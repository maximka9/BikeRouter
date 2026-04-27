"""Experimental ML recovery of missing OSM ``surface`` tags.

This module is intentionally standalone: it reads OSM edges and existing
satellite tiles, trains a local model, and writes experiment artifacts. It does
not mutate the routing graph and is not imported by the routing path.
"""

from __future__ import annotations

import ast
import json
import math
import os
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import joblib
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point
from shapely.ops import linemerge
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ..config import OSM_HIGHWAY_FILTER, Settings
from .area_graph_cache import (
    graph_base_path,
    graph_green_path,
    load_graphml_path,
    parse_precache_polygon,
)
from .graph import GraphBuilder

try:
    from PIL import Image

    PIL_AVAILABLE = True
except Exception:
    Image = None  # type: ignore[assignment]
    PIL_AVAILABLE = False

try:
    from pyproj import Transformer
except Exception:
    Transformer = None  # type: ignore[assignment]

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except Exception:
    tqdm = None  # type: ignore[assignment]
    TQDM_AVAILABLE = False


SURFACE_GROUP_MAP: Dict[str, str] = {
    "asphalt": "paved_good",
    "concrete": "paved_good",
    "paved": "paved_rough",
    "concrete:plates": "paved_rough",
    "paving_stones": "paved_rough",
    "sett": "paved_rough",
    "cobblestone": "paved_rough",
    "bricks": "paved_rough",
    "compacted": "unpaved_hard",
    "fine_gravel": "unpaved_hard",
    "gravel": "unpaved_hard",
    "pebblestone": "unpaved_hard",
    "unpaved": "unpaved_soft",
    "ground": "unpaved_soft",
    "dirt": "unpaved_soft",
    "earth": "unpaved_soft",
    "sand": "unpaved_soft",
    "grass": "unpaved_soft",
    "mud": "unpaved_soft",
}

SURFACE_GROUPS: Tuple[str, ...] = (
    "paved_good",
    "paved_rough",
    "unpaved_hard",
    "unpaved_soft",
)

OSM_TAG_COLUMNS: Tuple[str, ...] = (
    "highway",
    "surface",
    "service",
    "tracktype",
    "smoothness",
    "bicycle",
    "foot",
    "access",
    "lit",
    "oneway",
    "maxspeed",
    "lanes",
)

OSM_CATEGORICAL_FEATURES: Tuple[str, ...] = (
    "highway",
    "service",
    "tracktype",
    "smoothness",
    "bicycle",
    "foot",
    "access",
    "lit",
    "oneway",
    "maxspeed",
    "lanes",
)

GEOMETRY_FEATURES: Tuple[str, ...] = (
    "edge_length_m",
    "edge_bearing_deg",
    "edge_sinuosity",
    "edge_num_points",
    "edge_bbox_width_m",
    "edge_bbox_height_m",
)

SATELLITE_FEATURES: Tuple[str, ...] = (
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "rgb_std_r",
    "rgb_std_g",
    "rgb_std_b",
    "hsv_mean_h",
    "hsv_mean_s",
    "hsv_mean_v",
    "brightness_mean",
    "brightness_std",
    "saturation_mean",
    "saturation_std",
    "gray_pixel_share",
    "brown_pixel_share",
    "green_pixel_share",
    "dark_pixel_share",
    "texture_std",
    "tile_missing_share",
    "sampled_pixel_count",
)

NUMERIC_FEATURES: Tuple[str, ...] = GEOMETRY_FEATURES + SATELLITE_FEATURES

PREDICTION_COLUMNS: Tuple[str, ...] = (
    "edge_id",
    "u",
    "v",
    "key",
    "length_m",
    "highway",
    "surface_osm_raw",
    "surface_osm_norm",
    "surface_group_true",
    "is_surface_known",
    "surface_group_pred",
    "surface_group_effective",
    "surface_pred_confidence",
    "surface_source",
    "proba_paved_good",
    "proba_paved_rough",
    "proba_unpaved_hard",
    "proba_unpaved_soft",
    "surface_pred_proba_paved_good",
    "surface_pred_proba_paved_rough",
    "surface_pred_proba_unpaved_hard",
    "surface_pred_proba_unpaved_soft",
    "tile_samples_count",
    "tile_missing_share",
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "brightness_mean",
    "saturation_mean",
    "texture_std",
    "geometry_wkt",
)

SURFACE_COLORS: Dict[str, str] = {
    "paved_good": "#0B4F9C",
    "paved_rough": "#7B3294",
    "unpaved_hard": "#E08214",
    "unpaved_soft": "#8C2D04",
    "unknown": "#8A8A8A",
}

ProgressFactory = Callable[[Iterable[Any], str, Optional[int]], Iterable[Any]]


@dataclass(frozen=True)
class SurfaceMLConfig:
    sample_step_m: float = 7.0
    pixel_window: int = 5
    min_confidence: float = 0.65
    paved_good_min_confidence: float = 0.75
    spatial_grid_m: float = 300.0
    test_share: float = 0.20
    random_state: int = 42
    min_known_edges: int = 100
    min_class_edges_warning: int = 10
    n_estimators: int = 300
    max_tile_cache_items: int = 512


@dataclass
class ExperimentArtifacts:
    output_dir: Path
    dataset_csv: Path
    predictions_csv: Path
    predictions_geojson: Path
    map_png: Path
    legend_png: Path
    confusion_matrix_png: Path
    feature_importance_png: Path
    metrics_json: Path
    model_joblib: Path
    report_txt: Path
    model_info_xlsx: Path


def progress_iter(
    iterable: Iterable[Any],
    desc: str,
    total: Optional[int] = None,
) -> Iterable[Any]:
    if TQDM_AVAILABLE and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


def no_progress(
    iterable: Iterable[Any],
    desc: str,
    total: Optional[int] = None,
) -> Iterable[Any]:
    return iterable


def _single_progress(desc: str) -> None:
    for _ in progress_iter(range(1), desc, total=1):
        pass


def _first_osm_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (list, tuple, set)):
        for item in value:
            v = _first_osm_value(item)
            if v not in (None, ""):
                return v
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return None
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
            except Exception:
                return s
            return _first_osm_value(parsed)
        return s
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def normalize_osm_tag(value: Any) -> Optional[str]:
    v = _first_osm_value(value)
    if v is None:
        return None
    s = str(v).strip().lower()
    if not s or s in {"nan", "none", "null", "unknown"}:
        return None
    return s


def normalize_surface(value: Any) -> str:
    v = _first_osm_value(value)
    if v is None:
        return ""
    s = str(v).strip().lower()
    if not s or s in {"nan", "none", "null", "unknown", "no"}:
        return ""
    candidates = [p.strip() for p in s.replace("|", ";").split(";")]
    candidates = [p for p in candidates if p]
    for cand in candidates:
        if cand in SURFACE_GROUP_MAP:
            return cand
    if candidates:
        return candidates[0]
    return s


def surface_to_group(value: Any) -> str:
    return SURFACE_GROUP_MAP.get(normalize_surface(value), "unknown")


def _as_float(value: Any) -> float:
    v = _first_osm_value(value)
    if v is None:
        return float("nan")
    try:
        return float(v)
    except Exception:
        return float("nan")


def _ensure_line_geometry(geom: Any) -> Optional[Any]:
    if geom is None or getattr(geom, "is_empty", True):
        return None
    if isinstance(geom, (LineString, MultiLineString)):
        return geom
    if isinstance(geom, GeometryCollection):
        lines = [
            g
            for g in geom.geoms
            if isinstance(g, (LineString, MultiLineString)) and not g.is_empty
        ]
        if not lines:
            return None
        merged = linemerge(lines)
        if isinstance(merged, (LineString, MultiLineString)) and not merged.is_empty:
            return merged
    return None


def _line_parts(geom: Any) -> List[LineString]:
    geom = _ensure_line_geometry(geom)
    if geom is None:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return [g for g in geom.geoms if isinstance(g, LineString) and not g.is_empty]
    return []


def _coord_count(geom: Any) -> int:
    return sum(len(part.coords) for part in _line_parts(geom))


def _endpoints_wgs84(geom: Any) -> Optional[Tuple[float, float, float, float]]:
    parts = _line_parts(geom)
    if not parts:
        return None
    first = parts[0]
    last = parts[-1]
    lon1, lat1 = first.coords[0][0], first.coords[0][1]
    lon2, lat2 = last.coords[-1][0], last.coords[-1][1]
    return float(lon1), float(lat1), float(lon2), float(lat2)


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6_371_000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def _bearing_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _utm_crs_for(gdf: gpd.GeoDataFrame) -> Any:
    try:
        crs = gdf.estimate_utm_crs()
        if crs is not None:
            return crs
    except Exception:
        pass
    return "EPSG:3857"


def _tile_fraction(lat: float, lon: float, zoom: int) -> Tuple[int, int, int, int]:
    lat = max(-85.05112878, min(85.05112878, float(lat)))
    n = 2**int(zoom)
    xt = (float(lon) + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    yt = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    x = int(math.floor(xt))
    y = int(math.floor(yt))
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))
    px = int(max(0, min(255, math.floor((xt - x) * 256.0))))
    py = int(max(0, min(255, math.floor((yt - y) * 256.0))))
    return x, y, px, py


class ExistingTileReader:
    def __init__(
        self,
        tiles_dir: Path,
        *,
        server: str,
        zoom: int,
        max_items: int = 512,
    ) -> None:
        self.tiles_dir = Path(tiles_dir)
        self.server = str(server)
        self.zoom = int(zoom)
        self.max_items = max(1, int(max_items))
        self._cache: OrderedDict[Tuple[int, int], Optional[np.ndarray]] = OrderedDict()

    def _tile_path(self, x: int, y: int) -> Path:
        stem = f"{self.server}_{self.zoom}_{x}_{y}"
        for suffix in (".jpg", ".jpeg", ".png"):
            p = self.tiles_dir / f"{stem}{suffix}"
            if p.is_file():
                return p
        return self.tiles_dir / f"{stem}.jpg"

    def read(self, x: int, y: int) -> Optional[np.ndarray]:
        key = (int(x), int(y))
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        value: Optional[np.ndarray] = None
        p = self._tile_path(x, y)
        if PIL_AVAILABLE and p.is_file() and Image is not None:
            try:
                with Image.open(p) as im:
                    value = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
            except Exception:
                value = None
        self._cache[key] = value
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return value


def load_edges_for_precache_area(settings: Settings) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    if not settings.has_precache_area_polygon:
        raise ValueError("PRECACHE_AREA_POLYGON_WKT is empty or not a polygon")

    poly = parse_precache_polygon(settings)
    meta: Dict[str, Any] = {
        "precache_polygon_bounds_lonlat": tuple(float(x) for x in poly.bounds),
        "graph_source": None,
        "graph_path": None,
    }

    graph_path = graph_base_path(settings)
    G = load_graphml_path(graph_path)
    if G is None:
        graph_path = graph_green_path(settings)
        G = load_graphml_path(graph_path)

    if G is not None:
        meta["graph_source"] = "area_precache_graphml"
        meta["graph_path"] = str(graph_path)
        edges = GraphBuilder.to_geodataframe(G)
    else:
        meta["graph_source"] = "osmnx_overpass_fallback"
        ox.settings.cache_folder = settings.osmnx_cache_dir
        ox.settings.use_cache = True
        for tag in OSM_TAG_COLUMNS + ("smoothness",):
            if tag not in ox.settings.useful_tags_way:
                ox.settings.useful_tags_way += [tag]
        G = ox.graph_from_polygon(
            poly,
            custom_filter=OSM_HIGHWAY_FILTER,
            simplify=False,
            retain_all=True,
        )
        if not isinstance(G, nx.MultiDiGraph):
            G = nx.MultiDiGraph(G)
        edges = GraphBuilder.to_geodataframe(G)

    return edges, meta


def filter_edges_to_polygon(
    edges_gdf: gpd.GeoDataFrame,
    polygon: Any,
) -> gpd.GeoDataFrame:
    gdf = edges_gdf.copy()
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    else:
        gdf = gdf.to_crs("EPSG:4326")

    for col in OSM_TAG_COLUMNS:
        if col not in gdf.columns:
            gdf[col] = None
    if "length" not in gdf.columns:
        gdf["length"] = np.nan

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    clipped = gdf.geometry.intersection(polygon)
    clipped = clipped.map(_ensure_line_geometry)
    gdf["geometry"] = clipped
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    projected_crs = _utm_crs_for(gdf)
    projected = gdf.to_crs(projected_crs)
    length_m = projected.geometry.length.astype(float)
    gdf["length_m"] = length_m.values
    gdf["length"] = np.where(np.isfinite(gdf["length"].map(_as_float)), gdf["length"], gdf["length_m"])
    gdf = gdf[np.isfinite(gdf["length_m"]) & (gdf["length_m"] > 0.0)].copy()
    gdf = gdf[gdf["highway"].map(normalize_osm_tag).notna()].copy()
    gdf = gdf.reset_index()
    if "u" not in gdf.columns:
        gdf["u"] = None
    if "v" not in gdf.columns:
        gdf["v"] = None
    if "key" not in gdf.columns:
        gdf["key"] = 0
    gdf["edge_id"] = [
        f"{row.u}_{row.v}_{row.key}_{i}" for i, row in enumerate(gdf.itertuples())
    ]
    gdf["surface_osm_raw"] = gdf["surface"].map(_first_osm_value)
    gdf["surface_osm_norm"] = gdf["surface"].map(normalize_surface)
    gdf["surface_group_true"] = gdf["surface_osm_norm"].map(
        lambda s: SURFACE_GROUP_MAP.get(str(s), "unknown")
    )
    gdf["is_surface_known"] = gdf["surface_group_true"].isin(SURFACE_GROUPS)
    return gdf


def limit_edges_for_experiment(
    edges_gdf: gpd.GeoDataFrame,
    max_edges: Optional[int],
    *,
    random_state: int,
    min_known_edges: int = 100,
) -> gpd.GeoDataFrame:
    if max_edges is None or int(max_edges) <= 0 or len(edges_gdf) <= int(max_edges):
        return edges_gdf.copy().reset_index(drop=True)

    max_edges = int(max_edges)
    known = edges_gdf[edges_gdf["is_surface_known"]].copy()
    unknown = edges_gdf[~edges_gdf["is_surface_known"]].copy()
    rng = np.random.default_rng(int(random_state))

    known_target = min(len(known), max_edges)
    if len(unknown) > 0:
        known_target = min(
            len(known),
            max(min_known_edges, int(round(max_edges * 0.70))),
            max_edges - 1,
        )
    unknown_target = max(0, max_edges - known_target)

    selected_known_parts: List[pd.DataFrame] = []
    if known_target > 0:
        groups = list(known.groupby("surface_group_true", sort=True))
        base_each = max(1, min(10, known_target // max(1, len(groups))))
        used_idx: set = set()
        for _, group in groups:
            take = min(len(group), base_each)
            if take <= 0:
                continue
            sample = group.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1)))
            selected_known_parts.append(sample)
            used_idx.update(sample.index.tolist())
        selected_known = (
            pd.concat(selected_known_parts, axis=0)
            if selected_known_parts
            else known.iloc[0:0].copy()
        )
        remaining = known.drop(index=list(used_idx), errors="ignore")
        fill = known_target - len(selected_known)
        if fill > 0 and len(remaining) > 0:
            selected_known = pd.concat(
                [
                    selected_known,
                    remaining.sample(
                        n=min(fill, len(remaining)),
                        random_state=int(rng.integers(0, 2**31 - 1)),
                    ),
                ],
                axis=0,
            )
    else:
        selected_known = known.iloc[0:0].copy()

    if unknown_target > 0 and len(unknown) > 0:
        selected_unknown = unknown.sample(
            n=min(unknown_target, len(unknown)),
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
    else:
        selected_unknown = unknown.iloc[0:0].copy()

    out = pd.concat([selected_known, selected_unknown], axis=0)
    if len(out) < max_edges:
        rest = edges_gdf.drop(index=out.index, errors="ignore")
        if len(rest) > 0:
            out = pd.concat(
                [
                    out,
                    rest.sample(
                        n=min(max_edges - len(out), len(rest)),
                        random_state=int(rng.integers(0, 2**31 - 1)),
                    ),
                ],
                axis=0,
            )
    return out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def _geometry_feature_row(row: Any, projected_geom: Any) -> Dict[str, Any]:
    length_m = float(getattr(projected_geom, "length", float("nan")))
    endpoints = _endpoints_wgs84(row.geometry)
    if endpoints is None:
        bearing = float("nan")
        straight = float("nan")
    else:
        lon1, lat1, lon2, lat2 = endpoints
        bearing = _bearing_deg(lon1, lat1, lon2, lat2)
        straight = _haversine_m(lon1, lat1, lon2, lat2)
    bounds = projected_geom.bounds if projected_geom is not None else (np.nan,) * 4
    bbox_w = float(bounds[2] - bounds[0]) if len(bounds) == 4 else float("nan")
    bbox_h = float(bounds[3] - bounds[1]) if len(bounds) == 4 else float("nan")
    if straight and np.isfinite(straight) and straight > 0:
        sinuosity = max(1.0, length_m / straight)
    else:
        sinuosity = 1.0
    return {
        "edge_length_m": length_m,
        "edge_bearing_deg": bearing,
        "edge_sinuosity": sinuosity,
        "edge_num_points": int(_coord_count(row.geometry)),
        "edge_bbox_width_m": bbox_w,
        "edge_bbox_height_m": bbox_h,
    }


def build_osm_geometry_dataset(
    edges_gdf: gpd.GeoDataFrame,
    *,
    progress: ProgressFactory = no_progress,
) -> pd.DataFrame:
    projected_crs = _utm_crs_for(edges_gdf)
    projected = edges_gdf.to_crs(projected_crs)
    rows: List[Dict[str, Any]] = []
    iterator = zip(edges_gdf.itertuples(index=False), projected.geometry.values)
    for row, projected_geom in progress(
        iterator, "[3/7] OSM/geometry features", total=len(edges_gdf)
    ):
        item: Dict[str, Any] = {
            "edge_id": row.edge_id,
            "u": str(row.u),
            "v": str(row.v),
            "key": str(row.key),
            "length_m": float(getattr(row, "length_m", np.nan)),
            "surface_raw": getattr(row, "surface_osm_raw", None),
            "surface_norm": getattr(row, "surface_osm_norm", ""),
            "surface_osm_raw": getattr(row, "surface_osm_raw", None),
            "surface_osm_norm": getattr(row, "surface_osm_norm", ""),
            "surface_group_true": getattr(row, "surface_group_true", "unknown"),
            "is_surface_known": bool(getattr(row, "is_surface_known", False)),
            "geometry_wkt": row.geometry.wkt if row.geometry is not None else None,
        }
        for col in OSM_CATEGORICAL_FEATURES:
            item[col] = normalize_osm_tag(getattr(row, col, None))
        item.update(_geometry_feature_row(row, projected_geom))
        rows.append(item)
    return pd.DataFrame(rows)


def _sample_projected_points(geom: Any, step_m: float) -> List[Point]:
    parts = _line_parts(geom)
    points: List[Point] = []
    step_m = max(1.0, float(step_m))
    for part in parts:
        length = float(part.length)
        if not np.isfinite(length) or length <= 0:
            continue
        n_steps = max(1, int(math.floor(length / step_m)))
        distances = [min(length, i * step_m) for i in range(n_steps + 1)]
        if distances[-1] < length:
            distances.append(length)
        for dist in distances:
            try:
                points.append(part.interpolate(float(dist)))
            except Exception:
                continue
    return points


def _rgb_to_hsv(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb_f = rgb.astype(np.float64) / 255.0
    r = rgb_f[:, 0]
    g = rgb_f[:, 1]
    b = rgb_f[:, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc
    h = np.zeros_like(maxc)
    mask = delta > 1e-12
    rmask = mask & (maxc == r)
    gmask = mask & (maxc == g)
    bmask = mask & (maxc == b)
    h[rmask] = ((g[rmask] - b[rmask]) / delta[rmask]) % 6.0
    h[gmask] = ((b[gmask] - r[gmask]) / delta[gmask]) + 2.0
    h[bmask] = ((r[bmask] - g[bmask]) / delta[bmask]) + 4.0
    h = h * 60.0
    s = np.zeros_like(maxc)
    nonzero = maxc > 1e-12
    s[nonzero] = delta[nonzero] / maxc[nonzero]
    return h, s, maxc


def _nan_tile_features(
    *,
    samples_count: int = 0,
    missing_share: float = 1.0,
) -> Dict[str, Any]:
    out = {name: float("nan") for name in SATELLITE_FEATURES}
    out["tile_samples_count"] = int(samples_count)
    out["sampled_pixel_count"] = 0
    out["tile_missing_share"] = float(missing_share)
    out["tile_missing"] = True
    return out


def _tile_features_from_pixels(
    pixels: List[np.ndarray],
    *,
    samples_count: int,
    missing_count: int,
) -> Dict[str, Any]:
    if not pixels:
        return _nan_tile_features(
            samples_count=samples_count,
            missing_share=(missing_count / samples_count if samples_count else 1.0),
        )
    rgb = np.vstack(pixels).astype(np.float64)
    h, s, v = _rgb_to_hsv(rgb)
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float64)
    brightness = v * 255.0
    hue = h
    sample_total = max(1, int(samples_count))
    missing_share = float(missing_count) / float(sample_total)
    out = {
        "rgb_mean_r": float(np.nanmean(rgb[:, 0])),
        "rgb_mean_g": float(np.nanmean(rgb[:, 1])),
        "rgb_mean_b": float(np.nanmean(rgb[:, 2])),
        "rgb_std_r": float(np.nanstd(rgb[:, 0])),
        "rgb_std_g": float(np.nanstd(rgb[:, 1])),
        "rgb_std_b": float(np.nanstd(rgb[:, 2])),
        "hsv_mean_h": float(np.nanmean(hue)),
        "hsv_mean_s": float(np.nanmean(s)),
        "hsv_mean_v": float(np.nanmean(v)),
        "brightness_mean": float(np.nanmean(brightness)),
        "brightness_std": float(np.nanstd(brightness)),
        "saturation_mean": float(np.nanmean(s)),
        "saturation_std": float(np.nanstd(s)),
        "gray_pixel_share": float(np.mean((s < 0.18) & (v > 0.18) & (v < 0.92))),
        "brown_pixel_share": float(np.mean((hue >= 15.0) & (hue <= 55.0) & (s > 0.20) & (v > 0.15))),
        "green_pixel_share": float(np.mean((hue >= 70.0) & (hue <= 170.0) & (s > 0.20) & (v > 0.12))),
        "dark_pixel_share": float(np.mean(v < 0.22)),
        "texture_std": float(np.nanstd(gray)),
        "tile_missing_share": missing_share,
        "sampled_pixel_count": int(rgb.shape[0]),
        "tile_samples_count": int(samples_count),
        "tile_missing": bool(missing_count > 0 or rgb.shape[0] == 0),
    }
    return out


def extract_tile_features_for_edges(
    edges_gdf: gpd.GeoDataFrame,
    settings: Settings,
    config: SurfaceMLConfig,
    *,
    progress: ProgressFactory = no_progress,
) -> pd.DataFrame:
    projected_crs = _utm_crs_for(edges_gdf)
    projected = edges_gdf.to_crs(projected_crs)
    if Transformer is None:
        rows = []
        for row in progress(edges_gdf.itertuples(index=False), "[4/7] Tile features", total=len(edges_gdf)):
            item = {"edge_id": row.edge_id}
            item.update(_nan_tile_features())
            rows.append(item)
        return pd.DataFrame(rows)

    transformer = Transformer.from_crs(projected_crs, "EPSG:4326", always_xy=True)
    tiles_dir = Path(settings.cache_dir) / "tiles"
    reader = ExistingTileReader(
        tiles_dir,
        server=settings.tms_server,
        zoom=settings.satellite_zoom,
        max_items=config.max_tile_cache_items,
    )
    half = max(0, int(config.pixel_window) // 2)
    rows: List[Dict[str, Any]] = []
    iterator = zip(edges_gdf.itertuples(index=False), projected.geometry.values)
    for row, projected_geom in progress(iterator, "[4/7] Tile features", total=len(edges_gdf)):
        try:
            pixels: List[np.ndarray] = []
            sample_points = _sample_projected_points(projected_geom, config.sample_step_m)
            missing = 0
            for point in sample_points:
                try:
                    lon, lat = transformer.transform(point.x, point.y)
                    x, y, px, py = _tile_fraction(lat, lon, settings.satellite_zoom)
                except Exception:
                    missing += 1
                    continue
                tile = reader.read(x, y)
                if tile is None:
                    missing += 1
                    continue
                y0 = max(0, py - half)
                y1 = min(tile.shape[0], py + half + 1)
                x0 = max(0, px - half)
                x1 = min(tile.shape[1], px + half + 1)
                window = tile[y0:y1, x0:x1, :]
                if window.size == 0:
                    missing += 1
                    continue
                pixels.append(window.reshape(-1, 3))
            item = {"edge_id": row.edge_id}
            item.update(
                _tile_features_from_pixels(
                    pixels,
                    samples_count=len(sample_points),
                    missing_count=missing,
                )
            )
            rows.append(item)
        except Exception:
            item = {"edge_id": getattr(row, "edge_id", None)}
            item.update(
                _nan_tile_features(
                    samples_count=0,
                    missing_share=1.0,
                )
            )
            rows.append(item)
    return pd.DataFrame(rows)


def dedupe_dataframe_by_edge_id(
    df: pd.DataFrame,
    *,
    keep: str = "last",
    name: str = "df",
) -> pd.DataFrame:
    """Оставить одну строку на edge_id (например после concat тайловых чанков)."""
    if df.empty or "edge_id" not in df.columns:
        return df
    dup = df["edge_id"].duplicated(keep=False)
    if dup.any():
        import logging

        logging.getLogger(__name__).warning(
            "%s: %d строк с дублирующимся edge_id перед merge, keep=%s",
            name,
            int(dup.sum()),
            keep,
        )
    return df.drop_duplicates(subset=["edge_id"], keep=keep)


def build_dataset(
    edges_gdf: gpd.GeoDataFrame,
    settings: Settings,
    config: SurfaceMLConfig,
    *,
    progress: ProgressFactory = no_progress,
) -> pd.DataFrame:
    osm_df = build_osm_geometry_dataset(edges_gdf, progress=progress)
    tile_df = extract_tile_features_for_edges(
        edges_gdf,
        settings,
        config,
        progress=progress,
    )
    osm_df = osm_df.copy()
    tile_df = tile_df.copy()
    osm_df["edge_id"] = osm_df["edge_id"].map(lambda x: "" if pd.isna(x) else str(x).strip())
    tile_df["edge_id"] = tile_df["edge_id"].map(lambda x: "" if pd.isna(x) else str(x).strip())
    osm_df = dedupe_dataframe_by_edge_id(osm_df, keep="first", name="osm_df (surface_ml)")
    tile_df = dedupe_dataframe_by_edge_id(tile_df, keep="last", name="tile_df (surface_ml)")
    dataset = osm_df.merge(tile_df, on="edge_id", how="left", validate="one_to_one")
    for name in SATELLITE_FEATURES:
        if name not in dataset.columns:
            dataset[name] = np.nan
    if "tile_samples_count" not in dataset.columns:
        dataset["tile_samples_count"] = 0
    if "tile_missing" not in dataset.columns:
        dataset["tile_missing"] = True
    return dataset


def validate_training_data(dataset: pd.DataFrame, config: SurfaceMLConfig) -> None:
    known = dataset[dataset["is_surface_known"]].copy()
    known_count = len(known)
    if known_count < int(config.min_known_edges):
        raise ValueError(
            f"Not enough known surface edges for training: found {known_count}. "
            f"Need at least {config.min_known_edges}."
        )
    class_count = int(known["surface_group_true"].nunique())
    if class_count < 2:
        raise ValueError(
            f"Not enough surface classes for training: found {class_count}. "
            "Need at least 2."
        )


def spatial_train_test_split(
    known_df: pd.DataFrame,
    edges_gdf: gpd.GeoDataFrame,
    *,
    grid_m: float,
    test_share: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
    known_edge_ids = set(known_df["edge_id"].astype(str))
    known_edges = edges_gdf[edges_gdf["edge_id"].astype(str).isin(known_edge_ids)].copy()
    if len(known_edges) != len(known_df):
        known_edges = known_edges.set_index("edge_id").loc[known_df["edge_id"].astype(str)].reset_index()
    projected = known_edges.to_crs(_utm_crs_for(known_edges))
    cent = projected.geometry.centroid
    gx = np.floor(cent.x.to_numpy() / max(1.0, float(grid_m))).astype(np.int64)
    gy = np.floor(cent.y.to_numpy() / max(1.0, float(grid_m))).astype(np.int64)
    cells = pd.Series([f"{x}:{y}" for x, y in zip(gx, gy)], index=known_df.index)
    unique_cells = np.array(sorted(cells.unique()))
    if len(unique_cells) < 2:
        raise ValueError("Not enough spatial grid cells for train/test split")
    rng = np.random.default_rng(int(random_state))
    rng.shuffle(unique_cells)
    test_n = max(1, int(round(len(unique_cells) * float(test_share))))
    test_n = min(test_n, len(unique_cells) - 1)
    test_cells = set(unique_cells[:test_n])

    train_mask = ~cells.isin(test_cells).to_numpy()
    test_mask = cells.isin(test_cells).to_numpy()
    all_classes = set(known_df["surface_group_true"])
    train_classes = set(known_df.loc[train_mask, "surface_group_true"])
    missing_train = all_classes - train_classes
    for cls in sorted(missing_train):
        cls_cells = cells[known_df["surface_group_true"] == cls]
        move_candidates = [c for c in cls_cells.unique() if c in test_cells]
        if move_candidates:
            test_cells.remove(move_candidates[0])
    train_mask = ~cells.isin(test_cells).to_numpy()
    test_mask = cells.isin(test_cells).to_numpy()
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Spatial train/test split produced an empty partition")
    return train_mask, test_mask, cells


def _model_features(dataset: pd.DataFrame) -> pd.DataFrame:
    cols = list(OSM_CATEGORICAL_FEATURES) + list(NUMERIC_FEATURES)
    X = dataset.reindex(columns=cols).copy()
    for col in OSM_CATEGORICAL_FEATURES:
        X[col] = X[col].where(X[col].notna(), None)
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def _build_pipeline(config: SurfaceMLConfig) -> Pipeline:
    cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                        ("onehot", cat_encoder),
                    ]
                ),
                list(OSM_CATEGORICAL_FEATURES),
            ),
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                list(NUMERIC_FEATURES),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    clf = RandomForestClassifier(
        n_estimators=int(config.n_estimators),
        random_state=int(config.random_state),
        class_weight="balanced",
        n_jobs=-1,
        min_samples_leaf=2,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])


def _classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    labels: Sequence[str],
) -> Dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(labels),
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix_labels": list(labels),
        "confusion_matrix": cm.astype(int).tolist(),
        "classification_report": report,
        "precision_recall_paved_good": {
            "precision": float(report.get("paved_good", {}).get("precision", 0.0)),
            "recall": float(report.get("paved_good", {}).get("recall", 0.0)),
        },
        "precision_recall_unpaved_soft": {
            "precision": float(report.get("unpaved_soft", {}).get("precision", 0.0)),
            "recall": float(report.get("unpaved_soft", {}).get("recall", 0.0)),
        },
    }


def _feature_importance_df(pipeline: Pipeline) -> pd.DataFrame:
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    try:
        names = list(preprocess.get_feature_names_out())
    except Exception:
        names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
    values = np.asarray(model.feature_importances_, dtype=float)
    n = min(len(names), len(values))
    return (
        pd.DataFrame({"feature": names[:n], "importance": values[:n]})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def train_surface_model(
    dataset: pd.DataFrame,
    edges_gdf: gpd.GeoDataFrame,
    config: SurfaceMLConfig,
    *,
    progress: ProgressFactory = no_progress,
) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame]:
    validate_training_data(dataset, config)
    known = dataset[dataset["is_surface_known"]].copy().reset_index(drop=True)
    train_mask, test_mask, cells = spatial_train_test_split(
        known,
        edges_gdf,
        grid_m=config.spatial_grid_m,
        test_share=config.test_share,
        random_state=config.random_state,
    )
    X = _model_features(known)
    y = known["surface_group_true"].astype(str)
    pipeline = _build_pipeline(config)
    for _ in progress(range(1), "[5/7] Train model", total=1):
        pipeline.fit(X.loc[train_mask], y.loc[train_mask])
    test_pred = pipeline.predict(X.loc[test_mask])
    labels = list(SURFACE_GROUPS)
    metrics = _classification_metrics(y.loc[test_mask].tolist(), test_pred.tolist(), labels=labels)
    metrics.update(
        {
            "class_distribution_all_known": known["surface_group_true"].value_counts().sort_index().to_dict(),
            "class_distribution_train": y.loc[train_mask].value_counts().sort_index().to_dict(),
            "class_distribution_test": y.loc[test_mask].value_counts().sort_index().to_dict(),
            "known_edges_count": int(len(known)),
            "train_edges_count": int(train_mask.sum()),
            "test_edges_count": int(test_mask.sum()),
            "spatial_grid_m": float(config.spatial_grid_m),
            "test_share": float(config.test_share),
            "spatial_cells_total": int(cells.nunique()),
            "spatial_cells_train": int(cells.loc[train_mask].nunique()),
            "spatial_cells_test": int(cells.loc[test_mask].nunique()),
        }
    )
    small_classes = {
        cls: int(count)
        for cls, count in known["surface_group_true"].value_counts().items()
        if int(count) < int(config.min_class_edges_warning)
    }
    if small_classes:
        metrics["small_class_warning"] = small_classes
    return pipeline, metrics, _feature_importance_df(pipeline)


def _predict_proba_frame(pipeline: Pipeline, dataset: pd.DataFrame) -> pd.DataFrame:
    X = _model_features(dataset)
    classes = list(pipeline.named_steps["model"].classes_)
    proba = pipeline.predict_proba(X)
    out = pd.DataFrame(0.0, index=dataset.index, columns=[f"proba_{c}" for c in SURFACE_GROUPS])
    for i, cls in enumerate(classes):
        if cls in SURFACE_GROUPS:
            out[f"proba_{cls}"] = proba[:, i]
    return out


def _has_any_model_features(row: pd.Series) -> bool:
    for col in OSM_CATEGORICAL_FEATURES:
        v = normalize_osm_tag(row.get(col))
        if v:
            return True
    for col in NUMERIC_FEATURES:
        v = pd.to_numeric(row.get(col), errors="coerce")
        if pd.notna(v):
            return True
    return False


def apply_prediction_policy(
    pred_group: str,
    confidence: float,
    *,
    is_surface_known: bool,
    true_group: str,
    has_features: bool,
    config: SurfaceMLConfig,
) -> Tuple[str, str, str, float]:
    if is_surface_known:
        group = true_group if true_group in SURFACE_GROUPS else "unknown"
        return group, group, "osm", 1.0
    if not has_features:
        return pred_group or "unknown", "unknown", "default_no_features", 0.0
    pred_group = pred_group if pred_group in SURFACE_GROUPS else "unknown"
    confidence = float(confidence) if np.isfinite(confidence) else 0.0
    if pred_group == "paved_good" and confidence < float(config.paved_good_min_confidence):
        return pred_group, "paved_rough", "default_low_confidence", confidence
    if confidence >= float(config.min_confidence):
        return pred_group, pred_group, "ml", confidence
    return pred_group, "unknown", "default_low_confidence", confidence


def predict_all_edges(
    dataset: pd.DataFrame,
    pipeline: Pipeline,
    config: SurfaceMLConfig,
    *,
    progress: ProgressFactory = no_progress,
) -> pd.DataFrame:
    for _ in progress(range(1), "[6/7] Predict unknown surface", total=1):
        proba_df = _predict_proba_frame(pipeline, dataset)
        pred_labels = proba_df.idxmax(axis=1).str.replace("proba_", "", regex=False)
        confidences = proba_df.max(axis=1).astype(float)

    out = dataset.copy()
    out = pd.concat([out, proba_df], axis=1)
    out["surface_pred_proba_paved_good"] = out["proba_paved_good"]
    out["surface_pred_proba_paved_rough"] = out["proba_paved_rough"]
    out["surface_pred_proba_unpaved_hard"] = out["proba_unpaved_hard"]
    out["surface_pred_proba_unpaved_soft"] = out["proba_unpaved_soft"]
    pred_groups: List[str] = []
    effective_groups: List[str] = []
    sources: List[str] = []
    final_conf: List[float] = []
    has_features_col: List[bool] = []
    for idx, row in out.iterrows():
        has_features = _has_any_model_features(row)
        pred, eff, src, conf = apply_prediction_policy(
            str(pred_labels.loc[idx]),
            float(confidences.loc[idx]),
            is_surface_known=bool(row.get("is_surface_known", False)),
            true_group=str(row.get("surface_group_true") or "unknown"),
            has_features=has_features,
            config=config,
        )
        pred_groups.append(pred)
        effective_groups.append(eff)
        sources.append(src)
        final_conf.append(conf)
        has_features_col.append(has_features)
    out["surface_group_pred"] = pred_groups
    out["surface_group_effective"] = effective_groups
    out["surface_pred_confidence"] = final_conf
    out["surface_source"] = sources
    out["has_ml_features"] = has_features_col
    return out


def experiment_output_dir(settings: Settings, output_dir: Optional[Path] = None) -> Path:
    if output_dir is not None:
        out = Path(output_dir)
    else:
        stamp = datetime.now().strftime("surface_ml_%Y%m%d_%H%M%S")
        out = Path(settings.base_dir) / "experiments" / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def artifact_paths(output_dir: Path) -> ExperimentArtifacts:
    return ExperimentArtifacts(
        output_dir=output_dir,
        dataset_csv=output_dir / "surface_edges_dataset.csv",
        predictions_csv=output_dir / "surface_predictions.csv",
        predictions_geojson=output_dir / "surface_predictions.geojson",
        map_png=output_dir / "surface_ml_map.png",
        legend_png=output_dir / "surface_ml_legend.png",
        confusion_matrix_png=output_dir / "surface_ml_confusion_matrix.png",
        feature_importance_png=output_dir / "surface_ml_feature_importance.png",
        metrics_json=output_dir / "surface_ml_metrics.json",
        model_joblib=output_dir / "surface_ml_model.joblib",
        report_txt=output_dir / "surface_ml_report.txt",
        model_info_xlsx=output_dir / "surface_ml_model_info.xlsx",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return _json_safe(value.tolist())
    return value


def write_dataset_csv(dataset: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(path, index=False, encoding="utf-8")


def write_predictions_csv(predictions: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in PREDICTION_COLUMNS if c in predictions.columns]
    extra = [c for c in predictions.columns if c not in cols and c != "geometry"]
    predictions[cols + extra].to_csv(path, index=False, encoding="utf-8")


def write_predictions_geojson(
    predictions: pd.DataFrame,
    edges_gdf: gpd.GeoDataFrame,
    path: Path,
) -> None:
    base = edges_gdf[["edge_id", "geometry"]].copy()
    gdf = base.merge(predictions.drop(columns=["geometry"], errors="ignore"), on="edge_id", how="left")
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GeoJSON")


def _import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_surface_map(
    predictions: pd.DataFrame,
    edges_gdf: gpd.GeoDataFrame,
    path: Path,
    legend_path: Path,
) -> None:
    plt = _import_pyplot()
    gdf = edges_gdf[["edge_id", "geometry"]].copy()
    cols = ["edge_id", "surface_group_effective", "surface_source"]
    gdf = gdf.merge(predictions[cols], on="edge_id", how="left")
    gdf["surface_group_effective"] = gdf["surface_group_effective"].fillna("unknown")
    gdf["surface_source"] = gdf["surface_source"].fillna("unknown")

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_axis_off()
    source_styles = {
        "osm": ("solid", 1.5, 0.95),
        "ml": ((0, (5, 3)), 1.2, 0.90),
        "default_low_confidence": ((0, (1, 3)), 0.8, 0.75),
        "default_no_features": ((0, (1, 3)), 0.7, 0.65),
        "unknown": ((0, (1, 3)), 0.7, 0.65),
    }
    for group in list(SURFACE_GROUPS) + ["unknown"]:
        for source, style in source_styles.items():
            subset = gdf[
                (gdf["surface_group_effective"] == group)
                & (gdf["surface_source"] == source)
            ]
            if subset.empty:
                continue
            subset.plot(
                ax=ax,
                color=SURFACE_COLORS.get(group, SURFACE_COLORS["unknown"]),
                linewidth=style[1],
                linestyle=style[0],
                alpha=style[2],
            )
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=SURFACE_COLORS[g], lw=4, label=g)
        for g in list(SURFACE_GROUPS) + ["unknown"]
    ]
    handles.extend(
        [
            Line2D([0], [0], color="#333333", lw=3, linestyle="solid", label="osm"),
            Line2D([0], [0], color="#333333", lw=3, linestyle=(0, (5, 3)), label="ml"),
            Line2D([0], [0], color="#333333", lw=3, linestyle=(0, (1, 3)), label="default/unknown"),
        ]
    )
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(legend_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix_png(metrics: Dict[str, Any], path: Path) -> None:
    plt = _import_pyplot()
    labels = list(metrics.get("confusion_matrix_labels") or SURFACE_GROUPS)
    cm = np.asarray(metrics.get("confusion_matrix") or np.zeros((len(labels), len(labels))), dtype=float)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="#111111")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance_png(feature_importance: pd.DataFrame, path: Path, *, top_n: int = 25) -> None:
    plt = _import_pyplot()
    df = feature_importance.head(top_n).iloc[::-1].copy()
    fig_h = max(4.0, min(12.0, 0.35 * max(1, len(df))))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(df["feature"], df["importance"], color="#3B6EA8")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_metrics_json(metrics: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(metrics), f, ensure_ascii=False, indent=2)


def write_model_joblib(
    pipeline: Pipeline,
    feature_importance: pd.DataFrame,
    metrics: Dict[str, Any],
    config: SurfaceMLConfig,
    path: Path,
) -> None:
    bundle = {
        "pipeline": pipeline,
        "surface_groups": list(SURFACE_GROUPS),
        "surface_group_map": SURFACE_GROUP_MAP,
        "categorical_features": list(OSM_CATEGORICAL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "feature_importance": feature_importance,
        "metrics": _json_safe(metrics),
        "config": asdict(config),
    }
    joblib.dump(bundle, path)


def write_model_info_xlsx(
    path: Path,
    *,
    config: SurfaceMLConfig,
    metrics: Dict[str, Any],
    feature_importance: pd.DataFrame,
    run_meta: Dict[str, Any],
    artifacts: ExperimentArtifacts,
) -> None:
    rows = []
    for key, value in {
        "model": "RandomForestClassifier",
        "n_estimators": config.n_estimators,
        "random_state": config.random_state,
        "min_confidence": config.min_confidence,
        "paved_good_min_confidence": config.paved_good_min_confidence,
        "spatial_grid_m": config.spatial_grid_m,
        "test_share": config.test_share,
        "sample_step_m": config.sample_step_m,
        "pixel_window": config.pixel_window,
        "graph_source": run_meta.get("graph_source"),
        "graph_path": run_meta.get("graph_path"),
        "tiles_dir": run_meta.get("tiles_dir"),
        "satellite_zoom": run_meta.get("satellite_zoom"),
        "known_edges_count": metrics.get("known_edges_count"),
        "train_edges_count": metrics.get("train_edges_count"),
        "test_edges_count": metrics.get("test_edges_count"),
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "limitation": (
            "Satellite windows may include cars, shadows, tree crowns, roofs, "
            "yards, and road-adjacent pixels; tile features are auxiliary."
        ),
    }.items():
        rows.append({"parameter": key, "value": value})

    class_train = pd.Series(metrics.get("class_distribution_train", {}), name="count").reset_index()
    class_train.columns = ["surface_group", "train_count"]
    class_test = pd.Series(metrics.get("class_distribution_test", {}), name="count").reset_index()
    class_test.columns = ["surface_group", "test_count"]
    files = pd.DataFrame(
        [
            {"artifact": field, "path": str(getattr(artifacts, field))}
            for field in artifacts.__dataclass_fields__
            if field != "output_dir"
        ]
    )
    report_df = pd.DataFrame(metrics.get("classification_report", {})).T.reset_index()
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="model_info", index=False)
        pd.DataFrame([metrics]).drop(columns=["classification_report"], errors="ignore").to_excel(
            writer, sheet_name="metrics", index=False
        )
        class_train.to_excel(writer, sheet_name="train_classes", index=False)
        class_test.to_excel(writer, sheet_name="test_classes", index=False)
        report_df.to_excel(writer, sheet_name="classification", index=False)
        feature_importance.head(100).to_excel(writer, sheet_name="feature_importance", index=False)
        files.to_excel(writer, sheet_name="artifacts", index=False)


def dataset_summary(dataset: pd.DataFrame) -> Dict[str, Any]:
    total = int(len(dataset))
    known = int(dataset["is_surface_known"].sum()) if total else 0
    unknown = total - known
    return {
        "total_edges": total,
        "known_surface_edges": known,
        "unknown_surface_edges": unknown,
        "unknown_share": float(unknown / total) if total else 0.0,
        "surface_group_distribution": dataset["surface_group_true"].value_counts(dropna=False).to_dict(),
        "highway_distribution": dataset["highway"].value_counts(dropna=False).head(50).to_dict(),
    }


def write_report(
    path: Path,
    *,
    config: SurfaceMLConfig,
    summary: Dict[str, Any],
    metrics: Dict[str, Any],
    run_meta: Dict[str, Any],
    artifacts: ExperimentArtifacts,
) -> None:
    lines = [
        "Surface ML experiment report",
        "",
        f"Output dir: {artifacts.output_dir}",
        f"Graph source: {run_meta.get('graph_source')}",
        f"Graph path: {run_meta.get('graph_path')}",
        f"Tiles dir: {run_meta.get('tiles_dir')}",
        f"Satellite zoom: {run_meta.get('satellite_zoom')}",
        "",
        "Dataset:",
        f"  Total edges: {summary.get('total_edges')}",
        f"  Known surface edges: {summary.get('known_surface_edges')}",
        f"  Unknown surface edges: {summary.get('unknown_surface_edges')}",
        f"  Unknown share: {summary.get('unknown_share'):.3f}",
        f"  Surface group distribution: {summary.get('surface_group_distribution')}",
        f"  Highway distribution top: {summary.get('highway_distribution')}",
        "",
        "Model:",
        "  Estimator: RandomForestClassifier",
        f"  n_estimators: {config.n_estimators}",
        f"  random_state: {config.random_state}",
        f"  min_confidence: {config.min_confidence}",
        f"  paved_good_min_confidence: {config.paved_good_min_confidence}",
        f"  spatial_grid_m: {config.spatial_grid_m}",
        f"  test_share: {config.test_share}",
        "",
        "Metrics:",
        f"  accuracy: {metrics.get('accuracy')}",
        f"  balanced_accuracy: {metrics.get('balanced_accuracy')}",
        f"  macro_f1: {metrics.get('macro_f1')}",
        f"  weighted_f1: {metrics.get('weighted_f1')}",
        f"  precision/recall paved_good: {metrics.get('precision_recall_paved_good')}",
        f"  precision/recall unpaved_soft: {metrics.get('precision_recall_unpaved_soft')}",
        "",
        "Safety policy:",
        "  Low-confidence predictions are not marked as ML source.",
        "  If paved_good is predicted below paved_good_min_confidence, the effective group is paved_rough.",
        "",
        "Limitations:",
        "  Satellite tiles can include cars, shadows, tree crowns, roofs, yards, and adjacent land.",
        "  The first-stage pixel statistics are auxiliary features, not ground truth.",
        "",
        "Artifacts:",
    ]
    for field in artifacts.__dataclass_fields__:
        if field != "output_dir":
            lines.append(f"  {field}: {getattr(artifacts, field)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_surface_ml_experiment(
    *,
    settings: Optional[Settings] = None,
    config: Optional[SurfaceMLConfig] = None,
    max_edges: Optional[int] = None,
    output_dir: Optional[Path] = None,
    progress: ProgressFactory = progress_iter,
) -> ExperimentArtifacts:
    settings = settings or Settings()
    config = config or SurfaceMLConfig()
    out_dir = experiment_output_dir(settings, output_dir)
    artifacts = artifact_paths(out_dir)
    run_meta: Dict[str, Any] = {
        "tiles_dir": str(Path(settings.cache_dir) / "tiles"),
        "satellite_zoom": int(settings.satellite_zoom),
        "tms_server": str(settings.tms_server),
    }

    for _ in progress(range(1), "[1/7] Load graph/edges", total=1):
        edges_raw, graph_meta = load_edges_for_precache_area(settings)
        run_meta.update(graph_meta)

    polygon = parse_precache_polygon(settings)
    for _ in progress(range(1), "[2/7] Filter polygon", total=1):
        edges = filter_edges_to_polygon(edges_raw, polygon)
        edges = limit_edges_for_experiment(
            edges,
            max_edges,
            random_state=config.random_state,
            min_known_edges=config.min_known_edges,
        )

    dataset = build_dataset(edges, settings, config, progress=progress)
    summary = dataset_summary(dataset)
    write_dataset_csv(dataset, artifacts.dataset_csv)

    pipeline, metrics, feature_importance = train_surface_model(
        dataset,
        edges,
        config,
        progress=progress,
    )
    metrics["dataset_summary"] = summary
    metrics["run_meta"] = run_meta
    metrics["surface_group_map"] = SURFACE_GROUP_MAP
    metrics["config"] = asdict(config)

    predictions = predict_all_edges(dataset, pipeline, config, progress=progress)
    write_predictions_csv(predictions, artifacts.predictions_csv)
    write_predictions_geojson(predictions, edges, artifacts.predictions_geojson)

    for step in progress(range(4), "[7/7] Visualizations", total=4):
        if step == 0:
            plot_surface_map(predictions, edges, artifacts.map_png, artifacts.legend_png)
        elif step == 1:
            plot_confusion_matrix_png(metrics, artifacts.confusion_matrix_png)
        elif step == 2:
            plot_feature_importance_png(feature_importance, artifacts.feature_importance_png)
        else:
            pass

    write_metrics_json(metrics, artifacts.metrics_json)
    write_model_joblib(pipeline, feature_importance, metrics, config, artifacts.model_joblib)
    write_model_info_xlsx(
        artifacts.model_info_xlsx,
        config=config,
        metrics=metrics,
        feature_importance=feature_importance,
        run_meta=run_meta,
        artifacts=artifacts,
    )
    write_report(
        artifacts.report_txt,
        config=config,
        summary=summary,
        metrics=metrics,
        run_meta=run_meta,
        artifacts=artifacts,
    )
    return artifacts

