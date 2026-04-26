"""Final experimental AI module for concrete OSM ``surface`` recovery.

The module predicts concrete OSM surface values such as ``asphalt`` or
``paving_stones`` and derives routing-oriented groups afterwards. It is kept
outside the routing path: generated predictions are exported for analysis and
future integration, but the live graph is not mutated.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ..config import Settings
from .surface_ml import (
    SATELLITE_FEATURES,
    SURFACE_GROUP_MAP,
    SURFACE_GROUPS,
    _first_osm_value,
    _geometry_feature_row,
    _import_pyplot,
    _json_safe,
    _utm_crs_for,
    extract_tile_features_for_edges,
    filter_edges_to_polygon,
    limit_edges_for_experiment,
    load_edges_for_precache_area,
    no_progress,
    progress_iter,
)
from .area_graph_cache import parse_precache_polygon


ProgressFactory = Callable[[Iterable[Any], str, Optional[int]], Iterable[Any]]


CONCRETE_SURFACE_VALUES: Tuple[str, ...] = (
    "asphalt",
    "concrete",
    "paved",
    "concrete:plates",
    "paving_stones",
    "sett",
    "cobblestone",
    "bricks",
    "compacted",
    "fine_gravel",
    "gravel",
    "pebblestone",
    "unpaved",
    "ground",
    "dirt",
    "earth",
    "sand",
    "grass",
    "mud",
)

SUPPORTED_SURFACE_VALUES: Tuple[str, ...] = CONCRETE_SURFACE_VALUES + (
    "unknown",
    "rare_other",
)

AI_OSM_FEATURES: Tuple[str, ...] = (
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
    "bridge",
    "tunnel",
    "junction",
    "area",
)

AI_GEOMETRY_FEATURES: Tuple[str, ...] = (
    "edge_length_m",
    "edge_bearing_deg",
    "edge_sinuosity",
    "edge_num_points",
    "edge_bbox_width_m",
    "edge_bbox_height_m",
    "distance_to_polygon_center_m",
)

AI_SATELLITE_FEATURES: Tuple[str, ...] = SATELLITE_FEATURES

REQUIRED_PROBA_SURFACES: Tuple[str, ...] = (
    "asphalt",
    "concrete",
    "paved",
    "paving_stones",
    "sett",
    "compacted",
    "fine_gravel",
    "gravel",
    "ground",
    "dirt",
    "sand",
    "grass",
    "unknown",
)

BAD_SURFACES: Tuple[str, ...] = (
    "unpaved",
    "ground",
    "dirt",
    "earth",
    "sand",
    "grass",
    "mud",
)

SURFACE_COLORS: Dict[str, str] = {
    "asphalt": "#1f78b4",
    "concrete": "#6baed6",
    "paved": "#756bb1",
    "concrete:plates": "#9e9ac8",
    "paving_stones": "#7b3294",
    "sett": "#c2a5cf",
    "cobblestone": "#9970ab",
    "bricks": "#b35806",
    "compacted": "#fdb863",
    "fine_gravel": "#e08214",
    "gravel": "#d95f0e",
    "pebblestone": "#fe9929",
    "unpaved": "#a63603",
    "ground": "#8c510a",
    "dirt": "#6b3d1f",
    "earth": "#5d4037",
    "sand": "#dfc27d",
    "grass": "#1b9e77",
    "mud": "#4d2c19",
    "rare_other": "#525252",
    "unknown": "#969696",
}

GROUP_COLORS: Dict[str, str] = {
    "paved_good": "#0B4F9C",
    "paved_rough": "#7B3294",
    "unpaved_hard": "#E08214",
    "unpaved_soft": "#8C2D04",
    "unknown": "#8A8A8A",
}


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _split_csv(raw: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in raw.split(",") if x.strip())


@dataclass(frozen=True)
class SurfaceAIConfig:
    enabled: bool = False
    mode: str = "experiment"
    target: str = "concrete"
    min_class_count: int = 50
    model_candidates: Tuple[str, ...] = (
        "always_majority",
        "highway_heuristic",
        "osm_only_rf",
        "satellite_only_rf",
        "combined_rf",
        "combined_rf_balanced",
    )
    selection_metric: str = "macro_f1"
    min_bad_surface_recall: float = 0.50
    use_satellite_features: bool = True
    sample_step_m: float = 7.0
    pixel_window: int = 5
    tms_server: str = "google"
    tile_zoom: int = 20
    tiles_dir: str = ""
    spatial_grid_m: float = 300.0
    test_share: float = 0.20
    random_state: int = 42
    rf_n_estimators: int = 300
    rf_class_weight: str = "balanced_subsample"
    min_confidence: float = 0.65
    conf_high: float = 0.85
    conf_medium: float = 0.65
    enable_safety_policy: bool = True
    paved_good_min_confidence: float = 0.90
    bad_surface_min_confidence: float = 0.60
    min_margin: float = 0.15
    out_dir: str = "./data/experiments/surface_ai"
    min_known_edges: int = 100
    max_tile_cache_items: int = 512

    @classmethod
    def from_env(cls) -> "SurfaceAIConfig":
        return cls(
            enabled=_env_bool("SURFACE_AI_ENABLED", False),
            mode=_env_str("SURFACE_AI_MODE", "experiment"),
            target=_env_str("SURFACE_AI_TARGET", "concrete"),
            min_class_count=_env_int("SURFACE_AI_MIN_CLASS_COUNT", 50),
            model_candidates=_split_csv(
                _env_str(
                    "SURFACE_AI_MODEL_CANDIDATES",
                    "always_majority,highway_heuristic,osm_only_rf,satellite_only_rf,combined_rf,combined_rf_balanced",
                )
            ),
            selection_metric=_env_str("SURFACE_AI_SELECTION_METRIC", "macro_f1"),
            min_bad_surface_recall=_env_float("SURFACE_AI_MIN_BAD_SURFACE_RECALL", 0.50),
            use_satellite_features=_env_bool("SURFACE_AI_USE_SATELLITE_FEATURES", True),
            sample_step_m=_env_float("SURFACE_AI_SAMPLE_STEP_M", 7.0),
            pixel_window=_env_int("SURFACE_AI_PIXEL_WINDOW", 5),
            tms_server=_env_str("SURFACE_AI_TMS_SERVER", "google"),
            tile_zoom=_env_int("SURFACE_AI_TILE_ZOOM", 20),
            tiles_dir=_env_str("SURFACE_AI_TILES_DIR", ""),
            spatial_grid_m=_env_float("SURFACE_AI_SPATIAL_GRID_M", 300.0),
            test_share=_env_float("SURFACE_AI_TEST_SHARE", 0.20),
            random_state=_env_int("SURFACE_AI_RANDOM_STATE", 42),
            rf_n_estimators=_env_int("SURFACE_AI_RF_N_ESTIMATORS", 300),
            rf_class_weight=_env_str("SURFACE_AI_RF_CLASS_WEIGHT", "balanced_subsample"),
            min_confidence=_env_float("SURFACE_AI_MIN_CONFIDENCE", 0.65),
            conf_high=_env_float("SURFACE_AI_CONF_HIGH", 0.85),
            conf_medium=_env_float("SURFACE_AI_CONF_MEDIUM", 0.65),
            enable_safety_policy=_env_bool("SURFACE_AI_ENABLE_SAFETY_POLICY", True),
            paved_good_min_confidence=_env_float("SURFACE_AI_PAVED_GOOD_MIN_CONFIDENCE", 0.90),
            bad_surface_min_confidence=_env_float("SURFACE_AI_BAD_SURFACE_MIN_CONFIDENCE", 0.60),
            min_margin=_env_float("SURFACE_AI_MIN_MARGIN", 0.15),
            out_dir=_env_str("SURFACE_AI_OUT_DIR", "./data/experiments/surface_ai"),
        )


@dataclass
class SurfaceAIArtifacts:
    output_dir: Path
    dataset_csv: Path
    predictions_csv: Path
    predictions_geojson: Path
    baseline_csv: Path
    baseline_xlsx: Path
    baseline_png: Path
    metrics_json: Path
    model_joblib: Path
    model_card_json: Path
    report_txt: Path
    surface_map_png: Path
    confidence_map_png: Path
    unknown_predictions_map_png: Path
    errors_map_png: Path
    confusion_matrix_concrete_png: Path
    confusion_matrix_group_png: Path
    feature_importance_png: Path


def normalize_surface_value(raw: Any) -> Optional[str]:
    value = _first_osm_value(raw)
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s or s in {"none", "null", "nan", "unknown", "no"}:
        return None
    if s == "yes":
        return "paved"

    s = s.replace("\\", "/")
    parts = [p.strip() for p in re.split(r"[;|,/]", s) if p.strip()]
    if not parts:
        parts = [s]

    aliases = {
        "cement": "concrete",
        "concrete_slabs": "concrete:plates",
        "concrete:plate": "concrete:plates",
        "cobblestone:flattened": "cobblestone",
        "paving_stones:30": "paving_stones",
        "paving_stones:20": "paving_stones",
        "fine_gravel;gravel": "fine_gravel",
        "grass_paver": "grass",
        "woodchips": "ground",
        "soil": "earth",
    }
    for part in parts:
        part = aliases.get(part, part)
        if part.startswith("paving_stones:"):
            part = "paving_stones"
        elif part.startswith("sett:"):
            part = "sett"
        elif part.startswith("cobblestone:"):
            part = "cobblestone"
        elif part.startswith("concrete:") and part != "concrete:plates":
            part = "concrete"
        if part in CONCRETE_SURFACE_VALUES:
            return part
    first = aliases.get(parts[0], parts[0])
    return first if first in CONCRETE_SURFACE_VALUES else None


def surface_to_group(surface: Any) -> str:
    s = normalize_surface_value(surface) if surface not in (None, "rare_other") else surface
    if s == "rare_other":
        return "unknown"
    return SURFACE_GROUP_MAP.get(str(s or ""), "unknown")


def train_label_to_surface(label: Any) -> str:
    s = str(label or "unknown")
    if s in CONCRETE_SURFACE_VALUES:
        return s
    if s.startswith("rare_other_"):
        return "rare_other"
    return "unknown"


def train_label_to_group(label: Any) -> str:
    s = str(label or "unknown")
    if s.startswith("rare_other_"):
        return s.replace("rare_other_", "", 1) or "unknown"
    return surface_to_group(s)


def proba_column_for_surface(surface: str) -> str:
    clean = re.sub(r"[^0-9a-zA-Z_]+", "_", surface).strip("_")
    return f"proba_{clean}"


def predict_surface_by_highway_heuristic(edge_attrs: dict) -> str:
    highway = str(edge_attrs.get("highway") or "").strip().lower()
    tracktype = str(edge_attrs.get("tracktype") or "").strip().lower()

    if highway in {"primary", "secondary", "tertiary", "residential", "living_street"}:
        return "asphalt"
    if highway in {"cycleway"}:
        return "asphalt"
    if highway in {"footway", "pedestrian"}:
        return "paving_stones"
    if highway == "service":
        return "asphalt"
    if highway == "track":
        if tracktype in {"grade1"}:
            return "compacted"
        if tracktype in {"grade2", "grade3"}:
            return "gravel"
        return "ground"
    if highway == "path":
        return "ground"
    if highway == "steps":
        return "concrete"
    return "unknown"


def _norm_feature(value: Any) -> Optional[str]:
    v = _first_osm_value(value)
    if v is None:
        return None
    s = str(v).strip().lower()
    if not s or s in {"nan", "none", "null"}:
        return None
    return s


def _geometry_features_with_center(
    row: Any,
    projected_geom: Any,
    center_projected: Any,
) -> Dict[str, Any]:
    item = _geometry_feature_row(row, projected_geom)
    try:
        item["distance_to_polygon_center_m"] = float(projected_geom.centroid.distance(center_projected))
    except Exception:
        item["distance_to_polygon_center_m"] = float("nan")
    return item


def build_osm_geometry_dataset(
    edges_gdf: gpd.GeoDataFrame,
    polygon: Any,
    *,
    progress: ProgressFactory = no_progress,
) -> pd.DataFrame:
    for col in AI_OSM_FEATURES:
        if col not in edges_gdf.columns:
            edges_gdf[col] = None
    projected_crs = _utm_crs_for(edges_gdf)
    projected = edges_gdf.to_crs(projected_crs)
    center = gpd.GeoSeries([polygon.centroid], crs="EPSG:4326").to_crs(projected_crs).iloc[0]

    rows: List[Dict[str, Any]] = []
    iterator = zip(edges_gdf.itertuples(index=False), projected.geometry.values)
    for row, projected_geom in progress(
        iterator,
        "[3/8] OSM/geometry features",
        total=len(edges_gdf),
    ):
        surface_norm = normalize_surface_value(getattr(row, "surface", None))
        true_group = surface_to_group(surface_norm)
        item: Dict[str, Any] = {
            "edge_id": row.edge_id,
            "u": str(row.u),
            "v": str(row.v),
            "key": str(row.key),
            "length_m": float(getattr(row, "length_m", np.nan)),
            "surface_raw": _first_osm_value(getattr(row, "surface", None)),
            "surface_norm": surface_norm or "unknown",
            "surface_osm_raw": _first_osm_value(getattr(row, "surface", None)),
            "surface_osm_norm": surface_norm or "unknown",
            "surface_true_concrete": surface_norm or "unknown",
            "surface_true_group": true_group,
            "surface_group_true": true_group,
            "is_surface_known": bool(surface_norm in CONCRETE_SURFACE_VALUES),
            "geometry_wkt": row.geometry.wkt if row.geometry is not None else None,
        }
        for col in AI_OSM_FEATURES:
            item[col] = _norm_feature(getattr(row, col, None))
        item.update(_geometry_features_with_center(row, projected_geom, center))
        rows.append(item)
    return pd.DataFrame(rows)


def _surface_train_label(surface: str, frequent: set[str]) -> Tuple[str, bool]:
    if surface in frequent:
        return surface, False
    group = surface_to_group(surface)
    if group in SURFACE_GROUPS:
        return f"rare_other_{group}", True
    return "rare_other", True


def add_train_labels(dataset: pd.DataFrame, *, min_class_count: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = dataset.copy()
    known = df[df["is_surface_known"]]
    counts = known["surface_true_concrete"].value_counts().sort_index()
    frequent = {str(k) for k, v in counts.items() if int(v) >= int(min_class_count)}
    train_labels: List[str] = []
    rare_flags: List[bool] = []
    for _, row in df.iterrows():
        surface = str(row.get("surface_true_concrete") or "unknown")
        if not bool(row.get("is_surface_known")) or surface == "unknown":
            train_labels.append("unknown")
            rare_flags.append(False)
            continue
        label, is_rare = _surface_train_label(surface, frequent)
        train_labels.append(label)
        rare_flags.append(is_rare)
    df["surface_train_label"] = train_labels
    df["surface_is_rare_class"] = rare_flags
    meta = {
        "surface_counts": {str(k): int(v) for k, v in counts.items()},
        "frequent_surface_classes": sorted(frequent),
        "rare_surface_classes": sorted(set(counts.index.astype(str)) - frequent),
        "min_class_count": int(min_class_count),
    }
    return df, meta


def build_surface_ai_dataset(
    edges_gdf: gpd.GeoDataFrame,
    settings: Settings,
    config: SurfaceAIConfig,
    polygon: Any,
    *,
    progress: ProgressFactory = no_progress,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    osm_df = build_osm_geometry_dataset(edges_gdf, polygon, progress=progress)
    if config.use_satellite_features:
        cache_dir = Path(settings.cache_dir)
        if config.tiles_dir:
            cache_dir = Path(config.tiles_dir).expanduser().resolve().parent
        tile_settings = SimpleNamespace(
            cache_dir=str(cache_dir),
            tms_server=config.tms_server or settings.tms_server,
            satellite_zoom=int(config.tile_zoom or settings.satellite_zoom),
        )
        ml_config = _surface_ml_config_from_ai(config)
        tile_df = extract_tile_features_for_edges(
            edges_gdf,
            tile_settings,
            ml_config,
            progress=progress,
        )
    else:
        rows = []
        for edge_id in osm_df["edge_id"].astype(str):
            item = {"edge_id": edge_id, "tile_missing": True, "tile_samples_count": 0}
            for col in AI_SATELLITE_FEATURES:
                item[col] = np.nan
            item["tile_missing_share"] = 1.0
            item["sampled_pixel_count"] = 0
            rows.append(item)
        tile_df = pd.DataFrame(rows)
    dataset = osm_df.merge(tile_df, on="edge_id", how="left", validate="one_to_one")
    for col in AI_SATELLITE_FEATURES:
        if col not in dataset.columns:
            dataset[col] = np.nan
    dataset, label_meta = add_train_labels(dataset, min_class_count=config.min_class_count)
    return dataset, label_meta


def _surface_ml_config_from_ai(config: SurfaceAIConfig) -> Any:
    from .surface_ml import SurfaceMLConfig

    return SurfaceMLConfig(
        sample_step_m=config.sample_step_m,
        pixel_window=config.pixel_window,
        min_confidence=config.min_confidence,
        paved_good_min_confidence=config.paved_good_min_confidence,
        spatial_grid_m=config.spatial_grid_m,
        test_share=config.test_share,
        random_state=config.random_state,
        min_known_edges=config.min_known_edges,
        n_estimators=config.rf_n_estimators,
        max_tile_cache_items=config.max_tile_cache_items,
    )


def spatial_train_test_split(
    dataset: pd.DataFrame,
    edges_gdf: gpd.GeoDataFrame,
    config: SurfaceAIConfig,
) -> pd.Series:
    known = dataset[dataset["is_surface_known"]].copy()
    if len(known) < int(config.min_known_edges):
        raise ValueError(
            f"Not enough known surface edges for training: found {len(known)}. "
            f"Need at least {config.min_known_edges}."
        )
    if known["surface_train_label"].nunique() < 2:
        raise ValueError("Need at least two concrete surface classes for training")

    edge_lookup = edges_gdf.set_index("edge_id")
    known_edges = edge_lookup.loc[known["edge_id"].astype(str)].reset_index()
    projected = known_edges.to_crs(_utm_crs_for(known_edges))
    cent = projected.geometry.centroid
    grid = max(1.0, float(config.spatial_grid_m))
    cells = pd.Series(
        [f"{int(math.floor(x / grid))}:{int(math.floor(y / grid))}" for x, y in zip(cent.x, cent.y)],
        index=known.index,
    )
    unique_cells = np.array(sorted(cells.unique()))
    if len(unique_cells) < 2:
        raise ValueError("Not enough spatial cells for spatial train/test split")
    rng = np.random.default_rng(int(config.random_state))
    rng.shuffle(unique_cells)
    test_n = max(1, int(round(len(unique_cells) * float(config.test_share))))
    test_n = min(test_n, len(unique_cells) - 1)
    test_cells = set(unique_cells[:test_n])

    # Keep every learnable label represented in train if possible.
    labels = set(known["surface_train_label"].astype(str))
    for label in sorted(labels):
        train_labels = set(known.loc[~cells.isin(test_cells), "surface_train_label"].astype(str))
        if label in train_labels:
            continue
        label_cells = [c for c in cells[known["surface_train_label"].astype(str) == label].unique() if c in test_cells]
        if label_cells:
            test_cells.remove(label_cells[0])

    split = pd.Series("unknown", index=dataset.index, dtype=object)
    split.loc[known.index[~cells.isin(test_cells).to_numpy()]] = "train"
    split.loc[known.index[cells.isin(test_cells).to_numpy()]] = "test"
    if (split == "train").sum() == 0 or (split == "test").sum() == 0:
        raise ValueError("Spatial split produced an empty train or test partition")
    return split


def _features_for_model(dataset: pd.DataFrame, feature_set: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    if feature_set == "osm":
        cat_cols = list(AI_OSM_FEATURES)
        num_cols: List[str] = []
    elif feature_set == "satellite":
        cat_cols = []
        num_cols = list(AI_SATELLITE_FEATURES)
    elif feature_set == "combined":
        cat_cols = list(AI_OSM_FEATURES)
        num_cols = list(AI_GEOMETRY_FEATURES) + list(AI_SATELLITE_FEATURES)
    elif feature_set == "osm_geometry":
        cat_cols = list(AI_OSM_FEATURES)
        num_cols = list(AI_GEOMETRY_FEATURES)
    else:
        cat_cols = []
        num_cols = []
    cols = cat_cols + num_cols
    X = dataset.reindex(columns=cols).copy()
    for col in cat_cols:
        X[col] = X[col].where(X[col].notna(), None)
    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X, cat_cols, num_cols


def _build_preprocessor(cat_cols: Sequence[str], num_cols: Sequence[str]) -> ColumnTransformer:
    transformers = []
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                list(cat_cols),
            )
        )
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                list(num_cols),
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)


def _rf(config: SurfaceAIConfig, *, class_weight: Optional[str]) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=int(config.rf_n_estimators),
        random_state=int(config.random_state),
        class_weight=class_weight,
        n_jobs=-1,
        min_samples_leaf=2,
    )


def _build_pipeline(candidate: str, feature_set: str, config: SurfaceAIConfig) -> Pipeline:
    _, cat_cols, num_cols = _features_for_model(pd.DataFrame(), feature_set)
    preprocessor = _build_preprocessor(cat_cols, num_cols)
    if candidate == "combined_hist_gradient_boosting":
        model: Any = HistGradientBoostingClassifier(random_state=int(config.random_state))
    elif candidate == "combined_logistic_regression":
        model = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    else:
        cw = None
        if candidate in {"combined_rf_balanced", "combined_balanced_random_forest"}:
            cw = config.rf_class_weight or "balanced_subsample"
        model = _rf(config, class_weight=cw)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


class HighwayHeuristicClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, train_label_classes: Sequence[str], frequent_surfaces: Sequence[str]) -> None:
        self.classes_ = np.array(sorted(set(train_label_classes)))
        self.frequent_surfaces_ = set(frequent_surfaces)

    def fit(self, X: pd.DataFrame, y: Sequence[str]) -> "HighwayHeuristicClassifier":
        self.classes_ = np.array(sorted(set(y)))
        return self

    def _to_label(self, surface: str) -> str:
        if surface in self.frequent_surfaces_:
            return surface
        group = surface_to_group(surface)
        label = f"rare_other_{group}" if group in SURFACE_GROUPS else "rare_other"
        return label if label in set(self.classes_) else "unknown"

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        labels = []
        for _, row in X.iterrows():
            surface = predict_surface_by_highway_heuristic(row.to_dict())
            labels.append(self._to_label(surface))
        return np.array(labels, dtype=object)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pred = self.predict(X)
        proba = np.zeros((len(pred), len(self.classes_)), dtype=float)
        lookup = {c: i for i, c in enumerate(self.classes_)}
        for i, label in enumerate(pred):
            j = lookup.get(label)
            if j is not None:
                proba[i, j] = 1.0
        return proba


def _candidate_feature_set(candidate: str) -> str:
    if candidate == "osm_only_rf":
        return "osm"
    if candidate == "satellite_only_rf":
        return "satellite"
    if candidate in {
        "combined_rf",
        "combined_rf_balanced",
        "combined_balanced_random_forest",
        "combined_hist_gradient_boosting",
        "combined_logistic_regression",
    }:
        return "combined"
    if candidate == "highway_heuristic":
        return "osm"
    return "none"


def _candidate_display(candidate: str) -> Tuple[str, str]:
    mapping = {
        "always_majority": ("Always majority", "none"),
        "highway_heuristic": ("Highway heuristic", "highway only"),
        "osm_only_rf": ("OSM-only RF", "OSM tags"),
        "satellite_only_rf": ("Satellite-only RF", "satellite stats"),
        "combined_rf": ("Combined RF", "OSM + geometry + satellite"),
        "combined_rf_balanced": ("Combined balanced RF", "OSM + geometry + satellite"),
        "combined_balanced_random_forest": ("Combined balanced RF", "OSM + geometry + satellite"),
        "combined_hist_gradient_boosting": ("Combined HistGradientBoosting", "OSM + geometry + satellite"),
        "combined_logistic_regression": ("Combined logistic regression", "OSM + geometry + satellite"),
    }
    return mapping.get(candidate, (candidate, _candidate_feature_set(candidate)))


def _metrics_for_predictions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    labels: Sequence[str],
    target: str,
) -> Dict[str, Any]:
    labels = list(labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
        output_dict=True,
    )
    true_groups = pd.Series(y_true).map(train_label_to_group)
    pred_groups = pd.Series(y_pred).map(train_label_to_group)
    bad_mask = true_groups.isin({"unpaved_hard", "unpaved_soft"})
    recall_bad = float((pred_groups[bad_mask].isin({"unpaved_hard", "unpaved_soft"})).mean()) if bad_mask.any() else 0.0
    unpaved_soft_mask = true_groups == "unpaved_soft"
    recall_unpaved_soft = float((pred_groups[unpaved_soft_mask] == "unpaved_soft").mean()) if unpaved_soft_mask.any() else 0.0
    return {
        "target": target,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_bad_surface": recall_bad,
        "recall_unpaved_soft": recall_unpaved_soft,
        "recall_asphalt": float(report.get("asphalt", {}).get("recall", 0.0)),
        "recall_concrete": float(report.get("concrete", {}).get("recall", 0.0)),
        "recall_paving_stones": float(report.get("paving_stones", {}).get("recall", 0.0)),
        "classification_report": report,
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
    }


def _group_metrics_for_predictions(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, Any]:
    ytg = pd.Series(y_true).map(train_label_to_group).tolist()
    ypg = pd.Series(y_pred).map(train_label_to_group).tolist()
    labels = list(SURFACE_GROUPS)
    report = classification_report(ytg, ypg, labels=labels, zero_division=0, output_dict=True)
    return {
        "target": "group",
        "accuracy": float(accuracy_score(ytg, ypg)),
        "balanced_accuracy": float(balanced_accuracy_score(ytg, ypg)),
        "macro_f1": float(f1_score(ytg, ypg, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(ytg, ypg, average="weighted", zero_division=0)),
        "recall_unpaved_soft": float(report.get("unpaved_soft", {}).get("recall", 0.0)),
        "classification_report": report,
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(ytg, ypg, labels=labels).astype(int).tolist(),
    }


def train_evaluate_models(
    dataset: pd.DataFrame,
    config: SurfaceAIConfig,
    label_meta: Dict[str, Any],
    *,
    progress: ProgressFactory = no_progress,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    train = dataset[dataset["surface_ai_split"] == "train"].copy()
    test = dataset[dataset["surface_ai_split"] == "test"].copy()
    y_train = train["surface_train_label"].astype(str)
    y_test = test["surface_train_label"].astype(str)
    labels = sorted(set(y_train) | set(y_test))
    frequent = label_meta.get("frequent_surface_classes", [])
    models: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []
    test_predictions: Dict[str, List[str]] = {}

    candidates = list(dict.fromkeys(config.model_candidates))
    for candidate in progress(candidates, "[5/8] Train/evaluate models", total=len(candidates)):
        display, features_label = _candidate_display(candidate)
        feature_set = _candidate_feature_set(candidate)
        if candidate == "always_majority":
            model: Any = DummyClassifier(strategy="most_frequent")
            X_train = pd.DataFrame({"constant": np.ones(len(train))})
            X_test = pd.DataFrame({"constant": np.ones(len(test))})
        elif candidate == "highway_heuristic":
            model = HighwayHeuristicClassifier(labels, frequent)
            X_train, _, _ = _features_for_model(train, "osm")
            X_test, _, _ = _features_for_model(test, "osm")
        else:
            model = _build_pipeline(candidate, feature_set, config)
            X_train, _, _ = _features_for_model(train, feature_set)
            X_test, _, _ = _features_for_model(test, feature_set)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        pred = [str(x) for x in pred]
        test_predictions[candidate] = pred
        concrete_metrics = _metrics_for_predictions(
            y_test.tolist(),
            pred,
            labels=labels,
            target="surface_concrete",
        )
        group_metrics = _group_metrics_for_predictions(y_test.tolist(), pred)
        row = {
            "model_key": candidate,
            "model": display,
            "target": "concrete surface",
            "features": features_label,
            "accuracy": concrete_metrics["accuracy"],
            "balanced_accuracy": concrete_metrics["balanced_accuracy"],
            "macro_f1": concrete_metrics["macro_f1"],
            "weighted_f1": concrete_metrics["weighted_f1"],
            "recall_bad_surface": concrete_metrics["recall_bad_surface"],
            "recall_unpaved_soft": concrete_metrics["recall_unpaved_soft"],
            "recall_asphalt": concrete_metrics["recall_asphalt"],
            "recall_concrete": concrete_metrics["recall_concrete"],
            "recall_paving_stones": concrete_metrics["recall_paving_stones"],
        }
        rows.append(row)
        group_rows.append(
            {
                "model_key": candidate,
                "model": display,
                "target": "group",
                "features": features_label,
                "accuracy": group_metrics["accuracy"],
                "balanced_accuracy": group_metrics["balanced_accuracy"],
                "macro_f1": group_metrics["macro_f1"],
                "weighted_f1": group_metrics["weighted_f1"],
                "recall_bad_surface": concrete_metrics["recall_bad_surface"],
                "recall_unpaved_soft": group_metrics["recall_unpaved_soft"],
                "recall_asphalt": np.nan,
                "recall_concrete": np.nan,
                "recall_paving_stones": np.nan,
            }
        )
        models[candidate] = {
            "model": model,
            "feature_set": feature_set,
            "concrete_metrics": concrete_metrics,
            "group_metrics": group_metrics,
            "test_pred": pred,
        }

    baseline_table = pd.DataFrame(rows + group_rows)
    selected = select_best_model(models, baseline_table, config)
    models["_selected"] = selected
    return models, baseline_table, pd.DataFrame(rows)


def select_best_model(
    models: Dict[str, Any],
    baseline_table: pd.DataFrame,
    config: SurfaceAIConfig,
) -> Dict[str, Any]:
    concrete = baseline_table[baseline_table["target"] == "concrete surface"].copy()
    eligible = concrete[concrete["recall_bad_surface"] >= float(config.min_bad_surface_recall)].copy()
    safety_applied = True
    if eligible.empty:
        eligible = concrete
        safety_applied = False
    metric = config.selection_metric if config.selection_metric in eligible.columns else "macro_f1"
    eligible = eligible.sort_values(
        by=[metric, "balanced_accuracy", "recall_bad_surface"],
        ascending=[False, False, False],
    )
    best_key = str(eligible.iloc[0]["model_key"])
    return {
        "model_key": best_key,
        "model_name": str(eligible.iloc[0]["model"]),
        "selected_by": metric,
        "safety_recall_threshold": float(config.min_bad_surface_recall),
        "safety_recall_threshold_satisfied": bool(safety_applied),
        "metrics": models[best_key]["concrete_metrics"],
        "group_metrics": models[best_key]["group_metrics"],
    }


def _predict_model_proba(model: Any, X: pd.DataFrame, classes: Sequence[str]) -> pd.DataFrame:
    classes = list(classes)
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(X)
        model_classes = [str(c) for c in getattr(model, "classes_", classes)]
        out = pd.DataFrame(0.0, index=X.index, columns=classes)
        for i, cls in enumerate(model_classes):
            if cls in out.columns:
                out[cls] = raw[:, i]
        return out
    pred = [str(x) for x in model.predict(X)]
    out = pd.DataFrame(0.0, index=X.index, columns=classes)
    for i, cls in enumerate(pred):
        if cls in out.columns:
            out.loc[X.index[i], cls] = 1.0
    return out


def _has_features(row: pd.Series, feature_set: str) -> bool:
    _, cat_cols, num_cols = _features_for_model(pd.DataFrame(), feature_set)
    for col in cat_cols:
        if _norm_feature(row.get(col)):
            return True
    for col in num_cols:
        v = pd.to_numeric(row.get(col), errors="coerce")
        if pd.notna(v):
            return True
    return False


def apply_safety_policy(
    *,
    is_surface_known: bool,
    true_surface: str,
    pred_label: str,
    confidence: float,
    margin: float,
    has_features: bool,
    config: SurfaceAIConfig,
) -> Tuple[str, str, str]:
    pred_surface = train_label_to_surface(pred_label)
    pred_group = train_label_to_group(pred_label)
    if is_surface_known and true_surface != "unknown":
        return true_surface, surface_to_group(true_surface), "osm"
    if not has_features:
        return "unknown", "unknown", "default_no_features"
    if not config.enable_safety_policy:
        if confidence >= config.min_confidence:
            return pred_surface, pred_group, "ml"
        return "unknown", "unknown", "default_low_confidence"
    if margin < float(config.min_margin):
        return "unknown", "unknown", "ml_ambiguous"
    if confidence < float(config.min_confidence):
        return "unknown", "unknown", "default_low_confidence"
    if pred_surface in BAD_SURFACES and confidence >= float(config.bad_surface_min_confidence):
        return pred_surface, pred_group, "ml"
    if pred_group == "paved_good" and confidence < float(config.paved_good_min_confidence):
        return "paved", "paved_rough", "ml"
    if pred_surface == "rare_other":
        return "unknown", pred_group, "ml"
    return pred_surface, pred_group, "ml"


def predict_all_edges(
    dataset: pd.DataFrame,
    models: Dict[str, Any],
    config: SurfaceAIConfig,
    *,
    progress: ProgressFactory = no_progress,
) -> pd.DataFrame:
    selected = models["_selected"]
    model_key = selected["model_key"]
    model = models[model_key]["model"]
    feature_set = models[model_key]["feature_set"]
    classes = sorted(set(dataset.loc[dataset["is_surface_known"], "surface_train_label"].astype(str)))
    X, _, _ = _features_for_model(dataset, feature_set if feature_set != "none" else "osm")
    for _ in progress(range(1), "[6/8] Predict surfaces", total=1):
        if model_key == "always_majority":
            X = pd.DataFrame({"constant": np.ones(len(dataset))}, index=dataset.index)
        proba = _predict_model_proba(model, X, classes)

    pred_labels: List[str] = []
    confs: List[float] = []
    top2_labels: List[str] = []
    top2_confs: List[float] = []
    margins: List[float] = []
    for idx, row in proba.iterrows():
        ordered = row.sort_values(ascending=False)
        top1 = str(ordered.index[0]) if len(ordered) else "unknown"
        top1_conf = float(ordered.iloc[0]) if len(ordered) else 0.0
        top2 = str(ordered.index[1]) if len(ordered) > 1 else "unknown"
        top2_conf = float(ordered.iloc[1]) if len(ordered) > 1 else 0.0
        pred_labels.append(top1)
        confs.append(top1_conf)
        top2_labels.append(train_label_to_surface(top2))
        top2_confs.append(top2_conf)
        margins.append(top1_conf - top2_conf)

    out = dataset.copy()
    out["surface_pred_label"] = pred_labels
    out["surface_pred_concrete"] = [train_label_to_surface(x) for x in pred_labels]
    out["surface_pred"] = out["surface_pred_concrete"]
    out["surface_pred_group"] = [train_label_to_group(x) for x in pred_labels]
    out["surface_group_pred"] = out["surface_pred_group"]
    out["surface_pred_confidence"] = confs
    out["surface_pred_top2"] = top2_labels
    out["surface_pred_top2_confidence"] = top2_confs
    out["surface_pred_margin"] = margins

    for surface in sorted(set(classes) | set(REQUIRED_PROBA_SURFACES)):
        col = proba_column_for_surface(train_label_to_surface(surface))
        if col not in out.columns:
            out[col] = 0.0
    for label in proba.columns:
        surface = train_label_to_surface(str(label))
        col = proba_column_for_surface(surface)
        out[col] = out[col].to_numpy(dtype=float) + proba[label].to_numpy(dtype=float)
    if "proba_unknown" not in out.columns:
        out["proba_unknown"] = 0.0

    effective: List[str] = []
    effective_group: List[str] = []
    sources: List[str] = []
    for idx, row in out.iterrows():
        has_features = _has_features(row, feature_set)
        eff, eff_group, source = apply_safety_policy(
            is_surface_known=bool(row.get("is_surface_known")),
            true_surface=str(row.get("surface_true_concrete") or "unknown"),
            pred_label=str(row.get("surface_pred_label") or "unknown"),
            confidence=float(row.get("surface_pred_confidence") or 0.0),
            margin=float(row.get("surface_pred_margin") or 0.0),
            has_features=has_features,
            config=config,
        )
        effective.append(eff)
        effective_group.append(eff_group)
        sources.append(source)
    out["surface_source"] = sources
    out["surface_effective_for_routing"] = effective
    out["surface_effective_group_for_routing"] = effective_group
    return out


def dataset_summary(dataset: pd.DataFrame) -> Dict[str, Any]:
    total = int(len(dataset))
    known = int(dataset["is_surface_known"].sum()) if total else 0
    return {
        "total_edges": total,
        "known_surface_edges": known,
        "unknown_surface_edges": total - known,
        "unknown_share": float((total - known) / total) if total else 0.0,
        "surface_norm_distribution": dataset["surface_true_concrete"].value_counts(dropna=False).to_dict(),
        "surface_train_label_distribution": dataset.loc[
            dataset["is_surface_known"], "surface_train_label"
        ].value_counts(dropna=False).to_dict(),
        "surface_group_distribution": dataset["surface_true_group"].value_counts(dropna=False).to_dict(),
        "highway_distribution": dataset["highway"].value_counts(dropna=False).head(50).to_dict(),
    }


def experiment_output_dir(settings: Settings, config: SurfaceAIConfig, output_dir: Optional[Path]) -> Path:
    if output_dir is not None:
        out = Path(output_dir)
    else:
        stamp = datetime.now().strftime("surface_ai_%Y%m%d_%H%M%S")
        raw = (config.out_dir or "").strip()
        if raw:
            base = Path(raw)
            if not base.is_absolute():
                base = Path.cwd() / base
            out = base.parent / stamp if base.name == "surface_ai" else base / stamp
        else:
            out = Path(settings.base_dir) / "experiments" / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def artifact_paths(output_dir: Path) -> SurfaceAIArtifacts:
    return SurfaceAIArtifacts(
        output_dir=output_dir,
        dataset_csv=output_dir / "surface_ai_dataset.csv",
        predictions_csv=output_dir / "surface_ai_predictions.csv",
        predictions_geojson=output_dir / "surface_ai_predictions.geojson",
        baseline_csv=output_dir / "surface_ai_baseline_table.csv",
        baseline_xlsx=output_dir / "surface_ai_baseline_table.xlsx",
        baseline_png=output_dir / "surface_ai_baseline_table.png",
        metrics_json=output_dir / "surface_ai_metrics.json",
        model_joblib=output_dir / "surface_ai_model.joblib",
        model_card_json=output_dir / "surface_ai_model_card.json",
        report_txt=output_dir / "surface_ai_report.txt",
        surface_map_png=output_dir / "surface_ai_surface_map.png",
        confidence_map_png=output_dir / "surface_ai_confidence_map.png",
        unknown_predictions_map_png=output_dir / "surface_ai_unknown_predictions_map.png",
        errors_map_png=output_dir / "surface_ai_errors_map.png",
        confusion_matrix_concrete_png=output_dir / "surface_ai_confusion_matrix_concrete.png",
        confusion_matrix_group_png=output_dir / "surface_ai_confusion_matrix_group.png",
        feature_importance_png=output_dir / "surface_ai_feature_importance.png",
    )


def write_dataset_csv(dataset: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(path, index=False, encoding="utf-8")


def prediction_columns(predictions: pd.DataFrame) -> List[str]:
    required = [
        "edge_id",
        "u",
        "v",
        "key",
        "length_m",
        "highway",
        "surface_osm_raw",
        "surface_osm_norm",
        "surface_true_concrete",
        "surface_true_group",
        "is_surface_known",
        "surface_train_label",
        "surface_pred_concrete",
        "surface_pred_group",
        "surface_group_pred",
        "surface_pred_confidence",
        "surface_pred_top2",
        "surface_pred_top2_confidence",
        "surface_pred_margin",
        "surface_source",
    ]
    required.extend(proba_column_for_surface(s) for s in REQUIRED_PROBA_SURFACES)
    required.extend(
        [
            "surface_is_rare_class",
            "surface_effective_for_routing",
            "surface_effective_group_for_routing",
            "tile_missing_share",
            "sampled_pixel_count",
            "rgb_mean_r",
            "rgb_mean_g",
            "rgb_mean_b",
            "brightness_mean",
            "saturation_mean",
            "texture_std",
            "geometry_wkt",
        ]
    )
    cols = [c for c in required if c in predictions.columns]
    extra = [c for c in predictions.columns if c not in cols and c != "geometry"]
    return cols + extra


def write_predictions_csv(predictions: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions[prediction_columns(predictions)].to_csv(path, index=False, encoding="utf-8")


def write_predictions_geojson(predictions: pd.DataFrame, edges_gdf: gpd.GeoDataFrame, path: Path) -> None:
    base = edges_gdf[["edge_id", "geometry"]].copy()
    gdf = base.merge(predictions.drop(columns=["geometry"], errors="ignore"), on="edge_id", how="left")
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GeoJSON")


def write_baseline_tables(baseline_table: pd.DataFrame, artifacts: SurfaceAIArtifacts) -> None:
    baseline_table.to_csv(artifacts.baseline_csv, index=False, encoding="utf-8")
    with pd.ExcelWriter(artifacts.baseline_xlsx, engine="openpyxl") as writer:
        baseline_table.to_excel(writer, sheet_name="all", index=False)
        baseline_table[baseline_table["target"] == "concrete surface"].to_excel(
            writer, sheet_name="concrete_surface", index=False
        )
        baseline_table[baseline_table["target"] == "group"].to_excel(
            writer, sheet_name="group", index=False
        )
    plot_baseline_table_png(baseline_table, artifacts.baseline_png)


def plot_baseline_table_png(baseline_table: pd.DataFrame, path: Path) -> None:
    plt = _import_pyplot()
    cols = [
        "model",
        "target",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "recall_bad_surface",
        "recall_unpaved_soft",
    ]
    df = baseline_table[cols].copy()
    for col in cols[2:]:
        df[col] = df[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    fig_h = max(4.0, 0.42 * len(df) + 1.0)
    fig, ax = plt.subplots(figsize=(15, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.35)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_lines_by_column(
    edges_gdf: gpd.GeoDataFrame,
    predictions: pd.DataFrame,
    *,
    value_col: str,
    color_map: Dict[str, str],
    path: Path,
    only_mask: Optional[pd.Series] = None,
    title: str = "",
) -> None:
    plt = _import_pyplot()
    gdf = edges_gdf[["edge_id", "geometry"]].merge(
        predictions[["edge_id", value_col]],
        on="edge_id",
        how="left",
    )
    if only_mask is not None:
        allowed = set(predictions.loc[only_mask, "edge_id"].astype(str))
        gdf = gdf[gdf["edge_id"].astype(str).isin(allowed)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    if gdf.empty:
        ax.text(0.5, 0.5, "No edges", ha="center", va="center", transform=ax.transAxes)
    else:
        for value, color in color_map.items():
            subset = gdf[gdf[value_col] == value]
            if subset.empty:
                continue
            subset.plot(ax=ax, color=color, linewidth=1.3, alpha=0.9, label=value)
        rest = gdf[~gdf[value_col].isin(color_map.keys())]
        if not rest.empty:
            rest.plot(ax=ax, color="#969696", linewidth=0.8, alpha=0.7, label="other")
        ax.legend(loc="lower left", fontsize=7, frameon=True)
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_surface_maps(predictions: pd.DataFrame, edges_gdf: gpd.GeoDataFrame, artifacts: SurfaceAIArtifacts, config: SurfaceAIConfig) -> None:
    _plot_lines_by_column(
        edges_gdf,
        predictions,
        value_col="surface_pred_group",
        color_map=GROUP_COLORS,
        path=artifacts.surface_map_png,
        title="Predicted surface group",
    )
    unknown_mask = ~predictions["is_surface_known"].astype(bool)
    _plot_lines_by_column(
        edges_gdf,
        predictions,
        value_col="surface_pred_group",
        color_map=GROUP_COLORS,
        path=artifacts.unknown_predictions_map_png,
        only_mask=unknown_mask,
        title="Unknown OSM surface predictions",
    )
    plot_confidence_map(predictions, edges_gdf, artifacts.confidence_map_png, config)
    plot_errors_map(predictions, edges_gdf, artifacts.errors_map_png)


def plot_confidence_map(predictions: pd.DataFrame, edges_gdf: gpd.GeoDataFrame, path: Path, config: SurfaceAIConfig) -> None:
    plt = _import_pyplot()
    gdf = edges_gdf[["edge_id", "geometry"]].merge(
        predictions[["edge_id", "is_surface_known", "surface_source", "surface_pred_confidence"]],
        on="edge_id",
        how="left",
    )

    def bucket(row: pd.Series) -> str:
        if bool(row.get("is_surface_known")):
            return "osm_known"
        src = str(row.get("surface_source") or "")
        if src.startswith("default") or src in {"unknown", "ml_ambiguous"}:
            return "default"
        c = float(row.get("surface_pred_confidence") or 0.0)
        if c >= config.conf_high:
            return "high"
        if c >= config.conf_medium:
            return "medium"
        return "low"

    gdf["confidence_bucket"] = gdf.apply(bucket, axis=1)
    colors = {
        "high": "#1a9850",
        "medium": "#fee08b",
        "low": "#d73027",
        "default": "#969696",
        "osm_known": "#2b2b2b",
    }
    widths = {"osm_known": 0.6, "default": 0.8, "low": 1.2, "medium": 1.2, "high": 1.2}
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_axis_off()
    for key, color in colors.items():
        subset = gdf[gdf["confidence_bucket"] == key]
        if subset.empty:
            continue
        subset.plot(ax=ax, color=color, linewidth=widths[key], alpha=0.85, label=key)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_errors_map(predictions: pd.DataFrame, edges_gdf: gpd.GeoDataFrame, path: Path) -> None:
    err_mask = (
        (predictions["surface_ai_split"] == "test")
        & (predictions["surface_train_label"].astype(str) != predictions["surface_pred_label"].astype(str))
    )
    plt = _import_pyplot()
    gdf = edges_gdf[["edge_id", "geometry"]].merge(
        predictions[["edge_id"]],
        on="edge_id",
        how="inner",
    )
    error_ids = set(predictions.loc[err_mask, "edge_id"].astype(str))
    gdf = gdf[gdf["edge_id"].astype(str).isin(error_ids)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_axis_off()
    if gdf.empty:
        ax.text(0.5, 0.5, "No test errors", ha="center", va="center", transform=ax.transAxes)
    else:
        gdf.plot(ax=ax, color="#d73027", linewidth=1.8, alpha=0.95)
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_confusion_matrix(metrics: Dict[str, Any], path: Path, *, title: str) -> None:
    plt = _import_pyplot()
    labels = list(metrics.get("confusion_matrix_labels") or [])
    cm = np.asarray(metrics.get("confusion_matrix") or np.zeros((len(labels), len(labels))), dtype=float)
    fig_w = max(7.0, 0.55 * max(1, len(labels)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_w))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="#111111", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def feature_importance_df(model: Any) -> pd.DataFrame:
    if not isinstance(model, Pipeline):
        return pd.DataFrame(columns=["feature", "importance"])
    estimator = model.named_steps.get("model")
    if not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    pre = model.named_steps.get("preprocess")
    try:
        names = list(pre.get_feature_names_out())
    except Exception:
        names = [f"feature_{i}" for i in range(len(estimator.feature_importances_))]
    values = np.asarray(estimator.feature_importances_, dtype=float)
    n = min(len(names), len(values))
    return (
        pd.DataFrame({"feature": names[:n], "importance": values[:n]})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def plot_feature_importance(feature_importance: pd.DataFrame, path: Path) -> None:
    plt = _import_pyplot()
    if feature_importance.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No feature importance for selected model", ha="center", va="center")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return
    df = feature_importance.head(30).iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.34 * len(df))))
    ax.barh(df["feature"], df["importance"], color="#3B6EA8")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_model_card(
    artifacts: SurfaceAIArtifacts,
    *,
    selected: Dict[str, Any],
    config: SurfaceAIConfig,
    dataset: pd.DataFrame,
) -> Dict[str, Any]:
    known = dataset[dataset["is_surface_known"]]
    train = dataset[dataset["surface_ai_split"] == "train"]
    test = dataset[dataset["surface_ai_split"] == "test"]
    metrics = selected["metrics"]
    feature_set = models_feature_set_from_key(selected["model_key"])
    card = {
        "model_name": f"surface_ai_{selected['model_key']}",
        "target": "surface_concrete",
        "classes": sorted(known["surface_train_label"].astype(str).unique().tolist()),
        "features": feature_set,
        "training_edges": int(len(train)),
        "test_edges": int(len(test)),
        "spatial_split_m": float(config.spatial_grid_m),
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "selected_by": selected.get("selected_by"),
        "limitations": [
            "class imbalance",
            "rare surfaces are hard to learn",
            "satellite tiles contain shadows/cars/tree crowns",
        ],
        "safe_to_use_for_routing": False,
    }
    with open(artifacts.model_card_json, "w", encoding="utf-8") as f:
        json.dump(_json_safe(card), f, ensure_ascii=False, indent=2)
    return card


def models_feature_set_from_key(model_key: str) -> List[str]:
    fs = _candidate_feature_set(model_key)
    if fs == "osm":
        return ["osm"]
    if fs == "satellite":
        return ["satellite"]
    if fs == "combined":
        return ["osm", "geometry", "satellite"]
    if fs == "osm_geometry":
        return ["osm", "geometry"]
    return []


def write_report(
    artifacts: SurfaceAIArtifacts,
    *,
    summary: Dict[str, Any],
    label_meta: Dict[str, Any],
    selected: Dict[str, Any],
    baseline_table: pd.DataFrame,
    config: SurfaceAIConfig,
    run_meta: Dict[str, Any],
) -> None:
    top = baseline_table[baseline_table["target"] == "concrete surface"].sort_values(
        ["macro_f1", "balanced_accuracy"],
        ascending=False,
    )
    lines = [
        "Surface AI experiment report",
        "",
        f"Output dir: {artifacts.output_dir}",
        f"Graph source: {run_meta.get('graph_source')}",
        f"Graph path: {run_meta.get('graph_path')}",
        f"Tiles dir: {run_meta.get('tiles_dir')}",
        f"Tile zoom: {run_meta.get('tile_zoom')}",
        "",
        "Dataset:",
        f"  Total edges: {summary.get('total_edges')}",
        f"  Known surface edges: {summary.get('known_surface_edges')}",
        f"  Unknown surface edges: {summary.get('unknown_surface_edges')}",
        f"  Unknown share: {summary.get('unknown_share'):.3f}",
        f"  Frequent concrete classes: {label_meta.get('frequent_surface_classes')}",
        f"  Rare concrete classes: {label_meta.get('rare_surface_classes')}",
        "",
        "Model selection:",
        f"  Selected model: {selected.get('model_name')} ({selected.get('model_key')})",
        f"  Main criterion: {selected.get('selected_by')}",
        f"  Safety recall threshold: {selected.get('safety_recall_threshold')}",
        f"  Safety threshold satisfied by selected pool: {selected.get('safety_recall_threshold_satisfied')}",
        f"  Selected metrics: {selected.get('metrics')}",
        "",
        "Baseline ranking by macro_f1:",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"  {row['model']}: macro_f1={row['macro_f1']:.4f}, "
            f"balanced_accuracy={row['balanced_accuracy']:.4f}, "
            f"recall_bad_surface={row['recall_bad_surface']:.4f}"
        )
    lines.extend(
        [
            "",
            "Safety policy:",
            f"  enabled: {config.enable_safety_policy}",
            f"  min_confidence: {config.min_confidence}",
            f"  min_margin: {config.min_margin}",
            f"  paved_good_min_confidence: {config.paved_good_min_confidence}",
            f"  bad_surface_min_confidence: {config.bad_surface_min_confidence}",
            "  Known OSM surface is never replaced by ML.",
            "  Main routing is not connected to this model yet.",
            "",
            "Limitations:",
            "  Satellite tiles can include shadows, cars, tree crowns, roofs, and yards.",
            "  Rare surface classes are grouped into rare_other_* train labels.",
            "  The output is prepared for future routing integration but safe_to_use_for_routing=false.",
            "",
            "Artifacts:",
        ]
    )
    for field in artifacts.__dataclass_fields__:
        if field != "output_dir":
            lines.append(f"  {field}: {getattr(artifacts, field)}")
    artifacts.report_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metrics_json(
    artifacts: SurfaceAIArtifacts,
    *,
    selected: Dict[str, Any],
    models: Dict[str, Any],
    baseline_table: pd.DataFrame,
    summary: Dict[str, Any],
    label_meta: Dict[str, Any],
    config: SurfaceAIConfig,
    run_meta: Dict[str, Any],
) -> None:
    payload = {
        "selected_model": selected,
        "model_metrics": {
            key: {
                "concrete_metrics": value.get("concrete_metrics"),
                "group_metrics": value.get("group_metrics"),
                "feature_set": value.get("feature_set"),
            }
            for key, value in models.items()
            if not key.startswith("_")
        },
        "baseline_table": baseline_table.to_dict(orient="records"),
        "dataset_summary": summary,
        "label_meta": label_meta,
        "config": asdict(config),
        "run_meta": run_meta,
    }
    with open(artifacts.metrics_json, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2)


def write_model_joblib(
    artifacts: SurfaceAIArtifacts,
    *,
    selected: Dict[str, Any],
    models: Dict[str, Any],
    feature_importance: pd.DataFrame,
    config: SurfaceAIConfig,
    label_meta: Dict[str, Any],
) -> None:
    key = selected["model_key"]
    bundle = {
        "model": models[key]["model"],
        "model_key": key,
        "feature_set": models[key]["feature_set"],
        "selected": selected,
        "surface_values": CONCRETE_SURFACE_VALUES,
        "surface_group_map": SURFACE_GROUP_MAP,
        "label_meta": label_meta,
        "feature_importance": feature_importance,
        "config": asdict(config),
    }
    joblib.dump(bundle, artifacts.model_joblib)


def run_surface_ai_experiment(
    *,
    settings: Optional[Settings] = None,
    config: Optional[SurfaceAIConfig] = None,
    max_edges: Optional[int] = None,
    output_dir: Optional[Path] = None,
    progress: ProgressFactory = progress_iter,
) -> SurfaceAIArtifacts:
    settings = settings or Settings()
    config = config or SurfaceAIConfig.from_env()
    artifacts = artifact_paths(experiment_output_dir(settings, config, output_dir))
    run_meta: Dict[str, Any] = {
        "tiles_dir": config.tiles_dir or str(Path(settings.cache_dir) / "tiles"),
        "tile_zoom": int(config.tile_zoom or settings.satellite_zoom),
        "tms_server": config.tms_server or settings.tms_server,
    }

    for _ in progress(range(1), "[1/8] Load graph/edges", total=1):
        edges_raw, graph_meta = load_edges_for_precache_area(settings)
        run_meta.update(graph_meta)

    polygon = parse_precache_polygon(settings)
    for _ in progress(range(1), "[2/8] Filter polygon", total=1):
        edges = filter_edges_to_polygon(edges_raw, polygon)
        for col in AI_OSM_FEATURES:
            if col not in edges.columns:
                edges[col] = None
        edges = limit_edges_for_experiment(
            edges,
            max_edges,
            random_state=config.random_state,
            min_known_edges=config.min_known_edges,
        )

    dataset, label_meta = build_surface_ai_dataset(
        edges,
        settings,
        config,
        polygon,
        progress=progress,
    )
    dataset["surface_ai_split"] = spatial_train_test_split(dataset, edges, config)
    summary = dataset_summary(dataset)
    write_dataset_csv(dataset, artifacts.dataset_csv)

    models, baseline_table, _ = train_evaluate_models(
        dataset,
        config,
        label_meta,
        progress=progress,
    )
    selected = models["_selected"]
    predictions = predict_all_edges(dataset, models, config, progress=progress)
    write_predictions_csv(predictions, artifacts.predictions_csv)
    write_predictions_geojson(predictions, edges, artifacts.predictions_geojson)
    write_baseline_tables(baseline_table, artifacts)

    feature_importance = feature_importance_df(models[selected["model_key"]]["model"])
    for step in progress(range(6), "[7/8] Visualizations", total=6):
        if step == 0:
            plot_surface_maps(predictions, edges, artifacts, config)
        elif step == 1:
            plot_confusion_matrix(
                selected["metrics"],
                artifacts.confusion_matrix_concrete_png,
                title="Concrete surface confusion matrix",
            )
        elif step == 2:
            plot_confusion_matrix(
                selected["group_metrics"],
                artifacts.confusion_matrix_group_png,
                title="Surface group confusion matrix",
            )
        elif step == 3:
            plot_feature_importance(feature_importance, artifacts.feature_importance_png)
        else:
            pass

    for _ in progress(range(1), "[8/8] Export reports/model", total=1):
        write_metrics_json(
            artifacts,
            selected=selected,
            models=models,
            baseline_table=baseline_table,
            summary=summary,
            label_meta=label_meta,
            config=config,
            run_meta=run_meta,
        )
        write_model_joblib(
            artifacts,
            selected=selected,
            models=models,
            feature_importance=feature_importance,
            config=config,
            label_meta=label_meta,
        )
        write_model_card(artifacts, selected=selected, config=config, dataset=dataset)
        write_report(
            artifacts,
            summary=summary,
            label_meta=label_meta,
            selected=selected,
            baseline_table=baseline_table,
            config=config,
            run_meta=run_meta,
        )

    return artifacts
