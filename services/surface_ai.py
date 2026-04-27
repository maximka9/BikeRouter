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
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
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
    OSM_HIGHWAY_FILTER,
    OSM_TAG_COLUMNS,
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
    no_progress,
    progress_iter,
)
from .area_graph_cache import graph_base_path, graph_green_path, load_graphml_path, parse_precache_polygon
from .graph import GraphBuilder


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

SPATIAL_PRIOR_FEATURES: Tuple[str, ...] = (
    "distance_to_polygon_center_m",
    "distance_to_polygon_centroid_x",
    "distance_to_polygon_centroid_y",
    "edge_centroid_lat",
    "edge_centroid_lon",
    "tile_x",
    "tile_y",
)

NEIGHBOR_FEATURES: Tuple[str, ...] = (
    "neighbor_1hop_count",
    "neighbor_2hop_count",
    "neighbor_1hop_known_surface_share",
    "neighbor_2hop_known_surface_share",
    "neighbor_1hop_paved_good_share",
    "neighbor_1hop_paved_rough_share",
    "neighbor_1hop_unpaved_soft_share",
    "neighbor_2hop_paved_good_share",
    "neighbor_2hop_paved_rough_share",
    "neighbor_2hop_unpaved_soft_share",
    "neighbor_1hop_same_highway_share",
    "neighbor_2hop_same_highway_share",
    "neighbor_1hop_mean_length_m",
    "neighbor_2hop_mean_length_m",
    "neighbor_1hop_highway_residential_share",
    "neighbor_1hop_highway_footway_share",
    "neighbor_1hop_highway_service_share",
    "neighbor_1hop_highway_path_or_track_share",
)

CONCRETE_AI_LABEL_MAP: Dict[str, str] = {
    "asphalt": "asphalt",
    "paving_stones": "paving_stones",
    "concrete": "concrete",
    "unpaved": "unpaved",
    "ground": "ground",
    "paved": "other_paved",
    "concrete:plates": "other_paved",
    "sett": "other_paved",
    "cobblestone": "other_paved",
    "bricks": "other_paved",
    "compacted": "other_unpaved",
    "fine_gravel": "other_unpaved",
    "gravel": "other_unpaved",
    "pebblestone": "other_unpaved",
    "dirt": "other_unpaved",
    "earth": "other_unpaved",
    "sand": "other_unpaved",
    "grass": "other_unpaved",
    "mud": "other_unpaved",
}

CONCRETE_COMPACT_LABELS: Tuple[str, ...] = (
    "asphalt",
    "paving_stones",
    "unpaved",
    "ground",
    "concrete",
    "other_paved",
    "other_unpaved",
)

DANGEROUS_UPGRADE_SEVERITY: Dict[str, float] = {
    "paved_rough": 1.0,
    "unpaved_hard": 2.0,
    "unpaved_soft": 3.0,
    "rough_or_unpaved": 2.5,
}

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
        "combined_rf_without_spatial_prior",
        "combined_rf_balanced_without_spatial_prior",
        "combined_rf_with_neighbors",
    )
    group_model_candidates: Tuple[str, ...] = (
        "group_direct_always_majority",
        "group_direct_highway_heuristic",
        "group_direct_osm_only_rf",
        "group_direct_satellite_only_rf",
        "group_direct_combined_rf",
        "group_direct_combined_rf_balanced",
        "group_direct_combined_rf_calibrated",
        "group_direct_combined_rf_without_spatial_prior",
        "group_direct_combined_rf_balanced_without_spatial_prior",
        "group_direct_combined_rf_neighbor_features",
    )
    concrete_target_mode: str = "compact"
    group_target_mode: str = "3class_unpaved_hard_to_soft"
    train_direct_group_model: bool = True
    selection_metric: str = "macro_f1_safe"
    dangerous_error_penalty: float = 0.50
    min_bad_surface_recall: float = 0.50
    calibration_enabled: bool = True
    calibration_method: str = "sigmoid"
    calibration_share: float = 0.10
    calibration_min_class_count: int = 20
    enable_spatial_prior: bool = True
    run_spatial_prior_ablation: bool = True
    use_neighbor_features: bool = True
    neighbor_hops: int = 2
    neighbor_use_prediction_pass2: bool = True
    graph_source_priority: Tuple[str, ...] = (
        "area_precache",
        "tiles_coverage_graph",
        "osmnx_fallback",
    )
    strict_area_precache: bool = False
    use_tile_coverage_polygon: bool = True
    tile_usage_report: bool = True
    use_green_cache_features: bool = True
    use_tile_green_mask_features: bool = True
    max_dangerous_upgrade_rate_effective: float = 0.10
    max_calibration_ece: float = 0.08
    train_area_mode: str = "tile_coverage"
    predict_area_mode: str = "precache_polygon"
    train_on_all_tile_edges: bool = True
    predict_only_inside_polygon: bool = True
    train_graph_source_priority: Tuple[str, ...] = (
        "tile_coverage_graph",
        "area_precache",
        "osmnx_fallback",
    )
    predict_graph_source_priority: Tuple[str, ...] = (
        "area_precache",
        "tile_coverage_graph",
        "osmnx_fallback",
    )
    osmnx_retain_all: bool = True
    osmnx_truncate_by_edge: bool = True
    osmnx_simplify: bool = True
    osmnx_network_type: str = "all"
    osmnx_custom_filter: str = ""
    holdout_predict_polygon_known_share: float = 0.20
    tile_edge_match_mode: str = "samples_and_buffer"
    edge_tile_buffer_m: float = 10.0
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
                    "always_majority,highway_heuristic,osm_only_rf,satellite_only_rf,combined_rf,combined_rf_balanced,combined_rf_without_spatial_prior,combined_rf_balanced_without_spatial_prior,combined_rf_with_neighbors",
                )
            ),
            group_model_candidates=_split_csv(
                _env_str(
                    "SURFACE_AI_GROUP_MODEL_CANDIDATES",
                    "group_direct_always_majority,group_direct_highway_heuristic,group_direct_osm_only_rf,group_direct_satellite_only_rf,group_direct_combined_rf,group_direct_combined_rf_balanced,group_direct_combined_rf_calibrated,group_direct_combined_rf_without_spatial_prior,group_direct_combined_rf_balanced_without_spatial_prior,group_direct_combined_rf_neighbor_features",
                )
            ),
            concrete_target_mode=_env_str("SURFACE_AI_CONCRETE_TARGET_MODE", "compact"),
            group_target_mode=_env_str("SURFACE_AI_GROUP_TARGET_MODE", "3class_unpaved_hard_to_soft"),
            train_direct_group_model=_env_bool("SURFACE_AI_TRAIN_DIRECT_GROUP_MODEL", True),
            selection_metric=_env_str("SURFACE_AI_SELECTION_METRIC", "macro_f1_safe"),
            dangerous_error_penalty=_env_float("SURFACE_AI_DANGEROUS_ERROR_PENALTY", 0.50),
            min_bad_surface_recall=_env_float("SURFACE_AI_MIN_BAD_SURFACE_RECALL", 0.50),
            calibration_enabled=_env_bool("SURFACE_AI_CALIBRATION_ENABLED", True),
            calibration_method=_env_str("SURFACE_AI_CALIBRATION_METHOD", "sigmoid"),
            calibration_share=_env_float("SURFACE_AI_CALIBRATION_SHARE", 0.10),
            calibration_min_class_count=_env_int("SURFACE_AI_CALIBRATION_MIN_CLASS_COUNT", 20),
            enable_spatial_prior=_env_bool("SURFACE_AI_ENABLE_SPATIAL_PRIOR", True),
            run_spatial_prior_ablation=_env_bool("SURFACE_AI_RUN_SPATIAL_PRIOR_ABLATION", True),
            use_neighbor_features=_env_bool("SURFACE_AI_USE_NEIGHBOR_FEATURES", True),
            neighbor_hops=_env_int("SURFACE_AI_NEIGHBOR_HOPS", 2),
            neighbor_use_prediction_pass2=_env_bool("SURFACE_AI_NEIGHBOR_USE_PREDICTION_PASS2", True),
            graph_source_priority=_split_csv(
                _env_str(
                    "SURFACE_AI_GRAPH_SOURCE_PRIORITY",
                    "area_precache,tiles_coverage_graph,osmnx_fallback",
                )
            ),
            strict_area_precache=_env_bool("SURFACE_AI_STRICT_AREA_PRECACHE", False),
            use_tile_coverage_polygon=_env_bool("SURFACE_AI_USE_TILE_COVERAGE_POLYGON", True),
            tile_usage_report=_env_bool("SURFACE_AI_TILE_USAGE_REPORT", True),
            use_green_cache_features=_env_bool("SURFACE_AI_USE_GREEN_CACHE_FEATURES", True),
            use_tile_green_mask_features=_env_bool("SURFACE_AI_USE_TILE_GREEN_MASK_FEATURES", True),
            max_dangerous_upgrade_rate_effective=_env_float("SURFACE_AI_MAX_DANGEROUS_UPGRADE_RATE_EFFECTIVE", 0.10),
            max_calibration_ece=_env_float("SURFACE_AI_MAX_CALIBRATION_ECE", 0.08),
            train_area_mode=_env_str("SURFACE_AI_TRAIN_AREA_MODE", "tile_coverage"),
            predict_area_mode=_env_str("SURFACE_AI_PREDICT_AREA_MODE", "precache_polygon"),
            train_on_all_tile_edges=_env_bool("SURFACE_AI_TRAIN_ON_ALL_TILE_EDGES", True),
            predict_only_inside_polygon=_env_bool("SURFACE_AI_PREDICT_ONLY_INSIDE_POLYGON", True),
            train_graph_source_priority=_split_csv(
                _env_str(
                    "SURFACE_AI_TRAIN_GRAPH_SOURCE_PRIORITY",
                    "tile_coverage_graph,area_precache,osmnx_fallback",
                )
            ),
            predict_graph_source_priority=_split_csv(
                _env_str(
                    "SURFACE_AI_PREDICT_GRAPH_SOURCE_PRIORITY",
                    "area_precache,tile_coverage_graph,osmnx_fallback",
                )
            ),
            osmnx_retain_all=_env_bool("SURFACE_AI_OSMNX_RETAIN_ALL", True),
            osmnx_truncate_by_edge=_env_bool("SURFACE_AI_OSMNX_TRUNCATE_BY_EDGE", True),
            osmnx_simplify=_env_bool("SURFACE_AI_OSMNX_SIMPLIFY", True),
            osmnx_network_type=_env_str("SURFACE_AI_OSMNX_NETWORK_TYPE", "all"),
            osmnx_custom_filter=_env_str("SURFACE_AI_OSMNX_CUSTOM_FILTER", ""),
            holdout_predict_polygon_known_share=_env_float("SURFACE_AI_HOLDOUT_PREDICT_POLYGON_KNOWN_SHARE", 0.20),
            tile_edge_match_mode=_env_str("SURFACE_AI_TILE_EDGE_MATCH_MODE", "samples_and_buffer"),
            edge_tile_buffer_m=_env_float("SURFACE_AI_EDGE_TILE_BUFFER_M", 10.0),
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
    confusion_matrix_concrete_compact_png: Path
    confusion_matrix_concrete_legacy_png: Path
    confusion_matrix_group_direct_png: Path
    group_direct_model_joblib: Path
    group_direct_model_card_json: Path
    group_direct_metrics_json: Path
    calibration_curve_concrete_png: Path
    calibration_curve_group_direct_png: Path
    reliability_table_concrete_csv: Path
    reliability_table_group_direct_csv: Path
    dangerous_errors_map_raw_png: Path
    dangerous_errors_map_effective_png: Path
    dangerous_errors_csv: Path
    spatial_prior_ablation_txt: Path
    tile_usage_csv: Path
    tile_usage_map_png: Path
    neighbor_feature_importance_png: Path
    train_polygon_geojson: Path
    predict_polygon_geojson: Path
    tile_coverage_polygon_geojson: Path
    train_edges_geojson: Path
    predict_edges_geojson: Path
    dataset_all_tile_edges_csv: Path
    predictions_inside_polygon_csv: Path
    predictions_inside_polygon_geojson: Path
    tile_usage_map_train_png: Path
    tile_usage_map_predict_png: Path
    train_vs_predict_report_txt: Path


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


def normalize_surface_to_concrete_ai_label(surface_norm: Any) -> Optional[str]:
    surface = normalize_surface_value(surface_norm)
    if surface is None:
        return None
    return CONCRETE_AI_LABEL_MAP.get(surface)


def normalize_group_target(group: Any, mode: str = "3class_unpaved_hard_to_soft") -> str:
    g = str(group or "unknown")
    if mode == "4class_original":
        return g if g in SURFACE_GROUPS else "unknown"
    if mode == "3class_rough_or_unpaved":
        if g in {"unpaved_hard", "unpaved_soft"}:
            return "rough_or_unpaved"
        if g in {"paved_good", "paved_rough"}:
            return g
        return "unknown"
    if g == "unpaved_hard":
        return "unpaved_soft"
    if g in {"paved_good", "paved_rough", "unpaved_soft"}:
        return g
    return "unknown"


def group_target_labels(mode: str) -> Tuple[str, ...]:
    if mode == "4class_original":
        return SURFACE_GROUPS
    if mode == "3class_rough_or_unpaved":
        return ("paved_good", "paved_rough", "rough_or_unpaved")
    return ("paved_good", "paved_rough", "unpaved_soft")


def is_dangerous_upgrade(true_group: str, pred_group: str) -> bool:
    return str(true_group) in {"paved_rough", "unpaved_hard", "unpaved_soft", "rough_or_unpaved"} and str(pred_group) == "paved_good"


def train_label_to_surface(label: Any) -> str:
    s = str(label or "unknown")
    if s in CONCRETE_SURFACE_VALUES or s in CONCRETE_COMPACT_LABELS:
        return s
    if s.startswith("rare_other_"):
        return "rare_other"
    return "unknown"


def train_label_to_group(label: Any) -> str:
    s = str(label or "unknown")
    if s in {"paved_good", "paved_rough", "unpaved_hard", "unpaved_soft", "rough_or_unpaved"}:
        return s
    if s in {"other_paved"}:
        return "paved_rough"
    if s in {"other_unpaved"}:
        return "unpaved_soft"
    if s.startswith("rare_other_"):
        return s.replace("rare_other_", "", 1) or "unknown"
    return surface_to_group(s)


def label_to_group_or_direct(label: Any) -> str:
    return train_label_to_group(label)


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
            "surface_true_concrete_legacy": surface_norm or "unknown",
            "surface_true_concrete_compact": normalize_surface_to_concrete_ai_label(surface_norm) or "unknown",
            "surface_true_group": true_group,
            "surface_group_true": true_group,
            "is_surface_known": bool(surface_norm in CONCRETE_SURFACE_VALUES),
            "geometry_wkt": row.geometry.wkt if row.geometry is not None else None,
            "inside_train_area": bool(getattr(row, "inside_train_area", False)),
            "inside_predict_area": bool(getattr(row, "inside_predict_area", False)),
            "surface_ai_edge_source": getattr(row, "surface_ai_edge_source", None),
        }
        for col in AI_OSM_FEATURES:
            item[col] = _norm_feature(getattr(row, col, None))
        item.update(_geometry_features_with_center(row, projected_geom, center))
        rows.append(item)
    return pd.DataFrame(rows)


def _surface_train_label(surface: str, frequent: set[str], *, mode: str = "legacy_full") -> Tuple[str, bool]:
    if mode == "compact":
        label = normalize_surface_to_concrete_ai_label(surface)
        if label:
            return label, False
        return "unknown", False
    if surface in frequent:
        return surface, False
    group = surface_to_group(surface)
    if group in SURFACE_GROUPS:
        return f"rare_other_{group}", True
    return "rare_other", True


def add_train_labels(
    dataset: pd.DataFrame,
    *,
    min_class_count: int,
    concrete_target_mode: str = "compact",
    group_target_mode: str = "3class_unpaved_hard_to_soft",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = dataset.copy()
    known = df[df["is_surface_known"]]
    counts = known["surface_true_concrete"].value_counts().sort_index()
    frequent = {str(k) for k, v in counts.items() if int(v) >= int(min_class_count)}
    train_labels: List[str] = []
    legacy_labels: List[str] = []
    group_direct: List[str] = []
    rare_flags: List[bool] = []
    for _, row in df.iterrows():
        surface = str(row.get("surface_true_concrete") or "unknown")
        group = str(row.get("surface_true_group") or "unknown")
        if not bool(row.get("is_surface_known")) or surface == "unknown":
            train_labels.append("unknown")
            legacy_labels.append("unknown")
            group_direct.append("unknown")
            rare_flags.append(False)
            continue
        label, is_rare = _surface_train_label(surface, frequent, mode=concrete_target_mode)
        legacy_label, legacy_is_rare = _surface_train_label(surface, frequent, mode="legacy_full")
        train_labels.append(label)
        legacy_labels.append(legacy_label)
        group_direct.append(normalize_group_target(group, group_target_mode))
        rare_flags.append(is_rare)
    df["surface_train_label"] = train_labels
    df["surface_train_label_legacy"] = legacy_labels
    df["surface_group_direct_label"] = group_direct
    df["surface_is_rare_class"] = rare_flags
    meta = {
        "surface_counts": {str(k): int(v) for k, v in counts.items()},
        "frequent_surface_classes": sorted(frequent),
        "rare_surface_classes": sorted(set(counts.index.astype(str)) - frequent),
        "min_class_count": int(min_class_count),
        "concrete_target_mode": concrete_target_mode,
        "group_target_mode": group_target_mode,
        "concrete_compact_labels": list(CONCRETE_COMPACT_LABELS),
        "group_direct_labels": list(group_target_labels(group_target_mode)),
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
    dataset, label_meta = add_train_labels(
        dataset,
        min_class_count=config.min_class_count,
        concrete_target_mode=config.concrete_target_mode,
        group_target_mode=config.group_target_mode,
    )
    return dataset, label_meta


def add_neighbor_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Add 1/2-hop graph-neighbor features without using test labels.

    For edges in the test fold, known-surface shares use only non-test OSM labels,
    preventing target leakage from adjacent test edges.
    """
    df = dataset.copy()
    edge_ids = df["edge_id"].astype(str).tolist()
    edge_set = set(edge_ids)
    node_edges: Dict[str, set[str]] = {}
    for edge_id, u, v in zip(edge_ids, df["u"].astype(str), df["v"].astype(str)):
        node_edges.setdefault(u, set()).add(edge_id)
        node_edges.setdefault(v, set()).add(edge_id)
    edge_neighbors: Dict[str, set[str]] = {}
    for edge_id, u, v in zip(edge_ids, df["u"].astype(str), df["v"].astype(str)):
        n = (node_edges.get(u, set()) | node_edges.get(v, set())) - {edge_id}
        edge_neighbors[edge_id] = n
    lookup = df.set_index("edge_id", drop=False)
    allowed_known = (
        lookup["is_surface_known"].astype(bool)
        & (lookup.get("surface_ai_split", pd.Series("", index=lookup.index)) != "test")
    )

    def summarize(edge_id: str, neigh_ids: set[str], hop: int, base_highway: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        ids = [n for n in neigh_ids if n in edge_set]
        sub = lookup.loc[ids] if ids else lookup.iloc[0:0]
        out[f"neighbor_{hop}hop_count"] = float(len(ids))
        if sub.empty:
            for key in (
                "known_surface_share",
                "paved_good_share",
                "paved_rough_share",
                "unpaved_soft_share",
                "same_highway_share",
                "mean_length_m",
            ):
                out[f"neighbor_{hop}hop_{key}"] = 0.0
            if hop == 1:
                for key in ("residential", "footway", "service", "path_or_track"):
                    out[f"neighbor_1hop_highway_{key}_share"] = 0.0
            return out
        known_mask = allowed_known.reindex(sub.index).fillna(False)
        known = sub[known_mask]
        denom = max(1, len(sub))
        out[f"neighbor_{hop}hop_known_surface_share"] = float(len(known) / denom)
        groups = known["surface_true_group"].map(lambda g: normalize_group_target(g, "3class_unpaved_hard_to_soft"))
        for group in ("paved_good", "paved_rough", "unpaved_soft"):
            out[f"neighbor_{hop}hop_{group}_share"] = float((groups == group).sum() / max(1, len(known)))
        hws = sub["highway"].astype(str)
        out[f"neighbor_{hop}hop_same_highway_share"] = float((hws == base_highway).mean()) if len(hws) else 0.0
        out[f"neighbor_{hop}hop_mean_length_m"] = float(pd.to_numeric(sub["length_m"], errors="coerce").mean()) if len(sub) else 0.0
        if hop == 1:
            out["neighbor_1hop_highway_residential_share"] = float((hws == "residential").mean()) if len(hws) else 0.0
            out["neighbor_1hop_highway_footway_share"] = float((hws == "footway").mean()) if len(hws) else 0.0
            out["neighbor_1hop_highway_service_share"] = float((hws == "service").mean()) if len(hws) else 0.0
            out["neighbor_1hop_highway_path_or_track_share"] = float(hws.isin(["path", "track"]).mean()) if len(hws) else 0.0
        return out

    rows: List[Dict[str, float]] = []
    for edge_id, base_highway in zip(edge_ids, df["highway"].astype(str)):
        hop1 = edge_neighbors.get(edge_id, set())
        hop2: set[str] = set()
        for n in hop1:
            hop2.update(edge_neighbors.get(n, set()))
        hop2.discard(edge_id)
        hop2 -= hop1
        item = {"edge_id": edge_id}
        item.update(summarize(edge_id, hop1, 1, base_highway))
        item.update(summarize(edge_id, hop2, 2, base_highway))
        rows.append(item)
    neigh = pd.DataFrame(rows)
    out = df.merge(neigh, on="edge_id", how="left", validate="one_to_one")
    for col in NEIGHBOR_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


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
    candidate_mask = dataset.get("is_train_candidate", dataset["is_surface_known"]).astype(bool)
    known = dataset[candidate_mask].copy()
    if len(known) < int(config.min_known_edges):
        raise ValueError(
            f"Not enough known train-candidate surface edges for training: found {len(known)}. "
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
    cal_n = int(round(len(unique_cells) * float(config.calibration_share))) if config.calibration_enabled else 0
    test_n = min(test_n, len(unique_cells) - 1)
    cal_n = min(max(0, cal_n), max(0, len(unique_cells) - test_n - 1))
    test_cells = set(unique_cells[:test_n])
    calibration_cells = set(unique_cells[test_n : test_n + cal_n])

    # Keep every learnable label represented in train if possible.
    labels = set(known["surface_train_label"].astype(str))
    for label in sorted(labels):
        holdout_cells = test_cells | calibration_cells
        train_labels = set(known.loc[~cells.isin(holdout_cells), "surface_train_label"].astype(str))
        if label in train_labels:
            continue
        label_cells = [
            c
            for c in cells[known["surface_train_label"].astype(str) == label].unique()
            if c in holdout_cells
        ]
        if label_cells:
            c = label_cells[0]
            if c in test_cells:
                test_cells.remove(c)
            else:
                calibration_cells.discard(c)

    split = pd.Series("unknown", index=dataset.index, dtype=object)
    split.loc[known.index[~cells.isin(test_cells | calibration_cells).to_numpy()]] = "train"
    split.loc[known.index[cells.isin(calibration_cells).to_numpy()]] = "calibration"
    split.loc[known.index[cells.isin(test_cells).to_numpy()]] = "test"
    holdout_share = max(0.0, float(config.holdout_predict_polygon_known_share))
    if holdout_share > 0 and "inside_predict_area" in dataset.columns:
        train_inside = known.index[
            (split.loc[known.index] == "train").to_numpy()
            & dataset.loc[known.index, "inside_predict_area"].astype(bool).to_numpy()
        ]
        holdout_n = int(round(len(train_inside) * holdout_share))
        if holdout_n > 0 and len(train_inside) > holdout_n:
            rng = np.random.default_rng(int(config.random_state) + 17)
            chosen = rng.choice(np.array(train_inside), size=holdout_n, replace=False)
            split.loc[chosen] = "predict_holdout"
    if (split == "train").sum() == 0 or (split == "test").sum() == 0:
        raise ValueError("Spatial split produced an empty train or test partition")
    return split


def spatial_train_cal_test_split(
    dataset: pd.DataFrame,
    edges_gdf: gpd.GeoDataFrame,
    config: SurfaceAIConfig,
) -> pd.Series:
    return spatial_train_test_split(dataset, edges_gdf, config)


def _features_for_model(dataset: pd.DataFrame, feature_set: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    geometry_cols = list(AI_GEOMETRY_FEATURES)
    if feature_set.endswith("_without_spatial_prior") or feature_set == "combined_without_spatial_prior":
        geometry_cols = [c for c in geometry_cols if c not in SPATIAL_PRIOR_FEATURES]
    if feature_set == "osm":
        cat_cols = list(AI_OSM_FEATURES)
        num_cols: List[str] = []
    elif feature_set == "satellite":
        cat_cols = []
        num_cols = list(AI_SATELLITE_FEATURES)
    elif feature_set in {"combined", "combined_without_spatial_prior"}:
        cat_cols = list(AI_OSM_FEATURES)
        num_cols = geometry_cols + list(AI_SATELLITE_FEATURES)
    elif feature_set in {"combined_with_neighbors", "combined_with_neighbors_without_spatial_prior"}:
        cat_cols = list(AI_OSM_FEATURES)
        num_cols = geometry_cols + list(AI_SATELLITE_FEATURES) + list(NEIGHBOR_FEATURES)
    elif feature_set == "osm_geometry":
        cat_cols = list(AI_OSM_FEATURES)
        num_cols = geometry_cols
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
        if candidate in {
            "combined_rf_balanced",
            "combined_balanced_random_forest",
            "combined_rf_balanced_without_spatial_prior",
            "group_direct_combined_rf_balanced",
            "group_direct_combined_rf_balanced_without_spatial_prior",
        }:
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


class GroupHighwayHeuristicClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, group_mode: str = "3class_unpaved_hard_to_soft") -> None:
        self.group_mode = group_mode
        self.classes_ = np.array(group_target_labels(group_mode))

    def fit(self, X: pd.DataFrame, y: Sequence[str]) -> "GroupHighwayHeuristicClassifier":
        self.classes_ = np.array(sorted(set(str(v) for v in y)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        labels = []
        for _, row in X.iterrows():
            surface = predict_surface_by_highway_heuristic(row.to_dict())
            labels.append(normalize_group_target(surface_to_group(surface), self.group_mode))
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
    key = candidate.replace("group_direct_", "")
    if key == "osm_only_rf":
        return "osm"
    if key == "satellite_only_rf":
        return "satellite"
    if key in {
        "combined_rf",
        "combined_rf_balanced",
        "combined_balanced_random_forest",
        "combined_hist_gradient_boosting",
        "combined_logistic_regression",
        "combined_rf_calibrated",
    }:
        return "combined"
    if key in {"combined_rf_without_spatial_prior", "combined_rf_balanced_without_spatial_prior"}:
        return "combined_without_spatial_prior"
    if key in {"combined_rf_with_neighbors", "combined_with_neighbors", "combined_rf_neighbor_features"}:
        return "combined_with_neighbors"
    if key == "highway_heuristic":
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
        "combined_rf_without_spatial_prior": ("Combined RF without spatial prior", "OSM + geometry + satellite - spatial prior"),
        "combined_rf_balanced_without_spatial_prior": ("Combined balanced RF without spatial prior", "OSM + geometry + satellite - spatial prior"),
        "combined_rf_with_neighbors": ("Combined RF + neighbor features", "OSM + geometry + satellite + neighbors"),
        "combined_rf_neighbor_features": ("Combined RF + neighbor features", "OSM + geometry + satellite + neighbors"),
        "group_direct_always_majority": ("Group direct always majority", "none"),
        "group_direct_highway_heuristic": ("Group direct highway heuristic", "highway only"),
        "group_direct_osm_only_rf": ("Group direct OSM-only RF", "OSM tags"),
        "group_direct_satellite_only_rf": ("Group direct satellite-only RF", "satellite stats"),
        "group_direct_combined_rf": ("Group direct Combined RF", "OSM + geometry + satellite"),
        "group_direct_combined_rf_balanced": ("Group direct Combined balanced RF", "OSM + geometry + satellite"),
        "group_direct_combined_rf_calibrated": ("Group direct Combined calibrated RF", "OSM + geometry + satellite"),
        "group_direct_combined_rf_without_spatial_prior": ("Group direct Combined RF without spatial prior", "OSM + geometry + satellite - spatial prior"),
        "group_direct_combined_rf_balanced_without_spatial_prior": ("Group direct Combined balanced RF without spatial prior", "OSM + geometry + satellite - spatial prior"),
        "group_direct_combined_rf_neighbor_features": ("Group direct RF + neighbor features", "OSM + geometry + satellite + neighbors"),
        "group_direct_combined_rf_with_neighbors": ("Group direct RF + neighbor features", "OSM + geometry + satellite + neighbors"),
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
    if "rough_or_unpaved" in set(ytg) | set(ypg):
        labels = ["paved_good", "paved_rough", "rough_or_unpaved"]
    report = classification_report(ytg, ypg, labels=labels, zero_division=0, output_dict=True)
    bad_mask = pd.Series(ytg).isin({"unpaved_hard", "unpaved_soft", "rough_or_unpaved"})
    pred_bad = pd.Series(ypg).isin({"unpaved_hard", "unpaved_soft", "rough_or_unpaved"})
    return {
        "target": "group",
        "accuracy": float(accuracy_score(ytg, ypg)),
        "balanced_accuracy": float(balanced_accuracy_score(ytg, ypg)),
        "macro_f1": float(f1_score(ytg, ypg, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(ytg, ypg, average="weighted", zero_division=0)),
        "recall_bad_surface": float(pred_bad[bad_mask].mean()) if bad_mask.any() else 0.0,
        "recall_unpaved_soft": float(report.get("unpaved_soft", {}).get("recall", 0.0)),
        "classification_report": report,
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(ytg, ypg, labels=labels).astype(int).tolist(),
    }


def dangerous_upgrade_metrics(
    true_groups: Sequence[str],
    raw_pred_groups: Sequence[str],
    effective_pred_groups: Optional[Sequence[str]] = None,
    lengths_m: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    true = pd.Series(list(true_groups), dtype=object).fillna("unknown").astype(str)
    raw = pd.Series(list(raw_pred_groups), dtype=object).fillna("unknown").astype(str)
    effective = (
        pd.Series(list(effective_pred_groups), dtype=object).fillna("unknown").astype(str)
        if effective_pred_groups is not None
        else raw
    )
    lengths = pd.Series(list(lengths_m), dtype=float) if lengths_m is not None else pd.Series(np.ones(len(true)), dtype=float)
    non_good = true.isin({"paved_rough", "unpaved_hard", "unpaved_soft", "rough_or_unpaved"})

    def block(pred: pd.Series, suffix: str) -> Dict[str, Any]:
        mask = pd.Series([is_dangerous_upgrade(t, p) for t, p in zip(true, pred)])
        by_class = {g: int(mask[true == g].sum()) for g in sorted(true[non_good].unique())}
        rate_by = {
            g: float(mask[true == g].mean()) if int((true == g).sum()) else 0.0
            for g in sorted(true[non_good].unique())
        }
        sev = true.map(lambda g: DANGEROUS_UPGRADE_SEVERITY.get(str(g), 0.0))
        return {
            f"dangerous_upgrade_count_{suffix}": int(mask.sum()),
            f"dangerous_upgrade_rate_{suffix}_all": float(mask.mean()) if len(mask) else 0.0,
            f"dangerous_upgrade_rate_{suffix}_among_non_good": float(mask[non_good].mean()) if non_good.any() else 0.0,
            f"dangerous_upgrade_count_by_true_class_{suffix}": by_class,
            f"dangerous_upgrade_rate_by_true_class_{suffix}": rate_by,
            f"dangerous_upgrade_length_m_{suffix}": float(lengths[mask].sum()),
            f"dangerous_upgrade_length_share_{suffix}": float(lengths[mask].sum() / max(1e-9, lengths.sum())),
            f"dangerous_upgrade_weighted_score_{suffix}": float((sev[mask]).sum()),
        }

    out = {}
    out.update(block(raw, "raw"))
    out.update(block(effective, "effective"))
    # Flat aliases used by baseline tables.
    out["dangerous_upgrade_rate_raw"] = out["dangerous_upgrade_rate_raw_among_non_good"]
    out["dangerous_upgrade_rate_effective"] = out["dangerous_upgrade_rate_effective_among_non_good"]
    out["dangerous_upgrade_count_raw"] = out["dangerous_upgrade_count_raw"]
    out["dangerous_upgrade_count_effective"] = out["dangerous_upgrade_count_effective"]
    out["dangerous_upgrade_length_m_effective"] = out["dangerous_upgrade_length_m_effective"]
    return out


def macro_f1_safe(metrics: Dict[str, Any], config: SurfaceAIConfig) -> float:
    return float(metrics.get("macro_f1", 0.0)) - float(config.dangerous_error_penalty) * float(
        metrics.get("dangerous_upgrade_rate_effective", 0.0)
    )


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


def _effective_group_for_metrics(pred_group: str, confidence: float, margin: float, config: SurfaceAIConfig) -> str:
    if margin < float(config.min_margin):
        return "unknown"
    if confidence < float(config.min_confidence):
        return "unknown"
    if pred_group == "paved_good" and confidence < float(config.paved_good_min_confidence):
        return "paved_rough"
    return pred_group


def _task_candidate_kind(candidate: str) -> str:
    key = candidate.replace("group_direct_", "")
    if key == "always_majority":
        return "always_majority"
    if key == "highway_heuristic":
        return "highway_heuristic"
    return "model"


def _fit_candidate_model(
    candidate: str,
    feature_set: str,
    target: str,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    config: SurfaceAIConfig,
    labels: Sequence[str],
    label_meta: Dict[str, Any],
) -> Tuple[Any, pd.DataFrame]:
    train_fit = pd.concat([train, calibration], axis=0) if not calibration.empty else train
    y_train = train["surface_group_direct_label" if target == "group_direct" else "surface_train_label"].astype(str)
    y_train_fit = train_fit["surface_group_direct_label" if target == "group_direct" else "surface_train_label"].astype(str)
    kind = _task_candidate_kind(candidate)
    if kind == "always_majority":
        model: Any = DummyClassifier(strategy="most_frequent")
        X_train_fit = pd.DataFrame({"constant": np.ones(len(train_fit))}, index=train_fit.index)
        model.fit(X_train_fit, y_train_fit)
        return model, X_train_fit
    if kind == "highway_heuristic":
        if target == "group_direct":
            model = GroupHighwayHeuristicClassifier(config.group_target_mode)
        else:
            model = HighwayHeuristicClassifier(labels, label_meta.get("frequent_surface_classes", []))
        X_train_fit, _, _ = _features_for_model(train_fit, "osm")
        model.fit(X_train_fit, y_train_fit)
        return model, X_train_fit

    calibrate = "calibrated" in candidate and config.calibration_enabled and config.calibration_method != "none" and not calibration.empty
    base = _build_pipeline(candidate.replace("_calibrated", ""), feature_set, config)
    if calibrate:
        X_train, _, _ = _features_for_model(train, feature_set)
        base.fit(X_train, y_train)
        X_cal, _, _ = _features_for_model(calibration, feature_set)
        y_cal = calibration["surface_group_direct_label" if target == "group_direct" else "surface_train_label"].astype(str)
        if y_cal.nunique() >= 2 and len(y_cal) >= int(config.calibration_min_class_count):
            try:
                calibrated = CalibratedClassifierCV(
                    estimator=base,
                    method=config.calibration_method,
                    cv="prefit",
                )
                calibrated.fit(X_cal, y_cal)
                return calibrated, X_train
            except Exception:
                pass
        return base, X_train
    X_train_fit, _, _ = _features_for_model(train_fit, feature_set)
    base.fit(X_train_fit, y_train_fit)
    return base, X_train_fit


def _proba_top_fields(proba: pd.DataFrame) -> Tuple[List[str], List[float], List[str], List[float], List[float]]:
    labels: List[str] = []
    conf: List[float] = []
    labels2: List[str] = []
    conf2: List[float] = []
    margins: List[float] = []
    for _, row in proba.iterrows():
        ordered = row.sort_values(ascending=False)
        top1 = str(ordered.index[0]) if len(ordered) else "unknown"
        c1 = float(ordered.iloc[0]) if len(ordered) else 0.0
        top2 = str(ordered.index[1]) if len(ordered) > 1 else "unknown"
        c2 = float(ordered.iloc[1]) if len(ordered) > 1 else 0.0
        labels.append(top1)
        conf.append(c1)
        labels2.append(top2)
        conf2.append(c2)
        margins.append(c1 - c2)
    return labels, conf, labels2, conf2, margins


def evaluate_task_model(
    *,
    candidate: str,
    model: Any,
    feature_set: str,
    target: str,
    test: pd.DataFrame,
    labels: Sequence[str],
    config: SurfaceAIConfig,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    if _task_candidate_kind(candidate) == "always_majority":
        X_test = pd.DataFrame({"constant": np.ones(len(test))}, index=test.index)
    else:
        X_test, _, _ = _features_for_model(test, feature_set if feature_set != "none" else "osm")
    proba = _predict_model_proba(model, X_test, labels)
    pred, conf, _, _, margin = _proba_top_fields(proba)
    if target == "group_direct":
        y_true = test["surface_group_direct_label"].astype(str).tolist()
        true_groups = y_true
        pred_groups = pred
        group_metrics = _group_metrics_for_predictions(y_true, pred)
        metrics = dict(group_metrics)
        metrics["target"] = "group_direct"
        report_labels = list(labels)
    else:
        y_true = test["surface_train_label"].astype(str).tolist()
        true_groups = [train_label_to_group(x) for x in y_true]
        pred_groups = [train_label_to_group(x) for x in pred]
        metrics = _metrics_for_predictions(y_true, pred, labels=labels, target="concrete_compact")
        report_labels = list(labels)
    effective_groups = [
        _effective_group_for_metrics(pg, c, m, config)
        for pg, c, m in zip(pred_groups, conf, margin)
    ]
    metrics.update(
        dangerous_upgrade_metrics(
            true_groups,
            pred_groups,
            effective_groups,
            test["length_m"].astype(float).tolist(),
        )
    )
    metrics["macro_f1_safe"] = macro_f1_safe(metrics, config)
    metrics["confusion_matrix_labels"] = report_labels if target != "group_direct" else list(labels)
    details = pd.DataFrame(
        {
            "edge_id": test["edge_id"].astype(str).values,
            "y_true": y_true,
            "y_pred": pred,
            "pred_confidence": conf,
            "pred_margin": margin,
            "true_group": true_groups,
            "pred_group": pred_groups,
            "effective_group": effective_groups,
        },
        index=test.index,
    )
    return metrics, details


def train_evaluate_task_models(
    dataset: pd.DataFrame,
    config: SurfaceAIConfig,
    label_meta: Dict[str, Any],
    *,
    target: str,
    candidates: Sequence[str],
    progress: ProgressFactory = no_progress,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    train = dataset[dataset["surface_ai_split"] == "train"].copy()
    calibration = dataset[dataset["surface_ai_split"] == "calibration"].copy()
    test = dataset[dataset["surface_ai_split"] == "test"].copy()
    target_col = "surface_group_direct_label" if target == "group_direct" else "surface_train_label"
    labels = sorted(set(train[target_col].astype(str)) | set(test[target_col].astype(str)))
    models: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    for candidate in progress(list(dict.fromkeys(candidates)), f"[5/8] Train/evaluate {target}", total=len(list(dict.fromkeys(candidates)))):
        display, features_label = _candidate_display(candidate)
        feature_set = _candidate_feature_set(candidate)
        model, _ = _fit_candidate_model(
            candidate,
            feature_set,
            target,
            train,
            calibration,
            config,
            labels,
            label_meta,
        )
        metrics, details = evaluate_task_model(
            candidate=candidate,
            model=model,
            feature_set=feature_set,
            target=target,
            test=test,
            labels=labels,
            config=config,
        )
        rows.append(
            {
                "model_key": candidate,
                "model": display,
                "target": target,
                "features": features_label,
                "accuracy": metrics.get("accuracy", 0.0),
                "balanced_accuracy": metrics.get("balanced_accuracy", 0.0),
                "macro_f1": metrics.get("macro_f1", 0.0),
                "weighted_f1": metrics.get("weighted_f1", 0.0),
                "macro_f1_safe": metrics.get("macro_f1_safe", 0.0),
                "recall_bad_surface": metrics.get("recall_bad_surface", 0.0),
                "recall_unpaved_soft": metrics.get("recall_unpaved_soft", 0.0),
                "recall_asphalt": metrics.get("recall_asphalt", np.nan),
                "recall_concrete": metrics.get("recall_concrete", np.nan),
                "recall_paving_stones": metrics.get("recall_paving_stones", np.nan),
                "dangerous_upgrade_rate_raw": metrics.get("dangerous_upgrade_rate_raw", 0.0),
                "dangerous_upgrade_rate_effective": metrics.get("dangerous_upgrade_rate_effective", 0.0),
                "dangerous_upgrade_count_raw": metrics.get("dangerous_upgrade_count_raw", 0),
                "dangerous_upgrade_count_effective": metrics.get("dangerous_upgrade_count_effective", 0),
                "dangerous_upgrade_length_m_effective": metrics.get("dangerous_upgrade_length_m_effective", 0.0),
            }
        )
        models[candidate] = {
            "model": model,
            "feature_set": feature_set,
            "metrics": metrics,
            "details": details,
            "labels": labels,
            "target": target,
        }
    table = pd.DataFrame(rows)
    return models, table


def select_model_for_target(models: Dict[str, Any], table: pd.DataFrame, config: SurfaceAIConfig, *, target: str) -> Dict[str, Any]:
    if table.empty:
        raise ValueError(f"No models evaluated for {target}")
    if target == "group_direct":
        ranked = table.sort_values(
            by=[
                "dangerous_upgrade_rate_effective",
                "macro_f1",
                "recall_bad_surface",
                "recall_unpaved_soft",
            ],
            ascending=[True, False, False, False],
        )
        selected_by = "min_dangerous_upgrade_rate_effective_then_macro_f1"
    else:
        metric = config.selection_metric
        if metric not in table.columns:
            metric = "macro_f1"
        eligible = table[table["recall_bad_surface"] >= float(config.min_bad_surface_recall)]
        ranked = eligible if not eligible.empty else table
        ranked = ranked.sort_values(
            by=[metric, "balanced_accuracy", "recall_bad_surface"],
            ascending=[False, False, False],
        )
        selected_by = metric
    row = ranked.iloc[0]
    key = str(row["model_key"])
    return {
        "model_key": key,
        "model_name": str(row["model"]),
        "selected_by": selected_by,
        "metrics": models[key]["metrics"],
        "labels": models[key]["labels"],
        "target": target,
    }


def reliability_table_from_proba(y_true: Sequence[str], proba: pd.DataFrame, *, n_bins: int = 10) -> Tuple[pd.DataFrame, Dict[str, float]]:
    pred = proba.idxmax(axis=1).astype(str)
    conf = proba.max(axis=1).astype(float)
    correct = (pd.Series(list(y_true), index=proba.index).astype(str) == pred).astype(float)
    rows = []
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        mask = (conf >= lo) & (conf <= hi if i == n_bins - 1 else conf < hi)
        if mask.any():
            avg_conf = float(conf[mask].mean())
            acc = float(correct[mask].mean())
            gap = abs(avg_conf - acc)
            share = float(mask.mean())
            ece += share * gap
            mce = max(mce, gap)
            count = int(mask.sum())
        else:
            avg_conf = acc = gap = share = 0.0
            count = 0
        rows.append(
            {
                "bin_low": lo,
                "bin_high": hi,
                "count": count,
                "share": share,
                "avg_confidence": avg_conf,
                "accuracy": acc,
                "gap": gap,
            }
        )
    brier = float(np.mean((conf - correct) ** 2)) if len(conf) else 0.0
    metrics = {
        "ece_top_label": float(ece),
        "mce_top_label": float(mce),
        "brier_top_label": brier,
        "mean_confidence_correct": float(conf[correct == 1].mean()) if (correct == 1).any() else 0.0,
        "mean_confidence_incorrect": float(conf[correct == 0].mean()) if (correct == 0).any() else 0.0,
    }
    return pd.DataFrame(rows), metrics


def plot_calibration_curve(reliability: pd.DataFrame, path: Path, *, title: str) -> None:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="#666666", linestyle="--", linewidth=1)
    if not reliability.empty:
        ax.plot(reliability["avg_confidence"], reliability["accuracy"], marker="o", color="#1f78b4")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def calibration_artifacts_for_selected(
    dataset: pd.DataFrame,
    selected: Dict[str, Any],
    models: Dict[str, Any],
    artifacts: SurfaceAIArtifacts,
    *,
    target: str,
) -> Dict[str, Any]:
    key = selected["model_key"]
    entry = models[key]
    test = dataset[dataset["surface_ai_split"] == "test"].copy()
    target_col = "surface_group_direct_label" if target == "group_direct" else "surface_train_label"
    if _task_candidate_kind(key) == "always_majority":
        X = pd.DataFrame({"constant": np.ones(len(test))}, index=test.index)
    else:
        X, _, _ = _features_for_model(test, entry["feature_set"] if entry["feature_set"] != "none" else "osm")
    proba_model = entry.get("calibrated_model") or entry["model"]
    proba = _predict_model_proba(proba_model, X, entry["labels"])
    table, metrics = reliability_table_from_proba(test[target_col].astype(str).tolist(), proba)
    if target == "group_direct":
        table.to_csv(artifacts.reliability_table_group_direct_csv, index=False, encoding="utf-8")
        plot_calibration_curve(table, artifacts.calibration_curve_group_direct_png, title="Group direct calibration")
    else:
        table.to_csv(artifacts.reliability_table_concrete_csv, index=False, encoding="utf-8")
        plot_calibration_curve(table, artifacts.calibration_curve_concrete_png, title="Concrete compact calibration")
    return metrics


def evaluate_selected_on_split(
    dataset: pd.DataFrame,
    selected: Dict[str, Any],
    models: Dict[str, Any],
    config: SurfaceAIConfig,
    *,
    target: str,
    split_name: str,
) -> Dict[str, Any]:
    subset = dataset[dataset["surface_ai_split"] == split_name].copy()
    if subset.empty:
        return {"split": split_name, "target": target, "edge_count": 0}
    key = selected["model_key"]
    entry = models[key]
    model = entry.get("calibrated_model") or entry["model"]
    metrics, _ = evaluate_task_model(
        candidate=key,
        model=model,
        feature_set=entry["feature_set"],
        target=target,
        test=subset,
        labels=entry["labels"],
        config=config,
    )
    metrics["split"] = split_name
    metrics["edge_count"] = int(len(subset))
    return metrics


def apply_posthoc_calibration_to_selected(
    dataset: pd.DataFrame,
    selected: Dict[str, Any],
    models: Dict[str, Any],
    config: SurfaceAIConfig,
    *,
    target: str,
) -> bool:
    if not config.calibration_enabled or config.calibration_method == "none":
        selected["calibrated"] = False
        return False
    key = selected["model_key"]
    if _task_candidate_kind(key) != "model":
        selected["calibrated"] = False
        return False
    entry = models[key]
    if isinstance(entry.get("model"), CalibratedClassifierCV) or entry.get("calibrated_model") is not None:
        selected["calibrated"] = True
        selected["calibration_method"] = config.calibration_method
        return True

    train = dataset[dataset["surface_ai_split"] == "train"].copy()
    calibration = dataset[dataset["surface_ai_split"] == "calibration"].copy()
    if calibration.empty:
        selected["calibrated"] = False
        return False
    target_col = "surface_group_direct_label" if target == "group_direct" else "surface_train_label"
    y_train = train[target_col].astype(str)
    y_cal = calibration[target_col].astype(str)
    labels = set(entry.get("labels") or [])
    if y_cal.nunique() < 2 or len(y_cal) < int(config.calibration_min_class_count):
        selected["calibrated"] = False
        return False
    if labels and not labels.issubset(set(y_cal.unique())):
        selected["calibrated"] = False
        return False

    feature_set = entry["feature_set"]
    base_key = key.replace("_calibrated", "")
    base = _build_pipeline(base_key, feature_set, config)
    X_train, _, _ = _features_for_model(train, feature_set)
    X_cal, _, _ = _features_for_model(calibration, feature_set)
    try:
        base.fit(X_train, y_train)
        calibrated = CalibratedClassifierCV(
            estimator=base,
            method=config.calibration_method,
            cv="prefit",
        )
        calibrated.fit(X_cal, y_cal)
    except Exception:
        selected["calibrated"] = False
        return False
    entry["calibrated_model"] = calibrated
    entry["calibrated"] = True
    selected["calibrated"] = True
    selected["calibration_method"] = config.calibration_method
    return True


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


def _predict_entry_proba_for_all(
    dataset: pd.DataFrame,
    entry: Dict[str, Any],
    model_key: str,
    *,
    calibrated: bool = False,
) -> pd.DataFrame:
    if _task_candidate_kind(model_key) == "always_majority":
        X = pd.DataFrame({"constant": np.ones(len(dataset))}, index=dataset.index)
    else:
        X, _, _ = _features_for_model(dataset, entry["feature_set"] if entry["feature_set"] != "none" else "osm")
    model = entry.get("calibrated_model") if calibrated and entry.get("calibrated_model") is not None else entry["model"]
    return _predict_model_proba(model, X, entry["labels"])


def confidence_bucket(conf: float, config: SurfaceAIConfig) -> str:
    if conf >= float(config.conf_high):
        return "high"
    if conf >= float(config.conf_medium):
        return "medium"
    return "low"


def predict_all_edges(
    dataset: pd.DataFrame,
    concrete_models: Dict[str, Any],
    group_models: Dict[str, Any],
    config: SurfaceAIConfig,
    *,
    progress: ProgressFactory = no_progress,
) -> pd.DataFrame:
    concrete_selected = concrete_models["_selected"]
    concrete_key = concrete_selected["model_key"]
    concrete_entry = concrete_models[concrete_key]
    group_selected = group_models["_selected"]
    group_key = group_selected["model_key"]
    group_entry = group_models[group_key]

    for _ in progress(range(1), "[6/8] Predict surfaces/groups", total=1):
        concrete_proba_raw = _predict_entry_proba_for_all(dataset, concrete_entry, concrete_key)
        concrete_proba = _predict_entry_proba_for_all(dataset, concrete_entry, concrete_key, calibrated=True)
        group_proba_raw = _predict_entry_proba_for_all(dataset, group_entry, group_key)
        group_proba = _predict_entry_proba_for_all(dataset, group_entry, group_key, calibrated=True)

    concrete_pred_raw, concrete_conf_raw, _, _, concrete_margin_raw = _proba_top_fields(concrete_proba_raw)
    concrete_pred, concrete_conf, concrete_top2, concrete_top2_conf, concrete_margin = _proba_top_fields(concrete_proba)
    group_pred_raw, group_conf_raw, _, _, group_margin_raw = _proba_top_fields(group_proba_raw)
    group_pred, group_conf, group_top2, group_top2_conf, group_margin = _proba_top_fields(group_proba)

    out = dataset.copy()
    out["surface_pred_label"] = concrete_pred
    out["surface_pred_label_raw"] = concrete_pred_raw
    out["surface_pred_concrete_compact"] = [train_label_to_surface(x) for x in concrete_pred]
    out["surface_pred_concrete"] = out["surface_pred_concrete_compact"]
    out["surface_pred"] = out["surface_pred_concrete_compact"]
    out["surface_pred_group_derived_from_concrete"] = [train_label_to_group(x) for x in concrete_pred]
    out["surface_pred_group_direct"] = group_pred
    out["surface_pred_group_direct_raw"] = group_pred_raw
    out["surface_pred_group"] = out["surface_pred_group_direct"]
    out["surface_group_pred"] = out["surface_pred_group_direct"]
    out["surface_pred_confidence_raw"] = concrete_conf_raw
    out["surface_pred_confidence_calibrated"] = concrete_conf
    out["surface_pred_confidence"] = group_conf
    out["surface_pred_top2"] = [train_label_to_surface(x) for x in concrete_top2]
    out["surface_pred_top2_confidence"] = concrete_top2_conf
    out["surface_pred_margin_raw"] = concrete_margin_raw
    out["surface_pred_margin_calibrated"] = concrete_margin
    out["surface_pred_margin"] = group_margin
    out["surface_pred_group_direct_confidence_raw"] = group_conf_raw
    out["surface_pred_group_direct_confidence_calibrated"] = group_conf
    out["surface_pred_group_direct_margin_raw"] = group_margin_raw
    out["surface_pred_group_direct_margin_calibrated"] = group_margin

    for surface in sorted(set(concrete_proba.columns) | set(REQUIRED_PROBA_SURFACES) | set(CONCRETE_COMPACT_LABELS)):
        col = proba_column_for_surface(train_label_to_surface(surface))
        if col not in out.columns:
            out[col] = 0.0
    for label in concrete_proba.columns:
        surface = train_label_to_surface(str(label))
        col = proba_column_for_surface(surface)
        out[col] = out[col].to_numpy(dtype=float) + concrete_proba[label].to_numpy(dtype=float)
    if "proba_unknown" not in out.columns:
        out["proba_unknown"] = 0.0

    raw_eff: List[str] = []
    cal_eff: List[str] = []
    sources: List[str] = []
    concrete_effective: List[str] = []
    for idx, row in out.iterrows():
        known = bool(row.get("is_surface_known"))
        true_group = normalize_group_target(row.get("surface_true_group"), config.group_target_mode)
        pred_group = str(row.get("surface_pred_group_direct") or "unknown")
        conf = float(row.get("surface_pred_group_direct_confidence_calibrated") or 0.0)
        margin = float(row.get("surface_pred_group_direct_margin_calibrated") or 0.0)
        has_features = _has_features(row, group_entry["feature_set"])
        if known:
            eff_group = true_group
            source = "osm"
        elif not has_features:
            eff_group = "unknown"
            source = "default_no_features"
        elif margin < float(config.min_margin):
            eff_group = "unknown"
            source = "ml_ambiguous"
        elif conf < float(config.min_confidence):
            eff_group = "unknown"
            source = "default_low_confidence"
        else:
            eff_group = _effective_group_for_metrics(pred_group, conf, margin, config)
            source = "ml"
        raw_pred_group = str(row.get("surface_pred_group_direct_raw") or "unknown")
        raw_eff.append(raw_pred_group if not known else true_group)
        cal_eff.append(eff_group)
        sources.append(source)
        concrete_effective.append(str(row.get("surface_true_concrete") if known else row.get("surface_pred_concrete_compact")))
    out["surface_source"] = sources
    out["surface_effective_for_routing"] = concrete_effective
    out["surface_effective_group_for_routing_raw"] = raw_eff
    out["surface_effective_group_for_routing_calibrated"] = cal_eff
    out["surface_effective_group_for_routing"] = cal_eff
    true_direct = out["surface_group_direct_label"].astype(str)
    out["dangerous_upgrade_raw"] = [
        is_dangerous_upgrade(t, p) for t, p in zip(true_direct, out["surface_effective_group_for_routing_raw"].astype(str))
    ]
    out["dangerous_upgrade_effective"] = [
        is_dangerous_upgrade(t, p) for t, p in zip(true_direct, out["surface_effective_group_for_routing_calibrated"].astype(str))
    ]
    out["dangerous_upgrade_severity"] = [
        DANGEROUS_UPGRADE_SEVERITY.get(str(t), 0.0) if bool(flag) else 0.0
        for t, flag in zip(true_direct, out["dangerous_upgrade_effective"])
    ]
    out["confidence_bucket_raw"] = [confidence_bucket(float(x), config) for x in out["surface_pred_group_direct_confidence_raw"]]
    out["confidence_bucket_calibrated"] = [confidence_bucket(float(x), config) for x in out["surface_pred_group_direct_confidence_calibrated"]]
    out["tile_id_samples"] = ""
    out["tile_green_mask_available"] = False
    out["edge_green_cache_available"] = False
    return out


def dataset_summary(dataset: pd.DataFrame) -> Dict[str, Any]:
    total = int(len(dataset))
    known = int(dataset["is_surface_known"].sum()) if total else 0
    return {
        "total_edges": total,
        "known_surface_edges": known,
        "unknown_surface_edges": total - known,
        "unknown_share": float((total - known) / total) if total else 0.0,
        "train_edges_total": int(dataset.get("inside_train_area", pd.Series(False, index=dataset.index)).astype(bool).sum()),
        "train_edges_with_tile_features": int(
            (
                dataset.get("inside_train_area", pd.Series(False, index=dataset.index)).astype(bool)
                & dataset.get("has_tile_features", pd.Series(False, index=dataset.index)).astype(bool)
            ).sum()
        ),
        "train_known_surface_edges": int(dataset.get("is_train_candidate", pd.Series(False, index=dataset.index)).astype(bool).sum()),
        "train_unknown_surface_edges": int(
            (
                dataset.get("inside_train_area", pd.Series(False, index=dataset.index)).astype(bool)
                & ~dataset["is_surface_known"].astype(bool)
            ).sum()
        ),
        "predict_edges_inside_polygon": int(dataset.get("inside_predict_area", pd.Series(False, index=dataset.index)).astype(bool).sum()),
        "predict_unknown_surface_edges_inside_polygon": int(
            dataset.get("is_prediction_candidate", pd.Series(False, index=dataset.index)).astype(bool).sum()
        ),
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
        confusion_matrix_concrete_compact_png=output_dir / "surface_ai_confusion_matrix_concrete_compact.png",
        confusion_matrix_concrete_legacy_png=output_dir / "surface_ai_confusion_matrix_concrete_legacy.png",
        confusion_matrix_group_direct_png=output_dir / "surface_ai_confusion_matrix_group_direct.png",
        group_direct_model_joblib=output_dir / "surface_ai_group_direct_model.joblib",
        group_direct_model_card_json=output_dir / "surface_ai_group_direct_model_card.json",
        group_direct_metrics_json=output_dir / "surface_ai_group_direct_metrics.json",
        calibration_curve_concrete_png=output_dir / "surface_ai_calibration_curve_concrete.png",
        calibration_curve_group_direct_png=output_dir / "surface_ai_calibration_curve_group_direct.png",
        reliability_table_concrete_csv=output_dir / "surface_ai_reliability_table_concrete.csv",
        reliability_table_group_direct_csv=output_dir / "surface_ai_reliability_table_group_direct.csv",
        dangerous_errors_map_raw_png=output_dir / "surface_ai_dangerous_errors_map_raw.png",
        dangerous_errors_map_effective_png=output_dir / "surface_ai_dangerous_errors_map_effective.png",
        dangerous_errors_csv=output_dir / "surface_ai_dangerous_errors.csv",
        spatial_prior_ablation_txt=output_dir / "surface_ai_spatial_prior_ablation.txt",
        tile_usage_csv=output_dir / "surface_ai_tile_usage.csv",
        tile_usage_map_png=output_dir / "surface_ai_tile_usage_map.png",
        neighbor_feature_importance_png=output_dir / "surface_ai_neighbor_feature_importance.png",
        train_polygon_geojson=output_dir / "surface_ai_train_polygon.geojson",
        predict_polygon_geojson=output_dir / "surface_ai_predict_polygon.geojson",
        tile_coverage_polygon_geojson=output_dir / "surface_ai_tile_coverage_polygon.geojson",
        train_edges_geojson=output_dir / "surface_ai_train_edges.geojson",
        predict_edges_geojson=output_dir / "surface_ai_predict_edges.geojson",
        dataset_all_tile_edges_csv=output_dir / "surface_ai_dataset_all_tile_edges.csv",
        predictions_inside_polygon_csv=output_dir / "surface_ai_predictions_inside_polygon.csv",
        predictions_inside_polygon_geojson=output_dir / "surface_ai_predictions_inside_polygon.geojson",
        tile_usage_map_train_png=output_dir / "surface_ai_tile_usage_map_train.png",
        tile_usage_map_predict_png=output_dir / "surface_ai_tile_usage_map_predict.png",
        train_vs_predict_report_txt=output_dir / "surface_ai_train_vs_predict_report.txt",
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
        "surface_true",
        "inside_train_area",
        "inside_predict_area",
        "has_tile_features",
        "is_train_candidate",
        "is_prediction_candidate",
        "is_surface_known",
        "surface_train_label",
        "surface_pred_concrete",
        "surface_pred_concrete_compact",
        "surface_pred_group_derived_from_concrete",
        "surface_pred_group_direct",
        "surface_pred_group_direct_confidence_raw",
        "surface_pred_group_direct_confidence_calibrated",
        "surface_pred_group",
        "surface_group_pred",
        "surface_pred_confidence_raw",
        "surface_pred_confidence_calibrated",
        "surface_pred_confidence",
        "surface_pred_top2",
        "surface_pred_top2_confidence",
        "surface_pred_margin_raw",
        "surface_pred_margin_calibrated",
        "surface_pred_margin",
        "surface_source",
    ]
    required.extend(proba_column_for_surface(s) for s in REQUIRED_PROBA_SURFACES)
    required.extend(
        [
            "surface_is_rare_class",
            "surface_effective_for_routing",
            "surface_effective_group_for_routing_raw",
            "surface_effective_group_for_routing_calibrated",
            "surface_effective_group_for_routing",
            "dangerous_upgrade_raw",
            "dangerous_upgrade_effective",
            "dangerous_upgrade_severity",
            "confidence_bucket_raw",
            "confidence_bucket_calibrated",
            "neighbor_1hop_count",
            "neighbor_2hop_count",
            "neighbor_1hop_known_surface_share",
            "neighbor_1hop_paved_good_share",
            "neighbor_1hop_paved_rough_share",
            "neighbor_1hop_unpaved_soft_share",
            "neighbor_2hop_paved_good_share",
            "neighbor_2hop_paved_rough_share",
            "neighbor_2hop_unpaved_soft_share",
            "tile_id_samples",
            "tile_samples_count",
            "tile_missing_share",
            "tile_green_mask_available",
            "edge_green_cache_available",
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
    prediction_ids = set(predictions["edge_id"].astype(str)) if "edge_id" in predictions.columns else set()
    if predictions.empty:
        base = base.iloc[0:0]
    elif prediction_ids:
        base = base[base["edge_id"].astype(str).isin(prediction_ids)]
    gdf = base.merge(predictions.drop(columns=["geometry"], errors="ignore"), on="edge_id", how="left")
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GeoJSON")


def write_edges_geojson(edges_gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = edges_gdf.copy()
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    else:
        gdf = gdf.to_crs("EPSG:4326")
    gdf.to_file(path, driver="GeoJSON")


def write_baseline_tables(baseline_table: pd.DataFrame, artifacts: SurfaceAIArtifacts) -> None:
    baseline_table.to_csv(artifacts.baseline_csv, index=False, encoding="utf-8")
    with pd.ExcelWriter(artifacts.baseline_xlsx, engine="openpyxl") as writer:
        baseline_table.to_excel(writer, sheet_name="all", index=False)
        baseline_table[baseline_table["target"].astype(str).str.contains("concrete")].to_excel(
            writer, sheet_name="concrete", index=False
        )
        baseline_table[baseline_table["target"].astype(str).str.contains("group")].to_excel(
            writer, sheet_name="groups", index=False
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
        "dangerous_upgrade_rate_effective",
        "dangerous_upgrade_count_effective",
    ]
    df = baseline_table[[c for c in cols if c in baseline_table.columns]].copy()
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
        & (predictions["surface_group_direct_label"].astype(str) != predictions["surface_pred_group_direct"].astype(str))
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


def plot_dangerous_errors_map(
    predictions: pd.DataFrame,
    edges_gdf: gpd.GeoDataFrame,
    path: Path,
    *,
    mode: str,
) -> None:
    col = "dangerous_upgrade_raw" if mode == "raw" else "dangerous_upgrade_effective"
    mask = (predictions["surface_ai_split"] == "test") & predictions[col].astype(bool)
    gdf = edges_gdf[["edge_id", "geometry"]].merge(
        predictions[["edge_id", "surface_group_direct_label", "length_m", col]],
        on="edge_id",
        how="inner",
    )
    ids = set(predictions.loc[mask, "edge_id"].astype(str))
    gdf = gdf[gdf["edge_id"].astype(str).isin(ids)]
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_axis_off()
    colors = {"paved_rough": "#fdae61", "unpaved_soft": "#d73027", "unpaved_hard": "#7f0000", "rough_or_unpaved": "#d73027"}
    if gdf.empty:
        ax.text(0.5, 0.5, f"No dangerous upgrades ({mode})", ha="center", va="center", transform=ax.transAxes)
    else:
        for group, color in colors.items():
            subset = gdf[gdf["surface_group_direct_label"] == group]
            if subset.empty:
                continue
            subset.plot(ax=ax, color=color, linewidth=2.0, alpha=0.9, label=f"{group} -> paved_good")
        total_len = float(pd.to_numeric(gdf["length_m"], errors="coerce").sum())
        ax.text(
            0.01,
            0.99,
            f"Dangerous upgrades after safety policy: {len(gdf)} edges / {total_len:.0f} m",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def write_dangerous_errors_csv(predictions: pd.DataFrame, path: Path) -> None:
    mask = (
        (predictions["surface_ai_split"] == "test")
        & (predictions["dangerous_upgrade_raw"].astype(bool) | predictions["dangerous_upgrade_effective"].astype(bool))
    )
    cols = [
        "edge_id",
        "u",
        "v",
        "key",
        "length_m",
        "highway",
        "surface_true_group",
        "surface_group_direct_label",
        "surface_pred_group_direct",
        "surface_effective_group_for_routing_calibrated",
        "dangerous_upgrade_raw",
        "dangerous_upgrade_effective",
        "dangerous_upgrade_severity",
        "surface_pred_group_direct_confidence_calibrated",
        "geometry_wkt",
    ]
    predictions.loc[mask, [c for c in cols if c in predictions.columns]].to_csv(path, index=False, encoding="utf-8")


def _tile_to_lat_lon(x: int, y: int, z: int) -> Tuple[float, float]:
    n = 2**z
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    return math.degrees(lat_rad), lon


def _tile_bounds_geom(x: int, y: int, z: int) -> Any:
    lat1, lon1 = _tile_to_lat_lon(x, y, z)
    lat2, lon2 = _tile_to_lat_lon(x + 1, y + 1, z)
    return box(lon1, lat2, lon2, lat1)


def count_cached_tile_images(tiles_dir: Path | str) -> int:
    tiles_dir = Path(tiles_dir)
    if not tiles_dir.exists():
        return 0
    suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    return sum(1 for p in tiles_dir.iterdir() if p.is_file() and p.suffix.lower() in suffixes)


def scan_cached_tiles(
    tiles_dir: Path | str,
    tms_server: str,
    z: int,
    precache_polygon: Any = None,
) -> gpd.GeoDataFrame:
    tiles_dir = Path(tiles_dir)
    rows = []
    pat = re.compile(rf"^{re.escape(tms_server)}_{int(z)}_(\d+)_(\d+)\.(jpg|jpeg|png|webp)$", re.IGNORECASE)
    for path in tiles_dir.glob(f"{tms_server}_{int(z)}_*_*.*"):
        m = pat.match(path.name)
        if not m:
            continue
        x = int(m.group(1))
        y = int(m.group(2))
        geom = _tile_bounds_geom(x, y, int(z))
        rows.append(
            {
                "tile_id": f"{tms_server}_{int(z)}_{x}_{y}",
                "server": tms_server,
                "z": int(z),
                "x": x,
                "y": y,
                "path": str(path),
                "geometry": geom,
                "footprint_wgs84": geom.wkt,
            }
        )
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    if not gdf.empty:
        gdf["footprint_area_m2"] = gdf.to_crs(_utm_crs_for(gdf)).geometry.area.astype(float).values
    else:
        gdf["footprint_area_m2"] = []
    if precache_polygon is not None:
        gdf["inside_precache_polygon"] = gdf.geometry.intersects(precache_polygon) if not gdf.empty else []
    elif "inside_precache_polygon" not in gdf.columns:
        gdf["inside_precache_polygon"] = False if not gdf.empty else []
    return gdf


def _polygon_area_km2(polygon: Any) -> float:
    if polygon is None or polygon.is_empty:
        return 0.0
    gdf = gpd.GeoDataFrame({"name": ["area"]}, geometry=[polygon], crs="EPSG:4326")
    return float(gdf.to_crs(_utm_crs_for(gdf)).geometry.area.iloc[0] / 1_000_000.0)


def _write_polygon_geojson(path: Path, polygon: Any, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame({"name": [name]}, geometry=[polygon], crs="EPSG:4326")
    gdf.to_file(path, driver="GeoJSON")


def tile_coverage_polygon_from_tiles(tiles_gdf: gpd.GeoDataFrame, fallback_polygon: Any) -> Any:
    if tiles_gdf.empty:
        return fallback_polygon
    return unary_union(list(tiles_gdf.geometry.values))


def _area_polygon_from_mode(mode: str, tile_coverage_polygon: Any, precache_polygon: Any) -> Any:
    mode = str(mode or "precache_polygon")
    if mode == "tile_coverage":
        return tile_coverage_polygon
    if mode == "tile_coverage_intersect_precache_polygon":
        return tile_coverage_polygon.intersection(precache_polygon)
    if mode == "tile_coverage_union_precache_polygon":
        return tile_coverage_polygon.union(precache_polygon)
    return precache_polygon


def _load_edges_from_osmnx_polygon(
    polygon: Any,
    settings: Settings,
    config: SurfaceAIConfig,
    *,
    source: str,
) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    ox.settings.cache_folder = settings.osmnx_cache_dir
    ox.settings.use_cache = True
    for tag in OSM_TAG_COLUMNS + ("smoothness",):
        if tag not in ox.settings.useful_tags_way:
            ox.settings.useful_tags_way += [tag]
    kwargs: Dict[str, Any] = {
        "simplify": bool(config.osmnx_simplify),
        "retain_all": bool(config.osmnx_retain_all),
        "truncate_by_edge": bool(config.osmnx_truncate_by_edge),
    }
    custom_filter = str(config.osmnx_custom_filter or "").strip()
    if custom_filter:
        kwargs["custom_filter"] = custom_filter
    else:
        kwargs["network_type"] = str(config.osmnx_network_type or "all")
    graph = ox.graph_from_polygon(polygon, **kwargs)
    if not isinstance(graph, nx.MultiDiGraph):
        graph = nx.MultiDiGraph(graph)
    return GraphBuilder.to_geodataframe(graph), {
        "graph_source": source,
        "graph_path": None,
        "osmnx_retain_all": bool(config.osmnx_retain_all),
        "osmnx_truncate_by_edge": bool(config.osmnx_truncate_by_edge),
        "osmnx_simplify": bool(config.osmnx_simplify),
        "osmnx_network_type": str(config.osmnx_network_type or "all"),
        "osmnx_custom_filter": custom_filter,
    }


def load_edges_for_surface_ai_polygon(
    settings: Settings,
    config: SurfaceAIConfig,
    polygon: Any,
    *,
    priority: Sequence[str],
    scope: str,
) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    """Load edges for one Surface AI area using a scoped source priority."""
    meta: Dict[str, Any] = {
        "precache_polygon_bounds_lonlat": tuple(float(x) for x in polygon.bounds),
        "graph_source": None,
        "graph_path": None,
        "graph_source_priority": list(priority),
        "graph_fallback_reasons": [],
        "graph_scope": scope,
    }
    priorities = tuple(priority or ("area_precache", "tiles_coverage_graph", "osmnx_fallback"))

    for source in priorities:
        if source == "tile_coverage_graph":
            try:
                edges, osm_meta = _load_edges_from_osmnx_polygon(
                    polygon,
                    settings,
                    config,
                    source="tile_coverage_graph" if scope == "train" else "tile_coverage_graph_predict",
                )
                meta.update(osm_meta)
                return edges, meta
            except Exception as exc:
                meta["graph_fallback_reasons"].append(f"tile_coverage_graph_failed:{exc}")
                continue
        if source == "area_precache":
            loaded = False
            for label, path in (("graph_base", graph_base_path(settings)), ("graph_green", graph_green_path(settings))):
                if not path.exists():
                    meta["graph_fallback_reasons"].append(f"area_precache_{label}_missing")
                    continue
                graph = load_graphml_path(path)
                if graph is None:
                    meta["graph_fallback_reasons"].append(f"area_precache_{label}_load_failed")
                    continue
                meta["graph_source"] = "area_precache_graphml"
                meta["graph_path"] = str(path)
                return GraphBuilder.to_geodataframe(graph), meta
            if config.strict_area_precache and not loaded:
                raise FileNotFoundError(
                    "Surface AI strict area precache is enabled, but graph_base.graphml/graph_green.graphml were not usable"
                )
            continue
        if source == "osmnx_fallback":
            try:
                edges, osm_meta = _load_edges_from_osmnx_polygon(
                    polygon,
                    settings,
                    config,
                    source="osmnx_overpass_fallback",
                )
                meta.update(osm_meta)
                return edges, meta
            except Exception as exc:
                meta["graph_fallback_reasons"].append(f"osmnx_fallback_failed:{exc}")
                continue
        meta["graph_fallback_reasons"].append(f"unknown_graph_source:{source}")

    raise ValueError(f"No usable Surface AI graph source for {scope}: {priorities}; reasons={meta['graph_fallback_reasons']}")


def load_edges_for_surface_ai_area(settings: Settings, config: SurfaceAIConfig) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    if not settings.has_precache_area_polygon:
        raise ValueError("PRECACHE_AREA_POLYGON_WKT is empty or not a polygon")
    return load_edges_for_surface_ai_polygon(
        settings,
        config,
        parse_precache_polygon(settings),
        priority=config.graph_source_priority,
        scope="legacy",
    )


def build_train_predict_polygons(
    settings: Settings,
    config: SurfaceAIConfig,
    tiles_gdf: gpd.GeoDataFrame,
) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    precache_polygon = parse_precache_polygon(settings)
    tile_coverage_polygon = tile_coverage_polygon_from_tiles(tiles_gdf, precache_polygon)
    train_polygon = _area_polygon_from_mode(config.train_area_mode, tile_coverage_polygon, precache_polygon)
    predict_polygon = _area_polygon_from_mode(config.predict_area_mode, tile_coverage_polygon, precache_polygon)
    meta = {
        "train_area_mode": config.train_area_mode,
        "predict_area_mode": config.predict_area_mode,
        "tile_coverage_area_km2": _polygon_area_km2(tile_coverage_polygon),
        "precache_polygon_area_km2": _polygon_area_km2(precache_polygon),
        "train_polygon_area_km2": _polygon_area_km2(train_polygon),
        "predict_polygon_area_km2": _polygon_area_km2(predict_polygon),
    }
    return train_polygon, predict_polygon, tile_coverage_polygon, meta


def _edge_inside_polygon_flags(edges_gdf: gpd.GeoDataFrame, polygon: Any) -> pd.Series:
    if edges_gdf.empty:
        return pd.Series([], dtype=bool)
    gdf = edges_gdf.to_crs("EPSG:4326") if edges_gdf.crs else edges_gdf.set_crs("EPSG:4326", allow_override=True)
    return gdf.geometry.intersects(polygon)


def combine_train_predict_edges(
    train_edges: gpd.GeoDataFrame,
    predict_edges: gpd.GeoDataFrame,
    train_polygon: Any,
    predict_polygon: Any,
) -> gpd.GeoDataFrame:
    train = train_edges.copy()
    predict = predict_edges.copy()
    train["inside_train_area"] = _edge_inside_polygon_flags(train, train_polygon).values
    train["inside_predict_area"] = _edge_inside_polygon_flags(train, predict_polygon).values
    predict["inside_train_area"] = _edge_inside_polygon_flags(predict, train_polygon).values
    predict["inside_predict_area"] = _edge_inside_polygon_flags(predict, predict_polygon).values
    train["surface_ai_edge_source"] = "train_graph"
    predict["surface_ai_edge_source"] = "predict_graph"
    combined = pd.concat([train, predict], axis=0, ignore_index=True)
    if "edge_id" in combined.columns:
        combined["_prefer_predict"] = (combined["surface_ai_edge_source"] == "predict_graph").astype(int)
        combined = combined.sort_values("_prefer_predict").drop_duplicates("edge_id", keep="last").drop(columns="_prefer_predict")
    return gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")


def mark_dataset_area_flags(dataset: pd.DataFrame, config: SurfaceAIConfig) -> pd.DataFrame:
    df = dataset.copy()
    for col in ("inside_train_area", "inside_predict_area"):
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].fillna(False).astype(bool)
    if config.use_satellite_features:
        sampled = pd.to_numeric(df.get("sampled_pixel_count", 0), errors="coerce").fillna(0)
        missing = pd.to_numeric(df.get("tile_missing_share", 1.0), errors="coerce").fillna(1.0)
        df["has_tile_features"] = (sampled > 0) & (missing < 1.0)
    else:
        df["has_tile_features"] = True
    df["is_train_candidate"] = (
        df["inside_train_area"].astype(bool)
        & df["has_tile_features"].astype(bool)
        & df["is_surface_known"].astype(bool)
    )
    predict_area_mask = df["inside_predict_area"].astype(bool) if config.predict_only_inside_polygon else pd.Series(True, index=df.index)
    df["is_prediction_candidate"] = (
        predict_area_mask
        & df["has_tile_features"].astype(bool)
        & ~df["is_surface_known"].astype(bool)
    )
    df["surface_true"] = df["surface_true_concrete"].where(df["is_surface_known"].astype(bool), None)
    return df


def write_tile_usage_report(
    tiles_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    polygon: Any,
    artifacts: SurfaceAIArtifacts,
    *,
    predict_edges_gdf: Optional[gpd.GeoDataFrame] = None,
    train_polygon: Any = None,
    predict_polygon: Any = None,
    config: Optional[SurfaceAIConfig] = None,
    total_cached_tiles: Optional[int] = None,
) -> Dict[str, Any]:
    train_edges_gdf = edges_gdf
    predict_edges_gdf = predict_edges_gdf if predict_edges_gdf is not None else edges_gdf
    train_polygon = train_polygon if train_polygon is not None else polygon
    predict_polygon = predict_polygon if predict_polygon is not None else polygon
    total_cached_tiles = int(total_cached_tiles if total_cached_tiles is not None else len(tiles_gdf))

    if tiles_gdf.empty:
        pd.DataFrame(
            columns=[
                "tile_id",
                "x",
                "y",
                "path",
                "inside_train_area",
                "inside_predict_area",
                "intersects_train_edge",
                "intersects_predict_edge",
                "sampled_by_train_edge",
                "sampled_by_predict_edge",
                "sampled_pixel_count",
                "sampled_edge_count",
                "unused_reason",
            ]
        ).to_csv(artifacts.tile_usage_csv, index=False, encoding="utf-8")
        plot_tile_usage_map(tiles_gdf, artifacts.tile_usage_map_png)
        plot_tile_usage_map(tiles_gdf, artifacts.tile_usage_map_train_png, zone="train")
        plot_tile_usage_map(tiles_gdf, artifacts.tile_usage_map_predict_png, zone="predict")
        return {
            "tiles_total_in_cache": total_cached_tiles,
            "tiles_matching_server_zoom": 0,
            "tiles_in_train_area": 0,
            "tiles_intersect_train_edges": 0,
            "tiles_sampled_by_train_edges": 0,
            "tiles_in_predict_area": 0,
            "tiles_intersect_predict_edges": 0,
            "tiles_sampled_by_predict_edges": 0,
            "train_tile_usage_share": 0.0,
            "predict_tile_usage_share": 0.0,
            "edges_with_tile_samples": 0,
            "edges_without_tile_samples": int(len(train_edges_gdf)),
        }

    def usage_union(edges: gpd.GeoDataFrame) -> Any:
        if edges is None or edges.empty:
            return None
        gdf = edges.copy()
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        else:
            gdf = gdf.to_crs("EPSG:4326")
        mode = str((config.tile_edge_match_mode if config else "samples_and_buffer") or "samples_and_buffer")
        if "buffer" in mode:
            projected = gdf.to_crs(_utm_crs_for(gdf))
            buffered = projected.geometry.buffer(float(config.edge_tile_buffer_m if config else 10.0))
            return gpd.GeoSeries(buffered, crs=projected.crs).to_crs("EPSG:4326").unary_union
        return unary_union(list(gdf.geometry.values))

    tiles = tiles_gdf.copy()
    tiles["inside_train_area"] = tiles.geometry.intersects(train_polygon)
    tiles["inside_predict_area"] = tiles.geometry.intersects(predict_polygon)
    train_union = usage_union(train_edges_gdf)
    predict_union = usage_union(predict_edges_gdf)
    tiles["intersects_train_edge"] = tiles.geometry.intersects(train_union) if train_union is not None else False
    tiles["intersects_predict_edge"] = tiles.geometry.intersects(predict_union) if predict_union is not None else False
    tiles["sampled_by_train_edge"] = tiles["intersects_train_edge"]
    tiles["sampled_by_predict_edge"] = tiles["intersects_predict_edge"]
    tiles["sampled_edge_count"] = tiles["sampled_by_train_edge"].astype(int) + tiles["sampled_by_predict_edge"].astype(int)
    tiles["sampled_pixel_count"] = 0
    tiles["used_in_features"] = tiles["sampled_by_train_edge"] | tiles["sampled_by_predict_edge"]

    def unused_reason(row: pd.Series) -> str:
        if bool(row["sampled_by_train_edge"]) and bool(row["sampled_by_predict_edge"]):
            return "used_for_train_and_predict"
        if bool(row["sampled_by_train_edge"]):
            return "used_for_train_only"
        if bool(row["sampled_by_predict_edge"]):
            return "used_for_predict_only"
        if not bool(row["inside_train_area"]) and not bool(row["inside_predict_area"]):
            return "outside_train_and_predict_area"
        return "no_road_edge_intersection"

    tiles["unused_reason"] = tiles.apply(unused_reason, axis=1)
    tiles.drop(columns="geometry").to_csv(artifacts.tile_usage_csv, index=False, encoding="utf-8")
    plot_tile_usage_map(tiles, artifacts.tile_usage_map_png)
    plot_tile_usage_map(tiles, artifacts.tile_usage_map_train_png, zone="train")
    plot_tile_usage_map(tiles, artifacts.tile_usage_map_predict_png, zone="predict")
    train_inside = int(tiles["inside_train_area"].sum())
    train_used = int((tiles["inside_train_area"] & tiles["sampled_by_train_edge"]).sum())
    predict_inside = int(tiles["inside_predict_area"].sum())
    predict_used = int((tiles["inside_predict_area"] & tiles["sampled_by_predict_edge"]).sum())
    out = {
        "tiles_total_in_cache": total_cached_tiles,
        "tiles_matching_server_zoom": int(len(tiles)),
        "tiles_in_train_area": train_inside,
        "tiles_intersect_train_edges": int(tiles["intersects_train_edge"].sum()),
        "tiles_sampled_by_train_edges": train_used,
        "tiles_in_predict_area": predict_inside,
        "tiles_intersect_predict_edges": int(tiles["intersects_predict_edge"].sum()),
        "tiles_sampled_by_predict_edges": predict_used,
        "train_tile_usage_share": float(train_used / train_inside) if train_inside else 0.0,
        "predict_tile_usage_share": float(predict_used / predict_inside) if predict_inside else 0.0,
        "edges_with_tile_samples": int((train_edges_gdf["edge_id"].notna()).sum()),
        "edges_without_tile_samples": 0,
    }
    out.update(
        {
            "tiles_inside_precache_polygon": predict_inside,
            "tiles_intersect_graph_edges": int(tiles["intersects_train_edge"].sum()),
            "tiles_sampled_by_edges": train_used,
            "tile_usage_share_inside_polygon": float(predict_used / predict_inside) if predict_inside else 0.0,
            "unused_tiles_inside_polygon": int(max(0, predict_inside - predict_used)),
        }
    )
    return out


def write_spatial_prior_ablation(artifacts: SurfaceAIArtifacts, baseline_table: pd.DataFrame) -> Dict[str, Any]:
    concrete = baseline_table[baseline_table["target"].astype(str).str.contains("concrete")].copy()
    keys = [
        "combined_rf",
        "combined_rf_without_spatial_prior",
        "combined_rf_balanced",
        "combined_rf_balanced_without_spatial_prior",
    ]
    rows = concrete[concrete["model_key"].isin(keys)][
        [
            c
            for c in [
                "model",
                "macro_f1",
                "balanced_accuracy",
                "recall_bad_surface",
                "dangerous_upgrade_rate_effective",
            ]
            if c in concrete.columns
        ]
    ]
    lines = [
        "Surface AI spatial prior ablation",
        "",
        rows.to_string(index=False) if not rows.empty else "No ablation rows were evaluated.",
        "",
        "Interpretation:",
        "If quality drops only slightly, the spatial prior helps but is not critical.",
        "If quality drops strongly, the model is locally tied to this study area.",
    ]
    artifacts.spatial_prior_ablation_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out: Dict[str, Any] = {"enabled": True}
    by_key = concrete.set_index("model_key") if not concrete.empty else pd.DataFrame()
    if not by_key.empty and "combined_rf" in by_key.index and "combined_rf_without_spatial_prior" in by_key.index:
        out["macro_f1_delta_without_spatial_prior"] = float(by_key.loc["combined_rf_without_spatial_prior", "macro_f1"] - by_key.loc["combined_rf", "macro_f1"])
        out["dangerous_upgrade_delta_without_spatial_prior"] = float(
            by_key.loc["combined_rf_without_spatial_prior", "dangerous_upgrade_rate_effective"]
            - by_key.loc["combined_rf", "dangerous_upgrade_rate_effective"]
        )
    return out


def plot_tile_usage_map(tiles_gdf: gpd.GeoDataFrame, path: Path, *, zone: str = "any") -> None:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_axis_off()
    if tiles_gdf.empty:
        ax.text(0.5, 0.5, "No cached tiles", ha="center", va="center", transform=ax.transAxes)
    else:
        tiles = tiles_gdf.copy()
        if "inside_precache_polygon" not in tiles.columns:
            tiles["inside_precache_polygon"] = False
        if "used_in_features" not in tiles.columns:
            tiles["used_in_features"] = False
        if zone == "train":
            inside_col = "inside_train_area"
            used_col = "sampled_by_train_edge"
        elif zone == "predict":
            inside_col = "inside_predict_area"
            used_col = "sampled_by_predict_edge"
        else:
            inside_col = "inside_precache_polygon" if "inside_precache_polygon" in tiles.columns else "inside_predict_area"
            used_col = "used_in_features"
        if inside_col not in tiles.columns:
            tiles[inside_col] = False
        if used_col not in tiles.columns:
            tiles[used_col] = False
        outside = tiles[~tiles[inside_col].astype(bool)]
        unused = tiles[tiles[inside_col].astype(bool) & ~tiles[used_col].astype(bool)]
        used = tiles[tiles[used_col].astype(bool)]
        if not outside.empty:
            outside.boundary.plot(ax=ax, color="#bdbdbd", linewidth=0.4, label="outside")
        if not unused.empty:
            unused.boundary.plot(ax=ax, color="#ffd92f", linewidth=0.6, label="inside unused")
        if not used.empty:
            used.boundary.plot(ax=ax, color="#1a9850", linewidth=0.8, label="used")
        ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
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
    group_selected: Optional[Dict[str, Any]] = None,
    config: SurfaceAIConfig,
    dataset: pd.DataFrame,
    dangerous_metrics: Optional[Dict[str, Any]] = None,
    tile_coverage: Optional[Dict[str, Any]] = None,
    calibration_metrics: Optional[Dict[str, Any]] = None,
    spatial_ablation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    known = dataset[dataset["is_surface_known"]]
    train = dataset[dataset["surface_ai_split"] == "train"]
    test = dataset[dataset["surface_ai_split"] == "test"]
    metrics = selected["metrics"]
    feature_set = models_feature_set_from_key(selected["model_key"])
    card = {
        "model_name": f"surface_ai_{selected['model_key']}",
        "target": "surface_concrete",
        "concrete_target_mode": config.concrete_target_mode,
        "group_target_mode": config.group_target_mode,
        "direct_group_model": bool(config.train_direct_group_model),
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
        "calibrated": bool(selected.get("calibrated", False)),
        "calibration_method": selected.get("calibration_method", config.calibration_method),
        "ece_top_label_before": None,
        "ece_top_label_after": (calibration_metrics or {}).get("ece_top_label"),
        "spatial_prior_ablation": spatial_ablation or {"enabled": bool(config.run_spatial_prior_ablation)},
        "dangerous_error_metrics": dangerous_metrics or {},
        "tile_coverage": tile_coverage or {},
        "neighbor_features": {"enabled": bool(config.use_neighbor_features), "hops": int(config.neighbor_hops)},
        "limitations": [
            "class imbalance",
            "rare surfaces are hard to learn",
            "satellite tiles contain shadows/cars/tree crowns",
        ],
        "safe_to_use_for_routing": bool(
            (dangerous_metrics or {}).get("dangerous_upgrade_rate_effective", 1.0)
            <= float(config.max_dangerous_upgrade_rate_effective)
            and selected["metrics"].get("recall_bad_surface", 0.0) >= float(config.min_bad_surface_recall)
            and (calibration_metrics or {}).get("ece_top_label", 1.0) <= float(config.max_calibration_ece)
            and bool(selected.get("calibrated", False))
        ),
    }
    with open(artifacts.model_card_json, "w", encoding="utf-8") as f:
        json.dump(_json_safe(card), f, ensure_ascii=False, indent=2)
    return card


def write_group_direct_model_card(
    artifacts: SurfaceAIArtifacts,
    *,
    selected: Dict[str, Any],
    config: SurfaceAIConfig,
    dataset: pd.DataFrame,
    calibration_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    train = dataset[dataset["surface_ai_split"] == "train"]
    test = dataset[dataset["surface_ai_split"] == "test"]
    metrics = selected["metrics"]
    card = {
        "model_name": f"surface_ai_{selected['model_key']}",
        "target": "group_direct",
        "group_target_mode": config.group_target_mode,
        "classes": sorted(dataset.loc[dataset["is_surface_known"], "surface_group_direct_label"].astype(str).unique().tolist()),
        "features": models_feature_set_from_key(selected["model_key"]),
        "training_edges": int(len(train)),
        "test_edges": int(len(test)),
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "dangerous_upgrade_rate_effective": metrics.get("dangerous_upgrade_rate_effective"),
        "calibrated": bool(selected.get("calibrated", False)),
        "calibration_method": selected.get("calibration_method", config.calibration_method),
        "ece_top_label_after": (calibration_metrics or {}).get("ece_top_label"),
        "safe_to_use_for_routing": False,
    }
    with open(artifacts.group_direct_model_card_json, "w", encoding="utf-8") as f:
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
    group_selected: Optional[Dict[str, Any]] = None,
    baseline_table: pd.DataFrame,
    config: SurfaceAIConfig,
    run_meta: Dict[str, Any],
    tile_coverage: Optional[Dict[str, Any]] = None,
    area_meta: Optional[Dict[str, Any]] = None,
    holdout_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    top = baseline_table[baseline_table["target"].astype(str).str.contains("concrete")].sort_values(
        ["macro_f1", "balanced_accuracy"],
        ascending=False,
    )
    lines = [
        "Surface AI experiment report",
        "",
        f"Output dir: {artifacts.output_dir}",
        f"Train area mode: {config.train_area_mode}",
        f"Predict area mode: {config.predict_area_mode}",
        f"Train graph source: {run_meta.get('train_graph_source')}",
        f"Train graph path: {run_meta.get('train_graph_path')}",
        f"Predict graph source: {run_meta.get('predict_graph_source')}",
        f"Predict graph path: {run_meta.get('predict_graph_path')}",
        f"Graph source: {run_meta.get('graph_source')}",
        f"Graph path: {run_meta.get('graph_path')}",
        f"Tiles dir: {run_meta.get('tiles_dir')}",
        f"Tile zoom: {run_meta.get('tile_zoom')}",
        f"Total cached tiles: {(tile_coverage or {}).get('tiles_total_in_cache')}",
        f"Matching server/zoom tiles: {(tile_coverage or {}).get('tiles_matching_server_zoom')}",
        f"Train area tiles: {(tile_coverage or {}).get('tiles_in_train_area')}",
        f"Predict area tiles: {(tile_coverage or {}).get('tiles_in_predict_area')}",
        f"Area metrics: {area_meta or {}}",
        "",
        "Dataset:",
        f"  Total edges: {summary.get('total_edges')}",
        f"  Known surface edges: {summary.get('known_surface_edges')}",
        f"  Unknown surface edges: {summary.get('unknown_surface_edges')}",
        f"  Unknown share: {summary.get('unknown_share'):.3f}",
        f"  Train edges total: {summary.get('train_edges_total')}",
        f"  Train edges with tile features: {summary.get('train_edges_with_tile_features')}",
        f"  Train known surface edges: {summary.get('train_known_surface_edges')}",
        f"  Train unknown surface edges: {summary.get('train_unknown_surface_edges')}",
        f"  Predict edges inside polygon: {summary.get('predict_edges_inside_polygon')}",
        f"  Predict unknown surface edges inside polygon: {summary.get('predict_unknown_surface_edges_inside_polygon')}",
        f"  Predictions produced inside polygon: {summary.get('predictions_produced_inside_polygon')}",
        f"  Frequent concrete classes: {label_meta.get('frequent_surface_classes')}",
        f"  Rare concrete classes: {label_meta.get('rare_surface_classes')}",
        "",
        "Model selection:",
        f"  Selected model: {selected.get('model_name')} ({selected.get('model_key')})",
        f"  Main criterion: {selected.get('selected_by')}",
        f"  Safety recall threshold: {selected.get('safety_recall_threshold')}",
        f"  Safety threshold satisfied by selected pool: {selected.get('safety_recall_threshold_satisfied')}",
        f"  Selected metrics: {selected.get('metrics')}",
        f"  Selected group direct model: {(group_selected or {}).get('model_name')} ({(group_selected or {}).get('model_key')})",
        f"  Group direct metrics: {(group_selected or {}).get('metrics')}",
        f"  Global test metrics: {selected.get('metrics')}",
        f"  Predict polygon holdout metrics: {holdout_metrics or {}}",
        f"  Concrete target mode: {config.concrete_target_mode}",
        f"  Group target mode: {config.group_target_mode}",
        f"  Tile coverage: {tile_coverage or {}}",
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
            "Tile-driven graph warning:",
            "  If tile coverage area is large, but train graph has few edges, possible causes are cached tiles without roads, wrong server/zoom, graph-source fallback failure, restrictive filters, fragmented tile coverage, or partial Overpass response.",
            "",
            "Artifacts:",
        ]
    )
    for field in artifacts.__dataclass_fields__:
        if field != "output_dir":
            lines.append(f"  {field}: {getattr(artifacts, field)}")
    text = "\n".join(lines) + "\n"
    artifacts.report_txt.write_text(text, encoding="utf-8")
    artifacts.train_vs_predict_report_txt.write_text(text, encoding="utf-8")


def write_metrics_json(
    artifacts: SurfaceAIArtifacts,
    *,
    selected: Dict[str, Any],
    group_selected: Optional[Dict[str, Any]] = None,
    models: Dict[str, Any],
    group_models: Optional[Dict[str, Any]] = None,
    baseline_table: pd.DataFrame,
    summary: Dict[str, Any],
    label_meta: Dict[str, Any],
    config: SurfaceAIConfig,
    run_meta: Dict[str, Any],
    tile_coverage: Optional[Dict[str, Any]] = None,
    calibration_metrics: Optional[Dict[str, Any]] = None,
    area_meta: Optional[Dict[str, Any]] = None,
    holdout_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "selected_model": selected,
        "selected_group_direct_model": group_selected,
        "model_metrics": {
            key: {
                "metrics": value.get("metrics") or value.get("concrete_metrics"),
                "group_metrics": value.get("group_metrics"),
                "feature_set": value.get("feature_set"),
            }
            for key, value in models.items()
            if not key.startswith("_")
        },
        "group_direct_model_metrics": {
            key: {
                "metrics": value.get("metrics"),
                "feature_set": value.get("feature_set"),
            }
            for key, value in (group_models or {}).items()
            if not key.startswith("_")
        },
        "baseline_table": baseline_table.to_dict(orient="records"),
        "dataset_summary": summary,
        "label_meta": label_meta,
        "config": asdict(config),
        "run_meta": run_meta,
        "tile_coverage": tile_coverage or {},
        "tile_usage": tile_coverage or {},
        "area_meta": area_meta or {},
        "global_tile_coverage_test_metrics": selected.get("metrics"),
        "predict_polygon_holdout_metrics": holdout_metrics or {},
        "calibration_metrics": calibration_metrics or {},
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
        "calibrated_model": models[key].get("calibrated_model"),
        "model_key": key,
        "feature_set": models[key]["feature_set"],
        "selected": selected,
        "surface_values": CONCRETE_SURFACE_VALUES,
        "surface_group_map": SURFACE_GROUP_MAP,
        "label_meta": label_meta,
        "feature_importance": feature_importance,
        "config": asdict(config),
        "uses_calibrated_confidence": bool(selected.get("calibrated", False)),
    }
    joblib.dump(bundle, artifacts.model_joblib)


def write_group_direct_joblib(
    artifacts: SurfaceAIArtifacts,
    *,
    selected: Dict[str, Any],
    models: Dict[str, Any],
    config: SurfaceAIConfig,
) -> None:
    key = selected["model_key"]
    bundle = {
        "model": models[key]["model"],
        "calibrated_model": models[key].get("calibrated_model"),
        "model_key": key,
        "feature_set": models[key]["feature_set"],
        "selected": selected,
        "labels": models[key]["labels"],
        "target": "group_direct",
        "config": asdict(config),
        "uses_calibrated_confidence": bool(selected.get("calibrated", False)),
    }
    joblib.dump(bundle, artifacts.group_direct_model_joblib)


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

    precache_polygon = parse_precache_polygon(settings)
    tiles_dir = Path(run_meta["tiles_dir"])
    total_cached_tiles = count_cached_tile_images(tiles_dir)
    for _ in progress(range(1), "[1/10] Scan cached tiles", total=1):
        tiles_gdf = scan_cached_tiles(
            tiles_dir,
            str(run_meta["tms_server"]),
            int(run_meta["tile_zoom"]),
            precache_polygon=precache_polygon,
        )
        train_polygon, predict_polygon, tile_coverage_polygon, area_meta = build_train_predict_polygons(
            settings,
            config,
            tiles_gdf,
        )
        area_meta.update(
            {
                "tiles_total_in_cache": total_cached_tiles,
                "tiles_matching_server_zoom": int(len(tiles_gdf)),
            }
        )
        _write_polygon_geojson(artifacts.train_polygon_geojson, train_polygon, "train_polygon")
        _write_polygon_geojson(artifacts.predict_polygon_geojson, predict_polygon, "predict_polygon")
        _write_polygon_geojson(artifacts.tile_coverage_polygon_geojson, tile_coverage_polygon, "tile_coverage_polygon")

    for _ in progress(range(1), "[2/10] Load train graph", total=1):
        train_raw, train_meta = load_edges_for_surface_ai_polygon(
            settings,
            config,
            train_polygon,
            priority=config.train_graph_source_priority,
            scope="train",
        )
        run_meta["train_graph_source"] = train_meta.get("graph_source")
        run_meta["train_graph_path"] = train_meta.get("graph_path")
        run_meta["train_graph_fallback_reasons"] = train_meta.get("graph_fallback_reasons", [])
        run_meta["graph_source"] = run_meta["train_graph_source"]
        run_meta["graph_path"] = run_meta["train_graph_path"]

    for _ in progress(range(1), "[3/10] Filter train graph", total=1):
        train_edges = filter_edges_to_polygon(train_raw, train_polygon)
        for col in AI_OSM_FEATURES:
            if col not in train_edges.columns:
                train_edges[col] = None

    for _ in progress(range(1), "[4/10] Load predict graph", total=1):
        predict_raw, predict_meta = load_edges_for_surface_ai_polygon(
            settings,
            config,
            predict_polygon,
            priority=config.predict_graph_source_priority,
            scope="predict",
        )
        run_meta["predict_graph_source"] = predict_meta.get("graph_source")
        run_meta["predict_graph_path"] = predict_meta.get("graph_path")
        run_meta["predict_graph_fallback_reasons"] = predict_meta.get("graph_fallback_reasons", [])

    for _ in progress(range(1), "[5/10] Filter predict graph", total=1):
        predict_edges = filter_edges_to_polygon(predict_raw, predict_polygon)
        for col in AI_OSM_FEATURES:
            if col not in predict_edges.columns:
                predict_edges[col] = None

    write_edges_geojson(train_edges, artifacts.train_edges_geojson)
    write_edges_geojson(predict_edges, artifacts.predict_edges_geojson)
    edges = combine_train_predict_edges(train_edges, predict_edges, train_polygon, predict_polygon)
    if max_edges is not None:
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
        train_polygon,
        progress=progress,
    )
    dataset = mark_dataset_area_flags(dataset, config)
    dataset["surface_ai_split"] = spatial_train_cal_test_split(dataset, edges, config)
    if config.use_neighbor_features:
        dataset = add_neighbor_features(dataset)
    summary = dataset_summary(dataset)
    write_dataset_csv(dataset, artifacts.dataset_csv)
    write_dataset_csv(dataset, artifacts.dataset_all_tile_edges_csv)

    concrete_models, concrete_table = train_evaluate_task_models(
        dataset,
        config,
        label_meta,
        target="concrete_compact",
        candidates=config.model_candidates,
        progress=progress,
    )
    concrete_selected = select_model_for_target(concrete_models, concrete_table, config, target="concrete_compact")
    concrete_models["_selected"] = concrete_selected
    group_models, group_table = train_evaluate_task_models(
        dataset,
        config,
        label_meta,
        target="group_direct",
        candidates=config.group_model_candidates,
        progress=progress,
    )
    group_selected = select_model_for_target(group_models, group_table, config, target="group_direct")
    group_models["_selected"] = group_selected
    apply_posthoc_calibration_to_selected(
        dataset,
        concrete_selected,
        concrete_models,
        config,
        target="concrete_compact",
    )
    apply_posthoc_calibration_to_selected(
        dataset,
        group_selected,
        group_models,
        config,
        target="group_direct",
    )
    holdout_metrics = {
        "concrete_compact": evaluate_selected_on_split(
            dataset,
            concrete_selected,
            concrete_models,
            config,
            target="concrete_compact",
            split_name="predict_holdout",
        ),
        "group_direct": evaluate_selected_on_split(
            dataset,
            group_selected,
            group_models,
            config,
            target="group_direct",
            split_name="predict_holdout",
        ),
    }
    baseline_table = pd.concat([concrete_table, group_table], axis=0, ignore_index=True)

    predictions = predict_all_edges(dataset, concrete_models, group_models, config, progress=progress)
    predictions_inside = predictions[predictions.get("is_prediction_candidate", pd.Series(False, index=predictions.index)).astype(bool)].copy()
    summary["predictions_produced_inside_polygon"] = int(len(predictions_inside))
    write_predictions_csv(predictions, artifacts.predictions_csv)
    write_predictions_geojson(predictions, edges, artifacts.predictions_geojson)
    write_predictions_csv(predictions_inside, artifacts.predictions_inside_polygon_csv)
    write_predictions_geojson(predictions_inside, predict_edges, artifacts.predictions_inside_polygon_geojson)
    write_baseline_tables(baseline_table, artifacts)

    tile_coverage = (
        write_tile_usage_report(
            tiles_gdf,
            train_edges,
            train_polygon,
            artifacts,
            predict_edges_gdf=predict_edges,
            train_polygon=train_polygon,
            predict_polygon=predict_polygon,
            config=config,
            total_cached_tiles=total_cached_tiles,
        )
        if config.tile_usage_report
        else {}
    )
    dangerous_overall = dangerous_upgrade_metrics(
        predictions.loc[predictions["surface_ai_split"] == "test", "surface_group_direct_label"].astype(str).tolist(),
        predictions.loc[predictions["surface_ai_split"] == "test", "surface_effective_group_for_routing_raw"].astype(str).tolist(),
        predictions.loc[predictions["surface_ai_split"] == "test", "surface_effective_group_for_routing_calibrated"].astype(str).tolist(),
        predictions.loc[predictions["surface_ai_split"] == "test", "length_m"].astype(float).tolist(),
    )
    dangerous_flat = {
        "dangerous_upgrade_rate_raw": dangerous_overall.get("dangerous_upgrade_rate_raw", 0.0),
        "dangerous_upgrade_rate_effective": dangerous_overall.get("dangerous_upgrade_rate_effective", 0.0),
        "dangerous_upgrade_length_m_effective": dangerous_overall.get("dangerous_upgrade_length_m_effective", 0.0),
    }
    spatial_ablation = write_spatial_prior_ablation(artifacts, baseline_table)
    concrete_calibration = calibration_artifacts_for_selected(
        dataset,
        concrete_selected,
        concrete_models,
        artifacts,
        target="concrete_compact",
    )
    group_calibration = calibration_artifacts_for_selected(
        dataset,
        group_selected,
        group_models,
        artifacts,
        target="group_direct",
    )
    feature_importance = feature_importance_df(concrete_models[concrete_selected["model_key"]]["model"])
    neighbor_importance = feature_importance[
        feature_importance["feature"].astype(str).str.contains("neighbor_", regex=False)
    ].copy()
    if neighbor_importance.empty:
        neighbor_importance = feature_importance.head(30).copy()
    predictions_predict_area = predictions[predictions.get("inside_predict_area", pd.Series(False, index=predictions.index)).astype(bool)].copy()
    for step in progress(range(6), "[9/10] Visualizations", total=6):
        if step == 0:
            plot_surface_maps(predictions_predict_area, predict_edges, artifacts, config)
        elif step == 1:
            plot_confusion_matrix(
                concrete_selected["metrics"],
                artifacts.confusion_matrix_concrete_png,
                title="Concrete surface confusion matrix",
            )
            plot_confusion_matrix(
                concrete_selected["metrics"],
                artifacts.confusion_matrix_concrete_compact_png,
                title="Concrete compact confusion matrix",
            )
            plot_confusion_matrix(
                concrete_selected["metrics"],
                artifacts.confusion_matrix_concrete_legacy_png,
                title="Concrete legacy comparison matrix",
            )
        elif step == 2:
            plot_confusion_matrix(
                group_selected["metrics"],
                artifacts.confusion_matrix_group_png,
                title="Surface group direct confusion matrix",
            )
            plot_confusion_matrix(
                group_selected["metrics"],
                artifacts.confusion_matrix_group_direct_png,
                title="Surface group direct confusion matrix",
            )
        elif step == 3:
            plot_feature_importance(feature_importance, artifacts.feature_importance_png)
            plot_feature_importance(neighbor_importance, artifacts.neighbor_feature_importance_png)
        elif step == 4:
            plot_dangerous_errors_map(predictions, edges, artifacts.dangerous_errors_map_raw_png, mode="raw")
            plot_dangerous_errors_map(predictions, edges, artifacts.dangerous_errors_map_effective_png, mode="effective")
            write_dangerous_errors_csv(predictions, artifacts.dangerous_errors_csv)
        else:
            pass

    for _ in progress(range(1), "[10/10] Export reports/model", total=1):
        write_metrics_json(
            artifacts,
            selected=concrete_selected,
            group_selected=group_selected,
            models=concrete_models,
            group_models=group_models,
            baseline_table=baseline_table,
            summary=summary,
            label_meta=label_meta,
            config=config,
            run_meta=run_meta,
            tile_coverage=tile_coverage,
            calibration_metrics={"concrete": concrete_calibration, "group_direct": group_calibration},
            area_meta=area_meta,
            holdout_metrics=holdout_metrics,
        )
        with open(artifacts.group_direct_metrics_json, "w", encoding="utf-8") as f:
            json.dump(_json_safe(group_selected), f, ensure_ascii=False, indent=2)
        write_model_joblib(
            artifacts,
            selected=concrete_selected,
            models=concrete_models,
            feature_importance=feature_importance,
            config=config,
            label_meta=label_meta,
        )
        write_group_direct_joblib(
            artifacts,
            selected=group_selected,
            models=group_models,
            config=config,
        )
        write_model_card(
            artifacts,
            selected=concrete_selected,
            group_selected=group_selected,
            config=config,
            dataset=dataset,
            dangerous_metrics=dangerous_flat,
            tile_coverage=tile_coverage,
            calibration_metrics=concrete_calibration,
            spatial_ablation=spatial_ablation,
        )
        write_group_direct_model_card(
            artifacts,
            selected=group_selected,
            config=config,
            dataset=dataset,
            calibration_metrics=group_calibration,
        )
        write_report(
            artifacts,
            summary=summary,
            label_meta=label_meta,
            selected=concrete_selected,
            group_selected=group_selected,
            baseline_table=baseline_table,
            config=config,
            run_meta=run_meta,
            tile_coverage=tile_coverage,
            area_meta=area_meta,
            holdout_metrics=holdout_metrics,
        )

    return artifacts
