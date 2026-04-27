"""Быстрые unit-проверки Surface AI / ML (без обучения, без тайлов/OSM)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, box

from bike_router.services.surface_prediction_store import (
    REQUIRED_RUNTIME_COLUMNS,
    write_runtime_predictions_csv,
)
from bike_router.services.surface_resolve import ml_group_to_profile_surface


def test_normalize_surface_value() -> None:
    from bike_router.services.surface_ai import normalize_surface_value

    assert normalize_surface_value("asphalt;concrete") == "asphalt"
    assert normalize_surface_value("yes") == "paved"
    assert normalize_surface_value("no") is None


def test_ml_group_maps_to_profile_surface() -> None:
    assert ml_group_to_profile_surface("paved_good") == "asphalt"
    assert ml_group_to_profile_surface("paved_rough") == "compacted"
    assert ml_group_to_profile_surface("unpaved_soft") == "unpaved"


def test_train_label_to_group_concrete_mapping() -> None:
    from bike_router.services.surface_ai import train_label_to_group

    assert train_label_to_group("asphalt") == "paved_good"
    assert train_label_to_group("ground") == "unpaved_soft"


def test_dangerous_upgrade_detection() -> None:
    from bike_router.services.surface_ai import is_dangerous_upgrade

    assert is_dangerous_upgrade("unpaved_soft", "paved_good") is True
    assert is_dangerous_upgrade("paved_good", "unpaved_soft") is False


def test_safety_policy_safe_vs_effective() -> None:
    from bike_router.services.surface_ai import SurfaceAIConfig, apply_safety_policy

    cfg = SurfaceAIConfig(min_margin=0.15, min_confidence=0.65)
    assert apply_safety_policy(
        is_surface_known=True,
        true_surface="asphalt",
        pred_label="ground",
        confidence=0.95,
        margin=0.80,
        has_features=True,
        config=cfg,
    ) == ("asphalt", "paved_good", "osm")
    assert apply_safety_policy(
        is_surface_known=False,
        true_surface="unknown",
        pred_label="asphalt",
        confidence=0.80,
        margin=0.05,
        has_features=True,
        config=cfg,
    ) == ("unknown", "unknown", "ml_ambiguous")


def test_runtime_predictions_csv_required_columns(tmp_path: Path) -> None:
    row = {c: "" for c in REQUIRED_RUNTIME_COLUMNS}
    row.update(
        {
            "edge_id": "1_2_0",
            "surface_pred_group": "paved_rough",
            "surface_pred_confidence": "0.9",
            "surface_pred_margin": "0.3",
            "surface_ml_safe": "true",
        }
    )
    p = tmp_path / "rp.csv"
    write_runtime_predictions_csv(pd.DataFrame([row]), p)
    df = pd.read_csv(p)
    assert not df.empty
    assert all(c in df.columns for c in REQUIRED_RUNTIME_COLUMNS)


def test_neighbor_features_no_leak_from_predict_holdout_labels() -> None:
    from bike_router.services.surface_ai import add_neighbor_features

    df = pd.DataFrame(
        {
            "edge_id": ["e0", "e1", "e2"],
            "u": ["a", "b", "c"],
            "v": ["b", "c", "d"],
            "highway": ["residential", "residential", "path"],
            "length_m": [10.0, 20.0, 30.0],
            "is_surface_known": [True, True, True],
            "surface_true_group": ["paved_good", "paved_rough", "unpaved_soft"],
            "surface_ai_split": ["train", "predict_holdout", "train"],
        }
    )
    out = add_neighbor_features(df)
    row = out.set_index("edge_id").loc["e1"]
    assert row["neighbor_1hop_count"] == 2.0


def test_tile_usage_report_writes_csv_and_png(tmp_path: Path) -> None:
    from bike_router.services.surface_ai import artifact_paths, scan_cached_tiles, write_tile_usage_report

    tile_path = tmp_path / "google_20_1_2.jpg"
    tile_path.write_bytes(b"x")
    tiles = scan_cached_tiles(tmp_path, "google", 20)
    polygon = box(-180.0, 85.050, -179.998, 85.052)
    edges = gpd.GeoDataFrame(
        {"edge_id": ["e1"], "geometry": [LineString([(-179.9998, 85.0510), (-179.9995, 85.0512)])]},
        crs="EPSG:4326",
    )
    artifacts = artifact_paths(tmp_path)
    summary = write_tile_usage_report(tiles, edges, polygon, artifacts)
    assert artifacts.tile_usage_csv.exists()
    assert artifacts.tile_usage_map_png.exists()
    assert "tiles_total_in_cache" in summary
