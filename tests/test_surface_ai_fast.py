"""Быстрые unit-проверки Surface AI / ML (без обучения, без тайлов/OSM)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from dataclasses import replace

from shapely.geometry import LineString, box

from bike_router.config import Settings

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


def test_dedupe_dataframe_by_edge_id() -> None:
    from bike_router.services.surface_ml import dedupe_dataframe_by_edge_id

    df = pd.DataFrame({"edge_id": ["a", "a", "b"], "v": [1, 2, 3]})
    out = dedupe_dataframe_by_edge_id(df, keep="last", name="t")
    assert list(out["edge_id"]) == ["a", "b"]
    assert int(out.loc[out["edge_id"] == "a", "v"].iloc[0]) == 2


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


def test_surface_ai_edges_cache_fingerprint_stable(tmp_path: Path) -> None:
    from bike_router.services.surface_ai import (
        SurfaceAIConfig,
        surface_ai_edges_cache_fingerprint,
    )

    settings = replace(Settings(), base_dir=str(tmp_path))
    cfg = SurfaceAIConfig()
    train_poly = box(0, 0, 0.1, 0.1)
    predict_poly = box(0, 0, 0.2, 0.2)
    tile_cov = box(0, 0, 0.15, 0.15)
    tiles_gdf = gpd.GeoDataFrame(
        {"tile_id": ["srv_20_1_2"], "geometry": [box(-0.01, -0.01, 0.01, 0.01)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    fp1 = surface_ai_edges_cache_fingerprint(
        settings=settings,
        config=cfg,
        train_polygon=train_poly,
        predict_polygon=predict_poly,
        tile_coverage_polygon=tile_cov,
        tiles_gdf=tiles_gdf,
        tiles_dir=tiles_dir,
        tms_server="google",
        tile_zoom=20,
    )
    fp2 = surface_ai_edges_cache_fingerprint(
        settings=settings,
        config=cfg,
        train_polygon=train_poly,
        predict_polygon=predict_poly,
        tile_coverage_polygon=tile_cov,
        tiles_gdf=tiles_gdf,
        tiles_dir=tiles_dir,
        tms_server="google",
        tile_zoom=20,
    )
    assert fp1 == fp2
    assert len(fp1) == 40


def test_surface_ai_edges_cache_parquet_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")

    from bike_router.services.surface_ai import (
        SurfaceAIConfig,
        _try_load_surface_ai_edges_cache,
        _write_surface_ai_edges_cache,
        surface_ai_edges_cache_fingerprint,
        surface_ai_edges_cache_read_roots,
        surface_ai_edges_cache_write_root,
    )

    settings = replace(Settings(), base_dir=str(tmp_path))
    cfg = SurfaceAIConfig(
        edges_cache_write=True,
        edges_cache_dir=str(tmp_path / "edge_cache_sub"),
    )
    train_poly = box(0, 0, 0.1, 0.1)
    predict_poly = box(0, 0, 0.2, 0.2)
    tile_cov = box(0, 0, 0.15, 0.15)
    tiles_gdf = gpd.GeoDataFrame(
        {"tile_id": ["srv_20_5_5"], "geometry": [box(0, 0, 0.02, 0.02)]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    tiles_dir = tmp_path / "tiles2"
    tiles_dir.mkdir()
    fp = surface_ai_edges_cache_fingerprint(
        settings=settings,
        config=cfg,
        train_polygon=train_poly,
        predict_polygon=predict_poly,
        tile_coverage_polygon=tile_cov,
        tiles_gdf=tiles_gdf,
        tiles_dir=tiles_dir,
        tms_server="google",
        tile_zoom=20,
    )
    root = surface_ai_edges_cache_write_root(settings, cfg)
    train_edges = gpd.GeoDataFrame(
        {"edge_id": ["a"], "geometry": [LineString([(0, 0), (0.01, 0)])]},
        crs="EPSG:4326",
    )
    predict_edges = gpd.GeoDataFrame(
        {"edge_id": ["b"], "geometry": [LineString([(0.1, 0.1), (0.11, 0.1)])]},
        crs="EPSG:4326",
    )
    _write_surface_ai_edges_cache(
        settings,
        cfg,
        root,
        fp,
        train_edges,
        predict_edges,
        run_meta={"train_graph_source": "unit"},
    )
    loaded = _try_load_surface_ai_edges_cache(surface_ai_edges_cache_read_roots(settings, cfg), fp)
    assert loaded is not None
    t1, p1, rm, hit = loaded
    assert rm.get("train_graph_source") == "unit"
    assert list(t1["edge_id"]) == ["a"]
    assert list(p1["edge_id"]) == ["b"]
    assert "edge_cache_sub" in str(hit)


def test_surface_ai_edges_cache_refuses_write_inside_data_cache(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    from bike_router.services.surface_ai import (
        SurfaceAIConfig,
        _write_surface_ai_edges_cache,
        surface_ai_edges_cache_write_root,
    )

    settings = replace(Settings(), base_dir=str(tmp_path))
    bad_root = Path(settings.cache_dir) / "surface_ai_edges_should_not_appear"
    cfg = SurfaceAIConfig(edges_cache_write=True, edges_cache_dir=str(bad_root))
    root = surface_ai_edges_cache_write_root(settings, cfg)
    train_edges = gpd.GeoDataFrame(
        {"edge_id": ["a"], "geometry": [LineString([(0, 0), (0.01, 0)])]},
        crs="EPSG:4326",
    )
    predict_edges = gpd.GeoDataFrame(
        {"edge_id": ["b"], "geometry": [LineString([(0.1, 0.1), (0.11, 0.1)])]},
        crs="EPSG:4326",
    )
    _write_surface_ai_edges_cache(
        settings,
        cfg,
        root,
        "deadbeef",
        train_edges,
        predict_edges,
        run_meta={},
    )
    assert not (bad_root / "deadbeef" / "train_edges.parquet").is_file()
