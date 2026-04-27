from __future__ import annotations

import geopandas as gpd
from shapely.geometry import LineString, box


def test_tile_scanner_parses_supported_extensions(tmp_path) -> None:
    from bike_router.services.surface_ai import scan_cached_tiles

    (tmp_path / "google_20_1_2.jpg").write_bytes(b"x")
    (tmp_path / "google_20_3_4.png").write_bytes(b"x")
    (tmp_path / "google_20_5_6.webp").write_bytes(b"x")
    (tmp_path / "other_20_7_8.jpg").write_bytes(b"x")

    tiles = scan_cached_tiles(tmp_path, "google", 20)

    assert set(tiles["tile_id"]) == {
        "google_20_1_2",
        "google_20_3_4",
        "google_20_5_6",
    }
    assert {"tile_id", "server", "z", "x", "y", "path", "footprint_area_m2"} <= set(tiles.columns)


def test_tile_usage_report_fields_exist(tmp_path) -> None:
    from bike_router.services.surface_ai import SurfaceAIArtifacts, scan_cached_tiles, write_tile_usage_report

    tile_path = tmp_path / "google_20_1_2.jpg"
    tile_path.write_bytes(b"x")
    tiles = scan_cached_tiles(tmp_path, "google", 20)
    polygon = box(-180.0, 85.050, -179.998, 85.052)
    edges = gpd.GeoDataFrame(
        {"edge_id": ["e1"], "geometry": [LineString([(-179.9998, 85.0510), (-179.9995, 85.0512)])]},
        crs="EPSG:4326",
    )
    artifacts = SurfaceAIArtifacts(
        output_dir=tmp_path,
        dataset_csv=tmp_path / "dataset.csv",
        predictions_csv=tmp_path / "pred.csv",
        predictions_geojson=tmp_path / "pred.geojson",
        baseline_csv=tmp_path / "baseline.csv",
        baseline_xlsx=tmp_path / "baseline.xlsx",
        baseline_png=tmp_path / "baseline.png",
        metrics_json=tmp_path / "metrics.json",
        model_joblib=tmp_path / "model.joblib",
        model_card_json=tmp_path / "model_card.json",
        report_txt=tmp_path / "report.txt",
        surface_map_png=tmp_path / "surface.png",
        confidence_map_png=tmp_path / "confidence.png",
        unknown_predictions_map_png=tmp_path / "unknown.png",
        errors_map_png=tmp_path / "errors.png",
        confusion_matrix_concrete_png=tmp_path / "cm_concrete.png",
        confusion_matrix_group_png=tmp_path / "cm_group.png",
        feature_importance_png=tmp_path / "feature.png",
        confusion_matrix_concrete_compact_png=tmp_path / "cm_concrete_compact.png",
        confusion_matrix_concrete_legacy_png=tmp_path / "cm_concrete_legacy.png",
        confusion_matrix_group_direct_png=tmp_path / "cm_group_direct.png",
        group_direct_model_joblib=tmp_path / "group.joblib",
        group_direct_model_card_json=tmp_path / "group_card.json",
        group_direct_metrics_json=tmp_path / "group_metrics.json",
        calibration_curve_concrete_png=tmp_path / "cal_concrete.png",
        calibration_curve_group_direct_png=tmp_path / "cal_group.png",
        reliability_table_concrete_csv=tmp_path / "rel_concrete.csv",
        reliability_table_group_direct_csv=tmp_path / "rel_group.csv",
        dangerous_errors_map_raw_png=tmp_path / "danger_raw.png",
        dangerous_errors_map_effective_png=tmp_path / "danger_eff.png",
        dangerous_errors_csv=tmp_path / "danger.csv",
        spatial_prior_ablation_txt=tmp_path / "ablation.txt",
        tile_usage_csv=tmp_path / "tile_usage.csv",
        tile_usage_map_png=tmp_path / "tile_usage.png",
        neighbor_feature_importance_png=tmp_path / "neighbor.png",
    )

    summary = write_tile_usage_report(tiles, edges, polygon, artifacts)

    assert artifacts.tile_usage_csv.exists()
    assert artifacts.tile_usage_map_png.exists()
    assert "tiles_total_in_cache" in summary
    assert "tile_usage_share_inside_polygon" in summary
