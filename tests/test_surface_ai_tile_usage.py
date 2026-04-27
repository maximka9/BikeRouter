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
    assert "tile_usage_share_inside_polygon" in summary
