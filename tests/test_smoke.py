"""Дымовые проверки пакета без загрузки графа OSM."""

from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch


class SmokeTests(unittest.TestCase):
    def test_api_app_metadata(self) -> None:
        from bike_router.api import app

        self.assertIn("Bike", app.title)

    def test_routing_fingerprint_deterministic(self) -> None:
        from bike_router.config import routing_engine_cache_fingerprint

        a = routing_engine_cache_fingerprint()
        b = routing_engine_cache_fingerprint()
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)

    def test_models_route_quality(self) -> None:
        from bike_router.models import RouteQualityHints

        q = RouteQualityHints(
            warnings=[],
            na_surface_fraction=0.1,
            inferred_surface_fraction=0.2,
            inferred_highway_fraction=0.05,
        )
        self.assertEqual(q.na_surface_fraction, 0.1)

    def test_application_composition_root(self) -> None:
        from bike_router.app import Application

        app = Application()
        self.assertIsNotNone(app.router)
        self.assertIsNotNone(app.graph_builder)

    def test_settings_area_polygon_wkt_flag(self) -> None:
        from bike_router.config import Settings

        samara = (
            "POLYGON ((50.08 53.18, 50.22 53.18, 50.22 53.26, "
            "50.08 53.26, 50.08 53.18))"
        )
        with patch.dict(os.environ, {"AREA_POLYGON_WKT": samara}):
            s = Settings()
            self.assertTrue(s.has_area_polygon)
            self.assertEqual(s.area_polygon_wkt_stripped, samara)
        with patch.dict(os.environ, {"AREA_POLYGON_WKT": "LINESTRING (0 0, 1 1)"}):
            self.assertFalse(Settings().has_area_polygon)

    def test_corridor_mode_requires_no_fixed_area(self) -> None:
        import importlib

        import bike_router.config as cfg

        with patch.dict(
            os.environ,
            {
                "GRAPH_CORRIDOR_MODE": "true",
                "AREA_MIN_LAT": "53",
                "AREA_MAX_LAT": "54",
                "AREA_MIN_LON": "50",
                "AREA_MAX_LON": "51",
                "AREA_POLYGON_WKT": "",
            },
        ):
            importlib.reload(cfg)
            self.assertFalse(cfg.Settings().use_dynamic_corridor_graph)
        with patch.dict(
            os.environ,
            {
                "GRAPH_CORRIDOR_MODE": "true",
                "AREA_MIN_LAT": "0",
                "AREA_MAX_LAT": "0",
                "AREA_MIN_LON": "0",
                "AREA_MAX_LON": "0",
                "AREA_POLYGON_WKT": "",
            },
        ):
            importlib.reload(cfg)
            self.assertTrue(cfg.Settings().use_dynamic_corridor_graph)
        importlib.reload(cfg)

    def test_corridor_bbox_cache_key_stable(self) -> None:
        from bike_router.config import Settings
        from bike_router.services.corridor_graph_cache import corridor_bbox_cache_key

        s = Settings()
        a = corridor_bbox_cache_key(50.08, 53.18, 50.22, 53.26, s)
        b = corridor_bbox_cache_key(50.08, 53.18, 50.22, 53.26, s)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)

        c = corridor_bbox_cache_key(50.0800001, 53.18, 50.22, 53.26, s)
        self.assertEqual(a, c)

        stub = corridor_bbox_cache_key(
            50.08, 53.18, 50.22, 53.26, s, skip_satellite_green=True
        )
        full = corridor_bbox_cache_key(
            50.08, 53.18, 50.22, 53.26, s, skip_satellite_green=False
        )
        self.assertEqual(stub, full)

    def test_corridor_bbox_cache_key_grid_merges_nearby(self) -> None:
        from bike_router.config import Settings
        from bike_router.services.corridor_graph_cache import corridor_bbox_cache_key

        s = replace(Settings(), corridor_cache_grid_step_deg=0.01)
        a = corridor_bbox_cache_key(50.0800001, 53.1800001, 50.119999, 53.199999, s)
        b = corridor_bbox_cache_key(50.08, 53.18, 50.12, 53.20, s)
        self.assertEqual(a, b)

    def test_surface_resolve_pipeline(self) -> None:
        from bike_router.services.surface_resolve import resolve_surface_effective

        self.assertEqual(
            resolve_surface_effective("asphalt", "track", "grade5"),
            "asphalt",
        )
        self.assertEqual(resolve_surface_effective("", "track", None), "unpaved")
        self.assertEqual(resolve_surface_effective("", "path", "grade1"), "paved")
        self.assertEqual(resolve_surface_effective("", "pedestrian", None), "paved")
        self.assertEqual(resolve_surface_effective("", None, None), "unknown")

    def test_settings_has_area_bbox(self) -> None:
        from bike_router.config import Settings

        z = replace(
            Settings(),
            area_min_lat=0.0,
            area_max_lat=0.0,
            area_min_lon=0.0,
            area_max_lon=0.0,
        )
        self.assertFalse(z.has_area_bbox)
        ok = replace(
            Settings(),
            area_min_lat=53.0,
            area_max_lat=54.0,
            area_min_lon=50.0,
            area_max_lon=51.0,
        )
        self.assertTrue(ok.has_area_bbox)
        flat = replace(
            Settings(),
            area_min_lat=53.0,
            area_max_lat=53.0,
            area_min_lon=50.0,
            area_max_lon=51.0,
        )
        self.assertFalse(flat.has_area_bbox)

    def test_cache_service_atomic_wrapper(self) -> None:
        from bike_router.services.cache import CacheService

        tmp = tempfile.mkdtemp()
        try:
            cs = CacheService(tmp)
            path = os.path.join(cs.cache_dir, "sub", "x.pkl")
            self.assertTrue(cs.save(path, {"k": 2}))
            self.assertEqual(cs.load(path), {"k": 2})
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_run_scenarios_metrics_extraction(self) -> None:
        from bike_router.tools.run_route_scenarios import _route_stairs_and_quality

        ln, seg, na = _route_stairs_and_quality(
            {
                "stairs": {"total_length_m": 12.5, "count": 3},
                "quality_hints": {"na_surface_fraction": 0.07},
            }
        )
        self.assertEqual(ln, 12.5)
        self.assertEqual(seg, 3)
        self.assertEqual(na, "7.0")


if __name__ == "__main__":
    unittest.main()
