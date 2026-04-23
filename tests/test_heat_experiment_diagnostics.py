"""Диагностики heat_weather: стем ветра, merge KPI, QA, зимний reroute."""

from __future__ import annotations

from bike_router.tools._experiment_common import (
    _weather_case_wind_stem,
    build_heat_experiment_extra_sheet_blocks,
    build_heat_weather_kpi_extras_dict,
    build_heat_weather_kpi_rows,
    build_winter_reroute_explain_rows,
)


def test_weather_case_wind_stem() -> None:
    assert _weather_case_wind_stem("test_T250_R0_W0_C0_WD90") == "test_T250_R0_W0_C0"
    assert _weather_case_wind_stem("winter_T-10_F1_D100_W1_WD270") == "winter_T-10_F1_D100_W1"
    assert _weather_case_wind_stem("no_suffix") == "no_suffix"


def test_wind_stem_geometry_group_count() -> None:
    rows = [
        {
            "origin_point_id": "P01",
            "destination_point_id": "P02",
            "profile": "cyclist",
            "weather_test_case_id": "base_WD0",
            "geometry_json": '{"a":1}',
            "route_mean_wind_orientation_factor": 1.01,
            "route_mean_green_route_factor": 1.0,
            "route_covered_share": 0.0,
            "route_stairs_length_fraction": 0.01,
            "route_winter_harsh_surface_share": 0.0,
            "route_building_shade_share": 0.1,
            "route_open_sky_share": 0.5,
            "route_mean_stress_weather_factor": 1.0,
            "route_mean_surface_weather_factor": 1.0,
            "route_bad_wet_surface_share": 0.0,
        },
        {
            "origin_point_id": "P01",
            "destination_point_id": "P02",
            "profile": "cyclist",
            "weather_test_case_id": "base_WD90",
            "geometry_json": '{"b":2}',
            "route_mean_wind_orientation_factor": 1.04,
            "route_mean_green_route_factor": 1.0,
            "route_covered_share": 0.0,
            "route_stairs_length_fraction": 0.01,
            "route_winter_harsh_surface_share": 0.0,
            "route_building_shade_share": 0.1,
            "route_open_sky_share": 0.5,
            "route_mean_stress_weather_factor": 1.0,
            "route_mean_surface_weather_factor": 1.0,
            "route_bad_wet_surface_share": 0.0,
        },
    ]
    ex = build_heat_weather_kpi_extras_dict(rows)
    assert ex["wind_dir_only_geometry_change_groups"] == 1
    assert ex["mean_wind_orientation_factor_range_per_od_pair"] is not None


def test_merge_kpi_contains_green_note() -> None:
    rows = [
        {
            "origin_point_id": "P01",
            "destination_point_id": "P02",
            "profile": "cyclist",
            "weather_test_case_id": "c1",
            "weather_test_temperature_c": 0.0,
            "weather_test_precipitation_mm": 0.0,
            "weather_test_wind_speed_ms": 2.0,
            "weather_test_cloud_cover_pct": 50.0,
            "route_open_sky_share": 0.5,
            "geometry_json": "g",
            "route_mean_green_route_factor": 1.0,
            "route_mean_wind_orientation_factor": 1.0,
            "route_covered_share": 0.0,
            "route_stairs_length_fraction": 0.0,
            "route_winter_harsh_surface_share": 0.0,
            "route_building_shade_share": 0.05,
            "route_mean_stress_weather_factor": 1.0,
            "route_mean_surface_weather_factor": 1.0,
            "route_bad_wet_surface_share": 0.0,
        }
    ]
    k = build_heat_weather_kpi_rows(rows)[0]
    k.update(build_heat_weather_kpi_extras_dict(rows))
    assert "green_route_factor_heat_note_ru" in k
    assert "validation_coverage_summary_ru" in k


def test_winter_reroute_explain() -> None:
    rows = [
        {
            "origin_point_id": "P01",
            "destination_point_id": "P02",
            "profile": "pedestrian",
            "weather_test_case_id": "winter_T0_F0_D0_W0_WD0",
            "geometry_json": '{"x":1}',
            "route_stairs_length_fraction": 0.0,
            "route_winter_harsh_surface_share": 0.1,
            "route_open_sky_share": 0.4,
            "route_building_shade_share": 0.2,
            "route_mean_stress_weather_factor": 1.05,
        },
        {
            "origin_point_id": "P01",
            "destination_point_id": "P02",
            "profile": "pedestrian",
            "weather_test_case_id": "winter_T0_F0_D0_W0_WD90",
            "geometry_json": '{"y":2}',
            "route_stairs_length_fraction": 0.05,
            "route_winter_harsh_surface_share": 0.3,
            "route_open_sky_share": 0.5,
            "route_building_shade_share": 0.1,
            "route_mean_stress_weather_factor": 1.2,
        },
    ]
    out = build_winter_reroute_explain_rows(rows)
    assert len(out) == 1
    assert out[0]["winter_distinct_geometries"] == 2
    assert out[0]["strongest_relative_range_metric"]


def test_extra_blocks_structure() -> None:
    blocks = build_heat_experiment_extra_sheet_blocks([])
    assert len(blocks) == 2
    assert blocks[0][0].startswith("QA")
