"""KPI-сводка heat_weather_experiment (синтетика)."""

from __future__ import annotations

from bike_router.tools._experiment_common import build_heat_weather_kpi_rows


def _row(
    *,
    case: str,
    o: str,
    d: str,
    prof: str,
    T: float,
    rain_mm: float,
    ws: float,
    cc: float,
    open_s: float,
    geom: str,
    covered: float = 0.0,
) -> dict:
    return {
        "origin_point_id": o,
        "destination_point_id": d,
        "profile": prof,
        "direction_key": f"{o}->{d}",
        "weather_test_case_id": case,
        "weather_test_temperature_c": T,
        "weather_test_precipitation_mm": rain_mm,
        "weather_test_wind_speed_ms": ws,
        "weather_test_wind_gusts_ms": ws + 1.0,
        "weather_test_cloud_cover_pct": cc,
        "weather_test_humidity_pct": 55.0,
        "weather_test_shortwave_radiation_wm2": 400.0,
        "route_open_sky_share": open_s,
        "route_building_shade_share": 0.05,
        "route_covered_share": covered,
        "route_bad_wet_surface_share": 0.1,
        "geometry_json": geom,
    }


def test_build_heat_weather_kpi_changed_pairs_and_temp_share() -> None:
    rows = []
    base = ("P01", "P02", "cyclist")
    for T, g, o_s in (
        (0.0, "geomA", 0.9),
        (25.0, "geomB", 0.5),
    ):
        rows.append(
            _row(
                case=f"c_{T}",
                o=base[0],
                d=base[1],
                prof=base[2],
                T=T,
                rain_mm=0.0,
                ws=2.0,
                cc=50.0,
                open_s=o_s,
                geom=g,
            )
        )
    k = build_heat_weather_kpi_rows(rows)[0]
    assert k["changed_pairs_count"] == 1
    assert k["changed_pairs_share"] == 1.0
    assert k["mean_unique_geometries"] == 2.0
    assert k["temp25_vs_0_open_lower_share"] == 1.0
    assert k["covered_nonzero_share"] == 0.0


def test_build_heat_weather_kpi_rain_pair() -> None:
    rows = []
    for rain, g, o_s in ((0.0, "g1", 0.8), (1.5, "g2", 0.4)):
        rows.append(
            _row(
                case=f"r_{rain}",
                o="P01",
                d="P02",
                prof="cyclist",
                T=12.5,
                rain_mm=rain,
                ws=2.0,
                cc=50.0,
                open_s=o_s,
                geom=g,
            )
        )
    k = build_heat_weather_kpi_rows(rows)[0]
    assert k["rain_vs_dry_open_lower_share"] == 1.0


def test_build_heat_weather_kpi_covered_nonzero() -> None:
    rows = [
        _row(
            case="c1",
            o="P01",
            d="P02",
            prof="cyclist",
            T=0.0,
            rain_mm=0.0,
            ws=2.0,
            cc=0.0,
            open_s=0.5,
            geom="x",
            covered=0.02,
        )
    ]
    k = build_heat_weather_kpi_rows(rows)[0]
    assert k["covered_nonzero_share"] == 1.0
