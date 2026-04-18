"""Модульные тесты тепловой модели, стресса и профилей предпочтений."""

from bike_router.config import (
    routing_preference_profile,
    time_slot_for_hour,
    time_slot_key_for_hour,
)
from bike_router.services.heat import (
    angle_diff_deg,
    exposure_units_for_all_slots,
    heat_cost_for_edge,
)
from bike_router.services.routing import RouteService
from bike_router.services.stress import lts_from_osm_tags, stress_cost_for_edge


def test_angle_diff_deg_symmetric():
    assert angle_diff_deg(0, 180) == 180
    assert angle_diff_deg(350, 10) == 20


def test_time_slot_morning_noon_evening_night():
    assert time_slot_key_for_hour(8) == "morning"
    assert time_slot_key_for_hour(13) == "noon"
    assert time_slot_key_for_hour(18) == "evening"
    assert time_slot_key_for_hour(23) == "night"
    assert time_slot_for_hour(3).key == "night"


def test_routing_preference_profiles_exist():
    for k in ("balanced", "safe", "cool", "sport"):
        p = routing_preference_profile(k)
        assert p.key == k
        assert p.alpha > 0


def test_lts_cycleway_lower_than_primary():
    low = lts_from_osm_tags(
        {"highway": "cycleway", "maxspeed": "30", "lanes": "1"}
    )
    high = lts_from_osm_tags(
        {"highway": "primary", "maxspeed": "60", "lanes": "4"}
    )
    assert low < high


def test_stress_cost_scales_with_length_and_lts():
    a = stress_cost_for_edge(100.0, 2.0)
    b = stress_cost_for_edge(100.0, 4.0)
    assert b > a > 0


def test_heat_cost_nonnegative():
    h = heat_cost_for_edge(50.0, 0.4)
    assert h >= 0


def test_exposure_slots_differ_by_sun():
    tags = {"highway": "residential", "width": "12"}
    e_m = exposure_units_for_all_slots(90.0, 10.0, 5.0, tags)
    e_n = exposure_units_for_all_slots(90.0, 10.0, 5.0, tags)
    assert set(e_m.keys()) == {"morning", "noon", "evening", "night"}
    # ночной слот слабее инсоляции
    assert e_n["night"] <= e_n["noon"]


def test_combined_edge_weight_finite():
    pref = routing_preference_profile("balanced")
    d = {
        "weight_cyclist_full": 100.0,
        "heat_noon": 10.0,
        "stress_cost": 5.0,
    }
    w = RouteService.combined_edge_weight(
        d, "cyclist", "noon", pref
    )
    assert w > 0 and w < 1e9

