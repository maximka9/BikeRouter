from __future__ import annotations

import pandas as pd


def test_normalize_surface_value_concrete_cases() -> None:
    from bike_router.services.surface_ai import normalize_surface_value

    assert normalize_surface_value("asphalt;concrete") == "asphalt"
    assert normalize_surface_value("paving_stones:30") == "paving_stones"
    assert normalize_surface_value("concrete:plates") == "concrete:plates"
    assert normalize_surface_value("gravel;ground") == "gravel"
    assert normalize_surface_value("yes") == "paved"
    assert normalize_surface_value("no") is None


def test_highway_heuristic_baseline() -> None:
    from bike_router.services.surface_ai import predict_surface_by_highway_heuristic

    assert predict_surface_by_highway_heuristic({"highway": "residential"}) == "asphalt"
    assert predict_surface_by_highway_heuristic({"highway": "footway"}) == "paving_stones"
    assert (
        predict_surface_by_highway_heuristic(
            {"highway": "track", "tracktype": "grade2"}
        )
        == "gravel"
    )
    assert predict_surface_by_highway_heuristic({"highway": "path"}) == "ground"
    assert predict_surface_by_highway_heuristic({"highway": "steps"}) == "concrete"


def test_rare_class_train_labels() -> None:
    from bike_router.services.surface_ai import add_train_labels

    df = pd.DataFrame(
        {
            "is_surface_known": [True, True, True, True, True],
            "surface_true_concrete": [
                "asphalt",
                "asphalt",
                "ground",
                "mud",
                "paving_stones",
            ],
        }
    )
    out, meta = add_train_labels(df, min_class_count=2)
    assert out.loc[0, "surface_train_label"] == "asphalt"
    assert out.loc[2, "surface_train_label"] == "rare_other_unpaved_soft"
    assert bool(out.loc[3, "surface_is_rare_class"]) is True
    assert meta["frequent_surface_classes"] == ["asphalt"]


def test_safety_policy_ambiguous_and_known_osm() -> None:
    from bike_router.services.surface_ai import SurfaceAIConfig, apply_safety_policy

    config = SurfaceAIConfig(min_margin=0.15, min_confidence=0.65)
    assert apply_safety_policy(
        is_surface_known=True,
        true_surface="asphalt",
        pred_label="ground",
        confidence=0.95,
        margin=0.80,
        has_features=True,
        config=config,
    ) == ("asphalt", "paved_good", "osm")
    assert apply_safety_policy(
        is_surface_known=False,
        true_surface="unknown",
        pred_label="asphalt",
        confidence=0.80,
        margin=0.05,
        has_features=True,
        config=config,
    ) == ("unknown", "unknown", "ml_ambiguous")


def test_surface_ai_experiment_cli_has_main() -> None:
    from bike_router.tools import surface_ai_experiment as m

    assert callable(getattr(m, "main", None))
