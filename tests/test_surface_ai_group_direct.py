from __future__ import annotations

import pandas as pd


def test_compact_concrete_mapping() -> None:
    from bike_router.services.surface_ai import normalize_surface_to_concrete_ai_label

    assert normalize_surface_to_concrete_ai_label("asphalt") == "asphalt"
    assert normalize_surface_to_concrete_ai_label("paving_stones") == "paving_stones"
    assert normalize_surface_to_concrete_ai_label("concrete:plates") == "other_paved"
    assert normalize_surface_to_concrete_ai_label("gravel") == "other_unpaved"
    assert normalize_surface_to_concrete_ai_label("mud") == "other_unpaved"


def test_group_target_modes() -> None:
    from bike_router.services.surface_ai import group_target_labels, normalize_group_target

    assert normalize_group_target("unpaved_hard", "3class_unpaved_hard_to_soft") == "unpaved_soft"
    assert normalize_group_target("unpaved_soft", "3class_rough_or_unpaved") == "rough_or_unpaved"
    assert normalize_group_target("unpaved_hard", "4class_original") == "unpaved_hard"
    assert group_target_labels("3class_unpaved_hard_to_soft") == (
        "paved_good",
        "paved_rough",
        "unpaved_soft",
    )


def test_add_train_labels_direct_group_mode() -> None:
    from bike_router.services.surface_ai import add_train_labels

    df = pd.DataFrame(
        {
            "is_surface_known": [True, True, True],
            "surface_true_concrete": ["asphalt", "gravel", "paving_stones"],
            "surface_true_group": ["paved_good", "unpaved_hard", "paved_rough"],
        }
    )

    out, meta = add_train_labels(
        df,
        min_class_count=1,
        concrete_target_mode="compact",
        group_target_mode="3class_unpaved_hard_to_soft",
    )

    assert out["surface_train_label"].tolist() == [
        "asphalt",
        "other_unpaved",
        "paving_stones",
    ]
    assert out["surface_group_direct_label"].tolist() == [
        "paved_good",
        "unpaved_soft",
        "paved_rough",
    ]
    assert meta["group_target_mode"] == "3class_unpaved_hard_to_soft"
