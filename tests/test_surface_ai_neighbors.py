from __future__ import annotations

import pandas as pd


def test_neighbor_features_mask_test_labels() -> None:
    from bike_router.services.surface_ai import add_neighbor_features

    df = pd.DataFrame(
        {
            "edge_id": ["e0", "e1", "e2", "e3"],
            "u": ["a", "b", "c", "d"],
            "v": ["b", "c", "d", "e"],
            "highway": ["residential", "residential", "path", "service"],
            "length_m": [10.0, 20.0, 30.0, 40.0],
            "is_surface_known": [True, True, True, True],
            "surface_true_group": ["paved_good", "paved_rough", "unpaved_soft", "paved_good"],
            "surface_ai_split": ["train", "test", "test", "train"],
        }
    )

    out = add_neighbor_features(df)
    row = out.set_index("edge_id").loc["e1"]

    assert row["neighbor_1hop_count"] == 2.0
    assert row["neighbor_1hop_known_surface_share"] == 0.5
    assert row["neighbor_1hop_paved_good_share"] == 1.0
    assert row["neighbor_1hop_unpaved_soft_share"] == 0.0
    assert row["neighbor_1hop_same_highway_share"] == 0.5


def test_neighbor_features_columns_are_present_for_isolated_edge() -> None:
    from bike_router.services.surface_ai import NEIGHBOR_FEATURES, add_neighbor_features

    df = pd.DataFrame(
        {
            "edge_id": ["e0"],
            "u": ["a"],
            "v": ["b"],
            "highway": ["path"],
            "length_m": [5.0],
            "is_surface_known": [True],
            "surface_true_group": ["unpaved_soft"],
            "surface_ai_split": ["train"],
        }
    )

    out = add_neighbor_features(df)

    assert set(NEIGHBOR_FEATURES) <= set(out.columns)
    assert out.loc[0, "neighbor_1hop_count"] == 0.0
    assert out.loc[0, "neighbor_2hop_count"] == 0.0
