from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString


def test_spatial_split_has_calibration_and_no_test_overlap() -> None:
    from bike_router.services.surface_ai import SurfaceAIConfig, spatial_train_cal_test_split

    n = 12
    dataset = pd.DataFrame(
        {
            "edge_id": [f"e{i}" for i in range(n)],
            "is_surface_known": [True] * n,
            "surface_train_label": ["asphalt", "ground"] * (n // 2),
        }
    )
    edges = gpd.GeoDataFrame(
        {
            "edge_id": [f"e{i}" for i in range(n)],
            "geometry": [
                LineString([(i * 0.002, 0.0), (i * 0.002 + 0.0005, 0.0)])
                for i in range(n)
            ],
        },
        crs="EPSG:4326",
    )
    config = SurfaceAIConfig(
        min_known_edges=4,
        spatial_grid_m=50,
        test_share=0.25,
        calibration_share=0.25,
        calibration_enabled=True,
        random_state=7,
    )

    split = spatial_train_cal_test_split(dataset, edges, config)

    assert set(split.unique()) >= {"train", "calibration", "test"}
    assert set(split[split == "calibration"].index).isdisjoint(set(split[split == "test"].index))


def test_reliability_table_metrics() -> None:
    from bike_router.services.surface_ai import reliability_table_from_proba

    proba = pd.DataFrame(
        {
            "asphalt": [0.9, 0.6, 0.4],
            "ground": [0.1, 0.4, 0.6],
        }
    )
    table, metrics = reliability_table_from_proba(["asphalt", "ground", "ground"], proba, n_bins=3)

    assert len(table) == 3
    assert 0.0 <= metrics["ece_top_label"] <= 1.0
    assert 0.0 <= metrics["brier_top_label"] <= 1.0
    assert metrics["mean_confidence_correct"] > 0.0
