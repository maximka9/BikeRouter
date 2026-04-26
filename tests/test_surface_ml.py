from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString


def test_surface_group_mapping_and_normalization() -> None:
    from bike_router.services.surface_ml import (
        normalize_surface,
        surface_to_group,
    )

    assert normalize_surface(" Asphalt ") == "asphalt"
    assert normalize_surface("['gravel', 'ground']") == "gravel"
    assert surface_to_group("asphalt") == "paved_good"
    assert surface_to_group("concrete:plates") == "paved_rough"
    assert surface_to_group("fine_gravel") == "unpaved_hard"
    assert surface_to_group("mud") == "unpaved_soft"
    assert surface_to_group(None) == "unknown"


def test_prediction_policy_low_confidence_and_known_osm() -> None:
    from bike_router.services.surface_ml import (
        SurfaceMLConfig,
        apply_prediction_policy,
    )

    config = SurfaceMLConfig(min_confidence=0.65, paved_good_min_confidence=0.75)
    pred, effective, source, confidence = apply_prediction_policy(
        "paved_good",
        0.70,
        is_surface_known=False,
        true_group="unknown",
        has_features=True,
        config=config,
    )
    assert pred == "paved_good"
    assert effective == "paved_rough"
    assert source == "default_low_confidence"
    assert confidence == 0.70

    pred, effective, source, confidence = apply_prediction_policy(
        "unpaved_soft",
        0.80,
        is_surface_known=True,
        true_group="paved_good",
        has_features=True,
        config=config,
    )
    assert (pred, effective, source, confidence) == (
        "paved_good",
        "paved_good",
        "osm",
        1.0,
    )


def test_missing_tile_features_do_not_drop_edge(tmp_path: Path) -> None:
    from bike_router.config import Settings
    from bike_router.services.surface_ml import (
        SurfaceMLConfig,
        extract_tile_features_for_edges,
        no_progress,
    )

    base_dir = tmp_path / "data"
    (base_dir / "cache" / "tiles").mkdir(parents=True)
    settings = Settings()
    settings.base_dir = str(base_dir)
    settings.tms_server = "google"
    settings.satellite_zoom = 20

    gdf = gpd.GeoDataFrame(
        {
            "edge_id": ["e1"],
            "geometry": [LineString([(50.0, 53.0), (50.0001, 53.0001)])],
        },
        crs="EPSG:4326",
    )
    features = extract_tile_features_for_edges(
        gdf,
        settings,
        SurfaceMLConfig(sample_step_m=5.0, pixel_window=5),
        progress=no_progress,
    )

    assert len(features) == 1
    assert features.loc[0, "edge_id"] == "e1"
    assert bool(features.loc[0, "tile_missing"]) is True
    assert features.loc[0, "sampled_pixel_count"] == 0
    assert np.isnan(features.loc[0, "rgb_mean_r"])


def test_spatial_split_keeps_grid_cells_disjoint() -> None:
    from bike_router.services.surface_ml import spatial_train_test_split

    rows = []
    geoms = []
    groups = ["paved_good", "paved_rough", "unpaved_soft", "unpaved_hard"]
    for i in range(40):
        lon = 50.0 + i * 0.002
        geoms.append(LineString([(lon, 53.0), (lon + 0.0004, 53.0004)]))
        rows.append(
            {
                "edge_id": f"e{i}",
                "surface_group_true": groups[i % len(groups)],
                "is_surface_known": True,
            }
        )
    known = pd.DataFrame(rows)
    edges = gpd.GeoDataFrame(
        {"edge_id": known["edge_id"].tolist(), "geometry": geoms},
        crs="EPSG:4326",
    )

    train_mask, test_mask, cells = spatial_train_test_split(
        known,
        edges,
        grid_m=150.0,
        test_share=0.25,
        random_state=42,
    )

    assert train_mask.any()
    assert test_mask.any()
    assert set(cells.loc[train_mask]).isdisjoint(set(cells.loc[test_mask]))


def test_surface_ml_experiment_cli_has_main() -> None:
    from bike_router.tools import surface_ml_experiment as m

    assert callable(getattr(m, "main", None))

