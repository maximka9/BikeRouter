"""Интеграция Surface AI runtime predictions в surface_effective и веса."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import LineString

from bike_router.config import Settings
from bike_router.services.graph import GraphBuilder
from bike_router.services.surface_prediction_store import (
    SurfacePredictionStore,
    parse_runtime_artifact_graph_hash,
)
from bike_router.services.surface_resolve import apply_surface_resolution


def _minimal_edges_gdf() -> gpd.GeoDataFrame:
    geom = LineString([(0.0, 0.0), (0.001, 0.0)])
    return gpd.GeoDataFrame(
        {
            "u": [1],
            "v": [2],
            "key": [0],
            "edge_id": ["1_2_0"],
            "highway": ["residential"],
            "tracktype": [None],
            "surface": [None],
            "length": [50.0],
            "geometry": [geom],
        },
        crs="EPSG:4326",
    )


def _write_runtime_csv(path: Path, **kwargs) -> None:
    row = {
        "edge_id": "1_2_0",
        "u": 1,
        "v": 2,
        "key": 0,
        "surface_pred_group": "paved_rough",
        "surface_pred_concrete": "",
        "surface_pred_confidence": 0.9,
        "surface_pred_margin": 0.4,
        "surface_ml_safe": "true",
        "surface_ml_reject_reason": "",
        "surface_effective_ml": "",
        "model_key": "test",
        "model_target": "group_direct",
        "artifact_created_at": "2026-01-01",
        "artifact_schema_version": "1",
        "area_fingerprint": "",
        "graph_fingerprint": "",
        "geometry_hash_rounded": "",
        "osm_way_id": "",
    }
    row.update(kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)


def test_osm_surface_has_priority_over_ml(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "p.csv"
    _write_runtime_csv(csv_path, surface_pred_group="unpaved_soft")
    monkeypatch.setenv("SURFACE_AI_RUNTIME_ENABLED", "true")
    monkeypatch.setenv("SURFACE_AI_RUNTIME_PREDICTIONS_PATH", str(csv_path))
    monkeypatch.setenv("SURFACE_AI_RUNTIME_STRICT", "false")
    s = Settings()
    store = SurfacePredictionStore(s)
    store.load()
    gdf = _minimal_edges_gdf()
    gdf["surface"] = ["asphalt"]
    out, stats = apply_surface_resolution(gdf, prediction_store=store, settings=s)
    assert out["surface_source"].iloc[0] == "osm"
    assert out["surface_effective"].iloc[0] == "asphalt"
    assert stats.surface_source_osm_count == 1


def test_ml_surface_effective_maps_to_different_profile_coefficient(
    tmp_path, monkeypatch
) -> None:
    """Группа paved_rough → compacted в профиле; коэффициент отличается от unknown."""
    from bike_router.config import CYCLIST, DEFAULT_COEFFICIENT

    csv_path = tmp_path / "p.csv"
    _write_runtime_csv(csv_path)
    monkeypatch.setenv("SURFACE_AI_RUNTIME_ENABLED", "true")
    monkeypatch.setenv("SURFACE_AI_RUNTIME_PREDICTIONS_PATH", str(csv_path))
    s = Settings()
    store = SurfacePredictionStore(s)
    store.load()
    gdf = _minimal_edges_gdf()
    out, _ = apply_surface_resolution(gdf, prediction_store=store, settings=s)
    eff = str(out["surface_effective"].iloc[0])
    assert eff == "compacted"
    c_eff = float(CYCLIST.surface.get(eff, DEFAULT_COEFFICIENT))
    c_unk = float(CYCLIST.surface.get("unknown", DEFAULT_COEFFICIENT))
    assert c_eff != c_unk


def test_ml_used_only_when_surface_missing(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "p.csv"
    _write_runtime_csv(csv_path)
    monkeypatch.setenv("SURFACE_AI_RUNTIME_ENABLED", "true")
    monkeypatch.setenv("SURFACE_AI_RUNTIME_PREDICTIONS_PATH", str(csv_path))
    s = Settings()
    store = SurfacePredictionStore(s)
    store.load()
    gdf = _minimal_edges_gdf()
    out, stats = apply_surface_resolution(gdf, prediction_store=store, settings=s)
    assert out["surface_source"].iloc[0] == "ml"
    assert out["surface_effective"].iloc[0] == "compacted"
    assert stats.surface_source_ml_count == 1


def test_ml_rejected_when_low_confidence(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "p.csv"
    _write_runtime_csv(csv_path, surface_pred_confidence=0.1, surface_pred_margin=0.5)
    monkeypatch.setenv("SURFACE_AI_RUNTIME_ENABLED", "true")
    monkeypatch.setenv("SURFACE_AI_RUNTIME_PREDICTIONS_PATH", str(csv_path))
    s = Settings()
    store = SurfacePredictionStore(s)
    store.load()
    gdf = _minimal_edges_gdf()
    out, stats = apply_surface_resolution(gdf, prediction_store=store, settings=s)
    assert out["surface_source"].iloc[0] == "heuristic"
    assert stats.ml_predictions_rejected_low_confidence >= 1


def test_ml_rejected_when_low_margin(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "p.csv"
    _write_runtime_csv(csv_path, surface_pred_confidence=0.95, surface_pred_margin=0.01)
    monkeypatch.setenv("SURFACE_AI_RUNTIME_ENABLED", "true")
    monkeypatch.setenv("SURFACE_AI_RUNTIME_PREDICTIONS_PATH", str(csv_path))
    s = Settings()
    store = SurfacePredictionStore(s)
    store.load()
    gdf = _minimal_edges_gdf()
    out, stats = apply_surface_resolution(gdf, prediction_store=store, settings=s)
    assert out["surface_source"].iloc[0] == "heuristic"
    assert stats.ml_predictions_rejected_low_margin >= 1


def test_ml_paved_good_requires_high_confidence(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "p.csv"
    _write_runtime_csv(
        csv_path,
        surface_pred_group="paved_good",
        surface_pred_confidence=0.8,
        surface_pred_margin=0.5,
    )
    monkeypatch.setenv("SURFACE_AI_RUNTIME_ENABLED", "true")
    monkeypatch.setenv("SURFACE_AI_RUNTIME_PREDICTIONS_PATH", str(csv_path))
    s = Settings()
    store = SurfacePredictionStore(s)
    store.load()
    gdf = _minimal_edges_gdf()
    out, stats = apply_surface_resolution(gdf, prediction_store=store, settings=s)
    assert out["surface_source"].iloc[0] != "ml"
    assert stats.ml_predictions_rejected_paved_good_conf >= 1


def test_ml_fallback_to_heuristic_when_prediction_missing(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "p.csv"
    _write_runtime_csv(csv_path, edge_id="999_888_0", u=999, v=888, key=0)
    monkeypatch.setenv("SURFACE_AI_RUNTIME_ENABLED", "true")
    monkeypatch.setenv("SURFACE_AI_RUNTIME_PREDICTIONS_PATH", str(csv_path))
    s = Settings()
    store = SurfacePredictionStore(s)
    store.load()
    gdf = _minimal_edges_gdf()
    out, _stats = apply_surface_resolution(gdf, prediction_store=store, settings=s)
    assert out["surface_source"].iloc[0] == "heuristic"


def test_ml_fallback_to_unknown_when_no_heuristic(tmp_path, monkeypatch) -> None:
    from bike_router.services import surface_resolve as sr

    csv_path = tmp_path / "p.csv"
    _write_runtime_csv(csv_path, edge_id="999_888_0", u=999, v=888, key=0)
    monkeypatch.setenv("SURFACE_AI_RUNTIME_ENABLED", "true")
    monkeypatch.setenv("SURFACE_AI_RUNTIME_PREDICTIONS_PATH", str(csv_path))
    monkeypatch.setattr(sr, "SURFACE_AI_RUNTIME_FALLBACK_TO_HEURISTIC", False)
    s = Settings()
    store = SurfacePredictionStore(s)
    store.load()
    gdf = _minimal_edges_gdf()
    gdf["highway"] = [None]
    out, _stats = apply_surface_resolution(gdf, prediction_store=store, settings=s)
    assert out["surface_source"].iloc[0] == "unknown"


def test_prediction_store_rejects_missing_columns(tmp_path, monkeypatch) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("edge_id,x\n1_2_0,1\n", encoding="utf-8")
    monkeypatch.setenv("SURFACE_AI_RUNTIME_ENABLED", "true")
    monkeypatch.setenv("SURFACE_AI_RUNTIME_PREDICTIONS_PATH", str(p))
    monkeypatch.setenv("SURFACE_AI_RUNTIME_STRICT", "false")
    st = SurfacePredictionStore(Settings())
    st.load()
    assert not st.loaded


def test_prediction_store_rejects_duplicate_edge_ids(tmp_path, monkeypatch) -> None:
    p = tmp_path / "d.csv"
    rows = [
        {"edge_id": "1_2_0", "surface_pred_group": "paved_rough", "surface_pred_confidence": 0.9, "surface_pred_margin": 0.3, "surface_ml_safe": "true"},
        {"edge_id": "1_2_0", "surface_pred_group": "unpaved_soft", "surface_pred_confidence": 0.9, "surface_pred_margin": 0.3, "surface_ml_safe": "true"},
    ]
    pd.DataFrame(rows).to_csv(p, index=False)
    monkeypatch.setenv("SURFACE_AI_RUNTIME_ENABLED", "true")
    monkeypatch.setenv("SURFACE_AI_RUNTIME_PREDICTIONS_PATH", str(p))
    monkeypatch.setenv("SURFACE_AI_RUNTIME_STRICT", "false")
    st = SurfacePredictionStore(Settings())
    st.load()
    assert not st.loaded


def test_prediction_store_supports_undirected_edge_match(tmp_path, monkeypatch) -> None:
    from bike_router.services import surface_prediction_store as sps

    p = tmp_path / "u.csv"
    _write_runtime_csv(p, edge_id="x", u=2, v=1, key=0)
    monkeypatch.setenv("SURFACE_AI_RUNTIME_ENABLED", "true")
    monkeypatch.setenv("SURFACE_AI_RUNTIME_PREDICTIONS_PATH", str(p))
    monkeypatch.setattr(sps, "SURFACE_AI_RUNTIME_MATCH_BY", "undirected_edge_key")
    s = Settings()
    st = SurfacePredictionStore(s)
    st.load()
    gdf = _minimal_edges_gdf()
    out, _ = apply_surface_resolution(gdf, prediction_store=st, settings=s)
    assert out["surface_source"].iloc[0] == "ml"


def test_apply_weights_transfers_surface_diagnostics_to_graph(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "p.csv"
    _write_runtime_csv(csv_path)
    monkeypatch.setenv("SURFACE_AI_RUNTIME_ENABLED", "true")
    monkeypatch.setenv("SURFACE_AI_RUNTIME_PREDICTIONS_PATH", str(csv_path))
    s = Settings()
    store = SurfacePredictionStore(s)
    store.load()
    gdf = _minimal_edges_gdf()
    gdf, _ = apply_surface_resolution(gdf, prediction_store=store, settings=s)
    G = nx.MultiDiGraph()
    G.add_node(1, y=0.0, x=0.0)
    G.add_node(2, y=0.0, x=0.001)
    G.add_edge(1, 2, key=0, length=50.0)
    G2 = GraphBuilder.apply_weights(G, gdf)
    _, _, _, data = next(iter(G2.edges(keys=True, data=True)))
    assert data.get("surface_source") == "ml"
    assert "surface_resolution_reason" in data


def test_runtime_disabled_preserves_old_behavior(monkeypatch) -> None:
    monkeypatch.setenv("SURFACE_AI_RUNTIME_ENABLED", "false")
    s = Settings()
    gdf = _minimal_edges_gdf()
    out, stats = apply_surface_resolution(gdf, prediction_store=None, settings=s)
    assert out["surface_source"].iloc[0] == "heuristic"
    assert stats.surface_source_ml_count == 0


def test_osm_unknown_tag_allows_ml(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "p.csv"
    _write_runtime_csv(csv_path)
    monkeypatch.setenv("SURFACE_AI_RUNTIME_ENABLED", "true")
    monkeypatch.setenv("SURFACE_AI_RUNTIME_PREDICTIONS_PATH", str(csv_path))
    s = Settings()
    store = SurfacePredictionStore(s)
    store.load()
    gdf = _minimal_edges_gdf()
    gdf["surface"] = ["unknown"]
    out, _stats = apply_surface_resolution(gdf, prediction_store=store, settings=s)
    assert out["surface_source"].iloc[0] == "ml"


def test_osm_priority_env_does_not_allow_ml_over_osm(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "p.csv"
    _write_runtime_csv(csv_path, surface_pred_group="unpaved_soft")
    monkeypatch.setenv("SURFACE_AI_RUNTIME_ENABLED", "true")
    monkeypatch.setenv("SURFACE_AI_RUNTIME_PREDICTIONS_PATH", str(csv_path))
    s = Settings()
    store = SurfacePredictionStore(s)
    store.load()
    gdf = _minimal_edges_gdf()
    gdf["surface"] = ["asphalt"]
    out, _stats = apply_surface_resolution(gdf, prediction_store=store, settings=s)
    assert out["surface_source"].iloc[0] == "osm"
    assert out["surface_effective"].iloc[0] == "asphalt"


def test_graph_fingerprint_mismatch_disables_ml(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "p.csv"
    _write_runtime_csv(csv_path, graph_fingerprint="badbadbadbadbad0")
    monkeypatch.setenv("SURFACE_AI_RUNTIME_ENABLED", "true")
    monkeypatch.setenv("SURFACE_AI_RUNTIME_PREDICTIONS_PATH", str(csv_path))
    s = Settings()
    store = SurfacePredictionStore(s)
    store.load()
    assert store.loaded
    gdf = _minimal_edges_gdf()
    out, _stats = apply_surface_resolution(gdf, prediction_store=store, settings=s)
    assert out["surface_source"].iloc[0] != "ml"


def test_parse_runtime_json_graph_fingerprint() -> None:
    blob = json.dumps({"predict": {"graph_hash": "abc123def4567890"}, "train": {}})
    assert parse_runtime_artifact_graph_hash(blob) == "abc123def4567890"


