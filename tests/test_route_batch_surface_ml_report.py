from __future__ import annotations

from types import SimpleNamespace

import networkx as nx

from bike_router.tools._experiment_common import (
    add_surface_ml_report_summary_means,
    call_with_route_surface_source_report,
    route_surface_source_report_metrics,
)


class _Route:
    def __init__(self, edges):
        self.edges = edges


def _graph_for_surface_report() -> tuple[nx.MultiDiGraph, _Route]:
    G = nx.MultiDiGraph()
    for n in range(1, 7):
        G.add_node(n)
    rows = [
        (1, 2, 0, 40.0, "asphalt", "asphalt", "osm"),
        (2, 3, 0, 20.0, None, "compacted", "ml"),
        (3, 4, 0, 20.0, "", "paved", "heuristic"),
        (4, 5, 0, 10.0, "", "unknown", "unknown"),
        (5, 6, 0, 10.0, "unknown", "unknown", "ml"),
    ]
    for u, v, key, length, raw, effective, source in rows:
        G.add_edge(
            u,
            v,
            key=key,
            length=length,
            surface_osm=raw,
            surface_effective=effective,
            surface_source=source,
        )
    return G, _Route([(u, v, key) for u, v, key, *_rest in rows])


def test_route_surface_source_report_metrics_are_length_weighted() -> None:
    G, route = _graph_for_surface_report()

    metrics = route_surface_source_report_metrics(G, route)

    assert metrics["osm_surface_missing_share_pct"] == 60.0
    assert metrics["surface_from_osm_share_pct"] == 40.0
    assert metrics["surface_from_ml_share_pct"] == 20.0
    assert metrics["surface_from_heuristic_share_pct"] == 20.0
    assert metrics["surface_unknown_after_ml_share_pct"] == 20.0

    checksum = (
        metrics["surface_from_osm_share_pct"]
        + metrics["surface_from_ml_share_pct"]
        + metrics["surface_from_heuristic_share_pct"]
        + metrics["surface_unknown_after_ml_share_pct"]
    )
    assert abs(checksum - 100.0) <= 0.05
    assert metrics["surface_from_ml_share_pct"] <= metrics["osm_surface_missing_share_pct"]
    assert (
        metrics["surface_unknown_after_ml_share_pct"]
        <= metrics["osm_surface_missing_share_pct"]
    )


def test_call_with_route_surface_source_report_captures_engine_route() -> None:
    G, route = _graph_for_surface_report()

    class Engine:
        graph = G

        def _build_route_response(
            self,
            start,
            end,
            profile_key,
            mode,
            route_result,
            cost_weight_key,
            *,
            graph=None,
        ):
            return SimpleNamespace(mode=mode, route_result=route_result)

    engine = Engine()

    def call():
        return engine._build_route_response(
            (0.0, 0.0),
            (1.0, 1.0),
            "cyclist",
            "full",
            route,
            "weight_cyclist_full",
            graph=G,
        )

    result, report_by_mode = call_with_route_surface_source_report(engine, call)

    assert result.mode == "full"
    assert report_by_mode["full"]["surface_from_osm_share_pct"] == 40.0


def test_surface_ml_report_summary_means_group_by_profile_and_variant() -> None:
    raw_rows = [
        {
            "profile": "cyclist",
            "variant_key": "full",
            "osm_surface_missing_share_pct": 60.0,
            "surface_from_osm_share_pct": 40.0,
            "surface_from_ml_share_pct": 20.0,
            "surface_from_heuristic_share_pct": 20.0,
            "surface_unknown_after_ml_share_pct": 20.0,
        },
        {
            "profile": "cyclist",
            "variant_key": "full",
            "osm_surface_missing_share_pct": 20.0,
            "surface_from_osm_share_pct": 80.0,
            "surface_from_ml_share_pct": 10.0,
            "surface_from_heuristic_share_pct": 5.0,
            "surface_unknown_after_ml_share_pct": 5.0,
        },
    ]
    summary = [{"profile": "cyclist", "variant_key": "full"}]

    add_surface_ml_report_summary_means(summary, raw_rows)

    assert summary[0]["mean_osm_surface_missing_share_pct"] == 40.0
    assert summary[0]["mean_surface_from_osm_share_pct"] == 60.0
    assert summary[0]["mean_surface_from_ml_share_pct"] == 15.0
    assert summary[0]["mean_surface_from_heuristic_share_pct"] == 12.5
    assert summary[0]["mean_surface_unknown_after_ml_share_pct"] == 12.5
