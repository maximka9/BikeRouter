"""Пороги и поведение runtime Surface AI (не из .env)."""

from __future__ import annotations

# Safety / quality gates для применения ML к ребру
SURFACE_AI_RUNTIME_MIN_CONFIDENCE: float = 0.65
SURFACE_AI_RUNTIME_MIN_MARGIN: float = 0.15
SURFACE_AI_RUNTIME_PAVED_GOOD_MIN_CONFIDENCE: float = 0.90
SURFACE_AI_RUNTIME_USE_ONLY_SAFE: bool = True
SURFACE_AI_RUNTIME_FALLBACK_TO_HEURISTIC: bool = True
# Порядок ключей сопоставления строк CSV с геометрией графа
SURFACE_AI_RUNTIME_MATCH_BY: str = (
    "edge_id,undirected_edge_key,osm_way_id_geometry"
)
SURFACE_AI_RUNTIME_LOG_STATS: bool = True
