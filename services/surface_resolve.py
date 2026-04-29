"""Восстановление покрытия рёбер: OSM → ML (опционально) → эвристика highway/tracktype → unknown."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd

from ..config import DEFAULT_COEFFICIENT, Settings
from .surface_runtime_constants import (
    SURFACE_AI_RUNTIME_FALLBACK_TO_HEURISTIC,
    SURFACE_AI_RUNTIME_LOG_STATS,
    SURFACE_AI_RUNTIME_MIN_CONFIDENCE,
    SURFACE_AI_RUNTIME_MIN_MARGIN,
    SURFACE_AI_RUNTIME_PAVED_GOOD_MIN_CONFIDENCE,
    SURFACE_AI_RUNTIME_USE_ONLY_SAFE,
)

if TYPE_CHECKING:
    from .surface_prediction_store import SurfacePrediction, SurfacePredictionStore

logger = logging.getLogger(__name__)

_TRACKTYPE_SURFACE = {
    "grade1": "paved",
    "grade2": "paved",
    "grade3": "compacted",
    "grade4": "unpaved",
    "grade5": "unpaved",
}

_HIGHWAY_SURFACE = {
    "construction": "compacted",
    "cycleway": "paved",
    "footway": "paved",
    "living_street": "paved",
    "path": "compacted",
    "pedestrian": "paved",
    "platform": "paved",
    "primary": "paved",
    "primary_link": "paved",
    "residential": "paved",
    "secondary": "paved",
    "secondary_link": "paved",
    "service": "paved",
    "steps": "paved",
    "tertiary": "paved",
    "tertiary_link": "paved",
    "track": "unpaved",
    "unclassified": "compacted",
}

# Группа Surface AI → строка для ``ModeProfile.surface`` (ключи CYCLIST/PEDESTRIAN).
ML_GROUP_TO_ROUTING_PROFILE_SURFACE: Dict[str, str] = {
    "paved_good": "asphalt",
    "paved_rough": "compacted",
    "unpaved_soft": "unpaved",
    "unknown": "unknown",
}
_ML_GROUP_TO_PROFILE_SURFACE = ML_GROUP_TO_ROUTING_PROFILE_SURFACE


def ml_group_to_profile_surface(group: str) -> str:
    """Строка ``surface_effective`` для профиля по группе Surface AI."""
    g = (group or "").strip().lower()
    return ML_GROUP_TO_ROUTING_PROFILE_SURFACE.get(g, "unknown")


def _osm_surface_blocks_ml(surface_osm: str) -> bool:
    """True, если в OSM есть осмысленное покрытие — ML не применяется (инвариант)."""
    s = (surface_osm or "").strip().lower()
    if not s:
        return False
    if s == "unknown":
        return False
    return True


def _first_value(val: Any) -> Any:
    if isinstance(val, list):
        return val[0] if val else None
    return val


def _norm_tag(val: Any) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    if isinstance(val, list):
        val = val[0] if val else None
    s = str(val).strip().lower()
    return s


def infer_surface_from_tracktype_highway(
    highway: Any, tracktype: Any
) -> Optional[str]:
    """Эвристика, если в OSM нет тега ``surface``."""
    tt = _norm_tag(tracktype)
    if tt in _TRACKTYPE_SURFACE:
        return _TRACKTYPE_SURFACE[tt]
    hw = _norm_tag(highway)
    if not hw:
        return None
    return _HIGHWAY_SURFACE.get(hw)


def resolve_surface_effective(
    surface_osm: str,
    highway: Any,
    tracktype: Any,
) -> str:
    """Итоговая метка для коэффициента profile.surface (нижний регистр)."""
    s = (surface_osm or "").strip().lower()
    if s:
        return s

    inf = infer_surface_from_tracktype_highway(highway, tracktype)
    if inf:
        return inf

    return "unknown"


def build_surface_effective_column(edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Добавить ``surface_osm`` (нормализованный тег OSM или пусто)."""
    gdf = edges_gdf.copy()
    if "surface" in gdf.columns:
        gdf["surface_osm"] = gdf["surface"].map(
            lambda x: _norm_tag(_first_value(x))
        )
    else:
        gdf["surface_osm"] = ""

    return gdf


def _ensure_edge_id_column(gdf: gpd.GeoDataFrame) -> None:
    if "edge_id" in gdf.columns and gdf["edge_id"].astype(str).str.len().gt(0).any():
        return
    if not all(c in gdf.columns for c in ("u", "v", "key")):
        return
    gdf["edge_id"] = (
        pd.to_numeric(gdf["u"], errors="coerce").fillna(-1).astype(int).astype(str)
        + "_"
        + pd.to_numeric(gdf["v"], errors="coerce").fillna(-1).astype(int).astype(str)
        + "_"
        + pd.to_numeric(gdf["key"], errors="coerce").fillna(0).astype(int).astype(str)
    )


def _ml_surface_string(pred: "SurfacePrediction") -> str:
    """Строка surface_effective для весов (ключ словаря profile.surface)."""
    if pred.surface_effective_ml:
        s = pred.surface_effective_ml.strip().lower()
        if s:
            return s
    g = pred.surface_group
    if pred.surface_concrete:
        c = pred.surface_concrete.strip().lower()
        if c:
            return c
    return ml_group_to_profile_surface(g)


@dataclass
class SurfaceResolutionStats:
    """Счётчики после :func:`apply_surface_resolution` (граф целиком)."""

    edge_count: int = 0
    surface_source_osm_count: int = 0
    surface_source_ml_count: int = 0
    surface_source_heuristic_count: int = 0
    surface_source_unknown_count: int = 0
    ml_predictions_available_count: int = 0
    ml_predictions_used_count: int = 0
    ml_predictions_rejected_low_confidence: int = 0
    ml_predictions_rejected_low_margin: int = 0
    ml_predictions_rejected_paved_good_conf: int = 0
    ml_predictions_rejected_unsafe: int = 0
    ml_predictions_rejected_missing_match: int = 0
    ml_confidence_sum_used: float = 0.0

    def to_api_summary(self) -> Dict[str, Any]:
        n = max(1, int(self.edge_count))
        used = int(self.ml_predictions_used_count)
        avg_conf = (
            float(self.ml_confidence_sum_used) / used if used > 0 else 0.0
        )
        rejected = (
            int(self.ml_predictions_rejected_low_confidence)
            + int(self.ml_predictions_rejected_low_margin)
            + int(self.ml_predictions_rejected_paved_good_conf)
            + int(self.ml_predictions_rejected_unsafe)
            + int(self.ml_predictions_rejected_missing_match)
        )
        return {
            "osm_share": round(self.surface_source_osm_count / n, 4),
            "ml_share": round(self.surface_source_ml_count / n, 4),
            "heuristic_share": round(self.surface_source_heuristic_count / n, 4),
            "unknown_share": round(self.surface_source_unknown_count / n, 4),
            "ml_avg_confidence": round(avg_conf, 4),
            "ml_rejected_edges": rejected,
        }


def apply_surface_resolution(
    edges_gdf: gpd.GeoDataFrame,
    *,
    prediction_store: Optional["SurfacePredictionStore"] = None,
    settings: Optional[Settings] = None,
) -> tuple[gpd.GeoDataFrame, SurfaceResolutionStats]:
    """Колонка ``surface_effective`` и диагностика источника покрытия.

    Приоритет: осмысленный OSM ``surface`` (не пустой и не ``unknown``) → ML →
    эвристика highway/tracktype → unknown. Покрытие из OSM никогда не перезаписывается ML.
    """
    stats = SurfaceResolutionStats(edge_count=len(edges_gdf))
    gdf = build_surface_effective_column(edges_gdf)
    _ensure_edge_id_column(gdf)

    if prediction_store is not None:
        prediction_store.begin_graph_session(gdf)

    s = settings
    use_ml = (
        s is not None
        and bool(s.surface_ai_runtime_enabled)
        and prediction_store is not None
        and prediction_store.loaded
    )
    if s is not None and bool(s.surface_ai_runtime_enabled):
        if use_ml:
            logger.info(
                "surface_resolution: use_ml=True "
                "(SURFACE_AI_RUNTIME_ENABLED=True, SurfacePredictionStore.loaded=True)"
            )
        else:
            reason = "prediction_store_missing"
            if prediction_store is not None and not prediction_store.loaded:
                reason = (
                    getattr(prediction_store, "failure_reason", None)
                    or "store_not_loaded"
                )
            logger.warning(
                "surface_resolution: use_ml=False "
                "(SURFACE_AI_RUNTIME_ENABLED=True, reason=%s)",
                reason,
            )

    n = len(gdf)
    surf_eff: List[str] = []
    surf_src: List[str] = []
    ml_grp: List[str] = []
    ml_conc: List[str] = []
    ml_conf: List[float] = []
    ml_marg: List[float] = []
    ml_rej: List[str] = []
    res_reason: List[str] = []

    for idx in range(n):
        row = gdf.iloc[idx]
        surface_osm = str(row.get("surface_osm") or "")
        highway = row.get("highway")
        tracktype = row.get("tracktype")

        if s and _osm_surface_blocks_ml(surface_osm):
            eff = resolve_surface_effective(surface_osm, highway, tracktype)
            surf_eff.append(eff)
            surf_src.append("osm")
            ml_grp.append("")
            ml_conc.append("")
            ml_conf.append(float("nan"))
            ml_marg.append(float("nan"))
            ml_rej.append("")
            res_reason.append("osm")
            stats.surface_source_osm_count += 1
            continue

        pred: Optional["SurfacePrediction"] = None
        if use_ml:
            assert prediction_store is not None
            pred = prediction_store.get_for_edge(row)
            if pred is not None:
                stats.ml_predictions_available_count += 1

        ml_accepted = False
        reject_reason: Optional[str] = None

        if use_ml and pred is not None:
            if float(pred.confidence) < float(SURFACE_AI_RUNTIME_MIN_CONFIDENCE):
                reject_reason = "low_confidence"
                stats.ml_predictions_rejected_low_confidence += 1
            elif float(pred.margin) < float(SURFACE_AI_RUNTIME_MIN_MARGIN):
                reject_reason = "low_margin"
                stats.ml_predictions_rejected_low_margin += 1
            elif SURFACE_AI_RUNTIME_USE_ONLY_SAFE and not pred.is_safe:
                reject_reason = "unsafe"
                stats.ml_predictions_rejected_unsafe += 1
            elif (
                pred.surface_group == "paved_good"
                and float(pred.confidence)
                < float(SURFACE_AI_RUNTIME_PAVED_GOOD_MIN_CONFIDENCE)
            ):
                reject_reason = "paved_good_low_confidence"
                stats.ml_predictions_rejected_paved_good_conf += 1
            else:
                ml_accepted = True

        if ml_accepted and pred is not None:
            eff = _ml_surface_string(pred)
            surf_eff.append(eff)
            surf_src.append("ml")
            ml_grp.append(pred.surface_group)
            ml_conc.append(pred.surface_concrete or "")
            ml_conf.append(float(pred.confidence))
            ml_marg.append(float(pred.margin))
            ml_rej.append(pred.reject_reason or "")
            res_reason.append("ml_accepted")
            stats.surface_source_ml_count += 1
            stats.ml_predictions_used_count += 1
            stats.ml_confidence_sum_used += float(pred.confidence)
            continue

        if use_ml and pred is None:
            stats.ml_predictions_rejected_missing_match += 1

        ml_grp.append(pred.surface_group if pred is not None else "")
        ml_conc.append((pred.surface_concrete or "") if pred is not None else "")
        ml_conf.append(float(pred.confidence) if pred is not None else float("nan"))
        ml_marg.append(float(pred.margin) if pred is not None else float("nan"))
        if reject_reason:
            ml_rej.append(reject_reason)
        elif pred is not None and pred.reject_reason:
            ml_rej.append(str(pred.reject_reason))
        elif use_ml and pred is None:
            ml_rej.append("no_ml_row")
        else:
            ml_rej.append("")

        if (
            s is not None
            and SURFACE_AI_RUNTIME_FALLBACK_TO_HEURISTIC
            and not _osm_surface_blocks_ml(surface_osm)
        ):
            inf = infer_surface_from_tracktype_highway(highway, tracktype)
            if inf:
                surf_eff.append(inf)
                surf_src.append("heuristic")
                res_reason.append(
                    f"heuristic_after_ml_reject:{reject_reason}"
                    if reject_reason
                    else "heuristic"
                )
                stats.surface_source_heuristic_count += 1
                continue

        surf_eff.append("unknown")
        surf_src.append("unknown")
        res_reason.append(
            f"unknown_after_ml_reject:{reject_reason}" if reject_reason else "unknown"
        )
        stats.surface_source_unknown_count += 1

    gdf["surface_effective"] = surf_eff
    gdf["surface_source"] = surf_src
    gdf["surface_ml_group"] = ml_grp
    gdf["surface_ml_concrete"] = ml_conc
    gdf["surface_ml_confidence"] = ml_conf
    gdf["surface_ml_margin"] = ml_marg
    gdf["surface_ml_reject_reason"] = ml_rej
    gdf["surface_resolution_reason"] = res_reason

    if s and SURFACE_AI_RUNTIME_LOG_STATS:
        n2 = max(1, n)
        logger.info(
            "surface_resolution: osm=%.1f%% ml=%.1f%% heur=%.1f%% unk=%.1f%% | "
            "ml_used=%d ml_avail=%d rej(conf/margin/pgood/unsafe/miss)=%d/%d/%d/%d/%d",
            100 * stats.surface_source_osm_count / n2,
            100 * stats.surface_source_ml_count / n2,
            100 * stats.surface_source_heuristic_count / n2,
            100 * stats.surface_source_unknown_count / n2,
            stats.ml_predictions_used_count,
            stats.ml_predictions_available_count,
            stats.ml_predictions_rejected_low_confidence,
            stats.ml_predictions_rejected_low_margin,
            stats.ml_predictions_rejected_paved_good_conf,
            stats.ml_predictions_rejected_unsafe,
            stats.ml_predictions_rejected_missing_match,
        )

    return gdf, stats
