"""Загрузка и выдача ML-предсказаний покрытия для runtime-маршрутизации."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from ..config import Settings

logger = logging.getLogger(__name__)

RUNTIME_PREDICTIONS_SCHEMA_VERSION = "1"

# Минимум для сопоставления и safety; остальные колонки опциональны.
REQUIRED_RUNTIME_COLUMNS: Tuple[str, ...] = (
    "edge_id",
    "surface_pred_group",
    "surface_pred_confidence",
    "surface_pred_margin",
    "surface_ml_safe",
)

ROUTING_GROUPS = frozenset({"paved_good", "paved_rough", "unpaved_soft", "unknown"})


def _short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:16]


def compute_runtime_routing_graph_hash(edges: Any) -> str:
    """Совпадает с ``graph_hash`` из ``surface_ai.graph_fingerprint`` для того же набора рёбер."""
    if edges is None or getattr(edges, "empty", True):
        return "empty"
    try:
        u = edges.get("u", pd.Series([], dtype=object)).astype(str)
        v = edges.get("v", pd.Series([], dtype=object)).astype(str)
        eids = edges.get("edge_id", pd.Series([], dtype=object)).astype(str)
        payload = "|".join(sorted(eids.head(10000).tolist()))
        return _short_hash(payload)
    except Exception:
        return "empty"


def parse_runtime_artifact_graph_hash(raw: Any) -> str:
    """Из ячейки CSV: короткий hash или JSON с вложенным ``predict.graph_hash``."""
    s = str(raw or "").strip()
    if not s:
        return ""
    if s.startswith("{"):
        try:
            d = json.loads(s)
            if isinstance(d, dict):
                pred = d.get("predict")
                if isinstance(pred, dict) and pred.get("graph_hash") is not None:
                    return str(pred["graph_hash"]).strip()
                if d.get("graph_hash") is not None and d.get("edges_count") is not None:
                    return str(d["graph_hash"]).strip()
        except (json.JSONDecodeError, TypeError, ValueError):
            return s
    return s


def _norm_group(g: Any) -> str:
    s = str(g or "").strip().lower()
    if s in ROUTING_GROUPS:
        return s
    if s in {"rough_or_unpaved", "unpaved_hard"}:
        return "unpaved_soft"
    if s == "paved":
        return "paved_good"
    return "unknown"


def _rounded_geometry_hash(geometry: Any, precision: int = 6) -> str:
    """Стабильный хеш линии (как в surface_ai.graph_fingerprint)."""
    if geometry is None:
        return ""
    try:
        from shapely.geometry.base import BaseGeometry

        if not isinstance(geometry, BaseGeometry):
            return ""
    except Exception:
        return ""
    try:
        g = geometry
        if g.geom_type == "LineString":
            coords = list(g.coords)
        elif g.geom_type == "MultiLineString":
            coords = []
            for part in g.geoms:
                coords.extend(list(part.coords))
        else:
            return ""
        if not coords:
            return ""
        parts = []
        for x, y in coords:
            parts.append(f"{round(float(x), precision):.{precision}f},{round(float(y), precision):.{precision}f}")
        body = "|".join(parts)
        import hashlib

        return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return ""


def _first_osm_scalar(val: Any) -> Any:
    if isinstance(val, list):
        return val[0] if val else None
    return val


@dataclass(frozen=True)
class SurfacePrediction:
    edge_id: str
    surface_group: str
    surface_concrete: Optional[str]
    confidence: float
    margin: float
    source_model: str
    artifact_id: str
    is_safe: bool
    reject_reason: Optional[str] = None
    surface_effective_ml: Optional[str] = None


class SurfacePredictionStore:
    """Читает runtime_predictions.csv; индексирует по edge_id / undirected / way+geom."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._loaded = False
        self._failure_reason: Optional[str] = None
        self._artifact_graph_hash: str = ""
        self._ml_graph_ok: bool = True
        self._by_edge_id: Dict[str, SurfacePrediction] = {}
        self._by_undirected: Dict[str, SurfacePrediction] = {}
        self._by_way_geom: Dict[str, SurfacePrediction] = {}

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def failure_reason(self) -> Optional[str]:
        return self._failure_reason

    def load(self) -> None:
        self._loaded = False
        self._failure_reason = None
        self._artifact_graph_hash = ""
        self._ml_graph_ok = True
        self._by_edge_id.clear()
        self._by_undirected.clear()
        self._by_way_geom.clear()

        if not self._settings.surface_ai_runtime_enabled:
            return

        path: Path = self._settings.surface_ai_runtime_predictions_resolved_path
        if not path.is_file():
            msg = f"runtime predictions file missing: {path}"
            self._failure_reason = msg
            if self._settings.surface_ai_runtime_strict:
                raise FileNotFoundError(msg)
            logger.warning("%s — маршрутизация без ML", msg)
            return

        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
        except Exception as exc:
            msg = f"cannot read runtime predictions: {exc}"
            self._failure_reason = msg
            if self._settings.surface_ai_runtime_strict:
                raise RuntimeError(msg) from exc
            logger.warning("%s", msg)
            return

        if df.empty:
            msg = "runtime predictions CSV is empty"
            self._failure_reason = msg
            if self._settings.surface_ai_runtime_strict:
                raise ValueError(msg)
            logger.warning("%s", msg)
            return

        missing = [c for c in REQUIRED_RUNTIME_COLUMNS if c not in df.columns]
        if missing:
            msg = f"runtime predictions missing columns: {missing}"
            self._failure_reason = msg
            if self._settings.surface_ai_runtime_strict:
                raise ValueError(msg)
            logger.warning("%s", msg)
            return

        if "artifact_schema_version" in df.columns:
            bad = df["artifact_schema_version"].astype(str).str.strip()
            bad = bad[bad != ""]
            if not bad.empty and not bad.eq(RUNTIME_PREDICTIONS_SCHEMA_VERSION).all():
                msg = (
                    f"artifact_schema_version mismatch "
                    f"(expected {RUNTIME_PREDICTIONS_SCHEMA_VERSION})"
                )
                self._failure_reason = msg
                if self._settings.surface_ai_runtime_strict:
                    raise ValueError(msg)
                logger.warning("%s — отключаем ML", msg)
                return

        dup = df["edge_id"].astype(str).duplicated()
        if dup.any():
            msg = f"duplicate edge_id in runtime predictions ({int(dup.sum())} rows)"
            self._failure_reason = msg
            if self._settings.surface_ai_runtime_strict:
                raise ValueError(msg)
            logger.warning("%s — отключаем ML", msg)
            return

        area_fp = ""
        if "area_fingerprint" in df.columns and len(df):
            area_fp = str(df["area_fingerprint"].iloc[0] or "").strip()
        self._artifact_graph_hash = ""
        if "graph_fingerprint" in df.columns and len(df):
            gvals = df["graph_fingerprint"].astype(str).str.strip()
            gvals = gvals[gvals != ""]
            if not gvals.empty:
                uniq = gvals.unique()
                if len(uniq) > 1:
                    msg = "inconsistent graph_fingerprint values across CSV rows"
                    self._failure_reason = msg
                    if self._settings.surface_ai_runtime_strict:
                        raise ValueError(msg)
                    logger.warning("%s — отключаем ML", msg)
                    return
                self._artifact_graph_hash = parse_runtime_artifact_graph_hash(uniq[0])

        # Опциональная проверка отпечатков (если заданы и в Settings есть WKT полигона precache).
        if area_fp and self._settings.has_precache_area_polygon:
            from .area_graph_cache import parse_precache_polygon

            try:
                poly = parse_precache_polygon(self._settings)
                cur = hashlib.sha256(
                    getattr(poly, "wkt", str(poly)).encode("utf-8")
                ).hexdigest()[:24]
                if cur and area_fp != cur:
                    msg = f"area_fingerprint mismatch (csv={area_fp!r} current={cur!r})"
                    self._failure_reason = msg
                    if self._settings.surface_ai_runtime_strict:
                        raise ValueError(msg)
                    logger.warning("%s — отключаем ML", msg)
                    return
            except Exception as exc:
                msg = f"area_fingerprint check failed: {exc}"
                if self._settings.surface_ai_runtime_strict:
                    raise RuntimeError(msg) from exc
                logger.warning("%s", msg)
                return

        for _, row in df.iterrows():
            pred = self._row_to_prediction(row)
            eid = str(row["edge_id"]).strip()
            self._by_edge_id[eid] = pred
            u = row.get("u")
            v = row.get("v")
            k = row.get("key", 0)
            if u is not None and v is not None and str(u).strip() != "":
                try:
                    uu, vv, kk = int(u), int(v), int(float(k))
                    uk = self._undirected_key(uu, vv, kk)
                    self._by_undirected[uk] = pred
                except Exception:
                    pass
            ow = row.get("osm_way_id") or row.get("osmid") or row.get("way_id")
            gh = row.get("geometry_hash_rounded") or row.get("geometry_hash")
            if ow is not None and gh is not None and str(gh).strip():
                wgk = f"{int(float(str(ow).split('.')[0]))}|{str(gh).strip()}"
                self._by_way_geom[wgk] = pred

        self._loaded = True
        self._ml_graph_ok = True
        logger.info(
            "Surface AI runtime: загружено %d предсказаний из %s",
            len(self._by_edge_id),
            path,
        )

    def begin_graph_session(self, edges_gdf: Any) -> None:
        """Сверка ``graph_fingerprint`` артефакта с текущим графом (вызывать перед выборкой ML по рёбрам)."""
        self._ml_graph_ok = True
        if not self._loaded:
            return
        exp = (self._artifact_graph_hash or "").strip()
        if not exp:
            return
        cur = compute_runtime_routing_graph_hash(edges_gdf)
        if cur == exp:
            return
        self._ml_graph_ok = False
        msg = f"graph_fingerprint mismatch (artifact={exp!r} current_graph={cur!r})"
        if self._settings.surface_ai_runtime_strict:
            raise ValueError(msg)
        logger.warning("%s — ML-предсказания отключены для этого графа", msg)

    @staticmethod
    def _undirected_key(u: Any, v: Any, key: Any) -> str:
        a, b = int(u), int(v)
        lo, hi = (a, b) if a <= b else (b, a)
        return f"{lo}_{hi}_{int(key)}"

    def _row_to_prediction(self, row: pd.Series) -> SurfacePrediction:
        conf = float(pd.to_numeric(row.get("surface_pred_confidence"), errors="coerce") or 0.0)
        margin = float(pd.to_numeric(row.get("surface_pred_margin"), errors="coerce") or 0.0)
        grp = _norm_group(row.get("surface_pred_group"))
        conc = row.get("surface_pred_concrete")
        conc_s = str(conc).strip() if conc is not None and str(conc).strip() != "" else None
        safe_raw = row.get("surface_ml_safe")
        safe = str(safe_raw).strip().lower() in ("true", "1", "yes")
        rej = row.get("surface_ml_reject_reason")
        rej_s = str(rej).strip() if rej is not None and str(rej).strip() != "" else None
        eff_ml = row.get("surface_effective_ml")
        eff_s = str(eff_ml).strip().lower() if eff_ml is not None and str(eff_ml).strip() != "" else None
        mk = str(row.get("model_key") or "").strip()
        mt = str(row.get("model_target") or "").strip()
        art = str(row.get("artifact_created_at") or row.get("artifact_id") or "").strip()
        return SurfacePrediction(
            edge_id=str(row["edge_id"]).strip(),
            surface_group=grp,
            surface_concrete=conc_s,
            confidence=conf,
            margin=margin,
            source_model=mk,
            artifact_id=art,
            is_safe=safe,
            reject_reason=rej_s,
            surface_effective_ml=eff_s,
        )

    def get_for_edge(self, row: pd.Series) -> Optional[SurfacePrediction]:
        if not self._loaded or not self._ml_graph_ok:
            return None
        modes = [
            m.strip().lower().replace(" ", "_")
            for m in (self._settings.surface_ai_runtime_match_by or "").split(",")
            if m.strip()
        ]
        for mode in modes:
            if mode in ("edge_id",):
                eid = str(row.get("edge_id") or "").strip()
                if eid and eid in self._by_edge_id:
                    return self._by_edge_id[eid]
            elif mode in ("undirected_edge_key", "undirected"):
                u, v, k = row.get("u"), row.get("v"), row.get("key", 0)
                if u is None or v is None:
                    continue
                try:
                    uk = self._undirected_key(int(u), int(v), int(float(k)))
                    if uk in self._by_undirected:
                        return self._by_undirected[uk]
                except Exception:
                    continue
            elif mode in ("osm_way_id_geometry", "osm_way_id+geometry", "way_geom"):
                ow = row.get("osmid") if "osmid" in row.index else row.get("osm_id")
                if ow is None:
                    ow = row.get("way_id")
                geom = row.get("geometry")
                gh = _rounded_geometry_hash(geom)
                if ow is None or not gh:
                    continue
                try:
                    wid = int(float(str(_first_osm_scalar(ow)).split(".")[0]))
                except Exception:
                    continue
                wgk = f"{wid}|{gh}"
                if wgk in self._by_way_geom:
                    return self._by_way_geom[wgk]
        return None


def write_runtime_predictions_csv(df: pd.DataFrame, path: Path) -> None:
    """Записать CSV для :class:`SurfacePredictionStore` (проверка обязательных колонок)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    miss = [c for c in REQUIRED_RUNTIME_COLUMNS if c not in df.columns]
    if miss:
        raise ValueError(f"runtime predictions: missing columns {miss}")
    out = df.copy()
    if "artifact_schema_version" not in out.columns:
        out["artifact_schema_version"] = RUNTIME_PREDICTIONS_SCHEMA_VERSION
    out.to_csv(path, index=False, encoding="utf-8")
