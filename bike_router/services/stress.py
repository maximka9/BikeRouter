"""Транспортный стресс: конфигурируемая политика LTS, сегмент и пересечение."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Optional, Tuple

from ..config import STRESS_COST_SCALE
from .policy_data import load_stress_policy

# ---------------------------------------------------------------------------
# Парсинг OSM
# ---------------------------------------------------------------------------


def _parse_lanes(val: Any) -> float:
    if val is None:
        return 1.0
    s = str(val).strip()
    if not s or s.lower() in ("none", "no", "n/a"):
        return 1.0
    parts = re.split(r"[|;]", s)
    nums: list[float] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        m = re.search(r"(\d+(?:\.\d+)?)", p)
        if m:
            try:
                nums.append(float(m.group(1)))
            except ValueError:
                continue
    if not nums:
        return 1.0
    return float(max(nums))


def _parse_maxspeed_kmh(val: Any) -> float:
    if val is None:
        return 30.0
    s = str(val).strip().lower()
    if not s or s in ("none", "walk", "signals"):
        return 30.0
    if "mph" in s:
        m = re.search(r"(\d+)", s)
        if m:
            return float(m.group(1)) * 1.60934
        return 30.0
    m = re.search(r"(\d+)", s)
    if m:
        return float(m.group(1))
    return 30.0


def _has_protected_bike(tags: Dict[str, Any]) -> bool:
    for k in (
        "cycleway",
        "cycleway:left",
        "cycleway:right",
        "cycleway:both",
        "bicycle_road",
    ):
        v = tags.get(k)
        if v is None:
            continue
        sv = str(v).lower()
        if sv in ("", "no", "none"):
            continue
        if "track" in sv or "lane" in sv or sv in ("yes", "designated", "separate"):
            return True
    if str(tags.get("segregated", "")).lower() == "yes":
        return True
    return False


def _has_sidewalk(tags: Dict[str, Any]) -> bool:
    for k in ("sidewalk", "sidewalk:both", "sidewalk:left", "sidewalk:right"):
        v = tags.get(k)
        if v is None:
            continue
        sv = str(v).lower()
        if sv and sv not in ("no", "none"):
            return True
    return False


def _policy_hw_base(hw: str, policy: Dict[str, Any]) -> float:
    h = (hw or "unclassified").lower()
    table = policy.get("highway_base") or {}
    if isinstance(table, dict) and h in table:
        return float(table[h])
    if isinstance(table, dict) and "default" in table:
        return float(table["default"])
    return 2.0


def _policy_speed_penalty(ms: float, policy: Dict[str, Any]) -> float:
    rows = policy.get("speed_penalty_kmh")
    if not isinstance(rows, list):
        return 0.0
    pen = 0.0
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        thr, p = float(row[0]), float(row[1])
        if ms >= thr:
            pen = max(pen, p)
    return pen


def lts_from_osm_tags(
    tags: Dict[str, Any],
    policy: Optional[Dict[str, Any]] = None,
) -> float:
    """Уровень стресса сегмента 1..4 (конфигурируется stress_policy.json)."""
    pol = policy if policy is not None else load_stress_policy()
    if not pol:
        pol = {}
    hw = str(tags.get("highway") or "unclassified").lower()
    if hw == "steps":
        return float((pol.get("highway_base") or {}).get("steps", 3.5))

    base = _policy_hw_base(hw, pol)
    lanes = _parse_lanes(tags.get("lanes"))
    ms = _parse_maxspeed_kmh(tags.get("maxspeed"))

    thr = int(pol.get("lane_threshold_high", 2))
    per = float(pol.get("lane_penalty_per_extra_lane", 0.12))
    lane_pen = max(0.0, lanes - thr) * per if lanes > thr else 0.0

    speed_pen = _policy_speed_penalty(ms, pol)

    protected = _has_protected_bike(tags)
    red = float(pol.get("protected_bike_reduction", 0.9))
    if protected:
        base = max(float(pol.get("lts_min", 1.0)), base - red)
        lane_pen *= 0.45
        speed_pen *= 0.55

    j = str(tags.get("junction") or "").lower()
    if j in ("roundabout", "circular"):
        base += float(pol.get("roundabout_bonus", 0.4))
    elif j in ("yes", "merge", "intersection"):
        base += float(pol.get("junction_complex_bonus", 0.25))

    has_sig = str(tags.get("traffic_signals") or "").lower() in (
        "yes",
        "signal",
    )
    has_cross = bool(tags.get("crossing"))
    if has_sig:
        base += float(pol.get("traffic_signals_bonus", 0.18))
    if has_cross:
        base += float(pol.get("crossing_bonus", 0.15))
        if has_sig:
            base += float(pol.get("crossing_with_signals_bonus", 0.1))

    # нет велоинфраструктуры на проезжей части
    if hw in ("primary", "secondary", "tertiary", "trunk", "unclassified"):
        if not protected and not _has_sidewalk(tags):
            base += float(pol.get("no_cycle_infrastructure_penalty", 0.45))
        elif not protected:
            base += float(pol.get("no_sidewalk_where_expected_penalty", 0.15))

    lmn = float(pol.get("lts_min", 1.0))
    lmx = float(pol.get("lts_max", 4.0))
    raw = base + lane_pen + speed_pen
    return float(max(lmn, min(lmx, raw)))


def intersection_stress_score(
    tags: Dict[str, Any],
    policy: Optional[Dict[str, Any]] = None,
) -> float:
    """Относительный стресс пересечения/узла [0..1]."""
    pol = policy if policy is not None else load_stress_policy()
    if not pol:
        pol = {}
    j = str(tags.get("junction") or "").lower()
    if j in ("roundabout", "circular"):
        return float(pol.get("intersection_roundabout", 0.85))
    has_sig = str(tags.get("traffic_signals") or "").lower() in (
        "yes",
        "signal",
    )
    has_cross = bool(tags.get("crossing"))
    if has_sig and has_cross:
        return float(pol.get("intersection_crossing_signals", 0.9))
    if has_sig:
        return float(pol.get("intersection_signals", 0.65))
    if has_cross:
        return float(pol.get("intersection_crossing_only", 0.45))
    return float(pol.get("intersection_default", 0.0))


def stress_costs_for_edge(
    length_m: float,
    lts: float,
    intersection_score: float,
    policy: Optional[Dict[str, Any]] = None,
    *,
    scale: float = STRESS_COST_SCALE,
) -> Tuple[float, float, float]:
    """Сегментный стресс, надбавка за пересечение, сумма."""
    pol = policy if policy is not None else load_stress_policy()
    int_scale = float(pol.get("intersection_cost_scale", 0.35)) if pol else 0.35
    seg = stress_cost_for_edge(length_m, lts, scale=scale)
    inter = float(scale * length_m * intersection_score * int_scale)
    return seg, inter, seg + inter


def stress_cost_for_edge(
    length_m: float,
    lts: float,
    *,
    scale: float = STRESS_COST_SCALE,
) -> float:
    if length_m <= 0 or not math.isfinite(length_m):
        return 0.0
    return float(scale * length_m * ((lts - 1.0) / 3.0))


def stress_metrics_for_route(
    lengths: list[float],
    lts_values: list[float],
    *,
    high_threshold: float = 2.5,
) -> Dict[str, float]:
    if not lengths:
        return {
            "avg_lts": 1.0,
            "max_lts": 1.0,
            "high_stress_length_m": 0.0,
            "high_stress_fraction": 0.0,
            "total_stress_cost": 0.0,
        }
    total_m = sum(lengths)
    tw = sum(l * lv for l, lv in zip(lengths, lts_values))
    high_m = sum(
        l for l, lv in zip(lengths, lts_values) if lv >= high_threshold
    )
    costs = [stress_cost_for_edge(l, lv) for l, lv in zip(lengths, lts_values)]
    return {
        "avg_lts": float(tw / total_m) if total_m > 0 else 1.0,
        "max_lts": float(max(lts_values) if lts_values else 1.0),
        "high_stress_length_m": float(high_m),
        "high_stress_fraction": float(high_m / total_m) if total_m > 0 else 0.0,
        "total_stress_cost": float(sum(costs)),
    }
