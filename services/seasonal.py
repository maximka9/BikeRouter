"""Сезонные профили маршрутизации (зелень, тень, зимняя активность снега).

Календарь по умолчанию — север умеренных широт; даты и коэффициенты задаются в Settings / .env.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Optional, Tuple

__all__ = (
    "parse_route_calendar_date",
    "routing_season_label",
    "season_green_route_multiplier",
    "season_tree_heat_route_multiplier",
    "snow_route_model_strength",
)


def parse_route_calendar_date(reference_iso: Optional[str]) -> Optional[date]:
    """Дата календаря из ISO (для сезона и плавного включения зелени)."""
    if not reference_iso:
        return None
    raw = str(reference_iso).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return dt.date()


def routing_season_label(d: date, s: Any) -> str:
    """Метка сезона маршрутизации (4 режима + короткий ramp в апреле)."""
    m, dd = d.month, d.day
    ramp_s = int(getattr(s, "season_green_ramp_start_day", 10))
    ramp_e = int(getattr(s, "season_green_ramp_end_day", 20))
    early_end = int(getattr(s, "season_early_spring_end_day", 9))

    if m in (11, 12, 1, 2) or m == 3:
        return "winter"
    if m == 4 and dd <= early_end:
        return "early_spring"
    if m == 4 and ramp_s <= dd <= ramp_e:
        return "spring_ramp"
    if (m == 4 and dd > ramp_e) or m in (5, 6, 7, 8, 9) or (m == 10 and dd <= 15):
        return "green_season"
    if m == 10 and dd >= 16:
        return "late_autumn"
    return "winter"


def season_green_route_multiplier(d: date, s: Any) -> float:
    """Множитель вклада зелёного бонуса к физике на рёбрах [0..1]."""
    m, dd = d.month, d.day
    ramp_s = int(getattr(s, "season_green_ramp_start_day", 10))
    ramp_e = int(getattr(s, "season_green_ramp_end_day", 20))
    early_end = int(getattr(s, "season_early_spring_end_day", 9))
    lo = float(getattr(s, "season_green_ramp_start_mult", 0.18))
    hi = float(getattr(s, "season_green_ramp_end_mult", 1.0))

    if m == 4 and ramp_s <= dd <= ramp_e:
        span = max(1, ramp_e - ramp_s)
        t = (dd - ramp_s) / float(span)
        return float(lo + (hi - lo) * t)

    lab = routing_season_label(d, s)
    return float(
        {
            "winter": getattr(s, "season_green_mult_winter", 0.08),
            "early_spring": getattr(s, "season_green_mult_early_spring", 0.18),
            "spring_ramp": lo,
            "green_season": getattr(s, "season_green_mult_green", 1.0),
            "late_autumn": getattr(s, "season_green_mult_late_autumn", 0.42),
        }.get(lab, 1.0)
    )


def season_tree_heat_route_multiplier(season: str, s: Any) -> float:
    """Множитель роли тени деревьев в непрерывной heat-модели на рёбрах."""
    key = (season or "green_season").strip().lower()
    return float(
        {
            "winter": getattr(s, "season_tree_heat_mult_winter", 0.12),
            "early_spring": getattr(s, "season_tree_heat_mult_early_spring", 0.28),
            "spring_ramp": getattr(s, "season_tree_heat_mult_spring_ramp", 0.55),
            "green_season": getattr(s, "season_tree_heat_mult_green", 1.0),
            "late_autumn": getattr(s, "season_tree_heat_mult_late_autumn", 0.45),
        }.get(key, 1.0)
    )


def _snap_snow_depth_m(snap: Any) -> float:
    return max(0.0, float(getattr(snap, "snow_depth_m", 0.0) or 0.0))


def _snap_snowfall_cm_h(snap: Any) -> float:
    return max(0.0, float(getattr(snap, "snowfall_cm_h", 0.0) or 0.0))


def snow_route_model_strength(season: str, snap: Any, s: Any) -> float:
    """0..1: насколько включать зимние snow/ветер/open-поправки."""
    sd = _snap_snow_depth_m(snap)
    sf = _snap_snowfall_cm_h(snap)
    se = (season or "").strip().lower()
    floor_green = float(getattr(s, "snow_green_season_strength_floor", 0.06))
    d_on = float(getattr(s, "snow_green_season_depth_on_m", 0.02))
    f_on = float(getattr(s, "snow_green_season_fresh_on_cm_h", 0.25))

    if se == "green_season":
        if sd < d_on and sf < f_on:
            return floor_green
        # заметный снег вне сезона — поднять силу модели
        return float(
            min(
                1.0,
                floor_green
                + 0.55 * min(1.0, sd / max(d_on * 5, 1e-6))
                + 0.35 * min(1.0, sf / max(f_on * 4, 1e-6)),
            )
        )

    if se in ("winter", "early_spring", "late_autumn"):
        return 1.0
    if se == "spring_ramp":
        return float(getattr(s, "snow_spring_ramp_strength", 0.88))
    return float(getattr(s, "snow_default_strength", 0.9))


def snow_depth_phys_multiplier(depth_m: float, s: Any) -> float:
    """Ступени по глубине снега на земле (м) — множитель к физике."""
    d = max(0.0, float(depth_m))
    t0 = float(getattr(s, "snow_depth_tier0_max_m", 0.02))
    t1 = float(getattr(s, "snow_depth_tier1_max_m", 0.05))
    t2 = float(getattr(s, "snow_depth_tier2_max_m", 0.10))
    m0 = float(getattr(s, "snow_depth_mult_tier0", 1.0))
    m1 = float(getattr(s, "snow_depth_mult_tier1", 1.06))
    m2 = float(getattr(s, "snow_depth_mult_tier2", 1.14))
    m3 = float(getattr(s, "snow_depth_mult_tier3", 1.28))
    if d <= t0:
        return m0
    if d <= t1:
        return m1
    if d <= t2:
        return m2
    return m3


def snow_fresh_phys_multiplier(snowfall_cm_h: float, s: Any) -> float:
    """Ступени по свежему снегу см/ч."""
    x = max(0.0, float(snowfall_cm_h))
    t0 = float(getattr(s, "snow_fresh_tier0_max_cm_h", 0.2))
    t1 = float(getattr(s, "snow_fresh_tier1_max_cm_h", 1.0))
    t2 = float(getattr(s, "snow_fresh_tier2_max_cm_h", 3.0))
    m0 = float(getattr(s, "snow_fresh_mult_tier0", 1.0))
    m1 = float(getattr(s, "snow_fresh_mult_tier1", 1.04))
    m2 = float(getattr(s, "snow_fresh_mult_tier2", 1.12))
    m3 = float(getattr(s, "snow_fresh_mult_tier3", 1.22))
    if x <= t0:
        return m0
    if x <= t1:
        return m1
    if x <= t2:
        return m2
    return m3


def normalized_snow_signals(snap: Any, s: Any) -> Tuple[float, float]:
    """Нормы [0..1] для снега (глубина, интенсивность снегопада)."""
    sd = _snap_snow_depth_m(snap)
    sf = _snap_snowfall_cm_h(snap)
    dref = max(float(getattr(s, "snow_depth_norm_ref_m", 0.25)), 1e-6)
    fref = max(float(getattr(s, "snow_fresh_norm_ref_cm_h", 4.0)), 1e-6)
    return (
        max(0.0, min(1.0, sd / dref)),
        max(0.0, min(1.0, sf / fref)),
    )
