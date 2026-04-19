"""Сравнение маршрутов при разных погодных сценариях (ручной режим).

Запуск::

    python -m bike_router.tools.weather_scenarios_experiment

Требует настроенный коридор в .env (START_LAT, …) и прогретый движок.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple


def _ensure_pkg_path() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)


def main() -> None:
    _ensure_pkg_path()
    from bike_router.engine import RouteEngine
    from bike_router.config import Settings

    s = Settings()
    start = (s.start_lat, s.start_lon)
    end = (s.end_lat, s.end_lon)
    eng = RouteEngine()
    eng.warmup()

    scenarios: List[Tuple[str, Dict[str, float]]] = [
        ("neutral", {"temperature_c": 18.0, "precipitation_mm": 0.0, "wind_speed_ms": 3.0, "cloud_cover_pct": 50.0, "humidity_pct": 55.0}),
        ("hot_sun", {"temperature_c": 34.0, "precipitation_mm": 0.0, "wind_speed_ms": 2.0, "cloud_cover_pct": 10.0, "humidity_pct": 40.0}),
        ("rain", {"temperature_c": 15.0, "precipitation_mm": 4.0, "wind_speed_ms": 5.0, "cloud_cover_pct": 90.0, "humidity_pct": 92.0}),
        ("wind", {"temperature_c": 12.0, "precipitation_mm": 0.0, "wind_speed_ms": 14.0, "cloud_cover_pct": 40.0, "humidity_pct": 65.0}),
    ]

    rows: List[Dict[str, Any]] = []
    for name, wx in scenarios:
        try:
            out = eng.compute_alternatives(
                start,
                end,
                "cyclist",
                green_enabled=True,
                time_slot_override="noon",
                weather_mode="manual",
                use_live_weather=False,
                **wx,
            )
            r0 = next(
                (r for r in out.routes if r.mode == "heat_stress"),
                out.routes[0] if out.routes else None,
            )
            hs = r0.heat_stress if r0 else None
            g = r0.green if r0 else None
            rows.append(
                {
                    "scenario": name,
                    "weather": wx,
                    "length_m": r0.length_m if r0 else None,
                    "combined_cost": hs.combined_cost if hs else None,
                    "physical_in_breakdown": hs.combined_breakdown.physical if hs and hs.combined_breakdown else None,
                    "green_pct": g.percent if g else None,
                    "high_stress_frac": hs.high_stress_length_fraction if hs else None,
                    "weather_multipliers": (
                        hs.combined_breakdown.weather_multipliers
                        if hs and hs.combined_breakdown
                        else None
                    ),
                }
            )
        except Exception as exc:
            rows.append({"scenario": name, "error": str(exc)})

    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
