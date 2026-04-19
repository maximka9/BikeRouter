"""Скрипт сравнения режимов маршрутизации для отчёта / диплома.

Запуск (после ``python -m bike_router`` или из кода с загруженным движком)::

    python -m bike_router.tools.heat_stress_experiment

По умолчанию читает координаты из .env (START_LAT, …) и вызывает
``RouteEngine.compute_alternatives`` (единый ответ со всеми вариантами) по слотам времени.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List


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

    slots = ["morning", "noon", "evening"]

    rows: List[Dict[str, Any]] = []
    for slot in slots:
        try:
            out = eng.compute_alternatives(
                start,
                end,
                "cyclist",
                green_enabled=True,
                time_slot_override=slot,
            )
            for r in out.routes:
                hs = r.heat_stress
                rows.append(
                    {
                        "time_slot": slot,
                        "mode": r.mode,
                        "length_m": r.length_m,
                        "heat_total": hs.total_heat_cost if hs else None,
                        "avg_lts": hs.avg_stress_lts if hs else None,
                        "combined": hs.combined_cost if hs else None,
                        "turns": hs.turn_count if hs else None,
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "time_slot": slot,
                    "error": str(exc),
                }
            )

    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
