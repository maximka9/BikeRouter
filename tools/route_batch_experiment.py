"""Устарело: см. отдельные инструменты пакетных экспериментов.

Используйте вместо этого модуля:

- ``python -m bike_router.tools.route_variants_experiment`` — шесть вариантов
  маршрута при одном снимке погоды (Open-Meteo один раз в центре арены).
- ``python -m bike_router.tools.heat_weather_experiment`` — только ``heat``
  на сетке из 36 синтетических погод (без Open-Meteo).

Этот файл оставлен как тонкая обёртка на один релиз и завершает работу с кодом выхода 2.
"""

from __future__ import annotations

import sys


def main() -> None:
    msg = (
        "route_batch_experiment.py больше не используется.\n\n"
        "Вместо него:\n"
        "  python -m bike_router.tools.route_variants_experiment  # 6 вариантов, одна погода\n"
        "  python -m bike_router.tools.heat_weather_experiment     # только heat, 36 synthetic\n"
    )
    print(msg, file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
