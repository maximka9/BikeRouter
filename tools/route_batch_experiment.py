"""Устарело: см. отдельные инструменты пакетных экспериментов.

Используйте вместо этого модуля:

- ``python -m bike_router.tools.route_variants_experiment`` — шесть вариантов
  маршрута при одном снимке погоды (Open-Meteo один раз в центре арены).
- ``python -m bike_router.tools.heat_weather_experiment`` — только ``heat``
  на synthetic-сетке (``--grid summer`` 36 кейсов или ``--grid winter`` 96).

Этот файл оставлен как тонкая обёртка на один релиз и завершает работу с кодом выхода 2.
"""

from __future__ import annotations

import sys


def main() -> None:
    msg = (
        "route_batch_experiment.py больше не используется.\n\n"
        "Вместо него:\n"
        "  python -m bike_router.tools.route_variants_experiment  # 6 вариантов, одна погода\n"
        "  python -m bike_router.tools.heat_weather_experiment     # heat; --grid summer|winter\n"
    )
    print(msg, file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
