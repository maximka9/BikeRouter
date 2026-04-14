#!/usr/bin/env python3
"""Устарело: предсборка area precache только офлайн — ``python -m bike_router.tools.precache_area``."""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "ensure_precache удалён из основного контура. "
        "Соберите кэш до запуска API: python -m bike_router.tools.precache_area",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
