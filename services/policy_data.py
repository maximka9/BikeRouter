"""Загрузка JSON-политик тепла/сезона и стресса (редактируются без смены кода)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

_POLICIES_DIR = Path(__file__).resolve().parent.parent / "policies"


def _load_json(name: str) -> Dict[str, Any]:
    p = _POLICIES_DIR / name
    if not p.is_file():
        logger.warning("Политика не найдена %s — встроенные значения", p)
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Не удалось прочитать %s: %s", p, exc)
        return {}


def load_heat_season_policy() -> Dict[str, Any]:
    return _load_json("heat_season.json")


def load_stress_policy() -> Dict[str, Any]:
    return _load_json("stress_policy.json")
