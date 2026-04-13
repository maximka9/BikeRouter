"""Единая настройка логирования (избегаем повторного basicConfig)."""

from __future__ import annotations

import logging
import sys


def configure_root_logging() -> None:
    """UTF-8 для консоли Windows и один раз basicConfig на корневой логгер."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
