"""Статусы progressive-задачи — единый источник для backend и контрактов API."""

from __future__ import annotations

from enum import Enum


class AlternativesJobStatus(str, Enum):
    """Состояние записи POST /alternatives/start → GET /alternatives/job/{id}."""

    RUNNING = "running"
    PARTIAL = "partial"
    RUNNING_GREEN = "running_green"
    DONE = "done"
    FAILED = "failed"
