"""Фоновые задачи progressive POST /alternatives/start: in-memory хранилище по job_id."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..job_status import AlternativesJobStatus
from ..metrics import inc_green_job_failed

logger = logging.getLogger(__name__)


def new_job_id() -> str:
    """Идентификатор задачи; не привязан к классу store (удобно для api.py)."""
    return str(uuid.uuid4())


@dataclass
class JobRecord:
    job_id: str
    status: str
    routes: List[Dict[str, Any]] = field(default_factory=list)
    pending: List[str] = field(default_factory=list)
    error: Optional[Dict[str, Any]] = None
    green_warning: Optional[str] = None
    """Сериализованный criteria_bundle (JSON-маршруты по ключам критериев)."""
    criteria_bundle: Optional[Dict[str, List[Dict[str, Any]]]] = None
    created_at: float = field(default_factory=time.time)
    mut: threading.Lock = field(default_factory=threading.Lock)


class AlternativesJobStore:
    """Изоляция по job_id; TTL; thread-safe."""

    def __init__(self, ttl_sec: float = 1800.0) -> None:
        self._ttl = float(ttl_sec)
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def _purge_stale_unlocked(self) -> None:
        now = time.time()
        dead = [
            jid
            for jid, rec in self._jobs.items()
            if now - rec.created_at > self._ttl
        ]
        for jid in dead:
            del self._jobs[jid]
            logger.info("alternatives_job expired job_id=%s", jid)

    def create(self, job_id: str, record: JobRecord) -> None:
        with self._lock:
            self._purge_stale_unlocked()
            self._jobs[job_id] = record

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            self._purge_stale_unlocked()
            return self._jobs.get(job_id)

    def complete_green_success(self, job_id: str, route_dump: Dict[str, Any]) -> None:
        rec = self.get(job_id)
        if rec is None:
            return
        with rec.mut:
            rec.routes.append(route_dump)
            rec.pending = []
            rec.status = AlternativesJobStatus.DONE.value
            rec.error = None

    def complete_green_failure(
        self,
        job_id: str,
        *,
        error: Dict[str, Any],
        green_warning: Optional[str] = None,
    ) -> None:
        rec = self.get(job_id)
        if rec is None:
            return
        with rec.mut:
            rec.pending = []
            rec.status = AlternativesJobStatus.FAILED.value
            rec.error = error
            if green_warning:
                rec.green_warning = green_warning
        inc_green_job_failed()

    def finalize_progressive_job(
        self,
        job_id: str,
        *,
        routes: List[Dict[str, Any]],
        pending: List[str],
        status: str,
        error: Optional[Dict[str, Any]] = None,
        green_warning: Optional[str] = None,
    ) -> None:
        """Итог progressive 2.0: полный список маршрутов и статус (done / failed)."""
        with self._lock:
            rec = self._jobs.get(job_id)
        if rec is None:
            return
        with rec.mut:
            rec.routes = list(routes)
            rec.pending = list(pending)
            rec.status = status
            rec.error = error
            if green_warning is not None:
                rec.green_warning = green_warning
