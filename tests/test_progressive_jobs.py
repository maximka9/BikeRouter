"""Статусы progressive-задач (in-memory store)."""

from __future__ import annotations

from bike_router.job_status import AlternativesJobStatus
from bike_router.services.alternatives_jobs import AlternativesJobStore, JobRecord, new_job_id


def test_green_failure_sets_failed_and_error() -> None:
    st = AlternativesJobStore(ttl_sec=3600.0)
    jid = new_job_id()
    st.create(
        jid,
        JobRecord(
            job_id=jid,
            status=AlternativesJobStatus.RUNNING_GREEN.value,
            routes=[{"mode": "shortest"}],
            pending=["green"],
        ),
    )
    st.complete_green_failure(
        jid,
        error={"code": "GREEN_ROUTE_FAILED", "message": "test"},
        green_warning="Подсказка",
    )
    rec = st.get(jid)
    assert rec is not None
    assert rec.status == AlternativesJobStatus.FAILED.value
    assert rec.pending == []
    assert rec.error is not None
    assert rec.error["code"] == "GREEN_ROUTE_FAILED"
    assert rec.green_warning == "Подсказка"


def test_green_success_appends_route() -> None:
    st = AlternativesJobStore(ttl_sec=3600.0)
    jid = new_job_id()
    st.create(
        jid,
        JobRecord(
            job_id=jid,
            status=AlternativesJobStatus.RUNNING_GREEN.value,
            routes=[{"mode": "shortest"}],
            pending=["green"],
        ),
    )
    st.complete_green_success(jid, {"mode": "green", "x": 1})
    rec = st.get(jid)
    assert rec is not None
    assert rec.status == AlternativesJobStatus.DONE.value
    assert len(rec.routes) == 2
    assert rec.routes[-1]["mode"] == "green"
    assert rec.pending == []
