"""Unit-тесты retry/backoff."""

from __future__ import annotations

import pytest

from bike_router.services.retry import retry_call, sleep_backoff


def test_sleep_backoff_zero_jitter_monotonic() -> None:
    sleep_backoff(1, base_sec=0.1, max_sec=10.0, jitter=0.0)
    sleep_backoff(3, base_sec=0.1, max_sec=10.0, jitter=0.0)


def test_retry_call_succeeds_first_try() -> None:
    n = {"i": 0}

    def ok():
        n["i"] += 1
        return 42

    out = retry_call(
        ok,
        should_retry=lambda e: False,
        max_attempts=3,
        base_sec=0.01,
        max_sec=0.1,
        jitter=0.0,
        label="t",
    )
    assert out == 42
    assert n["i"] == 1


def test_retry_call_raises_after_exhausted() -> None:
    def boom():
        raise ConnectionError("x")

    with pytest.raises(ConnectionError):
        retry_call(
            boom,
            should_retry=lambda e: isinstance(e, ConnectionError),
            max_attempts=2,
            base_sec=0.01,
            max_sec=0.05,
            jitter=0.0,
            label="t",
        )
