"""Повторные попытки HTTP/сетевых вызовов с экспоненциальной задержкой и jitter."""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable, List, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def sleep_backoff(
    attempt: int,
    *,
    base_sec: float,
    max_sec: float,
    jitter: float,
) -> None:
    """Пауза перед попыткой ``attempt`` (1-based)."""
    exp = min(max_sec, base_sec * (2 ** max(0, attempt - 1)))
    if jitter > 0 and exp > 0:
        lo = max(0.0, exp * (1.0 - jitter))
        hi = exp * (1.0 + jitter)
        delay = lo + random.random() * (hi - lo)
    else:
        delay = exp
    time.sleep(delay)


def retry_call(
    fn: Callable[[], T],
    *,
    should_retry: Callable[[Exception], bool],
    max_attempts: int,
    base_sec: float,
    max_sec: float,
    jitter: float,
    label: str = "call",
) -> T:
    """Выполнить ``fn`` до ``max_attempts`` раз при временных ошибках."""
    last: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last = exc
            if not should_retry(exc) or attempt >= max_attempts:
                raise
            logger.warning(
                "%s retry %d/%d after %s: %s",
                label,
                attempt,
                max_attempts,
                type(exc).__name__,
                exc,
            )
            sleep_backoff(attempt, base_sec=base_sec, max_sec=max_sec, jitter=jitter)
    assert last is not None
    raise last


def _exception_chain(exc: BaseException) -> List[Any]:
    """Цепочка __cause__ / __context__ без зацикливания."""
    out: List[Any] = []
    seen: set[int] = set()
    e: Optional[BaseException] = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        out.append(e)
        e = e.__cause__ or e.__context__
    return out


def is_dns_resolution_failure(exc: BaseException) -> bool:
    """getaddrinfo / «Failed to resolve» — повтор к тому же хосту обычно бесполезен."""
    for c in _exception_chain(exc):
        msg = str(c).lower()
        tname = type(c).__name__.lower()
        if "gaierror" in tname or "resolution" in tname:
            return True
        if any(
            p in msg
            for p in (
                "failed to resolve",
                "name resolution",
                "temporary failure in name resolution",
                "name or service not known",
                "nodename nor servname",
            )
        ):
            return True
    return False


def is_transient_http(exc: Exception) -> bool:
    """requests / сеть / 5xx / таймаут."""
    import requests

    if is_dns_resolution_failure(exc):
        return False
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code >= 500
    if isinstance(exc, requests.ChunkedEncodingError):
        return True
    return False
