"""Логирование HTTP-запросов: метод, путь, статус, время ответа."""

from __future__ import annotations

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("bike_router.request")


class RequestLogMiddleware(BaseHTTPMiddleware):
    """Пишет в лог одну строку на каждый завершённый запрос."""

    async def dispatch(self, request: Request, call_next) -> Response:
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.exception(
                "request_error method=%s path=%s duration_ms=%.2f",
                request.method,
                request.url.path,
                elapsed_ms,
            )
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "request method=%s path=%s status=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response
