"""
Точка входа для локального запуска веб-сервера.

    cd NIR
    python -m bike_router

По умолчанию слушает ``0.0.0.0:8000`` — удобно открыть сайт с телефона в той же Wi‑Fi
(в браузере ``http://<IPv4_ПК>:8000``). Только этот компьютер: ``BIKE_ROUTER_HOST=127.0.0.1``.
"""

from __future__ import annotations

import os

from .logutil import configure_root_logging


def main() -> None:
    configure_root_logging()
    import uvicorn

    host = os.getenv("BIKE_ROUTER_HOST", "0.0.0.0").strip() or "0.0.0.0"
    port = int(os.getenv("BIKE_ROUTER_PORT", "8000").strip() or "8000")

    uvicorn.run(
        "bike_router.api:app",
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
