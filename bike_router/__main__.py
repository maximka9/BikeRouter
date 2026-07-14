"""
Точка входа для локального запуска веб-сервера.

    python -m bike_router

По умолчанию слушает ``127.0.0.1:8000``. Для доступа из сети задайте
``BIKE_ROUTER_HOST=0.0.0.0`` явно.
"""

from __future__ import annotations

import os

from .logutil import configure_root_logging


def main() -> None:
    configure_root_logging()
    import uvicorn

    host = os.getenv("BIKE_ROUTER_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.getenv("BIKE_ROUTER_PORT", "8000").strip() or "8000")

    uvicorn.run(
        "bike_router.api:app",
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
