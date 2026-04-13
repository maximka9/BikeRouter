# Сборка из каталога bike_router (контекст = этот каталог):
#   docker build -t bike-router .
#
# Кэш: монтируйте хост ./data → /data (BIKE_ROUTER_BASE_DIR), см. docker-compose.yml.

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    BIKE_ROUTER_BASE_DIR=/data

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app/bike_router

RUN mkdir -p /data/cache /data/osmnx_cache

EXPOSE 8000

# Один воркер: граф OSM в памяти на процесс; при >1 — копия графа на каждый воркер.
CMD ["gunicorn", "bike_router.api:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-w", "1", \
     "-b", "0.0.0.0:8000", \
     "--timeout", "600", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
