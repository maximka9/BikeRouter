#!/bin/sh
# Запускается до Gunicorn: при необходимости догружает area_precache (идемпотентно).
set -e
export PYTHONPATH="${PYTHONPATH:-/app}"

if [ "${PRECACHE_AREA_ENSURE_BEFORE_API:-false}" = "true" ]; then
  echo "docker-entrypoint: проверка area_precache (PRECACHE_AREA_ENSURE_BEFORE_API=true)…"
  python -m bike_router.tools.ensure_precache || exit $?
fi

exec "$@"
