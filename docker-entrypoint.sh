#!/bin/sh
# Только запуск процесса (Gunicorn). Предсборка полигона — офлайн: precache_area.
set -e

exec "$@"
