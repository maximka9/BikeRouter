#!/usr/bin/env python3
"""Прогон сценариев POST /alternatives против запущенного API.

Использование (из каталога NIR, PYTHONPATH должен содержать корень репозитория)::

    python bike_router/tools/run_route_scenarios.py --base http://127.0.0.1:8000

Или внутри контейнера::

    python /app/bike_router/tools/run_route_scenarios.py --base http://127.0.0.1:8000

Выход: TSV в stdout (id, статус, код ошибки, число маршрутов, лестницы, na_surface %%).
Код выхода 1, если есть несовпадение с полем expect в JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

# Корень bike_router на PYTHONPATH — requests уже в requirements.txt проекта


def _load_scenarios() -> list:
    p = Path(__file__).resolve().parent / "route_scenarios.json"
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return data["scenarios"]


def _post_alternatives(
    base: str, start: dict, end: dict, profile: str, timeout: float
) -> tuple[int, dict | None]:
    url = base.rstrip("/") + "/alternatives"
    try:
        r = requests.post(
            url,
            json={"start": start, "end": end, "profile": profile},
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
    except requests.RequestException as e:
        print(f"FATAL: не удалось подключиться к {url}: {e}", file=sys.stderr)
        sys.exit(2)
    try:
        payload = r.json() if r.content else None
    except json.JSONDecodeError:
        payload = None
    return r.status_code, payload


def _detail_code(payload: dict | None) -> str | None:
    if not payload or "detail" not in payload:
        return None
    d = payload["detail"]
    if isinstance(d, dict):
        return d.get("code")
    return None


def _route_stairs_and_quality(r0: dict) -> tuple[float, int, str]:
    """Соответствует models.RouteResponse: stairs.*, quality_hints.*."""
    stairs = r0.get("stairs")
    if isinstance(stairs, dict):
        stairs_len = float(stairs.get("total_length_m") or 0)
        stairs_seg = int(stairs.get("count") or 0)
    else:
        stairs_len = float(r0.get("stairs_total_length_m") or 0)
        stairs_seg = int(r0.get("stairs_segments") or 0)

    qh = r0.get("quality_hints") or r0.get("quality") or {}
    na_pct = ""
    if isinstance(qh, dict):
        if qh.get("na_surface_fraction") is not None:
            na_pct = f"{float(qh['na_surface_fraction']) * 100:.1f}"
        elif qh.get("na_surface_pct") is not None:
            na_pct = f"{float(qh['na_surface_pct']):.1f}"
    return stairs_len, stairs_seg, na_pct


def _check_expect(expect: str, status: int, code: str | None, n_routes: int) -> bool:
    if expect == "any":
        return True
    if expect == "ok":
        return status == 200 and n_routes > 0
    if expect == "no_path":
        return status == 404 and code == "NO_PATH"
    if expect == "point_outside":
        return status == 422 and code == "POINT_OUTSIDE_ZONE"
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Проверка сценариев маршрутизации")
    ap.add_argument("--base", default="http://127.0.0.1:8000", help="Базовый URL API")
    ap.add_argument("--timeout", type=float, default=180.0, help="Таймаут запроса, сек")
    args = ap.parse_args()

    scenarios = _load_scenarios()
    print("id\tok\tstatus\tcode\tn_routes\tstairs_len_m\tstairs_seg\tna_surface_pct\texpect\tlabel")

    failed = 0
    for s in scenarios:
        sid = s["id"]
        st = s["start"]
        en = s["end"]
        profile = s["profile"]
        expect = s.get("expect", "any")
        label = s.get("label", "")

        status, payload = _post_alternatives(
            args.base,
            {"lat": st["lat"], "lon": st["lon"]},
            {"lat": en["lat"], "lon": en["lon"]},
            profile,
            args.timeout,
        )
        code = _detail_code(payload)
        n_routes = 0
        stairs_len = 0.0
        stairs_seg = 0
        na_pct = ""
        if status == 200 and payload:
            routes = payload.get("routes") or []
            n_routes = len(routes)
            if routes:
                stairs_len, stairs_seg, na_pct = _route_stairs_and_quality(routes[0])

        ok = _check_expect(expect, status, code, n_routes)
        if not ok:
            failed += 1

        print(
            f"{sid}\t{ok}\t{status}\t{code or ''}\t{n_routes}\t{stairs_len:.1f}\t{stairs_seg}\t{na_pct}\t{expect}\t{label}",
        )

    if failed:
        print(f"\nНесовпадений с expect: {failed}", file=sys.stderr)
        sys.exit(1)
    print("\nВсе сценарии соответствуют expect.", file=sys.stderr)


if __name__ == "__main__":
    main()
