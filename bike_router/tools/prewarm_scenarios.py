#!/usr/bin/env python3
"""Прогрев кэшей по типовым сценариям перед демонстрацией (POST /alternatives/start).

Запуск (из корня репозитория NIR, с поднятым API)::

    python bike_router/tools/prewarm_scenarios.py --base http://127.0.0.1:8000

Прогоняет первые N сценариев с ``expect`` ok из ``route_scenarios.json`` с
``green_enabled=true``, чтобы заполнились route-cache, corridor graph cache,
тайлы и edge-кэш зелени. Повторный клик по тем же точкам в UI будет быстрее.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests


def _load_ok_scenarios(limit: int) -> list:
    p = Path(__file__).resolve().parent / "route_scenarios.json"
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for s in data["scenarios"]:
        if s.get("expect") != "ok":
            continue
        out.append(s)
        if len(out) >= limit:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Прогрев кэшей Bike Router")
    ap.add_argument(
        "--base",
        default="http://127.0.0.1:8000",
        help="URL API (без завершающего /)",
    )
    ap.add_argument(
        "-n",
        type=int,
        default=5,
        help="Число сценариев (по умолчанию 5 первых с expect=ok)",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Таймаут HTTP на один запрос (сек)",
    )
    args = ap.parse_args()
    base = args.base.rstrip("/")
    scenarios = _load_ok_scenarios(max(1, args.n))
    if not scenarios:
        print("Нет сценариев с expect=ok", file=sys.stderr)
        sys.exit(1)
    url = f"{base}/alternatives/start"
    t0 = time.perf_counter()
    for i, s in enumerate(scenarios, 1):
        sid = s.get("id", i)
        print(f"[{i}/{len(scenarios)}] {sid} …", flush=True)
        body = {
            "start": s["start"],
            "end": s["end"],
            "profile": s.get("profile", "cyclist"),
            "green_enabled": True,
        }
        try:
            r = requests.post(url, json=body, timeout=args.timeout)
            if r.status_code != 200:
                print(f"  HTTP {r.status_code}: {r.text[:200]}", file=sys.stderr)
                continue
            data = r.json()
            jid = data.get("job_id")
            if data.get("pending") and jid:
                poll = f"{base}/alternatives/job/{jid}"
                for _ in range(120):
                    time.sleep(1.5)
                    pr = requests.get(poll, timeout=30)
                    if pr.status_code != 200:
                        break
                    jd = pr.json()
                    if jd.get("status") == "failed" or not jd.get("pending"):
                        break
        except requests.RequestException as e:
            print(f"  Ошибка: {e}", file=sys.stderr)
    print(f"Готово за {time.perf_counter() - t0:.1f} с")


if __name__ == "__main__":
    main()
