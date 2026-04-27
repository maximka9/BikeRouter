"""Сравнение маршрутов с SURFACE_AI_RUNTIME off vs on.

Примеры::

    python -m bike_router.tools.surface_runtime_route_experiment --modes off,on
    python -m bike_router.tools.surface_runtime_route_experiment --pairs-csv pairs.csv --modes off,on
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple


def _ensure_pkg_path() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)


def _route_surface_shares(surfaces: Dict[str, int]) -> Dict[str, float]:
    tot = sum(int(v) for v in surfaces.values()) if surfaces else 0
    if tot <= 0:
        return {}
    return {k: round(int(v) / tot, 4) for k, v in surfaces.items()}


def _read_pairs_csv(path: str) -> List[Tuple[float, float, float, float]]:
    import pandas as pd

    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    for a, b, c, d in (
        ("start_lat", "start_lon", "end_lat", "end_lon"),
        ("lat1", "lon1", "lat2", "lon2"),
    ):
        if all(x in cols for x in (a, b, c, d)):
            la = df[cols[a]].astype(float)
            lo = df[cols[b]].astype(float)
            la2 = df[cols[c]].astype(float)
            lo2 = df[cols[d]].astype(float)
            return list(zip(la.tolist(), lo.tolist(), la2.tolist(), lo2.tolist()))
    raise ValueError(
        "Ожидаются колонки start_lat,start_lon,end_lat,end_lon "
        "или lat1,lon1,lat2,lon2 — не найдено в CSV."
    )


def run_modes_for_pair(
    modes_csv: str,
    start: Tuple[float, float],
    end: Tuple[float, float],
    *,
    profile: str = "cyclist",
    mode: str = "full",
) -> List[Dict[str, Any]]:
    _ensure_pkg_path()
    from bike_router.app import Application
    from bike_router.config import Settings
    from bike_router.engine import RouteEngine

    modes = [m.strip().lower() for m in modes_csv.split(",") if m.strip()]
    out: List[Dict[str, Any]] = []
    for m in modes:
        en = "true" if m == "on" else "false"
        os.environ["SURFACE_AI_RUNTIME_ENABLED"] = en
        app = Application(Settings())
        eng = RouteEngine(app)
        eng.warmup()
        rr = eng.compute_route(start, end, profile, mode)
        sr = rr.surface_resolution or {}
        surf = rr.surfaces.surfaces if rr.surfaces else {}
        out.append(
            {
                "surface_ai_runtime": m,
                "start_lat": start[0],
                "start_lon": start[1],
                "end_lat": end[0],
                "end_lon": end[1],
                "route_length_m": rr.length_m,
                "route_weight_full": rr.cost,
                "surface_resolution": sr,
                "route_surface_edge_shares": _route_surface_shares(surf),
                "geometry_points": len(rr.geometry),
            }
        )
    return out


def _compare_off_on(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by = {r["surface_ai_runtime"]: r for r in rows}
    off = by.get("off")
    on = by.get("on")
    if not off or not on:
        return {}
    dlen = off["route_length_m"] - on["route_length_m"]
    dwc = off["route_weight_full"] - on["route_weight_full"]
    len0 = max(off["route_length_m"], 1e-6)
    w0 = max(abs(off["route_weight_full"]), 1e-6)
    return {
        "delta_length_m": round(dlen, 2),
        "delta_length_percent": round(100.0 * dlen / len0, 4),
        "delta_weight_percent": round(100.0 * dwc / w0, 4),
        "route_changed_bool": bool(
            abs(dlen) > 1.0 or abs(dwc) > max(1e-3, 1e-4 * w0)
        ),
    }


def run_modes(modes_csv: str) -> List[Dict[str, Any]]:
    """Одна пара START/END из Settings — плоский список по режимам (совместимость с route_batch_experiment)."""
    _ensure_pkg_path()
    from bike_router.config import Settings

    s = Settings()
    return run_modes_for_pair(
        modes_csv,
        s.start_coords,
        s.end_coords,
    )


def run_modes_multi_pairs(
    modes_csv: str,
    pairs: List[Tuple[float, float, float, float]],
) -> List[Dict[str, Any]]:
    """Несколько O-D: каждая строка — pair_index, modes, compare_off_on."""
    batch: List[Dict[str, Any]] = []
    for i, tup in enumerate(pairs):
        slat, slon, elat, elon = tup
        rows = run_modes_for_pair(modes_csv, (slat, slon), (elat, elon))
        item: Dict[str, Any] = {"pair_index": i, "modes": rows}
        cmp = _compare_off_on(rows)
        if cmp:
            item["compare_off_on"] = cmp
        batch.append(item)
    return batch


def main() -> None:
    _ensure_pkg_path()
    parser = argparse.ArgumentParser(
        description="Сравнение маршрута(ов) с ML runtime off vs on.",
    )
    parser.add_argument(
        "--modes",
        default="off,on",
        help="Через запятую: off, on или оба.",
    )
    parser.add_argument(
        "--pairs-csv",
        default="",
        metavar="PATH",
        help="Несколько O-D: колонки start_lat,start_lon,end_lat,end_lon (или lat1,lon1,lat2,lon2).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if str(args.pairs_csv or "").strip():
        pairs = _read_pairs_csv(str(args.pairs_csv).strip())
        rows = run_modes_multi_pairs(args.modes, pairs)
    else:
        rows = run_modes(args.modes)
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
