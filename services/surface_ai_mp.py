"""Multiprocessing-воркеры Surface AI (spawn-safe, top-level функции)."""

from __future__ import annotations

import json
import logging
import time
from io import StringIO
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List

import geopandas as gpd
import pandas as pd

_log = logging.getLogger(__name__)

# Ниже этого числа рёбер тайловые признаки считаются последовательно (меньше оверхеда MP).
_TILE_MP_MIN_EDGES = 40


def extract_tile_features_chunk_worker(payload: dict) -> str:
    """Чанк рёбер → JSON ``records`` с признаками тайлов (как ``extract_tile_features_for_edges``).

    Не пишет файлов; только чтение тайлов из кэша. Ошибка по ребру → строка с NaN-признаками.
    """
    from bike_router.services.surface_ml import (
        SurfaceMLConfig,
        extract_tile_features_for_edges,
        no_progress,
    )

    geo = json.loads(payload["geojson"])
    crs = payload.get("crs") or "EPSG:4326"
    gdf = gpd.GeoDataFrame.from_features(geo.get("features") or [], crs=crs)
    tile_settings = SimpleNamespace(
        cache_dir=str(payload["cache_dir"]),
        tms_server=str(payload["tms_server"]),
        satellite_zoom=int(payload["satellite_zoom"]),
    )
    ml_cfg = SurfaceMLConfig(**dict(payload["ml_config"]))
    out = extract_tile_features_for_edges(
        gdf,
        tile_settings,
        ml_cfg,
        progress=no_progress,
    )
    return out.to_json(orient="records")


def parallel_extract_tile_features_for_edges(
    edges_gdf: gpd.GeoDataFrame,
    settings: Any,
    config: Any,
    ml_config: Any,
    *,
    progress: Callable[..., Any],
) -> pd.DataFrame:
    """Параллельное извлечение тайловых признаков по чанкам рёбер."""
    from bike_router.tools._parallel_utils import auto_cpu_count, auto_worker_count, chunked

    from bike_router.services.surface_ml import extract_tile_features_for_edges

    n = len(edges_gdf)
    cache_dir = Path(settings.cache_dir)
    if getattr(config, "tiles_dir", None):
        cache_dir = Path(config.tiles_dir).expanduser().resolve().parent
    tile_settings = SimpleNamespace(
        cache_dir=str(cache_dir),
        tms_server=config.tms_server or settings.tms_server,
        satellite_zoom=int(config.tile_zoom or settings.satellite_zoom),
    )

    if n < _TILE_MP_MIN_EDGES or not bool(getattr(config, "auto_parallel", True)):
        return extract_tile_features_for_edges(
            edges_gdf,
            tile_settings,
            ml_config,
            progress=progress,
        )

    cap = max(1, auto_cpu_count() - 1)
    target_workers = min(cap, max(2, n // 50))
    chunk_sz = max(35, (n + target_workers - 1) // target_workers)
    parts = list(chunked(list(range(n)), chunk_sz))
    workers = min(auto_worker_count(len(parts), memory_heavy=False), len(parts))

    ml_dict = {
        "sample_step_m": float(ml_config.sample_step_m),
        "pixel_window": int(ml_config.pixel_window),
        "min_confidence": float(ml_config.min_confidence),
        "paved_good_min_confidence": float(ml_config.paved_good_min_confidence),
        "spatial_grid_m": float(ml_config.spatial_grid_m),
        "test_share": float(ml_config.test_share),
        "random_state": int(ml_config.random_state),
        "min_known_edges": int(ml_config.min_known_edges),
        "n_estimators": int(ml_config.n_estimators),
        "max_tile_cache_items": int(ml_config.max_tile_cache_items),
    }

    t0 = time.perf_counter()
    _log.info(
        "Surface AI tile features: parallel edges=%d chunks=%d workers=%d chunk_sz=%d",
        n,
        len(parts),
        workers,
        chunk_sz,
    )

    payloads: List[dict] = []
    for part in parts:
        sub = edges_gdf.iloc[part].copy()
        payloads.append(
            {
                "geojson": sub.to_json(),
                "crs": str(sub.crs) if sub.crs is not None else "EPSG:4326",
                "cache_dir": str(cache_dir),
                "tms_server": str(config.tms_server or settings.tms_server),
                "satellite_zoom": int(config.tile_zoom or settings.satellite_zoom),
                "ml_config": ml_dict,
            }
        )

    frames: List[pd.DataFrame] = []
    if workers <= 1:
        for p in payloads:
            frames.append(
                pd.read_json(StringIO(extract_tile_features_chunk_worker(p)))
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for js in ex.map(extract_tile_features_chunk_worker, payloads):
                frames.append(pd.read_json(StringIO(js)))

    out = pd.concat(frames, axis=0, ignore_index=True)
    out["edge_id"] = out["edge_id"].map(lambda x: "" if pd.isna(x) else str(x).strip())
    before = len(out)
    out = out.drop_duplicates(subset=["edge_id"], keep="last")
    if len(out) < before:
        _log.warning("Surface AI tile features: снято %d дублирующихся edge_id после concat", before - len(out))
    order = {eid: i for i, eid in enumerate(edges_gdf["edge_id"].astype(str))}
    out["_ord"] = out["edge_id"].astype(str).map(lambda x: order.get(x, 10**9))
    out = out.sort_values("_ord").drop(columns=["_ord"], errors="ignore").reset_index(drop=True)
    elapsed = time.perf_counter() - t0
    _log.info("Surface AI tile features: parallel done rows=%d elapsed=%.1fs", len(out), elapsed)
    return out


def train_surface_ai_candidate_worker(payload: dict) -> dict:
    """Обучение и оценка одного кандидата (отдельный процесс)."""
    import joblib

    from bike_router.services.surface_ai import (
        SurfaceAIConfig,
        _candidate_display,
        _candidate_feature_set,
        _fit_candidate_model,
        evaluate_task_model,
    )

    cfg: SurfaceAIConfig = joblib.load(payload["config_path"])
    label_meta: dict = joblib.load(payload["label_meta_path"])
    train = pd.read_parquet(payload["train_path"])
    calibration = pd.read_parquet(payload["calibration_path"])
    test = pd.read_parquet(payload["test_path"])
    candidate = str(payload["candidate"])
    target = str(payload["target"])
    sklearn_n_jobs = int(payload.get("sklearn_n_jobs", 1))

    target_col = "surface_group_direct_label" if target == "group_direct" else "surface_train_label"
    labels = sorted(
        set(train[target_col].astype(str)) | set(test[target_col].astype(str))
    )

    display, features_label = _candidate_display(candidate)
    feature_set = _candidate_feature_set(candidate)
    model, _ = _fit_candidate_model(
        candidate,
        feature_set,
        target,
        train,
        calibration,
        cfg,
        labels,
        label_meta,
        sklearn_n_jobs=sklearn_n_jobs,
    )
    metrics, details = evaluate_task_model(
        candidate=candidate,
        model=model,
        feature_set=feature_set,
        target=target,
        test=test,
        labels=labels,
        config=cfg,
    )
    row = {
        "model_key": candidate,
        "model": display,
        "target": target,
        "features": features_label,
        "accuracy": metrics.get("accuracy", 0.0),
        "balanced_accuracy": metrics.get("balanced_accuracy", 0.0),
        "macro_f1": metrics.get("macro_f1", 0.0),
        "weighted_f1": metrics.get("weighted_f1", 0.0),
        "macro_f1_safe": metrics.get("macro_f1_safe", 0.0),
        "macro_f1_safe_rough_aware": metrics.get("macro_f1_safe_rough_aware", 0.0),
        "recall_bad_surface": metrics.get("recall_bad_surface", 0.0),
        "recall_unpaved_soft": metrics.get("recall_unpaved_soft", 0.0),
        "recall_paved_rough": metrics.get("recall_paved_rough", float("nan")),
        "precision_paved_rough": metrics.get("precision_paved_rough", float("nan")),
        "f1_paved_rough": metrics.get("f1_paved_rough", float("nan")),
        "paved_rough_to_paved_good_count": metrics.get("paved_rough_to_paved_good_count", 0),
        "paved_rough_to_paved_good_rate": metrics.get("paved_rough_to_paved_good_rate", 0.0),
        "rough_upgrade_rate_effective": metrics.get("rough_upgrade_rate_effective", 0.0),
        "rough_upgrade_length_m_effective": metrics.get("rough_upgrade_length_m_effective", 0.0),
        "recall_asphalt": metrics.get("recall_asphalt", float("nan")),
        "recall_concrete": metrics.get("recall_concrete", float("nan")),
        "recall_paving_stones": metrics.get("recall_paving_stones", float("nan")),
        "dangerous_upgrade_rate_raw": metrics.get("dangerous_upgrade_rate_raw", 0.0),
        "dangerous_upgrade_rate_effective": metrics.get("dangerous_upgrade_rate_effective", 0.0),
        "dangerous_upgrade_count_raw": metrics.get("dangerous_upgrade_count_raw", 0),
        "dangerous_upgrade_count_effective": metrics.get("dangerous_upgrade_count_effective", 0),
        "dangerous_upgrade_length_m_effective": metrics.get("dangerous_upgrade_length_m_effective", 0.0),
    }
    out_model = Path(payload["models_dir"]) / f"_model_{candidate.replace('/', '_')}.joblib"
    joblib.dump(model, out_model)
    det_path = Path(payload["models_dir"]) / f"_details_{candidate.replace('/', '_')}.parquet"
    details.to_parquet(det_path, index=False)
    return {
        "candidate": candidate,
        "row": row,
        "model_path": str(out_model),
        "details_path": str(det_path),
        "feature_set": feature_set,
        "metrics": metrics,
        "labels": labels,
        "target": target,
    }
