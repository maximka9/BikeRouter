"""FastAPI backend для маршрутизации велосипедистов и пешеходов.

Запуск::

    cd NIR
    python -m bike_router

или: ``uvicorn bike_router.api:app --reload``

Документация: http://127.0.0.1:8000/docs
Фронтенд:    http://127.0.0.1:8000/
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import ROUTING_ALGO_VERSION
from .engine import RouteEngine, _resolve_season_for_heat_alternatives
from .exceptions import (
    BikeRouterError,
    OverpassUnavailableError,
    PointOutsideZoneError,
    RouteNotFoundError,
    RouteTooLongError,
)
from .metrics import metrics_text
from .models import (
    AlternativesJobResponse,
    AlternativesRequest,
    AlternativesResponse,
    AlternativesStartRequest,
    AlternativesStartResponse,
    ErrorDetail,
    GeocodingResult,
    HealthResponse,
    RouteRequest,
    RouteResponse,
)
from .job_status import AlternativesJobStatus as AJS
from .services.alternatives_jobs import AlternativesJobStore, JobRecord, new_job_id
from .logutil import configure_root_logging
from .middleware import RequestLogMiddleware
from .services.geocoding import GeocodingService

logger = logging.getLogger(__name__)
reqlog = logging.getLogger("bike_router.alternatives")
routelog = logging.getLogger("bike_router.routing")


def _serialize_criteria_bundle(
    bundle: Optional[Dict[str, List[RouteResponse]]],
) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    if not bundle:
        return None
    return {
        k: [r.model_dump(mode="json") for r in lst]
        for k, lst in bundle.items()
    }


def _parse_stored_criteria_bundle(
    raw: Optional[Dict[str, List[Dict[str, Any]]]],
) -> Optional[Dict[str, List[RouteResponse]]]:
    if not raw:
        return None
    return {
        k: [RouteResponse.model_validate(x) for x in lst]
        for k, lst in raw.items()
    }


def _use_progressive_phase1(_req: AlternativesStartRequest) -> bool:
    """Progressive 2.0: фаза 1 (shortest+full), затем green/heat/stress/heat_stress в фоне (см. .env)."""
    return bool(engine.settings.progressive_alternatives_enabled)


def _progressive_pending_modes(req: AlternativesStartRequest) -> List[str]:
    raw = getattr(engine.settings, "progressive_background_variants", "") or ""
    modes = [x.strip() for x in str(raw).split(",") if x.strip()]
    allowed = {"green", "heat", "stress", "heat_stress"}
    modes = [m for m in modes if m in allowed]
    if not req.green_enabled:
        modes = [m for m in modes if m != "green"]
    return modes if modes else ["heat", "stress", "heat_stress"]


def _alternatives_kw_common(
    req: Union[AlternativesRequest, AlternativesStartRequest],
) -> Dict[str, Any]:
    season_ov = req.season.value if req.season is not None else None
    return {
        "green_enabled": req.green_enabled,
        "departure_time": req.departure_time,
        "time_slot_override": req.time_slot,
        "air_temperature_c": req.air_temperature_c,
        "weather_mode": req.weather_mode,
        "use_live_weather": req.use_live_weather,
        "weather_time": req.weather_time,
        "temperature_c": req.temperature_c,
        "apparent_temperature_c": getattr(req, "apparent_temperature_c", None),
        "precipitation_mm": req.precipitation_mm,
        "wind_speed_ms": req.wind_speed_ms,
        "wind_direction_deg": req.wind_direction_deg,
        "cloud_cover_pct": req.cloud_cover_pct,
        "humidity_pct": req.humidity_pct,
        "wind_gusts_ms": req.wind_gusts_ms,
        "shortwave_radiation_wm2": req.shortwave_radiation_wm2,
        "snowfall_cm_h": req.snowfall_cm_h,
        "snow_depth_m": req.snow_depth_m,
        "weather_code": req.weather_code,
        "season": season_ov,
    }


def _run_progressive_background_thread(
    job_id: str,
    start: tuple,
    end: tuple,
    profile_key: str,
    alt_kw: Dict[str, Any],
) -> None:
    try:
        out = engine.compute_alternatives(
            start=start,
            end=end,
            profile_key=profile_key,
            **alt_kw,
        )
        dumps = [r.model_dump(mode="json") for r in out.routes]
        alternatives_job_store.finalize_progressive_job(
            job_id,
            routes=dumps,
            pending=[],
            status=AJS.DONE.value,
        )
        reqlog.info(
            "alternatives_job progressive job_id=%s done routes=%d",
            job_id,
            len(dumps),
        )
    except BikeRouterError as exc:
        rec = alternatives_job_store.get(job_id)
        keep: List[Dict[str, Any]] = []
        if rec is not None:
            with rec.mut:
                keep = list(rec.routes)
        alternatives_job_store.finalize_progressive_job(
            job_id,
            routes=keep,
            pending=[],
            status=AJS.FAILED.value,
            error={"code": exc.code, "message": str(exc)},
            green_warning=(
                "Не удалось досчитать дополнительные варианты маршрута "
                "(зелёный, тепло, стресс и т.д.). Первые два варианта сохранены."
            ),
        )
        reqlog.warning(
            "alternatives_job progressive job_id=%s failed code=%s",
            job_id,
            exc.code,
        )
    except Exception as exc:
        rec = alternatives_job_store.get(job_id)
        keep = []
        if rec is not None:
            with rec.mut:
                keep = list(rec.routes)
        alternatives_job_store.finalize_progressive_job(
            job_id,
            routes=keep,
            pending=[],
            status=AJS.FAILED.value,
            error={"code": "INTERNAL_ERROR", "message": str(exc)},
            green_warning=(
                "Внутренняя ошибка при досчёте вариантов. Первые два маршрута сохранены."
            ),
        )
        reqlog.exception("alternatives_job progressive job_id=%s", job_id)


def _run_green_job_thread(
    job_id: str,
    start: tuple,
    end: tuple,
    profile_key: str,
) -> None:
    """Фон: третий маршрут (green); при ошибке — status=failed и структурированный error."""
    try:
        route = engine.compute_green_route_addon(start, end, profile_key)
        alternatives_job_store.complete_green_success(
            job_id, route.model_dump(mode="json")
        )
        reqlog.info("alternatives_job job_id=%s green ready", job_id)
    except BikeRouterError as exc:
        reqlog.warning(
            "alternatives_job job_id=%s green failed BikeRouterError: %s",
            job_id,
            exc,
        )
        alternatives_job_store.complete_green_failure(
            job_id,
            error={"code": exc.code, "message": str(exc)},
            green_warning=(
                "Не удалось построить вариант с учётом озеленения. "
                "Два первых маршрута остаются без изменений."
            ),
        )
    except Exception as exc:
        reqlog.warning("alternatives_job job_id=%s green failed: %s", job_id, exc)
        alternatives_job_store.complete_green_failure(
            job_id,
            error={
                "code": "GREEN_ROUTE_FAILED",
                "message": str(exc) or "Ошибка расчёта зелёного маршрута",
            },
            green_warning=(
                "Маршрут с учётом озеленения не удалось построить. "
                "Первые два варианта остаются доступны."
            ),
        )

_FRONTEND_DIR = Path(__file__).parent / "frontend"

# ── Глобальные сервисы (singleton) ───────────────────────────────

engine = RouteEngine()
alternatives_job_store = AlternativesJobStore(
    ttl_sec=engine.settings.alternatives_job_ttl_sec,
)
geocoder = GeocodingService(
    disk_cache_dir=(
        os.path.join(engine.settings.cache_dir, "nominatim_disk")
        if engine.settings.geocode_disk_cache
        else None
    ),
    settings=engine.settings,
)

# ── Lifespan ─────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_root_logging()
    logger.info("Инициализация движка (warmup)...")
    engine.warmup()
    if engine.settings.use_dynamic_corridor_graph:
        logger.info(
            "Сервер готов: граф подгрузится по первому маршруту (коридор ±BUFFER)."
        )
    else:
        logger.info("Граф загружен — сервер готов к запросам.")
    yield
    logger.info("Остановка сервера.")


# ── Helpers ──────────────────────────────────────────────────────

_STATUS_MAP = {
    PointOutsideZoneError: 422,
    RouteTooLongError: 422,
    RouteNotFoundError: 404,
    OverpassUnavailableError: 502,
}


def _route_error(exc: BikeRouterError) -> HTTPException:
    """Преобразовать доменное исключение в HTTPException с ErrorDetail."""
    status = _STATUS_MAP.get(type(exc), 500)
    body = ErrorDetail(code=exc.code, message=str(exc))
    return HTTPException(status_code=status, detail=body.model_dump())


# ── FastAPI application ──────────────────────────────────────────

app = FastAPI(
    title="Bike Router API",
    description=(
        "Оптимизация маршрутов для велосипедистов и пешеходов.\n\n"
        "Профили: ``cyclist``, ``pedestrian``. "
        "``POST /alternatives`` — синхронно до шести вариантов: "
        "``shortest``, ``full``, ``green``, ``heat``, ``stress``, ``heat_stress`` "
        "(набор зависит от ``green_enabled`` и доступности путей).\n\n"
        "``POST /alternatives/start`` при ``PROGRESSIVE_ALTERNATIVES_ENABLED=true``: "
        "сначала ``shortest`` и ``full``, остальные варианты досчитываются в фоне "
        "(см. ``pending`` и ``GET /alternatives/job/{job_id}``).\n\n"
        "Ответ маршрута: геометрия, длина, время, рельеф, озеленение, лестницы, "
        "покрытия, погодный контекст и др."
    ),
    version=__version__,
    lifespan=lifespan,
)

def _cors_allow_origins() -> List[str]:
    """``CORS_ALLOW_ORIGINS``: через запятую или ``*`` (по умолчанию — только для dev)."""
    raw = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
    if not raw or raw == "*":
        return ["*"]
    return [x.strip() for x in raw.split(",") if x.strip()]


def _cors_allow_credentials(origins: List[str]) -> bool:
    """С ``*`` нельзя включать credentials (спецификация CORS / Starlette)."""
    return not (origins == ["*"] or any(o == "*" for o in origins))


_cors_origins = _cors_allow_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_allow_credentials(_cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLogMiddleware)


# ══════════════════════════════════════════════════════════════════
# Endpoints — Routing
# ══════════════════════════════════════════════════════════════════


@app.post(
    "/route",
    response_model=RouteResponse,
    responses={
        404: {"model": ErrorDetail, "description": "Маршрут не найден"},
        422: {"model": ErrorDetail, "description": "Точка вне зоны / слишком далёкий"},
        502: {"model": ErrorDetail, "description": "Overpass/OSM недоступен"},
    },
    summary="Построить маршрут",
    tags=["Routing"],
)
async def route(req: RouteRequest):
    """Построить **один** маршрут.

    - **start / end** — координаты (lat, lon).
    - **profile** — `cyclist` или `pedestrian`.
    - **mode** — `full`, `green` или `shortest` (минимальная длина по OSM).
    """
    t0 = time.perf_counter()
    try:
        return engine.compute_route(
            start=(req.start.lat, req.start.lon),
            end=(req.end.lat, req.end.lon),
            profile_key=req.profile.value,
            mode=req.mode.value,
        )
    except BikeRouterError as exc:
        ms = (time.perf_counter() - t0) * 1000
        routelog.warning(
            "route_rejected code=%s profile=%s mode=%s duration_ms=%.1f start=%.5f,%.5f end=%.5f,%.5f",
            exc.code,
            req.profile.value,
            req.mode.value,
            ms,
            req.start.lat,
            req.start.lon,
            req.end.lat,
            req.end.lon,
        )
        raise _route_error(exc)
    except ValueError as exc:
        ms = (time.perf_counter() - t0) * 1000
        routelog.warning(
            "route_validation_error profile=%s duration_ms=%.1f detail=%s",
            req.profile.value,
            ms,
            exc,
        )
        raise HTTPException(status_code=422, detail=str(exc))


@app.post(
    "/alternatives",
    response_model=AlternativesResponse,
    responses={
        404: {"model": ErrorDetail},
        422: {"model": ErrorDetail},
        502: {"model": ErrorDetail},
    },
    summary="Несколько вариантов маршрута (до 6)",
    tags=["Routing"],
)
async def alternatives(req: AlternativesRequest):
    """Синхронно: shortest, full, green, heat, stress, heat_stress (по доступности)."""
    t0 = time.perf_counter()
    try:
        kw = _alternatives_kw_common(req)
        out = engine.compute_alternatives(
            start=(req.start.lat, req.start.lon),
            end=(req.end.lat, req.end.lon),
            profile_key=req.profile.value,
            **kw,
        )
        ms = (time.perf_counter() - t0) * 1000
        reqlog.info(
            "alternatives ok profile=%s routes=%d duration_ms=%.1f",
            req.profile.value,
            len(out.routes),
            ms,
        )
        return out
    except BikeRouterError as exc:
        ms = (time.perf_counter() - t0) * 1000
        routelog.warning(
            "alternatives_rejected code=%s profile=%s duration_ms=%.1f start=%.5f,%.5f end=%.5f,%.5f",
            exc.code,
            req.profile.value,
            ms,
            req.start.lat,
            req.start.lon,
            req.end.lat,
            req.end.lon,
        )
        reqlog.info(
            "alternatives fail profile=%s code=%s duration_ms=%.1f",
            req.profile.value,
            exc.code,
            ms,
        )
        raise _route_error(exc)
    except ValueError as exc:
        ms = (time.perf_counter() - t0) * 1000
        routelog.warning(
            "alternatives_validation_error profile=%s duration_ms=%.1f detail=%s",
            req.profile.value,
            ms,
            exc,
        )
        reqlog.info(
            "alternatives fail profile=%s code=validation duration_ms=%.1f",
            req.profile.value,
            ms,
        )
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        ms = (time.perf_counter() - t0) * 1000
        logger.exception(
            "alternatives_internal_error profile=%s duration_ms=%.1f",
            req.profile.value,
            ms,
        )
        routelog.error(
            "alternatives_internal_error profile=%s duration_ms=%.1f err=%s",
            req.profile.value,
            ms,
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(
                code="INTERNAL_ERROR",
                message=(
                    "Внутренняя ошибка при расчёте маршрута. "
                    "Подробности — в логе сервера."
                ),
            ).model_dump(),
        ) from exc


@app.get(
    "/graph/stress-overlay",
    tags=["Routing"],
    summary="GeoJSON маски стресса по рёбрам текущего графа (коридор start–end)",
)
async def graph_stress_overlay(
    start_lat: float = Query(..., ge=-90, le=90),
    start_lon: float = Query(..., ge=-180, le=180),
    end_lat: float = Query(..., ge=-90, le=90),
    end_lon: float = Query(..., ge=-180, le=180),
):
    """Тестовая визуализация: рёбра рабочего графа с полями stress_lts, stress_cost и т.д."""
    try:
        engine._ensure_graph_for_corridor((start_lat, start_lon), (end_lat, end_lon))
        return engine.build_stress_overlay_geojson()
    except BikeRouterError as exc:
        raise _route_error(exc) from exc


@app.post(
    "/alternatives/start",
    response_model=AlternativesStartResponse,
    responses={
        404: {"model": ErrorDetail},
        422: {"model": ErrorDetail},
        502: {"model": ErrorDetail},
    },
    summary="Старт progressive-расчёта вариантов",
    tags=["Routing"],
)
async def alternatives_start(req: AlternativesStartRequest):
    """При PROGRESSIVE_ALTERNATIVES_ENABLED: shortest+full сразу, остальное в фоне; иначе полный ответ или только green в фоне."""
    job_id = new_job_id()
    t0 = time.perf_counter()
    st = (req.start.lat, req.start.lon)
    en = (req.end.lat, req.end.lon)
    pk = req.profile.value
    reqlog.info(
        "alternatives_start job_id=%s green=%s",
        job_id,
        req.green_enabled,
    )
    try:
        if _use_progressive_phase1(req):
            alt_kw = _alternatives_kw_common(req)
            now_utc = datetime.now(timezone.utc)
            dep_for = req.departure_time or now_utc.replace(microsecond=0).isoformat()
            heat_season = _resolve_season_for_heat_alternatives(
                season_override=alt_kw.get("season"),
                weather_time_iso=req.weather_time,
                departure_time_iso=dep_for,
                now_utc=now_utc,
            )
            routes_phase1 = engine.compute_alternatives_phase1_two_routes(
                st, en, pk, season=heat_season
            )
            pending_modes = _progressive_pending_modes(req)
            ms = (time.perf_counter() - t0) * 1000
            reqlog.info(
                "alternatives_start job_id=%s progressive phase1 routes=%d pending=%s duration_ms=%.1f",
                job_id,
                len(routes_phase1),
                ",".join(pending_modes),
                ms,
            )
            alternatives_job_store.create(
                job_id,
                JobRecord(
                    job_id=job_id,
                    status=AJS.RUNNING.value,
                    routes=[r.model_dump(mode="json") for r in routes_phase1],
                    pending=list(pending_modes),
                    criteria_bundle=None,
                ),
            )
            threading.Thread(
                target=_run_progressive_background_thread,
                args=(job_id, st, en, pk, alt_kw),
                name=f"alt-progressive-{job_id[:8]}",
                daemon=True,
            ).start()
            return AlternativesStartResponse(
                job_id=job_id,
                status=AJS.RUNNING.value,
                routes=routes_phase1,
                pending=list(pending_modes),
                criteria_bundle=None,
            )

        kw = _alternatives_kw_common(req)
        out = engine.compute_alternatives(start=st, end=en, profile_key=pk, **kw)
        routes = list(out.routes)
        cb = out.criteria_bundle
        cb_dump = _serialize_criteria_bundle(cb)
        ms = (time.perf_counter() - t0) * 1000
        reqlog.info(
            "alternatives_start job_id=%s sync routes=%d duration_ms=%.1f",
            job_id,
            len(routes),
            ms,
        )

        has_green = any(getattr(r, "mode", "") == "green" for r in routes)
        needs_green_thread = bool(req.green_enabled and not has_green)

        if needs_green_thread:
            alternatives_job_store.create(
                job_id,
                JobRecord(
                    job_id=job_id,
                    status=AJS.RUNNING_GREEN.value,
                    routes=[r.model_dump(mode="json") for r in routes],
                    pending=["green"],
                    criteria_bundle=cb_dump,
                ),
            )
            threading.Thread(
                target=_run_green_job_thread,
                args=(job_id, st, en, pk),
                name=f"alt-green-{job_id[:8]}",
                daemon=True,
            ).start()
            return AlternativesStartResponse(
                job_id=job_id,
                status=AJS.RUNNING_GREEN.value,
                routes=routes,
                pending=["green"],
                criteria_bundle=cb,
            )

        alternatives_job_store.create(
            job_id,
            JobRecord(
                job_id=job_id,
                status="done",
                routes=[r.model_dump(mode="json") for r in routes],
                pending=[],
                criteria_bundle=cb_dump,
            ),
        )
        return AlternativesStartResponse(
            job_id=job_id,
            status="done",
            routes=routes,
            pending=[],
            criteria_bundle=cb,
        )
    except BikeRouterError as exc:
        ms = (time.perf_counter() - t0) * 1000
        reqlog.info(
            "alternatives_start fail job_id=%s code=%s duration_ms=%.1f",
            job_id,
            exc.code,
            ms,
        )
        raise _route_error(exc)
    except ValueError as exc:
        ms = (time.perf_counter() - t0) * 1000
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get(
    "/alternatives/job/{job_id}",
    response_model=AlternativesJobResponse,
    responses={404: {"model": ErrorDetail}},
    summary="Состояние задачи progressive-расчёта",
    tags=["Routing"],
)
async def alternatives_job_status(job_id: str):
    rec = alternatives_job_store.get(job_id)
    if rec is None:
        reqlog.info("alternatives_job poll job_id=%s not_found_or_expired", job_id)
        raise HTTPException(
            status_code=404,
            detail=ErrorDetail(
                code="JOB_NOT_FOUND",
                message="Задача не найдена или срок хранения истёк",
            ).model_dump(),
        )
    with rec.mut:
        routes_raw = list(rec.routes)
        pending = list(rec.pending)
        status = rec.status
        err = rec.error
        gw = rec.green_warning
        cb_stored = rec.criteria_bundle
    routes = [RouteResponse.model_validate(x) for x in routes_raw]
    criteria_bundle = _parse_stored_criteria_bundle(cb_stored)
    if status in (
        AJS.DONE.value,
        AJS.FAILED.value,
        AJS.RUNNING.value,
        AJS.PARTIAL.value,
        AJS.RUNNING_GREEN.value,
    ):
        reqlog.debug(
            "alternatives_job poll job_id=%s status=%s routes=%d pending=%s",
            job_id,
            status,
            len(routes),
            pending,
        )
    if status in (AJS.DONE.value, AJS.FAILED.value):
        reqlog.info(
            "alternatives_job completed job_id=%s status=%s routes=%d",
            job_id,
            status,
            len(routes),
        )
    return AlternativesJobResponse(
        job_id=job_id,
        status=status,
        routes=routes,
        pending=pending,
        error=ErrorDetail.model_validate(err) if err else None,
        green_warning=gw,
        criteria_bundle=criteria_bundle,
    )


# ══════════════════════════════════════════════════════════════════
# Endpoints — Geocoding
# ══════════════════════════════════════════════════════════════════


@app.get(
    "/geocode",
    response_model=List[GeocodingResult],
    summary="Адрес → координаты",
    tags=["Geocoding"],
)
async def geocode(
    q: str = Query(..., min_length=2, description="Адрес или название места"),
    limit: int = Query(5, ge=1, le=20),
):
    """Прямое геокодирование (Nominatim / OSM). Кэшируется.

    Для публичного экземпляра Nominatim действует политика OSMF: не использовать
    client-side autocomplete; не чаще ~1 запроса/сек; вызывать только по явному
    действию пользователя (кнопка поиска, Enter). Веб-интерфейс следует этой схеме.
    """
    t0 = time.perf_counter()
    try:
        results = geocoder.geocode(q, limit=limit)
        ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "geocode ok query_len=%d results=%d duration_ms=%.1f",
            len(q),
            len(results),
            ms,
        )
        return results
    except requests.Timeout:
        ms = (time.perf_counter() - t0) * 1000
        logger.warning("geocode timeout duration_ms=%.1f", ms)
        raise HTTPException(
            status_code=502,
            detail=ErrorDetail(
                code="GEOCODING_TIMEOUT",
                message="Геокодер не ответил вовремя. Повторите запрос позже.",
            ).model_dump(),
        )
    except requests.RequestException as exc:
        ms = (time.perf_counter() - t0) * 1000
        logger.warning("geocode network error: %s duration_ms=%.1f", exc, ms)
        raise HTTPException(
            status_code=502,
            detail=ErrorDetail(
                code="GEOCODING_NETWORK",
                message=(
                    "Не удалось связаться с сервисом геокодирования "
                    "(сеть, DNS или блокировка). Проверьте подключение к интернету."
                ),
            ).model_dump(),
        )
    except Exception as exc:
        ms = (time.perf_counter() - t0) * 1000
        logger.exception("geocode error duration_ms=%.1f", ms)
        raise HTTPException(
            status_code=502,
            detail=ErrorDetail(
                code="GEOCODING_ERROR", message=str(exc)
            ).model_dump(),
        )


@app.get(
    "/reverse-geocode",
    response_model=GeocodingResult,
    summary="Координаты → адрес",
    tags=["Geocoding"],
)
async def reverse_geocode(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
):
    """Обратное геокодирование (Nominatim / OSM). Кэшируется.

    Вызывается при выборе точки на карте / перетаскивании маркера — отдельные
    запросы; повторные координаты обслуживаются кэшем (память и при включении — диск).
    """
    try:
        result = geocoder.reverse_geocode(lat, lon)
    except requests.Timeout:
        raise HTTPException(
            status_code=502,
            detail=ErrorDetail(
                code="GEOCODING_TIMEOUT",
                message="Обратный геокодер не ответил вовремя.",
            ).model_dump(),
        )
    except requests.RequestException:
        raise HTTPException(
            status_code=502,
            detail=ErrorDetail(
                code="GEOCODING_NETWORK",
                message="Сетевая ошибка при обращении к геокодеру.",
            ).model_dump(),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=ErrorDetail(
                code="GEOCODING_ERROR", message=str(exc)
            ).model_dump(),
        )
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=ErrorDetail(
                code="GEOCODING_NOT_FOUND", message="Адрес не найден"
            ).model_dump(),
        )
    return result


# ══════════════════════════════════════════════════════════════════
# Endpoints — System
# ══════════════════════════════════════════════════════════════════


@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    """Prometheus при ``pip install prometheus_client``."""
    return Response(content=metrics_text(), media_type="text/plain; version=0.0.4")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    G = engine.graph
    return HealthResponse(
        status="ok",
        version=__version__,
        graph_loaded=engine.is_loaded,
        nodes=G.number_of_nodes() if G else 0,
        edges=G.number_of_edges() if G else 0,
        profiles=["cyclist", "pedestrian"],
        graph_built_at_utc=engine.graph_built_at_utc,
        routing_engine_fingerprint=RouteEngine.routing_weights_fingerprint(),
        routing_algo_version=ROUTING_ALGO_VERSION,
        satellite_green_enabled=not engine.settings.disable_satellite_green,
        graph_corridor_mode=engine.settings.use_dynamic_corridor_graph,
    )


# ══════════════════════════════════════════════════════════════════
# Frontend — статика
# ══════════════════════════════════════════════════════════════════


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(str(_FRONTEND_DIR / "index.html"))


@app.get("/about", include_in_schema=False)
async def about_page():
    return FileResponse(str(_FRONTEND_DIR / "about.html"))


app.mount(
    "/static",
    StaticFiles(directory=str(_FRONTEND_DIR)),
    name="static",
)
