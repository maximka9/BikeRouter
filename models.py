"""Pydantic-модели: замороженный формат ответа движка маршрутизации.

Все ответы API и чистой функции ``compute_route`` возвращают
экземпляры этих моделей — JSON-совместимые, самодокументируемые,
с валидацией на входе и выходе.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Перечисления ─────────────────────────────────────────────────


class ProfileEnum(str, Enum):
    cyclist = "cyclist"
    pedestrian = "pedestrian"


class ModeEnum(str, Enum):
    full = "full"
    green = "green"
    shortest = "shortest"


# ── Базовые типы ─────────────────────────────────────────────────


class LatLon(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Широта")
    lon: float = Field(..., ge=-180, le=180, description="Долгота")


# ── Запросы ──────────────────────────────────────────────────────


class RouteRequest(BaseModel):
    """Запрос на построение одного маршрута."""

    start: LatLon
    end: LatLon
    profile: ProfileEnum = ProfileEnum.cyclist
    mode: ModeEnum = ModeEnum.full

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "start": {"lat": 53.186243, "lon": 50.088031},
                    "end": {"lat": 53.195334, "lon": 50.124063},
                    "profile": "cyclist",
                    "mode": "full",
                }
            ]
        }
    }


class AlternativesRequest(BaseModel):
    """Запрос на несколько вариантов маршрута (оптимальный, зелёный, кратчайший/альтернатива)."""

    start: LatLon
    end: LatLon
    profile: ProfileEnum = ProfileEnum.cyclist
    green_enabled: bool = Field(
        True,
        description="False — только кратчайший и энергетический, без спутника и без зелёного маршрута",
    )


class AlternativesStartRequest(BaseModel):
    """Старт progressive-расчёта (два маршрута сразу, зелёный в фоне при green_enabled)."""

    start: LatLon
    end: LatLon
    profile: ProfileEnum = ProfileEnum.cyclist
    green_enabled: bool = True


class AlternativesStartResponse(BaseModel):
    job_id: str
    status: str = Field(
        ...,
        description="running_green | done | failed — фаза 1+ожидание зелёного или сразу готово",
    )
    routes: List["RouteResponse"] = Field(default_factory=list)
    pending: List[str] = Field(
        default_factory=list,
        description="Например [\"green\"] пока считается третий вариант",
    )


class AlternativesJobResponse(BaseModel):
    job_id: str
    status: str
    routes: List["RouteResponse"] = Field(default_factory=list)
    pending: List[str] = Field(default_factory=list)
    error: Optional[ErrorDetail] = None
    green_warning: Optional[str] = Field(
        default=None,
        description="Если зелёный маршрут не удалось добавить",
    )


# ── Подблоки ответа ──────────────────────────────────────────────


class ElevationMetrics(BaseModel):
    """Метрики рельефа маршрута."""

    climb_m: float = Field(..., description="Суммарный набор высоты (м)")
    descent_m: float = Field(..., description="Суммарный спуск (м)")
    max_gradient_pct: float = Field(..., description="Макс. уклон (%)")
    avg_gradient_pct: float = Field(..., description="Средний уклон (%)")
    max_above_start_m: float = Field(
        ..., description="Макс. подъём относительно старта (м)"
    )
    max_below_start_m: float = Field(
        ..., description="Макс. спуск относительно старта (м, ≤0)"
    )
    end_diff_m: float = Field(..., description="Перепад старт→финиш (м)")


class GreenMetrics(BaseModel):
    """Метрики озеленения маршрута."""

    percent: float = Field(..., description="Доля озеленённых рёбер (%)")
    avg_trees_pct: float = Field(..., description="Средняя доля деревьев (%)")
    avg_grass_pct: float = Field(..., description="Средняя доля травы (%)")
    categories: Dict[str, int] = Field(
        default_factory=dict, description="Категории озеленения → кол-во рёбер"
    )


class StairsInfo(BaseModel):
    """Информация о лестницах на маршруте."""

    count: int = 0
    total_length_m: float = 0.0


class SurfaceBreakdown(BaseModel):
    """Распределение покрытий и типов дорог."""

    surfaces: Dict[str, int] = Field(
        default_factory=dict, description="Покрытие → кол-во рёбер на маршруте"
    )
    highways: Dict[str, int] = Field(
        default_factory=dict, description="Тип дороги → кол-во рёбер на маршруте"
    )
    na_fraction: float = Field(
        ...,
        ge=0,
        le=1,
        description=(
            "Доля длины маршрута (сумма длин рёбер OSM) без тега surface в данных OSM; "
            "согласовано с quality_hints.na_surface_fraction"
        ),
    )


class ElevationPoint(BaseModel):
    """Точка профиля высот."""

    distance_m: float
    elevation_m: float


class RouteQualityHints(BaseModel):
    """Предупреждения о полноте OSM и эвристиках вдоль маршрута."""

    warnings: List[str] = Field(
        default_factory=list,
        description="Тексты для показа пользователю",
    )
    na_surface_fraction: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Доля длины маршрута (сумма длин рёбер) без тега surface",
    )
    inferred_surface_fraction: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Доля длины без surface или с неизвестным значением — коэффициент 1.0"
        ),
    )
    inferred_highway_fraction: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Доля длины без highway или с неизвестным типом — коэффициент 1.0"
        ),
    )


class MapLayersGeoJSON(BaseModel):
    """GeoJSON-слои сегментов выбранного маршрута (для MapLibre / аналитики)."""

    greenery: Dict[str, Any] = Field(
        ..., description="Сегменты с заметным озеленением"
    )
    stairs: Dict[str, Any] = Field(..., description="highway=steps")
    problematic: Dict[str, Any] = Field(
        ..., description="Крутые уклоны или магистрали/второстепенки с трафиком"
    )
    na_surface: Dict[str, Any] = Field(
        ..., description="Нет тега surface в OSM"
    )


# ── Основной ответ ───────────────────────────────────────────────


class RouteResponse(BaseModel):
    """Единый объект результата маршрутизации.

    Содержит всё для отображения маршрута: геометрию, длину,
    время, рельеф, озеленение, лестницы, покрытия и профиль высот.
    """

    profile: str = Field(..., description="cyclist | pedestrian")
    mode: str = Field(..., description="full | green | shortest")
    variant_label: str = Field(
        default="",
        description="Человекочитаемая подпись варианта (для UI)",
    )
    geometry: List[List[float]] = Field(
        ..., description="Полилиния маршрута [[lat, lon], ...]"
    )
    length_m: float = Field(..., description="Длина маршрута (м)")
    time_s: float = Field(..., description="Оценка времени (с)")
    time_display: str = Field(
        ..., description="Человекочитаемое время, напр. '12 мин 30 сек'"
    )
    cost: float = Field(
        ..., description="Стоимость маршрута (единицы энерго-модели)"
    )
    elevation: ElevationMetrics
    green: GreenMetrics
    stairs: StairsInfo
    surfaces: SurfaceBreakdown
    elevation_profile: List[ElevationPoint] = Field(
        ..., description="Профиль высот (абсолютные высоты)"
    )
    map_layers: Optional[MapLayersGeoJSON] = Field(
        default=None,
        description="Слои карты: озеленение, лестницы, проблемные участки, N/A surface",
    )
    quality_hints: Optional[RouteQualityHints] = Field(
        default=None,
        description="Предупреждения о качестве данных OSM и fallback-коэффициентах",
    )


class AlternativesResponse(BaseModel):
    """Несколько вариантов маршрута для одного профиля (обычно 2–3 шт.)."""

    routes: List[RouteResponse] = Field(
        ...,
        description="Список вариантов: оптимальный, зелёный, кратчайший по длине или альтернатива",
        min_length=1,
    )


# ── Геокодирование ───────────────────────────────────────────────


class GeocodingResult(BaseModel):
    lat: float
    lon: float
    display_name: str


# ── Ошибки ───────────────────────────────────────────────────────


class ErrorDetail(BaseModel):
    """Структурированная ошибка API."""

    code: str = Field(..., description="Машинно-читаемый код ошибки")
    message: str = Field(..., description="Человекочитаемое описание")


# ── Health ───────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = Field(..., description="ok — процесс отвечает; детали графа в graph_loaded")
    version: str
    graph_loaded: bool
    nodes: int = 0
    edges: int = 0
    profiles: List[str] = Field(default_factory=list)
    graph_built_at_utc: Optional[str] = Field(
        default=None,
        description="Момент последней сборки графа на сервере (UTC, ISO 8601)",
    )
    routing_engine_fingerprint: str = Field(
        default="",
        description="SHA-256 параметров весов; смена инвалидирует кэш маршрутов",
    )
    routing_algo_version: str = Field(
        default="",
        description="Версия логики весов (ROUTING_ALGO_VERSION); поднять при смене формул",
    )
    satellite_green_enabled: bool = Field(
        default=True,
        description="False если DISABLE_SATELLITE_GREEN — без анализа снимков, зелень в ответах нулевая",
    )
    graph_corridor_mode: bool = Field(
        default=False,
        description="True — граф по коридору POST ±BUFFER (GRAPH_CORRIDOR_MODE), без фиксированного AREA",
    )
