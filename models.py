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


class SeasonEnum(str, Enum):
    """Сезон для температурного множителя (см. policies/heat_season.json)."""

    summer = "summer"
    spring_autumn = "spring_autumn"


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
    departure_time: Optional[str] = Field(
        default=None,
        description="ISO 8601 локального времени, напр. 2026-07-15T13:00:00 — выбор теплового слота",
    )
    time_slot: Optional[str] = Field(
        default=None,
        description="Явный слот: morning | noon | evening | night (если задан — важнее departure_time)",
    )
    season: SeasonEnum = Field(
        SeasonEnum.summer,
        description="Сезон для множителя тепла (summer | spring_autumn)",
    )
    air_temperature_c: Optional[float] = Field(
        default=None,
        description="Опционально: температура воздуха °C для поправки тепловой модели",
    )
    weather_mode: str = Field(
        default="none",
        description="none | auto | manual | fixed-snapshot — см. resolve_weather_for_route",
    )
    use_live_weather: bool = Field(
        default=False,
        description="Если true — как auto (прогноз/кэш по координатам и времени)",
    )
    weather_time: Optional[str] = Field(
        default=None,
        description="ISO 8601 — время для почасовой погоды (иначе departure_time)",
    )
    temperature_c: Optional[float] = Field(default=None, description="Ручая погода °C")
    precipitation_mm: Optional[float] = Field(default=None, description="Осадки мм/ч (ручной режим)")
    wind_speed_ms: Optional[float] = Field(default=None, description="Ветер м/с")
    wind_direction_deg: Optional[float] = Field(
        default=None,
        description="Направление ветра ° (метео: откуда дует), для manual/fixed-snapshot",
    )
    cloud_cover_pct: Optional[float] = Field(default=None, description="Облачность %")
    humidity_pct: Optional[float] = Field(default=None, description="Влажность %")
    wind_gusts_ms: Optional[float] = Field(default=None, description="Порывы ветра м/с")
    shortwave_radiation_wm2: Optional[float] = Field(
        default=None, description="КВ радиация Вт/м² (ручной снимок)"
    )
    snowfall_cm_h: Optional[float] = Field(
        default=None, description="Снегопад см/ч (ручной снимок)"
    )
    snow_depth_m: Optional[float] = Field(
        default=None,
        description="Глубина снега м (модельная; ручной снимок)",
    )
    weather_code: Optional[int] = Field(
        default=None, description="Код погоды WMO (Open-Meteo), опционально"
    )


class AlternativesStartRequest(BaseModel):
    """Запрос на расчёт набора маршрутов (см. :class:`AlternativesRequest`)."""

    start: LatLon
    end: LatLon
    profile: ProfileEnum = ProfileEnum.cyclist
    green_enabled: bool = True
    departure_time: Optional[str] = Field(
        default=None,
        description="ISO 8601 локального времени выезда",
    )
    time_slot: Optional[str] = Field(
        default=None,
        description="Явный слот (важнее departure_time, если задан)",
    )
    season: SeasonEnum = Field(SeasonEnum.summer, description="Сезон для множителя тепла")
    air_temperature_c: Optional[float] = Field(
        default=None,
        description="Температура воздуха °C (опционально)",
    )
    weather_mode: str = Field(default="none")
    use_live_weather: bool = False
    weather_time: Optional[str] = None
    temperature_c: Optional[float] = None
    precipitation_mm: Optional[float] = None
    wind_speed_ms: Optional[float] = None
    cloud_cover_pct: Optional[float] = None
    humidity_pct: Optional[float] = None
    wind_gusts_ms: Optional[float] = None
    wind_direction_deg: Optional[float] = None
    shortwave_radiation_wm2: Optional[float] = None
    snowfall_cm_h: Optional[float] = None
    snow_depth_m: Optional[float] = None
    weather_code: Optional[int] = None


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
    criteria_bundle: Optional[Dict[str, List["RouteResponse"]]] = Field(
        default=None,
        description="При include_criteria_bundle — набор маршрутов по критериям",
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
    criteria_bundle: Optional[Dict[str, List["RouteResponse"]]] = Field(
        default=None,
        description="Сравнение критериев (если было в запросе)",
    )


# ── Подблоки ответа ──────────────────────────────────────────────


class ElevationMetrics(BaseModel):
    """Метрики рельефа маршрута."""

    climb_m: float = Field(..., description="Суммарный набор высоты (м)")
    descent_m: float = Field(..., description="Суммарный спуск (м)")
    max_gradient_pct: float = Field(
        ...,
        description=(
            "Макс. уклон по сегментам маршрута (%): |Δh|/длина на рёбрах длиннее порога, "
            "без жёсткого потолка 49%; единичные выбросы DEM приглушаются"
        ),
    )
    avg_gradient_pct: float = Field(
        ...,
        description="Средний |уклон| по тем же валидным сегментам (%)",
    )
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


class CombinedCostBreakdown(BaseModel):
    """Разложение комбинированной стоимости (для heat_stress)."""

    physical: float = 0.0
    heat_effective: float = 0.0
    heat_raw: float = 0.0
    stress: float = 0.0
    stress_segment: float = 0.0
    stress_intersection: float = 0.0
    turn_penalty: float = 0.0
    heat_context_multiplier: float = Field(
        default=1.0,
        description="κ: сезон и/или температура",
    )
    weather_multipliers: Optional[Dict[str, float]] = Field(
        default=None,
        description="Погодные множители (если режим погоды включён)",
    )


class RoutingContextMeta(BaseModel):
    """Служебный контекст расчёта для воспроизводимости и диплома."""

    time_slot: str = ""
    season: str = "summer"
    routing_profile: str = ""
    criterion: str = Field(
        default="",
        description="Служебная метка режима построения (например heat, full)",
    )
    air_temperature_c: Optional[float] = None
    heat_context_multiplier: float = 1.0
    thermal_model_proxy: bool = Field(
        default=False,
        description="True если использован упрощённый прокси (мало спутниковых данных)",
    )


class WeatherSnapshotValues(BaseModel):
    """Нормализованные поля погоды в ответе API."""

    temperature_c: float = 20.0
    apparent_temperature_c: Optional[float] = None
    precipitation_mm: float = 0.0
    precipitation_probability: Optional[float] = None
    wind_speed_ms: float = 3.0
    wind_gusts_ms: Optional[float] = None
    wind_direction_deg: Optional[float] = Field(
        default=None,
        description="Направление ветра ° по Open-Meteo: откуда дует (0° — север), не куда",
    )
    cloud_cover_pct: float = 50.0
    humidity_pct: float = 60.0
    shortwave_radiation_wm2: Optional[float] = None
    snowfall_cm_h: float = 0.0
    snow_depth_m: float = 0.0
    weather_code: Optional[int] = None


class WeatherRouteContext(BaseModel):
    """Погода, по которой строился маршрут, и итоговые множители модели."""

    enabled: bool = False
    mode: str = Field(
        default="none",
        description="none | auto | manual",
    )
    use_live_weather: bool = False
    weather_time: Optional[str] = Field(
        default=None,
        description="ISO 8601 — момент, для которого взята погода",
    )
    source: str = Field(default="", description="open_meteo | manual | none | …")
    snapshot: Optional[WeatherSnapshotValues] = None
    multipliers: Dict[str, float] = Field(
        default_factory=dict,
        description="physical, heat, green, stress, surface",
    )
    summary_ru: str = Field(
        default="",
        description="Кратко: как погода повлияла на веса",
    )
    heat_continuous: bool = Field(
        default=False,
        description="Непрерывная тепло-микроклиматическая модель рёбер включена",
    )
    heat_microclimate: Dict[str, float] = Field(
        default_factory=dict,
        description="Коэффициенты и нормированные сигналы (tree_shade_bonus, temp_norm, …)",
    )
    routing_season: str = Field(
        default="",
        description="Эффективный сезон: winter | early_spring | spring_ramp | green_season | late_autumn",
    )
    routing_season_calendar: str = Field(
        default="",
        description="Сезон по календарю (до adaptive-коррекции)",
    )
    routing_season_source: str = Field(
        default="",
        description="Источник эффективного сезона: calendar | adaptive",
    )
    season_green_route_mult: float = Field(
        default=1.0,
        description="Множитель зелёного бонуса к физике по календарю",
    )
    season_tree_heat_route_mult: float = Field(
        default=1.0,
        description="Множитель роли тени деревьев в heat (до весов winter_heat_*)",
    )
    season_stress_route_mult: float = Field(
        default=1.0,
        description="Сезонный множитель к stress_route_regime",
    )
    season_stairs_route_mult: float = Field(
        default=1.0,
        description="Сезонный множитель к критерию stairs",
    )
    season_wind_orientation_route_mult: float = Field(
        default=1.0,
        description="Сезонный множитель к ориентации улицы к ветру",
    )
    stress_route_regime_factor: float = Field(
        default=1.0,
        description="Итоговый глобальный stress-regime для маршрута",
    )
    snow_model_strength: float = Field(
        default=0.0,
        description="Сила зимней snow-модели 0..1",
    )
    snow_export_phys_amp: float = Field(default=1.0, description="Глобальный множитель физики от снега")
    snow_export_stress_amp: float = Field(
        default=1.0, description="Глобальный множитель stress от снега"
    )
    snow_export_surface_amp: float = Field(
        default=1.0, description="Показатель усиления покрытия зимой (экспорт)"
    )
    wind_direction_available: bool = Field(
        default=False,
        description="True если для рёбер использовалась модель ветра с направлением",
    )


class HeatStressMetrics(BaseModel):
    """Метрики тепловой нагрузки и транспортного стресса вдоль маршрута."""

    time_slot: str = Field(default="", description="morning | noon | evening | night")
    season: str = Field(default="summer", description="summer | spring_autumn")
    routing_profile: str = Field(
        default="",
        description="Ключ профиля α/β/γ/δ в config (внутренняя воспроизводимость)",
    )
    total_heat_cost: float = Field(
        default=0.0,
        description="Суммарная тепловая стоимость с учётом κ (сезон/T)",
    )
    heat_cost_raw: float = Field(
        default=0.0,
        description="Тепловая стоимость без κ (как в графе)",
    )
    stress_cost_total: float = Field(default=0.0, description="Суммарный stress_cost по рёбрам")
    exposed_high_length_m: float = Field(
        default=0.0,
        description="Длина с высокой экспозицией (интегральный показатель слота)",
    )
    exposed_open_unfavorable_length_m: float = Field(
        default=0.0,
        description="Длина открытых теплонеблагоприятных сегментов (порог экспозиции)",
    )
    avg_exposure_unit: float = Field(
        default=0.0,
        description="Средняя безразмерная экспозиция по длине",
    )
    vegetation_shade_share: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Средневзвешенная доля тени растительности по длине",
    )
    building_shade_share: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Средневзвешенная доля прокси-тени зданий по длине",
    )
    route_open_sky_share: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Средневзвешенная thermal_open_sky_share по длине маршрута",
    )
    route_building_shade_share: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Средневзвешенная thermal_building_shade_share по длине",
    )
    route_covered_share: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Средневзвешенная thermal_covered_share по длине",
    )
    route_bad_wet_surface_share: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Среднее wet_surface_edge_slip_factor по длине (мокрое плохое покрытие)",
    )
    route_winter_harsh_surface_share: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Доля «тяжёлого» зимнего покрытия и лестниц по длине (прокси)",
    )
    route_wind_direction_aware: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="1.0 если ниже учтено направление ветра (direction-aware)",
    )
    route_mean_wind_to_street_angle_deg: float = Field(
        default=0.0,
        ge=0.0,
        le=90.0,
        description="Средняя по длине мин. угол оси улицы к потоку ветра, °",
    )
    route_mean_heat_directional_wind_exp: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Средняя direction-aware ветровая экспозиция (heat)",
    )
    route_mean_heat_building_wind_factor: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Средний множитель экрана зданий для heat при данном ветре",
    )
    route_frac_wind_along_open_hostile: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Доля длины: ветер вдоль + открыто + заметная сила ветра",
    )
    route_frac_wind_cross_building_screen: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Доля длины: ветер поперёк + тень зданий + ветер",
    )
    stress_route_regime_factor: float = Field(
        default=0.0,
        ge=0.0,
        description="Глобальный множитель stress (blend×mults.stress + snow_add)×сезон",
    )
    route_mean_base_route_factor: float = Field(
        default=0.0,
        description="Длина-взвешенное среднее base_route_factor по рёбрам",
    )
    route_mean_slope_weather_factor: float = Field(default=0.0)
    route_mean_surface_weather_factor: float = Field(default=0.0)
    route_mean_green_route_factor: float = Field(default=0.0)
    route_mean_open_sky_weather_factor: float = Field(default=0.0)
    route_mean_building_shelter_factor: float = Field(default=0.0)
    route_mean_covered_shelter_factor: float = Field(default=0.0)
    route_mean_stress_weather_factor: float = Field(default=0.0)
    route_mean_stairs_weather_factor: float = Field(default=0.0)
    route_mean_wind_orientation_factor: float = Field(default=0.0)
    avg_stress_lts: float = Field(default=0.0, description="Средний уровень стресса (1–4)")
    max_stress_lts: float = Field(default=0.0, description="Максимальный стресс на сегменте")
    high_stress_length_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Доля длины с LTS ≥ порога (≈3)",
    )
    high_stress_segments_count: int = Field(
        default=0,
        description="Число рёбер с высоким LTS",
    )
    stressful_intersections_count: int = Field(
        default=0,
        description="Число рёбер со стрессом пересечения выше порога",
    )
    turn_count: int = Field(default=0, description="Число заметных поворотов")
    combined_cost: float = Field(
        default=0.0,
        description="α·physical + β·heat·κ + γ·stress + δ·turn",
    )
    combined_breakdown: Optional[CombinedCostBreakdown] = Field(
        default=None,
        description="Детализация для heat_stress",
    )


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
    variant_note_ru: Optional[str] = Field(
        default=None,
        description="Краткое пояснение режима (например, отличие теплового от зелёного)",
    )
    effect_summary_ru: Optional[str] = Field(
        default=None,
        description="Краткое сравнение с базовым (кратчайшим) маршрутом, без чисел модели",
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
    heat_stress: Optional[HeatStressMetrics] = Field(
        default=None,
        description="Заполняется для критериев heat, stress, heat_stress",
    )
    routing_context: Optional[RoutingContextMeta] = Field(
        default=None,
        description="Слот, сезон, κ и профиль",
    )
    weather: Optional[WeatherRouteContext] = Field(
        default=None,
        description="Погодный контекст и множители (если запрошено)",
    )
    heat_cost_total: float = Field(
        default=0.0,
        description="Дублирует тепловую сумму для аналитики (0 если не применимо)",
    )
    stress_cost_total: float = Field(default=0.0)
    exposed_length_m: float = Field(
        default=0.0,
        description="Длина с высокой тепловой экспозицией",
    )
    building_shade_share: float = Field(default=0.0, ge=0.0, le=1.0)
    route_open_sky_share: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Средняя открытость неба по длине"
    )
    route_building_shade_share: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Средняя тень зданий по длине"
    )
    route_covered_share: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Средняя доля укрытий по длине"
    )
    route_bad_wet_surface_share: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Средний индекс плохого мокрого покрытия по длине",
    )
    route_winter_harsh_surface_share: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Прокси доли тяжёлого зимнего покрытия и лестниц",
    )
    vegetation_shade_share: float = Field(default=0.0, ge=0.0, le=1.0)
    stressful_intersections_count: int = Field(default=0)
    high_stress_segments_count: int = Field(default=0)
    turn_count_analytics: int = Field(
        default=0,
        description="Число значимых поворотов (дублирует heat_stress.turn_count при наличии)",
    )
    route_built_at_utc: Optional[str] = Field(
        default=None,
        description="Момент построения этого варианта на сервере (UTC, ISO 8601); для UI «Построен: …»",
    )


class AlternativesResponse(BaseModel):
    """Несколько вариантов маршрута для одного профиля (обычно 2–3 шт.)."""

    routes: List[RouteResponse] = Field(
        ...,
        description="Список вариантов: оптимальный, зелёный, кратчайший по длине или альтернатива",
        min_length=1,
    )
    criteria_bundle: Optional[Dict[str, List[RouteResponse]]] = Field(
        default=None,
        description="При include_criteria_bundle: сравнение критериев по ключам",
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
