"""
Конфигурация проекта.

Параметры загружаются из переменных окружения (.env файл)
или используются значения по умолчанию.

Профили участников движения (велосипедист / пешеход) определяют
различия в физике, предпочтениях дорог, покрытий и озеленения.
"""

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from dotenv import load_dotenv

    _PACKAGE_ENV = Path(__file__).resolve().parent / ".env"
    # Сначала `.env` в cwd (без перезаписи уже заданных в ОС переменных).
    load_dotenv()
    # Затем `bike_router/.env` с override=True: иначе BIKE_ROUTER_BASE_DIR из
    # пользовательских/системных переменных Windows (например R:\\data) перебивает
    # проектный .env, и тайлы уезжают не в каталог репозитория.
    load_dotenv(_PACKAGE_ENV, override=True)
except ImportError:
    pass


def _env(key: str, default, cast_fn=str):
    """Чтение переменной окружения с приведением типа."""
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return cast_fn(raw)
    except (ValueError, TypeError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    """Чтение булевой переменной окружения."""
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# Профиль участника движения
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModeProfile:
    """Набор параметров для конкретного типа участника движения.

    Физическая модель:
        ``base = mg * length * (nu + gradient)``

    Итоговый вес ребра:
        ``weight = base * surface_coeff * highway_coeff``

    Вес с учётом озеленения:
        ``weight_green = weight * (1 + (green_coeff - 1) * green_sensitivity)``
    """

    key: str
    label: str
    icon: str
    # Физика энергозатратной модели
    mg: float
    nu: float
    min_descent_coeff: float
    max_gradient: float
    # Множитель влияния озеленения (>1 — сильнее реагирует на зелень)
    green_sensitivity: float
    # Предпочтения по типам дорог и покрытий
    highway: Dict[str, float] = field(default_factory=dict)
    surface: Dict[str, float] = field(default_factory=dict)
    # Цвета маршрутов на карте
    color_full: str = "#3498DB"
    color_green: str = "#27AE60"
    # Скоростная модель (для оценки времени)
    base_speed_ms: float = 5.0
    uphill_penalty: float = 5.0
    downhill_bonus: float = 3.0
    stairs_speed_ms: float = 0.3


CYCLIST = ModeProfile(
    key="cyclist",
    label="Велосипедист",
    icon="🚴",
    # mg = 1/nu ≈ 16.667 → при gradient=0 вес ребра = длине
    mg=16.667,
    nu=0.06,
    min_descent_coeff=0.02,
    max_gradient=0.30,
    # Озеленение влияет наравне с физикой; дифференциация — через highway-штрафы
    green_sensitivity=1.0,
    highway={
        "cycleway": 0.3,       # выделенные велодорожки — идеал
        "path": 0.7,           # тропинки — приемлемо
        "living_street": 0.9,  # жилые зоны — тихо
        "residential": 1.0,
        "service": 1.1,
        "unclassified": 1.1,
        "pedestrian": 1.2,     # пешеходные зоны — конфликт с пешеходами
        "tertiary": 1.2,
        "tertiary_link": 1.2,
        "footway": 1.3,        # тротуары — не для велосипеда
        "track": 1.3,
        "secondary": 1.4,      # трафик
        "secondary_link": 1.4,
        "primary": 1.6,        # интенсивный трафик — опасно
        "primary_link": 1.6,
        "steps": 10.0,         # лестницы — непроезжаемо
        "construction": 100.0,
    },
    surface={
        "asphalt": 0.9,
        "concrete": 1.1,
        "concrete:plates": 1.3,
        "paved": 1.2,
        "paving_stones": 1.4,
        "sett": 1.5,
        "compacted": 1.5,
        "fine_gravel": 1.7,
        "gravel": 2.0,
        "unpaved": 2.5,
        "dirt": 3.0,
        "ground": 3.0,
        "sand": 3.5,
        "grass": 4.0,
    },
    color_full="#3498DB",
    color_green="#27AE60",
    base_speed_ms=5.0,       # 18 км/ч
    uphill_penalty=5.0,
    downhill_bonus=3.0,
    stairs_speed_ms=0.3,     # нужно вести велосипед
)

PEDESTRIAN = ModeProfile(
    key="pedestrian",
    label="Пешеход",
    icon="🚶",
    # mg = 1/nu = 25.0 → при gradient=0 вес ребра = длине
    mg=25.0,
    nu=0.04,
    min_descent_coeff=0.01,
    # Пешеход преодолевает крутые подъёмы (и лестницы)
    max_gradient=0.50,
    # Пешеходу важнее тень, зелень, комфорт среды
    green_sensitivity=1.3,
    highway={
        "footway": 0.5,        # тротуары — идеал
        "pedestrian": 0.5,     # пешеходные зоны
        "path": 0.7,
        "living_street": 0.8,
        "residential": 1.0,
        "service": 1.1,
        "unclassified": 1.2,
        "cycleway": 1.3,       # велодорожка — не для пешехода
        "track": 1.4,
        "tertiary": 1.5,       # опасность при переходе
        "tertiary_link": 1.5,
        "secondary": 1.8,
        "secondary_link": 1.8,
        "steps": 2.0,          # лестницы — утомительно, но проходимо
        "primary": 2.5,        # очень опасный переход
        "primary_link": 2.5,
        "construction": 100.0,
    },
    surface={
        "asphalt": 0.9,
        "concrete": 1.0,
        "concrete:plates": 1.1,
        "paved": 1.0,
        "paving_stones": 1.1,
        "sett": 1.2,
        "compacted": 1.2,
        "fine_gravel": 1.3,
        "gravel": 1.5,
        "unpaved": 1.8,
        "grass": 1.8,          # трава — проходимо для пешехода
        "dirt": 2.0,
        "ground": 2.0,
        "sand": 2.5,
    },
    color_full="#E67E22",
    color_green="#8E44AD",
    base_speed_ms=1.39,      # 5 км/ч
    uphill_penalty=3.0,
    downhill_bonus=1.5,
    stairs_speed_ms=0.5,
)

PROFILES = [CYCLIST, PEDESTRIAN]


# ---------------------------------------------------------------------------
# Профиль предпочтений маршрутизации (тепло / безопасность / баланс)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutingPreferenceProfile:
    """Коэффициенты комбинированной стоимости:

    ``cost = alpha * physical + beta * heat + gamma * stress + delta * turn``

    ``physical`` — существующий ``weight_{mode}_full``; ``heat``/``stress`` —
    безразмерные штрафы на ребро (уже с длиной); ``turn`` — штраф за резкий
    поворот между рёбрами (масштабируется в :mod:`services.routing`).
    """

    key: str
    label: str
    alpha: float
    beta: float
    gamma: float
    delta: float


ROUTING_PREFERENCE_PROFILES: Dict[str, RoutingPreferenceProfile] = {
    "balanced": RoutingPreferenceProfile(
        key="balanced",
        label="Сбалансированный",
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        delta=1.0,
    ),
    "safe": RoutingPreferenceProfile(
        key="safe",
        label="Приоритет безопасности",
        alpha=1.0,
        beta=0.7,
        gamma=1.4,
        delta=1.2,
    ),
    "cool": RoutingPreferenceProfile(
        key="cool",
        label="Приоритет теплового комфорта",
        alpha=1.0,
        beta=1.5,
        gamma=0.8,
        delta=0.9,
    ),
    "sport": RoutingPreferenceProfile(
        key="sport",
        label="Скорость / меньше штрафа за стресс",
        alpha=1.0,
        beta=0.6,
        gamma=0.5,
        delta=0.5,
    ),
    # Тепловой маршрут: доминирует weight_*_full; тепло — корректирующий член (не «ещё один green»).
    "thermal_physical_base": RoutingPreferenceProfile(
        key="thermal_physical_base",
        label="Тепло на физической базе",
        alpha=1.0,
        beta=0.58,
        gamma=0.08,
        delta=0.72,
    ),
    # Стресс-маршрут на той же физической базе, что и full (не чистый stress_cost).
    "stress_physical_base": RoutingPreferenceProfile(
        key="stress_physical_base",
        label="Безопасность на физической базе",
        alpha=1.0,
        beta=0.12,
        gamma=0.94,
        delta=1.05,
    ),
    # Тепло+безопасность: явно выше вес тепла, ниже стресса, чем у balanced — меньше совпадений со stress.
    "heat_stress_physical_base": RoutingPreferenceProfile(
        key="heat_stress_physical_base",
        label="Тепло и безопасность на физической базе",
        alpha=1.0,
        beta=1.45,
        gamma=0.46,
        delta=0.92,
    ),
}


def routing_preference_profile(key: str) -> RoutingPreferenceProfile:
    """Профиль по ключу; неизвестный ключ → balanced."""
    return ROUTING_PREFERENCE_PROFILES.get(
        (key or "").strip().lower(),
        ROUTING_PREFERENCE_PROFILES["balanced"],
    )


# ---------------------------------------------------------------------------
# Временные слоты и тепловая модель (номинальное азимутальное солнце, ° от севера)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeSlotDef:
    key: str
    label: str
    hour_start: int  # включительно [0..23]
    hour_end: int  # не включительно; если start>end — интервал через полночь
    sun_azimuth_deg: float  # упрощённо для средних широт (~53°N, лето)
    insolation_scale: float  # 0..1 — общая интенсивность для слота


TIME_SLOTS: Tuple[TimeSlotDef, ...] = (
    TimeSlotDef(
        "morning",
        "Утро",
        6,
        11,
        sun_azimuth_deg=105.0,
        insolation_scale=0.55,
    ),
    TimeSlotDef(
        "noon",
        "Полдень",
        11,
        16,
        sun_azimuth_deg=195.0,
        insolation_scale=1.0,
    ),
    TimeSlotDef(
        "evening",
        "Вечер",
        16,
        21,
        sun_azimuth_deg=285.0,
        insolation_scale=0.65,
    ),
    TimeSlotDef(
        "night",
        "Ночь / низкая инсоляция",
        21,
        6,
        sun_azimuth_deg=180.0,
        insolation_scale=0.12,
    ),
)


def time_slot_for_hour(hour: int) -> TimeSlotDef:
    """Выбор слота по часу локального времени [0..23]."""
    h = int(hour) % 24
    for slot in TIME_SLOTS:
        if slot.hour_start < slot.hour_end:
            if slot.hour_start <= h < slot.hour_end:
                return slot
        else:
            # через полночь (night: 21..6)
            if h >= slot.hour_start or h < slot.hour_end:
                return slot
    return TIME_SLOTS[1]


def time_slot_key_for_hour(hour: int) -> str:
    return time_slot_for_hour(hour).key


# Масштабы безразмерных тепло/стресс штрафов относительно типичного physical weight
HEAT_COST_SCALE: float = _env("HEAT_COST_SCALE", 0.35, float)
STRESS_COST_SCALE: float = _env("STRESS_COST_SCALE", 2.5, float)
# Базовый штраф за поворот (масштабируется delta из RoutingPreferenceProfile)
TURN_PENALTY_BASE: float = _env("TURN_PENALTY_BASE", 8.0, float)
TURN_ANGLE_THRESHOLD_DEG: float = _env("TURN_ANGLE_THRESHOLD_DEG", 40.0, float)

# Версия тепло/стресс слоя для инвалидации кэша графа
HEAT_STRESS_MODEL_VERSION: str = _env("HEAT_STRESS_MODEL_VERSION", "1", str)

# Верхняя граница **отображаемого** уклона (доля 0–1). На рёбрах графа ``gradient``
# клипуется до ``max(p.max_gradient) == 0.50``, из‑за чего «макс. уклон» в UI часто
# был ровно 50%. Для статистики и подсказок используется ``gradient_raw`` и
# дополнительное ограничение ниже 50%, чтобы не показывать артефакт клипа.
MAX_ROUTE_GRADIENT_DISPLAY = 0.49


# ---------------------------------------------------------------------------
# Настройки приложения
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    """Основные настройки маршрутизации."""

    # --- Конечные точки маршрута (для CLI-демо) ---
    start_lat: float = _env("START_LAT", 53.186243, float)
    start_lon: float = _env("START_LON", 50.088031, float)
    end_lat: float = _env("END_LAT", 53.195334, float)
    end_lon: float = _env("END_LON", 50.124063, float)
    buffer: float = _env("BUFFER", 0.003, float)
    # При GRAPH_CORRIDOR_MODE: база для метаданных/fingerprint; основная последовательность
    # буферов задаётся в CORRIDOR_BUFFER_EXPAND_SCHEDULE.
    corridor_buffer_meters: float = _env("CORRIDOR_BUFFER_METERS", 400.0, float)
    # При NO_PATH: подряд пробовать буферы коридора (м), напр. 10,100,1000,5000.
    # Пустая строка — одна попытка только с CORRIDOR_BUFFER_METERS (>0), иначе без буфера (BUFFER в °).
    corridor_buffer_expand_schedule: str = field(
        default_factory=lambda: _env(
            "CORRIDOR_BUFFER_EXPAND_SCHEDULE", "10,100,1000,5000", str
        )
    )

    # --- Область покрытия графа (для API) ---
    # Если заданы все четыре AREA_*, граф загружается по ним, а не по
    # start/end + buffer. Это позволяет покрыть весь город/район.
    area_min_lat: float = _env("AREA_MIN_LAT", 0.0, float)
    area_max_lat: float = _env("AREA_MAX_LAT", 0.0, float)
    area_min_lon: float = _env("AREA_MIN_LON", 0.0, float)
    area_max_lon: float = _env("AREA_MAX_LON", 0.0, float)
    # WKT полигона/мультиполигона (координаты lon lat). Если задан — граф OSM режется по нему,
    # значения AREA_* для формы области игнорируются (см. engine.warmup).
    # default_factory: чтение env при каждом Settings(), не при import (как у остальных полей dataclass).
    area_polygon_wkt: str = field(
        default_factory=lambda: _env("AREA_POLYGON_WKT", "", str)
    )
    # Граф и спутниковая зелень только в прямоугольнике между точками запроса ± BUFFER.
    # Работает только если нет AREA_POLYGON_WKT и нет полного AREA_*; иначе игнорируется.
    graph_corridor_mode: bool = _env_bool("GRAPH_CORRIDOR_MODE", False)

    # --- Спутниковые снимки ---
    satellite_zoom: int = _env("SATELLITE_ZOOM", 20, int)
    road_buffer_meters: int = _env("ROAD_BUFFER_METERS", 10, int)
    analyze_corridor: bool = _env_bool("ANALYZE_CORRIDOR", True)
    # True — не качать спутник и не считать зелень по тайлам (заглушки); быстрый старт Docker/CI
    disable_satellite_green: bool = _env_bool("DISABLE_SATELLITE_GREEN", False)
    tile_download_threads: int = _env("TILE_DOWNLOAD_THREADS", 8, int)
    # Сколько тайлов обрабатывать за один проход (скачивание + маски T/G). Меньше — меньше RAM.
    # 0 — без разбиения (как раньше; на огромном полигоне возможен OOM).
    green_tile_batch_size: int = _env("GREEN_TILE_BATCH_SIZE", 4096, int)
    tms_server: str = _env("TMS_SERVER", "google")

    # --- Лимиты маршрутизации ---
    max_route_km: float = _env("MAX_ROUTE_KM", 50.0, float)
    max_snap_distance_m: float = _env("MAX_SNAP_DISTANCE_M", 500.0, float)
    # Макс. относительное удлинение vs маршрут «Оптимальный по энергии» (full) для thermal/stress режимов.
    heat_max_detour_ratio: float = _env("HEAT_MAX_DETOUR_RATIO", 0.42, float)
    stress_max_detour_ratio: float = _env("STRESS_MAX_DETOUR_RATIO", 0.55, float)

    # --- Тепловая маршрутизация: погода × микроклимат рёбер (см. services/routing.effective_edge_components) ---
    heat_hot_tree_bonus_scale: float = _env("HEAT_HOT_TREE_BONUS_SCALE", 1.45, float)
    heat_hot_open_sky_penalty_scale: float = _env(
        "HEAT_HOT_OPEN_SKY_PENALTY_SCALE", 1.55, float
    )
    heat_cold_building_canyon_bonus_scale: float = _env(
        "HEAT_COLD_BUILDING_CANYON_BONUS_SCALE", 1.4, float
    )
    # 0..1: насколько гасить выгоду тени деревьев в «холодном» режиме (выше — сильнее гашение).
    heat_cold_tree_bonus_damping: float = _env(
        "HEAT_COLD_TREE_BONUS_DAMPING", 0.65, float
    )
    # Общий усилитель чувствительности погодных множителей и тепловых поправок по рёбрам.
    heat_weather_response_scale: float = _env("HEAT_WEATHER_RESPONSE_SCALE", 1.35, float)

    # --- Непрерывная тепло-микроклиматическая модель (рёбра, без hot/cold/neutral) ---
    heat_continuous_enable: bool = _env_bool("HEAT_CONTINUOUS_ENABLE", True)
    heat_temp_ref_max: float = _env("HEAT_TEMP_REF_MAX", 30.0, float)
    heat_temp_cool_ref: float = _env("HEAT_TEMP_COOL_REF", 14.0, float)
    heat_temp_cool_range: float = _env("HEAT_TEMP_COOL_RANGE", 10.0, float)
    heat_rain_ref_max: float = _env("HEAT_RAIN_REF_MAX", 3.0, float)
    heat_wind_ref_max: float = _env("HEAT_WIND_REF_MAX", 12.0, float)
    heat_gust_delta_ref_max: float = _env("HEAT_GUST_DELTA_REF_MAX", 10.0, float)

    heat_tree_shade_temp_gain: float = _env("HEAT_TREE_SHADE_TEMP_GAIN", 0.42, float)
    heat_tree_shade_rain_damp: float = _env("HEAT_TREE_SHADE_RAIN_DAMP", 0.35, float)
    heat_tree_shade_cold_damp: float = _env("HEAT_TREE_SHADE_COLD_DAMP", 0.58, float)

    heat_open_sky_temp_gain: float = _env("HEAT_OPEN_SKY_TEMP_GAIN", 0.52, float)
    heat_open_sky_wind_gain: float = _env("HEAT_OPEN_SKY_WIND_GAIN", 0.28, float)
    heat_open_sky_gust_gain: float = _env("HEAT_OPEN_SKY_GUST_GAIN", 0.22, float)
    heat_open_sky_humid_gain: float = _env("HEAT_OPEN_SKY_HUMID_GAIN", 0.18, float)
    heat_apparent_temp_enable: bool = _env_bool("HEAT_APPARENT_TEMP_ENABLE", True)
    heat_apparent_temp_ref: float = _env("HEAT_APPARENT_TEMP_REF", 25.0, float)
    heat_apparent_temp_hot_ref: float = _env("HEAT_APPARENT_TEMP_HOT_REF", 32.0, float)
    heat_apparent_temp_extreme_ref: float = _env(
        "HEAT_APPARENT_TEMP_EXTREME_REF", 38.0, float
    )
    heat_open_sky_apparent_gain: float = _env(
        "HEAT_OPEN_SKY_APPARENT_GAIN", 0.30, float
    )
    heat_open_sky_radiation_gain: float = _env(
        "HEAT_OPEN_SKY_RADIATION_GAIN", 0.25, float
    )
    heat_open_sky_hot_extra_gain: float = _env(
        "HEAT_OPEN_SKY_HOT_EXTRA_GAIN", 0.15, float
    )
    heat_extreme_apparent_threshold: float = _env(
        "HEAT_EXTREME_APPARENT_THRESHOLD", 35.0, float
    )
    heat_extreme_route_beta_gain: float = _env(
        "HEAT_EXTREME_ROUTE_BETA_GAIN", 0.30, float
    )
    heat_extreme_open_route_gain: float = _env(
        "HEAT_EXTREME_OPEN_ROUTE_GAIN", 0.22, float
    )
    heat_extreme_tree_route_gain: float = _env(
        "HEAT_EXTREME_TREE_ROUTE_GAIN", 0.18, float
    )
    heat_tree_shade_apparent_gain: float = _env(
        "HEAT_TREE_SHADE_APPARENT_GAIN", 0.38, float
    )
    heat_tree_shade_radiation_gain: float = _env(
        "HEAT_TREE_SHADE_RADIATION_GAIN", 0.30, float
    )
    heat_tree_shade_extreme_gain: float = _env(
        "HEAT_TREE_SHADE_EXTREME_GAIN", 0.20, float
    )
    heat_building_shade_radiation_gain: float = _env(
        "HEAT_BUILDING_SHADE_RADIATION_GAIN", 0.14, float
    )

    heat_building_shade_wind_gain: float = _env(
        "HEAT_BUILDING_SHADE_WIND_GAIN", 0.32, float
    )
    heat_building_shade_rain_gain: float = _env(
        "HEAT_BUILDING_SHADE_RAIN_GAIN", 0.30, float
    )
    heat_building_shade_humid_gain: float = _env(
        "HEAT_BUILDING_SHADE_HUMID_GAIN", 0.14, float
    )
    heat_building_shade_gust_gain: float = _env(
        "HEAT_BUILDING_SHADE_GUST_GAIN", 0.22, float
    )

    heat_covered_rain_gain: float = _env("HEAT_COVERED_RAIN_GAIN", 0.28, float)
    heat_covered_wind_gain: float = _env("HEAT_COVERED_WIND_GAIN", 0.14, float)
    heat_covered_gust_gain: float = _env("HEAT_COVERED_GUST_GAIN", 0.12, float)

    heat_wet_surface_rain_gain: float = _env("HEAT_WET_SURFACE_RAIN_GAIN", 0.38, float)
    heat_wet_surface_humid_gain: float = _env("HEAT_WET_SURFACE_HUMID_GAIN", 0.22, float)

    heat_edge_k_open: float = _env("HEAT_EDGE_K_OPEN", 0.66, float)
    heat_edge_k_tree: float = _env("HEAT_EDGE_K_TREE", 0.54, float)
    heat_edge_k_building: float = _env("HEAT_EDGE_K_BUILDING", 0.50, float)
    heat_edge_k_covered: float = _env("HEAT_EDGE_K_COVERED", 0.16, float)
    heat_edge_k_wet: float = _env("HEAT_EDGE_K_WET", 0.34, float)
    heat_edge_k_wind: float = _env("HEAT_EDGE_K_WIND", 0.36, float)
    # Усиление дождевого отклика на рёбрах через O/B (rain_norm), без опоры на covered.
    heat_edge_rain_open_mult: float = _env("HEAT_EDGE_RAIN_OPEN_MULT", 0.48, float)
    heat_edge_rain_building_mult: float = _env(
        "HEAT_EDGE_RAIN_BUILDING_MULT", 0.56, float
    )
    heat_edge_rain_wind_exp_mult: float = _env(
        "HEAT_EDGE_RAIN_WIND_EXP_MULT", 0.22, float
    )

    heat_wind_exp_w1: float = _env("HEAT_WIND_EXP_W1", 0.45, float)
    heat_wind_exp_w2: float = _env("HEAT_WIND_EXP_W2", 0.35, float)
    heat_wind_exp_w3: float = _env("HEAT_WIND_EXP_W3", 0.35, float)

    heat_wet_surface_edge_bad_max: float = _env(
        "HEAT_WET_SURFACE_EDGE_BAD_MAX", 0.85, float
    )
    heat_open_wet_synergy: float = _env("HEAT_OPEN_WET_SYNERGY", 0.14, float)

    heat_wind_open_penalty_gain: float = _env("HEAT_WIND_OPEN_GAIN", 0.34, float)
    heat_wind_open_gust_gain: float = _env("HEAT_WIND_OPEN_GUST_EXTRA", 0.24, float)
    # Ветер с направлением (метео ° — откуда дует): вдоль улицы / поперёк и экран зданий.
    heat_wind_along_open_amp: float = _env("HEAT_WIND_ALONG_OPEN_AMP", 0.28, float)
    heat_wind_cross_build_amp: float = _env("HEAT_WIND_CROSS_BUILD_AMP", 0.35, float)
    heat_wind_along_build_damp: float = _env("HEAT_WIND_ALONG_BUILD_DAMP", 0.24, float)
    stress_wind_along_open_amp: float = _env("STRESS_WIND_ALONG_OPEN_AMP", 0.32, float)
    stress_wind_cross_shelter_amp: float = _env(
        "STRESS_WIND_CROSS_SHELTER_AMP", 0.40, float
    )

    heat_edge_factor_min: float = _env("HEAT_EDGE_FACTOR_MIN", 0.65, float)
    heat_edge_factor_max: float = _env("HEAT_EDGE_FACTOR_MAX", 1.75, float)

    heat_coeff_clamp_lo: float = _env("HEAT_COEFF_CLAMP_LO", 0.72, float)
    heat_coeff_clamp_hi: float = _env("HEAT_COEFF_CLAMP_HI", 1.60, float)
    heat_coeff_soft_overflow_gain: float = _env(
        "HEAT_COEFF_SOFT_OVERFLOW_GAIN", 0.18, float
    )
    heat_coeff_hard_hi: float = _env("HEAT_COEFF_HARD_HI", 1.85, float)

    # Погодный stress: доля глобального wm.stress (остальное — edge-specific в routing).
    weather_stress_global_blend: float = _env("WEATHER_STRESS_GLOBAL_BLEND", 0.28, float)
    weather_stress_edge_rain_slip: float = _env("WEATHER_STRESS_EDGE_RAIN_SLIP", 0.22, float)
    weather_stress_edge_wind_open: float = _env("WEATHER_STRESS_EDGE_WIND_OPEN", 0.20, float)
    weather_stress_edge_lts_fast: float = _env("WEATHER_STRESS_EDGE_LTS_FAST", 0.17, float)
    weather_stress_edge_building_shelter: float = _env(
        "WEATHER_STRESS_EDGE_BUILDING_SHELTER", 0.11, float
    )
    weather_stress_edge_factor_min: float = _env(
        "WEATHER_STRESS_EDGE_FACTOR_MIN", 0.82, float
    )
    weather_stress_edge_factor_max: float = _env(
        "WEATHER_STRESS_EDGE_FACTOR_MAX", 1.48, float
    )
    # Физика: дождевой штраф по покрытию на ребре (хорошие покрытия ~1.0).
    weather_phys_wet_penalty_tier0_cap: float = _env(
        "WEATHER_PHYS_WET_PENALTY_TIER0_CAP", 0.016, float
    )
    weather_phys_wet_penalty_tier1_coef: float = _env(
        "WEATHER_PHYS_WET_PENALTY_TIER1_COEF", 0.048, float
    )
    weather_phys_wet_penalty_tier2_coef: float = _env(
        "WEATHER_PHYS_WET_PENALTY_TIER2_COEF", 0.12, float
    )

    # --- Сезон маршрутизации (зелень / тень / снег): календарь и коэффициенты ---
    season_green_ramp_start_day: int = int(
        max(1, round(_env("SEASON_GREEN_RAMP_START_DAY", 10.0, float)))
    )
    season_green_ramp_end_day: int = int(
        max(1, round(_env("SEASON_GREEN_RAMP_END_DAY", 20.0, float)))
    )
    season_early_spring_end_day: int = int(
        max(1, round(_env("SEASON_EARLY_SPRING_END_DAY", 9.0, float)))
    )
    season_green_ramp_start_mult: float = _env(
        "SEASON_GREEN_RAMP_START_MULT", 0.18, float
    )
    season_green_ramp_end_mult: float = _env("SEASON_GREEN_RAMP_END_MULT", 1.0, float)
    season_green_mult_winter: float = _env("SEASON_GREEN_MULT_WINTER", 0.08, float)
    season_green_mult_early_spring: float = _env(
        "SEASON_GREEN_MULT_EARLY_SPRING", 0.18, float
    )
    season_green_mult_green: float = _env("SEASON_GREEN_MULT_GREEN", 1.0, float)
    season_green_mult_late_autumn: float = _env(
        "SEASON_GREEN_MULT_LATE_AUTUMN", 0.42, float
    )
    season_tree_heat_mult_winter: float = _env(
        "SEASON_TREE_HEAT_MULT_WINTER", 0.12, float
    )
    season_tree_heat_mult_early_spring: float = _env(
        "SEASON_TREE_HEAT_MULT_EARLY_SPRING", 0.28, float
    )
    season_tree_heat_mult_spring_ramp: float = _env(
        "SEASON_TREE_HEAT_MULT_SPRING_RAMP", 0.55, float
    )
    season_tree_heat_mult_green: float = _env(
        "SEASON_TREE_HEAT_MULT_GREEN", 1.0, float
    )
    season_tree_heat_mult_late_autumn: float = _env(
        "SEASON_TREE_HEAT_MULT_LATE_AUTUMN", 0.45, float
    )
    snow_spring_ramp_strength: float = _env("SNOW_SPRING_RAMP_STRENGTH", 0.88, float)
    snow_default_strength: float = _env("SNOW_DEFAULT_STRENGTH", 0.9, float)
    snow_green_season_strength_floor: float = _env(
        "SNOW_GREEN_SEASON_STRENGTH_FLOOR", 0.06, float
    )
    snow_green_season_depth_on_m: float = _env(
        "SNOW_GREEN_SEASON_DEPTH_ON_M", 0.02, float
    )
    snow_green_season_fresh_on_cm_h: float = _env(
        "SNOW_GREEN_SEASON_FRESH_ON_CM_H", 0.25, float
    )
    snow_depth_tier0_max_m: float = _env("SNOW_DEPTH_TIER0_MAX_M", 0.02, float)
    snow_depth_tier1_max_m: float = _env("SNOW_DEPTH_TIER1_MAX_M", 0.05, float)
    snow_depth_tier2_max_m: float = _env("SNOW_DEPTH_TIER2_MAX_M", 0.10, float)
    snow_depth_mult_tier0: float = _env("SNOW_DEPTH_MULT_TIER0", 1.0, float)
    snow_depth_mult_tier1: float = _env("SNOW_DEPTH_MULT_TIER1", 1.06, float)
    snow_depth_mult_tier2: float = _env("SNOW_DEPTH_MULT_TIER2", 1.14, float)
    snow_depth_mult_tier3: float = _env("SNOW_DEPTH_MULT_TIER3", 1.28, float)
    snow_fresh_tier0_max_cm_h: float = _env("SNOW_FRESH_TIER0_MAX_CM_H", 0.2, float)
    snow_fresh_tier1_max_cm_h: float = _env("SNOW_FRESH_TIER1_MAX_CM_H", 1.0, float)
    snow_fresh_tier2_max_cm_h: float = _env("SNOW_FRESH_TIER2_MAX_CM_H", 3.0, float)
    snow_fresh_mult_tier0: float = _env("SNOW_FRESH_MULT_TIER0", 1.0, float)
    snow_fresh_mult_tier1: float = _env("SNOW_FRESH_MULT_TIER1", 1.04, float)
    snow_fresh_mult_tier2: float = _env("SNOW_FRESH_MULT_TIER2", 1.12, float)
    snow_fresh_mult_tier3: float = _env("SNOW_FRESH_MULT_TIER3", 1.22, float)
    snow_depth_norm_ref_m: float = _env("SNOW_DEPTH_NORM_REF_M", 0.25, float)
    snow_fresh_norm_ref_cm_h: float = _env("SNOW_FRESH_NORM_REF_CM_H", 4.0, float)
    snow_stress_global_add_winter: float = _env(
        "SNOW_STRESS_GLOBAL_ADD_WINTER", 0.035, float
    )
    snow_stress_phys_coupling: float = _env("SNOW_STRESS_PHYS_COUPLING", 0.45, float)
    winter_heat_open_scale_winter: float = _env(
        "WINTER_HEAT_OPEN_SCALE_WINTER", 1.18, float
    )
    winter_heat_open_scale_transition: float = _env(
        "WINTER_HEAT_OPEN_SCALE_TRANSITION", 1.08, float
    )
    winter_heat_wind_scale_winter: float = _env(
        "WINTER_HEAT_WIND_SCALE_WINTER", 1.22, float
    )
    winter_heat_wind_scale_transition: float = _env(
        "WINTER_HEAT_WIND_SCALE_TRANSITION", 1.12, float
    )
    winter_heat_tree_damp_snow_depth: float = _env(
        "WINTER_HEAT_TREE_DAMP_SNOW_DEPTH", 0.38, float
    )
    winter_heat_tree_damp_snow_fresh: float = _env(
        "WINTER_HEAT_TREE_DAMP_SNOW_FRESH", 0.22, float
    )
    weather_stress_edge_snow_open: float = _env(
        "WEATHER_STRESS_EDGE_SNOW_OPEN", 0.18, float
    )
    weather_stress_edge_snow_surface: float = _env(
        "WEATHER_STRESS_EDGE_SNOW_SURFACE", 0.22, float
    )
    weather_stress_edge_snow_stairs: float = _env(
        "WEATHER_STRESS_EDGE_SNOW_STAIRS", 0.16, float
    )
    snow_surface_tier0_amp: float = _env("SNOW_SURFACE_TIER0_AMP", 0.012, float)
    snow_surface_tier1_amp: float = _env("SNOW_SURFACE_TIER1_AMP", 0.055, float)
    snow_surface_tier2_amp: float = _env("SNOW_SURFACE_TIER2_AMP", 0.14, float)
    snow_stairs_ped_base: float = _env("SNOW_STAIRS_PED_BASE", 1.08, float)
    snow_stairs_ped_frozen_boost: float = _env("SNOW_STAIRS_PED_FROZEN_BOOST", 1.12, float)
    snow_stairs_cyclist_base: float = _env("SNOW_STAIRS_CYCLIST_BASE", 1.18, float)
    snow_stairs_cyclist_frozen_boost: float = _env(
        "SNOW_STAIRS_CYCLIST_FROZEN_BOOST", 1.22, float
    )
    snow_phys_cyclist_mult_boost: float = _env(
        "SNOW_PHYS_CYCLIST_MULT_BOOST", 0.12, float
    )
    winter_clearance_high_mitigate: float = _env(
        "WINTER_CLEARANCE_HIGH_MITIGATE", 0.22, float
    )
    # Сезон: calendar_only | adaptive_if_possible (см. services.seasonal.resolve_season_routing_context).
    season_adaptive_mode: str = _env("SEASON_ADAPTIVE_MODE", "calendar_only", str)
    season_adaptive_warm_anomaly_temp_c: float = _env(
        "SEASON_ADAPTIVE_WARM_ANOMALY_TEMP_C", 9.0, float
    )
    season_adaptive_winter_snow_depth_max_m: float = _env(
        "SEASON_ADAPTIVE_WINTER_SNOW_DEPTH_MAX_M", 0.018, float
    )
    season_adaptive_winter_fresh_max_cm_h: float = _env(
        "SEASON_ADAPTIVE_WINTER_FRESH_MAX_CM_H", 0.35, float
    )
    season_adaptive_snow_depth_on_green_m: float = _env(
        "SEASON_ADAPTIVE_SNOW_DEPTH_ON_GREEN_M", 0.04, float
    )
    season_adaptive_fresh_on_green_cm_h: float = _env(
        "SEASON_ADAPTIVE_FRESH_ON_GREEN_CM_H", 0.45, float
    )
    season_adaptive_cold_on_green_c: float = _env(
        "SEASON_ADAPTIVE_COLD_ON_GREEN_C", 1.5, float
    )
    season_adaptive_early_april_warm_c: float = _env(
        "SEASON_ADAPTIVE_EARLY_APRIL_WARM_C", 11.0, float
    )
    # WMO weather_code — маршрутизация (см. services.weather.wmo_weather_code_profile).
    wc_snow_model_strength_amp: float = _env("WC_SNOW_MODEL_STRENGTH_AMP", 0.22, float)
    wc_wet_slip_physical_amp: float = _env("WC_WET_SLIP_PHYSICAL_AMP", 0.08, float)
    wc_mixed_precip_stress_amp: float = _env("WC_MIXED_PRECIP_STRESS_AMP", 0.12, float)
    wc_wet_slip_surface_amp: float = _env("WC_WET_SLIP_SURFACE_AMP", 0.26, float)
    wc_mixed_surface_amp: float = _env("WC_MIXED_SURFACE_AMP", 0.18, float)
    wc_wet_stairs_amp: float = _env("WC_WET_STAIRS_AMP", 0.14, float)
    wc_mixed_snow_stress_amp: float = _env("WC_MIXED_SNOW_STRESS_AMP", 0.16, float)
    winter_clearance_low_amp: float = _env("WINTER_CLEARANCE_LOW_AMP", 0.32, float)
    winter_clearance_low_amp_cyclist: float = _env(
        "WINTER_CLEARANCE_LOW_AMP_CYCLIST", 0.22, float
    )

    # --- Кэш ---
    cache_satellite: bool = _env_bool("CACHE_SATELLITE", True)
    cache_tile_analysis: bool = _env_bool("CACHE_TILE_ANALYSIS", True)
    force_recalculate: bool = _env_bool("FORCE_RECALCULATE", False)
    # green_edges: не доверять pickle, где у всех рёбер нулевая зелень (часто после сбоя тайлов).
    green_edge_reject_all_zero_cache: bool = _env_bool(
        "GREEN_EDGE_REJECT_ALL_ZERO_CACHE", True
    )
    # Персистентный кэш ответов Nominatim (forward/reverse) на диске
    geocode_disk_cache: bool = _env_bool("GEOCODE_DISK_CACHE", True)
    # Кэш JSON ответов POST /alternatives (инвалидация: узлы/рёбра + отпечаток весов)
    route_disk_cache: bool = _env_bool("ROUTE_DISK_CACHE", False)
    # Кэш взвешенных графов коридора (GraphML), только при GRAPH_CORRIDOR_MODE
    corridor_graph_disk_cache: bool = _env_bool("CORRIDOR_GRAPH_DISK_CACHE", True)
    # Шаг сетки (градусы) для расширения bbox коридора и ключа дискового кэша; 0 — только round(..., 6).
    # По умолчанию ~0.001° ≈ 111 м по широте — чаще попадание в один .graphml для близких POST.
    corridor_cache_grid_step_deg: float = _env("CORRIDOR_CACHE_GRID_STEP_DEG", 0.001, float)

    # --- Предкэш арены (не заменяет GRAPH_CORRIDOR_MODE / AREA_POLYGON_WKT; только дисковый граф для ускорения) ---
    precache_area_enabled: bool = _env_bool("PRECACHE_AREA_ENABLED", False)
    # WKT полигона (lon lat), отдельно от AREA_POLYGON_WKT — не переключает fixed-area режим.
    precache_area_polygon_wkt: str = field(
        default_factory=lambda: _env("PRECACHE_AREA_POLYGON_WKT", "", str)
    )
    precache_area_name: str = field(
        default_factory=lambda: _env("PRECACHE_AREA_NAME", "default", str)
    )
    # Сохранять/ожидать graph_green.graphml (со спутником); иначе только phase1 base.
    precache_area_use_green_graph: bool = _env_bool(
        "PRECACHE_AREA_USE_GREEN_GRAPH", True
    )

    # --- Legacy: было base + AUTO_EXPAND_STEP *(...) до AUTO_EXPAND_MAX_METERS;
    # сейчас приоритет у CORRIDOR_BUFFER_EXPAND_SCHEDULE (см. corridor_expand_schedule_meters).
    auto_expand_step_meters: float = _env("AUTO_EXPAND_STEP_METERS", 1000.0, float)
    auto_expand_max_meters: float = _env("AUTO_EXPAND_MAX_METERS", 5000.0, float)
    auto_expand_max_attempts: int = int(
        max(1, round(_env("AUTO_EXPAND_MAX_ATTEMPTS", 5.0, float)))
    )

    # --- Progressive alternatives / TTL job store ---
    alternatives_job_ttl_sec: float = _env("ALTERNATIVES_JOB_TTL_SEC", 1800.0, float)
    progressive_alternatives_enabled: bool = _env_bool(
        "PROGRESSIVE_ALTERNATIVES_ENABLED", False
    )
    progressive_background_variants: str = _env(
        "PROGRESSIVE_BACKGROUND_VARIANTS",
        "green,heat,stress,heat_stress",
        str,
    )
    # --- HTTP retry (Overpass через OSMnx, Nominatim, TMS; см. services/retry.py) ---
    http_retry_max_attempts: int = int(
        max(1, round(_env("HTTP_RETRY_MAX_ATTEMPTS", 4.0, float)))
    )
    http_retry_base_delay_sec: float = _env("HTTP_RETRY_BASE_DELAY_SEC", 0.5, float)
    http_retry_max_delay_sec: float = _env("HTTP_RETRY_MAX_DELAY_SEC", 30.0, float)
    http_retry_jitter: float = _env("HTTP_RETRY_JITTER", 0.25, float)
    # Повторы на одном endpoint Overpass перед переключением на следующий mirror.
    overpass_per_endpoint_retries: int = int(
        max(1, round(_env("OVERPASS_PER_ENDPOINT_RETRIES", 3.0, float)))
    )

    # --- Overpass API (OSMnx: первая загрузка графа) ---
    # При True OSMnx перед каждым запросом опрашивает /status; при ошибке сети из Docker
    # подставляется пауза до 60 с без ваших логов — кажется «зависанием» на «Загрузка дорог из OSM».
    # По умолчанию False; для публичного Overpass вежливее true (если статус стабильно доступен).
    osm_overpass_rate_limit: bool = _env_bool("OSM_OVERPASS_RATE_LIMIT", False)
    osm_requests_timeout: float = _env("OSM_REQUESTS_TIMEOUT", 180.0, float)
    # Первая sub-попытка к каждому Overpass URL — короткий таймаут (сек), затем повторы с OSM_REQUESTS_TIMEOUT
    # и следующие зеркала из OSM_OVERPASS_URLS. Работает и при одном endpoint. 0 — везде только OSM_REQUESTS_TIMEOUT.
    osm_overpass_first_attempt_timeout: float = _env(
        "OSM_OVERPASS_FIRST_TIMEOUT", 30.0, float
    )
    # Несколько URL через запятую — последовательные попытки (без параллельного fan-out).
    # Приоритет: OSM_OVERPASS_URLS → OSM_OVERPASS_URL → https://overpass-api.de/api
    osm_overpass_urls: str = _env("OSM_OVERPASS_URLS", "", str)
    # Пусто — один из списка выше или дефолт. Обратная совместимость с одним URL.
    osm_overpass_url: str = _env("OSM_OVERPASS_URL", "", str)

    # --- Пути ---
    base_dir: str = field(default="")

    @property
    def start_coords(self) -> Tuple[float, float]:
        """Координаты старта (lat, lon)."""
        return (self.start_lat, self.start_lon)

    @property
    def end_coords(self) -> Tuple[float, float]:
        """Координаты финиша (lat, lon)."""
        return (self.end_lat, self.end_lon)

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.base_dir, "cache")

    @property
    def osmnx_cache_dir(self) -> str:
        return os.path.join(self.base_dir, "osmnx_cache")

    @property
    def srtm_local_cache_dir(self) -> str:
        """Каталог для HGT библиотеки ``srtm`` (иначе пишет в ``~/.cache/srtm`` на системном диске)."""
        return os.path.join(self.base_dir, "cache", "srtm_hgt")

    @property
    def has_area_bbox(self) -> bool:
        """True если заданы все четыре границы и прямоугольник невырожден (не через truthy чисел)."""
        a, b, c, d = (
            self.area_min_lat,
            self.area_max_lat,
            self.area_min_lon,
            self.area_max_lon,
        )
        if not all(math.isfinite(x) for x in (a, b, c, d)):
            return False
        return a < b and c < d

    @property
    def area_polygon_wkt_stripped(self) -> str:
        return (self.area_polygon_wkt or "").strip()

    @property
    def has_area_polygon(self) -> bool:
        """True если в .env задан непустой WKT полигона для загрузки графа."""
        w = self.area_polygon_wkt_stripped.upper()
        return bool(w) and (
            w.startswith("POLYGON")
            or w.startswith("MULTIPOLYGON")
        )

    @property
    def use_dynamic_corridor_graph(self) -> bool:
        """Динамический bbox по координатам POST (без фиксированной зоны в .env)."""
        return (
            self.graph_corridor_mode
            and not self.has_area_polygon
            and not self.has_area_bbox
        )

    @property
    def precache_area_polygon_wkt_stripped(self) -> str:
        return (self.precache_area_polygon_wkt or "").strip()

    @property
    def has_precache_area_polygon(self) -> bool:
        w = self.precache_area_polygon_wkt_stripped.upper()
        return bool(w) and (
            w.startswith("POLYGON") or w.startswith("MULTIPOLYGON")
        )

    @property
    def corridor_expand_schedule_meters(self) -> List[float]:
        """Список буферов коридора (м) по возрастанию попыток при NO_PATH.

        Читается из ``CORRIDOR_BUFFER_EXPAND_SCHEDULE`` (числа через запятую).
        Если строка пустая — одна попытка с ``CORRIDOR_BUFFER_METERS``, если оно > 0.
        """
        raw = (self.corridor_buffer_expand_schedule or "").strip()
        out: List[float] = []
        for part in raw.split(","):
            p = part.strip()
            if not p:
                continue
            try:
                v = float(p)
            except ValueError:
                continue
            if v > 0:
                out.append(v)
        if out:
            return out
        b = float(self.corridor_buffer_meters)
        return [b] if b > 0 else []

    def resolved_overpass_endpoints(self) -> List[str]:
        """Цепочка Overpass: зеркало(а) и основной инстанс — по очереди, не параллельно."""
        raw = (self.osm_overpass_urls or "").strip()
        if raw:
            urls: List[str] = []
            for part in raw.split(","):
                u = part.strip().rstrip("/")
                if u:
                    urls.append(u)
            if urls:
                return urls
        one = (self.osm_overpass_url or "").strip().rstrip("/")
        if one:
            return [one]
        return ["https://overpass-api.de/api"]

    def __post_init__(self):
        env_base = os.getenv("BIKE_ROUTER_BASE_DIR", "").strip()
        if env_base:
            self.base_dir = os.path.abspath(env_base)
        elif not self.base_dir:
            # Данные по умолчанию: подкаталог ``bike_router/data`` (а не родитель пакета),
            # чтобы кэш был внутри репозитория: ``data/cache``, ``data/osmnx_cache``.
            self.base_dir = str(Path(__file__).resolve().parent / "data")


# ---------------------------------------------------------------------------
# Глобальные справочники
# ---------------------------------------------------------------------------

DEFAULT_COEFFICIENT: float = 1.0

TMS_SERVERS: Dict[str, str] = {
    "google": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    "esri": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
    ),
}

OSM_HIGHWAY_FILTER: str = (
    '["highway"~"construction|cycleway|footway|living_street|path|pedestrian|'
    "platform|primary|primary_link|residential|secondary|secondary_link|"
    'service|steps|tertiary|tertiary_link|track|unclassified"]'
)

# Увеличивайте при изменении формул весов в ``graph.calculate_weights``,
# ``apply_weights``, подсегментов высот, клипа озеленения и т.п.
ROUTING_ALGO_VERSION: str = _env("ROUTING_ALGO_VERSION", "3", str)


def routing_engine_cache_fingerprint() -> str:
    """Стабильный SHA-256 от параметров профилей и OSM-фильтра.

    Инвалидация дискового кэша маршрутов при смене коэффициентов без смены графа.
    """
    profs = []
    for p in PROFILES:
        d = asdict(p)
        d["highway"] = dict(sorted(d["highway"].items()))
        d["surface"] = dict(sorted(d["surface"].items()))
        profs.append(d)
    pref_profiles = {
        k: asdict(v) for k, v in sorted(ROUTING_PREFERENCE_PROFILES.items())
    }
    slots = [
        {
            "key": s.key,
            "hour_start": s.hour_start,
            "hour_end": s.hour_end,
            "insolation_scale": s.insolation_scale,
            "sun_azimuth_deg": s.sun_azimuth_deg,
            "label": s.label,
        }
        for s in TIME_SLOTS
    ]
    payload = {
        "algo_ver": ROUTING_ALGO_VERSION,
        "default_coefficient": DEFAULT_COEFFICIENT,
        "heat_cost_scale": HEAT_COST_SCALE,
        "heat_hot_tree_bonus_scale": _env("HEAT_HOT_TREE_BONUS_SCALE", 1.45, float),
        "heat_hot_open_sky_penalty_scale": _env(
            "HEAT_HOT_OPEN_SKY_PENALTY_SCALE", 1.55, float
        ),
        "heat_cold_building_canyon_bonus_scale": _env(
            "HEAT_COLD_BUILDING_CANYON_BONUS_SCALE", 1.4, float
        ),
        "heat_cold_tree_bonus_damping": _env(
            "HEAT_COLD_TREE_BONUS_DAMPING", 0.65, float
        ),
        "heat_weather_response_scale": _env(
            "HEAT_WEATHER_RESPONSE_SCALE", 1.35, float
        ),
        "heat_continuous_enable": _env_bool("HEAT_CONTINUOUS_ENABLE", True),
        "heat_temp_ref_max": _env("HEAT_TEMP_REF_MAX", 30.0, float),
        "heat_rain_ref_max": _env("HEAT_RAIN_REF_MAX", 3.0, float),
        "heat_wind_ref_max": _env("HEAT_WIND_REF_MAX", 12.0, float),
        "heat_gust_delta_ref_max": _env("HEAT_GUST_DELTA_REF_MAX", 10.0, float),
        "heat_wind_along_open_amp": _env("HEAT_WIND_ALONG_OPEN_AMP", 0.28, float),
        "heat_wind_cross_build_amp": _env("HEAT_WIND_CROSS_BUILD_AMP", 0.35, float),
        "heat_wind_along_build_damp": _env("HEAT_WIND_ALONG_BUILD_DAMP", 0.24, float),
        "stress_wind_along_open_amp": _env("STRESS_WIND_ALONG_OPEN_AMP", 0.32, float),
        "stress_wind_cross_shelter_amp": _env(
            "STRESS_WIND_CROSS_SHELTER_AMP", 0.40, float
        ),
        "heat_edge_factor_min": _env("HEAT_EDGE_FACTOR_MIN", 0.65, float),
        "heat_edge_factor_max": _env("HEAT_EDGE_FACTOR_MAX", 1.75, float),
        "heat_edge_k_open": _env("HEAT_EDGE_K_OPEN", 0.66, float),
        "heat_edge_k_tree": _env("HEAT_EDGE_K_TREE", 0.54, float),
        "heat_edge_k_building": _env("HEAT_EDGE_K_BUILDING", 0.50, float),
        "heat_edge_k_covered": _env("HEAT_EDGE_K_COVERED", 0.16, float),
        "heat_edge_rain_open_mult": _env("HEAT_EDGE_RAIN_OPEN_MULT", 0.48, float),
        "heat_edge_rain_building_mult": _env(
            "HEAT_EDGE_RAIN_BUILDING_MULT", 0.56, float
        ),
        "heat_edge_rain_wind_exp_mult": _env(
            "HEAT_EDGE_RAIN_WIND_EXP_MULT", 0.22, float
        ),
        "heat_building_shade_rain_gain": _env(
            "HEAT_BUILDING_SHADE_RAIN_GAIN", 0.30, float
        ),
        "heat_tree_shade_temp_gain": _env("HEAT_TREE_SHADE_TEMP_GAIN", 0.42, float),
        "heat_open_sky_temp_gain": _env("HEAT_OPEN_SKY_TEMP_GAIN", 0.52, float),
        "heat_apparent_temp_enable": _env_bool("HEAT_APPARENT_TEMP_ENABLE", True),
        "heat_apparent_temp_ref": _env("HEAT_APPARENT_TEMP_REF", 25.0, float),
        "heat_apparent_temp_hot_ref": _env("HEAT_APPARENT_TEMP_HOT_REF", 32.0, float),
        "heat_apparent_temp_extreme_ref": _env(
            "HEAT_APPARENT_TEMP_EXTREME_REF", 38.0, float
        ),
        "heat_open_sky_apparent_gain": _env(
            "HEAT_OPEN_SKY_APPARENT_GAIN", 0.30, float
        ),
        "heat_open_sky_radiation_gain": _env(
            "HEAT_OPEN_SKY_RADIATION_GAIN", 0.25, float
        ),
        "heat_open_sky_hot_extra_gain": _env(
            "HEAT_OPEN_SKY_HOT_EXTRA_GAIN", 0.15, float
        ),
        "heat_extreme_apparent_threshold": _env(
            "HEAT_EXTREME_APPARENT_THRESHOLD", 35.0, float
        ),
        "heat_extreme_route_beta_gain": _env(
            "HEAT_EXTREME_ROUTE_BETA_GAIN", 0.30, float
        ),
        "heat_extreme_open_route_gain": _env(
            "HEAT_EXTREME_OPEN_ROUTE_GAIN", 0.22, float
        ),
        "heat_extreme_tree_route_gain": _env(
            "HEAT_EXTREME_TREE_ROUTE_GAIN", 0.18, float
        ),
        "heat_tree_shade_apparent_gain": _env(
            "HEAT_TREE_SHADE_APPARENT_GAIN", 0.38, float
        ),
        "heat_tree_shade_radiation_gain": _env(
            "HEAT_TREE_SHADE_RADIATION_GAIN", 0.30, float
        ),
        "heat_tree_shade_extreme_gain": _env(
            "HEAT_TREE_SHADE_EXTREME_GAIN", 0.20, float
        ),
        "heat_building_shade_radiation_gain": _env(
            "HEAT_BUILDING_SHADE_RADIATION_GAIN", 0.14, float
        ),
        "heat_coeff_clamp_lo": _env("HEAT_COEFF_CLAMP_LO", 0.72, float),
        "heat_coeff_clamp_hi": _env("HEAT_COEFF_CLAMP_HI", 1.60, float),
        "heat_coeff_soft_overflow_gain": _env(
            "HEAT_COEFF_SOFT_OVERFLOW_GAIN", 0.18, float
        ),
        "heat_coeff_hard_hi": _env("HEAT_COEFF_HARD_HI", 1.85, float),
        "heat_temp_cool_ref": _env("HEAT_TEMP_COOL_REF", 14.0, float),
        "heat_temp_cool_range": _env("HEAT_TEMP_COOL_RANGE", 10.0, float),
        "heat_tree_shade_cold_damp": _env("HEAT_TREE_SHADE_COLD_DAMP", 0.58, float),
        "weather_stress_global_blend": _env("WEATHER_STRESS_GLOBAL_BLEND", 0.28, float),
        "season_green_mult_winter": _env("SEASON_GREEN_MULT_WINTER", 0.08, float),
        "season_green_ramp_end_day": _env("SEASON_GREEN_RAMP_END_DAY", 20.0, float),
        "snow_depth_mult_tier3": _env("SNOW_DEPTH_MULT_TIER3", 1.28, float),
        "snow_fresh_mult_tier3": _env("SNOW_FRESH_MULT_TIER3", 1.22, float),
        "season_adaptive_mode": _env("SEASON_ADAPTIVE_MODE", "calendar_only", str),
        "wc_snow_model_strength_amp": _env("WC_SNOW_MODEL_STRENGTH_AMP", 0.22, float),
        "wc_wet_slip_surface_amp": _env("WC_WET_SLIP_SURFACE_AMP", 0.26, float),
        "wc_mixed_surface_amp": _env("WC_MIXED_SURFACE_AMP", 0.18, float),
        "winter_clearance_low_amp": _env("WINTER_CLEARANCE_LOW_AMP", 0.32, float),
        "winter_clearance_low_amp_cyclist": _env(
            "WINTER_CLEARANCE_LOW_AMP_CYCLIST", 0.22, float
        ),
        "winter_clearance_high_mitigate": _env(
            "WINTER_CLEARANCE_HIGH_MITIGATE", 0.22, float
        ),
        "heat_stress_model_version": HEAT_STRESS_MODEL_VERSION,
        "osm_highway_filter": OSM_HIGHWAY_FILTER,
        "preference_profiles": pref_profiles,
        "profiles": profs,
        "stress_cost_scale": STRESS_COST_SCALE,
        "surface_resolve": "v2_osm_hw_tt",
        "time_slots": slots,
        "turn_angle_threshold_deg": TURN_ANGLE_THRESHOLD_DEG,
        "turn_penalty_base": TURN_PENALTY_BASE,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
