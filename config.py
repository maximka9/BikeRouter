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
    # При GRAPH_CORRIDOR_MODE: если > 0 — запас коридора в метрах (раздельно по широте/долготе).
    # По умолчанию 400 м (~как BUFFER=0.003° по широте); 0 — явный откат на BUFFER в градусах.
    corridor_buffer_meters: float = _env("CORRIDOR_BUFFER_METERS", 400.0, float)

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

    # --- Кэш ---
    cache_satellite: bool = _env_bool("CACHE_SATELLITE", True)
    # Отклонять однотонные/серые тайлы перед записью в cache/tiles (иначе зелень → нули)
    tile_validate_for_green: bool = _env_bool("TILE_VALIDATE_FOR_GREEN", True)
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
    # При GRAPH_CORRIDOR_MODE: при старте API один раз построить граф по START/END ± BUFFER (phase1 без тайлов).
    corridor_warmup_prebuild: bool = _env_bool("CORRIDOR_WARMUP_PREBUILD", False)

    # --- Предкэш арены (не заменяет GRAPH_CORRIDOR_MODE / AREA_POLYGON_WKT; только дисковый граф для ускорения) ---
    precache_area_enabled: bool = _env_bool("PRECACHE_AREA_ENABLED", False)
    # WKT полигона (lon lat), отдельно от AREA_POLYGON_WKT — не переключает fixed-area режим.
    precache_area_polygon_wkt: str = field(
        default_factory=lambda: _env("PRECACHE_AREA_POLYGON_WKT", "", str)
    )
    precache_area_name: str = field(
        default_factory=lambda: _env("PRECACHE_AREA_NAME", "default", str)
    )
    # Устарело для эксплуатации: предсборка полигона только офлайн (precache_area). Не включайте в production.
    precache_area_build_on_startup: bool = _env_bool(
        "PRECACHE_AREA_BUILD_ON_STARTUP", False
    )
    # Сохранять/ожидать graph_green.graphml (со спутником); иначе только phase1 base.
    precache_area_use_green_graph: bool = _env_bool(
        "PRECACHE_AREA_USE_GREEN_GRAPH", True
    )

    # --- Авторасширение коридора при NO_PATH (только при CORRIDOR_BUFFER_METERS > 0) ---
    auto_expand_step_meters: float = _env("AUTO_EXPAND_STEP_METERS", 1000.0, float)
    # Верхняя граница суммарной «добавки» к буферу (м); совместно с AUTO_EXPAND_MAX_ATTEMPTS.
    auto_expand_max_meters: float = _env("AUTO_EXPAND_MAX_METERS", 5000.0, float)
    # Максимум попыток с разным eff=base+extra (первая при extra=0). При исчерпании — NO_PATH.
    auto_expand_max_attempts: int = int(
        max(1, round(_env("AUTO_EXPAND_MAX_ATTEMPTS", 5.0, float)))
    )

    # --- Progressive alternatives / TTL job store ---
    alternatives_job_ttl_sec: float = _env("ALTERNATIVES_JOB_TTL_SEC", 1800.0, float)
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
ROUTING_ALGO_VERSION: str = _env("ROUTING_ALGO_VERSION", "1", str)


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
    payload = {
        "algo_ver": ROUTING_ALGO_VERSION,
        "default_coefficient": DEFAULT_COEFFICIENT,
        "osm_highway_filter": OSM_HIGHWAY_FILTER,
        "profiles": profs,
        "surface_resolve": "v2_osm_hw_tt",
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
