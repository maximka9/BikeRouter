"""Оркестратор приложения — composition root: связывает сервисы через DI.

Полный цикл загрузки графа и ответов API выполняет
:class:`~bike_router.engine.RouteEngine` (``warmup``, ``compute_route`` /
``compute_alternatives``). Этот класс только создаёт и предоставляет зависимости.

``green`` и ``graph_builder`` подгружаются лениво (тяжёлый ``geopandas``), чтобы
CLI вроде ``precache_area`` не зависал без вывода на минуты при холодном старте.
"""

from typing import TYPE_CHECKING, Any, Optional

from .config import Settings
from .services.cache import CacheService
from .services.elevation import ElevationService
from .services.routing import RouteService
from .services.tiles import TileService

if TYPE_CHECKING:
    from .services.graph import GraphBuilder
    from .services.green import GreenAnalyzer
    from .services.surface_prediction_store import SurfacePredictionStore


class Application:
    """Связывает сервисы маршрутизации (граф, рельеф, зелень, кэш, тайлы).

    Используется :class:`~bike_router.engine.RouteEngine` и тестами;
    веб-интерфейс обслуживается через ``bike_router.api``.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self.cache = CacheService(self.settings.base_dir)
        self.tiles = TileService(self.cache, self.settings)
        self.elevation = ElevationService(
            srtm_cache_dir=self.settings.srtm_local_cache_dir
        )
        self._green: Optional["GreenAnalyzer"] = None
        self._graph_builder: Optional["GraphBuilder"] = None
        self._surface_prediction_store: Optional[Any] = None
        self.router = RouteService()

    @property
    def green(self) -> "GreenAnalyzer":
        if self._green is None:
            from .services.green import GreenAnalyzer

            self._green = GreenAnalyzer(self.tiles, self.cache, self.settings)
        return self._green

    @property
    def surface_prediction_store(self) -> Any:
        """Runtime CSV Surface AI; ``None`` если выключено в настройках."""
        if not self.settings.surface_ai_runtime_enabled:
            return None
        if self._surface_prediction_store is None:
            from .services.surface_prediction_store import SurfacePredictionStore

            st = SurfacePredictionStore(self.settings)
            st.load()
            self._surface_prediction_store = st
        return self._surface_prediction_store

    @property
    def graph_builder(self) -> "GraphBuilder":
        if self._graph_builder is None:
            from .services.graph import GraphBuilder

            self._graph_builder = GraphBuilder(
                self.elevation,
                self.green,
                self.settings,
                surface_prediction_store=self.surface_prediction_store,
            )
        return self._graph_builder
