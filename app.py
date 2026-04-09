"""Оркестратор приложения — composition root: связывает сервисы через DI.

Полный цикл загрузки графа и ответов API выполняет
:class:`~bike_router.engine.RouteEngine` (``warmup``, ``compute_route`` /
``compute_alternatives``). Этот класс только создаёт и предоставляет зависимости.
"""

from typing import Optional

from .config import Settings
from .services.cache import CacheService
from .services.elevation import ElevationService
from .services.graph import GraphBuilder
from .services.green import GreenAnalyzer
from .services.routing import RouteService
from .services.tiles import TileService


class Application:
    """Связывает сервисы маршрутизации (граф, рельеф, зелень, кэш, тайлы).

    Используется :class:`~bike_router.engine.RouteEngine` и тестами;
    веб-интерфейс обслуживается через ``bike_router.api``.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self.cache = CacheService(self.settings.base_dir)
        self.tiles = TileService(self.cache, self.settings)
        self.elevation = ElevationService()
        self.green = GreenAnalyzer(self.tiles, self.cache, self.settings)
        self.graph_builder = GraphBuilder(
            self.elevation, self.green, self.settings
        )
        self.router = RouteService()
