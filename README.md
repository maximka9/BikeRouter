# Bike Router — веб-сервис маршрутов для велосипедистов и пешеходов

Сервис строит **несколько вариантов** маршрута между двумя точками в заданной области города, используя граф **OpenStreetMap**, **рельеф** (SRTM), теги **highway** / **surface** и оценку **озеленения** (спутниковый анализ коридора ребра или упрощения без тайлов). Состав: **FastAPI** + статический фронтенд (`frontend/`).

## Что умеет сервис

- Профили **велосипедист** и **пешеход** (разные коэффициенты дорог, лестниц, озеленения и оценка времени).
- Эндпоинт **`POST /alternatives`**: до трёх вариантов (энергетический, «зелёный», кратчайший по длине OSM) с геометрией и метриками качества данных.
- Прямой и обратный **геокодинг** через Nominatim (см. ограничения ниже).
- Веб-интерфейс: карта MapLibre, слои лестниц / озеленения / проблемных участков, **ссылка с состоянием** маршрута в query-параметрах.

## Структура проекта

Ниже все пути относительно каталога **`bike_router/`** (в репозитории это обычно `NIR/bike_router` или `/app/bike_router` в Docker).

### Исходный код и конфигурация

| Путь | Назначение |
|------|------------|
| `bike_router/__init__.py` | Версия пакета |
| `bike_router/__main__.py` | Точка входа: `python -m bike_router` → Uvicorn (`BIKE_ROUTER_HOST` / `PORT`) |
| `bike_router/api.py` | FastAPI: маршруты, lifespan, CORS, раздача `frontend/` |
| `bike_router/app.py` | Composition root: `Application`, связывание сервисов |
| `bike_router/engine.py` | `RouteEngine`: warmup, коридор, граф, `compute_route` / `compute_alternatives` |
| `bike_router/config.py` | `Settings`, профили `ModeProfile`, константы OSM/TMS, fingerprint кэша |
| `bike_router/models.py` | Pydantic-модели запросов/ответов и `HealthResponse` |
| `bike_router/exceptions.py` | Доменные ошибки маршрутизации |
| `bike_router/requirements.txt` | Зафиксированные версии зависимостей |
| `bike_router/Dockerfile` | Сборка образа приложения |
| `bike_router/docker-compose.yml` | Сервис, том `/data`, healthcheck |
| `bike_router/.dockerignore` | Исключения при сборке образа |
| `bike_router/.env.example` | Шаблон переменных окружения (копировать в `.env`) |
| `bike_router/.env` | Локальные секреты/настройки (не коммитить) |

### `bike_router/middleware/`

| Путь | Назначение |
|------|------------|
| `bike_router/middleware/__init__.py` | Пакет middleware |
| `bike_router/middleware/request_log.py` | Логирование HTTP (`bike_router.request`) |

### `bike_router/services/`

| Путь | Назначение |
|------|------------|
| `bike_router/services/__init__.py` | Пакет сервисов |
| `bike_router/services/cache.py` | Файловый pickle-кэш (обёртка с версией, атомарная запись) |
| `bike_router/services/corridor_graph_cache.py` | Дисковый кэш графов коридора (GraphML), ключ bbox + fingerprint |
| `bike_router/services/elevation.py` | SRTM, высоты рёбер |
| `bike_router/services/geocoding.py` | Nominatim, дисковый кэш |
| `bike_router/services/graph.py` | OSMnx: загрузка графа, веса, коридорный bbox |
| `bike_router/services/green.py` | Спутниковые маски T/G, коридор M, метрики зелени |
| `bike_router/services/route_cache.py` | Дисковый кэш ответов `POST /alternatives` |
| `bike_router/services/routing.py` | Поиск пути, метрики маршрута, GeoJSON слоёв |
| `bike_router/services/tiles.py` | Загрузка TMS-тайлов, параллельный batch |

### `bike_router/frontend/` (статика UI)

| Путь | Назначение |
|------|------------|
| `bike_router/frontend/index.html` | Главная страница с картой |
| `bike_router/frontend/about.html` | Страница «О системе» |
| `bike_router/frontend/app.js` | Логика карты (MapLibre), запросы API |
| `bike_router/frontend/style.css` | Стили основной страницы |
| `bike_router/frontend/about.css` | Стили `/about` |

### `bike_router/tools/` и документация

| Путь | Назначение |
|------|------------|
| `bike_router/tools/route_scenarios.json` | Набор сценариев для регрессии API |
| `bike_router/tools/run_route_scenarios.py` | Скрипт прогона сценариев против запущенного сервера |
| `bike_router/docs/ROUTE_TEST_MATRIX.md` | Чеклист ручной проверки UX |
| `bike_router/deploy/nginx.example.conf` | Пример конфигурации обратного прокси |

### `bike_router/tests/`

| Путь | Назначение |
|------|------------|
| `bike_router/tests/test_smoke.py` | Дымовые тесты без загрузки OSM |

### Дерево каталогов (кратко)

```
bike_router/
├── __init__.py
├── __main__.py
├── api.py
├── app.py
├── config.py
├── engine.py
├── exceptions.py
├── models.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env.example
├── middleware/
│   ├── __init__.py
│   └── request_log.py
├── services/
│   ├── __init__.py
│   ├── cache.py
│   ├── corridor_graph_cache.py
│   ├── elevation.py
│   ├── geocoding.py
│   ├── graph.py
│   ├── green.py
│   ├── route_cache.py
│   ├── routing.py
│   └── tiles.py
├── frontend/
│   ├── index.html
│   ├── about.html
│   ├── app.js
│   ├── style.css
│   └── about.css
├── tools/
│   ├── route_scenarios.json
│   └── run_route_scenarios.py
├── tests/
│   └── test_smoke.py
├── docs/
│   └── ROUTE_TEST_MATRIX.md
└── deploy/
    └── nginx.example.conf
```

### Данные и кэши на диске (под `BIKE_ROUTER_BASE_DIR`)

Корень данных задаётся **`BIKE_ROUTER_BASE_DIR`** (в Docker compose обычно **`/data`**). Подкаталоги создаются автоматически; в документации ниже `…` = этот корень.

| Путь на диске | Откуда в коде | Содержимое |
|---------------|----------------|------------|
| `{BASE}/cache/tiles/` | `services/cache.py` → `tile_dir()` | JPEG спутниковых тайлов (`{server}_{z}_{x}_{y}.jpg`) |
| `{BASE}/cache/tile_green_masks/` | `services/green.py` | Маски деревьев/травы по тайлу (`*.npz`) |
| `{BASE}/cache/green_edges/` | `services/green.py` | Pickle кэш анализа рёбер по bbox |
| `{BASE}/cache/corridor_graphs/` | `services/corridor_graph_cache.py` | GraphML взвешенных графов коридора |
| `{BASE}/cache/route_alternatives_cache/` | `services/route_cache.py` | JSON кэш `POST /alternatives` |
| `{BASE}/cache/*.pkl` (прочие) | `cache.py` → `get_path()` | Прочие pickle по старым ключам bbox |
| `{BASE}/cache/nominatim_disk/` | `api.py` / геокодинг | Дисковый кэш Nominatim (если включён) |
| `{BASE}/osmnx_cache/` | `services/graph.py` | Кэш загрузок OSMnx |

Локально, если **`BIKE_ROUTER_BASE_DIR`** не задан, корень по умолчанию — **родительский каталог** папки `bike_router` (см. `config.py`), то есть кэши окажутся рядом с `bike_router`, а не внутри неё.

## Быстрый старт (разработка)

Из каталога **NIR** (родительский для пакета `bike_router`):

```bash
pip install -r bike_router/requirements.txt
python -m bike_router
```

Откройте http://127.0.0.1:8000/ — интерфейс. OpenAPI: http://127.0.0.1:8000/docs  

Переменные **`BIKE_ROUTER_HOST`** и **`BIKE_ROUTER_PORT`** задают bind для `python -m bike_router` (по умолчанию **`0.0.0.0:8000`**, чтобы зайти с телефона в той же сети: `http://<IPv4 из ipconfig>:8000`). Только localhost: `BIKE_ROUTER_HOST=127.0.0.1`.

## Запуск в Docker (одна команда)

Перейдите в каталог **`bike_router/`** (где лежит `docker-compose.yml`).

**Первый раз:** скопируйте `.env.example` → `.env` и при необходимости отредактируйте `AREA_*` и порт.

| Действие | Команда |
|----------|---------|
| Собрать образ и запустить в foreground (логи в терминале) | `docker compose up --build` |
| Запустить в фоне | `docker compose up -d --build` |
| Остановить контейнеры (compose из этого каталога) | `docker compose stop` |
| Остановить и удалить контейнеры сети compose | `docker compose down` |
| Остановить и удалить контейнеры **и** том с данными (`bike_router_data`) | `docker compose down -v` |

Сервис: **http://127.0.0.1:8000** (или порт из **`BIKE_ROUTER_PORT`** в `.env`). Данные кэша и OSMnx монтируются в том **`bike_router_data`** → `/data` в контейнере (`BIKE_ROUTER_BASE_DIR`).

### Доступ из интернета: Cloudflare Tunnel (cloudflared)

Чтобы открыть тот же сервис **снаружи** (например, показать ссылку **научному руководителю** без белого IP), можно поднять временный HTTPS-туннель к локальному порту.

1. Запустите Bike Router (локально или Docker на **8000** или своём порте из `.env`).
2. В **отдельном** терминале выполните (путь к `cloudflared.exe` подставьте свой; порт — как у приложения):

```powershell
# Пример: exe лежит в папке Cloudflared рядом с проектом
.\Cloudflared\cloudflared.exe tunnel --url http://127.0.0.1:8000
```

**Где взять ссылку**

- **Режим quick tunnel** (команда выше): публичный HTTPS-адрес **появится в выводе** `cloudflared` в консоли — строка вида `https://xxxx.trycloudflare.com`. Её можно скопировать и отправить; при каждом новом запуске quick tunnel адрес обычно **новый**.
- **Именованный туннель** (постоянный домен в вашей зоне Cloudflare): настраивается в [Cloudflare Zero Trust](https://one.dash.cloudflare.com/) → **Networks** → **Tunnels**; публичный hostname задаётся там и привязывается к сервису на `localhost`.

Убедитесь, что порт в `--url` совпадает с тем, на котором слушает Bike Router (`BIKE_ROUTER_PORT` в `.env` для Docker).

**Долгий `POST /alternatives/start` и ошибка `context canceled` в cloudflared**

Первый расчёт в новом коридоре (Overpass + при необходимости зелень) может занимать **много минут**. В логах quick tunnel (`*.trycloudflare.com`) тогда часто появляется:

`Incoming request ended abruptly: context canceled`

Причины не в «бездействии» самого сервера, а в том, что **по цепочке Cloudflare → cloudflared → localhost** действуют **таймауты на ожидание ответа** на один HTTP-запрос: пока бэкенд считает, ответа ещё нет — соединение могут закрыть **edge или клиент**. Quick tunnel для таких сценариев **ненадёжен**.

Что делать:

1. **Проверять и тяжёлые запросы по `http://127.0.0.1:8000`** (или по IP ПК в LAN) без туннеля.
2. **Прогреть кэш** (один короткий маршрут), затем снова открыть ссылку через туннель — ответ может уложиться в лимит.
3. Использовать **именованный туннель** в [Zero Trust](https://one.dash.cloudflare.com/) и в конфиге задать **`originRequest`** с увеличенными таймаутами к origin, например:

```yaml
ingress:
  - hostname: bike.example.com
    service: http://127.0.0.1:8000
    originRequest:
      connectTimeout: 120s
      tlsTimeout: 120s
```

(имя хоста и путь к credentials — как в мастере создания туннеля; точные ключи см. [документацию Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/configuration/configuration-file/ingress/).)

4. Либо другой туннель/VPN с настраиваемым таймаутом до origin.

Спутниковая зелень по умолчанию **включена** (`DISABLE_SATELLITE_GREEN=false` в коде). Чтобы прогрев не тянул сотни тысяч тайлов, в **`.env`** задайте **узкий** **`AREA_*`** или **`AREA_POLYGON_WKT`**, снизьте **`SATELLITE_ZOOM`** (например 17 вместо 20 — в разы меньше тайлов) либо временно **`DISABLE_SATELLITE_GREEN=true`** (в т.ч. раскомментируйте строку в **`docker-compose.yml`**).

**Важно:** в образе по умолчанию **один** процесс Gunicorn + Uvicorn worker — граф OSM целиком в памяти процесса; увеличение `-w` дублирует потребление RAM.

## Production за пределами Docker

1. **Uvicorn** (как в `python -m bike_router`) или **Gunicorn** + `uvicorn.workers.UvicornWorker` (см. `bike_router/Dockerfile`, `CMD`).
2. **Обратный прокси** (TLS, заголовки `X-Forwarded-*`): пример — `bike_router/deploy/nginx.example.conf`.
3. Постоянный URL: DNS на ваш сервер + HTTPS (Let’s Encrypt и т.п.).
4. Таймаут прокси не меньше **120 с** на первый запрос после холодного старта (загрузка графа).

## Конфигурация и переменные окружения

Подробные комментарии — в **`.env.example`**.

| Тема | Переменные |
|------|------------|
| Каталог кэшей и данных | **`BIKE_ROUTER_BASE_DIR`** (в Docker: `/data`; локально по умолчанию — родитель каталога `bike_router`) |
| Полигон графа | **`AREA_POLYGON_WKT`** — WKT `POLYGON` / `MULTIPOLYGON` (lon lat); если задан, имеет приоритет над `AREA_*` |
| Bbox графа | **`AREA_MIN_LAT`**, **`AREA_MAX_LAT`**, **`AREA_MIN_LON`**, **`AREA_MAX_LON`** — если все заданы и ненулевые, граф строится по прямоугольнику |
| Коридор по запросу | **`GRAPH_CORRIDOR_MODE=true`** — без полигона и без ненулевого `AREA_*`: граф и спутниковая зелень в прямоугольнике между точками запроса; запас по умолчанию **`CORRIDOR_BUFFER_METERS=400`** (метры по широте/долготе); при **`CORRIDOR_BUFFER_METERS=0`** — запас **`BUFFER`** в градусах; кэш графов — **`cache/corridor_graphs/*.graphml`** при **`CORRIDOR_GRAPH_DISK_CACHE=true`** |
| CORS | **`CORS_ALLOW_ORIGINS`**: `*` (по умолчанию, dev) или список origin через запятую для публикации |
| Запасной bbox | **`START_*`**, **`END_*`**, **`BUFFER`** — если нет ни полигона, ни `AREA_*`, и не включён коридор |
| Лимиты | **`MAX_ROUTE_KM`**, **`MAX_SNAP_DISTANCE_M`** |
| Инвалидация кэша маршрутов | **`ROUTING_ALGO_VERSION`** — увеличить после изменения формул весов / логики в `bike_router/services/graph.py`, `bike_router/services/routing.py` и т.д. |
| Дисковый кэш | **`GEOCODE_DISK_CACHE`**, **`ROUTE_DISK_CACHE`**, **`CORRIDOR_GRAPH_DISK_CACHE`** — `true`/`false` |
| Прочие кэши | **`CACHE_SATELLITE`**, **`CACHE_TILE_ANALYSIS`**, **`FORCE_RECALCULATE`** |
| TMS для зелени | **`TMS_SERVER`**, **`SATELLITE_ZOOM`**, **`ROAD_BUFFER_METERS`** (ширина коридора **M** вокруг линии дороги в метрах), **`TILE_DOWNLOAD_THREADS`** (загрузка тайлов и построение масок T/G) |
| Покрытие без `surface` в OSM | Тег OSM при наличии → иначе эвристика **`tracktype`** / **`highway`** → `unknown` (коэфф. 1.0); см. `bike_router/services/surface_resolve.py` |

**Обязательные переменные:** для старта API ни одна не обязательна. Для **production** либо фиксированная зона (**`AREA_*`** / **`AREA_POLYGON_WKT`**), либо **`GRAPH_CORRIDOR_MODE`** без фиксированной области; при Docker задайте **`BIKE_ROUTER_BASE_DIR`**.

## HTTP API (основные endpoint’ы)

| Метод | Путь | Назначение |
|--------|------|------------|
| `GET` | `/` | Главная страница с картой |
| `GET` | `/about` | Страница «О системе» |
| `GET` | `/static/*` | CSS, JS |
| `GET` | `/health` | Состояние, версия, размер графа, fingerprint весов |
| `GET` | `/docs`, `/redoc` | OpenAPI (Swagger / ReDoc) |
| `POST` | `/alternatives` | Несколько вариантов маршрута (основной сценарий UI) |
| `POST` | `/route` | Один маршрут с выбором `mode`: `full` / `green` / `shortest` |
| `GET` | `/geocode` | Прямое геокодирование (query `q`) |
| `GET` | `/reverse-geocode` | Обратное геокодирование (`lat`, `lon`) |

Ответы с ошибками маршрутизации содержат JSON `detail`: `{ "code", "message" }` (например `NO_PATH`, `POINT_OUTSIDE_ZONE`, `ROUTE_TOO_LONG`).

## Профили `cyclist` и `pedestrian`

Задаются в **`bike_router/config.py`** (`ModeProfile`): физическая модель веса рёбер, штрафы по **`highway`** и **`surface`**, чувствительность к озеленению, оценка времени. Пешеход допускает лестницы с меньшим штрафом, сильнее учитывает зелень; велосипедист сильнее избегает лестниц и оживлённых дорог. Точные числа — в коде и на странице «О системе».

## Ограничения OSM и геокодинга

- **OSM** неполон и неравномерен: где нет тега `surface`, используется нейтральный коэффициент — интерфейс показывает долю «N/A» и предупреждения.
- **Граф** ограничен **bbox** сервера; точки вне области дают **`POINT_OUTSIDE_ZONE`**.
- **Nominatim** (публичный): лимит частоты запросов, [политика использования](https://operations.osmfoundation.org/policies/nominatim/); UI вызывает геокодинг только по кнопке / Enter. Для высоких нагрузок — свой инстанс или другой провайдер (см. `GeocodingProvider` в `bike_router/services/geocoding.py`).
- **Маршруты** — модель по статическим данным, не навигация в реальном времени.

**Спутниковая зелень:** по тайлу один раз строятся маски деревьев **T** и травы **G** (NDI/ExG), кэш `cache/tile_green_masks/*.npz`. Для ребра в режиме коридора — маска **M** (`ROAD_BUFFER_METERS`); доли в ответе: `100·Σ(M∩T)/Σ(M)` и `100·Σ(M∩G)/Σ(M)` по сумме пикселей по всем тайлам сегмента. Смена порогов — константа `TILE_VEG_MASK_VERSION` в `bike_router/services/green.py` и/или **`ROUTING_ALGO_VERSION`**.

## Данные OSM и пересборка графа

Граф загружается при старте (`engine.warmup()`), кроме режима **`GRAPH_CORRIDOR_MODE`**, где первая загрузка — по первому запросу маршрута. Свежие правки на openstreetmap.org не подтягиваются до перезапуска и при необходимости очистки кэша OSMnx. Смена **`AREA_*`**, **`AREA_POLYGON_WKT`** или логики загрузки требует перезапуска. После смены **`ROUTING_ALGO_VERSION`** или весов имеет смысл очистить дисковый кэш маршрутов.

## Логирование и диагностика

| Логгер | Содержание |
|--------|------------|
| `bike_router.request` | Каждый HTTP-запрос: метод, путь, статус, `duration_ms` |
| `bike_router.alternatives` | Успех/тип отказа `POST /alternatives`, длительность |
| `bike_router.routing` | **Отказы построения** (`route_rejected`, `alternatives_rejected`) с кодом, профилем, координатами |
| `bike_router` (корневой в `api`) | Геокодинг, прочие сообщения |

## Тестовые сценарии маршрутов

- **30 автоматических сценариев:** `bike_router/tools/route_scenarios.json`, прогон (нужен `requests`):  
  `python bike_router/tools/run_route_scenarios.py --base http://127.0.0.1:8000`  
  (из каталога **NIR**; при фиксированной зоне граф уже в памяти, при **`GRAPH_CORRIDOR_MODE`** первый запрос строит коридор — задайте **`--timeout`** при необходимости.)
- **Чеклист UX и мобильной проверки:** `bike_router/docs/ROUTE_TEST_MATRIX.md`.

## Юнит-тесты (без загрузки OSM)

```bash
cd NIR
python -m unittest discover -s bike_router/tests -p "test*.py" -v
```

## Архитектура (кратко)

HTTP и статика — `bike_router/api.py`; доменная логика — `bike_router/engine.py` и модули в `bike_router/services/`. Полный перечень файлов и каталогов кэша — в разделе **«Структура проекта»** выше. **`GRAPH_CORRIDOR_MODE`** в коде по умолчанию выключен; рекомендуемый сценарий — в **`bike_router/.env.example`**.

Атрибуция карт и данных — в интерфейсе (подпись под картой) и на `/about`.
