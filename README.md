# Bike Router — веб-сервис маршрутов для велосипедистов и пешеходов

Сервис строит **несколько вариантов** маршрута между двумя точками в заданной области города, используя граф **OpenStreetMap**, **рельеф** (SRTM), теги **highway** / **surface** и оценку **озеленения** (спутниковый анализ коридора ребра или упрощения без тайлов). Состав: **FastAPI** + статический фронтенд (`frontend/`).

## Репозиторий и публикация

**Публичный репозиторий релиза:** [github.com/maximka9/BikeRouter](https://github.com/maximka9/BikeRouter).

Если разработка ведётся внутри монорепозитория, а каталог проекта лежит по пути `NIR/bike_router/`, обновлять GitHub достаточно **только** для этого remote (не обязательно пушить весь монорепозиторий). В корне Git добавьте remote `bike-router` → `https://github.com/maximka9/BikeRouter.git`, затем из корня репозитория:

```powershell
.\NIR\bike_router\scripts\push_to_github_bike_router.ps1
```

Скрипт выполняет `git subtree split --prefix=NIR/bike_router` и пушит результат в `bike-router:main`.

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
| `bike_router/Dockerfile` | Сборка образа приложения, `ENTRYPOINT` → `docker-entrypoint.sh` |
| `bike_router/docker-compose.yml` | Сервис, bind mount `./data` → `/data`, healthcheck |
| `bike_router/docker-entrypoint.sh` | Только `exec` Gunicorn/Uvicorn; предсборка полигона — офлайн (`precache_area`) |
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
| `bike_router/services/area_graph_cache.py` | Предкэш графа по полигону арены (без смены corridor/fixed-area режима) |
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
| `bike_router/tools/precache_area.py` | CLI предсборки area precache (офлайн) |
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
| `{BASE}/cache/green_edges/` | `services/green.py` | Pickle кэш анализа рёбер по bbox (ключ включает TMS/zoom; схема `green_edges_v6`). Полностью нулевой кэш при большом графе при загрузке отбрасывается — см. `GREEN_EDGE_REJECT_ALL_ZERO_CACHE` |
| `{BASE}/cache/corridor_graphs/` | `services/corridor_graph_cache.py` | GraphML взвешенных графов коридора |
| `{BASE}/cache/route_alternatives_cache/` | `services/route_cache.py` | JSON кэш `POST /alternatives` |
| `{BASE}/cache/*.pkl` (прочие) | `cache.py` → `get_path()` | Прочие pickle по старым ключам bbox |
| `{BASE}/cache/nominatim_disk/` | `api.py` / геокодинг | Дисковый кэш Nominatim (если включён) |
| `{BASE}/osmnx_cache/` | `services/graph.py` | Кэш загрузок OSMnx |
| `{BASE}/cache/area_precache/<hash>/` | `services/area_graph_cache.py` | **Area precache**: OSM + веса по **`PRECACHE_AREA_POLYGON_WKT`**, `graph_base.graphml` / `graph_green.graphml`, `meta.json` |

**Два независимых типа кэша графа** (в конфиге и коде нет параллельных «режимов предсборки при старте API»: только эта пара + офлайн `precache_area`.)

1. **Area precache** — один раз по **полигону** (`PRECACHE_AREA_*`): быстрый разбор маршрутов, если коридор запроса целиком **внутри** этой арены. Файлы — в **`cache/area_precache/`** (см. таблицу выше).
2. **Corridor cache** — при **`GRAPH_CORRIDOR_MODE=true`** и точках **вне** арены (или если bbox не помещается в полигон): граф строится по **прямоугольнику между точками** и кладётся в **`cache/corridor_graphs/*.graphml`** (`corridor_graph_cache.py`). Это другой ключ и другая семантика, чем area precache.

Локально, если **`BIKE_ROUTER_BASE_DIR`** не задан, корень по умолчанию — **`bike_router/data`** (см. `config.py`).

## Быстрый старт (разработка)

Из каталога **NIR** (родительский для пакета `bike_router`):

```bash
pip install -r bike_router/requirements.txt
python -m bike_router
```

Откройте http://127.0.0.1:8000/ — интерфейс. OpenAPI: http://127.0.0.1:8000/docs  

Переменные **`BIKE_ROUTER_HOST`** и **`BIKE_ROUTER_PORT`** задают bind для `python -m bike_router` (по умолчанию **`0.0.0.0:8000`**, чтобы зайти с телефона в той же сети: `http://<IPv4 из ipconfig>:8000`). Только localhost: `BIKE_ROUTER_HOST=127.0.0.1`.

### Windows: «зависает» `import osmnx` или `import pyogrio`

К **Bike Router** это не относится: так бывает у связки **GeoPandas / OSMnx / pyogrio** (внутри — **GDAL**). После `print('start')` тишина означает, что интерпретатор грузит нативные DLL (иногда минуты, иногда бесконечно при конфликте версий).

**Что сделать по порядку:**

1. Убедиться, что в **`PATH`** нет «левого» GDAL (например **OSGeo4W** и одновременно колёса из pip) — временно уберите лишние пути к `gdal*.dll` и перезапустите терминал.
2. Переустановить движок: `pip install --upgrade --force-reinstall pyogrio` (или то же для `osmnx` и `geopandas`).
3. Самый стабильный вариант на Windows — отдельное окружение **conda/mamba** с **conda-forge**: `conda install -c conda-forge python=3.12 geopandas osmnx pyogrio` (там согласованы GDAL и бинарники).
4. Если **`import pyogrio`** зависает, а **`import geopandas`** — нет: тяжёлая подгрузка всё равно может случиться при первом реальном чтении геоданных или при **`import osmnx`** — ориентируйтесь на пункты выше, а не на проект.

## Запуск в Docker

Перейдите в каталог **`bike_router/`** (где лежит `docker-compose.yml`).

**Первый раз:** скопируйте `.env.example` → `.env` и при необходимости отредактируйте `AREA_*` и порт.

### Два официальных режима кэша

1. **Area precache (полигон Самары или другой арены)** — каталог **`cache/area_precache/<hash>/`**: `graph_base.graphml`, при включённой зелени — **`graph_green.graphml`**, если спутниковый анализ **семантически валиден** (`green_quality_state=ok` в **`meta.json`**). Собирается **только офлайн**: `python -m bike_router.tools.precache_area` (локально или `docker compose run --rm …`). API и `docker compose up` **не** выполняют тяжёлую предсборку полигона.
2. **Corridor graph cache** — при **`GRAPH_CORRIDOR_MODE`** и маршруте **вне** полигона precache: граф по bbox между точками ± буфер, дисковый кэш **`cache/corridor_graphs/`** (см. `CORRIDOR_GRAPH_DISK_CACHE`).

Тайлы (`cache/tiles`), маски (`cache/tile_green_masks`), `green_edges/*.pkl` — внутренние слои реализации, не отдельные «режимы эксплуатации».

Кэш (**`cache/`**, OSMnx, тайлы) лежит в **`./data`** на хосте и монтируется в контейнер как **`/data`** (`BIKE_ROUTER_BASE_DIR=/data`). Так можно **заранее** заполнить папку на диске (в т.ч. предсборкой area precache), а затем поднять только API.

### Предсборка area precache (отдельно от сервера)

С тем же `.env`, что и у приложения (в частности **`PRECACHE_AREA_POLYGON_WKT`**, **`PRECACHE_AREA_USE_GREEN_GRAPH`**):

```bash
docker compose run --rm bike-router python -m bike_router.tools.precache_area
```

Результат появится в **`./data/cache/area_precache/...`**. API при старте **не** собирает полигон на диск: только офлайн **`precache_area`**; в **`warmup`** — лишь опциональная предзагрузка уже готового precache в память.

Локально без Docker (чтобы всё лежало на нужном диске, напр. `R:`): из каталога **родителя** пакета (`NIR`, где лежит папка `bike_router`) задайте **`BIKE_ROUTER_BASE_DIR`** на каталог данных (например **`R:\Python\NIR\bike_router\data`**) и выполните **`python -m bike_router.tools.precache_area`**. Тогда **`cache/`** (тайлы, `area_precache`, **SRTM HGT** в `cache/srtm_hgt/`) и **`osmnx_cache/`** создаются под этим путём, а не в образе Docker и не в `C:\Users\…\.cache\srtm`.

### Запуск API

| Действие | Команда |
|----------|---------|
| Собрать образ и запустить в foreground (логи в терминале) | `docker compose up --build` |
| Запустить в фоне | `docker compose up -d --build` |
| Остановить контейнеры (compose из этого каталога) | `docker compose stop` |
| Остановить и удалить контейнеры сети compose | `docker compose down` |

Очистить кэш на диске: удалите содержимое **`bike_router/data/`** (или всю папку) на хосте — это не том Docker, а обычная директория.

Сервис: **http://127.0.0.1:8000** (или порт из **`BIKE_ROUTER_PORT`** в `.env`).

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
| Предкэш арены (area precache) | **`PRECACHE_AREA_ENABLED=true`** и **`PRECACHE_AREA_POLYGON_WKT`** — отдельно от **`AREA_POLYGON_WKT`**: при corridor-режиме, если bbox внутри полигона и **`meta.json`** валиден (**`green_quality_state=ok`** при зелени), граф с диска **`cache/area_precache/<hash>/`**. Сборка только офлайн: **`python -m bike_router.tools.precache_area`**. |
| CORS | **`CORS_ALLOW_ORIGINS`**: `*` (по умолчанию, dev) или список origin через запятую для публикации |
| Запасной bbox | **`START_*`**, **`END_*`**, **`BUFFER`** — если нет ни полигона, ни `AREA_*`, и не включён коридор |
| Лимиты | **`MAX_ROUTE_KM`**, **`MAX_SNAP_DISTANCE_M`** |
| Инвалидация кэша маршрутов | **`ROUTING_ALGO_VERSION`** — увеличить после изменения формул весов / логики в `bike_router/services/graph.py`, `bike_router/services/routing.py` и т.д. |
| Дисковый кэш | **`GEOCODE_DISK_CACHE`**, **`ROUTE_DISK_CACHE`** (опциональный UX-кэш ответов; по умолчанию `false` в `config`), **`CORRIDOR_GRAPH_DISK_CACHE`** — `true`/`false` |
| Прочие кэши | **`CACHE_SATELLITE`**, **`CACHE_TILE_ANALYSIS`** (оставьте **`true`** на большом precache: маски в `cache/tile_green_masks`, агрегация читает с диска и не раздувает RAM), **`FORCE_RECALCULATE`** |
| TMS для зелени | **`TMS_SERVER`**, **`SATELLITE_ZOOM`**, **`ROAD_BUFFER_METERS`** (ширина коридора **M** вокруг линии дороги в метрах), **`TILE_DOWNLOAD_THREADS`**, **`GREEN_TILE_BATCH_SIZE`** (сколько тайлов за один проход загрузки+масок; меньше — меньше RAM на большом полигоне; `0` — всё сразу) |
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

Граф загружается при старте (`engine.warmup()`), кроме режима **`GRAPH_CORRIDOR_MODE`**, где первая загрузка по **новому** bbox коридора — на пути обработки запроса (Overpass + веса + при необходимости тайлы). Свежие правки на openstreetmap.org не подтягиваются до перезапуска и при необходимости очистки кэша OSMnx. Смена **`AREA_*`**, **`AREA_POLYGON_WKT`** или логики загрузки требует перезапуска. После смены **`ROUTING_ALGO_VERSION`** или весов имеет смысл очистить дисковый кэш маршрутов.

В corridor-режиме «холодный старт» без area precache означает: первый **`POST /alternatives`** строит граф по bbox; отдельного прогрева коридора по **`START_*`/`END_*`** из `.env` в коде **нет** (legacy удалён).

### Предкэш арены (`PRECACHE_AREA_*`, каталог `cache/area_precache/`)

Режим **corridor** (`GRAPH_CORRIDOR_MODE=true` без `AREA_*` / `AREA_POLYGON_WKT`) можно дополнить **предсборкой** дорожного графа по выбранному **полигону** (арене): `graph_base.graphml` (phase1: высоты, surface, `weight_*_full`) и при **`PRECACHE_AREA_USE_GREEN_GRAPH=true`** — `graph_green.graphml` (полный спутниковый анализ с теми же `SATELLITE_ZOOM`, `ROAD_BUFFER_METERS`, `ANALYZE_CORRIDOR`, что и в обычном запросе). Это **не** замена corridor-режима и **не** то же самое, что **`AREA_POLYGON_WKT`** (фиксированная зона для всего сервиса).

При **`PRECACHE_AREA_ENABLED=true`**, если bbox коридора **целиком внутри** полигона **`PRECACHE_AREA_POLYGON_WKT`**, fingerprint в **`meta.json`** совпадает с настройками и для зелёной фазы зафиксировано **`green_quality_state=ok`** (или зелень не запрашивается), загружается готовый GraphML **без** Overpass. Иначе — corridor pipeline и **`corridor_graphs`**.

**Рекомендуемый порядок:** (1) собрать area precache в тот же каталог, что у runtime (**`BIKE_ROUTER_BASE_DIR`**); (2) поднять API — в `warmup` только проверка и опциональная предзагрузка **валидного** precache в память.

Сборка на диск (локально, с `PYTHONPATH` на пакет и переменными из `.env`):

```bash
python -m bike_router.tools.precache_area
```

В Docker (тот же `docker-compose.yml` и `./data:/data`):

```bash
docker compose run --rm bike-router python -m bike_router.tools.precache_area
```

Пример полигона — **`bike_router/tools/precache_samara_core.wkt.example`**.

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

HTTP и статика — `bike_router/api.py`; доменная логика — `bike_router/engine.py` и модули в `bike_router/services/`. Полный перечень файлов и каталогов кэша — в разделе **«Структура проекта»** выше. **`GRAPH_CORRIDOR_MODE`** в коде по умолчанию выключен; пример с коридором в **`bike_router/.env.example`** ориентирован на демо/разработку.

**Параллелизм и блокировка графа:** тяжёлая сборка коридора (OSM, расчёт весов, спутник) выполняется **вне** короткой блокировки `threading.Lock` в `RouteEngine`: под lock остаются проверка «текущий коридор уже покрывает запрос?» и атомарная замена активного графа. Запросы с разными ключами коридора могут перекрываться по CPU; для одного и того же bbox по-прежнему действует отдельный **gate** (один строитель на ключ). Это снимает глобальную сериализацию «всех POST на один lock» на время Dijkstra и построения ответа.

**Один процесс:** в образе по умолчанию один worker — граф целиком в памяти процесса; несколько workers Gunicorn умножают RAM (см. раздел про Docker выше). Это осознанная модель для дипломного single-process режима, а не кластерная репликация состояния.

**Алгоритмическое ядро** — `RouteEngine` (`engine.py`) и **`POST /alternatives`** (синхронный расчёт вариантов). Код в `alternatives_jobs.py` и эндпоинты **`/alternatives/start`**, **`/alternatives/job/{id}`** — тонкая HTTP-обёртка (очередь в памяти процесса) для UX; на корректность весов и кэшей графа **не влияют**. Если нужен минимальный контур для анализа или тестов — достаточно **`POST /alternatives`** без progressive-слоя.

**Progressive** остаётся в репозитории как опция фронтенда; при нескольких workers память графа не разделяется — см. ограничение про один worker.

Атрибуция карт и данных — в интерфейсе (подпись под картой) и на `/about`.
