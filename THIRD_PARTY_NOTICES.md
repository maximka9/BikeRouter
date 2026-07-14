# Third-Party Notices

The deposited author-owned source text excludes installed third-party libraries, external datasets, caches, satellite tiles and `bike_router/frontend/vendor`.

| Component | Version | Purpose | License / terms | Included in registration archive |
|---|---:|---|---|---|
| FastAPI | 0.124.4 | HTTP API framework | MIT | No, installed from dependency lock |
| Uvicorn | 0.38.0 | ASGI server | BSD-3-Clause | No |
| Gunicorn | 23.0.0 | Production process manager | MIT | No |
| Pydantic | 2.12.5 | Request and response models | MIT | No |
| NumPy | 2.3.5 | Numeric processing | BSD-3-Clause | No |
| GeoPandas | 1.1.1 | Geospatial data processing | BSD-3-Clause | No |
| Shapely | 2.1.2 | Geometry operations | BSD-3-Clause | No |
| NetworkX | 3.5 | Graph algorithms | BSD-3-Clause | No |
| OSMnx | 2.0.7 | OpenStreetMap graph loading | MIT | No |
| scikit-learn | 1.6.1 | Surface type models | BSD-3-Clause | No |
| pyarrow | 20.0.0 | Parquet support | Apache-2.0 | No |
| SRTM.py | 0.3.7 | Elevation tile access | MIT | No |
| Requests | 2.32.5 | HTTP client | Apache-2.0 | No |
| Pillow | 12.0.0 | Raster image processing | HPND | No |
| tqdm | 4.67.1 | Progress output | MPL-2.0 / MIT | No |
| python-dotenv | 1.2.1 | Local environment loading | BSD-3-Clause | No |
| prometheus-client | 0.21.1 | Metrics endpoint | Apache-2.0 | No |
| openpyxl | 3.1.5 | Experiment report workbooks | MIT | No |
| MapLibre GL JS | local vendor copy | Browser map rendering | BSD-3-Clause | No, excluded from deposited source archive |
| OpenStreetMap | external data | Road graph data | ODbL | Data not included |
| SRTM-compatible elevation data | external data | Elevation and slope values | Source-specific public data terms | Data not included |
| Satellite tile provider | external data | Optional greenery and surface features | Must permit machine analysis and caching | Data not included |

The default `.env.example` disables satellite analysis and does not name a satellite tile provider. A deployment that enables satellite-based analysis must document the selected provider, license, attribution and caching rights separately.
