# Changelog

## 3.0.0 - Registration release

- FastAPI HTTP API for building and comparing pedestrian and bicycle route alternatives.
- Multi-criteria routing engine using distance, road class, surface, elevation, greenery, weather, seasonality, heat load and transport stress.
- OpenStreetMap graph preparation, corridor and area graph caching, SRTM elevation processing and route analytics.
- Runtime surface type recovery for unknown road surfaces, with policy checks for prediction safety and persistence.
- Static JavaScript interface with interactive map, route comparison, profile switching and analytics panels.
- Docker deployment configuration and local command-line entry point through `python -m bike_router`.
- Registration support scripts for abstract validation, program size calculation and release archive assembly.
