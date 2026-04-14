"""Сервис анализа озеленения по спутниковым снимкам."""

import hashlib
import json
import logging
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional, Set, Tuple

import geopandas as gpd
import numpy as np

from ..config import ROUTING_ALGO_VERSION, Settings
from .cache import CacheService
from .tiles import TileService

logger = logging.getLogger(__name__)

# Пороги NDI/ExG в compute_vegetation_masks; при изменении — увеличить для сброса кэша .npz
# v2: после валидации тайлов в tiles.py (серые placeholder не кэшируются)
TILE_VEG_MASK_VERSION = "ndisexg_v2_tileval"
# Версия pickle green_edges: смена инвалидирует старые файлы (в т.ч. «залипшие» нули).
GREEN_EDGES_CACHE_SCHEMA = "green_edges_v5"

try:
    from PIL import Image, ImageDraw

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class GreenAnalyzer:
    """Озеленение по тайлам: маски **T** (деревья) и **G** (трава) считаются один раз на тайл.

    Для коридора **M** вокруг дороги доли в процентах:
    ``P_trees = 100·Σ(M∩T)/Σ(M)``, ``P_grass = 100·Σ(M∩G)/Σ(M)`` по сумме пикселей
    по всем тайлам ребра (не среднее по тайлам).
    """

    def __init__(
        self,
        tile_service: TileService,
        cache_service: CacheService,
        settings: Settings,
    ) -> None:
        self._tiles = tile_service
        self._cache = cache_service
        self._settings = settings
        self._satellite_warning: Optional[str] = None
        # LRU при чтении масок с диска в цикле по рёбрам (без гигабайтного dict в RAM)
        self._tile_mask_read_lru: OrderedDict[
            Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]
        ] = OrderedDict()
        self._tile_mask_read_lru_max: int = 8192

    def consume_satellite_warning(self) -> Optional[str]:
        """Одноразово отдать предупреждение о неполных тайлах (режим green в ответе)."""
        w = self._satellite_warning
        self._satellite_warning = None
        return w

    # ==================================================================
    # Анализ изображений
    # ==================================================================

    @staticmethod
    def compute_vegetation_masks(img: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Булевы маски деревьев **T** и травы **G** (H, W), взаимно не пересекаются."""
        if img is None:
            return None
        if img.mode != "RGB":
            img = img.convert("RGB")

        px = np.array(img, dtype=np.float32)
        r, g, b = px[:, :, 0], px[:, :, 1], px[:, :, 2]

        ndi = (g - r) / (r + g + 0.001)
        exg = g - (r + b) / 2.0
        brightness = (r + g + b) / 3.0
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        saturation = (max_rgb - min_rgb) / (max_rgb + 0.001)

        base_green = (
            (ndi > 0.01)
            & (exg > 1)
            & (g > r)
            & (g > 30)
            & (brightness > 5)
            & (brightness < 250)
        )

        trees_mask = base_green & (
            (brightness < 120)
            | ((brightness < 160) & (ndi > 0.05) & (saturation > 0.1))
        )
        grass_mask = base_green & ~trees_mask & (
            (brightness >= 60) & (brightness < 220) & (ndi > 0.01)
        )
        return trees_mask, grass_mask

    @staticmethod
    def analyze_image(
        img: Any,
        mask: Optional[np.ndarray] = None,
        return_details: bool = False,
    ) -> dict | float:
        """Доли зелени в %: при маске **M** — ``100·sum(M∩T)/sum(M)`` и аналогично трава."""
        empty: dict = {"trees": 0.0, "grass": 0.0, "total": 0.0}
        vm = GreenAnalyzer.compute_vegetation_masks(img)
        if vm is None:
            return empty if return_details else 0.0
        trees_mask, grass_mask = vm

        if mask is not None:
            mb = mask > 0
            denom = float(mb.sum())
            if denom <= 0:
                return empty if return_details else 0.0
            tp = float((trees_mask & mb).sum()) / denom * 100.0
            gp = float((grass_mask & mb).sum()) / denom * 100.0
        else:
            denom = float(trees_mask.size)
            if denom <= 0:
                return empty if return_details else 0.0
            tp = float(trees_mask.sum()) / denom * 100.0
            gp = float(grass_mask.sum()) / denom * 100.0

        if return_details:
            return {"trees": tp, "grass": gp, "total": tp + gp}
        return tp + gp

    @staticmethod
    def create_corridor_mask(
        edge_geom: Any,
        tile_bounds: Tuple[float, ...],
        img_size: Tuple[int, int],
        buffer_meters: int,
    ) -> np.ndarray:
        """Маска коридора вокруг линии дороги (для попиксельного анализа)."""
        if not PIL_AVAILABLE:
            return np.zeros((img_size[1], img_size[0]), dtype=np.uint8)

        min_lon, min_lat, max_lon, max_lat = tile_bounds
        w, h = img_size

        def _to_pixel(lon: float, lat: float) -> Tuple[int, int]:
            px = int((lon - min_lon) / (max_lon - min_lon) * w)
            py = int((max_lat - lat) / (max_lat - min_lat) * h)
            return px, py

        mask_img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask_img)

        lat_c = (min_lat + max_lat) / 2
        m_per_deg = 111_000 * np.cos(np.radians(lat_c))
        tw = (max_lon - min_lon) * m_per_deg
        th = (max_lat - min_lat) * 111_000
        if tw > 0 and th > 0:
            mpp = max(tw / w, th / h)
            lw = int((buffer_meters * 2) / mpp)
            lw = max(5, min(lw, min(w, h) // 3))
        else:
            lw = 20

        def _draw_line_coords(coords: list) -> None:
            if len(coords) < 2:
                return
            pixels = [_to_pixel(lon, lat) for lon, lat in coords]
            for i in range(len(pixels) - 1):
                draw.line(
                    [pixels[i], pixels[i + 1]], fill=255, width=lw
                )

        gt = getattr(edge_geom, "geom_type", None)
        if gt == "LineString":
            _draw_line_coords(list(edge_geom.coords))
        elif gt == "MultiLineString":
            for part in edge_geom.geoms:
                _draw_line_coords(list(part.coords))

        return np.array(mask_img)

    # ==================================================================
    # Пакетный спутниковый анализ
    # ==================================================================

    def calculate_satellite_batch(
        self, edges_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Пакетный расчёт зелени для каждого ребра по космоснимкам.

        1. Определяет покрывающие ребро TMS-тайлы.
        2. Строит маску коридора вокруг дороги.
        3. Анализирует зелень (NDI / ExG).
        4. Разделяет на **деревья** и **траву**.
        """
        zoom = self._settings.satellite_zoom
        buf_m = self._settings.road_buffer_meters
        buf_deg = buf_m / 111_000.0

        logger.info(
            "Анализ зелени (космоснимки): zoom=%d, буфер=%dм, режим=%s",
            zoom,
            buf_m,
            "коридор" if self._settings.analyze_corridor else "весь тайл",
        )

        if not self._tiles.pil_available:
            return self._fill_empty_green(edges_gdf)

        edge_tiles: dict = {}
        n_edges = len(edges_gdf)
        index_arr = edges_gdf.index.values
        geoms = edges_gdf.geometry.values

        for pos in range(n_edges):
            idx = index_arr[pos]
            geom = geoms[pos]
            if geom is None:
                continue
            b = geom.bounds
            x1, y1 = self._tiles.lat_lon_to_tile(
                b[3] + buf_deg, b[0] - buf_deg, zoom
            )
            x2, y2 = self._tiles.lat_lon_to_tile(
                b[1] - buf_deg, b[2] + buf_deg, zoom
            )
            tiles = [
                (x, y) for x in range(x1, x2 + 1) for y in range(y1, y2 + 1)
            ]
            edge_tiles[idx] = tiles

        edge_cache = self._load_edge_cache(edges_gdf, zoom, buf_m)
        n_tg_fail_total = 0
        invalid_tiles_total = 0

        tiles_needed: Set[Tuple[int, int]] = set()
        for pos in range(n_edges):
            idx = index_arr[pos]
            if idx in edge_cache:
                continue
            tl = edge_tiles.get(idx)
            if tl:
                tiles_needed.update(tl)

        tile_masks: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        if len(edge_cache) >= len(edges_gdf):
            logger.info("Все результаты в кэше (%d рёбер)", len(edge_cache))
        elif not tiles_needed:
            logger.debug("Тайлы не нужны: все рёбра уже в edge-кэше")
        else:
            self._satellite_warning = None
            self._tile_mask_read_lru.clear()
            n_need = len(tiles_needed)
            tiles_list = sorted(tiles_needed)
            batch_cap = self._settings.green_tile_batch_size
            if batch_cap <= 0:
                batch_cap = n_need
            else:
                batch_cap = min(batch_cap, n_need)

            n_tg_cache = n_tg_compute = n_tg_fail = 0
            n_ok_total = 0

            if n_need > batch_cap:
                logger.info(
                    "Спутник: разбиение на пакеты по %d тайлов (GREEN_TILE_BATCH_SIZE), всего %d",
                    batch_cap,
                    n_need,
                )
            if self._settings.cache_tile_analysis:
                logger.info(
                    "Маски T/G пишутся в tile_green_masks/*.npz; в RAM не копится словарь на все тайлы "
                    "(CACHE_TILE_ANALYSIS=true)"
                )

            for start in range(0, n_need, batch_cap):
                chunk_list = tiles_list[start : start + batch_cap]
                chunk_set = set(chunk_list)
                if n_need > batch_cap:
                    logger.info(
                        "Спутник: пакет тайлов %d–%d из %d",
                        start + 1,
                        min(start + batch_cap, n_need),
                        n_need,
                    )
                tile_images, tstats = self._tiles.download_batch(chunk_set, zoom)
                invalid_tiles_total += tstats.invalid_rejected
                n_ok = len(tile_images)
                n_ok_total += n_ok
                if n_ok < len(chunk_set):
                    self._satellite_warning = (
                        f"Загружено спутниковых тайлов {n_ok_total} из {n_need} "
                        f"(пакет {start // batch_cap + 1}); оценка озеленения может быть неполной."
                    )
                    logger.warning("%s", self._satellite_warning)
                part_masks, nc, ncm, nf = self._parallel_build_tile_masks(
                    tile_images, zoom, chunk_set
                )
                if self._settings.cache_tile_analysis:
                    # Маски уже в .npz; не держим сотни тысяч ndarray в одном dict (OOM/своп).
                    pass
                else:
                    tile_masks.update(part_masks)
                n_tg_cache += nc
                n_tg_compute += ncm
                n_tg_fail += nf
                n_tg_fail_total += nf
                del tile_images
                del part_masks

            if n_ok_total < n_need and self._satellite_warning is None:
                self._satellite_warning = (
                    f"Загружено спутниковых тайлов {n_ok_total} из {n_need}; "
                    "оценка озеленения по снимку может быть неполной."
                )
                logger.warning("%s", self._satellite_warning)

            if invalid_tiles_total > 0:
                msg = (
                    f"Отклонено {invalid_tiles_total} тайлов как серые/плоские "
                    "(не годятся для цветового анализа зелени); edge-кэш не сохраняем."
                )
                if self._satellite_warning:
                    self._satellite_warning = f"{self._satellite_warning} {msg}"
                else:
                    self._satellite_warning = msg
                logger.warning("tiles: %s", msg)

            masks_done = (
                len(tile_masks)
                if not self._settings.cache_tile_analysis
                else n_need
            )
            logger.info(
                "Маски T/G: готово %d / %d (tile_green_masks hit %d, miss %d, fail %d); "
                "%% по коридору: 100·Σ(M∩T)/Σ(M)",
                masks_done,
                len(tiles_needed),
                n_tg_cache,
                n_tg_compute,
                n_tg_fail,
            )

        trees_l, grass_l, total_l = [], [], []

        for pos in range(n_edges):
            idx = index_arr[pos]
            if idx in edge_cache:
                r = edge_cache[idx]
                trees_l.append(r["trees"])
                grass_l.append(r["grass"])
                total_l.append(r["total"])
                continue

            geom = geoms[pos]
            if idx not in edge_tiles or geom is None:
                trees_l.append(0.0)
                grass_l.append(0.0)
                total_l.append(0.0)
                continue

            sum_m = 0
            sum_t = 0
            sum_g = 0
            ram_masks: Optional[
                Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]
            ] = None if self._settings.cache_tile_analysis else tile_masks
            for tx, ty in edge_tiles[idx]:
                pair = self._resolve_tile_masks_pair(tx, ty, zoom, ram_masks)
                if pair is None:
                    continue
                t_mask, g_mask = pair
                h, w = t_mask.shape[0], t_mask.shape[1]
                tb = self._tiles.get_tile_bounds(tx, ty, zoom)

                if self._settings.analyze_corridor:
                    m_arr = self.create_corridor_mask(
                        geom, tb, (w, h), buf_m
                    )
                    mb = m_arr > 0
                else:
                    mb = np.ones((h, w), dtype=bool)

                sm = int(mb.sum())
                if sm == 0:
                    continue
                sum_m += sm
                sum_t += int(np.logical_and(t_mask, mb).sum())
                sum_g += int(np.logical_and(g_mask, mb).sum())

            if sum_m > 0:
                tp = 100.0 * float(sum_t) / float(sum_m)
                gp = 100.0 * float(sum_g) / float(sum_m)
                top = tp + gp
            else:
                tp = gp = top = 0.0

            trees_l.append(tp)
            grass_l.append(gp)
            total_l.append(top)
            edge_cache[idx] = {"trees": tp, "grass": gp, "total": top}

        skip_edge_persist = (
            self._satellite_warning is not None
            or n_tg_fail_total > 0
            or invalid_tiles_total > 0
        )
        self._save_edge_cache(
            edge_cache,
            edges_gdf,
            zoom,
            buf_m,
            skip_disk=skip_edge_persist,
        )

        edges_gdf["trees_percent"] = trees_l
        edges_gdf["grass_percent"] = grass_l
        edges_gdf["green_percent"] = total_l
        self._apply_green_coefficients(edges_gdf)
        self._log_green_stats(edges_gdf, buf_m)
        return edges_gdf

    # ==================================================================
    # Приватные методы
    # ==================================================================

    def _tile_mask_cache_path(self, tx: int, ty: int, zoom: int) -> str:
        key = hashlib.sha256(
            f"{self._settings.tms_server!s}|{zoom}|{tx}|{ty}|{TILE_VEG_MASK_VERSION}".encode(
                "utf-8"
            )
        ).hexdigest()[:28]
        sub = os.path.join(self._cache.cache_dir, "tile_green_masks")
        return os.path.join(sub, f"{key}.npz")

    def _load_green_tile_masks_npz(
        self, tx: int, ty: int, zoom: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Чтение готовых масок с диска (без PIL), для агрегации при CACHE_TILE_ANALYSIS."""
        path = self._tile_mask_cache_path(tx, ty, zoom)
        if not os.path.isfile(path):
            return None
        try:
            z = np.load(path)
            t = z["trees"].astype(bool)
            g = z["grass"].astype(bool)
            if t.shape != g.shape:
                return None
            return (t, g)
        except Exception as exc:
            logger.debug("tile mask npz read %s: %s", path, exc)
            return None

    def _resolve_tile_masks_pair(
        self,
        tx: int,
        ty: int,
        zoom: int,
        ram: Optional[Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]],
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if ram is not None:
            return ram.get((tx, ty))
        key = (tx, ty, zoom)
        od = self._tile_mask_read_lru
        if key in od:
            od.move_to_end(key)
            return od[key]
        pair = self._load_green_tile_masks_npz(tx, ty, zoom)
        if pair is None:
            return None
        od[key] = pair
        while len(od) > self._tile_mask_read_lru_max:
            od.popitem(last=False)
        return pair

    def _get_or_compute_tile_masks(
        self, img: Any, tx: int, ty: int, zoom: int
    ) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray]], str]:
        """Один раз NDI/ExG на тайл; чтение/запись ``.npz``.

        Returns:
            (маски или None, источник: ``cache`` | ``compute`` | ``fail``).
        """
        path = self._tile_mask_cache_path(tx, ty, zoom)
        h, w = img.size[1], img.size[0]
        if self._settings.cache_tile_analysis and os.path.isfile(path):
            try:
                z = np.load(path)
                t = z["trees"].astype(bool)
                g = z["grass"].astype(bool)
                if t.shape == (h, w) and g.shape == (h, w):
                    return (t, g), "cache"
            except Exception as exc:
                logger.debug("tile mask cache read %s: %s", path, exc)

        vm = self.compute_vegetation_masks(img)
        if vm is None:
            return None, "fail"
        tbool, gbool = vm
        if self._settings.cache_tile_analysis:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                # savez_compressed добавляет .npz, если путь не оканчивается на .npz
                root, ext = os.path.splitext(path)
                tmp = root + ".tmp" + (ext if ext else ".npz")
                np.savez_compressed(
                    tmp,
                    trees=tbool.astype(np.uint8),
                    grass=gbool.astype(np.uint8),
                )
                os.replace(tmp, path)
            except Exception as exc:
                logger.debug("tile mask cache write %s: %s", path, exc)
        return (tbool, gbool), "compute"

    def _parallel_build_tile_masks(
        self,
        tile_images: Dict[Tuple[int, int], Any],
        zoom: int,
        all_tiles: Set[Tuple[int, int]],
    ) -> Tuple[
        Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
        int,
        int,
        int,
    ]:
        """Собрать маски по тайлам; счётчики cache hit / compute / fail (нет снимка или NDI)."""
        out: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        n_cache = n_compute = n_fail = 0
        tiles_list = list(all_tiles)
        workers = max(1, self._settings.tile_download_threads)

        def job(xy: Tuple[int, int]):
            img = tile_images.get(xy)
            if img is None:
                return xy, None, "fail"
            pair, src = self._get_or_compute_tile_masks(img, xy[0], xy[1], zoom)
            return xy, pair, src

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(job, xy) for xy in tiles_list]
            for fut in as_completed(futures):
                xy, pair, src = fut.result()
                if src == "cache":
                    n_cache += 1
                elif src == "compute":
                    n_compute += 1
                else:
                    n_fail += 1
                if pair is not None:
                    out[xy] = pair
        return out, n_cache, n_compute, n_fail

    def _edge_cache_path(
        self, edges_gdf: gpd.GeoDataFrame, zoom: int, buf_m: int
    ) -> str:
        """Путь к pickle кэша по bbox + параметрам анализа (в т.ч. TMS и алгоритм)."""
        bounds = edges_gdf.total_bounds
        payload = {
            "analyze_corridor": self._settings.analyze_corridor,
            "bounds": [round(float(bounds[i]), 5) for i in range(4)],
            "buf_m": int(buf_m),
            "disable_satellite_green": self._settings.disable_satellite_green,
            "routing_algo_version": str(ROUTING_ALGO_VERSION),
            "schema": GREEN_EDGES_CACHE_SCHEMA,
            "tile_mask_ver": TILE_VEG_MASK_VERSION,
            "tms_server": self._settings.tms_server,
            "zoom": int(zoom),
        }
        h = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        sub = os.path.join(self._cache.cache_dir, "green_edges")
        os.makedirs(sub, exist_ok=True)
        return os.path.join(sub, f"{h}.pkl")

    @staticmethod
    def _edge_cache_is_degenerate_all_zeros(
        cached: Dict[Any, Any], n_edges: int
    ) -> bool:
        """Полный кэш, где у каждого ребра trees=grass=total=0 — часто артефакт сбоя тайлов/TMS.

        Не трогаем совсем маленькие графы: там нули могут быть осмысленны.
        """
        if n_edges < 100 or len(cached) < n_edges:
            return False
        for r in cached.values():
            if not isinstance(r, dict):
                return False
            if (
                float(r.get("trees", 0) or 0) != 0.0
                or float(r.get("grass", 0) or 0) != 0.0
                or float(r.get("total", 0) or 0) != 0.0
            ):
                return False
        return True

    def _load_edge_cache(
        self, edges_gdf: gpd.GeoDataFrame, zoom: int, buf_m: int
    ) -> dict:
        if not self._settings.cache_tile_analysis:
            return {}
        if self._settings.force_recalculate:
            logger.info("FORCE_RECALCULATE=True, пересчитываем")
            return {}
        path = self._edge_cache_path(edges_gdf, zoom, buf_m)
        cached = self._cache.load(path)
        if cached:
            logger.info(
                "Кэш анализа рёбер (%s…): %d записей",
                os.path.basename(path)[:16],
                len(cached),
            )
            n_edges = len(edges_gdf)
            if self._settings.green_edge_reject_all_zero_cache and self._edge_cache_is_degenerate_all_zeros(
                cached, n_edges
            ):
                logger.warning(
                    "green_edges: игнорируем pickle — полный кэш только из нулей "
                    "(вероятен старый сбой тайлов/TMS). Будет пересчёт; при необходимости "
                    "смените TMS_SERVER или удалите %s вручную",
                    os.path.dirname(path),
                )
                return {}
        return cached or {}

    def _save_edge_cache(
        self,
        cache: dict,
        edges_gdf: gpd.GeoDataFrame,
        zoom: int,
        buf_m: int,
        *,
        skip_disk: bool = False,
    ) -> None:
        if not self._settings.cache_tile_analysis:
            return
        if skip_disk:
            logger.warning(
                "green_edges: pickle на диск не пишем (неполные тайлы / fail масок — "
                "иначе нули закрепились бы как «готовый» кэш). Повторите запрос после "
                "исправления сети или TMS."
            )
            return
        path = self._edge_cache_path(edges_gdf, zoom, buf_m)
        self._cache.save(path, cache)

    @staticmethod
    def _apply_green_coefficients(gdf: gpd.GeoDataFrame) -> None:
        """Рассчитать ``green_coeff`` и ``green_type`` по процентам.

        ``green_coeff`` — базовый коэффициент озеленения (0.3–1.2).
        Профильная чувствительность применяется позже в GraphBuilder.
        """
        trees = gdf["trees_percent"].values
        grass = gdf["grass_percent"].values
        total = gdf["green_percent"].values

        penalty = np.maximum(0, (10 - total) / 100)
        gdf["green_coeff"] = np.clip(
            1.0 + penalty - trees / 100 * 0.6 - grass / 100 * 0.25,
            0.3,
            1.2,
        )

        n = len(gdf)
        tg = trees + grass
        gt = np.full(n, "none", dtype=object)
        gt[trees >= 15.0] = "trees"
        sel = gt == "none"
        gt[sel & (tg >= 25.0)] = "park"
        sel = gt == "none"
        gt[sel & (grass >= 10.0)] = "grass"
        sel = gt == "none"
        gt[sel & (tg >= 3.0)] = "sparse"
        gdf["green_type"] = gt

    @staticmethod
    def _fill_empty_green(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Заполнить нулями, если спутниковые данные недоступны."""
        gdf["green_percent"] = 0.0
        gdf["trees_percent"] = 0.0
        gdf["grass_percent"] = 0.0
        gdf["green_coeff"] = 1.0
        gdf["green_type"] = "none"
        return gdf

    @staticmethod
    def _log_green_stats(gdf: gpd.GeoDataFrame, buf_m: int) -> None:
        logger.info(
            "Статистика зелени (коридор %dм): "
            "деревья=%.1f%%, трава=%.1f%%, всего=%.1f%%",
            buf_m,
            gdf["trees_percent"].mean(),
            gdf["grass_percent"].mean(),
            gdf["green_percent"].mean(),
        )
        logger.info("Категории: %s", dict(gdf["green_type"].value_counts()))
