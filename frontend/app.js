/* ═══════════════════════════════════════════════════════════════
 * Bike Router — Frontend Application
 * ═══════════════════════════════════════════════════════════════ */

const API = '';
const SAMARA_CENTER = [50.10, 53.19];
/** Таймаут POST /alternatives (мс); первый коридор + спутник/зелень в Docker часто 3–10+ мин */
const ROUTE_FETCH_MS = 600000;
/** Подсказка «сервер долго считает» (раньше таймаута) */
const ROUTE_SLOW_HINT_MS = 20000;
/** Опрос /alternatives/job/{id}: чаще в начале, реже при долгом ожидании (мс). */
function alternativesJobPollDelayMs(iteration) {
  const base = 900;
  const cap = 12000;
  return Math.min(cap, base + iteration * iteration * 350);
}

/** С `/health`: false = на сервере DISABLE_SATELLITE_GREEN, метрики зелени в ответах нулевые */
let serverSatelliteGreenEnabled = true;

/** Ключи map_layers с бэкенда → слой MapLibre + подписи легенды */
const OVERLAY_LAYERS = [
  {
    key: 'greenery',
    color: '#1e8449',
    width: 6,
    legendLabel: 'Озеленение',
    legendSub: 'Выше зелени по снимку / типу',
  },
  {
    key: 'stairs',
    color: '#7d3c98',
    width: 6,
    legendLabel: 'Лестницы',
    legendSub: 'highway=steps',
  },
  {
    key: 'problematic',
    color: '#c0392b',
    width: 5,
    legendLabel: 'Проблемные',
    legendSub: 'Крутизна или оживлённая дорога',
  },
  {
    key: 'na_surface',
    color: '#7f8c8d',
    width: 4,
    dash: [2, 2],
    legendLabel: 'Нет surface',
    legendSub: 'В OSM не указано покрытие',
  },
];

/* ── State ─────────────────────────────────────────────────────── */

const state = {
  startCoords: null,
  endCoords: null,
  profile: 'cyclist',
  activeInput: 'start',
  routes: null,
  routeList: [],
  selectedVariantIndex: 0,
  loading: false,
  pendingVariantFromUrl: null,
  /** input | variants | detail — только при isMobileLayout() */
  mobileSheetPhase: 'input',
  /** В фазе detail: false — низкая шторка (больше карты), true — развёрнута */
  mobileSheetDetailExpanded: false,
  /** Мобильная фаза input без маршрутов: false — низкая «полка», true — полная форма */
  mobileInputSheetExpanded: false,
  /** Учёт озеленения (чекбокс + progressive 3-й маршрут) */
  greenEnabled: true,
  /** pending=['green'] от API; сбрасывается при пустом pending или при mode===green в routes */
  jobPendingGreen: false,
  /** Инкремент при сбросе — отмена фонового poll */
  routeFetchSeq: 0,
};

const N_ROUTE_LAYERS = 3;

const markers = { start: null, end: null };
let map;
let overlayPopup = null;
/** Координаты для пунктов ПКМ-меню (desktop) */
let mapContextMenuLngLat = null;
/** Подавляет replaceState при гидратации из URL */
let urlSyncSuppressed = false;

/* ── Route colors per profile ──────────────────────────────────── */

const COLORS = {
  cyclist: {
    full: '#2874a6',
    green: '#1e8449',
    shortest: '#ca6f1e',
  },
  pedestrian: {
    full: '#b8570a',
    green: '#6c3483',
    shortest: '#117864',
  },
};

/* ═══════════ DOM ═══════════════════════════════════════════════ */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const dom = {
  startInput:   $('#start-input'),
  endInput:     $('#end-input'),
  startSuggest: $('#start-suggest'),
  endSuggest:   $('#end-suggest'),
  swapBtn:      $('#swap-btn'),
  locateStartBtn: $('#locate-start-btn'),
  clearPointsBtn: $('#clear-points-btn'),
  resetRouteBtn: $('#reset-route-btn'),
  buildBtn:     $('#build-btn'),
  buildText:    $('.build-text'),
  spinner:      $('.spinner'),
  profileBtns:  $$('#profile-toggle .toggle-btn'),
  results:        $('#results'),
  variantTabs:    $('#variant-tabs'),
  routeDetailCard: $('#route-detail-card'),
  mapLegend:      $('#map-legend'),
  mapContextMenu: $('#map-context-menu'),
  fitRoutesBtn:   $('#fit-routes-btn'),
  elevChart:      $('#elev-chart'),
  errorToast:   $('#error-toast'),
  versionText:  $('#version-text'),
  versionTextMobile: $('#version-text-mobile'),
  tripStatus:   $('#trip-status'),
  routeProgress: $('#route-progress'),
  routeProgressFill: $('#route-progress-fill'),
  copyShareBtn: $('#copy-share-btn'),
  resultsEmpty: $('#results-empty'),
  mobileLocateBtn: $('#mobile-locate-btn'),
  mapHintToggle: $('#map-hint-toggle'),
  mapHintTooltip: $('#map-hint-tooltip'),
  mobileEditPointsBtn: $('#mobile-edit-points-btn'),
  mobileBackVariantsBtn: $('#mobile-back-variants-btn'),
  mobileRouteCompact: $('#mobile-route-compact'),
  mobileDetailSticky: $('#mobile-detail-sticky'),
  mobileDetailTitle: $('#mobile-detail-title'),
  mobileDetailExpandBtn: $('#mobile-detail-expand-btn'),
  sheetHandle: $('#sheet-handle'),
  greenEnabledCheckbox: $('#green-enabled-checkbox'),
};

const MQ_MOBILE = typeof window !== 'undefined' ? window.matchMedia('(max-width: 768px)') : null;

function isMobileLayout() {
  return MQ_MOBILE ? MQ_MOBILE.matches : false;
}

function truncMobileLabel(s, maxLen) {
  const t = (s == null ? '' : String(s)).trim();
  if (t.length <= maxLen) return t;
  return `${t.slice(0, Math.max(0, maxLen - 1))}…`;
}

/** Короткое имя варианта в мобильной строке вкладки (остальное — в title на desktop). */
const MOBILE_VARIANT_TAB_LABEL_MAX = 11;
const MOBILE_DETAIL_TITLE_MAX = 24;

function updateMobileRouteCompact() {
  if (!dom.mobileRouteCompact) return;
  const hasR = !!(state.routeList && state.routeList.length);
  const ph = state.mobileSheetPhase;
  const show = isMobileLayout() && hasR && (ph === 'variants' || ph === 'detail');
  if (!show) {
    dom.mobileRouteCompact.hidden = true;
    return;
  }
  const a = truncMobileLabel(dom.startInput && dom.startInput.value, 34);
  const b = truncMobileLabel(dom.endInput && dom.endInput.value, 34);
  dom.mobileRouteCompact.textContent = `${a || 'Старт'} → ${b || 'Финиш'}`;
  dom.mobileRouteCompact.hidden = false;
}

function updateMobileDetailHeader() {
  if (!dom.mobileDetailSticky || !dom.mobileDetailTitle) return;
  if (!isMobileLayout() || state.mobileSheetPhase !== 'detail') {
    dom.mobileDetailSticky.hidden = true;
    dom.mobileDetailTitle.textContent = '';
    dom.mobileDetailTitle.removeAttribute('title');
    if (dom.mobileDetailExpandBtn) {
      dom.mobileDetailExpandBtn.hidden = true;
    }
    return;
  }
  const routes = state.routeList;
  const r = routes && routes[state.selectedVariantIndex];
  if (!r) {
    dom.mobileDetailSticky.hidden = true;
    dom.mobileDetailTitle.removeAttribute('title');
    if (dom.mobileDetailExpandBtn) dom.mobileDetailExpandBtn.hidden = true;
    return;
  }
  const fullTitle = String(r.variant_label || r.mode || '').trim();
  dom.mobileDetailTitle.textContent = truncMobileLabel(fullTitle, MOBILE_DETAIL_TITLE_MAX);
  dom.mobileDetailTitle.title = fullTitle;
  dom.mobileDetailSticky.hidden = false;
  if (dom.mobileDetailExpandBtn) {
    dom.mobileDetailExpandBtn.hidden = false;
    const expanded = state.mobileSheetDetailExpanded;
    dom.mobileDetailExpandBtn.setAttribute('aria-expanded', expanded ? 'true' : 'false');
    dom.mobileDetailExpandBtn.setAttribute(
      'aria-label',
      expanded ? 'Свернуть панель, показать больше карты' : 'Развернуть панель деталей',
    );
    dom.mobileDetailExpandBtn.title = expanded ? 'Показать больше карты' : 'Развернуть панель деталей';
    const icon = dom.mobileDetailExpandBtn.querySelector('.mobile-detail-expand-icon');
    if (icon) icon.textContent = expanded ? '▾' : '▴';
  }
}

function syncMobileSheetUi() {
  const hasRoutes = !!(state.routeList && state.routeList.length);
  document.body.classList.toggle('mobile-sheet-has-routes', hasRoutes);
  document.body.classList.remove(
    'mobile-sheet-phase-input',
    'mobile-sheet-phase-variants',
    'mobile-sheet-phase-detail',
    'mobile-sheet-detail-expanded',
    'mobile-sheet-input-peeking',
  );
  if (isMobileLayout()) {
    document.body.classList.add(`mobile-sheet-phase-${state.mobileSheetPhase}`);
    if (state.mobileSheetPhase === 'detail' && state.mobileSheetDetailExpanded) {
      document.body.classList.add('mobile-sheet-detail-expanded');
    }
    const inputPeeking =
      state.mobileSheetPhase === 'input' &&
      !hasRoutes &&
      !state.mobileInputSheetExpanded;
    if (inputPeeking) {
      document.body.classList.add('mobile-sheet-input-peeking');
    }
    if (dom.sheetHandle) {
      const expanded = !inputPeeking;
      dom.sheetHandle.setAttribute('aria-expanded', expanded ? 'true' : 'false');
    }
  } else if (dom.sheetHandle) {
    dom.sheetHandle.setAttribute('aria-expanded', 'true');
  }
  updateMobileRouteCompact();
  updateMobileDetailHeader();
  requestAnimationFrame(() => {
    if (map) map.resize();
  });
}

function applyRouteHighlightEqual(nActive) {
  if (!map) return;
  for (let i = 0; i < N_ROUTE_LAYERS; i++) {
    const lid = `route-layer-${i}`;
    if (!map.getLayer(lid)) continue;
    if (i >= nActive) continue;
    map.setPaintProperty(lid, 'line-opacity', 0.9);
    map.setPaintProperty(lid, 'line-width', 6);
    map.setPaintProperty(lid, 'line-blur', 0);
  }
}

function syncRouteHighlightFromUi() {
  const n = state.routeList?.length || 0;
  if (!n || !map) return;
  if (isMobileLayout() && state.mobileSheetPhase === 'variants') {
    applyRouteHighlightEqual(n);
  } else {
    applyRouteHighlight(state.selectedVariantIndex, n);
  }
}

function getMapFitPadding() {
  if (isMobileLayout()) {
    if (state.mobileSheetPhase === 'variants') {
      return { top: 56, bottom: 100, left: 20, right: 20 };
    }
    if (state.mobileSheetPhase === 'detail') {
      const bottom = state.mobileSheetDetailExpanded ? 200 : 108;
      return { top: 56, bottom, left: 20, right: 20 };
    }
    const hasRoutes = !!(state.routeList && state.routeList.length);
    const peek =
      !hasRoutes && !state.mobileInputSheetExpanded;
    if (peek) {
      return { top: 56, bottom: 92, left: 20, right: 20 };
    }
    return { top: 56, bottom: 168, left: 20, right: 20 };
  }
  return { top: 72, bottom: 100, left: 72, right: 72 };
}

function dismissMapHint() {
  document.body.classList.add('map-hint-dismissed');
  if (dom.mapHintTooltip) dom.mapHintTooltip.classList.add('hidden');
}

function toggleMobileInputSheetPeek() {
  if (!isMobileLayout() || state.mobileSheetPhase !== 'input') return;
  if (state.routeList && state.routeList.length) return;
  state.mobileInputSheetExpanded = !state.mobileInputSheetExpanded;
  syncMobileSheetUi();
  if (map) {
    requestAnimationFrame(() => fitAllRoutesInView());
  }
}

function expandMobileInputSheet() {
  if (!isMobileLayout() || state.mobileSheetPhase !== 'input') return;
  if (state.mobileInputSheetExpanded) return;
  state.mobileInputSheetExpanded = true;
  syncMobileSheetUi();
}

function toggleMobileDetailSheetExpand() {
  if (!isMobileLayout() || state.mobileSheetPhase !== 'detail') return;
  state.mobileSheetDetailExpanded = !state.mobileSheetDetailExpanded;
  syncMobileSheetUi();
  updateMobileDetailHeader();
  if (map && state.routeList && state.routeList.length) {
    requestAnimationFrame(() => fitAllRoutesInView());
  }
}

function mobileBackToVariants() {
  const routes = state.routeList;
  if (!routes || !routes.length) return;
  state.mobileSheetPhase = 'variants';
  state.mobileSheetDetailExpanded = false;
  applyRouteHighlightEqual(routes.length);
  dom.routeDetailCard.innerHTML = '';
  dom.elevChart.innerHTML = '';
  dom.variantTabs.querySelectorAll('.variant-tab').forEach((b) => {
    b.classList.remove('active');
    b.setAttribute('aria-selected', 'false');
  });
  syncMobileSheetUi();
  updateTripStatus();
  const sb = document.getElementById('sidebar');
  if (sb) sb.scrollTo({ top: 0, behavior: 'smooth' });
}

function mobileEditPoints() {
  state.mobileSheetPhase = 'input';
  state.mobileInputSheetExpanded = true;
  syncMobileSheetUi();
  updateTripStatus();
  const sb = document.getElementById('sidebar');
  if (sb) sb.scrollTo({ top: 0, behavior: 'smooth' });
}

/* ═══════════ Map ══════════════════════════════════════════════ */

function lineColorForMode(mode) {
  const pal = COLORS[state.profile];
  if (mode === 'shortest') return pal.shortest;
  if (mode === 'green') return pal.green;
  return pal.full;
}

/** Порядок в UI: сначала кратчайший по длине на карте, остальные — в исходном порядке API. */
function orderRoutesShortestFirst(routes) {
  if (!routes || routes.length <= 1) return routes || [];
  return routes
    .map((r, i) => ({ r, i }))
    .sort((a, b) => {
      const d = a.r.length_m - b.r.length_m;
      if (Math.abs(d) < 1e-3) return a.i - b.i;
      return d;
    })
    .map(({ r }) => r);
}

function initMap() {
  /* Базовая подложка: CARTO Voyager (CDN). Не tile.openstreetmap.org —
   * публичные тайлы OSMF не рассчитаны на постоянную нагрузку без своего кэша/договора. */
  map = new maplibregl.Map({
    container: 'map',
    style: {
      version: 8,
      sources: {
        carto: {
          type: 'raster',
          tiles: [
            'https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}@2x.png',
            'https://b.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}@2x.png',
            'https://c.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}@2x.png',
          ],
          tileSize: 256,
          attribution: '© <a href="https://carto.com/">CARTO</a> © <a href="https://www.openstreetmap.org/copyright">OSM</a>',
        },
      },
      layers: [{ id: 'carto', type: 'raster', source: 'carto' }],
    },
    center: SAMARA_CENTER,
    zoom: 14,
  });

  map.addControl(new maplibregl.NavigationControl(), 'top-right');
  /* Атрибуция CARTO/OSM задаётся у raster-источника; MapLibre показывает её стандартным блоком. */

  map.on('load', () => {
    addRouteSources();
    addOverlaySources();
    void applyUrlFromQuery();
  });

  map.on('click', onMapClick);
  map.on('contextmenu', onMapContextMenu);
  map.on('movestart', () => {
    closeMapContextMenu();
  });
  map.on('mousemove', onMapMouseMove);
}

function getVisibleOverlayLayerIds() {
  if (!map) return [];
  return OVERLAY_LAYERS.map(({ key }) => `overlay-layer-${key}`).filter((id) => {
    try {
      return map.getLayer(id) && map.getLayoutProperty(id, 'visibility') === 'visible';
    } catch {
      return false;
    }
  });
}

function removeOverlayPopup() {
  if (overlayPopup) {
    overlayPopup.remove();
    overlayPopup = null;
  }
}

/** Текст popup по слою и свойствам GeoJSON с бэкенда. */
function buildOverlayPopupHtml(layerId, props) {
  const p = props || {};
  const hw = escHtml(String(p.highway || '—'));
  const len = p.length_m != null ? `${escHtml(String(p.length_m))} м` : '—';
  const grad = p.gradient_pct != null ? `${escHtml(String(p.gradient_pct))}%` : '—';

  if (layerId.includes('stairs')) {
    return `<div class="map-popup-inner"><strong>Лестница</strong><p>highway=steps · длина ~${len}</p></div>`;
  }
  if (layerId.includes('na_surface')) {
    return `<div class="map-popup-inner"><strong>Нет surface в OSM</strong><p>Для сегмента не указано покрытие · ~${len}</p><p class="map-popup-meta">Тип дороги: ${hw}</p></div>`;
  }
  if (layerId.includes('greenery')) {
    const gt = escHtml(String(p.green_type || '—'));
    const tr = p.trees_pct != null ? escHtml(String(p.trees_pct)) : '—';
    const gr = p.grass_pct != null ? escHtml(String(p.grass_pct)) : '—';
    return `<div class="map-popup-inner"><strong>Озеленённый участок</strong><p>Тип: ${gt} · деревья ${tr}% · трава ${gr}%</p><p class="map-popup-meta">~${len} · уклон ~${grad}</p></div>`;
  }
  if (layerId.includes('problematic')) {
    const reasons = String(p.reasons || '').split(',').filter(Boolean);
    const lines = [];
    if (reasons.includes('steep')) lines.push('Крутой уклон');
    if (reasons.includes('traffic_class')) lines.push('Класс дороги с интенсивным движением');
    const body = lines.length ? lines.map((t) => `<li>${escHtml(t)}</li>`).join('') : '<li>Участок повышенного внимания</li>';
    return `<div class="map-popup-inner"><strong>Проблемный участок</strong><ul class="map-popup-ul">${body}</ul><p class="map-popup-meta">${hw} · ~${len} · уклон ~${grad}</p></div>`;
  }
  return `<div class="map-popup-inner">Сегмент маршрута</div>`;
}

function showOverlayPopup(feature, lngLat) {
  removeOverlayPopup();
  const lid = feature.layer && feature.layer.id ? feature.layer.id : '';
  const html = buildOverlayPopupHtml(lid, feature.properties);
  overlayPopup = new maplibregl.Popup({
    closeButton: true,
    maxWidth: '300px',
    anchor: 'bottom',
  })
    .setLngLat(lngLat)
    .setHTML(html)
    .addTo(map);
}

function closeMapContextMenu() {
  if (!dom.mapContextMenu) return;
  dom.mapContextMenu.classList.add('hidden');
  dom.mapContextMenu.setAttribute('aria-hidden', 'true');
  mapContextMenuLngLat = null;
}

function openMapContextMenu(clientX, clientY, lngLat) {
  if (!dom.mapContextMenu) return;
  mapContextMenuLngLat = { lat: lngLat.lat, lng: lngLat.lng };
  const el = dom.mapContextMenu;
  el.classList.remove('hidden');
  el.setAttribute('aria-hidden', 'false');
  const pad = 8;
  el.style.left = `${clientX}px`;
  el.style.top = `${clientY}px`;
  requestAnimationFrame(() => {
    const r = el.getBoundingClientRect();
    let x = clientX;
    let y = clientY;
    if (r.right > window.innerWidth - pad) x = window.innerWidth - r.width - pad;
    if (r.bottom > window.innerHeight - pad) y = window.innerHeight - r.height - pad;
    if (x < pad) x = pad;
    if (y < pad) y = pad;
    el.style.left = `${x}px`;
    el.style.top = `${y}px`;
  });
}

function onMapContextMenu(e) {
  if (isMobileLayout()) return;
  e.preventDefault();
  if (e.originalEvent && typeof e.originalEvent.preventDefault === 'function') {
    e.originalEvent.preventDefault();
  }
  removeOverlayPopup();
  const oe = e.originalEvent;
  const cx = oe && typeof oe.clientX === 'number' ? oe.clientX : 0;
  const cy = oe && typeof oe.clientY === 'number' ? oe.clientY : 0;
  openMapContextMenu(cx, cy, e.lngLat);
}

function applyMapContextMenuChoice(which) {
  if (!mapContextMenuLngLat) return;
  const { lat, lng } = mapContextMenuLngLat;
  state.activeInput = which;
  setPoint(which, lat, lng);
  reverseGeocode(lat, lng, which);
  closeMapContextMenu();
  dismissMapHint();
}

function onMapClick(e) {
  dismissMapHint();
  const overlayIds = getVisibleOverlayLayerIds();
  if (overlayIds.length) {
    const feats = map.queryRenderedFeatures(e.point, { layers: overlayIds });
    if (feats.length) {
      showOverlayPopup(feats[0], e.lngLat);
      return;
    }
  }
  removeOverlayPopup();

  if (!isMobileLayout()) {
    return;
  }

  const { lng, lat } = e.lngLat;
  let which;
  if (!state.startCoords && !state.endCoords) {
    which = 'start';
  } else if (state.startCoords && !state.endCoords) {
    which = 'end';
  } else if (!state.startCoords && state.endCoords) {
    which = 'start';
  } else {
    which = state.activeInput;
  }
  setPoint(which, lat, lng);
  reverseGeocode(lat, lng, which);
}

function onMapMouseMove(e) {
  if (!map) return;
  const overlayIds = getVisibleOverlayLayerIds();
  let hit = false;
  if (overlayIds.length) {
    const feats = map.queryRenderedFeatures(e.point, { layers: overlayIds });
    hit = feats.length > 0;
  }
  map.getCanvas().style.cursor = hit ? 'pointer' : '';
}

function computeAllRoutesBounds() {
  const bounds = new maplibregl.LngLatBounds();
  let ok = false;
  (state.routeList || []).forEach((r) => {
    if (!r.geometry || !r.geometry.length) return;
    r.geometry.forEach(([plat, plon]) => {
      bounds.extend([plon, plat]);
      ok = true;
    });
  });
  return ok ? bounds : null;
}

function fitAllRoutesInView() {
  const b = computeAllRoutesBounds();
  if (!b || !map) return;
  const pad = isMobileLayout()
    ? {
        top: 64,
        bottom: state.mobileSheetPhase === 'variants' ? 130 : 210,
        left: 24,
        right: 24,
      }
    : { top: 100, bottom: 120, left: 100, right: 100 };
  map.fitBounds(b, {
    padding: pad,
    duration: 750,
    maxZoom: 16,
  });
}

function addRouteSources() {
  const empty = { type: 'FeatureCollection', features: [] };
  for (let i = 0; i < N_ROUTE_LAYERS; i++) {
    const sid = `route-src-${i}`;
    const lid = `route-layer-${i}`;
    map.addSource(sid, { type: 'geojson', data: empty });
    map.addLayer({
      id: lid,
      type: 'line',
      source: sid,
      layout: { 'line-cap': 'round', 'line-join': 'round' },
      paint: {
        'line-color': '#2874a6',
        'line-width': 5,
        'line-opacity': 0.9,
      },
    });
  }
}

function addOverlaySources() {
  const empty = { type: 'FeatureCollection', features: [] };
  OVERLAY_LAYERS.forEach(({ key, color, width, dash }) => {
    const sid = `overlay-src-${key}`;
    const lid = `overlay-layer-${key}`;
    map.addSource(sid, { type: 'geojson', data: empty });
    const paint = {
      'line-color': color,
      'line-width': width,
      'line-opacity': 0.88,
    };
    if (dash) paint['line-dasharray'] = dash;
    map.addLayer({
      id: lid,
      type: 'line',
      source: sid,
      layout: { 'line-cap': 'round', 'line-join': 'round', visibility: 'none' },
      paint,
    });
  });
}

function clearOverlaySources() {
  if (!map) return;
  const empty = { type: 'FeatureCollection', features: [] };
  OVERLAY_LAYERS.forEach(({ key }) => {
    const s = map.getSource(`overlay-src-${key}`);
    if (s) s.setData(empty);
  });
}

function syncOverlayVisibility() {
  /* Слои сегментов отключены; оставлено для совместимости вызовов. */
}

/** Сегментные слои (озеленение, лестницы и т.д.) на карте не показываем — только сплошные линии маршрутов. */
function updateOverlays(_route) {
  if (!map) return;
  clearOverlaySources();
}

function setMarker(type, lat, lon) {
  if (markers[type]) markers[type].remove();

  const el = document.createElement('div');
  el.className = `marker ${type}`;

  markers[type] = new maplibregl.Marker({ element: el, draggable: true })
    .setLngLat([lon, lat])
    .addTo(map);

  markers[type].on('dragend', () => {
    dismissMapHint();
    const { lng, lat: la } = markers[type].getLngLat();
    setPoint(type, la, lng);
    reverseGeocode(la, lng, type);
  });
}

function displayRoutes(data) {
  const routes = data.routes || [];
  state.routeList = routes;
  const empty = { type: 'FeatureCollection', features: [] };
  const bounds = new maplibregl.LngLatBounds();
  let hasBounds = false;

  for (let i = 0; i < N_ROUTE_LAYERS; i++) {
    const src = map.getSource(`route-src-${i}`);
    if (!src) continue;
    if (i < routes.length) {
      const coords = routes[i].geometry.map(([lat, lon]) => [lon, lat]);
      coords.forEach((c) => { bounds.extend(c); hasBounds = true; });
      src.setData({
        type: 'Feature',
        geometry: { type: 'LineString', coordinates: coords },
      });
      map.setPaintProperty(
        `route-layer-${i}`,
        'line-color',
        lineColorForMode(routes[i].mode),
      );
      map.setLayoutProperty(`route-layer-${i}`, 'visibility', 'visible');
    } else {
      src.setData(empty);
      map.setLayoutProperty(`route-layer-${i}`, 'visibility', 'none');
    }
  }

  syncRouteHighlightFromUi();
  clearOverlaySources();
  if (dom.fitRoutesBtn) dom.fitRoutesBtn.disabled = routes.length === 0;
  if (hasBounds) {
    const pad = getMapFitPadding();
    map.fitBounds(bounds, { padding: pad, duration: 900 });
  }
}

function applyRouteHighlight(selectedIdx, nActive) {
  if (!map) return;
  for (let i = 0; i < N_ROUTE_LAYERS; i++) {
    const lid = `route-layer-${i}`;
    if (!map.getLayer(lid)) continue;
    if (i >= nActive) continue;
    const on = i === selectedIdx;
    map.setPaintProperty(lid, 'line-opacity', on ? 1 : 0.5);
    map.setPaintProperty(lid, 'line-width', on ? 8 : 6);
    map.setPaintProperty(lid, 'line-blur', 0);
  }
}

function renderMapLegend(routes) {
  if (!routes.length) {
    dom.mapLegend.classList.add('hidden');
    if (dom.fitRoutesBtn) dom.fitRoutesBtn.disabled = true;
    return;
  }
  if (isMobileLayout()) {
    dom.mapLegend.classList.add('hidden');
    dom.mapLegend.innerHTML = '';
    if (dom.fitRoutesBtn) dom.fitRoutesBtn.disabled = false;
    return;
  }
  dom.mapLegend.classList.remove('hidden');
  if (dom.fitRoutesBtn) dom.fitRoutesBtn.disabled = false;
  const sel = state.selectedVariantIndex;
  const routeRows = routes.map((r, i) => {
    const col = lineColorForMode(r.mode);
    const name = escHtml(r.variant_label || r.mode);
    const active = i === sel ? ' lg-row-active' : '';
    const pill = i === sel ? ' <span class="lg-pill">на карте</span>' : '';
    return `<div class="lg-row${active}"><span class="lg-swatch lg-swatch-route" style="background:${col}"></span><span class="lg-text">${name}${pill}</span></div>`;
  }).join('');

  dom.mapLegend.innerHTML = `
    <div class="lg-section">
      <div class="lg-section-title">Маршруты на карте</div>
      <p class="lg-mini-hint">Все варианты на карте; выбранный — яркий и толще, остальные ~50% прозрачности. Первый в списке — кратчайший по длине.</p>
      ${routeRows}
    </div>
  `;
}

function clearRoutes() {
  const empty = { type: 'FeatureCollection', features: [] };
  for (let i = 0; i < N_ROUTE_LAYERS; i++) {
    const src = map.getSource(`route-src-${i}`);
    if (src) src.setData(empty);
    if (map.getLayer(`route-layer-${i}`)) {
      map.setLayoutProperty(`route-layer-${i}`, 'visibility', 'none');
    }
  }
  dom.results.classList.add('hidden');
  dom.results.classList.remove('mobile-has-routes', 'results--no-variants');
  dom.results.setAttribute('aria-hidden', 'true');
  dom.elevChart.innerHTML = '';
  dom.variantTabs.innerHTML = '';
  dom.routeDetailCard.innerHTML = '';
  dom.mapLegend.classList.add('hidden');
  dom.mapLegend.innerHTML = '';
  removeOverlayPopup();
  if (dom.fitRoutesBtn) dom.fitRoutesBtn.disabled = true;
  clearOverlaySources();
  state.routeList = [];
  state.routes = null;
  state.selectedVariantIndex = 0;
  state.mobileSheetPhase = 'input';
  state.routeFetchSeq += 1;
  state.jobPendingGreen = false;
  state.mobileSheetDetailExpanded = false;
  state.mobileInputSheetExpanded = false;
  if (dom.resultsEmpty) {
    dom.resultsEmpty.classList.add('hidden');
    dom.resultsEmpty.setAttribute('aria-hidden', 'true');
    dom.resultsEmpty.textContent = '';
  }
  syncMobileSheetUi();
  updateTripStatus();
}

/* ═══════════ URL share (query: slat, slon, elat, elon, p, v, sa, ea) ═══════════ */

function hydrateStateFromUrl() {
  const p = new URLSearchParams(window.location.search);
  urlSyncSuppressed = true;
  try {
    const prof = p.get('p');
    if (prof === 'pedestrian' || prof === 'cyclist') {
      state.profile = prof;
      dom.profileBtns.forEach((b) => b.classList.toggle('active', b.dataset.value === state.profile));
    }
    const sa = p.get('sa');
    const ea = p.get('ea');
    if (sa) dom.startInput.value = sa;
    if (ea) dom.endInput.value = ea;
    const g = p.get('green');
    if (g === '0' || g === 'false') {
      state.greenEnabled = false;
      if (dom.greenEnabledCheckbox) dom.greenEnabledCheckbox.checked = false;
    } else if (g === '1' || g === 'true') {
      state.greenEnabled = true;
      if (dom.greenEnabledCheckbox) dom.greenEnabledCheckbox.checked = true;
    }
  } finally {
    urlSyncSuppressed = false;
  }
}

function syncUrlFromState() {
  if (urlSyncSuppressed) return;
  const p = new URLSearchParams();
  if (state.startCoords) {
    p.set('slat', state.startCoords.lat.toFixed(5));
    p.set('slon', state.startCoords.lon.toFixed(5));
  }
  if (state.endCoords) {
    p.set('elat', state.endCoords.lat.toFixed(5));
    p.set('elon', state.endCoords.lon.toFixed(5));
  }
  if (state.profile === 'pedestrian') p.set('p', 'pedestrian');
  const si = dom.startInput.value.trim();
  const ei = dom.endInput.value.trim();
  if (si) p.set('sa', si.slice(0, 400));
  if (ei) p.set('ea', ei.slice(0, 400));
  if (state.routeList && state.routeList.length) {
    p.set('v', String(state.selectedVariantIndex));
  }
  const ge =
    dom.greenEnabledCheckbox && !dom.greenEnabledCheckbox.checked ? '0' : '1';
  if (ge === '0') p.set('green', '0');
  const qs = p.toString();
  const path = window.location.pathname || '/';
  const newUrl = qs ? `${path}?${qs}` : path;
  history.replaceState(null, '', newUrl);
}

function clearUrlParams() {
  urlSyncSuppressed = true;
  history.replaceState(null, '', window.location.pathname || '/');
  urlSyncSuppressed = false;
}

async function applyUrlFromQuery() {
  if (!map) return;
  const p = new URLSearchParams(window.location.search);
  const slat = parseFloat(p.get('slat'));
  const slon = parseFloat(p.get('slon'));
  const elat = parseFloat(p.get('elat'));
  const elon = parseFloat(p.get('elon'));
  if (![slat, slon, elat, elon].every(Number.isFinite)) {
    updateTripStatus();
    return;
  }
  urlSyncSuppressed = true;
  try {
    setPoint('start', slat, slon);
    setPoint('end', elat, elon);
    const vRaw = parseInt(p.get('v'), 10);
    state.pendingVariantFromUrl = Number.isFinite(vRaw) && vRaw >= 0 ? vRaw : null;
    const b = new maplibregl.LngLatBounds();
    b.extend([slon, slat]);
    b.extend([elon, elat]);
    map.fitBounds(b, {
      padding: { top: 56, bottom: 72, left: 56, right: 56 },
      duration: 0,
      maxZoom: 15,
    });
  } finally {
    urlSyncSuppressed = false;
  }
  syncUrlFromState();
  updateTripStatus();
  await buildRoute();
  if (state.startCoords) reverseGeocode(state.startCoords.lat, state.startCoords.lon, 'start');
  if (state.endCoords) reverseGeocode(state.endCoords.lat, state.endCoords.lon, 'end');
}

/** Прогресс построения: неопределённо на POST, ~68% пока ждём зелёный вариант по job_id. */
function updateRouteProgress() {
  const wrap = dom.routeProgress;
  const fill = dom.routeProgressFill;
  if (!wrap) return;

  const greenWait = state.jobPendingGreen && state.greenEnabled;
  const postWait = state.loading && !greenWait;

  if (!postWait && !greenWait) {
    wrap.classList.add('hidden');
    wrap.classList.remove('route-progress--indeterminate', 'route-progress--partial');
    wrap.setAttribute('aria-hidden', 'true');
    wrap.removeAttribute('role');
    wrap.removeAttribute('aria-busy');
    wrap.removeAttribute('aria-valuenow');
    wrap.removeAttribute('aria-valuemin');
    wrap.removeAttribute('aria-valuemax');
    if (fill) fill.style.width = '';
    return;
  }

  wrap.classList.remove('hidden');
  wrap.setAttribute('aria-hidden', 'false');
  wrap.setAttribute('role', 'progressbar');

  if (postWait) {
    wrap.classList.add('route-progress--indeterminate');
    wrap.classList.remove('route-progress--partial');
    wrap.setAttribute('aria-busy', 'true');
    wrap.removeAttribute('aria-valuenow');
    wrap.removeAttribute('aria-valuemin');
    wrap.removeAttribute('aria-valuemax');
    if (fill) fill.style.width = '';
  } else {
    wrap.classList.add('route-progress--partial');
    wrap.classList.remove('route-progress--indeterminate');
    wrap.removeAttribute('aria-busy');
    wrap.setAttribute('aria-valuenow', '68');
    wrap.setAttribute('aria-valuemin', '0');
    wrap.setAttribute('aria-valuemax', '100');
    if (fill) fill.style.width = '';
  }
}

function updateTripStatus() {
  if (!dom.tripStatus) return;
  const m = isMobileLayout();
  try {
    if (state.jobPendingGreen && state.greenEnabled) {
      dom.tripStatus.textContent = 'Строим маршрут с учётом озеленения…';
      dom.tripStatus.className = 'trip-status trip-status-loading';
      return;
    }
    if (state.loading) {
      dom.tripStatus.textContent = m ? 'Считаем маршрут…' : 'Строим маршрут…';
      dom.tripStatus.className = 'trip-status trip-status-loading';
      return;
    }
    const s = state.startCoords;
    const e = state.endCoords;
    if (!s && !e) {
      dom.tripStatus.textContent = m
        ? 'Старт и финиш — тап по карте или адрес.'
        : 'Укажите старт и финиш — клик по карте или поиск адреса (Enter / «Найти»).';
      dom.tripStatus.className = 'trip-status trip-status-idle';
      return;
    }
    if (s && !e) {
      dom.tripStatus.textContent = m ? 'Укажите финиш.' : 'Старт задан. Укажите финиш.';
      dom.tripStatus.className = 'trip-status trip-status-partial';
      return;
    }
    if (!s && e) {
      dom.tripStatus.textContent = m ? 'Укажите старт.' : 'Финиш задан. Укажите старт.';
      dom.tripStatus.className = 'trip-status trip-status-partial';
      return;
    }
    if (state.routeList && state.routeList.length) {
      if (m && state.mobileSheetPhase === 'input') {
        dom.tripStatus.textContent = 'Маршрут на карте. Поменяйте точки и нажмите «Построить» снова.';
      } else {
        dom.tripStatus.textContent = m
          ? 'Готово. Ниже варианты — тап для деталей.'
          : 'Маршрут построен. Первый вариант — кратчайший; вкладки и ссылка сохраняют выбор.';
      }
      dom.tripStatus.className = 'trip-status trip-status-done';
      return;
    }
    dom.tripStatus.textContent = m ? '«Построить маршрут»' : 'Нажмите «Построить маршрут».';
    dom.tripStatus.className = 'trip-status trip-status-ready';
  } finally {
    updateRouteProgress();
  }
}

/* ═══════════ Geocoding (явный запрос: кнопка / Enter — без автодополнения при вводе) ═══ */

async function runAddressSearch(type) {
  const input = type === 'start' ? dom.startInput : dom.endInput;
  const suggest = type === 'start' ? dom.startSuggest : dom.endSuggest;
  const btn = document.querySelector(`.addr-search-btn[data-field="${type}"]`);
  const q = input.value.trim();
  if (q.length < 3) {
    closeSuggest(suggest);
    showToast('Введите не меньше 3 символов для поиска.', 'warn', 5000);
    return;
  }
  if (btn) {
    btn.disabled = true;
    btn.classList.add('is-busy');
  }
  showToast('Ищем адрес…', 'info', 2500);
  try {
    const res = await fetch(`${API}/geocode?q=${encodeURIComponent(q)}&limit=5`);
    let body = null;
    try {
      body = await res.json();
    } catch {
      body = null;
    }
    if (!res.ok) {
      closeSuggest(suggest);
      const d = body && typeof body.detail === 'object' ? body.detail : null;
      const msg = d && d.message
        ? d.message
        : (typeof body?.detail === 'string' ? body.detail : `Ошибка геокодинга (${res.status})`);
      showToast(msg, 'error', 9000);
      return;
    }
    const items = Array.isArray(body) ? body : [];
    if (!items.length) {
      closeSuggest(suggest);
      showToast(
        'Ничего не найдено. Уточните запрос (улица, дом, город) или выберите точку на карте.',
        'warn',
        8000,
      );
      return;
    }
    renderSuggestions(suggest, items, type);
  } catch (e) {
    closeSuggest(suggest);
    const isNet = e instanceof TypeError && String(e.message || '').toLowerCase().includes('fetch');
    showToast(
      isNet
        ? 'Не удалось связаться с сервером геокодирования. Проверьте сеть и что API запущен.'
        : (e.message || 'Ошибка при поиске адреса'),
      'error',
      9000,
    );
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.classList.remove('is-busy');
    }
  }
}

function setupGeoInput(inputEl, suggestEl, type) {
  inputEl.addEventListener('input', () => {
    const q = inputEl.value.trim();
    if (q.length < 3) closeSuggest(suggestEl);
    updateMobileRouteCompact();
  });

  inputEl.addEventListener('focus', () => { state.activeInput = type; });

  inputEl.addEventListener('keydown', (e) => {
    const lis = suggestEl.querySelectorAll('li');
    const active = suggestEl.querySelector('li.active');
    let idx = [...lis].indexOf(active);

    if (e.key === 'ArrowDown') {
      if (!lis.length) return;
      e.preventDefault();
      idx = Math.min(idx + 1, lis.length - 1);
    } else if (e.key === 'ArrowUp') {
      if (!lis.length) return;
      e.preventDefault();
      idx = Math.max(idx - 1, 0);
    } else if (e.key === 'Enter' && active) {
      e.preventDefault();
      active.click();
      return;
    } else if (e.key === 'Enter') {
      e.preventDefault();
      runAddressSearch(type);
      return;
    } else if (e.key === 'Escape') {
      closeSuggest(suggestEl);
      return;
    } else {
      return;
    }

    lis.forEach((li) => li.classList.remove('active'));
    if (lis[idx]) lis[idx].classList.add('active');
  });
}

function renderSuggestions(listEl, items, type) {
  if (!items.length) { closeSuggest(listEl); return; }
  listEl.innerHTML = items.map((it) =>
    `<li data-lat="${it.lat}" data-lon="${it.lon}">${escHtml(it.display_name)}</li>`
  ).join('');

  listEl.classList.add('open');

  listEl.querySelectorAll('li').forEach((li) => {
    li.addEventListener('click', () => {
      const lat = parseFloat(li.dataset.lat);
      const lon = parseFloat(li.dataset.lon);
      setPoint(type, lat, lon);
      const input = type === 'start' ? dom.startInput : dom.endInput;
      input.value = li.textContent;
      closeSuggest(listEl);
      updateMobileRouteCompact();
    });
  });
}

function closeSuggest(el) { el.classList.remove('open'); el.innerHTML = ''; }

async function reverseGeocode(lat, lon, type) {
  try {
    const res = await fetch(`${API}/reverse-geocode?lat=${lat}&lon=${lon}`);
    let data = null;
    try {
      data = await res.json();
    } catch {
      data = null;
    }
    if (!res.ok) {
      const d = data && typeof data.detail === 'object' ? data.detail : null;
      const msg = d && d.message ? d.message : null;
      if (res.status >= 500 && msg) {
        showToast(msg, 'error', 7000);
      }
      const input = type === 'start' ? dom.startInput : dom.endInput;
      if (!input.value.trim()) input.value = `${lat.toFixed(5)}, ${lon.toFixed(5)}`;
      syncUrlFromState();
      updateMobileRouteCompact();
      return;
    }
    const input = type === 'start' ? dom.startInput : dom.endInput;
    input.value = data.display_name || `${lat.toFixed(5)}, ${lon.toFixed(5)}`;
    syncUrlFromState();
    updateMobileRouteCompact();
  } catch (e) {
    const isNet = e instanceof TypeError && String(e.message || '').toLowerCase().includes('fetch');
    if (isNet) {
      showToast('Сетевая ошибка обратного геокодирования.', 'error', 7000);
    }
    const input = type === 'start' ? dom.startInput : dom.endInput;
    if (!input.value.trim()) input.value = `${lat.toFixed(5)}, ${lon.toFixed(5)}`;
    syncUrlFromState();
    updateMobileRouteCompact();
  }
}

/* ═══════════ Point management ════════════════════════════════ */

function setPoint(type, lat, lon) {
  if (type === 'start') state.startCoords = { lat, lon };
  else state.endCoords = { lat, lon };

  setMarker(type, lat, lon);
  updateBuildBtn();
  clearRoutes();
  if (!urlSyncSuppressed) syncUrlFromState();
}

function clearAllPoints() {
  clearUrlParams();
  if (markers.start) { markers.start.remove(); markers.start = null; }
  if (markers.end) { markers.end.remove(); markers.end = null; }
  state.startCoords = null;
  state.endCoords = null;
  dom.startInput.value = '';
  dom.endInput.value = '';
  closeSuggest(dom.startSuggest);
  closeSuggest(dom.endSuggest);
  clearRoutes();
  updateBuildBtn();
}

function resetRouteOnly() {
  state.routes = null;
  clearRoutes();
  updateBuildBtn();
  syncUrlFromState();
}

function useMyLocationForStart() {
  if (!navigator.geolocation) {
    showError('Геолокация не поддерживается браузером');
    return;
  }
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      const { latitude, longitude } = pos.coords;
      state.activeInput = 'start';
      setPoint('start', latitude, longitude);
      reverseGeocode(latitude, longitude, 'start');
      if (map) {
        map.flyTo({
          center: [longitude, latitude],
          zoom: Math.max(map.getZoom(), 15),
          duration: 800,
        });
      }
    },
    () => {
      showError('Не удалось получить координаты. Разрешите доступ к геолокации.');
    },
    { enableHighAccuracy: true, timeout: 15000, maximumAge: 60000 },
  );
}

function swapPoints() {
  const oldStart = markers.start ? markers.start.getLngLat() : null;
  const oldEnd   = markers.end   ? markers.end.getLngLat()   : null;

  if (markers.start) { markers.start.remove(); markers.start = null; }
  if (markers.end)   { markers.end.remove();   markers.end   = null; }

  const tmp = state.startCoords;
  state.startCoords = state.endCoords;
  state.endCoords = tmp;

  const tmpVal = dom.startInput.value;
  dom.startInput.value = dom.endInput.value;
  dom.endInput.value = tmpVal;

  if (state.startCoords && oldEnd)   setMarker('start', oldEnd.lat,   oldEnd.lng);
  if (state.endCoords   && oldStart) setMarker('end',   oldStart.lat, oldStart.lng);

  updateBuildBtn();
  clearRoutes();
  syncUrlFromState();
  updateMobileRouteCompact();
}

/* ═══════════ Route building ═════════════════════════════════ */

function formatAlternativesError(res, body) {
  const detailStr =
    body && typeof body.detail === 'string' && body.detail.trim() ? body.detail.trim() : null;
  const d = body && typeof body.detail === 'object' && body.detail ? body.detail : null;
  const code = d && d.code ? d.code : null;
  const serverMsg = (d && d.message) || null;
  if (code === 'NO_PATH') {
    return serverMsg || 'Маршрут между точками не найден. Попробуйте другие улицы или точки ближе к велосипедной сети.';
  }
  if (code === 'POINT_OUTSIDE_ZONE') {
    return serverMsg || 'Точка вне зоны обслуживания карты. Сдвиньте маркер в город или расширьте AREA_* на сервере.';
  }
  if (code === 'ROUTE_TOO_LONG') {
    return serverMsg || 'Маршрут слишком длинный для расчёта.';
  }
  if (code === 'OVERPASS_UNAVAILABLE') {
    return (
      serverMsg
      || 'Сервер OpenStreetMap (Overpass) сейчас недоступен или ответил с ошибкой. Повторите позже или укажите OSM_OVERPASS_URL в настройках сервера.'
    );
  }
  if (res.status === 404 && serverMsg) return serverMsg;
  /* FastAPI отдаёт 422 с detail строкой (не JSON ErrorDetail) — иначе ложно показывали «далеко от дороги». */
  if (res.status === 422) {
    if (detailStr) {
      const one = detailStr.replace(/\s+/g, ' ');
      if (one.includes('<!DOCTYPE') || one.includes('</html>') || one.length > 500) {
        return (
          'Ошибка загрузки карты дорог (Overpass/OSM), не сами координаты. '
          + 'Проверьте лог сервера и сеть; при необходимости OSM_OVERPASS_URL в .env.'
        );
      }
      return one.length > 450 ? `${one.slice(0, 450)}…` : one;
    }
    return serverMsg || 'Некорректные координаты или точка слишком далеко от дороги.';
  }
  return serverMsg || detailStr || `Ошибка сервера (${res.status})`;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function buildRoute() {
  if (!state.startCoords || !state.endCoords || state.loading) return;

  const fetchSeq = ++state.routeFetchSeq;
  const greenEnabled = dom.greenEnabledCheckbox ? dom.greenEnabledCheckbox.checked : true;
  state.greenEnabled = greenEnabled;
  syncUrlFromState();

  setLoading(true);
  hideError();

  const ac = new AbortController();
  const tAbort = setTimeout(() => ac.abort(), ROUTE_FETCH_MS);
  const tSlow = setTimeout(() => {
    showToast(
      'Сервер долго считает маршрут… Первая загрузка дорог (Overpass) в новом коридоре может занять несколько минут — запрос не прерывайте.',
      'info',
      22000,
    );
  }, ROUTE_SLOW_HINT_MS);

  async function applyRoutesPayload(data, prefer) {
    if (Array.isArray(data.routes) && data.routes.length) {
      data.routes = orderRoutesShortestFirst(data.routes);
    }
    state.routes = data;

    const rlist = data.routes || [];
    let idx = 0;
    if (Number.isFinite(prefer) && prefer >= 0 && rlist[prefer]) {
      idx = prefer;
    }
    state.selectedVariantIndex = idx;

    if (isMobileLayout()) {
      if (rlist.length) {
        state.mobileSheetPhase =
          Number.isFinite(prefer) && prefer >= 0 && rlist[prefer] ? 'detail' : 'variants';
      } else {
        state.mobileSheetPhase = 'input';
      }
    } else {
      state.mobileSheetPhase = 'input';
    }

    displayRoutes(data);
    await showResults(data, prefer);
  }

  try {
    const res = await fetch(`${API}/alternatives/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        start: state.startCoords,
        end: state.endCoords,
        profile: state.profile,
        green_enabled: greenEnabled,
      }),
      signal: ac.signal,
    });

    let body = null;
    try {
      body = await res.json();
    } catch {
      body = null;
    }

    if (!res.ok) {
      throw new Error(formatAlternativesError(res, body));
    }

    if (fetchSeq !== state.routeFetchSeq) return;

    const prefer = state.pendingVariantFromUrl;
    state.pendingVariantFromUrl = null;

    const data = { routes: body.routes || [] };
    await applyRoutesPayload(data, prefer);
    syncUrlFromState();

    state.jobPendingGreen =
      greenEnabled && !!(body.pending && body.pending.includes('green'));
    const pendingGreen = state.jobPendingGreen && body.job_id;
    if (pendingGreen) {
      setLoading(false);
      updateTripStatus();
      const jobId = body.job_id;
      let pollN = 0;
      for (;;) {
        await sleep(alternativesJobPollDelayMs(pollN));
        pollN += 1;
        if (fetchSeq !== state.routeFetchSeq) {
          state.jobPendingGreen = false;
          return;
        }
        const jr = await fetch(`${API}/alternatives/job/${encodeURIComponent(jobId)}`, {
          signal: ac.signal,
        });
        let jd = null;
        try {
          jd = await jr.json();
        } catch {
          jd = null;
        }
        if (!jr.ok) {
          state.jobPendingGreen = false;
          updateTripStatus();
          const detail = jd && jd.detail;
          const code =
            detail && typeof detail === 'object' && detail.code ? detail.code : null;
          if (jr.status === 404 && code === 'JOB_NOT_FOUND') {
            showToast(
              'Задача устарела или сервер перезапущен. Нажмите «Построить маршрут» снова.',
              'warn',
              12000,
            );
          } else {
            showToast('Не удалось получить статус маршрута с озеленением.', 'warn', 8000);
          }
          break;
        }
        if (fetchSeq !== state.routeFetchSeq) {
          state.jobPendingGreen = false;
          return;
        }
        if (jd.status === 'failed') {
          state.jobPendingGreen = false;
          const em =
            jd.error && jd.error.message
              ? jd.error.message
              : 'Вариант с озеленением не удалось построить.';
          if (jd.green_warning) {
            showToast(`${jd.green_warning} (${em})`, 'warn', 12000);
          } else {
            showToast(em, 'warn', 10000);
          }
          await applyRoutesPayload({ routes: jd.routes || [] }, state.selectedVariantIndex);
          syncUrlFromState();
          updateTripStatus();
          break;
        }
        if (jd.green_warning && jd.status !== 'failed') {
          showToast(jd.green_warning, 'warn', 10000);
        }
        await applyRoutesPayload({ routes: jd.routes || [] }, state.selectedVariantIndex);
        syncUrlFromState();
        const rlist = jd.routes || [];
        state.jobPendingGreen =
          Array.isArray(jd.pending) && jd.pending.includes('green') && !rlist.some((r) => r.mode === 'green');
        if (!jd.pending || !jd.pending.length) {
          state.jobPendingGreen = false;
          updateTripStatus();
          break;
        }
        updateTripStatus();
      }
    } else {
      state.jobPendingGreen = false;
    }
  } catch (e) {
    state.jobPendingGreen = false;
    state.pendingVariantFromUrl = null;
    if (e.name === 'AbortError') {
      showToast(
        'Превышено время ожидания (10 мин). Повторите запрос: после прогрева кэша ответ обычно быстрый. Для ускорения первого запуска можно выключить спутник (DISABLE_SATELLITE_GREEN).',
        'error',
        12000,
      );
    } else {
      const isNet = e instanceof TypeError && String(e.message || '').toLowerCase().includes('fetch');
      if (isNet) {
        showToast(
          'Не удалось связаться с сервером маршрутизации. Запустите API (python -m bike_router) и обновите страницу.',
          'error',
        );
      } else {
        showToast(e.message || 'Не удалось построить маршрут', 'error');
      }
    }
  } finally {
    clearTimeout(tAbort);
    clearTimeout(tSlow);
    setLoading(false);
    updateTripStatus();
  }
}

/* ═══════════ UI — Results ═══════════════════════════════════ */

async function refreshSatelliteGreenFlag() {
  try {
    const res = await fetch(`${API}/health`);
    const data = await res.json();
    if (typeof data.satellite_green_enabled === 'boolean') {
      serverSatelliteGreenEnabled = data.satellite_green_enabled;
    }
  } catch {
    /* оставляем предыдущее значение */
  }
}

async function showResults(data, preferredVariantIndex = null) {
  await refreshSatelliteGreenFlag();
  const routes = data.routes || [];
  dom.results.classList.remove('hidden');
  dom.results.setAttribute('aria-hidden', 'false');
  dom.results.classList.toggle('mobile-has-routes', routes.length > 0);
  dom.results.classList.toggle('results--no-variants', routes.length === 0);

  let idx = 0;
  if (
    Number.isFinite(preferredVariantIndex)
    && preferredVariantIndex >= 0
    && routes[preferredVariantIndex]
  ) {
    idx = preferredVariantIndex;
  }
  state.selectedVariantIndex = idx;

  if (dom.resultsEmpty) {
    if (routes.length) {
      dom.resultsEmpty.classList.add('hidden');
      dom.resultsEmpty.setAttribute('aria-hidden', 'true');
      dom.resultsEmpty.textContent = '';
    } else {
      dom.resultsEmpty.textContent = 'Варианты маршрута не получены. Попробуйте другие точки или профиль.';
      dom.resultsEmpty.classList.remove('hidden');
      dom.resultsEmpty.setAttribute('aria-hidden', 'false');
    }
  }

  buildVariantTabs(routes);

  const isM = isMobileLayout();
  if (routes.length) {
    if (isM && state.mobileSheetPhase === 'variants') {
      applyRouteHighlightEqual(routes.length);
      dom.routeDetailCard.innerHTML = '';
      dom.elevChart.innerHTML = '';
      dom.variantTabs.querySelectorAll('.variant-tab').forEach((b) => {
        b.classList.remove('active');
        b.setAttribute('aria-selected', 'false');
      });
      updateOverlays(null);
      renderMapLegend(routes);
    } else {
      selectVariant(idx, routes);
    }
  } else {
    clearOverlaySources();
    renderMapLegend(routes);
  }
  syncUrlFromState();
  syncMobileSheetUi();
  updateTripStatus();
}

function selectVariant(idx, routes) {
  if (!routes || idx < 0 || idx >= routes.length) return;
  state.selectedVariantIndex = idx;
  if (isMobileLayout() && routes.length) {
    state.mobileSheetPhase = 'detail';
    state.mobileSheetDetailExpanded = false;
  }

  dom.variantTabs.querySelectorAll('.variant-tab').forEach((b, j) => {
    b.classList.toggle('active', j === idx);
    b.setAttribute('aria-selected', j === idx ? 'true' : 'false');
  });

  const r = routes[idx];
  renderRouteDetailCard(r, routes, idx);
  renderElevChartForRoute(r);
  applyRouteHighlight(idx, routes.length);
  updateOverlays(r);
  renderMapLegend(routes);
  syncUrlFromState();
  syncMobileSheetUi();
  updateTripStatus();
}

/** Placeholder зелёного маршрута: только пока API ждёт green и в routes ещё нет mode===green. */
function greenRoutePlaceholderVisible(routes) {
  if (!state.greenEnabled) return false;
  const rlist = routes || [];
  if (rlist.some((r) => r.mode === 'green')) return false;
  return state.jobPendingGreen;
}

/** Одна строка варианта: короткое время из time_s (м:сс или ч:мм). */
function formatVariantTimeShort(route) {
  const s = route?.time_s;
  if (Number.isFinite(s) && s >= 0) {
    const total = Math.round(s);
    const m = Math.floor(total / 60);
    const sec = total % 60;
    if (m >= 60) {
      const h = Math.floor(m / 60);
      const mm = m % 60;
      return `${h}:${String(mm).padStart(2, '0')}`;
    }
    return `${m}:${String(sec).padStart(2, '0')}`;
  }
  const td = route?.time_display;
  if (td != null && String(td).trim()) return escHtml(String(td).trim());
  return '—';
}

function buildVariantTabs(routes) {
  const main = routes.map((r, i) => {
    const col = lineColorForMode(r.mode);
    const rawLabel = String(r.variant_label || r.mode || '').trim();
    const labelDesk = escHtml(rawLabel);
    const labelMobile = escHtml(truncMobileLabel(rawLabel, MOBILE_VARIANT_TAB_LABEL_MAX));
    const active = i === state.selectedVariantIndex ? ' active' : '';
    const meta = `${fmtDist(r.length_m)} · ${escHtml(String(r.time_display ?? ''))} · ${r.elevation.climb_m.toFixed(0)} м`;
    const tShort = formatVariantTimeShort(r);
    const mobileLine = `<span class="variant-tab-mobile-line"><span class="swatch" style="background:${col}"></span><span class="vt-one-row"><strong title="${escHtml(rawLabel)}">${labelMobile}</strong><span class="vt-sep"> · </span><span class="vt-metrics">${fmtDist(r.length_m)} · ${tShort} · ${r.elevation.climb_m.toFixed(0)} м</span></span></span>`;
    const desk = `<span class="variant-tab-desktop-stack"><span class="variant-tab-row"><span class="swatch" style="background:${col}"></span><span class="variant-tab-title">${labelDesk}</span></span><span class="variant-tab-meta">${meta}</span></span>`;
    return `<button type="button" class="variant-tab${active}" data-idx="${i}" role="tab" aria-selected="${i === state.selectedVariantIndex}">
      ${mobileLine}
      ${desk}
    </button>`;
  }).join('');
  const pendingTab = greenRoutePlaceholderVisible(routes)
      ? `<div class="variant-tab variant-tab-pending" role="status" aria-live="polite">
          <span class="variant-tab-desktop-stack"><span class="variant-tab-row"><span class="swatch swatch-pending"></span><span class="variant-tab-title">С учётом озеленения — вычисляется…</span></span><span class="variant-tab-meta variant-tab-meta-muted">Подождите, спутниковый анализ</span></span>
          <span class="variant-tab-mobile-line variant-tab-mobile-pending"><span class="swatch swatch-pending"></span><span class="vt-one-row"><strong>С учётом озеленения</strong><span class="vt-sep"> · </span><span class="vt-metrics">вычисляется…</span></span></span>
        </div>`
      : '';
  dom.variantTabs.innerHTML = main + pendingTab;

  dom.variantTabs.querySelectorAll('.variant-tab[data-idx]').forEach((btn) => {
    btn.addEventListener('click', () => {
      selectVariant(parseInt(btn.dataset.idx, 10), routes);
    });
  });
}

/** Доли fallback по сумме длин рёбер (как на backend в quality_hints). */
function renderQualitySection(qh, compact) {
  const naS = ((qh?.na_surface_fraction ?? 0) * 100).toFixed(0);
  const infS = ((qh?.inferred_surface_fraction ?? 0) * 100).toFixed(0);
  const infH = ((qh?.inferred_highway_fraction ?? 0) * 100).toFixed(0);
  const warns = qh?.warnings || [];
  if (compact) {
    const wShort = warns.length
      ? ` · ⚠ ${warns.length}`
      : '';
    return `
      <div class="quality-block span-2">
        <span class="d-label">OSM</span>
        <p class="quality-metrics-line">Без surface ~${naS}% · surface ~${infS}% · highway ~${infH}%${wShort}</p>
      </div>
    `;
  }
  const warnBlock = warns.length
    ? `<div class="quality-warn-title">Предупреждения</div><ul class="quality-warn-list">${warns.map((w) => `<li>${escHtml(w)}</li>`).join('')}</ul>`
    : '<p class="quality-ok-note">Критичных предупреждений по порогам нет.</p>';
  return `
    <div class="quality-block span-2">
      <span class="d-label">Качество данных (OSM)</span>
      <p class="quality-metrics-line">Оценка неизвестных тегов — по <strong>сумме длин рёбер</strong> (совпадает с длиной маршрута в карточке): без surface ~${naS}% · коэфф. 1.0 по surface ~${infS}% · по highway ~${infH}%.</p>
      ${warnBlock}
    </div>
  `;
}

function renderVariantDiffMobile(route, routes, idx) {
  const base = routes[0];
  if (!base || routes.length < 2) return '';
  if (idx === 0) {
    return '<p class="variant-diff variant-diff-baseline mobile-diff-one">Опорный — кратчайший по длине.</p>';
  }
  const parts = [];
  const dl = route.length_m - base.length_m;
  if (Math.abs(dl) >= 1) {
    parts.push(dl > 0 ? `+${fmtDist(dl)}` : `−${fmtDist(-dl)}`);
  }
  const dc = route.elevation.climb_m - base.elevation.climb_m;
  if (Math.abs(dc) >= 3) {
    parts.push(dc > 0 ? `+${dc.toFixed(0)} м набора` : `−${(-dc).toFixed(0)} м набора`);
  }
  if (state.greenEnabled) {
    const dg = route.green.percent - base.green.percent;
    if (Math.abs(dg) >= 0.5) {
      const sign = dg > 0 ? '+' : '';
      parts.push(`${sign}${dg.toFixed(0)} п.п. зелени`);
    }
  }
  const text = parts.length ? parts.join(' · ') : 'Близко к опорному.';
  return `<p class="variant-diff mobile-diff-one"><strong>К опорному:</strong> ${escHtml(text)}</p>`;
}

function renderVariantDiff(route, routes, idx) {
  const base = routes[0];
  if (!base || routes.length < 2) return '';
  const baseName = escHtml(base.variant_label || base.mode);
  if (idx === 0) {
    return `<p class="variant-diff variant-diff-baseline">Опорный вариант «${baseName}» — <strong>кратчайший по длине на карте</strong> среди предложенных. Остальные строки в таблице отличаются от него (длина, время, рельеф, зелень и т.д.).</p>`;
  }
  const parts = [];
  const dl = route.length_m - base.length_m;
  if (Math.abs(dl) >= 1) {
    parts.push(dl > 0 ? `длиннее на ${fmtDist(dl)}` : `короче на ${fmtDist(-dl)}`);
  }
  const dt = route.time_s - base.time_s;
  if (Math.abs(dt) >= 30) {
    parts.push(dt > 0 ? `≈ на ${Math.round(dt / 60)} мин дольше по времени` : `≈ на ${Math.round(-dt / 60)} мин быстрее`);
  }
  const dc = route.elevation.climb_m - base.elevation.climb_m;
  if (Math.abs(dc) >= 5) {
    parts.push(dc > 0 ? `набор на ${dc.toFixed(0)} м больше` : `набор на ${(-dc).toFixed(0)} м меньше`);
  }
  const dmx = route.elevation.max_gradient_pct - base.elevation.max_gradient_pct;
  if (Math.abs(dmx) >= 0.5) {
    parts.push(dmx > 0 ? `макс. уклон на ${dmx.toFixed(1)} п.п. круче` : `макс. уклон на ${(-dmx).toFixed(1)} п.п. положе`);
  }
  if (state.greenEnabled) {
    const dg = route.green.percent - base.green.percent;
    if (Math.abs(dg) >= 0.5) {
      parts.push(dg > 0 ? `озеленение выше на ${dg.toFixed(1)} п.п.` : `озеленение ниже на ${(-dg).toFixed(1)} п.п.`);
    }
  }
  const ds = route.stairs.count - base.stairs.count;
  if (ds !== 0) {
    parts.push(ds > 0 ? `на ${ds} участ. лестниц больше` : `на ${-ds} участ. лестниц меньше`);
  }
  const text = parts.length
    ? parts.join('; ') + '.'
    : 'По ключевым метрикам близок к кратчайшему варианту; подробности — в таблице сравнения и карточке.';
  return `<p class="variant-diff"><strong>Чем отличается от кратчайшего «${baseName}»:</strong> ${text}</p>`;
}

function renderRouteDetailCard(route, routes, idx) {
  const e = route.elevation;
  const g = route.green;
  const s = route.stairs;
  const na = route.surfaces.na_fraction;
  const title = escHtml(route.variant_label || route.mode);
  const compact = isMobileLayout();
  const greenOff = !state.greenEnabled;
  const diffBlock = Array.isArray(routes)
    ? (compact ? renderVariantDiffMobile(route, routes, idx) : renderVariantDiff(route, routes, idx))
    : '';
  const qSection = renderQualitySection(route.quality_hints || {}, compact);
  const qFull = renderQualitySection(route.quality_hints || {}, false);
  const greenSrvNote = greenOff
    ? `<div class="detail-row span-2 green-satellite-off">
        <p class="green-satellite-off-text">Учёт озеленения отключён — метрики зелени по снимкам для этого запроса не использовались.</p>
      </div>`
    : !serverSatelliteGreenEnabled
      ? `<div class="detail-row span-2 green-satellite-off">
        <p class="green-satellite-off-text">Озеленение по снимкам на сервере выключено (<code>DISABLE_SATELLITE_GREEN</code>). Ниже — нули-заглушки; режим «С учётом озеленения» совпадает с энергетическим по весам.</p>
      </div>`
      : '';

  if (compact) {
    const moreInner = `
      ${diffBlock}
      ${greenSrvNote}
      <div class="detail-grid">
        <div class="detail-row">
          <span class="d-label">Деревья (средн.)</span>
          <span class="d-val">${greenOff ? '—' : `${g.avg_trees_pct.toFixed(1)}%`}</span>
          ${!greenOff && !serverSatelliteGreenEnabled ? '<span class="d-note">снимки выкл.</span>' : ''}
        </div>
        <div class="detail-row span-2">
          <span class="d-label">Нет surface в OSM</span>
          <span class="d-val small">${(na * 100).toFixed(1)}%</span>
          <span class="d-note">доля длины маршрута без тега surface в OSM</span>
        </div>
      </div>
      ${qFull}
    `;
    dom.routeDetailCard.innerHTML = `
      <h3 class="route-detail-title-fallback">${title}</h3>
      <div class="detail-grid">
        <div class="detail-row">
          <span class="d-label">Длина</span>
          <span class="d-val">${fmtDist(route.length_m)}</span>
        </div>
        <div class="detail-row">
          <span class="d-label">Время</span>
          <span class="d-val small">${escHtml(route.time_display)}</span>
        </div>
        <div class="detail-row">
          <span class="d-label">Набор</span>
          <span class="d-val">${e.climb_m.toFixed(0)} м</span>
          <span class="d-note">спуск ${e.descent_m.toFixed(0)} м</span>
        </div>
        <div class="detail-row">
          <span class="d-label">Макс. уклон</span>
          <span class="d-val">${e.max_gradient_pct.toFixed(1)}%</span>
        </div>
        <div class="detail-row">
          <span class="d-label">Зелень</span>
          <span class="d-val" style="color:var(--green)">${greenOff ? '—' : `${g.percent.toFixed(1)}%`}</span>
        </div>
        <div class="detail-row span-2">
          <span class="d-label">Лестницы</span>
          <span class="d-val small">${s.count} шт · ${s.total_length_m.toFixed(0)} м</span>
        </div>
      </div>
      <details class="mobile-route-more">
        <summary>Подробнее</summary>
        <div class="mobile-route-more-inner">${moreInner}</div>
      </details>
    `;
  } else {
    dom.routeDetailCard.innerHTML = `
    <h3>${title}</h3>
    ${diffBlock}
    <div class="detail-grid">
      <div class="detail-row">
        <span class="d-label">Длина</span>
        <span class="d-val">${fmtDist(route.length_m)}</span>
      </div>
      <div class="detail-row">
        <span class="d-label">Время</span>
        <span class="d-val small">${escHtml(route.time_display)}</span>
      </div>
      <div class="detail-row">
        <span class="d-label">Набор высоты</span>
        <span class="d-val">${e.climb_m.toFixed(0)} м</span>
        <span class="d-note">спуск ${e.descent_m.toFixed(0)} м</span>
      </div>
      <div class="detail-row">
        <span class="d-label">Макс. уклон</span>
        <span class="d-val">${e.max_gradient_pct.toFixed(1)}%</span>
      </div>
      ${greenSrvNote}
      <div class="detail-row">
        <span class="d-label">Озеленение</span>
        <span class="d-val" style="color:var(--green)">${greenOff ? '—' : `${g.percent.toFixed(1)}%`}</span>
        <span class="d-note">${greenOff ? 'не учитывалось в запросе' : `доля «зелёных» рёбер${!serverSatelliteGreenEnabled ? ' (снимки выкл.)' : ''}`}</span>
      </div>
      <div class="detail-row">
        <span class="d-label">Деревья (средн.)</span>
        <span class="d-val">${greenOff ? '—' : `${g.avg_trees_pct.toFixed(1)}%`}</span>
        ${!greenOff && !serverSatelliteGreenEnabled ? '<span class="d-note">анализ спутниковых тайлов отключён на сервере</span>' : ''}
      </div>
      <div class="detail-row span-2">
        <span class="d-label">Лестницы</span>
        <span class="d-val small">${s.count} шт · ${s.total_length_m.toFixed(0)} м суммарно по рёбрам</span>
      </div>
      <div class="detail-row span-2">
        <span class="d-label">Нет surface в OSM</span>
        <span class="d-val small">${(na * 100).toFixed(1)}%</span>
        <span class="d-note">доля длины маршрута без тега surface в OSM; см. блок «Качество данных» выше</span>
      </div>
      ${qSection}
    </div>
  `;
  }
  dom.routeDetailCard.classList.toggle('route-detail-card--mobile-compact', compact);
}

function renderElevChartForRoute(route) {
  const profile = route?.elevation_profile;
  if (!profile || profile.length < 2) { dom.elevChart.innerHTML = ''; return; }

  const W = 320, H = 70, PX = 4, PY = 4;
  const dists = profile.map((p) => p.distance_m);
  const elevs = profile.map((p) => p.elevation_m);
  const maxD = Math.max(...dists) || 1;
  const minE = Math.min(...elevs);
  const maxE = Math.max(...elevs);
  const rangeE = maxE - minE || 1;

  const x = (d) => PX + (d / maxD) * (W - 2 * PX);
  const y = (el) => PY + (1 - (el - minE) / rangeE) * (H - 2 * PY);

  const pathD = profile.map((p, i) =>
    `${i === 0 ? 'M' : 'L'}${x(p.distance_m).toFixed(1)},${y(p.elevation_m).toFixed(1)}`
  ).join(' ');

  const areaD = pathD
    + ` L${x(dists[dists.length - 1]).toFixed(1)},${(H - PY).toFixed(1)}`
    + ` L${x(0).toFixed(1)},${(H - PY).toFixed(1)} Z`;

  const lineColor = lineColorForMode(route.mode);
  const gid = `eg-${Date.now()}`;

  dom.elevChart.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      <defs>
        <linearGradient id="${gid}" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="${lineColor}" stop-opacity="0.28"/>
          <stop offset="100%" stop-color="${lineColor}" stop-opacity="0.02"/>
        </linearGradient>
      </defs>
      <path d="${areaD}" fill="url(#${gid})"/>
      <path d="${pathD}" fill="none" stroke="${lineColor}" stroke-width="1.5" stroke-linejoin="round"/>
      <text x="${PX}" y="${H - 1}" font-size="7" fill="#94a3b8">${minE.toFixed(0)} м</text>
      <text x="${W - PX}" y="${PY + 6}" font-size="7" fill="#94a3b8" text-anchor="end">${maxE.toFixed(0)} м</text>
    </svg>
  `;
}

/* ═══════════ UI — Helpers ═══════════════════════════════════ */

function updateBuildBtn() {
  dom.buildBtn.disabled = !(state.startCoords && state.endCoords);
}

function setToolbarDisabled(on) {
  if (dom.locateStartBtn) dom.locateStartBtn.disabled = on;
  if (dom.clearPointsBtn) dom.clearPointsBtn.disabled = on;
  if (dom.resetRouteBtn) dom.resetRouteBtn.disabled = on;
}

function setLoading(on) {
  state.loading = on;
  dom.buildText.classList.toggle('hidden', on);
  dom.spinner.classList.toggle('hidden', !on);
  if (on) {
    dom.buildBtn.disabled = true;
  } else {
    updateBuildBtn();
  }
  setToolbarDisabled(on);
  updateTripStatus();
}

function showToast(msg, variant = 'error', durationMs = 8000) {
  if (!dom.errorToast) return;
  dom.errorToast.textContent = msg;
  dom.errorToast.className = `toast toast-${variant}`;
  dom.errorToast.classList.remove('hidden');
  clearTimeout(dom.errorToast._hideT);
  dom.errorToast._hideT = setTimeout(() => hideError(), durationMs);
}

function showError(msg) {
  showToast(msg, 'error');
}

function hideError() {
  if (dom.errorToast) dom.errorToast.classList.add('hidden');
}

function fmtDist(m) {
  return m >= 1000 ? `${(m / 1000).toFixed(2)} км` : `${m.toFixed(0)} м`;
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

/* ═══════════ Event listeners ════════════════════════════════ */

function setupEvents() {
  let sheetTouchY0 = null;
  let sheetTouchMoved = false;
  let sheetSuppressClick = false;

  setupGeoInput(dom.startInput, dom.startSuggest, 'start');
  setupGeoInput(dom.endInput, dom.endSuggest, 'end');
  dom.startInput.addEventListener('focus', () => expandMobileInputSheet());
  dom.endInput.addEventListener('focus', () => expandMobileInputSheet());

  document.querySelectorAll('.addr-search-btn').forEach((btn) => {
    btn.addEventListener('click', () => runAddressSearch(btn.dataset.field));
  });

  dom.swapBtn.addEventListener('click', swapPoints);
  if (dom.sheetHandle) {
    const TH = 40;
    dom.sheetHandle.addEventListener(
      'touchstart',
      (e) => {
        if (e.touches.length !== 1) return;
        sheetTouchY0 = e.touches[0].clientY;
        sheetTouchMoved = false;
      },
      { passive: true },
    );
    dom.sheetHandle.addEventListener(
      'touchmove',
      (e) => {
        if (sheetTouchY0 == null || e.touches.length !== 1) return;
        if (Math.abs(e.touches[0].clientY - sheetTouchY0) > 10) sheetTouchMoved = true;
      },
      { passive: true },
    );
    dom.sheetHandle.addEventListener(
      'touchend',
      (e) => {
        if (sheetTouchY0 == null) return;
        const y = e.changedTouches[0].clientY;
        const dy = y - sheetTouchY0;
        sheetTouchY0 = null;
        if (
          !isMobileLayout()
          || state.mobileSheetPhase !== 'input'
          || (state.routeList && state.routeList.length)
        ) {
          sheetTouchMoved = false;
          return;
        }
        if (sheetTouchMoved) {
          let acted = false;
          if (dy < -TH && !state.mobileInputSheetExpanded) {
            state.mobileInputSheetExpanded = true;
            acted = true;
          } else if (dy > TH && state.mobileInputSheetExpanded) {
            state.mobileInputSheetExpanded = false;
            acted = true;
          }
          if (acted) {
            syncMobileSheetUi();
            requestAnimationFrame(() => fitAllRoutesInView());
            sheetSuppressClick = true;
          }
        }
        sheetTouchMoved = false;
      },
      { passive: true },
    );
    dom.sheetHandle.addEventListener('click', (ev) => {
      if (sheetSuppressClick) {
        sheetSuppressClick = false;
        ev.preventDefault();
        return;
      }
      toggleMobileInputSheetPeek();
    });
  }
  dom.buildBtn.addEventListener('click', buildRoute);

  if (dom.locateStartBtn) dom.locateStartBtn.addEventListener('click', useMyLocationForStart);
  if (dom.clearPointsBtn) dom.clearPointsBtn.addEventListener('click', clearAllPoints);
  if (dom.resetRouteBtn) dom.resetRouteBtn.addEventListener('click', resetRouteOnly);
  if (dom.fitRoutesBtn) dom.fitRoutesBtn.addEventListener('click', fitAllRoutesInView);

  dom.profileBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      dom.profileBtns.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      state.profile = btn.dataset.value;
      if (state.routes) clearRoutes();
      syncUrlFromState();
      updateTripStatus();
    });
  });

  if (dom.greenEnabledCheckbox) {
    dom.greenEnabledCheckbox.addEventListener('change', () => {
      state.greenEnabled = dom.greenEnabledCheckbox.checked;
      syncUrlFromState();
    });
  }

  if (dom.copyShareBtn) {
    dom.copyShareBtn.addEventListener('click', async () => {
      syncUrlFromState();
      const url = window.location.href;
      try {
        await navigator.clipboard.writeText(url);
        showToast('Ссылка скопирована в буфер обмена', 'info', 3500);
      } catch {
        showToast(`Не удалось скопировать автоматически. Адрес страницы:\n${url}`, 'warn', 12000);
      }
    });
  }

  document.addEventListener('click', (e) => {
    if (!e.target.closest('.addr-input-wrap')) {
      closeSuggest(dom.startSuggest);
      closeSuggest(dom.endSuggest);
    }
    if (
      dom.mapContextMenu
      && !dom.mapContextMenu.classList.contains('hidden')
      && !e.target.closest('#map-context-menu')
    ) {
      closeMapContextMenu();
    }
    if (
      dom.mapHintTooltip
      && !dom.mapHintTooltip.classList.contains('hidden')
      && !e.target.closest('#map-hint-toggle')
      && !e.target.closest('#map-hint-tooltip')
    ) {
      dom.mapHintTooltip.classList.add('hidden');
    }
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' || e.key === 'Esc') {
      closeMapContextMenu();
    }
  });

  window.addEventListener('resize', () => {
    closeMapContextMenu();
  });

  if (dom.mapContextMenu) {
    dom.mapContextMenu.addEventListener('click', (ev) => {
      ev.stopPropagation();
    });
    dom.mapContextMenu.addEventListener('mousedown', (ev) => {
      ev.stopPropagation();
    });
    dom.mapContextMenu.querySelectorAll('[data-which]').forEach((btn) => {
      btn.addEventListener('click', (ev) => {
        ev.preventDefault();
        const w = btn.getAttribute('data-which');
        if (w === 'start' || w === 'end') applyMapContextMenuChoice(w);
      });
    });
  }

  if (dom.mapHintToggle && dom.mapHintTooltip) {
    dom.mapHintToggle.addEventListener('click', (e) => {
      e.stopPropagation();
      dom.mapHintTooltip.classList.toggle('hidden');
    });
  }

  if (dom.mobileLocateBtn) {
    dom.mobileLocateBtn.addEventListener('click', () => useMyLocationForStart());
  }
  if (dom.mobileEditPointsBtn) {
    dom.mobileEditPointsBtn.addEventListener('click', () => mobileEditPoints());
  }
  if (dom.mobileBackVariantsBtn) {
    dom.mobileBackVariantsBtn.addEventListener('click', () => mobileBackToVariants());
  }
  if (dom.mobileDetailExpandBtn) {
    dom.mobileDetailExpandBtn.addEventListener('click', () => toggleMobileDetailSheetExpand());
  }
}

/* ═══════════ Health check & version ═════════════════════════ */

async function loadHealth() {
  try {
    const res = await fetch(`${API}/health`);
    const data = await res.json();
    const vLabel = `v${data.version}`;
    if (dom.versionText) dom.versionText.textContent = vLabel;
    if (dom.versionTextMobile) dom.versionTextMobile.textContent = vLabel;
    if (typeof data.satellite_green_enabled === 'boolean') {
      serverSatelliteGreenEnabled = data.satellite_green_enabled;
    }
  } catch { /* silent */ }
}

/* ═══════════ Init ═══════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
  hydrateStateFromUrl();
  initMap();
  setupEvents();
  loadHealth();
  if (MQ_MOBILE) {
    MQ_MOBILE.addEventListener('change', () => {
      closeMapContextMenu();
      syncMobileSheetUi();
      updateTripStatus();
      if (map && state.routeList && state.routeList.length) {
        if (!isMobileLayout()) {
          applyRouteHighlight(state.selectedVariantIndex, state.routeList.length);
        } else {
          syncRouteHighlightFromUi();
        }
      }
    });
  }
  syncMobileSheetUi();
  updateTripStatus();
});
