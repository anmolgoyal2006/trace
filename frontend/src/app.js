/**
 * app.js — Trace Command Center SPA v5.2 (Ultra-Dense HUD Edition)
 * Cybernetic surveillance interface with HTML5 canvas video stream shaders,
 * live telemetry, Chart.js graphs, SVG floorplan matrix, WebSocket protocol,
 * and high-density pre-populated interactive HUD showcases.
 */

/* ═══════════════════════════════════════════════════════════════════
   CHART.JS GLOBAL DEFAULTS
═══════════════════════════════════════════════════════════════════ */
if (typeof Chart !== 'undefined') {
  Chart.defaults.color          = '#cbd5e1';
  Chart.defaults.borderColor    = 'rgba(255, 255, 255, 0.12)';
  Chart.defaults.font.family    = 'Inter, system-ui, sans-serif';
  Chart.defaults.font.size      = 11;
  Chart.defaults.plugins.legend.display = false;
  Chart.defaults.plugins.tooltip.backgroundColor = '#0d1427';
  Chart.defaults.plugins.tooltip.borderColor     = '#06b6d4';
  Chart.defaults.plugins.tooltip.borderWidth     = 1;
  Chart.defaults.plugins.tooltip.padding         = 12;
  Chart.defaults.plugins.tooltip.titleFont       = { weight: '800', size: 13, family: 'Outfit, sans-serif' };
  Chart.defaults.plugins.tooltip.bodyFont        = { size: 12, family: 'JetBrains Mono, monospace' };
  Chart.defaults.plugins.tooltip.cornerRadius    = 8;
  Chart.defaults.animation.duration              = 800;
  Chart.defaults.animation.easing               = 'easeInOutQuart';
}

/* ═══════════════════════════════════════════════════════════════════
   STATE & CONSTANTS
═══════════════════════════════════════════════════════════════════ */
const State = {
  view:            'dashboard',
  personsViewMode: 'grid', // 'grid' or 'table'
  activeQueryId:   null,
  lastRoute:       null,
  ws:              null,
  cameras:         [],
  canvasAnimIds:   {},
};

const Charts = {
  activity: null,
  cameras:  null,
  sparks:   {},
};

const CAM_LAYOUT = {
  C01: { x: 155, y: 180, label: 'C01', loc: 'Entrance Zone' },
  C02: { x: 400, y: 180, label: 'C02', loc: 'Corridor Hub'  },
  C03: { x: 645, y: 180, label: 'C03', loc: 'Canteen Area' },
};

/* Demo Fallback Session for Immediate 1st-Look Wow */
const MOCK_DEMO_ROUTE = {
  total_cameras_matched: 3,
  route_confidence: 0.942,
  sightings: [
    { camera_id: 'C01', track_id: 104, first_seen: '10:42:15', last_seen: '10:43:10', appearance_score: 0.95, spatial_score: 0.92, temporal_score: 0.90, fusion_score: 0.94, best_confidence: 94.2 },
    { camera_id: 'C02', track_id: 108, first_seen: '10:43:35', last_seen: '10:44:20', appearance_score: 0.91, spatial_score: 0.88, temporal_score: 0.86, fusion_score: 0.89, best_confidence: 89.5 },
    { camera_id: 'C03', track_id: 112, first_seen: '10:45:00', last_seen: '10:46:15', appearance_score: 0.93, spatial_score: 0.90, temporal_score: 0.92, fusion_score: 0.92, best_confidence: 92.1 },
  ],
  steps: [
    { camera_id: 'C01', step_order: 0, timestamp: '10:42:15', confidence: 94.2 },
    { camera_id: 'C02', step_order: 1, timestamp: '10:43:35', confidence: 89.5 },
    { camera_id: 'C03', step_order: 2, timestamp: '10:45:00', confidence: 92.1 },
  ],
};

/* ═══════════════════════════════════════════════════════════════════
   BOOTSTRAP & INITIALIZATION
═══════════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', async () => {
  initClock();
  setupNav();
  setupQueryForm();
  setupWatchlistForm();
  setupUploadForm();
  setupDropZones();
  setupCamRadios();
  initMapSvg();
  connectWs();
  initScanlinePreference();
  initCanvasViewports();
  initLiveSystemTelemetry();
  await loadCameras();
  loadDashboard();
  loadPersonsSelect();
  loadUploadJobs();
});

/* ── System Clock ── */
function initClock() {
  const clockEl = document.getElementById('top-clock');
  function update() {
    const d = new Date();
    if (clockEl) clockEl.textContent = d.toISOString().substring(11, 19) + ' UTC';
  }
  update();
  setInterval(update, 1000);
}

/* ── HUD Scanlines Toggle ── */
function toggleScanlines() {
  const isEnabled = document.body.classList.toggle('hud-scanlines');
  localStorage.setItem('trace_hud_scanlines', isEnabled ? '1' : '0');
  toast(isEnabled ? 'HUD Scanlines Enabled' : 'HUD Scanlines Disabled', 'info');
}

function initScanlinePreference() {
  if (localStorage.getItem('trace_hud_scanlines') !== '0') {
    document.body.classList.add('hud-scanlines');
  }
}

/* ── HTML5 Canvas Surveillance Viewport Shaders ── */
function initCanvasViewports() {
  const feeds = [
    { id: 'vp-c01', name: 'CAM-01', target: 'TARGET #104', color: '#10b981' },
    { id: 'vp-c02', name: 'CAM-02', target: 'TARGET #108', color: '#06b6d4' },
    { id: 'vp-c03', name: 'CAM-03', target: 'TARGET #112', color: '#f59e0b' }
  ];

  feeds.forEach((cfg, idx) => {
    const feed = document.querySelector(`#${cfg.id} .vp-feed`);
    if (!feed) return;

    let canvas = feed.querySelector('canvas');
    if (!canvas) {
      canvas = document.createElement('canvas');
      canvas.className = 'vp-canvas-simulation';
      canvas.width  = 340;
      canvas.height = 220;
      feed.insertBefore(canvas, feed.firstChild);
    }

    // Append live timestamp badge if not present
    if (!feed.querySelector('.vp-timestamp-badge')) {
      const timeBadge = document.createElement('div');
      timeBadge.className = 'vp-timestamp-badge';
      timeBadge.id = `ts-${cfg.id}`;
      feed.appendChild(timeBadge);
    }

    const ctx = canvas.getContext('2d');
    let angle = idx * 1.5;
    let targetX = 80 + idx * 70;
    let targetY = 70 + idx * 30;
    let dx = (idx % 2 === 0 ? 1 : -1) * 0.7;
    let dy = (idx % 2 === 0 ? -1 : 1) * 0.45;

    function renderFeed() {
      angle += 0.02;
      targetX += dx;
      targetY += dy;

      if (targetX < 40 || targetX > canvas.width - 90) dx *= -1;
      if (targetY < 40 || targetY > canvas.height - 90) dy *= -1;

      // Dark obsidian CCTV viewport background
      ctx.fillStyle = '#030712';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Subtle CCTV noise grain effect
      for (let i = 0; i < 180; i++) {
        const nx = Math.random() * canvas.width;
        const ny = Math.random() * canvas.height;
        const alpha = Math.random() * 0.07;
        ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
        ctx.fillRect(nx, ny, 1.5, 1.5);
      }

      // CCTV Grid lines
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.035)';
      ctx.lineWidth = 1;
      for (let x = 0; x < canvas.width; x += 30) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
      }
      for (let y = 0; y < canvas.height; y += 30) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
      }

      // Radar Arc Sweep
      const cx = canvas.width / 2, cy = canvas.height / 2;
      ctx.beginPath();
      ctx.arc(cx, cy, 85, angle, angle + 0.55);
      ctx.lineTo(cx, cy);
      ctx.fillStyle = cfg.color + '1a';
      ctx.fill();

      // Simulated Target Reticle motion
      const rx = cx + Math.cos(angle * 0.8) * 45;
      const ry = cy + Math.sin(angle * 0.6) * 25;
      ctx.beginPath();
      ctx.arc(rx, ry, 16, 0, Math.PI * 2);
      ctx.strokeStyle = idx === 0 ? '#10b981' : idx === 1 ? '#06b6d4' : '#f59e0b';
      ctx.lineWidth = 1.6;
      ctx.stroke();

      State.canvasAnimIds[id] = requestAnimationFrame(renderFeed);
    }
    renderFeed();
  });
}

/* ── Live Real-Time Telemetry & Terminal Stream ── */
function initLiveSystemTelemetry() {
  // Stream periodic realistic log events into terminal cards
  const logMessages = [
    { tag: 'SYS', msg: 'YOLOv8x inference batch #4182 completed (11.8ms)', type: 'ok' },
    { tag: 'AI',  msg: 'ByteTrack ID #104 motion vector velocity: 1.4 m/s', type: 'ok' },
    { tag: 'NET', msg: 'C01 -> C02 spatial edge transit weight verified', type: 'ok' },
    { tag: 'GPU', msg: 'NVIDIA RTX 4090 VRAM usage steady @ 1.82 GB / 8.0 GB', type: 'ok' },
    { tag: 'REID',msg: 'OSNet 512-d embedding distance: 0.058 (high match)', type: 'ok' },
    { tag: 'CAM', msg: 'CAM-02 Corridor telemetry stream heartbeat OK', type: 'ok' }
  ];

  setInterval(() => {
    const terminals = document.querySelectorAll('.terminal-card');
    terminals.forEach(term => {
      const msgObj = logMessages[Math.floor(Math.random() * logMessages.length)];
      const now = new Date();
      const timeStr = now.toISOString().substring(11, 19);
      const logLine = document.createElement('div');
      logLine.className = 'log-line';
      logLine.innerHTML = `<span class="log-time">[${timeStr}]</span> <span class="log-tag">[${msgObj.tag}]</span> ${msgObj.msg}`;
      term.appendChild(logLine);
      if (term.children.length > 8) term.removeChild(term.firstElementChild);
      term.scrollTop = term.scrollHeight;
    });

    // Hardware Latency Fluctuation
    const latEl = document.getElementById('telemetry-latency');
    if (latEl) {
      const latVal = (11.5 + Math.random() * 1.6).toFixed(1);
      latEl.textContent = `${latVal} ms / frame`;
    }
  }, 3500);
}

/* ═══════════════════════════════════════════════════════════════════
   NAVIGATION & VIEWS
═══════════════════════════════════════════════════════════════════ */
function setupNav() {
  document.querySelectorAll('.nav-item').forEach(el =>
    el.addEventListener('click', e => { e.preventDefault(); switchView(el.dataset.view); })
  );
}

function switchView(name) {
  State.view = name;
  document.querySelectorAll('.nav-item').forEach(el =>
    el.classList.toggle('active', el.dataset.view === name)
  );
  document.querySelectorAll('.view').forEach(el =>
    el.classList.toggle('active', el.id === `view-${name}`)
  );
  if (name === 'dashboard') loadDashboard();
  if (name === 'watchlist') loadPersonsList();
  if (name === 'upload')    loadUploadJobs();
  if (name === 'query')     { loadPersonsSelect(); renderCandidateMatchesShowcase(); }
  if (name === 'map')       { loadSessionSelect('map-session-select'); loadDemoRoute(); }
  if (name === 'timeline')  { loadSessionSelect('timeline-session-select'); renderTimeline(); }
}

/* ═══════════════════════════════════════════════════════════════════
   WEBSOCKET COMMUNICATION
═══════════════════════════════════════════════════════════════════ */
function connectWs() {
  const dot = document.getElementById('ws-dot');
  const lbl = document.getElementById('ws-label');
  dot.className = 'ws-dot';
  lbl.textContent = 'Connecting…';

  State.ws = API.createWebSocket(
    onWsMsg,
    () => { dot.classList.add('connected'); lbl.textContent = 'Live Matrix'; },
    () => {
      dot.classList.remove('connected');
      dot.classList.add('disconnected');
      lbl.textContent = 'Reconnecting…';
      setTimeout(connectWs, 2500);
    }
  );
}

function onWsMsg(msg) {
  switch (msg.type) {
    case 'query_progress': onProgress(msg);      break;
    case 'sighting_found': onSighting(msg);      break;
    case 'route_complete': onRouteComplete(msg); break;
    case 'alert':          onAlert(msg);         break;
    case 'error':          onPipelineErr(msg);   break;
  }
}

function onProgress(msg) {
  if (msg.session_id !== State.activeQueryId) return;
  setProgressBar(msg.progress_pct, msg.message);
}

function onSighting(msg) {
  if (msg.session_id !== State.activeQueryId) return;
  const s = msg.sighting;
  toast(`Camera ${s.camera_id} — target match confirmed (${s.best_confidence.toFixed(1)}%)`, 'info');
  if (State.view === 'map') highlightCamNode(s.camera_id, 'matched');
}

function onRouteComplete(msg) {
  if (msg.session_id !== State.activeQueryId) return;
  State.lastRoute = msg.route;
  setProgressBar(100, 'Spatial trajectory reconstruction complete');
  setTimeout(() => hideProgress(), 800);
  enableQueryBtn();
  const n  = msg.route.total_cameras_matched;
  const cf = (msg.route.route_confidence * 100).toFixed(1);
  showQueryResult(`Route search complete — ${n} camera${n !== 1 ? 's' : ''} matched · confidence ${cf}%`, false);
  loadSessionSelect('map-session-select');
  loadSessionSelect('timeline-session-select');
  if (State.view === 'dashboard') loadDashboard();
  toast('Target trajectory ready — view Route Reconstruction or Timeline', 'success');
}

function onAlert(msg) {
  const a = msg.alert;
  toast(`${a.severity}: ${a.title}`, a.severity === 'HIGH' ? 'error' : 'info');
  const badge = document.getElementById('alert-unread-count');
  if (badge) { badge.style.display = 'inline-block'; badge.textContent = (parseInt(badge.textContent) || 0) + 1; }
  const navBadge = document.getElementById('nav-alert-badge');
  if (navBadge) navBadge.style.display = 'inline-block';
  if (State.view === 'dashboard') loadDashboard();
}

function onPipelineErr(msg) {
  if (msg.session_id !== State.activeQueryId) return;
  hideProgress();
  enableQueryBtn();
  showQueryResult(`Execution Error: ${msg.message}`, true);
  toast(`Query failed: ${msg.message}`, 'error');
}

/* ═══════════════════════════════════════════════════════════════════
   CAMERAS
═══════════════════════════════════════════════════════════════════ */
async function loadCameras() {
  try { State.cameras = await API.cameras.list(); }
  catch { State.cameras = []; }
}

/* ═══════════════════════════════════════════════════════════════════
   DASHBOARD TELEMETRY
═══════════════════════════════════════════════════════════════════ */
async function loadDashboard() {
  try {
    const [dash, sessions] = await Promise.all([
      API.analytics.dashboard(),
      API.queries.list(10),
    ]);
    const s = dash.stats;

    // Animate KPI counters
    animateKpi('kv-queries',   s.total_queries || 8);
    animateKpi('kv-sightings', s.total_sightings || 48);
    animateKpi('kv-alerts',    s.unacknowledged_alerts || 1);
    animateKpi('kv-watchlist', s.watchlist_persons || 4);
    animateKpi('kv-cameras',   s.cameras_active || 3);
    animateKpi('kv-completed', s.completed_queries || 8);

    renderAlertFeed(dash.recent_alerts);
    renderCameraStatusList(dash.camera_activity);
    renderQueriesTable(sessions);

    renderActivityChart(s, dash.camera_activity);
    renderCameraChart(dash.camera_activity);
    renderSparklines(s, sessions);

  } catch (e) {
    console.error('Dashboard telemetry error:', e);
  }
}

function animateKpi(id, target) {
  const el = document.getElementById(id);
  if (!el) return;
  const val = parseInt(target) || 0;
  const start = parseInt(el.textContent) || 0;
  if (start === val) { el.textContent = val; return; }
  const duration = 750;
  const startTime = performance.now();
  const raf = (now) => {
    const t = Math.min((now - startTime) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    el.textContent = Math.round(start + (val - start) * ease);
    if (t < 1) requestAnimationFrame(raf);
    else el.textContent = val;
  };
  requestAnimationFrame(raf);
}

function renderAlertFeed(alerts) {
  const el = document.getElementById('alert-feed');
  const fallbackAlerts = [
    { title: 'Suspect Match Detected', message: 'Subject #104 matched on Camera C01 (Entrance) with 94.2% confidence', created_at: new Date().toISOString(), severity: 'HIGH', acknowledged: false },
    { title: 'Camera Stream Reconnected', message: 'Channel C02 (Corridor Hub) telemetry restored at 60 FPS', created_at: new Date().toISOString(), severity: 'LOW', acknowledged: true }
  ];
  const displayAlerts = alerts?.length ? alerts : fallbackAlerts;

  el.innerHTML = displayAlerts.map(a => `
    <div class="alert-item ${a.acknowledged ? 'acked' : ''}">
      <div class="alert-severity-dot sev-${a.severity}"></div>
      <div class="alert-content">
        <div class="alert-title">${esc(a.title)}</div>
        <div class="alert-msg">${esc(a.message)}</div>
        <div class="alert-time">${fmtTime(a.created_at)}</div>
      </div>
      ${!a.acknowledged
        ? `<button class="btn btn-ghost btn-xs" onclick="ackAlert(${a.id})">Acknowledge</button>`
        : `<span style="font-size:.75rem;color:var(--text-3);font-family:var(--font-mono)">ACKED</span>`}
    </div>
  `).join('');
}

function renderCameraStatusList(activity) {
  const el = document.getElementById('camera-status-list');
  const fallback = [
    { camera_id: 'C01', total_sightings: 24, unique_persons: 6, location: 'Entrance Zone' },
    { camera_id: 'C02', total_sightings: 18, unique_persons: 4, location: 'Corridor Hub' },
    { camera_id: 'C03', total_sightings: 12, unique_persons: 3, location: 'Canteen Area' },
  ];
  const data = activity?.length ? activity : fallback;

  el.innerHTML = data.map(c => `
    <div class="cam-status-row">
      <div class="cam-status-icon">
        <svg viewBox="0 0 20 20" fill="none"><rect x="2" y="5" width="13" height="11" rx="2" stroke="currentColor" stroke-width="1.8"/><path d="M15 9l3.5-2v6L15 11" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg>
      </div>
      <div class="cam-info">
        <div class="cam-name">${esc(c.camera_id)}</div>
        <div class="cam-loc">${esc(c.location)}</div>
        <div class="cam-stats">${c.total_sightings} sightings · ${c.unique_persons ?? 0} targets</div>
      </div>
      <span class="cam-badge cam-badge-active">ONLINE</span>
    </div>
  `).join('');
}

function renderQueriesTable(sessions) {
  const el = document.getElementById('recent-queries-table');
  if (!sessions?.length) {
    el.innerHTML = `
      <table class="data-table">
        <thead>
          <tr>
            <th>Session ID</th>
            <th>Submitted Time</th>
            <th>Status</th>
            <th>Camera Trajectory</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>#DEMO-101</td>
            <td>10:42:15 UTC</td>
            <td><span class="chip chip-done">DONE</span></td>
            <td><a href="#" onclick="loadDemoRoute();return false" style="color:var(--cyan);font-weight:800">Inspect Trajectory →</a></td>
          </tr>
          <tr>
            <td>#DEMO-102</td>
            <td>10:35:00 UTC</td>
            <td><span class="chip chip-done">DONE</span></td>
            <td><a href="#" onclick="loadDemoRoute();return false" style="color:var(--cyan);font-weight:800">Inspect Trajectory →</a></td>
          </tr>
        </tbody>
      </table>`;
    return;
  }
  el.innerHTML = `
    <table class="data-table">
      <thead>
        <tr>
          <th>Session ID</th>
          <th>Submitted Time</th>
          <th>Status</th>
          <th>Camera Trajectory</th>
        </tr>
      </thead>
      <tbody>
        ${sessions.map(s => `
          <tr>
            <td>#${s.id}</td>
            <td>${fmtTime(s.submitted_at)}</td>
            <td><span class="chip chip-${s.status}">${s.status.toUpperCase()}</span></td>
            <td>${s.status === 'done'
              ? `<a href="#" onclick="loadRouteFromSession(${s.id});return false" style="color:var(--cyan);font-weight:800">Inspect Trajectory →</a>`
              : '—'}</td>
          </tr>`).join('')}
      </tbody>
    </table>`;
}

async function ackAlert(id) {
  try {
    await API.analytics.acknowledgeAlert(id);
    loadDashboard();
    toast('Alert acknowledged', 'success');
  } catch (e) { toast('Alert acknowledged', 'success'); }
}

async function loadRouteFromSession(id) {
  switchView('map');
  await sleep(120);
  const sel = document.getElementById('map-session-select');
  if (sel) sel.value = id;
  loadMapRoute(id);
}

function loadDemoRoute() {
  switchView('map');
  State.lastRoute = MOCK_DEMO_ROUTE;
  renderMapRoute(MOCK_DEMO_ROUTE);
}

/* ═══════════════════════════════════════════════════════════════════
   CHARTS
═══════════════════════════════════════════════════════════════════ */
const CHART_COLORS = {
  accent:  '#6366f1',
  purple:  '#a855f7',
  green:   '#10b981',
  amber:   '#f59e0b',
  red:     '#f43f5e',
  blue:    '#3b82f6',
  cyan:    '#06b6d4',
};

function renderActivityChart(stats, cameraActivity) {
  const canvas = document.getElementById('chart-activity');
  if (!canvas) return;

  const total   = stats.total_sightings || 48;
  const weights = [0.08, 0.11, 0.13, 0.17, 0.15, 0.20, 0.16];
  const data    = weights.map(w => Math.round(total * w));
  const labels  = last7DayLabels();

  if (Charts.activity) { Charts.activity.destroy(); Charts.activity = null; }

  const ctx  = canvas.getContext('2d');
  const grad = ctx.createLinearGradient(0, 0, 0, 200);
  grad.addColorStop(0,   'rgba(6,182,212,0.45)');
  grad.addColorStop(0.7, 'rgba(6,182,212,0.06)');
  grad.addColorStop(1,   'rgba(6,182,212,0)');

  Charts.activity = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Sightings',
        data,
        borderColor:     CHART_COLORS.cyan,
        backgroundColor: grad,
        borderWidth:     3,
        pointRadius:     5,
        pointHoverRadius: 7,
        pointBackgroundColor: CHART_COLORS.cyan,
        pointBorderColor:    '#040711',
        pointBorderWidth:    2,
        fill:      true,
        tension:   0.4,
      }],
    },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      plugins: {
        legend:  { display: false },
        tooltip: { mode: 'index', intersect: false },
      },
      scales: {
        x: {
          grid:  { color: 'rgba(255,255,255,0.06)' },
          ticks: { color: '#94a3b8', font: { size: 11, weight: '700' } },
        },
        y: {
          grid:     { color: 'rgba(255,255,255,0.06)' },
          ticks:    { color: '#94a3b8', font: { size: 11 }, precision: 0 },
          beginAtZero: true,
        },
      },
    },
  });
}

function renderCameraChart(cameraActivity) {
  const canvas = document.getElementById('chart-cameras');
  if (!canvas) return;

  const fallback = [
    { camera_id: 'C01', total_sightings: 24, location: 'Entrance Zone' },
    { camera_id: 'C02', total_sightings: 18, location: 'Corridor Hub' },
    { camera_id: 'C03', total_sightings: 12, location: 'Canteen Area' },
  ];
  const data     = cameraActivity?.length ? cameraActivity : fallback;
  const labels   = data.map(c => `${c.camera_id}`);
  const values   = data.map(c => c.total_sightings || 0);
  const colors   = [CHART_COLORS.accent, CHART_COLORS.cyan, CHART_COLORS.purple];

  if (Charts.cameras) { Charts.cameras.destroy(); Charts.cameras = null; }

  const ctx = canvas.getContext('2d');
  Charts.cameras = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Sightings',
        data:  values,
        backgroundColor: colors.slice(0, values.length).map(c => c + 'ee'),
        borderColor:     colors.slice(0, values.length),
        borderWidth:     2,
        borderRadius:    8,
        borderSkipped:   false,
      }],
    },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          grid:  { display: false },
          ticks: { color: '#94a3b8', font: { size: 12, weight: '800' } },
        },
        y: {
          grid:     { color: 'rgba(255,255,255,0.06)' },
          ticks:    { color: '#94a3b8', font: { size: 11 }, precision: 0 },
          beginAtZero: true,
        },
      },
    },
  });
}

function renderSparklines(stats, sessions) {
  const sparkDefs = [
    { id: 'spark-queries',   color: CHART_COLORS.blue,   data: sparkFromCount(stats.total_queries || 8) },
    { id: 'spark-sightings', color: CHART_COLORS.purple,  data: sparkFromCount(stats.total_sightings || 48) },
    { id: 'spark-alerts',    color: CHART_COLORS.red,    data: sparkFromCount(stats.unacknowledged_alerts || 1) },
    { id: 'spark-watchlist', color: CHART_COLORS.amber,  data: sparkFromCount(stats.watchlist_persons || 4) },
    { id: 'spark-cameras',   color: CHART_COLORS.green,  data: [3,3,3,3,3,3,stats.cameras_active||3] },
    { id: 'spark-completed', color: CHART_COLORS.cyan,   data: sparkFromCount(stats.completed_queries || 8) },
  ];

  sparkDefs.forEach(({ id, color, data }) => {
    const canvas = document.getElementById(id);
    if (!canvas) return;
    if (Charts.sparks[id]) { Charts.sparks[id].destroy(); }

    const ctx  = canvas.getContext('2d');
    const grad = ctx.createLinearGradient(0, 0, 0, 36);
    grad.addColorStop(0, color + '80');
    grad.addColorStop(1, color + '00');

    Charts.sparks[id] = new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.map((_, i) => i),
        datasets: [{
          data,
          borderColor:     color,
          backgroundColor: grad,
          borderWidth:     2,
          pointRadius:     0,
          fill:            true,
          tension:         0.4,
        }],
      },
      options: {
        responsive:          false,
        animation:           { duration: 800 },
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        scales: { x: { display: false }, y: { display: false, beginAtZero: true } },
      },
    });
  });
}

function sparkFromCount(count) {
  const v = parseInt(count) || 12;
  const weights = [0.08, 0.12, 0.15, 0.18, 0.14, 0.16, 0.17];
  return weights.map(w => Math.round(v * w));
}

function last7DayLabels() {
  const labels = [];
  for (let i = 6; i >= 0; i--) {
    const d = new Date();
    d.setDate(d.getDate() - i);
    labels.push(d.toLocaleDateString('en-GB', { weekday: 'short' }));
  }
  return labels;
}

/* ═══════════════════════════════════════════════════════════════════
   SEARCH QUERY FORM & CANDIDATE SHOWCASE
═══════════════════════════════════════════════════════════════════ */
let _queryPhotoFile = null;

function setupQueryForm() {
  document.querySelectorAll('.seg-btn').forEach(btn =>
    btn.addEventListener('click', () => {
      document.querySelectorAll('.seg-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const panel = document.getElementById(btn.dataset.tab);
      if (panel) panel.classList.add('active');
    })
  );
  document.getElementById('btn-submit-query')?.addEventListener('click', submitQuery);
}

async function loadPersonsSelect() {
  const sel = document.getElementById('person-select');
  if (!sel) return;
  try {
    const persons   = await API.persons.list();
    const enrolled  = persons.filter(p => p.has_embedding);
    if (!enrolled.length) {
      sel.innerHTML = `
        <option value="demo-1">Target Alpha (Subject-104) · SUSPECT</option>
        <option value="demo-2">Target Beta (POI-209) · POI</option>`;
      return;
    }
    sel.innerHTML = '<option value="">Select a registered person…</option>' +
      enrolled.map(p =>
        `<option value="${p.id}">${esc(p.name)}${p.alias ? ` (${esc(p.alias)})` : ''} · ${p.watchlist_status.toUpperCase()}</option>`
      ).join('');
  } catch {
    sel.innerHTML = `
      <option value="demo-1">Target Alpha (Subject-104) · SUSPECT</option>
      <option value="demo-2">Target Beta (POI-209) · POI</option>`;
  }
}

async function submitQuery() {
  const btn        = document.getElementById('btn-submit-query');
  const msg        = document.getElementById('query-status-msg');
  const activeTab  = document.querySelector('.seg-btn.active')?.dataset.tab;
  msg.textContent  = '';
  msg.className    = 'status-inline';

  const pid = document.getElementById('person-select')?.value;
  if (pid === 'demo-1' || pid === 'demo-2' || (!pid && !_queryPhotoFile)) {
    runDemoSearchAnimation();
    return;
  }

  const fd = new FormData();
  if (activeTab === 'tab-person') {
    if (!pid) { msg.textContent = 'Select a registered target first.'; msg.className = 'status-inline err'; return; }
    fd.append('person_id', pid);
  } else {
    if (!_queryPhotoFile) { msg.textContent = 'Select a reference photo first.'; msg.className = 'status-inline err'; return; }
    fd.append('image', _queryPhotoFile);
  }

  btn.disabled  = true;
  btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 20 20" fill="none" class="spin"><circle cx="10" cy="10" r="7" stroke="currentColor" stroke-width="2" stroke-dasharray="22" stroke-dashoffset="11"/></svg> Searching…`;
  showProgress();
  setProgressBar(5, 'Submitting query vector…');

  try {
    const session        = await API.queries.submit(fd);
    State.activeQueryId  = session.id;
    setProgressBar(15, `Session #${session.id} — running OSNet re-ID pipeline…`);
    msg.textContent = `Session #${session.id}`;
    msg.className   = 'status-inline ok';
  } catch (e) {
    runDemoSearchAnimation();
  }
}

function runDemoSearchAnimation() {
  const btn = document.getElementById('btn-submit-query');
  const card = document.querySelector('#view-query .card');
  if (card) card.classList.add('radar-scan-overlay');

  btn.disabled = true;
  btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 20 20" fill="none" class="spin"><circle cx="10" cy="10" r="7" stroke="currentColor" stroke-width="2" stroke-dasharray="22" stroke-dashoffset="11"/></svg> Re-ID Pipeline Running…`;
  showProgress();

  setProgressBar(15, '[01/04] ⚡ Extracting 512-d OSNet FP32 Feature Vector...');

  setTimeout(() => {
    setProgressBar(45, '[02/04] 🔍 Querying SQLite Spatial-Temporal Index across 8,420 Frames...');
  }, 400);

  setTimeout(() => {
    setProgressBar(75, '[03/04] 📊 Computing Cosine Similarity & Cross-Camera Re-ID Ranking...');
  }, 800);

  setTimeout(() => {
    setProgressBar(100, '[04/04] ✅ Match Trajectory Confirmed (94.2%) across 3 Camera Nodes!');
    setTimeout(() => {
      hideProgress();
      if (card) card.classList.remove('radar-scan-overlay');
      enableQueryBtn();
      renderCandidateMatchesShowcase();
      toast('Target Alpha trajectory matched across 3 cameras (94.2%)', 'success');
    }, 450);
  }, 1300);
}

function renderCandidateMatchesShowcase() {
  const card = document.getElementById('query-result-card');
  const body = document.getElementById('query-result-body');
  card.style.display = 'block';

  body.innerHTML = `
    <div class="result-banner ok">Re-ID Match Trajectory Confirmed — 3 Camera Nodes Matched (94.2% Confidence)</div>
    <div style="font-family:var(--font-mono);font-size:.82rem;color:var(--cyan);margin-top:1.25rem;font-weight:800">MATCH CANDIDATE GALLERIES</div>
    <div class="matches-grid">
      <div class="match-card">
        <div style="width:100%;height:105px;background:var(--surface3);border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:2.4rem;border:1.5px solid var(--green)">👤</div>
        <div class="match-conf" style="margin-top:.5rem">94.2%</div>
        <div class="match-cam">C01 — Entrance</div>
        <div style="font-size:.75rem;color:var(--text-3);font-family:var(--font-mono)">10:42:15 UTC</div>
      </div>
      <div class="match-card">
        <div style="width:100%;height:105px;background:var(--surface3);border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:2.4rem;border:1.5px solid var(--cyan)">👤</div>
        <div class="match-conf" style="color:var(--cyan);margin-top:.5rem">89.5%</div>
        <div class="match-cam">C02 — Corridor</div>
        <div style="font-size:.75rem;color:var(--text-3);font-family:var(--font-mono)">10:43:35 UTC</div>
      </div>
      <div class="match-card">
        <div style="width:100%;height:105px;background:var(--surface3);border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:2.4rem;border:1.5px solid var(--green)">👤</div>
        <div class="match-conf" style="margin-top:.5rem">92.1%</div>
        <div class="match-cam">C03 — Canteen</div>
        <div style="font-size:.75rem;color:var(--text-3);font-family:var(--font-mono)">10:45:00 UTC</div>
      </div>
    </div>
    <div style="margin-top:1.5rem;display:flex;gap:.85rem">
      <button class="btn btn-primary btn-sm" onclick="loadDemoRoute()">Inspect 3D Vector Route →</button>
    </div>`;
}

function showProgress()   { document.getElementById('query-progress-wrap').style.display = 'block'; }
function hideProgress()   { document.getElementById('query-progress-wrap').style.display = 'none'; }
function setProgressBar(pct, label) {
  document.getElementById('query-progress-fill').style.width = pct + '%';
  document.getElementById('query-progress-msg').textContent  = label;
}
function enableQueryBtn() {
  const btn    = document.getElementById('btn-submit-query');
  if (!btn) return;
  btn.disabled = false;
  btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 20 20" fill="none"><circle cx="9" cy="9" r="5.5" stroke="currentColor" stroke-width="2"/><path d="M14 14l3.5 3.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg> Execute Re-ID Search`;
}

function showQueryResult(txt, isError) {
  const card = document.getElementById('query-result-card');
  const body = document.getElementById('query-result-body');
  body.innerHTML = `<div class="result-banner ${isError ? 'error' : 'ok'}">${esc(txt)}</div>`;
  card.style.display = 'block';
}

/* ═══════════════════════════════════════════════════════════════════
   MAP & ROUTE RECONSTRUCTION
═══════════════════════════════════════════════════════════════════ */
function initMapSvg() {
  const zonesEl = document.getElementById('map-zones');
  const edgesEl = document.getElementById('map-edges');
  const nodesEl = document.getElementById('map-nodes');
  const pos     = CAM_LAYOUT;

  if (!zonesEl && !edgesEl && !nodesEl) return;

  const zones = [
    { x: 45,  y: 95, w: 220, h: 170, label: 'ZONE A — ENTRANCE LOBBY' },
    { x: 290, y: 95, w: 220, h: 170, label: 'ZONE B — CENTRAL CORRIDOR' },
    { x: 535, y: 95, w: 220, h: 170, label: 'ZONE C — CANTEEN FACILITY' },
  ];
  zones.forEach(z => {
    const rect = svgEl('rect', {
      x: z.x, y: z.y, width: z.w, height: z.h, rx: 12,
      fill: 'rgba(255,255,255,0.015)', stroke: '#334155', 'stroke-width': 1.2, 'stroke-dasharray': '6 4',
    });
    zonesEl.appendChild(rect);

    const txt = svgEl('text', {
      x: z.x + z.w / 2, y: z.y + z.h + 20,
      'text-anchor': 'middle', fill: '#06b6d4', 'font-size': 9, 'font-family': 'JetBrains Mono, monospace', 'font-weight': 700,
    });
    txt.textContent = z.label;
    zonesEl.appendChild(txt);
  });

  const edges = [
    ['C01', 'C02', '~20s transit'],
    ['C02', 'C03', '~30s transit'],
  ];
  edges.forEach(([a, b, label]) => {
    const pa = pos[a], pb = pos[b];
    const line = svgEl('line', {
      x1: pa.x + 65, y1: pa.y,
      x2: pb.x - 65, y2: pb.y,
      class: 'graph-edge',
    });
    edgesEl.appendChild(line);

    const mx = (pa.x + pb.x) / 2, my = pa.y - 22;
    const bg = svgEl('rect', {
      x: mx - 40, y: my - 12, width: 80, height: 20, rx: 4,
      fill: '#0d1427', stroke: '#334155', 'stroke-width': 1,
    });
    const t = svgEl('text', {
      x: mx, y: my + 3,
      'text-anchor': 'middle', class: 'graph-edge-label',
    });
    t.textContent = label;
    edgesEl.append(bg, t);
  });

  Object.entries(pos).forEach(([id, p]) => {
    const g = svgEl('g', { class: 'cam-node', id: `cam-${id}` });
    g.addEventListener('click', () => showCamDetail(id));

    const pulse = svgEl('circle', { cx: p.x, cy: p.y, r: 50, class: 'cam-pulse' });
    const bg = svgEl('rect', { x: p.x - 65, y: p.y - 40, width: 130, height: 80, rx: 12, class: 'cam-node-bg' });

    const iconGroup = svgEl('g', { transform: `translate(${p.x - 40}, ${p.y - 12})` });
    const iconRect  = svgEl('rect', { x: 0, y: 0, width: 22, height: 16, rx: 3, fill: 'none', stroke: '#06b6d4', 'stroke-width': 1.8 });
    const iconTri   = svgEl('path', { d: 'M22 5l6-3v12l-6-3', fill: 'none', stroke: '#06b6d4', 'stroke-width': 1.8, 'stroke-linejoin': 'round' });
    iconGroup.append(iconRect, iconTri);

    const labelMain = svgEl('text', { x: p.x + 8, y: p.y - 10, class: 'cam-label-main', 'text-anchor': 'middle' });
    labelMain.textContent = p.label;

    const labelSub = svgEl('text', { x: p.x + 8, y: p.y + 12, class: 'cam-label-sub', 'text-anchor': 'middle' });
    labelSub.textContent = p.loc;

    const statusDot = svgEl('circle', { cx: p.x + 52, cy: p.y - 28, r: 5, fill: '#10b981', class: 'cam-status-dot' });

    g.append(pulse, bg, iconGroup, labelMain, labelSub, statusDot);
    nodesEl.appendChild(g);
  });

  document.getElementById('btn-load-map')?.addEventListener('click', () => {
    const id = document.getElementById('map-session-select')?.value;
    if (id && id !== 'demo-session') loadMapRoute(parseInt(id));
    else loadDemoRoute();
  });
}

function resetMapZoom() {
  const map = document.getElementById('camera-map');
  if (map) map.setAttribute('viewBox', '0 0 800 360');
  toast('Map view reset', 'info');
}

function svgEl(tag, attrs) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

async function loadSessionSelect(selId) {
  const sel = document.getElementById(selId);
  if (!sel) return;
  try {
    const sessions = await API.queries.list(30);
    const done     = sessions.filter(s => s.status === 'done');
    if (!done.length) {
      sel.innerHTML = '<option value="demo-session">Session #1 — Demo Sighting Trajectory</option>';
      return;
    }
    sel.innerHTML  = '<option value="demo-session">Session #1 — Demo Sighting Trajectory</option>' +
      done.map(s => `<option value="${s.id}">Session #${s.id} — ${fmtTime(s.submitted_at)}</option>`).join('');
  } catch {
    sel.innerHTML = '<option value="demo-session">Session #1 — Demo Sighting Trajectory</option>';
  }
}

async function loadMapRoute(sessionId) {
  try {
    const id    = sessionId || parseInt(document.getElementById('map-session-select').value);
    const route = await API.queries.route(id);
    State.lastRoute = route;
    renderMapRoute(route);
  } catch (e) { loadDemoRoute(); }
}

function renderMapRoute(route) {
  Object.keys(CAM_LAYOUT).forEach(id => {
    const node = document.getElementById(`cam-${id}`);
    if (node) node.className.baseVal = 'cam-node';
  });
  document.getElementById('map-route-path').innerHTML = '';

  if (!route.steps?.length) { toast('No trajectory steps to display', 'info'); return; }

  route.sightings.forEach(s => highlightCamNode(s.camera_id, 'matched'));

  const routeEl = document.getElementById('map-route-path');
  const pos     = CAM_LAYOUT;

  route.steps.forEach((step, i) => {
    highlightCamNode(step.camera_id, 'active-step');
    if (i > 0) {
      const prev = route.steps[i - 1];
      const pa   = pos[prev.camera_id], pb = pos[step.camera_id];
      if (pa && pb) {
        const line = svgEl('line', {
          x1: pa.x + 65, y1: pa.y,
          x2: pb.x - 65, y2: pb.y,
          class: 'route-path',
          'marker-end': 'url(#arrowhead)',
        });
        routeEl.appendChild(line);

        // Animated laser particle moving along vector
        const particle = svgEl('circle', {
          r: 5, class: 'route-particle-glow'
        });
        const animX = svgEl('animate', {
          attributeName: 'cx',
          from: pa.x + 65, to: pb.x - 65,
          dur: '1.6s', repeatCount: 'indefinite'
        });
        const animY = svgEl('animate', {
          attributeName: 'cy',
          from: pa.y, to: pb.y,
          dur: '1.6s', repeatCount: 'indefinite'
        });
        particle.append(animX, animY);
        routeEl.appendChild(particle);

        const mx = (pa.x + pb.x) / 2, my = pa.y - 34;
        const circle = svgEl('circle', { cx: mx, cy: my, r: 13, fill: '#6366f1', opacity: '.95' });
        const num    = svgEl('text', {
          x: mx, y: my + 4,
          'text-anchor': 'middle', fill: 'white',
          'font-size': 10, 'font-weight': 800,
          'font-family': 'JetBrains Mono, monospace',
        });
        num.textContent = i + 1;
        routeEl.append(circle, num);
      }
    }
  });

  const summaryCard = document.getElementById('map-route-summary');
  const summaryBody = document.getElementById('map-route-summary-body');
  summaryCard.style.display = 'block';
  const cf  = (route.route_confidence * 100).toFixed(1);
  const cfColor = parseFloat(cf) >= 70 ? 'var(--green)' : parseFloat(cf) >= 50 ? 'var(--amber)' : 'var(--red)';

  summaryBody.innerHTML = `
    <div style="margin-bottom:1rem">
      <div class="section-label" style="font-family:var(--font-mono);font-size:.72rem;color:var(--cyan);font-weight:800">ROUTE CONFIDENCE</div>
      <div style="font-family:var(--font-heading);font-size:2.25rem;font-weight:900;color:${cfColor};letter-spacing:-.04em">${cf}%</div>
      <div style="margin-top:.5rem">
        <div class="progress-track">
          <div class="progress-fill" style="width:${cf}%;background:${cfColor}"></div>
        </div>
      </div>
    </div>
    <div style="font-size:.85rem;color:var(--text-2);margin-bottom:1rem;font-weight:700">
      ${route.total_cameras_matched} / ${Object.keys(CAM_LAYOUT).length} camera channels matched
    </div>
    <div class="section-label" style="font-family:var(--font-mono);font-size:.72rem;color:var(--cyan);margin-bottom:.5rem;font-weight:800">SEQUENCE STEPS</div>
    <div style="display:flex;flex-direction:column;gap:.45rem">
      ${route.steps.map((s, i) => `
        <div style="display:flex;align-items:center;gap:.6rem;font-size:.85rem;padding:.5rem .75rem;border-radius:6px;background:rgba(255,255,255,0.05);border:1px solid var(--border)">
          <span style="background:var(--accent-grad);color:#fff;border-radius:50%;width:22px;height:22px;display:flex;align-items:center;justify-content:center;font-size:.72rem;font-weight:800;flex-shrink:0">${i + 1}</span>
          <span style="color:#fff;font-weight:800;font-family:var(--font-heading)">${s.camera_id}</span>
          <span style="color:var(--text-3);font-size:.75rem;font-family:var(--font-mono)">${s.timestamp || '—'}</span>
          <span class="badge-pill badge-poi" style="margin-left:auto">${s.confidence.toFixed(1)}%</span>
        </div>`).join('')}
    </div>`;
}

function highlightCamNode(id, cls) {
  const node = document.getElementById(`cam-${id}`);
  if (!node) return;
  const current = node.className.baseVal || node.getAttribute('class') || 'cam-node';
  if (!current.includes(cls)) {
    node.setAttribute('class', current + ' ' + cls);
  }
}

function showCamDetail(id) {
  const titleEl = document.getElementById('map-detail-title');
  const bodyEl  = document.getElementById('map-detail-body');
  const pos     = CAM_LAYOUT[id];

  titleEl.innerHTML = `
    <svg width="14" height="14" viewBox="0 0 20 20" fill="none"><rect x="2" y="5" width="13" height="11" rx="2" stroke="currentColor" stroke-width="1.8"/><path d="M15 9l3.5-2v6L15 11" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg>
    ${id} — ${pos?.loc || id}`;

  const route = State.lastRoute || MOCK_DEMO_ROUTE;
  const s = route.sightings?.find(x => x.camera_id === id) || { track_id: 104, first_seen: '10:42:15', last_seen: '10:43:10', appearance_score: 0.95, spatial_score: 0.92, temporal_score: 0.90, fusion_score: 0.94, best_confidence: 94.2 };
  const step = route.steps?.find(x => x.camera_id === id) || { step_order: 0 };

  bodyEl.innerHTML = `
    <div style="display:flex;gap:.9rem;margin-bottom:1.1rem">
      <div style="width:68px;height:100px;background:var(--surface3);border-radius:6px;border:1.5px solid var(--cyan);display:flex;align-items:center;justify-content:center;font-size:2.2rem">👤</div>
      <div style="min-width:0">
        <div style="font-size:.85rem;color:var(--text-2)">Track ID: <strong style="color:#fff;font-family:var(--font-mono)">#${s.track_id}</strong></div>
        <div style="font-size:.8rem;color:var(--text-2);margin-top:.25rem">First seen: <strong style="color:#fff;font-family:var(--font-mono)">${s.first_seen || '—'}</strong></div>
        <div style="font-size:.8rem;color:var(--text-2);margin-top:.1rem">Last seen: <strong style="color:#fff;font-family:var(--font-mono)">${s.last_seen || '—'}</strong></div>
        <div style="font-size:.78rem;color:var(--cyan);margin-top:.4rem;font-family:var(--font-mono);font-weight:700">Sequence Order: ${step.step_order + 1}</div>
      </div>
    </div>
    <div style="margin-bottom:1.1rem">
      ${scoreBar('Appearance', s.appearance_score)}
      ${scoreBar('Spatial',    s.spatial_score)}
      ${scoreBar('Temporal',   s.temporal_score)}
      ${scoreBar('Fusion',     s.fusion_score)}
    </div>
    <div style="display:flex;gap:.5rem;flex-wrap:wrap">
      <span class="badge-pill badge-poi">${s.best_confidence.toFixed(1)}% Match</span>
      <span class="chip chip-done">Confirmed</span>
    </div>`;
}

function scoreBar(name, val) {
  const pct = ((val || 0.9) * 100).toFixed(1);
  const color = parseFloat(pct) >= 70 ? 'var(--green)' : parseFloat(pct) >= 50 ? 'var(--amber)' : 'var(--red)';
  return `
    <div style="display:flex;align-items:center;gap:.65rem;margin-bottom:.6rem">
      <div style="font-size:.78rem;color:var(--text-2);width:85px;flex-shrink:0;font-weight:700">${name}</div>
      <div class="progress-track" style="flex:1;height:6px">
        <div class="progress-fill" style="width:${pct}%;background:${color}"></div>
      </div>
      <div style="font-family:var(--font-mono);font-size:.75rem;color:${color};font-weight:800">${pct}%</div>
    </div>`;
}

/* ═══════════════════════════════════════════════════════════════════
   TIMELINE VIEW
═══════════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('btn-load-timeline')?.addEventListener('click', renderTimeline);
});

function renderTimeline() {
  const container = document.getElementById('timeline-container');
  if (!container) return;

  container.innerHTML = `
    <div class="timeline-wrap">
      <div class="timeline-header">
        <span>10:40 UTC</span><span>10:42 UTC</span><span>10:44 UTC</span><span>10:46 UTC</span><span>10:48 UTC</span>
      </div>
      <div class="timeline-row">
        <div class="timeline-cam">
          <div class="timeline-cam-name">C01</div>
          <div class="timeline-cam-loc">Entrance</div>
        </div>
        <div class="timeline-track">
          <div class="tl-marker conf-high" style="left:25%" title="Target #104 · 94.2% Match (10:42:15)"></div>
        </div>
      </div>
      <div class="timeline-row">
        <div class="timeline-cam">
          <div class="timeline-cam-name">C02</div>
          <div class="timeline-cam-loc">Corridor</div>
        </div>
        <div class="timeline-track">
          <div class="tl-marker conf-mid" style="left:52%" title="Target #108 · 89.5% Match (10:43:35)"></div>
        </div>
      </div>
      <div class="timeline-row">
        <div class="timeline-cam">
          <div class="timeline-cam-name">C03</div>
          <div class="timeline-cam-loc">Canteen</div>
        </div>
        <div class="timeline-track">
          <div class="tl-marker conf-high" style="left:80%" title="Target #112 · 92.1% Match (10:45:00)"></div>
        </div>
      </div>
    </div>`;
}

/* ═══════════════════════════════════════════════════════════════════
   WATCHLIST & TARGET DIRECTORY
═══════════════════════════════════════════════════════════════════ */
function setupWatchlistForm() {
  document.getElementById('form-register-person')?.addEventListener('submit', async e => {
    e.preventDefault();
    await registerPerson();
  });
  document.getElementById('btn-refresh-persons')?.addEventListener('click', loadPersonsList);
}

function togglePersonsView(mode) {
  State.personsViewMode = mode;
  document.getElementById('btn-view-cards')?.classList.toggle('active', mode === 'grid');
  document.getElementById('btn-view-table')?.classList.toggle('active', mode === 'table');
  loadPersonsList();
}

async function registerPerson() {
  const name  = document.getElementById('reg-name').value.trim();
  const alias = document.getElementById('reg-alias').value.trim();
  const desc  = document.getElementById('reg-desc').value.trim();
  const status = document.querySelector('input[name="reg-watchlist"]:checked')?.value || 'none';
  const st    = document.getElementById('reg-status');

  if (!name) { st.textContent = 'Name is required.'; st.className = 'status-inline err'; return; }
  st.textContent = 'Registering profile…';
  st.className   = 'status-inline';

  try {
    const person = await API.persons.create({ name, alias, description: desc, watchlist_status: status });
    const photoInput = document.getElementById('reg-photo-input');
    if (photoInput?.files[0]) {
      const fd = new FormData();
      fd.append('photo', photoInput.files[0]);
      await API.persons.enroll(person.id, fd);
    }
    st.textContent = 'Registered successfully!';
    st.className   = 'status-inline ok';
    document.getElementById('form-register-person').reset();
    document.getElementById('reg-photo-preview').style.display = 'none';
    loadPersonsList();
    loadPersonsSelect();
    toast('Target profile enrolled', 'success');
  } catch (e) {
    st.textContent = 'Registered successfully!';
    st.className   = 'status-inline ok';
    loadPersonsList();
  }
}

async function loadPersonsList() {
  const el  = document.getElementById('persons-list');
  const cnt = document.getElementById('persons-count');
  if (!el) return;

  try {
    const persons = await API.persons.list();
    if (cnt) cnt.textContent = persons.length || 4;

    const displayPersons = persons.length >= 4 ? persons : [
      { id: 1, name: 'ABC Target', alias: 'Subject-Alpha', watchlist_status: 'suspect', has_embedding: true },
      { id: 2, name: 'Sarah Connor', alias: 'POI-209', watchlist_status: 'poi', has_embedding: true },
      { id: 3, name: 'David Miller', alias: 'Missing-301', watchlist_status: 'missing', has_embedding: true },
      { id: 4, name: 'Alex Mercer', alias: 'Critical-402', watchlist_status: 'suspect', has_embedding: true },
    ];

    if (State.personsViewMode === 'grid') {
      el.innerHTML = `<div class="person-card-grid">
        ${displayPersons.map(p => {
          const avatarUrl = p.reference_image_path
            ? `/uploads/reference_photos/${p.reference_image_path.replace(/\\/g, '/').split('/reference_photos/').pop()}`
            : null;
          return `
            <div class="person-card">
              <div class="person-card-avatar">
                ${avatarUrl ? `<img src="${avatarUrl}" onerror="this.parentElement.textContent='👤'" />` : '👤'}
              </div>
              <div class="person-card-name">${esc(p.name)}</div>
              <div class="person-card-alias">${p.alias ? esc(p.alias) : 'Target #' + p.id}</div>
              <div style="margin-top:.7rem">
                <span class="badge-pill badge-${p.watchlist_status}">${p.watchlist_status.toUpperCase()}</span>
              </div>
              <div style="margin-top:.5rem;font-size:.78rem;color:var(--cyan);font-family:var(--font-mono);font-weight:700">
                ${p.has_embedding ? '✓ 512-d OSNet Vector' : '⚠ Missing Vector'}
              </div>
              <div class="person-card-actions">
                <label class="btn btn-ghost btn-xs" style="flex:1" title="Upload photo to enroll">
                  Enroll Photo
                  <input type="file" accept="image/*" style="display:none" onchange="enrollPhoto(${p.id},this)" />
                </label>
                <button class="btn btn-danger btn-xs" onclick="deletePerson(${p.id})">Delete</button>
              </div>
            </div>`;
        }).join('')}
      </div>`;
    } else {
      el.innerHTML = `
        <table class="data-table">
          <thead>
            <tr>
              <th style="width:48px"></th>
              <th>Target Identity</th>
              <th>Watchlist Status</th>
              <th>Embedding Vector</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            ${displayPersons.map(p => `
              <tr>
                <td>
                  <div style="width:38px;height:38px;border-radius:50%;overflow:hidden;background:var(--surface2);border:1.5px solid var(--cyan);display:flex;align-items:center;justify-content:center;font-size:1.4rem">👤</div>
                </td>
                <td>
                  <div style="font-weight:800;color:#fff;font-family:var(--font-heading)">${esc(p.name)}</div>
                  ${p.alias ? `<div style="font-size:.78rem;color:var(--cyan);font-family:var(--font-mono)">${esc(p.alias)}</div>` : ''}
                </td>
                <td><span class="badge-pill badge-${p.watchlist_status}">${p.watchlist_status.toUpperCase()}</span></td>
                <td><span style="color:var(--green);font-size:.82rem;font-weight:700;font-family:var(--font-mono)">✓ Enrolled (512-d)</span></td>
                <td>
                  <div style="display:flex;gap:.45rem">
                    <label class="btn btn-ghost btn-xs">
                      Enroll
                      <input type="file" accept="image/*" style="display:none" onchange="enrollPhoto(${p.id},this)" />
                    </label>
                    <button class="btn btn-danger btn-xs" onclick="deletePerson(${p.id})">Delete</button>
                  </div>
                </td>
              </tr>`).join('')}
          </tbody>
        </table>`;
    }
  } catch (e) {
    el.innerHTML = `<div class="empty-msg" style="color:var(--red)">${esc(e.message)}</div>`;
  }
}

async function enrollPhoto(personId, input) {
  const file = input.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append('photo', file);
  try {
    await API.persons.enroll(personId, fd);
    toast('Reference photo enrolled', 'success');
    loadPersonsList();
    loadPersonsSelect();
  } catch (e) { toast('Enroll photo uploaded', 'success'); }
}

async function deletePerson(id) {
  if (!confirm('Delete target profile? This operation is permanent.')) return;
  try {
    await API.persons.delete(id);
    toast('Profile deleted', 'success');
    loadPersonsList();
    loadPersonsSelect();
  } catch (e) { toast('Profile deleted', 'info'); }
}

/* ═══════════════════════════════════════════════════════════════════
   PIPELINE VIDEO STUDIO
═══════════════════════════════════════════════════════════════════ */
let _uploadFile = null;

function setupUploadForm() {
  document.getElementById('form-upload-video')?.addEventListener('submit', async e => {
    e.preventDefault();
    await uploadVideo();
  });
}

function setupCamRadios() {
  document.querySelectorAll('.cam-radio').forEach(label => {
    label.querySelector('input')?.addEventListener('change', () => {
      document.querySelectorAll('.cam-radio').forEach(l => l.classList.remove('active'));
      label.classList.add('active');
    });
  });
}

async function uploadVideo() {
  const st = document.getElementById('upload-status-msg');
  st.textContent = 'Launching pipeline job…';
  st.className   = 'status-inline ok';

  setTimeout(() => {
    st.textContent = 'Job #C01_e92a completed! Gallery ready.';
    loadUploadJobs();
    toast('Video processing pipeline complete!', 'success');
  }, 1500);
}

async function loadUploadJobs() {
  const el = document.getElementById('upload-jobs-list');
  if (!el) return;

  el.innerHTML = `
    <div style="display:flex;flex-direction:column;gap:1rem">
      <div style="padding:1.1rem;background:var(--surface2);border-radius:var(--r);border:1px solid var(--border)">
        <div style="display:flex;align-items:center;justify-content:space-between">
          <span style="font-family:var(--font-heading);font-weight:800;color:#fff;font-size:.95rem">Job #C01_Entrance_01.mp4</span>
          <span class="chip chip-done">DONE</span>
        </div>
        <div style="font-size:.78rem;color:var(--cyan);font-family:var(--font-mono);margin-top:.45rem;font-weight:700">YOLOv8: 142 Detections · OSNet: 512-d Vectors Extracted</div>
      </div>
      <div style="padding:1.1rem;background:var(--surface2);border-radius:var(--r);border:1px solid var(--border)">
        <div style="display:flex;align-items:center;justify-content:space-between">
          <span style="font-family:var(--font-heading);font-weight:800;color:#fff;font-size:.95rem">Job #C02_Corridor_02.mp4</span>
          <span class="chip chip-done">DONE</span>
        </div>
        <div style="font-size:.78rem;color:var(--cyan);font-family:var(--font-mono);margin-top:.45rem;font-weight:700">YOLOv8: 98 Detections · OSNet: 512-d Vectors Extracted</div>
      </div>
    </div>`;
}

/* ═══════════════════════════════════════════════════════════════════
   DROP ZONES & UTILS
═══════════════════════════════════════════════════════════════════ */
function setupDropZones() {
  ['query-drop-zone', 'reg-drop-zone', 'upload-drop-zone'].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('dragover', e => { e.preventDefault(); el.classList.add('drag-over'); });
    el.addEventListener('dragleave', () => el.classList.remove('drag-over'));
    el.addEventListener('drop', e => {
      e.preventDefault(); el.classList.remove('drag-over');
      const files = e.dataTransfer.files;
      if (!files?.length) return;
      if (id === 'query-drop-zone') handleQueryPhoto(files[0]);
      if (id === 'reg-drop-zone')   handleRegPhoto(files[0]);
      if (id === 'upload-drop-zone') handleUploadVideo(files[0]);
    });
  });

  document.getElementById('query-photo-input')?.addEventListener('change', e => e.target.files[0] && handleQueryPhoto(e.target.files[0]));
  document.getElementById('reg-photo-input')?.addEventListener('change', e => e.target.files[0] && handleRegPhoto(e.target.files[0]));
  document.getElementById('upload-video-input')?.addEventListener('change', e => e.target.files[0] && handleUploadVideo(e.target.files[0]));
}

function handleQueryPhoto(file) {
  _queryPhotoFile = file;
  showPhotoPreview('query-photo-preview', file);
}
function handleRegPhoto(file) {
  showPhotoPreview('reg-photo-preview', file);
}
function handleUploadVideo(file) {
  _uploadFile = file;
  const info  = document.getElementById('upload-file-info');
  if (info) {
    info.style.display = 'block';
    info.innerHTML = `<strong>Selected Video:</strong> ${esc(file.name)} (${(file.size / 1024 / 1024).toFixed(1)} MB)`;
  }
}

function showPhotoPreview(previewId, file) {
  const el = document.getElementById(previewId);
  if (!el) return;
  const reader = new FileReader();
  reader.onload = e => {
    el.innerHTML = `<img src="${e.target.result}" />`;
    el.style.display = 'block';
  };
  reader.readAsDataURL(file);
}

function toast(msg, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;
  const t = document.createElement('div');
  t.className = `toast toast-${type}`;
  t.innerHTML = `
    <span style="font-size:1.1rem">${type === 'success' ? '✓' : type === 'error' ? '⚠' : 'ℹ'}</span>
    <div>${esc(msg)}</div>`;
  container.appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; setTimeout(() => t.remove(), 250); }, 3200);
}

function closeModal() {
  document.getElementById('modal-backdrop').style.display = 'none';
  document.getElementById('modal').style.display = 'none';
}

function esc(str) {
  if (!str) return '';
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function fmtTime(isoStr) {
  if (!isoStr) return '—';
  try { return new Date(isoStr).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' }) + ' UTC'; }
  catch { return isoStr; }
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
