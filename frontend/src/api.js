/**
 * api.js — REST + WebSocket client for the Trace backend.
 *
 * Exports a global `API` object with methods for every endpoint.
 * All methods return Promises. Errors throw with a .message property.
 */

const API = (() => {
  const BASE = '';  // same origin — FastAPI serves the frontend

  // ── HTTP helpers ────────────────────────────────────────────────────────

  async function _request(method, path, body = null, isFormData = false) {
    const opts = { method, headers: {} };
    if (body !== null) {
      if (isFormData) {
        opts.body = body;  // FormData — browser sets Content-Type automatically
      } else {
        opts.headers['Content-Type'] = 'application/json';
        opts.body = JSON.stringify(body);
      }
    }
    const res = await fetch(BASE + path, opts);
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try { detail = (await res.json()).detail || detail; } catch (_) {}
      const err = new Error(detail);
      err.status = res.status;
      throw err;
    }
    if (res.status === 204) return null;
    return res.json();
  }

  const get    = (path)        => _request('GET',    path);
  const post   = (path, body)  => _request('POST',   path, body);
  const put    = (path, body)  => _request('PUT',    path, body);
  const del    = (path)        => _request('DELETE', path);
  const postForm = (path, fd)  => _request('POST',   path, fd, true);

  // ── Persons ─────────────────────────────────────────────────────────────

  const persons = {
    list:      (watchlistOnly = false) => get(`/api/persons${watchlistOnly ? '?watchlist_only=true' : ''}`),
    get:       (id)           => get(`/api/persons/${id}`),
    create:    (fd)           => postForm('/api/persons', fd),
    update:    (id, body)     => put(`/api/persons/${id}`, body),
    delete:    (id)           => del(`/api/persons/${id}`),
    enroll:    (id, fd)       => postForm(`/api/persons/${id}/enroll`, fd),
    search:    (fd)           => postForm('/api/persons/search', fd),
  };

  // ── Cameras ─────────────────────────────────────────────────────────────

  const cameras = {
    list:      ()   => get('/api/cameras'),
    get:       (id) => get(`/api/cameras/${id}`),
    sightings: (id, limit = 20) => get(`/api/cameras/${id}/sightings?limit=${limit}`),
  };

  // ── Queries ─────────────────────────────────────────────────────────────

  const queries = {
    submit:    (fd)  => postForm('/api/queries', fd),
    status:    (id)  => get(`/api/queries/${id}`),
    route:     (id)  => get(`/api/queries/${id}/route`),
    list:      (limit = 20) => get(`/api/queries?limit=${limit}`),
  };

  // ── Analytics ───────────────────────────────────────────────────────────

  const analytics = {
    dashboard:          ()        => get('/api/analytics/dashboard'),
    alerts:             (unack)   => get(`/api/analytics/alerts${unack ? '?unacknowledged_only=true' : ''}`),
    acknowledgeAlert:   (id)      => post(`/api/analytics/alerts/${id}/acknowledge`, {}),
    recentSightings:    (limit)   => get(`/api/analytics/sightings/recent?limit=${limit || 50}`),
  };

  // ── Upload ──────────────────────────────────────────────────────────────

  const upload = {
    video:     (fd)  => postForm('/api/upload/video', fd),
    status:    (id)  => get(`/api/upload/status/${id}`),
    jobs:      ()    => get('/api/upload/jobs'),
  };

  // ── Health ──────────────────────────────────────────────────────────────

  const health = () => get('/api/health');

  // ── WebSocket ───────────────────────────────────────────────────────────

  function createWebSocket(onMessage, onOpen, onClose) {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onmessage = (e) => {
      try { onMessage(JSON.parse(e.data)); } catch (_) {}
    };
    ws.onopen    = () => onOpen && onOpen(ws);
    ws.onclose   = () => onClose && onClose();
    ws.onerror   = () => onClose && onClose();
    return ws;
  }

  return { persons, cameras, queries, analytics, upload, health, createWebSocket };
})();
