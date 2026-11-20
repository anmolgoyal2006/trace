/**
 * api.ts — Typed REST client for the TRACE backend.
 * Mirrors backend/app/routers/*.py endpoints.
 */

import type {
  AnalyticsDashboard,
  AlertOut,
  CameraLocationUpdate,
  CameraMapsConfig,
  CameraSighting,
  CameraWithGraph,
  CameraUpdate,
  HealthResponse,
  PersonOut,
  PersonSearchResult,
  PersonUpdate,
  QueryRouteOut,
  QueryStatusOut,
  UploadJobOut,
} from "@/types/api";

const BASE = ""; // Same origin in production; Vite proxy in dev

// ── HTTP helpers ──────────────────────────────────────────────────────────────

async function request<T>(
  method: string,
  path: string,
  body?: BodyInit | null,
  headers?: Record<string, string>,
): Promise<T> {
  const opts: RequestInit = { method, headers: headers || {} };
  if (body !== null && body !== undefined) {
    opts.body = body;
  }
  const res = await fetch(BASE + path, opts);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const json = await res.json();
      detail = json.detail || detail;
    } catch {
      // response may not be JSON
    }
    const err = new Error(detail) as Error & { status: number };
    err.status = res.status;
    throw err;
  }
  if (res.status === 204) return null as T;
  return res.json() as Promise<T>;
}

function get<T>(path: string): Promise<T> {
  return request<T>("GET", path);
}

function postJson<T>(path: string, body: unknown): Promise<T> {
  return request<T>("POST", path, JSON.stringify(body), {
    "Content-Type": "application/json",
  });
}

function putJson<T>(path: string, body: unknown): Promise<T> {
  return request<T>("PUT", path, JSON.stringify(body), {
    "Content-Type": "application/json",
  });
}

function patchJson<T>(path: string, body: unknown): Promise<T> {
  return request<T>("PATCH", path, JSON.stringify(body), {
    "Content-Type": "application/json",
  });
}

function del<T>(path: string): Promise<T> {
  return request<T>("DELETE", path);
}

function postForm<T>(path: string, formData: FormData): Promise<T> {
  return request<T>("POST", path, formData);
}

// ── Persons API ─────────────────────────────────────────────────────────────

export const personsApi = {
  list: (watchlistOnly = false) =>
    get<PersonOut[]>(`/api/persons${watchlistOnly ? "?watchlist_only=true" : ""}`),

  get: (id: number) => get<PersonOut>(`/api/persons/${id}`),

  create: (fd: FormData) => postForm<PersonOut>("/api/persons", fd),

  update: (id: number, body: PersonUpdate) =>
    putJson<PersonOut>(`/api/persons/${id}`, body),

  delete: (id: number) => del<void>(`/api/persons/${id}`),

  enroll: (id: number, fd: FormData) =>
    postForm<PersonOut>(`/api/persons/${id}/enroll`, fd),

  search: (fd: FormData) =>
    postForm<PersonSearchResult[]>("/api/persons/search", fd),
};

// ── Cameras API ──────────────────────────────────────────────────────────────

export const camerasApi = {
  list: () => get<CameraWithGraph[]>("/api/cameras"),

  get: (id: string) => get<CameraWithGraph>(`/api/cameras/${id}`),

  patch: (id: string, body: CameraUpdate) =>
    patchJson<CameraWithGraph>(`/api/cameras/${id}`, body),

  sightings: (id: string, limit = 20) =>
    get<CameraSighting[]>(`/api/cameras/${id}/sightings?limit=${limit}`),

  mapsConfig: () => get<CameraMapsConfig>("/api/cameras/maps-config"),

  updateLocation: (id: string, body: CameraLocationUpdate) =>
    putJson<CameraWithGraph>(`/api/cameras/${id}/location`, body),
};

// ── Queries API ──────────────────────────────────────────────────────────────

export const queriesApi = {
  submit: (fd: FormData) => postForm<QueryStatusOut>("/api/queries", fd),

  status: (id: number) => get<QueryStatusOut>(`/api/queries/${id}`),

  route: (id: number) => get<QueryRouteOut>(`/api/queries/${id}/route`),

  list: (limit = 20) =>
    get<QueryStatusOut[]>(`/api/queries?limit=${limit}`),
};

// ── Analytics API ────────────────────────────────────────────────────────────

export const analyticsApi = {
  dashboard: () => get<AnalyticsDashboard>("/api/analytics/dashboard"),

  alerts: (unackOnly = false) =>
    get<AlertOut[]>(`/api/analytics/alerts${unackOnly ? "?unacknowledged_only=true" : ""}`),

  acknowledgeAlert: (id: number) =>
    postJson<AlertOut>(`/api/analytics/alerts/${id}/acknowledge`, {}),

  recentSightings: (limit = 50) =>
    get<Record<string, unknown>[]>(
      `/api/analytics/sightings/recent?limit=${limit}`,
    ),
};

// ── Upload API ────────────────────────────────────────────────────────────────

export const uploadApi = {
  video: (fd: FormData) => postForm<UploadJobOut>("/api/upload/video", fd),

  status: (jobId: string) => get<UploadJobOut>(`/api/upload/status/${jobId}`),

  jobs: () => get<UploadJobOut[]>("/api/upload/jobs"),
};

// ── Health ────────────────────────────────────────────────────────────────────

export const healthApi = {
  check: () => get<HealthResponse>("/api/health"),
};

// ── Aggregate export ──────────────────────────────────────────────────────────

export const api = {
  persons: personsApi,
  cameras: camerasApi,
  queries: queriesApi,
  analytics: analyticsApi,
  upload: uploadApi,
  health: healthApi,
};

export default api;
