/**
 * api.ts — TypeScript interfaces mirroring the backend Pydantic schemas.
 * Source of truth: backend/app/models/schemas.py
 */

// ── Person ──────────────────────────────────────────────────────────────────

export interface PersonOut {
  id: number;
  name: string;
  alias: string | null;
  description: string | null;
  watchlist_status: "none" | "missing" | "suspect" | "poi";
  reference_image_path: string | null;
  has_embedding: boolean;
  is_active: boolean;
  created_at: string; // ISO datetime
}

export interface PersonUpdate {
  name?: string;
  alias?: string;
  description?: string;
  watchlist_status?: "none" | "missing" | "suspect" | "poi";
}

export interface PersonSearchResult {
  person: PersonOut;
  similarity: number; // 0–1
  confidence: number; // 0–100
}

// ── Camera ──────────────────────────────────────────────────────────────────

export interface CameraGraphEdge {
  to_camera_id: string;
  avg_transit_sec: number;
  notes: string | null;
}

export interface CameraWithGraph {
  id: string;
  location: string;
  description: string | null;
  start_time: string | null;
  is_active: boolean;
  connects_to: CameraGraphEdge[];
}

export interface CameraUpdate {
  location?: string;
  start_time?: string;
  transit_updates?: Record<string, number>;
}

export interface CameraSighting {
  id: number;
  session_id: number;
  track_id: number;
  first_seen: string | null;
  best_confidence: number;
  fusion_score: number;
  crop_path: string | null;
}

// ── Sighting ─────────────────────────────────────────────────────────────────

export type MatchStatus =
  | "CONFIDENT_MATCH"
  | "POSSIBLE_MATCH_REVIEW"
  | "FACE_MISMATCH"
  | "NO_CONFIDENT_MATCH"
  | "NO_USABLE_EVIDENCE";

export interface SightingOut {
  id: number;
  session_id: number;
  camera_id: string;
  track_id: number;
  first_seen: string | null;
  last_seen: string | null;
  best_confidence: number; // 0–100, similarity-derived
  mean_confidence: number;
  crop_path: string | null;
  appearance_score: number;
  spatial_score: number;
  temporal_score: number;
  fusion_score: number;
  match_status: MatchStatus;
  face_veto_applied: boolean;
  face_veto_reason: string | null;
  face_sim: number | null;
  face_coverage: number | null;
  kpr_sim: number | null;
  fused_score: number;
  active_backbone: string;
  embedding_dim: number;
}

export interface CandidateTrackOut {
  camera_id: string;
  track_id: number;
  best_confidence: number;
  appearance_score: number;
  fused_score: number;
  match_status: MatchStatus;
  face_veto_applied: boolean;
  face_sim: number | null;
  face_coverage: number | null;
  kpr_sim: number | null;
  first_seen: string | null;
  crop_path: string | null;
  active_backbone: string;
}

// ── Route Step ───────────────────────────────────────────────────────────────

export interface RouteStepOut {
  step_order: number;
  camera_id: string;
  camera_location: string;
  track_id: number;
  timestamp: string | null;
  confidence: number;
  fusion_score: number;
  crop_path: string | null;
}

// ── Query Session ────────────────────────────────────────────────────────────

export type QueryStatus = "pending" | "running" | "done" | "failed";

export interface QueryStatusOut {
  id: number;
  person_id: number | null;
  query_image_path: string | null;
  status: QueryStatus;
  progress_pct: number;
  submitted_at: string;
  completed_at: string | null;
  error_message: string | null;
}

export type MatchDecision =
  | "CONFIDENT_MATCH"
  | "POSSIBLE_MATCH_REVIEW"
  | "NO_CONFIDENT_MATCH";

export interface QueryRouteOut {
  session_id: number;
  status: string;
  steps: RouteStepOut[];
  sightings: SightingOut[];
  total_cameras_matched: number;
  route_confidence: number;
  match_decision: MatchDecision;
  top_candidates: CandidateTrackOut[];
}

// ── Alert ────────────────────────────────────────────────────────────────────

export interface AlertOut {
  id: number;
  session_id: number;
  person_id: number | null;
  sighting_id: number | null;
  severity: string;
  title: string;
  message: string;
  acknowledged: boolean;
  created_at: string;
}

// ── Analytics ────────────────────────────────────────────────────────────────

export interface DashboardStats {
  total_persons: number;
  watchlist_persons: number;
  total_queries: number;
  completed_queries: number;
  total_sightings: number;
  total_alerts: number;
  unacknowledged_alerts: number;
  cameras_active: number;
}

export interface CameraActivityItem {
  camera_id: string;
  location: string;
  total_sightings: number;
  recent_queries: number;
}

export interface AnalyticsDashboard {
  stats: DashboardStats;
  recent_alerts: AlertOut[];
  camera_activity: CameraActivityItem[];
}

// ── Upload ────────────────────────────────────────────────────────────────────

export type UploadJobStatus = "queued" | "processing" | "done" | "failed";

export interface UploadJobOut {
  job_id: string;
  camera_id: string;
  filename: string;
  status: UploadJobStatus;
  message: string | null;
}

// ── Health ───────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  ws_connections: number;
}

// ── WebSocket Events ─────────────────────────────────────────────────────────

export interface WsQueryProgress {
  type: "query_progress";
  session_id: number;
  progress_pct: number;
  message: string;
}

export interface WsSightingFound {
  type: "sighting_found";
  session_id: number;
  sighting: SightingOut;
}

export interface WsRouteComplete {
  type: "route_complete";
  session_id: number;
  route: QueryRouteOut;
}

export interface WsAlert {
  type: "alert";
  alert: AlertOut;
}

export interface WsError {
  type: "error";
  session_id: number;
  message: string;
}

export type WsEvent =
  | WsQueryProgress
  | WsSightingFound
  | WsRouteComplete
  | WsAlert
  | WsError;
