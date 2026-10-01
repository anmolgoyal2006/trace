"""
schemas.py — Pydantic request / response schemas for the unified Trace API.

Organised by resource:
  Person          — registration, update, search
  Camera          — list, detail
  QuerySession    — submit, status poll, route result
  Sighting        — per-camera match detail
  RouteStep       — one hop in the reconstructed path
  Alert           — watchlist notification
  Analytics       — dashboard stats
  Upload          — video upload response
  WebSocket       — pushed event envelopes
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

class _Base(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ===========================================================================
# Person
# ===========================================================================

class PersonCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    alias: Optional[str] = None
    description: Optional[str] = None
    watchlist_status: str = Field(
        default="none",
        pattern="^(none|missing|suspect|poi)$",
    )


class PersonUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    alias: Optional[str] = None
    description: Optional[str] = None
    watchlist_status: Optional[str] = Field(
        default=None,
        pattern="^(none|missing|suspect|poi)$",
    )


class PersonOut(_Base):
    id: int
    name: str
    alias: Optional[str]
    description: Optional[str]
    watchlist_status: str
    reference_image_path: Optional[str]
    has_embedding: bool = Field(
        default=False,
        description="True if a reference OSNet embedding has been enrolled",
    )
    is_active: bool
    created_at: datetime

    @classmethod
    def from_orm_with_embedding(cls, person: Any) -> "PersonOut":
        return cls(
            id=person.id,
            name=person.name,
            alias=person.alias,
            description=person.description,
            watchlist_status=person.watchlist_status,
            reference_image_path=person.reference_image_path,
            has_embedding=person.reference_embedding is not None,
            is_active=person.is_active,
            created_at=person.created_at,
        )


class PersonSearchResult(_Base):
    """Returned by POST /api/persons/search — image-based identity lookup."""
    person: PersonOut
    similarity: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=100.0)


# ===========================================================================
# Camera
# ===========================================================================

class CameraOut(_Base):
    id: str
    location: str
    description: Optional[str]
    start_time: Optional[str]
    is_active: bool


class CameraGraphEdge(BaseModel):
    """One directed edge in the camera adjacency graph."""
    to_camera_id: str
    avg_transit_sec: int
    notes: Optional[str] = None


class CameraWithGraph(_Base):
    id: str
    location: str
    description: Optional[str]
    start_time: Optional[str]
    is_active: bool
    connects_to: list[CameraGraphEdge] = []


class CameraUpdate(BaseModel):
    """
    User-editable camera fields.

    location    — human-readable place name shown in the UI and map.
    start_time  — HH:MM:SS wall-clock time when the recording started.
    transit_updates — map of {neighbour_camera_id: avg_transit_sec} to
                      patch specific edges in camera_graph.json.
    """
    location: Optional[str] = Field(default=None, min_length=1, max_length=200)
    start_time: Optional[str] = Field(
        default=None,
        pattern=r"^\d{2}:\d{2}(:\d{2})?$",
        description="HH:MM or HH:MM:SS",
    )
    transit_updates: Optional[dict[str, int]] = Field(
        default=None,
        description="Neighbour camera ID → new avg_transit_sec",
    )


# ===========================================================================
# Sighting
# ===========================================================================

class SightingOut(_Base):
    id: int
    session_id: int
    camera_id: str
    track_id: int
    first_seen: Optional[str]
    last_seen: Optional[str]
    # Similarity-derived match score [0,100] — NOT an identity probability.
    # Label as "Match score" or "Similarity score" in the UI, NOT "Accuracy".
    best_confidence: float
    mean_confidence: float
    crop_path: Optional[str]
    appearance_score: float
    spatial_score: float
    temporal_score: float
    fusion_score: float
    # Identity decision fields
    match_status: str = "POSSIBLE_MATCH_REVIEW"
    face_veto_applied: bool = False
    face_veto_reason: Optional[str] = None
    # Signal breakdown for debugging / display
    face_sim: Optional[float] = None
    face_coverage: Optional[float] = None
    kpr_sim: Optional[float] = None
    fused_score: float = 0.0
    active_backbone: str = "OSNet x1_0"
    embedding_dim: int = 512
    # ---- Phase 4: face-first identity fields ----------------------------
    matching_mode: str = "body"            # "face" | "body"
    face_used: bool = False                # True when ArcFace was the identity signal
    face_similarity: Optional[float] = None  # ArcFace cosine sim, None = no face
    body_similarity: float = 0.0           # OSNet / SOLIDER body cosine sim
    identity_score: float = 0.0            # face_sim OR body_sim, never blended


class CandidateTrackOut(BaseModel):
    """
    A top candidate track returned for human review even when there is no
    confident match.  These are the strongest body/KPR/face candidates so
    an operator can visually verify them.

    Note: best_confidence is a similarity-derived match score, NOT a
    probability of identity.
    """
    camera_id: str
    track_id: int
    # Similarity-derived match score [0,100] — NOT identity probability
    best_confidence: float
    appearance_score: float
    fused_score: float
    match_status: str
    face_veto_applied: bool = False
    face_sim: Optional[float] = None
    face_coverage: Optional[float] = None
    kpr_sim: Optional[float] = None
    first_seen: Optional[str] = None
    crop_path: Optional[str] = None
    active_backbone: str = "OSNet x1_0"


# ===========================================================================
# Route Step
# ===========================================================================

class RouteStepOut(_Base):
    step_order: int
    camera_id: str
    camera_location: str
    track_id: int
    timestamp: Optional[str]        # first_seen for that sighting
    confidence: float               # best_confidence
    fusion_score: float
    crop_path: Optional[str]
    # ---- Phase 4: face-first identity fields ----------------------------
    matching_mode: str = "body"    # "face" | "body"
    face_used: bool = False         # True when ArcFace was the identity signal
    identity_score: float = 0.0    # face_sim OR body_sim, never blended


# ===========================================================================
# Query Session
# ===========================================================================

class QuerySubmit(BaseModel):
    """
    Submitted via POST /api/queries.

    Either person_id or the multipart image upload field is required.
    The endpoint also accepts multipart/form-data with an `image` file field.
    """
    person_id: Optional[int] = Field(
        default=None,
        description="ID of a registered person — uses their enrolled embedding",
    )


class QueryStatusOut(_Base):
    id: int
    person_id: Optional[int]
    query_image_path: Optional[str]
    status: str
    progress_pct: int
    submitted_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str]


class QueryRouteOut(BaseModel):
    """Full route result returned by GET /api/queries/{id}/route."""
    session_id: int
    status: str
    steps: list[RouteStepOut]
    sightings: list[SightingOut]
    total_cameras_matched: int
    route_confidence: float = Field(
        description="Mean fusion score across all route steps"
    )
    # Overall identity decision for this query session.
    # CONFIDENT_MATCH   : ≥1 camera returned a CONFIDENT_MATCH sighting
    # POSSIBLE_MATCH_REVIEW: some candidates above soft_floor but none confident
    # NO_CONFIDENT_MATCH : no track cleared even the soft_floor threshold
    # Note: even when match_decision != CONFIDENT_MATCH, top_candidates are
    # returned so an operator can visually verify the strongest tracks.
    match_decision: str = "NO_CONFIDENT_MATCH"
    # Up to 3 unique top candidate tracks for human review.
    # Always populated regardless of match_decision.
    top_candidates: list[CandidateTrackOut] = Field(default_factory=list)


# ===========================================================================
# Alert
# ===========================================================================

class AlertOut(_Base):
    id: int
    session_id: int
    person_id: Optional[int]
    sighting_id: Optional[int]
    severity: str
    title: str
    message: str
    acknowledged: bool
    created_at: datetime


class AlertAcknowledge(BaseModel):
    acknowledged: bool = True


# ===========================================================================
# Analytics
# ===========================================================================

class DashboardStats(BaseModel):
    total_persons: int
    watchlist_persons: int
    total_queries: int
    completed_queries: int
    total_sightings: int
    total_alerts: int
    unacknowledged_alerts: int
    cameras_active: int


class CameraActivityItem(BaseModel):
    camera_id: str
    location: str
    total_sightings: int
    recent_queries: int


class AnalyticsDashboard(BaseModel):
    stats: DashboardStats
    recent_alerts: list[AlertOut]
    camera_activity: list[CameraActivityItem]


# ===========================================================================
# Upload
# ===========================================================================

class UploadJobOut(BaseModel):
    job_id: str
    camera_id: str
    filename: str
    status: str      # queued | processing | done | failed
    message: Optional[str] = None


# ===========================================================================
# WebSocket event envelopes
# ===========================================================================

class WsQueryProgress(BaseModel):
    """Pushed when a query moves forward through the pipeline."""
    type: str = "query_progress"
    session_id: int
    progress_pct: int
    message: str


class WsSightingFound(BaseModel):
    """Pushed when a sighting is confirmed on a camera."""
    type: str = "sighting_found"
    session_id: int
    sighting: SightingOut


class WsRouteComplete(BaseModel):
    """Pushed when route reconstruction finishes."""
    type: str = "route_complete"
    session_id: int
    route: QueryRouteOut


class WsAlert(BaseModel):
    """Pushed for watchlist hits."""
    type: str = "alert"
    alert: AlertOut


class WsError(BaseModel):
    """Pushed when a session errors out."""
    type: str = "error"
    session_id: int
    message: str


# Generic envelope — used for serialising any WS event to JSON
WsEvent = WsQueryProgress | WsSightingFound | WsRouteComplete | WsAlert | WsError
