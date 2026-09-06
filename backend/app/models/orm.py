"""
orm.py — SQLAlchemy ORM table definitions for the unified Trace system.

Tables:
  persons         — registered persons with optional watchlist status + OSNet embedding
  cameras         — camera registry (C01–C03 MVP)
  query_sessions  — one record per submitted search query
  sightings       — per-camera, per-track match result within a session
  routes          — ordered steps in a reconstructed cross-camera path
  alerts          — watchlist hit notifications
"""

import json
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database import Base


# ---------------------------------------------------------------------------
# Persons
# ---------------------------------------------------------------------------

class Person(Base):
    """
    A registered person in the system.

    reference_embedding is stored as a JSON string (list of 512 floats).
    Accessor .embedding_vector returns it as a Python list[float].
    """
    __tablename__ = "persons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    alias: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # none | missing | suspect | poi
    watchlist_status: Mapped[str] = mapped_column(Text, nullable=False, default="none")
    # JSON-serialised float list — 512-D OSNet embedding
    reference_embedding: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Path to the uploaded reference image on disk
    reference_image_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    # Relationships
    sessions: Mapped[list["QuerySession"]] = relationship(
        "QuerySession", back_populates="person", lazy="selectin"
    )
    alerts: Mapped[list["Alert"]] = relationship(
        "Alert", back_populates="person", lazy="selectin"
    )

    @property
    def embedding_vector(self) -> Optional[list[float]]:
        """Deserialise the stored JSON embedding to a Python list."""
        if self.reference_embedding is None:
            return None
        return json.loads(self.reference_embedding)

    @embedding_vector.setter
    def embedding_vector(self, vec: list[float]) -> None:
        """Serialise a Python list to the stored JSON string."""
        self.reference_embedding = json.dumps(vec)

    def __repr__(self) -> str:
        return f"<Person id={self.id} name={self.name!r} watchlist={self.watchlist_status}>"


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

class Camera(Base):
    """
    A registered camera. Seeded from camera_graph.json on startup.
    """
    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(Text, primary_key=True)   # e.g. "C01"
    location: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    start_time: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # HH:MM:SS
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Relationships
    sightings: Mapped[list["Sighting"]] = relationship(
        "Sighting", back_populates="camera", lazy="selectin"
    )

    def __repr__(self) -> str:
        return f"<Camera id={self.id!r} location={self.location!r}>"


# ---------------------------------------------------------------------------
# Query Sessions
# ---------------------------------------------------------------------------

class QuerySession(Base):
    """
    One submitted search query.

    Either person_id is set (search for a registered person) or
    query_image_path is set (ad-hoc photo upload search) — or both.
    """
    __tablename__ = "query_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    person_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("persons.id", ondelete="SET NULL"), nullable=True
    )
    query_image_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # pending | running | done | failed
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    progress_pct: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    person: Mapped[Optional["Person"]] = relationship(
        "Person", back_populates="sessions", lazy="selectin"
    )
    sightings: Mapped[list["Sighting"]] = relationship(
        "Sighting", back_populates="session", lazy="selectin",
        cascade="all, delete-orphan",
    )
    route_steps: Mapped[list["RouteStep"]] = relationship(
        "RouteStep", back_populates="session", lazy="selectin",
        order_by="RouteStep.step_order",
        cascade="all, delete-orphan",
    )
    alerts: Mapped[list["Alert"]] = relationship(
        "Alert", back_populates="session", lazy="selectin",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<QuerySession id={self.id} status={self.status!r}>"


# ---------------------------------------------------------------------------
# Sightings
# ---------------------------------------------------------------------------

class Sighting(Base):
    """
    Aggregated per-camera match result within a query session.

    Stores both the raw appearance score and the fused score so the UI
    can show either the raw re-ID confidence or the full spatial+temporal
    fusion score.
    """
    __tablename__ = "sightings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("query_sessions.id", ondelete="CASCADE"), nullable=False
    )
    camera_id: Mapped[str] = mapped_column(
        Text, ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False
    )
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    first_seen: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # HH:MM:SS.ff
    last_seen: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Confidence values derived from OSNet similarity
    best_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    mean_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # Best crop image path for thumbnail display
    crop_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Raw appearance signal (cosine similarity)
    appearance_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # Spatial plausibility [0, 1]
    spatial_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # Temporal plausibility [0, 1]
    temporal_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # Weighted fusion of above three signals
    fusion_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Relationships
    session: Mapped["QuerySession"] = relationship(
        "QuerySession", back_populates="sightings", lazy="selectin"
    )
    camera: Mapped["Camera"] = relationship(
        "Camera", back_populates="sightings", lazy="selectin"
    )
    route_step: Mapped[Optional["RouteStep"]] = relationship(
        "RouteStep", back_populates="sighting", uselist=False, lazy="selectin"
    )

    def __repr__(self) -> str:
        return (
            f"<Sighting id={self.id} session={self.session_id} "
            f"camera={self.camera_id!r} track={self.track_id} "
            f"fusion={self.fusion_score:.3f}>"
        )


# ---------------------------------------------------------------------------
# Route Steps
# ---------------------------------------------------------------------------

class RouteStep(Base):
    """
    One ordered step in the reconstructed cross-camera route.

    A complete route is session.route_steps ordered by step_order.
    """
    __tablename__ = "routes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("query_sessions.id", ondelete="CASCADE"), nullable=False
    )
    step_order: Mapped[int] = mapped_column(Integer, nullable=False)
    sighting_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("sightings.id", ondelete="CASCADE"), nullable=False
    )

    # Relationships
    session: Mapped["QuerySession"] = relationship(
        "QuerySession", back_populates="route_steps", lazy="selectin"
    )
    sighting: Mapped["Sighting"] = relationship(
        "Sighting", back_populates="route_step", lazy="selectin"
    )

    def __repr__(self) -> str:
        return (
            f"<RouteStep session={self.session_id} "
            f"step={self.step_order} sighting={self.sighting_id}>"
        )


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

class Alert(Base):
    """
    A watchlist hit notification generated when a registered person
    (watchlist_status != 'none') is matched during a query session.
    """
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("query_sessions.id", ondelete="CASCADE"), nullable=False
    )
    person_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("persons.id", ondelete="SET NULL"), nullable=True
    )
    sighting_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("sightings.id", ondelete="SET NULL"), nullable=True
    )
    # HIGH | MEDIUM | LOW
    severity: Mapped[str] = mapped_column(Text, nullable=False, default="MEDIUM")
    title: Mapped[str] = mapped_column(Text, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    acknowledged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    # Relationships
    session: Mapped["QuerySession"] = relationship(
        "QuerySession", back_populates="alerts", lazy="selectin"
    )
    person: Mapped[Optional["Person"]] = relationship(
        "Person", back_populates="alerts", lazy="selectin"
    )

    def __repr__(self) -> str:
        return (
            f"<Alert id={self.id} severity={self.severity!r} "
            f"person={self.person_id} ack={self.acknowledged}>"
        )
