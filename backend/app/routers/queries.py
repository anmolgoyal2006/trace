"""
queries.py — /api/queries
Submit a search query, poll status, retrieve route results.

Single-search-at-a-time enforced: a new query is rejected with 409 if
another session is currently running.
"""

import json
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import settings
from backend.app.database import get_db
from backend.app.models.orm import Person, QuerySession, RouteStep, Sighting
from backend.app.models.schemas import (
    QueryRouteOut,
    QueryStatusOut,
    RouteStepOut,
    SightingOut,
)
from backend.app.services.embedding_service import embedding_service

router = APIRouter(prefix="/api/queries", tags=["queries"])

# Injected at startup by main.py
_pipeline_service = None


def set_pipeline_service(svc) -> None:
    global _pipeline_service
    _pipeline_service = svc


# ---------------------------------------------------------------------------
# Module-level camera graph cache
# Loaded once on first use — avoids repeated disk reads in get_route()
# ---------------------------------------------------------------------------
_cam_data_cache: Optional[dict] = None


def _get_cam_data() -> dict:
    """Return camera metadata dict from camera_graph.json, cached after first load."""
    global _cam_data_cache
    if _cam_data_cache is None:
        try:
            with open(settings.camera_graph_path) as f:
                _cam_data_cache = json.load(f).get("cameras", {})
        except FileNotFoundError:
            logger.warning(f"[queries] camera_graph.json not found: {settings.camera_graph_path}")
            _cam_data_cache = {}
    return _cam_data_cache


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

@router.post("", response_model=QueryStatusOut, status_code=status.HTTP_202_ACCEPTED)
async def submit_query(
    background_tasks: BackgroundTasks,
    person_id: Optional[int] = Form(default=None),
    image: Optional[UploadFile] = File(default=None),
    db: AsyncSession = Depends(get_db),
) -> QueryStatusOut:
    """
    Submit a search query.

    Accepts either:
      - person_id (form field) — uses the enrolled OSNet embedding
      - image (file upload) — embeds on the fly

    Returns 409 if another query is already running.
    Returns 400 if neither person_id nor image is provided.
    """
    # ── guard: pipeline service must be ready ─────────────────────────
    if _pipeline_service is None:
        raise HTTPException(status_code=503, detail="Pipeline service not initialised.")

    # ── single-search-at-a-time guard ─────────────────────────────────
    running = await db.execute(
        select(QuerySession).where(QuerySession.status == "running")
    )
    if running.scalars().first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A query is already running. Wait for it to complete before submitting a new one.",
        )

    if person_id is None and image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide either person_id or an image upload.",
        )

    # ── if image submitted, embedding model must be ready ─────────────
    if image is not None and person_id is None and not embedding_service.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Embedding model not loaded yet. Use a registered person with an enrolled photo, or wait for model startup.",
        )

    # ── validate person if given ───────────────────────────────────────
    query_image_path: Optional[str] = None
    if person_id is not None:
        person = await db.get(Person, person_id)
        if not person or not person.is_active:
            raise HTTPException(status_code=404, detail="Person not found")
        if person.reference_embedding is None:
            raise HTTPException(
                status_code=400,
                detail="Person has no enrolled embedding. Upload a reference photo first via POST /api/persons/{id}/enroll",
            )

    # ── save uploaded image ────────────────────────────────────────────
    if image is not None:
        dest_dir = settings.uploads_dir / "query_images"
        dest_dir.mkdir(parents=True, exist_ok=True)
        ext = Path(image.filename).suffix.lower() if image.filename else ".jpg"
        filename = f"query_{uuid.uuid4().hex}{ext}"
        dest = dest_dir / filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(image.file, f)
        query_image_path = str(dest)

    # ── create session record ──────────────────────────────────────────
    session = QuerySession(
        person_id=person_id,
        query_image_path=query_image_path,
        status="pending",
        progress_pct=0,
    )
    db.add(session)
    await db.commit()

    logger.info(f"[queries] Session {session.id} created (person_id={person_id})")

    background_tasks.add_task(_pipeline_service.run_session, session.id)

    return QueryStatusOut.model_validate(session)


# ---------------------------------------------------------------------------
# Poll status
# ---------------------------------------------------------------------------

@router.get("/{session_id}", response_model=QueryStatusOut)
async def get_session_status(
    session_id: int,
    db: AsyncSession = Depends(get_db),
) -> QueryStatusOut:
    session = await db.get(QuerySession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return QueryStatusOut.model_validate(session)


# ---------------------------------------------------------------------------
# Route result
# ---------------------------------------------------------------------------

@router.get("/{session_id}/route", response_model=QueryRouteOut)
async def get_route(
    session_id: int,
    db: AsyncSession = Depends(get_db),
) -> QueryRouteOut:
    """
    Retrieve the reconstructed route for a completed query session.
    Returns 404 if session not found, 400 if session hasn't completed yet.
    """
    session = await db.get(QuerySession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status not in ("done", "failed"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session is still {session.status}. Poll GET /api/queries/{session_id} for status.",
        )

    # Load sightings
    sightings_result = await db.execute(
        select(Sighting).where(Sighting.session_id == session_id)
    )
    sightings = sightings_result.scalars().all()

    # Load route steps ordered
    steps_result = await db.execute(
        select(RouteStep)
        .where(RouteStep.session_id == session_id)
        .order_by(RouteStep.step_order)
    )
    steps = steps_result.scalars().all()

    # Build a lookup: sighting_id → Sighting
    sighting_map = {s.id: s for s in sightings}

    cam_data = _get_cam_data()

    step_outs: list[RouteStepOut] = []
    for step in steps:
        s = sighting_map.get(step.sighting_id)
        if s is None:
            continue
        location = cam_data.get(s.camera_id, {}).get("location", s.camera_id)
        step_outs.append(RouteStepOut(
            step_order=step.step_order,
            camera_id=s.camera_id,
            camera_location=location,
            track_id=s.track_id,
            timestamp=s.first_seen,
            confidence=s.best_confidence,
            fusion_score=s.fusion_score,
            crop_path=s.crop_path,
        ))

    # Route confidence = mean fusion score
    route_confidence = (
        round(sum(s.fusion_score for s in sightings) / len(sightings), 4)
        if sightings else 0.0
    )

    return QueryRouteOut(
        session_id=session_id,
        status=session.status,
        steps=step_outs,
        sightings=[SightingOut.model_validate(s) for s in sightings],
        total_cameras_matched=len(sightings),
        route_confidence=route_confidence,
    )


# ---------------------------------------------------------------------------
# List all sessions
# ---------------------------------------------------------------------------

@router.get("", response_model=list[QueryStatusOut])
async def list_sessions(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
) -> list[QueryStatusOut]:
    result = await db.execute(
        select(QuerySession)
        .order_by(QuerySession.submitted_at.desc())
        .limit(limit)
    )
    sessions = result.scalars().all()
    return [QueryStatusOut.model_validate(s) for s in sessions]
