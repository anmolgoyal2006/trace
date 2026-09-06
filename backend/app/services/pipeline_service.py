"""
pipeline_service.py — PipelineService
End-to-end query pipeline orchestrator for the unified Trace system.

Handles one query session from start to finish:
  1. Load the query embedding (from registered person or uploaded photo).
  2. For each MVP camera: load its gallery embeddings.
  3. Run MatchingService.search_camera() per camera.
  4. Run RouteService.reconstruct_route() over the matches.
  5. Persist Sighting and RouteStep records to the database.
  6. Fire watchlist alerts if the queried person is on the watchlist.
  7. Push WebSocket progress events throughout.

The pipeline runs in a background asyncio task so the HTTP endpoint that
submitted the query can return immediately with the session ID.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import settings
from backend.app.database import async_session
from backend.app.models.orm import Alert, QuerySession, RouteStep, Sighting
from backend.app.models.schemas import (
    AlertOut,
    QueryRouteOut,
    RouteStepOut,
    SightingOut,
    WsAlert,
    WsError,
    WsQueryProgress,
    WsRouteComplete,
    WsSightingFound,
)
from backend.app.services.embedding_service import embedding_service
from backend.app.services.matching_service import CameraMatch, MatchingService
from backend.app.services.route_service import FusedSighting, RouteService

if TYPE_CHECKING:
    from backend.app.core.websocket_manager import WebSocketManager

# Module-level service singletons (shared across requests)
_matching_service = MatchingService()
_route_service: Optional[RouteService] = None


def get_route_service() -> RouteService:
    global _route_service
    if _route_service is None:
        _route_service = RouteService()
    return _route_service


# ---------------------------------------------------------------------------
# Gallery resolution
# ---------------------------------------------------------------------------

def _gallery_path_for_camera(camera_id: str) -> Optional[Path]:
    """
    Find the embeddings JSON file for a given camera.

    Looks for: dataset/embeddings_<camera_id>.json
    """
    path = settings.embeddings_dir / f"embeddings_{camera_id}.json"
    return path if path.exists() else None


def _load_galleries(camera_ids: list[str]) -> dict[str, list[dict]]:
    """Load all available gallery embedding files for the given cameras."""
    galleries: dict[str, list[dict]] = {}
    for cam_id in camera_ids:
        path = _gallery_path_for_camera(cam_id)
        if path is None:
            logger.warning(f"[PipelineService] No gallery found for camera {cam_id} — skipping")
            continue
        try:
            with open(path) as f:
                galleries[cam_id] = json.load(f)
            logger.info(f"[PipelineService] Loaded gallery: {cam_id} ({len(galleries[cam_id])} crops)")
        except Exception as e:
            logger.error(f"[PipelineService] Failed to load gallery {path}: {e}")
    return galleries


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PipelineService:
    """
    Orchestrates a full query session end-to-end.

    Instantiated once per app startup and shared across all sessions.
    """

    def __init__(self, ws_manager: "WebSocketManager") -> None:
        self._ws = ws_manager

    async def run_session(self, session_id: int) -> None:
        """
        Main entry point — runs in a background asyncio task.

        Loads the session from the DB, executes the pipeline, writes results,
        pushes WebSocket events, and updates the session status.
        """
        logger.info(f"[PipelineService] Session {session_id} starting")

        async with async_session() as db:
            session = await db.get(QuerySession, session_id)
            if session is None:
                logger.error(f"[PipelineService] Session {session_id} not found in DB")
                return

            # ── Mark running ─────────────────────────────────────────────
            session.status = "running"
            session.progress_pct = 0
            await db.commit()

            try:
                await self._execute(session, db)
            except Exception as e:
                logger.exception(f"[PipelineService] Session {session_id} failed: {e}")
                session.status = "failed"
                session.error_message = str(e)
                session.completed_at = datetime.utcnow()
                await db.commit()
                await self._ws.broadcast(WsError(
                    session_id=session_id,
                    message=str(e),
                ).model_dump())

    # ------------------------------------------------------------------ #
    # Internal execution                                                   #
    # ------------------------------------------------------------------ #

    async def _execute(self, session: QuerySession, db: AsyncSession) -> None:
        session_id = session.id

        # ── Step 1: resolve query embedding ──────────────────────────────
        await self._push_progress(session, db, 5, "Loading query embedding...")

        query_embedding = await self._resolve_query_embedding(session)
        if query_embedding is None:
            raise ValueError(
                "No embedding available. Enroll a reference photo for this person "
                "or upload a query image."
            )

        # ── Step 2: load camera galleries ────────────────────────────────
        await self._push_progress(session, db, 15, "Loading camera galleries...")

        mvp_cameras = ["C01", "C02", "C03"]
        galleries = _load_galleries(mvp_cameras)

        if not galleries:
            raise ValueError(
                "No gallery embeddings found for any MVP camera. "
                "Run the embedding pipeline first."
            )

        # ── Step 3: per-camera matching ───────────────────────────────────
        await self._push_progress(session, db, 25, "Running cross-camera matching...")

        camera_matches: dict[str, Optional[CameraMatch]] = {}
        n_cameras = len(galleries)

        for i, (cam_id, gallery) in enumerate(galleries.items()):
            pct = 25 + int((i / n_cameras) * 40)
            await self._push_progress(session, db, pct, f"Matching camera {cam_id}...")

            match = _matching_service.search_camera(
                query_embedding=query_embedding,
                gallery=gallery,
                camera_id=cam_id,
            )
            camera_matches[cam_id] = match

            # Persist sighting immediately if match found
            if match is not None:
                sighting = await self._persist_sighting(
                    db=db,
                    session_id=session_id,
                    match=match,
                    spatial=1.0,   # will be updated after route reconstruction
                    temporal=1.0,
                    fusion=match.appearance_score,
                )
                await db.commit()

                # Push sighting found event
                await self._ws.broadcast(WsSightingFound(
                    session_id=session_id,
                    sighting=SightingOut.model_validate(sighting),
                ).model_dump())

                # Fire watchlist alert if applicable
                if session.person and session.person.watchlist_status != "none":
                    await self._fire_alert(
                        db=db,
                        session=session,
                        sighting=sighting,
                    )

        # ── Step 4: route reconstruction ──────────────────────────────────
        await self._push_progress(session, db, 70, "Reconstructing route...")

        route_service = get_route_service()
        fused_route = route_service.reconstruct_route(camera_matches)
        route_confidence = route_service.compute_route_confidence(fused_route)

        # ── Step 5: persist route steps and update sighting fusion scores ─
        await self._push_progress(session, db, 85, "Persisting route...")

        # Load the sightings we just saved so we can update fusion scores
        # and create route step records
        from sqlalchemy import select
        existing_sightings_result = await db.execute(
            select(Sighting).where(Sighting.session_id == session_id)
        )
        existing_sightings: dict[str, Sighting] = {
            s.camera_id: s for s in existing_sightings_result.scalars().all()
        }

        for step_order, fused in enumerate(fused_route):
            cam_id = fused.match.camera_id
            sighting = existing_sightings.get(cam_id)
            if sighting is None:
                continue

            # Update fusion scores on the sighting record
            sighting.spatial_score = round(fused.spatial, 6)
            sighting.temporal_score = round(fused.temporal, 6)
            sighting.fusion_score = round(fused.fusion, 6)

            # Create route step
            route_step = RouteStep(
                session_id=session_id,
                step_order=step_order,
                sighting_id=sighting.id,
            )
            db.add(route_step)

        await db.commit()

        # ── Step 6: build route output and push completion event ──────────
        await self._push_progress(session, db, 95, "Finalising route...")

        # Reload sightings with updated fusion scores
        sightings_result = await db.execute(
            select(Sighting).where(Sighting.session_id == session_id)
        )
        all_sightings = sightings_result.scalars().all()

        route_steps_result = await db.execute(
            select(RouteStep)
            .where(RouteStep.session_id == session_id)
            .order_by(RouteStep.step_order)
        )
        all_steps = route_steps_result.scalars().all()

        # Build step objects for the WebSocket event
        step_outs: list[RouteStepOut] = []
        for step in all_steps:
            s = existing_sightings.get(
                next((fs.match.camera_id for fs in fused_route if fs.match.camera_id in existing_sightings), None)
            )
            # find matching sighting by id
            matching_sighting = next(
                (si for si in all_sightings if si.id == step.sighting_id), None
            )
            if matching_sighting is None:
                continue
            cam_data = route_service.get_graph().get("cameras", {}).get(
                matching_sighting.camera_id, {}
            )
            step_outs.append(RouteStepOut(
                step_order=step.step_order,
                camera_id=matching_sighting.camera_id,
                camera_location=cam_data.get("location", matching_sighting.camera_id),
                track_id=matching_sighting.track_id,
                timestamp=matching_sighting.first_seen,
                confidence=matching_sighting.best_confidence,
                fusion_score=matching_sighting.fusion_score,
                crop_path=matching_sighting.crop_path,
            ))

        route_out = QueryRouteOut(
            session_id=session_id,
            status="done",
            steps=step_outs,
            sightings=[SightingOut.model_validate(s) for s in all_sightings],
            total_cameras_matched=len(all_sightings),
            route_confidence=route_confidence,
        )

        # ── Step 7: mark complete ─────────────────────────────────────────
        session.status = "done"
        session.progress_pct = 100
        session.completed_at = datetime.utcnow()
        await db.commit()

        await self._ws.broadcast(WsRouteComplete(
            session_id=session_id,
            route=route_out,
        ).model_dump())

        logger.info(
            f"[PipelineService] Session {session_id} complete. "
            f"Cameras matched: {len(all_sightings)} / {len(galleries)}. "
            f"Route confidence: {route_confidence:.4f}"
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    async def _resolve_query_embedding(
        self, session: QuerySession
    ) -> Optional[list[float]]:
        """
        Get the 512-D query embedding:
          - If session has a registered person with an enrolled embedding → use it.
          - If session has a query image path → embed it on the fly.
        """
        if session.person and session.person.embedding_vector:
            logger.info(
                f"[PipelineService] Using enrolled embedding for person {session.person_id}"
            )
            return session.person.embedding_vector

        if session.query_image_path:
            img_path = Path(session.query_image_path)
            if not img_path.exists():
                raise FileNotFoundError(f"Query image not found: {img_path}")
            logger.info(f"[PipelineService] Embedding query image: {img_path.name}")
            # Run in executor so the async loop isn't blocked
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, embedding_service.embed_single, img_path
            )
            return embedding

        return None

    async def _persist_sighting(
        self,
        db: AsyncSession,
        session_id: int,
        match: CameraMatch,
        spatial: float,
        temporal: float,
        fusion: float,
    ) -> Sighting:
        """Create and add a Sighting ORM record (does not commit)."""
        sighting = Sighting(
            session_id=session_id,
            camera_id=match.camera_id,
            track_id=match.track_id,
            first_seen=match.first_seen,
            last_seen=match.last_seen,
            best_confidence=match.best_confidence,
            mean_confidence=match.mean_confidence,
            crop_path=match.best_crop_path,
            appearance_score=match.appearance_score,
            spatial_score=round(spatial, 6),
            temporal_score=round(temporal, 6),
            fusion_score=round(fusion, 6),
        )
        db.add(sighting)
        await db.flush()   # get sighting.id without committing
        return sighting

    async def _fire_alert(
        self,
        db: AsyncSession,
        session: QuerySession,
        sighting: Sighting,
    ) -> None:
        """Create an alert for a watchlist match and push it via WebSocket."""
        person = session.person
        status_label = person.watchlist_status.upper()
        severity = "HIGH" if status_label in ("MISSING", "SUSPECT") else "MEDIUM"

        alert = Alert(
            session_id=session.id,
            person_id=person.id,
            sighting_id=sighting.id,
            severity=severity,
            title=f"Watchlist Match: {person.name}",
            message=(
                f"{status_label} — {person.name} detected at "
                f"Camera {sighting.camera_id} "
                f"(confidence: {sighting.best_confidence:.1f})"
            ),
        )
        db.add(alert)
        await db.flush()

        await self._ws.broadcast(WsAlert(
            alert=AlertOut.model_validate(alert),
        ).model_dump())

        logger.warning(
            f"[PipelineService] [{severity}] Watchlist alert: "
            f"{person.name} @ {sighting.camera_id}"
        )

    async def _push_progress(
        self,
        session: QuerySession,
        db: AsyncSession,
        pct: int,
        message: str,
    ) -> None:
        """Update session progress in DB and push WebSocket event."""
        session.progress_pct = pct
        await db.commit()
        await self._ws.broadcast(WsQueryProgress(
            session_id=session.id,
            progress_pct=pct,
            message=message,
        ).model_dump())
