"""
pipeline_service.py — PipelineService
End-to-end query pipeline orchestrator for the unified Trace system.

Handles one query session from start to finish:
  1. Load the query body embedding (OSNet / SOLIDER — from person or photo).
  2. Load the query face embedding (SCRFD + ArcFace) if weights are configured.
  3. Load the query KPR part embeddings if weights are configured.
  4. For each MVP camera: load body, face, and KPR gallery embeddings.
  5. Run MatchingService.search_camera_top_k() per camera with all available
     signals → triple fusion (body + face + KPR) when all are present.
  6. Run RouteService.reconstruct_route() → picks the track per camera that
     maximises fused (appearance+spatial+temporal) score given route context.
  7. Persist one Sighting per route step and write RouteStep records.
  8. Fire watchlist alerts if the queried person is on the watchlist.
  9. Push WebSocket progress events throughout.

Signal availability:
  - Body embedding : always available (OSNet).
  - Face embedding : available when face_det_model + face_rec_model are set in
                     config AND the query image contains a detectable face.
  - KPR embedding  : available when kpr_weights_path + kpr_config_path are set
                     AND the KPR repo is importable.
  When a signal is unavailable it is silently excluded from fusion —
  existing body-only behaviour is fully preserved.

The pipeline runs in a background asyncio task so the HTTP endpoint that
submitted the query can return immediately with the session ID.
"""

from __future__ import annotations

import asyncio
import json
import sys
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
# Body gallery helpers
# ---------------------------------------------------------------------------

def _gallery_path_for_camera(camera_id: str) -> Optional[Path]:
    """Return path to body embeddings JSON for a camera, or None if absent."""
    path = settings.embeddings_dir / f"embeddings_{camera_id}.json"
    return path if path.exists() else None


def _load_galleries(camera_ids: list[str]) -> dict[str, list[dict]]:
    """Load body embedding gallery files for all given cameras."""
    galleries: dict[str, list[dict]] = {}
    for cam_id in camera_ids:
        path = _gallery_path_for_camera(cam_id)
        if path is None:
            logger.warning(
                f"[PipelineService] No body gallery for camera {cam_id} — skipping"
            )
            continue
        try:
            with open(path) as f:
                galleries[cam_id] = json.load(f)
            logger.info(
                f"[PipelineService] Body gallery {cam_id}: {len(galleries[cam_id])} crops"
            )
        except Exception as e:
            logger.error(f"[PipelineService] Failed to load body gallery {path}: {e}")
    return galleries


# ---------------------------------------------------------------------------
# Face gallery helpers
# ---------------------------------------------------------------------------

def _load_face_galleries(camera_ids: list[str]) -> dict[str, list[dict]]:
    """
    Load face embedding gallery files for all given cameras.
    Returns empty dict if no face gallery files exist (non-fatal).
    """
    galleries: dict[str, list[dict]] = {}
    for cam_id in camera_ids:
        path = settings.embeddings_dir / f"face_embeddings_{cam_id}.json"
        if not path.exists():
            logger.debug(
                f"[PipelineService] No face gallery for {cam_id} — face signal skipped"
            )
            continue
        try:
            with open(path) as f:
                galleries[cam_id] = json.load(f)
            logger.info(
                f"[PipelineService] Face gallery {cam_id}: {len(galleries[cam_id])} crops"
            )
        except Exception as e:
            logger.warning(
                f"[PipelineService] Failed to load face gallery {path}: {e}"
            )
    return galleries


def _embed_query_face(img_path: Path) -> Optional[list[float]]:
    """
    Run SCRFD face detection + ArcFace on a single query image.
    Returns 512-dim L2-normalised embedding, or None if no face detected
    or if the ONNX weights are not configured.

    Runs synchronously — call inside run_in_executor.
    """
    det_model = settings.face_det_model
    rec_model = settings.face_rec_model

    if not det_model or not rec_model:
        logger.debug("[PipelineService] Face weights not configured — skipping face query")
        return None

    det_path = Path(det_model)
    rec_path = Path(rec_model)

    if not det_path.exists() or not rec_path.exists():
        missing = [str(p) for p in (det_path, rec_path) if not p.exists()]
        logger.warning(
            f"[PipelineService] Face weight files missing: {missing} — skipping face query"
        )
        return None

    try:
        # Lazy import — keeps startup fast when onnxruntime is not installed
        _repo_root = Path(__file__).resolve().parents[3]
        if str(_repo_root) not in sys.path:
            sys.path.insert(0, str(_repo_root))

        from ai_pipeline.reid.face_embed import (  # noqa: PLC0415
            SCRFDDetector,
            ArcFaceRecogniser,
            process_crop,
        )

        detector   = SCRFDDetector(det_path, conf_thresh=settings.face_det_threshold)
        recogniser = ArcFaceRecogniser(rec_path)

        # Re-use process_crop with a minimal metadata record
        dummy_meta = {
            "crop_path": str(img_path),
            "camera_id": "query",
            "track_id":  -1,
            "frame":      0,
            "timestamp":  None,
            "bbox":       None,
        }
        result = process_crop(img_path, dummy_meta, detector, recogniser, min_face_size=20)

        if result.get("face_detected") and result.get("face_embedding"):
            logger.info("[PipelineService] Face detected in query image")
            return result["face_embedding"]

        logger.info("[PipelineService] No face detected in query image — face signal skipped")
        return None

    except Exception as exc:
        logger.warning(f"[PipelineService] Face query embedding failed (non-fatal): {exc}")
        return None


# ---------------------------------------------------------------------------
# KPR gallery helpers
# ---------------------------------------------------------------------------

def _load_kpr_galleries(camera_ids: list[str]) -> dict[str, list[dict]]:
    """
    Load KPR embedding gallery files for all given cameras.
    Returns empty dict if no KPR gallery files exist (non-fatal).
    """
    galleries: dict[str, list[dict]] = {}
    for cam_id in camera_ids:
        path = settings.embeddings_dir / f"kpr_embeddings_{cam_id}.json"
        if not path.exists():
            logger.debug(
                f"[PipelineService] No KPR gallery for {cam_id} — KPR signal skipped"
            )
            continue
        try:
            with open(path) as f:
                galleries[cam_id] = json.load(f)
            logger.info(
                f"[PipelineService] KPR gallery {cam_id}: {len(galleries[cam_id])} crops"
            )
        except Exception as e:
            logger.warning(
                f"[PipelineService] Failed to load KPR gallery {path}: {e}"
            )
    return galleries


def _embed_query_kpr(img_path: Path) -> Optional[dict]:
    """
    Run KPR on a single query image to produce holistic + part embeddings.
    Returns a dict with keys: holistic_embedding, part_embeddings,
    part_visibility — or None if KPR weights are not configured or import fails.

    Runs synchronously — call inside run_in_executor.
    """
    kpr_weights = settings.kpr_weights_path
    kpr_config  = settings.kpr_config_path

    if not kpr_weights or not kpr_config:
        logger.debug("[PipelineService] KPR weights not configured — skipping KPR query")
        return None

    kpr_w = Path(kpr_weights)
    kpr_c = Path(kpr_config)

    if not kpr_w.exists() or not kpr_c.exists():
        missing = [str(p) for p in (kpr_w, kpr_c) if not p.exists()]
        logger.warning(
            f"[PipelineService] KPR weight files missing: {missing} — skipping KPR query"
        )
        return None

    try:
        import torch  # noqa: PLC0415

        _repo_root = Path(__file__).resolve().parents[3]
        if str(_repo_root) not in sys.path:
            sys.path.insert(0, str(_repo_root))

        from ai_pipeline.reid.embed_kpr import (  # noqa: PLC0415
            load_kpr_model,
            probe_kpr_output,
            validate_and_preprocess,
            _parse_kpr_output,
        )
        from torchreid.utils.tools import extract_test_embeddings  # noqa: PLC0415

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Infer kpr_root from config path (walk up to find setup.py / torchreid dir)
        kpr_root = kpr_c.resolve().parent
        for parent in kpr_c.resolve().parents:
            if (parent / "setup.py").exists() or (parent / "torchreid").exists():
                kpr_root = parent
                break

        extractor = load_kpr_model(kpr_w, kpr_c, kpr_root, device)
        num_slots, holistic_idx, num_parts, part_dim = probe_kpr_output(extractor, device)

        # Preprocess the query image
        tensor, failure = validate_and_preprocess(
            img_path, str(img_path), min_crop_width=32, min_crop_height=64
        )
        if failure is not None:
            logger.warning(
                f"[PipelineService] KPR query preprocess failed: {failure.reason}"
            )
            return None

        # Run KPR forward pass on single image (batch size 1)
        batch = tensor.unsqueeze(0).to(device)  # [1, C, H, W]
        with torch.no_grad():
            embeddings_batch, vis_scores_batch = extract_test_embeddings(
                extractor, batch
            )

        holistic_emb, part_embs, part_vis = _parse_kpr_output(
            embeddings_batch, vis_scores_batch,
            batch_idx_in_batch=0,
            holistic_idx=holistic_idx,
        )

        logger.info(
            f"[PipelineService] KPR query embedded: "
            f"{num_parts} parts, holistic_dim={len(holistic_emb)}"
        )
        return {
            "holistic_embedding": holistic_emb,
            "part_embeddings":    part_embs,
            "part_visibility":    part_vis,
        }

    except Exception as exc:
        logger.warning(f"[PipelineService] KPR query embedding failed (non-fatal): {exc}")
        return None


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
        """
        logger.info(f"[PipelineService] Session {session_id} starting")

        async with async_session() as db:
            session = await db.get(QuerySession, session_id)
            if session is None:
                logger.error(f"[PipelineService] Session {session_id} not found in DB")
                return

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
        loop = asyncio.get_event_loop()

        # ── Step 1: body embedding ────────────────────────────────────────
        await self._push_progress(session, db, 5, "Loading body embedding...")
        query_embedding = await self._resolve_query_embedding(session)
        if query_embedding is None:
            raise ValueError(
                "No embedding available. Enroll a reference photo for this person "
                "or upload a query image."
            )

        # ── Step 2: face embedding for query image (optional) ─────────────
        await self._push_progress(session, db, 10, "Extracting face from query image...")
        query_face_embedding: Optional[list[float]] = None
        img_path = self._resolve_query_image_path(session)
        if img_path is not None:
            query_face_embedding = await loop.run_in_executor(
                None, _embed_query_face, img_path
            )
            if query_face_embedding:
                logger.info("[PipelineService] Face signal: active")
            else:
                logger.info("[PipelineService] Face signal: inactive (no face / no weights)")
        else:
            logger.info(
                "[PipelineService] Face signal: skipped "
                "(registered-person query — no image available)"
            )

        # ── Step 3: KPR embedding for query image (optional) ──────────────
        await self._push_progress(session, db, 15, "Extracting KPR parts from query image...")
        query_kpr: Optional[dict] = None
        if img_path is not None:
            query_kpr = await loop.run_in_executor(
                None, _embed_query_kpr, img_path
            )
            if query_kpr:
                logger.info("[PipelineService] KPR signal: active")
            else:
                logger.info("[PipelineService] KPR signal: inactive (no weights / import error)")
        else:
            logger.info(
                "[PipelineService] KPR signal: skipped "
                "(registered-person query — no image available)"
            )

        # ── Step 4: load all galleries ────────────────────────────────────
        await self._push_progress(session, db, 20, "Loading camera galleries...")
        mvp_cameras = ["C01", "C02", "C03"]
        body_galleries = _load_galleries(mvp_cameras)

        if not body_galleries:
            raise ValueError(
                "No gallery embeddings found for any MVP camera. "
                "Upload a video first to generate the galleries."
            )

        # Load face + KPR galleries only if we have query signals for them
        face_galleries: dict[str, list[dict]] = {}
        if query_face_embedding is not None:
            face_galleries = _load_face_galleries(mvp_cameras)
            if face_galleries:
                logger.info(
                    f"[PipelineService] Face galleries loaded for: "
                    f"{list(face_galleries.keys())}"
                )
            else:
                logger.info(
                    "[PipelineService] Face galleries not found — "
                    "face signal deactivated for this session"
                )
                query_face_embedding = None  # deactivate — nothing to match against

        kpr_galleries: dict[str, list[dict]] = {}
        if query_kpr is not None:
            kpr_galleries = _load_kpr_galleries(mvp_cameras)
            if kpr_galleries:
                logger.info(
                    f"[PipelineService] KPR galleries loaded for: "
                    f"{list(kpr_galleries.keys())}"
                )
            else:
                logger.info(
                    "[PipelineService] KPR galleries not found — "
                    "KPR signal deactivated for this session"
                )
                query_kpr = None  # deactivate — nothing to match against

        # Log active fusion mode
        active_signals = ["body"]
        if query_face_embedding is not None:
            active_signals.append("face")
        if query_kpr is not None:
            active_signals.append("KPR")
        logger.info(
            f"[PipelineService] Active fusion signals: {' + '.join(active_signals)}"
        )

        # ── Step 5: per-camera matching ───────────────────────────────────
        await self._push_progress(session, db, 30, "Running cross-camera matching...")
        camera_candidates: dict[str, list[CameraMatch]] = {}
        n_cameras = len(body_galleries)

        for i, (cam_id, gallery) in enumerate(body_galleries.items()):
            pct = 30 + int((i / n_cameras) * 40)
            await self._push_progress(
                session, db, pct,
                f"[{i+1}/{n_cameras}] Matching {cam_id} "
                f"({'+'.join(active_signals)})..."
            )

            candidates = _matching_service.search_camera_top_k(
                query_embedding=query_embedding,
                gallery=gallery,
                camera_id=cam_id,
                candidate_tracks=3,
                # face signal — pass only when gallery exists for this camera
                face_gallery=face_galleries.get(cam_id),
                query_face_embedding=query_face_embedding,
                # KPR signal — pass only when gallery exists for this camera
                kpr_gallery=kpr_galleries.get(cam_id),
                query_kpr=query_kpr,
            )
            if candidates:
                camera_candidates[cam_id] = candidates

        # ── Step 6: route reconstruction ──────────────────────────────────
        await self._push_progress(session, db, 75, "Reconstructing route...")
        route_service = get_route_service()
        fused_route   = route_service.reconstruct_route(camera_candidates)
        route_confidence = route_service.compute_route_confidence(fused_route)

        for fused in fused_route:
            sighting = await self._persist_sighting(
                db=db,
                session_id=session_id,
                match=fused.match,
                spatial=1.0,
                temporal=1.0,
                fusion=fused.match.appearance_score,
            )
            await db.commit()

            await self._ws.broadcast(WsSightingFound(
                session_id=session_id,
                sighting=SightingOut.model_validate(sighting),
            ).model_dump())

            if session.person and session.person.watchlist_status != "none":
                await self._fire_alert(db=db, session=session, sighting=sighting)

        # ── Step 7: persist route steps + update fusion scores ────────────
        await self._push_progress(session, db, 88, "Persisting route...")
        from sqlalchemy import select  # noqa: PLC0415

        existing_sightings_result = await db.execute(
            select(Sighting).where(Sighting.session_id == session_id)
        )
        existing_sightings: dict[str, Sighting] = {
            s.camera_id: s for s in existing_sightings_result.scalars().all()
        }

        for step_order, fused in enumerate(fused_route):
            cam_id  = fused.match.camera_id
            sighting = existing_sightings.get(cam_id)
            if sighting is None:
                continue
            sighting.spatial_score  = round(fused.spatial, 6)
            sighting.temporal_score = round(fused.temporal, 6)
            sighting.fusion_score   = round(fused.fusion,   6)
            db.add(RouteStep(
                session_id=session_id,
                step_order=step_order,
                sighting_id=sighting.id,
            ))

        await db.commit()

        # ── Step 8: build route output and push completion event ──────────
        await self._push_progress(session, db, 96, "Finalising route...")

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

        step_outs: list[RouteStepOut] = []
        for step in all_steps:
            ms = next((s for s in all_sightings if s.id == step.sighting_id), None)
            if ms is None:
                continue
            cam_data = route_service.get_graph().get("cameras", {}).get(ms.camera_id, {})
            step_outs.append(RouteStepOut(
                step_order=step.step_order,
                camera_id=ms.camera_id,
                camera_location=cam_data.get("location", ms.camera_id),
                track_id=ms.track_id,
                timestamp=ms.first_seen,
                confidence=ms.best_confidence,
                fusion_score=ms.fusion_score,
                crop_path=ms.crop_path,
            ))

        route_out = QueryRouteOut(
            session_id=session_id,
            status="done",
            steps=step_outs,
            sightings=[SightingOut.model_validate(s) for s in all_sightings],
            total_cameras_matched=len(all_sightings),
            route_confidence=route_confidence,
        )

        # ── Step 9: mark complete ──────────────────────────────────────────
        session.status       = "done"
        session.progress_pct = 100
        session.completed_at = datetime.utcnow()
        await db.commit()

        await self._ws.broadcast(WsRouteComplete(
            session_id=session_id,
            route=route_out,
        ).model_dump())

        logger.info(
            f"[PipelineService] Session {session_id} complete — "
            f"signals={'+'.join(active_signals)} "
            f"cameras={len(all_sightings)}/{len(body_galleries)} "
            f"confidence={route_confidence:.4f}"
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _resolve_query_image_path(
        self, session: QuerySession
    ) -> Optional[Path]:
        """
        Return the Path to the query image if one exists, else None.
        Used to run face and KPR extraction on the raw image.
        Registered-person queries (no image) return None — face/KPR
        are not available in that mode without an image to process.
        """
        if session.query_image_path:
            p = Path(session.query_image_path)
            return p if p.exists() else None
        return None

    async def _resolve_query_embedding(
        self, session: QuerySession
    ) -> Optional[list[float]]:
        """
        Get the body query embedding:
          - Registered person with enrolled embedding → use stored vector.
          - Uploaded query image → embed on the fly with OSNet.
        """
        if session.person and session.person.embedding_vector:
            logger.info(
                f"[PipelineService] Using enrolled OSNet embedding "
                f"for person {session.person_id}"
            )
            return session.person.embedding_vector

        if session.query_image_path:
            img_path = Path(session.query_image_path)
            if not img_path.exists():
                raise FileNotFoundError(f"Query image not found: {img_path}")
            logger.info(f"[PipelineService] Embedding query image: {img_path.name}")
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, embedding_service.embed_single, img_path
            )

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
        """Create and flush a Sighting ORM record (does not commit)."""
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
        await db.flush()
        return sighting

    async def _fire_alert(
        self,
        db: AsyncSession,
        session: QuerySession,
        sighting: Sighting,
    ) -> None:
        """Create a watchlist alert and push it via WebSocket."""
        person       = session.person
        status_label = person.watchlist_status.upper()
        severity     = "HIGH" if status_label in ("MISSING", "SUSPECT") else "MEDIUM"

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
