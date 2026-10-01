"""
pipeline_service.py — PipelineService
End-to-end query pipeline orchestrator for the unified Trace system.

Handles one query session from start to finish:
  1.  Load query body embedding (OSNet — from person or uploaded image).
  2.  Run SCRFD + ArcFace on the query image → query_face_embedding + confidence.
  3.  Run KPR on the query image → query_kpr.
  4.  Load body / face / KPR gallery files per camera.
  5.  Run MatchingService.search_camera_top_k() → per-track evidence hierarchy:
        face veto → KPR part-aware sim → body sim
  6.  Run RouteService.reconstruct_route() over CONFIDENT_MATCH candidates only.
  7.  Persist Sighting records (with match_status + veto details).
  8.  Compute overall match_decision and top_candidates for human review.
  9.  Fire watchlist alerts for CONFIDENT_MATCH sightings on the watchlist.
  10. Push WebSocket progress events throughout.

Signal availability:
  body    : always (OSNet gallery embeddings_<cam>.json).
  face    : when face_det_model + face_rec_model are set AND query has a face
            AND face_embeddings_<cam>.json gallery exists.
  KPR     : when kpr_weights_path + kpr_config_path are set AND query succeeds
            AND kpr_embeddings_<cam>.json gallery exists.
  Missing signals are silently excluded — body-only behaviour is preserved.

Identity decision logging
--------------------------
At session start we log:
  BODY MODEL: OSNet / SOLIDER
  BODY DIMENSION: 512 / 768
  FACE: ENABLED / DISABLED
  KPR: ENABLED / DISABLED
  ACTIVE THRESHOLD: value
  SOFT FLOOR: value

For every candidate track we log the full evidence breakdown.
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
    CandidateTrackOut,
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
from backend.app.services.matching_service import (
    CameraMatch,
    MatchingService,
    MatchStatus,
)
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
# Gallery loaders
# ---------------------------------------------------------------------------

def _gallery_path_for_camera(camera_id: str) -> Optional[Path]:
    path = settings.embeddings_dir / f"embeddings_{camera_id}.json"
    return path if path.exists() else None


def _load_galleries(camera_ids: list[str]) -> dict[str, list[dict]]:
    galleries: dict[str, list[dict]] = {}
    for cam_id in camera_ids:
        path = _gallery_path_for_camera(cam_id)
        if path is None:
            logger.warning(f"[PipelineService] No body gallery for {cam_id} — skipping")
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


def _load_face_galleries(camera_ids: list[str]) -> dict[str, list[dict]]:
    galleries: dict[str, list[dict]] = {}
    for cam_id in camera_ids:
        path = settings.embeddings_dir / f"face_embeddings_{cam_id}.json"
        if not path.exists():
            logger.debug(f"[PipelineService] No face gallery for {cam_id}")
            continue
        try:
            with open(path) as f:
                galleries[cam_id] = json.load(f)
            logger.info(
                f"[PipelineService] Face gallery {cam_id}: {len(galleries[cam_id])} crops"
            )
        except Exception as e:
            logger.warning(f"[PipelineService] Failed to load face gallery {path}: {e}")
    return galleries


def _load_kpr_galleries(camera_ids: list[str]) -> dict[str, list[dict]]:
    galleries: dict[str, list[dict]] = {}
    for cam_id in camera_ids:
        path = settings.embeddings_dir / f"kpr_embeddings_{cam_id}.json"
        if not path.exists():
            logger.debug(f"[PipelineService] No KPR gallery for {cam_id}")
            continue
        try:
            with open(path) as f:
                galleries[cam_id] = json.load(f)
            logger.info(
                f"[PipelineService] KPR gallery {cam_id}: {len(galleries[cam_id])} crops"
            )
        except Exception as e:
            logger.warning(f"[PipelineService] Failed to load KPR gallery {path}: {e}")
    return galleries


# ---------------------------------------------------------------------------
# Query-image embedding helpers
# ---------------------------------------------------------------------------

def _embed_query_face(img_path: Path) -> tuple[Optional[list[float]], Optional[float]]:
    """
    Run SCRFD + ArcFace on the query image.

    Returns (embedding, detection_confidence) or (None, None).
    detection_confidence is the SCRFD score for the best detected face —
    passed to search_camera_top_k() so it can decide whether the query
    face is reliable enough to trigger the face veto.
    """
    det_model = settings.face_det_model
    rec_model = settings.face_rec_model

    if not det_model or not rec_model:
        logger.debug("[PipelineService] Face weights not configured — skipping face query")
        return None, None

    det_path = Path(det_model)
    rec_path = Path(rec_model)

    if not det_path.exists() or not rec_path.exists():
        missing = [str(p) for p in (det_path, rec_path) if not p.exists()]
        logger.warning(
            f"[PipelineService] Face weight files missing: {missing} — skipping face query"
        )
        return None, None

    try:
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
            det_conf = result.get("face_confidence")  # SCRFD score
            logger.info(
                f"[PipelineService] Query face detected "
                f"(SCRFD confidence={det_conf:.3f if det_conf else 'N/A'})"
            )
            return result["face_embedding"], det_conf

        logger.info("[PipelineService] No face detected in query image — face signal skipped")
        return None, None

    except Exception as exc:
        logger.warning(f"[PipelineService] Face query embedding failed (non-fatal): {exc}")
        return None, None


def _embed_query_kpr(img_path: Path) -> Optional[dict]:
    """
    Run KPR on the query image.
    Returns {holistic_embedding, part_embeddings, part_visibility} or None.
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
        kpr_root = kpr_c.resolve().parent
        for parent in kpr_c.resolve().parents:
            if (parent / "setup.py").exists() or (parent / "torchreid").exists():
                kpr_root = parent
                break

        extractor = load_kpr_model(kpr_w, kpr_c, kpr_root, device)
        num_slots, holistic_idx, num_parts, part_dim = probe_kpr_output(extractor, device)

        tensor, failure = validate_and_preprocess(
            img_path, str(img_path), min_crop_width=32, min_crop_height=64
        )
        if failure is not None:
            logger.warning(f"[PipelineService] KPR query preprocess failed: {failure.reason}")
            return None

        batch = tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            embeddings_batch, vis_scores_batch = extract_test_embeddings(extractor, batch)

        holistic_emb, part_embs, part_vis = _parse_kpr_output(
            embeddings_batch, vis_scores_batch, 0, holistic_idx
        )
        logger.info(
            f"[PipelineService] KPR query embedded: {num_parts} parts, "
            f"holistic_dim={len(holistic_emb)}"
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
# Match decision helpers
# ---------------------------------------------------------------------------

def _compute_match_decision(
    all_candidates: list[CameraMatch],
) -> str:
    """
    Derive the overall session match_decision from all camera candidates.

    Returns one of:
        CONFIDENT_MATCH        — ≥1 camera has a CONFIDENT_MATCH candidate
        POSSIBLE_MATCH_REVIEW  — ≥1 candidate above soft_floor, none confident
        NO_CONFIDENT_MATCH     — no candidate above soft_floor
    """
    if not all_candidates:
        return MatchStatus.NO_CONFIDENT_MATCH.value

    statuses = {m.match_status for m in all_candidates}
    if MatchStatus.CONFIDENT_MATCH in statuses:
        return MatchStatus.CONFIDENT_MATCH.value
    if MatchStatus.POSSIBLE_MATCH_REVIEW in statuses or MatchStatus.FACE_MISMATCH in statuses:
        return MatchStatus.POSSIBLE_MATCH_REVIEW.value
    return MatchStatus.NO_CONFIDENT_MATCH.value


def _build_top_candidates(
    camera_candidates: dict[str, list[CameraMatch]],
    n: int = 3,
) -> list[CandidateTrackOut]:
    """
    Collect up to *n* unique (camera_id, track_id) candidates across all cameras,
    sorted by fused_score descending.

    One candidate per unique track — never three crops from the same track.
    All statuses are included (CONFIDENT_MATCH, POSSIBLE_MATCH_REVIEW,
    FACE_MISMATCH) so an operator can visually verify the best options.
    """
    seen: set[tuple[str, int]] = set()
    flat: list[CameraMatch] = []
    for cam_candidates in camera_candidates.values():
        for m in cam_candidates:
            key = (m.camera_id, m.track_id)
            if key not in seen:
                seen.add(key)
                flat.append(m)

    flat.sort(key=lambda m: m.fused_score, reverse=True)

    return [
        CandidateTrackOut(
            camera_id=m.camera_id,
            track_id=m.track_id,
            best_confidence=m.best_confidence,
            appearance_score=m.appearance_score,
            fused_score=m.fused_score,
            match_status=m.match_status.value,
            face_veto_applied=m.face_veto_applied,
            face_sim=m.face_sim,
            face_coverage=m.face_coverage,
            kpr_sim=m.kpr_sim,
            first_seen=m.first_seen,
            crop_path=m.best_crop_path,
            active_backbone=m.active_backbone,
        )
        for m in flat[:n]
    ]


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
        logger.info(f"[PipelineService] Session {session_id} starting")
        async with async_session() as db:
            session = await db.get(QuerySession, session_id)
            if session is None:
                logger.error(f"[PipelineService] Session {session_id} not found in DB")
                return

            session.status       = "running"
            session.progress_pct = 0
            await db.commit()

            try:
                await self._execute(session, db)
            except Exception as e:
                logger.exception(f"[PipelineService] Session {session_id} failed: {e}")
                session.status        = "failed"
                session.error_message = str(e)
                session.completed_at  = datetime.utcnow()
                await db.commit()
                await self._ws.broadcast(WsError(
                    session_id=session_id, message=str(e)
                ).model_dump())

    async def _execute(self, session: QuerySession, db: AsyncSession) -> None:
        session_id = session.id
        loop = asyncio.get_event_loop()

        # ── Step 1: body embedding ────────────────────────────────────────
        await self._push_progress(session, db, 5, "Loading body embedding...")
        query_embedding = await self._resolve_query_embedding(session)
        if query_embedding is None:
            raise ValueError(
                "No embedding available. Enroll a reference photo or upload an image."
            )

        # ── Step 2: face embedding (optional) ────────────────────────────
        await self._push_progress(session, db, 10, "Extracting face from query image...")
        query_face_embedding: Optional[list[float]] = None
        query_face_confidence: Optional[float] = None
        img_path = self._resolve_query_image_path(session)

        if img_path is not None:
            query_face_embedding, query_face_confidence = await loop.run_in_executor(
                None, _embed_query_face, img_path
            )
        else:
            logger.info("[PipelineService] Face signal: skipped (no image for registered-person query)")

        # ── Step 3: KPR embedding (optional) ─────────────────────────────
        await self._push_progress(session, db, 15, "Extracting KPR parts from query image...")
        query_kpr: Optional[dict] = None
        if img_path is not None:
            query_kpr = await loop.run_in_executor(None, _embed_query_kpr, img_path)
        else:
            logger.info("[PipelineService] KPR signal: skipped (no image for registered-person query)")

        # ── Step 4: load galleries ────────────────────────────────────────
        await self._push_progress(session, db, 20, "Loading camera galleries...")
        mvp_cameras   = ["C01", "C02", "C03"]
        body_galleries = _load_galleries(mvp_cameras)

        if not body_galleries:
            raise ValueError(
                "No gallery embeddings found for any MVP camera. "
                "Upload a video first to generate the galleries."
            )

        # Infer active backbone from first gallery record
        first_gallery = next(iter(body_galleries.values()), [])
        emb_dim = len(first_gallery[0]["embedding"]) if first_gallery else 512
        from ai_pipeline.reid.confidence_scaling import get_backbone_config  # noqa: PLC0415
        backbone_cfg = get_backbone_config(emb_dim)

        # Load face + KPR galleries only when query signals are active
        face_galleries: dict[str, list[dict]] = {}
        if query_face_embedding is not None:
            face_galleries = _load_face_galleries(mvp_cameras)
            if not face_galleries:
                logger.info(
                    "[PipelineService] Face galleries not found — "
                    "face signal deactivated (run face_embed.py to generate them)"
                )
                query_face_embedding  = None
                query_face_confidence = None

        kpr_galleries: dict[str, list[dict]] = {}
        if query_kpr is not None:
            kpr_galleries = _load_kpr_galleries(mvp_cameras)
            if not kpr_galleries:
                logger.info(
                    "[PipelineService] KPR galleries not found — "
                    "KPR signal deactivated (run embed_kpr.py to generate them)"
                )
                query_kpr = None

        # Log active configuration clearly
        active_signals = ["body"]
        if query_face_embedding is not None: active_signals.append("face")
        if query_kpr is not None:            active_signals.append("KPR")

        face_reliable = (
            query_face_confidence is not None
            and query_face_confidence >= settings.face_min_query_confidence
        )
        logger.info(
            f"[PipelineService] ══ ACTIVE CONFIGURATION ══ "
            f"BODY MODEL: {backbone_cfg.name} | "
            f"BODY DIMENSION: {emb_dim} | "
            f"FACE: {'ENABLED' if 'face' in active_signals else 'DISABLED'} | "
            f"KPR: {'ENABLED' if 'KPR' in active_signals else 'DISABLED'} | "
            f"FACE WEIGHTS: {settings.face_det_model or 'not set'} | "
            f"KPR WEIGHTS: {settings.kpr_weights_path or 'not set'} | "
            f"ACTIVE THRESHOLD: {backbone_cfg.no_match_threshold} | "
            f"SOFT FLOOR: {backbone_cfg.soft_floor} | "
            f"QUERY FACE RELIABLE: {face_reliable} "
            f"(conf={query_face_confidence}, "
            f"min={settings.face_min_query_confidence})"
        )

        # ── Step 5: per-camera matching ───────────────────────────────────
        await self._push_progress(session, db, 30, "Running cross-camera matching...")
        camera_candidates: dict[str, list[CameraMatch]] = {}
        n_cameras = len(body_galleries)

        for i, (cam_id, gallery) in enumerate(body_galleries.items()):
            pct = 30 + int((i / n_cameras) * 40)
            await self._push_progress(
                session, db, pct,
                f"[{i+1}/{n_cameras}] Matching {cam_id} ({'+'.join(active_signals)})..."
            )

            candidates = _matching_service.search_camera_top_k(
                query_embedding=query_embedding,
                gallery=gallery,
                camera_id=cam_id,
                candidate_tracks=3,
                face_gallery=face_galleries.get(cam_id),
                query_face_embedding=query_face_embedding,
                query_face_confidence=query_face_confidence,
                kpr_gallery=kpr_galleries.get(cam_id),
                query_kpr=query_kpr,
            )
            if candidates:
                camera_candidates[cam_id] = candidates

        # Compute overall match decision BEFORE route reconstruction
        all_candidates_flat = [
            m for cands in camera_candidates.values() for m in cands
        ]
        match_decision = _compute_match_decision(all_candidates_flat)
        top_candidates = _build_top_candidates(camera_candidates, n=3)

        logger.info(
            f"[PipelineService] Session {session_id} "
            f"match_decision={match_decision} "
            f"total_candidates={len(all_candidates_flat)}"
        )

        # ── Step 6: route reconstruction (CONFIDENT_MATCH only) ──────────
        await self._push_progress(session, db, 75, "Reconstructing route...")
        route_service = get_route_service()

        # Only route-walk over tracks that were actually confident matches.
        # POSSIBLE_MATCH_REVIEW and FACE_MISMATCH tracks are in top_candidates
        # for human review but must not anchor a route as "confirmed".
        confident_candidates: dict[str, list[CameraMatch]] = {
            cam_id: [m for m in cands if m.match_status == MatchStatus.CONFIDENT_MATCH]
            for cam_id, cands in camera_candidates.items()
            if any(m.match_status == MatchStatus.CONFIDENT_MATCH for m in cands)
        }

        fused_route = route_service.reconstruct_route(confident_candidates)
        route_confidence = route_service.compute_route_confidence(fused_route)

        for fused in fused_route:
            sighting = await self._persist_sighting(
                db=db, session_id=session_id, match=fused.match,
                spatial=1.0, temporal=1.0, fusion=fused.match.appearance_score,
            )
            await db.commit()

            await self._ws.broadcast(WsSightingFound(
                session_id=session_id,
                sighting=SightingOut.model_validate(sighting),
            ).model_dump())

            if session.person and session.person.watchlist_status != "none":
                await self._fire_alert(db=db, session=session, sighting=sighting)

        # ── Step 7: persist route steps ───────────────────────────────────
        await self._push_progress(session, db, 88, "Persisting route...")
        from sqlalchemy import select  # noqa: PLC0415

        existing_result = await db.execute(
            select(Sighting).where(Sighting.session_id == session_id)
        )
        existing_sightings: dict[str, Sighting] = {
            s.camera_id: s for s in existing_result.scalars().all()
        }

        for step_order, fused in enumerate(fused_route):
            cam_id   = fused.match.camera_id
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

        # ── Step 8: build output + push completion event ──────────────────
        await self._push_progress(session, db, 96, "Finalising route...")

        sightings_result = await db.execute(
            select(Sighting).where(Sighting.session_id == session_id)
        )
        all_sightings = sightings_result.scalars().all()

        steps_result = await db.execute(
            select(RouteStep)
            .where(RouteStep.session_id == session_id)
            .order_by(RouteStep.step_order)
        )
        all_steps = steps_result.scalars().all()

        sighting_map = {s.id: s for s in all_sightings}
        cam_data_map = route_service.get_graph().get("cameras", {})

        step_outs: list[RouteStepOut] = []
        for step in all_steps:
            s = sighting_map.get(step.sighting_id)
            if s is None:
                continue
            location = cam_data_map.get(s.camera_id, {}).get("location", s.camera_id)
            step_outs.append(RouteStepOut(
                step_order=step.step_order,
                camera_id=s.camera_id,
                camera_location=location,
                track_id=s.track_id,
                timestamp=s.first_seen,
                confidence=s.best_confidence,
                fusion_score=s.fusion_score,
                crop_path=s.crop_path,
                matching_mode=getattr(s, "matching_mode", "body"),
                face_used=getattr(s, "face_used", False),
                identity_score=getattr(s, "identity_score", s.appearance_score),
            ))

        route_out = QueryRouteOut(
            session_id=session_id,
            status="done",
            steps=step_outs,
            sightings=[SightingOut.model_validate(s) for s in all_sightings],
            total_cameras_matched=len(all_sightings),
            route_confidence=route_confidence,
            match_decision=match_decision,
            top_candidates=top_candidates,
        )

        # ── Step 9: mark complete ──────────────────────────────────────────
        session.status        = "done"
        session.progress_pct  = 100
        session.completed_at  = datetime.utcnow()
        session.match_decision = match_decision
        await db.commit()

        await self._ws.broadcast(WsRouteComplete(
            session_id=session_id, route=route_out
        ).model_dump())

        logger.info(
            f"[PipelineService] Session {session_id} complete — "
            f"signals={'+'.join(active_signals)} "
            f"match_decision={match_decision} "
            f"confident_cameras={len(fused_route)}/{len(body_galleries)} "
            f"confidence={route_confidence:.4f}"
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _resolve_query_image_path(self, session: QuerySession) -> Optional[Path]:
        if session.query_image_path:
            p = Path(session.query_image_path)
            return p if p.exists() else None
        return None

    async def _resolve_query_embedding(
        self, session: QuerySession
    ) -> Optional[list[float]]:
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
        """Create and flush a Sighting ORM record with full evidence details."""
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
            # Evidence fields
            match_status=match.match_status.value,
            face_veto_applied=match.face_veto_applied,
            face_veto_reason=match.face_veto_reason,
            face_sim=match.face_sim,
            face_coverage=match.face_coverage,
            kpr_sim=match.kpr_sim,
            fused_score=match.fused_score,
            active_backbone=match.active_backbone,
            embedding_dim=match.embedding_dim,
            # Phase 4: face-first identity fields
            matching_mode=match.matching_mode,
            face_used=match.face_used,
            face_similarity=match.face_similarity,
            body_similarity=match.body_similarity,
            identity_score=match.appearance_score,
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
        """Fire a watchlist alert for CONFIDENT_MATCH sightings only."""
        # Only alert on confirmed matches — not POSSIBLE_MATCH_REVIEW
        if sighting.match_status != MatchStatus.CONFIDENT_MATCH.value:
            return

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
                f"(match score: {sighting.best_confidence:.1f} — "
                f"similarity-derived, not a probability)"
            ),
        )
        db.add(alert)
        await db.flush()

        await self._ws.broadcast(WsAlert(
            alert=AlertOut.model_validate(alert)
        ).model_dump())

        logger.warning(
            f"[PipelineService] [{severity}] Watchlist alert: "
            f"{person.name} @ {sighting.camera_id} "
            f"status={sighting.match_status}"
        )

    async def _push_progress(
        self,
        session: QuerySession,
        db: AsyncSession,
        pct: int,
        message: str,
    ) -> None:
        session.progress_pct = pct
        await db.commit()
        await self._ws.broadcast(WsQueryProgress(
            session_id=session.id,
            progress_pct=pct,
            message=message,
        ).model_dump())
