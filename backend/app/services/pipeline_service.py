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


def _peek_gallery_dim(gallery: list[dict]) -> Optional[int]:
    """Return the embedding dimension of a gallery (None if empty/malformed)."""
    for rec in gallery:
        emb = rec.get("embedding")
        if isinstance(emb, list) and emb:
            return len(emb)
    return None


def _solider_paths() -> Optional[tuple[Path, Path]]:
    """
    Return (weights, config) paths when SOLIDER is configured AND the files
    exist, else None. NOTE: settings.solider_weights defaults to Path("")
    which normalises to Path(".") — always truthy and exists() — so compare
    the string form explicitly.
    """
    w, c = str(settings.solider_weights or ""), str(settings.solider_config_path or "")
    if w.strip() in ("", ".") or c.strip() in ("", "."):
        return None
    wp, cp = Path(w), Path(c)
    if not wp.exists() or not cp.exists():
        return None
    return wp, cp


# Cached SOLIDER model (one per weights path) — loaded once, reused for all
# queries. The SOLIDER-REID modules use unique top-level names for this
# process (config/model/datasets under its repo root, imported via path
# insert below), so no isolation surgery is needed unlike the KPR fork.
_solider_model_cache: dict[str, object] = {}


def _embed_query_solider(img_path: Path) -> Optional[list[float]]:
    """
    Embed a query image with SOLIDER Swin-Small (768-dim, L2-normalised).
    Returns None when SOLIDER is not configured or embedding fails.

    Runs synchronously — call inside run_in_executor.
    """
    paths = _solider_paths()
    if paths is None:
        logger.debug("[PipelineService] SOLIDER weights not configured — skipping")
        return None
    weights_path, config_path = paths

    try:
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        _repo_root = Path(__file__).resolve().parents[3]
        if str(_repo_root) not in sys.path:
            sys.path.insert(0, str(_repo_root))

        from ai_pipeline.reid.embed_solider import (  # noqa: PLC0415
            SOLIDER_TRANSFORM,
            load_solider_model,
            solider_forward,
        )

        solider_root = _repo_root / "SOLIDER-REID"
        cache_key = str(weights_path.resolve())
        model = _solider_model_cache.get(cache_key)
        if model is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = load_solider_model(
                weights_path, config_path, solider_root, device
            )
            _solider_model_cache[cache_key] = model
            logger.info("[PipelineService] SOLIDER model cached for queries")

        device = next(model.parameters()).device
        img = Image.open(img_path).convert("RGB")
        tensor = SOLIDER_TRANSFORM(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = solider_forward(model, tensor)
        vec = emb.cpu().squeeze(0).tolist()
        logger.info(f"[PipelineService] SOLIDER query embedded: dim={len(vec)}")
        return vec

    except Exception as exc:
        logger.warning(
            f"[PipelineService] SOLIDER query embedding failed (non-fatal): {exc}"
        )
        return None


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


# Module-level YOLO cache for query auto-crop (lazy — keeps startup fast)
_yolo_model = None


def _get_yolo_model():
    """
    Load YOLOv8n once and cache it for query-image person detection.
    Uses the same torch>=2.6 weights_only-safe load as the pipeline scripts.
    """
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    import torch  # noqa: PLC0415
    from ultralytics import YOLO  # noqa: PLC0415

    _orig_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    model_path = Path(settings.yolo_model)
    if not model_path.is_absolute():
        model_path = Path(__file__).resolve().parents[3] / model_path

    torch.load = _patched_load
    try:
        _yolo_model = YOLO(str(model_path))
    finally:
        torch.load = _orig_load
    logger.info(f"[PipelineService] YOLO query model ready: {model_path.name}")
    return _yolo_model


def _autocrop_query_person(img_path: Path) -> Path:
    """
    Crop a multi-person query image down to its largest detected person.

    Full-frame snaps (e.g. video screenshots) embed as one meaningless
    vector and match nothing. Cropping the most prominent subject makes
    every downstream signal (body / face / KPR) see a single person —
    the same framing the gallery crops have.

    Returns the crop path, or the original path unchanged when no person
    is detected or detection fails. Never raises.
    Runs synchronously — call inside run_in_executor.
    """
    try:
        import cv2  # noqa: PLC0415

        model = _get_yolo_model()
        results = model.predict(
            source=str(img_path),
            conf=settings.detection_confidence,
            classes=settings.detection_classes,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            logger.warning(
                "[PipelineService] No person detected in query image — "
                "embedding full image (may match nothing)"
            )
            return img_path

        xyxy = boxes.xyxy.tolist()
        confs = boxes.conf.tolist()
        # Largest box by area = most prominent subject
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in xyxy]
        best = max(range(len(xyxy)), key=lambda i: areas[i])
        x1, y1, x2, y2 = (int(v) for v in xyxy[best])

        img = cv2.imread(str(img_path))
        if img is None:
            return img_path
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 32 or y2 - y1 < 64:
            logger.warning(
                "[PipelineService] Detected person box too small "
                f"({x2 - x1}x{y2 - y1}) — embedding full image"
            )
            return img_path

        crop_path = img_path.parent / f"{img_path.stem}_person{img_path.suffix}"
        cv2.imwrite(str(crop_path), img[y1:y2, x1:x2])
        logger.info(
            f"[PipelineService] Query auto-crop: {len(xyxy)} person(s), "
            f"kept largest (conf={confs[best]:.2f}, "
            f"box={x2 - x1}x{y2 - y1}) → {crop_path.name}"
        )
        return crop_path
    except Exception as exc:
        logger.warning(
            f"[PipelineService] Query auto-crop failed (non-fatal, "
            f"using full image): {exc}"
        )
        return img_path


def _embed_query_kpr(img_path: Path) -> Optional[dict]:
    """
    Run KPR on a single query image to produce holistic + part embeddings.
    Returns a dict with keys: holistic_embedding, part_embeddings,
    part_visibility — or None if KPR weights are not configured or the
    subprocess fails.

    Runs KPR in a FRESH subprocess via embed_kpr.py's CLI (same reason as
    the upload path: the KPR torchreid fork cannot coexist in-process with
    the standard torchreid used by OSNet). The single image is wrapped in
    a 1-record metadata file, embedded, and the record parsed back.

    Runs synchronously — call inside run_in_executor.
    """
    import os
    import subprocess
    import tempfile

    kpr_weights = settings.kpr_weights_path
    kpr_config = settings.kpr_config_path

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

    _repo_root = Path(__file__).resolve().parents[3]
    _script = _repo_root / "ai_pipeline" / "reid" / "embed_kpr.py"

    # Infer kpr_root: prefer the standard clone location at the repo root,
    # then fall back to walking up from the config path.
    kpr_root = None
    _candidate = _repo_root / "keypoint_promptable_reidentification"
    if (_candidate / "torchreid" / "scripts" / "builder.py").exists():
        kpr_root = _candidate
    if kpr_root is None:
        kpr_root = kpr_c.resolve().parent
        for parent in kpr_c.resolve().parents:
            if (parent / "setup.py").exists() or (parent / "torchreid").exists():
                kpr_root = parent
                break

    tmp_dir = Path(tempfile.mkdtemp(prefix="kpr_query_"))
    try:
        meta_path = tmp_dir / "query_metadata.json"
        out_path = tmp_dir / "query_kpr.json"
        # Absolute crop_path: embed_kpr does _REPO_ROOT / rec["crop_path"],
        # and pathlib lets an absolute right-hand side win, so this resolves
        # back to the query image itself.
        with open(meta_path, "w") as f:
            json.dump([{
                "crop_path": str(img_path.resolve()),
                "camera_id": "QUERY",
                "track_id": 0,
                "frame": 0,
                "timestamp": "00:00:00.00",
                "bbox": [0, 0, 4096, 4096],
                "detection_confidence": 1.0,
            }], f)

        cmd = [
            sys.executable,
            str(_script),
            "--metadata", str(meta_path),
            "--crop-dir", str(img_path.resolve().parent),
            "--output", str(out_path),
            "--kpr-weights", str(kpr_w),
            "--kpr-config", str(kpr_c),
            "--kpr-root", str(kpr_root),
            "--overwrite",
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(_repo_root),
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            timeout=600,  # 10 min cap for a single query image on CPU
        )
        if proc.returncode != 0:
            out = proc.stdout.decode(errors="replace") if proc.stdout else "(no output)"
            logger.warning(
                f"[PipelineService] KPR query subprocess failed "
                f"(code {proc.returncode}, non-fatal):\n{out[-2000:]}"
            )
            return None
        if not out_path.exists():
            logger.warning("[PipelineService] KPR query produced no output — skipping")
            return None
        with open(out_path) as f:
            records = json.load(f)
        if not records:
            logger.warning("[PipelineService] KPR query embedded 0 crops — skipping")
            return None
        rec = records[0]
        logger.info("[PipelineService] KPR query embedded via subprocess")
        return {
            "holistic_embedding": rec["holistic_embedding"],
            "part_embeddings": rec["part_embeddings"],
            "part_visibility": rec["part_visibility"],
        }
    except (Exception, SystemExit) as exc:
        logger.warning(f"[PipelineService] KPR query embedding failed (non-fatal): {exc}")
        return None
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


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
            except (Exception, SystemExit) as e:
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
        # Normalise the query image first: multi-person snaps (e.g. full
        # video screenshots) are auto-cropped to the largest detected
        # person so every signal embeds one subject, not the whole scene.
        img_path = self._resolve_query_image_path(session)
        if img_path is not None:
            await self._push_progress(session, db, 3, "Detecting subject in query photo...")
            img_path = await loop.run_in_executor(
                None, _autocrop_query_person, img_path
            )
        query_embedding = await self._resolve_query_embedding(session, img_path)
        if query_embedding is None:
            raise ValueError(
                "No embedding available. Enroll a reference photo for this person "
                "or upload a query image."
            )

        # ── Step 2: face embedding for query image (optional) ─────────────
        await self._push_progress(session, db, 10, "Extracting face from query image...")
        query_face_embedding: Optional[list[float]] = None
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

        # ── Step 4b: SOLIDER query vector (optional, per-gallery dim) ──────
        # Galleries may mix backbones (e.g. C01 SOLIDER-768, C02 OSNet-512).
        # Build a dim → query-vector map and pick the matching vector per
        # camera in Step 5. Enrolled-person vectors participate with their
        # native dim; cameras with no matching vector are skipped.
        query_vecs: dict[int, list[float]] = {len(query_embedding): query_embedding}
        gallery_dims = {
            cam_id: _peek_gallery_dim(g) for cam_id, g in body_galleries.items()
        }
        logger.info(f"[PipelineService] Gallery dims: {gallery_dims}")
        need_solider_dim = settings.solider_embedding_dim
        if (
            need_solider_dim not in query_vecs
            and need_solider_dim in set(gallery_dims.values())
            and img_path is not None
        ):
            await self._push_progress(
                session, db, 17, "Embedding query with SOLIDER..."
            )
            solider_vec = await loop.run_in_executor(
                None, _embed_query_solider, img_path
            )
            if solider_vec is not None:
                query_vecs[len(solider_vec)] = solider_vec
                logger.info("[PipelineService] SOLIDER signal: active")
            else:
                logger.info(
                    "[PipelineService] SOLIDER signal: inactive "
                    "(weights missing or embedding failed)"
                )

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

            # Pick the query vector matching this gallery's backbone dim.
            # Galleries with no compatible query vector are skipped (e.g.
            # enrolled OSNet-512 vector vs SOLIDER-768 gallery — re-enroll
            # the person after switching gallery backbones).
            cam_dim = gallery_dims.get(cam_id)
            cam_query_vec = query_vecs.get(cam_dim) if cam_dim else None
            if cam_query_vec is None:
                logger.warning(
                    f"[PipelineService] Skipping {cam_id}: no query vector "
                    f"with dim={cam_dim} (have dims={sorted(query_vecs)})"
                )
                continue

            candidates = _matching_service.search_camera_top_k(
                query_embedding=cam_query_vec,
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

        if not camera_candidates and any(
            d is not None and d not in query_vecs
            for d in gallery_dims.values()
        ):
            raise ValueError(
                "Query embedding dimension does not match any gallery "
                f"(query dims={sorted(query_vecs)}, "
                f"gallery dims={gallery_dims}). If you switched gallery "
                "backbones (OSNet ↔ SOLIDER), re-enroll reference photos "
                "or search with a query image so the vector can be "
                "recomputed in the gallery's dimension."
            )

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
                match_tier=ms.match_tier,
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
        self, session: QuerySession, img_path: Optional[Path] = None
    ) -> Optional[list[float]]:
        """
        Get the body query embedding:
          - Registered person with enrolled embedding → use stored vector.
          - Uploaded query image → embed on the fly with OSNet.
        img_path, when given, is the (possibly auto-cropped) query image
        resolved by the caller; otherwise it falls back to the session path.
        """
        if session.person and session.person.embedding_vector:
            logger.info(
                f"[PipelineService] Using enrolled OSNet embedding "
                f"for person {session.person_id}"
            )
            return session.person.embedding_vector

        if img_path is None and session.query_image_path:
            img_path = Path(session.query_image_path)
        if img_path is not None:
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
        tier = "confident" if match.is_confident else "possible"
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
            match_tier=tier,
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
