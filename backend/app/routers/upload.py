"""
upload.py — /api/upload
Video file upload + batch processing pipeline trigger.

Flow:
  POST /api/upload/video  →  saves file, kicks off background task:
    detect → track → extract crops → embed (OSNet) → face embed → KPR embed
  GET  /api/upload/status/{job_id}  →  poll job status (in-memory store)

Face and KPR embedding steps run automatically after OSNet if their model
weights are configured (face_det_model, face_rec_model, kpr_weights_path,
kpr_config_path in config.py / .env).  If weights are missing the steps are
skipped gracefully — the OSNet gallery is always produced.
"""

import asyncio
import sys
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from loguru import logger

from backend.app.config import settings
from backend.app.models.schemas import UploadJobOut
from backend.app.services.detection_service import DetectionService
from backend.app.services.tracker_service import TrackerService
from backend.app.services.crop_service import CropService
from backend.app.services.embedding_service import embedding_service

router = APIRouter(prefix="/api/upload", tags=["upload"])

# In-memory job store (resets on server restart — acceptable for MVP)
_jobs: dict[str, dict] = {}

_ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


# ---------------------------------------------------------------------------
# Upload video
# ---------------------------------------------------------------------------

@router.post("/video", response_model=UploadJobOut, status_code=status.HTTP_202_ACCEPTED)
async def upload_video(
    background_tasks: BackgroundTasks,
    camera_id: str = Form(...),
    video: UploadFile = File(...),
) -> UploadJobOut:
    """
    Upload a video file for a specific camera and trigger the full
    detection → tracking → crop extraction → embedding pipeline.

    camera_id must be one of the MVP cameras: C01, C02, C03.
    """
    # Validate camera ID
    valid_cameras = {"C01", "C02", "C03"}
    if camera_id not in valid_cameras:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid camera_id '{camera_id}'. Must be one of: {sorted(valid_cameras)}",
        )

    # Validate file extension
    ext = Path(video.filename).suffix.lower() if video.filename else ""
    if ext not in _ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{ext}'. Allowed: {_ALLOWED_VIDEO_EXTENSIONS}",
        )

    # Save to disk
    dest_dir = settings.raw_videos_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{camera_id}_{uuid.uuid4().hex[:8]}{ext}"
    dest_path = dest_dir / filename

    with open(dest_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    logger.info(f"[upload] Saved video: {dest_path}")

    # Create job record
    job_id = uuid.uuid4().hex
    _jobs[job_id] = {
        "job_id": job_id,
        "camera_id": camera_id,
        "filename": filename,
        "status": "queued",
        "message": "Video saved. Pipeline queued: tracking → crops → OSNet → face → KPR.",
    }

    # Kick off pipeline in background
    background_tasks.add_task(_run_video_pipeline, job_id, dest_path, camera_id)

    return UploadJobOut(**_jobs[job_id])


# ---------------------------------------------------------------------------
# Poll job status
# ---------------------------------------------------------------------------

@router.get("/status/{job_id}", response_model=UploadJobOut)
async def get_upload_status(job_id: str) -> UploadJobOut:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return UploadJobOut(**job)


@router.get("/jobs", response_model=list[UploadJobOut])
async def list_jobs() -> list[UploadJobOut]:
    """List all upload jobs (most recent first by insertion order)."""
    jobs = list(reversed(list(_jobs.values())))
    return [UploadJobOut(**j) for j in jobs]


# ---------------------------------------------------------------------------
# Background pipeline
# ---------------------------------------------------------------------------

async def _run_video_pipeline(job_id: str, video_path: Path, camera_id: str) -> None:
    """
    Full processing pipeline for an uploaded video:
      1. Detection (YOLOv8) + Tracking (ByteTrack)
      2. Crop extraction (quality-filtered)
      3. OSNet body embeddings  → embeddings_<cam>.json       (always)
      4. Face embeddings        → face_embeddings_<cam>.json  (if weights set)
      5. KPR part embeddings    → kpr_embeddings_<cam>.json   (if weights set)

    Steps 4 and 5 are skipped gracefully if their model weight paths are not
    configured — the OSNet gallery is always produced regardless.
    """

    def _update(status: str, message: str) -> None:
        _jobs[job_id]["status"] = status
        _jobs[job_id]["message"] = message
        logger.info(f"[upload job {job_id}] {status}: {message}")

    try:
        trk_svc  = TrackerService()
        crop_svc = CropService()

        crop_dir      = settings.data_dir / "crops" / camera_id
        metadata_path = settings.data_dir / f"crops_metadata_{camera_id}.json"

        # ── Step 1: Tracking (includes detection) ──────────────────────
        _update("processing", f"[1/5] Running detection + tracking for {camera_id}...")
        tracks_path = await trk_svc.run(video_path, camera_id)

        # ── Step 2: Crop extraction ─────────────────────────────────────
        _update("processing", f"[2/5] Extracting crops for {camera_id}...")
        await crop_svc.run(
            video_path=video_path,
            tracks_path=tracks_path,
            camera_id=camera_id,
            output_dir=crop_dir,
            metadata_path=metadata_path,
        )

        # ── Step 3: OSNet body embeddings (always) ─────────────────────
        _update("processing", f"[3/5] Generating OSNet body embeddings for {camera_id}...")
        embeddings_path = settings.embeddings_dir / f"embeddings_{camera_id}.json"
        await embedding_service.batch_embed(
            metadata_path=metadata_path,
            crop_dir=crop_dir,
            output_path=embeddings_path,
        )
        logger.info(f"[upload job {job_id}] OSNet gallery ready: {embeddings_path.name}")

        # ── Step 4: Face embeddings (SCRFD + ArcFace, optional) ────────
        face_det   = Path(settings.face_det_model)  if settings.face_det_model  else None
        face_rec   = Path(settings.face_rec_model)  if settings.face_rec_model  else None

        if face_det and face_rec and face_det.exists() and face_rec.exists():
            _update("processing", f"[4/5] Generating face embeddings for {camera_id}...")
            face_output = settings.embeddings_dir / f"face_embeddings_{camera_id}.json"
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    _run_face_embed,
                    metadata_path,
                    crop_dir,
                    face_output,
                    face_det,
                    face_rec,
                )
                logger.info(
                    f"[upload job {job_id}] Face gallery ready: {face_output.name}"
                )
            except Exception as exc:
                logger.warning(
                    f"[upload job {job_id}] Face embedding failed (non-fatal): {exc}"
                )
        else:
            missing = []
            if not face_det or not face_det.exists():
                missing.append("face_det_model (det_10g.onnx)")
            if not face_rec or not face_rec.exists():
                missing.append("face_rec_model (w600k_r50.onnx)")
            logger.info(
                f"[upload job {job_id}] Skipping face embeddings — "
                f"weights not configured: {', '.join(missing)}"
            )
            _update("processing", f"[4/5] Skipped face embeddings (weights not set)")

        # ── Step 5: KPR part embeddings (optional) ─────────────────────
        kpr_weights = Path(settings.kpr_weights_path) if settings.kpr_weights_path else None
        kpr_config  = Path(settings.kpr_config_path)  if settings.kpr_config_path  else None

        if kpr_weights and kpr_config and kpr_weights.exists() and kpr_config.exists():
            _update("processing", f"[5/5] Generating KPR part embeddings for {camera_id}...")
            kpr_output = settings.embeddings_dir / f"kpr_embeddings_{camera_id}.json"
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    _run_kpr_embed,
                    metadata_path,
                    crop_dir,
                    kpr_output,
                    kpr_weights,
                    kpr_config,
                )
                logger.info(
                    f"[upload job {job_id}] KPR gallery ready: {kpr_output.name}"
                )
            except Exception as exc:
                logger.warning(
                    f"[upload job {job_id}] KPR embedding failed (non-fatal): {exc}"
                )
        else:
            missing = []
            if not kpr_weights or not kpr_weights.exists():
                missing.append("kpr_weights_path (.pth.tar)")
            if not kpr_config or not kpr_config.exists():
                missing.append("kpr_config_path (.yaml)")
            logger.info(
                f"[upload job {job_id}] Skipping KPR embeddings — "
                f"weights not configured: {', '.join(missing)}"
            )
            _update("processing", f"[5/5] Skipped KPR embeddings (weights not set)")

        _update(
            "done",
            f"Pipeline complete for {camera_id}. "
            f"OSNet gallery ready"
            + (f" · face gallery ready" if (face_det and face_det.exists() and face_rec and face_rec.exists()) else "")
            + (f" · KPR gallery ready"  if (kpr_weights and kpr_weights.exists() and kpr_config and kpr_config.exists()) else "")
            + "."
        )

    except Exception as e:
        logger.exception(f"[upload job {job_id}] Pipeline failed: {e}")
        _update("failed", str(e))


# ---------------------------------------------------------------------------
# Sync helpers (run in executor so they don't block the async loop)
# ---------------------------------------------------------------------------

def _run_face_embed(
    metadata_path: Path,
    crop_dir: Path,
    output_path: Path,
    det_weight: Path,
    rec_weight: Path,
) -> None:
    """
    Thin synchronous wrapper around face_embed.run_face_embedding_pipeline().
    Runs in a thread-pool executor so it doesn't block the event loop.
    """
    # Lazy import — keeps startup fast when face weights are not installed
    _repo_root = Path(__file__).resolve().parents[3]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

    from ai_pipeline.reid.face_embed import run_face_embedding_pipeline  # noqa: PLC0415

    import torch
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    run_face_embedding_pipeline(
        metadata_path=metadata_path,
        crop_dir=crop_dir,
        output_path=output_path,
        det_weight=det_weight,
        rec_weight=rec_weight,
        face_threshold=settings.face_det_threshold,
        min_face_size=20,
        overwrite=True,
    )


def _run_kpr_embed(
    metadata_path: Path,
    crop_dir: Path,
    output_path: Path,
    kpr_weights: Path,
    kpr_config: Path,
) -> None:
    """
    Thin synchronous wrapper around embed_kpr.run_kpr_embedding_pipeline().
    Runs in a thread-pool executor so it doesn't block the event loop.
    """
    # Lazy import — KPR requires its own torchreid fork; keep startup clean
    _repo_root = Path(__file__).resolve().parents[3]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

    from ai_pipeline.reid.embed_kpr import run_kpr_embedding_pipeline  # noqa: PLC0415

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # kpr_root is the cloned KPR repo root — expected one level above kpr_config
    # e.g.  kpr_config = /path/to/keypoint_promptable_reidentification/configs/...yaml
    #        kpr_root  = /path/to/keypoint_promptable_reidentification
    kpr_root = kpr_config.resolve().parents[
        next(
            i for i, p in enumerate(kpr_config.resolve().parents)
            if (p / "setup.py").exists() or (p / "torchreid").exists()
        )
    ]

    run_kpr_embedding_pipeline(
        metadata_path=metadata_path,
        crop_dir=crop_dir,
        output_path=output_path,
        batch_size=16,
        kpr_weights=kpr_weights,
        kpr_config=kpr_config,
        kpr_root=kpr_root,
        min_crop_width=32,
        min_crop_height=64,
        device=device,
        overwrite=True,
    )
