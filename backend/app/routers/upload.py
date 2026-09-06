"""
upload.py — /api/upload
Video file upload + batch processing pipeline trigger.

Flow:
  POST /api/upload/video  →  saves file, kicks off background task:
    detect → track → extract crops → embed → gallery stored
  GET  /api/upload/status/{job_id}  →  poll job status (in-memory store)
"""

import asyncio
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
        "message": "Video saved. Processing queued.",
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
      1. Detection (YOLOv8)
      2. Tracking (ByteTrack)
      3. Crop extraction (quality-filtered)
      4. Batch embedding (OSNet)
    """

    def _update(status: str, message: str) -> None:
        _jobs[job_id]["status"] = status
        _jobs[job_id]["message"] = message
        logger.info(f"[upload job {job_id}] {status}: {message}")

    try:
        det_svc = DetectionService()
        trk_svc = TrackerService()
        crop_svc = CropService()

        # ── Step 1: Tracking (includes detection) ──────────────────────
        _update("processing", f"[1/3] Running detection + tracking for {camera_id}...")
        tracks_path = await trk_svc.run(video_path, camera_id)

        # ── Step 2: Crop extraction ─────────────────────────────────────
        _update("processing", f"[2/3] Extracting crops for {camera_id}...")
        crop_dir = settings.data_dir / "crops" / camera_id
        metadata_path = settings.data_dir / f"crops_metadata_{camera_id}.json"
        await crop_svc.run(
            video_path=video_path,
            tracks_path=tracks_path,
            camera_id=camera_id,
            output_dir=crop_dir,
            metadata_path=metadata_path,
        )

        # ── Step 3: Batch embedding ────────────────────────────────────
        _update("processing", f"[3/3] Generating OSNet embeddings for {camera_id}...")
        embeddings_path = settings.embeddings_dir / f"embeddings_{camera_id}.json"
        await embedding_service.batch_embed(
            metadata_path=metadata_path,
            crop_dir=crop_dir,
            output_path=embeddings_path,
        )

        _update("done", f"Pipeline complete. Gallery: {embeddings_path.name}")

    except Exception as e:
        logger.exception(f"[upload job {job_id}] Pipeline failed: {e}")
        _update("failed", str(e))
