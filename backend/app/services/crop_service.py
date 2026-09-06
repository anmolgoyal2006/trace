"""
crop_service.py — CropService
Thin async wrapper around ai_pipeline/reid/crop_extractor.py.

Extracts quality-filtered person crop JPEGs from a video using pre-computed
ByteTrack bounding boxes, then writes metadata JSON ready for batch embedding.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from backend.app.config import settings

# Force UTF-8 stdout/stderr in all subprocesses — prevents cp1252 crash on
# Windows when pipeline scripts print unicode characters (e.g. ≥ in messages).
_SUBPROCESS_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}

_CROP_SCRIPT = settings.repo_root / "ai_pipeline" / "reid" / "crop_extractor.py"


class CropService:
    """
    Extracts person crops from a video using pre-computed tracks.

    Usage:
        svc = CropService()
        metadata_path = await svc.run(video_path, tracks_path, camera_id="C01")
    """

    async def run(
        self,
        video_path: Path,
        tracks_path: Path,
        camera_id: str,
        output_dir: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ) -> Path:
        """
        Extract crops and return the path to the metadata JSON.
        """
        if output_dir is None:
            output_dir = settings.data_dir / "crops" / camera_id
        if metadata_path is None:
            metadata_path = settings.data_dir / f"crops_metadata_{camera_id}.json"

        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(_CROP_SCRIPT),
            "--video", str(video_path),
            "--tracks", str(tracks_path),
            "--output-dir", str(output_dir),
            "--metadata", str(metadata_path),
        ]

        logger.info(f"[CropService] Extracting crops: camera={camera_id}")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=_SUBPROCESS_ENV,
        )
        stdout, _ = await proc.communicate()

        if proc.returncode != 0:
            msg = stdout.decode(errors="replace") if stdout else "(no output)"
            raise RuntimeError(
                f"Crop extraction failed for camera {camera_id}.\n"
                f"Output:\n{msg}"
            )

        logger.info(f"[CropService] Done: {metadata_path}")
        return metadata_path

    @staticmethod
    def load_metadata(metadata_path: Path) -> list[dict]:
        """Load crop metadata records from JSON."""
        if not metadata_path.exists():
            raise FileNotFoundError(f"Crop metadata not found: {metadata_path}")
        with open(metadata_path) as f:
            return json.load(f)
