"""
detection_service.py — DetectionService
Thin async wrapper around ai_pipeline/detection/detect.py.

Runs YOLO person detection on a video file and returns the detections JSON
path (the detection script writes its own output file).  Heavy lifting stays
in the existing, tested pipeline script — this service just invokes it as a
subprocess so the async FastAPI event loop is never blocked.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from backend.app.config import settings

_SUBPROCESS_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}

# ── resolve the pipeline script path ──────────────────────────────────────
_DETECT_SCRIPT = settings.repo_root / "ai_pipeline" / "detection" / "detect.py"


class DetectionService:
    """
    Runs YOLO person detection for one camera video.

    Usage:
        svc = DetectionService()
        output_path = await svc.run(video_path, camera_id="C01")
        detections = svc.load_detections(output_path)
    """

    async def run(
        self,
        video_path: Path,
        camera_id: str,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Run detection on *video_path* and return the path to the output JSON.

        The output defaults to dataset/detections_<camera_id>.json inside
        the repo's dataset directory if not specified.
        """
        if output_path is None:
            output_path = settings.data_dir / f"detections_{camera_id}.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(_DETECT_SCRIPT),
            "--video", str(video_path),
            "--camera-id", camera_id,
            "--output", str(output_path),
        ]

        logger.info(f"[DetectionService] Starting detection: camera={camera_id} video={video_path.name}")

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
                f"Detection failed for camera {camera_id}.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Output:\n{msg}"
            )

        logger.info(f"[DetectionService] Done: {output_path}")
        return output_path

    @staticmethod
    def load_detections(output_path: Path) -> list[dict]:
        """Load and return the detection records from the output JSON."""
        if not output_path.exists():
            raise FileNotFoundError(f"Detection output not found: {output_path}")
        with open(output_path) as f:
            return json.load(f)
