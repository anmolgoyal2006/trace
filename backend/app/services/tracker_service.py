"""
tracker_service.py — TrackerService
Thin async wrapper around ai_pipeline/tracking/track.py.

Runs YOLOv8 + ByteTrack on a single camera video and returns the tracks
JSON path.  Reuses the existing, tested pipeline script via subprocess.
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

_TRACK_SCRIPT = settings.repo_root / "ai_pipeline" / "tracking" / "track.py"


class TrackerService:
    """
    Runs ByteTrack person tracking for one camera video.

    Usage:
        svc = TrackerService()
        tracks_path = await svc.run(video_path, camera_id="C01")
        tracks = svc.load_tracks(tracks_path)
    """

    async def run(
        self,
        video_path: Path,
        camera_id: str,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Run tracking on *video_path* and return the path to the output JSON.
        """
        if output_path is None:
            output_path = settings.data_dir / f"tracks_{camera_id}.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(_TRACK_SCRIPT),
            "--video", str(video_path),
            "--camera-id", camera_id,
            "--output", str(output_path),
        ]

        logger.info(f"[TrackerService] Starting tracking: camera={camera_id} video={video_path.name}")

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
                f"Tracking failed for camera {camera_id}.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Output:\n{msg}"
            )

        logger.info(f"[TrackerService] Done: {output_path}")
        return output_path

    @staticmethod
    def load_tracks(tracks_path: Path) -> list[dict]:
        """Load and return track records from the output JSON."""
        if not tracks_path.exists():
            raise FileNotFoundError(f"Tracks output not found: {tracks_path}")
        with open(tracks_path) as f:
            return json.load(f)

    @staticmethod
    def group_by_track(tracks: list[dict]) -> dict[int, list[dict]]:
        """
        Group flat track records by track_id.

        Returns {track_id: [records sorted by frame]}.
        """
        grouped: dict[int, list[dict]] = {}
        for rec in tracks:
            tid = rec["track_id"]
            grouped.setdefault(tid, []).append(rec)
        for recs in grouped.values():
            recs.sort(key=lambda r: r["frame"])
        return grouped
