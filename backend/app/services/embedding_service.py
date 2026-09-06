"""
embedding_service.py — EmbeddingService
Wraps the OSNet embedding pipeline from ai_pipeline/reid/embed.py.

Two modes:
  1. batch_embed()  — runs embed.py as a subprocess for a full crop directory
                      (used during video processing pipeline)
  2. embed_single() — runs OSNet in-process for a single image
                      (used for enrolling a reference photo or an ad-hoc query)

The in-process path loads the model once and caches it on the service instance
to avoid reloading weights on every request.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from PIL import Image

from backend.app.config import settings

# ── pipeline script for batch mode ────────────────────────────────────────
_EMBED_SCRIPT = settings.repo_root / "ai_pipeline" / "reid" / "embed.py"

# ── add repo root to path so we can import the pipeline module ────────────
if str(settings.repo_root) not in sys.path:
    sys.path.insert(0, str(settings.repo_root))

_SUBPROCESS_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


class EmbeddingService:
    """
    OSNet x1_0 embedding service.

    Keeps a single model instance alive for the lifetime of the process.
    Call load_model() once during app startup (inside the FastAPI lifespan).
    """

    def __init__(self) -> None:
        self._model: Optional[torch.nn.Module] = None
        self._device: Optional[torch.device] = None
        self._transform = None   # set when model loads

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def load_model(self) -> None:
        """
        Load OSNet x1_0 with pretrained weights.  Call once at startup.

        This is intentionally synchronous — it runs during the FastAPI
        lifespan startup phase before the event loop handles requests.
        """
        from ai_pipeline.reid.embed import load_model, select_device, _TRANSFORM

        logger.info("[EmbeddingService] Loading OSNet x1_0...")
        self._device = select_device()
        self._model = load_model(settings.reid_model, self._device)
        self._transform = _TRANSFORM
        logger.info(f"[EmbeddingService] Model ready on {self._device}")

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------ #
    # Single-image embedding (in-process)                                  #
    # ------------------------------------------------------------------ #

    def embed_single(self, image_path: Path) -> list[float]:
        """
        Embed one image through OSNet and return a 512-D float list.

        Used for reference photo enrollment and ad-hoc query embedding.

        Raises:
            RuntimeError: if model not loaded or image cannot be processed.
        """
        if not self.is_ready:
            raise RuntimeError(
                "EmbeddingService: model not loaded. "
                "Call load_model() during app startup."
            )

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Cannot open image {image_path}: {e}")

        tensor = self._transform(img).unsqueeze(0).to(self._device)  # [1, 3, 256, 128]

        with torch.no_grad():
            embedding = self._model(tensor)  # [1, 512]

        return embedding.cpu().squeeze(0).tolist()

    # ------------------------------------------------------------------ #
    # Batch embedding (subprocess)                                         #
    # ------------------------------------------------------------------ #

    async def batch_embed(
        self,
        metadata_path: Path,
        crop_dir: Path,
        output_path: Path,
        batch_size: int = 32,
    ) -> Path:
        """
        Run the full Phase 4 batch embedding pipeline via subprocess.

        Returns the path to the output embeddings JSON.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(_EMBED_SCRIPT),
            "--metadata", str(metadata_path),
            "--crop-dir", str(crop_dir),
            "--output", str(output_path),
            "--batch-size", str(batch_size),
            "--overwrite",
        ]

        logger.info(f"[EmbeddingService] Starting batch embed → {output_path.name}")

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
                f"Batch embedding failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Output:\n{msg}"
            )

        logger.info(f"[EmbeddingService] Batch embed done: {output_path}")
        return output_path

    # ------------------------------------------------------------------ #
    # Gallery loading                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def load_gallery(embeddings_path: Path) -> list[dict]:
        """Load and return embedding records from a JSON file."""
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        with open(embeddings_path) as f:
            return json.load(f)

    @staticmethod
    def save_gallery(records: list[dict], output_path: Path) -> None:
        """Persist embedding records to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(records, f, indent=2)


# Module-level singleton — imported by the FastAPI app and routers
embedding_service = EmbeddingService()
