"""
matching_service.py — MatchingService
Cross-camera Re-ID matching using OSNet embeddings.

Given a query embedding and one or more per-camera gallery files, this
service finds the best matching track in each camera and returns a list
of CameraMatch objects ranked by appearance score.

Uses:
  - ai_pipeline/reid/similarity.py  → cosine_similarity
  - ai_pipeline/reid/track_aggregation.py → aggregate_by_track
  - ai_pipeline/reid/confidence_scaling.py → similarity_to_confidence
  - config.no_match_threshold (0.74) for filtering low-confidence matches
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

from backend.app.config import settings

# Import existing, tested pipeline modules
if str(settings.repo_root) not in sys.path:
    sys.path.insert(0, str(settings.repo_root))

from ai_pipeline.reid.similarity import cosine_similarity          # noqa: E402
from ai_pipeline.reid.track_aggregation import aggregate_by_track  # noqa: E402
from ai_pipeline.reid.confidence_scaling import similarity_to_confidence  # noqa: E402


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CameraMatch:
    """
    Best matching track found in a single camera's gallery.

    All scores are in [0.0, 1.0] unless noted.
    confidence is in [0.0, 100.0].
    """
    camera_id: str
    track_id: int
    appearance_score: float          # max cosine similarity for best track
    mean_top3_similarity: float      # mean of top-3 crops for the best track
    best_confidence: float           # similarity_to_confidence(appearance_score), [0,100]
    mean_confidence: float           # similarity_to_confidence(mean_top3), [0,100]
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    best_crop_path: Optional[str] = None
    # crop-level records for the winning track (sorted by similarity desc)
    top_crops: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class MatchingService:
    """
    Cross-camera appearance matching.

    search_camera() is the primary entry point: call it once per camera
    with the gallery embeddings for that camera and the query embedding.
    """

    def __init__(self) -> None:
        self._threshold = settings.no_match_threshold

    # ------------------------------------------------------------------ #
    # Core search                                                          #
    # ------------------------------------------------------------------ #

    def search_camera(
        self,
        query_embedding: list[float],
        gallery: list[dict],
        camera_id: str,
        source_crop_path: Optional[str] = None,
        top_k: int = 5,
    ) -> Optional[CameraMatch]:
        """
        Find the best matching track in *gallery* for *query_embedding*.

        Returns None if no track exceeds the no-match threshold.

        Args:
            query_embedding:  512-D float list from OSNet.
            gallery:          List of embedding records (from embeddings_*.json).
            camera_id:        Camera identifier (for logging and result tagging).
            source_crop_path: Exclude this crop from results (anti-leakage).
            top_k:            Number of per-crop candidates to consider.

        Returns:
            CameraMatch with the best track, or None if below threshold.
        """
        if not gallery:
            logger.warning(f"[MatchingService] Empty gallery for camera {camera_id}")
            return None

        # ── step 1: compute crop-level similarities ──────────────────────
        crop_sims: list[dict] = []
        for rec in gallery:
            if source_crop_path and rec.get("crop_path") == source_crop_path:
                continue  # anti-leakage exclusion
            emb = rec.get("embedding")
            if not emb:
                continue
            try:
                sim = cosine_similarity(query_embedding, emb)
            except (ValueError, Exception) as e:
                logger.debug(f"[MatchingService] Skipping crop {rec.get('crop_path')}: {e}")
                continue
            crop_sims.append({
                "track_id":   rec.get("track_id"),
                "similarity": sim,
                "timestamp":  rec.get("timestamp"),
                "crop_path":  rec.get("crop_path"),
                "frame":      rec.get("frame"),
            })

        if not crop_sims:
            logger.warning(f"[MatchingService] No valid crops for camera {camera_id}")
            return None

        # ── step 2: aggregate per-track ──────────────────────────────────
        track_stats = aggregate_by_track(crop_sims)  # {track_id: {max, mean, mean_top3, n}}

        # ── step 3: pick the best track by max_similarity ────────────────
        best_track_id = max(track_stats, key=lambda t: track_stats[t]["max_similarity"])
        best_stats = track_stats[best_track_id]
        appearance_score = best_stats["max_similarity"]

        # ── step 4: apply no-match threshold ─────────────────────────────
        if appearance_score < self._threshold:
            logger.info(
                f"[MatchingService] camera={camera_id} best_sim={appearance_score:.4f} "
                f"< threshold={self._threshold} → no confident match"
            )
            return None

        # ── step 5: collect crop details for the winning track ───────────
        track_crops = sorted(
            [c for c in crop_sims if c["track_id"] == best_track_id],
            key=lambda c: c["similarity"],
            reverse=True,
        )
        top_crops = track_crops[:top_k]

        # Derive timestamps from crop records
        all_track_timestamps = [
            c["timestamp"] for c in track_crops if c.get("timestamp")
        ]
        first_seen = min(all_track_timestamps) if all_track_timestamps else None
        last_seen = max(all_track_timestamps) if all_track_timestamps else None
        best_crop_path = top_crops[0]["crop_path"] if top_crops else None

        best_confidence = round(similarity_to_confidence(appearance_score), 1)
        mean_confidence = round(
            similarity_to_confidence(best_stats["mean_top3_similarity"]), 1
        )

        logger.info(
            f"[MatchingService] camera={camera_id} "
            f"track={best_track_id} "
            f"sim={appearance_score:.4f} "
            f"conf={best_confidence:.1f}"
        )

        return CameraMatch(
            camera_id=camera_id,
            track_id=int(best_track_id),
            appearance_score=round(appearance_score, 6),
            mean_top3_similarity=round(best_stats["mean_top3_similarity"], 6),
            best_confidence=best_confidence,
            mean_confidence=mean_confidence,
            first_seen=first_seen,
            last_seen=last_seen,
            best_crop_path=best_crop_path,
            top_crops=top_crops,
        )

    def search_all_cameras(
        self,
        query_embedding: list[float],
        galleries: dict[str, list[dict]],
        source_crop_path: Optional[str] = None,
    ) -> dict[str, Optional[CameraMatch]]:
        """
        Run search_camera() for every camera in *galleries*.

        Returns {camera_id: CameraMatch | None}.
        """
        results: dict[str, Optional[CameraMatch]] = {}
        for camera_id, gallery in galleries.items():
            results[camera_id] = self.search_camera(
                query_embedding=query_embedding,
                gallery=gallery,
                camera_id=camera_id,
                source_crop_path=source_crop_path,
            )
        return results
