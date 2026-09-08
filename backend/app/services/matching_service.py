"""
matching_service.py — MatchingService
Cross-camera Re-ID matching using OSNet embeddings.

Given a query embedding and one or more per-camera gallery files, this
service finds matching tracks in each camera and returns them ranked by
appearance score.

Cross-camera ranking fix (Phase 5.4):
    The original implementation returned only the single best-scoring track
    per camera.  This caused the route walk in RouteService to commit to the
    wrong track when the top-appearance-score track was not temporally or
    spatially consistent with the rest of the route.

    search_camera_top_k() now returns up to `candidate_tracks` CameraMatch
    objects per camera (one per qualifying track, sorted by appearance_score
    descending).  RouteService can then evaluate all candidates against the
    spatial+temporal context and pick the genuinely best one per camera
    during the greedy walk — not the one that merely scored highest in
    isolation.

    The original search_camera() is retained unchanged for backwards
    compatibility (persons/search endpoint, enroll flow).

Uses:
  - ai_pipeline/reid/similarity.py        → cosine_similarity
  - ai_pipeline/reid/track_aggregation.py → aggregate_by_track
  - ai_pipeline/reid/confidence_scaling.py→ similarity_to_confidence
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
    A matching track found in a single camera's gallery.

    All scores are in [0.0, 1.0] unless noted.
    confidence is in [0.0, 100.0].
    """
    camera_id: str
    track_id: int
    appearance_score: float          # max cosine similarity for this track
    mean_top3_similarity: float      # mean of top-3 crops for this track
    best_confidence: float           # similarity_to_confidence(appearance_score), [0,100]
    mean_confidence: float           # similarity_to_confidence(mean_top3), [0,100]
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    best_crop_path: Optional[str] = None
    # crop-level records for this track (sorted by similarity desc)
    top_crops: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class MatchingService:
    """
    Cross-camera appearance matching.

    Primary entry point for the pipeline:
        search_camera_top_k() — returns up to N CameraMatch objects per camera,
                                 sorted by appearance_score descending.

    Legacy entry point (persons/search, enroll):
        search_camera()       — returns only the single best CameraMatch or None.
    """

    def __init__(self) -> None:
        self._threshold = settings.no_match_threshold

    # ------------------------------------------------------------------ #
    # Shared internals                                                     #
    # ------------------------------------------------------------------ #

    def _compute_crop_similarities(
        self,
        query_embedding: list[float],
        gallery: list[dict],
        source_crop_path: Optional[str],
    ) -> list[dict]:
        """
        Compute cosine similarity between query and every gallery crop.

        Returns a flat list of dicts with keys:
            track_id, similarity, timestamp, crop_path, frame
        Crops matching source_crop_path are excluded (anti-leakage).
        """
        crop_sims: list[dict] = []
        for rec in gallery:
            if source_crop_path and rec.get("crop_path") == source_crop_path:
                continue
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
        return crop_sims

    def _build_camera_match(
        self,
        camera_id: str,
        track_id: int,
        stats: dict,
        crop_sims: list[dict],
        crops_top_k: int,
    ) -> CameraMatch:
        """Build a CameraMatch for one track given its aggregated stats."""
        track_crops = sorted(
            [c for c in crop_sims if c["track_id"] == track_id],
            key=lambda c: c["similarity"],
            reverse=True,
        )
        top_crops = track_crops[:crops_top_k]

        all_timestamps = [c["timestamp"] for c in track_crops if c.get("timestamp")]
        first_seen = min(all_timestamps) if all_timestamps else None
        last_seen  = max(all_timestamps) if all_timestamps else None
        best_crop  = top_crops[0]["crop_path"] if top_crops else None

        appearance_score = stats["max_similarity"]

        return CameraMatch(
            camera_id=camera_id,
            track_id=int(track_id),
            appearance_score=round(appearance_score, 6),
            mean_top3_similarity=round(stats["mean_top3_similarity"], 6),
            best_confidence=round(similarity_to_confidence(appearance_score), 1),
            mean_confidence=round(
                similarity_to_confidence(stats["mean_top3_similarity"]), 1
            ),
            first_seen=first_seen,
            last_seen=last_seen,
            best_crop_path=best_crop,
            top_crops=top_crops,
        )

    # ------------------------------------------------------------------ #
    # Primary pipeline entry point                                         #
    # ------------------------------------------------------------------ #

    def search_camera_top_k(
        self,
        query_embedding: list[float],
        gallery: list[dict],
        camera_id: str,
        source_crop_path: Optional[str] = None,
        candidate_tracks: int = 3,
        crops_top_k: int = 5,
    ) -> list[CameraMatch]:
        """
        Return up to *candidate_tracks* CameraMatch objects for *camera_id*,
        one per qualifying track, sorted by appearance_score descending.

        All returned matches have appearance_score >= no_match_threshold.
        Returns an empty list when no track clears the threshold.

        Args:
            query_embedding:   512-D float list from OSNet.
            gallery:           Embedding records from embeddings_<cam>.json.
            camera_id:         Camera identifier.
            source_crop_path:  Crop to exclude for anti-leakage.
            candidate_tracks:  Max number of tracks to return (default 3).
            crops_top_k:       Number of crop-level records to keep per track.

        Returns:
            List of CameraMatch, best-appearance first.  Empty → no match.
        """
        if not gallery:
            logger.warning(f"[MatchingService] Empty gallery for camera {camera_id}")
            return []

        crop_sims = self._compute_crop_similarities(
            query_embedding, gallery, source_crop_path
        )
        if not crop_sims:
            logger.warning(f"[MatchingService] No valid crops for camera {camera_id}")
            return []

        # Aggregate per-track stats
        track_stats = aggregate_by_track(crop_sims)

        # Sort all tracks by max_similarity descending
        ranked_tracks = sorted(
            track_stats.items(),
            key=lambda kv: kv[1]["max_similarity"],
            reverse=True,
        )

        matches: list[CameraMatch] = []
        for track_id, stats in ranked_tracks:
            if stats["max_similarity"] < self._threshold:
                break  # list is sorted — nothing below here will qualify
            if len(matches) >= candidate_tracks:
                break

            match = self._build_camera_match(
                camera_id, track_id, stats, crop_sims, crops_top_k
            )
            matches.append(match)

            logger.info(
                f"[MatchingService] camera={camera_id} "
                f"track={track_id} "
                f"sim={stats['max_similarity']:.4f} "
                f"conf={match.best_confidence:.1f} "
                f"(candidate {len(matches)}/{candidate_tracks})"
            )

        if not matches:
            logger.info(
                f"[MatchingService] camera={camera_id} "
                f"best_sim={ranked_tracks[0][1]['max_similarity']:.4f} "
                f"< threshold={self._threshold} → no confident match"
            )

        return matches

    # ------------------------------------------------------------------ #
    # Legacy single-best entry point (kept for backwards compatibility)   #
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
        Find the single best matching track in *gallery* for *query_embedding*.

        Returns None if no track exceeds the no-match threshold.
        Used by the /api/persons/search endpoint and reference-photo enrollment.
        For the main query pipeline use search_camera_top_k() instead.
        """
        results = self.search_camera_top_k(
            query_embedding=query_embedding,
            gallery=gallery,
            camera_id=camera_id,
            source_crop_path=source_crop_path,
            candidate_tracks=1,
            crops_top_k=top_k,
        )
        return results[0] if results else None

    def search_all_cameras(
        self,
        query_embedding: list[float],
        galleries: dict[str, list[dict]],
        source_crop_path: Optional[str] = None,
    ) -> dict[str, Optional[CameraMatch]]:
        """
        Run search_camera() for every camera in *galleries*.

        Returns {camera_id: CameraMatch | None}.
        Retained for backwards compatibility; pipeline uses search_camera_top_k.
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
