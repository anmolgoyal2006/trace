"""
matching_service.py — MatchingService
Cross-camera Re-ID matching using appearance embeddings.

Given a query embedding and one or more per-camera gallery files, this
service finds matching tracks in each camera and returns them ranked by
appearance score.

Embedding backbone compatibility
---------------------------------
This service is dimension-agnostic.  cosine_similarity() (similarity.py)
operates on plain NumPy arrays and accepts any vector length, so it works
identically with:

  • OSNet x1_0  (embed.py)         → 512-dim embeddings
  • SOLIDER Swin-Small (embed_solider.py) → 768-dim embeddings

No code changes are required here when switching backbones.  The only
runtime requirement is that the query embedding and gallery embeddings
were produced by the same backbone (same dimension).  Mixed-backbone
galleries will produce a ValueError from cosine_similarity() (dimension
mismatch), which is caught per-crop in _compute_crop_similarities() and
logged as a debug skip.

To switch backbones end-to-end:
  1. Re-run embed_solider.py to produce new gallery files
     (e.g. dataset/embeddings_solider_C01.json).
  2. Update the embeddings_dir / file-naming convention so the query
     router loads the correct gallery files.
  3. Update config.embedding_dim to 768 (or leave it — the service does
     not use embedding_dim directly; it is used by the upload router for
     validation only).

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

Face fusion (Step 2):
    When face_gallery is provided to search_camera_top_k(), a second signal
    is fused with the body appearance score per track:

        fused_score = (body_weight * body_sim + face_weight * face_sim)
                      / (body_weight + face_weight)

    Face similarity is only used when face_coverage >= 0.3 for that track
    (at least 30 % of its crops had a detected face).  Below that threshold
    face_weight is set to 0 and the body score is used unchanged — this
    makes the face path fully additive: it can only help, never silently
    penalise tracks where faces are occluded.

    face_gallery=None (default) reproduces byte-for-byte identical output
    to the pre-face-fusion code.  Existing callers are unaffected.

Uses:
  - ai_pipeline/reid/similarity.py        → cosine_similarity (dimension-agnostic)
  - ai_pipeline/reid/track_aggregation.py → aggregate_by_track
  - ai_pipeline/reid/confidence_scaling.py→ similarity_to_confidence
  - ai_pipeline/reid/face_similarity.py   → face_cosine_similarity,
                                            aggregate_face_by_track
  - config.no_match_threshold (0.74) for filtering low-confidence matches
  - config.fusion_body_weight / fusion_face_weight for face fusion
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
from ai_pipeline.reid.face_similarity import (                     # noqa: E402
    face_cosine_similarity,
    aggregate_face_by_track,
)

# Minimum face_coverage for a track before its face signal is trusted.
# Below this fraction face_weight is dropped to 0 (body-only scoring).
_FACE_COVERAGE_MIN: float = 0.3


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CameraMatch:
    """
    A matching track found in a single camera's gallery.

    All scores are in [0.0, 1.0] unless noted.
    confidence is in [0.0, 100.0].

    Face fusion fields (only populated when face_gallery is provided):
        face_sim    — face cosine similarity for this track, or None when
                      face_coverage < 0.3 or no face was detected.
        face_weight — the weight actually applied to face_sim in the fusion
                      (0.0 when face was not used, settings.fusion_face_weight
                      otherwise).
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
    # face fusion signal — None when face_gallery was not provided
    face_sim: Optional[float] = None
    face_weight: float = 0.0


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class MatchingService:
    """
    Cross-camera appearance matching.

    Primary entry point for the pipeline:
        search_camera_top_k() — returns up to N CameraMatch objects per camera,
                                 sorted by appearance_score (or fused score when
                                 face_gallery is provided) descending.

    Legacy entry point (persons/search, enroll):
        search_camera()       — returns only the single best CameraMatch or None.

    Face fusion:
        Pass face_gallery to search_camera_top_k() to activate body+face fusion.
        face_gallery=None (default) is a strict no-op — produces byte-for-byte
        identical output to the pre-fusion code path.
    """

    def __init__(self) -> None:
        self._threshold   = settings.no_match_threshold
        self._body_weight = settings.fusion_body_weight
        self._face_weight = settings.fusion_face_weight

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

    def _compute_face_similarities(
        self,
        query_face_embedding: list[float],
        face_gallery: list[dict],
    ) -> dict[int | str, dict]:
        """
        Aggregate face embeddings from *face_gallery* by track and return
        per-track face statistics via aggregate_face_by_track().

        Then compute a per-track face similarity score between
        *query_face_embedding* and the mean (centroid) face embedding of
        each gallery track.

        Returns
        -------
        dict keyed by track_id, each value containing:
            face_sim      float | None  — cosine sim between query face and
                                          track mean face; None if coverage low
                                          or no faces detected.
            face_coverage float         — fraction of crops with detected face.

        Only called when face_gallery is not None.
        """
        face_track_stats = aggregate_face_by_track(face_gallery)
        result: dict[int | str, dict] = {}

        for track_id, stats in face_track_stats.items():
            coverage = stats["face_coverage"]

            if coverage < _FACE_COVERAGE_MIN or stats["num_face_detected"] == 0:
                # Not enough face evidence — mark as unusable
                result[track_id] = {"face_sim": None, "face_coverage": coverage}
                continue

            # Build mean face embedding for this track from all detected crops
            track_embs = [
                rec["face_embedding"]
                for rec in face_gallery
                if rec.get("track_id") == track_id
                and rec.get("face_detected")
                and rec.get("face_embedding") is not None
            ]

            if not track_embs:
                result[track_id] = {"face_sim": None, "face_coverage": coverage}
                continue

            import numpy as np
            stacked  = np.array(track_embs, dtype=np.float64)
            centroid = stacked.mean(axis=0)
            norm_c   = float(np.linalg.norm(centroid))
            if norm_c > 0.0:
                centroid /= norm_c

            face_sim = face_cosine_similarity(query_face_embedding, centroid.tolist())
            result[track_id] = {"face_sim": face_sim, "face_coverage": coverage}

        return result

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
        face_gallery: Optional[list[dict]] = None,
        query_face_embedding: Optional[list[float]] = None,
    ) -> list[CameraMatch]:
        """
        Return up to *candidate_tracks* CameraMatch objects for *camera_id*,
        one per qualifying track, sorted by fused score descending.

        When face_gallery is None (default) the method behaves identically to
        the pre-face-fusion version — no new computation, no changed output.

        Args:
            query_embedding:       512-D body embedding from OSNet.
            gallery:               Body embedding records (embeddings_<cam>.json).
            camera_id:             Camera identifier string.
            source_crop_path:      Crop path to exclude (anti-leakage).
            candidate_tracks:      Max tracks to return per camera (default 3).
            crops_top_k:           Crop-level records to keep per track (default 5).
            face_gallery:          Face embedding records (face_embeddings_<cam>.json),
                                   or None to skip face fusion entirely.
            query_face_embedding:  512-D ArcFace embedding of the query person's face,
                                   or None.  Ignored when face_gallery is None.

        Returns:
            List of CameraMatch sorted by effective score descending.
            Empty list when no track clears the no_match_threshold.
        """
        if not gallery:
            logger.warning(f"[MatchingService] Empty gallery for camera {camera_id}")
            return []

        # ---- body similarities (unchanged path) --------------------------
        crop_sims = self._compute_crop_similarities(
            query_embedding, gallery, source_crop_path
        )
        if not crop_sims:
            logger.warning(f"[MatchingService] No valid crops for camera {camera_id}")
            return []

        track_stats = aggregate_by_track(crop_sims)

        # ---- optional face similarities ----------------------------------
        # Only computed when both face_gallery and query_face_embedding are
        # supplied.  If either is absent, the face path is fully skipped and
        # every track gets face_weight=0, face_sim=None.
        use_face = face_gallery is not None and query_face_embedding is not None
        face_info: dict[int | str, dict] = {}
        if use_face:
            try:
                face_info = self._compute_face_similarities(
                    query_face_embedding, face_gallery  # type: ignore[arg-type]
                )
            except Exception as exc:
                # Face pipeline failure must never break the body-only result
                logger.warning(
                    f"[MatchingService] Face similarity failed for {camera_id}: {exc}"
                    " — falling back to body-only scoring."
                )
                use_face = False

        # ---- compute effective (possibly fused) score per track ----------
        def _effective_score(track_id: int | str, body_sim: float) -> float:
            """Return fused score when face is available, else body_sim."""
            if not use_face:
                return body_sim

            fi = face_info.get(track_id, {})
            f_sim = fi.get("face_sim")

            if f_sim is None:
                # No usable face for this track — body only
                return body_sim

            # Weighted fusion: normalise by the sum of active weights
            bw = self._body_weight
            fw = self._face_weight
            fused = (bw * body_sim + fw * f_sim) / (bw + fw)
            return fused

        # ---- sort by effective score descending --------------------------
        ranked_tracks = sorted(
            track_stats.items(),
            key=lambda kv: _effective_score(kv[0], kv[1]["max_similarity"]),
            reverse=True,
        )

        matches: list[CameraMatch] = []
        for track_id, stats in ranked_tracks:
            effective = _effective_score(track_id, stats["max_similarity"])

            if effective < self._threshold:
                break  # list is sorted — nothing below will qualify
            if len(matches) >= candidate_tracks:
                break

            match = self._build_camera_match(
                camera_id, track_id, stats, crop_sims, crops_top_k
            )

            # Populate face fusion fields when face was used
            if use_face:
                fi       = face_info.get(track_id, {})
                f_sim    = fi.get("face_sim")
                f_weight = self._face_weight if f_sim is not None else 0.0
                match.face_sim    = round(f_sim, 6) if f_sim is not None else None
                match.face_weight = f_weight

            matches.append(match)

            logger.info(
                f"[MatchingService] camera={camera_id} "
                f"track={track_id} "
                f"body_sim={stats['max_similarity']:.4f} "
                f"effective={effective:.4f} "
                f"face_sim={match.face_sim} "
                f"conf={match.best_confidence:.1f} "
                f"(candidate {len(matches)}/{candidate_tracks})"
            )

        if not matches:
            best_body = ranked_tracks[0][1]["max_similarity"] if ranked_tracks else 0.0
            best_eff  = _effective_score(
                ranked_tracks[0][0], best_body
            ) if ranked_tracks else 0.0
            logger.info(
                f"[MatchingService] camera={camera_id} "
                f"best_effective={best_eff:.4f} "
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

        Does NOT accept face_gallery — face fusion is only available via
        search_camera_top_k() to keep this legacy method a strict no-op.
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
