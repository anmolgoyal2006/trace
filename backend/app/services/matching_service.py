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

  • OSNet x1_0  (embed.py)              → 512-dim embeddings
  • SOLIDER Swin-Small (embed_solider.py) → 768-dim embeddings

Cross-camera ranking fix (Phase 5.4):
    search_camera_top_k() returns up to `candidate_tracks` CameraMatch
    objects per camera (one per qualifying track, sorted by effective score
    descending).  RouteService evaluates all candidates against spatial and
    temporal context during the greedy walk.

    search_camera() is retained for backwards compatibility.

Face fusion (Step 2):
    When face_gallery and query_face_embedding are both provided, a face
    signal is fused with the body score per track:

        fused = (body_w * body_sim + face_w * face_sim) / (body_w + face_w)

    Face is only used when face_coverage >= 0.3 for that track.
    face_gallery=None is a strict no-op.

KPR part-based fusion (Step 3):
    When kpr_gallery and query_kpr are both provided, part-aware similarity
    (kpr_similarity.part_aware_similarity) replaces the body signal as the
    primary appearance term.  KPR dominates because it is strictly better
    at partial / occluded bodies.

    Triple fusion formula (all three signals active simultaneously):

        fused = (kpr_w * kpr_sim
                 + face_w * face_flag * face_sim
                 + body_w * body_sim)
                /
                (kpr_w + face_w * face_flag + body_w)

        kpr_w    = settings.fusion_kpr_weight   (default 0.60)
        face_w   = settings.fusion_face_weight  (default 0.30)
        body_w   = settings.fusion_body_weight  (default 0.40)
        face_flag = 1 if face_coverage >= 0.3 else 0

    Each signal can be active independently:
        body only          : kpr_gallery=None, face_gallery=None  → unchanged
        body + face        : kpr_gallery=None, face provided
        body + KPR         : face_gallery=None, kpr provided
        body + face + KPR  : all three provided

    kpr_gallery=None AND face_gallery=None reproduces byte-for-byte identical
    output to the pre-fusion codebase. Zero regressions.

Uses:
  - ai_pipeline/reid/similarity.py        → cosine_similarity
  - ai_pipeline/reid/track_aggregation.py → aggregate_by_track
  - ai_pipeline/reid/confidence_scaling.py→ similarity_to_confidence
  - ai_pipeline/reid/face_similarity.py   → face_cosine_similarity,
                                            aggregate_face_by_track
  - ai_pipeline/reid/kpr_similarity.py    → aggregate_kpr_by_track
  - config.no_match_threshold             (0.74)
  - config.fusion_kpr_weight / fusion_face_weight / fusion_body_weight
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

from ai_pipeline.reid.similarity import cosine_similarity                  # noqa: E402
from ai_pipeline.reid.track_aggregation import aggregate_by_track          # noqa: E402
from ai_pipeline.reid.confidence_scaling import similarity_to_confidence   # noqa: E402
from ai_pipeline.reid.face_similarity import (                             # noqa: E402
    face_cosine_similarity,
    aggregate_face_by_track,
)
from ai_pipeline.reid.kpr_similarity import aggregate_kpr_by_track         # noqa: E402

# Minimum face_coverage for a track before its face signal is trusted.
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

    Optional fusion fields (populated only when the relevant gallery is
    provided to search_camera_top_k):

        face_sim         — face cosine similarity, or None (coverage < 0.3
                           or face_gallery not provided).
        face_weight      — weight applied to face_sim in the fusion
                           (0.0 when face not used).
        kpr_sim          — part-aware similarity from KPR, or None
                           (kpr_gallery not provided).
        mean_visible_parts — mean number of mutually visible KPR parts across
                             gallery crops for this track, or None.
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
    # face fusion fields — None when face_gallery not provided
    face_sim: Optional[float] = None
    face_weight: float = 0.0
    # KPR fusion fields — None when kpr_gallery not provided
    kpr_sim: Optional[float] = None
    mean_visible_parts: Optional[float] = None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class MatchingService:
    """
    Cross-camera appearance matching with optional face and KPR fusion.

    Fusion modes (controlled by optional params to search_camera_top_k):

        body only          → kpr_gallery=None, face_gallery=None (default)
        body + face        → face_gallery + query_face_embedding provided
        body + KPR         → kpr_gallery + query_kpr provided
        body + face + KPR  → all four optional params provided

    All modes degrade gracefully: a missing or erroring signal is dropped
    and the remaining active signals are re-normalised. The body signal is
    always present and is never dropped.

    Legacy methods search_camera() and search_all_cameras() call
    search_camera_top_k() with no optional params — identical to pre-fusion
    behaviour, byte-for-byte.
    """

    def __init__(self) -> None:
        self._threshold   = settings.no_match_threshold
        self._body_weight = settings.fusion_body_weight
        self._face_weight = settings.fusion_face_weight
        self._kpr_weight  = settings.fusion_kpr_weight
        self._kpr_vis_thr = settings.kpr_vis_threshold

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
                logger.debug(
                    f"[MatchingService] Skipping crop {rec.get('crop_path')}: {e}"
                )
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
        Return per-track face similarity info.

        Returns dict keyed by track_id:
            face_sim      float | None  — query-vs-track-centroid cosine sim
            face_coverage float         — fraction of crops with detected face
        """
        import numpy as np

        face_track_stats = aggregate_face_by_track(face_gallery)
        result: dict[int | str, dict] = {}

        for track_id, stats in face_track_stats.items():
            coverage = stats["face_coverage"]
            if coverage < _FACE_COVERAGE_MIN or stats["num_face_detected"] == 0:
                result[track_id] = {"face_sim": None, "face_coverage": coverage}
                continue

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

            stacked  = np.array(track_embs, dtype=np.float64)
            centroid = stacked.mean(axis=0)
            norm_c   = float(np.linalg.norm(centroid))
            if norm_c > 0.0:
                centroid /= norm_c

            face_sim = face_cosine_similarity(query_face_embedding, centroid.tolist())
            result[track_id] = {"face_sim": face_sim, "face_coverage": coverage}

        return result

    def _compute_kpr_similarities(
        self,
        query_kpr: dict,
        kpr_gallery: list[dict],
    ) -> dict[int | str, dict]:
        """
        Return per-track KPR part-aware similarity info.

        Returns dict keyed by track_id:
            kpr_sim             float  — max part-aware similarity
            mean_visible_parts  float  — average visible-parts count per crop
        """
        try:
            track_stats = aggregate_kpr_by_track(
                kpr_gallery, query_kpr, vis_threshold=self._kpr_vis_thr
            )
        except Exception as exc:
            logger.warning(f"[MatchingService] aggregate_kpr_by_track failed: {exc}")
            return {}

        result: dict[int | str, dict] = {}
        for track_id, stats in track_stats.items():
            result[track_id] = {
                "kpr_sim":           stats["max_part_aware_sim"],
                "mean_visible_parts": stats["mean_visible_parts"],
            }
        return result

    def _build_camera_match(
        self,
        camera_id: str,
        track_id: int,
        stats: dict,
        crop_sims: list[dict],
        crops_top_k: int,
    ) -> CameraMatch:
        """Build a CameraMatch for one track given its aggregated body stats."""
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
        # ---- Step 2: face fusion (optional) ----------------------------
        face_gallery: Optional[list[dict]] = None,
        query_face_embedding: Optional[list[float]] = None,
        # ---- Step 3: KPR fusion (optional) ----------------------------
        kpr_gallery: Optional[list[dict]] = None,
        query_kpr: Optional[dict] = None,
    ) -> list[CameraMatch]:
        """
        Return up to *candidate_tracks* CameraMatch objects for *camera_id*,
        one per qualifying track, sorted by effective fused score descending.

        When all optional gallery parameters are None (the default) this
        method is byte-for-byte identical to the pre-fusion implementation.

        Fusion modes
        ------------
        Signals present at call time determine the active fusion:

            body only:         kpr_gallery=None, face_gallery=None
            body + face:       face_gallery + query_face_embedding provided
            body + KPR:        kpr_gallery + query_kpr provided
            body + face + KPR: all four provided

        Triple-fusion formula:
            fused = (kpr_w * kpr_sim
                     + face_w * face_flag * face_sim
                     + body_w * body_sim)
                    / (kpr_w + face_w * face_flag + body_w)

            face_flag = 1 if face_coverage >= 0.3, else 0.
            Each inactive signal (gallery=None) contributes 0 weight.

        Args:
            query_embedding:      Body embedding (any dim, matches gallery).
            gallery:              Body embedding records for this camera.
            camera_id:            Camera identifier string.
            source_crop_path:     Crop to exclude for anti-leakage.
            candidate_tracks:     Max qualifying tracks to return (default 3).
            crops_top_k:          Crop records to keep per track (default 5).
            face_gallery:         Face embedding records, or None.
            query_face_embedding: ArcFace query embedding, or None.
            kpr_gallery:          KPR embedding records, or None.
            query_kpr:            KPR query record dict, or None.

        Returns:
            List of CameraMatch sorted by effective score descending.
            Empty list when no track clears no_match_threshold.
        """
        if not gallery:
            logger.warning(f"[MatchingService] Empty gallery for camera {camera_id}")
            return []

        # ---- body similarities (always computed) -------------------------
        crop_sims = self._compute_crop_similarities(
            query_embedding, gallery, source_crop_path
        )
        if not crop_sims:
            logger.warning(
                f"[MatchingService] No valid crops for camera {camera_id}"
            )
            return []

        track_stats = aggregate_by_track(crop_sims)

        # ---- optional face similarities ----------------------------------
        use_face = face_gallery is not None and query_face_embedding is not None
        face_info: dict[int | str, dict] = {}
        if use_face:
            try:
                face_info = self._compute_face_similarities(
                    query_face_embedding, face_gallery  # type: ignore[arg-type]
                )
            except Exception as exc:
                logger.warning(
                    f"[MatchingService] Face similarity failed for {camera_id}: {exc}"
                    " — dropping face signal."
                )
                use_face = False

        # ---- optional KPR similarities -----------------------------------
        use_kpr = kpr_gallery is not None and query_kpr is not None
        kpr_info: dict[int | str, dict] = {}
        if use_kpr:
            try:
                kpr_info = self._compute_kpr_similarities(
                    query_kpr, kpr_gallery  # type: ignore[arg-type]
                )
            except Exception as exc:
                logger.warning(
                    f"[MatchingService] KPR similarity failed for {camera_id}: {exc}"
                    " — dropping KPR signal."
                )
                use_kpr = False

        # ---- effective (possibly fused) score per track ------------------
        def _effective_score(track_id: int | str, body_sim: float) -> float:
            """
            Compute fused score for one track.

            Body-only path:
                Returns body_sim unchanged — identical to pre-fusion output.

            Dual / triple fusion path:
                fused = (kpr_w * kpr_sim + face_w * face_flag * face_sim
                         + body_w * body_sim)
                        / (kpr_w + face_w * face_flag + body_w)
            """
            # Fast path: no optional signals active
            if not use_face and not use_kpr:
                return body_sim

            numerator   = self._body_weight * body_sim
            denominator = self._body_weight

            # KPR term
            if use_kpr:
                k_sim = kpr_info.get(track_id, {}).get("kpr_sim")
                if k_sim is not None:
                    numerator   += self._kpr_weight * k_sim
                    denominator += self._kpr_weight

            # Face term
            if use_face:
                fi    = face_info.get(track_id, {})
                f_sim = fi.get("face_sim")
                if f_sim is not None:
                    # face_flag = 1 (coverage already verified inside
                    # _compute_face_similarities; None means flag=0)
                    numerator   += self._face_weight * f_sim
                    denominator += self._face_weight

            # denominator is always >= body_weight > 0
            return float(numerator / denominator)

        # ---- rank all tracks by effective score --------------------------
        ranked_tracks = sorted(
            track_stats.items(),
            key=lambda kv: _effective_score(kv[0], kv[1]["max_similarity"]),
            reverse=True,
        )

        matches: list[CameraMatch] = []
        for track_id, stats in ranked_tracks:
            effective = _effective_score(track_id, stats["max_similarity"])

            if effective < self._threshold:
                break   # sorted — nothing below qualifies
            if len(matches) >= candidate_tracks:
                break

            match = self._build_camera_match(
                camera_id, track_id, stats, crop_sims, crops_top_k
            )

            # Populate face fields
            if use_face:
                fi       = face_info.get(track_id, {})
                f_sim    = fi.get("face_sim")
                f_weight = self._face_weight if f_sim is not None else 0.0
                match.face_sim    = round(f_sim, 6) if f_sim is not None else None
                match.face_weight = f_weight

            # Populate KPR fields
            if use_kpr:
                ki    = kpr_info.get(track_id, {})
                k_sim = ki.get("kpr_sim")
                match.kpr_sim            = round(k_sim, 6) if k_sim is not None else None
                match.mean_visible_parts = ki.get("mean_visible_parts")

            matches.append(match)

            logger.info(
                f"[MatchingService] camera={camera_id} "
                f"track={track_id} "
                f"body_sim={stats['max_similarity']:.4f} "
                f"kpr_sim={match.kpr_sim} "
                f"face_sim={match.face_sim} "
                f"effective={effective:.4f} "
                f"conf={match.best_confidence:.1f} "
                f"(candidate {len(matches)}/{candidate_tracks})"
            )

        if not matches:
            best_body = (
                ranked_tracks[0][1]["max_similarity"] if ranked_tracks else 0.0
            )
            best_eff = (
                _effective_score(ranked_tracks[0][0], best_body)
                if ranked_tracks else 0.0
            )
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

        Does NOT accept face_gallery / kpr_gallery — fusion is only available
        via search_camera_top_k() to keep this legacy method a strict no-op.
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
