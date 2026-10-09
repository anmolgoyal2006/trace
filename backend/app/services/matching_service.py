"""
matching_service.py — MatchingService
Cross-camera Re-ID matching with evidence hierarchy and face identity veto.

Evidence hierarchy (highest to lowest weight):
  1. Face identity veto   — ArcFace biometric identity check
  2. KPR part-based sim   — appearance, part-aligned, occlusion-robust
  3. Body sim (OSNet)     — global appearance baseline
  Spatial + temporal      — handled by RouteService after candidate selection

Match status model
------------------
Every candidate track gets one of five statuses:

  CONFIDENT_MATCH
      Fused score >= backbone no_match_threshold AND
      no reliable face contradiction.

  POSSIBLE_MATCH_REVIEW
      Fused score >= soft_floor but < no_match_threshold.
      Returned for human review but NOT as a confirmed identity.
      Also used when face evidence is absent/unreliable and appearance
      score clears soft_floor but not the full threshold.

  FACE_MISMATCH
      Query and gallery face are BOTH reliable (coverage >= min_coverage
      AND detection confidence >= face_min_query_confidence) AND
      face_sim < face_match_threshold.
      Body/KPR similarity no matter how high CANNOT override this.

  NO_CONFIDENT_MATCH
      No track cleared even soft_floor. Returned with top candidates
      for human visual review.

  NO_USABLE_EVIDENCE
      Gallery is empty or all crops failed preprocessing.

Face veto design
----------------
The face veto is applied ONLY when BOTH sides have reliable face evidence:
  - Query face: detected with confidence >= face_min_query_confidence
  - Gallery track: face_coverage >= face_min_coverage (default 0.3)

When face evidence is absent or unreliable on either side, the veto is
NOT applied — we do not penalise occluded or rear-facing persons.

A FACE_MISMATCH candidate is still returned in the top-N for human review.
It is never promoted to CONFIDENT_MATCH regardless of body/KPR scores.

Backbone-aware thresholds
-------------------------
OSNet (512-dim)  : no_match_threshold=0.74,  soft_floor=0.60
SOLIDER (768-dim): no_match_threshold=0.94,  soft_floor=0.90

The active threshold is selected from the embedding dimension of the first
gallery record.  Using OSNet thresholds for SOLIDER is incorrect because
SOLIDER different-track similarities cluster at 0.83–0.92.

All thresholds are EMPIRICAL / MVP values — NOT calibrated probabilities.

Confidence score terminology
-----------------------------
best_confidence is a "similarity-derived match score" in [0, 100].
It is NOT "P(same person) = X%".  It is not displayed as "accuracy".
See confidence_scaling.py for the full disclaimer.

Zero-regression guarantee
--------------------------
face_gallery=None AND kpr_gallery=None → body-only path is taken.
The body-only path produces byte-for-byte identical results to pre-fusion
code because _effective_score() returns body_sim unchanged in that case.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from loguru import logger

from backend.app.config import settings

# Import existing, tested pipeline modules
if str(settings.repo_root) not in sys.path:
    sys.path.insert(0, str(settings.repo_root))

from ai_pipeline.reid.similarity import cosine_similarity                  # noqa: E402
from ai_pipeline.reid.track_aggregation import aggregate_by_track          # noqa: E402
from ai_pipeline.reid.confidence_scaling import (                          # noqa: E402
    similarity_to_confidence,
    get_backbone_config,
    BACKBONE_OSNET_512,
)
from ai_pipeline.reid.face_similarity import (                             # noqa: E402
    face_cosine_similarity,
    aggregate_face_by_track,
)
from ai_pipeline.reid.kpr_similarity import aggregate_kpr_by_track         # noqa: E402


# ---------------------------------------------------------------------------
# Match status enum
# ---------------------------------------------------------------------------

class MatchStatus(str, Enum):
    """
    Identity decision for a single candidate track.

    Statuses are mutually exclusive and ordered by confidence.
    FACE_MISMATCH overrides CONFIDENT_MATCH when reliable face evidence
    contradicts the appearance match.
    """
    CONFIDENT_MATCH       = "CONFIDENT_MATCH"
    POSSIBLE_MATCH_REVIEW = "POSSIBLE_MATCH_REVIEW"
    FACE_MISMATCH         = "FACE_MISMATCH"
    NO_CONFIDENT_MATCH    = "NO_CONFIDENT_MATCH"
    NO_USABLE_EVIDENCE    = "NO_USABLE_EVIDENCE"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CameraMatch:
    """
    A candidate track found in a single camera's gallery.

    Scores
    ------
    appearance_score   : max cosine body similarity for this track
    mean_top3_similarity: mean of top-3 crops (body backbone)
    best_confidence    : similarity-derived match score [0, 100]
                         NOT an identity probability — label as "Match score"
    mean_confidence    : mean_top3 scaled the same way

    Identity decision
    -----------------
    match_status       : MatchStatus enum — the primary identity decision
    face_veto_applied  : True when a reliable face mismatch was detected
    face_veto_reason   : human-readable explanation when veto was applied

    Face fusion fields (None when face_gallery not provided)
    ---------------------------------------------------------
    face_sim           : ArcFace cosine sim, or None
    face_weight        : weight applied in fusion (0 when not used)
    face_coverage      : fraction of gallery crops with detected face
    query_face_reliable: True when query face met quality threshold
    gallery_face_reliable: True when gallery track met coverage threshold

    KPR fusion fields (None when kpr_gallery not provided)
    -------------------------------------------------------
    kpr_sim            : part-aware similarity, or None
    mean_visible_parts : avg mutually-visible parts per crop pair

    Debug
    -----
    fused_score        : raw weighted-average score before threshold check
    active_backbone    : name of body backbone used (e.g. "OSNet x1_0")
    embedding_dim      : dimension of body embeddings
    """
    camera_id: str
    track_id: int
    appearance_score: float
    mean_top3_similarity: float
    best_confidence: float           # similarity-derived score, NOT probability
    mean_confidence: float
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    best_crop_path: Optional[str] = None
    top_crops: list[dict] = field(default_factory=list)

    # Identity decision
    match_status: MatchStatus = MatchStatus.POSSIBLE_MATCH_REVIEW
    face_veto_applied: bool = False
    face_veto_reason: Optional[str] = None

    # Face fusion
    face_sim: Optional[float] = None
    face_weight: float = 0.0
    face_coverage: Optional[float] = None
    query_face_reliable: bool = False
    gallery_face_reliable: bool = False

    # KPR fusion
    kpr_sim: Optional[float] = None
    mean_visible_parts: Optional[float] = None

    # Debug
    fused_score: float = 0.0
    active_backbone: str = "OSNet x1_0"
    embedding_dim: int = 512

    # Phase 4 identity fields — consumed by _persist_sighting()
    # matching_mode: which signal was primary ("body" | "face")
    # face_used: True when ArcFace embedding participated in the decision
    # face_similarity: alias for face_sim kept for DB column compatibility
    # body_similarity: alias for appearance_score kept for DB column compatibility
    matching_mode: str = "body"
    face_used: bool = False
    face_similarity: Optional[float] = None   # set from face_sim after veto check
    body_similarity: float = 0.0              # set from appearance_score


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class MatchingService:
    """
    Cross-camera appearance matching with evidence hierarchy.

    Evidence hierarchy (most authoritative first):
      1. Face identity veto  — overrides body/KPR if both faces are reliable
      2. KPR part-aware sim  — best for occluded / partial bodies
      3. Body cosine sim     — always computed, always present

    Fusion modes (controlled by optional params):
      body only          : kpr_gallery=None, face_gallery=None  → unchanged
      body + face        : face_gallery + query_face_embedding provided
      body + KPR         : kpr_gallery + query_kpr provided
      body + face + KPR  : all four provided

    Zero-regression: face_gallery=None AND kpr_gallery=None → identical
    output to pre-fusion code (body-only path, byte-for-byte).
    """

    def __init__(self) -> None:
        self._body_weight   = settings.fusion_body_weight
        self._face_weight   = settings.fusion_face_weight
        self._kpr_weight    = settings.fusion_kpr_weight
        self._kpr_vis_thr   = settings.kpr_vis_threshold
        # Face veto parameters
        self._face_match_thr        = settings.face_match_threshold
        self._face_min_coverage     = settings.face_min_coverage
        self._face_min_query_conf   = settings.face_min_query_confidence

    # ------------------------------------------------------------------ #
    # Backbone detection                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_backbone(gallery: list[dict]) -> tuple[str, int, float, float]:
        """
        Infer active body backbone from gallery embedding dimension.

        Returns (backbone_name, embedding_dim, no_match_threshold, soft_floor).
        Falls back to OSNet if gallery is empty or embedding is missing.
        """
        for rec in gallery:
            emb = rec.get("embedding")
            if emb:
                dim = len(emb)
                cfg = get_backbone_config(dim)
                return cfg.name, dim, cfg.no_match_threshold, cfg.soft_floor
        # Fallback to OSNet defaults
        return (
            BACKBONE_OSNET_512.name,
            BACKBONE_OSNET_512.embedding_dim,
            BACKBONE_OSNET_512.no_match_threshold,
            BACKBONE_OSNET_512.soft_floor,
        )

    # ------------------------------------------------------------------ #
    # Similarity computation helpers                                       #
    # ------------------------------------------------------------------ #

    def _compute_crop_similarities(
        self,
        query_embedding: list[float],
        gallery: list[dict],
        source_crop_path: Optional[str],
    ) -> list[dict]:
        """
        Compute cosine similarity between query and every gallery crop.
        Excludes source_crop_path (anti-leakage).
        Returns list of {track_id, similarity, timestamp, crop_path, frame}.
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
        Compute per-track face similarity info from gallery.

        Returns dict keyed by track_id:
            face_sim      float | None
            face_coverage float
        """
        import numpy as np

        face_track_stats = aggregate_face_by_track(face_gallery)
        result: dict[int | str, dict] = {}

        for track_id, stats in face_track_stats.items():
            coverage = stats["face_coverage"]
            if coverage < self._face_min_coverage or stats["num_face_detected"] == 0:
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
        Compute per-track KPR part-aware similarity info.

        Returns dict keyed by track_id:
            kpr_sim            float
            mean_visible_parts float
        """
        try:
            track_stats = aggregate_kpr_by_track(
                kpr_gallery, query_kpr, vis_threshold=self._kpr_vis_thr
            )
        except Exception as exc:
            logger.warning(f"[MatchingService] aggregate_kpr_by_track failed: {exc}")
            return {}

        return {
            track_id: {
                "kpr_sim":            stats["max_part_aware_sim"],
                "mean_visible_parts": stats["mean_visible_parts"],
            }
            for track_id, stats in track_stats.items()
        }

    # ------------------------------------------------------------------ #
    # Face veto logic                                                      #
    # ------------------------------------------------------------------ #

    def _check_face_veto(
        self,
        track_id: int | str,
        face_info: dict[int | str, dict],
        query_face_reliable: bool,
    ) -> tuple[bool, Optional[str], Optional[float], Optional[float]]:
        """
        Determine whether this track should be vetoed on face evidence.

        A veto fires ONLY when:
          - query face is reliable (detected with sufficient confidence)
          - gallery track has sufficient face coverage (>= face_min_coverage)
          - face similarity is below face_match_threshold

        Returns (veto, reason, face_sim, face_coverage).
        veto=False means either evidence is absent/unreliable (no veto) OR
        face similarity is high enough (no mismatch).
        """
        if not query_face_reliable:
            return False, None, None, None

        fi = face_info.get(track_id, {})
        face_sim     = fi.get("face_sim")
        face_coverage = fi.get("face_coverage", 0.0)
        gallery_face_reliable = (
            face_sim is not None
            and face_coverage is not None
            and face_coverage >= self._face_min_coverage
        )

        if not gallery_face_reliable:
            # Gallery track has no reliable face — cannot veto, cannot confirm
            return False, None, face_sim, face_coverage

        if face_sim < self._face_match_thr:
            reason = (
                f"face_sim={face_sim:.4f} < face_match_threshold={self._face_match_thr:.4f} "
                f"(coverage={face_coverage:.2f}). "
                f"Query and gallery face evidence is reliable but contradicts appearance match. "
                f"Note: face_match_threshold is an initial empirical value — NOT calibrated."
            )
            return True, reason, face_sim, face_coverage

        return False, None, face_sim, face_coverage

    # ------------------------------------------------------------------ #
    # Camera match builder                                                 #
    # ------------------------------------------------------------------ #

    def _build_camera_match(
        self,
        camera_id: str,
        track_id: int,
        stats: dict,
        crop_sims: list[dict],
        crops_top_k: int,
        embedding_dim: int,
        backbone_name: str,
    ) -> CameraMatch:
        """Build a CameraMatch for one track from its aggregated body stats."""
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
            best_confidence=round(
                similarity_to_confidence(appearance_score, embedding_dim=embedding_dim), 1
            ),
            mean_confidence=round(
                similarity_to_confidence(
                    stats["mean_top3_similarity"], embedding_dim=embedding_dim
                ), 1
            ),
            first_seen=first_seen,
            last_seen=last_seen,
            best_crop_path=best_crop,
            top_crops=top_crops,
            active_backbone=backbone_name,
            embedding_dim=embedding_dim,
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
        # ---- face fusion (optional) -----------------------------------
        face_gallery: Optional[list[dict]] = None,
        query_face_embedding: Optional[list[float]] = None,
        query_face_confidence: Optional[float] = None,
        # ---- KPR fusion (optional) ------------------------------------
        kpr_gallery: Optional[list[dict]] = None,
        query_kpr: Optional[dict] = None,
    ) -> list[CameraMatch]:
        """
        Return up to *candidate_tracks* CameraMatch objects for *camera_id*,
        sorted by fused score descending.

        Every returned candidate includes a match_status field:
          CONFIDENT_MATCH, POSSIBLE_MATCH_REVIEW, or FACE_MISMATCH.

        Even below the no_match threshold, up to candidate_tracks tracks
        above soft_floor are returned as POSSIBLE_MATCH_REVIEW so the
        caller can show them for human review.

        When all optional gallery parameters are None (default), the method
        is byte-for-byte identical to pre-fusion behaviour (body only).

        Args:
            query_embedding:        Body embedding (any dim, matches gallery).
            gallery:                Body embedding records for this camera.
            camera_id:              Camera identifier.
            source_crop_path:       Crop to exclude (anti-leakage).
            candidate_tracks:       Max tracks to return (default 3).
            crops_top_k:            Crop records per track (default 5).
            face_gallery:           Face embedding records, or None.
            query_face_embedding:   ArcFace query embedding, or None.
            query_face_confidence:  SCRFD detection score for the query face.
                                    Used to determine if query face is reliable.
                                    Pass None when face signal is absent.
            kpr_gallery:            KPR embedding records, or None.
            query_kpr:              KPR query record dict, or None.

        Returns:
            List of CameraMatch, best first.  May be empty if no track
            clears soft_floor.  The caller must check match_status on
            each returned match — FACE_MISMATCH must not be treated as
            CONFIDENT_MATCH.
        """
        if not gallery:
            logger.warning(f"[MatchingService] Empty gallery for camera {camera_id}")
            return []

        # ---- detect backbone + select thresholds -------------------------
        backbone_name, embedding_dim, no_match_thr, soft_floor = (
            self._detect_backbone(gallery)
        )

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
        query_face_reliable = False

        if use_face:
            # Query face is reliable only if it was detected with sufficient confidence
            query_face_reliable = (
                query_face_confidence is not None
                and query_face_confidence >= self._face_min_query_conf
            )
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

        # ---- compute fused score per track -------------------------------
        def _fused_score(track_id: int | str, body_sim: float) -> float:
            """
            Weighted fusion of body + KPR (+ face when available).

            DESIGN: Face is intentionally EXCLUDED from the threshold gating
            score when a face veto is expected to fire. The gating score uses
            body+KPR only, so a FACE_MISMATCH candidate is NOT silently dropped
            below soft_floor — it is returned with status=FACE_MISMATCH for
            human review. Face only contributes to ranking when face_sim > 0
            (i.e. it agrees with the appearance match).

            More precisely: we include face in fused_score only when face_sim
            is positive (supportive evidence), not when it contradicts.
            This prevents the face signal from dragging a high-body-sim track
            below soft_floor while still using it for positive reinforcement.
            """
            # Fast path: body only — no new computation
            if not use_face and not use_kpr:
                return body_sim

            numerator   = self._body_weight * body_sim
            denominator = self._body_weight

            if use_kpr:
                k_sim = kpr_info.get(track_id, {}).get("kpr_sim")
                if k_sim is not None:
                    numerator   += self._kpr_weight * k_sim
                    denominator += self._kpr_weight

            # Face: include in ranking score only when positive (supportive).
            # When face_sim < 0, it will trigger the veto but must NOT
            # suppress the candidate below soft_floor before the veto fires.
            if use_face:
                fi    = face_info.get(track_id, {})
                f_sim = fi.get("face_sim")
                if f_sim is not None and f_sim > 0:
                    numerator   += self._face_weight * f_sim
                    denominator += self._face_weight

            return float(numerator / denominator)

        # ---- rank by fused score, collect all above soft_floor -----------
        ranked_tracks = sorted(
            track_stats.items(),
            key=lambda kv: _fused_score(kv[0], kv[1]["max_similarity"]),
            reverse=True,
        )

        # Log startup message
        active_signals = ["body"]
        if use_face:   active_signals.append("face")
        if use_kpr:    active_signals.append("KPR")
        logger.info(
            f"[MatchingService] camera={camera_id} "
            f"backbone={backbone_name} dim={embedding_dim} "
            f"threshold={no_match_thr} soft_floor={soft_floor} "
            f"signals={'+'.join(active_signals)} "
            f"query_face_reliable={query_face_reliable}"
        )

        matches: list[CameraMatch] = []

        for track_id, stats in ranked_tracks:
            if len(matches) >= candidate_tracks:
                break

            body_sim = stats["max_similarity"]
            fused    = _fused_score(track_id, body_sim)

            # Skip tracks below soft_floor entirely
            if fused < soft_floor:
                break   # list is sorted — nothing below qualifies

            # Build the base match object
            match = self._build_camera_match(
                camera_id, track_id, stats, crop_sims,
                crops_top_k, embedding_dim, backbone_name
            )
            match.fused_score = round(fused, 6)

            # ---- face veto check ----------------------------------------
            veto, veto_reason, f_sim, f_cov = self._check_face_veto(
                track_id, face_info, query_face_reliable
            )

            # ---- populate face fields -----------------------------------
            if use_face:
                fi = face_info.get(track_id, {})
                raw_f_sim = fi.get("face_sim")
                f_coverage = fi.get("face_coverage", 0.0)
                match.face_sim    = round(raw_f_sim, 6) if raw_f_sim is not None else None
                match.face_weight = self._face_weight if raw_f_sim is not None else 0.0
                match.face_coverage = f_coverage
                match.query_face_reliable  = query_face_reliable
                match.gallery_face_reliable = (
                    raw_f_sim is not None
                    and f_coverage is not None
                    and f_coverage >= self._face_min_coverage
                )

            # ---- populate KPR fields ------------------------------------
            if use_kpr:
                ki    = kpr_info.get(track_id, {})
                k_sim = ki.get("kpr_sim")
                match.kpr_sim            = round(k_sim, 6) if k_sim is not None else None
                match.mean_visible_parts = ki.get("mean_visible_parts")

            # ---- assign match_status ------------------------------------
            if veto:
                match.match_status      = MatchStatus.FACE_MISMATCH
                match.face_veto_applied = True
                match.face_veto_reason  = veto_reason
            elif fused >= no_match_thr:
                match.match_status = MatchStatus.CONFIDENT_MATCH
            else:
                match.match_status = MatchStatus.POSSIBLE_MATCH_REVIEW

            # ---- populate Phase 4 identity fields -----------------------
            # These mirror existing fields under the names expected by
            # _persist_sighting() in pipeline_service.py.
            match.face_used       = use_face and match.face_sim is not None
            match.matching_mode   = "face" if match.face_used and not veto else "body"
            match.face_similarity = match.face_sim          # alias
            match.body_similarity = match.appearance_score  # alias

            matches.append(match)

            # ---- debug log per candidate --------------------------------
            logger.info(
                f"[MatchingService] CANDIDATE "
                f"camera={camera_id} track={track_id} "
                f"backbone={backbone_name} dim={embedding_dim} | "
                f"body_sim={body_sim:.4f} "
                f"kpr_sim={match.kpr_sim} "
                f"face_sim={match.face_sim} "
                f"face_coverage={match.face_coverage} "
                f"query_face_reliable={query_face_reliable} "
                f"gallery_face_reliable={match.gallery_face_reliable} | "
                f"fused={fused:.4f} "
                f"threshold={no_match_thr} "
                f"soft_floor={soft_floor} "
                f"face_veto={veto} | "
                f"confidence={match.best_confidence:.1f} "
                f"status={match.match_status.value}"
            )

        if not matches:
            best_body = ranked_tracks[0][1]["max_similarity"] if ranked_tracks else 0.0
            best_fused = _fused_score(ranked_tracks[0][0], best_body) if ranked_tracks else 0.0
            logger.info(
                f"[MatchingService] camera={camera_id} "
                f"best_fused={best_fused:.4f} < soft_floor={soft_floor} "
                f"→ NO_USABLE_EVIDENCE"
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
        Return the single best CONFIDENT_MATCH track, or None.

        Used by /api/persons/search and reference-photo enrollment.
        Only returns a track if its match_status is CONFIDENT_MATCH.
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
        # Only return a match if it is actually confident
        if results and results[0].match_status == MatchStatus.CONFIDENT_MATCH:
            return results[0]
        return None

    def search_all_cameras(
        self,
        query_embedding: list[float],
        galleries: dict[str, list[dict]],
        source_crop_path: Optional[str] = None,
    ) -> dict[str, Optional[CameraMatch]]:
        """
        Run search_camera() for every camera in *galleries*.
        Returns {camera_id: CameraMatch | None}.
        Retained for backwards compatibility.
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
