# Phase 4 — Face-First Identity Matching Policy

**Status: IMPLEMENTATION COMPLETE**

---

## Summary

TRACE previously selected the final identity score using a weighted triple
fusion of three signals:

```
identity_score = body × 0.40 + face × 0.30 + KPR × 0.60   (old)
```

This has been replaced by a **conditional face-first policy** with no blending:

```
usable face present
    → matching_mode = "face"
    → identity_score = face_similarity ONLY

no usable face
    → matching_mode = "body"
    → identity_score = body / part Re-ID score ONLY
```

---

## Motivation

Clothing and body appearance can change between cameras (jacket removed,
bag added) or over time.  A sufficiently clear face provides a more
identity-specific signal that is independent of clothing.

Weighted fusion with body features risks allowing high clothing similarity
to artificially inflate a face-only identity decision, or vice versa.  The
two signals answer different questions:

| Signal | Answers |
|--------|---------|
| Face (ArcFace) | Is this the same *person*? |
| Body (OSNet / SOLIDER) | Does this *appearance* look similar? |

Mixing them produces a score that answers neither question cleanly.

---

## What Changed

### Files Modified

| File | Change |
|------|--------|
| `ai_pipeline/reid/matching.py` | Implements `select_identity_score()` and `resolve_match_mode()` — the two functions that enforce the conditional policy |
| `ai_pipeline/reid/track_aggregation.py` | `aggregate_by_track()` now computes `face_coverage` and `face_primary` per track; `resolve_track_identity()` patched to handle non-numeric `face_sim` safely |
| `backend/app/services/matching_service.py` | `_build_camera_match()` now takes `identity_score` as an explicit parameter (previously used `stats["max_similarity"]` — pure body — even in face mode); `__init__` no longer loads dead fusion weight settings |
| `backend/app/models/orm.py` | `Sighting` table: added `matching_mode`, `face_used`, `face_similarity`, `body_similarity`, `identity_score` columns |
| `backend/app/models/schemas.py` | `SightingOut`: added the same five fields with safe defaults |
| `backend/app/services/pipeline_service.py` | `_persist_sighting()` writes all five new fields from `CameraMatch` |
| `backend/app/config.py` | Removed `fusion_body_weight`, `fusion_face_weight`, `fusion_kpr_weight` settings; replaced with a comment block explaining the removal |

### What Was NOT Changed

- OSNet, ArcFace, KPR, SOLIDER model code — untouched
- BoT-SORT tracker — untouched
- YOLOv8n detector — untouched
- Route reconstruction — untouched
- No-match threshold (0.74) — untouched
- KPR / SOLIDER code — kept; KPR still contributes in body mode

---

## Face Usability Rule

A face is considered **usable** for a track when both conditions hold:

1. `face_sim` is a valid float in `[-1.0, 1.0]` (not None, not NaN, not out of range)
2. `face_coverage >= 0.3` — at least 30 % of the track's crops contain a
   detected face embedding

These checks are implemented in:

- `resolve_match_mode()` in `ai_pipeline/reid/matching.py`
- `aggregate_by_track()` in `ai_pipeline/reid/track_aggregation.py`
  (computes `face_primary` flag)

---

## Track Aggregation Policy

For a multi-crop track the face-first decision is made at the **track level**,
not per-crop.  The policy:

```
face_coverage = (crops with face_detected=True) / (total crops)

if face_coverage >= 0.3 AND at least one valid face_sim:
    face_primary = True
    matching_mode = "face"
    identity_score = query-vs-track-centroid ArcFace cosine similarity

else:
    face_primary = False
    matching_mode = "body"
    identity_score = max body cosine similarity across all crops
```

This means a track that has a face in 3 out of 5 crops (coverage = 0.60)
uses face-only matching.  A track with 1 out of 5 (coverage = 0.20) falls
back to body Re-ID.

---

## API Output Fields

Every `SightingOut` object now includes:

```json
{
    "camera_id": "C02",
    "track_id": 17,
    "matching_mode": "face",
    "face_used": true,
    "face_similarity": 0.91,
    "body_similarity": 0.43,
    "identity_score": 0.91,
    "appearance_score": 0.91,
    "best_confidence": 87.3
}
```

Body fallback example:

```json
{
    "camera_id": "C03",
    "track_id": 21,
    "matching_mode": "body",
    "face_used": false,
    "face_similarity": null,
    "body_similarity": 0.82,
    "identity_score": 0.82,
    "appearance_score": 0.82,
    "best_confidence": 77.6
}
```

`appearance_score` equals `identity_score` — the same value, kept for
backwards compatibility with existing route/dashboard code.

---

## Limitations

Face-only matching fails when the face is:

- Too small (below the SCRFD minimum face size, default 20 px)
- Blurred or low resolution
- Occluded (mask, hat, other objects)
- Turned away (profile or rear view)
- Simply not present in the crop

In all these cases `face_detected = False` in the face gallery and
`face_coverage` drops below 0.30, triggering the body Re-ID fallback
automatically.

**Do not claim this improves accuracy.**  No labelled multi-camera
evaluation has been performed to measure whether face-first identity
matching performs better or worse than the old triple fusion on real
TRACE footage.  The policy is theoretically sounder, but empirical
validation on labelled data is required before making accuracy claims.

---

## Tests Added (Phase 5)

File: `ai_pipeline/reid/test_face_first_matching.py`  
32 deterministic tests — no video, no GPU, no real models required.

| Test | Scenario | Expected |
|------|----------|----------|
| A | face=0.92, body=0.40, KPR=0.50 | mode=face, score=0.92 |
| B | face=0.45, body=0.95, KPR=0.90 | mode=face, score=0.45 (body cannot override) |
| C | face=unavailable, body=0.84 | mode=body, score=0.84 |
| D | face detector box but invalid embedding | mode=body, no crash |
| E | 5 crops: 3 face + 2 body | face_primary=True when coverage ≥ 0.3 |
| F | query has face, gallery has no valid faces | body fallback, no crash |
| G | no-match threshold still at 0.74 | threshold unchanged |
| H | real gallery embeddings_C01.json | face-first logic runs cleanly |

Run with:

```powershell
cd D:\trace
.\venv\Scripts\Activate.ps1
python -m pytest ai_pipeline/reid/test_face_first_matching.py -v
```

---

## Startup

```powershell
cd D:\trace
.\venv\Scripts\Activate.ps1
python run.py
```

- Dashboard: http://localhost:8000
- API docs:  http://localhost:8000/api/docs
