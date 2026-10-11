# Commit History Notes

This file documents the actual work done in commits that had unclear or missing messages.
All commits are by ANMOL GOYAL.

---

## commit `e1cc23d` — Oct 2, 2026 (titled "risk")

**What was actually done:**

- Added entire `docs/assignment2/` folder for SE Lab Assignment 2:
  - `SRS.md` — Software Requirements Specification (501 lines)
  - `REPORT.md` — Full project report (1222 lines)
  - DFD Level 0, 1, 2 (Matching, Route, Video Processing) as `.mmd` + rendered PNGs
  - UML diagrams: Activity, Class, Component, Sequence, Use Case (`.puml` + rendered PNGs)
  - `generate_pdf.py`, `render_diagrams.py` — diagram generation scripts
  - `traceability.md` — requirements traceability matrix
- Added `Trace_SE_Lab_Assignment_2.pdf` — compiled assignment submission
- Added `docs/phase4_identity_policy.md` — identity matching policy documentation
- Added `ai_pipeline/reid/matching.py` — new face-first matching pipeline module
- Added `ai_pipeline/reid/test_face_first_matching.py` — 400-line test suite for face-first matching
- Expanded `ai_pipeline/reid/test_track_aggregation.py` (+293 lines)
- Extended `ai_pipeline/reid/track_aggregation.py` (+164 lines)
- Extended `ai_pipeline/reid/embed.py` (+31 lines)
- Updated `backend/app/main.py` (+60 lines)
- Updated `backend/app/models/orm.py` and `schemas.py` — new fields
- Updated `backend/app/routers/queries.py` — minor additions
- Updated `backend/app/services/pipeline_service.py` — minor additions
- Rewrote `frontend/src/app.js` (288 lines changed)

---

## commit `e33353a` — Oct 2, 2026 (titled "risk1")

**What was actually done:**

- Added `_dev_app.js` — 1,925-line development backup/snapshot of `app.js` saved before further edits
- Updated `frontend/src/app.js` (+295 lines changed) — continued frontend iteration after dev snapshot
- Fixed `ai_pipeline/reid/embed.py` (+21 lines) — minor embed pipeline fixes
- Fixed `ai_pipeline/reid/test_crop_extractor.py`, `test_embed.py`, `test_similarity.py`, `test_target_search.py` — import and compatibility fixes for Python 3.9

---

## commit `72d28a4` — Sep 9, 2026 (titled "fix")

**What was actually done:**

- Major overhaul of `backend/app/services/matching_service.py` (256 lines changed) — rewrote multi-camera vector similarity search logic
- Major refactor of `backend/app/services/route_service.py` (212 lines changed) — rewrote spatial-temporal route reconstruction
- Significant updates to `backend/app/services/pipeline_service.py` (+80 lines) — pipeline orchestration fixes
- Fixed `ai_pipeline/tracking/track.py` (+16 lines) — tracker integration bug fixes
- Updated `run.py` — startup script fixes

---

## commit `a9b0e9f` — Sep 14, 2026 (titled "frontend")

**What was actually done:**

- Major expansion of `frontend/src/app.js` (+288 lines) — added query submission flow, route display, person enrollment UI
- Updated `frontend/index.html` (+108 lines) — new UI sections and layout changes
- Added `frontend/src/styles.css` (+141 lines) — new styles for query results, route timeline, camera detail panels
- Added `backend/app/routers/cameras.py` (+90 lines) — camera adjacency graph endpoints
- Added `dataset/camera_graph.json` — camera topology/adjacency data (60 lines)
- Updated `backend/app/models/schemas.py` (+21 lines) — new Pydantic DTO fields
- Added `frontend/src/api.js` minor addition
- Added 7 new query image files to the dataset

---

## commit `611f9c4` — Sep 16, 2026 (titled "fix")

**What was actually done:**

- Re-ran the C01 crop extraction pipeline on updated video footage
- Added ~60 new person crop JPEGs for C01 tracks 1–37 (multiple frames per track)
- Regenerated `dataset/tracks_C01.json` — full re-run output (78k lines changed)
- Updated `dataset/crops_metadata_C01.json` — refreshed crop metadata (5,182 lines changed)
- Updated `dataset/crop_quality_report.json` — refreshed quality scores (598 lines)

---

## commit `19ff357` — Sep 20, 2026 (titled "fix")

**What was actually done:**

- `backend/app/services/pipeline_service.py` (+510 lines) — wired face embedding and KPR embedding steps into the upload processing pipeline
- `backend/app/routers/upload.py` (+207 lines) — updated upload router to trigger face + KPR embedding after video processing
- Regenerated `dataset/embeddings_C01.json` — trimmed and refreshed OSNet gallery embeddings
- Added 17 new query JPEG images to the dataset queries folder

---

## commit `6dfca50` — Sep 20, 2026 (titled "fix")

**What was actually done:**

- Added 4 new query JPEG images to the dataset queries folder
- No code changes — pure dataset addition
