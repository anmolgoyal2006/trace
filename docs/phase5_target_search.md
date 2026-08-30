# Phase 5 — Target Search (Re-ID Matching)

## Overview

Phase 5 connects the embedding pipeline built in Phases 1–4 to real-world query
scenarios. Its goal is to accept a query image of a person and retrieve the most
similar person crops from the C01 gallery using cosine similarity over OSNet
embeddings.

**Environment split:**

| Step | Where |
|------|-------|
| Phase 5.1 preparation (query creation, metadata, validation) | VS Code |
| Phase 5.2 matching experiment (OSNet inference + cosine similarity) | Google Colab |

The production embeddings (`dataset/embeddings_C01.json`) are stored in Google
Drive and are NOT available in the VS Code environment. No embedding generation
or modification occurs here.

---

## Phase 5.1 — Derived-Query Robustness Test

### Important caveat

> **This is a derived-query robustness test. It is NOT equivalent to using an
> independently captured real-world photograph.**

A separate real-world photograph of a C01 subject is not currently available.
Therefore Phase 5.1 constructs a synthetic query by applying mild, realistic
image-processing transformations to an existing high-quality C01 crop. The
purpose is to verify that the Re-ID pipeline can tolerate common image
degradations (rotation, lighting change, scale shift, blur/compression) while
still retrieving the correct person.

This does **not** prove cross-camera Re-ID performance, and it does **not**
prove that the system can match against a genuinely separate photograph taken
from an independent source.

---

### Source crop selection

The source crop was selected automatically by scoring all crops from the
`dataset/crops_metadata_phase3_final.json` metadata (353 verified crops) using:

```
score = detection_confidence × sqrt(bbox_area)
```

Only crops from tracks with ≥ 5 crops were considered, ensuring enough
same-track alternatives for evaluation. The highest-scoring crop with good
proportions (height > width, minimum 40 × 80 px) was chosen.

| Field | Value |
|-------|-------|
| Source crop path | `dataset/crops_phase3_final/C01_track29_frame0270.jpg` |
| Track ID | 29 |
| Frame | 270 |
| Camera | C01 |
| Detection confidence | 0.8947 |
| Image dimensions | 222 × 634 px |

---

### Gallery composition for evaluation

| Set | Definition | Count |
|-----|-----------|-------|
| **Query** | One transformed variant of the source crop | 1 |
| **Positive candidates** | Other crops with `track_id = 29` (source crop excluded) | 4 |
| **Negative candidates** | All crops with `track_id ≠ 29` | 348 |
| **Total gallery** | Positives + negatives (source crop excluded) | 352 |

The source crop (`C01_track29_frame0270.jpg`) is **excluded from the gallery**
to prevent trivial self-matching. This is enforced by the
`source_crop_excluded_from_gallery = true` flag in every metadata entry.

Same-track positive crops (frames of the same person, available for positive
retrieval evaluation):

- `C01_track29_frame0155.jpg`
- `C01_track29_frame0185.jpg`
- `C01_track29_frame0215.jpg`
- `C01_track29_frame0300.jpg`

---

### Query variants created

All variants are derived exclusively from `C01_track29_frame0270.jpg` using
Pillow image-processing operations. No generative AI is used. No objects are
added or removed. The person's identity is unchanged.

| File | Transformation | Description |
|------|---------------|-------------|
| `test_query_1.jpg` | `original_quality` | High-quality re-save (JPEG quality=95). No geometric or photometric change. Baseline variant. |
| `test_query_1_rotated.jpg` | `rotation` | 5-degree clockwise rotation, bicubic resampling. Simulates slight camera tilt. |
| `test_query_1_lighting.jpg` | `brightness_contrast` | Brightness +15%, contrast +10%. Simulates different ambient lighting. |
| `test_query_1_scaled.jpg` | `crop_scale` | Centre-crop to 90% extent, resized back with Lanczos. Simulates framing/zoom difference. |
| `test_query_1_blur.jpg` | `blur_compression` | Gaussian blur radius=0.8, JPEG quality=70. Simulates lower-resolution or compressed input. |

All query images are stored in `dataset/query_photos/`.

---

### Anti-leakage design

The anti-leakage rule is enforced at three levels:

1. **Metadata flag** — every entry in `query_metadata.json` carries
   `"source_crop_excluded_from_gallery": true`.
2. **Physical separation** — query files live in `dataset/query_photos/`;
   the source crop lives in `dataset/crops_phase3_final/`. They are never
   the same file.
3. **Colab experiment contract** — Phase 5.2 must build its gallery from
   `crops_metadata_phase3_final.json` after filtering out the row whose
   `crop_path` matches `source_crop_path` from `query_metadata.json`.

---

### Files produced by Phase 5.1

```
dataset/query_photos/
    query_metadata.json          ← 5 entries, one per variant
    test_query_1.jpg
    test_query_1_rotated.jpg
    test_query_1_lighting.jpg
    test_query_1_scaled.jpg
    test_query_1_blur.jpg

ai_pipeline/reid/
    validate_query.py            ← standalone validation script
    test_query.py                ← pytest test suite (GPU-free)
```

---

### Files NOT modified by Phase 5.1

The following production files remain completely untouched:

- `dataset/embeddings_C01.json`
- `dataset/detections_C01.json`
- `dataset/tracks_C01.json`
- `dataset/crops_phase3_final/` (all 353 crops)
- `dataset/crops_metadata_phase3_final.json`
- All Phase 1–4 scripts and results

---

### Validation

Run the standalone validation script from the workspace root:

```bash
python ai_pipeline/reid/validate_query.py
```

Expected output ends with:

```
RESULT: PASS  (10/10 checks passed)
```

Run the pytest suite:

```bash
pytest ai_pipeline/reid/test_query.py -v
```

No GPU is required for either command.

---

## Phase 5.2 — OSNet Matching Experiment (Colab only)

> **Not yet implemented in VS Code. Will be executed in Google Colab.**

Phase 5.2 will:

1. Load `dataset/embeddings_C01.json` (353 embeddings from Google Drive).
2. Encode each `dataset/query_photos/test_query_1*.jpg` variant through OSNet
   (the same `osnet_x1_0` checkpoint used in Phase 4).
3. Compute cosine similarity between each query embedding and all gallery
   embeddings, excluding the source crop.
4. Rank gallery crops by similarity.
5. Report Rank-1 / Rank-5 retrieval and whether any of the 4 same-track
   positive crops appear in the top results.

This will be documented in a Phase 5.2 section after Colab execution.

---

## Phase 5.3 — Real Independent-Photo Test (future)

When a genuine external photograph of a C01 subject becomes available, Phase
5.3 will repeat the matching experiment using that image as the query. Only
Phase 5.3 results can be cited as evidence of photo-to-video Re-ID capability.
