# Phase 5.5 — Rank and Return Top-K Candidates

**Status: IMPLEMENTATION COMPLETE — Real Colab Test Pending**

Phase 5.5 implements the integration layer that connects the existing Re-ID components into a target search system. It accepts a query photo, embeds it using the Phase 4 pipeline, computes cosine similarity against the C01 gallery, and returns top-K candidates with similarity-derived confidence scores.

---

## Implementation Summary

### Files Created

| File | Purpose |
|------|---------|
| `ai_pipeline/reid/target_search.py` | Main integration script with CLI interface |
| `ai_pipeline/reid/test_target_search.py` | GPU-free unit test suite (26 tests) |

### Files NOT Modified

- ✅ `dataset/embeddings_C01.json` — Production embeddings unchanged
- ✅ OSNet model — No changes
- ✅ Preprocessing pipeline — Uses existing Phase 4 pipeline from `embed.py`
- ✅ Cosine similarity — Uses existing `similarity.py`
- ✅ Confidence scaling — Uses existing `confidence_scaling.py`
- ✅ All Phase 1–4 production outputs — Unchanged

---

## Architecture

### Integration Layer Design

Phase 5.5 is an **integration layer only**. It does not redesign the Re-ID pipeline:

1. **Query Embedding** — Uses EXACT Phase 4 preprocessing + OSNet pipeline from `embed.py`
   - Same ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   - Same resize dimensions (256×128)
   - Same OSNet x1_0 model with pretrained weights

2. **Similarity Computation** — Uses existing `cosine_similarity()` from `similarity.py`
   - Same formula: `dot(A, B) / (‖A‖ · ‖B‖)`
   - Same zero-vector guard
   - Same clamping to [-1, 1]

3. **Confidence Conversion** — Uses existing `similarity_to_confidence()` from `confidence_scaling.py`
   - Same Phase 4.5 observed range (min=0.3315..., max=0.9730...)
   - Same min-max scaling formula
   - Same clamping to [0, 100]

---

## CLI Interface

### Usage

```bash
python ai_pipeline/reid/target_search.py \
    --query dataset/query_photos/query_track13/test_query.jpg \
    --embeddings dataset/embeddings_C01.json \
    --query-metadata dataset/query_photos/query_track13/query_metadata.json \
    --top-k 5 \
    --output dataset/target_search/phase5_5_results.json
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--query` | Yes | — | Path to query image |
| `--embeddings` | Yes | — | Path to gallery embeddings JSON |
| `--query-metadata` | No | None | Path to query metadata JSON (for source crop exclusion) |
| `--top-k` | No | 5 | Number of candidates to return |
| `--output` | No | stdout | Path to output JSON file |
| `--config` | No | `ai_pipeline/config.yaml` | Path to config.yaml |

---

## Output Contract

### JSON Schema

The output is a clean JSON array with exactly these fields per candidate:

```json
[
  {
    "camera_id": "C01",
    "timestamp": "10:02:04.89",
    "confidence": 87.3,
    "crop_path": "dataset/crops_phase3_final/..."
  },
  ...
]
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `camera_id` | string | Camera that recorded this observation (e.g., "C01") |
| `timestamp` | string | Wall-clock timestamp in "HH:MM:SS.ss" format |
| `confidence` | float | Similarity-derived confidence score in [0, 100], rounded to 1 decimal |
| `crop_path` | string | Relative path to the crop image file |

### Backend Contract Suitability

This JSON contract is designed for direct handoff to the backend/API component:
- Minimal fields required for display and retrieval
- Numeric confidence for sorting/filtering
- Crop path for image loading
- Timestamp for temporal context
- Camera ID for multi-camera scenarios

---

## Source-Crop Leakage Prevention

### Metadata-Based Exclusion

When a query is derived from a gallery crop (e.g., Phase 5.1 derived queries), the query metadata identifies the source crop:

```json
{
  "query_id": "track13_original",
  "query_path": "dataset/query_photos/query_track13/test_query.jpg",
  "source_crop_path": "dataset/crops_phase3_final/C01_track13_frame0149.jpg",
  "source_crop_excluded_from_gallery": true
}
```

### Exclusion Logic

1. If `--query-metadata` is provided and contains a matching entry
2. And `source_crop_excluded_from_gallery` is `true`
3. Then the exact `source_crop_path` is excluded from similarity computation
4. The exclusion count is reported in the log output

This prevents the query from returning its own source crop as the top result.

---

## Error Handling

### Clear Error Messages

| Error Condition | Handling |
|----------------|----------|
| Missing query image | `FileNotFoundError` with clear message |
| Missing embeddings file | `FileNotFoundError` with clear message |
| Invalid top-k (< 1) | Error message, exit code 1 |
| Unreadable image | `RuntimeError` with PIL error details |
| Invalid embedding dimension | `ValueError` with expected vs actual dimension |
| Empty gallery | `ValueError` with file path |
| Invalid JSON in metadata | Returns None, logs error |

### No Silent Failures

All error conditions are explicitly reported. The script does not silently continue with invalid data.

---

## Unit Tests

### Test Coverage

```
pytest ai_pipeline/reid/test_target_search.py -v

26 passed in 7.20s
```

### Test Categories

| Category | Tests | Coverage |
|----------|-------|----------|
| `TestLoadQueryMetadata` | 4 | Metadata loading, matching, error handling |
| `TestLoadGallery` | 6 | Gallery loading, validation, dimension checking |
| `TestComputeSimilarities` | 4 | Similarity computation, source crop exclusion |
| `TestRankAndConvert` | 5 | Ranking, top-K limit, confidence conversion |
| `TestFormatOutput` | 3 | JSON output schema, field filtering |
| `TestIntegrationScenarios` | 4 | End-to-end pipeline scenarios |

### Key Test Validations

- ✅ Top-K ordering (similarity descending)
- ✅ Top-K limit enforcement
- ✅ Confidence conversion using Phase 4.5 range
- ✅ JSON output schema matches backend contract
- ✅ Source-crop exclusion logic
- ✅ Missing input handling (files, invalid JSON)
- ✅ 512-D embedding compatibility
- ✅ GPU-free (uses synthetic data)

---

## Real Colab Test Instructions

### Prerequisites

The real Colab test requires:

1. **Google Colab environment** with GPU access
2. **Dataset mounted** from Google Drive containing:
   - `dataset/embeddings_C01.json` (353 crops, 512-D embeddings)
   - `dataset/query_photos/query_track13/` (query images and metadata)
   - `dataset/query_photos/query_track16/` (query images and metadata)
3. **torchreid installed**: `!pip install torchreid`

### Test Commands

#### Track 13 Original Query

```bash
python ai_pipeline/reid/target_search.py \
    --query dataset/query_photos/query_track13/test_query.jpg \
    --embeddings dataset/embeddings_C01.json \
    --query-metadata dataset/query_photos/query_track13/query_metadata.json \
    --top-k 5 \
    --output dataset/target_search/track13_original_top5.json
```

#### Track 16 Original Query

```bash
python ai_pipeline/reid/target_search.py \
    --query dataset/query_photos/query_track16/test_query.jpg \
    --embeddings dataset/embeddings_C01.json \
    --query-metadata dataset/query_photos/query_track16/query_metadata.json \
    --top-k 5 \
    --output dataset/target_search/track16_original_top5.json
```

#### Transformed Query (Rotation)

```bash
python ai_pipeline/reid/target_search.py \
    --query dataset/query_photos/query_track13/test_query_rotated.jpg \
    --embeddings dataset/embeddings_C01.json \
    --query-metadata dataset/query_photos/query_track13/query_metadata.json \
    --top-k 5 \
    --output dataset/target_search/track13_rotated_top5.json
```

### Expected Results Format

Each test should produce a JSON file with top-5 candidates:

```json
[
  {
    "camera_id": "C01",
    "timestamp": "10:02:XX.XX",
    "confidence": XX.X,
    "crop_path": "dataset/crops_phase3_final/C01_trackXX_frameXXXX.jpg"
  },
  ...
]
```

### Results Aggregation

After running all tests, combine results into:

```bash
dataset/target_search/phase5_5_results.json
```

This file should contain a summary of all test runs with:
- Query metadata
- Gallery size
- Source crop exclusion count
- Top-K candidates with confidence scores
- Timestamps

---

## Limitations

### Important Disclaimers

1. **Not a calibrated probability**
   - The confidence score is a "similarity-derived confidence score"
   - It is NOT "P(person is the same) = X%"
   - The scaling has not been statistically calibrated against ground-truth identity data

2. **Controlled query experiments only**
   - Phase 5.5 uses derived queries from Phase 5.1 (transformed gallery crops)
   - This does NOT prove real-world missing-person identification
   - Real independent photograph testing is deferred to future phases

3. **Single-camera gallery**
   - Current gallery is C01 only (353 crops)
   - Cross-camera Re-ID (C01 → C02, C01 → C03) is not tested
   - Multi-camera evaluation is deferred to future phases

4. **No temporal filtering**
   - Results are ranked purely by similarity
   - No temporal consistency or trajectory analysis
   - No cross-frame smoothing

---

## Production Data Confirmation

### Unchanged Files

- ✅ `dataset/embeddings_C01.json` — No modifications
- ✅ `dataset/crops_phase3_final/` — No modifications
- ✅ `dataset/crops_metadata_phase3_final.json` — No modifications
- ✅ `dataset/detections_C01.json` — No modifications
- ✅ `dataset/tracks_C01.json` — No modifications
- ✅ All Phase 1–4 outputs — No modifications

### Integration-Only Changes

Phase 5.5 only adds:
- `ai_pipeline/reid/target_search.py` — New integration script
- `ai_pipeline/reid/test_target_search.py` — New test suite
- `docs/phase5_5_results.md` — This documentation

No existing pipeline components were modified.

---

## Next Steps

### After Real Colab Test

1. Update this document with actual results:
   - Track 13 Top-5 candidates and confidence scores
   - Track 16 Top-5 candidates and confidence scores
   - Rotated query results
   - Source crop exclusion counts

2. Analyze results:
   - Does the source track appear in top-K?
   - Are confidence scores reasonable?
   - Do transformed queries perform similarly to originals?

3. Backend handoff:
   - Confirm JSON contract meets backend requirements
   - Provide API integration examples
   - Document deployment considerations

---

## Files Summary

### Created

| File | Lines | Purpose |
|------|-------|---------|
| `ai_pipeline/reid/target_search.py` | ~330 | Main integration script |
| `ai_pipeline/reid/test_target_search.py` | ~520 | Unit test suite |
| `docs/phase5_5_results.md` | This file | Documentation |

### Tests

| Metric | Value |
|--------|-------|
| Total tests | 26 |
| Passed | 26 |
| Failed | 0 |
| Duration | 7.20s |
| GPU required | No (synthetic data) |

### Production Impact

| Component | Status |
|-----------|--------|
| Embeddings | Unchanged |
| OSNet | Unchanged |
| Preprocessing | Unchanged |
| Similarity | Unchanged |
| Confidence scaling | Unchanged |
| Phase 1–4 outputs | Unchanged |
