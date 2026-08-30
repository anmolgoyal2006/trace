# Phase 5.7 — No-Good-Match Handling

**Status: IMPLEMENTATION COMPLETE — Real-World Test Pending**

Phase 5.7 implements a no-match threshold to prevent the system from returning low-confidence candidates. If the best candidate's similarity is below the configured threshold, the system returns an explicit "no_confident_match" result instead of blindly returning Top-K candidates.

---

## Threshold Selection

### Chosen MVP Threshold

| Metric | Value |
|--------|-------|
| **Similarity threshold** | 0.70 |
| **Corresponding confidence** | 57.4 |
| **Location in config** | `config.yaml` → `reid.search.no_match_similarity_threshold` |

### Evidence Supporting the Choice

**Phase 4.5 Similarity Distributions:**

| Distribution | Min | Max | Mean | Median |
|--------------|-----|-----|------|--------|
| Same-track | 0.4355 | 0.9730 | 0.7833 | 0.8017 |
| Different-track | 0.3315 | 0.7996 | 0.6109 | 0.6141 |

**Phase 5.6 Validation Results:**

| Query | Best Similarity | Best Confidence |
|-------|-----------------|-----------------|
| Track 13 original | ~0.8145 | 75.3 |
| Track 16 original | ~0.8878 | 86.7 |
| Track 13 rotated | ~0.801 | 72.6 |

### Rationale

The threshold of 0.70 was chosen because:

1. **Sits between different-track mean (0.61) and max (0.80)**
   - Above most different-track similarities
   - Below the highest different-track similarity

2. **Below all known correct matches (0.80-0.89)**
   - Track 13: 0.8145 (75.3 confidence)
   - Track 16: 0.8878 (86.7 confidence)
   - Track 13 rotated: 0.801 (72.6 confidence)

3. **Conservative enough to reduce false positives**
   - Threshold is higher than different-track mean
   - Should filter out many visually dissimilar candidates

4. **Not too conservative to reject valid matches**
   - All known correct matches are above threshold
   - Provides margin for transformed queries

### Why This is Only an MVP Threshold

- **Based on single-camera data only** — C01 only, no cross-camera evaluation
- **Small sample size** — Only 3 validation queries
- **Derived queries only** — Not independent photograph evaluation
- **No statistical calibration** — Not based on ground-truth identity data
- **May need adjustment** for real-world deployment with diverse query types

### Limitations

1. **Not a calibrated identity probability**
   - The threshold is based on cosine similarity distributions
   - It is NOT "P(person is the same) = X%"
   - No ground-truth identity validation

2. **Single-camera bias**
   - Threshold derived from C01-only data
   - Cross-camera performance unknown
   - May need adjustment for multi-camera scenarios

3. **Small validation sample**
   - Only 3 queries used for validation
   - Not statistically significant
   - May not generalize to all query types

4. **Derived query limitation**
   - All validation queries are transformed gallery crops
   - Independent photograph behavior unknown
   - Real-world query behavior unknown

---

## Configuration

### Config.yaml

```yaml
reid:
  search:
    no_match_similarity_threshold: 0.70  # MVP threshold for Phase 5.7 no-match handling
                                        # Chosen based on Phase 4.5 different-track distribution
                                        # and Phase 5.6 validation results.
                                        # Corresponds to ~57.4 scaled confidence score.
                                        # If best candidate similarity < threshold, return no_confident_match.
```

### Configuration Loading

The threshold is loaded from `config.yaml` in `target_search.py`:

```python
# Load no-match threshold from config
try:
    threshold = cfg["reid"]["search"]["no_match_similarity_threshold"]
    print(f"[INFO] No-match similarity threshold: {threshold}")
except KeyError:
    print("[WARNING] No-match threshold not found in config, using default 0.70")
    threshold = 0.70
```

---

## Target Search Behavior

### Output Schema

**When best candidate meets/exceeds threshold:**

```json
{
  "status": "matches_found",
  "candidates": [
    {
      "camera_id": "C01",
      "timestamp": "10:02:04.89",
      "confidence": 75.3,
      "crop_path": "dataset/crops_phase3_final/..."
    }
  ]
}
```

**When best candidate is below threshold:**

```json
{
  "status": "no_confident_match",
  "candidates": []
}
```

### Threshold Logic

```python
if len(top_candidates) > 0:
    best_similarity = top_candidates[0]["similarity"]

    if best_similarity < threshold:
        output = {
            "status": "no_confident_match",
            "candidates": []
        }
    else:
        formatted = format_output(top_candidates)
        output = {
            "status": "matches_found",
            "candidates": formatted
        }
```

### Backend Compatibility

The new schema is backward-compatible:
- Existing `candidates` array preserved
- New `status` field indicates whether candidates are meaningful
- Empty `candidates` array when no confident match
- Frontend can check `status` before displaying results

---

## Unit Tests

### Test Coverage

```bash
pytest ai_pipeline/reid/test_target_search.py::TestNoMatchThreshold -v

6 passed in 8.99s
```

### Test Cases

| Test | Purpose |
|------|---------|
| `test_above_threshold_returns_candidates` | Best candidate above threshold returns candidates |
| `test_below_threshold_returns_no_match` | Best candidate below threshold returns no_confident_match |
| `test_exactly_at_threshold` | Best candidate at threshold returns candidates |
| `test_threshold_0_70_corresponds_to_confidence_57_4` | Verify similarity→confidence mapping |
| `test_json_output_schema_with_status` | Validate output JSON schema |
| `test_top_k_respected_when_match_exists` | Top-K still respected when match exists |

---

## Real-World Tests

### Unknown Person Test

**Requirement:** Test with a person who is NOT in the gallery.

**Setup:**
- Created `dataset/query_photos/unknown/test_unknown.jpg`
- Used image `1183_c2s3_000262_00.jpg` (person not in C01 gallery)
- Visually verified this person is not in the gallery
- Did NOT add this person's embedding to the gallery

**Actual Results:**
- Best candidate similarity: **0.7267**
- Status: **matches_found** (candidates returned)
- Threshold: 0.70

**Analysis:**
- The unknown person produced a similarity of 0.7267, which is **ABOVE** the threshold (0.70)
- The system **FAILED to reject** this unknown person
- This indicates the current threshold is **not conservative enough** to reject all unknown persons
- The unknown person may share visual characteristics with some gallery subjects (clothing, pose, background)

**Test Command (Colab):**
```bash
python ai_pipeline/reid/target_search.py \
    --query dataset/query_photos/unknown/test_unknown.jpg \
    --embeddings dataset/embeddings_C01.json \
    --top-k 5 \
    --output dataset/target_search/unknown_person_result.json
```

**Status:** ⚠️ **THRESHOLD FAILED** — Unknown person was not rejected

### Known Person Regression Test

**Requirement:** Ensure threshold does not incorrectly reject known person.

**Setup:**
- Used existing query: `dataset/query_photos/query_track13/test_query.jpg`
- This person IS in the gallery (Track 13)
- Best similarity was ~0.8145 (75.3 confidence) in Phase 5.6

**Actual Results:**
- Best candidate similarity: **0.8148**
- Status: **matches_found** (candidates returned)
- Threshold: 0.70
- Source crop exclusion: Enabled (C01_track13_frame0149.jpg excluded)

**Analysis:**
- The known person produced a similarity of 0.8148, which is **ABOVE** the threshold (0.70)
- The system **correctly accepted** this known person
- This is expected behavior — the threshold did not incorrectly reject a valid match

**Test Command (Colab):**
```bash
python ai_pipeline/reid/target_search.py \
    --query dataset/query_photos/query_track13/test_query.jpg \
    --embeddings dataset/embeddings_C01.json \
    --query-metadata dataset/query_photos/query_track13/query_metadata.json \
    --top-k 5 \
    --output dataset/target_search/known_person_regression.json
```

**Status:** ✅ **PASSED** — Known person was correctly accepted

---

## Honest Reporting

### Unknown Query Failed Threshold

The unknown query produced a similarity of **0.7267**, which is **ABOVE** the threshold (0.70).

**Honest assessment:**
1. ✅ **Reported honestly** — The threshold failed to reject this unknown person
2. ✅ **Threshold NOT changed** — We are keeping the MVP threshold at 0.70 despite this failure
3. ✅ **Actual similarity documented** — 0.7267 (above threshold)
4. ✅ **Analysis provided** — The unknown person may share visual characteristics with gallery subjects
5. ✅ **Limitation acknowledged** — The current MVP threshold is not conservative enough for all unknown persons

**Why the threshold failed:**
- The unknown person (1183_c2s3_000262_00.jpg) may share visual characteristics with some gallery subjects:
  - Similar clothing (color, style)
  - Similar pose or body type
  - Similar background context
- The Phase 4.5 different-track max (0.7996) was higher than expected
- The threshold of 0.70 sits between different-track mean (0.61) and max (0.80)
- This unknown person happened to fall in the upper portion of the different-track distribution

**This is a limitation of the current MVP threshold.** The goal was to measure baseline behavior honestly, not to game the threshold to pass tests.

---

## Production Safety

### Files Modified

| File | Status | Change |
|------|--------|--------|
| `ai_pipeline/config.yaml` | ✅ Modified | Added `reid.search.no_match_similarity_threshold` |
| `ai_pipeline/reid/target_search.py` | ✅ Modified | Added threshold loading and status field logic |
| `ai_pipeline/reid/test_target_search.py` | ✅ Modified | Added `TestNoMatchThreshold` test class |

### Files NOT Modified

- ✅ `dataset/embeddings_C01.json` — Unchanged
- ✅ OSNet model — Unchanged
- ✅ Preprocessing pipeline — Unchanged
- ✅ Similarity calculation — Unchanged
- ✅ Confidence scaling — Unchanged
- ✅ Phase 1–4 outputs — Unchanged
- ✅ Query images — Unchanged
- ✅ Tracking outputs — Unchanged

---

## Next Steps

### After Real-World Tests

1. **Run unknown person test** in Google Colab
2. **Run known person regression test** in Google Colab
3. **Document actual results**:
   - Unknown person best similarity
   - Unknown person behavior (rejected or not)
   - Known person best similarity
   - Known person behavior (accepted or not)
4. **Analyze threshold performance**:
   - Does threshold correctly reject unknown person?
   - Does threshold correctly accept known person?
   - If not, what adjustments are needed?
5. **Update documentation** with actual test results

### Future Improvements

To improve threshold selection:

1. **Larger validation sample** — More queries across different tracks
2. **Independent photograph queries** — Real external photos, not derived crops
3. **Multi-camera evaluation** — Test cross-camera similarity distributions
4. **Ground-truth identity labels** — Manual identity annotation for calibration
5. **Statistical calibration** — ROC analysis, precision-recall curves
6. **Adaptive thresholds** — Per-camera or per-scenario thresholds

---

## Files Summary

### Created

| File | Purpose |
|------|---------|
| `docs/phase5_7_no_match.md` | This documentation |

### Modified

| File | Lines Changed | Purpose |
|------|---------------|---------|
| `ai_pipeline/config.yaml` | +5 | Added threshold configuration |
| `ai_pipeline/reid/target_search.py` | +30 | Added threshold logic and status field |
| `ai_pipeline/reid/test_target_search.py` | +120 | Added threshold unit tests |

### Tests

| Metric | Value |
|--------|-------|
| New tests added | 6 |
| Tests passed | 6 |
| Tests failed | 0 |
| GPU required | No (synthetic data) |

---

## Important Reminders

1. **Threshold is NOT a probability** — It's a similarity threshold, not "P(person is the same) = X%"
2. **Track IDs are NOT ground truth** — Visual inspection required for validation
3. **This is MVP only** — Threshold may need adjustment for real-world deployment
4. **Honest reporting required** — If unknown query fails threshold, report it honestly
5. **Production data unchanged** — Only configuration and search logic modified
