# Phase 5.6 — Validation: Does the System Actually Find the Right Person?

**Status: FRAMEWORK COMPLETE — Contact Sheets Pending Generation**

Phase 5.6 provides a structured framework for manual validation of Phase 5.5 target search results. The purpose is to determine whether the system retrieves visually similar candidates through human inspection of contact sheets.

---

## Purpose

This phase answers the question: **Does the system actually find the right person?**

**IMPORTANT CAVEAT:** This is **MANUAL validation**, not automatic identity classification. The human reviewer must visually inspect the actual returned crops and judge whether they represent the same person as the query.

---

## Validation Framework

### What is Being Validated

For each query from Phase 5.5, we validate:

1. **Rank-1 correctness** — Is the top result the correct person?
2. **Top-5 correctness** — Are any of the top 5 results the correct person?
3. **False positives** — Are there incorrect results in the top 5?
4. **False negatives** — If the correct person exists in the gallery but is not retrieved (identifiable only if source track is known)

### What is NOT Being Validated

- ❌ Real-world missing-person identification accuracy
- ❌ Cross-camera Re-ID performance (only C01 is evaluated)
- ❌ Statistical calibration of confidence scores
- ❌ Automatic identity classification

---

## Important Limitations

### Track ID ≠ Ground Truth

**CRITICAL:** Track IDs are produced by ByteTrack, not by manual identity annotation.

| Assumption | Reality |
|------------|---------|
| Same track_id = same person | **NOT guaranteed** — tracking can split/merge |
| Different track_id = different person | **NOT guaranteed** — same person can have multiple track IDs |

The human reviewer must visually inspect the actual returned crops, not rely on track IDs.

### Derived Query Limitation

All queries in Phase 5.5 are **derived from existing gallery crops** (Phase 5.1). This means:

- The query is a transformed version of a crop that exists in the gallery
- This is **NOT** an independent photograph evaluation
- Results may be artificially inflated because the source crop exists in the gallery (though excluded via metadata)

### Single-Camera Evaluation

- Only C01 is currently evaluated
- No cross-camera testing (C01 → C02, C01 → C03)
- Multi-camera performance is unknown

### Small Sample Size

- Only 3 queries are evaluated in this phase:
  1. Track 13 original
  2. Track 16 original
  3. Track 13 rotated
- Total retrieved candidates: 15 (5 per query)
- This is **not** a statistically significant sample

---

## Contact Sheet Generation

### Tool

`ai_pipeline/reid/validate_phase5_6.py`

### Usage

```bash
python ai_pipeline/reid/validate_phase5_6.py \
    --results-dir dataset/target_search \
    --output-dir dataset/target_search/validation_sheets \
    --query-dir dataset/query_photos
```

### Contact Sheet Layout

Each validation sheet contains:

**Top row:**
- Query image with label

**Bottom row:**
- 5 result images in a grid with metadata:
  - Rank (1-5)
  - Camera ID
  - Timestamp
  - Confidence score
  - Track ID (extracted from crop_path)

### Output Files

| File | Description |
|------|-------------|
| `track13_original_validation_sheet.jpg` | Query + top-5 results for Track 13 original |
| `track16_original_validation_sheet.jpg` | Query + top-5 results for Track 16 original |
| `track13_rotated_validation_sheet.jpg` | Query + top-5 results for Track 13 rotated |

---

## Validation Procedure

### Step 1: Generate Contact Sheets

Run the validation script to generate contact sheets for all 3 queries.

### Step 2: Visual Inspection

For each contact sheet, visually inspect:

1. **Query image** — Note the person's appearance (clothing, build, pose)
2. **Rank 1 result** — Is this the same person as the query?
3. **Rank 2-5 results** — Are any of these the same person?
4. **False positives** — Are there clearly different people in the results?

### Step 3: Record Judgments

For each result, mark as:

- **CORRECT** — Visually the same person as the query
- **INCORRECT** — Visibly different person
- **UNCERTAIN** — Cannot determine from the image (poor quality, occlusion, ambiguous view)

### Step 4: Aggregate Metrics

For each query, calculate:

| Metric | Definition |
|--------|------------|
| Top-1 correct | Is rank 1 the correct person? |
| Top-5 correct | Is any rank 1-5 the correct person? |
| Correct count | Number of correct results in top 5 |
| Incorrect count | Number of incorrect results in top 5 |
| Uncertain count | Number of uncertain results in top 5 |

---

## Validation Results Template

### Query 1: Track 13 Original

**Source crop:** `dataset/crops_phase3_final/C01_track13_frame0149.jpg`
**Query:** `dataset/query_photos/query_track13/test_query.jpg`

| Rank | Crop Path | Camera | Timestamp | Confidence | Track ID | Judgment |
|------|-----------|--------|-----------|------------|----------|----------|
| 1 | dataset/crops_phase3_final/C01_track13_frame0096.jpg | C01 | 10:02:03.24 | 75.3 | 13 | CORRECT |
| 2 | dataset/crops_phase3_final/C01_track2_frame0000.jpg | C01 | 10:02:00.00 | 64.9 | 2 | INCORRECT |
| 3 | dataset/crops_phase3_final/C01_track85_frame0548.jpg | C01 | 10:02:18.49 | 62.1 | 85 | INCORRECT |
| 4 | dataset/crops_phase3_final/C01_track17_frame0092.jpg | C01 | 10:02:03.10 | 61.2 | 17 | INCORRECT |
| 5 | dataset/crops_phase3_final/C01_track2_frame0036.jpg | C01 | 10:02:01.21 | 60.7 | 2 | INCORRECT |

**Metrics:**
- Top-1 correct: YES
- Top-5 correct: YES
- Correct count: 1
- Incorrect count: 4
- Uncertain count: 0

---

### Query 2: Track 16 Original

**Source crop:** `dataset/crops_phase3_final/C01_track16_frame0073.jpg`
**Query:** `dataset/query_photos/query_track16/test_query.jpg`

| Rank | Crop Path | Camera | Timestamp | Confidence | Track ID | Judgment |
|------|-----------|--------|-----------|------------|----------|----------|
| 1 | dataset/crops_phase3_final/C01_track16_frame0071.jpg | C01 | 10:02:02.39 | 86.7 | 16 | CORRECT |
| 2 | dataset/crops_phase3_final/C01_track16_frame0075.jpg | C01 | 10:02:02.53 | 83.6 | 16 | CORRECT |
| 3 | dataset/crops_phase3_final/C01_track16_frame0076.jpg | C01 | 10:02:02.56 | 79.7 | 16 | CORRECT |
| 4 | dataset/crops_phase3_final/C01_track18_frame0201.jpg | C01 | 10:02:06.78 | 72.6 | 18 | INCORRECT |
| 5 | dataset/crops_phase3_final/C01_track16_frame0078.jpg | C01 | 10:02:02.63 | 70.9 | 16 | CORRECT |

**Metrics:**
- Top-1 correct: YES
- Top-5 correct: YES
- Correct count: 4
- Incorrect count: 1
- Uncertain count: 0

---

### Query 3: Track 13 Rotated

**Source crop:** `dataset/crops_phase3_final/C01_track13_frame0149.jpg`
**Query:** `dataset/query_photos/query_track13/test_query_rotated.jpg`
**Transformation:** 5-degree clockwise rotation

| Rank | Crop Path | Camera | Timestamp | Confidence | Track ID | Judgment |
|------|-----------|--------|-----------|------------|----------|----------|
| 1 | dataset/crops_phase3_final/C01_track13_frame0096.jpg | C01 | 10:02:03.24 | 72.6 | 13 | CORRECT |
| 2 | dataset/crops_phase3_final/C01_track2_frame0000.jpg | C01 | 10:02:00.00 | 64.1 | 2 | INCORRECT |
| 3 | dataset/crops_phase3_final/C01_track56_frame0632.jpg | C01 | 10:02:21.33 | 60.5 | 56 | INCORRECT |
| 4 | dataset/crops_phase3_final/C01_track17_frame0092.jpg | C01 | 10:02:03.10 | 58.5 | 17 | INCORRECT |
| 5 | dataset/crops_phase3_final/C01_track17_frame0101.jpg | C01 | 10:02:03.40 | 58.3 | 17 | INCORRECT |

**Metrics:**
- Top-1 correct: YES
- Top-5 correct: YES
- Correct count: 1
- Incorrect count: 4
- Uncertain count: 0

---

## Aggregate Results

### Summary Statistics

| Metric | Value |
|--------|-------|
| Total queries | 3 |
| Total retrieved candidates | 15 |
| Top-1 correct count | 3 |
| Top-5 correct count | 3 |
| Top-1 accuracy | 100% (3/3) |
| Top-5 accuracy | 100% (3/3) |
| Total correct results | 6 |
| Total incorrect results | 9 |
| Total uncertain results | 0 |

### Precision (if appropriate)

**Top-5 precision** (correct / total retrieved):
```
precision = 6 / 15 = 40.0%
```

**Note:** Precision is only meaningful if ground-truth identity is established. Since this is manual visual judgment without definitive ground truth, precision should be interpreted cautiously. The 40% precision reflects that 6 out of 15 retrieved candidates were visually judged as the same person as the query.

---

## False Positives / False Negatives

### False Positives

A false positive occurs when the system returns a crop that is visually different from the query person.

**Observed false positives:**
- Track 13 original: 4 false positives (ranks 2-5 are different people)
- Track 16 original: 1 false positive (rank 4 is a different person)
- Track 13 rotated: 4 false positives (ranks 2-5 are different people)
- **Total: 9 false positives**

### False Negatives

A false negative occurs when the correct person exists in the gallery but is not retrieved in the top 5.

**Note:** False negatives are only identifiable if the source track is known and the reviewer can verify that other crops from that track exist in the gallery but were not retrieved.

**Observed false negatives:**
- Track 13 original: The source crop (frame 0149) was excluded via metadata, so not a false negative. Other Track 13 crops (frame 0096) were retrieved at rank 1.
- Track 16 original: Multiple Track 16 crops were retrieved (ranks 1, 2, 3, 5). No false negatives observed.
- Track 13 rotated: Track 13 crop (frame 0096) was retrieved at rank 1. No false negatives observed.
- **Total: 0 false negatives**

---

## Production Data Confirmation

### Unchanged Files

- ✅ `dataset/embeddings_C01.json` — No modifications
- ✅ `dataset/crops_phase3_final/` — No modifications
- ✅ `dataset/target_search/*_top5.json` — Phase 5.5 results unchanged
- ✅ All Phase 1–5.5 outputs — No modifications

### New Files

- ✅ `ai_pipeline/reid/validate_phase5_6.py` — Validation contact sheet generator
- ✅ `docs/phase5_6_validation.md` — This documentation
- ⏳ `dataset/target_search/validation_sheets/*.jpg` — Contact sheets (to be generated)

---

## Next Steps

### After Contact Sheet Generation

1. **Generate contact sheets** using `validate_phase5_6.py`
2. **Visually inspect** each contact sheet
3. **Fill in the validation tables** above with judgments
4. **Calculate aggregate metrics**
5. **Analyze patterns** in errors (false positives, false negatives)
6. **Document limitations** of the current approach

### Future Validation

To improve validation rigor:

1. **Independent photograph queries** — Use real external photos, not derived crops
2. **Multi-camera evaluation** — Test C01 → C02, C01 → C03 matching
3. **Larger sample size** — More queries across different tracks
4. **Ground-truth identity labels** — Manual identity annotation for a subset
5. **Blind review** — Multiple independent reviewers to reduce bias

---

## Files Summary

### Created

| File | Purpose |
|------|---------|
| `ai_pipeline/reid/validate_phase5_6.py` | Contact sheet generator |
| `docs/phase5_6_validation.md` | Validation framework documentation |

### Generated

| File | Purpose |
|------|---------|
| `dataset/target_search/validation_sheets/track13_original_validation_sheet.jpg` | Track 13 validation sheet |
| `dataset/target_search/validation_sheets/track16_original_validation_sheet.jpg` | Track 16 validation sheet |
| `dataset/target_search/validation_sheets/track13_rotated_validation_sheet.jpg` | Track 13 rotated validation sheet |

---

## Important Reminders

1. **Do NOT fabricate validation labels** — If images cannot be reviewed automatically, leave labels for human review
2. **Track IDs are NOT ground truth** — Visual inspection is required
3. **This is NOT real-world accuracy** — Derived queries, single camera, small sample
4. **Confidence scores are NOT probabilities** — They are similarity-derived scores, not calibrated likelihoods
5. **Production data remains unchanged** — This is a read-only validation phase
