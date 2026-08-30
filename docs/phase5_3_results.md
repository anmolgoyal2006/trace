# Phase 5.3 — Track-Level Aggregation

## Status

| Sub-phase | Status | Location |
|-----------|--------|----------|
| 5.3.1 — Define aggregation statistics | **COMPLETE** (VS Code) | `ai_pipeline/reid/track_aggregation.py` |
| 5.3.2 — Run aggregation on real data | **PENDING** (Google Colab) | — |

---

## Motivation

Phase 5.2 produced one cosine-similarity score per gallery crop. The crop-level
result for the derived query (`test_query_1.jpg`) was:

| Rank | Track | Similarity |
|------|-------|------------|
| 1 | 66 | 0.8076 |
| **2** | **29 (correct)** | **0.7928** |

Track 29 is the source person's track. It ranked second, not first, at the
individual-crop level.

A natural question is whether **multiple observations from the same track**,
taken together, provide stronger evidence than any single crop. A person moves
through the scene; some of their crops will be at better angles, under better
lighting, or captured more completely than others. Aggregating across all crops
of a track reduces the effect of any one poor crop and rewards tracks that are
consistently similar to the query.

This does **not** guarantee that aggregation will change the ranking. That must
be measured on the real data in Colab.

---

## Phase 5.3.1 — Aggregation Statistics

### Implementation

Module: `ai_pipeline/reid/track_aggregation.py`
Entry point: `aggregate_by_track(records) -> dict`

The function accepts a list of crop-level similarity records. Each record must
contain at minimum:

```python
{
    "track_id":   int | str,   # identifies the track
    "similarity": float,       # cosine similarity for this crop
}
```

Additional fields are silently ignored, so the function can be passed the raw
output list from Phase 5.2 without preprocessing.

### The four statistics

For every track in the input, `aggregate_by_track` returns:

#### `max_similarity`
The highest single-crop cosine similarity for that track.

```
max_similarity = max(sims)
```

Interpretation: the best moment the Re-ID model saw this person. Robust to
occlusion or poor-quality crops elsewhere in the track.

#### `mean_similarity`
Arithmetic mean over **all** crops for that track.

```
mean_similarity = sum(sims) / n
```

Interpretation: average evidence across the whole track. Penalises tracks that
include many low-quality crops alongside a few strong ones.

#### `mean_top3_similarity`
Mean of the **three highest** similarities for that track. If fewer than three
crops exist, the mean is taken over all available crops (no padding).

```python
sorted_desc = sorted(sims, reverse=True)
top_k = sorted_desc[:3]          # <= 3 elements
mean_top3 = sum(top_k) / len(top_k)
```

Interpretation: a balance between robustness (not relying on a single crop as
`max_similarity` does) and resilience to outlier-low crops (not penalised as
heavily as `mean_similarity`).

#### `num_observations`
Count of gallery crops that contributed to the statistics.

```
num_observations = len(sims)
```

Interpretation: indicates how well-represented the track is in the gallery.
Tracks with very few crops are more susceptible to noise.

### Why these four

| Statistic | Strength | Weakness |
|-----------|----------|---------|
| `max_similarity` | Captures best view; immune to bad crops | One outlier-high crop can dominate |
| `mean_similarity` | Smooth; reflects overall track quality | Sensitive to outlier-low crops |
| `mean_top3_similarity` | Balanced; reduces noise, retains peak | Arbitrary choice of k=3 |
| `num_observations` | Context for interpreting the others | Informational only |

Choosing which statistic to use for final ranking is a Phase 5.3.2 decision,
to be made after measuring on the real data.

### Return type guarantee

All values in the returned dict are plain Python `float` or `int`. No NumPy
scalar types are used, so the result is directly JSON-serialisable.

---

## Anti-contamination guarantee

The aggregation function:

- does **not** know which track is the correct source track
- does **not** apply any threshold or ranking
- does **not** use Track 29's identity in any way
- is purely mechanical: group by `track_id`, compute four statistics

Track 29 is mentioned in this document only as historical context from Phase
5.2. It will only be used for evaluation **after** Phase 5.3.2 is run in Colab.

---

## Phase 5.3.2 — Pending Colab Experiment

> **The aggregation has NOT yet been evaluated on the real Phase 5.2 data.**
> No measured Phase 5.3 results exist. No ranking is claimed.
> Do not interpret any numbers in this document as Phase 5.3 outcomes.

### Colab execution script

`ai_pipeline/reid/phase5_3_colab_aggregation.py`

Run this script in Google Colab once your Drive is mounted. It:

1. Loads `dataset/embeddings_C01.json` (353 gallery crops).
2. Loads `dataset/query_photos/test_query_1.jpg` and produces a query
   embedding using the Phase 4.4-verified OSNet x1_0 preprocessing
   (`Resize(256,128)` → `ToTensor` → `Normalize(ImageNet)`).
3. Computes cosine similarity against every gallery crop.
4. Excludes `C01_track29_frame0270.jpg` (the source crop) from the gallery.
5. Calls `aggregate_by_track()` on the crop-level similarity records.
6. Ranks all tracks by `max_similarity`, `mean_similarity`, and
   `mean_top3_similarity`.
7. Prints Track 29's rank and score under each method, and whether it
   improves from the Phase 5.2 crop-level Rank #2 baseline.
8. Saves results to `dataset/target_search/track_aggregation_test_query_1.json`.

### How to run

```python
# In Colab — mount Drive first, then:
# !pip install torchreid
# exec(open("/content/drive/MyDrive/Trace/ai_pipeline/reid/phase5_3_colab_aggregation.py").read())
# OR run each cell block sequentially
```

Adjust `REPO_ROOT` at the top of the script to match your Drive path.

### Procedure once run

1. Load the crop-level similarity records produced by the script.
2. For each of the three aggregation statistics, rank all tracks descending.
3. Record the rank of Track 29 under each statistic.
4. Compare with the Phase 5.2 crop-level baseline (Track 29 at Rank 2).
5. Document whether aggregation improves, maintains, or worsens the ranking.

Only after step 5 can any claim be made about whether track-level aggregation
helps Re-ID for this dataset.

**Phase 5.3 results section will be filled in after Colab execution.**

---

## Files

| File | Purpose |
|------|---------|
| `ai_pipeline/reid/track_aggregation.py` | Aggregation implementation |
| `ai_pipeline/reid/test_track_aggregation.py` | 13-class pytest suite (CPU-only) |
| `ai_pipeline/reid/phase5_3_colab_aggregation.py` | Colab execution script (GPU, real data) |
| `docs/phase5_3_results.md` | This document |
| `dataset/target_search/track_aggregation_test_query_1.json` | Output (written by Colab script after execution) |

## Files NOT modified

- `dataset/embeddings_C01.json`
- `dataset/crops_metadata_phase3_final.json`
- `dataset/crops_phase3_final/` (353 crops)
- All Phase 1–4 scripts and outputs
- `ai_pipeline/reid/similarity.py`
- `ai_pipeline/reid/target_search.py` (not yet implemented)
