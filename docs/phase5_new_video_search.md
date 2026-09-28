# Phase 5.8 — Target Person Search (Query Matching)

## Overview

Phase 5.8 provides a query-search entry point that accepts a photo of a target
person, embeds it using the existing OSNet pipeline, and retrieves the most
similar person crops from **any compatible gallery** (original C01 or new video).

```
Uploaded target photo
        ↓
OSNet query embedding (256×128, ImageNet normalization)
        ↓
Load gallery embeddings (512-D JSON)
        ↓
Cosine similarity
        ↓
Rank candidates (descending)
        ↓
Return top-K results with confidence scores
```

**Key design decisions:**

- Reuses the exact Phase 4 preprocessing pipeline (`_TRANSFORM` from `embed.py`)
- Reuses the existing cosine similarity implementation (`similarity.py`)
- Reuses the Phase 5.4 confidence scaling (`confidence_scaling.py`)
- Supports **any compatible query image** via CLI argument
- Supports **any compatible gallery** via CLI argument
- Does NOT modify production data or original embeddings
- GPU preferred but CPU fallback supported

---

## CLI Usage

```bash
python ai_pipeline/reid/target_search.py \
    --query <path-to-query-image> \
    --gallery <path-to-gallery-embeddings> \
    --top-k <number> \
    --output <path-to-output-json> \
    --format <full|backend>
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--query` | Yes | — | Path to query image (JPEG/PNG) |
| `--gallery` / `--embeddings` | Yes | — | Path to gallery embeddings JSON |
| `--query-metadata` | No | None | Path to query metadata JSON (for source crop exclusion) |
| `--top-k` | No | 5 | Number of candidates to return |
| `--output` | No | stdout | Path to output JSON file |
| `--format` | No | `full` | Output format: `full` or `backend` |
| `--config` | No | `ai_pipeline/config.yaml` | Path to config YAML |

### Example: Search new video gallery

```bash
python ai_pipeline/reid/target_search.py \
    --query dataset/query_photos/test_query_1.jpg \
    --gallery dataset/new_video_test/embeddings_C01.json \
    --top-k 10 \
    --output dataset/new_video_test/target_search_results.json
```

### Example: Search original C01 gallery

```bash
python ai_pipeline/reid/target_search.py \
    --query dataset/query_photos/test_query_1.jpg \
    --gallery dataset/embeddings_C01.json \
    --query-metadata dataset/query_photos/query_metadata.json \
    --top-k 5 \
    --output dataset/target_search/results.json
```

---

## Input Format

### Query Image

Any standard image file (JPEG, PNG). The image will be:
1. Loaded via PIL and converted to RGB
2. Resized to 256×128 pixels
3. Normalized with ImageNet mean/std
4. Passed through OSNet `osnet_x1_0` to produce a 512-D embedding

### Gallery Embeddings JSON

A JSON array where each record contains at minimum:

```json
[
  {
    "crop_path": "dataset/new_video_test/crops/C01_track5_frame0120.jpg",
    "camera_id": "C01",
    "track_id": 5,
    "frame": 120,
    "timestamp": "10:02:04.05",
    "bbox": [100.0, 200.0, 300.0, 600.0],
    "detection_confidence": 0.92,
    "embedding": [0.123, 0.456, ...]
  }
]
```

The `embedding` field must be a 512-dimensional float array produced by
OSNet `osnet_x1_0` with the same preprocessing pipeline as Phase 4.

---

## Output Format

### Full Format (default: `--format full`)

```json
{
  "query": "dataset/query_photos/test_query_1.jpg",
  "gallery": "dataset/new_video_test/embeddings_C01.json",
  "top_k": 10,
  "model": "osnet_x1_0",
  "embedding_dim": 512,
  "gallery_size": 67,
  "threshold": 0.74,
  "status": "matches_found",
  "results": [
    {
      "rank": 1,
      "similarity": 0.852345,
      "confidence": 81.2,
      "crop_path": "dataset/new_video_test/crops/C01_track5_frame0120.jpg",
      "camera_id": "C01",
      "track_id": 5,
      "frame": 120
    }
  ]
}
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | Path to the query image used |
| `gallery` | string | Path to the gallery embeddings used |
| `top_k` | int | Requested number of candidates |
| `model` | string | OSNet model name |
| `embedding_dim` | int | Embedding dimension (512) |
| `gallery_size` | int | Total records in gallery |
| `threshold` | float | No-match similarity threshold |
| `status` | string | `"matches_found"` or `"no_confident_match"` |
| `results` | array | Ranked candidates (empty if no confident match) |

**Per-result fields:**

| Field | Type | Description |
|-------|------|-------------|
| `rank` | int | 1-based rank position |
| `similarity` | float | Raw cosine similarity (rounded to 6 decimals) |
| `confidence` | float | Scaled confidence score 0–100 (1 decimal) |
| `crop_path` | string | Path to the matched gallery crop |
| `camera_id` | string | Camera identifier |
| `track_id` | int | ByteTrack tracking ID |
| `frame` | int | Video frame number |

### Backend Format (`--format backend`)

Minimal format for backend handoff:

```json
{
  "status": "matches_found",
  "candidates": [
    {
      "camera_id": "C01",
      "timestamp": "10:02:04.05",
      "confidence": 81.2,
      "crop_path": "dataset/new_video_test/crops/C01_track5_frame0120.jpg"
    }
  ]
}
```

---

## Confidence Score Interpretation

> **The confidence score is NOT a calibrated probability of identity.**

The confidence score is a **similarity-derived heuristic** computed using
min-max normalization:

```
confidence = ((similarity - 0.3315) / (0.9730 - 0.3315)) × 100
```

Clamped to [0, 100].

The normalization constants are from Phase 4.5 empirical analysis on the
original C01 gallery:
- `OBSERVED_MIN = 0.3315` (minimum different-track similarity)
- `OBSERVED_MAX = 0.9730` (maximum same-track similarity)

### Limitations

1. **Not calibrated**: A confidence of 85% does NOT mean 85% probability that
   the person is the same identity.
2. **Dataset-specific**: The normalization constants were derived from C01 gallery
   distributions. A different video/camera may have a different similarity
   distribution, shifting the effective midpoint.
3. **No-match threshold**: If the best candidate's raw similarity is below 0.74
   (configurable in `config.yaml` → `reid.search.no_match_similarity_threshold`),
   the system returns `"no_confident_match"` with an empty results list.
4. **Track IDs are not identity labels**: ByteTrack track IDs represent tracking
   trajectories, not verified person identities. Visual confirmation is required.

---

## How to Run in Google Colab

### Step 1: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2: Set the TRACE repository root

```python
import os

# Adjust this path to match YOUR Google Drive layout
TRACE_ROOT = "/content/drive/MyDrive/Trace"

# Verify the path exists
assert os.path.isdir(TRACE_ROOT), f"TRACE root not found: {TRACE_ROOT}"
os.chdir(TRACE_ROOT)
print(f"Working directory: {os.getcwd()}")
```

### Step 3: Install dependencies

```python
!pip install -q torch torchvision torchreid pillow numpy pyyaml
```

### Step 4: Upload or place the query photo

**Option A — Upload from local machine:**
```python
from google.colab import files
uploaded = files.upload()  # Select your query image

# Move to query_photos directory
import shutil
for filename in uploaded:
    shutil.move(filename, f"dataset/query_photos/{filename}")
    print(f"Saved query photo: dataset/query_photos/{filename}")
```

**Option B — Use an existing query photo:**
```python
# List available query photos
!ls dataset/query_photos/*.jpg
```

### Step 5: Run target search against the new video gallery

```bash
!python ai_pipeline/reid/target_search.py \
    --query dataset/query_photos/test_query_1.jpg \
    --gallery dataset/new_video_test/embeddings_C01.json \
    --top-k 10 \
    --output dataset/new_video_test/target_search_results.json
```

### Step 6: Inspect the JSON results

```python
import json

with open("dataset/new_video_test/target_search_results.json") as f:
    results = json.load(f)

print(f"Status: {results['status']}")
print(f"Gallery size: {results['gallery_size']}")
print(f"Results: {len(results['results'])}")
print()

for r in results["results"]:
    print(f"  Rank {r['rank']}: similarity={r['similarity']:.4f}  "
          f"confidence={r['confidence']}%  "
          f"track={r['track_id']}  frame={r['frame']}  "
          f"crop={r['crop_path']}")
```

### Step 7 (Optional): View top candidate crops

```python
from PIL import Image
import matplotlib.pyplot as plt

top_n = min(5, len(results["results"]))
fig, axes = plt.subplots(1, top_n + 1, figsize=(4 * (top_n + 1), 6))

# Show query
query_img = Image.open(results["query"])
axes[0].imshow(query_img)
axes[0].set_title("QUERY", fontsize=12, fontweight='bold')
axes[0].axis("off")

# Show top candidates
for i, r in enumerate(results["results"][:top_n]):
    crop_img = Image.open(r["crop_path"])
    axes[i + 1].imshow(crop_img)
    axes[i + 1].set_title(
        f"Rank {r['rank']}\n"
        f"sim={r['similarity']:.3f}\n"
        f"conf={r['confidence']}%\n"
        f"track={r['track_id']}",
        fontsize=10
    )
    axes[i + 1].axis("off")

plt.tight_layout()
plt.savefig("dataset/new_video_test/target_search_visual.png", dpi=150)
plt.show()
```

---

## Using with Different Galleries

The `--gallery` argument accepts any compatible embeddings JSON file:

| Gallery | Path | Description |
|---------|------|-------------|
| Original C01 | `dataset/embeddings_C01.json` | 353 crops from original video |
| New video test | `dataset/new_video_test/embeddings_C01.json` | 67 crops from new test video |

Both galleries use the same embedding format (512-D OSNet `osnet_x1_0`).
The confidence scaling constants were calibrated on the original C01 gallery
but will work with any gallery (values are clamped to [0, 100]).

---

## Files

| File | Purpose |
|------|---------|
| `ai_pipeline/reid/target_search.py` | CLI entry point and search logic |
| `ai_pipeline/reid/confidence_scaling.py` | Similarity → confidence conversion |
| `ai_pipeline/reid/similarity.py` | Cosine similarity (pure NumPy) |
| `ai_pipeline/reid/embed.py` | OSNet model loading and `_TRANSFORM` |
| `ai_pipeline/config.yaml` | Configuration (model, threshold, etc.) |
| `ai_pipeline/reid/test_target_search.py` | Unit tests (GPU-free) |
| `ai_pipeline/reid/test_confidence_scaling.py` | Unit tests (GPU-free) |

---

## Files NOT Modified

The following production files remain untouched:

- `dataset/embeddings_C01.json`
- `dataset/detections_C01.json`
- `dataset/tracks_C01.json`
- `dataset/crops_metadata_phase3_final.json`
- `dataset/crops_phase3_final/`
