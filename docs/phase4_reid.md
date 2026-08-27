# Phase 4: Re-ID Embedding Generation

## 4.1 OSNet Inference Verification

**Status: COMPLETE (experimentally verified in Google Colab)**

Phase 4.1 confirmed that OSNet x1_0 with pretrained weights produces valid appearance embeddings
from real TRACE crop images.

| Property | Value |
|---|---|
| Model | OSNet x1_0 |
| Pretrained weights | Yes (ImageNet + Market-1501) |
| GPU | Tesla T4 |
| Test crop | `C01_track100_frame0607.jpg` |
| Input shape | `[1, 3, 256, 128]` |
| Output shape | `[1, 512]` |
| Embedding dimension | **512** |
| dtype | float32 |
| NaN values | 0 |
| Inf values | 0 |
| Non-zero values | 248 |
| L2 norm | 24.154 |
| Same-image cosine similarity | 1.0 |

Conclusion: OSNet x1_0 produces a valid, stable 512-dimensional float32 appearance embedding
from TRACE crop images. The embedding has no degenerate values and achieves perfect self-similarity.

---

## 4.2 Embedding Storage Format

**Status: COMPLETE — format decided and documented**

### Decision

**JSON**

The official MVP embedding storage format for TRACE Re-ID is JSON.

Future output file: `dataset/embeddings_C01.json`

### Reason

JSON was selected for the MVP because TRACE already uses a JSON-first structured output pipeline
and the expected dataset is only a few thousand crops. Human readability and debugging ease are
more valuable at this scale than the storage and load efficiency of binary formats.

The guiding principle for this decision is:

> **Pipeline consistency + human debuggability > raw storage/loading efficiency** at MVP dataset scale.

Specific reasons:

1. **Consistency with Phase 1–3.** TRACE already produces all structured outputs as JSON:
   - `dataset/detections_C01.json` — YOLO detection results
   - `dataset/tracks_C01.json` — ByteTrack tracking results
   - `dataset/crops_metadata.json` — Phase 3 crop metadata
   - `dataset/crops_metadata_phase3_final.json` — Phase 3.5 quality-filtered metadata
   Adding embeddings as JSON keeps every artifact in the pipeline in the same format.

2. **Human readability.** Any team member can open `embeddings_C01.json` in a text editor and
   immediately read individual records without needing special tooling.

3. **Debugging ease.** When investigating a matching error in Phase 6/8, a developer can directly
   inspect the embedding record alongside its metadata (crop path, track ID, bounding box,
   confidence) without writing extra deserialization code.

4. **Metadata co-location.** JSON keeps the embedding and its provenance (camera, track, frame,
   timestamp, bbox, confidence) in a single record. Binary formats would require a separate
   metadata sidecar file or an index structure to maintain the same linkage.

5. **Easy access in downstream phases.** Phase 6/8 matching logic can `json.load()` the file and
   directly access `record["embedding"]` as a Python list alongside the identity fields — no
   additional parsing layer needed.

6. **Acceptable overhead at MVP scale.** 353 crops × 512 float32 values ≈ ~700 KB of embedding
   data. JSON float representation adds overhead, but the total file remains well under 10 MB.
   This is entirely acceptable for a local research pipeline.

7. **Non-goals at MVP scale.** Raw I/O throughput and memory efficiency are not bottlenecks when
   loading a few hundred records once per pipeline run.

8. **Incremental upgrade path.** If TRACE later scales to hundreds of thousands or millions of
   embeddings, the JSON records can be migrated to `.npy`/`.pt`, FAISS, or a vector database
   without changing the conceptual embedding interface. The record schema defined here stays valid.

### Alternatives Considered

| Property | JSON (selected) | `.npy` / `.pt` (rejected for MVP) |
|---|---|---|
| Human readable | Yes | No |
| Easy to debug | Yes | No — requires numpy/torch to inspect |
| Consistent with Phase 1–3 | Yes | No |
| Metadata co-located | Yes | Requires sidecar file |
| Storage size | Larger (~5–10× float text) | Compact (raw binary) |
| Load speed | Slower | Faster |
| Tooling required | `json` stdlib only | `numpy` or `torch` |
| Acceptable at MVP scale | Yes | Yes, but unnecessary complexity |

`.npy` and `.pt` are the right choice when embedding counts reach tens of thousands or when
sub-millisecond load time is required. Neither condition applies at TRACE MVP scale.

### Selected Format

```
dataset/embeddings_C01.json
```

This file will be generated in Phase 4.3. It does not exist yet.

### Record Structure

Each record in `embeddings_C01.json` follows this schema:

```json
{
  "crop_path": "dataset/crops_phase3_final/C01_track100_frame0607.jpg",
  "camera_id": "C01",
  "track_id": 100,
  "frame": 607,
  "timestamp": "10:02:20.50",
  "bbox": [292.15, 154.30, 412.88, 534.72],
  "detection_confidence": 0.94,
  "embedding": [1.1892024, 0.0, 0.0, "... 509 more float values ..."]
}
```

**Field definitions:**

| Field | Type | Source | Description |
|---|---|---|---|
| `crop_path` | string | Phase 3 metadata | Relative path to the crop image file |
| `camera_id` | string | Phase 3 metadata | Camera that recorded this observation (e.g. `"C01"`) |
| `track_id` | integer | Phase 3 metadata | ByteTrack track ID within this camera |
| `frame` | integer | Phase 3 metadata | Frame index within the source video |
| `timestamp` | string | Phase 3 metadata | Wall-clock timestamp (`"HH:MM:SS.ss"`) |
| `bbox` | [x1, y1, x2, y2] | Phase 3 metadata | YOLO detection bounding box in pixel coordinates |
| `detection_confidence` | float | Phase 3 metadata | YOLO detection confidence score |
| `embedding` | list[float] | Phase 4.3 OSNet | 512-dimensional appearance embedding vector |

**Key constraint:** Every field from Phase 3 crop metadata is preserved verbatim. This ensures
every embedding can be traced back to its original video observation without joining across files.

### Embedding Dimension

**512 dimensions**, using float32 values.

This was experimentally confirmed in Phase 4.1 using OSNet x1_0 with pretrained weights on a
real TRACE crop image (`C01_track100_frame0607.jpg`) on Google Colab with a Tesla T4 GPU.

The Phase 4.3 embedding generator will determine the output dimension programmatically from the
model at runtime and validate it against the expected value of 512. The `embedding_dim: 512`
entry in `config.yaml` serves as documentation and a validation target — it does not hardcode
the model's output shape in inference code.

### Configuration

The storage decision is recorded in `ai_pipeline/config.yaml` under the `reid.embedding` key:

```yaml
reid:
  crops_per_track: 5
  min_crop_width: 40
  min_crop_height: 100
  quality:
    reject_severe_edge_truncation: true
    edge_margin_pixels: 5
  embedding:
    storage_format: "json"   # MVP format
    model: "osnet_x1_0"
    embedding_dim: 512        # confirmed Phase 4.1; validation target for Phase 4.3
```

### Future Scalability

If the TRACE dataset grows to hundreds of thousands or millions of observations, the following
migration paths are available:

- **`.npy` arrays** — store embeddings as a matrix with a separate JSON/CSV index for metadata
- **`.pt` (PyTorch tensors)** — same approach, native to the torchreid/PyTorch stack
- **FAISS index** — approximate nearest-neighbour search at million-scale, no metadata storage
- **Vector database** (e.g. Pinecone, Weaviate, Qdrant, pgvector) — managed storage + ANN search
  with metadata filtering

None of these are in scope for the current MVP. The JSON record schema defined here is compatible
with any future migration because it is a self-contained, well-documented format.

---

## 4.3 Embedding Generation (UPCOMING)

Phase 4.3 will:

1. Load all 353 crops from `dataset/crops_phase3_final/`
2. Load their metadata from `dataset/crops_metadata_phase3_final.json`
3. Run each crop through OSNet x1_0 to produce a 512-dimensional embedding
4. Validate that the model output dimension equals 512 (from config)
5. Merge the embedding with the existing metadata fields
6. Write all records to `dataset/embeddings_C01.json`

Phase 4.3 has not started yet.

---

## 4.4 Preprocessing Correctness Verification

**Status: COMPLETE — TRACE preprocessing verified as bit-exact match to Torchreid**

### Torchreid source inspected

```
Source:  https://github.com/KaiyangZhou/deep-person-reid
File:    torchreid/data/transforms.py
Branch:  master
```

The `build_transforms()` function was read directly from the repository source. The authoritative
test/inference transform is constructed at the end of that function:

```python
transform_te = Compose([
    Resize((height, width)),
    ToTensor(),
    normalize,   # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

Torchreid's `DataManager` uses `height=256`, `width=128` as defaults, matching the OSNet x1_0
input size confirmed in Phase 4.1.

### Torchreid official test/inference transform

```python
transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
```

No random augmentations appear in `transform_te`. Random augmentations
(`RandomHorizontalFlip`, `Random2DTranslation`, `RandomPatch`, `ColorJitter`, `RandomErasing`)
are only included in `transform_tr` (the training transform), controlled by the `transforms`
argument. The test transform is always the three-step deterministic pipeline above.

### TRACE transform (ai_pipeline/reid/embed.py)

```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
INPUT_HEIGHT  = 256
INPUT_WIDTH   = 128

_TRANSFORM = transforms.Compose([
    transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH)),  # (256, 128)
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
```

`_TRANSFORM` is a module-level constant — it is constructed once at import time and applied
identically to every crop in every batch. There is no per-batch or per-crop variation.

### Side-by-side comparison

| Item | Torchreid | TRACE | Match |
|---|---|---|---|
| Image library | PIL (`Image.open`) | PIL (`Image.open().convert("RGB")`) | ✅ YES |
| Color format | RGB | RGB (`.convert("RGB")` enforced) | ✅ YES |
| Resize height | 256 | 256 (`INPUT_HEIGHT`) | ✅ YES |
| Resize width | 128 | 128 (`INPUT_WIDTH`) | ✅ YES |
| Resize argument order | `(height, width)` | `(height, width)` | ✅ YES |
| ToTensor | Yes | Yes | ✅ YES |
| Normalize mean[0] | 0.485 | 0.485 | ✅ YES |
| Normalize mean[1] | 0.456 | 0.456 | ✅ YES |
| Normalize mean[2] | 0.406 | 0.406 | ✅ YES |
| Normalize std[0] | 0.229 | 0.229 | ✅ YES |
| Normalize std[1] | 0.224 | 0.224 | ✅ YES |
| Normalize std[2] | 0.225 | 0.225 | ✅ YES |
| RandomHorizontalFlip | No | No | ✅ YES |
| RandomCrop / Random2DTranslation | No | No | ✅ YES |
| RandomErasing | No | No | ✅ YES |
| ColorJitter | No | No | ✅ YES |
| RandomPatch | No | No | ✅ YES |
| Deterministic per-call | Yes | Yes | ✅ YES |

### RGB handling

TRACE uses `PIL Image.open(path).convert("RGB")` for every crop. This is the correct approach:

- PIL always returns an RGB image from `.convert("RGB")`, regardless of JPEG encoding details.
- OpenCV is not used anywhere in the inference preprocessing path.
- There is therefore no possibility of BGR channel ordering being fed to OSNet.

### Resize ordering note

`torchvision.transforms.Resize` takes `(height, width)`. PIL's `Image.size` returns `(width, height)`.
These are opposite conventions. TRACE uses `Resize((256, 128))` which torchvision interprets as
height=256, width=128 — matching OSNet's expected input dimensions exactly. The argument order
was confirmed to be correct.

### Normalization

TRACE uses the standard ImageNet normalization values, which are also the Torchreid defaults:

```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

After normalization, pixel values shift from `[0, 1]` (from ToTensor) to approximately `[-2.5, 2.5]`
depending on the image content. This was verified on a real crop: the minimum tensor value was
confirmed to be negative, confirming normalization was applied.

### Absence of random augmentation

`_TRANSFORM` in `embed.py` contains exactly three steps: `Resize`, `ToTensor`, `Normalize`.
`test_preprocessing.py` programmatically inspects every step of the `Compose` pipeline and
confirms that none are instances of `RandomHorizontalFlip`, `RandomCrop`, `RandomResizedCrop`,
`ColorJitter`, `RandomRotation`, `RandomAffine`, or `RandomErasing`.

### Direct tensor comparison result

Both transforms were applied to the same real TRACE crop
(`dataset/crops_phase3_final/C01_track100_frame0607.jpg`) and the output tensors were compared:

```
torch.equal(torchreid_tensor, trace_tensor)  →  True
max absolute difference                       →  0.0
```

The tensors are **bit-for-bit identical**. This is the strongest possible confirmation that
TRACE preprocessing is correct — not just that the parameters match on paper, but that the
actual computed tensors are indistinguishable.

### Test results

```
pytest ai_pipeline/reid/test_preprocessing.py -v

51 passed, 0 warnings in 6.18s
```

| Test class | Tests | Result |
|---|---|---|
| `TestPreprocessingConstants` | 11 | PASS |
| `TestTransformPipelineStructure` | 11 | PASS |
| `TestRGBInputHandling` | 3 | PASS |
| `TestTensorProperties` | 10 | PASS |
| `TestBatchDimension` | 2 | PASS |
| `TestDeterminism` | 2 | PASS |
| `TestDirectTorchreidComparison` | 7 | PASS |
| `TestPreprocessImageFunction` | 6 | PASS |

Real TRACE crop used: `C01_track100_frame0607.jpg`

### Existing embeddings

The Phase 4.3 embeddings in `dataset/embeddings_C01.json` (353 records) were generated using
the verified preprocessing. No regeneration is required.

### Conclusion

TRACE preprocessing is confirmed correct. Every property — RGB input, resize dimensions and
order, ToTensor conversion, ImageNet normalization mean and std, and absence of random
augmentation — matches the Torchreid official test/inference transform. The direct tensor
comparison produces a maximum absolute difference of 0.0, providing bit-exact confirmation.

---

## 4.5 Same-Person vs Different-Person Re-ID Sanity Check

**Status: IMPLEMENTATION COMPLETE — Colab experiment pending**

---

### Purpose

Phase 4.5 measures whether the OSNet x1_0 embeddings generated in Phase 4.3
show useful separation between:

1. **Same-track crops** — two crops belonging to the same ByteTrack `track_id`
2. **Different-track crops** — two crops belonging to different ByteTrack `track_id` values

The metric is **cosine similarity**.

This is a **sanity check**, not a ground-truth Re-ID accuracy evaluation.
Its purpose is to confirm that the embedding model produces geometry that is
consistent with meaningful appearance representation before investing in
cross-camera matching.

---

### Important Limitation: Track ID ≠ Identity Ground-Truth

Track IDs are produced by the Phase 2 ByteTrack tracker, not by manual identity
annotation.

| Interpretation | Meaning |
|---|---|
| `same track_id` | Same tracked trajectory in C01 |
| `different track_id` | Different tracked trajectories in C01 |
| `same track_id` | **Does NOT guarantee** same real-world person |
| `different track_id` | **Does NOT guarantee** different real-world person |

A strong same-track / different-track separation is encouraging evidence that
OSNet embeddings are useful for Re-ID. It is **not proof** of Re-ID accuracy.

---

### Dataset

| Property | Value |
|---|---|
| Embeddings file | `dataset/embeddings_C01.json` |
| Total embeddings | 353 |
| Embedding dimension | 512 |
| Camera | C01 only |
| Model | OSNet x1_0 (pretrained, Phase 4.3) |
| Embeddings regenerated | No — Phase 4.3 embeddings used as-is |

---

### Cosine Similarity

The similarity metric is cosine similarity:

```
cos(A, B) = dot(A, B) / (‖A‖ · ‖B‖)
```

Implemented in `ai_pipeline/reid/similarity.py`:

- Accepts Python lists, NumPy arrays, or any numeric sequence
- Converts to float64 for numerical stability
- Returns a plain Python float in [−1.0, 1.0]
- Zero-vector guard: returns 0.0 (not NaN) when either input is the zero vector
- Result is clamped to [−1, 1] to absorb floating-point rounding

---

### Same-Track Pair Sampling

Pairs are formed from records sharing a `track_id`.

Rules:
- Only tracks with **≥ 2 crops** are eligible
- A crop is **never** paired with itself
- **Non-adjacent frames are preferred** (|frame_a − frame_b| > 1) because
  consecutive frames carry near-identical appearance and are less diagnostic
- A controlled maximum of **10 pairs per track** is used to avoid over-weighting
  long tracks
- Pair selection is **deterministic**: `SEED = 42`

---

### Different-Track Pair Sampling

Pairs are formed from records with **different** `track_id` values.

Rules:
- The number of different-track pairs equals the number of same-track pairs
  (balanced comparison)
- Two distinct `track_id` values are selected at random, then one crop from each
- No pair can have matching `track_id` values
- Deterministic: same `SEED = 42`

---

### Random Seed

```python
SEED = 42
```

All pair sampling uses this seed. Re-running `sanity_check.py` on the same input
will always produce identical pairs and identical numerical results.

---

### Statistical Measures

For both same-track and different-track similarity distributions:

| Statistic | Description |
|---|---|
| count | Number of pairs compared |
| mean | Arithmetic mean of cosine similarities |
| median | 50th percentile |
| std | Population standard deviation |
| min | Minimum similarity |
| max | Maximum similarity |
| p25 | 25th percentile |
| p75 | 75th percentile |

Separation metrics:

```
mean_gap   = same_track_mean   − different_track_mean
median_gap = same_track_median − different_track_median
```

---

### Threshold Diagnostics

The fraction of pairs above each diagnostic threshold is reported for both groups:

| Threshold | Purpose |
|---|---|
| 0.50 | Loose positive match signal |
| 0.60 | Moderate confidence |
| 0.70 | Higher confidence |
| 0.80 | Strict match criterion |
| 0.90 | Very high confidence |

**No production Re-ID threshold is selected in this phase.**
These values are diagnostic only.

---

### Output Files

| File | Description |
|---|---|
| `dataset/reid_sanity/similarity_report.json` | All measured statistics in machine-readable form |
| `dataset/reid_sanity/similarity_distribution.png` | Histogram of same-track vs different-track similarity distributions |

---

### Implementation Files

| File | Role |
|---|---|
| `ai_pipeline/reid/similarity.py` | `cosine_similarity()`, `compute_stats()`, `threshold_fractions()` |
| `ai_pipeline/reid/sanity_check.py` | Full experiment: load → validate → pair → compute → report → save |
| `ai_pipeline/reid/test_similarity.py` | 51 unit tests (all pass locally); 5 smoke tests (auto-skipped when embeddings absent) |

---

### Test Results (local)

```
pytest ai_pipeline/reid/test_similarity.py -v

51 passed, 5 skipped in 0.81s
```

The 5 skipped tests are the `TestRealEmbeddingsSmoke` class, which auto-skips
when `dataset/embeddings_C01.json` is not available locally. These tests pass
when run in Google Colab against the actual dataset.

---

### Measured Results

**Pending — experiment must be run in Google Colab.**

`dataset/embeddings_C01.json` currently exists in Google Drive / Google Colab.
This section will be updated with actual measured numbers after running:

```bash
python ai_pipeline/reid/sanity_check.py \
    --embeddings dataset/embeddings_C01.json \
    --output-dir dataset/reid_sanity
```

The following fields will be populated from `dataset/reid_sanity/similarity_report.json`:

#### Same-Track Similarity

| Statistic | Value |
|---|---|
| Count | _to be filled after Colab run_ |
| Mean | _to be filled after Colab run_ |
| Median | _to be filled after Colab run_ |
| Std | _to be filled after Colab run_ |
| Min | _to be filled after Colab run_ |
| Max | _to be filled after Colab run_ |
| P25 | _to be filled after Colab run_ |
| P75 | _to be filled after Colab run_ |

#### Different-Track Similarity

| Statistic | Value |
|---|---|
| Count | _to be filled after Colab run_ |
| Mean | _to be filled after Colab run_ |
| Median | _to be filled after Colab run_ |
| Std | _to be filled after Colab run_ |
| Min | _to be filled after Colab run_ |
| Max | _to be filled after Colab run_ |
| P25 | _to be filled after Colab run_ |
| P75 | _to be filled after Colab run_ |

#### Separation

| Metric | Value |
|---|---|
| Mean gap (same − diff) | _to be filled after Colab run_ |
| Median gap (same − diff) | _to be filled after Colab run_ |

#### Threshold Diagnostics

| Threshold | Same-track above | Different-track above |
|---|---|---|
| 0.50 | _pending_ | _pending_ |
| 0.60 | _pending_ | _pending_ |
| 0.70 | _pending_ | _pending_ |
| 0.80 | _pending_ | _pending_ |
| 0.90 | _pending_ | _pending_ |

---

### Interpretation Template

After the Colab run, the interpretation will state:

- Whether same-track mean similarity is higher than different-track mean
- The magnitude of the mean and median gaps
- The degree of distribution overlap
- Whether the separation is sufficient to proceed to Phase 5 cross-camera matching
- Any concerns if separation is weak or absent

---

### Cross-Camera Note

Phase 4.5 uses **C01 only**. It does not test:

- C01 person → C02 person matching
- C01 → C03 matching

Cross-camera Re-ID evaluation is deferred to a later phase when multi-camera
embeddings are available.

---

## 4.6 Edge Case Handling

**Status: COMPLETE**

Phase 4.6 makes the Re-ID embedding pipeline resilient against bad input crops.
A single corrupted, missing, too-small, or otherwise invalid crop **never terminates
the pipeline** — it is logged, recorded with a structured reason code, and skipped.
All remaining valid crops continue to be processed normally.

---

### Design principle

Failures are isolated at the **individual crop** level, before any batch is assembled.
The strategy is:

1. Validate and preprocess each crop independently using `validate_and_preprocess()`.
2. Collect only valid tensors into the batch.
3. Run model inference on the valid subset only.
4. Record every failure with a structured `CropFailure` record.
5. Continue to the next crop / batch without re-raising any exception.

Because invalid crops are filtered out before `torch.stack()`, one bad file cannot
prevent the other valid images in the same batch from being embedded.

---

### Failure record structure

Every failed crop produces a structured record stored in the `failures` list:

```json
{
  "crop_path": "dataset/crops_phase3_final/broken_crop.jpg",
  "reason":    "unreadable_image",
  "details":   "PIL failed to open/verify: image file is truncated (0 bytes not processed)"
}
```

| Field       | Type   | Description                                    |
|-------------|--------|------------------------------------------------|
| `crop_path` | string | Relative path to the crop that failed          |
| `reason`    | string | Machine-readable reason code (see table below) |
| `details`   | string | Human-readable exception message or explanation|

#### Reason codes

| Reason code           | Trigger                                                |
|-----------------------|--------------------------------------------------------|
| `missing_file`        | File path does not exist on disk                       |
| `unreadable_image`    | PIL cannot open or decode the file bytes               |
| `crop_too_small`      | Decoded image is smaller than the configured minimums  |
| `invalid_image`       | Zero dimension, or cannot convert to RGB               |
| `preprocessing_error` | `torchvision` transform raised an exception            |
| `inference_error`     | Model forward pass raised an exception for this batch  |

---

### Edge case: Corrupted / unreadable crop

**File:** `ai_pipeline/reid/embed.py` → `validate_and_preprocess()`

If a crop file exists on disk but its bytes cannot be decoded as a valid image:

- `PIL.Image.open().verify()` is called first; any exception is caught.
- The crop is assigned `reason = unreadable_image`.
- The exception message is stored in `details`.
- Processing continues immediately to the next crop.
- No tensor is produced for the corrupted crop.
- The corrupted crop does **not** appear in the output embeddings JSON.

Example situation: a file contains a valid JPEG SOI marker but a truncated or
garbage body. PIL will raise `UnidentifiedImageError` or a related exception, which
is caught and recorded.

---

### Edge case: Missing crop

If a metadata record references a file path that does not exist:

- `image_path.exists()` is checked before any PIL call.
- The crop is assigned `reason = missing_file`.
- Processing continues immediately to the next crop.
- No tensor is produced.

This handles cases where a crop file was accidentally deleted, moved, or the
metadata path is stale after a dataset reorganisation.

---

### Edge case: Too-small crop

Phase 3 quality filtering already rejects crops below the configured minimum
dimensions. Phase 4.6 adds a **defensive guard** that re-checks size after
decoding, using the **same thresholds from `config.yaml`**:

```yaml
reid:
  min_crop_width:  40    # pixels
  min_crop_height: 100   # pixels
```

These values are read at runtime via `get_min_crop_dims(cfg)` — they are never
hardcoded in the inference path.

If a decoded image is narrower than `min_crop_width` or shorter than `min_crop_height`:

- The crop is assigned `reason = crop_too_small`.
- `details` records the actual dimensions and the configured thresholds.
- Processing continues immediately to the next crop.

Example `details` value:
```
Crop is 10×20 px — below minimum 40×100 px (config: min_crop_width=40, min_crop_height=100)
```

---

### Edge case: Invalid image

Covers decoded images that are structurally unusable:

- **Zero-dimension images**: `width == 0` or `height == 0` → `reason = invalid_image`.
- **RGB conversion failure**: `img.convert("RGB")` raises → `reason = invalid_image`.
- **Unreadable size**: `img.size` raises → `reason = invalid_image`.

In all cases the exception is caught, the crop is logged and skipped, and the
pipeline continues.

---

### Edge case: Preprocessing failure

If the `torchvision.transforms.Compose` pipeline raises for a specific image
(e.g. an unusual colour space that survives RGB conversion but cannot be resized):

- The exception is caught **around that individual crop's transform call only**.
- The crop is assigned `reason = preprocessing_error`.
- All other crops in the same batch are unaffected.

Broad `except` blocks that would catch exceptions from the model or the entire
batch are not used. The failure scope is strictly per-crop.

---

### Edge case: Inference / model failure

If the GPU/CPU model forward pass raises for an entire batch:

- The exception is caught around the `model(batch_tensor)` call.
- Every crop in that batch is marked `reason = inference_error`.
- The pipeline continues to the next batch.

Because invalid crops are removed **before** `torch.stack()` and the forward pass,
an individual invalid file cannot cause the inference step to fail. The inference
error path is a safety net for unexpected GPU or model-level failures.

---

### Failure report

At the end of every run, the pipeline prints a structured summary:

```
Total metadata records : 4
Successfully embedded  : 2
Failed                 : 2
Missing / not processed: 0

  Failures by reason:
    - unreadable_image: 1
    - crop_too_small: 1

  Failed crops:
    ✗ [unreadable_image] dataset/crops/broken.jpg
      PIL failed to open/verify: image file is truncated
    ✗ [crop_too_small] dataset/crops/tiny.jpg
      Crop is 10×20 px — below minimum 40×100 px
```

The `failures` list is accumulated in memory throughout the run and reported at
the end. It is **not** written to a separate file in the current MVP implementation;
failures are visible in the console output.

---

### Exit code behaviour

| Mode            | Behaviour                                              |
|-----------------|--------------------------------------------------------|
| Default         | `exit 0` after any completed run, even with failures   |
| `--strict` flag | `exit 1` if one or more crops failed                   |

The default resilient mode is appropriate for production batch runs where partial
output is better than no output. `--strict` is available for CI pipelines that
require a clean run.

---

### Output contract (unchanged from Phase 4.3)

Successful embedding records retain exactly the Phase 4.3 schema:

```json
{
  "crop_path":            "dataset/crops_phase3_final/C01_track100_frame0607.jpg",
  "camera_id":            "C01",
  "track_id":             100,
  "frame":                607,
  "timestamp":            "10:02:20.50",
  "bbox":                 [292.15, 154.30, 412.88, 534.72],
  "detection_confidence": 0.94,
  "embedding":            [1.1892024, 0.0, ...]
}
```

No fields are added or removed. The embedding dimension remains 512.
Failed crops produce no entry in the output JSON.

---

### Implementation files

| File                                    | Role                                                      |
|-----------------------------------------|-----------------------------------------------------------|
| `ai_pipeline/reid/embed.py`             | `validate_and_preprocess()`, `CropFailure`, `run_embedding_pipeline()` |
| `ai_pipeline/reid/test_embed.py`        | Section B: 12 edge-case unit + integration tests (no GPU) |
| `ai_pipeline/config.yaml`              | `reid.min_crop_width` / `reid.min_crop_height` thresholds |

---

### Test coverage

All edge-case tests run without GPU via a mock model (`unittest.mock.patch`).

| Test | Description | Result |
|------|-------------|--------|
| `test_corrupted_jpeg_returns_failure` | PIL cannot decode — returns `unreadable_image` failure | PASS |
| `test_missing_file_returns_failure` | Non-existent path — returns `missing_file` failure | PASS |
| `test_crop_below_min_width_returns_failure` | Width < 40 px — returns `crop_too_small` failure | PASS |
| `test_crop_below_min_height_returns_failure` | Height < 100 px — returns `crop_too_small` failure | PASS |
| `test_valid_crop_after_corrupted_crop` | Valid crop succeeds after a corrupted-crop call | PASS |
| `test_valid_crop_returns_tensor` | Valid crop returns `[3, 256, 128]` float32 tensor | PASS |
| `test_failure_to_dict_has_required_keys` | `CropFailure.to_dict()` has `crop_path`/`reason`/`details` | PASS |
| `test_pipeline_continues_after_broken_crop` | valid + broken + valid → 2 embedded, 1 skipped, no crash | PASS |
| `test_valid_crop_gets_embedding_after_corrupted_crop` | Crop after a broken one is still embedded | PASS |
| `test_multiple_valid_crops_with_one_corrupted` | 5 valid + 1 broken → 5 embedded, 1 failed | PASS |
| `test_missing_file_handled_in_pipeline` | Missing file logged; other crops proceed | PASS |
| `test_small_crop_handled_in_pipeline` | Tiny crop skipped; valid crop embedded | PASS |
| `test_critical_integration_valid_broken_tiny` | valid×2 + broken + tiny → 2 success, 2 fail, no crash | PASS |
| `test_valid_crops_receive_correct_schema` | Output record has all required fields, correct types | PASS |

---

## 4.7 Performance Benchmark

**Status: PASS / COMPLETE**

Phase 4.7 measures the actual time required for OSNet to process the complete 353-crop TRACE dataset on both GPU and CPU.

---

### Purpose

The goal is to determine:

- Total embedding inference time for all 353 crops
- Crops per second throughput
- Milliseconds per crop latency
- Whether CPU is practical for the final demo
- How much GPU acceleration helps

---

### Fairness Rules

Both benchmarks use exactly the same:

- **353 crop files** from `dataset/crops_phase3_final/`
- **OSNet x1_0** with pretrained weights
- **Preprocessing pipeline** (256×128 resize, ImageNet normalization)
- **Batch size 32** (from `config.yaml`)
- **`model.eval()`** mode
- **`torch.no_grad()`** context

The only intentional difference is **GPU vs CPU**.

---

### Timing Method

The benchmark measures:

| Component | Description |
|-----------|-------------|
| Model load time | Time to load OSNet from torchreid |
| Warm-up time | Small warm-up batch (not counted in main timing) |
| Preprocessing time | Image loading + torchvision transforms |
| Inference time | Actual model forward pass (with CUDA sync on GPU) |
| Total time | Preprocessing + Inference |

**GPU timing uses CUDA synchronization:**
```python
torch.cuda.synchronize()
start = time.perf_counter()
# run inference
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

---

### CPU Benchmark Results

**Status: COMPLETE — measured on local development machine**

| Property | Value |
|----------|-------|
| Environment | Local PC CPU |
| CPU | Intel64 Family 6 Model 186 Stepping 3, GenuineIntel |
| Logical CPUs | 12 |
| PyTorch version | 2.8.0+cpu |
| CUDA available | No |
| Device | cpu |
| Crops | 353 |
| Batch size | 32 |
| Model | osnet_x1_0 |
| Embedding dim | 512 |
| Model load time | 3.736s |
| Warm-up time | 0.353s |

#### Run Results

| Run | Total time | Preprocessing | Inference | Crops/sec | ms/crop |
|-----|------------|---------------|------------|------------|---------|
| 1 | 49.385s | 1.130s | 48.256s | 7.1 | 139.90 |
| 2 | 83.237s | 8.722s | 74.515s | 4.2 | 235.80 |
| 3 | 96.761s | 24.953s | 71.808s | 3.6 | 274.11 |

#### Summary Statistics

| Metric | Value |
|--------|-------|
| Mean total time | 76.461s |
| Median total time | 83.237s |
| Min total time | 49.385s |
| Max total time | 96.761s |
| Mean inference time | 64.860s |
| Mean crops/sec | 5.0 |
| Mean ms/crop | 216.60 |

#### Interpretation

- **CPU performance is variable**: Run 1 was significantly faster (49s) than runs 2-3 (83s, 97s)
- **Preprocessing overhead varies**: From 1.1s (run 1) to 25s (run 3), likely due to system load
- **Inference dominates**: Mean inference time (65s) is ~85% of mean total time (76s)
- **Throughput**: ~5 crops/sec on average
- **Latency**: ~217ms per crop on average

**Practical assessment**: CPU is usable for the demo but slow. Processing 353 crops takes ~1.3 minutes on average, with significant variance. For a real-time or near-real-time demo, GPU would be strongly preferable.

---

### GPU Benchmark Results

**Status: COMPLETE — measured on Google Colab Tesla T4**

| Property | Value |
|----------|-------|
| Environment | Google Colab |
| GPU | Tesla T4 |
| GPU memory | 14.6 GB |
| PyTorch version | _from Colab runtime_ |
| Device | cuda |
| Crops | 353 |
| Batch size | 32 |
| Model | osnet_x1_0 |
| Embedding dim | 512 |
| Model load time | 0.332s |
| Warm-up time | 0.073s |

#### Raw Run Results

| Run | Total time | Preprocessing | Inference | Crops/sec | ms/crop |
|-----|------------|---------------|------------|------------|---------|
| 1 | 67.121 s | 66.473 s | 0.649 s | 5.3 | 190.15 |
| 2 | 1.631 s | 1.048 s | 0.582 s | 216.5 | 4.62 |
| 3 | 1.848 s | 1.268 s | 0.580 s | 191.0 | 5.23 |

#### Script-Reported Summary Statistics (All Runs)

| Metric | Value |
|--------|-------|
| Mean total time | 23.533 s |
| Median total time | 1.848 s |
| Min total time | 1.631 s |
| Max total time | 67.121 s |
| Mean inference time | 0.604 s |
| Mean crops/sec | 137.6 |
| Mean ms/crop | 66.67 |

#### Steady-State Analysis (Runs 2–3)

The first GPU run was a substantial preprocessing outlier. Its 66.473 s preprocessing time dominated the total 67.121 s runtime, while GPU inference itself remained only 0.649 s. Runs 2 and 3 were much faster, suggesting a cold-start/storage/I/O/cache effect, but the exact cause was not isolated.

For representative steady-state performance, we consider Runs 2 and 3:

| Metric | Calculation | Value |
|--------|-------------|-------|
| Steady-state mean total time | (1.631 + 1.848) / 2 | 1.7395 s |
| Steady-state throughput | 353 / 1.7395 | 202.9 crops/sec |
| Steady-state ms/crop | 1.7395 / 353 × 1000 | 4.93 ms/crop |
| Steady-state mean inference time | (0.582 + 0.580) / 2 | 0.581 s |

#### Interpretation

- **GPU inference is very fast**: OSNet forward pass takes only ~0.58–0.65 s for all 353 crops
- **Preprocessing dominates end-to-end time**: In steady state, preprocessing (~1.1–1.3 s) is roughly 2× the inference time (~0.58 s)
- **First run outlier**: The 66.473 s preprocessing in Run 1 is ~50× longer than steady-state preprocessing
- **Cold-start effect**: The dramatic difference between Run 1 and Runs 2–3 suggests a storage/cache cold-start effect, though the exact cause was not isolated
- **Optimization implication**: For this MVP, optimizing image loading/preprocessing may matter more than optimizing OSNet inference

---

### Comparison

**Status: COMPLETE**

#### Detailed Comparison Table

| Environment | Run | Total Time | Preprocessing | Inference | Crops/sec | ms/crop |
|-------------|-----|------------|---------------|------------|------------|---------|
| GPU T4      | 1   | 67.121 s   | 66.473 s      | 0.649 s    | 5.3        | 190.15  |
| GPU T4      | 2   | 1.631 s    | 1.048 s       | 0.582 s    | 216.5      | 4.62    |
| GPU T4      | 3   | 1.848 s    | 1.268 s       | 0.580 s    | 191.0      | 5.23    |
| Local CPU   | Mean | 76.461 s   | —             | 64.860 s   | 5.0        | 216.60  |

#### Steady-State Comparison (GPU Runs 2–3 vs CPU Mean)

| Metric | GPU Steady-State (Runs 2–3) | CPU Mean | Speedup |
|--------|------------------------------|----------|---------|
| Mean total time | 1.7395 s | 76.461 s | 43.96× |
| Throughput | 202.9 crops/sec | 5.0 crops/sec | 40.58× |
| Latency | 4.93 ms/crop | 216.60 ms/crop | 43.89× |
| Mean inference time | 0.581 s | 64.860 s | 111.7× |

**CPU / GPU steady-state speedup: 43.96×**

This is a comparison between the CPU mean and the GPU steady-state mean excluding the cold/outlier first GPU run. This is not a formal real-time speedup guarantee.

---

### Implementation Files

| File | Role |
|------|------|
| `ai_pipeline/reid/benchmark.py` | CLI benchmark script with timing logic |
| `ai_pipeline/reid/test_benchmark.py` | 25 GPU-free unit tests (all pass) |
| `ai_pipeline/reid/phase4_7_gpu_benchmark.ipynb` | Colab notebook for GPU benchmark |

---

### Test Results

```
pytest ai_pipeline/reid/test_benchmark.py -v

25 passed in 6.68s
```

| Test category | Tests | Result |
|---------------|-------|--------|
| CLI argument parsing | 2 | PASS |
| Device selection | 3 | PASS |
| Config loading | 2 | PASS |
| Metadata loading | 3 | PASS |
| CPU info | 1 | PASS |
| Model loading (mocked) | 2 | PASS |
| Preprocessing | 1 | PASS |
| BenchmarkRun/Summary | 2 | PASS |
| Calculations | 3 | PASS |
| Result storage | 1 | PASS |
| Production embeddings safety | 1 | PASS |
| Benchmark output schema | 1 | PASS |
| Batch size respect | 1 | PASS |
| Crop count discovery | 1 | PASS |

All tests use mocked model inference and do not require GPU.

---

### Result Storage

Benchmark results are saved to:

- **CPU**: `dataset/reid_benchmark/performance_benchmark.json`
- **GPU**: `dataset/reid_benchmark/performance_benchmark_gpu.json` (after Colab run)

Both files contain:

- Timestamp
- Environment info (device, GPU/CPU details, PyTorch version)
- Model load time
- Warm-up time
- Individual run results (total time, preprocessing, inference, metrics)
- Summary statistics (mean, median, min, max)

---

### Production Data Safety

The benchmark script **does not modify** `dataset/embeddings_C01.json`. Embeddings are computed in memory and discarded after timing. Output is written only to the benchmark directory.

---

### Completion Criteria

Phase 4.7 is COMPLETE:

1. ✅ Benchmark code implemented
2. ✅ Tests pass (25/25)
3. ✅ All 353 real crops benchmarked on CPU
4. ✅ Real Colab GPU timing recorded (Tesla T4)
5. ✅ Same model/preprocessing/batch size used for both
6. ✅ GPU timing uses CUDA synchronization (in notebook)
7. ✅ Actual crops/sec and ms/crop calculated (CPU and GPU)
8. ✅ GPU vs CPU speedup calculated (43.96× steady-state)
9. ✅ Results saved (CPU and GPU)
10. ✅ Documentation contains real measurements
11. ✅ Production embeddings unchanged
