# Phase 3 — Re-ID Data Preparation

**Status:** Complete  
**Camera:** C01  
**Video:** `dataset/raw_videos/clip_001_C01.mp4`  
**Completed phases:** 3.1 Sampling · 3.2 Crop Extraction · 3.3 Metadata Linking ·
3.4 Visual Verification · 3.5 Quality Filtering · 3.6 Storage & Count Sanity Check

---

## 3.1 Sampling Strategy

For each track, TRACE selects up to K evenly spaced frames across that track's
available lifetime, where K is configured in `config.yaml`.

```yaml
reid:
  crops_per_track: 5
```

Frames are selected by `ai_pipeline/reid/sampling.py` using integer linspace across
the sorted frame list for that track.  Tracks with fewer than K frames yield fewer
than K candidates; no padding or duplication is applied.

Observed in the C01 run:

| Track | Sampled | Reason for < 5 |
|------:|--------:|:---------------|
| 10    | 1       | Very short track (single observation window) |
| 39    | 1       | Very short track |
| 30    | 3       | Short track |
| 93    | 3       | Short track |
| 79    | 4       | Short track |

All other 71 tracks produced exactly 5 candidates each.

---

## 3.2 Crop Extraction

### Input video

| Property    | Value              |
|:------------|:-------------------|
| File        | `clip_001_C01.mp4` |
| Resolution  | 1920 × 1080 px     |
| FPS         | 29.625             |
| Frame count | 633                |
| Duration    | 21.37 s            |
| Camera ID   | C01                |

### Tracks input

| Property       | Value                    |
|:---------------|:-------------------------|
| File           | `dataset/tracks_C01.json` |
| Unique tracks  | 76                        |
| Total records  | (all frames for 76 tracks) |

### Extraction method

`ai_pipeline/reid/crop_extractor.py` — reads the tracks JSON, calls
`sampling.sample_track_frames()` to select up to K frames per track, then seeks
the video with `cv2.VideoCapture.set(CAP_PROP_POS_FRAMES)` and crops the
bounding box region from the decoded frame.  No YOLO or ByteTrack is re-run; the
extractor only reads pre-computed bounding boxes.

### Crop naming convention

```
{camera_id}_track{track_id}_frame{frame:04d}.jpg
```

Example: `C01_track37_frame0195.jpg`

---

## 3.3 Metadata Linking

### Metadata file

`dataset/crops_metadata_phase3_final.json`

### Metadata fields (per record)

| Field                  | Type         | Description                                      |
|:-----------------------|:-------------|:-------------------------------------------------|
| `crop_path`            | `str`        | Relative path to the JPEG from repo root         |
| `camera_id`            | `str`        | Camera identifier (e.g. `"C01"`)                 |
| `track_id`             | `int`        | ByteTrack track ID                               |
| `frame`                | `int`        | 0-based frame index in the source video          |
| `timestamp`            | `str`        | Wall-clock time derived from camera start time   |
| `bbox`                 | `list[float]`| `[x1, y1, x2, y2]` in pixels                    |
| `detection_confidence` | `float`      | YOLO detection confidence score ∈ [0, 1]         |

### Crop-to-metadata relationship

Every accepted JPEG has exactly one metadata record.  The relationship is
enforced by writing the metadata list only from the list of successfully saved
crops, so no record can exist without its file.

### Validation result (Phase 3.3 validator — run in Phase 3.6)

```
Metadata records        : 353
Actual crop images      : 353
Referenced images found : 353
Missing crop files      : 0
Orphan crop images      : 0
Orphan metadata entries : 0
Duplicate crop paths    : 0
Invalid metadata records: 0
Unreadable images       : 0
RESULT: PASS
```

---

## 3.4 Visual Verification

### Random crop inspection

A random sample of accepted crops was rendered as a contact sheet and inspected
visually (`dataset/crop_verification/random_samples.jpg`).  All sampled crops
showed recognisable person silhouettes with correct bounding boxes.

### Full-track inspection

Track 37 was inspected at the full-frame level (`dataset/crop_verification/track_37.jpg`)
to investigate apparent bounding-box movement across frames.  The trajectory was
consistent with a person walking through the scene.

### Track 37 investigation

Track 37's frames were examined to determine whether an ID switch had occurred.
The full-frame trajectory showed continuous motion along a plausible path.  No
confirmed hard ID switch was established from the available evidence.  The
investigation was inconclusive — it was not possible to rule out a soft merge, but
no clear discontinuity was observed.

### Conclusion

Visual inspection did not surface any systematic extraction error.  Bounding boxes
aligned with detected persons across all spot-checked tracks.

---

## 3.5 Quality Filtering

### Configuration

```yaml
reid:
  min_crop_width: 40          # raised from 30 in Phase 3.5
  min_crop_height: 100        # raised from 60 in Phase 3.5
  quality:
    reject_severe_edge_truncation: true
    edge_margin_pixels: 5
```

### Filtering rules (applied in order)

**Rule 0 — bbox sanity**  
Reject if the bounding box is `None` or does not contain exactly 4 coordinates,
or if it degenerates to zero area after clipping to the frame.

**Rule 1 — minimum size**  
Reject if the clipped crop width < `min_crop_width` (40 px)
OR the clipped crop height < `min_crop_height` (100 px).

**Rule 2 — severe edge truncation** *(optional, enabled)*  
Reject when a bbox simultaneously:
- touches or is within `edge_margin_pixels` (5 px) of any frame edge, **and**
- the crop dimension along that axis is less than half the frame dimension
  (i.e. < 960 px wide or < 540 px tall).

This accepts people who are merely entering or exiting frame at the edge
(they touch the boundary but occupy a reasonable fraction of the frame).
It rejects only bboxes that are both edge-hugging and heavily cut off,
meaning most of the person's body is off-screen.

### Rejection counts (Phase 3.5 dry-run)

The dry-run in Phase 3.5 predicted:

- Selected: 367 · Accepted: 353 · Rejected: 14 · Rejection rate: 3.8 %
- All 14 rejected due to `severely_truncated`

The dry-run did not write any files.

---

## 3.6 Final Storage & Count Sanity Check

### Command used

```bash
python ai_pipeline/reid/crop_extractor.py \
    --video          dataset/raw_videos/clip_001_C01.mp4 \
    --tracks         dataset/tracks_C01.json \
    --output-dir     dataset/crops_phase3_final \
    --metadata       dataset/crops_metadata_phase3_final.json \
    --quality-report dataset/crop_quality_report_phase3_final.json
```

Run from repo root `c:\Users\anmol\OneDrive\Desktop\Trace`.  No `--dry-run` flag.

### Measured results

| Metric                      | Measured Value     |
|:----------------------------|-------------------:|
| Video duration              | 21.37 s            |
| FPS                         | 29.625             |
| Frames                      | 633                |
| Resolution                  | 1920 × 1080 px     |
| Unique tracks               | 76                 |
| Sampled candidates          | 367                |
| Accepted crops              | 353                |
| Rejected crops              | 14                 |
| Rejection rate              | 3.8 %              |
| Tracks with zero valid crops| 0                  |
| Crop images on disk         | 353                |
| Metadata records            | 353                |
| Orphan crops                | 0                  |
| Orphan metadata             | 0                  |
| Total storage               | 3.77 MB            |
| Average crop size           | 10.94 KB           |
| Smallest crop file          | 2.71 KB            |
| Largest crop file           | 57.26 KB           |

### Rejection details (all 14)

All 14 rejections were caused by **`severely_truncated`** — the person's bounding
box was hugging a frame edge and the visible crop was less than half the frame
width (< 960 px).  No crops were rejected for being too small.

| Track | Frame | Edge     | Crop size   | Details |
|------:|------:|:---------|:------------|:--------|
| 1     | 388   | right    | 188 × 956   | x2=1920 ≥ 1915, crop_w=188 < 960 |
| 8     | 135   | left     | 137 × 566   | x1=0 ≤ 5, crop_w=137 < 960 |
| 13    | 202   | right    | 238 × 616   | x2=1920 ≥ 1915, crop_w=238 < 960 |
| 23    | 214   | left     | 100 × 539   | x1=0 ≤ 5, crop_w=100 < 960 |
| 37    | 195   | left     | 102 × 813   | x1=3 ≤ 5, crop_w=102 < 960 |
| 37    | 405   | right    | 369 × 1001  | x2=1920 ≥ 1915, crop_w=369 < 960 |
| 45    | 257   | right    | 131 × 583   | x2=1915 ≥ 1915, crop_w=131 < 960 |
| 57    | 303   | right    | 60 × 436    | x2=1919 ≥ 1915, crop_w=60 < 960 |
| 64    | 362   | right    | 159 × 819   | x2=1920 ≥ 1915, crop_w=159 < 960 |
| 64    | 368   | right    | 93 × 585    | x2=1920 ≥ 1915, crop_w=93 < 960 |
| 86    | 524   | right    | 83 × 585    | x2=1917 ≥ 1915, crop_w=83 < 960 |
| 86    | 619   | right    | 270 × 894   | x2=1920 ≥ 1915, crop_w=270 < 960 |
| 92    | 567   | right    | 65 × 561    | x2=1918 ≥ 1915, crop_w=65 < 960 |
| 100   | 620   | right    | 243 × 626   | x2=1920 ≥ 1915, crop_w=243 < 960 |

All 14 are persons entering or exiting the right or left frame edge with most of
their body outside the frame boundary.  These crops contain too little visual
information to be useful for Re-ID embedding.

### Per-track summary

| track_id | sampled | accepted | rejected | status |
|---------:|--------:|---------:|---------:|:-------|
| 1        | 5       | 4        | 1        | OK     |
| 2        | 5       | 5        | 0        | OK     |
| 3        | 5       | 5        | 0        | OK     |
| 4        | 5       | 5        | 0        | OK     |
| 5        | 5       | 5        | 0        | OK     |
| 6        | 5       | 5        | 0        | OK     |
| 7        | 5       | 5        | 0        | OK     |
| 8        | 5       | 4        | 1        | OK     |
| 9        | 5       | 5        | 0        | OK     |
| 10       | 1       | 1        | 0        | OK     |
| 11       | 5       | 5        | 0        | OK     |
| 13       | 5       | 4        | 1        | OK     |
| 15       | 5       | 5        | 0        | OK     |
| 16       | 5       | 5        | 0        | OK     |
| 17       | 5       | 5        | 0        | OK     |
| 18       | 5       | 5        | 0        | OK     |
| 20       | 5       | 5        | 0        | OK     |
| 22       | 5       | 5        | 0        | OK     |
| 23       | 5       | 4        | 1        | OK     |
| 25       | 5       | 5        | 0        | OK     |
| 28       | 5       | 5        | 0        | OK     |
| 29       | 5       | 5        | 0        | OK     |
| 30       | 3       | 3        | 0        | OK     |
| 32       | 5       | 5        | 0        | OK     |
| 34       | 5       | 5        | 0        | OK     |
| 35       | 5       | 5        | 0        | OK     |
| 37       | 5       | 3        | 2        | OK     |
| 39       | 1       | 1        | 0        | OK     |
| 40       | 5       | 5        | 0        | OK     |
| 42       | 5       | 5        | 0        | OK     |
| 44       | 5       | 5        | 0        | OK     |
| 45       | 5       | 4        | 1        | OK     |
| 46       | 5       | 5        | 0        | OK     |
| 47       | 5       | 5        | 0        | OK     |
| 48       | 5       | 5        | 0        | OK     |
| 49       | 5       | 5        | 0        | OK     |
| 51       | 5       | 5        | 0        | OK     |
| 52       | 5       | 5        | 0        | OK     |
| 53       | 5       | 5        | 0        | OK     |
| 54       | 5       | 5        | 0        | OK     |
| 55       | 5       | 5        | 0        | OK     |
| 56       | 5       | 5        | 0        | OK     |
| 57       | 5       | 4        | 1        | OK     |
| 60       | 5       | 5        | 0        | OK     |
| 61       | 5       | 5        | 0        | OK     |
| 62       | 5       | 5        | 0        | OK     |
| 64       | 5       | 3        | 2        | OK     |
| 65       | 5       | 5        | 0        | OK     |
| 66       | 5       | 5        | 0        | OK     |
| 67       | 5       | 5        | 0        | OK     |
| 69       | 5       | 5        | 0        | OK     |
| 70       | 5       | 5        | 0        | OK     |
| 71       | 5       | 5        | 0        | OK     |
| 73       | 5       | 5        | 0        | OK     |
| 74       | 5       | 5        | 0        | OK     |
| 75       | 5       | 5        | 0        | OK     |
| 76       | 5       | 5        | 0        | OK     |
| 79       | 4       | 4        | 0        | OK     |
| 80       | 5       | 5        | 0        | OK     |
| 82       | 5       | 5        | 0        | OK     |
| 83       | 5       | 5        | 0        | OK     |
| 85       | 5       | 5        | 0        | OK     |
| 86       | 5       | 3        | 2        | OK     |
| 87       | 5       | 5        | 0        | OK     |
| 88       | 5       | 5        | 0        | OK     |
| 89       | 5       | 5        | 0        | OK     |
| 91       | 5       | 5        | 0        | OK     |
| 92       | 5       | 4        | 1        | OK     |
| 93       | 3       | 3        | 0        | OK     |
| 94       | 5       | 5        | 0        | OK     |
| 96       | 5       | 5        | 0        | OK     |
| 100      | 5       | 4        | 1        | OK     |
| 102      | 5       | 5        | 0        | OK     |
| 105      | 5       | 5        | 0        | OK     |
| 107      | 5       | 5        | 0        | OK     |
| 110      | 5       | 5        | 0        | OK     |
| **TOTAL**| **367** | **353**  | **14**   |        |

### Tracks with zero valid crops

None — every track produced at least one valid crop.

### Accepted crop dimensions

| Dimension | Minimum | Maximum |
|:----------|--------:|--------:|
| Width     | 44 px   | 367 px  |
| Height    | 128 px  | 1036 px |

All accepted crops meet the minimum 40 × 100 px threshold.

### Comparison with Phase 3.5 dry-run prediction

| Metric     | Phase 3.5 dry-run | Phase 3.6 actual run |
|:-----------|------------------:|---------------------:|
| Selected   | 367               | 367                  |
| Accepted   | 353               | 353                  |
| Rejected   | 14                | 14                   |
| Reasons    | severely_truncated (14) | severely_truncated (14) |

**The actual run matched the dry-run prediction exactly.**

### Output files

| File                                           | Description                              |
|:-----------------------------------------------|:-----------------------------------------|
| `dataset/crops_phase3_final/`                  | 353 accepted JPEG crops                  |
| `dataset/crops_metadata_phase3_final.json`     | 353 metadata records (1-to-1 with crops) |
| `dataset/crop_quality_report_phase3_final.json`| Full rejection log with per-crop details |
| `dataset/crop_verification/phase3_final_samples.jpg` | 16-thumbnail contact sheet (sanity check) |

### Validation

The Phase 3.3 metadata validator was run against the final dataset:

```
RESULT: PASS
```

All checks passed: required fields, data types, confidence range, bounding boxes,
path safety, no missing files, no orphan images, no orphan metadata, no duplicates,
all images readable by OpenCV.

---

## Phase 3 Complete

The C01 Re-ID crop dataset is ready for Phase 4 embedding generation.

| Deliverable                    | Location                                      |
|:-------------------------------|:----------------------------------------------|
| Accepted crops                 | `dataset/crops_phase3_final/` (353 JPEGs)     |
| Metadata                       | `dataset/crops_metadata_phase3_final.json`    |
| Quality report                 | `dataset/crop_quality_report_phase3_final.json` |
| Verification image             | `dataset/crop_verification/phase3_final_samples.jpg` |
| Per-track sanity script        | `ai_pipeline/reid/per_track_sanity.py`        |
| Crop extractor                 | `ai_pipeline/reid/crop_extractor.py`          |
| Quality filter                 | `ai_pipeline/reid/quality_filter.py`          |
| Metadata validator             | `ai_pipeline/reid/validate_crop_metadata.py`  |
