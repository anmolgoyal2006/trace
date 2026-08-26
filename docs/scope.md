# Trace — MVP Scope Document (v1)

> **Status: DRAFT — needs sign-off from all team members before implementation begins.**
> Add your name and date to the sign-off table at the bottom once you've read and agreed.

---

## 1. What We're Building

Trace is a multi-camera person tracking and re-identification system. Given a query person (e.g., a reference image or a short clip), the system searches a fixed set of cameras, reconstructs the person's route across them, and presents the result on an interactive dashboard.

The v1 goal is a working, demonstrable end-to-end system — not a production deployment.

---

## 2. In Scope (v1)

These are the only things we commit to finishing before the final demo.

### Cameras & Video
- **3 cameras** to start. Camera topology is defined in `dataset/camera_graph.json`.
- **Batch video processing only** — videos are pre-recorded files, not live streams.
- Each camera contributes a fixed-length video clip per session.

### AI Pipeline
- **Detection**: YOLOv8 (pretrained `yolov8n.pt` or `yolov8m.pt`) — no fine-tuning.
- **Tracking**: ByteTrack for within-camera identity continuity.
- **Re-ID**: Pretrained torchreid model (e.g., `osnet_x1_0`) — no custom training.
- **Matching / Fusion**: Basic weighted score combining appearance similarity, spatial plausibility (camera adjacency from the graph), and temporal plausibility (time-gap between sightings). No learned fusion — hand-tuned weights are fine for v1.
- **Route Reconstruction**: Graph traversal over matched sightings to output an ordered list of (camera, timestamp, confidence) tuples.

### Backend
- FastAPI REST API.
- **SQLite** for all persistence (sightings, tracks, routes, query sessions).
- Single-search-at-a-time — one active query session processes to completion before the next can be submitted.
- Endpoints: submit query, poll status, fetch route result, list cameras.

### Frontend
- React dashboard with two views:
  1. **Map view** — camera nodes on a static floor-plan or schematic, route highlighted as a path.
  2. **Timeline view** — horizontal timeline showing each camera sighting with timestamp and confidence.
- Query submission form (upload a reference image or select a tracked ID).
- No authentication, no multi-user support.

### Demo Criteria (Definition of Done for the Final Demo)
> The system passes the demo if:
> 1. A **blind test** video set (not used during development) is fed into the pipeline.
> 2. The reconstructed route matches the known ground-truth route (correct camera order).
> 3. Confidence scores are **sensible** — higher at cameras where the person was clearly visible, lower at ambiguous sightings. No single sighting should return 100% or 0% confidence without cause.
> 4. The dashboard renders the route and timeline without errors.
> 5. End-to-end latency (video ingestion → route displayed) is under **5 minutes** for 3 cameras × ~10-minute clips on a dev laptop.

---

## 3. Out of Scope / Stretch (post-v1)

These are explicitly deferred. If someone brings them up during v1 development, point them here.

| Feature | Why Deferred |
|---|---|
| Multi-search concurrency | Requires job queue (Celery/Redis), complicates state management — not worth it for a single-operator demo. |
| Real-time / live streaming | Batch processing is sufficient for the demo; streaming adds latency pipeline complexity. |
| Gait analysis | No labeled gait dataset; adds a separate model with uncertain accuracy gain. |
| Face recognition fusion | Raises privacy/ethical review requirements; out of scope for a research demo. |
| PostgreSQL migration | SQLite is adequate for single-user, single-process v1. Migration path is straightforward post-demo. |
| Re-ID model fine-tuning | Requires annotated multi-camera dataset we don't have; pretrained models are the baseline. |
| Learned fusion weights | Requires labeled matching data; hand-tuned weights are sufficient for v1. |
| User authentication | Single-operator demo — no user accounts needed. |
| Deployment / containerization | Local dev only for v1; Docker/cloud is a post-demo concern. |
| More than 3 cameras | Adding cameras is an extension, not a v1 requirement. |

---

## 4. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Pretrained re-ID model underperforms on our footage | Medium | Test on sample clips in week 1; if accuracy is poor, try a heavier model (e.g., `osnet_x0_25` → `osnet_x1_0`) before considering fine-tuning. |
| ByteTrack ID switches within a camera | Medium | Tune `track_thresh` and `match_thresh` on dev clips. Short-term ID switches are acceptable if cross-camera matching still recovers the route. |
| SQLite concurrency issues | Low | Single-search-at-a-time constraint avoids write contention entirely. |
| Scope creep ("can we just add X?") | High | This document. Point to the stretch table. |

---

## 5. Architecture Snapshot (v1)

```
raw_videos/
    cam_01.mp4  cam_02.mp4  cam_03.mp4
         │
    [Detection + Tracking]  ← YOLOv8 + ByteTrack (per camera)
         │
    [Re-ID Embedding]       ← pretrained torchreid
         │
    [Cross-Camera Matching] ← appearance + spatial + temporal fusion
         │
    [Route Reconstruction]  ← graph traversal on camera_graph.json
         │
    SQLite DB  ←→  FastAPI backend  ←→  React dashboard
                                          ├── Map view
                                          └── Timeline view
```

---

## 6. Decisions Already Made (Not Up for Re-Discussion in v1)

- **SQLite over PostgreSQL** — simple, zero-config, adequate for v1.
- **Pretrained models only** — no training infra needed.
- **Batch over streaming** — simpler pipeline, sufficient for demo.
- **3 cameras** — enough to prove multi-camera routing without over-engineering the graph.
- **Single search at a time** — avoids concurrency complexity with no demo cost.

---

## 7. Performance Baseline

Measured on Phase 1.7 (detection) and Phase 2.5 (detection + tracking). Same video and settings on two environments.

| Property | Value |
|---|---|
| Test video | `dataset/raw_videos/clip_001_C01.mp4` |
| Duration | 21.37 s |
| Total frames | 633 |
| FPS | 29.63 |
| Resolution | 1920 × 1080 |
| YOLO model | yolov8n.pt (pretrained, no fine-tuning) |
| Confidence threshold | 0.6 |
| Classes | [0] — person only |
| Processing mode | Every frame |

### Phase 1.7 — Detection only (YOLO `predict`)

| Environment | Total time | Time/frame | Processing / min of video | Realtime ratio |
|---|---|---|---|---|
| Local CPU | 135.42 s | ~214 ms | ~380 s/min | 6.3× slower |
| Colab Tesla T4 | 12.25 s | ~19 ms | ~34.4 s/min | 0.57× (faster than RT) |

### Phase 2.5 — Detection + ByteTrack (`track` stream mode)

| Environment | Total time | Time/frame | Processing / min of video | Realtime ratio |
|---|---|---|---|---|
| Local CPU | 109.87 s | ~174 ms | ~308.5 s/min | 5.1× slower |
| Colab Tesla T4 | (not yet measured) | — | — | — |

> ByteTrack adds negligible overhead. The tracking run is ~18% **faster** than detection-only because `model.track(stream=True)` uses an internal generator loop that is more efficient than calling `model.predict()` per frame in Python.

### Decision

Full-frame detection + tracking is feasible on the Colab T4 but too slow on CPU for the original 3-camera × 10-minute target.

For the MVP, **retain full-frame processing** while developing the pipeline. Consider frame sampling only if end-to-end processing becomes a bottleneck after all pipeline stages are integrated.

| Environment | 3 cameras × 10-min clips (estimated) | Feasible? |
|---|---|---|
| Local CPU (detection only) | ~190 minutes | No |
| Local CPU (detection + tracking) | ~154 minutes | No |
| Colab T4 (detection only) | ~17 minutes | Yes — acceptable for demo |
| Colab T4 (detection + tracking) | TBD | Expected yes |

---

## 8. Sign-Off

This document locks the v1 scope. Add your name below once you've read it and agree.

| Name | Role | Date |
|---|---|---|
| | | |
| | | |
| | | |

> Once all team members have signed off, update the status at the top from **DRAFT** to **AGREED**.
