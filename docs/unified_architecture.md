# Trace — Unified Architecture (v2)

> Merges the best of the original **Trace** pipeline (body re-ID, route
> reconstruction, calibrated thresholds, quality filtering) with the best
> of **TraceAI** (live dashboard, WebSocket alerts, watchlist, identity
> registry, video upload, analytics).

---

## 1. What Changed From v1 Scope

| Area | v1 Scope | Unified v2 |
|---|---|---|
| Re-ID model | OSNet x1_0 (body) | **OSNet x1_0 (body) — kept** — more robust than face-only ArcFace for CCTV |
| Processing | Batch only | Batch + **video file upload via API** (same batch engine, triggered by endpoint) |
| Route reconstruction | Planned but not wired | **Fully wired** — spatial + temporal fusion over camera_graph.json |
| Dashboard | React map + timeline | Map + timeline + **live alert feed + watchlist panel + analytics** |
| Backend | FastAPI stub | **Full FastAPI** — persons, cameras, queries, analytics, upload, WebSocket |
| Real-time | Out of scope | **WebSocket** for query progress updates and alert push (no live stream — batch only) |
| Identity registry | None | **Persons registry** — register persons with reference photo, watchlist status |
| Face search | None | **Image-based search** — upload a photo, system finds matching tracks |

---

## 2. Architecture Diagram

```
dataset/raw_videos/
  cam_01.mp4  cam_02.mp4  cam_03.mp4
        │
        ▼  (triggered by API upload or pre-processed files)
┌─────────────────────────────────────────────────────────────────┐
│                       AI PIPELINE                               │
│                                                                 │
│  DetectionService      → YOLOv8n  (detect.py wrapper)          │
│  TrackerService        → ByteTrack (track.py wrapper)           │
│  CropExtractorService  → crop_extractor.py wrapper              │
│  EmbeddingService      → OSNet x1_0 (embed.py wrapper)         │
│  MatchingService       → cosine similarity + track aggregation  │
│  RouteService          → spatial+temporal fusion + graph walk   │
└───────────────────────────────┬─────────────────────────────────┘
                                │ persists to
                                ▼
                    ┌───────────────────────┐
                    │  SQLite  (trace.db)   │
                    │  persons              │
                    │  cameras              │
                    │  sightings            │
                    │  detections           │
                    │  routes               │
                    │  query_sessions       │
                    │  alerts               │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   FastAPI Backend     │
                    │   :8000               │
                    │                       │
                    │  /api/persons         │
                    │  /api/cameras         │
                    │  /api/queries         │
                    │  /api/analytics       │
                    │  /api/upload          │
                    │  /ws  (WebSocket)     │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   React Frontend      │
                    │                       │
                    │  Map View             │
                    │  Timeline View        │
                    │  Query Form           │
                    │  Alert Feed           │
                    │  Watchlist Panel      │
                    │  Analytics            │
                    └───────────────────────┘
```

---

## 3. Project File Structure (v2)

```
Trace/
├── ai_pipeline/                   ← EXISTING — no changes to core logic
│   ├── config.yaml
│   ├── detection/detect.py
│   ├── tracking/track.py
│   └── reid/
│       ├── crop_extractor.py
│       ├── embed.py
│       ├── similarity.py
│       ├── confidence_scaling.py
│       ├── track_aggregation.py
│       └── target_search.py
│
├── backend/
│   ├── app/
│   │   ├── main.py                ← FastAPI app + lifespan + static mount
│   │   ├── config.py              ← pydantic settings (paths, thresholds)
│   │   ├── database.py            ← SQLAlchemy engine + session factory
│   │   ├── models/
│   │   │   ├── orm.py             ← SQLAlchemy ORM table definitions
│   │   │   └── schemas.py         ← Pydantic request/response schemas
│   │   ├── routers/
│   │   │   ├── persons.py         ← CRUD + face enroll + image search
│   │   │   ├── cameras.py         ← CRUD + camera graph
│   │   │   ├── queries.py         ← submit, poll, results
│   │   │   ├── analytics.py       ← dashboard stats, timeline, alerts
│   │   │   └── upload.py          ← video file upload + processing trigger
│   │   ├── services/
│   │   │   ├── detection_service.py   ← wraps ai_pipeline/detection/detect.py
│   │   │   ├── tracker_service.py     ← wraps ai_pipeline/tracking/track.py
│   │   │   ├── embedding_service.py   ← wraps ai_pipeline/reid/embed.py
│   │   │   ├── matching_service.py    ← cosine sim + track aggregation
│   │   │   └── route_service.py       ← spatial+temporal fusion + graph walk
│   │   └── core/
│   │       └── websocket_manager.py   ← broadcast manager
│   └── requirements.txt
│
├── frontend/
│   ├── index.html
│   └── src/
│       ├── app.js
│       ├── api.js
│       └── styles.css
│
├── dataset/
│   ├── camera_graph.json          ← EXISTING
│   ├── raw_videos/                ← input videos
│   ├── crops/                     ← extracted person crops
│   └── embeddings_*.json          ← per-camera embedding stores
│
└── docs/
    ├── scope.md
    └── unified_architecture.md    ← this file
```

---

## 4. Database Schema

### 4.1 `persons`
| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | |
| name | TEXT | |
| alias | TEXT | optional |
| description | TEXT | optional |
| watchlist_status | TEXT | none / missing / suspect / poi |
| reference_embedding | TEXT | JSON-serialised float list (512-d OSNet) |
| reference_image_path | TEXT | path to uploaded reference photo |
| is_active | BOOLEAN | soft delete |
| created_at | DATETIME | |

### 4.2 `cameras`
| Column | Type | Notes |
|---|---|---|
| id | TEXT PK | C01, C02, C03 |
| location | TEXT | from camera_graph.json |
| description | TEXT | |
| start_time | TEXT | wall-clock HH:MM:SS |
| is_active | BOOLEAN | |

### 4.3 `query_sessions`
| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | |
| person_id | INTEGER FK→persons | null if ad-hoc image query |
| query_image_path | TEXT | uploaded query photo path |
| status | TEXT | pending / running / done / failed |
| submitted_at | DATETIME | |
| completed_at | DATETIME | |
| error_message | TEXT | if failed |

### 4.4 `sightings`
Aggregated per-camera match result for a query session.

| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | |
| session_id | INTEGER FK→query_sessions | |
| camera_id | TEXT FK→cameras | |
| track_id | INTEGER | best matching ByteTrack ID |
| first_seen | TEXT | wall-clock timestamp |
| last_seen | TEXT | |
| best_confidence | REAL | highest confidence crop in this track |
| mean_confidence | REAL | mean of top-3 crops |
| crop_path | TEXT | path to best matching crop |
| appearance_score | REAL | raw cosine similarity |
| spatial_score | REAL | camera adjacency plausibility [0,1] |
| temporal_score | REAL | time-gap plausibility [0,1] |
| fusion_score | REAL | weighted combination of above |

### 4.5 `routes`
Ordered reconstruction of the person's path across cameras.

| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | |
| session_id | INTEGER FK→query_sessions | |
| step_order | INTEGER | 0-indexed position in route |
| sighting_id | INTEGER FK→sightings | |

### 4.6 `alerts`
| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | |
| session_id | INTEGER FK→query_sessions | |
| sighting_id | INTEGER FK→sightings | |
| severity | TEXT | HIGH / MEDIUM / LOW |
| title | TEXT | |
| message | TEXT | |
| acknowledged | BOOLEAN | |
| created_at | DATETIME | |

---

## 5. API Contract

### Persons
| Method | Path | Description |
|---|---|---|
| POST | /api/persons | Register person (name, watchlist, photo) |
| GET | /api/persons | List all persons |
| GET | /api/persons/{id} | Get person + sighting history |
| PUT | /api/persons/{id} | Update person profile |
| DELETE | /api/persons/{id} | Soft delete |
| POST | /api/persons/{id}/enroll | Upload reference photo → generate OSNet embedding |
| POST | /api/persons/search | Upload photo → find matching persons |

### Cameras
| Method | Path | Description |
|---|---|---|
| GET | /api/cameras | List cameras with graph info |
| GET | /api/cameras/{id} | Camera detail + recent sightings |

### Queries
| Method | Path | Description |
|---|---|---|
| POST | /api/queries | Submit query (person_id or image upload) |
| GET | /api/queries/{id} | Poll status |
| GET | /api/queries/{id}/route | Get reconstructed route |
| GET | /api/queries/{id}/sightings | All sightings for session |

### Analytics
| Method | Path | Description |
|---|---|---|
| GET | /api/analytics/dashboard | Aggregated stats |
| GET | /api/analytics/alerts | Recent alerts |
| POST | /api/analytics/alerts/{id}/acknowledge | Ack an alert |
| GET | /api/analytics/activity | Per-camera activity summary |

### Upload
| Method | Path | Description |
|---|---|---|
| POST | /api/upload/video | Upload a video file for batch processing |
| GET | /api/upload/status/{job_id} | Poll processing job status |

### WebSocket
| Path | Events |
|---|---|
| /ws | query_progress, sighting_found, route_complete, alert, error |

---

## 6. Key Design Decisions

### 6.1 Body Re-ID over Face Recognition
- **Kept OSNet x1_0** from the original Trace pipeline.
- TraceAI's ArcFace fails when subjects face away, wear masks, or are far from camera.
- Body appearance is more reliable for typical CCTV overhead angles.
- The reference image enrollment uses the same OSNet pipeline — consistent embedding space.

### 6.2 Route Reconstruction (unique to this system)
The route service does three things TraceAI doesn't:
1. **Appearance score** — OSNet cosine similarity (already built).
2. **Spatial score** — camera adjacency from `camera_graph.json`. Non-adjacent cameras get a heavy penalty.
3. **Temporal score** — gaussian decay around expected transit time from the graph edge weights. A sighting that arrives exactly `avg_transit_sec` after the previous one scores 1.0; sightings that arrive impossibly fast or too late decay toward 0.
4. **Fusion** — weighted sum: `0.6 × appearance + 0.25 × spatial + 0.15 × temporal`.
5. **Graph walk** — greedy best-first traversal over the camera adjacency graph using fusion scores.

### 6.3 WebSocket — Progress Not Live Video
- No live stream, no frame preview (batch system, consistent with scope).
- WebSocket pushes: query progress %, sighting found per camera, route complete, watchlist alert.
- Keeps the real-time feel of TraceAI's dashboard without the complexity of live video pipelines.

### 6.4 Single Search at a Time
- Maintained from v1 scope — avoids job queue complexity.
- Backend enforces: reject new query if one is already `running`.
- WebSocket lets the frontend show progress without polling.

### 6.5 Fusion Weights
```
appearance_weight = 0.60   # OSNet cosine similarity is the primary signal
spatial_weight    = 0.25   # camera graph adjacency
temporal_weight   = 0.15   # transit time plausibility
```
Hand-tuned for MVP. Adjustable in `backend/app/config.py`.

---

## 7. Frontend Views

### Map View
- Static SVG floor plan (or schema diagram) with camera nodes C01–C03.
- Route highlighted as an animated path as each sighting is confirmed.
- Click a camera node → shows sighting detail panel (timestamp, confidence, crop thumbnail).

### Timeline View
- Horizontal timeline, one row per camera.
- Each sighting rendered as a marker at the correct timestamp position.
- Marker color encodes confidence (green ≥ 80, yellow 60–80, red < 60).
- Hover → tooltip with camera, timestamp, confidence score.

### Query Form
- Upload a reference photo or select a registered person from dropdown.
- Submit button → triggers POST /api/queries.
- Progress bar driven by WebSocket `query_progress` events.

### Alert Feed
- Real-time list of alerts pushed via WebSocket.
- Severity badge (HIGH = red, MEDIUM = amber, LOW = grey).
- Acknowledge button per alert.

### Watchlist Panel
- Table of registered persons with watchlist status badges.
- Quick-add person form.
- Each row links to that person's full sighting history.

### Analytics Panel
- Stat cards: total queries, total sightings, watchlist hits, cameras active.
- Bar chart: queries per day.
- Per-camera detection count.

---

## 8. Fusion Score Formula

```
For each candidate sighting (camera Ci, track Ti):

  appearance_score(Ci, Ti) = mean_top3_cosine_similarity(Ti)

  spatial_score(Ci, Ti) =
    1.0   if Ci is adjacent to previous camera in route
    0.3   if Ci is two hops away
    0.0   if Ci is not reachable from previous camera

  temporal_score(Ci, Ti) =
    gaussian(actual_gap_sec, mean=avg_transit_sec, sigma=avg_transit_sec/2)
    clamped to [0, 1]

  fusion_score = 0.60 × appearance + 0.25 × spatial + 0.15 × temporal
```

---

## 9. Implementation Order

1. `backend/app/database.py` + `models/orm.py` — schema first
2. `models/schemas.py` — API contracts
3. `services/` — AI service wrappers (detection, tracking, embedding, matching, route)
4. `routers/` + `main.py` — FastAPI app
5. `frontend/` — React dashboard
6. End-to-end smoke test

