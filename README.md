# TRACE — Multi-Camera Person Re-Identification

TRACE is an AI-powered surveillance system.  
Upload CCTV videos from multiple cameras, submit a reference photo of any person, and the system automatically finds where they appeared and reconstructs their route with timestamps.

**Full pipeline:** YOLOv8n detection → BoT-SORT tracking → OSNet body embeddings → SCRFD + ArcFace face embeddings → KPR part-based embeddings → triple-signal fusion → cross-camera route reconstruction.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10 or 3.11 | 3.9 also works |
| pip | latest | comes with Python |
| RAM | 8 GB minimum | 16 GB recommended |
| Disk space | ~3 GB | for weights + deps |
| GPU | optional | CUDA speeds things up; CPU works fine |

> **Windows:** During Python installation tick **"Add Python to PATH"**.

---

## Setup (one-time)

### 1. Clone the repository

```bash
git clone https://github.com/anmolgoyal2006/trace.git
cd trace
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install all Python dependencies

```bash
pip install -r backend/requirements.txt
```

This installs every dependency in one shot: FastAPI, PyTorch, Ultralytics (YOLOv8 + BoT-SORT), torchreid (OSNet), ONNX Runtime, OpenCV, and everything else.

> ⚠️ PyTorch is ~800 MB. This step can take 5–30 minutes on a slow connection. That is normal.

**If torchreid fails to install from PyPI, install it from source:**
```bash
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
pip install -r backend/requirements.txt
```

**macOS (Apple Silicon):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt
```

**If you see a libGL / OpenCV DLL error on any platform:**
```bash
pip uninstall opencv-python -y
pip install opencv-python-headless==4.9.0.80
```

### 4. Install KPR dependencies

KPR is already cloned in the repo at `keypoint_promptable_reidentification/`. Install its dependencies:

```bash
cd keypoint_promptable_reidentification
pip install -r requirements.txt
python setup.py develop
cd ..
```

> If `python setup.py develop` fails, try `pip install -e .` from the same directory.

### 5. Install ONNX Runtime (for face pipeline)

```bash
# CPU (works everywhere)
pip install onnxruntime

# GPU (CUDA 11+, faster)
pip install onnxruntime-gpu
```

### 6. Copy the environment config

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

The `.env` file already has all model paths pre-configured pointing at the weight files that ship with this repo. **No edits needed.**

---

## Run the server

From the **project root** (`trace/` directory):

```bash
python run.py
```

Or equivalently:

```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

> ⚠️ Always run from the project root, not from inside `backend/` or any subdirectory.

On first startup you will see:
```
[startup] OSNet model loaded
[startup] Cameras seeded
[startup] PipelineService initialised
[startup] Dashboard → http://0.0.0.0:8000
```

OSNet downloads its pretrained weights (~5 MB) automatically on first startup if they are not already cached. This requires an internet connection once.

**Open the dashboard:**
```
http://localhost:8000
```

**API docs (Swagger UI):**
```
http://localhost:8000/api/docs
```

---

## Using TRACE

### Step 1 — Upload a camera video

1. Open **Pipeline Video Studio** in the left sidebar
2. Select camera **C01**, **C02**, or **C03**
3. Upload an MP4 / AVI / MOV file
4. Watch progress in the UI — the full pipeline runs automatically:
   - Detect people with YOLOv8n
   - Track them with BoT-SORT
   - Extract and quality-filter person crops
   - Generate OSNet body embeddings
   - Generate SCRFD + ArcFace face embeddings
   - Generate KPR part-based embeddings

Upload all cameras you have footage for before searching.

**Expected processing time per video:**

| Hardware | Time (35-second video, full pipeline) |
|---|---|
| CPU (Intel/AMD) | 15–30 min |
| Apple Silicon M1/M2 | 8–15 min |
| NVIDIA GPU (CUDA) | 2–5 min |

### Step 2 — Search for a person

1. Open **Re-ID Search Query** in the sidebar
2. Click the **Reference Photo Upload** tab
3. Upload a clear photo of the person you are looking for
4. Click **Execute Re-ID Search**
5. Results appear live — route, timestamps, match scores

### Step 3 — View the route

Results show:
- Which cameras the person appeared on
- Timestamps per camera
- Match confidence per sighting
- Reconstructed route with spatial + temporal scoring
- Body similarity, face similarity, and KPR part similarity per match

---

## How the pipeline works

```
Video uploaded
    │
    ▼
YOLOv8n  →  detects every person in every frame  (conf ≥ 0.6)
    │
    ▼
BoT-SORT  →  assigns stable Track IDs across frames
    │
    ▼
Crop extraction  →  5 crops per track, min 40×100 px, edge-truncation filter
    │
    ├─────────────────────┬──────────────────────────┐
    ▼                     ▼                          ▼
OSNet x1_0           SCRFD 10G                  KPR Swin-Small
512-dim body         face detection             8 part embeddings
embedding            + ArcFace R50              768-dim each
                     512-dim face               + visibility scores
                     embedding (L2-norm)        (ECCV 2024)

Query photo submitted
    │
    ▼
Same three models run on the query photo
    │
    ▼
Cosine similarity  →  query vs every gallery embedding per camera
    │
    ▼
Triple fusion:
  fused = (0.40×body + 0.60×KPR + 0.30×face) / sum_active_weights
  Missing signals auto-excluded → remaining signals re-normalised to 1.0
    │
    ▼
Threshold  →  fused ≥ 0.74 → CONFIDENT_MATCH
    │
    ▼
Route reconstruction  →  greedy walk over camera graph
  score = 0.60×appearance + 0.25×spatial + 0.15×temporal
    │
    ▼
Results pushed via WebSocket  →  displayed on dashboard
```

---

## Model weights included

All weights ship with the repository and are already configured in `.env`:

| Model | File | Size |
|---|---|---|
| SCRFD 10G (face detector) | `weights/det_10g.onnx` | 16 MB |
| ArcFace R50 (face ID) | `weights/w600k_r50.onnx` | 174 MB |
| KPR Swin-Small (part ReID) | `pretrained_models/kpr_occ_pt_IN_82.34_92.33_42323828.pth.tar` | 403 MB |
| SOLIDER Swin-Small (body ReID alt) | `pretrained_models/SOLIDER_REID_swin_small.pth` | 198 MB |
| OSNet x1_0 (body ReID) | auto-downloaded by torchreid on first run (~5 MB) | — |
| YOLOv8n (detection) | auto-downloaded by Ultralytics on first run (~6 MB) | — |

---

## Supported cameras

| ID | Location |
|---|---|
| C01 | Main Door |
| C02 | Main Corridor |
| C03 | Canteen / Common Area |

Camera adjacency and average transit times are defined in `dataset/camera_graph.json`.

---

## Project structure

```
trace/
├── ai_pipeline/
│   ├── detection/          YOLOv8 detection scripts
│   ├── tracking/           BoT-SORT tracking scripts
│   ├── reid/
│   │   ├── embed.py        OSNet body embedding pipeline
│   │   ├── face_embed.py   SCRFD + ArcFace face pipeline
│   │   ├── embed_kpr.py    KPR part-based embedding pipeline
│   │   ├── similarity.py   Cosine similarity
│   │   ├── track_aggregation.py
│   │   ├── matching.py
│   │   └── test_*.py       Unit tests
│   └── config.yaml         AI pipeline config (thresholds, model names)
├── backend/
│   └── app/
│       ├── main.py         FastAPI app entry point + startup
│       ├── config.py       Settings (reads .env, resolves paths)
│       ├── database.py     Async SQLAlchemy + SQLite
│       ├── models/
│       │   ├── orm.py      Database tables
│       │   └── schemas.py  Pydantic request/response models
│       ├── routers/
│       │   ├── upload.py   POST /api/upload/video
│       │   ├── queries.py  POST /api/queries, GET /api/queries/{id}/route
│       │   ├── cameras.py
│       │   ├── persons.py
│       │   └── analytics.py
│       ├── services/
│       │   ├── pipeline_service.py  End-to-end query orchestrator
│       │   ├── matching_service.py  Triple-fusion cross-camera matching
│       │   ├── route_service.py     Spatial-temporal route reconstruction
│       │   ├── embedding_service.py OSNet in-process service
│       │   ├── tracker_service.py   BoT-SORT subprocess wrapper
│       │   └── crop_service.py      Crop extraction subprocess wrapper
│       └── requirements.txt
├── frontend/
│   ├── index.html
│   └── src/
│       ├── app.js          Full dashboard (vanilla JS, no framework)
│       └── styles.css
├── dataset/
│   ├── camera_graph.json   Camera topology + transit times
│   ├── embeddings_*.json   Body galleries (generated on upload)
│   ├── face_embeddings_*.json  Face galleries (generated on upload)
│   ├── kpr_embeddings_*.json   KPR galleries (generated on upload)
│   ├── crops/              Extracted person crops
│   └── raw_videos/         Uploaded video files
├── weights/                SCRFD + ArcFace ONNX weights
├── pretrained_models/      KPR + SOLIDER weights
├── keypoint_promptable_reidentification/  KPR source repo
├── db/
│   └── trace.db            SQLite database (auto-created on startup)
├── .env                    Your local config
├── .env.example            Config template
└── run.py                  Server entry point
```

---

## Environment variables (.env)

The `.env.example` contains all defaults. After copying to `.env`, no edits are needed — all model paths point to weight files that are already present in the repo.

Key variables for reference:

```env
# Face pipeline
FACE_DET_MODEL=weights/det_10g.onnx
FACE_REC_MODEL=weights/w600k_r50.onnx
FACE_DET_THRESHOLD=0.5

# KPR
KPR_WEIGHTS_PATH=pretrained_models/kpr_occ_pt_IN_82.34_92.33_42323828.pth.tar
KPR_CONFIG_PATH=keypoint_promptable_reidentification/configs/kpr/imagenet/kpr_occ_posetrack_test.yaml
KPR_VIS_THRESHOLD=0.30

# Fusion weights
FUSION_BODY_WEIGHT=0.40
FUSION_FACE_WEIGHT=0.30
FUSION_KPR_WEIGHT=0.60

# Thresholds
NO_MATCH_THRESHOLD=0.74
DETECTION_CONFIDENCE=0.6
```

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/upload/video` | Upload video, trigger full pipeline |
| GET | `/api/upload/jobs` | List all upload jobs |
| GET | `/api/upload/status/{job_id}` | Poll upload job status |
| POST | `/api/queries` | Submit a Re-ID search query |
| GET | `/api/queries/{id}` | Poll query session status |
| GET | `/api/queries/{id}/route` | Get completed route and candidates |
| GET | `/api/cameras` | List cameras |
| GET | `/api/persons` | List registered persons |
| POST | `/api/persons` | Register a new person |
| POST | `/api/persons/{id}/enroll` | Enroll reference photo |
| GET | `/api/analytics/summary` | Dashboard stats |
| GET | `/api/health` | Health check |
| WS | `/ws` | WebSocket — live progress + results |

Full interactive docs: `http://localhost:8000/api/docs`

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'backend'`**
You are not in the project root. Always run from the `trace/` directory:
```bash
cd trace
python run.py
```

**`OSNet model failed to load` on startup**
torchreid downloads OSNet weights on first use (~5 MB). Requires internet on first run. If you are behind a proxy, set `HTTP_PROXY` and `HTTPS_PROXY` environment variables.

**`Tracking failed` / PyTorch 2.6 weights_only error**
Upgrade Ultralytics:
```bash
pip install "ultralytics>=8.3.0"
```

**`torchreid` not found**
```bash
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

**KPR step silently skipped**
Check that `keypoint_promptable_reidentification/` exists and its `setup.py develop` was run. If the KPR repo was not set up, the pipeline logs a warning and continues with body + face only.

**OpenCV DLL / libGL error**
```bash
pip uninstall opencv-python -y
pip install opencv-python-headless==4.9.0.80
```

**macOS — `zsh: command not found: python`**
```bash
brew install python@3.11
python3.11 -m venv venv
source venv/bin/activate
```

**Database schema errors after a `git pull`**
Delete the database and restart — it is recreated automatically:
```bash
# Windows
del db\trace.db

# macOS / Linux
rm db/trace.db
```

**Video upload says `Invalid camera_id`**
Use only `C01`, `C02`, or `C03`.

---

## Tech stack

| Layer | Technology |
|---|---|
| Detection | YOLOv8n (Ultralytics) |
| Tracking | BoT-SORT |
| Body ReID | OSNet x1_0 (torchreid) |
| Face detection | SCRFD 10G (ONNX Runtime) |
| Face ReID | ArcFace ResNet-50 (ONNX Runtime) |
| Part-based ReID | KPR Swin-Small (ECCV 2024) |
| Backend | FastAPI + Uvicorn |
| Database | SQLite + SQLAlchemy 2.0 async |
| Validation | Pydantic v2 |
| Real-time | WebSocket |
| Deep learning | PyTorch 2.x |
| Frontend | Vanilla JavaScript |

---

## License

For research and educational use.
