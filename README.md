# Trace — Multi-Camera Person Re-Identification System

Trace is an AI-powered surveillance system that tracks a specific person across multiple cameras and reconstructs their route with timestamps. Give it a reference photo, get back: which cameras they appeared on, in what order, and at what time.

---

## What It Does

- Upload CCTV footage from up to 5 cameras
- Submit a reference photo of any person
- System automatically finds that person across all cameras
- Returns a timestamped route: `Main Door 10:42:15 → Corridor 10:43:35 → Canteen 10:45:00`
- Live dashboard with real-time progress, route map, timeline, and watchlist alerts

---

## AI Models Used

| Model | Purpose | Dim |
|---|---|---|
| YOLOv8n | Detect people in video frames | — |
| ByteTrack | Track person IDs within a camera | — |
| OSNet x1_0 | Body appearance embedding | 512 |
| SOLIDER Swin-Small | Cloth-change robust body embedding | 768 |
| SCRFD 10G | Face detection in crops | — |
| ArcFace ResNet-50 | Face identity embedding | 512 |
| KPR Swin-Small | Part-based body embedding (ECCV 2024) | 768 |

---

## Project Structure

```
Trace/
├── ai_pipeline/
│   ├── reid/
│   │   ├── embed.py              # OSNet body embeddings
│   │   ├── embed_solider.py      # SOLIDER body embeddings
│   │   ├── face_embed.py         # SCRFD + ArcFace face embeddings
│   │   ├── embed_kpr.py          # KPR part-based embeddings
│   │   ├── similarity.py         # Cosine similarity util
│   │   ├── face_similarity.py    # Face-specific similarity
│   │   ├── kpr_similarity.py     # Part-aware similarity
│   │   ├── track_aggregation.py  # Per-track stats aggregation
│   │   └── confidence_scaling.py # Similarity → confidence %
│   ├── detection/                # YOLOv8 detection
│   ├── tracking/                 # ByteTrack tracking
│   ├── config.yaml               # AI pipeline config
│   └── requirements.txt          # AI pipeline deps
├── backend/
│   └── app/
│       ├── main.py               # FastAPI app entry point
│       ├── config.py             # Settings (reads .env)
│       ├── database.py           # SQLAlchemy async setup
│       ├── models/
│       │   ├── orm.py            # SQLAlchemy ORM models
│       │   └── schemas.py        # Pydantic schemas
│       ├── routers/
│       │   ├── upload.py         # Video upload + pipeline trigger
│       │   ├── queries.py        # Re-ID query submission + results
│       │   ├── cameras.py        # Camera registry
│       │   ├── persons.py        # Watchlist management
│       │   └── analytics.py      # Dashboard stats
│       └── services/
│           ├── pipeline_service.py   # End-to-end query orchestrator
│           ├── matching_service.py   # Triple fusion matching
│           ├── route_service.py      # Spatial-temporal route reconstruction
│           ├── embedding_service.py  # OSNet inference service
│           ├── tracker_service.py    # ByteTrack wrapper
│           ├── crop_service.py       # Crop extraction
│           └── detection_service.py  # YOLOv8 wrapper
├── frontend/
│   ├── index.html                # Single-page dashboard
│   └── src/
│       ├── app.js                # All dashboard JS (vanilla, no framework)
│       ├── api.js                # API client
│       └── styles.css            # HUD dark theme
├── dataset/
│   ├── camera_graph.json         # Camera topology + transit times
│   ├── crops/                    # Extracted person crops per camera
│   ├── embeddings_C01.json       # OSNet gallery (generated)
│   ├── face_embeddings_C01.json  # Face gallery (generated, optional)
│   ├── kpr_embeddings_C01.json   # KPR gallery (generated, optional)
│   └── raw_videos/               # Uploaded videos
├── db/
│   └── trace.db                  # SQLite database (auto-created)
├── weights/                      # ONNX model weights (download separately)
├── pretrained_models/            # KPR + SOLIDER weights (download separately)
├── .env                          # Local config (not committed)
├── .env.example                  # Template for .env
└── backend/requirements.txt      # All Python dependencies
```

---

## Prerequisites

- Python 3.11 — https://www.python.org/downloads/release/python-3110/
- pip (comes with Python)
- Git — https://git-scm.com/downloads
- 4GB+ RAM (8GB recommended for KPR)
- GPU optional but significantly faster for KPR (use Google Colab for KPR on CPU machines)

> **Windows users:** Make sure Python is added to PATH during installation. Check "Add Python to PATH" in the installer.

> **macOS users:** If `python3.11` is not available, install via `brew install python@3.11`

---

## Quick Start (Body-Only Mode — No Optional Downloads)

This gets the full system running with OSNet body matching only. No face weights, no KPR weights needed.

### 1. Clone the repo

```bash
git clone https://github.com/anmolgoyal2006/trace.git
cd trace
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r backend/requirements.txt
```

> This installs FastAPI, SQLAlchemy, PyTorch, Ultralytics (YOLOv8), torchreid (OSNet), OpenCV, and all other backend + AI deps in one shot.

> ⚠️ **PyTorch is ~800MB.** This step can take 5–30 minutes depending on your internet speed. This is normal.

> ⚠️ **macOS (Apple Silicon M1/M2/M3):** PyTorch runs on CPU by default. It works correctly but has no CUDA GPU support (Apple uses MPS, not CUDA). All models will run on CPU. KPR will be slow — use Google Colab for KPR embedding generation.
> If you see a `torch` version conflict, install PyTorch for Apple Silicon first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install -r backend/requirements.txt
> ```

> ⚠️ **macOS — `libGL` error with OpenCV:** If you see `ImportError: dlopen ... libGL`, install the headless variant:
> ```bash
> pip uninstall opencv-python
> pip install opencv-python-headless==4.9.0.80
> ```

> ⚠️ **torchreid install issue?** If `torchreid` fails to install from PyPI, install it directly from source:
> ```bash
> pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
> ```
> Then re-run `pip install -r backend/requirements.txt`

> ⚠️ **Windows + OpenCV error?** If you see a DLL error with `opencv-python`, install the headless version instead:
> ```bash
> pip uninstall opencv-python
> pip install opencv-python-headless==4.9.0.80
> ```

### 4. Set up environment config

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` — the defaults work for body-only mode. No changes needed for basic operation.

### 5. Run the server

```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

> ⚠️ **Must be run from the project root** (`Trace/` directory), not from inside `backend/`. The import paths assume the project root.

> On first startup, OSNet weights (~5MB) are downloaded automatically from the internet. You'll see:
> `Successfully loaded imagenet pretrained weights from "...osnet_x1_0_imagenet.pth"`
> This is normal — happens once only.

### 6. Open the dashboard

```
http://localhost:8000
```

API docs available at:
```
http://localhost:8000/api/docs
```

---

## Using the System

### Step 1 — Upload camera footage

1. Go to **Pipeline Video Studio** in the left sidebar
2. Select a camera (C01, C02, or C03)
3. Upload an MP4/AVI/MOV video file
4. Wait for processing to complete (progress shown in the UI)
   - On CPU: ~6–10 min for a 35-second video (body only)
   - On GPU: ~1–2 min

### Step 2 — Register or search for a person

**Option A — Search with a photo directly:**
1. Go to **Re-ID Search Query**
2. Click "Reference Photo Upload" tab
3. Upload any photo of the person you want to find
4. Click "Execute Re-ID Search"

**Option B — Register a person first (watchlist):**
1. Go to **Watchlist & Persons**
2. Click "Register Target Profile"
3. Fill in name, set watchlist status (suspect/poi/missing)
4. Upload a reference photo to enroll their embedding
5. Then search using "Registered Person" tab in Re-ID Search Query

### Step 3 — View results

Results appear in real-time:
- **Route Reconstruction** — animated map showing which cameras they appeared on and when
- **Sighting Timeline** — horizontal timeline view per camera
- **Dashboard** — confidence scores, match thumbnails, alerts

---

## Activating Face Embedding (Optional — Improves Accuracy)

Face matching adds a second signal alongside body matching. Useful when people change clothing between cameras.

### Download ONNX weights

```bash
# Create weights directory
mkdir weights

# Download SCRFD face detector (16 MB)
curl -L -o weights/det_10g.onnx https://huggingface.co/yakhyo/scrfd/resolve/main/det_10g.onnx

# Download ArcFace recogniser (166 MB)
curl -L -o weights/w600k_r50.onnx https://huggingface.co/yakhyo/arcface/resolve/main/w600k_r50.onnx
```

Or download manually from:
- SCRFD: https://huggingface.co/yakhyo/scrfd/resolve/main/det_10g.onnx
- ArcFace: https://huggingface.co/yakhyo/arcface/resolve/main/w600k_r50.onnx

### Install ONNX Runtime

```bash
pip install onnxruntime        # CPU
# OR
pip install onnxruntime-gpu    # GPU (CUDA 11+)
```

### Enable in .env

```env
FACE_DET_MODEL=weights/det_10g.onnx
FACE_REC_MODEL=weights/w600k_r50.onnx
```

Restart the server. Face embedding now runs automatically on every new video upload and every query photo.

---

## Activating KPR Part-Based Matching (Optional — Best Accuracy, Needs GPU)

KPR (Keypoint Promptable Re-Identification) handles partial bodies — when cameras only show waist-up, or legs are cut off. It compares only the body parts that are visible in both images.

> ⚠️ KPR is very slow on CPU (8–15 min per video). Strongly recommended to run on Google Colab (free T4 GPU) and copy the output `kpr_embeddings_C01.json` file back to your `dataset/` folder.

### Step 1 — Clone the KPR repo

```bash
git clone https://github.com/VlSomers/keypoint_promptable_reidentification
```

### Step 2 — Install KPR dependencies

```bash
cd keypoint_promptable_reidentification
pip install -r requirements.txt
python setup.py develop
cd ..
```

> ⚠️ **macOS:** If `python setup.py develop` fails, try:
> ```bash
> pip install -e .
> ```

### Step 3 — Download KPR pretrained weights

Download from Google Drive:
```
https://drive.google.com/file/d/1Np5wu3nQa_Fl_z7Zw2kchJNC8JZVwsh5/view
```

Save as:
```
pretrained_models/kpr_occ_pt_IN_82.34_92.33_42323828.pth.tar
```

### Step 4 — Enable in .env

```env
KPR_WEIGHTS_PATH=pretrained_models/kpr_occ_pt_IN_82.34_92.33_42323828.pth.tar
KPR_CONFIG_PATH=keypoint_promptable_reidentification/configs/kpr/market1501/kpr_swin_small.yaml
```

Restart the server. KPR now runs automatically on video upload.

### Running KPR on Google Colab (recommended for CPU machines)

If you're on CPU, run KPR separately on Colab:

```python
# In Google Colab:
!git clone https://github.com/anmolgoyal2006/trace.git
!git clone https://github.com/VlSomers/keypoint_promptable_reidentification
!cd keypoint_promptable_reidentification && pip install -r requirements.txt && python setup.py develop

# Upload your crops and metadata, then run:
!python trace/ai_pipeline/reid/embed_kpr.py \
    --metadata trace/dataset/crops_metadata_C01.json \
    --crop-dir trace/dataset/crops/C01 \
    --output trace/dataset/kpr_embeddings_C01.json \
    --kpr-weights pretrained_models/kpr_occ_pt_IN_82.34_92.33_42323828.pth.tar \
    --kpr-config keypoint_promptable_reidentification/configs/kpr/market1501/kpr_swin_small.yaml
```

Then download `kpr_embeddings_C01.json` and put it in your local `dataset/` folder. The query pipeline will automatically use it on next search.

---

## Activating SOLIDER Body Embedding (Optional — Cloth-Change Robust)

SOLIDER uses a Swin-Small transformer trained on human-centric semantics. Better than OSNet when people change clothes between cameras.

### Download weights

From: https://github.com/tinyvision/SOLIDER-REID/releases

Save as: `pretrained_models/SOLIDER_REID_swin_small.pth`

### Clone SOLIDER-REID repo

```bash
git clone https://github.com/tinyvision/SOLIDER-REID
```

### Install dependencies

```bash
pip install timm
```

### Run SOLIDER embedding (replaces OSNet for the gallery)

```bash
python ai_pipeline/reid/embed_solider.py \
    --metadata dataset/crops_metadata_C01.json \
    --crop-dir dataset/crops/C01 \
    --output dataset/embeddings_C01.json \
    --weights pretrained_models/SOLIDER_REID_swin_small.pth \
    --solider-config SOLIDER-REID/configs/market/swin_small.yml \
    --overwrite
```

Update `.env`:
```env
SOLIDER_WEIGHTS=pretrained_models/SOLIDER_REID_swin_small.pth
SOLIDER_CONFIG_PATH=SOLIDER-REID/configs/market/swin_small.yml
SOLIDER_EMBEDDING_DIM=768
```

---

## Environment Variables Reference (.env)

```env
# Server
HOST=0.0.0.0
PORT=8000
DEBUG=true

# Face embedding (optional)
FACE_DET_MODEL=weights/det_10g.onnx
FACE_REC_MODEL=weights/w600k_r50.onnx
FACE_DET_THRESHOLD=0.5

# KPR part-based matching (optional)
KPR_WEIGHTS_PATH=pretrained_models/kpr_occ_pt_IN_82.34_92.33_42323828.pth.tar
KPR_CONFIG_PATH=keypoint_promptable_reidentification/configs/kpr/market1501/kpr_swin_small.yaml
KPR_VIS_THRESHOLD=0.30

# SOLIDER body embedding (optional)
SOLIDER_WEIGHTS=pretrained_models/SOLIDER_REID_swin_small.pth
SOLIDER_CONFIG_PATH=SOLIDER-REID/configs/market/swin_small.yml
SOLIDER_EMBEDDING_DIM=768

# Matching thresholds
NO_MATCH_THRESHOLD=0.74

# Fusion weights
FUSION_BODY_WEIGHT=0.40
FUSION_FACE_WEIGHT=0.30
FUSION_KPR_WEIGHT=0.60

# YOLOv8
YOLO_MODEL=yolov8n.pt
DETECTION_CONFIDENCE=0.6
```

---

## Supported Cameras

The system supports 5 cameras by default (C01–C05). MVP cameras (C01, C02, C03) are active by default.

| ID | Location | Status |
|---|---|---|
| C01 | Main Door | MVP (active) |
| C02 | Main Corridor | MVP (active) |
| C03 | Canteen / Common Area | MVP (active) |
| C04 | Stairwell / 2nd Floor | Stretch (disabled) |
| C05 | Outdoor Courtyard | Stretch (disabled) |

Camera connections and transit times are defined in `dataset/camera_graph.json`.

---

## Processing Times (35-second video)

| Mode | CPU (Intel/AMD) | Apple Silicon (M1/M2) | GPU (NVIDIA CUDA) |
|---|---|---|---|
| Body only (OSNet) | 6–11 min | 3–5 min | 1–2 min |
| Body + Face | 7–13 min | 4–7 min | 1.5–2.5 min |
| Body + Face + KPR | 14–26 min | 8–14 min | 2–4 min |

> KPR on CPU is slow regardless of platform. Use Google Colab (free T4 GPU) for KPR embedding generation if you don't have an NVIDIA GPU.
> Apple Silicon has no CUDA support — MPS (Metal) is not used by this project. CPU times apply.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/upload/video` | Upload video, trigger pipeline |
| GET | `/api/upload/jobs` | List all upload jobs + status |
| GET | `/api/upload/status/{job_id}` | Poll single job status |
| POST | `/api/queries` | Submit a re-ID search query |
| GET | `/api/queries/{id}` | Poll query session status |
| GET | `/api/queries/{id}/route` | Get completed route result |
| GET | `/api/cameras` | List all cameras |
| PATCH | `/api/cameras/{id}` | Update camera location/config |
| GET | `/api/persons` | List registered persons |
| POST | `/api/persons` | Register a new person |
| POST | `/api/persons/{id}/enroll` | Enroll reference photo |
| GET | `/api/analytics/dashboard` | Dashboard stats |
| GET | `/api/health` | Health check |
| WS | `/ws` | WebSocket for live events |

Full interactive docs: `http://localhost:8000/api/docs`

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'backend'`**
You're running uvicorn from the wrong directory. Always run from the project root:
```bash
cd Trace   # make sure you're here
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

**macOS — `zsh: command not found: python`**
macOS ships with Python 2 or no Python. Use `python3` or install Python 3.11:
```bash
brew install python@3.11
python3.11 -m venv venv
source venv/bin/activate
```
After activating the venv, `python` and `pip` will point to 3.11 correctly.

**macOS — `OSError: dlopen libgomp` or OpenMP error with torchreid**
Install OpenMP via Homebrew:
```bash
brew install libomp
```
Then retry the pip install.

**macOS — `SSL: CERTIFICATE_VERIFY_FAILED` when downloading OSNet weights**
Run the Python certificate installer:
```bash
/Applications/Python\ 3.11/Install\ Certificates.command
```
Or:
```bash
pip install certifi
python -c "import ssl; ssl.create_default_context()"
```

**`torchreid` not found**
```bash
pip install torchreid
# or
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

**`OSNet model failed to load` on startup**
OSNet weights are downloaded automatically by torchreid on first use. Requires internet connection on first run. If behind a proxy, set `HTTP_PROXY` environment variable.

**Face embedding skipped silently**
Check that `weights/det_10g.onnx` and `weights/w600k_r50.onnx` exist at the paths in `.env`. The pipeline logs exactly which weights are missing.

**KPR import error**
The KPR repo must be cloned and `python setup.py develop` must have been run inside it. The `torchreid` inside KPR is a fork — it conflicts with the standard torchreid. Use separate virtual environments if running both.

**Database schema errors after pulling updates**
If new columns were added to ORM models, delete the database and restart:
```bash
del db\trace.db    # Windows
rm db/trace.db     # macOS/Linux
```
The database is recreated automatically on startup.

**Video upload fails with `Invalid camera_id`**
Only C01, C02, C03 are valid for upload. C04 and C05 are stretch cameras not yet active.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Person detection | YOLOv8n (Ultralytics) |
| Tracking | ByteTrack |
| Body ReID | OSNet x1_0 (torchreid) |
| Semantic body ReID | SOLIDER Swin-Small |
| Face detection | SCRFD 10G (ONNX Runtime) |
| Face recognition | ArcFace ResNet-50 (ONNX Runtime) |
| Part-based ReID | KPR Swin-Small (ECCV 2024) |
| Backend framework | FastAPI |
| ORM | SQLAlchemy 2.0 async |
| Database | SQLite + aiosqlite |
| Data validation | Pydantic v2 |
| Real-time | WebSocket |
| Deep learning | PyTorch 2.x |
| ONNX inference | ONNX Runtime |
| Frontend | Vanilla JavaScript (no framework) |
| Charts | Chart.js |
| Map | SVG floor-plan |
| Server | Uvicorn |

---

## License

This project is for research and educational use.
