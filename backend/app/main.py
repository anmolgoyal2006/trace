"""
main.py — FastAPI application entry point for the unified Trace system.

Startup sequence:
  1. Create all DB tables (idempotent).
  2. Seed cameras from camera_graph.json.
  3. Load OSNet model into EmbeddingService.
  4. Initialise PipelineService with the WebSocket manager.
  5. Mount frontend static files.
  6. Register all routers.
  7. Register WebSocket endpoint.
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from backend.app.config import settings
from backend.app.core.websocket_manager import ws_manager
from backend.app.database import async_session, engine, Base
from backend.app.models.orm import Camera  # noqa: F401 — needed for metadata
import backend.app.models.orm  # noqa: F401 — register all ORM models with Base
from backend.app.routers import analytics, cameras, persons, queries, upload
from backend.app.services.embedding_service import embedding_service
from backend.app.services.pipeline_service import PipelineService


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  Trace — Unified Surveillance Intelligence")
    logger.info("=" * 60)

    # 1. Create DB schema
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("[startup] Database schema ready")

    # 2. Seed cameras from camera_graph.json
    await _seed_cameras()
    logger.info("[startup] Cameras seeded")

    # 3. Load OSNet model
    try:
        embedding_service.load_model()
        logger.info("[startup] OSNet model loaded")
    except Exception as e:
        logger.warning(f"[startup] OSNet model failed to load: {e}")
        logger.warning("[startup] Embedding/search endpoints will return 503 until model is available")

    # 4. Init PipelineService and inject into queries router
    pipeline_svc = PipelineService(ws_manager)
    queries.set_pipeline_service(pipeline_svc)
    logger.info("[startup] PipelineService initialised")

    # 5. Ensure data directories exist
    for d in [
        settings.uploads_dir,
        settings.snapshots_dir,
        settings.raw_videos_dir,
        settings.uploads_dir / "reference_photos",
        settings.uploads_dir / "query_images",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f"[startup] Dashboard → http://{settings.host}:{settings.port}")
    logger.info(f"[startup] API docs  → http://{settings.host}:{settings.port}/api/docs")
    logger.info(f"[startup] WebSocket → ws://{settings.host}:{settings.port}/ws")
    logger.info("=" * 60)

    yield

    # ── Shutdown ───────────────────────────────────────────────────────
    await engine.dispose()
    logger.info("[shutdown] Database connections closed")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Trace — Unified Surveillance Intelligence",
    description=(
        "Multi-camera person tracking and re-identification system. "
        "Body re-ID (OSNet), spatial+temporal route reconstruction, "
        "watchlist alerts, and live dashboard."
    ),
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# ── CORS (dev: allow all origins; tighten for production) ─────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API routers ────────────────────────────────────────────────────────────
app.include_router(persons.router)
app.include_router(cameras.router)
app.include_router(queries.router)
app.include_router(analytics.router)
app.include_router(upload.router)

# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive; clients can send pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# ---------------------------------------------------------------------------
# Health check  (must be registered BEFORE the catch-all static mount)
# ---------------------------------------------------------------------------

@app.get("/api/health", tags=["system"])
async def health():
    return {
        "status": "ok",
        "model_loaded": embedding_service.is_ready,
        "ws_connections": ws_manager.connection_count,
    }


# ── Serve crop images (dataset/crops) ─────────────────────────────────────
crops_dir = settings.data_dir / "crops"
crops_dir.mkdir(parents=True, exist_ok=True)
app.mount("/crops", StaticFiles(directory=str(crops_dir)), name="crops")

# ── Serve uploaded reference photos ───────────────────────────────────────
uploads_dir = settings.uploads_dir
uploads_dir.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")

# ── Frontend static assets (JS, CSS) — mounted BEFORE the catch-all ──────
_frontend_dir = settings.frontend_dir
if _frontend_dir.exists():
    # Serve /src/* assets (styles.css, app.js, api.js)
    _src_dir = _frontend_dir / "src"
    if _src_dir.exists():
        app.mount("/src", StaticFiles(directory=str(_src_dir)), name="frontend-src")

    # SPA catch-all: serve index.html for any non-API, non-asset path
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        index = _frontend_dir / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return {"detail": "Frontend not found"}

else:
    logger.warning(f"[startup] Frontend directory not found: {_frontend_dir}")


# ---------------------------------------------------------------------------
# Camera seeding
# ---------------------------------------------------------------------------

async def _seed_cameras() -> None:
    """
    Upsert cameras from camera_graph.json into the database.
    Only seeds cameras where mvp=true.
    Existing cameras are updated, new ones inserted.
    """
    from sqlalchemy import select

    if not settings.camera_graph_path.exists():
        logger.warning(f"[seed] camera_graph.json not found: {settings.camera_graph_path}")
        return

    with open(settings.camera_graph_path) as f:
        graph = json.load(f)

    import yaml
    with open(settings.ai_config_path) as f:
        ai_cfg = yaml.safe_load(f)
    camera_times = ai_cfg.get("cameras", {})

    async with async_session() as db:
        cameras_data = graph.get("cameras", {})
        for cam_id, cam_info in cameras_data.items():
            if not cam_info.get("mvp", False):
                continue

            existing = await db.get(Camera, cam_id)
            start_time = camera_times.get(cam_id, {}).get("start_time")

            if existing is None:
                cam = Camera(
                    id=cam_id,
                    location=cam_info.get("location", cam_id),
                    description=cam_info.get("description"),
                    start_time=start_time,
                    is_active=True,
                )
                db.add(cam)
                logger.info(f"[seed] Inserted camera {cam_id}: {cam_info.get('location')}")
            else:
                existing.location = cam_info.get("location", cam_id)
                existing.description = cam_info.get("description")
                if start_time:
                    existing.start_time = start_time
                logger.debug(f"[seed] Camera {cam_id} already exists — updated")

        await db.commit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )
