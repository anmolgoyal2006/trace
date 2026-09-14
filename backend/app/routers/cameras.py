"""
cameras.py — /api/cameras
Camera registry + graph info.
Cameras are seeded from camera_graph.json at startup.
PATCH /{id} lets the user update location, start_time, and transit times
without editing any config files — the changes are persisted to the DB
and written back to camera_graph.json so the route service picks them up.
"""

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import settings
from backend.app.database import get_db
from backend.app.models.orm import Camera, Sighting
from backend.app.models.schemas import CameraGraphEdge, CameraOut, CameraUpdate, CameraWithGraph
from backend.app.services.route_service import RouteService

router = APIRouter(prefix="/api/cameras", tags=["cameras"])

# Lazy-loaded route service (reuses the same instance)
_route_svc: RouteService | None = None


def _get_route_svc() -> RouteService:
    global _route_svc
    if _route_svc is None:
        _route_svc = RouteService()
    return _route_svc


@router.get("", response_model=list[CameraWithGraph])
async def list_cameras(db: AsyncSession = Depends(get_db)) -> list[CameraWithGraph]:
    """List all active cameras with their adjacency graph edges."""
    result = await db.execute(select(Camera).where(Camera.is_active == True))
    cameras = result.scalars().all()

    graph = _get_route_svc().get_graph()
    cam_graph_data = graph.get("cameras", {})

    out = []
    for cam in cameras:
        edges_raw = cam_graph_data.get(cam.id, {}).get("connects_to", {})
        edges = [
            CameraGraphEdge(
                to_camera_id=neighbour,
                avg_transit_sec=edge_data.get("avg_transit_sec", 0),
                notes=edge_data.get("notes"),
            )
            for neighbour, edge_data in edges_raw.items()
        ]
        out.append(CameraWithGraph(
            id=cam.id,
            location=cam.location,
            description=cam.description,
            start_time=cam.start_time,
            is_active=cam.is_active,
            connects_to=edges,
        ))
    return out


@router.get("/{camera_id}", response_model=CameraWithGraph)
async def get_camera(
    camera_id: str,
    db: AsyncSession = Depends(get_db),
) -> CameraWithGraph:
    cam = await db.get(Camera, camera_id)
    if not cam or not cam.is_active:
        raise HTTPException(status_code=404, detail="Camera not found")

    graph = _get_route_svc().get_graph()
    edges_raw = graph.get("cameras", {}).get(cam.id, {}).get("connects_to", {})
    edges = [
        CameraGraphEdge(
            to_camera_id=neighbour,
            avg_transit_sec=edge_data.get("avg_transit_sec", 0),
            notes=edge_data.get("notes"),
        )
        for neighbour, edge_data in edges_raw.items()
    ]
    return CameraWithGraph(
        id=cam.id,
        location=cam.location,
        description=cam.description,
        start_time=cam.start_time,
        is_active=cam.is_active,
        connects_to=edges,
    )


@router.get("/{camera_id}/sightings")
async def get_camera_sightings(
    camera_id: str,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """Recent sightings for a specific camera across all sessions."""
    cam = await db.get(Camera, camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    result = await db.execute(
        select(Sighting)
        .where(Sighting.camera_id == camera_id)
        .order_by(Sighting.id.desc())
        .limit(limit)
    )
    sightings = result.scalars().all()
    return [
        {
            "id": s.id,
            "session_id": s.session_id,
            "track_id": s.track_id,
            "first_seen": s.first_seen,
            "best_confidence": s.best_confidence,
            "fusion_score": s.fusion_score,
            "crop_path": s.crop_path,
        }
        for s in sightings
    ]


@router.patch("/{camera_id}", response_model=CameraWithGraph)
async def update_camera(
    camera_id: str,
    body: CameraUpdate,
    db: AsyncSession = Depends(get_db),
) -> CameraWithGraph:
    """
    Update camera location, start_time, and/or transit times.

    Changes are written to:
      1. The SQLite cameras table (location, start_time).
      2. camera_graph.json (location, transit times) so the route service
         uses the new values immediately without a server restart.
    """
    cam = await db.get(Camera, camera_id)
    if not cam or not cam.is_active:
        raise HTTPException(status_code=404, detail="Camera not found")

    # ── 1. Update DB columns ───────────────────────────────────────────
    if body.location is not None:
        cam.location = body.location.strip()
    if body.start_time is not None:
        # Normalise to HH:MM:SS
        t = body.start_time.strip()
        cam.start_time = t if len(t) == 8 else t + ":00"

    await db.commit()
    await db.refresh(cam)

    # ── 2. Patch camera_graph.json ─────────────────────────────────────
    graph_path = settings.camera_graph_path
    if graph_path.exists():
        with open(graph_path) as f:
            graph = json.load(f)

        cam_data = graph.get("cameras", {}).get(camera_id, {})

        if body.location is not None:
            cam_data["location"] = cam.location

        if body.transit_updates:
            connects_to = cam_data.get("connects_to", {})
            for neighbour, secs in body.transit_updates.items():
                if neighbour in connects_to:
                    connects_to[neighbour]["avg_transit_sec"] = int(secs)
            cam_data["connects_to"] = connects_to

        graph["cameras"][camera_id] = cam_data

        with open(graph_path, "w") as f:
            json.dump(graph, f, indent=2)

        # Reset the cached route service so it re-reads the updated graph
        global _route_svc
        _route_svc = None

    # ── 3. Return updated camera with graph edges ──────────────────────
    route_svc = _get_route_svc()
    updated_graph = route_svc.get_graph()
    edges_raw = updated_graph.get("cameras", {}).get(cam.id, {}).get("connects_to", {})
    edges = [
        CameraGraphEdge(
            to_camera_id=neighbour,
            avg_transit_sec=edge_data.get("avg_transit_sec", 0),
            notes=edge_data.get("notes"),
        )
        for neighbour, edge_data in edges_raw.items()
    ]
    return CameraWithGraph(
        id=cam.id,
        location=cam.location,
        description=cam.description,
        start_time=cam.start_time,
        is_active=cam.is_active,
        connects_to=edges,
    )
