"""
cameras.py — /api/cameras
Read-only camera registry + graph info.
Cameras are seeded from camera_graph.json at startup — not created via API.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.database import get_db
from backend.app.models.orm import Camera, Sighting
from backend.app.models.schemas import CameraGraphEdge, CameraOut, CameraWithGraph
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
