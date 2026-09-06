"""
analytics.py — /api/analytics
Dashboard stats, alert management, camera activity.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.database import get_db
from backend.app.models.orm import Alert, Camera, Person, QuerySession, Sighting
from backend.app.models.schemas import (
    AlertOut,
    AnalyticsDashboard,
    CameraActivityItem,
    DashboardStats,
)

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@router.get("/dashboard", response_model=AnalyticsDashboard)
async def dashboard(db: AsyncSession = Depends(get_db)) -> AnalyticsDashboard:
    """Aggregated stats for the dashboard header cards."""

    # Counts
    total_persons = (
        await db.scalar(select(func.count()).where(Person.is_active == True))
    ) or 0

    watchlist_persons = (
        await db.scalar(
            select(func.count())
            .select_from(Person)
            .where(Person.is_active == True, Person.watchlist_status != "none")
        )
    ) or 0

    total_queries = (await db.scalar(select(func.count()).select_from(QuerySession))) or 0

    completed_queries = (
        await db.scalar(
            select(func.count())
            .select_from(QuerySession)
            .where(QuerySession.status == "done")
        )
    ) or 0

    total_sightings = (await db.scalar(select(func.count()).select_from(Sighting))) or 0

    total_alerts = (await db.scalar(select(func.count()).select_from(Alert))) or 0

    unacknowledged_alerts = (
        await db.scalar(
            select(func.count())
            .select_from(Alert)
            .where(Alert.acknowledged == False)
        )
    ) or 0

    cameras_active = (
        await db.scalar(select(func.count()).where(Camera.is_active == True))
    ) or 0

    stats = DashboardStats(
        total_persons=total_persons,
        watchlist_persons=watchlist_persons,
        total_queries=total_queries,
        completed_queries=completed_queries,
        total_sightings=total_sightings,
        total_alerts=total_alerts,
        unacknowledged_alerts=unacknowledged_alerts,
        cameras_active=cameras_active,
    )

    # Recent alerts (last 10 unacknowledged first)
    alerts_result = await db.execute(
        select(Alert)
        .order_by(Alert.acknowledged.asc(), Alert.created_at.desc())
        .limit(10)
    )
    recent_alerts = [AlertOut.model_validate(a) for a in alerts_result.scalars().all()]

    # Per-camera activity — two batch aggregates instead of 2×N round-trips
    cameras_result = await db.execute(select(Camera).where(Camera.is_active == True))
    cameras = cameras_result.scalars().all()

    # Batch 1: total sightings per camera
    sightings_agg_result = await db.execute(
        select(Sighting.camera_id, func.count().label("cnt"))
        .group_by(Sighting.camera_id)
    )
    sightings_by_cam: dict[str, int] = {row.camera_id: row.cnt for row in sightings_agg_result}

    # Batch 2: completed-query sightings per camera
    done_agg_result = await db.execute(
        select(Sighting.camera_id, func.count().label("cnt"))
        .join(QuerySession, Sighting.session_id == QuerySession.id)
        .where(QuerySession.status == "done")
        .group_by(Sighting.camera_id)
    )
    done_by_cam: dict[str, int] = {row.camera_id: row.cnt for row in done_agg_result}

    activity: list[CameraActivityItem] = [
        CameraActivityItem(
            camera_id=cam.id,
            location=cam.location,
            total_sightings=sightings_by_cam.get(cam.id, 0),
            recent_queries=done_by_cam.get(cam.id, 0),
        )
        for cam in cameras
    ]

    return AnalyticsDashboard(
        stats=stats,
        recent_alerts=recent_alerts,
        camera_activity=activity,
    )


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

@router.get("/alerts", response_model=list[AlertOut])
async def list_alerts(
    unacknowledged_only: bool = False,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
) -> list[AlertOut]:
    """List alerts, most recent first."""
    q = select(Alert).order_by(Alert.created_at.desc()).limit(limit)
    if unacknowledged_only:
        q = q.where(Alert.acknowledged == False)
    result = await db.execute(q)
    return [AlertOut.model_validate(a) for a in result.scalars().all()]


@router.post("/alerts/{alert_id}/acknowledge", response_model=AlertOut)
async def acknowledge_alert(
    alert_id: int,
    db: AsyncSession = Depends(get_db),
) -> AlertOut:
    alert = await db.get(Alert, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.acknowledged = True
    await db.commit()
    return AlertOut.model_validate(alert)


# ---------------------------------------------------------------------------
# Sightings timeline
# ---------------------------------------------------------------------------

@router.get("/sightings/recent", response_model=list[dict])
async def recent_sightings(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """Most recent sightings across all sessions for the timeline view."""
    result = await db.execute(
        select(Sighting)
        .order_by(Sighting.id.desc())
        .limit(limit)
    )
    sightings = result.scalars().all()
    return [
        {
            "id": s.id,
            "session_id": s.session_id,
            "camera_id": s.camera_id,
            "track_id": s.track_id,
            "first_seen": s.first_seen,
            "last_seen": s.last_seen,
            "best_confidence": s.best_confidence,
            "fusion_score": s.fusion_score,
            "crop_path": s.crop_path,
        }
        for s in sightings
    ]
