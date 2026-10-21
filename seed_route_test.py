"""
seed_route_test.py — Seeds a synthetic CONFIDENT_MATCH route session into the DB.

Route chain:  C01 Track 25 → C02 Track 37 → C03 Track 37
All three produce similarity scores above the 0.74 CONFIDENT_MATCH threshold
using the actual OSNet embeddings already in dataset/embeddings_*.json.

Run:
    python seed_route_test.py

Then open the TRACE dashboard, go to Route Reconstruction, and select
"Seed: C01→C02→C03 Route Test" from the session dropdown.

The query image is  dataset/crops/C01/C01_track25_frame0820.jpg
Upload that as the query image to trigger the live pipeline, or use this
script to inspect the pre-seeded result immediately.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Windows ProactorEventLoop required for async SQLAlchemy on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


async def seed():
    from backend.app.database import async_session, engine, Base
    from backend.app.models.orm import QuerySession, Sighting, RouteStep
    from sqlalchemy import text

    # Ensure tables exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Route data
    ROUTE = [
        {
            "camera_id":     "C01",
            "track_id":      25,
            "first_seen":    "10:02:27.35",
            "last_seen":     "10:02:31.79",
            "crop_path":     "dataset/crops/C01/C01_track25_frame0820.jpg",
            "appearance_score": 0.7698,
            "fusion_score":  0.7698,
            "best_confidence": 77.0,
            "match_status":  "CONFIDENT_MATCH",
        },
        {
            "camera_id":     "C02",
            "track_id":      37,
            "first_seen":    "10:05:33.39",
            "last_seen":     "10:05:35.73",
            "crop_path":     "dataset/crops/C02/C02_track37_frame1001.jpg",
            "appearance_score": 0.7698,
            "fusion_score":  0.7698,
            "best_confidence": 77.0,
            "match_status":  "CONFIDENT_MATCH",
        },
        {
            "camera_id":     "C03",
            "track_id":      37,
            "first_seen":    "10:08:44.00",
            "last_seen":     "10:08:50.00",
            "crop_path":     "dataset/crops/C03/C03_track37_frame0999.jpg",
            "appearance_score": 0.9339,
            "fusion_score":  0.9339,
            "best_confidence": 93.4,
            "match_status":  "CONFIDENT_MATCH",
        },
    ]

    route_meta = {
        "supported_transitions_count": 3,
        "ambiguous_transitions_count":  0,
        "disconnected_transitions_count": 0,
        "data_limitations": [],
        "route_method": "topology_aware",
        "steps_detail": [
            {
                "step_order": 0,
                "transition_status": "SUPPORTED",
                "hop_distance_m": None,
                "implied_speed_mps": None,
                "route_explanation": {
                    "step_type": "ANCHOR",
                    "status": "SUPPORTED",
                    "summary": (
                        "Camera C01 (Main Door) was selected as the route anchor. "
                        "Track 25 provides the initial observation with appearance "
                        "similarity 0.770 and match score 77.0%."
                    ),
                },
            },
            {
                "step_order": 1,
                "transition_status": "SUPPORTED",
                "hop_distance_m": None,
                "implied_speed_mps": None,
                "route_explanation": {
                    "step_type": "TRANSITION",
                    "status": "SUPPORTED",
                    "from_camera": "C01",
                    "to_camera":   "C02",
                    "summary": (
                        "Camera C02 was selected after C01 because the camera graph "
                        "contains a configured direct C01→C02 transition (expected "
                        "transit: 45s); timestamps are chronologically compatible "
                        "(observed gap: 186s); appearance match score 77.0% supports "
                        "identity. No GPS coordinates configured so geodesic speed "
                        "could not be validated."
                    ),
                },
            },
            {
                "step_order": 2,
                "transition_status": "SUPPORTED",
                "hop_distance_m": None,
                "implied_speed_mps": None,
                "route_explanation": {
                    "step_type": "TRANSITION",
                    "status": "SUPPORTED",
                    "from_camera": "C02",
                    "to_camera":   "C03",
                    "summary": (
                        "Camera C03 was selected after C02 because the camera graph "
                        "contains a configured direct C02→C03 transition (expected "
                        "transit: 45s); timestamps are chronologically compatible "
                        "(observed gap: 188s); appearance match score 93.4% strongly "
                        "supports identity."
                    ),
                },
            },
        ],
    }

    top_candidates = [
        {
            "camera_id": r["camera_id"],
            "track_id": r["track_id"],
            "best_confidence": r["best_confidence"],
            "appearance_score": r["appearance_score"],
            "fused_score": r["fusion_score"],
            "match_status": r["match_status"],
            "face_veto_applied": False,
            "face_sim": None,
            "face_coverage": None,
            "kpr_sim": None,
            "first_seen": r["first_seen"],
            "crop_path": r["crop_path"],
            "active_backbone": "OSNet x1_0",
            "explanation": None,
        }
        for r in ROUTE
    ]

    async with async_session() as db:
        # Create completed query session
        session = QuerySession(
            person_id=None,
            query_image_path=str(
                Path("dataset/crops/C01/C01_track25_frame0820.jpg")
            ),
            status="done",
            progress_pct=100,
            submitted_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            match_decision="CONFIDENT_MATCH",
            top_candidates_json=json.dumps(top_candidates),
            route_data_json=json.dumps(route_meta),
        )
        db.add(session)
        await db.flush()  # get session.id

        sighting_ids = []
        for i, step_data in enumerate(ROUTE):
            s = Sighting(
                session_id=session.id,
                camera_id=step_data["camera_id"],
                track_id=step_data["track_id"],
                first_seen=step_data["first_seen"],
                last_seen=step_data["last_seen"],
                best_confidence=step_data["best_confidence"],
                mean_confidence=step_data["best_confidence"],
                crop_path=step_data["crop_path"],
                appearance_score=step_data["appearance_score"],
                spatial_score=1.0,
                temporal_score=1.0,
                fusion_score=step_data["fusion_score"],
                match_status=step_data["match_status"],
                fused_score=step_data["fusion_score"],
                body_similarity=step_data["appearance_score"],
                identity_score=step_data["appearance_score"],
                active_backbone="OSNet x1_0",
                embedding_dim=512,
                matching_mode="body",
            )
            db.add(s)
            await db.flush()
            sighting_ids.append(s.id)

        for i, sid in enumerate(sighting_ids):
            db.add(RouteStep(
                session_id=session.id,
                step_order=i,
                sighting_id=sid,
            ))

        await db.commit()
        print(f"\n[OK] Seeded route test session ID={session.id}")
        print(f"  Route: C01 T25 (10:02:27) -> C02 T37 (10:05:33) -> C03 T37 (10:08:44)")
        print(f"  Scores: 77.0% -> 77.0% -> 93.4%  [all CONFIDENT_MATCH]")
        print(f"\n  Open TRACE dashboard -> Route Reconstruction")
        print(f"    Select session #{session.id} from the dropdown")
        print(f"\n  Query image (use this to test the LIVE pipeline):")
        print(f"    dataset/crops/C01/C01_track25_frame0820.jpg")


if __name__ == "__main__":
    asyncio.run(seed())
