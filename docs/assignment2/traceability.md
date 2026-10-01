# Traceability Matrix

## Requirement → Component → Implementation → Diagram

| Req ID | Requirement Summary | System Component | Implementation File(s) | DFD Ref | UML Ref |
|---|---|---|---|---|---|
| FR-01 | Video upload | Upload Router | `backend/app/routers/upload.py` | DFD L1 §1.0 | Use Case UC1, Seq |
| FR-02 | Input validation | Upload Router | `backend/app/routers/upload.py` | DFD L1 §1.0, DFD L2 §1.1 | Use Case UC2 |
| FR-03 | Person detection | Detection Service + AI Script | `backend/app/services/detection_service.py`, `ai_pipeline/detection/detect.py` | DFD L2 §2.1 | Component |
| FR-04 | Multi-object tracking | Tracker Service + AI Script | `backend/app/services/tracker_service.py`, `ai_pipeline/tracking/track.py` | DFD L2 §2.2 | Component |
| FR-05 | Quality-filtered crop extraction | Crop Service + AI Script | `backend/app/services/crop_service.py`, `ai_pipeline/reid/crop_extractor.py` | DFD L2 §3.1–3.3 | Component |
| FR-06 | OSNet body embedding | Embedding Service + AI Script | `backend/app/services/embedding_service.py`, `ai_pipeline/reid/embed.py` | DFD L2 §3.4 | Component |
| FR-07 | Face embedding (optional) | Upload Router + AI Script | `backend/app/routers/upload.py`, `ai_pipeline/reid/face_embed.py` | DFD L2 §3.5 | Component |
| FR-08 | KPR embedding (optional) | Upload Router + AI Script | `backend/app/routers/upload.py`, `ai_pipeline/reid/embed_kpr.py` | DFD L2 §3.6 | Component |
| FR-09 | Person registration | Persons Router | `backend/app/routers/persons.py` | DFD L1 | Use Case UC4 |
| FR-10 | Reference photo enrollment | Persons Router + EmbeddingService | `backend/app/routers/persons.py`, `backend/app/services/embedding_service.py` | DFD L1 | Use Case UC5 |
| FR-11 | Search query submission | Queries Router | `backend/app/routers/queries.py` | DFD L1 §4.0 | Use Case UC9, Seq |
| FR-12 | Query image auto-crop | PipelineService | `backend/app/services/pipeline_service.py` | DFD L2 §4.1 | Activity, Seq |
| FR-13 | Cross-camera matching | MatchingService | `backend/app/services/matching_service.py` | DFD L2 §4.4 | Activity, Seq |
| FR-14 | Triple fusion matching | MatchingService | `backend/app/services/matching_service.py` | DFD L2 §4.5 | Activity |
| FR-15 | No-confident-match threshold | MatchingService | `backend/app/services/matching_service.py` | DFD L2 §4.6 | Activity |
| FR-16 | Route reconstruction | RouteService | `backend/app/services/route_service.py` | DFD L2 §5.1–5.7 | Activity, Seq |
| FR-17 | Sighting/route persistence | PipelineService | `backend/app/services/pipeline_service.py` | DFD L1 §6.0 | Seq |
| FR-18 | Watchlist alert generation | PipelineService | `backend/app/services/pipeline_service.py` | DFD L1 §6.0 | Seq, Use Case UC14 |
| FR-19 | WebSocket real-time progress | WebSocketManager | `backend/app/core/websocket_manager.py` | DFD L1 §7.0 | Seq, Component |
| FR-20 | Route result retrieval | Queries Router | `backend/app/routers/queries.py` | DFD L1 §7.0 | Use Case UC11 |
| FR-21 | Dashboard analytics | Analytics Router | `backend/app/routers/analytics.py` | DFD L1 §7.0 | Use Case UC13 |
| FR-22 | Camera configuration | Cameras Router | `backend/app/routers/cameras.py` | DFD L1 | Use Case UC17 |
| FR-23 | Alert acknowledgement | Analytics Router | `backend/app/routers/analytics.py` | DFD L1 §7.0 | Use Case UC15 |
| FR-24 | Upload job status polling | Upload Router | `backend/app/routers/upload.py` | DFD L1 §1.0 | Use Case UC3 |

## Diagram Cross-Reference

| Diagram | File | Requirements Covered |
|---|---|---|
| DFD Level 0 (Context) | `DFD_Level_0.mmd` | All (system boundary) |
| DFD Level 1 | `DFD_Level_1.mmd` | FR-01–FR-24 (process decomposition) |
| DFD Level 2 — Video Processing | `DFD_Level_2_Video_Processing.mmd` | FR-01–FR-08 |
| DFD Level 2 — Matching | `DFD_Level_2_Matching.mmd` | FR-11–FR-15 |
| DFD Level 2 — Route | `DFD_Level_2_Route.mmd` | FR-16–FR-19 |
| UML Use Case | `UML_Use_Case.puml` | FR-01, FR-09–FR-11, FR-18, FR-20–FR-24 |
| UML Activity | `UML_Activity.puml` | FR-11–FR-18 |
| UML Sequence | `UML_Sequence.puml` | FR-11–FR-20 |
| UML Component | `UML_Component.puml` | All (architectural view) |
| UML Class | `UML_Class.puml` | All (structural view) |

## Verification Notes

- All requirements FR-01 through FR-24 are verified as **IMPLEMENTED** from source code inspection.
- No requirement references a feature that is absent from the codebase.
- Optional features (FR-07, FR-08) are explicitly marked as dependent on external model weights.
- The no-match threshold value (0.74) is sourced from `ai_pipeline/config.yaml` and `backend/app/config.py`.
- Fusion weights are sourced from `backend/app/config.py` (Settings class).