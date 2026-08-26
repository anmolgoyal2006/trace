# Scope

## Overview

Trace is a multi-camera person tracking and re-identification system designed for surveillance and analytics workflows.

## Core Capabilities

- **Detection**: YOLO-based person detection from video frames
- **Tracking**: ByteTrack for within-camera identity tracking
- **Re-Identification**: Deep appearance embeddings for cross-camera matching
- **Matching**: Fusion-based scoring combining appearance, spatial, and temporal cues
- **Route Reconstruction**: Graph-based path reconstruction across camera network
- **Prediction**: Next-location scoring using historical movement patterns

## Architecture

- **AI Pipeline** (`ai_pipeline/`): Core ML and tracking logic
- **Backend** (`backend/`): FastAPI service exposing pipeline results via REST API
- **Frontend** (`frontend/`): React dashboard for visualization
- **Dataset** (`dataset/`): Raw video, camera topology graph, metadata
- **Database** (`db/`): SQLite for persistent storage

## Tech Stack

- Python 3.11+
- Ultralytics YOLOv8
- ByteTrack
- PyTorch / torchreid
- FastAPI + SQLAlchemy
- React
- SQLite
