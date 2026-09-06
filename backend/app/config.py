"""
config.py — Unified Trace backend configuration.

All tunable values live here. Import `settings` anywhere in the backend.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


# Repository root: backend/app/config.py → parents[2] = Trace/
_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Paths                                                                 #
    # ------------------------------------------------------------------ #
    repo_root: Path = _REPO_ROOT
    db_path: Path = _REPO_ROOT / "db" / "trace.db"
    data_dir: Path = _REPO_ROOT / "dataset"
    uploads_dir: Path = _REPO_ROOT / "dataset" / "uploads"
    snapshots_dir: Path = _REPO_ROOT / "dataset" / "snapshots"
    embeddings_dir: Path = _REPO_ROOT / "dataset"
    raw_videos_dir: Path = _REPO_ROOT / "dataset" / "raw_videos"
    camera_graph_path: Path = _REPO_ROOT / "dataset" / "camera_graph.json"
    ai_config_path: Path = _REPO_ROOT / "ai_pipeline" / "config.yaml"
    frontend_dir: Path = _REPO_ROOT / "frontend"

    # ------------------------------------------------------------------ #
    # Server                                                               #
    # ------------------------------------------------------------------ #
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # ------------------------------------------------------------------ #
    # Re-ID / matching                                                     #
    # ------------------------------------------------------------------ #
    reid_model: str = "osnet_x1_0"
    embedding_dim: int = 512
    no_match_threshold: float = 0.74      # Phase 5.7 evidence-based value
    top_k: int = 5

    # ------------------------------------------------------------------ #
    # Fusion weights (appearance + spatial + temporal = 1.0)              #
    # ------------------------------------------------------------------ #
    fusion_appearance_weight: float = 0.60
    fusion_spatial_weight: float = 0.25
    fusion_temporal_weight: float = 0.15

    # ------------------------------------------------------------------ #
    # Confidence scaling (from Phase 4.5 observed distributions)          #
    # ------------------------------------------------------------------ #
    similarity_observed_min: float = 0.3315049352393543
    similarity_observed_max: float = 0.9730031552165505

    # ------------------------------------------------------------------ #
    # Detection                                                            #
    # ------------------------------------------------------------------ #
    yolo_model: str = "yolov8n.pt"
    detection_confidence: float = 0.6
    detection_classes: list[int] = [0]   # person only


settings = Settings()
