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
    # Fusion weights                                                       #
    # Body + face (Step 2):  body=0.70, face=0.30                        #
    # Body + face + KPR (Step 3): KPR dominates; body reduced to 0.40    #
    # Weights are re-normalised at runtime by the active signal count,    #
    # so these values only need to reflect relative importance.           #
    # ------------------------------------------------------------------ #
    fusion_body_weight: float = 0.40   # reduced: KPR handles body better
    fusion_face_weight: float = 0.30
    fusion_kpr_weight: float = 0.60   # KPR dominates — best at partial bodies

    # ------------------------------------------------------------------ #
    # KPR (Keypoint Promptable Re-Identification, ECCV 2024)              #
    # ------------------------------------------------------------------ #
    # Paths are intentionally empty — supply at runtime via environment
    # variables (KPR_WEIGHTS_PATH, KPR_CONFIG_PATH) or a .env file.
    kpr_weights_path: str = ""   # path to .pth.tar checkpoint
    kpr_config_path: str = ""    # path to KPR yaml config
    kpr_vis_threshold: float = 0.30  # minimum per-part visibility to include

    # ------------------------------------------------------------------ #
    # Face detection / recognition (SCRFD + ArcFace, ONNX Runtime)        #
    # ------------------------------------------------------------------ #
    face_det_threshold: float = 0.50
    # Paths are intentionally empty by default — supply at runtime via
    # environment variables (FACE_DET_MODEL, FACE_REC_MODEL) or a .env file.
    face_det_model: str = ""     # path to det_10g.onnx
    face_rec_model: str = ""     # path to w600k_r50.onnx

    # ------------------------------------------------------------------ #
    # Confidence scaling (from Phase 4.5 observed distributions)          #
    # ------------------------------------------------------------------ #
    similarity_observed_min: float = 0.3315049352393543
    similarity_observed_max: float = 0.9730031552165505

    # ------------------------------------------------------------------ #
    # SOLIDER-REID (Swin-Small, 768-dim)                                  #
    # ------------------------------------------------------------------ #
    # Paths are intentionally empty by default — supply them at runtime
    # via environment variables (SOLIDER_WEIGHTS, SOLIDER_CONFIG_PATH) or
    # override in a .env file.  embed_solider.py accepts these values via
    # its own --weights / --solider-config CLI flags independently of the
    # backend settings; these fields exist so the backend can load a
    # SOLIDER embedding file and know its dimension without re-running
    # inference.
    solider_weights: Path = Path("")
    solider_config_path: Path = Path("")
    solider_embedding_dim: int = 768

    # ------------------------------------------------------------------ #
    # Detection                                                            #
    # ------------------------------------------------------------------ #
    yolo_model: str = "yolov8n.pt"
    detection_confidence: float = 0.6
    detection_classes: list[int] = [0]   # person only


settings = Settings()
