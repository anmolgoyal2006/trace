"""
config.py — Unified Trace backend configuration.

All tunable values live here. Import `settings` anywhere in the backend.
"""

from pathlib import Path
from pydantic import model_validator
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
    google_maps_api_key: str = ""

    # ------------------------------------------------------------------ #
    # Re-ID / matching — OSNet (512-dim)                                  #
    # ------------------------------------------------------------------ #
    reid_model: str = "osnet_x1_0"
    embedding_dim: int = 512
    # Phase 5.7 evidence-based threshold (small validation set only).
    # NOT a universally calibrated decision boundary.
    no_match_threshold: float = 0.74
    # Candidates above soft_floor but below no_match_threshold are returned
    # as POSSIBLE_MATCH for human review (not as CONFIDENT_MATCH).
    soft_floor: float = 0.60
    top_k: int = 5

    # ------------------------------------------------------------------ #
    # Re-ID / matching — SOLIDER (768-dim)                                #
    # IMPORTANT: These are INITIAL EMPIRICAL values, NOT calibrated.      #
    # Different-track pairs cluster at 0.83-0.92 for SOLIDER — OSNet's   #
    # 0.74 threshold is completely invalid here.                          #
    # Recalibrate with labelled genuine/impostor data before production.  #
    # ------------------------------------------------------------------ #
    solider_no_match_threshold: float = 0.94   # initial empirical — NOT calibrated
    solider_soft_floor: float = 0.90           # initial empirical — NOT calibrated
    solider_observed_min: float = 0.83         # empirical lower bound
    solider_observed_max: float = 0.995        # empirical upper bound

    # ------------------------------------------------------------------ #
    # Fusion weights (appearance + spatial + temporal = 1.0)              #
    # ------------------------------------------------------------------ #
    fusion_appearance_weight: float = 0.60
    fusion_spatial_weight: float = 0.25
    fusion_temporal_weight: float = 0.15

    # ------------------------------------------------------------------ #
    # Fusion weights — body + face + KPR                                  #
    # Re-normalised at runtime by active signal count.                    #
    # ------------------------------------------------------------------ #
    fusion_body_weight: float = 0.40   # reduced: KPR handles body better
    fusion_face_weight: float = 0.30
    fusion_kpr_weight: float = 0.60   # KPR dominates — best at partial bodies

    # ------------------------------------------------------------------ #
    # Route fusion weights — five components.                             #
    # Renormalized by active weight sum at runtime.                       #
    # INITIAL DEFAULTS — not proven optimal; calibrate with labelled data.#
    # ------------------------------------------------------------------ #
    route_body_weight: float = 0.25
    route_face_weight: float = 0.20
    route_kpr_weight: float = 0.15
    route_spatial_weight: float = 0.25
    route_temporal_weight: float = 0.15

    # Per-modality normalization bounds.
    # body: derived at runtime from confidence_scaling.get_backbone_config(embedding_dim).
    # face: ARBITRARY UN-CALIBRATED PLACEHOLDER. Replace after labelled ArcFace calibration.
    face_norm_min: float = 0.0
    face_norm_max: float = 0.70
    # kpr: PLACEHOLDER from Swin/SOLIDER range (kpr_swin_small shares Swin family). Recalibrate.
    kpr_norm_min: float = 0.83
    kpr_norm_max: float = 0.995

    # Route-transition face identity veto
    route_face_mismatch_threshold: float = 0.38
    route_face_min_coverage: float = 0.30
    route_face_min_query_confidence: float = 0.50
    route_face_confidence_threshold: float = 0.38  # must be >= route_face_mismatch_threshold
    route_face_include_uncertain: bool = True

    # Physical contradiction
    route_max_speed_mps: float = 6.0

    # Soft penalties (non-veto cases only)
    route_penalty_disconnected: float = 0.40
    route_penalty_too_fast: float = 0.30      # sprint-without-GPS only; GPS impossible = hard veto
    route_penalty_simultaneous: float = 0.50
    route_penalty_early: float = 0.0
    route_penalty_late: float = 0.0

    # Beam search
    route_beam_width: int = 3

    # ------------------------------------------------------------------ #
    # Face identity veto                                                   #
    # When both the query and gallery track have reliable face evidence,   #
    # a low face similarity can VETO a confident body/KPR match.          #
    #                                                                      #
    # face_match_threshold: face similarity must EXCEED this for a        #
    #   reliable face to be considered a match rather than a mismatch.    #
    #   INITIAL EMPIRICAL value — not statistically calibrated.           #
    #                                                                      #
    # face_min_coverage: minimum fraction of gallery track crops where     #
    #   a face was detected before the track's face signal is trusted.    #
    #   Below this → face evidence is unreliable → no veto applied.      #
    #                                                                      #
    # face_min_query_confidence: minimum SCRFD detection confidence for   #
    #   the query face to be considered reliable.                         #
    # ------------------------------------------------------------------ #
    face_match_threshold: float = 0.38      # initial empirical — NOT calibrated
    face_min_coverage: float = 0.30         # ≥30% crops must have a detected face
    face_min_query_confidence: float = 0.50 # SCRFD score for query face

    # ------------------------------------------------------------------ #
    # KPR (Keypoint Promptable Re-Identification, ECCV 2024)              #
    # ------------------------------------------------------------------ #
    kpr_weights_path: str = ""   # path to .pth.tar checkpoint
    kpr_config_path: str = ""    # path to KPR yaml config
    kpr_vis_threshold: float = 0.30  # minimum per-part visibility to include

    # ------------------------------------------------------------------ #
    # Face detection / recognition (SCRFD + ArcFace, ONNX Runtime)        #
    # ------------------------------------------------------------------ #
    face_det_threshold: float = 0.50
    face_det_model: str = ""     # path to det_10g.onnx
    face_rec_model: str = ""     # path to w600k_r50.onnx

    # ------------------------------------------------------------------ #
    # Path resolution helpers                                              #
    # Resolve model weight paths relative to repo root so the server can  #
    # be started from any working directory.                               #
    # ------------------------------------------------------------------ #
    @property
    def face_det_model_path(self) -> Path:
        """Absolute path to SCRFD weights; empty string → disabled."""
        if not self.face_det_model:
            return Path("")
        p = Path(self.face_det_model)
        return p if p.is_absolute() else _REPO_ROOT / p

    @property
    def face_rec_model_path(self) -> Path:
        """Absolute path to ArcFace weights; empty string → disabled."""
        if not self.face_rec_model:
            return Path("")
        p = Path(self.face_rec_model)
        return p if p.is_absolute() else _REPO_ROOT / p

    @property
    def kpr_weights_path_abs(self) -> Path:
        """Absolute path to KPR checkpoint; empty string → disabled."""
        if not self.kpr_weights_path:
            return Path("")
        p = Path(self.kpr_weights_path)
        return p if p.is_absolute() else _REPO_ROOT / p

    @property
    def kpr_config_path_abs(self) -> Path:
        """Absolute path to KPR config yaml; empty string → disabled."""
        if not self.kpr_config_path:
            return Path("")
        p = Path(self.kpr_config_path)
        return p if p.is_absolute() else _REPO_ROOT / p

    # ------------------------------------------------------------------ #
    # Confidence scaling (from Phase 4.5 observed distributions)          #
    # These are OSNet-specific — SOLIDER uses its own range above.        #
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

    @model_validator(mode="after")
    def _validate_route_config(self) -> "Settings":
        weights = [
            self.route_body_weight, self.route_face_weight, self.route_kpr_weight,
            self.route_spatial_weight, self.route_temporal_weight,
        ]
        if any(w < 0 for w in weights):
            raise ValueError("No route weight may be negative.")
        if all(w == 0 for w in weights):
            raise ValueError("All five route weights cannot be zero simultaneously.")
        if self.route_beam_width < 1:
            raise ValueError("route_beam_width must be >= 1.")
        if self.route_max_speed_mps <= 0:
            raise ValueError("route_max_speed_mps must be > 0.")
        if self.face_norm_max <= self.face_norm_min:
            raise ValueError("face_norm_max must be > face_norm_min.")
        if self.kpr_norm_max <= self.kpr_norm_min:
            raise ValueError("kpr_norm_max must be > kpr_norm_min.")
        if self.route_face_confidence_threshold < self.route_face_mismatch_threshold:
            raise ValueError(
                "route_face_confidence_threshold must be >= route_face_mismatch_threshold."
            )
        soft_penalties = [
            self.route_penalty_disconnected, self.route_penalty_too_fast,
            self.route_penalty_simultaneous, self.route_penalty_early, self.route_penalty_late,
        ]
        if any(p < 0 for p in soft_penalties):
            raise ValueError("Soft penalty values must be >= 0.")
        return self


settings = Settings()
