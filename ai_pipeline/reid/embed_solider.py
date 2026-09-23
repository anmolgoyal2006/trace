"""
embed_solider.py — SOLIDER Body Embedding Pipeline for TRACE.

Runs SOLIDER Swin-Small (transformer ReID, 768-dim) over quality-filtered
person crops and writes one L2-normalised appearance embedding per crop.
Output records use the SAME schema as embed.py (OSNet), so the gallery file
can replace dataset/embeddings_<CAM>.json directly — the matching pipeline
is dimension-agnostic (cosine similarity works on any vector length).

Model nuance: SOLIDER eval forward returns (features, featmaps); with
TEST.NECK_FEAT='before' (see SOLIDER-REID/configs/market/swin_small.yml)
`features` is the 768-dim global feature BEFORE the BN neck — the standard
retrieval feature. We L2-normalise, matching SOLIDER's own evaluation.

Usage:
    python ai_pipeline/reid/embed_solider.py \\
        --metadata  dataset/crops_metadata_C01.json \\
        --crop-dir  dataset/crops/C01 \\
        --output    dataset/embeddings_C01.json \\
        --weights   pretrained_models/SOLIDER_REID_swin_small.pth \\
        --solider-config SOLIDER-REID/configs/market/swin_small.yml \\
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
import yaml
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]   # .../Trace/
_SOLIDER_DEFAULT_ROOT = _REPO_ROOT / "SOLIDER-REID"

# ---------------------------------------------------------------------------
# SOLIDER expected input: 384 (height) × 128 (width), RGB, mean/std 0.5
# (SOLIDER-REID INPUT.SIZE_TEST / PIXEL_MEAN / PIXEL_STD)
# ---------------------------------------------------------------------------

INPUT_HEIGHT = 384
INPUT_WIDTH = 128

SOLIDER_MEAN = [0.5, 0.5, 0.5]
SOLIDER_STD = [0.5, 0.5, 0.5]

SOLIDER_TRANSFORM = transforms.Compose([
    transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH)),
    transforms.ToTensor(),                                      # → [0, 1] float32
    transforms.Normalize(mean=SOLIDER_MEAN, std=SOLIDER_STD),
])

# ---------------------------------------------------------------------------
# Failure reason codes — same vocabulary as embed.py (Phase 4.6)
# ---------------------------------------------------------------------------

REASON_MISSING_FILE        = "missing_file"
REASON_UNREADABLE_IMAGE    = "unreadable_image"
REASON_CROP_TOO_SMALL      = "crop_too_small"
REASON_INVALID_IMAGE       = "invalid_image"
REASON_PREPROCESSING_ERROR = "preprocessing_error"
REASON_INFERENCE_ERROR     = "inference_error"


class CropFailure(NamedTuple):
    """Structured record for a crop that could not be embedded."""
    crop_path: str
    reason: str
    details: str

    def to_dict(self) -> dict:
        return {
            "crop_path": self.crop_path,
            "reason":    self.reason,
            "details":   self.details,
        }


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def select_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] PyTorch version   : {torch.__version__}")
    print(f"[INFO] CUDA available    : {torch.cuda.is_available()}")
    print(f"[INFO] Device            : {device}")
    return device


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _ensure_solider_on_path(solider_root: Path) -> None:
    """Add the SOLIDER-REID repo root to sys.path for its model modules."""
    if not solider_root.exists():
        sys.exit(
            f"[ERROR] SOLIDER root not found: {solider_root}\n"
            "Clone the repo first:\n"
            "  git clone https://github.com/tinyvision/SOLIDER-REID"
        )
    if not (solider_root / "model" / "make_model.py").exists():
        sys.exit(
            f"[ERROR] {solider_root / 'model' / 'make_model.py'} not found.\n"
            "Ensure --solider-root points to the cloned SOLIDER-REID repository."
        )
    root_str = str(solider_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _load_solider_cfg(solider_config: Path):
    """Load SOLIDER yacs config (defaults + yaml override)."""
    from config import cfg  # noqa: PLC0415  (SOLIDER-REID/config)

    cfg.defrost()
    cfg.merge_from_file(str(solider_config))
    cfg.freeze()
    return cfg


def load_solider_model(
    weights_path: Path,
    solider_config: Path,
    solider_root: Path,
    device: torch.device,
) -> torch.nn.Module:
    """
    Build SOLIDER transformer (Swin-Small per config) and load a ReID
    checkpoint via the repo's own load_param (tolerates classifier shape
    mismatch — only backbone/neck weights matter for retrieval).
    """
    if not weights_path.exists():
        sys.exit(f"[ERROR] SOLIDER weights not found: {weights_path}")
    if not solider_config.exists():
        sys.exit(f"[ERROR] SOLIDER config not found: {solider_config}")

    _ensure_solider_on_path(solider_root)

    try:
        from model.make_model import make_model  # noqa: PLC0415
    except ImportError as exc:
        sys.exit(f"[ERROR] Cannot import SOLIDER modules: {exc}")

    print(f"\n[INFO] Loading SOLIDER config  : {solider_config}")
    cfg = _load_solider_cfg(solider_config)

    print(f"[INFO] Building {cfg.MODEL.TRANSFORMER_TYPE} ...")
    # num_class / camera / view only size the (unused-at-inference)
    # classifier head; checkpoint classifier mismatch is skipped on load.
    model = make_model(
        cfg,
        num_class=1,
        camera_num=0,
        view_num=0,
        semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
    )
    print(f"[INFO] Loading SOLIDER weights : {weights_path}")
    try:
        model.load_param(str(weights_path))
    except Exception as exc:
        sys.exit(f"[ERROR] Failed to load SOLIDER weights: {exc}")

    model.to(device)
    model.eval()
    # Remember the neck setting for the forward-pass unpacking below.
    model._trace_neck_feat = cfg.TEST.NECK_FEAT
    print(f"[INFO] SOLIDER ready on {device} (neck_feat={cfg.TEST.NECK_FEAT})")
    return model


def solider_forward(
    model: torch.nn.Module, batch: torch.Tensor
) -> torch.Tensor:
    """
    Run SOLIDER eval forward and return L2-normalised [B, 768] features.
    Handles the (features, featmaps) tuple the repo returns in eval mode.
    """
    out = model(batch)
    feats = out[0] if isinstance(out, (tuple, list)) else out
    return F.normalize(feats.float(), p=2, dim=-1)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def load_metadata(metadata_path: Path) -> list[dict]:
    """Load and return the crop metadata list."""
    if not metadata_path.exists():
        sys.exit(f"[ERROR] Metadata file not found: {metadata_path}")
    with open(metadata_path) as f:
        records = json.load(f)
    if not isinstance(records, list):
        sys.exit(f"[ERROR] Expected a JSON list in {metadata_path}, got {type(records)}")
    print(f"[INFO] Loaded metadata   : {len(records)} records from {metadata_path}")
    return records


# ---------------------------------------------------------------------------
# Per-crop validation and preprocessing (same edge cases as embed.py)
# ---------------------------------------------------------------------------

def validate_and_preprocess(
    image_path: Path,
    crop_path_str: str,
    min_crop_width: int,
    min_crop_height: int,
) -> tuple[Optional[torch.Tensor], Optional[CropFailure]]:
    """Validate a single crop image and return its preprocessed tensor."""
    if not image_path.exists():
        return None, CropFailure(
            crop_path=crop_path_str,
            reason=REASON_MISSING_FILE,
            details=f"File does not exist: {image_path}",
        )

    try:
        img = Image.open(image_path)
        img.verify()
    except (UnidentifiedImageError, Exception) as exc:
        return None, CropFailure(
            crop_path=crop_path_str,
            reason=REASON_UNREADABLE_IMAGE,
            details=f"PIL failed to open/verify: {exc}",
        )

    try:
        img = Image.open(image_path)
    except Exception as exc:
        return None, CropFailure(
            crop_path=crop_path_str,
            reason=REASON_UNREADABLE_IMAGE,
            details=f"PIL re-open failed: {exc}",
        )

    try:
        w, h = img.size
    except Exception as exc:
        return None, CropFailure(
            crop_path=crop_path_str,
            reason=REASON_INVALID_IMAGE,
            details=f"Cannot read image size: {exc}",
        )

    if w == 0 or h == 0:
        return None, CropFailure(
            crop_path=crop_path_str,
            reason=REASON_INVALID_IMAGE,
            details=f"Zero dimension: width={w}, height={h}",
        )

    if w < min_crop_width or h < min_crop_height:
        return None, CropFailure(
            crop_path=crop_path_str,
            reason=REASON_CROP_TOO_SMALL,
            details=f"Crop is {w}×{h} px — below minimum {min_crop_width}×{min_crop_height} px",
        )

    try:
        img_rgb = img.convert("RGB")
    except Exception as exc:
        return None, CropFailure(
            crop_path=crop_path_str,
            reason=REASON_INVALID_IMAGE,
            details=f"Cannot convert to RGB: {exc}",
        )

    try:
        tensor = SOLIDER_TRANSFORM(img_rgb)   # [3, INPUT_HEIGHT, INPUT_WIDTH]
    except Exception as exc:
        return None, CropFailure(
            crop_path=crop_path_str,
            reason=REASON_PREPROCESSING_ERROR,
            details=f"torchvision transform failed: {exc}",
        )

    return tensor, None


# ---------------------------------------------------------------------------
# Embedding validation
# ---------------------------------------------------------------------------

def validate_batch_output(
    embeddings: torch.Tensor,
    expected_batch_size: int,
    expected_dim: int,
) -> list[str]:
    """Validate a batch of embeddings straight off the model."""
    errors = []

    if not isinstance(embeddings, torch.Tensor):
        errors.append(f"Output is not a tensor: {type(embeddings)}")
        return errors

    if embeddings.ndim != 2:
        errors.append(f"Expected 2D tensor [B, D], got shape {list(embeddings.shape)}")
        return errors

    actual_batch, actual_dim = embeddings.shape

    if actual_batch != expected_batch_size:
        errors.append(
            f"Batch size mismatch: expected {expected_batch_size}, got {actual_batch}"
        )

    if actual_dim != expected_dim:
        errors.append(
            f"Embedding dim mismatch: expected {expected_dim}, got {actual_dim}"
        )

    if torch.isnan(embeddings).any():
        errors.append(f"NaN values detected: {torch.isnan(embeddings).sum().item()}")

    if torch.isinf(embeddings).any():
        errors.append(f"Infinite values detected: {torch.isinf(embeddings).sum().item()}")

    return errors


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_solider_embedding_pipeline(
    metadata_path: Path,
    crop_dir: Path,
    output_path: Path,
    batch_size: int,
    weights_path: Path,
    solider_config: Path,
    solider_root: Path,
    expected_dim: int,
    min_crop_width: int,
    min_crop_height: int,
    device: torch.device,
    overwrite: bool,
) -> int:
    """
    Run the full SOLIDER embedding pipeline.

    Returns
    -------
    int  — number of failed crops (0 if all succeeded)
    """
    if output_path.exists() and not overwrite:
        sys.exit(
            f"[ERROR] Output file already exists: {output_path}\n"
            "Use --overwrite to replace it, or delete it manually first."
        )
    if output_path.exists() and overwrite:
        print(f"[WARNING] Overwriting existing file: {output_path}")

    metadata = load_metadata(metadata_path)
    total = len(metadata)

    if not crop_dir.exists():
        sys.exit(f"[ERROR] Crop directory not found: {crop_dir}")

    model = load_solider_model(weights_path, solider_config, solider_root, device)

    print(f"\n[INFO] Probing model output dimension...")
    with torch.no_grad():
        dummy = torch.zeros(1, 3, INPUT_HEIGHT, INPUT_WIDTH, device=device)
        actual_dim = solider_forward(model, dummy).shape[1]
    print(f"[INFO] Model output dim  : {actual_dim}")

    if actual_dim != expected_dim:
        print(
            f"[WARNING] Model output dimension ({actual_dim}) differs from "
            f"expected ({expected_dim}). Proceeding with actual dimension."
        )

    n_batches = math.ceil(total / batch_size)

    print(f"\n[INFO] Starting SOLIDER embedding pipeline")
    print(f"[INFO] Total crops       : {total}")
    print(f"[INFO] Batch size        : {batch_size}")
    print(f"[INFO] Number of batches : {n_batches}")
    print(f"[INFO] Output path       : {output_path}")
    print()

    results: list[dict] = []
    failures: list[CropFailure] = []
    t_start = time.perf_counter()

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total)
        batch_meta = metadata[batch_start:batch_end]

        tensors: list[torch.Tensor] = []
        valid_meta: list[dict] = []

        for rec in batch_meta:
            img_path = _REPO_ROOT / rec["crop_path"]
            tensor, failure = validate_and_preprocess(
                image_path=img_path,
                crop_path_str=rec["crop_path"],
                min_crop_width=min_crop_width,
                min_crop_height=min_crop_height,
            )
            if failure is not None:
                failures.append(failure)
                print(
                    f"  [SKIP] {failure.crop_path} "
                    f"— reason={failure.reason} — {failure.details}"
                )
            else:
                tensors.append(tensor)
                valid_meta.append(rec)

        if not tensors:
            print(
                f"  [WARN] Batch {batch_idx + 1}/{n_batches}: "
                "all images failed pre-processing validation, skipping inference"
            )
            continue

        try:
            batch_tensor = torch.stack(tensors).to(device)   # [B, 3, H, W]
            with torch.no_grad():
                batch_embeddings = solider_forward(model, batch_tensor)
        except Exception as exc:
            for rec in valid_meta:
                failures.append(CropFailure(
                    crop_path=rec["crop_path"],
                    reason=REASON_INFERENCE_ERROR,
                    details=f"Model forward pass raised: {exc}",
                ))
                print(f"  [FAIL] Inference error for {rec['crop_path']}: {exc}")
            continue

        errors = validate_batch_output(batch_embeddings, len(tensors), actual_dim)
        if errors:
            for err in errors:
                print(f"  [ERROR] Batch {batch_idx + 1}/{n_batches} validation: {err}")
            for rec in valid_meta:
                failures.append(CropFailure(
                    crop_path=rec["crop_path"],
                    reason=REASON_INFERENCE_ERROR,
                    details=f"Batch validation failed: {'; '.join(errors)}",
                ))
            continue

        emb_cpu = batch_embeddings.cpu().numpy()

        for i, rec in enumerate(valid_meta):
            results.append({
                "crop_path":            rec["crop_path"],
                "camera_id":            rec["camera_id"],
                "track_id":             rec["track_id"],
                "frame":                rec["frame"],
                "timestamp":            rec["timestamp"],
                "bbox":                 rec.get("bbox"),
                "detection_confidence": rec.get("detection_confidence"),
                "embedding":            emb_cpu[i].tolist(),
            })

        done = batch_end
        elapsed = time.perf_counter() - t_start
        print(
            f"  [Batch {batch_idx + 1}/{n_batches}] "
            f"{done}/{total} crops, {elapsed:.1f}s elapsed"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f)

    total_elapsed = time.perf_counter() - t_start
    print(f"\n[DONE] Embedded {len(results)}/{total} crops → {output_path}")
    print(f"[DONE] Failures: {len(failures)} | Time: {total_elapsed:.1f}s")
    if failures:
        reasons: dict[str, int] = {}
        for fl in failures:
            reasons[fl.reason] = reasons.get(fl.reason, 0) + 1
        print(f"[DONE] Failure breakdown: {reasons}")

    return len(failures)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SOLIDER Swin-Small batch embedding")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--crop-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--solider-config", type=Path, required=True)
    parser.add_argument("--solider-root", type=Path, default=_SOLIDER_DEFAULT_ROOT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--expected-dim", type=int, default=768)
    parser.add_argument("--min-crop-width", type=int, default=32)
    parser.add_argument("--min-crop-height", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = select_device()
    n_failed = run_solider_embedding_pipeline(
        metadata_path=args.metadata,
        crop_dir=args.crop_dir,
        output_path=args.output,
        batch_size=args.batch_size,
        weights_path=args.weights,
        solider_config=args.solider_config,
        solider_root=args.solider_root,
        expected_dim=args.expected_dim,
        min_crop_width=args.min_crop_width,
        min_crop_height=args.min_crop_height,
        device=device,
        overwrite=args.overwrite,
    )
    print(f"[EXIT] {n_failed} failed crop(s)")


if __name__ == "__main__":
    main()
