"""
embed.py — Phase 4.6
Batch Re-ID Embedding Pipeline for TRACE — with edge-case handling.

Reads the final Phase 3 quality-filtered crop metadata, processes every
crop image through pretrained OSNet x1_0 in batches, and writes one
appearance embedding per crop to dataset/embeddings_C01.json.

Edge cases handled (Phase 4.6):
  - Missing file         → logged, skipped, reason=missing_file
  - Corrupted/unreadable → logged, skipped, reason=unreadable_image
  - Too-small crop       → logged, skipped, reason=crop_too_small  (uses config thresholds)
  - Invalid image        → logged, skipped, reason=invalid_image
  - Preprocessing error  → logged, skipped, reason=preprocessing_error
  - Inference error      → logged, skipped (per-crop isolated), reason=inference_error

One bad crop never terminates the batch or the pipeline.

Does NOT run YOLO, ByteTrack, or any training step.
Does NOT implement cross-camera matching.

Usage (recommended — run on Google Colab T4 GPU):
    python ai_pipeline/reid/embed.py \\
        --metadata  dataset/crops_metadata_phase3_final.json \\
        --crop-dir  dataset/crops_phase3_final \\
        --output    dataset/embeddings_C01.json \\
        --batch-size 32

Options:
    --metadata      Path to Phase 3 final metadata JSON (required)
    --crop-dir      Directory containing the crop images (required)
    --output        Destination path for embeddings JSON (required)
    --batch-size    Crops per GPU forward pass (default: config value or 32)
    --overwrite     Overwrite an existing output file without prompting
    --config        Path to config.yaml (default: ai_pipeline/config.yaml)
    --strict        Exit non-zero if any crops failed (default: exit 0 always)

Expected result on the TRACE MVP dataset:
    Total metadata records : 353
    Successfully embedded  : 353
    Failed                 : 0
    Embedding dimension    : 512
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import NamedTuple

import torch
import yaml
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT   = Path(__file__).resolve().parents[2]   # .../Trace/
_CONFIG_PATH = _REPO_ROOT / "ai_pipeline" / "config.yaml"

# ---------------------------------------------------------------------------
# ImageNet normalisation — identical to Phase 4.4 verified preprocessing
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# OSNet expected input: 256 (height) × 128 (width), RGB
INPUT_HEIGHT = 256
INPUT_WIDTH  = 128

_TRANSFORM = transforms.Compose([
    transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH)),
    transforms.ToTensor(),                                      # → [0, 1] float32
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# Failure reason codes — structured and traceable
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
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """Return the full config dict from config.yaml."""
    if not config_path.exists():
        sys.exit(f"[ERROR] Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_embedding_config(cfg: dict) -> dict:
    """Extract and validate the reid.embedding block."""
    try:
        return cfg["reid"]["embedding"]
    except KeyError as e:
        sys.exit(f"[ERROR] Missing config key: {e}. "
                 "Ensure reid.embedding exists in config.yaml.")


def get_min_crop_dims(cfg: dict) -> tuple[int, int]:
    """
    Read minimum crop dimensions from the reid block of config.yaml.

    Returns (min_width, min_height).  These are the same thresholds
    used by Phase 3 quality filtering — Phase 4 uses them as a
    defensive guard only.
    """
    reid_cfg = cfg.get("reid", {})
    min_w = int(reid_cfg.get("min_crop_width",  40))
    min_h = int(reid_cfg.get("min_crop_height", 100))
    return min_w, min_h


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def select_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] PyTorch version   : {torch.__version__}")
    print(f"[INFO] CUDA available    : {torch.cuda.is_available()}")
    print(f"[INFO] Device            : {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU               : {torch.cuda.get_device_name(0)}")
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[INFO] GPU memory        : {mem_total:.1f} GB")
    else:
        print("[INFO] GPU               : N/A (running on CPU)")
    return device


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: torch.device) -> torch.nn.Module:
    """Load pretrained OSNet from torchreid and set to eval mode."""
    try:
        import torchreid
    except ImportError:
        sys.exit(
            "[ERROR] torchreid is not installed.\n"
            "Install it with: pip install torchreid\n"
            "Or on Colab: !pip install torchreid"
        )

    print(f"\n[INFO] Loading {model_name} with pretrained weights...")
    model = torchreid.models.build_model(
        name=model_name,
        num_classes=1000,
        pretrained=True,
    )
    model.to(device)
    model.eval()
    print(f"[INFO] Model loaded and set to eval() on {device}")
    return model


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def load_metadata(metadata_path: Path) -> list[dict]:
    """Load and return the Phase 3 final crop metadata list."""
    if not metadata_path.exists():
        sys.exit(f"[ERROR] Metadata file not found: {metadata_path}")
    with open(metadata_path) as f:
        records = json.load(f)
    if not isinstance(records, list):
        sys.exit(f"[ERROR] Expected a JSON list in {metadata_path}, got {type(records)}")
    print(f"[INFO] Loaded metadata   : {len(records)} records from {metadata_path}")
    return records


# ---------------------------------------------------------------------------
# Per-crop validation and preprocessing (Phase 4.6)
# ---------------------------------------------------------------------------

def validate_and_preprocess(
    image_path: Path,
    crop_path_str: str,
    min_crop_width: int,
    min_crop_height: int,
) -> tuple[torch.Tensor | None, CropFailure | None]:
    """
    Validate a single crop image and return its preprocessed tensor.

    Returns
    -------
    (tensor, None)          on success
    (None, CropFailure)     on any failure — the caller should log and skip

    Failure reasons:
        missing_file        — file does not exist on disk
        unreadable_image    — PIL cannot decode the file bytes
        invalid_image       — decoded image has zero/unexpected dimensions
                              or cannot be converted to RGB
        crop_too_small      — image is smaller than config-defined minimums
        preprocessing_error — torchvision transform raised an exception
    """
    # ---- Edge Case 2: missing file ----------------------------------------
    if not image_path.exists():
        return None, CropFailure(
            crop_path=crop_path_str,
            reason=REASON_MISSING_FILE,
            details=f"File not found: {image_path}",
        )

    # ---- Edge Case 1: corrupted / unreadable image ------------------------
    try:
        img = Image.open(image_path)
        img.verify()          # raises if file is truncated / corrupt
    except (UnidentifiedImageError, Exception) as exc:
        return None, CropFailure(
            crop_path=crop_path_str,
            reason=REASON_UNREADABLE_IMAGE,
            details=f"PIL failed to open/verify: {exc}",
        )

    # Re-open after verify() (verify() leaves file in an unusable state)
    try:
        img = Image.open(image_path)
    except Exception as exc:
        return None, CropFailure(
            crop_path=crop_path_str,
            reason=REASON_UNREADABLE_IMAGE,
            details=f"PIL re-open failed: {exc}",
        )

    # ---- Edge Case 4: invalid image dimensions ----------------------------
    try:
        w, h = img.size   # PIL: (width, height)
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

    # ---- Edge Case 3: too-small crop (defensive Phase 4 check) -----------
    if w < min_crop_width or h < min_crop_height:
        return None, CropFailure(
            crop_path=crop_path_str,
            reason=REASON_CROP_TOO_SMALL,
            details=(
                f"Crop is {w}×{h} px — below minimum "
                f"{min_crop_width}×{min_crop_height} px "
                f"(config: min_crop_width={min_crop_width}, "
                f"min_crop_height={min_crop_height})"
            ),
        )

    # ---- Edge Case 4 continued: convert to RGB ----------------------------
    try:
        img_rgb = img.convert("RGB")
    except Exception as exc:
        return None, CropFailure(
            crop_path=crop_path_str,
            reason=REASON_INVALID_IMAGE,
            details=f"Cannot convert to RGB: {exc}",
        )

    # ---- Edge Case 5: preprocessing failure --------------------------------
    try:
        tensor = _TRANSFORM(img_rgb)   # [3, INPUT_HEIGHT, INPUT_WIDTH]
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
    """
    Validate a batch of embeddings straight off the model.

    Returns a list of error strings (empty = valid).
    """
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
        n = torch.isnan(embeddings).sum().item()
        errors.append(f"NaN values detected: {n}")

    if torch.isinf(embeddings).any():
        n = torch.isinf(embeddings).sum().item()
        errors.append(f"Infinite values detected: {n}")

    return errors


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_embedding_pipeline(
    metadata_path: Path,
    crop_dir: Path,
    output_path: Path,
    batch_size: int,
    model_name: str,
    expected_dim: int,
    min_crop_width: int,
    min_crop_height: int,
    device: torch.device,
    overwrite: bool,
) -> int:
    """
    Run the full embedding pipeline.

    Returns
    -------
    int  — number of failed crops (0 if all succeeded)
    """

    # --- guard: output already exists ---
    if output_path.exists() and not overwrite:
        sys.exit(
            f"[ERROR] Output file already exists: {output_path}\n"
            "Use --overwrite to replace it, or delete it manually first."
        )
    if output_path.exists() and overwrite:
        print(f"[WARNING] Overwriting existing file: {output_path}")

    # --- load inputs ---
    metadata = load_metadata(metadata_path)
    total    = len(metadata)

    # --- verify crop directory ---
    if not crop_dir.exists():
        sys.exit(f"[ERROR] Crop directory not found: {crop_dir}")

    # --- load model ---
    model = load_model(model_name, device)

    # --- determine actual embedding dim from a dummy forward pass ---
    print(f"\n[INFO] Probing model output dimension...")
    with torch.no_grad():
        dummy = torch.zeros(1, 3, INPUT_HEIGHT, INPUT_WIDTH, device=device)
        probe = model(dummy)
        actual_dim = probe.shape[1]
    print(f"[INFO] Model output dim  : {actual_dim}")

    if actual_dim != expected_dim:
        print(
            f"[WARNING] Model output dimension ({actual_dim}) differs from "
            f"config expected_dim ({expected_dim}). Proceeding with actual "
            f"model output dimension."
        )
    else:
        print(f"[INFO] Embedding dim matches config: {actual_dim} ✓")

    print(f"[INFO] Min crop size     : {min_crop_width}×{min_crop_height} px (from config)")

    n_batches = math.ceil(total / batch_size)

    print(f"\n[INFO] Starting batch embedding pipeline")
    print(f"[INFO] Total crops       : {total}")
    print(f"[INFO] Batch size        : {batch_size}")
    print(f"[INFO] Number of batches : {n_batches}")
    print(f"[INFO] Output path       : {output_path}")
    print()

    results: list[dict]       = []   # successfully embedded records
    failures: list[CropFailure] = [] # structured failure records
    t_start = time.perf_counter()

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end   = min(batch_start + batch_size, total)
        batch_meta  = metadata[batch_start:batch_end]

        # --- Phase 4.6: validate and preprocess each crop individually ----
        #
        # One bad image must NOT prevent the other valid images in the batch
        # from being embedded.  Strategy:
        #   1. Validate + preprocess each image independently.
        #   2. Collect only the valid tensors and their metadata records.
        #   3. Run inference on the valid subset.
        #   4. Record failures separately with structured reason codes.

        tensors    : list[torch.Tensor] = []
        valid_meta : list[dict]         = []

        for rec in batch_meta:
            img_path = _REPO_ROOT / rec["crop_path"]
            tensor, failure = validate_and_preprocess(
                image_path=img_path,
                crop_path_str=rec["crop_path"],
                min_crop_width=min_crop_width,
                min_crop_height=min_crop_height,
            )

            if failure is not None:
                # Log the failure and continue — do NOT abort the batch
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

        # --- Edge Case 6: isolate inference failures ----------------------
        #
        # Run inference on the valid subset.  If the entire batch raises,
        # catch it and record every crop in that batch as inference_error,
        # then continue to the next batch.

        try:
            batch_tensor = torch.stack(tensors).to(device)   # [B, 3, H, W]
            with torch.no_grad():
                batch_embeddings = model(batch_tensor)        # [B, actual_dim]
        except Exception as exc:
            # Entire batch inference failed — mark each valid crop failed
            for rec in valid_meta:
                failures.append(CropFailure(
                    crop_path=rec["crop_path"],
                    reason=REASON_INFERENCE_ERROR,
                    details=f"Model forward pass raised: {exc}",
                ))
                print(
                    f"  [FAIL] Inference error for {rec['crop_path']}: {exc}"
                )
            continue

        # --- validate batch output shape and values -----------------------
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

        # --- move to CPU and serialise to Python lists for JSON -----------
        emb_cpu = batch_embeddings.cpu().numpy()   # [B, actual_dim], float32 ndarray

        for i, rec in enumerate(valid_meta):
            embedding_list = emb_cpu[i].tolist()
            results.append({
                "crop_path":            rec["crop_path"],
                "camera_id":            rec["camera_id"],
                "track_id":             rec["track_id"],
                "frame":                rec["frame"],
                "timestamp":            rec["timestamp"],
                "bbox":                 rec["bbox"],
                "detection_confidence": rec["detection_confidence"],
                "embedding":            embedding_list,
            })

        n_failed_this_batch = len(batch_meta) - len(tensors)
        print(
            f"  Batch {batch_idx + 1:>2}/{n_batches}: "
            f"{batch_end}/{total} crops  "
            f"({len(tensors)} embedded, {n_failed_this_batch} skipped this batch)"
        )

    t_elapsed = time.perf_counter() - t_start

    # --- summary report ---------------------------------------------------
    n_success = len(results)
    n_failed  = len(failures)
    n_missing = total - n_success - n_failed

    print()
    print("=" * 60)
    print("TRACE Phase 4.6 — Embedding Pipeline Results")
    print("=" * 60)
    print(f"  Total metadata records : {total}")
    print(f"  Successfully embedded  : {n_success}")
    print(f"  Failed                 : {n_failed}")
    print(f"  Missing / not processed: {n_missing}")
    print(f"  Embedding dimension    : {actual_dim}")
    print(f"  Total time             : {t_elapsed:.2f}s")
    if n_success > 0:
        print(f"  Crops/second           : {n_success / t_elapsed:.1f}")
        print(f"  ms/crop                : {1000 * t_elapsed / n_success:.2f}")
    print("=" * 60)

    # --- structured failure breakdown by reason --------------------------
    if n_failed > 0:
        from collections import Counter
        reason_counts = Counter(f.reason for f in failures)

        print(f"\n  Failures by reason:")
        for reason in [
            REASON_UNREADABLE_IMAGE,
            REASON_MISSING_FILE,
            REASON_CROP_TOO_SMALL,
            REASON_INVALID_IMAGE,
            REASON_PREPROCESSING_ERROR,
            REASON_INFERENCE_ERROR,
        ]:
            count = reason_counts.get(reason, 0)
            if count:
                print(f"    - {reason}: {count}")

        print(f"\n  Failed crops:")
        for f in failures:
            print(f"    ✗ [{f.reason}] {f.crop_path}")
            print(f"      {f.details}")

    if n_success != total:
        print(
            f"\n[WARNING] Expected {total} embeddings but produced {n_success}. "
            f"{n_failed} crop(s) were skipped due to errors."
        )
    else:
        print(f"\n[INFO] All {total} crops successfully embedded ✓")

    # --- write output JSON -----------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] Embeddings written : {output_path}  ({n_success} records)")

    if n_success == total and n_failed == 0:
        print("\n[DONE] Phase 4 embedding complete — all crops embedded successfully.")
    else:
        print("\n[DONE] Phase 4 embedding complete — see failure summary above.")

    return n_failed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRACE Phase 4.6 — Batch Re-ID Embedding Pipeline (edge-case robust)"
    )
    parser.add_argument(
        "--metadata",
        required=True,
        type=Path,
        help="Path to crops_metadata_phase3_final.json",
    )
    parser.add_argument(
        "--crop-dir",
        required=True,
        type=Path,
        help="Directory containing the Phase 3 final crop images",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination path for embeddings JSON (e.g. dataset/embeddings_C01.json)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Crops per GPU forward pass (default: value from config.yaml)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file without prompting",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_CONFIG_PATH,
        help=f"Path to config.yaml (default: {_CONFIG_PATH})",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit non-zero if any crops failed. "
            "Default behaviour: always exit 0 after a completed run."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg     = load_config(args.config)
    emb_cfg = get_embedding_config(cfg)

    model_name   = emb_cfg.get("model", "osnet_x1_0")
    expected_dim = int(emb_cfg.get("embedding_dim", 512))
    config_batch = int(emb_cfg.get("batch_size", 32))
    batch_size   = args.batch_size if args.batch_size is not None else config_batch

    min_crop_width, min_crop_height = get_min_crop_dims(cfg)

    print("=" * 60)
    print("TRACE — Phase 4.6: Batch Re-ID Embedding Pipeline")
    print("=" * 60)
    print(f"[INFO] Model             : {model_name}")
    print(f"[INFO] Expected emb dim  : {expected_dim}")
    print(f"[INFO] Batch size        : {batch_size}")
    print(f"[INFO] Min crop width    : {min_crop_width} px")
    print(f"[INFO] Min crop height   : {min_crop_height} px")
    print(f"[INFO] Metadata          : {args.metadata}")
    print(f"[INFO] Crop directory    : {args.crop_dir}")
    print(f"[INFO] Output            : {args.output}")
    print(f"[INFO] Overwrite         : {args.overwrite}")
    print(f"[INFO] Strict mode       : {args.strict}")
    print()

    device = select_device()

    n_failed = run_embedding_pipeline(
        metadata_path=args.metadata,
        crop_dir=args.crop_dir,
        output_path=args.output,
        batch_size=batch_size,
        model_name=model_name,
        expected_dim=expected_dim,
        min_crop_width=min_crop_width,
        min_crop_height=min_crop_height,
        device=device,
        overwrite=args.overwrite,
    )

    # Default: exit 0 even if some crops failed (resilient mode).
    # --strict: exit non-zero if any crop failed.
    if args.strict and n_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
