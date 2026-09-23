"""
embed_kpr.py — TRACE KPR Part-Based Embedding Pipeline (ECCV 2024)
===================================================================

Reads the same Phase 3 crop metadata JSON used by embed.py / embed_solider.py
and produces a parallel KPR embedding file.  For each body crop it produces:

  - holistic_embedding : 768-dim global body embedding (L2-normalised)
  - part_embeddings    : list of per-body-part embeddings (each 768-dim, L2-norm'd)
  - part_visibility    : per-part visibility score in [0, 1]

The key advantage over embed.py: when comparing two crops, kpr_similarity.py
uses ONLY the parts that are visible in BOTH images.  This fixes the
waist-up / partial-body failure case directly.

═══════════════════════════════════════════════════════════════════
 SETUP INSTRUCTIONS (run once before using this script)
═══════════════════════════════════════════════════════════════════

Step 1 — Clone the KPR repo
-----------------------------
  git clone https://github.com/VlSomers/keypoint_promptable_reidentification
  cd keypoint_promptable_reidentification

Step 2 — Create environment and install dependencies
-----------------------------------------------------
  conda create --name kpr python=3.10 pytorch==1.13.0 torchvision==0.14.0 \\
      pytorch-cuda=11.7 -c pytorch -c nvidia -y
  conda activate kpr
  pip install -r requirements.txt
  python setup.py develop   # installs torchreid from the KPR fork

  On Google Colab (CUDA 11.8, PyTorch 2.x):
    pip install torch==2.0.1 torchvision==0.15.2
    pip install yacs timm
    # then run setup.py develop from inside the cloned KPR directory

Step 3 — Download pretrained KPR weights
-----------------------------------------
  Market-1501 fine-tuned KPR (Swin-Small, ImageNet pretrained):
    https://drive.google.com/file/d/1Np5wu3nQa_Fl_z7Zw2kchJNC8JZVwsh5/view
    Save as: pretrained_models/kpr_occ_pt_IN_82.34_92.33_42323828.pth.tar

  SOLIDER pretrained KPR (higher accuracy, requires SOLIDER backbone):
    See https://github.com/VlSomers/keypoint_promptable_reidentification
    → README → "Download the pre-trained models"

  Pass the .pth.tar path via --kpr-weights.

Step 4 — Locate the KPR config file
-------------------------------------
  Use one of the provided configs, e.g.:
    configs/kpr/imagenet/kpr_occ_posetrack_test.yaml
    configs/kpr/market1501/kpr_swin_small.yaml   (if testing on Market)

  Pass the config path via --kpr-config.

Step 5 — Set model.load_weights in the yaml config to your weights path
  OR pass --kpr-weights on the CLI (this script overrides it automatically).

═══════════════════════════════════════════════════════════════════
 EXAMPLE RUN COMMAND
═══════════════════════════════════════════════════════════════════

  python ai_pipeline/reid/embed_kpr.py \\
      --metadata   dataset/crops_metadata_phase3_final.json \\
      --crop-dir   dataset/crops_phase3_final \\
      --output     dataset/kpr_embeddings_C01.json \\
      --kpr-weights pretrained_models/kpr_occ_pt_IN_82.34_92.33_42323828.pth.tar \\
      --kpr-config  configs/kpr/imagenet/kpr_occ_posetrack_test.yaml \\
      --kpr-root    /path/to/keypoint_promptable_reidentification \\
      --batch-size  16 \\
      --overwrite

═══════════════════════════════════════════════════════════════════
 OUTPUT SCHEMA  (one record per crop)
═══════════════════════════════════════════════════════════════════

  {
    "crop_path":            "dataset/crops_phase3_final/C01_track13_frame0042.jpg",
    "camera_id":            "C01",
    "track_id":             13,
    "frame":                42,
    "timestamp":            "10:02:03.400",
    "bbox":                 [x1, y1, x2, y2],
    "detection_confidence": 0.91,
    "holistic_embedding":   [...],           # 768-dim L2-normalised list
    "part_embeddings":      [[...], ...],    # num_parts × 768-dim lists
    "part_visibility":      [0.9, 0.7, 0.0, 0.8, 0.6]  # per-part float in [0, 1]
  }

Edge cases handled (identical to embed.py Phase 4.6):
  - Missing file         → logged, skipped, reason=missing_file
  - Corrupted/unreadable → logged, skipped, reason=unreadable_image
  - Too-small crop       → logged, skipped, reason=crop_too_small
  - Invalid image        → logged, skipped, reason=invalid_image
  - Preprocessing error  → logged, skipped, reason=preprocessing_error
  - Inference error      → logged, skipped per-crop, reason=inference_error
  - KPR root not found   → fatal error with actionable message
  - Unexpected output shape → logged, skipped, reason=inference_error

Prompt-optional mode: KPR is called with image only (no keypoint prompts).
The model supports this natively — no pose estimator is required.

Python 3.11 / Windows 11 + Google Colab compatible.
All paths are relative to the repository root.
"""

from __future__ import annotations

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

_REPO_ROOT   = Path(__file__).resolve().parents[2]   # .../trace/
_CONFIG_PATH = _REPO_ROOT / "ai_pipeline" / "config.yaml"

# ---------------------------------------------------------------------------
# KPR preprocessing — ImageNet normalisation, same as embed.py
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# KPR default input size (height × width)
INPUT_HEIGHT = 256
INPUT_WIDTH  = 128

_TRANSFORM = transforms.Compose([
    transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# Failure reason codes (identical to embed.py)
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
        return {"crop_path": self.crop_path, "reason": self.reason, "details": self.details}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """Return the full TRACE config dict from config.yaml."""
    if not config_path.exists():
        sys.exit(f"[ERROR] Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_kpr_config(cfg: dict) -> dict:
    """Extract and validate the reid.kpr block from TRACE config.yaml."""
    try:
        return cfg["reid"]["kpr"]
    except KeyError as e:
        sys.exit(
            f"[ERROR] Missing config key: {e}. "
            "Ensure reid.kpr exists in ai_pipeline/config.yaml."
        )


def get_min_crop_dims(cfg: dict) -> tuple[int, int]:
    """Return (min_width, min_height) from reid block of config.yaml."""
    reid_cfg = cfg.get("reid", {})
    return int(reid_cfg.get("min_crop_width", 40)), int(reid_cfg.get("min_crop_height", 100))


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
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"[INFO] GPU memory        : {mem_gb:.1f} GB")
    else:
        print("[INFO] GPU               : N/A (running on CPU)")
    return device


# ---------------------------------------------------------------------------
# KPR model loading
# ---------------------------------------------------------------------------

def _ensure_kpr_on_path(kpr_root: Path) -> None:
    """
    Add the KPR repo root to sys.path so its internal torchreid modules
    can be imported.

    Args:
        kpr_root: Path to the cloned KPR repository root.
                  Must contain torchreid/ and torchreid/scripts/builder.py.

    Raises:
        SystemExit: if the directory or required files are missing.
    """
    if not kpr_root.exists():
        sys.exit(
            f"[ERROR] KPR root not found: {kpr_root}\n"
            "Clone the repo first:\n"
            "  git clone https://github.com/VlSomers/keypoint_promptable_reidentification"
        )
    builder_path = kpr_root / "torchreid" / "scripts" / "builder.py"
    if not builder_path.exists():
        sys.exit(
            f"[ERROR] {builder_path} not found.\n"
            "Ensure --kpr-root points to the root of the cloned KPR repository "
            "(the directory that contains torchreid/ and main.py)."
        )
    root_str = str(kpr_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def load_kpr_model(
    kpr_weights: Path,
    kpr_config: Path,
    kpr_root: Path,
    device: torch.device,
):
    """
    Build and return a KPRFeatureExtractor ready for inference.

    The extractor is configured in prompt-optional mode: no keypoint prompts
    are passed (only the image).  KPR handles this natively.

    Args:
        kpr_weights: Path to the .pth.tar checkpoint file.
        kpr_config:  Path to the KPR yaml config file.
        kpr_root:    Root of the cloned KPR repository.
        device:      torch.device to run inference on.

    Returns:
        Configured KPRFeatureExtractor in eval mode.

    Raises:
        SystemExit: on missing files or import failures.
    """
    if not kpr_weights.exists():
        sys.exit(
            f"[ERROR] KPR weights not found: {kpr_weights}\n"
            "Download from:\n"
            "  https://drive.google.com/file/d/1Np5wu3nQa_Fl_z7Zw2kchJNC8JZVwsh5/view\n"
            "then pass the path via --kpr-weights."
        )
    if not kpr_config.exists():
        sys.exit(
            f"[ERROR] KPR config not found: {kpr_config}\n"
            "Use one of the configs under configs/kpr/ in the cloned KPR repo, "
            "e.g. configs/kpr/imagenet/kpr_occ_posetrack_test.yaml"
        )

    _ensure_kpr_on_path(kpr_root)

    try:
        from torchreid.scripts.builder import build_config   # noqa: PLC0415
        from torchreid.tools.feature_extractor import KPRFeatureExtractor  # noqa: PLC0415
    except ImportError as exc:
        sys.exit(
            f"[ERROR] Cannot import KPR modules: {exc}\n"
            "Ensure --kpr-root points to the cloned KPR repo and that you have\n"
            "run 'python setup.py develop' inside it."
        )

    print(f"\n[INFO] Loading KPR config    : {kpr_config}")
    kpr_cfg = build_config(config_path=str(kpr_config))

    # Override weights path and GPU usage from CLI / runtime context
    kpr_cfg.model.load_weights = str(kpr_weights)
    kpr_cfg.use_gpu = (device.type == "cuda")

    # Prompt-optional mode: disable inference-time keypoint prompting so we
    # do not need a pose estimator.  KPR still produces full part embeddings.
    if hasattr(kpr_cfg.model, "promptable_trans"):
        kpr_cfg.model.promptable_trans.disable_inference_prompting = True

    print(f"[INFO] Loading KPR weights   : {kpr_weights}")
    extractor = KPRFeatureExtractor(kpr_cfg, verbose=True)
    extractor.model.eval()
    print(f"[INFO] KPRFeatureExtractor ready on {device}")

    return extractor


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
# Per-crop validation and preprocessing (identical to embed.py Phase 4.6)
# ---------------------------------------------------------------------------

def validate_and_preprocess(
    image_path: Path,
    crop_path_str: str,
    min_crop_width: int,
    min_crop_height: int,
) -> tuple[torch.Tensor | None, CropFailure | None]:
    """
    Validate a single crop image and return its preprocessed tensor.

    Returns (tensor, None) on success or (None, CropFailure) on any failure.
    """
    if not image_path.exists():
        return None, CropFailure(crop_path_str, REASON_MISSING_FILE,
                                 f"File not found: {image_path}")

    try:
        img = Image.open(image_path)
        img.verify()
    except (UnidentifiedImageError, Exception) as exc:
        return None, CropFailure(crop_path_str, REASON_UNREADABLE_IMAGE,
                                 f"PIL failed to open/verify: {exc}")

    try:
        img = Image.open(image_path)
    except Exception as exc:
        return None, CropFailure(crop_path_str, REASON_UNREADABLE_IMAGE,
                                 f"PIL re-open failed: {exc}")

    try:
        w, h = img.size
    except Exception as exc:
        return None, CropFailure(crop_path_str, REASON_INVALID_IMAGE,
                                 f"Cannot read image size: {exc}")

    if w == 0 or h == 0:
        return None, CropFailure(crop_path_str, REASON_INVALID_IMAGE,
                                 f"Zero dimension: width={w}, height={h}")

    if w < min_crop_width or h < min_crop_height:
        return None, CropFailure(
            crop_path_str, REASON_CROP_TOO_SMALL,
            f"Crop is {w}×{h} px — below minimum {min_crop_width}×{min_crop_height} px",
        )

    try:
        img_rgb = img.convert("RGB")
    except Exception as exc:
        return None, CropFailure(crop_path_str, REASON_INVALID_IMAGE,
                                 f"Cannot convert to RGB: {exc}")

    try:
        tensor = _TRANSFORM(img_rgb)
    except Exception as exc:
        return None, CropFailure(crop_path_str, REASON_PREPROCESSING_ERROR,
                                 f"torchvision transform failed: {exc}")

    return tensor, None


# ---------------------------------------------------------------------------
# KPR output parsing
# ---------------------------------------------------------------------------

def _parse_kpr_output(
    embeddings_batch: "torch.Tensor",  # [B, P+H, D]
    vis_scores_batch: "torch.Tensor",  # [B, P+H]
    batch_idx_in_batch: int,
    holistic_idx: int,
) -> tuple[list[float], list[list[float]], list[float]]:
    """
    Parse KPR model output for a single sample.

    KPR returns embeddings shaped [B, num_parts + 1, D] where the last
    (or first, depending on config) slot is the holistic/foreground embedding.
    The holistic index is probed from the output and passed in.

    Returns
    -------
    holistic_embedding : list[float]          — 1-D, D-dim, L2-normalised
    part_embeddings    : list[list[float]]    — num_parts × D-dim, each L2-norm'd
    part_visibility    : list[float]          — num_parts floats in [0, 1]
    """
    import torch.nn.functional as F

    sample_embs = embeddings_batch[batch_idx_in_batch]   # [P+1, D]
    sample_vis  = vis_scores_batch[batch_idx_in_batch]   # [P+1]

    num_slots = sample_embs.shape[0]

    # L2-normalise every slot
    normed = F.normalize(sample_embs.float(), p=2, dim=-1)   # [P+1, D]

    # Holistic slot
    holistic = normed[holistic_idx].detach().cpu().tolist()

    # Part slots — everything except the holistic slot
    part_indices = [i for i in range(num_slots) if i != holistic_idx]
    part_embeddings = [normed[i].detach().cpu().tolist() for i in part_indices]
    part_visibility  = [float(sample_vis[i].item()) for i in part_indices]

    return holistic, part_embeddings, part_visibility


# ---------------------------------------------------------------------------
# Probe model output shape
# ---------------------------------------------------------------------------

def probe_kpr_output(
    extractor,
    device: torch.device,
) -> tuple[int, int, int, int]:
    """
    Run a dummy forward pass to detect:
        num_slots    — total output slots (parts + holistic)
        holistic_idx — index of the holistic/foreground embedding
        num_parts    — number of body-part slots (num_slots - 1)
        part_dim     — embedding dimension per part

    The holistic slot is identified as the one with the highest mean
    visibility score across a random batch, consistent with KPR's design
    where the foreground (holistic) branch always has score ≈ 1.0.

    Returns (num_slots, holistic_idx, num_parts, part_dim).
    """
    try:
        import torch.nn.functional as F
        dummy_img = torch.zeros(1, 3, INPUT_HEIGHT, INPUT_WIDTH)
        sample = {"image": dummy_img[0]}   # KPRFeatureExtractor expects numpy/BGR normally
                                            # but we pass a pre-tensored dict for the probe
        # Prefer direct model forward pass for probing
        with torch.no_grad():
            dummy_batch = dummy_img.to(device)
            model = extractor.model
            # KPR model forward without prompts
            output = model(images=dummy_batch)
            from torchreid.utils.tools import extract_test_embeddings  # noqa: PLC0415
            embs, vis, _, _ = extract_test_embeddings(output, extractor.cfg.model.kpr.test_embeddings)
            # embs: [1, num_slots, D], vis: [1, num_slots]

        num_slots = embs.shape[1]
        part_dim  = embs.shape[2]

        # Holistic slot = highest mean visibility (foreground branch ≈ 1.0)
        # NOTE: vis can be a Bool tensor on some builds — argmax is not
        # implemented for Bool on CPU, so cast to float first.
        mean_vis = vis[0].detach().cpu().float()  # [num_slots]
        holistic_idx = int(mean_vis.argmax().item())

        num_parts = num_slots - 1

        print(f"[INFO] KPR output shape  : embeddings=[1, {num_slots}, {part_dim}]")
        print(f"[INFO] Holistic slot idx : {holistic_idx}")
        print(f"[INFO] Number of parts   : {num_parts}")
        print(f"[INFO] Part embedding dim: {part_dim}")

        return num_slots, holistic_idx, num_parts, part_dim

    except Exception as exc:
        sys.exit(
            f"[ERROR] KPR output probing failed: {exc}\n"
            "Check that --kpr-weights and --kpr-config are compatible."
        )


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_kpr_embedding_pipeline(
    metadata_path: Path,
    crop_dir: Path,
    output_path: Path,
    batch_size: int,
    kpr_weights: Path,
    kpr_config: Path,
    kpr_root: Path,
    min_crop_width: int,
    min_crop_height: int,
    device: torch.device,
    overwrite: bool,
) -> int:
    """
    Run the full KPR part-based embedding pipeline.

    Returns
    -------
    int — number of failed crops (0 if all succeeded)
    """

    # --- guard: output already exists ------------------------------------
    if output_path.exists() and not overwrite:
        sys.exit(
            f"[ERROR] Output file already exists: {output_path}\n"
            "Use --overwrite to replace it, or delete it manually first."
        )
    if output_path.exists() and overwrite:
        print(f"[WARNING] Overwriting existing file: {output_path}")

    # --- load metadata ---------------------------------------------------
    metadata = load_metadata(metadata_path)
    total    = len(metadata)

    if not crop_dir.exists():
        sys.exit(f"[ERROR] Crop directory not found: {crop_dir}")

    # --- load KPR model --------------------------------------------------
    extractor = load_kpr_model(kpr_weights, kpr_config, kpr_root, device)

    # --- probe output shape ----------------------------------------------
    num_slots, holistic_idx, num_parts, part_dim = probe_kpr_output(extractor, device)

    # --- import KPR internals needed for batched forward -----------------
    try:
        from torchreid.utils.tools import extract_test_embeddings  # noqa: PLC0415
    except ImportError as exc:
        sys.exit(f"[ERROR] Cannot import KPR tools: {exc}")

    n_batches = math.ceil(total / batch_size)

    print(f"\n[INFO] Starting KPR embedding pipeline")
    print(f"[INFO] Total crops       : {total}")
    print(f"[INFO] Batch size        : {batch_size}")
    print(f"[INFO] Number of batches : {n_batches}")
    print(f"[INFO] Output path       : {output_path}")
    print()

    results:  list[dict]       = []
    failures: list[CropFailure] = []
    t_start = time.perf_counter()

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end   = min(batch_start + batch_size, total)
        batch_meta  = metadata[batch_start:batch_end]

        # --- validate and preprocess each crop individually --------------
        tensors:    list[torch.Tensor] = []
        valid_meta: list[dict]         = []

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

        # --- KPR forward pass (prompt-optional: images only) -------------
        try:
            batch_tensor = torch.stack(tensors).to(device)   # [B, 3, H, W]
            with torch.no_grad():
                model_output = extractor.model(images=batch_tensor)
                batch_embs, batch_vis, _, _ = extract_test_embeddings(
                    model_output, extractor.cfg.model.kpr.test_embeddings
                )
                # batch_embs: [B, num_slots, D]
                # batch_vis:  [B, num_slots]

        except Exception as exc:
            for rec in valid_meta:
                failures.append(CropFailure(
                    rec["crop_path"], REASON_INFERENCE_ERROR,
                    f"KPR forward pass raised: {exc}",
                ))
                print(f"  [FAIL] Inference error for {rec['crop_path']}: {exc}")
            continue

        # --- validate output shape ---------------------------------------
        if batch_embs.shape[1] != num_slots or batch_embs.shape[2] != part_dim:
            err = (
                f"Expected output [B, {num_slots}, {part_dim}], "
                f"got {list(batch_embs.shape)}"
            )
            print(f"  [ERROR] Batch {batch_idx + 1} shape mismatch: {err}")
            for rec in valid_meta:
                failures.append(CropFailure(
                    rec["crop_path"], REASON_INFERENCE_ERROR,
                    f"Output shape mismatch: {err}",
                ))
            continue

        # --- NaN / Inf guard ---------------------------------------------
        if torch.isnan(batch_embs).any() or torch.isinf(batch_embs).any():
            for rec in valid_meta:
                failures.append(CropFailure(
                    rec["crop_path"], REASON_INFERENCE_ERROR,
                    "NaN/Inf values in KPR output embeddings",
                ))
            print(f"  [ERROR] Batch {batch_idx + 1}: NaN/Inf in embeddings, skipping")
            continue

        # --- parse and serialise per-crop outputs -----------------------
        for i, rec in enumerate(valid_meta):
            try:
                holistic, part_embeddings, part_visibility = _parse_kpr_output(
                    batch_embs, batch_vis, i, holistic_idx
                )
            except Exception as exc:
                failures.append(CropFailure(
                    rec["crop_path"], REASON_INFERENCE_ERROR,
                    f"Output parsing failed: {exc}",
                ))
                print(f"  [FAIL] Parsing error for {rec['crop_path']}: {exc}")
                continue

            results.append({
                "crop_path":            rec["crop_path"],
                "camera_id":            rec["camera_id"],
                "track_id":             rec["track_id"],
                "frame":                rec["frame"],
                "timestamp":            rec["timestamp"],
                "bbox":                 rec["bbox"],
                "detection_confidence": rec["detection_confidence"],
                "holistic_embedding":   holistic,
                "part_embeddings":      part_embeddings,
                "part_visibility":      part_visibility,
            })

        n_failed_this_batch = len(batch_meta) - len(tensors)
        print(
            f"  Batch {batch_idx + 1:>2}/{n_batches}: "
            f"{batch_end}/{total} crops  "
            f"({len(tensors)} embedded, {n_failed_this_batch} skipped this batch)"
        )

    t_elapsed = time.perf_counter() - t_start

    # --- summary ---------------------------------------------------------
    n_success = len(results)
    n_failed  = len(failures)
    n_missing = total - n_success - n_failed

    print()
    print("=" * 60)
    print("TRACE — KPR Part-Based Embedding Pipeline Results")
    print("=" * 60)
    print(f"  Total metadata records : {total}")
    print(f"  Successfully embedded  : {n_success}")
    print(f"  Failed                 : {n_failed}")
    print(f"  Missing / not processed: {n_missing}")
    print(f"  Holistic dim           : {part_dim}")
    print(f"  Part dim               : {part_dim}")
    print(f"  Num parts              : {num_parts}")
    print(f"  Total time             : {t_elapsed:.2f}s")
    if n_success > 0:
        print(f"  Crops/second           : {n_success / t_elapsed:.1f}")
        print(f"  ms/crop                : {1000 * t_elapsed / n_success:.2f}")
    print("=" * 60)

    if n_failed > 0:
        from collections import Counter
        reason_counts = Counter(f.reason for f in failures)
        print("\n  Failures by reason:")
        for reason in [
            REASON_UNREADABLE_IMAGE, REASON_MISSING_FILE,
            REASON_CROP_TOO_SMALL, REASON_INVALID_IMAGE,
            REASON_PREPROCESSING_ERROR, REASON_INFERENCE_ERROR,
        ]:
            count = reason_counts.get(reason, 0)
            if count:
                print(f"    - {reason}: {count}")
        print("\n  Failed crops:")
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
    print(f"[INFO] KPR embeddings written : {output_path}  ({n_success} records)")

    if n_success == total and n_failed == 0:
        print("\n[DONE] KPR embedding complete — all crops embedded successfully.")
    else:
        print("\n[DONE] KPR embedding complete — see failure summary above.")

    return n_failed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "TRACE — KPR Part-Based Embedding Pipeline.\n"
            "Reads Phase 3 crop metadata and writes per-crop KPR embeddings\n"
            "(holistic + per-part embeddings + visibility scores)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--metadata",    required=True,  type=Path,
                        help="Path to crops_metadata_phase3_final.json")
    parser.add_argument("--crop-dir",    required=True,  type=Path,
                        help="Directory containing Phase 3 crop images")
    parser.add_argument("--output",      required=True,  type=Path,
                        help="Destination path for KPR embeddings JSON "
                             "(e.g. dataset/kpr_embeddings_C01.json)")
    parser.add_argument("--kpr-weights", required=True,  type=Path,
                        help="Path to the KPR .pth.tar checkpoint")
    parser.add_argument("--kpr-config",  required=True,  type=Path,
                        help="Path to the KPR yaml config file, "
                             "e.g. configs/kpr/imagenet/kpr_occ_posetrack_test.yaml")
    parser.add_argument("--kpr-root",    type=Path, default=Path.cwd(),
                        help="Root of the cloned KPR repo "
                             "(must contain torchreid/). Default: cwd.")
    parser.add_argument("--batch-size",  type=int, default=None,
                        help="Crops per GPU forward pass "
                             "(default: reid.kpr.batch_size from config.yaml)")
    parser.add_argument("--overwrite",   action="store_true",
                        help="Overwrite an existing output file without prompting")
    parser.add_argument("--config",      type=Path, default=_CONFIG_PATH,
                        help=f"Path to TRACE config.yaml (default: {_CONFIG_PATH})")
    parser.add_argument("--strict",      action="store_true",
                        help="Exit non-zero if any crops failed "
                             "(default: always exit 0 after a completed run)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg     = load_config(args.config)
    kpr_cfg = get_kpr_config(cfg)

    config_batch    = int(kpr_cfg.get("batch_size", 16))
    batch_size      = args.batch_size if args.batch_size is not None else config_batch
    min_w, min_h    = get_min_crop_dims(cfg)

    print("=" * 60)
    print("TRACE — KPR Part-Based Embedding Pipeline (ECCV 2024)")
    print("=" * 60)
    print(f"[INFO] KPR weights       : {args.kpr_weights}")
    print(f"[INFO] KPR config        : {args.kpr_config}")
    print(f"[INFO] KPR root          : {args.kpr_root}")
    print(f"[INFO] Batch size        : {batch_size}")
    print(f"[INFO] Min crop size     : {min_w}×{min_h} px")
    print(f"[INFO] Metadata          : {args.metadata}")
    print(f"[INFO] Crop directory    : {args.crop_dir}")
    print(f"[INFO] Output            : {args.output}")
    print(f"[INFO] Overwrite         : {args.overwrite}")
    print(f"[INFO] Strict mode       : {args.strict}")
    print()

    device = select_device()

    n_failed = run_kpr_embedding_pipeline(
        metadata_path=args.metadata,
        crop_dir=args.crop_dir,
        output_path=args.output,
        batch_size=batch_size,
        kpr_weights=args.kpr_weights,
        kpr_config=args.kpr_config,
        kpr_root=args.kpr_root,
        min_crop_width=min_w,
        min_crop_height=min_h,
        device=device,
        overwrite=args.overwrite,
    )

    if args.strict and n_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
