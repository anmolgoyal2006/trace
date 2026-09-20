"""
face_embed.py — TRACE Face Embedding Pipeline (SCRFD + ArcFace, ONNX Runtime)
==============================================================================

Reads the same Phase 3 crop metadata JSON used by embed.py and produces a
parallel face-embedding file.  For each body crop it:

  1. Runs SCRFD 10G face detection to locate any face in the crop.
  2. If a face is found with confidence >= face_det_threshold:
       - aligns the face patch using the 5-point keypoints returned by SCRFD
       - runs ArcFace ResNet-50 to produce a 512-dim embedding
       - L2-normalises the embedding so cosine similarity == dot product
  3. If NO face is detected, records face_detected=False and embedding=null.

ONNX models (download before first run)
----------------------------------------
SCRFD 10G  — face detection (16.1 MB)
  https://huggingface.co/yakhyo/scrfd/resolve/main/det_10g.onnx

ArcFace ResNet-50  — face recognition (166 MB)
  https://huggingface.co/yakhyo/arcface/resolve/main/w600k_r50.onnx

Quick-download (Linux / macOS / Windows Git-Bash):
  curl -L -o weights/det_10g.onnx   https://huggingface.co/yakhyo/scrfd/resolve/main/det_10g.onnx
  curl -L -o weights/w600k_r50.onnx https://huggingface.co/yakhyo/arcface/resolve/main/w600k_r50.onnx

Install runtime:
  pip install onnxruntime          # CPU
  pip install onnxruntime-gpu      # GPU (CUDA 11+)
  pip install opencv-python-headless scikit-image numpy

Example run:
  python ai_pipeline/reid/face_embed.py \\
      --metadata   dataset/crops_metadata_phase3_final.json \\
      --crop-dir   dataset/crops_phase3_final \\
      --output     dataset/face_embeddings_C01.json \\
      --det-weight weights/det_10g.onnx \\
      --rec-weight weights/w600k_r50.onnx \\
      --face-threshold 0.5 \\
      --overwrite

Output schema (one record per crop):
  {
    "crop_path":       "dataset/crops_phase3_final/C01_track13_frame0042.jpg",
    "camera_id":       "C01",
    "track_id":        13,
    "frame":           42,
    "timestamp":       "10:02:03.400",
    "bbox":            [x1, y1, x2, y2],
    "face_detected":   true,
    "face_confidence": 0.92,      # SCRFD score; null when not detected
    "face_embedding":  [...]      # 512-dim L2-normalised list; null when not detected
  }

Edge cases handled
------------------
  - Missing ONNX weight files        → fatal error with actionable message
  - Unreadable / corrupted crop      → logged, face_detected=False
  - No face found in crop            → face_detected=False, embedding=null
  - Face bounding box too small      → skipped (min_face_size, default 20 px)
  - ONNX inference error (per crop)  → logged, face_detected=False
  - scikit-image not installed       → graceful fallback with alignment warning

Python 3.11 / Windows 11 compatible. No torchreid, no insightface, no C++ builds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]   # .../trace/

# ---------------------------------------------------------------------------
# ArcFace alignment constants (from the face-reidentification reference impl)
# ---------------------------------------------------------------------------

# 5-point reference landmarks for 112×112 aligned output
_ARCFACE_REFERENCE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

_FACE_INPUT_SIZE = (112, 112)   # ArcFace ResNet-50 input
_ARCFACE_MEAN    = 127.5
_ARCFACE_SCALE   = 127.5        # pixel = (pixel - 127.5) / 127.5

_SCRFD_INPUT_SIZE = (640, 640)  # SCRFD 10G input
_SCRFD_MEAN       = 127.5
_SCRFD_STD        = 128.0

_SCRFD_FEAT_STRIDES  = [8, 16, 32]
_SCRFD_NUM_ANCHORS   = 2
_SCRFD_FMC           = 3        # feature map count per stride group
_SCRFD_IOU_THRESH    = 0.4


# ---------------------------------------------------------------------------
# Failure reason codes (mirroring embed.py conventions)
# ---------------------------------------------------------------------------

REASON_MISSING_WEIGHT   = "missing_weight"
REASON_UNREADABLE_IMAGE = "unreadable_image"
REASON_INFERENCE_ERROR  = "inference_error"
REASON_NO_FACE          = "no_face_detected"
REASON_FACE_TOO_SMALL   = "face_too_small"


# ===========================================================================
# Low-level geometry helpers (self-contained — no external utils module)
# ===========================================================================

def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Decode SCRFD distance predictions to [x1, y1, x2, y2] bounding boxes."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Decode SCRFD distance predictions to 5-point facial keypoints."""
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def _nms(dets: np.ndarray, iou_thresh: float) -> list[int]:
    """Greedy NMS. dets shape: (N, 5) with columns [x1, y1, x2, y2, score]."""
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w    = np.maximum(0.0, xx2 - xx1 + 1)
        h    = np.maximum(0.0, yy2 - yy1 + 1)
        ovr  = (w * h) / (areas[i] + areas[order[1:]] - w * h)
        order = order[np.where(ovr <= iou_thresh)[0] + 1]
    return keep


def _estimate_face_transform(landmark: np.ndarray) -> np.ndarray:
    """
    Compute the 2×3 similarity-transform matrix that maps *landmark* to
    the ArcFace 112×112 reference positions.

    Uses scikit-image SimilarityTransform when available.
    Falls back to a simple cv2.estimateAffinePartial2D when not installed.

    Returns a (2, 3) float32 warp matrix suitable for cv2.warpAffine.
    """
    try:
        from skimage.transform import SimilarityTransform  # type: ignore
        tform = SimilarityTransform()
        tform.estimate(landmark, _ARCFACE_REFERENCE)
        return tform.params[0:2, :].astype(np.float32)
    except ImportError:
        # Fallback: OpenCV partial affine (3 DOF: scale, rotation, translation)
        M, _ = cv2.estimateAffinePartial2D(
            landmark.reshape(-1, 1, 2),
            _ARCFACE_REFERENCE.reshape(-1, 1, 2),
        )
        if M is None:
            raise RuntimeError("cv2.estimateAffinePartial2D returned None")
        return M.astype(np.float32)


def _align_face(image_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """
    Align a face patch to the ArcFace 112×112 canonical view.

    Args:
        image_bgr: Full crop image in BGR format.
        kps:       5-point facial keypoints, shape (5, 2).

    Returns:
        Aligned face image, shape (112, 112, 3), BGR.
    """
    M = _estimate_face_transform(kps)
    aligned = cv2.warpAffine(image_bgr, M, _FACE_INPUT_SIZE, borderValue=0.0)
    return aligned


# ===========================================================================
# SCRFD detector (self-contained, ONNX Runtime only)
# ===========================================================================

class SCRFDDetector:
    """
    SCRFD face detector backed by ONNX Runtime.

    Wraps the forward pass, anchor decoding, and NMS into detect(),
    which returns (detections, keypoints) exactly as the reference
    implementation does.
    """

    def __init__(self, model_path: Path, conf_thresh: float = 0.5) -> None:
        """
        Args:
            model_path:   Path to det_10g.onnx.
            conf_thresh:  Minimum detection confidence (default 0.5).

        Raises:
            FileNotFoundError: if model_path does not exist.
            RuntimeError:      if ONNX session fails to load.
        """
        if not model_path.exists():
            raise FileNotFoundError(
                f"SCRFD weight not found: {model_path}\n"
                "Download: curl -L -o weights/det_10g.onnx "
                "https://huggingface.co/yakhyo/scrfd/resolve/main/det_10g.onnx"
            )
        try:
            from onnxruntime import InferenceSession
            self._session = InferenceSession(
                str(model_path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialise SCRFD ONNX session from {model_path}: {exc}"
            ) from exc

        self._input_name   = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]
        self._conf_thresh  = conf_thresh
        self._center_cache: dict[tuple, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _forward(
        self, image: np.ndarray, threshold: float
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """Run one SCRFD forward pass on a pre-resized image."""
        input_h, input_w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0 / _SCRFD_STD,
            size=(input_w, input_h),
            mean=(_SCRFD_MEAN, _SCRFD_MEAN, _SCRFD_MEAN),
            swapRB=True,
        )
        outputs = self._session.run(
            self._output_names, {self._input_name: blob}
        )

        scores_list: list[np.ndarray] = []
        bboxes_list: list[np.ndarray] = []
        kpss_list:   list[np.ndarray] = []

        for idx, stride in enumerate(_SCRFD_FEAT_STRIDES):
            scores    = outputs[idx]                           # (H*W*A, 1)
            bbox_pred = outputs[idx + _SCRFD_FMC] * stride    # (H*W*A, 4)
            kps_pred  = outputs[idx + _SCRFD_FMC * 2] * stride  # (H*W*A, 10)

            height = input_h // stride
            width  = input_w // stride
            key    = (height, width, stride)

            if key not in self._center_cache:
                # Build anchor centre grid
                centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1
                ).astype(np.float32)
                centers = (centers * stride).reshape(-1, 2)
                if _SCRFD_NUM_ANCHORS > 1:
                    centers = np.stack(
                        [centers] * _SCRFD_NUM_ANCHORS, axis=1
                    ).reshape(-1, 2)
                if len(self._center_cache) < 100:
                    self._center_cache[key] = centers
            else:
                centers = self._center_cache[key]

            pos_idx    = np.where(scores >= threshold)[0]
            bboxes     = _distance2bbox(centers, bbox_pred)
            pos_scores = scores[pos_idx]
            pos_bboxes = bboxes[pos_idx]
            kpss       = _distance2kps(centers, kps_pred)
            kpss       = kpss.reshape(kpss.shape[0], -1, 2)
            pos_kpss   = kpss[pos_idx]

            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            kpss_list.append(pos_kpss)

        return scores_list, bboxes_list, kpss_list

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def detect(
        self, image_bgr: np.ndarray
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Detect faces in *image_bgr*.

        Returns:
            detections:  np.ndarray shape (N, 5), columns [x1, y1, x2, y2, score].
                         Empty array when no face detected.
            keypoints:   np.ndarray shape (N, 5, 2) or None.
        """
        w_in, h_in = _SCRFD_INPUT_SIZE
        img_h, img_w = image_bgr.shape[:2]

        # Aspect-preserving resize + letterbox pad
        scale = min(h_in / img_h, w_in / img_w)
        new_h = int(img_h * scale)
        new_w = int(img_w * scale)
        det_scale = float(new_h) / img_h

        resized    = cv2.resize(image_bgr, (new_w, new_h))
        padded     = np.zeros((h_in, w_in, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        scores_list, bboxes_list, kpss_list = self._forward(
            padded, self._conf_thresh
        )

        if not any(len(s) > 0 for s in scores_list):
            return np.empty((0, 5), dtype=np.float32), None

        scores = np.vstack(scores_list).ravel()
        bboxes = np.vstack(bboxes_list) / det_scale
        kpss   = np.vstack(kpss_list)   / det_scale

        order    = scores.argsort()[::-1]
        pre_det  = np.hstack([bboxes, scores[:, None]]).astype(np.float32)[order]
        keep     = _nms(pre_det, _SCRFD_IOU_THRESH)
        det      = pre_det[keep]
        kpss     = kpss[order][keep]

        return det, kpss


# ===========================================================================
# ArcFace recogniser (self-contained, ONNX Runtime only)
# ===========================================================================

class ArcFaceRecogniser:
    """
    ArcFace ResNet-50 face embedding backed by ONNX Runtime.

    Input:  aligned 112×112 BGR face image
    Output: 512-dim float32 embedding (raw model output)
    """

    def __init__(self, model_path: Path) -> None:
        """
        Args:
            model_path: Path to w600k_r50.onnx.

        Raises:
            FileNotFoundError: if model_path does not exist.
            RuntimeError:      if ONNX session fails to load.
        """
        if not model_path.exists():
            raise FileNotFoundError(
                f"ArcFace weight not found: {model_path}\n"
                "Download: curl -L -o weights/w600k_r50.onnx "
                "https://huggingface.co/yakhyo/arcface/resolve/main/w600k_r50.onnx"
            )
        try:
            from onnxruntime import InferenceSession
            self._session = InferenceSession(
                str(model_path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialise ArcFace ONNX session from {model_path}: {exc}"
            ) from exc

        self._input_name   = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]
        self.embedding_dim: int = self._session.get_outputs()[0].shape[1]

    def get_embedding(self, aligned_face_bgr: np.ndarray) -> np.ndarray:
        """
        Produce a raw (non-normalised) 512-dim face embedding.

        Args:
            aligned_face_bgr: 112×112 BGR image, aligned via _align_face().

        Returns:
            np.ndarray shape (512,), float32, NOT L2-normalised.

        Raises:
            RuntimeError: on ONNX inference failure.
        """
        try:
            blob = cv2.dnn.blobFromImage(
                aligned_face_bgr,
                scalefactor=1.0 / _ARCFACE_SCALE,
                size=_FACE_INPUT_SIZE,
                mean=(_ARCFACE_MEAN,) * 3,
                swapRB=True,
            )
            output = self._session.run(self._output_names, {self._input_name: blob})
            return output[0].flatten().astype(np.float32)
        except Exception as exc:
            raise RuntimeError(f"ArcFace inference failed: {exc}") from exc


# ===========================================================================
# Per-crop face processing
# ===========================================================================

def _l2_normalise(vec: np.ndarray) -> np.ndarray:
    """Return vec / ||vec||₂. Returns the zero vector unchanged."""
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def process_crop(
    crop_path: Path,
    rec: dict,
    detector: SCRFDDetector,
    recogniser: ArcFaceRecogniser,
    min_face_size: int,
) -> dict:
    """
    Run face detection + embedding for a single crop image.

    Args:
        crop_path:     Absolute path to the crop JPEG.
        rec:           Metadata record from Phase 3 JSON.
        detector:      Loaded SCRFDDetector instance.
        recogniser:    Loaded ArcFaceRecogniser instance.
        min_face_size: Minimum face bounding-box dimension in pixels.

    Returns:
        A dict matching the output JSON schema (see module docstring).
        face_detected=False and face_embedding=null on any failure.
    """
    # ---- base record (always present in output) --------------------------
    base: dict = {
        "crop_path":       rec["crop_path"],
        "camera_id":       rec["camera_id"],
        "track_id":        rec["track_id"],
        "frame":           rec["frame"],
        "timestamp":       rec["timestamp"],
        "bbox":            rec["bbox"],
        "face_detected":   False,
        "face_confidence": None,
        "face_embedding":  None,
    }

    # ---- read image -------------------------------------------------------
    if not crop_path.exists():
        return {**base, "_skip_reason": REASON_UNREADABLE_IMAGE,
                "_skip_details": f"File not found: {crop_path}"}

    image_bgr = cv2.imread(str(crop_path))
    if image_bgr is None:
        return {**base, "_skip_reason": REASON_UNREADABLE_IMAGE,
                "_skip_details": f"cv2.imread returned None for {crop_path}"}

    # ---- SCRFD detection --------------------------------------------------
    try:
        detections, kpss = detector.detect(image_bgr)
    except Exception as exc:
        return {**base, "_skip_reason": REASON_INFERENCE_ERROR,
                "_skip_details": f"SCRFD inference error: {exc}"}

    if detections.shape[0] == 0 or kpss is None:
        return {**base, "_skip_reason": REASON_NO_FACE,
                "_skip_details": "SCRFD found no faces above threshold"}

    # ---- pick best (highest-confidence) face -----------------------------
    best_idx   = int(np.argmax(detections[:, 4]))
    best_det   = detections[best_idx]            # [x1, y1, x2, y2, score]
    best_kps   = kpss[best_idx]                  # (5, 2)
    face_score = float(best_det[4])

    # ---- minimum face size guard -----------------------------------------
    face_w = float(best_det[2] - best_det[0])
    face_h = float(best_det[3] - best_det[1])
    if face_w < min_face_size or face_h < min_face_size:
        return {**base, "_skip_reason": REASON_FACE_TOO_SMALL,
                "_skip_details": (
                    f"Face bbox {face_w:.1f}×{face_h:.1f} px "
                    f"< min_face_size={min_face_size} px"
                )}

    # ---- face alignment ---------------------------------------------------
    try:
        aligned = _align_face(image_bgr, best_kps)
    except Exception as exc:
        return {**base, "_skip_reason": REASON_INFERENCE_ERROR,
                "_skip_details": f"Face alignment failed: {exc}"}

    # ---- ArcFace embedding -----------------------------------------------
    try:
        raw_embedding = recogniser.get_embedding(aligned)
    except RuntimeError as exc:
        return {**base, "_skip_reason": REASON_INFERENCE_ERROR,
                "_skip_details": str(exc)}

    # ---- L2 normalisation ------------------------------------------------
    normed = _l2_normalise(raw_embedding)

    return {
        "crop_path":       rec["crop_path"],
        "camera_id":       rec["camera_id"],
        "track_id":        rec["track_id"],
        "frame":           rec["frame"],
        "timestamp":       rec["timestamp"],
        "bbox":            rec["bbox"],
        "face_detected":   True,
        "face_confidence": round(face_score, 6),
        "face_embedding":  normed.tolist(),
    }


# ===========================================================================
# Pipeline
# ===========================================================================

def run_face_embedding_pipeline(
    metadata_path: Path,
    crop_dir: Path,
    output_path: Path,
    det_weight: Path,
    rec_weight: Path,
    face_threshold: float,
    min_face_size: int,
    overwrite: bool,
) -> int:
    """
    Full face-embedding pipeline.

    Returns:
        Number of crops where no face was detected (includes all skip reasons).
    """

    # ---- guard: output already exists ------------------------------------
    if output_path.exists() and not overwrite:
        sys.exit(
            f"[ERROR] Output already exists: {output_path}\n"
            "Use --overwrite to replace it."
        )
    if output_path.exists() and overwrite:
        print(f"[WARNING] Overwriting existing file: {output_path}")

    # ---- load metadata ---------------------------------------------------
    if not metadata_path.exists():
        sys.exit(f"[ERROR] Metadata file not found: {metadata_path}")
    with open(metadata_path) as fh:
        records: list[dict] = json.load(fh)
    if not isinstance(records, list):
        sys.exit(f"[ERROR] Expected JSON list in {metadata_path}")
    total = len(records)
    print(f"[INFO] Loaded metadata   : {total} records")

    # ---- verify crop directory -------------------------------------------
    if not crop_dir.exists():
        sys.exit(f"[ERROR] Crop directory not found: {crop_dir}")

    # ---- load models (fail fast if weights missing) ----------------------
    print(f"[INFO] Loading SCRFD detector  : {det_weight}")
    try:
        detector = SCRFDDetector(det_weight, conf_thresh=face_threshold)
    except (FileNotFoundError, RuntimeError) as exc:
        sys.exit(f"[ERROR] {exc}")

    print(f"[INFO] Loading ArcFace recogniser: {rec_weight}")
    try:
        recogniser = ArcFaceRecogniser(rec_weight)
    except (FileNotFoundError, RuntimeError) as exc:
        sys.exit(f"[ERROR] {exc}")

    print(f"[INFO] ArcFace embedding dim : {recogniser.embedding_dim}")
    print(f"[INFO] Face det threshold    : {face_threshold}")
    print(f"[INFO] Min face size         : {min_face_size} px")
    print(f"[INFO] Output path           : {output_path}")
    print()

    # ---- per-crop processing ---------------------------------------------
    results:       list[dict] = []
    skip_reasons:  dict[str, int] = {}
    n_detected   = 0
    t_start      = time.perf_counter()

    for i, rec in enumerate(records):
        crop_path = _REPO_ROOT / rec["crop_path"]
        result    = process_crop(crop_path, rec, detector, recogniser, min_face_size)

        # Strip internal debug keys before writing to output
        skip_reason  = result.pop("_skip_reason",  None)
        skip_details = result.pop("_skip_details", None)

        if result["face_detected"]:
            n_detected += 1
        else:
            reason = skip_reason or REASON_NO_FACE
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            if skip_details:
                print(
                    f"  [SKIP] {rec['crop_path']} "
                    f"— reason={reason} — {skip_details}"
                )

        results.append(result)

        # Progress dot every 50 crops
        if (i + 1) % 50 == 0 or (i + 1) == total:
            elapsed = time.perf_counter() - t_start
            print(
                f"  [{i + 1:>4}/{total}] faces detected so far: {n_detected}"
                f"  ({elapsed:.1f}s elapsed)"
            )

    t_elapsed    = time.perf_counter() - t_start
    n_missed     = total - n_detected
    det_rate_pct = 100.0 * n_detected / total if total > 0 else 0.0

    # ---- summary ---------------------------------------------------------
    print()
    print("=" * 60)
    print("TRACE — Face Embedding Pipeline Results")
    print("=" * 60)
    print(f"  Total crops         : {total}")
    print(f"  Faces detected      : {n_detected}")
    print(f"  Faces missed        : {n_missed}")
    print(f"  Detection rate      : {det_rate_pct:.1f}%")
    print(f"  Total time          : {t_elapsed:.2f}s")
    if n_detected > 0:
        print(f"  Crops/second        : {total / t_elapsed:.1f}")
    if skip_reasons:
        print(f"\n  Missed breakdown:")
        for reason, count in sorted(skip_reasons.items()):
            print(f"    - {reason}: {count}")
    print("=" * 60)

    # ---- write output JSON -----------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[INFO] Face embeddings written: {output_path}  ({total} records)")

    return n_missed


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "TRACE — Face Embedding Pipeline (SCRFD + ArcFace, ONNX Runtime).\n"
            "Reads Phase 3 crop metadata and writes per-crop face embeddings."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        help="Directory containing Phase 3 crop images",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination path for face embeddings JSON "
             "(e.g. dataset/face_embeddings_C01.json)",
    )
    parser.add_argument(
        "--det-weight",
        required=True,
        type=Path,
        help="Path to det_10g.onnx (SCRFD 10G detector)",
    )
    parser.add_argument(
        "--rec-weight",
        required=True,
        type=Path,
        help="Path to w600k_r50.onnx (ArcFace ResNet-50)",
    )
    parser.add_argument(
        "--face-threshold",
        type=float,
        default=0.5,
        help="SCRFD minimum detection confidence (default: 0.5)",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=20,
        help="Minimum face bounding-box dimension in pixels (default: 20)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file without prompting",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print("=" * 60)
    print("TRACE — Face Embedding Pipeline (SCRFD 10G + ArcFace R50)")
    print("=" * 60)
    print(f"[INFO] Metadata          : {args.metadata}")
    print(f"[INFO] Crop directory    : {args.crop_dir}")
    print(f"[INFO] Output            : {args.output}")
    print(f"[INFO] Det weight        : {args.det_weight}")
    print(f"[INFO] Rec weight        : {args.rec_weight}")
    print(f"[INFO] Face threshold    : {args.face_threshold}")
    print(f"[INFO] Min face size     : {args.min_face_size} px")
    print(f"[INFO] Overwrite         : {args.overwrite}")
    print()

    run_face_embedding_pipeline(
        metadata_path=args.metadata,
        crop_dir=args.crop_dir,
        output_path=args.output,
        det_weight=args.det_weight,
        rec_weight=args.rec_weight,
        face_threshold=args.face_threshold,
        min_face_size=args.min_face_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
