"""
gen_query_variants.py — Phase 5 Query Image Generator
======================================================
Creates five controlled query-image variants from a source crop and
writes them to a target directory with a JSON metadata sidecar.

Transformation parameters are identical to those used in Phase 5.1
for Track 29.  This script is the authoritative generator for all
future per-track query sets.

Transformations applied
-----------------------
  original     : High-quality re-save at JPEG quality=95.
                 No geometric or photometric change.
  rotated      : Clockwise 5-degree rotation, bicubic resampling.
                 Simulates slight camera tilt.
  lighting     : Brightness ×1.15, contrast ×1.10.
                 Simulates different ambient lighting conditions.
  scaled       : Centre-crop to 90 % of original extent, then
                 Lanczos-resized back to original dimensions.
                 Simulates slight zoom / framing difference.
  blur         : Gaussian blur radius=0.8, then saved at JPEG
                 quality=70.  Simulates lower-resolution or heavily
                 compressed query image.

Output per track
----------------
  test_query.jpg
  test_query_rotated.jpg
  test_query_lighting.jpg
  test_query_scaled.jpg
  test_query_blur.jpg
  query_metadata.json

Usage
-----
  # Generate for Track 13:
  python ai_pipeline/reid/gen_query_variants.py \\
      --source  dataset/crops_phase3_final/C01_track13_frame0149.jpg \\
      --track-id 13 \\
      --frame   149 \\
      --output-dir dataset/query_photos/query_track13

  # Generate for Track 16:
  python ai_pipeline/reid/gen_query_variants.py \\
      --source  dataset/crops_phase3_final/C01_track16_frame0073.jpg \\
      --track-id 16 \\
      --frame   73 \\
      --output-dir dataset/query_photos/query_track16

DO NOT MODIFY:
  dataset/query_photos/test_query_1*.jpg  (Track 29 queries — untouched)
  dataset/embeddings_C01.json
  Any Phase 1–4 output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Transformation parameters — identical to Phase 5.1
# ---------------------------------------------------------------------------

JPEG_QUALITY_ORIGINAL = 95    # original variant save quality
JPEG_QUALITY_BLUR     = 70    # blur variant save quality (lossy compression)
ROTATION_DEGREES      = 5     # clockwise
BRIGHTNESS_FACTOR     = 1.15  # +15 %
CONTRAST_FACTOR       = 1.10  # +10 %
CROP_FRACTION         = 0.90  # centre-crop to 90 % of extent
BLUR_RADIUS           = 0.8   # Gaussian blur radius


# ---------------------------------------------------------------------------
# Individual transform functions
# ---------------------------------------------------------------------------

def _apply_original(img: Image.Image, out_path: Path) -> tuple[int, int]:
    """Re-save at JPEG quality=95. No geometric or photometric change."""
    rgb = img.convert("RGB")
    rgb.save(str(out_path), format="JPEG", quality=JPEG_QUALITY_ORIGINAL)
    return rgb.size   # (width, height)


def _apply_rotated(img: Image.Image, out_path: Path) -> tuple[int, int]:
    """Clockwise 5° rotation with bicubic resampling, expand=False so
    output dimensions match input (same as Phase 5.1 behaviour)."""
    rgb = img.convert("RGB")
    # PIL rotate is counter-clockwise; negate for clockwise
    rotated = rgb.rotate(
        -ROTATION_DEGREES,
        resample=Image.BICUBIC,
        expand=False,
    )
    rotated.save(str(out_path), format="JPEG", quality=JPEG_QUALITY_ORIGINAL)
    return rotated.size


def _apply_lighting(img: Image.Image, out_path: Path) -> tuple[int, int]:
    """Brightness +15 %, contrast +10 %."""
    rgb = img.convert("RGB")
    rgb = ImageEnhance.Brightness(rgb).enhance(BRIGHTNESS_FACTOR)
    rgb = ImageEnhance.Contrast(rgb).enhance(CONTRAST_FACTOR)
    rgb.save(str(out_path), format="JPEG", quality=JPEG_QUALITY_ORIGINAL)
    return rgb.size


def _apply_scaled(img: Image.Image, out_path: Path) -> tuple[int, int]:
    """Centre-crop to 90 % of original extent, Lanczos-resize back to
    original dimensions."""
    rgb    = img.convert("RGB")
    orig_w, orig_h = rgb.size

    # Compute crop box — centred, 90 % of each dimension
    crop_w = int(orig_w * CROP_FRACTION)
    crop_h = int(orig_h * CROP_FRACTION)
    left   = (orig_w - crop_w) // 2
    top    = (orig_h - crop_h) // 2
    right  = left + crop_w
    bottom = top + crop_h

    cropped = rgb.crop((left, top, right, bottom))
    resized = cropped.resize((orig_w, orig_h), Image.LANCZOS)
    resized.save(str(out_path), format="JPEG", quality=JPEG_QUALITY_ORIGINAL)
    return resized.size


def _apply_blur(img: Image.Image, out_path: Path) -> tuple[int, int]:
    """Gaussian blur radius=0.8, saved at JPEG quality=70."""
    rgb     = img.convert("RGB")
    blurred = rgb.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
    blurred.save(str(out_path), format="JPEG", quality=JPEG_QUALITY_BLUR)
    return blurred.size


# Ordered list of (variant_id, filename_suffix, apply_fn, transform_name, description)
_VARIANTS: list[tuple] = [
    (
        "original",
        "test_query.jpg",
        _apply_original,
        "original_quality",
        f"High-quality re-save of source crop at JPEG quality={JPEG_QUALITY_ORIGINAL}. "
        "No geometric or photometric change.",
    ),
    (
        "rotated",
        "test_query_rotated.jpg",
        _apply_rotated,
        "rotation",
        f"Clockwise {ROTATION_DEGREES}-degree rotation with bicubic resampling. "
        "Simulates slight camera tilt.",
    ),
    (
        "lighting",
        "test_query_lighting.jpg",
        _apply_lighting,
        "brightness_contrast",
        f"Brightness +{int((BRIGHTNESS_FACTOR-1)*100)}%, "
        f"contrast +{int((CONTRAST_FACTOR-1)*100)}%. "
        "Simulates different ambient lighting conditions.",
    ),
    (
        "scaled",
        "test_query_scaled.jpg",
        _apply_scaled,
        "crop_scale",
        f"Centre-crop to {int(CROP_FRACTION*100)}% of original extent, "
        "resized back to original dimensions with Lanczos. "
        "Simulates slight zoom/framing difference.",
    ),
    (
        "blur",
        "test_query_blur.jpg",
        _apply_blur,
        "blur_compression",
        f"Gaussian blur radius={BLUR_RADIUS} + JPEG quality={JPEG_QUALITY_BLUR}. "
        "Simulates lower-resolution or heavily compressed query image.",
    ),
]


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_query_variants(
    source_path:  Path,
    track_id:     int,
    source_frame: int,
    output_dir:   Path,
    camera_id:    str = "C01",
) -> dict:
    """
    Generate five query variants from *source_path* and write them to
    *output_dir* along with a query_metadata.json sidecar.

    Parameters
    ----------
    source_path:
        Path to the source crop JPEG.
    track_id:
        Track ID the source crop belongs to.
    source_frame:
        Frame number of the source crop.
    output_dir:
        Directory to write the five variant images and metadata into.
        Created automatically if it does not exist.
    camera_id:
        Camera ID string (default "C01").

    Returns
    -------
    dict
        The metadata dict written to query_metadata.json.
    """
    # ── validate source ──────────────────────────────────────────────────
    if not source_path.exists():
        sys.exit(f"[ERROR] Source crop not found: {source_path}")
    try:
        src_img = Image.open(source_path)
        src_img.verify()
    except (UnidentifiedImageError, Exception) as exc:
        sys.exit(f"[ERROR] Cannot read source crop: {exc}")

    src_img   = Image.open(source_path).convert("RGB")
    src_w, src_h = src_img.size
    print(f"  Source crop  : {source_path.name}")
    print(f"  Source dims  : {src_w}×{src_h} px")

    # ── build relative source-crop path (forward slashes) ────────────────
    # Normalise to a path relative to the repo root for the metadata field
    try:
        repo_root  = Path(__file__).resolve().parents[2]
        src_rel    = str(source_path.resolve().relative_to(repo_root)).replace("\\", "/")
    except ValueError:
        # source_path is outside the repo — store absolute
        src_rel = str(source_path).replace("\\", "/")

    # ── create output directory ──────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── apply all five transforms ─────────────────────────────────────────
    metadata_entries: list[dict] = []
    created_files:    list[Path] = []

    for (variant_id, filename, apply_fn, transform_name, description) in _VARIANTS:
        out_path = output_dir / filename
        try:
            dims = apply_fn(src_img, out_path)
        except Exception as exc:
            sys.exit(f"[ERROR] Failed to apply '{variant_id}' transform: {exc}")

        # Verify the written file is a readable RGB JPEG
        try:
            check = Image.open(out_path).convert("RGB")
            check_w, check_h = check.size
            assert check_w > 0 and check_h > 0
        except Exception as exc:
            sys.exit(f"[ERROR] Verification failed for {out_path.name}: {exc}")

        rel_out = str(out_path.relative_to(output_dir.parent.parent
                      if output_dir.parts[-2:] else output_dir)).replace("\\", "/")
        # Store path relative to repo root
        try:
            rel_out = str(out_path.resolve().relative_to(repo_root)).replace("\\", "/")
        except ValueError:
            rel_out = str(out_path).replace("\\", "/")

        entry = {
            "query_id":                    f"track{track_id}_{variant_id}",
            "query_path":                  rel_out,
            "source_crop_path":            src_rel,
            "source_track_id":             track_id,
            "source_frame":                source_frame,
            "source_camera_id":            camera_id,
            "transformation":              transform_name,
            "transformation_description":  description,
            "original_dimensions":         [src_w, src_h],
            "query_dimensions":            list(dims),
            "external_validation":         False,
            "derived_from_c01_crop":       True,
            # The source crop MUST be excluded from the gallery when this
            # query is used for Re-ID evaluation (anti-leakage rule).
            "source_crop_excluded_from_gallery": True,
        }
        metadata_entries.append(entry)
        created_files.append(out_path)

        print(
            f"  [{variant_id:>8}]  {filename:<30}  "
            f"{dims[0]}×{dims[1]} px  ✓"
        )

    # ── write metadata sidecar ───────────────────────────────────────────
    meta_path = output_dir / "query_metadata.json"
    meta_path.write_text(
        json.dumps(metadata_entries, indent=2),
        encoding="utf-8",
    )
    print(f"  Metadata     : {meta_path}")

    return {
        "track_id":       track_id,
        "source_crop":    src_rel,
        "source_frame":   source_frame,
        "source_dims":    [src_w, src_h],
        "output_dir":     str(output_dir),
        "files_created":  [str(p) for p in created_files] + [str(meta_path)],
        "metadata":       metadata_entries,
    }


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate_query_dir(output_dir: Path) -> list[dict]:
    """
    Open every expected variant image in *output_dir*, confirm it is a
    readable RGB JPEG, and return a list of validation result dicts.
    """
    expected = [v[1] for v in _VARIANTS]  # filenames
    results  = []
    for fname in expected:
        p = output_dir / fname
        entry: dict = {"file": fname, "path": str(p)}
        if not p.exists():
            entry.update({"status": "MISSING", "error": "File not found"})
        else:
            try:
                img = Image.open(p).convert("RGB")
                w, h = img.size
                assert w > 0 and h > 0, "Zero dimension"
                entry.update({
                    "status":     "OK",
                    "width_px":   w,
                    "height_px":  h,
                    "size_bytes": p.stat().st_size,
                    "error":      None,
                })
            except Exception as exc:
                entry.update({"status": "FAIL", "error": str(exc)})
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate five Phase-5.1-style query variants from a source crop.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the source crop JPEG.",
    )
    p.add_argument(
        "--track-id",
        type=int,
        required=True,
        help="Track ID the source crop belongs to.",
    )
    p.add_argument(
        "--frame",
        type=int,
        required=True,
        help="Frame number of the source crop.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the five variant images and metadata into.",
    )
    p.add_argument(
        "--camera-id",
        type=str,
        default="C01",
        help="Camera ID string (default: C01).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    print(f"\nGenerating query variants for Track {args.track_id}")
    print(f"  Output dir   : {args.output_dir}")

    result = generate_query_variants(
        source_path=args.source.resolve(),
        track_id=args.track_id,
        source_frame=args.frame,
        output_dir=args.output_dir.resolve(),
        camera_id=args.camera_id,
    )

    # Final validation pass
    print(f"\nValidating output in {args.output_dir} ...")
    val_results = validate_query_dir(args.output_dir.resolve())
    all_ok = all(r["status"] == "OK" for r in val_results)
    for r in val_results:
        status_sym = "✓" if r["status"] == "OK" else "✗"
        if r["status"] == "OK":
            print(
                f"  [{status_sym}] {r['file']:<30}  "
                f"{r['width_px']}×{r['height_px']} px  "
                f"{r['size_bytes']:,} bytes"
            )
        else:
            print(f"  [{status_sym}] {r['file']:<30}  {r['status']}: {r['error']}")

    if not all_ok:
        print("\n[ERROR] One or more variants failed validation.", file=sys.stderr)
        return 1

    print(f"\nAll 5 variants OK for Track {args.track_id}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
