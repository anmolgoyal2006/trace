"""
validate_crop_metadata.py — Phase 3.3
Validate the contract between crops_metadata.json and dataset/crops/*.jpg.

Checks:
  - Metadata file is valid JSON and a list.
  - Every record has exactly the required fields with correct types.
  - detection_confidence is in [0, 1].
  - bbox is a list of exactly 4 numerics.
  - crop_path is relative and points inside the expected crops directory.
  - Every referenced crop file exists and is readable by OpenCV.
  - No duplicate crop_path values in metadata.
  - No orphan images (image with no metadata entry).
  - No orphan metadata entries (metadata with no image).

Exit code 0 — all checks pass.
Exit code 1 — one or more checks fail.

Usage:
    python ai_pipeline/reid/validate_crop_metadata.py \
        --metadata dataset/crops_metadata.json \
        --crops-dir dataset/crops
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

# Required fields and their expected Python types
REQUIRED_FIELDS: dict[str, type | tuple] = {
    "crop_path":            str,
    "camera_id":            str,
    "track_id":             int,
    "frame":                int,
    "timestamp":            str,
    "bbox":                 list,
    "detection_confidence": (int, float),
}


# ---------------------------------------------------------------------------
# Individual record validators
# ---------------------------------------------------------------------------

def validate_record(rec: dict, idx: int, crops_dir: Path, repo_root: Path) -> list[str]:
    """
    Validate a single metadata record.
    Returns a list of failure strings (empty = record is valid).
    """
    errors: list[str] = []
    prefix = f"Record {idx}"

    # --- required fields present ---
    for field in REQUIRED_FIELDS:
        if field not in rec:
            errors.append(f"{prefix}: missing field '{field}'")

    if errors:          # no point checking types if fields are absent
        return errors

    # --- type checks ---
    for field, expected_type in REQUIRED_FIELDS.items():
        val = rec[field]
        if not isinstance(val, expected_type):
            errors.append(
                f"{prefix}: '{field}' expected {expected_type}, "
                f"got {type(val).__name__} (value={val!r})"
            )

    # --- detection_confidence in [0, 1] ---
    conf = rec["detection_confidence"]
    if isinstance(conf, (int, float)) and not (0.0 <= conf <= 1.0):
        errors.append(
            f"{prefix}: detection_confidence out of range [0,1]: {conf}"
        )

    # --- bbox: exactly 4 numeric values ---
    bbox = rec["bbox"]
    if isinstance(bbox, list):
        if len(bbox) != 4:
            errors.append(f"{prefix}: bbox must have 4 elements, got {len(bbox)}")
        elif not all(isinstance(v, (int, float)) for v in bbox):
            errors.append(f"{prefix}: bbox contains non-numeric values: {bbox}")
    # (type mismatch already caught above)

    # --- crop_path: relative, forward-slashes, inside crops_dir ---
    crop_path_str = rec["crop_path"]
    if isinstance(crop_path_str, str):
        # Must not be an absolute path
        if Path(crop_path_str).is_absolute():
            errors.append(
                f"{prefix}: crop_path must be relative, got absolute: {crop_path_str!r}"
            )
        else:
            # Resolve against repo root and confirm it's inside crops_dir
            resolved = (repo_root / crop_path_str).resolve()
            crops_resolved = crops_dir.resolve()
            try:
                resolved.relative_to(crops_resolved)
            except ValueError:
                errors.append(
                    f"{prefix}: crop_path resolves outside crops_dir: {crop_path_str!r}"
                )

    return errors


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def validate(metadata_path: Path, crops_dir: Path, repo_root: Path) -> bool:
    """
    Run all validation checks.
    Returns True if everything passes, False otherwise.
    Prints a structured report to stdout.
    """
    W = 44
    print("=" * W)
    print("TRACE Crop Metadata Validation")
    print("=" * W)
    print(f"Metadata file : {metadata_path}")
    print(f"Crop directory: {crops_dir}")
    print()

    all_pass = True

    def fail(msg: str) -> None:
        nonlocal all_pass
        all_pass = False
        print(f"[FAIL] {msg}")

    # ------------------------------------------------------------------ #
    # 1. Metadata file exists and is valid JSON
    # ------------------------------------------------------------------ #
    if not metadata_path.exists():
        fail(f"Metadata file not found: {metadata_path}")
        print()
        print("=" * W)
        print("RESULT: FAIL")
        print("=" * W)
        return False

    try:
        with open(metadata_path) as f:
            records: list[dict] = json.load(f)
    except json.JSONDecodeError as e:
        fail(f"Invalid JSON in metadata file: {e}")
        print("=" * W)
        print("RESULT: FAIL")
        print("=" * W)
        return False

    if not isinstance(records, list):
        fail("Metadata must be a JSON array (list), got: " + type(records).__name__)
        print("=" * W)
        print("RESULT: FAIL")
        print("=" * W)
        return False

    # ------------------------------------------------------------------ #
    # 2. Crops directory exists
    # ------------------------------------------------------------------ #
    if not crops_dir.exists():
        fail(f"Crops directory not found: {crops_dir}")
        print("=" * W)
        print("RESULT: FAIL")
        print("=" * W)
        return False

    # ------------------------------------------------------------------ #
    # 3. Per-record field/type/range/path checks
    # ------------------------------------------------------------------ #
    record_errors: list[str] = []
    for i, rec in enumerate(records):
        errs = validate_record(rec, i, crops_dir, repo_root)
        record_errors.extend(errs)

    for err in record_errors:
        fail(err)

    invalid_records = len(record_errors)

    # ------------------------------------------------------------------ #
    # 4. Duplicate crop_path values
    # ------------------------------------------------------------------ #
    all_paths = [r["crop_path"] for r in records if "crop_path" in r]
    seen: set[str] = set()
    duplicates: list[str] = []
    for p in all_paths:
        if p in seen:
            duplicates.append(p)
        seen.add(p)
    for dup in duplicates:
        fail(f"Duplicate crop_path: {dup!r}")

    # ------------------------------------------------------------------ #
    # 5. Referenced files exist + are readable
    # ------------------------------------------------------------------ #
    missing_files: list[str] = []
    unreadable_files: list[str] = []

    for i, rec in enumerate(records):
        if "crop_path" not in rec:
            continue
        cp = rec["crop_path"]
        abs_path = (repo_root / cp).resolve()

        if not abs_path.exists():
            missing_files.append(cp)
            fail(f"Metadata record {i}: crop_path does not exist: {cp!r}")
            continue

        img = cv2.imread(str(abs_path))
        if img is None:
            unreadable_files.append(cp)
            fail(f"Metadata record {i}: image unreadable by OpenCV: {cp!r}")

    # ------------------------------------------------------------------ #
    # 6. Actual crop files vs metadata (orphan detection)
    # ------------------------------------------------------------------ #
    actual_files: set[str] = set()
    for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
        for p in crops_dir.glob(ext):
            # Store as relative forward-slash path from repo root
            try:
                rel = p.resolve().relative_to(repo_root.resolve())
                actual_files.add(str(rel).replace("\\", "/"))
            except ValueError:
                actual_files.add(str(p).replace("\\", "/"))

    metadata_paths: set[str] = set()
    for r in records:
        if "crop_path" in r:
            # Normalise: resolve and re-relativise
            abs_p = (repo_root / r["crop_path"]).resolve()
            try:
                rel = abs_p.relative_to(repo_root.resolve())
                metadata_paths.add(str(rel).replace("\\", "/"))
            except ValueError:
                metadata_paths.add(r["crop_path"].replace("\\", "/"))

    orphan_images    = actual_files - metadata_paths
    orphan_metadata  = metadata_paths - actual_files

    for p in sorted(orphan_images):
        fail(f"Orphan image (no metadata entry): {p!r}")
    for p in sorted(orphan_metadata):
        fail(f"Orphan metadata entry (file missing): {p!r}")

    # ------------------------------------------------------------------ #
    # Summary table
    # ------------------------------------------------------------------ #
    print()
    print(f"Metadata records        : {len(records)}")
    print(f"Actual crop images      : {len(actual_files)}")
    print(f"Referenced images found : {len(records) - len(missing_files)}")
    print(f"Missing crop files      : {len(missing_files)}")
    print(f"Orphan crop images      : {len(orphan_images)}")
    print(f"Orphan metadata entries : {len(orphan_metadata)}")
    print(f"Duplicate crop paths    : {len(duplicates)}")
    print(f"Invalid metadata records: {invalid_records}")
    print(f"Unreadable images       : {len(unreadable_files)}")
    print()

    # Check line per logical group
    def check_line(label: str, ok: bool) -> None:
        status = "PASS" if ok else "FAIL"
        print(f"  {label:<28}: {status}")

    check_line("Required fields",      invalid_records == 0)
    check_line("Data types",           invalid_records == 0)
    check_line("Confidence range",     invalid_records == 0)
    check_line("Bounding boxes",       invalid_records == 0)
    check_line("Path safety",          invalid_records == 0 and len(duplicates) == 0)
    check_line("No missing files",     len(missing_files) == 0)
    check_line("No orphan images",     len(orphan_images) == 0)
    check_line("No orphan metadata",   len(orphan_metadata) == 0)
    check_line("No duplicates",        len(duplicates) == 0)
    check_line("All images readable",  len(unreadable_files) == 0)

    print()
    print("=" * W)
    result = "PASS" if all_pass else "FAIL"
    print(f"RESULT: {result}")
    print("=" * W)

    return all_pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRACE — validate crops_metadata.json against dataset/crops/."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("dataset/crops_metadata.json"),
    )
    parser.add_argument(
        "--crops-dir",
        type=Path,
        default=Path("dataset/crops"),
    )
    return parser.parse_args()


def main() -> None:
    args      = parse_args()
    repo_root = Path(__file__).resolve().parents[2]   # Trace/
    passed    = validate(
        metadata_path=args.metadata.resolve() if args.metadata.is_absolute()
                      else (Path.cwd() / args.metadata),
        crops_dir=args.crops_dir.resolve() if args.crops_dir.is_absolute()
                  else (Path.cwd() / args.crops_dir),
        repo_root=repo_root,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
