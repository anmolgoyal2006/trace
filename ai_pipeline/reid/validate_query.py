"""
Phase 5.1 - Query Validation Script
====================================
Verifies that all query photo files and metadata are correct before
Phase 5 matching experiments are run.

Checks performed:
  - query directory exists
  - every query file listed in metadata exists on disk
  - every image can be opened (not corrupted)
  - RGB conversion succeeds
  - image dimensions are valid (>= 40x80)
  - query_metadata.json exists and is parseable
  - source crop exists on disk
  - source track ID is a positive integer
  - every entry is flagged derived_from_c01_crop = True
  - every entry is flagged source_crop_excluded_from_gallery = True
  - external_validation is correctly set to False

Usage:
    python ai_pipeline/reid/validate_query.py
    (run from the workspace root)
"""

import json
import os
import sys

from PIL import Image

# ---------------------------------------------------------------------------
# Paths  (relative to workspace root)
# ---------------------------------------------------------------------------
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

QUERY_DIR       = os.path.join(BASE, 'dataset', 'query_photos')
METADATA_PATH   = os.path.join(QUERY_DIR, 'query_metadata.json')
CROPS_BASE      = os.path.join(BASE, 'dataset', 'crops_phase3_final')

EXPECTED_QUERY_IDS = [
    'test_query_1',
    'test_query_1_rotated',
    'test_query_1_lighting',
    'test_query_1_scaled',
    'test_query_1_blur',
]

MIN_WIDTH  = 40
MIN_HEIGHT = 80

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fail(msg: str) -> None:
    print(f'  FAIL  {msg}', file=sys.stderr)


def _ok(msg: str) -> None:
    print(f'  OK    {msg}')


def _section(title: str) -> None:
    print(f'\n=== {title} ===')


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_query_dir_exists() -> bool:
    _section('Query directory')
    if os.path.isdir(QUERY_DIR):
        _ok(f'Directory exists: {QUERY_DIR}')
        return True
    _fail(f'Directory missing: {QUERY_DIR}')
    return False


def check_metadata_exists() -> bool:
    _section('Metadata file')
    if os.path.isfile(METADATA_PATH):
        _ok(f'File exists: {METADATA_PATH}')
        return True
    _fail(f'File missing: {METADATA_PATH}')
    return False


def load_metadata() -> list | None:
    try:
        with open(METADATA_PATH, encoding='utf-8') as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            _fail('query_metadata.json root must be a JSON array')
            return None
        _ok(f'Parsed {len(data)} entries from query_metadata.json')
        return data
    except json.JSONDecodeError as exc:
        _fail(f'JSON parse error: {exc}')
        return None


def check_expected_query_ids(metadata: list) -> bool:
    _section('Expected query IDs')
    found_ids = {entry.get('query_id') for entry in metadata}
    all_ok = True
    for qid in EXPECTED_QUERY_IDS:
        if qid in found_ids:
            _ok(f'query_id present: {qid}')
        else:
            _fail(f'query_id missing: {qid}')
            all_ok = False
    return all_ok


def check_query_files(metadata: list) -> bool:
    _section('Query files exist and are readable')
    all_ok = True
    for entry in metadata:
        qid = entry.get('query_id', '<unknown>')
        rel_path = entry.get('query_path', '')
        full_path = os.path.join(BASE, rel_path.replace('/', os.sep))

        # existence
        if not os.path.isfile(full_path):
            _fail(f'[{qid}] file missing: {full_path}')
            all_ok = False
            continue

        # open image
        try:
            img = Image.open(full_path)
        except Exception as exc:
            _fail(f'[{qid}] cannot open image: {exc}')
            all_ok = False
            continue

        # RGB conversion
        try:
            rgb = img.convert('RGB')
        except Exception as exc:
            _fail(f'[{qid}] RGB conversion failed: {exc}')
            all_ok = False
            continue

        w, h = rgb.size

        # dimension check
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            _fail(f'[{qid}] image too small: {w}x{h} (min {MIN_WIDTH}x{MIN_HEIGHT})')
            all_ok = False
            continue

        # metadata dimension consistency
        stored_dims = entry.get('query_dimensions')
        if stored_dims and stored_dims != [w, h]:
            _fail(
                f'[{qid}] dimension mismatch: file={w}x{h}, metadata={stored_dims[0]}x{stored_dims[1]}'
            )
            all_ok = False
            continue

        _ok(f'[{qid}] readable, RGB OK, {w}x{h}')

    return all_ok


def check_source_crop(metadata: list) -> bool:
    _section('Source crop')
    # All entries should point to the same source crop
    source_paths = {entry.get('source_crop_path', '') for entry in metadata}
    all_ok = True
    for rel_path in source_paths:
        full_path = os.path.join(BASE, rel_path.replace('/', os.sep))
        if not os.path.isfile(full_path):
            _fail(f'Source crop missing: {full_path}')
            all_ok = False
        else:
            _ok(f'Source crop exists: {full_path}')

    return all_ok


def check_source_track_id(metadata: list) -> bool:
    _section('Source track ID')
    all_ok = True
    for entry in metadata:
        qid = entry.get('query_id', '<unknown>')
        track_id = entry.get('source_track_id')
        if not isinstance(track_id, int) or track_id <= 0:
            _fail(f'[{qid}] invalid source_track_id: {track_id!r}')
            all_ok = False
        else:
            _ok(f'[{qid}] source_track_id = {track_id}')
    return all_ok


def check_derived_from_c01(metadata: list) -> bool:
    _section('derived_from_c01_crop flag')
    all_ok = True
    for entry in metadata:
        qid = entry.get('query_id', '<unknown>')
        flag = entry.get('derived_from_c01_crop')
        if flag is not True:
            _fail(f'[{qid}] derived_from_c01_crop is {flag!r}, expected True')
            all_ok = False
        else:
            _ok(f'[{qid}] derived_from_c01_crop = True')
    return all_ok


def check_source_crop_excluded(metadata: list) -> bool:
    _section('source_crop_excluded_from_gallery flag (anti-leakage)')
    all_ok = True
    for entry in metadata:
        qid = entry.get('query_id', '<unknown>')
        flag = entry.get('source_crop_excluded_from_gallery')
        if flag is not True:
            _fail(f'[{qid}] source_crop_excluded_from_gallery is {flag!r}, expected True')
            all_ok = False
        else:
            _ok(f'[{qid}] source_crop_excluded_from_gallery = True')
    return all_ok


def check_external_validation(metadata: list) -> bool:
    _section('external_validation flag')
    all_ok = True
    for entry in metadata:
        qid = entry.get('query_id', '<unknown>')
        flag = entry.get('external_validation')
        if flag is not False:
            _fail(f'[{qid}] external_validation is {flag!r}, expected False')
            all_ok = False
        else:
            _ok(f'[{qid}] external_validation = False')
    return all_ok


def check_transformations(metadata: list) -> bool:
    _section('Transformation field completeness')
    valid_transformations = {
        'original_quality',
        'rotation',
        'brightness_contrast',
        'crop_scale',
        'blur_compression',
    }
    all_ok = True
    seen = set()
    for entry in metadata:
        qid = entry.get('query_id', '<unknown>')
        t = entry.get('transformation', '')
        if not t:
            _fail(f'[{qid}] transformation field is empty')
            all_ok = False
        elif t not in valid_transformations:
            _fail(f'[{qid}] unrecognised transformation: {t!r}')
            all_ok = False
        elif t in seen:
            _fail(f'[{qid}] duplicate transformation: {t!r}')
            all_ok = False
        else:
            seen.add(t)
            _ok(f'[{qid}] transformation = {t!r}')
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print('Phase 5.1 Query Validation')
    print('=' * 50)
    print(f'Workspace root : {BASE}')
    print(f'Query directory: {QUERY_DIR}')
    print(f'Metadata file  : {METADATA_PATH}')

    results = []

    results.append(check_query_dir_exists())

    meta_exists = check_metadata_exists()
    results.append(meta_exists)

    if not meta_exists:
        print('\nFATAL: metadata file missing — cannot continue.')
        return 1

    metadata = load_metadata()
    if metadata is None:
        return 1

    results.append(check_expected_query_ids(metadata))
    results.append(check_query_files(metadata))
    results.append(check_source_crop(metadata))
    results.append(check_source_track_id(metadata))
    results.append(check_derived_from_c01(metadata))
    results.append(check_source_crop_excluded(metadata))
    results.append(check_external_validation(metadata))
    results.append(check_transformations(metadata))

    print('\n' + '=' * 50)
    passed = sum(results)
    total  = len(results)
    if all(results):
        print(f'RESULT: PASS  ({passed}/{total} checks passed)')
        return 0
    else:
        failed = total - passed
        print(f'RESULT: FAIL  ({failed}/{total} checks failed)', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
