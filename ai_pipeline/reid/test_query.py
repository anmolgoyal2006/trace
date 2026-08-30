"""
Phase 5.1 - Pytest Tests for Query Photos
==========================================
Verifies all query files, metadata, images, transformations, and
anti-leakage flags without requiring a GPU.

Run from the workspace root:
    pytest ai_pipeline/reid/test_query.py -v
"""

import json
import os

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

QUERY_DIR       = os.path.join(BASE, 'dataset', 'query_photos')
METADATA_PATH   = os.path.join(QUERY_DIR, 'query_metadata.json')
CROPS_BASE      = os.path.join(BASE, 'dataset', 'crops_phase3_final')

EXPECTED_FILENAMES = [
    'test_query_1.jpg',
    'test_query_1_rotated.jpg',
    'test_query_1_lighting.jpg',
    'test_query_1_scaled.jpg',
    'test_query_1_blur.jpg',
]

EXPECTED_QUERY_IDS = [os.path.splitext(f)[0] for f in EXPECTED_FILENAMES]

EXPECTED_TRANSFORMATIONS = {
    'original_quality',
    'rotation',
    'brightness_contrast',
    'crop_scale',
    'blur_compression',
}

SOURCE_TRACK_ID  = 29
SOURCE_FRAME     = 270
SOURCE_CAMERA_ID = 'C01'
SOURCE_CROP_REL  = 'dataset/crops_phase3_final/C01_track29_frame0270.jpg'

MIN_WIDTH  = 40
MIN_HEIGHT = 80


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def metadata() -> list:
    """Load and return parsed query_metadata.json."""
    assert os.path.isfile(METADATA_PATH), (
        f'query_metadata.json not found at {METADATA_PATH}'
    )
    with open(METADATA_PATH, encoding='utf-8') as fh:
        data = json.load(fh)
    assert isinstance(data, list), 'Metadata root must be a JSON array'
    return data


@pytest.fixture(scope='module')
def metadata_by_id(metadata) -> dict:
    return {entry['query_id']: entry for entry in metadata}


# ---------------------------------------------------------------------------
# 1. Query directory
# ---------------------------------------------------------------------------

class TestQueryDirectory:
    def test_query_dir_exists(self):
        assert os.path.isdir(QUERY_DIR), (
            f'Query directory does not exist: {QUERY_DIR}'
        )

    def test_query_dir_not_empty(self):
        files = os.listdir(QUERY_DIR)
        assert len(files) > 0, 'Query directory is empty'


# ---------------------------------------------------------------------------
# 2. Query files exist
# ---------------------------------------------------------------------------

class TestQueryFiles:
    @pytest.mark.parametrize('filename', EXPECTED_FILENAMES)
    def test_file_exists(self, filename):
        path = os.path.join(QUERY_DIR, filename)
        assert os.path.isfile(path), f'Query file missing: {path}'

    @pytest.mark.parametrize('filename', EXPECTED_FILENAMES)
    def test_file_not_empty(self, filename):
        path = os.path.join(QUERY_DIR, filename)
        assert os.path.getsize(path) > 0, f'Query file is empty: {filename}'


# ---------------------------------------------------------------------------
# 3. Image readability and RGB conversion
# ---------------------------------------------------------------------------

class TestImageReadability:
    @pytest.mark.parametrize('filename', EXPECTED_FILENAMES)
    def test_image_opens(self, filename):
        path = os.path.join(QUERY_DIR, filename)
        try:
            img = Image.open(path)
            img.verify()       # checks for corruption
        except Exception as exc:
            pytest.fail(f'Cannot open/verify {filename}: {exc}')

    @pytest.mark.parametrize('filename', EXPECTED_FILENAMES)
    def test_rgb_conversion(self, filename):
        path = os.path.join(QUERY_DIR, filename)
        try:
            img = Image.open(path).convert('RGB')
            assert img.mode == 'RGB'
        except Exception as exc:
            pytest.fail(f'RGB conversion failed for {filename}: {exc}')

    @pytest.mark.parametrize('filename', EXPECTED_FILENAMES)
    def test_valid_dimensions(self, filename):
        path = os.path.join(QUERY_DIR, filename)
        img = Image.open(path).convert('RGB')
        w, h = img.size
        assert w >= MIN_WIDTH,  f'{filename}: width {w} < minimum {MIN_WIDTH}'
        assert h >= MIN_HEIGHT, f'{filename}: height {h} < minimum {MIN_HEIGHT}'


# ---------------------------------------------------------------------------
# 4. Metadata structure
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_metadata_file_exists(self):
        assert os.path.isfile(METADATA_PATH)

    def test_metadata_is_list(self, metadata):
        assert isinstance(metadata, list)

    def test_metadata_has_five_entries(self, metadata):
        assert len(metadata) == 5, (
            f'Expected 5 metadata entries, got {len(metadata)}'
        )

    def test_all_expected_query_ids_present(self, metadata):
        found = {e['query_id'] for e in metadata}
        for qid in EXPECTED_QUERY_IDS:
            assert qid in found, f'query_id missing from metadata: {qid}'

    @pytest.mark.parametrize('field', [
        'query_id', 'query_path', 'source_crop_path', 'source_track_id',
        'source_frame', 'source_camera_id', 'transformation',
        'transformation_description', 'original_dimensions', 'query_dimensions',
        'external_validation', 'derived_from_c01_crop',
        'source_crop_excluded_from_gallery',
    ])
    def test_required_fields_present(self, metadata, field):
        for entry in metadata:
            assert field in entry, (
                f'Field "{field}" missing from entry {entry.get("query_id")}'
            )

    def test_query_paths_match_files(self, metadata):
        for entry in metadata:
            rel = entry['query_path']
            full = os.path.join(BASE, rel.replace('/', os.sep))
            assert os.path.isfile(full), (
                f'query_path points to missing file: {rel}'
            )

    def test_dimensions_match_actual_images(self, metadata):
        for entry in metadata:
            rel = entry['query_path']
            full = os.path.join(BASE, rel.replace('/', os.sep))
            img = Image.open(full).convert('RGB')
            w, h = img.size
            stored = entry['query_dimensions']
            assert stored == [w, h], (
                f'{entry["query_id"]}: stored dims {stored} != actual {[w, h]}'
            )


# ---------------------------------------------------------------------------
# 5. Source crop
# ---------------------------------------------------------------------------

class TestSourceCrop:
    def test_source_crop_file_exists(self):
        full = os.path.join(BASE, SOURCE_CROP_REL.replace('/', os.sep))
        assert os.path.isfile(full), f'Source crop missing: {full}'

    def test_source_crop_is_readable(self):
        full = os.path.join(BASE, SOURCE_CROP_REL.replace('/', os.sep))
        img = Image.open(full).convert('RGB')
        w, h = img.size
        assert w >= MIN_WIDTH
        assert h >= MIN_HEIGHT

    def test_metadata_source_crop_path_consistent(self, metadata):
        for entry in metadata:
            assert entry['source_crop_path'] == SOURCE_CROP_REL, (
                f'{entry["query_id"]}: unexpected source_crop_path '
                f'{entry["source_crop_path"]!r}'
            )


# ---------------------------------------------------------------------------
# 6. Source track ID
# ---------------------------------------------------------------------------

class TestSourceTrackId:
    def test_source_track_id_is_integer(self, metadata):
        for entry in metadata:
            tid = entry.get('source_track_id')
            assert isinstance(tid, int), (
                f'{entry["query_id"]}: source_track_id is not int: {tid!r}'
            )

    def test_source_track_id_is_positive(self, metadata):
        for entry in metadata:
            tid = entry['source_track_id']
            assert tid > 0, (
                f'{entry["query_id"]}: source_track_id must be positive, got {tid}'
            )

    def test_source_track_id_matches_expected(self, metadata):
        for entry in metadata:
            assert entry['source_track_id'] == SOURCE_TRACK_ID, (
                f'{entry["query_id"]}: expected track_id {SOURCE_TRACK_ID}, '
                f'got {entry["source_track_id"]}'
            )

    def test_source_frame_matches_expected(self, metadata):
        for entry in metadata:
            assert entry['source_frame'] == SOURCE_FRAME, (
                f'{entry["query_id"]}: expected frame {SOURCE_FRAME}, '
                f'got {entry["source_frame"]}'
            )

    def test_source_camera_id_matches_expected(self, metadata):
        for entry in metadata:
            assert entry['source_camera_id'] == SOURCE_CAMERA_ID, (
                f'{entry["query_id"]}: expected camera {SOURCE_CAMERA_ID!r}, '
                f'got {entry["source_camera_id"]!r}'
            )


# ---------------------------------------------------------------------------
# 7. Transformations
# ---------------------------------------------------------------------------

class TestTransformations:
    def test_all_transformations_present(self, metadata):
        found = {e['transformation'] for e in metadata}
        for t in EXPECTED_TRANSFORMATIONS:
            assert t in found, f'Transformation type missing: {t!r}'

    def test_no_duplicate_transformations(self, metadata):
        transformations = [e['transformation'] for e in metadata]
        assert len(transformations) == len(set(transformations)), (
            'Duplicate transformation types found in metadata'
        )

    def test_each_entry_has_transformation_description(self, metadata):
        for entry in metadata:
            desc = entry.get('transformation_description', '')
            assert isinstance(desc, str) and len(desc) > 10, (
                f'{entry["query_id"]}: transformation_description is empty or too short'
            )

    def test_rotated_variant_differs_from_original(self):
        """Rotated image should differ from the original (not pixel-identical)."""
        orig_path = os.path.join(QUERY_DIR, 'test_query_1.jpg')
        rot_path  = os.path.join(QUERY_DIR, 'test_query_1_rotated.jpg')
        orig = list(Image.open(orig_path).convert('RGB').getdata())
        rot  = list(Image.open(rot_path).convert('RGB').getdata())
        # at least some pixels must differ
        diff_count = sum(1 for a, b in zip(orig, rot) if a != b)
        assert diff_count > 100, (
            'Rotated image appears identical to original — rotation not applied'
        )

    def test_lighting_variant_differs_from_original(self):
        orig_path = os.path.join(QUERY_DIR, 'test_query_1.jpg')
        lit_path  = os.path.join(QUERY_DIR, 'test_query_1_lighting.jpg')
        orig = list(Image.open(orig_path).convert('RGB').getdata())
        lit  = list(Image.open(lit_path).convert('RGB').getdata())
        diff_count = sum(1 for a, b in zip(orig, lit) if a != b)
        assert diff_count > 100, (
            'Lighting image appears identical to original — enhancement not applied'
        )

    def test_scaled_variant_differs_from_original(self):
        orig_path  = os.path.join(QUERY_DIR, 'test_query_1.jpg')
        scale_path = os.path.join(QUERY_DIR, 'test_query_1_scaled.jpg')
        orig  = list(Image.open(orig_path).convert('RGB').getdata())
        scale = list(Image.open(scale_path).convert('RGB').getdata())
        diff_count = sum(1 for a, b in zip(orig, scale) if a != b)
        assert diff_count > 100, (
            'Scaled image appears identical to original — scaling not applied'
        )

    def test_blur_variant_differs_from_original(self):
        orig_path = os.path.join(QUERY_DIR, 'test_query_1.jpg')
        blur_path = os.path.join(QUERY_DIR, 'test_query_1_blur.jpg')
        orig = list(Image.open(orig_path).convert('RGB').getdata())
        blur = list(Image.open(blur_path).convert('RGB').getdata())
        diff_count = sum(1 for a, b in zip(orig, blur) if a != b)
        assert diff_count > 100, (
            'Blur image appears identical to original — blur not applied'
        )


# ---------------------------------------------------------------------------
# 8. Anti-leakage flag
# ---------------------------------------------------------------------------

class TestAntiLeakage:
    def test_external_validation_is_false(self, metadata):
        for entry in metadata:
            assert entry.get('external_validation') is False, (
                f'{entry["query_id"]}: external_validation must be False, '
                f'got {entry.get("external_validation")!r}'
            )

    def test_derived_from_c01_crop_is_true(self, metadata):
        for entry in metadata:
            assert entry.get('derived_from_c01_crop') is True, (
                f'{entry["query_id"]}: derived_from_c01_crop must be True, '
                f'got {entry.get("derived_from_c01_crop")!r}'
            )


# ---------------------------------------------------------------------------
# 9. Source crop exclusion requirement
# ---------------------------------------------------------------------------

class TestSourceCropExclusion:
    def test_source_crop_excluded_flag_is_true(self, metadata):
        for entry in metadata:
            assert entry.get('source_crop_excluded_from_gallery') is True, (
                f'{entry["query_id"]}: source_crop_excluded_from_gallery must be True, '
                f'got {entry.get("source_crop_excluded_from_gallery")!r}'
            )

    def test_query_path_not_same_as_source_crop(self, metadata):
        """Query files must not be the same filesystem path as the source crop."""
        src_full = os.path.abspath(
            os.path.join(BASE, SOURCE_CROP_REL.replace('/', os.sep))
        )
        for entry in metadata:
            query_full = os.path.abspath(
                os.path.join(BASE, entry['query_path'].replace('/', os.sep))
            )
            assert query_full != src_full, (
                f'{entry["query_id"]}: query_path resolves to the source crop itself — '
                'this would cause self-matching during gallery evaluation'
            )

    def test_source_crop_not_in_query_directory(self):
        """The source crop (in crops_phase3_final/) must not physically reside
        inside the query_photos/ directory."""
        src_full = os.path.abspath(
            os.path.join(BASE, SOURCE_CROP_REL.replace('/', os.sep))
        )
        query_dir_abs = os.path.abspath(QUERY_DIR)
        assert not src_full.startswith(query_dir_abs), (
            'Source crop is inside the query_photos/ directory — '
            'it must remain in crops_phase3_final/ and be excluded from gallery'
        )
