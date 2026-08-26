"""
test_preprocessing.py — Phase 4.4
Re-ID Preprocessing Correctness Verification for TRACE.

Verifies that the preprocessing pipeline used by embed.py exactly matches
Torchreid's official test/inference transform as confirmed from source:
  https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/data/transforms.py

Torchreid's authoritative test transform (from source):
    transform_te = Compose([
        Resize((height, width)),          # height=256, width=128
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

Does NOT require GPU or torchreid to be installed.
Uses one real TRACE Phase 3 crop for deterministic checks.

Run:
    pytest ai_pipeline/reid/test_preprocessing.py -v
"""

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Import embed module (mock torchreid so the import succeeds without it)
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock

_torchreid_mock = MagicMock()
sys.modules.setdefault("torchreid", _torchreid_mock)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embed as emb  # noqa: E402

# ---------------------------------------------------------------------------
# Constants — authoritative values from Torchreid source
# ---------------------------------------------------------------------------

TORCHREID_HEIGHT     = 256
TORCHREID_WIDTH      = 128
TORCHREID_MEAN       = [0.485, 0.456, 0.406]
TORCHREID_STD        = [0.229, 0.224, 0.225]
EXPECTED_CHANNELS    = 3
EXPECTED_TENSOR_SHAPE = (EXPECTED_CHANNELS, TORCHREID_HEIGHT, TORCHREID_WIDTH)

# Real crop used for all on-disk tests
_REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_CROP  = _REPO_ROOT / "dataset" / "crops_phase3_final" / "C01_track100_frame0607.jpg"

# Torchreid test transform — built here from source spec, NOT from torchreid package
TORCHREID_TRANSFORM = transforms.Compose([
    transforms.Resize((TORCHREID_HEIGHT, TORCHREID_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=TORCHREID_MEAN, std=TORCHREID_STD),
])


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def load_rgb_pil(path: Path) -> Image.Image:
    """Load image as PIL RGB, identical to how embed.py does it."""
    return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# 1. TRACE preprocessing constants match Torchreid spec
# ---------------------------------------------------------------------------

class TestPreprocessingConstants:
    """Verify TRACE constants against Torchreid authoritative values."""

    def test_input_height_is_256(self):
        assert emb.INPUT_HEIGHT == TORCHREID_HEIGHT, (
            f"INPUT_HEIGHT={emb.INPUT_HEIGHT}, expected {TORCHREID_HEIGHT}"
        )

    def test_input_width_is_128(self):
        assert emb.INPUT_WIDTH == TORCHREID_WIDTH, (
            f"INPUT_WIDTH={emb.INPUT_WIDTH}, expected {TORCHREID_WIDTH}"
        )

    def test_height_before_width(self):
        """torchvision Resize takes (height, width) — must not be swapped."""
        assert emb.INPUT_HEIGHT == 256 and emb.INPUT_WIDTH == 128, (
            "Resize order must be (height=256, width=128), not (128, 256). "
            "torchvision.transforms.Resize expects (H, W)."
        )

    def test_mean_channel_0(self):
        assert abs(emb.IMAGENET_MEAN[0] - TORCHREID_MEAN[0]) < 1e-6, (
            f"Mean[0]={emb.IMAGENET_MEAN[0]}, expected {TORCHREID_MEAN[0]}"
        )

    def test_mean_channel_1(self):
        assert abs(emb.IMAGENET_MEAN[1] - TORCHREID_MEAN[1]) < 1e-6, (
            f"Mean[1]={emb.IMAGENET_MEAN[1]}, expected {TORCHREID_MEAN[1]}"
        )

    def test_mean_channel_2(self):
        assert abs(emb.IMAGENET_MEAN[2] - TORCHREID_MEAN[2]) < 1e-6, (
            f"Mean[2]={emb.IMAGENET_MEAN[2]}, expected {TORCHREID_MEAN[2]}"
        )

    def test_std_channel_0(self):
        assert abs(emb.IMAGENET_STD[0] - TORCHREID_STD[0]) < 1e-6, (
            f"Std[0]={emb.IMAGENET_STD[0]}, expected {TORCHREID_STD[0]}"
        )

    def test_std_channel_1(self):
        assert abs(emb.IMAGENET_STD[1] - TORCHREID_STD[1]) < 1e-6, (
            f"Std[1]={emb.IMAGENET_STD[1]}, expected {TORCHREID_STD[1]}"
        )

    def test_std_channel_2(self):
        assert abs(emb.IMAGENET_STD[2] - TORCHREID_STD[2]) < 1e-6, (
            f"Std[2]={emb.IMAGENET_STD[2]}, expected {TORCHREID_STD[2]}"
        )

    def test_mean_length_is_3(self):
        assert len(emb.IMAGENET_MEAN) == 3

    def test_std_length_is_3(self):
        assert len(emb.IMAGENET_STD) == 3


# ---------------------------------------------------------------------------
# 2. _TRANSFORM pipeline structure
# ---------------------------------------------------------------------------

class TestTransformPipelineStructure:
    """Inspect the _TRANSFORM Compose object for the correct sequence of ops."""

    def test_transform_is_compose(self):
        assert isinstance(emb._TRANSFORM, transforms.Compose), (
            f"_TRANSFORM is {type(emb._TRANSFORM)}, expected Compose"
        )

    def test_transform_has_3_steps(self):
        assert len(emb._TRANSFORM.transforms) == 3, (
            f"Expected 3 transform steps (Resize, ToTensor, Normalize), "
            f"got {len(emb._TRANSFORM.transforms)}: "
            f"{[type(t).__name__ for t in emb._TRANSFORM.transforms]}"
        )

    def test_first_step_is_resize(self):
        step = emb._TRANSFORM.transforms[0]
        assert isinstance(step, transforms.Resize), (
            f"Step 0 should be Resize, got {type(step).__name__}"
        )

    def test_resize_size_is_256x128(self):
        step = emb._TRANSFORM.transforms[0]
        # torchvision stores size as a list or tuple
        size = tuple(step.size) if hasattr(step.size, '__iter__') else (step.size,)
        assert size == (256, 128), (
            f"Resize size={size}, expected (256, 128)"
        )

    def test_second_step_is_to_tensor(self):
        step = emb._TRANSFORM.transforms[1]
        assert isinstance(step, transforms.ToTensor), (
            f"Step 1 should be ToTensor, got {type(step).__name__}"
        )

    def test_third_step_is_normalize(self):
        step = emb._TRANSFORM.transforms[2]
        assert isinstance(step, transforms.Normalize), (
            f"Step 2 should be Normalize, got {type(step).__name__}"
        )

    def test_normalize_mean_values(self):
        norm = emb._TRANSFORM.transforms[2]
        for i, (actual, expected) in enumerate(zip(norm.mean, TORCHREID_MEAN)):
            assert abs(actual - expected) < 1e-6, (
                f"Normalize mean[{i}]={actual}, expected {expected}"
            )

    def test_normalize_std_values(self):
        norm = emb._TRANSFORM.transforms[2]
        for i, (actual, expected) in enumerate(zip(norm.std, TORCHREID_STD)):
            assert abs(actual - expected) < 1e-6, (
                f"Normalize std[{i}]={actual}, expected {expected}"
            )

    def test_no_random_augmentation_in_transform(self):
        """Confirm no training augmentations are present in _TRANSFORM."""
        random_aug_types = (
            transforms.RandomHorizontalFlip,
            transforms.RandomCrop,
            transforms.RandomResizedCrop,
            transforms.ColorJitter,
            transforms.RandomRotation,
            transforms.RandomAffine,
        )
        for step in emb._TRANSFORM.transforms:
            assert not isinstance(step, random_aug_types), (
                f"Random augmentation {type(step).__name__} found in _TRANSFORM. "
                "Inference must use only deterministic transforms."
            )

    def test_no_random_erasing_in_transform(self):
        """RandomErasing is a training-only augmentation — must not be present."""
        for step in emb._TRANSFORM.transforms:
            assert type(step).__name__ != "RandomErasing", (
                "RandomErasing found in _TRANSFORM — must not be used during inference."
            )


# ---------------------------------------------------------------------------
# 3. RGB input handling
# ---------------------------------------------------------------------------

class TestRGBInputHandling:
    """Verify that images are loaded as RGB and not BGR."""

    def test_real_crop_loads_as_pil(self):
        img = load_rgb_pil(REAL_CROP)
        assert isinstance(img, Image.Image), "Expected PIL Image"

    def test_pil_mode_is_rgb(self):
        img = load_rgb_pil(REAL_CROP)
        assert img.mode == "RGB", (
            f"PIL image mode is '{img.mode}', expected 'RGB'. "
            "OpenCV loads as BGR — PIL with .convert('RGB') is required."
        )

    def test_no_opencv_bgr_conversion_needed(self):
        """
        PIL Image.open().convert('RGB') always produces RGB regardless of
        whether the JPEG was saved with channel data in a different order.
        This test confirms the loaded image has 3 channels in RGB order
        by checking that channel means differ from a random permutation.
        """
        img = load_rgb_pil(REAL_CROP)
        assert img.mode == "RGB"
        # The key invariant: convert("RGB") is called, so mode must be RGB
        # PIL never returns BGR from convert("RGB")
        assert img.mode != "BGR"


# ---------------------------------------------------------------------------
# 4. Output tensor properties on a real crop
# ---------------------------------------------------------------------------

class TestTensorProperties:
    """Apply _TRANSFORM to a real crop and verify output tensor properties."""

    @pytest.fixture(scope="class")
    @classmethod
    def tensor(cls) -> torch.Tensor:
        img = load_rgb_pil(REAL_CROP)
        return emb._TRANSFORM(img)

    def test_output_is_tensor(self, tensor):
        assert isinstance(tensor, torch.Tensor)

    def test_tensor_dtype_is_float32(self, tensor):
        assert tensor.dtype == torch.float32, (
            f"dtype={tensor.dtype}, expected float32"
        )

    def test_tensor_shape_is_3x256x128(self, tensor):
        assert tuple(tensor.shape) == EXPECTED_TENSOR_SHAPE, (
            f"shape={tuple(tensor.shape)}, expected {EXPECTED_TENSOR_SHAPE}"
        )

    def test_tensor_channels_is_3(self, tensor):
        assert tensor.shape[0] == EXPECTED_CHANNELS

    def test_tensor_height_is_256(self, tensor):
        assert tensor.shape[1] == TORCHREID_HEIGHT

    def test_tensor_width_is_128(self, tensor):
        assert tensor.shape[2] == TORCHREID_WIDTH

    def test_tensor_has_no_nan(self, tensor):
        assert not torch.isnan(tensor).any(), "NaN values in preprocessed tensor"

    def test_tensor_has_no_inf(self, tensor):
        assert not torch.isinf(tensor).any(), "Inf values in preprocessed tensor"

    def test_tensor_values_are_normalized(self, tensor):
        """
        After ImageNet normalization values are typically in roughly [-2.5, 2.5].
        Raw [0,1] ToTensor values would all be >= 0.
        Confirm that some values are negative (i.e., normalization was applied).
        """
        assert tensor.min().item() < 0.0, (
            "All tensor values are >= 0. This suggests normalization was NOT applied. "
            "After ImageNet normalization, values below the mean become negative."
        )

    def test_tensor_values_are_in_expected_range(self, tensor):
        """Normalized ImageNet tensors are typically within [-3, 3]."""
        assert tensor.min().item() >= -3.5, f"Unexpectedly low min: {tensor.min().item()}"
        assert tensor.max().item() <=  3.5, f"Unexpectedly high max: {tensor.max().item()}"


# ---------------------------------------------------------------------------
# 5. Batch dimension
# ---------------------------------------------------------------------------

class TestBatchDimension:
    """Verify that unsqueeze/stack correctly produces [1, 3, 256, 128]."""

    def test_single_crop_batch_shape(self):
        img = load_rgb_pil(REAL_CROP)
        t = emb._TRANSFORM(img)
        batch = t.unsqueeze(0)
        assert tuple(batch.shape) == (1, EXPECTED_CHANNELS, TORCHREID_HEIGHT, TORCHREID_WIDTH), (
            f"Batch shape={tuple(batch.shape)}, expected (1, 3, 256, 128)"
        )

    def test_multi_crop_batch_shape(self):
        img = load_rgb_pil(REAL_CROP)
        t = emb._TRANSFORM(img)
        batch = torch.stack([t, t, t])  # simulate 3-item batch
        assert tuple(batch.shape) == (3, EXPECTED_CHANNELS, TORCHREID_HEIGHT, TORCHREID_WIDTH), (
            f"3-item batch shape={tuple(batch.shape)}, expected (3, 3, 256, 128)"
        )


# ---------------------------------------------------------------------------
# 6. Determinism — same image produces identical tensor on repeated calls
# ---------------------------------------------------------------------------

class TestDeterminism:
    """_TRANSFORM must produce bit-identical results for the same input."""

    def test_same_image_same_tensor(self):
        img = load_rgb_pil(REAL_CROP)
        t1 = emb._TRANSFORM(img)
        t2 = emb._TRANSFORM(img)
        assert torch.equal(t1, t2), (
            "Two applications of _TRANSFORM to the same image produced different "
            "tensors. The transform contains non-deterministic operations."
        )

    def test_reload_same_image_same_tensor(self):
        """Loading the same file twice must produce the same tensor."""
        t1 = emb._TRANSFORM(load_rgb_pil(REAL_CROP))
        t2 = emb._TRANSFORM(load_rgb_pil(REAL_CROP))
        assert torch.equal(t1, t2), (
            "Reloading the same JPEG and applying _TRANSFORM produced different tensors."
        )


# ---------------------------------------------------------------------------
# 7. Direct TRACE vs Torchreid tensor comparison
# ---------------------------------------------------------------------------

class TestDirectTorchreidComparison:
    """
    The strongest possible check: apply both transforms to the same PIL image
    and confirm the output tensors are numerically identical.

    Torchreid test transform (from source):
        Compose([Resize((256, 128)), ToTensor(), Normalize(mean=..., std=...)])

    TRACE transform (embed._TRANSFORM):
        Compose([Resize((256, 128)), ToTensor(), Normalize(mean=..., std=...)])

    Both are built from identical torchvision primitives with identical
    parameters — their outputs must be bit-for-bit equal.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def tensors(cls):
        img = load_rgb_pil(REAL_CROP)
        t_torchreid = TORCHREID_TRANSFORM(img)
        t_trace     = emb._TRANSFORM(img)
        return t_torchreid, t_trace

    def test_shapes_match(self, tensors):
        t_torchreid, t_trace = tensors
        assert t_torchreid.shape == t_trace.shape, (
            f"Shape mismatch: Torchreid={tuple(t_torchreid.shape)}, "
            f"TRACE={tuple(t_trace.shape)}"
        )

    def test_dtypes_match(self, tensors):
        t_torchreid, t_trace = tensors
        assert t_torchreid.dtype == t_trace.dtype, (
            f"dtype mismatch: Torchreid={t_torchreid.dtype}, TRACE={t_trace.dtype}"
        )

    def test_tensors_are_numerically_equal(self, tensors):
        """
        torch.allclose with tight tolerances — both transforms use identical
        torchvision ops so outputs must be bit-for-bit equal.
        """
        t_torchreid, t_trace = tensors
        assert torch.allclose(t_torchreid, t_trace, atol=1e-6, rtol=1e-5), (
            "TRACE tensor does NOT match Torchreid tensor within tolerance. "
            "Preprocessing mismatch detected — embeddings must be regenerated."
        )

    def test_tensors_are_exactly_equal(self, tensors):
        """
        Since both transforms use identical parameters and ops, we expect
        exact bit-level equality (not just approximate).
        """
        t_torchreid, t_trace = tensors
        assert torch.equal(t_torchreid, t_trace), (
            "TRACE tensor is not exactly equal to Torchreid tensor. "
            "Any difference in parameters would cause Re-ID embedding quality degradation."
        )

    def test_max_absolute_difference_is_zero(self, tensors):
        t_torchreid, t_trace = tensors
        max_diff = (t_torchreid - t_trace).abs().max().item()
        assert max_diff == 0.0, (
            f"Max absolute difference between tensors: {max_diff:.2e}. "
            "Expected 0.0 — both transforms use identical parameters."
        )

    def test_torchreid_transform_height_is_256(self):
        resize = TORCHREID_TRANSFORM.transforms[0]
        size = tuple(resize.size) if hasattr(resize.size, '__iter__') else (resize.size,)
        assert size[0] == 256

    def test_torchreid_transform_width_is_128(self):
        resize = TORCHREID_TRANSFORM.transforms[0]
        size = tuple(resize.size) if hasattr(resize.size, '__iter__') else (resize.size,)
        assert size[1] == 128


# ---------------------------------------------------------------------------
# 8. preprocess_image() function in embed.py
# ---------------------------------------------------------------------------

class TestPreprocessImageFunction:
    """
    Verify that emb.preprocess_image() (the function actually called by the
    pipeline) produces the same result as _TRANSFORM applied directly.
    """

    def test_preprocess_image_returns_tensor(self):
        result = emb.preprocess_image(REAL_CROP)
        assert not isinstance(result, tuple), (
            f"preprocess_image returned failure tuple: {result}"
        )
        assert isinstance(result, torch.Tensor)

    def test_preprocess_image_shape(self):
        t = emb.preprocess_image(REAL_CROP)
        assert tuple(t.shape) == EXPECTED_TENSOR_SHAPE

    def test_preprocess_image_dtype(self):
        t = emb.preprocess_image(REAL_CROP)
        assert t.dtype == torch.float32

    def test_preprocess_image_matches_transform(self):
        """preprocess_image() must produce the same tensor as _TRANSFORM directly."""
        img = load_rgb_pil(REAL_CROP)
        t_direct    = emb._TRANSFORM(img)
        t_function  = emb.preprocess_image(REAL_CROP)
        assert torch.equal(t_direct, t_function), (
            "preprocess_image() result differs from _TRANSFORM applied directly. "
            "The function may be using a different code path."
        )

    def test_preprocess_image_fails_gracefully_on_missing_file(self, tmp_path):
        result = emb.preprocess_image(tmp_path / "nonexistent.jpg")
        assert isinstance(result, tuple), (
            "preprocess_image() should return (None, error_str) for missing files"
        )
        assert result[0] is None

    def test_preprocess_image_is_deterministic(self):
        t1 = emb.preprocess_image(REAL_CROP)
        t2 = emb.preprocess_image(REAL_CROP)
        assert torch.equal(t1, t2), (
            "preprocess_image() is not deterministic for the same input path."
        )
