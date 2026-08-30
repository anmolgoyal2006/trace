"""
Tests for Phase 5.4 — Similarity to Confidence Scaling

Tests the similarity_to_confidence() function and edge cases.

Run:
    pytest ai_pipeline/reid/test_confidence_scaling.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Add repo root to path for imports
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from ai_pipeline.reid.confidence_scaling import (
    similarity_to_confidence,
    get_observed_range,
    OBSERVED_MIN,
    OBSERVED_MAX,
)


class TestObservedRangeConstants:
    """Test that the observed range constants match Phase 4.5 results."""

    def test_observed_min_matches_phase_4_5(self):
        """OBSERVED_MIN should match Phase 4.5 different-track min."""
        assert OBSERVED_MIN == pytest.approx(0.3315049352393543)

    def test_observed_max_matches_phase_4_5(self):
        """OBSERVED_MAX should match Phase 4.5 same-track max."""
        assert OBSERVED_MAX == pytest.approx(0.9730031552165505)

    def test_get_observed_range(self):
        """get_observed_range() should return the correct tuple."""
        min_val, max_val = get_observed_range()
        assert min_val == pytest.approx(0.3315049352393543)
        assert max_val == pytest.approx(0.9730031552165505)


class TestBasicScaling:
    """Test basic similarity-to-confidence conversion."""

    def test_exact_minimum_returns_zero(self):
        """Similarity at observed_min should return 0.0."""
        result = similarity_to_confidence(OBSERVED_MIN)
        assert result == pytest.approx(0.0)

    def test_exact_maximum_returns_100(self):
        """Similarity at observed_max should return 100.0."""
        result = similarity_to_confidence(OBSERVED_MAX)
        assert result == pytest.approx(100.0)

    def test_midpoint_returns_50(self):
        """Similarity at midpoint should return 50.0."""
        midpoint = (OBSERVED_MIN + OBSERVED_MAX) / 2
        result = similarity_to_confidence(midpoint)
        assert result == pytest.approx(50.0)

    def test_example_0_8148(self):
        """Test with example similarity 0.8148 from request."""
        result = similarity_to_confidence(0.8148)
        # Manual calculation:
        # (0.8148 - 0.3315049352393543) / (0.9730031552165505 - 0.3315049352393543) * 100
        # = 0.4832950647606457 / 0.6414982199771962 * 100
        # = 0.753619... * 100
        # = 75.3619...
        expected = ((0.8148 - OBSERVED_MIN) / (OBSERVED_MAX - OBSERVED_MIN)) * 100
        assert result == pytest.approx(expected)
        # Verify it's in reasonable range
        assert 70 < result < 80

    def test_example_0_8878(self):
        """Test with example similarity 0.8878 from request."""
        result = similarity_to_confidence(0.8878)
        expected = ((0.8878 - OBSERVED_MIN) / (OBSERVED_MAX - OBSERVED_MIN)) * 100
        assert result == pytest.approx(expected)
        # Verify it's in reasonable range (higher than 0.8148)
        assert 80 < result < 90


class TestClampingBelowMinimum:
    """Test clamping behavior for similarities below observed_min."""

    def test_below_minimum_clamps_to_zero(self):
        """Similarity below observed_min should return 0.0."""
        result = similarity_to_confidence(OBSERVED_MIN - 0.1)
        assert result == pytest.approx(0.0)

    def test_far_below_minimum_clamps_to_zero(self):
        """Similarity far below observed_min should return 0.0."""
        result = similarity_to_confidence(-1.0)
        assert result == pytest.approx(0.0)

    def test_negative_similarity_clamps_to_zero(self):
        """Negative cosine similarity should clamp to 0.0."""
        result = similarity_to_confidence(-0.5)
        assert result == pytest.approx(0.0)

    def test_zero_similarity_clamps_to_zero(self):
        """Zero similarity (orthogonal vectors) should clamp to 0.0."""
        result = similarity_to_confidence(0.0)
        assert result == pytest.approx(0.0)


class TestClampingAboveMaximum:
    """Test clamping behavior for similarities above observed_max."""

    def test_above_maximum_clamps_to_100(self):
        """Similarity above observed_max should return 100.0."""
        result = similarity_to_confidence(OBSERVED_MAX + 0.1)
        assert result == pytest.approx(100.0)

    def test_far_above_maximum_clamps_to_100(self):
        """Similarity far above observed_max should return 100.0."""
        result = similarity_to_confidence(2.0)
        assert result == pytest.approx(100.0)

    def test_perfect_similarity_clamps_to_100(self):
        """Similarity of 1.0 (perfect match) should clamp to 100.0."""
        result = similarity_to_confidence(1.0)
        assert result == pytest.approx(100.0)


class TestEdgeCases:
    """Test various edge cases and boundary conditions."""

    def test_just_above_minimum(self):
        """Similarity just above observed_min should return small positive value."""
        result = similarity_to_confidence(OBSERVED_MIN + 0.001)
        assert result > 0.0
        assert result < 1.0

    def test_just_below_maximum(self):
        """Similarity just below observed_max should return value just below 100."""
        result = similarity_to_confidence(OBSERVED_MAX - 0.001)
        assert result > 99.0
        assert result < 100.0

    def test_same_track_mean_similarity(self):
        """Test with Phase 4.5 same-track mean (0.7832829535698886)."""
        same_track_mean = 0.7832829535698886
        result = similarity_to_confidence(same_track_mean)
        expected = ((same_track_mean - OBSERVED_MIN) / (OBSERVED_MAX - OBSERVED_MIN)) * 100
        assert result == pytest.approx(expected)
        # Should be reasonable confidence
        assert 60 < result < 80

    def test_different_track_mean_similarity(self):
        """Test with Phase 4.5 different-track mean (0.6109137615451901)."""
        diff_track_mean = 0.6109137615451901
        result = similarity_to_confidence(diff_track_mean)
        expected = ((diff_track_mean - OBSERVED_MIN) / (OBSERVED_MAX - OBSERVED_MIN)) * 100
        assert result == pytest.approx(expected)
        # Should be lower confidence than same-track mean
        assert 40 < result < 60

    def test_integer_input(self):
        """Function should accept integer similarity values."""
        result = similarity_to_confidence(0)
        assert result == pytest.approx(0.0)

    def test_float_input(self):
        """Function should accept float similarity values."""
        result = similarity_to_confidence(0.5)
        # 0.5 is above observed_min (0.3315), so should produce a positive value
        expected = ((0.5 - OBSERVED_MIN) / (OBSERVED_MAX - OBSERVED_MIN)) * 100
        assert result == pytest.approx(expected)


class TestCustomRange:
    """Test with custom observed_min and observed_max values."""

    def test_custom_range_minimum(self):
        """Custom range minimum should return 0.0."""
        result = similarity_to_confidence(0.0, observed_min=0.0, observed_max=1.0)
        assert result == pytest.approx(0.0)

    def test_custom_range_maximum(self):
        """Custom range maximum should return 100.0."""
        result = similarity_to_confidence(1.0, observed_min=0.0, observed_max=1.0)
        assert result == pytest.approx(100.0)

    def test_custom_range_midpoint(self):
        """Custom range midpoint should return 50.0."""
        result = similarity_to_confidence(0.5, observed_min=0.0, observed_max=1.0)
        assert result == pytest.approx(50.0)

    def test_custom_range_clamping_below(self):
        """Custom range should clamp below minimum to 0.0."""
        result = similarity_to_confidence(-0.5, observed_min=0.0, observed_max=1.0)
        assert result == pytest.approx(0.0)

    def test_custom_range_clamping_above(self):
        """Custom range should clamp above maximum to 100.0."""
        result = similarity_to_confidence(1.5, observed_min=0.0, observed_max=1.0)
        assert result == pytest.approx(100.0)


class TestInvalidRange:
    """Test error handling for invalid range configurations."""

    def test_equal_min_max_raises_error(self):
        """Equal observed_min and observed_max should raise ValueError."""
        with pytest.raises(ValueError, match="observed_max.*equals observed_min"):
            similarity_to_confidence(0.5, observed_min=0.5, observed_max=0.5)

    def test_equal_min_max_error_message(self):
        """Error message should be descriptive."""
        with pytest.raises(ValueError, match="division by zero"):
            similarity_to_confidence(0.5, observed_min=0.5, observed_max=0.5)

    def test_equal_min_max_with_defaults(self):
        """Default constants should not be equal (sanity check)."""
        # This ensures our Phase 4.5 data is valid
        assert OBSERVED_MIN != OBSERVED_MAX


class TestReturnValueType:
    """Test return value types and properties."""

    def test_returns_float(self):
        """Function should return a float."""
        result = similarity_to_confidence(0.5)
        assert isinstance(result, float)

    def test_return_in_range_0_to_100(self):
        """All valid inputs should return values in [0, 100]."""
        test_values = [
            OBSERVED_MIN,
            OBSERVED_MAX,
            (OBSERVED_MIN + OBSERVED_MAX) / 2,
            OBSERVED_MIN - 0.5,
            OBSERVED_MAX + 0.5,
            0.0,
            1.0,
            -1.0,
            2.0,
        ]
        for sim in test_values:
            result = similarity_to_confidence(sim)
            assert 0.0 <= result <= 100.0, f"Failed for similarity={sim}"

    def test_deterministic_results(self):
        """Same input should always produce same output."""
        result1 = similarity_to_confidence(0.8)
        result2 = similarity_to_confidence(0.8)
        assert result1 == result2
