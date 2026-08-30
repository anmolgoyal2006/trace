"""
Phase 5.3.1 — Pytest Tests for track_aggregation.py
=====================================================
All tests use synthetic, controlled similarity records.
No real embeddings, no GPU, no torchreid required.

Run from the workspace root:
    pytest ai_pipeline/reid/test_track_aggregation.py -v
"""

import math
import sys
from pathlib import Path

import pytest

# Ensure the workspace root is on sys.path so that the ai_pipeline package
# resolves correctly regardless of where pytest is invoked from.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from ai_pipeline.reid.track_aggregation import aggregate_by_track


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _approx(a: float, b: float, tol: float = 1e-9) -> bool:
    """Return True if a and b are within tol of each other."""
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# Shared fixture: the canonical example from the spec
# ---------------------------------------------------------------------------

@pytest.fixture
def spec_records():
    return [
        {"track_id": 29, "similarity": 0.79},
        {"track_id": 29, "similarity": 0.70},
        {"track_id": 29, "similarity": 0.85},
        {"track_id": 66, "similarity": 0.81},
        {"track_id": 66, "similarity": 0.75},
    ]


# ---------------------------------------------------------------------------
# 1. One track with several observations
# ---------------------------------------------------------------------------

class TestSingleTrack:
    def test_single_track_returns_one_key(self):
        records = [
            {"track_id": 10, "similarity": 0.60},
            {"track_id": 10, "similarity": 0.70},
            {"track_id": 10, "similarity": 0.80},
        ]
        result = aggregate_by_track(records)
        assert list(result.keys()) == [10]

    def test_single_track_num_observations(self):
        records = [
            {"track_id": 10, "similarity": 0.60},
            {"track_id": 10, "similarity": 0.70},
            {"track_id": 10, "similarity": 0.80},
        ]
        result = aggregate_by_track(records)
        assert result[10]["num_observations"] == 3

    def test_single_track_all_fields_present(self):
        records = [{"track_id": 1, "similarity": 0.5}]
        result = aggregate_by_track(records)
        assert "max_similarity" in result[1]
        assert "mean_similarity" in result[1]
        assert "mean_top3_similarity" in result[1]
        assert "num_observations" in result[1]


# ---------------------------------------------------------------------------
# 2. Multiple tracks remain independent
# ---------------------------------------------------------------------------

class TestMultipleTracks:
    def test_correct_number_of_tracks(self, spec_records):
        result = aggregate_by_track(spec_records)
        assert len(result) == 2

    def test_both_track_ids_present(self, spec_records):
        result = aggregate_by_track(spec_records)
        assert 29 in result
        assert 66 in result

    def test_tracks_are_independent(self, spec_records):
        """Track 66's stats must not be contaminated by track 29's values."""
        result = aggregate_by_track(spec_records)
        # Track 66 has only 0.81 and 0.75 — max must be 0.81, not 0.85
        assert result[66]["max_similarity"] < result[29]["max_similarity"]
        assert _approx(result[66]["max_similarity"], 0.81)

    def test_num_observations_per_track(self, spec_records):
        result = aggregate_by_track(spec_records)
        assert result[29]["num_observations"] == 3
        assert result[66]["num_observations"] == 2


# ---------------------------------------------------------------------------
# 3. Track with exactly 3 observations — top-3 mean == overall mean
# ---------------------------------------------------------------------------

class TestExactlyThreeObservations:
    def test_mean_top3_equals_mean_when_exactly_3(self):
        records = [
            {"track_id": 5, "similarity": 0.60},
            {"track_id": 5, "similarity": 0.70},
            {"track_id": 5, "similarity": 0.80},
        ]
        result = aggregate_by_track(records)
        assert _approx(
            result[5]["mean_top3_similarity"],
            result[5]["mean_similarity"],
        ), (
            "With exactly 3 observations top-3 mean must equal overall mean"
        )

    def test_num_observations_is_3(self):
        records = [
            {"track_id": 5, "similarity": 0.60},
            {"track_id": 5, "similarity": 0.70},
            {"track_id": 5, "similarity": 0.80},
        ]
        result = aggregate_by_track(records)
        assert result[5]["num_observations"] == 3


# ---------------------------------------------------------------------------
# 4. Track with fewer than 3 observations — top-3 uses all available
# ---------------------------------------------------------------------------

class TestFewerThanThreeObservations:
    def test_one_observation_top3_equals_that_value(self):
        records = [{"track_id": 7, "similarity": 0.55}]
        result = aggregate_by_track(records)
        assert _approx(result[7]["mean_top3_similarity"], 0.55)
        assert _approx(result[7]["max_similarity"], 0.55)
        assert _approx(result[7]["mean_similarity"], 0.55)
        assert result[7]["num_observations"] == 1

    def test_two_observations_top3_equals_mean(self):
        records = [
            {"track_id": 8, "similarity": 0.60},
            {"track_id": 8, "similarity": 0.80},
        ]
        result = aggregate_by_track(records)
        expected_mean = (0.60 + 0.80) / 2
        assert _approx(result[8]["mean_top3_similarity"], expected_mean)
        assert result[8]["num_observations"] == 2

    def test_spec_track_66_top3_equals_mean(self, spec_records):
        """Spec example: track 66 has 2 crops so mean_top3 == mean."""
        result = aggregate_by_track(spec_records)
        assert _approx(
            result[66]["mean_top3_similarity"],
            result[66]["mean_similarity"],
        )


# ---------------------------------------------------------------------------
# 5. Track with more than 3 observations — top-3 ignores lower similarities
# ---------------------------------------------------------------------------

class TestMoreThanThreeObservations:
    def test_top3_ignores_lowest_values(self):
        records = [
            {"track_id": 9, "similarity": 0.90},
            {"track_id": 9, "similarity": 0.85},
            {"track_id": 9, "similarity": 0.80},
            {"track_id": 9, "similarity": 0.10},   # outlier low
            {"track_id": 9, "similarity": 0.05},   # outlier low
        ]
        result = aggregate_by_track(records)
        expected_top3 = (0.90 + 0.85 + 0.80) / 3
        assert _approx(result[9]["mean_top3_similarity"], expected_top3), (
            f"Expected {expected_top3}, got {result[9]['mean_top3_similarity']}"
        )

    def test_top3_higher_than_overall_mean_when_low_outliers(self):
        records = [
            {"track_id": 9, "similarity": 0.90},
            {"track_id": 9, "similarity": 0.85},
            {"track_id": 9, "similarity": 0.80},
            {"track_id": 9, "similarity": 0.10},
            {"track_id": 9, "similarity": 0.05},
        ]
        result = aggregate_by_track(records)
        assert result[9]["mean_top3_similarity"] > result[9]["mean_similarity"]

    def test_num_observations_counts_all_not_just_top3(self):
        records = [
            {"track_id": 9, "similarity": 0.90},
            {"track_id": 9, "similarity": 0.85},
            {"track_id": 9, "similarity": 0.80},
            {"track_id": 9, "similarity": 0.10},
            {"track_id": 9, "similarity": 0.05},
        ]
        result = aggregate_by_track(records)
        assert result[9]["num_observations"] == 5


# ---------------------------------------------------------------------------
# 6. Correct max similarity
# ---------------------------------------------------------------------------

class TestMaxSimilarity:
    def test_max_is_highest_value(self):
        records = [
            {"track_id": 1, "similarity": 0.50},
            {"track_id": 1, "similarity": 0.95},
            {"track_id": 1, "similarity": 0.72},
        ]
        result = aggregate_by_track(records)
        assert _approx(result[1]["max_similarity"], 0.95)

    def test_spec_track29_max(self, spec_records):
        result = aggregate_by_track(spec_records)
        assert _approx(result[29]["max_similarity"], 0.85)

    def test_spec_track66_max(self, spec_records):
        result = aggregate_by_track(spec_records)
        assert _approx(result[66]["max_similarity"], 0.81)


# ---------------------------------------------------------------------------
# 7. Correct mean similarity
# ---------------------------------------------------------------------------

class TestMeanSimilarity:
    def test_mean_is_arithmetic_mean(self):
        records = [
            {"track_id": 2, "similarity": 0.60},
            {"track_id": 2, "similarity": 0.80},
            {"track_id": 2, "similarity": 0.70},
        ]
        result = aggregate_by_track(records)
        expected = (0.60 + 0.80 + 0.70) / 3
        assert _approx(result[2]["mean_similarity"], expected)

    def test_spec_track29_mean(self, spec_records):
        result = aggregate_by_track(spec_records)
        expected = (0.79 + 0.70 + 0.85) / 3
        assert _approx(result[29]["mean_similarity"], expected)

    def test_spec_track66_mean(self, spec_records):
        result = aggregate_by_track(spec_records)
        expected = (0.81 + 0.75) / 2
        assert _approx(result[66]["mean_similarity"], expected)


# ---------------------------------------------------------------------------
# 8. Correct top-3 mean
# ---------------------------------------------------------------------------

class TestTop3Mean:
    def test_top3_uses_three_highest(self):
        records = [
            {"track_id": 3, "similarity": 0.50},
            {"track_id": 3, "similarity": 0.90},
            {"track_id": 3, "similarity": 0.70},
            {"track_id": 3, "similarity": 0.85},
        ]
        result = aggregate_by_track(records)
        # top 3: 0.90, 0.85, 0.70 — NOT 0.50
        expected = (0.90 + 0.85 + 0.70) / 3
        assert _approx(result[3]["mean_top3_similarity"], expected)

    def test_spec_track29_top3_equals_mean_because_exactly_3(self, spec_records):
        result = aggregate_by_track(spec_records)
        expected = (0.79 + 0.70 + 0.85) / 3
        assert _approx(result[29]["mean_top3_similarity"], expected)

    def test_spec_track66_top3_equals_mean_because_only_2(self, spec_records):
        result = aggregate_by_track(spec_records)
        expected = (0.81 + 0.75) / 2
        assert _approx(result[66]["mean_top3_similarity"], expected)


# ---------------------------------------------------------------------------
# 9. Correct num_observations
# ---------------------------------------------------------------------------

class TestNumObservations:
    def test_single_record(self):
        result = aggregate_by_track([{"track_id": 1, "similarity": 0.5}])
        assert result[1]["num_observations"] == 1

    def test_multiple_records_same_track(self):
        records = [{"track_id": 1, "similarity": float(i) / 10} for i in range(7)]
        result = aggregate_by_track(records)
        assert result[1]["num_observations"] == 7

    def test_num_obs_is_int(self):
        result = aggregate_by_track([{"track_id": 1, "similarity": 0.5}])
        assert isinstance(result[1]["num_observations"], int)


# ---------------------------------------------------------------------------
# 10. Tracks remain independent
# ---------------------------------------------------------------------------

class TestTrackIndependence:
    def test_adding_track_does_not_affect_other(self):
        base = [
            {"track_id": 10, "similarity": 0.70},
            {"track_id": 10, "similarity": 0.80},
        ]
        extended = base + [
            {"track_id": 99, "similarity": 0.99},
            {"track_id": 99, "similarity": 0.98},
            {"track_id": 99, "similarity": 0.97},
        ]
        r_base = aggregate_by_track(base)
        r_ext  = aggregate_by_track(extended)
        # track 10 stats must be identical in both runs
        assert _approx(r_base[10]["max_similarity"],       r_ext[10]["max_similarity"])
        assert _approx(r_base[10]["mean_similarity"],      r_ext[10]["mean_similarity"])
        assert _approx(r_base[10]["mean_top3_similarity"], r_ext[10]["mean_top3_similarity"])
        assert r_base[10]["num_observations"] == r_ext[10]["num_observations"]

    def test_high_similarity_track_does_not_pollute_low_track(self):
        records = [
            {"track_id": "A", "similarity": 0.99},
            {"track_id": "B", "similarity": 0.10},
        ]
        result = aggregate_by_track(records)
        assert _approx(result["A"]["max_similarity"], 0.99)
        assert _approx(result["B"]["max_similarity"], 0.10)


# ---------------------------------------------------------------------------
# 11. Empty input
# ---------------------------------------------------------------------------

class TestEmptyInput:
    def test_empty_list_returns_empty_dict(self):
        result = aggregate_by_track([])
        assert result == {}

    def test_empty_result_is_dict(self):
        assert isinstance(aggregate_by_track([]), dict)


# ---------------------------------------------------------------------------
# 12. Output contains plain Python numbers (not NumPy scalars)
# ---------------------------------------------------------------------------

class TestOutputTypes:
    def test_max_similarity_is_plain_float(self, spec_records):
        result = aggregate_by_track(spec_records)
        val = result[29]["max_similarity"]
        assert type(val) is float, f"Expected float, got {type(val)}"

    def test_mean_similarity_is_plain_float(self, spec_records):
        result = aggregate_by_track(spec_records)
        val = result[29]["mean_similarity"]
        assert type(val) is float, f"Expected float, got {type(val)}"

    def test_mean_top3_is_plain_float(self, spec_records):
        result = aggregate_by_track(spec_records)
        val = result[29]["mean_top3_similarity"]
        assert type(val) is float, f"Expected float, got {type(val)}"

    def test_num_observations_is_plain_int(self, spec_records):
        result = aggregate_by_track(spec_records)
        val = result[29]["num_observations"]
        assert type(val) is int, f"Expected int, got {type(val)}"

    def test_result_is_json_serialisable(self, spec_records):
        import json
        result = aggregate_by_track(spec_records)
        # Should not raise
        serialised = json.dumps(result)
        assert isinstance(serialised, str)


# ---------------------------------------------------------------------------
# 13. Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_non_list_input_raises_typeerror(self):
        with pytest.raises(TypeError):
            aggregate_by_track({"track_id": 1, "similarity": 0.5})

    def test_missing_track_id_raises_valueerror(self):
        with pytest.raises(ValueError, match="track_id"):
            aggregate_by_track([{"similarity": 0.5}])

    def test_missing_similarity_raises_valueerror(self):
        with pytest.raises(ValueError, match="similarity"):
            aggregate_by_track([{"track_id": 1}])

    def test_extra_fields_are_ignored(self):
        records = [{"track_id": 1, "similarity": 0.75, "extra": "ignored"}]
        result = aggregate_by_track(records)
        assert _approx(result[1]["max_similarity"], 0.75)
