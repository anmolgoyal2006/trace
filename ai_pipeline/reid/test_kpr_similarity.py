"""
test_kpr_similarity.py — XAI Phase 1a
Characterization tests for ai_pipeline/reid/kpr_similarity.py.

These lock in the CURRENT behaviour of part_aware_similarity() and
aggregate_kpr_by_track() before the XAI layer adds per-part evidence.
Any refactor of kpr_similarity.py must keep every test here green.

These tests:
  - Do NOT require GPU, torchreid, or the KPR model.
  - Use only small synthetic vectors and records.

Run:
    pytest ai_pipeline/reid/test_kpr_similarity.py -v

Coverage:
  PA1  Visibility gating — both sides must be >= vis_threshold (inclusive)
  PA2  Weighting — weight = min(vis_a, vis_b), weighted mean of part cosines
  PA3  Holistic fallback — used only when no part is mutually visible
  PA4  Zero fallback — no mutual parts and no usable holistic → 0.0
  PA5  Per-part errors — skipped silently, never counted in the weight
  PA6  Input validation — part / visibility length mismatches raise
  PA7  Reference parity — matches an independent NumPy implementation
  AG1  Input validation and empty input
  AG2  Per-track statistics (max, mean_top3, mean_visible_parts, counts)
  AG3  Unusable crops — mismatched part counts, holistic fallback, all-unusable
  BD1  Breakdown score parity with part_aware_similarity()
  BD2  Breakdown per-part evidence (status, weight, share, contribution)
  BD3  Breakdown fallback reporting and JSON-safety
  AG4  aggregate_kpr_by_track(include_evidence=True) — best-crop evidence
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from ai_pipeline.reid.kpr_similarity import (
    aggregate_kpr_by_track,
    part_aware_similarity,
    part_aware_similarity_breakdown,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_E0 = [1.0, 0.0]
_E1 = [0.0, 1.0]
_NEG_E0 = [-1.0, 0.0]


def _reference_part_aware(emb_a, vis_a, emb_b, vis_b, holistic_a=None,
                          holistic_b=None, vis_threshold=0.3) -> float:
    """Independent re-statement of the documented algorithm (no shared code)."""
    def cos(x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        nx, ny = np.linalg.norm(x), np.linalg.norm(y)
        if nx == 0.0 or ny == 0.0:
            return 0.0
        return float(np.clip(np.dot(x, y) / (nx * ny), -1.0, 1.0))

    num = den = 0.0
    for ea, va, eb, vb in zip(emb_a, vis_a, emb_b, vis_b):
        if va < vis_threshold or vb < vis_threshold:
            continue
        w = min(va, vb)
        num += w * cos(ea, eb)
        den += w
    if den > 0.0:
        return num / den
    if holistic_a is not None and holistic_b is not None:
        return cos(holistic_a, holistic_b)
    return 0.0


def _record(track_id, part_embeddings, part_visibility, holistic=None) -> dict:
    rec = {
        "track_id":        track_id,
        "part_embeddings": part_embeddings,
        "part_visibility": part_visibility,
    }
    if holistic is not None:
        rec["holistic_embedding"] = holistic
    return rec


# ===========================================================================
# PA1 — Visibility gating
# ===========================================================================

class TestPA1_VisibilityGating:

    def test_part_at_threshold_is_included(self):
        # Part 1 is orthogonal; at exactly 0.3 it is included (>=) → mean drops
        sim = part_aware_similarity(
            emb_a=[_E0, _E0], vis_a=[0.9, 0.3],
            emb_b=[_E0, _E1], vis_b=[0.9, 0.9],
        )
        assert sim == pytest.approx(0.9 / 1.2)

    def test_part_just_below_threshold_is_excluded(self):
        sim = part_aware_similarity(
            emb_a=[_E0, _E0], vis_a=[0.9, 0.29],
            emb_b=[_E0, _E1], vis_b=[0.9, 0.9],
        )
        assert sim == 1.0

    def test_gating_requires_both_sides(self):
        # Part 1 invisible on side B only — still excluded
        sim = part_aware_similarity(
            emb_a=[_E0, _E0], vis_a=[0.9, 0.9],
            emb_b=[_E0, _E1], vis_b=[0.9, 0.1],
        )
        assert sim == 1.0

    def test_custom_vis_threshold(self):
        kwargs = dict(emb_a=[_E0, _E0], vis_a=[0.9, 0.5],
                      emb_b=[_E0, _E1], vis_b=[0.9, 0.9])
        assert part_aware_similarity(**kwargs, vis_threshold=0.6) == 1.0
        assert part_aware_similarity(**kwargs, vis_threshold=0.3) == pytest.approx(0.9 / 1.4)

    def test_occluded_legs_docstring_example(self):
        sim = part_aware_similarity(
            emb_a=[_E0, _E1, _E0, [0, 0], [0, 0]],
            vis_a=[0.9, 0.8, 0.7, 0.1, 0.05],
            emb_b=[_E0, _E1, _E0, _E1, _E1],
            vis_b=[0.8, 0.9, 0.7, 0.8, 0.9],
        )
        assert sim == 1.0


# ===========================================================================
# PA2 — Weighting
# ===========================================================================

class TestPA2_Weighting:

    def test_weight_is_min_visibility(self):
        # part0: cos=1, w=min(0.9,0.6)=0.6 ; part1: cos=0, w=min(0.4,0.8)=0.4
        sim = part_aware_similarity(
            emb_a=[_E0, _E0], vis_a=[0.9, 0.4],
            emb_b=[_E0, _E1], vis_b=[0.6, 0.8],
        )
        assert sim == pytest.approx(0.6)

    def test_negative_part_cosine_pulls_score_down(self):
        # part0: cos=1, w=0.5 ; part1: cos=-1, w=0.5 → 0.0
        sim = part_aware_similarity(
            emb_a=[_E0, _E0], vis_a=[0.5, 0.5],
            emb_b=[_E0, _NEG_E0], vis_b=[0.5, 0.5],
        )
        assert sim == pytest.approx(0.0)

    def test_zero_vector_part_counts_as_zero_similarity(self):
        # A visible zero-vector part yields cosine 0.0 but KEEPS its weight
        sim = part_aware_similarity(
            emb_a=[_E0, [0.0, 0.0]], vis_a=[0.8, 0.8],
            emb_b=[_E0, _E0],        vis_b=[0.8, 0.8],
        )
        assert sim == pytest.approx(0.5)

    def test_holistic_ignored_when_parts_visible(self):
        sim = part_aware_similarity(
            emb_a=[_E0], vis_a=[0.9], emb_b=[_E0], vis_b=[0.9],
            holistic_a=_E0, holistic_b=_NEG_E0,
        )
        assert sim == 1.0

    def test_symmetry(self):
        a = dict(emb=[_E0, [0.6, 0.8], _E1], vis=[0.9, 0.5, 0.35])
        b = dict(emb=[[0.8, 0.6], _E0, _E1], vis=[0.4, 0.95, 0.7])
        ab = part_aware_similarity(a["emb"], a["vis"], b["emb"], b["vis"])
        ba = part_aware_similarity(b["emb"], b["vis"], a["emb"], a["vis"])
        assert ab == pytest.approx(ba, abs=1e-15)

    def test_returns_plain_python_float(self):
        sim = part_aware_similarity([_E0], [0.9], [_E0], [0.9])
        assert type(sim) is float


# ===========================================================================
# PA3 / PA4 — Fallbacks
# ===========================================================================

class TestPA3_HolisticFallback:

    def test_no_mutual_parts_uses_holistic(self):
        sim = part_aware_similarity(
            emb_a=[_E0, _E0], vis_a=[0.9, 0.1],
            emb_b=[_E0, _E0], vis_b=[0.1, 0.9],
            holistic_a=[1.0, 1.0], holistic_b=_E0,
        )
        assert sim == pytest.approx(1.0 / math.sqrt(2.0))

    def test_zero_parts_uses_holistic(self):
        sim = part_aware_similarity([], [], [], [], holistic_a=_E0, holistic_b=_NEG_E0)
        assert sim == pytest.approx(-1.0)

    def test_holistic_length_mismatch_falls_to_zero(self):
        sim = part_aware_similarity(
            [_E0], [0.1], [_E0], [0.1], holistic_a=[1.0, 0.0], holistic_b=[1.0, 0.0, 0.0],
        )
        assert sim == 0.0


class TestPA4_ZeroFallback:

    def test_no_mutual_parts_and_no_holistic(self):
        assert part_aware_similarity([_E0], [0.1], [_E0], [0.9]) == 0.0

    def test_only_one_holistic_present(self):
        assert part_aware_similarity(
            [_E0], [0.1], [_E0], [0.9], holistic_a=_E0, holistic_b=None
        ) == 0.0


# ===========================================================================
# PA5 — Per-part errors are skipped silently
# ===========================================================================

class TestPA5_PartErrors:

    def test_dimension_mismatch_part_is_skipped(self):
        # Part 0 errors (2-D vs 3-D) → skipped, contributes no weight
        sim = part_aware_similarity(
            emb_a=[_E0, _E0],        vis_a=[0.9, 0.5],
            emb_b=[[1.0, 0, 0], _E0], vis_b=[0.9, 0.5],
        )
        assert sim == 1.0

    def test_empty_part_vector_is_skipped(self):
        sim = part_aware_similarity(
            emb_a=[[], _E0], vis_a=[0.9, 0.5],
            emb_b=[[], _E1], vis_b=[0.9, 0.5],
        )
        assert sim == 0.0

    def test_all_parts_error_falls_back_to_holistic(self):
        sim = part_aware_similarity(
            emb_a=[_E0], vis_a=[0.9],
            emb_b=[[1.0, 0, 0]], vis_b=[0.9],
            holistic_a=_E0, holistic_b=_E1,
        )
        assert sim == pytest.approx(0.0)
        sim2 = part_aware_similarity(
            emb_a=[_E0], vis_a=[0.9],
            emb_b=[[1.0, 0, 0]], vis_b=[0.9],
            holistic_a=_E0, holistic_b=_E0,
        )
        assert sim2 == 1.0


# ===========================================================================
# PA6 — Input validation
# ===========================================================================

class TestPA6_Validation:

    def test_part_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="Part count mismatch"):
            part_aware_similarity([_E0, _E0], [0.9, 0.9], [_E0], [0.9])

    def test_vis_a_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="vis_a length"):
            part_aware_similarity([_E0, _E0], [0.9], [_E0, _E0], [0.9, 0.9])

    def test_vis_b_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="vis_b length"):
            part_aware_similarity([_E0, _E0], [0.9, 0.9], [_E0, _E0], [0.9])


# ===========================================================================
# PA7 — Parity with an independent reference implementation
# ===========================================================================

class TestPA7_ReferenceParity:

    @pytest.mark.parametrize("seed", range(20))
    def test_random_five_part_inputs(self, seed):
        rng = random.Random(seed)
        dim = 16

        def emb():
            return [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(5)]

        def vis():
            return [rng.random() for _ in range(5)]

        kwargs = dict(
            emb_a=emb(), vis_a=vis(), emb_b=emb(), vis_b=vis(),
            holistic_a=[rng.gauss(0, 1) for _ in range(dim)],
            holistic_b=[rng.gauss(0, 1) for _ in range(dim)],
        )
        got = part_aware_similarity(**kwargs)
        want = _reference_part_aware(**kwargs)
        assert got == pytest.approx(want, abs=1e-12)
        assert -1.0 <= got <= 1.0


# ===========================================================================
# AG1 — aggregate_kpr_by_track input validation
# ===========================================================================

_QUERY = {"part_embeddings": [_E0, _E1], "part_visibility": [0.9, 0.9]}


class TestAG1_Validation:

    def test_non_list_raises_type_error(self):
        with pytest.raises(TypeError):
            aggregate_kpr_by_track(({"track_id": 1},), _QUERY)  # type: ignore[arg-type]

    @pytest.mark.parametrize("missing", ["track_id", "part_embeddings", "part_visibility"])
    def test_missing_record_field_raises(self, missing):
        rec = _record(1, [_E0, _E1], [0.9, 0.9])
        del rec[missing]
        with pytest.raises(ValueError, match=missing):
            aggregate_kpr_by_track([rec], _QUERY)

    @pytest.mark.parametrize("missing", ["part_embeddings", "part_visibility"])
    def test_query_missing_field_raises_key_error(self, missing):
        q = dict(_QUERY)
        del q[missing]
        with pytest.raises(KeyError):
            aggregate_kpr_by_track([_record(1, [_E0, _E1], [0.9, 0.9])], q)

    def test_empty_gallery_returns_empty_dict(self):
        assert aggregate_kpr_by_track([], _QUERY) == {}


# ===========================================================================
# AG2 — Per-track statistics
# ===========================================================================

class TestAG2_TrackStats:

    def _gallery(self):
        # Track 7 crops with part-aware sims 1.0, 0.5, 0.0, -1.0 vs _QUERY
        return [
            _record(7, [_E0, _E1],      [0.9, 0.9]),        # 1.0, 2 visible
            _record(7, [_E0, _NEG_E0],  [0.9, 0.9]),        # (1+0)/2 = 0.5
            _record(7, [_E1, _E0],      [0.9, 0.9]),        # 0.0
            _record(7, [_NEG_E0, _E1],  [0.9, 0.1]),        # -1.0, 1 visible
            _record("t2", [_E0, _E1],   [0.9, 0.9]),        # str id preserved
        ]

    def test_exact_output_keys(self):
        result = aggregate_kpr_by_track(self._gallery(), _QUERY)
        assert set(result) == {7, "t2"}
        for stats in result.values():
            assert set(stats) == {
                "max_part_aware_sim", "mean_top3_part_aware_sim",
                "mean_visible_parts", "num_observations",
            }

    def test_stats_values(self):
        s = aggregate_kpr_by_track(self._gallery(), _QUERY)[7]
        assert s["max_part_aware_sim"] == pytest.approx(1.0)
        assert s["mean_top3_part_aware_sim"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
        assert s["mean_visible_parts"] == pytest.approx((2 + 2 + 2 + 1) / 4)
        assert s["num_observations"] == 4

    def test_fewer_than_three_crops(self):
        s = aggregate_kpr_by_track(self._gallery(), _QUERY)["t2"]
        assert s["max_part_aware_sim"] == pytest.approx(1.0)
        assert s["mean_top3_part_aware_sim"] == pytest.approx(1.0)
        assert s["num_observations"] == 1

    def test_plain_python_types(self):
        s = aggregate_kpr_by_track(self._gallery(), _QUERY)[7]
        assert type(s["max_part_aware_sim"]) is float
        assert type(s["mean_top3_part_aware_sim"]) is float
        assert type(s["mean_visible_parts"]) is float
        assert type(s["num_observations"]) is int

    def test_max_matches_part_aware_similarity(self):
        gallery = self._gallery()
        s = aggregate_kpr_by_track(gallery, _QUERY)[7]
        direct = max(
            part_aware_similarity(
                _QUERY["part_embeddings"], _QUERY["part_visibility"],
                r["part_embeddings"], r["part_visibility"],
            )
            for r in gallery if r["track_id"] == 7
        )
        assert s["max_part_aware_sim"] == direct

    def test_vis_threshold_passthrough(self):
        gallery = [_record(1, [_E0, _NEG_E0], [0.9, 0.5])]
        default = aggregate_kpr_by_track(gallery, _QUERY)[1]
        strict  = aggregate_kpr_by_track(gallery, _QUERY, vis_threshold=0.6)[1]
        assert default["mean_visible_parts"] == 2.0
        assert strict["mean_visible_parts"] == 1.0
        assert strict["max_part_aware_sim"] == pytest.approx(1.0)
        assert default["max_part_aware_sim"] < strict["max_part_aware_sim"]


# ===========================================================================
# AG3 — Unusable crops
# ===========================================================================

class TestAG3_UnusableCrops:

    def test_part_count_mismatch_uses_holistic(self):
        q = dict(_QUERY, holistic_embedding=_E0)
        gallery = [_record(1, [_E0, _E1, _E0], [0.9, 0.9, 0.9], holistic=[1.0, 1.0])]
        s = aggregate_kpr_by_track(gallery, q)[1]
        assert s["max_part_aware_sim"] == pytest.approx(1.0 / math.sqrt(2.0))
        assert s["num_observations"] == 1

    def test_part_count_mismatch_without_holistic_is_dropped(self):
        # Unusable crop is excluded from sims / num_observations, but its
        # visible-part count is still included in mean_visible_parts.
        gallery = [
            _record(1, [_E0, _E1],      [0.9, 0.9]),         # usable, 2 visible
            _record(1, [_E0, _E1, _E0], [0.9, 0.1, 0.9]),    # unusable, 1 visible
        ]
        s = aggregate_kpr_by_track(gallery, _QUERY)[1]
        assert s["max_part_aware_sim"] == pytest.approx(1.0)
        assert s["num_observations"] == 1
        assert s["mean_visible_parts"] == pytest.approx(1.5)

    def test_all_crops_unusable_returns_zeros(self):
        gallery = [
            _record(3, [_E0], [0.9]),
            _record(3, [_E0], [0.9]),
        ]
        assert aggregate_kpr_by_track(gallery, _QUERY)[3] == {
            "max_part_aware_sim":       0.0,
            "mean_top3_part_aware_sim": 0.0,
            "mean_visible_parts":       0.0,
            "num_observations":         2,
        }


# ===========================================================================
# BD1 — Breakdown score parity
# ===========================================================================

def _random_kwargs(seed: int, dim: int = 16, n_parts: int = 5) -> dict:
    rng = random.Random(seed)
    return dict(
        emb_a=[[rng.gauss(0, 1) for _ in range(dim)] for _ in range(n_parts)],
        vis_a=[rng.random() for _ in range(n_parts)],
        emb_b=[[rng.gauss(0, 1) for _ in range(dim)] for _ in range(n_parts)],
        vis_b=[rng.random() for _ in range(n_parts)],
        holistic_a=[rng.gauss(0, 1) for _ in range(dim)],
        holistic_b=[rng.gauss(0, 1) for _ in range(dim)],
    )


class TestBD1_ScoreParity:

    @pytest.mark.parametrize("seed", range(50))
    def test_score_equals_part_aware_similarity_exactly(self, seed):
        kwargs = _random_kwargs(seed)
        assert part_aware_similarity_breakdown(**kwargs)["score"] == \
            part_aware_similarity(**kwargs)

    def test_validation_errors_unchanged(self):
        with pytest.raises(ValueError, match="Part count mismatch"):
            part_aware_similarity_breakdown([_E0, _E0], [0.9, 0.9], [_E0], [0.9])
        with pytest.raises(ValueError, match="vis_b length"):
            part_aware_similarity_breakdown([_E0], [0.9], [_E0], [])


# ===========================================================================
# BD2 — Per-part evidence
# ===========================================================================

class TestBD2_PartEvidence:

    def _bd(self):
        # part0: used, cos=1,  w=0.6
        # part1: used, cos=0,  w=0.4
        # part2: below threshold on side B
        # part3: mutually visible but errors (dimension mismatch)
        return part_aware_similarity_breakdown(
            emb_a=[_E0, _E0, _E0, _E0],
            vis_a=[0.9, 0.4, 0.9, 0.8],
            emb_b=[_E0, _E1, _E0, [1.0, 0.0, 0.0]],
            vis_b=[0.6, 0.8, 0.1, 0.8],
        )

    def test_statuses(self):
        assert [p["status"] for p in self._bd()["parts"]] == \
            ["used", "used", "below_threshold", "error"]

    def test_mutual_visibility_and_count(self):
        bd = self._bd()
        assert [p["mutually_visible"] for p in bd["parts"]] == [True, True, False, True]
        assert bd["num_mutually_visible"] == 3

    def test_used_part_values(self):
        p0, p1, _, _ = self._bd()["parts"]
        assert p0["cosine"] == pytest.approx(1.0)
        assert p0["weight"] == pytest.approx(0.6)
        assert p0["weight_share"] == pytest.approx(0.6)
        assert p0["contribution"] == pytest.approx(0.6)
        assert p1["cosine"] == pytest.approx(0.0)
        assert p1["weight_share"] == pytest.approx(0.4)
        assert p1["contribution"] == pytest.approx(0.0)

    def test_unused_parts_carry_no_weight(self):
        for p in self._bd()["parts"][2:]:
            assert p["cosine"] is None
            assert p["weight"] == 0.0
            assert p["weight_share"] == 0.0
            assert p["contribution"] == 0.0

    def test_visibilities_echoed(self):
        bd = self._bd()
        assert [p["vis_a"] for p in bd["parts"]] == [0.9, 0.4, 0.9, 0.8]
        assert [p["vis_b"] for p in bd["parts"]] == [0.6, 0.8, 0.1, 0.8]
        assert [p["index"] for p in bd["parts"]] == [0, 1, 2, 3]

    def test_weight_total(self):
        assert self._bd()["weight_total"] == pytest.approx(1.0)

    @pytest.mark.parametrize("seed", range(20))
    def test_contributions_sum_to_score(self, seed):
        bd = part_aware_similarity_breakdown(**_random_kwargs(seed))
        if bd["fallback"] == "none":
            assert sum(p["contribution"] for p in bd["parts"]) == \
                pytest.approx(bd["score"], abs=1e-12)
            assert sum(p["weight_share"] for p in bd["parts"]) == \
                pytest.approx(1.0, abs=1e-12)


# ===========================================================================
# BD3 — Fallback reporting and JSON-safety
# ===========================================================================

class TestBD3_Fallbacks:

    def test_fallback_none(self):
        bd = part_aware_similarity_breakdown([_E0], [0.9], [_E0], [0.9])
        assert bd["fallback"] == "none"

    def test_fallback_holistic(self):
        bd = part_aware_similarity_breakdown(
            [_E0], [0.1], [_E0], [0.9], holistic_a=[1.0, 1.0], holistic_b=_E0,
        )
        assert bd["fallback"] == "holistic"
        assert bd["score"] == pytest.approx(1.0 / math.sqrt(2.0))
        assert bd["weight_total"] == 0.0
        assert all(p["contribution"] == 0.0 for p in bd["parts"])

    def test_fallback_holistic_when_all_parts_error(self):
        bd = part_aware_similarity_breakdown(
            [_E0], [0.9], [[1.0, 0, 0]], [0.9], holistic_a=_E0, holistic_b=_E0,
        )
        assert bd["fallback"] == "holistic"
        assert bd["num_mutually_visible"] == 1
        assert bd["parts"][0]["status"] == "error"

    def test_fallback_zero(self):
        bd = part_aware_similarity_breakdown([_E0], [0.1], [_E0], [0.9])
        assert bd["fallback"] == "zero"
        assert bd["score"] == 0.0

    def test_fallback_zero_on_holistic_error(self):
        bd = part_aware_similarity_breakdown(
            [_E0], [0.1], [_E0], [0.1], holistic_a=[1.0, 0.0], holistic_b=[1.0, 0.0, 0.0],
        )
        assert bd["fallback"] == "zero"

    def test_json_serialisable_plain_types(self):
        import json
        bd = part_aware_similarity_breakdown(
            **_random_kwargs(0) | dict(
                vis_a=np.array([0.9, 0.2, 0.5, 0.7, 0.4]),
                vis_b=np.array([0.8, 0.9, 0.1, 0.6, 0.35]),
            )
        )
        assert json.loads(json.dumps(bd)) == bd
        assert type(bd["score"]) is float
        for p in bd["parts"]:
            assert type(p["vis_a"]) is float and type(p["weight"]) is float


# ===========================================================================
# AG4 — aggregate_kpr_by_track(include_evidence=True)
# ===========================================================================

_STAT_KEYS = {
    "max_part_aware_sim", "mean_top3_part_aware_sim",
    "mean_visible_parts", "num_observations",
}


class TestAG4_TrackEvidence:

    def _gallery(self):
        return [
            _record(7, [_E0, _NEG_E0], [0.9, 0.9]) | {"crop_path": "c7_f0.jpg", "frame": 0},
            _record(7, [_E0, _E1],     [0.9, 0.9]) | {"crop_path": "c7_f5.jpg", "frame": 5,
                                                      "timestamp": "00:00:05"},
            _record(7, [_E1, _E0],     [0.9, 0.9]) | {"crop_path": "c7_f9.jpg"},
            _record(8, [_E0],          [0.9]),                       # unusable
        ]

    def test_default_output_has_no_evidence_key(self):
        for stats in aggregate_kpr_by_track(self._gallery(), _QUERY).values():
            assert set(stats) == _STAT_KEYS

    def test_stats_identical_with_and_without_evidence(self):
        plain = aggregate_kpr_by_track(self._gallery(), _QUERY)
        rich  = aggregate_kpr_by_track(self._gallery(), _QUERY, include_evidence=True)
        assert set(plain) == set(rich)
        for tid in plain:
            assert set(rich[tid]) == _STAT_KEYS | {"best_crop_evidence"}
            assert {k: rich[tid][k] for k in _STAT_KEYS} == plain[tid]

    @pytest.mark.parametrize("seed", range(20))
    def test_stats_identical_random(self, seed):
        rng = random.Random(seed)
        e = lambda: [[rng.gauss(0, 1) for _ in range(8)] for _ in range(5)]
        v = lambda: [rng.random() for _ in range(5)]
        gallery = [_record(rng.randint(0, 2), e(), v(), [rng.gauss(0, 1) for _ in range(8)])
                   for _ in range(8)]
        q = {"part_embeddings": e(), "part_visibility": v()}
        plain = aggregate_kpr_by_track(gallery, q)
        rich  = aggregate_kpr_by_track(gallery, q, include_evidence=True)
        for tid in plain:
            assert {k: rich[tid][k] for k in _STAT_KEYS} == plain[tid]
            assert rich[tid]["best_crop_evidence"]["score"] == plain[tid]["max_part_aware_sim"]

    def test_evidence_is_best_crop(self):
        ev = aggregate_kpr_by_track(self._gallery(), _QUERY, include_evidence=True)[7]["best_crop_evidence"]
        assert ev["crop_path"] == "c7_f5.jpg"
        assert ev["frame"] == 5
        assert ev["timestamp"] == "00:00:05"
        assert ev["score"] == pytest.approx(1.0)
        assert ev["fallback"] == "none"
        assert ev["part_count_mismatch"] is False
        assert [p["status"] for p in ev["parts"]] == ["used", "used"]

    def test_missing_crop_metadata_is_none(self):
        gallery = [_record(1, [_E0, _E1], [0.9, 0.9])]
        ev = aggregate_kpr_by_track(gallery, _QUERY, include_evidence=True)[1]["best_crop_evidence"]
        assert ev["crop_path"] is None and ev["frame"] is None and ev["timestamp"] is None

    def test_tie_picks_first_crop(self):
        gallery = [
            _record(1, [_E0, _E1], [0.9, 0.9]) | {"crop_path": "first.jpg"},
            _record(1, [_E0, _E1], [0.9, 0.9]) | {"crop_path": "second.jpg"},
        ]
        ev = aggregate_kpr_by_track(gallery, _QUERY, include_evidence=True)[1]["best_crop_evidence"]
        assert ev["crop_path"] == "first.jpg"

    def test_all_unusable_track_has_none_evidence(self):
        rich = aggregate_kpr_by_track(self._gallery(), _QUERY, include_evidence=True)
        assert rich[8]["best_crop_evidence"] is None

    def test_part_count_mismatch_holistic_evidence(self):
        q = dict(_QUERY, holistic_embedding=_E0)
        gallery = [_record(1, [_E0, _E1, _E0], [0.9, 0.9, 0.9], holistic=[1.0, 1.0])
                   | {"crop_path": "odd.jpg"}]
        ev = aggregate_kpr_by_track(gallery, q, include_evidence=True)[1]["best_crop_evidence"]
        assert ev["part_count_mismatch"] is True
        assert ev["fallback"] == "holistic"
        assert ev["parts"] == []
        assert ev["score"] == pytest.approx(1.0 / math.sqrt(2.0))
        assert ev["crop_path"] == "odd.jpg"

    def test_evidence_json_serialisable(self):
        import json
        rich = aggregate_kpr_by_track(self._gallery(), _QUERY, include_evidence=True)
        assert json.loads(json.dumps(rich[7])) == rich[7]
