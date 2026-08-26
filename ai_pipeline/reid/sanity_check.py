"""
sanity_check.py — Phase 4.5
TRACE Re-ID Sanity Check: Same-Track vs Different-Track Cosine Similarity.

Reads the pre-computed 512-dimensional OSNet embeddings from
dataset/embeddings_C01.json and measures whether same-track crops have
higher cosine similarity than different-track crops.

This script:
  - Does NOT regenerate embeddings.
  - Does NOT load torchreid or OSNet.
  - Does NOT require a GPU.
  - Only performs mathematical analysis on the existing embedding vectors.

This is a SANITY CHECK, not a ground-truth Re-ID accuracy evaluation.
Track IDs come from the Phase 2 ByteTrack tracker:
  same track_id  = same tracked trajectory in C01
  different track_id = different tracked trajectory in C01
This is NOT ground-truth identity annotation.

Usage:
    python ai_pipeline/reid/sanity_check.py \\
        --embeddings dataset/embeddings_C01.json \\
        --output-dir dataset/reid_sanity

Optional flags:
    --max-pairs-per-track INT   Max same-track pairs per track (default: 10)
    --seed INT                  Random seed (default: 42)
    --no-plot                   Skip generating the distribution PNG

Outputs:
    dataset/reid_sanity/similarity_report.json
    dataset/reid_sanity/similarity_distribution.png  (unless --no-plot)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Repository root and path setup
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]   # .../Trace/
sys.path.insert(0, str(_REPO_ROOT))

from ai_pipeline.reid.similarity import (
    cosine_similarity,
    compute_stats,
    threshold_fractions,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
EXPECTED_DIM = 512
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90]
MAX_PAIRS_PER_TRACK = 10  # controlled upper bound per track for same-track pairs


# ---------------------------------------------------------------------------
# 1. Load and validate embeddings
# ---------------------------------------------------------------------------

def load_embeddings(path: Path) -> list[dict]:
    """
    Load embeddings_C01.json and return the validated record list.
    Prints a summary and exits on critical errors.
    """
    if not path.exists():
        sys.exit(
            f"[ERROR] Embeddings file not found: {path}\n"
            "This file currently exists in Google Colab / Google Drive.\n"
            "Run this script there against the actual dataset."
        )

    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        sys.exit(f"[ERROR] Expected a JSON list, got {type(data).__name__}")

    print(f"[INFO] Loaded {len(data)} records from {path}")
    return data


def validate_records(records: list[dict]) -> list[dict]:
    """
    Validate every record; remove invalid ones with a warning rather than
    crashing, so the experiment degrades gracefully on partial datasets.

    Checks per record:
      - 'embedding' field exists and is a list
      - length == EXPECTED_DIM (512)
      - all values are numeric
      - no NaN or Inf values
      - 'track_id' field exists
    """
    valid: list[dict] = []
    n_invalid = 0

    for i, rec in enumerate(records):
        path_label = rec.get("crop_path", f"<record #{i}>")
        emb = rec.get("embedding")

        if emb is None or not isinstance(emb, list):
            print(f"  [WARN] {path_label}: missing or non-list embedding — skipped")
            n_invalid += 1
            continue

        if len(emb) != EXPECTED_DIM:
            print(f"  [WARN] {path_label}: embedding length {len(emb)} ≠ {EXPECTED_DIM} — skipped")
            n_invalid += 1
            continue

        non_numeric = [j for j, v in enumerate(emb) if not isinstance(v, (int, float))]
        if non_numeric:
            print(f"  [WARN] {path_label}: non-numeric values at indices {non_numeric[:3]} — skipped")
            n_invalid += 1
            continue

        bad_floats = [j for j, v in enumerate(emb) if isinstance(v, float) and (math.isnan(v) or math.isinf(v))]
        if bad_floats:
            print(f"  [WARN] {path_label}: NaN/Inf at indices {bad_floats[:3]} — skipped")
            n_invalid += 1
            continue

        if "track_id" not in rec:
            print(f"  [WARN] {path_label}: missing track_id — skipped")
            n_invalid += 1
            continue

        valid.append(rec)

    if n_invalid:
        print(f"[WARN] {n_invalid} record(s) failed validation and were excluded.")
    else:
        print(f"[INFO] All {len(valid)} records passed validation.")

    return valid


# ---------------------------------------------------------------------------
# 2. Grouping
# ---------------------------------------------------------------------------

def group_by_track(records: list[dict]) -> dict[int, list[dict]]:
    """Group records by track_id, sorted by frame within each group."""
    groups: dict[int, list[dict]] = {}
    for r in records:
        groups.setdefault(r["track_id"], []).append(r)
    for tid in groups:
        groups[tid].sort(key=lambda x: x["frame"])
    return groups


# ---------------------------------------------------------------------------
# 3. Pair sampling
# ---------------------------------------------------------------------------

def sample_same_track_pairs(
    records: list[dict],
    max_pairs_per_track: int = MAX_PAIRS_PER_TRACK,
    seed: int = SEED,
) -> list[tuple[dict, dict]]:
    """
    For each track with ≥ 2 crops, sample up to max_pairs_per_track pairs.

    Strategy:
      1. Build all valid (i < j) pairs for the track where i ≠ j.
      2. Prefer non-adjacent frames (|frame_a − frame_b| > 1) because
         consecutive frames are very similar and less diagnostic.
      3. If only adjacent pairs exist, fall back to those.
      4. Shuffle the candidate pool with the fixed seed, then take the first
         max_pairs_per_track entries — deterministic every run.

    A record is never paired with itself (enforced by crop_path equality check).
    """
    rng = random.Random(seed)
    groups = group_by_track(records)
    pairs: list[tuple[dict, dict]] = []

    for tid in sorted(groups):
        crops = groups[tid]
        if len(crops) < 2:
            continue

        # All distinct ordered pairs (i < j)
        candidates = [
            (crops[i], crops[j])
            for i in range(len(crops))
            for j in range(i + 1, len(crops))
            if crops[i]["crop_path"] != crops[j]["crop_path"]
        ]

        # Prefer non-adjacent frames
        non_adjacent = [
            (a, b) for a, b in candidates
            if abs(a["frame"] - b["frame"]) > 1
        ]
        pool = non_adjacent if non_adjacent else candidates

        rng.shuffle(pool)
        pairs.extend(pool[:max_pairs_per_track])

    return pairs


def sample_different_track_pairs(
    records: list[dict],
    n_pairs: int,
    seed: int = SEED,
) -> list[tuple[dict, dict]]:
    """
    Sample n_pairs pairs where each pair has different track_ids.

    Strategy:
      - Repeatedly pick two distinct track_ids at random, then one random
        crop from each.
      - Stop when n_pairs pairs have been collected or max_attempts is hit.
      - Deterministic via seed.

    Note: different track_id does NOT guarantee a different real-world person.
    This limitation is documented in the report.
    """
    rng = random.Random(seed)
    groups = group_by_track(records)
    track_ids = sorted(groups)

    if len(track_ids) < 2:
        print("[WARN] Fewer than 2 tracks — cannot create different-track pairs.")
        return []

    pairs: list[tuple[dict, dict]] = []
    max_attempts = n_pairs * 50   # generous headroom
    attempts = 0

    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        tid_a, tid_b = rng.sample(track_ids, 2)
        rec_a = rng.choice(groups[tid_a])
        rec_b = rng.choice(groups[tid_b])
        if rec_a["track_id"] != rec_b["track_id"]:
            pairs.append((rec_a, rec_b))

    if len(pairs) < n_pairs:
        print(
            f"[WARN] Requested {n_pairs} different-track pairs but only "
            f"collected {len(pairs)} after {max_attempts} attempts."
        )

    return pairs


# ---------------------------------------------------------------------------
# 4. Similarity computation
# ---------------------------------------------------------------------------

def compute_pair_similarities(pairs: list[tuple[dict, dict]]) -> list[float]:
    """Return a list of cosine similarity values for the given pairs."""
    return [
        cosine_similarity(a["embedding"], b["embedding"])
        for a, b in pairs
    ]


# ---------------------------------------------------------------------------
# 5. Report printing
# ---------------------------------------------------------------------------

def _fmt(value: float, decimals: int = 4) -> str:
    return f"{value:.{decimals}f}"


def print_report(
    *,
    embeddings_path: Path,
    n_total: int,
    n_tracks: int,
    same_sims: list[float],
    diff_sims: list[float],
    same_stats: dict,
    diff_stats: dict,
    mean_gap: float,
    median_gap: float,
    threshold_same: dict,
    threshold_diff: dict,
) -> None:
    """Print the structured Phase 4.5 report to stdout."""

    SEP  = "=" * 50
    DASH = "-" * 32

    print()
    print(SEP)
    print("TRACE PHASE 4.5 — RE-ID SANITY CHECK")
    print(SEP)
    print()
    print(f"Input:                    {embeddings_path}")
    print(f"Total embeddings:         {n_total}")
    print(f"Unique tracks:            {n_tracks}")
    print(f"Same-track comparisons:   {same_stats['count']}")
    print(f"Different-track comps:    {diff_stats['count']}")
    print()

    print(DASH)
    print("SAME-TRACK SIMILARITY")
    print(DASH)
    print(f"Mean:      {_fmt(same_stats['mean'])}")
    print(f"Median:    {_fmt(same_stats['median'])}")
    print(f"Std:       {_fmt(same_stats['std'])}")
    print(f"Min:       {_fmt(same_stats['min'])}")
    print(f"Max:       {_fmt(same_stats['max'])}")
    print(f"P25:       {_fmt(same_stats['p25'])}")
    print(f"P75:       {_fmt(same_stats['p75'])}")
    print()

    print(DASH)
    print("DIFFERENT-TRACK SIMILARITY")
    print(DASH)
    print(f"Mean:      {_fmt(diff_stats['mean'])}")
    print(f"Median:    {_fmt(diff_stats['median'])}")
    print(f"Std:       {_fmt(diff_stats['std'])}")
    print(f"Min:       {_fmt(diff_stats['min'])}")
    print(f"Max:       {_fmt(diff_stats['max'])}")
    print(f"P25:       {_fmt(diff_stats['p25'])}")
    print(f"P75:       {_fmt(diff_stats['p75'])}")
    print()

    print(DASH)
    print("SEPARATION")
    print(DASH)
    print(f"Mean gap   (same − diff): {_fmt(mean_gap)}")
    print(f"Median gap (same − diff): {_fmt(median_gap)}")
    print()

    print(DASH)
    print("THRESHOLD DIAGNOSTICS")
    print(DASH)
    print("(Fraction of pairs with cosine similarity > threshold)")
    print()
    for t in THRESHOLDS:
        s_frac = threshold_same.get(t, 0.0)
        d_frac = threshold_diff.get(t, 0.0)
        print(f"  Threshold {t:.2f}:")
        print(f"    Same-track above:      {s_frac * 100:6.1f}%")
        print(f"    Different-track above: {d_frac * 100:6.1f}%")
    print()

    print(DASH)
    print("INTERPRETATION")
    print(DASH)

    if mean_gap > 0.05:
        separation_label = "meaningful"
    elif mean_gap > 0.0:
        separation_label = "small but positive"
    else:
        separation_label = "absent or negative — investigate further"

    if same_stats["mean"] > diff_stats["mean"]:
        direction = "HIGHER than"
    else:
        direction = "NOT higher than"

    print(
        f"Same-track mean similarity ({_fmt(same_stats['mean'])}) is {direction} "
        f"different-track mean ({_fmt(diff_stats['mean'])})."
    )
    print(f"Mean gap: {_fmt(mean_gap)} — separation is {separation_label}.")
    print()

    # Distribution overlap heuristic: how much of the different-track
    # distribution sits above the same-track median?
    same_med  = same_stats["median"]
    diff_above_same_median = float(np.mean(np.array(diff_sims) > same_med))
    print(
        f"Distribution overlap: {diff_above_same_median * 100:.1f}% of different-track "
        f"pairs have similarity > same-track median ({_fmt(same_med)})."
    )

    if diff_above_same_median > 0.40:
        print(
            "Overlap is substantial. The two distributions are not well-separated "
            "at the same-track median. Further investigation or model tuning may be needed."
        )
    elif diff_above_same_median > 0.15:
        print(
            "Moderate overlap exists. The distributions are partially separated "
            "but there is no clean decision boundary at this threshold."
        )
    else:
        print(
            "Overlap is low. The same-track and different-track distributions "
            "are reasonably well separated around the same-track median."
        )

    print()
    print("NOTE: Track IDs come from the Phase 2 ByteTrack tracker.")
    print("      same track_id  ≠ guaranteed same real-world person.")
    print("      different track_id ≠ guaranteed different real-world person.")
    print("      This experiment is a sanity check, not a ground-truth Re-ID evaluation.")
    print()
    print(SEP)


# ---------------------------------------------------------------------------
# 6. Distribution plot
# ---------------------------------------------------------------------------

def plot_distributions(
    same_sims: list[float],
    diff_sims: list[float],
    output_path: Path,
) -> bool:
    """
    Save a histogram / KDE overlay of same-track vs different-track similarities.
    Returns True on success, False if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend — safe in all environments
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available — skipping distribution plot.")
        return False

    fig, ax = plt.subplots(figsize=(9, 5))

    bins = np.linspace(-0.1, 1.0, 45)

    ax.hist(
        same_sims,
        bins=bins,
        alpha=0.55,
        color="#2196F3",
        label=f"Same-track  (n={len(same_sims)}, mean={np.mean(same_sims):.3f})",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.hist(
        diff_sims,
        bins=bins,
        alpha=0.55,
        color="#F44336",
        label=f"Diff-track  (n={len(diff_sims)}, mean={np.mean(diff_sims):.3f})",
        edgecolor="white",
        linewidth=0.4,
    )

    # Vertical lines for the means
    ax.axvline(
        float(np.mean(same_sims)),
        color="#1565C0",
        linestyle="--",
        linewidth=1.6,
        label=f"Same-track mean ({np.mean(same_sims):.3f})",
    )
    ax.axvline(
        float(np.mean(diff_sims)),
        color="#B71C1C",
        linestyle="--",
        linewidth=1.6,
        label=f"Diff-track mean ({np.mean(diff_sims):.3f})",
    )

    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        "TRACE Phase 4.5 — Re-ID Sanity Check\n"
        "Same-Track vs Different-Track Cosine Similarity (C01)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.set_xlim(-0.1, 1.05)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"[INFO] Distribution plot saved: {output_path}")
    return True


# ---------------------------------------------------------------------------
# 7. JSON report
# ---------------------------------------------------------------------------

def save_json_report(
    *,
    output_path: Path,
    embeddings_path: Path,
    n_total: int,
    n_tracks: int,
    seed: int,
    same_stats: dict,
    diff_stats: dict,
    mean_gap: float,
    median_gap: float,
    threshold_same: dict,
    threshold_diff: dict,
) -> None:
    """Serialise all measured results to dataset/reid_sanity/similarity_report.json."""

    # Convert threshold dict keys to strings for valid JSON
    def _str_keys(d: dict) -> dict:
        return {str(k): v for k, v in d.items()}

    report = {
        "phase": "4.5",
        "description": (
            "TRACE Re-ID sanity check: same-track vs different-track cosine similarity "
            "using pre-computed OSNet x1_0 embeddings."
        ),
        "input_file": str(embeddings_path),
        "n_total_embeddings": n_total,
        "n_unique_tracks": n_tracks,
        "random_seed": seed,
        "same_track": {
            "n_pairs": same_stats["count"],
            "statistics": same_stats,
        },
        "different_track": {
            "n_pairs": diff_stats["count"],
            "statistics": diff_stats,
        },
        "separation": {
            "mean_gap":   mean_gap,
            "median_gap": median_gap,
        },
        "threshold_diagnostics": {
            "thresholds": THRESHOLDS,
            "same_track_fraction_above":      _str_keys(threshold_same),
            "different_track_fraction_above": _str_keys(threshold_diff),
        },
        "limitations": [
            "same track_id does not guarantee same real-world person identity",
            "different track_id does not guarantee different real-world person identity",
            "experiment uses single camera C01 only",
            "cross-camera Re-ID is not evaluated in this phase",
            "this is a sanity check, not a ground-truth accuracy evaluation",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[INFO] JSON report saved:       {output_path}")


# ---------------------------------------------------------------------------
# 8. Main experiment
# ---------------------------------------------------------------------------

def run_sanity_check(
    embeddings_path: Path,
    output_dir: Path,
    max_pairs_per_track: int = MAX_PAIRS_PER_TRACK,
    seed: int = SEED,
    generate_plot: bool = True,
) -> None:

    # --- load and validate ---
    raw_records = load_embeddings(embeddings_path)
    records = validate_records(raw_records)

    if len(records) == 0:
        sys.exit("[ERROR] No valid records after validation — cannot proceed.")

    groups = group_by_track(records)
    n_tracks = len(groups)

    print(f"[INFO] Valid records:           {len(records)}")
    print(f"[INFO] Unique track IDs:        {n_tracks}")
    tracks_with_multi = sum(1 for crops in groups.values() if len(crops) >= 2)
    print(f"[INFO] Tracks with ≥ 2 crops:   {tracks_with_multi}")

    # --- sample pairs ---
    print(f"\n[INFO] Sampling pairs (seed={seed}) ...")
    same_pairs = sample_same_track_pairs(
        records, max_pairs_per_track=max_pairs_per_track, seed=seed
    )
    n_same = len(same_pairs)
    print(f"[INFO] Same-track pairs:        {n_same}")

    diff_pairs = sample_different_track_pairs(records, n_pairs=n_same, seed=seed)
    n_diff = len(diff_pairs)
    print(f"[INFO] Different-track pairs:   {n_diff}")

    if n_same == 0:
        sys.exit("[ERROR] No same-track pairs could be formed — check that tracks have ≥ 2 crops.")
    if n_diff == 0:
        sys.exit("[ERROR] No different-track pairs could be formed — check that ≥ 2 track IDs exist.")

    # --- compute similarities ---
    print("\n[INFO] Computing cosine similarities ...")
    same_sims = compute_pair_similarities(same_pairs)
    diff_sims  = compute_pair_similarities(diff_pairs)

    # --- statistics ---
    same_stats = compute_stats(same_sims)
    diff_stats  = compute_stats(diff_sims)

    mean_gap   = same_stats["mean"]   - diff_stats["mean"]
    median_gap = same_stats["median"] - diff_stats["median"]

    threshold_same = threshold_fractions(same_sims, THRESHOLDS)
    threshold_diff  = threshold_fractions(diff_sims,  THRESHOLDS)

    # --- print report ---
    print_report(
        embeddings_path=embeddings_path,
        n_total=len(records),
        n_tracks=n_tracks,
        same_sims=same_sims,
        diff_sims=diff_sims,
        same_stats=same_stats,
        diff_stats=diff_stats,
        mean_gap=mean_gap,
        median_gap=median_gap,
        threshold_same=threshold_same,
        threshold_diff=threshold_diff,
    )

    # --- save JSON report ---
    json_path = output_dir / "similarity_report.json"
    save_json_report(
        output_path=json_path,
        embeddings_path=embeddings_path,
        n_total=len(records),
        n_tracks=n_tracks,
        seed=seed,
        same_stats=same_stats,
        diff_stats=diff_stats,
        mean_gap=mean_gap,
        median_gap=median_gap,
        threshold_same=threshold_same,
        threshold_diff=threshold_diff,
    )

    # --- distribution plot ---
    if generate_plot:
        plot_path = output_dir / "similarity_distribution.png"
        plot_distributions(same_sims, diff_sims, plot_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRACE Phase 4.5 — Re-ID Sanity Check"
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=_REPO_ROOT / "dataset" / "embeddings_C01.json",
        help="Path to embeddings_C01.json (default: dataset/embeddings_C01.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "dataset" / "reid_sanity",
        help="Directory to save outputs (default: dataset/reid_sanity/)",
    )
    parser.add_argument(
        "--max-pairs-per-track",
        type=int,
        default=MAX_PAIRS_PER_TRACK,
        help=f"Max same-track pairs sampled per track (default: {MAX_PAIRS_PER_TRACK})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed for deterministic pair sampling (default: {SEED})",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating the distribution PNG",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_sanity_check(
        embeddings_path=args.embeddings,
        output_dir=args.output_dir,
        max_pairs_per_track=args.max_pairs_per_track,
        seed=args.seed,
        generate_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
