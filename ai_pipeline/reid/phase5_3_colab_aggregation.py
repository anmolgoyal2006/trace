"""
Phase 5.3 — Track-Level Aggregation Experiment
===============================================
Run this script in Google Colab against the REAL Phase 5.2 data.

What this script does
---------------------
1. Loads the real dataset/embeddings_C01.json (353 gallery crops).
2. Loads the Phase 5.1 query image: dataset/query_photos/test_query_1.jpg
3. Produces a query embedding using the SAME OSNet x1_0 preprocessing and
   model checkpoint used in Phase 4 (bit-exact match verified in Phase 4.4).
4. Computes cosine similarity between the query and every gallery crop.
5. Excludes the source crop (C01_track29_frame0270.jpg) from the gallery
   to prevent trivial self-matching (Phase 5.1 anti-leakage rule).
6. Passes the crop-level similarity records to aggregate_by_track().
7. Saves results to dataset/target_search/track_aggregation_test_query_1.json
8. Prints a full ranked report for each of the four aggregation statistics,
   explicitly reporting Track 29's rank and score under each method.

Prerequisites (Colab)
---------------------
    !pip install torchreid
    (torchvision, numpy, Pillow are pre-installed in Colab)

Mount your Google Drive first and set REPO_ROOT to your TRACE folder path.

IMPORTANT
---------
- This script does NOT modify embeddings_C01.json.
- This script does NOT modify any Phase 1–4 files.
- This script does NOT apply confidence scaling or thresholds.
- This script does NOT select a "winner" — it only reports statistics.
- No result is pre-declared. The numbers here are measured, not invented.
"""

# ===========================================================================
# CELL 1 — Configuration
# ===========================================================================

import json
import os
import sys
from pathlib import Path

# ── Set TRACE_REPO_ROOT env var to override, or edit the fallback path ──────
REPO_ROOT = Path(
    os.environ.get(
        "TRACE_REPO_ROOT",
        "/content/drive/.shortcut-targets-by-id/1-_b6Eoxlyvf-BvJKPeT0DshfzmKirCPq/Trace",
    )
)

# Paths derived from REPO_ROOT
EMBEDDINGS_PATH  = REPO_ROOT / "dataset" / "embeddings_C01.json"
QUERY_IMAGE_PATH = REPO_ROOT / "dataset" / "query_photos" / "test_query_1.jpg"
QUERY_META_PATH  = REPO_ROOT / "dataset" / "query_photos" / "query_metadata.json"
OUTPUT_DIR       = REPO_ROOT / "dataset" / "target_search"
OUTPUT_PATH      = OUTPUT_DIR / "track_aggregation_test_query_1.json"

# The source crop that was used to generate the query — MUST be excluded
SOURCE_CROP_REL  = "dataset/crops_phase3_final/C01_track29_frame0270.jpg"
SOURCE_TRACK_ID  = 29

# OSNet preprocessing — identical to Phase 4.4 verified pipeline
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]
INPUT_HEIGHT   = 256
INPUT_WIDTH    = 128
MODEL_NAME     = "osnet_x1_0"

print("=" * 60)
print("TRACE Phase 5.3 — Track-Level Aggregation Experiment")
print("=" * 60)
print(f"  REPO_ROOT     : {REPO_ROOT}")
print(f"  Embeddings    : {EMBEDDINGS_PATH}")
print(f"  Query image   : {QUERY_IMAGE_PATH}")
print(f"  Output        : {OUTPUT_PATH}")
print(f"  Source crop   : {SOURCE_CROP_REL}  [EXCLUDED from gallery]")
print()


# ===========================================================================
# CELL 2 — Verify files exist before doing any GPU work
# ===========================================================================

def _require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

_require(EMBEDDINGS_PATH,  "embeddings_C01.json")
_require(QUERY_IMAGE_PATH, "query image")
_require(QUERY_META_PATH,  "query metadata")
print("File existence check  OK")


# ===========================================================================
# CELL 3 — Load gallery embeddings
# ===========================================================================

with open(EMBEDDINGS_PATH) as fh:
    gallery_records = json.load(fh)

print(f"Gallery records loaded : {len(gallery_records)}")

# Exclude the exact source crop (anti-leakage)
gallery_filtered = [
    r for r in gallery_records
    if r["crop_path"] != SOURCE_CROP_REL
]
n_excluded = len(gallery_records) - len(gallery_filtered)
print(f"Source crop excluded   : {n_excluded} record(s) removed")
print(f"Gallery size (clean)   : {len(gallery_filtered)}")

# Sanity check: each record must have a 512-D embedding
sample_dim = len(gallery_filtered[0]["embedding"])
assert sample_dim == 512, f"Unexpected embedding dimension: {sample_dim}"
print(f"Embedding dimension    : {sample_dim}  OK")


# ===========================================================================
# CELL 4 — Load model and produce query embedding
# ===========================================================================

import torch
import torchreid
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice : {device}")
if device.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

# Preprocessing transform — bit-exact match with Phase 4.4 verified pipeline
_TRANSFORM = transforms.Compose([
    transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Load OSNet x1_0 with pretrained weights
print(f"\nLoading {MODEL_NAME} with pretrained weights...")
model = torchreid.models.build_model(
    name=MODEL_NAME,
    num_classes=1000,
    pretrained=True,
)
model.to(device)
model.eval()
print(f"Model loaded on {device}")

# Preprocess and embed the query image
query_img = Image.open(QUERY_IMAGE_PATH).convert("RGB")
query_tensor = _TRANSFORM(query_img).unsqueeze(0).to(device)  # [1, 3, 256, 128]

with torch.no_grad():
    query_embedding_tensor = model(query_tensor)              # [1, 512]

query_embedding = query_embedding_tensor.cpu().numpy()[0]     # (512,)  float32
print(f"\nQuery embedding shape : {query_embedding.shape}")
print(f"Query embedding dtype : {query_embedding.dtype}")

# Sanity check
import numpy as np
query_norm = float(np.linalg.norm(query_embedding))
print(f"Query L2 norm         : {query_norm:.4f}")
assert not np.isnan(query_embedding).any(),  "NaN in query embedding!"
assert not np.isinf(query_embedding).any(),  "Inf in query embedding!"
assert query_norm > 0,                       "Zero-norm query embedding!"
print("Query embedding validation  OK")


# ===========================================================================
# CELL 5 — Compute cosine similarity for every gallery crop
# ===========================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Pure-Python/NumPy cosine similarity, matching similarity.py."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    raw = float(np.dot(a, b) / (norm_a * norm_b))
    return float(max(-1.0, min(1.0, raw)))


print(f"\nComputing cosine similarity for {len(gallery_filtered)} gallery crops...")

crop_level_records = []
for rec in gallery_filtered:
    gallery_emb = np.array(rec["embedding"], dtype=np.float32)
    sim = cosine_similarity(query_embedding, gallery_emb)
    crop_level_records.append({
        "crop_path":  rec["crop_path"],
        "track_id":   rec["track_id"],
        "camera_id":  rec["camera_id"],
        "frame":      rec["frame"],
        "similarity": sim,
    })

print(f"Similarity computed for {len(crop_level_records)} crops  OK")

# Crop-level top-10 (for reference / baseline comparison)
crop_sorted = sorted(crop_level_records, key=lambda x: -x["similarity"])
print("\n--- Crop-level top-10 (baseline) ---")
print(f"  {'Rank':<5} {'Track':<8} {'Similarity':<12} {'Crop'}")
for rank, r in enumerate(crop_sorted[:10], 1):
    marker = " ← SOURCE TRACK" if r["track_id"] == SOURCE_TRACK_ID else ""
    print(f"  {rank:<5} {r['track_id']:<8} {r['similarity']:.6f}     "
          f"{os.path.basename(r['crop_path'])}{marker}")


# ===========================================================================
# CELL 6 — Add REPO_ROOT to sys.path so track_aggregation imports correctly
# ===========================================================================

_REPO_STR = str(REPO_ROOT)
if _REPO_STR not in sys.path:
    sys.path.insert(0, _REPO_STR)

from ai_pipeline.reid.track_aggregation import aggregate_by_track
print("\ntrack_aggregation imported  OK")


# ===========================================================================
# CELL 7 — Run track-level aggregation
# ===========================================================================

aggregated = aggregate_by_track(crop_level_records)

print(f"Tracks aggregated : {len(aggregated)}")
print(f"Source track ({SOURCE_TRACK_ID}) present : {SOURCE_TRACK_ID in aggregated}")


# ===========================================================================
# CELL 8 — Rank tracks by each statistic and report
# ===========================================================================

STATS = [
    "max_similarity",
    "mean_similarity",
    "mean_top3_similarity",
]

TOP_N = 15  # how many tracks to print in the ranked table

def _rank_tracks(aggregated: dict, stat: str) -> list[tuple]:
    """Return list of (rank, track_id, stat_value, num_obs) sorted by stat desc."""
    ranked = sorted(aggregated.items(), key=lambda kv: -kv[1][stat])
    return [
        (rank + 1, tid, stats[stat], stats["num_observations"])
        for rank, (tid, stats) in enumerate(ranked)
    ]


print("\n" + "=" * 60)
print("TRACK-LEVEL AGGREGATION RESULTS")
print("=" * 60)

ranking_results = {}   # will be saved to JSON

for stat in STATS:
    ranked = _rank_tracks(aggregated, stat)
    ranking_results[stat] = [
        {"rank": r, "track_id": tid, "score": score, "num_observations": n}
        for r, tid, score, n in ranked
    ]

    # Find Track 29's rank and score
    track29_rank  = next((r for r, tid, _, _ in ranked if tid == SOURCE_TRACK_ID), None)
    track29_score = aggregated[SOURCE_TRACK_ID][stat] if SOURCE_TRACK_ID in aggregated else None

    # Top competitor (rank 1 if track29 is not rank 1, else rank 2)
    competitor_rank, competitor_tid, competitor_score, _ = ranked[0]
    if competitor_tid == SOURCE_TRACK_ID and len(ranked) > 1:
        competitor_rank, competitor_tid, competitor_score, _ = ranked[1]

    print(f"\n── {stat} ──")
    print(f"  {'Rank':<6} {'Track':<8} {'Score':<12} {'#Obs':<6}")
    print(f"  {'-'*4:<6} {'-'*5:<8} {'-'*10:<12} {'-'*4:<6}")
    for r, tid, score, n in ranked[:TOP_N]:
        marker = " ◄ SOURCE TRACK" if tid == SOURCE_TRACK_ID else ""
        print(f"  {r:<6} {tid:<8} {score:<12.6f} {n:<6}{marker}")
    if len(ranked) > TOP_N:
        print(f"  ... ({len(ranked) - TOP_N} more tracks not shown)")

    print()
    print(f"  Track {SOURCE_TRACK_ID} rank  : {track29_rank}")
    print(f"  Track {SOURCE_TRACK_ID} score : {track29_score:.6f}" if track29_score else "  Track 29 : NOT FOUND")
    print(f"  Top competitor : Track {competitor_tid}  score={competitor_score:.6f}")

    # Previous crop-level baseline: Track 29 was Rank #2
    crop_baseline_rank = next(
        (i + 1 for i, r in enumerate(crop_sorted) if r["track_id"] == SOURCE_TRACK_ID),
        None,
    )
    improved = (track29_rank is not None and track29_rank < 2)  # Rank 1 = improvement
    print(f"  Crop-level baseline: Track {SOURCE_TRACK_ID} was Rank #{crop_baseline_rank}")
    print(f"  Rank improvement   : {'YES — Track 29 is now Rank #1' if improved else 'NO'}")


# ===========================================================================
# CELL 9 — Summary table
# ===========================================================================

print("\n" + "=" * 60)
print("SUMMARY: Track 29 Performance Across All Methods")
print("=" * 60)
print(f"  {'Method':<30} {'Track 29 Rank':<16} {'Track 29 Score':<16} {'Is Rank #1?'}")
print(f"  {'-'*28:<30} {'-'*13:<16} {'-'*13:<16} {'-'*10}")

baseline_crop_rank = next(
    (i + 1 for i, r in enumerate(crop_sorted) if r["track_id"] == SOURCE_TRACK_ID),
    None,
)
baseline_crop_score = next(
    (r["similarity"] for r in crop_sorted if r["track_id"] == SOURCE_TRACK_ID),
    None,
)
print(f"  {'crop_level (baseline)':<30} {str(baseline_crop_rank):<16} "
      f"{baseline_crop_score:<16.6f} {'YES' if baseline_crop_rank == 1 else 'NO'}")

for stat in STATS:
    ranked = _rank_tracks(aggregated, stat)
    track29_rank  = next((r for r, tid, _, _ in ranked if tid == SOURCE_TRACK_ID), None)
    track29_score = aggregated[SOURCE_TRACK_ID][stat] if SOURCE_TRACK_ID in aggregated else None
    is_rank1 = "YES" if track29_rank == 1 else "NO"
    score_str = f"{track29_score:.6f}" if track29_score is not None else "N/A"
    print(f"  {stat:<30} {str(track29_rank):<16} {score_str:<16} {is_rank1}")

print()
print("NOTE: This is a derived-query robustness test.")
print("      Track 29 is the source track — not ground-truth identity.")
print("      These results do NOT prove cross-camera Re-ID performance.")


# ===========================================================================
# CELL 10 — Save results to JSON
# ===========================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_payload = {
    "experiment": "phase_5_3_track_aggregation",
    "query_image": str(QUERY_IMAGE_PATH.relative_to(REPO_ROOT)),
    "source_crop_excluded": SOURCE_CROP_REL,
    "source_track_id": SOURCE_TRACK_ID,
    "gallery_size": len(gallery_filtered),
    "num_tracks_aggregated": len(aggregated),
    "crop_level_baseline": {
        "track29_rank":  baseline_crop_rank,
        "track29_score": round(float(baseline_crop_score), 6) if baseline_crop_score else None,
    },
    "aggregated_stats": {
        track_id: {k: v for k, v in stats.items()}
        for track_id, stats in aggregated.items()
    },
    "rankings": ranking_results,
}

with open(OUTPUT_PATH, "w") as fh:
    json.dump(output_payload, fh, indent=2)

print(f"\nResults saved → {OUTPUT_PATH}")
print("Phase 5.3 aggregation experiment complete.")
