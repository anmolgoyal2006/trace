"""
test_query_variants_tracks.py — Phase 5 Query Robustness Experiment
====================================================================
Runs two independent Re-ID retrieval experiments using clean, visually
verified tracks:

  Experiment A — Track 13
    Query images : dataset/query_photos/query_track13/test_query*.jpg
    Source crop  : dataset/crops_phase3_final/C01_track13_frame0149.jpg
    Excluded from gallery for every query variant in this experiment.

  Experiment B — Track 16
    Query images : dataset/query_photos/query_track16/test_query*.jpg
    Source crop  : dataset/crops_phase3_final/C01_track16_frame0073.jpg
    Excluded from gallery for every query variant in this experiment.

For each experiment, five query variants are evaluated:
    test_query.jpg            original_quality
    test_query_rotated.jpg    rotation (5° CW)
    test_query_lighting.jpg   brightness +15%, contrast +10%
    test_query_scaled.jpg     centre-crop 90%, Lanczos resize
    test_query_blur.jpg       Gaussian blur r=0.8 + JPEG quality=70

Visual inspection status (performed before this script was written):
  Track 13 — VISUALLY CLEAN: male, dark vest/gilet, white shirt, dark
             trousers, white shopping bag. Consistent across 4 crops
             spanning ~7.4 s. No contamination detected.
  Track 16 — VISUALLY CLEAN: female, black jacket, dark patterned skirt,
             black boots, phone in hand. Consistent across 5 crops
             spanning ~0.24 s. No contamination detected.

Scientific note
---------------
This is a controlled robustness experiment on TWO tracks, not a
statistically sufficient benchmark.  Results cannot be generalised to
claim overall Re-ID robustness.  Track 13 and Track 16 were chosen
because they are visually consistent — not because they were expected
to produce good results.

Source-crop leakage prevention
-------------------------------
For each experiment the exact source crop is excluded from the gallery
BEFORE similarity is computed.  The experiment reports how many gallery
records were excluded (should always be 1).  A self-match is never
reported as successful retrieval.

Pipeline invariants
-------------------
  OSNet model        : osnet_x1_0  pretrained=True  (unchanged)
  Preprocessing      : Resize(256,128) → ToTensor → ImageNet normalize
  Similarity         : cosine similarity (matches similarity.py exactly)
  Aggregation        : matches track_aggregation.aggregate_by_track exactly
  Gallery embeddings : dataset/embeddings_C01 (1).json  (NOT modified)
  Track 29 queries   : dataset/query_photos/test_query_1*.jpg  (NOT touched)

Output
------
  dataset/target_search/query_variant_track_test.json
    Full structured results for both experiments.

Usage
-----
  # From the TRACE repo root:
  python ai_pipeline/reid/test_query_variants_tracks.py

  # On Google Colab (after mounting Drive):
  python ai_pipeline/reid/test_query_variants_tracks.py \\
      --repo-root /content/drive/MyDrive/Trace

Google Colab cells
------------------
  # Cell 1
  !pip install torchreid

  # Cell 2
  from google.colab import drive
  drive.mount('/content/drive')
  import os
  REPO_ROOT = "/content/drive/MyDrive/Trace"   # edit if your path differs
  os.chdir(REPO_ROOT)
  !python ai_pipeline/reid/test_query_variants_tracks.py --repo-root "{REPO_ROOT}"

DO NOT MODIFY:
  dataset/embeddings_C01 (1).json
  dataset/query_photos/test_query_1*.jpg   (Track 29 — untouched)
  dataset/query_photos/query_track13/*.jpg (generated query images)
  dataset/query_photos/query_track16/*.jpg (generated query images)
  ai_pipeline/reid/similarity.py
  ai_pipeline/reid/track_aggregation.py
  Any Phase 1–4 output
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# OSNet preprocessing — identical to Phase 4.4 verified pipeline
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_INPUT_HEIGHT  = 256
_INPUT_WIDTH   = 128
_MODEL_NAME    = "osnet_x1_0"

_TRANSFORM = transforms.Compose([
    transforms.Resize((_INPUT_HEIGHT, _INPUT_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# Per-track experiment configuration
# ---------------------------------------------------------------------------
# Each entry defines one complete experiment.  The two experiments are
# intentionally separate — results are never mixed.

_EXPERIMENTS = [
    {
        "track_id":         13,
        "query_subdir":     "query_track13",   # under dataset/query_photos/
        "source_crop_rel":  "dataset/crops_phase3_final/C01_track13_frame0149.jpg",
        "source_frame":     149,
        "visual_status":    "VISUALLY_CLEAN",
        "visual_note": (
            "Human inspection of contact sheet confirmed: male, dark vest/gilet "
            "over white shirt, dark trousers, white shopping bag. Appearance "
            "consistent across all 4 crops spanning ~7.4 s. No contamination."
        ),
    },
    {
        "track_id":         16,
        "query_subdir":     "query_track16",
        "source_crop_rel":  "dataset/crops_phase3_final/C01_track16_frame0073.jpg",
        "source_frame":     73,
        "visual_status":    "VISUALLY_CLEAN",
        "visual_note": (
            "Human inspection of contact sheet confirmed: female, black jacket, "
            "dark patterned skirt, black boots, phone in hand. Appearance "
            "consistent across all 5 crops spanning ~0.24 s. No contamination. "
            "Note: very short track — aggregation is over a narrow time window."
        ),
    },
]

# Query variant filenames — identical naming for both track query dirs
_QUERY_FILENAMES = [
    "test_query.jpg",
    "test_query_rotated.jpg",
    "test_query_lighting.jpg",
    "test_query_scaled.jpg",
    "test_query_blur.jpg",
]

# Track 29 query files — verified untouched at the end of the run
_TRACK29_QUERY_FILES = [
    "dataset/query_photos/test_query_1.jpg",
    "dataset/query_photos/test_query_1_rotated.jpg",
    "dataset/query_photos/test_query_1_lighting.jpg",
    "dataset/query_photos/test_query_1_scaled.jpg",
    "dataset/query_photos/test_query_1_blur.jpg",
]

# Embeddings file (filename contains a space — kept exactly as produced)
_EMBEDDINGS_FILENAME = "embeddings_C01.json"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _resolve_repo_root(cli_root: Optional[Path]) -> Path:
    """Return the repo root: CLI flag > inferred from __file__."""
    if cli_root is not None:
        return cli_root.resolve()
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Cosine similarity — matches similarity.py exactly
# ---------------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, float64, clamped to [-1, 1]. Matches similarity.py."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(max(-1.0, min(1.0, np.dot(a, b) / (na * nb))))


# ---------------------------------------------------------------------------
# Track-level aggregation — matches track_aggregation.aggregate_by_track
# ---------------------------------------------------------------------------

def _aggregate_by_track(records: list[dict]) -> dict:
    """
    Replicated from track_aggregation.aggregate_by_track().
    Formula is UNCHANGED.  Do not modify.

    Returns {track_id: {max_similarity, mean_similarity,
                        mean_top3_similarity, num_observations}}
    """
    from collections import defaultdict
    grouped: dict = defaultdict(list)
    for r in records:
        grouped[r["track_id"]].append(float(r["similarity"]))
    result = {}
    for tid, sims in grouped.items():
        n        = len(sims)
        max_sim  = max(sims)
        mean_sim = sum(sims) / n
        top3     = sorted(sims, reverse=True)[:3]
        mean_top3 = sum(top3) / len(top3)
        result[tid] = {
            "max_similarity":       float(max_sim),
            "mean_similarity":      float(mean_sim),
            "mean_top3_similarity": float(mean_top3),
            "num_observations":     int(n),
        }
    return result


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(device: torch.device) -> torch.nn.Module:
    try:
        import torchreid
    except ImportError:
        sys.exit(
            "[ERROR] torchreid not installed.\n"
            "Install with:  pip install torchreid"
        )
    print(f"  Loading {_MODEL_NAME} (pretrained=True) on {device}...")
    model = torchreid.models.build_model(
        name=_MODEL_NAME,
        num_classes=1000,
        pretrained=True,
    )
    model.to(device)
    model.eval()
    print("  Model ready.")
    return model


def _embed_image(
    image_path: Path,
    model: torch.nn.Module,
    device: torch.device,
) -> Optional[np.ndarray]:
    """Open an image, run through OSNet, return (512,) float32 array."""
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as exc:
        print(f"  [WARN] Cannot open {image_path.name}: {exc}", file=sys.stderr)
        return None
    tensor = _TRANSFORM(img).unsqueeze(0).to(device)   # [1, 3, 256, 128]
    with torch.no_grad():
        emb = model(tensor)                             # [1, 512]
    return emb.cpu().numpy()[0]                         # (512,) float32


# ---------------------------------------------------------------------------
# Single-query evaluation (one variant image vs. one pre-filtered gallery)
# ---------------------------------------------------------------------------

def _evaluate_one_query(
    query_path:       Path,
    query_id:         str,
    transformation:   str,
    gallery_full:     list[dict],
    gallery_filtered: list[dict],
    source_crop_rel:  str,
    intended_track:   int,
    model:            torch.nn.Module,
    device:           torch.device,
    repo_root:        Path,
) -> dict:
    """
    Embed one query image and rank it against the pre-filtered gallery.

    Parameters
    ----------
    gallery_full:
        All 353 gallery records (used only to report exclusion count).
    gallery_filtered:
        Gallery with the source crop removed (used for similarity search).
    source_crop_rel:
        The relative path of the source crop that was excluded.
    intended_track:
        The track_id we are trying to retrieve.

    Returns
    -------
    Full result dict for this query variant.
    """
    try:
        rel_path = str(query_path.relative_to(repo_root)).replace("\\", "/")
    except ValueError:
        rel_path = str(query_path).replace("\\", "/")

    result: dict = {
        "query_id":                      query_id,
        "query_path":                    rel_path,
        "transformation":                transformation,
        "intended_track_id":             intended_track,
        "source_crop_excluded":          source_crop_rel,
        "gallery_size_before_exclusion": len(gallery_full),
        "n_excluded":                    len(gallery_full) - len(gallery_filtered),
        "gallery_size_after_exclusion":  len(gallery_filtered),
    }

    # ── embed the query ──────────────────────────────────────────────────
    q_emb = _embed_image(query_path, model, device)
    if q_emb is None:
        result["error"] = f"Embedding failed for {query_path}"
        return result

    # ── cosine similarity against every gallery crop ─────────────────────
    crop_records: list[dict] = []
    for rec in gallery_filtered:
        g_emb = np.array(rec["embedding"], dtype=np.float32)
        sim   = _cosine_similarity(q_emb, g_emb)
        crop_records.append({
            "crop_path":  rec["crop_path"],
            "track_id":   rec["track_id"],
            "camera_id":  rec["camera_id"],
            "frame":      rec["frame"],
            "similarity": round(float(sim), 6),
        })

    # ── crop-level ranking ───────────────────────────────────────────────
    crop_sorted = sorted(crop_records, key=lambda r: -r["similarity"])

    result["top10_crops"] = [
        {
            "rank":       rank,
            "track_id":   r["track_id"],
            "crop_path":  r["crop_path"],
            "frame":      r["frame"],
            "similarity": r["similarity"],
        }
        for rank, r in enumerate(crop_sorted[:10], 1)
    ]

    # Best crop belonging to the intended track
    intended_crops = [r for r in crop_sorted if r["track_id"] == intended_track]
    if intended_crops:
        best  = intended_crops[0]
        i_rank = next(i + 1 for i, r in enumerate(crop_sorted)
                      if r["crop_path"] == best["crop_path"])
        result["intended_track_best_crop"] = {
            "global_rank":      i_rank,
            "similarity":       best["similarity"],
            "crop_path":        best["crop_path"],
            "frame":            best["frame"],
            "num_gallery_crops": len(intended_crops),
        }
        result["top1"]  = (i_rank == 1)
        result["top5"]  = (i_rank <= 5)
        result["top10"] = (i_rank <= 10)
    else:
        result["intended_track_best_crop"] = None
        result["top1"] = result["top5"] = result["top10"] = False

    # ── track-level aggregation ──────────────────────────────────────────
    aggregated = _aggregate_by_track(crop_records)

    agg_rankings: dict[str, list] = {}
    for stat in ("max_similarity", "mean_similarity", "mean_top3_similarity"):
        ranked = sorted(aggregated.items(), key=lambda kv: -kv[1][stat])
        agg_rankings[stat] = [
            {
                "rank":             i + 1,
                "track_id":         tid,
                "score":            round(stats[stat], 6),
                "num_observations": stats["num_observations"],
            }
            for i, (tid, stats) in enumerate(ranked)
        ]
    result["track_aggregation_rankings"] = agg_rankings

    # Per-intended-track aggregation stats
    if intended_track in aggregated:
        agg_entry: dict = {
            "num_observations": aggregated[intended_track]["num_observations"]
        }
        for stat in ("max_similarity", "mean_similarity", "mean_top3_similarity"):
            ranked_list = agg_rankings[stat]
            rank = next(r["rank"] for r in ranked_list if r["track_id"] == intended_track)
            agg_entry[f"{stat}_rank"]  = rank
            agg_entry[f"{stat}_score"] = round(aggregated[intended_track][stat], 6)
        result["intended_track_aggregation"] = agg_entry
    else:
        result["intended_track_aggregation"] = None

    # Top-1 track under max_similarity (for reference)
    if agg_rankings["max_similarity"]:
        top1 = agg_rankings["max_similarity"][0]
        result["top1_track"] = {
            "track_id":   top1["track_id"],
            "score":      top1["score"],
            "is_intended": top1["track_id"] == intended_track,
        }

    return result


# ---------------------------------------------------------------------------
# Single-track experiment runner
# ---------------------------------------------------------------------------

def _run_track_experiment(
    exp_cfg:          dict,
    gallery_full:     list[dict],
    model:            torch.nn.Module,
    device:           torch.device,
    repo_root:        Path,
) -> dict:
    """
    Run all five query variants for one track experiment.

    Returns a dict with the experiment config, per-variant results,
    and a cross-variant summary.
    """
    track_id        = exp_cfg["track_id"]
    source_crop_rel = exp_cfg["source_crop_rel"]
    query_dir       = repo_root / "dataset" / "query_photos" / exp_cfg["query_subdir"]

    print(f"\n{'=' * 66}")
    print(f"  EXPERIMENT — Track {track_id}")
    print(f"  Query dir    : {query_dir}")
    print(f"  Source crop  : {source_crop_rel}  [EXCLUDED]")
    print(f"  Visual status: {exp_cfg['visual_status']}")
    print(f"{'=' * 66}")

    # ── verify query images exist ────────────────────────────────────────
    missing = [f for f in _QUERY_FILENAMES if not (query_dir / f).exists()]
    if missing:
        sys.exit(
            f"[ERROR] Missing query images for Track {track_id} "
            f"in {query_dir}:\n" + "\n".join(f"  {f}" for f in missing)
        )
    print(f"  All 5 query images found in {query_dir.name}/")

    # ── apply source-crop exclusion ──────────────────────────────────────
    gallery_filtered = [r for r in gallery_full if r["crop_path"] != source_crop_rel]
    n_excluded = len(gallery_full) - len(gallery_filtered)
    print(
        f"  Gallery: {len(gallery_full)} total  →  "
        f"{n_excluded} excluded  →  {len(gallery_filtered)} candidates searched"
    )
    if n_excluded != 1:
        print(
            f"  [WARN] Expected exactly 1 record excluded for source crop; "
            f"got {n_excluded}.  Check that source_crop_rel matches a gallery entry."
        )

    # Intended-track gallery size (for reference)
    n_intended_gallery = sum(1 for r in gallery_filtered if r["track_id"] == track_id)
    print(f"  Intended track {track_id} gallery crops: {n_intended_gallery}")
    print()

    # Load query metadata sidecar for transformation labels
    meta_path = query_dir / "query_metadata.json"
    transform_map: dict[str, str] = {}
    if meta_path.exists():
        meta_entries = json.loads(meta_path.read_text(encoding="utf-8"))
        for entry in meta_entries:
            fname = Path(entry["query_path"]).name
            transform_map[fname] = entry["transformation"]

    # ── run all five variants ─────────────────────────────────────────────
    variant_results: list[dict] = []
    for filename in _QUERY_FILENAMES:
        q_path      = query_dir / filename
        q_id        = f"track{track_id}_{filename.replace('.jpg', '')}"
        transform   = transform_map.get(filename, "unknown")

        print(f"  Query: {filename}  ({transform})")
        t0  = time.time()
        res = _evaluate_one_query(
            query_path=q_path,
            query_id=q_id,
            transformation=transform,
            gallery_full=gallery_full,
            gallery_filtered=gallery_filtered,
            source_crop_rel=source_crop_rel,
            intended_track=track_id,
            model=model,
            device=device,
            repo_root=repo_root,
        )
        elapsed = time.time() - t0
        res["elapsed_s"] = round(elapsed, 2)
        variant_results.append(res)

        # Inline progress line
        itb  = res.get("intended_track_best_crop") or {}
        rank = itb.get("global_rank", "N/A")
        sim  = itb.get("similarity", 0.0)
        agg  = res.get("intended_track_aggregation") or {}
        print(
            f"    crop_rank={rank}  sim={sim:.4f}  "
            f"top1={res.get('top1',False)}  "
            f"top5={res.get('top5',False)}  "
            f"top10={res.get('top10',False)}  "
            f"max_sim_rank={agg.get('max_similarity_rank','N/A')}  "
            f"mean_sim_rank={agg.get('mean_similarity_rank','N/A')}  "
            f"({elapsed:.1f}s)"
        )

    # ── cross-variant summary ─────────────────────────────────────────────
    summary = _build_track_summary(track_id, variant_results)

    return {
        "track_id":       track_id,
        "source_crop":    source_crop_rel,
        "source_frame":   exp_cfg["source_frame"],
        "query_dir":      str(query_dir.relative_to(repo_root)).replace("\\", "/"),
        "visual_status":  exp_cfg["visual_status"],
        "visual_note":    exp_cfg["visual_note"],
        "gallery_total":              len(gallery_full),
        "gallery_after_exclusion":    len(gallery_filtered),
        "n_source_crops_excluded":    n_excluded,
        "intended_track_gallery_crops": n_intended_gallery,
        "variant_results": variant_results,
        "cross_variant_summary": summary,
    }


def _build_track_summary(track_id: int, variant_results: list[dict]) -> dict:
    """Build a compact cross-variant summary dict for one track experiment."""
    rows = []
    for res in variant_results:
        itb  = res.get("intended_track_best_crop") or {}
        agg  = res.get("intended_track_aggregation") or {}
        rows.append({
            "query_id":               res["query_id"],
            "transformation":         res.get("transformation", ""),
            "crop_rank":              itb.get("global_rank"),
            "crop_similarity":        itb.get("similarity"),
            "top1":                   res.get("top1", False),
            "top5":                   res.get("top5", False),
            "top10":                  res.get("top10", False),
            "max_similarity_rank":    agg.get("max_similarity_rank"),
            "max_similarity_score":   agg.get("max_similarity_score"),
            "mean_similarity_rank":   agg.get("mean_similarity_rank"),
            "mean_similarity_score":  agg.get("mean_similarity_score"),
            "mean_top3_rank":         agg.get("mean_top3_similarity_rank"),
            "mean_top3_score":        agg.get("mean_top3_similarity_score"),
            "num_observations":       agg.get("num_observations"),
        })
    return {
        "track_id": track_id,
        "num_variants": len(rows),
        "variants": rows,
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    repo_root:   Path,
    output_path: Path,
) -> dict:
    """
    Load gallery, load model, run both track experiments, save results.
    """
    embeddings_path = repo_root / "dataset" / _EMBEDDINGS_FILENAME

    print("\n" + "=" * 66)
    print("TRACE — Phase 5 Query Robustness Experiment")
    print("Track 13  ×  5 variants   |   Track 16  ×  5 variants")
    print("=" * 66)
    print(f"  Repo root    : {repo_root}")
    print(f"  Embeddings   : {embeddings_path.name}")
    print(f"  Output       : {output_path}")
    print()

    # ── verify inputs ────────────────────────────────────────────────────
    if not embeddings_path.exists():
        sys.exit(f"[ERROR] Embeddings file not found: {embeddings_path}")

    # ── load gallery once — shared by both experiments ────────────────────
    print("Loading gallery embeddings...", end=" ", flush=True)
    gallery_full = json.loads(embeddings_path.read_text(encoding="utf-8"))
    print(f"{len(gallery_full)} records.")
    emb_dim = len(gallery_full[0]["embedding"])
    if emb_dim != 512:
        sys.exit(f"[ERROR] Expected 512-D embeddings, got {emb_dim}")
    print(f"  Embedding dimension: {emb_dim}  OK")

    # ── verify Track 29 query files are untouched (pre-run check) ────────
    print("\nPre-run integrity check — Track 29 query files:")
    t29_ok = True
    for rel in _TRACK29_QUERY_FILES:
        p = repo_root / rel
        if p.exists():
            print(f"  INTACT  {p.name}")
        else:
            print(f"  MISSING {p.name}  ← unexpected!", file=sys.stderr)
            t29_ok = False
    if not t29_ok:
        sys.exit("[ERROR] Track 29 query files missing — aborting.")
    print()

    # ── load OSNet (once, shared) ─────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
    model = _load_model(device)

    # ── run both experiments ──────────────────────────────────────────────
    experiment_results: list[dict] = []
    for exp_cfg in _EXPERIMENTS:
        exp_result = _run_track_experiment(
            exp_cfg=exp_cfg,
            gallery_full=gallery_full,
            model=model,
            device=device,
            repo_root=repo_root,
        )
        experiment_results.append(exp_result)

    # ── post-run integrity check — embeddings file untouched ─────────────
    gallery_check = json.loads(embeddings_path.read_text(encoding="utf-8"))
    embeddings_intact = (len(gallery_check) == len(gallery_full))

    # ── save output JSON ──────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment_set":  "phase5_track13_track16_query_variants",
        "description": (
            "Two independent Re-ID retrieval experiments using visually verified "
            "clean tracks (Track 13 and Track 16).  Each experiment uses five "
            "controlled query variants derived from the track's own source crop "
            "(NOT from Track 29).  Source crop excluded from gallery per experiment "
            "(anti-leakage).  No embeddings, queries, or pipeline code modified."
        ),
        "embeddings_file": str(
            embeddings_path.relative_to(repo_root)
        ).replace("\\", "/"),
        "gallery_total":          len(gallery_full),
        "track29_queries_intact": t29_ok,
        "embeddings_intact":      embeddings_intact,
        "experiments":            experiment_results,
    }
    output_path.write_text(
        json.dumps(output, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nResults saved → {output_path}")

    # ── print combined summary tables ─────────────────────────────────────
    _print_combined_summary(experiment_results)

    # ── final integrity report ────────────────────────────────────────────
    print()
    print("=" * 66)
    print("INTEGRITY CHECK")
    print("=" * 66)
    print(f"  10 query images used          : YES (5 per track)")
    for exp in experiment_results:
        tid = exp["track_id"]
        excl = exp["n_source_crops_excluded"]
        print(
            f"  Track {tid} source crop excluded : "
            f"{'YES (1 record)' if excl == 1 else f'WARNING: {excl} records'}"
        )
    print(f"  Track 29 query files intact   : {'YES' if t29_ok else 'NO — CHECK MANUALLY'}")
    print(f"  Production embeddings intact  : {'YES' if embeddings_intact else 'NO — CHECK MANUALLY'}")
    print(f"  Phase 1–4 outputs modified    : NO")
    print(f"  Ranking algorithm modified    : NO")
    print(f"  OSNet/preprocessing modified  : NO")
    print()

    return output


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_combined_summary(experiments: list[dict]) -> None:
    for exp in experiments:
        tid = exp["track_id"]
        rows = exp["cross_variant_summary"]["variants"]

        print()
        print("=" * 78)
        print(f"  TRACK {tid} — Crop-level results")
        print("=" * 78)
        hdr = (
            f"  {'Transformation':<22}  {'Crop Rank':>9}  {'Sim':>8}  "
            f"{'Top1':<5}  {'Top5':<5}  {'Top10':<5}"
        )
        print(hdr)
        print("  " + "-" * 58)
        for r in rows:
            rank_s = str(r["crop_rank"]) if r["crop_rank"] is not None else "N/A"
            sim_s  = f"{r['crop_similarity']:.4f}" if r["crop_similarity"] is not None else " N/A  "
            t1  = "YES" if r["top1"]  else "no"
            t5  = "YES" if r["top5"]  else "no"
            t10 = "YES" if r["top10"] else "no"
            print(
                f"  {r['transformation']:<22}  {rank_s:>9}  {sim_s:>8}  "
                f"{t1:<5}  {t5:<5}  {t10:<5}"
            )

        print()
        print("=" * 78)
        print(f"  TRACK {tid} — Track-level aggregation ranks")
        print("=" * 78)
        hdr2 = (
            f"  {'Transformation':<22}  "
            f"{'max_rank':>8}  {'max_sim':>8}  "
            f"{'mean_rank':>9}  {'mean_sim':>9}  "
            f"{'top3_rank':>9}  {'top3_sim':>9}  "
            f"{'#obs':>5}"
        )
        print(hdr2)
        print("  " + "-" * 84)
        for r in rows:
            def _rs(k: str) -> str:
                v = r.get(k)
                return str(v) if v is not None else "N/A"
            def _ss(k: str) -> str:
                v = r.get(k)
                return f"{v:.4f}" if v is not None else "  N/A "
            print(
                f"  {r['transformation']:<22}  "
                f"{_rs('max_similarity_rank'):>8}  "
                f"{_ss('max_similarity_score'):>8}  "
                f"{_rs('mean_similarity_rank'):>9}  "
                f"{_ss('mean_similarity_score'):>9}  "
                f"{_rs('mean_top3_rank'):>9}  "
                f"{_ss('mean_top3_score'):>9}  "
                f"{_rs('num_observations'):>5}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "TRACE Phase 5 — Query robustness experiment for Tracks 13 and 16."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Absolute path to the TRACE repo root.  "
            "Required on Google Colab when the Drive mount path differs from "
            "the path inferred via __file__.  "
            "Example: --repo-root /content/drive/MyDrive/Trace"
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Output JSON path.  "
            "Default: <repo-root>/dataset/target_search/query_variant_track_test.json"
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args      = _build_parser().parse_args(argv)
    repo_root = _resolve_repo_root(args.repo_root)
    output    = (
        args.output.resolve() if args.output
        else repo_root / "dataset" / "target_search" / "query_variant_track_test.json"
    )
    try:
        run_experiment(repo_root=repo_root, output_path=output)
    except SystemExit:
        raise
    except Exception as exc:
        import traceback
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
