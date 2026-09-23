"""
target_search.py — Phase 5.5 / Phase 5.8
Target Search Integration Layer for TRACE Re-ID

Accepts a query photo, embeds it using the existing Phase 4 pipeline,
computes cosine similarity against any compatible gallery, and returns top-K
candidates with similarity-derived confidence scores.

IMPORTANT:
- This is an integration layer — it does NOT redesign the Re-ID pipeline
- Uses EXISTING Phase 4 preprocessing + OSNet pipeline from embed.py
- Uses EXISTING cosine similarity from similarity.py
- Uses EXISTING confidence scaling from confidence_scaling.py
- Does NOT modify embeddings_C01.json
- Does NOT modify Phase 1–4 production outputs

Usage (run in Google Colab with GPU):
    python ai_pipeline/reid/target_search.py \\
        --query dataset/query_photos/test_query_1.jpg \\
        --gallery dataset/new_video_test/embeddings_C01.json \\
        --top-k 10 \\
        --output dataset/new_video_test/target_search_results.json

Options:
    --query            Path to query image (required)
    --gallery          Path to gallery embeddings JSON (required; alias: --embeddings)
    --query-metadata   Path to query metadata JSON (optional, for source crop exclusion)
    --top-k            Number of candidates to return (default: 5)
    --output           Path to output JSON (default: stdout)
    --format           Output format: 'full' (default) or 'backend'
    --config           Path to config.yaml (default: ai_pipeline/config.yaml)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch

# Add repo root to path for imports
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from ai_pipeline.reid.embed import (
    _TRANSFORM,
    load_config,
    load_model,
    select_device,
)
from ai_pipeline.reid.similarity import cosine_similarity
from ai_pipeline.reid.confidence_scaling import similarity_to_confidence


# ---------------------------------------------------------------------------
# Query metadata loading
# ---------------------------------------------------------------------------

def load_query_metadata(metadata_path: Path, query_path: Path) -> Optional[dict]:
    """
    Load query metadata and find the entry matching the query path.

    Returns the metadata dict if found, None otherwise.
    """
    if not metadata_path.exists():
        print(f"[INFO] Query metadata not found: {metadata_path}")
        return None

    try:
        with open(metadata_path) as f:
            queries = json.load(f)
    except json.JSONDecodeError:
        print(f"[ERROR] Invalid JSON in {metadata_path}")
        return None

    if not isinstance(queries, list):
        print(f"[ERROR] Expected JSON list in {metadata_path}")
        return None

    # Find the entry matching the query path
    for entry in queries:
        if entry.get("query_path") == str(query_path):
            return entry

    print(f"[INFO] No metadata entry found for query: {query_path}")
    return None


# ---------------------------------------------------------------------------
# Query embedding
# ---------------------------------------------------------------------------

def embed_query(
    query_path: Path,
    model: torch.nn.Module,
    device: torch.device,
) -> list[float]:
    """
    Embed a single query image using the existing Phase 4 preprocessing pipeline.

    Args:
        query_path: Path to query image
        model: Loaded OSNet model
        device: torch device (cuda/cpu)

    Returns:
        512-dimensional embedding as a Python list of floats

    Raises:
        FileNotFoundError: If query image does not exist
        RuntimeError: If image cannot be loaded or embedded
    """
    from PIL import Image, UnidentifiedImageError

    if not query_path.exists():
        raise FileNotFoundError(f"Query image not found: {query_path}")

    # Load and preprocess using EXACT Phase 4 pipeline
    try:
        img = Image.open(query_path).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise RuntimeError(f"Failed to load query image: {e}")

    # Apply Phase 4 preprocessing (256x128 resize, ImageNet normalization)
    tensor = _TRANSFORM(img)  # [3, 256, 128]

    # Add batch dimension
    tensor = tensor.unsqueeze(0).to(device)  # [1, 3, 256, 128]

    # Embed using OSNet
    with torch.no_grad():
        embedding = model(tensor)  # [1, 512]

    # Convert to Python list
    embedding_list = embedding.cpu().squeeze(0).tolist()

    return embedding_list


# ---------------------------------------------------------------------------
# Gallery loading
# ---------------------------------------------------------------------------

def load_gallery(embeddings_path: Path) -> list[dict]:
    """
    Load the gallery embeddings from JSON.

    Args:
        embeddings_path: Path to embeddings_C01.json

    Returns:
        List of embedding records

    Raises:
        FileNotFoundError: If embeddings file does not exist
        ValueError: If embeddings file is invalid
    """
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    with open(embeddings_path) as f:
        gallery = json.load(f)

    if not isinstance(gallery, list):
        raise ValueError(f"Expected JSON list in {embeddings_path}")

    if len(gallery) == 0:
        raise ValueError(f"Empty gallery in {embeddings_path}")

    # Validate embedding dimension
    first_record = gallery[0]
    if "embedding" not in first_record:
        raise ValueError("Gallery record missing 'embedding' field")

    embedding_dim = len(first_record["embedding"])
    if embedding_dim != 512:
        raise ValueError(f"Expected 512-D embeddings, got {embedding_dim}")

    print(f"[INFO] Loaded gallery: {len(gallery)} crops, {embedding_dim}-D embeddings")

    return gallery


# ---------------------------------------------------------------------------
# Similarity computation and ranking
# ---------------------------------------------------------------------------

def compute_similarities(
    query_embedding: list[float],
    gallery: list[dict],
    source_crop_path: Optional[str] = None,
) -> list[dict]:
    """
    Compute cosine similarity between query and all gallery crops.

    Args:
        query_embedding: 512-D query embedding
        gallery: List of gallery embedding records
        source_crop_path: If provided, exclude this crop from results

    Returns:
        List of dicts with similarity and metadata for each gallery crop
    """
    results = []
    excluded_count = 0

    for record in gallery:
        crop_path = record.get("crop_path")

        # Exclude source crop if specified
        if source_crop_path and crop_path == source_crop_path:
            excluded_count += 1
            continue

        gallery_embedding = record.get("embedding")
        if not gallery_embedding:
            continue

        # Compute cosine similarity
        similarity = cosine_similarity(query_embedding, gallery_embedding)

        results.append({
            "similarity": similarity,
            "camera_id": record.get("camera_id"),
            "timestamp": record.get("timestamp"),
            "crop_path": crop_path,
            "track_id": record.get("track_id"),
            "frame": record.get("frame"),
        })

    if excluded_count > 0:
        print(f"[INFO] Excluded {excluded_count} source crop(s) from results")

    return results


# ---------------------------------------------------------------------------
# Ranking and confidence conversion
# ---------------------------------------------------------------------------

def rank_and_convert(
    results: list[dict],
    top_k: int,
) -> list[dict]:
    """
    Rank results by similarity descending, convert to confidence, return top-K.

    Args:
        results: List of similarity results
        top_k: Maximum number of candidates to return

    Returns:
        Top-K candidates with confidence scores
    """
    # Sort by similarity descending
    ranked = sorted(results, key=lambda x: x["similarity"], reverse=True)

    # Take top-K
    top_k_results = ranked[:top_k]

    # Convert similarity to confidence score
    for result in top_k_results:
        similarity = result["similarity"]
        confidence = similarity_to_confidence(similarity)
        result["confidence"] = round(confidence, 1)

    return top_k_results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_output(candidates: list[dict]) -> list[dict]:
    """
    Format candidates according to the backend contract.

    Returns only the fields required by the current backend contract:
    - camera_id
    - timestamp
    - confidence
    - crop_path
    """
    formatted = []
    for candidate in candidates:
        formatted.append({
            "camera_id": candidate["camera_id"],
            "timestamp": candidate["timestamp"],
            "confidence": candidate["confidence"],
            "crop_path": candidate["crop_path"],
        })
    return formatted


def format_full_output(
    candidates: list[dict],
    query_path: str,
    gallery_path: str,
    top_k: int,
    model_name: str,
    embedding_dim: int,
    gallery_size: int,
    threshold: float,
    status: str,
) -> dict:
    """
    Format results with full metadata for analysis and debugging.

    Includes rank, raw similarity, confidence, crop_path, camera_id,
    track_id, and frame for each candidate, plus query/gallery metadata.

    This is the comprehensive output format. For the minimal backend
    contract, use format_output() instead.
    """
    results = []
    for i, candidate in enumerate(candidates, start=1):
        results.append({
            "rank": i,
            "similarity": round(candidate["similarity"], 6),
            "confidence": candidate["confidence"],
            "crop_path": candidate.get("crop_path"),
            "camera_id": candidate.get("camera_id"),
            "track_id": candidate.get("track_id"),
            "frame": candidate.get("frame"),
        })

    return {
        "query": str(query_path),
        "gallery": str(gallery_path),
        "top_k": top_k,
        "model": model_name,
        "embedding_dim": embedding_dim,
        "gallery_size": gallery_size,
        "threshold": threshold,
        "status": status,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 5.8 Target Search — Query-to-gallery Re-ID"
    )
    parser.add_argument(
        "--query",
        type=Path,
        required=True,
        help="Path to query image"
    )
    parser.add_argument(
        "--gallery", "--embeddings",
        dest="gallery",
        type=Path,
        required=True,
        help="Path to gallery embeddings JSON"
    )
    parser.add_argument(
        "--query-metadata",
        type=Path,
        default=None,
        help="Path to query metadata JSON (for source crop exclusion)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of candidates to return (default: 5)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output JSON (default: stdout)"
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=["full", "backend"],
        default="full",
        help="Output format: 'full' (default) includes all metadata, "
             "'backend' uses minimal backend contract"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_REPO_ROOT / "ai_pipeline" / "config.yaml",
        help="Path to config.yaml"
    )

    args = parser.parse_args()

    # Validate top-k
    if args.top_k < 1:
        print("[ERROR] --top-k must be at least 1")
        sys.exit(1)

    print("=" * 60)
    print("Phase 5.8 Target Search — Query Matching")
    print("=" * 60)
    print(f"[INFO] Query: {args.query}")
    print(f"[INFO] Gallery: {args.gallery}")
    print(f"[INFO] Top-K: {args.top_k}")
    print(f"[INFO] Output format: {args.output_format}")

    # Load config
    cfg = load_config(args.config)
    embedding_cfg = cfg["reid"]["embedding"]
    model_name = embedding_cfg["model"]

    # Load no-match threshold from config
    try:
        threshold = cfg["reid"]["search"]["no_match_similarity_threshold"]
        print(f"[INFO] No-match similarity threshold: {threshold}")
    except KeyError:
        print("[WARNING] No-match threshold not found in config, using default 0.70")
        threshold = 0.70

    # Select device
    device = select_device()

    # Load model
    model = load_model(model_name, device)

    # Load gallery
    try:
        gallery = load_gallery(args.gallery)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # Load query metadata (if provided)
    query_metadata = None
    source_crop_path = None
    if args.query_metadata:
        query_metadata = load_query_metadata(args.query_metadata, args.query)
        if query_metadata and query_metadata.get("source_crop_excluded_from_gallery"):
            source_crop_path = query_metadata.get("source_crop_path")
            if source_crop_path:
                print(f"[INFO] Source crop exclusion enabled: {source_crop_path}")

    # Embed query
    print(f"[INFO] Embedding query...")
    try:
        query_embedding = embed_query(args.query, model, device)
        print(f"[INFO] Query embedding dimension: {len(query_embedding)}")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # Compute similarities
    print(f"[INFO] Computing similarities against {len(gallery)} gallery crops...")
    results = compute_similarities(query_embedding, gallery, source_crop_path)

    if len(results) == 0:
        print("[ERROR] No valid results after source crop exclusion")
        sys.exit(1)

    # Rank and convert
    print(f"[INFO] Ranking and converting to confidence scores...")
    top_candidates = rank_and_convert(results, args.top_k)

    # Determine status based on threshold
    if len(top_candidates) > 0:
        best_similarity = top_candidates[0]["similarity"]
        print(f"[INFO] Best candidate similarity: {best_similarity:.4f}")

        if best_similarity < threshold:
            print(f"[INFO] Best similarity below threshold ({threshold}), returning no_confident_match")
            status = "no_confident_match"
        else:
            print(f"[INFO] Best similarity meets/exceeds threshold, returning candidates")
            status = "matches_found"
    else:
        print("[INFO] No candidates available, returning no_confident_match")
        status = "no_confident_match"

    # Build output based on format
    active_candidates = top_candidates if status == "matches_found" else []

    if args.output_format == "full":
        output = format_full_output(
            candidates=active_candidates,
            query_path=str(args.query),
            gallery_path=str(args.gallery),
            top_k=args.top_k,
            model_name=model_name,
            embedding_dim=512,
            gallery_size=len(gallery),
            threshold=threshold,
            status=status,
        )
        result_count = len(output["results"])
    else:
        formatted = format_output(active_candidates)
        output = {
            "status": status,
            "candidates": formatted,
        }
        result_count = len(output["candidates"])

    # Output
    output_json = json.dumps(output, indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"[INFO] Results written to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(output_json)

    print(f"\n[INFO] Returned {result_count} candidates (top-{args.top_k} requested)")
    print("[INFO] Done")


if __name__ == "__main__":
    main()
