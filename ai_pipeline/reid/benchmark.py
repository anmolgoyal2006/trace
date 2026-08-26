"""
benchmark.py — Phase 4.7
Re-ID Performance Benchmark for TRACE.

Measures the actual time required for OSNet to process the complete 353-crop
TRACE dataset on both GPU and CPU.

Usage:
    python ai_pipeline/reid/benchmark.py \\
        --metadata dataset/crops_metadata_phase3_final.json \\
        --crop-dir dataset/crops_phase3_final \\
        --batch-size 32 \\
        --device cpu \\
        --runs 3

    python ai_pipeline/reid/benchmark.py \\
        --metadata dataset/crops_metadata_phase3_final.json \\
        --crop-dir dataset/crops_phase3_final \\
        --batch-size 32 \\
        --device cuda \\
        --runs 3

The script does NOT modify production embeddings (dataset/embeddings_C01.json).
"""

import argparse
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import torch
import yaml
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT   = Path(__file__).resolve().parents[2]   # .../Trace/
_CONFIG_PATH = _REPO_ROOT / "ai_pipeline" / "config.yaml"
_BENCHMARK_DIR = _REPO_ROOT / "dataset" / "reid_benchmark"

# ---------------------------------------------------------------------------
# ImageNet normalisation — identical to Phase 4.4 verified preprocessing
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# OSNet expected input: 256 (height) × 128 (width), RGB
INPUT_HEIGHT = 256
INPUT_WIDTH  = 128

_TRANSFORM = transforms.Compose([
    transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH)),
    transforms.ToTensor(),                                      # → [0, 1] float32
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# Timing result structure
# ---------------------------------------------------------------------------

class BenchmarkRun(NamedTuple):
    """Results from a single benchmark run."""
    run_number: int
    total_time: float           # seconds
    model_load_time: float      # seconds
    warmup_time: float          # seconds
    inference_time: float       # seconds (actual forward pass)
    preprocessing_time: float   # seconds (image loading + transforms)
    crops_processed: int
    crops_per_second: float
    ms_per_crop: float

    def to_dict(self) -> dict:
        return {
            "run_number": self.run_number,
            "total_time": self.total_time,
            "model_load_time": self.model_load_time,
            "warmup_time": self.warmup_time,
            "inference_time": self.inference_time,
            "preprocessing_time": self.preprocessing_time,
            "crops_processed": self.crops_processed,
            "crops_per_second": self.crops_per_second,
            "ms_per_crop": self.ms_per_crop,
        }


class BenchmarkSummary(NamedTuple):
    """Aggregated statistics across all runs."""
    mean_total_time: float
    median_total_time: float
    min_total_time: float
    max_total_time: float
    mean_inference_time: float
    mean_crops_per_second: float
    mean_ms_per_crop: float
    total_runs: int

    def to_dict(self) -> dict:
        return {
            "mean_total_time": self.mean_total_time,
            "median_total_time": self.median_total_time,
            "min_total_time": self.min_total_time,
            "max_total_time": self.max_total_time,
            "mean_inference_time": self.mean_inference_time,
            "mean_crops_per_second": self.mean_crops_per_second,
            "mean_ms_per_crop": self.mean_ms_per_crop,
            "total_runs": self.total_runs,
        }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """Return the full config dict from config.yaml."""
    if not config_path.exists():
        sys.exit(f"[ERROR] Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_embedding_config(cfg: dict) -> dict:
    """Extract and validate the reid.embedding block."""
    try:
        return cfg["reid"]["embedding"]
    except KeyError as e:
        sys.exit(f"[ERROR] Missing config key: {e}. "
                 "Ensure reid.embedding exists in config.yaml.")


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def select_device(device_arg: str) -> torch.device:
    """
    Select device based on CLI argument.
    
    Forces CPU if device_arg is 'cpu', otherwise uses CUDA if available.
    """
    if device_arg == "cpu":
        device = torch.device("cpu")
        print(f"[INFO] Forced device    : {device}")
    elif device_arg == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[INFO] CUDA available    : True")
            print(f"[INFO] GPU name          : {torch.cuda.get_device_name(0)}")
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[INFO] GPU memory        : {mem_total:.1f} GB")
        else:
            print("[WARNING] CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        print(f"[INFO] Using device      : {device}")
    else:
        sys.exit(f"[ERROR] Invalid device argument: {device_arg}. Use 'cpu' or 'cuda'.")
    
    return device


def get_cpu_info() -> dict:
    """Get CPU information without installing extra packages."""
    info = {
        "processor": platform.processor(),
        "logical_cpus": None,
    }
    try:
        import os
        info["logical_cpus"] = os.cpu_count()
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: torch.device) -> tuple[torch.nn.Module, float]:
    """
    Load pretrained OSNet from torchreid and set to eval mode.
    
    Returns (model, load_time_seconds).
    """
    try:
        import torchreid
    except ImportError:
        sys.exit(
            "[ERROR] torchreid is not installed.\n"
            "Install it with: pip install torchreid\n"
            "Or on Colab: !pip install torchreid"
        )

    print(f"\n[INFO] Loading {model_name} with pretrained weights...")
    t_start = time.perf_counter()
    
    model = torchreid.models.build_model(
        name=model_name,
        num_classes=1000,
        pretrained=True,
    )
    model.to(device)
    model.eval()
    
    t_elapsed = time.perf_counter() - t_start
    print(f"[INFO] Model loaded in {t_elapsed:.3f}s on {device}")
    
    return model, t_elapsed


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def load_metadata(metadata_path: Path) -> list[dict]:
    """Load and return the Phase 3 final crop metadata list."""
    if not metadata_path.exists():
        sys.exit(f"[ERROR] Metadata file not found: {metadata_path}")
    with open(metadata_path) as f:
        records = json.load(f)
    if not isinstance(records, list):
        sys.exit(f"[ERROR] Expected a JSON list in {metadata_path}, got {type(records)}")
    print(f"[INFO] Loaded metadata   : {len(records)} records from {metadata_path}")
    return records


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(image_path: Path) -> torch.Tensor:
    """
    Load and preprocess a single crop image.
    
    Returns tensor of shape [3, INPUT_HEIGHT, INPUT_WIDTH].
    """
    img = Image.open(image_path).convert("RGB")
    tensor = _TRANSFORM(img)
    return tensor


# ---------------------------------------------------------------------------
# Benchmark pipeline
# ---------------------------------------------------------------------------

def run_benchmark(
    metadata_path: Path,
    crop_dir: Path,
    batch_size: int,
    model_name: str,
    device: torch.device,
    num_runs: int,
) -> tuple[list[BenchmarkRun], BenchmarkSummary, dict]:
    """
    Run the complete benchmark pipeline.
    
    Returns (runs, summary, environment_info).
    """
    
    # Load metadata once
    metadata = load_metadata(metadata_path)
    total_crops = len(metadata)
    
    # Verify crop directory
    if not crop_dir.exists():
        sys.exit(f"[ERROR] Crop directory not found: {crop_dir}")
    
    # Collect crop file paths
    crop_paths = []
    for rec in metadata:
        # Extract filename from the crop_path and construct full path using crop_dir
        crop_filename = Path(rec["crop_path"]).name
        img_path = crop_dir / crop_filename
        if not img_path.exists():
            sys.exit(f"[ERROR] Crop file not found: {img_path}")
        crop_paths.append(img_path)
    
    print(f"[INFO] Crop directory    : {crop_dir}")
    print(f"[INFO] Total crops       : {total_crops}")
    print(f"[INFO] Batch size        : {batch_size}")
    print(f"[INFO] Number of runs    : {num_runs}")
    
    # Load model once (measured separately)
    model, model_load_time = load_model(model_name, device)
    
    # Determine embedding dimension
    print(f"\n[INFO] Probing model output dimension...")
    with torch.no_grad():
        dummy = torch.zeros(1, 3, INPUT_HEIGHT, INPUT_WIDTH, device=device)
        probe = model(dummy)
        embedding_dim = probe.shape[1]
    print(f"[INFO] Model output dim  : {embedding_dim}")
    
    # Warm-up (not counted in main timing)
    print(f"\n[INFO] Performing warm-up...")
    t_warmup_start = time.perf_counter()
    
    # Process a small batch for warm-up
    warmup_size = min(4, len(crop_paths))
    warmup_tensors = []
    for i in range(warmup_size):
        warmup_tensors.append(preprocess_image(crop_paths[i]))
    
    if warmup_tensors:
        warmup_batch = torch.stack(warmup_tensors).to(device)
        with torch.no_grad():
            _ = model(warmup_batch)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    warmup_time = time.perf_counter() - t_warmup_start
    print(f"[INFO] Warm-up completed in {warmup_time:.3f}s")
    
    # Main benchmark runs
    runs: list[BenchmarkRun] = []
    
    for run_idx in range(1, num_runs + 1):
        print(f"\n[INFO] === Run {run_idx}/{num_runs} ===")
        
        # Preprocessing timing
        t_preprocess_start = time.perf_counter()
        all_tensors = []
        for crop_path in crop_paths:
            tensor = preprocess_image(crop_path)
            all_tensors.append(tensor)
        preprocessing_time = time.perf_counter() - t_preprocess_start
        
        # Inference timing
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        t_inference_start = time.perf_counter()
        
        n_batches = math.ceil(total_crops / batch_size)
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_crops)
            batch_tensors = all_tensors[batch_start:batch_end]
            
            batch_tensor = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                _ = model(batch_tensor)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        inference_time = time.perf_counter() - t_inference_start
        
        # Total time for this run
        total_time = preprocessing_time + inference_time
        
        # Calculate metrics
        crops_per_second = total_crops / total_time
        ms_per_crop = 1000 * total_time / total_crops
        
        run_result = BenchmarkRun(
            run_number=run_idx,
            total_time=total_time,
            model_load_time=model_load_time,
            warmup_time=warmup_time,
            inference_time=inference_time,
            preprocessing_time=preprocessing_time,
            crops_processed=total_crops,
            crops_per_second=crops_per_second,
            ms_per_crop=ms_per_crop,
        )
        runs.append(run_result)
        
        print(f"[INFO] Total time        : {total_time:.3f}s")
        print(f"[INFO] Preprocessing     : {preprocessing_time:.3f}s")
        print(f"[INFO] Inference         : {inference_time:.3f}s")
        print(f"[INFO] Crops/sec         : {crops_per_second:.1f}")
        print(f"[INFO] ms/crop           : {ms_per_crop:.2f}")
    
    # Calculate summary statistics
    total_times = [r.total_time for r in runs]
    inference_times = [r.inference_time for r in runs]
    crops_per_sec_values = [r.crops_per_second for r in runs]
    ms_per_crop_values = [r.ms_per_crop for r in runs]
    
    summary = BenchmarkSummary(
        mean_total_time=statistics.mean(total_times),
        median_total_time=statistics.median(total_times),
        min_total_time=min(total_times),
        max_total_time=max(total_times),
        mean_inference_time=statistics.mean(inference_times),
        mean_crops_per_second=statistics.mean(crops_per_sec_values),
        mean_ms_per_crop=statistics.mean(ms_per_crop_values),
        total_runs=num_runs,
    )
    
    # Environment info
    env_info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "model": model_name,
        "embedding_dim": embedding_dim,
        "crop_count": total_crops,
        "batch_size": batch_size,
    }
    
    if device.type == "cuda":
        env_info["gpu_name"] = torch.cuda.get_device_name(0)
        env_info["cuda_version"] = torch.version.cuda
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        env_info["gpu_memory_gb"] = round(gpu_mem, 2)
    else:
        cpu_info = get_cpu_info()
        env_info["cpu_processor"] = cpu_info["processor"]
        if cpu_info["logical_cpus"]:
            env_info["cpu_logical_cpus"] = cpu_info["logical_cpus"]
    
    return runs, summary, env_info


# ---------------------------------------------------------------------------
# Result storage
# ---------------------------------------------------------------------------

def save_benchmark_results(
    runs: list[BenchmarkRun],
    summary: BenchmarkSummary,
    env_info: dict,
    output_path: Path,
) -> None:
    """Save benchmark results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "environment": env_info,
        "model_load_time": runs[0].model_load_time,  # Same for all runs
        "warmup_time": runs[0].warmup_time,  # Same for all runs
        "runs": [r.to_dict() for r in runs],
        "summary": summary.to_dict(),
    }
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n[INFO] Benchmark results saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRACE Phase 4.7 — Re-ID Performance Benchmark"
    )
    parser.add_argument(
        "--metadata",
        required=True,
        type=Path,
        help="Path to crops_metadata_phase3_final.json",
    )
    parser.add_argument(
        "--crop-dir",
        required=True,
        type=Path,
        help="Directory containing the Phase 3 final crop images",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Crops per forward pass (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run benchmark on (default: cpu)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of benchmark runs (default: 3)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_CONFIG_PATH,
        help=f"Path to config.yaml (default: {_CONFIG_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_BENCHMARK_DIR / "performance_benchmark.json",
        help=f"Output path for benchmark results (default: {_BENCHMARK_DIR / 'performance_benchmark.json'})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    cfg = load_config(args.config)
    emb_cfg = get_embedding_config(cfg)
    
    model_name = emb_cfg.get("model", "osnet_x1_0")
    
    print("=" * 60)
    print("TRACE — Phase 4.7: Re-ID Performance Benchmark")
    print("=" * 60)
    print(f"[INFO] PyTorch version   : {torch.__version__}")
    print(f"[INFO] Model             : {model_name}")
    print(f"[INFO] Metadata          : {args.metadata}")
    print(f"[INFO] Crop directory    : {args.crop_dir}")
    print(f"[INFO] Batch size        : {args.batch_size}")
    print(f"[INFO] Device argument   : {args.device}")
    print(f"[INFO] Number of runs    : {args.runs}")
    print(f"[INFO] Output path       : {args.output}")
    print()
    
    device = select_device(args.device)
    
    runs, summary, env_info = run_benchmark(
        metadata_path=args.metadata,
        crop_dir=args.crop_dir,
        batch_size=args.batch_size,
        model_name=model_name,
        device=device,
        num_runs=args.runs,
    )
    
    save_benchmark_results(runs, summary, env_info, args.output)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Environment          : {env_info['device']}")
    if env_info['device'] == 'cuda':
        print(f"GPU                  : {env_info.get('gpu_name', 'N/A')}")
    else:
        print(f"CPU                  : {env_info.get('cpu_processor', 'N/A')}")
        if env_info.get('cpu_logical_cpus'):
            print(f"Logical CPUs         : {env_info['cpu_logical_cpus']}")
    print(f"Crops                : {env_info['crop_count']}")
    print(f"Batch size           : {env_info['batch_size']}")
    print(f"Model                : {env_info['model']}")
    print(f"Embedding dim        : {env_info['embedding_dim']}")
    print(f"Model load time      : {runs[0].model_load_time:.3f}s")
    print(f"Warm-up time         : {runs[0].warmup_time:.3f}s")
    print()
    print(f"Runs                 : {summary.total_runs}")
    print(f"Mean total time      : {summary.mean_total_time:.3f}s")
    print(f"Median total time    : {summary.median_total_time:.3f}s")
    print(f"Min total time       : {summary.min_total_time:.3f}s")
    print(f"Max total time       : {summary.max_total_time:.3f}s")
    print(f"Mean inference time  : {summary.mean_inference_time:.3f}s")
    print(f"Mean crops/sec       : {summary.mean_crops_per_second:.1f}")
    print(f"Mean ms/crop         : {summary.mean_ms_per_crop:.2f}")
    print("=" * 60)
    
    print("\n[DONE] Phase 4.7 benchmark complete.")


if __name__ == "__main__":
    main()
