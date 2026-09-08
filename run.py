"""
run.py — Start the unified Trace server.

Usage:
    python run.py              # default: http://127.0.0.1:8000
    python run.py --port 9000  # custom port
    python run.py --host 0.0.0.0 --port 8000  # bind all interfaces
"""

import argparse
import sys
from pathlib import Path

# Ensure the repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(description="Trace — Unified Surveillance Intelligence")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Trace - Unified Surveillance Intelligence")
    print("=" * 60)
    print(f"  Dashboard  ->  http://{args.host}:{args.port}")
    print(f"  API Docs   ->  http://{args.host}:{args.port}/api/docs")
    print(f"  WebSocket  ->  ws://{args.host}:{args.port}/ws")
    print("=" * 60)

    import uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
