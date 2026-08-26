"""
conftest.py — ai_pipeline/reid
Registers pytest CLI options shared across reid test modules.
"""

from pathlib import Path

_REPO_ROOT         = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDINGS = _REPO_ROOT / "dataset" / "embeddings_C01.json"
DEFAULT_METADATA   = _REPO_ROOT / "dataset" / "crops_metadata_phase3_final.json"


def pytest_addoption(parser):
    # Guard against duplicate registration when multiple conftest files are
    # loaded (e.g. root conftest + this one).
    try:
        parser.addoption(
            "--embeddings",
            default=str(DEFAULT_EMBEDDINGS),
            help=f"Path to embeddings JSON (default: {DEFAULT_EMBEDDINGS})",
        )
    except ValueError:
        pass  # already registered by a parent conftest

    try:
        parser.addoption(
            "--metadata",
            default=str(DEFAULT_METADATA),
            help=f"Path to Phase 3 metadata JSON (default: {DEFAULT_METADATA})",
        )
    except ValueError:
        pass  # already registered by a parent conftest
