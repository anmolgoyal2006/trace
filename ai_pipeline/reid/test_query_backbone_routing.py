"""
test_query_backbone_routing.py
Tests that the query is embedded with the same body backbone as each camera
gallery (OSNet 512-d vs SOLIDER 768-d).

Before this fix the query was always OSNet (512-d). A camera whose gallery had
been rebuilt with SOLIDER (768-d) — which every upload does when SOLIDER
weights are configured — silently matched nothing ("No valid crops").

These tests:
  - Stub out the embedding models (no GPU, weights, or server needed).

Run:
    pytest ai_pipeline/reid/test_query_backbone_routing.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import backend.app.services.pipeline_service as ps
from backend.app.config import settings


@pytest.fixture
def stub_models(monkeypatch):
    calls: list[str] = []

    def osnet(path):
        calls.append("osnet")
        return [0.1] * 512

    def solider(path):
        calls.append("solider")
        return [0.2] * 768

    monkeypatch.setattr(ps.embedding_service, "embed_single", osnet)
    monkeypatch.setattr(ps.embedding_service, "embed_single_solider", solider)
    return calls


class TestGalleryDim:

    def test_first_record_with_embedding(self):
        assert ps._gallery_dim([{"embedding": None}, {"embedding": [0.0] * 768}]) == 768

    def test_empty_or_missing(self):
        assert ps._gallery_dim([]) is None
        assert ps._gallery_dim([{"track_id": 1}]) is None


class TestEmbedQueryForDim:

    def test_solider_dim_uses_solider(self, stub_models, tmp_path):
        emb = ps._embed_query_for_dim(settings.solider_embedding_dim, tmp_path / "q.jpg")
        assert len(emb) == 768
        assert stub_models == ["solider"]

    def test_osnet_dim_uses_osnet(self, stub_models, tmp_path):
        emb = ps._embed_query_for_dim(512, tmp_path / "q.jpg")
        assert len(emb) == 512
        assert stub_models == ["osnet"]

    def test_no_image_returns_none(self, stub_models):
        assert ps._embed_query_for_dim(768, None) is None
        assert stub_models == []

    def test_unknown_dim_returns_none(self, stub_models, tmp_path):
        assert ps._embed_query_for_dim(1024, tmp_path / "q.jpg") is None
        assert stub_models == []

    def test_backbone_failure_returns_none(self, monkeypatch, tmp_path):
        def broken(path):
            raise RuntimeError("SOLIDER weights not configured")
        monkeypatch.setattr(ps.embedding_service, "embed_single_solider", broken)
        assert ps._embed_query_for_dim(768, tmp_path / "q.jpg") is None


class TestSoliderEmbedderGuards:

    def test_missing_weights_raise_runtime_error(self, monkeypatch, tmp_path):
        from backend.app.services.embedding_service import EmbeddingService
        img = tmp_path / "q.jpg"
        img.write_bytes(b"not really a jpeg")
        svc = EmbeddingService()
        monkeypatch.setattr(settings, "solider_weights", tmp_path / "missing.pth")
        with pytest.raises(RuntimeError, match="SOLIDER weights"):
            svc.embed_single_solider(img)

    def test_missing_image_raises(self, tmp_path):
        from backend.app.services.embedding_service import EmbeddingService
        with pytest.raises(FileNotFoundError):
            EmbeddingService().embed_single_solider(tmp_path / "nope.jpg")
