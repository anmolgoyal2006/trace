"""
test_query_kpr_subprocess.py
Tests that the query KPR embedding runs through the same embed_kpr.py
subprocess as the upload pipeline.

KPR ships its own fork of `torchreid`; the standard torchreid is already
imported server-wide for OSNet, so importing the fork in-process failed with
"No module named 'torchreid.utils'" and the KPR signal was silently dropped
for every query.

These tests stub the subprocess (no GPU, weights, or KPR install needed).

Run:
    pytest ai_pipeline/reid/test_query_kpr_subprocess.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import backend.app.routers.upload as upload_module
import backend.app.services.pipeline_service as ps
from backend.app.config import settings

_RECORD = {
    "crop_path": "ignored", "camera_id": "QUERY", "track_id": 0,
    "holistic_embedding": [0.5] * 4,
    "part_embeddings":    [[0.1] * 4, [0.2] * 4],
    "part_visibility":    [0.9, 0.1],
}


@pytest.fixture
def kpr_configured(monkeypatch, tmp_path):
    weights, config = tmp_path / "kpr.pth.tar", tmp_path / "kpr.yaml"
    weights.write_bytes(b"w")
    config.write_text("c")
    monkeypatch.setattr(settings, "kpr_weights_path", str(weights), raising=False)
    monkeypatch.setattr(settings, "kpr_config_path", str(config), raising=False)
    monkeypatch.setattr(type(settings), "kpr_weights_path_abs", property(lambda s: weights), raising=False)
    monkeypatch.setattr(type(settings), "kpr_config_path_abs", property(lambda s: config), raising=False)
    img = tmp_path / "photos" / "query.jpg"
    img.parent.mkdir()
    img.write_bytes(b"jpg")
    return img, weights, config


def _stub_subprocess(monkeypatch, records, calls):
    def fake(metadata_path, crop_dir, output_path, kpr_weights, kpr_config):
        calls.append({
            "metadata":   json.loads(Path(metadata_path).read_text()),
            "crop_dir":   crop_dir,
            "weights":    kpr_weights,
            "config":     kpr_config,
        })
        if records is not None:
            Path(output_path).write_text(json.dumps(records))
    monkeypatch.setattr(upload_module, "_run_kpr_embed", fake)


class TestQueryKprSubprocess:

    def test_returns_embedding_from_subprocess_output(self, monkeypatch, kpr_configured):
        img, weights, config = kpr_configured
        calls: list[dict] = []
        _stub_subprocess(monkeypatch, [_RECORD], calls)

        out = ps._embed_query_kpr(img)

        assert out == {
            "holistic_embedding": _RECORD["holistic_embedding"],
            "part_embeddings":    _RECORD["part_embeddings"],
            "part_visibility":    _RECORD["part_visibility"],
        }
        [call] = calls
        assert call["metadata"][0]["crop_path"] == str(img.resolve())
        assert call["crop_dir"] == img.resolve().parent
        assert (call["weights"], call["config"]) == (weights, config)

    def test_empty_output_returns_none(self, monkeypatch, kpr_configured):
        img, _, _ = kpr_configured
        _stub_subprocess(monkeypatch, [], [])
        assert ps._embed_query_kpr(img) is None

    def test_missing_output_returns_none(self, monkeypatch, kpr_configured):
        img, _, _ = kpr_configured
        _stub_subprocess(monkeypatch, None, [])
        assert ps._embed_query_kpr(img) is None

    def test_subprocess_failure_is_non_fatal(self, monkeypatch, kpr_configured):
        img, _, _ = kpr_configured
        def boom(*args, **kwargs):
            raise RuntimeError("KPR subprocess exited with code 1")
        monkeypatch.setattr(upload_module, "_run_kpr_embed", boom)
        assert ps._embed_query_kpr(img) is None

    def test_not_configured_skips_subprocess(self, monkeypatch, tmp_path):
        monkeypatch.setattr(settings, "kpr_weights_path", "", raising=False)
        calls: list[dict] = []
        _stub_subprocess(monkeypatch, [_RECORD], calls)
        assert ps._embed_query_kpr(tmp_path / "q.jpg") is None
        assert calls == []
