"""
test_query_face_embedding.py
Regression: the query face signal must survive a successful face detection.

The "face detected" log line used an invalid f-string format spec
({x:.3f if x else 'N/A'}), which raised ValueError exactly when a face was
found, so the face signal was dropped for every query photo containing a face.

These tests stub the SCRFD / ArcFace models (no weights needed).

Run:
    pytest ai_pipeline/reid/test_query_face_embedding.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import backend.app.services.pipeline_service as ps
from backend.app.config import settings

# ===========================================================================
# Query face embedding — detected face must be returned, not crash on logging
# ===========================================================================

class TestQueryFaceEmbedding:
    """
    Regression: the "face detected" log line used an invalid f-string format
    spec ({x:.3f if x else 'N/A'}), raising ValueError exactly when a face was
    found — so the face signal was dropped for every query with a face.
    """

    def _stub_face(self, monkeypatch, tmp_path, result):
        import ai_pipeline.reid.face_embed as fe
        det, rec = tmp_path / "det.onnx", tmp_path / "rec.onnx"
        det.write_bytes(b"x"); rec.write_bytes(b"x")
        monkeypatch.setattr(type(settings), "face_det_model_path", property(lambda s: det), raising=False)
        monkeypatch.setattr(type(settings), "face_rec_model_path", property(lambda s: rec), raising=False)
        monkeypatch.setattr(settings, "face_det_model", "det.onnx", raising=False)
        monkeypatch.setattr(settings, "face_rec_model", "rec.onnx", raising=False)
        monkeypatch.setattr(fe, "SCRFDDetector", lambda *a, **k: object())
        monkeypatch.setattr(fe, "ArcFaceRecogniser", lambda *a, **k: object())
        monkeypatch.setattr(fe, "process_crop", lambda *a, **k: result)

    @pytest.mark.parametrize("conf", [0.787, None])
    def test_detected_face_is_returned(self, monkeypatch, tmp_path, conf):
        emb = [0.1] * 512
        self._stub_face(monkeypatch, tmp_path,
                        {"face_detected": True, "face_embedding": emb, "face_confidence": conf})
        assert ps._embed_query_face(tmp_path / "q.jpg") == (emb, conf)

    def test_no_face_returns_none(self, monkeypatch, tmp_path):
        self._stub_face(monkeypatch, tmp_path, {"face_detected": False})
        assert ps._embed_query_face(tmp_path / "q.jpg") == (None, None)
