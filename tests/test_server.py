import pytest
import os
from fastapi.testclient import TestClient
import subprocess

# Set environment variables for the "configured" test client.
os.environ["TTS_API_KEY"] = "test-key"
os.environ["ENABLE_MANAGEMENT_ENDPOINTS"] = "true"
os.environ["ENABLE_WEB_UI"] = "true"

from server import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Import safety (no torch / torchaudio required at import time)
# ---------------------------------------------------------------------------
def test_import_server_without_torchaudio():
    result = subprocess.run(
        ["python", "-c", "import server"], capture_output=True, text=True
    )
    assert result.returncode == 0, f"import server failed: {result.stderr}"


def test_import_utils_without_torchaudio():
    result = subprocess.run(
        ["python", "-c", "import utils"], capture_output=True, text=True
    )
    assert result.returncode == 0, f"import utils failed: {result.stderr}"


# ---------------------------------------------------------------------------
# Health endpoints remain public
# ---------------------------------------------------------------------------
def test_health_remains_available():
    response = client.get("/health")
    assert response.status_code in (200, 503)


def test_api_tts_health_remains_available():
    response = client.get("/api/tts/health")
    assert response.status_code in (200, 503)


# ---------------------------------------------------------------------------
# Auth: Bearer + X-API-Key, invalid/missing keys
# ---------------------------------------------------------------------------
def test_valid_bearer_accepted():
    response = client.post(
        "/api/tts/speak",
        headers={"Authorization": "Bearer test-key"},
        json={"text": "Hello", "voice": "default", "format": "wav"},
    )
    assert response.status_code != 401
    assert response.status_code != 403


def test_valid_x_api_key_accepted():
    response = client.post(
        "/api/tts/speak",
        headers={"X-API-Key": "test-key"},
        json={"text": "Hello", "voice": "default", "format": "wav"},
    )
    assert response.status_code != 401
    assert response.status_code != 403


def test_invalid_key_rejected():
    response = client.post(
        "/api/tts/speak",
        headers={"Authorization": "Bearer wrong-key"},
        json={"text": "Hello", "voice": "default", "format": "wav"},
    )
    assert response.status_code == 403


def test_missing_key_rejected():
    response = client.post(
        "/api/tts/speak",
        json={"text": "Hello", "voice": "default", "format": "wav"},
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Fail-closed when TTS_API_KEY is unset
# ---------------------------------------------------------------------------
def test_protected_endpoints_fail_closed_when_key_unset(monkeypatch):
    monkeypatch.delenv("TTS_API_KEY", raising=False)
    unauth = TestClient(app)
    for path, method in [
        ("/api/tts/speak", "post"),
        ("/v1/audio/speech", "post"),
        ("/tts", "post"),
        ("/api/unload", "post"),
        ("/save_settings", "post"),
        ("/reset_settings", "post"),
        ("/restart_server", "post"),
        ("/api/cancel-loading", "post"),
        ("/api/ui/initial-data", "get"),
    ]:
        if method == "post":
            resp = unauth.post(path, json={"text": "x", "voice": "default"})
        else:
            resp = unauth.get(path)
        assert resp.status_code in (
            401,
            403,
            503,
        ), f"{path} unexpectedly returned {resp.status_code}"


# ---------------------------------------------------------------------------
# Management endpoints blocked by default (flag off)
# ---------------------------------------------------------------------------
def test_management_blocked_when_flag_false(monkeypatch):
    monkeypatch.setenv("ENABLE_MANAGEMENT_ENDPOINTS", "false")
    response = client.post(
        "/api/unload", headers={"Authorization": "Bearer test-key"}
    )
    assert response.status_code == 403


def test_restart_server_blocked_when_flag_false(monkeypatch):
    monkeypatch.setenv("ENABLE_MANAGEMENT_ENDPOINTS", "false")
    response = client.post(
        "/restart_server", headers={"Authorization": "Bearer test-key"}
    )
    assert response.status_code == 403


def test_save_settings_blocked_when_flag_false(monkeypatch):
    monkeypatch.setenv("ENABLE_MANAGEMENT_ENDPOINTS", "false")
    response = client.post(
        "/save_settings",
        headers={"Authorization": "Bearer test-key"},
        json={},
    )
    assert response.status_code == 403


def test_reset_settings_blocked_when_flag_false(monkeypatch):
    monkeypatch.setenv("ENABLE_MANAGEMENT_ENDPOINTS", "false")
    response = client.post(
        "/reset_settings", headers={"Authorization": "Bearer test-key"}
    )
    assert response.status_code == 403


def test_cancel_loading_blocked_when_flag_false(monkeypatch):
    monkeypatch.setenv("ENABLE_MANAGEMENT_ENDPOINTS", "false")
    response = client.post(
        "/api/cancel-loading", headers={"Authorization": "Bearer test-key"}
    )
    assert response.status_code == 403


def test_management_blocked_without_key():
    response = client.post("/api/unload")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Web UI disabled by default
# ---------------------------------------------------------------------------
def test_web_ui_disabled_when_flag_false(monkeypatch):
    monkeypatch.setenv("ENABLE_WEB_UI", "false")
    response = client.get("/", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 403


def test_ui_initial_data_blocked_when_web_ui_false(monkeypatch):
    monkeypatch.setenv("ENABLE_WEB_UI", "false")
    response = client.get(
        "/api/ui/initial-data", headers={"Authorization": "Bearer test-key"}
    )
    assert response.status_code == 403


def test_ui_initial_data_blocked_without_key():
    response = client.get("/api/ui/initial-data")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# MP3 rejected with clear error
# ---------------------------------------------------------------------------
def test_mp3_rejected_on_tts(monkeypatch):
    monkeypatch.setenv("TTS_API_KEY", "test-key")
    response = client.post(
        "/api/tts/speak",
        headers={"Authorization": "Bearer test-key"},
        json={"text": "Hello", "voice": "default", "output_format": "mp3"},
    )
    assert response.status_code == 400
    assert "MP3 output is not supported" in response.json().get("detail", "")


def test_mp3_rejected_on_openai(monkeypatch):
    monkeypatch.setenv("TTS_API_KEY", "test-key")
    response = client.post(
        "/v1/audio/speech",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "kitten-tts",
            "input": "Hello world",
            "voice": "default",
            "response_format": "mp3",
        },
    )
    assert response.status_code == 400
    assert "MP3 output is not supported" in response.json().get("detail", "")


# ---------------------------------------------------------------------------
# /v1/audio/speech streaming: must be a single valid WAV describing the FULL
# concatenated audio (correct RIFF/data sizes, no truncation).
# ---------------------------------------------------------------------------
SR = 24000
CHUNK_FRAMES = SR // 4  # 0.25s per chunk


def _fake_synthesize_factory(monkeypatch, n_chunks_seen):
    import numpy as np
    import engine

    silence = np.zeros(CHUNK_FRAMES, dtype=np.float32)

    def fake_synthesize(text, voice, speed=None):
        n_chunks_seen[0] += 1
        return silence.copy(), SR

    monkeypatch.setattr(engine, "MODEL_LOADED", True)
    monkeypatch.setattr("engine.synthesize", fake_synthesize)


def _assert_full_wav(data, expected_chunks):
    import io
    import wave

    assert data[:4] == b"RIFF", "WAV must start with a RIFF header"
    assert data.count(b"RIFF") == 1, "WAV must contain exactly one RIFF header"
    with wave.open(io.BytesIO(data), "rb") as wf:
        assert wf.getnchannels() == 1, "expected mono"
        assert wf.getframerate() == SR
        total_frames = wf.getnframes()
        expected_frames = expected_chunks * CHUNK_FRAMES
        assert total_frames == expected_frames, (
            f"frame count {total_frames} should equal sum of chunks "
            f"({expected_frames}), not just the first chunk"
        )
        pcm = wf.readframes(total_frames)
        assert len(pcm) == total_frames * 2, "PCM bytes must match full frame count (16-bit)"
        assert len(pcm) > CHUNK_FRAMES * 2, "audio must be longer than a single chunk"


def test_openai_stream_wav_full_audio(monkeypatch):
    seen = [0]
    _fake_synthesize_factory(monkeypatch, seen)
    long_text = "Hello there. " * 60  # > 300 chars -> multi-chunk path
    response = client.post(
        "/v1/audio/speech",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "kitten-tts",
            "input": long_text,
            "voice": "default",
            "response_format": "wav",
        },
    )
    assert response.status_code == 200, response.text
    assert seen[0] > 1, "expected multiple synthesized chunks"
    _assert_full_wav(response.content, seen[0])


def test_openai_stream_wav_latency_headers(monkeypatch):
    seen = [0]
    _fake_synthesize_factory(monkeypatch, seen)
    long_text = "Hello there. " * 60
    response = client.post(
        "/v1/audio/speech",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "kitten-tts",
            "input": long_text,
            "voice": "default",
            "response_format": "wav",
        },
    )
    assert response.status_code == 200
    for h in ("X-TTS-Total-Time-Ms", "X-TTS-Inference-Time-Ms", "X-TTS-Model", "X-TTS-Voice"):
        assert h in response.headers, f"missing latency header {h}"


def test_tts_stream_wav_full_audio(monkeypatch):
    seen = [0]
    _fake_synthesize_factory(monkeypatch, seen)
    long_text = "Hello there. " * 60  # triggers split > 1 chunk
    response = client.post(
        "/api/tts/speak",
        headers={"Authorization": "Bearer test-key"},
        json={
            "text": long_text,
            "voice": "default",
            "format": "wav",
            "stream": True,
        },
    )
    assert response.status_code == 200, response.text
    assert seen[0] > 1, "expected multiple synthesized chunks"
    _assert_full_wav(response.content, seen[0])
    for h in ("X-TTS-Total-Time-Ms", "X-TTS-Inference-Time-Ms", "X-TTS-Model", "X-TTS-Voice"):
        assert h in response.headers, f"missing latency header {h}"


# ---------------------------------------------------------------------------
# CORS is safe (no wildcard with credentials)
# ---------------------------------------------------------------------------
def test_cors_no_wildcard_with_credentials():
    # The CORS middleware must not allow "*" origins together with credentials.
    cors_mw = None
    for mw in app.user_middleware:
        if getattr(mw, "cls", None) is not None and mw.cls.__name__ == "CORSMiddleware":
            cors_mw = mw
    assert cors_mw is not None, "CORSMiddleware not configured"
    allow_origins = cors_mw.kwargs.get("allow_origins")
    allow_credentials = cors_mw.kwargs.get("allow_credentials")
    assert allow_credentials is False, "allow_credentials must be False"
    assert "*" not in allow_origins, "Wildcard origin must not be used"
    assert "null" not in allow_origins, "Null origin must not be used"


# We can't fully test streamed WAV RIFF header in pytest easily because engine might not be loaded.
# But we can test if the route exists and requires auth.
def test_streamed_wav_auth():
    response = client.post(
        "/api/tts/speak",
        json={"text": "Hello", "voice": "default", "format": "wav", "stream": True},
    )
    assert response.status_code == 401
