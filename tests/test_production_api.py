import io
import wave
import asyncio
import numpy as np
from fastapi.testclient import TestClient

import server
import engine


def _patch_engine(monkeypatch):
    monkeypatch.setattr(engine, "MODEL_LOADED", True)
    monkeypatch.setattr(engine, "WARMUP_COMPLETED", True)
    monkeypatch.setattr(engine, "get_available_voices", lambda: ["Jasper", "Bella"])
    monkeypatch.setattr(engine, "get_all_accepted_voices", lambda: ["Jasper", "Bella"])
    monkeypatch.setattr(engine, "get_default_voice", lambda: "Jasper")
    monkeypatch.setattr(engine, "get_model_info", lambda: {"repo_id": "KittenML/kitten-tts-nano-0.8-int8", "device": "cpu"})
    monkeypatch.setattr(engine, "get_onnx_provider_info", lambda: {"active": "CPUExecutionProvider"})
    monkeypatch.setattr(engine, "synthesize", lambda text, voice, speed=1.0: (np.zeros(2400, dtype=np.float32), 24000))


def _client(monkeypatch):
    monkeypatch.setenv("TTS_API_KEY", "secret")
    monkeypatch.setenv("ENABLE_MANAGEMENT_ENDPOINTS", "false")
    monkeypatch.setenv("MAX_TTS_TEXT_CHARS", "50")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "1000")
    _patch_engine(monkeypatch)
    return TestClient(server.app)


def test_health_does_not_require_auth(monkeypatch):
    c = _client(monkeypatch)
    assert c.get("/health").status_code == 200
    r = c.get("/api/tts/health")
    assert r.status_code == 200
    assert r.json()["default_voice"] == "Jasper"


def test_tts_rejects_missing_and_accepts_valid_key(monkeypatch):
    c = _client(monkeypatch)
    assert c.post("/tts", json={"text": "Hello", "voice": "Jasper"}).status_code == 401
    r = c.post("/tts", headers={"Authorization": "Bearer secret"}, json={"text": "Hello", "voice": "Jasper"})
    assert r.status_code == 200
    assert r.headers["x-tts-voice"] == "Jasper"


def test_api_key_header_alias(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/api/tts/speak", headers={"X-API-Key": "secret"}, json={"text": "Welcome to Pathwisse", "voice": "default", "stream": True})
    assert r.status_code == 200
    assert r.content.count(b"RIFF") == 1
    wave.open(io.BytesIO(r.content)).close()


def test_management_blocked(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/api/unload", headers={"X-API-Key": "secret"})
    assert r.status_code == 403


def test_validation_errors(monkeypatch):
    c = _client(monkeypatch)
    h = {"X-API-Key": "secret"}
    assert c.post("/api/tts/speak", headers=h, json={"text": ""}).status_code == 400
    assert c.post("/api/tts/speak", headers=h, json={"text": "   "}).status_code == 400
    assert c.post("/api/tts/speak", headers=h, json={"text": "x" * 51}).status_code == 413
    assert c.post("/api/tts/speak", headers=h, json={"text": "Hello", "voice": "Nope"}).status_code == 400
    assert c.post("/tts", headers=h, json={"voice": "Jasper"}).status_code == 422


def test_openai_endpoint_and_latency_headers(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/v1/audio/speech", headers={"X-API-Key": "secret"}, json={"model": "kitten", "input": "Hello", "voice": "Jasper", "response_format": "wav"})
    assert r.status_code == 200
    assert "x-tts-total-time-ms" in r.headers
    assert "x-tts-inference-time-ms" in r.headers


def test_concurrent_requests_and_health(monkeypatch):
    c = _client(monkeypatch)
    async def one():
        return await asyncio.to_thread(c.post, "/api/tts/speak", headers={"X-API-Key": "secret"}, json={"text": "Hello"})
    async def run():
        responses = await asyncio.gather(*[one() for _ in range(5)])
        health = c.get("/health")
        return responses, health
    responses, health = asyncio.run(run())
    assert all(r.status_code == 200 for r in responses)
    assert health.status_code == 200
