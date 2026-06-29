"""
WO-02 HTTP API tests.
Uses FastAPI's TestClient (httpx under the hood); no live server required.
Detection tests skip when the ONNX model is absent (same guard as test_detection.py).
"""

import base64
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

FIXTURES = Path(__file__).parent / "fixtures"
MODEL_PATH = Path(__file__).parent.parent / "models" / "yolo11n.onnx"

VALID_KEY = "test-secret-key"
HEADERS = {"Authorization": f"Bearer {VALID_KEY}"}

# Encode the fixture image once for reuse
_BUS_B64 = base64.b64encode((FIXTURES / "bus.jpg").read_bytes()).decode()
BUS_DATA_URL = f"data:image/jpeg;base64,{_BUS_B64}"

MODEL_REQUIRES = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Model not exported — run: python scripts/export_model.py",
)


def _image_request(data_url: str, model: str = "yolo11n", stream: bool = False) -> dict:
    return {
        "model": model,
        "stream": stream,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": data_url}}],
            }
        ],
    }


@pytest.fixture(autouse=True)
def set_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", VALID_KEY)


@pytest.fixture()
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ---------------------------------------------------------------------------
# Health and models endpoints (no model needed)
# ---------------------------------------------------------------------------
class TestInfraEndpoints:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_models_list(self, client):
        r = client.get("/v1/models", headers=HEADERS)
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "list"
        ids = [m["id"] for m in body["data"]]
        assert "yolo11n" in ids

    def test_models_requires_auth(self, client):
        r = client.get("/v1/models")
        assert r.status_code == 401
        assert "error" in r.json()


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
class TestAuth:
    def test_missing_auth_header(self, client):
        r = client.post("/v1/chat/completions", json=_image_request(BUS_DATA_URL))
        assert r.status_code == 401
        assert "error" in r.json()

    def test_wrong_key(self, client):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer wrong-key"},
            json=_image_request(BUS_DATA_URL),
        )
        assert r.status_code == 401
        assert "error" in r.json()

    def test_malformed_auth_scheme(self, client):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Basic dXNlcjpwYXNz"},
            json=_image_request(BUS_DATA_URL),
        )
        assert r.status_code == 401
        assert "error" in r.json()


# ---------------------------------------------------------------------------
# Request validation (no model needed)
# ---------------------------------------------------------------------------
class TestRequestValidation:
    def test_unknown_model(self, client):
        r = client.post(
            "/v1/chat/completions",
            headers=HEADERS,
            json=_image_request(BUS_DATA_URL, model="gpt-4o"),
        )
        assert r.status_code == 404
        assert "error" in r.json()

    def test_no_image(self, client):
        body = {
            "model": "yolo11n",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        }
        r = client.post("/v1/chat/completions", headers=HEADERS, json=body)
        assert r.status_code == 400
        assert "error" in r.json()

    def test_two_images(self, client):
        body = {
            "model": "yolo11n",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": BUS_DATA_URL}},
                        {"type": "image_url", "image_url": {"url": BUS_DATA_URL}},
                    ],
                }
            ],
        }
        r = client.post("/v1/chat/completions", headers=HEADERS, json=body)
        assert r.status_code == 400
        assert "error" in r.json()

    def test_remote_url_rejected(self, client):
        body = {
            "model": "yolo11n",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
                    ],
                }
            ],
        }
        r = client.post("/v1/chat/completions", headers=HEADERS, json=body)
        assert r.status_code == 400
        err = r.json()["error"]
        assert "error" in r.json()
        assert "remote" in err["message"].lower() or "base64" in err["message"].lower()

    def test_malformed_body(self, client):
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {VALID_KEY}", "Content-Type": "application/json"},
            content=b"not json at all",
        )
        assert r.status_code in (400, 422)

    def test_stream_true_returns_error(self, client):
        body = _image_request(BUS_DATA_URL, stream=True)
        r = client.post("/v1/chat/completions", headers=HEADERS, json=body)
        assert r.status_code == 400
        err = r.json()["error"]
        assert "streaming" in err["message"].lower()

    def test_extra_fields_ignored(self, client):
        """temperature, top_p, max_tokens etc. must be accepted and ignored."""
        # Without a model we can only check it's not a 400 validation error.
        # Model-gated behaviour is tested in TestHappyPath.
        body = {
            "model": "yolo11n",
            "messages": [{"role": "user", "content": "no image"}],
            "temperature": 0.9,
            "top_p": 1.0,
            "max_tokens": 512,
            "n": 1,
        }
        r = client.post("/v1/chat/completions", headers=HEADERS, json=body)
        # Must fail with 400 (no image), not 422 (unknown field)
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Happy path: requires exported model
# ---------------------------------------------------------------------------
class TestHappyPath:
    @MODEL_REQUIRES
    def test_valid_request_returns_200(self, client):
        r = client.post(
            "/v1/chat/completions",
            headers=HEADERS,
            json=_image_request(BUS_DATA_URL),
        )
        assert r.status_code == 200

    @MODEL_REQUIRES
    def test_response_is_chat_completion_envelope(self, client):
        r = client.post(
            "/v1/chat/completions",
            headers=HEADERS,
            json=_image_request(BUS_DATA_URL),
        )
        body = r.json()
        assert body["object"] == "chat.completion"
        assert body["model"] == "yolo11n"
        assert len(body["choices"]) == 1
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["choices"][0]["message"]["role"] == "assistant"

    @MODEL_REQUIRES
    def test_usage_is_zero(self, client):
        r = client.post(
            "/v1/chat/completions",
            headers=HEADERS,
            json=_image_request(BUS_DATA_URL),
        )
        usage = r.json()["usage"]
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == 0

    @MODEL_REQUIRES
    def test_content_parses_as_detection_schema(self, client):
        r = client.post(
            "/v1/chat/completions",
            headers=HEADERS,
            json=_image_request(BUS_DATA_URL),
        )
        content_str = r.json()["choices"][0]["message"]["content"]
        payload = json.loads(content_str)

        assert "detections" in payload
        assert "image" in payload
        assert "model" in payload
        assert "count" in payload
        assert payload["model"] == "yolo11n"
        assert payload["count"] == len(payload["detections"])
        assert payload["image"]["width"] == 810
        assert payload["image"]["height"] == 1080

    @MODEL_REQUIRES
    def test_bus_detected_in_content(self, client):
        r = client.post(
            "/v1/chat/completions",
            headers=HEADERS,
            json=_image_request(BUS_DATA_URL),
        )
        payload = json.loads(r.json()["choices"][0]["message"]["content"])
        labels = [d["label"] for d in payload["detections"]]
        assert "bus" in labels, f"Expected 'bus' in detections, got: {labels}"

    @MODEL_REQUIRES
    def test_detection_box_fields_present(self, client):
        r = client.post(
            "/v1/chat/completions",
            headers=HEADERS,
            json=_image_request(BUS_DATA_URL),
        )
        payload = json.loads(r.json()["choices"][0]["message"]["content"])
        for d in payload["detections"]:
            assert {"label", "class_id", "confidence", "box"} <= d.keys()
            assert {"x1", "y1", "x2", "y2"} <= d["box"].keys()

    @MODEL_REQUIRES
    def test_text_content_block_accepted_ignored(self, client):
        """Text blocks alongside the image must not cause an error."""
        body = {
            "model": "yolo11n",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {"type": "image_url", "image_url": {"url": BUS_DATA_URL}},
                    ],
                }
            ],
        }
        r = client.post("/v1/chat/completions", headers=HEADERS, json=body)
        assert r.status_code == 200
