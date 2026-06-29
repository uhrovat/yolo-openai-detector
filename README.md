# cpu-yolo-openai-detect

A **CPU-only object-detection service** that speaks the **OpenAI API**. Point the
ordinary OpenAI SDK at it, authenticate with one fixed key, send **one image**,
and get back **structured JSON detections**. No GPU required. Designed to run on
a GPU-less laptop (developed for Apple Silicon).

> **Status: governed scaffold.** This repository currently contains the project
> constitution, design docs, and structure. The service itself is implemented
> incrementally by AI coding agents following the work orders in
> [`docs/work-orders/`](docs/work-orders/). See [`AGENTS.md`](AGENTS.md) before
> contributing or running an agent here.

---

## What it is (and is not)

**It is:** stateless, single-image object **detection** wearing an OpenAI vision
model's interface.

**It is not:** a tracker, a segmenter, a multi-image or video API, a text model,
or a per-user metered gateway. See [`docs/non-goals.md`](docs/non-goals.md).

---

## How clients use it

Once the service is running and you have the fixed key, existing OpenAI client
code works unchanged except for `base_url` and the key:

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="YOUR_FIXED_KEY")

with open("street.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

resp = client.chat.completions.create(
    model="yolo11n",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "detect"},  # optional, ignored
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ],
    }],
)

# The detections are a JSON string in the assistant message content:
print(resp.choices[0].message.content)
```

Example response content (see [`docs/response-schema.md`](docs/response-schema.md)):

```json
{
  "detections": [
    {"label": "person", "class_id": 0, "confidence": 0.92,
     "box": {"x1": 34.0, "y1": 50.2, "x2": 220.1, "y2": 410.7}}
  ],
  "image": {"width": 640, "height": 480},
  "model": "yolo11n",
  "count": 1
}
```

---

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/chat/completions` | Detection (the workhorse) |
| GET | `/v1/models` | Lists the advertised model |
| GET | `/health` | Non-OpenAI liveness probe |

---

## Deliberate deviations from OpenAI

These are intentional and enforced (full list in
[`docs/openai-compatibility.md`](docs/openai-compatibility.md)):

- **Base64 images only.** Remote `http(s)` image URLs are rejected.
- **Exactly one image** per request. Zero or multiple are rejected.
- The assistant message content is **detection JSON**, not prose.
- `usage` token counts are always **zero** (no LLM tokens exist).

---

## Configuration

| Variable | Meaning |
|----------|---------|
| `API_KEY` | The single fixed bearer token clients must present. **Required.** |

Copy `.env.example` to `.env` and set a strong value. The key is never logged.

---

## Running (after implementation lands)

Implementation is delivered via the work orders. Once `app/` exists:

```bash
# install runtime deps (exact versions come from the committed lockfile)
pip install -r requirements.txt

# export weights to ONNX (offline, dev tooling)
pip install -r requirements-dev.txt
python scripts/export_model.py --model yolo11n

# run
API_KEY=changeme uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## For contributors and AI agents

Read [`AGENTS.md`](AGENTS.md) (canonical) — it defines the mission, invariants,
forbidden actions, workflow, testing, and reporting rules. `CLAUDE.md` mirrors
it for Claude Code.

---

## Licensing

Ultralytics YOLO and the standard pretrained COCO weights are **AGPL-3.0**
(commercial license available separately). Serving these weights over a network
carries obligations. Choose this project's license and confirm posture before
public deployment. See [`NOTICE.md`](NOTICE.md).
