# Architecture

This is the durable architecture note for the service. It is repository truth;
keep it current when behavior changes.

## One process, no services

The entire system is a single FastAPI/Uvicorn process. There is no database, no
cache, no message broker, no worker, no scheduler. This is deliberate: the
product is stateless per-image detection, so persistence and background work
would be accidental complexity and are forbidden (see `AGENTS.md` §5).

```
client (OpenAI SDK)
      │  POST /v1/chat/completions  (Bearer fixed key, 1 base64 image)
      ▼
┌─────────────────────────────────────────────┐
│ FastAPI process                              │
│                                              │
│  auth (constant-time)  ─ fail closed         │
│        │                                     │
│  request validation (Pydantic)               │
│        │                                     │
│  extract exactly one base64 image            │
│        │                                     │
│  decode (Pillow/NumPy)                       │
│        │                                     │
│  ┌─ off-event-loop executor, bounded ─────┐  │
│  │  letterbox → normalize → NCHW          │  │
│  │  ONNX Runtime (CPU EP) inference        │  │
│  │  confidence threshold → NMS → rescale   │  │
│  └────────────────────────────────────────┘  │
│        │                                     │
│  format detection JSON                       │
│        │                                     │
│  wrap in chat.completion envelope            │
└──────────────────────────────────────────────┘
      │
      ▼
  response (OpenAI-shaped; JSON detections in message content)
```

## Why ONNX Runtime on CPU

Exporting YOLO weights to ONNX and serving them with ONNX Runtime's CPU
execution provider gives a substantial speedup over raw PyTorch on CPU and runs
natively on Apple Silicon (ARM64). Critically, it lets the runtime avoid pulling
in `torch`/`ultralytics`, keeping the served process small. `ultralytics` is
used **only offline** to produce the `.onnx` file (see `scripts/`).

## Concurrency model

CPU inference is blocking and CPU-bound. Two rules follow:

1. **Off the event loop.** Inference runs in a threadpool/executor so a single
   request cannot stall the server.
2. **Bounded concurrency.** A semaphore sized to the available core count caps
   simultaneous inferences so a laptop does not thrash. Excess requests queue or
   are rejected with a clear status rather than degrading everything.

The model is loaded **once** at startup and shared (ONNX Runtime sessions are
usable across threads). No per-request model construction.

## Pre/post-processing we own

Because v1 uses raw ONNX Runtime (not the `ultralytics` runtime), this project
owns the steps `ultralytics` would otherwise hide:

- **Preprocess:** letterbox-resize the decoded image to the model input size
  (e.g. 640×640) preserving aspect ratio, normalize to `[0,1]`, convert HWC→CHW,
  add batch dim.
- **Postprocess:** read the model output, apply a confidence threshold, run
  **non-maximum suppression**, and **rescale** surviving boxes from letterboxed
  model space back to original image pixel coordinates.

The exact output tensor shape depends on the exported model and must be verified
against the real `.onnx` file, not assumed. The real-detection test is the guard
that this pipeline is correct end-to-end.

> Note: YOLO11n/YOLOv8n exports require this external NMS step. YOLO26n's
> NMS-free head would remove it — recorded as a future option in `AGENTS.md`
> §11, not implemented now.

## Trust boundary

- The only secret is `API_KEY`, read from the environment.
- No outbound network at request time (base64 only, no URL fetch) → no SSRF
  surface.
- No request data persisted → minimal data-handling risk.
