# Requirements

Durable record of what the service must do and the qualities it must have.
Acceptance criteria are phrased so success can be judged before code is written.

## Functional requirements

- **FR1** Accept `POST /v1/chat/completions` in the OpenAI Chat Completions
  request shape.
- **FR2** Authenticate every request with a single fixed bearer key, compared
  constant-time; reject all others.
- **FR3** Accept exactly one image per request, delivered as a base64 data URL
  inside the user message content.
- **FR4** Run YOLO object detection on the decoded image on CPU.
- **FR5** Return detections as the JSON in `response-schema.md`, inside a valid
  OpenAI `chat.completion` envelope.
- **FR6** Serve `GET /v1/models` listing the advertised model in OpenAI shape.
- **FR7** Serve `GET /health` for liveness/readiness.
- **FR8** Support `stream: true` via emulated single-chunk SSE (decision D2).
- **FR9** Return OpenAI-style error objects with correct HTTP codes for every
  rejection path.

## Non-functional requirements

- **NFR1 Compatibility.** The unmodified `openai` Python client works against the
  service with only `base_url` and `api_key` changed. Proven by test.
- **NFR2 Statelessness.** No request data persisted anywhere.
- **NFR3 Responsiveness under load.** Inference runs off the event loop with
  bounded concurrency; one slow request does not stall others.
- **NFR4 CPU-only.** Runs on a GPU-less Apple Silicon laptop. No CUDA, no
  discrete-GPU dependency.
- **NFR5 Lean runtime.** Runtime dependencies exclude `torch` and `ultralytics`.
- **NFR6 Security.** Key never logged/committed; no outbound calls at request
  time; no SSRF surface; fail closed.
- **NFR7 Reproducibility.** Pinned dependencies with a committed lockfile;
  documented export step; deterministic given the same model and thresholds.
- **NFR8 Honesty.** Empty detection is a valid success; `usage` is zeroed, never
  fabricated; release language matches evidence.

## Acceptance criteria (v1 "done")

- A real `openai` client call with one base64 image returns parseable detection
  JSON in the documented schema.
- A fixture image with a known object yields that object with a plausible box and
  confidence (real-detection test passes).
- All negative paths (no/again-multiple image, remote URL, bad key, malformed
  body) return the documented OpenAI-style errors.
- The event loop is never blocked by inference (verified by design/review).
- No `torch`/`ultralytics` in runtime deps; lockfile committed.
- `README`, `openai-compatibility.md`, and `response-schema.md` match actual
  behavior.
