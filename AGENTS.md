# AGENTS.md — Project Constitution

> This file is the operational law of this repository. Any AI coding agent
> (Codex CLI, Claude Code, or other) **must read this file before doing any
> work** and must obey it. `CLAUDE.md` is a faithful mirror of this file for
> Claude Code; if you change one, change the other. **AGENTS.md is canonical.**

---

## 0. How to use this document

- Read this whole file first.
- Then read the work order you were given in `docs/work-orders/`.
- If the live repository state differs from your work order, **stop and report
  the difference** rather than building on stale assumptions.
- If a requested action would violate a rule here, **refuse and report**, do not
  "helpfully" work around it.

---

## 1. Discovery Summary

**Domain problem.** Provide object detection from images on a machine with no
GPU, exposed through an interface that existing OpenAI client code can talk to
unchanged, gated by a single shared API key.

**Chosen product shape.** A *stateless, single-image object-detection service
that presents itself as an OpenAI vision model.* A client points the ordinary
OpenAI SDK at this service's `base_url`, authenticates with one fixed API key,
sends exactly one base64-attached image per `chat/completions` request, and
receives structured JSON detections in the assistant message. Nothing persists
between requests.

**Architecture & stack rationale (summary).** A single FastAPI process. YOLO
weights are exported offline to ONNX; at runtime the service uses ONNX Runtime
(CPU execution provider, native on Apple Silicon) for inference, run off the
event loop. No database, no queue, no background jobs, no external services.
ONNX gives a substantial CPU speedup over raw PyTorch and keeps the runtime
lean.

**Important alternatives rejected.**
- *Tracking / streaming sessions* — rejected: out of scope; the product is
  per-image detection only.
- *Whole-clip video tracking* — rejected: out of scope.
- *Per-user keys, quota, accounting* (the SLAIF-gateway pattern) — rejected:
  this service uses one fixed key and does no accounting.
- *OpenVINO* — rejected: Intel-only, irrelevant on Apple Silicon.
- *Heavy runtime via the `ultralytics`/`torch` stack for serving* — rejected for
  v1 runtime to keep the service lean; `ultralytics` is used **offline only**
  for export. (See §11 for the documented future option to reconsider.)

**First release scope.** Detection over one image per request, JSON output,
fixed-key auth, OpenAI-compatible `/v1/chat/completions` and `/v1/models`, plus
a non-OpenAI `/health` probe.

---

## 2. Mission

- **What this repository is for:** serving CPU object detection through an
  OpenAI-compatible HTTP API.
- **The user promise that must not be broken:** *unchanged OpenAI client code,
  pointed at this service with the fixed key, can send one base64 image and get
  back honest, structured detections.* Compatibility and honesty of results are
  the product. Do not break the OpenAI request/response envelope, and never
  return a fabricated or partial detection result.

---

## 3. Architecture

**Components (single process):**
- HTTP / async layer: **FastAPI + Uvicorn**.
- Validation: **Pydantic v2** request/response models (validation *is* the
  compatibility contract).
- Inference: **ONNX Runtime**, CPU execution provider. Model loaded **once** at
  startup and reused.
- Image decode: **Pillow + NumPy**.
- Pre/post-processing owned by this project: letterbox resize to model input,
  normalization, NCHW layout, then confidence thresholding, **non-maximum
  suppression (NMS)**, and rescaling boxes back to original image coordinates.

**Request flow (must be preserved):**
1. Receive `POST /v1/chat/completions`.
2. Authenticate: constant-time comparison of the bearer token against the fixed
   key. Fail closed on any mismatch.
3. Validate the body against the OpenAI-shaped schema.
4. Extract **exactly one** base64 image from the user message.
5. Decode the image.
6. Run inference **off the event loop** (threadpool/executor), under a bounded
   concurrency limit.
7. Format detections as the JSON schema in `docs/response-schema.md`.
8. Wrap in an OpenAI `chat.completion` envelope and return.

**Stack and versions.** Pinned in `pyproject.toml` / `requirements*.txt`. Exact
versions must be locked and committed as a lockfile in the first implementation
PR (see §8). Do not add dependencies not listed there without a work order that
authorizes it.

**Ownership boundaries.**
- `app/` — the service (routes, schemas, auth, inference wrapper). Built by the
  execution agent under work orders.
- `scripts/` — offline tooling (the weight→ONNX export script).
- `models/` — exported `.onnx` weights (artifact location).
- `tests/` — evidence.
- `docs/` — durable project truth (architecture, compatibility, schema,
  requirements, non-goals, work orders).

**Data flow assumption.** Stateless. No request data is stored. The only
long-lived in-memory object is the loaded model.

---

## 4. Non-negotiable invariants

- **Stateless.** No tracking, no sessions, no cross-request memory beyond the
  loaded model.
- **One image per request.** Exactly one. Zero or more than one → error.
- **Inference runs off the event loop.** Blocking the event loop is a defect.
- **Bounded concurrency.** A semaphore sized to available cores prevents CPU
  thrash on a laptop.
- **Fail closed.** Any auth failure, malformed body, wrong image count, or
  decode failure returns a clean OpenAI-style error — never a partial,
  defaulted, or guessed detection result.
- **Constant-time key comparison.** The fixed key is read from the environment
  (`API_KEY`), never hard-coded, never logged, never committed.
- **No image persistence.** Image bytes and decoded pixels are never written to
  disk, logs, or telemetry.
- **Model loaded once.** No per-request model construction.
- **Honest envelope.** The response is a valid OpenAI `chat.completion` object.
  `usage` token counts are reported as zero because no LLM tokens exist; they are
  never fabricated.

---

## 5. Forbidden actions

- Do **not** implement tracking, persistent IDs, or any temporal/cross-frame
  logic.
- Do **not** implement segmentation, masks, pose, classification, or OBB.
- Do **not** accept more than one image, video, or frame sequences.
- Do **not** fetch remote `http(s)` image URLs. **Base64 data URLs only.**
  (This closes SSRF and removes outbound network requirements.)
- Do **not** add a database, ORM, migration tool, Redis, Celery, or any
  background job or scheduler.
- Do **not** add per-user keys, quota, rate-accounting, or billing.
- Do **not** implement real text generation, embeddings, audio, image
  generation, or any non-detection OpenAI endpoint.
- Do **not** log, store, or transmit image content or the API key.
- Do **not** add `torch` or `ultralytics` to the **runtime** dependencies.
  (`ultralytics` is permitted in the **dev/export** group only.)
- Do **not** make outbound network calls at request time.
- Do **not** commit weights, secrets, `.env`, or large binaries that are not
  intended artifacts.
- Do **not** merge your own pull request.

---

## 6. Decisions adopted (v1)

These were the strategic recommendations adopted for v1. They are changeable by
the human lead, but until changed here, they are law.

| ID | Decision | Adopted value |
|----|----------|---------------|
| D1 | Shipped weights | **YOLO11n** (COCO-80). YOLOv8n is a documented drop-in. YOLO26n recorded for future evaluation (NMS-free → simpler/faster CPU). |
| D2 | `stream: true` handling | **Emulate**: emit the full JSON as a single SSE `chat.completion.chunk`, then `data: [DONE]`. Never error on `stream`. |
| D3 | Bounding-box format | **`xyxy`, absolute pixels**, floats, origin top-left, in original image coordinates. |
| D4 | Advertised model name | **`yolo11n`** — the real weight name, echoed in responses and listed by `/v1/models`. |

---

## 7. Workflow

- All implementation work happens on a **feature branch**, never directly on
  `main`. (Exception: the initial governance commit, performed by the human.)
- One work order → one **PR-sized** change. Keep diffs reviewable.
- Commit **only files related to the work order**.
- Push the branch and open a **pull request**. **Do not merge it.**
- If you installed local tools or dependencies inside the execution VM,
  document the exact commands in the PR. Do not ask the human to perform routine
  dependency setup unless a real safety boundary blocks you.
- If you discover the work order is wrong or impossible, stop and report rather
  than improvising scope.

---

## 8. Testing

- **Schema tests:** a valid request returns a valid `chat.completion` envelope,
  and the message content parses as the schema in `docs/response-schema.md`.
- **Real-detection test:** a committed fixture image containing a known object
  is detected with a plausible label, confidence, and box. This test is what
  proves letterbox/NMS/rescale correctness — it must be meaningful, not a
  tautology.
- **Negative tests:** zero images, two images, a remote URL, a missing/invalid
  key, and a malformed body each return the correct OpenAI-style error and HTTP
  status.
- **Compatibility test:** drive the running service with the genuine `openai`
  Python client to prove unchanged client code works.
- **Dependency pinning:** the first implementation PR must produce and commit a
  lockfile (e.g. `uv.lock` or a `pip-compile` output) with exact versions, and
  must verify every import path actually resolves (guard against hallucinated
  APIs).
- **Reporting test status:** use the vocabulary in §9. Never say "all tests
  passed" unless the full relevant suite literally passed. "Skipped" and "not
  run" are not "passed".

---

## 9. Documentation & reporting

**Docs that must change when behavior changes:**
- `docs/openai-compatibility.md` — every accepted/rejected/ignored field.
- `docs/response-schema.md` — the detection JSON schema.
- `README.md` — anything user-facing (how to call it, deviations, limits).

**Required final report format for any work order:**
- Branch name.
- Commit hash(es).
- PR URL.
- Summary of what changed.
- Tests run and results, using the status vocabulary:
  `passed` / `failed` / `skipped` / `not run` / `blocked` / `out of scope`.
- Local tools or dependencies installed (with commands).
- Docs changed.
- Risks, known gaps, and anything you are least confident about.

State limitations honestly. "Implemented" is not "production-ready". Release
language must match evidence.

---

## 10. Licensing note (read before shipping)

Ultralytics YOLO code and the standard pretrained COCO weights are distributed
under **AGPL-3.0** (with a separate commercial/enterprise license available).
Using the `ultralytics` package to export weights, and serving those pretrained
weights over a network, carries AGPL-3.0 obligations. The human lead must choose
this project's own license and confirm the licensing posture before any public
deployment. See `NOTICE.md`. This is a human decision, not an agent decision.

---

## 11. Documented future options (NOT in scope now)

These are recorded so they are not silently implemented and not forgotten:
- Evaluate **YOLO26n** (NMS-free, end-to-end head) to remove our own NMS code
  and gain CPU speed.
- Optional **CoreML export** path to use the Apple Neural Engine (still not a
  discrete GPU).
- Optional **normalized `[0,1]`** box output variant alongside absolute pixels.

Implementing any of these requires a new work order and a constitution update.
