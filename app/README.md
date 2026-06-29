# app/

The FastAPI service. **Implemented by the execution agent via work orders** —
do not hand-write the full app outside the OAP workflow.

Intended modules (created across WO-01..WO-03):
- `detection.py` — pure, HTTP-free inference (image bytes -> detections). (WO-01)
- `main.py` — FastAPI app factory and routes. (WO-02)
- `schemas.py` — Pydantic models for the OpenAI request/response shapes. (WO-02)
- `auth.py` — constant-time fixed-key bearer check. (WO-02)
- `errors.py` — OpenAI-style error envelope helpers. (WO-02/03)

Rules: inference off the event loop, bounded concurrency, model loaded once,
runtime deps exclude torch/ultralytics. See `../AGENTS.md`.
