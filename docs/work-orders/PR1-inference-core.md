# Work Order WO-01 — Inference Core

You are working in repository: `cpu-yolo-openai-detect`.

## Governing instructions
- Read `AGENTS.md` first and obey it in full.
- Follow the repository workflow (feature branch, PR, do not merge).
- If live repository state differs from this work order, **report the
  difference** before proceeding.

## Current verified state
- The repository contains governance and docs only (`AGENTS.md`, `CLAUDE.md`,
  `README.md`, `docs/`, config files). `app/`, `scripts/`, `models/`, `tests/`
  exist but contain no implementation.
- No model has been exported yet.

## Goal
Produce (a) an offline script that exports YOLO11n weights to ONNX, and (b) a
**pure, HTTP-free detection function** that takes raw image bytes and returns a
list of detections matching `docs/response-schema.md`. Prove correctness with a
real-detection test.

## Scope
- `scripts/export_model.py`: uses `ultralytics` (dev/export dependency only) to
  download `yolo11n` and export it to `models/yolo11n.onnx`. Accept a
  `--model` argument defaulting to `yolo11n`. Print the resolved output path and
  the model input size.
- `app/detection.py` (or similar): a function such as
  `detect(image_bytes: bytes, conf: float = ..., iou: float = ...) -> DetectionResult`
  that:
  - decodes the image with Pillow,
  - letterbox-resizes to the model input size, normalizes to `[0,1]`, HWC→CHW,
    adds batch dim,
  - runs inference with **ONNX Runtime (CPU execution provider)** using a
    session created **once** and reused,
  - applies confidence thresholding and **non-maximum suppression**,
  - rescales boxes to original image pixel coordinates,
  - returns detections sorted by confidence descending, plus original image
    width/height, in the exact shape of `docs/response-schema.md`.
- COCO class names mapping for `label` / `class_id`.

## Non-goals (do not do)
- Do **not** add any FastAPI/HTTP code. This work order is inference only.
- Do **not** add `torch` or `ultralytics` to **runtime** dependencies; they
  belong in the dev/export group. The runtime path uses `onnxruntime` only.
- Do **not** fetch remote URLs, persist images, or add a database/queue.
- Do **not** implement segmentation, tracking, or multi-image handling.

## Files to inspect
- `AGENTS.md`, `docs/architecture.md`, `docs/response-schema.md`,
  `pyproject.toml`, `requirements*.txt`.

## Required behavior
- Verify the **actual** ONNX output tensor shape from the exported model; do not
  assume it. Decode it correctly for YOLO11n (non-end2end export requires NMS).
- The detection function must be deterministic given the same image, model, and
  thresholds.

## Tests required
- **Real-detection test:** commit a small fixture image containing an obvious
  object (e.g. a person, dog, or car) under `tests/fixtures/`. Assert that the
  expected class is detected with confidence above a sensible threshold and a
  box within plausible bounds. This test must genuinely exercise
  letterbox+NMS+rescale.
- **Empty/edge test:** a blank image returns an empty detection list and a valid
  result object (not an error).
- Tests must run in CI without a GPU.

## Local setup allowed
- Install `ultralytics`, `onnxruntime`, `pillow`, `numpy`, `pytest` inside the
  execution VM as needed. Document exact commands in the PR.
- Produce and commit a dependency lockfile with exact versions (see `AGENTS.md`
  §8). Verify every import path resolves.

## Documentation required
- Update `models/README.md` and `scripts/README.md` if their intended contents
  change.
- If you discover the real output shape or any detail that contradicts
  `docs/architecture.md` or `docs/response-schema.md`, report it; the strategic
  layer will update those docs (do not silently diverge).

## Workflow
- Start from a fresh `main`.
- Create a feature branch (e.g. `wo-01-inference-core`).
- Commit only related files (do **not** commit the exported `.onnx` weight or
  any `.env`).
- Push the branch and open a pull request. **Do not merge.**

## Final report (required format)
- Branch, commit hash(es), PR URL.
- Summary of what changed.
- Tests run and results, using the vocabulary in `AGENTS.md` §9
  (`passed`/`failed`/`skipped`/`not run`/`blocked`/`out of scope`).
- The real ONNX output tensor shape you observed.
- Local tools/dependencies installed (with commands) and the lockfile produced.
- Docs changed.
- Risks, known gaps, and what you are least confident about.
