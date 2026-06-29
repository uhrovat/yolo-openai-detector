# NOTICE — Licensing and Attribution

This project depends on third-party components with license obligations the
human lead must review **before any public deployment**.

## Ultralytics YOLO and pretrained weights

- The `ultralytics` package and the standard pretrained YOLO COCO weights
  (e.g. `yolo11n`) are distributed under **AGPL-3.0**, with a separate
  commercial/enterprise license available from Ultralytics.
- **Implication:** AGPL-3.0 has network-use ("copyleft over a network")
  obligations. Exporting weights with `ultralytics` and **serving those weights
  over an API** can trigger source-availability obligations for AGPL-covered
  components.
- This affects (a) the offline export tooling and (b) the served model weights —
  even though `ultralytics`/`torch` are not in the runtime dependency set, the
  **weights themselves** carry the license.

## What the human lead must decide

1. This project's **own license** (not yet chosen — see `README.md`).
2. Whether to use AGPL pretrained weights, custom-trained weights with a
   different license, or obtain an Ultralytics commercial license.
3. The licensing posture for any public deployment.

These are **human decisions**, not agent decisions. AI agents must not assume,
assert, or change the license.

## Other runtime dependencies

ONNX Runtime, FastAPI, Uvicorn, Pydantic, Pillow, and NumPy are under permissive
licenses (MIT / Apache-2.0 / BSD / HPND-style). Confirm exact terms from the
committed lockfile before release.
