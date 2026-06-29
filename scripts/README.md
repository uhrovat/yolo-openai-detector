# scripts/

Offline tooling. Runs on a developer machine, not part of the served runtime.

- `export_model.py` — exports YOLO11n weights to `../models/yolo11n.onnx` using
  `ultralytics` (a dev/export dependency only). Created in WO-01.

`ultralytics` and `torch` are allowed here because this is offline; they must
never enter runtime dependencies. See `../AGENTS.md` §5.
