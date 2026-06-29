# models/

Location for exported ONNX weights (e.g. `yolo11n.onnx`).

Weights are **build artifacts, not source**: they are produced by
`../scripts/export_model.py` and are git-ignored (see `../.gitignore`). Do not
commit `.onnx`/`.pt` files. Re-export them on each machine.

Licensing: the pretrained COCO weights are AGPL-3.0. See `../NOTICE.md`.
