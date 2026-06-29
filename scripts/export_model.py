"""
Offline export: downloads a YOLO model and exports it to ONNX.
Run this once per machine before starting the service.

Usage:
    python scripts/export_model.py [--model yolo11n] [--out models/yolo11n.onnx]
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO weights to ONNX.")
    parser.add_argument(
        "--model", default="yolo11n", help="Ultralytics model name (default: yolo11n)"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path for the .onnx file (default: models/<model>.onnx)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    out_path = Path(args.out) if args.out else repo_root / "models" / f"{args.model}.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO  # dev/export only

    model = YOLO(f"{args.model}.pt")
    exported = model.export(format="onnx", dynamic=False, simplify=True)

    # ultralytics writes the file next to the .pt; move it to the target path.
    exported_path = Path(exported)
    if exported_path.resolve() != out_path.resolve():
        out_path.write_bytes(exported_path.read_bytes())
        exported_path.unlink(missing_ok=True)

    # Print the model input size so the caller can confirm the expected shape.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    print(f"Exported: {out_path.resolve()}")
    print(f"Input name : {inp.name}")
    print(f"Input shape: {inp.shape}")
    print(f"Input dtype: {inp.type}")
    out0 = sess.get_outputs()[0]
    print(f"Output[0] name : {out0.name}")
    print(f"Output[0] shape: {out0.shape}")
    print(f"Output[0] dtype: {out0.type}")


if __name__ == "__main__":
    main()
