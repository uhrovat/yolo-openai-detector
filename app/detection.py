"""
Pure inference module: image bytes -> DetectionResult.
No FastAPI, no network I/O. This is the runtime inference path (onnxruntime only;
torch/ultralytics are never imported here).

Output tensor shape for yolo11n (non-end2end export): [1, 84, 8400]
  dim 1: [cx, cy, w, h, score_cls0 … score_cls79]
  dim 2: 8400 anchor proposals
External NMS is required.
"""

import io
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np
import onnxruntime as ort
from PIL import Image

# ---------------------------------------------------------------------------
# COCO-80 class names (index == class_id)
# ---------------------------------------------------------------------------
COCO_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

MODEL_NAME = "yolo11n"
_INPUT_SIZE = 640  # model was exported at 640×640

# ---------------------------------------------------------------------------
# Singleton ONNX session (loaded once at module import, reused across requests)
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "yolo11n.onnx"


def _load_session(model_path: Path | None = None) -> ort.InferenceSession:
    path = model_path or _DEFAULT_MODEL_PATH
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


_session: ort.InferenceSession | None = None


def get_session() -> ort.InferenceSession:
    global _session
    if _session is None:
        _session = _load_session()
    return _session


# ---------------------------------------------------------------------------
# Shared threadpool for off-event-loop execution (used by the HTTP layer)
# ---------------------------------------------------------------------------
_executor = ThreadPoolExecutor(max_workers=max(1, os.cpu_count() or 1))


def get_executor() -> ThreadPoolExecutor:
    return _executor


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
class Box(NamedTuple):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(NamedTuple):
    label: str
    class_id: int
    confidence: float
    box: Box


class DetectionResult(NamedTuple):
    detections: list[Detection]
    image_width: int
    image_height: int
    model: str
    count: int


# ---------------------------------------------------------------------------
# Preprocessing: letterbox resize
# ---------------------------------------------------------------------------
def _letterbox(
    img: Image.Image, target: int = _INPUT_SIZE
) -> tuple[np.ndarray, float, int, int]:
    """
    Resize img to target×target with grey padding, preserving aspect ratio.
    Returns (array_NCHW_float32, scale, pad_left, pad_top).
    """
    orig_w, orig_h = img.size
    scale = min(target / orig_w, target / orig_h)
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)

    resized = img.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (target, target), (114, 114, 114))
    pad_left = (target - new_w) // 2
    pad_top = (target - new_h) // 2
    canvas.paste(resized, (pad_left, pad_top))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0  # HWC, [0,1]
    arr = arr.transpose(2, 0, 1)[np.newaxis]  # NCHW
    return arr, scale, pad_left, pad_top


# ---------------------------------------------------------------------------
# Postprocessing: NMS and rescale
# ---------------------------------------------------------------------------
def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one box against an array of boxes (all xyxy)."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return np.where(union > 0, inter / union, 0.0)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy NMS. boxes is (N, 4) xyxy; scores is (N,). Returns kept indices."""
    order = scores.argsort()[::-1]
    kept: list[int] = []
    while order.size > 0:
        i = order[0]
        kept.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        ious = _iou(boxes[i], boxes[rest])
        order = rest[ious <= iou_threshold]
    return kept


def _postprocess(
    raw: np.ndarray,
    scale: float,
    pad_left: int,
    pad_top: int,
    orig_w: int,
    orig_h: int,
    conf_threshold: float,
    iou_threshold: float,
) -> list[Detection]:
    """
    raw shape: [1, 84, 8400].
    Returns detections sorted by confidence descending.
    """
    # Transpose to [8400, 84]: each row is [cx, cy, w, h, cls0..cls79]
    preds = raw[0].T  # (8400, 84)

    boxes_cxywh = preds[:, :4]
    class_scores = preds[:, 4:]  # (8400, 80)

    confidences = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    mask = confidences >= conf_threshold
    if not mask.any():
        return []

    boxes_cxywh = boxes_cxywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    # Convert cx,cy,w,h (letterboxed model space) -> xyxy (letterboxed model space)
    x1 = boxes_cxywh[:, 0] - boxes_cxywh[:, 2] / 2
    y1 = boxes_cxywh[:, 1] - boxes_cxywh[:, 3] / 2
    x2 = boxes_cxywh[:, 0] + boxes_cxywh[:, 2] / 2
    y2 = boxes_cxywh[:, 1] + boxes_cxywh[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    kept_indices = _nms(boxes_xyxy, confidences, iou_threshold)

    detections: list[Detection] = []
    for idx in kept_indices:
        bx1, by1, bx2, by2 = boxes_xyxy[idx]

        # Rescale from letterboxed model space to original image pixels
        rx1 = float(np.clip((bx1 - pad_left) / scale, 0, orig_w))
        ry1 = float(np.clip((by1 - pad_top) / scale, 0, orig_h))
        rx2 = float(np.clip((bx2 - pad_left) / scale, 0, orig_w))
        ry2 = float(np.clip((by2 - pad_top) / scale, 0, orig_h))

        cid = int(class_ids[idx])
        label = COCO_CLASSES[cid] if 0 <= cid < len(COCO_CLASSES) else str(cid)
        detections.append(
            Detection(
                label=label,
                class_id=cid,
                confidence=float(confidences[idx]),
                box=Box(x1=rx1, y1=ry1, x2=rx2, y2=ry2),
            )
        )

    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def detect(
    image_bytes: bytes,
    conf: float = 0.25,
    iou: float = 0.45,
    session: ort.InferenceSession | None = None,
) -> DetectionResult:
    """
    Decode image_bytes, run YOLO11n inference, return DetectionResult.
    Deterministic for the same inputs. Never raises on empty detections.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = img.size

    sess = session or get_session()
    input_arr, scale, pad_left, pad_top = _letterbox(img)

    input_name = sess.get_inputs()[0].name
    raw: np.ndarray = sess.run(None, {input_name: input_arr})[0]

    detections = _postprocess(raw, scale, pad_left, pad_top, orig_w, orig_h, conf, iou)

    return DetectionResult(
        detections=detections,
        image_width=orig_w,
        image_height=orig_h,
        model=MODEL_NAME,
        count=len(detections),
    )
