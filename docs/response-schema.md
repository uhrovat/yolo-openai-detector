# Detection Response Schema

The assistant message content in every successful `/v1/chat/completions`
response is a **JSON string** with the structure below. This is the durable
contract for clients parsing detections.

## Schema

```json
{
  "detections": [
    {
      "label": "string",       // COCO class name, e.g. "person"
      "class_id": 0,            // integer COCO class index
      "confidence": 0.0,        // float in [0,1]
      "box": {
        "x1": 0.0,              // left   (pixels, original image coords)
        "y1": 0.0,              // top
        "x2": 0.0,              // right
        "y2": 0.0               // bottom
      }
    }
  ],
  "image": {"width": 0, "height": 0},  // original image dimensions, pixels
  "model": "yolo11n",
  "count": 0                            // == len(detections)
}
```

## Rules

- **Box format:** `xyxy`, **absolute pixels**, floats, origin at top-left, in the
  **original** image's coordinate space (already rescaled from letterboxed model
  space). This is decision **D3** in `AGENTS.md`.
- **Order:** detections sorted by `confidence` descending.
- **Empty result:** if nothing is detected, `detections` is `[]` and `count` is
  `0`. This is a valid, successful response — not an error.
- **Thresholds:** detections below the configured confidence threshold are
  omitted; overlapping boxes are reduced by NMS before serialization.
- **`model`:** the advertised name echoed from the request (`yolo11n`).
- **No extra fields** in v1. A normalized `[0,1]` box variant is a documented
  future option (`AGENTS.md` §11), not part of v1.

## Example

```json
{
  "detections": [
    {"label": "dog", "class_id": 16, "confidence": 0.94,
     "box": {"x1": 120.0, "y1": 200.5, "x2": 410.2, "y2": 540.9}},
    {"label": "bicycle", "class_id": 1, "confidence": 0.81,
     "box": {"x1": 12.0, "y1": 300.0, "x2": 260.0, "y2": 600.0}}
  ],
  "image": {"width": 800, "height": 600},
  "model": "yolo11n",
  "count": 2
}
```
