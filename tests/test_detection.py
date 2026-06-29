"""
WO-01 detection tests.

Real-detection test: bus.jpg (ultralytics sample image, AGPL-3.0 scope)
contains a bus and multiple people. We assert the bus is detected with
reasonable confidence and a plausible bounding box.

All tests run on CPU with no GPU, using the pre-exported models/yolo11n.onnx.
The .onnx file must be exported locally before running tests (scripts/export_model.py).
"""

import io
from pathlib import Path

import pytest
from PIL import Image

from app.detection import DetectionResult, detect

FIXTURES = Path(__file__).parent / "fixtures"
MODEL_PATH = Path(__file__).parent.parent / "models" / "yolo11n.onnx"


def _blank_image_bytes(width: int = 64, height: int = 64) -> bytes:
    img = Image.new("RGB", (width, height), (114, 114, 114))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Skip guard: tests need the exported model
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason=f"Model not exported — run: python scripts/export_model.py (expected at {MODEL_PATH})",
)


# ---------------------------------------------------------------------------
# Real-detection test
# ---------------------------------------------------------------------------
class TestRealDetection:
    """Exercises the full letterbox + inference + NMS + rescale pipeline."""

    def test_bus_detected_in_bus_jpg(self):
        image_bytes = (FIXTURES / "bus.jpg").read_bytes()
        result = detect(image_bytes)

        assert isinstance(result, DetectionResult)
        assert result.model == "yolo11n"
        assert result.image_width == 810
        assert result.image_height == 1080
        assert result.count == len(result.detections)
        assert result.count > 0, "Expected at least one detection in bus.jpg"

        labels = [d.label for d in result.detections]
        assert "bus" in labels, (
            f"Expected 'bus' among detections, got: {labels}"
        )

    def test_bus_confidence_above_threshold(self):
        image_bytes = (FIXTURES / "bus.jpg").read_bytes()
        result = detect(image_bytes)

        bus_detections = [d for d in result.detections if d.label == "bus"]
        assert bus_detections, "No bus detected"
        best = bus_detections[0]
        assert best.confidence >= 0.5, (
            f"Bus confidence {best.confidence:.3f} is below expected 0.5"
        )

    def test_bus_box_plausible(self):
        """Bus box should occupy a meaningful fraction of the 810×1080 image."""
        image_bytes = (FIXTURES / "bus.jpg").read_bytes()
        result = detect(image_bytes)

        bus_detections = [d for d in result.detections if d.label == "bus"]
        assert bus_detections
        box = bus_detections[0].box

        # Box coordinates must be within image bounds
        assert 0 <= box.x1 < box.x2 <= 810
        assert 0 <= box.y1 < box.y2 <= 1080

        # Bus should cover at least 5% of the image area
        box_area = (box.x2 - box.x1) * (box.y2 - box.y1)
        image_area = 810 * 1080
        assert box_area / image_area >= 0.05, (
            f"Bus box area {box_area:.0f}px² is implausibly small "
            f"({100 * box_area / image_area:.1f}% of image)"
        )

    def test_detections_sorted_by_confidence_descending(self):
        image_bytes = (FIXTURES / "bus.jpg").read_bytes()
        result = detect(image_bytes)
        confs = [d.confidence for d in result.detections]
        assert confs == sorted(confs, reverse=True), "Detections not sorted by confidence desc"

    def test_deterministic(self):
        """Same image must produce identical results on two calls."""
        image_bytes = (FIXTURES / "bus.jpg").read_bytes()
        r1 = detect(image_bytes)
        r2 = detect(image_bytes)
        assert r1 == r2, "detect() is not deterministic"

    def test_person_also_detected(self):
        """bus.jpg contains people; they should also be detected."""
        image_bytes = (FIXTURES / "bus.jpg").read_bytes()
        result = detect(image_bytes)
        labels = [d.label for d in result.detections]
        assert "person" in labels, f"Expected persons in bus.jpg, got: {labels}"


# ---------------------------------------------------------------------------
# Edge / negative tests
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_blank_image_returns_empty_detections(self):
        result = detect(_blank_image_bytes())
        assert isinstance(result, DetectionResult)
        assert result.detections == []
        assert result.count == 0
        assert result.model == "yolo11n"
        assert result.image_width == 64
        assert result.image_height == 64

    def test_blank_image_valid_result_not_error(self):
        """Empty detections must not raise; result must be well-formed."""
        result = detect(_blank_image_bytes())
        assert result.count == len(result.detections)

    def test_non_square_image_letterboxed_correctly(self):
        """Wide image (640×100) box coords must stay within image bounds."""
        img = Image.new("RGB", (640, 100), (255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        result = detect(buf.getvalue())
        for d in result.detections:
            assert 0 <= d.box.x1 < d.box.x2 <= 640
            assert 0 <= d.box.y1 < d.box.y2 <= 100

    def test_class_ids_in_valid_range(self):
        image_bytes = (FIXTURES / "bus.jpg").read_bytes()
        result = detect(image_bytes)
        for d in result.detections:
            assert 0 <= d.class_id <= 79, f"class_id {d.class_id} out of COCO-80 range"

    def test_confidences_in_unit_interval(self):
        image_bytes = (FIXTURES / "bus.jpg").read_bytes()
        result = detect(image_bytes)
        for d in result.detections:
            assert 0.0 <= d.confidence <= 1.0, f"confidence {d.confidence} outside [0,1]"

    def test_result_fields_match_schema(self):
        """Verify all fields required by docs/response-schema.md are present."""
        image_bytes = (FIXTURES / "bus.jpg").read_bytes()
        result = detect(image_bytes)
        # Top-level fields
        assert hasattr(result, "detections")
        assert hasattr(result, "image_width")
        assert hasattr(result, "image_height")
        assert hasattr(result, "model")
        assert hasattr(result, "count")
        # Per-detection fields
        for d in result.detections:
            assert hasattr(d, "label")
            assert hasattr(d, "class_id")
            assert hasattr(d, "confidence")
            assert hasattr(d, "box")
            assert hasattr(d.box, "x1")
            assert hasattr(d.box, "y1")
            assert hasattr(d.box, "x2")
            assert hasattr(d.box, "y2")
