"""
FastAPI application.
Routes:
  POST /v1/chat/completions  — OpenAI-compatible detection endpoint
  GET  /v1/models            — advertised model list
  GET  /health               — liveness probe
"""

import asyncio
import base64
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.auth import verify_bearer
from app.detection import DetectionResult, detect, get_executor, get_session
from app.errors import openai_error, openai_error_response
from app.schemas import (
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ModelList,
    ModelObject,
)

ADVERTISED_MODEL = "yolo11n"
_MAX_CONCURRENT = max(1, os.cpu_count() or 1)


# ---------------------------------------------------------------------------
# Semaphore: limits simultaneous CPU-bound inferences to available cores
# ---------------------------------------------------------------------------
_semaphore: asyncio.Semaphore | None = None


def get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(_MAX_CONCURRENT)
    return _semaphore


# ---------------------------------------------------------------------------
# App lifecycle: load the ONNX session once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    get_session()  # warm the singleton before accepting requests
    yield


app = FastAPI(title="YOLO OpenAI Detector", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Exception handlers: map FastAPI/Pydantic validation errors to OpenAI shape
# ---------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # Our openai_error() raises HTTPException with detail={"error": {...}}.
    # Return that dict directly instead of FastAPI's default {"detail": ...} wrap.
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": str(exc.detail),
                "type": "api_error",
                "param": None,
                "code": None,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return openai_error_response(
        status_code=400,
        message=f"Request body validation failed: {exc.errors()[0]['msg']}",
        error_type="invalid_request_error",
    )


# ---------------------------------------------------------------------------
# Helper: extract exactly one base64 image from the request messages
# ---------------------------------------------------------------------------
def _extract_image_bytes(req: ChatCompletionRequest) -> bytes:
    """
    Scan all message content blocks for image_url entries.
    Rules:
    - Exactly one image must be present (zero or >1 → 400).
    - Remote http(s) URLs are rejected → 400.
    - Only base64 data URLs are accepted.
    Returns the decoded image bytes.
    """
    found: list[str] = []

    for msg in req.messages:
        if msg.content is None:
            continue
        if isinstance(msg.content, str):
            continue
        for block in msg.content:
            if block.type != "image_url" or block.image_url is None:
                continue
            url = block.image_url.url
            if url.startswith("http://") or url.startswith("https://"):
                raise openai_error(
                    status_code=400,
                    message=(
                        "Remote image URLs are not supported. "
                        "Send the image as a base64 data URL "
                        "(data:image/<type>;base64,<data>)."
                    ),
                )
            found.append(url)

    if len(found) == 0:
        raise openai_error(
            status_code=400,
            message="No image found in the request. "
            "Include exactly one base64 image_url content block.",
        )
    if len(found) > 1:
        raise openai_error(
            status_code=400,
            message=f"Expected exactly one image, got {len(found)}. "
            "Send one image per request.",
        )

    data_url = found[0]
    # Parse data:<mediatype>;base64,<data>
    if not data_url.startswith("data:"):
        raise openai_error(
            status_code=400,
            message="image_url must be a base64 data URL starting with 'data:'.",
        )
    try:
        header, encoded = data_url.split(",", 1)
    except ValueError:
        raise openai_error(
            status_code=400,
            message="Malformed data URL: missing comma separator.",
        ) from None
    if "base64" not in header:
        raise openai_error(
            status_code=400,
            message="Only base64-encoded images are supported.",
        )
    try:
        return base64.b64decode(encoded)
    except Exception as b64_err:
        raise openai_error(
            status_code=400,
            message="Failed to decode base64 image data.",
        ) from b64_err


def _serialize_result(result: DetectionResult) -> str:
    """Serialize DetectionResult to the docs/response-schema.md JSON string."""
    payload = {
        "detections": [
            {
                "label": d.label,
                "class_id": d.class_id,
                "confidence": d.confidence,
                "box": {"x1": d.box.x1, "y1": d.box.y1, "x2": d.box.x2, "y2": d.box.y2},
            }
            for d in result.detections
        ],
        "image": {"width": result.image_width, "height": result.image_height},
        "model": result.model,
        "count": result.count,
    }
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models", response_model=ModelList)
async def list_models(request: Request):
    verify_bearer(request)
    return ModelList(data=[ModelObject(id=ADVERTISED_MODEL)])


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    verify_bearer(request)

    # Parse body; JSON errors → 400, Pydantic validation errors → handled above.
    try:
        body = await request.json()
    except Exception as json_err:
        raise openai_error(
            status_code=400, message="Request body is not valid JSON."
        ) from json_err
    req = ChatCompletionRequest.model_validate(body)

    # Model check
    if req.model != ADVERTISED_MODEL:
        raise openai_error(
            status_code=404,
            message=f"Model '{req.model}' not found. Available: {ADVERTISED_MODEL}",
            code="model_not_found",
        )

    # Streaming not yet supported (WO-03)
    if req.stream:
        raise openai_error(
            status_code=400,
            message="Streaming is not yet supported by this service.",
            code="streaming_not_supported",
        )

    image_bytes = _extract_image_bytes(req)

    # Run inference off the event loop under the bounded concurrency semaphore.
    loop = asyncio.get_event_loop()
    try:
        async with get_semaphore():
            result: DetectionResult = await loop.run_in_executor(
                get_executor(), detect, image_bytes
            )
    except Exception as exc:
        raise openai_error(
            status_code=500,
            message=f"Inference failed: {exc}",
            error_type="api_error",
        ) from exc

    content = _serialize_result(result)
    response = ChatCompletionResponse(
        model=ADVERTISED_MODEL,
        choices=[Choice(message=AssistantMessage(content=content))],
    )
    return response.model_dump()
