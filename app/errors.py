"""
OpenAI-style error envelope helpers.
All error responses use {"error": {"message", "type", "param", "code"}}.
"""

from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from app.schemas import ErrorDetail, ErrorResponse


def openai_error(
    status_code: int,
    message: str,
    error_type: str = "invalid_request_error",
    param: Any = None,
    code: Any = None,
) -> HTTPException:
    """
    Return an HTTPException whose detail is the OpenAI error JSON.
    Raise the result; FastAPI's exception handler will serialize it.
    """
    body = ErrorResponse(
        error=ErrorDetail(message=message, type=error_type, param=param, code=code)
    )
    return HTTPException(status_code=status_code, detail=body.model_dump())


def openai_error_response(
    status_code: int,
    message: str,
    error_type: str = "invalid_request_error",
    param: Any = None,
    code: Any = None,
) -> JSONResponse:
    """
    Return a JSONResponse directly (for exception handlers that can't raise).
    """
    body = ErrorResponse(
        error=ErrorDetail(message=message, type=error_type, param=param, code=code)
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())
