"""
Constant-time bearer-key authentication.
The key is read from the API_KEY environment variable.
Never logs the key value. Fails closed on any mismatch.
"""

import hmac
import os

from fastapi import Request
from fastapi.security.utils import get_authorization_scheme_param

from app.errors import openai_error


def _get_expected_key() -> str:
    key = os.environ.get("API_KEY", "")
    if not key:
        raise RuntimeError("API_KEY environment variable is not set")
    return key


def verify_bearer(request: Request) -> None:
    """
    Raise an HTTP 401 JSONResponse if the bearer token is missing or wrong.
    Uses hmac.compare_digest for constant-time comparison.
    """
    authorization = request.headers.get("Authorization", "")
    scheme, token = get_authorization_scheme_param(authorization)

    if not authorization or scheme.lower() != "bearer" or not token:
        raise openai_error(
            status_code=401,
            message="Missing or malformed Authorization header. "
            "Provide: Authorization: Bearer <key>",
            error_type="invalid_request_error",
        )

    expected = _get_expected_key()
    if not hmac.compare_digest(token.encode(), expected.encode()):
        raise openai_error(
            status_code=401,
            message="Invalid API key.",
            error_type="invalid_request_error",
        )
