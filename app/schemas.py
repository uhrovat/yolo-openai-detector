"""
Pydantic v2 models for the OpenAI chat/completions request and response shapes.
Only the fields this service reads or emits are modelled; all others are accepted
and silently dropped by Pydantic's model_config extra="ignore".
"""

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Shared config: extra fields are ignored (not rejected)
# ---------------------------------------------------------------------------
class _OpenAIBase(BaseModel):
    model_config = ConfigDict(extra="ignore")


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------
class ImageURL(_OpenAIBase):
    url: str
    detail: str | None = None


class ContentBlock(_OpenAIBase):
    type: str
    text: str | None = None
    image_url: ImageURL | None = None


class Message(_OpenAIBase):
    role: str
    content: str | list[ContentBlock] | None = None


class ChatCompletionRequest(_OpenAIBase):
    model: str
    messages: list[Message]
    stream: bool = False
    # Everything else (temperature, top_p, max_tokens, n, …) is accepted and
    # dropped by extra="ignore".


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------
class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: AssistantMessage
    finish_reason: Literal["stop"] = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


# ---------------------------------------------------------------------------
# OpenAI models list
# ---------------------------------------------------------------------------
class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "local"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelObject]


# ---------------------------------------------------------------------------
# OpenAI error envelope
# ---------------------------------------------------------------------------
class ErrorDetail(BaseModel):
    message: str
    type: str
    param: Any = None
    code: Any = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
