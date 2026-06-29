# OpenAI Compatibility

This is the contract. Existing OpenAI client code must work against this service
with only `base_url` and `api_key` changed. Anything that deviates is listed
here and enforced in code and tests. Keep this file in sync with behavior.

## Endpoints implemented

| Method | Path | Notes |
|--------|------|-------|
| POST | `/v1/chat/completions` | The detection endpoint. |
| GET | `/v1/models` | Returns the advertised model(s) in OpenAI list shape. |
| GET | `/health` | Non-OpenAI. Liveness/readiness probe. |

Any other `/v1/...` path returns an OpenAI-style `404` error object.

## Authentication

- `Authorization: Bearer <key>` is required.
- The key is compared **constant-time** against the fixed `API_KEY`.
- Missing/invalid key → `401` with an OpenAI-style error object. Fail closed.

## Request fields

| Field | Support |
|-------|---------|
| `model` | Validated against the advertised name (`yolo11n`); echoed back. Unknown model → `404` model-not-found error. |
| `messages` | Required. Exactly one image must appear across the user message content. |
| content block `type: "text"` | Accepted and **ignored** (the image is the input). |
| content block `type: "image_url"` with base64 data URL | **Required.** Exactly one. |
| `image_url` with remote `http(s)` URL | **Rejected** → `400` error. Base64 only. |
| more than one image | **Rejected** → `400` error. |
| zero images | **Rejected** → `400` error. |
| `detail` (low/high/original/auto) | Accepted and **ignored**. |
| `stream` | `false`/absent → normal response. `true` → emulated SSE (see below). |
| `temperature`, `top_p`, `max_tokens`, `n`, etc. | Accepted and **ignored** (no effect on detection). |
| `response_format` | Accepted; content is always the detection JSON regardless. |

## Response shape

A standard `chat.completion` object:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1750000000,
  "model": "yolo11n",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "<detection JSON string>"},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}
```

- The detection payload is a **JSON string** in `choices[0].message.content`,
  conforming to `response-schema.md`.
- `usage` is always zero — there are no LLM tokens. We never fabricate counts.

## Streaming (`stream: true`)

To avoid breaking streaming clients, streaming is **emulated**, not real:

1. Emit a single `data:` line containing one `chat.completion.chunk` whose
   `delta.content` is the full detection JSON string.
2. Emit `data: [DONE]`.

There is no token-by-token streaming because the result is computed in one shot.

## Errors

All errors use the OpenAI error envelope and appropriate HTTP status codes:

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": null, "code": null}}
```

| Condition | HTTP | `type` |
|-----------|------|--------|
| Missing/invalid key | 401 | `invalid_request_error` |
| Unknown model | 404 | `invalid_request_error` |
| No image / >1 image / remote URL | 400 | `invalid_request_error` |
| Undecodable or unsupported image | 400 | `invalid_request_error` |
| Server/inference failure | 500 | `api_error` |

## Compatibility test requirement

A test must exercise the **real `openai` Python client** against the running
service (base64 image in, JSON detections out) so "unchanged client code works"
is proven, not assumed.
