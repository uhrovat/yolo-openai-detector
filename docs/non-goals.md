# Non-Goals

These are out of scope **by decision**, not by omission. They are forbidden in
`AGENTS.md` §5. This file records the rationale so the boundaries are not eroded
by well-meaning "improvements".

| Non-goal | Why it is excluded |
|----------|--------------------|
| Object **tracking** / persistent IDs | The product is per-image detection. Tracking is stateful and temporal; it would change the whole architecture. |
| **Streaming video** / frame sequences / sessions | Same reason. The OpenAI request/response model is stateless; we keep it that way. |
| **Segmentation**, masks, pose, OBB, classification | Detection (boxes) only. Other tasks expand scope, model size, and output schema. |
| **Multiple images** per request | One image, one response. Keeps the contract and resource use predictable on a laptop. |
| **Remote image URL** fetching | Base64 only. Avoids SSRF and any outbound network requirement at request time. |
| **Database / persistence** | Nothing needs to be stored. Avoids data-handling risk and operational weight. |
| **Background jobs / queue / scheduler** | Detection is synchronous and fast enough; async infrastructure is accidental complexity. |
| **Per-user keys, quota, accounting, billing** | One fixed shared key. This is not a metered multi-tenant gateway. |
| **Real text generation / embeddings / audio / image generation** | This is a detector wearing an OpenAI interface, not a general model. |
| **`torch` / `ultralytics` at runtime** | Keep the served process lean; export happens offline. |
| **GPU / CUDA acceleration** | The target is a GPU-less laptop. |

## How to change a non-goal

A non-goal becomes scope only by: (1) a human decision, (2) updating `AGENTS.md`
and this file, and (3) a new work order. Agents must not cross these lines on
their own initiative, even if a change looks small or helpful.
