# Work Orders

Implementation proceeds as a sequence of **PR-sized** tasks. Each work order is
self-contained: an execution agent reads `AGENTS.md` first, then the work order,
does exactly that task on a feature branch, opens a PR, and does **not** merge.

The repository's **initial governance commit** (this constitution, docs, and
structure) is made by the human directly, not via PR — so the implementation
sequence below starts at the first coding task.

## Planned sequence

| WO | Title | Goal | Status |
|----|-------|------|--------|
| WO-01 | Inference core | Offline export script + pure detection function (image bytes → detections), with a real-detection test. No HTTP. | **Ready** → `PR1-inference-core.md` |
| WO-02 | OpenAI endpoint | `/v1/chat/completions` + `/v1/models`, Pydantic schemas, single-image extraction, off-loop inference, fixed-key auth, fail-closed errors, `/health`. | Pending WO-01 |
| WO-03 | Compatibility hardening | `stream:true` emulation, full OpenAI error taxonomy, `usage` stub, and an end-to-end test using the real `openai` client. | Pending WO-02 |

Later work orders (e.g. YOLO26n evaluation, CoreML path) require a constitution
update first — see `AGENTS.md` §11.

## Authoring rule

Work orders are written by the strategic layer (with the human lead), not
improvised by the executor. If a work order is wrong or blocked, the executor
reports rather than expanding scope.
