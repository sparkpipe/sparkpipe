# SparkPipe OpenAI-Compatible HTTP API

Serving endpoint for SparkPipe residents. One gateway fronts every model:
clients speak the OpenAI Chat Completions / Completions wire format (plus an
Anthropic Messages variant), and the gateway folds requests into a
server-side chat template, submits them to the service runtime, and streams
decoded tokens back over SSE.

- Implementation: `api/http_gateway.c`, `api/compat_api.c`,
  `api/gateway/http_server.c` (gateway line; headers
  `include/sparkpipe/spark_http_gateway.h`, `spark_compat_api.h`,
  `spark_service.h`)
- Gateway ABI version: `SPARK_HTTP_GATEWAY_ABI_VERSION = 1`
- Compat request ABI version: `SPARK_COMPAT_API_ABI_VERSION = 2`
- Executable: `build/sparkpipe_http_gateway`

Status: this document describes implemented behavior, including known
deviations from the OpenAI wire format (see [Deviations](#10-deviations-from-openai)).

---

## 1. Endpoints

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET | `/` | Built-in demo UI (browser page) |
| GET | `/health` | Runtime observation document (JSON) |
| POST | `/v1/chat/completions` | OpenAI chat completions |
| POST | `/v1/completions` | OpenAI text completions (`prompt`) |
| POST | `/v1/messages` | Anthropic Messages (`system` + `messages`) |
| OPTIONS | *any* | CORS preflight → `200`, empty body |
| other | *any* | `404` `{"error":{"type":"not_found",...}}` |

The service is single-model: the resident loaded behind the gateway answers
every request. The client `model` field is accepted but ignored.

## 2. Authentication

Optional Bearer auth.

- If the gateway was started **without** `--api-key` / `--api-key-file`
  (or the key is empty), all requests are accepted.
- Otherwise the `Authorization` header must match `Bearer <key>` exactly.
  On mismatch:

  ```json
  HTTP/1.1 401 Unauthorized
  {"error":{"type":"unauthorized","message":"missing or invalid bearer token"}}
  ```

The key applies to every route except `OPTIONS` preflight. Auth failures are
checked before backend availability.

## 3. Request schema (chat completions)

```json
{
  "model": "ignored",
  "messages": [
    {"role": "system",    "content": "..."},
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "tool",      "content": "..."}
  ],
  "max_tokens": 1024,
  "stream": true,
  "thinking_budget_tokens": 512,
  "priority": 0,
  "files": [
    {"filename": "notes.txt", "content": "attached text"}
  ]
}
```

### 3.1 Fields

| Field | Type | Default | Notes |
| ----- | ---- | ------- | ----- |
| `messages` | array | — | Required unless `prompt` is used. Must be non-empty. Roles: `user`, `system`, `assistant`, `tool`. Rendered through the server-side chat template. |
| `prompt` | string | — | `/v1/completions` form; raw text, no template wrapping. |
| `max_tokens` | integer | service default | Output token budget. Alias `max_completion_tokens`; last one present wins. |
| `stream` | bool | `false` | Detected by a textual scan of the body for `"stream"` followed by `true`. |
| `thinking_budget_tokens` | integer | see below | Thinking budget. Alias `thinking_token_budget`; specifying both with different values is an error. |
| `priority` | integer | `0` | Passed to the scheduler. |
| `files` | array | — | SparkPipe extension; also accepted as `attachments`. See §3.2. |

Thinking-mode defaults: with neither thinking field present the template runs
with generation prompt **and thinking enabled** (the default template flags).
If either field is present, thinking is enabled only when the value is > 0.

Sampling parameters (`temperature`, `top_p`, `stop`, `n`, `tools`, ...) are
not parsed; unknown members are ignored.

### 3.2 Message content variants and file attachments

Each message's `content` may be:

- a plain string, or
- an array whose elements are strings or objects:

  - `{"text": "..."}` — inline text;
  - `{"filename"|"file_name"|"name": "...", "content"|"file_content"|"file_text"|"text"|"data": "..."}`
    — an attached text file, folded into the prompt as
    `\n[uploaded file: NAME]\n…content…\n[/uploaded file]\n`
    (a file object without any content field is rejected);
  - `{"source": { ... }}` — Anthropic-style nested source; its content
    fields are read with the same alias set.

Top-level `files` / `attachments` arrays use the same file-object shape.

## 4. Responses

Submission is asynchronous. Both modes return **202 Accepted** first.

### 4.1 Non-streaming

```json
HTTP/1.1 202 Accepted
Content-Type: application/json

{"id":"spreq-1844674407370955161","object":"sparkpipe.request","client_request_id":42,"serving_request_id":1844674407370955161,"sequence_id":7,"prompt_tokens":128,"output_token_budget":1024,"status":"queued"}
```

- `id` / `serving_request_id`: gateway-assigned request identity
  (`spreq-` prefix). Non-stream submissions use the high bit (`2^63`) of the
  request-id space.
- `sequence_id`: assigned conversation sequence in the service runtime.
- Completion content for non-streaming requests is delivered through the
  service event channel, not on the submit response.

### 4.2 Streaming (SSE)

With `"stream": true` the 202 response begins a `text/event-stream` body on
the same connection:

```
event: accepted
data: {"client_request_id":42,"serving_request_id":1844674407370955161,"sequence_id":7,"prompt_tokens":128,"output_token_budget":1024}

event: token
data: {"kind":3,"status":0,"client_id":1,"client_request_id":42,"serving_request_id":1844674407370955161,"sequence_id":7,"token_id":333,"token_index":4,"text":" Hel"}

event: done
data: {"kind":4,"status":0,"client_id":1,"client_request_id":42,"serving_request_id":1844674407370955161,"sequence_id":7,"token_id":0,"token_index":9,"text":""}
```

Event names map from service-event kinds:

| SSE `event:` | Service kind | Meaning |
| ------------ | ------------ | ------- |
| `accepted` | (submit result) | Request admitted |
| `event` | `REQUEST_ACCEPTED`, `PREFILL_PROGRESS`, `BACKPRESSURE`, `CLIENT_CONNECTED/DISCONNECTED`, `STATS` | Progress |
| `token` | `TOKEN` | One decoded token; `text` is the detokenized piece (special tokens skipped) |
| `done` | `REQUEST_COMPLETED`, `REQUEST_CANCELLED`, `ERROR` | Terminal; check `kind`/`status` |

Every frame carries the same envelope: `kind`, `status`, `client_id`,
`client_request_id`, `serving_request_id`, `sequence_id`, `token_id`,
`token_index`, `text` (JSON-escaped; control characters emitted as short
escapes or `\u00xx`). Token frames decode one token at a time with
`SKIP_SPECIAL_TOKENS`; if decoding fails the frame carries empty `text`.

## 5. Errors

All error bodies share the OpenAI-style envelope:

```json
{"error":{"type":"<type>","message":"<human-readable>"}}
```

| HTTP | `type` | When | Streaming variant |
| ---- | ------ | ---- | ----------------- |
| 400 | `bad_request` | Malformed HTTP request framing (oversized/unparseable head) | no |
| 401 | `unauthorized` | Bearer token configured and missing/mismatched | no |
| 404 | `not_found` | Unknown method/path | no |
| 503 | `backend_unavailable` | Backend not attached or runtime not initialized; **also** returned when request parsing or submission fails (invalid JSON, empty `messages`, bad role, oversized prompt) | yes (`event: error` frame) |
| 504 | `request_timeout` | Submission reported `BUSY` — no terminal event within the stream poll budget | yes (`event: error` frame) |

Streaming errors are emitted as a final `event: error` frame carrying the
same error object, with the matching HTTP status on the stream start.

Known deviation: JSON validation failures currently surface as
`503 backend_unavailable` rather than `400`; clients should treat a 503
whose message does not indicate a missing backend as a request problem.

## 6. Health and observability

`GET /health` returns a `sparkpipe.runtime_observation.v1` document.
Fields include (non-exhaustive):

- Identity: `release_identity_status`, `release_id`, `release_git_commit`,
  `release_generation`.
- Readiness: `runtime_initialized`, `local_control_ready`.
- Capacity: `configured_kv_context_limit_tokens`,
  `configured_max_active_sequences`, `adaptive_decode_batch_width`,
  `decode_batch_capacity`, `prefill_wave_token_count`.
- Feature truth flags (fail-closed vocabulary): `end_to_end_observation_status`,
  `multi_sequence_batching_status`, `jit_kv_status`, `dspark_status`,
  `mtp_status`, `transport_binding_status` — each `OBSERVED` /
  `NOT_MEASURED` / `NOT_WORKING`. `accuracy_status` and
  `performance_status` are fixed `NOT_MEASURED` placeholders.
- Live counters: `connected_clients`, `live_requests`, `queued_requests`,
  `completed_streams`, `prefill_dispatches`, `prefill_batch_dispatches`,
  `prefill_tokens`, `decode_dispatches`, `decoded_tokens`, lane maxima,
  `event_backlog`, `dropped_events`, JIT-KV prefetch counters, MTP
  draft/verify/accept counters, and `first_blocker`.

When no backend is attached a minimal variant is served
(`runtime_initialized` / `local_control_ready` only).

`GET /` serves a small demo page that exercises streaming chat with
attachments against `/v1/chat/completions`.

## 7. Running the gateway

```
build/sparkpipe_http_gateway [flags]
```

Defaults: binds `127.0.0.1:8080`; DSpark speculation **on** (disable with
`--no-dspark` or env `SPARKPIPE_DISABLE_DSPARK`); MTP off unless enabled;
`--max-active` defaults to 1024; ring transport base port 52100.

Flags:

| Flag | Purpose |
| ---- | ------- |
| `--bind ADDR`, `--port N` | Listen address/port (default `127.0.0.1:8080`) |
| `--api-key KEY`, `--api-key-file F` | Enable Bearer auth (file contents trimmed of trailing newline/space) |
| `--service-backend-so SO` | Attach the model/service backend library (implies `--require-service-backend`) |
| `--require-service-backend` | Refuse to serve without a backend |
| `--pump-steps N` | Service pump iterations per non-stream poll |
| `--chat-template NAME` | Server-side chat template selection |
| `--tokenizer PATH` | Tokenizer for SSE detokenization |
| `--moe-pack-root DIR`, `--stagepack-root DIR` | Pack roots |
| `--transport-so SO`, `--driver-so SO`, `--node-context-builder-so SO` | Component libraries |
| `--embedding-pack PATH`, `--cuda-resident-socket PATH` | Resident attach points |
| `--program NAME`, `--node-target T` | Driver program / node target |
| `--max-active N`, `--kv-logical-blocks N`, `--model-quantization M` | Capacity tuning |
| `--dspark`, `--no-dspark`, `--mtp B` | Speculation modes |
| `--port-base N` | Ring transport base port |
| `--final-event-bind A`, `--final-event-return-host H` | Final-event channel binding |

Environment: `SPARKPIPE_DISABLE_DSPARK` (set = speculation off),
`SPARKPIPE_GATEWAY_DRAIN_EVENTS` (event-drain policy override).

## 8. Limits

| Limit | Value |
| ----- | ----- |
| Max upload (request body) | 64 MiB (`SPARK_SERVICE_MAX_TEXT_BYTES`) |
| Rendered prompt budget | Same 64 MiB pool (`SPARK_GATEWAY_COMPAT_TEXT_BYTES`) |
| Per-request read buffer | upload limit + 1 MiB |
| SSE frame buffer | 4 KiB per pending-stream frame |
| Concurrent streams | Pending-stream slot table sized by a 16-bit slot space |
| Output context ceiling | `SPARK_SERVICE_MAX_TOKEN_FRAME_COUNT` token frames |

Oversized or unparseable requests fail closed (§5).

## 9. Examples

Non-streaming chat:

```sh
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'authorization: Bearer SECRET' \
  -d '{"messages":[{"role":"user","content":"Summarize this."}],"max_tokens":256}'
# → 202 {"id":"spreq-...","object":"sparkpipe.request",...,"status":"queued"}
```

Streaming chat with attachments:

```sh
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"stream":true,"max_tokens":1024,"messages":[{"role":"user","content":"Summarize the attached file."}],"files":[{"filename":"a.txt","content":"..."}]}'
# → event: accepted … event: token … event: done
```

Legacy completion prompt:

```sh
curl -sS http://127.0.0.1:8080/v1/completions \
  -H 'content-type: application/json' \
  -d '{"prompt":"Once upon a time","max_tokens":64}'
```

Anthropic messages:

```sh
curl -sS http://127.0.0.1:8080/v1/messages \
  -H 'content-type: application/json' \
  -d '{"system":"Be brief.","messages":[{"role":"user","content":[{"type":"text","text":"Hi"}]}]}'
```

Health:

```sh
curl -sS http://127.0.0.1:8080/health
```

## 10. Deviations from OpenAI

Clients ported from OpenAI SDKs should account for:

1. **Asynchronous submission** — success is `202 Accepted` with a queued
   receipt (`object: "sparkpipe.request"`), not a synchronous completion.
2. **SSE dialect** — frames are named `accepted`/`token`/`done`/`event`/
   `error` with a flat data envelope; there are no `chat.completion.chunk`
   objects and no `choices[]` wrapper.
3. **No usage block** today; `prompt_tokens` and `output_token_budget`
   appear at accept time instead.
4. `model` and sampling parameters are ignored (single-model service,
   server-configured decoding).
5. Request-validation problems surface as `503 backend_unavailable`
   (see §5 note) rather than `400`.
6. File attachments (`files`/`attachments`) and the thinking-budget fields
   are SparkPipe extensions.

---

*Source of truth: the gateway implementation tree (`api/http_gateway.c`,
`api/compat_api.c`, `api/gateway/http_server.c`) as of 2026-08-23; the
executable spec lives in `tests/test_glm52_http_gateway.c` and
`tests/test_glm52_compat_api.c`. Update this document alongside those
files.*
