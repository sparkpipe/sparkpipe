# LiteLLM front end — one door for the SparkPipe fleet

Status: operational on the controller Mac (LiteLLM proxy v1.74.0), routing
table covers glm5_next TP16 and the Qwen 3.8 27B TP1 staging pair. All three
upstreams were DOWN at bring-up time (glm53 mid-debug; 27B staged but not
launched), so the proof of routing is the per-deployment connect evidence in
`docs/AGENT_LANE_BRIEFS/reports/litellm-2026-08-28.md` plus a
contract-exact mock-upstream verification of the passthrough bridge.

## What this is

- The standard open-source LiteLLM proxy runs on the controller Mac only.
  Sparks keep serving their per-deployment `model_api`; nothing changes on
  any spark (front end is controller-side).
- Clients get one OpenAI-compatible door: `http://<mac>:4000`, bearer-key
  auth, model-name routing (`glm-5.3-flash` → spark0:8433), access logging.

## The model_api contract (read first)

`model_api` (`node/model_api.c`) accepts OpenAI-shaped requests. Text
prompts need a tokenizer sidecar in the deployment config; token IDs always
work.

| Route | Request | Response |
| --- | --- | --- |
| `GET /health` | — | `{"status":"ok","served":N,"tokenizer":bool}` |
| `GET /v1/models` | — | `{"object":"list","data":[{"id":...}]}` |
| `POST /v1/completions` | `prompt` (text) or `prompt_token_ids` | `text_completion` with `choices[0].text`, `finish_reason`, `usage`, `tokens` |
| `POST /v1/chat/completions` | `messages` (text), or `prompt_token_ids` | `chat.completion` with `choices[0].message`, `finish_reason`, `usage`, `tokens` |
| anything else | — | `404 {"error":"not found"}` |

Optional request fields:

| Field | Meaning |
| --- | --- |
| `max_tokens` | output budget, default 32, cap 8192 |
| `stop_token_ids` | extra stop tokens, added to the model's EOS set |
| `stream` | `true` answers `text/event-stream`: one `data:` event per batch of new tokens (`text_completion` or `chat.completion.chunk` with a text or `delta.content` piece that never splits a UTF-8 sequence), a final event with `finish_reason` and `usage`, then `data: [DONE]` |
| `priority` | unsigned integer; higher runs first, with aging in the batch engine so lower priorities cannot starve |
| `deadline_ms` | relative deadline; queued requests are submitted earliest-deadline first within a priority, and a request still running at its deadline is cancelled and answered `504` with code `deadline_exceeded` (or an error event on a stream) |

A malformed `stream`, `priority` or `deadline_ms` is a `400
invalid_option`, never a silent default. A prompt plus `max_tokens` that the
deployment's context or KV pages cannot hold is a `400
context_length_exceeded`; the batch engine's own admission check decides, so
the API never duplicates the limits. Without a tokenizer sidecar the
response carries `tokens` only; with one it carries both text and `tokens`.

A stream opens only after the engine has accepted the request, so every
admission failure is an ordinary status code. Failures after admission
arrive as an error event before `data: [DONE]`. A client that disconnects
mid-stream cancels its request in the engine.

When every engine request slot is taken, new requests wait in the API's
queue instead of failing. The queue submits by priority, then earliest
deadline, then arrival, as slots free up. A queued request still honours
its deadline.

Every request writes one `request_measurements` JSON line to the API log.
It records the prompt's SHA-256, the adapter, model and driver identities,
the session fingerprint, the priority, the stream flag, the finish reason and
every output token with its timestamp. That is enough to replay a
completion off-node and compare it bit for bit.

The passthrough notes below were verified against the earlier token-ID-only
upstream (contract-exact mock in `tools/litellm_mock_upstream.py`, LiteLLM
1.74.0). Now that the upstream accepts OpenAI-shaped text bodies, the
transformed routes are expected to work too, but they have no receipt yet;
the passthrough stays the verified integration until one exists.

1. LiteLLM's **transformed routes** (`/v1/completions`, `/v1/chat/completions`)
   rewrite the body into the OpenAI shape and ignore unknown fields. A body
   carrying only `prompt_token_ids` dies at the proxy (`KeyError 'prompt'`).
   Against the token-ID-only upstream a text prompt was answered
   `400 {"error":"prompt_token_ids required"}`; with a tokenizer sidecar it is
   now accepted, but that path has no LiteLLM receipt yet.
2. LiteLLM's **pass-through route is the integration point.** Clients POST to

   ```
   POST http://<mac>:4000/vllm/v1/completions
   Authorization: Bearer <LITELLM_MASTER_KEY>
   {"model": "glm-5.3-flash", "prompt_token_ids": [...], "max_tokens": 64}
   ```

   The `/vllm/<path>` prefix selects the passthrough handler; `model` names
   the deployment in the config; the rest of the JSON body is forwarded
   **verbatim** (only `model` is rewritten to the deployment's internal name)
   to the deployment's `api_base` + path, and the upstream's response
   bytes/status are returned to the client as-is. Verified: the mock upstream
   received `{"model":"mock-model","prompt_token_ids":[151644,872,198],
   "max_tokens":3}` byte-for-byte and the client got the mock's
   `{"object":"text_completion",...}` unchanged.
3. Deployment model strings **must use the `vllm/` prefix** — the vllm
   provider is the one with a registered passthrough config. `openai/`
   prefix → `Provider openai not found` (500).
4. **Error fidelity caveat:** non-2xx upstream responses surface to the
   client as HTTP 500 with a generic body. The upstream's status text IS
   preserved server-side in the proxy log (`VLLMError: {"error": ...}`,
   `Cannot connect to host spark0:8433 ...`) — triage lives in the log, not
   the client response. Down upstream → 500 + `Cannot connect to host
   <host:port>` in the log naming the exact deployment.
5. LiteLLM's own `GET /health` (per-deployment probing) uses the OpenAI chat
   shape, which our upstream always rejects — it reports our deployments
   "unhealthy" even when they serve fine. Ignore it; probe upstreams directly
   (`curl http://spark0:8433/health`) or send a 1-token completion through
   the passthrough.

## Install (controller Mac)

```sh
python3.13 -m venv /Users/mac/sparkpipe/litellm-venv
/Users/mac/sparkpipe/litellm-venv/bin/pip install 'litellm[proxy]==1.74.0'
/Users/mac/sparkpipe/litellm-venv/bin/litellm --version   # litellm-1.74.0
```

Pinned: `litellm[proxy]==1.74.0` (with `litellm-proxy-extras-0.2.6`), on
Homebrew Python 3.13.2. The venv lives outside the repo at
`/Users/mac/sparkpipe/litellm-venv`.

## Config

Committed at `config/litellm-config.yaml`. Secrets come from the
environment (`os.environ/LITELLM_MASTER_KEY`,
`os.environ/SPARKPIPE_UPSTREAM_KEY`) — generated once into
`/Users/mac/sparkpipe/.env` (mode 600, never committed):

```sh
grep -q LITELLM_MASTER_KEY /Users/mac/sparkpipe/.env || \
  echo 'LITELLM_MASTER_KEY="sk-sparkpipe-<random>"' >> /Users/mac/sparkpipe/.env
```

Routing table (model_api HTTP ports, NOT residentd control ports):

| model_name | api_base | evidence |
| --- | --- | --- |
| `glm-5.3-flash` | `http://spark0:8433` | glm53 lane LAUNCH-STATE.md + `/tmp/g5n_api.log` on spark0 |
| `qwen-3.8-27b` | `http://sparka:8534` + `http://spark9:8434` (pool) | sparka port proven in `~/sparkdata/qwen38.fp8.tp1/co_resident_api.log`; **spark9:8434 provisional** (its api never reached ready; confirm at launch) |
| `qwen-3.8-27b-a` | `http://sparka:8534` | pin one instance |
| `qwen-3.8-27b-9` | `http://spark9:8434` | pin one instance |

## Start / stop

```sh
# start
cd /tmp && set -a; source /Users/mac/sparkpipe/.env; set +a; \
  nohup /Users/mac/sparkpipe/litellm-venv/bin/litellm \
    --config <repo>/config/litellm-config.yaml --port 4000 \
    > /tmp/litellm-sparkpipe.stdout 2>&1 & echo $! > /tmp/litellm-sparkpipe.pid
# stop
kill $(cat /tmp/litellm-sparkpipe.pid)
```

Binds `0.0.0.0:4000`. Uvicorn access lines (`POST /vllm/... 500/200`) plus
LiteLLM error details land in `/tmp/litellm-sparkpipe.stdout` — that is the
usage log in the DB-free setup. Per-key budgets/virtual keys need a Postgres
`database_url`; deliberately not configured yet (single master key is the
auth model today).

## Client contract (the one true way to call)

```sh
KEY=$(grep -o 'sk-sparkpipe-[a-f0-9]*' /Users/mac/sparkpipe/.env)
curl http://<mac>:4000/vllm/v1/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"glm-5.3-flash","prompt_token_ids":[151644,872,198],"max_tokens":64}'
# -> {"object":"text_completion","tokens":[...],"status":0}   (upstream bytes, verbatim)
```

`GET /v1/models` (with the key) lists the routed model names. Deployments
with a tokenizer sidecar accept text prompts and return text; others speak
token IDs end to end.

## Add-a-model procedure

1. The deployment's lane brings its `model_api` up on a known port on its
   head node (record the port in the deployment dir / LAUNCH-STATE.md).
2. Edit `config/litellm-config.yaml`: add

   ```yaml
   - model_name: <fleet-facing-name>
     litellm_params:
       model: vllm/<anything>          # vllm/ prefix is REQUIRED (see above)
       api_base: http://<sparkN>:<api-port>
       api_key: os.environ/SPARKPIPE_UPSTREAM_KEY
   ```

3. Restart the proxy (stop/start above).
4. Prove routing: `GET /v1/models` shows the name; a completion POST
   reaches the upstream (`curl http://<sparkN>:<port>/health` served count
   increments, or the connect-error in the proxy log names the right
   host:port if it is down).
5. Commit config + note the port's evidence in this file's routing table.

## Verification receipts

All raw outputs (model list, per-deployment connect evidence, the mock
passthrough byte-for-byte proof, error-surfacing behavior) are in
`docs/AGENT_LANE_BRIEFS/reports/litellm-2026-08-28.md`.
