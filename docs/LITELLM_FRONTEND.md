# LiteLLM front end — one door for the SparkPipe fleet

Status, 2026-09-28: `config/litellm-config.yaml` routes `glm-5.3-flash` to
the GLM API on the rtx5090 hub. From the controller Mac that upstream answers
`GET /health` with `"tokenizer":true` and `GET /v1/models`. No completion
through the proxy against it has a receipt yet, and no proxy on the
controller Mac runs the committed config (the one LiteLLM process there
listens on `127.0.0.1:4000` with a different config file). The browser checks
below used a mock upstream.

## What this is

- A standard open-source LiteLLM proxy on the controller Mac: one
  OpenAI-compatible door at `http://<mac>:4000` with bearer-key auth,
  model-name routing and access logging.
- Its upstreams are per-deployment `model_api` instances. GLM 5.3 Flash's is
  `sparkpipe_model_api` in the systemd user unit `g53-api` on the rtx5090,
  port 8433, an x86 build of the engines' source commit
  ([FLEET_RELEASE_RUNBOOK.md](FLEET_RELEASE_RUNBOOK.md) §6; the COMPSEC-17
  receipt below lists the build). No Spark serves an API: the serving drop-in sets
  `G5_API_DISABLED=1`, and `ensure_api` in `tools/fleet_node_agent.sh` then
  starts none.
- Nothing changes on the Sparks or the hub to add the door.

## The model_api contract (read first)

`model_api` (`node/model_api.c`) accepts OpenAI-shaped requests. Text
prompts need a tokenizer sidecar in the deployment config; token IDs always
work. `model_api` checks no API key and ignores the request's `model` field.

| Route | Request | Response |
| --- | --- | --- |
| `GET /health` | — | `{"status":"ok","served":N,"tokenizer":bool}` |
| `GET /v1/models` | — | `{"object":"list","data":[{"id":...}]}`; the id is `SPARK_MODEL_ID`, default `sparkpipe-model` |
| `POST /v1/completions` | `prompt` (text) or `prompt_token_ids` | `text_completion` with `choices[0].text`, `finish_reason`, `usage`, `tokens` |
| `POST /v1/chat/completions` | `messages`, or `prompt` or `prompt_token_ids` | `chat.completion` with `choices[0].message`, `finish_reason`, `usage`, `tokens` |
| anything else | — | `404 {"error":"not found"}` |

A request with `messages` and no `prompt` or `prompt_token_ids` gets the GLM
chat layout (`api_build_chat_prompt`): `[gMASK]<sop>`, a role marker before
each turn (`<|system|>`, `<|user|>`, `<|assistant|>`, `<|observation|>`) and
a trailing `<|assistant|>\n`. Unless the request names `stop_token_ids`, the
reply stops at the next turn marker. The layout does not append
`<think></think>`, so the model reasons before it answers.

Optional request fields:

| Field | Meaning |
| --- | --- |
| `max_tokens` | output budget, default 32, cap 8192 |
| `stop_token_ids` | extra stop tokens, added to the model's EOS set |
| `stream` | `true` answers `text/event-stream`: one `data:` event per batch of new tokens (`text_completion` or `chat.completion.chunk` with a text or `delta.content` piece that never splits a UTF-8 sequence), a final event with `finish_reason` and `usage`, then `data: [DONE]` |
| `priority` | unsigned integer; higher runs first, with aging in the batch engine so lower priorities cannot starve |
| `deadline_ms` | relative deadline; queued requests are submitted earliest-deadline first within a priority, and a request still running at its deadline is cancelled and answered `504` with code `deadline_exceeded` (or an error event on a stream) |
| `temperature` | `0` (the default) decodes greedily; `0.0001` to `2` samples from softmax(logits / T) with Gumbel-max noise keyed by (seed, position, token) |
| `seed` | unsigned 64-bit; the same seed, prompt and deployment reproduce a sampled completion token for token. Without one, the API draws a random seed and logs it |
| `top_p` | accepted only as `1`: nucleus sampling is not implemented |

A malformed `stream`, `priority`, `deadline_ms`, `temperature`, `seed` or
`top_p` is a `400 invalid_option`, never a silent default. A nonzero
temperature on a deployment whose adapter cannot sample is a `400
sampling_unsupported`. A prompt plus `max_tokens` that the
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
the session fingerprint, the priority, the stream flag, the temperature and
seed, the finish reason and
every output token with its timestamp. That is enough to replay a
completion off-node and compare it bit for bit.

## Calling GLM 5.3 Flash through the door

Chat, thinking on (the default path for clients):

```sh
curl http://<mac>:4000/v1/chat/completions \
  -H "Authorization: Bearer $SPARK_LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Hello"}],"max_tokens":256}'
```

Thinking off, or any exact prompt: send the templated text as `prompt` to
`/v1/completions`. The API's tokenizer maps the markers to their special
tokens.

```json
{"model": "glm-5.3-flash",
 "prompt": "[gMASK]<sop><|user|>\n{question}<|assistant|>\n<think></think>\n",
 "max_tokens": 512}
```

This is the COMPSEC-17 protocol, which scores 14/17 sent straight to the API
([receipt](../qualification/ds4_eval/runs/glm5-next-tp16-20260928-dd3526b-thinkoff/REPORT.md));
it has not been run through the proxy.

Token IDs go straight to `model_api`, not through the door: LiteLLM's
routes rewrite the body into the OpenAI shape, and a body with only
`prompt_token_ids` failed at the proxy with `KeyError 'prompt'` (LiteLLM
1.74.0, 2026-08-28). A token-ID prompt gets no chat layout, so it must carry
the markers itself. GLM 5.3 Flash IDs: `[gMASK]` 154822, `<sop>` 154824,
`<|user|>` 154827, `<|assistant|>` 154828, `<think>` 154841, `</think>`
154842 (`qualification/ds4_eval/tokenizer/glm-5.3-flash-tokenizer.json`).
Qwen's `<|im_start|>user\n` IDs (151644, 872, 198) are ordinary text pieces
to GLM.

## Browser clients (the sparkpipe.ai playground)

`site/playground.html` calls the door from a browser. Checked on 2026-09-27
against LiteLLM 1.74.0 and a mock upstream that speaks the chat contract
above, with deployments as `tools/generate_litellm_config.py` writes them
(`openai/` provider, `api_base` ending in `/v1`):

- CORS allows any origin, and preflights pass for `authorization` and
  `content-type`, so the page need not share the proxy's origin.
- `GET /health/liveliness` needs no key; the page uses it for its status
  light. `GET /v1/models` needs the key.
- Streamed chat works. With `stream_options: {"include_usage": true}` the
  last event before `[DONE]` carries `usage`, which the page uses for its
  decode rate.
- `seed`, `temperature`, `max_tokens` and `deadline_ms` reach the upstream.
  `priority` does not: LiteLLM takes it for its own scheduler.
- `vllm/` deployments fail the chat route with `No module named 'vllm'`:
  LiteLLM's `vllm/` provider runs a local vLLM engine rather than calling
  the upstream.

## Install (controller Mac)

```sh
python3.13 -m venv ~/litellm-venv
~/litellm-venv/bin/pip install 'litellm[proxy]==1.74.0'
~/litellm-venv/bin/litellm --version   # litellm-1.74.0
```

Pinned: `litellm[proxy]==1.74.0` (with `litellm-proxy-extras-0.2.6`) on
Homebrew Python 3.13.2, the versions every check here used. The 2026-09-05
integration notes give the reasons: newer LiteLLM releases need Postgres for
any auth, and Python 3.14 breaks uvloop. Keep the venv outside the
repository.

## Config

`config/litellm-config.yaml` is generated, never edited by hand:

```sh
python3 tools/generate_litellm_config.py \
  --pair glm-5.3-flash=http://100.123.97.61:8433 \
  --master-key --out config/litellm-config.yaml
```

The generator writes one `openai/<name>` deployment per `--pair` (or per
`--registry models.json` entry, `[{"name": ..., "base_url": ...}]`), with
`api_base` ending in `/v1`, `drop_params: true`, `num_retries: 1` and
`request_timeout: 600`. `--master-key` adds bearer auth through
`SPARK_LITELLM_MASTER_KEY`; the generator omits it by default, so always pass
it for the door. Upstream calls send `SPARK_API_KEY`, which `model_api`
ignores; set it to any value. Older setups named these `LITELLM_MASTER_KEY`
and `SPARKPIPE_UPSTREAM_KEY`.

Keep both secrets in their own mode-600 file outside the repository, not in
the repository's `.env`, which also holds the GitHub PAT:

```sh
mkdir -p ~/.config/sparkpipe
( umask 077; printf 'SPARK_LITELLM_MASTER_KEY="sk-sparkpipe-%s"\nSPARK_API_KEY="unused"\n' \
    "$(openssl rand -hex 16)" > ~/.config/sparkpipe/litellm.env )
```

Routing table (`model_api` HTTP ports, not residentd control ports):

| model_name | Upstream | Evidence |
| --- | --- | --- |
| `glm-5.3-flash` | `g53-api` on the rtx5090, `http://100.123.97.61:8433` (tailscale; `10.10.250.2` from the Sparks) | `GET /health` from the controller Mac on 2026-09-28; `docs/FLEET_RELEASE_RUNBOOK.md` §6 |

Qwen 3.8 27B has no route: all 16 Sparks run the `glm53flash.fp8.tp16` root
(`docs/FLEET_RELEASE_RUNBOOK.md` topology, COMPSEC-17 receipt).

## Start / stop

```sh
# start
cd /tmp && set -a; . ~/.config/sparkpipe/litellm.env; set +a; \
  nohup ~/litellm-venv/bin/litellm \
    --config <repo>/config/litellm-config.yaml --port 4000 \
    > /tmp/litellm-sparkpipe.stdout 2>&1 & echo $! > /tmp/litellm-sparkpipe.pid
# stop
kill $(cat /tmp/litellm-sparkpipe.pid)
```

Uvicorn access lines and LiteLLM error details land in
`/tmp/litellm-sparkpipe.stdout`, the usage log of the database-free setup.
Probe an upstream directly with `curl http://100.123.97.61:8433/health`.
Smoke the door with the standard tester, which sends two temperature-0 text
completions per listed model and requires identical output:

```sh
python3 tools/model_api_smoke.py --endpoint http://127.0.0.1:4000 \
  --token "$SPARK_LITELLM_MASTER_KEY"
```

## Postgres (virtual keys and spend logs)

Per-key budgets, virtual keys and spend tracking in the admin UI (`/ui`)
need a Postgres `database_url`. Install it on the controller Mac, not on a
Spark:

```sh
brew install postgresql@16
/opt/homebrew/opt/postgresql@16/bin/pg_ctl -D /opt/homebrew/var/postgresql@16 start
/opt/homebrew/opt/postgresql@16/bin/createdb litellm
python3 tools/generate_litellm_config.py \
  --pair glm-5.3-flash=http://100.123.97.61:8433 --master-key \
  --database-url postgresql://mac@localhost:5432/litellm \
  --out config/litellm-config.yaml
```

LiteLLM ships a prisma schema; generate the client and push the schema once:

```sh
cd ~/litellm-venv/lib/python3.13/site-packages/litellm/proxy
PATH=~/litellm-venv/bin:$PATH DATABASE_URL=postgresql://mac@localhost:5432/litellm prisma generate
PATH=~/litellm-venv/bin:$PATH DATABASE_URL=postgresql://mac@localhost:5432/litellm prisma db push
```

Then start the proxy with `DATABASE_URL` exported next to
`SPARK_LITELLM_MASTER_KEY`. On 2026-09-05 this setup, with mock upstreams,
served the admin page, routed completions by name and wrote one
`LiteLLM_SpendLogs` row per request.

## Add a model

1. Bring the deployment's `model_api` up on a known host and port and record
   them with the deployment.
2. Regenerate `config/litellm-config.yaml` with one `--pair` per route, the
   existing ones included.
3. Restart the proxy.
4. Prove routing: `GET /v1/models` through the door lists the name, and
   `tools/model_api_smoke.py` through the door passes.
5. Commit the config and add the route and its evidence to the table above.

## History

Until 2026-09-28 the committed config used `vllm/` deployments and LiteLLM's
`/vllm/<path>` passthrough, which forwards a token-ID body verbatim. That
path cannot serve chat clients, and its documented example sent Qwen token
IDs to GLM. The 2026-08-28 bring-up receipts, including the mock passthrough
proof, were removed with `docs/AGENT_LANE_BRIEFS` in 27a2620;
`git show 27a2620^:docs/AGENT_LANE_BRIEFS/reports/litellm-2026-08-28.md`
prints them. The generator and Postgres notes came from
[`archive/LITELLM_INTEGRATION.md`](archive/LITELLM_INTEGRATION.md).
