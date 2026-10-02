# LiteLLM front end — one door for the SparkPipe fleet

Status, 2026-10-02: GLM-5.3 Full (TP16, 65,536 positions per sequence) serves
through this stack on the rtx5090 hub:

| Port | Process | Protocol |
| --- | --- | --- |
| 4000 | LiteLLM proxy (`config/litellm-config.yaml`, user unit `sparkpipe-litellm`) | OpenAI `/v1/chat/completions` and Anthropic `/v1/messages`; any model name routes to `glm-5.3` |
| 8433 | chat layer (`serving/chat_frontend.py`, user unit `sparkpipe-chat`) | OpenAI `/v1/chat/completions`, `/v1/models`, `/health`; the sparkpipe.ai tunnel (`sparkpipe-ai-door`) forwards here |
| 8446 | `sparkpipe_model_api` (user unit `glmfull-api6`) | the engine endpoint: `prompt_token_ids` in, token ids, usage and `cached_tokens` out |

Smoke receipts (2026-10-02, build de944ae): chat answered "Paris" with the
reasoning returned separately; a two-turn tool call (`get_weather`) returned
standard `tool_calls`, then answered from the tool result with 192 of 225
prompt tokens served from the prefix cache; LiteLLM `/v1/messages` returned a
`thinking` block and a `tool_use` block with `stop_reason: tool_use`, and
streamed `thinking_delta` events; `https://sparkpipe.ai/health/liveliness`
and `/v1/models` answered through the tunnel.

## The chat layer

`serving/chat_frontend.py` is what vLLM is behind LiteLLM elsewhere, and
nothing more:

- it renders the model's own `chat_template.jinja` with jinja2 in the same
  sandboxed environment and filters Hugging Face uses (`trim_blocks`,
  `lstrip_blocks`, `loopcontrols`, `tojson`), so tools, tool results and
  reasoning render exactly as the model was trained;
- it tokenizes with the model's `tokenizer.json` through Hugging Face
  `tokenizers` and sends `prompt_token_ids` to the engine endpoint;
- it decodes the returned token ids incrementally and splits the model's
  reasoning and tool-call markup into standard `reasoning_content` and
  `tool_calls` (tool arguments typed from the tool's JSON schema);
- it keeps no state; prefix reuse happens in the engine.

The layer is model-neutral. A model family supplies its configuration
(`model-families/glm52/serving/chat_frontend.json`): the model directory
(`chat_template.jinja`, `tokenizer.json`, `tokenizer_config.json`), the
engine URL, the served context, stop tokens, the reasoning and tool-call
markers, the accepted `chat_template_kwargs` and the mapping of OpenAI
`reasoning_effort` onto the template's own variable.

Known limits:

- GLM-5.3 Full decodes greedily: the glm52 adapter has no sampled head, so
  `temperature` and `top_p` do not change the output.
- `n`, `logprobs`, `logit_bias` and a `response_format` other than text
  answer 400.
- Prompts past `context_tokens` answer 400 `context_length_exceeded`; a
  client must compact before 65,536 tokens.
- A 60K-token prompt takes about 90 s to prefill the first time; later turns
  that share its prefix reuse it.

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

A request with `messages` and no `prompt` or `prompt_token_ids` is rendered
with the chat template the deployment declares (`chat_template` in
`model_resident.json`, see below). A deployment without one answers every
`messages` request `400 chat_template_missing`; send `prompt` or
`prompt_token_ids` instead. The API never falls back to another model's
layout: before this rule it rendered every model with the GLM layout, and a
Gemma 4 channel answered `messages` with garbage (`LCBCBCBC`, 2026-09-29).
Unless the request names `stop_token_ids`, the reply also stops at the
template's `stop_markers`.

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
| `chat_template_kwargs` | `messages` requests only, exactly `{"enable_thinking": bool}`; default `false` |

### Declared chat template

The `chat_template` member of the API channel's `model_resident.json` has
exactly these members; the model family ships its declaration
(`model-families/glm5_next/chat_template.json`,
`model-families/gemma4/chat_template.json`), and
`tools/generate_model_resident_deployment.py` copies a `chat_template`
object from the deployment specification after the same checks.

| Member | Kind | Meaning |
| --- | --- | --- |
| `prefix` | string | text before the first turn (`[gMASK]<sop>`, `<bos>`) |
| `thinking_prefix` | string | with `enable_thinking`, text after the prefix when the first message is not `system` (Gemma's `<|think|>` system turn); `""` otherwise |
| `system` | string or null | header of a `system` turn; null refuses `system` messages |
| `system_thinking` | string or null | header of a leading `system` turn with `enable_thinking`; declared together with `system` |
| `user` | nonempty string | header of a `user` turn |
| `observation` | string or null | header of an `observation` turn; null refuses them |
| `assistant` | nonempty string | header of a past `assistant` turn |
| `assistant_thinking` | string or null | the same with `enable_thinking`; declared together with `generation_thinking` |
| `turn_suffix` | string | text after every turn's content (`""`, `<turn|>\n`) |
| `generation` | nonempty string | generation prompt after the last turn |
| `generation_thinking` | string or null | generation prompt with `enable_thinking`; null answers `400 thinking_unsupported` |
| `stop_markers` | 1 to 8 nonempty strings | each must be exactly one special token of the channel tokenizer, or `model_api` refuses to start |

A turn is `header + content + turn_suffix`. Roles other than `system`,
`user`, `assistant` and `observation`, and roles the template declares null,
are `400 role_unsupported`; a message that is not an object with string
`role` and `content`, or an empty `messages` array, is `400
invalid_messages`. Both answers name the code; nothing is dropped or mapped
to another role.

The GLM declaration reproduces the earlier fixed layout byte for byte:
`[gMASK]<sop>`, `<|system|>\n`, `<|user|>\n`, `<|observation|>\n`, and the
assistant header `<|assistant|>\n<think></think>\n` with thinking off (the
layout COMPSEC-17 validates, `tools/glm5_next_compsec17.py --thinking off`)
or `<|assistant|>\n<think>` with `enable_thinking`. Past assistant turns use
the same header as the generation prompt, so the next turn's prompt extends
the previous prompt and reply and reuses its cached prefix.

The Gemma 4 declaration follows the publisher template for text turns:
`<bos>`, `<|turn>user\n...<turn|>\n`, generation `<|turn>model\n` plus
`<|channel>thought\n<channel|>` with thinking off, and with
`enable_thinking` a `<|turn>system\n<|think|>\n` system turn that also
carries a leading system message. It does not declare tool turns
(`observation` is null), and system content is sent as given (the publisher
template trims it).

An older `model_api` refuses a `model_resident.json` that carries
`chat_template` (unknown member), so a channel gains the block in the same
install as the API binary that reads it.

A malformed `stream`, `priority`, `deadline_ms`, `temperature`, `seed`,
`top_p` or `chat_template_kwargs`, or `chat_template_kwargs` on a
`prompt` or `prompt_token_ids` request, is a `400 invalid_option`, never a
silent default. A nonzero
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

## Calling GLM-5.3 Full through the door

OpenAI clients (tools, streaming, `reasoning_effort` low, medium or high):

```sh
curl http://100.123.97.61:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5.3","messages":[{"role":"user","content":"Hello"}],"max_tokens":2048}'
```

Anthropic clients such as Claude Code: `ANTHROPIC_BASE_URL=http://100.123.97.61:4000`,
any `ANTHROPIC_AUTH_TOKEN`, and `ANTHROPIC_MODEL=glm-5.3`. The proxy maps any
model name to `glm-5.3`, so the client's small-model calls land there too.

The sections below describe the earlier GLM 5.3 Flash door.

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
| `glm-5.3` and `*` | the chat layer on the rtx5090, `http://127.0.0.1:8433/v1` (LiteLLM runs on the hub) | smoke receipts above, 2026-10-02 |

Qwen 3.8 27B has no route: all 16 Sparks run the `glm53flash.fp8.tp16` root
(`docs/FLEET_RELEASE_RUNBOOK.md` §2.1, COMPSEC-17 receipt).

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
