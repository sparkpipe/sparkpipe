# LiteLLM front end — one door for the SparkPipe fleet

Status, 2026-10-02: GLM-5.3 Full (TP16, 65,536 positions per sequence) serves
through this stack on the rtx5090 hub:

| Port | Process | Protocol |
| --- | --- | --- |
| 4000 | LiteLLM proxy (`config/litellm-config.yaml`, user unit `sparkpipe-litellm`) | OpenAI `/v1/chat/completions` and Anthropic `/v1/messages`; any model name routes to `glm-5.3` |
| 8433 | chat layer (`tools/serving/chat_frontend.py`, user unit `sparkpipe-chat`, 127.0.0.1 only) | OpenAI `/v1/chat/completions`, `/v1/models`, `/health`; the tunnel forwards sparkpipe.ai:8443/fleet here |
| 8446 | `sparkpipe_model_api` (user unit `glmfull-api6`) | the engine endpoint: `prompt_token_ids` in, token ids, usage and `cached_tokens` out |

Smoke receipts (2026-10-02, build de944ae): chat answered "Paris" with the
reasoning returned separately; a two-turn tool call (`get_weather`) returned
standard `tool_calls`, then answered from the tool result with 192 of 225
prompt tokens served from the prefix cache; LiteLLM `/v1/messages` returned a
`thinking` block and a `tool_use` block with `stop_reason: tool_use`, and
streamed `thinking_delta` events; `https://sparkpipe.ai/health/liveliness`
and `/v1/models` answered through the tunnel.

## The chat layer

`tools/serving/chat_frontend.py` is what vLLM is behind LiteLLM elsewhere, and
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
- it keeps no request state; prefix reuse happens in the engine;
- while the engine is starting (its 503 says `"status":"starting"`) or is
  unreachable, a request waits in the model's `warm_queue` (`depth`,
  `wait_seconds`, and `activation_url`, which is POSTed once per cold period
  or null when an operator starts the engine) until `/health` turns ready.
  A full queue or an expired wait answers 503 with Retry-After; `/health`
  shows the queue.

The layer is model-neutral. A model family supplies its configuration
(`model-families/glm52/serving/chat_frontend.json`): the model directory
(`chat_template.jinja`, `tokenizer.json`, `tokenizer_config.json`), the
engine URL, the served context, stop tokens, the reasoning and tool-call
markers, the accepted `chat_template_kwargs` and the mapping of OpenAI
`reasoning_effort` onto the template's own variable.

Known limits:

- The layer forwards `temperature`, `top_p`, `top_k` (vLLM's extension;
  `-1` and `0` mean off), `seed`, `logprobs` and `top_logprobs` (up to 20)
  to the engine. A request without `temperature` decodes greedily. Logprobs
  cover every generated token except the stop token, reasoning included, as
  vLLM reports them; a logprob too small for a float is `-9999.0`.
- `n`, `logit_bias`, `best_of`, `echo`, `suffix`, a nonzero
  `frequency_penalty`, `presence_penalty` or `min_p`, a `repetition_penalty`
  other than 1 and a `response_format` other than text answer 400.
- Prompts past `context_tokens` answer 400 `context_length_exceeded`; a
  client must compact before 65,536 tokens.
- A 60K-token prompt takes about 90 s to prefill the first time; later turns
  that share its prefix reuse it.

### Queueing and cancel (2026-10-18)

The door queues every request and rejects none for load (owner,
2026-10-18). A caller manages its own backlog and cancels a request by
closing its connection, whether the request is still queued or already
streaming:

- LiteLLM runs with `general_settings.cancel_on_disconnect: true`, so a
  client that goes away cancels LiteLLM's upstream call; mid-stream LiteLLM
  closes the upstream stream itself. `request_timeout` is 604,800 s (7
  days): it is the read timeout of LiteLLM's upstream call, and a request
  that waits in the queue sends no byte until its first token.
- The chat layer serves with `handler_cancellation`, so a closed client
  connection cancels its handler and closes the engine request; it logs
  `"finish_reason": "client_gone"`. Its engine session has no connection
  cap (aiohttp's default capped it at 100 requests in flight).
- `model_api` polls each waiting connection every 250 ms and cancels the
  engine request of a client that has gone (`api_await`).
- `model_api` and the chat layer raise their open-files soft limit to the
  hard limit at start and log it (`open files limit`). LiteLLM cannot, so
  its units start with `-p LimitNOFILE=524288`. When `model_api` still runs
  out of descriptors, `accept` waits 50 ms between retries and logs once.

The engine orders queued work by priority and then by deadline. The chat
layer sends priority 2 for interactive requests, 1 for a request to
`<model>-bulk` (every model has that second name at both LiteLLM doors
and in `/v1/models`) and 0 for the memory curator. Bulk work fills idle
capacity and is preempted, by checkpoint and requeue, when interactive
work needs its pages (PR #1500). The warm queue in front of a starting engine is
separate: it still answers 503 when full or after `wait_seconds`.

## Three models and on-demand swaps (2026-10-08)

The door serves `glm-5.3` (GLM-5.3 Full), `glm-5.3-flash` (GLM-5.3 Flash) and
`kimi-k3` (Kimi K3). One model holds the fleet at a time; a request for
another model waits while the fleet swaps.

| Piece | Where | What |
| --- | --- | --- |
| Swap controller | `tools/serving/model_swap.py`, config `deployment/hub/model_swap.json`, hub unit `sparkpipe-swap` on 127.0.0.1:8440 | `GET /status`, `POST /activate {"model": id}`, `POST /force`; every call needs the controller token (see Control-plane access) |
| Chat layer | `tools/serving/chat_frontend.py`, config `deployment/hub/chat_frontend.json`, hub unit `sparkpipe-chat` on 127.0.0.1:8433 | every model's `warm_queue.activation_url` is the controller; `GET /v1/sparkpipe/status` passes through to it; `/` redirects to the fleet page |
| LiteLLM | `config/litellm-config.yaml`, hub unit `sparkpipe-litellm` on 4000 | the three names, and `*` routes to `glm-5.3` |

A swap to model T, in order:

1. Wait up to `drain_seconds` for live requests on each model that will be stopped.
2. Stop every other model's API and engines.
3. Release their detached KV pools on every node: `weightd_warm SOCKET --drop-kv LABEL` (weightd `KV_POOL_DROP`, IPC kinds 59/60; an attached pool is refused and the swap fails).
4. Start T's engines and wait until every rank logs `model_residentd ready`. A dead rank fails the swap.
5. Warm T's weights where the model has a warm step.
6. Start T's API and wait until it connects every rank.

The KV of a stopped model stays on NVMe in its snapshot store. Weightd no longer keeps it in RAM, so the next model gets that memory. When the model comes back, its cached prefixes restore from the snapshot.

Measured on 16 Sparks (weightd 2ff18388, controller cycle, 4.2K-token document asked before and after):

| Swap | Time | Prefix cached on return |
| --- | --- | --- |
| cold start to K3 | 42 s (ranks 22 s, warm 17 s) | 4,224 of 4,227 |
| K3 to GLM Full | 23.5 s | 4,224 of 4,288 |
| GLM Full to GLM Flash | 10.9 s | 4,224 of 4,284 |
| GLM Flash to K3 | 42.3 s | 4,224 of 4,227, identical output |

Without step 3, weightd keeps each stopped model's KV pool resident (GLM Full 3.1 GB, Flash 0.4 GB). With K3 loaded, nodes then fell to earlyoom's 3% line, and earlyoom terminated a K3 rank.

The controller assumes one user steering the fleet. The last requested model wins, and two clients alternating between models will thrash it.

### Kimi K3 in the chat layer

K3 has no Jinja template. Its model directory holds the published files:

- `tiktoken.model` and `tokenizer_config.json`, the tokenizer (`"tokenizer": {"kind": "tiktoken", ...}` with the split pattern from `tokenization_kimi.py`);
- `encoding_k3.py`, the encoder (`"renderer": {"kind": "segments", "module": "encoding_k3.py", "function": "build_chat_segments"}`).

The renderer encodes each segment the encoder returns. Special tokens are allowed only in the structural segments, so user text never becomes a control token.

The reply is parsed as `think`, `response` and `tools` channels:

- `reasoning` ends at `<|close|>think<|sep|>`;
- the response and tools wrappers are `content_markers` and are dropped;
- calls use the `tagged` tool-call format, with typed `argument` elements or a `json` object;
- `<|close|>message<|sep|>` is an end marker.

OpenAI `reasoning_effort` maps onto K3's `thinking_effort`: low → low, medium → high, high → max. `minimal` turns thinking off.

### ZCode

ZCode reaches the door as a custom provider in `~/.zcode/v2/provider_config.json`:

- `"providerId": "custom:sparkpipe"`;
- `"api": {"type": "anthropic-messages", "baseUrl": "http://rtx5090:4000"}`;
- all 16 served model ids, with each context window in `modelConfigRules` taken from the chat layer's `context_tokens` (2026-10-16: `kimi-k3` 737280, `glm-5.3` and `qwen3.8-27b` 262144, `glm-5.3-flash` 131072, `deepseek-v4.1-flash` 16384, the rest 32768).

LiteLLM runs without a master key, so the provider's API key is a placeholder. Picking a model in ZCode swaps the fleet on its first request.

## The model_api contract (read first)

`model_api` (`node/model_api.c`) accepts OpenAI-shaped requests. Text
prompts need a tokenizer sidecar in the deployment config; token IDs always
work. `model_api` checks no API key and ignores the request's `model` field.

| Route | Request | Response |
| --- | --- | --- |
| `GET /health` | — | `200 {"status":"ok","served":N,"tokenizer":bool,"ranks":R,"connected_ranks":R,"missing_rank":-1,"degraded_rank":-1,"degraded_path":"none","engine_status":"OK","live_requests":L}`; `503` with `"status":"degraded"` while a rank is disconnected (`missing_rank` names the first one), a rank's driver runs on a degraded path (`degraded_rank`, `"degraded_path":"eager"` after a GLM Flash graph replay stuck or timed out), or the engine has latched a failure |
| `GET /v1/models` | — | `{"object":"list","data":[{"id":...}]}`; the id is `SPARK_MODEL_ID`, default `sparkpipe-model` |
| `POST /v1/completions` | `prompt` (text) or `prompt_token_ids` | `text_completion` with `choices[0].text`, `finish_reason`, `usage`, `tokens` |
| `POST /v1/chat/completions` | `messages`, or `prompt` or `prompt_token_ids` | `chat.completion` with `choices[0].message`, `finish_reason`, `usage`, `tokens` |
| anything else | — | `404 {"error":"not found"}` |

Reply text is always valid UTF-8. When the emitted tokens leave a byte
sequence incomplete or invalid (a reply cut inside a character, a
byte-fallback token), each maximal invalid subpart comes out as U+FFFD, as
in a lossy UTF-8 decode; `tokens` keeps the exact ids.

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
| `top_p` | above 0 and at most 1, default 1; samples from the smallest set of top tokens whose probability at the request's temperature reaches `top_p` |
| `top_k` | unsigned integer up to 1048576, default 0 (off); samples from the `top_k` most likely tokens, applied before `top_p` |
| `logprobs` | unsigned integer up to 20: return each emitted token's log-probability and that many top alternatives. The response adds `token_logprobs`, one entry per token in `tokens`: an array of `[token_id, logprob]` pairs, the emitted token first and the alternatives in descending order. Logprobs are the model's log_softmax at temperature 1 (before truncation); `null` stands for a value too small for a float |
| `speculation` | boolean, default `true`. `false` makes the engine draft nothing for this request, neither lookup nor model drafts, so every token is one plain decode step. Chained decode steps are unchanged, so this measures decode without speculation. A request that asks for `logprobs` never drafts either, but it also decodes one step per submission |
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
`top_p`, `top_k`, `logprobs`, `speculation` or `chat_template_kwargs`, or
`chat_template_kwargs` on a `prompt` or `prompt_token_ids` request, is a
`400 invalid_option`, never a silent default. With `temperature` 0,
`top_p` and `top_k` cannot change the greedy token and are not forwarded.
A nonzero temperature on a deployment whose adapter cannot sample is a
`400 sampling_unsupported`; `top_p` below 1 or a `top_k` on an adapter
that samples by temperature only is a `400 truncation_unsupported`;
`logprobs` on an adapter that cannot return them is a `400
logprobs_unsupported`. GLM-5.3 Full (glm52), GLM Flash (glm5_next) and K3
serve all three. A prompt plus `max_tokens` that the
deployment's context or KV pages cannot hold is a `400
context_length_exceeded`; the batch engine's own admission check decides, so
the API never duplicates the limits. Without a tokenizer sidecar the
response carries `tokens` only; with one it carries both text and `tokens`.

A request that reaches its `deadline_ms` while a rank is disconnected
answers `504 deadline_exceeded` with `missing_rank`. When the engine has
latched a failure other than a disconnect (a rank or adapter protocol
failure), every request answers `503 engine_failed` naming the status until
the ranks restart or reset; the API no longer reopens a failed engine for
each request.

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

## sparkpipe.ai

`https://sparkpipe.ai` is static, with no API and no login. It holds the home page, the Fleet List, the terms, the privacy policy and the checkout preview. They are copied from `site/` to `/var/www/html` on the KCInfra VPS (27.102.120.99). `tools/site_terms.py` generates the terms pages from `site/terms/`; `tests/test_site.py` fails when they differ.

Apache serves only these pages: no API, no login, no proxy. Fleet-hour customers will reach their fleet through the customer API that TECHDEBT lists under fleet-hours.

Every page with a Korean twin (`index`, `fleets`, `terms`, `privacy`, `hourly-terms`, `agreement`, each with a `-ko.html` page) carries the EN / 한국어 switch and `hreflang` alternates. `deployment/vps/sparkpipe-lang.conf`, included in the port-443 virtual host after the canonical-host rule, chooses the language:

- `?lang=ko` or `?lang=en` on any page sets the cookie `lang` (one year, Secure, HttpOnly, SameSite=Lax) and redirects to the same page without the query. The switch links carry it.
- A request that does not come from one of our own pages (no `https://sparkpipe.ai/` referrer) for an English page with a Korean twin goes to the Korean twin when the cookie says `ko`, or when there is no `lang` cookie and the client's /24 is in the Korean map. Links between our pages are followed as written, so an English link on a Korean page stays English. This needs `Referrer-Policy: same-origin` on the virtual host: other sites still get no referrer.
- The map `/etc/apache2/sparkpipe/kr24` (SDBM) holds every /24 that APNIC lists as allocated or assigned to KR, keyed `a.b.c`. Build it from APNIC's daily file and install it on the VPS:

```sh
curl -sSfO https://ftp.apnic.net/stats/apnic/delegated-apnic-latest
curl -sSfO https://ftp.apnic.net/stats/apnic/delegated-apnic-latest.md5
python3 tools/site_kr_networks.py delegated-apnic-latest kr24.txt
sudo httxt2dbm -f SDBM -i kr24.txt -o /etc/apache2/sparkpipe/kr24.new
sudo mv /etc/apache2/sparkpipe/kr24.new.pag /etc/apache2/sparkpipe/kr24.pag
sudo mv /etc/apache2/sparkpipe/kr24.new.dir /etc/apache2/sparkpipe/kr24.dir
```

Check the md5 against the downloaded `.md5` first. The builder refuses a file with fewer than 100,000 Korean networks; APNIC lists about 440,000.

| VPS path | Goes to |
| --- | --- |
| `/` and the pages | `/var/www/html` |
| `:8443/` | VPS 127.0.0.1:18082, then hub 8090 (Open WebUI, next section) |
| `:8443/fleet`, `/fleet/status`, `/fleet/switch-now` | VPS 127.0.0.1:18081, then hub 8433 (the chat layer's admin page) |

The tunnel is the hub system unit `sparkpipe-ai-door` (root, key `/root/.ssh/sparkpipe_ai_tunnel_ed25519`). It also forwards 127.0.0.1:18080 to the hub's port 4001, where nothing listens, and the VPS key may listen on 18080. Dropping 18080 from both is an SSH change, which the owner decides.

## SparkPipe Chat (Open WebUI, sparkpipe.ai:8443)

`https://sparkpipe.ai:8443` is the chat app for invited people: Open WebUI
v0.12.0, pinned by digest, with its own accounts. Owner decisions
(2026-10-14): no sub-domain, so it gets its own port on the main domain;
Open WebUI's branding stays visible; anyone may sign up but an account stays
pending until the admin approves it; web search will use a self-hosted
SearXNG (not enabled yet).

| Piece | Where | What |
| --- | --- | --- |
| Open WebUI | hub Docker container `sparkpipe-chatui`, `127.0.0.1:8090` -> 8080, data `/srv/workspace/open-webui/data` | settings `deployment/hub/open_webui.env`, secrets `/srv/workspace/open-webui/secrets.env` (root, 600: `WEBUI_SECRET_KEY`, `OPENAI_API_KEYS`) |
| Chat LiteLLM | hub user unit `sparkpipe-litellm-chatui` on `172.17.0.1:4002` (Docker bridge only), `~/sparkpipe-chat/litellm-chatui.yaml` (copy of `deployment/hub/litellm-chatui.yaml`), key in `~/sparkpipe-chat/chatui.env` | serves every fleet model; K3 is the default (owner 2026-10-14) |
| Tunnel | hub `sparkpipe-ai-door`, `-R 127.0.0.1:18082:127.0.0.1:8090` | the VPS key may listen on 18082 |
| Apache | VPS `/etc/apache2/sites-available/sparkpipe-chat-8443.conf` | TLS with the sparkpipe.ai certificate, websocket upgrade, no Basic auth; includes `/etc/apache2/sparkpipe/chat-local.conf` (copy of `deployment/vps/sparkpipe-chat-local.conf`) before its proxy lines |

**Controls and languages.** `deployment/hub/open_webui_loader.js` is mounted
as Open WebUI's `static/loader.js` (`open_webui_run.sh`; the hub copy is
`/srv/workspace/open-webui/branding/loader.js`). It adds an EN / 한국어 switch,
which sets Open WebUI's own locale in the browser, and above the message box
the memory switch, the compaction slider with Compact and Undo, a Context
button and the Local mode link. The buttons send the chat layer's commands
(`/memory on|off`, `/compact N%|full|undo`, `/context`), which the chat layer
answers itself and leaves out of every later prompt. A user who has not
chosen a language gets the browser's (`ui.default_locale` is empty).

**Local mode.** `https://sparkpipe.ai:8443/local/` (`deployment/vps/chat-local/`,
served by the VPS from `/var/www/chat-local`) keeps chats in the browser's
IndexedDB and sends each request to Open WebUI's `/api/chat/completions` with
the user's Open WebUI session. The request carries no chat id, so Open WebUI
saves no chat and the chat layer opens no memory; compaction runs in the
page. `tests/test_chat_controls.py` checks that the page calls nothing else.

Requests run browser -> Apache 8443 -> tunnel -> Open WebUI -> chat LiteLLM ->
chat layer -> engine. The chat layer gives requests without sampling fields
the model's published defaults and caps the reply budget at the model's
output limit, so Open WebUI sends neither. Attachments go into the prompt
whole (`BYPASS_EMBEDDING_AND_RETRIEVAL`), which uses the long context windows
and the prefix cache instead of retrieval chunks. Users cannot change
sampling, see other models, share chats publicly, create API keys or add
tools.

SparkPipe Chat provides generative AI directly to its users, so it carries
the notice and labelling duties of Article 31 of Korea's AI Basic Act. The
`ai-notice` banner (`ui.banners`, `WEBUI_BANNERS`) tells users on every new
chat, before their first message, that the replies come from generative AI;
Open WebUI shows it in Korean when the interface language is Korean. It is
not dismissible, so closing it hides it only until the next visit. Every
reply copied with the copy button ends with the bilingual label in
`ui.watermark` (`RESPONSE_WATERMARK`). A new account's waiting screen
(`ui.pending_user_overlay_title` and `_content`, `PENDING_USER_OVERLAY_*`)
says, in English and Korean, that the replies come from generative AI and
links the privacy policy, before the account is approved or used. Customers who serve their own users
with fleet-hours carry these duties themselves (agreement 8.10, Hourly Terms
7.2).

Start or replace the container with `sudo sh /srv/workspace/open-webui/run.sh`,
a copy of `deployment/hub/open_webui_run.sh`. It pins v0.12.0 by digest,
mounts the data directory and mounts the SparkPipe icons from
`/srv/workspace/open-webui/branding` (a copy of
`deployment/hub/open_webui_branding`) over Open WebUI's own.

Visitors' IP addresses stay out of the hub. Open WebUI trusts forwarded
addresses only from `127.0.0.1` (`FORWARDED_ALLOW_IPS`), and its requests
arrive from the Docker bridge, so its log shows the bridge address. The
Apache logs on the VPS are the only access logs with visitor addresses, kept
92 days as the privacy policy states. The container log is capped at three
10 MB files.

Retention follows the privacy policy:

- **Chat memory goes with its chat.**
  - A root timer, `open-webui-chat-list.timer`, runs every 5 minutes. It reads `webui.db` read-only and writes Open WebUI's chats, as hashed (user, chat) names, to `/srv/workspace/sparkpipe/open-webui-chats.json`. The chat layer's user cannot read the database itself.
  - Every 10 minutes the chat layer deletes the memory file of each chat not on that list and drops that chat from RAM. Files written in the last hour are kept, for chats whose row is not stored yet.
  - It deletes nothing when the list is missing, older than 30 minutes, or names no users.
  - Install:
    1. Copy `tools/serving/open_webui_chat_list.py` to `/usr/local/lib/sparkpipe/` (root, 644).
    2. Copy `deployment/hub/open-webui-chat-list.service` and `.timer` to `/etc/systemd/system/`.
    3. Run `systemctl enable --now open-webui-chat-list.timer`.
- **Operational records:** `chat.log`, the LiteLLM logs and the model APIs' logs rotate monthly under `/etc/logrotate.d/sparkpipe-hub` (`deployment/hub/logrotate-sparkpipe-hub`). A rotated file is deleted 334 days after its last write, so no record is older than a year.

`open_webui.env` is copied from `deployment/hub/open_webui.env`. v0.12.0
writes every persistent setting into the `config` table (`key`, JSON `value`)
at its first start and reads it from there afterwards, so an env change only
reaches the settings it reads at every start (`env.py`, for example
`BYPASS_MODEL_ACCESS_CONTROL`). Change the others in the admin panel, or with
the container stopped and the data directory backed up, update the row in
`webui.db` and recreate the container; keep `open_webui.env` matching so a
fresh data directory starts the same. That is how `ui.default_models`
(`kimi-k3`), `ui.enable_signup` and `openai.api_keys` were changed on
2026-10-14.

The first account created becomes the admin: it was created through
`ssh -N -L 8090:127.0.0.1:8090 rtx5090` and http://localhost:8090 before port
8443 was opened. Creating it turns sign-up off by itself (`ui.enable_signup`
false); it was turned back on because the owner chose open sign-up with
approval. New accounts arrive with the `pending` role and see only a waiting
screen; the admin approves one in Admin Panel -> Users by changing its role to
`user`, or deletes it.

Users see only models that have an access grant, and models from a connection
have none until someone configures them in the admin panel, so a `user` would
see an empty model list. `BYPASS_MODEL_ACCESS_CONTROL=true` shows every model
of the chat LiteLLM to every approved user, which is the intent while it
serves one model.

The chat LiteLLM key is shared by `chatui.env` (`LITELLM_MASTER_KEY`),
`secrets.env` (`OPENAI_API_KEYS`) and the `openai.api_keys` row. It was
replaced on 2026-10-14 after a config dump printed it; the old one is refused.

K3 takes about 70 s to swap in from GLM Full (69.9 s on 2026-10-14). The LAN
and sparkpipe.ai doors still serve every model, so a request for another
model there swaps the fleet away, and the next chat waits for K3 again.

### All models, Qwen for the admin, and model switches (2026-10-14)

The chat LiteLLM serves all 16 fleet models (`deployment/hub/litellm-chatui.yaml`,
generated from `deployment/hub/open_webui_models.json`, contexts checked
against the chat layer). Each has an Open WebUI model entry from that file,
applied inside the container with `tools/serving/open_webui/open_webui_models.py`: name,
description (context window and switch time), text-only input, web search
and the code interpreter on by default, the GitHub tool, the trimmed
built-in tools. Open WebUI's own context compaction stays off: SparkPipe chat
memory serves every model (docs/CHAT_MEMORY.md), and the chat layer refuses a
memory session once a compaction summary appears. The global "Full context"
filter (`tools/serving/open_webui/functions/memory_off.py`, installed with
`open_webui_functions.py` from `deployment/hub/open_webui_functions.json`) is
the user's switch to turn memory off for a chat.

The 15 public models carry a public read grant. Qwen 3.8 Flash carries none,
so only the admin sees it (admins bypass grants): its license requires a
separate agreement with Qwen for any model-as-a-service business, and the
owner's own use is internal. Qwen 3.8 27B (Apache 2.0) and Qwen 3.8 Max
(whose license asks for a separate agreement only above US$50 million of
yearly revenue) are public.
`BYPASS_MODEL_ACCESS_CONTROL` is off again; a model with no entry or no grant
stays hidden from users.

Only one model is resident on the fleet. Picking another model makes the
chat layer queue the request and ask the swap controller to switch. The
controller waits for the current model's replies to finish: it stops the old
engine only after it has had no live request for `drain_quiet_seconds` (15),
up to `drain_seconds` (600, the owner did not want a stuck engine to hold a
switch for long). The quiet window covers the short gaps between the
requests of one tool-calling reply (search, fetch, answer), which a plain
"no live request" check would cut (`tests/test_model_swap_drain_quiet.py`).
Queued requests wait up to `warm_queue.wait_seconds` (3900 s). New requests
for the old model that arrive during the drain are still served and extend
it; the fair queue (oldest waiter first, minimum residency) is the next step.

**Switch now.** `https://sparkpipe.ai:8443/fleet` (`site/fleet.html`, served
inside the Open WebUI container on the chat origin, so its `/` and
`/static/favicon.*` links are Open WebUI's own paths; `tests/test_site.py`
lists exactly those as the page's chat-app links) shows what is loaded, a switch in progress with its drain
(live replies, time waited, quiet time) and recent switches, refreshed every
2 s. For the admin it adds a red **Switch now** button, which ends the drain
at once and cuts any reply still running on the old model, and **Load now**
for any model. The page reads the browser's SparkPipe Chat login (same
origin); the chat layer's `POST /v1/sparkpipe/switch-now` asks Open WebUI
(`control.admin_url`, `/api/v1/auths/`) who the login belongs to and forwards
to the controller's `POST /force` only for the admin
(`tests/test_chat_frontend_switch_now.py`, `tests/test_model_swap_force.py`).
Forced switches are marked `forced` in the controller's history. The VPS
8443 site proxies only `/fleet`, `/fleet/status` and `/fleet/switch-now` to
the chat layer.

### Fleet access for the admin (2026-10-14)

Owner decisions: in the admin's own chats the model may read the fleet and
run commands; regular users get none of it. First version: every command
approved. Owner, later the same day: approving every command made it
useless, so the admin grants access once per chat and commands then run
without asking, bounded by what their identity can do.

- **Gateway.** `tools/serving/fleet_gateway.py` runs on the hub as `spec`
  (user unit `sparkpipe-fleet-gateway`, `172.17.0.1:8441`; config
  `deployment/hub/fleet_gateway.json` in `~/sparkpipe-fleet/`). It holds the
  SSH access, so Open WebUI never does. Every call needs the token in
  `/srv/workspace/sparkpipe/fleet-gateway/token` (spec, 600). The token and
  the network are the real boundary: the approver and grant fields are audit
  records, and any holder of the token can use every endpoint. It serves the
  swap controller's state, each machine's uptime, load, memory, disk, GPU and
  busiest processes, the configured logs, per-chat grants and `POST /v1/run`.
- **Sandboxed commands (the default).** Inside a chat whose grant is live
  (`POST /v1/grant` with a saved chat's id and the approver; 8 hours; at most
  64 live grants; `DELETE /v1/grant` revokes; a gateway restart ends them),
  each command runs as a transient systemd service through `sudo -n
  systemd-run` on the hub, or through ssh and the same on a spark: a dynamic
  user with the machine owner's group, `ProtectSystem=strict`,
  `ProtectHome=read-only`, a private `/tmp`, no devices (no GPU, no RDMA
  verbs), `NoNewPrivileges` (no sudo), only `AF_UNIX` and `AF_NETLINK`
  sockets plus `IPAddressDeny=any` (no TCP or UDP, so none of the fleet's
  services: the swap controller, the chat layer, LiteLLM, the engine APIs),
  the owner's `.ssh`, `.env`, `.config`, the known secret files and both
  token directories hidden, `CPUQuota=200%`, `MemoryMax=4G`, `TasksMax=128` and
  `RuntimeMaxSec` of the timeout plus 5 s. It can read the machine and the
  SparkPipe trees, logs and configs, list processes and read interface and
  RDMA counters; it cannot change anything. Checked live on the hub and
  spark3 on 2026-10-14. Nothing is created on the hosts; there is no account
  to set up.
- **Elevated commands.** With `elevated` the command runs as `spec` (on a
  spark under `timeout -k 5`) after the admin approves that exact command.
  This is the path for anything that changes state or needs the network or
  the GPU.
- **Both paths** pass the refusal rules in the config first (a global
  `sync`, killing flychess, power-off, deleting a root, home, workspace or
  mount directory, raw disk writes, secret files; on sparks also anything
  that adds to the boot path). These are a best-effort backstop against
  mistakes, not a security boundary. Runs are written before and after to
  `/srv/workspace/sparkpipe/logs/fleet-gateway-audit.jsonl` (600) with the
  chat. Output is masked for secrets and cut at 20,000 characters. A timeout
  kills the command's whole process group, and a drain of the output is
  bounded. `tests/test_fleet_gateway.py`.
- **Tool.** `tools/serving/open_webui/tools/fleet.py`, the Open WebUI tool
  `fleet` with no access grant, so only the admin (its owner) can use it; it
  also checks the caller's role. `fleet_status`, `fleet_nodes` and
  `fleet_logs` run directly; `fleet_run` asks once per chat ("Give Kimi
  fleet access for this chat?", showing the first command), then runs in the
  sandbox without asking; with `elevated` it asks for each command.
  `tests/test_open_webui_fleet_tool.py`. The tool is attached to no model;
  the admin switches it on per chat from the tools menu.
- **Review.** An adversarial review (2026-10-14) found that a permanent
  restricted account would still reach the fleet's unauthenticated control
  services, GPUs and RDMA devices, which led to the sandbox. The exposures it
  listed outside this tool are in Control-plane access below.
- **Not done.** Two-factor login stays off (owner, 2026-10-14).

### Control-plane access (2026-10-14)

The adversarial review of the fleet tool listed exposures that LAN users and
local accounts could still reach. Verified on the hub before the fix:

| Exposure | State |
| --- | --- |
| The swap controller (127.0.0.1:8440) switched or stopped the fleet for any local caller | fixed: every route needs the controller token |
| The chat layer listened on 0.0.0.0:8433, and its `POST /v1/sparkpipe/activate` started a switch for anyone | fixed: the route is gone and the layer serves loopback only |
| LiteLLM 4000 (`config/litellm-config.yaml`) serves the LAN and the tailnet without a master key | open, owner decision: a key breaks ZCode until ZCode carries it |
| Six root `sparkpipe_model_api` test processes from 2026-09-27 (`--runtime-root /tmp/chatfix`) and one spec test instance from 2026-09-28 answer on 0.0.0.0 | open, owner step: stop them |
| `glm-5.3-fp8` (145 files) and `glm-5.3-bf16` (11 files) on `/mnt/model-warm` are 0666, and five directories there are 2777 | open, owner step: remove the world write bit |
| `/etc/exports.d/qpn.exports` exports `/srv/qpn` rw with `no_root_squash` to 10.10.0.0/16 and 192.168.50.0/24; no client mounts it and it holds two empty probe files | open, owner step: drop the export or squash root |

**The controller token.** `model_swap.py` refuses to start without
`--token-file`, or when the file can be read by its group or others or holds
fewer than 32 characters, and answers 401 to any call without
`Authorization: Bearer <token>`, `/status` included
(`tests/test_model_swap_token.py`). The token lives in
`/srv/workspace/sparkpipe/swap-controller/token` (spec, 600, in a 2700
directory). Its callers:

- the chat layer reads the file named by `controller_token_file` in its
  config and presents it on warm-queue activations, status and Switch now,
  and nowhere else: not to the engines and not to SparkPipe Chat's login
  check (`tests/test_chat_frontend_warm_queue.py`,
  `tests/test_chat_frontend_switch_now.py`). It refuses a config whose
  `control` block or activation URLs have no token file;
- the fleet gateway reads `controller_token_file` from its config for the
  status and engine-log routes; a refusal from the controller is a 503, never
  an empty status (`tests/test_fleet_gateway.py`). The sandbox cannot see
  either token directory;
- the release scripts call `swap` in `scripts/rollover_lib.sh`, which reads
  the token on the hub and gives curl the header on stdin, so the token never
  appears in a process list (`tests/test_swap_token_shell.py`);
  `tools/ds4_eval92_session.py` and `tools/chat_memory_phase1.py` read the
  file the chat layer's config names.

A request for a model that is not loaded still swaps the fleet through the
warm queue: whoever can reach the chat layer can switch models. Only the hub
reaches it now: `chat_frontend.py` refuses a `--host` that is not a loopback
address. Its clients are LiteLLM 4000 and 4002 and the sparkpipe.ai tunnel
(`-R 127.0.0.1:18081:127.0.0.1:8433`, for `/fleet`), all on the hub. The
admin's Load now on `/fleet` loads a model at once.

Start or restart the two units (hub, as spec):

```sh
install -d -m 2700 /srv/workspace/sparkpipe/swap-controller
( umask 077; [ -s /srv/workspace/sparkpipe/swap-controller/token ] || \
  openssl rand -hex 32 > /srv/workspace/sparkpipe/swap-controller/token )
systemd-run --user --unit=sparkpipe-swap -p WorkingDirectory=/home/spec/sparkpipe-swap \
  /usr/bin/bash -c 'exec python3 model_swap.py --config model_swap.json --token-file /srv/workspace/sparkpipe/swap-controller/token --host 127.0.0.1 --port 8440 >> swap.log 2>&1'
systemd-run --user --unit=sparkpipe-chat -p WorkingDirectory=/home/spec/sparkpipe-chat \
  /usr/bin/bash -c 'exec python3 chat_frontend.py --config config.json --host 127.0.0.1 --port 8433 >> chat.log 2>&1'
```

Read the controller from a hub shell without putting the token on a command
line:

```sh
printf 'Authorization: Bearer %s\n' "$(cat /srv/workspace/sparkpipe/swap-controller/token)" | curl -s -H @- localhost:8440/status
```

### Chat memory (K3, 2026-10-14)

SparkPipe Chat's K3 conversations use the model-curated ledger described in
`docs/CHAT_MEMORY.md` (stages A and B). The chat layer loads `chat_memory.py` because
its config has a `chat_memory` block (store `/srv/workspace/sparkpipe/chat-memory`,
one private directory per hashed user id; JWT secret
`~/sparkpipe-chat/memory-jwt.secret`, 600). Open WebUI sends the signed
`X-OpenWebUI-User-Jwt` (`ENABLE_FORWARD_USER_INFO_HEADERS=true`, the same secret
as `FORWARD_USER_INFO_HEADER_JWT_SECRET` in `secrets.env`) and the chat and
message id headers from the connection's header templates (`openai.api_configs`
row, mirrored as `OPENAI_API_CONFIGS` in `open_webui.env`); its memory feature
no longer writes into the system message (`memories.system_context.enable`
false). The chat LiteLLM forwards `x-` headers
(`forward_client_headers_to_llm_api: true`) and drops the client's
Authorization header. Requests without a valid token (ZCode, API clients, other
doors) are served raw. Rotating the secret means updating both files and
recreating the container and the chat layer. Deployed 2026-10-14 with backups
stamped 20261014T232328Z (Open WebUI database and secrets) and the chat-layer
backups of the same night under `/srv/workspace/sparkpipe/backups`.

Stage B (curator continuation, pre-warm, regroups, fit) went live on
2026-10-15 at 00:15Z by copying `chat_frontend.py` and `chat_memory.py` into
`~/sparkpipe-chat` and restarting `sparkpipe-chat` while K3 was idle; the
previous files are `/srv/workspace/sparkpipe/backups/*.prev-20261015T001524Z`.
The K3 model API runs the stop-token fix (`engine/stop-token-completes`, built
from 1d89e0377 with only that patch; previous binary
`~/k3-kv-api/bin/sparkpipe_model_api.prev-20261015T000602Z`) until the next
fleet release carries it.

### Web search (SearXNG)

Since 2026-10-14 K3 has Open WebUI's `search_web` and `fetch_url` tools. The
search engine is a self-hosted SearXNG (owner decision 2026-10-14): hub
Docker container `sparkpipe-searxng` (2026.10.13-22c64b8d3, pinned by digest)
on `172.17.0.1:8888` only, settings `deployment/hub/searxng_settings.yml`
copied to `/srv/workspace/searxng/settings.yml` (JSON output on, limiter off
because only Open WebUI calls it, moderate safe search), `SEARXNG_SECRET` in
`/srv/workspace/searxng/secrets.env` (root, 600). It queries the upstream
engines from the hub's address; on the first day Brave and DuckDuckGo
rate-limited it and Google and the others answered.

Open WebUI offers the two tools only when web search is on globally, the user
has the `web_search` permission, the model has the `web_search` capability
and the chat's Web Search toggle is on. The `kimi-k3` model entry (Admin
Panel -> Settings -> Models) turns the toggle on by default
(`defaultFeatureIds`), so K3 itself decides when to search or read a page;
the entry also marks K3 as text-only, so image uploads are not offered.
`fetch_url` refuses private and loopback addresses
(`ENABLE_RAG_LOCAL_WEB_FETCH=false`, checked against the swap controller,
the hub, the LAN and 10/8), so a user cannot make the hub read internal
services, and it cuts a page at 40,000 characters.

```sh
sudo docker run -d --name sparkpipe-searxng --restart unless-stopped \
  -p 172.17.0.1:8888:8080 \
  -v /srv/workspace/searxng/settings.yml:/etc/searxng/settings.yml:ro \
  -v /srv/workspace/searxng/cache:/var/cache/searxng \
  --env-file /srv/workspace/searxng/secrets.env \
  searxng/searxng@sha256:9b61996edcc679fb6665290f95fda15389a14eda6060e41a25c63ab2758f2436
```

### Other tools (2026-10-14)

The owner asked for these four after K3 told a user it could not browse:

- **Code interpreter.** `execute_code` runs Python with Pyodide in the
  user's browser (numpy, pandas and matplotlib ship in the image), so no code
  runs on the hub. `code_interpreter.enable`, the `code_interpreter` user
  permission and the K3 capability are on, and K3's entry turns the toggle on
  by default.
- **Scanned documents.** Attachments go through Docling instead of plain text
  extraction, so scanned PDFs are read with OCR and tables come out as
  Markdown tables. Hub Docker container `sparkpipe-docling` on
  `172.17.0.1:5001` only (`docling-serve-cpu`, pinned by digest, 7.5 GB image,
  limited to 8 CPUs and 16 GB), `rag.content_extraction_engine` `docling`.
  A generated one-page scan with a table converted in 6 s. Plain text files
  still bypass Docling.

  ```sh
  sudo docker run -d --name sparkpipe-docling --restart unless-stopped \
    -p 172.17.0.1:5001:5001 --cpus 8 --memory 16g \
    -e DOCLING_SERVE_ENABLE_UI=false -e DOCLING_SERVE_MAX_SYNC_WAIT=600 \
    ghcr.io/docling-project/docling-serve-cpu@sha256:5a8367f3f657c26755605c3b2255546674833209a18afc2011598f4c045c6db3
  ```
- **GitHub.** `tools/serving/open_webui/tools/github.py`, installed as the
  Open WebUI tool `github` with public read access and attached to K3
  (`toolIds`): `github_search` (repositories, code, issues), `github_read`
  (files and directories, the README with the top level), `github_list`
  (issues, pulls, commits, releases) and `github_item` (one issue or pull
  request with its comments and changed files). It only sends GET requests to
  the configured API and refuses malformed repositories and paths before any
  request (`tests/test_open_webui_github_tool.py`). Without a token it uses
  GitHub's unauthenticated limit of 60 requests an hour for the hub's address
  and cannot search code. To lift that, the owner creates a fine-grained
  token with read-only access to public repositories and enters it in
  Workspace -> Tools -> GitHub -> Valves; never use the SparkPipe PAT there.
  To change the tool, edit the file, run the test and paste it into the same
  editor.
- **Fewer built-in tools.** K3's entry turns off the built-in tool groups
  nobody here uses: notes, calendar, chat search, skills, tasks, automations,
  ask-user and knowledge bases (`meta.builtinTools`). They took about 4,300
  of the 5,600 tokens every new chat sent before its first message; time,
  memory, web search and the code interpreter stay.

Install or update the K3 model entry and the GitHub tool with Open WebUI's
own code inside the container (`Models.update_model_by_id`,
`Tools.insert_new_tool` after `load_tool_module_by_id`), or through the admin
panel. To upgrade, pull the new release, check its
release notes and security advisories, replace the digest here, recreate the
container with the same data directory, and run a chat, an attachment and a
login. Back up `/srv/workspace/open-webui/data` before an upgrade.

Open WebUI's license (BSD-3 with a branding clause, from v0.6.6; clause 4
of the v0.12.0 LICENSE) forbids altering or replacing its branding unless the
deployment has at most 50 end users in any rolling 30-day period, or holds
written permission or an enterprise license. The owner expects about six
users (2026-10-14), so the icons, splash, logo and web manifest are the
SparkPipe bolt from `site/favicon.svg`, rendered with `rsvg-convert`. Keep the
user count (Admin Panel -> Users) at 50 or below; if it could pass that,
drop the branding mounts from `run.sh` first. The name is set to "SparkPipe
Chat", which v0.12.0 shows as "SparkPipe Chat (Open WebUI)" because it adds
the suffix in code. The repository holds only configuration and the icons, no
Open WebUI code.

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
