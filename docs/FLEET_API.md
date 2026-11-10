# Fleet API

This is the specification of the API and CLI a customer uses on a fleet it holds, as it ships on 4 January 2027. Every feature removes waiting or repeated work from an engineer's day. Section 1 says, for each saving, where the engine starts today and the work that delivers it. Section 18 assigns that work and lists the acceptance tests that close it.

The product rules (who holds a fleet, bookings, working-state deletion) are in the Customer Agreement and `docs/FLEET_BOOKINGS.md`. The engine side of working state is `docs/WORKING_STATE.md` (PR #1691, with the hub in PR #1692). This file is the customer-facing contract built on top of them.

## 1. Where the time goes, and what removes it

An engineer working with a coding agent waits for three things: prefill of context the model has seen before, generation, and work they could have left running overnight. Per-token APIs make the first one recur, because their caches expire after 5 minutes by default. A held fleet removes all three. The table lists the engine work that makes each saving hold.

| Saving | Mechanism | Starting point on `main` | Work | Acceptance test |
| --- | --- | --- | --- | --- |
| No re-prefill after a break | Automatic prefix cache, kept while the account holds the fleet | Prefix cache exists: hash-chained 64-token blocks, LRU (`cache/prefix_cache.c`). Retention is 24 h from last use (`spark_kv_snapshot.h:27`) | Held-hours retention and the wipe at a change of user (#1691, #1692). Snapshot store proven at the full per-node budget; today only 8 GiB per rank is proven (`TECHDEBT.md:579-599`). Prefix index that survives a release and a restart (`FLEET_RELEASE_RUNBOOK.md:298` drops it today) | Gate A |
| Every turn resumes where the last one ended | Assistant turns replayed token-exact, so the next request matches up to the end of the previous reply | The chat layer keeps reply text byte-exact for chat (`LITELLM_FRONTEND.md:33-37`). Kimi K3 resumes only at 16,384-token checkpoints or at the end of a reply (§5.1) | The same guarantee for Anthropic Messages and Responses, including tool calls and reasoning (§3.2) | Gate A |
| Long agent sessions never fail | Bounded recurrent-state store | A K3 conversation fails after about 977 checkpointed requests (`PERFORMANCE_STATUS.md:450-452`) | Evict the oldest checkpoints instead of failing | Gate A |
| No accidental cold fleet | Several models resident at once when they fit; a model that does not fit replaces another only on purpose | One model at a time, and "the last requested model wins" (`LITELLM_FRONTEND.md:132`). A switch takes 16–46 s, at worst 79 s (`PERFORMANCE_STATUS.md:174-202`). The cache is saved and restored: 137,600 tokens back in 2.7 s | Resident sets with a per-node memory budget for weights and cache, per-model cache pools, the alias table, and explicit replacement (§4.5). A background "haiku" call is served by a small resident model or the main one, and never swaps the fleet | Gate A |
| Big prompts paid for once | Long prefill publishes its cache | The fast streamed 1M prefill (16.2 min) publishes no cache blocks and stops decode meanwhile (`TECHDEBT.md:308-326`). A reusable 1M prefill takes about 76 min (`PERFORMANCE_STATUS.md:68`) | Streamed prefill that publishes its blocks to the prefix cache and snapshot store | Gate C |
| The codebase is already in context | Context packs (§5): a repository snapshot placed first in every prompt, kept warm | None | Pack builder, versioned aliases, and refresh in idle time | Gate D |
| A session starts warm | Prefill-only warm-up of an exact future prefix, and pins that protect it | `max_tokens` must be 1 or more (`node/model_api.c:1622-1631`); `max_tokens` 1 commits a prompt. No pinning | `POST /v1/cache/warm` and pins (§4) | Gate B |
| Faster generation | Speculation | Lookup drafts from the request's own text, plus a model drafter for K3 and DeepSeek V4.1 Flash. 126.4 tok/s on K3 whole-file rewrites, against 15.3 plain. No speculation at batch 2 or more (`PERFORMANCE_STATUS.md:50-51`) | Per-fleet drafter choice (§8), and speculation at batch 2 and more | tok/s per request in telemetry |
| Parallel agents | Batching | 8 sequences for K3, and 4 requests at once per key (`LITELLM_FRONTEND.md:401`) | Holder keys exempt from the per-key cap (#1692). Two sequences kept for interactive use while Functions run | Aggregate tok/s in telemetry |
| A lost node pair costs hours, not days | The hub reports which part of the cache a pair failure destroyed. The market credits the rebuild hours (Agreement 6.6), and a Function rebuilds the cache | Pair mirror on branch (`docs/WORKING_STATE.md`, #1691/#1692); no loss report | Loss report from the hub, automatic credit, and the rebuild Function (§7.2) | Gate A |
| Work done while you are away | Functions (§7) on the held fleet, with a sandbox for code that runs | None. No sandbox. About 11 GB of free memory per node with K3 loaded, and a 16-core CPU tenant cut K3 from 16.2 to 10.0 tok/s (`PERFORMANCE_STATUS.md:197-199`) | Function runner, plus a sandbox host per fleet so that code never runs on the Sparks | Gate E |

Telemetry (§3.6) puts the numbers on every response, so the saving is measured per request rather than claimed.

## 2. Surface

All paths are under `https://sparkpipe.ai/pay/api`. Every request authenticates with an API key from the account page or the CLI.

| Area | Paths |
| --- | --- |
| Inference | `POST /v1/messages`, `POST /v1/messages/count_tokens`, `POST /v1/responses`, `POST /v1/chat/completions`, `GET /v1/models`, `GET /v1/codex/models`, `HEAD /api/hello`, WebSocket `/v1/responses` |
| Working state | `GET /v1/state`, `POST /v1/cache/warm`, `GET/POST/DELETE /v1/cache/pins`, `DELETE /v1/cache`, `GET/PUT /v1/models/resident`, `GET/PUT /v1/models/aliases` |
| Usage records | `GET /v1/requests` |
| Context packs | `GET/POST /v1/packs`, `GET/DELETE /v1/packs/{name}`, `POST /v1/packs/{name}/refresh`, `POST/GET /v1/packs/trees` |
| Anticipator | `GET/PUT /v1/anticipator`, `GET /v1/anticipator/plan`, `POST /v1/anticipator/plan/approve`, `PUT /v1/anticipator/focus`, `GET /v1/anticipator/report` |
| Repositories | `GET/POST /v1/repos`, `DELETE /v1/repos/{name}`, git over HTTPS at `/git/{repo}.git` |
| Storage | `GET /v1/storage`, `PUT/GET/DELETE /v1/storage/{path}`, `GET /v1/storage/changes`, `POST /v1/export`, `POST /v1/restore` |
| Fetch | `GET/POST /v1/fetch`, `GET/DELETE /v1/fetch/{id}`, `POST /v1/fetch/{id}/resume` |
| Functions | `GET /v1/functions`, `POST /v1/functions/{name}/runs`, `GET /v1/runs`, `GET /v1/runs/{id}`, `GET /v1/runs/{id}/log`, `DELETE /v1/runs/{id}`, `GET/PUT/DELETE /v1/schedules`, `GET /v1/reports/{date}` |
| Draft models | `GET /v1/drafters`, `PUT /v1/drafters/active` |
| Customer models | `GET/POST /v1/custom-models`, `DELETE /v1/custom-models/{id}` |
| Codebase tools (MCP) | `POST /mcp` (Streamable HTTP), and `sparkpipe mcp` for stdio |
| Batch jobs | `GET/POST /v1/batches`, `GET/DELETE /v1/batches/{id}` |
| Teams, keys, audit | `GET/POST /v1/members`, `DELETE /v1/members/{id}`, `GET/POST /v1/keys`, `DELETE /v1/keys/{id}`, `GET /v1/audit` |
| Notifications | `GET/PUT /v1/webhooks`, `DELETE /v1/webhooks/{id}` |
| Several fleets | key option `fleet: "any"`, `POST /v1/state/move` |
| Inference extras | `POST /v1/tokenize`, `POST /v1/detokenize`, `DELETE /v1/requests/{id}` |
| Account | the existing `/market/...` endpoints, with a bearer session for the CLI, and auto-extend (§16) |

Everything here ships for 4 January 2027. §18 orders the work into parallel components and lists the acceptance tests.

## 3. Inference

### 3.1 Common rules

- **Keys.** Send the key as `Authorization: Bearer <key>` or `X-Api-Key: <key>`, because Claude Code sends the second when `ANTHROPIC_API_KEY` is set. A key belongs to one account. It can be limited to one fleet (`fleet`) and may be allowed to switch models (`allow_swap`, off by default). Keys are created on the account page or with `sparkpipe keys create`.
- **Which fleet.** A request goes to a fleet the account holds at that moment.
  - The fleet is the key's `fleet` if it has one.
  - Otherwise it is the `X-SparkPipe-Fleet` header, or the existing body field `"sparkpipe": {"fleet": "F1A"}` on chat completions.
  - Otherwise, if the account holds exactly one fleet, that one.
  - Holding several fleets and naming none answers 409 `fleet_required`.
- **Which model.** `model` names an offered model (`kimi-k3`), a pack alias (`kimi-k3+core`, §5) or an alias from the fleet's alias table (§4.5). A fleet keeps several models resident when they fit (§4.5).
  - A request for a resident model is served at once.
  - A request for a model that fits next to the resident ones loads it without unloading anything; the request waits 16–79 s.
  - A request for a model that would need another one unloaded answers 409 `model_not_loaded`, naming the resident models, unless the key has `allow_swap`. A fleet never loses a model, or its cache, by accident.
- **Size.** Request bodies up to 32 MiB. The gateway's cap rises from 8 MiB (`tools/sparkpipe_pay.py`), because a 1M-token conversation is about 4 MB of text before images.
- **Busy.** When all sequences are in use, the request waits up to 60 s for one, then answers 429 with an integer `retry-after` and `x-should-retry: true`, in the error each client retries: `rate_limit_error` on Messages, `rate_limit_exceeded` (type `requests`, with "try again in 5s" in the message, which Codex parses) on Responses, and `busy` on chat completions. Over the WebSocket the same refusal is an `error` event with `status` 429 and a `retry-after` header entry.
- **Keep-alives.** During long prefills the chat layer sends something every 10 s:
  - Messages streams send `ping` events. `message_start` waits for the first token, so it carries the engine's prompt and cache counts.
  - Responses streams, over SSE or WebSocket, repeat `response.in_progress`. Codex's SSE reader skips comment lines, so a comment would not reset its `stream_idle_timeout_ms`.
  - Chat completions streams send an SSE comment line.
  - Non-streamed Messages and Responses requests get their 200 headers at the first keep-alive, then a newline every 10 s before the JSON body; JSON parsers skip leading whitespace. An error after the first newline arrives as the shape's error body under that 200.
- **Path.** gateway (VPS) → SSH tunnel → the hub's chat layer (`tools/serving/chat_frontend.py`) → the model's `sparkpipe_model_api` → the 16 ranks. All three new API shapes are translated in the chat layer, because the tokenizer, chat templates and tool-call parsers live there (`chat_frontend.py:525`). The gateway passes the new paths through unchanged, with `X-SparkPipe-Account` and `X-SparkPipe-Fleet`.
- **No per-key limit.** No key has a request or concurrency limit of its own. A holder's 64 concurrent requests all wait for sequences; the busy 429 after 60 s is the only refusal for load on the Fleet. The gateway's own limit (§17) is a share of what it can process, so a holder meets it only when the gateway itself is the bottleneck.
- **Class.** A request is `interactive` (the default) or `background`. It names its class with the header `x-sparkpipe-priority: interactive|background`, or with the same key in its body's `metadata` (`"metadata": {"x-sparkpipe-priority": "background"}`, on all three shapes). The header wins when both are sent, and any other value is a 400 `invalid_request`. Claude Code's compaction and auxiliary calls (§3.8), a key made with `--priority background`, `<model>-bulk`, chat-memory jobs and Functions (§7.1) are background.
  - While background work runs, two sequences stay free for interactive requests. A freed sequence goes to a waiting interactive request first.
  - A background request never delays an interactive one by more than the engine submission already in flight. Background prefill yields between waves, background decode rides only in interactive decode steps while interactive work runs, and a background prompt never takes the streamed prefill.
  - Background work runs whenever no interactive work is queued or running, so it completes when interactive load stops. The class changes timing only, never the tokens.
- **Cancel.** `DELETE /v1/requests/{id}` cancels the caller's own queued or running request: the same account and, behind the gateway, the same member. The id is the `X-SparkPipe-Request` header of the reply, sent when a stream opens.
  - A client can choose the id itself: it sends `X-SparkPipe-Request: r_` followed by 24 lowercase hex digits on the request, and can then cancel the request while it still waits for a sequence, before any reply byte. Another value answers 400 `invalid_request`, and an id already queued or running answers 409 `request_id_in_use`.
  - It answers `{"id", "object": "request", "cancelled": true, "released"}` once the request's sequence is free (`released`), within 5 s.
  - The cancelled request ends with 409 `request_cancelled` in its own shape: an error response, or an error event on a stream.
  - Another account's or another member's id, or one that is not queued or running, answers 404 `request_not_found`.
  - Closing the connection cancels the same way. Either way the engine frees the sequence at the end of the submission in flight: a decode step, a chain or one prefill wave. KV already committed follows the retention rules.

### 3.2 Token-exact replay

The prefix cache matches token ids. A turn is warm only if the client's copy of the conversation renders to exactly the tokens the fleet produced. Clients send previous turns back as structured JSON: tool calls as objects, reasoning as blocks. A naive re-render changes whitespace, key order or reasoning text, and the cache then misses from that point on. With K3, a miss falls back to the last 16,384-token checkpoint.

The rule: for every assistant turn it returns, the chat layer stores the exact generated token ids, keyed by an opaque handle it hands to the client in a field the client already round-trips.

- Anthropic Messages: the `id` of each `tool_use` block, and the `signature` of each `thinking` block.
- Responses: the `id` of each output item, and `encrypted_content` of each reasoning item.
- Chat completions: `tool_calls[].id`, and the reply text, which already works.

When a later request carries those handles, the turn is rendered from the stored tokens. Unknown or missing handles fall back to a normal render, which is correct but may miss the cache.

The store is working state: it lives on the fleet, follows the same deletion rule, and holds only what the fleet generated for this account.

### 3.3 `POST /v1/messages` (Claude Code)

Anthropic Messages, streaming and non-streaming. What Claude Code sends and how the fleet answers:

- **System.** `system` as a string or an array of text blocks. `cache_control` markers are accepted and ignored: caching is automatic and has no TTL while the fleet is held. Mid-conversation `role: "system"` messages are rendered in place as system turns.
- **Content blocks.** `text` and `tool_use`. `tool_result`, whose content is a string or blocks. `thinking` with `signature`, replayed per §3.2. `image` only on models that take images; otherwise 400 `image_not_supported`.
- **Tools.** `tools` with `input_schema`, and `tool_choice` set to `auto`, `any`, `tool` or `none`, mapped to the model's tool-call format by the chat layer's parsers. `defer_loading` and `tool_reference` (MCP tool search) are not supported; the setup turns that beta off.
- **Reasoning.** `thinking` (`enabled` with `budget_tokens`, or `adaptive`) and `output_config.effort` map to the model's reasoning switch and, where the model has one, its budget. Otherwise they are ignored.
- **Other fields.** `max_tokens`, `temperature`, `top_p`, `top_k`, `stop_sequences` and `stream` work. Unknown fields are accepted and ignored, and so are `anthropic-beta` headers. `metadata` is read for the session id in `user_id` (§3.8) and for the class key `x-sparkpipe-priority`.
- **Response.** `id`, `content` blocks, and `stop_reason`: `end_turn`, `tool_use`, `max_tokens` or `stop_sequence`.
  - **Usage:**
    - `cache_read_input_tokens` is the prompt tokens the engine matched in its cache (for an API key, at most the prefix that key's account itself sent, in 64-token blocks);
    - `cache_creation_input_tokens` is 0 until the engine reports the prompt tokens it commits;
    - `input_tokens` is the rest;
    - `output_tokens` is what was generated.

    The prompt length and the cache count are the engine's own, never a count made in the chat layer. The sum of the three input fields is always the prompt length, in `message_start` and in `message_delta`: Claude Code keeps a `message_start` field wherever the `message_delta` field is 0, so both carry the same numbers. An engine that reports its counts only at the end gets zeros in `message_start`.
- **Stream.** `ping` while the prompt is prefilled, then `message_start`, `content_block_start/delta/stop` (`text_delta`, `input_json_delta`, `thinking_delta`, `signature_delta`), `message_delta` with cumulative usage, `message_stop`, and `error`. The stream never ends before `message_delta`.
- **Errors.** Errors use Anthropic's body, `{"type":"error","error":{"type","message"}}`. 429 and 529 carry `retry-after`.

`POST /v1/messages/count_tokens` returns the exact count: the length of the token ids the same request would send the engine, replay included. `HEAD /api/hello` answers 200. `GET /v1/models` lists the resident models, the other offered models, the account's pack aliases and its custom models.

**Claude Code setup** (written by `sparkpipe setup claude-code`):

```sh
ANTHROPIC_BASE_URL=https://sparkpipe.ai/pay/api
ANTHROPIC_AUTH_TOKEN=<key>
ANTHROPIC_MODEL=kimi-k3+core
ANTHROPIC_DEFAULT_HAIKU_MODEL=kimi-k3+core
CLAUDE_CODE_MAX_CONTEXT_TOKENS=1000000
CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
CLAUDE_CODE_ATTRIBUTION_HEADER=0
CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000
```

- **The Haiku alias points at a small resident model** when one fits next to the main model (GLM 5.3 Flash beside GLM 5.3, for example), and at the main model otherwise. Claude Code's background calls therefore never load a model that does not fit. The alias table catches any other `claude-*` name as well.
- **The context limit is the model's served window.** Claude Code otherwise assumes 200K.
- **The attribution header is turned off,** so the first system block stays the same from session to session.
- **The stream idle timeout is 30 minutes.** Claude Code 2.1.308 counts a `ping` towards its stream watchdog for only 30 heartbeats in a row (300 s), then aborts 300 s later, so a cold prefill longer than about 10 minutes would fail with the default. Setup sets `CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000`, which moves the abort to about 35 minutes.
- **The fleet's MCP server (§10) is registered at user scope:** `mcpServers.sparkpipe` (`type: "http"`, the `/mcp` URL and the key as a bearer header) in `~/.claude.json`, or `$CLAUDE_CONFIG_DIR/.claude.json` when that is set, the file `claude mcp add --scope user` writes. Claude Code 2.1.308's `claude mcp get sparkpipe` lists it from both places.
- **Every Claude Code release is tested** against this endpoint before we tell customers to upgrade, because Anthropic does not test Claude Code against other models ([LLM gateway docs](https://code.claude.com/docs/en/llm-gateway.md)).
- **Claude Code's terms keep it on the customer's own machine.** Functions never run Claude Code on our servers (§7).

### 3.4 `POST /v1/responses` (Codex)

OpenAI Responses, which is the only wire format Codex supports for custom providers.

- **Request fields:**
  - `input` as a string or items: `message`, `function_call`, `function_call_output` and `reasoning`;
  - `instructions` and function `tools`;
  - `tool_choice`, `parallel_tool_calls`, `max_output_tokens`, `temperature`, `top_p`, `stream`, `store` and `previous_response_id`.
- **Previous responses.** `previous_response_id` works for responses stored as working state, which is the default here. `store: false` stores no response for later reference, but the replay store of §3.2 still keeps the generated tokens. `prompt_cache_key` is the session: it reaches the telemetry record as `session` and drives affinity (§3.8).
- **Stored responses.** `GET /v1/responses/{id}` returns a stored response as it was answered, and `DELETE /v1/responses/{id}` deletes it. Both answer 404 for an id of another account or of another member of the same account, as `previous_response_id` does. Stored responses are working state (§4): `DELETE /v1/cache`, a wipe at a change of user and the 24-hour retention remove them.
- **WebSocket.** `GET /v1/responses` with a WebSocket upgrade is Codex's transport for `supports_websockets`. Each `response.create` message carries the request body and gets the same events as the SSE stream, one JSON text message each.
  - A `previous_response_id` that names the connection's last response works even with `store: false`.
  - `generate: false` keeps the input on the connection without generating.
  - Errors before a response starts are `error` events with `status`, the error body and the `retry-after` header.
  - The telemetry record's `transport` (`http` or `websocket`) and `turn_ms` compare the two transports.
- **Stream events:**
  - `response.created` and `response.in_progress`;
  - `response.output_item.added` and `response.content_part.added`;
  - `response.output_text.delta` and `response.output_text.done`;
  - `response.function_call_arguments.delta` and `response.function_call_arguments.done`;
  - `response.reasoning_summary_text.delta`, where the model reasons;
  - `response.output_item.done`, `response.completed` and `error`.
- **Usage:** `input_tokens` (the engine's prompt length), `input_tokens_details.cached_tokens` (the engine's cache count), `output_tokens` and `output_tokens_details.reasoning_tokens`.
- **Catalogue.** `GET /v1/codex/models` gives each model's served window from its lane record as `context_window`, `effective_context_window_percent` 100 and `auto_compact_token_limit` (the window less the output limit), so Codex needs no manual window setting.

**Codex setup** (written by `sparkpipe setup codex` into `~/.codex/config.toml`):

```toml
model = "kimi-k3+core"
model_provider = "sparkpipe"

[model_providers.sparkpipe]
name = "SparkPipe"
base_url = "https://sparkpipe.ai/pay/api/v1"
env_key = "SPARKPIPE_API_KEY"
wire_api = "responses"
stream_idle_timeout_ms = 600000
model_catalog_url = "https://sparkpipe.ai/pay/api/v1/codex/models"
```

### 3.5 `POST /v1/chat/completions`

This endpoint already exists in the gateway; OpenAI-compatible tools and SDKs use it. It gains:
- `usage.prompt_tokens_details.cached_tokens`;
- the alias and pack names;
- keep-alive comments;
- replay per §3.2.

### 3.6 Telemetry

Every response carries what it cost in time:
- the non-streaming body and the last stream event (`message_delta`, `response.completed` or the final chunk) include a `sparkpipe` object;
- headers carry what is known before the first byte.

```json
"sparkpipe": {
  "request": "r_7Q2…", "fleet": "F1A", "model": "kimi-k3", "pack": "core@a41f09",
  "prompt_tokens": 182340, "cached_tokens": 181760, "prefill_tokens": 580, "engine_rounds": 1,
  "queue_ms": 0, "prefill_ms": 1240, "ttft_ms": 1310,
  "output_tokens": 912, "decode_tps": 15.8, "draft_accept": 0.71,
  "cold_prefill_ms_estimate": 312000
}
```

`cold_prefill_ms_estimate` is the measured prefill rate for this model at this length, applied to the tokens that came from the cache: the wait the cache removed. `engine_rounds` counts the model's generations behind the response. It is more than one when context memory answered one of its own tool calls before the reply; then the prompt and cache counts are the last generation's, `output_tokens` counts every generation, and `prefill_ms` and `ttft_ms` run from the first. `GET /v1/requests?since=&until=` returns these records, without prompts or replies; they are the operational records the privacy policy already lists, kept for a year. `sparkpipe stats` sums them into "prefill avoided this week", which is the number the Customer Zero report publishes.

### 3.7 More inference features

- **Structured outputs.** `response_format: {"type": "json_schema", ...}` on chat completions, `text.format` on Responses, and forced tool use on Messages always produce output that matches the schema. The engine's sampler masks tokens with a grammar compiled from the schema. Today `response_format` other than text answers 400 (`LITELLM_FRONTEND.md:59-62`).
- **Parallel samples.** `n` up to 8 on chat completions and `n` in the body of Messages and Responses. The samples share the prompt's cache and run as one batch: best-of-N patches cost one prefill. Today `n` answers 400.
- **Log probabilities.** `logprobs` and `top_logprobs` up to 20, for evals and confidence checks.
- **Tokens.** `POST /v1/tokenize` and `POST /v1/detokenize` with the model's tokenizer and chat template, so tools can count and place context exactly.
- **Cancellation.** A client disconnect, or `DELETE /v1/requests/{id}`, stops generation within one decode step and frees the sequence. Agents interrupt often; an abandoned generation must not hold a sequence.
- **Priority.** `X-SparkPipe-Priority: interactive | background` (and a default per key). Background requests yield sequences and prefill to interactive ones, and Functions and batches always run as background. The chat layer passes priority to `sparkpipe_model_api`.
  - **One name, two places.** A client that can set headers sends the `X-SparkPipe-Priority` header. One that can only set body fields sends the same name as a key of the request's `metadata` object, `"metadata": {"x-sparkpipe-priority": "background"}`, on any of the three shapes. The header wins when both are present, and any other value is a 400 `invalid_request`.
  - **Harness hints.** Without either, the harness's own marks decide: Claude Code's `x-claude-code-request-class` `compaction` and `auxiliary`, and Codex's `x-openai-subagent: compact`, run at background priority; everything else is interactive.
- **Context memory.** `X-SparkPipe-Memory: on` gives a long session the context memory SparkPipe Chat uses: earlier exchanges are summarized, the newest are kept in full, and the model can read older detail through memory's own tools, whose calls the chat layer routes to the memory program, so the client never sees them ([MEMORY_PROTOCOL.md](MEMORY_PROTOCOL.md)). The session is the key's account and member plus the client's session id: `X-Claude-Code-Session-Id`, `X-Session-Id`, the Responses `prompt_cache_key` or the Messages `metadata` session; each Claude Code subagent has its own. Opting in without a session id answers 400, and the harness's own compaction requests stay raw. The client keeps sending its whole history and nothing in it is lost; usage counts the tokens the model read. While memory is unavailable, the request is served on its raw history.
- **Images** on models that take them, in all three API shapes.
- **Seeds.** `seed` gives reproducible sampling for the same prompt, model build and batch.

### 3.8 The three harnesses we support first-class

Senior engineers use Claude Code most, by the widest margin, then Codex and OpenCode. Sources:
- [Pragmatic Engineer, 2026](https://newsletter.pragmaticengineer.com/p/ai-tooling-2026): median 11–15 years of experience, Claude Code at 75% in the smallest companies. Staff+ engineers name OpenCode as a favourite as often as Copilot;
- [Stack Overflow, 2026](https://survey.stackoverflow.co/2026/ai);
- [JetBrains, 2026](https://blog.jetbrains.com/research/2026/08/ai-coding-agent-adoption-2026/).

We support these three as deeply as each one allows:
- a one-command setup;
- every extension point the harness offers;
- a release check that runs a scripted session against staging within a day of each release, before customers are told to upgrade.

Any other tool that speaks Anthropic Messages, OpenAI Responses or chat completions works too, without a setup command or a release check.

**Undo.** Each setup records, per file, the backup of the version it first replaced (`setup.json` beside the CLI's `config.json`). `sparkpipe setup <harness> --undo`, or `sparkpipe setup --undo` for all three, puts back the pre-setup value of every key setup owns and keeps every other setting as it is now, including what the user or the tool wrote since. Claude Code adds permissions to `settings.json` and rewrites `~/.claude.json` all the time, so restoring whole files would lose them. A file setup created is removed when nothing else was written into it. The undo needs no network and no key, backs up the file it changes, and refuses, changing nothing, when the recorded backup is gone.

**Release check.** `tools/client_release_check.py --endpoint <staging> --tool claude-code=<release> --tool codex=<release> --tool opencode=<release> --out result.json` (`latest` resolves through `npm view`), with the staging key in `SPARKPIPE_API_KEY`:
- installs each release with `npm install --prefix` into a temporary directory, never globally, and gives each tool a fresh temporary home;
- runs `sparkpipe setup <harness>` against a local relay (the proxy of §15 with a recorder), so the tool only ever holds the relay's one-time key and the staging key reaches only the fleet;
- runs a three-turn session in a scratch repository (read a file, recall it, run a shell command) through each tool's non-interactive mode (`claude -p`, `codex exec`, `opencode run`), with `HTTPS_PROXY` pointing at a recorder that refuses and lists every other host;
- writes the result as JSON, with one boolean per check: installed, setup, every turn finished, the session kept, the answers, tool calls, cache hits after the first turn, an MCP handshake answered, the affinity key on every inference request, Claude Code's hint headers, and no OpenCode request to opencode.ai. It also lists the request classes and background requests the relay saw, and the outside hosts each tool tried. It exits 0 only when every check passes.

The timer that runs it within a day of each release, and the internal status page, are run by the lead (H6).

**Claude Code** (Anthropic Messages, §3.3)
- **Setup.** `sparkpipe setup claude-code` writes the environment of §3.3, with three additions:
  - `CLAUDE_CODE_SUBAGENT_MODEL` set to the same model, so subagents never swap the fleet;
  - `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1`;
  - a `modelOverrides` entry, so `/model` lists our models.
- **Priority from hint headers.** With hint headers on, Claude Code marks each request with `x-claude-code-request-class`. Claude Code 2.1.308 sends `main`, `subagent`, `workflow`, `compaction`, `auxiliary` and `side_reply`: `compaction` and `auxiliary` (titles, summaries, quota checks and the other calls outside the main thread and its subagents) run at background priority, and the rest at interactive priority.
- **Affinity.** `X-Claude-Code-Session-Id`, which Claude Code 2.1.308 sends on every request, and `x-claude-code-agent-id` drive pack-version affinity (§5), routing between fleets (§14) and fair share (§12). Without the header, the session is the `session_id` in `metadata.user_id`, a JSON string in 2.1.x and a `_session_<id>` suffix in older releases.
- **The SparkPipe plugin for Claude Code**, from our plugin marketplace on GitHub, bundles:
  - the MCP server of §10;
  - slash commands: `/sparkpipe:status`, `/sparkpipe:report`, `/sparkpipe:stats`, `/sparkpipe:run` (tests on the sandbox host with the local diff) and `/sparkpipe:fn`;
  - a `SessionStart` hook that warms the session's pack after a resume, a compaction or a long pause (§4.2);
  - a skill that tells the agent when to use the fleet's `run`, `search_code` and `pack_info` instead of local tools.
- **Status line.** `sparkpipe statusline` shows the fleet, the resident models, the cached share of the last turn and the prefill avoided today.

**Codex** (OpenAI Responses, §3.4)
- **Setup.** `sparkpipe setup codex` writes two files in `$CODEX_HOME` (`~/.codex`):
  - In `config.toml`: the `model`, `model_provider` and provider of §3.4, whose `model_catalog_url` points at `GET /v1/codex/models`, a Codex-native catalogue of our models with context windows and reasoning settings. Also `[mcp_servers.sparkpipe]` for §10, with `bearer_token_env_var = "SPARKPIPE_API_KEY"`.
  - The profile `sparkpipe.config.toml`, which `codex --profile sparkpipe` lays over `config.toml` (Codex 0.134.0 and later read profiles only from such files, with top-level keys, not from `[profiles.<name>]` tables). It holds:
    - `model` and `model_provider`;
    - `model_context_window` and `model_auto_compact_token_limit` (the context window minus the output limit), set from the model;
    - `[agents]` with `default_subagent_model` set to the same model, and `max_concurrent_threads_per_session` set to the fleet's sequences less the main thread's.

  It merges into `config.toml` without changing any other setting, and refuses, leaving the file as it was, when it cannot. The profile file is SparkPipe's own and is rewritten whole.
- **Affinity.** Codex's session-derived `prompt_cache_key` drives pack-version affinity and routing. Its local compaction runs as a subagent marked `x-openai-subagent: compact`, which runs at background priority.
- **WebSocket transport.** `/v1/responses` also accepts Codex's optional WebSocket transport (`supports_websockets`), which removes a connection set-up per turn.

**OpenCode** (Anthropic Messages through its `@ai-sdk/anthropic` provider, so tools and reasoning keep full fidelity)
- **Setup.** `sparkpipe setup opencode` writes `opencode.json`:
  - the provider, with `baseURL` and `{env:SPARKPIPE_API_KEY}`;
  - `limit.context` and `limit.output` per model;
  - the subagent model set to the same model (the `general` and `explore` agents, and `compaction`);
  - `small_model` and the hidden `title` and `summary` agents on the fleet's background model, so titles and summaries never go to another provider;
  - the remote MCP server of §10;
  - `prune` off, because pruning rewrites history and misses the cache. Off is OpenCode's default and the key is deprecated in OpenCode 2, so setup only turns off a `prune` the file turned on;
  - `"share": "disabled"`, so sessions are never uploaded outside Korea.

  These are OpenCode 1's keys (`provider`, `small_model`, `agent`, `mcp`), which OpenCode 2 still reads.
- **The SparkPipe plugin for OpenCode** does what the Claude Code plugin does: it warms on session start, reports status and adds the fleet commands.
- **models.dev.** Our models and their served context windows are published there, so OpenCode knows them without any manual setting.

## 4. Working state

### 4.1 `GET /v1/state`

Returns:
- the resident models, their memory use and the alias table;
- cache use against capacity: device pages, snapshot store and mirror state;
- pins, packs and their warm status;
- the active drafter;
- the Function runs in progress;
- the time the working state would be deleted if the account stopped holding the fleet.

### 4.2 `POST /v1/cache/warm`

Prefills a prompt and commits it to the cache without generating anything.

- **Body.** The body is a Messages, Responses or chat request (`format`: `anthropic`, `openai_responses` or `openai_chat`). It is rendered exactly as a real request would be, so the warmed prefix is the prefix a later request will match.
- **Response.** `prompt_tokens`, `cached_tokens`, `committed_tokens` and `prefill_ms`.
- **Priority and limit.** Warm requests run at low priority: they wait for interactive requests and never delay them. They count against a per-fleet budget of the cache, published on the Fleet List.
- **Build.** The engine gets a commit-only request flag. Until then the chat layer sends `max_tokens` 1 and discards the token, as chat memory's prewarm does.

Warm-ups are used:
- by the CLI and Functions after a handover, a release or a pack refresh;
- by an optional Claude Code `SessionStart` hook that warms the current pack when a session starts after a long pause.

### 4.3 Pins

- **`POST /v1/cache/pins {"request": "r_…"}` or `{"pack": "core"}`** pins the prompt prefix of a past request or warm-up, or a pack's current version. A pinned chain is evicted only when the cache would otherwise refuse a request, and then after every unpinned chain.
- **Limits.** Pins are limited to half the fleet's cache.
- **`GET` and `DELETE`** list and remove pins.
- **Build.** The page cache already evicts by priority (`TECHDEBT.md:608-630`). Pins add a priority the coordinator's LRU skips (`cache/prefix_cache.c:739-800`).

### 4.4 `DELETE /v1/cache`

Deletes the account's working-state cache on that fleet now, as clause 6.2 of the Agreement allows ("earlier when Customer asks"). With `?storage=true`, the customer storage goes too.

- The answer is `{"deleted", "not_deleted", "note"}`. `deleted` names what went: the replay store, the alias table, stored responses, packs (and repositories with `?storage=true`), and `memory` once the memory program confirms it forgot the account's context-memory sessions. `not_deleted` names anything still to go.
- The account's cached-token ledger (§3.3) is cleared with it, and only that account's.

### 4.5 Resident models and aliases

A fleet keeps as many models resident as fit in its memory. Kimi K3 (1.56 TB) fills most of a 16-Spark fleet on its own. Smaller models share it: for example GLM 5.3 with GLM 5.3 Flash, or several 27–31B models together.

- **`GET /v1/models/resident`** returns:
  - the resident models;
  - each one's memory for weights and cache on the busiest node;
  - the headroom left;
  - which offered models would fit next to them.
- **`PUT /v1/models/resident {"models": ["glm-5.3", "glm-5.3-flash"]}`** makes that set resident, if it fits. Models outside the set are unloaded, and their cache is saved to the snapshot store and comes back when they return. A set that does not fit answers 409 `does_not_fit`, with the numbers. This needs a key with `allow_swap`, or the account session.
- **Cache per model.** Each resident model has its own cache pool within the per-node budget. A request for one model never evicts another model's cache.
- **Sharing sequences.** Each resident model has its own sequences. Interactive requests come first across all of them (§3.7).
- **Aliases.**
  - `PUT /v1/models/aliases {"claude-haiku-*": "glm-5.3-flash", "claude-*": "glm-5.3+core", "gpt-*": "glm-5.3+core"}` maps names that clients send by themselves.
  - The default table maps `claude-*` and `gpt-*` to the first resident model.

## 5. Context packs and the anticipator

A pack is a snapshot of a repository rendered into one deterministic text block. The fleet places it first in every prompt that names the pack's alias: before the client's own system prompt. Every session, subagent and Function then shares one warm prefix that already holds the code. The agent reads less and searches less, and none of it is prefilled twice.

Precomputed context only saves time when a real request starts with exactly those tokens. Claude Code, Codex and OpenCode begin every prompt with their own system prompt and the conversation, and the files arrive later through tool calls. So the fleet controls what a prompt starts with by placing a pack first. The anticipator (§5.3) decides which packs to build and keep warm, and §5.4 says how a session lands on one.

### 5.1 Packs

- **Build.** `POST /v1/packs {"name": "core", "repo": "main", "include": ["src/**", "include/**"], "exclude": [...], "budget_tokens": 400000}`.
  - Files come from an ingested repository (§6). They are ordered by how rarely they changed in the last 90 days, least-changed first, with ties broken by path.
  - Each file is rendered as `<file path="…" commit="…">…</file>` after a header that names the repository and gives the instruction to read a file before editing it. `commit` is the last commit that changed that file.
  - The snapshot's own commit and its file count come after the last file, not in the header, so a push changes no token before the first changed file.
  - The header, each file block and the footer are tokenized separately with the model's tokenizer and joined, so a file's tokens never depend on its neighbours.
  - Binary and generated files and anything over the budget are left out and listed in the response.
- **Alias.** `<model>+<pack>`, for example `kimi-k3+core`.
- **Placement.** The pack is one system turn before all of the client's messages: before the Messages `system`, a chat request's own system messages and Codex's `instructions`. The template renders it like any system turn. What a template puts ahead of the messages, such as tool declarations, stays ahead of the pack, so the stored pack prefix is shared by requests with the same tools and template settings. Warm-ups therefore render the pack with each tool set recently seen on it.
- **Refresh.**
  - `POST /v1/packs/{name}/refresh`, or automatically after a push (§6), builds a new version.
  - Only the part after the first changed file has to be prefilled again. For K3 that means everything after the last 16,384-token checkpoint before it, which is why rarely changed files come first.
  - The new version is prefilled at low priority, and the alias moves to it only once it is warm. No request ever waits for a rebuild.
- **Session affinity.** A conversation that started on version A keeps version A, matched by its prompt prefix and by `x-claude-code-session-id` when present, because switching it would invalidate the conversation's whole cache. New sessions get the newest warm version. Old versions are dropped when no session has used them for 24 hours.
- **Cost.** A pack makes every prompt longer, so decode slows a little: K3 runs at 16.2 tok/s at 8K, 15.1 at 73K and 12.0–12.4 at 1M (`PERFORMANCE_STATUS.md:36-38,65-81`). The budget is the customer's choice. Gate D records what a pack saves on a real task list, for the Customer Zero report.

### 5.2 Pack trees

A pack tree shares one stored prefix across many packs. Each level is a contiguous run of files:

| Level | What it holds | Changes |
| --- | --- | --- |
| Root | Specs, frozen headers and interfaces, project instructions (`AGENTS.md`, `CLAUDE.md`), the files every task needs | Monthly |
| Trunk | One subsystem's headers, types, tables and utilities | Weekly |
| Leaf | A working set: a module or module pair, its tests and the recent diff | Daily |

A leaf pack is root, trunk and leaf rendered one after another, named `<tree>/<leaf>` (for example `kimi-k3+core/net-crypto`).

- **Stored once.** The prefix cache is hash-chained in 64-token blocks, so every leaf of a tree reuses the stored root and trunk; only each leaf's own tail is extra.
- **Aligned to resume points.** Each level ends on the model's resume boundary: 16,384 tokens for K3 and 4,096 for GLM 5.3 Flash, and any 64-token block for models that resume anywhere. The builder pads a level with a fixed separator up to the boundary, so moving from one leaf to another never recomputes the shared part.
- **Budget.** `max_leaf_tokens` caps a whole leaf pack (default 400K, up to the model's window).
- **API.**
  - `POST /v1/packs/trees {"name": "core", "repo": "main", "root": [...], "trunks": {...}, "leaves": "auto" | {...}}` defines a tree. Leaves can be named by the customer, proposed by the anticipator, or both.
  - `GET /v1/packs/trees/{name}` lists its levels, versions, tokens and warm state.

### 5.3 The anticipator

Every night, the anticipator chooses which leaves to build and keep warm for the next working day, within the budgets the customer sets. It explains each choice, and it measures the next day whether its choices were used.

**Evidence.** Each source can be switched off:
1. **Structure.** The include and import graph and the directory layout, from the repository index (§6).
2. **History.** Files changed together over the last 90 days, the day's commits, and branches with recent pushes.
3. **Sessions.** Which files the account's sessions read, edited or searched in the last 7 days, and which packs they started on.
   - This is taken from the paths in tool calls that the chat layer already sees, never from file contents.
   - It is kept as working state on the fleet and deleted with it.
4. **Hints.** What the customer says tomorrow is about: `sparkpipe anticipate focus src/consensus/** "allocator migration"`, a task list file, or pinned leaves.

**Planning.**
- **Candidates.** Clusters of files that the evidence ties together, each fitted into a tree as a leaf: directories, pairs of directories, and each recent session's working set.
- **Score.** Each candidate is scored as:
  - the expected uses tomorrow (recent sessions on those files, decayed by age, raised by hints)
  - × the prefill time a use would save (its tokens at the model's measured cold prefill rate)
  - − the night's prefill time to build it, counted at a tenth of a daytime wait, since night prefill runs on an idle fleet
  - − the cache it takes.
- **Selection.** The plan is chosen greedily by score per gigabyte of new cache, counting shared roots and trunks once, until the night's compute budget or the cache budget runs out.
- **Output.** Every planned leaf lists its files, the signals behind it, its expected uses and its build time.

**When it runs.**
- It runs in the night window, only in time the fleet is not serving interactive requests.
- It runs after any Function that unloads the model (drafter training), so the cache is built for the model that serves in the morning.
- It finishes before the window ends.
- Cache is per model, so it builds for the account's main resident model unless told otherwise.

**Scale.** These are estimated from our measurements, not yet measured as a whole:
- **Build rate.** K3 prefills about 460–590 tokens per second at long context, so a 14-hour night builds about 25 million tokens of new context. That is about 50 unshared 500K-token contexts, and many more when leaves share a root and trunks.
- **Storage.** K3's cache takes about 1.9 KB per token per node, so 500 GB per node holds about 260 million tokens.
- **Payoff.** A 500K-token context costs about 18 minutes to prefill cold, against 3–6 seconds warm.

### 5.4 Starting a session on a pack

Three ways, each per project directory:
1. **Named.** `sparkpipe setup claude-code --pack core/net-crypto` writes the project's settings, so every session started there uses that leaf (Claude Code `.claude/settings.local.json`, a Codex profile, OpenCode's project `opencode.json`).
2. **Automatic.** The alias `<model>+auto` lets the chat layer pick a leaf at a session's first request. It matches:
   - the working directory and git branch, which Claude Code and Codex include in their first prompt;
   - the files the first prompt names.

   The pick is fixed for the whole session, shown in the response telemetry (`pack`) and the status line, and recorded as session evidence. When no leaf matches well enough, the session starts without a pack.
3. **None.** The plain model name.

The fleet never changes the prefix of a running conversation, and never removes files from a client's messages, even when they are already in the pack. Rewriting the client's history would break token-exact replay (§3.2) and surprise the agent. Instead:
- The pack header lists its files and commit.
- The MCP tool `pack_info` (§10) answers the same question, so the agent can skip reading what it already has, and reads the current version of any file changed since.

### 5.5 During the day

After each push, the leaves that contain changed files are rebuilt from the first changed file, starting at the resume boundary before it. This happens at background priority, between interactive turns.
- New sessions get the new version once it is warm.
- Running sessions keep theirs (§5.1).
- Daytime refresh uses only sequences that interactive use leaves free.

### 5.6 Controls

| Setting | Values | Default |
| --- | --- | --- |
| `mode` | `off`; `plan` (plan only, build what the customer approves); `auto` (build within budget) | `auto` |
| `window` | Local start and end time, and time zone | 19:00–09:00, account time zone |
| `max_hours` | Prefill hours a night | All idle time in the window |
| `cache_share` | Share of the fleet's cache that anticipated leaves may use | 50% |
| `max_leaf_tokens` | Largest leaf pack | 400,000 |
| `signals` | `structure`, `history`, `sessions`, `hints`, each on or off | All on |
| `exclude` | Paths never put in a pack | `.gitignore`, `.env*`, keys and certificates, vendored and generated files |
| `model` | Model the cache is built for | The main resident model |

API and CLI:
- `GET/PUT /v1/anticipator` holds the settings.
- `GET /v1/anticipator/plan` shows tonight's plan with reasons.
- `POST /v1/anticipator/plan/approve` approves all of it or chosen leaves, in `plan` mode.
- `PUT /v1/anticipator/focus` gives hints for the next day.
- `GET /v1/anticipator/report?date=` shows the results.
- `sparkpipe anticipate status | plan | approve | focus | report | off | auto`.
- Pins (§4.3) keep any leaf warm regardless of score.

### 5.7 Does it work

Every morning report includes:
- what the anticipator built and why;
- how yesterday went: sessions started, the share that started on an anticipated leaf, the share of those sessions' first-turn tokens served from it, and the prefill time avoided (from telemetry, §3.6);
- the leaves that were built but never used, and the hours they took.

In `auto` mode, a leaf unused for three nights loses score.

The acceptance test is Gate D2 (§18). It is published in the Customer Zero report whatever the result, so customers see the measured value, not a promise.

## 6. Storage and repositories

Customer storage is on the fleet's nodes: 500 GB per node, mirrored to the pair partner, 4 TB in all, in the scratch area of `docs/WORKING_STATE.md`. It follows the working-state deletion rule.

- **Small files.** `PUT /v1/storage/{path}` (with `Content-Range` for resumable chunks of up to 32 MiB), `GET`, `DELETE`, and `GET /v1/storage?prefix=` with sizes and totals.
- **Large files** are fetched by the fleet itself (§6.1), so model weights never pass through the public gateway or the customer's laptop.
- **Repositories.** The customer pushes to the fleet; the fleet never holds a token for the customer's git host.
  - `POST /v1/repos {"name": "main"}` creates a repository on the fleet, served as a git remote at `https://sparkpipe.ai/pay/api/git/main.git` and authenticated with the account's SparkPipe key.
  - `git remote set-url --add --push origin https://sparkpipe.ai/pay/api/git/main.git`, or one step in the customer's CI, pushes every commit to the fleet as it goes to the usual host.
  - Each push triggers the refreshes the account has scheduled (§5).
  - A first copy of a large repository can also come in through a fetch (§6.1).
- **Export and restore.**
  - `POST /v1/export {"paths": [...], "include": ["drafter", "packs"]}` writes an archive to storage, for download or for fetching from another fleet.
  - `POST /v1/restore {"url" | "path"}` brings one back.
  - Cache snapshots are not exported: they only work on the same model build and topology, and rebuilding them is a warm-up.
- **Daily backup.** The customer's own machine or server pulls what changed.
  - `GET /v1/storage/changes?since=` lists the files added, changed or deleted since a point.
  - `sparkpipe storage sync models/ ./backup/` (or to a bucket the customer's machine can write to) copies only those changes. It is meant to run from the customer's cron or CI, with the customer's own cloud credentials, which never reach us.
  - The morning report says when each path was last pulled.
  - Clause 6.3 of the Agreement tells customers to treat working state as temporary, even on a fleet held many weeks ahead, and recommends a daily export. This makes that one scheduled command.

### 6.1 Fetch

Terabytes move between the fleet and the source over the fleet's own line, while the customer's machine sends only the request. An engineer on a plane can start a 1.5 TB model download, or a repository and its datasets, and get a notification when it is in storage.

- **Request.** `POST /v1/fetch {"source": "hf://org/model@revision", "to": "models/x/", "include": ["*.safetensors", "*.json"], "then": "pack"}`, with the source's token in the `X-SparkPipe-Source-Token` header when the source needs one, returns a fetch id at once. `GET /v1/fetch` lists the account's fetches as `{"data": [...]}`, each with `id`, `state`, `source`, `to`, `bytes` and `total_bytes`. `GET /v1/fetch/{id}` reports files, bytes, rate and checks; `DELETE` cancels. `sparkpipe fetch` and `sparkpipe fetch ls` are the same in the CLI.
- **Sources:**
  - `hf://` models and datasets, pinned to a revision (resolved to a commit hash when the fetch starts);
  - `s3://` and S3-compatible buckets (R2, GCS through S3 compatibility, MinIO), through presigned URLs or a key sent with the request;
  - `https://` URLs, with stored headers;
  - `git+https://` repositories with LFS.
- **Transfer:**
  - The files are spread over the fleet's nodes, each downloading its share straight into its own storage, which is then mirrored to its pair partner.
  - Large files are downloaded in parallel ranges and resume after an interruption.
  - Each file is checked against the source's published SHA-256 (Hugging Face's LFS hashes, S3 checksums or ETags). A mismatch is downloaded again, and the fetch fails if it keeps happening.
- **Chaining.** `"then": "pack"` qualifies and packs a customer model (§9) when the download finishes; `"then": "warm"` warms a pack (§5). The result appears in the account's model list or pack status.
- **Notices.** `fetch.finished`, `fetch.failed` and `fetch.needs_token` webhooks and emails (§13), and a line in the morning report.

Where a fetch may go:
- **Only into the account's storage** on a fleet it holds, within the storage quota, and only while it holds the fleet. A fetch still running when the account stops holding the fleet is cancelled, and partial files follow the working-state deletion rule.
- **Only to the public internet.** The fetch service resolves each host, pins the address it connects to, and refuses:
  - private, loopback and link-local addresses;
  - our own hosts and networks;
  - any redirect that leads to one of them.

  The fleet can therefore never be used to reach the hub, the nodes or the facility's internal services. Only HTTPS is used, plus git over HTTPS.
- **Downloads are data, never run.** The fetch service writes files and runs nothing in them. The packer accepts weights in safetensors and GGUF only, and refuses pickle-based formats (`.bin`, `.pt`, `.pkl`), which can execute code when loaded. Converting those is the customer's own job, in the sandbox (§7.1).
- **Bandwidth is shared** with the facility's other traffic and is not promised.

### 6.2 Tokens stay with the customer

We never store a customer's credentials for other services.

- **With each request.** A fetch carries the token it needs in `X-SparkPipe-Source-Token`.
  - The CLI keeps tokens in the operating system's keychain (`sparkpipe tokens set hf`, read without echo).
  - The account page can keep one in the browser's own storage on that device.
  - Either sends the token only with a fetch to the host it belongs to.
- **Held in memory, for that fetch only.** The fetch worker keeps the token in memory while its fetch runs. It never writes it to disk or a log, and never sends it to any host but the source's. A Hugging Face token never goes anywhere but huggingface.co, even through a redirect. The token is dropped when the fetch ends, fails or is cancelled.
- **Long fetches.** If the worker restarts in the middle of a fetch, the fetch pauses, keeps the files done so far, and sends `fetch.needs_token`. `sparkpipe fetch resume ID`, or a click on the account page, sends the token again from where it is kept.
- **Unattended work needs no token from us.** Repositories are pushed to the fleet, and backups are pulled by the customer's machine (§6). Functions that run code get only what the customer passes to that run.
- **Guidance.** `sparkpipe tokens set hf` recommends a fine-grained, read-only Hugging Face token limited to the repositories needed, with an expiry.

## 7. Functions

A Function is software we build and run for the account on a fleet it holds. It is started, or scheduled, through the API, the CLI or the account page, and it does its own setup. Results go to storage and to the morning report.

### 7.1 How they run

- **Runner.** A runner for each fleet orchestrates its runs. It calls the fleet's own inference with the account's identity and low priority. Two sequences always stay free for interactive requests, and interactive requests go first.
- **Sandbox.** Code that a Function builds or runs (tests, fuzzers, builds, an agent's shell commands) runs in the fleet's sandbox, never on the Sparks.
  - Each run gets a fresh microVM with a copy of the repository and no network unless the run allows it.
  - The VM is destroyed when the run ends, and the fleet's sandbox is wiped at a change of user.
  - The Sparks have about 11 GB of memory to spare with K3 loaded, and a CPU tenant slows decode by more than a third (§1). So every fleet's sandbox has its own reserved CPU, memory and disk off the Sparks, and shares nothing with another fleet's.
- **Our agent loop.** Agent Functions use our own agent loop, published with the engine, with read, search, edit and run tools against the sandbox. They never use Claude Code.
- **API.**
  - `POST /v1/functions/{name}/runs` takes parameters and returns a run id.
  - `GET /v1/runs/{id}` returns the state and results, and `GET /v1/runs/{id}/log` streams the log. `DELETE` cancels.
  - `PUT /v1/schedules {"function", "params", "cron", "tz"}` schedules runs. The default window for nightly runs is 19:00–09:00 in the account's time zone.
  - `GET /v1/reports/{date}` returns the morning report as JSON and Markdown.

### 7.2 Catalogue

| Function | What it does | Needs |
| --- | --- | --- |
| Ingest a repository | Clone and follow a repository, and map its files, symbols and history (§6) | Storage |
| Build and refresh the cache | Build and refresh packs, warm them, and pin them if asked (§5) | Packs, warm |
| Anticipate tomorrow | Build and warm the leaves the next day's sessions are most likely to need, within the customer's budgets, and report how they were used (§5.3) | Pack trees, warm |
| Prefill ahead | Warm named leaves now, and re-warm after a handover or release | Warm |
| Resume sessions | Re-warm the last prompts of recent sessions from the usage records, after a release or restart | Warm |
| Export and restore | §6 | Storage |
| Daily backup | §6: list the day's changes for the customer's own scheduled `sparkpipe storage sync` to pull | Storage |
| Rebuild after a loss | When a node pair loses part of the cache (clause 6.6), re-warm packs and recent sessions at background priority, and report what was rebuilt | Warm, loss report |
| Review the day's changes | For each change since the last run, review it against the pack and the files it touches, and report findings ranked by severity | Packs |
| Second opinions | Run the same review on a second model: one already resident, or one loaded for the night and unloaded before the morning (§4.5) | Resident models |
| Ask the whole codebase | Run one question or instruction over every file or module, and collect the answers | Packs |
| Keep docs current | Draft updates to architecture notes, API docs and a map of the code after each push, as a branch | Packs, repo push token |
| Morning report | Everything the night produced, with the week's prefill avoided (§3.6) | Runner |
| Benchmark your repository | For a demo day or any time: ingest the repository, replay a standard set of agent turns on it, and report first-token times warm and cold, decode speed and prefill avoided | Packs, warm |
| Scheduled model | Change the fleet's resident models on a schedule, for example a second model for the night, with the cache of each saved and restored (§4.5) | Model switch |
| Tests and fuzzing | Write and run tests and fuzzers for changed code, and report crashes with reproductions | Sandbox |
| Confirmed findings | Re-run review findings as tests in the sandbox, and keep only those that reproduce | Sandbox |
| Security audit | Memory safety, concurrency, injection, secrets and dependencies, with reproductions | Sandbox |
| Overnight agent | Work through a task list and leave one branch per task, with test results | Sandbox, agent loop |
| Codebase-wide change | A rename, an API migration or a port, done file by file, built and tested, as one branch | Sandbox, agent loop |
| Evaluate models on your code | Run the account's tests or eval set against offered models and its own | Sandbox, model switch |
| Train a draft model | Train the fleet's drafter on the account's repository and sessions, with an acceptance report against the current drafter | Training (§8) |
| Qualify your model | Compare a customer model's outputs with its reference outputs (§9) | Customer models |
| Train an adapter | Train an adapter for an offered model or a customer model | Adapter support in the engine |
| Make training data | Generate training data for a customer model, only from models whose licence allows it | License list |


## 8. Draft models

- **Today.**
  - The drafter is fixed per lane (`deployment/lanes.json` `draftd`), and a request can only turn speculation off with `"speculation": false` (`node/model_api.c:1099-1101`).
  - The K3 drafter runs where the fleet's drafts run, with 0.5 GB of memory to spare there (`docs/NIGHTLY_DRAFTER.md`, branch `drafter/nightly-retrain`).
- **API.** `GET /v1/drafters` lists the fleet's drafters, ours and the account's. `PUT /v1/drafters/active {"drafter": "acct-2027-01-12"}` loads one where that fleet's drafts run, replacing the current one.
- **Training.**
  - The branch measured one GB10 at 16 blocks/s, with 3.2M positions taking about 52 min on 16 nodes (derived).
  - It fits only with K3 swapped out, so a nightly training run costs a swap out and back (two switches of 16–79 s each) plus the run.
  - The Function measures the new drafter's acceptance against the current one on the account's own recent requests, keeps whichever accepts more, and reports both.

## 9. Customer models

The product promise is that a team can serve its own model on its own fleet, listed only for it. What a "driver" is depends on the architecture (`README.md:100-106`, `docs/MODULE_MAP.md`):

| Tier | The model | What the customer supplies |
| --- | --- | --- |
| A | Same architecture as an offered model, such as a fine-tune of Qwen, Gemma, GLM or K3 | Weights in storage (§6) and the base model's name. Our packer for that family builds the stage packs on the sandbox host. No code |
| B | A new model in a family we already run | A model contract, a module of about 200 lines and a packer of about 250 (K-EXAONE 2.0 on the Gemma family: 212 and 255 lines), from our guide for coding agents. We review it before it runs |
| C | A new architecture | A family stage of about 1,100 lines of CUDA (`gemma4_stage.cuh` is 1,173), written from our guide. We review it, then qualify it like any model |

- **API.** `POST /v1/custom-models {"name", "tier", "base", "path", "contract"}` packs and qualifies the model and adds it to the account's model list. `DELETE` removes it.
- **Isolation.** A customer model is working state. Clause 6.3 of the Agreement restores the fleet's software image before another customer's booking starts, which covers tier B and C code.


## 10. Codebase tools over MCP

The fleet serves an MCP server (Streamable HTTP) at `/mcp`, authenticated with the account's key. Claude Code, Codex, OpenCode and any MCP client can call tools that run next to the ingested repository, its indexes and the sandbox host. A search, a type check or a test run is answered on the fleet in milliseconds or seconds, where it would otherwise cost the agent several turns and the laptop several minutes. The tools' indexes, build caches and dependency mirror are working state (§4): kept while the account holds the fleet and deleted with it. `sparkpipe mcp` is a stdio bridge for clients that only start local servers.

| Tool | Does | Why it saves time |
| --- | --- | --- |
| `search_code` | ripgrep, plus a semantic index built at ingest, over a repository at any commit | Answers in milliseconds from the fleet's copy, with no files shipped back and forth |
| `search_history` | The commits, messages and diffs that touched a path or match a question ("when and why did this change?"), from the same semantic index | The agent finds why code is the way it is without walking `git log` by hand |
| `find_symbol`, `references` | Definitions and uses from a tree-sitter index built at ingest | Replaces several grep-and-read turns |
| `definition`, `hover`, `diagnostics`, `rename_preview` | Answers from language servers kept running for the repository at the account's working commit, with the project's own build settings: clangd, pyright, rust-analyzer, gopls and the TypeScript server, among others | Exact, type-aware answers and compile errors without a build, where grep can only guess |
| `deps` | The source and docs of the repository's dependencies at the versions its lockfiles pin (pip, npm, cargo, Go modules, Maven), fetched when the account turns the mirror on and searched like the repository | The agent reads the real API of the exact version instead of guessing from memory |
| `read_file` | A file or line range at a commit | |
| `pack_info` | Which files and commit a pack holds | The agent knows what is already in its context and does not read it again |
| `run` | A command in a fresh sandbox microVM on the repository at a commit, with an optional patch (the local `git diff`) applied; no network unless allowed. Build caches (ccache, sccache, pip, npm, cargo, Gradle) persist between runs for the account. With `tests: "affected"`, only the tests the change touches run, from a coverage map a nightly Function refreshes, split across the sandbox host's cores; flaky tests are reported apart | A test loop that takes minutes on a laptop comes back in seconds, while the laptop stays free |
| `browse` | A headless browser in the sandbox opens a page served by a `run`, or a URL the account allows, follows a short script and returns screenshots and the console and network errors | Checks a web app without the engineer's machine; the model reads the screenshots |
| `map` | Runs one instruction over many files or chunks at once (summarize, extract, classify, check) with a small model beside the fleet's main model, and returns the collected answers | Fan-out reading spends no turns and no decode time of the main model, which stays on the reasoning |
| `start_function`, `run_status`, `morning_report` | §7 | |
| `warm` | §4.2 | |

Setup is written by `sparkpipe setup` and the plugins (§3.8). For Claude Code that is the user-scope `mcpServers.sparkpipe` entry in `~/.claude.json` that `claude mcp add --scope user --transport http sparkpipe https://sparkpipe.ai/pay/api/mcp --header "Authorization: Bearer <key>"` would write; for Codex, `[mcp_servers.sparkpipe]` in `config.toml`; for OpenCode, a remote MCP entry in `opencode.json`. Other clients use the stdio bridge.

## 11. Batch jobs

`POST /v1/batches {"input": "storage path of a JSONL file" | [...], "format": "anthropic" | "openai_responses" | "openai_chat", "window": "now" | "tonight"}` runs many requests at background priority, using sequences that interactive use leaves free.
- Requests are ordered by shared prefix, so the cache serves the common part once.
- Results are written as JSONL to storage, with telemetry per line.
- `GET` reports progress, and `DELETE` cancels.

Uses: evals, bulk edits, test generation, data for a customer model.

## 12. Teams, keys and audit

- **Members.** The account owner invites members by email. Each signs in with their own email code.
  - **Roles:**
    - **owner:** bookings, payments and members;
    - **admin:** keys, Functions, models and storage;
    - **member:** their own keys, inference, and their own Function runs.
- **Keys** belong to a member.
  - Options: scopes (`inference`, `state`, `storage`, `functions`, `models`, `account:read`), `fleet`, `allow_swap`, expiry, and an IP allowlist.
  - `POST /v1/keys/{id}/rotate` issues a new key and keeps the old one valid for an hour.
- **Fair share.** When several members' requests are waiting, the chat layer shares sequences between members. No member takes more than three quarters of them while another is waiting.
  - The unit is the member, by the key each request comes with, never the session: the session and agent headers of §3.8 drive affinity and routing, not fair share.
- **Usage per member** in `/v1/requests` and `sparkpipe stats`.
- **Audit log.** `GET /v1/audit` lists events, without prompts or replies, kept like the operational records:
  - sign-ins, member and key changes;
  - bookings and waitlist locks;
  - model switches, wipes and storage access;
  - each fetch, with its source and size;
  - Function runs.

## 13. Notifications

`PUT /v1/webhooks {"url", "events", "secret"}` delivers signed JSON (HMAC-SHA256 in `X-SparkPipe-Signature`), Slack-compatible on request. The same events can go by email to chosen members.

Deliveries run in parallel and are retried when a receiver fails, so they can arrive out of order or more than once. A JSON delivery carries its notice's `id` and `created` time, and `X-SparkPipe-Delivery` repeats the `id`: order notices by `created` and drop repeats by `id`.

| Event | When |
| --- | --- |
| `booking.deadline` | 72 h, 48 h, 6 h and 2 h before 00:00 on the Sunday of the last booked week, while the next week is not booked; early when auto-extend cannot book (§16), and when it stops |
| `booking.confirmed`, `fleet.allocated`, `waitlist.position` | A booking, a waitlist allocation, a change in the account's own position |
| `run.finished`, `run.failed`, `report.ready` | §7 |
| `fetch.finished`, `fetch.failed`, `fetch.needs_token` | §6.1, §6.2 |
| `pack.warm`, `cache.restored` | A pack version is warm; the cache is back after a handover, a release or a restart |
| `models.changed` | A model loaded or unloaded on the fleet |
| `downtime.credited` | HC credited for downtime or maintenance |
| `fleet.failover` | A held fleet is down and requests go to a free fleet without working state (Agreement 15.7), and when that ends |
| `working_state.lost` | A node pair lost part of the working state: what was lost, the rebuild credit under clause 6.6 (24 hours of the fleet), and the rebuild run started |

## 14. Several fleets

- **Routing.** A key with `fleet: "any"` sends each request to the held fleet where its prefix is warm, by the prefix hash the chat layer already computes, and otherwise to the least busy one. A session stays on its fleet.
- **Bursts on idle fleets.** An account can book hours on any fleet that nobody holds, up to the end of the current week (Agreement 14.1), with `sparkpipe book hours` or `/market/...`.
  - With `fleet: "any"`, requests reach a burst fleet like a held one.
  - When the booking starts, the account's packs are warmed on that fleet in the background.
  - Uses: a big refactor, a parallel review, or an overnight batch on a second fleet.
- **Bursts on idle fleets.** An account can book hours on any fleet that nobody holds, up to the end of the current week (Agreement 14.1), with `sparkpipe book hours` or `/market/...`.
  - With `fleet: "any"`, requests reach a burst fleet like a held one.
  - When the booking starts, the account's packs are warmed on that fleet in the background.
  - Uses: a big refactor, a parallel review, or an overnight batch on a second fleet.
- **Packs** are warmed on every fleet that serves their alias.
- **Moving working state.** `POST /v1/state/move {"from": "F1A", "to": "F2A"}` copies the snapshot store and the storage between two held fleets of the same model build and topology over the facility network. A fleet change or a long repair then costs no cache.

## 15. CLI

`sparkpipe` is a single-file Python command with no dependencies outside the standard library. It is published on PyPI and in this repository (`tools/sparkpipe_cli.py`). Every command is one or a few API calls.

| Command | Does |
| --- | --- |
| `sparkpipe login` | Email code sign-in, then creates a key for this machine |
| `sparkpipe status` | Holdings, the fleet's model, cache use, pins, packs, runs, deadlines |
| `sparkpipe book week F3 4`, `book hours F1 2027-01-05T10 2`, `waitlist lock 144` | Bookings (§16). A booking shows the market's quote and sends back the quoted first week and price; if either has changed by then, the market books nothing and answers `409 quote_changed`, stating the new first week and price |
| `sparkpipe setup claude-code`, `setup codex`, `setup opencode`, `setup [<harness>] --undo` | Writes the client settings and installs the plugins of §3.8; `--undo` restores the keys setup owns (§3.8). With `--fleet F1A` (or `SPARKPIPE_FLEET`), every model and MCP request the harness sends names that fleet in `X-SparkPipe-Fleet`: `ANTHROPIC_CUSTOM_HEADERS` and the MCP entry's headers for Claude Code, `http_headers` in the provider and MCP tables for Codex, the provider's `options.headers` and the MCP headers for OpenCode. Without a fleet, setup removes a fleet an earlier run wrote. Codex and OpenCode read the key from `SPARKPIPE_API_KEY`, and setup prints `export SPARKPIPE_API_KEY="$(sparkpipe keys show)"` for the shell profile, never the key itself |
| `sparkpipe statusline` | The Claude Code status line of §3.8 |
| `sparkpipe models resident [names]`, `aliases` | §4.5 |
| `sparkpipe repo add NAME` (prints the push URL) | §6 |
| `sparkpipe pack create`, `pack refresh`, `pack ls`, `pin` | §5, §4.3 |
| `sparkpipe anticipate status`, `plan`, `approve`, `focus`, `report`, `off`, `auto` | §5.3, §5.6 |
| `sparkpipe warm FILE` | §4.2 |
| `sparkpipe fn ls`, `fn run NAME`, `fn schedule`, `fn logs ID`, `fn cancel ID` | §7 |
| `sparkpipe report [DATE]` | The morning report |
| `sparkpipe stats [--since 7d]` | Cached share, prefill avoided and decode speed, from §3.6 |
| `sparkpipe storage ls`, `put`, `get`, `rm`, `export`, `restore` | §6 |
| `sparkpipe fetch SOURCE PATH [--token NAME] [--then pack]`, `fetch ls`, `fetch resume ID`, `fetch cancel ID` | §6.1 |
| `sparkpipe tokens set NAME`, `tokens ls`, `tokens rm NAME` (in the local keychain only) | §6.2 |
| `sparkpipe storage sync PATH DEST [--delete]` | §6 daily backup. It keeps a cursor per API, account, fleet, path and destination, so one fleet's or account's sync never skips another's changes. Files deleted on the fleet are kept in `DEST` and remembered until a run with `--delete` removes them |
| `sparkpipe drafter ls`, `drafter use` | §8 |
| `sparkpipe models add`, `models rm` | §9 |
| `sparkpipe keys ls`, `keys create`, `keys revoke`, `keys show` | Keys, with `--fleet` and `--allow-swap`; `keys show` prints this machine's key for `$(sparkpipe keys show)` in a shell profile |
| `sparkpipe wipe [--storage]` | §4.4 |
| `sparkpipe proxy` | A local Anthropic- and OpenAI-compatible endpoint on `127.0.0.1` that adds the key and fleet, retries `429` with `retry-after`, and serves tools that cannot set headers. It forwards only the inference paths of §2, the alias table `GET /v1/models/aliases` that `sparkpipe setup` reads, and `/mcp`, including the `/v1/responses` WebSocket. Clients send a random key it prints at each start, and it refuses requests addressed to another host name or carrying an `Origin`, so other local users and web pages cannot borrow the account's key |
| `sparkpipe mcp` | The stdio MCP bridge (§10) |
| `sparkpipe batch submit FILE`, `batch ls`, `batch get ID` | §11 |
| `sparkpipe team invite EMAIL --role`, `team ls`, `audit` | §12 |
| `sparkpipe webhooks add URL --events` | §13 |
| `sparkpipe auto-extend F3 --weeks 4` | §16 |

## 16. Account and bookings

The account endpoints already exist in the payment service under `/market/...`: state, quote and book weeks, book hours, the waitlist, keys and the ledger (`docs/FLEET_BOOKINGS.md`). For the CLI they also accept `Authorization: Bearer <session>` from the email-code sign-in, besides the page's cookie. Keys gain the options of §12.

**Auto-extend.** `PUT /market/auto-extend {"fleet": "F1A", "weeks_ahead": 4}` keeps that many weeks booked ahead from the account's HC. It is on by default, one week ahead, from the first week an account books or the waitlist grants it on a fleet; `weeks_ahead: 0`, or one click on the account page, turns it off.
- It books each new week as soon as the rules allow, at the discount for the weeks held.
- If the balance runs short, it sends `booking.deadline` early, with the HC needed.

A held fleet is then never lost to a forgotten Sunday.

## 17. Limits and errors

| Limit | Value | Where it comes from |
| --- | --- | --- |
| Sequences at once | 8 (K3, GLM 5.3), 16 (GLM 5.3 Flash) | `PERFORMANCE_STATUS.md:40` |
| Sequences kept for interactive use while background work runs | 2 | §3.1, §7.1 |
| Body size | 32 MiB | §3.1 |
| Pins | Half the fleet's cache | §4.3 |
| Context | The model's served window (Fleet List) | |
| Storage | 4 TB per fleet | §6 |
| Gateway requests, account holding an active rental | `SPARKPIPE_GATEWAY_CAPACITY` / N per 10 minutes, N = Fleets with an active rental now (at least 1) | `throttle_gateway` in `tools/sparkpipe_pay.py`; capacity measured with `tools/gateway_capacity_check.py` |
| Gateway requests, any other account | 6,000 per 10 minutes | `LIMITS["gateway"]` in `tools/sparkpipe_pay.py` |

The gateway's limits:
- An account holds an active rental when one of its bookings' hours includes now. Such an account is not subject to the 6,000 limit. Instead, the holders of the N Fleets rented at that moment share the gateway's measured capacity equally, so each gets `SPARKPIPE_GATEWAY_CAPACITY / N` requests in any 10 minutes.
- `SPARKPIPE_GATEWAY_CAPACITY` is the number of requests the gateway sustains in 10 minutes. It is required: the gateway refuses to start without a whole number of at least 1, and there is no default.
- It is measured with `tools/gateway_capacity_check.py` against the sandbox or a staging gateway, never production: the tool refuses a URL unless its host is loopback or has a `sandbox` or `staging` DNS label, or its path starts with `/pay/sandbox/api/` with no dot segments; a query, fragment or user name never counts. Set it to the result's `capacity_per_window`, which is the sustained 2xx requests per second over a 60-second run times 600 (docs/PAYMENTS.md).
- Both limits answer 429 `busy` with `Retry-After: 60` in each shape's error body.

Errors use each API shape's own error body, with these `type` or `code` values:
- `fleet_required`, `fleet_not_held`, `fleet_down`, `model_not_loaded`;
- `image_not_supported`, `context_too_long`, `busy`;
- `request_cancelled`, `request_not_found`;
- `invalid_request`, `invalid_key`, `upstream_error`.

`context_too_long` is sent in each shape's native form, so the harnesses compact on their own: on Messages, 400 `invalid_request_error` with the message `prompt is too long: N tokens > M maximum`; on Responses and chat completions, 400 with code `context_length_exceeded`.

`busy` is sent in the native form too, so the harnesses retry on their own: on Messages, 429 `rate_limit_error`; on Responses, 429 with type `requests` and code `rate_limit_exceeded`; on chat completions, 429 with code `busy`. Each carries `retry-after` and `x-should-retry: true` (§3.1).

## 18. Build order and acceptance tests

Owners are components, so the work splits into parallel sessions with no shared files. `docs/FLEET_API_ISSUES.md` breaks it into issues with sizes, dependencies and done-when tests.

| # | Component | Work | Depends on |
| --- | --- | --- | --- |
| 1 | Engine and working state | Merge #1691 and #1692. Prove the snapshot store at the full per-node budget. Keep the prefix index across a release and a restart. Bound K3's recurrent store. Commit-only requests. Pins. Streamed prefill that publishes its blocks | — |
| 2 | Chat layer (hub) | Messages and Responses translation, token-exact replay (§3.2), alias table and residency rules, telemetry fields, `count_tokens`, keep-alives | Engine: commit-only and pins for §4 |
| 3 | Gateway (VPS) | Pass the new paths through, `X-Api-Key`, 32 MiB bodies, key options, the usage-record store and `/v1/requests`, bearer sessions for the CLI | — |
| 4 | Packs, repositories, storage | Repository ingest, the pack builder, versioned aliases with session affinity, refresh and warm, storage and fetch | 1, 2 |
| 5 | Function runner | The runner, the Functions that only read code, the morning report | 2, 4 |
| 6 | Sandbox | Each fleet's reserved sandbox, microVMs, the agent loop, the Functions that run code | 5 |
| 7 | CLI | `tools/sparkpipe_cli.py` and its tests | 3; grows with each component |
| 8 | Drafters | Productize `drafter/nightly-retrain` with the acceptance gate (§8) | 5 |
| 9 | Customer models | A packer per family (tier A), the driver guide and review (tiers B and C) | 6 |

Acceptance tests, each recorded in `PERFORMANCE_STATUS.md`, all passed before 4 January 2027:

- **Gate A (a held week of Claude Code).** A 300-turn Claude Code session on K3 through `/v1/messages`, with tool use and two pauses of 2 and 14 hours.
  - After the first turn, at least 95% of prompt tokens come from the cache.
  - The first token after each pause comes in under 6 s at 150K tokens.
  - There is no failed request and no unrequested model swap.
  - The cache survives a hub release and one node reboot during the session.
- **Gate B (warm-up).** A warmed prefix of 150K tokens is fully matched by the next real request, and the warm-up never delays an interactive request by more than one decode step.
- **Gate C (big prompts).** A 1M-token prompt prefilled once is answered from the cache by a follow-up, with the first token under 10 s.
- **Gate D (packs).** On the Customer Zero repository, the same task list with and without a 300K-token pack. Report turns, tokens read through tools, wall time and decode speed, for the Customer Zero report.
- **Gate D2 (the anticipator).** On the Customer Zero repository over 10 working days, in `auto` mode with default budgets:
  - at least 60% of sessions start on an anticipated leaf;
  - in those sessions, at least 90% of first-turn prompt tokens come from the cache;
  - nightly compute stays within budget, and no interactive request waits for anticipation.

  The report also gives built-but-unused hours.
- **Gate E (a night of Functions).** On the Customer Zero repository, one night runs every Function in §7.2 on its schedule.
  - Every run finishes.
  - Interactive first-token time during the night stays within 10% of an idle fleet's.
  - The morning report is ready by 08:30.
- **Gate A also covers losing a pair:** one node pair fails during the session. The loss is reported, the 24-hour credit appears within an hour, and the rebuild Function restores the session's pack and conversation before its next turn.
- **Gate F (customer models).** A tier A fine-tune and a tier B model each go from upload to the account's model list, pass qualification, and serve through all three API shapes.
