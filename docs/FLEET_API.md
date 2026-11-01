# Fleet API

This is the specification of the API and CLI a customer uses on a fleet it holds, as it ships on 4 January 2027. Every feature removes waiting or repeated work from an engineer's day. Section 1 says, for each saving, where the engine starts today and the work that delivers it. Section 18 assigns that work and lists the acceptance tests that close it.

The product rules (who holds a fleet, bookings, working-state deletion) are in the Customer Agreement and `docs/FLEET_BOOKINGS.md`. The engine side of working state is `docs/WORKING_STATE.md` (PR #1691, with the hub in PR #1692). This file is the customer-facing contract built on top of them.

## 1. Where the time goes, and what removes it

An engineer working with a coding agent waits for three things: prefill of context the model has seen before, generation, and work they could have left running overnight. Per-token APIs make the first one recur, because their caches expire after 5 minutes by default. A held fleet removes all three. The table lists the engine work that makes each saving hold.

| Saving | Mechanism | Starting point on `main` | Work | Acceptance test |
| --- | --- | --- | --- | --- |
| No re-prefill after a break | Automatic prefix cache, kept while the account holds the fleet | Prefix cache exists: hash-chained 64-token blocks, LRU (`cache/prefix_cache.c`). Retention is 24 h from last use (`spark_kv_snapshot.h:27`) | Held-hours retention and the wipe at a change of user (#1691, #1692). Snapshot store proven at the full per-node budget; today only 8 GiB per rank is proven (`TECHDEBT.md:579-599`). Prefix index that survives a release and a restart (`FLEET_RELEASE_RUNBOOK.md:298` drops it today) | Gate A |
| Every turn resumes where the last one ended | Assistant turns replayed token-exact, so the next request matches up to the end of the previous reply | The chat layer keeps reply text byte-exact for chat (`LITELLM_FRONTEND.md:33-37`). Kimi K3 resumes only at 16,384-token checkpoints or at the end of a reply (`CHAT_MEMORY.md:22-33`) | The same guarantee for Anthropic Messages and Responses, including tool calls and reasoning (§3.2) | Gate A |
| Long agent sessions never fail | Bounded recurrent-state store | A K3 conversation fails after about 977 checkpointed requests (`PERFORMANCE_STATUS.md:450-452`) | Evict the oldest checkpoints instead of failing | Gate A |
| No accidental cold fleet | One model per fleet, switched only on purpose | "The last requested model wins" (`LITELLM_FRONTEND.md:132`). A switch takes 16–46 s, at worst 79 s (`PERFORMANCE_STATUS.md:174-202`). The cache is saved and restored: 137,600 tokens back in 2.7 s | Alias table and explicit switches (§4.5), so a background "haiku" call never swaps the fleet | Gate A |
| Big prompts paid for once | Long prefill publishes its cache | The fast streamed 1M prefill (16.2 min) publishes no cache blocks and stops decode meanwhile (`TECHDEBT.md:308-326`). A reusable 1M prefill takes about 76 min (`PERFORMANCE_STATUS.md:68`) | Streamed prefill that publishes its blocks to the prefix cache and snapshot store | Gate C |
| The codebase is already in context | Context packs (§5): a repository snapshot placed first in every prompt, kept warm | None | Pack builder, versioned aliases, and refresh in idle time | Gate D |
| A session starts warm | Prefill-only warm-up of an exact future prefix, and pins that protect it | `max_tokens` must be 1 or more (`node/model_api.c:1622-1631`); `max_tokens` 1 commits a prompt (`CHAT_MEMORY.md:38`). No pinning | `POST /v1/cache/warm` and pins (§4) | Gate B |
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
| Working state | `GET /v1/state`, `POST /v1/cache/warm`, `GET/POST/DELETE /v1/cache/pins`, `DELETE /v1/cache`, `GET/PUT /v1/model`, `GET/PUT /v1/model/aliases` |
| Usage records | `GET /v1/requests` |
| Context packs | `GET/POST /v1/packs`, `GET/DELETE /v1/packs/{name}`, `POST /v1/packs/{name}/refresh` |
| Repositories | `GET/POST /v1/repos`, `DELETE /v1/repos/{name}`, `POST /v1/repos/{name}/sync`, `POST /v1/repos/{name}/hook` |
| Storage | `GET /v1/storage`, `PUT/GET/DELETE /v1/storage/{path}`, `POST /v1/storage/fetch`, `POST /v1/export`, `POST /v1/restore` |
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
  - Otherwise it is the `X-SparkPipe-Fleet` header, or the existing body field `"sparkpipe": {"fleet": "F3"}` on chat completions.
  - Otherwise, if the account holds exactly one fleet, that one.
  - Holding several fleets and naming none answers 409 `fleet_required`.
- **Which model.** `model` names an offered model (`kimi-k3`), a pack alias (`kimi-k3+core`, §5) or an alias from the fleet's alias table (§4.5). A request for a model that is not loaded, from a key without `allow_swap`, answers 409 `model_not_loaded` and names the loaded model. It never swaps the fleet by accident.
- **Size.** Request bodies up to 32 MiB. The gateway's cap rises from 8 MiB (`tools/sparkpipe_pay.py`), because a 1M-token conversation is about 4 MB of text before images.
- **Busy.** When all sequences are in use, the request waits up to 60 s for one, then answers 429 with an integer `retry-after` and `x-should-retry: true`. During long prefills, streams send keep-alive events every 10 s: `ping` on Messages and an SSE comment line on the others. Clients with idle timeouts therefore stay connected.
- **Path.** gateway (VPS) → SSH tunnel → the hub's chat layer (`tools/serving/chat_frontend.py`) → the model's `sparkpipe_model_api` → the 16 ranks. All three new API shapes are translated in the chat layer, because the tokenizer, chat templates and tool-call parsers live there (`chat_frontend.py:525`). The gateway passes the new paths through unchanged, with `X-SparkPipe-Account` and `X-SparkPipe-Fleet`.

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
- **Other fields.** `max_tokens`, `temperature`, `top_p`, `top_k`, `stop_sequences` and `stream` work. `metadata` and unknown fields are accepted and ignored, and `anthropic-beta` headers are ignored.
- **Response.** `id`, `content` blocks, and `stop_reason`: `end_turn`, `tool_use`, `max_tokens` or `stop_sequence`.
  - **Usage:**
    - `cache_read_input_tokens` is the prompt tokens matched in the cache;
    - `cache_creation_input_tokens` is the prompt tokens newly committed;
    - `input_tokens` is the rest;
    - `output_tokens` is what was generated.

    The sum of the three input fields is always the prompt length.
- **Stream.** `message_start`, `content_block_start/delta/stop` (`text_delta`, `input_json_delta`, `thinking_delta`, `signature_delta`), `message_delta` with cumulative usage, `message_stop`, `ping`, and `error`. The stream never ends before `message_delta`.
- **Errors.** Errors use Anthropic's body, `{"type":"error","error":{"type","message"}}`. 429 and 529 carry `retry-after`.

`POST /v1/messages/count_tokens` returns the exact count, using the model's tokenizer and template. `HEAD /api/hello` answers 200. `GET /v1/models` lists the loaded model, the other offered models, the account's pack aliases and its custom models.

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
```

- **The Haiku alias points at the same model,** because Claude Code's background calls would otherwise ask for another model. The alias table catches any `claude-*` name as well.
- **The context limit is the model's served window.** Claude Code otherwise assumes 200K.
- **The attribution header is turned off,** so the first system block stays the same from session to session.
- **Every Claude Code release is tested** against this endpoint before we tell customers to upgrade, because Anthropic does not test Claude Code against other models ([LLM gateway docs](https://code.claude.com/docs/en/llm-gateway.md)).
- **Claude Code's terms keep it on the customer's own machine.** Functions never run Claude Code on our servers (§7).

### 3.4 `POST /v1/responses` (Codex)

OpenAI Responses, which is the only wire format Codex supports for custom providers.

- **Request fields:**
  - `input` as a string or items: `message`, `function_call`, `function_call_output` and `reasoning`;
  - `instructions` and function `tools`;
  - `tool_choice`, `parallel_tool_calls`, `max_output_tokens`, `temperature`, `top_p`, `stream`, `store` and `previous_response_id`.
- **Previous responses.** `previous_response_id` works for responses stored as working state, which is the default here. `store: false` stores no response for later reference, but the replay store of §3.2 still keeps the generated tokens. `prompt_cache_key` is accepted and ignored.
- **Stream events:**
  - `response.created` and `response.in_progress`;
  - `response.output_item.added` and `response.content_part.added`;
  - `response.output_text.delta` and `response.output_text.done`;
  - `response.function_call_arguments.delta` and `response.function_call_arguments.done`;
  - `response.reasoning_summary_text.delta`, where the model reasons;
  - `response.output_item.done`, `response.completed` and `error`.
- **Usage:** `input_tokens`, `input_tokens_details.cached_tokens`, `output_tokens` and `output_tokens_details.reasoning_tokens`.

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
  "request": "r_7Q2…", "fleet": "F3", "model": "kimi-k3", "pack": "core@a41f09",
  "prompt_tokens": 182340, "cached_tokens": 181760, "prefill_tokens": 580,
  "queue_ms": 0, "prefill_ms": 1240, "ttft_ms": 1310,
  "output_tokens": 912, "decode_tps": 15.8, "draft_accept": 0.71,
  "cold_prefill_ms_estimate": 312000
}
```

`cold_prefill_ms_estimate` is the measured prefill rate for this model at this length, applied to the tokens that came from the cache: the wait the cache removed. `GET /v1/requests?since=&until=` returns these records, without prompts or replies; they are the operational records the privacy policy already lists, kept for a year. `sparkpipe stats` sums them into "prefill avoided this week", which is the number the Customer Zero report publishes.

### 3.7 More inference features

- **Structured outputs.** `response_format: {"type": "json_schema", ...}` on chat completions, `text.format` on Responses, and forced tool use on Messages always produce output that matches the schema. The engine's sampler masks tokens with a grammar compiled from the schema. Today `response_format` other than text answers 400 (`LITELLM_FRONTEND.md:59-62`).
- **Parallel samples.** `n` up to 8 on chat completions and `n` in the body of Messages and Responses. The samples share the prompt's cache and run as one batch: best-of-N patches cost one prefill. Today `n` answers 400.
- **Log probabilities.** `logprobs` and `top_logprobs` up to 20, for evals and confidence checks.
- **Tokens.** `POST /v1/tokenize` and `POST /v1/detokenize` with the model's tokenizer and chat template, so tools can count and place context exactly.
- **Cancellation.** A client disconnect, or `DELETE /v1/requests/{id}`, stops generation within one decode step and frees the sequence. Agents interrupt often; an abandoned generation must not hold a sequence.
- **Priority.** `X-SparkPipe-Priority: interactive | background` (and a default per key). Background requests yield sequences and prefill to interactive ones, and Functions and batches always run as background. The chat layer passes priority to `sparkpipe_model_api`; LiteLLM drops it today (`TECHDEBT.md:1121`).
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

**Claude Code** (Anthropic Messages, §3.3)
- **Setup.** `sparkpipe setup claude-code` writes the environment of §3.3, with three additions:
  - `CLAUDE_CODE_SUBAGENT_MODEL` set to the same model, so subagents never swap the fleet;
  - `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1`;
  - a `modelOverrides` entry, so `/model` lists our models.
- **Priority from hint headers.** With hint headers on, Claude Code marks each request as main, subagent, compaction or auxiliary. Compaction and auxiliary requests run at background priority, and main requests at interactive priority.
- **Affinity.** `x-claude-code-session-id` and `x-claude-code-agent-id` drive pack-version affinity (§5), routing between fleets (§14) and fair share (§12).
- **The SparkPipe plugin for Claude Code**, from our plugin marketplace on GitHub, bundles:
  - the MCP server of §10;
  - slash commands: `/sparkpipe:status`, `/sparkpipe:report`, `/sparkpipe:stats`, `/sparkpipe:run` (tests on the sandbox host with the local diff) and `/sparkpipe:fn`;
  - a `SessionStart` hook that warms the session's pack after a resume, a compaction or a long pause (§4.2);
  - a skill that tells the agent when to use the fleet's `run`, `search_code` and `pack_info` instead of local tools.
- **Status line.** `sparkpipe statusline` shows the fleet, the loaded model, the cached share of the last turn and the prefill avoided today.

**Codex** (OpenAI Responses, §3.4)
- **Setup.** `sparkpipe setup codex` writes a `[profiles.sparkpipe]` profile and the provider of §3.4, with:
  - `model_context_window` and `model_auto_compact_token_limit` set from the model;
  - `model_catalog_url` pointing at `GET /v1/codex/models`, a Codex-native catalogue of our models with context windows and reasoning settings;
  - `agents.default_subagent_model` set to the same model;
  - `agents.max_concurrent_threads_per_session` set to fit the fleet's sequences;
  - `[mcp_servers.sparkpipe]` for §10.
- **Affinity.** Codex's session-derived `prompt_cache_key` drives pack-version affinity and routing.
- **WebSocket transport.** `/v1/responses` also accepts Codex's optional WebSocket transport (`supports_websockets`), which removes a connection set-up per turn.

**OpenCode** (Anthropic Messages through its `@ai-sdk/anthropic` provider, so tools and reasoning keep full fidelity)
- **Setup.** `sparkpipe setup opencode` writes `opencode.json`:
  - the provider, with `baseURL` and `{env:SPARKPIPE_API_KEY}`;
  - `limit.context` and `limit.output` per model;
  - the subagent model set to the same model;
  - the remote MCP server of §10;
  - `prune` off, because pruning rewrites history and misses the cache;
  - `"share": "disabled"`, so sessions are never uploaded outside Korea.
- **The SparkPipe plugin for OpenCode** does what the Claude Code plugin does: it warms on session start, reports status and adds the fleet commands.
- **models.dev.** Our models and their served context windows are published there, so OpenCode knows them without any manual setting.

## 4. Working state

### 4.1 `GET /v1/state`

Returns:
- the loaded model and the alias table;
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
- **Build.** The engine gets a commit-only request flag. Until then the chat layer sends `max_tokens` 1 and discards the token, as chat memory does (`tools/serving/chat_memory.py:1358-1365`).

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

### 4.5 Model and aliases

- **`GET /v1/model`** returns the loaded model.
- **`PUT /v1/model {"model": "glm-5.3"}`** switches the fleet on purpose. The cache of the old model is saved to the snapshot store and comes back when that model does. This needs a key with `allow_swap`, or the account session.
- **Aliases.**
  - `PUT /v1/model/aliases {"claude-*": "kimi-k3+core", "gpt-*": "kimi-k3+core"}` maps names that clients send by themselves.
  - The default table maps `claude-*` and `gpt-*` to the loaded model.

## 5. Context packs

A pack is a snapshot of a repository rendered into one deterministic text block. The fleet places it first in every prompt that names the pack's alias: before the client's own system prompt. Every session, subagent and Function then shares one warm prefix that already holds the code. The agent reads less and searches less, and none of it is prefilled twice.

- **Build.** `POST /v1/packs {"name": "core", "repo": "main", "include": ["src/**", "include/**"], "exclude": [...], "budget_tokens": 400000}`.
  - Files come from an ingested repository (§6). They are ordered by how rarely they changed in the last 90 days, least-changed first, with ties broken by path.
  - Each file is rendered as `<file path="…" commit="…">…</file>` after a header that names the repository, the commit, and the instruction to read a file before editing it.
  - Binary and generated files and anything over the budget are left out and listed in the response.
- **Alias.** `<model>+<pack>`, for example `kimi-k3+core`.
- **Refresh.**
  - `POST /v1/packs/{name}/refresh`, or automatically after a push (§6), builds a new version.
  - Only the part after the first changed file has to be prefilled again. For K3 that means everything after the last 16,384-token checkpoint before it, which is why rarely changed files come first.
  - The new version is prefilled at low priority, and the alias moves to it only once it is warm. No request ever waits for a rebuild.
- **Session affinity.** A conversation that started on version A keeps version A, matched by its prompt prefix and by `x-claude-code-session-id` when present, because switching it would invalidate the conversation's whole cache. New sessions get the newest warm version. Old versions are dropped when no session has used them for 24 hours.
- **Cost.** A pack makes every prompt longer, so decode slows a little: K3 runs at 16.2 tok/s at 8K, 15.1 at 73K and 12.0–12.4 at 1M (`PERFORMANCE_STATUS.md:36-38,65-81`). The budget is the customer's choice. Gate D records what a pack saves on a real task list, for the Customer Zero report.

## 6. Storage and repositories

Customer storage is on the fleet's nodes: 500 GB per node, mirrored to the pair partner, 4 TB in all, in the scratch area of `docs/WORKING_STATE.md`. It follows the working-state deletion rule.

- **Small files.** `PUT /v1/storage/{path}` (with `Content-Range` for resumable chunks of up to 32 MiB), `GET`, `DELETE`, and `GET /v1/storage?prefix=` with sizes and totals.
- **Large files.** `POST /v1/storage/fetch {"url": "https://…", "path": "models/x/", "headers": {...}}` has the fleet download straight from an HTTPS URL, a presigned bucket URL or a Hugging Face repository over the facility's line. Model weights never pass through the public gateway.
- **Repositories.**
  - `POST /v1/repos {"name": "main", "url": "https://github.com/acme/app.git", "token": "…", "branch": "main"}` clones into storage. The token is kept as working state and used only to pull.
  - `POST /v1/repos/{name}/sync` pulls now.
  - `POST /v1/repos/{name}/hook` returns a webhook URL for push events, which triggers a sync and then the refreshes the account has scheduled.
- **Export and restore.**
  - `POST /v1/export {"paths": [...], "include": ["drafter", "packs"]}` writes an archive to storage, for download or for fetching from another fleet.
  - `POST /v1/restore {"url" | "path"}` brings one back.
  - Cache snapshots are not exported: they only work on the same model build and topology, and rebuilding them is a warm-up.
- **Daily backup.** `PUT /v1/backups {"paths": [...], "destination": {"type": "s3" | "https" | "sftp", ...}, "at": "03:00", "tz": "Asia/Seoul"}` copies the chosen storage paths to the customer's own bucket or server every day, sending only what changed, and reports each run in the morning report.
  - Clause 6.3 of the Agreement tells customers to treat working state as temporary, even on a fleet held many weeks ahead, and recommends a daily export. This makes that one setting.

## 7. Functions

A Function is software we build and run for the account on a fleet it holds. It is started, or scheduled, through the API, the CLI or the account page, and it does its own setup. Results go to storage and to the morning report.

### 7.1 How they run

- **Runner.** A per-fleet runner on the fleet's hub orchestrates runs. It calls the fleet's own inference with the account's identity and low priority. Two sequences always stay free for interactive requests, and interactive requests go first.
- **Sandbox host.** Code that a Function builds or runs (tests, fuzzers, builds, an agent's shell commands) runs on a sandbox host attached to the fleet, never on the Sparks.
  - Each run gets a fresh microVM with a copy of the repository and no network unless the run allows it.
  - The VM is destroyed when the run ends, and the host is wiped at a change of user.
  - The Sparks have about 11 GB of memory to spare with K3 loaded, and a CPU tenant slows decode by more than a third (§1), so every fleet comes with its own sandbox host.
- **Our agent loop.** Agent Functions use our own open-source agent loop with read, search, edit and run tools against the sandbox. They never use Claude Code.
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
| Prefill ahead | Warm the named branches' packs, and re-warm after a handover or release | Warm |
| Resume sessions | Re-warm the last prompts of recent sessions from the usage records, after a release or restart | Warm |
| Export and restore | §6 | Storage |
| Daily backup | §6: push chosen paths to the customer's own destination every day | Storage |
| Rebuild after a loss | When a node pair loses part of the cache (clause 6.6), re-warm packs and recent sessions at background priority, and report what was rebuilt | Warm, loss report |
| Review the day's changes | For each change since the last run, review it against the pack and the files it touches, and report findings ranked by severity | Packs |
| Second opinions | Run the same review on a second model at night, then switch back before the morning (§4.5) | Model switch |
| Ask the whole codebase | Run one question or instruction over every file or module, and collect the answers | Packs |
| Keep docs current | Draft updates to architecture notes, API docs and a map of the code after each push, as a branch | Packs, repo push token |
| Morning report | Everything the night produced, with the week's prefill avoided (§3.6) | Runner |
| Benchmark your repository | For a demo hour or any time: ingest the repository, replay a standard set of agent turns on it, and report first-token times warm and cold, decode speed and prefill avoided | Packs, warm |
| Scheduled model | Switch the fleet's model on a schedule, for example a second model for the night, with the cache of each saved and restored (§4.5) | Model switch |
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
  - The K3 drafter runs on the hub's RTX 5090, which has 0.5 GB to spare (`docs/NIGHTLY_DRAFTER.md`, branch `drafter/nightly-retrain`).
- **API.** `GET /v1/drafters` lists the fleet's drafters, ours and the account's. `PUT /v1/drafters/active {"drafter": "acct-2027-01-12"}` loads one on that fleet's hub GPU, replacing the current one.
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

The fleet serves an MCP server (Streamable HTTP) at `/mcp`, authenticated with the account's key. Claude Code, Codex, OpenCode and any MCP client can call tools that run next to the ingested repository and the sandbox host. `sparkpipe mcp` is a stdio bridge for clients that only start local servers.

| Tool | Does | Why it saves time |
| --- | --- | --- |
| `search_code` | ripgrep, plus a semantic index built at ingest, over a repository at any commit | Answers in milliseconds from the fleet's copy, with no files shipped back and forth |
| `find_symbol`, `references` | Definitions and uses from a tree-sitter index built at ingest | Replaces several grep-and-read turns |
| `read_file` | A file or line range at a commit | |
| `pack_info` | Which files and commit a pack holds | The agent knows what is already in its context and does not read it again |
| `run` | A command in a fresh sandbox microVM on the repository at a commit, with an optional patch (the local `git diff`) applied; no network unless allowed | Builds and test suites run on the fleet's sandbox host while the laptop stays free |
| `start_function`, `run_status`, `morning_report` | §7 | |
| `warm` | §4.2 | |

Setup is written by `sparkpipe setup` and the plugins (§3.8). For Claude Code that is `claude mcp add --transport http sparkpipe https://sparkpipe.ai/pay/api/mcp --header "Authorization: Bearer <key>"`; for Codex, `[mcp_servers.sparkpipe]` in its profile; for OpenCode, a remote MCP entry in `opencode.json`. Other clients use the stdio bridge.

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
- **Usage per member** in `/v1/requests` and `sparkpipe stats`.
- **Audit log.** `GET /v1/audit` lists events, without prompts or replies, kept like the operational records:
  - sign-ins, member and key changes;
  - bookings and waitlist locks;
  - model switches, wipes and storage access;
  - Function runs.

## 13. Notifications

`PUT /v1/webhooks {"url", "events", "secret"}` delivers signed JSON (HMAC-SHA256 in `X-SparkPipe-Signature`), Slack-compatible on request. The same events can go by email to chosen members.

| Event | When |
| --- | --- |
| `booking.deadline` | 48 h and 6 h before 00:00 on the Sunday of the last booked week, while the next week is not booked |
| `booking.confirmed`, `fleet.allocated`, `waitlist.position` | A booking, a waitlist allocation, a change in the account's own position |
| `run.finished`, `run.failed`, `report.ready` | §7 |
| `pack.warm`, `cache.restored` | A pack version is warm; the cache is back after a handover, a release or a restart |
| `model.switched` | Any model switch on the fleet |
| `downtime.credited` | HC credited for downtime or maintenance |
| `working_state.lost` | A node pair lost part of the working state: what was lost, the rebuild credit under clause 6.6 (24 hours of the fleet), and the rebuild run started |

## 14. Several fleets

- **Routing.** A key with `fleet: "any"` sends each request to the held fleet where its prefix is warm, by the prefix hash the chat layer already computes, and otherwise to the least busy one. A session stays on its fleet.
- **Packs** are warmed on every fleet that serves their alias.
- **Moving working state.** `POST /v1/state/move {"from": "F3", "to": "F4"}` copies the snapshot store and the storage between two held fleets of the same model build and topology over the facility network. A fleet change or a long repair then costs no cache.

## 15. CLI

`sparkpipe` is a single-file Python command with no dependencies outside the standard library. It is published on PyPI and in this repository (`tools/sparkpipe_cli.py`). Every command is one or a few API calls.

| Command | Does |
| --- | --- |
| `sparkpipe login` | Email code sign-in, then creates a key for this machine |
| `sparkpipe status` | Holdings, the fleet's model, cache use, pins, packs, runs, deadlines |
| `sparkpipe book week F3 4`, `book hours F1 2027-01-05T10 2`, `waitlist lock 144` | Bookings (§16) |
| `sparkpipe setup claude-code`, `setup codex`, `setup opencode` | Writes the client settings and installs the plugins of §3.8 |
| `sparkpipe statusline` | The Claude Code status line of §3.8 |
| `sparkpipe model [name]`, `aliases` | §4.5 |
| `sparkpipe repo add`, `repo sync`, `repo hook` | §6 |
| `sparkpipe pack create`, `pack refresh`, `pack ls`, `pin` | §5, §4.3 |
| `sparkpipe warm FILE` | §4.2 |
| `sparkpipe fn ls`, `fn run NAME`, `fn schedule`, `fn logs ID`, `fn cancel ID` | §7 |
| `sparkpipe report [DATE]` | The morning report |
| `sparkpipe stats [--since 7d]` | Cached share, prefill avoided and decode speed, from §3.6 |
| `sparkpipe storage ls`, `put`, `get`, `rm`, `fetch`, `export`, `restore` | §6 |
| `sparkpipe drafter ls`, `drafter use` | §8 |
| `sparkpipe models add`, `models rm` | §9 |
| `sparkpipe keys ls`, `keys create`, `keys revoke` | Keys, with `--fleet` and `--allow-swap` |
| `sparkpipe wipe [--storage]` | §4.4 |
| `sparkpipe proxy` | A local Anthropic- and OpenAI-compatible endpoint on `localhost` that adds the key and fleet, retries `429` with `retry-after`, and serves tools that cannot set headers |
| `sparkpipe mcp` | The stdio MCP bridge (§10) |
| `sparkpipe batch submit FILE`, `batch ls`, `batch get ID` | §11 |
| `sparkpipe team invite EMAIL --role`, `team ls`, `audit` | §12 |
| `sparkpipe webhooks add URL --events` | §13 |
| `sparkpipe auto-extend F3 --weeks 4` | §16 |

## 16. Account and bookings

The account endpoints already exist in the payment service under `/market/...`: state, quote and book weeks, book hours, the waitlist, keys and the ledger (`docs/FLEET_BOOKINGS.md`). For the CLI they also accept `Authorization: Bearer <session>` from the email-code sign-in, besides the page's cookie. Keys gain the options of §12.

**Auto-extend.** `PUT /market/auto-extend {"fleet": "F3", "weeks_ahead": 4}` keeps that many weeks booked ahead from the account's HC.
- It books each new week as soon as the rules allow, at the discount for the weeks held.
- If the balance runs short, it sends `booking.deadline` early, with the HC needed.

A held fleet is then never lost to a forgotten Sunday.

## 17. Limits and errors

| Limit | Value | Where it comes from |
| --- | --- | --- |
| Sequences at once | 8 (K3, GLM 5.3), 16 (GLM 5.3 Flash) | `PERFORMANCE_STATUS.md:40` |
| Sequences kept for interactive use while Functions run | 2 | §7.1 |
| Body size | 32 MiB | §3.1 |
| Pins | Half the fleet's cache | §4.3 |
| Context | The model's served window (Fleet List) | |
| Storage | 4 TB per fleet | §6 |

Errors use each API shape's own error body, with these `type` or `code` values:
- `fleet_required`, `fleet_not_held`, `model_not_loaded`;
- `image_not_supported`, `context_too_long`, `busy`;
- `invalid_request`, `invalid_key`, `upstream_error`.

## 18. Build order and acceptance tests

Owners are components, so the work splits into parallel sessions with no shared files. `docs/FLEET_API_ISSUES.md` breaks it into issues with sizes, dependencies and done-when tests.

| # | Component | Work | Depends on |
| --- | --- | --- | --- |
| 1 | Engine and working state | Merge #1691 and #1692. Prove the snapshot store at the full per-node budget. Keep the prefix index across a release and a restart. Bound K3's recurrent store. Commit-only requests. Pins. Streamed prefill that publishes its blocks | — |
| 2 | Chat layer (hub) | Messages and Responses translation, token-exact replay (§3.2), alias table and explicit switches, telemetry fields, `count_tokens`, keep-alives | Engine: commit-only and pins for §4 |
| 3 | Gateway (VPS) | Pass the new paths through, `X-Api-Key`, 32 MiB bodies, key options, the usage-record store and `/v1/requests`, bearer sessions for the CLI | — |
| 4 | Packs, repositories, storage | Repository ingest, the pack builder, versioned aliases with session affinity, refresh and warm, storage and fetch | 1, 2 |
| 5 | Function runner | The runner, the Functions that only read code, the morning report | 2, 4 |
| 6 | Sandbox host | Hardware per fleet, microVMs, the agent loop, the Functions that run code | 5 |
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
- **Gate E (a night of Functions).** On the Customer Zero repository, one night runs every Function in §7.2 on its schedule.
  - Every run finishes.
  - Interactive first-token time during the night stays within 10% of an idle fleet's.
  - The morning report is ready by 08:30.
- **Gate A also covers losing a pair:** one node pair fails during the session. The loss is reported, the 24-hour credit appears within an hour, and the rebuild Function restores the session's pack and conversation before its next turn.
- **Gate F (customer models).** A tier A fine-tune and a tier B model each go from upload to the account's model list, pass qualification, and serve through all three API shapes.
