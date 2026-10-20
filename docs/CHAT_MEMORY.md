# Chat memory: a model-curated ledger with a shared prefix

Status: design, revision 2 (2026-10-14), after an adversarial review against
the chat layer, the templates of all 16 models, Open WebUI v0.12.0 and the
engine. Phases 1 (prefix mechanics) and 2 (quality) passed on K3. Stage A of
the chat-layer implementation is built (see "Implementation").

## Goal

Keep a SparkPipe Chat conversation at a steady size without losing anything,
and without a re-prefill on every turn. The model decides what it sees: every
exchange becomes one summary line in an append-only ledger, the raw text stays
available behind the line, and the model opens what a question needs. The
memory work runs as continuations of the answer the engine has just computed,
so it shares that answer's prefix and costs only its own suffix (owner's
design).

## Engine facts this rests on

Verified in code (review, 2026-10-14):

- A request resumes from the longest committed prefix, cut back to the last
  resume point at or below it. Resume points are: every block end on GLM
  Full; on the other models only multiples of the checkpoint stride (16384
  tokens; GLM Flash 4096), a committed prompt's checkpoint at
  `(P - 1) / B * B`, and the end of a reply that finished on a stop token
  (`runtime/model_batch_engine.c:1979-2033`, `:1258-1313`).
- Generated tokens are reused when the next prompt repeats them exactly (a
  reply-end resume point). A finished answer can therefore be continued for
  the cost of the continuation alone. On K3, which decodes in multi-token
  chains, a reply that stops inside a chain resumes at the last 64-token
  block it crossed (measured on 8 of 8 curator requests, 2026-10-15); a
  reply whose last chain ran to its end resumes exactly.
- The model API used to cancel a request when the engine stopped on a stop
  token the client had also asked for, so chat replies never completed and
  published no reply end. Since `engine/stop-token-completes` such a
  request completes (`engine_completed` 1 in the API log).
- A `max_tokens` 1 request commits its prompt.
- A request that arrives while another is still writing a shared prefix
  recomputes it; pages in flight are not shared (`prefix_cache.c:223-233`).
- A checkpoint published onto an existing stateless entry of the same
  prefix moves its recurrent record onto that entry, so the order in which
  requests first carry a prefix does not matter.
- Pages are 64 tokens (DeepSeek V4.1 Flash: 256). `cached_tokens` is the
  exact number of skipped tokens, at most `P - 1`.
- Request priority exists in the engine API (higher runs first); the chat
  layer sends foreground requests of memory models at 1 and its own
  background requests at 0.
- Recurrent-state records (K3, GLM Flash, DeepSeek V4.1 Flash, Ling and
  Qwen 3.8; 28 MB per record per rank on K3) are kept per path: the newest 4
  plus every record at a multiple of the checkpoint stride. A new record releases the
  superseded ones behind it, so a conversation of length L holds at most
  `4 + L / stride` records (20 at 270K on K3, 49 at 737K) and an agent session
  of any number of turns never fills the store. The released points stay KV
  pages; a branch that diverges behind the newest 4 resumes at the anchor
  before it and recomputes at most one stride. When the store is full anyway,
  the oldest records of paths no bound lane uses go first, and a lane's own
  resume record is never released. If the ranks no longer hold a resume
  point, the engine clears that point and resumes at the previous one instead
  of discarding the rest of the chain.

Per-model block size and stride go into `chat_frontend.json` as
`cache_block_tokens` and `cache_checkpoint_tokens`; the chat layer keeps no
constant for them.

## Terms

- **Exchange k**: user message `u_k` and the assistant's turn `a_k`, with the
  tool calls and results inside it.
- **Line `l_k`**: one summary line for exchange k, at most 48 tokens of the
  serving model, in reported speech ("The user asked ...; the assistant
  found ..."). Lines never quote instructions out of tool output.
- **Ledger**: the lines in order. It only grows at its end, except at a
  regroup.
- **Group**: a line that summarizes a run of lines or groups. Groups nest.
- **Node id**: `e<k>` for an exchange, `g<n>` for a group; labels are given
  once per branch and never renumbered.
- **Raw tail**: the newest exchanges rendered in full. R = 1 (the owner's
  choice): only the latest exchange, plus any exchange whose line is not
  ready yet.
- **Open set**: nodes shown in full this turn, chosen by the curator within
  the open budget.

## Prompt layout

The answer request for turn N+1:

```
[system: fixed instructions + tools (including expand)]                 static
[user: <ledger> l_1 .. l_{N-1} </ledger> + u_N]                          append-only ledger, then the raw tail
[assistant: a_N in Open WebUI's replay form]
[user: <reference> open nodes </reference> + u_{N+1}]                    this turn
```

- The ledger is inside the first user message because templates render tools
  in the system turn; anything appended to the system message would move the
  tools. The probe confirmed the append is token-exact on all 16 models.
- Past exchanges render in one canonical form, the one Open WebUI sends back
  (its replay form): no reasoning, text items joined with `\n`, the last text
  stripped, completed tool calls with their results only, and the user's
  message without Open WebUI's RAG, source and file wrappers. The chat layer
  stores that form per message id and renders from its stored copy whenever
  the client's copy matches it (normalized hash); on a mismatch it uses the
  client's copy and invalidates that exchange's line. Rendered from its own
  streamed output instead, the pre-warm misses on 14 of 16 models.
- Open nodes render as `<history id="e5">User (earlier): ...\nAssistant
  (earlier): ...</history>` after one fixed sentence saying they are
  reference, not new requests; the static system text says the same.
- The reference block is always present, even when empty, so the pre-warm cut
  is in the same place every turn.

## One turn

After answer `A_N` finishes on a stop token, in the background:

1. **Curator `C_N`**, an exact continuation of `A_N`'s final request: its
   prompt is `A_N`'s `prompt_token_ids`, then the generated ids the engine
   returned, then the model's turn close, the curator instruction as a user
   turn, the generation prompt and the model's `curator_suffix` (an empty
   reasoning block, so no thinking is spent). It resumes at `A_N`'s reply end
   and prefills only the instruction. It sees the raw tail, the open set and
   the answer, and writes the line for exchange N, lines for any older raw
   exchange still missing one, the open set for turn N+1, and a regroup when
   asked.
2. **Stem `S_N`** (models with a checkpoint stride only): `max_tokens` 1,
   prompt = the next answer's prefix up to the end of its ledger. It must be
   the first request that carries the newest ledger line, so the checkpoint
   lands at the ledger end; it resumes at the previous stem and prefills one
   line.
3. **Pre-warm `W_N`**: `max_tokens` 1, prompt = the next answer's prefix up to
   the end of the reference block, cut by a sentinel in the same `render()`
   call. It resumes at the stem and prefills the raw tail and the open set.

Then `A_{N+1}` resumes at `W_N`'s checkpoint and prefills `u_{N+1}` plus at
most one block. Its token ids are `W_N`'s stored ids plus the encoded rest.

Costs, steady state:

| Request | Prefills | Decodes |
|---|---|---|
| `A_{N+1}` (foreground) | `u_{N+1}` + at most one block | the answer |
| `C_N` | the curator instruction (about 200 tokens) | about 60 to 160 tokens |
| `S_N` | one line + at most one block | 1 |
| `W_N` | `u_N` + `a_N` + the open set | 1 |

GLM Full resumes at every block, so it needs no stem.

If `A_N` did not end on a stop token (the client stopped it, it hit the token
limit, the chat layer cut it at a stop string), there is no reply resume
point: `C_N` is skipped and becomes a catch-up for the next curator.

## Scheduling

- Background requests bypass the warm queue (it would ask the controller to
  swap the fleet back). They run only while the conversation's model is
  active and ready and the controller is neither swapping nor draining; a
  swap request cancels them.
- At most one background job per engine, admitted only when no foreground
  request is active or queued, at a lower priority than foreground.
- When the user's request arrives: if it extends an in-flight stem or
  pre-warm, it waits for that request's first token (the prefix is committed
  by then); otherwise the background request is cancelled by closing its
  stream. A cancelled curator becomes a catch-up.
- Each pre-warm carries the branch's next-turn id and is dropped when stale.
- With several users on one model, background work only fills idle time; it
  never slows an answer.

## Lines: pending, fallback, immutable

- **Pending**: an exchange without a ready line stays raw. The ledger stays
  append-only when the line arrives.
- **Catch-up**: the next curator writes lines for every raw exchange in its
  prompt that lacks one.
- **Fallback**: only after a definitive failure (invalid output twice, the
  second try when the engine is idle): about 80 characters of `u_k` plus the
  first sentence of `a_k`, tagged `(auto)`.
- **Immutable**: a line rendered into any answer prompt never changes. Only a
  regroup (a ledger version bump) changes ledger text.
- Curator output is validated in parts (line, open set, regroup); an open set
  over budget is trimmed in the curator's order; a tool call in the curator's
  reply is invalid. The curator never goes through the client request path
  (`prepare()` would strip the tools and refuse JSON response formats).

## The curator's instruction

The instruction gives the current open ids, each node's size in tokens, the
open budget, the ledger budget and, when over it, the oldest runs to
regroup. It asks for JSON only:

```
{"lines": {"e12": "<reported speech, at most 48 tokens>"},
 "open": ["e3", "g1"],
 "regroup": [{"ids": "e1-e40", "line": "<one line for the run>"}]}
```

A regroup brings the ledger to about half its budget in one step, and it is
pre-warmed through the stem and pre-warm like any other turn.

## The expand tool

`expand(id, part)` returns at most about 2K tokens: the raw exchange `e<k>`
(in parts when longer) or the child lines of `g<n>`. The chat layer answers
it itself, never streams the call or its result to the client, and records
every expansion in the exchange's frozen layout so follow-up requests render
it again at the same position (Open WebUI rebuilds follow-ups from its own
output, which never contains it). Raw tool results over a size cap become
stubs with an expand id.

## Budgets

Per model, as fractions of `context_tokens`: measured system and tools, the
ledger, the raw tail, the open set, documents, and an output reserve of at
least max(4K, 15%). Before every answer a synchronous fit step folds the
oldest lines into a stub group instead of returning `context_length_exceeded`
(DeepSeek V4.1 Flash serves 16K and ten models serve 32K; a switch from K3
to a 32K model can arrive with a ledger over budget).

## Bookkeeping

- **Identity.** Open WebUI forwards a signed user header
  (`ENABLE_FORWARD_USER_INFO_HEADERS` with `FORWARD_USER_INFO_HEADER_JWT_SECRET`;
  HS256, `sub` = user id, with expiry) and, through `OPENAI_API_CONFIGS`
  header templates, the chat id, the user message id and its parent, the
  assistant message id and the task. The chat LiteLLM passes them through
  (`forward_client_headers_to_llm_api`). Memory is on only when the JWT
  verifies, the task header is empty and a user message id is present; every
  other request (ZCode, API clients, Open WebUI's internal tasks) is served
  raw as today.
- **Keys.** A conversation is (user, chat id); an exchange is its user
  message id; ancestry comes from the parent ids. Lines are also cached by a
  per-exchange content hash, so clones, forks and edits reuse them. Forks
  never cancel each other's jobs; work on a non-current branch runs last.
- **One layout per exchange.** Frozen per (user, chat, user message id)
  through the end of the reference block: system message, ordered tools,
  template kwargs, ledger version and lines, which exchanges are raw, the
  open set and expansions. `u_N` and the tool-result region are rendered
  again on each request. A change in kwargs, tools or system message counts
  as a ledger version change.
- **Prefix discipline.** Muse's template renders today's date at token 22;
  the chat layer passes a fixed `current_date` per conversation. Open WebUI's
  memory feature writes into the system message every turn, so
  `memories.system_context.enable` is off while ledger memory is on. Open
  WebUI's context compaction is never on for a ledger model; a request
  carrying `[CONVERSATION SUMMARY]` turns ledger memory off for that chat and
  is logged.
- **Restarts.** State is on disk; a torn last record is tolerated. Lines that
  were never rendered are rebuilt lazily on the conversation's next request.
- **Checks.** Each request's expected `cached_tokens` is the largest resume
  point at or below min(shared prefix, P - 1) on the latest request of an
  ancestor branch. A shortfall is logged with the engine's
  `stale_prefix_recomputes` and the ranks' `stateful_dedup_lost_count`.

## Storage

`/srv/workspace/sparkpipe/chat-memory/<sha256(user id)>/<chat id>.jsonl`:
canonical exchanges, lines, groups, open sets, regroups, frozen layouts and
curator replies, append-only. Temporary, local and channel chats stay in
memory only. Deleted chats are removed when Open WebUI no longer knows them.

## Integration

- In the chat layer (`tools/serving/`), model-neutral; per-model data
  (`curator_suffix`, cache geometry, budgets) lives in `chat_frontend.json`.
- `curator_suffix` per model, read from the templates (unverified until
  phase 1): GLM ×2, DeepSeek, Ling ×2, Laguna `</think>`; K3
  `<|close|>think<|sep|><|open|>response<|sep|>`; Qwen ×3 `\n</think>\n\n`;
  Hy4 `</think:opensource>`; MiMo ×2 `<think></think>`; Muse
  ` to=user<|message|>`; Gemma none. Reasoning is not switched off through
  template kwargs: that changes the prompt within its first 37 tokens on 11
  of 16 models.

## Evaluation

**Phase 1, prefix mechanics** (about 15 minutes on K3, the worst case; GLM
Full if it is loaded): a standalone script renders with the chat layer's
renderer and sends `prompt_token_ids` to the engine. Eight exchanges at
temperature 0 with about 4K tokens of system and tools and a 3K-token
ledger; per turn `A_N`, `C_N`, `S_N`, `W_N`, `A_{N+1}`. Pass when `C_N`
skips `|A_N prompt| + |generated| - 1` tokens, `S_N` and `W_N` skip at least
their resume point, `A_{N+1}` skips at least `|W_N| - 64` (exactly `|W_N|` on
GLM Full), with no recomputes and no lost checkpoints. Controls: the
revision-1 curator layout (expected to skip 0 or 16384 on K3) and a fast user
who arrives 0.5 s into `W_N`. Also recorded: the curator's JSON validity with
the suffix, and its decode length.

**Phase 1 result, K3, 2026-10-14: passed.** `tools/chat_memory_phase1.py`
(run on the hub against the loaded engine), raw rows in
`qualification/chat-memory-phase1-k3-20261014.jsonl`. A 3.7K to 4.3K token
conversation (system, ten tools, an 80-line ledger), eight turns, temperature
0, `thinking_effort` low, 33 requests:

| Request | Resumed at | Prefilled | Time |
|---|---|---|---|
| First answer, nothing warm | 0 | 3,678 tokens | 6.6 s to first token |
| Answer after the pre-warm (7 turns) | the pre-warm's checkpoint, every turn | 37 to 92 tokens | 0.33 to 0.91 s to first token |
| Curator (8) | the reply end on 3 turns, the block below it on 5 (at most 53 tokens short) | 248 to 343 tokens | 7.1 to 9.3 s (85 to 118 tokens decoded) |
| Stem (7 after the first) | the previous stem's checkpoint, every turn | one line plus at most a block | 0.37 to 0.85 s |
| Pre-warm (8) | the stem's checkpoint, every turn | raw tail plus reference block | 0.78 to 1.29 s |

- Every stem, pre-warm and next answer was a token prefix of the next request
  (asserted), and the engine logged zero `stale_prefix_recomputes`.
- The curator's JSON was valid 8 of 8 times with K3's `curator_suffix`; its
  lines were accurate reported speech with names and numbers kept.
- Answers that needed the ledger were right: the figure recorded for item 12,
  the topic of item 30, and a speed stated two answers back (by then only in
  a line).
- A user arriving 0.5 s into the pre-warm still reused all of it (the engine
  finished the pre-warm's prefill first); first token 1.0 s.
- Background time per turn: about 9 to 11 s, mostly the curator's decode;
  idle fleet time while the user reads.
- Not covered yet: long contexts (stride checkpoints at 16K), other models,
  several users, answer quality at scale (phase 2).

**Phase 2, quality** (hours, same model): 10 scripted chats of 15 to 25
exchanges, each with 5 exact-match probes whose answer sits 3 or more
exchanges back. Arms: raw, ledger + expand, ledger without expand, ledger
with an oracle open set. If ledger + expand is within 5 points of raw with
expand recall of at least 80%, build. If ledger without expand scores like
ledger + expand, the models are not using expand: add lexical retrieval into
the reference block when the question arrives, and a raw-tail budget, before
building.

**Phase 2 result, K3, 2026-10-14: passed; build.** `tools/chat_memory_phase2.py`
(through the chat layer on the hub), rows in
`qualification/chat-memory-phase2-k3-20261014/`. 12 generated chats of 17 to
24 exchanges, 96 probes (12 per pattern: a code fix, title drafts, options,
a pasted memo, a search result, a standing instruction, a price table, a
correction), each asked 3 or more exchanges after its fact; 384 answers and
249 curator lines.

| Arm | Correct | Minus raw (95% CI, paired bootstrap) |
|---|---|---|
| Raw history | 96/96 | |
| Ledger + expand | 95/96 | -1.0 points (-3.1 to 0) |
| Ledger with the right exchange opened | 95/96 | -1.0 points |
| Ledger only | 80/96 | -16.7 points (-25 to -9.4) |

- The curator's JSON was valid 249 of 249 times; its line kept the probed
  fact 84 of 96 times.
- Expand: in the 12 probes where the fact was hidden and its line lacked it
  (all 12 were the memo's room number), K3 expanded the right exchange 12 of
  12 times and answered all 12; it expanded nothing anywhere else.
- Ledger only lost all 12 memo probes (it answered that the room was not
  given) and 4 others whose line did hold the fact.
- The two misses in the ledger + expand and oracle arms were empty replies,
  not wrong answers.
- Prompts were about a third smaller with the ledger (median 1,221 against
  1,867 tokens); these chats are short, and the gap grows with length.
- Latency here is not the production picture: this test sent no pre-warms
  and every probe changed the ledger position, so raw history reused the
  cache better (median 5.2 s against 7.4 s, p90 7.8 s against 27.6 s with
  expand rounds). Phase 1 measured the pre-warmed path.
- Limits: generated chats, one model, short conversations.

**Ship gate per model**: at least 150 exact-match probes across 8 patterns
(code diffs, multiple drafts, an option given N answers back, an attached
document, a tool result two or more back, a standing instruction, a numeric
table, a correction) plus prompt-injection and replay probes, LoCoMo (10
conversations) and 30 LongMemEval questions; arms raw, raw + pre-warm,
compaction on chats past its threshold, ledger + expand, ledger without
expand, oracle open set. Ship when accuracy is at least raw minus 3 points
overall and raw minus 5 per pattern (paired-bootstrap lower bound) and at
least compaction's, expand recall is at least 80%, curator fallbacks at most
2%, zero injection or replay failures, p95 first-token time with expand
rounds at or below raw + pre-warm, and foreground decode with two users at
most 5% slower. A model that fails keeps raw history.

## Implementation (stage A, 2026-10-14)

`tools/serving/chat_memory.py`, loaded by the chat layer when its config has
`chat_memory`; per-model settings under each model's `chat_memory` (K3 only so
far). Tests: `tests/test_chat_memory_core.py`,
`tests/test_chat_memory_session.py`; live driver `tools/chat_memory_live.py`.

- Identity: the signed `X-OpenWebUI-User-Jwt` (HS256, issuer `open-webui`),
  chat id, user message id, its parent and the assistant message id from
  Open WebUI's per-connection header templates, passed through by the chat
  LiteLLM. Anything else is served raw.
- Lines: past exchanges are keyed by a content hash of their replay form; the
  current exchange's line waits under its assistant message id and binds on
  the next turn through the parent id. The open set follows the same path.
- Layout: from the first memory turn the system note and the `expand` tool are
  present; the ledger covers the gap-free prefix of lined exchanges and always
  leaves the newest finished exchange raw. One layout is frozen per user
  message; expansions are replayed in follow-ups.
- Expand: answered inside the chat layer, never shown to the client, at most
  three rounds per request, 8,000 characters per part.
- Background: after an answer that finished on its own, the curator (same
  model, rendered as the answer plus an instruction, with the model's
  `curator_suffix` closing the reasoning block) writes lines for the newest
  exchange and up to six older raw ones, oldest first; then a ledger-end stem
  on checkpointed models. Background requests go straight to the engine at
  priority 0 (foreground carries 1), only while the controller reports the
  model active and idle, are cancelled by any foreground request, and are
  retried; two invalid curator replies leave fallback lines.
- Stage B (2026-10-15) added the curator continuation, the pre-warm,
  regroups and the fit step; see below. Not yet: other models (stage C).

Live, K3, 2026-10-14, a separate chat-layer instance driven like Open WebUI
(8 turns, 3 s between turns, a unique first message): the ledger grew to
cover every exchange but the newest, each line was written 5 to 8 s after its
answer, the prompt grew from 367 to 650 tokens over 8 turns, each turn
prefilled 133 to 210 new tokens, and both recall questions were answered
from the lines.

## Implementation (stage B, 2026-10-15)

- **Curator continuation.** The curator's first attempt is the answer's
  `prompt_token_ids`, the generated ids without a trailing stop token, the
  template's tokens after an assistant message's content (minus any the model
  already wrote, such as K3's response and message close tags), the
  instruction as a user turn, the generation prompt and `curator_suffix`.
  The template tail is cut out of one render with two sentinel messages, so
  it is the model's own template, not a per-model table. A retry re-renders.
- **Pre-warm.** After the curator and the stem, the chat layer builds the
  next turn's layout (this exchange in Open WebUI's replay form: the joined
  content, stripped) with a sentinel as the new question, cuts it there and
  runs it for one token. The next answer resumes at its prompt checkpoint and
  prefills its question plus at most one block.
- **Regroups.** Each model has a ledger budget (`ledger_tokens`, default a
  fifth of `context_tokens`). When the ledger is over it, the curator is
  asked for one group line per run of `group_size` (default 16) of the
  oldest entries it can see, until the ledger is about half its budget.
  Groups are stored by their members (exchange keys or older groups), nest,
  render as `[g3: e1-e16] line`, and `expand g3` returns their child lines.
  A follow-up keeps the groups of its frozen layout. A curator that fails
  twice leaves stub group lines tagged `(auto)`.
- **Fit.** Before an answer is rendered, a prompt over the model's budget
  (`prompt_tokens`, default `context_tokens` minus the larger of 4096 and
  15%) is made to fit without a model call: the open set is dropped, raw
  exchanges other than the newest get fallback lines, and the oldest half of
  the ledger entries is folded into a stub group, until it fits. A prompt
  that fits is never touched.

Live, K3, 2026-10-15, through the production chat layer (8 turns, 30 s
between turns, a unique first message):

| Turn | Prompt | Cached | Prefilled | TTFT (s) |
|---|---|---|---|---|
| 1 | 366 | 320 | 46 | 0.34 |
| 2 | 458 | 320 | 138 | 0.76 |
| 3 | 528 | 448 | 80 | 0.87 |
| 4 | 547 | 512 | 35 | 0.41 |
| 5 | 551 | 512 | 39 | 0.33 |
| 6 | 591 | 512 | 79 | 0.63 |
| 7 | 621 | 576 | 45 | 0.36 |
| 8 | 660 | 576 | 84 | 0.66 |

From turn 3 on (the first turn with a ledger) every answer resumed at its
pre-warm's checkpoint. Each curator resumed at the last block of the reply
it continues. Both recall questions were answered from the lines.

Regroup, live, K3, same driver against a second chat-layer instance with a
150-token ledger budget and runs of 3: after turn 5 the curator folded e1-e4
into `[g1: e1-e4]` (it kept the budget code and room, and summarized the
title ideas without the titles). On turn 7 K3 called `expand g1`, read the
child lines and gave the second title exactly; on turn 8 it answered the
budget code and room from the group line. The turn after the regroup
resumed at the pre-warm (612 prompt tokens, 576 cached). Evidence:
`qualification/chat-memory-stageb-k3-20261015/`.

## Open choices

- **Raw tail**: R = 1 (owner) or a token budget for the raw tail (25 to 40%
  of context, never less than one exchange, collapsed to half in one
  pre-warmed step). Phase 2 measures both.
- **Cross-chat memory** per user is a separate layer.
- **Engine follow-up**: a model-neutral `cache_checkpoints` field on
  `/v1/completions` would replace the stem request.
