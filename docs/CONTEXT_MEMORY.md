# Context memory: an optimized context rebuilt in the background

Status, 2026-11-07: built for SparkPipe Chat; agent rounds and the API are
next. It extends [CHAT_MEMORY.md](CHAT_MEMORY.md) for both SparkPipe Chat and
the API. The ledger, curator, expand and search tools, tool-output forms, stem,
pre-warm and store described there stay.

What is built (`tools/serving/chat_memory.py`):
- A pre-warm that finishes records its layout (the committed exchanges, the
  groups and the tool-output forms it rendered) as the chat's newest base
  (store record `base`).
- A new turn is laid out from the newest base whose exchanges its history
  holds: those exchanges are in the ledger, and every later exchange, even
  one the curator has committed since, stays raw. A turn's layout stays
  frozen for its follow-ups, as before. A chat with no base yet uses its
  latest commits.
- `tests/test_chat_memory_session.py` holds a commit raw while its pre-warm
  runs and in the ledger for the first turn after it.
- **Concurrent builds.** The chat layer settings `background_jobs` (default
  1) and `background_with_foreground` (default off) set how many chats build
  at once per engine and whether builds run beside foreground requests. A
  chat never runs two builds at once; its newer job waits for the running
  one.
- **Import.** `POST /v1/sparkpipe/memory/import` (the chat layer, with a
  memory identity: `model`, `messages`, and `batch` exchanges per curator
  call, default 8) brings a given conversation into memory without
  generating any answer: for each batch it opens the turn of the batch's last
  exchange and runs the curator, which writes the lines of the batch through
  its catch-up, commits and regroups as in a live chat, and the batch's
  pre-warm publishes a base. The reply counts the exchanges, the lined ones,
  the commits and the bases. A customer moving a long conversation in uses
  it, and so do the recall benchmarks, whose histories are given.

## Goal

Keep every session, a chat or an agent's tool loop, at a steady prompt size
without losing anything, while each request processes only what is new. Long
prompts cost twice: each new token is computed against the whole context, and
a context the cache no longer holds is read again in full. This design never
reads a long context again and keeps the context short.

## Terms

- **Event**: one message of the session in its canonical replay form: a user
  message, an assistant message (with its tool calls), or a tool result.
- **Base** `B_k`: an optimized context covering events `1..c_k` (its cut).
  It is the system prompt and tools, then a ledger with one line per event run
  that left the raw window, then the raw window: the newest covered events in
  full. A base is immutable once published.
- **Bridge**: the events after a base's cut, in full, exactly as the client
  sent them.
- **Build**: the background work that produces the next base.

## The two rules

1. **At the end of each answer**, if no build is running for the session,
   start one with every event so far, the new answer included. A build may
   take longer than one exchange; requests keep using the previous base
   meanwhile.
2. **At the start of each request**, use the latest published base whose
   covered events match the client's history, and append the bridge: every
   event after the base's cut, then the new request.

Both rules are the same for a chat turn and for an agent's tool round. An
agent's every assistant message is an answer.

## Prompt layout

```
[system: instructions, tools, the memory note]                      fixed per session
[user: <ledger> lines </ledger>, then the raw window's first event]  the base
[raw window events ...]                                               the base
[bridge events ...]                                                   since the cut
[new request]
```

- The ledger only grows at its end between regroups, so a new base reuses
  the previous base's cache up to the old ledger's last line. A build
  computes only the new lines and the raw window.
- A base ends on a message boundary, and the base followed by the bridge
  renders as the base's token ids followed by the bridge's. The renderer
  checks this per request; a request whose render does not start with its
  base's ids uses the previous base or no base.
- The raw window keeps the newest covered events up to its budget, and never
  splits a tool call from its result.

## A build

1. **Lines.** The curator writes a line for each event run leaving the raw
   window, as a continuation of the answer that started the build: it
   resumes at that answer's end and prefills only its instruction.
2. **Forms.** A tool result leaving the raw window becomes a stub plus the
   sections the curator keeps (CHAT_MEMORY_STAGE_C.md, "Tool output").
3. **Compose.** Ledger = the previous base's lines plus the new ones;
   raw window = the newest covered events within budget. Over the ledger
   budget, the oldest runs fold into group lines (a regroup), which changes
   the ledger's text once.
4. **Warm.** One commit request at background priority puts the base's
   tokens in the engine's prefix cache. The build is published when the
   engine confirms the commit.

At most one build runs per session, and up to `background_jobs` sessions
build at once per engine. Builds run at background priority. By default a
build waits for the engine to have no foreground request, and a foreground
request cancels it; the cancelled build comes back as a catch-up. An agent's
tool loop never leaves the engine idle, so agent sessions need
`background_with_foreground`: builds then run beside foreground requests and
share the engine with them, which the benchmarks measure as foreground
slowdown.

## Nothing is lost

Every event stays in the session's store. The model reads any of it with
`expand` and `search_history`, which the chat layer executes itself; an API
client never sees those calls. Compaction can therefore be aggressive: tool
output that can be fetched again leaves the raw window first.

## Identity

- SparkPipe Chat: the signed Open WebUI user and chat id, as now.
- The API: the account and a session id, from the `x-sparkpipe-session`
  header, or the harness's own session id (Claude Code's session in
  `metadata`, Codex's `prompt_cache_key`). Memory is opt-in per request with
  `x-sparkpipe-memory: on`, and through the providers' own compaction
  requests (Anthropic `context_management`, OpenAI `context_management` and
  `/v1/responses/compact`).

## Engine contract

The engine:
- takes `prompt_token_ids` and resumes from the longest committed prefix it
  holds;
- commits a prompt without generating (`max_tokens` 1) and saves recurrent
  state at a committed prompt's end and at a reply's end, so a later request
  resumes exactly there;
- runs requests at a priority and cancels on stream close.

## Measured against

Same model, same tasks, three arms: the full history, provider-style
compaction (one summary at a threshold), and this design.

- Recall: LongMemEval_S (a balanced subset) and LoCoMo.
- Agent sessions: task success on a SWE-bench Verified subset and
  Terminal-Bench.
- Cost: new prefill per request, time to first token, prompt size, and how
  often a request waits for a build (never, by design).

Called state of the art only with receipts: recall within 2 points of the full
history, task success equal to the full history and above compaction, and at
least 5 times less new prefill per agent step on long sessions.
