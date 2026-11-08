# Context memory: an optimized context rebuilt in the background

Status, 2026-11-07: built for SparkPipe Chat and for API sessions, one
exchange at a time, and for the tool results of the rounds inside an
agent's turn; a line per round is next. It extends [CHAT_MEMORY.md](CHAT_MEMORY.md) for both SparkPipe Chat and
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
  it, and so do the recall benchmarks, whose histories are given. The import
  takes a chat request's template settings (`reasoning_effort`,
  `chat_template_kwargs`) and pre-warms at them, so it must be sent with the
  settings the conversation will continue with: Qwen 3.8's template opens its
  system block with the reasoning effort, so a base warmed at another effort
  shares nothing past the first few tokens with the next request, which then
  prefills the whole layout cold. The import checks its memory identity once,
  when it arrives, and opens every batch as that identity under the batch's
  own turn ids: a long history takes as long as its batches do, past the life
  of the token it was sent with.
- **Curator replies.** One curator reply writes the lines asked for, then the
  open list, the groups, the kept sections and the commits. Its output budget
  is 160 tokens, plus 80 per line asked and per group, 10 per listed tool
  part, and 60 per exchange the commit is expected to take: the oldest open
  exchanges that free what it needs, the same ones a fill would commit. A
  reply cut short keeps every field it finished. Its lines stand, and a
  commit it did not finish is filled oldest first, as when the curator
  commits too little. In the LongMemEval_S bench's imports on Qwen 3.8 27B,
  the old budget (160 plus 80 per line) cut 340 of the 419 renders that also
  committed. A cut reply was rendered again, and a second cut fell back, so
  about one line in eight was only the first words of its exchange.

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

## Rounds inside an agent's turn

An agent's turn is one user message followed by tens to hundreds of rounds:
an assistant message with tool calls, then the tool results. Exchange-level
memory leaves the whole turn raw, so the prompt still grows with every
round.

Built (`tools/serving/chat_memory.py`, `round_cut` and `rounds`): when a
request's raw rounds hold more large tool results than the working budget,
a round build folds the older rounds back to the budget's low mark, newest
round always raw. The curator picks the sections of the folded results to
keep, the build pre-warms the next request's layout, and only then
publishes the cut (store record `round_base`). Later requests of the turn
render the folded results as stubs plus the kept sections; the assistant
messages and every newer round stay in full, and `expand` reads any result
back. `tests/test_chat_memory_session.py` checks the cut, that a request
during the pre-warm still sees every round in full, the folded rendering
and the pre-warmed prefix.

The full design, with the line per round still to build, applies the same
two rules with the round as the unit:
- **Lines.** A round leaving the turn's raw window gets one line ("r12: ran
  the tests; 3 failures in parse_args, KeyError 'mode'"). The curator writes
  it as a continuation of the round's own tokens, so it prefills only its
  instruction.
- **Forms.** A tool result leaving the raw window becomes its stub plus the
  sections the curator keeps, as for exchanges; `expand r12.t1` reads it
  back in full.
- **Layout.** The round lines sit right after the user message, as the
  model's own earlier `expand` call and its result, so the client's messages
  stay byte-identical and the block only grows at its end. Then come the
  raw rounds of the base, then the bridge.
- **Trigger.** A round is complete only when the next request brings its
  tool results, so a build starts when a request arrives, the turn's raw
  rounds pass the round budget and no build runs for the session. It runs
  beside the request at background priority (`background_with_foreground`).
- **Switch.** A request uses the newest base whose rounds its history holds.
  Within one base every round replays its exact tokens, so the cache extends
  by one round at a time. Moving to a newer base prefills, once, the rounds
  of its bridge.

## Nothing is lost

Every event stays in the session's store. The model reads any of it with
`expand` and `search_history`, which the chat layer executes itself; an API
client never sees those calls. Compaction can therefore be aggressive: tool
output that can be fetched again leaves the raw window first.

## Identity

- SparkPipe Chat: the signed Open WebUI user and chat id, as now.
- The API (built): memory is opt-in per request with
  `X-SparkPipe-Memory: on`, on chat completions, Messages and Responses.
  - The user is the key's scope and member. The session is the client's
    session id: `X-Claude-Code-Session-Id`, `X-Session-Id`, the Responses
    `prompt_cache_key` or the Messages `metadata` session. A Claude Code
    subagent's id follows the session id, so each subagent has its own
    memory.
  - Turn ids come from the conversation. A turn is named by the digest of
    the conversation up to its user message, its reply by the same digest,
    and its parent by the previous turn's. A client that resends its history
    therefore binds the curator's lines exactly as Open WebUI's message ids
    do. Open WebUI message-id headers on an API request are ignored.
  - Opting in without a session id answers 400. The harness's own
    compaction and auxiliary requests stay raw.
  - Replay keeps working inside a memory session. A frozen layout renders
    the same messages on every round, so each reply's exact tokens are found
    again and a tool round extends the previous round's prompt and reply in
    the engine's cache. A reply that used `expand` or `search_history` is
    not recorded for replay.
  - `tests/test_chat_memory_api.py` checks each of these.
- Later: the providers' own compaction requests (Anthropic
  `context_management`, OpenAI `context_management` and
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
