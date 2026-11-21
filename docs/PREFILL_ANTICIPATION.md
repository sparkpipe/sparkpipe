# Prefill anticipation: audit and design

Status: approved by the owner on 21 November 2026, with the decisions in section 9. Code was read at main `c2b714343`, and draft PR #1821 at `c5d5ea0cd`. Nothing here is built yet. Section 10 lists the sources and how each number was obtained.

## 1. Verdict

None of the four cache Functions the site lists (Build and refresh the cache, Anticipate tomorrow, Prefill ahead, Resume sessions) can be used today.

The work so far anticipates **files**: packs of files, and nightly leaves of file combinations. The cache does not reuse files. It reuses an exact token prefix from position 0, and an agent's prompt is built by the client: tools, system prompt, project instructions, then an append-only conversation in which file content arrives as tool results. A bundle prefilled at night is reused only by a request that begins with exactly that bundle.

Public agent traces show where the time goes:

- **Inside a session** that keeps its own prefix, prefill is 2–10% of model time, provided a small tail costs in proportion to its size. That has not been measured at long context on the current release (the turn floor, section 3).
- **The minutes are lost when the prefix is lost**: coming back after a break, a new session's opening, a subagent's opening, a compaction, a model swap, an engine restart or release, and the engine's deletion of anything unused for 24 hours.

The design therefore does three things, in this order:

1. **Make reuse exact by rule.** A request gives the same tokens whether it runs cold or from the cache. Today that is unproven for a cached prefix followed by a new tail.
2. **Keep what was computed.** A session's chain, each client's opening and any prefix the customer names survive nights and weekends, eviction, swaps, restarts and, where the arithmetic is unchanged, releases. Nothing is deleted because of its age while the customer holds the fleet. Kept prefixes come back from NVMe in seconds.
3. **Anticipate only prefixes that are known exactly**, on events and client signals, never by guessing file sets.

Repository packs stay, restructured, for work that wants the repository in the prompt. They stop being the centre.

## 2. What exists

| Piece | State | What is wrong |
| --- | --- | --- |
| Context packs, PR #1820 (`tools/serving/context_packs.py`) | Merged. The deployed configuration (`deployment/hub/chat_frontend.json`) has no `repositories` section, so every pack route answers 404 | Order, profiles, re-warming, truth (below) |
| Anticipator, PR #1768 (`tools/anticipator.py`) | Merged as an offline command. Nothing imports, schedules or serves it | Its output cannot reach the pack builder (below) |
| Cache controls, draft PR #1821 | Untouched since 6 November. Conflicts with main in 7 files | Pins never reach the ranks, and the warm-up is not a commit (below) |
| Save after read, PR #1949 | Released on 20 November | Correct for ordinary tails. It drops every warm-up that is evicted before its first read |
| Resume sessions | No code | The spec's method cannot work: no prompt tokens are stored anywhere on the fleet |
| Working state, draft PR #1691 | Open since 1 November, conflicts with main | Main's engine still deletes cached state 24 hours after its last use, whoever holds the fleet (launch issue #1789) |
| Deployed chat layer | The copy of 4 November | Predates packs. Under the menu model all of this must run on the renter's fleet, and the code assumes a shared hub behind our gateway |

### Defects found

**Packs**

1. **The order moves by itself.** Files are re-sorted on every build by their change count in the 90 days before the pack commit (`tools/serving/context_packs.py:241`). A file whose old changes leave the window moves earlier although nobody touched it. Simulated on this repository over 26 days with a 400K-token budget, the pack changed on 17 days and a median 3.5% of the old prefix survived a change; on 12 days more than 90% was lost. "Recomputes only what changed" is not true of this code.
2. **One pack is several prefixes.** Whatever the template renders before the pack is part of the prefix. Where a template puts tools first, every tool set is its own copy, and the chat layer warms up to 4 of them (`MAX_PROFILES`). GLM 5.3 Full and DeepSeek V4.1 Flash with thinking start with a reasoning-effort header, so each effort level is another copy.
3. **An unchanged pack is never warmed again.** A refresh returns early when the version is the same, so a pack lost to eviction, a swap or a restart stays cold. The `warm` state is a flag in the chat layer; the engine is never asked.
4. **A background warm-up can swap the fleet.** `warm_prefix` goes through `engine_stream`, whose not-ready path asks the swap controller to activate the pack's model.
5. **Reported reuse is overstated.** `resume_tokens` defaults to 64 for every model, while most adapters can resume only every 16,384 tokens. Warm-ups declare no cache points.
6. **Packs are skipped silently** when a memory session is active.
7. **The developer has to choose.** A pack is selected by changing the model name to `<model>+<pack>` in every harness. The first request on an unbuilt pack fails with 409.

**Anticipator**

8. **Not wired, and not wireable as it is.** Leaf names contain `/` and `+`, which the pack name rule and the alias syntax refuse. The pack builder re-sorts globally and cannot render root, trunk and leaf. The two tools count churn differently.
9. **The report counts nothing on real records.** It reads `time`; request records carry `at`.
10. **Profiles exist only for Kimi K3**, at 460 tok/s. GLM is refused, and Kimi K3.1 Pro, DeepSeek V4.1 Flash and the Qwen models have none.
11. **It plans combinations.** A hint on one directory, with no session evidence, planned 9 leaves that share only a 32,768-token root: 2,490,368 new tokens, about 90 minutes of build on Kimi K3. One hinted file anywhere in a leaf earns the leaf the full bonus.
12. **Its evidence command reads agent logs on the developer's machine.** Evidence must come from the fleet's own request records.

**Cache controls (#1821)**

13. **The pin flag collides.** Its rank-side `PINNED` flag is 0x40, which is `REUSED` on main since #1949. Its batch ABI number is taken.
14. **Pins protect one layer of five.** They skip rank eviction only. The API's prefix index, recurrent states, the snapshot quota and the save rule are untouched, and the engine endpoint answers 501.
15. **`commit_only` still samples a token** and drops it, so the request ends with a one-row step and leaves no state at the prompt end.

**Engine**

16. **A cached prefix followed by a new tail is not shown equal to a cold run on any model.** The release kit repeats identical prompts only, and no stage adapter has passed an I27 session (`TECHDEBT.md`). Section 5.1 explains why it does not hold by construction.
17. **Recurrent state is the binding capacity.** An 8-sequence lane holds 36 state records per rank. One Kimi K3.1 state is 28.4 MB per rank. Only the newest 4 non-anchor states on a path survive, so a branch point many sessions share is released once any one of them has put 4 newer states after it. Unread states are dropped, not spilled.
18. **The API's prefix index caps what is reachable**: 131,072 entries, 8.39M tokens for Kimi K3.1 and GLM 5.3 Full, evicted by plain LRU. Its protection call (`SparkPrefixCacheProtectPromptLookahead`) has no caller.
19. **Every release empties the cache.** The KV layout digest includes the adapter binary's SHA-256.
20. **No wave can be aborted.** An interactive request waits for a background wave to finish: about 4 s on Kimi K3.1, and 17–31 s on GLM 5.3 Full at 670–750K of context.
21. **Close saves newest first within 20 s.** Old, never-read warm-ups are saved last and lost first at a swap.

**Signals the serving layer drops**

22. **Codex's own warm-up.** Codex sends `response.create` with `generate: false` at session start and when it restores a thread. The chat layer answers it with zero usage and never calls the engine (`tools/serving/chat_frontend.py`, `socket_warmup`). Setup does not enable the WebSocket transport it needs.
23. **Claude Code's compaction signal.** `x-claude-code-context-compacted` says the old prefix is no longer used. Only the request class header is read.
24. **System blocks are joined** into one string (`tools/serving/chat_shapes.py`), which hides the block boundaries where sessions branch. A per-conversation attribution block, when a client sends it, then ends prefix sharing at the first system token.
25. **`cache_creation_input_tokens` is always 0**, so a client cannot see its own misses.

## 3. Where prefill time goes

Figures come from public traces of 80,000 agent sessions (5.1 million model requests) across Claude Code, the Claude Agent SDK, OpenCode, OpenHands and SWE-agent, plus 28 donated Claude Code sessions on real projects. Token counts in the first group are characters divided by 4; engine counts ran 12–34% higher where both were available.

### Inside a session

| Measure | Value |
| --- | --- |
| New tokens a request appends | p50 40–288, p90 560–2,400, p99 2,300–7,500 |
| Context the request lands on | p50 17K–50K, p90 60K–135K |
| Input tokens served from the cache | 98.2% (28 real sessions, 663M input tokens, a provider's cache) |
| Prefill share of model time with the prefix kept, if a tail costs in proportion to its tokens | Qwen 3.8 27B 1.8–7.9%, Kimi K3.1 2.5–10.4% |
| The same, if every turn on Kimi K3.1 pays for a whole 2,048-row wave | 9–30% |
| Prefill share with no reuse | Qwen 3.8 27B 65–86%, Kimi K3.1 71–89% |
| Mean per session, Agent SDK set, Kimi K3.1, proportional cost | 59 s of prefill with the prefix kept, 2,348 s of decode |

**The turn floor is not known.** The two Kimi K3.1 rows differ by what a small tail costs at long context. The only published figures are for Kimi K3: a cached follow-up takes 4.7–5.1 s at 137.7K of context, and 8.0 s for 1,872 new tokens at 337K (`PERFORMANCE_STATUS.md`). They are a month old and nothing records what they consist of. If that floor still holds, it is the largest prefill term inside a session, and anticipation does not remove it. Stage 0 measures it and names its parts.

### When the prefix is lost

| Event | Size | Evidence |
| --- | --- | --- |
| Back after more than an hour | 15 of 16 such gaps ended in a rebuild of the whole context: 71,753–668,954 tokens, median 124,043 | 28 real sessions, on a provider cache with a one-hour lifetime |
| All rebuilds | 45 rebuilds, median 108,033 tokens, were 52.8% of every missed token | Same |
| Session start | 46,328 tokens of tools and system prompt, then 3.4–6.2K per directory and 4–7K of project instructions | One user's capture of the Claude Code CLI |
| Subagent start | About 37K tokens, about 3% of them unique | One user's measurement |
| Compaction | 4 of 439 sessions | Claude Code on Terminal-Bench |
| A harness that rewrites history | Masking old observations broke the prefix on 76–94% of requests. On SWE-agent it raised uncached tokens per request from about 360 to about 7,000 | SWE-agent and OpenHands |

### Files

| Measure | Value |
| --- | --- |
| Distinct files read in a session | p50 2–14, p90 7–29 (benchmarks); median 16.5 (real sessions) |
| Share of read tokens in the top 5 files | 77–100% (session median) |
| A session's working set | p50 1.6K–15.8K tokens |
| Re-reads of an unedited file | 12–23% of read tokens in most sets; byte-identical repeats mostly under 2%, because a re-read usually asks for another range |
| Read sets of two tasks on one repository | Jaccard 0.03–0.21 |
| A snapshot of the files earlier tasks read most | Covers 35–37% of later read tokens at 50K tokens and 46–53% at 400K (Open-SWE-Traces); 53% and 65% (SWE-rebench) |

**Same few files, or unpredictable combinations?** Both, at different levels. Within a session the agent works on the same few files, and those are not misses: once read, a file sits in the session's cached prefix. Across tasks the file sets are mostly new, and a snapshot of popular files stops helping at about half the reads. The misses that cost minutes are not about files at all. They are lost prefixes.

**Limits.** These are single-task benchmark sessions and a small donated sample. One developer's weeks in one repository are not in any public set, and probably overlap more. The fleet's own miss ledger (section 5.6) has to answer this for real customers.

### What context costs

A repository in front of every prompt is not free. Drafted whole-file rewrites on Kimi K3 run at 76.8 tok/s at 73K of context and 41.8 at 337K; on GLM 5.3 Full, 105 at 46K and 86.2 at 198K (`PERFORMANCE_STATUS.md`). Decode is most of a session, so a large pack can cost more than the reads it saves.

## 4. Principles

1. **Exact only.** Reuse never changes a token.
2. **The unit is a prefix from token 0.** Prefixes are anticipated; files are not.
3. **Keep before predicting.** What was computed once is not computed again, and it is never deleted because of its age while the customer holds the fleet.
4. **Predict only what is known exactly**: a session's own history, a client's opening, and prompts we build ourselves.
5. **A client's messages are never rewritten.** The one thing removed is a billing-attribution line that some clients put first; it carries no instruction.
6. **Nothing for the developer to choose.** One setup, then it works.
7. **Background work has a stated bound** on how long it can delay a request, and never swaps a model.
8. **Every stage has a number and a gate**, and every miss has a recorded cause.

## 5. Design

```
[template start: BOS, and the effort header where a family has one]
[pack base]             optional; kept; shared by every session on that version
[pack delta]            optional; small; replaced at each push
[tool block]            per client and tool set
[client system blocks]  per client version
[project context]       CLAUDE.md, AGENTS.md, memory files
------------------------ the opening ends here; sessions branch ------------------------
[first prompt] [reply] [tool results] [reply] ...        the session's own chain
```

### 5.1 Exactness

**Contract.** The tokens a request produces depend on the request, never on what the cache held. This is invariant I27 extended from "restored equals uninterrupted" to "resumed equals cold".

**Why it does not hold today.** Prefill runs in waves. `SparkModelBatchPrefillSpan` (`runtime/model_batch_engine.c`) starts each wave where the last one ended, so a request that resumes at position r computes on a grid shifted by r, and kernels take different paths by wave size (16 rows, 256 rows, attention plans). On top of that:

- A warm-up today is a one-token completion. The later request resumes one token before the warm-up's end, in the middle of a block.
- The chat layer declares a cache point at the position its history ledger has seen (`cache_points(seen, total)`), so the cut depends on earlier traffic.
- A follow-up turn reuses KV that decode steps computed for the reply; a cold run computes those tokens in prefill waves.
- A capacity refusal halves the wave, so load changes the cuts.

**Rules.**

| Rule | Text |
| --- | --- |
| R1 | **Cuts come from the request alone.** A wave ends at each block boundary of the opening (everything before the first assistant message: system blocks, project context, the first prompt), one token before each assistant message begins (where that turn's prompt ended), where each assistant message ends, at each checkpoint anchor, and one token before the prompt's end. |
| R2 | **A request resumes only at a cut**, on every model, including those that can resume at any block. |
| R3 | **Keeping state never moves a cut.** State is kept at chosen cuts (5.2). Whether it is kept is policy; where waves end is not. |
| R4 | **A warm-up is a commit.** It computes to its last cut and keeps state there. It samples nothing. |
| R5 | **Nothing else moves a cut**: not history (the `seen` cache point goes), not load (a wave that cannot get its rows waits), not the request's class. |
| R6 | **Replies have one arithmetic.** Either a finished reply is committed in prefill arithmetic at once, during the tool pause, or a cold run recomputes replies with the decode kernels. The gate's measured cost per turn picks one per model. |
| R7 | **Concurrency.** Exactness under load also needs batch-invariant kernels, which is the open batch-invariance item in `TECHDEBT.md`. Background commits run alone, so they compute what a solo cold run computes. |

**Client marks.** A request's own `cache_control` marks become requests to keep state at those cuts. They never add a cut.

**Gate GX**, per served model, at 8K, 100K and 300K, with tails of 1, 40, 500 and 5,000 tokens:

- commit P, then P+T, against cold P+T;
- a three-turn session against a cold run of its last prompt;
- a chain evicted to NVMe and restored, against the uninterrupted chain.

Tokens and logprobs must be equal. `tools/i27_session.py` has the shared-prefix cases to extend. The release kit runs GX from then on.

### 5.2 A durable prefix store

PR #1949's rule (write a prefix only after something reads it) stays for ordinary tails. Three classes replace "unread means disposable":

| Class | What | Saved | Evicted |
| --- | --- | --- | --- |
| tail | Anything else | After its first read, as today | First, unsaved |
| kept | The newest prompt of every live session | When the session has been idle 120 s, before any eviction, and first at close | After tails, and never before it is saved |
| pinned | Members of the keep set (5.3) | At creation | Last, and never before it is saved |

**Every layer honours the class**, not only rank eviction:

- rank page eviction and pool shrink;
- the API's prefix index (protected entries);
- recurrent states: a kept or pinned state is exempt from the newest-4 rule and is spilled, not dropped, when the store is full;
- the snapshot store's quota eviction;
- the close-time save, which runs pinned, then kept, then newest tails, with a time limit that scales with what is unsaved.

**No deletion by age.** While the customer holds the fleet, no tier deletes cached state because it is old: not the device and host page caches, the weightd pools, the snapshot files, the API's prefix index, the batch engine's copy of a finished request's tokens, the draft servers' contexts or the serving layer's stores. Today each of them deletes what was unused for 24 hours. Cached state leaves in three ways only:

- **Capacity.** A full tier evicts in class order: tails, then kept, then pinned.
- **The customer's request.** `DELETE /v1/cache` empties every tier at once and fails loudly if one cannot be emptied. Today it leaves the engine's cache in place.
- **The end of the hold.** Working state is deleted when another customer's booking of the fleet starts, or 24 hours after the customer's last hour (Customer Agreement 6.2). Our administration software does this when it restores the fleet to the base image, whatever software the customer installed. The engine does not need to know the bookings.

The invariant "Cached request state expires" in `sparkpipe_invariants.md` is rewritten to this rule in the change that removes the age limit.

**State points.** State is kept at the last cut of every request, at commit ends, at cuts the client marks, at checkpoint anchors, and at opening cuts where a second session has branched. A saved prefix needs its state on NVMe, not in memory.

**Truth.** The engine answers, for a prefix identity: tokens resident, tokens saved, the nearest state point and the class. `GET /v1/state` and every "is it warm?" answer use that, never a flag in the serving layer.

**Restore before the request.** The keeper can ask for a saved chain to be made resident. This is the existing `CACHE_HINT` path, which still needs fleet proof.

**Releases.** The adapter declares a KV epoch, and the layout digest uses the epoch instead of the binary's hash. The release kit proves the epoch: the KV pages of its prompts must hash the same under the old and the new binary. Same epoch, and the saved cache is still valid after the release. A new epoch, and the keeper rebuilds the keep set from kept prompts.

**Kept prompts.** An entry that a Function keeps (resume, named, pack) stores its token ids on the fleet, as working state in Customer storage, so it can be rebuilt after a new epoch. Entries the engine keeps by itself (live sessions and openings) store none: after a new epoch they are computed again on first use. Nothing else stores prompt tokens.

**Writes.** A kept session writes one state when it goes idle: 454 MB across the 16 nodes for Kimi K3.1, far less elsewhere. Saves are rate-limited so they cannot stall a wave.

### 5.3 The keep set and the keeper

The keep set is a short list of prefixes that are known exactly. The keeper is the scheduler in the engine's serving layer that maintains it.

| Kind | Known exactly because | Kept how |
| --- | --- | --- |
| live | It is a running session's last prompt | Class kept, automatically |
| resume | It is the last prompt of a session that is no longer live | Pinned on NVMe; made resident on a signal or a schedule |
| opening | Two sessions with the same client, tool set and project context branched from it | Pinned; a state at its last cut. One per subagent type as well |
| named | The customer sent it: `POST /v1/cache/warm`, `sparkpipe warm FILE`, a pin | Pinned |
| pack | We built it (5.5) | Pinned, with its delta |

A session is live from its first request until it has been idle for 30 minutes, a setting. It then joins the resume set and stays there: nothing leaves the keep set because of its age. The set is bounded by its budgets alone, and when they are short the entries that save the fewest recompute seconds per gigabyte go first. Before the day starts, the night run makes the sessions used most recently resident.

**Automatic, or a Function.** The engine does this by itself, as part of its cache: exact reuse, keeping live sessions, learning openings, honouring a client's own commit, and restoring on use. A Function the customer starts or schedules adds the rest: Resume sessions (the resume set, making it resident before the day, rebuilding it after a new epoch), Prefill ahead (named entries), Build and refresh the cache (packs) and Anticipate tomorrow (the night run and its report).

**Events.**

| Event | Seen by | Action |
| --- | --- | --- |
| A request with a session id | Serving layer | Mark the chain live; record the miss cause (5.6) |
| A session idle for 120 s | Timer | Save its tail and state |
| Codex `generate: false` | WebSocket | Commit the offered prefix |
| `max_tokens: 0` on Messages, `commit_only` elsewhere | API | Commit |
| Claude Code session start or resume | A hook that setup installs | Make that session's chain and opening resident |
| A subagent class or agent type header | Header | Keep that type's opening |
| A compaction request, then `context-compacted` | Headers | Keep the opening resident; release the old chain |
| A push to the fleet's git remote | Git route | Rebuild the pack delta and commit it per profile; queue a re-base if the delta is large |
| A model about to be swapped out | Swap controller | Save the keep set first; restore openings and packs when the model returns |
| An engine start | Lane start | Same epoch: make openings and packs resident; sessions restore on use. New epoch: rebuild from kept prompts |
| The night window | Schedule | Re-base packs, rebuild what a release invalidated, make tomorrow's resume set resident before the day starts, write the morning report |

**Scheduling rules.**

- Background work runs only when no interactive request is queued or running on the lane, one job at a time, and gives way between waves.
- The bound on an interactive request is one background wave. Where one wave is too long (GLM 5.3 Full at long context), background commits stop at a context limit until a wave can be aborted.
- The keeper never activates or swaps a model. An entry for a model that is not resident waits.
- A commit the client asks for at a session's start is that session's own work and runs in the interactive class. Only the keeper's own work is background.
- A prefix is computed once. A request whose prefix another request is already computing waits for it to be published, so eight subagents that start together on a cold opening cost one prefill, not eight.
- A saved prefix is restored, never recomputed. On Kimi K3 a restore is 200–500 times cheaper per token than prefill.
- When the budget is short, openings are kept first, then packs, then sessions ranked by the recompute seconds they save per gigabyte. Entries are dropped from the other end.
- Each data-parallel lane has its own cache, so the keep set is per lane and a session stays on its lane.

**Budgets** are settings with defaults: a share of device pages for pinned entries, a number of state records, a share of the NVMe quota, and the night window.

### 5.4 Client signals and setup

**Claude Code**

- Keep system blocks apart, and strip a first block that starts with `x-anthropic-billing-header:` (setup already turns it off; a customer who sets the base URL by hand does not).
- Read `x-claude-code-agent-type`, `x-claude-code-compaction` and `x-claude-code-context-compacted` as well as the request class.
- Report `cache_creation_input_tokens` truthfully.
- Setup installs a session-start hook that calls the keeper, and sets the compaction window the client really honours when model names are mapped (to verify on a live client).

**Codex**

- Setup enables the WebSocket transport. `generate: false` becomes a commit, and later turns can send deltas with `previous_response_id`.

**Harnesses that rewrite history** (observation masking, pruning) defeat any exact cache. Setup turns off the ones it knows, and the miss ledger names the position where a client changed its own history.

**Tool lists that change mid-session** (an MCP server that connects late) show in the ledger the same way.

### 5.5 Repository packs, restructured

**What a pack is for.** Work that wants the repository in the prompt: questions over the whole codebase, reviews, second opinions, and agent sessions on a model where the measured trade is positive. It is an option with published figures, not the default for agent sessions.

**One base per repository and model.** No trees, no leaves.

- **Stable slots.** A file gets its place when it first enters the pack, coldest tier first, and keeps it. New files are appended. Nothing is re-sorted on a push.
- **Delta.** A push does not touch the base. Changes since the base commit go into a second block after it: a unified diff per changed file, or the whole file when it is new or small. A new session gets base plus the current delta. A running session keeps the pair it started with.
- **Re-base.** When the delta passes a size limit, or at night, a new base is built. It reuses the old base's KV up to the first slot whose file changed, and hot files are last.
- **Placement.** Ahead of the tool block wherever a family's template allows it, so one copy serves every tool set. Where it does not, each profile is a copy and budgets count it. The effort header is fixed per pack session.
- **Branches.** The delta is computed against the session's branch when the client tells us its commit (Codex sends it; for Claude Code the hook does), otherwise against the default branch. Unpushed edits reach the model through its own reads; the pack header says which commit it shows.
- **Selection.** `sparkpipe setup --pack` in a project writes that project's alias. No model name is edited by hand, and a request for an unbuilt pack is served without the pack.
- **Memory and packs** combine in a defined order, or the response says which one was used.

**Kept from #1820:** the git remote, the build, per-file tokenization, version ids and per-session version pinning.

**Measured before it is offered for agent sessions:** task success, turns and wall time with and without the pack, per model, on the agent task evaluation.

### 5.6 Telemetry

**Miss ledger.** For every request, on all three API shapes, the fleet records: prompt tokens; tokens matched; the cut it resumed at; tokens recomputed; restore time; time spent waiting behind a background wave; and a cause for everything prefilled beyond the new tail:

- first sight of an opening, or an opening that changed (which block);
- session restored from NVMe;
- session lost (evicted, restart, new epoch, swap);
- compaction;
- history changed by the client, with the position;
- a new pack profile;
- model not resident.

No content is recorded.

**`GET /v1/state` and `sparkpipe status`** show the keep set with engine truth, the budgets and the last night run.

**Morning report.** Yesterday's cold seconds by cause, what the keeper did overnight, and what it could not keep.

**Benchmark arms** added to `tools/agent_turn_bench.py`: typical turns (tails of 50, 300 and 2,400 tokens at 20K to 300K of context) and each cold event of section 3.

### 5.7 Where it runs

All of it runs on the renter's fleet, as part of the engine the renter installs. Repositories, packs, kept prompts, the keep set, the miss ledger and the reports are working state in Customer storage, and none of them leaves the fleet. The serving configuration that assumes a shared hub behind our gateway (a loopback bind behind the gateway's door, account scopes taken from gateway headers, hub paths) is replaced by the renter's own keys and the fleet's own paths.

### 5.8 Considered and not built

| Idea | Why not |
| --- | --- |
| Nightly leaves of predicted file combinations | Read sets across tasks overlap at Jaccard 0.03–0.21; a leaf is reused only by a session that starts with it; the count grows with tool sets and effort levels |
| Warming files, or a file's call-graph neighbours, on a save or a checkout | A file's KV depends on everything before it. Reads land at the end of a conversation, so free-standing file KV is never a prefix. The events are right; the thing to warm is a session, an opening or a pack delta |
| Stitching per-file KV at query time (CacheBlend, EPIC, KVLink) | Not exact. Impossible where layers carry recurrent or compressor state: Kimi K3.1, Qwen 3.8, GLM 5.3 Flash, DeepSeek V4.1 Flash. In the KVLink authors' re-run on Llama 3.1 8B, CacheBlend scored 6.0% on the multi-hop MuSiQue set where full encoding scored 49.1%. An MLA latent has no position in it, but it is still a function of every earlier token. It could only ever be a separately measured variant on GLM 5.3 Full |
| Replacing a tool result with a reference to the pack's copy | It changes what the model reads. Claude Code shipped such a stub for unchanged re-reads and removed it. Possible later as a measured variant |
| Prefilling a predicted tool result during the tool pause | The pause is 0.1–0.4 s at the median, and Codex output starts with timing values. Revisit if the ledger shows long pauses before large reads on the slow models |
| Predicting tomorrow's date inside a client's prompt | Not needed. Codex warms its own opening at session start; Claude Code puts the date after the first prompt |

## 6. What it buys

Cold times below are tokens divided by release c2b714343's measured prefill rate on a 7K prompt. Long prompts run slower on several models, so these are lower bounds. The DeepSeek V4.1 Flash row uses main's lane (130,819 tokens in 81.8 s).

| Model | Rate, tok/s | New session opening, 46K | Subagent opening, 37K | Back to a 124K session |
| --- | ---: | ---: | ---: | ---: |
| Kimi K3.1 Pro | 490 | 95 s | 76 s | 253 s |
| Kimi K3 | 709 | 65 s | 52 s | 175 s |
| Qwen 3.8 Max | 858 | 54 s | 43 s | 145 s |
| GLM 5.3 Full | 1,260 | 37 s | 29 s | 98 s |
| DeepSeek V4.1 Flash | 1,599 | 29 s | 23 s | 78 s |
| GLM 5.3 Flash | 2,112 | 22 s | 18 s | 59 s |
| Qwen 3.8 27B | 3,286 | 14 s | 11 s | 38 s |

With the design each of these becomes a restore plus the new tail. The one restore measured so far is Kimi K3 after a model swap: 137,600 tokens in 2.7 s. The other models' restore rates are measured in stage 0.

Per-turn time inside a session does not change with this design. Stage 0 measures the turn floor of section 3. If it is still seconds per turn at long context on the Kimi models, cutting it is separate engine work and is worth more than any single row above.

## 7. Build order and gates

| Stage | Work | Gate |
| --- | --- | --- |
| 0. Measure | The miss ledger and engine truth; benchmark arms; per model: time to first token against context and tail size, decode rate against context, restore and save rates, wait behind a background wave | A `PERFORMANCE_STATUS.md` section with those numbers on the current release |
| 1. Exactness | R1–R7; the commit request with state at its end; client marks as state points; the `seen` cache point removed | GX passes on every served model and joins the release kit |
| 2. Durable store | No deletion by age in any tier, and `DELETE /v1/cache` through every tier, first; then the three classes in every layer; idle and close saves; state rules; index protection; restore hints proven; warm-ups that cannot swap | A session left on Friday is a hit on Monday. A chain survives eviction, a swap away and back, a clean restart and a kill, and continues bit-equal (I27). After `DELETE /v1/cache` a scan of every node finds nothing |
| 3. Keeper and signals | The keep set (live, resume, opening, named); Codex warm-up; Claude Code headers, hook and setup; `POST /v1/cache/warm`, pins, `GET /v1/state`; the CLI | The day table below, per model, with Claude Code and Codex |
| 4. Releases | KV epoch proven by the release kit; rebuild from kept prompts | A release with an unchanged epoch keeps every pinned prefix; one with a new epoch rebuilds them in order |
| 5. Packs | Stable slots, delta, re-base, placement, selection; the whole-repository Functions on top | A push recomputes only the delta; a re-base reuses the base up to the first changed slot; the agent-session comparison is published |
| 6. Night run | The keeper's night window and the morning report | The report's cold seconds match the ledger |

Stage 0 and stage 1 can start together. Stage 5 needs stages 1 and 2 only.

**The day table (gate for stage 3).**

| Moment | Scenario | Pass when |
| --- | --- | --- |
| 09:00 new session | 14 hours after the last request | Only the first prompt, and any project context that changed, is prefilled |
| 09:05 resume | Yesterday's 150K session resumed | First token after a restore plus the new tail; nothing recomputed before the last cut |
| 10:45 after a meeting | 105 minutes idle | Same |
| Monday 09:00 | Friday's session resumed after 64 hours | Same |
| Subagents | 8 started together | Each prefills its own brief only |
| After a swap | Another model, then back | Same as resume |
| After a restart | Clean, and killed | Same as resume |
| 14:10 other branch | A new session on another branch | The opening is a hit; with a pack, base plus that branch's delta |
| Under background work | A request arrives during a commit | It waits at most one wave |

**Launch issues.** #1792 becomes stages 1–3. #1789's engine rule (kept while the fleet is held, deleted on request) is the first part of stage 2. #1791's "survives releases and restarts" is stages 2 and 4. #1800 is stage 5. #1801 and Gate D2 become stage 6. The client hook belongs to #1788.

## 8. What happens to existing work

The owner gave the go-ahead for these on 21 November 2026 (decision 3).

- **#1821** is closed without a rebase. Its request and response shapes (`commit_only`, the `cache.commit` object, the three warm formats), its `GET /v1/state` outline, its pin test, its warm-gate check tool, its cold-prefill seeds and its per-request draft counts are carried into stages 1–3.
- **`tools/serving/context_packs.py`** keeps what section 5.5 lists and loses the re-sort, the per-profile guessing and the belief-based `warm` state, in stage 5.
- **`tools/anticipator.py`** is removed. It was the leaf planner (candidates, tree, scoring), the command that read agent logs, and a settings, approval and report shell that handled leaf plans only. The night run's settings, approval mode and report are specified in `docs/FLEET_API.md` §5 and built in stage 6. The receipt of its one run stays in `qualification/anticipator/20261105/` (decision 4).
- **`docs/ANTICIPATOR.md`** goes with the tool, and **`docs/FLEET_API.md` §4–5** are rewritten to this design.
- **Issues #1792, #1800 and #1801** are rewritten to the stages of section 7.
- **#1691** (working state, draft) is not part of the go-ahead and is untouched. Its rule kept cached state until 24 hours after the later of its last use and the held hours, through a directive every tier reads. Under decision 1 the engine has no age rule, so the held hours need not reach the engine. What stage 2 takes from it is the deletion on request through every tier and its tests.

## 9. Decisions

The owner decided these on 21 November 2026.

1. **Expiry.** "As long as the fleet lease continues, there should be no time based purging." No tier deletes cached state because of its age while the customer holds the fleet (section 5.2). This goes further than the option the review recommended, which exempted only kept prefixes.
2. **Site wording.** The four Function lines and the day table on the home page are aligned with this design:
   - Resume sessions keeps sessions through nights, weekends, swaps, restarts and engine updates.
   - Prefill ahead keeps what the customer names.
   - Build and refresh the cache builds the pack and recomputes only what changed after a push.
   - Anticipate tomorrow is the night run and its report.
   - The day table's "4.7–5.1 s" stays until stage 0 measures the current release.
3. **Go-ahead for section 8**: close #1821, rewrite issues #1792, #1800 and #1801, and retire the leaf planner.
4. **The #1768 receipt** (`qualification/anticipator/20261105/`), built from agent session logs on a workstation, stays in the tree.

## 10. Sources and method

**Repository, at main `c2b714343`:** `cache/kv_page_cache.c`, `cache/prefix_cache.c`, `cache/kv_snapshot.c`, `runtime/model_batch_engine.c`, `runtime/stage_kv_binding.c`, `node/model_api.c`, `tools/serving/chat_frontend.py`, `tools/serving/context_packs.py`, `tools/serving/chat_shapes.py`, `tools/anticipator.py`, `tools/sparkpipe_cli.py`, `tools/stage_lane.py`, `deployment/lanes.json`, `sparkpipe_invariants.md`, `TECHDEBT.md`, `PERFORMANCE_STATUS.md`, `docs/FLEET_API.md`, and the release receipts in `qualification/releases/c2b714343/`. Capacity figures (state records, index entries, bytes per token) are computed from the lane formulas, not measured.

**Pack order simulation:** this repository's own history, daily snapshots from 21 October to 19 November 2026, tokens estimated as bytes divided by 3.5.

**Public traces** (figures are aggregates; no trace text is reproduced):

- nvidia/Open-SWE-Traces v1.2 (CC BY 4.0): Claude Agent SDK, OpenCode and OpenHands sessions.
- nebius/SWE-rebench-openhands-trajectories (CC BY 4.0).
- JetBrains-Research/the-complexity-trap (Apache-2.0): SWE-agent and OpenHands with per-request prompts and engine cache counters.
- harborframework/terminal-bench-2-leaderboard: one Claude Code run.
- trace-commons/agent-traces (CC BY 4.0): 28 donated Claude Code sessions.

The models that produced these traces are not the models the fleet serves, and no Codex traces were available.

**Timing model for the shares in section 3:** prefill at 2,600 tok/s for Qwen 3.8 27B, and for Kimi K3.1 a 2,048-row wave of 3.94 s plus 1.36 ms per 1K tokens of context; decode at 60 and 16 tok/s. Decode is held constant, and tool time, queueing and concurrency are left out.

**Client behaviour**, read on 21 November 2026: the Claude Code documentation (prompt caching, gateway protocol, sub-agents, hooks, model configuration) and changelog; the Codex source (openai/codex at `7edf91b13`); user measurements in anthropics/claude-code issues 94417 and 74318.

**Prior art:** SGLang RadixAttention (arXiv 2312.07104); Marconi (arXiv 2411.19379); the DeepSeek API's cache units; CacheBlend (arXiv 2405.16444), EPIC (arXiv 2410.15332), KVLink and its re-run of the others (arXiv 2502.16002); DeepSeek-V2's MLA (arXiv 2405.04434); Continuum (arXiv 2511.02230); Anthropic's prompt-caching documentation, which defines the `max_tokens: 0` warm-up.

**Not used:** any agent session log from a developer's machine.
