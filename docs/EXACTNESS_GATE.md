# Gate GX: reuse is exact

**Contract.** The tokens and logprobs a request produces depend on the
request, never on what the cache held. This extends invariant I27 from
"restored equals uninterrupted" to "resumed equals cold".

`tools/gx_gate.py` checks it on a live engine API, and
`scripts/gx_gate.sh` walks a served model through it with the swap
controller. Both need the fleet alone.

## How the engine keeps it

Prefill runs in waves, and kernels take different paths by wave size. So a
request's tokens are fixed only when its waves are fixed. The engine derives
them from the request and from nothing else.

- **Cuts.** A wave ends at the next of: the row budget (whole blocks), a
  checkpoint anchor, a cut the request names, one token before the prompt's
  end, and the prompt's end. A request names its cuts in `cuts`: token
  counts where its structure ends a wave. The chat layer names one token
  before each replayed reply began and where it ends.
- **Resume.** A request resumes only at a wave end of its own cold run, and
  only where every wave up to there is the same as in the run that left the
  state. The prefix index stores a tag of the waves with each resume point,
  and a block's cache identity names the waves that computed it, so two
  runs share a block only when they computed it alike.
- **State.** State is kept at anchors, one token before the prompt's end,
  at a commit's end and at cuts the request lists in `keep`. Keeping state
  never moves a wave end.
- **Commit.** `commit_only` computes a prompt to its end, keeps the state
  there and samples nothing. It is the warm-up.
- **Replies.** The pages a reply's decode steps write belong to that request
  alone. A later prompt that carries the reply computes it again as prefill,
  from the earlier prompt's last cut.
- **Load.** A wave the ranks refuse for capacity waits and goes out again
  with the same span.
- **Streamed prefill.** A prompt that takes the streamed prefill is computed
  whole every time and shares nothing, because its arithmetic is not the
  wave arithmetic. No lane in `deployment/lanes.json` streams.

Under concurrency the rows of other requests share a wave, and kernels are
not yet batch-invariant (`TECHDEBT.md`). The gate runs alone.

## What is compared

Every prompt is a run of token ids that the engine has never seen: they are
derived from a seed that is new for each state. A unit is one last prompt,
and it runs three times.

| Arm | Cache when the last prompt runs | Step |
| --- | --- | --- |
| chain | Holds the unit's own earlier requests, in memory | `chain` |
| restore | Holds them again after the engine went down and came back with its device pool dropped | `park:GROUP`, then `restore:GROUP` |
| cold | Holds nothing of the unit | `cold:TAIL` |

The last prompt asks for 32 tokens, greedy, one step at a time, with the
top 4 logprobs.

| Case | Earlier requests | Last prompt |
| --- | --- | --- |
| `prefix` | A commit of the prefix P | P and a new tail, with a cut at P's end |
| `kept` | A one-token completion of P and eight other tokens, with a cut at P's end and state kept there | P and a new tail, with the same cut and keep |
| `session` | Turn 1 (opening and first prompt) and turn 2 (plus reply 1 and a tail), each answered with 96 tokens as decode chains, speculation off | Turn 3: plus reply 2 and a tail |
| `drafted` | The same with speculation left at the engine's default | The same |

A session's turns name their cuts as the chat layer does: where the opening
ends, one token before each reply began and where each reply ends. The last
prompt is the same request in every arm, cuts included.

Two verdicts follow for each unit:

- **reuse**: chain against cold.
- **restored**: restore against chain.

A verdict is `PASS` only when tokens and logprobs are equal, the reusing run
resumed at the cut its structure names (P's end, or one token before reply 2
began), and the cold run reported no cached tokens. The receipt also holds
the first differing token, the first differing logprob entry and the largest
logprob difference up to there.

## Sizes

- Contexts: 8,000, 100,000 and 300,000 tokens for `prefix` and `session`;
  8,000 and 20,000 for `kept`; 8,000 for `drafted`.
- Tails: 1, 40, 500 and 5,000 tokens. A session's three tails have the
  unit's size.
- A prefix ends 21 tokens past its nominal size, in the middle of a block.
  A `kept` prefix ends one token past a multiple of 256 that is 768 past
  a multiple of 1,024: one token into a block for every model, and never on
  the row grid or a checkpoint anchor.
- A context the deployment cannot serve beside three 5,000-token tails is
  fitted to the largest that it can. It is skipped when that is under 1.5
  times a context already in the plan.

## Steps and cache changes

`gx_gate.py plan` writes the state and prints the steps with the cache
change each needs before it:

```
chain none
park:20000 clear
restore:20000 evict
park:100000 none
restore:100000 evict
park:300000 none
restore:300000 evict
cold:1 clear
cold:40 clear
cold:500 clear
cold:5000 clear
crowd clear
```

The `crowd` line appears only when the plan asks for it.

Prefixes of up to 32,768 tokens share one `park` and `restore` step, named
after the largest of them. A longer prefix has its own, so the states that
must survive one eviction stay few.

- **clear**: the controller stops every model, the lane's `kvsnapshot` and
  `kvcache` directories are emptied on all sixteen nodes, the hub API's
  `runtime/prefix_index.spi` is removed, and the controller starts the model
  again. The driver refuses to remove anything while an engine runs.
- **evict**: the stop and the start alone. The engines save their chains as
  they close, and the controller drops every device KV pool, so the next
  read comes from NVMe.

Cold runs take one cache lifetime per tail size. The units of one tail size
start from different prefixes, so none of them finds another's KV. Units of
different tail sizes share a prefix, so they cannot share a lifetime.

Inside a step the order is fixed. `chain` commits the prefix, runs its tails,
then runs the sessions. `park` runs the sessions' first two turns, then
commits the prefix, so the commit that nothing has read is the newest chain
when the engine closes.

## Requests sent together

`--crowd N` (`GX_PLAN="--crowd 4"`) adds a last step, `crowd`, after a cache
clear. It commits each prefix alone, then sends that prefix's tails N at a
time, so their tails share prefill waves and their replies share decode
steps. `verdict` compares each with the same request from the `chain` step,
which ran alone from the same committed prefix, and prints
`concurrent equal=… different=… not-run=…`.

This is measured, not judged: it does not change the verdict. Kernels take
different paths by the rows in a wave, so a request's tokens can depend on
the requests beside it until the batch kernels are invariant
(`TECHDEBT.md`, Dynamic batching). The receipt holds the first differing
token and the largest logprob difference for each unit.

## What stops a PASS

| Outcome | Meaning |
| --- | --- |
| `NOT-RUN` | An arm is missing |
| `NOT-COLD` | The cold run reported cached tokens, or ran another prompt |
| `NOT-CACHED` | The chain arm's last prompt reported no cached tokens |
| `SHORT` | The chain arm's tokens and logprobs equal the cold run's, but it resumed before the cut its structure names, so the reuse under test did not happen |
| `LOST` | The restore arm's last prompt reported no cached tokens: the chain did not come back |
| `DIVERGED` | The replayed session produced other replies, so its last prompt is another prompt |
| `ERROR` | A request of the unit failed while the engine stayed whole; the receipt holds the engine's answer |
| `alone=no` | The API served a request the gate did not send |
| a step not `done` | It stopped, or was never run |

Each of these leaves the verdict `INCOMPLETE`. A step that needs a cold run
and finds cached tokens stops at once.

`recomputed_after_restore` is the number of tokens the restore arm computed
again that the chain arm reused. Above zero, part of the chain or one of its
states did not come back.

## Interruptions

A block is one model's run, and it can be run again after anything stops it.

- **A request fails and the engine stays whole** (every rank connected,
  health `ok`): the unit is marked `ERROR` with the engine's answer and the
  step goes on. A prefix that does not commit fails its own units only.
- **The engine is not whole, or a cold run finds cached tokens**: the step
  stops. Every finished unit is already in the state file, which is written
  after each unit.
- **Run again** (`gx_gate.sh` on the same `GX_DIR`): finished steps are
  skipped. The cache is cleared before the first open step, because nothing
  is known about what the interruption left in it. A `chain` or `cold` step
  then does only its open units; a chain's shared prefix is committed again
  for them. A block that resumes at a `restore` step first runs its `park` step
  again, for the units the restore step still needs, and then evicts.
- A step still running on the hub is left alone: the driver refuses to start
  beside it.
- `GX_REST` idles that many seconds after every request longer than
  `GX_REST_OVER` (60) seconds.

An attempt that the engine cut short cannot read the API's served count
afterwards. It reads the engine's completed count from the API log instead,
and counts as alone unless that shows another client's request.

## Running it

```
GX_RELEASE=<sha> GX_OUT=qualification/<dir> scripts/gx_gate.sh MODEL...
```

- In a release, `scripts/release_rollout.sh gx` runs it after `qualify` for
  the served models (`GX_MODELS`; `ONLY` limits them), keeps its hub files in
  the release's work directory, copies receipts to the kit state's `gx`
  directory and leaves `qwen3.8-max` active. It fails unless every model
  passes.
- State and logs are kept on the hub under `$HUB_WORK/gx/<sha>/<model>/`.
  A state is planned once: its prompts must be new to the cache. Running
  the driver again on a directory that holds a state resumes it.
- `GX_PLAN` passes plan options, for example
  `--contexts 8000 --tails 1,40`.
- `GX_END` names the model to leave active; the default leaves none.
- Each step runs detached on the hub, so a dropped connection does not end
  it.
- The receipt `<model>.json`, the verdict table `<model>.txt` and the run
  log `<model>.log` are copied to `GX_OUT`.

## What a receipt measures

The receipts are also the fleet's turn-floor record, so no second matrix has
to run for it.

Per request (`steps.<step>.requests[]`):

| Field | Meaning |
| --- | --- |
| `label`, `attempt`, `max_tokens` | The unit and turn, and which attempt of the step sent it. A commit has `max_tokens` 0 and `finish_reason` `commit` |
| `cuts`, `keep` | The cuts the request named and where it asked for state |
| `prompt_tokens`, `cached_tokens`, `prefilled_tokens` | The prompt, the position the request resumed at, and what it computed |
| `seconds` | Wall time at the client |
| `queue_ms`, `first_token_ms`, `last_token_ms` | From the engine's own request record: accepted to first dispatch, to the first token and to the last |
| `decode_tokens_per_second` | Over the tokens after the first. Session replies are 96 tokens of chain decode |
| `completion_tokens`, `finish_reason`, `stale_prefix_recomputes` | |

Per attempt of a step, before and after: the running engine's KV store
counters summed over its ranks (pages and nanoseconds saved and restored,
restore misses, store files and bytes, device pool pages and bytes), the
slowest rank's save and restore time, and the engine's prefix counters (hits,
misses, hit tokens, stale-prefix recomputes, refused waves, draft tokens).
The attempt also names the engine: adapter id, model revision and driver
artifact SHA-256.

- **Cold fill at a context**: the `*/warm` requests (a commit's `seconds`
  is its prefill) and everything in `cold:*`.
- **A turn on a kept prefix**: `prefix/<context>/<tail>` in `chain`. Each
  unit's verdict names `resume_cut`, the cut the reuse should resume at.
- **A turn of a session**: `session/<context>/<tail>/turn2` and the last
  prompt in `chain`. Each computes the previous reply again as prefill, so
  its `prefilled_tokens` are the reply plus the new tail plus one.
- **After an eviction**: the same requests in `restore:*`. Their extra
  `first_token_ms` against `chain` is the restore the user waits for.
  Restored bytes are restored pages times pool bytes per pool page. A
  `restore` step with no restored pages did not read from NVMe.
