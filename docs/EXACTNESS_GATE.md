# Gate GX: reuse is exact

**Contract.** The tokens and logprobs a request produces depend on the
request, never on what the cache held. This extends invariant I27 from
"restored equals uninterrupted" to "resumed equals cold".

`tools/gx_gate.py` checks it on a live engine API, and
`scripts/gx_gate.sh` walks a served model through it with the swap
controller. Both need the fleet alone.

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
| `prefix` | A one-token completion of the prefix P | P and a new tail |
| `declared` | The same | The same, declaring a cache point one token before P's end, which is a block boundary off the wave grid. A cold run then cuts its waves where the reusing run resumes |
| `session` | Turn 1 (opening and first prompt) and turn 2 (plus reply 1 and a tail), each answered with 96 tokens as decode chains, speculation off | Turn 3: plus reply 2 and a tail |
| `drafted` | The same with speculation left at the engine's default | The same |

Two verdicts follow for each unit:

- **reuse**: chain against cold.
- **restored**: restore against chain.

A verdict is `PASS` only when tokens and logprobs are equal, the reusing run
reported cached tokens, and the cold run reported none. The receipt also
holds the first differing token, the first differing logprob entry and the
largest logprob difference up to there.

## Sizes

- Contexts: 8,000, 100,000 and 300,000 tokens for `prefix` and `session`;
  8,000 and 20,000 for `declared`; 8,000 for `drafted`.
- Tails: 1, 40, 500 and 5,000 tokens. A session's three tails have the
  unit's size.
- A prefix ends 21 tokens past its nominal size, in the middle of a block.
  A `declared` prefix ends one token past a multiple of 256 that is 768 past
  a multiple of 1,024: a block boundary for every model, and never a wave
  boundary of a cold run or a checkpoint anchor.
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
```

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

Inside a step the order is fixed. `chain` warms the prefix, runs its tails,
then runs the sessions. `park` runs the sessions' first two turns, then
warms the prefix, so the warm-up that nothing has read is the newest chain
when the engine closes.

## What stops a PASS

| Outcome | Meaning |
| --- | --- |
| `NOT-RUN` | An arm is missing |
| `NOT-COLD` | The cold run reported cached tokens, or ran another prompt |
| `NOT-CACHED` | The chain arm's last prompt reported no cached tokens |
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
  step goes on. A prefix that does not warm fails its own units only.
- **The engine is not whole, or a cold run finds cached tokens**: the step
  stops. Every finished unit is already in the state file, which is written
  after each unit.
- **Run again** (`gx_gate.sh` on the same `GX_DIR`): finished steps are
  skipped. The cache is cleared before the first open step, because nothing
  is known about what the interruption left in it. A `chain` or `cold` step
  then does only its open units; a chain's shared prefix is warmed again for
  them. A block that resumes at a `restore` step first runs its `park` step
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
| `label`, `attempt`, `max_tokens` | The unit and turn, and which attempt of the step sent it |
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

- **Cold fill at a context**: the `*/warm` requests and everything in
  `cold:*`.
- **A turn on a kept prefix**: `prefix/<context>/<tail>` in `chain`.
- **After an eviction**: the same requests in `restore:*`. Their extra
  `first_token_ms` against `chain` is the restore the user waits for.
  Restored bytes are restored pages times pool bytes per pool page. A
  `restore` step with no restored pages did not read from NVMe.
