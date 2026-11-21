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
| restore | Holds them again after the engine went down and came back with its device pool dropped | `park:CONTEXT`, then `restore:CONTEXT` |
| cold | Holds nothing of the unit | `cold:TAIL` |

The last prompt asks for 32 tokens, greedy, one step at a time, with the
top 4 logprobs.

| Case | Earlier requests | Last prompt |
| --- | --- | --- |
| `prefix` | A one-token completion of the prefix P | P and a new tail |
| `declared` | The same | The same, declaring a cache point one token before P's end, which is a block boundary off the wave grid. A cold run then cuts its waves where the reusing run resumes |
| `session` | Turn 1 (opening and first prompt) and turn 2 (plus reply 1 and a tail), each answered with 48 tokens as decode chains, speculation off | Turn 3: plus reply 2 and a tail |
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
| `alone=no` | The API served a request the gate did not send, or a step failed |

Each of these leaves the verdict `INCOMPLETE`. A step that needs a cold run
and finds cached tokens stops at once.

`recomputed_after_restore` is the number of tokens the restore arm computed
again that the chain arm reused. Above zero, part of the chain or one of its
states did not come back.

## Running it

```
GX_RELEASE=<sha> GX_OUT=qualification/<dir> scripts/gx_gate.sh MODEL...
```

- State and logs are kept on the hub under `$HUB_WORK/gx/<sha>/<model>/`.
  A state is planned once: its prompts must be new to the cache.
- `GX_PLAN` passes plan options, for example
  `--contexts 8000 --tails 1,40`.
- `GX_END` names the model to leave active; the default leaves none.
- Each step runs detached on the hub, so a dropped connection does not end
  it.
- The receipt `<model>.json`, the verdict table `<model>.txt` and the run
  log `<model>.log` are copied to `GX_OUT`.

Every request's record holds its prompt length, cached tokens, wall seconds
and, from the API log, its queue time, time to first token and stale-prefix
recomputes. The warm-ups and the cold runs are cold fills of each context,
and the tails are the cost of a turn at that context.

Each step also records, before and after, the running engine's KV store
counters summed over its ranks (pages and nanoseconds saved and restored,
restore misses, store files and bytes, pages adopted from the device pool)
and its prefix counters (hits, misses, hit tokens, stale-prefix recomputes,
refused waves, draft tokens). A `restore` step with no restored pages did
not read from NVMe.
