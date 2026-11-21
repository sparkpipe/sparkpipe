# Gate GX: reuse is exact

**Contract.** The tokens and logprobs a request produces depend on the
request, never on what the cache held. This extends invariant I27 from
"restored equals uninterrupted" to "resumed equals cold".

`tools/gx_gate.py` checks it on a live engine API, and
`scripts/gx_gate.sh` walks a served model through it with the swap
controller. Both need the fleet alone.

## Where it runs

Gate GX belongs to release qualification, on a fleet that holds no customer
data. Its `clear` deletes a lane's KV: the snapshot store and the backing
files on every node, and the API's prefix index. On a fleet a customer
holds, that KV is the customer's, so the gate never runs there. A request
from any other client leaves the receipt `INCOMPLETE`.

Fleet operations are not in this repository. Host roles, the hold that
gives a session the fleet and the release steps are in
`datacenter:operations/FLEET_RELEASE_RUNBOOK.md`. Read it before touching a
fleet host, and take the hold before the first fleet command of
[First runs](#first-runs).

Status: not yet measured on the fleet. No lane has been gated.

## How the engine keeps it

Prefill runs in waves, and kernels take different paths by wave size. So a
request's tokens are fixed only when its waves are fixed. The engine derives
them from the request and from nothing else.

- **Cuts.** A wave ends at the next of: the row budget (whole blocks), a
  checkpoint anchor, a cut the request names, one token before the prompt's
  end, and the prompt's end. A request names its cuts in `cuts`: token
  counts where its structure ends a wave. The chat layer names where the
  prompt's opening ends, where each block of the head ends (the tools, each
  system block, each block of the first user message but its last), where a
  context pack ends, one token before each replayed reply began and where it
  ends. A block's end is not a token boundary, so its cut is the last cache
  block at or before the tokens of the text through the block, tokenized by
  itself, less a margin (`tools/serving/prompt_cuts.py`); the gate's
  `opening` case takes its cuts from the same function.
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
  alone. After its last cut a request publishes nothing and names no block,
  and the ranks drop the blocks it sealed when its sequence is released, so
  what a turn leaves in the cache does not grow with its reply. A later
  prompt that carries the reply computes it again as prefill, from the
  earlier prompt's last cut. A client may start that work early: it
  commits the earlier prompt with the reply, naming one token before the
  earlier prompt's end as a cut, and the later prompt resumes where the
  reply ends. The waves are the same either way. The chat layer does this
  for a model with `reply_commit` set.
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
| `opening` | A one-token completion of another prompt whose head shares this one's first two blocks, with the head's cuts, and state kept at the second block's cut and the head's end | A head of three blocks and a new tail, with the same cuts and keep |
| `session` | Turn 1 (opening and first prompt) and turn 2 (plus reply 1 and a tail), each answered with 96 tokens as decode chains, speculation off | Turn 3: plus reply 2 and a tail |
| `drafted` | The same with speculation left at the engine's default | The same |
| `committed` | As `session`, and after each reply a commit of that turn's prompt with the reply, naming the turn's cuts and one token before the turn's prompt ended | The same |

A session's turns name their cuts as the chat layer does: where the opening
ends, one token before each reply began and where each reply ends. The last
prompt is the same request in every arm, cuts included.

An `opening` unit stands for a second conversation that shares its tools and
its first system block with an earlier one. Its head has three blocks: the
first ends a quarter of the way in, the second three quarters in. The cuts of
those two ends come from the chat layer's own rule
(`structural_cut` in `tools/serving/prompt_cuts.py`), so each lies a few
tokens before the end of the shared text, on a cache block; the third cut is
the head's end. State is kept as the chat layer keeps it: at the head's end,
and at the second block's cut, where a client's `cache_control` mark would
sit; the first block's cut has none. Every unit has a third block of its
own, so it can resume only at the second block's cut.

Two verdicts follow for each unit:

- **reuse**: chain against cold.
- **restored**: restore against chain.

A verdict is `PASS` only when tokens and logprobs are equal, the reusing run
resumed at the cut its structure names (P's end; for `opening`, the cut of
the head's second block; one token before reply 2 began; for `committed`,
where reply 2 ends), and the cold run reported no cached tokens. The receipt
also holds the first differing token, the first differing logprob entry and
the largest logprob difference up to there.

## Sizes

- Contexts: 8,000, 100,000 and 300,000 tokens for `prefix` and `session`;
  8,000 and 20,000 for `kept`; 8,000 for `drafted`, for `committed`
  (`--committed-contexts`) and for `opening` (`--opening-contexts`).
- Tails: 1, 40, 500 and 5,000 tokens. A session's three tails have the
  unit's size.
- A prefix ends 21 tokens past its nominal size, in the middle of a block.
  A `kept` prefix ends one token past a multiple of 256 that is 768 past
  a multiple of 1,024: one token into a block for every model, and never on
  the row grid or a checkpoint anchor.
- A context the deployment cannot serve beside three 5,000-token tails is
  fitted to the largest that it can. It is skipped when that is under 1.5
  times a context already in the plan.

## Lanes and plans

`scripts/release_rollout.sh gx` runs every lane of `deployment/lanes.json`.

| Lanes | Plan |
| --- | --- |
| The seven of `GX_MODELS`: `kimi-k3.1-pro`, `kimi-k3`, `glm-5.3`, `glm-5.3-flash`, `qwen3.8-27b`, `qwen3.8-max`, `deepseek-v4.1-flash` | The full plan: the sizes above |
| The other fourteen | The short plan, `GX_SHORT_PLAN`: `--contexts 8000,limit --kept-contexts 8000 --tails 1,500`. Every case at 8,000 tokens, `prefix` and `session` also at the lane's context limit, and tails of 1 and 500 tokens |
| `GX_NOT_GATED` | None |

- `limit` in a plan is the context length the lane's API reports. It is
  fitted beside the tails like any other context.
- The two tails of the short plan take both kernel paths of a last wave:
  one row, and several blocks.
- The short plan computes a prompt of the lane's context limit cold eight
  times (two prefixes in `chain`, two in `park`, four cold runs), beside
  the 8,000-token units. Its time is those tokens at the lane's prefill
  rate.
- A lane that cannot run its plan is moved to `GX_NOT_GATED` in
  `scripts/release_rollout.sh`, and this table names it with the reason.
  The step prints the list. A lane in neither list gets the short plan, so
  a new lane is gated unless someone says why it is not.

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

Inside a step the order is fixed. `chain` warms each prefix (a commit; for
`kept` and `opening`, the other prompt), runs its tails, then runs the
sessions. `park` runs the sessions' first two turns, then warms the
prefixes, so the commit that nothing has read is the newest chain when the
engine closes.

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
| `ERROR` | A request of the unit failed while the engine stayed whole; the receipt holds the engine's answer. A request that waited out the in-flight budget on a wave the ranks kept refusing ends this way |
| `alone=no` | The API served a request the gate did not send |
| a step not `done` | It stopped, or was never run |

Each of these leaves the verdict `INCOMPLETE`. A step that needs a cold run
and finds cached tokens stops at once.

`recomputed_after_restore` is the number of tokens the restore arm computed
again that the chain arm reused. Above zero, part of the chain or one of its
states did not come back.

**Waves that waited.** A wave the ranks refuse for capacity is never made
smaller: it waits and goes out again with the same span, and a request whose
wave is refused for the whole in-flight budget fails. The receipt counts
both for each step under `load` (`rejected_waves_capacity`,
`retry_deadlines`), from the engine's counters before and after each
attempt, and the verdict table prints their sums. An attempt that starts on
an engine that has not yet reported counts from that engine's start.
`gx_gate.py check` reports refused waves as a finding and fails on a request
that waited out the budget.

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
- `GX_REST` (60) idles that many seconds after every request that ran
  `GX_REST_OVER` (60) seconds or longer. Each request's record says whether a
  gap followed it, so the gap never counts into a timing.

An attempt that the engine cut short cannot read the API's served count
afterwards. It reads the engine's completed count from the API log instead,
and counts as alone unless that shows another client's request.

## Running it

```
GX_RELEASE=<sha> GX_OUT=qualification/<dir> scripts/gx_gate.sh MODEL...
```

- A fresh block starts with a preflight (`GX_PREFLIGHT=0` skips it): one 8K
  prefix with one 40-token tail runs its `chain` step, a cache clear and its
  `cold` step in a state of its own, and `gx_gate.py check` runs the tool's
  own checks on it: every step done, the commit confirmed, each request
  found in the API log, logprobs for every token, the reuse resumed at its
  cut and the cold run cold. A difference between reuse and cold is a
  finding and does not stop the block; anything else does, before the
  block's plan is written.
- The block notes what the controller had active when it started, its lane
  record, API and port, the release, and every cache change with the
  seconds the stop and the start took and rank 0's last
  `kv snapshot store close` line. The receipt holds the notes under `block`.
- When the block ends, the model that was active before it is activated
  again and must answer a real request (`GX_END=none` leaves the fleet idle;
  a model id names another).
- `GX_BACKEND=lane` runs the gate on a lane staged with
  `tools/stage_lane.sh` instead of a served model: a group of four nodes, for
  example. The argument is the lane's `STAGE_MODEL`, and the lane's `STAGE_`
  settings are those it was set up with (`setup` and `api-setup` come
  first). The driver then never calls the swap controller. It refuses
  unless production is stopped and no engine runs on any node, starts the
  lane and its hub API itself, and changes the cache with the lane's own
  stop and start: `lane.sh drop-kv` on the lane's nodes for an eviction, and
  the lane's `kvsnapshot`, `kvcache` and prefix index removed as well for a
  clear. It ends with the lane and its API stopped and the lane reclaimed.
  Whoever holds the fleet restores production afterwards.
- A model's long contexts can run as a block of their own: give the second
  pass its own `GX_DIR` and `GX_PLAN="--cases prefix,session --contexts 300000"`.
- In a release, `scripts/release_rollout.sh gx` runs it after `qualify` for
  every lane ([Lanes and plans](#lanes-and-plans); `ONLY` limits them),
  keeps its hub files in the release's work directory, copies receipts to
  the kit state's `gx` directory and leaves `qwen3.8-max` active. It fails
  unless every lane passes.
- `GX_SHORT` names the models that get `GX_SHORT_PLAN` instead of
  `GX_PLAN`. The block's first note says which plan it ran (`plan`).
- The driver copies `tools/gx_gate.py` and the chat layer's cut rule,
  `tools/serving/prompt_cuts.py`, to `$HUB_WORK/gx/<sha>/`. A plan with an
  `opening` unit fails when the rule is not beside the tool.
- State and logs are kept on the hub under `$HUB_WORK/gx/<sha>/<model>/`.
  A state is planned once: its prompts must be new to the cache. Running
  the driver again on a directory that holds a state resumes it.
- `GX_MINUTES` starts no further step once that many minutes have passed
  since the driver began; the block is left resumable and the fleet is
  released as at any end.
- `GX_PLAN` passes plan options, for example
  `--contexts 8000 --tails 1,40`.
- Each step runs detached on the hub, so a dropped connection does not end
  it.
- The receipt `<model>.json`, the verdict table `<model>.txt`, the run log
  `<model>.log` and `<model>.apilog.tgz`, the slice of the API log each
  attempt of a step covers, are copied to `GX_OUT`. The API log itself does
  not outlive a release.

## First runs

The gate has not yet run on a real engine. These are the runs that come
first, in order, with their commands. `<sha>` is the nine-character commit
of the candidate. Every command runs from a checkout of it, in a shell that
has read the kit library, which defines `hub` and `HUB_WORK`. From step 2 on
they run under the hold.

```
. scripts/rollover_lib.sh
```

Cache identities changed format with these rules, so KV that an earlier
release saved is not found by this one. The gate's `clear` removes it lane
by lane; nothing has to be cleared by hand first.

**1. Build the candidate.**

```
REPO=$PWD RELEASE_OUT=<directory for the node files> scripts/release_build.sh <sha>
```

It builds the hub API in `$HUB_WORK/api-build-<sha>/build` and the node
files on the build host in `~/build-rp/fw-<sha>-src/build`, for the
adapters of `deployment/lanes.json`. A four-rank lane's adapter is not
among them and is built in both trees afterwards:

```
ssh ${BUILD_HOST:-lenovo7} 'cd ~/build-rp/fw-<sha>-src && PATH=/usr/local/cuda/bin:$PATH make -j8 build/libqwen38_27b_tp4_serving_adapter.so build/libdsv41_flash_tp4_serving_adapter.so'
hub "cd $HUB_WORK/api-build-<sha> && PATH=/usr/local/cuda/bin:\$PATH make -j16 build/libqwen38_27b_tp4_serving_adapter.so build/libdsv41_flash_tp4_serving_adapter.so"
```

**2. Slot 1: Qwen 3.8 27B FP8 on four ranks**, about 30 minutes. A staged
lane on four nodes, with production stopped by whoever holds the fleet. The
mesh lane and the first node are the ones the hold names; the values below
are an example.

```
export STAGE_MODEL=qwen38_27b_fp8_tp4 STAGE_LANE=3 STAGE_FIRST_NODE=4
export STAGE_WEIGHTD_SOCKET=/tmp/spark_weightd.sock STAGE_MEMORY_MAX=16G
export STAGE_SEQUENCES=8 STAGE_ROWS=1024 STAGE_KV_PAGES=2048 STAGE_KV_PHYSICAL_PAGES_CAP=8192
export STAGE_FIRMWARE_HOST=${BUILD_HOST:-lenovo7} STAGE_FIRMWARE=build-rp/fw-<sha>-src/build
export STAGE_API_FIRMWARE=$HUB_WORK/api-build-<sha>/build
tools/stage_lane.sh setup && tools/stage_lane.sh api-setup
GX_BACKEND=lane GX_RELEASE=<sha> GX_MINUTES=25 \
  GX_DIR=$HUB_WORK/gx/<sha>-qwen38-27b-fp8-tp4 GX_OUT=$PWD/qualification/gx-<date>-tp4 \
  GX_PLAN="--cases prefix,kept,opening,session,committed --contexts 8000,32000 --kept-contexts 8000 --tails 1,40,500,5000 --crowd 4" \
  scripts/gx_gate.sh qwen38_27b_fp8_tp4
```

The driver ends with the lane and its API stopped and the lane reclaimed.
`GX-BLOCK-PAUSED` means the minutes ran out: the same command goes on from
the first open step.

**3. Slot 2: DeepSeek V4.1 Flash on four ranks**, about 30 minutes. The
other settings of slot 1 stay as they are:

```
export STAGE_MODEL=dsv41_flash_tp4 STAGE_MEMORY_MAX=32G
export STAGE_SEQUENCES=8 STAGE_ROWS=4096 STAGE_KV_PAGES=1024 STAGE_KV_PHYSICAL_PAGES_CAP=8192
tools/stage_lane.sh setup && tools/stage_lane.sh api-setup
GX_BACKEND=lane GX_RELEASE=<sha> GX_MINUTES=25 \
  GX_DIR=$HUB_WORK/gx/<sha>-dsv41-flash-tp4 GX_OUT=$PWD/qualification/gx-<date>-tp4 \
  GX_PLAN="--cases prefix,opening,session,committed --contexts 8000,32000 --tails 1,40,500,5000" \
  scripts/gx_gate.sh dsv41_flash_tp4
```

After both slots pass, the claim is: reuse is exact on those two four-rank
lanes at 8,000 and 32,000 tokens. The served lanes are other builds and are
not gated by it.

**4. Pass 1: every lane, short stretches.** The candidate is installed as a
release root by the kit's steps up to `qualify`
(`datacenter:operations/FLEET_RELEASE_RUNBOOK.md`), with the same `FW`,
`APIB`, `OLD_WD` and `STATE` as those steps.

```
GX_PLAN="--contexts 8000,100000" scripts/release_rollout.sh gx
```

The seven run every case with prefixes of 8,000 and 100,000 tokens; the
other fourteen run the short plan. Receipts land in the kit state's `gx`
directory (`scripts/state-<sha>/gx` unless `STATE` names another). A lane
that stops is run again with `ONLY="<model>"` and the same command; it
goes on from its first open step.

**5. Pass 2: the long contexts** of the lanes that serve more than 200,000
tokens, in a state of their own. A lane that serves fewer than 300,000 runs
the largest prefix that fits.

```
ONLY="glm-5.3 kimi-k3 kimi-k3.1-pro deepseek-v4.1-flash qwen3.8-27b" \
  GX_DIR=$HUB_WORK/release-<sha>/gx-long GX_OUT=${STATE:-scripts/state-<sha>}/gx-long \
  GX_PLAN="--cases prefix,session --contexts 300000" scripts/release_rollout.sh gx
```

**6. Keep the result.** Copy the receipts (`<model>.json`, `.txt`, `.log`,
`.apilog.tgz`) into `qualification/gx-<date>/` with a README that says
which commit, which lanes, which plan and the verdict of each. The cost of
the rules goes into `PERFORMANCE_STATUS.md`, "Exact reuse", from the
receipt fields that section names.

**What a verdict needs.** `verdict=PASS` and `alone=yes` in every
`<model>.txt`. `INCOMPLETE` names what is missing ([What stops a
PASS](#what-stops-a-pass)); `FAIL` is a difference between a reusing run
and a cold run, and the receipt holds the first differing token and
logprob. A `FAIL` is a finding: stop and report it with the receipt.

## What a receipt measures

The receipts are also the fleet's turn-floor record, so no second matrix has
to run for it.

Per request (`steps.<step>.requests[]`):

| Field | Meaning |
| --- | --- |
| `label`, `attempt`, `max_tokens` | The unit and turn, and which attempt of the step sent it. A commit has `max_tokens` 0 and `finish_reason` `commit` |
| `started_utc`, `finished_utc` | Wall clock at the client, to the millisecond |
| `sent_ns`, `first_ns`, `end_ns` | The hub's monotonic clock just before the request is written, at the first streamed tokens and at the end of the stream. Requests are streamed, as the chat layer sends them |
| `request_id`, `boot_pid`, `accepted_ns`, `first_dispatch_ns`, `first_ready_ns`, `last_ready_ns` | The engine's own record of the request, on the same clock |
| `rested_seconds` | The idle gap that followed the request |
| `cuts`, `keep` | The cuts the request named and where it asked for state |
| `prompt_tokens`, `cached_tokens`, `prefilled_tokens` | The prompt, the position the request resumed at, and what it computed |
| `seconds` | Wall time at the client |
| `queue_ms`, `first_token_ms`, `last_token_ms` | From the engine's own request record: accepted to first dispatch, to the first token and to the last. The first dispatch is the first one the ranks admit, so in a `restore` step the queue time holds the wait for the restore |
| `decode_tokens_per_second` | Over the tokens after the first. Session replies are 96 tokens of chain decode |
| `completion_tokens`, `finish_reason`, `stale_prefix_recomputes` | |

A result in a `restore` step also carries `restore`: the pages the ranks
read back for it, the seconds of the slowest rank and of all ranks
together, the bytes (pages times the pool's bytes a page) and the misses.

`steps.chain.served` holds, for each context, one reply of 128 tokens with
speculation at the engine's default on a prompt that is already cached: the
decode rate as served. Session turns are 96 tokens with speculation off.

Per attempt of a step: the idle gap in force, and the API log's path with
the byte offsets the attempt covers and the name of its copy.

Per attempt of a step, before and after: the running engine's KV store
counters summed over its ranks (pages and nanoseconds saved and restored,
restore misses, store files and bytes, device pool pages and bytes), the
slowest rank's save and restore time, and the engine's prefix counters (hits,
misses, hit tokens, stale-prefix recomputes, waves refused as busy, as not
found and for capacity, requests that waited out the in-flight budget, draft
tokens).
The attempt also names the engine: adapter id, model revision and driver
artifact SHA-256.

- **Cold fill at a context**: the `*/warm` requests (a commit's `seconds`
  is its prefill) and everything in `cold:*`.
- **A turn on a kept prefix**: `prefix/<context>/<tail>` in `chain`. Each
  unit's verdict names `resume_cut`, the cut the reuse should resume at.
- **A conversation that shares a head's first blocks**:
  `opening/<context>/<tail>` in `chain`. Its `prefilled_tokens` are the
  head past the second block's cut plus the tail.
- **A turn of a session**: `session/<context>/<tail>/turn2` and the last
  prompt in `chain`. Each computes the previous reply again as prefill, so
  its `prefilled_tokens` are the reply plus the new tail plus one.
- **A turn after a committed reply**: `committed/<context>/<tail>/turn2` and
  the last prompt in `chain`. Their `prefilled_tokens` are the new tail
  alone. `committed/<context>/<tail>/commit1` and `commit2` are the commits:
  a commit's `seconds` is the reply's prefill, run before the next turn
  instead of inside it. Against the `session` unit of the same context and
  tail, the difference in `first_token_ms` is what a committed reply saves a
  turn, and a commit's `seconds` is what it costs when no turn follows.
- **After an eviction**: the same requests in `restore:*`. Their extra
  `first_token_ms` against `chain` is the restore the user waits for.
  Restored bytes are restored pages times pool bytes per pool page. A
  `restore` step with no restored pages did not read from NVMe.
