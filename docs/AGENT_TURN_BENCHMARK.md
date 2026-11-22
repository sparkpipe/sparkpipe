# Agent-turn benchmark

`tools/agent_turn_bench.py` answers the buyer's second question: how long does
a whole agent turn take, with one agent and with many at once? It replays
agent sessions through the hub chat layer, the path customers use. The
sessions take two shapes:

- Claude Code's Messages requests: a system prompt and tools shared by every
  session, a history of `Read` tool calls and their results, and reasoning.
- Codex's Responses requests: `instructions`, Codex's tools and the full
  input each turn.

Each live turn's output is sent back as it came, and its tool call gets a
result. The next request is therefore a prefill of that result on top of a
cached prefix, as in a real session. The tool times every turn. Nothing in
this file has been run on the fleet yet; the results are the CUDA dev's to
produce.

## What one run measures

- **Per turn:**
  - time to first token (any reasoning, text or tool output), counted from
    the first attempt, so 429 waits are included;
  - time to the end of the turn, and time to the first text or tool call;
  - prompt, cached, prefill and output tokens;
  - the chat layer's own `queue_ms`, `prefill_ms` (admission to first token)
    and `ttft_ms`;
  - decode rate, accepted drafts, 429 retries and the stop reason;
  - the assistant output, as a receipt.
- **Per arm (1, 4, 8 and 16 agents arriving `--stagger` seconds apart):**
  - output tok/s over the arm, and over the window in which every agent is
    busy;
  - output per agent, and prefill tok/s;
  - first-token and turn percentiles;
  - the share of turn time spent waiting for the first token, in prefill
    (server-measured), in the queue and decoding, plus first-token and turn
    medians for each 50K-token context band;
  - the hub's load average before and after;
  - with `--engine-health`, the engine's `live_requests`, which shows other
    traffic;
  - with `--memory-hosts`, the memory gate below.

## Defaults and why

| Setting | Default | Reason |
| --- | --- | --- |
| `--context-low` / `--context-high` | 100,000 / 300,000 | the range a long agent session lives in |
| start contexts | spread from 100K to the top that leaves room for the turns (about 227K), in van der Corput order | the first 1, 4 and 8 sessions already cover the range |
| `--turns` | 6 | each session grows by six tool results |
| `--result-tokens` | 1,500:8,000 | file reads and command output |
| `--preamble-tokens` | 20,000 | system prompt and tool definitions, shared by every session |
| `--effort` | low | short agent turns, read then call the next tool; `medium` and `high` lengthen reasoning |
| `--max-tokens` | 4,096 | bounds a turn |
| `--stagger` / `--jitter` | 20 s / 5 s | agents arrive one after another |
| `--seed` | `agent-turn-1` | seeds depend only on the session index and this, so a warm pass serves every arm and an A/B sees the same conversations |

## Running it

Run it on the hub, inside the CUDA dev's hold, from a directory under
`/srv/workspace/sparkpipe`. It needs only Python 3, and
`tools/fleet_memory_watch.py` must sit beside it.

```sh
W=/srv/workspace/sparkpipe/agentturn-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p $W/corpus
git archive <main-sha> docs tools runtime inference node include | tar -C $W/corpus -x
cp tools/agent_turn_bench.py tools/fleet_memory_watch.py $W/
```

### The key

With `--key-file`, requests carry `X-SparkPipe-Public: 1` and an `sk-sp-`
key, so they take the public door: fair share by key, a 60 s busy wait,
then 429 with `Retry-After`. That is what a customer sees at 16 agents on an
8-sequence cap. Without it, requests are internal: own scope, with no busy
timeout.

Make a dedicated key, keep it in a mode-600 file, and revoke it afterwards:

```sh
umask 077; mkdir -p ~/.sparkpipe-bench
cd ~/sparkpipe-chat && python3 api_keys.py --file api_keys.json create --days 7 --note "agent-turn benchmark" > ~/.sparkpipe-bench/key
python3 api_keys.py --file api_keys.json revoke "$(cut -d' ' -f1 ~/.sparkpipe-bench/key)"
```

The second line creates the key; the third revokes it after the run. The
benchmark reads only the `sk-sp-` word of the file and refuses a file other
users can read.

### Plan first

`plan` sizes every seed exactly with `/v1/messages/count_tokens` and runs no
generation:

```sh
python3 $W/agent_turn_bench.py plan --endpoint http://127.0.0.1:8433 --model glm-5.3 --corpus $W/corpus \
  --sessions 1,4,8,16 --label glm53 --prefill-rate 700 --output $W/glm53.plan.json
```

It prints each seed's tokens, the cold prefill of the warm pass (the shared
preamble counted once) and the warm pass's time at `--prefill-rate`. Use
prefill rates from `PERFORMANCE_STATUS.md` at 100K-200K.

With the defaults, 8 seeds come to about 1.1M cold tokens and 16 to about
2.3M. That is DERIVED from the start-context spread, not counted.

| Model | Prefill rate used | Warm 8 seeds | Warm 16 seeds |
| --- | ---: | ---: | ---: |
| GLM 5.3 Full | about 700 tok/s (197K) | about 26 min | about 54 min |
| Kimi K3 | about 460 tok/s (137.7K) | about 40 min | about 82 min |
| DeepSeek V4.1 Flash | about 1,300-1,600 tok/s | about 12-14 min | about 24-29 min |

The arms then add a few minutes each at 1, 4 and 8 agents. The 16-agent arm
on an 8-sequence cap queues, so allow 15-20 minutes.

### Split across holds of 60 minutes or less

KV state outlives the hold: retention is 24 h, and pages that overflow the
pool park on NVMe. Run the pieces with the same `--seed`, `--label` and
`--output`:

```sh
B="python3 $W/agent_turn_bench.py run --endpoint http://127.0.0.1:8433 --key-file ~/.sparkpipe-bench/key --model glm-5.3 \
   --corpus $W/corpus --engine-health http://127.0.0.1:8475 --label glm53 --output $W/glm53"
$B --sessions 1,4,8 --warm-only
$B --sessions 1,4,8 --no-warm
$B --sessions 16 --warm-only
$B --sessions 16 --no-warm --memory-hosts spark0,spark1,spark2,spark3,spark4,spark5,spark6,spark7,spark8,spark9,sparka,sparkb,sparkc,sparkd,sparke,sparkf
python3 $W/agent_turn_bench.py report $W/glm53
```

1. The first piece is the warm pass for 8 seeds.
2. The second runs the 1-, 4- and 8-agent arms.
3. The third warms the remaining 8 seeds. The first 8 hit the cache.
4. The fourth runs the 16-agent arm with the memory gate.
5. The last line prints the table.

For the engine-health port, use the active model's `engine` URL in
`deployment/hub/chat_frontend.json`: GLM 5.3 8475, Kimi K3 8459, DeepSeek
V4.1 Flash 8463.

For Codex's shape, run the same pieces with `--shapes responses` and another
label (for example `glm53-codex`) at 1 and 8 agents. `--shapes
messages,responses` alternates the shapes within one team.

A warm request is a normal request with `max_tokens` 1. Its `prefill_ms` is
the seed's cold prefill, and the summary reports it under `warm`.

## The turn floor and the cold events

Two more commands time one session at a time, where the arms above time
many. Both write `turns.jsonl` in `--output`, and `report` prints their
table.

**`floor`** answers what a typical turn costs at each context length. One
session per shape grows through `--contexts` (default 20,000, 50,000,
135,000 and 300,000 tokens). At each level it sends the level request
(`floor_grow`), then `--repeats` (default 3) turns for each of `--tails`
(default 50, 300 and 2,400 tokens): the last reply goes back as it came,
with a tool result of that size, so the request is the reply plus the tail
on a cached prefix. `--reply-tokens` (default 64) bounds each reply. The
session is saved as `session-<shape>.json` for the resume event.

**`cold`** times the events that lose a prefix (`--events`, default
`opening,subagent`):

| Event | Requests | What it shows |
| --- | --- | --- |
| `opening` | a new session on a `--opening-tokens` (46,000) opening, then a second session on the same opening | the first is cold from its first tool on; the second pays from the engine's nearest resume point |
| `subagent` | a subagent of one type on a `--subagent-tokens` (37,000) opening with five tools, then a second with another brief | the same pair, with the subagent class and agent type headers |
| `resume` | one turn with a 300-token tail on the saved `floor` session; needs `--event NAME` | what the session costs after the operator did `NAME` to it: a swap away and back, an engine restart, an eviction |

**`render`** times the chat layer alone: it fills a session to each of
`--contexts` and asks `/v1/messages/count_tokens` for it `--repeats` times.
That request is translated, rendered and tokenized like a real one and never
reaches an engine, so the model need not be resident.

`--nonce` goes into the first tool's description and the system prompt, so
each `cold` run is a first sight unless the nonce is repeated. A request
without a key carries the chat layer's miss ledger
(`docs/LITELLM_FRONTEND.md`), and the table shows its cause and missed
tokens; a keyed request shows the times only.

```sh
T="--endpoint http://127.0.0.1:8433 --model glm-5.3 --corpus $W/corpus --label glm53 --output $W/glm53-floor"
python3 $W/agent_turn_bench.py floor $T --shapes messages,responses
python3 $W/agent_turn_bench.py cold $T --shapes messages
python3 $W/agent_turn_bench.py cold $T --shapes messages --events resume --event swap
python3 $W/agent_turn_bench.py report $W/glm53-floor
```

### The engine's share: `tools/turn_floor.py`

`tools/turn_floor.py` sends token ids straight to one engine API, with no
chat layer, and splits each request's time to first token with the engine's
own log. It is the measurement behind the turn floor in
`PERFORMANCE_STATUS.md`. Run it on the engine's host, so its clock is the
engine's.

```sh
python3 turn_floor.py run --endpoint http://127.0.0.1:8459 --tokens tokens.txt --api-log /srv/workspace/sparkpipe/logs/k3-kv-api.log \
    --label kimi-k3 --output $W/kimi-k3 --arms turns,openings,wave
python3 turn_floor.py run ... --arms resume --event swap
python3 turn_floor.py parts $W/kimi-k3
```

- `--tokens` is a file of token ids from the model's own tokenizer
  (`sparkpipe_tokenize_prompt --tokenizer-compiled ... --text-file ...`,
  one id a line). A run's prompts start at a random rotation of it (the
  `salt`, kept in `session.json`), so nothing an earlier run cached is
  reused.
- **`turns`**: at each of `--contexts`, a `grow` request, then `--repeats`
  plain turns for each of `--tails` with speculation off, `--spec-repeats`
  turns as served, a `decode` sample of `--decode-tokens` on the whole
  history with speculation off, and with `--served-tokens` a `decode_served`
  sample of that many tokens on the same prompt with speculation as the
  engine serves it. A sample's new tail is 0. On an engine that reuses the
  pages a reply wrote, the first token of either sample is the last prompt
  token's wave alone, the floor of the floor. On an engine that resumes only
  at cuts, the `decode` sample also computes the last reply again, and the
  `decode_served` sample, which repeats its prompt, is the one whose first
  token is that wave alone. The decode rate is read from the tokens after
  the first, so it is the same reading on both. A tail of
  `--once-from` tokens or more runs once and only plain, so a long tail is
  not prefilled again for each repeat.
- A level that does not fit the engine's context beside its turns is
  skipped and named in the run's output; when none fits, the run stops with
  an error instead of measuring nothing.
- A run that stops inside a level keeps the level's finished steps in
  `session.json` and goes on from the next one when the same command runs
  again. The first session request of such a run is marked `first_of_run`
  and left out of the level's figures: what the engine still holds of the
  session is not what a turn in a row finds.
- **`decode`**: grows the session to `--decode-context` and samples there.
- **`openings`**: a 46,000-token opening, a second session on it, a
  37,000-token subagent opening, and a second subagent that shares 97% of it.
- **`wave`**: a `--wave-tokens` prompt at background priority, and a
  2,048-token cached probe sent at interactive priority again and again
  while it runs. The probe's time beyond its solo time is the wait behind a
  background wave.
- **`resume`**: one turn on the saved session after `--event`.
- **`truth`**: asks the engine what it holds of the saved session
  (`POST /v1/cache/state` with the session's token ids) and keeps the answer
  in the run's record under `--event`; `parts` lists the answers in
  `summary.json` as `cache_states`. An engine without the endpoint answers
  404, which is recorded and stops nothing.

Every request names its cuts, as the chat layer does for a conversation. A
session request sends `cuts`: for each earlier reply of the session, one
token before that turn's prompt ended and where its reply ended. The list
rises and names a count once: after a one-token tail a reply ends one token
before the next prompt does, and an engine that reads cuts refuses a list
that repeats a count. The question of the `truth` arm sends the same list.
An opening and the second request on it send the length they share as a cut
and in `keep`, so state is kept there. An engine that does not know the
members ignores them; one that resumes only at the cuts a request names
needs them to resume a turn at all.

Every completion's `cache` object, when the engine sends one, is kept with
the request (`cache_matched_tokens`, `cache_resumed_tokens`,
`cache_restore_ms`, `cache_background_wait_ms`, `cache_boot`,
`cache_artifact`), so what the engine says of a request stands beside what
its log says.

`parts` joins each request to its `request_measurements` line by the
SHA-256 of its token ids and reads the `batch_step` and `batch_rejected`
lines of that request:

| Part | From |
| --- | --- |
| send | client send to the engine's acceptance: the HTTP request, the body's JSON and the hand-off to the engine thread |
| queue | acceptance to the first admitted dispatch, when the ranks never answered pending |
| restore | the same interval when they did (`batch_rejected status=18`): the ranks were restoring the prefix |
| prefill | first admitted dispatch to the dispatch of the last prompt token's wave: the waves over every new token but the last, and what runs between them |
| first step | the last prompt token's wave, one row, whose result is the first token: one decode step's worth of work |

An adapter that keeps checkpoints puts one a token before the prompt's end
(`SparkModelBatchPromptCheckpoint`), so the engine prefills every new token
but the last in waves and then runs the last token as a wave of one row; the
first token is ready when that wave returns, and the chains of the reply
come after it. The split reads that wave from the log (the request's last
`kind=prefill` line, `rows=1`) and takes its wall time as the first step. An
adapter without checkpoints (GLM 5.3 Full on release c2b714343) and a prompt
shorter than a block have no such wave: the last token rides in the last
prefill wave, the first step reads 0 and everything before the first token
counts as prefill. `decode_tok_s` is the tokens after the first
over the time from the first token's stamp to the last; the tokens of one
chain share a stamp. `summary.json` also holds the change in the ranks'
`kv_store_report` counters over the run (`save_pages`, `save_ns`,
`restore_pages`, `restore_ns`), which give the save and restore rates.

`parts --api-log FILE --log-base N` reads a copied part of the engine's log
that starts at byte `N` of the original, so a run can be split away from the
host that made it. A request that failed stays in `parts.jsonl` with its
error and takes no line of the log; when it was sent again, the line belongs
to the request that succeeded.

Send, and the reply leg from the engine's first token to the client's,
compare the client's clock with the engine's. The engine stamps
`CLOCK_MONOTONIC`, which is Python's monotonic clock on Linux, so they hold
only when the tool ran on the engine's host. `parts` checks it: a request
the engine accepted before the client sent it, or a first token the client
read before the engine produced it, shows two clocks, and then every send
and reply time of the run is left out, `summary.json` says
`"clock_shared": false` and the tables say so. Queue, restore, prefill and
the first step are differences of the engine's own stamps and stay. The
`clock` field is the clock the client read. On macOS some Python builds
count that clock from each process's start, which is the case this catches
on a workstation.

`turn_floor.py excerpt DIR --output FILE` writes the lines of the engine's
log that a split reads (steps, refused waves, request records, and each
rank's store counters before, first and last in the run) and prints the
`--log-base` that goes with the file. That is the log a receipt keeps.

### On the fleet: `scripts/turn_floor_fleet.sh`

The driver walks served models through the tool on the hub, from the
workstation, with the release kit's helpers:

```sh
TF_RUN=tf-20261122 TF_OUT=qualification/turn-floor-20261122 TF_TOKENS=/srv/workspace/sparkpipe/turn-floor/tokens \
    scripts/turn_floor_fleet.sh kimi-k3=k3=8000,100000 deepseek-v4.1-flash=dsv41=8000,100000,300000=turns,wave,openings,restart
```

- A model is `MODEL=FAMILY=LEVELS[=ARMS]`: its served name, the token file
  `tokens-FAMILY.txt` in `TF_TOKENS` on the hub, its context levels, and its
  arms (`turns,wave,restart` unless named; `openings` is the fourth).
- The lane, the engine's port and its log come from the model's lane record
  (`tools/lane_config.py release-row`).
- Each model: the swap controller activates it; `turns` (every level: a
  grow, tails of 1, 40, 500 and 5,000 tokens, three times each below 2,000
  tokens and once above, a 96-token plain decode sample and a 128-token one
  as served); `wave` (a 20,000-token background prefill with interactive
  probes); `openings`; `restart` (the controller stops and starts the model,
  then one turn on the saved session, which has to come back from NVMe);
  then `parts`, `excerpt`, and the receipts are copied to `TF_OUT`.
- Every step runs detached on the hub, so a lost connection does not stop
  it, and is recorded when it ends. Running the same command again goes on
  where the run stopped: finished steps are skipped and `turns` continues
  inside its open level. The first failure stops the run; no later model
  starts.
- `TF_DRY=1` reaches no host and prints every step and command.
- The driver changes the active model and leaves the last one active. It
  does not arrange who may use the fleet or restore the production model:
  whoever runs it does both, before and after.

### A test lane of a build: `scripts/turn_floor_slot.sh`

The slot script runs the tool against a build that is not released, on a
test lane of one group of nodes, and adds what only such a build can
answer: the `truth` arm after the turns, and before and after a turn that
follows a stop and start of the lane.

```sh
SLOT_MODEL=dsv41_flash_tp4 SLOT_LABEL=deepseek-v4.1-flash-tp4 SLOT_FW=fw-$SHA \
    SLOT_API_BUILD=/srv/workspace/sparkpipe/turn-floor-slot/api-build-$SHA/build \
    SLOT_TOKENS=/srv/workspace/sparkpipe/turn-floor/tokens/tokens-dsv41.txt SLOT_OUT=qualification/turn-floor-slot-$RUN \
    scripts/turn_floor_slot.sh $RUN
```

- `SLOT_MODEL` is a `tools/stage_lane.py` model and has no default,
  `SLOT_FIRST` is its first node (default 4, so spark4 to spark7 for a lane
  of four), `SLOT_LANE` its mesh lane (default 1), `SLOT_CONTEXTS` its
  levels (default `8000,50000`). The `STAGE_*` sizes of
  `tools/stage_lane.sh` keep the script's defaults unless the caller sets
  them. Every setting is checked before it is used: the model and the label
  against a pattern, the node, the lane and the port as whole numbers in
  range, the paths as plain paths.
- It refuses to start unless the swap controller reads `idle None None`,
  both builds are complete and of one commit (`SOURCE_COMMIT` in `SLOT_FW`
  and beside `SLOT_API_BUILD`), and every node of the group has
  `SLOT_MIN_FREE_GIB` (default 100) GiB available.
- Then: the lane (`stage_lane.sh setup`, `start`, wait for every rank), the
  API as a transient unit on the hub, the `turns`, `truth` and `wave` arms,
  a stop and start of the API and the lane that keeps the API's prefix
  index, `truth`, `resume --event restart`, `truth`, the split, the excerpt,
  and the receipts copied to `SLOT_OUT`.
- On every exit it stops the API unit, stops the lane and reclaims it, and
  prints each node's available memory.
- `SLOT_DRY=1` reaches no host and prints the plan. A run id is used once:
  the lane's logs are named by it.

## Starting the fleet measurements

Nothing below has run on the fleet yet except one cold request
(`qualification/turn-floor-20261121/`). Both scripts have run only as dry
runs. This section is what a session needs to start either run without
other knowledge.

Who may use the fleet, and when, is in
`datacenter:operations/FLEET_HOLD.md`; read it first. Neither script takes
or releases the fleet, stops the production model or restores it: the
caller does, before and after, as that document says.

**Token files.** Each model family needs a file of token ids from its own
tokenizer, one id a line, at least 450,000 ids for a 300,000-token level.
On the hub, from a text of a few megabytes and the compiled tokenizer of a
served model's API folder:

```sh
split -b 1500000 corpus.txt part.
for part in part.??; do ~/k3-kv-api/bin/sparkpipe_tokenize_prompt --tokenizer-compiled ~/k3-kv-api/runtime/tokenizer/tokenizer.compiled --text-file $part --output $part.ids; done
cat part.??.ids > tokens-k3.txt
```

The families are `k3` (Kimi K3 and K3.1 Pro), `glm53` (GLM 5.3 Full and
Flash), `qwen38` (Qwen 3.8 27B and Max) and `dsv41` (DeepSeek V4.1 Flash);
the API folder of a model is the fourth field of
`python3 tools/lane_config.py release-row MODEL`. The tokenizer takes
about 1.5 MB of text a call.

**The sixteen-node plan.** One chain per model, cold once per level, every
tail resumed from it. Print it first, then run it:

```sh
export TF_RUN=tf-$(date -u +%Y%m%d) TF_OUT=qualification/turn-floor-$(date -u +%Y%m%d) TF_TOKENS=/srv/workspace/sparkpipe/turn-floor/tokens
PLAN="qwen3.8-max=qwen38=8000,24000 glm-5.3-flash=glm53=8000,100000=turns,wave,openings,restart qwen3.8-27b=qwen38=8000,100000,240000=turns,wave,openings,restart deepseek-v4.1-flash=dsv41=8000,100000,300000=turns,wave,openings,restart glm-5.3=glm53=8000,100000 kimi-k3=k3=8000,100000 kimi-k3.1-pro=k3=8000,100000"
TF_DRY=1 scripts/turn_floor_fleet.sh $PLAN
scripts/turn_floor_fleet.sh $PLAN
python3 tools/turn_floor.py report $TF_OUT/*/
```

Run it from bash, or from zsh as `scripts/turn_floor_fleet.sh ${=PLAN}`:
zsh does not split `$PLAN` into words.

| Model | Levels | Estimated minutes | Longest single prefill, estimated |
| --- | --- | ---: | ---: |
| Qwen 3.8 Max | 8K, 24K | 4 | 0.5 min |
| GLM 5.3 Flash | 8K, 100K, openings | 4 | 0.8 min |
| Qwen 3.8 27B | 8K, 100K, 240K, openings | 5 | 1.5 min |
| DeepSeek V4.1 Flash | 8K, 100K, 300K, openings | 7 | 2.5 min |
| GLM 5.3 Full | 8K, 100K | 4 | 1.9 min |
| Kimi K3 | 8K, 100K | 5 | 2.9 min |
| Kimi K3.1 Pro | 8K, 100K | 7 | 4.1 min |
| All seven | | 36 | |

The minutes are estimates from release c2b714343's 7.4K prefill rates
(`qualification/releases/c2b714343/`) and the long-prompt rates above in
`PERFORMANCE_STATUS.md`; none is measured at these depths on this release.
They include each model's swap in and its stop and start. A run that stops
is continued by the same command with the same `TF_RUN`.

It gives, per model: the first token of a turn at each level for 1, 40, 500
and 5,000 new tokens with its parts, plain and as-served decode at each
level, the wait behind a 20,000-token background prefill, the 46K and 37K
openings on the three models that run them, and one turn after a stop and
start. It does not give 300K on GLM 5.3 Full, Kimi K3 and Kimi K3.1 Pro, or
anything beyond 300K.

Afterwards: the receipts under `TF_OUT` go into the pull request, the
`report` tables go into the "First token of a turn" section of
`PERFORMANCE_STATUS.md` with the largest part named for each model, and the
K3 follow-up line under "KV cache and long context" is replaced by K3's
row.

**The four-node slot.** It proves the instruments of a build on real
ranks. For commit `$SHA` of the build under test, from bash on a workstation
whose ssh knows the release kit's host names (`lenovo7`, the build host,
and `rtx5090`, the hub):

```sh
SHA=$(git rev-parse --short=9 HEAD)
COPYFILE_DISABLE=1 git archive --format=tar $SHA > /tmp/src-$SHA.tar
NODE="build/sparkpipe_model_residentd build/weightd_warm build/libhidden_transport_spark_host_rdma_verbs.so build/runtime_libs/libsparkpipe_device_cuda.so build/libdsv41_flash_tp4_serving_adapter.so"
API="build/sparkpipe_model_api build/sparkpipe_tokenize_prompt build/libdsv41_flash_tp4_serving_adapter.so build/runtime_libs/libsparkpipe_device_cuda.so"
python3 tools/sparkcp.py /tmp/src-$SHA.tar lenovo7:build-rp/
ssh lenovo7 "cd ~/build-rp && rm -rf fw-$SHA-src && mkdir fw-$SHA-src && tar -C fw-$SHA-src -xf src-$SHA.tar && cd fw-$SHA-src && export PATH=/usr/local/cuda/bin:\$PATH && make -j8 $NODE"
mkdir -p fw-$SHA/runtime_libs && echo $SHA > fw-$SHA/SOURCE_COMMIT
python3 tools/sparkcp.py lenovo7:build-rp/fw-$SHA-src/build/sparkpipe_model_residentd lenovo7:build-rp/fw-$SHA-src/build/weightd_warm lenovo7:build-rp/fw-$SHA-src/build/libdsv41_flash_tp4_serving_adapter.so lenovo7:build-rp/fw-$SHA-src/build/libhidden_transport_spark_host_rdma_verbs.so fw-$SHA/
python3 tools/sparkcp.py lenovo7:build-rp/fw-$SHA-src/build/runtime_libs/libsparkpipe_device_cuda.so fw-$SHA/runtime_libs/
ssh rtx5090 "mkdir -p /srv/workspace/sparkpipe/turn-floor-slot"
python3 tools/sparkcp.py /tmp/src-$SHA.tar hub:/srv/workspace/sparkpipe/turn-floor-slot/
ssh rtx5090 "cd /srv/workspace/sparkpipe/turn-floor-slot && rm -rf api-build-$SHA && mkdir api-build-$SHA && tar -C api-build-$SHA -xf src-$SHA.tar && echo $SHA > api-build-$SHA/SOURCE_COMMIT && cd api-build-$SHA && export PATH=/usr/local/cuda/bin:\$PATH && make -j8 $API"
```

Both builds took under a minute on 2026-11-21. The node files are copied to
the nodes from the workstation by `tools/stage_lane.sh`, so none of them
has to sit on the hub. Then, holding the fleet with the production model
stopped:

```sh
RUN=s0$(date -u +%H%M)
export SLOT_MODEL=dsv41_flash_tp4 SLOT_LABEL=deepseek-v4.1-flash-tp4 SLOT_FW=fw-$SHA SLOT_API_BUILD=/srv/workspace/sparkpipe/turn-floor-slot/api-build-$SHA/build
export SLOT_TOKENS=/srv/workspace/sparkpipe/turn-floor/tokens/tokens-dsv41.txt SLOT_OUT=qualification/turn-floor-slot-$RUN
SLOT_DRY=1 scripts/turn_floor_slot.sh $RUN
scripts/turn_floor_slot.sh $RUN
```

About 15 minutes of the lane are expected, as an estimate: its start, two
levels of turns, a 60,000-token background prefill, a stop and start, one
resumed turn. What to read in `summary.json` and `parts.jsonl` afterwards:

- `cache_states`: after the turns every rank should hold the session and
  have it resident (`held_tokens`, `resident_tokens`, `ranks_failed` 0).
  Before the resume no rank should hold it (`held_tokens` 0), and
  `saved_tokens` says how much of it the snapshot store has: a session a
  clean stop saved should read as saved. After the resume it should be held
  again. A probe that reads otherwise where the store's own counters show
  the pages restored is a fault of the probe.
- The resumed turn: `cache_restore_ms`, which the engine reports, against
  `restore_ms`, which the split reads from the refused waves in the log.
- The probes behind the wave: `cache_background_wait_ms` against the log.
  A probe dispatched while the background wave is still running waits at
  the ranks, so its wait shows as `first_step_ms` above
  `first_step_service_ms`, beyond the same difference for a probe sent
  alone; a probe that could not be dispatched shows it in `queue_ms`.
- Every turn: `cache_matched_tokens` and `cache_resumed_tokens` against the
  `cached_tokens` of its usage.

A difference in any of these is a fault in the instruments, to be fixed
before the sixteen-node numbers are read through them.

`turn_floor.py report DIR...` reads several runs' `summary.json` and prints
the tables that compare models: first token by context for each tail with
its largest part, the parts for the smallest tail, plain decode rate by
context, the cold events, and restore, save and wave-wait figures.

## The memory gate

`--memory-hosts` (or `tools/fleet_memory_watch.py` on its own) samples
`MemAvailable` on each node at 10 Hz over ssh: the Q19 probe method of #1753.
For each node it records the lowest value, the steady level and the dip.

A node passes when its lowest reading stays at least 1 GiB above earlyoom's
line, 2% of MemTotal. The arm passes when every node is sampled and passes.

## Qualifying a higher sequence cap

TECHDEBT (Dynamic batching) asks for arrival/completion and memory gates
before the cap rises. The serving cap is `sequences` in `deployment/lanes.json`
(rendered by `tools/stage_lane.py` as `max_active_sequences`) and the
chat layer's `sequences` in `deployment/hub/chat_frontend.json`. The
`SequenceGate` constants `INTERACTIVE_RESERVE` (2) and `FAIR_SHARE` (0.75)
are in `tools/serving/chat_frontend.py`.

1. **Baseline:** at today's cap, run the 8- and 16-agent arms with the memory
   gate, as above. The 16-agent arm shows what the cap costs: `share.queue`,
   429 retries and first-token p90.
2. **Candidate:** a separate test root rendered with the higher `sequences`
   (I33: the exact committed revision, never the production root). Its chat
   layer `sequences` must match. Same seed, same runs.
3. **Pass:**
   - no turn errors at 16 agents;
   - the memory gate passes on all sixteen nodes for the whole arm;
   - output tok/s at 16 agents is above the baseline's;
   - turn p90 at 16 agents is no worse than the baseline's 16-agent arm;
   - every agent finishes its turns: no session's output rate falls below
     half the median.
4. **Then:** a config PR changes `sequences` for that model in both files.
   Raising it on the served fleet is a release, with the user's OK.

Expect memory to bind before the slot count. On 2026-11-05, GLM 5.3 Full's
engine reported `kv_capacity_max_tokens` 1,803,968: the resident KV pages
of all its sequences together, which is physical pages times 64
(`node/model_api.c`). Eight agents at 200K already fill that, so sixteen park
pages on NVMe. The restore time shows up in first-token time and in
`share.prefill`.

## Receipts and the ledger

- **Receipts:** copy each run's directory to
  `qualification/agent-turns/<yyyymmdd>/<label>/`: `turns.jsonl`,
  `summary.json` and the plan.
- **RUN.md:** add one beside them, naming:
  - the hub release (its root `MANIFEST` sha256) and the chat layer's file
    hashes;
  - the model and pack;
  - the hold window;
  - the hub load average from the arm records;
  - any overlapping traffic, from `engine_live_requests.max` above the run's
    own agents.
- **PERFORMANCE_STATUS.md:** put the `report` table in the "Agent turns"
  section, with the per-context medians from `summary.json`, and grade each
  figure MEASURED.
