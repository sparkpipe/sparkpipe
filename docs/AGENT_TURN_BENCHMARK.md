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
  history with speculation off (its new tail is 0, so its first token is the
  floor of the floor), and with `--served-tokens` a `decode_served` sample
  of that many tokens with speculation as the engine serves it. A tail of
  `--once-from` tokens or more runs once and only plain, so a long tail is
  not prefilled again for each repeat.
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
