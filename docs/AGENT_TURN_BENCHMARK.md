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
