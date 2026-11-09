# Terminal-Bench 2.1 with Claude Code on DeepSeek V4.1 Flash, Fleet 1, 2026-11-09

**Result: partial.** 21 of the 89 tasks ran before the 21:00 KST stop.
10 of the 21 passed, which is 48% as Harbor counts. This is not a
Terminal-Bench 2.1 score and is not comparable to DeepSeek's published 88.0
for Claude Code (`../PUBLISHED_SCORES.md`):
- the 21 tasks are a prefix of a seeded shuffle, not the benchmark;
- 5 of them never reached the model;
- 3 ended on our chat layer's refusal of images.

## What ran

- **Model:** `deepseek-v4.1-flash` on Fleet 1 at its configured TP16
  (lane 10, 16 ranks).
  - It was activated through the swap controller's `/activate`. `/force`
    was never used.
  - Hub core `MANIFEST` sha256 prefix: `118ab360e1a58126`.
  - The pack's `model_resident.json` sha256 prefix: `597cef7391846c98`.
- **Agent:** Claude Code 2.1.311, installed by Harbor in each task
  container with `--ak version=2.1.311`.
  - `--ak reasoning_effort=max`, with the settings in `cc-deepseek.json`
    (context 1,048,576).
  - One attempt per task (`-k 1`). The tasks' own agent and verifier time
    limits were used: no timeout multiplier.
- **Runner:** Harbor 0.25.0, dataset `terminal-bench/terminal-bench-2-1`
  (89 tasks), on this workstation:
  - an Apple-silicon MacBook Air;
  - colima 0.10.3: an aarch64 VM with 6 CPUs and 11.65 GiB;
  - Docker 29.5.2 and Docker Compose 5.7.0.
  - The task images are Harbor's prebuilt amd64 images, run under Rosetta.
- **Path:** each container reached the Mac's 127.0.0.1:18433 at
  192.168.5.2. An SSH tunnel carried it to the hub chat layer's
  127.0.0.1:8433.
- **Door: internal, with no key.** The requests carried no
  `X-SparkPipe-Public` header, so the chat layer took them as internal
  traffic: own scope, no key check, no 60 s busy wait.
  - The env file's `ANTHROPIC_AUTH_TOKEN` was the placeholder
    `sk-sp-internal-door-carries-no-key`. It is not a key.
  - Two transcripts show it where the agent printed its environment.
  - No key was created for this run.
- **Tool:** `tools/agent_task_eval.py` at `6ac3a5e38` when the run started.
  The scoring changes that separate trials whose agent never started, API
  errors and the partial marking are in the commit that adds these
  receipts.

## Order and window

- **Order:** `task-order.txt` is the 89 tasks in a seeded shuffle (seed
  `tb21-20261109`).
- **Batches:** `task-batches.txt` cuts it into 16 batches: 3 tasks, then
  6s.
- **Driver:** `driver.sh` started each batch only after checking four
  things:
  - the time was before 21:00 KST;
  - the controller read `deepseek-v4.1-flash idle`, with nothing wanted;
  - the engine had no live requests;
  - colima had at least 15 GB free.

| Time (KST) | Event | Receipt |
| --- | --- | --- |
| 18:16:53 | `qwen3.8-max` active and idle, nothing wanted | `status-before.json` |
| 18:17:04 | `/activate deepseek-v4.1-flash` | `activate-deepseek.json` |
| 18:18:17 | `deepseek-v4.1-flash` active and idle after a 60.5 s swap, not forced; projected 44 GiB, 38.4-38.7 GiB used per node, tightest spark0 | `swap-progress.log`, `status-deepseek-active.json`, `engine-health-start.json` |
| 18:18:51-18:42:41 | batch 1, 3 tasks | `driver.log` |
| 18:42:42-19:07:34 | batch 2, 6 tasks | |
| 19:07:35-20:15:56 | batch 3, 6 tasks | |
| 20:15:59-21:20:12 | batch 4, 6 tasks; it started before 21:00 and ran to its last task's 3600 s limit | |
| 21:20:12 | no batch 5: past 21:00 KST | `driver.log` |
| 21:20:36 | still `deepseek-v4.1-flash` idle; the engine had served 371 requests on 16 of 16 ranks, none live | `status-deepseek-end.json`, `engine-health-end.json` |
| 21:20:50 | `/activate qwen3.8-max` | `activate-qwen.json` |
| 21:21:58 | `qwen3.8-max` active and idle after a 61.0 s swap, not forced | `swap-back-progress.log`, `status-after.json` |

**Overlapping traffic: none.**
- The chat layer logged 371 request records for `deepseek-v4.1-flash`. That
  matches the engine's 371 served requests.
- They come from 16 sessions, one for each trial whose agent ran.
- No other model's record follows the first of them.
- `chat-request-records.jsonl` holds the 371 records.

## Per task

| Batch | Task | Result | Agent time | Cause |
| ---: | --- | --- | ---: | --- |
| 1 | feal-linear-cryptanalysis | pass | 1114 s | |
| 1 | multi-source-data-merger | pass | 290 s | |
| 1 | raman-fitting | fail | 260 s | chat layer 400: `deepseek-v4.1-flash does not take images` |
| 2 | adaptive-rejection-sampler | not run | – | agent setup over Harbor's 360 s limit |
| 2 | bn-fit-modify | pass | 902 s | |
| 2 | crack-7z-hash | pass | 1086 s | |
| 2 | extract-elf | fail | 900 s | the task's agent time limit |
| 2 | sparql-university | not run | – | agent setup over 360 s |
| 2 | vulnerable-secret | pass | 189 s | |
| 3 | db-wal-recovery | pass | 200 s | |
| 3 | hf-model-inference | pass | 233 s | |
| 3 | portfolio-optimization | pass | 718 s | |
| 3 | qemu-startup | not run | – | agent install failed: `apt-get install nodejs npm` got 404s from the Debian bullseye security pool |
| 3 | regex-log | not run | – | agent setup over 360 s |
| 3 | train-fasttext | fail | 3601 s | the task's agent time limit |
| 4 | circuit-fibsqrt | pass | 3067 s | |
| 4 | install-windows-3.11 | fail | 365 s | chat layer 400: no images |
| 4 | merge-diff-arc-agi-task | not run | – | agent setup over 360 s |
| 4 | mteb-leaderboard | pass | 1381 s | |
| 4 | regex-chess | fail | 3601 s | the task's agent time limit |
| 4 | sam-cell-seg | fail | 678 s | chat layer 400: no images |

**Counts.** No task ended with a finished answer that the verifier
rejected: every 0 is a timeout or an error.

| | Tasks |
| --- | ---: |
| Ran (of 89) | 21 |
| Passed | 10 |
| Agent never started | 5 |
| Ended by the image refusal | 3 |
| Agent time limit | 3 |
| Pass rate as Harbor counts, partial | 10 / 21 = 48% |
| Pass rate over the 16 tasks whose agent started, partial | 10 / 16 = 62.5% |

Neither rate is a benchmark score.

## Findings

1. **The image refusal ends Claude Code's session.**
   - In the three tasks the agent used Claude Code's `Read` tool on a PNG it
     had made. Claude Code put the image in the tool result.
   - `tools/serving/chat_frontend.py` refuses any request with an image for
     a model without image input (400 `image_not_supported`).
   - Claude Code takes a 400 as fatal, so the whole task ended at that turn.
   - Whether a text-only model should get such turns with the image left
     out, or refused as now, is a product decision. It is not made here.
2. **Agent setup under Rosetta is near Harbor's limit.**
   - Harbor installs node, npm and Claude Code with apt and npm in every
     amd64 container. Under Rosetta that took 56-353 s where it finished,
     and 4 of 21 trials passed the 360 s limit.
   - `--agent-setup-timeout-multiplier` raises only that limit, not the
     agent's own task time.
   - A native x86_64 Docker host avoids Rosetta altogether.
3. **Serving under six agent sessions.** From the 371 records:
   - first token: median 0.88 s, p90 3.3 s;
   - prompts: median 38.8K tokens, p90 82.1K, largest 98.9K;
   - 95.3% of the 16.7M prompt tokens were served from cache;
   - chat-layer queue: median 103 ms, p90 191 ms, largest 1.08 s;
   - decode: median 45.6 tok/s, p10 25.7 tok/s.
4. **Two bulk-priority requests waited about 250 s in the engine.**
   - Both were in extract-elf's session, at the priority the chat layer
     gives Claude Code's background request class.
   - Their prompts were 838 and 319 tokens. Their first token came after
     253 s and 249 s, with 5-6 ms of chat-layer queue.
   - Whether the agent waited on them is not established.
5. **Pace.** 21 tasks took 3 h 1 min at six at a time on this test bed. All
   89 need about 13 hours here, or a faster Docker host.

## Files

- **Controller:**
  - `status-before.json`, `status-deepseek-active.json`,
    `status-deepseek-end.json`, `status-after.json`;
  - `activate-deepseek.json`, `activate-qwen.json`;
  - `swap-progress.log`, `swap-back-progress.log`.
- **Engine:** `engine-health-start.json`, `engine-health-end.json`.
- **Driver:** `driver.sh`, `driver.log`, `task-order.txt`,
  `task-batches.txt`, and the agent settings `cc-deepseek.json`.
- **Chat layer:** `chat-request-records.jsonl`.
- **Harbor jobs:** `terminal-bench-2-1/tb21-cc-b01` to `-b04`, holding
  each trial's:
  - `result.json`, `config.json` and `trial.log`;
  - `exception.txt`;
  - `verifier/`;
  - the console output in `harbor-console.txt`.
- **Agent logs:** each trial's `agent/` directory holds the Claude Code
  stream, the trajectory, the session files and the setup log.
  - They come to 42 MB, so they stay in this workstation's run directory,
    `~/sparkpipe-agent-e2e/fleet-20261109/jobs/`.
  - `terminal-bench-2-1/agent-logs.SHA256SUMS` lists their 122 files,
    relative to `jobs/`.
- **Score:** `terminal-bench-2-1/dsv41-cc-tb21.score.json`, from:

```sh
python3 tools/agent_task_eval.py harbor-score --job jobs/tb21-cc-b01 --job jobs/tb21-cc-b02 --job jobs/tb21-cc-b03 --job jobs/tb21-cc-b04 \
  --harness claude-code --model deepseek-v4.1-flash --benchmark terminal-bench-2.1 --label dsv41-cc-tb21 \
  --task-list task-order.txt --output dsv41-cc-tb21.score.json
```

The paths in Harbor's own records are this workstation's run directory,
`~/sparkpipe-agent-e2e/fleet-20261109/`.
