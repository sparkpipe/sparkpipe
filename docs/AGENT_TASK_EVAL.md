# Task success with real agents

`tools/agent_task_eval.py` answers the buyer's first question: is an open
model good enough for my work? It runs the agents customers use, Claude Code
and Codex, on public coding benchmarks against the fleet, keeps every patch
and transcript, and scores with each benchmark's own grader. Nothing in this
file has been run on the fleet yet; the results are the CUDA dev's to produce.

## Which benchmark first

Read `qualification/agent-tasks/PUBLISHED_SCORES.md`, checked on 2026-11-05.

- **Terminal-Bench 2.1 first.** It is the one benchmark every provider
  reports, and some report it for these very agents. GLM 5.3 publishes 88.2
  with Claude Code 2.1.207. DeepSeek V4.1 Flash publishes 88.0 with Claude
  Code and 84.1 with Codex. A run with the same agent on our served model is
  a direct check of SparkPipe serving against the provider's own number.
- **SWE-bench Verified, 50 tasks, as a baseline.** None of the three
  providers publishes it now, so it compares only with the frontier, for
  example Opus 5's 96.0% as secondary sources quote its system card.
- **Terminal-Bench 4.0 later.** It is what Anthropic's newest page reports,
  but it contains GPU tasks and needs a GPU-capable sandbox.

## Where it runs

Use a Docker host that is neither the hub (its workspace disk is full) nor a
fleet node. The task containers reach the hub's chat layer through an SSH
tunnel that the Docker host opens. Bind the tunnel to the Docker bridge
address, never to 0.0.0.0: a request to the chat layer without
`X-SparkPipe-Public` is an internal request and needs no key.

```sh
ssh -N -o ExitOnForwardFailure=yes -L 172.17.0.1:18433:127.0.0.1:8433 spec@100.123.97.61 &
docker run --rm --add-host host.docker.internal:host-gateway curlimages/curl -s http://host.docker.internal:18433/api/hello
```

The first line opens the tunnel; the second checks it from a container.
`/api/hello` needs no key and touches no engine.

The published SWE-bench images are x86_64. On an aarch64 host, build them
from the task repo: `swebench images build`, or `swebench eval ...
--task-repo`. They keep the dataset's image names, which the runner uses.

## The key

Create a dedicated key on the hub into a mode-600 file. Copy it to the
Docker host through stdin, never as an argument. Revoke it when the runs end.

```sh
ssh spec@100.123.97.61 'umask 077; mkdir -p ~/.sparkpipe-bench; cd ~/sparkpipe-chat && python3 api_keys.py --file api_keys.json create --days 14 --note "agent task benchmarks" > ~/.sparkpipe-bench/key'
ssh spec@100.123.97.61 'cat ~/.sparkpipe-bench/key' | (umask 077; cat > ~/.sparkpipe-bench-key)
ssh spec@100.123.97.61 'cd ~/sparkpipe-chat && python3 api_keys.py --file api_keys.json revoke "$(cut -d" " -f1 ~/.sparkpipe-bench/key)"'
```

The first line creates the key on the hub. The second copies it to the
Docker host. The third revokes it after the runs.

The runner mounts the file read-only into each container. Claude Code reads
it through `apiKeyHelper`, and the shell that starts Codex reads it into
`SPARKPIPE_API_KEY`. The key never appears in a docker argument or a
container file. For Harbor, put it in a mode-600 env file.

## Agents

Pin the versions the release tests use, and record them:

```sh
python3 tools/agent_task_eval.py prepare-agents --platform linux-x64 --claude-code 2.1.311 --codex 0.166.0 --output ~/sparkpipe-agents
```

`VERSIONS` records each package and binary sha256. Use `--platform
linux-arm64` on an aarch64 host.

The settings are those of `sparkpipe setup claude-code` and `setup codex`,
plus `X-SparkPipe-Public: 1`:

- **Claude Code:** `ANTHROPIC_BASE_URL`, `ANTHROPIC_MODEL`, the haiku and
  subagent models set to the same model, `CLAUDE_CODE_MAX_CONTEXT_TOKENS`,
  experimental betas and non-essential traffic off.
- **Codex:** a `sparkpipe` provider with `wire_api = "responses"`, the model
  catalog URL, `model_context_window` and the auto-compact limit.

`agent-config` writes the same file for Harbor.

## SWE-bench Verified

`qualification/agent-tasks/swebench-verified-50.json` is the subset: seed
`swebench-verified-1`, stratified by repository, from the 500-row dataset
with SHA-256 `faf8fab9…`. Run each model and harness under its own label.

```sh
python3 tools/agent_task_eval.py run --tasks qualification/agent-tasks/swebench-verified-50.json --harness claude-code --model glm-5.3 \
  --endpoint http://host.docker.internal:18433 --add-host host.docker.internal:host-gateway --key-file ~/.sparkpipe-bench-key \
  --agents ~/sparkpipe-agents --context 786432 --timeout 3600 --parallel 4 --stop-after 1800 --label glm53-cc --output runs/glm53-cc
swebench eval verified -p runs/glm53-cc/predictions.jsonl --run-id glm53-cc -j 4 --report-dir runs/glm53-cc/eval \
  $(python3 -c "import json;print(' '.join('-i '+t['instance_id'] for t in json.load(open('qualification/agent-tasks/swebench-verified-50.json'))['tasks']))")
python3 tools/agent_task_eval.py score --run runs/glm53-cc --report runs/glm53-cc/eval/claude-code__glm-5.3.glm53-cc.json
```

The three commands run the agent, evaluate the patches, then score the run.

- **Context:** `--context` is the model's `context_tokens` in
  `deployment/hub/chat_frontend.json`. Today that is GLM 5.3 786,432, and
  Kimi K3 and DeepSeek V4.1 Flash 1,048,576.
- **Parallel:** `--parallel` is the number of agents on the fleet at once.
  Keep it below the model's sequence cap.
- **Holds of an hour or less:** past `--stop-after` seconds, no new task
  starts, and in-flight tasks finish within `--timeout`. Rerun the same
  command in the next hold and it resumes with the unfinished tasks.
- **Codex:** use `--harness codex` and labels such as `glm53-cx`.

## Terminal-Bench 2.1 with Harbor

[Harbor](https://github.com/harbor-framework/harbor) is the official runner.
Install it with `uv tool install harbor`; 0.25.0 was current on 2026-11-05.
Write the agent's native config with `agent-config`, and put the key and
base URL in a mode-600 env file. On Linux, point the base URL at the bridge
address the tunnel binds, because Harbor's task networks do not map
`host.docker.internal`.

```sh
python3 tools/agent_task_eval.py agent-config --harness claude-code --endpoint http://172.17.0.1:18433 --model glm-5.3 --context 786432 --output cc-glm53.json
(umask 077; printf 'ANTHROPIC_BASE_URL=http://172.17.0.1:18433\nANTHROPIC_AUTH_TOKEN=%s\n' "$(grep -o 'sk-sp-[^ ]*' ~/.sparkpipe-bench-key)" > cc.env)
harbor run -d terminal-bench/terminal-bench-2-1 -a claude-code -m glm-5.3 --ak version=2.1.311 --ak config=cc-glm53.json \
  --env-file cc.env -n 4 -k 1 -o jobs --job-name glm53-cc-tb21
python3 tools/agent_task_eval.py harbor-score --job jobs/glm53-cc-tb21 --harness claude-code --model glm-5.3 --label glm53-cc-tb21 --output jobs/glm53-cc-tb21.score.json
```

The four commands write the settings, write the env file, run the
benchmark, then score the job.

- **Codex:** use `agent-config --harness codex`, an env file holding
  `SPARKPIPE_API_KEY`, `-a codex --ak version=0.166.0 --ak
  config=codex-glm53.toml`.
- **Preflight:** run two or three tasks with `-l 3` or a task filter, and
  check the job's transcripts show requests reaching the chat layer.
  DeepSeek ran Terminal-Bench 2.1 without network access, while here the
  agent inside the task container must reach the fleet.
- **Scoring:** `harbor-score` averages each task's attempts, so `-k 3` gives
  pass@1 over 3, as DeepSeek reports.

## Receipts and the ledger

- **Directory:** `qualification/agent-tasks/<yyyymmdd>/<label>/`, holding:
  - `run.json` and `runs.jsonl`;
  - `predictions.jsonl` and `patches/`;
  - the SWE-bench report, or the Harbor job's `result.json` files;
  - `score.json`.
- **Transcripts** (`logs/`): commit them gzipped when a run's logs come to
  about 20 MB or less. Otherwise keep them under
  `/srv/workspace/sparkpipe/agent-tasks/<label>/` and commit their
  `SHA256SUMS`.
- **RUN.md:** add one naming:
  - the hub release (its root `MANIFEST` sha256) and the model's pack;
  - the agents' `VERSIONS`;
  - the hold windows;
  - any overlapping traffic.
- **PERFORMANCE_STATUS.md:** `report runs/* jobs/*.score.json` prints the
  "Task success" table. Put each row beside the provider's figure for the
  same benchmark and harness from `PUBLISHED_SCORES.md`, graded MEASURED
  and quoted with its date.
