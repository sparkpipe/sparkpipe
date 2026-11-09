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

### Check the plumbing first, with no fleet

`tools/agent_endpoint_stub.py` is a canned Messages and Responses endpoint.
Point every command below at it instead of the tunnel.

- **What it replies:** for each task named in `--solve`, the first turn
  calls the shell tool with `git apply` of the task's gold patch (`Bash`,
  `exec_command` or `shell`, whichever the agent offers). Any other task
  gets a harmless look around, and the next turn ends.
- **What a working setup gives:** the named tasks resolve, the others score
  as empty patches, and Terminal-Bench tasks get reward 0.
- **Its log:** every request, with whether `X-SparkPipe-Public: 1` came, and
  a 12-digit hash of the key, never the key itself.

```sh
python3 tools/agent_endpoint_stub.py --port 18433 --model stub-model --dataset verified.jsonl \
  --solve psf__requests-2931,sympy__sympy-13480 --log stub-requests.jsonl
```

`verified.jsonl` is the dataset with its `patch` field, for example
`load_swebench_dataset` written out as JSON lines. The run of 2026-11-05 on
an arm64 Mac with colima is in `qualification/agent-tasks/stub-20261105/`:
Claude Code and Codex each resolved the 2 of 3 SWE-bench tasks the stub
solved, and both scored 0 on Terminal-Bench's `fix-git`.

On colima, containers reach the Mac's loopback at `host.docker.internal`
(with `--add-host host.docker.internal:host-gateway`), and Harbor's task
networks reach it at 192.168.5.2. Docker volumes see only `$HOME`, so keep
the agents, the key file and the runs there.

### Images on an arm64 host

The published SWE-bench images (`swebench/sweb.eval.x86_64.*`) are x86_64.
SWE-bench 4.1's Dockerfiles carry the aarch64 conda builds, but its command
line always asks for x86_64. `agent_task_eval.py swebench --arch arm64` runs
that command line with the arch the host has.

```sh
uv venv swe4 && uv pip install --python swe4/bin/python swebench==4.1.0
swe4/bin/python tools/agent_task_eval.py swebench --arch arm64 prepare_images --dataset_name princeton-nlp/SWE-bench_Verified \
  --instance_ids $(python3 -c "import json;print(' '.join(t['instance_id'] for t in json.load(open('qualification/agent-tasks/swebench-verified-50.json'))['tasks']))") \
  --max_workers 2 --namespace none --tag latest --env_image_tag latest
```

The first line installs SWE-bench 4.1. The second builds the 50 tasks'
images. On the Mac, the stub run's base, three environment and three task
images took 5.5 minutes. Docker reports each task image at about 3.3 GB and
each environment at 3.2 GB, but they share layers. On colima, the
SDK-based grader needs
`DOCKER_HOST=unix://$HOME/.colima/default/docker.sock`.

The images are named `sweb.eval.arm64.<instance_id>:latest`. Pass
`--image-template 'sweb.eval.arm64.{instance_id_lower}:latest'` to `run`,
and grade with the same wrapper, as in the next section.

## The key

Without a key, use `--door internal` on `run` and `agent-config`. The
agents then send no `X-SparkPipe-Public` header, and the chat layer takes
their requests as internal: own scope, no key check and no 60 s busy wait.
The agents still need a key-shaped token to start. Give `run` a mode-600
file holding a placeholder such as `sk-sp-internal-door-carries-no-key`, and
put the same placeholder in Harbor's env file. The first fleet run of
2026-11-09 used the internal door, because no key was to be made on the
production pay service.

With the public door, the customer's path, create a dedicated key on the
hub into a mode-600 file. Copy it to the Docker host through stdin, never as
an argument. Revoke it when the runs end.

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

On an arm64 host, the same run uses the local images and the 4.1 grader:

```sh
python3 tools/agent_task_eval.py run --tasks qualification/agent-tasks/swebench-verified-50.json --harness claude-code --model glm-5.3 \
  --endpoint http://host.docker.internal:18433 --add-host host.docker.internal:host-gateway --key-file ~/.sparkpipe-bench-key \
  --agents ~/sparkpipe-agents --context 786432 --timeout 3600 --parallel 4 --stop-after 1800 \
  --image-template 'sweb.eval.arm64.{instance_id_lower}:latest' --label glm53-cc --output runs/glm53-cc
mkdir -p runs/glm53-cc/eval && (cd runs/glm53-cc/eval && ../../../swe4/bin/python ../../../tools/agent_task_eval.py swebench --arch arm64 run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified --predictions_path ../predictions.jsonl --run_id glm53-cc --namespace none --max_workers 3 --report_dir .)
python3 tools/agent_task_eval.py score --run runs/glm53-cc --report runs/glm53-cc/eval/claude-code__glm-5.3.glm53-cc.json
```

The paths in the second command assume `runs/` and `swe4/` sit in the
repository root. `--instance ID`, repeated, limits a run and its planned set
to those tasks, for a preflight of two or three.

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
It needs `docker compose`. On a Mac with Homebrew's Docker CLI, run `brew
install docker-compose` and link it into `~/.docker/cli-plugins`.

Write the agent's native config with `agent-config`, and put the key and
base URL in a mode-600 env file. Harbor's task networks do not map
`host.docker.internal`:
- on Linux, point the base URL at the bridge address the tunnel binds;
- on colima, at 192.168.5.2.

Harbor installs the agent in each task container from `--ak version=`.
Task filters take the dataset's prefixed names, such as `-i
terminal-bench/fix-git`.

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
- **Preflight:** first run `-i terminal-bench/fix-git` against the stub. It
  is a light task: a Python slim base, 1 CPU, 2 GB, internet allowed. It
  takes about three minutes on the Mac, and its trial must finish with no
  exception and reward 0. Then run two or three tasks against the fleet with
  `-l 3`, and check that the transcripts show requests reaching the chat
  layer.
- **Network:** every Terminal-Bench 2.1 task sets `allow_internet = true`,
  which the agent needs to reach the endpoint. DeepSeek's own runs had no
  network access, so check this before comparing.
- **Scoring:** `harbor-score` averages each task's attempts, so `-k 3` gives
  pass@1 over 3, as DeepSeek reports.
  - Pass `--task-list` with the run's whole task list. A run that did not
    reach every task is then marked partial, and `report` prints it as
    partial with its task count.
  - The score counts apart the trials whose agent never started, and each
    API error message.
- **Agent setup under Rosetta:** Harbor installs the agent with apt and npm
  in every amd64 task container, which took up to 353 s on the Mac against
  Harbor's 360 s limit. On an emulated host pass
  `--agent-setup-timeout-multiplier 3`. It raises only the install limit,
  never the agent's task time.
- **Images:** an agent that reads an image file sends the image in a tool
  result. The chat layer refuses that with 400 `image_not_supported` for a
  model without image input, and Claude Code ends the task there. Count
  these trials from the score's `api_errors`.

## Receipts and the ledger

- **Directory:** `qualification/agent-tasks/<yyyymmdd>/<label>/`, holding:
  - `run.json` and `runs.jsonl`;
  - `predictions.jsonl` and `patches/`;
  - the SWE-bench report, or the Harbor job's `result.json` files;
  - `score.json`.
- **Transcripts** (`logs/`, or a Harbor trial's `agent/`): commit them as
  plain files when a run's logs come to about 20 MB or less. Otherwise keep
  them under `/srv/workspace/sparkpipe/agent-tasks/<label>/` and commit
  their `SHA256SUMS`. Never commit a tarball: the source package refuses
  `.tar.gz`.
- **RUN.md:** add one naming:
  - the hub release (its root `MANIFEST` sha256) and the model's pack;
  - the agents' `VERSIONS`;
  - the hold windows;
  - any overlapping traffic.
- **PERFORMANCE_STATUS.md:** `report runs/* jobs/*.score.json` prints the
  "Task success" table. Put each row beside the provider's figure for the
  same benchmark and harness from `PUBLISHED_SCORES.md`, graded MEASURED
  and quoted with its date.
