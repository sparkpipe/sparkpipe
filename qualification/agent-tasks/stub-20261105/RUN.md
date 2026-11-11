# Harness plumbing check against a stub model, 2026-11-05

These are not model results. Every reply came from
`tools/agent_endpoint_stub.py`, a canned Anthropic Messages and OpenAI
Responses endpoint. The check proves that the real agents, the real task
containers and the real graders work together through
`tools/agent_task_eval.py`:

- the request shapes, the public-door header and the key;
- tool calls and their results;
- patch capture, scoring and receipts.

## The script the stub follows

The stub serves model `stub-model`.

- **Tasks it solves:** for `psf__requests-2931` and `sympy__sympy-13480`,
  the first turn calls the shell tool with `git apply` of the task's gold
  patch: `Bash` for Claude Code, `exec_command` for Codex.
- **Every other task:** the first turn only runs `pwd && ls && git status
  --short`.
- **The next turn** ends with a text answer.

So the grader must resolve exactly two of the three SWE-bench tasks, record
`pytest-dev__pytest-7982` as an empty patch, and give the Terminal-Bench task
reward 0.

## Host and versions

- **Host:** this workstation, an Apple-silicon MacBook Air.
  - colima 0.10.3: an aarch64 VM with 6 CPUs and 11.65 GiB;
  - Docker 29.5.2, arm64, with Docker Compose 5.7.0.
- **Network:** no fleet, hub or Spark was reached. The stub listened on the
  Mac's 127.0.0.1:18433. Containers reached it at `host.docker.internal`
  with `--add-host host.docker.internal:host-gateway`; Harbor's task
  networks reached it at colima's host address, 192.168.5.2.
- **Agents:** `AGENT_VERSIONS` holds Claude Code 2.1.311 (linux-arm64) and
  Codex 0.166.0 (linux-arm64, musl), with their sha256s.
- **SWE-bench:** 4.1.0, dataset `princeton-nlp/SWE-bench_Verified` at
  revision `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`.
  - The images were built natively for arm64 with `agent_task_eval.py
    swebench --arch arm64 prepare_images`.
  - SWE-bench's Dockerfiles carry the aarch64 conda builds, but its command
    line asks for x86_64; the wrapper sets the arch.

| Image | Id |
| --- | --- |
| `sweb.base.py.arm64:latest` | `sha256:e49a87226b9dc2c27fabf153d0a0898023049fa2099e29ec01664c2e90c61e14` |
| `sweb.eval.arm64.psf__requests-2931:latest` | `sha256:43b8be44f34ed010d2a5e5890539e18bceb61aa2a5b03b55d87aa9f20ec619b7` |
| `sweb.eval.arm64.pytest-dev__pytest-7982:latest` | `sha256:95a30bbfc0e9d10f7f14c44874ac0dd8a96aebfb1828ba5bdd1dcec8566e6141` |
| `sweb.eval.arm64.sympy__sympy-13480:latest` | `sha256:f4e4bbfdb73d50bf7d3d6dd36e6e58ad277c777a3f9fa31aad806c35c0fe902a` |

- **Harbor:** 0.25.0, dataset `terminal-bench/terminal-bench-2-1`, task
  `terminal-bench/fix-git`.
  - Harbor installed each agent in the task container itself, with
    `--ak version=`.

## Results

| Run | Harness | Benchmark | Resolved | Empty patches | Errors | Receipts |
| --- | --- | --- | ---: | ---: | ---: | --- |
| e2e-cc | Claude Code | SWE-bench Verified, 3 tasks | 2 / 3 | 1 | 0 | `swebench-arm64/e2e-cc/` |
| e2e-cx | Codex | SWE-bench Verified, 3 tasks | 2 / 3 | 1 | 0 | `swebench-arm64/e2e-cx/` |
| stub-cc-tb21 | Claude Code | Terminal-Bench 2.1, fix-git | 0 / 1 | – | 0 | `terminal-bench-2-1/stub-cc-tb21/` |
| stub-cx-tb21 | Codex | Terminal-Bench 2.1, fix-git | 0 / 1 | – | 0 | `terminal-bench-2-1/stub-cx-tb21/` |

All four match the script. Each SWE-bench run directory holds:

- `run.json` and `runs.jsonl`;
- `predictions.jsonl` and `patches/`;
- the agent transcripts in `logs/`;
- SWE-bench's report and per-instance logs in `eval/`, including
  `test_output.txt`;
- `score.json`.

Each Terminal-Bench directory is Harbor's job directory: the trial
`result.json`, the agent's logs and the verifier's output.

Two of `stub-cx-tb21`'s Harbor logs are kept on the hub, not in the
repository: its `job.log` and `fix-git__mFJ7mJz/trial.log`.
- **Where:** `/srv/workspace/sparkpipe/agent-tasks/stub-20261105/terminal-bench-2-1/stub-cx-tb21/`.
- **Check:** `stub-cx-tb21/hub-logs.SHA256SUMS` lists both files. It was
  checked there with `sha256sum -c` on 2026-11-09.
- **Why:** the logs record the Codex installer's own `ln -sf` inside its
  task container. The repository's no-symlinks check
  (`tests/test_no_symlinks.py`) rejects any file that creates a link, and
  it makes no exception for evidence while an archive exists.
`stub-requests.jsonl` is the stub's log of every request. Each entry records
the path, the step, the action taken, whether `X-SparkPipe-Public: 1` was
sent, and the first 12 hex digits of the presented key's SHA-256.

- **Every agent turn carried the public header and a key.** That covers all
  four runs and Codex's `/v1/codex/models` fetch.
- **No key value appears anywhere in these receipts.** The key was a
  throwaway made for the stub, valid nowhere.

## Commands

From `~/sparkpipe-agent-e2e`. The tools were run from this branch. `swe4` is
a virtualenv with `swebench==4.1.0`, and
`DOCKER_HOST=unix://$HOME/.colima/default/docker.sock` was set.

```sh
python3 tools/agent_endpoint_stub.py --port 18433 --model stub-model --dataset stub-dataset.jsonl \
  --solve smoke__calc-1,psf__requests-2931,sympy__sympy-13480 --log stub-requests.jsonl
swe4/bin/python tools/agent_task_eval.py swebench --arch arm64 prepare_images --dataset_name princeton-nlp/SWE-bench_Verified \
  --instance_ids psf__requests-2931 pytest-dev__pytest-7982 sympy__sympy-13480 --max_workers 2 --namespace none --tag latest --env_image_tag latest
python3 tools/agent_task_eval.py run --tasks swebench-verified-50.json --harness claude-code --model stub-model \
  --endpoint http://host.docker.internal:18433 --add-host host.docker.internal:host-gateway --key-file stub.key --agents agents \
  --context 262144 --timeout 600 --parallel 3 --image-template 'sweb.eval.arm64.{instance_id_lower}:latest' \
  --instance psf__requests-2931 --instance pytest-dev__pytest-7982 --instance sympy__sympy-13480 --label e2e-cc --output runs/e2e-cc
(cd runs/e2e-cc/eval && swe4/bin/python tools/agent_task_eval.py swebench --arch arm64 run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path runs/e2e-cc/predictions.jsonl --run_id e2e-cc --instance_ids psf__requests-2931 pytest-dev__pytest-7982 sympy__sympy-13480 \
  --namespace none --max_workers 3 --report_dir runs/e2e-cc/eval)
python3 tools/agent_task_eval.py score --run runs/e2e-cc --report runs/e2e-cc/eval/claude-code__stub-model.e2e-cc.json
python3 tools/agent_task_eval.py agent-config --harness claude-code --endpoint http://192.168.5.2:18433 --model stub-model --context 262144 --output cc-stub.json
harbor run -d terminal-bench/terminal-bench-2-1 -i terminal-bench/fix-git -a claude-code -m stub-model --ak version=2.1.311 \
  --ak config=cc-stub.json --env-file cc.env -n 1 -k 1 -o jobs --job-name stub-cc-tb21
python3 tools/agent_task_eval.py harbor-score --job jobs/stub-cc-tb21 --harness claude-code --model stub-model --label stub-cc-tb21 --output stub-cc-tb21.score.json
```

In order, the commands:
1. start the stub;
2. build the arm64 images;
3. run Claude Code on the three tasks;
4. grade the patches;
5. score the run;
6. write Harbor's settings;
7. run the Terminal-Bench task;
8. score the job.

The Codex runs are the same with `--harness codex` and `-a codex --ak
version=0.166.0 --ak config=codex-stub.toml`. `cc.env` held
`ANTHROPIC_BASE_URL` and `ANTHROPIC_AUTH_TOKEN`, and `cx.env` held
`SPARKPIPE_API_KEY`; both were mode 600.

## What this found and fixed

- **The stub's streams were reset.** Its first version ended streams by
  closing the connection, and Claude Code saw `ECONNRESET` through colima's
  port forward. The stub now streams with chunked transfer encoding, as real
  servers do.
- **`--instance` scored against the wrong set.** `score` divided by every
  task in the file even when `--instance` limited the run. `run.json` now
  records only the tasks asked for.
- **Harbor needs `docker compose`**, and its task filter takes the dataset's
  prefixed names, such as `terminal-bench/fix-git`.
- **Claude Code prints `[claude-code:unrecognized_model]`** to stderr for a
  model name it does not know, then works normally.
