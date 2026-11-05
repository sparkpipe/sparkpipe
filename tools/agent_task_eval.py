#!/usr/bin/env python3
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
from pathlib import Path

HARNESSES = ("claude-code", "codex")
PACKAGES = {"claude-code": "@anthropic-ai/claude-code-{platform}", "codex": "@openai/codex"}
CODEX_PLATFORM_TAGS = {"linux-arm64": "linux-arm64", "linux-x64": "linux-x64"}
AGENTS_MOUNT = "/opt/sparkpipe-agents"
KEY_MOUNT = "/run/sparkpipe-key"
KEY_PREFIX = "sk-sp-"
PUBLIC_HEADER = "X-SparkPipe-Public"
WORKDIR = "/testbed"
TASK_FILE = "/tmp/sparkpipe-task.md"
READ_KEY = f"tr ' ' '\\n' < {KEY_MOUNT} | grep '^{KEY_PREFIX}'"
NAME_CHARACTERS = re.compile(r"[^a-zA-Z0-9_.-]+")
PROMPT = """{problem}

The repository for this issue is checked out at {workdir}. Resolve the issue by changing the code there. The project's environment is the conda environment testbed: run `source /opt/miniconda3/bin/activate testbed` before running Python or the tests. Do not change the existing tests; the fix is judged by tests you cannot see. When the change is complete, stop.
"""


def read_key(path):
    mode = os.stat(path).st_mode & 0o777
    if mode & 0o077:
        raise SystemExit(f"agent_task_eval: {path} has mode {mode:o}; the key file must be readable by its owner only")
    keys = [word for word in Path(path).read_text().split() if word.startswith(KEY_PREFIX)]
    if len(keys) != 1:
        raise SystemExit(f"agent_task_eval: {path} must hold exactly one {KEY_PREFIX} key")
    return keys[0]


def load_rows(path):
    text = Path(path).read_text()
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def order_key(seed, instance_id):
    return hashlib.sha256(f"{seed}:{instance_id}".encode()).hexdigest()


def stratified(rows, count, seed):
    by_repo = {}
    for row in rows:
        by_repo.setdefault(row["repo"], []).append(row)
    total = len(rows)
    quotas = {repo: count * len(group) / total for repo, group in by_repo.items()}
    taken = {repo: int(quota) for repo, quota in quotas.items()}
    remainder = count - sum(taken.values())
    for repo in sorted(quotas, key=lambda name: (-(quotas[name] - taken[name]), name))[:remainder]:
        taken[repo] += 1
    chosen = []
    for repo in sorted(by_repo):
        group = sorted(by_repo[repo], key=lambda row: order_key(seed, row["instance_id"]))
        chosen.extend(group[:taken[repo]])
    return sorted(chosen, key=lambda row: order_key(seed, row["instance_id"]))


def select(arguments):
    rows = load_rows(arguments.dataset)
    chosen = stratified(rows, arguments.count, arguments.seed)
    digest = hashlib.sha256(Path(arguments.dataset).read_bytes()).hexdigest()
    fields = ("instance_id", "repo", "difficulty", "image", "base_commit", "version", "problem_statement")
    document = {"dataset": arguments.name, "dataset_rows": len(rows), "dataset_sha256": digest, "seed": arguments.seed, "count": len(chosen),
                "tasks": [{field: row.get(field) for field in fields} for row in chosen]}
    Path(arguments.output).write_text(json.dumps(document, indent=1) + "\n")
    repos = {}
    for row in chosen:
        repos[row["repo"]] = repos.get(row["repo"], 0) + 1
    print(json.dumps({"count": len(chosen), "repos": repos}))


def container_name(label, instance_id):
    return NAME_CHARACTERS.sub("-", f"ate-{label}-{instance_id}")[:120]


def claude_settings(endpoint, model, context, key_helper=True):
    settings = {"env": {
        "ANTHROPIC_BASE_URL": endpoint, "ANTHROPIC_MODEL": model, "ANTHROPIC_DEFAULT_HAIKU_MODEL": model, "CLAUDE_CODE_SUBAGENT_MODEL": model,
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(context), "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0", "CLAUDE_CODE_GATEWAY_HINT_HEADERS": "1", "ANTHROPIC_CUSTOM_HEADERS": f"{PUBLIC_HEADER}: 1",
        "DISABLE_AUTOUPDATER": "1", "IS_SANDBOX": "1"}}
    if key_helper:
        settings["apiKeyHelper"] = READ_KEY
    return settings


def toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{json.dumps(key)} = {toml_value(item)}" for key, item in value.items()) + " }"
    return json.dumps(value)


def codex_config(endpoint, model, context, output_tokens):
    root = [("model", model), ("model_provider", "sparkpipe"), ("model_context_window", context),
            ("model_auto_compact_token_limit", context - output_tokens), ("approval_policy", "never"), ("sandbox_mode", "danger-full-access")]
    provider = [("name", "SparkPipe"), ("base_url", endpoint + "/v1"), ("env_key", "SPARKPIPE_API_KEY"), ("wire_api", "responses"),
                ("stream_idle_timeout_ms", 600000), ("model_catalog_url", endpoint + "/v1/codex/models"), ("http_headers", {PUBLIC_HEADER: "1"})]
    lines = [f"{key} = {toml_value(value)}" for key, value in root]
    lines += ["", "[model_providers.sparkpipe]"] + [f"{key} = {toml_value(value)}" for key, value in provider]
    return "\n".join(lines) + "\n"


def agent_command(harness, timeout, max_turns):
    if harness == "claude-code":
        turns = f" --max-turns {int(max_turns)}" if max_turns else ""
        return (f'exec timeout {int(timeout)} {AGENTS_MOUNT}/claude -p "$(cat {TASK_FILE})" --output-format stream-json --verbose '
                f"--dangerously-skip-permissions{turns} < /dev/null")
    return (f'SPARKPIPE_API_KEY="$({READ_KEY})" exec timeout {int(timeout)} {AGENTS_MOUNT}/codex exec --json --skip-git-repo-check '
            f'--dangerously-bypass-approvals-and-sandbox -C {WORKDIR} "$(cat {TASK_FILE})" < /dev/null')


def parse_claude(lines):
    summary = {"assistant_messages": 0, "tool_uses": 0, "result": None}
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "assistant":
            summary["assistant_messages"] += 1
            content = (event.get("message") or {}).get("content") or []
            summary["tool_uses"] += sum(isinstance(block, dict) and block.get("type") == "tool_use" for block in content)
        elif event.get("type") == "result":
            summary["result"] = {key: event.get(key) for key in ("subtype", "is_error", "num_turns", "duration_ms", "duration_api_ms", "usage")}
    return summary


def parse_codex(lines):
    summary = {"turns": 0, "items": {}, "usage": {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}, "errors": []}
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "turn.completed":
            summary["turns"] += 1
            for key in summary["usage"]:
                summary["usage"][key] += int((event.get("usage") or {}).get(key) or 0)
        elif kind == "item.completed":
            item = (event.get("item") or {}).get("type") or (event.get("item") or {}).get("item_type") or "unknown"
            summary["items"][item] = summary["items"].get(item, 0) + 1
        elif kind in ("turn.failed", "error"):
            summary["errors"].append(event.get("error") or event.get("message"))
    return summary


class Docker:
    def __init__(self, executable="docker"):
        self.executable = executable

    def call(self, arguments, stdin=None, timeout=None, stdout=None):
        return subprocess.run([self.executable] + arguments, input=stdin, stdout=stdout or subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)

    def write(self, name, path, text):
        directory = os.path.dirname(path)
        command = f"mkdir -p {shlex.quote(directory)} && cat > {shlex.quote(path)}"
        done = self.call(["exec", "-i", name, "sh", "-c", command], stdin=text.encode())
        if done.returncode != 0:
            raise RuntimeError(f"writing {path} failed: {done.stderr.decode(errors='replace')[-300:]}")


def run_task(task, arguments, docker, output, lock, deadline):
    if deadline is not None and time.monotonic() >= deadline:
        return None
    name = container_name(arguments.label, task["instance_id"])
    record = {"instance_id": task["instance_id"], "repo": task.get("repo"), "difficulty": task.get("difficulty"), "harness": arguments.harness,
              "model": arguments.model, "image": task["image"], "label": arguments.label, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    started = time.monotonic()
    log_path = output / "logs" / f"{task['instance_id']}.jsonl"
    docker.call(["rm", "-f", name])
    try:
        run = ["run", "-d", "--name", name, "-w", WORKDIR, "-v", f"{arguments.agents}:{AGENTS_MOUNT}:ro",
               "-v", f"{Path(arguments.key_file).resolve()}:{KEY_MOUNT}:ro"]
        for host in arguments.add_host or []:
            run += ["--add-host", host]
        if arguments.network:
            run += ["--network", arguments.network]
        made = docker.call(run + [task["image"], "sleep", "infinity"], timeout=1800)
        if made.returncode != 0:
            record["error"] = "container: " + made.stderr.decode(errors="replace")[-400:]
            return record
        head = docker.call(["exec", name, "git", "-C", WORKDIR, "rev-parse", "HEAD"]).stdout.decode().strip()
        record["base_commit_ok"] = head == task.get("base_commit")
        docker.write(name, TASK_FILE, PROMPT.format(problem=task["problem_statement"], workdir=WORKDIR))
        if arguments.harness == "claude-code":
            docker.write(name, "/root/.claude/settings.json", json.dumps(claude_settings(arguments.endpoint, arguments.model, arguments.context), indent=1))
        else:
            docker.write(name, "/root/.codex/config.toml", codex_config(arguments.endpoint, arguments.model, arguments.context, arguments.output_tokens))
        agent_started = time.monotonic()
        with open(log_path, "wb") as log, open(output / "logs" / f"{task['instance_id']}.stderr", "wb") as errors:
            done = subprocess.run([docker.executable, "exec", "-w", WORKDIR, "-e", "HOME=/root", name, "sh", "-c",
                                   agent_command(arguments.harness, arguments.timeout, arguments.max_turns)], stdout=log, stderr=errors,
                                  timeout=arguments.timeout + 300)
        record.update(agent_seconds=round(time.monotonic() - agent_started, 1), exit_code=done.returncode, timed_out=done.returncode == 124)
        lines = log_path.read_text(errors="replace").splitlines()
        record["agent"] = parse_claude(lines) if arguments.harness == "claude-code" else parse_codex(lines)
        diff = docker.call(["exec", name, "sh", "-c", f"cd {WORKDIR} && git add -A && git diff --cached --binary HEAD"])
        patch = diff.stdout.decode(errors="replace")
        (output / "patches" / f"{task['instance_id']}.diff").write_text(patch)
        record.update(patch_bytes=len(patch.encode()), patch_sha256=hashlib.sha256(patch.encode()).hexdigest(), error=None)
        with lock:
            with open(output / "predictions.jsonl", "a") as handle:
                handle.write(json.dumps({"instance_id": task["instance_id"], "model_name_or_path": f"{arguments.harness}__{arguments.model}",
                                         "model_patch": patch}) + "\n")
        return record
    except (RuntimeError, subprocess.TimeoutExpired, OSError) as error:
        record["error"] = f"{type(error).__name__}: {error}"[-400:]
        return record
    finally:
        docker.call(["rm", "-f", name])
        record["seconds"] = round(time.monotonic() - started, 1)


def finished(output):
    path = output / "runs.jsonl"
    if not path.exists():
        return set()
    return {json.loads(line)["instance_id"] for line in path.read_text().splitlines() if line.strip()}


def run(arguments):
    read_key(arguments.key_file)
    if arguments.harness not in HARNESSES:
        raise SystemExit(f"agent_task_eval: harness is one of {', '.join(HARNESSES)}")
    agent = Path(arguments.agents) / ("claude" if arguments.harness == "claude-code" else "codex")
    if not os.access(agent, os.X_OK):
        raise SystemExit(f"agent_task_eval: {agent} is not an executable; run prepare-agents first")
    output = Path(arguments.output)
    for directory in ("logs", "patches"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    document = json.loads(Path(arguments.tasks).read_text())
    done = finished(output)
    tasks = [task for task in document["tasks"] if task["instance_id"] not in done][:arguments.limit or None]
    meta = output / "run.json"
    if not meta.exists():
        meta.write_text(json.dumps({"label": arguments.label, "harness": arguments.harness, "model": arguments.model, "endpoint": arguments.endpoint,
                                    "dataset": document.get("dataset"), "dataset_sha256": document.get("dataset_sha256"), "seed": document.get("seed"),
                                    "tasks": [task["instance_id"] for task in document["tasks"]], "agent_sha256": hashlib.sha256(agent.read_bytes()).hexdigest(),
                                    "agent_version": (Path(arguments.agents) / "VERSIONS").read_text() if (Path(arguments.agents) / "VERSIONS").exists() else None,
                                    "context": arguments.context, "timeout_s": arguments.timeout, "max_turns": arguments.max_turns,
                                    "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1) + "\n")
    docker, lock = Docker(arguments.docker), threading.Lock()
    deadline = time.monotonic() + arguments.stop_after if arguments.stop_after else None
    skipped = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=arguments.parallel) as pool:
        futures = [pool.submit(run_task, task, arguments, docker, output, lock, deadline) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            if record is None:
                skipped += 1
                continue
            with lock:
                with open(output / "runs.jsonl", "a") as handle:
                    handle.write(json.dumps(record) + "\n")
            print(json.dumps({key: record.get(key) for key in ("instance_id", "seconds", "exit_code", "timed_out", "patch_bytes", "error")}), flush=True)
    if skipped:
        print(json.dumps({"stopped_after_s": arguments.stop_after, "not_started": skipped}), flush=True)


def score(arguments):
    output = Path(arguments.run)
    report = json.loads(Path(arguments.report).read_text())
    runs = [json.loads(line) for line in (output / "runs.jsonl").read_text().splitlines() if line.strip()]
    meta = json.loads((output / "run.json").read_text())
    planned = meta["tasks"]
    resolved = set(report.get("resolved_ids") or [])
    empty = set(report.get("empty_patch_ids") or [])
    errors = set(report.get("error_ids") or [])
    infra = set(report.get("infra_failure_ids") or [])
    agent_failed = {run["instance_id"] for run in runs if run.get("error")}
    timed_out = {run["instance_id"] for run in runs if run.get("timed_out")}
    result = {"label": meta["label"], "harness": meta["harness"], "model": meta["model"], "planned": len(planned), "ran": len({run["instance_id"] for run in runs}),
              "resolved": len(resolved & set(planned)), "pass_rate": round(len(resolved & set(planned)) / len(planned), 4) if planned else None,
              "empty_patches": len(empty), "evaluation_errors": len(errors), "infrastructure_failures": len(infra),
              "agent_failures": len(agent_failed), "agent_timeouts": len(timed_out), "resolved_ids": sorted(resolved & set(planned)),
              "report": str(arguments.report), "report_sha256": hashlib.sha256(Path(arguments.report).read_bytes()).hexdigest(),
              "agent_seconds_median": sorted(run.get("agent_seconds") or 0 for run in runs)[len(runs) // 2] if runs else None}
    (output / "score.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "resolved_ids"}))


def trial_results(job):
    trials = []
    for path in sorted(Path(job).glob("*/result.json")):
        document = json.loads(path.read_text())
        if "task_name" in document and "trial_name" in document:
            trials.append(document)
    return trials


def harbor_score(arguments):
    trials = trial_results(arguments.job)
    if not trials:
        raise SystemExit(f"agent_task_eval: {arguments.job} holds no Harbor trial results")
    tasks = {}
    for trial in trials:
        rewards = (trial.get("verifier_result") or {}).get("rewards") or {}
        reward = rewards.get("reward")
        failed = trial.get("exception_info") is not None
        tasks.setdefault(trial["task_name"], []).append({"reward": reward, "exception": (trial.get("exception_info") or {}).get("exception_type") if failed else None})
    passes = {name: sum((attempt["reward"] or 0) >= 1 for attempt in attempts) / len(attempts) for name, attempts in tasks.items()}
    exceptions = {}
    for attempts in tasks.values():
        for attempt in attempts:
            if attempt["exception"]:
                exceptions[attempt["exception"]] = exceptions.get(attempt["exception"], 0) + 1
    tokens = {key: sum(int((trial.get("agent_result") or {}).get(key) or 0) for trial in trials) for key in ("n_input_tokens", "n_cache_tokens", "n_output_tokens")}
    result = {"label": arguments.label, "harness": arguments.harness, "model": arguments.model, "benchmark": arguments.benchmark, "planned": len(tasks),
              "trials": len(trials), "resolved": round(sum(passes.values()), 2), "pass_rate": round(sum(passes.values()) / len(tasks), 4),
              "empty_patches": 0, "evaluation_errors": sum(exceptions.values()), "infrastructure_failures": 0, "agent_timeouts": exceptions.get("AgentTimeoutError", 0),
              "exceptions": exceptions, "tokens": tokens, "agent_seconds_median": None, "passed_tasks": sorted(name for name, rate in passes.items() if rate > 0),
              "job": str(arguments.job)}
    Path(arguments.output).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "passed_tasks"}))


def agent_config(arguments):
    if arguments.harness == "claude-code":
        text = json.dumps(claude_settings(arguments.endpoint, arguments.model, arguments.context, key_helper=False), indent=1) + "\n"
    else:
        text = codex_config(arguments.endpoint, arguments.model, arguments.context, arguments.output_tokens)
    Path(arguments.output).write_text(text)
    print(arguments.output)


def report(arguments):
    rows = ["| Model | Harness | Resolved | Pass rate | Empty patches | Agent timeouts | Evaluation errors | Median agent time |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for path in arguments.runs:
        path = Path(path)
        result = json.loads((path / "score.json" if path.is_dir() else path).read_text())
        median = "–" if result["agent_seconds_median"] is None else f"{result['agent_seconds_median']} s"
        rows.append(f"| {result['model']} | {result['harness']} | {result['resolved']:g} / {result['planned']} | {result['pass_rate']:.0%} | "
                    f"{result['empty_patches']} | {result['agent_timeouts']} | {result['evaluation_errors']} | {median} |")
    print("\n".join(rows))


def prepare_agents(arguments):
    target = Path(arguments.output)
    target.mkdir(parents=True, exist_ok=True)
    versions = []
    with tempfile.TemporaryDirectory() as scratch:
        for harness, version in (("claude-code", arguments.claude_code), ("codex", arguments.codex)):
            if not version:
                continue
            if harness == "claude-code":
                spec = PACKAGES[harness].format(platform=arguments.platform) + "@" + version
            else:
                spec = f"{PACKAGES[harness]}@{version}-{CODEX_PLATFORM_TAGS[arguments.platform]}"
            packed = subprocess.run(["npm", "pack", "--silent", "--pack-destination", scratch, spec], capture_output=True, text=True, check=True)
            archive = Path(scratch) / packed.stdout.strip().splitlines()[-1]
            unpacked = Path(scratch) / harness
            with tarfile.open(archive) as bundle:
                bundle.extractall(unpacked, filter="data")
            name = "claude" if harness == "claude-code" else "codex"
            candidates = sorted((path for path in unpacked.rglob(name) if path.is_file() and path.stat().st_size > 1 << 20), key=lambda path: -path.stat().st_size)
            if not candidates:
                raise SystemExit(f"agent_task_eval: {spec} holds no {name} executable")
            destination = target / name
            shutil.copyfile(candidates[0], destination)
            destination.chmod(0o755)
            versions.append(f"{spec} {hashlib.sha256(destination.read_bytes()).hexdigest()} {candidates[0].relative_to(unpacked)}")
    (target / "VERSIONS").write_text("\n".join(versions) + "\n")
    print("\n".join(versions))


def build_parser():
    parser = argparse.ArgumentParser(description="Run Claude Code or Codex on SWE-bench tasks against a SparkPipe fleet and score the patches")
    commands = parser.add_subparsers(dest="command", required=True)
    chooser = commands.add_parser("select")
    chooser.add_argument("--dataset", required=True)
    chooser.add_argument("--name", default="SWE-bench/SWE-bench_Verified")
    chooser.add_argument("--count", type=int, default=50)
    chooser.add_argument("--seed", default="swebench-verified-1")
    chooser.add_argument("--output", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--tasks", required=True)
    runner.add_argument("--harness", required=True, choices=HARNESSES)
    runner.add_argument("--model", required=True)
    runner.add_argument("--endpoint", required=True)
    runner.add_argument("--key-file", required=True)
    runner.add_argument("--agents", required=True)
    runner.add_argument("--context", type=int, required=True)
    runner.add_argument("--output-tokens", type=int, default=32768)
    runner.add_argument("--timeout", type=int, default=3600)
    runner.add_argument("--max-turns", type=int, default=0)
    runner.add_argument("--parallel", type=int, default=1)
    runner.add_argument("--limit", type=int, default=0)
    runner.add_argument("--stop-after", type=float, default=0.0)
    runner.add_argument("--network")
    runner.add_argument("--add-host", action="append")
    runner.add_argument("--docker", default="docker")
    runner.add_argument("--label", required=True)
    runner.add_argument("--output", required=True)
    scorer = commands.add_parser("score")
    scorer.add_argument("--run", required=True)
    scorer.add_argument("--report", required=True)
    harbor = commands.add_parser("harbor-score")
    harbor.add_argument("--job", required=True)
    harbor.add_argument("--harness", required=True, choices=HARNESSES)
    harbor.add_argument("--model", required=True)
    harbor.add_argument("--benchmark", default="terminal-bench/terminal-bench-2-1")
    harbor.add_argument("--label", required=True)
    harbor.add_argument("--output", required=True)
    configure = commands.add_parser("agent-config")
    configure.add_argument("--harness", required=True, choices=HARNESSES)
    configure.add_argument("--endpoint", required=True)
    configure.add_argument("--model", required=True)
    configure.add_argument("--context", type=int, required=True)
    configure.add_argument("--output-tokens", type=int, default=32768)
    configure.add_argument("--output", required=True)
    reporter = commands.add_parser("report")
    reporter.add_argument("runs", nargs="+")
    preparer = commands.add_parser("prepare-agents")
    preparer.add_argument("--platform", required=True, choices=sorted(CODEX_PLATFORM_TAGS))
    preparer.add_argument("--claude-code")
    preparer.add_argument("--codex")
    preparer.add_argument("--output", required=True)
    return parser


def main():
    arguments = build_parser().parse_args()
    {"select": select, "run": run, "score": score, "harbor-score": harbor_score, "agent-config": agent_config, "report": report, "prepare-agents": prepare_agents}[arguments.command](arguments)


if __name__ == "__main__":
    main()
