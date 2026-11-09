#!/usr/bin/env python3
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("agent_task_eval", ROOT / "tools" / "agent_task_eval.py")
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)

KEY = "sk-sp-" + "q" * 43
FAKE_DOCKER = r'''#!/usr/bin/env python3
import json, os, sys
arguments = sys.argv[1:]
with open(os.environ["FAKE_DOCKER_LOG"], "a") as handle:
    handle.write(json.dumps(arguments) + "\n")
state = os.environ["FAKE_DOCKER_STATE"]
if arguments[0] == "rm":
    sys.exit(0)
if arguments[0] == "run":
    sys.exit(1 if "broken/image:latest" in arguments else 0)
index = 1
while arguments[index].startswith("-"):
    index += 2 if arguments[index] in ("-w", "-e") else 1
name, rest = arguments[index], arguments[index + 1:]
if rest[:2] == ["git", "-C"]:
    print(os.environ["FAKE_BASE"])
    sys.exit(0)
script = rest[2] if rest[:2] == ["sh", "-c"] else ""
if "cat >" in script:
    target = script.split("cat > ")[1].strip("'")
    os.makedirs(os.path.join(state, name), exist_ok=True)
    with open(os.path.join(state, name, os.path.basename(target)), "w") as handle:
        handle.write(sys.stdin.read())
elif "/claude -p" in script:
    print(json.dumps({"type": "system", "subtype": "init", "model": "kimi-k3"}))
    print(json.dumps({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}]}}))
    print("not json")
    print(json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "done"}]}}))
    print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "num_turns": 2, "duration_ms": 1234,
                      "usage": {"input_tokens": 10, "cache_read_input_tokens": 900, "output_tokens": 50}}))
    sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
elif "/codex exec" in script:
    for event in ({"type": "thread.started"}, {"type": "turn.started"}, {"type": "item.completed", "item": {"type": "command_execution"}},
                  {"type": "item.completed", "item": {"type": "file_change"}}, {"type": "turn.completed", "usage": {"input_tokens": 700, "cached_input_tokens": 600, "output_tokens": 40}}):
        print(json.dumps(event))
elif "git diff --cached" in script:
    sys.stdout.write("diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n")
'''
FAKE_NPM = r'''#!/usr/bin/env python3
import io, os, sys, tarfile
destination = sys.argv[sys.argv.index("--pack-destination") + 1]
spec = sys.argv[-1]
name = "codex" if spec.startswith("@openai/codex") else "claude"
path = "package/vendor/aarch64-unknown-linux-musl/codex/codex" if name == "codex" else "package/claude"
archive = os.path.join(destination, name + ".tgz")
with tarfile.open(archive, "w:gz") as bundle:
    for member, size in ((path, 2 << 20), ("package/README.md", 10)):
        info = tarfile.TarInfo(member)
        info.size = size
        info.mode = 0o755
        bundle.addfile(info, io.BytesIO(b"x" * size))
print(name + ".tgz")
'''


class Checks:
    def __init__(self):
        self.failures = []

    def __call__(self, condition, message):
        if not condition:
            self.failures.append(message)


def rows():
    made = []
    for repo, count in (("a/a", 6), ("b/b", 3), ("c/c", 1)):
        for index in range(count):
            made.append({"instance_id": f"{repo.split('/')[0]}__{repo.split('/')[0]}-{index}", "repo": repo, "difficulty": "<15 min fix",
                         "image": f"swebench/sweb.eval.x86_64.{repo.split('/')[0]}-{index}:latest", "base_commit": f"commit{index}",
                         "version": "1.0", "problem_statement": f"Problem {repo} {index}", "patch": "gold", "test_patch": "tests"})
    return made


def arguments(*extra):
    return tool.build_parser().parse_args(list(extra))


def docker_calls(log):
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def main():
    check = Checks()
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        dataset = tmp / "verified.jsonl"
        dataset.write_text("".join(json.dumps(row) + "\n" for row in rows()))
        tasks = tmp / "tasks.json"
        tool.select(arguments("select", "--dataset", str(dataset), "--count", "5", "--seed", "s1", "--output", str(tasks)))
        chosen = json.loads(tasks.read_text())
        repos = sorted(task["repo"] for task in chosen["tasks"])
        check(repos == ["a/a", "a/a", "a/a", "b/b", "b/b"], f"the stratified sample is wrong: {repos}")
        check(chosen["dataset_rows"] == 10 and len(chosen["dataset_sha256"]) == 64 and "patch" not in chosen["tasks"][0],
              "the task file lacks the dataset identity or leaks the gold patch")
        again = tmp / "again.json"
        tool.select(arguments("select", "--dataset", str(dataset), "--count", "5", "--seed", "s1", "--output", str(again)))
        check(again.read_text() == tasks.read_text(), "the same seed did not choose the same tasks")
        base = {task["instance_id"]: task["base_commit"] for task in chosen["tasks"]}

        bin_dir = tmp / "bin"
        bin_dir.mkdir()
        docker = bin_dir / "docker"
        docker.write_text(FAKE_DOCKER)
        docker.chmod(0o755)
        npm = bin_dir / "npm"
        npm.write_text(FAKE_NPM)
        npm.chmod(0o755)
        os.environ["PATH"] = f"{bin_dir}:{os.environ['PATH']}"
        agents = tmp / "agents"
        tool.prepare_agents(arguments("prepare-agents", "--platform", "linux-arm64", "--claude-code", "2.1.311", "--codex", "0.166.0", "--output", str(agents)))
        versions = (agents / "VERSIONS").read_text()
        check(os.access(agents / "claude", os.X_OK) and os.access(agents / "codex", os.X_OK), "prepare-agents did not place both executables")
        check("@anthropic-ai/claude-code-linux-arm64@2.1.311" in versions and "@openai/codex@0.166.0-linux-arm64" in versions,
              f"VERSIONS does not name the platform packages: {versions}")

        key = tmp / "key"
        key.write_text(f"k_bench {KEY}\n")
        key.chmod(0o644)
        try:
            tool.read_key(key)
            check(False, "a world-readable key file was accepted")
        except SystemExit:
            pass
        key.chmod(0o600)
        log, state = tmp / "docker.log", tmp / "state"
        os.environ.update(FAKE_DOCKER_LOG=str(log), FAKE_DOCKER_STATE=str(state))
        claude_out = tmp / "claude"
        first = chosen["tasks"][0]
        os.environ["FAKE_BASE"] = first["base_commit"]
        common = ["--tasks", str(tasks), "--model", "kimi-k3", "--endpoint", "http://host.docker.internal:18433", "--key-file", str(key),
                  "--agents", str(agents), "--context", "1048576", "--add-host", "host.docker.internal:host-gateway"]
        tool.run(arguments("run", *common, "--harness", "claude-code", "--label", "k3cc", "--output", str(claude_out), "--limit", "2"))
        records = [json.loads(line) for line in (claude_out / "runs.jsonl").read_text().splitlines()]
        check(len(records) == 2 and all(record["error"] is None for record in records), f"two claude-code tasks did not run cleanly: {records}")
        by_id = {record["instance_id"]: record for record in records}
        check(by_id[first["instance_id"]]["base_commit_ok"] is True, "the base commit check failed for the matching task")
        check(sum(record["base_commit_ok"] for record in records) == sum(base[record["instance_id"]] == first["base_commit"] for record in records),
              "a mismatched base commit was reported as matching")
        record = by_id[first["instance_id"]]
        check(record["agent"]["result"]["num_turns"] == 2 and record["agent"]["tool_uses"] == 1 and record["agent"]["assistant_messages"] == 2,
              f"the Claude Code transcript was parsed wrong: {record['agent']}")
        check(record["patch_bytes"] > 0 and (claude_out / "patches" / f"{first['instance_id']}.diff").read_text().startswith("diff --git"),
              "the patch was not saved")
        predictions = [json.loads(line) for line in (claude_out / "predictions.jsonl").read_text().splitlines()]
        check(len(predictions) == 2 and predictions[0]["model_name_or_path"] == "claude-code__kimi-k3" and predictions[0]["model_patch"].startswith("diff"),
              "predictions.jsonl is not in the SWE-bench format")
        name = tool.container_name("k3cc", first["instance_id"])
        settings = json.loads((state / name / "settings.json").read_text())
        check(settings["env"]["ANTHROPIC_BASE_URL"] == "http://host.docker.internal:18433" and settings["env"]["ANTHROPIC_MODEL"] == "kimi-k3"
              and settings["env"]["ANTHROPIC_CUSTOM_HEADERS"] == "X-SparkPipe-Public: 1" and "/run/sparkpipe-key" in settings["apiKeyHelper"],
              f"the Claude Code settings are wrong: {settings}")
        check("Problem" in (state / name / "sparkpipe-task.md").read_text(), "the task prompt was not written")
        calls = docker_calls(log)
        check(not any(KEY in " ".join(call) for call in calls), "the key appeared in a docker argument")
        check(not any(KEY in path.read_text() for path in state.rglob("*") if path.is_file()), "the key was written into a container file")
        runs = [call for call in calls if call[0] == "run"]
        check(all(f"{key.resolve()}:/run/sparkpipe-key:ro" in call and f"{agents}:/opt/sparkpipe-agents:ro" in call and
                  "--add-host" in call for call in runs), "the container does not mount the key and agents read-only")
        for instance in by_id:
            container = tool.container_name("k3cc", instance)
            touched = [index for index, call in enumerate(calls) if container in call]
            check(calls[touched[-1]][:2] == ["rm", "-f"], f"{container} was not removed last")
        tool.run(arguments("run", *common, "--harness", "claude-code", "--label", "k3cc", "--output", str(claude_out)))
        records = [json.loads(line) for line in (claude_out / "runs.jsonl").read_text().splitlines()]
        check(len(records) == 5 and len({record["instance_id"] for record in records}) == 5, "a rerun did not resume with only the unfinished tasks")

        stopped = tmp / "stopped"
        tool.run(arguments("run", *common, "--harness", "claude-code", "--label", "k3s", "--output", str(stopped), "--stop-after", "0.000001"))
        check(not (stopped / "runs.jsonl").exists() and not any(call[0] == "run" and any("k3s" in part for part in call) for call in docker_calls(log)),
              "a task started after --stop-after")
        picked = tmp / "picked"
        tool.run(arguments("run", *common, "--harness", "claude-code", "--label", "k3p", "--output", str(picked), "--instance", first["instance_id"]))
        check(json.loads((picked / "run.json").read_text())["tasks"] == [first["instance_id"]] and
              [json.loads(line)["instance_id"] for line in (picked / "runs.jsonl").read_text().splitlines()] == [first["instance_id"]],
              "--instance did not limit both the run and its planned set")
        os.environ["FAKE_EXIT"] = "124"
        timed = tmp / "timed"
        tool.run(arguments("run", *common, "--harness", "claude-code", "--label", "k3t", "--output", str(timed), "--limit", "1"))
        check(json.loads((timed / "runs.jsonl").read_text())["timed_out"] is True, "exit status 124 was not recorded as a timeout")
        os.environ["FAKE_EXIT"] = "0"

        codex_out = tmp / "codex"
        tool.run(arguments("run", *common, "--harness", "codex", "--label", "k3cx", "--output", str(codex_out), "--limit", "1"))
        record = json.loads((codex_out / "runs.jsonl").read_text())
        check(record["agent"]["turns"] == 1 and record["agent"]["usage"]["cached_input_tokens"] == 600 and record["agent"]["items"]["file_change"] == 1,
              f"the Codex events were parsed wrong: {record['agent']}")
        config = (state / tool.container_name("k3cx", record["instance_id"]) / "config.toml").read_text()
        check('base_url = "http://host.docker.internal:18433/v1"' in config and 'env_key = "SPARKPIPE_API_KEY"' in config
              and 'wire_api = "responses"' in config and 'http_headers = { "X-SparkPipe-Public" = "1" }' in config
              and "model_auto_compact_token_limit = 1015808" in config, f"the Codex config is wrong:\n{config}")
        codex_exec = [call for call in docker_calls(log) if call[0] == "exec" and any("/codex exec" in part for part in call)]
        check(codex_exec and 'SPARKPIPE_API_KEY="$(' in codex_exec[0][-1], "Codex does not read its key from the mounted file")

        broken = tmp / "broken.json"
        document = json.loads(tasks.read_text())
        document["tasks"] = [dict(document["tasks"][0], image="broken/image:latest")]
        broken.write_text(json.dumps(document))
        failed = tmp / "failed"
        tool.run(arguments("run", *(common[:1] + [str(broken)] + common[2:]), "--harness", "codex", "--label", "bad", "--output", str(failed)))
        record = json.loads((failed / "runs.jsonl").read_text())
        check(record["error"].startswith("container:"), f"a failed container start was not recorded: {record}")
        check(docker_calls(log)[-1][:2] == ["rm", "-f"], "a failed container was not removed")

        report = tmp / "report.json"
        resolved = [task["instance_id"] for task in chosen["tasks"][:2]]
        report.write_text(json.dumps({"resolved_ids": resolved + ["not__planned-1"], "empty_patch_ids": [], "error_ids": [chosen["tasks"][4]["instance_id"]],
                                      "infra_failure_ids": []}))
        tool.score(arguments("score", "--run", str(claude_out), "--report", str(report)))
        result = json.loads((claude_out / "score.json").read_text())
        check(result["resolved"] == 2 and result["planned"] == 5 and result["pass_rate"] == 0.4 and result["evaluation_errors"] == 1,
              f"the score is wrong: {result}")
        table = subprocess.run([sys.executable, str(ROOT / "tools" / "agent_task_eval.py"), "report", str(claude_out)], capture_output=True, text=True).stdout
        check("| kimi-k3 | claude-code | 2 / 5 | 40% |" in table, f"the report row is wrong:\n{table}")
        job = tmp / "jobs" / "k3-tb21"
        for name, task, reward, exception in (("a__1", "a", 1.0, None), ("a__2", "a", 0.0, None), ("b__1", "b", 0.0, "AgentTimeoutError"), ("c__1", "c", 1.0, None)):
            (job / name).mkdir(parents=True)
            (job / name / "result.json").write_text(json.dumps({"task_name": task, "trial_name": name, "verifier_result": {"rewards": {"reward": reward}},
                                                                 "exception_info": {"exception_type": exception} if exception else None,
                                                                 "agent_result": {"n_input_tokens": 100, "n_cache_tokens": 80, "n_output_tokens": 10}}))
        (job / "result.json").write_text(json.dumps({"id": "job", "stats": {}}))
        scored = tmp / "tb21.json"
        tool.harbor_score(arguments("harbor-score", "--job", str(job), "--harness", "claude-code", "--model", "kimi-k3", "--label", "k3tb", "--output", str(scored)))
        second = tmp / "jobs" / "k3-tb21-b"
        (second / "d__1").mkdir(parents=True)
        (second / "d__1" / "result.json").write_text(json.dumps({"task_name": "d", "trial_name": "d__1", "verifier_result": {"rewards": {"reward": 1.0}}}))
        merged = tmp / "tb21-merged.json"
        tool.harbor_score(arguments("harbor-score", "--job", str(job), "--job", str(second), "--harness", "claude-code", "--model", "kimi-k3", "--label", "k3tb",
                                    "--output", str(merged)))
        check(json.loads(merged.read_text())["planned"] == 4 and json.loads(merged.read_text())["resolved"] == 2.5, "harbor-score did not merge two jobs")
        result = json.loads(scored.read_text())
        check(result["planned"] == 3 and result["trials"] == 4 and result["resolved"] == 1.5 and result["pass_rate"] == 0.5,
              f"the Harbor score does not average attempts per task: {result}")
        check(result["agent_timeouts"] == 1 and result["tokens"]["n_cache_tokens"] == 320 and result["passed_tasks"] == ["a", "c"],
              f"the Harbor score lost timeouts, tokens or passed tasks: {result}")
        table = subprocess.run([sys.executable, str(ROOT / "tools" / "agent_task_eval.py"), "report", str(scored)], capture_output=True, text=True).stdout
        check("| kimi-k3 | claude-code | 1.5 / 3 | 50% |" in table, f"the Harbor report row is wrong:\n{table}")
        native = tmp / "settings.json"
        tool.agent_config(arguments("agent-config", "--harness", "claude-code", "--endpoint", "http://127.0.0.1:18433", "--model", "glm-5.3",
                                    "--context", "786432", "--output", str(native)))
        written = json.loads(native.read_text())
        check("apiKeyHelper" not in written and written["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:18433"
              and written["env"]["ANTHROPIC_CUSTOM_HEADERS"] == "X-SparkPipe-Public: 1" and written["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "786432",
              f"the Harbor Claude Code settings are wrong: {written}")
        native = tmp / "config.toml"
        tool.agent_config(arguments("agent-config", "--harness", "codex", "--endpoint", "http://127.0.0.1:18433", "--model", "glm-5.3",
                                    "--context", "786432", "--output", str(native)))
        check(native.read_text() == tool.codex_config("http://127.0.0.1:18433", "glm-5.3", 786432, 32768), "the Harbor Codex config differs from the runner's")
        internal = tool.claude_settings("http://127.0.0.1:18433", "glm-5.3", 786432, key_helper=False, door="internal")
        check("ANTHROPIC_CUSTOM_HEADERS" not in internal["env"] and "X-SparkPipe-Public" not in tool.codex_config("http://x", "m", 1000, 10, "internal"),
              "the internal door still sends the public header")
    for failure in check.failures:
        print(f"FAIL test_agent_task_eval: {failure}")
    if check.failures:
        raise SystemExit(1)
    print("PASS test_agent_task_eval: the SWE-bench sample is stratified by repository and repeatable; prepare-agents places the pinned "
          "Linux binaries; each task runs Claude Code or Codex in its own container with the key and agents mounted read-only, never in an "
          "argument or a container file, configured for the public door; transcripts, patches and SWE-bench predictions are saved, timeouts and "
          "failed starts recorded, every container removed, no task starts after --stop-after, --instance limits the run and its planned set, a rerun resumes, the score reads the SWE-bench report, and Harbor "
          "Terminal-Bench jobs score as the mean pass rate per task")


if __name__ == "__main__":
    main()
