#!/usr/bin/env python3
"""Run every GPU-only test on a GPU host, a head revision against a base.

CI's host-tests job has no GPU, so the Python tests that need an sm_121 device
skip there and the `make test-*` GPU targets never run. This gate runs that
whole set on a GPU host (a Lenovo or a GB10 node), in git worktrees of both
revisions built the way `make test` builds them, plus the artifacts some tests
name that `make test` does not build. Each test runs on the head; a test that
fails there runs on the base too, so every result is classified. When base
and head are the same commit, one worktree serves both:

  PASS        passes on the head
  REGRESSION  fails on the head and passes on the base
  KNOWN       fails on both; the base failure is a bug to fix, not to carry
  FIXED       passes on the head and fails on the base

The kernel log must pass tools/nvrm_kernel_log.sh, the one kernel-log rule
of the GPU gates: the gate takes its mark before the first build and checks it
after the last test. That log is host-wide, so the gate runs in a GPU slot held
for its whole run, where any new line is its own. Tests run at nice 19 and
best-effort I/O priority 7 with the default action for every signal, each in
its own process group. While a build or a test runs, the gate reads available
memory every second; below 20 GiB it kills that command's process group,
prints STOP naming the command, and ends with the kernel-log check and the
summary of what ran. Each verdict is printed as it is decided, so a gate that
is itself killed still leaves every finished result in its output. It exits 1
on a regression or a kernel-log failure, 3 when it had to stop, else 0.

Usage:
    tools/gpu_test_gate.sh --base REV --head REV [--work DIR] [--jobs N] [--plan]
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPU_MARKERS = (
    re.compile(r"^\s*(?:from\s+gpu_test_runner\s+import|import\s+gpu_test_runner)\b", re.M),
    re.compile(r"--query-gpu=compute_cap"),
)
GPU_TARGET_EXCLUDED = ("test-metal-kernels", "test-nvrm")
PREREQUISITE_TESTS = {
    "tests/test_host_turn_bench.py": "build/host_turn_bench",
    "tests/test_dsv41_flash_pack_contract.py": "build/dsv41_flash_pack_synthesize",
    "tests/test_spark_tiktoken_compile.py": "build/test_tiktoken_compiled",
}
HOST_CI_REQUIREMENTS = "tools/ci/host-tests-requirements.txt"
MEMORY_FLOOR_KIB = 20 * 1024 * 1024
MEMORY_POLL_SECONDS = 1.0
TEST_TIMEOUT_SECONDS = 1800
KERNEL_LOG_RULE = "tools/nvrm_kernel_log.sh"
MISSING_MODULE = re.compile(r"ModuleNotFoundError: No module named '([^']+)'")


def make_variable(tree, name):
    printer = "print-%:\n\t@echo $($*)\n"
    run = subprocess.run(["make", "-s", "--no-print-directory", "-f", "Makefile", "-f", "-", "print-" + name], cwd=tree,
                         input=printer, capture_output=True, text=True)
    if run.returncode != 0:
        raise SystemExit("make cannot print %s in %s: %s" % (name, tree, run.stderr.strip()[-300:]))
    return run.stdout.split()


def wants_device(test, text):
    return test.endswith("_cuda.py") or any(marker.search(text) for marker in GPU_MARKERS)


def gpu_python_tests(tree):
    found = []
    for test in make_variable(tree, "PYTHON_TESTS"):
        path = Path(tree) / test
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")
        if wants_device(test, text):
            found.append(test)
    return found


def gpu_make_targets(tree):
    text = (Path(tree) / "Makefile").read_text()
    targets = sorted(set(re.findall(r"^(test-[a-z0-9-]+):", text, re.M)))
    return [target for target in targets if target not in GPU_TARGET_EXCLUDED]


def plan(tree):
    return {
        "python": gpu_python_tests(tree) + sorted(PREREQUISITE_TESTS),
        "targets": gpu_make_targets(tree),
        "build": sorted(set(PREREQUISITE_TESTS.values())),
    }


def classify(head_passed, base_passed):
    if head_passed:
        return "PASS" if base_passed is None or base_passed else "FIXED"
    return "KNOWN" if base_passed is False else "REGRESSION"


def kernel_log_rule(root):
    rule = Path(root) / KERNEL_LOG_RULE
    if not rule.is_file():
        raise SystemExit("the kernel-log rule %s is missing from %s" % (KERNEL_LOG_RULE, root))
    return rule


def kernel_log_mark(rule):
    run = subprocess.run(["sh", str(rule), "mark"], capture_output=True, text=True)
    mark = run.stdout.strip()
    if run.returncode != 0 or not mark:
        raise SystemExit("%s mark failed: %s" % (rule, (run.stdout + run.stderr).strip()))
    return mark


def kernel_log_check(rule, mark):
    run = subprocess.run(["sh", str(rule), "check", mark], capture_output=True, text=True)
    return run.returncode == 0, (run.stdout + run.stderr).strip()


def available_kib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise SystemExit("no MemAvailable in /proc/meminfo")


class Stopped(Exception):
    pass


def default_signals():
    for number in (signal.SIGINT, signal.SIGQUIT, signal.SIGHUP, signal.SIGPIPE, signal.SIGTERM):
        signal.signal(number, signal.SIG_DFL)
    os.setpgid(0, 0)
    os.nice(19)


def kill_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_logged(command, tree, log, name):
    started = time.monotonic()
    with open(log, "w") as output:
        process = subprocess.Popen(command, cwd=tree, stdout=output, stderr=subprocess.STDOUT, preexec_fn=default_signals)
        while True:
            try:
                code = process.wait(timeout=MEMORY_POLL_SECONDS)
                break
            except subprocess.TimeoutExpired:
                pass
            if available_kib() < MEMORY_FLOOR_KIB:
                kill_group(process)
                output.write("\ngpu test gate: available memory fell below 20 GiB; this command's process group was killed\n")
                raise Stopped("during " + name)
            if time.monotonic() - started > TEST_TIMEOUT_SECONDS:
                kill_group(process)
                code = 124
                break
    return code, time.monotonic() - started


def commit_of(repository, revision):
    return subprocess.run(["git", "-C", str(repository), "rev-parse", revision + "^{commit}"], capture_output=True, text=True, check=True).stdout.strip()


def registered_worktrees(repository):
    listing = subprocess.run(["git", "-C", str(repository), "worktree", "list", "--porcelain"], capture_output=True, text=True, check=True).stdout
    return {Path(line[len("worktree "):]).resolve() for line in listing.splitlines() if line.startswith("worktree ")}


def worktree(repository, work, label, wanted):
    path = Path(work) / label
    if path.exists():
        if path.resolve() in registered_worktrees(repository):
            current = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
            if current == wanted:
                return path
            subprocess.run(["git", "-C", str(repository), "worktree", "remove", "--force", str(path)], check=True)
        else:
            shutil.rmtree(path)
    subprocess.run(["git", "-C", str(repository), "worktree", "prune"], check=True)
    subprocess.run(["git", "-C", str(repository), "worktree", "add", "--detach", str(path), wanted], check=True, capture_output=True)
    return path


def trees(repository, work, head_revision, base_revision):
    head_commit = commit_of(repository, head_revision)
    base_commit = commit_of(repository, base_revision)
    head = worktree(repository, work, "head", head_commit)
    if base_commit == head_commit:
        return head, head, [("head", head)]
    base = worktree(repository, work, "base", base_commit)
    return head, base, [("head", head), ("base", base)]


def build(tree, extra, jobs, log):
    targets = ["all"] + make_variable(tree, "TEST_BINARIES") + make_variable(tree, "PYTHON_TEST_BINARIES") + extra
    code, _ = run_logged(["ionice", "-c2", "-n7", "make", "-k", "-j%d" % jobs] + targets, tree, log, "the build in %s" % tree)
    missing = [target for target in extra if not (Path(tree) / target).exists()]
    if code != 0 or missing:
        raise SystemExit("build in %s failed (exit %d, missing %s); see %s" % (tree, code, " ".join(missing) or "nothing", log))


def run_one(kind, name, command, head, base, logs, pins):
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    head_code, seconds = run_logged(command, head, logs / ("head.%s.log" % stem), name)
    base_code = None
    if head_code != 0:
        base_code, _ = run_logged(command, base, logs / ("base.%s.log" % stem), name + " on the base")
    verdict = classify(head_code == 0, None if base_code is None else base_code == 0)
    module = missing_module(logs / ("head.%s.log" % stem)) if head_code != 0 else None
    if module is not None and verdict == "KNOWN" and module.lower() in pins:
        verdict = "HOST-CI-ONLY"
    print("%-12s %-7s %s head=%d base=%s %.0fs%s" % (verdict, kind, name, head_code, "-" if base_code is None else base_code, seconds,
          " needs %s, pinned in %s" % (module, HOST_CI_REQUIREMENTS) if verdict == "HOST-CI-ONLY" else ""))
    return verdict


def missing_module(log):
    match = MISSING_MODULE.search(Path(log).read_text(errors="replace"))
    return match.group(1) if match else None


def host_ci_pins(tree):
    path = Path(tree) / HOST_CI_REQUIREMENTS
    return {line.split("==")[0].strip().lower() for line in path.read_text().splitlines() if "==" in line} if path.is_file() else set()


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--work", default=str(Path.home() / "nvfree" / "gate"))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--plan", action="store_true")
    options = parser.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    selected = plan(ROOT)
    if options.plan:
        for test in selected["python"]:
            print("python %s" % test)
        for target in selected["targets"]:
            print("target %s" % target)
        for target in selected["build"]:
            print("build %s" % target)
        return 0
    work = Path(options.work)
    logs = work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    head, base, builds = trees(ROOT, work, options.head, options.base)
    if base == head:
        print("base and head are the same commit; one worktree serves both")
    selected = plan(head)
    rule = kernel_log_rule(ROOT)
    mark = kernel_log_mark(rule)
    commands = [("python", test, ["python3", test]) for test in selected["python"]]
    commands += [("target", target, ["make", "-s", target]) for target in selected["targets"]]
    results = []
    stopped = None
    try:
        for label, tree in builds:
            build(tree, selected["build"], options.jobs, logs / ("build.%s.log" % label))
        pins = host_ci_pins(head)
        for kind, name, command in commands:
            if available_kib() < MEMORY_FLOOR_KIB:
                raise Stopped("before " + name)
            results.append(run_one(kind, name, command, head, base, logs, pins))
    except Stopped as stop:
        stopped = str(stop)
        print("%-12s available memory below 20 GiB %s; the gate ends here" % ("STOP", stopped))
    kernel_log_clean, kernel_log_report = kernel_log_check(rule, mark)
    print(kernel_log_report)
    counts = {}
    for verdict in results:
        counts[verdict] = counts.get(verdict, 0) + 1
    print("gpu test gate: %d run, %s; kernel log %s%s" % (
        len(results), ", ".join("%d %s" % (counts[key], key) for key in sorted(counts)) or "nothing", "clean" if kernel_log_clean else "FAILED",
        "" if stopped is None else "; stopped %s, %d of %d commands not run" % (stopped, len(commands) - len(results), len(commands))))
    if stopped is not None:
        return 3
    return 1 if counts.get("REGRESSION") or not kernel_log_clean else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
