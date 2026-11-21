#!/usr/bin/env python3
"""Run every GPU-only test on a GPU host, a head revision against a base.

CI's host-tests job has no GPU, so the Python tests that need an sm_121 device
skip there and the `make test-*` GPU targets never run. This gate runs that
whole set on a GPU host (a Lenovo or a GB10 node), in git worktrees of both
revisions built the way `make test` builds them, plus the artifacts some tests
name that `make test` does not build. Each test runs on the head; a test that
fails there runs on the base too, so every result is classified:

  PASS        passes on the head
  REGRESSION  fails on the head and passes on the base
  KNOWN       fails on both; the base failure is a bug to fix, not to carry
  FIXED       passes on the head and fails on the base

The kernel log must pass tools/nvrm_kernel_log.sh, the one kernel-log rule
of the GPU gates: the gate takes its mark before the first build and checks it
after the last test. That log is host-wide, so the gate runs in a GPU slot held
for its whole run, where any new line is its own. Tests run at nice 19 and best-effort I/O priority 7 with the default
action for every signal, and the gate stops when available memory falls below
20 GiB. It exits 1 on a regression or a kernel-log failure, 3 when it had to
stop, else 0.

Usage:
    tools/gpu_test_gate.sh --base REV --head REV [--work DIR] [--jobs N] [--plan]
"""
from __future__ import annotations

import argparse
import os
import re
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


def default_signals():
    for number in (signal.SIGINT, signal.SIGQUIT, signal.SIGHUP, signal.SIGPIPE, signal.SIGTERM):
        signal.signal(number, signal.SIG_DFL)
    os.nice(19)


def run_logged(command, tree, log):
    started = time.monotonic()
    with open(log, "w") as output:
        try:
            result = subprocess.run(command, cwd=tree, stdout=output, stderr=subprocess.STDOUT, preexec_fn=default_signals,
                                    timeout=TEST_TIMEOUT_SECONDS)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
    return code, time.monotonic() - started


def worktree(work, label, revision):
    path = Path(work) / label
    wanted = subprocess.run(["git", "-C", str(ROOT), "rev-parse", revision + "^{commit}"], capture_output=True, text=True, check=True).stdout.strip()
    if path.exists():
        current = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if current == wanted:
            return path
        subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force", str(path)], check=True)
    subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach", str(path), wanted], check=True, capture_output=True)
    return path


def build(tree, extra, jobs, log):
    targets = ["all"] + make_variable(tree, "TEST_BINARIES") + make_variable(tree, "PYTHON_TEST_BINARIES") + extra
    code, _ = run_logged(["ionice", "-c2", "-n7", "make", "-k", "-j%d" % jobs] + targets, tree, log)
    missing = [target for target in extra if not (Path(tree) / target).exists()]
    if code != 0 or missing:
        raise SystemExit("build in %s failed (exit %d, missing %s); see %s" % (tree, code, " ".join(missing) or "nothing", log))


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
    head = worktree(work, "head", options.head)
    base = worktree(work, "base", options.base)
    selected = plan(head)
    rule = kernel_log_rule(ROOT)
    mark = kernel_log_mark(rule)
    for label, tree in (("head", head), ("base", base)):
        build(tree, selected["build"], options.jobs, logs / ("build.%s.log" % label))
    pins = host_ci_pins(head)
    results = []
    stopped = False
    commands = [("python", test, ["python3", test]) for test in selected["python"]]
    commands += [("target", target, ["make", "-s", target]) for target in selected["targets"]]
    for kind, name, command in commands:
        if available_kib() < MEMORY_FLOOR_KIB:
            print("STOP available memory below 20 GiB before %s" % name)
            stopped = True
            break
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        head_code, seconds = run_logged(command, head, logs / ("head.%s.log" % stem))
        base_code = None
        if head_code != 0:
            base_code, _ = run_logged(command, base, logs / ("base.%s.log" % stem))
        verdict = classify(head_code == 0, None if base_code is None else base_code == 0)
        module = missing_module(logs / ("head.%s.log" % stem)) if head_code != 0 else None
        if module is not None and verdict == "KNOWN" and module.lower() in pins:
            verdict = "HOST-CI-ONLY"
        results.append((verdict, name, head_code, base_code, seconds, module))
        print("%-12s %-7s %s head=%d base=%s %.0fs%s" % (verdict, kind, name, head_code, "-" if base_code is None else base_code, seconds,
              "" if module is None else " needs %s, pinned in %s" % (module, HOST_CI_REQUIREMENTS) if verdict == "HOST-CI-ONLY" else ""))
    kernel_log_clean, kernel_log_report = kernel_log_check(rule, mark)
    print(kernel_log_report)
    counts = {}
    for verdict, *_ in results:
        counts[verdict] = counts.get(verdict, 0) + 1
    print("gpu test gate: %d run, %s; kernel log %s" % (
        len(results), ", ".join("%d %s" % (counts[key], key) for key in sorted(counts)) or "nothing", "clean" if kernel_log_clean else "FAILED"))
    if stopped:
        return 3
    return 1 if counts.get("REGRESSION") or not kernel_log_clean else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
