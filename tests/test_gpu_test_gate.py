"""tools/gpu_test_gate.py selects every GPU-only test and classifies results.

CI's host-tests job has no GPU, so GPU-only tests skip there or are not in
`make test` at all, and a test that only a GPU host runs can rot unseen. The
gate runs that set on a GPU host. This test checks, without a GPU, that its
plan holds every Python test that asks for an sm_121 device and no other test
beyond the listed prerequisite ones, that it runs every `make test-*` GPU
target, that each artifact it builds for a prerequisite test has a rule, and
that its verdicts mean what the gate reports, that a base at the head's
commit shares the head's worktree while any other base gets its own, that a
head/ or base/ directory another repository or an earlier run left behind is
replaced by this repository's own worktree, and that it hands its kernel-log mark to the shared rule, tools/nvrm_kernel_log.sh, and
takes that rule's verdict. It also checks the memory stop: a command that
runs while available memory is under 20 GiB, or past its timeout, has its whole
process group killed; a stop ends the gate with exit 3, a STOP line and the
summary; and a gate killed outright still leaves every verdict it decided in
its output.
"""
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("gpu_test_gate", ROOT / "tools" / "gpu_test_gate.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)

DRIVER = """
import importlib.util, os, signal, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("gpu_test_gate", sys.argv[1])
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
work, mode = Path(sys.argv[2]), sys.argv[3]
checks = []
gate.plan = lambda tree: {"python": ["first", "second", "third"], "targets": [], "build": []}
gate.trees = lambda repository, work_dir, head, base: (work, work, [])
gate.kernel_log_rule = lambda root: "rule"
gate.kernel_log_mark = lambda rule: "mark"
gate.kernel_log_check = lambda rule, mark: (True, "kernel log: no new lines")
def available():
    checks.append(1)
    return 1 if mode == "stop" and len(checks) == 3 else 64 * 1024 * 1024
def run_logged(command, tree, log, name):
    if mode == "killed" and name == "second":
        os.kill(os.getpid(), signal.SIGKILL)
    return 0, 0.5
gate.available_kib = available
gate.run_logged = run_logged
sys.exit(gate.main(["--base", "HEAD", "--head", "HEAD", "--work", str(work)]))
"""


def gone(pid):
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        if subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip().startswith("Z"):
            return True
        time.sleep(0.1)
    return False


def check_stops(failures):
    with tempfile.TemporaryDirectory(prefix="gpu-gate-stop-") as scratch:
        scratch = Path(scratch)
        family = 'sleep 60 & echo $! > child.pid; echo $$ > shell.pid; wait'
        original = gate.available_kib
        calls = []
        gate.available_kib = lambda: calls.append(1) or (1 if len(calls) >= 2 else 64 * 1024 * 1024)
        started = time.monotonic()
        try:
            gate.run_logged(["sh", "-c", family], scratch, scratch / "low.log", "a test")
            failures.append("a command running under 20 GiB of available memory was not stopped")
        except gate.Stopped as stop:
            if str(stop) != "during a test":
                failures.append("the stop names %r, not the command it stopped" % str(stop))
        finally:
            gate.available_kib = original
        if time.monotonic() - started > 15:
            failures.append("a command under the memory floor ran on for %.0f s instead of being killed within seconds" % (time.monotonic() - started))
        pids = [int((scratch / name).read_text()) for name in ("shell.pid", "child.pid") if (scratch / name).is_file()]
        if len(pids) != 2 or not all(gone(pid) for pid in pids):
            failures.append("stopping a command under the memory floor left part of its process group running: %s" % pids)
        if "available memory fell below 20 GiB" not in (scratch / "low.log").read_text():
            failures.append("the stopped command's log does not say why it was killed")
        gate.available_kib = lambda: 64 * 1024 * 1024
        timeout = gate.TEST_TIMEOUT_SECONDS
        gate.TEST_TIMEOUT_SECONDS = 1
        try:
            for name in ("shell.pid", "child.pid"):
                (scratch / name).unlink(missing_ok=True)
            code, seconds = gate.run_logged(["sh", "-c", family], scratch, scratch / "slow.log", "a slow test")
            pids = [int((scratch / name).read_text()) for name in ("shell.pid", "child.pid") if (scratch / name).is_file()]
            if code != 124 or seconds > 15 or len(pids) != 2 or not all(gone(pid) for pid in pids):
                failures.append("a command past its timeout did not exit 124 with its whole process group killed within seconds: %s %.0f s %s" % (code, seconds, pids))
            code, _ = gate.run_logged(["sh", "-c", "exit 7"], scratch, scratch / "quick.log", "a quick test")
            if code != 7:
                failures.append("a command that ends on its own did not return its exit code: %s" % code)
        finally:
            gate.available_kib = original
            gate.TEST_TIMEOUT_SECONDS = timeout
        for mode, wanted_code, wanted in (
                ("stop", 3, ["PASS         python  first head=0 base=- 0s", "PASS         python  second head=0 base=- 0s",
                             "STOP         available memory below 20 GiB before third; the gate ends here", "kernel log: no new lines",
                             "gpu test gate: 2 run, 2 PASS; kernel log clean; stopped before third, 1 of 3 commands not run"]),
                ("killed", -9, ["PASS         python  first head=0 base=- 0s"])):
            run = subprocess.run([sys.executable, "-c", DRIVER, str(ROOT / "tools" / "gpu_test_gate.py"), str(scratch / mode), mode],
                                 capture_output=True, text=True, timeout=60)
            lines = [line for line in run.stdout.splitlines() if not line.startswith("base and head")]
            if run.returncode != wanted_code or lines != wanted:
                failures.append("a gate %s exited %d with %r, expected %d with %r" % (mode, run.returncode, lines, wanted_code, wanted))


def main():
    failures = []
    selected = gate.plan(ROOT)
    python_tests = gate.make_variable(ROOT, "PYTHON_TESTS")
    for test in python_tests:
        path = ROOT / test
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")
        asks = test.endswith("_cuda.py") or re.search(r"^\s*(?:from|import)\s+gpu_test_runner\b", text, re.M) or "query-gpu=" + "compute_cap" in text
        if asks and test not in selected["python"] and test not in gate.GPU_PYTHON_PARKED:
            failures.append("%s asks for an sm_121 device but the gate does not run it" % test)
    for test, reason in sorted(gate.GPU_PYTHON_PARKED.items()):
        if test not in python_tests:
            failures.append("parked test %s is gone from make test; drop it from the parked list" % test)
        if test in selected["python"] or selected["parked"].get(test) != reason:
            failures.append("the gate runs parked test %s or does not report it as parked" % test)
        if not reason.startswith("TECHDEBT.md"):
            failures.append("parked test %s names no TECHDEBT entry as its reason" % test)
        environment = {key: value for key, value in os.environ.items() if key != "SPARK_RUN_PARKED"}
        run = subprocess.run([sys.executable, str(ROOT / test)], capture_output=True, text=True, timeout=60, env=environment)
        if run.returncode != 0 or not run.stdout.startswith("SKIP ") or "TECHDEBT.md" not in run.stdout:
            failures.append("parked test %s runs without SPARK_RUN_PARKED=1: exit %d %r" % (test, run.returncode, run.stdout[:200]))
    for test in selected["python"]:
        text = (ROOT / test).read_text(errors="replace")
        if test not in gate.PREREQUISITE_TESTS and not gate.wants_device(test, text):
            failures.append("the gate runs %s, which needs no GPU" % test)
        if test not in python_tests:
            failures.append("the gate runs %s, which make test does not list" % test)
    makefile = (ROOT / "Makefile").read_text()
    targets = set(re.findall(r"^(test-[a-z0-9-]+):", makefile, re.M))
    for target in gate.GPU_TARGET_EXCLUDED:
        if target not in targets:
            failures.append("excluded target %s is gone from the Makefile; drop it from the exclusions" % target)
    for target in sorted(targets - set(gate.GPU_TARGET_EXCLUDED)):
        if target not in selected["targets"]:
            failures.append("the gate does not run make %s" % target)
    for artifact in selected["build"]:
        if not re.search(r"^" + re.escape(artifact) + r":", makefile, re.M):
            failures.append("no Makefile rule builds %s" % artifact)
    cases = ((True, None, "PASS"), (True, True, "PASS"), (True, False, "FIXED"), (False, True, "REGRESSION"), (False, False, "KNOWN"))
    for head, base, expected in cases:
        if gate.classify(head, base) != expected:
            failures.append("classify(head=%s, base=%s) gave %s, expected %s" % (head, base, gate.classify(head, base), expected))
    with tempfile.TemporaryDirectory(prefix="gpu-gate-") as scratch:
        rule = Path(scratch) / "rule.sh"
        rule.write_text('case "$1" in mark) echo 12.5;; check) [ "$2" = 12.5 ] && { echo "FAIL Xid 13"; exit 1; }; exit 2;; esac\n')
        mark = gate.kernel_log_mark(rule)
        clean, report = gate.kernel_log_check(rule, mark)
        if mark != "12.5" or clean or report != "FAIL Xid 13":
            failures.append("the gate does not pass its mark to the kernel-log rule or read its verdict: %r %r %r" % (mark, clean, report))
    with tempfile.TemporaryDirectory(prefix="gpu-gate-trees-") as scratch:
        repository = Path(scratch) / "repository"
        git = ["git", "-c", "user.name=gate", "-c", "user.email=gate@example.invalid", "-C", str(repository)]
        gate.subprocess.run(["git", "init", "-q", str(repository)], check=True)
        gate.subprocess.run(git + ["commit", "-q", "--allow-empty", "-m", "first"], check=True)
        gate.subprocess.run(git + ["tag", "first"], check=True)
        gate.subprocess.run(git + ["commit", "-q", "--allow-empty", "-m", "second"], check=True)
        head, base, builds = gate.trees(repository, Path(scratch) / "same", "HEAD", "HEAD")
        if base != head or [label for label, _ in builds] != ["head"]:
            failures.append("a base at the head's commit does not reuse the head's worktree: %s %s %s" % (head, base, builds))
        head, base, builds = gate.trees(repository, Path(scratch) / "two", "HEAD", "first")
        if base == head or [label for label, _ in builds] != ["head", "base"] or gate.commit_of(base, "HEAD") != gate.commit_of(repository, "first"):
            failures.append("a base at another commit does not get its own worktree at that commit: %s %s %s" % (head, base, builds))
        other = Path(scratch) / "other"
        other_git = ["git", "-c", "user.name=gate", "-c", "user.email=gate@example.invalid", "-C", str(other)]
        gate.subprocess.run(["git", "init", "-q", str(other)], check=True)
        gate.subprocess.run(other_git + ["commit", "-q", "--allow-empty", "-m", "other"], check=True)
        stale = Path(scratch) / "stale"
        gate.trees(other, stale, "HEAD", "HEAD")
        (stale / "base").mkdir()
        (stale / "base" / "leftover").write_text("not a worktree")
        try:
            head, base, builds = gate.trees(repository, stale, "HEAD", "first")
        except gate.subprocess.CalledProcessError as error:
            failures.append("a work directory left by another repository stops the gate: %s" % error)
        else:
            registered = gate.registered_worktrees(repository)
            if gate.commit_of(head, "HEAD") != gate.commit_of(repository, "HEAD") or gate.commit_of(base, "HEAD") != gate.commit_of(repository, "first") or \
                    head.resolve() not in registered or base.resolve() not in registered or (base / "leftover").exists():
                failures.append("stale head/ and base/ directories that are not worktrees of this repository are not replaced by its own worktrees")
    check_stops(failures)
    if gate.KERNEL_LOG_RULE != "tools/nvrm_kernel_log.sh":
        failures.append("the gate does not use the shared kernel-log rule tools/nvrm_kernel_log.sh")
    for failure in failures:
        print("  FAIL %s" % failure)
    if failures:
        print("\nFAIL (%d)" % len(failures))
        return 1
    print("PASS the GPU test gate plans %d Python tests and %d GPU targets, classifies results as reported, kills a command's process group "
          "under the memory floor or past its timeout, ends a stopped run with STOP and the summary, and keeps every decided verdict when killed" % (
              len(selected["python"]), len(selected["targets"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
