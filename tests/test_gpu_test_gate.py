"""tools/gpu_test_gate.py selects every GPU-only test and classifies results.

CI's host-tests job has no GPU, so GPU-only tests skip there or are not in
`make test` at all, and a test that only a GPU host runs can rot unseen. The
gate runs that set on a GPU host. This test checks, without a GPU, that its
plan holds every Python test that asks for an sm_121 device and no other test
beyond the listed prerequisite ones, that it runs every `make test-*` GPU
target, that each artifact it builds for a prerequisite test has a rule, and
that its verdicts mean what the gate reports, that a base at the head's
commit shares the head's worktree while any other base gets its own, and that
it hands its kernel-log mark to the shared rule, tools/nvrm_kernel_log.sh, and
takes that rule's verdict.
"""
import importlib.util
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("gpu_test_gate", ROOT / "tools" / "gpu_test_gate.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


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
        if asks and test not in selected["python"]:
            failures.append("%s asks for an sm_121 device but the gate does not run it" % test)
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
    if gate.KERNEL_LOG_RULE != "tools/nvrm_kernel_log.sh":
        failures.append("the gate does not use the shared kernel-log rule tools/nvrm_kernel_log.sh")
    for failure in failures:
        print("  FAIL %s" % failure)
    if failures:
        print("\nFAIL (%d)" % len(failures))
        return 1
    print("PASS the GPU test gate plans %d Python tests and %d GPU targets and classifies results as reported" % (len(selected["python"]), len(selected["targets"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
