#!/usr/bin/env python3
import re
import subprocess
import sys

LAZY_ATTACH = re.compile(r"weightd lazy-attach model=(\S+) experts=(\d+) arena=(\d+) pool=(\d+)")
SPINE_RECEIPTS = "/tmp/spark-weightd-spine"
FAILURES = []


def ssh(node, command, timeout=30):
    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", node, command],
            capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout.strip()
    except (subprocess.TimeoutExpired, OSError) as error:
        return 1, str(error)


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS: {name} {detail}".rstrip())
    else:
        print(f"  FAIL: {name} {detail}".rstrip())
        FAILURES.append(f"{name}: {detail}")


def lazy_attach(node):
    rc, out = ssh(node, "grep 'weightd lazy-attach' ~/weightd.log | tail -1")
    match = LAZY_ATTACH.search(out) if rc == 0 else None
    check("weightd attached the pack lazily", match is not None, f"rc={rc} line={out!r}")
    if match is None:
        return None
    model = match.group(1)
    experts, arena, pool = (int(value) for value in match.groups()[1:])
    check("expert table is non-empty", experts > 0, f"model={model} experts={experts}")
    check("arena is non-empty", arena > 0, f"arena={arena}")
    check("expert pool budget is non-zero", pool > 0, f"pool={pool}")
    return arena


def spine_receipt(node, arena):
    rc, out = ssh(node, f"ls {SPINE_RECEIPTS}")
    names = out.split() if rc == 0 else []
    pattern = re.compile(rf"[0-9a-f]{{64}}-{arena}-\d+\.receipt")
    check("spine receipt exists for the attached pack", any(pattern.fullmatch(name) for name in names),
          f"arena={arena} receipts={len(names)}")


def main():
    if len(sys.argv) != 2:
        print("usage: test_expert_io_perf.py <node>", file=sys.stderr)
        return 2
    node = sys.argv[1]
    print(f"=== weightd expert residency on {node} ===")
    arena = lazy_attach(node)
    if arena is not None:
        spine_receipt(node, arena)
    print(f"=== summary: {len(FAILURES)} failures ===")
    for failure in FAILURES:
        print(f"  {failure}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
