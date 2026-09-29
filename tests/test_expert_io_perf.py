#!/usr/bin/env python3
import re
import subprocess
import sys

LAZY_ATTACH = re.compile(r"weightd lazy-attach model=(\S+) experts=(\d+) arena=(\d+) pool=(\d+)")
PACK_VERIFY = re.compile(r"weightd pack-verify path=(\S+) mode=(receipt|sha256|ck128) ")
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
    rc, out = ssh(node, "grep 'weightd pack-verify path=' ~/weightd.log | grep ' mode=' | tail -1")
    match = PACK_VERIFY.search(out) if rc == 0 else None
    check("weightd verified the attached pack", match is not None, f"rc={rc} line={out!r}")
    if match is None:
        return
    path = match.group(1)
    rc, out = ssh(node, f"r=$(readlink -f {path}).verified; test -f $r && echo $r || ls ~/.local/state/sparkpipe/verified")
    check("verify-once receipt exists for the attached pack", rc == 0 and out != "", f"arena={arena} path={path} mode={match.group(2)} out={out!r}")


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
