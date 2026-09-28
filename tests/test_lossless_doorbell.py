#!/usr/bin/env python3
import re
import subprocess
import sys

MESH_RANKS = 16
WIRED = re.compile(r"WD-WIRED rank=(\d+) peer=(\d+) ")
COMPLETIONS = re.compile(r"WD-MESH-CQ ok=(\d+) err=(\d+)")
FAILURES = []


def ssh(node, command, timeout=30):
    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", node, command],
            capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout
    except (subprocess.TimeoutExpired, OSError) as error:
        return 1, str(error)


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS: {name} {detail}".rstrip())
    else:
        print(f"  FAIL: {name} {detail}".rstrip())
        FAILURES.append(f"{name}: {detail}")


def main():
    if len(sys.argv) != 2:
        print("usage: test_lossless_doorbell.py <node>", file=sys.stderr)
        return 2
    node = sys.argv[1]
    rc, log = ssh(node, "test -r ~/weightd.log && { grep -E 'WD-WIRED |WD-MESH-CQ|WD-SHIP-POST-FAIL' ~/weightd.log || true; }")
    check("weightd log readable", rc == 0, f"rc={rc}")
    if rc != 0:
        return 1
    print(f"=== lossless doorbell relay on {node} ===")
    wired = [(int(rank), int(peer)) for rank, peer in WIRED.findall(log)]
    ranks = {rank for rank, _ in wired}
    check("one local mesh rank", len(ranks) == 1, f"ranks={sorted(ranks)}")
    if len(ranks) == 1:
        rank = ranks.pop()
        peers = {peer if peer < rank else peer + 1 for _, peer in wired}
        expected = set(range(MESH_RANKS)) - {rank}
        check("every peer rank wired", peers == expected, f"rank={rank} missing={sorted(expected - peers)} unexpected={sorted(peers - expected)}")
    completions = [(int(ok), int(err)) for ok, err in COMPLETIONS.findall(log)]
    check("relay completions reported", len(completions) > 0, f"reports={len(completions)}")
    check("no failed completions", all(err == 0 for _, err in completions), f"last={completions[-1] if completions else None}")
    check("completion counter never regresses", all(later[0] >= earlier[0] for earlier, later in zip(completions, completions[1:])),
          f"reports={len(completions)}")
    check("no completion errors", "WD-MESH-CQERR" not in log, f"count={log.count('WD-MESH-CQERR')}")
    check("no failed ship posts", "WD-SHIP-POST-FAIL" not in log, f"count={log.count('WD-SHIP-POST-FAIL')}")
    print(f"=== summary: {len(FAILURES)} failures ===")
    for failure in FAILURES:
        print(f"  {failure}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
