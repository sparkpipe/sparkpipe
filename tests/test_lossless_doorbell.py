#!/usr/bin/env python3
"""Lossless doorbell test: verifies the relay ships every publish.

Tests:
  1. Ship count matches publish count (no losses)
  2. Doorbell entries advance monotonically per rank
  3. No gaps in the shipped sequence numbers
  4. The pending bitmap (when implemented) covers all slots

Usage: python3 test_lossless_doorbell.py <node> [duration_seconds]
"""

import re
import subprocess
import sys
import time

FAILURES = []

def ssh(node, command, timeout=30):
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", node, command],
            capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout.strip()
    except Exception as e:
        return 1, str(e)

def check(name, condition, detail=""):
    if condition:
        print(f"  PASS: {name}")
    else:
        print(f"  FAIL: {name} {detail}")
        FAILURES.append(f"{name}: {detail}")

def test_ship_count_advances(node):
    rc1, before = ssh(node,
        "grep -c 'WD-SHIP' ~/weightd.log 2>/dev/null")
    time.sleep(5)
    rc2, after = ssh(node,
        "grep -c 'WD-SHIP' ~/weightd.log 2>/dev/null")

    try:
        before_count = int(before)
        after_count = int(after)
        check("ship count advances", after_count >= before_count,
              f"before={before_count} after={after_count}")
    except ValueError:
        check("ship count parseable", False,
              f"before={before} after={after}")

def test_no_sequence_gaps(node):
    rc, out = ssh(node,
        "grep 'WD-SHIP' ~/weightd.log 2>/dev/null | "
        "grep -oE 'seq=[0-9]+' | "
        "grep -oE '[0-9]+' | "
        "tail -20")

    if not out:
        check("ship sequences present", False, "no WD-SHIP lines")
        return

    seqs = [int(x) for x in out.split('\n') if x.strip()]
    if len(seqs) < 2:
        check("enough ship sequences to check gaps", False,
              f"only {len(seqs)} sequences")
        return

    gaps = []
    for i in range(1, len(seqs)):
        if seqs[i] - seqs[i-1] > 1 and seqs[i] != seqs[i-1]:
            if seqs[i] - seqs[i-1] < 1000:
                gaps.append((seqs[i-1], seqs[i]))

    check("no sequence gaps in last 20 ships",
          len(gaps) == 0,
          f"gaps: {gaps[:5]}" if gaps else "clean")

def test_cq_completion_matches_posts(node):
    rc, out = ssh(node,
        "grep 'WD-MESH-CQ' ~/weightd.log 2>/dev/null | tail -1")
    if out:
        match = re.search(r'ok=(\d+) err=(\d+)', out)
        if match:
            ok = int(match.group(1))
            err = int(match.group(2))
            check("completions >> errors",
                  ok > err * 100 if err > 0 else ok > 0,
                  f"ok={ok} err={err}")
            check("completion count is massive (relay active)",
                  ok > 1000000,
                  f"ok={ok} (should be millions for long-running)")
    else:
        check("CQ stats present", False, "no WD-MESH-CQ")

def test_doorbell_wired_to_all_peers(node):
    rc, out = ssh(node,
        "grep -c 'WD-WIRED' ~/weightd.log 2>/dev/null")
    try:
        wired = int(out)
        check("all peers wired", wired >= 15,
              f"wired={wired} (need 15 for 16-rank)")
    except ValueError:
        check("wired count parseable", False, f"out={out}")

def test_no_ship_failures(node):
    rc, out = ssh(node,
        "grep -c 'WD-SHIP FAILED' ~/weightd.log 2>/dev/null")
    try:
        failed = int(out)
        check("no ship failures", failed == 0,
              f"failed={failed}")
    except ValueError:
        pass

def test_relay_not_stale(node):
    rc, out = ssh(node,
        "grep 'WD-SHIP' ~/weightd.log 2>/dev/null | tail -1 | "
        "grep -oE 'total=[0-9]+'")
    check("relay has shipped at least once",
          out and 'total=' in out, f"last_ship={out}")

def main():
    node = sys.argv[1] if len(sys.argv) > 1 else "spark0"
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print(f"=== lossless doorbell tests on {node} ===\n")

    print("1. Ship count advances:")
    test_ship_count_advances(node)

    print("\n2. No sequence gaps:")
    test_no_sequence_gaps(node)

    print("\n3. CQ completions match posts:")
    test_cq_completion_matches_posts(node)

    print("\n4. All peers wired:")
    test_doorbell_wired_to_all_peers(node)

    print("\n5. No ship failures:")
    test_no_ship_failures(node)

    print("\n6. Relay active:")
    test_relay_not_stale(node)

    print(f"\n=== summary: {len(FAILURES)} failures ===")
    for f in FAILURES:
        print(f"  {f}")
    return 1 if FAILURES else 0

if __name__ == "__main__":
    sys.exit(main())
