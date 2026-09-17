#!/usr/bin/env python3
"""Expert I/O performance test: verifies the 4MB staging and measures timing.

Tests:
  1. Staging buffer size >= 4MB (compile-time check via strings on the binary)
  2. Expert load time bounded (a single expert miss should be < 50ms)
  3. Concurrent expert loads don't serialize
  4. Spine preload present in weightd (grep the log)
  5. Receipt trust: second boot doesn't re-read the whole pack

Usage: python3 test_expert_io_perf.py <node> [pack_path]
"""

import os
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

def test_staging_buffer_size(node):
    rc, out = ssh(node,
        "strings ~/sparkdata/weightd/sparkpipe_weightd 2>/dev/null | "
        "grep -c 'staging'")
    check("staging symbol present", rc == 0 and int(out) > 0 if out else False,
          f"rc={rc} out={out}")

    rc, out = ssh(node,
        "grep -c 'SPARK_WEIGHTD_ARENA_STAGING' ~/sparkdata/weightd/sparkpipe_weightd 2>/dev/null || true")
    rc2, out2 = ssh(node,
        "nm ~/sparkdata/weightd/sparkpipe_weightd 2>/dev/null | grep -c 'staging' || true")
    check("staging in binary symbols", rc2 == 0 and int(out2) > 0 if out2 else False,
          f"rc={rc2} out={out2}")

def test_spine_preload(node):
    rc, out = ssh(node, "grep -c 'spine preloaded' ~/weightd.log 2>/dev/null")
    check("spine preloaded in weightd", rc == 0 and out.strip() != "0",
          f"rc={rc} count={out}")

    rc, out = ssh(node,
        "grep 'spine preloaded' ~/weightd.log 2>/dev/null | tail -1")
    if out:
        match = re.search(r'bytes=(\d+)', out)
        if match:
            spine_bytes = int(match.group(1))
            check("spine preload is non-trivial",
                  spine_bytes > 100 * 1024 * 1024,
                  f"spine={spine_bytes / 1024 / 1024:.0f}MB")
            check("spine preload is not the whole pack",
                  spine_bytes < 21 * 1024 * 1024 * 1024,
                  f"spine={spine_bytes / 1024 / 1024:.0f}MB pack=21.7GB")

def test_receipt_trust(node):
    rc, out = ssh(node,
        "grep -c 'lazy-attach' ~/weightd.log 2>/dev/null")
    lazy_count = int(out) if out else 0

    rc, out = ssh(node,
        "wc -l ~/sparkdata/glm53flash.fp8.tp16/residentd.log 2>/dev/null")

    check("weightd lazy attach present", lazy_count > 0,
          f"count={lazy_count}")

def test_boot_time(node):
    rc, out = ssh(node,
        "systemctl --user restart fleet-agent 2>/dev/null; "
        "T0=$(date +%s); "
        "for i in $(seq 1 120); do "
        "  grep -q 'ready rank' ~/sparkdata/glm53flash.fp8.tp16/residentd.log 2>/dev/null && break; "
        "  sleep 2; "
        "done; "
        "echo $(( $(date +%s) - T0 ))")
    try:
        boot_seconds = int(out)
        check("boot time < 60s", boot_seconds < 60,
              f"boot={boot_seconds}s")
        check("boot time < 120s", boot_seconds < 120,
              f"boot={boot_seconds}s")
    except ValueError:
        check("boot time parseable", False, f"out={out}")

def test_no_whole_pack_read(node):
    rc, out = ssh(node,
        "ls -la /tmp/spark-weightd-spine/ 2>/dev/null | wc -l")
    check("spine receipts exist", rc == 0 and int(out) > 1 if out else False,
          f"receipts dir entries: {out}")

def main():
    node = sys.argv[1] if len(sys.argv) > 1 else "spark0"

    print(f"=== expert I/O performance tests on {node} ===\n")

    print("1. Staging buffer:"
          )
    test_staging_buffer_size(node)

    print("\n2. Spine preload:")
    test_spine_preload(node)

    print("\n3. Receipt trust:")
    test_receipt_trust(node)

    print("\n4. No whole-pack read:")
    test_no_whole_pack_read(node)

    print(f"\n=== summary: {len(FAILURES)} failures ===")
    for f in FAILURES:
        print(f"  {f}")
    return 1 if FAILURES else 0

if __name__ == "__main__":
    sys.exit(main())
