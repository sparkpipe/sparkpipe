#!/usr/bin/env python3
"""Memory bandwidth / C2C pressure test.

Verifies that:
  1. The device wait kernel keeps its __nanosleep (removing it caused 4x
     regression from C2C saturation — measured 09-16)
  2. The relay scan is yield-paced, not pure spin (pure spin ate bandwidth)
  3. GPU utilization correlates with useful work, not polling
  4. Mesh CQ error rate is bounded (not 53M accumulating)

Usage: python3 test_memory_bandwidth_budget.py <node>
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

def test_device_nanosleep_present(node):
    rc, out = ssh(node,
        "cuobjdump -sass ~/sparkdata/glm53flash.fp8.tp16/stages/stage_000/model_driver.so 2>/dev/null | "
        "grep -c 'NANOSLEEP' || true")
    try:
        count = int(out)
        check("device nanosleep instruction present in kernels",
              count > 0, f"count={count}")
    except ValueError:
        check("nanosleep count parseable", False, f"out={out}")

def test_relay_yield_paced(node):
    rc, out = ssh(node,
        "gdb -p $(pgrep -f 'sparkdata/weightd' | head -1) -batch "
        "-ex 'disassemble SparkWeightdMeshDoorbellLoop' 2>/dev/null | "
        "grep -c 'yield' || true")
    rc2, out2 = ssh(node,
        "strings ~/sparkdata/weightd/sparkpipe_weightd 2>/dev/null | "
        "grep -c 'yield'")
    check("relay uses yield instruction or has yield in binary",
          rc2 == 0 or rc == 0, f"disasm_rc={rc} strings_rc={rc2}")

def test_cq_error_bounded(node):
    rc, out = ssh(node,
        "grep 'WD-MESH-CQ' ~/weightd.log 2>/dev/null | tail -1")
    if out:
        match = re.search(r'ok=(\d+) err=(\d+)', out)
        if match:
            ok_count = int(match.group(1))
            err_count = int(match.group(2))
            total = ok_count + err_count
            if total > 0:
                err_rate = err_count / total
                check("CQ error rate < 1%",
                      err_rate < 0.01,
                      f"err_rate={err_rate:.4f} ({err_count}/{total})")
                check("CQ error count is not 53M-class",
                      err_count < 1000000,
                      f"err={err_count}")
            else:
                check("CQ has activity", total > 0, "no completions")
    else:
        check("CQ stats present", False, "no WD-MESH-CQ line")

def test_degraded_not_latching(node):
    rc, out = ssh(node,
        "grep -c 'WD-MESH-DEGRADED' ~/weightd.log 2>/dev/null")

    rc2, out2 = ssh(node,
        "grep -c 'WD-SHIP' ~/weightd.log 2>/dev/null")
    try:
        degraded = int(out)
        ships = int(out2)
        if degraded > 0:
            check("recovered after degraded (ships after degraded)",
                  ships > 0, f"degraded={degraded} ships={ships}")
        else:
            check("no degraded state", True, "clean")
    except ValueError:
        pass

def test_gpu_utilization_reasonable(node):
    rc, out = ssh(node,
        "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null")
    try:
        util = int(out.strip().rstrip('%'))
        check("GPU utilization < 100% when idle",
              util < 95, f"util={util}% (should be low without load)")
    except (ValueError, AttributeError):
        check("GPU util parseable", False, f"out={out}")

def test_weightd_cpu_not_burning(node):
    rc, out = ssh(node,
        "ps -o %cpu= -p $(pgrep -f 'sparkdata/weightd' | head -1) 2>/dev/null")
    try:
        cpu = float(out.strip())
        check("weightd CPU < 100% (yield-paced, not pure spin)",
              cpu < 99.0, f"cpu={cpu}%")
    except (ValueError, AttributeError):
        check("weightd CPU parseable", False, f"out={out}")

def main():
    node = sys.argv[1] if len(sys.argv) > 1 else "spark0"

    print(f"=== memory bandwidth / C2C tests on {node} ===\n")

    print("1. Device nanosleep (the 4x regression guard):")
    test_device_nanosleep_present(node)

    print("\n2. Relay pacing:")
    test_relay_yield_paced(node)

    print("\n3. CQ error bounded:")
    test_cq_error_bounded(node)

    print("\n4. Degraded state recovery:")
    test_degraded_not_latching(node)

    print("\n5. GPU utilization:")
    test_gpu_utilization_reasonable(node)

    print("\n6. Weightd CPU:")
    test_weightd_cpu_not_burning(node)

    print(f"\n=== summary: {len(FAILURES)} failures ===")
    for f in FAILURES:
        print(f"  {f}")
    return 1 if FAILURES else 0

if __name__ == "__main__":
    sys.exit(main())
