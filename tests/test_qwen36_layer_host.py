#!/usr/bin/env python3
"""Run a whole Qwen 3.6 layer on a CPU and check where its data went.

tests/host_cuda/qwen36_layer_host.cu includes inference/llms/qwen_3_6/layer.cuh
UNMODIFIED, with the GEMM recorded through the include shim and every other
kernel the one that ships. The per-kernel harnesses cannot see this driver's
defect class - a cache stored from a buffer nothing wrote, a state slot half
the kernel's stride, a convolution over one of three tensors - because each
kernel did exactly what it was told. This checks the wiring:

  attention   the cache slot holds this step's K and V, in the [K|V] layout,
              over a poisoned pool, and attending over them returns the value
  recurrent   both GEMM widths, all 48 value-head state slices advanced to
              the reference value, the window committed on all 10240 channels,
              and the canaries past both pools intact
  expansion   head h of the expanded row is source head h / 3, with values
              that discriminate (uniform projections cannot see a bad mapping)
  mlp         the two GEMM widths and a nonzero gated activation
"""
import math
import subprocess
import sys
from pathlib import Path

from host_cuda_compiler import host_cuda_cxx

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tests" / "host_cuda" / "qwen36_layer_host.cu"
BINARY = Path("/tmp") / "lm_qwen36_layer_host"


def build():
    result = subprocess.run(
        [host_cuda_cxx(), "-std=c++17", "-O0",
         f"-I{ROOT}/tests/host_cuda/shim", f"-I{ROOT}",
         f"-I{ROOT}/tests/host_cuda",
         "-x", "c++", str(SOURCE), "-o", str(BINARY)],
        capture_output=True, text=True)
    if result.returncode != 0:
        errors = [l for l in result.stderr.split("\n") if "error" in l]
        print("FAIL host build:", (errors or [result.stderr])[0][:200])
        return False
    return True


def parse(text):
    values = {}
    gemms = {"attn_gemm": [], "gdn_gemm": [], "mlp_gemm": []}
    for line in text.strip().split("\n"):
        parts = line.split()
        if not parts:
            continue
        if parts[0] in gemms and parts[1] == "in":
            gemms[parts[0]].append((int(parts[2]), int(parts[4])))
        elif parts[0].startswith("state_h"):
            values.setdefault(parts[0], []).append(float(parts[1]))
        elif len(parts) == 2:
            try:
                values[parts[0]] = float(parts[1])
            except ValueError:
                values[parts[0]] = parts[1]
    return values, gemms


def main():
    if not build():
        return 1
    run = subprocess.run([str(BINARY)], capture_output=True, text=True)
    if run.returncode != 0:
        print("FAIL host run:", (run.stdout + run.stderr).strip()[-300:])
        return 1
    values, gemms = parse(run.stdout)
    failures = 0

    def check(condition, label):
        nonlocal failures
        if not condition:
            print(f"  FAIL {label}")
            failures += 1

    check(gemms["attn_gemm"] == [(5120, 8192), (6144, 5120)],
          f"attention projection widths: {gemms['attn_gemm']}")
    check(values.get("slot_kv_maxdiff") == 0.0,
          f"cache slot does not hold this step's K and V "
          f"(maxdiff {values.get('slot_kv_maxdiff')})")
    check(values.get("attn_maxdiff") == 0.0,
          f"attention over the stored slot is not the stored value "
          f"(maxdiff {values.get('attn_maxdiff')})")

    check(gemms["gdn_gemm"] == [(5120, 10240), (6144, 5120)],
          f"recurrent projection widths: {gemms['gdn_gemm']}")
    # One decode step from a zero state: S = beta * v k^T with unit q and k,
    # so every element of every value-head slice is beta * v * k_elem. The
    # harness prints the convolved value c0; the norm is the kernel's own.
    c0 = values.get("conv_c0", 0.0)
    k_elem = c0 / math.sqrt(128.0 * c0 * c0 + 1e-6)
    expected = 0.25 * c0 * k_elem
    for head in ("state_h0", "state_h47"):
        got = values.get(head, [])
        check(len(got) == 8 and all(abs(g - expected) < 1e-6 for g in got),
              f"{head}: expected {expected:.6g}, got {got[:2] or 'nothing'}")
    check(values.get("window_tap3_min") == 0.125
          and values.get("window_tap3_max") == 0.125,
          f"convolution window not committed on all 10240 channels "
          f"({values.get('window_tap3_min')}..{values.get('window_tap3_max')})")
    check(values.get("canary_state") == 1.0,
          "writes past the GDN state pool (slot stride wider than the slot)")
    check(values.get("canary_window") == 1.0,
          "writes past the convolution window pool")
    check(values.get("expand_mismatch") == 0.0,
          f"head expansion maps the wrong source head "
          f"({values.get('expand_mismatch')} mismatches)")

    check(gemms["mlp_gemm"] == [(5120, 34816), (17408, 5120)],
          f"mlp projection widths: {gemms['mlp_gemm']}")
    check(values.get("mlp_intermediate_max", 0.0) > 0.0,
          "the gated activation is all zero")

    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("a whole Qwen 3.6 layer, run on a CPU: attention, recurrence and MLP")
    print("all move data where the reference says it goes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
