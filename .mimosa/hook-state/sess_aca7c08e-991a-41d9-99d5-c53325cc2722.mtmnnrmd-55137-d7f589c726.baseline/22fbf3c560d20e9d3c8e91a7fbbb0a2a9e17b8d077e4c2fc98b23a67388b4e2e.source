#!/usr/bin/env python3
"""Run MiMo 2.5's attention layers on a CPU and check where their data went.

tests/host_cuda/mimo25_layer_host.cu includes inference/llms/mimo_2_5/layer.cuh
UNMODIFIED, with the GEMM recorded through the include shim and every other
kernel the one that ships. What it checks, because each was a live defect or
is this driver's distinctive shape:

  full    the fused projection's widths, the slot holding this step's K and
          the 0.707-scaled V in the [K|V] layout over a poisoned pool, and
          attention returning exactly the scaled value
  swa     the wider SWA projection, the slot at the windowed position, and
          the selected-position list attending over only that position
  mlp     the dense SwiGLU widths and a nonzero gated activation
"""
import subprocess
import sys
from pathlib import Path

from host_cuda_compiler import host_cuda_cxx

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tests" / "host_cuda" / "mimo25_layer_host.cu"
BINARY = Path("/tmp") / "lm_mimo25_layer_host"


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
    gemms = {"full_gemm": [], "swa_gemm": [], "mlp_gemm": []}
    for line in text.strip().split("\n"):
        parts = line.split()
        if not parts:
            continue
        if parts[0] in gemms and parts[1] == "in":
            gemms[parts[0]].append((int(parts[2]), int(parts[4])))
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

    check(gemms["full_gemm"] == [(4096, 13568), (8192, 4096)],
          f"full-attention projection widths: {gemms['full_gemm']}")
    check(values.get("full_slot_maxdiff") == 0.0,
          f"full slot does not hold this step's K and scaled V "
          f"(maxdiff {values.get('full_slot_maxdiff')})")
    check(values.get("full_attn_maxdiff") == 0.0,
          f"full attention is not the stored value "
          f"(maxdiff {values.get('full_attn_maxdiff')})")

    check(gemms["swa_gemm"] == [(4096, 14848), (8192, 4096)],
          f"swa projection widths: {gemms['swa_gemm']}")
    check(values.get("swa_slot_maxdiff") == 0.0,
          f"swa slot's value region is wrong "
          f"(maxdiff {values.get('swa_slot_maxdiff')})")
    check(values.get("swa_slot_poisoned") == 0.0,
          f"swa slot still holds pool poison in "
          f"{values.get('swa_slot_poisoned')} elements - the store missed it")
    check(values.get("swa_attn_maxdiff") == 0.0,
          f"windowed attention is not the stored value "
          f"(maxdiff {values.get('swa_attn_maxdiff')})")

    check(gemms["mlp_gemm"] == [(4096, 32768), (16384, 4096)],
          f"mlp projection widths: {gemms['mlp_gemm']}")
    check(values.get("mlp_intermediate_max", 0.0) > 0.0,
          "the gated activation is all zero")

    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("MiMo 2.5's full and sliding-window attention layers, run on a CPU,")
    print("move data where the reference says it goes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
