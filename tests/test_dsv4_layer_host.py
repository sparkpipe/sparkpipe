#!/usr/bin/env python3
"""Run a whole DSv4 layer on a CPU and check where its data went.

The DSv4 instance of tests/test_k3_layer_host.py's argument: per-kernel
harnesses cannot see dataflow, and this driver's defects are dataflow. The
questions are the 2026-08-01 audit's:

  the KV latent GEMM must read the low-rank path's quantised input - the
    second quantise of the normed rows was deleted, so the recorded
    activation pointer IS the query scratch or the dedup regressed;
  the expert GEMMs must be grouped and see rows * top_k rows;
  the router must read the full hidden and write f32 logits;
  hidden must be the routed result plus the shared expert's, checked
    against the route weights the run actually emitted.
"""
import re
import subprocess
import sys
from pathlib import Path

from host_cuda_compiler import host_cuda_cxx

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tests" / "host_cuda" / "dsv4_layer_host.cu"
BINARY = Path("/tmp") / "lm_dsv4_layer_host"


def main():
    build = subprocess.run(
        [host_cuda_cxx(), "-std=c++17", "-O0", f"-I{ROOT}/tests/host_cuda/shim", f"-I{ROOT}",
         f"-I{ROOT}/tests/host_cuda", "-x", "c++", str(SOURCE), "-o", str(BINARY)],
        capture_output=True, text=True)
    if build.returncode != 0:
        errors = [l for l in build.stderr.split("\n") if "error" in l]
        print("FAIL host build:", (errors or [build.stderr])[0][:200])
        return 1
    run = subprocess.run([str(BINARY)], capture_output=True, text=True)
    if run.returncode != 0:
        print(f"FAIL the layer faulted (signal {-run.returncode})")
        return 1

    gemms = []
    for line in run.stdout.split("\n"):
        match = re.match(r"gemm \d+ dest (\w+) in (\d+) out (\d+) rows (\d+) "
                         r"grouped (\d)", line)
        if match:
            gemms.append(dict(dest=match.group(1), inp=int(match.group(2)),
                              out=int(match.group(3)), rows=int(match.group(4)),
                              grouped=match.group(5) == "1"))
    values = dict(re.findall(r"^([A-Za-z0-9_\[\] ]*?[A-Za-z0-9_\[\]]) "
                             r"([-\d.e]+)$", run.stdout, re.M))
    failures = 0

    if "attention 0" not in run.stdout or "moe 0" not in run.stdout:
        print("  FAIL a layer half returned a launch error")
        failures += 1
    if len(gemms) != 9:
        print(f"  FAIL {len(gemms)} GEMMs recorded, expected 9 "
              "(4 attention + router + 2 shared + 2 expert)")
        failures += 1
    if values.get("kv activation is query scratch") != "1":
        print("  FAIL the KV latent GEMM does not read the low-rank "
              "quantised input; the dedup regressed")
        failures += 1
    grouped = [g for g in gemms if g["grouped"]]
    if len(grouped) != 2 or any(g["rows"] != 12 for g in grouped):
        print(f"  FAIL expert GEMMs {[(g['rows'], g['grouped']) for g in gemms]}; "
              "expected two grouped at rows * top_k = 12")
        failures += 1
    if gemms and gemms[4]["inp"] != 4096:
        print(f"  FAIL the router reads {gemms[4]['inp']}, not the full hidden")
        failures += 1
    hidden = float(values.get("hidden[0]", 0.0))
    expected = float(values.get("hidden expected", -1.0))
    weight_sum = float(values.get("route weight sum", 0.0))
    if weight_sum <= 0.0:
        print("  FAIL the router emitted no weight; the routed branch is dead")
        failures += 1
    if abs(hidden - expected) > 1e-3:
        print(f"  FAIL hidden is {hidden}, expected {expected} "
              "(shared + expert x route weights); the finalize or the "
              "shared add regressed")
        failures += 1

    print(f"gemms {len(gemms)}, grouped {len(grouped)}, hidden {hidden}, "
          f"expected {expected}")
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nthe KV GEMM rides the low-rank quantise, the experts see every "
          "route, the routed branch and the shared expert both land")
    return 0


if __name__ == "__main__":
    sys.exit(main())
