#!/usr/bin/env python3
"""Run a whole K3 MoE layer on a CPU and check where its data went.

Every per-kernel harness passes because every kernel is individually correct.
An external audit found six P0s in K3 and its closing observation was the
useful one: they live in the paths those harnesses do not execute. Three were
dataflow, and no per-kernel test can see dataflow.

This runs the real K3LayerLatentMoe with only the GEMM replaced by a recorder.
On its first successful run it found a divide-by-zero in production code - the
quantise helper divided a width by LmBf16Format's kScaleGroup, which is
correctly zero because BF16 has no groups, at all 20 non-expert projections.

The checks below are the audit's questions, asked of a running layer:

  the shared expert must not write the buffer the routed branch wrote
  the routed result must survive into the output, which means the add ran
  the expert GEMMs must be grouped and see rows * top_k rows
  the router must read the full hidden, not the latent
"""
import re
import subprocess
import sys
from pathlib import Path

from host_cuda_compiler import host_cuda_cxx

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tests" / "host_cuda" / "k3_layer_host.cu"
BINARY = Path("/tmp") / "lm_k3_layer_host"


def main():
    build = subprocess.run(
        [host_cuda_cxx(), "-std=c++17", "-O0", f"-I{ROOT}/tests/host_cuda/shim", f"-I{ROOT}",
         f"-I{ROOT}/tests/host_cuda", f"-I{ROOT}/model-families/common/include", f"-I{ROOT}/include",
         "-x", "c++", str(SOURCE), "-o", str(BINARY)],
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
                         r"grouped (\d) ind (\d)", line)
        if match:
            gemms.append(dict(dest=match.group(1), inp=int(match.group(2)),
                              out=int(match.group(3)), rows=int(match.group(4)),
                              grouped=match.group(5) == "1",
                              indirect=match.group(6) == "1"))
    values = dict(re.findall(r"(\w+)\[0\] ([\d.\-]+)", run.stdout))
    failures = 0

    destinations = [g["dest"] for g in gemms]
    if "shared_out" not in destinations:
        print("  FAIL the shared expert does not write its own buffer")
        failures += 1
    if destinations.count("hidden") != 1:
        print(f"  FAIL {destinations.count('hidden')} GEMMs write hidden; "
              f"the routed branch and the shared branch must not share it")
        failures += 1
    # every GEMM writes 0.125 * its index, so the output tells you what
    # happened. THE SUM NO LONGER LIVES IN HIDDEN: both module-output halves
    # fold into the AttnRes partial in their own epilogues (the layer host
    # binds no partial, so the accumulate is a no-op here and the slice gate
    # owns the summed trajectory). Hidden must hold the routed result ALONE -
    # a hidden that equals routed + shared would mean the deleted AddRows
    # came back.
    hidden = float(values.get("hidden", 0.0))
    shared = float(values.get("shared_out", 0.0))
    routed_index = destinations.index("hidden") + 1
    if abs(hidden - 0.125 * routed_index) > 1e-3:
        print(f"  FAIL hidden is {hidden}, not the routed result "
              f"({0.125*routed_index}) alone; the epilogue fusion regressed")
        failures += 1
    if shared == 0.0:
        print("  FAIL the shared branch produced nothing")
        failures += 1
    grouped = [g for g in gemms if g["grouped"]]
    if len(grouped) != 2:
        print(f"  FAIL {len(grouped)} grouped GEMMs, expected the two expert ones")
        failures += 1
    for g in grouped:
        if g["rows"] <= 2:
            print(f"  FAIL an expert GEMM sees {g['rows']} rows; "
                  f"routes should be rows * top_k")
            failures += 1
    # THE GATHER DELETION, OBSERVED: the w1 expert GEMM (the wide one, gate|up
    # out) must arrive INDIRECT - A rows read through route_source_token, no
    # packed activation copy - and the w2 must not be (its input is the SiTU
    # output, already expert-major).
    wide = [g for g in grouped if g["out"] == max(x["out"] for x in grouped)]
    down = [g for g in grouped if g["out"] != max(x["out"] for x in grouped)]
    if len(wide) != 1 or not wide[0]["indirect"]:
        print("  FAIL the w1 expert GEMM did not come in indirect; the route "
              "gather double-touch is back (roadmap D9)")
        failures += 1
    if len(down) != 1 or down[0]["indirect"]:
        print("  FAIL the w2 expert GEMM is indirect; its A rows are the "
              "packed SiTU output, which has no route-map meaning")
        failures += 1
    router = gemms[0]
    if router["inp"] != 7168:
        print(f"  FAIL the router reads {router['inp']}, not the full hidden")
        failures += 1

    print(f"gemms {len(gemms)}, grouped {len(grouped)}, "
          f"hidden {hidden} = routed + shared {shared}")
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nthe routed branch survives, the shared expert adds, "
          "the experts see every route")
    return 0


if __name__ == "__main__":
    sys.exit(main())
