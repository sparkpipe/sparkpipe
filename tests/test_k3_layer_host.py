#!/usr/bin/env python3
"""Run a whole K3 MoE layer on a CPU and check where its data went.

Every per-kernel harness passes because every kernel is individually correct.
An external audit found six P0s in K3 and its closing observation was the
useful one: they live in the paths those harnesses do not execute. Three were
dataflow, and no per-kernel test can see dataflow.

This runs the real K3LayerLatentMoe with only the GEMM replaced by a recorder,
three times: with the non-interleaved expert stream, and with the pack V2
interleaved stream (the production layout) at tile_k 128 and at tile_k 32
(the TP16 repack granularity). Each run is asked the audit's questions:

  the shared expert must not write the buffer the routed branch wrote
  the routed result must survive into hidden alone, the shared half folds
  into the AttnRes partial in its own epilogue
  the expert GEMMs must be grouped and see rows * top_k routes, w1 indirect
  through the route map and w2 packed
  the router must read the full hidden, not the latent
  the interleave flag must reach the interleaved GEMMs at the pack's tile_k,
  and the flag off must reach the plain ones
  only the two routed-expert GEMMs take the 4-bit weight format, and every
  GEMM reads BF16 activations: the checkpoint quantises weights only
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from host_cuda_compiler import host_cuda_cxx

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tests" / "host_cuda" / "k3_layer_host.cu"
CONFIGS = [(0, 128), (1, 128), (1, 32)]


def parse(block):
    gemms = []
    for match in re.finditer(
            r"gemm \d+ dest (\w+) in (\d+) out (\d+) rows (\d+) grouped (\d) "
            r"ind (\d) il (\d) tk (\d+) abits (\d+) wbits (\d+)", block):
        gemms.append(dict(dest=match.group(1), inp=int(match.group(2)),
                          out=int(match.group(3)), rows=int(match.group(4)),
                          grouped=match.group(5) == "1",
                          indirect=match.group(6) == "1",
                          interleaved=match.group(7) == "1",
                          tile_k=int(match.group(8)),
                          activation_bits=int(match.group(9)),
                          weight_bits=int(match.group(10))))
    values = dict(re.findall(r"(\w+)\[0\] ([\d.\-]+)", block))
    return gemms, values


def check_config(label, interleave, tile_k, gemms, values):
    failures = []
    if not gemms:
        return [f"{label}: the layer recorded no GEMMs"]
    destinations = [g["dest"] for g in gemms]
    if "shared_out" not in destinations:
        failures.append(f"{label}: the shared expert does not write its own buffer")
    if destinations.count("hidden") != 1:
        failures.append(f"{label}: {destinations.count('hidden')} GEMMs write "
                        f"hidden; the routed and shared branches must not share it")
        return failures
    hidden = float(values.get("hidden", 0.0))
    shared = float(values.get("shared_out", 0.0))
    routed_index = destinations.index("hidden") + 1
    if abs(hidden - 0.125 * routed_index) > 1e-3:
        failures.append(f"{label}: hidden is {hidden}, not the routed result "
                        f"({0.125 * routed_index}) alone")
    if shared == 0.0:
        failures.append(f"{label}: the shared branch produced nothing")
    grouped = [g for g in gemms if g["grouped"]]
    if len(grouped) != 2:
        failures.append(f"{label}: {len(grouped)} grouped GEMMs, expected the "
                        f"two expert ones")
        return failures
    for g in grouped:
        if g["rows"] <= 2:
            failures.append(f"{label}: an expert GEMM sees {g['rows']} rows; "
                            f"routes should be rows * top_k")
    widest = max(g["out"] for g in grouped)
    wide = [g for g in grouped if g["out"] == widest]
    down = [g for g in grouped if g["out"] != widest]
    if len(wide) != 1 or not wide[0]["indirect"]:
        failures.append(f"{label}: the w1 expert GEMM did not come in indirect")
    if len(down) != 1 or down[0]["indirect"]:
        failures.append(f"{label}: the w2 expert GEMM is indirect; its A rows "
                        f"are the packed SiTU output")
    for g in grouped:
        if g["interleaved"] != (interleave == 1):
            failures.append(f"{label}: an expert GEMM into {g['dest']} ran "
                            f"{'interleaved' if g['interleaved'] else 'plain'}")
        if interleave == 1 and g["tile_k"] != tile_k:
            failures.append(f"{label}: an interleaved expert GEMM ran at tile_k "
                            f"{g['tile_k']}, not the pack's {tile_k}")
        if g["weight_bits"] != 4:
            failures.append(f"{label}: an expert GEMM reads "
                            f"{g['weight_bits']}-bit weights, not the MXFP4 grid")
    for g in gemms:
        if not g["grouped"] and (g["interleaved"] or g["weight_bits"] != 16):
            failures.append(f"{label}: a projection into {g['dest']} reads "
                            f"{g['weight_bits']}-bit weights; the checkpoint's "
                            f"ignore list keeps it BF16")
        if g["activation_bits"] != 16:
            failures.append(f"{label}: a GEMM into {g['dest']} quantises its "
                            f"activations to {g['activation_bits']} bits")
    if gemms[0]["inp"] != 7168:
        failures.append(f"{label}: the router reads {gemms[0]['inp']}, not the "
                        f"full hidden")
    print(f"{label}: gemms {len(gemms)}, grouped {len(grouped)}, "
          f"hidden {hidden} is the routed result alone, shared {shared}")
    return failures


def main():
    with tempfile.TemporaryDirectory() as scratch:
        binary = Path(scratch) / "lm_k3_layer_host"
        build = subprocess.run(
            [host_cuda_cxx(), "-std=c++17", "-O0", f"-I{ROOT}/tests/host_cuda/shim",
             f"-I{ROOT}", f"-I{ROOT}/tests/host_cuda",
             f"-I{ROOT}/model-families/common/include", f"-I{ROOT}/include",
             "-x", "c++", str(SOURCE), "-o", str(binary)],
            capture_output=True, text=True)
        if build.returncode != 0:
            errors = [l for l in build.stderr.split("\n") if "error" in l]
            print("FAIL host build:", (errors or [build.stderr])[0][:200])
            return 1
        run = subprocess.run([str(binary)], capture_output=True, text=True)
    if run.returncode != 0 or "survived" not in run.stdout:
        print(f"FAIL the layer faulted (returncode {run.returncode})")
        print(run.stdout[-400:])
        return 1

    parts = re.split(r"^config interleave (\d+) tile_k (\d+)$", run.stdout,
                     flags=re.M)
    blocks = {(int(parts[i]), int(parts[i + 1])): parts[i + 2]
              for i in range(1, len(parts) - 2, 3)}
    failures = []
    for interleave, tile_k in CONFIGS:
        label = f"interleave {interleave} tile_k {tile_k}"
        block = blocks.get((interleave, tile_k))
        if block is None:
            failures.append(f"{label}: the harness did not run it")
            continue
        if "launch refused" in block:
            failures.append(f"{label}: the layer refused to launch")
            continue
        gemms, values = parse(block)
        failures += check_config(label, interleave, tile_k, gemms, values)

    for failure in failures:
        print(f"  FAIL {failure}")
    if failures:
        print(f"\nFAIL ({len(failures)})")
        return 1
    print("\nthe routed branch survives, the shared expert adds, the experts "
          "see every route, and the interleave flag reaches the interleaved "
          "GEMMs at the pack's tile_k")
    return 0


if __name__ == "__main__":
    sys.exit(main())
