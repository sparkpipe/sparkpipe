#!/usr/bin/env python3
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from host_cuda_compiler import host_cuda_cxx

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tests" / "host_cuda" / "k3_kda_rank_heads_host.cu"
TP_DEGREE = 4
KDA_HEADS = 96


def run_harness():
    with tempfile.TemporaryDirectory() as scratch:
        binary = Path(scratch) / "lm_k3_kda_rank_heads_host"
        build = subprocess.run(
            [host_cuda_cxx(), "-std=c++17", "-O1",
             f"-I{ROOT}/tests/host_cuda/shim", f"-I{ROOT}",
             f"-I{ROOT}/tests/host_cuda",
             f"-I{ROOT}/model-families/common/include", f"-I{ROOT}/include",
             "-x", "c++", str(SOURCE), "-o", str(binary)],
            capture_output=True, text=True)
        if build.returncode != 0:
            errors = [l for l in build.stderr.split("\n") if "error" in l]
            return None, "host build: " + (errors or [build.stderr])[0][:240]
        run = subprocess.run([str(binary)], capture_output=True, text=True)
    if run.returncode != 0 or "done" not in run.stdout:
        return None, f"harness exit {run.returncode}: {run.stdout[-400:]}"
    return run.stdout, None


def check(stdout):
    failures = []
    distinct = re.search(r"^distinct_heads (\d+)$", stdout, re.M)
    if distinct is None or int(distinct.group(1)) != KDA_HEADS - 1:
        failures.append("the per-head decay tables must give every head its "
                        "own retention, or the rank check proves nothing")
    ranks = {int(m.group(1)): tuple(int(g) for g in m.groups()[1:])
             for m in re.finditer(r"^rank (\d+) retention_mismatch (\d+) "
                                  r"state_mismatch (\d+) out_mismatch (\d+)$",
                                  stdout, re.M)}
    if sorted(ranks) != list(range(TP_DEGREE)):
        failures.append(f"expected {TP_DEGREE} rank lines, got {sorted(ranks)}")
    for rank, (retention, state, out) in sorted(ranks.items()):
        if retention or state or out:
            failures.append(
                f"TP{TP_DEGREE} rank {rank} disagrees with the full-width layer "
                f"on its own heads: retention {retention}, state {state}, "
                f"output {out} mismatches")
    past = re.search(r"^rank_past_heads_status (-?\d+)$", stdout, re.M)
    if past is None or int(past.group(1)) == 0:
        failures.append("a rank whose head range runs past the model's heads "
                        "must be refused")
    return failures


def main():
    stdout, error = run_harness()
    if error is not None:
        print("FAIL", error)
        return 1
    failures = check(stdout)
    for failure in failures:
        print("FAIL", failure)
    if failures:
        return 1
    print(f"PASS K3 KDA TP{TP_DEGREE} ranks read their own heads' decay bias "
          f"and log scale over 3 recurrent steps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
