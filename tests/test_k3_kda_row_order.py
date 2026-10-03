#!/usr/bin/env python3
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from host_cuda_compiler import host_cuda_cxx

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tests" / "host_cuda" / "k3_kda_row_order_host.cu"


def main():
    with tempfile.TemporaryDirectory() as scratch:
        binary = Path(scratch) / "lm_k3_kda_row_order_host"
        build = subprocess.run(
            [host_cuda_cxx(), "-std=c++17", "-O1",
             f"-I{ROOT}/tests/host_cuda/shim", f"-I{ROOT}",
             f"-I{ROOT}/tests/host_cuda",
             f"-I{ROOT}/model-families/common/include", f"-I{ROOT}/include",
             "-x", "c++", str(SOURCE), "-o", str(binary)],
            capture_output=True, text=True)
        if build.returncode != 0:
            print("FAIL host build: " + build.stderr[-400:])
            return 1
        run = subprocess.run([str(binary)], capture_output=True, text=True)
    out = run.stdout
    failures = []
    if run.returncode != 0 or "done" not in out:
        failures.append(f"harness exit {run.returncode}: {out[-400:]}")
    nonzero = re.search(r"^grouped_state_nonzero (\d+)$", out, re.M)
    indexed = re.search(r"^interleaved_indexed status (-?\d+) mismatch (\d+)$", out, re.M)
    unindexed = re.search(r"^interleaved_unindexed status (-?\d+) mismatch (\d+)$", out, re.M)
    if nonzero is None or int(nonzero.group(1)) == 0:
        failures.append("the grouped run leaves no recurrent state, so the comparison proves nothing")
    if indexed is None or indexed.group(1) != "0" or indexed.group(2) != "0":
        failures.append("rows interleaved across sequences, walked through sequence_row_indices, must give "
                        "the grouped run's outputs, recurrent state and conv windows bit for bit: " + str(indexed and indexed.groups()))
    if unindexed is None or unindexed.group(2) == "0":
        failures.append("the same interleaved rows without sequence_row_indices must differ, or the check is blind")
    for failure in failures:
        print("FAIL " + failure)
    if failures:
        return 1
    print("PASS K3 KDA walks each sequence's own rows when a prefill interleaves sequences")
    return 0


if __name__ == "__main__":
    sys.exit(main())
