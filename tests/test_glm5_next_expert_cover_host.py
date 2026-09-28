#!/usr/bin/env python3
"""Run the working-set detection kernels and the step verdict on the host shim.

The real cover kernel, poison kernel, head maxloc pack and the module's own
head maxloc unpack run on the CUDA CPU shim: covered routes stay untouched,
misses substitute a held expert and fill a ring that never wraps, an empty
layer traps, no float orders to the poison value, one poisoned rank reaches
every row through a 16-rank MAX fold, and the verdict table classifies it.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from host_cuda_compiler import host_cuda_cxx

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/host_cuda/glm5_next_expert_cover_host.cu"
CUDA = ROOT / "modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_cuda.cu"


def function(source, name):
    match = re.search(r"(?m)^[^\n;]*\b" + re.escape(name) + r"\s*\(", source)
    if match is None:
        raise SystemExit(f"FAIL {name} not found in {CUDA}")
    opening = source.index("{", match.end())
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]


def main():
    unpack = function(CUDA.read_text(), "SparkGlm5NextHeadMaxlocUnpackKernel")
    with tempfile.TemporaryDirectory(prefix="glm5-next-expert-cover-") as directory:
        Path(directory, "glm5_next_head_unpack.inc").write_text(unpack + "\n")
        binary = str(Path(directory) / "probe")
        build = subprocess.run([host_cuda_cxx(), "-std=c++17", "-O2", "-Itests/host_cuda/shim", f"-I{directory}",
                                "-I.", "-Itests/host_cuda", "-Iinclude", "-x", "c++", str(SOURCE), "-o", binary],
                               cwd=ROOT, capture_output=True, text=True)
        if build.returncode != 0:
            print("FAIL host build:", build.stderr.strip()[:2000])
            return 1
        run = subprocess.run([binary], capture_output=True, text=True, timeout=600)
    print(run.stdout, end="")
    if run.returncode != 0 or "PASS (0 failures)" not in run.stdout:
        print(run.stderr, end="")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
