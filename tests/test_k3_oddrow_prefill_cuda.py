#!/usr/bin/env python3
"""K3 odd-row paired prefill regression gate: verifies K3RunnerPairRows and K3DispatchPairView keep all
per-lane buffers 16-byte aligned across odd prefill waves (1703, 1705, 513, 65, 33) so stream_gemm never
fails with -41, and verifies bit-exact parity between paired odd-row and single-lane KDA execution."""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_test_runner import NVCC, ROOT, has_device  # noqa: E402

NAME = "test_k3_oddrow_prefill_cuda"


def main():
    if not has_device(NAME):
        return 0
    if shutil.which(NVCC) is None:
        raise SystemExit(f"FAIL {NAME}: an sm_121 device is present but nvcc is unavailable")
    subprocess.run(["make", "-s", "build/skl2_k3_embedded.h"], cwd=ROOT, check=True)
    with tempfile.TemporaryDirectory(prefix=NAME + "-") as directory:
        binary = str(Path(directory) / "probe")
        subprocess.run([NVCC, "-O3", "-std=c++17", "-gencode", "arch=compute_121a,code=sm_121a", "-I.", "-Ibuild", "-Iinclude", "-Isrc",
                        "-Imodel-families/common/include", "-Imodel-families/k3/include",
                        "-Imodules/k3_resident_decode_stage/include",
                        "tests/cuda/k3_oddrow_prefill_cuda.cu",
                        "runtime/json.c",
                        "runtime/filesystem.c",
                        "runtime/named_pack.c",
                        "model-families/k3/src/spark_k3_bind.c",
                        "model-families/k3/src/spark_k3_pack_load.c",
                        "model-families/k3/src/spark_k3_resident_decode_stage_module.c",
                        "-lcuda", "-o", binary], cwd=ROOT, check=True)
        subprocess.run([binary], check=True, timeout=300)
    return 0


if __name__ == "__main__":
    sys.exit(main())
