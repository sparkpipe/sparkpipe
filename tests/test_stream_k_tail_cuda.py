#!/usr/bin/env python3
"""The stream GEMM's K tail: an unscaled (bf16) input that is a multiple of 8 but not of the 64-wide k-tile runs with its last tile
zero-filled, so its bf16 and fp32 outputs are bit-identical to the same wave zero-padded to the next multiple of 64, for dense and
grouped waves at every tile height, both bf16 formats and K3's routed_up wave (2048 x 224 -> 7168), and agree with an fp64 reference;
each row gets the same bits in a wave of any size from 17 to 2048 rows wherever the wave starts, so prefill waves split differently, as
prefix reuse splits them, agree with a cold run;
an input that is not a multiple of 8 refuses through SPARK_FAIL."""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_test_runner import NVCC, ROOT, has_device  # noqa: E402

NAME = "test_stream_k_tail_cuda"


def main():
    if not has_device(NAME):
        return 0
    if shutil.which(NVCC) is None:
        raise SystemExit(f"FAIL {NAME}: an sm_121 device is present but nvcc is unavailable")
    with tempfile.TemporaryDirectory(prefix=NAME + "-") as directory:
        binary = str(Path(directory) / "probe")
        subprocess.run([NVCC, "-O3", "-std=c++17", "-gencode", "arch=compute_121a,code=sm_121a", "-I.", "-Iinclude",
                        "-Imodel-families/common/include", "tests/cuda/stream_k_tail_cuda.cu", "-lcuda", "-o", binary], cwd=ROOT, check=True)
        subprocess.run([binary], check=True, timeout=600)
    return 0


if __name__ == "__main__":
    sys.exit(main())
