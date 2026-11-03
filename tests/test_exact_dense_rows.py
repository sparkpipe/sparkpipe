"""On an sm_121 device, a skinny dense, split or paired launch of any frame reproduces one-row decode bit for bit.

Builds build/test_exact_dense_rows. At per-rank projection shapes of GLM 5.3 Full and A.X K2 (FP8), Gemma 4, Solar
Open 2, Ling 3.0, Hy4, Qwen 3.8, GLM 5.3 Flash and Kimi K3 (bf16), Qwen 3.8 (FP8) and Hy4 (MXFP8), through every skinny
entry a stage uses (the automatic launch with its split path, the row launch the glm52 stage calls, the forced split
launch K3 calls, and the paired launch with output stride and column offset), every row of a 1, 2, 7, 16, 17, 40, 256
or 512-row launch with any_rows set equals the same row launched alone, padding columns and rows past the frame stay
untouched, and the same launch above 16 rows without any_rows is refused.
"""

import subprocess
import sys
from pathlib import Path

import gpu_test_runner

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_exact_dense_rows"


def main():
    if gpu_test_runner.built_binary("test_exact_dense_rows", BINARY) is None:
        return 0
    result = subprocess.run([str(BINARY)], capture_output=True, text=True, timeout=900)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS exact dense rows" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
