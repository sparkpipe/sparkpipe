"""On an sm_121 device, a grouped skinny expert launch of any frame reproduces the per-route kernel bit for bit.

Builds build/test_exact_grouped_experts. At the per-rank expert shapes of GLM 5.3 Full and A.X K2 (FP8), Gemma 4 26B,
Laguna S 2.1, Solar Open 2 and Ling 3.0 (bf16) and Hy4 (MXFP8), for frames of 1, 2, 7, 16, 17, 40, 256 and 512 rows (two lanes of 256) with
uniform and skewed routing (up to every row on the same few experts, far above the bounded launch's 16 routes per
expert), every local route of the exact grouped launch (any_rows set) equals the per-route kernel one-row decode uses
where every expert is local, and equals the same row launched alone through the grouped kernel. Speculative verify
frames read each touched expert's weights once per frame through this launch and stay exact.
"""

import subprocess
import sys
from pathlib import Path

import gpu_test_runner

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_exact_grouped_experts"


def main():
    if gpu_test_runner.built_binary("test_exact_grouped_experts", BINARY) is None:
        return 0
    result = subprocess.run([str(BINARY)], capture_output=True, text=True, timeout=900)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS exact grouped experts" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
