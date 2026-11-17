#!/usr/bin/env python3
"""GB10 bitwise check of the DSV4.1 shared-expert split for small steps.

Builds build/test_dsv41_shared_split and runs it on an sm_121 device. A decode or verification step
on the sixteen-rank lane splits the shared expert's 2,304 neurons into granules of 128 over the
ranks, so every rank computes a slice instead of one rank computing it whole. The granules of every
split cover the expert once, in order. Each rank's partial (its local routes plus its shared slice)
stays in FP32, the partials are summed in rank order and the sum is rounded to BF16 once, matching
the host's sums bit for bit, and a one-row step gives the same bits as its row of a nine-row step,
which is what keeps speculative verification exact.
"""

import subprocess
import sys
from pathlib import Path

import gpu_test_runner

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_dsv41_shared_split"


def main():
    if gpu_test_runner.built_binary("test_dsv41_shared_split", BINARY) is None:
        return 0
    result = subprocess.run([str(BINARY)], capture_output=True, text=True)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS dsv41 shared split" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
