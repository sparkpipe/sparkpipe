#!/usr/bin/env python3
"""GB10 check of the DSV4.1 rows-owned MoE exchange (dsv41_flash_stage_kernels.cuh).

Builds build/test_dsv41_exchange and runs it on an sm_121 device, sixteen ranks simulated on one
GPU. Each rank numbers its routed rows for every rank that holds one of their six experts, in row
order; the counts it reports must match. A packed row must arrive at each such rank exactly once,
with its MXFP8 codes decoded to the same BF16 values, its expert ids and its weights, and padding
rows up to the shared capacity must carry no routes. The owner must sum the ranks' partials in
rank order in FP32, round once to BF16 as the reduce-scatter it replaces does, add the shared
expert's BF16 output as the publisher's MoE adds its shared expert, and write zero for rows past
its live count. Cases cover uneven routing with two hot ranks, a partial last rank and no rows.
"""

import subprocess
import sys
from pathlib import Path

import gpu_test_runner

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_dsv41_exchange"


def main():
    if gpu_test_runner.built_binary("test_dsv41_exchange", BINARY) is None:
        return 0
    result = subprocess.run([str(BINARY)], capture_output=True, text=True)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS dsv41 exchange" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
