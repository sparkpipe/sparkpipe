#!/usr/bin/env python3
"""GB10 check of the DSV4.1 rows-owned MoE exchange (dsv41_flash_stage_kernels.cuh).

Builds build/test_dsv41_exchange and runs it on an sm_121 device, sixteen ranks simulated on one
GPU. Each owner first moves its owned rows' MXFP8 codes, scales, experts and weights to the MoE owner
named by the row's index modulo the ranks; each owner's per-expert route counts must add up to what
the MoE owners route. Every expert has a second copy on another rank; the test picks home or second
copy at random for each MoE owner and expert, as the per-step balance may. Each route must reach
the chosen rank exactly once, numbered in route order, with its codes decoded to the same BF16
values, its expert and weight, mapped to the rank's own or second-copy group. The route's output
must return to its place in the padded return. The MoE owner must sum each home rank's routes in
route order in FP32 and round once to BF16, sum those partials in rank order and round once, as the
reduce-scatter it replaces does, and the row's owner must add the shared expert's BF16 output as the
publisher's MoE adds its shared expert and write zero for rows past its live count. The result must
not depend on which copy ran. Cases cover uneven routing with two hot ranks, a partial last rank and no rows.
The engram exchange numbers every looked-up engram row by the rank that holds it and the rank
that owns its token row; each holder packs its rows' FP8 codes and scales, and each owner must
decode them to the BF16 values the holder's own lookup gives, with zero for ids no rank holds
and for padding rows. Cases cover a full 16,384-row step, odd row counts and a table that does
not split evenly over the ranks.
Window keys gather only each rank's last rows: a rank's own rows, the 127 rows before them and
the step's last 128 rows must assemble to the values the full gather of every row gives, for full
steps, steps with padding rows and the smallest owned blocks.
The tail is split over the ranks again: every tail row's four streams and stream weights must reach
the rank that owns it in the tail's own split, from the rank that owned it in the step, for full and
odd steps, a short step whose tail starts near its first row, and a tail shorter than a full reach.
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
