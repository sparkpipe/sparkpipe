#!/usr/bin/env python3
"""GB10 check of the DSV4.1 rows-owned MoE exchange (dsv41_flash_stage_kernels.cuh).

Builds build/test_dsv41_exchange and runs it on an sm_121 device, sixteen ranks simulated on one
GPU. Each owner first moves its owned rows' MXFP8 codes, scales, experts and weights to the MoE
owner named by the row's index modulo the ranks, so that every MoE owner holds rows spread over the
whole step; each owner's counts (rows by MoE owner and expert rank, routes by expert, and rows whose
only route into an expert's rank is that expert, by MoE owner and expert) must match the routes,
and their sums over the owners must add up. A row's group on an expert rank (its routes into that
rank) is one record. Each MoE owner may send up to a planned number of rows whose only route into a
rank is one expert, the first such rows in row order, to the rank that holds that expert's first
backup copy and the next ones to the rank holding its second copy instead; the plan is the one the
stage makes from the summed counts, or random numbers of rows for each copy up to every such row. Each MoE owner numbers its records for every rank in row order and rank of
the group; every group must reach its rank or the backup rank exactly once, as the planned counts
say, with its MXFP8 codes and scales unchanged (the expert rank multiplies them on the block-scaled
kernel), its expert ids and its weights; a backup record carries only the flagged expert, and a
rank localizes home experts to its own slots and flagged experts to its backup slots; a rank
unpacks only real records, packed source by source with no padding between them. Its finalize must
sum each record's local expert outputs in route order in FP32, round once to BF16 and write the
result to the record's place in the padded return layout. The MoE owner must sum the groups'
partials in rank order in FP32 and round once to BF16 as the reduce-scatter it replaces does, and
the row's owner must add the shared expert's BF16 output as the publisher's MoE adds its shared
expert and write zero for rows past its live count. Expert outputs depend only on the token and the
expert, so the result must equal the same sums computed without backups, bit for bit, wherever a
group was computed. Cases cover uneven routing with two hot ranks, a partial last rank and no rows.
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
