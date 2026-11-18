#!/usr/bin/env python3
"""GB10 bitwise check of the DSV4.1 window fold against the sequential fold it replaced.

Builds build/test_dsv41_window_fold and runs it on an sm_121 device. The fold scores a batch of
window keys before updating the online softmax; every record it writes must be the same bits as
the key-by-key fold, for decode rows, verify frames, prefill waves, local-head landings, early
positions and partial records. Both folds take the publisher's flash form, as DeepSeek's
sparse_attn does: the values are multiplied by each weight exp(s - m) rounded to BF16, while the
running sum adds the FP32 weight. Perturbing one key must change the result. A wave of more than 16
rows on a rank that keeps the whole window attends its window and its selected compressed rows in
one pass (dsv41_flash_attend.cuh) on tensor cores; its weights are rounded against a running
maximum taken a tile at a time, so for the window alone its attention output must match the
sequential fold and sink merge within that rounding (worst element 1/32, mean 0.001 on values up to
4), not bit for bit, for one and two sequences, shuffled rows, early positions and ring keys. Taking
two 16-head groups a pass must give the outputs and MXFP8 codes of one group a launch bit for bit. Record
initialization must set every record to -inf/0 for waves past 262,144 records at TP16 (4100
rows x 4 heads x 16 destinations) and must not touch padding between destinations.
"""

import subprocess
import sys
from pathlib import Path

import gpu_test_runner

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_dsv41_window_fold"


def main():
    if gpu_test_runner.built_binary("test_dsv41_window_fold", BINARY) is None:
        return 0
    result = subprocess.run([str(BINARY)], capture_output=True, text=True)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS dsv41 window fold" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
