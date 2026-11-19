#!/usr/bin/env python3
"""GB10 bitwise check of the DSV4.1 prompt fold's state capture.

Builds build/test_dsv41_fold_capture and runs it on an sm_121 device. A prompt's last prefill step
runs through the prompt's last token, while the cache checkpoint it publishes is the state one
token earlier. For each folded lane the step saves the ratio-2 compressor's pending row and the
n-gram history as they stand before the last row. Those pieces must be the same bits that the
compressor save and history kernels leave in a lane's live state after a step that ends one token
earlier, for lanes of two rows or more, shuffled rows, several lanes a step and both parities of
the checkpoint. A checkpoint whose last group is complete keeps no pending row, so its compressor
piece is left as it was, and lanes that do not fold are not touched.
"""

import subprocess
import sys
from pathlib import Path

import gpu_test_runner

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_dsv41_fold_capture"


def main():
    if gpu_test_runner.built_binary("test_dsv41_fold_capture", BINARY) is None:
        return 0
    result = subprocess.run([str(BINARY)], capture_output=True, text=True)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS dsv41 fold capture" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
