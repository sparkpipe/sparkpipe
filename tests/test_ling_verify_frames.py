"""On an sm_121 device, ling-stage verify frames reproduce one-row decode bit for bit and keep_rows restores the kept rows' recurrent state.

Builds build/test_ling_verify_frames: synthetic small-weight stages derived from the Solar Open 2 geometry (GQA with
KDA) and the Ling 3.0 Flash geometry (MLA with KDA, verify rows switched on for the test), every other feature switch
kept, TP degree 1. For frames of 1, 2, 7, 16, 17, 40 and 256 rows, every row's stage output equals the same row decoded
alone, the frame leaves the decodes' KV and the recurrent state untouched, and keep_rows of k rows (k from 0 to the
frame) leaves the recurrent state of k one-row decodes; the next two decodes and the KV up to them then equal a run
that never saw the draft. A frame over two sequences, every row kept, matches too.
"""

import subprocess
import sys
from pathlib import Path

import gpu_test_runner

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_ling_verify_frames"


def main():
    if gpu_test_runner.built_binary("test_ling_verify_frames", BINARY) is None:
        return 0
    result = subprocess.run([str(BINARY)], capture_output=True, text=True, timeout=900)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS ling verify frames" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
