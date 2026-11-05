"""On an sm_121 device, mimo26-stage verify frames reproduce one-row decode bit for bit and a rejected tail leaves no trace.

Builds build/test_mimo26_verify_frames: synthetic stages derived from the MiMo V2.6 Flash and Pro geometry structs at
their production TP16 split (16 ranks in threads on one GPU, emulated all-gather, all-to-all and all-reduce,
16-way context-sharded paged KV, rank-local experts), with FP8 block-scaled projections, MXFP4 routed experts,
sliding-window layers with attention sinks and full-attention layers. For frames of 1, 2, 7, 16, 17, 40 and 256 rows,
every row's stage output equals the same row decoded alone, and every rank's KV equals that of the one-row decodes.
After keeping k rows, the next two decodes and the KV up to them equal a run that never saw the draft. A frame over
two sequences matches too.
"""

import subprocess
import sys
from pathlib import Path

import gpu_test_runner

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_mimo26_verify_frames"


def main():
    if gpu_test_runner.built_binary("test_mimo26_verify_frames", BINARY) is None:
        return 0
    result = subprocess.run([str(BINARY)], capture_output=True, text=True, timeout=1800)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS mimo26 verify frames" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
