"""On an sm_121 device, hy4-stage verify frames reproduce one-row decode bit for bit and a rejected tail leaves no trace.

Builds build/test_hy4_verify_frames: a synthetic small-weight stage derived from the Hy4 Preview geometry (four
hyper-connection streams, MLA with attention sinks and an output gate, the DSA indexer selecting 64 keys on its
indexer layers and reusing the selection on the others, MXFP8 linears and experts with UE8M0 block scales, a dense
first layer, half the experts on the rank) at TP degree 1 with the four heads a rank holds. For frames of 1, 2, 7, 16,
17, 40 and 256 rows, every row's stage output equals the same row decoded alone and the frame leaves the decodes'
latent and indexer KV at every position. After keeping k rows (k from 0 to the frame), the next two decodes and the
KV up to them equal a run that never saw the draft. A frame over two sequences matches too.
"""

import subprocess
import sys
from pathlib import Path

import gpu_test_runner

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_hy4_verify_frames"


def main():
    if gpu_test_runner.built_binary("test_hy4_verify_frames", BINARY) is None:
        return 0
    result = subprocess.run([str(BINARY)], capture_output=True, text=True, timeout=900)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS hy4 verify frames" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
