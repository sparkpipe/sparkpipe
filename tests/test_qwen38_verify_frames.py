"""On an sm_121 device, qwen38-stage verify frames reproduce one-row decode bit for bit and keep_rows restores the kept rows' state.

Builds build/test_qwen38_verify_frames and build/test_qwen38_verify_frames_bf16: synthetic small-weight stages derived from
the Qwen 3.8 27B geometry (dense, with FP8 and with BF16 linear weights), the Qwen 3.8 Max geometry (rank-head attention,
NVFP4 experts) and the Qwen 3.8 Flash geometry (hyper-connection streams, the n-gram embedding with its dilated convolution
and token history, and the block indexer with its key tail; NVFP4 experts with an FP8 n-gram table in the first build, the
publisher's BF16 experts and BF16 n-gram table in the second), TP degree 1. For frames of 1, 2, 7, 16, 17, 40 and 256 rows, every row's stage output equals the
same row decoded alone, the frame leaves the decodes' attention and index KV and leaves every recurrent part untouched, and
keep_rows of k rows leaves the state of k one-row decodes; the next two decodes, the KV and the state then equal a run that
never saw the draft. A frame over two sequences, every row kept, matches too. The binary stops at the first mismatch.
"""

import subprocess
import sys
from pathlib import Path

import gpu_test_runner

ROOT = Path(__file__).resolve().parents[1]
BINARIES = ("test_qwen38_verify_frames", "test_qwen38_verify_frames_bf16")


def main():
    failures = 0
    for name in BINARIES:
        binary = ROOT / "build" / name
        if gpu_test_runner.built_binary(name, binary) is None:
            return 0
        result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=900)
        print(result.stdout + result.stderr, end="")
        failures += 0 if result.returncode == 0 and "PASS qwen38 verify frames" in result.stdout else 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
