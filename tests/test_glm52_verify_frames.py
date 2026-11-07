"""On an sm_121 device, glm52-stage verify frames reproduce one-row decode bit for bit and a rejected tail leaves no trace.

Builds build/test_glm52_verify_frames and build/test_glm52_verify_frames_bf16: synthetic small-weight stages derived from the
GLM 5.3 Full and A.X K2 geometries (MLA with the DSA indexer selecting 64 keys, FP8 linears and experts with block scales, or
GLM 5.3 Full's BF16 linears and experts without scales in the second build, A.X K2's query gate, query concat,
norm gates, expert groups and YaRN table), TP degree 1 with the four heads a rank holds. For frames of 1, 2, 7, 16, 17,
40 and 256 rows, every row's stage output equals the same row decoded alone and the frame leaves the decodes' latent and
indexer KV at every position. After keeping k rows (k from 0 to the frame), the next two decodes and the KV up to them
equal a run that never saw the draft. A frame over two sequences matches too.
"""

import subprocess
import sys
from pathlib import Path

import gpu_test_runner

ROOT = Path(__file__).resolve().parents[1]
BINARIES = ("test_glm52_verify_frames", "test_glm52_verify_frames_bf16")


def main():
    failures = 0
    for name in BINARIES:
        binary = ROOT / "build" / name
        if gpu_test_runner.built_binary(name, binary) is None:
            return 0
        result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=900)
        print(result.stdout + result.stderr, end="")
        failures += 0 if result.returncode == 0 and "PASS glm52 verify frames" in result.stdout else 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
