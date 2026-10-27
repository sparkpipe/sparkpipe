#!/usr/bin/env python3
"""GB10 check that skinny matrix-vector launches do not depend on how many rows they carry.

Builds build/test_skinny_rows and runs it on an sm_121 device. For mxfp8 and bf16 weights at the
DSV4.1 projection shapes, with bf16 and f32 outputs, every row of a 1 to 256 row launch (row-chunked
above 16 rows) and of a paired-target launch must be the same bits as a one-row launch of that row
alone. Every route of a grouped expert launch must match that route launched alone, for frames of
1 to 233 rows. Speculative verify frames rely on both to reproduce decode exactly. Perturbing one
activation must change only its own row.
"""

import subprocess
import sys
from pathlib import Path

import gpu_test_runner

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_skinny_rows"


def main():
    if gpu_test_runner.built_binary("test_skinny_rows", BINARY) is None:
        return 0
    result = subprocess.run([str(BINARY)], capture_output=True, text=True)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS skinny rows" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
