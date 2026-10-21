#!/usr/bin/env python3
"""GB10 check that skinny matrix-vector launches do not depend on how many rows they carry.

Builds build/test_skinny_rows and runs it on an sm_121 device. For mxfp8 and bf16 weights at the
DSV4.1 projection shapes, with bf16 and f32 outputs, every row of a 1 to 256 row launch (row-chunked
above 16 rows) and of a paired-target launch must be the same bits as a one-row launch of that row
alone. Every route of a grouped expert launch must match that route launched alone, for frames of
1 to 233 rows. Speculative verify frames rely on both to reproduce decode exactly. Perturbing one
activation must change only its own row.
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_skinny_rows"


def main():
    built = subprocess.run(["make", "-s", str(BINARY.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True)
    if built.returncode != 0:
        print(built.stdout + built.stderr)
        return 1
    if not BINARY.is_file():
        print("SKIP test_skinny_rows (nvcc unavailable on this host)")
        return 0
    if shutil.which("nvidia-smi") is None:
        print("SKIP test_skinny_rows (no GPU on this host)")
        return 0
    capabilities = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout.split()
    if "12.1" not in capabilities:
        print(f"SKIP test_skinny_rows (needs an sm_121 device; this host reports {capabilities or 'none'})")
        return 0
    result = subprocess.run([str(BINARY)], capture_output=True, text=True)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS skinny rows" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
