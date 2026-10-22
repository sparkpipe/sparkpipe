#!/usr/bin/env python3
"""GB10 check that the split-K skinny launch can stand in for the per-rank sliced launches.

Builds build/test_skinny_split and runs it on an sm_121 device. For bf16 weights at the K3 MLA
down-projection shapes and one generic shape, with bf16 and f32 outputs, a full-width split launch
of 1 to 64 rows (row-chunked above 16) must equal, bit for bit, the sixteen per-rank slices that
one-row decode computes and the one-row split launch of each row. The default kernel at full width
must differ, which is why a verify frame above 16 rows takes the split launch. Launches it cannot
carry are refused.
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_skinny_split"


def main():
    built = subprocess.run(["make", "-s", str(BINARY.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True)
    if built.returncode != 0:
        print(built.stdout + built.stderr)
        return 1
    if not BINARY.is_file():
        print("SKIP test_skinny_split (nvcc unavailable on this host)")
        return 0
    if shutil.which("nvidia-smi") is None:
        print("SKIP test_skinny_split (no GPU on this host)")
        return 0
    capabilities = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout.split()
    if "12.1" not in capabilities:
        print(f"SKIP test_skinny_split (needs an sm_121 device; this host reports {capabilities or 'none'})")
        return 0
    result = subprocess.run([str(BINARY)], capture_output=True, text=True)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS skinny split" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
