#!/usr/bin/env python3
"""GB10 round trip of the DSV4.1 indexer's per-rank query pack through the all-gather layout.

Builds build/test_dsv41_flash_index_pack and runs it on an sm_121 device: at TP16, 8, 4, 2 and 1 each emulated rank packs its
index heads' queries and weights at the stage's pack width, the packs are laid out rank by rank as the all-gather leaves them,
and the unpack must return every query element and weight at its global head, bit for bit.
"""

import subprocess
import sys
from pathlib import Path

import gpu_test_runner

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_dsv41_flash_index_pack"


def main():
    if gpu_test_runner.built_binary("test_dsv41_flash_index_pack", BINARY) is None:
        return 0
    result = subprocess.run([str(BINARY)], capture_output=True, text=True)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS dsv41 index pack" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
