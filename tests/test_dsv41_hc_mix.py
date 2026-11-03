#!/usr/bin/env python3
"""GB10 bitwise check of the DSV4.1 row-blocked hyper-connection mix against the task-parallel mix.

Builds build/test_dsv41_hc_mix and runs it on an sm_121 device. The row-blocked kernel reads each
row's streams once for all 25 mix tasks; every warp partial it writes must be the same bits as the
task-parallel kernel's, for 1 to 1024 rows including row counts that are not a multiple of the
block. Perturbing one stream element must change the result.
"""

import subprocess
import sys
from pathlib import Path

import gpu_test_runner

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/test_dsv41_hc_mix"


def main():
    if gpu_test_runner.built_binary("test_dsv41_hc_mix", BINARY) is None:
        return 0
    result = subprocess.run([str(BINARY)], capture_output=True, text=True)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS dsv41 hc mix" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
