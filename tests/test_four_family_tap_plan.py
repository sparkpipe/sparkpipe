"""On an sm_121 device, verify TapPlan registration, error contracts, zero-overhead disablement, and exact layer-output residual capture across glm-5.3, a.x-k2, glm-5.3-flash, k-exaone-2.0, and gemma4-31b."""

import subprocess
import sys
from pathlib import Path

import gpu_test_runner

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build" / "test_four_family_tap_plan"


def main():
    if gpu_test_runner.built_binary("test_four_family_tap_plan", BINARY) is None:
        return 0
    result = subprocess.run([str(BINARY)], capture_output=True, text=True, timeout=900)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS four-family tap plan" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
