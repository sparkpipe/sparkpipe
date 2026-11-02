"""Refuse a stage-runner graph capture on an sm_121 device after a copy and a collective failure were recorded, and check the failures survive the refusal."""
import subprocess
import sys
from pathlib import Path

import gpu_test_runner

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/runner_capture_flags_cuda"


def main():
    if gpu_test_runner.built_binary("test_runner_capture_flags_cuda", BINARY) is None:
        return 0
    result = subprocess.run([str(BINARY)], capture_output=True, text=True, timeout=600)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS runner_capture_flags_cuda" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
