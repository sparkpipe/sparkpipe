#!/usr/bin/env python3
"""make test-nvrm-gpu is a pass only where the open kernel module runs every GPU test.

Without /dev/nvidiactl the gate fails before it builds or runs anything,
naming the missing device. With it, a test binary that prints a SKIP line
fails the gate: test_device_nvrm skips its no-device checks on a GPU host, so
a gate run over it alone must fail and name it.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL = Path("/dev/nvidiactl")


def make(*arguments):
    return subprocess.run(["make", "--no-print-directory", "SPARKPIPE_DEVICE=nvrm", *arguments], cwd=ROOT, capture_output=True, text=True)


def main():
    failures = []
    if not CONTROL.exists():
        for goal in ("nvrm-gpu-present", "test-nvrm-gpu"):
            run = make(goal)
            if run.returncode == 0 or "/dev/nvidiactl is missing" not in run.stdout:
                failures.append(f"make {goal} on a host without the open kernel module must fail naming the device: {run.stdout.strip()[-300:]}")
        case = "without the open kernel module the gate fails before running anything"
    else:
        run = make("test-nvrm-gpu", "NVRM_GPU_TEST_BINARIES=build/test_device_nvrm")
        if run.returncode == 0 or "FAIL build/test_device_nvrm skipped part of the GPU gate" not in run.stdout:
            failures.append(f"a GPU gate whose test binary printed SKIP must fail and name it: {run.stdout.strip()[-300:]}")
        case = "on a GPU host a test that skips fails the gate"
    for failure in failures:
        print(f"FAIL {failure}")
    if failures:
        return 1
    print(f"PASS make test-nvrm-gpu passes only where every GPU test runs: {case}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    sys.exit(main())
