#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_test_runner import ROOT, built_binary  # noqa: E402

MODES = ("--run", "--peer-reduce", "--reduce-scatter", "--pipelined", "--all-to-all", "--pair-first", "--pair-first-gather", "--pair-first-boundary")


def main():
    binary = built_binary("test_tp_mesh_hardware_probe", ROOT / "build/tp_mesh_hardware_probe")
    if binary is None:
        return 0
    environment = dict(os.environ, CUDA_DEVICE_MAX_CONNECTIONS="32")
    failures = []
    for mode in MODES:
        result = subprocess.run([str(binary), mode], cwd=ROOT, capture_output=True, text=True, timeout=1800, env=environment)
        last = (result.stdout.strip().splitlines() or [""])[-1]
        if result.returncode != 0 or not last.startswith("PASS tp_mesh_hardware_probe"):
            failures.append(mode)
            print("".join(line + "\n" for line in (result.stdout + result.stderr).splitlines() if not line.startswith("TAIL "))[-4000:], end="")
        print(f"{mode}: {last}")
    for mode in failures:
        print(f"FAIL tp_mesh_hardware_probe {mode}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
