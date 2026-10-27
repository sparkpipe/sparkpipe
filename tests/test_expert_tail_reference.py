#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_test_runner import ROOT, built_binary  # noqa: E402


def main():
    binary = built_binary("test_expert_tail_reference", ROOT / "build/test_expert_tail_reference")
    if binary is None:
        return 0
    result = subprocess.run([str(binary), "--run"], cwd=ROOT, capture_output=True, text=True, timeout=1800)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "OK expert tail reference" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
