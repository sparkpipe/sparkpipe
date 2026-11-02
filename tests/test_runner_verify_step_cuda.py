"""On an sm_121 device, the CUDA stage runner hands the model every verify dispatch as a step that does not commit, for models with and without recurrent state, and commits decode and prefill steps."""
import subprocess
import sys
from pathlib import Path

import gpu_test_runner

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/runner_verify_step_cuda"


def main():
    if gpu_test_runner.built_binary("test_runner_verify_step_cuda", BINARY) is None:
        return 0
    result = subprocess.run([str(BINARY)], capture_output=True, text=True, timeout=300)
    print(result.stdout + result.stderr, end="")
    return 0 if result.returncode == 0 and "PASS runner_verify_step_cuda" in result.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
