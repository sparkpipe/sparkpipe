#!/usr/bin/env python3
"""The gemma4 serving adapter .so must dlopen in any host process.

The minimax lane-10 connect failure (71fb5d97): an adapter linked from the
runtime static archives without -lcudart leaves cudaFree/cudaMalloc/
cudaMemcpyAsync/cudaStreamSynchronize undefined - the residentd provides
them, but sparkpipe_model_api does not and its RTLD_NOW load fails closed
as NOT_FOUND (engine_connect status=3). module_build_release.sh now fails
the release on adapter undefined symbols (adapter-dependencies.log), so the
gemma4 adapter build must link the CUDA runtime library.
Run: python3 tests/test_gemma4_adapter_selfcontained.py
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]

MODULE = REPOSITORY / "modules/gemma4_resident_decode_stage"


def main() -> int:
    failures: list[str] = []
    built = subprocess.run(["make", "-n", "--no-print-directory", "MAKE=true", "-C", str(MODULE), "adapter", "CUDA_HOME=/spark-cuda"], text=True, capture_output=True)
    commands = [tokens for tokens in map(shlex.split, built.stdout.replace("\\\n", " ").splitlines()) if "-o" in tokens and any(token.endswith("spark_gemma4_serving_adapter.c") for token in tokens)]
    if built.returncode != 0 or len(commands) != 1:
        failures.append("make -n adapter in the gemma4 module does not compile the gemma4 adapter once: " + built.stderr.strip())
    else:
        if "-lcudart" not in commands[0]:
            failures.append("adapter build does not link -lcudart (undefined CUDA runtime symbols)")
        if "-L/spark-cuda/lib64" not in commands[0]:
            failures.append("adapter build does not search $(CUDA_HOME)/lib64 for libcudart")

    if failures:
        for failure in failures:
            print(f"MISMATCH {failure}")
        print(f"FAILED {len(failures)} check(s)")
        return 1
    print("PASS gemma4 adapter links the CUDA runtime (self-contained .so)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
