#!/usr/bin/env python3
"""Compile the mimo26 stagepack format header and pin its census.

The header is the C-side contract for the M26P wire tools/mimo26_stagepack.py
emits; this test compiles it standalone (-Wall -Wextra -Werror) and asserts
the expected-tensor-count formula against the MEASURED pack directories:
pro TP8 rank packs carry 831 tensors, flash TP4 carries 568 (the rank1 pro
pack receipt on spark1 is 831 tensors / 71,479,218,944 bytes), and the
synthetic mini fixture in tests/test_mimo26_stagepack.py plans 59.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "modules/mimo26_resident_decode_stage/source/spark_mimo26_stagepack_format.h"


def main() -> int:
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        print("no C compiler available; skipping (CI always has one)")
        return 0
    with tempfile.TemporaryDirectory(prefix="mimo26-fmt-") as raw:
        tmp = Path(raw)
        source = tmp / "probe.c"
        source.write_text(
            "#include <stdio.h>\n"
            "#include \"" + str(HEADER) + "\"\n"
            "int main(void) {\n"
            "  struct SparkMimo26StagePackHeader h = {0};\n"
            "  struct SparkMimo26StagePackEntry e = {0};\n"
            "  (void)h; (void)e;\n"
            "  if (SparkMimo26StagePackExpectedTensorCount(70u, 60u, 69u) != 831u) return 1;\n"
            "  if (SparkMimo26StagePackExpectedTensorCount(48u, 39u, 47u) != 568u) return 2;\n"
            "  if (SparkMimo26StagePackExpectedTensorCount(5u, 3u, 4u) != 59u) return 3;\n"
            "  if (SPARK_MIMO26_STAGEPACK_TENSOR_EXPERT_DOWN != 33u) return 4;\n"
            "  if (SPARK_MIMO26_STAGEPACK_WEIGHT_MXFP4_E2M1_E8M0G32 != 9u) return 5;\n"
            "  if (SparkMimo26StagePackIsExpert(SPARK_MIMO26_STAGEPACK_TENSOR_EXPERT_GATE) != 1u) return 6;\n"
            "  if (SparkMimo26StagePackIsExpert(SPARK_MIMO26_STAGEPACK_TENSOR_O_PROJ) != 0u) return 7;\n"
            "  puts(\"OK\");\n"
            "  return 0;\n"
            "}\n")
        binary = tmp / "probe"
        build = subprocess.run(
            [cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "include"),
             str(source), "-o", str(binary)],
            capture_output=True, text=True)
        if build.returncode != 0:
            print(build.stderr)
            return 1
        run = subprocess.run([str(binary)], capture_output=True, text=True)
        if run.returncode != 0 or run.stdout.strip() != "OK":
            print("probe failed:", run.returncode, run.stdout, run.stderr)
            return 1
    print("PASS mimo26 stagepack format header (census 831/568/59 pinned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
