#!/usr/bin/env python3
"""Compile the mimo26 stagepack format header and pin its census.

The header is the C-side contract for the M26P wire tools/mimo26_stagepack.py
emits; this test compiles it standalone (-Wall -Wextra -Werror) and asserts
the expected-tensor-count formula for format version 2 (one fused per-rank
QKV entry per layer): pro TP8 rank packs carry 691 tensors, flash TP4 472,
and the synthetic mini fixture in tests/test_mimo26_stagepack.py plans 49.
The per-rank qkv segment rows and scale rows are pinned against the census
grids: flash full 3392 rows / 27 blocks (x4 = 108), flash swa 3712 / 29
(x4 = 116), pro 3392 / 27 (x8 = 216).
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
            "  if (SparkMimo26StagePackExpectedTensorCount(70u, 60u, 69u) != 691u) return 1;\n"
            "  if (SparkMimo26StagePackExpectedTensorCount(48u, 39u, 47u) != 472u) return 2;\n"
            "  if (SparkMimo26StagePackExpectedTensorCount(5u, 3u, 4u) != 49u) return 3;\n"
            "  if (SparkMimo26StagePackQkvRankRows(64u, 4u, 192u, 128u, 4u) != 3392u) return 8;\n"
            "  if (SparkMimo26StagePackQkvRankScaleRows(3392u) * 4u != 108u) return 9;\n"
            "  if (SparkMimo26StagePackQkvRankScaleRows(SparkMimo26StagePackQkvRankRows(64u, 8u, 192u, 128u, 4u)) * 4u != 116u) return 10;\n"
            "  if (SparkMimo26StagePackQkvRankScaleRows(SparkMimo26StagePackQkvRankRows(128u, 8u, 192u, 128u, 8u)) * 8u != 216u) return 11;\n"
            "  if (SPARK_MIMO26_STAGEPACK_FORMAT_VERSION != 2u || SPARK_MIMO26_STAGEPACK_TENSOR_QKV != 24u) return 12;\n"
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
    print("PASS mimo26 stagepack format header (v2 counts 691/472/49, qkv segments 108/116/216 pinned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
