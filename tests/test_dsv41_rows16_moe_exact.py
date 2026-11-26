#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import gpu_test_runner

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/cuda/dsv41_rows16_moe_exact.cu"
LIBS = [
    "build/libsparkpipe_runtime.a",
    "build/libsparkpipe_model_common.a",
    "build/libsparkpipe_core.a",
]


def main():
    if not gpu_test_runner.has_device("test_dsv41_rows16_moe_exact"):
        return 0
    nvcc = os.environ.get("NVCC", "nvcc")
    if shutil.which(nvcc) is None:
        raise SystemExit("FAIL test_dsv41_rows16_moe_exact: sm_121 device present but nvcc unavailable")
    built = subprocess.run(["make", "-s", "-j4", *LIBS], cwd=ROOT, capture_output=True, text=True)
    if built.returncode != 0:
        print(built.stdout + built.stderr, end="")
        return 1
    with tempfile.TemporaryDirectory(prefix="dsv41_rows16_moe_exact_") as tmp:
        binary = Path(tmp) / "test_dsv41_rows16_moe_exact"
        cmd = [
            nvcc, "-O3", "-std=c++17",
            "-gencode", "arch=compute_121a,code=sm_121a",
            "-DDSV41_FLASH_TP_DEGREE=16u",
            "-DDSV41_FLASH_ROWS=1",
            '-DDSV41_FLASH_ADAPTER_ID="dsv41_flash-rows16"',
            '-DDSV41_FLASH_MODULE_TAG="dsv41_flash_rows16_stage"',
            '-DDSV41_FLASH_CONTRACT_SHA256="0"',
            f"-I{ROOT}",
            f"-I{ROOT / 'include'}",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'modules/dsv41_flash_resident_decode_stage/include'}",
            f"-I{ROOT / 'model-families/dsv41_flash/source'}",
            f"-I{ROOT / 'model-families/dsv41_flash/include'}",
            f"-I{ROOT / 'model-families/common/include'}",
            str(SOURCE),
            str(ROOT / "runtime/json.c"),
            str(ROOT / "runtime/filesystem.c"),
            str(ROOT / "src/spark_status.c"),
            *[str(ROOT / lib) for lib in LIBS],
            "-lpthread", "-ldl", "-lcuda",
            "-o", str(binary),
        ]
        comp = subprocess.run(cmd, capture_output=True, text=True)
        if comp.returncode != 0:
            print(comp.stdout + comp.stderr, end="")
            return 1
        run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=600)
        print(run.stdout + run.stderr, end="")
        return 0 if run.returncode == 0 and "PASS dsv41 rows16 moe exact" in run.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
