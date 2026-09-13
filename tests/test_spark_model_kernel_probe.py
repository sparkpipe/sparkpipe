#!/usr/bin/env python3

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE_SHA = "a" * 64


def run(command: list[str], expected: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != expected:
        raise AssertionError(
            f"command returned {result.returncode}, expected {expected}: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def main() -> int:
    with tempfile.TemporaryDirectory() as directory_name:
        directory = pathlib.Path(directory_name)
        module = directory / "libfake_model_kernel_probe.so"
        harness = directory / "spark_model_kernel_characterize"
        receipt = directory / "receipt.json"
        run([
            "cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2", "-fPIC", "-shared",
            "-I", str(ROOT / "include"),
            str(ROOT / "tests" / "fake_model_kernel_probe_module.c"),
            "-o", str(module),
        ])
        run([
            "cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2",
            "-I", str(ROOT / "include"),
            str(ROOT / "tools" / "hardware" / "spark_model_kernel_characterize.c"),
            "-ldl", "-lm", "-o", str(harness),
        ])
        run([
            str(harness),
            "--question", "GB10-NATIVE-MMA-001",
            "--provider", str(module),
            "--source-package-sha256", SOURCE_SHA,
            "--run-id", "model-kernel-probe-unit",
            "--topology", "unit",
            "--node", "spark0",
            "--model", "glm52_fp8_experts_bf16_rest",
            "--role", "expert",
            "--candidate", "fp8_e4m3",
            "--kernel-class", "gemm",
            "--route-distribution", "uniform",
            "--batch", "8",
            "--context", "2048",
            "--iterations", "16",
            "--output", str(receipt),
        ])
        run([
            sys.executable,
            str(ROOT / "tools" / "spark_hardware_qualify.py"),
            "validate-receipt", str(receipt),
        ])
        document = json.loads(receipt.read_text())
        observation = document["answers"][0]["observations"][0]
        assert observation["parameters"]["model_id"] == "glm52_fp8_experts_bf16_rest"
        assert observation["parameters"]["candidate"] == "fp8_e4m3"
        assert observation["metrics"]["latency_p99_ns"] == 2000
        assert observation["metrics"]["numerical_pass"] is True
        assert document["answers"][0]["summary"]["native_tensor_core"] is True
        assert document["answers"][0]["summary"]["provider_build_identity"] == "b" * 64

        run([
            str(harness),
            "--question", "GB10-NATIVE-MMA-001",
            "--provider", str(ROOT / "build" / "does-not-exist.so"),
            "--source-package-sha256", SOURCE_SHA,
            "--run-id", "model-kernel-probe-unit",
            "--topology", "unit",
            "--node", "spark0",
            "--model", "glm52_fp8_experts_bf16_rest",
            "--candidate", "fp8_e4m3",
            "--batch", "8",
            "--output", str(receipt),
        ], 1)
    print("PASS exact production model-kernel probe ABI and receipt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
