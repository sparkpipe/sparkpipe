#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CUDA_SOURCE = ROOT / "tools" / "hardware" / "spark_cuda_characterize.cu"
NVME_SOURCE = ROOT / "tools" / "hardware" / "spark_nvme_characterize.cu"
PMTU_SOURCE = ROOT / "tools" / "hardware" / "spark_pmtu_characterize.c"
COMMON_HEADER = ROOT / "tools" / "hardware" / "spark_probe_common.h"
CUDA_GATE = ROOT / "tools" / "cuda13_sm121a_compile_gate.sh"
CUDA_STUB = ROOT / "tests" / "hardware_cuda_stub"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run(command: list[str]) -> None:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise AssertionError(f"command failed: {' '.join(command)}")


def main() -> int:
    cuda = CUDA_SOURCE.read_text(encoding="utf-8")
    nvme = NVME_SOURCE.read_text(encoding="utf-8")
    common = COMMON_HEADER.read_text(encoding="utf-8")
    gate = CUDA_GATE.read_text(encoding="utf-8")

    for required in (
        "gpu_bandwidth_ratio",
        'baseline_options.candidate = "gpu_only"',
        "options.batch_size",
        "options.kernel_count",
        r'\"sample_phase\": \"all\"',
        "cudaSetDeviceFlags(cudaDeviceMapHost)",
        "cudaFuncSetAttribute",
        "SparkCudaProbeMeasureReadReuse",
        "SparkCudaProbeMeasurePointerChase",
        "cudaDevAttrMemoryClockRate",
        '"GB10-GRAPH-001"',
        'SparkCudaProbeWriteJsonString(output, options.mode)',
        "SparkCudaProbeParseTelemetryValue",
        "memory_clock_text",
        "if (loaded && load_stream != nullptr)",
        r'\"dynamic_shared_bytes\": %u, \"iterations\": %u',
        r'\"sustained_seconds\": %u, \"iterations\": %u',
    ):
        require(required in cuda, f"CUDA hardware probe is missing {required}")
    atomic_start = cuda.index('"GB10-ATOMIC-001"')
    thermal_start = cuda.index('"GB10-THERMAL-001"')
    require('SparkCudaProbeWriteJsonString(output, options.mode)' in
            cuda[atomic_start:thermal_start],
            "CUDA atomic receipt omits the planned mode")
    require("properties.memoryClockRate" not in cuda,
            "CUDA hardware probe uses the removed cudaDeviceProp memoryClockRate field")

    for required in (
        "SparkNvmeProbeVerifyPattern",
        "SparkNvmeProbeCopyAndFingerprint",
        "value += 0x9e3779b97f4a7c15ull",
        "SparkNvmeProbeDeviceMix64(words[index] ^",
        "absolute_word_offset + index",
        "initial_fingerprint = SparkProbeMix64(",
        "word_count ^ (absolute_byte_offset / sizeof(uint64_t))",
        "cudaMemcpyHostToDevice, slot->stream",
        "cpu_fingerprint != device_fingerprint",
        "initialization_failed",
        "execution_failed",
        "thread.joinable()",
    ):
        require(required in nvme, f"NVMe hardware probe is missing {required}")

    for required in (
        "SparkProbeFingerprintWords",
        "SparkProbeSummarizeLatency",
        "SparkProbeWriteJsonString",
        "SparkProbeHexSha256IsValid",
    ):
        require(required in common, f"common probe support is missing {required}")

    require("tools/hardware/spark_cuda_characterize.cu" in gate,
            "CUDA 13 gate does not compile the GB10 probe")
    require("tools/hardware/spark_nvme_characterize.cu" in gate,
            "CUDA 13 gate does not compile the NVMe-to-GPU probe")
    require(
        "modules/dsv4_resident_decode_stage/source/"
        "spark_dsv4_resident_decode_stage_cuda.cu" in gate,
        "CUDA 13 gate does not compile the DSV4 resident module",
    )
    require(
        "modules/glm52_resident_decode_stage/source/"
        "spark_glm52_resident_decode_stage_cuda.cu" in gate,
        "CUDA 13 gate does not compile the GLM resident module",
    )
    require(
        "glm_codecs=(int6 int7 int8 fp8 nvfp4 mxfp4)" in gate,
        "CUDA 13 gate does not qualify every selectable GLM expert codec",
    )
    require(
        "--print-build-identity" in gate,
        "CUDA 13 gate does not consume canonical package build identity",
    )
    require(
        "inference/llms/glm5_2" not in gate,
        "CUDA 13 gate still compiles the removed GLM inference path",
    )
    require(
        "inference/llms/deepseek_v4" not in gate,
        "CUDA 13 gate still compiles the removed DSV4 inference path",
    )

    run([
        "cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
        "-Itools/hardware", "-fsyntax-only", str(PMTU_SOURCE),
    ])

    clang = shutil.which("clang++") or "/usr/local/swift/usr/bin/clang++"
    if pathlib.Path(clang).is_file():
        common_command = [
            clang,
            "-std=c++17",
            "-x", "cuda",
            "--cuda-host-only",
            "-nocudainc",
            "-nocudalib",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Itests/hardware_cuda_stub",
            "-Itools/hardware",
            "-fsyntax-only",
        ]
        run(common_command + [str(CUDA_SOURCE)])
        run(common_command + [str(NVME_SOURCE)])
    else:
        print("SKIP Clang CUDA host syntax: clang++ unavailable")

    print("PASS hardware CUDA, NVMe, PMTU, shared-support, and exact-target source contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
