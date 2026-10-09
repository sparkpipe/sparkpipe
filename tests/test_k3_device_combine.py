#!/usr/bin/env python3
"""The stage runner's device collective reduces every rank's bf16 partial in fp32, in rank order, converting once.

The device collective's host round uses a registered fused combine when there is
one and otherwise adds the peers into a zeroed bf16 buffer one at a time, rounding
after every add (fifteen roundings at TP16). The stage runner runs every collective,
wide payloads included, on one device collective. It must register the fused
combine, and that combine must be the shared fixed-order fp32 rank sum, so every
rank converts the same fp32 total to bf16 once and all ranks hold identical bits.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "inference/runner/stage_runner.cu"
KERNELS = ROOT / "model-families/common/include/sparkpipe/spark_tp_mesh_kernels.cuh"
COLLECTIVE = ROOT / "ring/transport/tp_device_collective.c"


def body(source, name):
    match = re.search(r"static SparkStatus " + name + r"\([^)]*\)\s*\{(.*?)\n\}", source, re.S)
    return match.group(1) if match else None


def main():
    failures = []
    runner = RUNNER.read_text()
    fused = body(runner, "StageRunnerCombineSumRanksF32")
    if fused is None or "SparkTpLaunchSumRanksF32(" not in fused:
        failures.append("StageRunnerCombineSumRanksF32 does not launch the shared fp32 rank sum")
    if not re.search(r"device_config\.combine_fused_bf16_function\s*=\s*StageRunnerCombineSumRanksF32;", runner):
        failures.append("device_config does not register the fp32 fused combine")
    if "wide_config" in runner:
        failures.append("the runner still configures a second device collective")
    kernel = re.search(r"SparkTpSumRanksF32Kernel\((.*?)\n\}", KERNELS.read_text(), re.S)
    if kernel is None or not re.search(r"for \( source = 0u; source < source_count; source\+\+ \)", kernel.group(1)) \
            or "float2 acc" not in KERNELS.read_text():
        failures.append("the shared rank sum is not an fp32 accumulation in rank order")
    combine = body(COLLECTIVE.read_text(), "SparkTpDeviceCollectiveRoundCombine")
    if combine is None or combine.find("combine_fused_bf16") < 0 or \
            combine.find("combine_fused_bf16") > combine.find("SparkTpDeviceCollectiveCombineSerial"):
        failures.append("the host round no longer prefers the fused combine over the serial bf16 adds")
    for failure in failures:
        print("FAIL " + failure)
    if failures:
        return 1
    print("PASS the stage runner's one device collective reduces in fp32 in rank order, converting once")
    return 0


if __name__ == "__main__":
    sys.exit(main())
