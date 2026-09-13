#!/usr/bin/env python3
"""Absolute pins on the hardware-topology constants - both profiles, all homes.

The C header (include/sparkpipe/spark_hardware_topology.h) and the Python
mirror (tools/hardware_topology.py) must carry exactly these numbers, and
the shared consumers must keep aliasing them instead of growing fresh
copies. Unlike test_hardware_topology_mirror.py, which checks the two
homes against each other, every expected value here is an independent
literal: if BOTH sides drift together, this gate still fails.

Changing a pinned number is a performance event, not an edit: update the
C header, the mirror, this file, and the C pin binary
(tests/test_hardware_topology_pins.c) in the same commit.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import hardware_topology as ht  # noqa: E402

HEADER_PATH = os.path.join(ROOT, "include", "sparkpipe",
                           "spark_hardware_topology.h")

# The ground truth. GB10 values are the measured/calibrated facts cited in
# the header; MI350P pins only what is frozen or spec-pinned (capability,
# wavefront lanes) and keeps explicit zero sentinels elsewhere.
EXPECTED = {
    "cuda.sm121.gb10": {
        "multiprocessor_count": 48,
        "wavefront_lanes": 32,
        "compute_capability_major": 12,
        "compute_capability_minor": 1,
        "static_shared_limit_bytes": 49152,
        "max_dynamic_shared_bytes": 101376,
        "shared_per_sm_bytes": 131072,
        "l2_cache_bytes": 0,
        "max_clock_khz": 2550000,
        "memory_bandwidth_gbps": 273.0,
    },
    "rocm.gfx950.mi350p": {
        "multiprocessor_count": 0,
        "wavefront_lanes": 64,
        "compute_capability_major": 9,
        "compute_capability_minor": 5,
        "static_shared_limit_bytes": 0,
        "max_dynamic_shared_bytes": 0,
        "shared_per_sm_bytes": 0,
        "l2_cache_bytes": 0,
        "max_clock_khz": 0,
        "memory_bandwidth_gbps": 0.0,
    },
}

FIELD_MAP = {
    "SPARK_HW_MULTIPROCESSOR_COUNT": "multiprocessor_count",
    "SPARK_HW_WAVEFRONT_LANES": "wavefront_lanes",
    "SPARK_HW_COMPUTE_CAPABILITY_MAJOR": "compute_capability_major",
    "SPARK_HW_COMPUTE_CAPABILITY_MINOR": "compute_capability_minor",
    "SPARK_HW_STATIC_SHARED_LIMIT_BYTES": "static_shared_limit_bytes",
    "SPARK_HW_MAX_DYNAMIC_SHARED_BYTES": "max_dynamic_shared_bytes",
    "SPARK_HW_SHARED_PER_SM_BYTES": "shared_per_sm_bytes",
    "SPARK_HW_L2_CACHE_BYTES": "l2_cache_bytes",
    "SPARK_HW_MAX_CLOCK_KHZ": "max_clock_khz",
}

GB10_MARKER = "#if SPARK_HW_TARGET_ID == SPARK_HW_TARGET_CUDA_SM121_GB10"
MI350P_MARKER = "#elif SPARK_HW_TARGET_ID == SPARK_HW_TARGET_ROCM_GFX950_MI350P"
END_MARKER = "#else"


def parse_numeric(token):
    token = token.strip()
    if token.startswith("("):
        return None
    cleaned = token.rstrip("uUlL")
    try:
        return float(cleaned) if "." in cleaned else int(cleaned, 0)
    except ValueError:
        return None


def profile_block(header_text, start_marker, end_marker):
    start = header_text.index(start_marker)
    end = header_text.index(end_marker, start)
    values = {}
    for line in header_text[start:end].splitlines():
        stripped = line.strip()
        if not stripped.startswith("#define"):
            continue
        parts = stripped.split()
        if len(parts) < 3 or not parts[1].startswith("SPARK_HW_"):
            continue
        parsed = parse_numeric(parts[2])
        if parsed is not None:
            values[parts[1]] = parsed
    return values


def main() -> int:
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    with open(HEADER_PATH, "r", encoding="utf-8") as handle:
        header_text = handle.read()

    blocks = {
        ht.TARGET_CUDA_SM121_GB10: profile_block(
            header_text, GB10_MARKER, MI350P_MARKER),
        ht.TARGET_ROCM_GFX950_MI350P: profile_block(
            header_text, MI350P_MARKER, END_MARKER),
    }

    for target_id, expected in EXPECTED.items():
        profile = ht.select_target(target_id)
        for field, want in expected.items():
            got_py = getattr(profile, field)
            check(got_py == want,
                  f"python mirror {target_id}.{field}: {got_py!r} != pinned {want!r}")
            c_field = next((c for c, f in FIELD_MAP.items() if f == field), None)
            c_block = blocks[target_id]
            if c_field is not None:
                check(c_field in c_block,
                      f"c header {target_id}: {c_field} missing/unparsable")
                if c_field in c_block:
                    got_c = c_block[c_field]
                    check(got_c == want,
                          f"c header {target_id}.{field} ({c_field}): "
                          f"{got_c!r} != pinned {want!r}")

    # Wiring contract: shared consumers must alias the header constants.
    wiring = [
        ("inference/kernels/layout.cuh",
         "#define LM_SMEM_STATIC_LIMIT SPARK_HW_STATIC_SHARED_LIMIT_BYTES"),
        ("inference/kernels/layout.cuh",
         "#define LM_SMEM_SM_TOTAL SPARK_HW_SHARED_PER_SM_BYTES"),
        ("runtime/workspace.h",
         "#define LM_WS_SHARED_LIMIT SPARK_HW_SHARED_PER_SM_BYTES"),
        ("model-families/common/include/sparkpipe/spark_lm_kernels.cuh",
         "#define SPARK_LM_WARP_LANES SPARK_HW_WAVEFRONT_LANES"),
    ]
    for rel_path, needle in wiring:
        path = os.path.join(ROOT, rel_path)
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        check(needle in text,
              f"{rel_path}: lost the topology alias ({needle}); "
              f"a literal copy has been reintroduced")

    # The qwen GDN chunk gates must still compare against the header ceiling.
    for rel_path in (
        "modules/qwen38_27b_resident_decode_stage/source/"
        "spark_qwen38_27b_resident_decode_stage_cuda.cu",
        "modules/qwen38_max_resident_decode_stage/source/"
        "spark_qwen38_max_resident_decode_stage_cuda.cu",
    ):
        path = os.path.join(ROOT, rel_path)
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        check("<= SPARK_HW_MAX_DYNAMIC_SHARED_BYTES" in text,
              f"{rel_path}: target-fit assert no longer uses the header ceiling")

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        print(f"FAIL {len(failures)} topology constant pin(s)")
        return 1
    print("topology constant pins hold across c header + python mirror + "
          "consumer wiring (gb10 smem/warp/clock/bandwidth, mi350p capability)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
