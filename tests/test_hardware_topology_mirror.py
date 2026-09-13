#!/usr/bin/env python3
"""The Python topology mirror must never drift from the C header.

tools/hardware_topology.py is a hand-maintained mirror of
include/sparkpipe/spark_hardware_topology.h. This test parses the C
header's per-profile #define blocks directly and asserts every field of
both HardwareTopology profiles matches - value, sentinel state (0 =
unpinned), and the frozen target-id strings. A number changed on the C
side without updating the mirror (or vice versa) fails here, not in some
estimate six months from now.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import hardware_topology as ht  # noqa: E402

HEADER_PATH = os.path.join(ROOT, "include", "sparkpipe",
                           "spark_hardware_topology.h")

# C define name -> HardwareTopology field.
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

DEFINE_RE = re.compile("^#define[ ]+([A-Z_0-9]+)[ ]+([^ ]+)")


def parse_numeric(token: str):
    token = token.strip()
    if token.startswith("(") :  # parenthesised expressions are not pins
        return None
    cleaned = token.rstrip("uUlL")
    try:
        return float(cleaned) if "." in cleaned else int(cleaned, 0)
    except ValueError:
        return None


def profile_block(header_text: str, start_marker: str, end_marker: str) -> dict:
    start = header_text.index(start_marker)
    end = header_text.index(end_marker, start)
    values = {}
    for line in header_text[start:end].splitlines():
        match = DEFINE_RE.match(line.strip())
        if not match:
            continue
        parsed = parse_numeric(match.group(2))
        if parsed is not None:
            values[match.group(1)] = parsed
    return values


def main() -> int:
    with open(HEADER_PATH, "r", encoding="utf-8") as handle:
        header_text = handle.read()

    gb10_c = profile_block(header_text, GB10_MARKER, MI350P_MARKER)
    mi350p_c = profile_block(header_text, MI350P_MARKER, END_MARKER)

    for target_id, c_values in (
        (ht.TARGET_CUDA_SM121_GB10, gb10_c),
        (ht.TARGET_ROCM_GFX950_MI350P, mi350p_c),
    ):
        profile = ht.select_target(target_id)
        assert profile.target_id == target_id, target_id
        for c_name, field in FIELD_MAP.items():
            assert c_name in c_values, f"{target_id}: {c_name} missing from header"
            expected = c_values[c_name]
            actual = getattr(profile, field)
            assert actual == expected, (
                f"{target_id}.{field}: python mirror has {actual!r}, "
                f"C header pins {expected!r} ({c_name})")

    # Sentinel contract: unpinned fields are exactly zero, pinned helper agrees.
    for target_id in (ht.TARGET_CUDA_SM121_GB10, ht.TARGET_ROCM_GFX950_MI350P):
        profile = ht.select_target(target_id)
        for field in ("multiprocessor_count", "wavefront_lanes",
                      "static_shared_limit_bytes", "max_dynamic_shared_bytes",
                      "shared_per_sm_bytes"):
            value = getattr(profile, field)
            assert ht.HardwareTopology.pinned(profile, value) == (value != 0), (
                f"{target_id}.{field}: pinned() disagrees with sentinel value")
        # The GB10 profile pins everything numeric except L2; the mirror must
        # keep that shape so cost models can rely on the pins.
        if target_id == ht.TARGET_CUDA_SM121_GB10:
            for field in ("multiprocessor_count", "wavefront_lanes",
                          "static_shared_limit_bytes",
                          "max_dynamic_shared_bytes", "shared_per_sm_bytes",
                          "max_clock_khz", "memory_bandwidth_gbps"):
                assert getattr(profile, field) != 0, (
                    f"gb10.{field} unexpectedly unpinned")

    # MI350P capability is spec-pinned (gfx950 reports 9.5, matching the
    # in-tree fail-closed guard); an unpinning must be a deliberate act that
    # updates this assertion alongside both sources.
    mi350p = ht.select_target(ht.TARGET_ROCM_GFX950_MI350P)
    assert (mi350p.compute_capability_major, mi350p.compute_capability_minor)         == (9, 5), "mi350p capability pin drifted from (9, 5)"
    assert gb10_c["SPARK_HW_COMPUTE_CAPABILITY_MAJOR"] == 12, "gb10 major"
    assert gb10_c["SPARK_HW_COMPUTE_CAPABILITY_MINOR"] == 1, "gb10 minor"

    print("hardware topology mirror matches the C header "
          "(gb10 + rocm.gfx950.mi350p, capability pins included)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
