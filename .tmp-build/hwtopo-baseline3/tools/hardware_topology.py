"""Python mirror of include/sparkpipe/spark_hardware_topology.h.

One source of truth per hardware target for the Python cost models, exactly
matching the C header's profiles so a number can never drift between a kernel
build and an estimate. The C header is normative for device code; this mirror
is normative for tools/ estimates and must be kept field-for-field identical
with it (same frozen target ids, same pins, same sentinel semantics).

Target ids are the frozen strings from hwiface_v1.md section 6:

    cuda.sm121.gb10      NVIDIA GB10 (Spark workstation GPU), SM 12.1
    rocm.gfx950.mi350p   AMD Instinct MI350P (gfx950 / CDNA4)

Selection mirrors the C header: an explicit id wins; then the
SPARK_HW_TARGET_ID environment variable; then the documented fallback to
cuda.sm121.gb10 (every deployment in today's tree is GB10). An unknown id is
a hard error - the C header's #error, translated.

Sentinel contract (same as the C header): a 0 value means "this profile does
not pin the number at compile time". Treat 0 as unpinned - take the value
from the live SparkHwTarget descriptor instead; never use 0 as a budget or
bound. Use HardwareTopology.pinned() to make the check explicit.

Provenance per constant matches the C header's citations: "measured" values
come from receipts or docs in-tree; "advisory" values are the frozen advisor
notes awaiting on-hardware confirmation
(docs/coord/plan_amd_gfx950_mi350p.md section 3.3: measure, then pin).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

TARGET_CUDA_SM121_GB10 = "cuda.sm121.gb10"
TARGET_ROCM_GFX950_MI350P = "rocm.gfx950.mi350p"

# The C header falls back to GB10 when no accelerator compiler names a target;
# this mirror falls back identically.
DEFAULT_TARGET_ID = TARGET_CUDA_SM121_GB10


@dataclass(frozen=True)
class HardwareTopology:
    """Field-for-field mirror of the C header's per-profile constants."""

    target_id: str
    # SM/CU count: capacity math and tests only - launch geometry still takes
    # its count from the runtime query into the SparkHwTarget descriptor.
    multiprocessor_count: int
    wavefront_lanes: int
    compute_capability_major: int
    compute_capability_minor: int
    static_shared_limit_bytes: int
    max_dynamic_shared_bytes: int
    shared_per_sm_bytes: int
    l2_cache_bytes: int
    max_clock_khz: int
    memory_bandwidth_gbps: float

    def pinned(self, value) -> bool:
        """SPARK_HW_VALUE_IS_PINNED: False means the profile leaves it unpinned."""
        return value != 0


# cuda.sm121.gb10 - measured facts:
#   48 SMs, 128 KB L1/shared per SM, up to 2.55 GHz
#   (docs/archive/GB10_CUDA_COST_MODEL_CALIBRATION.md:23);
#   101376 B dynamic-shared opt-in ceiling
#   (hwiface_v1.md section 4.1); LPDDR5X 273 GB/s (roadmap:29).
#   L2 cache size has no in-tree measurement yet and stays unpinned.
_GB10 = HardwareTopology(
    target_id=TARGET_CUDA_SM121_GB10,
    multiprocessor_count=48,
    wavefront_lanes=32,
    compute_capability_major=12,
    compute_capability_minor=1,
    static_shared_limit_bytes=49152,
    max_dynamic_shared_bytes=101376,
    shared_per_sm_bytes=131072,
    l2_cache_bytes=0,
    max_clock_khz=2550000,
    memory_bandwidth_gbps=273.0,
)

# rocm.gfx950.mi350p - advisor-provided until measured on hardware
# (docs/coord/advisor_amd.md; plan section 3.3: measure, then pin):
#   64-lane wavefronts is frozen (hwiface_v1.md section 4.1, informational
#   per REV2 A12 - it feeds no gate); everything numeric stays unpinned.
_MI350P = HardwareTopology(
    target_id=TARGET_ROCM_GFX950_MI350P,
    multiprocessor_count=0,
    wavefront_lanes=64,
    compute_capability_major=0,
    compute_capability_minor=0,
    static_shared_limit_bytes=0,
    max_dynamic_shared_bytes=0,
    shared_per_sm_bytes=0,
    l2_cache_bytes=0,
    max_clock_khz=0,
    memory_bandwidth_gbps=0.0,
)

PROFILES = {
    TARGET_CUDA_SM121_GB10: _GB10,
    TARGET_ROCM_GFX950_MI350P: _MI350P,
}


def select_target(target_id: str | None = None) -> HardwareTopology:
    """Resolve a profile: argument > SPARK_HW_TARGET_ID env > default.

    Unknown ids raise ValueError - the C header's #error, translated.
    """
    resolved = target_id or os.environ.get("SPARK_HW_TARGET_ID") or DEFAULT_TARGET_ID
    profile = PROFILES.get(resolved)
    if profile is None:
        known = ", ".join(sorted(PROFILES))
        raise ValueError(
            f"unknown hardware target {resolved!r}; expected one of: {known}")
    return profile


__all__ = [
    "TARGET_CUDA_SM121_GB10",
    "TARGET_ROCM_GFX950_MI350P",
    "DEFAULT_TARGET_ID",
    "HardwareTopology",
    "PROFILES",
    "select_target",
]
