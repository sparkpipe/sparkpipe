/*
 * Compile-time pin gate for the rocm.gfx950.mi350p topology profile.
 *
 * The GB10 profile is compiled into tests/test_hardware_topology_pins via
 * the default target selection. Nothing else in the tree selects MI350P,
 * so without this TU the #elif branch of spark_hardware_topology.h is
 * parsed by zero C compilers - only the python mirror test reads it as
 * text. This file forces the branch through a real compiler and pins its
 * frozen/spec values at compile time:
 *
 *   - wavefront width 64 and capability 9.5 (spec-pinned, matching the
 *     fail-closed expectation in the rocm target guard), and
 *   - every other numeric staying an exact zero sentinel (unpinned until
 *     measured on hardware - plan_amd_gfx950_mi350p.md section 3.3).
 *
 * Deliberately header-only: layout.cuh refuses unpinned SMEM profiles by
 * design, which is the correct loud failure for DEVICE consumers, while
 * this gate exists to prove the PROFILE itself compiles and holds.
 */
#include "sparkpipe/spark_hardware_topology.h"

#include <stdio.h>

#if SPARK_HW_TARGET_ID != SPARK_HW_TARGET_ROCM_GFX950_MI350P
#error "this pin binary builds the MI350P profile only"
#endif

static_assert(SPARK_HW_WAVEFRONT_LANES == 64u,
	"gfx950 wavefront pin moved - frozen advisory, feeds no gate");
static_assert(SPARK_HW_COMPUTE_CAPABILITY_MAJOR == 9u &&
	SPARK_HW_COMPUTE_CAPABILITY_MINOR == 5u,
	"gfx950 capability pin moved from the spec/9.5 guard agreement");

static_assert(SPARK_HW_MULTIPROCESSOR_COUNT == 0u,
	"CU count must stay unpinned until measured on hardware");
static_assert(SPARK_HW_STATIC_SHARED_LIMIT_BYTES == 0u,
	"LDS static limit must stay unpinned (never import GB10 numbers)");
static_assert(SPARK_HW_MAX_DYNAMIC_SHARED_BYTES == 0ull,
	"LDS opt-in ceiling must stay unpinned (never import 101376)");
static_assert(SPARK_HW_SHARED_PER_SM_BYTES == 0u,
	"per-CU shared total must stay unpinned until measured");
static_assert(SPARK_HW_L2_CACHE_BYTES == 0ull, "L2 must stay unpinned");
static_assert(SPARK_HW_MAX_CLOCK_KHZ == 0u, "clock must stay unpinned");

/* Sentinel helper agrees with the raw values on both sides of the split. */
static_assert(SPARK_HW_VALUE_IS_PINNED(SPARK_HW_WAVEFRONT_LANES),
	"wavefront lanes are pinned and must report pinned");
static_assert(!SPARK_HW_VALUE_IS_PINNED(SPARK_HW_MAX_DYNAMIC_SHARED_BYTES),
	"dynamic-shared ceiling is unpinned and must report so");

int main(void)
{
	if ( SPARK_HW_TARGET_ID != SPARK_HW_TARGET_ROCM_GFX950_MI350P )
		return(1);
	puts("PASS hardware topology performance pins (mi350p: capability 9.5 "
		"pinned, advisories still sentinel-zero)");
	return(0);
}
