/*
 * Performance-pin gate for the hardware-topology constants (GB10 profile).
 *
 * The topology constants are performance contracts, not decoration: the
 * shared-memory limits size every GEMM pipeline in the tree and the warp
 * width drives every reduction shape. This file makes changing them LOUD
 * twice over:
 *
 *   1. Compile time - the static_asserts below carry the expected numbers
 *      as independent literals. Anyone editing
 *      include/sparkpipe/spark_hardware_topology.h without updating this
 *      gate breaks the build right here.
 *   2. Run time - the workspace pipeline selectors turn the shared-memory
 *      budget into concrete stage depths and occupancy counts. The exact
 *      integers are pinned: a different SMEM limit provably changes them.
 *
 * The MI350P profile cannot be compiled into this one binary (target
 * selection is compile-time); its pins live in
 * tests/test_topology_constant_pins.py, which parses both header blocks.
 */
#include "sparkpipe/spark_hardware_topology.h"
#include "inference/kernels/layout.cuh"
#include "runtime/workspace.h"

#include <stdio.h>

#if SPARK_HW_TARGET_ID != SPARK_HW_TARGET_CUDA_SM121_GB10
#error "pins binary builds the GB10 profile only"
#endif

/* ---- compile-time value pins (independent literals, not tautologies) ---- */
static_assert(SPARK_HW_MULTIPROCESSOR_COUNT == 48u,
	"GB10 SM count pin moved - update calibration docs and this gate");
static_assert(SPARK_HW_WAVEFRONT_LANES == 32u,
	"GB10 warp width pin moved - reductions and launches shift everywhere");
static_assert(SPARK_HW_COMPUTE_CAPABILITY_MAJOR == 12u &&
	SPARK_HW_COMPUTE_CAPABILITY_MINOR == 1u,
	"GB10 capability gate pin moved");
static_assert(SPARK_HW_STATIC_SHARED_LIMIT_BYTES == 49152u,
	"static __shared__ ceiling moved - every static smem budget changes");
static_assert(SPARK_HW_MAX_DYNAMIC_SHARED_BYTES == 101376ull,
	"dynamic-shared opt-in ceiling moved - qwen/dsv4 opt-ins re-derive");
static_assert(SPARK_HW_SHARED_PER_SM_BYTES == 131072u,
	"L1/shared per SM moved - pipeline stage selection shifts tree-wide");
static_assert(SPARK_HW_MAX_CLOCK_KHZ == 2550000u,
	"boost clock pin moved - roofline models shift");

/* Consumer aliases must keep pointing at the header, not at fresh copies. */
static_assert(LM_SMEM_STATIC_LIMIT == SPARK_HW_STATIC_SHARED_LIMIT_BYTES,
	"layout.cuh static-smem alias drifted from the topology header");
static_assert(LM_SMEM_SM_TOTAL == SPARK_HW_SHARED_PER_SM_BYTES,
	"layout.cuh per-SM alias drifted from the topology header");
static_assert(LM_WS_SHARED_LIMIT == SPARK_HW_SHARED_PER_SM_BYTES,
	"workspace.h shared-limit alias drifted from the topology header");

static int failures;

static void expect(int condition, const char *message)
{
	if ( !condition )
	{
		failures++;
		printf("FAIL %s\n", message);
		return;
	}
	printf("ok   %s\n", message);
}

int main(void)
{
	uint64_t bytes;
	uint32_t stages,ctas;

	/* Stage arithmetic at a fixed NVFP4 geometry: 2 stages cost 20512 B,
	 * which must fit the pinned 131072 B budget with room for six CTAs.
	 * A smaller SMEM budget flips depth or occupancy - both pinned. */
	bytes = LmWorkspaceshared_bytes(16u,64u,256u,2u,4u);
	expect(bytes == 20512u, "fixed-tile two-stage footprint is 20512 B");
	stages = LmWorkspaceselect_stages(16u,64u,256u,4u,LM_WS_SHARED_LIMIT);
	expect(stages == 2u, "small NVFP4 tile runs two stages under the budget");
	ctas = LmWorkspacectas_per_sm(bytes,LM_WS_SHARED_LIMIT);
	expect(ctas == 6u, "six CTAs per SM at the small tile under 131072 B");

	/* Wide BF16 tile: 2 stages need 73760 B - fits today, would NOT fit a
	 * half-sized budget (the old 64 KB assumption), so this catches the
	 * classic regression of silently shrinking the constant. */
	bytes = LmWorkspaceshared_bytes(128u,16u,256u,2u,8u);
	expect(bytes == 73760u, "wide-BF16 two-stage footprint is 73760 B");
	stages = LmWorkspaceselect_stages(128u,16u,256u,8u,LM_WS_SHARED_LIMIT);
	expect(stages == 2u, "wide-BF16 tile still pipelines two deep");
	ctas = LmWorkspacectas_per_sm(bytes,LM_WS_SHARED_LIMIT);
	expect(ctas == 1u, "one CTA per SM at the wide tile (73760 B leaves no room for a second)");

	/* Deliberate over-budget case: 131104 B is 528 B OVER the pinned
	 * ceiling, so the selector must refuse (stages 0). If someone raises
	 * SMEM_PER_SM past 131104 this flips to 2 and fails here on purpose -
	 * raising a hardware constant is exactly when we want a human look. */
	bytes = LmWorkspaceshared_bytes(240u,16u,256u,2u,8u);
	expect(bytes == 131104u, "over-budget footprint is 131104 B");
	stages = LmWorkspaceselect_stages(240u,16u,256u,8u,LM_WS_SHARED_LIMIT);
	expect(stages == 0u, "selector refuses what the pinned budget cannot hold");

	if ( failures != 0 )
	{
		printf("FAIL %d topology performance pin(s)\n",failures);
		return(1);
	}
	printf("PASS hardware topology performance pins "
		"(smem limits, warp-width consumers, stage economics)\n");
	return(0);
}
