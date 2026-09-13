#pragma once
// launch.h carries the plan machinery the real GEMM needs; the recorder does
// not, and the error codes it defines live in the gemm shim so a caller sees
// exactly one definition.
#include <stdint.h>

// THE REAL CODES, COPIED, NOT CHOSEN. runtime/launch.h assigns these and a
// harness that made up its own would let a kernel return -1 where the tree
// means -41 and call it the same failure.
#define LM_LAUNCH_OK 0
#define LM_LAUNCH_ERR_SHAPE (-41)
#define LM_LAUNCH_ERR_TILE (-42)
#define LM_LAUNCH_ERR_SHARED (-43)
#define LM_LAUNCH_ERR_MAP (-44)
#define LM_LAUNCH_ERR_ATTRIBUTE (-45)
#define LM_LAUNCH_ERR_LAUNCH (-46)

// THE REAL CHOICE, COPIED. The layer prices its expert tile tables with the
// planner's tile height; the shim must return the same number the real
// launch.h would or the recorded tables lie about the launches they price.
static uint32_t LmLaunchGroupedTileM(uint32_t tokens, uint32_t top_k, uint32_t expert_count)
{
	uint64_t mean;
	uint32_t peak;
	if ( expert_count == 0u )
		return(16u);
	mean = (((uint64_t)tokens * top_k) + expert_count - 1u) / expert_count;
	peak = (uint32_t)(mean * 2u);
	if ( peak <= 16u )
		return(16u);
	if ( peak <= 32u )
		return(32u);
	return(64u);
}

// THE REAL TILE-K FALLBACK, COPIED. LmGemmLaunchTileK (the GEMM-008 shim)
// selects TILE_K from input_dimension exactly as the real launch.h does, so
// the recorded launch matches the dispatch the serving tier would make.
static inline uint32_t LmGemmSelectTileK(uint32_t preferred_tile_k, uint32_t input_dimension)
{
	if ( (input_dimension % preferred_tile_k) == 0u )
		return(preferred_tile_k);
	if ( (input_dimension % 32u) == 0u )
		return(32u);
	return(0u);
}

// THE HOST NO-OP, COPIED. The real launch.h defines this under both branches:
// the device one grants dynamic shared past 48 KiB, the host one has no launch
// to attribute and returns OK. The K3 layer calls it through K3DeltaRuleOptIn,
// so without it here the harness never compiled the layer it exists to run.
static int32_t LmKernelSharedMemoryOptIn(const void *kernel, uint32_t shared_bytes)
{
	(void)kernel;
	(void)shared_bytes;
	return(LM_LAUNCH_OK);
}
