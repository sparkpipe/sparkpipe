#pragma once
#include <stdint.h>

#define LM_LAUNCH_OK 0
#define LM_LAUNCH_ERR_SHAPE (-41)
#define LM_LAUNCH_ERR_TILE (-42)
#define LM_LAUNCH_ERR_SHARED (-43)
#define LM_LAUNCH_ERR_MAP (-44)
#define LM_LAUNCH_ERR_ATTRIBUTE (-45)
#define LM_LAUNCH_ERR_LAUNCH (-46)

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

static inline uint32_t LmGemmSelectTileK(uint32_t preferred_tile_k, uint32_t input_dimension)
{
	if ( (input_dimension % preferred_tile_k) == 0u )
		return(preferred_tile_k);
	if ( (input_dimension % 32u) == 0u )
		return(32u);
	return(0u);
}

static int32_t LmKernelSharedMemoryOptIn(const void *kernel, uint32_t shared_bytes)
{
	(void)kernel;
	(void)shared_bytes;
	return(LM_LAUNCH_OK);
}
