#pragma once


#include "inference/kernels/layout.cuh"
#include <stdint.h>

#define LM_UNPAREN(...) __VA_ARGS__
#ifdef __CUDACC__
#define LM_LAUNCH(kernel, grid, block, shared, stream, ...) \
	LM_UNPAREN kernel<<<(grid), (block), (shared), (stream)>>>(__VA_ARGS__)
#endif

#define LM_LAUNCH_OK 0
#define LM_LAUNCH_ERR_SHAPE (-41)
#define LM_LAUNCH_ERR_TILE (-42)
#define LM_LAUNCH_ERR_SHARED (-43)
#define LM_LAUNCH_ERR_MAP (-44)
#define LM_LAUNCH_ERR_ATTRIBUTE (-45)
#define LM_LAUNCH_ERR_LAUNCH (-46)
#define LM_LAUNCH_ERR_OUTPUT (-47)

#define LM_LAUNCH_TILE_MIN 16u
#define LM_LAUNCH_TILE_MAX 64u

static inline uint32_t LmGemmSelectTileK(uint32_t preferred_tile_k, uint32_t input_dimension)
{
	if ( (input_dimension % preferred_tile_k) == 0u )
		return(preferred_tile_k);
	if ( (input_dimension % 32u) == 0u )
		return(32u);
	return(0u);
}

typedef struct LmLaunchShape
{
	uint32_t tokens,top_k,expert_count,input_dimension,output_dimension;
	uint32_t stored_bits,stored_bits_a,tile_n,tile_k,stages;
	uint32_t interleaved_b;
}
LmLaunchShape;

typedef struct LmLaunchPlan
{
	uint32_t tile_m;
	uint32_t shared_bytes;
	uint32_t grid_blocks;
	uint32_t block_threads;
	uint64_t workspace_bytes;
	uint32_t swizzle_span;
}
LmLaunchPlan;

static uint32_t LmLaunchPeakRowsPerGroup(const LmLaunchShape *shape)
{
	uint64_t mean;
	if ( shape->expert_count == 0u )
		return(0u);
	mean = (((uint64_t)shape->tokens * shape->top_k) + shape->expert_count - 1u)
		/ shape->expert_count;
	return((uint32_t)(mean * 2u));
}

static uint32_t LmLaunchSelectTile(uint32_t peak_rows)
{
	if ( peak_rows <= 16u )
		return(16u);
	if ( peak_rows <= 32u )
		return(32u);
	return(64u);
}

static inline uint32_t LmLaunchGroupedTileM(uint32_t tokens, uint32_t top_k, uint32_t expert_count)
{
	LmLaunchShape shape = {};
	shape.tokens = tokens;
	shape.top_k = top_k;
	shape.expert_count = expert_count;
	return(LmLaunchSelectTile(LmLaunchPeakRowsPerGroup(&shape)));
}

static uint32_t LmLaunchSharedBytes(const LmLaunchShape *shape, uint32_t tile_m)
{
	uint32_t bits_a = shape->stored_bits_a != 0u ? shape->stored_bits_a : shape->stored_bits;
	uint32_t a = (tile_m * shape->tile_k * bits_a) / 8u;
	uint32_t b = shape->interleaved_b != 0u
		? (17u * (shape->tile_n / 16u) * (shape->tile_k / 2u))
		: (shape->tile_n * shape->tile_k * shape->stored_bits) / 8u;
	return((shape->stages * (a + b)) + (shape->stages * 8u));
}

static int32_t LmLaunchPlanBuild(const LmLaunchShape *shape, uint32_t multiprocessors, LmLaunchPlan *plan)
{
	uint32_t pitch;
	if ( shape == 0 || plan == 0 || shape->expert_count == 0u || shape->tile_n == 0u
		|| shape->tile_k == 0u || shape->stages < 2u || multiprocessors == 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	plan->tile_m = LmLaunchSelectTile(LmLaunchPeakRowsPerGroup(shape));
	if ( plan->tile_m < LM_LAUNCH_TILE_MIN || plan->tile_m > LM_LAUNCH_TILE_MAX )
		return(LM_LAUNCH_ERR_TILE);
	if ( (shape->input_dimension % shape->tile_k) != 0u )
		return(LM_LAUNCH_ERR_SHAPE);
	plan->shared_bytes = LmLaunchSharedBytes(shape,plan->tile_m);
	if ( plan->shared_bytes > LM_SMEM_SM_TOTAL )
		return(LM_LAUNCH_ERR_SHARED);
	pitch = (shape->tile_k * shape->stored_bits) / 8u;
	plan->swizzle_span = LmSwizzleSpanFor(pitch);
	if ( (plan->swizzle_span == 0u || pitch != plan->swizzle_span) &&
		!(shape->interleaved_b != 0u && pitch == 16u) )
		return(LM_LAUNCH_ERR_MAP);
	if ( shape->stored_bits_a != 0u )
	{
		uint32_t activation_pitch =
			(shape->tile_k * shape->stored_bits_a) / 8u;
		uint32_t activation_span = LmSwizzleSpanFor(activation_pitch);
		if ( activation_span == 0u ||
			(shape->interleaved_b == 0u
				? activation_pitch != activation_span
				: (activation_pitch % activation_span) != 0u) )
			return(LM_LAUNCH_ERR_MAP);
	}
	plan->grid_blocks = multiprocessors;
	plan->block_threads = 8u * 32u;
	plan->workspace_bytes = 0u;
	return(LM_LAUNCH_OK);
}

#define LM_LAUNCH_MAX_OPTIN_KERNELS 16u
#define LM_LAUNCH_MAX_TRACKED_DEVICES 64u

#ifdef __CUDACC__
#include <mutex>

typedef struct LmKernelOptInEntry
{
	const void *kernel;
	uint64_t device_mask;
}
LmKernelOptInEntry;

static int32_t LmKernelSharedMemoryOptIn(const void *kernel, uint32_t shared_bytes)
{
	static std::mutex grant_mutex;
	static LmKernelOptInEntry grants[LM_LAUNCH_MAX_OPTIN_KERNELS];
	static uint32_t grant_count = 0u;
	cudaError_t status;
	uint64_t device_bit;
	uint32_t index;
	int device_index;

	if ( kernel == 0 )
		return(LM_LAUNCH_ERR_ATTRIBUTE);
	status = cudaGetDevice(&device_index);
	if ( status != cudaSuccess )
		return(LM_LAUNCH_ERR_ATTRIBUTE);
	if ( device_index < 0
		|| (uint32_t)device_index >= LM_LAUNCH_MAX_TRACKED_DEVICES )
		return(LM_LAUNCH_ERR_ATTRIBUTE);
	device_bit = UINT64_C(1) << (uint32_t)device_index;
	{
		std::lock_guard<std::mutex> lock(grant_mutex);
		for ( index = 0u; index < grant_count; index++ )
		{
			if ( grants[index].kernel == kernel )
			{
				if ( (grants[index].device_mask & device_bit) != 0u )
					return(LM_LAUNCH_OK);
				break;
			}
		}
		if ( index == grant_count )
		{
			if ( grant_count >= LM_LAUNCH_MAX_OPTIN_KERNELS )
				return(LM_LAUNCH_ERR_ATTRIBUTE);
			grants[grant_count].kernel = kernel;
			grants[grant_count].device_mask = 0u;
			grant_count++;
		}
		status = cudaFuncSetAttribute(kernel,
			cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shared_bytes);
		if ( status == cudaSuccess )
			grants[index].device_mask |= device_bit;
	}
	return(status == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_ATTRIBUTE);
}
#else
static int32_t LmKernelSharedMemoryOptIn(const void *kernel, uint32_t shared_bytes)
{
	(void)kernel;
	(void)shared_bytes;
	return(LM_LAUNCH_OK);
}
#endif
