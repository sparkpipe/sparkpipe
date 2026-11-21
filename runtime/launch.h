#pragma once


#include "inference/kernels/layout.cuh"
#include "sparkpipe/spark_error_site.h"
#include <stdint.h>
#include <stdio.h>

#define LM_UNPAREN(...) __VA_ARGS__

#define LM_LAUNCH_OK 0
#define LM_LAUNCH_ERR_SHAPE (-41)
#define LM_LAUNCH_ERR_TILE (-42)
#define LM_LAUNCH_ERR_SHARED (-43)
#define LM_LAUNCH_ERR_MAP (-44)
#define LM_LAUNCH_ERR_ATTRIBUTE (-45)
#define LM_LAUNCH_ERR_LAUNCH (-46)
#define LM_LAUNCH_ERR_OUTPUT (-47)

#define LM_LAUNCH_REFUSE(...) \
	do { (void)fprintf(stderr, __VA_ARGS__); SPARK_FAIL(LM_LAUNCH_ERR_SHAPE); } while (0)

#define LM_GPU_REFUSE(...) \
	do { (void)fprintf(stderr, __VA_ARGS__); SPARK_FAIL(SPARK_GPU_ERROR_INVALID_VALUE); } while (0)

#define LM_LAUNCH_GRID_Y_MAX 65535u

static inline uint32_t LmLaunchGridRows(uint32_t rows)
{
	return(rows < LM_LAUNCH_GRID_Y_MAX ? rows : LM_LAUNCH_GRID_Y_MAX);
}

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
	uint32_t k_tail;
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
	if ( shape->k_tail == 0u && (shape->input_dimension % shape->tile_k) != 0u )
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

#ifdef SPARKPIPE_DEVICE_NVRM
#include "runtime/device/nvrm/launch_nvrm.h"
#else
#include "runtime/device/cuda/launch_cuda.h"
#endif
