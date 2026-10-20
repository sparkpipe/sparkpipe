#pragma once

#include <string.h>

#include "inference/kernels/skinny.cuh"
#include "runtime/gemm.cuh"

#define LM_MOE_GEMM_TILE_N 128u
#define LM_MOE_GEMM_STAGES 2u
#define LM_MOE_GEMM_WARPS 8u

typedef struct LmLocalExpertRoutes
{
	const uint32_t *group_row_offset;
	const uint32_t *route_source_token;
	uint32_t *tile_prefix_up;
	uint32_t *tile_prefix_down;
	uint32_t rows;
	uint32_t routes;
	uint32_t top_k;
	uint32_t local_experts;
	uint32_t multiprocessors;
} LmLocalExpertRoutes;

template<class Format>
static int32_t LmLocalExpertsUp(const LmLocalExpertRoutes &routes, const void *weight, LmScaleTensor scale, const uint16_t *activation,
	uint16_t *output, uint32_t input_dimension, uint32_t output_dimension, cudaStream_t stream)
{
	int32_t launch = LmSkinnyGroupedExperts<Format>(weight, scale, activation, output, routes.group_row_offset, routes.route_source_token,
		routes.local_experts, routes.routes, 0u, input_dimension, output_dimension, stream);
	if ( launch == LM_LAUNCH_ERR_SHAPE )
	{
		LmGemmArguments gemm;
		memset(&gemm, 0, sizeof(gemm));
		gemm.scale_a = LmScaleTensorNone();
		gemm.scale_b = scale;
		gemm.prefix_built = 1u;
		gemm.group_row_offset = routes.group_row_offset;
		gemm.group_tile_prefix = routes.tile_prefix_up;
		gemm.source_row_map = routes.route_source_token;
		gemm.source_row_count = routes.rows;
		gemm.output_bf16 = output;
		launch = LmGemmWeightOnlyIndirectLaunch<Format,LM_MOE_GEMM_TILE_N,LM_MOE_GEMM_STAGES,LM_MOE_GEMM_WARPS>(&gemm, activation, weight,
			routes.routes, routes.rows, routes.top_k, routes.local_experts, input_dimension, output_dimension, routes.multiprocessors, stream);
	}
	return launch;
}

template<class Format>
static int32_t LmLocalExpertsDown(const LmLocalExpertRoutes &routes, const void *weight, LmScaleTensor scale, const uint16_t *activation,
	uint16_t *output, uint32_t input_dimension, uint32_t output_dimension, cudaStream_t stream)
{
	int32_t launch = LmSkinnyGroupedExperts<Format>(weight, scale, activation, output, routes.group_row_offset, routes.route_source_token,
		routes.local_experts, routes.routes, 1u, input_dimension, output_dimension, stream);
	if ( launch == LM_LAUNCH_ERR_SHAPE )
	{
		LmGemmArguments gemm;
		memset(&gemm, 0, sizeof(gemm));
		gemm.scale_a = LmScaleTensorNone();
		gemm.scale_b = scale;
		gemm.prefix_built = 1u;
		gemm.group_row_offset = routes.group_row_offset;
		gemm.group_tile_prefix = routes.tile_prefix_down;
		gemm.output_bf16 = output;
		launch = LmGemmWeightOnlyLaunch<Format,LM_MOE_GEMM_TILE_N,LM_MOE_GEMM_STAGES,LM_MOE_GEMM_WARPS>(&gemm, activation, weight, routes.routes,
			routes.rows, routes.top_k, routes.local_experts, input_dimension, output_dimension, routes.multiprocessors, true, stream);
	}
	return launch;
}
