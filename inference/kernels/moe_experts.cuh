#pragma once

#include <string.h>

#include "inference/kernels/cell_mma.cuh"
#include "inference/kernels/skinny.cuh"
#include "runtime/gemm.cuh"
#include "sparkpipe/spark_gpu.h"

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
static uint32_t LmLocalExpertsSkinny(const LmLocalExpertRoutes &routes, uint32_t row_exact)
{
	return row_exact != 0u || routes.routes <= LM_SKINNY_GROUPED_MAX_MEAN_ROWS * routes.local_experts ? 1u : 0u;
}

#define LM_LOCAL_EXPERTS_REFUSE(kernel, routes, input_dimension, output_dimension, needs) \
	LM_LAUNCH_REFUSE("local experts: the %s refused %u routes over %u experts of a %u x %u projection; it needs %s\n", kernel, (routes).routes, \
		(routes).local_experts, output_dimension, input_dimension, needs)

#define LM_LOCAL_EXPERTS_SKINNY_NEEDS "an input that is a multiple of the format's elements, 16-byte aligned weights and activation, and a scale that covers every expert"
#define LM_LOCAL_EXPERTS_CELL_NEEDS "input and output dimensions that are multiples of 32, 16-byte aligned weights and activation, and 4-byte aligned scales whose strides are multiples of 4 entries"

template<class Format>
static int32_t LmLocalExpertsUp(const LmLocalExpertRoutes &routes, const void *weight, LmScaleTensor scale, const uint16_t *activation,
	uint16_t *output, uint32_t input_dimension, uint32_t output_dimension, SparkGpuStream stream, uint32_t row_exact = 0u)
{
	int32_t launch;
	if ( LmLocalExpertsSkinny<Format>(routes, row_exact) != 0u )
	{
		launch = LmSkinnyGroupedExperts<Format>(weight, scale, activation, output, routes.group_row_offset, routes.route_source_token,
			routes.local_experts, routes.routes, 0u, input_dimension, output_dimension, stream, row_exact);
		if ( launch != LM_LAUNCH_ERR_SHAPE )
			return launch;
		LM_LOCAL_EXPERTS_REFUSE("grouped skinny kernel", routes, input_dimension, output_dimension, LM_LOCAL_EXPERTS_SKINNY_NEEDS);
	}
	if constexpr ( LmCellMmaFormat<Format>::kScales != 0u )
	{
		launch = LmCellMmaStridedExperts<Format>(weight, scale, activation, output, routes.group_row_offset, routes.route_source_token, routes.local_experts,
			routes.routes, 0u, input_dimension, output_dimension, stream);
		if ( launch != LM_LAUNCH_ERR_SHAPE )
			return launch;
		LM_LOCAL_EXPERTS_REFUSE("expert cell kernel", routes, input_dimension, output_dimension, LM_LOCAL_EXPERTS_CELL_NEEDS);
	}
	else
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
		return LmGemmWeightOnlyIndirectLaunch<Format,LM_MOE_GEMM_TILE_N,LM_MOE_GEMM_STAGES,LM_MOE_GEMM_WARPS>(&gemm, activation, weight,
			routes.routes, routes.rows, routes.top_k, routes.local_experts, input_dimension, output_dimension, routes.multiprocessors, stream);
	}
}

template<class Format>
static int32_t LmLocalExpertsDown(const LmLocalExpertRoutes &routes, const void *weight, LmScaleTensor scale, const uint16_t *activation,
	uint16_t *output, uint32_t input_dimension, uint32_t output_dimension, SparkGpuStream stream, uint32_t row_exact = 0u)
{
	int32_t launch;
	if ( LmLocalExpertsSkinny<Format>(routes, row_exact) != 0u )
	{
		launch = LmSkinnyGroupedExperts<Format>(weight, scale, activation, output, routes.group_row_offset, routes.route_source_token,
			routes.local_experts, routes.routes, 1u, input_dimension, output_dimension, stream, row_exact);
		if ( launch != LM_LAUNCH_ERR_SHAPE )
			return launch;
		LM_LOCAL_EXPERTS_REFUSE("grouped skinny kernel", routes, input_dimension, output_dimension, LM_LOCAL_EXPERTS_SKINNY_NEEDS);
	}
	if constexpr ( LmCellMmaFormat<Format>::kScales != 0u )
	{
		launch = LmCellMmaStridedExperts<Format>(weight, scale, activation, output, routes.group_row_offset, 0, routes.local_experts, routes.routes, 1u,
			input_dimension, output_dimension, stream);
		if ( launch != LM_LAUNCH_ERR_SHAPE )
			return launch;
		LM_LOCAL_EXPERTS_REFUSE("expert cell kernel", routes, input_dimension, output_dimension, LM_LOCAL_EXPERTS_CELL_NEEDS);
	}
	else
	{
		LmGemmArguments gemm;
		memset(&gemm, 0, sizeof(gemm));
		gemm.scale_a = LmScaleTensorNone();
		gemm.scale_b = scale;
		gemm.prefix_built = 1u;
		gemm.group_row_offset = routes.group_row_offset;
		gemm.group_tile_prefix = routes.tile_prefix_down;
		gemm.output_bf16 = output;
		return LmGemmWeightOnlyLaunch<Format,LM_MOE_GEMM_TILE_N,LM_MOE_GEMM_STAGES,LM_MOE_GEMM_WARPS>(&gemm, activation, weight, routes.routes,
			routes.rows, routes.top_k, routes.local_experts, input_dimension, output_dimension, routes.multiprocessors, true, stream);
	}
}
