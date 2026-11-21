#pragma once

#include <string.h>

#include "inference/kernels/skinny.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "sparkpipe/spark_gpu.h"

template<class Format>
static int32_t LmDenseRows(const void *weight, LmScaleTensor scale, const uint16_t *activation, uint16_t *output_bf16, float *output_f32, uint32_t rows,
	uint32_t input_dimension, uint32_t output_dimension, uint32_t row_exact, uint32_t multiprocessors, SparkGpuStream stream)
{
	if ( rows <= LM_SKINNY_ROWS_WIDE || row_exact != 0u )
	{
		LmSkinnyArguments args;
		int32_t status;
		memset(&args, 0, sizeof(args));
		args.weight = (const uint8_t *)weight;
		args.activation = activation;
		args.output_bf16 = output_bf16;
		args.output_f32 = output_f32;
		args.scale = scale;
		args.rows = rows;
		args.input_dimension = input_dimension;
		args.output_dimension = output_dimension;
		status = LmSkinnyLaunch<Format>(&args, stream, row_exact);
		if ( status != LM_LAUNCH_ERR_SHAPE )
			return status;
		LM_LAUNCH_REFUSE("dense rows: the skinny kernel refused %u rows of a %u x %u projection; it needs an input that is a multiple of %u elements, "
			"16-byte aligned weight and activation, one output and a scale that covers the weight\n", rows, output_dimension, input_dimension,
			LmSkinnyFormat<Format>::kElements);
	}
	return LmStreamGemmDense<Format>(weight, scale, activation, output_bf16, output_f32, rows, input_dimension, output_dimension, 0u, 0u, multiprocessors,
		stream);
}
