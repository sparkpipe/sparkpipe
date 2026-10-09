#pragma once

#include "inference/kernels/skinny.cuh"
#include "runtime/gemm.cuh"

template<>
int32_t LmSkinnyDense<LmBf16Format>(const void *weight, const uint16_t *activation, uint16_t *output_bf16, float *output_f32, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t output_row_stride, uint32_t output_column_offset, cudaStream_t stream)
{
	LmSkinnyArguments args;
	LmRecordedGemm record;
	uint32_t row, element;
	int32_t status;
	float value;
	memset(&args,0,sizeof(args));
	args.weight = (const uint8_t *)weight;
	args.activation = activation;
	args.output_bf16 = output_bf16;
	args.output_f32 = output_f32;
	args.scale = LmScaleTensorNone();
	args.rows = rows;
	args.input_dimension = input_dimension;
	args.output_dimension = output_dimension;
	args.output_row_stride = output_row_stride != 0u ? output_row_stride : output_dimension;
	args.output_column_offset = output_column_offset;
	status = LmSkinnyValidate<LmBf16Format>(&args);
	if ( status != LM_LAUNCH_OK )
		return(status);
	memset(&record,0,sizeof(record));
	record.activation = activation;
	record.weight = weight;
	record.output = output_f32 != 0 ? (const void *)output_f32 : (const void *)output_bf16;
	record.activation_scale = LmScaleTensorNone();
	record.weight_scale = LmScaleTensorNone();
	record.input_dimension = input_dimension;
	record.output_dimension = output_dimension;
	record.packed_rows = rows;
	record.activation_stored_bits = LmBf16Format::kStoredBits;
	record.weight_stored_bits = LmBf16Format::kStoredBits;
	lm_recorded_gemms.push_back(record);
	value = 0.125f * (float)lm_recorded_gemms.size();
	for ( row = 0u; row < rows; ++row )
		for ( element = 0u; element < output_dimension; ++element )
		{
			uint64_t index = ((uint64_t)row * args.output_row_stride) + args.output_column_offset + element;
			if ( output_f32 != 0 )
				output_f32[index] = value;
			else
				output_bf16[index] = LmFloatToBf16(value);
		}
	(void)stream;
	return(LM_LAUNCH_OK);
}
