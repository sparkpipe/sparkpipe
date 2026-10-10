#pragma once

#include "inference/kernels/skinny.cuh"

static int32_t LmCellMmaExperts(const void *weight, const uint16_t *activation, uint16_t *output_bf16, const uint32_t *group_row_offset, const uint32_t *route_source_token, uint32_t groups, uint32_t packed_rows, uint32_t activation_packed, uint32_t input_dimension, uint32_t output_dimension, uint32_t tile_k, cudaStream_t stream)
{
	return(LmSkinnyCellExperts(weight,activation,output_bf16,group_row_offset,route_source_token,groups,packed_rows,packed_rows,activation_packed,input_dimension,output_dimension,tile_k,stream));
}
