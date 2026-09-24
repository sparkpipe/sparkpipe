#pragma once

static int32_t SPARK_FAMILY(CudaStatus)(cudaError_t status)
{
	return(status == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

static __device__ __forceinline__ uint32_t SPARK_FAMILY(OrderedHeadScore)(float score)
{
	uint32_t bits;
	if ( isnan(score) )
		return(0u);
	bits = __float_as_uint(score);
	return(bits ^ ((bits & UINT32_C(0x80000000)) != 0u ? UINT32_MAX : UINT32_C(0x80000000)));
}

static __global__ void SPARK_FAMILY(HeadMaxlocPackKernel)(
	const float *scores,
	const uint32_t *token_ids,
	uint64_t *maxloc,
	uint32_t row_count,
	uint32_t rank_offset)
{
	uint32_t row;
	row = blockIdx.x * blockDim.x + threadIdx.x;
	if ( row < row_count )
		maxloc[row] = ((uint64_t)SPARK_FAMILY(OrderedHeadScore)(scores[row]) << 32u) |
			(UINT32_MAX - (token_ids[row] + rank_offset));
}
