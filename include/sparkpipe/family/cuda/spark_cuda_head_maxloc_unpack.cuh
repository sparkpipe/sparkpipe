#pragma once

static __global__ void SPARK_FAMILY(HeadMaxlocUnpackKernel)(
	const uint64_t *maxloc,
	uint32_t *token_ids,
	uint32_t row_count)
{
	uint32_t row;
	row = blockIdx.x * blockDim.x + threadIdx.x;
	if ( row < row_count )
		token_ids[row] = maxloc[row] == UINT64_MAX ? UINT32_MAX :
			UINT32_MAX - (uint32_t)maxloc[row];
}
