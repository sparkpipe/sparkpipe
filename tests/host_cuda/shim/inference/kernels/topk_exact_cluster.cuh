#pragma once

static inline uint32_t LmTopkExactClusterFits(uint32_t rows, uint32_t n, uint32_t k)
{
	(void)rows;
	(void)n;
	(void)k;
	return(0u);
}

template<uint32_t THREADS>
static inline cudaError_t LmTopkExactRowsLaunch(const float *scores, uint32_t rows, uint32_t n, uint32_t k, uint32_t *out_indices, cudaStream_t stream,
	const uint32_t *sequence_of_row = 0, const uint32_t *context_length = 0, const uint32_t *row_position = 0, uint32_t dense_limit = 0u, uint32_t pool_stride = 0u)
{
	(void)sequence_of_row;
	(void)context_length;
	(void)row_position;
	(void)dense_limit;
	(void)pool_stride;
	LM_LAUNCH((LmTopkExactKernel<THREADS>),rows,THREADS,0,stream,scores,n,k,out_indices);
	return(cudaPeekAtLastError());
}
