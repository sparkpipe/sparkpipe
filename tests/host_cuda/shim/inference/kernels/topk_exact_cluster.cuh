#pragma once

static inline uint32_t LmTopkExactClusterFits(uint32_t rows, uint32_t n, uint32_t k)
{
	(void)rows;
	(void)n;
	(void)k;
	return(0u);
}

template<uint32_t THREADS>
static inline cudaError_t LmTopkExactRowsLaunch(const float *scores, uint32_t rows, uint32_t n, uint32_t k, uint32_t *out_indices, cudaStream_t stream)
{
	LM_LAUNCH((LmTopkExactKernel<THREADS>),rows,THREADS,0,stream,scores,n,k,out_indices);
	return(cudaPeekAtLastError());
}
