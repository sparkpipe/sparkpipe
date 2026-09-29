#pragma once

#include <cuda_runtime.h>
#include "inference/kernels/head_score.cuh"
#include "sparkpipe/spark_glm5_next_model.h"

extern "C" cudaError_t SparkGlm5NextLaunchHeadScore(cudaStream_t stream,const uint16_t *normed_bf16,const void *head_bf16,float *logits,uint32_t rows,uint32_t width,uint32_t id_base,const uint32_t *probe_offsets,const uint32_t *probe_local,float *probe_logits,SparkScoreDumpStats *stats)
{
	cudaError_t error;
	if ( normed_bf16 == 0 || head_bf16 == 0 || logits == 0 || rows == 0u || width < SPARK_SCORE_DUMP_TOP_K || probe_offsets == 0 || probe_local == 0 || probe_logits == 0 || stats == 0 )
		return(cudaErrorInvalidValue);
	LmHeadScoreLogitsKernel<LM_HEAD_SCORE_ROWS_PER_PASS><<<dim3(LmHeadScoreLogitBlocks(width),LmHeadScoreRowPasses(rows)),LM_HEAD_SCORE_THREADS,0,stream>>>(normed_bf16,(const uint16_t *)head_bf16,logits,rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,width);
	error = cudaPeekAtLastError();
	if ( error != cudaSuccess )
		return(error);
	LmHeadScoreRowsKernel<LM_HEAD_SCORE_THREADS><<<rows,LM_HEAD_SCORE_THREADS,0,stream>>>(logits,rows,width,id_base,probe_offsets,probe_local,probe_logits,stats);
	return(cudaPeekAtLastError());
}
