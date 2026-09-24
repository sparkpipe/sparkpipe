#pragma once

static __global__ void SPARK_FAMILY(ResidualAddKernel)(void *hidden_bf16, const void *delta_bf16, uint32_t row_count, uint32_t dimension)
{
	uint64_t pair = ((uint64_t)blockIdx.x * blockDim.x) + threadIdx.x,pair_count = ((uint64_t)row_count * dimension) >> 1u;
	float2 hidden_pair,delta_pair;
	if ( pair >= pair_count )
		return;
	hidden_pair = SparkLmLoadBf16Pair(hidden_bf16,pair);
	delta_pair = SparkLmLoadBf16Pair(delta_bf16,pair);
	SparkLmStoreBf16Pair(hidden_bf16,pair,hidden_pair.x + delta_pair.x,hidden_pair.y + delta_pair.y);
	if ( pair == 0u && (((uint64_t)row_count * dimension) & 1u) != 0u )
		SparkLmFloatToBf16(hidden_bf16,((uint64_t)row_count * dimension) - 1u,SparkLmBf16ToFloat(hidden_bf16,((uint64_t)row_count * dimension) - 1u) + SparkLmBf16ToFloat(delta_bf16,((uint64_t)row_count * dimension) - 1u));
}

extern "C" cudaError_t SPARK_FAMILY(LaunchResidualAdd)(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, uint32_t row_count, uint32_t dimension)
{
	uint64_t pairs = ((uint64_t)row_count * dimension + 1u) >> 1u;
	SPARK_FAMILY(ResidualAddKernel)<<<(uint32_t)((pairs + SPARK_LM_CTA_THREADS - 1u) / SPARK_LM_CTA_THREADS),SPARK_LM_CTA_THREADS,0,stream>>>(hidden_bf16,delta_bf16,row_count,dimension);
	return(cudaGetLastError());
}

extern "C" cudaError_t SPARK_FAMILY(LaunchHeadArgmax)(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint32_t *token_ids, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count)
{
	SparkLmHeadArgmaxKernel<<<row_count,SPARK_LM_CTA_THREADS,0,stream>>>(hidden_bf16,head_weight_bf16,token_ids,output_token_ids,row_count,SPARK_FAMILY_CONST(MODEL_HIDDEN_DIMENSION),candidate_count);
	return(cudaGetLastError());
}
