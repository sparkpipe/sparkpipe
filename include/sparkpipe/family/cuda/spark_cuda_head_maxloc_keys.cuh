#pragma once

extern "C" cudaError_t SPARK_FAMILY(LaunchFusedResidualRmsNorm)(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon)
{
    size_t shared_memory_bytes = (size_t)dimension * sizeof(float);

    SparkLmFusedResidualRmsNormKernel<<<row_count, SPARK_LM_CTA_THREADS, shared_memory_bytes, stream>>>(hidden_bf16, delta_bf16, gain_bf16, output_bf16, row_count, dimension, epsilon);
    return cudaGetLastError();
}

static __device__ __forceinline__ uint32_t SPARK_FAMILY(HeadOrderKey)(float score)
{
	uint32_t bits = __float_as_uint(score);
	return((bits & 0x80000000u) != 0u ? ~bits : bits | 0x80000000u);
}

static __global__ void SPARK_FAMILY(HeadMaxLocPackKernel)(const float *scores_f32, const uint32_t *token_ids_u32, uint64_t *keys_u64, uint32_t row_count)
{
	uint32_t row = blockIdx.x;
	if ( row >= row_count )
		return;
	keys_u64[row] = ((uint64_t)SPARK_FAMILY(HeadOrderKey)(scores_f32[row]) << 32u) | (uint64_t)token_ids_u32[row];
}

extern "C" cudaError_t SPARK_FAMILY(LaunchHeadMaxLocPack)(cudaStream_t stream, const float *scores_f32, const uint32_t *token_ids_u32, uint64_t *keys_u64, uint32_t row_count)
{
	SPARK_FAMILY(HeadMaxLocPackKernel)<<<row_count,1u,0,stream>>>(scores_f32,token_ids_u32,keys_u64,row_count);
	return(cudaGetLastError());
}

static __global__ void SPARK_FAMILY(HeadMaxLocUnpackKernel)(const uint64_t *keys_u64, uint32_t *token_ids_u32, uint32_t row_count)
{
	uint32_t row = blockIdx.x;
	if ( row >= row_count )
		return;
	token_ids_u32[row] = (uint32_t)keys_u64[row];
}

extern "C" cudaError_t SPARK_FAMILY(LaunchHeadMaxLocUnpack)(cudaStream_t stream, const uint64_t *keys_u64, uint32_t *token_ids_u32, uint32_t row_count)
{
	SPARK_FAMILY(HeadMaxLocUnpackKernel)<<<row_count,1u,0,stream>>>(keys_u64,token_ids_u32,row_count);
	return(cudaGetLastError());
}
