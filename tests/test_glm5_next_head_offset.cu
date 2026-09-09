#include <assert.h>
#include "modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_cuda.cu"

template<typename T> static T *TestAllocate(uint64_t count)
{
	T *pointer = 0;
	assert(cudaMallocManaged(&pointer,count * sizeof(T)) == cudaSuccess);
	assert(cudaMemset(pointer,0,count * sizeof(T)) == cudaSuccess);
	assert(cudaDeviceSynchronize() == cudaSuccess);
	return(pointer);
}

static void TestHead(SparkGlm5NextCudaWave *wave,uint32_t rank,uint32_t rows)
{
	uint32_t row,expected;
	wave->tp_rank = rank;
	wave->row_count = rows;
	expected = rank * (SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT / 16u) + 7u;
	assert(SparkGlm5NextRunHead(wave) == LM_LAUNCH_OK);
	assert(SparkGlm5NextLaunchHeadMaxlocUnpack(0,wave->slot->head_maxloc_u64,wave->slot->output_token,rows) == cudaSuccess);
	assert(cudaDeviceSynchronize() == cudaSuccess);
	for (row=0u; row<rows; row++)
	{
		printf("rank=%u rows=%u row=%u expected=%u actual=%u\n",rank,rows,row,expected,wave->slot->output_token[row]);
		assert(wave->slot->output_token[row] == expected);
	}
}

int32_t main(void)
{
	SparkGlm5NextCudaWave wave = {};
	SparkGlm5NextExecutionSlot slot = {};
	SparkGlm5NextLayerWeights layer = {};
	uint32_t ordinal = UINT32_MAX,index,rank;
	uint32_t vocabulary = SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT / 16u;
	uint32_t tiles = (vocabulary + GLM5_NEXT_HEAD_TILE - 1u) / GLM5_NEXT_HEAD_TILE;
	uint16_t *head = TestAllocate<uint16_t>((uint64_t)vocabulary * GLM5_NEXT_HIDDEN);
	uint16_t *norm = TestAllocate<uint16_t>(GLM5_NEXT_HIDDEN);
	uint8_t *payload = TestAllocate<uint8_t>((uint64_t)vocabulary * GLM5_NEXT_HIDDEN);
	float *scale = TestAllocate<float>((uint64_t)vocabulary * GLM5_NEXT_HIDDEN / 32u);
	float *bound = TestAllocate<float>((uint64_t)vocabulary * GLM5_NEXT_HIDDEN / 32u);
	slot.hidden_bf16 = TestAllocate<uint16_t>(3u * GLM5_NEXT_HC * GLM5_NEXT_HIDDEN);
	slot.residual_bf16 = TestAllocate<uint16_t>(3u * GLM5_NEXT_HIDDEN);
	slot.hc_mean_bf16 = TestAllocate<uint16_t>(3u * GLM5_NEXT_HIDDEN);
	slot.normed_bf16 = TestAllocate<uint16_t>(3u * GLM5_NEXT_HIDDEN);
	slot.head_candidate_score = TestAllocate<float>(3u * tiles);
	slot.head_candidate_token = TestAllocate<uint32_t>(3u * tiles);
	slot.output_score = TestAllocate<float>(3u);
	slot.output_token = TestAllocate<uint32_t>(3u);
	slot.head_maxloc_u64 = TestAllocate<uint64_t>(3u);
	slot.head_certified_scratch = TestAllocate<uint8_t>(SparkHeadCertifiedFp8ScratchBytes(vocabulary,GLM5_NEXT_HIDDEN));
	slot.head_certified_candidates = TestAllocate<uint32_t>(vocabulary);
	slot.head_screened_count = TestAllocate<uint32_t>(1u);
	for (index=0u; index<GLM5_NEXT_HIDDEN; index++)
		norm[index] = head[7u * GLM5_NEXT_HIDDEN + index] = 0x3f80u;
	for (index=0u; index<3u * GLM5_NEXT_HC * GLM5_NEXT_HIDDEN; index++)
		slot.hidden_bf16[index] = 0x3f80u;
	assert(SparkGlm5NextLaunchHeadCertifiedQuantize(0,head,payload,scale,bound,vocabulary,GLM5_NEXT_HIDDEN) == cudaSuccess);
	wave.slot = &slot;
	wave.layers = &layer;
	wave.layer_count = 1u;
	wave.tp_degree = 16u;
	wave.owns_final_head = 1u;
	wave.kda_ordinal_by_local_layer = wave.index_ordinal_by_local_layer = &ordinal;
	wave.final_norm_bf16 = norm;
	wave.lm_head_bf16 = head;
	wave.head_certified_fp8_payload = payload;
	wave.head_certified_fp8_scale_f32 = scale;
	wave.head_certified_fp8_norm_f32 = bound;
	for (rank=0u; rank<16u; rank++)
	{
		TestHead(&wave,rank,1u);
		TestHead(&wave,rank,3u);
	}
	return(0);
}
