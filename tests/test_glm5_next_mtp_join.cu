#include <assert.h>
#include <math.h>
#include "modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_cuda.cu"

#define TEST_MTP_ROWS 6u
#define TEST_MTP_SHARD_ROW 5u

template<typename T> static T *TestAllocate(uint64_t count)
{
	T *pointer = 0;
	assert(cudaMallocManaged(&pointer,count * sizeof(T)) == cudaSuccess);
	assert(cudaMemset(pointer,0,count * sizeof(T)) == cudaSuccess);
	assert(cudaDeviceSynchronize() == cudaSuccess);
	return(pointer);
}

static uint16_t TestBf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	return((uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16));
}

static float TestFloat(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16;
	float result;
	memcpy(&result,&bits,sizeof(result));
	return(result);
}

static float TestEmbedValue(uint32_t index)
{
	return(1.0f + (float)(index % 7u) * 0.125f);
}

static float TestHiddenValue(uint32_t index)
{
	return(((index & 1u) != 0u ? -1.0f : 1.0f) * (0.5f + (float)(index % 11u) * 0.0625f));
}

static void TestNormed(const uint16_t *input,float weight,float *output)
{
	double sum = 0.0;
	float inverse;
	uint32_t index;
	for (index=0u; index<GLM5_NEXT_HIDDEN; index++)
		sum += (double)TestFloat(input[index]) * (double)TestFloat(input[index]);
	inverse = (float)(1.0 / sqrt(sum / (double)GLM5_NEXT_HIDDEN + (double)GLM5_NEXT_RMS_EPSILON));
	for (index=0u; index<GLM5_NEXT_HIDDEN; index++)
		output[index] = TestFloat(TestBf16(TestFloat(input[index]) * inverse * weight));
}

typedef struct TestMtpReduce
{
	const uint16_t *remote_bf16;
	uint32_t calls;
} TestMtpReduce;

static SparkStatus TestMtpReduceRows(void *context,uint16_t *rows_bf16,uint32_t rows,uint32_t width)
{
	TestMtpReduce *reduce = (TestMtpReduce *)context;
	uint32_t index;
	assert(rows == 1u && width == GLM5_NEXT_HIDDEN);
	assert(cudaDeviceSynchronize() == cudaSuccess);
	for (index=0u; index<width; index++)
		rows_bf16[index] = TestBf16(TestFloat(rows_bf16[index]) + TestFloat(reduce->remote_bf16[index]));
	reduce->calls++;
	return(SPARK_STATUS_OK);
}

static void TestSelect(uint16_t *eh_proj,uint32_t half)
{
	uint32_t index;
	assert(cudaMemset(eh_proj,0,(uint64_t)GLM5_NEXT_HIDDEN * 2u * GLM5_NEXT_HIDDEN * sizeof(uint16_t)) == cudaSuccess);
	assert(cudaDeviceSynchronize() == cudaSuccess);
	for (index=0u; index<GLM5_NEXT_HIDDEN; index++)
		eh_proj[(uint64_t)index * 2u * GLM5_NEXT_HIDDEN + half * GLM5_NEXT_HIDDEN + index] = 0x3f80u;
}

static void TestExpect(const char *label,const SparkGlm5NextCudaWave *wave,const SparkGlm5NextMtpDraftOps *ops,const uint16_t *hidden,const float *expected)
{
	uint32_t index,bad = 0u;
	float got,want;
	assert(cudaMemset(wave->slot->mtp_hidden_bf16,0,GLM5_NEXT_HIDDEN * sizeof(uint16_t)) == cudaSuccess);
	assert(SparkGlm5NextMtpJoin(wave,ops,hidden,0) == LM_LAUNCH_OK);
	assert(cudaDeviceSynchronize() == cudaSuccess);
	for (index=0u; index<GLM5_NEXT_HIDDEN; index++)
	{
		got = TestFloat(wave->slot->mtp_hidden_bf16[index]);
		want = expected[index];
		if ( fabsf(got - want) > fabsf(want) / 64.0f + 1e-3f )
		{
			if ( bad < 4u )
				printf("%s index=%u got=%g want=%g\n",label,index,(double)got,(double)want);
			bad++;
		}
	}
	printf("%s tp=%u rank=%u mismatches=%u\n",label,wave->tp_degree,wave->tp_rank,bad);
	assert(bad == 0u);
}

static void TestJoin(SparkGlm5NextCudaWave *wave,uint16_t *embedding,uint16_t *eh_proj,const uint16_t *hidden,uint32_t tp_degree,uint32_t rank,uint32_t token_rank)
{
	SparkGlm5NextMtpDraftOps ops;
	TestMtpReduce reduce;
	uint16_t *remote = TestAllocate<uint16_t>(GLM5_NEXT_HIDDEN);
	uint16_t *embed = TestAllocate<uint16_t>(GLM5_NEXT_HIDDEN);
	float *embed_normed = TestAllocate<float>(GLM5_NEXT_HIDDEN);
	float *hidden_normed = TestAllocate<float>(GLM5_NEXT_HIDDEN);
	uint32_t index,shard = GLM5_NEXT_VOCAB / tp_degree;
	for (index=0u; index<GLM5_NEXT_HIDDEN; index++)
		embed[index] = embedding[(uint64_t)TEST_MTP_SHARD_ROW * GLM5_NEXT_HIDDEN + index];
	TestNormed(embed,0.5f,embed_normed);
	TestNormed(hidden,2.0f,hidden_normed);
	wave->tp_degree = tp_degree;
	wave->tp_rank = rank;
	wave->slot->token_ids[0] = token_rank * shard + TEST_MTP_SHARD_ROW;
	memset(&ops,0,sizeof(ops));
	reduce.remote_bf16 = remote;
	reduce.calls = 0u;
	if ( token_rank != rank )
		for (index=0u; index<GLM5_NEXT_HIDDEN; index++)
			remote[index] = embed[index];
	ops.context = &reduce;
	ops.reduce_rows_bf16 = TestMtpReduceRows;
	TestSelect(eh_proj,0u);
	TestExpect("mtp-join embed-half",wave,tp_degree > 1u ? &ops : 0,hidden,embed_normed);
	TestSelect(eh_proj,1u);
	TestExpect("mtp-join hidden-half",wave,tp_degree > 1u ? &ops : 0,hidden,hidden_normed);
	assert(reduce.calls == (tp_degree > 1u ? 2u : 0u));
	assert(cudaFree(remote) == cudaSuccess && cudaFree(embed) == cudaSuccess);
	assert(cudaFree(embed_normed) == cudaSuccess && cudaFree(hidden_normed) == cudaSuccess);
}

int32_t main(void)
{
	SparkGlm5NextCudaWave wave = {};
	SparkGlm5NextExecutionSlot slot = {};
	uint16_t *embedding = TestAllocate<uint16_t>((uint64_t)TEST_MTP_ROWS * GLM5_NEXT_HIDDEN);
	uint16_t *eh_proj = TestAllocate<uint16_t>((uint64_t)GLM5_NEXT_HIDDEN * 2u * GLM5_NEXT_HIDDEN);
	uint16_t *enorm = TestAllocate<uint16_t>(GLM5_NEXT_HIDDEN);
	uint16_t *hnorm = TestAllocate<uint16_t>(GLM5_NEXT_HIDDEN);
	uint16_t *hidden = TestAllocate<uint16_t>(GLM5_NEXT_HIDDEN);
	int device = 0;
	uint32_t index;
	setvbuf(stdout,0,_IONBF,0);
	slot.token_ids = TestAllocate<uint32_t>(1u);
	slot.hidden_bf16 = TestAllocate<uint16_t>((uint64_t)GLM5_NEXT_HC * GLM5_NEXT_HIDDEN);
	slot.mtp_concat_bf16 = TestAllocate<uint16_t>(2u * GLM5_NEXT_HIDDEN);
	slot.mtp_hidden_bf16 = TestAllocate<uint16_t>(GLM5_NEXT_HIDDEN);
	slot.dense_row_offset = TestAllocate<uint32_t>(2u);
	slot.dense_tile_prefix = TestAllocate<uint32_t>(2u);
	slot.dense_row_offset[1] = 1u;
	for (index=0u; index<GLM5_NEXT_HIDDEN; index++)
	{
		embedding[(uint64_t)TEST_MTP_SHARD_ROW * GLM5_NEXT_HIDDEN + index] = TestBf16(TestEmbedValue(index));
		hidden[index] = TestBf16(TestHiddenValue(index));
		enorm[index] = TestBf16(0.5f);
		hnorm[index] = TestBf16(2.0f);
	}
	assert(cudaGetDevice(&device) == cudaSuccess);
	assert(cudaDeviceGetAttribute((int *)&wave.multiprocessor_count,cudaDevAttrMultiProcessorCount,device) == cudaSuccess);
	wave.slot = &slot;
	wave.embedding_bf16 = embedding;
	wave.mtp_eh_proj_bf16 = eh_proj;
	wave.mtp_enorm_bf16 = enorm;
	wave.mtp_hnorm_bf16 = hnorm;
	TestJoin(&wave,embedding,eh_proj,hidden,1u,0u,0u);
	TestJoin(&wave,embedding,eh_proj,hidden,16u,3u,3u);
	TestJoin(&wave,embedding,eh_proj,hidden,16u,3u,11u);
	printf("PASS mtp join: eh_proj input is [enorm(embed) | hnorm(hidden)] at tp 1 and 16\n");
	return(0);
}
