#include "tests/host_cuda/lm_host_cuda.cuh"
#include <stdio.h>
#include <assert.h>
LmHostDim3 blockIdx,threadIdx,blockDim,gridDim;
#include "modules/glm5_next_resident_decode_stage/source/cuda/index_kv.cuh"

#include "modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_internal.h"
#include "glm_kv_view_body.h"
#include "inference/kernels/formats/bf16.cuh"
#undef LM_WARP_LANES
#define LM_WARP_LANES 1u
#include "inference/kernels/norm.cuh"
#include "sparkpipe/spark_glm5_next_index_cp.h"
#include "glm_pool_kernels.h"

static void check_causal_pools(void)
{
	constexpr uint32_t dim = 128u;
	uint16_t pool[64u * (2u * dim + 1u)] = {},query[2u * dim],weight[2] = {0x3f80u,0x3f80u};
	uint32_t table[1] = {0u},sequence[2] = {0u,0u},positions[2] = {0u,2u},context[1] = {4u};
	uint32_t selected[4] = {0u,1u,0u,1u},expanded[12];
	float ape[2u * dim] = {},scores[4],reference[4];
	LmKvAccessError error = {};
	LmKvView view = {};
	view.pool = (uint8_t *)pool;
	view.access_error = &error;
	view.page_table = table;
	view.page_table_stride = view.sequence_count = view.pool_page_count = 1u;
	for (uint32_t i=0u; i<2u * dim; i++) query[i] = 0x3f80u;
	for (uint32_t pos=0u; pos<4u; pos++)
		for (uint32_t i=0u; i<dim; i++) pool[pos * (2u * dim + 1u) + i] = LmFloatToBf16((float)(pos + 1u));
	LM_LAUNCH((Glm5NextPoolScoreKernel<1u,dim,2u,1u>),dim3(2u,2u),1u,0,0,query,weight,view,sequence,context,positions,ape,2u,0u,1u,2u,1.0f,1.0f,scores);
	for (uint32_t row=0u; row<2u; row++)
	{
		context[0] = positions[row] + 1u;
		LM_LAUNCH((Glm5NextPoolScoreKernel<1u,dim,2u,1u>),dim3(2u,1u),1u,0,0,query+row*dim,weight+row,view,sequence,context,positions+row,ape,2u,0u,1u,2u,1.0f,1.0f,reference+row*2u);
	}
	assert(scores[0] == 128.0f && scores[1] == -INFINITY && scores[2] == 192.0f && scores[3] == 384.0f);
	assert(memcmp(scores,reference,sizeof(scores)) == 0);
	context[0] = 4u;
	LM_LAUNCH((Glm5NextPoolExpandKernel<1u,2u,4u,6u>),dim3(2u),1u,0,0,selected,sequence,context,positions,expanded,2u);
	for (uint32_t row=0u; row<2u; row++)
		for (uint32_t i=0u; i<6u; i++)
			assert(expanded[row*6u+i] == UINT32_MAX || expanded[row*6u+i] <= positions[row]);
	assert(expanded[0] == 0u && expanded[1] == UINT32_MAX && expanded[6+2] == 2u);
}

static void check_context_parallel_pools(void)
{
	constexpr uint32_t dim = 128u,context_tokens = 150u,pools = context_tokens / 2u;
	static uint16_t pool[3u * 64u * (2u * dim + 1u)];
	uint16_t query[dim],weight[1] = {0x3f80u};
	uint32_t table[3] = {2u,0u,1u},sequence[1] = {0u},positions[1] = {context_tokens - 1u},context[1] = {context_tokens};
	float ape[2u * dim],reference[pools],local[16u * 64u],gathered[16u * 64u],permuted[pools];
	uint32_t degree,rank,stride,state = 7u;
	LmKvAccessError error = {};
	LmKvView view = {};
	view.pool = (uint8_t *)pool;
	view.access_error = &error;
	view.page_table = table;
	view.page_table_stride = 3u;
	view.sequence_count = 1u;
	view.pool_page_count = 3u;
	for (uint32_t i=0u; i<sizeof(pool) / sizeof(pool[0]); i++) { state = state * 1664525u + 1013904223u; pool[i] = LmFloatToBf16((float)((int32_t)(state >> 20u) - 2048) / 2048.0f); }
	for (uint32_t i=0u; i<dim; i++) { state = state * 1664525u + 1013904223u; query[i] = LmFloatToBf16((float)((int32_t)(state >> 20u) - 2048) / 2048.0f); }
	for (uint32_t i=0u; i<2u * dim; i++) ape[i] = (float)(i % 7u) / 64.0f;
	LM_LAUNCH((Glm5NextPoolScoreKernel<1u,dim,2u,1u>),dim3(pools,1u),1u,0,0,query,weight,view,sequence,context,positions,ape,pools,0u,1u,pools,1.0f,1.0f,reference);
	for (degree=2u; degree<=16u; degree++)
	{
		stride = SparkGlm5NextIndexCpLocalStride(pools,degree);
		assert(stride <= 64u);
		for (rank=0u; rank<degree; rank++)
		{
			LM_LAUNCH((Glm5NextPoolScoreKernel<1u,dim,2u,1u>),dim3(stride,1u),1u,0,0,query,weight,view,sequence,context,positions,ape,pools,rank,degree,stride,1.0f,1.0f,local);
			memcpy(gathered + rank * stride,local,stride * sizeof(float));
		}
		LM_LAUNCH((Glm5NextPoolPermuteKernel<1u>),dim3(1u),1u,0,0,gathered,permuted,pools,stride,(uint64_t)stride,degree);
		assert(memcmp(reference,permuted,sizeof(reference)) == 0);
	}
	assert(error.error_code == LM_KV_ACCESS_ERROR_NONE);
}

int main(void)
{
	check_causal_pools();
	check_context_parallel_pools();
	constexpr uint32_t sequences = 3u,pages_per_sequence = 2u;
	constexpr uint32_t page_count = (sequences * pages_per_sequence);
	constexpr uint32_t slot_bytes = (SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u);
	constexpr uint64_t layer_bytes = ((uint64_t)page_count * 64u * slot_bytes);
	static uint8_t pool[SPARK_GLM5_NEXT_MODEL_DSA_LAYER_COUNT * layer_bytes];
	uint32_t table[page_count] = {0u,1u,2u,3u,4u,5u};
	uint32_t layer,sequence,position;
	uint64_t expected;
	uint8_t *address;
	LmKvView view = {};
	LmKvAccessError error = {};
	view.access_error = &error;
	view.page_table = table;
	view.page_table_stride = pages_per_sequence;
	view.sequence_count = sequences;
	view.pool_page_count = page_count;
	for (layer=0u; layer<SPARK_GLM5_NEXT_MODEL_DSA_LAYER_COUNT; layer++)
	{
		view.pool = pool + (layer * layer_bytes);
		for (sequence=0u; sequence<sequences; sequence++)
			for (position=0u; position<128u; position++)
			{
				expected = (((uint64_t)sequence * 128u + position) * slot_bytes);
				address = LmKvSlotMutableRequired<Glm5NextIndexKv>(view,sequence,position,sequence);
				if ( address != view.pool + expected )
				{
					fprintf(stderr,"wrong index address: layer=%u sequence=%u position=%u\n",layer,sequence,position);
					return(1);
				}
				if ( address + slot_bytes > view.pool + layer_bytes )
					return(2);
				address[0] = (uint8_t)(sequence + 1u);
			}
	}
	SparkGlm5NextExecutionSlot slot = {};
	SparkGlm5NextCudaWave wave = {};
	wave.slot = &slot;
	wave.page_table = table;
	wave.pages_per_sequence = pages_per_sequence;
	wave.resident_sequence_capacity = sequences;
	wave.physical_page_count = 2u;
	slot.kv_access_error = (uint32_t *)&error;
	SparkGlm5NextBuildKvView(&view,pool,&wave);
	assert(view.pool_page_count == 2u);
	table[0] = 2u;
	assert(LmKvSlotMutable<Glm5NextIndexKv>(view,0u,0u) == 0);
	table[0] = 1u;
	assert(LmKvSlotMutable<Glm5NextIndexKv>(view,0u,0u) == pool + 64u * slot_bytes);
	puts("PASS GLM index KV: physical bounds, per-row causal pool scores/selection, and context-parallel pool scores equal to replicated for degrees 2-16");
	return(0);
}
