#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "spark_dsv4_pool_layout.h"

#define SPARK_DSV4_TEST_RESIDENT 128u
#define SPARK_DSV4_TEST_MAX_SEQUENCE_POSITIONS 4096u

typedef struct SparkDsv4TestPoolLayout
{
	uint64_t cache_offsets[SPARK_DSV4_MODEL_LAYER_COUNT];
	uint64_t cache_strides[SPARK_DSV4_MODEL_LAYER_COUNT];
	uint64_t state_offsets[SPARK_DSV4_MODEL_LAYER_COUNT];
	uint64_t state_strides[SPARK_DSV4_MODEL_LAYER_COUNT];
	uint64_t cache_elements;
	uint64_t state_elements;
} SparkDsv4TestPoolLayout;

static void SparkDsv4TestBuildLayout(uint32_t first_layer,uint32_t layer_count,SparkDsv4TestPoolLayout *layout)
{
	assert(SparkDsv4PoolBuildLayout(first_layer,layer_count,SPARK_DSV4_TEST_RESIDENT,SPARK_DSV4_TEST_MAX_SEQUENCE_POSITIONS,layout->cache_offsets,layout->cache_strides,layout->state_offsets,layout->state_strides,&layout->cache_elements,&layout->state_elements) == 0);
}

static void SparkDsv4TestAssertBounds(uint32_t first_layer,uint32_t layer_count,const SparkDsv4TestPoolLayout *layout)
{
	uint64_t cache_cursor = 0u,state_cursor = 0u,lane_offset;
	uint32_t kind,layer;
	for (layer=first_layer; layer<first_layer+layer_count; layer++)
	{
		kind = SparkDsv4ModelLayerKind(layer);
		assert(layout->cache_offsets[layer] == cache_cursor);
		lane_offset = SparkDsv4PoolLaneOffset(layout->cache_offsets[layer],layout->cache_strides[layer],SPARK_DSV4_TEST_RESIDENT - 1u);
		cache_cursor += ((uint64_t)SPARK_DSV4_TEST_RESIDENT * layout->cache_strides[layer]);
		assert(lane_offset + layout->cache_strides[layer] == cache_cursor);
		if ( kind == SPARK_DSV4_MODEL_LAYER_KIND_SWA )
		{
			assert(layout->state_offsets[layer] == 0u);
			assert(layout->state_strides[layer] == 0u);
			continue;
		}
		assert(layout->state_offsets[layer] == state_cursor);
		lane_offset = SparkDsv4PoolLaneOffset(layout->state_offsets[layer],layout->state_strides[layer],SPARK_DSV4_TEST_RESIDENT - 1u);
		state_cursor += ((uint64_t)SPARK_DSV4_TEST_RESIDENT * layout->state_strides[layer]);
		assert(lane_offset + layout->state_strides[layer] == state_cursor);
	}
	assert(cache_cursor == layout->cache_elements);
	assert(state_cursor == layout->state_elements);
}

static void SparkDsv4TestMixedLayout(void)
{
	SparkDsv4TestPoolLayout layout;
	SparkDsv4TestBuildLayout(0u,4u,&layout);
	SparkDsv4TestAssertBounds(0u,4u,&layout);
	assert(layout.cache_offsets[0] == UINT64_C(0));
	assert(layout.cache_offsets[1] == UINT64_C(8388608));
	assert(layout.cache_offsets[2] == UINT64_C(16777216));
	assert(layout.cache_offsets[3] == UINT64_C(92274688));
	assert(layout.cache_strides[0] == UINT64_C(65536));
	assert(layout.cache_strides[2] == UINT64_C(589824));
	assert(layout.cache_strides[3] == UINT64_C(81920));
	assert(layout.state_offsets[2] == UINT64_C(0));
	assert(layout.state_offsets[3] == UINT64_C(1048576));
	assert(layout.state_strides[2] == UINT64_C(8192));
	assert(layout.state_strides[3] == UINT64_C(65536));
	assert(layout.cache_elements == UINT64_C(102760448));
	assert(layout.state_elements == UINT64_C(9437184));
}

static uint64_t SparkDsv4TestStageBytes(uint32_t first_layer,uint32_t layer_count)
{
	SparkDsv4TestPoolLayout layout;
	uint32_t csa_count = 0u,layer;
	SparkDsv4TestBuildLayout(first_layer,layer_count,&layout);
	SparkDsv4TestAssertBounds(first_layer,layer_count,&layout);
	for (layer=first_layer; layer<first_layer+layer_count; layer++)
		csa_count += SparkDsv4ModelLayerKind(layer) == SPARK_DSV4_MODEL_LAYER_KIND_CSA ? 1u : 0u;
	return(SparkDsv4PoolResidentStateBytes(layout.cache_elements,layout.state_elements,csa_count,SPARK_DSV4_TEST_RESIDENT,SPARK_DSV4_TEST_MAX_SEQUENCE_POSITIONS));
}

static void SparkDsv4TestStagePartition(void)
{
	const uint32_t layer_counts[13] = {3u,3u,3u,3u,3u,3u,3u,4u,4u,4u,4u,4u,2u};
	uint64_t bytes,max_bytes = 0u;
	uint32_t first_layer = 0u,stage;
	for (stage=0u; stage<13u; stage++)
	{
		bytes = SparkDsv4TestStageBytes(first_layer,layer_counts[stage]);
		if ( stage >= 7u && stage <= 11u )
			assert(bytes == UINT64_C(566231040));
		if ( bytes > max_bytes )
			max_bytes = bytes;
		first_layer += layer_counts[stage];
	}
	assert(first_layer == SPARK_DSV4_MODEL_LAYER_COUNT);
	assert(max_bytes == UINT64_C(566231040));
}

static void SparkDsv4TestIndexerLaneBounds(void)
{
	uint64_t cache_stride,state_stride,last_lane;
	cache_stride = SparkDsv4PoolIndexCacheLaneElements(SPARK_DSV4_TEST_MAX_SEQUENCE_POSITIONS);
	state_stride = SparkDsv4PoolIndexStateLaneElements();
	assert(cache_stride == UINT64_C(131072));
	assert(state_stride == UINT64_C(2048));
	last_lane = SparkDsv4PoolLaneOffset(0u,cache_stride,SPARK_DSV4_TEST_RESIDENT - 1u);
	assert(last_lane + cache_stride == (uint64_t)SPARK_DSV4_TEST_RESIDENT * cache_stride);
	last_lane = SparkDsv4PoolLaneOffset(0u,state_stride,SPARK_DSV4_TEST_RESIDENT - 1u);
	assert(last_lane + state_stride == (uint64_t)SPARK_DSV4_TEST_RESIDENT * state_stride);
}

static void SparkDsv4TestPagedLayoutIsOneBackingRecordPerPage(void)
{
	SparkDsv4PagedPoolLayout layout;
	const SparkDsv4PagedLayerLayout *swa,*csa,*hca;
	assert(SparkDsv4PagedPoolBuildLayout(0u,4u,&layout) == 0);
	assert(layout.block_token_count == 128u);
	assert(layout.page_stride_bytes != 0u);
	assert(layout.page_stride_bytes % SPARK_DSV4_PAGED_POOL_ALIGNMENT_BYTES == 0u);
	swa = &layout.layers[0u];
	csa = &layout.layers[2u];
	hca = &layout.layers[3u];
	assert(swa->attention_entry_capacity == 128u);
	assert(swa->compressor_kv_offset_bytes == 0u);
	assert(csa->attention_entry_capacity == 160u);
	assert(csa->index_entry_capacity == 32u);
	assert(csa->compressor_kv_offset_bytes != 0u);
	assert(csa->index_cache_offset_bytes != 0u);
	assert(hca->attention_entry_capacity == 129u);
	assert(hca->index_entry_capacity == 0u);
	assert(hca->compressor_kv_offset_bytes != 0u);
	assert(hca->index_cache_offset_bytes == 0u);
	assert(hca->compressor_score_offset_bytes < layout.page_stride_bytes);
}

int main(void)
{
	SparkDsv4TestMixedLayout();
	SparkDsv4TestStagePartition();
	SparkDsv4TestIndexerLaneBounds();
	SparkDsv4TestPagedLayoutIsOneBackingRecordPerPage();
	return(0);
}
