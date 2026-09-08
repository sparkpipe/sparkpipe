#include "tests/host_cuda/lm_host_cuda.cuh"
#include <stdio.h>
LmHostDim3 blockIdx,threadIdx,blockDim,gridDim;
#include "modules/glm5_next_resident_decode_stage/source/cuda/index_kv.cuh"

int main(void)
{
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
	puts("PASS GLM index KV: every sequence and page stays in its layer slab");
	return(0);
}
