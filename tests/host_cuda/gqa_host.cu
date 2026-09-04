
#include "tests/host_cuda/lm_host_cuda.cuh"

#include <stdio.h>

LmHostDim3 blockIdx, threadIdx, blockDim, gridDim;

#include "inference/kernels/dtype.cuh"

#include "inference/kernels/mma.cuh"
#undef LM_WARP_LANES
#define LM_WARP_LANES LM_HOST_WARP_LANES

#include "inference/kernels/kv.cuh"
#include "inference/kernels/gqa.cuh"

#define KV_HEADS 2u
#define HEAD_DIM 8u
#define VALUE_DIM 4u
#define PAGE_SLOTS 2u
#define HEADS 4u
#define SEQUENCES 2u
#define CONTEXT 4u
#define WINDOW 2u
#define PAGES_PER_SEQUENCE 2u
#define TOTAL_PAGES (SEQUENCES * PAGES_PER_SEQUENCE)
#define THREADS 1u
#define QK_SCALE 0.5f

using HostGqaKv = LmKvGeometry<(KV_HEADS * (HEAD_DIM + VALUE_DIM) * 2u), PAGE_SLOTS, true>;

static uint32_t seed = 24680u;
static float NextRandom(void)
{
	seed = (seed * 1664525u) + 1013904223u;
	return (float)((seed >> 8) & 0xffffu) / 32768.0f - 1.0f;
}

int main(void)
{
	static uint8_t pool[TOTAL_PAGES * PAGE_SLOTS * HostGqaKv::kSlotBytes];
	static uint32_t page_table[SEQUENCES * PAGES_PER_SEQUENCE];
	static uint16_t key_rows[SEQUENCES * CONTEXT * KV_HEADS * HEAD_DIM];
	static uint16_t value_rows[SEQUENCES * CONTEXT * KV_HEADS * VALUE_DIM];
	static uint32_t sequence_of_row[SEQUENCES * CONTEXT];
	static uint32_t position_of_row[SEQUENCES * CONTEXT];
	static uint16_t query[SEQUENCES * HEADS * HEAD_DIM];
	static uint16_t output_full[SEQUENCES * HEADS * VALUE_DIM];
	static uint16_t output_window[SEQUENCES * HEADS * VALUE_DIM];
	static uint32_t context_length[SEQUENCES];
	static uint32_t query_sequence[SEQUENCES];
	static uint32_t window_positions[SEQUENCES * WINDOW];
	static LmKvAccessError access_error;
	uint32_t index, sequence, position;

	page_table[0] = 0u; page_table[1] = 2u;
	page_table[2] = 1u; page_table[3] = 3u;

	printf("KV_HEADS %u HEAD_DIM %u VALUE_DIM %u HEADS %u SEQUENCES %u CONTEXT %u WINDOW %u SLOT %u SCALE %.9g\n",
		KV_HEADS, HEAD_DIM, VALUE_DIM, HEADS, SEQUENCES, CONTEXT, WINDOW,
		(uint32_t)HostGqaKv::kSlotBytes, (double)QK_SCALE);

	for (sequence = 0u; sequence < SEQUENCES; ++sequence)
	{
		context_length[sequence] = CONTEXT;
		query_sequence[sequence] = sequence;
		for (position = 0u; position < CONTEXT; ++position)
		{
			uint32_t row = (sequence * CONTEXT) + position;
			sequence_of_row[row] = sequence;
			position_of_row[row] = position;
			for (index = 0u; index < KV_HEADS * HEAD_DIM; ++index)
				key_rows[(row * KV_HEADS * HEAD_DIM) + index] =
					LmFloatToBf16(NextRandom());
			for (index = 0u; index < KV_HEADS * VALUE_DIM; ++index)
				value_rows[(row * KV_HEADS * VALUE_DIM) + index] =
					LmFloatToBf16(NextRandom());
		}
	}
	for (index = 0u; index < SEQUENCES * HEADS * HEAD_DIM; ++index)
		query[index] = LmFloatToBf16(NextRandom());
	for (sequence = 0u; sequence < SEQUENCES; ++sequence)
		for (index = 0u; index < WINDOW; ++index)
			window_positions[(sequence * WINDOW) + index] =
				(CONTEXT - WINDOW) + index;

	for (index = 0u; index < SEQUENCES * CONTEXT * KV_HEADS * HEAD_DIM; ++index)
		printf("key %.9g\n", (double)LmBf16ToFloat(key_rows[index]));
	for (index = 0u; index < SEQUENCES * CONTEXT * KV_HEADS * VALUE_DIM; ++index)
		printf("value %.9g\n", (double)LmBf16ToFloat(value_rows[index]));
	for (index = 0u; index < SEQUENCES * HEADS * HEAD_DIM; ++index)
		printf("query %.9g\n", (double)LmBf16ToFloat(query[index]));

	LmKvView view;
	LmKvAccessErrorReset(&access_error);
	view.pool = pool;
	view.page_table = page_table;
	view.page_table_stride = PAGES_PER_SEQUENCE;
	view.sequence_count = SEQUENCES;
	view.pool_page_count = TOTAL_PAGES;
	view.access_error = &access_error;

	LM_HOST_LAUNCH(dim3(SEQUENCES * CONTEXT),
		(LmGqaKvStoreKernel<HostGqaKv, THREADS, KV_HEADS, HEAD_DIM, VALUE_DIM>(
			view, key_rows, value_rows, sequence_of_row, position_of_row,
			SEQUENCES * CONTEXT)));

	LM_HOST_LAUNCH(dim3(SEQUENCES, HEADS),
		(LmGqaAttentionDecodeKernel<HostGqaKv, THREADS, KV_HEADS, HEAD_DIM, VALUE_DIM>(
			query, view, query_sequence, context_length,
			0, 0u, HEADS, QK_SCALE, output_full, 0)));

	LM_HOST_LAUNCH(dim3(SEQUENCES, HEADS),
		(LmGqaAttentionDecodeKernel<HostGqaKv, THREADS, KV_HEADS, HEAD_DIM, VALUE_DIM>(
			query, view, query_sequence, context_length,
			window_positions, WINDOW, HEADS, QK_SCALE, output_window, 0)));

	for (index = 0u; index < SEQUENCES * HEADS * VALUE_DIM; ++index)
		printf("out_full %.9g\n", (double)LmBf16ToFloat(output_full[index]));
	for (index = 0u; index < SEQUENCES * HEADS * VALUE_DIM; ++index)
		printf("out_window %.9g\n", (double)LmBf16ToFloat(output_window[index]));
	return 0;
}
