// Run K3's MLA path on a CPU: store into a paged cache, then attend over it.
//
// This is the last of K3's forward that had never executed. It needs more setup
// than the other harnesses because the cache is paged - a page table per
// sequence, a pool, and slots addressed through LmKvSlot - and that addressing
// is exactly what wants testing. LmKvSlot returns null for an unmapped page
// rather than an address into page zero, and the comment on it says why: the
// alternative is reading another sequence's keys and getting output that is
// fluent and wrong.
//
// TWO SEQUENCES, NOT ONE. A single sequence cannot tell correct page
// addressing from ignoring the page table, because with one sequence every
// mapping is the identity. The second sequence's pages are deliberately
// interleaved with the first's so a kernel that indexed by position alone would
// read the wrong tokens.
//
// The geometry is K3's shape at small size: a latent row plus an unrotated
// slice, which is what LmKvLatent describes and what the absorbed attention
// kernel reads as its key.

#include "tests/host_cuda/lm_host_cuda.cuh"

#include <stdio.h>

LmHostDim3 blockIdx, threadIdx, blockDim, gridDim;

uint32_t lm_topk_shared[LM_HOST_SHARED_BYTES / sizeof(uint32_t)];
float lm_norm_shared[LM_HOST_SHARED_BYTES / sizeof(float)];
float lm_fused_shared[LM_HOST_SHARED_BYTES / sizeof(float)];
float lm_quant_shared[LM_HOST_SHARED_BYTES / sizeof(float)];

#include "inference/kernels/dtype.cuh"

#include "inference/kernels/mma.cuh"
#undef LM_WARP_LANES
#define LM_WARP_LANES LM_HOST_WARP_LANES

// kv.cuh guards its store kernel on __CUDACC__ - it is the only such guard in
// the kernel tree, and the effect was that the one kernel responsible for
// WRITING the cache could not be compiled for the host and therefore could not
// be tested. Declaring it here says the harness is a device-code compiler for
// this purpose, which it is.
#define __CUDACC__ 1
#include "inference/kernels/kv.cuh"
#include "inference/kernels/attn.cuh"

#define LATENT 8u
#define ROPE 8u
#define PAGE_SLOTS 2u
#define HEADS 2u
#define SEQUENCES 2u
#define CONTEXT 4u
#define PAGES_PER_SEQUENCE 2u
#define TOTAL_PAGES (SEQUENCES * PAGES_PER_SEQUENCE)
#define THREADS 1u

using HostKv = LmKvLatent<16u, LATENT, ROPE, PAGE_SLOTS>;

static uint32_t seed = 13579u;
static float NextRandom(void)
{
	seed = (seed * 1664525u) + 1013904223u;
	return (float)((seed >> 8) & 0xffffu) / 32768.0f - 1.0f;
}

int main(void)
{
	static uint8_t pool[TOTAL_PAGES * PAGE_SLOTS * HostKv::kSlotBytes];
	static uint32_t page_table[SEQUENCES * PAGES_PER_SEQUENCE];
	static uint16_t slot_rows[SEQUENCES * CONTEXT * (LATENT + ROPE)];
	static uint32_t sequence_of_row[SEQUENCES * CONTEXT];
	static uint32_t position_of_row[SEQUENCES * CONTEXT];
	static uint16_t query[SEQUENCES * HEADS * (LATENT + ROPE)];
	static uint16_t output[SEQUENCES * HEADS * LATENT];
	static uint32_t context_length[SEQUENCES];
	static uint32_t query_sequence[SEQUENCES];
	static LmKvAccessError access_error;
	uint32_t index, sequence, position, head;

	// INTERLEAVED PAGES. Sequence 0 gets pages 0 and 2, sequence 1 gets 1 and 3,
	// so position-only addressing lands in the other sequence's tokens.
	page_table[0] = 0u; page_table[1] = 2u;
	page_table[2] = 1u; page_table[3] = 3u;

	printf("LATENT %u ROPE %u HEADS %u SEQUENCES %u CONTEXT %u SLOT %u\n",
		LATENT, ROPE, HEADS, SEQUENCES, CONTEXT, (uint32_t)HostKv::kSlotBytes);

	for (sequence = 0u; sequence < SEQUENCES; ++sequence)
	{
		context_length[sequence] = CONTEXT;
		query_sequence[sequence] = sequence;
		for (position = 0u; position < CONTEXT; ++position)
		{
			uint32_t row = (sequence * CONTEXT) + position;
			sequence_of_row[row] = sequence;
			position_of_row[row] = position;
			for (index = 0u; index < LATENT + ROPE; ++index)
				slot_rows[(row * (LATENT + ROPE)) + index] =
					LmFloatToBf16(NextRandom());
		}
	}
	for (index = 0u; index < SEQUENCES * HEADS * (LATENT + ROPE); ++index)
		query[index] = LmFloatToBf16(NextRandom());

	for (index = 0u; index < SEQUENCES * CONTEXT * (LATENT + ROPE); ++index)
		printf("slot %.9g\n", (double)LmBf16ToFloat(slot_rows[index]));
	for (index = 0u; index < SEQUENCES * HEADS * (LATENT + ROPE); ++index)
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
		(LmKvStoreKernel<HostKv, THREADS>(
			view, slot_rows, sequence_of_row, position_of_row,
			SEQUENCES * CONTEXT, LATENT + ROPE)));

	LM_HOST_LAUNCH(dim3(SEQUENCES, HEADS),
		(LmAttentionDecodeKernel<HostKv, THREADS, LATENT, ROPE>(
			query, query, view, query_sequence, context_length,
			0, 0u, HEADS, 0.25f, output, 0)));

	for (index = 0u; index < SEQUENCES * HEADS * LATENT; ++index)
		printf("out %.9g\n", (double)LmBf16ToFloat(output[index]));
	return 0;
}
