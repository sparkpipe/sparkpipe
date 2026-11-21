#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "runtime/launch.h"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/index_score.cuh"

#define SLOTS_PAGE 64u
#define SLOTS_DIM 128u

template<uint32_t SLOT_BYTES>
struct SlotsIndexKv
{
	static constexpr uint32_t kSlotBytes = SLOT_BYTES;
	static constexpr uint32_t kPageSlots = SLOTS_PAGE;
	static constexpr uint32_t kPageBytes = kSlotBytes * SLOTS_PAGE;
	static constexpr bool kGrows = true;
	static __host__ __device__ constexpr uint32_t PageOf(uint32_t position) { return position / SLOTS_PAGE; }
	static __host__ __device__ constexpr uint32_t SlotInPage(uint32_t position) { return position % SLOTS_PAGE; }
	static __host__ __device__ constexpr uint64_t PagesForTokens(uint64_t tokens) { return (tokens + SLOTS_PAGE - 1u) / SLOTS_PAGE; }
};

static int failures;
static uint32_t state_word = 20261118u;

static uint32_t Next(void)
{
	state_word = state_word * 1664525u + 1013904223u;
	return state_word;
}

static uint16_t Bf16(float value)
{
	uint32_t bits;
	memcpy(&bits, &value, 4u);
	return (uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

#define CHECK(condition, what) do { if ( !(condition) ) { printf("FAIL %s\n", what); failures++; } } while ( 0 )

template<class T>
static T *Upload(const std::vector<T> &host)
{
	T *device = 0;
	CHECK(cudaMalloc((void **)&device, host.size() * sizeof(T) + 16u) == cudaSuccess, "allocate");
	CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess, "upload");
	return device;
}

static LmKvView MakeView(const LmKvView &view, LmKvView *)
{
	return view;
}

static LmKvShardView MakeView(const LmKvView &view, LmKvShardView *)
{
	LmKvShardView shard;
	shard.pages = view;
	shard.shard.degree = 1u;
	shard.shard.rank = 0u;
	shard.shard.grain = 8u;
	return shard;
}

static const char *ViewName(LmKvView *)
{
	return "pages";
}

static const char *ViewName(LmKvShardView *)
{
	return "shard";
}

template<uint32_t HEADS, class View, uint32_t SLOT_BYTES>
static void Case(uint32_t rows, uint32_t context, uint32_t mixed, uint32_t stride_extra)
{
	typedef SlotsIndexKv<SLOT_BYTES> Kv;
	const uint32_t sequences = mixed != 0u ? 2u : 1u, pages = (context + SLOTS_PAGE - 1u) / SLOTS_PAGE, stride = context + stride_extra;
	const float scale = 0.0883883f;
	std::vector<uint32_t> table((uint64_t)sequences * pages), sequence_of_row(rows), position(rows), key_sequence((uint64_t)sequences * context),
		key_position((uint64_t)sequences * context), context_length(sequences, context);
	std::vector<uint16_t> keys((uint64_t)sequences * context * SLOTS_DIM), query((uint64_t)rows * HEADS * SLOTS_DIM), weight((uint64_t)rows * HEADS);
	for ( uint32_t index = 0u; index < table.size(); ++index )
		table[index] = (uint32_t)((index * 7u + 3u) % table.size());
	for ( uint32_t s = 0u; s < sequences; ++s )
		for ( uint32_t p = 0u; p < context; ++p )
		{
			key_sequence[(uint64_t)s * context + p] = s;
			key_position[(uint64_t)s * context + p] = p;
		}
	for ( auto &value : keys )
		value = Bf16((float)((int32_t)(Next() >> 20u) - 2048) / 2048.0f);
	for ( auto &value : query )
		value = Bf16((float)((int32_t)(Next() >> 20u) - 2048) / 2048.0f);
	for ( auto &value : weight )
		value = Bf16((float)(Next() >> 24u) / 256.0f);
	for ( uint32_t row = 0u; row < rows; ++row )
	{
		sequence_of_row[row] = mixed != 0u ? (row / 3u) % 2u : 0u;
		position[row] = row % 5u == 0u ? context - 1u : row % 5u == 1u ? (uint32_t)(Next() % (context < 40u ? context : 40u)) : (uint32_t)(Next() % context);
	}
	LmKvAccessError *error = 0;
	CHECK(cudaMallocManaged((void **)&error, sizeof(*error)) == cudaSuccess, "error cell");
	LmKvAccessErrorReset(error);
	uint8_t *pool = 0;
	CHECK(cudaMalloc((void **)&pool, (uint64_t)table.size() * Kv::kPageBytes) == cudaSuccess, "pool");
	uint32_t *d_table = Upload(table), *d_sequence = Upload(sequence_of_row), *d_position = Upload(position), *d_key_sequence = Upload(key_sequence),
		*d_key_position = Upload(key_position), *d_context = Upload(context_length);
	uint16_t *d_keys = Upload(keys), *d_query = Upload(query), *d_weight = Upload(weight);
	float *d_slots = 0, *d_reference = 0;
	CHECK(cudaMalloc((void **)&d_slots, (uint64_t)rows * stride * sizeof(float)) == cudaSuccess, "block-slot scores");
	CHECK(cudaMalloc((void **)&d_reference, (uint64_t)rows * stride * sizeof(float)) == cudaSuccess, "reference scores");
	LmKvView pages_view;
	CHECK(LmKvViewInitialize(&pages_view, pool, d_table, pages, sequences, (uint32_t)table.size(), error) == 0, "view");
	LmKvStoreKernel<Kv,128u><<<sequences * context, 128u>>>(pages_view, d_keys, d_key_sequence, d_key_position, sequences * context, SLOTS_DIM);
	CHECK(cudaDeviceSynchronize() == cudaSuccess, "store keys");
	const View view = MakeView(pages_view, (View *)0);
	CHECK((LmWeightedSparseScoreLaunch<Kv,HEADS,SLOTS_DIM>(d_query, d_weight, view, d_sequence, d_context, d_position, rows, stride, scale,
		d_slots, 0)) == cudaSuccess, "block-slot launch");
	LmWeightedSparseScoreKernel<Kv,View,HEADS,SLOTS_DIM,LM_INDEX_SCORE_SHARED_ROWS><<<dim3(LmIndexScoreBlocks(stride),
		(rows + LM_INDEX_SCORE_SHARED_ROWS - 1u) / LM_INDEX_SCORE_SHARED_ROWS), LM_INDEX_SCORE_THREADS>>>(d_query, d_weight, view, d_sequence, d_context,
		d_position, rows, stride, scale, d_reference);
	CHECK(cudaDeviceSynchronize() == cudaSuccess, "reference launch");
	std::vector<float> slots((uint64_t)rows * stride), reference((uint64_t)rows * stride);
	cudaMemcpy(slots.data(), d_slots, slots.size() * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(reference.data(), d_reference, reference.size() * sizeof(float), cudaMemcpyDeviceToHost);
	uint64_t differ = 0u, finite = 0u;
	for ( uint64_t index = 0u; index < slots.size(); ++index )
	{
		differ += memcmp(&slots[index], &reference[index], sizeof(float)) != 0 ? 1u : 0u;
		finite += reference[index] > -INFINITY ? 1u : 0u;
	}
	CHECK(differ == 0u && finite != 0u && error->error_code == 0u, "block-slot scores equal the per-row scorer's bit for bit");
	printf("index score rows %u context %u heads %u %s slot %u%s stride %u: %llu of %llu scores differ (%llu finite)\n", rows, context, HEADS,
		ViewName((View *)0), SLOT_BYTES, mixed != 0u ? " two sequences" : "", stride, (unsigned long long)differ,
		(unsigned long long)slots.size(), (unsigned long long)finite);
	cudaFree(pool); cudaFree(d_table); cudaFree(d_sequence); cudaFree(d_position); cudaFree(d_key_sequence); cudaFree(d_key_position); cudaFree(d_context);
	cudaFree(d_keys); cudaFree(d_query); cudaFree(d_weight); cudaFree(d_slots); cudaFree(d_reference); cudaFree(error);
}

int main(void)
{
	Case<32u,LmKvView,256u>(64u, 300u, 0u, 0u);
	Case<32u,LmKvView,256u>(77u, 4096u, 0u, 5u);
	Case<32u,LmKvView,256u>(200u, 9000u, 0u, 0u);
	Case<32u,LmKvView,256u>(96u, 4096u, 1u, 0u);
	Case<64u,LmKvView,256u>(72u, 2500u, 0u, 3u);
	Case<32u,LmKvShardView,1280u>(130u, 33000u, 0u, 0u);
	Case<32u,LmKvShardView,1280u>(1024u, 8192u, 0u, 64u);
	Case<32u,LmKvShardView,1280u>(160u, 70000u, 0u, 0u);
	Case<32u,LmKvShardView,1280u>(104u, 5000u, 1u, 0u);
	if ( failures != 0 )
	{
		printf("FAIL index score block slots: %d failure(s)\n", failures);
		return 1;
	}
	printf("PASS index score block slots: one slot lookup a position a block, scores bit-identical to the per-row scorer\n");
	return 0;
}
