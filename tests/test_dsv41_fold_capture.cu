#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include "model-families/dsv41_flash/source/dsv41_flash_stage_kernels.cuh"

#define WIDTH 512u
#define ORDERS 3u
#define SLOTS 8u
#define STATE_FLOATS (2u * WIDTH)
#define PIECE_FLOATS (2u * WIDTH + 64u)
#define HISTORY_WORDS 4u

#define REQUIRE(condition) do { if ( !(condition) ) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #condition); exit(1); } } while ( 0 )
#define CUDA(call) REQUIRE((call) == cudaSuccess)

static uint32_t state_word = 20261119u;

static uint32_t Next(void)
{
	state_word ^= state_word << 13u;
	state_word ^= state_word >> 17u;
	state_word ^= state_word << 5u;
	return state_word;
}

static float Value(void)
{
	return (float)((int32_t)(Next() % 20001u) - 10000) / 997.0f;
}

template<class T>
static T *Upload(const std::vector<T> &host)
{
	T *device = 0;
	CUDA(cudaMalloc(&device, host.size() * sizeof(T)));
	CUDA(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
	return device;
}

template<class T>
static std::vector<T> Download(const T *device, size_t count)
{
	std::vector<T> host(count);
	CUDA(cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost));
	return host;
}

struct Case
{
	uint32_t first_position;
	uint32_t count;
	uint32_t slot;
	uint32_t folded;
};

static uint32_t RunCase(const std::vector<Case> &lanes)
{
	const uint32_t sequences = (uint32_t)lanes.size();
	uint32_t rows = 0u, checked = 0u;
	for ( const Case &lane : lanes )
		rows += lane.count;
	std::vector<uint32_t> order(rows), positions(rows), tokens(rows), row_lane(rows), row_ordinal(rows), begin(sequences + 1u), indices(rows);
	for ( uint32_t row = 0u; row < rows; ++row )
		order[row] = row;
	for ( uint32_t row = rows; row > 1u; --row )
		std::swap(order[row - 1u], order[Next() % row]);
	uint32_t cursor = 0u;
	for ( uint32_t lane = 0u; lane < sequences; ++lane )
	{
		begin[lane] = cursor;
		for ( uint32_t ordinal = 0u; ordinal < lanes[lane].count; ++ordinal, ++cursor )
		{
			const uint32_t row = order[cursor];
			indices[cursor] = row;
			positions[row] = lanes[lane].first_position + ordinal;
			tokens[row] = Next() % 4096u;
			row_lane[row] = lane;
			row_ordinal[row] = ordinal;
		}
	}
	begin[sequences] = cursor;
	std::vector<uint32_t> slots(sequences), fold(sequences), truncated(sequences);
	for ( uint32_t lane = 0u; lane < sequences; ++lane )
	{
		slots[lane] = lanes[lane].slot;
		fold[lane] = lanes[lane].folded;
		truncated[lane] = lanes[lane].count - 1u;
	}
	std::vector<float> kv((uint64_t)rows * WIDTH), score((uint64_t)rows * WIDTH), live_state((uint64_t)SLOTS * STATE_FLOATS);
	std::vector<float> pieces((uint64_t)SLOTS * PIECE_FLOATS);
	std::vector<int32_t> live_history((uint64_t)SLOTS * HISTORY_WORDS), token_map(4096u);
	for ( float &value : kv )
		value = Value();
	for ( float &value : score )
		value = Value();
	for ( float &value : live_state )
		value = Value();
	for ( float &value : pieces )
		value = -7777.0f;
	for ( int32_t &value : live_history )
		value = (int32_t)(Next() % 100000u);
	for ( uint32_t token = 0u; token < 4096u; ++token )
		token_map[token] = (int32_t)(Next() % 50000u);
	const std::vector<float> sentinel = pieces;
	float *d_kv = Upload(kv), *d_score = Upload(score), *d_state = Upload(live_state), *d_pieces = Upload(pieces);
	int32_t *d_history = Upload(live_history), *d_map = Upload(token_map);
	uint32_t *d_positions = Upload(positions), *d_tokens = Upload(tokens), *d_row_lane = Upload(row_lane), *d_row_ordinal = Upload(row_ordinal);
	uint32_t *d_begin = Upload(begin), *d_indices = Upload(indices), *d_slots = Upload(slots), *d_fold = Upload(fold), *d_truncated = Upload(truncated);
	D41CompressorFoldKernel<<<sequences, 128u>>>(d_kv, d_score, d_pieces, d_fold, d_positions, d_begin, d_indices, d_slots, sequences, 2u,
		PIECE_FLOATS, WIDTH);
	D41HistoryFoldKernel<<<1u, 64u>>>(d_tokens, d_map, d_begin, d_indices, d_slots, d_history, HISTORY_WORDS,
		(int32_t *)(d_pieces + 2u * WIDTH), PIECE_FLOATS, d_fold, sequences, ORDERS);
	CUDA(cudaGetLastError());
	const std::vector<float> folded = Download(d_pieces, pieces.size());
	D41CompressorSaveKernel<<<rows, 128u>>>(d_kv, d_score, d_state, d_positions, d_row_lane, d_row_ordinal, d_begin, d_slots, rows, 2u,
		STATE_FLOATS, WIDTH, 1u, d_truncated);
	LmNgramHistoryKernel<<<1u, 64u>>>(d_tokens, d_map, d_begin, d_indices, d_slots, d_history, HISTORY_WORDS, sequences, ORDERS, 1u, d_truncated);
	CUDA(cudaDeviceSynchronize());
	const std::vector<float> unfolded_state = Download(d_state, live_state.size());
	const std::vector<int32_t> unfolded_history = Download(d_history, live_history.size());
	for ( uint32_t lane = 0u; lane < sequences; ++lane )
	{
		const uint32_t slot = lanes[lane].slot, target = lanes[lane].first_position + lanes[lane].count - 1u;
		const float *piece = folded.data() + (uint64_t)slot * PIECE_FLOATS, *before = sentinel.data() + (uint64_t)slot * PIECE_FLOATS;
		const int32_t *piece_history = (const int32_t *)(piece + 2u * WIDTH);
		if ( lanes[lane].folded == 0u )
		{
			REQUIRE(memcmp(piece, before, PIECE_FLOATS * sizeof(float)) == 0);
			continue;
		}
		if ( target % 2u != 0u )
			REQUIRE(memcmp(piece, unfolded_state.data() + (uint64_t)slot * STATE_FLOATS, STATE_FLOATS * sizeof(float)) == 0);
		else
			REQUIRE(memcmp(piece, before, STATE_FLOATS * sizeof(float)) == 0);
		REQUIRE(memcmp(piece_history, unfolded_history.data() + (uint64_t)slot * HISTORY_WORDS, ORDERS * sizeof(int32_t)) == 0);
		checked++;
	}
	cudaFree(d_kv); cudaFree(d_score); cudaFree(d_state); cudaFree(d_pieces); cudaFree(d_history); cudaFree(d_map); cudaFree(d_positions);
	cudaFree(d_tokens); cudaFree(d_row_lane); cudaFree(d_row_ordinal); cudaFree(d_begin); cudaFree(d_indices); cudaFree(d_slots); cudaFree(d_fold);
	cudaFree(d_truncated);
	return checked;
}

int main(void)
{
	int devices = 0;
	if ( cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0 )
	{
		printf("SKIP dsv41 fold capture: no CUDA device\n");
		return 0;
	}
	uint32_t checked = 0u;
	checked += RunCase({ { 12u, 6u, 5u, 1u } });
	checked += RunCase({ { 13u, 6u, 5u, 1u } });
	checked += RunCase({ { 40u, 2u, 1u, 1u } });
	checked += RunCase({ { 41u, 2u, 2u, 1u } });
	checked += RunCase({ { 7u, 9u, 3u, 1u }, { 100u, 4u, 0u, 0u }, { 64u, 3u, 6u, 1u }, { 5u, 17u, 7u, 1u } });
	for ( uint32_t round = 0u; round < 40u; ++round )
	{
		std::vector<Case> lanes;
		const uint32_t count = 1u + Next() % 6u;
		std::vector<uint32_t> slots = { 0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u };
		for ( uint32_t lane = 0u; lane < count; ++lane )
		{
			std::swap(slots[lane], slots[lane + Next() % (SLOTS - lane)]);
			lanes.push_back({ Next() % 5000u, 2u + Next() % 40u, slots[lane], Next() % 3u != 0u ? 1u : 0u });
		}
		checked += RunCase(lanes);
	}
	printf("PASS dsv41 fold capture: %u folded lanes match the state a step ending one token earlier leaves, lanes without the fold untouched\n", checked);
	return 0;
}
