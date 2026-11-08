#pragma once

#include "inference/kernels/attn_prefill.cuh"
#include "inference/kernels/attn_shard.cuh"
#include "inference/kernels/tma.cuh"

#define LM_LATENT_SHARD_PREFILL_MIN_ROWS 17u

#define LM_LATENT_SHARD_QUERIES 32u
#define LM_LATENT_SHARD_ROW_PAD 16u
#define LM_LATENT_SHARD_PLANS 4u
#define LM_LATENT_SHARD_PLAN_WORDS (LM_LATENT_SHARD_QUERIES + 8u)
#define LM_LATENT_SHARD_PREFILL_THREADS ((LM_PREFILL_ATTN_WARPS + 2u) * 32u)

typedef struct LmLatentShardSelection
{
	const uint32_t *selected_positions;
	const uint32_t *split_positions;
	const uint32_t *split_offsets;
	uint32_t selected_count;
	uint32_t dense_limit;
	uint32_t record_heads;
	uint32_t record_copies;
	uint32_t empty_rows;
	uint32_t padded_heads;
} LmLatentShardSelection;

static __host__ __device__ inline LmLatentShardSelection LmLatentShardDense(uint32_t heads_per_rank)
{
	LmLatentShardSelection selection = { 0, 0, 0, 0u, 0u, heads_per_rank, 1u, 0u, 0u };
	return selection;
}

static __host__ __device__ constexpr uint64_t LmLatentShardListBytes(uint32_t selected_count)
{
	return ((uint64_t)selected_count + LM_LATENT_SHARD_THREADS) * sizeof(uint32_t);
}

template<uint32_t DIM>
static __host__ __device__ constexpr uint32_t LmLatentShardPrefillSharedBytes(void)
{
	return(2u * LM_PREFILL_ATTN_TILE * (DIM * 2u + LM_LATENT_SHARD_ROW_PAD)
		+ 2u * LM_LATENT_SHARD_QUERIES * LM_PREFILL_ATTN_SCORE_STRIDE * 4u
		+ LM_LATENT_SHARD_QUERIES * LM_PREFILL_ATTN_WEIGHT_STRIDE * 2u
		+ 3u * LM_LATENT_SHARD_QUERIES * 4u
		+ (5u + 2u * LM_LATENT_SHARD_PLANS) * 8u
		+ LM_LATENT_SHARD_PLANS * LM_LATENT_SHARD_PLAN_WORDS * 4u);
}

static inline uint32_t LmLatentShardPrefillMultiprocessors(void)
{
	int device = 0, count = 0;
	if (cudaGetDevice(&device) != cudaSuccess || cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess || count < 1)
		return 1u;
	return (uint32_t)count;
}

template<class Geometry>
static __device__ __forceinline__ void LmLatentShardSlotPrefetch(const LmKvShardView &view, uint32_t sequence, uint32_t position)
{
	const uint32_t page = Geometry::PageOf(position);
	if (sequence < view.pages.sequence_count && page < view.pages.page_table_stride)
		asm volatile("prefetch.global.L1 [%0];\n" :: "l"(view.pages.page_table + (uint64_t)sequence * view.pages.page_table_stride + page));
}

template<class Geometry>
static __device__ __forceinline__ void LmLatentShardSlotPrefetch(const LmKvShardReplicaView &view, uint32_t sequence, uint32_t position)
{
	const uint32_t page = Geometry::PageOf(position);
	if (sequence < view.pages.sequence_count && page < view.pages.page_table_stride)
		asm volatile("prefetch.global.L1 [%0];\n" :: "l"(view.pages.page_table + (uint64_t)sequence * view.pages.page_table_stride + page));
}

template<class Geometry>
static __device__ __forceinline__ void LmLatentShardSlotPrefetch(const LmKvShardGatherView &, uint32_t, uint32_t)
{
}

static __device__ __forceinline__ void LmLatentShardConsumerBarrier(void)
{
	asm volatile("bar.sync 1, %0;\n" :: "n"(LM_PREFILL_ATTN_THREADS) : "memory");
}

static __device__ __forceinline__ uint32_t LmLatentShardCompactWarp(
	SparkKvShard shard,
	const uint32_t *__restrict__ selected,
	uint32_t selected_count,
	uint32_t bound,
	uint32_t *list)
{
	constexpr uint32_t BATCH = 8u;
	const uint32_t lane = threadIdx.x & 31u, below = (1u << lane) - 1u;
	uint32_t base, item, count = 0u, keep, mask, value[BATCH], upcoming[BATCH];
#pragma unroll
	for (item = 0u; item < BATCH; item++)
		upcoming[item] = item * 32u + lane < selected_count ? selected[item * 32u + lane] : 0xffffffffu;
	for (base = 0u; base < selected_count; base += BATCH * 32u)
	{
#pragma unroll
		for (item = 0u; item < BATCH; item++)
		{
			value[item] = upcoming[item];
			upcoming[item] = base + (BATCH + item) * 32u + lane < selected_count ? selected[base + (BATCH + item) * 32u + lane] : 0xffffffffu;
		}
#pragma unroll
		for (item = 0u; item < BATCH; item++)
		{
			keep = base + item * 32u + lane < selected_count && value[item] <= bound && SparkKvShardOwns(shard, value[item]) != 0u ? 1u : 0u;
			mask = __ballot_sync(0xffffffffu, keep != 0u);
			if (keep != 0u)
				list[count + (uint32_t)__popc(mask & below)] = value[item];
			count += (uint32_t)__popc(mask);
		}
	}
	return count;
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE, uint32_t ROWS>
__global__ __launch_bounds__(LM_LATENT_SHARD_PREFILL_THREADS, 1)
void LmLatentShardPrefillKernel(
	Pages cache,
	const uint16_t *__restrict__ query_bf16,
	uint64_t query_rank_stride,
	uint32_t heads_per_rank,
	const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ context_length,
	const uint32_t *__restrict__ row_position,
	float qk_scale,
	float *__restrict__ partials,
	uint64_t partial_rank_stride,
	uint32_t rows,
	uint32_t split_count,
	uint64_t split_stride,
	LmLatentShardSelection selection)
{
	constexpr uint32_t DIM = LATENT + ROPE;
	constexpr uint32_t ROW_BYTES = DIM * 2u;
	constexpr uint32_t PITCH = ROW_BYTES + LM_LATENT_SHARD_ROW_PAD;
	constexpr uint32_t CHUNKS = DIM / 8u;
	constexpr uint32_t QUERIES = LM_LATENT_SHARD_QUERIES;
	constexpr uint32_t TILE = LM_PREFILL_ATTN_TILE;
	constexpr uint32_t K_STEPS = DIM / 16u;
	constexpr uint32_t HALF_STEPS = K_STEPS / 2u;
	constexpr uint32_t WARP_COLUMNS = LATENT / LM_PREFILL_ATTN_WARPS;
	constexpr uint32_t COLUMN_FRAGS = WARP_COLUMNS / 8u;
	constexpr uint32_t TILE_CHUNKS = TILE * CHUNKS;
	constexpr uint32_t BLOCK_HEADS = QUERIES / ROWS;
	constexpr uint32_t QUERY_KEYS = QUERIES * TILE / LM_PREFILL_ATTN_THREADS;
	constexpr uint32_t QUERY_THREADS = TILE / QUERY_KEYS;
	constexpr uint32_t PLANNER = LM_PREFILL_ATTN_WARPS, ISSUER = LM_PREFILL_ATTN_WARPS + 1u;
	constexpr uint32_t PLANS = LM_LATENT_SHARD_PLANS;
	constexpr uint32_t PLAN_WORDS = LM_LATENT_SHARD_PLAN_WORDS;
	constexpr uint32_t PLAN_HEAD = PLAN_WORDS - QUERIES;
	static_assert(ROWS >= 1u && QUERIES % ROWS == 0u && ROWS <= 32u, "a block holds whole heads of each of its rows, one row per producer lane");
	static_assert(QUERIES == 32u && LM_PREFILL_ATTN_WARPS == 8u && TILE == 32u, "two 16-query tiles, a k-half and an 8-key pair per warp, one key per lane");
	static_assert(QUERY_KEYS == 4u && QUERY_THREADS == 8u, "eight threads hold a query's keys k, k+8, k+16 and k+24");
	static_assert(PLAN_HEAD == 8u, "a plan holds eight row words and one key count per query");
	static_assert(DIM % 64u == 0u && K_STEPS % 2u == 0u, "the query-key width must split into two halves of 16-wide steps");
	static_assert(LATENT % (LM_PREFILL_ATTN_WARPS * 16u) == 0u, "each warp owns whole 16-column value slices");
	static_assert(Geometry::kSlotBytes >= ROW_BYTES, "the slot holds the latent and its rope part");
	static_assert(PITCH % 128u == 16u, "a padded row starts four banks after the row before it");
	static_assert(LmLatentShardPrefillSharedBytes<DIM>() <= LM_PREFILL_ATTN_SHARED_LIMIT, "the tiles exceed the shared memory an SM has");
	extern __shared__ __align__(128) uint8_t lm_shard_prefill_shared[];
	uint8_t *keys = lm_shard_prefill_shared;
	float *score_parts = (float *)(keys + 2u * TILE * PITCH);
	uint16_t *weight_high = (uint16_t *)(score_parts + 2u * QUERIES * LM_PREFILL_ATTN_SCORE_STRIDE);
	float *running_max = (float *)(weight_high + QUERIES * LM_PREFILL_ATTN_WEIGHT_STRIDE);
	float *running_sum = running_max + QUERIES;
	float *rescale = running_sum + QUERIES;
	uint64_t *full = (uint64_t *)(rescale + QUERIES);
	uint64_t *empty = full + 2u;
	uint64_t *planned = empty + 2u;
	uint64_t *read = planned + PLANS;
	uint64_t *listed_free = read + PLANS;
	uint32_t *plans = (uint32_t *)(listed_free + 1u);
	uint32_t *list_shared = plans + PLANS * PLAN_WORDS;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t matrix = lane >> 3u, within = lane & 7u;
	const uint32_t own = threadIdx.x / QUERY_THREADS, part = threadIdx.x % QUERY_THREADS;
	const uint32_t group = blockIdx.y;
	const uint32_t blocks = (rows + ROWS - 1u) / ROWS;
	const uint32_t key_base = (uint32_t)__cvta_generic_to_shared(keys);
	const uint32_t qk_tile = warp >> 2u, qk_half = (warp >> 1u) & 1u, qk_pair = warp & 1u;
	uint32_t block, index, step, frag, q, head, query_row, m, shift, copy_index, used, ordinal;
	uint64_t base;

	partials += (uint64_t)blockIdx.z * split_stride;
	if (threadIdx.x == 0u)
	{
		LmMbarrierInit(&full[0], 1u);
		LmMbarrierInit(&full[1], 1u);
		LmMbarrierInit(&empty[0], LM_PREFILL_ATTN_WARPS);
		LmMbarrierInit(&empty[1], LM_PREFILL_ATTN_WARPS);
		for (index = 0u; index < PLANS; index++)
		{
			LmMbarrierInit(&planned[index], 1u);
			LmMbarrierInit(&read[index], LM_PREFILL_ATTN_WARPS + 1u);
		}
		LmMbarrierInit(listed_free, 1u);
		LmMbarrierInitFence();
	}
	__syncthreads();
	if (warp == PLANNER)
	{
		uint32_t compactions = 0u;
		for (block = blockIdx.x, ordinal = 0u; block < blocks; block += gridDim.x, ordinal++)
		{
			const uint32_t record = ordinal % PLANS, row = block * ROWS, last_row = (row + ROWS < rows ? row + ROWS : rows) - 1u;
			const uint32_t mine = row + (lane < ROWS && row + lane <= last_row ? lane : 0u);
			const uint32_t lane_sequence = sequence_of_row[mine], lane_position = row_position[mine];
			uint32_t sequence, valid, listed, count, list_first = 0u, limit, window_begin, tiles, tile_low, span, tile_first, tile_end, lane_dense, lane_keys, own_keys;
			sequence = __shfl_sync(0xffffffffu, lane_sequence, 0u);
			if (__ballot_sync(0xffffffffu, lane_sequence != sequence) != 0u)
				sequence = cache.pages.sequence_count;
			valid = LmKvViewIsConfigured(cache.pages) && sequence < cache.pages.sequence_count ? 1u : 0u;
			lane_dense = valid != 0u ? context_length[sequence] : 0u;
			if (selection.empty_rows != 0u && lane_position == 0xffffffffu)
				lane_dense = 0u;
			else if (lane_position < lane_dense)
				lane_dense = lane_position + 1u;
			listed = ROWS == 1u && selection.selected_positions != 0 && valid != 0u && lane_dense > selection.dense_limit ? 1u : 0u;
			count = 0u;
			if (listed != 0u && selection.split_positions != 0)
			{
				const uint32_t *offsets = selection.split_offsets + (uint64_t)row * (cache.shard.degree + 1u) + cache.shard.rank;
				list_first = offsets[0];
				count = offsets[1] - list_first;
			}
			else if (listed != 0u)
			{
				if (compactions > 0u)
					LmMbarrierWait(listed_free, (compactions - 1u) & 1u);
				count = LmLatentShardCompactWarp(cache.shard, selection.selected_positions + (uint64_t)row * selection.selected_count, selection.selected_count,
					__shfl_sync(0xffffffffu, lane_position, 0u), list_shared);
				compactions++;
			}
			lane_keys = lane < ROWS && row + lane <= last_row ? SparkKvShardLocalKeys(cache.shard, lane_dense) : 0u;
			if (listed == 0u)
			{
				count = lane_keys;
#pragma unroll
				for (shift = 16u; shift > 0u; shift >>= 1u)
				{
					const uint32_t other = __shfl_xor_sync(0xffffffffu, count, shift);
					count = other > count ? other : count;
				}
			}
			window_begin = listed != 0u ? 0u : LmKvKeyWindowBegin(cache);
			limit = listed != 0u ? count : LmKvKeyWindowEnd(cache);
			limit = count < limit ? count : limit;
			tile_low = window_begin / TILE;
			tiles = limit > window_begin ? (limit + TILE - 1u) / TILE : tile_low;
			span = tiles - tile_low;
			tile_first = tile_low + (span + split_count - 1u) / split_count * blockIdx.z;
			tile_end = tile_first + (span + split_count - 1u) / split_count;
			tile_first = tile_first < tiles ? tile_first : tiles;
			tile_end = tile_end < tiles ? tile_end : tiles;
			if (valid == 0u)
				tile_first = tile_end = 0u;
			own_keys = __shfl_sync(0xffffffffu, lane_keys, lane / BLOCK_HEADS);
			own_keys = row + lane / BLOCK_HEADS > last_row ? 0u : listed != 0u ? count : own_keys;
			own_keys = own_keys < limit ? own_keys : limit;
			if (ordinal >= PLANS)
				LmMbarrierWait(&read[record], ((ordinal / PLANS) - 1u) & 1u);
			plans[record * PLAN_WORDS + PLAN_HEAD + lane] = own_keys;
			if (lane == 0u)
			{
				plans[record * PLAN_WORDS] = tile_first;
				plans[record * PLAN_WORDS + 1u] = tile_end;
				plans[record * PLAN_WORDS + 2u] = window_begin;
				plans[record * PLAN_WORDS + 3u] = valid;
				plans[record * PLAN_WORDS + 4u] = limit;
				plans[record * PLAN_WORDS + 5u] = listed;
				plans[record * PLAN_WORDS + 6u] = list_first;
				plans[record * PLAN_WORDS + 7u] = sequence;
			}
			__syncwarp();
			if (lane == 0u)
				LmMbarrierArrive(&planned[record]);
		}
		return;
	}
	if (warp == ISSUER)
	{
		uint32_t row = 0u, sequence = 0u, listed = 0u, limit = 0u, window_begin = 0u, tile = 0u, tile_end = 0u, ahead = 0u, ahead_tile = 0xffffffffu;
		const uint32_t *list = 0;
		const uint8_t *slot = 0;
		auto next = [&]() -> uint32_t
		{
			if (tile + 1u < tile_end)
			{
				tile++;
				return 1u;
			}
			while (block < blocks)
			{
				const uint32_t record = ordinal % PLANS;
				uint32_t first;
				LmMbarrierWait(&planned[record], (ordinal / PLANS) & 1u);
				first = plans[record * PLAN_WORDS];
				tile_end = plans[record * PLAN_WORDS + 1u];
				window_begin = plans[record * PLAN_WORDS + 2u];
				limit = plans[record * PLAN_WORDS + 4u];
				listed = plans[record * PLAN_WORDS + 5u];
				sequence = plans[record * PLAN_WORDS + 7u];
				row = block * ROWS;
				list = listed == 0u ? 0 : selection.split_positions != 0 ? selection.split_positions + (uint64_t)row * selection.selected_count + plans[record * PLAN_WORDS + 6u] : list_shared;
				ahead_tile = 0xffffffffu;
				__syncwarp();
				if (lane == 0u)
				{
					LmMbarrierArrive(&read[record]);
					if (listed != 0u && selection.split_positions == 0 && first >= tile_end)
						LmMbarrierArrive(listed_free);
				}
				block += gridDim.x;
				ordinal++;
				if (first < tile_end)
				{
					tile = first;
					return 1u;
				}
			}
			return 0u;
		};
		auto listing = [&](uint32_t which) -> uint32_t
		{
			const uint32_t local = which * TILE + lane;
			if (local >= limit || local < window_begin)
				return 0u;
			return listed == 0u ? SparkKvShardLocalPosition(cache.shard, local) : list[local];
		};
		auto resolve = [&]() -> const uint8_t *
		{
			const uint32_t local = tile * TILE + lane;
			const uint32_t position = ahead_tile == tile ? ahead : listing(tile);
			ahead_tile = 0xffffffffu;
			if (tile + 1u < tile_end)
			{
				ahead = listing(tile + 1u);
				ahead_tile = tile + 1u;
				if ((tile + 1u) * TILE + lane < limit && (tile + 1u) * TILE + lane >= window_begin)
					LmLatentShardSlotPrefetch<Geometry>(cache, sequence, ahead);
			}
			else if (listed != 0u && selection.split_positions == 0)
			{
				__syncwarp();
				if (lane == 0u)
					LmMbarrierArrive(listed_free);
			}
			if (local >= limit || local < window_begin)
				return 0;
			return LmKvShardSlotRequired<Geometry>(cache, sequence, position, row, LM_KV_ACCESS_READ);
		};
		block = blockIdx.x;
		ordinal = 0u;
		tile_end = 0u;
		if (next() != 0u)
		{
			slot = resolve();
			for (used = 0u;; used++)
			{
				const uint32_t buffer = used & 1u, uses = used >> 1u;
				const uint32_t present = __ballot_sync(0xffffffffu, slot != 0);
				if (uses > 0u)
					LmMbarrierWait(&empty[buffer], (uses - 1u) & 1u);
				if (present != 0xffffffffu)
					for (index = lane; index < TILE_CHUNKS; index += 32u)
						if (((present >> (index / CHUNKS)) & 1u) == 0u)
							*(uint4 *)(keys + (buffer * TILE + index / CHUNKS) * PITCH + (index % CHUNKS) * 16u) = make_uint4(0u, 0u, 0u, 0u);
				LmMbarrierInitFence();
				__syncwarp();
				if (lane == 0u)
					LmMbarrierArriveExpect(&full[buffer], (uint32_t)__popc(present) * ROW_BYTES);
				if (slot != 0)
					LmTmaLoadBulk1d(keys + (buffer * TILE + lane) * PITCH, slot, &full[buffer], ROW_BYTES);
				if (next() == 0u)
					break;
				slot = resolve();
			}
		}
		return;
	}
	uint32_t query_frag[HALF_STEPS][4];
	float output[2][COLUMN_FRAGS][4];
	auto fetch = [&](uint32_t which)
	{
		const uint32_t row = which * ROWS, last_row = (row + ROWS < rows ? row + ROWS : rows) - 1u;
		const uint32_t low = qk_tile * 16u + (lane >> 2u), high = low + 8u;
		const uint32_t low_head = selection.padded_heads != 0u ? (low % BLOCK_HEADS) % selection.padded_heads : group * BLOCK_HEADS + low % BLOCK_HEADS;
		const uint32_t high_head = selection.padded_heads != 0u ? (high % BLOCK_HEADS) % selection.padded_heads : group * BLOCK_HEADS + high % BLOCK_HEADS;
		const uint32_t low_row = row + low / BLOCK_HEADS <= last_row ? row + low / BLOCK_HEADS : last_row;
		const uint32_t high_row = row + high / BLOCK_HEADS <= last_row ? row + high / BLOCK_HEADS : last_row;
		const uint32_t *low_query = (const uint32_t *)(query_bf16 + (uint64_t)(low_head / heads_per_rank) * query_rank_stride +
			((uint64_t)low_row * heads_per_rank + low_head % heads_per_rank) * DIM) + qk_half * HALF_STEPS * 8u + (lane & 3u);
		const uint32_t *high_query = (const uint32_t *)(query_bf16 + (uint64_t)(high_head / heads_per_rank) * query_rank_stride +
			((uint64_t)high_row * heads_per_rank + high_head % heads_per_rank) * DIM) + qk_half * HALF_STEPS * 8u + (lane & 3u);
#pragma unroll
		for (step = 0u; step < HALF_STEPS; step++)
		{
			query_frag[step][0] = low_query[step * 8u];
			query_frag[step][1] = high_query[step * 8u];
			query_frag[step][2] = low_query[step * 8u + 4u];
			query_frag[step][3] = high_query[step * 8u + 4u];
		}
	};
	used = 0u;
	if (blockIdx.x < blocks)
		fetch(blockIdx.x);
	for (block = blockIdx.x, ordinal = 0u; block < blocks; block += gridDim.x, ordinal++)
	{
		const uint32_t record = ordinal % PLANS, row = block * ROWS, last_row = (row + ROWS < rows ? row + ROWS : rows) - 1u;
		const uint32_t following = block + gridDim.x;
		uint32_t tile, tile_first, tile_end, window_begin, valid, own_count;
		LmMbarrierWait(&planned[record], (ordinal / PLANS) & 1u);
		tile_first = plans[record * PLAN_WORDS];
		tile_end = plans[record * PLAN_WORDS + 1u];
		window_begin = plans[record * PLAN_WORDS + 2u];
		valid = plans[record * PLAN_WORDS + 3u];
		own_count = plans[record * PLAN_WORDS + PLAN_HEAD + own];
		__syncwarp();
		if (lane == 0u)
			LmMbarrierArrive(&read[record]);
		if (valid == 0u)
		{
			if (threadIdx.x == 0u)
				LmKvReportRequiredAccessFailure(cache.pages, !LmKvViewIsConfigured(cache.pages) ? LM_KV_ACCESS_ERROR_INVALID_VIEW : LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE, LM_KV_ACCESS_READ, row, sequence_of_row[row], 0xffffffffu, 0xffffffffu);
			if (following < blocks)
				fetch(following);
			continue;
		}
		if (threadIdx.x < QUERIES)
		{
			running_max[threadIdx.x] = -INFINITY;
			running_sum[threadIdx.x] = 0.0f;
		}
#pragma unroll
		for (m = 0u; m < 2u; m++)
#pragma unroll
			for (frag = 0u; frag < COLUMN_FRAGS; frag++)
				output[m][frag][0] = output[m][frag][1] = output[m][frag][2] = output[m][frag][3] = 0.0f;
		for (tile = tile_first; tile < tile_end; tile++, used++)
		{
			const uint32_t buffer = used & 1u;
			const uint32_t tile_base = key_base + buffer * TILE * PITCH;
			LmMbarrierWait(&full[buffer], (used >> 1u) & 1u);
			{
				float score[2][4] = { { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } };
				uint32_t b[4], first[2], second[2];
				const uint32_t b_row = (qk_pair * 2u + (matrix >> 1u)) * 8u + within;
#pragma unroll
				for (step = 0u; step < HALF_STEPS; step++)
				{
					const uint32_t b_chunk = (qk_half * HALF_STEPS + step) * 2u + (matrix & 1u);
					LmPrefillLoadX4(tile_base + b_row * PITCH + b_chunk * 16u, b);
					first[0] = b[0]; first[1] = b[1];
					second[0] = b[2]; second[1] = b[3];
					LmMmaBf16(score[0], query_frag[step], first);
					LmMmaBf16(score[1], query_frag[step], second);
				}
#pragma unroll
				for (m = 0u; m < 2u; m++)
#pragma unroll
					for (index = 0u; index < 4u; index++)
						score_parts[(qk_half * QUERIES + qk_tile * 16u + (lane >> 2u) + (index >> 1u) * 8u) * LM_PREFILL_ATTN_SCORE_STRIDE + (qk_pair * 2u + m) * 8u + (lane & 3u) * 2u + (index & 1u)] = score[m][index];
			}
			if (tile + 1u == tile_end && following < blocks)
				fetch(following);
			LmLatentShardConsumerBarrier();
			{
				float value[QUERY_KEYS], weight[QUERY_KEYS], peak, previous_max, next_max, scale, total;
#pragma unroll
				for (index = 0u; index < QUERY_KEYS; index++)
				{
					const uint32_t key = part + index * QUERY_THREADS, local = tile * TILE + key;
					value[index] = (score_parts[own * LM_PREFILL_ATTN_SCORE_STRIDE + key] + score_parts[(QUERIES + own) * LM_PREFILL_ATTN_SCORE_STRIDE + key]) * qk_scale;
					value[index] = local < own_count && local >= window_begin ? value[index] : -INFINITY;
				}
				peak = fmaxf(fmaxf(value[0], value[2]), fmaxf(value[1], value[3]));
#pragma unroll
				for (shift = QUERY_THREADS / 2u; shift > 0u; shift >>= 1u)
					peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, shift));
				previous_max = running_max[own];
				next_max = fmaxf(previous_max, peak);
				scale = previous_max == -INFINITY ? 0.0f : __expf(previous_max - next_max);
#pragma unroll
				for (index = 0u; index < QUERY_KEYS; index++)
				{
					weight[index] = next_max == -INFINITY || value[index] == -INFINITY ? 0.0f : __expf(value[index] - next_max);
					weight_high[own * LM_PREFILL_ATTN_WEIGHT_STRIDE + part + index * QUERY_THREADS] = LmFloatToBf16(weight[index]);
				}
				total = (weight[0] + weight[2]) + (weight[1] + weight[3]);
#pragma unroll
				for (shift = QUERY_THREADS / 2u; shift > 0u; shift >>= 1u)
					total += __shfl_xor_sync(0xffffffffu, total, shift);
				__syncwarp();
				if (part == 0u)
				{
					running_sum[own] = running_sum[own] * scale + total;
					running_max[own] = next_max;
					rescale[own] = scale;
				}
			}
			LmLatentShardConsumerBarrier();
			{
				uint32_t a_high[2][4], b[4], first[2], second[2];
#pragma unroll
				for (m = 0u; m < 2u; m++)
				{
					const float low_scale = rescale[m * 16u + (lane >> 2u)], high_scale = rescale[m * 16u + (lane >> 2u) + 8u];
#pragma unroll
					for (frag = 0u; frag < COLUMN_FRAGS; frag++)
					{
						output[m][frag][0] *= low_scale;
						output[m][frag][1] *= low_scale;
						output[m][frag][2] *= high_scale;
						output[m][frag][3] *= high_scale;
					}
				}
#pragma unroll
				for (step = 0u; step < TILE / 16u; step++)
				{
#pragma unroll
					for (m = 0u; m < 2u; m++)
					{
						const uint32_t a_row = m * 16u + (matrix & 1u) * 8u + within, a_column = step * 16u + (matrix >> 1u) * 8u;
						LmPrefillLoadX4((uint32_t)__cvta_generic_to_shared(weight_high + a_row * LM_PREFILL_ATTN_WEIGHT_STRIDE + a_column), a_high[m]);
					}
#pragma unroll
					for (frag = 0u; frag < COLUMN_FRAGS; frag += 2u)
					{
						const uint32_t b_row = step * 16u + (matrix & 1u) * 8u + within;
						const uint32_t b_chunk = (warp * WARP_COLUMNS + frag * 8u) / 8u + (matrix >> 1u);
						LmPrefillLoadX4Trans(tile_base + b_row * PITCH + b_chunk * 16u, b);
						first[0] = b[0]; first[1] = b[1];
						second[0] = b[2]; second[1] = b[3];
#pragma unroll
						for (m = 0u; m < 2u; m++)
						{
							LmMmaBf16(output[m][frag], a_high[m], first);
							LmMmaBf16(output[m][frag + 1u], a_high[m], second);
						}
					}
				}
			}
			__syncwarp();
			if (lane == 0u)
				LmMbarrierArrive(&empty[buffer]);
		}
		if (tile_first >= tile_end && following < blocks)
			fetch(following);
#pragma unroll
		for (m = 0u; m < 2u; m++)
#pragma unroll
			for (index = 0u; index < 4u; index += 2u)
			{
				q = m * 16u + (lane >> 2u) + (index >> 1u) * 8u;
				head = group * BLOCK_HEADS + q % BLOCK_HEADS;
				query_row = row + q / BLOCK_HEADS;
				if (query_row > last_row || (selection.padded_heads != 0u && q % BLOCK_HEADS >= selection.padded_heads))
					continue;
				for (copy_index = 0u; copy_index < selection.record_copies; copy_index++)
				{
					float *target = partials + ((uint64_t)(head / selection.record_heads) * selection.record_copies + copy_index) * partial_rank_stride +
						((uint64_t)query_row * selection.record_heads + head % selection.record_heads) * LM_LATENT_SHARD_RECORD_FLOATS(LATENT) + 2u + warp * WARP_COLUMNS + (lane & 3u) * 2u;
#pragma unroll
					for (frag = 0u; frag < COLUMN_FRAGS; frag++)
						*(float2 *)(target + frag * 8u) = make_float2(output[m][frag][index], output[m][frag][index + 1u]);
				}
			}
		if (threadIdx.x < QUERIES && row + threadIdx.x / BLOCK_HEADS <= last_row &&
			(selection.padded_heads == 0u || threadIdx.x % BLOCK_HEADS < selection.padded_heads))
		{
			head = group * BLOCK_HEADS + threadIdx.x % BLOCK_HEADS;
			query_row = row + threadIdx.x / BLOCK_HEADS;
			for (copy_index = 0u; copy_index < selection.record_copies; copy_index++)
			{
				base = ((uint64_t)(head / selection.record_heads) * selection.record_copies + copy_index) * partial_rank_stride +
					((uint64_t)query_row * selection.record_heads + head % selection.record_heads) * LM_LATENT_SHARD_RECORD_FLOATS(LATENT);
				partials[base] = running_max[threadIdx.x];
				partials[base + 1u] = running_sum[threadIdx.x];
			}
		}
	}
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE, uint32_t ROWS>
static inline cudaError_t LmLatentShardPrefillRowsLaunch(
	Pages cache,
	const uint16_t *query_bf16,
	uint64_t query_rank_stride,
	uint32_t heads_per_rank,
	uint32_t total_heads,
	const uint32_t *sequence_of_row,
	const uint32_t *context_length,
	const uint32_t *row_position,
	float qk_scale,
	float *partials,
	uint64_t partial_rank_stride,
	uint32_t rows,
	uint32_t split_count,
	uint64_t split_stride,
	cudaStream_t stream,
	LmLatentShardSelection selection)
{
	constexpr uint32_t block_heads = LM_LATENT_SHARD_QUERIES / ROWS;
	const uint64_t shared = LmLatentShardPrefillSharedBytes<LATENT + ROPE>() +
		(selection.selected_positions != 0 && selection.split_positions == 0 ? LmLatentShardListBytes(selection.selected_count) : 0u);
	static bool granted = false;
	if (rows == 0u || heads_per_rank == 0u || query_bf16 == 0 || partials == 0 || row_position == 0 ||
		total_heads == 0u || (selection.padded_heads != 0u ? ROWS != 1u || selection.padded_heads != total_heads || total_heads >= block_heads :
			(total_heads % block_heads) != 0u) ||
		selection.record_heads == 0u || selection.record_copies == 0u || total_heads % selection.record_heads != 0u ||
		(selection.selected_positions != 0 && (ROWS != 1u || split_count != 1u || (selection.split_positions == 0) != (selection.split_offsets == 0))) ||
		shared > LM_PREFILL_ATTN_SHARED_LIMIT ||
		SparkKvShardValid(cache.shard, Geometry::kPageSlots) == 0u ||
		query_rank_stride < (uint64_t)rows * heads_per_rank * (LATENT + ROPE) ||
		partial_rank_stride < (uint64_t)rows * selection.record_heads * LM_LATENT_SHARD_RECORD_FLOATS(LATENT) ||
		(partial_rank_stride % 2u) != 0u)
		return cudaErrorInvalidValue;
	if (!granted)
	{
		cudaError_t error = cudaFuncSetAttribute((const void *)LmLatentShardPrefillKernel<Geometry, Pages, LATENT, ROPE, ROWS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)LM_PREFILL_ATTN_SHARED_LIMIT);
		if (error != cudaSuccess)
			return error;
		granted = true;
	}
	const uint32_t blocks = (rows + ROWS - 1u) / ROWS, groups = selection.padded_heads != 0u ? 1u : total_heads / block_heads;
	const uint32_t resident = LmLatentShardPrefillMultiprocessors() / (groups * split_count);
	const uint32_t grid = blocks < resident ? blocks : resident > 0u ? resident : 1u;
	LmLatentShardPrefillKernel<Geometry, Pages, LATENT, ROPE, ROWS><<<dim3(grid, groups, split_count), LM_LATENT_SHARD_PREFILL_THREADS, shared, stream>>>(
		cache, query_bf16, query_rank_stride, heads_per_rank, sequence_of_row, context_length, row_position, qk_scale, partials, partial_rank_stride, rows,
		split_count, split_stride, selection);
	return cudaPeekAtLastError();
}

template<uint32_t LATENT>
__global__ __launch_bounds__(LM_LATENT_SHARD_THREADS)
void LmLatentShardSplitMergeKernel(
	const float *__restrict__ split_partials,
	uint64_t split_rank_stride,
	uint64_t split_stride,
	uint32_t split_count,
	float *__restrict__ partials,
	uint64_t partial_rank_stride,
	uint32_t heads_per_rank)
{
	const uint32_t row = blockIdx.x, head = blockIdx.y;
	const uint64_t record = ((uint64_t)row * heads_per_rank + head % heads_per_rank) * LM_LATENT_SHARD_RECORD_FLOATS(LATENT);
	const uint64_t source = (uint64_t)(head / heads_per_rank) * split_rank_stride + record;
	const uint64_t base = (uint64_t)(head / heads_per_rank) * partial_rank_stride + record;
	float maximum = -INFINITY, sum = 0.0f, weight, value;
	uint32_t split, element;
	for (split = 0u; split < split_count; split++)
		maximum = fmaxf(maximum, split_partials[(uint64_t)split * split_stride + source]);
	for (split = 0u; split < split_count; split++)
	{
		const float local = split_partials[(uint64_t)split * split_stride + source];
		weight = maximum == -INFINITY || local == -INFINITY ? 0.0f : __expf(local - maximum);
		sum = fmaf(split_partials[(uint64_t)split * split_stride + source + 1u], weight, sum);
	}
	for (element = threadIdx.x; element < LATENT; element += LM_LATENT_SHARD_THREADS)
	{
		value = 0.0f;
		for (split = 0u; split < split_count; split++)
		{
			const float local = split_partials[(uint64_t)split * split_stride + source];
			weight = maximum == -INFINITY || local == -INFINITY ? 0.0f : __expf(local - maximum);
			value = fmaf(split_partials[(uint64_t)split * split_stride + source + 2u + element], weight, value);
		}
		partials[base + 2u + element] = value;
	}
	if (threadIdx.x == 0u)
	{
		partials[base] = maximum;
		partials[base + 1u] = sum;
	}
}

static inline uint32_t LmLatentShardDecodeSplits(uint32_t rows)
{
	return rows >= 16u ? 1u : 16u / rows;
}

static inline uint64_t LmLatentShardDecodeSplitFloats(uint32_t rows, uint32_t total_heads, uint32_t latent)
{
	return (uint64_t)LmLatentShardDecodeSplits(rows) * rows * total_heads * LM_LATENT_SHARD_RECORD_FLOATS(latent);
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline cudaError_t LmLatentShardSplitLaunch(
	Pages cache,
	const uint16_t *query_bf16,
	uint64_t query_rank_stride,
	uint32_t heads_per_rank,
	const uint32_t *sequence_of_row,
	const uint32_t *context_length,
	const uint32_t *row_position,
	float qk_scale,
	float *split_partials,
	uint64_t split_capacity_floats,
	float *partials,
	uint64_t partial_rank_stride,
	uint32_t rows,
	uint32_t splits,
	cudaStream_t stream)
{
	const uint32_t total_heads = heads_per_rank * cache.shard.degree;
	const uint64_t row_floats = (uint64_t)splits * total_heads * LM_LATENT_SHARD_RECORD_FLOATS(LATENT);
	uint32_t first, count, chunk;
	cudaError_t error;
	if (rows == 0u || splits == 0u || total_heads == 0u || total_heads % LM_LATENT_SHARD_QUERIES != 0u)
		return cudaErrorInvalidValue;
	if (splits == 1u)
		return LmLatentShardPrefillRowsLaunch<Geometry, Pages, LATENT, ROPE, 1u>(cache, query_bf16, query_rank_stride, heads_per_rank, total_heads,
			sequence_of_row, context_length, row_position, qk_scale, partials, partial_rank_stride, rows, 1u, 0u, stream, LmLatentShardDense(heads_per_rank));
	if (split_partials == 0 || split_capacity_floats < row_floats)
		return cudaErrorInvalidValue;
	chunk = split_capacity_floats / row_floats < rows ? (uint32_t)(split_capacity_floats / row_floats) : rows;
	for (first = 0u; first < rows; first += count)
	{
		count = rows - first < chunk ? rows - first : chunk;
		const uint64_t split_rank_stride = (uint64_t)count * heads_per_rank * LM_LATENT_SHARD_RECORD_FLOATS(LATENT);
		const uint64_t split_stride = (uint64_t)cache.shard.degree * split_rank_stride;
		const uint64_t records = (uint64_t)first * heads_per_rank;
		error = LmLatentShardPrefillRowsLaunch<Geometry, Pages, LATENT, ROPE, 1u>(cache, query_bf16 + records * (LATENT + ROPE), query_rank_stride, heads_per_rank, total_heads,
			sequence_of_row + first, context_length, row_position + first, qk_scale, split_partials, split_rank_stride, count, splits, split_stride, stream, LmLatentShardDense(heads_per_rank));
		if (error != cudaSuccess)
			return error;
		LmLatentShardSplitMergeKernel<LATENT><<<dim3(count, total_heads), LM_LATENT_SHARD_THREADS, 0, stream>>>(split_partials, split_rank_stride, split_stride, splits,
			partials + records * LM_LATENT_SHARD_RECORD_FLOATS(LATENT), partial_rank_stride, heads_per_rank);
		error = cudaPeekAtLastError();
		if (error != cudaSuccess)
			return error;
	}
	return cudaSuccess;
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline cudaError_t LmLatentShardDecodeLaunch(
	Pages cache,
	const uint16_t *query_bf16,
	uint64_t query_rank_stride,
	uint32_t heads_per_rank,
	const uint32_t *sequence_of_row,
	const uint32_t *context_length,
	const uint32_t *row_position,
	float qk_scale,
	float *split_partials,
	uint64_t split_capacity_floats,
	float *partials,
	uint64_t partial_rank_stride,
	uint32_t rows,
	cudaStream_t stream)
{
	const uint32_t total_heads = heads_per_rank * cache.shard.degree, splits = rows != 0u ? LmLatentShardDecodeSplits(rows) : 1u;
	if (rows == 0u || rows >= LM_LATENT_SHARD_PREFILL_MIN_ROWS || total_heads % LM_LATENT_SHARD_QUERIES != 0u)
		return cudaErrorInvalidValue;
	if (splits > 1u && (split_partials == 0 || split_capacity_floats < LmLatentShardDecodeSplitFloats(rows, total_heads, LATENT)))
		return cudaErrorInvalidValue;
	return LmLatentShardSplitLaunch<Geometry, Pages, LATENT, ROPE>(cache, query_bf16, query_rank_stride, heads_per_rank, sequence_of_row, context_length,
		row_position, qk_scale, split_partials, split_capacity_floats, partials, partial_rank_stride, rows, splits, stream);
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline cudaError_t LmLatentShardPrefillLaunch(
	Pages cache,
	const uint16_t *query_bf16,
	uint64_t query_rank_stride,
	uint32_t heads_per_rank,
	const uint32_t *sequence_of_row,
	const uint32_t *context_length,
	const uint32_t *row_position,
	float qk_scale,
	float *partials,
	uint64_t partial_rank_stride,
	uint32_t rows,
	cudaStream_t stream)
{
	if ((heads_per_rank * cache.shard.degree) % LM_LATENT_SHARD_QUERIES != 0u)
		return cudaErrorInvalidValue;
	return LmLatentShardPrefillRowsLaunch<Geometry, Pages, LATENT, ROPE, 1u>(cache, query_bf16, query_rank_stride, heads_per_rank,
		heads_per_rank * cache.shard.degree, sequence_of_row, context_length, row_position, qk_scale, partials, partial_rank_stride, rows, 1u, 0u, stream, LmLatentShardDense(heads_per_rank));
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline cudaError_t LmLatentGatherPrefillLaunch(
	Pages cache,
	const uint16_t *query_bf16,
	uint32_t heads_per_rank,
	const uint32_t *sequence_of_row,
	const uint32_t *context_length,
	const uint32_t *row_position,
	float qk_scale,
	float *partials,
	uint32_t rows,
	cudaStream_t stream)
{
	const uint64_t query_stride = (uint64_t)rows * heads_per_rank * (LATENT + ROPE);
	const uint64_t partial_stride = (uint64_t)rows * heads_per_rank * LM_LATENT_SHARD_RECORD_FLOATS(LATENT);
	if (heads_per_rank % 32u == 0u)
		return LmLatentShardPrefillRowsLaunch<Geometry, Pages, LATENT, ROPE, 1u>(cache, query_bf16, query_stride, heads_per_rank, heads_per_rank, sequence_of_row, context_length, row_position, qk_scale, partials, partial_stride, rows, 1u, 0u, stream, LmLatentShardDense(heads_per_rank));
	if (heads_per_rank % 16u == 0u)
		return LmLatentShardPrefillRowsLaunch<Geometry, Pages, LATENT, ROPE, 2u>(cache, query_bf16, query_stride, heads_per_rank, heads_per_rank, sequence_of_row, context_length, row_position, qk_scale, partials, partial_stride, rows, 1u, 0u, stream, LmLatentShardDense(heads_per_rank));
	if (heads_per_rank % 8u == 0u)
		return LmLatentShardPrefillRowsLaunch<Geometry, Pages, LATENT, ROPE, 4u>(cache, query_bf16, query_stride, heads_per_rank, heads_per_rank, sequence_of_row, context_length, row_position, qk_scale, partials, partial_stride, rows, 1u, 0u, stream, LmLatentShardDense(heads_per_rank));
	if (heads_per_rank % 4u == 0u)
		return LmLatentShardPrefillRowsLaunch<Geometry, Pages, LATENT, ROPE, 8u>(cache, query_bf16, query_stride, heads_per_rank, heads_per_rank, sequence_of_row, context_length, row_position, qk_scale, partials, partial_stride, rows, 1u, 0u, stream, LmLatentShardDense(heads_per_rank));
	if (heads_per_rank % 2u == 0u)
		return LmLatentShardPrefillRowsLaunch<Geometry, Pages, LATENT, ROPE, 16u>(cache, query_bf16, query_stride, heads_per_rank, heads_per_rank, sequence_of_row, context_length, row_position, qk_scale, partials, partial_stride, rows, 1u, 0u, stream, LmLatentShardDense(heads_per_rank));
	return LmLatentShardPrefillRowsLaunch<Geometry, Pages, LATENT, ROPE, 32u>(cache, query_bf16, query_stride, heads_per_rank, heads_per_rank, sequence_of_row, context_length, row_position, qk_scale, partials, partial_stride, rows, 1u, 0u, stream, LmLatentShardDense(heads_per_rank));
}

template<class Geometry, uint32_t LATENT, uint32_t ROPE>
static __host__ __device__ constexpr bool LmLatentShardTensorFits(void)
{
	return (LATENT + ROPE) % 64u == 0u && LATENT % (LM_PREFILL_ATTN_WARPS * 16u) == 0u && Geometry::kSlotBytes >= (LATENT + ROPE) * 2u &&
		LmLatentShardPrefillSharedBytes<LATENT + ROPE>() <= LM_PREFILL_ATTN_SHARED_LIMIT;
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline cudaError_t LmLatentShardAttendHeadsLaunch(
	Pages cache,
	const uint16_t *query_bf16,
	uint64_t query_rank_stride,
	uint32_t heads_per_rank,
	uint32_t total_heads,
	const uint32_t *sequence_of_row,
	const uint32_t *context_length,
	const uint32_t *row_position,
	const uint32_t *selected_positions,
	uint32_t selected_count,
	uint32_t dense_limit,
	float qk_scale,
	float *partials,
	uint64_t partial_rank_stride,
	uint32_t record_heads,
	uint32_t record_copies,
	uint32_t rows,
	cudaStream_t stream,
	const uint32_t *split_positions = 0,
	const uint32_t *split_offsets = 0,
	uint32_t heads_per_block_limit = 4u)
{
	if constexpr (LmLatentShardTensorFits<Geometry, LATENT, ROPE>())
	{
		const uint64_t list = selected_positions != 0 && split_positions == 0 ? LmLatentShardListBytes(selected_count) : 0u;
		const uint32_t padded = total_heads < LM_LATENT_SHARD_QUERIES ? total_heads : 0u;
		if ( (total_heads % LM_LATENT_SHARD_QUERIES == 0u || padded != 0u) && row_position != 0 && (partial_rank_stride % 2u) == 0u &&
			LmLatentShardPrefillSharedBytes<LATENT + ROPE>() + list <= LM_PREFILL_ATTN_SHARED_LIMIT)
		{
			const LmLatentShardSelection selection = { selected_positions, split_positions, split_offsets, selected_count, dense_limit, record_heads, record_copies, 1u, padded };
			return LmLatentShardPrefillRowsLaunch<Geometry, Pages, LATENT, ROPE, 1u>(cache, query_bf16, query_rank_stride, heads_per_rank, total_heads,
				sequence_of_row, context_length, row_position, qk_scale, partials, partial_rank_stride, rows, 1u, 0u, stream, selection);
		}
	}
	return LmLatentShardPartialHeadsLaunch<Geometry, Pages, LATENT, ROPE>(cache, query_bf16, query_rank_stride, heads_per_rank, total_heads, sequence_of_row,
		context_length, row_position, selected_positions, selected_count, dense_limit, qk_scale, partials, partial_rank_stride, record_heads, record_copies, rows,
		stream, split_positions, split_offsets, heads_per_block_limit);
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline cudaError_t LmLatentShardAttendLaunch(
	Pages cache,
	const uint16_t *query_bf16,
	uint64_t query_rank_stride,
	uint32_t heads_per_rank,
	const uint32_t *sequence_of_row,
	const uint32_t *context_length,
	const uint32_t *row_position,
	const uint32_t *selected_positions,
	uint32_t selected_count,
	uint32_t dense_limit,
	float qk_scale,
	float *partials,
	uint64_t partial_rank_stride,
	uint32_t rows,
	cudaStream_t stream)
{
	return LmLatentShardAttendHeadsLaunch<Geometry, Pages, LATENT, ROPE>(cache, query_bf16, query_rank_stride, heads_per_rank, heads_per_rank * cache.shard.degree,
		sequence_of_row, context_length, row_position, selected_positions, selected_count, dense_limit, qk_scale, partials, partial_rank_stride, heads_per_rank, 1u,
		rows, stream);
}
