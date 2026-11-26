#pragma once

#include "inference/kernels/attn_prefill.cuh"
#include "inference/kernels/attn_shard.cuh"
#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/tma.cuh"
#include "runtime/launch.h"
#include "sparkpipe/spark_gpu.h"

#define LM_LATENT_SHARD_PREFILL_MIN_ROWS 17u

#define LM_LATENT_SHARD_QUERIES 32u
#define LM_LATENT_SHARD_ROW_PAD 16u
#define LM_LATENT_SHARD_PREFILL_THREADS ((LM_PREFILL_ATTN_WARPS + 1u) * 32u)
#define LM_LATENT_SHARD_REG_STEPS 16u
#define LM_LATENT_SHARD_SMEM_STEPS_SPARSE 2u
#define LM_LATENT_SHARD_SMEM_BYTES_SPARSE (LM_LATENT_SHARD_SMEM_STEPS_SPARSE * 4u * 32u * 16u)
#define LM_LATENT_SHARD_SMEM_STEP_BYTES (4u * 32u * 16u)

typedef struct LmLatentShardSelection
{
	const uint32_t *selected_positions;
	const uint32_t *split_positions;
	const uint32_t *split_offsets;
	const uint32_t *fork_positions;
	const uint32_t *slot_offsets;
	uint32_t selected_count;
	uint32_t dense_limit;
	uint32_t record_heads;
	uint32_t record_copies;
	uint32_t empty_rows;
	uint32_t padded_heads;
} LmLatentShardSelection;

static __host__ __device__ inline LmLatentShardSelection LmLatentShardDense(
	uint32_t heads_per_rank,
	const uint32_t *fork_positions = 0,
	const uint32_t *slot_offsets = 0)
{
	LmLatentShardSelection selection = { 0, 0, 0, fork_positions, slot_offsets, 0u, 0u, heads_per_rank, 1u, 0u, 0u };
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
		+ 4u * LM_LATENT_SHARD_QUERIES * 4u
		+ 8u * 8u
		+ ((DIM / 32u) > LM_LATENT_SHARD_REG_STEPS ? LM_LATENT_SHARD_SMEM_BYTES_SPARSE : 0u));
}

template<uint32_t DIM>
static __host__ __device__ constexpr uint32_t LmLatentShardPrefillFullQueryBytes(void)
{
	return(2u * LM_PREFILL_ATTN_TILE * (DIM * 2u + LM_LATENT_SHARD_ROW_PAD)
		+ 2u * LM_LATENT_SHARD_QUERIES * LM_PREFILL_ATTN_SCORE_STRIDE * 4u
		+ LM_LATENT_SHARD_QUERIES * LM_PREFILL_ATTN_WEIGHT_STRIDE * 2u
		+ 4u * LM_LATENT_SHARD_QUERIES * 4u
		+ 8u * 8u
		+ ((DIM / 32u) > LM_LATENT_SHARD_REG_STEPS ? ((DIM / 32u) - LM_LATENT_SHARD_REG_STEPS) * LM_LATENT_SHARD_SMEM_STEP_BYTES : 0u));
}

#define LM_LATENT_SHARD_DEVICES 64u

static inline uint32_t LmLatentShardPrefillMultiprocessors(void)
{
	static uint32_t known[LM_LATENT_SHARD_DEVICES];
	int device = 0, count = 0;
	if (SparkGpuGetDevice(&device) != SPARK_GPU_SUCCESS || device < 0)
		return 1u;
	if ((uint32_t)device < LM_LATENT_SHARD_DEVICES && known[device] != 0u)
		return known[device];
	if (SparkGpuDeviceGetAttribute(&count, SPARK_GPU_ATTRIBUTE_MULTIPROCESSOR_COUNT, device) != SPARK_GPU_SUCCESS || count < 1)
		return 1u;
	if ((uint32_t)device < LM_LATENT_SHARD_DEVICES)
		known[device] = (uint32_t)count;
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

template<class Geometry>
static __device__ __forceinline__ void LmLatentShardSlotPrefetch(const LmKvShardGatherAllView &, uint32_t, uint32_t)
{
}

static __device__ __forceinline__ void LmLatentShardGroupBarrier(uint32_t name)
{
	asm volatile("bar.sync %0, %1;\n" :: "r"(name), "n"(LM_PREFILL_ATTN_THREADS / 2u) : "memory");
}

static __device__ __forceinline__ uint32_t LmLatentShardCompactWarp(
	SparkKvShard shard,
	const uint32_t *__restrict__ selected,
	uint32_t selected_count,
	uint32_t bound,
	uint32_t *list)
{
	constexpr uint32_t BATCH = 4u;
	const uint32_t lane = threadIdx.x & 31u, below = (1u << lane) - 1u;
	uint32_t base, item, count = 0u, keep, mask, value[BATCH], upcoming[BATCH];
#pragma unroll
	for (item = 0u; item < BATCH; item++)
		upcoming[item] = item * 32u + lane < selected_count ? __ldg(selected + item * 32u + lane) : 0xffffffffu;
	for (base = 0u; base < selected_count; base += BATCH * 32u)
	{
#pragma unroll
		for (item = 0u; item < BATCH; item++)
		{
			value[item] = upcoming[item];
			upcoming[item] = base + (BATCH + item) * 32u + lane < selected_count ? __ldg(selected + base + (BATCH + item) * 32u + lane) : 0xffffffffu;
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

template<uint32_t THREADS>
static __device__ __forceinline__ uint32_t LmLatentShardCompactBlock(
	SparkKvShard shard,
	const uint32_t *__restrict__ selected,
	uint32_t selected_count,
	uint32_t bound,
	uint32_t *list,
	uint32_t *tally)
{
	constexpr uint32_t WARPS = THREADS / 32u;
	constexpr uint32_t ROUNDS = (LM_LATENT_SHARD_LIST_CAPACITY + THREADS - 1u) / THREADS;
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, below = (1u << lane) - 1u;
	uint32_t round, count = 0u, keep, mask, before, total, other;
#pragma unroll 1
	for (round = 0u; round < ROUNDS; round++)
	{
		if (round * THREADS >= selected_count)
			break;
		const uint32_t val = round * THREADS + threadIdx.x < selected_count ? __ldg(selected + round * THREADS + threadIdx.x) : 0xffffffffu;
		keep = round * THREADS + threadIdx.x < selected_count && val <= bound && SparkKvShardOwns(shard, val) != 0u ? 1u : 0u;
		mask = __ballot_sync(0xffffffffu, keep != 0u);
		if (lane == 0u)
			tally[(round & 1u) * WARPS + warp] = (uint32_t)__popc(mask);
		__syncthreads();
		before = 0u;
		total = 0u;
#pragma unroll
		for (other = 0u; other < WARPS; other++)
		{
			before += other < warp ? tally[(round & 1u) * WARPS + other] : 0u;
			total += tally[(round & 1u) * WARPS + other];
		}
		if (keep != 0u)
			list[count + before + (uint32_t)__popc(mask & below)] = val;
		count += total;
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
	constexpr uint32_t GROUP_WARPS = LM_PREFILL_ATTN_WARPS / 2u;
	constexpr uint32_t WARP_COLUMNS = LATENT / GROUP_WARPS;
	constexpr uint32_t COLUMN_FRAGS = WARP_COLUMNS / 8u;
	constexpr uint32_t BLOCK_HEADS = QUERIES / ROWS;
	constexpr uint32_t QUERY_KEYS = QUERIES * TILE / LM_PREFILL_ATTN_THREADS;
	constexpr uint32_t QUERY_THREADS = TILE / QUERY_KEYS;
	constexpr uint32_t ISSUER = LM_PREFILL_ATTN_WARPS;
	constexpr uint32_t REG_STEPS = HALF_STEPS > LM_LATENT_SHARD_REG_STEPS ? LM_LATENT_SHARD_REG_STEPS : HALF_STEPS;
	constexpr uint32_t FULL_TAIL_STEPS = HALF_STEPS - REG_STEPS;
	constexpr uint32_t SPARSE_SMEM_STEPS = FULL_TAIL_STEPS < LM_LATENT_SHARD_SMEM_STEPS_SPARSE ? FULL_TAIL_STEPS : LM_LATENT_SHARD_SMEM_STEPS_SPARSE;
	static_assert(ROWS >= 1u && QUERIES % ROWS == 0u && ROWS <= 32u, "a block holds whole heads of each of its rows, one row per lane");
	static_assert(QUERIES == 32u && LM_PREFILL_ATTN_WARPS == 8u && TILE == 32u, "two 16-query tiles, a k-half and an 8-key pair per warp, one key per lane");
	static_assert(QUERY_KEYS == 4u && QUERY_THREADS == 8u, "eight threads hold a query's keys k, k+8, k+16 and k+24");
	static_assert(DIM % 64u == 0u && K_STEPS % 2u == 0u, "the query-key width must split into two halves of 16-wide steps");
	static_assert(LATENT % (GROUP_WARPS * 16u) == 0u, "each warp of a query group owns whole 16-column value slices");
	static_assert(Geometry::kSlotBytes >= ROW_BYTES, "the slot holds the latent and its rope part");
	static_assert(PITCH % 128u == 16u, "a padded row starts four banks after the row before it");
	static_assert(LmLatentShardPrefillSharedBytes<DIM>() <= LM_PREFILL_ATTN_SHARED_LIMIT, "the tiles exceed the shared memory an SM has");
	static_assert(LmLatentShardPrefillFullQueryBytes<DIM>() <= LM_PREFILL_ATTN_SHARED_LIMIT, "full query staging exceeds shared memory");
	extern __shared__ __align__(128) uint8_t lm_shard_prefill_shared[];
	uint8_t *keys = lm_shard_prefill_shared;
	float *score_parts = (float *)(keys + 2u * TILE * PITCH);
	uint16_t *weight_high = (uint16_t *)(score_parts + 2u * QUERIES * LM_PREFILL_ATTN_SCORE_STRIDE);
	float *running_max = (float *)(weight_high + QUERIES * LM_PREFILL_ATTN_WEIGHT_STRIDE);
	float *running_sum = running_max + QUERIES;
	float *rescale = running_sum + QUERIES;
	uint32_t *own_counts = (uint32_t *)(rescale + QUERIES);
	uint64_t *full = (uint64_t *)(own_counts + QUERIES);
	uint64_t *empty = full + 2u;
	uint32_t *meta = (uint32_t *)(empty + 2u);
	uint4 *query_smem = (uint4 *)(full + 8u);
	uint32_t *list_shared = (uint32_t *)((uint8_t *)query_smem + (HALF_STEPS > LM_LATENT_SHARD_REG_STEPS ? LM_LATENT_SHARD_SMEM_BYTES_SPARSE : 0u));
	LmDependentWait();
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t matrix = lane >> 3u, within = lane & 7u;
	const uint32_t own = threadIdx.x / QUERY_THREADS, part = threadIdx.x % QUERY_THREADS;
	const uint32_t group = blockIdx.x, block = blockIdx.y;
	const uint32_t row = block * ROWS, last_row = (row + ROWS < rows ? row + ROWS : rows) - 1u;
	const uint32_t key_base = (uint32_t)__cvta_generic_to_shared(keys);
	const uint32_t qk_tile = warp >> 2u, qk_half = (warp >> 1u) & 1u, qk_pair = warp & 1u, column_warp = warp % GROUP_WARPS;
	const uint32_t consumers = selection.padded_heads != 0u && selection.padded_heads <= 16u ? GROUP_WARPS : LM_PREFILL_ATTN_WARPS;
	uint32_t index, step, frag, q, head, query_row, m, shift, copy_index;
	uint64_t base;

	partials += (uint64_t)blockIdx.z * split_stride;
	uint32_t sequence = __ldg(sequence_of_row + row);
	for (query_row = row + 1u; query_row <= last_row && ROWS > 1u; query_row++)
		if (__ldg(sequence_of_row + query_row) != sequence)
			sequence = cache.pages.sequence_count;
	if (!LmKvViewIsConfigured(cache.pages) || sequence >= cache.pages.sequence_count)
	{
		if (threadIdx.x == 0u)
			LmKvReportRequiredAccessFailure(cache.pages, !LmKvViewIsConfigured(cache.pages) ? LM_KV_ACCESS_ERROR_INVALID_VIEW : LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE, LM_KV_ACCESS_READ, row, sequence_of_row[row], 0xffffffffu, 0xffffffffu);
		LmDependentRelease();
		return;
	}
	uint32_t listed = 0u, count = 0u, list_first = 0u;
	const uint32_t ctx_len = __ldg(context_length + sequence);
	if (ROWS == 1u && selection.selected_positions != 0)
	{
		const uint32_t rpos = __ldg(row_position + row);
		uint32_t dense = ctx_len;
		if (selection.empty_rows != 0u && rpos == 0xffffffffu)
			dense = 0u;
		else if (rpos < dense)
			dense = rpos + 1u;
		listed = dense > selection.dense_limit ? 1u : 0u;
	}
	if (listed != 0u && selection.split_positions != 0)
	{
		const uint32_t *offsets = selection.split_offsets + (uint64_t)row * (cache.shard.degree + 1u) + cache.shard.rank;
		list_first = __ldg(offsets);
		count = __ldg(offsets + 1u) - list_first;
	}
	else if (listed != 0u)
	{
		count = LmLatentShardCompactBlock<LM_LATENT_SHARD_PREFILL_THREADS>(cache.shard,
			selection.selected_positions + (uint64_t)row * selection.selected_count,
			selection.selected_count, __ldg(row_position + row), list_shared, list_shared + selection.selected_count);
	}
	if (warp == ISSUER)
	{
		const uint32_t mine = row + (lane < ROWS && row + lane <= last_row ? lane : 0u);
		const uint32_t lane_position = __ldg(row_position + mine);
		uint32_t lane_dense = ctx_len;
		if (selection.empty_rows != 0u && lane_position == 0xffffffffu)
			lane_dense = 0u;
		else if (lane_position < lane_dense)
			lane_dense = lane_position + 1u;
		const uint32_t lane_keys = lane < ROWS && row + lane <= last_row ? SparkKvShardLocalKeys(cache.shard, lane_dense) : 0u;
		uint32_t max_keys = count;
		if (listed == 0u)
		{
			max_keys = lane_keys;
#pragma unroll
			for (shift = 16u; shift > 0u; shift >>= 1u)
			{
				const uint32_t other = __shfl_xor_sync(0xffffffffu, max_keys, shift);
				max_keys = other > max_keys ? other : max_keys;
			}
		}
		const uint32_t win_end = listed != 0u ? max_keys : LmKvKeyWindowEnd(cache);
		const uint32_t lim = max_keys < win_end ? max_keys : win_end;
		uint32_t okeys = __shfl_sync(0xffffffffu, lane_keys, lane / BLOCK_HEADS);
		okeys = row + lane / BLOCK_HEADS > last_row ? 0u : listed != 0u ? max_keys : okeys;
		okeys = okeys < lim ? okeys : lim;
		own_counts[lane] = okeys;
		if (lane == 0u)
		{
			LmMbarrierInit(&full[0], 1u);
			LmMbarrierInit(&full[1], 1u);
			LmMbarrierInit(&empty[0], consumers);
			LmMbarrierInit(&empty[1], consumers);
			meta[0] = lim;
			LmMbarrierInitFence();
		}
	}
	__syncthreads();
	const uint32_t limit = meta[0];
	const uint32_t window_begin = listed != 0u ? 0u : LmKvKeyWindowBegin(cache);
	const uint32_t tile_low = window_begin / TILE;
	const uint32_t tiles = limit > window_begin ? (limit + TILE - 1u) / TILE : tile_low;
	const uint32_t span = tiles - tile_low;
	const uint32_t per_split = (span + split_count - 1u) / split_count;
	const uint32_t tile_first = tile_low + per_split * blockIdx.z < tiles ? tile_low + per_split * blockIdx.z : tiles;
	const uint32_t tile_end = tile_first + per_split < tiles ? tile_first + per_split : tiles;
	if (warp == ISSUER)
	{
		const uint32_t fork = selection.fork_positions != 0 ? __ldg(selection.fork_positions + row) : 0xffffffffu;
		const uint32_t offset = selection.slot_offsets != 0 ? __ldg(selection.slot_offsets + row) : 0u;
		const uint32_t *list_ptr = listed == 0u ? 0 : selection.split_positions != 0 ? selection.split_positions + (uint64_t)row * selection.selected_count + list_first : list_shared;
		uint32_t zeroed0 = 0u, zeroed1 = 0u, ahead = 0u;
		auto listing = [&](uint32_t which) -> uint32_t
		{
			const uint32_t local = which * TILE + lane;
			if (local >= limit || local < window_begin)
				return 0u;
			const uint32_t pos = listed == 0u ? SparkKvShardLocalPosition(cache.shard, local) : list_ptr[local];
			return pos >= fork ? pos + offset : pos;
		};
		if (tile_first < tile_end)
		{
			ahead = listing(tile_first);
			for (uint32_t used = 0u; tile_first + used < tile_end; used++)
			{
				const uint32_t tile = tile_first + used, buffer = used & 1u, uses = used >> 1u;
				const uint32_t local = tile * TILE + lane;
				const uint32_t position = ahead;
				if (tile + 1u < tile_end)
				{
					ahead = listing(tile + 1u);
					if ((tile + 1u) * TILE + lane < limit && (tile + 1u) * TILE + lane >= window_begin)
						LmLatentShardSlotPrefetch<Geometry>(cache, sequence, ahead);
				}
				const uint8_t *slot = (local < limit && local >= window_begin) ? LmKvShardSlotRequired<Geometry>(cache, sequence, position, row, LM_KV_ACCESS_READ) : 0;
				const uint32_t present = __ballot_sync(0xffffffffu, slot != 0);
				uint32_t clear = ~present & ~(buffer == 0u ? zeroed0 : zeroed1);
				if (uses > 0u)
					LmMbarrierWait(&empty[buffer], (uses - 1u) & 1u);
				(buffer == 0u ? zeroed0 : zeroed1) = ~present;
				if (slot != 0)
					LmTmaLoadBulk1d(keys + (buffer * TILE + lane) * PITCH, slot, &full[buffer], ROW_BYTES);
				const uint32_t had_clear = clear;
				for (; clear != 0u; clear &= clear - 1u)
					for (index = lane; index < CHUNKS; index += 32u)
						*(uint4 *)(keys + (buffer * TILE + (uint32_t)(__ffs(clear) - 1)) * PITCH + index * 16u) = make_uint4(0u, 0u, 0u, 0u);
				if (had_clear != 0u)
					LmMbarrierInitFence();
				__syncwarp();
				if (lane == 0u)
					LmMbarrierArriveExpect(&full[buffer], (uint32_t)__popc(present) * ROW_BYTES);
			}
		}
		LmDependentRelease();
		return;
	}
	if (warp >= consumers)
	{
		LmDependentRelease();
		return;
	}
	const uint32_t own_count = own_counts[own];
	const bool full_query_smem = selection.selected_positions == 0 || selection.split_positions != 0;
	uint32_t query_frag[REG_STEPS][4];
	const uint32_t *low_query = 0, *high_query = 0;
	float output[COLUMN_FRAGS][4];
	{
		const uint32_t low = qk_tile * 16u + (lane >> 2u), high = low + 8u;
		const uint32_t low_head = selection.padded_heads != 0u ? (low % BLOCK_HEADS) % selection.padded_heads : group * BLOCK_HEADS + low % BLOCK_HEADS;
		const uint32_t high_head = selection.padded_heads != 0u ? (high % BLOCK_HEADS) % selection.padded_heads : group * BLOCK_HEADS + high % BLOCK_HEADS;
		const uint32_t low_row = row + low / BLOCK_HEADS <= last_row ? row + low / BLOCK_HEADS : last_row;
		const uint32_t high_row = row + high / BLOCK_HEADS <= last_row ? row + high / BLOCK_HEADS : last_row;
		low_query = (const uint32_t *)(query_bf16 + (uint64_t)(low_head / heads_per_rank) * query_rank_stride +
			((uint64_t)low_row * heads_per_rank + low_head % heads_per_rank) * DIM) + qk_half * HALF_STEPS * 8u + (lane & 3u);
		high_query = (const uint32_t *)(query_bf16 + (uint64_t)(high_head / heads_per_rank) * query_rank_stride +
			((uint64_t)high_row * heads_per_rank + high_head % heads_per_rank) * DIM) + qk_half * HALF_STEPS * 8u + (lane & 3u);
#pragma unroll
		for (step = 0u; step < REG_STEPS; step++)
		{
			query_frag[step][0] = __ldg(low_query + step * 8u);
			query_frag[step][1] = __ldg(high_query + step * 8u);
			query_frag[step][2] = __ldg(low_query + step * 8u + 4u);
			query_frag[step][3] = __ldg(high_query + step * 8u + 4u);
		}
		if constexpr (FULL_TAIL_STEPS > 0u)
		{
			if (qk_pair == 0u)
			{
				const uint32_t smem_count = full_query_smem ? FULL_TAIL_STEPS : SPARSE_SMEM_STEPS;
				uint4 *dst = query_smem + (qk_tile * 2u + qk_half) * smem_count * 32u + lane;
				for (step = 0u; step < smem_count; step++)
				{
					const uint32_t s = REG_STEPS + step;
					dst[step * 32u] = make_uint4(__ldg(low_query + s * 8u), __ldg(high_query + s * 8u),
						__ldg(low_query + s * 8u + 4u), __ldg(high_query + s * 8u + 4u));
				}
			}
		}
	}
	if (threadIdx.x % (32u * GROUP_WARPS) < 16u)
	{
		running_max[qk_tile * 16u + threadIdx.x % (32u * GROUP_WARPS)] = -INFINITY;
		running_sum[qk_tile * 16u + threadIdx.x % (32u * GROUP_WARPS)] = 0.0f;
	}
#pragma unroll
	for (frag = 0u; frag < COLUMN_FRAGS; frag++)
		output[frag][0] = output[frag][1] = output[frag][2] = output[frag][3] = 0.0f;
	if constexpr (FULL_TAIL_STEPS > 0u)
		LmLatentShardGroupBarrier(1u + qk_tile);

#pragma unroll 1
	for (uint32_t used = 0u; tile_first + used < tile_end; used++)
	{
		const uint32_t tile = tile_first + used;
		const uint32_t buffer = used & 1u;
		const uint32_t tile_base = key_base + buffer * TILE * PITCH;
		LmMbarrierWait(&full[buffer], (used >> 1u) & 1u);
		{
			float score[2][4] = { { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } };
			uint32_t b[2][4], first[2], second[2];
			const uint32_t b_row = (qk_pair * 2u + (matrix >> 1u)) * 8u + within;
			const uint32_t b_row_base = tile_base + b_row * PITCH;
			LmPrefillLoadX4(b_row_base + ((qk_half * HALF_STEPS) * 2u + (matrix & 1u)) * 16u, b[0]);
#pragma unroll
			for (step = 0u; step < REG_STEPS; step++)
			{
				const uint32_t cur = step & 1u, nxt = cur ^ 1u;
				if (step + 1u < REG_STEPS)
					LmPrefillLoadX4(b_row_base + ((qk_half * HALF_STEPS + step + 1u) * 2u + (matrix & 1u)) * 16u, b[nxt]);
				first[0] = b[cur][0]; first[1] = b[cur][1];
				second[0] = b[cur][2]; second[1] = b[cur][3];
				LmMmaBf16(score[0], query_frag[step], first);
				LmMmaBf16(score[1], query_frag[step], second);
			}
			if constexpr (FULL_TAIL_STEPS > 0u)
			{
				if (full_query_smem)
				{
					const uint4 *src = query_smem + (qk_tile * 2u + qk_half) * FULL_TAIL_STEPS * 32u + lane;
					LmPrefillLoadX4(b_row_base + ((qk_half * HALF_STEPS + REG_STEPS) * 2u + (matrix & 1u)) * 16u, b[0]);
#pragma unroll
					for (step = 0u; step < FULL_TAIL_STEPS; step++)
					{
						const uint32_t cur = step & 1u, nxt = cur ^ 1u;
						const uint4 qv = src[step * 32u];
						uint32_t q_sm[4] = { qv.x, qv.y, qv.z, qv.w };
						if (step + 1u < FULL_TAIL_STEPS)
							LmPrefillLoadX4(b_row_base + ((qk_half * HALF_STEPS + REG_STEPS + step + 1u) * 2u + (matrix & 1u)) * 16u, b[nxt]);
						first[0] = b[cur][0]; first[1] = b[cur][1]; second[0] = b[cur][2]; second[1] = b[cur][3];
						LmMmaBf16(score[0], q_sm, first);
						LmMmaBf16(score[1], q_sm, second);
					}
				}
				else
				{
					const uint4 *src = query_smem + (qk_tile * 2u + qk_half) * SPARSE_SMEM_STEPS * 32u + lane;
					LmPrefillLoadX4(b_row_base + ((qk_half * HALF_STEPS + REG_STEPS) * 2u + (matrix & 1u)) * 16u, b[0]);
#pragma unroll
					for (step = 0u; step < SPARSE_SMEM_STEPS; step++)
					{
						const uint32_t cur = step & 1u, nxt = cur ^ 1u;
						const uint4 qv = src[step * 32u];
						uint32_t q_sm[4] = { qv.x, qv.y, qv.z, qv.w };
						if (step + 1u < SPARSE_SMEM_STEPS)
							LmPrefillLoadX4(b_row_base + ((qk_half * HALF_STEPS + REG_STEPS + step + 1u) * 2u + (matrix & 1u)) * 16u, b[nxt]);
						first[0] = b[cur][0]; first[1] = b[cur][1]; second[0] = b[cur][2]; second[1] = b[cur][3];
						LmMmaBf16(score[0], q_sm, first);
						LmMmaBf16(score[1], q_sm, second);
					}
#pragma unroll
					for (step = REG_STEPS + SPARSE_SMEM_STEPS; step < HALF_STEPS; step++)
					{
						const uint32_t b_chunk = (qk_half * HALF_STEPS + step) * 2u + (matrix & 1u);
						uint32_t q_tail[4] = {
							__ldg(low_query + step * 8u), __ldg(high_query + step * 8u),
							__ldg(low_query + step * 8u + 4u), __ldg(high_query + step * 8u + 4u)
						};
						LmPrefillLoadX4(b_row_base + b_chunk * 16u, b[0]);
						first[0] = b[0][0]; first[1] = b[0][1]; second[0] = b[0][2]; second[1] = b[0][3];
						LmMmaBf16(score[0], q_tail, first);
						LmMmaBf16(score[1], q_tail, second);
					}
				}
			}
#pragma unroll
			for (m = 0u; m < 2u; m++)
#pragma unroll
				for (index = 0u; index < 4u; index++)
					score_parts[(qk_half * QUERIES + qk_tile * 16u + (lane >> 2u) + (index >> 1u) * 8u) * LM_PREFILL_ATTN_SCORE_STRIDE + (qk_pair * 2u + m) * 8u + (lane & 3u) * 2u + (index & 1u)] = score[m][index];
		}
		LmLatentShardGroupBarrier(1u + qk_tile);
		{
			float value[QUERY_KEYS], weight[QUERY_KEYS], peak, previous_max, next_max, scale, total;
			const bool full_tile = tile * TILE >= window_begin && (tile + 1u) * TILE <= own_count;
#pragma unroll
			for (index = 0u; index < QUERY_KEYS; index++)
			{
				const uint32_t key = part + index * QUERY_THREADS, local = tile * TILE + key;
				value[index] = (score_parts[own * LM_PREFILL_ATTN_SCORE_STRIDE + key] + score_parts[(QUERIES + own) * LM_PREFILL_ATTN_SCORE_STRIDE + key]) * qk_scale;
				if (!full_tile)
					value[index] = local < own_count && local >= window_begin ? value[index] : -INFINITY;
			}
			peak = fmaxf(fmaxf(value[0], value[2]), fmaxf(value[1], value[3]));
#pragma unroll
			for (shift = QUERY_THREADS / 2u; shift > 0u; shift >>= 1u)
				peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, shift));
			previous_max = running_max[own];
			next_max = fmaxf(previous_max, peak);
			scale = previous_max == -INFINITY ? 0.0f : __expf(previous_max - next_max);
			if (full_tile)
			{
#pragma unroll
				for (index = 0u; index < QUERY_KEYS; index++)
				{
					weight[index] = __expf(value[index] - next_max);
					weight_high[own * LM_PREFILL_ATTN_WEIGHT_STRIDE + part + index * QUERY_THREADS] = LmFloatToBf16(weight[index]);
				}
			}
			else
			{
#pragma unroll
				for (index = 0u; index < QUERY_KEYS; index++)
				{
					weight[index] = next_max == -INFINITY || value[index] == -INFINITY ? 0.0f : __expf(value[index] - next_max);
					weight_high[own * LM_PREFILL_ATTN_WEIGHT_STRIDE + part + index * QUERY_THREADS] = LmFloatToBf16(weight[index]);
				}
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
		LmLatentShardGroupBarrier(1u + qk_tile);
		{
			const float low_scale = rescale[qk_tile * 16u + (lane >> 2u)], high_scale = rescale[qk_tile * 16u + (lane >> 2u) + 8u];
			uint32_t a_high[4], b[2][4], first[2], second[2];
			if (low_scale != 1.0f || high_scale != 1.0f)
			{
#pragma unroll
				for (frag = 0u; frag < COLUMN_FRAGS; frag++)
				{
					output[frag][0] *= low_scale;
					output[frag][1] *= low_scale;
					output[frag][2] *= high_scale;
					output[frag][3] *= high_scale;
				}
			}
#pragma unroll 1
			for (step = 0u; step < TILE / 16u; step++)
			{
				const uint32_t a_row = qk_tile * 16u + (matrix & 1u) * 8u + within, a_column = step * 16u + (matrix >> 1u) * 8u;
				const uint32_t b_row = step * 16u + (matrix & 1u) * 8u + within;
				const uint32_t b_row_base = tile_base + b_row * PITCH;
				LmPrefillLoadX4((uint32_t)__cvta_generic_to_shared(weight_high + a_row * LM_PREFILL_ATTN_WEIGHT_STRIDE + a_column), a_high);
				LmPrefillLoadX4Trans(b_row_base + ((column_warp * WARP_COLUMNS) / 8u + (matrix >> 1u)) * 16u, b[0]);
#pragma unroll
				for (frag = 0u; frag < COLUMN_FRAGS; frag += 2u)
				{
					const uint32_t cur = (frag >> 1u) & 1u, nxt = cur ^ 1u;
					if (frag + 2u < COLUMN_FRAGS)
						LmPrefillLoadX4Trans(b_row_base + ((column_warp * WARP_COLUMNS + (frag + 2u) * 8u) / 8u + (matrix >> 1u)) * 16u, b[nxt]);
					first[0] = b[cur][0]; first[1] = b[cur][1];
					second[0] = b[cur][2]; second[1] = b[cur][3];
					LmMmaBf16(output[frag], a_high, first);
					LmMmaBf16(output[frag + 1u], a_high, second);
				}
			}
		}
		LmSharedReadsDone();
		if (lane == 0u)
			LmMbarrierArrive(&empty[buffer]);
	}
#pragma unroll
	for (index = 0u; index < 4u; index += 2u)
	{
		q = qk_tile * 16u + (lane >> 2u) + (index >> 1u) * 8u;
		head = group * BLOCK_HEADS + q % BLOCK_HEADS;
		query_row = row + q / BLOCK_HEADS;
		if (query_row > last_row || (selection.padded_heads != 0u && q % BLOCK_HEADS >= selection.padded_heads))
			continue;
		for (copy_index = 0u; copy_index < selection.record_copies; copy_index++)
		{
			float *target = partials + ((uint64_t)(head / selection.record_heads) * selection.record_copies + copy_index) * partial_rank_stride +
				((uint64_t)query_row * selection.record_heads + head % selection.record_heads) * LM_LATENT_SHARD_RECORD_FLOATS(LATENT) + 2u + column_warp * WARP_COLUMNS + (lane & 3u) * 2u;
#pragma unroll
			for (frag = 0u; frag < COLUMN_FRAGS; frag++)
				__stcs((float2 *)(target + frag * 8u), make_float2(output[frag][index], output[frag][index + 1u]));
		}
	}
	q = qk_tile * 16u + threadIdx.x % (32u * GROUP_WARPS);
	if (threadIdx.x % (32u * GROUP_WARPS) < 16u && row + q / BLOCK_HEADS <= last_row &&
		(selection.padded_heads == 0u || q % BLOCK_HEADS < selection.padded_heads))
	{
		head = group * BLOCK_HEADS + q % BLOCK_HEADS;
		query_row = row + q / BLOCK_HEADS;
		for (copy_index = 0u; copy_index < selection.record_copies; copy_index++)
		{
			base = ((uint64_t)(head / selection.record_heads) * selection.record_copies + copy_index) * partial_rank_stride +
				((uint64_t)query_row * selection.record_heads + head % selection.record_heads) * LM_LATENT_SHARD_RECORD_FLOATS(LATENT);
			__stcs((float2 *)(partials + base), make_float2(running_max[q], running_sum[q]));
		}
	}
	LmDependentRelease();
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE, uint32_t ROWS>
static inline SparkGpuError LmLatentShardPrefillRowsLaunch(
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
	SparkGpuStream stream,
	LmLatentShardSelection selection)
{
	constexpr uint32_t block_heads = LM_LATENT_SHARD_QUERIES / ROWS;
	const bool unsplit_sparse = selection.selected_positions != 0 && selection.split_positions == 0;
	const uint64_t shared = unsplit_sparse
		? LmLatentShardPrefillSharedBytes<LATENT + ROPE>() + LmLatentShardListBytes(selection.selected_count)
		: LmLatentShardPrefillFullQueryBytes<LATENT + ROPE>();
	static bool granted = false;
	if (rows == 0u || heads_per_rank == 0u || query_bf16 == 0 || partials == 0 || row_position == 0 ||
		total_heads == 0u || (selection.padded_heads != 0u ? ROWS != 1u || selection.padded_heads != total_heads || total_heads >= block_heads :
			(total_heads % block_heads) != 0u) ||
		selection.record_heads == 0u || selection.record_copies == 0u || total_heads % selection.record_heads != 0u ||
		(selection.fork_positions == 0) != (selection.slot_offsets == 0) || (selection.fork_positions != 0 && ROWS != 1u) ||
		(selection.selected_positions != 0 && (ROWS != 1u || split_count != 1u || (selection.split_positions == 0) != (selection.split_offsets == 0))) ||
		shared > LM_PREFILL_ATTN_SHARED_LIMIT ||
		SparkKvShardValid(cache.shard, Geometry::kPageSlots) == 0u ||
		query_rank_stride < (uint64_t)rows * heads_per_rank * (LATENT + ROPE) ||
		partial_rank_stride < (uint64_t)rows * selection.record_heads * LM_LATENT_SHARD_RECORD_FLOATS(LATENT) ||
		(partial_rank_stride % 2u) != 0u)
		SPARK_FAIL(SPARK_GPU_ERROR_INVALID_VALUE);
	if (!granted)
	{
		SparkGpuError error = SparkGpuFuncSetAttribute((const void *)LmLatentShardPrefillKernel<Geometry, Pages, LATENT, ROPE, ROWS>, SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)LM_PREFILL_ATTN_SHARED_LIMIT);
		if (error != SPARK_GPU_SUCCESS)
			return error;
		granted = true;
	}
	const uint32_t blocks = (rows + ROWS - 1u) / ROWS, groups = selection.padded_heads != 0u ? 1u : total_heads / block_heads;
	LM_LAUNCH_DEPENDENT((LmLatentShardPrefillKernel<Geometry, Pages, LATENT, ROPE, ROWS>), dim3(groups, blocks, split_count), LM_LATENT_SHARD_PREFILL_THREADS, shared, stream,
		cache, query_bf16, query_rank_stride, heads_per_rank, sequence_of_row, context_length, row_position, qk_scale, partials, partial_rank_stride, rows,
		split_count, split_stride, selection);
	return SparkGpuPeekAtLastError();
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
	__shared__ float shared_weights[32];
	__shared__ float shared_max;
	LmDependentWait();
	const uint32_t row = blockIdx.x, head = blockIdx.y;
	const uint64_t record = ((uint64_t)row * heads_per_rank + head % heads_per_rank) * LM_LATENT_SHARD_RECORD_FLOATS(LATENT);
	const uint64_t source = (uint64_t)(head / heads_per_rank) * split_rank_stride + record;
	const uint64_t base = (uint64_t)(head / heads_per_rank) * partial_rank_stride + record;
	float maximum = -INFINITY, sum = 0.0f;
	uint32_t split, element;
	if (threadIdx.x == 0u)
	{
		for (split = 0u; split < split_count; split++)
			maximum = fmaxf(maximum, __ldg(split_partials + (uint64_t)split * split_stride + source));
		for (split = 0u; split < split_count; split++)
		{
			const float local = __ldg(split_partials + (uint64_t)split * split_stride + source);
			const float weight = maximum == -INFINITY || local == -INFINITY ? 0.0f : __expf(local - maximum);
			if (split < 32u)
				shared_weights[split] = weight;
			sum = fmaf(__ldg(split_partials + (uint64_t)split * split_stride + source + 1u), weight, sum);
		}
		shared_max = maximum;
		partials[base] = maximum;
		partials[base + 1u] = sum;
	}
	__syncthreads();
	maximum = shared_max;
	if (split_count <= 32u && (LATENT & 1u) == 0u && ((base + 2u) & 1u) == 0u && ((source + 2u) & 1u) == 0u && (split_stride & 1u) == 0u)
	{
		for (element = threadIdx.x; element < (LATENT >> 1u); element += LM_LATENT_SHARD_THREADS)
		{
			float v0 = 0.0f, v1 = 0.0f;
			for (split = 0u; split < split_count; split++)
			{
				const float weight = shared_weights[split];
				const float2 s2 = __ldg((const float2 *)(split_partials + (uint64_t)split * split_stride + source + 2u) + element);
				v0 = fmaf(s2.x, weight, v0);
				v1 = fmaf(s2.y, weight, v1);
			}
			*((float2 *)(partials + base + 2u) + element) = make_float2(v0, v1);
		}
	}
	else
	{
		for (element = threadIdx.x; element < LATENT; element += LM_LATENT_SHARD_THREADS)
		{
			float value = 0.0f;
			for (split = 0u; split < split_count; split++)
			{
				const float local = __ldg(split_partials + (uint64_t)split * split_stride + source);
				const float weight = split < 32u ? shared_weights[split] : (maximum == -INFINITY || local == -INFINITY ? 0.0f : __expf(local - maximum));
				value = fmaf(__ldg(split_partials + (uint64_t)split * split_stride + source + 2u + element), weight, value);
			}
			partials[base + 2u + element] = value;
		}
	}
	LmDependentRelease();
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
static inline SparkGpuError LmLatentShardSplitLaunch(
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
	SparkGpuStream stream,
	const uint32_t *fork_positions = 0,
	const uint32_t *slot_offsets = 0)
{
	const uint32_t total_heads = heads_per_rank * cache.shard.degree;
	const uint64_t row_floats = (uint64_t)splits * total_heads * LM_LATENT_SHARD_RECORD_FLOATS(LATENT);
	uint32_t first, count, chunk;
	SparkGpuError error;
	if (rows == 0u || splits == 0u || total_heads == 0u || total_heads % LM_LATENT_SHARD_QUERIES != 0u)
		SPARK_FAIL(SPARK_GPU_ERROR_INVALID_VALUE);
	if (splits == 1u)
		return LmLatentShardPrefillRowsLaunch<Geometry, Pages, LATENT, ROPE, 1u>(cache, query_bf16, query_rank_stride, heads_per_rank, total_heads,
			sequence_of_row, context_length, row_position, qk_scale, partials, partial_rank_stride, rows, 1u, 0u, stream,
			LmLatentShardDense(heads_per_rank, fork_positions, slot_offsets));
	if (split_partials == 0 || split_capacity_floats < row_floats)
		SPARK_FAIL(SPARK_GPU_ERROR_INVALID_VALUE);
	chunk = split_capacity_floats / row_floats < rows ? (uint32_t)(split_capacity_floats / row_floats) : rows;
	for (first = 0u; first < rows; first += count)
	{
		count = rows - first < chunk ? rows - first : chunk;
		const uint64_t split_rank_stride = (uint64_t)count * heads_per_rank * LM_LATENT_SHARD_RECORD_FLOATS(LATENT);
		const uint64_t split_stride = (uint64_t)cache.shard.degree * split_rank_stride;
		const uint64_t records = (uint64_t)first * heads_per_rank;
		error = LmLatentShardPrefillRowsLaunch<Geometry, Pages, LATENT, ROPE, 1u>(cache, query_bf16 + records * (LATENT + ROPE), query_rank_stride, heads_per_rank, total_heads,
			sequence_of_row + first, context_length, row_position + first, qk_scale, split_partials, split_rank_stride, count, splits, split_stride, stream,
			LmLatentShardDense(heads_per_rank, fork_positions != 0 ? fork_positions + first : 0, slot_offsets != 0 ? slot_offsets + first : 0));
		if (error != SPARK_GPU_SUCCESS)
			return error;
		LM_LAUNCH_DEPENDENT((LmLatentShardSplitMergeKernel<LATENT>), dim3(count, total_heads), LM_LATENT_SHARD_THREADS, 0, stream, split_partials, split_rank_stride, split_stride, splits,
			partials + records * LM_LATENT_SHARD_RECORD_FLOATS(LATENT), partial_rank_stride, heads_per_rank);
		error = SparkGpuPeekAtLastError();
		if (error != SPARK_GPU_SUCCESS)
			return error;
	}
	return SPARK_GPU_SUCCESS;
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline SparkGpuError LmLatentShardDecodeLaunch(
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
	SparkGpuStream stream)
{
	const uint32_t total_heads = heads_per_rank * cache.shard.degree, splits = rows != 0u ? LmLatentShardDecodeSplits(rows) : 1u;
	if (rows == 0u || rows >= LM_LATENT_SHARD_PREFILL_MIN_ROWS || total_heads % LM_LATENT_SHARD_QUERIES != 0u)
		SPARK_FAIL(SPARK_GPU_ERROR_INVALID_VALUE);
	if (splits > 1u && (split_partials == 0 || split_capacity_floats < LmLatentShardDecodeSplitFloats(rows, total_heads, LATENT)))
		SPARK_FAIL(SPARK_GPU_ERROR_INVALID_VALUE);
	return LmLatentShardSplitLaunch<Geometry, Pages, LATENT, ROPE>(cache, query_bf16, query_rank_stride, heads_per_rank, sequence_of_row, context_length,
		row_position, qk_scale, split_partials, split_capacity_floats, partials, partial_rank_stride, rows, splits, stream);
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline SparkGpuError LmLatentShardPrefillLaunch(
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
	SparkGpuStream stream)
{
	if ((heads_per_rank * cache.shard.degree) % LM_LATENT_SHARD_QUERIES != 0u)
		SPARK_FAIL(SPARK_GPU_ERROR_INVALID_VALUE);
	return LmLatentShardPrefillRowsLaunch<Geometry, Pages, LATENT, ROPE, 1u>(cache, query_bf16, query_rank_stride, heads_per_rank,
		heads_per_rank * cache.shard.degree, sequence_of_row, context_length, row_position, qk_scale, partials, partial_rank_stride, rows, 1u, 0u, stream, LmLatentShardDense(heads_per_rank));
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline SparkGpuError LmLatentGatherPrefillLaunch(
	Pages cache,
	const uint16_t *query_bf16,
	uint32_t heads_per_rank,
	const uint32_t *sequence_of_row,
	const uint32_t *context_length,
	const uint32_t *row_position,
	float qk_scale,
	float *partials,
	uint32_t rows,
	SparkGpuStream stream)
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
static inline SparkGpuError LmLatentShardAttendHeadsLaunch(
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
	SparkGpuStream stream,
	const uint32_t *split_positions = 0,
	const uint32_t *split_offsets = 0,
	uint32_t heads_per_block_limit = 4u,
	const uint32_t *fork_positions = 0,
	const uint32_t *slot_offsets = 0)
{
	static_assert(LmLatentShardTensorFits<Geometry, LATENT, ROPE>(), "shard prefill attention runs on the tensor-core kernel: the latent plus rope width must be "
		"a multiple of 64, the latent a multiple of the warps' 16-wide tiles, the KV slot must hold the bf16 key row, and the tiles must fit shared memory");
	const uint64_t list = selected_positions != 0 && split_positions == 0 ? LmLatentShardListBytes(selected_count) : 0u;
	const uint32_t padded = total_heads < LM_LATENT_SHARD_QUERIES ? total_heads : 0u;
	(void)heads_per_block_limit;
	if ( total_heads % LM_LATENT_SHARD_QUERIES != 0u && padded == 0u )
		LM_GPU_REFUSE("shard prefill attention: %u heads over all ranks; the tensor-core kernel takes fewer than %u or a multiple of %u\n", total_heads,
			LM_LATENT_SHARD_QUERIES, LM_LATENT_SHARD_QUERIES);
	if ( row_position == 0 || (partial_rank_stride % 2u) != 0u )
		LM_GPU_REFUSE("shard prefill attention: the tensor-core kernel needs each row's position and an even partial stride (got %llu)\n",
			(unsigned long long)partial_rank_stride);
	if ( LmLatentShardPrefillSharedBytes<LATENT + ROPE>() + list > LM_PREFILL_ATTN_SHARED_LIMIT )
		LM_GPU_REFUSE("shard prefill attention: %u selected positions need %llu list bytes beside %u tile bytes, past the %u shared bytes an SM has; "
			"select at most %llu positions or pass them split\n", selected_count, (unsigned long long)list, LmLatentShardPrefillSharedBytes<LATENT + ROPE>(),
			LM_PREFILL_ATTN_SHARED_LIMIT, (unsigned long long)((LM_PREFILL_ATTN_SHARED_LIMIT - LmLatentShardPrefillSharedBytes<LATENT + ROPE>()) / sizeof(uint32_t)) -
			LM_LATENT_SHARD_THREADS);
	const LmLatentShardSelection selection = { selected_positions, split_positions, split_offsets, fork_positions, slot_offsets, selected_count, dense_limit, record_heads, record_copies, 1u, padded };
	return LmLatentShardPrefillRowsLaunch<Geometry, Pages, LATENT, ROPE, 1u>(cache, query_bf16, query_rank_stride, heads_per_rank, total_heads,
		sequence_of_row, context_length, row_position, qk_scale, partials, partial_rank_stride, rows, 1u, 0u, stream, selection);
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline SparkGpuError LmLatentShardAttendLaunch(
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
	SparkGpuStream stream,
	const uint32_t *fork_positions = 0,
	const uint32_t *slot_offsets = 0)
{
	return LmLatentShardAttendHeadsLaunch<Geometry, Pages, LATENT, ROPE>(cache, query_bf16, query_rank_stride, heads_per_rank, heads_per_rank * cache.shard.degree,
		sequence_of_row, context_length, row_position, selected_positions, selected_count, dense_limit, qk_scale, partials, partial_rank_stride, heads_per_rank, 1u,
		rows, stream, (const uint32_t *)0, (const uint32_t *)0, 4u, fork_positions, slot_offsets);
}
