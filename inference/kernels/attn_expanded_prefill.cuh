#pragma once

#include "inference/kernels/attn_shard_prefill.cuh"

#define LM_EXPANDED_PREFILL_THREADS 256u
#define LM_EXPANDED_PREFILL_WARPS 8u
#define LM_EXPANDED_PREFILL_ROWS 32u
#define LM_EXPANDED_PREFILL_HEADS 2u
#define LM_EXPANDED_PREFILL_TILE 16u
#define LM_EXPANDED_PREFILL_STAGES 3u
#define LM_EXPANDED_PREFILL_EXCHANGE_FLOATS 8u

#define LM_LATENT_EXPAND_THREADS 256u
#define LM_LATENT_EXPAND_KEYS 128u
#define LM_LATENT_EXPAND_CHUNK 64u
#define LM_LATENT_EXPAND_STAGES 3u

#define LM_EXPANDED_PREFILL_PAD 16u

template<uint32_t DIM, uint32_t VALUE>
static __host__ __device__ constexpr uint32_t LmExpandedPrefillStageBytes(void)
{
	return(LM_EXPANDED_PREFILL_TILE * (DIM * 2u + LM_EXPANDED_PREFILL_PAD)
		+ LM_EXPANDED_PREFILL_HEADS * LM_EXPANDED_PREFILL_TILE * (VALUE * 2u + LM_EXPANDED_PREFILL_PAD));
}

template<uint32_t DIM, uint32_t VALUE>
static __host__ __device__ constexpr uint32_t LmExpandedPrefillSharedBytes(void)
{
	return(LM_EXPANDED_PREFILL_STAGES * LmExpandedPrefillStageBytes<DIM, VALUE>()
		+ 2u * LM_EXPANDED_PREFILL_WARPS * 32u * LM_EXPANDED_PREFILL_EXCHANGE_FLOATS * 4u);
}

template<uint32_t VALUE>
static __host__ __device__ constexpr uint32_t LmLatentExpandSharedBytes(void)
{
	return(LM_LATENT_EXPAND_STAGES * (LM_LATENT_EXPAND_KEYS + VALUE) * LM_LATENT_EXPAND_CHUNK * 2u);
}

static __device__ __forceinline__ uint32_t LmExpandedPackBf16(uint16_t low, uint16_t high)
{
	return((uint32_t)low | ((uint32_t)high << 16u));
}

static __device__ __forceinline__ void LmExpandedNamedBarrier(uint32_t name, uint32_t threads)
{
	asm volatile("bar.sync %0, %1;\n" :: "r"(name), "r"(threads) : "memory");
}

template<uint32_t LATENT, uint32_t ROW_DIM, uint32_t VALUE>
__global__ __launch_bounds__(LM_LATENT_EXPAND_THREADS, 1)
void LmLatentValueExpandKernel(
	const uint8_t *__restrict__ keys,
	uint64_t key_source_stride,
	uint32_t keys_per_source,
	const uint16_t *__restrict__ weight_bf16,
	uint32_t heads,
	uint8_t *__restrict__ values,
	uint64_t value_source_stride)
{
	constexpr uint32_t ROW_BYTES = ROW_DIM * 2u;
	constexpr uint32_t CHUNKS = LM_LATENT_EXPAND_CHUNK / 8u;
	constexpr uint32_t STEPS = LM_LATENT_EXPAND_CHUNK / 16u;
	constexpr uint32_t ROUNDS = LATENT / LM_LATENT_EXPAND_CHUNK;
	constexpr uint32_t FRAGS = VALUE / 8u;
	constexpr uint32_t A_BYTES = LM_LATENT_EXPAND_KEYS * LM_LATENT_EXPAND_CHUNK * 2u;
	constexpr uint32_t STAGE_BYTES = (LM_LATENT_EXPAND_KEYS + VALUE) * LM_LATENT_EXPAND_CHUNK * 2u;
	static_assert(LATENT % LM_LATENT_EXPAND_CHUNK == 0u && ROUNDS >= LM_LATENT_EXPAND_STAGES, "the latent splits into pipelined chunks");
	static_assert(LM_LATENT_EXPAND_KEYS == 16u * (LM_LATENT_EXPAND_THREADS / 32u), "each warp expands sixteen keys");
	static_assert(VALUE % 16u == 0u && CHUNKS == 8u, "whole value fragment pairs and eight swizzled chunks per row");
	static_assert(LATENT <= ROW_DIM, "the latent sits at the front of each key row");
	static_assert(LmLatentExpandSharedBytes<VALUE>() <= LM_PREFILL_ATTN_SHARED_LIMIT, "the expansion tiles exceed the shared memory an SM has");
	extern __shared__ __align__(128) uint8_t lm_expand_shared[];
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t matrix = lane >> 3u, within = lane & 7u;
	const uint32_t first_key = blockIdx.x * LM_LATENT_EXPAND_KEYS, head = blockIdx.y, source = blockIdx.z;
	const uint32_t base = (uint32_t)__cvta_generic_to_shared(lm_expand_shared);
	const uint8_t *source_keys = keys + (uint64_t)source * key_source_stride;
	const uint16_t *head_weight = weight_bf16 + (uint64_t)head * VALUE * LATENT;
	float output[FRAGS][4];
	uint32_t round, step, frag, item, row, chunk, a[4], b[4];

	if (first_key >= keys_per_source || head >= heads)
		return;
	auto load = [&](uint32_t which)
	{
		const uint32_t stage = base + (which % LM_LATENT_EXPAND_STAGES) * STAGE_BYTES;
		for (item = threadIdx.x; item < (LM_LATENT_EXPAND_KEYS + VALUE) * CHUNKS; item += LM_LATENT_EXPAND_THREADS)
		{
			row = item / CHUNKS;
			chunk = item % CHUNKS;
			if (row < LM_LATENT_EXPAND_KEYS)
			{
				const uint32_t key = first_key + row;
				LmPrefillCopy(stage + row * LM_LATENT_EXPAND_CHUNK * 2u + LmPrefillSwizzle(row,chunk) * 16u,
					key < keys_per_source ? (const void *)(source_keys + (uint64_t)key * ROW_BYTES + (which * LM_LATENT_EXPAND_CHUNK + chunk * 8u) * 2u) : (const void *)source_keys,
					key < keys_per_source ? 16u : 0u);
			}
			else
				LmPrefillCopy(stage + A_BYTES + (row - LM_LATENT_EXPAND_KEYS) * LM_LATENT_EXPAND_CHUNK * 2u + LmPrefillSwizzle(row - LM_LATENT_EXPAND_KEYS,chunk) * 16u,
					head_weight + (uint64_t)(row - LM_LATENT_EXPAND_KEYS) * LATENT + which * LM_LATENT_EXPAND_CHUNK + chunk * 8u, 16u);
		}
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	};
#pragma unroll
	for (frag = 0u; frag < FRAGS; frag++)
		output[frag][0] = output[frag][1] = output[frag][2] = output[frag][3] = 0.0f;
	for (round = 0u; round + 1u < LM_LATENT_EXPAND_STAGES; round++)
		load(round);
	for (round = 0u; round < ROUNDS; round++)
	{
		asm volatile("cp.async.wait_group %0;\n" :: "n"(LM_LATENT_EXPAND_STAGES - 2u) : "memory");
		__syncthreads();
		if (round + LM_LATENT_EXPAND_STAGES - 1u < ROUNDS)
			load(round + LM_LATENT_EXPAND_STAGES - 1u);
		else
			asm volatile("cp.async.commit_group;\n" ::: "memory");
		const uint32_t stage = base + (round % LM_LATENT_EXPAND_STAGES) * STAGE_BYTES;
#pragma unroll
		for (step = 0u; step < STEPS; step++)
		{
			const uint32_t a_row = warp * 16u + (matrix & 1u) * 8u + within;
			LmPrefillLoadX4(stage + a_row * LM_LATENT_EXPAND_CHUNK * 2u + LmPrefillSwizzle(a_row,step * 2u + (matrix >> 1u)) * 16u, a);
#pragma unroll
			for (frag = 0u; frag < FRAGS; frag += 2u)
			{
				const uint32_t b_row = frag * 8u + (matrix >> 1u) * 8u + within;
				uint32_t first[2], second[2];
				LmPrefillLoadX4(stage + A_BYTES + b_row * LM_LATENT_EXPAND_CHUNK * 2u + LmPrefillSwizzle(b_row,step * 2u + (matrix & 1u)) * 16u, b);
				first[0] = b[0]; first[1] = b[1];
				second[0] = b[2]; second[1] = b[3];
				LmMmaBf16(output[frag], a, first);
				LmMmaBf16(output[frag + 1u], a, second);
			}
		}
	}
	{
		const uint32_t key_low = first_key + warp * 16u + (lane >> 2u), key_high = key_low + 8u;
		uint8_t *target = values + (uint64_t)source * value_source_stride;
#pragma unroll
		for (frag = 0u; frag < FRAGS; frag++)
		{
			const uint32_t column = frag * 8u + (lane & 3u) * 2u;
			if (key_low < keys_per_source)
				*(uint32_t *)(target + (((uint64_t)key_low * heads + head) * VALUE + column) * 2u) =
					LmExpandedPackBf16(LmFloatToBf16(output[frag][0]), LmFloatToBf16(output[frag][1]));
			if (key_high < keys_per_source)
				*(uint32_t *)(target + (((uint64_t)key_high * heads + head) * VALUE + column) * 2u) =
					LmExpandedPackBf16(LmFloatToBf16(output[frag][2]), LmFloatToBf16(output[frag][3]));
		}
	}
}

template<uint32_t LATENT, uint32_t ROW_DIM, uint32_t VALUE>
static inline cudaError_t LmLatentValueExpandLaunch(
	const uint8_t *keys,
	uint64_t key_source_stride,
	uint32_t sources,
	uint32_t keys_per_source,
	const uint16_t *weight_bf16,
	uint32_t heads,
	uint8_t *values,
	uint64_t value_source_stride,
	cudaStream_t stream)
{
	constexpr uint32_t shared = LmLatentExpandSharedBytes<VALUE>();
	static bool granted = false;
	if (keys == 0 || weight_bf16 == 0 || values == 0 || sources == 0u || heads == 0u ||
		(key_source_stride % 16u) != 0u || (value_source_stride % 16u) != 0u ||
		(sources > 1u && (key_source_stride < (uint64_t)keys_per_source * ROW_DIM * 2u ||
			value_source_stride < (uint64_t)keys_per_source * heads * VALUE * 2u)))
		return cudaErrorInvalidValue;
	if (keys_per_source == 0u)
		return cudaSuccess;
	if (!granted)
	{
		cudaError_t error = cudaFuncSetAttribute((const void *)LmLatentValueExpandKernel<LATENT, ROW_DIM, VALUE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared);
		if (error != cudaSuccess)
			return error;
		granted = true;
	}
	LmLatentValueExpandKernel<LATENT, ROW_DIM, VALUE><<<dim3((keys_per_source + LM_LATENT_EXPAND_KEYS - 1u) / LM_LATENT_EXPAND_KEYS, heads, sources), LM_LATENT_EXPAND_THREADS, shared, stream>>>(
		keys, key_source_stride, keys_per_source, weight_bf16, heads, values, value_source_stride);
	return cudaPeekAtLastError();
}

template<class Geometry, uint32_t LATENT, uint32_t ROPE, uint32_t VALUE>
__global__ __launch_bounds__(LM_EXPANDED_PREFILL_THREADS, 1)
void LmExpandedGatherPrefillKernel(
	LmKvShardGatherView cache,
	const uint8_t *__restrict__ values,
	const uint16_t *__restrict__ query_bf16,
	uint32_t heads_per_rank,
	const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ context_length,
	const uint32_t *__restrict__ row_position,
	float qk_scale,
	float *__restrict__ partials,
	uint32_t rows)
{
	constexpr uint32_t DIM = LATENT + ROPE;
	constexpr uint32_t ROW_BYTES = DIM * 2u;
	constexpr uint32_t VALUE_BYTES = VALUE * 2u;
	constexpr uint32_t CHUNKS = DIM / 8u;
	constexpr uint32_t VALUE_CHUNKS = VALUE / 8u;
	constexpr uint32_t TILE = LM_EXPANDED_PREFILL_TILE;
	constexpr uint32_t STAGES = LM_EXPANDED_PREFILL_STAGES;
	constexpr uint32_t QUERIES = LM_EXPANDED_PREFILL_ROWS * LM_EXPANDED_PREFILL_HEADS;
	constexpr uint32_t K_STEPS = DIM / 16u;
	constexpr uint32_t HALF_STEPS = K_STEPS / 2u;
	constexpr uint32_t HALF_COLUMNS = VALUE / 2u;
	constexpr uint32_t FRAGS = HALF_COLUMNS / 8u;
	constexpr uint32_t KEY_PITCH = ROW_BYTES + LM_EXPANDED_PREFILL_PAD;
	constexpr uint32_t VALUE_PITCH = VALUE_BYTES + LM_EXPANDED_PREFILL_PAD;
	constexpr uint32_t KEY_BYTES = TILE * KEY_PITCH;
	constexpr uint32_t STAGE_BYTES = LmExpandedPrefillStageBytes<DIM, VALUE>();
	constexpr uint32_t KEY_ITEMS = TILE * CHUNKS;
	constexpr uint32_t STAGE_ITEMS = KEY_ITEMS + LM_EXPANDED_PREFILL_HEADS * TILE * VALUE_CHUNKS;
	constexpr uint32_t ROUNDS = (STAGE_ITEMS + LM_EXPANDED_PREFILL_THREADS - 1u) / LM_EXPANDED_PREFILL_THREADS;
	constexpr uint32_t RECORD = LM_LATENT_SHARD_RECORD_FLOATS(VALUE);
	static_assert(LM_EXPANDED_PREFILL_WARPS * 32u == LM_EXPANDED_PREFILL_THREADS && LM_EXPANDED_PREFILL_WARPS == 2u * QUERIES / 16u,
		"each sixteen-query tile is shared by two warps, one per half of the query-key width");
	static_assert(TILE == 16u && LM_EXPANDED_PREFILL_ROWS == 32u && LM_EXPANDED_PREFILL_HEADS == 2u, "a tile is one 16-key fragment pair; a block holds two heads of 32 rows");
	static_assert(DIM % 32u == 0u && K_STEPS % 2u == 0u && CHUNKS % 8u == 0u, "the query-key width splits into two halves of 16-wide steps");
	static_assert(VALUE % 32u == 0u && FRAGS % 2u == 0u && VALUE_CHUNKS % 8u == 0u, "each warp owns whole 16-column value slices");
	static_assert((KEY_PITCH / 4u) % 32u == 4u && (VALUE_PITCH / 4u) % 32u == 4u, "padded rows put eight consecutive rows on distinct banks");
	static_assert(ROUNDS <= HALF_STEPS, "the next tile's copies spread over the query-key steps");
	static_assert(Geometry::kSlotBytes >= ROW_BYTES, "the slot holds the latent and its rope part");
	static_assert(QUERIES * ROW_BYTES <= STAGES * STAGE_BYTES, "the queries stage through the key buffers before the pipeline starts");
	static_assert(LmExpandedPrefillSharedBytes<DIM, VALUE>() <= LM_PREFILL_ATTN_SHARED_LIMIT, "the tiles exceed the shared memory an SM has");
	extern __shared__ __align__(128) uint8_t lm_expanded_shared[];
	float *exchange = (float *)(lm_expanded_shared + STAGES * STAGE_BYTES);
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t matrix = lane >> 3u, within = lane & 7u;
	const uint32_t tile_id = warp >> 1u, half = warp & 1u, local_head = tile_id >> 1u;
	const uint32_t row = blockIdx.x * LM_EXPANDED_PREFILL_ROWS, head = blockIdx.y * LM_EXPANDED_PREFILL_HEADS;
	const uint32_t last_row = (row + LM_EXPANDED_PREFILL_ROWS < rows ? row + LM_EXPANDED_PREFILL_ROWS : rows) - 1u;
	const uint32_t base = (uint32_t)__cvta_generic_to_shared(lm_expanded_shared);
	const uint32_t row_low = row + (tile_id & 1u) * 16u + (lane >> 2u), row_high = row_low + 8u;
	const uint64_t value_row_bytes = (uint64_t)heads_per_rank * VALUE_BYTES;
	uint32_t query_frag[HALF_STEPS][4];
	float output[FRAGS][4];
	float max_low = -INFINITY, max_high = -INFINITY, sum_low = 0.0f, sum_high = 0.0f;
	uint32_t sequence, dense, count, count_low, count_high, window_begin, limit, available, tile_first, tile_end, tile, index, chunk, step, frag, q, query_row;
	uint64_t key_origin;

	sequence = sequence_of_row[row];
	for (query_row = row + 1u; query_row <= last_row; query_row++)
		if (sequence_of_row[query_row] != sequence)
			sequence = cache.pages.sequence_count;
	if (!LmKvViewIsConfigured(cache.pages) || sequence >= cache.pages.sequence_count)
	{
		if (threadIdx.x == 0u)
			LmKvReportRequiredAccessFailure(cache.pages, !LmKvViewIsConfigured(cache.pages) ? LM_KV_ACCESS_ERROR_INVALID_VIEW : LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE, LM_KV_ACCESS_READ, row, sequence_of_row[row], 0xffffffffu, 0xffffffffu);
		return;
	}
	count = 0u;
	for (query_row = row; query_row <= last_row; query_row++)
	{
		dense = context_length[sequence];
		if (row_position[query_row] < dense)
			dense = row_position[query_row] + 1u;
		dense = SparkKvShardLocalKeys(cache.shard, dense);
		count = dense > count ? dense : count;
	}
	dense = context_length[sequence];
	count_low = row_low <= last_row ? SparkKvShardLocalKeys(cache.shard, row_position[row_low] < dense ? row_position[row_low] + 1u : dense) : 0u;
	count_high = row_high <= last_row ? SparkKvShardLocalKeys(cache.shard, row_position[row_high] < dense ? row_position[row_high] + 1u : dense) : 0u;
	window_begin = cache.key_begin;
	limit = count < cache.key_end ? count : cache.key_end;
	available = SparkKvShardLocalKeys(cache.shard, cache.key_context[sequence]);
	if (limit > window_begin && limit > available)
	{
		if (threadIdx.x == 0u)
			LmKvReportRequiredAccessFailure(cache.pages, LM_KV_ACCESS_ERROR_PAGE_TABLE_OUT_OF_RANGE, LM_KV_ACCESS_READ, row, sequence,
				SparkKvShardLocalPosition(cache.shard, available), Geometry::PageOf(SparkKvShardLocalPosition(cache.shard, available)));
		return;
	}
	key_origin = (uint64_t)cache.key_offset[sequence] - window_begin;
	tile_first = window_begin / TILE;
	tile_end = limit > window_begin ? (limit + TILE - 1u) / TILE : tile_first;

	auto copy = [&](uint32_t which, uint32_t item)
	{
		const uint32_t stage = base + ((which - tile_first) % STAGES) * STAGE_BYTES;
		uint32_t key, piece, local, which_head;
		if (item >= STAGE_ITEMS)
			return;
		if (item < KEY_ITEMS)
		{
			key = item / CHUNKS;
			piece = item % CHUNKS;
			local = which * TILE + key;
			const bool valid = local >= window_begin && local < limit;
			LmPrefillCopy(stage + key * KEY_PITCH + piece * 16u,
				valid ? (const void *)(cache.keys + (key_origin + local) * Geometry::kSlotBytes + piece * 16u) : (const void *)cache.keys, valid ? 16u : 0u);
			return;
		}
		item -= KEY_ITEMS;
		which_head = item / (TILE * VALUE_CHUNKS);
		key = item / VALUE_CHUNKS % TILE;
		piece = item % VALUE_CHUNKS;
		local = which * TILE + key;
		const bool valid = local >= window_begin && local < limit;
		LmPrefillCopy(stage + KEY_BYTES + (which_head * TILE + key) * VALUE_PITCH + piece * 16u,
			valid ? (const void *)(values + (key_origin + local) * value_row_bytes + (uint64_t)(head + which_head) * VALUE_BYTES + piece * 16u) : (const void *)values, valid ? 16u : 0u);
	};

	for (index = threadIdx.x; index < QUERIES * CHUNKS; index += LM_EXPANDED_PREFILL_THREADS)
	{
		q = index / CHUNKS;
		chunk = index % CHUNKS;
		query_row = row + (q >> 4u & 1u) * 16u + (q & 15u);
		query_row = query_row <= last_row ? query_row : last_row;
		LmPrefillCopy(base + q * ROW_BYTES + LmPrefillSwizzle(q,chunk) * 16u,
			query_bf16 + ((uint64_t)query_row * heads_per_rank + head + (q >> 5u)) * DIM + chunk * 8u, 16u);
	}
	asm volatile("cp.async.commit_group;\n" ::: "memory");
	asm volatile("cp.async.wait_group 0;\n" ::: "memory");
	__syncthreads();
	{
		const uint32_t a_row = tile_id * 16u + (matrix & 1u) * 8u + within;
#pragma unroll
		for (step = 0u; step < HALF_STEPS; step++)
			LmPrefillLoadX4(base + a_row * ROW_BYTES + LmPrefillSwizzle(a_row,(half * HALF_STEPS + step) * 2u + (matrix >> 1u)) * 16u, query_frag[step]);
	}
#pragma unroll
	for (frag = 0u; frag < FRAGS; frag++)
		output[frag][0] = output[frag][1] = output[frag][2] = output[frag][3] = 0.0f;
	__syncthreads();
	for (index = 0u; index + 1u < STAGES; index++)
	{
		if (tile_first + index < tile_end)
			for (chunk = 0u; chunk < ROUNDS; chunk++)
				copy(tile_first + index, threadIdx.x + chunk * LM_EXPANDED_PREFILL_THREADS);
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	}
	for (tile = tile_first; tile < tile_end; tile++)
	{
		const bool refill = tile + STAGES - 1u < tile_end;
		asm volatile("cp.async.wait_group %0;\n" :: "n"(STAGES - 2u) : "memory");
		__syncthreads();
		const uint32_t stage = base + ((tile - tile_first) % STAGES) * STAGE_BYTES;
		float score[2][4] = { { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } };
		{
			uint32_t b[4], first[2], second[2];
			const uint32_t b_row = (matrix >> 1u) * 8u + within;
#pragma unroll
			for (step = 0u; step < HALF_STEPS; step++)
			{
				if (refill && (step * ROUNDS) % HALF_STEPS < ROUNDS)
					copy(tile + STAGES - 1u, threadIdx.x + (step * ROUNDS / HALF_STEPS) * LM_EXPANDED_PREFILL_THREADS);
				LmPrefillLoadX4(stage + b_row * KEY_PITCH + ((half * HALF_STEPS + step) * 2u + (matrix & 1u)) * 16u, b);
				first[0] = b[0]; first[1] = b[1];
				second[0] = b[2]; second[1] = b[3];
				LmMmaBf16(score[0], query_frag[step], first);
				LmMmaBf16(score[1], query_frag[step], second);
			}
			asm volatile("cp.async.commit_group;\n" ::: "memory");
		}
		{
			float *mine = exchange + (((tile & 1u) * LM_EXPANDED_PREFILL_WARPS + warp) * 32u + lane) * LM_EXPANDED_PREFILL_EXCHANGE_FLOATS;
			const float *theirs = exchange + (((tile & 1u) * LM_EXPANDED_PREFILL_WARPS + (warp ^ 1u)) * 32u + lane) * LM_EXPANDED_PREFILL_EXCHANGE_FLOATS;
			*(float4 *)mine = make_float4(score[0][0], score[0][1], score[0][2], score[0][3]);
			*(float4 *)(mine + 4u) = make_float4(score[1][0], score[1][1], score[1][2], score[1][3]);
			LmExpandedNamedBarrier(1u + tile_id, 64u);
			const float4 other_low = *(const float4 *)theirs, other_high = *(const float4 *)(theirs + 4u);
			score[0][0] += other_low.x; score[0][1] += other_low.y; score[0][2] += other_low.z; score[0][3] += other_low.w;
			score[1][0] += other_high.x; score[1][1] += other_high.y; score[1][2] += other_high.z; score[1][3] += other_high.w;
		}
		uint32_t weight_high[4], weight_low[4];
		{
			float peak_low = -INFINITY, peak_high = -INFINITY, next_low, next_high, scale_low, scale_high, total_low = 0.0f, total_high = 0.0f, value;
			uint16_t high[2][4], low[2][4];
			uint32_t local;
#pragma unroll
			for (frag = 0u; frag < 2u; frag++)
#pragma unroll
				for (index = 0u; index < 4u; index++)
				{
					local = tile * TILE + frag * 8u + (lane & 3u) * 2u + (index & 1u);
					value = score[frag][index] * qk_scale;
					value = local < (index < 2u ? count_low : count_high) && local < limit && local >= window_begin ? value : -INFINITY;
					score[frag][index] = value;
					if (index < 2u)
						peak_low = fmaxf(peak_low, value);
					else
						peak_high = fmaxf(peak_high, value);
				}
			peak_low = fmaxf(peak_low, __shfl_xor_sync(0xffffffffu, peak_low, 1u));
			peak_high = fmaxf(peak_high, __shfl_xor_sync(0xffffffffu, peak_high, 1u));
			peak_low = fmaxf(peak_low, __shfl_xor_sync(0xffffffffu, peak_low, 2u));
			peak_high = fmaxf(peak_high, __shfl_xor_sync(0xffffffffu, peak_high, 2u));
			next_low = fmaxf(max_low, peak_low);
			next_high = fmaxf(max_high, peak_high);
			scale_low = max_low == -INFINITY ? 0.0f : __expf(max_low - next_low);
			scale_high = max_high == -INFINITY ? 0.0f : __expf(max_high - next_high);
#pragma unroll
			for (frag = 0u; frag < 2u; frag++)
#pragma unroll
				for (index = 0u; index < 4u; index++)
				{
					const float next = index < 2u ? next_low : next_high;
					value = next == -INFINITY || score[frag][index] == -INFINITY ? 0.0f : __expf(score[frag][index] - next);
					high[frag][index] = LmFloatToBf16(value);
					low[frag][index] = LmFloatToBf16(value - LmBf16ToFloat(high[frag][index]));
					if (index < 2u)
						total_low += value;
					else
						total_high += value;
				}
			total_low += __shfl_xor_sync(0xffffffffu, total_low, 1u);
			total_high += __shfl_xor_sync(0xffffffffu, total_high, 1u);
			total_low += __shfl_xor_sync(0xffffffffu, total_low, 2u);
			total_high += __shfl_xor_sync(0xffffffffu, total_high, 2u);
			sum_low = sum_low * scale_low + total_low;
			sum_high = sum_high * scale_high + total_high;
			max_low = next_low;
			max_high = next_high;
#pragma unroll
			for (frag = 0u; frag < FRAGS; frag++)
			{
				output[frag][0] *= scale_low;
				output[frag][1] *= scale_low;
				output[frag][2] *= scale_high;
				output[frag][3] *= scale_high;
			}
			weight_high[0] = LmExpandedPackBf16(high[0][0], high[0][1]);
			weight_high[1] = LmExpandedPackBf16(high[0][2], high[0][3]);
			weight_high[2] = LmExpandedPackBf16(high[1][0], high[1][1]);
			weight_high[3] = LmExpandedPackBf16(high[1][2], high[1][3]);
			weight_low[0] = LmExpandedPackBf16(low[0][0], low[0][1]);
			weight_low[1] = LmExpandedPackBf16(low[0][2], low[0][3]);
			weight_low[2] = LmExpandedPackBf16(low[1][0], low[1][1]);
			weight_low[3] = LmExpandedPackBf16(low[1][2], low[1][3]);
		}
		{
			const uint32_t value_base = stage + KEY_BYTES + local_head * TILE * VALUE_PITCH;
			const uint32_t b_row = (matrix & 1u) * 8u + within;
			uint32_t b[4], first[2], second[2];
#pragma unroll
			for (frag = 0u; frag < FRAGS; frag += 2u)
			{
				LmPrefillLoadX4Trans(value_base + b_row * VALUE_PITCH + ((half * HALF_COLUMNS + frag * 8u) / 8u + (matrix >> 1u)) * 16u, b);
				first[0] = b[0]; first[1] = b[1];
				second[0] = b[2]; second[1] = b[3];
				LmMmaBf16(output[frag], weight_high, first);
				LmMmaBf16(output[frag], weight_low, first);
				LmMmaBf16(output[frag + 1u], weight_high, second);
				LmMmaBf16(output[frag + 1u], weight_low, second);
			}
		}
	}
	{
		const uint32_t column = half * HALF_COLUMNS + (lane & 3u) * 2u;
		float *record_low = partials + ((uint64_t)row_low * heads_per_rank + head + local_head) * RECORD;
		float *record_high = partials + ((uint64_t)row_high * heads_per_rank + head + local_head) * RECORD;
#pragma unroll
		for (frag = 0u; frag < FRAGS; frag++)
		{
			if (row_low <= last_row)
				*(float2 *)(record_low + 2u + column + frag * 8u) = make_float2(output[frag][0], output[frag][1]);
			if (row_high <= last_row)
				*(float2 *)(record_high + 2u + column + frag * 8u) = make_float2(output[frag][2], output[frag][3]);
		}
		if (half == 0u && (lane & 3u) == 0u)
		{
			if (row_low <= last_row)
			{
				record_low[0] = max_low;
				record_low[1] = sum_low;
			}
			if (row_high <= last_row)
			{
				record_high[0] = max_high;
				record_high[1] = sum_high;
			}
		}
	}
}

template<class Geometry, uint32_t LATENT, uint32_t ROPE, uint32_t VALUE>
static inline cudaError_t LmExpandedGatherPrefillLaunch(
	LmKvShardGatherView cache,
	const uint8_t *values,
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
	constexpr uint32_t shared = LmExpandedPrefillSharedBytes<LATENT + ROPE, VALUE>();
	static bool granted = false;
	if (rows == 0u || heads_per_rank == 0u || (heads_per_rank % LM_EXPANDED_PREFILL_HEADS) != 0u || values == 0 || query_bf16 == 0 ||
		partials == 0 || row_position == 0 || cache.keys == 0 || cache.key_offset == 0 || cache.key_context == 0 ||
		SparkKvShardValid(cache.shard, Geometry::kPageSlots) == 0u || (cache.key_begin % LM_EXPANDED_PREFILL_TILE) != 0u)
		return cudaErrorInvalidValue;
	if (!granted)
	{
		cudaError_t error = cudaFuncSetAttribute((const void *)LmExpandedGatherPrefillKernel<Geometry, LATENT, ROPE, VALUE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared);
		if (error != cudaSuccess)
			return error;
		granted = true;
	}
	LmExpandedGatherPrefillKernel<Geometry, LATENT, ROPE, VALUE><<<dim3((rows + LM_EXPANDED_PREFILL_ROWS - 1u) / LM_EXPANDED_PREFILL_ROWS, heads_per_rank / LM_EXPANDED_PREFILL_HEADS, 1u), LM_EXPANDED_PREFILL_THREADS, shared, stream>>>(
		cache, values, query_bf16, heads_per_rank, sequence_of_row, context_length, row_position, qk_scale, partials, rows);
	return cudaPeekAtLastError();
}
