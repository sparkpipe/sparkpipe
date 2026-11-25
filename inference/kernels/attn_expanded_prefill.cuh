#pragma once

#include "inference/kernels/attn_shard_prefill.cuh"
#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/tma.cuh"
#include "runtime/launch.h"
#include "sparkpipe/spark_gpu.h"

#define LM_EXPANDED_PREFILL_COMPUTE_WARPS 8u
#define LM_EXPANDED_PREFILL_THREADS (LM_EXPANDED_PREFILL_COMPUTE_WARPS * 32u)
#define LM_EXPANDED_PREFILL_ROWS 64u
#define LM_EXPANDED_PREFILL_HEADS 2u
#define LM_EXPANDED_PREFILL_TILE 16u
#define LM_EXPANDED_PREFILL_RING 2u
#define LM_EXPANDED_PREFILL_EXCHANGE_FLOATS 8u
#define LM_EXPANDED_PREFILL_REG_STEPS 14u

#define LM_LATENT_EXPAND_THREADS 256u
#define LM_LATENT_EXPAND_KEYS 128u
#define LM_LATENT_EXPAND_CHUNK 32u
#define LM_LATENT_EXPAND_STAGES 3u

#define LM_EXPANDED_PREFILL_PAD 16u

template<uint32_t DIM, uint32_t VALUE>
static __host__ __device__ constexpr uint32_t LmExpandedPrefillSharedBytes(void)
{
	constexpr uint32_t HALF_STEPS = (DIM / 16u) / 2u;
	constexpr uint32_t REG_STEPS = HALF_STEPS < LM_EXPANDED_PREFILL_REG_STEPS ? HALF_STEPS : LM_EXPANDED_PREFILL_REG_STEPS;
	constexpr uint32_t TAIL_STEPS = HALF_STEPS - REG_STEPS;
	constexpr uint32_t RING_AND_EXCHANGE =
		LM_EXPANDED_PREFILL_RING * LM_EXPANDED_PREFILL_TILE * (DIM * 2u + LM_EXPANDED_PREFILL_PAD)
		+ LM_EXPANDED_PREFILL_RING * LM_EXPANDED_PREFILL_HEADS * LM_EXPANDED_PREFILL_TILE * (VALUE * 2u + LM_EXPANDED_PREFILL_PAD)
		+ LM_EXPANDED_PREFILL_COMPUTE_WARPS * 32u * LM_EXPANDED_PREFILL_EXCHANGE_FLOATS * 4u;
	constexpr uint32_t STAGE_BYTES = (LM_EXPANDED_PREFILL_ROWS / 2u) * LM_EXPANDED_PREFILL_HEADS * (2u * REG_STEPS * 16u * 2u);
	constexpr uint32_t TAIL_BYTES = LM_EXPANDED_PREFILL_ROWS * LM_EXPANDED_PREFILL_HEADS * (2u * TAIL_STEPS * 16u * 2u);
	return((RING_AND_EXCHANGE > STAGE_BYTES ? RING_AND_EXCHANGE : STAGE_BYTES) + TAIL_BYTES);
}

template<uint32_t VALUE>
static __host__ __device__ constexpr uint32_t LmLatentExpandSharedBytes(void)
{
	return(LM_LATENT_EXPAND_STAGES * (LM_LATENT_EXPAND_KEYS + VALUE) * LM_LATENT_EXPAND_CHUNK * 2u);
}

static __device__ __forceinline__ uint32_t LmLatentExpandSwizzle(uint32_t row, uint32_t chunk)
{
	return(chunk ^ ((row >> 1u) & 3u));
}

static __device__ __forceinline__ void LmExpandedNamedBarrier(uint32_t name, uint32_t threads)
{
	asm volatile("bar.sync %0, %1;\n" :: "r"(name), "r"(threads) : "memory");
}

static __device__ __forceinline__ uint32_t LmExpandedSource(void)
{
	uint32_t source;
	asm volatile("mov.u32 %0, %%ctaid.z;\n" : "=r"(source));
	return(source);
}

static __device__ __forceinline__ void LmExpandedCopy16(uint32_t destination, const void *source)
{
	asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(destination), "l"(source) : "memory");
}

template<uint32_t LATENT, uint32_t ROW_DIM, uint32_t VALUE>
__global__ __launch_bounds__(LM_LATENT_EXPAND_THREADS, 2)
void LmLatentValueExpandKernel(
	const uint8_t *__restrict__ keys,
	uint64_t key_source_stride,
	uint32_t keys_per_source,
	const uint16_t *__restrict__ weight_bf16,
	uint32_t heads,
	uint8_t *__restrict__ values,
	uint64_t value_source_stride,
	uint64_t key_pitch,
	uint64_t head_pitch)
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
	static_assert(VALUE % 16u == 0u && CHUNKS == 4u, "whole value fragment pairs and four swizzled chunks per row");
	static_assert(LATENT <= ROW_DIM, "the latent sits at the front of each key row");
	static_assert(LmLatentExpandSharedBytes<VALUE>() <= LM_PREFILL_ATTN_SHARED_LIMIT, "the expansion tiles exceed the shared memory an SM has");
	extern __shared__ __align__(128) uint8_t lm_expand_shared[];
	LmDependentWait();
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t matrix = lane >> 3u, within = lane & 7u;
	const uint32_t first_key = blockIdx.y * LM_LATENT_EXPAND_KEYS, head = blockIdx.x, source = blockIdx.z;
	const uint32_t base = (uint32_t)__cvta_generic_to_shared(lm_expand_shared);
	const uint8_t *source_keys = keys + (uint64_t)source * key_source_stride;
	const uint16_t *head_weight = weight_bf16 + (uint64_t)head * VALUE * LATENT;
	float output[FRAGS][4];
	uint32_t round, step, frag, item, row, chunk, a[4], b[4];

	if (first_key >= keys_per_source || head >= heads)
	{
		LmDependentRelease();
		return;
	}
	auto load = [&](uint32_t which)
	{
		const uint32_t stage = base + (which % LM_LATENT_EXPAND_STAGES) * STAGE_BYTES;
		const uint32_t k_off = which * LM_LATENT_EXPAND_CHUNK;
		#pragma unroll
		for (item = threadIdx.x; item < (LM_LATENT_EXPAND_KEYS + VALUE) * CHUNKS; item += LM_LATENT_EXPAND_THREADS)
		{
			row = item >> 2u;
			chunk = item & 3u;
			if (row < LM_LATENT_EXPAND_KEYS)
			{
				const uint32_t key = first_key + row;
				LmPrefillCopy(stage + row * (LM_LATENT_EXPAND_CHUNK * 2u) + LmLatentExpandSwizzle(row, chunk) * 16u,
					key < keys_per_source ? (const void *)(source_keys + (uint64_t)key * ROW_BYTES + (k_off + chunk * 8u) * 2u) : (const void *)source_keys,
					key < keys_per_source ? 16u : 0u);
			}
			else
			{
				const uint32_t wrow = row - LM_LATENT_EXPAND_KEYS;
				LmPrefillCopy(stage + A_BYTES + wrow * (LM_LATENT_EXPAND_CHUNK * 2u) + LmLatentExpandSwizzle(wrow, chunk) * 16u,
					head_weight + (uint64_t)wrow * LATENT + k_off + chunk * 8u, 16u);
			}
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
			LmPrefillLoadX4(stage + a_row * (LM_LATENT_EXPAND_CHUNK * 2u) + LmLatentExpandSwizzle(a_row, step * 2u + (matrix >> 1u)) * 16u, a);
#pragma unroll
			for (frag = 0u; frag < FRAGS; frag += 2u)
			{
				const uint32_t b_row = frag * 8u + (matrix >> 1u) * 8u + within;
				uint32_t first[2], second[2];
				LmPrefillLoadX4(stage + A_BYTES + b_row * (LM_LATENT_EXPAND_CHUNK * 2u) + LmLatentExpandSwizzle(b_row, step * 2u + (matrix & 1u)) * 16u, b);
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
		const uint64_t low_base = ((uint64_t)key_low * key_pitch + (uint64_t)head * head_pitch + (lane & 3u) * 2u) * 2u;
		const uint64_t high_base = ((uint64_t)key_high * key_pitch + (uint64_t)head * head_pitch + (lane & 3u) * 2u) * 2u;
		if (key_low < keys_per_source)
		{
#pragma unroll
			for (frag = 0u; frag < FRAGS; frag++)
				*(uint32_t *)(target + low_base + frag * 16u) = LmPackBf16Pair(output[frag][0], output[frag][1]);
		}
		if (key_high < keys_per_source)
		{
#pragma unroll
			for (frag = 0u; frag < FRAGS; frag++)
				*(uint32_t *)(target + high_base + frag * 16u) = LmPackBf16Pair(output[frag][2], output[frag][3]);
		}
	}
	LmDependentRelease();
}

template<uint32_t LATENT, uint32_t ROW_DIM, uint32_t VALUE>
static inline SparkGpuError LmLatentValueExpandLaunch(
	const uint8_t *keys,
	uint64_t key_source_stride,
	uint32_t sources,
	uint32_t keys_per_source,
	const uint16_t *weight_bf16,
	uint32_t heads,
	uint8_t *values,
	uint64_t value_source_stride,
	SparkGpuStream stream)
{
	constexpr uint32_t shared = LmLatentExpandSharedBytes<VALUE>();
	static bool granted = false;
	if (keys == 0 || weight_bf16 == 0 || values == 0 || sources == 0u || heads == 0u ||
		(key_source_stride % 16u) != 0u || (value_source_stride % 16u) != 0u ||
		(sources > 1u && (key_source_stride < (uint64_t)keys_per_source * ROW_DIM * 2u ||
			value_source_stride < (uint64_t)keys_per_source * heads * VALUE * 2u)))
		SPARK_FAIL(SPARK_GPU_ERROR_INVALID_VALUE);
	if (keys_per_source == 0u)
		return SPARK_GPU_SUCCESS;
	if (!granted)
	{
		SparkGpuError error = SparkGpuFuncSetAttribute((const void *)LmLatentValueExpandKernel<LATENT, ROW_DIM, VALUE>, SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)shared);
		if (error != SPARK_GPU_SUCCESS)
			return error;
		granted = true;
	}
	if ((keys_per_source + LM_LATENT_EXPAND_KEYS - 1u) / LM_LATENT_EXPAND_KEYS > 65535u)
		SPARK_FAIL(SPARK_GPU_ERROR_INVALID_VALUE);
	LM_LAUNCH_DEPENDENT((LmLatentValueExpandKernel<LATENT, ROW_DIM, VALUE>), dim3(heads, (keys_per_source + LM_LATENT_EXPAND_KEYS - 1u) / LM_LATENT_EXPAND_KEYS, sources), LM_LATENT_EXPAND_THREADS, shared, stream,
		keys, key_source_stride, keys_per_source, weight_bf16, heads, values, value_source_stride, (uint64_t)heads * VALUE, VALUE);
	return SparkGpuPeekAtLastError();
}

template<uint32_t LATENT, uint32_t ROW_DIM>
__global__ __launch_bounds__(256, 2)
void LmLatentRopeBroadcastKernel(
	const uint8_t *__restrict__ keys,
	uint32_t key_count,
	uint32_t heads,
	uint8_t *__restrict__ output,
	uint64_t key_pitch,
	uint64_t head_pitch,
	uint64_t rope_offset)
{
	constexpr uint32_t ROPE_DIM = ROW_DIM - LATENT;
	constexpr uint32_t ROPE_WORDS = ROPE_DIM / 2u;
	__shared__ __align__(16) uint32_t rope_shared[ROPE_WORDS];
	LmDependentWait();
	const uint32_t key = blockIdx.x;
	if (key >= key_count)
	{
		LmDependentRelease();
		return;
	}
	const uint32_t *rope = (const uint32_t *)(keys + ((uint64_t)key * ROW_DIM + LATENT) * 2u);
	for (uint32_t w = threadIdx.x; w < ROPE_WORDS; w += blockDim.x)
		rope_shared[w] = __ldg(rope + w);
	__syncthreads();
	uint8_t *dst_key = output + ((uint64_t)key * key_pitch + rope_offset) * 2u;
	if constexpr ((ROPE_DIM % 8u) == 0u)
	{
		if ((((uintptr_t)dst_key | (head_pitch * 2u)) & 15u) == 0u)
		{
			constexpr uint32_t ROPE_VECS = ROPE_DIM / 8u;
			const uint4 *src4 = (const uint4 *)rope_shared;
			for (uint32_t item = threadIdx.x; item < heads * ROPE_VECS; item += blockDim.x)
			{
				const uint32_t h = item / ROPE_VECS;
				const uint32_t v = item - h * ROPE_VECS;
				((uint4 *)(dst_key + (uint64_t)h * head_pitch * 2u))[v] = src4[v];
			}
			LmDependentRelease();
			return;
		}
	}
	for (uint32_t item = threadIdx.x; item < heads * ROPE_WORDS; item += blockDim.x)
	{
		const uint32_t h = item / ROPE_WORDS;
		const uint32_t w = item - h * ROPE_WORDS;
		*(uint32_t *)(dst_key + (uint64_t)h * head_pitch * 2u + w * 4u) = rope_shared[w];
	}
	LmDependentRelease();
}

template<uint32_t LATENT, uint32_t ROW_DIM>
static inline SparkGpuError LmLatentRopeBroadcastLaunch(
	const uint8_t *keys,
	uint32_t key_count,
	uint32_t heads,
	uint8_t *output,
	uint64_t key_pitch,
	uint64_t head_pitch,
	uint64_t rope_offset,
	SparkGpuStream stream)
{
	static_assert(ROW_DIM > LATENT && (ROW_DIM - LATENT) % 2u == 0u, "the rope part follows the latent and is whole bf16 pairs");
	if (keys == 0 || output == 0 || heads == 0u || (uintptr_t)output % 4u != 0u || rope_offset % 2u != 0u || head_pitch % 2u != 0u || key_pitch % 2u != 0u ||
		head_pitch < rope_offset + (ROW_DIM - LATENT) || key_pitch < (uint64_t)(heads - 1u) * head_pitch + rope_offset + (ROW_DIM - LATENT))
		SPARK_FAIL(SPARK_GPU_ERROR_INVALID_VALUE);
	if (key_count == 0u)
		return SPARK_GPU_SUCCESS;
	LM_LAUNCH_DEPENDENT((LmLatentRopeBroadcastKernel<LATENT, ROW_DIM>), key_count, 256, 0, stream, keys, key_count, heads, output, key_pitch, head_pitch, rope_offset);
	return SparkGpuPeekAtLastError();
}

template<uint32_t LATENT, uint32_t ROW_DIM, uint32_t VALUE>
static inline SparkGpuError LmLatentExpandPitchedLaunch(
	const uint8_t *keys,
	uint32_t key_count,
	const uint16_t *weight_bf16,
	uint32_t heads,
	uint8_t *output,
	uint64_t key_pitch,
	uint64_t head_pitch,
	SparkGpuStream stream)
{
	constexpr uint32_t shared = LmLatentExpandSharedBytes<VALUE>();
	static bool granted = false;
	if (keys == 0 || weight_bf16 == 0 || output == 0 || heads == 0u || (uintptr_t)output % 4u != 0u || head_pitch < VALUE ||
		key_pitch < (uint64_t)(heads - 1u) * head_pitch + VALUE || key_pitch % 2u != 0u || head_pitch % 2u != 0u)
		SPARK_FAIL(SPARK_GPU_ERROR_INVALID_VALUE);
	if (key_count == 0u)
		return SPARK_GPU_SUCCESS;
	if (!granted)
	{
		SparkGpuError error = SparkGpuFuncSetAttribute((const void *)LmLatentValueExpandKernel<LATENT, ROW_DIM, VALUE>, SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)shared);
		if (error != SPARK_GPU_SUCCESS)
			return error;
		granted = true;
	}
	if ((key_count + LM_LATENT_EXPAND_KEYS - 1u) / LM_LATENT_EXPAND_KEYS > 65535u)
		SPARK_FAIL(SPARK_GPU_ERROR_INVALID_VALUE);
	LM_LAUNCH_DEPENDENT((LmLatentValueExpandKernel<LATENT, ROW_DIM, VALUE>), dim3(heads, (key_count + LM_LATENT_EXPAND_KEYS - 1u) / LM_LATENT_EXPAND_KEYS, 1u), LM_LATENT_EXPAND_THREADS, shared, stream,
		keys, 0u, key_count, weight_bf16, heads, output, 0u, key_pitch, head_pitch);
	return SparkGpuPeekAtLastError();
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
	uint32_t rows,
	uint64_t key_source_stride,
	uint64_t value_source_stride,
	uint64_t partial_source_stride)
{
	constexpr uint32_t DIM = LATENT + ROPE;
	constexpr uint32_t ROW_BYTES = DIM * 2u;
	constexpr uint32_t VALUE_BYTES = VALUE * 2u;
	constexpr uint32_t CHUNKS = DIM / 8u;
	constexpr uint32_t VALUE_CHUNKS = VALUE / 8u;
	constexpr uint32_t TILE = LM_EXPANDED_PREFILL_TILE;
	constexpr uint32_t ROW_TILES = 2u;
	constexpr uint32_t HALF_QUERIES = (LM_EXPANDED_PREFILL_ROWS / ROW_TILES) * LM_EXPANDED_PREFILL_HEADS;
	constexpr uint32_t ALL_QUERIES = LM_EXPANDED_PREFILL_ROWS * LM_EXPANDED_PREFILL_HEADS;
	constexpr uint32_t K_STEPS = DIM / 16u;
	constexpr uint32_t HALF_STEPS = K_STEPS / 2u;
	constexpr uint32_t REG_STEPS = HALF_STEPS < LM_EXPANDED_PREFILL_REG_STEPS ? HALF_STEPS : LM_EXPANDED_PREFILL_REG_STEPS;
	constexpr uint32_t TAIL_STEPS = HALF_STEPS - REG_STEPS;
	constexpr uint32_t REG_HALF_CHUNKS = REG_STEPS * 2u;
	constexpr uint32_t TAIL_HALF_CHUNKS = TAIL_STEPS * 2u;
	constexpr uint32_t REG_CHUNKS = 2u * REG_HALF_CHUNKS;
	constexpr uint32_t TAIL_CHUNKS = 2u * TAIL_HALF_CHUNKS;
	constexpr uint32_t REG_ROW_BYTES = REG_CHUNKS * 16u;
	constexpr uint32_t TAIL_ROW_BYTES = TAIL_CHUNKS * 16u;
	constexpr uint32_t HALF_COLUMNS = VALUE / 2u;
	constexpr uint32_t FRAGS = HALF_COLUMNS / 8u;
	constexpr uint32_t KEY_PITCH = ROW_BYTES + LM_EXPANDED_PREFILL_PAD;
	constexpr uint32_t VALUE_PITCH = VALUE_BYTES + LM_EXPANDED_PREFILL_PAD;
	constexpr uint32_t KEY_SLOT = TILE * KEY_PITCH;
	constexpr uint32_t VALUE_SLOT = LM_EXPANDED_PREFILL_HEADS * TILE * VALUE_PITCH;
	constexpr uint32_t RING_AND_EXCHANGE =
		LM_EXPANDED_PREFILL_RING * KEY_SLOT
		+ LM_EXPANDED_PREFILL_RING * VALUE_SLOT
		+ LM_EXPANDED_PREFILL_COMPUTE_WARPS * 32u * LM_EXPANDED_PREFILL_EXCHANGE_FLOATS * 4u;
	constexpr uint32_t STAGE_BYTES = HALF_QUERIES * REG_ROW_BYTES;
	constexpr uint32_t TAIL_OFFSET = RING_AND_EXCHANGE > STAGE_BYTES ? RING_AND_EXCHANGE : STAGE_BYTES;
	constexpr uint32_t RECORD = LM_LATENT_SHARD_RECORD_FLOATS(VALUE);
	static_assert(TILE == 16u && LM_EXPANDED_PREFILL_COMPUTE_WARPS == 8u && VALUE_CHUNKS == 16u, "two keys per warp and sixteen value chunks per head");
	static_assert(REG_CHUNKS % 8u == 0u && TAIL_CHUNKS % 8u == 0u, "swizzled query staging uses multiples of eight chunks");
	static_assert(LmExpandedPrefillSharedBytes<DIM, VALUE>() <= LM_PREFILL_ATTN_SHARED_LIMIT, "the rings exceed the shared memory an SM has");
	extern __shared__ __align__(128) uint8_t lm_expanded_shared[];
	uint8_t *value_ring = lm_expanded_shared + LM_EXPANDED_PREFILL_RING * KEY_SLOT;
	float *exchange = (float *)(value_ring + LM_EXPANDED_PREFILL_RING * VALUE_SLOT);
	LmDependentWait();
	const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
	const uint32_t matrix = lane >> 3u, within = lane & 7u;
	const uint32_t tile_id = warp >> 1u, half = warp & 1u, local_head = tile_id >> 1u;
	const uint32_t row = blockIdx.x * LM_EXPANDED_PREFILL_ROWS, head = blockIdx.y * LM_EXPANDED_PREFILL_HEADS;
	const uint32_t last_row = (row + LM_EXPANDED_PREFILL_ROWS < rows ? row + LM_EXPANDED_PREFILL_ROWS : rows) - 1u;
	const uint32_t base = (uint32_t)__cvta_generic_to_shared(lm_expanded_shared);
	const uint32_t key_base = base, value_base = base + LM_EXPANDED_PREFILL_RING * KEY_SLOT;
	const uint32_t tail_base = base + TAIL_OFFSET;
	const uint64_t value_row_bytes = (uint64_t)heads_per_rank * VALUE_BYTES;
	const float scale_log2 = qk_scale * 1.4426950408889634f;
	const uint32_t source_id = LmExpandedSource();
	const uint8_t *source_keys = cache.keys + (uint64_t)source_id * key_source_stride;
	const uint8_t *source_values = values + (uint64_t)source_id * value_source_stride;
	uint32_t sequence, dense, count, window_begin, limit, available, tile_first, tile_end, tiles, index, step, frag, q, query_row;
	uint32_t count_low[ROW_TILES], count_high[ROW_TILES], safe_end[ROW_TILES];
	uint64_t key_origin;

	SparkKvShard shard = cache.shard;
	shard.rank += source_id;
	sequence = __ldg(sequence_of_row + row);
	for (query_row = row + 1u; query_row <= last_row; query_row++)
		if (__ldg(sequence_of_row + query_row) != sequence)
			sequence = cache.pages.sequence_count;
	if (!LmKvViewIsConfigured(cache.pages) || sequence >= cache.pages.sequence_count)
	{
		if (threadIdx.x == 0u)
			LmKvReportRequiredAccessFailure(cache.pages, !LmKvViewIsConfigured(cache.pages) ? LM_KV_ACCESS_ERROR_INVALID_VIEW : LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE, LM_KV_ACCESS_READ, row, sequence_of_row[row], 0xffffffffu, 0xffffffffu);
		LmDependentRelease();
		return;
	}
	dense = __ldg(context_length + sequence);
	count = 0u;
	for (query_row = row; query_row <= last_row; query_row++)
	{
		const uint32_t rp = __ldg(row_position + query_row);
		const uint32_t d = SparkKvShardLocalKeys(shard, rp < dense ? rp + 1u : dense);
		count = d > count ? d : count;
	}
	window_begin = cache.key_begin;
	limit = count < cache.key_end ? count : cache.key_end;
	available = SparkKvShardLocalKeys(shard, __ldg(cache.key_context + sequence));
	if (count > available)
	{
		if (threadIdx.x == 0u)
			LmKvReportRequiredAccessFailure(cache.pages, LM_KV_ACCESS_ERROR_PAGE_TABLE_OUT_OF_RANGE, LM_KV_ACCESS_READ, row, sequence,
				SparkKvShardLocalPosition(shard, available), Geometry::PageOf(SparkKvShardLocalPosition(shard, available)));
		LmDependentRelease();
		return;
	}
#pragma unroll
	for (uint32_t m = 0u; m < ROW_TILES; m++)
	{
		const uint32_t rl = row + m * 32u + (tile_id & 1u) * 16u + (lane >> 2u);
		const uint32_t rh = rl + 8u;
		count_low[m] = rl <= last_row ? SparkKvShardLocalKeys(shard, __ldg(row_position + rl) < dense ? __ldg(row_position + rl) + 1u : dense) : 0u;
		count_high[m] = rh <= last_row ? SparkKvShardLocalKeys(shard, __ldg(row_position + rh) < dense ? __ldg(row_position + rh) + 1u : dense) : 0u;
		const uint32_t cmin = count_low[m] < count_high[m] ? count_low[m] : count_high[m];
		safe_end[m] = cmin < limit ? cmin : limit;
	}
	key_origin = (uint64_t)__ldg(cache.key_offset + sequence) - window_begin;
	tile_first = window_begin / TILE;
	tile_end = limit > window_begin ? (limit + TILE - 1u) / TILE : tile_first;
	tiles = tile_end - tile_first;

	if constexpr (TAIL_STEPS > 0u)
	{
		for (index = threadIdx.x; index < ALL_QUERIES * TAIL_CHUNKS; index += LM_EXPANDED_PREFILL_THREADS)
		{
			q = index / TAIL_CHUNKS;
			const uint32_t tc = index - q * TAIL_CHUNKS;
			const uint32_t h_side = tc >= TAIL_HALF_CHUNKS ? 1u : 0u;
			const uint32_t src_chunk = h_side * (HALF_STEPS * 2u) + REG_HALF_CHUNKS + (tc - h_side * TAIL_HALF_CHUNKS);
			const uint32_t m_idx = q >> 6u;
			const uint32_t q_local = q & 63u;
			query_row = row + m_idx * 32u + ((q_local >> 4u) & 1u) * 16u + (q_local & 15u);
			query_row = query_row <= last_row ? query_row : last_row;
			LmExpandedCopy16(tail_base + q * TAIL_ROW_BYTES + LmPrefillSwizzle(q, tc) * 16u,
				query_bf16 + ((uint64_t)query_row * heads_per_rank + head + (q_local >> 5u)) * DIM + src_chunk * 8u);
		}
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	}
	uint32_t query_frag[ROW_TILES][REG_STEPS][4];
#pragma unroll
	for (uint32_t m = 0u; m < ROW_TILES; m++)
	{
		for (index = threadIdx.x; index < HALF_QUERIES * REG_CHUNKS; index += LM_EXPANDED_PREFILL_THREADS)
		{
			q = index / REG_CHUNKS;
			const uint32_t rc = index - q * REG_CHUNKS;
			const uint32_t h_side = rc >= REG_HALF_CHUNKS ? 1u : 0u;
			const uint32_t src_chunk = h_side * (HALF_STEPS * 2u) + (rc - h_side * REG_HALF_CHUNKS);
			query_row = row + m * 32u + ((q >> 4u) & 1u) * 16u + (q & 15u);
			query_row = query_row <= last_row ? query_row : last_row;
			LmExpandedCopy16(base + q * REG_ROW_BYTES + LmPrefillSwizzle(q, rc) * 16u,
				query_bf16 + ((uint64_t)query_row * heads_per_rank + head + (q >> 5u)) * DIM + src_chunk * 8u);
		}
		asm volatile("cp.async.commit_group;\n" ::: "memory");
		asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		__syncthreads();
		const uint32_t a_row = tile_id * 16u + (matrix & 1u) * 8u + within;
#pragma unroll
		for (step = 0u; step < REG_STEPS; step++)
			LmPrefillLoadX4(base + a_row * REG_ROW_BYTES + LmPrefillSwizzle(a_row, half * REG_HALF_CHUNKS + step * 2u + (matrix >> 1u)) * 16u, query_frag[m][step]);
		__syncthreads();
	}

	auto load_tile = [&](uint32_t used)
	{
		const uint32_t which = tile_first + used;
		const uint32_t slot = used & 1u;
		const uint32_t first_key = which * TILE;
		const uint32_t low = window_begin > first_key ? window_begin - first_key : 0u;
		const uint32_t high = limit < first_key + TILE ? limit - first_key : TILE;
		const bool full = low == 0u && high == TILE;
		const uint8_t *key_tile = source_keys + (key_origin + first_key) * Geometry::kSlotBytes;
		const uint8_t *value_tile = source_values + (key_origin + first_key) * value_row_bytes + (uint64_t)head * VALUE_BYTES;
		const uint32_t key_slot = key_base + slot * KEY_SLOT;
		const uint32_t value_slot = value_base + slot * VALUE_SLOT;
		const uint32_t k0 = warp * 2u;
		const uint32_t k1 = k0 + 1u;
		constexpr uint32_t WARP_KEY_ITEMS = 2u * CHUNKS;
		if (full)
		{
#pragma unroll
			for (uint32_t idx = lane; idx < WARP_KEY_ITEMS; idx += 32u)
			{
				const uint32_t is_k1 = idx >= CHUNKS ? 1u : 0u;
				const uint32_t k = k0 + is_k1;
				const uint32_t piece = idx - is_k1 * CHUNKS;
				LmExpandedCopy16(key_slot + k * KEY_PITCH + piece * 16u, key_tile + (uint64_t)k * Geometry::kSlotBytes + piece * 16u);
			}
			{
				const uint32_t h = lane >> 4u;
				const uint32_t piece = lane & 15u;
				LmExpandedCopy16(value_slot + (h * TILE + k0) * VALUE_PITCH + piece * 16u,
					value_tile + (uint64_t)k0 * value_row_bytes + h * VALUE_BYTES + piece * 16u);
				LmExpandedCopy16(value_slot + (h * TILE + k1) * VALUE_PITCH + piece * 16u,
					value_tile + (uint64_t)k1 * value_row_bytes + h * VALUE_BYTES + piece * 16u);
			}
		}
		else
		{
#pragma unroll
			for (uint32_t idx = lane; idx < WARP_KEY_ITEMS; idx += 32u)
			{
				const uint32_t is_k1 = idx >= CHUNKS ? 1u : 0u;
				const uint32_t k = k0 + is_k1;
				const uint32_t piece = idx - is_k1 * CHUNKS;
				const bool valid = k >= low && k < high;
				LmPrefillCopy(key_slot + k * KEY_PITCH + piece * 16u,
					valid ? (const void *)(key_tile + (uint64_t)k * Geometry::kSlotBytes + piece * 16u) : (const void *)source_keys,
					valid ? 16u : 0u);
			}
			{
				const uint32_t h = lane >> 4u;
				const uint32_t piece = lane & 15u;
				const bool v0 = k0 >= low && k0 < high;
				const bool v1 = k1 >= low && k1 < high;
				LmPrefillCopy(value_slot + (h * TILE + k0) * VALUE_PITCH + piece * 16u,
					v0 ? (const void *)(value_tile + (uint64_t)k0 * value_row_bytes + h * VALUE_BYTES + piece * 16u) : (const void *)source_values,
					v0 ? 16u : 0u);
				LmPrefillCopy(value_slot + (h * TILE + k1) * VALUE_PITCH + piece * 16u,
					v1 ? (const void *)(value_tile + (uint64_t)k1 * value_row_bytes + h * VALUE_BYTES + piece * 16u) : (const void *)source_values,
					v1 ? 16u : 0u);
			}
		}
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	};

	float output[ROW_TILES][FRAGS][4];
	float max_low[ROW_TILES], max_high[ROW_TILES], sum_low[ROW_TILES], sum_high[ROW_TILES];
#pragma unroll
	for (uint32_t m = 0u; m < ROW_TILES; m++)
	{
		max_low[m] = -INFINITY;
		max_high[m] = -INFINITY;
		sum_low[m] = 0.0f;
		sum_high[m] = 0.0f;
#pragma unroll
		for (frag = 0u; frag < FRAGS; frag++)
			output[m][frag][0] = output[m][frag][1] = output[m][frag][2] = output[m][frag][3] = 0.0f;
	}
	if (tiles > 0u)
		load_tile(0u);
	float *mine_ex = exchange + (warp * 32u + lane) * LM_EXPANDED_PREFILL_EXCHANGE_FLOATS;
	const float *theirs_ex = exchange + ((warp ^ 1u) * 32u + lane) * LM_EXPANDED_PREFILL_EXCHANGE_FLOATS;
	for (uint32_t used = 0u; used < tiles; used++)
	{
		const uint32_t slot = used & 1u;
		const uint32_t tile = tile_first + used;
		const uint32_t tile_start = tile * TILE;
		if (used + 1u < tiles)
		{
			load_tile(used + 1u);
			asm volatile("cp.async.wait_group 1;\n" ::: "memory");
		}
		else
			asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		__syncthreads();
		const uint32_t key_slot_base = key_base + slot * KEY_SLOT;
		const uint32_t val_slot_base = value_base + slot * VALUE_SLOT + local_head * TILE * VALUE_PITCH;
		const uint32_t b_key_row = (matrix >> 1u) * 8u + within;
		const uint32_t b_val_row = (matrix & 1u) * 8u + within;
		float part[ROW_TILES][2][4];
		uint32_t weight_high[ROW_TILES][4];
#pragma unroll
		for (uint32_t m = 0u; m < ROW_TILES; m++)
		{
			part[m][0][0] = part[m][0][1] = part[m][0][2] = part[m][0][3] = 0.0f;
			part[m][1][0] = part[m][1][1] = part[m][1][2] = part[m][1][3] = 0.0f;
		}
#pragma unroll
		for (uint32_t s = 0u; s < REG_STEPS; s++)
		{
			uint32_t b[4], first[2], second[2];
			LmPrefillLoadX4(key_slot_base + b_key_row * KEY_PITCH + ((half * HALF_STEPS + s) * 2u + (matrix & 1u)) * 16u, b);
			first[0] = b[0]; first[1] = b[1];
			second[0] = b[2]; second[1] = b[3];
#pragma unroll
			for (uint32_t m = 0u; m < ROW_TILES; m++)
			{
				LmMmaBf16(part[m][0], query_frag[m][s], first);
				LmMmaBf16(part[m][1], query_frag[m][s], second);
			}
		}
		if constexpr (TAIL_STEPS > 0u)
		{
			const uint32_t q_row0 = tile_id * 16u + (matrix & 1u) * 8u + within;
			const uint32_t q_row1 = HALF_QUERIES + q_row0;
#pragma unroll
			for (uint32_t s = 0u; s < TAIL_STEPS; s++)
			{
				uint32_t qa0[4], qa1[4], b[4], first[2], second[2];
				const uint32_t tchunk = half * TAIL_HALF_CHUNKS + s * 2u + (matrix >> 1u);
				LmPrefillLoadX4(key_slot_base + b_key_row * KEY_PITCH + ((half * HALF_STEPS + REG_STEPS + s) * 2u + (matrix & 1u)) * 16u, b);
				LmPrefillLoadX4(tail_base + q_row0 * TAIL_ROW_BYTES + LmPrefillSwizzle(q_row0, tchunk) * 16u, qa0);
				LmPrefillLoadX4(tail_base + q_row1 * TAIL_ROW_BYTES + LmPrefillSwizzle(q_row1, tchunk) * 16u, qa1);
				first[0] = b[0]; first[1] = b[1];
				second[0] = b[2]; second[1] = b[3];
				LmMmaBf16(part[0][0], qa0, first);
				LmMmaBf16(part[0][1], qa0, second);
				LmMmaBf16(part[1][0], qa1, first);
				LmMmaBf16(part[1][1], qa1, second);
			}
		}
#pragma unroll
		for (uint32_t m = 0u; m < ROW_TILES; m++)
		{
			float score[2][4];
			*(float4 *)mine_ex = make_float4(part[m][0][0], part[m][0][1], part[m][0][2], part[m][0][3]);
			*(float4 *)(mine_ex + 4u) = make_float4(part[m][1][0], part[m][1][1], part[m][1][2], part[m][1][3]);
			LmExpandedNamedBarrier(1u + tile_id, 64u);
			const float4 other_low = *(const float4 *)theirs_ex;
			const float4 other_high = *(const float4 *)(theirs_ex + 4u);
			LmExpandedNamedBarrier(1u + tile_id, 64u);
			score[0][0] = part[m][0][0] + other_low.x; score[0][1] = part[m][0][1] + other_low.y; score[0][2] = part[m][0][2] + other_low.z; score[0][3] = part[m][0][3] + other_low.w;
			score[1][0] = part[m][1][0] + other_high.x; score[1][1] = part[m][1][1] + other_high.y; score[1][2] = part[m][1][2] + other_high.z; score[1][3] = part[m][1][3] + other_high.w;
			const bool full_tile = tile_start >= window_begin && tile_start + TILE <= safe_end[m];
			float peak_low = -INFINITY, peak_high = -INFINITY, next_low, next_high, scale_low, scale_high, total_low, total_high;
			float w00, w01, w02, w03, w10, w11, w12, w13;
			if (full_tile)
			{
				peak_low = fmaxf(fmaxf(score[0][0], score[0][1]), fmaxf(score[1][0], score[1][1]));
				peak_high = fmaxf(fmaxf(score[0][2], score[0][3]), fmaxf(score[1][2], score[1][3]));
			}
			else
			{
#pragma unroll
				for (frag = 0u; frag < 2u; frag++)
#pragma unroll
					for (index = 0u; index < 4u; index++)
					{
						const uint32_t local = tile_start + frag * 8u + (lane & 3u) * 2u + (index & 1u);
						const float v = local < (index < 2u ? count_low[m] : count_high[m]) && local < limit && local >= window_begin ? score[frag][index] : -INFINITY;
						score[frag][index] = v;
						if (index < 2u)
							peak_low = fmaxf(peak_low, v);
						else
							peak_high = fmaxf(peak_high, v);
					}
			}
			peak_low = fmaxf(peak_low, __shfl_xor_sync(0xffffffffu, peak_low, 1u));
			peak_high = fmaxf(peak_high, __shfl_xor_sync(0xffffffffu, peak_high, 1u));
			peak_low = fmaxf(peak_low, __shfl_xor_sync(0xffffffffu, peak_low, 2u));
			peak_high = fmaxf(peak_high, __shfl_xor_sync(0xffffffffu, peak_high, 2u));
			next_low = fmaxf(max_low[m], peak_low);
			next_high = fmaxf(max_high[m], peak_high);
			scale_low = max_low[m] == -INFINITY ? 0.0f : SparkMathExp2Fast((max_low[m] - next_low) * scale_log2);
			scale_high = max_high[m] == -INFINITY ? 0.0f : SparkMathExp2Fast((max_high[m] - next_high) * scale_log2);
			const float shift_low = next_low == -INFINITY ? 0.0f : next_low * scale_log2;
			const float shift_high = next_high == -INFINITY ? 0.0f : next_high * scale_log2;
			if (full_tile)
			{
				w00 = SparkMathExp2Fast(fmaf(score[0][0], scale_log2, -shift_low));
				w01 = SparkMathExp2Fast(fmaf(score[0][1], scale_log2, -shift_low));
				w02 = SparkMathExp2Fast(fmaf(score[0][2], scale_log2, -shift_high));
				w03 = SparkMathExp2Fast(fmaf(score[0][3], scale_log2, -shift_high));
				w10 = SparkMathExp2Fast(fmaf(score[1][0], scale_log2, -shift_low));
				w11 = SparkMathExp2Fast(fmaf(score[1][1], scale_log2, -shift_low));
				w12 = SparkMathExp2Fast(fmaf(score[1][2], scale_log2, -shift_high));
				w13 = SparkMathExp2Fast(fmaf(score[1][3], scale_log2, -shift_high));
			}
			else
			{
				w00 = next_low == -INFINITY || score[0][0] == -INFINITY ? 0.0f : SparkMathExp2Fast(fmaf(score[0][0], scale_log2, -shift_low));
				w01 = next_low == -INFINITY || score[0][1] == -INFINITY ? 0.0f : SparkMathExp2Fast(fmaf(score[0][1], scale_log2, -shift_low));
				w02 = next_high == -INFINITY || score[0][2] == -INFINITY ? 0.0f : SparkMathExp2Fast(fmaf(score[0][2], scale_log2, -shift_high));
				w03 = next_high == -INFINITY || score[0][3] == -INFINITY ? 0.0f : SparkMathExp2Fast(fmaf(score[0][3], scale_log2, -shift_high));
				w10 = next_low == -INFINITY || score[1][0] == -INFINITY ? 0.0f : SparkMathExp2Fast(fmaf(score[1][0], scale_log2, -shift_low));
				w11 = next_low == -INFINITY || score[1][1] == -INFINITY ? 0.0f : SparkMathExp2Fast(fmaf(score[1][1], scale_log2, -shift_low));
				w12 = next_high == -INFINITY || score[1][2] == -INFINITY ? 0.0f : SparkMathExp2Fast(fmaf(score[1][2], scale_log2, -shift_high));
				w13 = next_high == -INFINITY || score[1][3] == -INFINITY ? 0.0f : SparkMathExp2Fast(fmaf(score[1][3], scale_log2, -shift_high));
			}
			total_low = (w00 + w01) + (w10 + w11);
			total_high = (w02 + w03) + (w12 + w13);
			total_low += __shfl_xor_sync(0xffffffffu, total_low, 1u);
			total_high += __shfl_xor_sync(0xffffffffu, total_high, 1u);
			total_low += __shfl_xor_sync(0xffffffffu, total_low, 2u);
			total_high += __shfl_xor_sync(0xffffffffu, total_high, 2u);
			sum_low[m] = fmaf(sum_low[m], scale_low, total_low);
			sum_high[m] = fmaf(sum_high[m], scale_high, total_high);
			max_low[m] = next_low;
			max_high[m] = next_high;
			if (scale_low != 1.0f || scale_high != 1.0f)
			{
#pragma unroll
				for (frag = 0u; frag < FRAGS; frag++)
				{
					output[m][frag][0] *= scale_low;
					output[m][frag][1] *= scale_low;
					output[m][frag][2] *= scale_high;
					output[m][frag][3] *= scale_high;
				}
			}
			weight_high[m][0] = LmPackBf16Pair(w00, w01);
			weight_high[m][1] = LmPackBf16Pair(w02, w03);
			weight_high[m][2] = LmPackBf16Pair(w10, w11);
			weight_high[m][3] = LmPackBf16Pair(w12, w13);
		}
		{
			uint32_t b[4], first[2], second[2];
#pragma unroll
			for (frag = 0u; frag < FRAGS; frag += 2u)
			{
				LmPrefillLoadX4Trans(val_slot_base + b_val_row * VALUE_PITCH + ((half * HALF_COLUMNS + frag * 8u) / 8u + (matrix >> 1u)) * 16u, b);
				first[0] = b[0]; first[1] = b[1];
				second[0] = b[2]; second[1] = b[3];
#pragma unroll
				for (uint32_t m = 0u; m < ROW_TILES; m++)
				{
					LmMmaBf16(output[m][frag], weight_high[m], first);
					LmMmaBf16(output[m][frag + 1u], weight_high[m], second);
				}
			}
		}
		__syncthreads();
	}
	{
		const uint32_t column = half * HALF_COLUMNS + (lane & 3u) * 2u;
		float *source_partials = partials + (uint64_t)source_id * partial_source_stride;
#pragma unroll
		for (uint32_t m = 0u; m < ROW_TILES; m++)
		{
			const uint32_t rl = row + m * 32u + (tile_id & 1u) * 16u + (lane >> 2u);
			const uint32_t rh = rl + 8u;
			float *record_low = source_partials + ((uint64_t)rl * heads_per_rank + head + local_head) * RECORD;
			float *record_high = source_partials + ((uint64_t)rh * heads_per_rank + head + local_head) * RECORD;
			if (rl <= last_row)
			{
#pragma unroll
				for (frag = 0u; frag < FRAGS; frag++)
					__stcs((float2 *)(record_low + 2u + column + frag * 8u), make_float2(output[m][frag][0], output[m][frag][1]));
			}
			if (rh <= last_row)
			{
#pragma unroll
				for (frag = 0u; frag < FRAGS; frag++)
					__stcs((float2 *)(record_high + 2u + column + frag * 8u), make_float2(output[m][frag][2], output[m][frag][3]));
			}
			if (half == 0u && (lane & 3u) == 0u)
			{
				if (rl <= last_row)
					__stcs((float2 *)record_low, make_float2(max_low[m] == -INFINITY ? -INFINITY : max_low[m] * qk_scale, sum_low[m]));
				if (rh <= last_row)
					__stcs((float2 *)record_high, make_float2(max_high[m] == -INFINITY ? -INFINITY : max_high[m] * qk_scale, sum_high[m]));
			}
		}
	}
	LmDependentRelease();
}

template<class Geometry, uint32_t LATENT, uint32_t ROPE, uint32_t VALUE>
static inline SparkGpuError LmExpandedGatherPrefillSourcesLaunch(
	LmKvShardGatherView cache,
	uint32_t sources,
	uint64_t key_source_stride,
	const uint8_t *values,
	uint64_t value_source_stride,
	const uint16_t *query_bf16,
	uint32_t heads_per_rank,
	const uint32_t *sequence_of_row,
	const uint32_t *context_length,
	const uint32_t *row_position,
	float qk_scale,
	float *partials,
	uint64_t partial_source_stride,
	uint32_t rows,
	SparkGpuStream stream)
{
	constexpr uint32_t shared = LmExpandedPrefillSharedBytes<LATENT + ROPE, VALUE>();
	static bool granted = false;
	if (rows == 0u || heads_per_rank == 0u || (heads_per_rank % LM_EXPANDED_PREFILL_HEADS) != 0u || values == 0 || query_bf16 == 0 ||
		partials == 0 || row_position == 0 || cache.keys == 0 || cache.key_offset == 0 || cache.key_context == 0 ||
		SparkKvShardValid(cache.shard, Geometry::kPageSlots) == 0u || (cache.key_begin % LM_EXPANDED_PREFILL_TILE) != 0u ||
		sources == 0u || sources > 65535u || (uint64_t)cache.shard.rank + sources > cache.shard.degree ||
		(sources > 1u && (key_source_stride == 0u || value_source_stride == 0u ||
			partial_source_stride < (uint64_t)rows * heads_per_rank * LM_LATENT_SHARD_RECORD_FLOATS(VALUE))) ||
		(key_source_stride % 16u) != 0u || (value_source_stride % 16u) != 0u || (partial_source_stride % 2u) != 0u)
		SPARK_FAIL(SPARK_GPU_ERROR_INVALID_VALUE);
	if (!granted)
	{
		SparkGpuError error = SparkGpuFuncSetAttribute((const void *)LmExpandedGatherPrefillKernel<Geometry, LATENT, ROPE, VALUE>, SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)shared);
		if (error != SPARK_GPU_SUCCESS)
			return error;
		granted = true;
	}
	LM_LAUNCH_DEPENDENT((LmExpandedGatherPrefillKernel<Geometry, LATENT, ROPE, VALUE>), dim3((rows + LM_EXPANDED_PREFILL_ROWS - 1u) / LM_EXPANDED_PREFILL_ROWS, heads_per_rank / LM_EXPANDED_PREFILL_HEADS, sources), LM_EXPANDED_PREFILL_THREADS, shared, stream,
		cache, values, query_bf16, heads_per_rank, sequence_of_row, context_length, row_position, qk_scale, partials, rows,
		key_source_stride, value_source_stride, partial_source_stride);
	return SparkGpuPeekAtLastError();
}

template<class Geometry, uint32_t LATENT, uint32_t ROPE, uint32_t VALUE>
static inline SparkGpuError LmExpandedGatherPrefillLaunch(
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
	SparkGpuStream stream)
{
	return LmExpandedGatherPrefillSourcesLaunch<Geometry, LATENT, ROPE, VALUE>(cache, 1u, 0u, values, 0u, query_bf16, heads_per_rank,
		sequence_of_row, context_length, row_position, qk_scale, partials, 0u, rows, stream);
}
