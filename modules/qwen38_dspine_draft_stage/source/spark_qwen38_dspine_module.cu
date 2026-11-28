#include "inference/kernels/dspark_draft_stage_common.cuh"
#include "inference/kernels/mma.cuh"

#define Q38_HIDDEN 5120u
#define Q38_TAPS 5u
#define Q38_TAP_WIDTH (Q38_TAPS * Q38_HIDDEN)
#define Q38_LAYERS 5u
#define Q38_HEADS 32u
#define Q38_KV_HEADS 8u
#define Q38_HEAD_DIM 128u
#define Q38_HALF (Q38_HEAD_DIM / 2u)
#define Q38_GROUP (Q38_HEADS / Q38_KV_HEADS)
#define Q38_Q_WIDTH (Q38_HEADS * Q38_HEAD_DIM)
#define Q38_KV_WIDTH (Q38_KV_HEADS * Q38_HEAD_DIM)
#define Q38_MLP 17408u
#define Q38_BLOCK 16u
#define Q38_DRAFTS (Q38_BLOCK - 1u)
#define Q38_SLIDING 8192u
#define Q38_WINDOW (Q38_SLIDING - Q38_BLOCK)
#define Q38_COLUMNS (Q38_WINDOW + Q38_BLOCK)
#define Q38_VOCAB 248320u
#define Q38_CODE 1024u
#define Q38_RANK 512u
#define Q38_PACKED (3u * Q38_RANK)
#define Q38_LANES 16u
#define Q38_BRANCH_LEN 4u
#define Q38_POOL 64u
#define Q38_EARLY (Q38_LAYERS - 1u)
#define Q38_INGEST_ROWS 256u
#define Q38_SPLIT_KEYS 256u
#define Q38_SPLITS ((Q38_WINDOW + Q38_SPLIT_KEYS - 1u) / Q38_SPLIT_KEYS)
#define Q38_QUERIES (Q38_GROUP * Q38_BLOCK)
#define Q38_ATTENTION_TILE 32u
#define Q38_STREAM_OUTPUTS 4096u
#define Q38_ATTENTION_THREADS (Q38_GROUP * DSPARK_WARP)
#define Q38_SEQUENCES_MAX 256u
#define Q38_LAYER_KV ((uint64_t)Q38_KV_HEADS * Q38_COLUMNS * Q38_HEAD_DIM)
#define Q38_SEQUENCE_ELEMENTS (2u * Q38_LAYERS * Q38_LAYER_KV)
#define Q38_SEQUENCE_BYTES (Q38_SEQUENCE_ELEMENTS * sizeof(uint16_t))
#define Q38_CONFIDENCE_FLOOR 1e-6f
#define Q38_TINY 1.1754943508222875e-38f

typedef struct Q38Layer
{
	const uint16_t *input_norm;
	const uint16_t *post_norm;
	const uint16_t *q;
	const uint16_t *k;
	const uint16_t *v;
	const uint16_t *o;
	const uint16_t *q_norm;
	const uint16_t *k_norm;
	const uint16_t *gate;
	const uint16_t *up;
	const uint16_t *down;
	const uint16_t *read;
	const uint16_t *write;
} Q38Layer;

using Q38Sequence = DsparkRingSequence<Q38_WINDOW>;
using Q38RingPool = DsparkRingPool<Q38_WINDOW, Q38_SEQUENCES_MAX>;

typedef struct Q38Draft
{
	uint16_t *cache;
	uint64_t seed;
	uint32_t positions[Q38_BLOCK];
	uint32_t anchor;
	uint32_t mask;
	uint32_t greedy;
	uint32_t top_k;
	float inverse_temperature;
	float nucleus;
	uint32_t start;
	uint32_t block_at;
	uint32_t masked;
	uint32_t splits;
	uint32_t tree;
	uint8_t valid[Q38_WINDOW];
} Q38Draft;

typedef struct Q38Picks
{
	uint32_t tokens[Q38_DRAFTS];
	float scores[Q38_DRAFTS];
	uint32_t branch_tokens[Q38_DRAFTS][Q38_BRANCH_LEN];
	float branch_scores[Q38_DRAFTS][Q38_BRANCH_LEN];
} Q38Picks;

static __device__ __forceinline__ float Q38Sigmoid(float value)
{
	return 1.0f / (1.0f + expf(-value));
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void Q38DualRmsKernel(const uint16_t *__restrict__ input, const uint16_t *__restrict__ weight, uint16_t *__restrict__ out_fused,
	uint16_t *__restrict__ out_plain, float epsilon)
{
	LmDependentWait();
	__shared__ float reduction[DSPARK_WARPS];
	const uint64_t row = blockIdx.x;
	const uint16_t *src = input + row * Q38_HIDDEN;
	uint16_t *dst_f = out_fused + row * Q38_HIDDEN;
	uint16_t *dst_p = out_plain + row * Q38_HIDDEN;
	float total = 0.0f, scale;
	for ( uint32_t index = threadIdx.x; index < Q38_HIDDEN; index += DSPARK_THREADS )
	{
		const float value = LmBf16ToFloat(__ldg(src + index));
		total += value * value;
	}
	total = LmBlockSum<DSPARK_THREADS>(total, reduction);
	scale = rsqrtf(total / (float)Q38_HIDDEN + epsilon);
	for ( uint32_t index = threadIdx.x; index < Q38_HIDDEN; index += DSPARK_THREADS )
	{
		const float value = LmBf16ToFloat(__ldg(src + index)) * scale;
		const float w = LmBf16ToFloat(__ldg(weight + index));
		dst_f[index] = LmFloatToBf16(LmBf16ToFloat(LmFloatToBf16(value)) * w);
		dst_p[index] = LmFloatToBf16(value * w);
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void Q38ResidualRmsKernel(const float *__restrict__ hidden_f32, const uint16_t *__restrict__ hidden, const uint16_t *__restrict__ addend,
	const uint16_t *__restrict__ weight, uint16_t *__restrict__ sum_out, uint16_t *__restrict__ norm_out, float epsilon)
{
	__shared__ float reduction[DSPARK_WARPS];
	const uint64_t row = blockIdx.x;
	const float *src_f32 = hidden_f32 != 0 ? hidden_f32 + row * Q38_HIDDEN : (const float *)0;
	const uint16_t *src_h = hidden + row * Q38_HIDDEN;
	const uint16_t *src_a = addend + row * Q38_HIDDEN;
	uint16_t *dst_sum = sum_out + row * Q38_HIDDEN;
	uint16_t *dst_norm = norm_out + row * Q38_HIDDEN;
	float total = 0.0f, scale;
	for ( uint32_t index = threadIdx.x; index < Q38_HIDDEN; index += DSPARK_THREADS )
	{
		const float base = src_f32 != 0 ? LmBf16ToFloat(LmFloatToBf16(src_f32[index])) : LmBf16ToFloat(__ldg(src_h + index));
		const uint16_t sum_bf16 = LmFloatToBf16(base + LmBf16ToFloat(__ldg(src_a + index)));
		const float value = LmBf16ToFloat(sum_bf16);
		dst_sum[index] = sum_bf16;
		total += value * value;
	}
	total = LmBlockSum<DSPARK_THREADS>(total, reduction);
	scale = rsqrtf(total / (float)Q38_HIDDEN + epsilon);
	for ( uint32_t index = threadIdx.x; index < Q38_HIDDEN; index += DSPARK_THREADS )
		dst_norm[index] = LmFloatToBf16(LmBf16ToFloat(dst_sum[index]) * scale * LmBf16ToFloat(__ldg(weight + index)));
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void Q38IngestKeysKernel(const uint16_t *__restrict__ raw, const uint16_t *__restrict__ weight, const float *__restrict__ inverse,
	const uint32_t *__restrict__ positions, float epsilon, uint16_t *__restrict__ keys)
{
	__shared__ float reduction[DSPARK_WARPS];
	__shared__ float normed[Q38_KV_HEADS][Q38_HEAD_DIM];
	const uint32_t row = blockIdx.x, position = positions[row], slot = position % Q38_WINDOW;
	for ( uint32_t head = 0u; head < Q38_KV_HEADS; head += DSPARK_THREADS / Q38_HEAD_DIM )
	{
		const uint32_t own = head + threadIdx.x / Q38_HEAD_DIM, lane = threadIdx.x % Q38_HEAD_DIM;
		const float value = LmBf16ToFloat(__ldg(raw + ((uint64_t)row * Q38_KV_HEADS + own) * Q38_HEAD_DIM + lane));
		float total = DsparkWarpSum(value * value);
		if ( threadIdx.x % DSPARK_WARP == 0u )
			reduction[threadIdx.x / DSPARK_WARP] = total;
		__syncthreads();
		total = 0.0f;
		for ( uint32_t warp = 0u; warp < Q38_HEAD_DIM / DSPARK_WARP; warp++ )
			total += reduction[(threadIdx.x / Q38_HEAD_DIM) * (Q38_HEAD_DIM / DSPARK_WARP) + warp];
		normed[own][lane] = LmBf16ToFloat(LmFloatToBf16(value * rsqrtf(total / (float)Q38_HEAD_DIM + epsilon) * LmBf16ToFloat(__ldg(weight + lane))));
		__syncthreads();
	}
	for ( uint32_t index = threadIdx.x; index < Q38_KV_WIDTH; index += DSPARK_THREADS )
	{
		const uint32_t head = index / Q38_HEAD_DIM, lane = index % Q38_HEAD_DIM;
		float cosine, sine, rotated;
		DsparkRotationUnit(position, __ldg(inverse + (lane % Q38_HALF)), &cosine, &sine);
		rotated = lane < Q38_HALF ? normed[head][lane] * cosine - normed[head][lane + Q38_HALF] * sine :
			normed[head][lane] * cosine + normed[head][lane - Q38_HALF] * sine;
		keys[((uint64_t)head * Q38_COLUMNS + slot) * Q38_HEAD_DIM + lane] = LmFloatToBf16(rotated);
	}
}

__global__ __launch_bounds__(Q38_ATTENTION_THREADS, 1)
void Q38AttentionSplitKernel(const uint16_t *__restrict__ q, uint64_t keys_offset, uint64_t values_offset,
	const Q38Draft *__restrict__ draft, float scale, float *__restrict__ partials)
{
	LmDependentWait();
	constexpr uint32_t TILE = Q38_ATTENTION_TILE, ROW_BYTES = Q38_HEAD_DIM * 2u, CHUNKS = Q38_HEAD_DIM / 8u, K_STEPS = Q38_HEAD_DIM / 16u;
	constexpr uint32_t KEY_FRAGS = TILE / 8u, P_STEPS = TILE / 16u, VALUE_FRAGS = Q38_HEAD_DIM / 8u;
	__shared__ __align__(128) uint8_t key_tiles[2u * TILE * ROW_BYTES];
	__shared__ __align__(128) uint8_t value_tiles[2u * TILE * ROW_BYTES];
	__shared__ uint8_t visible[2u * TILE];
	const uint32_t group = blockIdx.x, split = blockIdx.y, warp = threadIdx.x / DSPARK_WARP, lane = threadIdx.x % DSPARK_WARP;
	const uint32_t matrix = lane >> 3u, within = lane & 7u, head = group * Q38_GROUP + warp, masked = draft->masked;
	const uint32_t first = draft->start + split * Q38_SPLIT_KEYS, end = draft->block_at;
	const uint32_t stop = first + Q38_SPLIT_KEYS < end ? first + Q38_SPLIT_KEYS : end, tiles = first < stop ? (stop - first + TILE - 1u) / TILE : 0u;
	const uint16_t *group_keys = draft->cache + keys_offset + (uint64_t)group * Q38_COLUMNS * Q38_HEAD_DIM;
	const uint16_t *group_values = draft->cache + values_offset + (uint64_t)group * Q38_COLUMNS * Q38_HEAD_DIM;
	const float scale_log2 = scale * 1.4426950408889634f;
	float running_max[2] = { -INFINITY, -INFINITY }, running_sum[2] = { 0.0f, 0.0f }, output[VALUE_FRAGS][4];
	uint32_t query[K_STEPS][4];
	float *base = partials + ((uint64_t)group * Q38_SPLITS + split) * Q38_QUERIES * (Q38_HEAD_DIM + 2u);
#pragma unroll
	for ( uint32_t step = 0u; step < K_STEPS; step++ )
#pragma unroll
		for ( uint32_t index = 0u; index < 4u; index++ )
			query[step][index] = *(const uint32_t *)(q + ((uint64_t)((lane >> 2u) + 8u * (index % 2u)) * Q38_HEADS + head) * Q38_HEAD_DIM + step * 16u +
				2u * (lane & 3u) + 8u * (index / 2u));
#pragma unroll
	for ( uint32_t frag = 0u; frag < VALUE_FRAGS; frag++ )
		output[frag][0] = output[frag][1] = output[frag][2] = output[frag][3] = 0.0f;
	auto load = [&](uint32_t tile, uint32_t buffer)
	{
		const uint32_t key_base = (uint32_t)__cvta_generic_to_shared(key_tiles + buffer * TILE * ROW_BYTES);
		const uint32_t value_base = (uint32_t)__cvta_generic_to_shared(value_tiles + buffer * TILE * ROW_BYTES);
		const uint32_t tile_first = first + tile * TILE;
		for ( uint32_t item = threadIdx.x; item < TILE * 2u * CHUNKS; item += Q38_ATTENTION_THREADS )
		{
			const uint32_t key = item / (2u * CHUNKS), piece = item % (2u * CHUNKS), column = tile_first + key, inside = column < stop ? 1u : 0u;
			const uint64_t row = (uint64_t)(inside != 0u ? column : first) * Q38_HEAD_DIM;
			if ( piece < CHUNKS )
				DsparkStage16Cg(key_base + key * ROW_BYTES + DsparkSwizzle(key, piece) * 16u, group_keys + row + piece * 8u, inside != 0u ? 16u : 0u);
			else
				DsparkStage16Cg(value_base + key * ROW_BYTES + DsparkSwizzle(key, piece - CHUNKS) * 16u, group_values + row + (piece - CHUNKS) * 8u,
					inside != 0u ? 16u : 0u);
		}
		if ( threadIdx.x < TILE )
			visible[buffer * TILE + threadIdx.x] = tile_first + threadIdx.x < stop && (masked == 0u || draft->valid[tile_first + threadIdx.x] != 0u) ? 1u : 0u;
		asm volatile("cp.async.commit_group;\n" ::: "memory");
	};
	if ( tiles != 0u )
		load(0u, 0u);
	for ( uint32_t tile = 0u; tile < tiles; tile++ )
	{
		const uint32_t buffer = tile & 1u;
		const uint32_t tile_keys = (uint32_t)__cvta_generic_to_shared(key_tiles + buffer * TILE * ROW_BYTES);
		const uint32_t tile_values = (uint32_t)__cvta_generic_to_shared(value_tiles + buffer * TILE * ROW_BYTES);
		float score[KEY_FRAGS][4], peak[2], rescale[2], shift[2];
		if ( tile + 1u < tiles )
		{
			load(tile + 1u, buffer ^ 1u);
			asm volatile("cp.async.wait_group 1;\n" ::: "memory");
		}
		else
			asm volatile("cp.async.wait_group 0;\n" ::: "memory");
		__syncthreads();
#pragma unroll
		for ( uint32_t frag = 0u; frag < KEY_FRAGS; frag++ )
			score[frag][0] = score[frag][1] = score[frag][2] = score[frag][3] = 0.0f;
#pragma unroll
		for ( uint32_t step = 0u; step < K_STEPS; step++ )
#pragma unroll
			for ( uint32_t frag = 0u; frag < KEY_FRAGS; frag += 2u )
			{
				const uint32_t b_row = (frag + (matrix >> 1u)) * 8u + within, b_chunk = 2u * step + (matrix & 1u);
				uint32_t b[4];
				DsparkLoadX4(tile_keys + b_row * ROW_BYTES + DsparkSwizzle(b_row, b_chunk) * 16u, b);
				LmMmaBf16(score[frag], query[step], b);
				LmMmaBf16(score[frag + 1u], query[step], b + 2);
			}
		peak[0] = peak[1] = -INFINITY;
#pragma unroll
		for ( uint32_t frag = 0u; frag < KEY_FRAGS; frag++ )
#pragma unroll
			for ( uint32_t entry = 0u; entry < 4u; entry++ )
			{
				const uint32_t key = frag * 8u + 2u * (lane & 3u) + (entry & 1u);
				score[frag][entry] = visible[buffer * TILE + key] != 0u ? score[frag][entry] : -INFINITY;
				peak[entry / 2u] = fmaxf(peak[entry / 2u], score[frag][entry]);
			}
#pragma unroll
		for ( uint32_t side = 0u; side < 2u; side++ )
		{
			float next;
			peak[side] = fmaxf(peak[side], __shfl_xor_sync(0xffffffffu, peak[side], 1u));
			peak[side] = fmaxf(peak[side], __shfl_xor_sync(0xffffffffu, peak[side], 2u));
			next = fmaxf(running_max[side], peak[side]);
			rescale[side] = running_max[side] == -INFINITY ? 0.0f : SparkMathExp2Fast((running_max[side] - next) * scale_log2);
			running_max[side] = next;
			shift[side] = next == -INFINITY ? 0.0f : next * scale_log2;
			running_sum[side] *= rescale[side];
		}
#pragma unroll
		for ( uint32_t frag = 0u; frag < VALUE_FRAGS; frag++ )
		{
			output[frag][0] *= rescale[0];
			output[frag][1] *= rescale[0];
			output[frag][2] *= rescale[1];
			output[frag][3] *= rescale[1];
		}
#pragma unroll
		for ( uint32_t frag = 0u; frag < KEY_FRAGS; frag++ )
#pragma unroll
			for ( uint32_t entry = 0u; entry < 4u; entry++ )
			{
				const float weight = SparkMathExp2Fast(fmaf(score[frag][entry], scale_log2, -shift[entry / 2u]));
				running_sum[entry / 2u] += weight;
				score[frag][entry] = weight;
			}
#pragma unroll
		for ( uint32_t step = 0u; step < P_STEPS; step++ )
		{
			uint32_t weights[4];
#pragma unroll
			for ( uint32_t index = 0u; index < 4u; index++ )
				weights[index] = LmPackBf16Pair(score[2u * step + index / 2u][2u * (index % 2u)], score[2u * step + index / 2u][2u * (index % 2u) + 1u]);
#pragma unroll
			for ( uint32_t frag = 0u; frag < VALUE_FRAGS; frag += 2u )
			{
				const uint32_t b_row = step * 16u + (matrix & 1u) * 8u + within, b_chunk = frag + (matrix >> 1u);
				uint32_t b[4];
				DsparkLoadX4Trans(tile_values + b_row * ROW_BYTES + DsparkSwizzle(b_row, b_chunk) * 16u, b);
				LmMmaBf16(output[frag], weights, b);
				LmMmaBf16(output[frag + 1u], weights, b + 2);
			}
		}
		__syncthreads();
	}
#pragma unroll
	for ( uint32_t side = 0u; side < 2u; side++ )
	{
		float *out = base + (uint64_t)(warp * Q38_BLOCK + (lane >> 2u) + 8u * side) * (Q38_HEAD_DIM + 2u);
		running_sum[side] += __shfl_xor_sync(0xffffffffu, running_sum[side], 1u);
		running_sum[side] += __shfl_xor_sync(0xffffffffu, running_sum[side], 2u);
#pragma unroll
		for ( uint32_t frag = 0u; frag < VALUE_FRAGS; frag++ )
		{
			out[frag * 8u + 2u * (lane & 3u)] = output[frag][2u * side];
			out[frag * 8u + 2u * (lane & 3u) + 1u] = output[frag][2u * side + 1u];
		}
		if ( (lane & 3u) == 0u )
		{
			out[Q38_HEAD_DIM] = running_max[side] == -INFINITY ? -INFINITY : running_max[side] * scale;
			out[Q38_HEAD_DIM + 1u] = running_sum[side];
		}
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void Q38ResidualBridgeScaleKernel(const uint16_t *__restrict__ residual, const uint16_t *__restrict__ mlp, uint16_t *__restrict__ raw,
	float *__restrict__ rms, uint16_t *__restrict__ normalized, float epsilon)
{
	__shared__ float reduction[DSPARK_WARPS];
	const uint64_t row = blockIdx.x;
	const uint16_t *src_r = residual + row * Q38_HIDDEN, *src_m = mlp + row * Q38_HIDDEN;
	uint16_t *dst_raw = raw + row * Q38_HIDDEN, *dst_norm = normalized + row * Q38_HIDDEN;
	float total = 0.0f, scale;
	for ( uint32_t index = threadIdx.x; index < Q38_HIDDEN; index += DSPARK_THREADS )
	{
		const uint16_t sum_bf16 = LmFloatToBf16(LmBf16ToFloat(__ldg(src_r + index)) + LmBf16ToFloat(__ldg(src_m + index)));
		const float value = LmBf16ToFloat(sum_bf16);
		dst_raw[index] = sum_bf16;
		total += value * value;
	}
	total = LmBlockSum<DSPARK_THREADS>(total, reduction);
	scale = sqrtf(fmaxf(total / (float)Q38_HIDDEN, Q38_TINY));
	if ( threadIdx.x == 0u )
		rms[row] = scale;
	const float denom = scale + epsilon;
	for ( uint32_t index = threadIdx.x; index < Q38_HIDDEN; index += DSPARK_THREADS )
		dst_norm[index] = LmFloatToBf16(LmBf16ToFloat(dst_raw[index]) / denom);
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void Q38BridgeCodesKernel(const uint16_t *__restrict__ projected, const uint16_t *__restrict__ codebook, const Q38Draft *__restrict__ draft,
	float epsilon, uint16_t *__restrict__ codes)
{
	__shared__ float reduction[DSPARK_WARPS];
	const uint64_t row = blockIdx.x;
	const float root = sqrtf((float)Q38_CODE);
	float total = 0.0f, length;
	if ( row == 0u )
	{
		for ( uint32_t index = threadIdx.x; index < Q38_CODE; index += DSPARK_THREADS )
			codes[index] = LmFloatToBf16(root * LmBf16ToFloat(__ldg(codebook + (uint64_t)draft->anchor * Q38_CODE + index)));
		LmDependentRelease();
		return;
	}
	for ( uint32_t index = threadIdx.x; index < Q38_CODE; index += DSPARK_THREADS )
	{
		const float value = LmBf16ToFloat(__ldg(projected + row * Q38_CODE + index));
		total += value * value;
	}
	length = sqrtf(LmBlockSum<DSPARK_THREADS>(total, reduction));
	for ( uint32_t index = threadIdx.x; index < Q38_CODE; index += DSPARK_THREADS )
		codes[row * Q38_CODE + index] = LmFloatToBf16(root * LmBf16ToFloat(__ldg(projected + row * Q38_CODE + index)) / (length + epsilon));
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void Q38BridgeMessageKernel(const uint16_t *__restrict__ packed, const uint16_t *__restrict__ bias, uint16_t *__restrict__ message)
{
	const uint32_t row = blockIdx.x;
	for ( uint32_t index = threadIdx.x; index < Q38_RANK; index += DSPARK_THREADS )
	{
		float value = 0.0f;
		if ( row != 0u )
		{
			const float query = LmBf16ToFloat(__ldg(packed + (uint64_t)row * Q38_PACKED + index));
			const float sender = LmBf16ToFloat(__ldg(packed + (uint64_t)(row - 1u) * Q38_PACKED + Q38_RANK + index));
			const float prior = LmBf16ToFloat(__ldg(packed + (uint64_t)(row - 1u) * Q38_PACKED + 2u * Q38_RANK + index));
			value = Q38Sigmoid(query + sender + LmBf16ToFloat(__ldg(bias + index))) * prior;
		}
		message[(uint64_t)row * Q38_RANK + index] = LmFloatToBf16(value);
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void Q38BridgeOutKernel(const uint16_t *__restrict__ raw, const float *__restrict__ rms, const uint16_t *__restrict__ delta, uint16_t *__restrict__ output)
{
	const uint64_t row = blockIdx.x;
	const float scale = rms[row];
	for ( uint32_t index = threadIdx.x; index < Q38_HIDDEN; index += DSPARK_THREADS )
	{
		const float value = LmBf16ToFloat(__ldg(raw + row * Q38_HIDDEN + index));
		output[row * Q38_HIDDEN + index] = LmFloatToBf16(row == 0u ? value : value + scale * LmBf16ToFloat(__ldg(delta + row * Q38_HIDDEN + index)));
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void Q38PoolKernel(const uint32_t *__restrict__ pooled, const uint16_t *__restrict__ normed, const uint16_t *__restrict__ head,
	uint32_t *__restrict__ candidates)
{
	LmDependentWait();
	__shared__ uint32_t tokens[Q38_POOL];
	__shared__ uint32_t sorted[Q38_POOL];
	__shared__ float exact[Q38_POOL];
	__shared__ uint32_t chosen[Q38_LANES];
	const uint32_t row = blockIdx.x, warp = threadIdx.x / DSPARK_WARP, lane = threadIdx.x % DSPARK_WARP;
	const uint16_t *own = normed + (uint64_t)(row + 1u) * Q38_HIDDEN;
	if ( threadIdx.x < Q38_POOL )
		tokens[threadIdx.x] = pooled[(uint64_t)row * Q38_POOL + threadIdx.x];
	__syncthreads();
	if ( threadIdx.x < Q38_POOL )
	{
		uint32_t rank = 0u;
		for ( uint32_t other = 0u; other < Q38_POOL; other++ )
			rank += tokens[other] < tokens[threadIdx.x] ? 1u : 0u;
		sorted[rank] = tokens[threadIdx.x];
	}
	__syncthreads();
	for ( uint32_t item = warp; item < Q38_POOL; item += DSPARK_WARPS )
	{
		const uint16_t *weights = head + (uint64_t)sorted[item] * Q38_HIDDEN;
		float dot = 0.0f;
		for ( uint32_t index = lane; index < Q38_HIDDEN; index += DSPARK_WARP )
			dot += LmBf16ToFloat(__ldg(own + index)) * LmBf16ToFloat(__ldg(weights + index));
		dot = DsparkWarpSum(dot);
		if ( lane == 0u )
			exact[item] = LmBf16ToFloat(LmFloatToBf16(dot));
	}
	__syncthreads();
	if ( threadIdx.x < Q38_POOL )
	{
		const float value = exact[threadIdx.x];
		uint32_t rank = 0u;
		for ( uint32_t other = 0u; other < Q38_POOL; other++ )
			rank += exact[other] > value || (exact[other] == value && other < threadIdx.x) ? 1u : 0u;
		if ( rank < Q38_LANES )
			chosen[rank] = sorted[threadIdx.x];
	}
	__syncthreads();
	if ( threadIdx.x < Q38_LANES )
	{
		uint32_t rank = 0u;
		for ( uint32_t other = 0u; other < Q38_LANES; other++ )
			rank += chosen[other] < chosen[threadIdx.x] ? 1u : 0u;
		candidates[(uint64_t)row * Q38_LANES + rank] = chosen[threadIdx.x];
	}
	LmDependentRelease();
}

static __device__ __forceinline__ uint32_t Q38Sender(const Q38Draft *__restrict__ draft, const uint32_t *__restrict__ candidates, uint32_t row,
	uint32_t lane)
{
	return row == 0u ? draft->anchor : candidates[(uint64_t)(row - 1u) * Q38_LANES + lane];
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void Q38GreedyTableKernel(const Q38Draft *__restrict__ draft, const uint32_t *__restrict__ candidates, const uint16_t *__restrict__ raw,
	const float *__restrict__ rms, const uint16_t *__restrict__ query, const uint16_t *__restrict__ norm, const uint16_t *__restrict__ head,
	const uint16_t *__restrict__ tg, const uint16_t *__restrict__ tsv, const uint16_t *__restrict__ bias, uint32_t *__restrict__ table,
	uint32_t *__restrict__ alt_table, float *__restrict__ confidence, float *__restrict__ alt_confidence)
{
	LmDependentWait();
	__shared__ float base[Q38_LANES];
	__shared__ float message[Q38_LANES][Q38_RANK];
	__shared__ float scores[Q38_LANES][Q38_LANES];
	const uint32_t row = blockIdx.x, warp = threadIdx.x / DSPARK_WARP, lane = threadIdx.x % DSPARK_WARP, source = row + 1u;
	const float scale = rms[source];
	for ( uint32_t item = warp; item < Q38_LANES; item += DSPARK_WARPS )
	{
		const uint16_t *weights = head + (uint64_t)candidates[(uint64_t)row * Q38_LANES + item] * Q38_HIDDEN;
		float dot = 0.0f;
		for ( uint32_t index = lane; index < Q38_HIDDEN; index += DSPARK_WARP )
			dot += LmBf16ToFloat(__ldg(weights + index)) * (LmBf16ToFloat(__ldg(raw + (uint64_t)source * Q38_HIDDEN + index)) * LmBf16ToFloat(__ldg(norm + index)));
		dot = DsparkWarpSum(dot);
		if ( lane == 0u )
			base[item] = dot;
	}
	for ( uint32_t index = threadIdx.x; index < Q38_LANES * Q38_RANK; index += DSPARK_THREADS )
	{
		const uint32_t sender = index / Q38_RANK, column = index % Q38_RANK;
		const uint16_t *pair = tsv + (uint64_t)Q38Sender(draft, candidates, row, sender) * (2u * Q38_RANK);
		message[sender][column] = Q38Sigmoid(LmBf16ToFloat(__ldg(query + (uint64_t)source * Q38_PACKED + column)) + LmBf16ToFloat(__ldg(pair + column)) +
			LmBf16ToFloat(__ldg(bias + column))) * LmBf16ToFloat(__ldg(pair + Q38_RANK + column));
	}
	__syncthreads();
	for ( uint32_t item = warp; item < Q38_LANES * Q38_LANES; item += DSPARK_WARPS )
	{
		const uint32_t sender = item / Q38_LANES, choice = item % Q38_LANES;
		const uint16_t *weights = tg + (uint64_t)candidates[(uint64_t)row * Q38_LANES + choice] * Q38_RANK;
		float dot = 0.0f;
		for ( uint32_t index = lane; index < Q38_RANK; index += DSPARK_WARP )
			dot += message[sender][index] * LmBf16ToFloat(__ldg(weights + index));
		dot = DsparkWarpSum(dot);
		if ( lane == 0u )
			scores[sender][choice] = base[choice] + scale * dot;
	}
	__syncthreads();
	if ( threadIdx.x < Q38_LANES )
	{
		const uint32_t sender = threadIdx.x;
		const float divisor = fmaxf(scale, Q38_CONFIDENCE_FLOOR);
		float best = scores[sender][0], second = -INFINITY, total = 0.0f;
		uint32_t pick = 0u, alt_pick = 1u;
		for ( uint32_t choice = 1u; choice < Q38_LANES; choice++ )
		{
			const float value = scores[sender][choice];
			if ( value > best )
			{
				second = best;
				alt_pick = pick;
				best = value;
				pick = choice;
			}
			else if ( value > second )
			{
				second = value;
				alt_pick = choice;
			}
		}
		for ( uint32_t choice = 0u; choice < Q38_LANES; choice++ )
			total += expf((scores[sender][choice] - best) / divisor);
		table[(uint64_t)row * Q38_LANES + sender] = pick;
		alt_table[(uint64_t)row * Q38_LANES + sender] = alt_pick;
		confidence[(uint64_t)row * Q38_LANES + sender] = 1.0f / total;
		alt_confidence[(uint64_t)row * Q38_LANES + sender] = expf((second - best) / divisor) / total;
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_THREADS, 2)
void Q38RefinedKernel(const Q38Draft *__restrict__ draft, const uint32_t *__restrict__ candidates, const uint16_t *__restrict__ raw,
	const float *__restrict__ rms, const uint16_t *__restrict__ query, const uint16_t *__restrict__ codebook, const uint16_t *__restrict__ sender_weight,
	const uint16_t *__restrict__ value_weight, const uint16_t *__restrict__ bias, const uint16_t *__restrict__ write, const uint16_t *__restrict__ norm,
	const uint16_t *__restrict__ head, float epsilon, float *__restrict__ refined_table)
{
	LmDependentWait();
	__shared__ float codes[Q38_CODE];
	__shared__ float message[Q38_RANK];
	__shared__ float refined[Q38_HIDDEN];
	__shared__ float reduction[DSPARK_WARPS];
	const uint32_t row = blockIdx.x / Q38_LANES, sender = blockIdx.x % Q38_LANES, source = row + 1u;
	if ( row == 0u && sender != 0u )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t warp = threadIdx.x / DSPARK_WARP, lane = threadIdx.x % DSPARK_WARP, token = Q38Sender(draft, candidates, row, sender);
	const float root = sqrtf((float)Q38_CODE), scale = rms[source];
	float total = 0.0f, inverse;
	for ( uint32_t index = threadIdx.x; index < Q38_CODE; index += DSPARK_THREADS )
		codes[index] = root * LmBf16ToFloat(__ldg(codebook + (uint64_t)token * Q38_CODE + index));
	__syncthreads();
	for ( uint32_t output = warp; output < Q38_RANK; output += DSPARK_WARPS )
	{
		const uint16_t *gate_row = sender_weight + (uint64_t)output * Q38_CODE, *value_row = value_weight + (uint64_t)output * Q38_CODE;
		float gate = 0.0f, value = 0.0f;
		for ( uint32_t index = lane; index < Q38_CODE; index += DSPARK_WARP )
		{
			const float c = codes[index];
			gate += c * LmBf16ToFloat(__ldg(gate_row + index));
			value += c * LmBf16ToFloat(__ldg(value_row + index));
		}
		gate = DsparkWarpSum(gate);
		value = DsparkWarpSum(value);
		if ( lane == 0u )
			message[output] = Q38Sigmoid(LmBf16ToFloat(__ldg(query + (uint64_t)source * Q38_PACKED + output)) + gate + LmBf16ToFloat(__ldg(bias + output))) * value;
	}
	__syncthreads();
	for ( uint32_t output = warp; output < Q38_HIDDEN; output += DSPARK_WARPS )
	{
		const uint16_t *write_row = write + (uint64_t)output * Q38_RANK;
		float dot = 0.0f;
		for ( uint32_t index = lane; index < Q38_RANK; index += DSPARK_WARP )
			dot += message[index] * LmBf16ToFloat(__ldg(write_row + index));
		dot = DsparkWarpSum(dot);
		if ( lane == 0u )
			refined[output] = LmBf16ToFloat(__ldg(raw + (uint64_t)source * Q38_HIDDEN + output)) + scale * dot;
	}
	__syncthreads();
	for ( uint32_t index = threadIdx.x; index < Q38_HIDDEN; index += DSPARK_THREADS )
		total += refined[index] * refined[index];
	inverse = 1.0f / sqrtf(LmBlockSum<DSPARK_THREADS>(total, reduction) / (float)Q38_HIDDEN + epsilon);
	for ( uint32_t index = threadIdx.x; index < Q38_HIDDEN; index += DSPARK_THREADS )
		refined[index] = refined[index] * inverse * LmBf16ToFloat(__ldg(norm + index));
	__syncthreads();
	for ( uint32_t choice = warp; choice < Q38_LANES; choice += DSPARK_WARPS )
	{
		const uint16_t *weights = head + (uint64_t)candidates[(uint64_t)row * Q38_LANES + choice] * Q38_HIDDEN;
		float dot = 0.0f;
		for ( uint32_t index = lane; index < Q38_HIDDEN; index += DSPARK_WARP )
			dot += refined[index] * LmBf16ToFloat(__ldg(weights + index));
		dot = DsparkWarpSum(dot);
		if ( lane == 0u )
			refined_table[((uint64_t)row * Q38_LANES + sender) * Q38_LANES + choice] = dot;
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_WARP, 1)
void Q38ChainKernel(const Q38Draft *__restrict__ draft, const uint32_t *__restrict__ candidates, const uint32_t *__restrict__ table,
	const uint32_t *__restrict__ alt_table, const float *__restrict__ confidence, const float *__restrict__ alt_confidence,
	const float *__restrict__ refined_table, Q38Picks *__restrict__ picks)
{
	LmDependentWait();
	const uint32_t lane = threadIdx.x;
	const uint32_t tree = draft->tree;
	uint32_t choice = 0u, previous = 0u, sampled_lane = 0u;
	if ( draft->greedy != 0u || lane < Q38_LANES )
	{
		if ( lane == 0u || draft->greedy == 0u )
			choice = table[0];
		for ( uint32_t row = 0u; row < Q38_DRAFTS; row++ )
		{
			const uint32_t greedy_token = candidates[(uint64_t)row * Q38_LANES + choice];
			const float score = confidence[(uint64_t)row * Q38_LANES + previous];
			uint32_t token = greedy_token;
			uint32_t alt_lane = alt_table[(uint64_t)row * Q38_LANES + previous];
			uint32_t alt_token = candidates[(uint64_t)row * Q38_LANES + alt_lane];
			float alt_score = alt_confidence[(uint64_t)row * Q38_LANES + previous];
			if ( draft->greedy == 0u )
			{
				const float value = refined_table[((uint64_t)row * Q38_LANES + sampled_lane) * Q38_LANES + lane] * draft->inverse_temperature;
				const uint32_t candidate = candidates[(uint64_t)row * Q38_LANES + lane];
				float alt_prob = 0.0f;
				DsparkWarpSampleSubwarp<Q38_LANES>(value, candidate, lane, draft->seed, draft->positions[0] + row, draft->nucleus, draft->top_k,
					&sampled_lane, &alt_lane, &alt_prob);
				token = candidates[(uint64_t)row * Q38_LANES + sampled_lane];
				alt_token = candidates[(uint64_t)row * Q38_LANES + alt_lane];
				alt_score = alt_prob;
			}
			if ( lane == 0u )
			{
				picks->tokens[row] = token;
				picks->scores[row] = score;
				picks->branch_tokens[row][0] = alt_token;
				picks->branch_scores[row][0] = alt_score;
			}
			if ( tree != 0u )
			{
				uint32_t branch_lane = alt_lane;
				for ( uint32_t k = 1u; k < Q38_BRANCH_LEN && row + k < Q38_DRAFTS; k++ )
				{
					if ( draft->greedy != 0u )
					{
						if ( lane == 0u )
						{
							const uint64_t next_at = (uint64_t)(row + k) * Q38_LANES + branch_lane;
							branch_lane = table[next_at];
							picks->branch_tokens[row][k] = candidates[(uint64_t)(row + k) * Q38_LANES + branch_lane];
							picks->branch_scores[row][k] = confidence[next_at];
						}
					}
					else
					{
						const uint32_t active = (1u << Q38_LANES) - 1u;
						const float value = refined_table[((uint64_t)(row + k) * Q38_LANES + branch_lane) * Q38_LANES + lane] * draft->inverse_temperature;
						float top = value, denom;
						unsigned long long key = ((unsigned long long)SparkMathSampleOrdered(value) << 32u) | (unsigned long long)(0xffffffffu - lane);
						for ( uint32_t mask = Q38_LANES / 2u; mask > 0u; mask >>= 1u )
						{
							const unsigned long long other = __shfl_xor_sync(active, key, mask);
							key = other > key ? other : key;
							top = fmaxf(top, __shfl_xor_sync(active, top, mask));
						}
						denom = expf(value - top);
						for ( uint32_t mask = Q38_LANES / 2u; mask > 0u; mask >>= 1u )
							denom += __shfl_xor_sync(active, denom, mask);
						branch_lane = 0xffffffffu - (uint32_t)key;
						if ( lane == 0u )
						{
							picks->branch_tokens[row][k] = candidates[(uint64_t)(row + k) * Q38_LANES + branch_lane];
							picks->branch_scores[row][k] = denom > 0.0f ? 1.0f / denom : 0.0f;
						}
					}
				}
			}
			previous = choice;
			if ( row + 1u < Q38_DRAFTS )
				choice = table[(uint64_t)(row + 1u) * Q38_LANES + choice];
		}
	}
	LmDependentRelease();
}

__global__ __launch_bounds__(DSPARK_THREADS, 1)
void Q38ScaledRowsKernel(const uint16_t *__restrict__ input, const uint16_t *__restrict__ weight, float factor, uint32_t width, uint64_t count,
	uint16_t *__restrict__ output)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * DSPARK_THREADS + threadIdx.x; index < count; index += (uint64_t)gridDim.x * DSPARK_THREADS )
		output[index] = LmFloatToBf16(LmBf16ToFloat(input[index]) * (weight != 0 ? LmBf16ToFloat(weight[index % width]) : factor));
}

__global__ __launch_bounds__(DSPARK_THREADS, 1)
void Q38TransposeKernel(const uint16_t *__restrict__ input, uint32_t rows, uint32_t columns, uint16_t *__restrict__ output)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * DSPARK_THREADS + threadIdx.x; index < (uint64_t)rows * columns; index += (uint64_t)gridDim.x * DSPARK_THREADS )
		output[(index % columns) * rows + index / columns] = input[index];
}

typedef struct Q38Forward
{
	Q38Draft *host_draft;
	Q38Draft *draft;
	Q38Picks *host_picks;
	Q38Picks *picks;
	float *hidden_f32;
	uint16_t *hidden;
	uint16_t *x;
	uint16_t *q;
	uint16_t *q_roped;
	uint16_t *k;
	uint16_t *v;
	float *partials;
	uint16_t *attended;
	uint16_t *projected;
	uint16_t *residual;
	uint16_t *gate;
	uint16_t *up;
	uint16_t *act;
	uint16_t *mlp;
	uint16_t *raw;
	float *rms;
	uint16_t *bridge_normed;
	uint16_t *codes_projected;
	uint16_t *codes;
	uint16_t *packed;
	uint16_t *message;
	uint16_t *delta;
	uint16_t *normed;
	uint16_t *y;
	uint16_t *base;
	SparkDrafterTopkBf16Workspace topk_ws;
	uint32_t *pooled;
	uint32_t *candidates;
	uint32_t *table;
	uint32_t *alt_table;
	float *confidence;
	float *alt_confidence;
	float *refined_table;
} Q38Forward;

typedef struct Q38State
{
	SparkWeightdLazyPack *pack;
	SparkGpuStream stream;
	uint32_t multiprocessors;
	uint32_t mask_token;
	float epsilon;
	float bridge_epsilon;
	float softmax_scale;
	float *rope_inverse;
	const uint16_t *fc;
	const uint16_t *hidden_norm;
	const uint16_t *norm;
	const uint16_t *embed;
	const uint16_t *head;
	const uint16_t *codebook;
	const uint16_t *gate_bias;
	const uint16_t *bridge_gate;
	const uint16_t *bridge_sender;
	const uint16_t *bridge_value;
	Q38Layer layers[Q38_LAYERS];
	uint16_t *packed_weight;
	uint16_t *write_transposed;
	uint16_t *tg;
	uint16_t *tsv;
	Q38Forward forward;
	SparkDrafterGraphCache graphs;
	uint16_t *host_taps;
	uint32_t *host_positions;
	uint16_t *taps;
	uint32_t *positions;
	uint16_t *features_raw;
	uint16_t *features;
	uint16_t *keys_raw;
	uint16_t *values_raw;
	uint16_t *caches;
	Q38RingPool ring;
	uint64_t frames;
	uint64_t rows;
} Q38State;

static SparkStatus Q38Shape(Q38State *state, const SparkJsonDocument *document)
{
	const int32_t config = DsparkMember(document, SparkJsonGetRootToken(document), "config");
	const int32_t checkpoint = DsparkMember(document, config, "checkpoint_config");
	const int32_t dflash = DsparkMember(document, checkpoint, "dflash_config");
	const int32_t bridge = DsparkMember(document, dflash, "selected_prefix_code_bridge");
	const int32_t second = DsparkMember(document, checkpoint, "code_bridge_second_pass");
	const int32_t rope = DsparkMember(document, checkpoint, "rope_parameters");
	const DsparkExpectedU64 expected[] = {
		{checkpoint, "hidden_size", Q38_HIDDEN}, {checkpoint, "num_hidden_layers", Q38_LAYERS}, {checkpoint, "num_attention_heads", Q38_HEADS},
		{checkpoint, "num_key_value_heads", Q38_KV_HEADS}, {checkpoint, "head_dim", Q38_HEAD_DIM}, {checkpoint, "intermediate_size", Q38_MLP},
		{checkpoint, "block_size", Q38_BLOCK}, {checkpoint, "sliding_window", Q38_SLIDING}, {checkpoint, "vocab_size", Q38_VOCAB},
		{bridge, "code_size", Q38_CODE}, {bridge, "rank", Q38_RANK}, {second, "head_topk", Q38_LANES}, {second, "first_exact", Q38_LAYERS - 1u},
	};
	uint64_t value = 0u;
	float theta = 0.0f;
	int32_t token;
	SparkStatus status = DsparkCheckU64Table(document, "qwen38 dspine", expected, (uint32_t)(sizeof(expected) / sizeof(expected[0])));
	if ( status != SPARK_STATUS_OK )
		return status;
	token = DsparkMember(document, dflash, "mask_token_id");
	if ( token < 0 || SparkJsonGetUInt64(document, token, &value) != SPARK_STATUS_OK || value >= Q38_VOCAB )
		return SPARK_STATUS_SCHEMA_ERROR;
	state->mask_token = (uint32_t)value;
	token = DsparkMember(document, checkpoint, "rms_norm_eps");
	if ( token < 0 || SparkJsonGetFloat(document, token, &state->epsilon) != SPARK_STATUS_OK )
		return SPARK_STATUS_SCHEMA_ERROR;
	token = DsparkMember(document, bridge, "eps");
	if ( token < 0 || SparkJsonGetFloat(document, token, &state->bridge_epsilon) != SPARK_STATUS_OK )
		return SPARK_STATUS_SCHEMA_ERROR;
	token = DsparkMember(document, rope, "rope_theta");
	if ( token < 0 || SparkJsonGetFloat(document, token, &theta) != SPARK_STATUS_OK )
		return SPARK_STATUS_SCHEMA_ERROR;
	state->softmax_scale = (float)(1.0 / sqrt((double)Q38_HEAD_DIM));
	if ( SparkGpuMalloc((void **)&state->rope_inverse, Q38_HALF * sizeof(float)) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	LM_LAUNCH((DsparkRopeTableKernel<Q38_HALF, Q38_HEAD_DIM>), 1u, DSPARK_THREADS, 0, state->stream, state->rope_inverse, theta);
	return SparkGpuGetLastError() == SPARK_GPU_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus Q38Weights(Q38State *state, const char *pack_path)
{
	SparkJsonDocument document;
	uint64_t payload_base = 0u;
	const uint16_t *early = 0, *last_read = 0, *writes = 0;
	SparkStatus status;
	memset(&document, 0, sizeof(document));
	status = DsparkLoadManifest(pack_path, &document, &payload_base);
	if ( status == SPARK_STATUS_OK )
		status = Q38Shape(state, &document);
	if ( status == SPARK_STATUS_OK )
	{
		const DsparkBf16Spec plain[] = {
			{"fc.weight", (uint64_t)Q38_HIDDEN * Q38_TAP_WIDTH, &state->fc}, {"hidden_norm.weight", Q38_HIDDEN, &state->hidden_norm},
			{"norm.weight", Q38_HIDDEN, &state->norm}, {"target.embed", (uint64_t)Q38_VOCAB * Q38_HIDDEN, &state->embed},
			{"target.head", (uint64_t)Q38_VOCAB * Q38_HIDDEN, &state->head},
			{"selected_prefix_bridge.codebook", (uint64_t)Q38_VOCAB * Q38_CODE, &state->codebook},
			{"selected_prefix_bridge.early_read_weights", (uint64_t)Q38_EARLY * Q38_CODE * Q38_HIDDEN, &early},
			{"selected_prefix_bridge.read_weight", (uint64_t)Q38_CODE * Q38_HIDDEN, &last_read},
			{"selected_prefix_bridge.feature_write_weights", (uint64_t)Q38_LAYERS * Q38_HIDDEN * Q38_RANK, &writes},
			{"selected_prefix_bridge.gate.weight", (uint64_t)Q38_RANK * Q38_CODE, &state->bridge_gate},
			{"selected_prefix_bridge.gate.bias", Q38_RANK, &state->gate_bias},
			{"selected_prefix_bridge.predecessor_gate.weight", (uint64_t)Q38_RANK * Q38_CODE, &state->bridge_sender},
			{"selected_prefix_bridge.prev.weight", (uint64_t)Q38_RANK * Q38_CODE, &state->bridge_value},
		};
		status = DsparkLoadBf16Table(state->pack, &document, payload_base, "qwen38 dspine", plain, (uint32_t)(sizeof(plain) / sizeof(plain[0])));
	}
	for ( uint32_t layer = 0u; layer < Q38_LAYERS && status == SPARK_STATUS_OK; layer++ )
	{
		Q38Layer *w = &state->layers[layer];
		const DsparkBf16Spec tensors[] = {
			{"input_layernorm.weight", Q38_HIDDEN, &w->input_norm}, {"post_attention_layernorm.weight", Q38_HIDDEN, &w->post_norm},
			{"self_attn.q_proj.weight", (uint64_t)Q38_Q_WIDTH * Q38_HIDDEN, &w->q}, {"self_attn.k_proj.weight", (uint64_t)Q38_KV_WIDTH * Q38_HIDDEN, &w->k},
			{"self_attn.v_proj.weight", (uint64_t)Q38_KV_WIDTH * Q38_HIDDEN, &w->v}, {"self_attn.o_proj.weight", (uint64_t)Q38_HIDDEN * Q38_Q_WIDTH, &w->o},
			{"self_attn.q_norm.weight", Q38_HEAD_DIM, &w->q_norm}, {"self_attn.k_norm.weight", Q38_HEAD_DIM, &w->k_norm},
			{"mlp.gate_proj.weight", (uint64_t)Q38_MLP * Q38_HIDDEN, &w->gate}, {"mlp.up_proj.weight", (uint64_t)Q38_MLP * Q38_HIDDEN, &w->up},
			{"mlp.down_proj.weight", (uint64_t)Q38_HIDDEN * Q38_MLP, &w->down},
		};
		status = DsparkLoadPrefixedBf16Table(state->pack, &document, payload_base, "qwen38 dspine", "layers.%u.%s", layer, tensors,
			(uint32_t)(sizeof(tensors) / sizeof(tensors[0])));
		if ( status == SPARK_STATUS_OK )
		{
			w->read = layer < Q38_EARLY ? early + (uint64_t)layer * Q38_CODE * Q38_HIDDEN : last_read;
			w->write = writes + (uint64_t)layer * Q38_HIDDEN * Q38_RANK;
		}
	}
	SparkJsonDocumentDestroy(&document);
	return status;
}

static SparkStatus Q38Gemm(Q38State *state, const uint16_t *weight, const uint16_t *input, uint16_t *output, uint32_t rows, uint32_t input_dimension,
	uint32_t output_dimension)
{
	if ( rows == Q38_BLOCK && output_dimension >= Q38_STREAM_OUTPUTS )
		return LmStreamGemmDense<LmBf16Format>(weight, LmScaleTensorNone(), input, output, (float *)0, rows, input_dimension, output_dimension, 0u, 0u,
			state->multiprocessors, state->stream) == LM_LAUNCH_OK ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
	return DsparkGemm(weight, input, output, rows, input_dimension, output_dimension, state->multiprocessors, state->stream);
}

static SparkStatus Q38Tables(Q38State *state)
{
	const uint32_t chunk = 8192u;
	uint16_t *scaled = 0;
	SparkStatus status = SPARK_STATUS_OK;
	if ( !DSPARK_DEVICE(state->packed_weight, (uint64_t)Q38_PACKED * Q38_CODE) || !DSPARK_DEVICE(state->write_transposed, (uint64_t)Q38_RANK * Q38_HIDDEN) ||
		!DSPARK_DEVICE(state->tg, (uint64_t)Q38_VOCAB * Q38_RANK) || !DSPARK_DEVICE(state->tsv, (uint64_t)Q38_VOCAB * 2u * Q38_RANK) ||
		!DSPARK_DEVICE(scaled, (uint64_t)chunk * Q38_HIDDEN) )
	{
		(void)SparkGpuFree(scaled);
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	if ( SparkGpuMemcpyAsync(state->packed_weight, state->bridge_gate, (size_t)Q38_RANK * Q38_CODE * sizeof(uint16_t), SPARK_GPU_COPY_DEVICE_TO_DEVICE,
		state->stream) != SPARK_GPU_SUCCESS ||
		SparkGpuMemcpyAsync(state->packed_weight + (uint64_t)Q38_RANK * Q38_CODE, state->bridge_sender, (size_t)Q38_RANK * Q38_CODE * sizeof(uint16_t),
		SPARK_GPU_COPY_DEVICE_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS ||
		SparkGpuMemcpyAsync(state->packed_weight + 2u * (uint64_t)Q38_RANK * Q38_CODE, state->bridge_value, (size_t)Q38_RANK * Q38_CODE * sizeof(uint16_t),
		SPARK_GPU_COPY_DEVICE_TO_DEVICE, state->stream) != SPARK_GPU_SUCCESS )
		status = SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK )
		LM_LAUNCH((Q38TransposeKernel), 4u * state->multiprocessors, DSPARK_THREADS, 0, state->stream, state->layers[Q38_LAYERS - 1u].write, Q38_HIDDEN,
			Q38_RANK, state->write_transposed);
	for ( uint32_t first = 0u; first < Q38_VOCAB && status == SPARK_STATUS_OK; first += chunk )
	{
		const uint32_t rows = Q38_VOCAB - first < chunk ? Q38_VOCAB - first : chunk;
		LM_LAUNCH((Q38ScaledRowsKernel), 4u * state->multiprocessors, DSPARK_THREADS, 0, state->stream, state->head + (uint64_t)first * Q38_HIDDEN,
			state->norm, 1.0f, Q38_HIDDEN, (uint64_t)rows * Q38_HIDDEN, scaled);
		status = Q38Gemm(state, state->write_transposed, scaled, state->tg + (uint64_t)first * Q38_RANK, rows, Q38_HIDDEN, Q38_RANK);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((Q38ScaledRowsKernel), 4u * state->multiprocessors, DSPARK_THREADS, 0, state->stream, state->codebook + (uint64_t)first * Q38_CODE,
			(const uint16_t *)0, sqrtf((float)Q38_CODE), Q38_CODE, (uint64_t)rows * Q38_CODE, scaled);
		status = Q38Gemm(state, state->packed_weight + (uint64_t)Q38_RANK * Q38_CODE, scaled, state->tsv + (uint64_t)first * 2u * Q38_RANK, rows,
			Q38_CODE, 2u * Q38_RANK);
	}
	status = DsparkSettle(state->stream, status);
	(void)SparkGpuFree(scaled);
	return status;
}

static SparkStatus Q38Scratch(Q38State *state, uint64_t context_bytes)
{
	Q38Forward *f = &state->forward;
	int multiprocessors = 0;
	if ( !state->ring.Init(context_bytes, Q38_SEQUENCE_ELEMENTS, &state->caches) ||
		SparkGpuDeviceGetAttribute(&multiprocessors, SPARK_GPU_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0) != SPARK_GPU_SUCCESS ||
		!DsparkAllocHost(&state->host_taps, (size_t)Q38_INGEST_ROWS * Q38_TAP_WIDTH) ||
		!DsparkAllocHost(&state->host_positions, Q38_INGEST_ROWS) || !DsparkAllocHost(&f->host_draft, 1u) ||
		!DsparkAllocHost(&f->host_picks, 1u) ||
		!DSPARK_DEVICE(state->taps, (uint64_t)Q38_INGEST_ROWS * Q38_TAP_WIDTH) || !DSPARK_DEVICE(state->positions, Q38_INGEST_ROWS) ||
		!DSPARK_DEVICE(state->features_raw, (uint64_t)Q38_INGEST_ROWS * Q38_HIDDEN) ||
		!DSPARK_DEVICE(state->features, (uint64_t)Q38_INGEST_ROWS * Q38_HIDDEN) ||
		!DSPARK_DEVICE(state->keys_raw, (uint64_t)Q38_INGEST_ROWS * Q38_KV_WIDTH) ||
		!DSPARK_DEVICE(state->values_raw, (uint64_t)Q38_INGEST_ROWS * Q38_KV_WIDTH) || !DSPARK_DEVICE(f->draft, 1u) || !DSPARK_DEVICE(f->picks, 1u) ||
		!DSPARK_DEVICE(f->hidden_f32, Q38_BLOCK * Q38_HIDDEN) || !DSPARK_DEVICE(f->hidden, Q38_BLOCK * Q38_HIDDEN) ||
		!DSPARK_DEVICE(f->x, Q38_BLOCK * Q38_HIDDEN) || !DSPARK_DEVICE(f->q, Q38_BLOCK * Q38_Q_WIDTH) ||
		!DSPARK_DEVICE(f->q_roped, Q38_BLOCK * Q38_Q_WIDTH) || !DSPARK_DEVICE(f->k, Q38_BLOCK * Q38_KV_WIDTH) ||
		!DSPARK_DEVICE(f->v, Q38_BLOCK * Q38_KV_WIDTH) ||
		!DSPARK_DEVICE(f->partials, (uint64_t)Q38_KV_HEADS * Q38_SPLITS * Q38_QUERIES * (Q38_HEAD_DIM + 2u)) ||
		!DSPARK_DEVICE(f->attended, Q38_BLOCK * Q38_Q_WIDTH) || !DSPARK_DEVICE(f->projected, Q38_BLOCK * Q38_HIDDEN) ||
		!DSPARK_DEVICE(f->residual, Q38_BLOCK * Q38_HIDDEN) || !DSPARK_DEVICE(f->gate, Q38_BLOCK * Q38_MLP) ||
		!DSPARK_DEVICE(f->up, Q38_BLOCK * Q38_MLP) || !DSPARK_DEVICE(f->act, Q38_BLOCK * Q38_MLP) ||
		!DSPARK_DEVICE(f->mlp, Q38_BLOCK * Q38_HIDDEN) || !DSPARK_DEVICE(f->raw, Q38_BLOCK * Q38_HIDDEN) ||
		!DSPARK_DEVICE(f->rms, Q38_BLOCK) || !DSPARK_DEVICE(f->bridge_normed, Q38_BLOCK * Q38_HIDDEN) ||
		!DSPARK_DEVICE(f->codes_projected, Q38_BLOCK * Q38_CODE) || !DSPARK_DEVICE(f->codes, Q38_BLOCK * Q38_CODE) ||
		!DSPARK_DEVICE(f->packed, Q38_BLOCK * Q38_PACKED) || !DSPARK_DEVICE(f->message, Q38_BLOCK * Q38_RANK) ||
		!DSPARK_DEVICE(f->delta, Q38_BLOCK * Q38_HIDDEN) || !DSPARK_DEVICE(f->normed, Q38_BLOCK * Q38_HIDDEN) ||
		!DSPARK_DEVICE(f->y, Q38_BLOCK * Q38_HIDDEN) || !DSPARK_DEVICE(f->base, (uint64_t)Q38_BLOCK * Q38_VOCAB) ||
		SparkDrafterTopkBf16Alloc(&f->topk_ws, Q38_DRAFTS) != SPARK_STATUS_OK || !DSPARK_DEVICE(f->pooled, Q38_DRAFTS * Q38_POOL) ||
		!DSPARK_DEVICE(f->candidates, Q38_DRAFTS * Q38_LANES) || !DSPARK_DEVICE(f->table, Q38_DRAFTS * Q38_LANES) ||
		!DSPARK_DEVICE(f->alt_table, Q38_DRAFTS * Q38_LANES) || !DSPARK_DEVICE(f->confidence, Q38_DRAFTS * Q38_LANES) ||
		!DSPARK_DEVICE(f->alt_confidence, Q38_DRAFTS * Q38_LANES) || !DSPARK_DEVICE(f->refined_table, Q38_DRAFTS * Q38_LANES * Q38_LANES) )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	state->multiprocessors = (uint32_t)multiprocessors;
	return SPARK_STATUS_OK;
}

static uint16_t *Q38Keys(const Q38Sequence *sequence, uint32_t layer)
{
	return sequence->cache + (uint64_t)(2u * layer) * Q38_LAYER_KV;
}

static uint16_t *Q38Values(const Q38Sequence *sequence, uint32_t layer)
{
	return sequence->cache + (uint64_t)(2u * layer + 1u) * Q38_LAYER_KV;
}

static SparkStatus Q38IngestRows(Q38State *state, Q38Sequence *sequence, uint32_t rows)
{
	SparkStatus status = Q38Gemm(state, state->fc, state->taps, state->features_raw, rows, Q38_TAP_WIDTH, Q38_HIDDEN);
	if ( status == SPARK_STATUS_OK )
		LM_LAUNCH((DsparkRmsKernel<false>), rows, DSPARK_THREADS, 0, state->stream, state->features_raw, state->hidden_norm, state->features,
			Q38_HIDDEN, state->epsilon);
	for ( uint32_t layer = 0u; layer < Q38_LAYERS && status == SPARK_STATUS_OK; layer++ )
	{
		status = Q38Gemm(state, state->layers[layer].k, state->features, state->keys_raw, rows, Q38_HIDDEN, Q38_KV_WIDTH);
		if ( status == SPARK_STATUS_OK )
			status = Q38Gemm(state, state->layers[layer].v, state->features, state->values_raw, rows, Q38_HIDDEN, Q38_KV_WIDTH);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((Q38IngestKeysKernel), rows, DSPARK_THREADS, 0, state->stream, state->keys_raw, state->layers[layer].k_norm, state->rope_inverse,
			state->positions, state->epsilon, Q38Keys(sequence, layer));
		LM_LAUNCH((DsparkStoreValuesRingKernel<Q38_KV_WIDTH, Q38_HEAD_DIM, Q38_WINDOW, Q38_COLUMNS, Q38Draft>), rows, DSPARK_THREADS, 0, state->stream,
			state->values_raw, state->positions, (const Q38Draft *)0, 0u, Q38Values(sequence, layer));
	}
	if ( status == SPARK_STATUS_OK && SparkGpuGetLastError() != SPARK_GPU_SUCCESS )
		status = SPARK_STATUS_INTERNAL_ERROR;
	return DsparkSettle(state->stream, status);
}

static SparkStatus Q38Ingest(void *opaque, const SparkTapStreamReaderFrame *frame)
{
	Q38State *state = (Q38State *)opaque;
	return state->ring.IngestFrame<Q38_TAPS, Q38_HIDDEN, Q38_INGEST_ROWS>(frame, state->host_taps, state->host_positions, state->taps,
		state->positions, state->stream, &state->frames, &state->rows,
		[&](Q38Sequence *sequence, uint32_t part) { return Q38IngestRows(state, sequence, part); });
}

static SparkStatus Q38Prepare(Q38State *state, Q38Sequence *sequence, uint32_t anchor, uint64_t position, uint64_t checked,
	const SparkRowSampling *sampling, uint32_t tree, uint32_t *splits_out)
{
	Q38Draft *draft = state->forward.host_draft;
	uint32_t splits;
	if ( anchor >= Q38_VOCAB || position + Q38_BLOCK > UINT32_MAX )
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(draft, 0, offsetof(Q38Draft, valid));
	draft->cache = sequence->cache;
	for ( uint32_t row = 0u; row < Q38_BLOCK; row++ )
		draft->positions[row] = (uint32_t)(position + row);
	draft->anchor = anchor;
	draft->mask = state->mask_token;
	draft->greedy = sampling->inverse_temperature > 0.0f ? 0u : 1u;
	draft->seed = sampling->seed;
	draft->inverse_temperature = sampling->inverse_temperature;
	draft->nucleus = sampling->nucleus < 1.0f ? sampling->nucleus : 0.0f;
	draft->top_k = sampling->top_k;
	draft->tree = tree;
	Q38RingPool::Span<Q38_BLOCK, Q38_SLIDING, false>(sequence, checked, position, draft);
	splits = (draft->block_at - draft->start + Q38_SPLIT_KEYS - 1u) / Q38_SPLIT_KEYS;
	draft->splits = splits;
	if ( splits_out != 0 )
		*splits_out = splits;
	return SparkGpuMemcpyAsync(state->forward.draft, draft, sizeof(*draft), SPARK_GPU_COPY_HOST_TO_DEVICE, state->stream) == SPARK_GPU_SUCCESS ?
		SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
}

static SparkStatus Q38Bridge(Q38State *state, uint32_t layer, uint16_t *output)
{
	Q38Forward *f = &state->forward;
	const SparkGpuStream stream = state->stream;
	SparkStatus status;
	LM_LAUNCH((Q38ResidualBridgeScaleKernel), Q38_BLOCK, DSPARK_THREADS, 0, stream, f->residual, f->mlp, f->raw, f->rms, f->bridge_normed,
		state->bridge_epsilon);
	status = Q38Gemm(state, state->layers[layer].read, f->bridge_normed, f->codes_projected, Q38_BLOCK, Q38_HIDDEN, Q38_CODE);
	if ( status != SPARK_STATUS_OK )
		return status;
	LM_LAUNCH((Q38BridgeCodesKernel), Q38_BLOCK, DSPARK_THREADS, 0, stream, f->codes_projected, state->codebook, f->draft, state->bridge_epsilon,
		f->codes);
	status = Q38Gemm(state, state->packed_weight, f->codes, f->packed, Q38_BLOCK, Q38_CODE, Q38_PACKED);
	if ( status != SPARK_STATUS_OK )
		return status;
	LM_LAUNCH((Q38BridgeMessageKernel), Q38_BLOCK, DSPARK_THREADS, 0, stream, f->packed, state->gate_bias, f->message);
	status = Q38Gemm(state, state->layers[layer].write, f->message, f->delta, Q38_BLOCK, Q38_RANK, Q38_HIDDEN);
	if ( status != SPARK_STATUS_OK )
		return status;
	LM_LAUNCH((Q38BridgeOutKernel), Q38_BLOCK, DSPARK_THREADS, 0, stream, f->raw, f->rms, f->delta, output);
	return SPARK_STATUS_OK;
}

static SparkStatus Q38RunForward(Q38State *state, uint32_t layers, uint32_t splits)
{
	Q38Forward *f = &state->forward;
	const SparkGpuStream stream = state->stream;
	const uint32_t *positions = (const uint32_t *)((const uint8_t *)f->draft + offsetof(Q38Draft, positions));
	SparkStatus status = SPARK_STATUS_OK;
	LM_LAUNCH((DsparkEmbedF32Kernel<Q38_HIDDEN, Q38Draft>), Q38_BLOCK, DSPARK_THREADS, 0, stream, state->embed, f->draft, f->hidden_f32);
	for ( uint32_t layer = 0u; layer < layers && status == SPARK_STATUS_OK; layer++ )
	{
		const Q38Layer *w = &state->layers[layer];
		const uint64_t keys_offset = (uint64_t)(2u * layer) * Q38_LAYER_KV;
		const uint64_t values_offset = (uint64_t)(2u * layer + 1u) * Q38_LAYER_KV;
		if ( layer == 0u )
			LM_LAUNCH((DsparkRmsKernel<false, false, float>), Q38_BLOCK, DSPARK_THREADS, 0, stream, f->hidden_f32, w->input_norm, f->x, Q38_HIDDEN,
				state->epsilon);
		else
			LM_LAUNCH((DsparkRmsKernel<true>), Q38_BLOCK, DSPARK_THREADS, 0, stream, f->hidden, w->input_norm, f->x, Q38_HIDDEN, state->epsilon);
		status = Q38Gemm(state, w->q, f->x, f->q, Q38_BLOCK, Q38_HIDDEN, Q38_Q_WIDTH);
		if ( status == SPARK_STATUS_OK )
			status = Q38Gemm(state, w->k, f->x, f->k, Q38_BLOCK, Q38_HIDDEN, Q38_KV_WIDTH);
		if ( status == SPARK_STATUS_OK )
			status = Q38Gemm(state, w->v, f->x, f->v, Q38_BLOCK, Q38_HIDDEN, Q38_KV_WIDTH);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((DsparkHeadNormRopeRingKernel<Q38_HEAD_DIM, Q38_HEAD_DIM, Q38_WINDOW, Q38_COLUMNS, DSPARK_NORM_BF16_OUT, false, Q38Draft>),
			Q38_BLOCK * Q38_HEADS, Q38_HEAD_DIM, 0, stream, f->q, w->q_norm, state->rope_inverse, 1.0f, positions, Q38_HEADS, state->epsilon,
			f->q_roped, (uint16_t *)0, f->draft, 0u, 0u);
		if ( LM_LAUNCH_DEPENDENT((DsparkHeadNormRopeRingKernel<Q38_HEAD_DIM, Q38_HEAD_DIM, Q38_WINDOW, Q38_COLUMNS, DSPARK_NORM_BF16_OUT, false,
					Q38Draft>),
				Q38_BLOCK * Q38_KV_HEADS, Q38_HEAD_DIM, 0, stream, f->k, w->k_norm, state->rope_inverse, 1.0f, positions, Q38_KV_HEADS,
				state->epsilon, (uint16_t *)0, (uint16_t *)0, f->draft, keys_offset, 0u) != SPARK_GPU_SUCCESS ||
			LM_LAUNCH_DEPENDENT((DsparkStoreValuesRingKernel<Q38_KV_WIDTH, Q38_HEAD_DIM, Q38_WINDOW, Q38_COLUMNS, Q38Draft>), Q38_BLOCK,
				DSPARK_THREADS, 0, stream, f->v, (const uint32_t *)0, f->draft, values_offset, (uint16_t *)0) != SPARK_GPU_SUCCESS )
			return SPARK_STATUS_INTERNAL_ERROR;
		if ( splits != 0u &&
			LM_LAUNCH_DEPENDENT((Q38AttentionSplitKernel), dim3(Q38_KV_HEADS, splits), Q38_ATTENTION_THREADS, 0, stream, f->q_roped, keys_offset,
				values_offset, f->draft, state->softmax_scale, f->partials) != SPARK_GPU_SUCCESS )
			return SPARK_STATUS_INTERNAL_ERROR;
		if ( LM_LAUNCH_DEPENDENT((DsparkAttentionMergeKernel<Q38_HEADS, Q38_KV_HEADS, Q38_BLOCK, Q38_COLUMNS, Q38_HEAD_DIM, Q38_SPLITS, true, false,
					true, Q38Draft>),
				Q38_BLOCK * Q38_HEADS, Q38_HEAD_DIM, 0, stream, f->q_roped, (const uint16_t *)0, (const uint16_t *)0, keys_offset, values_offset,
				f->draft, state->softmax_scale, f->partials, f->attended) != SPARK_GPU_SUCCESS )
			return SPARK_STATUS_INTERNAL_ERROR;
		status = Q38Gemm(state, w->o, f->attended, f->projected, Q38_BLOCK, Q38_Q_WIDTH, Q38_HIDDEN);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((Q38ResidualRmsKernel), Q38_BLOCK, DSPARK_THREADS, 0, stream, layer == 0u ? f->hidden_f32 : (const float *)0, f->hidden,
			f->projected, w->post_norm, f->residual, f->x, state->epsilon);
		status = Q38Gemm(state, w->gate, f->x, f->gate, Q38_BLOCK, Q38_HIDDEN, Q38_MLP);
		if ( status == SPARK_STATUS_OK )
			status = Q38Gemm(state, w->up, f->x, f->up, Q38_BLOCK, Q38_HIDDEN, Q38_MLP);
		if ( status != SPARK_STATUS_OK )
			break;
		LM_LAUNCH((DsparkSwigluKernel), 2u * state->multiprocessors, DSPARK_THREADS, 0, stream, f->gate, f->up, f->act, Q38_BLOCK * Q38_MLP);
		status = Q38Gemm(state, w->down, f->act, f->mlp, Q38_BLOCK, Q38_MLP, Q38_HIDDEN);
		if ( status != SPARK_STATUS_OK )
			break;
		status = Q38Bridge(state, layer, f->hidden);
	}
	if ( status != SPARK_STATUS_OK || layers < Q38_LAYERS )
		return status;
	if ( LM_LAUNCH_DEPENDENT((Q38DualRmsKernel), Q38_BLOCK, DSPARK_THREADS, 0, stream, f->hidden, state->norm, f->normed, f->y, state->epsilon) !=
		SPARK_GPU_SUCCESS )
		return SPARK_STATUS_INTERNAL_ERROR;
	status = Q38Gemm(state, state->head, f->y, f->base, Q38_BLOCK, Q38_HIDDEN, Q38_VOCAB);
	if ( status != SPARK_STATUS_OK )
		return status;
	return SparkGpuGetLastError() == SPARK_GPU_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus Q38Heads(Q38State *state, uint32_t greedy)
{
	Q38Forward *f = &state->forward;
	const SparkGpuStream stream = state->stream;
	SparkStatus status = SparkDrafterTopkBf16Launch(f->base + Q38_VOCAB, Q38_DRAFTS, Q38_VOCAB, Q38_POOL, &f->topk_ws, f->pooled, stream);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( LM_LAUNCH_DEPENDENT((Q38PoolKernel), Q38_DRAFTS, DSPARK_THREADS, 0, stream, f->pooled, f->normed, state->head, f->candidates) !=
			SPARK_GPU_SUCCESS ||
		LM_LAUNCH_DEPENDENT((Q38GreedyTableKernel), Q38_DRAFTS, DSPARK_THREADS, 0, stream, f->draft, f->candidates, f->raw, f->rms, f->packed,
			state->norm, state->head, state->tg, state->tsv, state->gate_bias, f->table, f->alt_table, f->confidence, f->alt_confidence) !=
			SPARK_GPU_SUCCESS )
		return SPARK_STATUS_INTERNAL_ERROR;
	if ( greedy == 0u &&
		LM_LAUNCH_DEPENDENT((Q38RefinedKernel), Q38_DRAFTS * Q38_LANES, DSPARK_THREADS, 0, stream, f->draft, f->candidates, f->raw, f->rms, f->packed,
			state->codebook, state->bridge_sender, state->bridge_value, state->gate_bias, state->layers[Q38_LAYERS - 1u].write, state->norm,
			state->head, state->epsilon, f->refined_table) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_INTERNAL_ERROR;
	if ( LM_LAUNCH_DEPENDENT((Q38ChainKernel), 1u, DSPARK_WARP, 0, stream, f->draft, f->candidates, f->table, f->alt_table, f->confidence,
			f->alt_confidence, f->refined_table, f->picks) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_INTERNAL_ERROR;
	if ( SparkGpuMemcpyAsync(f->host_picks, f->picks, sizeof(Q38Picks), SPARK_GPU_COPY_DEVICE_TO_HOST, stream) != SPARK_GPU_SUCCESS )
		return SPARK_STATUS_IO_ERROR;
	return SparkGpuGetLastError() == SPARK_GPU_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus Q38Close(void *opaque)
{
	Q38State *state = (Q38State *)opaque;
	SparkStatus status = SPARK_STATUS_OK;
	if ( state == 0 )
		return SPARK_STATUS_OK;
	if ( state->stream != 0 )
		(void)SparkGpuStreamSynchronize(state->stream);
	SparkDrafterGraphCacheDestroy(&state->graphs);
	{
		Q38Forward *f = &state->forward;
		void *buffers[] = {
			f->draft, f->picks, f->hidden_f32, f->hidden, f->x, f->q, f->q_roped, f->k, f->v, f->partials, f->attended, f->projected, f->residual,
			f->gate, f->up, f->act, f->mlp, f->raw, f->rms, f->bridge_normed, f->codes_projected, f->codes, f->packed, f->message, f->delta,
			f->normed, f->y, f->base, f->pooled, f->candidates, f->table, f->alt_table, f->confidence, f->alt_confidence, f->refined_table,
			state->caches, state->taps, state->positions, state->features_raw, state->features, state->keys_raw, state->values_raw,
			state->packed_weight, state->write_transposed, state->tg, state->tsv, state->rope_inverse,
		};
		void *host_buffers[] = { f->host_draft, f->host_picks, state->host_taps, state->host_positions };
		DsparkFreeDeviceBuffers(buffers, (uint32_t)(sizeof(buffers) / sizeof(buffers[0])));
		SparkDrafterTopkBf16Free(&f->topk_ws);
		DsparkFreeHostBuffers(host_buffers, (uint32_t)(sizeof(host_buffers) / sizeof(host_buffers[0])));
	}
	if ( state->stream != 0 )
		(void)SparkGpuStreamDestroy(state->stream);
	if ( state->pack != 0 )
		status = SparkWeightdLazyPackDestroy(state->pack);
	free(state->ring.sequences);
	free(state);
	return status;
}

static SparkStatus Q38Open(const SparkDrafterModuleLane *lane, void **state_out, SparkDrafterModuleBind *bind)
{
	return DsparkDrafterOpen<Q38State>(lane, Q38_SEQUENCE_BYTES, "lane %s has a context cap below one sequence's window",
		"lane %s: the pack %s is not a Qwen3.8-27B DSpine pack this module reads", true, state_out, bind, Q38Close,
		[&](Q38State *state) -> SparkStatus {
			SparkStatus status = Q38Weights(state, lane->pack_path);
			if ( status == SPARK_STATUS_OK )
				status = Q38Scratch(state, lane->context_bytes);
			if ( status == SPARK_STATUS_OK )
				status = Q38Tables(state);
			return status;
		},
		[](const Q38State *state) { return state->ring.capacity; });
}

static SparkStatus Q38Propose(void *opaque, const SparkDraftServerRequest *request, SparkDraftBridgeNode *nodes, uint32_t capacity, uint32_t *count)
{
	Q38State *state = (Q38State *)opaque;
	const uint32_t source = request->speculator_mask & (~request->speculator_mask + 1u);
	const uint64_t first = request->position > Q38_WINDOW ? request->position - Q38_WINDOW : 0u;
	const uint32_t requested = request->depth < request->max_depth ? request->depth : request->max_depth;
	const uint32_t limit = request->node_capacity != 0u && request->node_capacity < capacity ? request->node_capacity : capacity;
	uint32_t drafts = requested, splits = 0u;
	Q38Sequence *sequence = 0;
	uint64_t checked = first;
	SparkStatus status;
	*count = 0u;
	if ( limit == 0u )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	status = state->ring.PrepareProposeContext(request, first, state->stream, &sequence, &checked,
		[&](Q38Sequence *dst, const Q38Sequence *src, uint32_t slot, uint32_t n) {
			return Q38RingPool::CopyKvRange<Q38_LAYERS, Q38_KV_HEADS, Q38_COLUMNS, Q38_HEAD_DIM>(dst, src, slot, n, state->stream);
		});
	if ( status != SPARK_STATUS_OK )
		return status;
	drafts = drafts < Q38_DRAFTS ? drafts : Q38_DRAFTS;
	drafts = drafts < limit - 1u ? drafts : limit - 1u;
	{
		const uint32_t tree = limit > (requested > drafts ? requested : drafts) + 1u ? 1u : 0u;
		const uint32_t greedy = request->sampling.inverse_temperature > 0.0f ? 0u : 1u;
		status = Q38Prepare(state, sequence, request->anchor_token_id, request->position, checked, &request->sampling, tree, &splits);
		if ( status == SPARK_STATUS_OK )
		{
			const uint64_t key = ((uint64_t)splits << 1u) | (uint64_t)greedy;
			status = SparkDrafterGraphRun(&state->graphs, state->stream, key, [&]() -> SparkStatus {
				SparkStatus inner = Q38RunForward(state, Q38_LAYERS, splits);
				if ( inner == SPARK_STATUS_OK )
					inner = Q38Heads(state, greedy);
				return inner;
			});
		}
		status = DsparkSettle(state->stream, status);
		if ( status != SPARK_STATUS_OK )
			return status;
		*count = DsparkPackGridTree<Q38_DRAFTS, Q38_BRANCH_LEN>(request->anchor_token_id, source, state->forward.host_picks, tree, drafts, requested,
			limit, nodes);
	}
	return SPARK_DRAFTER_MODULE_PROPOSE_DRAFTS;
}

static const SparkDrafterModuleInterface q38_interface = {
	SPARK_DRAFTER_MODULE_ABI_VERSION,
	(uint32_t)sizeof(SparkDrafterModuleInterface),
	"qwen38-dspine",
	Q38Open,
	Q38Ingest,
	Q38Propose,
	Q38Close,
};

extern "C" __attribute__((visibility("default"))) const SparkDrafterModuleInterface *spark_drafter_module_interface(void)
{
	return &q38_interface;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_qwen38_dspine_cache(void *opaque, uint64_t sequence_id, uint32_t layer,
	uint16_t *keys, uint16_t *values, int64_t *positions)
{
	Q38State *state = (Q38State *)opaque;
	if ( state == 0 )
		return SPARK_STATUS_NOT_FOUND;
	return state->ring.CacheReadback<Q38_LAYERS, Q38_LAYER_KV>(sequence_id, layer, keys, values, positions);
}

static SparkStatus Q38RunDiag(Q38State *state, uint64_t sequence_id, uint32_t anchor, uint64_t position, uint32_t layers, uint32_t run_heads,
	SparkStatus missing)
{
	Q38Sequence *sequence = state != 0 ? state->ring.Live(sequence_id) : 0;
	const uint64_t first = position > Q38_WINDOW ? position - Q38_WINDOW : 0u;
	uint32_t splits = 0u;
	SparkRowSampling greedy;
	SparkStatus status;
	if ( sequence == 0 )
		return missing;
	memset(&greedy, 0, sizeof(greedy));
	status = Q38Prepare(state, sequence, anchor, position, Q38RingPool::CheckedFrom(sequence, first, position), &greedy, 0u, &splits);
	if ( status == SPARK_STATUS_OK )
		status = Q38RunForward(state, layers, splits);
	if ( status == SPARK_STATUS_OK && run_heads != 0u )
		status = Q38Heads(state, 1u);
	return DsparkSettle(state->stream, status);
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_qwen38_dspine_forward(void *opaque, uint64_t sequence_id, uint32_t anchor,
	uint64_t position, uint16_t *y, uint16_t *normed, uint32_t *candidates, uint64_t *elapsed_ns)
{
	Q38State *state = (Q38State *)opaque;
	const uint64_t started = SparkMonotonicNs();
	SparkStatus status = Q38RunDiag(state, sequence_id, anchor, position, Q38_LAYERS, 1u, SPARK_STATUS_NOT_FOUND);
	if ( elapsed_ns != 0 )
		*elapsed_ns = SparkMonotonicNs() - started;
	if ( status == SPARK_STATUS_OK && y != 0 )
		status = DsparkReadback(y, state->forward.y, (size_t)Q38_BLOCK * Q38_HIDDEN * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK && normed != 0 )
		status = DsparkReadback(normed, state->forward.normed, (size_t)Q38_BLOCK * Q38_HIDDEN * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK && candidates != 0 )
		status = DsparkReadback(candidates, state->forward.candidates, (size_t)Q38_DRAFTS * Q38_LANES * sizeof(uint32_t));
	return status;
}

extern "C" __attribute__((visibility("default"))) SparkStatus spark_qwen38_dspine_trace(void *opaque, uint64_t sequence_id, uint32_t anchor,
	uint64_t position, uint32_t layers, uint16_t *hidden)
{
	Q38State *state = (Q38State *)opaque;
	SparkStatus status;
	if ( state == 0 || hidden == 0 || layers == 0u || layers > Q38_LAYERS )
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = Q38RunDiag(state, sequence_id, anchor, position, layers, 0u, SPARK_STATUS_INVALID_ARGUMENT);
	if ( status == SPARK_STATUS_OK )
		status = DsparkReadback(hidden, state->forward.hidden, (size_t)Q38_BLOCK * Q38_HIDDEN * sizeof(uint16_t));
	return status;
}
