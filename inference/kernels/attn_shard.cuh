#pragma once

#include <stdint.h>
#include <math.h>

#include "inference/kernels/kv_shard.cuh"

#define LM_LATENT_SHARD_WARPS 8u
#define LM_LATENT_SHARD_THREADS (LM_LATENT_SHARD_WARPS * 32u)
#define LM_LATENT_SHARD_UNROLL 2u
#define LM_LATENT_SHARD_LIST_CAPACITY 4096u
#define LM_LATENT_SHARD_RECORD_FLOATS(latent) ((latent) + 2u)

template<uint32_t PER_LANE>
static __device__ __forceinline__ void LmLatentShardLoad(const uint16_t *__restrict__ source, float *values)
{
    uint32_t chunk, word, bits[4];
    uint4 packed;
    if (PER_LANE % 8u == 0u)
    {
        for (chunk = 0u; chunk < PER_LANE / 8u; ++chunk)
        {
            packed = ((const uint4 *)source)[chunk];
            bits[0] = packed.x; bits[1] = packed.y; bits[2] = packed.z; bits[3] = packed.w;
            for (word = 0u; word < 4u; ++word)
            {
                values[chunk * 8u + word * 2u] = __uint_as_float(bits[word] << 16u);
                values[chunk * 8u + word * 2u + 1u] = __uint_as_float(bits[word] & 0xffff0000u);
            }
        }
        return;
    }
    for (word = 0u; word < PER_LANE / 2u; ++word)
    {
        bits[0] = ((const uint32_t *)source)[word];
        values[word * 2u] = __uint_as_float(bits[0] << 16u);
        values[word * 2u + 1u] = __uint_as_float(bits[0] & 0xffff0000u);
    }
}

template<uint32_t HEADS, uint32_t PER_LANE>
static __device__ __forceinline__ void LmLatentShardStep(const float (*query)[PER_LANE], const float *value, float qk_scale, float *running_max, float *running_sum, float (*accumulator)[PER_LANE])
{
    uint32_t head, element, mask;
    float score, previous, scaled_previous, scaled_current;
    for (head = 0u; head < HEADS; ++head)
    {
        score = 0.0f;
        for (element = 0u; element < PER_LANE; ++element)
            score += query[head][element] * value[element];
        for (mask = 16u; mask > 0u; mask >>= 1u)
            score += __shfl_xor_sync(0xffffffffu, score, mask);
        score *= qk_scale;
        previous = running_max[head];
        running_max[head] = fmaxf(previous, score);
        scaled_previous = __expf(previous - running_max[head]);
        scaled_current = __expf(score - running_max[head]);
        running_sum[head] = (running_sum[head] * scaled_previous) + scaled_current;
        for (element = 0u; element < PER_LANE; ++element)
            accumulator[head][element] = (accumulator[head][element] * scaled_previous) + (scaled_current * value[element]);
    }
}

static __device__ __forceinline__ uint32_t LmLatentShardCompact(
    SparkKvShard shard,
    const uint32_t *__restrict__ selected,
    uint32_t selected_count,
    uint32_t bound,
    uint32_t bounded,
    uint32_t *list,
    uint32_t *scan)
{
    uint32_t base, offset, add, count = 0u, keep, position, index;
    for (base = 0u; base < selected_count; base += LM_LATENT_SHARD_THREADS)
    {
        index = base + threadIdx.x;
        position = index < selected_count ? selected[index] : 0xffffffffu;
        keep = index < selected_count && (bounded == 0u || position <= bound) && SparkKvShardOwns(shard, position) != 0u ? 1u : 0u;
        scan[threadIdx.x] = keep;
        __syncthreads();
        for (offset = 1u; offset < LM_LATENT_SHARD_THREADS; offset <<= 1u)
        {
            add = threadIdx.x >= offset ? scan[threadIdx.x - offset] : 0u;
            __syncthreads();
            scan[threadIdx.x] += add;
            __syncthreads();
        }
        if (keep != 0u)
            list[count + scan[threadIdx.x] - 1u] = position;
        count += scan[LM_LATENT_SHARD_THREADS - 1u];
        __syncthreads();
    }
    return count;
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE, uint32_t HEADS>
__global__ __launch_bounds__(LM_LATENT_SHARD_THREADS, 1)
void LmLatentShardPartialKernel(
    Pages cache,
    const uint16_t *__restrict__ query_bf16,
    uint64_t query_rank_stride,
    uint32_t heads_per_rank,
    const uint32_t *__restrict__ sequence_of_row,
    const uint32_t *__restrict__ context_length,
    const uint32_t *__restrict__ row_position,
    const uint32_t *__restrict__ selected_positions,
    uint32_t selected_count,
    uint32_t dense_limit,
    float qk_scale,
    float *__restrict__ partials,
    uint64_t partial_rank_stride)
{
    constexpr uint32_t WIDTH = LATENT + ROPE;
    constexpr uint32_t PER_LANE = WIDTH / 32u;
    static_assert(WIDTH % 64u == 0u, "each lane loads whole bf16 pairs");
    static_assert(Geometry::kSlotBytes >= WIDTH * 2u, "the slot holds the latent and its rope part");
    __shared__ uint32_t local_list[LM_LATENT_SHARD_LIST_CAPACITY];
    __shared__ uint32_t scan[LM_LATENT_SHARD_THREADS];
    __shared__ float warp_max[LM_LATENT_SHARD_WARPS][HEADS];
    __shared__ float warp_sum[LM_LATENT_SHARD_WARPS][HEADS];
    __shared__ float merged[HEADS][LATENT];
    const uint8_t *slot[LM_LATENT_SHARD_UNROLL];
    float query[HEADS][PER_LANE], accumulator[HEADS][PER_LANE], value[LM_LATENT_SHARD_UNROLL][PER_LANE];
    float running_max[HEADS], running_sum[HEADS], global_max[HEADS], global_sum[HEADS], scale;
    uint32_t row = blockIdx.x, group = blockIdx.y, warp = threadIdx.x / 32u, lane = threadIdx.x % 32u;
    uint32_t sequence = sequence_of_row[row], head, global_head, index, element, dense, listed, count, step, unroll, failed, position;
    uint64_t base;

    if (!LmKvViewIsConfigured(cache.pages) || sequence >= cache.pages.sequence_count)
    {
        LmKvReportRequiredAccessFailure(cache.pages, !LmKvViewIsConfigured(cache.pages) ? LM_KV_ACCESS_ERROR_INVALID_VIEW : LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE, LM_KV_ACCESS_READ, row, sequence, 0xffffffffu, 0xffffffffu);
        return;
    }
    dense = context_length[sequence];
    if (row_position != 0 && row_position[row] < dense)
        dense = row_position[row] + 1u;
    listed = selected_positions != 0 && dense > dense_limit ? 1u : 0u;
    if (listed != 0u && selected_count > LM_LATENT_SHARD_LIST_CAPACITY)
    {
        LmKvReportRequiredAccessFailure(cache.pages, (LmKvAccessErrorCode)LM_FRAME_ERROR_SPARSE_INDEX_OUT_OF_RANGE, LM_KV_ACCESS_READ, row, sequence, selected_count, 0xffffffffu);
        return;
    }
    count = listed != 0u
        ? LmLatentShardCompact(cache.shard, selected_positions + (uint64_t)row * selected_count, selected_count, row_position != 0 ? row_position[row] : 0u, row_position != 0 ? 1u : 0u, local_list, scan)
        : SparkKvShardLocalKeys(cache.shard, dense);
    for (head = 0u; head < HEADS; ++head)
    {
        global_head = group * HEADS + head;
        running_max[head] = -INFINITY;
        running_sum[head] = 0.0f;
        LmLatentShardLoad<PER_LANE>(query_bf16 + (uint64_t)(global_head / heads_per_rank) * query_rank_stride + ((uint64_t)row * heads_per_rank + global_head % heads_per_rank) * WIDTH + lane * PER_LANE, query[head]);
        for (element = 0u; element < PER_LANE; ++element)
            accumulator[head][element] = 0.0f;
    }
    failed = 0u;
    for (step = warp; step < count && failed == 0u; step += LM_LATENT_SHARD_UNROLL * LM_LATENT_SHARD_WARPS)
    {
        for (unroll = 0u; unroll < LM_LATENT_SHARD_UNROLL; ++unroll)
        {
            index = step + unroll * LM_LATENT_SHARD_WARPS;
            slot[unroll] = 0;
            if (index >= count || failed != 0u)
                continue;
            position = listed != 0u ? local_list[index] : SparkKvShardLocalPosition(cache.shard, index);
            slot[unroll] = LmKvShardSlotRequired<Geometry>(cache, sequence, position, row, LM_KV_ACCESS_READ);
            failed = slot[unroll] == 0 ? 1u : 0u;
        }
        for (unroll = 0u; unroll < LM_LATENT_SHARD_UNROLL; ++unroll)
            if (slot[unroll] != 0)
                LmLatentShardLoad<PER_LANE>((const uint16_t *)slot[unroll] + lane * PER_LANE, value[unroll]);
        for (unroll = 0u; unroll < LM_LATENT_SHARD_UNROLL; ++unroll)
            if (slot[unroll] != 0)
                LmLatentShardStep<HEADS, PER_LANE>(query, value[unroll], qk_scale, running_max, running_sum, accumulator);
    }
    if (lane == 0u)
        for (head = 0u; head < HEADS; ++head)
        {
            warp_max[warp][head] = running_max[head];
            warp_sum[warp][head] = running_sum[head];
        }
    for (index = threadIdx.x; index < HEADS * LATENT; index += LM_LATENT_SHARD_THREADS)
        merged[index / LATENT][index % LATENT] = 0.0f;
    __syncthreads();
    for (head = 0u; head < HEADS; ++head)
    {
        global_max[head] = -INFINITY;
        global_sum[head] = 0.0f;
        for (index = 0u; index < LM_LATENT_SHARD_WARPS; ++index)
            global_max[head] = fmaxf(global_max[head], warp_max[index][head]);
        for (index = 0u; index < LM_LATENT_SHARD_WARPS; ++index)
        {
            scale = global_max[head] == -INFINITY || warp_max[index][head] == -INFINITY ? 0.0f : __expf(warp_max[index][head] - global_max[head]);
            global_sum[head] = fmaf(warp_sum[index][head], scale, global_sum[head]);
        }
    }
    for (index = 0u; index < LM_LATENT_SHARD_WARPS; ++index)
    {
        if (warp == index)
            for (head = 0u; head < HEADS; ++head)
            {
                scale = global_max[head] == -INFINITY || warp_max[index][head] == -INFINITY ? 0.0f : __expf(warp_max[index][head] - global_max[head]);
                for (element = 0u; element < PER_LANE; ++element)
                    if (lane * PER_LANE + element < LATENT)
                        merged[head][lane * PER_LANE + element] += accumulator[head][element] * scale;
            }
        __syncthreads();
    }
    for (index = threadIdx.x; index < HEADS * LATENT; index += LM_LATENT_SHARD_THREADS)
    {
        head = index / LATENT;
        element = index % LATENT;
        global_head = group * HEADS + head;
        base = (uint64_t)(global_head / heads_per_rank) * partial_rank_stride + ((uint64_t)row * heads_per_rank + global_head % heads_per_rank) * LM_LATENT_SHARD_RECORD_FLOATS(LATENT);
        partials[base + 2u + element] = merged[head][element];
        if (element == 0u)
        {
            partials[base] = global_max[head];
            partials[base + 1u] = global_sum[head];
        }
    }
}

template<uint32_t THREADS, uint32_t LATENT>
__global__ __launch_bounds__(THREADS, 1)
void LmLatentShardMergeKernel(
    const float *__restrict__ partials,
    uint64_t source_stride,
    uint32_t sources,
    uint32_t heads_per_rank,
    uint16_t *__restrict__ output_bf16)
{
    __shared__ float scales[SPARK_KV_SHARD_MAX_DEGREE];
    __shared__ float denominator_shared;
    uint32_t row = blockIdx.x, head = blockIdx.y, source, element;
    uint64_t block_base = ((uint64_t)row * heads_per_rank + head) * LM_LATENT_SHARD_RECORD_FLOATS(LATENT);
    float global_max, denominator, source_max, scale, merged;
    if (threadIdx.x == 0u)
    {
        global_max = -INFINITY;
        for (source = 0u; source < sources; ++source)
        {
            source_max = partials[(uint64_t)source * source_stride + block_base];
            global_max = source_max > global_max ? source_max : global_max;
        }
        denominator = 0.0f;
        for (source = 0u; source < sources; ++source)
        {
            source_max = partials[(uint64_t)source * source_stride + block_base];
            scale = global_max == -INFINITY || source_max == -INFINITY ? 0.0f : __expf(source_max - global_max);
            scales[source] = scale;
            denominator = fmaf(partials[(uint64_t)source * source_stride + block_base + 1u], scale, denominator);
        }
        denominator_shared = denominator;
    }
    __syncthreads();
    for (element = threadIdx.x; element < LATENT; element += THREADS)
    {
        merged = 0.0f;
        for (source = 0u; source < sources; ++source)
            merged = fmaf(partials[(uint64_t)source * source_stride + block_base + 2u + element], scales[source], merged);
        output_bf16[((uint64_t)row * heads_per_rank + head) * LATENT + element] = LmFloatToBf16(merged / fmaxf(denominator_shared, 1.0e-20f));
    }
}

static inline uint32_t LmLatentShardHeadsPerBlock(uint32_t heads_per_rank)
{
    return heads_per_rank % 4u == 0u ? 4u : heads_per_rank % 2u == 0u ? 2u : 1u;
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline cudaError_t LmLatentShardPartialHeadsLaunch(
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
    uint32_t rows,
    cudaStream_t stream)
{
    uint32_t per_block;
    if (rows == 0u || heads_per_rank == 0u || query_bf16 == 0 || partials == 0 ||
        SparkKvShardValid(cache.shard, Geometry::kPageSlots) == 0u ||
        (selected_positions != 0 && selected_count > LM_LATENT_SHARD_LIST_CAPACITY) ||
        query_rank_stride < (uint64_t)rows * heads_per_rank * (LATENT + ROPE) ||
        partial_rank_stride < (uint64_t)rows * heads_per_rank * LM_LATENT_SHARD_RECORD_FLOATS(LATENT))
        return cudaErrorInvalidValue;
    per_block = LmLatentShardHeadsPerBlock(heads_per_rank);
    if (per_block == 4u)
        LM_LAUNCH((LmLatentShardPartialKernel<Geometry, Pages, LATENT, ROPE, 4u>), dim3(rows, total_heads / 4u), LM_LATENT_SHARD_THREADS, 0, stream, cache, query_bf16, query_rank_stride, heads_per_rank, sequence_of_row, context_length, row_position, selected_positions, selected_count, dense_limit, qk_scale, partials, partial_rank_stride);
    else if (per_block == 2u)
        LM_LAUNCH((LmLatentShardPartialKernel<Geometry, Pages, LATENT, ROPE, 2u>), dim3(rows, total_heads / 2u), LM_LATENT_SHARD_THREADS, 0, stream, cache, query_bf16, query_rank_stride, heads_per_rank, sequence_of_row, context_length, row_position, selected_positions, selected_count, dense_limit, qk_scale, partials, partial_rank_stride);
    else
        LM_LAUNCH((LmLatentShardPartialKernel<Geometry, Pages, LATENT, ROPE, 1u>), dim3(rows, total_heads), LM_LATENT_SHARD_THREADS, 0, stream, cache, query_bf16, query_rank_stride, heads_per_rank, sequence_of_row, context_length, row_position, selected_positions, selected_count, dense_limit, qk_scale, partials, partial_rank_stride);
    return cudaPeekAtLastError();
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline cudaError_t LmLatentShardPartialLaunch(
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
    return LmLatentShardPartialHeadsLaunch<Geometry, Pages, LATENT, ROPE>(cache, query_bf16, query_rank_stride, heads_per_rank, heads_per_rank * cache.shard.degree, sequence_of_row, context_length, row_position, selected_positions, selected_count, dense_limit, qk_scale, partials, partial_rank_stride, rows, stream);
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline cudaError_t LmLatentShardPartialOwnHeadsLaunch(
    Pages cache,
    const uint16_t *query_bf16,
    uint32_t heads_per_rank,
    const uint32_t *sequence_of_row,
    const uint32_t *context_length,
    const uint32_t *row_position,
    const uint32_t *selected_positions,
    uint32_t selected_count,
    uint32_t dense_limit,
    float qk_scale,
    float *partials,
    uint32_t rows,
    cudaStream_t stream)
{
    return LmLatentShardPartialHeadsLaunch<Geometry, Pages, LATENT, ROPE>(cache, query_bf16, (uint64_t)rows * heads_per_rank * (LATENT + ROPE), heads_per_rank, heads_per_rank, sequence_of_row, context_length, row_position, selected_positions, selected_count, dense_limit, qk_scale, partials, (uint64_t)rows * heads_per_rank * LM_LATENT_SHARD_RECORD_FLOATS(LATENT), rows, stream);
}

template<uint32_t LATENT>
static inline cudaError_t LmLatentShardMergeLaunch(
    const float *partials,
    uint64_t source_stride,
    uint32_t sources,
    uint32_t heads_per_rank,
    uint16_t *output_bf16,
    uint32_t rows,
    cudaStream_t stream)
{
    if (rows == 0u || heads_per_rank == 0u || sources == 0u || sources > SPARK_KV_SHARD_MAX_DEGREE || partials == 0 || output_bf16 == 0 ||
        source_stride < (uint64_t)rows * heads_per_rank * LM_LATENT_SHARD_RECORD_FLOATS(LATENT))
        return cudaErrorInvalidValue;
    LM_LAUNCH((LmLatentShardMergeKernel<LM_LATENT_SHARD_THREADS, LATENT>), dim3(rows, heads_per_rank), LM_LATENT_SHARD_THREADS, 0, stream, partials, source_stride, sources, heads_per_rank, output_bf16);
    return cudaPeekAtLastError();
}
