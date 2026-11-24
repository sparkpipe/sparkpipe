#pragma once

#include <stdint.h>
#include <math.h>

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/shard_reduce.cuh"
#include "sparkpipe/spark_gpu.h"

#define LM_LATENT_SHARD_WARPS 8u
#define LM_LATENT_SHARD_THREADS (LM_LATENT_SHARD_WARPS * 32u)
#define LM_LATENT_SHARD_UNROLL 2u
#define LM_LATENT_SHARD_LIST_CAPACITY 4096u
#define LM_LATENT_SHARD_RECORD_FLOATS(latent) ((latent) + 2u)

static __device__ __forceinline__ uint32_t LmLatentShardCompact(
    SparkKvShard shard,
    const uint32_t *__restrict__ selected,
    uint32_t selected_count,
    uint32_t bound,
    uint32_t bounded,
    uint32_t *list,
    uint32_t *scan)
{
    const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, below = (1u << lane) - 1u;
    uint32_t base, count = 0u, keep, position, index, mask, rank_in_warp, phase = 0u;
    for (base = 0u; base < selected_count; base += LM_LATENT_SHARD_THREADS)
    {
        uint32_t *warp_scan = scan + phase * 16u;
        index = base + threadIdx.x;
        position = index < selected_count ? selected[index] : 0xffffffffu;
        keep = index < selected_count && (bounded == 0u || position <= bound) && SparkKvShardOwns(shard, position) != 0u ? 1u : 0u;
        mask = __ballot_sync(0xffffffffu, keep != 0u);
        rank_in_warp = (uint32_t)__popc(mask & below);
        if (lane == 0u)
            warp_scan[warp] = (uint32_t)__popc(mask);
        __syncthreads();
        if (threadIdx.x == 0u)
        {
            uint32_t running = 0u, w, here;
            for (w = 0u; w < LM_LATENT_SHARD_WARPS; ++w)
            {
                here = warp_scan[w];
                warp_scan[w] = running;
                running += here;
            }
            warp_scan[LM_LATENT_SHARD_WARPS] = running;
        }
        __syncthreads();
        if (keep != 0u)
            list[count + warp_scan[warp] + rank_in_warp] = position;
        count += warp_scan[LM_LATENT_SHARD_WARPS];
        phase ^= 1u;
    }
    if (selected_count != 0u)
        __syncthreads();
    return count;
}

static __global__ __launch_bounds__(LM_LATENT_SHARD_THREADS)
void LmLatentShardSplitKernel(
    SparkKvShard shard,
    const uint32_t *__restrict__ selected_positions,
    uint32_t selected_count,
    const uint32_t *__restrict__ row_position,
    uint32_t *__restrict__ split_positions,
    uint32_t *__restrict__ split_offsets)
{
    __shared__ uint32_t warp_counts[LM_LATENT_SHARD_WARPS][SPARK_KV_SHARD_MAX_DEGREE];
    __shared__ uint32_t chunk_total[SPARK_KV_SHARD_MAX_DEGREE];
    __shared__ uint32_t cursor[SPARK_KV_SHARD_MAX_DEGREE];
    __shared__ uint32_t starts[SPARK_KV_SHARD_MAX_DEGREE + 1u];
    const uint32_t row = blockIdx.x, warp = threadIdx.x / 32u, lane = threadIdx.x % 32u, below = (1u << lane) - 1u;
    const uint32_t *selected = selected_positions + (uint64_t)row * selected_count;
    uint32_t *split = split_positions + (uint64_t)row * selected_count;
    uint32_t base, index, position, owner, source, mask, rank_in_warp, running, here, w;
    if (threadIdx.x < shard.degree)
        cursor[threadIdx.x] = 0u;
    __syncthreads();
    for (base = 0u; base < selected_count; base += LM_LATENT_SHARD_THREADS)
    {
        index = base + threadIdx.x;
        position = index < selected_count ? selected[index] : 0xffffffffu;
        owner = index < selected_count && (row_position == 0 || position <= row_position[row]) ? SparkKvShardOwner(shard, position) : 0xffffffffu;
        for (source = 0u; source < shard.degree; ++source)
        {
            mask = __ballot_sync(0xffffffffu, owner == source);
            if (lane == 0u && mask != 0u)
                atomicAdd(&cursor[source], (uint32_t)__popc(mask));
        }
    }
    __syncthreads();
    if (threadIdx.x == 0u)
    {
        starts[0] = 0u;
        for (source = 0u; source < shard.degree; ++source)
            starts[source + 1u] = starts[source] + cursor[source];
    }
    __syncthreads();
    if (threadIdx.x <= shard.degree)
        split_offsets[(uint64_t)row * (shard.degree + 1u) + threadIdx.x] = starts[threadIdx.x];
    if (threadIdx.x < shard.degree)
        cursor[threadIdx.x] = 0u;
    __syncthreads();
    for (base = 0u; base < selected_count; base += LM_LATENT_SHARD_THREADS)
    {
        index = base + threadIdx.x;
        position = index < selected_count ? selected[index] : 0xffffffffu;
        owner = index < selected_count && (row_position == 0 || position <= row_position[row]) ? SparkKvShardOwner(shard, position) : 0xffffffffu;
        rank_in_warp = 0u;
        for (source = 0u; source < shard.degree; ++source)
        {
            mask = __ballot_sync(0xffffffffu, owner == source);
            if (lane == 0u)
                warp_counts[warp][source] = (uint32_t)__popc(mask);
            if (owner == source)
                rank_in_warp = (uint32_t)__popc(mask & below);
        }
        __syncthreads();
        if (threadIdx.x < shard.degree)
        {
            running = 0u;
            for (w = 0u; w < LM_LATENT_SHARD_WARPS; ++w)
            {
                here = warp_counts[w][threadIdx.x];
                warp_counts[w][threadIdx.x] = running;
                running += here;
            }
            chunk_total[threadIdx.x] = running;
        }
        __syncthreads();
        if (owner != 0xffffffffu)
            split[starts[owner] + cursor[owner] + warp_counts[warp][owner] + rank_in_warp] = position;
        __syncthreads();
        if (threadIdx.x < shard.degree)
            cursor[threadIdx.x] += chunk_total[threadIdx.x];
        __syncthreads();
    }
}

static inline SparkGpuError LmLatentShardSplitLaunch(SparkKvShard shard, const uint32_t *selected_positions, uint32_t selected_count,
    const uint32_t *row_position, uint32_t *split_positions, uint32_t *split_offsets, uint32_t rows, SparkGpuStream stream)
{
    if (rows == 0u || selected_positions == 0 || split_positions == 0 || split_offsets == 0 || shard.degree == 0u ||
        shard.degree > SPARK_KV_SHARD_MAX_DEGREE || shard.grain == 0u || selected_count > LM_LATENT_SHARD_LIST_CAPACITY)
        return SPARK_GPU_ERROR_INVALID_VALUE;
    LM_LAUNCH((LmLatentShardSplitKernel), rows, LM_LATENT_SHARD_THREADS, 0, stream, shard, selected_positions, selected_count, row_position, split_positions,
        split_offsets);
    return SparkGpuPeekAtLastError();
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
    uint64_t partial_rank_stride,
    uint32_t record_heads,
    uint32_t record_copies,
    const uint32_t *__restrict__ split_positions,
    const uint32_t *__restrict__ split_offsets)
{
    constexpr uint32_t WIDTH = LATENT + ROPE;
    constexpr uint32_t PER_LANE = WIDTH / 32u;
    constexpr uint32_t MERGE_SLICE = (HEADS <= 2u && LATENT <= 512u) ? LATENT : (LATENT < 256u ? LATENT : 256u);
    static_assert(WIDTH % 64u == 0u, "each lane loads whole bf16 pairs");
    static_assert(Geometry::kSlotBytes >= WIDTH * 2u, "the slot holds the latent and its rope part");
    __shared__ union
    {
        struct
        {
            uint32_t list[LM_LATENT_SHARD_LIST_CAPACITY];
            uint32_t scan[LM_LATENT_SHARD_THREADS];
        } compact;
        float warp_acc[LM_LATENT_SHARD_WARPS][HEADS][MERGE_SLICE];
    } shared_buf;
    __shared__ float warp_max[LM_LATENT_SHARD_WARPS][HEADS];
    __shared__ float warp_sum[LM_LATENT_SHARD_WARPS][HEADS];
    __shared__ float warp_scale[LM_LATENT_SHARD_WARPS][HEADS];
    __shared__ float global_max_s[HEADS];
    const uint8_t *slot[LM_LATENT_SHARD_UNROLL];
    float query[HEADS][PER_LANE], accumulator[HEADS][PER_LANE], value[LM_LATENT_SHARD_UNROLL][PER_LANE];
    float running_max[HEADS], running_sum[HEADS];
    uint32_t row = blockIdx.x, group = blockIdx.y, warp = threadIdx.x / 32u, lane = threadIdx.x % 32u;
    uint32_t sequence = sequence_of_row[row], head, global_head, index, element, dense, listed, count, step, unroll, failed, position, copy;
    uint64_t base;

    if (!LmKvCheckSequence(cache.pages, row, sequence))
        return;
    dense = context_length[sequence];
    if (row_position != 0 && row_position[row] == 0xffffffffu)
        dense = 0u;
    else if (row_position != 0 && row_position[row] < dense)
        dense = row_position[row] + 1u;
    listed = selected_positions != 0 && dense > dense_limit ? 1u : 0u;
    if (listed != 0u && selected_count > LM_LATENT_SHARD_LIST_CAPACITY)
    {
        LmKvReportRequiredAccessFailure(cache.pages, (LmKvAccessErrorCode)LM_FRAME_ERROR_SPARSE_INDEX_OUT_OF_RANGE, LM_KV_ACCESS_READ, row, sequence, selected_count, 0xffffffffu);
        return;
    }
    const uint32_t *split_list = 0;
    if (listed != 0u && split_positions != 0)
    {
        const uint32_t *offsets = split_offsets + (uint64_t)row * (cache.shard.degree + 1u) + cache.shard.rank;
        split_list = split_positions + (uint64_t)row * selected_count + offsets[0];
        count = offsets[1] - offsets[0];
    }
    else
        count = listed != 0u
            ? LmLatentShardCompact(cache.shard, selected_positions + (uint64_t)row * selected_count, selected_count, row_position != 0 ? row_position[row] : 0u, row_position != 0 ? 1u : 0u, shared_buf.compact.list, shared_buf.compact.scan)
            : SparkKvShardLocalKeys(cache.shard, dense);
    for (head = 0u; head < HEADS; ++head)
    {
        global_head = group * HEADS + head;
        running_max[head] = -INFINITY;
        running_sum[head] = 0.0f;
        LmLatentVecLoad<PER_LANE>(query_bf16 + (uint64_t)(global_head / heads_per_rank) * query_rank_stride + ((uint64_t)row * heads_per_rank + global_head % heads_per_rank) * WIDTH + lane * PER_LANE, query[head]);
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
            position = split_list != 0 ? split_list[index] : listed != 0u ? shared_buf.compact.list[index] : SparkKvShardLocalPosition(cache.shard, index);
            slot[unroll] = LmKvShardSlotRequired<Geometry>(cache, sequence, position, row, LM_KV_ACCESS_READ);
            failed = slot[unroll] == 0 ? 1u : 0u;
        }
        for (unroll = 0u; unroll < LM_LATENT_SHARD_UNROLL; ++unroll)
            if (slot[unroll] != 0)
                LmLatentVecLoad<PER_LANE>((const uint16_t *)slot[unroll] + lane * PER_LANE, value[unroll]);
        for (unroll = 0u; unroll < LM_LATENT_SHARD_UNROLL; ++unroll)
            if (slot[unroll] != 0)
                LmLatentOnlineStep<HEADS, PER_LANE>(query, value[unroll], qk_scale, running_max, running_sum, accumulator);
    }
    LmWarpScaleMax<LM_LATENT_SHARD_WARPS, HEADS>(running_max, running_sum, warp_max, warp_sum, warp_scale, global_max_s, warp, lane);
    for (uint32_t slice = 0u; slice < LATENT; slice += MERGE_SLICE)
    {
        const uint32_t slice_end = slice + MERGE_SLICE < LATENT ? slice + MERGE_SLICE : LATENT;
        const uint32_t slice_len = slice_end - slice;
        for (head = 0u; head < HEADS; ++head)
            for (element = 0u; element < PER_LANE; ++element)
            {
                const uint32_t col = lane * PER_LANE + element;
                if (col >= slice && col < slice_end)
                    shared_buf.warp_acc[warp][head][col - slice] = accumulator[head][element];
            }
        __syncthreads();
        for (index = threadIdx.x; index < HEADS * slice_len; index += LM_LATENT_SHARD_THREADS)
        {
            head = index / slice_len;
            const uint32_t local_col = index % slice_len;
            element = slice + local_col;
            global_head = group * HEADS + head;
            float total = 0.0f;
            #pragma unroll
            for (uint32_t w = 0u; w < LM_LATENT_SHARD_WARPS; ++w)
                total += shared_buf.warp_acc[w][head][local_col] * warp_scale[w][head];
            float gs = 0.0f;
            if (element == 0u)
                for (uint32_t w = 0u; w < LM_LATENT_SHARD_WARPS; ++w)
                    gs = fmaf(warp_sum[w][head], warp_scale[w][head], gs);
            for (copy = 0u; copy < record_copies; ++copy)
            {
                base = ((uint64_t)(global_head / record_heads) * record_copies + copy) * partial_rank_stride + ((uint64_t)row * record_heads + global_head % record_heads) * LM_LATENT_SHARD_RECORD_FLOATS(LATENT);
                partials[base + 2u + element] = total;
                if (element == 0u)
                {
                    partials[base] = global_max_s[head];
                    partials[base + 1u] = gs;
                }
            }
        }
        if (slice_end < LATENT)
            __syncthreads();
    }
}

static __global__ __launch_bounds__(LM_LATENT_SHARD_THREADS, 1)
void LmQueryRopePackKernel(
    const uint16_t *__restrict__ latent_bf16,
    const uint16_t *__restrict__ q_bf16,
    uint16_t *__restrict__ query_bf16,
    uint32_t heads,
    uint32_t latent_dim,
    uint32_t nope_dim,
    uint32_t rope_dim)
{
    const uint64_t pair = (uint64_t)blockIdx.x * heads + blockIdx.y;
    const uint32_t total = latent_dim + rope_dim;
    if (((latent_dim | nope_dim | rope_dim) & 7u) == 0u &&
        ((((uintptr_t)latent_bf16 | (uintptr_t)q_bf16 | (uintptr_t)query_bf16) & 15u) == 0u))
    {
        const uint32_t latent_vecs = latent_dim >> 3u, total_vecs = total >> 3u;
        const uint4 *lat4 = (const uint4 *)(latent_bf16 + pair * latent_dim);
        const uint4 *rope4 = (const uint4 *)(q_bf16 + pair * (nope_dim + rope_dim) + nope_dim);
        uint4 *out4 = (uint4 *)(query_bf16 + pair * total);
        for (uint32_t v = threadIdx.x; v < total_vecs; v += blockDim.x)
            out4[v] = v < latent_vecs ? lat4[v] : rope4[v - latent_vecs];
        return;
    }
    for (uint32_t element = threadIdx.x; element < total; element += blockDim.x)
        query_bf16[pair * total + element] = element < latent_dim
            ? latent_bf16[pair * latent_dim + element]
            : q_bf16[pair * (nope_dim + rope_dim) + nope_dim + element - latent_dim];
}

template<uint32_t LATENT, uint32_t ROPE>
static inline SparkGpuError LmLatentShardQueryPackLaunch(const uint16_t *latent_bf16, const uint16_t *rope_bf16, uint32_t heads, uint32_t rows, uint16_t *query_bf16, SparkGpuStream stream)
{
    if (latent_bf16 == 0 || rope_bf16 == 0 || query_bf16 == 0 || heads == 0u || rows == 0u)
        return SPARK_GPU_ERROR_INVALID_VALUE;
    LM_LAUNCH((LmQueryRopePackKernel), dim3(rows, heads), LM_LATENT_SHARD_THREADS, 0, stream, latent_bf16, rope_bf16, query_bf16, heads, LATENT, 0u, ROPE);
    return SparkGpuPeekAtLastError();
}

template<uint32_t THREADS, uint32_t LATENT>
__global__ __launch_bounds__(THREADS, 1)
void LmLatentShardMergeKernel(
    const float *__restrict__ partials,
    uint64_t source_stride,
    uint32_t sources,
    uint32_t heads_per_rank,
    uint16_t *__restrict__ output_bf16,
    const float *__restrict__ sinks)
{
    __shared__ float scales[SPARK_KV_SHARD_MAX_DEGREE];
    __shared__ float denom_shared;
    const uint32_t row = blockIdx.x, head = blockIdx.y;
    const uint64_t block_base = ((uint64_t)row * heads_per_rank + head) * LM_LATENT_SHARD_RECORD_FLOATS(LATENT);
    if (threadIdx.x < 32u)
    {
        float gm = -INFINITY;
        float den = LmWarpMergeScales(partials + block_base, source_stride, sources, scales, &gm);
        if (threadIdx.x == 0u)
        {
            if (sinks != 0)
                den += gm == -INFINITY ? __expf(sinks[head]) : __expf(sinks[head] - gm);
            denom_shared = fmaxf(den, 1.0e-20f);
        }
    }
    __syncthreads();
    LmWeightedSourcesMergeBf16<LATENT, THREADS>(
        partials + block_base + 2u, source_stride, sources, scales, denom_shared,
        output_bf16 + ((uint64_t)row * heads_per_rank + head) * LATENT);
}

template<uint32_t THREADS, uint32_t LATENT>
__global__ __launch_bounds__(THREADS, 1)
void LmLatentShardFoldKernel(
    const float *__restrict__ partials,
    uint64_t source_stride,
    uint32_t sources,
    uint32_t heads_per_rank,
    float *__restrict__ running,
    uint32_t first)
{
    __shared__ float scales[SPARK_KV_SHARD_MAX_DEGREE + 1u];
    const uint32_t row = blockIdx.x, head = blockIdx.y;
    const uint64_t block_base = ((uint64_t)row * heads_per_rank + head) * LM_LATENT_SHARD_RECORD_FLOATS(LATENT);
    if (threadIdx.x < 32u)
    {
        const float run_max = first != 0u ? -INFINITY : running[block_base];
        const float run_den = first != 0u ? 0.0f : running[block_base + 1u];
        float gm = -INFINITY;
        const float den = LmWarpMergeScales(
            partials + block_base, source_stride, sources, scales, &gm, run_max, run_den, &scales[sources]);
        if (threadIdx.x == 0u)
        {
            running[block_base] = gm;
            running[block_base + 1u] = den;
        }
    }
    __syncthreads();
    LmWeightedSourcesFoldF32<LATENT, THREADS>(
        partials + block_base + 2u, source_stride, sources, scales, scales[sources], first,
        running + block_base + 2u);
}

static inline uint32_t LmLatentShardHeadsPerBlock(uint32_t heads_per_rank)
{
    return heads_per_rank % 4u == 0u ? 4u : heads_per_rank % 2u == 0u ? 2u : 1u;
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline SparkGpuError LmLatentShardPartialHeadsLaunch(
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
    uint32_t heads_per_block_limit = 4u)
{
    uint32_t per_block;
    if (rows == 0u || heads_per_rank == 0u || query_bf16 == 0 || partials == 0 ||
        record_heads == 0u || record_copies == 0u || total_heads % record_heads != 0u ||
        SparkKvShardValid(cache.shard, Geometry::kPageSlots) == 0u ||
        (selected_positions != 0 && selected_count > LM_LATENT_SHARD_LIST_CAPACITY) ||
        query_rank_stride < (uint64_t)rows * heads_per_rank * (LATENT + ROPE) ||
        partial_rank_stride < (uint64_t)rows * record_heads * LM_LATENT_SHARD_RECORD_FLOATS(LATENT))
        return SPARK_GPU_ERROR_INVALID_VALUE;
    per_block = LmLatentShardHeadsPerBlock(heads_per_rank);
    while (per_block > heads_per_block_limit && per_block > 1u)
        per_block /= 2u;
    if (per_block == 4u)
        LM_LAUNCH((LmLatentShardPartialKernel<Geometry, Pages, LATENT, ROPE, 4u>), dim3(rows, total_heads / 4u), LM_LATENT_SHARD_THREADS, 0, stream, cache, query_bf16, query_rank_stride, heads_per_rank, sequence_of_row, context_length, row_position, selected_positions, selected_count, dense_limit, qk_scale, partials, partial_rank_stride, record_heads, record_copies, split_positions, split_offsets);
    else if (per_block == 2u)
        LM_LAUNCH((LmLatentShardPartialKernel<Geometry, Pages, LATENT, ROPE, 2u>), dim3(rows, total_heads / 2u), LM_LATENT_SHARD_THREADS, 0, stream, cache, query_bf16, query_rank_stride, heads_per_rank, sequence_of_row, context_length, row_position, selected_positions, selected_count, dense_limit, qk_scale, partials, partial_rank_stride, record_heads, record_copies, split_positions, split_offsets);
    else
        LM_LAUNCH((LmLatentShardPartialKernel<Geometry, Pages, LATENT, ROPE, 1u>), dim3(rows, total_heads), LM_LATENT_SHARD_THREADS, 0, stream, cache, query_bf16, query_rank_stride, heads_per_rank, sequence_of_row, context_length, row_position, selected_positions, selected_count, dense_limit, qk_scale, partials, partial_rank_stride, record_heads, record_copies, split_positions, split_offsets);
    return SparkGpuPeekAtLastError();
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline SparkGpuError LmLatentShardPartialLaunch(
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
    SparkGpuStream stream)
{
    return LmLatentShardPartialHeadsLaunch<Geometry, Pages, LATENT, ROPE>(cache, query_bf16, query_rank_stride, heads_per_rank, heads_per_rank * cache.shard.degree, sequence_of_row, context_length, row_position, selected_positions, selected_count, dense_limit, qk_scale, partials, partial_rank_stride, heads_per_rank, 1u, rows, stream);
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline SparkGpuError LmLatentShardPartialOwnHeadsLaunch(
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
    SparkGpuStream stream)
{
    return LmLatentShardPartialHeadsLaunch<Geometry, Pages, LATENT, ROPE>(cache, query_bf16, (uint64_t)rows * heads_per_rank * (LATENT + ROPE), heads_per_rank, heads_per_rank, sequence_of_row, context_length, row_position, selected_positions, selected_count, dense_limit, qk_scale, partials, (uint64_t)rows * heads_per_rank * LM_LATENT_SHARD_RECORD_FLOATS(LATENT), heads_per_rank, 1u, rows, stream);
}

template<uint32_t LATENT>
static inline SparkGpuError LmLatentShardFoldLaunch(
    const float *partials,
    uint64_t source_stride,
    uint32_t sources,
    uint32_t heads_per_rank,
    float *running,
    uint32_t rows,
    uint32_t first,
    SparkGpuStream stream)
{
    if (rows == 0u || heads_per_rank == 0u || sources == 0u || sources > SPARK_KV_SHARD_MAX_DEGREE || partials == 0 || running == 0 ||
        source_stride < (uint64_t)rows * heads_per_rank * LM_LATENT_SHARD_RECORD_FLOATS(LATENT))
        return SPARK_GPU_ERROR_INVALID_VALUE;
    LM_LAUNCH((LmLatentShardFoldKernel<LM_LATENT_SHARD_THREADS, LATENT>), dim3(rows, heads_per_rank), LM_LATENT_SHARD_THREADS, 0, stream, partials, source_stride, sources, heads_per_rank, running, first);
    return SparkGpuPeekAtLastError();
}

template<uint32_t LATENT>
static inline SparkGpuError LmLatentShardMergeLaunch(
    const float *partials,
    uint64_t source_stride,
    uint32_t sources,
    uint32_t heads_per_rank,
    uint16_t *output_bf16,
    uint32_t rows,
    SparkGpuStream stream,
    const float *sinks = 0)
{
    if (rows == 0u || heads_per_rank == 0u || sources == 0u || sources > SPARK_KV_SHARD_MAX_DEGREE || partials == 0 || output_bf16 == 0 ||
        source_stride < (uint64_t)rows * heads_per_rank * LM_LATENT_SHARD_RECORD_FLOATS(LATENT))
        return SPARK_GPU_ERROR_INVALID_VALUE;
    LM_LAUNCH((LmLatentShardMergeKernel<LM_LATENT_SHARD_THREADS, LATENT>), dim3(rows, heads_per_rank), LM_LATENT_SHARD_THREADS, 0, stream, partials, source_stride, sources, heads_per_rank, output_bf16, sinks);
    return SparkGpuPeekAtLastError();
}
