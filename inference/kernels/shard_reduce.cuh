#pragma once

#include <math.h>
#include <stdint.h>

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/mma.cuh"

static __device__ __forceinline__ bool LmKvCheckSequence(LmKvView view, uint32_t row, uint32_t sequence, uint32_t position = 0xffffffffu, bool report = true)
{
    if (LmKvViewIsConfigured(view) && sequence < view.sequence_count)
        return true;
    if (report)
        LmKvReportRequiredAccessFailure(view, !LmKvViewIsConfigured(view) ? LM_KV_ACCESS_ERROR_INVALID_VIEW : LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE, LM_KV_ACCESS_READ, row, sequence, position, 0xffffffffu);
    return false;
}

template<uint32_t PER_LANE>
static __device__ __forceinline__ void LmLatentVecLoad(const uint16_t *__restrict__ source, float *values)
{
    if constexpr ((PER_LANE % 8u) == 0u)
    {
        for (uint32_t chunk = 0u; chunk < PER_LANE / 8u; ++chunk)
        {
            const uint4 packed = ((const uint4 *)source)[chunk];
            const uint32_t bits[4] = {packed.x, packed.y, packed.z, packed.w};
            for (uint32_t word = 0u; word < 4u; ++word)
            {
                values[chunk * 8u + word * 2u] = __uint_as_float(bits[word] << 16u);
                values[chunk * 8u + word * 2u + 1u] = __uint_as_float(bits[word] & 0xffff0000u);
            }
        }
    }
    else
    {
        for (uint32_t word = 0u; word < PER_LANE / 2u; ++word)
        {
            const uint32_t bits = ((const uint32_t *)source)[word];
            values[word * 2u] = __uint_as_float(bits << 16u);
            values[word * 2u + 1u] = __uint_as_float(bits & 0xffff0000u);
        }
    }
}

template<uint32_t HEADS, uint32_t PER_LANE>
static __device__ __forceinline__ void LmLatentOnlineStep(const float (*query)[PER_LANE], const float *value, float qk_scale, float *running_max, float *running_sum, float (*accumulator)[PER_LANE])
{
    for (uint32_t head = 0u; head < HEADS; ++head)
    {
        float score = 0.0f;
        for (uint32_t element = 0u; element < PER_LANE; ++element)
            score += query[head][element] * value[element];
        for (uint32_t mask = 16u; mask > 0u; mask >>= 1u)
            score += __shfl_xor_sync(0xffffffffu, score, mask);
        score *= qk_scale;
        const float previous = running_max[head];
        running_max[head] = fmaxf(previous, score);
        const float scaled_previous = __expf(previous - running_max[head]);
        const float scaled_current = __expf(score - running_max[head]);
        running_sum[head] = (running_sum[head] * scaled_previous) + scaled_current;
        for (uint32_t element = 0u; element < PER_LANE; ++element)
            accumulator[head][element] = (accumulator[head][element] * scaled_previous) + (SparkMathAttentionValueWeight(scaled_current) * value[element]);
    }
}

template<uint32_t WARPS, uint32_t HEADS>
static __device__ __forceinline__ void LmWarpScaleMax(const float *running_max, const float *running_sum, float (*warp_max)[HEADS], float (*warp_sum)[HEADS], float (*warp_scale)[HEADS], float *global_max_s, uint32_t warp, uint32_t lane)
{
    if (lane == 0u)
        for (uint32_t head = 0u; head < HEADS; ++head)
        {
            warp_max[warp][head] = running_max[head];
            warp_sum[warp][head] = running_sum[head];
        }
    __syncthreads();
    if (warp != 0u)
        return;
    for (uint32_t head = 0u; head < HEADS; ++head)
    {
        const float wm = lane < WARPS ? warp_max[lane][head] : -INFINITY;
        float gm = wm;
        #pragma unroll
        for (uint32_t offset = WARPS / 2u; offset != 0u; offset >>= 1u)
            gm = fmaxf(gm, __shfl_xor_sync(0xffffffffu, gm, offset));
        if (lane < WARPS)
            warp_scale[lane][head] = gm == -INFINITY || wm == -INFINITY ? 0.0f : __expf(wm - gm);
        if (lane == 0u)
            global_max_s[head] = gm;
    }
}

static __device__ __forceinline__ float LmWarpMergeScales(const float *__restrict__ meta_base, uint64_t source_stride, uint32_t sources, float *scales, float *out_global_max = 0, float init_max = -INFINITY, float init_den = 0.0f, float *out_init_scale = 0)
{
    const uint32_t lane = threadIdx.x & 31u;
#if defined(__CUDA_ARCH__)
    float sm = -INFINITY, sd = 0.0f;
    if (lane < sources)
    {
        const float *meta = meta_base + (uint64_t)lane * source_stride;
        sm = meta[0]; sd = meta[1];
    }
    float gm = lane < sources && sm > -INFINITY ? sm : -INFINITY;
    if (lane == 0u && init_max > gm)
        gm = init_max;
    #pragma unroll
    for (uint32_t offset = 16u; offset != 0u; offset >>= 1u)
    {
        const float other = __shfl_xor_sync(0xffffffffu, gm, offset);
        gm = other > gm ? other : gm;
    }
    const float sc = lane < sources ? (gm == -INFINITY || sm == -INFINITY ? 0.0f : __expf(sm - gm)) : 0.0f;
    if (lane < sources)
        scales[lane] = sc;
    __syncwarp();
#else
    if (lane != 0u)
        return 0.0f;
    float gm = init_max;
    for (uint32_t s = 0u; s < sources; ++s)
    {
        const float cand = meta_base[(uint64_t)s * source_stride];
        if (cand > gm)
            gm = cand;
    }
#endif
    const float init_sc = (gm == -INFINITY || init_max == -INFINITY) ? 0.0f : __expf(init_max - gm);
    if (lane == 0u && out_init_scale != 0)
        *out_init_scale = init_sc;
    if (lane == 0u && out_global_max != 0)
        *out_global_max = gm;
    float den = out_init_scale != 0 ? init_den * init_sc : init_den;
    for (uint32_t s = 0u; s < sources; ++s)
    {
#if defined(__CUDA_ARCH__)
        const float d = __shfl_sync(0xffffffffu, sd, s);
        if (lane == 0u)
            den = fmaf(d, scales[s], den);
#else
        const float *meta = meta_base + (uint64_t)s * source_stride;
        const float w = (gm == -INFINITY || meta[0] == -INFINITY) ? 0.0f : __expf(meta[0] - gm);
        scales[s] = w;
        den = fmaf(meta[1], w, den);
#endif
    }
    return den;
}

template<bool FOLD_F32, uint32_t ALIGN_MOD, uint32_t THREADS, uint32_t STRIDE_MOD, class Dst, class SlotAt, class ScaleAt, class DenomAt>
static __device__ __forceinline__ void LmWeightedSourcesApply(const float *__restrict__ src0, uint64_t source_stride, uint32_t sources, uint32_t elem_begin, uint32_t elem_end, Dst *__restrict__ dst, SlotAt slot_at, ScaleAt scale_at, DenomAt denom_at, uint32_t head_align, float running_scale, uint32_t first)
{
    const uint32_t step = THREADS != 0u ? THREADS : blockDim.x;
    const uint32_t dst_align4 = FOLD_F32 ? 15u : 7u, dst_align2 = FOLD_F32 ? 7u : 3u;
    if constexpr (((ALIGN_MOD | STRIDE_MOD) % 4u) == 0u)
    {
        if ((((head_align | elem_begin | elem_end) & 3u) == 0u) && ((((uintptr_t)src0) & 15u) == 0u) && (sources == 1u || (source_stride & 3u) == 0u) && ((((uintptr_t)dst) & dst_align4) == 0u))
        {
            for (uint32_t v = (elem_begin >> 2u) + threadIdx.x; v < (elem_end >> 2u); v += step)
            {
                const uint32_t slot = slot_at(v << 2u);
                const float d = FOLD_F32 ? 1.0f : denom_at(slot);
                float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
                if constexpr (FOLD_F32)
                    if (first == 0u) { const float4 p = ((const float4 *)dst)[v]; m0 = p.x * running_scale; m1 = p.y * running_scale; m2 = p.z * running_scale; m3 = p.w * running_scale; }
                for (uint32_t s = 0u; s < sources; ++s)
                {
                    const float w = scale_at(slot, s);
                    const float4 x = ((const float4 *)(src0 + (uint64_t)s * source_stride))[v];
                    m0 = fmaf(x.x, w, m0); m1 = fmaf(x.y, w, m1); m2 = fmaf(x.z, w, m2); m3 = fmaf(x.w, w, m3);
                }
                if constexpr (FOLD_F32)
                    ((float4 *)dst)[v] = float4{m0, m1, m2, m3};
                else
                    ((uint2 *)dst)[v] = make_uint2(LmPackBf16Pair(m0 / d, m1 / d), LmPackBf16Pair(m2 / d, m3 / d));
            }
            return;
        }
    }
    if constexpr (((ALIGN_MOD | STRIDE_MOD) % 2u) == 0u)
    {
        if ((((head_align | elem_begin | elem_end) & 1u) == 0u) && ((((uintptr_t)src0) & 7u) == 0u) && (sources == 1u || (source_stride & 1u) == 0u) && ((((uintptr_t)dst) & dst_align2) == 0u))
        {
            for (uint32_t v = (elem_begin >> 1u) + threadIdx.x; v < (elem_end >> 1u); v += step)
            {
                const uint32_t slot = slot_at(v << 1u);
                const float d = FOLD_F32 ? 1.0f : denom_at(slot);
                float m0 = 0.0f, m1 = 0.0f;
                if constexpr (FOLD_F32)
                    if (first == 0u) { const float2 p = ((const float2 *)dst)[v]; m0 = p.x * running_scale; m1 = p.y * running_scale; }
                for (uint32_t s = 0u; s < sources; ++s)
                {
                    const float w = scale_at(slot, s);
                    const float2 x = ((const float2 *)(src0 + (uint64_t)s * source_stride))[v];
                    m0 = fmaf(x.x, w, m0); m1 = fmaf(x.y, w, m1);
                }
                if constexpr (FOLD_F32)
                    ((float2 *)dst)[v] = make_float2(m0, m1);
                else
                    ((uint32_t *)dst)[v] = LmPackBf16Pair(m0 / d, m1 / d);
            }
            return;
        }
    }
    for (uint32_t e = elem_begin + threadIdx.x; e < elem_end; e += step)
    {
        const uint32_t slot = slot_at(e);
        float m = 0.0f;
        if constexpr (FOLD_F32)
            m = first != 0u ? 0.0f : ((const float *)dst)[e] * running_scale;
        for (uint32_t s = 0u; s < sources; ++s)
            m = fmaf(src0[(uint64_t)s * source_stride + e], scale_at(slot, s), m);
        if constexpr (FOLD_F32)
            ((float *)dst)[e] = m;
        else
            ((uint16_t *)dst)[e] = LmFloatToBf16(m / denom_at(slot));
    }
}

template<uint32_t ALIGN_MOD, uint32_t THREADS = 0u, uint32_t STRIDE_MOD = 0u, class SlotAt, class ScaleAt, class DenomAt>
static __device__ __forceinline__ void LmWeightedSourcesMergeRangeBf16(const float *__restrict__ src0, uint64_t source_stride, uint32_t sources, uint32_t elem_begin, uint32_t elem_end, uint16_t *__restrict__ out_ptr, SlotAt slot_at, ScaleAt scale_at, DenomAt denom_at, uint32_t head_align = 0u)
{
    LmWeightedSourcesApply<false, ALIGN_MOD, THREADS, STRIDE_MOD>(src0, source_stride, sources, elem_begin, elem_end, out_ptr, slot_at, scale_at, denom_at, head_align, 0.0f, 1u);
}

template<uint32_t WIDTH, uint32_t THREADS = 0u, uint32_t STRIDE_MOD = 0u>
static __device__ __forceinline__ void LmWeightedSourcesMergeBf16(const float *__restrict__ src0, uint64_t source_stride, uint32_t sources, const float *__restrict__ scales, float denom, uint16_t *__restrict__ out_ptr)
{
    LmWeightedSourcesApply<false, WIDTH, THREADS, STRIDE_MOD>(src0, source_stride, sources, 0u, WIDTH, out_ptr, [&](uint32_t) { return 0u; }, [&](uint32_t, uint32_t s) { return scales[s]; }, [&](uint32_t) { return denom; }, 0u, 0.0f, 1u);
}

template<uint32_t WIDTH, uint32_t THREADS = 0u, uint32_t STRIDE_MOD = 0u>
static __device__ __forceinline__ void LmWeightedSourcesFoldF32(const float *__restrict__ src0, uint64_t source_stride, uint32_t sources, const float *__restrict__ scales, float running_scale, uint32_t first, float *__restrict__ run_ptr)
{
    LmWeightedSourcesApply<true, WIDTH, THREADS, STRIDE_MOD>(src0, source_stride, sources, 0u, WIDTH, run_ptr, [&](uint32_t) { return 0u; }, [&](uint32_t, uint32_t s) { return scales[s]; }, [&](uint32_t) { return 1.0f; }, 0u, running_scale, first);
}
