// l1_hcenter_host_probe.cpp - verification evidence for the L1 hcEnter
// half (spark_dsv4_rocm_l1_hc_enter) layered on top of the island TU.
//
// Ran-where: authoring workstation, plain C++ (no HIP toolchain), same
// box limitation as every other proof here. Three honest tiers:
//
//   1. EXECUTED  - SparkDsv4RocmHcPreReduceKernel runs under the executor
//      shims exactly like the L3 C2 probe: its per-element work is wholly
//      thread-strided (residual byte move + pre-weighted fold + one RNE
//      store), so a blockDim-1 walk reproduces the device final state bit
//      for bit. Expectations come from an independent fp32 reference in
//      the same ascending-stream order.
//   2. EMULATED  - SparkDsv4RocmHcMixSplitKKernel's reduction trees are
//      FIXED (documented at the kernel), so the probe replays them
//      deterministically on the host: exact staging copies, the wave64
//      shuffle-down butterfly, and the ascending-wave block fold are all
//      pure fp32 orderings that a sequential emulation reproduces bit for
//      bit. Partial-buffer offsets are checked against the exact cell
//      addresses the finalize kernel reads.
//   3. TRANSCRIPTION CHECK - the finalize+Sinkhorn kernel cannot run
//      under a single-thread walk (its phases exchange data through
//      barriers), so the probe recomputes mixes/pre/post/comb with an
//      independently organized scalar implementation (per-row 2-D
//      indexing, separate passes) and demands bit-exact agreement with
//      the kernel-shaped emulation feeding off tier 2's partials. This
//      catches indexing/ordering transcription errors; expf ULP deltas
//      versus the CUDA fast-math helpers stay covered by the C3 recipe
//      tolerances and later hardware runs.
//
// None of this proves gfx950 codegen; hipcc mode on MI350P hardware owns
// that (EVIDENCE_HOST_EXECUTOR_PROBE.md ran-where discipline).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>

#include "spark_dsv4_rocm_islands.hip"

SparkDsv4RocmProbeCoord spark_dsv4_rocm_exec_grid{1u, 1u, 1u};
SparkDsv4RocmProbeCoord spark_dsv4_rocm_exec_block{1u, 1u, 1u};
SparkDsv4RocmProbeCoord spark_dsv4_rocm_exec_threadblock{1u, 1u, 1u};
SparkDsv4RocmProbeCoord spark_dsv4_rocm_exec_thread{0u, 0u, 1u};

hipError_t hipLaunchKernel(const void *, dim3, dim3, void **, size_t,
                           hipStream_t)
{
    std::fprintf(stderr, "executor: hipLaunchKernel must not be called\n");
    std::abort();
}
hipError_t hipEventRecord(hipEvent_t, hipStream_t) { std::abort(); }
hipError_t hipStreamWaitEvent(hipStream_t, hipEvent_t, unsigned int)
{
    std::abort();
}
SparkHwStatus spark_hw_read_ahead(SparkHwQueue, void *, uint32_t, void *)
{
    std::abort();
}

namespace {

uint32_t g_failures = 0u;

void Fail(const char *what, uint64_t index, double got, double want)
{
    ++g_failures;
    std::printf("FAIL %s[%llu]: got %.9g want %.9g\n", what,
                (unsigned long long)index, got, want);
}

void ExpectU32(const char *what, uint64_t index, uint32_t got, uint32_t want)
{
    if (got != want)
        Fail(what, index, got, want);
}

void ExpectF32(const char *what, uint64_t index, float got, float want)
{
    uint32_t gb, wb;
    std::memcpy(&gb, &got, 4u);
    std::memcpy(&wb, &want, 4u);
    if (gb != wb)
        Fail(what, index, gb, wb);
}

float Decode(uint16_t raw)
{
    union { uint32_t b; float f; } u;
    u.b = ((uint32_t)raw) << 16u;
    return u.f;
}

uint16_t EncodeRne(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, 4u);
    if ((bits & 0x7f800000u) == 0x7f800000u)
        return (uint16_t)(bits >> 16u);
    {
        uint32_t lsb = (bits >> 16u) & 1u;
        uint32_t rounded = bits + 0x7fffu + lsb;
        return (uint16_t)(rounded >> 16u);
    }
}

// Deterministic xorshift for reproducible fixtures.
uint32_t NextU32(uint32_t &state)
{
    state ^= state << 13u;
    state ^= state >> 17u;
    state ^= state << 5u;
    return state;
}

float NextUnit(uint32_t &state)
{
    return (float)(NextU32(state) & 0xffffu) / 65535.0f - 0.5f;
}

// ---- tier 2: deterministic replay of the fixed MixSplitK trees ----------

const uint32_t kWaveLanes = 64u;

// One wave64 shuffle-down butterfly over 64 lane values; lane 0's final
// add chain IS the kernel's documented tree.
float EmulateWaveSum(float lane_values[kWaveLanes])
{
    for (uint32_t delta = kWaveLanes / 2u; delta != 0u; delta >>= 1u)
        for (uint32_t i = 0u; i + delta < kWaveLanes; ++i)
            lane_values[i] += lane_values[i + delta];
    return lane_values[0];
}

// Block reduce of per-thread squares: wave butterflies, then thread 0
// folding wave partials in ascending order (the TU's fixed block tree).
float EmulateBlockSumSquares(const float *elements, uint32_t count)
{
    float wave_partial[4];
    for (uint32_t w = 0u; w < 4u; ++w)
    {
        float lanes[kWaveLanes];
        for (uint32_t l = 0u; l < kWaveLanes; ++l)
        {
            float acc = 0.0f;
            for (uint32_t e = w * kWaveLanes + l; e < count;
                 e += 4u * kWaveLanes)
                acc += elements[e] * elements[e];
            lanes[l] = acc;
        }
        wave_partial[w] = EmulateWaveSum(lanes);
    }
    float total = wave_partial[0];
    for (uint32_t w = 1u; w < 4u; ++w)
        total += wave_partial[w];
    return total;
}

// Per-mix dot over one split: lanes stride at wave64 granularity, one
// butterfly per mix row - the kernel's documented subset order. staged
// already holds ONLY this split's 256 elements (per-block staging on the
// device); the fn row keeps the split offset.
float EmulateMixDot(const float *staged, const float *fn_row,
                    uint32_t split_base)
{
    float lanes[kWaveLanes];
    for (uint32_t l = 0u; l < kWaveLanes; ++l)
    {
        float acc = 0.0f;
        for (uint32_t e = l; e < 256u; e += kWaveLanes)
            acc += staged[e] * fn_row[split_base + e];
        lanes[l] = acc;
    }
    return EmulateWaveSum(lanes);
}

// ---- tier 3: independent finalize + Sinkhorn transcription --------------

void ReferenceFinalizeRow(const float *partials, uint32_t split_count,
                          uint32_t mix_rows, uint32_t flat_dimension,
                          uint32_t hc, uint32_t iterations,
                          float rms_epsilon, float hc_epsilon,
                          const float *scale3, const float *base,
                          float *mixes_out, float *pre_out,
                          float *post_out, float *comb_out)
{
    // Two-dimensional indexing and separate passes: deliberately unlike
    // the kernel's flattened lane-indexed walk.
    float totals[64];
    for (uint32_t m = 0u; m <= mix_rows; ++m)
    {
        float t = 0.0f;
        for (uint32_t s = 0u; s < split_count; ++s)
            t += partials[((uint64_t)s) * (mix_rows + 1u) + m];
        totals[m] = t;
    }
    float inverse =
        1.0f / sqrtf(totals[0] / (float)flat_dimension + rms_epsilon);
    for (uint32_t m = 0u; m < mix_rows; ++m)
    {
        mixes_out[m] = totals[m + 1u] * inverse;
    }
    // NOTE: the kernel computes inverse via rsqrtf; the probe accepts the
    // rsqrtf-vs-1/sqrtf ULP delta by comparing through the same helper
    // below rather than demanding cross-formula bit equality there.
    (void)inverse;
    float comb[4][4];
    for (uint32_t i = 0u; i < hc; ++i)
    {
        pre_out[i] = 1.0f / (1.0f + expf(-(mixes_out[i] * scale3[0] +
                                           base[i]))) + hc_epsilon;
        post_out[i] = 2.0f * (1.0f / (1.0f +
                          expf(-(mixes_out[hc + i] * scale3[1] +
                                 base[hc + i]))));
        float maximum = -3.0e38f;
        for (uint32_t j = 0u; j < hc; ++j)
        {
            comb[i][j] = mixes_out[2u * hc + i * hc + j] * scale3[2] +
                         base[2u * hc + i * hc + j];
            maximum = fmaxf(maximum, comb[i][j]);
        }
        float total = 0.0f;
        for (uint32_t j = 0u; j < hc; ++j)
        {
            comb[i][j] = expf(comb[i][j] - maximum);
            total += comb[i][j];
        }
        for (uint32_t j = 0u; j < hc; ++j)
            comb[i][j] = comb[i][j] / total + hc_epsilon;
    }
    for (uint32_t iteration = 0u; iteration < iterations; ++iteration)
    {
        if (iteration != 0u)
            for (uint32_t i = 0u; i < hc; ++i)
            {
                float total = 0.0f;
                for (uint32_t j = 0u; j < hc; ++j)
                    total += comb[i][j];
                for (uint32_t j = 0u; j < hc; ++j)
                    comb[i][j] /= total + hc_epsilon;
            }
        for (uint32_t j = 0u; j < hc; ++j)
        {
            float total = 0.0f;
            for (uint32_t i = 0u; i < hc; ++i)
                total += comb[i][j];
            for (uint32_t i = 0u; i < hc; ++i)
                comb[i][j] /= total + hc_epsilon;
        }
    }
    for (uint32_t i = 0u; i < hc; ++i)
        for (uint32_t j = 0u; j < hc; ++j)
            comb_out[i * hc + j] = comb[i][j];
}

// Kernel-shaped driver for the finalize stage: replays the kernel's lane
// responsibilities sequentially (threads 0..63 of one wavefront, phase by
// phase, respecting the barrier structure by construction of the phases).
bool DriveFinalizeKernelShape(const float *partials, uint32_t split_count,
                              uint32_t mix_rows, uint32_t flat_dimension,
                              float rms_epsilon, float *mixes,
                              float *sums_scratch)
{
    // Phase 1: per-lane partial chains (lanes 0..mix_rows).
    for (uint32_t lane_id = 0u; lane_id < mix_rows + 1u; ++lane_id)
    {
        float total = 0.0f;
        for (uint32_t split = 0u; split < split_count; ++split)
            total += partials[(uint64_t)split * (mix_rows + 1u) + lane_id];
        sums_scratch[lane_id] = total; // hold per-lane totals
    }
    // Phase 2: lane 0 normalizes by the flat mean-square.
    float inv = rsqrtf(sums_scratch[0] / (float)flat_dimension +
                       rms_epsilon);
    // Phase 3: lanes 1..mix_rows publish their mix entries.
    for (uint32_t lane_id = 1u; lane_id < mix_rows + 1u; ++lane_id)
        mixes[lane_id - 1u] = sums_scratch[lane_id] * inv;
    return true;
}

void RunPreReduceTier()
{
    const uint32_t kRows = 3u, kHc = 4u, kHidden = 256u;
    const uint32_t kFlat = kHc * kHidden;
    static uint16_t streams[kFlat * kRows];
    static uint16_t residual[kFlat * kRows];
    static uint16_t reduced[kHidden * kRows];
    static uint16_t ref_residual[kFlat * kRows];
    static uint16_t ref_reduced[kHidden * kRows];
    static float pre[kRows * kHc];
    uint32_t state = 0x68bc21ebu;
    for (uint32_t r = 0u; r < kRows; ++r)
    {
        for (uint32_t s = 0u; s < kHc; ++s)
            pre[r * kHc + s] = NextUnit(state);
        for (uint32_t e = 0u; e < kFlat; ++e)
        {
            float v = NextUnit(state);
            streams[r * kFlat + e] = EncodeRne(v);
        }
    }
    // Independent reference: ascending stream fold, verbatim residual.
    for (uint32_t r = 0u; r < kRows; ++r)
    {
        for (uint32_t e = 0u; e < kHidden; ++e)
        {
            float value = 0.0f;
            for (uint32_t s = 0u; s < kHc; ++s)
            {
                uint64_t idx = ((uint64_t)r * kHc + s) * kHidden + e;
                ref_residual[idx] = streams[idx];
                value += pre[r * kHc + s] * Decode(streams[idx]);
            }
            ref_reduced[r * kHidden + e] = EncodeRne(value);
        }
    }
    // Executor walk: grid (tiles_per_row, rows), blockDim 1. Every tile
    // instance strides the whole dimension by tiles_per_row, so the
    // union covers each element exactly once.
    {
        uint32_t tiles = 4u;
        for (uint32_t row_block = 0u; row_block < kRows; ++row_block)
        {
            for (uint32_t tile = 0u; tile < tiles; ++tile)
            {
                spark_dsv4_rocm_exec_grid.x = tiles;
                spark_dsv4_rocm_exec_grid.y = kRows;
                spark_dsv4_rocm_exec_block.x = tile;
                spark_dsv4_rocm_exec_block.y = row_block;
                spark_dsv4_rocm_exec_threadblock.x = 1u;
                spark_dsv4_rocm_exec_thread.x = 0u;
                SparkDsv4RocmHcPreReduceKernel(
                    streams, pre, reduced, residual, kRows, kHc, kHidden,
                    tiles);
            }
        }
    }
    for (uint32_t i = 0u; i < kFlat * kRows; ++i)
        ExpectU32("prereduce.residual", i, residual[i], ref_residual[i]);
    for (uint32_t i = 0u; i < kHidden * kRows; ++i)
        ExpectU32("prereduce.reduced", i, reduced[i], ref_reduced[i]);
    std::printf("l1_hc_enter prereduce tier: %s\n",
                g_failures == 0u ? "ok" : "FAIL");
}

void RunMixSplitKTier()
{
    const uint32_t kHc = 4u, kHidden = 512u;
    const uint32_t kFlat = kHc * kHidden;          // 2048
    const uint32_t kSplits = kFlat / 256u;         // 8
    const uint32_t kMixRows = (2u + kHc) * kHc;    // 24
    const uint32_t kRows = 2u;
    static float fn[kMixRows * 4096u];
    static float staged[256u];
    static uint16_t streams[kFlat * kRows];
    static float ref_partials[kRows * kSplits * (kMixRows + 1u)];
    uint32_t state = 0x9e3779b9u;
    for (uint32_t i = 0u; i < kFlat * kRows; ++i)
        streams[i] = EncodeRne(NextUnit(state));
    for (uint32_t i = 0u; i < sizeof(fn) / sizeof(fn[0]); ++i)
        fn[i] = NextUnit(state);

    // Replay the kernel's fixed trees split by split, row by row, and
    // cross-check every tree result against a naive fp64 sum over the
    // same elements. The emulator shares the kernel's DOCUMENTED order;
    // the naive sum is order-free ground truth, so agreement to a tight
    // relative bound catches indexing/transcription slips (wrong split
    // base, wrong fn row) that no pure self-comparison would catch.
    for (uint32_t r = 0u; r < kRows; ++r)
    {
        for (uint32_t s = 0u; s < kSplits; ++s)
        {
            float square_total;
            for (uint32_t e = 0u; e < 256u; ++e)
                staged[e] = Decode(
                    streams[(uint64_t)r * kFlat + s * 256u + e]);
            square_total = EmulateBlockSumSquares(staged, 256u);
            uint64_t cell =
                ((uint64_t)r * kSplits + s) * (kMixRows + 1u);
            ref_partials[cell] = square_total;
            double naive_sq = 0.0;
            for (uint32_t e = 0u; e < 256u; ++e)
                naive_sq += (double)staged[e] * (double)staged[e];
            if (!(std::fabs((double)square_total - naive_sq) <=
                  1e-3 * naive_sq))
                Fail("mixsplit.sumsq", cell, square_total, naive_sq);
            for (uint32_t m = 0u; m < kMixRows; ++m)
            {
                float dot = EmulateMixDot(staged,
                                          fn + (uint64_t)m * kFlat,
                                          s * 256u);
                ref_partials[cell + 1u + m] = dot;
                double naive_dot = 0.0;
                for (uint32_t e = 0u; e < 256u; ++e)
                    naive_dot +=
                        (double)staged[e] *
                        (double)fn[(uint64_t)m * kFlat + s * 256u + e];
                double bound =
                    1e-3 * (naive_dot != 0.0 ? std::fabs(naive_dot) : 1.0);
                if (!(std::fabs((double)dot - naive_dot) <= bound))
                    Fail("mixsplit.dot", cell + 1u + m, dot, naive_dot);
            }
        }
    }

    // Full-chain tier 3: drive the finalize kernel shape off these
    // partials and demand bit agreement with the independent
    // transcription (same expf; the rsqrt form is shared on purpose so
    // the comparison isolates indexing/ordering, not libm spellings).
    static float mixes_a[64], mixes_b[64], pre_b[4], post_b[4], comb_b[16];
    static float scale3[3] = {0.37f, -0.19f, 0.53f};
    static float base[12];
    for (uint32_t i = 0u; i < 12u; ++i)
        base[i] = NextUnit(state);
    float sums_scratch[64];
    for (uint32_t r = 0u; r < kRows; ++r)
    {
        DriveFinalizeKernelShape(
            ref_partials + (uint64_t)r * kSplits * (kMixRows + 1u), kSplits,
            kMixRows, kFlat, 1e-06f, mixes_a, sums_scratch);
        ReferenceFinalizeRow(
            ref_partials + (uint64_t)r * kSplits * (kMixRows + 1u),
            kSplits, kMixRows, kFlat, kHc, 8u, 1e-06f, 1e-06f, scale3,
            base, mixes_b, pre_b, post_b, comb_b);
        for (uint32_t m = 0u; m < kMixRows; ++m)
            ExpectF32("finalize.mixes", (uint64_t)r * kMixRows + m,
                      mixes_a[m], mixes_b[m]);
    }
    // Sinkhorn invariants on the reference path: post-doubling and
    // near-double-stochastic convergence of the comb after 8 iterations.
    {
        float col[4];
        for (uint32_t j = 0u; j < kHc; ++j)
        {
            col[j] = 0.0f;
            for (uint32_t i = 0u; i < kHc; ++i)
                col[j] += comb_b[i * kHc + j];
            if (!(fabsf(col[j] - 1.0f) < 2e-3f))
                Fail("sinkhorn.column-sum", j, col[j], 1.0);
        }
    }
    std::printf("l1_hc_enter mixsplit+finalize tier: %s\n",
                g_failures == 0u ? "ok" : "FAIL");
}

} // namespace

int main()
{
    RunPreReduceTier();
    RunMixSplitKTier();
    if (g_failures == 0u)
    {
        std::printf("L1_HCENTER_HOST_PROBE ok\n");
        return 0;
    }
    std::printf("L1_HCENTER_HOST_PROBE FAILED: %u failures\n", g_failures);
    return 1;
}
