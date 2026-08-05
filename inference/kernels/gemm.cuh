#pragma once

// The GEMM. One body, one mma, every format, every model family.
//
// A dense linear is a grouped GEMM with one group, and every weight format is a
// decoder into the same BF16 fragment. Those two facts are why this file is one
// kernel where the old tree had seven - a CUTLASS dense call, a CUTLASS grouped
// call, generated B12x kernels, a reference, two BF16 tile policies behind a
// macro, and an FP8 tile excluded from every build. Four of the seven never
// executed.
//
// WHY EVERYTHING DECODES TO BF16. GB10's BF16 ridge is 573 FLOP/byte. Decode
// arithmetic intensity is 8 at B128 and 64 at B1024, so the kernel needs between
// 1.4 and 11 percent of BF16 peak. Unpacking INT7 costs about 2 percent of the
// CUDA cores. Compute is free by roughly sixty times, and the only number that
// matters is bytes crossing the bus.
//
// So a format's job is to be narrow in memory and to hand back a BF16 register.
// Nothing else varies. There is no integer accumulator, no block-scaled mma
// variant, no rescale pass, and no second code path - all of which existed here
// two commits ago and all of which were buying arithmetic on a machine with
// arithmetic to spare.
//
// WHAT A FORMAT MUST PROVIDE. kStoredBits, the four fragment coordinate helpers,
// and Fragment(). That is the whole interface, and it is why an entropy coder
// drops in as a replacement for one function rather than as a second kernel.
//
// TILE HEIGHT. Rows per group at decode is batch * top_k / experts, so it grows
// with the batch while the tile height is compile-time. Each M tile re-reads its
// group's weight tile, so the moment rows exceed TILE_M every group splits and
// the weight stream doubles. TILE_M is a template parameter selected per token
// bucket, and rounding is one-directional: up wastes mma throughput on padded
// rows, which is free here; down costs bandwidth, which is not.

#include "inference/kernels/formats/bf16.cuh"
#include "inference/kernels/scale.cuh"
#include "inference/kernels/tile.cuh"
#include <cuda.h>
#include <stdint.h>

// Bytes the launcher must request and pass. Constexpr so a host can compute it
// without a device query and so it cannot drift from what the kernel carves.
template<class FormatA, class FormatB, uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t STAGES>
static __host__ __device__ constexpr uint32_t LmGemmSharedBytes(void)
{
	return((STAGES * LmTileBytes(TILE_M,TILE_K,FormatA::kStoredBits))
		+ (STAGES * LmTileBytes(TILE_N,TILE_K,FormatB::kStoredBits))
		+ (STAGES * 8u));
}

struct LmGemmArguments
{
	LmScaleTensor scale_a;
	LmScaleTensor scale_b;
	// A caller that pre-priced the grouped tile table for this launch's tile
	// heights sets this; the launcher then makes no prefix launch. The dense
	// case never needs either - the kernel derives one group's two values.
	uint32_t prefix_built;
	const uint32_t *group_row_offset;
	// FILLED BY THE LAUNCHER, NOT THE CALLER. Tile counts depend on this
	// launch's tile height and neuron width, so one caller-built prefix cannot
	// serve two GEMMs of different output widths - and it was serving twenty.
	// The caller provides group_count + 1 words of device scratch; the launcher
	// prices them for the launch it is about to make.
	uint32_t *group_tile_prefix;
	const void *activation_bytes;
	const uint32_t *source_row_map;
	uint32_t source_row_count;
	void *output_bf16;
	void *output_f32;
	// Optional second life for the result: the epilogue ADDS it into this
	// buffer (read-modify-write; single writer per element by tile
	// ownership, stream-ordered against other producers). accumulate ==
	// output means the output itself accumulates. This is how a module's
	// final projection folds the AttnRes partial add - and the shared
	// expert's sum - into the store it was already making: one fewer
	// kernel, one fewer full-width read, per fused add.
	void *accumulate_bf16;
	// Logical columns produced by this GEMM may occupy a slice of a wider
	// row. A zero row stride means tightly packed output_dimension columns.
	// The column offset is applied to output, output_f32 and accumulate alike.
	uint32_t output_row_stride;
	uint32_t output_column_offset;
	uint32_t group_count;
	uint32_t input_dimension;
	uint32_t output_dimension;
};

// Accumulate one staged K tile.
//
// The scale is fetched once per fragment rather than once per element: a
// fragment spans two adjacent k, and the scale group is at least 16, so both
// halves always share it. That is asserted rather than assumed.
//
// Formats that dequantise for free hand back raw BF16 bit patterns holding
// code + bias, and the correction is applied here with the multiply that had to
// happen anyway - (v - bias) * scale becomes one fma against a precomputed
// -bias*scale. Formats already in a real numeric form skip it. Both branches are
// on a compile-time constant, so only one exists in any instantiation.
template<class FormatA, class FormatB, uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t WARPS>
static __device__ void LmGemmConsume(
    float (*total)[4],
    const uint8_t *stage_a,
    const uint8_t *stage_b,
    const LmScaleTensor &scale_a,
    const LmScaleTensor &scale_b,
    uint32_t group_index,
    uint32_t row_base,
    uint32_t neuron_base,
    uint32_t global_k_base,
    uint32_t warp,
    uint32_t lane)
{
    static_assert(
        FormatA::kMmaM == FormatB::kMmaM &&
        FormatA::kMmaN == FormatB::kMmaN &&
        FormatA::kMmaK == FormatB::kMmaK,
        "both operands must decode into the same MMA geometry");
    static_assert(
        FormatA::kScaleGroup == 0u || FormatA::kScaleGroup >= 2u,
        "an activation scale group below two splits a fragment");
    static_assert(
        FormatB::kScaleGroup == 0u || FormatB::kScaleGroup >= 2u,
        "a weight scale group below two splits a fragment");
    const uint32_t m_frags = TILE_M / FormatA::kMmaM;
    const uint32_t n_frags = TILE_N / WARPS / FormatA::kMmaN;
    const uint32_t pitch_a = LmTileBytes(1u, TILE_K, FormatA::kStoredBits);
    const uint32_t pitch_b = LmTileBytes(1u, TILE_K, FormatB::kStoredBits);
    const uint32_t steps = TILE_K / FormatA::kMmaK;
    uint32_t step, mi, ni, neuron, k_base, reg;
    uint32_t a[4], b[2];

    for (step = 0u; step < steps; ++step)
    {
        k_base = step * FormatA::kMmaK;
        for (mi = 0u; mi < m_frags; ++mi)
        {
            for (reg = 0u; reg < 4u; ++reg)
            {
                const uint32_t local_row =
                    (mi * FormatA::kMmaM) + FormatA::OperandARow(lane, reg);
                const uint32_t local_k = k_base + FormatA::OperandAK(lane, reg);
                const float scale = LmScaleTensorLoad(
                    &scale_a,
                    0u,
                    row_base + local_row,
                    global_k_base + local_k);
                a[reg] = FormatA::Fragment(
                    stage_a,
                    local_row,
                    local_k,
                    pitch_a,
                    scale);
            }
            for (ni = 0u; ni < n_frags; ++ni)
            {
                neuron =
                    (warp * (TILE_N / WARPS)) + (ni * FormatA::kMmaN);
                for (reg = 0u; reg < 2u; ++reg)
                {
                    const uint32_t local_row =
                        neuron + FormatB::OperandBRow(lane);
                    const uint32_t local_k =
                        k_base + FormatB::OperandBK(lane, reg);
                    const float scale = LmScaleTensorLoad(
                        &scale_b,
                        group_index,
                        neuron_base + local_row,
                        global_k_base + local_k);
                    b[reg] = FormatB::Fragment(
                        stage_b,
                        local_row,
                        local_k,
                        pitch_b,
                        scale);
                }
                LmMmaBf16(total[(mi * n_frags) + ni], a, b);
            }
        }
    }
}

// Rows past a group's own count are dropped rather than written. The tile height
// covers the busiest group, so most groups have a ragged tail and this is the
// steady state, not an edge case.
template<class Format, uint32_t TILE_M, uint32_t TILE_N, uint32_t WARPS>
static __device__ void LmGemmStore(const LmGemmArguments &args, float (*total)[4], uint32_t count, uint32_t row_base, uint32_t row_limit, uint32_t neuron_base, uint32_t warp, uint32_t lane)
{
	const uint32_t n_frags = TILE_N / WARPS / Format::kMmaN;
	uint16_t *output = (uint16_t *)args.output_bf16;
	float *output_f32 = (float *)args.output_f32;
	uint16_t *accumulate = (uint16_t *)args.accumulate_bf16;
	uint64_t at;
	uint32_t i,e,mi,ni,row,column;
	for (i = 0u; i < count; ++i)
	{
		mi = i / n_frags;
		ni = i % n_frags;
		for (e = 0u; e < 4u; ++e)
		{
			row = row_base + (mi * Format::kMmaM) + LmMmaAccumulatorRow(lane,e);
			column = neuron_base + (warp * (TILE_N / WARPS)) + (ni * Format::kMmaN)
				+ LmMmaAccumulatorColumn(lane,e);
			if ( row >= row_limit || column >= args.output_dimension )
				continue;
			at = ((uint64_t)row * args.output_row_stride) +
				args.output_column_offset + column;
			if ( accumulate != 0 )
			{
				float sum = LmBf16ToFloat(accumulate[at]) + total[i][e];
				accumulate[at] = LmFloatToBf16(sum);
				if ( accumulate == output )
					continue;
			}
			if ( output_f32 != 0 )
				output_f32[at] = total[i][e];
			else
				output[at] = LmFloatToBf16(total[i][e]);
		}
	}
}

static __device__ __forceinline__ void LmGemmZero(float (*acc)[4], uint32_t count)
{
	uint32_t i,e;
	for (i = 0u; i < count; ++i)
		for (e = 0u; e < 4u; ++e)
			acc[i][e] = 0.0f;
}

// The persistent grid walks tiles strided by gridDim, so it is sized to the
// machine rather than the problem and a short group never leaves an SM idle
// behind a long one. The tile total comes from the device-side prefix, never a
// host estimate: an estimate from average rows launches tiles for groups that
// received none, and each still streams a full weight tile before its stores
// are rejected.
//
// Not static: a model's unity.cu names this in an explicit instantiation, and
// explicit instantiation of an internal symbol is ill-formed.
template<class FormatA, class FormatB, uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t STAGES, uint32_t WARPS, bool INDIRECT_A = false, uint32_t ACTIVATION_CODEC = SPARK_ACTIVATION_CODEC_NONE>
__global__ __launch_bounds__(WARPS * LM_WARP_LANES, 1)
void LmGemmKernel(
    __grid_constant__ const LmGemmArguments args,
    __grid_constant__ const CUtensorMap tensor_map_a,
    __grid_constant__ const CUtensorMap tensor_map_b,
    LmTileGeometry geometry_a,
    LmTileGeometry geometry_b,
    bool grouped)
{
    static_assert(
        LmTileKIsTmaLoadable(TILE_K,FormatA::kStoredBits,FormatA::kTmaSwizzle),
        "the activation row pitch is not TMA-loadable");
    static_assert(
        LmTileKIsTmaLoadable(TILE_K,FormatB::kStoredBits,FormatB::kTmaSwizzle),
        "the weight row pitch is not TMA-loadable");
    static_assert(
        LmPipelineSharedBytesSplit(
            TILE_M,
            TILE_N,
            TILE_K,
            STAGES,
            FormatA::kStoredBits,
            FormatB::kStoredBits) <= LM_SMEM_SM_TOTAL,
        "tile exceeds the shared memory an SM has");
    static_assert(
        FormatA::kMmaM == FormatB::kMmaM &&
        FormatA::kMmaN == FormatB::kMmaN &&
        FormatA::kMmaK == FormatB::kMmaK,
        "both operands must decode into the same MMA geometry");
    static_assert(
        TILE_M % FormatA::kMmaM == 0u &&
        TILE_N % (WARPS * FormatA::kMmaN) == 0u,
        "tile is not a whole number of MMA fragments");
    static_assert(
        TILE_K % FormatA::kMmaK == 0u,
        "tile depth is not a whole number of MMA steps");
	static_assert(ACTIVATION_CODEC <= SPARK_ACTIVATION_CODEC_FP8_E4M3_UE8M0,
		"unknown activation codec");
	static_assert((!INDIRECT_A && ACTIVATION_CODEC == SPARK_ACTIVATION_CODEC_NONE) || FormatA::kScaleGroup == 0u,
		"manual activation staging requires unscaled stored activations");
    extern __shared__ __align__(LM_TMA_ALIGNMENT_BYTES) uint8_t lm_shared[];
    const uint32_t a_bytes =
        LmTileBytes(TILE_M, TILE_K, FormatA::kStoredBits);
    const uint32_t b_bytes =
        LmTileBytes(TILE_N, TILE_K, FormatB::kStoredBits);
    uint8_t (*stage_a)[LmTileBytes(TILE_M, TILE_K, FormatA::kStoredBits)] =
        (uint8_t (*)[LmTileBytes(TILE_M, TILE_K, FormatA::kStoredBits)])
            lm_shared;
    uint8_t (*stage_b)[LmTileBytes(TILE_N, TILE_K, FormatB::kStoredBits)] =
        (uint8_t (*)[LmTileBytes(TILE_N, TILE_K, FormatB::kStoredBits)])
            (lm_shared + (STAGES * a_bytes));
    uint64_t *barrier =
        (uint64_t *)(lm_shared + (STAGES * (a_bytes + b_bytes)));
    const uint32_t count =
        (TILE_M / FormatA::kMmaM) *
        (TILE_N / WARPS / FormatA::kMmaN);
    const uint32_t neuron_tiles =
        (args.output_dimension + TILE_N - 1u) / TILE_N;
    const uint32_t k_tiles = args.input_dimension / TILE_K;
    const uint32_t dense_rows = args.group_count == 1u
        ? args.group_row_offset[1] - args.group_row_offset[0]
        : 0u;
    const uint32_t dense_tiles = args.group_count == 1u
        ? ((dense_rows + TILE_M - 1u) / TILE_M) * neuron_tiles
        : 0u;
    float total[
        (TILE_M / FormatA::kMmaM) *
        (TILE_N / WARPS / FormatA::kMmaN)][4];
    const uint32_t warp = threadIdx.x / LM_WARP_LANES;
    const uint32_t lane = threadIdx.x % LM_WARP_LANES;
    uint32_t tile, group, in_group, row_base, row_limit;
    uint32_t neuron_base, k, stage, ahead, total_tiles;
    // One wait-parity bit per stage, carried across output tiles: the barriers
    // initialise once, so a stage's phase accumulates over every tile this CTA
    // runs and cannot be derived from k - (k / STAGES) & 1 is exact only when
    // k_tiles % (2 * STAGES) == 0, and silently waits on a stale phase
    // otherwise.
    uint32_t phase;

    LmPipelineInitialise<STAGES>(barrier, 1u);
    phase = 0u;
    total_tiles = args.group_count == 1u
        ? dense_tiles
        : LmTotalTiles(args.group_tile_prefix, args.group_count);
    for (tile = blockIdx.x; tile < total_tiles; tile += gridDim.x)
    {
        group = args.group_count == 1u
            ? 0u
            : LmGroupOfTile(args.group_tile_prefix, args.group_count, tile);
        in_group = args.group_count == 1u
            ? tile
            : tile - args.group_tile_prefix[group];
        row_base =
            args.group_row_offset[group] +
            ((in_group / neuron_tiles) * TILE_M);
        row_limit = args.group_row_offset[group + 1u];
        neuron_base = (in_group % neuron_tiles) * TILE_N;
        LmGemmZero(total, count);
        for (stage = 0u;
             stage + 1u < STAGES && stage < k_tiles;
             ++stage)
        {
				if constexpr ( INDIRECT_A || ACTIVATION_CODEC != SPARK_ACTIVATION_CODEC_NONE )
					LmPipelineProduceManualA<FormatA,TILE_M,TILE_K,ACTIVATION_CODEC>(&geometry_a,&geometry_b,&tensor_map_b,
						(const uint8_t *)args.activation_bytes,args.source_row_map,
						stage_a[stage],stage_b[stage],&barrier[stage],row_base,row_limit,
						INDIRECT_A ? args.source_row_count : args.group_row_offset[args.group_count],
						args.input_dimension,neuron_base,stage,group,grouped);
			else
				LmPipelineProduce(&geometry_a,&geometry_b,&tensor_map_a,&tensor_map_b,
					stage_a[stage],stage_b[stage],&barrier[stage],row_base,neuron_base,
					stage,group,grouped);
        }
        for (k = 0u; k < k_tiles; ++k)
        {
            stage = LmPipelineStage(k, STAGES);
            ahead = LmPipelineAhead(k, STAGES);
            if (ahead < k_tiles)
            {
				if constexpr ( INDIRECT_A || ACTIVATION_CODEC != SPARK_ACTIVATION_CODEC_NONE )
					LmPipelineProduceManualA<FormatA,TILE_M,TILE_K,ACTIVATION_CODEC>(&geometry_a,&geometry_b,&tensor_map_b,
						(const uint8_t *)args.activation_bytes,args.source_row_map,
						stage_a[ahead % STAGES],stage_b[ahead % STAGES],&barrier[ahead % STAGES],
						row_base,row_limit,INDIRECT_A ? args.source_row_count : args.group_row_offset[args.group_count],args.input_dimension,
						neuron_base,ahead,group,grouped);
				else
					LmPipelineProduce(&geometry_a,&geometry_b,&tensor_map_a,&tensor_map_b,
						stage_a[ahead % STAGES],stage_b[ahead % STAGES],&barrier[ahead % STAGES],
						row_base,neuron_base,ahead,group,grouped);
            }
            LmMbarrierWait(
                &barrier[stage],
                (phase >> stage) & 1u);
			if constexpr ( INDIRECT_A || ACTIVATION_CODEC != SPARK_ACTIVATION_CODEC_NONE )
				__syncthreads();
            phase ^= 1u << stage;
            LmGemmConsume<
                FormatA,
                FormatB,
                TILE_M,
                TILE_N,
                TILE_K,
                WARPS>(
                    total,
                    stage_a[stage],
                    stage_b[stage],
                    args.scale_a,
                    args.scale_b,
                    group,
                    row_base,
                    neuron_base,
                    k * TILE_K,
                    warp,
                    lane);
            __syncthreads();
        }
        LmGemmStore<FormatA, TILE_M, TILE_N, WARPS>(
            args,
            total,
            count,
            row_base,
            row_limit,
            neuron_base,
            warp,
            lane);
        __syncthreads();
    }
    LmPipelineRelease<STAGES>(barrier);
}

// Price this launch's tiles from the row offsets: tiles for group g are
// ceil(rows_g / tile_m) * neuron_tiles, prefixed. Serial over the groups by
// one thread - 897 additions is not a scan problem - and launched by the GEMM
// launcher on the same stream, so the GEMM's binary search reads a prefix
// built for exactly its own geometry.
template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmGemmTilePrefixKernel(const uint32_t *__restrict__ group_row_offset, uint32_t group_count, uint32_t tile_m, uint32_t neuron_tiles, uint32_t *__restrict__ tile_prefix)
{
	uint32_t index,rows,total = 0u;
	if ( threadIdx.x != 0u || blockIdx.x != 0u )
		return;
	for (index = 0u; index < group_count; ++index)
	{
		rows = group_row_offset[index + 1u] - group_row_offset[index];
		tile_prefix[index] = total;
		total += ((rows + tile_m - 1u) / tile_m) * neuron_tiles;
	}
	tile_prefix[group_count] = total;
}
