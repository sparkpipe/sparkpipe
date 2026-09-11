#pragma once


#include "inference/kernels/formats/bf16.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/scale.cuh"
#include "inference/kernels/tile.cuh"
#include <cuda.h>
#include <stdint.h>

template<class FormatA, class FormatB, uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t STAGES, bool INTERLEAVED_B = false>
static __host__ __device__ constexpr uint32_t LmGemmSharedBytes(void)
{
	constexpr uint32_t b_bytes = INTERLEAVED_B
		? (17u * (TILE_N / 16u) * (TILE_K / 2u))
		: LmTileBytes(TILE_N,TILE_K,FormatB::kStoredBits);
	return((STAGES * LmTileBytes(TILE_M,TILE_K,FormatA::kStoredBits))
		+ (STAGES * b_bytes)
		+ (STAGES * 8u));
}

struct LmGemmArguments
{
	LmScaleTensor scale_a;
	LmScaleTensor scale_b;
	uint32_t prefix_built;
	const uint32_t *group_row_offset;
	uint32_t *group_tile_prefix;
	const void *activation_bytes;
	const uint32_t *source_row_map;
	uint32_t source_row_count;
	void *output_bf16;
	void *output_f32;
	void *accumulate_bf16;
	uint32_t output_row_stride;
	uint32_t output_column_offset;
	uint32_t group_count;
	uint32_t input_dimension;
	uint32_t output_dimension;
	LmFrameError *frame_error;
};

template<class FormatA, class FormatB, uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t WARPS, bool INTERLEAVED_B = false>
static __device__ void LmGemmConsume(
    float (*total)[4],
    const uint8_t *stage_a,
    const uint8_t *stage_b,
    const LmScaleTensor &scale_a,
    const LmScaleTensor &scale_b,
    const uint32_t *source_row_map,
    uint32_t group_index,
    uint32_t row_base,
    uint32_t row_limit,
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
    const uint32_t pitch_b = INTERLEAVED_B
        ? (TILE_K / 2u)
        : LmTileBytes(1u, TILE_K, FormatB::kStoredBits);
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
                uint32_t scale_row = row_base + local_row;
                if ( source_row_map != 0 &&
                    scale_a.encoding != LM_SCALE_ENCODING_NONE )
                {
                    if ( scale_row >= row_limit )
                        scale_row = row_base;
                    scale_row = LmRouteSourceRow(source_row_map, scale_row);
                }
                const float scale = LmScaleTensorLoad(
                    &scale_a,
                    0u,
                    scale_row,
                    global_k_base + local_k);
                if constexpr ( INTERLEAVED_B && TILE_K == 128u )
                {
                    a[reg] = FormatA::Fragment(
                        stage_a + ((local_k >= 64u) ? (TILE_M * 128u) : 0u),
                        local_row,
                        local_k & 63u,
                        128u,
                        scale);
                }
                else
                {
                    a[reg] = FormatA::Fragment(
                        stage_a,
                        local_row,
                        local_k,
                        pitch_a,
                        scale);
                }
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
                    if constexpr ( INTERLEAVED_B )
                    {
                        const uint32_t cell = local_row / 16u;
                        const uint32_t r16 = local_row % 16u;
                        const uint32_t scale_byte = (TILE_K == 128u)
                            ? LmSwizzledOffset((cell * 17u) + 16u,
                                (r16 * 4u) + (local_k / 32u),
                                pitch_b, 64u)
                            : ((cell * 17u) + 16u) * pitch_b + r16;
                        const float scale = FormatB::ScaleDecode(
                            stage_b[scale_byte]);
                        b[reg] = FormatB::Fragment(
                            stage_b,
                            (cell * 17u) + r16,
                            local_k,
                            pitch_b,
                            scale);
                    }
                    else
                    {
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
                }
                LmMmaBf16(total[(mi * n_frags) + ni], a, b);
            }
        }
    }
}

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

template<class FormatA, uint32_t TILE_M, uint32_t TILE_K, bool INDIRECT_A, uint32_t ACTIVATION_CODEC, bool INTERLEAVED_B = false>
static __device__ __forceinline__ void LmGemmProduce(
    const LmGemmArguments &args,
    const LmTileGeometry *geometry_a,
    const LmTileGeometry *geometry_b,
    const void *tensor_map_a,
    const void *tensor_map_b,
    void *stage_a,
    void *stage_b,
    uint64_t *barrier,
    uint32_t row_base,
    uint32_t row_limit,
    uint32_t neuron_base,
    uint32_t k_tile,
    uint32_t group,
    bool grouped)
{
    if constexpr ( ACTIVATION_CODEC != SPARK_ACTIVATION_CODEC_NONE )
    {
        static_assert(!INTERLEAVED_B,
            "the activation codec and interleaved B do not combine");
        LmPipelineProduceManualA<FormatA,TILE_M,TILE_K,ACTIVATION_CODEC>(
            geometry_a,geometry_b,tensor_map_b,
            (const uint8_t *)args.activation_bytes,args.source_row_map,
            stage_a,stage_b,barrier,row_base,row_limit,
            INDIRECT_A ? args.source_row_count : args.group_row_offset[args.group_count],
            args.input_dimension,neuron_base,k_tile,group,grouped,
            args.frame_error);
        return;
    }
    if constexpr ( INDIRECT_A )
    {
        LmPipelineProduceIndirectA<FormatA,TILE_K,INTERLEAVED_B>(
            geometry_a,geometry_b,tensor_map_b,
            (const uint8_t *)args.activation_bytes,args.source_row_map,
            args.source_row_count,args.input_dimension,
            stage_a,stage_b,barrier,row_base,row_limit,
            neuron_base,k_tile,group,grouped,
            args.output_dimension / 16u,
            args.frame_error);
        return;
    }
    LmPipelineProduce<TILE_K>(
        geometry_a,geometry_b,tensor_map_a,tensor_map_b,
        stage_a,stage_b,barrier,row_base,neuron_base,k_tile,group,grouped,
        INTERLEAVED_B,args.output_dimension / 16u);
}

template<class FormatA, class FormatB, uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t STAGES, uint32_t WARPS, bool INDIRECT_A = false, uint32_t ACTIVATION_CODEC = SPARK_ACTIVATION_CODEC_NONE, bool INTERLEAVED_B = false>
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
        !INTERLEAVED_B || LmTileKIsTmaLoadable(TILE_K / 2u,8u,TILE_K == 128u),
        "the interleaved cell row is not TMA-loadable");
    static_assert(
        INTERLEAVED_B || LmTileKIsTmaLoadable(TILE_K,FormatB::kStoredBits,FormatB::kTmaSwizzle),
        "the weight row pitch is not TMA-loadable");
    static_assert(!INTERLEAVED_B || TILE_K == 128u || TILE_K == 32u,
        "interleaved B stages one pack k-tile (128 or 32 elements) per box");
    static_assert(!INTERLEAVED_B || (TILE_N % 16u) == 0u,
        "interleaved B tiles a whole number of 16-neuron cells");
    static_assert(!INTERLEAVED_B || (TILE_M % 8u) == 0u,
        "the two-block A stage needs TILE_M sectors-aligned rows");
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
    constexpr uint32_t b_stride = INTERLEAVED_B
        ? (17u * (TILE_N / 16u) * (TILE_K / 2u))
        : LmTileBytes(TILE_N, TILE_K, FormatB::kStoredBits);
    uint8_t (*stage_a)[LmTileBytes(TILE_M, TILE_K, FormatA::kStoredBits)] =
        (uint8_t (*)[LmTileBytes(TILE_M, TILE_K, FormatA::kStoredBits)])
            lm_shared;
    uint8_t *stage_b_base = lm_shared + (STAGES * a_bytes);
    uint64_t *barrier =
        (uint64_t *)(lm_shared + (STAGES * (a_bytes + b_stride)));
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
            LmGemmProduce<FormatA,TILE_M,TILE_K,INDIRECT_A,ACTIVATION_CODEC,INTERLEAVED_B>(
                args,
                &geometry_a,
                &geometry_b,
                &tensor_map_a,
                &tensor_map_b,
                stage_a[stage],
                stage_b_base + (stage * b_stride),
                &barrier[stage],
                row_base,
                row_limit,
                neuron_base,
                stage,
                group,
                grouped);
        }
        for (k = 0u; k < k_tiles; ++k)
        {
            stage = LmPipelineStage(k, STAGES);
            ahead = LmPipelineAhead(k, STAGES);
            if (ahead < k_tiles)
            {
                LmGemmProduce<FormatA,TILE_M,TILE_K,INDIRECT_A,ACTIVATION_CODEC,INTERLEAVED_B>(
                    args,
                    &geometry_a,
                    &geometry_b,
                    &tensor_map_a,
                    &tensor_map_b,
                    stage_a[ahead % STAGES],
                    stage_b_base + ((ahead % STAGES) * b_stride),
                    &barrier[ahead % STAGES],
                    row_base,
                    row_limit,
                    neuron_base,
                    ahead,
                    group,
                    grouped);
            }
            LmMbarrierWait(
                &barrier[stage],
                (phase >> stage) & 1u);
			if constexpr ( ACTIVATION_CODEC != SPARK_ACTIVATION_CODEC_NONE )
				__syncthreads();
            phase ^= 1u << stage;
            LmGemmConsume<
                FormatA,
                FormatB,
                TILE_M,
                TILE_N,
                TILE_K,
                WARPS,
                INTERLEAVED_B>(
                    total,
                    stage_a[stage],
                    stage_b_base + (stage * b_stride),
                    args.scale_a,
                    args.scale_b,
                    args.source_row_map,
                    group,
                    row_base,
                    row_limit,
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
