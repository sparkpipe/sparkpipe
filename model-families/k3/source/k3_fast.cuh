#pragma once

#include <cstddef>
#include <stdint.h>
#include "inference/kernels/cell_mma.cuh"
#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "inference/kernels/topk_warp.cuh"
#include "inference/kernels/weight_prefetch.cuh"
#include "model-families/k3/source/config.h"
#include "runtime/launch.h"
#include "sparkpipe/spark_gpu.h"
#include "skl2_k3_embedded.h"

struct K3Skl2BundleTag {};

template<typename... Args>
static inline SparkGpuError K3Skl2Launch(uint32_t id, dim3 grid, dim3 block, size_t shared, SparkGpuStream stream, Args... args)
{
	return LmLaunchEmbedded<K3Skl2BundleTag>(kK3Skl2Cubin, sizeof(kK3Skl2Cubin), (const char *)kK3Skl2Manifest,
		sizeof(kK3Skl2Manifest) - 1u, kK3Skl2Names, (uint32_t)K3SKL2_COUNT, id, grid, block, shared, stream, args...);
}

#define K3_ATTNRES_PERSISTENT_BLOCKS 96u

static int32_t K3MoeCellsMmaFastLaunch(const void *w1_weight, const void *w2_weight,
	const uint16_t *latent_full_bf16, uint16_t *gate_up_bf16, uint16_t *intermediate_bf16, uint16_t *latent_bf16,
	const uint32_t *group_row_offset, const uint32_t *route_source_token, const uint32_t *route_packed_row,
	const float *route_weight, const uint32_t *active_experts, uint32_t rows, uint32_t packed_rows,
	uint32_t channels, SparkGpuStream stream)
{
	const uint32_t w1_k_tiles = K3_ROUTED_EXPERT_HIDDEN / LM_SKINNY_CELL_TILE_K;
	const uint32_t half_cells = channels / LM_SKINNY_CELL_NEURONS;
	const uint32_t w1_col_blocks = half_cells / 4u;
	const uint32_t max_active = packed_rows < K3_EXPERTS ? packed_rows : K3_EXPERTS;
	const uint32_t w1_group_bytes = w1_k_tiles * (half_cells * 2u) * LM_CELL_MMA_CELL_BYTES;
	if ( K3Skl2Launch(packed_rows <= 1024u ? K3SKL2_k3_cell_mma_w1_situ_c2 : K3SKL2_k3_cell_mma_w1_situ_c4,
		max_active * w1_col_blocks, LM_CELL_MMA_THREADS, 0u, stream,
		(const uint8_t *)w1_weight, latent_full_bf16, intermediate_bf16, group_row_offset, route_source_token,
		active_experts, w1_group_bytes, w1_k_tiles, half_cells, (uint32_t)K3_ROUTED_EXPERT_HIDDEN, channels,
		(float)K3_SITU_BETA, (float)K3_SITU_LINEAR_BETA) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	const uint32_t w2_k_tiles = channels / LM_SKINNY_CELL_TILE_K;
	const uint32_t w2_total_cells = K3_ROUTED_EXPERT_HIDDEN / LM_SKINNY_CELL_NEURONS;
	const uint32_t w2_group_bytes = w2_k_tiles * w2_total_cells * LM_CELL_MMA_CELL_BYTES;
	const uint64_t full_out_bytes = (uint64_t)packed_rows * K3_ROUTED_EXPERT_HIDDEN * sizeof(uint16_t);
	uint32_t slice_cells = w2_total_cells;
	if ( full_out_bytes > 16u * 1024u * 1024u )
	{
		const uint64_t target_bytes = 10u * 1024u * 1024u;
		uint32_t raw_cells = (uint32_t)(target_bytes / ((uint64_t)packed_rows * LM_SKINNY_CELL_NEURONS * sizeof(uint16_t)));
		raw_cells = (raw_cells / LM_CELL_MMA_BLOCK_CELLS) * LM_CELL_MMA_BLOCK_CELLS;
		if ( raw_cells < LM_CELL_MMA_BLOCK_CELLS )
			raw_cells = LM_CELL_MMA_BLOCK_CELLS;
		if ( raw_cells > w2_total_cells )
			raw_cells = w2_total_cells;
		slice_cells = raw_cells;
	}
	if ( slice_cells == w2_total_cells )
	{
		const uint32_t w2_col_blocks = w2_total_cells / LM_CELL_MMA_BLOCK_CELLS;
		if ( K3Skl2Launch(packed_rows <= 2048u ? K3SKL2_k3_cell_mma_w2_slice_c2 : K3SKL2_k3_cell_mma_w2_slice_c4_stream,
			max_active * w2_col_blocks, LM_CELL_MMA_THREADS, 0u, stream,
			(const uint8_t *)w2_weight, intermediate_bf16, gate_up_bf16, group_row_offset, active_experts,
			w2_group_bytes, w2_k_tiles, w2_total_cells, 0u, w2_total_cells, channels) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		const SparkGpuError err = K3Skl2Launch(K3SKL2_k3_moe_finalize_token224_224, rows, 224u, 0u, stream,
			gate_up_bf16, route_packed_row, route_weight, latent_bf16, rows, (uint32_t)K3_ROUTED_EXPERT_HIDDEN);
		return err == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
	}
	for ( uint32_t cell_begin = 0u; cell_begin < w2_total_cells; cell_begin += slice_cells )
	{
		const uint32_t cur_cells = w2_total_cells - cell_begin < slice_cells ? w2_total_cells - cell_begin : slice_cells;
		const uint32_t cur_col_blocks = cur_cells / LM_CELL_MMA_BLOCK_CELLS;
		const uint32_t cur_cols = cur_cells * LM_SKINNY_CELL_NEURONS;
		if ( K3Skl2Launch(K3SKL2_k3_cell_mma_w2_slice_c4, max_active * cur_col_blocks, LM_CELL_MMA_THREADS, 0u, stream,
			(const uint8_t *)w2_weight, intermediate_bf16, gate_up_bf16, group_row_offset, active_experts,
			w2_group_bytes, w2_k_tiles, w2_total_cells, cell_begin, cur_cells, channels) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
		if ( K3Skl2Launch(K3SKL2_k3_moe_finalize_warp_slice_128, (rows + 3u) / 4u, 128u, 0u, stream,
			gate_up_bf16, route_packed_row, route_weight, latent_bf16, rows,
			cell_begin * LM_SKINNY_CELL_NEURONS, cur_cols, (uint32_t)K3_ROUTED_EXPERT_HIDDEN) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	}
	return LM_LAUNCH_OK;
}

static int32_t K3MoeCellsSkinnyFastLaunch(const void *w1_weight, const void *w2_weight,
	const uint16_t *latent_full_bf16, uint16_t *gate_up_bf16, uint16_t *intermediate_bf16, uint16_t *latent_bf16,
	const uint32_t *group_row_offset, const uint32_t *route_expert, const uint32_t *route_source_token,
	const uint32_t *route_packed_row, const float *route_weight, const uint32_t *active_experts,
	uint32_t rows, uint32_t packed_rows, uint32_t channels, SparkGpuStream stream)
{
	const uint32_t w1_k_tiles = K3_ROUTED_EXPERT_HIDDEN / LM_SKINNY_CELL_TILE_K;
	const uint32_t half_cells = channels / LM_SKINNY_CELL_NEURONS;
	const uint32_t max_active = packed_rows < K3_EXPERTS ? packed_rows : K3_EXPERTS;
	const uint32_t w1_group_bytes = w1_k_tiles * (half_cells * 2u) * LM_CELL_MMA_CELL_BYTES;
	if ( K3Skl2Launch(rows == 1u ? K3SKL2_k3_cell_skinny_w1_situ_b1 : K3SKL2_k3_cell_skinny_w1_situ_b4,
		max_active * half_cells, 128u, 0u, stream,
		(const uint8_t *)w1_weight, latent_full_bf16, intermediate_bf16, group_row_offset, route_source_token,
		active_experts, w1_group_bytes, w1_k_tiles, half_cells, (uint32_t)K3_ROUTED_EXPERT_HIDDEN, channels,
		(float)K3_SITU_BETA, (float)K3_SITU_LINEAR_BETA) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	const uint32_t w2_k_tiles = channels / LM_SKINNY_CELL_TILE_K;
	const uint32_t w2_total_cells = K3_ROUTED_EXPERT_HIDDEN / LM_SKINNY_CELL_NEURONS;
	const uint32_t w2_group_bytes = w2_k_tiles * w2_total_cells * LM_CELL_MMA_CELL_BYTES;
	if ( rows == 1u )
	{
		const SparkGpuError err = K3Skl2Launch(K3SKL2_k3_cell_skinny_w2_finalize_row1, w2_total_cells, 128u, 0u, stream,
			(const uint8_t *)w2_weight, intermediate_bf16, route_expert, route_packed_row, route_weight,
			latent_bf16, w2_group_bytes, w2_k_tiles, w2_total_cells, channels);
		return err == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
	}
	const uint32_t w2_col_blocks = w2_total_cells / 8u;
	if ( K3Skl2Launch(K3SKL2_k3_cell_skinny_w2_b4, max_active * w2_col_blocks, 128u, 0u, stream,
		(const uint8_t *)w2_weight, intermediate_bf16, gate_up_bf16, group_row_offset, active_experts,
		w2_group_bytes, w2_k_tiles, w2_total_cells, channels, (uint32_t)K3_ROUTED_EXPERT_HIDDEN) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	const SparkGpuError err = K3Skl2Launch(K3SKL2_k3_moe_finalize_token224_224, rows, 224u, 0u, stream,
		gate_up_bf16, route_packed_row, route_weight, latent_bf16, rows, (uint32_t)K3_ROUTED_EXPERT_HIDDEN);
	return err == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

struct K3SplitKTarget
{
	const uint8_t *weight;
	uint16_t *out_bf16;
	float *out_f32;
	uint32_t n;
	uint32_t stride;
	uint32_t col_offset;
	uint32_t first_tile;
};

struct K3SplitKArgs
{
	K3SplitKTarget t[2];
	uint32_t count;
	uint32_t total_tiles;
	uint32_t total_cols;
	uint32_t k_tiles;
	uint32_t split_k;
	uint32_t rows;
	uint32_t input_dim;
	const uint16_t *activation;
	float *scratch;
};

static_assert(sizeof(K3SplitKTarget) == 40u && sizeof(K3SplitKArgs) == 128u &&
	offsetof(K3SplitKArgs, count) == 80u && offsetof(K3SplitKArgs, total_cols) == 88u &&
	offsetof(K3SplitKArgs, k_tiles) == 92u && offsetof(K3SplitKArgs, split_k) == 96u &&
	offsetof(K3SplitKArgs, rows) == 100u && offsetof(K3SplitKArgs, input_dim) == 104u &&
	offsetof(K3SplitKArgs, activation) == 112u && offsetof(K3SplitKArgs, scratch) == 120u,
	"K3SplitKArgs layout matches skl2 blob[128, 8]");

static int32_t K3SplitKMmaLaunch(const K3SplitKTarget *targets, uint32_t count, const uint16_t *activation,
	float *scratch, uint32_t rows, uint32_t input_dim, uint32_t split_k, uint16_t *situ_out, SparkGpuStream stream)
{
	K3SplitKArgs args = {};
	args.count = count;
	args.k_tiles = input_dim / 64u;
	args.split_k = split_k;
	args.rows = rows;
	args.input_dim = input_dim;
	args.activation = activation;
	args.scratch = scratch;
	uint32_t tiles = 0u;
	for ( uint32_t p = 0u; p < count; ++p )
	{
		args.t[p] = targets[p];
		args.t[p].first_tile = tiles;
		tiles += (targets[p].n + 31u) / 32u;
		LmWeightPrefetchNote(stream, targets[p].weight, (uint64_t)targets[p].n * input_dim * sizeof(uint16_t));
	}
	args.total_tiles = tiles;
	args.total_cols = tiles * 32u;
	const dim3 grid(tiles, split_k);
	if ( K3Skl2Launch(rows <= 32u ? K3SKL2_k3_splitk_mma_32x32 : K3SKL2_k3_splitk_mma_64x32,
		grid, 128u, 0u, stream, args) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(LM_LAUNCH_ERR_LAUNCH);
	SparkGpuError err;
	if ( situ_out != 0 )
	{
		const uint32_t ch = targets[0].n / 2u;
		err = K3Skl2Launch(K3SKL2_k3_splitk_reduce_situ_256, dim3(((ch / 2u) + 255u) / 256u, rows), 256u, 0u, stream,
			scratch, situ_out, rows, ch, split_k, (float)K3_SITU_BETA, (float)K3_SITU_LINEAR_BETA);
	}
	else
		err = K3Skl2Launch(K3SKL2_k3_splitk_reduce_256, dim3(((args.total_cols / 2u) + 255u) / 256u, rows), 256u, 0u, stream, args);
	return err == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

struct K3KdaProjMmaArgs
{
	const uint8_t *w_qkvb;
	const uint8_t *w_gate;
	const uint8_t *w_decay;
	const uint16_t *activation;
	uint16_t *q;
	uint16_t *k;
	uint16_t *v;
	uint16_t *beta;
	uint16_t *gate;
	uint16_t *latent;
	uint16_t *rq;
	uint16_t *rk;
	uint16_t *rv;
	uint32_t rows;
	uint32_t qkvb_n;
	uint32_t gate_n;
	uint32_t decay_n;
	uint32_t rank_qk;
	uint32_t rank_v;
	uint32_t rank_heads;
	uint32_t tiles_qkvb;
	uint32_t tiles_gate;
};

static_assert(sizeof(K3KdaProjMmaArgs) == 144u &&
	offsetof(K3KdaProjMmaArgs, activation) == 24u && offsetof(K3KdaProjMmaArgs, q) == 32u &&
	offsetof(K3KdaProjMmaArgs, rq) == 80u && offsetof(K3KdaProjMmaArgs, rows) == 104u &&
	offsetof(K3KdaProjMmaArgs, tiles_qkvb) == 132u && offsetof(K3KdaProjMmaArgs, tiles_gate) == 136u,
	"K3KdaProjMmaArgs layout matches skl2 blob[144, 8]");

static int32_t K3WideMmaLaunch(const void *weight, const uint16_t *activation, uint16_t *output_bf16,
	uint32_t rows, uint32_t input_dim, uint32_t output_dim, SparkGpuStream stream)
{
	LmWeightPrefetchNote(stream, weight, (uint64_t)output_dim * input_dim * sizeof(uint16_t));
	const uint32_t tiles = (output_dim + 31u) / 32u;
	const SparkGpuError err = K3Skl2Launch(rows <= 32u ? K3SKL2_k3_wide_mma_32x32 : K3SKL2_k3_wide_mma_64x32,
		tiles, 128u, 0u, stream, (const uint8_t *)weight, activation, output_bf16, rows, input_dim, output_dim);
	return err == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

static int32_t K3IndexQMmaWideSkipLaunch(const void *weight, const uint16_t *activation, uint16_t *output,
	const uint32_t *sequence_of_row, const uint32_t *context_length, const uint32_t *positions,
	uint32_t rows, SparkGpuStream stream)
{
	LmWeightPrefetchNote(stream, weight, (uint64_t)K3_INDEX_QUERY * K3_Q_LORA_RANK * sizeof(uint16_t));
	const bool wide = rows >= 128u;
	const uint32_t shared = wide ? (3u * (128u + 128u) * 128u) : (3u * (64u + 128u) * 128u);
	const uint32_t tiles = (wide ? ((rows + 127u) / 128u) : ((rows + 63u) / 64u)) * (K3_INDEX_QUERY / 128u);
	const SparkGpuError err = K3Skl2Launch(wide ? K3SKL2_k3_index_q_mma_wide_skip_128x128 : K3SKL2_k3_index_q_mma_wide_skip_64x128,
		tiles, LM_STREAM_THREADS, shared, stream, (const uint8_t *)weight, activation, output,
		sequence_of_row, context_length, positions, rows,
		(uint32_t)K3_Q_LORA_RANK, (uint32_t)K3_INDEX_QUERY, (uint32_t)K3_INDEX_TOPK);
	return err == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

static int32_t K3KdaProjPrefillMmaLaunch(const K3KdaProjMmaArgs &args, SparkGpuStream stream)
{
	constexpr uint32_t SHARED = 3u * (128u + 128u) * 128u;
	LmWeightPrefetchNote(stream, args.w_qkvb, (uint64_t)args.qkvb_n * K3_HIDDEN * sizeof(uint16_t));
	LmWeightPrefetchNote(stream, args.w_gate, (uint64_t)args.gate_n * K3_HIDDEN * sizeof(uint16_t));
	if ( args.decay_n != 0u )
		LmWeightPrefetchNote(stream, args.w_decay, (uint64_t)args.decay_n * K3_HIDDEN * sizeof(uint16_t));
	const uint32_t total_n = args.tiles_qkvb + args.tiles_gate + (args.decay_n + 127u) / 128u;
	const uint32_t tiles = ((args.rows + 127u) / 128u) * total_n;
	const SparkGpuError err = K3Skl2Launch(K3SKL2_k3_kda_proj_prefill_mma_128x128,
		tiles, LM_STREAM_THREADS, SHARED, stream, args, (uint32_t)K3_HIDDEN);
	return err == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

static int32_t K3MoeDownRoutePrefillMmaLaunch(const void *w_down, const void *w_router, const uint16_t *activation,
	uint16_t *latent_bf16, float *router_f32, uint32_t rows, uint32_t down_n, uint32_t router_n, SparkGpuStream stream)
{
	constexpr uint32_t SHARED = 3u * (64u + 64u) * 128u;
	LmWeightPrefetchNote(stream, w_down, (uint64_t)down_n * K3_HIDDEN * sizeof(uint16_t));
	LmWeightPrefetchNote(stream, w_router, (uint64_t)router_n * K3_HIDDEN * sizeof(uint16_t));
	const uint32_t total_n = (down_n + 63u) / 64u + (router_n + 63u) / 64u;
	const uint32_t tiles = ((rows + 63u) / 64u) * total_n;
	const SparkGpuError err = K3Skl2Launch(K3SKL2_k3_moe_down_route_prefill_mma_64x64,
		tiles, LM_STREAM_THREADS, SHARED, stream, (const uint8_t *)w_down, (const uint8_t *)w_router,
		activation, latent_bf16, router_f32, rows, (uint32_t)K3_HIDDEN, down_n, router_n);
	return err == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

static int32_t K3SharedW1SituPrefillMmaLaunch(const void *weight, const uint16_t *activation, uint16_t *output_bf16,
	uint32_t rows, uint32_t channels, SparkGpuStream stream)
{
	LmWeightPrefetchNote(stream, weight, (uint64_t)channels * 2u * K3_HIDDEN * sizeof(uint16_t));
	const uint32_t n_tiles = (channels + 63u) / 64u;
	const bool wide = rows >= 1024u;
	const uint32_t shared = wide ? (3u * (128u + 128u) * 128u) : (3u * (64u + 128u) * 128u);
	const uint32_t tiles = (wide ? ((rows + 127u) / 128u) : ((rows + 63u) / 64u)) * n_tiles;
	const SparkGpuError err = K3Skl2Launch(wide ? K3SKL2_k3_shared_w1_situ_prefill_mma_128x128 : K3SKL2_k3_shared_w1_situ_prefill_mma_64x128,
		tiles, LM_STREAM_THREADS, shared, stream, (const uint8_t *)weight, activation, output_bf16,
		rows, (uint32_t)K3_HIDDEN, channels, (float)K3_SITU_BETA, (float)K3_SITU_LINEAR_BETA);
	return err == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}

static int32_t K3RoutedUpAddPrefillMmaLaunch(const void *weight, const uint16_t *activation,
	const uint16_t *addend_bf16, uint16_t *output_bf16, uint32_t rows, uint32_t input_dim, uint32_t output_dim,
	SparkGpuStream stream)
{
	constexpr uint32_t SHARED = 3u * (128u + 128u) * 64u;
	LmWeightPrefetchNote(stream, weight, (uint64_t)output_dim * input_dim * sizeof(uint16_t));
	const uint32_t tiles = ((rows + 127u) / 128u) * ((output_dim + 127u) / 128u);
	const SparkGpuError err = K3Skl2Launch(K3SKL2_k3_routed_up_add_prefill_mma_128x128,
		tiles, LM_STREAM_THREADS, SHARED, stream, (const uint8_t *)weight, activation, addend_bf16,
		output_bf16, rows, input_dim, output_dim);
	return err == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}
