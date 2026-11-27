#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "runtime/launch.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weight_codec.h"
#include "inference/runner/stage_model_cuda.cuh"
#include "inference/kernels/attn.cuh"
#include "inference/kernels/dense_rows.cuh"
#include "inference/kernels/gqa.cuh"
#include "inference/kernels/gqa_shard.cuh"
#include "inference/kernels/hyper_mix.cuh"
#include "inference/kernels/kv.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "inference/kernels/linear_attn.cuh"
#include "inference/kernels/moe_experts.cuh"
#include "inference/kernels/moe_local.cuh"
#include "inference/kernels/ngram_hash.cuh"
#include "inference/kernels/norm.cuh"
#include "inference/kernels/project.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/stream_gemm.cuh"
#include "inference/kernels/topk_warp.cuh"
#include "inference/kernels/weight_codec.cuh"
#include "runtime/gemm.cuh"
#include "sparkpipe/spark_qwen38_family.h"
#include "sparkpipe/spark_gpu.h"
#include "skl2_qwen38_embedded.h"

struct QwenSkl2BundleTag {};

template<typename... Args>
static inline SparkGpuError QwenSkl2Launch(uint32_t id, dim3 grid, dim3 block, size_t shared, SparkGpuStream stream, Args... args)
{
	return LmLaunchEmbedded<QwenSkl2BundleTag>(kQwenSkl2Cubin, sizeof(kQwenSkl2Cubin), (const char *)kQwenSkl2Manifest,
		sizeof(kQwenSkl2Manifest) - 1u, kQwenSkl2Names, (uint32_t)QWENSKL2_COUNT, id, grid, block, shared, stream, args...);
}

#define QH_THREADS 256u
#define QH_SCALE_K 128u

constexpr uint32_t QwenHybridGatherParts(uint32_t ranks, uint32_t width, uint32_t limit)
{
	uint32_t parts = ranks == 1u ? 1u : (width + limit - 1u) / limit;
	while ( width % parts != 0u )
		++parts;
	return parts;
}

template<class G>
struct QwenHybridShape : QwenHybridLayout<G>
{
	using Kv = LmKvHeads<16u, G::kKvHeads, G::kHeadDim, G::kPageSlots>;
	static constexpr uint32_t kGatherParts = QwenHybridGatherParts(G::kTpDegree, QwenHybridLayout<G>::kQkvWidth / G::kTpDegree, G::kHidden);
	static_assert(QwenHybridLayout<G>::kQkvWidth / G::kTpDegree % kGatherParts == 0u &&
		(G::kTpDegree == 1u || QwenHybridLayout<G>::kQkvWidth / G::kTpDegree / kGatherParts <= G::kHidden),
		"a rank's fused attention rows gather as whole collective rows of at most the hidden width");
	using IndexKv = LmKvGeometry<QwenHybridLayout<G>::kIndexer ? G::kIndexDim * 2u : 16u, G::kPageSlots / G::kIndexRatio, true>;
	static_assert(!G::kMoe || LM_SKINNY_ROWS_WIDE * G::kTopK <= LM_SKINNY_ROWS * G::kLocalExperts,
		"a 16-row verify chunk keeps its routes on the grouped skinny expert variants whose per-route sums match one-row decode");
};

typedef struct QwenHybridLayer
{
	const uint8_t *mix_weight, *out_weight, *ffn_gate_up, *ffn_down;
	const uint8_t *moe_w1, *moe_w1_scale, *moe_w3, *moe_w3_scale, *moe_w2, *moe_w2_scale;
	const float *mix_scale, *out_scale, *ffn_gate_up_scale, *ffn_down_scale;
	const float *gdn_a_log, *gdn_dt_bias, *q_norm, *k_norm, *input_norm, *post_norm;
	const float *hc_norm[2], *index_q_norm, *index_k_norm;
	const uint16_t *gdn_ba, *gdn_conv, *gdn_norm, *moe_router, *moe_shared_gate_up, *moe_shared_down, *moe_shared_gate;
	const uint16_t *hc_down[2], *hc_up[2], *hc_inject[2], *index_qk;
	uint32_t recurrent_index, kv_index;
} QwenHybridLayer;

template<class G>
struct QwenHybridState : CudaStagePackState
{
	const char *linear_kind;
	uint32_t linear_bytes, ffn_rows, expert_first, gdn_heads, gdn_qk, gdn_v, gdn_channels, attn_rows, attn_width;
	uint64_t gdn_state_bytes, gdn_window_bytes, record_stride, tap_stride, ple_window_bytes;
	QwenHybridLayer layers[G::kLayers];
	FILE *dump;
	int16_t tap_output[G::kLayers];
	uint16_t *dump_rows, *tap_slab, *normed, *partial, *gate_up, *intermediate;
	uint16_t *qkv, *z, *ba, *gdn_qkvz, *value, *query_heads, *key_heads, *mixed;
	uint16_t *attn_local, *attn_gathered, *attn_query, *merged, *merged_gate;
	uint16_t *router_logits, *expert_gate, *expert_up, *expert_act, *expert_out, *shared_gate_up, *shared_act, *shared_out, *shared_logit;
	uint16_t *gdn_window, *replay_qkv, *replay_ba, *replay_ple, *replay_index, *residual, *hc_normed, *hc_low, *hc_up;
	uint16_t *ple_embed, *ple_key, *ple_query, *ple_value, *ple_gated, *ple_conv_in, *ple_conv_out, *ple_window;
	uint16_t *index_qk, *index_query, *index_key, *index_tail;
	float *retention, *write_gate, *send, *received, *router_probs, *route_weight, *packed_weight, *gdn_snapshots, *hc_inject;
	float *index_scores;
	uint32_t *dump_positions, *route_expert, *route_global, *group_row_offset, *route_packed_row, *route_source_token, *packed_expert, *tile_prefix_up, *tile_prefix_down;
	uint32_t *row_lane, *row_ordinal, *index_local_selected, *index_mask;
	uint8_t *gdn_state;
	const float *final_hc_norm, *ple_norm_key, *ple_norm_query, *ple_norm_conv, *ple_table_scale;
	const uint16_t *final_hc_down, *final_hc_up, *ple_key_weight, *ple_value_weight, *ple_conv_weight;
	const uint8_t *ple_table;
	const int64_t *ple_multipliers, *ple_vocab_sizes, *ple_offsets;
	int64_t *ple_ids;
	int32_t *ple_history;
	uint32_t ple_table_first, ple_table_rows, max_positions, index_local_stride, index_global_stride, index_mask_words, hc_pending_inject;
	uint64_t split_gqa_floats;
	uint2 *index_candidates, *index_gathered;
	LmKvShardView index_views[QwenHybridShape<G>::kAttnLayers], kv_views[QwenHybridShape<G>::kAttnLayers];
};

#define QH_PREFILL_MAX_SEQS 64u

typedef struct QwenPrefillSeqPlan
{
	uint32_t count;
	uint32_t total_prior;
	uint32_t total_peer;
	uint32_t seq_id[QH_PREFILL_MAX_SEQS];
	uint32_t row_begin[QH_PREFILL_MAX_SEQS];
	uint32_t row_len[QH_PREFILL_MAX_SEQS];
	uint32_t old_ctx[QH_PREFILL_MAX_SEQS];
	uint32_t prior_off[QH_PREFILL_MAX_SEQS];
	uint32_t peer_off[QH_PREFILL_MAX_SEQS];
	uint32_t peer_keys[QH_PREFILL_MAX_SEQS];
}
QwenPrefillSeqPlan;

static constexpr uint32_t QH_GQA_MAX_SPLITS = 12u;
static constexpr uint32_t QH_VERIFY_INVARIANT_SEQS = 8u;

template<uint32_t KEY_DIM, uint32_t VALUE_DIM>
static inline SparkGpuError QwenGdnScalarChunkLaunch(uint8_t *state_pool, uint32_t slot_bytes, const uint32_t *state_index,
	const uint32_t *sequence_row_begin, const uint32_t *sequence_row_count, const uint32_t *positions, const uint16_t *query_qk, const uint16_t *key_qk,
	const uint16_t *value_bf16, const uint16_t *ba_bf16, const float *a_log, const float *dt_bias, uint16_t *output_bf16,
	uint32_t qk_heads, uint32_t heads, uint32_t sequences, uint32_t commit, const uint32_t *sequence_row_indices, SparkGpuStream stream)
{
	if ( sequences == 0u || heads == 0u )
		return SPARK_GPU_ERROR_INVALID_VALUE;
	SparkGpuError err = SPARK_GPU_SUCCESS;
	if ( positions != 0 )
		err = QwenSkl2Launch(QWENSKL2_qwen_gdn_exact_edge_pre_128_128_pdl, dim3(sequences, heads, VALUE_DIM / LM_WARP_LANES),
			LM_DELTA_COLUMN_THREADS, 0, stream, state_pool, slot_bytes, state_index, sequence_row_begin, sequence_row_count, positions, query_qk, key_qk,
			value_bf16, ba_bf16, a_log, dt_bias, output_bf16, qk_heads, heads, sequences, commit, sequence_row_indices);
	if ( err == SPARK_GPU_SUCCESS )
		err = QwenSkl2Launch(QWENSKL2_qwen_gdn_scalar_chunk_128_128_128_pdl, dim3(sequences, heads, 1u), 256u, 62464u, stream,
			state_pool, slot_bytes, state_index, sequence_row_begin, sequence_row_count, positions, query_qk, key_qk, value_bf16,
			ba_bf16, a_log, dt_bias, output_bf16, qk_heads, heads, sequences, commit, sequence_row_indices);
	if ( err == SPARK_GPU_SUCCESS && positions != 0 )
		err = QwenSkl2Launch(QWENSKL2_qwen_gdn_exact_edge_post_128_128_pdl, dim3(sequences, heads, VALUE_DIM / LM_WARP_LANES),
			LM_DELTA_COLUMN_THREADS, 0, stream, state_pool, slot_bytes, state_index, sequence_row_begin, sequence_row_count, positions, query_qk, key_qk,
			value_bf16, ba_bf16, a_log, dt_bias, output_bf16, qk_heads, heads, sequences, commit, sequence_row_indices);
	return err;
}

typedef struct QwenBf16Part
{
	const uint16_t *weight;
	uint16_t *output;
	uint32_t out_dim;
	uint32_t first_block;
} QwenBf16Part;

typedef struct QwenBf16MultiArgs
{
	QwenBf16Part part[3];
	uint32_t count;
	uint32_t total_blocks;
	const uint16_t *activation;
	uint32_t rows;
	uint32_t input_dim;
} QwenBf16MultiArgs;

#define QH_MOE_LANE_ROWS 1024u

template<class G>
struct QwenHybridModel : CudaStage<QwenHybridModel<G>>
{
	using S = QwenHybridShape<G>;
	using State = QwenHybridState<G>;
	using Layer = QwenHybridLayer;
	using Schedule = QwenHybridSchedule<G, QwenHybridModel>;
	using KvViews = CudaStageKvViews<typename S::Kv, &State::kv_views, 0u, S::kAttnLayers>;
	using IndexViews = CudaStageKvViews<typename S::IndexKv, &State::index_views, 1u, S::kAttnLayers>;
	using Kv = typename std::conditional<S::kIndexer, CudaStageKv<KvViews, IndexViews>, CudaStageKv<KvViews>>::type;
	using LaneShared = CudaStageFields<&State::gdn_state, &State::gdn_window, &State::ple_window, &State::ple_history, &State::index_tail, &State::kv_error>;

	static constexpr bool kLanes = G::kTpDegree > 1u;
	static constexpr bool kKeepRows = G::kVerifyRows != 0u;
	static constexpr bool kTreeVerify = !S::kPle && !S::kIndexer && G::kVerifyRows != 0u;
	static constexpr bool kTapPlan = true;
	static constexpr bool kGraphKey = true;
	static constexpr uint32_t kChunkRows = G::kMoe ? QH_MOE_LANE_ROWS : 0u;
	static constexpr uint32_t kTpDegree = G::kTpDegree;
	static constexpr uint32_t kHidden = G::kHidden;
	static constexpr uint32_t kVocab = G::kVocab;
	static constexpr uint32_t kLayers = G::kLayers;
	static constexpr float kEpsilon = QH_EPSILON;
	static constexpr uint32_t kPackAlignment = QH_PACK_ALIGNMENT;

	struct Frame
	{
		const SparkStageRunnerStep *step;
		SparkGpuStream stream;
	};

	static const char *Tag(void)
	{
		return G::Tag();
	}

	static const char *ReportTag(void)
	{
		return G::ReportTag();
	}

	static const char *WeightdModel(void)
	{
		return G::Name();
	}

	static bool PrefillWave(const SparkStageRunnerStep *step)
	{
		return step->commit != 0u && step->last_rows_only != 0u && step->tree == 0u;
	}

	static bool PrefillUnified(const SparkStageRunnerStep *step)
	{
		return PrefillWave(step) && step->sequences == 1u;
	}

	static int32_t LaunchExactBf16Multi(QwenBf16MultiArgs args, SparkGpuStream stream)
	{
		const uint32_t rows = args.rows;
		const uint32_t tile_n = rows <= 8u ? 16u : (rows <= 16u ? 8u : 4u);
		uint32_t blk = 0u;
		for ( uint32_t i = 0u; i < args.count; ++i )
		{
			args.part[i].first_block = blk;
			blk += (args.part[i].out_dim + tile_n - 1u) / tile_n;
		}
		args.total_blocks = blk;
		SparkGpuError err;
		if ( rows <= 4u )
			err = QwenSkl2Launch(QWENSKL2_qwen_exact_bf16_multi_w1x4_r4_n4, dim3(blk, 1u), 128u, 0, stream, args);
		else if ( rows <= 8u )
			err = QwenSkl2Launch(QWENSKL2_qwen_exact_bf16_multi_w2x4_r4_n4, dim3(blk, 1u), 256u, 0, stream, args);
		else if ( rows <= 16u )
			err = QwenSkl2Launch(QWENSKL2_qwen_exact_bf16_multi_w4x2_r4_n4, dim3(blk, 1u), 256u, 0, stream, args);
		else
			err = QwenSkl2Launch(QWENSKL2_qwen_exact_bf16_multi_w4x2_r8_n2, dim3(blk, (rows + 31u) / 32u), 256u, 0, stream, args);
		return err == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
	}

	static int32_t LaunchBf16Mma(const uint16_t *weight, const uint16_t *source, uint16_t *destination,
		uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, SparkGpuStream stream)
	{
		if ( output_dimension < 512u && (input_dimension % 256u) == 0u )
		{
			const uint32_t nb = (output_dimension + 15u) / 16u;
			SparkGpuError err;
			if ( rows > 16u && (output_dimension >= 16u || rows >= 1024u) )
				err = QwenSkl2Launch(QWENSKL2_qwen_bf16_smalln_mma_wide_32, ((rows + 31u) / 32u) * nb, 128u, 49152u, stream,
					weight, source, destination, rows, input_dimension, output_dimension);
			else
				err = QwenSkl2Launch(QWENSKL2_qwen_bf16_smalln_mma_16, ((rows + 15u) / 16u) * nb, 128u, 0, stream,
					weight, source, destination, rows, input_dimension, output_dimension);
			return err == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
		}
		if ( output_dimension < 512u || rows > 128u || (rows > 16u && (output_dimension >= 8192u || rows >= 32u)) )
			return LM_LAUNCH_ERR_SHAPE;
		const uint32_t nb = (output_dimension + 63u) / 64u;
		SparkGpuError err;
		if ( rows <= 16u )
			err = QwenSkl2Launch(QWENSKL2_qwen_bf16_dense_mma_16x64, ((rows + 15u) / 16u) * nb, 128u, 0, stream,
				weight, source, destination, rows, input_dimension, output_dimension);
		else if ( rows <= 32u )
			err = QwenSkl2Launch(QWENSKL2_qwen_bf16_dense_mma_32x64, ((rows + 31u) / 32u) * nb, 256u, 0, stream,
				weight, source, destination, rows, input_dimension, output_dimension);
		else if ( rows <= 64u )
			err = QwenSkl2Launch(QWENSKL2_qwen_bf16_dense_mma_64x64, ((rows + 63u) / 64u) * nb, 256u, 0, stream,
				weight, source, destination, rows, input_dimension, output_dimension);
		else
			err = QwenSkl2Launch(QWENSKL2_qwen_bf16_dense_mma_128x64, ((rows + 127u) / 128u) * nb, 256u, 0, stream,
				weight, source, destination, rows, input_dimension, output_dimension);
		return err == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
	}

	template<uint32_t TILE_M>
	static int32_t LaunchSharedGateUpSwigluTile(const uint16_t *weight, const uint16_t *source, uint16_t *destination,
		uint32_t rows, uint32_t input_dimension, uint32_t shared_rows, SparkGpuStream stream)
	{
		constexpr uint32_t smem = 4u * 2u * (TILE_M + 32u) * 128u;
		constexpr uint32_t kid = TILE_M == 32u ? QWENSKL2_qwen_shared_gate_up_swiglu_wide_32 : QWENSKL2_qwen_shared_gate_up_swiglu_wide_16;
		const uint32_t blocks = ((rows + TILE_M - 1u) / TILE_M) * ((shared_rows + 15u) / 16u);
		const SparkGpuError err = QwenSkl2Launch(kid, blocks, 128u, smem, stream,
			weight, source, destination, rows, input_dimension, shared_rows);
		return err == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
	}

	static int32_t LaunchFusedGateUpSwiglu(const uint16_t *weight, const uint16_t *source, uint16_t *destination,
		uint32_t rows, uint32_t input_dimension, uint32_t ffn_rows, uint32_t exact, bool force_stream, SparkGpuStream stream)
	{
		const bool use_mma = (force_stream || (exact == 0u && rows > LM_SKINNY_ROWS_WIDE)) && (input_dimension % 64u) == 0u;
		SparkGpuError err;
		if ( use_mma )
		{
			if ( rows <= 16u )
				err = QwenSkl2Launch(QWENSKL2_qwen_bf16_fused_swiglu_mma_16x64,
					((rows + 15u) / 16u) * ((ffn_rows + 63u) / 64u), 128u, 0, stream, weight, source, destination, rows, input_dimension, ffn_rows);
			else if ( rows <= 32u )
				err = QwenSkl2Launch(QWENSKL2_qwen_bf16_fused_swiglu_mma_32x64,
					((rows + 31u) / 32u) * ((ffn_rows + 63u) / 64u), 256u, 0, stream, weight, source, destination, rows, input_dimension, ffn_rows);
			else if ( rows <= 64u )
				err = QwenSkl2Launch(QWENSKL2_qwen_bf16_fused_swiglu_mma_64x32,
					((rows + 63u) / 64u) * ((ffn_rows + 31u) / 32u), 256u, 0, stream, weight, source, destination, rows, input_dimension, ffn_rows);
			else
				err = QwenSkl2Launch(QWENSKL2_qwen_bf16_fused_swiglu_mma_128x64,
					((rows + 127u) / 128u) * ((ffn_rows + 63u) / 64u), 256u, 3u * (128u + 128u) * 128u, stream,
					weight, source, destination, rows, input_dimension, ffn_rows);
			return err == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
		}
		if ( rows == 1u || rows > 32u )
			return LM_LAUNCH_ERR_SHAPE;
		if ( rows <= 4u )
			err = QwenSkl2Launch(QWENSKL2_qwen_exact_bf16_gate_up_swiglu_w1x4_r4_n4, dim3((ffn_rows + 15u) / 16u, 1u), 128u, 0, stream,
				weight, source, destination, rows, input_dimension, ffn_rows);
		else if ( rows <= 8u )
			err = QwenSkl2Launch(QWENSKL2_qwen_exact_bf16_gate_up_swiglu_w2x4_r4_n4, dim3((ffn_rows + 15u) / 16u, 1u), 256u, 0, stream,
				weight, source, destination, rows, input_dimension, ffn_rows);
		else if ( rows <= 16u )
			err = QwenSkl2Launch(QWENSKL2_qwen_exact_bf16_gate_up_swiglu_w4x2_r4_n4, dim3((ffn_rows + 7u) / 8u, 1u), 256u, 0, stream,
				weight, source, destination, rows, input_dimension, ffn_rows);
		else
			err = QwenSkl2Launch(QWENSKL2_qwen_exact_bf16_gate_up_swiglu_w4x2_r8_n2, dim3((ffn_rows + 3u) / 4u, 1u), 256u, 0, stream,
				weight, source, destination, rows, input_dimension, ffn_rows);
		return err == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
	}

	template<class Format>
	static int32_t Project(const void *weight, const float *scale_rows, const uint16_t *source, uint16_t *destination,
		uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t exact, uint32_t multiprocessors, SparkGpuStream stream,
		bool force_stream = false)
	{
		if constexpr ( std::is_same<Format, LmBf16Format>::value )
		{
			const bool use_mma = force_stream || (exact == 0u && rows > LM_SKINNY_ROWS_WIDE);
			if ( use_mma && (input_dimension % 64u) == 0u && (output_dimension & 1u) == 0u )
			{
				const int32_t status = LaunchBf16Mma((const uint16_t *)weight, source, destination, rows, input_dimension, output_dimension, stream);
				if ( status != LM_LAUNCH_ERR_SHAPE )
					return status;
			}
			else if ( !use_mma && rows >= 2u && rows <= 32u && input_dimension >= 256u && (input_dimension & 7u) == 0u && output_dimension > 384u )
			{
				QwenBf16MultiArgs args;
				args.part[0] = { (const uint16_t *)weight, destination, output_dimension, 0u };
				args.count = 1u;
				args.activation = source;
				args.rows = rows;
				args.input_dim = input_dimension;
				return LaunchExactBf16Multi(args, stream);
			}
		}
		const LmScaleTensor scale = scale_rows != 0 ?
			LmScaleTensorBuild(scale_rows, LM_SCALE_ENCODING_F32, 1u, output_dimension, input_dimension, 1u, QH_SCALE_K) : LmScaleTensorNone();
		if ( force_stream )
		{
			const int32_t status = LmStreamGemmDense<Format>(weight, scale, source, destination, (float *)0, rows, input_dimension,
				output_dimension, 0u, 0u, multiprocessors, stream);
			if ( status != LM_LAUNCH_ERR_SHAPE )
				return status;
		}
		return LmDenseRows<Format>(weight, scale, source, destination, (float *)0, rows, input_dimension, output_dimension, exact, multiprocessors,
			stream);
	}

	static int32_t Linear(const State *state, const uint8_t *weight, const float *scale_rows, const uint16_t *source,
		uint16_t *destination, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension, uint32_t exact, uint32_t multiprocessors,
		SparkGpuStream stream, bool force_stream = false)
	{
		return state->linear_bytes == 1u ?
			Project<LmFp8>(weight, scale_rows, source, destination, rows, input_dimension, output_dimension, exact, multiprocessors, stream, force_stream) :
			Project<LmBf16Format>(weight, 0, source, destination, rows, input_dimension, output_dimension, exact, multiprocessors, stream, force_stream);
	}

	static uint32_t Exact(const SparkStageRunnerStep *step)
	{
		return step->commit == 0u && step->sequences < QH_VERIFY_INVARIANT_SEQS ? 1u : 0u;
	}

	static SparkStatus GdnRecur(State *state, const QwenHybridLayer *layer, const SparkStageRunnerStep *step, const uint16_t *qkv, const uint16_t *ba,
		const uint32_t *kept, uint32_t commit, SparkGpuStream stream, uint32_t qkv_stride = 0u, uint16_t *replay_qkv_out = 0)
	{
		const uint32_t rows = step->rows, sequences = step->sequences, heads = state->gdn_heads;
		const uint32_t tree = step->tree != 0u && commit == 0u ? 1u : 0u;
		uint8_t *state_pool = state->gdn_state + (uint64_t)layer->recurrent_index * state->slots * state->gdn_state_bytes;
		uint16_t *window = state->gdn_window + (uint64_t)layer->recurrent_index * state->slots * (state->gdn_window_bytes / sizeof(uint16_t));
		if ( tree == 0u && commit != 0u && kept == 0 && PrefillWave(step) )
		{
			const uint32_t input_stride = qkv_stride != 0u ? qkv_stride : state->gdn_channels;
			const uint32_t max_row_tiles = (rows + LM_CAUSAL_CONV_TILE_ROWS - 1u) / LM_CAUSAL_CONV_TILE_ROWS;
			const uint32_t ch_tiles = (state->gdn_channels + LM_CAUSAL_CONV_TILE_CHANNELS - 1u) / LM_CAUSAL_CONV_TILE_CHANNELS;
			if ( QwenSkl2Launch(QWENSKL2_qwen_gdn_conv_prefill_2d_4_128_pdl,
				dim3(sequences, ch_tiles, max_row_tiles), LM_CAUSAL_CONV_TILE_THREADS, 0, stream,
				window, step->recurrent_index, step->sequence_row_begin, kept, qkv, layer->gdn_conv,
				state->query_heads, state->key_heads, state->value, state->gdn_qk, state->gdn_v,
				input_stride, sequences, max_row_tiles == 1u ? 1u : 0u, step->sequence_row_indices) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			if ( max_row_tiles > 1u && QwenSkl2Launch(QWENSKL2_qwen_gdn_conv_window_commit_4_pdl,
				dim3(sequences, (state->gdn_channels + QH_THREADS - 1u) / QH_THREADS), QH_THREADS, 0, stream,
				window, step->recurrent_index, step->sequence_row_begin, kept, qkv,
				state->gdn_channels, input_stride, sequences, step->sequence_row_indices) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS ||
				QwenGdnScalarChunkLaunch<QH_KEY_DIM, QH_VALUE_DIM>(state_pool, (uint32_t)state->gdn_state_bytes, step->recurrent_index,
					step->sequence_row_begin, kept, step->positions, state->query_heads, state->key_heads, state->value, ba,
					layer->gdn_a_log, layer->gdn_dt_bias, state->mixed, state->gdn_qk / QH_KEY_DIM, heads,
					sequences, commit, step->sequence_row_indices, stream) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			return CudaStageLaunched();
		}
		if ( tree == 0u )
		{
			const uint32_t input_stride = qkv_stride != 0u ? qkv_stride : state->gdn_channels;
			const uint32_t gate_blocks = (heads + (QH_THREADS / LM_WARP_LANES) - 1u) / (QH_THREADS / LM_WARP_LANES);
			if ( rows <= LM_CAUSAL_CONV_SERIAL_ROWS * sequences )
			{
				const uint32_t conv_blocks = (state->gdn_channels + QH_THREADS - 1u) / QH_THREADS;
				if ( QwenSkl2Launch(QWENSKL2_qwen_gdn_conv_split_serial_4_128_pdl,
					dim3(sequences, conv_blocks + gate_blocks), QH_THREADS, 0, stream,
					window, step->recurrent_index, step->sequence_row_begin, kept, qkv, layer->gdn_conv,
					ba, layer->gdn_a_log, layer->gdn_dt_bias,
					state->query_heads, state->key_heads, state->value, state->retention, state->write_gate,
					replay_qkv_out, state->gdn_qk, state->gdn_v, heads, input_stride, conv_blocks, sequences, commit,
					step->sequence_row_indices) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			else
			{
				if ( QwenSkl2Launch(QWENSKL2_gdn_conv_split_tiled_4_128_pdl,
					dim3(sequences, (state->gdn_channels + LM_CAUSAL_CONV_TILE_CHANNELS - 1u) / LM_CAUSAL_CONV_TILE_CHANNELS),
					LM_CAUSAL_CONV_TILE_THREADS, 0, stream,
					window, step->recurrent_index, step->sequence_row_begin, kept, qkv, layer->gdn_conv,
					state->query_heads, state->key_heads, state->value, replay_qkv_out,
					state->gdn_qk, state->gdn_v, heads, input_stride, sequences, commit, step->sequence_row_indices) != SPARK_GPU_SUCCESS ||
					QwenSkl2Launch(QWENSKL2_qwen_gdn_gate_only_128_pdl,
					dim3(gate_blocks, rows), QH_THREADS, 0, stream,
					ba, layer->gdn_a_log, layer->gdn_dt_bias, state->retention, state->write_gate, heads, rows) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else
		{
			if ( LmCausalConvLaunch<QH_CONV,LM_CONV_SWISH,uint16_t>(window, step->recurrent_index, step->sequence_row_begin, kept, qkv, layer->gdn_conv,
				state->qkv, state->gdn_channels, sequences, commit, step->sequence_row_indices, rows, stream, step->parent_rows) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			if ( QwenSkl2Launch(QWENSKL2_qwen_gdn_split_expand_gate_256_128_pdl, rows, QH_THREADS, 0, stream,
				state->qkv, ba, layer->gdn_a_log, layer->gdn_dt_bias, state->query_heads, state->key_heads, state->value,
				state->retention, state->write_gate, state->gdn_qk, state->gdn_v, heads, rows) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		if ( commit != 0u && kept == 0 && PrefillWave(step) )
		{
			if ( LmDeltaRuleChunkLaunch<QH_THREADS,QH_KEY_DIM,QH_VALUE_DIM,float>(state_pool, (uint32_t)state->gdn_state_bytes, step->recurrent_index,
					step->sequence_row_begin, kept, state->query_heads, state->key_heads, state->value, state->retention, state->write_gate, state->mixed,
					heads, 1u, sequences, commit, step->sequence_row_indices, stream) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else if ( LmDeltaRuleLaunch<QH_THREADS,QH_KEY_DIM,QH_VALUE_DIM,float>(rows, state_pool, (uint32_t)state->gdn_state_bytes, step->recurrent_index,
				step->sequence_row_begin, kept, state->query_heads, state->key_heads, state->value, state->retention, state->write_gate, state->mixed,
				heads, 1u, sequences, commit, step->sequence_row_indices, stream,
				tree != 0u ? step->state_loads : (const uint32_t *)0, tree != 0u ? step->state_saves : (const uint32_t *)0,
				tree != 0u ? state->gdn_snapshots : (float *)0) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return CudaStageLaunched();
	}

	static SparkStatus Gdn(State *state, const Frame &frame, const QwenHybridLayer &entry)
	{
		const QwenHybridLayer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		SparkGpuStream stream = frame.stream;
		const uint32_t rows = step->rows, heads = state->gdn_heads, exact = Exact(step);
		const uint32_t sms = step->multiprocessors;
		const uint64_t replay = (uint64_t)layer->recurrent_index * G::kVerifyRows;
		const uint32_t qkvz_width = state->gdn_channels + state->gdn_v;
		const bool prefill_wave = PrefillWave(step);
		const bool fused_qkvz = (prefill_wave || rows > LM_SKINNY_ROWS_WIDE) && exact == 0u && step->tree == 0u;
		SparkStatus status;
		if ( !prefill_wave && (rows <= LM_SKINNY_ROWS_WIDE || exact != 0u) )
		{
			if ( state->linear_bytes == 2u )
			{
				if ( rows >= 2u && rows <= 32u && G::kHidden >= 256u && (G::kHidden & 7u) == 0u )
				{
					QwenBf16MultiArgs args;
					args.part[0] = { (const uint16_t *)layer->mix_weight, state->qkv, state->gdn_channels, 0u };
					args.part[1] = { (const uint16_t *)(layer->mix_weight + (uint64_t)state->gdn_channels * G::kHidden * 2u), state->z, state->gdn_v, 0u };
					args.part[2] = { layer->gdn_ba, state->ba, 2u * heads, 0u };
					args.count = 3u;
					args.activation = state->normed;
					args.rows = rows;
					args.input_dim = G::kHidden;
					if ( LaunchExactBf16Multi(args, stream) != LM_LAUNCH_OK )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				}
				else
				{
					const LmSkinnyDenseTarget targets[3] =
					{
						{ layer->mix_weight, state->qkv, (float *)0, state->gdn_channels, state->gdn_channels, 0u, LmScaleTensorNone() },
						{ layer->mix_weight + (uint64_t)state->gdn_channels * G::kHidden * 2u, state->z, (float *)0, state->gdn_v, state->gdn_v, 0u, LmScaleTensorNone() },
						{ layer->gdn_ba, state->ba, (float *)0, 2u * heads, 2u * heads, 0u, LmScaleTensorNone() }
					};
					if ( LmSkinnyDenseMulti<LmBf16Format>(targets, 3u, state->normed, rows, G::kHidden, stream, exact) != LM_LAUNCH_OK )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				}
			}
			else
			{
				const LmScaleTensor s0 = LmScaleTensorBuild(layer->mix_scale, LM_SCALE_ENCODING_F32, 1u, state->gdn_channels, G::kHidden, 1u, QH_SCALE_K);
				const LmScaleTensor s1 = LmScaleTensorBuild(layer->mix_scale + (uint64_t)state->gdn_channels * (G::kHidden / QH_SCALE_K),
					LM_SCALE_ENCODING_F32, 1u, state->gdn_v, G::kHidden, 1u, QH_SCALE_K);
				const LmSkinnyDenseTarget targets[2] =
				{
					{ layer->mix_weight, state->qkv, (float *)0, state->gdn_channels, state->gdn_channels, 0u, s0 },
					{ layer->mix_weight + (uint64_t)state->gdn_channels * G::kHidden, state->z, (float *)0, state->gdn_v, state->gdn_v, 0u, s1 }
				};
				if ( LmSkinnyDenseMulti<LmFp8>(targets, 2u, state->normed, rows, G::kHidden, stream, exact) != LM_LAUNCH_OK ||
					Project<LmBf16Format>(layer->gdn_ba, 0, state->normed, state->ba, rows, G::kHidden, 2u * heads, exact, sms, stream) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
		}
		else if ( fused_qkvz )
		{
			if ( Linear(state, layer->mix_weight, layer->mix_scale, state->normed, state->gdn_qkvz, rows, G::kHidden, qkvz_width, 0u, sms, stream, prefill_wave) != LM_LAUNCH_OK ||
				Project<LmBf16Format>(layer->gdn_ba, 0, state->normed, state->ba, rows, G::kHidden, 2u * heads, 0u, sms, stream, prefill_wave) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else
		{
			if ( Linear(state, layer->mix_weight, layer->mix_scale, state->normed, state->qkv, rows, G::kHidden, state->gdn_channels, exact, sms, stream, prefill_wave) != LM_LAUNCH_OK ||
				Linear(state, layer->mix_weight + (uint64_t)state->gdn_channels * G::kHidden * state->linear_bytes,
					layer->mix_scale == 0 ? (const float *)0 : layer->mix_scale + (uint64_t)state->gdn_channels * (G::kHidden / QH_SCALE_K),
					state->normed, state->z, rows, G::kHidden, state->gdn_v, exact, sms, stream, prefill_wave) != LM_LAUNCH_OK ||
				Project<LmBf16Format>(layer->gdn_ba, 0, state->normed, state->ba, rows, G::kHidden, 2u * heads, exact, sms, stream, prefill_wave) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		if ( step->commit == 0u )
		{
			if ( !fused_qkvz &&
				SparkGpuMemcpyAsync(state->replay_qkv + replay * state->gdn_channels, state->qkv, (uint64_t)rows * state->gdn_channels * sizeof(uint16_t),
					SPARK_GPU_COPY_DEVICE_TO_DEVICE, stream) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
			if ( SparkGpuMemcpyAsync(state->replay_ba + replay * 2u * heads, state->ba, (uint64_t)rows * 2u * heads * sizeof(uint16_t),
					SPARK_GPU_COPY_DEVICE_TO_DEVICE, stream) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
		CudaStageRowState(state, layer->recurrent_index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
		status = GdnRecur(state, layer, step,
			fused_qkvz ? state->gdn_qkvz : (step->tree != 0u ? state->replay_qkv + replay * state->gdn_channels : state->qkv),
			state->ba, (const uint32_t *)0, step->commit, stream, fused_qkvz ? qkvz_width : 0u,
			fused_qkvz && step->commit == 0u ? state->replay_qkv + replay * state->gdn_channels : (uint16_t *)0);
		CudaStageRowState(state, layer->recurrent_index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( QwenSkl2Launch(G::kGdnSigmoidGate ? QWENSKL2_head_rmsnorm_sigmoid_gate_pdl_exact_128_bf16 : QWENSKL2_head_rmsnorm_silu_gate_pdl_exact_128_bf16,
				dim3((heads + 7u) >> 3u, rows), QH_THREADS, 0, stream,
				state->mixed, fused_qkvz ? state->gdn_qkvz : state->z, layer->gdn_norm, state->mixed, rows, heads,
				fused_qkvz ? qkvz_width : state->gdn_v, fused_qkvz ? state->gdn_channels : 0u) != SPARK_GPU_SUCCESS ||
			Linear(state, layer->out_weight, layer->out_scale, state->mixed, state->partial, rows, state->gdn_v, G::kHidden, exact, sms, stream, prefill_wave) !=
				LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus IndexPrepare(State *state, const QwenHybridLayer *layer, const SparkStageRunnerStep *step, SparkGpuStream stream,
		bool force_stream = false, bool host_dense = false)
	{
		const uint32_t rows = step->rows, sms = step->multiprocessors, exact = Exact(step);
		uint16_t *tail = state->index_tail + (uint64_t)layer->kv_index * state->slots * S::kIndexTail * G::kIndexDim;
		constexpr uint32_t Q_DIM = G::kIndexHeads * G::kIndexDim;
		constexpr bool kFastIndexSkip = G::kIndexDim <= 256u && (G::kIndexDim & 7u) == 0u && G::kIndexHeads + 1u <= QH_THREADS / LM_WARP_LANES &&
			G::kRopeDim == 64u && (G::kHidden % 64u) == 0u && (Q_DIM % 64u) == 0u;
	const float index_rope_step = -2.0f * log2f(G::kRopeTheta) / (float)G::kRopeDim;
		constexpr uint32_t qprep_qonly_id = G::kIndexHeads <= 4u ? QWENSKL2_index_q_prep_qonly_128 : QWENSKL2_index_q_prep_qonly_256;
		constexpr uint32_t qprep_qonly_blk = G::kIndexHeads <= 4u ? 128u : QH_THREADS;
		if constexpr ( kFastIndexSkip )
		{
			if ( host_dense )
			{
				if ( Project<LmBf16Format>(layer->index_qk + (uint64_t)Q_DIM * G::kHidden, 0, state->normed, state->index_key, rows, G::kHidden,
					G::kIndexDim, exact, sms, stream, force_stream) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			else if ( !force_stream && (rows <= LM_SKINNY_ROWS_WIDE || exact != 0u) )
			{
				if ( QwenSkl2Launch(QWENSKL2_index_qk_exact_skip_128, dim3((S::kIndexWidth + 7u) >> 3u, (rows + 7u) >> 3u), 128u, 0, stream,
						layer->index_qk, state->normed, state->index_qk, state->index_key, step->sequence_of_row, step->context_length, step->positions,
						rows, G::kHidden, Q_DIM, G::kIndexDim, G::kIndexRatio, S::kIndexBlocks) != SPARK_GPU_SUCCESS ||
					QwenSkl2Launch(qprep_qonly_id, rows, qprep_qonly_blk, 0, stream,
						state->index_qk, layer->index_q_norm, step->positions, step->sequence_of_row, step->context_length,
						state->index_query, state->index_key, rows, G::kIndexHeads, G::kIndexDim, G::kRopeDim, G::kIndexRatio, S::kIndexBlocks,
						index_rope_step, QH_EPSILON) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			else
			{
				if ( Project<LmBf16Format>(layer->index_qk + (uint64_t)Q_DIM * G::kHidden, 0, state->normed, state->index_key, rows, G::kHidden,
					G::kIndexDim, exact, sms, stream, force_stream) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				const uint32_t mma_id = rows > 32u ? QWENSKL2_index_q_mma_skip_64x64 : QWENSKL2_index_q_mma_skip_32x64;
				const dim3 mma_grid = rows > 32u ? dim3(Q_DIM / 64u, (rows + 63u) / 64u) : dim3(Q_DIM / 64u, (rows + 31u) / 32u);
				if ( QwenSkl2Launch(mma_id, mma_grid, QH_THREADS, 0, stream,
						layer->index_qk, state->normed, state->index_qk, step->sequence_of_row, step->context_length, step->positions,
						rows, G::kHidden, Q_DIM, G::kIndexRatio, S::kIndexBlocks) != SPARK_GPU_SUCCESS ||
					QwenSkl2Launch(qprep_qonly_id, rows, qprep_qonly_blk, 0, stream,
						state->index_qk, layer->index_q_norm, step->positions, step->sequence_of_row, step->context_length,
						state->index_query, state->index_key, rows, G::kIndexHeads, G::kIndexDim, G::kRopeDim, G::kIndexRatio, S::kIndexBlocks,
						index_rope_step, QH_EPSILON) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
		}
		else
		{
			if ( Project<LmBf16Format>(layer->index_qk, 0, state->normed, state->index_qk, rows, G::kHidden, S::kIndexWidth, exact, sms, stream, force_stream) !=
				LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			if constexpr ( G::kIndexDim <= 256u && (G::kIndexDim & 7u) == 0u && G::kIndexHeads + 1u <= QH_THREADS / LM_WARP_LANES && G::kRopeDim == 64u )
			{
				if ( QwenSkl2Launch(QWENSKL2_index_q_prep_256, rows, QH_THREADS, 0, stream,
						state->index_qk, layer->index_q_norm, step->positions, step->sequence_of_row, step->context_length,
						state->index_query, state->index_key, rows, G::kIndexHeads, G::kIndexDim, G::kRopeDim, G::kIndexRatio, S::kIndexBlocks,
						index_rope_step, QH_EPSILON) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			else
			{
				LM_LAUNCH((LmSplit2RowsKernel<QH_THREADS>), rows, QH_THREADS, 0, stream, state->index_qk, state->index_query, state->index_key, rows,
					G::kIndexHeads * G::kIndexDim, G::kIndexDim);
				LM_LAUNCH((LmHeadRmsNormKernel<QH_THREADS,float>), dim3(G::kIndexHeads,rows), QH_THREADS, 0, stream,
					state->index_query, layer->index_q_norm, state->index_query, rows, G::kIndexHeads, G::kIndexDim, QH_EPSILON, 1.0f);
				LM_LAUNCH((LmRopePerHeadKernel<QH_THREADS>), dim3(rows,G::kIndexHeads), QH_THREADS, 0, stream,
					state->index_query, step->positions, G::kIndexHeads, G::kIndexDim, G::kRopeDim, G::kRopeTheta, (const float *)0, 1.0f, 0u);
			}
		}
		if ( step->commit == 0u && SparkGpuMemcpyAsync(state->replay_index + (uint64_t)layer->kv_index * G::kVerifyRows * G::kIndexDim, state->index_key,
			(uint64_t)rows * G::kIndexDim * sizeof(uint16_t), SPARK_GPU_COPY_DEVICE_TO_DEVICE, stream) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		constexpr uint32_t bstore_id = G::kIndexDim == 64u ? QWENSKL2_index_block_store_64 : QWENSKL2_index_block_store_128;
		if ( QwenSkl2Launch(bstore_id, rows, G::kIndexDim, 0, stream, state->index_key, tail, layer->index_k_norm,
				state->index_views[layer->kv_index], step->sequence_of_row, step->positions, state->row_lane, state->row_ordinal,
				step->sequence_row_begin, step->sequence_row_indices, step->recurrent_index, G::kIndexDim, G::kIndexRatio,
				S::IndexKv::kPageSlots, G::kRopeDim, index_rope_step, QH_EPSILON) != SPARK_GPU_SUCCESS ||
			QwenSkl2Launch(QWENSKL2_index_tail_128, step->sequences, 128u, 0, stream, state->index_key, tail, step->positions,
				step->sequence_row_begin, step->sequence_row_indices, step->recurrent_index, step->commit, (const uint32_t *)0,
				G::kIndexDim, G::kIndexRatio) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return CudaStageLaunched();
	}

	static SparkStatus IndexSelect(State *state, const QwenHybridLayer *layer, const SparkStageRunnerStep *step, SparkGpuStream stream,
		bool prefill_unified, bool host_dense)
	{
		if ( host_dense )
			return SPARK_STATUS_OK;
		const uint32_t rows = step->rows, degree = state->tp_degree;
		const LmKvShardView *view = &state->index_views[layer->kv_index];
		SparkStatus status;
		const uint32_t total_ctx = prefill_unified ? step->gather_context : 0u;
		const uint32_t total_blocks = total_ctx / G::kIndexRatio;
		const uint32_t max_local = total_blocks > S::kIndexBlocks ? SparkKvShardGatherKeys(view->shard, total_blocks) : 0u;
		const uint64_t cand_cap_bytes = (uint64_t)state->max_rows * S::kIndexBlocks * sizeof(uint2);
		const uint64_t gath_cap_bytes = (uint64_t)degree * cand_cap_bytes;
		const uint64_t pack_bytes = (uint64_t)max_local * G::kIndexDim * sizeof(uint16_t);
		const uint64_t keys_bytes = ((uint64_t)degree * pack_bytes + 63u) & ~(uint64_t)63u;
		const uint64_t score_floats = gath_cap_bytes > keys_bytes ? (gath_cap_bytes - keys_bytes) / sizeof(float) : 0u;
		const bool use_compact_index = prefill_unified && total_blocks > S::kIndexBlocks && max_local != 0u &&
			max_local <= state->max_rows && pack_bytes <= cand_cap_bytes &&
			score_floats >= total_blocks && G::kIndexDim <= G::kHidden && ((G::kIndexHeads * G::kIndexDim) & 7u) == 0u;
		if ( use_compact_index )
		{
			uint16_t *pack_buf = (uint16_t *)state->index_candidates;
			uint16_t *keys_buf = (uint16_t *)state->index_gathered;
			float *scores_buf = (float *)((uint8_t *)state->index_gathered + keys_bytes);
			const uint32_t chunk_cap = (uint32_t)(score_floats / total_blocks);
			const uint32_t chunk_rows = chunk_cap < rows ? chunk_cap : rows;
			const float scale = 1.0f / sqrtf((float)G::kIndexDim);
			constexpr uint32_t gscore_id = G::kIndexHeads == 4u ? QWENSKL2_index_gathered_score_h4_256 : QWENSKL2_index_gathered_score_gen_256;
			if ( QwenSkl2Launch(QWENSKL2_index_pack_local_keys_256, max_local, QH_THREADS, 0, stream, *view,
					step->sequence_of_row, total_blocks, max_local, degree > 1u ? pack_buf : keys_buf,
					G::kIndexDim, S::IndexKv::kPageSlots) != SPARK_GPU_SUCCESS ||
				CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			if ( degree > 1u )
			{
				status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, max_local, G::kIndexDim, pack_buf, keys_buf);
				if ( status != SPARK_STATUS_OK )
					return status;
			}
			for ( uint32_t first = 0u; first < rows; first += chunk_rows )
			{
				const uint32_t count = rows - first < chunk_rows ? rows - first : chunk_rows;
				if ( QwenSkl2Launch(gscore_id, dim3((total_blocks + QH_THREADS - 1u) / QH_THREADS, count), QH_THREADS, 0, stream,
						keys_buf, max_local, degree, state->index_query, step->positions, first, count, total_blocks, scores_buf, scale,
						G::kIndexHeads, G::kIndexDim, G::kIndexRatio, S::kIndexBlocks) != SPARK_GPU_SUCCESS ||
					QwenSkl2Launch(QWENSKL2_index_local_mask_from_scores_256, count, QH_THREADS, 0, stream,
						scores_buf, total_blocks, step->positions, first, count, state->index_mask, state->index_mask_words,
						G::kIndexRatio, S::kIndexBlocks) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			return CudaStageLaunched();
		}
		constexpr uint32_t score_id = G::kIndexHeads == 4u ? QWENSKL2_index_score_sharded_h4_256 : QWENSKL2_index_score_sharded_gen_256;
		if ( QwenSkl2Launch(score_id, dim3((state->index_local_stride + QH_THREADS - 1u) / QH_THREADS, rows), QH_THREADS, 0, stream, *view,
				state->index_query, step->sequence_of_row, step->context_length, step->positions, state->index_scores, state->index_candidates,
				state->index_local_stride, 1.0f / sqrtf((float)G::kIndexDim), G::kIndexHeads, G::kIndexDim, G::kIndexRatio, S::kIndexBlocks,
				S::IndexKv::kPageSlots) != SPARK_GPU_SUCCESS ||
			QwenSkl2Launch(QWENSKL2_index_local_topk_pack_256, rows, QH_THREADS, 0, stream, view->shard.degree, view->shard.rank, view->shard.grain,
				state->index_scores, state->index_local_stride, step->sequence_of_row, step->context_length, step->positions,
				state->index_local_selected, state->index_candidates, G::kIndexRatio, S::kIndexBlocks) != SPARK_GPU_SUCCESS ||
			CudaStageLaunched() != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows,
			S::kIndexBlocks * (uint32_t)(sizeof(uint2) / sizeof(uint16_t)), state->index_candidates, state->index_gathered);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( QwenSkl2Launch(QWENSKL2_index_gathered_select_mask_256, rows, QH_THREADS, 0, stream, state->index_gathered,
				(uint32_t)((uint64_t)rows * S::kIndexBlocks), state->tp_degree, step->sequence_of_row, step->context_length, step->positions,
				state->index_mask, state->index_mask_words, G::kIndexRatio, S::kIndexBlocks) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return CudaStageLaunched();
	}

	static SparkStatus Attention(State *state, const Frame &frame, const QwenHybridLayer &entry)
	{
		const QwenHybridLayer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		SparkGpuStream stream = frame.stream;
		const uint32_t rows = step->rows, sms = step->multiprocessors, degree = state->tp_degree;
		const uint64_t units = ((uint64_t)rows * LmGqaShardRecordFloats(G::kHeads, G::kHeadDim, degree) * sizeof(float) +
			G::kHidden * sizeof(uint16_t) - 1u) / (G::kHidden * sizeof(uint16_t));
		const uint64_t stride = units * G::kHidden * sizeof(uint16_t) / sizeof(float);
		const LmKvShardView *view = &state->kv_views[layer->kv_index];
		const uint32_t exact = Exact(step);
		const bool prefill_wave = PrefillWave(step);
		const bool prefill_unified = PrefillUnified(step);
		SparkStatus status;
		if ( state->kv_attached == 0u || stride > state->record_stride )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( Linear(state, layer->mix_weight, layer->mix_scale, state->normed, state->attn_local, rows, G::kHidden,
			state->attn_rows, exact, sms, stream, prefill_wave) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		const uint32_t group = G::kHeads / G::kKvHeads;
		const uint32_t elem_first = state->tp_rank * state->attn_width;
		const uint32_t q_first = elem_first / G::kHeadDim;
		const uint32_t q_last = (elem_first + state->attn_width - 1u) / G::kHeadDim;
		const uint32_t rank_heads = q_last - q_first + 1u;
		const uint32_t kv_first = q_first / group;
		const uint32_t kv_last = q_last / group;
		const uint32_t local_kvh = kv_last - kv_first + 1u;
		bool uniform_kvh = true;
		for ( uint32_t peer = 0u; peer < degree; ++peer )
		{
			const uint32_t p_first = (peer * state->attn_width) / G::kHeadDim;
			const uint32_t p_last = (peer * state->attn_width + state->attn_width - 1u) / G::kHeadDim;
			if ( (p_last / group) - (p_first / group) + 1u != local_kvh )
				uniform_kvh = false;
		}
		const uint32_t grain = view->shard.grain;
		QwenPrefillSeqPlan plan;
		memset(&plan, 0, sizeof(plan));
		bool plan_ok = false;
		uint32_t max_seq_rows = 0u, max_seq_prior = 0u, max_seq_peer = 0u, max_seq_total_ctx = 0u;
		if ( prefill_wave && uniform_kvh && G::kHeadDim == 256u && (G::kHeads % G::kKvHeads) == 0u &&
			step->sequences != 0u && step->sequences <= QH_PREFILL_MAX_SEQS )
		{
			if ( step->sequences == 1u )
			{
				const uint32_t total_ctx = step->gather_context;
				if ( total_ctx >= rows )
				{
					const uint32_t old_ctx = total_ctx - rows;
					const uint32_t b1 = (old_ctx + grain - 1u) / grain;
					const uint32_t peer_keys = old_ctx != 0u ? (((b1 + degree - 1u) / degree) * grain) : 0u;
					plan.count = 1u;
					plan.row_begin[0] = 0u;
					plan.row_len[0] = rows;
					plan.old_ctx[0] = old_ctx;
					plan.prior_off[0] = 0u;
					plan.peer_off[0] = 0u;
					plan.peer_keys[0] = peer_keys;
					plan.total_prior = old_ctx;
					plan.total_peer = peer_keys;
					max_seq_rows = rows;
					max_seq_prior = old_ctx;
					max_seq_peer = peer_keys;
					max_seq_total_ctx = total_ctx;
					plan_ok = true;
				}
			}
			else if ( step->host_positions != 0 && step->host_sequence_of_row != 0 )
			{
				uint32_t r = 0u, s = 0u;
				bool valid = true;
				while ( r < rows && s < step->sequences )
				{
					const uint32_t seq_id = step->host_sequence_of_row[r];
					const uint32_t old_ctx = step->host_positions[r];
					const uint32_t r0 = r;
					++r;
					while ( r < rows && step->host_sequence_of_row[r] == seq_id && step->host_positions[r] == old_ctx + (r - r0) )
						++r;
					const uint32_t len = r - r0;
					const uint32_t total_ctx = old_ctx + len;
					const uint32_t b1 = (old_ctx + grain - 1u) / grain;
					const uint32_t peer_keys = old_ctx != 0u ? (((b1 + degree - 1u) / degree) * grain) : 0u;
					plan.seq_id[s] = seq_id;
					plan.row_begin[s] = r0;
					plan.row_len[s] = len;
					plan.old_ctx[s] = old_ctx;
					plan.prior_off[s] = plan.total_prior;
					plan.peer_off[s] = plan.total_peer;
					plan.peer_keys[s] = peer_keys;
					plan.total_prior += old_ctx;
					plan.total_peer += peer_keys;
					if ( len > max_seq_rows )
						max_seq_rows = len;
					if ( old_ctx > max_seq_prior )
						max_seq_prior = old_ctx;
					if ( peer_keys > max_seq_peer )
						max_seq_peer = peer_keys;
					if ( total_ctx > max_seq_total_ctx )
						max_seq_total_ctx = total_ctx;
					++s;
				}
				if ( r != rows || s != step->sequences )
					valid = false;
				plan.count = s;
				plan_ok = valid;
			}
		}
		const bool host_dense = S::kIndexer && plan_ok && max_seq_total_ctx != 0u && (max_seq_total_ctx / G::kIndexRatio) <= S::kIndexBlocks;
		const uint64_t unit_bytes = (uint64_t)G::kHidden * sizeof(uint16_t);
		const uint64_t peer_bytes = (uint64_t)plan.total_peer * local_kvh * (2u * G::kHeadDim) * sizeof(uint16_t);
		const uint32_t a2a_units = (degree > 1u && plan.total_prior != 0u) ? (uint32_t)((peer_bytes + unit_bytes - 1u) / unit_bytes) : 0u;
		const uint64_t a2a_stride_bytes = (uint64_t)a2a_units * unit_bytes;
		const uint64_t kv_cur_bytes = (uint64_t)local_kvh * rows * (2u * G::kHeadDim) * sizeof(uint16_t);
		const uint64_t prior_bytes = (uint64_t)local_kvh * plan.total_prior * (2u * G::kHeadDim) * sizeof(uint16_t);
		const uint64_t buf_cap_bytes = (uint64_t)degree * state->record_stride * sizeof(float);
		const uint64_t query_cap_elems = (uint64_t)state->max_rows * S::kQWidth;
		const uint64_t query_used_elems = (uint64_t)rows * rank_heads * G::kHeadDim;
		constexpr bool kDirectPaged = G::kTpDegree == 1u && (S::Kv::kPageSlots % 32u) == 0u;
		const bool kv_cur_in_query = degree > 1u && query_used_elems + (kv_cur_bytes / sizeof(uint16_t)) <= query_cap_elems;
		const bool tp1_buf_fits = degree == 1u && (kv_cur_bytes + prior_bytes) <= buf_cap_bytes;
		const bool tp_buf_fits = degree > 1u && kv_cur_in_query && (uint64_t)degree * a2a_stride_bytes <= buf_cap_bytes && prior_bytes <= buf_cap_bytes;
		const bool use_local_prefill = plan_ok && (tp1_buf_fits || tp_buf_fits);
		if ( use_local_prefill )
		{
			constexpr bool kKvOnlyGather = G::kRankHeads && G::kTpDegree > 1u && (2u * S::kKvWidth / G::kTpDegree) <= G::kHidden;
			if constexpr ( G::kTpDegree > 1u )
			{
				if constexpr ( kKvOnlyGather )
				{
					constexpr uint32_t KV_LOCAL = 2u * S::kKvWidth / G::kTpDegree;
					constexpr uint32_t Q2_LOCAL_V = (2u * S::kQWidth / G::kTpDegree) >> 3u;
					constexpr uint32_t KV_LOCAL_V = KV_LOCAL >> 3u;
					if ( QwenSkl2Launch(QWENSKL2_qwen_pack_local_kv_256,
						LmGridBlocks<QH_THREADS>((uint64_t)rows * KV_LOCAL_V), QH_THREADS, 0, stream,
						state->attn_local, state->attn_query, rows, state->attn_rows, Q2_LOCAL_V, KV_LOCAL_V) != SPARK_GPU_SUCCESS )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
					status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, KV_LOCAL,
						state->attn_query, state->attn_gathered);
				}
				else
					status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows * S::kGatherParts,
						state->attn_rows / S::kGatherParts, state->attn_local, state->attn_gathered);
				if ( status != SPARK_STATUS_OK )
					return status;
			}
			uint16_t *kv_cur = degree > 1u ? (state->attn_query + query_used_elems) : (uint16_t *)state->send;
			uint16_t *kv_prior = degree > 1u ? (uint16_t *)state->send : (uint16_t *)((uint8_t *)state->send + kv_cur_bytes);
			CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
			{
				constexpr uint32_t warps = QH_THREADS / LM_WARP_LANES;
				constexpr uint32_t prep_flags = (G::kRankHeads ? 1u : 0u) | (kKvOnlyGather ? 2u : 0u) | (G::kRopeRoundedTerms ? 4u : 0u);
				const uint32_t local_tasks = rank_heads + 2u * G::kKvHeads;
				if ( QwenSkl2Launch(QWENSKL2_qwen_attn_prep_local_gen, dim3((local_tasks + warps - 1u) / warps, rows), QH_THREADS, 0, stream,
					state->attn_local, state->attn_gathered, layer->q_norm, layer->k_norm, step->positions, step->sequence_of_row, *view,
					state->attn_query, state->merged_gate, kv_cur, rows, state->attn_rows, state->tp_rank, state->attn_width,
					q_first, rank_heads, kv_first, local_kvh, G::kHeads, G::kKvHeads, G::kTpDegree, prep_flags) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			if constexpr ( S::kIndexer )
			{
				status = IndexPrepare(state, layer, step, stream, prefill_wave, host_dense);
				CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
				if ( status == SPARK_STATUS_OK )
					status = IndexSelect(state, layer, step, stream, prefill_unified, host_dense);
				if ( status != SPARK_STATUS_OK )
					return status;
			}
			else
				CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
			if ( CudaStageLaunched() != SPARK_STATUS_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			if ( plan.total_prior != 0u )
			{
				if ( degree > 1u )
				{
					const uint32_t peer_blocks = (max_seq_peer + 7u) >> 3u;
					if ( QwenSkl2Launch(QWENSKL2_qwen_prior_kv_pack_gen, dim3(peer_blocks, degree, plan.count), QH_THREADS, 0, stream,
						*view, step->sequence_of_row, plan, state->attn_width, group, local_kvh, (uint64_t)a2a_stride_bytes,
						(uint8_t *)state->send, G::kKvHeads) != SPARK_GPU_SUCCESS )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
					status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, a2a_units, 0u, state->send, state->received);
					if ( status != SPARK_STATUS_OK )
						return status;
					if ( QwenSkl2Launch(QWENSKL2_qwen_prior_kv_unpack_gen, dim3(peer_blocks, degree, plan.count), QH_THREADS, 0, stream,
						view->shard.degree, view->shard.grain, (const uint8_t *)state->received, (uint64_t)a2a_stride_bytes,
						plan, local_kvh, kv_prior) != SPARK_GPU_SUCCESS )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				}
				else if constexpr ( !kDirectPaged )
				{
					if ( QwenSkl2Launch(QWENSKL2_qwen_prior_kv_direct_gen, dim3(max_seq_prior, local_kvh, plan.count), QH_THREADS, 0, stream,
						*view, step->sequence_of_row, plan, kv_first, local_kvh, kv_prior, G::kKvHeads) != SPARK_GPU_SUCCESS )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				}
			}
			constexpr bool kGateFromLocal = G::kTpDegree == 1u || kKvOnlyGather;
			const uint32_t q_blocks = (max_seq_rows + 31u) / 32u;
			const uint16_t *gate_ptr = kGateFromLocal ? state->attn_local : state->merged_gate;
			const uint32_t gate_stride = kGateFromLocal ? state->attn_rows : state->attn_width;
			const dim3 prefill_grid = kDirectPaged ? dim3(group * q_blocks, local_kvh, plan.count) : dim3(rank_heads, q_blocks, plan.count);
			const bool masked = S::kIndexer && !host_dense;
			const uint32_t prefill_flags = (masked ? 1u : 0u) | (kGateFromLocal ? 2u : 0u) | (kDirectPaged ? 4u : 0u);
			if ( QwenSkl2Launch(QWENSKL2_qwen_local_prefill_attn_gen_256, prefill_grid, 128u, 41088u, stream,
				state->attn_query, kv_prior, kv_cur, *view, step->sequence_of_row, gate_ptr, gate_stride, step->positions, plan, rows, rank_heads,
				q_first, group, kv_first, elem_first, state->attn_width, 1.0f / sqrtf((float)G::kHeadDim),
				masked ? state->index_mask : (const uint32_t *)0, masked ? state->index_mask_words : 0u, G::kIndexRatio, S::kIndexBlocks,
				state->merged, G::kKvHeads, prefill_flags) != SPARK_GPU_SUCCESS ||
				Linear(state, layer->out_weight, layer->out_scale, state->merged, state->partial, rows, state->attn_width, G::kHidden, exact, sms,
					stream, prefill_wave) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			return SPARK_STATUS_OK;
		}
		constexpr bool kQkvNoGateGather = G::kRankHeads && G::kTpDegree > 1u &&
			(S::kQWidth % (G::kTpDegree * G::kHeadDim)) == 0u && (S::kKvWidth % (G::kTpDegree * 8u)) == 0u &&
			((S::kQWidth + 2u * S::kKvWidth) / G::kTpDegree) <= G::kHidden;
		if constexpr ( G::kTpDegree > 1u )
		{
			if constexpr ( kQkvNoGateGather )
			{
				constexpr uint32_t QKV_LOCAL = (S::kQWidth + 2u * S::kKvWidth) / G::kTpDegree;
				constexpr uint32_t Q_LOCAL_V = (S::kQWidth / G::kTpDegree) >> 3u;
				constexpr uint32_t KV_LOCAL_V = (2u * S::kKvWidth / G::kTpDegree) >> 3u;
				constexpr uint32_t QKV_LOCAL_V = QKV_LOCAL >> 3u;
				if ( QwenSkl2Launch(QWENSKL2_qwen_pack_local_qkv_no_gate_256,
					LmGridBlocks<QH_THREADS>((uint64_t)rows * QKV_LOCAL_V), QH_THREADS, 0, stream,
					state->attn_local, state->attn_query, rows, state->attn_rows, Q_LOCAL_V, KV_LOCAL_V) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows, QKV_LOCAL,
					state->attn_query, state->attn_gathered);
			}
			else
				status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, rows * S::kGatherParts, state->attn_rows / S::kGatherParts,
					state->attn_local, state->attn_gathered);
			if ( status != SPARK_STATUS_OK )
				return status;
		}
		CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
		{
			constexpr uint32_t tasks = G::kHeads + 2u * G::kKvHeads;
			constexpr uint32_t warps = QH_THREADS / LM_WARP_LANES;
			constexpr uint32_t prep_flags = (G::kRankHeads ? 1u : 0u) | (kQkvNoGateGather ? 2u : 0u) | (G::kRopeRoundedTerms ? 4u : 0u);
			if ( QwenSkl2Launch(QWENSKL2_qwen_hybrid_attn_prep_gen, dim3((tasks + warps - 1u) / warps, rows), QH_THREADS, 0, stream,
				state->attn_gathered, layer->q_norm, layer->k_norm, step->positions, step->tree != 0u ? step->slots : step->positions,
				step->sequence_of_row, *view, state->attn_query, state->merged_gate, rows, state->attn_rows, state->tp_rank, state->attn_width,
				G::kHeads, G::kKvHeads, G::kTpDegree, prep_flags) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		if constexpr ( S::kIndexer )
		{
			status = IndexPrepare(state, layer, step, stream, prefill_wave, host_dense);
			CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
			if ( status == SPARK_STATUS_OK )
				status = IndexSelect(state, layer, step, stream, prefill_unified, host_dense);
			if ( status != SPARK_STATUS_OK )
				return status;
		}
		else
			CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
		constexpr bool kSplitGqaCapable = G::kTpDegree == 1u && !S::kIndexer && G::kHeadDim == 256u && (G::kHeads % G::kKvHeads) == 0u;
		bool split_done = false;
		if constexpr ( kSplitGqaCapable )
		{
			constexpr uint32_t GROUP = G::kHeads / G::kKvHeads;
			constexpr uint32_t GQA_BLOCK = GROUP <= 4u ? 128u : (GROUP <= 6u ? 192u : 512u);
			constexpr uint32_t SPLIT_ID = GROUP <= 4u ? QWENSKL2_qwen_gqa_split_decode_gen_128 :
				(GROUP <= 6u ? QWENSKL2_qwen_gqa_split_decode_gen_192 : QWENSKL2_qwen_gqa_split_decode_gen_512);
			constexpr uint32_t DIRECT_ID = GROUP <= 4u ? QWENSKL2_qwen_gqa_direct_decode_gate_gen_128 :
				(GROUP <= 6u ? QWENSKL2_qwen_gqa_direct_decode_gate_gen_192 : QWENSKL2_qwen_gqa_direct_decode_gate_gen_512);
			const uint64_t per_row_floats = (uint64_t)QH_GQA_MAX_SPLITS * G::kHeads * (G::kHeadDim + 4u);
			const uint32_t chunk_rows = (uint32_t)(state->split_gqa_floats / per_row_floats);
			if ( !prefill_wave && chunk_rows > 0u && (rows <= 16u || (step->commit == 0u && step->sequences < QH_VERIFY_INVARIANT_SEQS)) )
			{
				constexpr uint32_t warps = QH_THREADS / LM_WARP_LANES;
				for ( uint32_t first = 0u; first < rows; first += chunk_rows )
				{
					const uint32_t count = rows - first < chunk_rows ? rows - first : chunk_rows;
					if ( QwenSkl2Launch(SPLIT_ID, dim3(count, G::kKvHeads, QH_GQA_MAX_SPLITS), GQA_BLOCK, 0, stream,
						*view, state->attn_query + (uint64_t)first * G::kHeads * G::kHeadDim,
						state->attn_local + (uint64_t)first * S::kQkvWidth,
						step->sequence_of_row + first, step->context_length, step->positions + first,
						step->tree != 0u ? step->fork_positions + first : (const uint32_t *)0,
						step->tree != 0u ? step->slot_offsets + first : (const uint32_t *)0,
						1.0f / sqrtf((float)G::kHeadDim), state->send,
						state->merged + (uint64_t)first * G::kHeads * G::kHeadDim, count,
						G::kHeads, G::kKvHeads, (uint32_t)(S::kQkvWidth >> 3u)) != SPARK_GPU_SUCCESS ||
						QwenSkl2Launch(QWENSKL2_qwen_gqa_split_reduce_gate_gen, dim3((G::kHeads + warps - 1u) / warps, count), QH_THREADS, 0, stream,
						state->send, state->attn_local + (uint64_t)first * S::kQkvWidth,
						step->sequence_of_row + first, step->context_length, step->positions + first,
						state->merged + (uint64_t)first * G::kHeads * G::kHeadDim, count,
						G::kHeads, (uint32_t)(S::kQkvWidth >> 3u)) != SPARK_GPU_SUCCESS )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				}
				split_done = true;
			}
			else if ( !prefill_wave && step->tree == 0u && rows == step->sequences && rows > 16u )
			{
				if ( QwenSkl2Launch(DIRECT_ID, dim3(rows, G::kKvHeads), GQA_BLOCK, 0, stream,
					*view, state->attn_query, state->attn_local, step->sequence_of_row, step->context_length, step->positions,
					1.0f / sqrtf((float)G::kHeadDim), state->merged, rows,
					G::kHeads, G::kKvHeads, (uint32_t)(S::kQkvWidth >> 3u)) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				split_done = true;
			}
		}
		if ( !split_done )
		{
			if ( CudaStageLaunched() != SPARK_STATUS_OK ||
				LmGqaShardPartialLaunch<typename S::Kv,LmKvShardView,G::kKvHeads,G::kHeadDim,G::kHeadDim>(*view, state->attn_query, G::kHeads,
					step->sequence_of_row, step->context_length, step->positions, 1.0f / sqrtf((float)G::kHeadDim), state->send, stride, rows, stream,
					(S::kIndexer && !host_dense) ? state->index_mask : (const uint32_t *)0, state->index_mask_words, G::kIndexRatio, 0u,
					step->tree != 0u ? step->fork_positions : (const uint32_t *)0,
					step->tree != 0u ? step->slot_offsets : (const uint32_t *)0) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			if constexpr ( G::kTpDegree > 1u )
			{
				status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL, (uint32_t)units, 0u, state->send, state->received);
				if ( status != SPARK_STATUS_OK )
					return status;
				if ( QwenSkl2Launch(QWENSKL2_qwen_hybrid_multi_merge_gate_gen, rows, QH_THREADS, 0, stream,
					state->received, (uint64_t)stride, degree, G::kHeads, state->tp_rank,
					kQkvNoGateGather ? state->attn_local : state->merged_gate, state->merged, rows,
					kQkvNoGateGather ? state->attn_rows : 0u, (uint32_t)(kQkvNoGateGather ? 1u : 0u)) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			else
			{
				constexpr uint32_t warps = QH_THREADS / LM_WARP_LANES;
				if ( QwenSkl2Launch(QWENSKL2_qwen_hybrid_tp1_merge_gate_gen, dim3((G::kHeads + warps - 1u) / warps, rows), QH_THREADS, 0, stream,
					state->send, state->attn_local, state->merged, rows, G::kHeads, (uint32_t)S::kQkvWidth) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
		}
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			Linear(state, layer->out_weight, layer->out_scale, state->merged, state->partial, rows, state->attn_width, G::kHidden, exact, sms,
				stream, prefill_wave) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static LmLocalExpertRoutes ExpertRoutes(const State *state, uint32_t rows, uint32_t routes, uint32_t sms)
	{
		LmLocalExpertRoutes expert_routes;
		expert_routes.group_row_offset = state->group_row_offset;
		expert_routes.route_source_token = state->route_source_token;
		expert_routes.tile_prefix_up = state->tile_prefix_up;
		expert_routes.tile_prefix_down = state->tile_prefix_down;
		expert_routes.rows = rows;
		expert_routes.routes = routes;
		expert_routes.top_k = G::kTopK;
		expert_routes.local_experts = G::kLocalExperts;
		expert_routes.multiprocessors = sms;
		return expert_routes;
	}

	static SparkStatus Moe(State *state, const QwenHybridLayer *layer, const SparkStageRunnerStep *step, uint32_t first, uint32_t rows,
		SparkGpuStream stream)
	{
		const uint32_t sms = step->multiprocessors, routes = rows * G::kTopK, exact = Exact(step);
		const bool prefill_wave = PrefillWave(step);
		const uint16_t *normed = state->normed + (uint64_t)first * G::kHidden;
		using ExpertFormat = typename LmWeightCodec<G::kExpertCodec>::Format;
		const LmScaleTensor w2_scale = LmWeightCodecScaleTensor<G::kExpertCodec>(layer->moe_w2_scale, G::kLocalExperts, G::kHidden, G::kMoeInter);
		const LmLocalExpertRoutes expert_routes = ExpertRoutes(state, rows, routes, sms);
		const bool multi_shared_done = !prefill_wave && (rows <= LM_SKINNY_ROWS_WIDE || exact != 0u);
		int32_t launch;
		if ( multi_shared_done )
		{
			const LmSkinnyDenseTarget targets[3] =
			{
				{ layer->moe_router, state->router_logits, (float *)0, G::kExperts, G::kExperts, 0u, LmScaleTensorNone() },
				{ layer->moe_shared_gate_up, state->shared_gate_up, (float *)0, 2u * G::kSharedRows, 2u * G::kSharedRows, 0u, LmScaleTensorNone() },
				{ layer->moe_shared_gate, state->shared_logit, (float *)0, 1u, 1u, 0u, LmScaleTensorNone() }
			};
			if ( LmSkinnyDenseMulti<LmBf16Format>(targets, 3u, normed, rows, G::kHidden, stream, exact) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else if ( Project<LmBf16Format>(layer->moe_router, 0, normed, state->router_logits, rows, G::kHidden, G::kExperts, exact, sms, stream, prefill_wave) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if constexpr ( G::kExperts <= LM_TOPK_WARP_PER_LANE * LM_WARP_LANES )
		{
			const uint32_t tile_m = LmLaunchGroupedTileM(rows, G::kTopK, G::kLocalExperts + 1u);
			const uint32_t n_up = (G::kMoeInter + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
			const uint32_t n_down = (G::kHidden + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
			if ( routes <= 32u )
			{
				if ( QwenSkl2Launch(QWENSKL2_moe_route_small_256, 1u, QH_THREADS, 0, stream,
					state->router_logits, rows, G::kExperts, G::kTopK, state->expert_first, G::kLocalExperts,
					step->route_fault, state->route_expert, state->route_weight, state->route_global, state->group_row_offset,
					state->route_packed_row, state->route_source_token, state->packed_expert, state->packed_weight, tile_m, n_up,
					state->tile_prefix_up, n_down, state->tile_prefix_down) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			else
			{
				if ( QwenSkl2Launch(QWENSKL2_moe_softmax_topk_local_256, rows, QH_THREADS, 0, stream,
					state->router_logits, G::kExperts, G::kTopK, state->expert_first, G::kLocalExperts,
					step->route_fault, state->route_expert, state->route_weight, state->route_global) != SPARK_GPU_SUCCESS ||
					QwenSkl2Launch(QWENSKL2_moe_route_build_packed_256, 1u, QH_THREADS, 0, stream,
					state->route_expert, state->route_weight, routes, G::kTopK, G::kLocalExperts + 1u,
					state->group_row_offset, state->route_packed_row, state->route_source_token,
					state->packed_expert, state->packed_weight, tile_m, n_up, state->tile_prefix_up,
					n_down, state->tile_prefix_down) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
		}
		else
		{
			LM_LAUNCH((LmRouterSoftmaxKernel<QH_THREADS,G::kExperts>), rows, QH_THREADS, 0, stream, state->router_logits, state->router_probs);
			if ( LmTopkRouteLaunch<QH_THREADS,G::kTopK,true,LM_TOPK_SCORE_IDENTITY>(rows, state->router_probs, G::kExperts, state->route_expert,
				state->route_weight, (const float *)0, (const uint16_t *)0, 1.0f, step->route_fault, stream) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			LM_LAUNCH((LmRouteLocalKernel), (routes + 255u) / 256u, 256u, 0, stream, state->route_expert, state->route_weight, state->route_global,
				routes, state->expert_first, G::kLocalExperts);
			const uint32_t tile_m = LmLaunchGroupedTileM(rows, G::kTopK, G::kLocalExperts + 1u);
			const uint32_t n_up = (G::kMoeInter + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
			const uint32_t n_down = (G::kHidden + LM_MOE_GEMM_TILE_N - 1u) / LM_MOE_GEMM_TILE_N;
			if ( QwenSkl2Launch(QWENSKL2_moe_route_build_packed_256, 1u, QH_THREADS, 0, stream,
				state->route_expert, state->route_weight, routes, G::kTopK, G::kLocalExperts + 1u,
				state->group_row_offset, state->route_packed_row, state->route_source_token,
				state->packed_expert, state->packed_weight, tile_m, n_up, state->tile_prefix_up,
				n_down, state->tile_prefix_down) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		if constexpr ( G::kExpertCodec == SPARK_WEIGHT_CODEC_NVFP4_E2M1 && (G::kHidden % 128u) == 0u && (G::kMoeInter % 128u) == 0u )
		{
			constexpr uint32_t UP_CHUNKS = G::kHidden >> 5u;
			constexpr uint32_t UP_LANES = UP_CHUNKS >= 32u ? 32u : (UP_CHUNKS >= 16u ? 16u : 8u);
			constexpr uint32_t UP_PER_BLK = QH_THREADS / UP_LANES;
			constexpr uint32_t DN_CHUNKS = G::kMoeInter >> 5u;
			constexpr uint32_t DN_LANES = DN_CHUNKS >= 32u ? 32u : (DN_CHUNKS >= 16u ? 16u : 8u);
			constexpr uint32_t DN_PER_BLK = QH_THREADS / DN_LANES;
			constexpr uint32_t DN_SINGLE_PER_BLK = DN_PER_BLK * 4u;
			const uint32_t local_mean = (routes + G::kExperts - 1u) / G::kExperts;
			const bool use_mma = prefill_wave || (exact == 0u && rows > LM_SKINNY_ROWS_WIDE && (G::kHidden >= 4096u || local_mean > 2u));
			if ( use_mma )
			{
				const uint32_t up_blocks = G::kLocalExperts * ((G::kMoeInter + 63u) / 64u);
				const uint32_t dn_blocks = G::kLocalExperts * ((G::kHidden + 127u) / 128u);
				const uint32_t up_id = rows <= 128u ? QWENSKL2_moe_nvfp4_w1w3_m16 : (rows < 512u ? QWENSKL2_moe_nvfp4_w1w3_m32 : QWENSKL2_moe_nvfp4_w1w3_m48);
				const uint32_t dn_id = rows <= 128u ? QWENSKL2_moe_nvfp4_w2_m16 : (rows < 512u ? QWENSKL2_moe_nvfp4_w2_m32 : QWENSKL2_moe_nvfp4_w2_m48);
				if ( QwenSkl2Launch(up_id, up_blocks, 256u, 0, stream,
					layer->moe_w1, layer->moe_w1_scale, layer->moe_w3, layer->moe_w3_scale, normed, state->packed_weight, state->expert_act,
					state->group_row_offset, state->route_source_token, G::kLocalExperts, G::kHidden, G::kMoeInter) != SPARK_GPU_SUCCESS ||
					QwenSkl2Launch(dn_id, dn_blocks, 256u, 0, stream,
					layer->moe_w2, layer->moe_w2_scale, (const uint8_t *)0, (const uint8_t *)0, state->expert_act, (const float *)0,
					state->expert_out, state->group_row_offset, (const uint32_t *)0, G::kLocalExperts, G::kMoeInter, G::kHidden) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			else if ( rows == 1u )
			{
				constexpr uint32_t up_id = UP_LANES == 32u ? QWENSKL2_moe_nvfp4_skinny_w1w3_single_l32 :
					(UP_LANES == 16u ? QWENSKL2_moe_nvfp4_skinny_w1w3_single_l16 : QWENSKL2_moe_nvfp4_skinny_w1w3_single_l8);
				constexpr uint32_t dn_id = DN_LANES == 32u ? QWENSKL2_moe_nvfp4_skinny_w2_single_l32 :
					(DN_LANES == 16u ? QWENSKL2_moe_nvfp4_skinny_w2_single_l16 : QWENSKL2_moe_nvfp4_skinny_w2_single_l8);
				if ( QwenSkl2Launch(up_id, dim3((G::kMoeInter + UP_PER_BLK - 1u) / UP_PER_BLK, routes), QH_THREADS, 0, stream,
					layer->moe_w1, layer->moe_w1_scale, layer->moe_w3, layer->moe_w3_scale, normed, state->packed_weight,
					state->group_row_offset, state->packed_expert, state->expert_act, G::kHidden, G::kMoeInter, G::kLocalExperts) != SPARK_GPU_SUCCESS ||
					QwenSkl2Launch(dn_id, dim3((G::kHidden + DN_SINGLE_PER_BLK - 1u) / DN_SINGLE_PER_BLK, routes), QH_THREADS, 0, stream,
					layer->moe_w2, layer->moe_w2_scale, state->expert_act, state->group_row_offset, state->packed_expert,
					state->expert_out, G::kMoeInter, G::kHidden, G::kLocalExperts) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			else
			{
				constexpr uint32_t up_id = UP_LANES == 32u ? QWENSKL2_moe_nvfp4_skinny_w1w3_batched_l32 :
					(UP_LANES == 16u ? QWENSKL2_moe_nvfp4_skinny_w1w3_batched_l16 : QWENSKL2_moe_nvfp4_skinny_w1w3_batched_l8);
				constexpr uint32_t dn_id = DN_LANES == 32u ? QWENSKL2_moe_nvfp4_skinny_w2_batched_l32 :
					(DN_LANES == 16u ? QWENSKL2_moe_nvfp4_skinny_w2_batched_l16 : QWENSKL2_moe_nvfp4_skinny_w2_batched_l8);
				if ( QwenSkl2Launch(up_id, dim3((G::kMoeInter + UP_PER_BLK - 1u) / UP_PER_BLK, G::kLocalExperts), QH_THREADS, 0, stream,
					layer->moe_w1, layer->moe_w1_scale, layer->moe_w3, layer->moe_w3_scale, normed, state->packed_weight,
					state->group_row_offset, state->route_source_token, state->expert_act, G::kHidden, G::kMoeInter, G::kLocalExperts) != SPARK_GPU_SUCCESS ||
					QwenSkl2Launch(dn_id, dim3((G::kHidden + DN_SINGLE_PER_BLK - 1u) / DN_SINGLE_PER_BLK, G::kLocalExperts), QH_THREADS, 0, stream,
					layer->moe_w2, layer->moe_w2_scale, state->expert_act, state->group_row_offset, state->expert_out,
					G::kMoeInter, G::kHidden, G::kLocalExperts) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
		}
		else if constexpr ( G::kExpertCodec == SPARK_WEIGHT_CODEC_BF16 && (G::kHidden & 63u) == 0u && (G::kMoeInter & 63u) == 0u && G::kLocalExperts <= 256u )
		{
			const uint32_t local_mean = (routes + G::kExperts - 1u) / G::kExperts;
			const bool use_mma = prefill_wave || (exact == 0u && rows > LM_SKINNY_ROWS_WIDE);
			const uint16_t *w1_bf16 = (const uint16_t *)layer->moe_w1;
			const uint16_t *w3_bf16 = (const uint16_t *)layer->moe_w3;
			const uint16_t *w2_bf16 = (const uint16_t *)layer->moe_w2;
			if ( use_mma )
			{
				const uint32_t up_id = local_mean <= 16u ? QWENSKL2_moe_bf16_mma_w1w3_16x64 :
					(local_mean <= 32u ? QWENSKL2_moe_bf16_mma_w1w3_32x64 : QWENSKL2_moe_bf16_mma_w1w3_64x64);
				const uint32_t dn_id = local_mean <= 16u ? QWENSKL2_moe_bf16_mma_w2_16x64 :
					(local_mean <= 32u ? QWENSKL2_moe_bf16_mma_w2_32x64 : QWENSKL2_moe_bf16_mma_w2_64x128);
				const uint32_t up_blk = local_mean <= 16u ? sms * 4u : sms * 2u;
				const uint32_t dn_blk = local_mean <= 16u ? sms * 3u : sms * 2u;
				const uint32_t thr = local_mean <= 16u ? 128u : 256u;
				if ( QwenSkl2Launch(up_id, up_blk, thr, 0, stream,
					w1_bf16, w3_bf16, normed, state->packed_weight, state->group_row_offset, state->route_source_token,
					state->expert_act, G::kLocalExperts, G::kHidden, G::kMoeInter) != SPARK_GPU_SUCCESS ||
					QwenSkl2Launch(dn_id, dn_blk, thr, 0, stream,
					w2_bf16, state->expert_act, state->group_row_offset, state->expert_out, G::kLocalExperts, G::kMoeInter, G::kHidden) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			else
			{
				constexpr uint32_t W1_BLOCKS = (G::kMoeInter + 3u) / 4u;
				constexpr uint32_t W2_CHUNKS = G::kMoeInter >> 3u;
				constexpr uint32_t W2_LANES = (W2_CHUNKS >= 32u && (W2_CHUNKS % 32u) == 0u) ? 32u : (W2_CHUNKS >= 16u ? 16u : 8u);
				constexpr uint32_t W2_GROUPS = 128u / W2_LANES;
				constexpr uint32_t W2_BLOCKS = (G::kHidden + W2_GROUPS - 1u) / W2_GROUPS;
				if ( rows == 1u )
				{
					constexpr uint32_t w2_id = W2_LANES == 32u ? QWENSKL2_moe_bf16_skinny_w2_single_l32 :
						(W2_LANES == 16u ? QWENSKL2_moe_bf16_skinny_w2_single_l16 : QWENSKL2_moe_bf16_skinny_w2_single_l8);
					if ( QwenSkl2Launch(QWENSKL2_moe_bf16_skinny_w1w3_single, dim3(W1_BLOCKS, routes), 128u, 0, stream,
						w1_bf16, w3_bf16, normed, state->packed_weight, state->group_row_offset, state->packed_expert,
						state->expert_act, G::kHidden, G::kMoeInter, G::kLocalExperts) != SPARK_GPU_SUCCESS ||
						QwenSkl2Launch(w2_id, dim3(W2_BLOCKS, routes), 128u, 0, stream,
						w2_bf16, state->expert_act, state->group_row_offset, state->packed_expert, state->expert_out,
						G::kMoeInter, G::kHidden, G::kLocalExperts) != SPARK_GPU_SUCCESS )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				}
				else
				{
					constexpr uint32_t w2_id = W2_LANES == 32u ? QWENSKL2_moe_bf16_skinny_w2_batched_l32 :
						(W2_LANES == 16u ? QWENSKL2_moe_bf16_skinny_w2_batched_l16 : QWENSKL2_moe_bf16_skinny_w2_batched_l8);
					if ( QwenSkl2Launch(QWENSKL2_moe_bf16_skinny_w1w3_batched, dim3(W1_BLOCKS, G::kLocalExperts), 128u, 0, stream,
						w1_bf16, w3_bf16, normed, state->packed_weight, state->group_row_offset, state->route_source_token,
						state->expert_act, G::kHidden, G::kMoeInter, G::kLocalExperts) != SPARK_GPU_SUCCESS ||
						QwenSkl2Launch(w2_id, dim3(W2_BLOCKS, G::kLocalExperts), 128u, 0, stream,
						w2_bf16, state->expert_act, state->group_row_offset, state->expert_out,
						G::kMoeInter, G::kHidden, G::kLocalExperts) != SPARK_GPU_SUCCESS )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				}
			}
		}
		else
		{
			const bool use_stream = prefill_wave || (exact == 0u && rows > LM_SKINNY_ROWS_WIDE);
			const LmScaleTensor w1_scale = LmWeightCodecScaleTensor<G::kExpertCodec>(layer->moe_w1_scale, G::kLocalExperts, G::kMoeInter, G::kHidden);
			const LmScaleTensor w3_scale = LmWeightCodecScaleTensor<G::kExpertCodec>(layer->moe_w3_scale, G::kLocalExperts, G::kMoeInter, G::kHidden);
			if ( use_stream )
			{
				launch = LmStreamGemmGrouped<ExpertFormat>(layer->moe_w1, w1_scale, normed, state->expert_gate, state->group_row_offset,
					state->route_source_token, G::kLocalExperts, routes, G::kHidden, G::kMoeInter, sms, stream);
				if ( launch == LM_LAUNCH_OK )
					launch = LmStreamGemmGrouped<ExpertFormat>(layer->moe_w3, w3_scale, normed, state->expert_up, state->group_row_offset,
						state->route_source_token, G::kLocalExperts, routes, G::kHidden, G::kMoeInter, sms, stream);
			}
			else
			{
				launch = LmLocalExpertsUp<ExpertFormat>(expert_routes, layer->moe_w1, w1_scale, normed, state->expert_gate, G::kHidden, G::kMoeInter, stream, exact);
				if ( launch == LM_LAUNCH_OK )
					launch = LmLocalExpertsUp<ExpertFormat>(expert_routes, layer->moe_w3, w3_scale, normed, state->expert_up, G::kHidden, G::kMoeInter, stream, exact);
			}
			if ( launch != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			if ( QwenSkl2Launch(QWENSKL2_moe_swiglu_packed_256, dim3(LmGridBlocks<QH_THREADS>(G::kMoeInter), routes), QH_THREADS, 0, stream,
				state->expert_gate, state->expert_up, state->packed_weight, state->group_row_offset, G::kLocalExperts, state->expert_act, G::kMoeInter) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			if ( use_stream )
			{
				if ( LmStreamGemmGrouped<ExpertFormat>(layer->moe_w2, w2_scale, state->expert_act, state->expert_out, state->group_row_offset,
					(const uint32_t *)0, G::kLocalExperts, routes, G::kMoeInter, G::kHidden, sms, stream) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			else if ( LmLocalExpertsDown<ExpertFormat>(expert_routes, layer->moe_w2, w2_scale, state->expert_act, state->expert_out, G::kMoeInter, G::kHidden,
				stream, exact) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		bool shared_act_ready = false;
		if ( !multi_shared_done )
		{
			if constexpr ( G::kSharedRows <= 256u && (G::kHidden % 256u) == 0u )
			{
				const int32_t sh_st = rows > 16u ?
					LaunchSharedGateUpSwigluTile<32u>(layer->moe_shared_gate_up, normed, state->shared_act, rows, G::kHidden, G::kSharedRows, stream) :
					LaunchSharedGateUpSwigluTile<16u>(layer->moe_shared_gate_up, normed, state->shared_act, rows, G::kHidden, G::kSharedRows, stream);
				if ( sh_st != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				shared_act_ready = true;
			}
			else if ( Project<LmBf16Format>(layer->moe_shared_gate_up, 0, normed, state->shared_gate_up, rows, G::kHidden, 2u * G::kSharedRows, exact, sms, stream, prefill_wave) !=
				LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			if ( QwenSkl2Launch(QWENSKL2_stage_row_dot_pdl_bf16x8, rows, QH_THREADS, 0, stream, normed, layer->moe_shared_gate, state->shared_logit, G::kHidden) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		if constexpr ( G::kSharedRows <= 128u && (G::kSharedRows & 7u) == 0u && G::kTopK <= LM_MOE_LOCAL_MAX_TOPK )
		{
			if ( multi_shared_done )
			{
				constexpr uint32_t S_CHUNKS = G::kSharedRows >> 3u;
				constexpr uint32_t S_LANES = S_CHUNKS >= 16u ? 16u : 8u;
				constexpr uint32_t S_GROUPS = 128u / S_LANES;
				constexpr uint32_t S_BLOCKS = (G::kHidden + S_GROUPS - 1u) / S_GROUPS;
				constexpr uint32_t sh_id = S_LANES == 16u ? QWENSKL2_moe_shared_down_finalize_l16 : QWENSKL2_moe_shared_down_finalize_l8;
				if ( QwenSkl2Launch(sh_id, dim3(S_BLOCKS, rows), 128u, 0, stream,
					state->expert_out, state->route_packed_row, state->route_expert, state->shared_gate_up,
					layer->moe_shared_down, state->shared_logit, state->partial + (uint64_t)first * G::kHidden,
					rows, G::kHidden, G::kSharedRows, G::kTopK, G::kLocalExperts) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				return SPARK_STATUS_OK;
			}
		}
		if ( !shared_act_ready &&
			QwenSkl2Launch(QWENSKL2_swiglu_pdl_exact_bf16x8, rows, QH_THREADS, 0, stream, state->shared_gate_up, state->shared_act, G::kSharedRows) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if ( CudaStageLaunched() != SPARK_STATUS_OK ||
			Project<LmBf16Format>(layer->moe_shared_down, 0, state->shared_act, state->shared_out, rows, G::kSharedRows, G::kHidden, exact, sms, stream, prefill_wave) !=
				LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if ( QwenSkl2Launch(QWENSKL2_moe_local_finalize_gated_256, dim3(LmVecBlocks<QH_THREADS>(G::kHidden), rows), QH_THREADS, 0, stream,
			state->expert_out, state->route_packed_row, state->route_expert, state->shared_out, state->shared_logit,
			state->partial + (uint64_t)first * G::kHidden, G::kTopK, G::kLocalExperts, G::kHidden) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Ffn(State *state, const Frame &frame, const QwenHybridLayer &entry)
	{
		const QwenHybridLayer *layer = &entry;
		const SparkStageRunnerStep *step = frame.step;
		SparkGpuStream stream = frame.stream;
		if constexpr ( G::kMoe )
			return Moe(state, layer, step, 0u, step->rows, stream);
		else
		{
			const uint32_t rows = step->rows, sms = step->multiprocessors, exact = Exact(step);
			const bool prefill_wave = PrefillWave(step);
			bool fused_done = false;
			if ( state->linear_bytes == 2u && G::kHidden >= 256u && (G::kHidden & 7u) == 0u && (state->ffn_rows & 1u) == 0u )
			{
				const int32_t status = LaunchFusedGateUpSwiglu((const uint16_t *)layer->ffn_gate_up, state->normed, state->intermediate,
					rows, G::kHidden, state->ffn_rows, exact, prefill_wave, stream);
				if ( status == LM_LAUNCH_OK )
					fused_done = true;
				else if ( status != LM_LAUNCH_ERR_SHAPE )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			if ( !fused_done )
			{
				if ( Linear(state, layer->ffn_gate_up, layer->ffn_gate_up_scale, state->normed, state->gate_up, rows, G::kHidden,
					2u * state->ffn_rows, exact, sms, stream, prefill_wave) != LM_LAUNCH_OK ||
					QwenSkl2Launch(QWENSKL2_swiglu_pdl_exact_bf16x8, rows, QH_THREADS, 0, stream, state->gate_up, state->intermediate, state->ffn_rows) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			if ( CudaStageLaunched() != SPARK_STATUS_OK ||
				Linear(state, layer->ffn_down, layer->ffn_down_scale, state->intermediate, state->partial, rows, state->ffn_rows, G::kHidden, exact, sms,
					stream, prefill_wave) != LM_LAUNCH_OK )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			return SPARK_STATUS_OK;
		}
	}

	static SparkStatus DumpLayer(State *state, uint32_t layer, const SparkStageRunnerStep *step, const uint16_t *hidden,
		const uint16_t *partial, SparkGpuStream stream)
	{
		const uint32_t width = S::kHyper && layer != 0xffffffffu ? S::kResidual : G::kHidden;
		const uint32_t header[4] = { S::kHyper ? 0x484c5751u : 0x444c5751u, layer, step->rows, width };
		const uint64_t values = (uint64_t)step->rows * width, partials = (uint64_t)step->rows * G::kHidden;
		if ( SparkGpuStreamSynchronize(stream) != SPARK_GPU_SUCCESS ||
			SparkGpuMemcpy(state->dump_positions, step->positions, (size_t)step->rows * sizeof(uint32_t), SPARK_GPU_COPY_DEVICE_TO_HOST) != SPARK_GPU_SUCCESS ||
			SparkGpuMemcpy(state->dump_rows, hidden, (size_t)values * sizeof(uint16_t), SPARK_GPU_COPY_DEVICE_TO_HOST) != SPARK_GPU_SUCCESS ||
			(partial != 0 && SparkGpuMemcpy(state->dump_rows + values, partial, (size_t)partials * sizeof(uint16_t), SPARK_GPU_COPY_DEVICE_TO_HOST) != SPARK_GPU_SUCCESS) )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		if ( partial == 0 )
			memset(state->dump_rows + values, 0, (size_t)partials * sizeof(uint16_t));
		fwrite(header, sizeof(uint32_t), S::kHyper ? 4u : 3u, state->dump);
		fwrite(state->dump_positions, sizeof(uint32_t), step->rows, state->dump);
		if ( S::kHyper && layer == 0xffffffffu )
		{
			if ( step->token_ids == 0 ||
				SparkGpuMemcpy(state->dump_positions, step->token_ids, (size_t)step->rows * sizeof(uint32_t), SPARK_GPU_COPY_DEVICE_TO_HOST) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
			fwrite(state->dump_positions, sizeof(uint32_t), step->rows, state->dump);
		}
		fwrite(state->dump_rows, sizeof(uint16_t), (size_t)(values + partials), state->dump);
		fflush(state->dump);
		return SPARK_STATUS_OK;
	}

	static SparkStatus HcPre(State *state, const float *norm, const uint16_t *down, const uint16_t *up, const uint16_t *inject,
		uint16_t *mixed, uint32_t rows, uint32_t exact, uint32_t sms, SparkGpuStream stream, bool prefill_unified = false)
	{
		if constexpr ( S::kHyper )
		{
			const bool skinny = !prefill_unified && (rows <= LM_SKINNY_ROWS_WIDE || exact != 0u);
			if ( state->hc_pending_inject != 0u )
			{
				state->hc_pending_inject = 0u;
				constexpr uint32_t in_id = G::kHidden <= 2048u ? QWENSKL2_hc_inject_norm_256_w2048 :
					(G::kHidden <= 4096u ? QWENSKL2_hc_inject_norm_256_w4096 : QWENSKL2_hc_inject_norm_256_w8192);
				if ( QwenSkl2Launch(in_id, dim3(G::kStreams, rows), QH_THREADS, 0, stream,
					state->residual, state->partial, state->hc_inject, norm, state->hc_normed, G::kStreams, G::kHidden, QH_EPSILON) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			else
				LM_LAUNCH((LmGroupRmsNormKernel<QH_THREADS>), dim3(G::kStreams, rows), QH_THREADS, 0, stream, state->residual, norm, state->hc_normed,
					G::kStreams, G::kHidden, QH_EPSILON);
			if ( skinny )
			{
				constexpr uint32_t HC_THREADS = S::kResidual >= 2048u ? 256u : (S::kResidual >= 1024u ? 128u : 32u);
				const uint32_t n_blocks = G::kHcLowRank + (inject != 0 ? G::kStreams : 0u);
				const uint32_t rg_dn = rows == 1u ? 1u : (rows <= 4u ? 4u : 8u);
				const uint32_t dn_id = HC_THREADS == 256u ?
					(rg_dn == 1u ? QWENSKL2_hc_down_gate_inject_skinny_t256_r1 : (rg_dn == 4u ? QWENSKL2_hc_down_gate_inject_skinny_t256_r4 : QWENSKL2_hc_down_gate_inject_skinny_t256_r8)) :
					(HC_THREADS == 128u ?
					(rg_dn == 1u ? QWENSKL2_hc_down_gate_inject_skinny_t128_r1 : (rg_dn == 4u ? QWENSKL2_hc_down_gate_inject_skinny_t128_r4 : QWENSKL2_hc_down_gate_inject_skinny_t128_r8)) :
					(rg_dn == 1u ? QWENSKL2_hc_down_gate_inject_skinny_t32_r1 : (rg_dn == 4u ? QWENSKL2_hc_down_gate_inject_skinny_t32_r4 : QWENSKL2_hc_down_gate_inject_skinny_t32_r8)));
				if ( QwenSkl2Launch(dn_id, n_blocks * ((rows + rg_dn - 1u) / rg_dn), HC_THREADS, 0, stream,
					down, inject, state->hc_normed, state->hc_low, inject != 0 ? state->hc_inject : (float *)0,
					rows, G::kStreams, G::kHidden, G::kHcLowRank) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				constexpr uint32_t CHUNKS = G::kHcLowRank >> 3u;
				constexpr uint32_t LANES = CHUNKS >= 32u ? 32u : (CHUNKS >= 16u ? 16u : 8u);
				constexpr uint32_t GROUPS = QH_THREADS / LANES;
				constexpr uint32_t COL_BLOCKS = (G::kHidden + GROUPS - 1u) / GROUPS;
				const uint32_t up_id = rows == 1u ?
					(LANES == 32u ? QWENSKL2_hc_up_mix_skinny_r1_l32 : (LANES == 16u ? QWENSKL2_hc_up_mix_skinny_r1_l16 : QWENSKL2_hc_up_mix_skinny_r1_l8)) :
					(LANES == 32u ? QWENSKL2_hc_up_mix_skinny_r4_l32 : (LANES == 16u ? QWENSKL2_hc_up_mix_skinny_r4_l16 : QWENSKL2_hc_up_mix_skinny_r4_l8));
				const uint32_t up_blks = rows == 1u ? COL_BLOCKS : COL_BLOCKS * ((rows + 3u) / 4u);
				if ( QwenSkl2Launch(up_id, up_blks, QH_THREADS, 0, stream,
					up, state->hc_low, state->hc_normed, mixed, rows, G::kStreams, G::kHidden, G::kHcLowRank) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			else if constexpr ( (S::kResidual % 64u) == 0u && (G::kHcLowRank % 64u) == 0u && (G::kHidden % 64u) == 0u )
			{
				const bool use_32 = ((S::kResidual % 256u) == 0u) ? (rows <= 160u) : (rows <= 32u);
				const uint32_t dn_id = use_32 ?
					(inject != 0 ? QWENSKL2_qwen_hc_down_gate_inject_mma_32x32 : QWENSKL2_qwen_hc_down_gate_mma_32x32) :
					(inject != 0 ? QWENSKL2_qwen_hc_down_gate_inject_mma_64x64 : QWENSKL2_qwen_hc_down_gate_mma_64x64);
				const uint32_t tile_dn = use_32 ? 32u : 64u;
				const uint32_t m_dn = (rows + tile_dn - 1u) / tile_dn;
				const uint32_t n_dn = (G::kHcLowRank + (inject != 0 ? G::kStreams : 0u) + tile_dn - 1u) / tile_dn;
				if ( QwenSkl2Launch(dn_id, m_dn * n_dn, 128u, 0, stream,
					down, inject, state->hc_normed, state->hc_low, inject != 0 ? state->hc_inject : (float *)0,
					rows, G::kStreams, G::kHidden, G::kHcLowRank) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				SparkGpuError up_err;
				if ( rows >= 128u )
					up_err = QwenSkl2Launch(QWENSKL2_qwen_hc_up_mix_mma_128, ((rows + 127u) / 128u) * (G::kHidden / 64u), 256u, 0, stream,
						up, state->hc_low, state->hc_normed, mixed, rows, G::kStreams, G::kHidden, G::kHcLowRank);
				else if ( rows > 32u )
					up_err = QwenSkl2Launch(QWENSKL2_qwen_hc_up_mix_mma_64, ((rows + 63u) / 64u) * (G::kHidden / 64u), 128u, 0, stream,
						up, state->hc_low, state->hc_normed, mixed, rows, G::kStreams, G::kHidden, G::kHcLowRank);
				else if ( rows > 16u || (S::kResidual % 256u) != 0u )
					up_err = QwenSkl2Launch(QWENSKL2_qwen_hc_up_mix_mma_32, ((rows + 31u) / 32u) * (G::kHidden / 64u), 128u, 0, stream,
						up, state->hc_low, state->hc_normed, mixed, rows, G::kStreams, G::kHidden, G::kHcLowRank);
				else
					up_err = QwenSkl2Launch(QWENSKL2_qwen_hc_up_mix_mma_16, ((rows + 15u) / 16u) * (G::kHidden / 64u), 128u, 0, stream,
						up, state->hc_low, state->hc_normed, mixed, rows, G::kStreams, G::kHidden, G::kHcLowRank);
				if ( up_err != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			else
			{
				if ( Project<LmBf16Format>(down, 0, state->hc_normed, state->hc_low, rows, S::kResidual, G::kHcLowRank, exact, sms, stream, prefill_unified) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				LM_LAUNCH((LmHyperGateKernel<QH_THREADS>), LmGridBlocks<QH_THREADS>((uint64_t)rows * G::kHcLowRank), QH_THREADS, 0, stream, state->hc_low,
					(uint64_t)rows * G::kHcLowRank, (float)G::kStreams);
				if ( Project<LmBf16Format>(up, 0, state->hc_low, state->hc_up, rows, G::kHcLowRank, S::kResidual, exact, sms, stream, prefill_unified) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				if ( QwenSkl2Launch(QWENSKL2_hc_mix_vec_256, rows, QH_THREADS, 0, stream,
					state->hc_up, state->hc_normed, mixed, G::kStreams, G::kHidden) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
				if ( inject != 0 && QwenSkl2Launch(QWENSKL2_hc_inject_weights_vec_256, rows, QH_THREADS, 0, stream,
					state->hc_normed, inject, state->hc_inject, G::kStreams, G::kHidden) != SPARK_GPU_SUCCESS )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
		}
		return CudaStageLaunched();
	}

	static SparkStatus HcInject(State *state, uint32_t rows, SparkGpuStream stream)
	{
		LM_LAUNCH((LmHyperInjectKernel<QH_THREADS>), dim3(LmGridBlocks<QH_THREADS>(S::kResidual), rows), QH_THREADS, 0, stream, state->residual, state->partial,
			state->hc_inject, G::kStreams, G::kHidden);
		return CudaStageLaunched();
	}

	static SparkStatus PleRecur(State *state, const SparkStageRunnerStep *step, const uint16_t *conv_in, const uint32_t *kept, uint32_t commit,
		SparkGpuStream stream)
	{
		if ( LmCausalConvLaunch<G::kPleConv,LM_CONV_SWISH,uint16_t,G::kPleDilation>(state->ple_window, step->recurrent_index, step->sequence_row_begin,
			kept, conv_in, state->ple_conv_weight, state->ple_conv_out, S::kResidual, step->sequences, commit, step->sequence_row_indices, step->rows,
			stream) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LM_LAUNCH((LmNgramHistoryKernel), (step->sequences + 63u) / 64u, 64u, 0, stream, step->token_ids, (const int32_t *)0, step->sequence_row_begin,
			step->sequence_row_indices, step->recurrent_index, state->ple_history, S::kPleHistory, step->sequences, G::kPleOrders, commit, kept);
		return CudaStageLaunched();
	}

	static SparkStatus Ple(State *state, const Frame &frame)
	{
		const SparkStageRunnerStep *step = frame.step;
		SparkGpuStream stream = frame.stream;
		const uint32_t rows = step->rows, sms = step->multiprocessors, exact = Exact(step);
		const uint64_t elements = (uint64_t)rows * S::kResidual;
		SparkStatus status;
		if ( step->token_ids == 0 )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( state->hc_pending_inject != 0u )
		{
			state->hc_pending_inject = 0u;
			status = HcInject(state, rows, stream);
			if ( status != SPARK_STATUS_OK )
				return status;
		}
		CudaStageRowState(state, G::kPleLayer, SPARK_STAGE_RUNNER_ROW_STATE_READ);
		LM_LAUNCH((LmNgramHashKernel), (rows + 63u) / 64u, 64u, 0, stream, step->token_ids, (const int32_t *)0, step->positions, state->row_lane,
			state->row_ordinal, step->sequence_row_begin, step->sequence_row_indices, step->recurrent_index, state->ple_history, S::kPleHistory,
			state->ple_multipliers, state->ple_vocab_sizes, state->ple_offsets, state->ple_ids, rows, G::kPleOrders, G::kPleHeadsPerOrder,
			(int32_t)G::kPleEos, (int32_t)G::kPleEos, 1u);
		if constexpr ( G::kPleTableCodec == SPARK_WEIGHT_CODEC_BF16 )
			LM_LAUNCH((LmNgramGatherBf16Kernel), dim3(rows, S::kPleColumns), G::kPleHeadDim, 0, stream, state->ple_ids, (const uint16_t *)state->ple_table,
				state->ple_embed, state->ple_table_first, state->ple_table_rows, S::kPleColumns, G::kPleHeadDim);
		else
			LM_LAUNCH((LmNgramGatherScaledKernel), dim3(rows, S::kPleColumns), G::kPleHeadDim, 0, stream, state->ple_ids, state->ple_table, state->ple_table_scale,
				state->ple_embed, state->ple_table_first, state->ple_table_rows, S::kPleColumns, G::kPleHeadDim);
		status = CudaStageLaunched();
		if ( status == SPARK_STATUS_OK )
			status = CudaStageRound(state, stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, rows, 0u, state->ple_embed, state->ple_embed);
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( !PrefillWave(step) && (rows <= LM_SKINNY_ROWS_WIDE || exact != 0u) )
		{
			if ( rows >= 2u && rows <= 32u && G::kHidden >= 256u && (G::kHidden & 7u) == 0u )
			{
				QwenBf16MultiArgs args;
				args.part[0] = { state->ple_key_weight, state->ple_key, S::kResidual, 0u };
				args.part[1] = { state->ple_value_weight, state->ple_value, G::kHidden, 0u };
				args.count = 2u;
				args.activation = state->ple_embed;
				args.rows = rows;
				args.input_dim = G::kHidden;
				if ( LaunchExactBf16Multi(args, stream) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
			else
			{
				const LmSkinnyDenseTarget targets[2] =
				{
					{ state->ple_key_weight, state->ple_key, (float *)0, S::kResidual, S::kResidual, 0u, LmScaleTensorNone() },
					{ state->ple_value_weight, state->ple_value, (float *)0, G::kHidden, G::kHidden, 0u, LmScaleTensorNone() }
				};
				if ( LmSkinnyDenseMulti<LmBf16Format>(targets, 2u, state->ple_embed, rows, G::kHidden, stream, exact) != LM_LAUNCH_OK )
					SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
		}
		else if ( Project<LmBf16Format>(state->ple_key_weight, 0, state->ple_embed, state->ple_key, rows, G::kHidden, S::kResidual, exact, sms, stream, PrefillWave(step)) !=
				LM_LAUNCH_OK ||
			Project<LmBf16Format>(state->ple_value_weight, 0, state->ple_embed, state->ple_value, rows, G::kHidden, G::kHidden, exact, sms, stream, PrefillWave(step)) !=
				LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		if constexpr ( (G::kHidden & 7u) == 0u && G::kHidden <= QH_THREADS * LM_RMS_NORM_STAGED )
		{
			constexpr uint32_t ple_id = G::kHidden <= 2048u ? QWENSKL2_ple_key_query_gate_norm_256_w2048 :
				(G::kHidden <= 4096u ? QWENSKL2_ple_key_query_gate_norm_256_w4096 : QWENSKL2_ple_key_query_gate_norm_256_w8192);
			if ( QwenSkl2Launch(ple_id, dim3(G::kStreams, rows), QH_THREADS, 0, stream,
					state->ple_key, state->residual, state->ple_value, state->ple_norm_key, state->ple_norm_query, state->ple_norm_conv,
					state->ple_gated, state->ple_conv_in, G::kStreams, G::kHidden, QH_EPSILON) != SPARK_GPU_SUCCESS )
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		}
		else
		{
			LM_LAUNCH((LmGroupRmsNormKernel<QH_THREADS>), dim3(G::kStreams, rows), QH_THREADS, 0, stream, state->ple_key, state->ple_norm_key, state->ple_key,
				G::kStreams, G::kHidden, QH_EPSILON);
			LM_LAUNCH((LmGroupRmsNormKernel<QH_THREADS>), dim3(G::kStreams, rows), QH_THREADS, 0, stream, state->residual, state->ple_norm_query, state->ple_query,
				G::kStreams, G::kHidden, QH_EPSILON);
			LM_LAUNCH((LmHyperKeyGateKernel<QH_THREADS>), dim3(G::kStreams, rows), QH_THREADS, 0, stream, state->ple_key, state->ple_query, state->ple_value,
				state->ple_gated, G::kStreams, G::kHidden);
			LM_LAUNCH((LmGroupRmsNormKernel<QH_THREADS>), dim3(G::kStreams, rows), QH_THREADS, 0, stream, state->ple_gated, state->ple_norm_conv, state->ple_conv_in,
				G::kStreams, G::kHidden, QH_EPSILON);
		}
		if ( step->commit == 0u && SparkGpuMemcpyAsync(state->replay_ple, state->ple_conv_in, elements * sizeof(uint16_t), SPARK_GPU_COPY_DEVICE_TO_DEVICE, stream) !=
			SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		status = PleRecur(state, step, state->ple_conv_in, (const uint32_t *)0, step->commit, stream);
		if ( status != SPARK_STATUS_OK )
			return status;
		LM_LAUNCH((LmHyperAddKernel<QH_THREADS>), LmGridBlocks<QH_THREADS>(elements), QH_THREADS, 0, stream, state->residual, state->ple_gated, state->ple_conv_out, elements);
		CudaStageRowState(state, G::kPleLayer, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
		return CudaStageLaunched();
	}

	static SparkStatus Begin(State *state, const SparkStageRunnerStep *step, void *stream_void, Frame *frame)
	{
		SparkGpuStream stream = (SparkGpuStream)stream_void;
		SparkStatus status = SPARK_STATUS_OK;
		if ( step->rows == 0u || step->rows > state->max_rows || step->sequences > state->slots || (step->commit == 0u && step->rows > G::kVerifyRows) )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		if ( step->tree != 0u && (!kTreeVerify || step->commit != 0u || step->sequences != 1u || state->gdn_snapshots == 0 || step->slots == 0 ||
			step->fork_positions == 0 || step->slot_offsets == 0 || step->parent_rows == 0 || step->state_loads == 0 || step->state_saves == 0) )
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		state->hc_pending_inject = 0u;
		frame->step = step;
		frame->stream = stream;
		if ( state->dump != 0 )
		{
			SparkGpuCaptureStatus capturing = SPARK_GPU_CAPTURE_STATUS_NONE;
			if ( SparkGpuStreamIsCapturing(stream, &capturing) != SPARK_GPU_SUCCESS || capturing != SPARK_GPU_CAPTURE_STATUS_NONE )
				SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
			status = DumpLayer(state, 0xffffffffu, step, step->hidden_bf16, 0, stream);
		}
		return status;
	}

	static SparkStatus RmsNorm(State *state, const Frame &frame, const float *weight)
	{
		SparkGpuError err;
		if constexpr ( G::kHidden == 5120u )
			err = QwenSkl2Launch(QWENSKL2_rmsnorm_pdl_exact_5120_bf16x8, frame.step->rows, QH_THREADS, 0, frame.stream,
				frame.step->hidden_bf16, (const uint16_t *)0, weight, state->normed, QH_EPSILON);
		else
			err = QwenSkl2Launch(QWENSKL2_fused_residual_rmsnorm_f32_generic_pdl_exact_bf16x8, frame.step->rows, QH_THREADS, 0, frame.stream,
				frame.step->hidden_bf16, (const uint16_t *)0, weight, (uint16_t *)0, state->normed, G::kHidden, G::kHidden, QH_EPSILON);
		return err == SPARK_GPU_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
	}

	static SparkStatus AddRmsNorm(State *state, const Frame &frame, const float *weight)
	{
		uint16_t *hidden = frame.step->hidden_bf16;
		SparkGpuError err;
		if constexpr ( G::kHidden == 5120u )
			err = QwenSkl2Launch(QWENSKL2_fused_residual_rmsnorm_pdl_exact_5120_bf16x8, frame.step->rows, QH_THREADS, 0, frame.stream,
				hidden, state->partial, weight, state->normed, QH_EPSILON);
		else
			err = QwenSkl2Launch(QWENSKL2_fused_residual_rmsnorm_f32_generic_pdl_exact_bf16x8, frame.step->rows, QH_THREADS, 0, frame.stream,
				hidden, state->partial, weight, hidden, state->normed, G::kHidden, G::kHidden, QH_EPSILON);
		return err == SPARK_GPU_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
	}

	static SparkStatus AllReduce(State *state, const Frame &frame)
	{
		if constexpr ( G::kTpDegree == 1u )
			return SPARK_STATUS_OK;
		else
			return CudaStageRound(state, frame.stream, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, frame.step->rows, 0u, state->partial, state->partial);
	}

	static SparkStatus LayerDone(State *state, const Frame &frame, uint32_t index)
	{
		SparkStatus status = SPARK_STATUS_OK;
		if ( state->dump != 0 )
		{
			const uint16_t *rows;
			if constexpr ( S::kHyper )
				rows = state->residual;
			else
				rows = frame.step->hidden_bf16;
			status = DumpLayer(state, index, frame.step, rows, state->partial, frame.stream);
		}
		if ( status == SPARK_STATUS_OK && state->tap_slab != 0 && state->tap_output[index] >= 0 )
		{
			LM_LAUNCH((LmAddRowsStridedKernel<QH_THREADS>), dim3(LmGridBlocks<QH_THREADS>(G::kHidden), frame.step->rows), QH_THREADS, 0, frame.stream,
				frame.step->hidden_bf16, state->partial, state->tap_slab + (uint64_t)frame.step->dispatch_row_offset * state->tap_stride +
				(uint64_t)state->tap_output[index] * G::kHidden, frame.step->rows, G::kHidden, state->tap_stride);
			status = CudaStageLaunched();
		}
		if ( status == SPARK_STATUS_OK && state->services->layer_done != 0 )
			state->services->layer_done(state->services->context, index);
		return status;
	}

	static SparkStatus AddRows(State *state, const Frame &frame)
	{
		const uint64_t width = (uint64_t)frame.step->rows * G::kHidden;
		LM_LAUNCH((LmAddElementsKernel), LmVecBlocks<QH_THREADS>(width), QH_THREADS, 0, frame.stream, frame.step->hidden_bf16, state->partial, width);
		return SPARK_STATUS_OK;
	}

	static SparkStatus HyperBegin(State *state, const Frame &frame)
	{
		const SparkStageRunnerStep *step = frame.step;
		LM_LAUNCH((LmHyperExpandKernel<QH_THREADS>), dim3(LmGridBlocks<QH_THREADS>(S::kResidual), step->rows), QH_THREADS, 0, frame.stream, step->hidden_bf16, state->residual,
			G::kStreams, G::kHidden);
		if constexpr ( S::kPle || S::kIndexer )
			LM_LAUNCH((LmRowLanesKernel), step->sequences, QH_THREADS, 0, frame.stream, step->sequence_row_begin, step->sequence_row_indices, state->row_lane,
				state->row_ordinal);
		return CudaStageLaunched();
	}

	static SparkStatus HyperPre(State *state, const Frame &frame, const QwenHybridLayer &layer, uint32_t part)
	{
		return HcPre(state, layer.hc_norm[part], layer.hc_down[part], layer.hc_up[part], layer.hc_inject[part], state->normed, frame.step->rows,
			Exact(frame.step), frame.step->multiprocessors, frame.stream, PrefillWave(frame.step));
	}

	static SparkStatus HyperPost(State *state, const Frame &frame)
	{
		if ( state->dump == 0 )
		{
			state->hc_pending_inject = 1u;
			return SPARK_STATUS_OK;
		}
		return HcInject(state, frame.step->rows, frame.stream);
	}

	static SparkStatus HyperEnd(State *state, const Frame &frame)
	{
		return HcPre(state, state->final_hc_norm, state->final_hc_down, state->final_hc_up, (const uint16_t *)0, frame.step->hidden_bf16,
			frame.step->rows, Exact(frame.step), frame.step->multiprocessors, frame.stream, PrefillWave(frame.step));
	}

	static uint64_t Plan(State *state, uint8_t *base, uint32_t lane)
	{
		const uint64_t rows = state->max_rows, slots = lane == 0u ? state->slots : 0u, routes = rows * (G::kMoe ? G::kTopK : 0u);
		const uint64_t width = rows * G::kHidden * sizeof(uint16_t), replay = lane == 0u ? G::kVerifyRows : 0u;
		constexpr bool kFusedPle = S::kPle && (G::kHidden & 7u) == 0u && G::kHidden <= QH_THREADS * LM_RMS_NORM_STAGED;
		const uint64_t send_floats = (uint64_t)state->tp_degree * state->record_stride > state->split_gqa_floats ?
			(uint64_t)state->tp_degree * state->record_stride : state->split_gqa_floats;
		const CudaStageSlice persistent[] =
		{
			{ (void **)&state->normed, width },
			{ (void **)&state->partial, width },
			{ (void **)&state->residual, S::kHyper ? rows * S::kResidual * sizeof(uint16_t) : 0u },
			{ (void **)&state->hc_inject, S::kHyper ? rows * G::kStreams * sizeof(float) : 0u },
			{ (void **)&state->row_lane, S::kPle || S::kIndexer ? rows * sizeof(uint32_t) : 0u },
			{ (void **)&state->row_ordinal, S::kPle || S::kIndexer ? rows * sizeof(uint32_t) : 0u },
			{ (void **)&state->gdn_state, (uint64_t)S::kGdnLayers * slots * state->gdn_state_bytes },
			{ (void **)&state->gdn_window, (uint64_t)S::kGdnLayers * slots * state->gdn_window_bytes },
			{ (void **)&state->gdn_snapshots, replay != 0u && kTreeVerify ? LmDeltaRuleSnapshotBytes(state->gdn_heads, QH_KEY_DIM, QH_VALUE_DIM) : 0u },
			{ (void **)&state->replay_qkv, (uint64_t)S::kGdnLayers * replay * state->gdn_channels * sizeof(uint16_t) },
			{ (void **)&state->replay_ba, (uint64_t)S::kGdnLayers * replay * 2u * state->gdn_heads * sizeof(uint16_t) },
			{ (void **)&state->replay_ple, S::kPle ? replay * S::kResidual * sizeof(uint16_t) : 0u },
			{ (void **)&state->replay_index, (uint64_t)S::kAttnLayers * replay * G::kIndexDim * sizeof(uint16_t) },
			{ (void **)&state->ple_window, slots * state->ple_window_bytes },
			{ (void **)&state->ple_history, S::kPle ? slots * S::kPleHistory * sizeof(int32_t) : 0u },
			{ (void **)&state->index_tail, (uint64_t)S::kAttnLayers * slots * S::kIndexTail * G::kIndexDim * sizeof(uint16_t) },
			{ (void **)&state->kv_error, lane == 0u ? sizeof(LmKvAccessError) : 0u },
		};
		uint64_t fixed = CudaStagePlace(persistent, base);
		fixed = (fixed + CUDA_STAGE_SLICE_ALIGNMENT - 1u) & ~(uint64_t)(CUDA_STAGE_SLICE_ALIGNMENT - 1u);
		uint8_t *scratch_base = base != 0 ? base + fixed : 0;
		uint64_t scratch = 0u;
		auto overlay = [&](uint64_t bytes) { if ( bytes > scratch ) scratch = bytes; };
		if constexpr ( G::kMoe )
		{
			constexpr bool kFusedNvfp4 = G::kExpertCodec == SPARK_WEIGHT_CODEC_NVFP4_E2M1 &&
				(G::kHidden % 128u) == 0u && (G::kMoeInter % 128u) == 0u;
			constexpr bool kWarpTopk = G::kExperts <= LM_TOPK_WARP_PER_LANE * LM_WARP_LANES;
			const CudaStageSlice ffn_moe[] =
			{
				{ (void **)&state->router_logits, rows * G::kExperts * sizeof(uint16_t) },
				{ (void **)&state->router_probs, rows * (!kWarpTopk ? G::kExperts : 0u) * sizeof(float) },
				{ (void **)&state->route_expert, routes * sizeof(uint32_t) },
				{ (void **)&state->route_global, routes * sizeof(uint32_t) },
				{ (void **)&state->route_weight, routes * sizeof(float) },
				{ (void **)&state->group_row_offset, (G::kLocalExperts + 2u) * sizeof(uint32_t) },
				{ (void **)&state->route_packed_row, routes * sizeof(uint32_t) },
				{ (void **)&state->route_source_token, routes * sizeof(uint32_t) },
				{ (void **)&state->packed_expert, routes * sizeof(uint32_t) },
				{ (void **)&state->tile_prefix_up, (G::kLocalExperts + 2u) * sizeof(uint32_t) },
				{ (void **)&state->tile_prefix_down, (G::kLocalExperts + 2u) * sizeof(uint32_t) },
				{ (void **)&state->packed_weight, routes * sizeof(float) },
				{ (void **)&state->expert_gate, routes * (!kFusedNvfp4 ? G::kMoeInter : 0u) * sizeof(uint16_t) },
				{ (void **)&state->expert_up, routes * (!kFusedNvfp4 ? G::kMoeInter : 0u) * sizeof(uint16_t) },
				{ (void **)&state->expert_act, routes * G::kMoeInter * sizeof(uint16_t) },
				{ (void **)&state->expert_out, routes * G::kHidden * sizeof(uint16_t) },
				{ (void **)&state->shared_gate_up, rows * 2u * G::kSharedRows * sizeof(uint16_t) },
				{ (void **)&state->shared_act, rows * G::kSharedRows * sizeof(uint16_t) },
				{ (void **)&state->shared_out, width },
				{ (void **)&state->shared_logit, rows * sizeof(uint16_t) },
			};
			overlay(CudaStagePlace(ffn_moe, scratch_base));
		}
		else
		{
			const CudaStageSlice ffn_dense[] =
			{
				{ (void **)&state->gate_up, rows * 2u * state->ffn_rows * sizeof(uint16_t) },
				{ (void **)&state->intermediate, rows * state->ffn_rows * sizeof(uint16_t) },
			};
			overlay(CudaStagePlace(ffn_dense, scratch_base));
		}
		const CudaStageSlice gdn_qkv[] =
		{
			{ (void **)&state->qkv, rows * state->gdn_channels * sizeof(uint16_t) },
			{ (void **)&state->z, rows * state->gdn_v * sizeof(uint16_t) },
		};
		const CudaStageSlice gdn_qkvz[] =
		{
			{ (void **)&state->gdn_qkvz, rows * (state->gdn_channels + state->gdn_v) * sizeof(uint16_t) },
		};
		const uint64_t gdn_a = CudaStagePlace(gdn_qkv, scratch_base);
		const uint64_t gdn_b = CudaStagePlace(gdn_qkvz, scratch_base);
		uint64_t gdn_head = gdn_a > gdn_b ? gdn_a : gdn_b;
		gdn_head = (gdn_head + CUDA_STAGE_SLICE_ALIGNMENT - 1u) & ~(uint64_t)(CUDA_STAGE_SLICE_ALIGNMENT - 1u);
		const CudaStageSlice gdn_rest[] =
		{
			{ (void **)&state->ba, rows * 2u * state->gdn_heads * sizeof(uint16_t) },
			{ (void **)&state->value, rows * state->gdn_v * sizeof(uint16_t) },
			{ (void **)&state->query_heads, rows * state->gdn_heads * QH_KEY_DIM * sizeof(uint16_t) },
			{ (void **)&state->key_heads, rows * state->gdn_heads * QH_KEY_DIM * sizeof(uint16_t) },
			{ (void **)&state->retention, rows * state->gdn_heads * QH_KEY_DIM * sizeof(float) },
			{ (void **)&state->write_gate, rows * state->gdn_heads * sizeof(float) },
			{ (void **)&state->mixed, rows * state->gdn_v * sizeof(uint16_t) },
		};
		overlay(gdn_head + CudaStagePlace(gdn_rest, scratch_base != 0 ? scratch_base + gdn_head : 0));
		const CudaStageSlice attn_group[] =
		{
			{ (void **)&state->attn_local, rows * state->attn_rows * sizeof(uint16_t) },
			{ (void **)&state->attn_gathered, G::kTpDegree > 1u ? rows * S::kQkvWidth * sizeof(uint16_t) : 0u },
			{ (void **)&state->attn_query, rows * S::kQWidth * sizeof(uint16_t) },
			{ (void **)&state->send, send_floats * sizeof(float) },
			{ (void **)&state->received, G::kTpDegree > 1u ? (uint64_t)state->tp_degree * state->record_stride * sizeof(float) : 0u },
			{ (void **)&state->merged, rows * state->attn_width * sizeof(uint16_t) },
			{ (void **)&state->merged_gate, G::kTpDegree > 1u ? rows * state->attn_width * sizeof(uint16_t) : 0u },
			{ (void **)&state->index_qk, S::kIndexer ? rows * S::kIndexWidth * sizeof(uint16_t) : 0u },
			{ (void **)&state->index_query, S::kIndexer ? rows * G::kIndexHeads * G::kIndexDim * sizeof(uint16_t) : 0u },
			{ (void **)&state->index_key, S::kIndexer ? rows * G::kIndexDim * sizeof(uint16_t) : 0u },
			{ (void **)&state->index_scores, S::kIndexer ? rows * state->index_local_stride * sizeof(float) : 0u },
			{ (void **)&state->index_local_selected, S::kIndexer ? rows * S::kIndexBlocks * sizeof(uint32_t) : 0u },
			{ (void **)&state->index_candidates, S::kIndexer ? rows * S::kIndexBlocks * sizeof(uint2) : 0u },
			{ (void **)&state->index_gathered, S::kIndexer ? (uint64_t)state->tp_degree * rows * S::kIndexBlocks * sizeof(uint2) : 0u },
			{ (void **)&state->index_mask, S::kIndexer ? rows * state->index_mask_words * sizeof(uint32_t) : 0u },
		};
		overlay(CudaStagePlace(attn_group, scratch_base));
		const CudaStageSlice hc_ple[] =
		{
			{ (void **)&state->hc_normed, S::kHyper ? rows * S::kResidual * sizeof(uint16_t) : 0u },
			{ (void **)&state->hc_low, S::kHyper ? rows * G::kHcLowRank * sizeof(uint16_t) : 0u },
			{ (void **)&state->hc_up, S::kHyper ? rows * S::kResidual * sizeof(uint16_t) : 0u },
			{ (void **)&state->ple_ids, S::kPle ? rows * S::kPleColumns * sizeof(int64_t) : 0u },
			{ (void **)&state->ple_embed, S::kPle ? width : 0u },
			{ (void **)&state->ple_key, S::kPle ? rows * S::kResidual * sizeof(uint16_t) : 0u },
			{ (void **)&state->ple_query, S::kPle && !kFusedPle ? rows * S::kResidual * sizeof(uint16_t) : 0u },
			{ (void **)&state->ple_value, S::kPle ? width : 0u },
			{ (void **)&state->ple_gated, S::kPle ? rows * S::kResidual * sizeof(uint16_t) : 0u },
			{ (void **)&state->ple_conv_in, S::kPle ? rows * S::kResidual * sizeof(uint16_t) : 0u },
			{ (void **)&state->ple_conv_out, S::kPle ? rows * S::kResidual * sizeof(uint16_t) : 0u },
		};
		overlay(CudaStagePlace(hc_ple, scratch_base));
		if ( base != 0 && G::kTpDegree == 1u )
		{
			state->attn_gathered = state->attn_local;
			state->received = state->send;
			state->merged_gate = 0;
		}
		return fixed + scratch;
	}

	static void Release(State *state)
	{
		if ( state->dump != 0 )
			fclose(state->dump);
		free(state->dump_rows);
		free(state->dump_positions);
	}

	static SparkStatus Config(State *state, const SparkStageRunnerConfiguration *configuration)
	{
		uint32_t degree = 0u, rank = 0u, hidden = 0u, layers = 0u, vocab_rows = 0u, experts = 0u, local = 0u, top_k = 0u;
		if ( SparkNamedPackConfigU32(&state->pack, "tp_degree", &degree) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "tp_rank", &rank) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "hidden", &hidden) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "layers", &layers) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->pack, "vocab_rows", &vocab_rows) != SPARK_STATUS_OK )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( degree != configuration->tp_degree || rank != configuration->tp_rank || hidden != G::kHidden || layers != G::kLayers ||
			vocab_rows * degree != G::kVocab )
		{
			fprintf(stderr, "%s: pack is tp %u rank %u hidden %u layers %u, the runner is tp %u rank %u\n",
				G::Tag(), degree, rank, hidden, layers, configuration->tp_degree, configuration->tp_rank);
			SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
		}
		if constexpr ( G::kMoe )
		{
			if ( SparkNamedPackConfigU32(&state->pack, "experts", &experts) != SPARK_STATUS_OK ||
				SparkNamedPackConfigU32(&state->pack, "local_experts", &local) != SPARK_STATUS_OK ||
				SparkNamedPackConfigU32(&state->pack, "expert_first", &state->expert_first) != SPARK_STATUS_OK ||
				SparkNamedPackConfigU32(&state->pack, "top_k", &top_k) != SPARK_STATUS_OK ||
				experts != G::kExperts || local != G::kLocalExperts || top_k != G::kTopK || state->expert_first != rank * G::kLocalExperts )
			{
				fprintf(stderr, "%s: pack experts %u local %u first %u top_k %u do not match the model\n", G::Tag(), experts, local,
					state->expert_first, top_k);
				SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
			}
		}
		else if ( SparkNamedPackConfigU32(&state->pack, "ffn_rows", &state->ffn_rows) != SPARK_STATUS_OK || state->ffn_rows == 0u ||
			state->ffn_rows % QH_SCALE_K != 0u )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if constexpr ( S::kHyper )
		{
			uint32_t streams = 0u, lowrank = 0u, shared = 0u;
			if ( SparkNamedPackConfigU32(&state->pack, "hc_streams", &streams) != SPARK_STATUS_OK ||
				SparkNamedPackConfigU32(&state->pack, "hc_lowrank", &lowrank) != SPARK_STATUS_OK ||
				SparkNamedPackConfigU32(&state->pack, "shared_rows", &shared) != SPARK_STATUS_OK ||
				streams != G::kStreams || lowrank != G::kHcLowRank || shared != G::kSharedRows )
			{
				fprintf(stderr, "%s: pack hyper-connection streams %u lowrank %u shared rows %u do not match the model\n", G::Tag(),
					streams, lowrank, shared);
				SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
			}
		}
		if constexpr ( S::kPle )
		{
			uint32_t layer = 0u, eos = 0u;
			if ( SparkNamedPackConfigU32(&state->pack, "ple_layer", &layer) != SPARK_STATUS_OK ||
				SparkNamedPackConfigU32(&state->pack, "ple_eos", &eos) != SPARK_STATUS_OK ||
				SparkNamedPackConfigU32(&state->pack, "ple_table_first", &state->ple_table_first) != SPARK_STATUS_OK ||
				SparkNamedPackConfigU32(&state->pack, "ple_table_rows", &state->ple_table_rows) != SPARK_STATUS_OK ||
				layer != G::kPleLayer || eos != G::kPleEos || state->ple_table_rows == 0u )
			{
				fprintf(stderr, "%s: pack n-gram embedding layer %u eos %u rows %u do not match the model\n", G::Tag(), layer, eos,
					state->ple_table_rows);
				SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
			}
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Shape(State *state)
	{
		state->gdn_heads = G::kGdnValueHeads / state->tp_degree;
		state->gdn_qk = G::kGdnKeyHeads / state->tp_degree * QH_KEY_DIM;
		state->gdn_v = state->gdn_heads * QH_VALUE_DIM;
		state->gdn_channels = 2u * state->gdn_qk + state->gdn_v;
		state->attn_rows = S::kQkvWidth / state->tp_degree;
		state->attn_width = S::kQWidth / state->tp_degree;
		state->gdn_state_bytes = (uint64_t)state->gdn_heads * QH_KEY_DIM * QH_VALUE_DIM * sizeof(float);
		state->gdn_window_bytes = (uint64_t)state->gdn_channels * QH_CONV * sizeof(uint16_t);
		state->ple_window_bytes = S::kPle ? (uint64_t)S::kResidual * S::kPleSpan * sizeof(uint16_t) : 0u;
		if constexpr ( S::kIndexer )
		{
			SparkKvShard shard;
			shard.degree = state->tp_degree;
			shard.rank = state->tp_rank;
			shard.grain = 1u;
			const uint32_t blocks = state->max_positions / G::kIndexRatio;
			const uint32_t local = ((SparkKvShardGatherKeys(shard, blocks) + 63u) / 64u) * 64u, global = ((blocks + 63u) / 64u) * 64u;
			state->index_local_stride = local > S::kIndexBlocks ? local : S::kIndexBlocks;
			state->index_global_stride = global > S::kIndexBlocks ? global : S::kIndexBlocks;
			state->index_mask_words = state->index_global_stride / 32u;
		}
		state->record_stride = (((uint64_t)state->max_rows * LmGqaShardRecordFloats(G::kHeads, G::kHeadDim, state->tp_degree) * sizeof(float) +
			G::kHidden * sizeof(uint16_t) - 1u) / (G::kHidden * sizeof(uint16_t))) * G::kHidden * sizeof(uint16_t) / sizeof(float);
		if constexpr ( G::kTpDegree == 1u && !S::kIndexer && G::kHeadDim == 256u && (G::kHeads % G::kKvHeads) == 0u )
		{
			const uint32_t split_rows = state->max_rows < 64u ? state->max_rows : 64u;
			state->split_gqa_floats = (uint64_t)split_rows * QH_GQA_MAX_SPLITS * G::kHeads * (G::kHeadDim + 4u);
		}
		else
			state->split_gqa_floats = 0u;
		return SPARK_STATUS_OK;
	}

	static SparkStatus Prepare(State *state, const SparkStageRunnerConfiguration *configuration)
	{
#ifdef DEBUG
		if ( getenv(G::DumpEnv()) != 0 )
		{
			state->dump = fopen(getenv(G::DumpEnv()), "wb");
			state->dump_rows = (uint16_t *)malloc((size_t)state->max_rows * (S::kResidual + G::kHidden) * sizeof(uint16_t));
			state->dump_positions = (uint32_t *)malloc((size_t)state->max_rows * sizeof(uint32_t));
			if ( state->dump == 0 || state->dump_rows == 0 || state->dump_positions == 0 )
				SPARK_FAIL(SPARK_STATUS_IO_ERROR);
			fprintf(stderr, "%s: this rank dumps every layer's residual and sublayer output to %s (debug; graphs off)\n",
				G::Tag(), getenv(G::DumpEnv()));
		}
#endif
		if ( configuration->linear_weight_codec == SPARK_WEIGHT_CODEC_FP8_E4M3 && !G::kMoe )
		{
			state->linear_bytes = 1u;
			state->linear_kind = "fp8_e4m3";
		}
		else if ( configuration->linear_weight_codec == SPARK_WEIGHT_CODEC_BF16 )
		{
			state->linear_bytes = 2u;
			state->linear_kind = "bf16";
		}
		else
		{
			fprintf(stderr, "%s: linear weight codec %u is not supported by this model\n", G::Tag(), configuration->linear_weight_codec);
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		}
		state->max_positions = configuration->kv_pages_per_sequence * G::kPageSlots;
		return SPARK_STATUS_OK;
	}

	static SparkStatus Head(State *state, SparkStageRunnerModelGeometry *geometry)
	{
		geometry->tree_verify = kTreeVerify ? 1u : 0u;
		geometry->graph_ignores_gather = 0u;
		geometry->head_prenormed = S::kHyper ? 1u : 0u;
		return CudaStage<QwenHybridModel>::HeadRanges(state, geometry, false, S::kHyper);
	}

	template<class T>
	static void Elements(CudaStageTensors *tensors, const char *name, const char *kind, T **field, uint64_t elements)
	{
		tensors->Add(name, kind, elements * sizeof(T), field);
	}

	static void LinearTensors(const State *state, CudaStageTensors *tensors, const char *name, const uint8_t **field, const float **scale_field, uint64_t rows,
		uint64_t columns)
	{
		tensors->Add(name, state->linear_kind, rows * columns * state->linear_bytes, field);
		*scale_field = 0;
		if ( state->linear_bytes == 1u )
			tensors->AddScaled(name, ".scale", "f32", rows * ((columns + QH_SCALE_K - 1u) / QH_SCALE_K) * sizeof(float), scale_field);
	}

	static void ExpertTensors(CudaStageTensors *tensors, const char *name, const uint8_t **field, const uint8_t **scale_field, uint64_t rows, uint64_t columns)
	{
		constexpr bool scaled = LmWeightCodec<G::kExpertCodec>::kScaleEncoding != LM_SCALE_ENCODING_NONE;
		tensors->Add(name, scaled ? "nvfp4_e2m1" : "bf16", SparkWeightCodecPayloadBytes(G::kExpertCodec, G::kLocalExperts * rows, columns), field);
		*scale_field = 0;
		if ( scaled )
			tensors->AddScaled(name, ".scale", "ue4m3_f32_global", SparkWeightCodecScaleBytes(G::kExpertCodec, G::kLocalExperts, rows, columns), scale_field);
	}

	static void PleTensors(State *state, CudaStageTensors *t)
	{
		Elements(t, "ple_key", "bf16", &state->ple_key_weight, (uint64_t)S::kResidual * G::kHidden);
		Elements(t, "ple_value", "bf16", &state->ple_value_weight, (uint64_t)G::kHidden * G::kHidden);
		Elements(t, "ple_norm_key", "f32", &state->ple_norm_key, S::kResidual);
		Elements(t, "ple_norm_query", "f32", &state->ple_norm_query, S::kResidual);
		Elements(t, "ple_norm_conv", "f32", &state->ple_norm_conv, S::kResidual);
		Elements(t, "ple_conv", "bf16", &state->ple_conv_weight, (uint64_t)S::kResidual * G::kPleConv);
		if constexpr ( G::kPleTableCodec == SPARK_WEIGHT_CODEC_BF16 )
		{
			state->ple_table_scale = 0;
			Elements(t, "ple_table", "bf16", &state->ple_table, (uint64_t)state->ple_table_rows * G::kPleHeadDim * 2u);
		}
		else
		{
			static_assert(G::kPleTableCodec == SPARK_WEIGHT_CODEC_FP8_E4M3, "the PLE table is fp8 with one scale or bf16");
			Elements(t, "ple_table", "fp8_e4m3", &state->ple_table, (uint64_t)state->ple_table_rows * G::kPleHeadDim);
			Elements(t, "ple_table_scale", "f32", &state->ple_table_scale, 1u);
		}
		Elements(t, "ple_multipliers", "i64", &state->ple_multipliers, G::kPleOrders + 1u);
		Elements(t, "ple_vocab_sizes", "i64", &state->ple_vocab_sizes, S::kPleColumns);
		Elements(t, "ple_offsets", "i64", &state->ple_offsets, S::kPleColumns);
	}

	static SparkStatus GlobalTensors(State *state, CudaStageTensors *t)
	{
		if constexpr ( S::kHyper )
		{
			Elements(t, "final_hc_norm", "f32", &state->final_hc_norm, S::kResidual);
			Elements(t, "final_hc_down", "bf16", &state->final_hc_down, (uint64_t)G::kHcLowRank * S::kResidual);
			Elements(t, "final_hc_up", "bf16", &state->final_hc_up, (uint64_t)S::kResidual * G::kHcLowRank);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus LayerTensors(State *state, uint32_t index, CudaStageTensors *t)
	{
		QwenHybridLayer *layer = &state->layers[index];
		const uint32_t attention_layer = index % G::kPeriod == G::kPhase;
		if constexpr ( G::kMoe )
		{
			Elements(t, "moe_router", "bf16", &layer->moe_router, (uint64_t)G::kExperts * G::kHidden);
			ExpertTensors(t, "moe_w1", &layer->moe_w1, &layer->moe_w1_scale, G::kMoeInter, G::kHidden);
			ExpertTensors(t, "moe_w3", &layer->moe_w3, &layer->moe_w3_scale, G::kMoeInter, G::kHidden);
			ExpertTensors(t, "moe_w2", &layer->moe_w2, &layer->moe_w2_scale, G::kHidden, G::kMoeInter);
			Elements(t, "moe_shared_gate_up", "bf16", &layer->moe_shared_gate_up, (uint64_t)2u * G::kSharedRows * G::kHidden);
			Elements(t, "moe_shared_down", "bf16", &layer->moe_shared_down, (uint64_t)G::kHidden * G::kSharedRows);
			Elements(t, "moe_shared_gate", "bf16", &layer->moe_shared_gate, G::kHidden);
		}
		else
		{
			LinearTensors(state, t, "ffn_gate_up", &layer->ffn_gate_up, &layer->ffn_gate_up_scale, 2u * state->ffn_rows, G::kHidden);
			LinearTensors(state, t, "ffn_down", &layer->ffn_down, &layer->ffn_down_scale, G::kHidden, state->ffn_rows);
		}
		if ( attention_layer )
		{
			LinearTensors(state, t, "attn_qkv", &layer->mix_weight, &layer->mix_scale, S::kQkvWidth / G::kTpDegree, G::kHidden);
			LinearTensors(state, t, "attn_out", &layer->out_weight, &layer->out_scale, G::kHidden, S::kQWidth / G::kTpDegree);
		}
		else
		{
			LinearTensors(state, t, "gdn_qkvz", &layer->mix_weight, &layer->mix_scale, 2u * state->gdn_qk + 2u * state->gdn_v, G::kHidden);
			LinearTensors(state, t, "gdn_out", &layer->out_weight, &layer->out_scale, G::kHidden, state->gdn_v);
		}
		if ( !S::kHyper )
		{
			Elements(t, "input_norm", "f32", &layer->input_norm, G::kHidden);
			Elements(t, "post_norm", "f32", &layer->post_norm, G::kHidden);
		}
		for ( uint32_t part = 0u; part < 2u && S::kHyper; ++part )
		{
			const char *prefix = part == 0u ? "attn_hc_" : "mlp_hc_";
			char name[SPARK_NAMED_PACK_MAX_NAME_BYTES];
			snprintf(name, sizeof(name), "%snorm", prefix);
			Elements(t, name, "f32", &layer->hc_norm[part], S::kResidual);
			snprintf(name, sizeof(name), "%sdown", prefix);
			Elements(t, name, "bf16", &layer->hc_down[part], (uint64_t)G::kHcLowRank * S::kResidual);
			snprintf(name, sizeof(name), "%sup", prefix);
			Elements(t, name, "bf16", &layer->hc_up[part], (uint64_t)S::kResidual * G::kHcLowRank);
			snprintf(name, sizeof(name), "%sinject", prefix);
			Elements(t, name, "bf16", &layer->hc_inject[part], (uint64_t)G::kStreams * S::kResidual);
		}
		if constexpr ( S::kPle )
			if ( index == G::kPleLayer )
				PleTensors(state, t);
		if ( attention_layer )
		{
			Elements(t, "attn_q_norm", "f32", &layer->q_norm, G::kHeadDim);
			Elements(t, "attn_k_norm", "f32", &layer->k_norm, G::kHeadDim);
			if ( S::kIndexer )
			{
				Elements(t, "attn_index_qk", "bf16", &layer->index_qk, (uint64_t)S::kIndexWidth * G::kHidden);
				Elements(t, "attn_index_q_norm", "f32", &layer->index_q_norm, G::kIndexDim);
				Elements(t, "attn_index_k_norm", "f32", &layer->index_k_norm, G::kIndexDim);
			}
		}
		else
		{
			Elements(t, "gdn_ba", "bf16", &layer->gdn_ba, (uint64_t)2u * state->gdn_heads * G::kHidden);
			Elements(t, "gdn_conv", "bf16", &layer->gdn_conv, (uint64_t)state->gdn_channels * QH_CONV);
			Elements(t, "gdn_norm", "bf16", &layer->gdn_norm, QH_VALUE_DIM);
			Elements(t, "gdn_a_log", "f32", &layer->gdn_a_log, state->gdn_heads);
			Elements(t, "gdn_dt_bias", "f32", &layer->gdn_dt_bias, state->gdn_heads);
		}
		return SPARK_STATUS_OK;
	}

	static SparkStatus Arrange(State *state)
	{
		uint32_t recurrent = 0u, attention = 0u;
		for ( uint32_t index = 0u; index < G::kLayers; ++index )
			if ( index % G::kPeriod == G::kPhase )
				state->layers[index].kv_index = attention++;
			else
				state->layers[index].recurrent_index = recurrent++;
		return SPARK_STATUS_OK;
	}

	static uint32_t LaneReady(const State *state)
	{
		return state->lane_arena != 0 && state->dump == 0 ? 1u : 0u;
	}

	static SparkStatus LaneAdmit(State *state, const SparkStageRunnerStep *step)
	{
		(void)state;
		(void)step;
		return SPARK_STATUS_OK;
	}

	static uint32_t RecurrentParts(const State *state, CudaStageRecurrent *parts)
	{
		parts[0] = { state->gdn_state, state->gdn_state_bytes, S::kGdnLayers };
		parts[1] = { (uint8_t *)state->gdn_window, state->gdn_window_bytes, S::kGdnLayers };
		uint32_t count = 2u;
		if constexpr ( S::kPle )
		{
			parts[count++] = { (uint8_t *)state->ple_window, state->ple_window_bytes, 1u };
			parts[count++] = { (uint8_t *)state->ple_history, S::kPleHistory * sizeof(int32_t), 1u };
		}
		if constexpr ( S::kIndexer )
			parts[count++] = { (uint8_t *)state->index_tail, (uint64_t)S::kIndexTail * G::kIndexDim * sizeof(uint16_t), S::kAttnLayers };
		return count;
	}

	static uint32_t ReportFields(const State *state, CudaStageReportField *fields)
	{
		const CudaStageReportField report[] =
		{
			{ "arena_bytes", state->arena_bytes }, { "slots", state->slots }, { "rows", state->max_rows }, { "ffn_rows", state->ffn_rows },
			{ "expert_first", state->expert_first }
		};
		return CudaStageReportOut(fields, report);
	}

	static SparkStatus TapPlan(void *model, const SparkStageRunnerTapPlan *plan)
	{
		State *state = (State *)model;
		int16_t outputs[G::kLayers];
		if ( state == 0 || plan == 0 )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if constexpr ( S::kHyper )
		{
			fprintf(stderr, "%s: taps refused: a layer output here is several hyper-connection streams, not one residual row\n", G::Tag());
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		}
		for ( uint32_t layer = 0u; layer < G::kLayers; ++layer )
			outputs[layer] = -1;
		for ( uint32_t index = 0u; index < plan->entry_count; ++index )
		{
			if ( plan->entries[index].form == SPARK_STAGE_RUNNER_TAP_FORM_FINAL )
				continue;
			if ( plan->entries[index].form != SPARK_STAGE_RUNNER_TAP_FORM_OUTPUT || plan->entries[index].layer >= G::kLayers )
				SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
			outputs[plan->entries[index].layer] = (int16_t)index;
		}
		memcpy(state->tap_output, outputs, sizeof(outputs));
		state->tap_slab = plan->capture != 0u ? plan->slab : 0;
		state->tap_stride = plan->row_stride_elements;
		fprintf(stderr, "%s: taps entries=%u capture=%u; a layer output is its residual after the feed-forward add\n", G::Tag(), plan->entry_count, plan->capture);
		return SPARK_STATUS_OK;
	}

	static uint32_t GraphKey(void *model, const SparkStageRunnerStep *step)
	{
		if ( ((const State *)model)->dump != 0 )
			return 0u;
		if ( step->commit == 0u && step->last_rows_only == 0u && step->sequences == 1u && step->rows > 1u && step->rows <= G::kVerifyRows )
			return step->tree != 0u && !kTreeVerify ? 0u : 2u;
		return 1u;
	}

	static SparkStatus KeepRows(void *model, const SparkStageRunnerStep *step, const uint32_t *kept, void *stream_void)
	{
		State *state = (State *)model;
		SparkGpuStream stream = (SparkGpuStream)stream_void;
		SparkStatus status = SPARK_STATUS_OK;
		if ( G::kVerifyRows == 0u || kept == 0 || step->commit != 0u || step->rows == 0u || step->rows > G::kVerifyRows || step->sequences == 0u ||
			step->sequences > state->slots || (S::kPle && step->token_ids == 0) )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( step->tree != 0u && (!kTreeVerify || step->sequences != 1u || step->slots == 0 || step->sequence_row_indices == 0) )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for ( uint32_t index = 0u; index < G::kLayers && status == SPARK_STATUS_OK; ++index )
		{
			const QwenHybridLayer *layer = &state->layers[index];
			const uint64_t replay = (uint64_t)layer->recurrent_index * G::kVerifyRows;
			if ( index % G::kPeriod == G::kPhase )
			{
				if constexpr ( S::kIndexer )
				{
					if ( QwenSkl2Launch(QWENSKL2_index_tail_128, step->sequences, 128u, 0, stream, state->replay_index +
							(uint64_t)layer->kv_index * G::kVerifyRows * G::kIndexDim, state->index_tail + (uint64_t)layer->kv_index * state->slots * S::kIndexTail *
							G::kIndexDim, step->positions, step->sequence_row_begin, step->sequence_row_indices, step->recurrent_index, 1u, kept,
							G::kIndexDim, G::kIndexRatio) != SPARK_GPU_SUCCESS )
						SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
					status = CudaStageLaunched();
				}
				else if ( step->tree != 0u )
				{
					CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_READ);
					LM_LAUNCH((LmKvShardPathMoveKernel<typename S::Kv,QH_THREADS>), G::kVerifyRows, QH_THREADS, 0, stream,
						state->kv_views[layer->kv_index], step->sequence_of_row, step->sequence_row_indices, kept, step->slots, 2u * S::kKvWidth);
					CudaStageRowState(state, layer->kv_index, SPARK_STAGE_RUNNER_ROW_STATE_WRITTEN);
					status = CudaStageLaunched();
				}
				continue;
			}
			status = GdnRecur(state, layer, step, state->replay_qkv + replay * state->gdn_channels, state->replay_ba + replay * 2u * state->gdn_heads, kept,
				1u, stream);
		}
		if constexpr ( S::kPle )
			if ( status == SPARK_STATUS_OK )
				status = PleRecur(state, step, state->replay_ple, kept, 1u, stream);
		return status;
	}
};
