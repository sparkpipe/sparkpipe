/* Large stage packs exceed 2 GB: 64-bit file offsets are required. */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <inttypes.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_qwen38_27b_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_admission.h"
#include "sparkpipe/spark_stage_kv_client.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "spark_qwen38_27b_stagepack_format.h"
#include "spark_qwen38_27b_dspark_format.h"
#include "spark_qwen38_27b_tp.h"

/*
 * Qwen 3.6 27B resident decode stage host module, PP-Nx native.
 *
 * One process is one STAGE: configuration names the stage count, the stage
 * index and the layer slice; the pack must declare exactly that slice and
 * exactly the computed tensor inventory. Embedding and head ownership are
 * derived from slice position, and the frame transport flags must agree with
 * the position in both directions - a mid-pipeline stage without both
 * transports, or an edge stage with the wrong one, is a refused frame.
 *
 * Execute serves two frame modes, exactly one per frame. DECODE: one next
 * token per row for up to max_active_sequence_count distinct lanes. PREFILL:
 * one lane's consecutive prompt positions, projections and attention batched
 * over every position, the GDN core walked in 64-token chunks on the slot
 * stream; a base-zero frame resets the lane's recurrent state and conv
 * tails, a nonzero base requires a warm lane, and the head stage samples
 * only the final position. Execute is synchronous.
 */

#define SPARK_QWEN38_27B_MODULE_TAG "qwen38_27b_stage"
#define SPARK_QWEN38_27B_MODULE_FUSED_QUERY_COMPONENT_COUNT 2u


static inline float SparkQwen38_27bModuleBf16ToFloat(uint16_t h)
{
	uint32_t u = (uint32_t)h << 16u;
	float f;
	memcpy(&f,&u,sizeof(f));
	return(f);
}

/* fp32 -> bf16 with round-to-nearest-even, matching CUDA __float2bfloat16 and
 * torch's bf16 cast (vLLM truncates the Markov bias and the base+bias sum to
 * bf16 before argmax, so the host sampler must round identically). */
static inline uint16_t SparkQwen38_27bModuleFloatToBf16(float f)
{
	uint32_t u,lsb;
	memcpy(&u,&f,sizeof(u));
	lsb = (u >> 16u) & 1u;
	u += 0x7FFFu + lsb;
	return((uint16_t)(u >> 16u));
}
#define SPARK_QWEN38_27B_MODULE_STAGED_ROW_CAPACITY (SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT + SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS)

typedef struct SparkQwen38_27bModuleSlot
{
	void *cuda_stream;
	uint32_t *input_token_ids;
	uint32_t *output_token_ids;
	void *head_logits_bf16;
	uint32_t *head_candidate_ids_u32;
	uint32_t *head_candidate_counts_u32;
	uint32_t *row_lane_indices;
	uint32_t *slot_mapping;
	uint32_t *context_lengths;
	uint32_t *row_cold;
	uint64_t *row_positions;
	void *hidden_bf16;
	void *normalized_bf16;
	void *delta_bf16;
	void *qkv_bf16;
	void *conv_out_bf16;
	void *z_bf16;
	void *beta_pre_bf16;
	void *decay_pre_bf16;
	float *log_decay_f32;
	float *beta_f32;
	void *core_bf16;
	void *gated_bf16;
	void *q_fused_bf16;
	void *k_bf16;
	void *v_bf16;
	void *head_out_bf16;
	void *ffn_gate_bf16;
	void *ffn_up_bf16;
	float *chunk_qn_f32;
	float *chunk_kn_f32;
	float *chunk_cum_g_f32;
	float *chunk_decay_f32;
	float *chunk_attn_f32;
	float *chunk_w_f32;
	float *chunk_kg_f32;
	float *head_scores_f32;
	uint64_t *head_maxloc_u64;
	uint32_t *mtp_draft_ids;
	void *dspark_tap_buffer;
	void *dspark_scratch;
	/* Set per frame in RunFrame: replay frames walk the GDN STEP path. */
	uint32_t replay_frame;
	uint32_t verify_frame;
	/* per-(rows,prefill) frame graph: warm run first (shared-memory opt-ins
	 * precede capture), capture on second sighting, replay after. */
	cudaGraphExec_t graph_exec;
	uint32_t graph_live;
	uint32_t graph_warm;
	uint32_t graph_rows;
	uint32_t graph_prefill;
	uint32_t capturing;
	uint16_t *dspark_logits_host;
	uint16_t *dspark_hidden_host;
	uint32_t *dspark_mask_token_ids;
	uint32_t mtp_seed_row;
	uint32_t host_row_cold[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t host_slot_mapping[SPARK_QWEN38_27B_MODULE_STAGED_ROW_CAPACITY];
	uint32_t host_context_lengths[SPARK_QWEN38_27B_MODULE_STAGED_ROW_CAPACITY];
	uint32_t host_row_lane_indices[SPARK_QWEN38_27B_MODULE_STAGED_ROW_CAPACITY];
	uint64_t host_row_positions[SPARK_QWEN38_27B_MODULE_STAGED_ROW_CAPACITY];
} SparkQwen38_27bModuleSlot;

typedef struct SparkQwen38_27bDsparkLayerWeights
{
	SparkQwen38_27bLinearView q;
	SparkQwen38_27bLinearView k;
	SparkQwen38_27bLinearView v;
	SparkQwen38_27bLinearView o;
	const void *q_norm_bf16;
	const void *k_norm_bf16;
	const void *input_norm_bf16;
	const void *post_norm_bf16;
	SparkQwen38_27bLinearView gate;
	SparkQwen38_27bLinearView up;
	SparkQwen38_27bLinearView down;
	/* DFlash2 grouped dynamic depthwise conv, one module per sublayer. */
	SparkQwen38_27bLinearView conv_attn_proj;
	const void *conv_attn_base_bf16;
	SparkQwen38_27bLinearView conv_mlp_proj;
	const void *conv_mlp_base_bf16;
} SparkQwen38_27bDsparkLayerWeights;

typedef struct SparkQwen38_27bDsparkWeights
{
	SparkQwen38_27bDsparkLayerWeights layer[SPARK_QWEN38_27B_DSPARK_LAYER_COUNT];
	SparkQwen38_27bLinearView projector;
	SparkQwen38_27bLinearView selector_pred;
	SparkQwen38_27bLinearView selector_succ;
	SparkQwen38_27bLinearView selector_hidden_proj;
	const void *final_norm_bf16;
	const void *hidden_norm_bf16;
	uint16_t *selector_pred_host;
	uint16_t *selector_succ_host;
	uint16_t *selector_hidden_proj_host;
	uint32_t armed;
} SparkQwen38_27bDsparkWeights;

typedef struct SparkQwen38_27bModuleState
{
	SparkStageModuleLedger ledger;
	uint32_t stage_count;
	uint32_t stage_index;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t owns_embedding;
	uint32_t owns_final_head;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint32_t max_active_sequence_count;
	uint32_t pipeline_slot_count;
	uint32_t kv_block_count;
	uint32_t gdn_layer_count;
	uint32_t attn_layer_count;
	uint32_t cache_layer_count;
	uint32_t mtp_armed;
	uint32_t mtp_cache_ordinal;
	uint32_t gdn_snapshot_slot_count;
	float *snapshot_state_f32;
	void *snapshot_tail_bf16;
	float *prefix_state_f32;
	void *prefix_tail_bf16;
	uint32_t gdn_ordinal_by_layer[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	uint32_t attn_ordinal_by_layer[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	uint32_t layer_seen_bits[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	uint32_t global_seen_bits;
	uint32_t mtp_seen_bits;
	SparkQwen38_27bMtpWeights mtp;
	const void *token_embedding_bf16;
	const void *final_norm_weight_bf16;
	const void *lm_head_weight_bf16;
	uint8_t *head_shadow_payload;
	uint8_t *head_shadow_scale;
	float *head_error_norm_f32;
	const void *attention_norm_by_layer[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	const void *mlp_norm_by_layer[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	SparkQwen38_27bGdnLayerWeights gdn_by_layer[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	SparkQwen38_27bAttnLayerWeights attn_by_layer[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	SparkQwen38_27bFfnLayerWeights ffn_by_layer[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	SparkQwen38_27bGdnStatePool gdn_pool;
	void *kv_cache_bf16;
	uint64_t cache_layer_stride;
	uint64_t cache_block_stride;
	uint8_t lane_warm[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_sequence_ids[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_request_generations[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_next_positions[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	SparkQwen38_27bModuleSlot slots[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	atomic_uint slot_states[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	atomic_uint lane_states[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	SparkStageKvClient kv_client;
	SparkQwen38_27bTpState tp;
	SparkQwen38_27bDsparkWeights dspark_weights;
	/* DFlash2 context-KV machinery (upstream precompute_and_store_context_kv):
	 * per-position tap history [8192][5][H] bf16, the fc/normed context
	 * window [2048][H] x2, the staged per-layer K/V [5][2][2056][1024] bf16
	 * (context window + block rows), and the prep positions. */
	void *dflash_taps_history;
	/* persistent draft-side block KV history (the HF DynamicCache shape):
	 * raw k/v rows of every block the drafter ever ran, keyed by position;
	 * specforge/vLLM semantics - the drafter attends its own past blocks */
	void *dflash_block_hist_k;
	void *dflash_block_hist_v;
	uint64_t dflash_hist_pos_host[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_DFLASH_BLOCK_KV_CAP];
	uint32_t dflash_hist_count;
	/* device-side selector front-end output (ids/scores/hproj, compact) */
	void *dspark_sel_out_dev;
	uint32_t *dspark_sel_out_host;
	void *dflash_fc_out;
	void *dflash_ctx_normed;
	void *dflash_ctx_kv;
	/* incremental context cache (position-keyed): the per-layer K/V
	 * projections of committed rows, plus the fc/normed watermark. Committed
	 * positions' taps are final (accepted rows walked their true tokens), so
	 * cached rows equal the full recompute up to kernel-shape rounding. */
	void *dflash_ctx_kv_cache;
	uint64_t dflash_ctx_valid_to;
	uint64_t *dflash_positions;

	uint64_t dflash_positions_host[SPARK_QWEN38_27B_DFLASH2_FRAME_KV_ROWS];
	void *tp_stream;
	atomic_ullong submitted_count;
	atomic_ullong completed_count;
	atomic_ullong rejected_count;
	atomic_ullong failed_count;
	atomic_ullong tokens_emitted;
	/* decode-frame GPU phase profiling (SPARK_QWEN38_27B_PROFILE=1). The host
	 * blocks inside every TP reduce spin, which drains the queued GPU work, so
	 * the spin durations measure the GPU execution of the phase between two
	 * reduces: GDN branch, ATTN branch, FFN, and the head tail. */
	uint32_t profile_enabled;
	uint32_t tap_capture_enabled;
	uint32_t dflash2_state_select;
	uint32_t tap_dump_nth;
	uint32_t tap_capture_count;
	uint64_t profile_gdn_spin_nanos;
	uint64_t profile_attn_spin_nanos;
	uint64_t profile_ffn_spin_nanos;
	uint64_t profile_head_spin_nanos;
	uint64_t profile_stage_nanos;
	uint64_t profile_walk_nanos;
	uint64_t profile_tail_nanos;
	uint32_t graph_frames_replayed;
	uint32_t graph_frames_captured;
	uint32_t graph_frames_plain;
	/* GPU-side per-phase timing via stream events (plain frames; the
	 * spin counters measure host reduce/sync spans, not kernels) */
	cudaEvent_t profile_ev[2];
	float profile_gdn_ms;
	float profile_attn_ms;
	float profile_ffn_ms;
	uint32_t profile_event_frames;
	uint64_t profile_frame_nanos;
	uint32_t profile_frame_count;
	uint32_t graphs_broken;
} SparkQwen38_27bModuleState;

static uint64_t SparkQwen38_27bProfileNow(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ((uint64_t)ts.tv_sec * 1000000000ull) + (uint64_t)ts.tv_nsec;
}

static void SparkQwen38_27bProfilePrint(SparkQwen38_27bModuleState *state, uint64_t frame_nanos)
{
	state->profile_frame_nanos += frame_nanos;
	state->profile_frame_count++;
	if ( (state->profile_frame_count & 7u) == 0u || state->profile_frame_count == 1u )
		fprintf(stderr, "%s gpu_spin_profile frames=%u frame_ms=%.2f gdn_ms=%.2f attn_ms=%.2f ffn_ms=%.2f head_ms=%.2f stage_ms=%.2f walk_ms=%.2f tail_ms=%.2f\n",
			SPARK_QWEN38_27B_MODULE_TAG, state->profile_frame_count,
			(double)state->profile_frame_nanos / 1000000.0,
			(double)state->profile_gdn_spin_nanos / 1000000.0,
			(double)state->profile_attn_spin_nanos / 1000000.0,
			(double)state->profile_ffn_spin_nanos / 1000000.0,
			(double)state->profile_head_spin_nanos / 1000000.0,
			(double)state->profile_stage_nanos / 1000000.0,
			(double)state->profile_walk_nanos / 1000000.0,
			(double)state->profile_tail_nanos / 1000000.0);
	if ( (state->profile_frame_count & 63u) == 0u )
		fprintf(stderr, "%s graph_profile replayed=%u captured=%u plain=%u broken=%u evframes=%u evgdn_ms=%.0f evattn_ms=%.0f evffn_ms=%.0f\n",
			SPARK_QWEN38_27B_MODULE_TAG, state->graph_frames_replayed,
			state->graph_frames_captured, state->graph_frames_plain,
			state->graphs_broken, state->profile_event_frames,
			state->profile_gdn_ms, state->profile_attn_ms,
			state->profile_ffn_ms);
}

extern cudaError_t SparkQwen38_27bConfigureCudaKernels(void);
extern cudaError_t SparkQwen38_27bLaunchRmsNorm(cudaStream_t stream, const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon);
extern cudaError_t SparkQwen38_27bLaunchFusedResidualRmsNorm(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon);
extern cudaError_t SparkQwen38_27bLaunchLinear(cudaStream_t stream, const SparkQwen38_27bLinearView *view, const void *input_bf16, void *output_bf16, uint32_t row_count);
extern cudaError_t SparkQwen38_27bLaunchDsparkTapStore(cudaStream_t stream, const void *hidden_bf16, const uint64_t *row_positions, void *taps_bf16, uint32_t rows, uint32_t tap_index, uint32_t hidden_dim, uint32_t tap_layers);
extern cudaError_t SparkQwen38_27bLaunchDsparkKPrep(cudaStream_t stream, void *k_bf16, const void *k_norm_bf16, const uint64_t *positions, uint32_t rows);
extern cudaError_t SparkQwen38_27bLaunchDsparkQPrep(cudaStream_t stream, void *q_bf16, const void *q_norm_bf16, const uint64_t *positions, uint32_t rows);
extern cudaError_t SparkQwen38_27bLaunchDsparkCacheAttn(cudaStream_t stream, const void *q_bf16, const void *k_bf16, const void *v_bf16, const void *q_norm_bf16, const void *k_norm_bf16, const uint64_t *positions, void *attn_out_bf16, uint32_t block_rows, uint32_t nkv, uint32_t window);
extern cudaError_t SparkQwen38_27bLaunchDsparkSelect(cudaStream_t stream, const void *logits, const void *hidden, const void *hproj_w, void *out, uint32_t block_rows, uint32_t vocab, uint32_t hidden_dim, uint32_t rank, uint32_t top_k);
extern cudaError_t SparkQwen38_27bLaunchDsparkMarkov(cudaStream_t stream, const void *markov_w1_bf16, const void *markov_w2_bf16, const uint32_t *prev_token_ids, uint32_t draft_count, uint32_t rank, void *bias_out, uint32_t vocab);
extern cudaError_t SparkQwen38_27bLaunchDsparkConv(cudaStream_t stream, const void *x_bf16, const void *delta_bf16, const void *base_bf16, void *out_bf16, uint32_t block_size, uint32_t num_groups, uint32_t group_size, uint32_t side);
/* Small-batch GEMM geometry, mirrors the cuda translation unit. */
#define SPARK_QWEN38_27B_SMALL_BATCH_MAX_ROWS 8u
#define SPARK_QWEN38_27B_SMALL_BATCH_TILE_N 64u
#define SPARK_QWEN38_27B_SMALL_BATCH_K_CHUNK 128u
extern cudaError_t SparkQwen38_27bLaunchFfnGateUp(cudaStream_t stream, const void *gate_weight_bf16, const void *up_weight_bf16, const void *input_bf16, void *gated_up_bf16, uint32_t row_count, uint32_t input_dimension, uint32_t output_dimension);
extern cudaError_t SparkQwen38_27bLaunchEmbeddingGather(cudaStream_t stream, const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count);
extern cudaError_t SparkQwen38_27bLaunchConvUpdate(cudaStream_t stream, const void *qkv_bf16, const SparkQwen38_27bGdnLayerWeights *weights, void *conv_out_bf16, const SparkQwen38_27bGdnStatePool *pool, const uint32_t *row_lane_indices, uint32_t row_count, uint32_t gdn_layer_ordinal, uint8_t *row_snap_tails, uint64_t snap_tail_lane_stride, uint64_t snap_tail_layer_stride);
extern cudaError_t SparkQwen38_27bLaunchDecayBeta(cudaStream_t stream, const void *decay_pre_bf16, const void *beta_pre_bf16, const SparkQwen38_27bGdnLayerWeights *weights, float *log_decay_f32, float *beta_f32, uint32_t row_count);
extern cudaError_t SparkQwen38_27bLaunchGdnStep(cudaStream_t stream, const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, const SparkQwen38_27bGdnStatePool *pool, void *core_out_bf16, const uint32_t *row_lane_indices, uint32_t row_count, uint32_t gdn_layer_ordinal, float *row_snap_states, uint64_t snap_lane_stride, uint64_t snap_layer_stride);
extern cudaError_t SparkQwen38_27bLaunchGatedNorm(cudaStream_t stream, const void *core_bf16, const void *z_bf16, const SparkQwen38_27bGdnLayerWeights *weights, void *output_bf16, uint32_t row_count, float epsilon);
extern cudaError_t SparkQwen38_27bLaunchAttnPrepare(cudaStream_t stream, void *q_fused_bf16, const void *k_bf16, const void *v_bf16, const SparkQwen38_27bAttnLayerWeights *weights, void *kv_cache_bf16, const uint32_t *slot_mapping, const uint64_t *row_positions, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, float epsilon);
extern cudaError_t SparkQwen38_27bLaunchAttnDecode(cudaStream_t stream, const void *q_fused_bf16, const void *kv_cache_bf16, const SparkQwen38_27bKvBlockTableView *table, const uint32_t *row_lane_indices, const uint32_t *context_lengths, void *head_out_bf16, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride);
extern cudaError_t SparkQwen38_27bLaunchChunkConv(cudaStream_t stream, const void *qkv_bf16, const SparkQwen38_27bGdnLayerWeights *weights, void *conv_out_bf16, const SparkQwen38_27bGdnStatePool *pool, uint32_t lane_index, uint32_t token_count, uint32_t gdn_layer_ordinal);
extern cudaError_t SparkQwen38_27bLaunchGdnChunk(cudaStream_t stream, const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, float *workspace_qn, float *workspace_kn, float *workspace_cum_g, float *workspace_decay, float *workspace_attn, float *workspace_w, float *workspace_kg, const SparkQwen38_27bGdnStatePool *pool, void *core_out_bf16, uint32_t lane_index, uint32_t token_count, uint32_t gdn_layer_ordinal);
extern cudaError_t SparkQwen38_27bLaunchResidualAdd(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, uint32_t row_count, uint32_t dimension);
extern cudaError_t SparkQwen38_27bLaunchSwiGlu(cudaStream_t stream, const void *gate_bf16, void *up_bf16, uint32_t row_count, uint32_t dimension);
extern cudaError_t SparkQwen38_27bLaunchHeadArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint32_t *token_ids, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count);
extern cudaError_t SparkQwen38_27bLaunchHeadShadowQuantize(cudaStream_t stream, const void *head_bf16, uint8_t *shadow_payload, uint8_t *shadow_scale, float *error_norm, uint32_t candidate_count, uint32_t hidden_dimension);
extern cudaError_t SparkQwen38_27bLaunchHeadScreenedArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint8_t *shadow_payload, const uint8_t *shadow_scale, const float *error_norm, void *logits_bf16, uint32_t *candidate_ids, uint32_t *candidate_counts, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count);
extern cudaError_t SparkQwen38_27bLaunchHeadScreenedArgmaxScore(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint8_t *shadow_payload, const uint8_t *shadow_scale, const float *error_norm, void *scratch_bf16, uint32_t *candidate_ids, uint32_t *candidate_counts, uint32_t *output_token_ids, float *output_scores, uint32_t candidate_offset, uint32_t row_count, uint32_t candidate_count);
extern cudaError_t SparkQwen38_27bLaunchHeadMaxLocPack(cudaStream_t stream, const float *scores_f32, const uint32_t *token_ids_u32, uint64_t *keys_u64, uint32_t row_count);
extern cudaError_t SparkQwen38_27bLaunchHeadMaxLocUnpack(cudaStream_t stream, const uint64_t *keys_u64, uint32_t *token_ids_u32, uint32_t row_count);
extern cudaError_t SparkQwen38_27bTpSetGeometry(uint32_t gdn_qk_channels,uint32_t gdn_value_channels,uint32_t gdn_conv_channels,uint32_t gdn_key_heads,uint32_t gdn_value_heads,uint32_t attn_query_heads,uint32_t attn_kv_heads,uint32_t gdn_qk_channel_base,uint32_t gdn_value_channel_base,uint32_t gdn_key_head_base,uint32_t gdn_value_head_base);

static SparkStatus SparkQwen38_27bModuleConfigure(SparkQwen38_27bModuleState *state)
{
	SparkStatus status;
	status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN38_27B_MODULE_TAG,"SPARK_QWEN38_27B_STAGE_COUNT",1u,SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT,&state->stage_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN38_27B_MODULE_TAG,"SPARK_QWEN38_27B_STAGE_INDEX",0u,SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT - 1u,&state->stage_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN38_27B_MODULE_TAG,"SPARK_QWEN38_27B_STAGE_FIRST_LAYER",0u,SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_LAYER_COUNT - 1u,&state->first_layer_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN38_27B_MODULE_TAG,"SPARK_QWEN38_27B_STAGE_LAYER_COUNT",1u,SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_LAYER_COUNT,&state->layer_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN38_27B_MODULE_TAG,"SPARK_QWEN38_27B_TP_DEGREE",1u,16u,&state->tp_degree);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN38_27B_MODULE_TAG,"SPARK_QWEN38_27B_TP_RANK",0u,15u,&state->tp_rank);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN38_27B_MODULE_TAG,"SPARK_QWEN38_27B_STAGE_MAX_ACTIVE_SEQUENCES",1u,SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,&state->max_active_sequence_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN38_27B_MODULE_TAG,"SPARK_QWEN38_27B_STAGE_PIPELINE_SLOTS",1u,SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,&state->pipeline_slot_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN38_27B_MODULE_TAG,"SPARK_QWEN38_27B_STAGE_KV_BLOCKS",1u,1u << 20u,&state->kv_block_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN38_27B_MODULE_TAG,"SPARK_QWEN38_27B_STAGE_MTP",0u,1u,&state->mtp_armed);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN38_27B_MODULE_TAG,"SPARK_QWEN38_27B_STAGE_GDN_SNAPSHOT_SLOTS",0u,SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_GDN_SNAPSHOT_SLOTS,&state->gdn_snapshot_slot_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( state->stage_index >= state->stage_count || state->first_layer_index + state->layer_count > SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_LAYER_COUNT )
	{
		fprintf(stderr,"%s config_slice_invalid stage=%u/%u slice=%u+%u\n",SPARK_QWEN38_27B_MODULE_TAG,state->stage_index,state->stage_count,state->first_layer_index,state->layer_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	/* MTP is TP-safe: the pack slices the MTP decoder's attention/FFN
	 * exactly like main layers, the fc and norms replicate, the decoder
	 * pass reuses the reduced RunAttnLayer/RunFfn, and the draft argmax
	 * reduces the sharded head with a u64 maxloc. */
	if ( state->tp_rank >= state->tp_degree ||
		(state->tp_degree > 1u && (state->stage_count != 1u || state->stage_index != 0u || state->first_layer_index != 0u || state->layer_count != SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_LAYER_COUNT)) )
	{
		fprintf(stderr,"%s config_tp_invalid degree=%u rank=%u stage=%u/%u slice=%u+%u\n",SPARK_QWEN38_27B_MODULE_TAG,state->tp_degree,state->tp_rank,state->stage_index,state->stage_count,state->first_layer_index,state->layer_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	state->owns_embedding = state->first_layer_index == 0u ? 1u : 0u;
	state->owns_final_head = state->first_layer_index + state->layer_count == SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_LAYER_COUNT ? 1u : 0u;
	if ( (state->stage_index == 0u) != (state->owns_embedding != 0u) || (state->stage_index + 1u == state->stage_count) != (state->owns_final_head != 0u) )
	{
		fprintf(stderr,"%s config_position_mismatch stage=%u/%u slice=%u+%u\n",SPARK_QWEN38_27B_MODULE_TAG,state->stage_index,state->stage_count,state->first_layer_index,state->layer_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( state->mtp_armed != 0u && state->owns_final_head == 0u )
	{
		fprintf(stderr,"%s config_mtp_without_head stage=%u/%u\n",SPARK_QWEN38_27B_MODULE_TAG,state->stage_index,state->stage_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	return(SPARK_STATUS_OK);
}

static void SparkQwen38_27bModuleBuildOrdinals(SparkQwen38_27bModuleState *state)
{
	uint32_t layer;
	for (layer = 0; layer < SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_LAYER_COUNT; layer++)
	{
		state->gdn_ordinal_by_layer[layer] = UINT32_MAX;
		state->attn_ordinal_by_layer[layer] = UINT32_MAX;
	}
	for (layer = state->first_layer_index; layer < state->first_layer_index + state->layer_count; layer++)
	{
		if ( SPARK_QWEN38_27B_MODEL_LAYER_IS_GDN(layer) != 0u )
			state->gdn_ordinal_by_layer[layer] = state->gdn_layer_count++;
		else
			state->attn_ordinal_by_layer[layer] = state->attn_layer_count++;
	}
}

static void SparkQwen38_27bModuleFillLinearView(SparkQwen38_27bLinearView *view, const SparkQwen38_27bStagePackEntry *entry, void *payload, void *scale)
{
	view->abi_version = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION;
	view->weight_format = entry->weight_format;
	view->input_dimension = entry->columns;
	view->output_dimension = entry->rows;
	view->weight_payload = payload;
	view->weight_scale_e8m0 = (const uint8_t *)scale;
	view->weight_payload_bytes = entry->payload_bytes;
	view->weight_scale_bytes = entry->scale_bytes;
}

static SparkStatus SparkQwen38_27bModuleValidateEntry(SparkQwen38_27bModuleState *state, const SparkQwen38_27bStagePackEntry *entry, uint64_t file_bytes, uint32_t *is_global)
{
	SparkQwen38_27bStagePackTensorShape shape;
	uint32_t global = entry->layer_index == SPARK_QWEN38_27B_STAGEPACK_GLOBAL_LAYER ? 1u : 0u;
	memset(&shape, 0, sizeof(shape));
	if ( SparkQwen38_27bStagePackResolvedShape(entry->tensor_kind,global != 0u ? 0u : entry->layer_index,global,state->tp_degree,&shape) != 0 || entry->rows != shape.rows || entry->columns != shape.columns )
	{
		fprintf(stderr,"%s dbg_shape kind=%u layer=%u rows=%u/%u cols=%u/%u\n",SPARK_QWEN38_27B_MODULE_TAG,entry->tensor_kind,entry->layer_index,entry->rows,shape.rows,entry->columns,shape.columns);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	if ( entry->weight_format == SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16_RANS )
	{
		if ( shape.quantizable == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
		if ( (entry->rows % 64u) != 0u || (entry->columns % 128u) != 0u || entry->scale_bytes != 0u || entry->scale_group_size != 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
	}
	else if ( shape.quantizable != 0u )
	{
		if ( entry->weight_format != SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 && entry->weight_format != SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 && entry->weight_format != SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 && entry->weight_format != SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_E8M0B128 )
			return(SPARK_STATUS_VALIDATION_FAILED);
	}
	else if ( entry->weight_format != shape.natural_format )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->weight_format == SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 ? entry->scale_group_size != 32u : ((entry->weight_format == SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 || entry->weight_format == SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_E8M0B128) ? entry->scale_group_size != 128u : entry->scale_group_size != 0u) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->weight_format != SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16_RANS && (entry->payload_bytes != SparkQwen38_27bStagePackPayloadBytes(entry->weight_format,entry->rows,entry->columns) || entry->scale_bytes != SparkQwen38_27bStagePackScaleBytes(entry->weight_format,entry->rows,entry->columns)) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->payload_offset > file_bytes || entry->payload_bytes > file_bytes - entry->payload_offset )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->scale_bytes != 0u && (entry->scale_offset > file_bytes || entry->scale_bytes > file_bytes - entry->scale_offset) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->layer_index == SPARK_QWEN38_27B_STAGEPACK_MTP_LAYER || (global != 0u && (entry->tensor_kind >= SPARK_QWEN38_27B_STAGEPACK_TENSOR_MTP_FC && entry->tensor_kind <= SPARK_QWEN38_27B_STAGEPACK_TENSOR_MTP_FINAL_NORM)) )
	{
		if ( state->owns_final_head == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
	}
	else if ( global == 0u && (entry->layer_index < state->first_layer_index || entry->layer_index >= state->first_layer_index + state->layer_count) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	*is_global = global;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen38_27bModuleBindMtp(SparkQwen38_27bModuleState *state, const SparkQwen38_27bStagePackEntry *entry, void *payload, void *scale)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_MTP_FC: SparkQwen38_27bModuleFillLinearView(&state->mtp.fc,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTENTION_NORM: state->mtp.attention_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_MLP_NORM: state->mtp.mlp_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_GATE: SparkQwen38_27bModuleFillLinearView(&state->mtp.ffn.gate,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_UP: SparkQwen38_27bModuleFillLinearView(&state->mtp.ffn.up,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_DOWN: SparkQwen38_27bModuleFillLinearView(&state->mtp.ffn.down,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_QUERY: SparkQwen38_27bModuleFillLinearView(&state->mtp.attention.query,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_KEY: SparkQwen38_27bModuleFillLinearView(&state->mtp.attention.key,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_VALUE: SparkQwen38_27bModuleFillLinearView(&state->mtp.attention.value,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_OUTPUT: SparkQwen38_27bModuleFillLinearView(&state->mtp.attention.output,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_QUERY_NORM: state->mtp.attention.query_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_KEY_NORM: state->mtp.attention.key_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
}

static SparkStatus SparkQwen38_27bModuleBindGlobal(SparkQwen38_27bModuleState *state, const SparkQwen38_27bStagePackEntry *entry, void *payload)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_EMBEDDING:
		if ( state->owns_embedding == 0u && state->owns_final_head == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
		state->token_embedding_bf16 = payload;
		return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_FINAL_NORM:
		if ( state->owns_final_head == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
		state->final_norm_weight_bf16 = payload;
		return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_LM_HEAD:
		if ( state->owns_final_head == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
		state->lm_head_weight_bf16 = payload;
		return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_MTP_EMBED_NORM: state->mtp.embed_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_MTP_HIDDEN_NORM: state->mtp.hidden_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_MTP_FINAL_NORM: state->mtp.final_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
}

static SparkStatus SparkQwen38_27bModuleBindLayer(SparkQwen38_27bModuleState *state, const SparkQwen38_27bStagePackEntry *entry, void *payload, void *scale)
{
	uint32_t layer = entry->layer_index;
	switch ( entry->tensor_kind )
	{
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTENTION_NORM: state->attention_norm_by_layer[layer] = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_MLP_NORM: state->mlp_norm_by_layer[layer] = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_GATE: SparkQwen38_27bModuleFillLinearView(&state->ffn_by_layer[layer].gate,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_UP: SparkQwen38_27bModuleFillLinearView(&state->ffn_by_layer[layer].up,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_DOWN: SparkQwen38_27bModuleFillLinearView(&state->ffn_by_layer[layer].down,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_QKV: SparkQwen38_27bModuleFillLinearView(&state->gdn_by_layer[layer].qkv,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_GATE: SparkQwen38_27bModuleFillLinearView(&state->gdn_by_layer[layer].gate,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_BETA: SparkQwen38_27bModuleFillLinearView(&state->gdn_by_layer[layer].beta,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_DECAY: SparkQwen38_27bModuleFillLinearView(&state->gdn_by_layer[layer].decay,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_OUTPUT: SparkQwen38_27bModuleFillLinearView(&state->gdn_by_layer[layer].output,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_CONV_WEIGHT: state->gdn_by_layer[layer].conv_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_A_LOG: state->gdn_by_layer[layer].a_log_f32 = (const float *)payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_DT_BIAS: state->gdn_by_layer[layer].dt_bias_f32 = (const float *)payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_NORM: state->gdn_by_layer[layer].gdn_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_QUERY: SparkQwen38_27bModuleFillLinearView(&state->attn_by_layer[layer].query,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_KEY: SparkQwen38_27bModuleFillLinearView(&state->attn_by_layer[layer].key,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_VALUE: SparkQwen38_27bModuleFillLinearView(&state->attn_by_layer[layer].value,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_OUTPUT: SparkQwen38_27bModuleFillLinearView(&state->attn_by_layer[layer].output,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_QUERY_NORM: state->attn_by_layer[layer].query_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_KEY_NORM: state->attn_by_layer[layer].key_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
}

static SparkStatus SparkQwen38_27bModuleLoadEntry(SparkQwen38_27bModuleState *state, FILE *file, const SparkQwen38_27bStagePackEntry *entry, uint64_t file_bytes)
{
	SparkStatus status;
	uint32_t is_global = 0u,bit = 1u << entry->tensor_kind;
	uint32_t *seen;
	void *payload = 0,*scale = 0;
	status = SparkQwen38_27bModuleValidateEntry(state,entry,file_bytes,&is_global);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s pack_entry_invalid kind=%u layer=%u\n",SPARK_QWEN38_27B_MODULE_TAG,entry->tensor_kind,entry->layer_index);
		return(status);
	}
	if ( entry->layer_index == SPARK_QWEN38_27B_STAGEPACK_MTP_LAYER || (is_global != 0u && entry->tensor_kind >= SPARK_QWEN38_27B_STAGEPACK_TENSOR_MTP_FC && entry->tensor_kind <= SPARK_QWEN38_27B_STAGEPACK_TENSOR_MTP_FINAL_NORM) )
		seen = &state->mtp_seen_bits;
	else
		seen = is_global != 0u ? &state->global_seen_bits : &state->layer_seen_bits[entry->layer_index];
	if ( (*seen & bit) != 0u )
	{
		fprintf(stderr,"%s pack_entry_duplicate kind=%u layer=%u\n",SPARK_QWEN38_27B_MODULE_TAG,entry->tensor_kind,entry->layer_index);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	*seen |= bit;
	status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,entry->payload_offset,entry->payload_bytes,&payload);
	if ( status == SPARK_STATUS_OK && entry->scale_bytes != 0u )
		status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,entry->scale_offset,entry->scale_bytes,&scale);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( entry->layer_index == SPARK_QWEN38_27B_STAGEPACK_MTP_LAYER || entry->tensor_kind == SPARK_QWEN38_27B_STAGEPACK_TENSOR_MTP_FC )
		return(SparkQwen38_27bModuleBindMtp(state,entry,payload,scale));
	return(is_global != 0u ? SparkQwen38_27bModuleBindGlobal(state,entry,payload) : SparkQwen38_27bModuleBindLayer(state,entry,payload,scale));
}

static SparkStatus SparkQwen38_27bModuleVerifyCoverage(SparkQwen38_27bModuleState *state)
{
	uint32_t layer,expected_global = 0u,expected_layer;
	if ( state->owns_embedding != 0u || state->owns_final_head != 0u )
		expected_global |= 1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_EMBEDDING;
	if ( state->owns_final_head != 0u )
		expected_global |= (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_FINAL_NORM) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_LM_HEAD);
	if ( state->owns_final_head != 0u )
	{
		uint32_t expected_mtp = (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_MTP_FC) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_MTP_EMBED_NORM) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_MTP_HIDDEN_NORM) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_MTP_FINAL_NORM) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTENTION_NORM) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_MLP_NORM) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_GATE) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_UP) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_DOWN) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_QUERY) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_KEY) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_VALUE) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_OUTPUT) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_QUERY_NORM) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_KEY_NORM);
		if ( state->mtp_seen_bits != expected_mtp )
		{
			fprintf(stderr,"%s pack_mtp_incomplete seen=%08x expected=%08x\n",SPARK_QWEN38_27B_MODULE_TAG,state->mtp_seen_bits,expected_mtp);
			return(SPARK_STATUS_VALIDATION_FAILED);
		}
	}
	if ( state->global_seen_bits != expected_global )
	{
		fprintf(stderr,"%s pack_globals_incomplete seen=%08x expected=%08x\n",SPARK_QWEN38_27B_MODULE_TAG,state->global_seen_bits,expected_global);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	for (layer = state->first_layer_index; layer < state->first_layer_index + state->layer_count; layer++)
	{
		expected_layer = (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTENTION_NORM) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_MLP_NORM) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_GATE) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_UP) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_DOWN);
		if ( SPARK_QWEN38_27B_MODEL_LAYER_IS_GDN(layer) != 0u )
			expected_layer |= (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_QKV) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_GATE) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_BETA) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_DECAY) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_OUTPUT) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_CONV_WEIGHT) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_A_LOG) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_DT_BIAS) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_GDN_NORM);
		else
			expected_layer |= (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_QUERY) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_KEY) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_VALUE) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_OUTPUT) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_QUERY_NORM) | (1u << SPARK_QWEN38_27B_STAGEPACK_TENSOR_ATTN_KEY_NORM);
		if ( state->layer_seen_bits[layer] != expected_layer )
		{
			fprintf(stderr,"%s pack_layer_incomplete layer=%u seen=%08x expected=%08x\n",SPARK_QWEN38_27B_MODULE_TAG,layer,state->layer_seen_bits[layer],expected_layer);
			return(SPARK_STATUS_VALIDATION_FAILED);
		}
	}
	return(SPARK_STATUS_OK);
}

/* Resolve one DSpark drafter pack entry into the dspark_weights struct. */
static SparkStatus SparkQwen38_27bModuleLoadDsparkEntry(
	SparkQwen38_27bModuleState *state,
	const SparkQwen38_27bStagePackEntry *entry,
	void *payload,
	void *scale)
{
	SparkQwen38_27bDsparkWeights *w = &state->dspark_weights;
	uint32_t layer = entry->layer_index;
	/* Global tensors (projector/selector/final-norm/hidden-norm) carry the
	 * 0xFFFFFFFF layer sentinel and resolve to w->... not w->layer[...]. */
	if ( layer >= SPARK_QWEN38_27B_DSPARK_LAYER_COUNT && layer != 0xFFFFFFFFu )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	SparkQwen38_27bDsparkLayerWeights *lw = layer < SPARK_QWEN38_27B_DSPARK_LAYER_COUNT ? &w->layer[layer] : 0;
	switch ( entry->tensor_kind )
	{
	case SPARK_QWEN38_27B_DSPARK_TENSOR_ATTN_QUERY: SparkQwen38_27bModuleFillLinearView(&lw->q,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_DSPARK_TENSOR_ATTN_KEY: SparkQwen38_27bModuleFillLinearView(&lw->k,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_DSPARK_TENSOR_ATTN_VALUE: SparkQwen38_27bModuleFillLinearView(&lw->v,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_DSPARK_TENSOR_ATTN_OUTPUT: SparkQwen38_27bModuleFillLinearView(&lw->o,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_DSPARK_TENSOR_ATTN_QUERY_NORM: lw->q_norm_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_DSPARK_TENSOR_ATTN_KEY_NORM: lw->k_norm_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_DSPARK_TENSOR_ATTENTION_NORM: lw->input_norm_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_DSPARK_TENSOR_MLP_NORM: lw->post_norm_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_DSPARK_TENSOR_FFN_GATE: SparkQwen38_27bModuleFillLinearView(&lw->gate,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_DSPARK_TENSOR_FFN_UP: SparkQwen38_27bModuleFillLinearView(&lw->up,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_DSPARK_TENSOR_FFN_DOWN: SparkQwen38_27bModuleFillLinearView(&lw->down,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_DSPARK_TENSOR_PROJECTOR: SparkQwen38_27bModuleFillLinearView(&w->projector,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_DSPARK_TENSOR_SELECTOR_PRED: SparkQwen38_27bModuleFillLinearView(&w->selector_pred,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_DSPARK_TENSOR_SELECTOR_SUCC: SparkQwen38_27bModuleFillLinearView(&w->selector_succ,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_DSPARK_TENSOR_SELECTOR_HIDDEN_PROJ: SparkQwen38_27bModuleFillLinearView(&w->selector_hidden_proj,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_DSPARK_TENSOR_FINAL_NORM: w->final_norm_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_DSPARK_TENSOR_HIDDEN_NORM: w->hidden_norm_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_DSPARK_TENSOR_CONV_ATTN_BASE: lw->conv_attn_base_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_DSPARK_TENSOR_CONV_ATTN_PROJ: SparkQwen38_27bModuleFillLinearView(&lw->conv_attn_proj,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_DSPARK_TENSOR_CONV_MLP_BASE: lw->conv_mlp_base_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_27B_DSPARK_TENSOR_CONV_MLP_PROJ: SparkQwen38_27bModuleFillLinearView(&lw->conv_mlp_proj,entry,payload,scale); return(SPARK_STATUS_OK);
	default: return(SPARK_STATUS_INVALID_ARGUMENT);
	}
}

/* Load the separate DSpark drafter pack (optional, spec-method dspark only). */
static SparkStatus SparkQwen38_27bModuleLoadDsparkPack(SparkQwen38_27bModuleState *state, const char *path)
{
	SparkQwen38_27bStagePackHeader header;
	SparkQwen38_27bStagePackEntry *directory;
	FILE *file;
	SparkStatus status;
	uint32_t index;
	if ( path == 0 || path[0] == '\0' )
		return(SPARK_STATUS_OK);
	file = fopen(path,"rb");
	if ( file == 0 )
	{
		fprintf(stderr,"%s dspark_pack_open_failed path=%s\n",SPARK_QWEN38_27B_MODULE_TAG,path);
		return(SPARK_STATUS_IO_ERROR);
	}
	status = SparkStageModulePackRead(SPARK_QWEN38_27B_MODULE_TAG,file,0u,&header,sizeof(header));
	if ( status == SPARK_STATUS_OK && (header.magic != SPARK_QWEN38_27B_STAGEPACK_MAGIC || header.hidden_dimension != SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION || header.layer_count != SPARK_QWEN38_27B_DSPARK_LAYER_COUNT || header.attn_query_head_count != SPARK_QWEN38_27B_DSPARK_ATTN_QUERY_HEADS || header.attn_kv_head_count != SPARK_QWEN38_27B_DSPARK_ATTN_KV_HEADS || header.attn_head_dimension != SPARK_QWEN38_27B_DSPARK_ATTN_HEAD_DIMENSION || header.ffn_intermediate_dimension != SPARK_QWEN38_27B_DSPARK_FFN_INTERMEDIATE || header.output_vocab_count != SPARK_QWEN38_27B_MODEL_OUTPUT_VOCAB_COUNT) )
		status = SPARK_STATUS_VALIDATION_FAILED;
	directory = status == SPARK_STATUS_OK ? (SparkQwen38_27bStagePackEntry *)malloc((size_t)header.tensor_count * sizeof(SparkQwen38_27bStagePackEntry)) : 0;
	if ( status == SPARK_STATUS_OK && directory == 0 )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModulePackRead(SPARK_QWEN38_27B_MODULE_TAG,file,header.directory_offset,directory,(uint64_t)header.tensor_count * sizeof(SparkQwen38_27bStagePackEntry));
	for (index = 0u; status == SPARK_STATUS_OK && index < header.tensor_count; index++)
	{
		void *payload = 0, *scale = 0;
		status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,directory[index].payload_offset,directory[index].payload_bytes,&payload);
		if ( status == SPARK_STATUS_OK && directory[index].scale_bytes != 0u )
			status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,directory[index].scale_offset,directory[index].scale_bytes,&scale);
		if ( status == SPARK_STATUS_OK )
			status = SparkQwen38_27bModuleLoadDsparkEntry(state,&directory[index],payload,scale);
	}
	if ( status == SPARK_STATUS_OK )
		state->dspark_weights.armed = 1u;
	if ( status == SPARK_STATUS_OK )
	{
		/* Host mirrors for the host-side selector pass: the two [vocab, rank]
		 * codebooks (the old Markov W1/W2 slots at identical shapes) plus the
		 * small [rank, hidden] hidden projection. */
		const uint64_t codebook_bytes = state->dspark_weights.selector_pred.weight_payload_bytes;
		const uint64_t hidden_proj_bytes = state->dspark_weights.selector_hidden_proj.weight_payload_bytes;
		state->dspark_weights.selector_pred_host = (uint16_t *)malloc((size_t)codebook_bytes);
		state->dspark_weights.selector_succ_host = (uint16_t *)malloc((size_t)codebook_bytes);
		state->dspark_weights.selector_hidden_proj_host = (uint16_t *)malloc((size_t)hidden_proj_bytes);
		if ( state->dspark_weights.selector_pred_host == 0 || state->dspark_weights.selector_succ_host == 0 || state->dspark_weights.selector_hidden_proj_host == 0 )
			status = SPARK_STATUS_CAPACITY_EXCEEDED;
		if ( status == SPARK_STATUS_OK )
		{
			cudaError_t d2h = cudaMemcpy(state->dspark_weights.selector_pred_host,state->dspark_weights.selector_pred.weight_payload,(size_t)codebook_bytes,cudaMemcpyDeviceToHost);
			if ( d2h == cudaSuccess )
				d2h = cudaMemcpy(state->dspark_weights.selector_succ_host,state->dspark_weights.selector_succ.weight_payload,(size_t)codebook_bytes,cudaMemcpyDeviceToHost);
			if ( d2h == cudaSuccess )
				d2h = cudaMemcpy(state->dspark_weights.selector_hidden_proj_host,state->dspark_weights.selector_hidden_proj.weight_payload,(size_t)hidden_proj_bytes,cudaMemcpyDeviceToHost);
			status = SparkStageModuleCudaStatus(SPARK_QWEN38_27B_MODULE_TAG,d2h,"dspark_selector_d2h");
		}
	}
	if ( status == SPARK_STATUS_OK )
	{
		/* context-KV machinery: taps history 8192x5xH, fc/norm window
		 * 2048xH x2, per-layer staged KV 5x2x2056x1024, positions. */
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)8192u * 5u * SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION * 2u,&state->dflash_taps_history);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)2048u * SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION * 2u,&state->dflash_fc_out);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)2048u * SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION * 2u,&state->dflash_ctx_normed);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)5u * 2u * (2048u + 8u + SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_DFLASH_BLOCK_KV_CAP) * 1024u * 2u,&state->dflash_ctx_kv);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)5u * 2u * 2048u * 1024u * 2u,&state->dflash_ctx_kv_cache);
		state->dflash_ctx_valid_to = 0u;
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)5u * SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_DFLASH_BLOCK_KV_CAP * 1024u * 2u,&state->dflash_block_hist_k);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)5u * SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_DFLASH_BLOCK_KV_CAP * 1024u * 2u,&state->dflash_block_hist_v);
		if ( status == SPARK_STATUS_OK )
			state->dflash_hist_count = 0u;
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,(2048u + 8u + SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_DFLASH_BLOCK_KV_CAP) * sizeof(uint64_t),(void **)&state->dflash_positions);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)(SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_DSPARK_BLOCK_SIZE - 1u) * (2u * SPARK_QWEN38_27B_DSPARK_SELECTOR_TOP_K + SPARK_QWEN38_27B_DSPARK_SELECTOR_RANK) * 4u,&state->dspark_sel_out_dev);
		if ( status == SPARK_STATUS_OK )
		{
			state->dspark_sel_out_host = (uint32_t *)malloc((size_t)(SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_DSPARK_BLOCK_SIZE - 1u) * (2u * SPARK_QWEN38_27B_DSPARK_SELECTOR_TOP_K + SPARK_QWEN38_27B_DSPARK_SELECTOR_RANK) * 4u);
			if ( state->dspark_sel_out_host == 0 )
				status = SPARK_STATUS_CAPACITY_EXCEEDED;
		}
	}
	free(directory);
	fclose(file);
	return(status);
}

static SparkStatus SparkQwen38_27bModuleLoadPack(SparkQwen38_27bModuleState *state, const char *path)
{
	SparkQwen38_27bStagePackHeader header,expected;
	SparkQwen38_27bStagePackEntry *directory;
	FILE *file;
	SparkStatus status;
	int32_t compare;
	uint32_t index;
	file = fopen(path,"rb");
	if ( file == 0 )
	{
		fprintf(stderr,"%s pack_open_failed path=%s\n",SPARK_QWEN38_27B_MODULE_TAG,path);
		return(SPARK_STATUS_IO_ERROR);
	}
	status = SparkStageModulePackRead(SPARK_QWEN38_27B_MODULE_TAG,file,0u,&header,sizeof(header));
	if ( status == SPARK_STATUS_OK )
	{
		SparkQwen38_27bStagePackExpectedGeometry(&expected,state->first_layer_index,state->layer_count);
		expected.tp_degree = state->tp_degree;
		expected.tp_rank = state->tp_rank;
		compare = SparkQwen38_27bStagePackCompareGeometry(&header,&expected);
		if ( compare != 0 || header.directory_offset != SPARK_QWEN38_27B_STAGEPACK_HEADER_BYTES )
		{
			fprintf(stderr,"%s pack_geometry_mismatch field=%s\n",SPARK_QWEN38_27B_MODULE_TAG,compare != 0 ? SparkQwen38_27bStagePackGeometryFieldName(compare) : "directory_offset");
			status = SPARK_STATUS_VALIDATION_FAILED;
		}
	}
	/* Device-memory preflight (the watchdog-restart SEGV fix): a second
	 * daemon instance starting while another holds the GPU reaches
	 * cudaMemcpy with corrupted CUDA state and jumps to a garbage PC
	 * inside the pack loop (measured: core at LoadDeviceRegion -> 0x2480).
	 * Refusing cleanly here turns that crash into the daemon's normal
	 * phase=adapter_initialize capacity_exceeded report. */
	if ( status == SPARK_STATUS_OK )
	{
		size_t device_free = 0u,device_total = 0u;
		if ( cudaMemGetInfo(&device_free,&device_total) == cudaSuccess &&
			(uint64_t)device_free < header.file_bytes )
		{
			fprintf(stderr,"%s pack_device_memory_insufficient free=%llu pack=%llu (another instance holding the GPU?)\n",
				SPARK_QWEN38_27B_MODULE_TAG,(unsigned long long)device_free,(unsigned long long)header.file_bytes);
			fclose(file);
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
	}
	directory = status == SPARK_STATUS_OK ? (SparkQwen38_27bStagePackEntry *)malloc((size_t)header.tensor_count * sizeof(SparkQwen38_27bStagePackEntry)) : 0;
	if ( status == SPARK_STATUS_OK && directory == 0 )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModulePackRead(SPARK_QWEN38_27B_MODULE_TAG,file,header.directory_offset,directory,(uint64_t)header.tensor_count * sizeof(SparkQwen38_27bStagePackEntry));
	for (index = 0; status == SPARK_STATUS_OK && index < header.tensor_count; index++)
		status = SparkQwen38_27bModuleLoadEntry(state,file,&directory[index],header.file_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen38_27bModuleVerifyCoverage(state);
	free(directory);
	fclose(file);
	return(status);
}

// One-time MXFP4 shadow of the lm_head plus per-neuron certified error
// norms, the mimo25 screened-head pattern; head stage only, built
// synchronously at initialize.
static SparkStatus SparkQwen38_27bModuleBuildHeadShadow(SparkQwen38_27bModuleState *state)
{
	uint64_t vocab = state->tp.head_rows,dim = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
	SparkStatus status;
	if ( state->owns_final_head == 0u )
		return(SPARK_STATUS_OK);
	status = SparkStageModuleDeviceAllocate(&state->ledger,(vocab * dim) / 2u,(void **)&state->head_shadow_payload);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(vocab * dim) / 32u,(void **)&state->head_shadow_scale);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,vocab * sizeof(float),(void **)&state->head_error_norm_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_QWEN38_27B_MODULE_TAG,SparkQwen38_27bLaunchHeadShadowQuantize(0,state->lm_head_weight_bf16,state->head_shadow_payload,state->head_shadow_scale,state->head_error_norm_f32,(uint32_t)vocab,(uint32_t)dim),"head_shadow_quantize");
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_QWEN38_27B_MODULE_TAG,cudaDeviceSynchronize(),"head_shadow_sync");
	return(status);
}

/* Tensor-parallel bootstrap: one dedicated stream for the synchronous
 * collective submissions, the collective itself, and the per-rank device
 * geometry table. Must precede every allocation that derives per-rank
 * dimensions. */
static SparkStatus SparkQwen38_27bModuleInitializeTp(SparkQwen38_27bModuleState *state)
{
	cudaStream_t stream = 0;
	SparkStatus status = SparkStageModuleCudaStatus(SPARK_QWEN38_27B_MODULE_TAG,cudaStreamCreate(&stream),"tp_stream_create");
	if ( status != SPARK_STATUS_OK )
		return(status);
	state->tp_stream = stream;
	status = SparkQwen38_27bTpInitialize(&state->tp,state->tp_degree,state->tp_rank,state->max_active_sequence_count,state->pipeline_slot_count,stream);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkStageModuleCudaStatus(SPARK_QWEN38_27B_MODULE_TAG,
		SparkQwen38_27bTpSetGeometry(
			state->tp.gdn_qk_channels,state->tp.gdn_value_channels,
			state->tp.gdn_conv_channels,state->tp.gdn_key_heads,
			state->tp.gdn_value_heads,state->tp.attn_query_heads,
			state->tp.attn_kv_heads,
			state->tp_rank * state->tp.gdn_qk_channels,
			state->tp_rank * state->tp.gdn_value_channels,
			state->tp_rank * state->tp.gdn_key_heads,
			state->tp_rank * state->tp.gdn_value_heads),
		"tp_set_geometry"));
}

/* Stream-ordered BF16 hidden all-reduce of slot->delta_bf16: the reduction
 * is enqueued on the slot stream between the producing projection and the
 * consuming kernels, so no stream drain is needed; the frame's own end-of-
 * execute synchronization covers every collective in flight. */
static SparkStatus SparkQwen38_27bModuleTpReduceDelta(SparkQwen38_27bModuleState *state, SparkQwen38_27bModuleSlot *slot, uint32_t rows)
{
	SparkStatus status;
	if ( state->tp_degree <= 1u )
	{
		/* profile-only sync: capture-aware (the FFN's copy was the THIRD
		 * capture blocker - it invalidated the capture round with the
		 * sticky error surfacing at the next checked ffn launch) */
		if ( state->profile_enabled != 0u && slot->capturing == 0u )
			(void)cudaStreamSynchronize((cudaStream_t)slot->cuda_stream);
		return(SPARK_STATUS_OK);
	}
	status = SparkQwen38_27bTpReduceHidden(&state->tp,slot->delta_bf16,rows,slot->cuda_stream);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr, "%s tp_reduce_delta_failed status=%d rows=%u\n", SPARK_QWEN38_27B_MODULE_TAG, (int)status, rows);
	return status;
}

static SparkStatus SparkQwen38_27bModuleAllocatePools(SparkQwen38_27bModuleState *state)
{
	SparkStatus status = SPARK_STATUS_OK;
	uint64_t state_elements,tail_elements,cache_elements;
	state->gdn_pool.abi_version = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_GDN_STATE_POOL_ABI_VERSION;
	state->gdn_pool.lane_capacity = state->max_active_sequence_count;
	state->gdn_pool.gdn_layer_count = state->gdn_layer_count;
	state->gdn_pool.state_layer_stride_elements = (uint64_t)state->tp.gdn_value_heads * SPARK_QWEN38_27B_MODEL_GDN_HEAD_KEY_DIMENSION * SPARK_QWEN38_27B_MODEL_GDN_HEAD_VALUE_DIMENSION;
	state->gdn_pool.state_lane_stride_elements = state->gdn_pool.state_layer_stride_elements * state->gdn_layer_count;
	state->gdn_pool.conv_tail_layer_stride_elements = (uint64_t)state->tp.gdn_conv_channels * (SPARK_QWEN38_27B_MODEL_GDN_CONV_KERNEL - 1u);
	state->gdn_pool.conv_tail_lane_stride_elements = state->gdn_pool.conv_tail_layer_stride_elements * state->gdn_layer_count;

	if ( state->gdn_layer_count != 0u )
	{
		state_elements = state->gdn_pool.state_lane_stride_elements * state->max_active_sequence_count;
		tail_elements = state->gdn_pool.conv_tail_lane_stride_elements * state->max_active_sequence_count;
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,state_elements * sizeof(float),(void **)&state->gdn_pool.state_f32);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,tail_elements * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES,&state->gdn_pool.conv_tail_bf16);
	}
	// The MTP decoder owns the LAST cache layer on an armed head stage; its
	// ordinal is attn_layer_count, so main-layer ordinals are undisturbed.
	state->mtp_cache_ordinal = state->attn_layer_count;
	state->cache_layer_count = state->attn_layer_count + (state->owns_final_head != 0u && state->mtp_armed != 0u ? SPARK_QWEN38_27B_MODEL_MTP_LAYER_COUNT : 0u);
	if ( status == SPARK_STATUS_OK && state->cache_layer_count != 0u )
	{
		state->cache_layer_stride = (uint64_t)SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS * (uint64_t)state->tp.attn_kv_heads * SPARK_QWEN38_27B_MODEL_ATTN_HEAD_DIMENSION * 2u;
		state->cache_block_stride = state->cache_layer_stride * state->cache_layer_count;
		cache_elements = state->cache_block_stride * state->kv_block_count;
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,cache_elements * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES,&state->kv_cache_bf16);
	}
	if ( status == SPARK_STATUS_OK && state->gdn_layer_count != 0u && state->gdn_snapshot_slot_count != 0u )
	{
		status = SparkStageModuleDeviceAllocate(&state->ledger,state->gdn_pool.state_lane_stride_elements * sizeof(float) * state->gdn_snapshot_slot_count,(void **)&state->snapshot_state_f32);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,state->gdn_pool.conv_tail_lane_stride_elements * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES * state->gdn_snapshot_slot_count,&state->snapshot_tail_bf16);
	}
	/* persistent prefix-cache GDN pool: 8 slots, same strides as the verify
	 * pool but owned by the adapter's prefix entries (survives sequences) */
	if ( status == SPARK_STATUS_OK && state->gdn_layer_count != 0u )
	{
		status = SparkStageModuleDeviceAllocate(&state->ledger,state->gdn_pool.state_lane_stride_elements * sizeof(float) * SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_PREFIX_GDN_SLOT_COUNT,(void **)&state->prefix_state_f32);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,state->gdn_pool.conv_tail_lane_stride_elements * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES * SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_PREFIX_GDN_SLOT_COUNT,&state->prefix_tail_bf16);
	}
	return(status);
}

static SparkStatus SparkQwen38_27bModuleAllocateSlotControl(SparkQwen38_27bModuleState *state, SparkQwen38_27bModuleSlot *slot)
{
	uint64_t rows = state->max_active_sequence_count,staged = rows + SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS;
	SparkStatus status;
	cudaStream_t stream = 0;
	status = SparkStageModuleCudaStatus(SPARK_QWEN38_27B_MODULE_TAG,cudaStreamCreate(&stream),"cudaStreamCreate");
	if ( status != SPARK_STATUS_OK )
		return(status);
	slot->cuda_stream = stream;
	status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->input_token_ids);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->output_token_ids);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * SPARK_QWEN38_27B_MODEL_OUTPUT_VOCAB_COUNT * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES,&slot->head_logits_bf16);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_HEAD_SCREEN_CAP * sizeof(uint32_t),(void **)&slot->head_candidate_ids_u32);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->head_candidate_counts_u32);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(float),(void **)&slot->head_scores_f32);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint64_t),(void **)&slot->head_maxloc_u64);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,staged * sizeof(uint32_t),(void **)&slot->row_lane_indices);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,staged * sizeof(uint32_t),(void **)&slot->slot_mapping);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,staged * sizeof(uint32_t),(void **)&slot->context_lengths);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->row_cold);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,staged * sizeof(uint64_t),(void **)&slot->row_positions);
	if ( status == SPARK_STATUS_OK && state->mtp_armed != 0u )
		status = SparkStageModuleDeviceAllocate(&state->ledger,SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS * sizeof(uint32_t),(void **)&slot->mtp_draft_ids);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,SPARK_QWEN38_27B_DSPARK_TARGET_TAP_COUNT * SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES,&slot->dspark_tap_buffer);
	/* DFlash2 scratch (bf16): context [H] + block hidden + conv-prepared h +
	 * the two per-layer conv coefficient planes [B, 2*taps*groups] + Q + K/V +
	 * attn out + norm + ffn (17408) + logits, all x block_size (8). */
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)(1u*SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION + 3u*SPARK_QWEN38_27B_DSPARK_BLOCK_SIZE*SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION + 2u*SPARK_QWEN38_27B_DSPARK_CONV_PROJ_ROWS*SPARK_QWEN38_27B_DSPARK_BLOCK_SIZE + 2u*(SPARK_QWEN38_27B_DSPARK_BLOCK_SIZE+1u)*SPARK_QWEN38_27B_DSPARK_ATTN_KV_HEADS*SPARK_QWEN38_27B_DSPARK_ATTN_HEAD_DIMENSION + 2u*SPARK_QWEN38_27B_DSPARK_BLOCK_SIZE*SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION + 2u*SPARK_QWEN38_27B_DSPARK_BLOCK_SIZE*SPARK_QWEN38_27B_DSPARK_FFN_INTERMEDIATE + SPARK_QWEN38_27B_DSPARK_BLOCK_SIZE*SPARK_QWEN38_27B_MODEL_OUTPUT_VOCAB_COUNT) * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES,&slot->dspark_scratch);
	if ( status == SPARK_STATUS_OK )
		slot->dspark_logits_host = (uint16_t *)malloc((size_t)SPARK_QWEN38_27B_DSPARK_BLOCK_SIZE * SPARK_QWEN38_27B_MODEL_OUTPUT_VOCAB_COUNT * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		slot->dspark_hidden_host = (uint16_t *)malloc((size_t)SPARK_QWEN38_27B_DSPARK_BLOCK_SIZE * SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK && (slot->dspark_logits_host == 0 || slot->dspark_hidden_host == 0) )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(SPARK_QWEN38_27B_DSPARK_BLOCK_SIZE - 1u) * sizeof(uint32_t),(void **)&slot->dspark_mask_token_ids);
	if ( status == SPARK_STATUS_OK )
	{
		uint32_t host_mask[SPARK_QWEN38_27B_DSPARK_BLOCK_SIZE - 1u];
		uint32_t i;
		for (i = 0u; i < SPARK_QWEN38_27B_DSPARK_BLOCK_SIZE - 1u; i++)
			host_mask[i] = SPARK_QWEN38_27B_DSPARK_MASK_TOKEN_ID;
		status = SparkStageModuleCudaStatus(SPARK_QWEN38_27B_MODULE_TAG,cudaMemcpy(slot->dspark_mask_token_ids,host_mask,(SPARK_QWEN38_27B_DSPARK_BLOCK_SIZE - 1u) * sizeof(uint32_t),cudaMemcpyHostToDevice),"dspark_mask_ids");
	}
	return(status);
}

// Slot-owned GDN chunk workspace, the exact view layout the chunk launcher
// documents (per head: qn/kn/w/kg 64 x 128, decay/attn 64 x 64, cum_g 64).
// Only stages that own GDN layers pay for it.
static SparkStatus SparkQwen38_27bModuleAllocateSlotChunkWorkspace(SparkQwen38_27bModuleState *state, SparkQwen38_27bModuleSlot *slot)
{
	uint64_t heads = state->tp.gdn_value_heads,chunk = SPARK_QWEN38_27B_MODEL_GDN_CHUNK_TOKENS;
	uint64_t vector_bytes = heads * chunk * SPARK_QWEN38_27B_MODEL_GDN_HEAD_KEY_DIMENSION * sizeof(float);
	uint64_t matrix_bytes = heads * chunk * chunk * sizeof(float);
	SparkStatus status;
	if ( state->gdn_layer_count == 0u )
		return(SPARK_STATUS_OK);
	status = SparkStageModuleDeviceAllocate(&state->ledger,vector_bytes,(void **)&slot->chunk_qn_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,vector_bytes,(void **)&slot->chunk_kn_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,heads * chunk * sizeof(float),(void **)&slot->chunk_cum_g_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,matrix_bytes,(void **)&slot->chunk_decay_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,matrix_bytes,(void **)&slot->chunk_attn_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,heads * chunk * SPARK_QWEN38_27B_MODEL_GDN_HEAD_VALUE_DIMENSION * sizeof(float),(void **)&slot->chunk_w_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,vector_bytes,(void **)&slot->chunk_kg_f32);
	return(status);
}

static SparkStatus SparkQwen38_27bModuleAllocateSlot(SparkQwen38_27bModuleState *state, SparkQwen38_27bModuleSlot *slot)
{
	uint64_t rows = state->max_active_sequence_count;
	uint64_t attn_query_dim = (uint64_t)state->tp.attn_query_heads * SPARK_QWEN38_27B_MODEL_ATTN_HEAD_DIMENSION;
	uint64_t attn_kv_dim = (uint64_t)state->tp.attn_kv_heads * SPARK_QWEN38_27B_MODEL_ATTN_HEAD_DIMENSION;
	SparkStatus status = SparkQwen38_27bModuleAllocateSlotControl(state,slot);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES,&slot->hidden_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES,&slot->normalized_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES,&slot->delta_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * state->tp.gdn_conv_channels * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES,&slot->qkv_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * state->tp.gdn_conv_channels * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES,&slot->conv_out_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN38_27B_MODEL_GDN_VALUE_DIMENSION * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES,&slot->z_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN38_27B_MODEL_GDN_VALUE_HEAD_COUNT * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES,&slot->beta_pre_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN38_27B_MODEL_GDN_VALUE_HEAD_COUNT * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES,&slot->decay_pre_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * state->tp.gdn_value_heads * sizeof(float),(void **)&slot->log_decay_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * state->tp.gdn_value_heads * sizeof(float),(void **)&slot->beta_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * state->tp.gdn_value_channels * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES,&slot->core_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * state->tp.gdn_value_channels * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES,&slot->gated_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN38_27B_MODULE_FUSED_QUERY_COMPONENT_COUNT * attn_query_dim * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES,&slot->q_fused_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * attn_kv_dim * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES,&slot->k_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * attn_kv_dim * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES,&slot->v_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * attn_query_dim * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES,&slot->head_out_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * state->tp.ffn_intermediate * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES,&slot->ffn_gate_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * state->tp.ffn_intermediate * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES,&slot->ffn_up_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen38_27bModuleAllocateSlotChunkWorkspace(state,slot);
	return(status);
}

static cudaError_t SparkQwen38_27bModuleRunGdnCoreDecodeSnap(SparkQwen38_27bModuleState *state, SparkQwen38_27bModuleSlot *slot, const SparkQwen38_27bGdnLayerWeights *weights, uint32_t rows, uint32_t ordinal, float *row_snap_states, uint64_t snap_lane_stride, uint64_t snap_layer_stride, uint8_t *row_snap_tails, uint64_t snap_tail_lane_stride, uint64_t snap_tail_layer_stride);

static cudaError_t SparkQwen38_27bModuleRunGdnCoreDecode(SparkQwen38_27bModuleState *state, SparkQwen38_27bModuleSlot *slot, const SparkQwen38_27bGdnLayerWeights *weights, uint32_t rows, uint32_t ordinal)
{
	return(SparkQwen38_27bModuleRunGdnCoreDecodeSnap(state,slot,weights,rows,ordinal,0,0,0,0,0,0));
}

static cudaError_t SparkQwen38_27bModuleRunGdnCoreDecodeSnap(SparkQwen38_27bModuleState *state, SparkQwen38_27bModuleSlot *slot, const SparkQwen38_27bGdnLayerWeights *weights, uint32_t rows, uint32_t ordinal, float *row_snap_states, uint64_t snap_lane_stride, uint64_t snap_layer_stride, uint8_t *row_snap_tails, uint64_t snap_tail_lane_stride, uint64_t snap_tail_layer_stride)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	/* Cold flags are per-frame (the slot's uploaded row_cold), not pool
	 * state: hand the kernels a per-call pool view so concurrent slots never
	 * race the shared pool's pointer. */
	SparkQwen38_27bGdnStatePool pool = state->gdn_pool;
	pool.state_cold_by_row = slot->row_cold;
	error = SparkQwen38_27bLaunchConvUpdate(stream,slot->qkv_bf16,weights,slot->conv_out_bf16,&pool,slot->row_lane_indices,rows,ordinal,row_snap_tails,snap_tail_lane_stride,snap_tail_layer_stride);
	if ( error != cudaSuccess && rows >= 32u )
		fprintf(stderr, "%s gdn_core conv_failed rows=%u err=%d\n", SPARK_QWEN38_27B_MODULE_TAG, rows, (int)error);
	if ( error == cudaSuccess )
		error = SparkQwen38_27bLaunchDecayBeta(stream,slot->decay_pre_bf16,slot->beta_pre_bf16,weights,slot->log_decay_f32,slot->beta_f32,rows);
	if ( error != cudaSuccess && rows >= 32u )
		fprintf(stderr, "%s gdn_core decaybeta_failed rows=%u err=%d\n", SPARK_QWEN38_27B_MODULE_TAG, rows, (int)error);
	if ( error == cudaSuccess )
		error = SparkQwen38_27bLaunchGdnStep(stream,slot->conv_out_bf16,slot->log_decay_f32,slot->beta_f32,&pool,slot->core_bf16,slot->row_lane_indices,rows,ordinal,row_snap_states,snap_lane_stride,snap_layer_stride);
	if ( error != cudaSuccess && rows >= 32u )
		fprintf(stderr, "%s gdn_core gdnstep_failed rows=%u err=%d\n", SPARK_QWEN38_27B_MODULE_TAG, rows, (int)error);
	return(error);
}

// Prefill GDN core: conv over the whole frame with the carried tail, then
// the 64-token chunk sequence per slice of the frame; looping chunks on the
// one slot stream serializes the state dependency for free.
static cudaError_t SparkQwen38_27bModuleRunGdnCorePrefill(SparkQwen38_27bModuleState *state, SparkQwen38_27bModuleSlot *slot, const SparkQwen38_27bGdnLayerWeights *weights, uint32_t lane, uint32_t rows, uint32_t ordinal)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint32_t start,count;
	uint64_t conv_bytes,core_bytes,head_offset;
	cudaError_t error;
	error = SparkQwen38_27bLaunchChunkConv(stream,slot->qkv_bf16,weights,slot->conv_out_bf16,&state->gdn_pool,lane,rows,ordinal);
	if ( error == cudaSuccess )
		error = SparkQwen38_27bLaunchDecayBeta(stream,slot->decay_pre_bf16,slot->beta_pre_bf16,weights,slot->log_decay_f32,slot->beta_f32,rows);
	for (start = 0; error == cudaSuccess && start < rows; start += SPARK_QWEN38_27B_MODEL_GDN_CHUNK_TOKENS)
	{
		count = rows - start < SPARK_QWEN38_27B_MODEL_GDN_CHUNK_TOKENS ? rows - start : SPARK_QWEN38_27B_MODEL_GDN_CHUNK_TOKENS;
		conv_bytes = (uint64_t)start * SPARK_QWEN38_27B_MODEL_GDN_CONV_CHANNELS * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES;
		core_bytes = (uint64_t)start * SPARK_QWEN38_27B_MODEL_GDN_VALUE_DIMENSION * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES;
		head_offset = (uint64_t)start * SPARK_QWEN38_27B_MODEL_GDN_VALUE_HEAD_COUNT;
		error = SparkQwen38_27bLaunchGdnChunk(stream,(const void *)((const uint8_t *)slot->conv_out_bf16 + conv_bytes),slot->log_decay_f32 + head_offset,slot->beta_f32 + head_offset,slot->chunk_qn_f32,slot->chunk_kn_f32,slot->chunk_cum_g_f32,slot->chunk_decay_f32,slot->chunk_attn_f32,slot->chunk_w_f32,slot->chunk_kg_f32,&state->gdn_pool,(void *)((uint8_t *)slot->core_bf16 + core_bytes),lane,count,ordinal);
	}
	return(error);
}

/*
 * Replay GDN core (the DSV4 session's silent-divergence fix, unified 2bd2673):
 * the committed positions a verify frame destroyed, re-walked through the
 * DECODE path so the rebuilt state is bit-identical to the state a no-spec run
 * would hold at the same positions. The step kernels are row-indexed, so the
 * frame's rows are staged as one lane repeated - the same shape a decode batch
 * of that many rows would present - and every row is WARM (the restore put the
 * lane's state back before this walk).
 */
/* Verify GDN core: the step path (one lane repeated) with per-row state
 * checkpoints into dflash_row_state[layer][row] (the vLLM select shape). */
static cudaError_t SparkQwen38_27bModuleRunGdnCoreReplaySnap(SparkQwen38_27bModuleState *state, SparkQwen38_27bModuleSlot *slot, const SparkQwen38_27bGdnLayerWeights *weights, uint32_t lane, uint32_t rows, uint32_t ordinal)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint32_t row;
	cudaError_t error;
	if ( rows > 8u || state->snapshot_state_f32 == 0 || state->gdn_snapshot_slot_count < SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_VERIFY_CHECKPOINT_SLOT_BASE + 8u )
		return(cudaErrorInvalidValue);
	for (row = 0u; row < rows; row++)
	{
		slot->host_row_lane_indices[row] = lane;
		slot->host_row_cold[row] = 0u;
	}
	error = cudaMemcpyAsync(slot->row_lane_indices,slot->host_row_lane_indices,(size_t)rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->row_cold,slot->host_row_cold,(size_t)rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = SparkQwen38_27bModuleRunGdnCoreDecodeSnap(state,slot,weights,rows,ordinal,(float *)state->snapshot_state_f32 + (uint64_t)SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_VERIFY_CHECKPOINT_SLOT_BASE * state->gdn_pool.state_lane_stride_elements,state->gdn_pool.state_lane_stride_elements,state->gdn_pool.state_layer_stride_elements,(uint8_t *)state->snapshot_tail_bf16 + (uint64_t)SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_VERIFY_CHECKPOINT_SLOT_BASE * state->gdn_pool.conv_tail_lane_stride_elements * 2u,state->gdn_pool.conv_tail_lane_stride_elements * 2u,state->gdn_pool.conv_tail_layer_stride_elements * 2u);
	return(error);
}

static cudaError_t SparkQwen38_27bModuleRunGdnCoreReplay(SparkQwen38_27bModuleState *state, SparkQwen38_27bModuleSlot *slot, const SparkQwen38_27bGdnLayerWeights *weights, uint32_t lane, uint32_t rows, uint32_t ordinal)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint32_t row;
	cudaError_t error;
	if ( rows > SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT )
		return(cudaErrorInvalidValue);
	for (row = 0u; row < rows; row++)
	{
		slot->host_row_lane_indices[row] = lane;
		slot->host_row_cold[row] = 0u;
	}
	error = cudaMemcpyAsync(slot->row_lane_indices,slot->host_row_lane_indices,(size_t)rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->row_cold,slot->host_row_cold,(size_t)rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = SparkQwen38_27bModuleRunGdnCoreDecode(state,slot,weights,rows,ordinal);
	return(error);
}

static SparkStatus SparkQwen38_27bModuleRunGdnLayer(SparkQwen38_27bModuleState *state, SparkQwen38_27bModuleSlot *slot, const SparkQwen38_27bPrefillFrameView *prefill, uint32_t layer, uint32_t rows)
{
	const SparkQwen38_27bGdnLayerWeights *weights = &state->gdn_by_layer[layer];
	uint32_t ordinal = state->gdn_ordinal_by_layer[layer];
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	error = SparkQwen38_27bLaunchLinear(stream,&weights->qkv,slot->normalized_bf16,slot->qkv_bf16,rows);
	if ( error != cudaSuccess && rows >= 32u )
		fprintf(stderr, "%s gdn_qkv_failed rows=%u err=%d\n", SPARK_QWEN38_27B_MODULE_TAG, rows, (int)error);
	if ( error == cudaSuccess )
		error = SparkQwen38_27bLaunchLinear(stream,&weights->gate,slot->normalized_bf16,slot->z_bf16,rows);
	if ( error != cudaSuccess && rows >= 32u )
		fprintf(stderr, "%s gdn_gate_failed rows=%u err=%d\n", SPARK_QWEN38_27B_MODULE_TAG, rows, (int)error);
	if ( error == cudaSuccess )
		error = SparkQwen38_27bLaunchLinear(stream,&weights->beta,slot->normalized_bf16,slot->beta_pre_bf16,rows);
	if ( error != cudaSuccess && rows >= 32u )
		fprintf(stderr, "%s gdn_beta_failed rows=%u err=%d\n", SPARK_QWEN38_27B_MODULE_TAG, rows, (int)error);
	if ( error == cudaSuccess )
		error = SparkQwen38_27bLaunchLinear(stream,&weights->decay,slot->normalized_bf16,slot->decay_pre_bf16,rows);
	if ( error != cudaSuccess && rows >= 32u )
		fprintf(stderr, "%s gdn_decay_failed rows=%u err=%d\n", SPARK_QWEN38_27B_MODULE_TAG, rows, (int)error);
	/* PATH CHOICE IS A LOSSLESSNESS CONTRACT: the chunk path (prompt
	 * prefill) and the step path (decodes, REPLAY, and VERIFY) compute
	 * the same recurrence with different fp32 rounding, and the recurrence
	 * accumulates the difference round after round. The VERIFY walks the
	 * step path WITH per-row state checkpoints, so the accept loop can
	 * SELECT the accepted-prefix state (the vLLM shape) instead of paying
	 * a replay re-walk. */

	if ( error == cudaSuccess )
	{
		if ( prefill != 0 && slot->replay_frame != 0u )
			error = SparkQwen38_27bModuleRunGdnCoreReplay(state,slot,weights,prefill->lane_index,rows,ordinal);
		else if ( prefill != 0 && slot->verify_frame != 0u && state->snapshot_state_f32 != 0 && state->dflash2_state_select != 0u && state->gdn_snapshot_slot_count >= SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_VERIFY_CHECKPOINT_SLOT_BASE + 8u )
			error = SparkQwen38_27bModuleRunGdnCoreReplaySnap(state,slot,weights,prefill->lane_index,rows,ordinal);
		else
			error = prefill != 0 ? SparkQwen38_27bModuleRunGdnCorePrefill(state,slot,weights,prefill->lane_index,rows,ordinal) : SparkQwen38_27bModuleRunGdnCoreDecode(state,slot,weights,rows,ordinal);
	}
	if ( error != cudaSuccess && rows >= 32u )
		fprintf(stderr, "%s gdn_core_failed rows=%u err=%d\n", SPARK_QWEN38_27B_MODULE_TAG, rows, (int)error);
	if ( error == cudaSuccess )
		error = SparkQwen38_27bLaunchGatedNorm(stream,slot->core_bf16,slot->z_bf16,weights,slot->gated_bf16,rows,SPARK_QWEN38_27B_MODEL_RMS_NORM_EPSILON);
	if ( error != cudaSuccess && rows >= 32u )
		fprintf(stderr, "%s gatednorm_failed rows=%u err=%d\n", SPARK_QWEN38_27B_MODULE_TAG, rows, (int)error);
	if ( error == cudaSuccess )
		error = SparkQwen38_27bLaunchLinear(stream,&weights->output,slot->gated_bf16,slot->delta_bf16,rows);
	if ( error != cudaSuccess && rows >= 32u )
		fprintf(stderr, "%s gdn_output_failed rows=%u err=%d\n", SPARK_QWEN38_27B_MODULE_TAG, rows, (int)error);
	{
		uint64_t spin_start = state->profile_enabled != 0u ? SparkQwen38_27bProfileNow() : 0ull;
		if ( error == cudaSuccess && SparkQwen38_27bModuleTpReduceDelta(state,slot,rows) != SPARK_STATUS_OK )
			error = cudaErrorUnknown;
		if ( state->profile_enabled != 0u )
			state->profile_gdn_spin_nanos += SparkQwen38_27bProfileNow() - spin_start;
	}
	return(SparkStageModuleCudaStatus(SPARK_QWEN38_27B_MODULE_TAG,error,"gdn_layer"));
}

// Device row-control pointers for one attention pass. Main layers bind the
// slot arrays from row zero; an MTP draft step binds them offset to its one
// staged draft row, so the SAME attention path serves both.
typedef struct SparkQwen38_27bAttnRowsView
{
	const uint32_t *slot_mapping;
	const uint64_t *row_positions;
	const uint32_t *row_lane_indices;
	const uint32_t *context_lengths;
} SparkQwen38_27bAttnRowsView;

static SparkStatus SparkQwen38_27bModuleRunAttnLayer(SparkQwen38_27bModuleState *state, SparkQwen38_27bModuleSlot *slot, const SparkQwen38_27bKvBlockTableView *table, const SparkQwen38_27bAttnLayerWeights *weights, uint32_t ordinal, const SparkQwen38_27bAttnRowsView *rows_view, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	error = SparkQwen38_27bLaunchLinear(stream,&weights->query,slot->normalized_bf16,slot->q_fused_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen38_27bLaunchLinear(stream,&weights->key,slot->normalized_bf16,slot->k_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen38_27bLaunchLinear(stream,&weights->value,slot->normalized_bf16,slot->v_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen38_27bLaunchAttnPrepare(stream,slot->q_fused_bf16,slot->k_bf16,slot->v_bf16,weights,state->kv_cache_bf16,rows_view->slot_mapping,rows_view->row_positions,rows,ordinal,state->cache_layer_stride,state->cache_block_stride,SPARK_QWEN38_27B_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkQwen38_27bLaunchAttnDecode(stream,slot->q_fused_bf16,state->kv_cache_bf16,table,rows_view->row_lane_indices,rows_view->context_lengths,slot->head_out_bf16,rows,ordinal,state->cache_layer_stride,state->cache_block_stride);
	if ( error == cudaSuccess )
		error = SparkQwen38_27bLaunchLinear(stream,&weights->output,slot->head_out_bf16,slot->delta_bf16,rows);
	{
		uint64_t spin_start = state->profile_enabled != 0u ? SparkQwen38_27bProfileNow() : 0ull;
		if ( error == cudaSuccess && SparkQwen38_27bModuleTpReduceDelta(state,slot,rows) != SPARK_STATUS_OK )
			error = cudaErrorUnknown;
		if ( state->profile_enabled != 0u )
			state->profile_attn_spin_nanos += SparkQwen38_27bProfileNow() - spin_start;
	}
	return(SparkStageModuleCudaStatus(SPARK_QWEN38_27B_MODULE_TAG,error,"attn_layer"));
}

static SparkStatus SparkQwen38_27bModuleRunFfn(SparkQwen38_27bModuleState *state, SparkQwen38_27bModuleSlot *slot, const void *mlp_norm_bf16, const SparkQwen38_27bFfnLayerWeights *weights, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	const char *ffn_gate_env;
	error = SparkQwen38_27bLaunchFusedResidualRmsNorm(stream,slot->hidden_bf16,slot->delta_bf16,mlp_norm_bf16,slot->normalized_bf16,rows,SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION,SPARK_QWEN38_27B_MODEL_RMS_NORM_EPSILON);
	ffn_gate_env = getenv("SPARK_QWEN38_27B_SMALL_BATCH_GEMM");
	if ( error == cudaSuccess && (ffn_gate_env == 0 || strcmp(ffn_gate_env, "0") != 0) &&
		rows >= 5u && rows <= SPARK_QWEN38_27B_SMALL_BATCH_MAX_ROWS &&
		weights->gate.weight_format == SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 &&
		weights->up.weight_format == SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 &&
		weights->gate.input_dimension == weights->up.input_dimension &&
		weights->gate.output_dimension == weights->up.output_dimension &&
		(weights->gate.input_dimension % SPARK_QWEN38_27B_SMALL_BATCH_K_CHUNK) == 0u &&
		(weights->gate.output_dimension % SPARK_QWEN38_27B_SMALL_BATCH_TILE_N) == 0u )
	{
		/* fused gate+up+swiglu: both projections stream once and the product
		 * lands directly in ffn_up_bf16, bit-identical to the three kernels */
		error = SparkQwen38_27bLaunchFfnGateUp(stream,weights->gate.weight_payload,weights->up.weight_payload,slot->normalized_bf16,slot->ffn_up_bf16,rows,weights->gate.input_dimension,weights->gate.output_dimension);
	}
	else
	{
		if ( error == cudaSuccess )
			error = SparkQwen38_27bLaunchLinear(stream,&weights->gate,slot->normalized_bf16,slot->ffn_gate_bf16,rows);
		if ( error == cudaSuccess )
			error = SparkQwen38_27bLaunchLinear(stream,&weights->up,slot->normalized_bf16,slot->ffn_up_bf16,rows);
		if ( error == cudaSuccess )
			error = SparkQwen38_27bLaunchSwiGlu(stream,slot->ffn_gate_bf16,slot->ffn_up_bf16,rows,state->tp.ffn_intermediate);
	}
	if ( error == cudaSuccess )
		error = SparkQwen38_27bLaunchLinear(stream,&weights->down,slot->ffn_up_bf16,slot->delta_bf16,rows);
	{
		uint64_t spin_start = state->profile_enabled != 0u ? SparkQwen38_27bProfileNow() : 0ull;
		if ( error == cudaSuccess && SparkQwen38_27bModuleTpReduceDelta(state,slot,rows) != SPARK_STATUS_OK )
			error = cudaErrorUnknown;
		if ( state->profile_enabled != 0u )
			state->profile_ffn_spin_nanos += SparkQwen38_27bProfileNow() - spin_start;
	}
	if ( error == cudaSuccess )
		error = SparkQwen38_27bLaunchResidualAdd(stream,slot->hidden_bf16,slot->delta_bf16,rows,SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION);
	return(SparkStageModuleCudaStatus(SPARK_QWEN38_27B_MODULE_TAG,error,"ffn"));
}

static SparkStatus SparkQwen38_27bModuleRunLayer(SparkQwen38_27bModuleState *state, SparkQwen38_27bModuleSlot *slot, const SparkQwen38_27bKvBlockTableView *table, const SparkQwen38_27bPrefillFrameView *prefill, uint32_t layer, uint32_t rows)
{
	SparkQwen38_27bAttnRowsView rows_view;
	SparkStatus status;
	cudaError_t error = SparkQwen38_27bLaunchRmsNorm((cudaStream_t)slot->cuda_stream,slot->hidden_bf16,state->attention_norm_by_layer[layer],slot->normalized_bf16,rows,SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION,SPARK_QWEN38_27B_MODEL_RMS_NORM_EPSILON);
	rows_view.slot_mapping = slot->slot_mapping;
	rows_view.row_positions = slot->row_positions;
	rows_view.row_lane_indices = slot->row_lane_indices;
	rows_view.context_lengths = slot->context_lengths;
	status = SparkStageModuleCudaStatus(SPARK_QWEN38_27B_MODULE_TAG,error,"attention_norm");
	if ( state->profile_enabled != 0u && slot->capturing == 0u )
	{
		(void)cudaEventRecord(state->profile_ev[0],(cudaStream_t)slot->cuda_stream);
		if ( layer == state->first_layer_index )
			state->profile_event_frames++;
	}
	if ( status == SPARK_STATUS_OK )
		status = SPARK_QWEN38_27B_MODEL_LAYER_IS_GDN(layer) != 0u ? SparkQwen38_27bModuleRunGdnLayer(state,slot,prefill,layer,rows) : SparkQwen38_27bModuleRunAttnLayer(state,slot,table,&state->attn_by_layer[layer],state->attn_ordinal_by_layer[layer],&rows_view,rows);
	if ( status == SPARK_STATUS_OK && state->profile_enabled != 0u && slot->capturing == 0u )
	{
		float ms;
		(void)cudaEventRecord(state->profile_ev[1],(cudaStream_t)slot->cuda_stream);
		(void)cudaEventSynchronize(state->profile_ev[1]);
		if ( cudaEventElapsedTime(&ms,state->profile_ev[0],state->profile_ev[1]) == cudaSuccess )
		{
			if ( SPARK_QWEN38_27B_MODEL_LAYER_IS_GDN(layer) != 0u )
				state->profile_gdn_ms += ms;
			else
				state->profile_attn_ms += ms;
		}
		(void)cudaEventRecord(state->profile_ev[0],(cudaStream_t)slot->cuda_stream);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen38_27bModuleRunFfn(state,slot,state->mlp_norm_by_layer[layer],&state->ffn_by_layer[layer],rows);
	if ( status == SPARK_STATUS_OK && state->profile_enabled != 0u && slot->capturing == 0u )
	{
		float ms;
		(void)cudaEventRecord(state->profile_ev[1],(cudaStream_t)slot->cuda_stream);
		(void)cudaEventSynchronize(state->profile_ev[1]);
		if ( cudaEventElapsedTime(&ms,state->profile_ev[0],state->profile_ev[1]) == cudaSuccess )
			state->profile_ffn_ms += ms;
	}
	return(status);
}

static SparkStatus SparkQwen38_27bModuleValidateDecodeView(
    SparkQwen38_27bModuleState *state,
    const SparkModelDriverFrame *frame,
    const SparkQwen38_27bResidentDecodeStageFrameContext *context)
{
    const SparkQwen38_27bDecodeBatchView *batch;
    uint32_t row;

    batch = context->decode_batch;
    if (batch == 0 ||
        batch->abi_version !=
            SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION ||
        batch->descriptor_bytes < (uint32_t)sizeof(*batch) ||
        batch->reserved0 != 0u ||
        batch->row_count == 0u ||
        batch->row_count > state->max_active_sequence_count ||
        batch->row_count != frame->active_slot_count ||
        batch->row_count != frame->new_token_count ||
        batch->row_lane_indices == 0 ||
        batch->row_positions == 0 ||
        batch->row_sequence_ids == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    for (row = 0u; row < batch->row_count; row++)
    {
        uint32_t previous_row;

        if (batch->row_lane_indices[row] >= state->max_active_sequence_count ||
            batch->row_sequence_ids[row] == 0u ||
            batch->row_positions[row] >=
                SPARK_QWEN38_27B_MODEL_MAXIMUM_CONTEXT_TOKENS)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        for (previous_row = 0u; previous_row < row; previous_row++)
        {
            if (batch->row_lane_indices[previous_row] == batch->row_lane_indices[row] ||
                batch->row_sequence_ids[previous_row] == batch->row_sequence_ids[row])
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    return SPARK_STATUS_OK;
}

// The base-zero rule: a fresh lane starts at position zero and gets its
// recurrent state reset; a continuation frame is only meaningful on a lane
// warmed by the preceding frames of the same prompt.
static SparkStatus SparkQwen38_27bModuleValidatePrefillView(
    SparkQwen38_27bModuleState *state,
    const SparkModelDriverFrame *frame,
    const SparkQwen38_27bResidentDecodeStageFrameContext *context)
{
    const SparkQwen38_27bPrefillFrameView *view;

    view = context->prefill_frame;
    if (view == 0 ||
        view->abi_version !=
            SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_PREFILL_FRAME_VIEW_ABI_VERSION ||
        view->descriptor_bytes < (uint32_t)sizeof(*view) ||
        view->lane_index >= state->max_active_sequence_count ||
        view->sequence_id == 0u ||
        view->token_count == 0u ||
        view->token_count > state->max_active_sequence_count ||
        view->token_count != frame->new_token_count ||
        frame->active_slot_count != 1u ||
        view->base_position != frame->sequence_position ||
        view->sequence_id != frame->sequence_id ||
        SparkModelDriverRangeFitsWithinCapacity(
            view->base_position,
            view->token_count,
            SPARK_QWEN38_27B_MODEL_MAXIMUM_CONTEXT_TOKENS) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkQwen38_27bModuleValidateSpeculation(
    SparkQwen38_27bModuleState *state,
    const SparkQwen38_27bResidentDecodeStageFrameContext *context)
{
    const SparkQwen38_27bPrefillFrameView *prefill;
    const SparkQwen38_27bMtpDraftView *draft;
    const SparkQwen38_27bGdnSnapshotView *snapshot;
    const SparkQwen38_27bDecodeBatchView *decode_batch;
    uint32_t drafted;
    uint32_t prefix_restore;
    uint32_t prefix_snapshot;
    uint32_t restore;
    uint32_t verify;
    uint32_t row;
    uint32_t matching_row_found;

    prefill = context->prefill_frame;
    draft = context->mtp_draft;
    snapshot = context->gdn_snapshot;
    decode_batch = context->decode_batch;
    verify = context->flags &
        SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_SPECULATIVE_VERIFY;
    /* prefix-cache transfers carry a snapshot view too: restore-in borrows
     * from the persistent prefix pool, snapshot-out publishes into it */
    prefix_restore = context->flags &
        SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_PREFIX_RESTORE_IN;
    prefix_snapshot = context->flags &
        SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_PREFIX_SNAPSHOT_OUT;
    restore = context->flags &
        SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_RESTORE_FIRST;
    drafted = context->flags &
        SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MTP_DRAFT_AFTER;

    if (verify != 0u || restore != 0u || prefix_restore != 0u ||
        prefix_snapshot != 0u)
    {
        if (prefill == 0 || (verify != 0u && restore != 0u) ||
            (verify != 0u && prefill->base_position == 0u) ||
            snapshot == 0 ||
            snapshot->abi_version !=
                SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_GDN_SNAPSHOT_VIEW_ABI_VERSION ||
            snapshot->descriptor_bytes < (uint32_t)sizeof(*snapshot) ||
            snapshot->reserved0 != 0u ||
            (state->gdn_layer_count != 0u &&
             ((prefix_restore != 0u || prefix_snapshot != 0u)
                  ? (snapshot->snapshot_index >=
                     SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_PREFIX_GDN_SLOT_COUNT)
                  : (state->gdn_snapshot_slot_count == 0u ||
                     snapshot->snapshot_index >=
                         state->gdn_snapshot_slot_count))))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
}
    else if (snapshot != 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (drafted == 0u)
    {
        return draft == 0 ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (state->mtp_armed == 0u || state->owns_final_head == 0u ||
        draft == 0 ||
        draft->abi_version !=
            SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MTP_DRAFT_VIEW_ABI_VERSION ||
        draft->descriptor_bytes < (uint32_t)sizeof(*draft) ||
        draft->draft_token_count == 0u ||
        draft->draft_token_count >
            SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS ||
        draft->lane_index >= state->max_active_sequence_count ||
        (state->owns_embedding == 0u && draft->row_token_ids == 0))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (prefill != 0)
    {
        if (draft->lane_index != prefill->lane_index ||
            draft->sequence_id != prefill->sequence_id ||
            draft->base_position !=
                prefill->base_position + prefill->token_count)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        return SPARK_STATUS_OK;
    }

    matching_row_found = 0u;
    for (row = 0u; row < decode_batch->row_count; row++)
    {
        if (decode_batch->row_lane_indices[row] == draft->lane_index)
        {
            if (draft->sequence_id != decode_batch->row_sequence_ids[row] ||
                draft->base_position != decode_batch->row_positions[row] + 1u)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            matching_row_found = 1u;
            break;
        }
    }
    return matching_row_found != 0u
        ? SPARK_STATUS_OK
        : SPARK_STATUS_INVALID_ARGUMENT;
}

static SparkStatus SparkQwen38_27bModuleValidateFrame(
    SparkQwen38_27bModuleState *state,
    const SparkModelDriverFrame *frame,
    const SparkQwen38_27bResidentDecodeStageFrameContext **context_out)
{
    const uint32_t known_frame_flags = SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL;
    const uint32_t known_context_flags =
        SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE |
        SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW |
        SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT |
        SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT |
        SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW |
        SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MTP_DRAFT_AFTER |
        SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_SPECULATIVE_VERIFY |
        SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_RESTORE_FIRST |
        SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_RESTORE_VERIFY_ROW |
        SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_PREFIX_SNAPSHOT_OUT |
        SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_PREFIX_RESTORE_IN |
        SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DSPARK_DRAFT_AFTER;
    const SparkQwen38_27bResidentDecodeStageFrameContext *context;
    const SparkQwen38_27bKvBlockTableView *block_table;
    uint32_t expected_buffer_count;
    uint32_t is_prefill;
    uint32_t mode;
    uint32_t needs_hidden_input;
    uint32_t needs_hidden_output;
    uint32_t output_buffer_index;
    uint32_t output_token_count;
    uint32_t row_count;
    uint64_t token_bytes;
    SparkStatus status;

    if (state == 0 || frame == 0 || context_out == 0 ||
        frame->program_id == 0u || frame->tokens_per_sequence != 1u ||
        (frame->flags & ~known_frame_flags) != 0u ||
        (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID) != 0u ||
        frame->new_token_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    expected_buffer_count = state->owns_embedding + state->owns_final_head;
    if (frame->buffer_count != expected_buffer_count ||
        (expected_buffer_count != 0u && frame->buffers == 0) ||
        frame->user_context == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    context = (const SparkQwen38_27bResidentDecodeStageFrameContext *)frame->user_context;
    if (context->abi_version !=
            SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION ||
        context->descriptor_bytes < (uint32_t)sizeof(*context) ||
        context->reserved0 != 0u ||
        (context->flags & ~known_context_flags) != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    mode = context->flags &
        (SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW |
         SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW);
    is_prefill = (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u
        ? 1u
        : 0u;
    if ((is_prefill != 0u &&
         mode != SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW) ||
        (is_prefill == 0u &&
         mode != SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    needs_hidden_input = state->stage_index > 0u ? 1u : 0u;
    needs_hidden_output = state->stage_index + 1u < state->stage_count ? 1u : 0u;
    if (((context->flags &
          SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT) != 0u) !=
            (needs_hidden_input != 0u) ||
        ((context->flags &
          SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT) != 0u) !=
            (needs_hidden_output != 0u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((needs_hidden_input != 0u &&
         (context->hidden_input_transport_session == 0 ||
          context->hidden_input_post_receive_function == 0)) ||
        (needs_hidden_input == 0u &&
         (context->hidden_input_transport_session != 0 ||
          context->hidden_input_post_receive_function != 0)) ||
        (needs_hidden_output != 0u &&
         (context->hidden_output_transport_session == 0 ||
          context->hidden_output_send_function == 0)) ||
        (needs_hidden_output == 0u &&
         (context->hidden_output_transport_session != 0 ||
          context->hidden_output_send_function != 0)))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    block_table = context->kv_block_table;
    if (state->attn_layer_count != 0u)
    {
        if ((context->flags &
             SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE) == 0u ||
            block_table == 0 ||
            block_table->abi_version !=
                SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_KV_BLOCK_TABLE_ABI_VERSION ||
            block_table->descriptor_bytes < (uint32_t)sizeof(*block_table) ||
            block_table->block_token_count !=
                SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS ||
            block_table->lane_count != state->max_active_sequence_count ||
            block_table->lane_capacity < block_table->lane_count ||
            block_table->lane_stride == 0u ||
            block_table->physical_block_indices == 0 ||
            block_table->lane_physical_block_counts == 0 ||
            block_table->host_physical_block_indices == 0 ||
            block_table->host_lane_physical_block_counts == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    else if ((context->flags &
              SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE) != 0u ||
             block_table != 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = is_prefill != 0u
        ? SparkQwen38_27bModuleValidatePrefillView(state, frame, context)
        : SparkQwen38_27bModuleValidateDecodeView(state, frame, context);
    if (status == SPARK_STATUS_OK)
    {
        status = SparkQwen38_27bModuleValidateSpeculation(state, context);
    }
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    row_count = is_prefill != 0u
        ? context->prefill_frame->token_count
        : context->decode_batch->row_count;
    token_bytes = (uint64_t)row_count * sizeof(uint32_t);
    if (state->owns_embedding != 0u)
    {
        status = SparkModelDriverValidateBuffer(
            frame,
            0u,
            0u,
            SPARK_MODEL_DRIVER_BUFFER_FLAG_READ,
            token_bytes);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    if (state->owns_final_head != 0u)
    {
        output_token_count = is_prefill != 0u &&
            (context->flags &
             SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_SPECULATIVE_VERIFY) == 0u
            ? 1u
            : row_count;
        if ((context->flags &
             SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MTP_DRAFT_AFTER) != 0u)
        {
            output_token_count += context->mtp_draft->draft_token_count;
        }
        output_buffer_index = state->owns_embedding != 0u ? 1u : 0u;
        status = SparkModelDriverValidateBuffer(
            frame,
            output_buffer_index,
            1u,
            SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE,
            (uint64_t)output_token_count * sizeof(uint32_t));
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }

    *context_out = context;
    return SPARK_STATUS_OK;
}

/*
 * Host staging for one decode microbatch: distinct-lane check, cold flags
 * from the lane warm map, and for attention stages the physical slot and
 * context length per row proven against the host block-table mirrors. Any
 * uncovered position is a refused frame, never a stray cache write.
 */
/*
 * One staged row: lane and position mirrored into the host arrays, and for
 * cache-bearing stages the physical slot and context length proven against
 * the host block-table mirrors. Decode rows, prefill rows and MTP draft
 * rows all pass through here; an uncovered position is a refused frame,
 * never a stray cache write.
 */
static SparkStatus SparkQwen38_27bModuleStagePosition(SparkQwen38_27bModuleState *state, SparkQwen38_27bModuleSlot *slot, const SparkQwen38_27bKvBlockTableView *table, uint32_t lane, uint64_t position, uint32_t index)
{
	uint32_t block_ordinal,block;
	slot->host_row_lane_indices[index] = lane;
	slot->host_row_positions[index] = position;
	if ( state->attn_layer_count == 0u )
		return(SPARK_STATUS_OK);
	if ( position + 1u > (uint64_t)table->lane_stride * SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	block_ordinal = (uint32_t)(position / SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
	if ( lane >= table->lane_count || block_ordinal >= table->host_lane_physical_block_counts[lane] )
	{
		fprintf(stderr,"qwen38_27b_debug stage_pos lane=%u ord=%u counts=%u stride=%u pos=%llu\n",lane,block_ordinal,lane < table->lane_count ? table->host_lane_physical_block_counts[lane] : 0u,table->lane_stride,(unsigned long long)position);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	block = table->host_physical_block_indices[((uint64_t)lane * table->lane_stride) + block_ordinal];
	if ( block == SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_NO_BLOCK || block >= state->kv_block_count )
	{
		fprintf(stderr,"qwen38_27b_debug stage_noblock lane=%u ord=%u block=%u kv=%u\n",lane,block_ordinal,block,state->kv_block_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	slot->host_slot_mapping[index] = (block * SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS) + (uint32_t)(position % SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
	slot->host_context_lengths[index] = (uint32_t)(position + 1u);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen38_27bModuleStageRows(SparkQwen38_27bModuleState *state, SparkQwen38_27bModuleSlot *slot, const SparkQwen38_27bResidentDecodeStageFrameContext *context, uint8_t *lane_used)
{
	const SparkQwen38_27bDecodeBatchView *batch = context->decode_batch;
	uint32_t row,lane;
	SparkStatus status;
	for (row = 0; row < batch->row_count; row++)
	{
		lane = batch->row_lane_indices[row];
		if ( lane >= state->max_active_sequence_count || lane_used[lane] != 0u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		lane_used[lane] = 1u;
		slot->host_row_cold[row] = state->lane_warm[lane] != 0u ? 0u : 1u;
		status = SparkQwen38_27bModuleStagePosition(state,slot,context->kv_block_table,lane,batch->row_positions[row],row);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	return(SPARK_STATUS_OK);
}

/*
 * Host staging for one prefill frame: every position of
 * [base_position, base_position + token_count) becomes a row of the one
 * lane, with the physical slot and context length proven against the host
 * block-table mirrors exactly as decode does per row. Any uncovered
 * position is a refused frame, never a stray cache write.
 */
static SparkStatus SparkQwen38_27bModulePrefillStage(SparkQwen38_27bModuleState *state, SparkQwen38_27bModuleSlot *slot, const SparkQwen38_27bResidentDecodeStageFrameContext *context)
{
	const SparkQwen38_27bPrefillFrameView *view = context->prefill_frame;
	uint32_t index;
	SparkStatus status;
	for (index = 0; index < view->token_count; index++)
	{
		status = SparkQwen38_27bModuleStagePosition(state,slot,context->kv_block_table,view->lane_index,view->base_position + index,index);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	return(SPARK_STATUS_OK);
}

/*
 * Draft staging appends draft_token_count rows after the frame's rows. The
 * seed row is the last prefill row, or the decode batch row carrying the
 * drafted lane; the first draft position must be exactly one past it, so a
 * mispointed draft view is refused before anything launches.
 */
static SparkStatus SparkQwen38_27bModuleStageMtpDraft(SparkQwen38_27bModuleState *state, SparkQwen38_27bModuleSlot *slot, const SparkQwen38_27bResidentDecodeStageFrameContext *context, const SparkQwen38_27bPrefillFrameView *prefill, uint32_t rows)
{
	const SparkQwen38_27bMtpDraftView *view = context->mtp_draft;
	uint32_t seed_row = rows - 1u,row,draft;
	SparkStatus status;
	if ( prefill == 0 )
	{
		for (row = 0; row < rows; row++)
			if ( context->decode_batch->row_lane_indices[row] == view->lane_index )
				break;
		if ( row == rows )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		seed_row = row;
	}
	if ( slot->host_row_positions[seed_row] + 1u != view->base_position )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	slot->mtp_seed_row = seed_row;
	for (draft = 0; draft + 1u < view->draft_token_count; draft++)
	{
		status = SparkQwen38_27bModuleStagePosition(state,slot,context->kv_block_table,view->lane_index,view->base_position + draft,rows + draft);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen38_27bModuleUploadRows(SparkQwen38_27bModuleState *state, SparkQwen38_27bModuleSlot *slot, const SparkQwen38_27bResidentDecodeStageFrameContext *context, const SparkModelDriverFrame *frame, const SparkQwen38_27bPrefillFrameView *prefill, uint32_t rows)
{
	uint32_t token_guard;
	uint32_t drafted = (context->flags & SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MTP_DRAFT_AFTER) != 0u ? 1u : 0u;
	uint32_t staged = rows + (drafted != 0u ? context->mtp_draft->draft_token_count - 1u : 0u);
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	error = cudaMemcpyAsync(slot->row_lane_indices,slot->host_row_lane_indices,staged * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->row_positions,slot->host_row_positions,staged * sizeof(uint64_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess && prefill == 0 )
		error = cudaMemcpyAsync(slot->row_cold,slot->host_row_cold,rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess && state->attn_layer_count != 0u )
		error = cudaMemcpyAsync(slot->slot_mapping,slot->host_slot_mapping,staged * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess && state->attn_layer_count != 0u )
		error = cudaMemcpyAsync(slot->context_lengths,slot->host_context_lengths,staged * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess && state->owns_embedding != 0u )
	{
		for (token_guard = 0; token_guard < rows; token_guard++)
			if ( ((const uint32_t *)frame->buffers[0].address)[token_guard] >= SPARK_QWEN38_27B_MODEL_OUTPUT_VOCAB_COUNT )
			{
				fprintf(stderr,"%s token_id_out_of_range row=%u\n",SPARK_QWEN38_27B_MODULE_TAG,token_guard);
				return(SPARK_STATUS_INVALID_ARGUMENT);
			}
		error = cudaMemcpyAsync(slot->input_token_ids,frame->buffers[0].address,rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	}
	if ( error == cudaSuccess && drafted != 0u && state->owns_embedding == 0u )
		error = cudaMemcpyAsync(slot->input_token_ids,context->mtp_draft->row_token_ids,(prefill != 0 ? rows : 1u) * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	return(SparkStageModuleCudaStatus(SPARK_QWEN38_27B_MODULE_TAG,error,"stage_upload"));
}

// A base-zero prefill claims the lane fresh: the chunk kernels read the
// resident state unconditionally, so a reused lane's stale delta state and
// conv tails are zeroed on the slot stream before the walk.
static SparkStatus SparkQwen38_27bModuleResetLaneState(SparkQwen38_27bModuleState *state, SparkQwen38_27bModuleSlot *slot, uint32_t lane)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint64_t state_bytes = state->gdn_pool.state_lane_stride_elements * sizeof(float);
	uint64_t tail_bytes = state->gdn_pool.conv_tail_lane_stride_elements * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES;
	cudaError_t error;
	if ( state->gdn_layer_count == 0u )
		return(SPARK_STATUS_OK);
	error = cudaMemsetAsync(state->gdn_pool.state_f32 + ((uint64_t)lane * state->gdn_pool.state_lane_stride_elements),0,state_bytes,stream);
	if ( error == cudaSuccess )
		error = cudaMemsetAsync((uint8_t *)state->gdn_pool.conv_tail_bf16 + ((uint64_t)lane * tail_bytes),0,tail_bytes,stream);
	return(SparkStageModuleCudaStatus(SPARK_QWEN38_27B_MODULE_TAG,error,"lane_reset"));
}

// Only the GDN recurrence is destructive under speculation: attention and
// MTP K/V at rejected positions are simply overwritten when the position
// re-executes. A verify frame copies the lane's delta state and conv tails
// OUT to the runtime-assigned snapshot slot before the walk; the replay
// frame copies them back IN before re-advancing over the accepted tokens.
static SparkStatus SparkQwen38_27bModuleGdnSnapshot(SparkQwen38_27bModuleState *state, SparkQwen38_27bModuleSlot *slot, uint32_t lane, uint32_t snapshot_index, uint32_t restore)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint64_t state_bytes = state->gdn_pool.state_lane_stride_elements * sizeof(float);
	uint64_t tail_bytes = state->gdn_pool.conv_tail_lane_stride_elements * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES;
	float *lane_state = state->gdn_pool.state_f32 + ((uint64_t)lane * state->gdn_pool.state_lane_stride_elements);
	float *shot_state = state->snapshot_state_f32 + ((uint64_t)snapshot_index * state->gdn_pool.state_lane_stride_elements);
	uint8_t *lane_tail = (uint8_t *)state->gdn_pool.conv_tail_bf16 + ((uint64_t)lane * tail_bytes);
	uint8_t *shot_tail = (uint8_t *)state->snapshot_tail_bf16 + ((uint64_t)snapshot_index * tail_bytes);
	cudaError_t error;
	if ( state->gdn_layer_count == 0u )
		return(SPARK_STATUS_OK);
	error = cudaMemcpyAsync(restore != 0u ? lane_state : shot_state,restore != 0u ? shot_state : lane_state,state_bytes,cudaMemcpyDeviceToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(restore != 0u ? lane_tail : shot_tail,restore != 0u ? shot_tail : lane_tail,tail_bytes,cudaMemcpyDeviceToDevice,stream);
	return(SparkStageModuleCudaStatus(SPARK_QWEN38_27B_MODULE_TAG,error,"gdn_snapshot"));
}

/* Prefix-cache GDN transfer: lane state <-> a PERSISTENT pool slot (the
 * adapter's prefix entries own these slots; the verify snapshot slots are
 * transient per-round). restore=0 snapshots OUT (after the publish-
 * boundary walk), restore=1 restores IN (before a prefix-hit lane's walk). */
static SparkStatus SparkQwen38_27bModuleGdnPrefixTransfer(SparkQwen38_27bModuleState *state, SparkQwen38_27bModuleSlot *slot, uint32_t lane, uint32_t prefix_slot, uint32_t restore)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint64_t state_bytes = state->gdn_pool.state_lane_stride_elements * sizeof(float);
	uint64_t tail_bytes = state->gdn_pool.conv_tail_lane_stride_elements * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES;
	float *lane_state = state->gdn_pool.state_f32 + ((uint64_t)lane * state->gdn_pool.state_lane_stride_elements);
	float *slot_state = state->prefix_state_f32 + ((uint64_t)prefix_slot * state->gdn_pool.state_lane_stride_elements);
	uint8_t *lane_tail = (uint8_t *)state->gdn_pool.conv_tail_bf16 + ((uint64_t)lane * tail_bytes);
	uint8_t *slot_tail = (uint8_t *)state->prefix_tail_bf16 + ((uint64_t)prefix_slot * tail_bytes);
	cudaError_t error;
	if ( state->gdn_layer_count == 0u || state->prefix_state_f32 == 0 || state->prefix_tail_bf16 == 0 || prefix_slot >= SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_PREFIX_GDN_SLOT_COUNT )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	error = cudaMemcpyAsync(restore != 0u ? lane_state : slot_state,restore != 0u ? slot_state : lane_state,state_bytes,cudaMemcpyDeviceToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(restore != 0u ? lane_tail : slot_tail,restore != 0u ? slot_tail : lane_tail,tail_bytes,cudaMemcpyDeviceToDevice,stream);
	return(SparkStageModuleCudaStatus(SPARK_QWEN38_27B_MODULE_TAG,error,"gdn_prefix_transfer"));
}

static SparkStatus SparkQwen38_27bModuleBeginHidden(SparkQwen38_27bModuleState *state, SparkQwen38_27bModuleSlot *slot, SparkQwen38_27bResidentDecodeStageFrameContext *context, uint32_t rows)
{
	SparkStatus status;
	cudaError_t error;
	if ( state->owns_embedding != 0u )
	{
		error = SparkQwen38_27bLaunchEmbeddingGather((cudaStream_t)slot->cuda_stream,slot->input_token_ids,state->token_embedding_bf16,slot->hidden_bf16,rows);
		return(SparkStageModuleCudaStatus(SPARK_QWEN38_27B_MODULE_TAG,error,"embedding"));
	}
	if ( context->hidden_input_post_receive_function == 0 || context->hidden_input_transport_session == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = context->hidden_input_post_receive_function(context->hidden_input_transport_session,&context->hidden_input_packet);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( context->hidden_input_packet.active_sequence_count != rows || context->hidden_input_packet.hidden_dimension != SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION || context->hidden_input_packet.hidden_bf16 == 0 )
		return(SPARK_STATUS_VALIDATION_FAILED);
	error = cudaMemcpyAsync(slot->hidden_bf16,context->hidden_input_packet.hidden_bf16,(uint64_t)rows * SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES,cudaMemcpyDeviceToDevice,(cudaStream_t)slot->cuda_stream);
	return(SparkStageModuleCudaStatus(SPARK_QWEN38_27B_MODULE_TAG,error,"hidden_receive"));
}

/*
 * Head emission. Decode samples every row. Prefill samples ONLY the final
 * position: the norm and the fused matvec+argmax run on one row addressed
 * at the frame's last hidden row, and exactly one token id lands in the
 * output buffer - a 512-row argmax over a 248320 vocabulary for positions
 * nothing reads would be pure waste.
 */
static cudaError_t SparkQwen38_27bModuleEmitHead(SparkQwen38_27bModuleState *state, SparkQwen38_27bModuleSlot *slot, SparkModelDriverFrame *frame, const SparkQwen38_27bPrefillFrameView *prefill, uint32_t emit_all, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint32_t out_index = state->owns_embedding != 0u ? 1u : 0u,head_rows = prefill != 0 && emit_all == 0u ? 1u : rows;
	const void *head_hidden = head_rows == 1u && rows != 1u ? (const void *)((const uint8_t *)slot->hidden_bf16 + ((uint64_t)(rows - 1u) * SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES)) : slot->hidden_bf16;
	cudaError_t error;
	error = SparkQwen38_27bLaunchRmsNorm(stream,head_hidden,state->final_norm_weight_bf16,slot->normalized_bf16,head_rows,SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION,SPARK_QWEN38_27B_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkQwen38_27bLaunchHeadScreenedArgmaxScore(stream,slot->normalized_bf16,state->lm_head_weight_bf16,state->head_shadow_payload,state->head_shadow_scale,state->head_error_norm_f32,slot->head_logits_bf16,slot->head_candidate_ids_u32,slot->head_candidate_counts_u32,slot->output_token_ids,slot->head_scores_f32,state->tp_rank * state->tp.head_rows,head_rows,state->tp.head_rows);
	if ( error == cudaSuccess )
		error = SparkQwen38_27bLaunchHeadMaxLocPack(stream,slot->head_scores_f32,slot->output_token_ids,slot->head_maxloc_u64,head_rows);
	{
		uint64_t spin_start = state->profile_enabled != 0u ? SparkQwen38_27bProfileNow() : 0ull;
		if ( error == cudaSuccess && SparkQwen38_27bTpReduceU64Max(&state->tp,slot->head_maxloc_u64,head_rows,stream) != SPARK_STATUS_OK )
			error = cudaErrorUnknown;
		if ( state->profile_enabled != 0u )
		{
			/* profile-only sync: ILLEGAL during graph capture (it invalidated
			 * the very first capture round; skip while capturing) */
			if ( state->tp_degree <= 1u && slot->capturing == 0u )
				(void)cudaStreamSynchronize(stream);
			state->profile_head_spin_nanos += SparkQwen38_27bProfileNow() - spin_start;
		}
	}
	if ( error == cudaSuccess )
		error = SparkQwen38_27bLaunchHeadMaxLocUnpack(stream,slot->head_maxloc_u64,slot->output_token_ids,head_rows);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(frame->buffers[out_index].address,slot->output_token_ids,head_rows * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream);
	return(error);
}

/*
 * MTP input packing for rows_p rows: embed the token ids, the two pre-norms
 * into dense scratch, then two strided device copies interleave them as the
 * [enorm | hnorm] halves of each fc input row, and fc lands the decoder
 * input in hidden rows [0, rows_p). The pre-norm sources are consumed
 * before fc overwrites hidden - the slot stream serializes the hazard.
 */
static SparkStatus SparkQwen38_27bModuleRunMtpPackInput(SparkQwen38_27bModuleState *state, SparkQwen38_27bModuleSlot *slot, const uint32_t *token_src, const void *hnorm_src, uint32_t rows_p)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint64_t half_bytes = SPARK_QWEN38_27B_MODEL_HIDDEN_BF16_BYTES,pack_pitch = 2u * half_bytes;
	cudaError_t error;
	error = SparkQwen38_27bLaunchEmbeddingGather(stream,token_src,state->token_embedding_bf16,slot->gated_bf16,rows_p);
	if ( error == cudaSuccess )
		error = SparkQwen38_27bLaunchRmsNorm(stream,slot->gated_bf16,state->mtp.embed_norm_weight_bf16,slot->normalized_bf16,rows_p,SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION,SPARK_QWEN38_27B_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkQwen38_27bLaunchRmsNorm(stream,hnorm_src,state->mtp.hidden_norm_weight_bf16,slot->delta_bf16,rows_p,SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION,SPARK_QWEN38_27B_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = cudaMemcpy2DAsync(slot->qkv_bf16,pack_pitch,slot->normalized_bf16,half_bytes,half_bytes,rows_p,cudaMemcpyDeviceToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpy2DAsync((uint8_t *)slot->qkv_bf16 + half_bytes,pack_pitch,slot->delta_bf16,half_bytes,half_bytes,rows_p,cudaMemcpyDeviceToDevice,stream);
	if ( error == cudaSuccess )
		error = SparkQwen38_27bLaunchLinear(stream,&state->mtp.fc,slot->qkv_bf16,slot->hidden_bf16,rows_p);
	return(SparkStageModuleCudaStatus(SPARK_QWEN38_27B_MODULE_TAG,error,"mtp_pack"));
}

static SparkStatus SparkQwen38_27bModuleRunMtpDecoderPass(SparkQwen38_27bModuleState *state, SparkQwen38_27bModuleSlot *slot, const SparkQwen38_27bKvBlockTableView *table, const SparkQwen38_27bAttnRowsView *rows_view, uint32_t rows_p)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	SparkStatus status;
	cudaError_t error = SparkQwen38_27bLaunchRmsNorm(stream,slot->hidden_bf16,state->mtp.attention_norm_weight_bf16,slot->normalized_bf16,rows_p,SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION,SPARK_QWEN38_27B_MODEL_RMS_NORM_EPSILON);
	status = SparkStageModuleCudaStatus(SPARK_QWEN38_27B_MODULE_TAG,error,"mtp_attn_norm");
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen38_27bModuleRunAttnLayer(state,slot,table,&state->mtp.attention,state->mtp_cache_ordinal,rows_view,rows_p);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_QWEN38_27B_MODULE_TAG,SparkQwen38_27bLaunchResidualAdd(stream,slot->hidden_bf16,slot->delta_bf16,rows_p,SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION),"mtp_residual");
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen38_27bModuleRunFfn(state,slot,state->mtp.mlp_norm_weight_bf16,&state->mtp.ffn,rows_p);
	return(status);
}

static SparkStatus SparkQwen38_27bModuleRunMtpArgmaxRow(SparkQwen38_27bModuleState *state, SparkQwen38_27bModuleSlot *slot, uint32_t row, uint32_t draft_index)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	const void *row_hidden = (const uint8_t *)slot->hidden_bf16 + ((uint64_t)row * SPARK_QWEN38_27B_MODEL_HIDDEN_BF16_BYTES);
	cudaError_t error;
	error = SparkQwen38_27bLaunchRmsNorm(stream,row_hidden,state->mtp.final_norm_weight_bf16,slot->normalized_bf16,1u,SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION,SPARK_QWEN38_27B_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess && state->tp_degree > 1u )
	{
		/* TP4: argmax over the rank's head shard, then the u64 maxloc
		 * collective picks the global winner across ranks. */
		error = SparkQwen38_27bLaunchHeadScreenedArgmaxScore(stream,slot->normalized_bf16,state->lm_head_weight_bf16,state->head_shadow_payload,state->head_shadow_scale,state->head_error_norm_f32,slot->head_logits_bf16,slot->head_candidate_ids_u32,slot->head_candidate_counts_u32,slot->mtp_draft_ids + draft_index,slot->head_scores_f32,state->tp_rank * state->tp.head_rows,1u,state->tp.head_rows);
		if ( error == cudaSuccess )
			error = SparkQwen38_27bLaunchHeadMaxLocPack(stream,slot->head_scores_f32,slot->mtp_draft_ids + draft_index,slot->head_maxloc_u64,1u);
		if ( error == cudaSuccess && SparkQwen38_27bTpReduceU64Max(&state->tp,slot->head_maxloc_u64,1u,stream) != SPARK_STATUS_OK )
			error = cudaErrorUnknown;
		if ( error == cudaSuccess )
			error = SparkQwen38_27bLaunchHeadMaxLocUnpack(stream,slot->head_maxloc_u64,slot->mtp_draft_ids + draft_index,1u);
	}
	else if ( error == cudaSuccess )
		error = SparkQwen38_27bLaunchHeadScreenedArgmax(stream,slot->normalized_bf16,state->lm_head_weight_bf16,state->head_shadow_payload,state->head_shadow_scale,state->head_error_norm_f32,slot->head_logits_bf16,slot->head_candidate_ids_u32,slot->head_candidate_counts_u32,slot->mtp_draft_ids + draft_index,1u,SPARK_QWEN38_27B_MODEL_OUTPUT_VOCAB_COUNT);
	return(SparkStageModuleCudaStatus(SPARK_QWEN38_27B_MODULE_TAG,error,"mtp_argmax"));
}

/*
 * The seed pass batches the MTP decoder over the frame's own rows, feeding
 * committed token ids against committed backbone hiddens, which rewrites
 * the lane's MTP K/V at those positions and yields draft one at the final
 * row. Chain steps then extend one position each, embedding the previous
 * draft against the previous MTP hidden through the identical pass. The
 * frame's backbone hiddens are dead by now: the head consumed them.
 */
static SparkStatus SparkQwen38_27bModuleRunMtpDraftChain(SparkQwen38_27bModuleState *state, SparkQwen38_27bModuleSlot *slot, SparkQwen38_27bResidentDecodeStageFrameContext *context, SparkModelDriverFrame *frame, const SparkQwen38_27bPrefillFrameView *prefill, uint32_t rows)
{
	const SparkQwen38_27bMtpDraftView *view = context->mtp_draft;
	uint32_t verify = (context->flags & SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_SPECULATIVE_VERIFY) != 0u ? 1u : 0u;
	uint32_t head_rows = prefill != 0 && verify == 0u ? 1u : rows;
	uint32_t rows_p = prefill != 0 ? rows : 1u,row_base = prefill != 0 ? 0u : slot->mtp_seed_row,step;
	uint32_t out_index = state->owns_embedding != 0u ? 1u : 0u;
	const uint32_t *seed_ids = slot->input_token_ids + (state->owns_embedding != 0u ? row_base : 0u);
	const void *seed_hidden = (const uint8_t *)slot->hidden_bf16 + ((uint64_t)row_base * SPARK_QWEN38_27B_MODEL_HIDDEN_BF16_BYTES);
	SparkQwen38_27bAttnRowsView rows_view;
	SparkStatus status;
	rows_view.slot_mapping = slot->slot_mapping + row_base;
	rows_view.row_positions = slot->row_positions + row_base;
	rows_view.row_lane_indices = slot->row_lane_indices + row_base;
	rows_view.context_lengths = slot->context_lengths + row_base;
	status = SparkQwen38_27bModuleRunMtpPackInput(state,slot,seed_ids,seed_hidden,rows_p);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen38_27bModuleRunMtpDecoderPass(state,slot,context->kv_block_table,&rows_view,rows_p);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen38_27bModuleRunMtpArgmaxRow(state,slot,rows_p - 1u,0u);
	for (step = 1; status == SPARK_STATUS_OK && step < view->draft_token_count; step++)
	{
		rows_view.slot_mapping = slot->slot_mapping + rows + (step - 1u);
		rows_view.row_positions = slot->row_positions + rows + (step - 1u);
		rows_view.row_lane_indices = slot->row_lane_indices + rows + (step - 1u);
		rows_view.context_lengths = slot->context_lengths + rows + (step - 1u);
		status = SparkQwen38_27bModuleRunMtpPackInput(state,slot,slot->mtp_draft_ids + (step - 1u),step == 1u ? (const void *)((const uint8_t *)slot->hidden_bf16 + ((uint64_t)(rows_p - 1u) * SPARK_QWEN38_27B_MODEL_HIDDEN_BF16_BYTES)) : slot->hidden_bf16,1u);
		if ( status == SPARK_STATUS_OK )
			status = SparkQwen38_27bModuleRunMtpDecoderPass(state,slot,context->kv_block_table,&rows_view,1u);
		if ( status == SPARK_STATUS_OK )
			status = SparkQwen38_27bModuleRunMtpArgmaxRow(state,slot,0u,step);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_QWEN38_27B_MODULE_TAG,cudaMemcpyAsync((uint8_t *)frame->buffers[out_index].address + ((uint64_t)head_rows * sizeof(uint32_t)),slot->mtp_draft_ids,view->draft_token_count * sizeof(uint32_t),cudaMemcpyDeviceToHost,(cudaStream_t)slot->cuda_stream),"mtp_emit");
	return(status);
}

static SparkStatus SparkQwen38_27bModuleFinish(SparkQwen38_27bModuleState *state, SparkQwen38_27bModuleSlot *slot, SparkQwen38_27bResidentDecodeStageFrameContext *context, SparkModelDriverFrame *frame, const SparkQwen38_27bPrefillFrameView *prefill, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint32_t emit_all = (context->flags & SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_SPECULATIVE_VERIFY) != 0u ? 1u : 0u;
	SparkStatus status = SPARK_STATUS_OK;
	if ( state->owns_final_head != 0u )
		status = SparkStageModuleCudaStatus(SPARK_QWEN38_27B_MODULE_TAG,SparkQwen38_27bModuleEmitHead(state,slot,frame,prefill,emit_all,rows),"head");
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u && (context->flags & SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MTP_DRAFT_AFTER) != 0u )
		status = SparkQwen38_27bModuleRunMtpDraftChain(state,slot,context,frame,prefill,rows);
	if ( state->owns_final_head == 0u )
	{
		if ( context->hidden_output_send_function == 0 || context->hidden_output_transport_session == 0 )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		context->hidden_output_packet.active_sequence_count = rows;
		context->hidden_output_packet.hidden_dimension = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
		context->hidden_output_packet.bytes_per_sequence = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION * SPARK_QWEN38_27B_MODEL_BF16_ELEMENT_BYTES;
		context->hidden_output_packet.hidden_bf16 = slot->hidden_bf16;
		context->hidden_output_packet.cuda_stream = stream;
		context->hidden_output_packet.sideband_payload = 0;
		context->hidden_output_packet.sideband_kind = 0u;
		context->hidden_output_packet.sideband_bytes_per_sequence = 0u;
	}
	/* the sync is the module's completion contract; during graph capture it
	 * is ILLEGAL (it invalidated the very first capture round) - the frame
	 * wrapper syncs after the replay instead, preserving the contract */
	if ( status == SPARK_STATUS_OK && slot->capturing == 0u )
		status = SparkStageModuleCudaStatus(SPARK_QWEN38_27B_MODULE_TAG,cudaStreamSynchronize(stream),"sync");
	if ( status == SPARK_STATUS_OK && state->owns_final_head == 0u )
		status = context->hidden_output_send_function(context->hidden_output_transport_session,&context->hidden_output_packet);
	return(status);
}

static uint64_t SparkQwen38_27bModuleFingerprint(const void *bytes, uint64_t count, uint64_t basis)
{
	const uint8_t *data = (const uint8_t *)bytes;
	uint64_t hash = basis,index;
	for (index = 0; index < count; index++)
		hash = (hash ^ data[index]) * 1099511628211ull;
	return(hash);
}

static SparkStatus SparkQwen38_27bModuleOpenKvTier(SparkQwen38_27bModuleState *state)
{
	SparkQwen38_27bStagePackHeader geometry;
	const char *provider = 0,*service = 0,*socket_path = 0;
	uint64_t pool_bytes = 0u,model_fp,layout_fp,layout_bits[3];
	uint32_t workers = 0u;
	SparkStatus status = SparkStageModuleEnvironmentText(SPARK_QWEN38_27B_MODULE_TAG,"SPARK_QWEN38_27B_STAGE_KV_STORE",&provider);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( strcmp(provider,"none") == 0 )
		return(SparkStageKvClientOpen(&state->kv_client,SPARK_QWEN38_27B_MODULE_TAG,provider,0u,0u,0u,0u,0u,0,0,0u,0u));
	status = SparkStageModuleEnvironmentText(SPARK_QWEN38_27B_MODULE_TAG,"SPARK_QWEN38_27B_STAGE_KV_SERVICE",&service);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentText(SPARK_QWEN38_27B_MODULE_TAG,"SPARK_QWEN38_27B_STAGE_KV_SOCKET",&socket_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned64(SPARK_QWEN38_27B_MODULE_TAG,"SPARK_QWEN38_27B_STAGE_KV_POOL_BYTES",1u,1ull << 40u,&pool_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN38_27B_MODULE_TAG,"SPARK_QWEN38_27B_STAGE_KV_WORKERS",1u,64u,&workers);
	if ( status != SPARK_STATUS_OK )
		return(status);
	SparkQwen38_27bStagePackExpectedGeometry(&geometry,state->first_layer_index,state->layer_count);
	geometry.tp_degree = state->tp_degree;
	geometry.tp_rank = state->tp_rank;
	model_fp = SparkQwen38_27bModuleFingerprint(&geometry,sizeof(geometry),14695981039346656037ull);
	layout_bits[0] = state->cache_layer_stride;
	layout_bits[1] = state->cache_block_stride;
	layout_bits[2] = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
	layout_fp = SparkQwen38_27bModuleFingerprint(layout_bits,sizeof(layout_bits),model_fp);
	return(SparkStageKvClientOpen(&state->kv_client,SPARK_QWEN38_27B_MODULE_TAG,provider,state->stage_index,state->first_layer_index,state->layer_count,model_fp,layout_fp,service,socket_path,pool_bytes,workers));
}

static SparkStatus SparkQwen38_27bModuleValidateLaneSequenceContinuity(
    SparkQwen38_27bModuleState *state,
    const SparkQwen38_27bResidentDecodeStageFrameContext *context,
    const SparkQwen38_27bPrefillFrameView *prefill,
    uint64_t request_generation,
    uint8_t *lane_requires_reset)
{
    uint32_t row_count;
    uint32_t row;

    if (state == 0 || context == 0 || lane_requires_reset == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    row_count = prefill != 0 ? 1u : context->decode_batch->row_count;
    memset(lane_requires_reset, 0, row_count * sizeof(*lane_requires_reset));
    if (prefill != 0)
    {
        uint64_t current_sequence_id;
        uint64_t expected_position;
        uint32_t prefix_restore_in;
        uint32_t restore_first;

        current_sequence_id = state->lane_sequence_ids[prefill->lane_index];
        expected_position = state->lane_next_positions[prefill->lane_index];
        /* A VERIFY_ROW restore also re-establishes the lane at a branch
         * point (the one-frame round's verify starts behind the previous
         * verify's end whenever the chain breaks early), so it gets the
         * same at-or-behind continuity rule as a full restore. */
        restore_first = (context->flags &
            (SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_RESTORE_FIRST |
             SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_RESTORE_VERIFY_ROW)) != 0u
            ? 1u
            : 0u;
        /* A prefix-cache borrow legitimately lands a NEW sequence mid-lane:
         * the restore-in transfer re-seeds the lane's GDN state from the
         * published prefix snapshot and the KV blocks come pinned from the
         * prefix store, so the position-zero rule does not apply. The lane
         * reset below still clears the previous sequence's residue. */
        prefix_restore_in = (context->flags &
            SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_PREFIX_RESTORE_IN) != 0u
            ? 1u
            : 0u;
        if (current_sequence_id == prefill->sequence_id &&
            state->lane_request_generations[prefill->lane_index] == request_generation)
        {
            if (prefill->base_position == 0u && expected_position != 0u)
            {
                /* a re-used sequence id restarting at zero (a fresh client
                 * run of the same batch re-sends the same ids; generations
                 * collide across client processes). Position 0 is always a
                 * legitimate restart - the lane resets below. */
                lane_requires_reset[0] = 1u;
            }
            else if ((restore_first == 0u && prefill->base_position != expected_position) ||
                (restore_first != 0u && prefill->base_position > expected_position))
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
        }
        else
        {
            if (prefill->base_position != 0u && prefix_restore_in == 0u)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            lane_requires_reset[0] = 1u;
        }
        return SPARK_STATUS_OK;
    }

    for (row = 0u; row < context->decode_batch->row_count; row++)
    {
        uint32_t lane;
        uint64_t sequence_id;
        uint64_t position;
        uint64_t current_sequence_id;

        lane = context->decode_batch->row_lane_indices[row];
        sequence_id = context->decode_batch->row_sequence_ids[row];
        position = context->decode_batch->row_positions[row];
        current_sequence_id = state->lane_sequence_ids[lane];
        if (current_sequence_id == sequence_id)
        {
            if (position != state->lane_next_positions[lane])
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
        }
        else
        {
            if (position != 0u)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            lane_requires_reset[row] = 1u;
        }
    }
    return SPARK_STATUS_OK;
}

static void SparkQwen38_27bModuleCommitLaneSequenceContinuity(
    SparkQwen38_27bModuleState *state,
    const SparkQwen38_27bResidentDecodeStageFrameContext *context,
    const SparkQwen38_27bPrefillFrameView *prefill,
    uint64_t request_generation)
{
    uint32_t row;

    if (prefill != 0)
    {
        state->lane_sequence_ids[prefill->lane_index] = prefill->sequence_id;
        state->lane_request_generations[prefill->lane_index] = request_generation;
        state->lane_next_positions[prefill->lane_index] =
            prefill->base_position + prefill->token_count;
        state->lane_warm[prefill->lane_index] = 1u;
        return;
    }
    for (row = 0u; row < context->decode_batch->row_count; row++)
    {
        uint32_t lane;

        lane = context->decode_batch->row_lane_indices[row];
        state->lane_sequence_ids[lane] = context->decode_batch->row_sequence_ids[row];
        state->lane_request_generations[lane] = request_generation;
        state->lane_next_positions[lane] = context->decode_batch->row_positions[row] + 1u;
        state->lane_warm[lane] = 1u;
    }
}

static void SparkQwen38_27bModuleInvalidateLaneSequenceContinuity(
    SparkQwen38_27bModuleState *state,
    const SparkQwen38_27bResidentDecodeStageFrameContext *context,
    const SparkQwen38_27bPrefillFrameView *prefill)
{
    uint32_t row;

    if (prefill != 0)
    {
        state->lane_sequence_ids[prefill->lane_index] = 0u;
        state->lane_next_positions[prefill->lane_index] = 0u;
        state->lane_warm[prefill->lane_index] = 0u;
        return;
    }
    for (row = 0u; row < context->decode_batch->row_count; row++)
    {
        uint32_t lane;

        lane = context->decode_batch->row_lane_indices[row];
        state->lane_sequence_ids[lane] = 0u;
        state->lane_next_positions[lane] = 0u;
        state->lane_warm[lane] = 0u;
    }
}

/* Copy the post-layer hidden into the DSpark tap buffer when the decode
 * reaches one of the 5 target tap layers {4,16,28,40,52}. B1 only: the tap
 * holds one position (the committed token's hidden). */
/* The DSpark drafter: a 5-layer full-attention decoder that emits a 7-token
 * block. Weights are loaded from the separate drafter pack; the target's token
 * embedding and lm_head are shared (the drafter pack carries neither). */


/* DFlash2 block-diffusion draft forward, cache-based (upstream semantics):
 * the drafter attends over a per-layer context KV built from the target's
 * per-position taps (last 2048 positions -> fc -> hidden_norm -> k/v proj ->
 * k_norm -> rope) plus the block's own rows. Block = [embed(anchor), masks].
 * The conv-wrapped 5-layer walk, shared lm_head, host top-16/selector/walk
 * are unchanged from the dual-source version. */
static SparkStatus SparkQwen38_27bModuleRunDsparkBlockForward(
	SparkQwen38_27bModuleState *state,
	SparkQwen38_27bModuleSlot *slot,
	const SparkQwen38_27bDsparkDraftView *view,
	const uint32_t *row_tokens_host,
	uint32_t rows)
{
	SparkQwen38_27bDsparkWeights *w = &state->dspark_weights;
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint8_t *scr = (uint8_t *)slot->dspark_scratch;
	const uint32_t B = SPARK_QWEN38_27B_DSPARK_BLOCK_SIZE;
	const uint32_t H = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
	const uint32_t V = SPARK_QWEN38_27B_MODEL_OUTPUT_VOCAB_COUNT;
	const uint32_t R = SPARK_QWEN38_27B_DSPARK_SELECTOR_RANK;
	const uint32_t K = SPARK_QWEN38_27B_DSPARK_SELECTOR_TOP_K;
	const uint32_t conv_groups = H / SPARK_QWEN38_27B_DSPARK_CONV_GROUP_SIZE;
	const uint32_t conv_side_base = SPARK_QWEN38_27B_DSPARK_CONV_KERNEL_SIZE * H;
	/* scratch carve-out (bf16 elements) */
	uint16_t *block_hidden = (uint16_t *)scr + H;
	uint16_t *conv_h = block_hidden + (uint64_t)B * H;
	uint16_t *conv_proj_a = conv_h + (uint64_t)B * H;
	uint16_t *conv_proj_m = conv_proj_a + (uint64_t)B * SPARK_QWEN38_27B_DSPARK_CONV_PROJ_ROWS;
	uint16_t *q = conv_proj_m + (uint64_t)B * SPARK_QWEN38_27B_DSPARK_CONV_PROJ_ROWS;
	uint16_t *attn_out = q + (uint64_t)B * H;
	uint16_t *norm = attn_out + (uint64_t)B * H;
	uint16_t *ffn = norm + (uint64_t)B * H;
	uint16_t *up = ffn + (uint64_t)B * SPARK_QWEN38_27B_DSPARK_FFN_INTERMEDIATE;
	uint16_t *logits = up + (uint64_t)B * SPARK_QWEN38_27B_DSPARK_FFN_INTERMEDIATE;
	/* Dual-source degeneration: window = 1 (the LAST tap position only,
	 * position base-1) + block rows at base..base+B-1, anchor = the frame's
	 * emission. This is the convention whose iteration-1 drafts were
	 * PERFECT ([270x7]); the per-position cache context made iteration 1
	 * worse, and the recurrence races (now serialized) were what killed
	 * iterations 2+. */
	const uint64_t base = view->base_position;
	uint32_t wnd_bound;
	uint32_t ctx_tail;
	uint32_t block_kv_on;
	/* env-tunable recent-context bound (default full; small windows measured
	 * best post-fix). */
	{
		const char *wenv = getenv("SPARK_QWEN38_27B_DFLASH2_WINDOW");
		wnd_bound = wenv != 0 ? (uint32_t)strtoul(wenv,0,0) : 2048u;
	}
	{
		const char *benv = getenv("SPARK_QWEN38_27B_DFLASH2_BLOCK_KV");
		block_kv_on = benv != 0 && benv[0] != '0' ? 1u : 0u;
	}
	/* The verify-forward tail rows (rejected drafts' hiddens); measured
	 * negative, default 0. */
	{
		const char *tenv = getenv("SPARK_QWEN38_27B_DFLASH2_CTX_TAIL");
		ctx_tail = tenv != 0 ? (uint32_t)strtoul(tenv,0,0) : 0u;
	}
	uint16_t *ctx_kv = (uint16_t *)state->dflash_ctx_kv;
	uint16_t *kv_k;
	uint16_t *kv_v;
	uint32_t layer;
	uint32_t prev;
	SparkStatus status;
	cudaError_t error;
	uint32_t blk;
	(void)rows;
	if ( w->armed == 0u )
		return(SPARK_STATUS_OK);
	if ( view == 0 || view->draft_token_ids == 0 || state->dflash_taps_history == 0 )
	{
		fprintf(stderr,"dflash2_trace drafter_invalid view=%p ids=%p taps=%p\n",(void*)view,(void*)(view!=0?(void*)view->draft_token_ids:0),(void*)state->dflash_taps_history);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	/* Multi-block (padding) drafting: block i anchors on output row i's
	 * emission at base+i - vLLM's per-row draft-then-select shape; the host
	 * picks block m (the accept depth) from the matrix after the verify.
	 * Each block's context window covers taps < base+i, so block m sees the
	 * accepted-prefix rows fresh and none of the rejected tail (the deferred
	 * seed pair, per block). multi_block_count <= 1 keeps the legacy
	 * anchor-row-0 single block. */
	{
		uint32_t multi = view->multi_block_count;
		uint32_t sel_block = 0u;
		if ( multi < 1u )
			multi = 1u;
		if ( multi > SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_DSPARK_MAX_MULTI_BLOCKS )
			multi = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_DSPARK_MAX_MULTI_BLOCKS;
		if ( multi > 1u && row_tokens_host != 0 )
		{
			/* padding-select: compute the verify's own accept depth (its
			 * emissions vs the walked draft rows) and draft ONLY that block -
			 * the host discards every other block's output anyway, and the
			 * per-block selector pass is host-bound. The adapter recomputes
			 * the identical m from the same data after the frame. */
			uint32_t emissions[SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS];
			if ( rows > SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS )
				rows = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS;
			if ( cudaStreamSynchronize(stream) == cudaSuccess && cudaMemcpy(emissions,slot->output_token_ids,(size_t)rows * sizeof(uint32_t),cudaMemcpyDeviceToHost) == cudaSuccess )
			{
				while ( sel_block + 1u < rows && emissions[sel_block] == row_tokens_host[sel_block + 1u] )
					sel_block++;
			}
			multi = sel_block + 1u;
		}
		status = SPARK_STATUS_OK;
		for (blk = sel_block; status == SPARK_STATUS_OK && blk < multi; blk++)
		{
			const uint64_t base_blk = base + blk;
			uint32_t avail = (uint32_t)(base_blk < 2048u ? base_blk : 2048u);
			const uint32_t window = wnd_bound < avail ? wnd_bound : avail;
			/* the window INCLUDES the walked row's tap g_P (the deferred pair) */
			const uint64_t window_base = base_blk - window;
			const uint32_t nkv = window + ctx_tail + B;
	/* 1) context window: fc(cat 5 taps per position) -> hidden_norm.
	 * INCREMENTAL: only positions committed since the last block forward
	 * are projected (the reference's precompute-and-store semantics - it
	 * processes each round's NEW rows only). A backward base resets the
	 * watermark (new sequence on the lane); ctx-tail mode keeps the full
	 * recompute (its tail rows are per-round). Kill-switch:
	 * SPARK_QWEN38_27B_DFLASH2_CTX_CACHE=0. */
	{
		const char *cenv = getenv("SPARK_QWEN38_27B_DFLASH2_CTX_CACHE");
		uint32_t cache_on = (cenv == 0 || cenv[0] != '0') && ctx_tail == 0u ? 1u : 0u;
		error = cudaSuccess;
		if ( cache_on != 0u && base_blk < state->dflash_ctx_valid_to )
		{
			if ( base_blk + 64u < state->dflash_ctx_valid_to )
			{
				/* a FAR backward base = a NEW sequence on this lane (a fresh
				 * request starts at 0, a prefix borrow at the prefix edge):
				 * the block-KV history still holds the previous sequence's
				 * rows (keyed by positions that collide with the new one) -
				 * without this reset, later requests on the same daemon
				 * attend stale rows and acceptance collapses (measured:
				 * run 1 E=2.80, run 2 E=1.80 on the identical prompt) */
				state->dflash_ctx_valid_to = 0u;
				state->dflash_hist_count = 0u;
			}
			else
			{
				/* an intra-sequence rollback (rejection): the walk depth is
				 * <= k+verify, so a small backward step is NOT a new
				 * sequence. Rewind the projection watermark so the
				 * re-committed rows re-project from their final taps; KEEP
				 * the block history (resetting it on every rejection
				 * measured E 5.66 -> 4.81, +11 rounds on O512). History
				 * rows at positions >= base_blk are stale-by-definition and
				 * are filtered at assembly below. */
				state->dflash_ctx_valid_to = base_blk;
			}
		}
		if ( cache_on != 0u )
		{
			uint64_t new_from = window_base > state->dflash_ctx_valid_to ? window_base : state->dflash_ctx_valid_to;
			/* per-row projection: the wide fc (input 25600) at rows 2..4
			 * lands on the library scalar kernel whose 100KB dynamic
			 * shared request fails without the opt-in; rows=1 rides the
			 * module's lean wide-B1 kernel. New rows per round are few
			 * (m+1 <= 8), so the per-row loop is cheap. */
			while ( new_from < base_blk && error == cudaSuccess )
			{
				error = SparkQwen38_27bLaunchLinear(stream,&w->projector,(const uint8_t *)state->dflash_taps_history + new_from * 5u * H * 2u,(uint8_t *)state->dflash_fc_out + new_from * H * 2u,1u);
				if ( error == cudaSuccess )
					error = SparkQwen38_27bLaunchRmsNorm(stream,(const uint8_t *)state->dflash_fc_out + new_from * H * 2u,w->hidden_norm_bf16,(uint8_t *)state->dflash_ctx_normed + new_from * H * 2u,1u,H,SPARK_QWEN38_27B_MODEL_RMS_NORM_EPSILON);
				{
					uint32_t cl;
					for (cl = 0u; cl < SPARK_QWEN38_27B_DSPARK_LAYER_COUNT && error == cudaSuccess; cl++)
					{
						SparkQwen38_27bDsparkLayerWeights *cw = &w->layer[cl];
						const uint64_t cache_lk = (uint64_t)cl * 2u * 2048u * 1024u;
						const uint64_t cache_lv = cache_lk + (uint64_t)2048u * 1024u;
						error = SparkQwen38_27bLaunchLinear(stream,&cw->k,(const uint8_t *)state->dflash_ctx_normed + new_from * H * 2u,(uint16_t *)state->dflash_ctx_kv_cache + cache_lk + new_from * 1024u,1u);
						if ( error == cudaSuccess )
							error = SparkQwen38_27bLaunchLinear(stream,&cw->v,(const uint8_t *)state->dflash_ctx_normed + new_from * H * 2u,(uint16_t *)state->dflash_ctx_kv_cache + cache_lv + new_from * 1024u,1u);
					}
				}
				new_from++;
			}
			if ( error == cudaSuccess )
				state->dflash_ctx_valid_to = base_blk;
		}
		else
		{
			error = SparkQwen38_27bLaunchLinear(stream,&w->projector,(const uint8_t *)state->dflash_taps_history + window_base * 5u * H * 2u,state->dflash_fc_out,window);
			if ( error == cudaSuccess )
				error = SparkQwen38_27bLaunchRmsNorm(stream,state->dflash_fc_out,w->hidden_norm_bf16,state->dflash_ctx_normed,window,H,SPARK_QWEN38_27B_MODEL_RMS_NORM_EPSILON);
		}
	}
	/* 2) block[0] = embed(anchor = the frame's EMISSION), roped at ITS OWN
	 * position base, one past the g_P context row at base-1 (the deferred
	 * seed pair). Block rows at base..base+7; row r's walk output predicts
	 * the token at base+r (own-position). */
	if ( error == cudaSuccess )
		error = SparkQwen38_27bLaunchEmbeddingGather(stream,slot->output_token_ids + blk,state->token_embedding_bf16,block_hidden,1u);
	if ( error == cudaSuccess )
		error = SparkQwen38_27bLaunchEmbeddingGather(stream,slot->dspark_mask_token_ids,state->token_embedding_bf16,block_hidden + H,B - 1u);
	/* 3) prep positions: window rows at window_base..base-1 (g_P included),
	 * block rows at base..base+B-1 - the anchor row sits one PAST its paired
	 * g_P row (llama.cpp's DEFERRED pair: t_{P+1} at P+1, g_P at P; a shared
	 * position measurably zeroes deep drafts). (k norm+rope and q rope are
	 * absolute.) */
	for (layer = 0u; layer < window + ctx_tail; layer++)
		state->dflash_positions_host[layer] = window_base + layer;
	for (layer = 0u; layer < B; layer++)
		state->dflash_positions_host[window + ctx_tail + layer] = base_blk + layer;
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(state->dflash_positions,state->dflash_positions_host,(size_t)nkv * sizeof(uint64_t),cudaMemcpyHostToDevice,stream);
	status = SparkStageModuleCudaStatus(SPARK_QWEN38_27B_MODULE_TAG,error,"dflash2_head_init");
	if ( status == SPARK_STATUS_OK && blk == 0u && getenv("SPARK_QWEN38_27B_DFLASH2_CTX_DUMP") != 0 )
	{
		static uint32_t ctx_dump_count = 0u;
		uint32_t ctx_dump_want = (uint32_t)strtoul(getenv("SPARK_QWEN38_27B_DFLASH2_CTX_DUMP"),0,0);
		ctx_dump_count++;
		if ( ctx_dump_want == 0u || ctx_dump_count == ctx_dump_want )
		{
			FILE *file;
			uint16_t *host = (uint16_t *)malloc((size_t)5u * H * 2u);
			if ( cudaMemcpy(host,(const uint8_t *)state->dflash_taps_history + (base - 1u) * 5u * H * 2u,(size_t)5u * H * 2u,cudaMemcpyDeviceToHost) == cudaSuccess )
			{
				file = fopen("/tmp/ctxdump_taps_last.bin","wb");
				if ( file != 0 ) { fwrite(host,1,(size_t)5u * H * 2u,file); fclose(file); }
			}
			if ( cudaMemcpy(host,(const uint8_t *)state->dflash_fc_out + (uint64_t)(window - 1u) * H * 2u,(size_t)H * 2u,cudaMemcpyDeviceToHost) == cudaSuccess )
			{
				file = fopen("/tmp/ctxdump_fc_last.bin","wb");
				if ( file != 0 ) { fwrite(host,1,(size_t)H * 2u,file); fclose(file); }
			}
			if ( cudaMemcpy(host,(const uint8_t *)state->dflash_ctx_normed + (uint64_t)(window - 1u) * H * 2u,(size_t)H * 2u,cudaMemcpyDeviceToHost) == cudaSuccess )
			{
				file = fopen("/tmp/ctxdump_normed_last.bin","wb");
				if ( file != 0 ) { fwrite(host,1,(size_t)H * 2u,file); fclose(file); }
			}
			free(host);
			{
				/* the full neighborhood: every position any frame walked around
				 * this draft (verify rows incl. rejected, replay rows), so the
				 * numpy sweep can test any context-window convention */
				uint64_t nb_base = 0u;
				uint32_t nb = (uint32_t)(base < 2048u ? base + 8u : 2056u);
				uint16_t *win = (uint16_t *)malloc((size_t)nb * 5u * H * 2u);
				if ( win != 0 && cudaMemcpy(win,(const uint8_t *)state->dflash_taps_history + nb_base * 5u * H * 2u,(size_t)nb * 5u * H * 2u,cudaMemcpyDeviceToHost) == cudaSuccess )
				{
					FILE *wf = fopen("/tmp/ctxwin_taps.bin","wb");
					if ( wf != 0 ) { fwrite(win,1,(size_t)nb * 5u * H * 2u,wf); fclose(wf); }
				}
				free(win);
				file = fopen("/tmp/ctxwin.meta","w");
				if ( file != 0 ) { fprintf(file,"base=%llu window=%u nb_base=%llu nb=%u\n",(unsigned long long)base,window,(unsigned long long)nb_base,nb); fclose(file); }
			}
			{
				uint32_t anchor_id = 0u;
				/* the anchor the block actually embeds: the frame's EMISSION
				 * (output_token_ids[0]); the old input-row read dumped the
				 * walked token instead and mislabeled the oracle sweeps */
				cudaMemcpy(&anchor_id,slot->output_token_ids,sizeof(uint32_t),cudaMemcpyDeviceToHost);
				fprintf(stderr,"ctxdump run=%u base=%llu window=%u anchor=%u\n",ctx_dump_count,(unsigned long long)base,window,anchor_id);
				{
					FILE *mf = fopen("/tmp/ctxwin_anchor","w");
					if ( mf != 0 ) { fprintf(mf,"%u\n",anchor_id); fclose(mf); }
				}
			}
		}
	}
	for (layer = 0u; status == SPARK_STATUS_OK && layer < SPARK_QWEN38_27B_DSPARK_LAYER_COUNT; layer++)
	{
		SparkQwen38_27bDsparkLayerWeights *lw = &w->layer[layer];
		const uint64_t kv_stride = (2048u + 8u + SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_DFLASH_BLOCK_KV_CAP) * 1024u;
		kv_k = ctx_kv + (uint64_t)(layer * 2u + 0u) * kv_stride;
		kv_v = ctx_kv + (uint64_t)(layer * 2u + 1u) * kv_stride;
		(void)kv_stride;
		/* attention half: norm -> conv.prepare -> q from the prepared h;
		 * K/V = [context window (from the target taps) || block rows]. */
		error = SparkQwen38_27bLaunchRmsNorm(stream,block_hidden,lw->input_norm_bf16,norm,B,H,SPARK_QWEN38_27B_MODEL_RMS_NORM_EPSILON);
		if ( error == cudaSuccess )
			error = SparkQwen38_27bLaunchLinear(stream,&lw->conv_attn_proj,norm,conv_proj_a,B);
		if ( error == cudaSuccess )
			error = SparkQwen38_27bLaunchDsparkConv(stream,norm,conv_proj_a,lw->conv_attn_base_bf16,conv_h,B,conv_groups,SPARK_QWEN38_27B_DSPARK_CONV_GROUP_SIZE,0u);
		if ( error == cudaSuccess )
			error = SparkQwen38_27bLaunchLinear(stream,&lw->q,conv_h,q,B);
		if ( error == cudaSuccess && state->dflash_ctx_valid_to >= base_blk )
		{
			/* cached: assemble this round's window rows from the
			 * position-keyed per-layer cache (contiguous D2D copy) */
			const uint64_t cache_lk = (uint64_t)layer * 2u * 2048u * 1024u;
			error = cudaMemcpyAsync(kv_k,(const uint16_t *)state->dflash_ctx_kv_cache + cache_lk + window_base * 1024u,(size_t)window * 1024u * 2u,cudaMemcpyDeviceToDevice,stream);
			if ( error == cudaSuccess )
				error = cudaMemcpyAsync(kv_v,(const uint16_t *)state->dflash_ctx_kv_cache + cache_lk + (uint64_t)2048u * 1024u + window_base * 1024u,(size_t)window * 1024u * 2u,cudaMemcpyDeviceToDevice,stream);
		}
		else if ( error == cudaSuccess )
		{
			error = SparkQwen38_27bLaunchLinear(stream,&lw->k,state->dflash_ctx_normed,kv_k,window);
			if ( error == cudaSuccess )
				error = SparkQwen38_27bLaunchLinear(stream,&lw->v,state->dflash_ctx_normed,kv_v,window);
		}
		if ( error == cudaSuccess )
			error = SparkQwen38_27bLaunchLinear(stream,&lw->k,conv_h,kv_k + (uint64_t)(window + ctx_tail) * 1024u,B);
		if ( error == cudaSuccess )
			error = SparkQwen38_27bLaunchLinear(stream,&lw->v,conv_h,kv_v + (uint64_t)(window + ctx_tail) * 1024u,B);
		{
			/* the persistent block-KV history rides AFTER the current block
			 * rows: [ctx || this block || past blocks]. Raw rows - the
			 * attention kernel norms and ropes each row at its own position. */
			const uint32_t hist_base = (window + ctx_tail + B) * 1024u;
			const uint64_t hist_lk = (uint64_t)layer * SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_DFLASH_BLOCK_KV_CAP * 1024u;
			uint32_t nkv_eff = nkv;
			uint32_t hi;
			if ( block_kv_on != 0u && state->dflash_hist_count != 0u )
			{
				uint32_t live = 0u;
				for (hi = 0u; hi < state->dflash_hist_count; hi++)
				{
					/* rows at positions >= base_blk are residue of the
					 * rejected walk this round re-produces; they must not
					 * be attended (the walk overwrites them after use) */
					if ( state->dflash_hist_pos_host[hi] >= base_blk )
						continue;
					if ( error == cudaSuccess )
						error = cudaMemcpyAsync(kv_k + hist_base + live * 1024u,(const uint16_t *)state->dflash_block_hist_k + hist_lk + (uint64_t)hi * 1024u,1024u * 2u,cudaMemcpyDeviceToDevice,stream);
					if ( error == cudaSuccess )
						error = cudaMemcpyAsync(kv_v + hist_base + live * 1024u,(const uint16_t *)state->dflash_block_hist_v + hist_lk + (uint64_t)hi * 1024u,1024u * 2u,cudaMemcpyDeviceToDevice,stream);
					if ( window + ctx_tail + B + live >= SPARK_QWEN38_27B_DFLASH2_FRAME_KV_ROWS )
					{
						fprintf(stderr,"qwen38_27b_stage dflash2_block_kv_frame_overflow nkv_eff=%u frame_rows=%u\n",
							window + ctx_tail + B + live,SPARK_QWEN38_27B_DFLASH2_FRAME_KV_ROWS);
						return(SPARK_STATUS_CAPACITY_EXCEEDED);
					}
					state->dflash_positions_host[window + ctx_tail + B + live] = state->dflash_hist_pos_host[hi];
					live++;
				}
				nkv_eff = nkv + live;
			}
			if ( error == cudaSuccess )
				error = cudaMemcpyAsync(state->dflash_positions,state->dflash_positions_host,(size_t)nkv_eff * sizeof(uint64_t),cudaMemcpyHostToDevice,stream);
			if ( error == cudaSuccess )
				error = SparkQwen38_27bLaunchDsparkCacheAttn(stream,q,kv_k,kv_v,lw->q_norm_bf16,lw->k_norm_bf16,state->dflash_positions,attn_out,B,nkv_eff,window + ctx_tail);
			/* append this block's raw k/v rows to the per-layer history,
			 * keyed by position (re-walked positions overwrite their slot) */
			if ( block_kv_on != 0u && error == cudaSuccess )
			{
				for (hi = 0u; hi < B; hi++)
				{
					const uint64_t bpos = state->dflash_positions_host[window + ctx_tail + hi];
					uint32_t slot = state->dflash_hist_count;
					uint32_t si;
					for (si = 0u; si < state->dflash_hist_count; si++)
						if ( state->dflash_hist_pos_host[si] == bpos )
						{
							slot = si;
							break;
						}
					if ( slot >= SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_DFLASH_BLOCK_KV_CAP )
						continue;
					if ( slot == state->dflash_hist_count )
						state->dflash_hist_count++;
					state->dflash_hist_pos_host[slot] = bpos;
					error = cudaMemcpyAsync((uint16_t *)state->dflash_block_hist_k + hist_lk + (uint64_t)slot * 1024u,kv_k + (uint64_t)(window + ctx_tail + hi) * 1024u,1024u * 2u,cudaMemcpyDeviceToDevice,stream);
					if ( error == cudaSuccess )
						error = cudaMemcpyAsync((uint16_t *)state->dflash_block_hist_v + hist_lk + (uint64_t)slot * 1024u,kv_v + (uint64_t)(window + ctx_tail + hi) * 1024u,1024u * 2u,cudaMemcpyDeviceToDevice,stream);
					if ( error != cudaSuccess )
						break;
				}
			}
		}
		if ( layer == 0u && getenv("SPARK_QWEN38_27B_DFLASH2_CTX_DUMP") != 0 )
		{
			static uint32_t l0_dump_count = 0u;
			l0_dump_count++;
			uint32_t l0_want = (uint32_t)strtoul(getenv("SPARK_QWEN38_27B_DFLASH2_CTX_DUMP"),0,0);
			if ( l0_want == 0u || l0_dump_count == l0_want )
			{
				uint32_t rows_sample[5];
				uint32_t sample_count = window >= 2u ? 5u : window + 2u;
				uint32_t si;
				FILE *sf;
				rows_sample[0] = 0u;
				rows_sample[1] = window >= 2u ? window - 1u : 0u;
				rows_sample[2] = window;
				rows_sample[3] = window + 1u;
				rows_sample[4] = nkv - 1u;
				cudaStreamSynchronize(stream);
				/* debug-dump null guards (the 2026-08-23 spec-does-not-work
				 * report: CTX_DUMP/L0_DUMP env + a failed fopen (fd
				 * exhaustion, /tmp state) = fwrite(NULL) SEGV on the first
				 * decode round - misdiagnosed as 'speculation broken') */
				sf = fopen("/tmp/l0_sample_rows.txt","w");
				if ( sf != 0 )
				{
					for (si = 0u; si < sample_count; si++)
						fprintf(sf,"%u\n",rows_sample[si]);
					fclose(sf);
				}
				sf = fopen("/tmp/l0_kv_k.bin","wb");
				if ( sf != 0 )
				{
					for (si = 0u; si < sample_count; si++)
						{ uint16_t rowbuf[1024]; cudaMemcpy(rowbuf,kv_k + (uint64_t)rows_sample[si] * 1024u,sizeof(rowbuf),cudaMemcpyDeviceToHost); fwrite(rowbuf,1,sizeof(rowbuf),sf); }
					fclose(sf);
				}
				sf = fopen("/tmp/l0_kv_v.bin","wb");
				if ( sf != 0 )
				{
					for (si = 0u; si < sample_count; si++)
						{ uint16_t rowbuf[1024]; cudaMemcpy(rowbuf,kv_v + (uint64_t)rows_sample[si] * 1024u,sizeof(rowbuf),cudaMemcpyDeviceToHost); fwrite(rowbuf,1,sizeof(rowbuf),sf); }
					fclose(sf);
				}
				sf = fopen("/tmp/l0_q.bin","wb");
				if ( sf != 0 )
				{
					for (si = 0u; si < 2u; si++)
						{ uint16_t rowbuf[4096]; cudaMemcpy(rowbuf,q + (uint64_t)si * 4096u,sizeof(rowbuf),cudaMemcpyDeviceToHost); fwrite(rowbuf,1,sizeof(rowbuf),sf); }
					fclose(sf);
				}
				sf = fopen("/tmp/l0_attn.bin","wb");
				if ( sf != 0 )
				{
					{ uint16_t rowbuf[4096]; cudaMemcpy(rowbuf,attn_out,sizeof(rowbuf),cudaMemcpyDeviceToHost); fwrite(rowbuf,1,sizeof(rowbuf),sf); }
					fclose(sf);
				}
			}
		}
		if ( error == cudaSuccess )
			error = SparkQwen38_27bLaunchLinear(stream,&lw->o,attn_out,q,B);
		if ( error == cudaSuccess )
			error = SparkQwen38_27bLaunchDsparkConv(stream,q,conv_proj_a,(const char *)lw->conv_attn_base_bf16 + (uint64_t)conv_side_base * 2u,conv_h,B,conv_groups,SPARK_QWEN38_27B_DSPARK_CONV_GROUP_SIZE,1u);
		if ( error == cudaSuccess )
			error = SparkQwen38_27bLaunchResidualAdd(stream,block_hidden,conv_h,B,H);
		/* mlp half: the same conv wrap around gate/up/swiglu/down. */
		if ( error == cudaSuccess )
			error = SparkQwen38_27bLaunchRmsNorm(stream,block_hidden,lw->post_norm_bf16,norm,B,H,SPARK_QWEN38_27B_MODEL_RMS_NORM_EPSILON);
		if ( error == cudaSuccess )
			error = SparkQwen38_27bLaunchLinear(stream,&lw->conv_mlp_proj,norm,conv_proj_m,B);
		if ( error == cudaSuccess )
			error = SparkQwen38_27bLaunchDsparkConv(stream,norm,conv_proj_m,lw->conv_mlp_base_bf16,conv_h,B,conv_groups,SPARK_QWEN38_27B_DSPARK_CONV_GROUP_SIZE,0u);
		if ( error == cudaSuccess )
			error = SparkQwen38_27bLaunchLinear(stream,&lw->gate,conv_h,ffn,B);
		if ( error == cudaSuccess )
			error = SparkQwen38_27bLaunchLinear(stream,&lw->up,conv_h,up,B);
		if ( error == cudaSuccess )
			error = SparkQwen38_27bLaunchSwiGlu(stream,ffn,up,B,SPARK_QWEN38_27B_DSPARK_FFN_INTERMEDIATE);
		if ( error == cudaSuccess )
			error = SparkQwen38_27bLaunchLinear(stream,&lw->down,up,q,B);
		if ( error == cudaSuccess )
			error = SparkQwen38_27bLaunchDsparkConv(stream,q,conv_proj_m,(const char *)lw->conv_mlp_base_bf16 + (uint64_t)conv_side_base * 2u,conv_h,B,conv_groups,SPARK_QWEN38_27B_DSPARK_CONV_GROUP_SIZE,1u);
		if ( error == cudaSuccess )
			error = SparkQwen38_27bLaunchResidualAdd(stream,block_hidden,conv_h,B,H);
		status = SparkStageModuleCudaStatus(SPARK_QWEN38_27B_MODULE_TAG,error,"dflash2_layer");
	}
	/* 3) final norm + shared lm_head -> logits (B x vocab); D2H the mask-row
	 * logits, the final hidden (selector gate input) and the anchor C0. */
	if ( status == SPARK_STATUS_OK )
	{
		SparkQwen38_27bLinearView lm_head;
		memset(&lm_head,0,sizeof(lm_head));
		lm_head.abi_version = SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION;
		lm_head.weight_format = 0u; /* BF16 */
		lm_head.input_dimension = SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION;
		lm_head.output_dimension = SPARK_QWEN38_27B_MODEL_OUTPUT_VOCAB_COUNT;
		lm_head.weight_payload = state->lm_head_weight_bf16;
		lm_head.weight_payload_bytes = (uint64_t)SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION * SPARK_QWEN38_27B_MODEL_OUTPUT_VOCAB_COUNT * 2u;
		error = SparkQwen38_27bLaunchRmsNorm(stream,block_hidden,w->final_norm_bf16,norm,B,H,SPARK_QWEN38_27B_MODEL_RMS_NORM_EPSILON);
		if ( error == cudaSuccess )
			error = SparkQwen38_27bLaunchLinear(stream,&lm_head,norm,logits,B);
		if ( error == cudaSuccess )
			error = SparkQwen38_27bLaunchDsparkSelect(stream,logits,norm,w->selector_hidden_proj.weight_payload,state->dspark_sel_out_dev,B,V,H,R,K);
		if ( error == cudaSuccess )
			error = cudaMemcpyAsync(slot->dspark_hidden_host,norm,(size_t)B * H * sizeof(uint16_t),cudaMemcpyDeviceToHost,stream);
		if ( error == cudaSuccess )
			error = cudaMemcpyAsync(state->dspark_sel_out_host,state->dspark_sel_out_dev,(size_t)(B - 1u) * (2u * K + R) * 4u,cudaMemcpyDeviceToHost,stream);
		/* Anchor = the frame's emission: the decode's output (iteration 1)
		 * or the replay's output (replay-tail draft, iterations 2+). Both are
		 * the first candidate token AFTER the tap position. */
		if ( error == cudaSuccess )
			error = cudaMemcpyAsync(&prev,slot->output_token_ids + blk,sizeof(uint32_t),cudaMemcpyDeviceToHost,stream);
		if ( error == cudaSuccess )
			error = cudaStreamSynchronize(stream);
		if ( error == cudaSuccess )
		{
			static int dflash2_dump_done = 0;
			if ( dflash2_dump_done == 0 )
			{
				/* one-shot parity dump: taps + C0 + logits + final hidden */
				dflash2_dump_done = 1;
				FILE *df;
				cudaMemcpy(slot->dspark_logits_host,logits,(size_t)B * V * 2u,cudaMemcpyDeviceToHost);
				uint16_t *taps_host = (uint16_t *)malloc((size_t)5u * H * 2u);
				cudaMemcpy(taps_host,slot->dspark_tap_buffer,(size_t)5u * H * 2u,cudaMemcpyDeviceToHost);
				/* same null-guard discipline for the one-shot parity dump */
				df = fopen("/tmp/dflash2_taps.bin","wb");
				if ( df != 0 ) { fwrite(taps_host,1,(size_t)5u * H * 2u,df); fclose(df); }
				free(taps_host);
				df = fopen("/tmp/dflash2_c0.bin","wb");
				if ( df != 0 ) { fwrite(&prev,1,4u,df); fclose(df); }
				df = fopen("/tmp/dflash2_logits.bin","wb");
				if ( df != 0 ) { fwrite(slot->dspark_logits_host,1,(size_t)B * V * 2u,df); fclose(df); }
				df = fopen("/tmp/dflash2_hidden.bin","wb");
				if ( df != 0 ) { fwrite(slot->dspark_hidden_host,1,(size_t)B * H * 2u,df); fclose(df); }
				fprintf(stderr,"dflash2_dump c0=%u\n",prev);
			}
		}
		/* 4) candidate selector, host pass (numpy reference rounding order):
		 * top-16 per mask row (value desc, index asc), hidden projection gate,
		 * [slots, K, K] edge lattice, greedy walk from the C0 anchor. */
		if ( error == cudaSuccess && getenv("SPARK_QWEN38_27B_DSPARK_SEL_CHECK") != 0 )
		{
			/* parity oracle: recompute the device front-end's outputs with the
			 * original scalar host pass and print the first divergence */
			const uint32_t *dev_ids = state->dspark_sel_out_host;
			const float *dev_scores = (const float *)(dev_ids + (B - 1u) * K);
			const float *dev_hproj = dev_scores + (B - 1u) * K;
			uint32_t slot_i;
			cudaMemcpy(slot->dspark_logits_host,logits,(size_t)B * V * 2u,cudaMemcpyDeviceToHost);
			for (slot_i = 0u; slot_i < B - 1u; slot_i++)
			{
				const uint16_t *row = slot->dspark_logits_host + (uint64_t)(slot_i + 1u) * V;
				uint32_t hid[K];
				float hun[K];
				float hp[R];
				uint32_t fill, v, c2;
				for (fill = 0u; fill < K; fill++)
				{
					hid[fill] = 0u;
					hun[fill] = -3.4028235e38f;
				}
				for (v = 0u; v < V; v++)
				{
					const float value = SparkQwen38_27bModuleBf16ToFloat(row[v]);
					uint32_t insert = K;
					uint32_t shift;
					while ( insert > 0u && value > hun[insert - 1u] )
						insert--;
					if ( insert == K )
						continue;
					for (shift = K - 1u; shift > insert; shift--)
					{
						hun[shift] = hun[shift - 1u];
						hid[shift] = hid[shift - 1u];
					}
					hun[insert] = value;
					hid[insert] = v;
				}
				for (fill = 0u; fill < R; fill++)
				{
					const uint16_t *hw = w->selector_hidden_proj_host + (uint64_t)fill * H;
					const uint16_t *hidden = slot->dspark_hidden_host + (uint64_t)(slot_i + 1u) * H;
					float acc = 0.0f;
					uint32_t c;
					for (c = 0u; c < H; c++)
						acc += SparkQwen38_27bModuleBf16ToFloat(hidden[c]) * SparkQwen38_27bModuleBf16ToFloat(hw[c]);
					hp[fill] = SparkQwen38_27bModuleBf16ToFloat(SparkQwen38_27bModuleFloatToBf16(acc));
				}
				for (c2 = 0u; c2 < K; c2++)
				{
					if ( hid[c2] != dev_ids[slot_i * K + c2] || hun[c2] != dev_scores[slot_i * K + c2] )
						fprintf(stderr,"sel_check slot=%u rank=%u host=(%u,%f) dev=(%u,%f)\n",slot_i,c2,hid[c2],hun[c2],dev_ids[slot_i * K + c2],dev_scores[slot_i * K + c2]);
				}
				for (c2 = 0u; c2 < R; c2++)
				{
					if ( hp[c2] != dev_hproj[slot_i * R + c2] )
					{
						fprintf(stderr,"sel_check hproj slot=%u r=%u host=%f dev=%f\n",slot_i,c2,hp[c2],dev_hproj[slot_i * R + c2]);
						break;
					}
				}
			}
		}
		if ( error == cudaSuccess )
		{
			/* the selector front-end ran on device: top-16 ids/scores (the
			 * host pass's exact value-desc/index-asc order) plus the
			 * bf16-rounded hidden projection, all in one compact copy */
			const uint32_t *top_ids = state->dspark_sel_out_host;
			uint32_t slot_i;
			/* Draft selection = the serving semantics of the SOTA reference:
			 * per-mask-row full-vocab ARGMAX (v1 DFlashSpeculator.sample_draft
			 * -> gumbel_sample(compute_logits(mask hiddens)); greedy == argmax).
			 * The codebook lattice walk (dflash2 speculator) is NOT the serving
			 * path - it never loads in vllm serve. The top-16 is value-desc /
			 * index-asc, so rank 0 IS the argmax (lowest index on ties, like
			 * torch.argmax). Validated on the reference's own dumped inputs:
			 * 96-100% draft agreement at every position vs 87% best-case for
			 * the walk (tools/qwen38_27b_dflash2_vllm_input_parity.py). */
			for (slot_i = 0u; slot_i < B - 1u; slot_i++)
			{
				view->draft_token_ids[blk * (B - 1u) + slot_i] = top_ids[slot_i * K];
			}
			if ( blk == 0u && getenv("SPARK_QWEN38_27B_DFLASH2_CTX_DUMP") != 0 )
				{
					/* Per-run parity capture, read AFTER the section-3 stream
					 * sync (race-free): a taps slice wide enough for any
					 * convention sweep, the true walk anchor (prev), and the
					 * device's own walk output. Consumed round-by-round by
					 * tools/qwen38_27b_dflash2_deep_parity.py against the numpy
					 * reference to localize the deep-draft acceptance gap. */
					static uint32_t parity_runs = 0u;
					if ( parity_runs < 80u )
					{
						char path[128];
						/* full-prefix neighborhood: the parity replay needs every
						 * position the context window can cover (window =
						 * min(2048, base)), not the old 41-row sweep slice -
						 * the oracle skips rounds it cannot cover */
						uint32_t lo = 0u;
						uint32_t count = (uint32_t)(base + 8u - lo);
						uint16_t *host = (uint16_t *)malloc((size_t)count * 5u * H * 2u);
						if ( host != 0 && cudaMemcpy(host,(const uint8_t *)state->dflash_taps_history + (uint64_t)lo * 5u * H * 2u,(size_t)count * 5u * H * 2u,cudaMemcpyDeviceToHost) == cudaSuccess )
						{
							FILE *tf;
							snprintf(path,sizeof(path),"/tmp/ctxrun_%u_taps.bin",parity_runs);
							tf = fopen(path,"wb");
							if ( tf != 0 ) { fwrite(host,1,(size_t)count * 5u * H * 2u,tf); fclose(tf); }
						}
						free(host);
						snprintf(path,sizeof(path),"/tmp/ctxrun_%u.meta",parity_runs);
						{
							FILE *mf = fopen(path,"w");
							if ( mf != 0 )
							{
								uint32_t di;
								fprintf(mf,"run=%u lo=%u base=%llu window=%u anchor=%u drafts=",(unsigned int)parity_runs,(unsigned int)lo,(unsigned long long)base,window,prev);
								for (di=0u; di<B - 1u; di++)
									fprintf(mf,"%u%c",view->draft_token_ids[di],di + 1u < B - 1u ? ' ' : '\n');
								fclose(mf);
							}
						}
					parity_runs++;
				}
			}
		}
		status = SparkStageModuleCudaStatus(SPARK_QWEN38_27B_MODULE_TAG,error,"dflash2_head");
	}
	}
	}
	return(status);
}

static SparkStatus SparkQwen38_27bModuleCaptureDsparkTap(
	SparkQwen38_27bModuleState *state,
	SparkQwen38_27bModuleSlot *slot,
	uint32_t layer,
	uint32_t rows)
{
	uint32_t tap_index;
	cudaError_t error;
	switch ( layer )
	{
	/* Tap layers = the config's target_layer_ids [5,19,33,47,61], i.e. the
	 * OUTPUT of model layers 5/19/33/47/61. The spark0 vLLM reference prints
	 * "auxiliary layers (6,20,34,48,62)" - those are hidden-state LIST
	 * indices (index i+1 = layer i's output), the SAME tensors. Measured on
	 * this build with the tap-store fixed: [5,19,...] round-1 accepts 7/7
	 * (mean 0.37); capturing one layer later [6,20,...] accepts 3/7 (mean
	 * 0.175). The two engines' tap tensors agree; the earlier cross-test
	 * predates the tap-store fix and compared noise. */
	case 5u: tap_index = 0u; break;
	case 19u: tap_index = 1u; break;
	case 33u: tap_index = 2u; break;
	case 47u: tap_index = 3u; break;
	case 61u: tap_index = 4u; break;
	default: return(SPARK_STATUS_OK);
	}
	error = SparkQwen38_27bLaunchDsparkTapStore((cudaStream_t)slot->cuda_stream,slot->hidden_bf16,slot->row_positions,state->dflash_taps_history,rows,tap_index,SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION,SPARK_QWEN38_27B_DSPARK_TARGET_TAP_COUNT);
	return(SparkStageModuleCudaStatus(SPARK_QWEN38_27B_MODULE_TAG,error,"dflash_tap_store"));
}

static SparkStatus SparkQwen38_27bModuleRunFrame(SparkQwen38_27bModuleState *state, SparkQwen38_27bModuleSlot *slot, SparkQwen38_27bResidentDecodeStageFrameContext *context, SparkModelDriverFrame *frame, const SparkQwen38_27bPrefillFrameView *prefill, uint32_t rows)
{
	uint64_t frame_start = state->profile_enabled != 0u ? SparkQwen38_27bProfileNow() : 0ull;
	uint64_t profile_stage_end = frame_start, profile_walk_end = frame_start;
	SparkStatus status = SparkQwen38_27bModuleUploadRows(state,slot,context,frame,prefill,rows);
	slot->replay_frame = (context->flags & SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_RESTORE_FIRST) != 0u ? 1u : 0u;
	slot->verify_frame = (context->flags & SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_SPECULATIVE_VERIFY) != 0u ? 1u : 0u;
	/* verify frames no longer capture the pre-verify snapshot: the per-row
	 * step checkpoints (slot = row) replace it. */
	if ( status == SPARK_STATUS_OK && (context->flags & SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_RESTORE_FIRST) != 0u )
		status = SparkQwen38_27bModuleGdnSnapshot(state,slot,prefill->lane_index,context->gdn_snapshot->snapshot_index,1u);
	if ( status == SPARK_STATUS_OK && (context->flags & SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_RESTORE_VERIFY_ROW) != 0u && state->gdn_snapshot_slot_count >= SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_VERIFY_CHECKPOINT_SLOT_BASE + 8u )
	{
		/* SELECT the verify's row-N checkpoint (the vLLM shape): the step
		 * kernels wrote slots 8+row during the verify walk; this restores
		 * the accepted-prefix state+tail, replacing the re-walk. */
			status = SparkQwen38_27bModuleGdnSnapshot(state,slot,prefill->lane_index,SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_VERIFY_CHECKPOINT_SLOT_BASE + context->gdn_snapshot->snapshot_index,1u);
	}
	/* prefix-cache borrow: the lane resumes from a published GDN state
	 * (the vLLM mamba-radix shape - hybrid models snapshot recurrent state
	 * at block boundaries). Must land BEFORE the first layer touches the
	 * lane's GDN pool storage. */
	if ( status == SPARK_STATUS_OK && (context->flags & SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_PREFIX_RESTORE_IN) != 0u && context->gdn_snapshot != 0 )
		status = SparkQwen38_27bModuleGdnPrefixTransfer(state,slot,prefill->lane_index,context->gdn_snapshot->snapshot_index,1u);
	/* FRAME GRAPH (the launch-gap lever: ~8-10ms/frame no-spec measured).
	 * Eligible = the plain path (no drafter tail - it has the
	 * padding-select sync; no GDN restore - the snapshot slot varies per
	 * round and is a baked kernel arg; no verify). All frame inputs are
	 * device-buffered (row_positions, cold, slot_mapping, context_lengths,
	 * input_token_ids - uploaded eagerly in UploadRows), so replay reads
	 * current contents. Warm run first; capture on second sighting; the
	 * capture round REPLAYS immediately (recording does not execute).
	 * Kill-switch SPARK_QWEN38_27B_FRAME_GRAPH=0; any capture failure sets
	 * graphs_broken and this frame returns an error (the GDN state
	 * mutation makes a direct RERUN unsafe). */
	{
			const uint32_t graph_blocked = (context->flags & (SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DSPARK_DRAFT_AFTER | SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MTP_DRAFT_AFTER | SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_RESTORE_FIRST | SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_RESTORE_VERIFY_ROW | SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_PREFIX_RESTORE_IN | SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_PREFIX_SNAPSHOT_OUT | SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_SPECULATIVE_VERIFY)) != 0u;
		/* OPT-IN (WIP): the capture round still invalidates from an
		 * unchecked call in the layer path (three sync blockers found and
		 * guarded - Finish, profile-head, both syncs now capture-aware -
		 * but a fourth remains; enable with SPARK_QWEN38_27B_FRAME_GRAPH=1
		 * to continue the hunt from the graphs_broken diagnostics) */
		/* DEFAULT ON (2026-08-21): the spec-graph anomaly is resolved - the
		 * FFN TP-reduce capture guard fixed the silent replay corruption.
		 * Verified bit-identical on both paths: spec O512 20.8s / 77 rounds
		 * (was 21.1), no-spec 66.3s (was 67.5). Kill-switch
		 * SPARK_QWEN38_27B_FRAME_GRAPH=0. */
		/* Default ON: measured 78% of frames replay with zero capture
		 * failures; the feared fourth blocker never materialized. The
		 * opt-in era cost a free win. Kill switch: FRAME_GRAPH=0. */
		const char *genv = getenv("SPARK_QWEN38_27B_FRAME_GRAPH");
		int graph_off = genv != 0 && genv[0] == '0';
		int replayed = 0;
		int capturing = 0;
		cudaGraph_t cap = 0;
		if ( graph_off == 0 && state->graphs_broken == 0u && graph_blocked == 0u && status == SPARK_STATUS_OK )
		{
			if ( slot->graph_live != 0u && slot->graph_rows == rows && slot->graph_prefill == (prefill != 0 ? 1u : 0u) )
			{
				if ( slot->graph_exec != 0 )
				{
					if ( cudaGraphLaunch(slot->graph_exec,(cudaStream_t)slot->cuda_stream) == cudaSuccess )
					{
						if ( cudaStreamSynchronize((cudaStream_t)slot->cuda_stream) == cudaSuccess )
						{
							replayed = 1;
							state->graph_frames_replayed++;
						}
						else
							state->graphs_broken = 1u;
					}
					else
						state->graphs_broken = 1u;
				}
				else if ( slot->graph_warm != 0u )
				{
					if ( cudaStreamBeginCapture((cudaStream_t)slot->cuda_stream,cudaStreamCaptureModeRelaxed) == cudaSuccess )
						capturing = 1;
					else
						state->graphs_broken = 1u;
				}
			}
			else
			{
				slot->graph_live = 1u;
				slot->graph_warm = 1u;
				slot->graph_rows = rows;
				slot->graph_prefill = prefill != 0 ? 1u : 0u;
				if ( slot->graph_exec != 0 )
					(void)cudaGraphExecDestroy(slot->graph_exec);
				slot->graph_exec = 0;
			}
		}
		if ( replayed == 0 )
		{
			state->graph_frames_plain++;
			slot->capturing = capturing != 0 ? 1u : 0u;
			if ( state->profile_enabled != 0u )
			{
				profile_stage_end = SparkQwen38_27bProfileNow();
				state->profile_stage_nanos += profile_stage_end - frame_start;
			}
			if ( status == SPARK_STATUS_OK )
				status = SparkQwen38_27bModuleBeginHidden(state,slot,context,rows);
			{
				uint32_t layer;
				for (layer = state->first_layer_index; status == SPARK_STATUS_OK && layer < state->first_layer_index + state->layer_count; layer++)
			{
						status = SparkQwen38_27bModuleRunLayer(state,slot,context->kv_block_table,prefill,layer,rows);
					/* Per-position tap capture on EVERY armed frame (prefill, decode,
					 * verify, replay): the drafter's context KV is built from the target
					 * hidden states at every committed position (upstream
					 * precompute_and_store_context_kv). Rejected draft positions are
					 * overwritten when re-walked, exactly like the main KV. */
					if ( status == SPARK_STATUS_OK && state->dspark_weights.armed != 0u )
						status = SparkQwen38_27bModuleCaptureDsparkTap(state,slot,layer,rows);
					if ( capturing != 0 && getenv("SPARK_QWEN38_27B_CAPTRACE") != 0 )
						fprintf(stderr,"captrace: post-layer %u status=%d\n",layer,(int)status);
				}
			}
			if ( status == SPARK_STATUS_OK )
				status = SparkQwen38_27bModuleFinish(state,slot,context,frame,prefill,rows);
			if ( state->profile_enabled != 0u )
			{
				profile_walk_end = SparkQwen38_27bProfileNow();
				state->profile_walk_nanos += profile_walk_end - profile_stage_end;
			}
			if ( status == SPARK_STATUS_OK && (context->flags & SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_PREFIX_SNAPSHOT_OUT) != 0u && context->gdn_snapshot != 0 )
				status = SparkQwen38_27bModuleGdnPrefixTransfer(state,slot,prefill->lane_index,context->gdn_snapshot->snapshot_index,0u);
			if ( capturing != 0 )
			{
				if ( cudaStreamEndCapture((cudaStream_t)slot->cuda_stream,&cap) == cudaSuccess && cap != 0 && status == SPARK_STATUS_OK )
				{
					cudaGraphExec_t exec = 0;
					if ( cudaGraphInstantiate(&exec,cap,0ull) == cudaSuccess && exec != 0 )
					{
						state->graph_frames_captured++;
						slot->graph_exec = exec;
						/* the capture round did not execute - replay now so this
						 * frame produces its output (the K3 pattern) */
						if ( cudaGraphLaunch(exec,(cudaStream_t)slot->cuda_stream) == cudaSuccess )
							status = SparkStageModuleCudaStatus(SPARK_QWEN38_27B_MODULE_TAG,cudaStreamSynchronize((cudaStream_t)slot->cuda_stream),"graph_sync");
						else
						{
							state->graphs_broken = 1u;
							status = SPARK_STATUS_INTERNAL_ERROR;
						}
					}
					else
					{
						state->graphs_broken = 1u;
						status = SPARK_STATUS_INTERNAL_ERROR;
					}
					if ( cap != 0 )
						(void)cudaGraphDestroy(cap);
				}
				else
				{
					if ( cap != 0 )
						(void)cudaGraphDestroy(cap);
					state->graphs_broken = 1u;
					/* the recorded work did NOT execute and the GDN state mutation
					 * makes a blind rerun unsafe - fail the frame loudly */
					status = SPARK_STATUS_INTERNAL_ERROR;
				}
			}
			slot->capturing = 0u;
		}
	}
	if ( status == SPARK_STATUS_OK && prefill == 0 && state->tap_capture_enabled != 0u )
	{
		state->tap_capture_count++;
		if ( state->tap_capture_count == state->tap_dump_nth || state->tap_dump_nth == 0u )
		{
			char path[128];
			FILE *file;
			uint16_t *taps_host = (uint16_t *)malloc((size_t)SPARK_QWEN38_27B_DSPARK_TARGET_TAP_COUNT * SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION * 2u);
			uint32_t c0;
			snprintf(path,sizeof(path),"/tmp/dflash2_tapdump_%u.bin",state->tap_capture_count);
			if ( taps_host != 0 && cudaMemcpy(taps_host,slot->dspark_tap_buffer,(size_t)SPARK_QWEN38_27B_DSPARK_TARGET_TAP_COUNT * SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION * 2u,cudaMemcpyDeviceToHost) == cudaSuccess )
			{
				file = fopen(path,"wb");
				if ( file != 0 ) { fwrite(taps_host,1,(size_t)SPARK_QWEN38_27B_DSPARK_TARGET_TAP_COUNT * SPARK_QWEN38_27B_MODEL_HIDDEN_DIMENSION * 2u,file); fclose(file); }
			}
			free(taps_host);
			if ( cudaMemcpy(&c0,slot->output_token_ids,sizeof(uint32_t),cudaMemcpyDeviceToHost) == cudaSuccess )
			{
				snprintf(path,sizeof(path),"/tmp/dflash2_tapdump_%u.meta",state->tap_capture_count);
				file = fopen(path,"w");
				if ( file != 0 ) { fprintf(file,"capture=%u c0=%u position=%llu\n",state->tap_capture_count,c0,(unsigned long long)slot->host_row_positions[0]); fclose(file); }
			}
		}
	}
	if ( state->profile_enabled != 0u )
	{
		{
			uint64_t tail_now = SparkQwen38_27bProfileNow();
			state->profile_tail_nanos += tail_now - (profile_walk_end != 0 ? profile_walk_end : profile_stage_end);
		}
		SparkQwen38_27bProfilePrint(state, SparkQwen38_27bProfileNow() - frame_start);
	}
	return(status);
}

SparkStatus SparkQwen38_27bResidentDecodeStageExecute(
    void *module_state,
    SparkModelDriverFrame *frame)
{
    SparkQwen38_27bModuleState *state;
    SparkQwen38_27bResidentDecodeStageFrameContext *context;
    const SparkQwen38_27bPrefillFrameView *prefill;
    const uint32_t *claimed_lane_indices;
    uint32_t prefill_lane_index;
    uint8_t lane_requires_reset[
        SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
    uint8_t lane_used[
        SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
    SparkQwen38_27bModuleSlot *slot;
    uint32_t claimed_lane_count;
    uint32_t slot_index;
    uint32_t rows;
    uint32_t row;
    uint32_t lanes_claimed;
    SparkStatus status;

    state = (SparkQwen38_27bModuleState *)module_state;
    if (state == 0 || frame == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    context = 0;
    status = SparkQwen38_27bModuleValidateFrame(
        state,
        frame,
        (const SparkQwen38_27bResidentDecodeStageFrameContext **)&context);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "%s frame_validate_failed status=%d frame_flags=0x%x buffers=%u tps=%u new_tokens=%u active_slots=%u seq_pos=%" PRIu64 " program_id=%u exp_buffers=%u\n", SPARK_QWEN38_27B_MODULE_TAG, (int)status, frame->flags, frame->buffer_count, frame->tokens_per_sequence, frame->new_token_count, frame->active_slot_count, frame->sequence_position, frame->program_id, state->owns_embedding + state->owns_final_head);
        atomic_fetch_add_explicit(
            &state->rejected_count,
            1u,
            memory_order_relaxed);
        return status;
    }

    prefill = (context->flags &
        SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW) != 0u
        ? context->prefill_frame
        : 0;
    rows = prefill != 0 ? prefill->token_count : context->decode_batch->row_count;
    if (prefill != 0)
    {
        prefill_lane_index = prefill->lane_index;
        claimed_lane_indices = &prefill_lane_index;
        claimed_lane_count = 1u;
    }
    else
    {
        claimed_lane_indices = context->decode_batch->row_lane_indices;
        claimed_lane_count = rows;
    }

    lanes_claimed = 0u;
    status = SparkStageModuleIndexSetClaim(
        state->lane_states,
        state->max_active_sequence_count,
        claimed_lane_indices,
        claimed_lane_count);
    lanes_claimed = status == SPARK_STATUS_OK ? 1u : 0u;
    if (status == SPARK_STATUS_OK)
    {
        status = SparkQwen38_27bModuleValidateLaneSequenceContinuity(
            state,
            context,
            prefill,
            frame->scalar[0], /* the adapter passes the client request_generation */
            lane_requires_reset);
    }
    if (status != SPARK_STATUS_OK)
    {
        if (lanes_claimed != 0u)
        {
            SparkStageModuleIndexSetRelease(
                state->lane_states,
                state->max_active_sequence_count,
                claimed_lane_indices,
                claimed_lane_count);
        }
        atomic_fetch_add_explicit(
            &state->rejected_count,
            1u,
            memory_order_relaxed);
        return status;
    }

    slot_index = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
    status = SparkStageModuleSlotClaim(
        state->slot_states,
        state->pipeline_slot_count,
        &slot_index);
    if (status != SPARK_STATUS_OK)
    {
        SparkStageModuleIndexSetRelease(
            state->lane_states,
            state->max_active_sequence_count,
            claimed_lane_indices,
            claimed_lane_count);
        atomic_fetch_add_explicit(
            &state->rejected_count,
            1u,
            memory_order_relaxed);
        return status;
    }
    slot = &state->slots[slot_index];

    for (row = 0u; status == SPARK_STATUS_OK && row < claimed_lane_count; row++)
    {
        if (lane_requires_reset[row] != 0u)
        {
            state->lane_warm[claimed_lane_indices[row]] = 0u;
            status = SparkQwen38_27bModuleResetLaneState(
                state,
                slot,
                claimed_lane_indices[row]);
        }
    }
    if (status == SPARK_STATUS_OK && prefill != 0)
    {
        status = SparkQwen38_27bModulePrefillStage(state, slot, context);
    }
    else if (status == SPARK_STATUS_OK)
    {
        memset(lane_used, 0, sizeof(lane_used));
        status = SparkQwen38_27bModuleStageRows(state, slot, context, lane_used);
    }
    if (status == SPARK_STATUS_OK &&
        (context->flags &
         SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MTP_DRAFT_AFTER) != 0u)
    {
        status = SparkQwen38_27bModuleStageMtpDraft(
            state,
            slot,
            context,
            prefill,
            rows);
    }
    if (status == SPARK_STATUS_OK)
    {
        atomic_fetch_add_explicit(
            &state->submitted_count,
            1u,
            memory_order_relaxed);
        status = SparkQwen38_27bModuleRunFrame(
            state,
            slot,
            context,
            frame,
            prefill,
            rows);
    }
    if (status == SPARK_STATUS_OK && (context->flags & SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DSPARK_DRAFT_AFTER) != 0u)
        status = SparkQwen38_27bModuleRunDsparkBlockForward(state,slot,context->dspark_draft,
            (state->owns_embedding != 0u && frame->buffer_count > 0u && frame->buffers != 0)
                ? (const uint32_t *)frame->buffers[0].address : 0,
            rows);
    if (status == SPARK_STATUS_OK)
    {
        SparkQwen38_27bModuleCommitLaneSequenceContinuity(
            state,
            context,
            prefill,
            frame->scalar[0]);
        atomic_fetch_add_explicit(
            &state->completed_count,
            1u,
            memory_order_relaxed);
        atomic_fetch_add_explicit(
            &state->tokens_emitted,
            rows,
            memory_order_relaxed);
    }
    else
    {
        SparkQwen38_27bModuleInvalidateLaneSequenceContinuity(
            state,
            context,
            prefill);
        atomic_fetch_add_explicit(
            &state->failed_count,
            1u,
            memory_order_relaxed);
    }

    SparkStageModuleIndexSetRelease(
        state->lane_states,
        state->max_active_sequence_count,
        claimed_lane_indices,
        claimed_lane_count);
    SparkStageModuleSlotRelease(state->slot_states, slot_index);
    return status;
}

static void SparkQwen38_27bAdmissionCost(
    void *context,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision)
{
    SparkQwen38_27bModuleState *state = (SparkQwen38_27bModuleState *)context;
    (void)state;
    decision->host_staging_bytes = (uint64_t)request->new_token_count *
        (sizeof(uint32_t) *
             (uint64_t)(state->owns_embedding + state->owns_final_head + 3u) +
         sizeof(uint64_t));
    decision->device_memcpy_bytes = decision->host_staging_bytes;
}

static SparkStatus SparkQwen38_27bAdmissionKvPredicate(
    void *context,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision)
{
    (void)context;
    if ((request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u &&
        SparkModelDriverRangeFitsWithinCapacity(
            request->sequence_position,
            request->new_token_count,
            SPARK_QWEN38_27B_MODEL_MAXIMUM_CONTEXT_TOKENS) == 0u)
    {
        SparkModelDriverRejectAdmission(
            decision,
            SPARK_MODEL_DRIVER_ADMISSION_REJECTED_KV_CAPACITY,
            decision->available_dispatch_slot_count);
    }
    else
    {
        decision->accepted = 1u;
        decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkQwen38_27bResidentDecodeStageAdmit(
    void *module_state,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision)
{
    SparkQwen38_27bModuleState *state;
    SparkAdmissionPolicyTable table;
    uint32_t available_slot_count;
    SparkStatus status;

    state = (SparkQwen38_27bModuleState *)module_state;
    if (state == 0 || request == 0 || decision == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    available_slot_count = SparkStageModuleSlotCountFree(
        state->slot_states,
        state->pipeline_slot_count);
    memset(&table, 0, sizeof(table));
    table.abi_version = SPARK_ADMISSION_ABI_VERSION;
    table.descriptor_bytes = (uint32_t)sizeof(table);
    table.max_active_sequence_count = state->max_active_sequence_count;
    table.max_input_row_count = state->max_active_sequence_count;
    table.max_sequence_positions = SPARK_QWEN38_27B_MODEL_MAXIMUM_CONTEXT_TOKENS;
    table.flags = SPARK_ADMISSION_POLICY_FLAG_PREFILL_SINGLE_SLOT |
        SPARK_ADMISSION_POLICY_FLAG_DECODE_EQUALS_SLOTS;
    table.predicate = SparkQwen38_27bAdmissionKvPredicate;
    table.predicate_context = state;
    table.cost = SparkQwen38_27bAdmissionCost;
    table.cost_context = state;
    status = SparkAdmissionEvaluateShape(
        &table, available_slot_count, request, decision);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (decision->accepted == 0u)
    {
        atomic_fetch_add_explicit(
            &state->rejected_count,
            1u,
            memory_order_relaxed);
    }
    return status;
}

SparkStatus SparkQwen38_27bResidentDecodeStageSnapshot(
    void *module_state,
    uint32_t program_id,
    SparkModelDriverRuntimeSnapshot *snapshot)
{
    SparkQwen38_27bModuleState *state;

    state = (SparkQwen38_27bModuleState *)module_state;
    if (state == 0 || snapshot == 0 || program_id == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkStageModuleRuntimeSnapshotInitialize(
        snapshot,
        program_id,
        state->slot_states,
        state->pipeline_slot_count);
    snapshot->submitted_count = atomic_load_explicit(
        &state->submitted_count,
        memory_order_relaxed);
    snapshot->completed_count = atomic_load_explicit(
        &state->completed_count,
        memory_order_relaxed);
    snapshot->rejected_count = atomic_load_explicit(
        &state->rejected_count,
        memory_order_relaxed);
    snapshot->kv_token_capacity = (uint64_t)state->kv_block_count * SPARK_QWEN38_27B_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
    return SPARK_STATUS_OK;
}

void SparkQwen38_27bResidentDecodeStageDestroy(void *module_state)
{
    SparkQwen38_27bModuleState *state;
    uint32_t slot_index;

    state = (SparkQwen38_27bModuleState *)module_state;
    if (state == 0)
    {
        return;
    }
    if (SparkStageModuleWaitForSlots(
            SPARK_QWEN38_27B_MODULE_TAG,
            state->slot_states,
            state->pipeline_slot_count,
            SPARK_STAGE_MODULE_DESTROY_QUIESCE_TIMEOUT_NS) != SPARK_STATUS_OK)
    {
        return;
    }
    for (slot_index = 0u; slot_index < state->pipeline_slot_count; ++slot_index)
    {
        if (state->slots[slot_index].cuda_stream != 0)
        {
            cudaStreamDestroy((cudaStream_t)state->slots[slot_index].cuda_stream);
        }
        if (state->slots[slot_index].dspark_logits_host != 0)
        {
            free(state->slots[slot_index].dspark_logits_host);
        }
        if (state->slots[slot_index].dspark_hidden_host != 0)
        {
            free(state->slots[slot_index].dspark_hidden_host);
        }
    }
    SparkQwen38_27bTpDestroy(&state->tp);
    if ( state->tp_stream != 0 )
        cudaStreamDestroy((cudaStream_t)state->tp_stream);
    if (state->dspark_weights.selector_pred_host != 0)
        free(state->dspark_weights.selector_pred_host);
    if (state->dspark_weights.selector_succ_host != 0)
        free(state->dspark_weights.selector_succ_host);
    if (state->dspark_weights.selector_hidden_proj_host != 0)
        free(state->dspark_weights.selector_hidden_proj_host);
    SparkStageKvClientClose(&state->kv_client);
    SparkStageModuleLedgerRelease(&state->ledger);
    free(state);
}

SparkStatus SparkQwen38_27bResidentDecodeStageInitialize(
    const SparkFirmwareModuleConfiguration *configuration,
    const SparkFirmwareModuleHostServices *host_services,
    void **module_state)
{
    SparkQwen38_27bModuleState *state;
    const char *pack_path;
    uint32_t allow_unqualified_execution;
    uint32_t slot_index;
    SparkStatus status;

    pack_path = 0;
    allow_unqualified_execution = 0u;
    status = SparkFirmwareModuleValidateInitialization(
        configuration,
        host_services,
        module_state);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkStageModuleEnvironmentUnsigned(
        SPARK_QWEN38_27B_MODULE_TAG,
        "SPARK_QWEN38_27B_ALLOW_UNQUALIFIED_EXECUTION",
        1u,
        1u,
        &allow_unqualified_execution);
    if (status != SPARK_STATUS_OK || allow_unqualified_execution != 1u)
    {
        return SPARK_STATUS_MODULE_NOT_VALIDATED;
    }

    state = (SparkQwen38_27bModuleState *)calloc(1u, sizeof(*state));
    if (state == 0)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    state->ledger.module_tag = SPARK_QWEN38_27B_MODULE_TAG;
    {
        const char *profile_env = getenv("SPARK_QWEN38_27B_PROFILE");
        state->profile_enabled = profile_env != 0 && strcmp(profile_env, "0") != 0 ? 1u : 0u;
        if ( state->profile_enabled != 0u )
        {
            (void)cudaEventCreate(&state->profile_ev[0]);
            (void)cudaEventCreate(&state->profile_ev[1]);
        }
    }
    {
        /* Diagnostics: capture drafter taps on EVERY decode frame (even
         * without the drafter armed) and dump the Nth capture set, so
         * spec-run taps diff against no-spec taps at the same position.
         * N=0 dumps every capture. */
        const char *capture_env = getenv("SPARK_QWEN38_27B_STAGE_TAP_CAPTURE");
        const char *dump_env = getenv("SPARK_QWEN38_27B_DFLASH2_TAP_DUMP_N");
        const char *sel_env = getenv("SPARK_QWEN38_27B_DFLASH2_STATE_SELECT");
        state->tap_capture_enabled = capture_env != 0 && strcmp(capture_env, "0") != 0 ? 1u : 0u;
        state->dflash2_state_select = sel_env != 0 && strcmp(sel_env, "0") != 0 ? 1u : 0u;
        state->tap_dump_nth = dump_env != 0 ? (uint32_t)strtoul(dump_env,0,0) : 0xFFFFFFFFu;
        state->tap_capture_count = 0u;
    }
    atomic_init(&state->submitted_count, 0u);
    atomic_init(&state->completed_count, 0u);
    atomic_init(&state->rejected_count, 0u);
    atomic_init(&state->failed_count, 0u);
    atomic_init(&state->tokens_emitted, 0u);

    status = SparkQwen38_27bModuleConfigure(state);
    if (status == SPARK_STATUS_OK)
    {
        status = SparkStageModuleCudaStatus(
            SPARK_QWEN38_27B_MODULE_TAG,
            SparkQwen38_27bConfigureCudaKernels(),
            "configure_cuda_kernels");
    }
    if (status == SPARK_STATUS_OK)
    {
        SparkStageModuleAtomicStateArrayInitialize(
            state->slot_states,
            state->pipeline_slot_count);
        SparkStageModuleAtomicStateArrayInitialize(
            state->lane_states,
            state->max_active_sequence_count);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkStageModuleEnvironmentText(
            SPARK_QWEN38_27B_MODULE_TAG,
            "SPARK_QWEN38_27B_STAGE_PACK_PATH",
            &pack_path);
    }
    if (status == SPARK_STATUS_OK)
    {
        SparkQwen38_27bModuleBuildOrdinals(state);
        status = SparkQwen38_27bModuleLoadPack(state, pack_path);
    }
    if (status == SPARK_STATUS_OK)
    {
        /* Optional: the DSpark drafter pack is loaded only when the env var
         * is set (spec-method dspark). getenv (not the required-text helper)
         * so a no-spec / MTP deploy without the var still initializes. */
        const char *dspark_path = getenv("SPARK_QWEN38_27B_DSPARK_PACK_PATH");
        if ( dspark_path != 0 && dspark_path[0] != '\0' )
            status = SparkQwen38_27bModuleLoadDsparkPack(state, dspark_path);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkQwen38_27bModuleInitializeTp(state);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkQwen38_27bModuleAllocatePools(state);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkQwen38_27bModuleBuildHeadShadow(state);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkQwen38_27bModuleOpenKvTier(state);
    }
    for (slot_index = 0u;
         status == SPARK_STATUS_OK && slot_index < state->pipeline_slot_count;
         slot_index++)
    {
        status = SparkQwen38_27bModuleAllocateSlot(state, &state->slots[slot_index]);
    }
    if (status != SPARK_STATUS_OK)
    {
        SparkQwen38_27bResidentDecodeStageDestroy(state);
        return status;
    }

    fprintf(
        stderr,
        "%s ready stage=%u/%u slice=%u+%u tp=%u/%u gdn=%u attn=%u lanes=%u kv_blocks=%u device_gib=%.1f\n",
        SPARK_QWEN38_27B_MODULE_TAG,
        state->stage_index,
        state->stage_count,
        state->first_layer_index,
        state->layer_count,
        state->tp_degree,
        state->tp_rank,
        state->gdn_layer_count,
        state->attn_layer_count,
        state->max_active_sequence_count,
        state->kv_block_count,
        (double)state->ledger.device_bytes_resident /
            (1024.0 * 1024.0 * 1024.0));
    *module_state = state;
    return SPARK_STATUS_OK;
}
