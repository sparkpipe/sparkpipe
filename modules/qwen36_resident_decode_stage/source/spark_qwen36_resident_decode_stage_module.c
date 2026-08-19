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

#include "sparkpipe/spark_qwen36_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_admission.h"
#include "sparkpipe/spark_stage_kv_client.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "spark_qwen36_stagepack_format.h"
#include "spark_qwen36_dspark_format.h"
#include "spark_qwen36_tp.h"

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

#define SPARK_QWEN36_MODULE_TAG "qwen36_stage"
#define SPARK_QWEN36_MODULE_FUSED_QUERY_COMPONENT_COUNT 2u


static inline float SparkQwen36ModuleBf16ToFloat(uint16_t h)
{
	uint32_t u = (uint32_t)h << 16u;
	float f;
	memcpy(&f,&u,sizeof(f));
	return(f);
}

/* fp32 -> bf16 with round-to-nearest-even, matching CUDA __float2bfloat16 and
 * torch's bf16 cast (vLLM truncates the Markov bias and the base+bias sum to
 * bf16 before argmax, so the host sampler must round identically). */
static inline uint16_t SparkQwen36ModuleFloatToBf16(float f)
{
	uint32_t u,lsb;
	memcpy(&u,&f,sizeof(u));
	lsb = (u >> 16u) & 1u;
	u += 0x7FFFu + lsb;
	return((uint16_t)(u >> 16u));
}
#define SPARK_QWEN36_MODULE_STAGED_ROW_CAPACITY (SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT + SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS)

typedef struct SparkQwen36ModuleSlot
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
	uint16_t *dspark_logits_host;
	uint16_t *dspark_conv_delta;
	void *dspark_conv_out;
	uint32_t *dspark_mask_token_ids;
	uint64_t *dspark_selector_chunk_keys;
	uint32_t *dspark_selector_candidate_ids;
	float *dspark_selector_unary;
	uint16_t *dspark_selector_gate;
	float *dspark_selector_edges;
	uint32_t *dspark_selector_slots;
	uint32_t mtp_seed_row;
	uint32_t host_row_cold[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t host_slot_mapping[SPARK_QWEN36_MODULE_STAGED_ROW_CAPACITY];
	uint32_t host_context_lengths[SPARK_QWEN36_MODULE_STAGED_ROW_CAPACITY];
	uint32_t host_row_lane_indices[SPARK_QWEN36_MODULE_STAGED_ROW_CAPACITY];
	uint64_t host_row_positions[SPARK_QWEN36_MODULE_STAGED_ROW_CAPACITY];
} SparkQwen36ModuleSlot;

typedef struct SparkQwen36DsparkLayerWeights
{
	SparkQwen36LinearView q;
	SparkQwen36LinearView k;
	SparkQwen36LinearView v;
	SparkQwen36LinearView o;
	const void *q_norm_bf16;
	const void *k_norm_bf16;
	const void *input_norm_bf16;
	const void *post_norm_bf16;
	SparkQwen36LinearView gate;
	SparkQwen36LinearView up;
	SparkQwen36LinearView down;
	const void *conv_attn_base;
	SparkQwen36LinearView conv_attn_proj;
	const void *conv_mlp_base;
	SparkQwen36LinearView conv_mlp_proj;
} SparkQwen36DsparkLayerWeights;

typedef struct SparkQwen36DsparkWeights
{
	SparkQwen36DsparkLayerWeights layer[SPARK_QWEN36_DSPARK_LAYER_COUNT];
	SparkQwen36LinearView projector;
	SparkQwen36LinearView markov_w1;
	SparkQwen36LinearView markov_w2;
	SparkQwen36LinearView confidence;
	const void *final_norm_bf16;
	const void *hidden_norm_bf16;
	uint16_t *markov_w1_host;
	uint16_t *markov_w2_host;
	uint32_t armed;
} SparkQwen36DsparkWeights;

typedef struct SparkQwen36ModuleState
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
	uint32_t gdn_ordinal_by_layer[SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	uint32_t attn_ordinal_by_layer[SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	uint32_t layer_seen_bits[SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	uint32_t global_seen_bits;
	uint32_t mtp_seen_bits;
	SparkQwen36MtpWeights mtp;
	const void *token_embedding_bf16;
	const void *final_norm_weight_bf16;
	const void *lm_head_weight_bf16;
	uint8_t *head_shadow_payload;
	uint8_t *head_shadow_scale;
	float *head_error_norm_f32;
	const void *attention_norm_by_layer[SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	const void *mlp_norm_by_layer[SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	SparkQwen36GdnLayerWeights gdn_by_layer[SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	SparkQwen36AttnLayerWeights attn_by_layer[SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	SparkQwen36FfnLayerWeights ffn_by_layer[SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	SparkQwen36GdnStatePool gdn_pool;
	void *kv_cache_bf16;
	uint64_t cache_layer_stride;
	uint64_t cache_block_stride;
	uint8_t lane_warm[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_sequence_ids[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_next_positions[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	SparkQwen36ModuleSlot slots[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	atomic_uint slot_states[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	atomic_uint lane_states[SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	SparkStageKvClient kv_client;
	SparkQwen36TpState tp;
	SparkQwen36DsparkWeights dspark_weights;
	void *tp_stream;
	atomic_ullong submitted_count;
	atomic_ullong completed_count;
	atomic_ullong rejected_count;
	atomic_ullong failed_count;
	atomic_ullong tokens_emitted;
	/* decode-frame GPU phase profiling (SPARK_QWEN36_PROFILE=1). The host
	 * blocks inside every TP reduce spin, which drains the queued GPU work, so
	 * the spin durations measure the GPU execution of the phase between two
	 * reduces: GDN branch, ATTN branch, FFN, and the head tail. */
	uint32_t profile_enabled;
	uint64_t profile_gdn_spin_nanos;
	uint64_t profile_attn_spin_nanos;
	uint64_t profile_ffn_spin_nanos;
	uint64_t profile_head_spin_nanos;
	uint64_t profile_frame_nanos;
	uint32_t profile_frame_count;
} SparkQwen36ModuleState;

static uint64_t SparkQwen36ProfileNow(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ((uint64_t)ts.tv_sec * 1000000000ull) + (uint64_t)ts.tv_nsec;
}

static void SparkQwen36ProfilePrint(SparkQwen36ModuleState *state, uint64_t frame_nanos)
{
	state->profile_frame_nanos += frame_nanos;
	state->profile_frame_count++;
	if ( (state->profile_frame_count & 7u) == 0u || state->profile_frame_count == 1u )
		fprintf(stderr, "%s gpu_spin_profile frames=%u frame_ms=%.2f gdn_ms=%.2f attn_ms=%.2f ffn_ms=%.2f head_ms=%.2f\n",
			SPARK_QWEN36_MODULE_TAG, state->profile_frame_count,
			(double)state->profile_frame_nanos / 1000000.0,
			(double)state->profile_gdn_spin_nanos / 1000000.0,
			(double)state->profile_attn_spin_nanos / 1000000.0,
			(double)state->profile_ffn_spin_nanos / 1000000.0,
			(double)state->profile_head_spin_nanos / 1000000.0);
}

extern cudaError_t SparkQwen36ConfigureCudaKernels(void);
extern cudaError_t SparkQwen36LaunchRmsNorm(cudaStream_t stream, const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon);
extern cudaError_t SparkQwen36LaunchFusedResidualRmsNorm(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon);
extern cudaError_t SparkQwen36LaunchLinear(cudaStream_t stream, const SparkQwen36LinearView *view, const void *input_bf16, void *output_bf16, uint32_t row_count);
extern cudaError_t SparkQwen36LaunchDsparkAttn(cudaStream_t stream, const void *q_bf16, const void *k_bf16, const void *v_bf16, const void *q_norm_bf16, const void *k_norm_bf16, void *attn_out_bf16, uint32_t block_size, uint64_t base_position);
extern cudaError_t SparkQwen36LaunchDsparkMarkov(cudaStream_t stream, const void *markov_w1_bf16, const void *markov_w2_bf16, const uint32_t *prev_token_ids, uint32_t draft_count, uint32_t rank, void *bias_out, uint32_t vocab);
extern cudaError_t SparkQwen36LaunchDsparkConv(cudaStream_t stream, const void *x_bf16, const void *delta_f32, const void *base_bf16, void *out_bf16, uint32_t block_size, uint32_t num_groups, uint32_t group_size, uint32_t side);
extern uint64_t SparkQwen36DsparkHeadTopKChunkKeyCount(uint32_t row_count, uint32_t top_k);
extern cudaError_t SparkQwen36LaunchDsparkHeadTopK(cudaStream_t stream, const SparkQwen36LinearView *head, const void *hidden_bf16, uint64_t *chunk_keys, uint32_t *top_candidate_ids, float *top_scores_f32, void *top_scores_bf16, uint32_t row_count, uint32_t candidate_offset, uint32_t top_k);
extern cudaError_t SparkQwen36LaunchDsparkSelector(cudaStream_t stream, const void *hidden_bf16, const void *hidden_projection_bf16, const void *predecessor_bf16, const void *successor_bf16, const uint32_t *candidate_ids, const uint32_t *anchor_token_ids, const float *unary_f32, void *context_gate_bf16, float *edges_f32, uint32_t *draft_token_ids, uint32_t *draft_candidate_slots, uint32_t batch_count, uint32_t slot_count, uint32_t top_k, uint32_t rank, uint32_t hidden_dimension);
/* Small-batch GEMM geometry, mirrors the cuda translation unit. */
#define SPARK_QWEN36_SMALL_BATCH_MAX_ROWS 8u
#define SPARK_QWEN36_SMALL_BATCH_TILE_N 64u
#define SPARK_QWEN36_SMALL_BATCH_K_CHUNK 128u
extern cudaError_t SparkQwen36LaunchFfnGateUp(cudaStream_t stream, const void *gate_weight_bf16, const void *up_weight_bf16, const void *input_bf16, void *gated_up_bf16, uint32_t row_count, uint32_t input_dimension, uint32_t output_dimension);
extern cudaError_t SparkQwen36LaunchEmbeddingGather(cudaStream_t stream, const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count);
extern cudaError_t SparkQwen36LaunchConvUpdate(cudaStream_t stream, const void *qkv_bf16, const SparkQwen36GdnLayerWeights *weights, void *conv_out_bf16, const SparkQwen36GdnStatePool *pool, const uint32_t *row_lane_indices, uint32_t row_count, uint32_t gdn_layer_ordinal);
extern cudaError_t SparkQwen36LaunchDecayBeta(cudaStream_t stream, const void *decay_pre_bf16, const void *beta_pre_bf16, const SparkQwen36GdnLayerWeights *weights, float *log_decay_f32, float *beta_f32, uint32_t row_count);
extern cudaError_t SparkQwen36LaunchGdnStep(cudaStream_t stream, const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, const SparkQwen36GdnStatePool *pool, void *core_out_bf16, const uint32_t *row_lane_indices, uint32_t row_count, uint32_t gdn_layer_ordinal);
extern cudaError_t SparkQwen36LaunchGatedNorm(cudaStream_t stream, const void *core_bf16, const void *z_bf16, const SparkQwen36GdnLayerWeights *weights, void *output_bf16, uint32_t row_count, float epsilon);
extern cudaError_t SparkQwen36LaunchAttnPrepare(cudaStream_t stream, void *q_fused_bf16, const void *k_bf16, const void *v_bf16, const SparkQwen36AttnLayerWeights *weights, void *kv_cache_bf16, const uint32_t *slot_mapping, const uint64_t *row_positions, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, float epsilon);
extern cudaError_t SparkQwen36LaunchAttnDecode(cudaStream_t stream, const void *q_fused_bf16, const void *kv_cache_bf16, const SparkQwen36KvBlockTableView *table, const uint32_t *row_lane_indices, const uint32_t *context_lengths, void *head_out_bf16, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride);
extern cudaError_t SparkQwen36LaunchChunkConv(cudaStream_t stream, const void *qkv_bf16, const SparkQwen36GdnLayerWeights *weights, void *conv_out_bf16, const SparkQwen36GdnStatePool *pool, uint32_t lane_index, uint32_t token_count, uint32_t gdn_layer_ordinal);
extern cudaError_t SparkQwen36LaunchGdnChunk(cudaStream_t stream, const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, float *workspace_qn, float *workspace_kn, float *workspace_cum_g, float *workspace_decay, float *workspace_attn, float *workspace_w, float *workspace_kg, const SparkQwen36GdnStatePool *pool, void *core_out_bf16, uint32_t lane_index, uint32_t token_count, uint32_t gdn_layer_ordinal);
extern cudaError_t SparkQwen36LaunchResidualAdd(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, uint32_t row_count, uint32_t dimension);
extern cudaError_t SparkQwen36LaunchSwiGlu(cudaStream_t stream, const void *gate_bf16, void *up_bf16, uint32_t row_count, uint32_t dimension);
extern cudaError_t SparkQwen36LaunchHeadArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint32_t *token_ids, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count);
extern cudaError_t SparkQwen36LaunchHeadShadowQuantize(cudaStream_t stream, const void *head_bf16, uint8_t *shadow_payload, uint8_t *shadow_scale, float *error_norm, uint32_t candidate_count, uint32_t hidden_dimension);
extern cudaError_t SparkQwen36LaunchHeadScreenedArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint8_t *shadow_payload, const uint8_t *shadow_scale, const float *error_norm, void *logits_bf16, uint32_t *candidate_ids, uint32_t *candidate_counts, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count);
extern cudaError_t SparkQwen36LaunchHeadScreenedArgmaxScore(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint8_t *shadow_payload, const uint8_t *shadow_scale, const float *error_norm, void *scratch_bf16, uint32_t *candidate_ids, uint32_t *candidate_counts, uint32_t *output_token_ids, float *output_scores, uint32_t candidate_offset, uint32_t row_count, uint32_t candidate_count);
extern cudaError_t SparkQwen36LaunchHeadMaxLocPack(cudaStream_t stream, const float *scores_f32, const uint32_t *token_ids_u32, uint64_t *keys_u64, uint32_t row_count);
extern cudaError_t SparkQwen36LaunchHeadMaxLocUnpack(cudaStream_t stream, const uint64_t *keys_u64, uint32_t *token_ids_u32, uint32_t row_count);
extern cudaError_t SparkQwen36TpSetGeometry(uint32_t gdn_qk_channels,uint32_t gdn_value_channels,uint32_t gdn_conv_channels,uint32_t gdn_key_heads,uint32_t gdn_value_heads,uint32_t attn_query_heads,uint32_t attn_kv_heads,uint32_t gdn_qk_channel_base,uint32_t gdn_value_channel_base,uint32_t gdn_key_head_base,uint32_t gdn_value_head_base);

static SparkStatus SparkQwen36ModuleConfigure(SparkQwen36ModuleState *state)
{
	SparkStatus status;
	status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN36_MODULE_TAG,"SPARK_QWEN36_STAGE_COUNT",1u,SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT,&state->stage_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN36_MODULE_TAG,"SPARK_QWEN36_STAGE_INDEX",0u,SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT - 1u,&state->stage_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN36_MODULE_TAG,"SPARK_QWEN36_STAGE_FIRST_LAYER",0u,SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT - 1u,&state->first_layer_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN36_MODULE_TAG,"SPARK_QWEN36_STAGE_LAYER_COUNT",1u,SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT,&state->layer_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN36_MODULE_TAG,"SPARK_QWEN36_TP_DEGREE",1u,16u,&state->tp_degree);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN36_MODULE_TAG,"SPARK_QWEN36_TP_RANK",0u,15u,&state->tp_rank);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN36_MODULE_TAG,"SPARK_QWEN36_STAGE_MAX_ACTIVE_SEQUENCES",1u,SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,&state->max_active_sequence_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN36_MODULE_TAG,"SPARK_QWEN36_STAGE_PIPELINE_SLOTS",1u,SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,&state->pipeline_slot_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN36_MODULE_TAG,"SPARK_QWEN36_STAGE_KV_BLOCKS",1u,1u << 20u,&state->kv_block_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN36_MODULE_TAG,"SPARK_QWEN36_STAGE_MTP",0u,1u,&state->mtp_armed);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN36_MODULE_TAG,"SPARK_QWEN36_STAGE_GDN_SNAPSHOT_SLOTS",0u,SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_GDN_SNAPSHOT_SLOTS,&state->gdn_snapshot_slot_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( state->stage_index >= state->stage_count || state->first_layer_index + state->layer_count > SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT )
	{
		fprintf(stderr,"%s config_slice_invalid stage=%u/%u slice=%u+%u\n",SPARK_QWEN36_MODULE_TAG,state->stage_index,state->stage_count,state->first_layer_index,state->layer_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	/* MTP is TP-safe: the pack slices the MTP decoder's attention/FFN
	 * exactly like main layers, the fc and norms replicate, the decoder
	 * pass reuses the reduced RunAttnLayer/RunFfn, and the draft argmax
	 * reduces the sharded head with a u64 maxloc. */
	if ( state->tp_rank >= state->tp_degree ||
		(state->tp_degree > 1u && (state->stage_count != 1u || state->stage_index != 0u || state->first_layer_index != 0u || state->layer_count != SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT)) )
	{
		fprintf(stderr,"%s config_tp_invalid degree=%u rank=%u stage=%u/%u slice=%u+%u\n",SPARK_QWEN36_MODULE_TAG,state->tp_degree,state->tp_rank,state->stage_index,state->stage_count,state->first_layer_index,state->layer_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	state->owns_embedding = state->first_layer_index == 0u ? 1u : 0u;
	state->owns_final_head = state->first_layer_index + state->layer_count == SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT ? 1u : 0u;
	if ( (state->stage_index == 0u) != (state->owns_embedding != 0u) || (state->stage_index + 1u == state->stage_count) != (state->owns_final_head != 0u) )
	{
		fprintf(stderr,"%s config_position_mismatch stage=%u/%u slice=%u+%u\n",SPARK_QWEN36_MODULE_TAG,state->stage_index,state->stage_count,state->first_layer_index,state->layer_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( state->mtp_armed != 0u && state->owns_final_head == 0u )
	{
		fprintf(stderr,"%s config_mtp_without_head stage=%u/%u\n",SPARK_QWEN36_MODULE_TAG,state->stage_index,state->stage_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	return(SPARK_STATUS_OK);
}

static void SparkQwen36ModuleBuildOrdinals(SparkQwen36ModuleState *state)
{
	uint32_t layer;
	for (layer = 0; layer < SPARK_QWEN36_RESIDENT_DECODE_STAGE_LAYER_COUNT; layer++)
	{
		state->gdn_ordinal_by_layer[layer] = UINT32_MAX;
		state->attn_ordinal_by_layer[layer] = UINT32_MAX;
	}
	for (layer = state->first_layer_index; layer < state->first_layer_index + state->layer_count; layer++)
	{
		if ( SPARK_QWEN36_MODEL_LAYER_IS_GDN(layer) != 0u )
			state->gdn_ordinal_by_layer[layer] = state->gdn_layer_count++;
		else
			state->attn_ordinal_by_layer[layer] = state->attn_layer_count++;
	}
}

static void SparkQwen36ModuleFillLinearView(SparkQwen36LinearView *view, const SparkQwen36StagePackEntry *entry, void *payload, void *scale)
{
	view->abi_version = SPARK_QWEN36_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION;
	view->weight_format = entry->weight_format;
	view->input_dimension = entry->columns;
	view->output_dimension = entry->rows;
	view->weight_payload = payload;
	view->weight_scale_e8m0 = (const uint8_t *)scale;
	view->weight_payload_bytes = entry->payload_bytes;
	view->weight_scale_bytes = entry->scale_bytes;
}

static SparkStatus SparkQwen36ModuleValidateEntry(SparkQwen36ModuleState *state, const SparkQwen36StagePackEntry *entry, uint64_t file_bytes, uint32_t *is_global)
{
	SparkQwen36StagePackTensorShape shape;
	uint32_t global = entry->layer_index == SPARK_QWEN36_STAGEPACK_GLOBAL_LAYER ? 1u : 0u;
	memset(&shape, 0, sizeof(shape));
	if ( SparkQwen36StagePackResolvedShape(entry->tensor_kind,global != 0u ? 0u : entry->layer_index,global,state->tp_degree,&shape) != 0 || entry->rows != shape.rows || entry->columns != shape.columns )
	{
		fprintf(stderr,"%s dbg_shape kind=%u layer=%u rows=%u/%u cols=%u/%u\n",SPARK_QWEN36_MODULE_TAG,entry->tensor_kind,entry->layer_index,entry->rows,shape.rows,entry->columns,shape.columns);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	if ( entry->weight_format == SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16_RANS )
	{
		if ( shape.quantizable == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
		if ( (entry->rows % 64u) != 0u || (entry->columns % 128u) != 0u || entry->scale_bytes != 0u || entry->scale_group_size != 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
	}
	else if ( shape.quantizable != 0u )
	{
		if ( entry->weight_format != SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 && entry->weight_format != SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 && entry->weight_format != SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 )
			return(SPARK_STATUS_VALIDATION_FAILED);
	}
	else if ( entry->weight_format != shape.natural_format )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->weight_format == SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 ? entry->scale_group_size != 32u : (entry->weight_format == SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 ? entry->scale_group_size != 128u : entry->scale_group_size != 0u) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->weight_format != SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16_RANS && (entry->payload_bytes != SparkQwen36StagePackPayloadBytes(entry->weight_format,entry->rows,entry->columns) || entry->scale_bytes != SparkQwen36StagePackScaleBytes(entry->weight_format,entry->rows,entry->columns)) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->payload_offset > file_bytes || entry->payload_bytes > file_bytes - entry->payload_offset )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->scale_bytes != 0u && (entry->scale_offset > file_bytes || entry->scale_bytes > file_bytes - entry->scale_offset) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->layer_index == SPARK_QWEN36_STAGEPACK_MTP_LAYER || (global != 0u && (entry->tensor_kind >= SPARK_QWEN36_STAGEPACK_TENSOR_MTP_FC && entry->tensor_kind <= SPARK_QWEN36_STAGEPACK_TENSOR_MTP_FINAL_NORM)) )
	{
		if ( state->owns_final_head == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
	}
	else if ( global == 0u && (entry->layer_index < state->first_layer_index || entry->layer_index >= state->first_layer_index + state->layer_count) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	*is_global = global;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen36ModuleBindMtp(SparkQwen36ModuleState *state, const SparkQwen36StagePackEntry *entry, void *payload, void *scale)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_QWEN36_STAGEPACK_TENSOR_MTP_FC: SparkQwen36ModuleFillLinearView(&state->mtp.fc,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTENTION_NORM: state->mtp.attention_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_MLP_NORM: state->mtp.mlp_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_FFN_GATE: SparkQwen36ModuleFillLinearView(&state->mtp.ffn.gate,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_FFN_UP: SparkQwen36ModuleFillLinearView(&state->mtp.ffn.up,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_FFN_DOWN: SparkQwen36ModuleFillLinearView(&state->mtp.ffn.down,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_QUERY: SparkQwen36ModuleFillLinearView(&state->mtp.attention.query,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_KEY: SparkQwen36ModuleFillLinearView(&state->mtp.attention.key,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_VALUE: SparkQwen36ModuleFillLinearView(&state->mtp.attention.value,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_OUTPUT: SparkQwen36ModuleFillLinearView(&state->mtp.attention.output,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_QUERY_NORM: state->mtp.attention.query_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_KEY_NORM: state->mtp.attention.key_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
}

static SparkStatus SparkQwen36ModuleBindGlobal(SparkQwen36ModuleState *state, const SparkQwen36StagePackEntry *entry, void *payload)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_QWEN36_STAGEPACK_TENSOR_EMBEDDING:
		if ( state->owns_embedding == 0u && state->owns_final_head == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
		state->token_embedding_bf16 = payload;
		return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_FINAL_NORM:
		if ( state->owns_final_head == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
		state->final_norm_weight_bf16 = payload;
		return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_LM_HEAD:
		if ( state->owns_final_head == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
		state->lm_head_weight_bf16 = payload;
		return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_MTP_EMBED_NORM: state->mtp.embed_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_MTP_HIDDEN_NORM: state->mtp.hidden_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_MTP_FINAL_NORM: state->mtp.final_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
}

static SparkStatus SparkQwen36ModuleBindLayer(SparkQwen36ModuleState *state, const SparkQwen36StagePackEntry *entry, void *payload, void *scale)
{
	uint32_t layer = entry->layer_index;
	switch ( entry->tensor_kind )
	{
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTENTION_NORM: state->attention_norm_by_layer[layer] = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_MLP_NORM: state->mlp_norm_by_layer[layer] = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_FFN_GATE: SparkQwen36ModuleFillLinearView(&state->ffn_by_layer[layer].gate,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_FFN_UP: SparkQwen36ModuleFillLinearView(&state->ffn_by_layer[layer].up,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_FFN_DOWN: SparkQwen36ModuleFillLinearView(&state->ffn_by_layer[layer].down,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_GDN_QKV: SparkQwen36ModuleFillLinearView(&state->gdn_by_layer[layer].qkv,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_GDN_GATE: SparkQwen36ModuleFillLinearView(&state->gdn_by_layer[layer].gate,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_GDN_BETA: SparkQwen36ModuleFillLinearView(&state->gdn_by_layer[layer].beta,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_GDN_DECAY: SparkQwen36ModuleFillLinearView(&state->gdn_by_layer[layer].decay,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_GDN_OUTPUT: SparkQwen36ModuleFillLinearView(&state->gdn_by_layer[layer].output,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_GDN_CONV_WEIGHT: state->gdn_by_layer[layer].conv_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_GDN_A_LOG: state->gdn_by_layer[layer].a_log_f32 = (const float *)payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_GDN_DT_BIAS: state->gdn_by_layer[layer].dt_bias_f32 = (const float *)payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_GDN_NORM: state->gdn_by_layer[layer].gdn_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_QUERY: SparkQwen36ModuleFillLinearView(&state->attn_by_layer[layer].query,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_KEY: SparkQwen36ModuleFillLinearView(&state->attn_by_layer[layer].key,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_VALUE: SparkQwen36ModuleFillLinearView(&state->attn_by_layer[layer].value,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_OUTPUT: SparkQwen36ModuleFillLinearView(&state->attn_by_layer[layer].output,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_QUERY_NORM: state->attn_by_layer[layer].query_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_KEY_NORM: state->attn_by_layer[layer].key_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
}

static SparkStatus SparkQwen36ModuleLoadEntry(SparkQwen36ModuleState *state, FILE *file, const SparkQwen36StagePackEntry *entry, uint64_t file_bytes)
{
	SparkStatus status;
	uint32_t is_global = 0u,bit = 1u << entry->tensor_kind;
	uint32_t *seen;
	void *payload = 0,*scale = 0;
	status = SparkQwen36ModuleValidateEntry(state,entry,file_bytes,&is_global);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s pack_entry_invalid kind=%u layer=%u\n",SPARK_QWEN36_MODULE_TAG,entry->tensor_kind,entry->layer_index);
		return(status);
	}
	if ( entry->layer_index == SPARK_QWEN36_STAGEPACK_MTP_LAYER || (is_global != 0u && entry->tensor_kind >= SPARK_QWEN36_STAGEPACK_TENSOR_MTP_FC && entry->tensor_kind <= SPARK_QWEN36_STAGEPACK_TENSOR_MTP_FINAL_NORM) )
		seen = &state->mtp_seen_bits;
	else
		seen = is_global != 0u ? &state->global_seen_bits : &state->layer_seen_bits[entry->layer_index];
	if ( (*seen & bit) != 0u )
	{
		fprintf(stderr,"%s pack_entry_duplicate kind=%u layer=%u\n",SPARK_QWEN36_MODULE_TAG,entry->tensor_kind,entry->layer_index);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	*seen |= bit;
	status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,entry->payload_offset,entry->payload_bytes,&payload);
	if ( status == SPARK_STATUS_OK && entry->scale_bytes != 0u )
		status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,entry->scale_offset,entry->scale_bytes,&scale);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( entry->layer_index == SPARK_QWEN36_STAGEPACK_MTP_LAYER || entry->tensor_kind == SPARK_QWEN36_STAGEPACK_TENSOR_MTP_FC )
		return(SparkQwen36ModuleBindMtp(state,entry,payload,scale));
	return(is_global != 0u ? SparkQwen36ModuleBindGlobal(state,entry,payload) : SparkQwen36ModuleBindLayer(state,entry,payload,scale));
}

static SparkStatus SparkQwen36ModuleVerifyCoverage(SparkQwen36ModuleState *state)
{
	uint32_t layer,expected_global = 0u,expected_layer;
	if ( state->owns_embedding != 0u || state->owns_final_head != 0u )
		expected_global |= 1u << SPARK_QWEN36_STAGEPACK_TENSOR_EMBEDDING;
	if ( state->owns_final_head != 0u )
		expected_global |= (1u << SPARK_QWEN36_STAGEPACK_TENSOR_FINAL_NORM) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_LM_HEAD);
	if ( state->owns_final_head != 0u )
	{
		uint32_t expected_mtp = (1u << SPARK_QWEN36_STAGEPACK_TENSOR_MTP_FC) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_MTP_EMBED_NORM) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_MTP_HIDDEN_NORM) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_MTP_FINAL_NORM) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_ATTENTION_NORM) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_MLP_NORM) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_FFN_GATE) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_FFN_UP) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_FFN_DOWN) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_QUERY) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_KEY) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_VALUE) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_OUTPUT) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_QUERY_NORM) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_KEY_NORM);
		if ( state->mtp_seen_bits != expected_mtp )
		{
			fprintf(stderr,"%s pack_mtp_incomplete seen=%08x expected=%08x\n",SPARK_QWEN36_MODULE_TAG,state->mtp_seen_bits,expected_mtp);
			return(SPARK_STATUS_VALIDATION_FAILED);
		}
	}
	if ( state->global_seen_bits != expected_global )
	{
		fprintf(stderr,"%s pack_globals_incomplete seen=%08x expected=%08x\n",SPARK_QWEN36_MODULE_TAG,state->global_seen_bits,expected_global);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	for (layer = state->first_layer_index; layer < state->first_layer_index + state->layer_count; layer++)
	{
		expected_layer = (1u << SPARK_QWEN36_STAGEPACK_TENSOR_ATTENTION_NORM) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_MLP_NORM) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_FFN_GATE) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_FFN_UP) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_FFN_DOWN);
		if ( SPARK_QWEN36_MODEL_LAYER_IS_GDN(layer) != 0u )
			expected_layer |= (1u << SPARK_QWEN36_STAGEPACK_TENSOR_GDN_QKV) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_GDN_GATE) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_GDN_BETA) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_GDN_DECAY) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_GDN_OUTPUT) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_GDN_CONV_WEIGHT) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_GDN_A_LOG) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_GDN_DT_BIAS) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_GDN_NORM);
		else
			expected_layer |= (1u << SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_QUERY) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_KEY) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_VALUE) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_OUTPUT) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_QUERY_NORM) | (1u << SPARK_QWEN36_STAGEPACK_TENSOR_ATTN_KEY_NORM);
		if ( state->layer_seen_bits[layer] != expected_layer )
		{
			fprintf(stderr,"%s pack_layer_incomplete layer=%u seen=%08x expected=%08x\n",SPARK_QWEN36_MODULE_TAG,layer,state->layer_seen_bits[layer],expected_layer);
			return(SPARK_STATUS_VALIDATION_FAILED);
		}
	}
	return(SPARK_STATUS_OK);
}

/* Resolve one DSpark drafter pack entry into the dspark_weights struct. */
static SparkStatus SparkQwen36ModuleLoadDsparkEntry(
	SparkQwen36ModuleState *state,
	const SparkQwen36StagePackEntry *entry,
	void *payload,
	void *scale)
{
	SparkQwen36DsparkWeights *w = &state->dspark_weights;
	uint32_t layer = entry->layer_index;
	/* Global tensors (projector/markov/final-norm/hidden-norm) carry the
	 * 0xFFFFFFFF layer sentinel; the confidence bias rides at 0xFFFFFFFE.
	 * Both resolve to w->... not w->layer[...]. */
	if ( layer >= SPARK_QWEN36_DSPARK_LAYER_COUNT && layer != 0xFFFFFFFFu && layer != 0xFFFFFFFEu )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	SparkQwen36DsparkLayerWeights *lw = layer < SPARK_QWEN36_DSPARK_LAYER_COUNT ? &w->layer[layer] : 0;
	switch ( entry->tensor_kind )
	{
	case SPARK_QWEN36_DSPARK_TENSOR_ATTN_QUERY: SparkQwen36ModuleFillLinearView(&lw->q,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_DSPARK_TENSOR_ATTN_KEY: SparkQwen36ModuleFillLinearView(&lw->k,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_DSPARK_TENSOR_ATTN_VALUE: SparkQwen36ModuleFillLinearView(&lw->v,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_DSPARK_TENSOR_ATTN_OUTPUT: SparkQwen36ModuleFillLinearView(&lw->o,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_DSPARK_TENSOR_ATTN_QUERY_NORM: lw->q_norm_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_DSPARK_TENSOR_ATTN_KEY_NORM: lw->k_norm_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_DSPARK_TENSOR_ATTENTION_NORM: lw->input_norm_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_DSPARK_TENSOR_MLP_NORM: lw->post_norm_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_DSPARK_TENSOR_FFN_GATE: SparkQwen36ModuleFillLinearView(&lw->gate,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_DSPARK_TENSOR_FFN_UP: SparkQwen36ModuleFillLinearView(&lw->up,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_DSPARK_TENSOR_FFN_DOWN: SparkQwen36ModuleFillLinearView(&lw->down,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_DSPARK_TENSOR_CONV_ATTN_BASE: lw->conv_attn_base = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_DSPARK_TENSOR_CONV_ATTN_PROJ: SparkQwen36ModuleFillLinearView(&lw->conv_attn_proj,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_DSPARK_TENSOR_CONV_MLP_BASE: lw->conv_mlp_base = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_DSPARK_TENSOR_CONV_MLP_PROJ: SparkQwen36ModuleFillLinearView(&lw->conv_mlp_proj,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_DSPARK_TENSOR_PROJECTOR: SparkQwen36ModuleFillLinearView(&w->projector,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_DSPARK_TENSOR_SELECTOR_PRED: SparkQwen36ModuleFillLinearView(&w->markov_w1,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_DSPARK_TENSOR_SELECTOR_SUCC: SparkQwen36ModuleFillLinearView(&w->markov_w2,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_DSPARK_TENSOR_SELECTOR_HIDDEN_PROJ: SparkQwen36ModuleFillLinearView(&w->confidence,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN36_DSPARK_TENSOR_FINAL_NORM: w->final_norm_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN36_DSPARK_TENSOR_HIDDEN_NORM: w->hidden_norm_bf16 = payload; return(SPARK_STATUS_OK);
	default: return(SPARK_STATUS_INVALID_ARGUMENT);
	}
}

/* Load the separate DSpark drafter pack (optional, spec-method dspark only). */
static SparkStatus SparkQwen36ModuleLoadDsparkPack(SparkQwen36ModuleState *state, const char *path)
{
	SparkQwen36StagePackHeader header;
	SparkQwen36StagePackEntry *directory;
	FILE *file;
	SparkStatus status;
	uint32_t index;
	if ( path == 0 || path[0] == '\0' )
		return(SPARK_STATUS_OK);
	file = fopen(path,"rb");
	if ( file == 0 )
	{
		fprintf(stderr,"%s dspark_pack_open_failed path=%s\n",SPARK_QWEN36_MODULE_TAG,path);
		return(SPARK_STATUS_IO_ERROR);
	}
	status = SparkStageModulePackRead(SPARK_QWEN36_MODULE_TAG,file,0u,&header,sizeof(header));
	if ( status == SPARK_STATUS_OK && (header.magic != SPARK_QWEN36_STAGEPACK_MAGIC || header.hidden_dimension != SPARK_QWEN36_MODEL_HIDDEN_DIMENSION || header.layer_count != SPARK_QWEN36_DSPARK_LAYER_COUNT || header.attn_query_head_count != SPARK_QWEN36_DSPARK_ATTN_QUERY_HEADS || header.attn_kv_head_count != SPARK_QWEN36_DSPARK_ATTN_KV_HEADS || header.attn_head_dimension != SPARK_QWEN36_DSPARK_ATTN_HEAD_DIMENSION || header.ffn_intermediate_dimension != SPARK_QWEN36_DSPARK_FFN_INTERMEDIATE || header.output_vocab_count != SPARK_QWEN36_MODEL_OUTPUT_VOCAB_COUNT) )
		status = SPARK_STATUS_VALIDATION_FAILED;
	directory = status == SPARK_STATUS_OK ? (SparkQwen36StagePackEntry *)malloc((size_t)header.tensor_count * sizeof(SparkQwen36StagePackEntry)) : 0;
	if ( status == SPARK_STATUS_OK && directory == 0 )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModulePackRead(SPARK_QWEN36_MODULE_TAG,file,header.directory_offset,directory,(uint64_t)header.tensor_count * sizeof(SparkQwen36StagePackEntry));
	for (index = 0u; status == SPARK_STATUS_OK && index < header.tensor_count; index++)
	{
		void *payload = 0, *scale = 0;
		status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,directory[index].payload_offset,directory[index].payload_bytes,&payload);
		if ( status == SPARK_STATUS_OK && directory[index].scale_bytes != 0u )
			status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,directory[index].scale_offset,directory[index].scale_bytes,&scale);
		if ( status == SPARK_STATUS_OK )
			status = SparkQwen36ModuleLoadDsparkEntry(state,&directory[index],payload,scale);
	}
	if ( status == SPARK_STATUS_OK )
		state->dspark_weights.armed = 1u;
	if ( status == SPARK_STATUS_OK )
	{
		const uint64_t markov_bytes = state->dspark_weights.markov_w1.weight_payload_bytes;
		state->dspark_weights.markov_w1_host = (uint16_t *)malloc((size_t)markov_bytes);
		state->dspark_weights.markov_w2_host = (uint16_t *)malloc((size_t)markov_bytes);
		if ( state->dspark_weights.markov_w1_host == 0 || state->dspark_weights.markov_w2_host == 0 )
			status = SPARK_STATUS_CAPACITY_EXCEEDED;
		if ( status == SPARK_STATUS_OK )
		{
			cudaError_t d2h = cudaMemcpy(state->dspark_weights.markov_w1_host,state->dspark_weights.markov_w1.weight_payload,(size_t)markov_bytes,cudaMemcpyDeviceToHost);
			if ( d2h == cudaSuccess )
				d2h = cudaMemcpy(state->dspark_weights.markov_w2_host,state->dspark_weights.markov_w2.weight_payload,(size_t)markov_bytes,cudaMemcpyDeviceToHost);
			status = SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,d2h,"dspark_markov_d2h");
		}
	}
	free(directory);
	fclose(file);
	return(status);
}

static SparkStatus SparkQwen36ModuleLoadPack(SparkQwen36ModuleState *state, const char *path)
{
	SparkQwen36StagePackHeader header,expected;
	SparkQwen36StagePackEntry *directory;
	FILE *file;
	SparkStatus status;
	int32_t compare;
	uint32_t index;
	file = fopen(path,"rb");
	if ( file == 0 )
	{
		fprintf(stderr,"%s pack_open_failed path=%s\n",SPARK_QWEN36_MODULE_TAG,path);
		return(SPARK_STATUS_IO_ERROR);
	}
	status = SparkStageModulePackRead(SPARK_QWEN36_MODULE_TAG,file,0u,&header,sizeof(header));
	if ( status == SPARK_STATUS_OK )
	{
		SparkQwen36StagePackExpectedGeometry(&expected,state->first_layer_index,state->layer_count);
		expected.tp_degree = state->tp_degree;
		expected.tp_rank = state->tp_rank;
		compare = SparkQwen36StagePackCompareGeometry(&header,&expected);
		if ( compare != 0 || header.directory_offset != SPARK_QWEN36_STAGEPACK_HEADER_BYTES )
		{
			fprintf(stderr,"%s pack_geometry_mismatch field=%s\n",SPARK_QWEN36_MODULE_TAG,compare != 0 ? SparkQwen36StagePackGeometryFieldName(compare) : "directory_offset");
			status = SPARK_STATUS_VALIDATION_FAILED;
		}
	}
	directory = status == SPARK_STATUS_OK ? (SparkQwen36StagePackEntry *)malloc((size_t)header.tensor_count * sizeof(SparkQwen36StagePackEntry)) : 0;
	if ( status == SPARK_STATUS_OK && directory == 0 )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModulePackRead(SPARK_QWEN36_MODULE_TAG,file,header.directory_offset,directory,(uint64_t)header.tensor_count * sizeof(SparkQwen36StagePackEntry));
	for (index = 0; status == SPARK_STATUS_OK && index < header.tensor_count; index++)
		status = SparkQwen36ModuleLoadEntry(state,file,&directory[index],header.file_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen36ModuleVerifyCoverage(state);
	free(directory);
	fclose(file);
	return(status);
}

// One-time MXFP4 shadow of the lm_head plus per-neuron certified error
// norms, the mimo25 screened-head pattern; head stage only, built
// synchronously at initialize.
static SparkStatus SparkQwen36ModuleBuildHeadShadow(SparkQwen36ModuleState *state)
{
	uint64_t vocab = state->tp.head_rows,dim = SPARK_QWEN36_MODEL_HIDDEN_DIMENSION;
	SparkStatus status;
	if ( state->owns_final_head == 0u )
		return(SPARK_STATUS_OK);
	status = SparkStageModuleDeviceAllocate(&state->ledger,(vocab * dim) / 2u,(void **)&state->head_shadow_payload);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(vocab * dim) / 32u,(void **)&state->head_shadow_scale);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,vocab * sizeof(float),(void **)&state->head_error_norm_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,SparkQwen36LaunchHeadShadowQuantize(0,state->lm_head_weight_bf16,state->head_shadow_payload,state->head_shadow_scale,state->head_error_norm_f32,(uint32_t)vocab,(uint32_t)dim),"head_shadow_quantize");
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,cudaDeviceSynchronize(),"head_shadow_sync");
	return(status);
}

/* Tensor-parallel bootstrap: one dedicated stream for the synchronous
 * collective submissions, the collective itself, and the per-rank device
 * geometry table. Must precede every allocation that derives per-rank
 * dimensions. */
static SparkStatus SparkQwen36ModuleInitializeTp(SparkQwen36ModuleState *state)
{
	cudaStream_t stream = 0;
	SparkStatus status = SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,cudaStreamCreate(&stream),"tp_stream_create");
	if ( status != SPARK_STATUS_OK )
		return(status);
	state->tp_stream = stream;
	status = SparkQwen36TpInitialize(&state->tp,state->tp_degree,state->tp_rank,state->max_active_sequence_count,state->pipeline_slot_count,stream);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,
		SparkQwen36TpSetGeometry(
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
static SparkStatus SparkQwen36ModuleTpReduceDelta(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot, uint32_t rows)
{
	SparkStatus status;
	if ( state->tp_degree <= 1u )
	{
		if ( state->profile_enabled != 0u )
			(void)cudaStreamSynchronize((cudaStream_t)slot->cuda_stream);
		return(SPARK_STATUS_OK);
	}
	status = SparkQwen36TpReduceHidden(&state->tp,slot->delta_bf16,rows,slot->cuda_stream);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr, "%s tp_reduce_delta_failed status=%d rows=%u\n", SPARK_QWEN36_MODULE_TAG, (int)status, rows);
	return status;
}

static SparkStatus SparkQwen36ModuleAllocatePools(SparkQwen36ModuleState *state)
{
	SparkStatus status = SPARK_STATUS_OK;
	uint64_t state_elements,tail_elements,cache_elements;
	state->gdn_pool.abi_version = SPARK_QWEN36_RESIDENT_DECODE_STAGE_GDN_STATE_POOL_ABI_VERSION;
	state->gdn_pool.lane_capacity = state->max_active_sequence_count;
	state->gdn_pool.gdn_layer_count = state->gdn_layer_count;
	state->gdn_pool.state_layer_stride_elements = (uint64_t)state->tp.gdn_value_heads * SPARK_QWEN36_MODEL_GDN_HEAD_KEY_DIMENSION * SPARK_QWEN36_MODEL_GDN_HEAD_VALUE_DIMENSION;
	state->gdn_pool.state_lane_stride_elements = state->gdn_pool.state_layer_stride_elements * state->gdn_layer_count;
	state->gdn_pool.conv_tail_layer_stride_elements = (uint64_t)state->tp.gdn_conv_channels * (SPARK_QWEN36_MODEL_GDN_CONV_KERNEL - 1u);
	state->gdn_pool.conv_tail_lane_stride_elements = state->gdn_pool.conv_tail_layer_stride_elements * state->gdn_layer_count;
	if ( state->gdn_layer_count != 0u )
	{
		state_elements = state->gdn_pool.state_lane_stride_elements * state->max_active_sequence_count;
		tail_elements = state->gdn_pool.conv_tail_lane_stride_elements * state->max_active_sequence_count;
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,state_elements * sizeof(float),(void **)&state->gdn_pool.state_f32);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,tail_elements * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&state->gdn_pool.conv_tail_bf16);
	}
	// The MTP decoder owns the LAST cache layer on an armed head stage; its
	// ordinal is attn_layer_count, so main-layer ordinals are undisturbed.
	state->mtp_cache_ordinal = state->attn_layer_count;
	state->cache_layer_count = state->attn_layer_count + (state->owns_final_head != 0u && state->mtp_armed != 0u ? SPARK_QWEN36_MODEL_MTP_LAYER_COUNT : 0u);
	if ( status == SPARK_STATUS_OK && state->cache_layer_count != 0u )
	{
		state->cache_layer_stride = (uint64_t)SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS * (uint64_t)state->tp.attn_kv_heads * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION * 2u;
		state->cache_block_stride = state->cache_layer_stride * state->cache_layer_count;
		cache_elements = state->cache_block_stride * state->kv_block_count;
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,cache_elements * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&state->kv_cache_bf16);
	}
	if ( status == SPARK_STATUS_OK && state->gdn_layer_count != 0u && state->gdn_snapshot_slot_count != 0u )
	{
		status = SparkStageModuleDeviceAllocate(&state->ledger,state->gdn_pool.state_lane_stride_elements * sizeof(float) * state->gdn_snapshot_slot_count,(void **)&state->snapshot_state_f32);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,state->gdn_pool.conv_tail_lane_stride_elements * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES * state->gdn_snapshot_slot_count,&state->snapshot_tail_bf16);
	}
	return(status);
}

static SparkStatus SparkQwen36ModuleAllocateSlotControl(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot)
{
	uint64_t rows = state->max_active_sequence_count,staged = rows + SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS;
	SparkStatus status;
	cudaStream_t stream = 0;
	status = SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,cudaStreamCreate(&stream),"cudaStreamCreate");
	if ( status != SPARK_STATUS_OK )
		return(status);
	slot->cuda_stream = stream;
	status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->input_token_ids);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->output_token_ids);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * SPARK_QWEN36_MODEL_OUTPUT_VOCAB_COUNT * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->head_logits_bf16);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * SPARK_QWEN36_RESIDENT_DECODE_STAGE_HEAD_SCREEN_CAP * sizeof(uint32_t),(void **)&slot->head_candidate_ids_u32);
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
		status = SparkStageModuleDeviceAllocate(&state->ledger,SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS * sizeof(uint32_t),(void **)&slot->mtp_draft_ids);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,SPARK_QWEN36_DSPARK_TARGET_TAP_COUNT * SPARK_QWEN36_MODEL_HIDDEN_DIMENSION * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->dspark_tap_buffer);
	/* DFlash2 scratch: context (5120) + block hidden + Q + K/V + attn out + norm + ffn (17408) + logits, all x block_size (8). */
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)(1u*SPARK_QWEN36_MODEL_HIDDEN_DIMENSION + SPARK_QWEN36_DSPARK_BLOCK_SIZE*SPARK_QWEN36_MODEL_HIDDEN_DIMENSION + SPARK_QWEN36_DSPARK_BLOCK_SIZE*SPARK_QWEN36_MODEL_HIDDEN_DIMENSION + 2u*(SPARK_QWEN36_DSPARK_BLOCK_SIZE+1u)*SPARK_QWEN36_DSPARK_ATTN_KV_HEADS*SPARK_QWEN36_DSPARK_ATTN_HEAD_DIMENSION + SPARK_QWEN36_DSPARK_BLOCK_SIZE*SPARK_QWEN36_MODEL_HIDDEN_DIMENSION + SPARK_QWEN36_DSPARK_BLOCK_SIZE*SPARK_QWEN36_MODEL_HIDDEN_DIMENSION + 2u*SPARK_QWEN36_DSPARK_BLOCK_SIZE*SPARK_QWEN36_DSPARK_FFN_INTERMEDIATE + SPARK_QWEN36_DSPARK_BLOCK_SIZE*SPARK_QWEN36_MODEL_OUTPUT_VOCAB_COUNT) * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->dspark_scratch);
	if ( status == SPARK_STATUS_OK )
		slot->dspark_logits_host = (uint16_t *)malloc((size_t)SPARK_QWEN36_DSPARK_BLOCK_SIZE * SPARK_QWEN36_MODEL_OUTPUT_VOCAB_COUNT * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		slot->dspark_conv_delta = (uint16_t *)malloc((size_t)SPARK_QWEN36_DSPARK_BLOCK_SIZE * 2u * SPARK_QWEN36_DSPARK_CONV_KERNEL_SIZE * (SPARK_QWEN36_MODEL_HIDDEN_DIMENSION / SPARK_QWEN36_DSPARK_CONV_GROUP_SIZE) * sizeof(uint16_t));
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)SPARK_QWEN36_DSPARK_BLOCK_SIZE * SPARK_QWEN36_MODEL_HIDDEN_DIMENSION * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->dspark_conv_out);
	if ( status == SPARK_STATUS_OK && slot->dspark_logits_host == 0 )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(SPARK_QWEN36_DSPARK_BLOCK_SIZE - 1u) * sizeof(uint32_t),(void **)&slot->dspark_mask_token_ids);
	if ( status == SPARK_STATUS_OK )
	{
		uint32_t host_mask[SPARK_QWEN36_DSPARK_BLOCK_SIZE - 1u];
		uint32_t i;
		for (i = 0u; i < SPARK_QWEN36_DSPARK_BLOCK_SIZE - 1u; i++)
			host_mask[i] = SPARK_QWEN36_DSPARK_MASK_TOKEN_ID;
		status = SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,cudaMemcpy(slot->dspark_mask_token_ids,host_mask,(SPARK_QWEN36_DSPARK_BLOCK_SIZE - 1u) * sizeof(uint32_t),cudaMemcpyHostToDevice),"dspark_mask_ids");
	}
	/* DFlash2 candidate-selector buffers: top-16 chunk workspace + candidate ids/unary
	 * over the (B-1) mask rows, the hidden-projection context gate, and the K x K
	 * edge lattice. Sized for one sequence (batch 1) x (B-1) draft slots. */
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,SparkQwen36DsparkHeadTopKChunkKeyCount(SPARK_QWEN36_DSPARK_BLOCK_SIZE - 1u,SPARK_QWEN36_DSPARK_SELECTOR_TOP_K) * sizeof(uint64_t),(void **)&slot->dspark_selector_chunk_keys);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)(SPARK_QWEN36_DSPARK_BLOCK_SIZE - 1u) * SPARK_QWEN36_DSPARK_SELECTOR_TOP_K * sizeof(uint32_t),(void **)&slot->dspark_selector_candidate_ids);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)(SPARK_QWEN36_DSPARK_BLOCK_SIZE - 1u) * SPARK_QWEN36_DSPARK_SELECTOR_TOP_K * sizeof(float),(void **)&slot->dspark_selector_unary);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)(SPARK_QWEN36_DSPARK_BLOCK_SIZE - 1u) * SPARK_QWEN36_DSPARK_SELECTOR_RANK * sizeof(uint16_t),(void **)&slot->dspark_selector_gate);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)(SPARK_QWEN36_DSPARK_BLOCK_SIZE - 1u) * SPARK_QWEN36_DSPARK_SELECTOR_TOP_K * SPARK_QWEN36_DSPARK_SELECTOR_TOP_K * sizeof(float),(void **)&slot->dspark_selector_edges);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)(SPARK_QWEN36_DSPARK_BLOCK_SIZE - 1u) * sizeof(uint32_t),(void **)&slot->dspark_selector_slots);
	return(status);
}

// Slot-owned GDN chunk workspace, the exact view layout the chunk launcher
// documents (per head: qn/kn/w/kg 64 x 128, decay/attn 64 x 64, cum_g 64).
// Only stages that own GDN layers pay for it.
static SparkStatus SparkQwen36ModuleAllocateSlotChunkWorkspace(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot)
{
	uint64_t heads = state->tp.gdn_value_heads,chunk = SPARK_QWEN36_MODEL_GDN_CHUNK_TOKENS;
	uint64_t vector_bytes = heads * chunk * SPARK_QWEN36_MODEL_GDN_HEAD_KEY_DIMENSION * sizeof(float);
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
		status = SparkStageModuleDeviceAllocate(&state->ledger,heads * chunk * SPARK_QWEN36_MODEL_GDN_HEAD_VALUE_DIMENSION * sizeof(float),(void **)&slot->chunk_w_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,vector_bytes,(void **)&slot->chunk_kg_f32);
	return(status);
}

static SparkStatus SparkQwen36ModuleAllocateSlot(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot)
{
	uint64_t rows = state->max_active_sequence_count;
	uint64_t attn_query_dim = (uint64_t)state->tp.attn_query_heads * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION;
	uint64_t attn_kv_dim = (uint64_t)state->tp.attn_kv_heads * SPARK_QWEN36_MODEL_ATTN_HEAD_DIMENSION;
	SparkStatus status = SparkQwen36ModuleAllocateSlotControl(state,slot);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN36_MODEL_HIDDEN_DIMENSION * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->hidden_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN36_MODEL_HIDDEN_DIMENSION * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->normalized_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN36_MODEL_HIDDEN_DIMENSION * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->delta_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * state->tp.gdn_conv_channels * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->qkv_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * state->tp.gdn_conv_channels * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->conv_out_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN36_MODEL_GDN_VALUE_DIMENSION * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->z_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->beta_pre_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->decay_pre_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * state->tp.gdn_value_heads * sizeof(float),(void **)&slot->log_decay_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * state->tp.gdn_value_heads * sizeof(float),(void **)&slot->beta_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * state->tp.gdn_value_channels * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->core_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * state->tp.gdn_value_channels * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->gated_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN36_MODULE_FUSED_QUERY_COMPONENT_COUNT * attn_query_dim * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->q_fused_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * attn_kv_dim * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->k_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * attn_kv_dim * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->v_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * attn_query_dim * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->head_out_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * state->tp.ffn_intermediate * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->ffn_gate_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * state->tp.ffn_intermediate * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,&slot->ffn_up_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen36ModuleAllocateSlotChunkWorkspace(state,slot);
	return(status);
}

static cudaError_t SparkQwen36ModuleRunGdnCoreDecode(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot, const SparkQwen36GdnLayerWeights *weights, uint32_t rows, uint32_t ordinal)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	/* Cold flags are per-frame (the slot's uploaded row_cold), not pool
	 * state: hand the kernels a per-call pool view so concurrent slots never
	 * race the shared pool's pointer. */
	SparkQwen36GdnStatePool pool = state->gdn_pool;
	pool.state_cold_by_row = slot->row_cold;
	error = SparkQwen36LaunchConvUpdate(stream,slot->qkv_bf16,weights,slot->conv_out_bf16,&pool,slot->row_lane_indices,rows,ordinal);
	if ( error != cudaSuccess && rows >= 32u )
		fprintf(stderr, "%s gdn_core conv_failed rows=%u err=%d\n", SPARK_QWEN36_MODULE_TAG, rows, (int)error);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchDecayBeta(stream,slot->decay_pre_bf16,slot->beta_pre_bf16,weights,slot->log_decay_f32,slot->beta_f32,rows);
	if ( error != cudaSuccess && rows >= 32u )
		fprintf(stderr, "%s gdn_core decaybeta_failed rows=%u err=%d\n", SPARK_QWEN36_MODULE_TAG, rows, (int)error);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchGdnStep(stream,slot->conv_out_bf16,slot->log_decay_f32,slot->beta_f32,&pool,slot->core_bf16,slot->row_lane_indices,rows,ordinal);
	if ( error != cudaSuccess && rows >= 32u )
		fprintf(stderr, "%s gdn_core gdnstep_failed rows=%u err=%d\n", SPARK_QWEN36_MODULE_TAG, rows, (int)error);
	return(error);
}

// Prefill GDN core: conv over the whole frame with the carried tail, then
// the 64-token chunk sequence per slice of the frame; looping chunks on the
// one slot stream serializes the state dependency for free.
static cudaError_t SparkQwen36ModuleRunGdnCorePrefill(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot, const SparkQwen36GdnLayerWeights *weights, uint32_t lane, uint32_t rows, uint32_t ordinal)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint32_t start,count;
	uint64_t conv_bytes,core_bytes,head_offset;
	cudaError_t error;
	error = SparkQwen36LaunchChunkConv(stream,slot->qkv_bf16,weights,slot->conv_out_bf16,&state->gdn_pool,lane,rows,ordinal);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchDecayBeta(stream,slot->decay_pre_bf16,slot->beta_pre_bf16,weights,slot->log_decay_f32,slot->beta_f32,rows);
	for (start = 0; error == cudaSuccess && start < rows; start += SPARK_QWEN36_MODEL_GDN_CHUNK_TOKENS)
	{
		count = rows - start < SPARK_QWEN36_MODEL_GDN_CHUNK_TOKENS ? rows - start : SPARK_QWEN36_MODEL_GDN_CHUNK_TOKENS;
		conv_bytes = (uint64_t)start * SPARK_QWEN36_MODEL_GDN_CONV_CHANNELS * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES;
		core_bytes = (uint64_t)start * SPARK_QWEN36_MODEL_GDN_VALUE_DIMENSION * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES;
		head_offset = (uint64_t)start * SPARK_QWEN36_MODEL_GDN_VALUE_HEAD_COUNT;
		error = SparkQwen36LaunchGdnChunk(stream,(const void *)((const uint8_t *)slot->conv_out_bf16 + conv_bytes),slot->log_decay_f32 + head_offset,slot->beta_f32 + head_offset,slot->chunk_qn_f32,slot->chunk_kn_f32,slot->chunk_cum_g_f32,slot->chunk_decay_f32,slot->chunk_attn_f32,slot->chunk_w_f32,slot->chunk_kg_f32,&state->gdn_pool,(void *)((uint8_t *)slot->core_bf16 + core_bytes),lane,count,ordinal);
	}
	return(error);
}

static SparkStatus SparkQwen36ModuleRunGdnLayer(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot, const SparkQwen36PrefillFrameView *prefill, uint32_t layer, uint32_t rows)
{
	const SparkQwen36GdnLayerWeights *weights = &state->gdn_by_layer[layer];
	uint32_t ordinal = state->gdn_ordinal_by_layer[layer];
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	error = SparkQwen36LaunchLinear(stream,&weights->qkv,slot->normalized_bf16,slot->qkv_bf16,rows);
	if ( error != cudaSuccess && rows >= 32u )
		fprintf(stderr, "%s gdn_qkv_failed rows=%u err=%d\n", SPARK_QWEN36_MODULE_TAG, rows, (int)error);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchLinear(stream,&weights->gate,slot->normalized_bf16,slot->z_bf16,rows);
	if ( error != cudaSuccess && rows >= 32u )
		fprintf(stderr, "%s gdn_gate_failed rows=%u err=%d\n", SPARK_QWEN36_MODULE_TAG, rows, (int)error);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchLinear(stream,&weights->beta,slot->normalized_bf16,slot->beta_pre_bf16,rows);
	if ( error != cudaSuccess && rows >= 32u )
		fprintf(stderr, "%s gdn_beta_failed rows=%u err=%d\n", SPARK_QWEN36_MODULE_TAG, rows, (int)error);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchLinear(stream,&weights->decay,slot->normalized_bf16,slot->decay_pre_bf16,rows);
	if ( error != cudaSuccess && rows >= 32u )
		fprintf(stderr, "%s gdn_decay_failed rows=%u err=%d\n", SPARK_QWEN36_MODULE_TAG, rows, (int)error);
	if ( error == cudaSuccess )
		error = prefill != 0 ? SparkQwen36ModuleRunGdnCorePrefill(state,slot,weights,prefill->lane_index,rows,ordinal) : SparkQwen36ModuleRunGdnCoreDecode(state,slot,weights,rows,ordinal);
	if ( error != cudaSuccess && rows >= 32u )
		fprintf(stderr, "%s gdn_core_failed rows=%u err=%d\n", SPARK_QWEN36_MODULE_TAG, rows, (int)error);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchGatedNorm(stream,slot->core_bf16,slot->z_bf16,weights,slot->gated_bf16,rows,SPARK_QWEN36_MODEL_RMS_NORM_EPSILON);
	if ( error != cudaSuccess && rows >= 32u )
		fprintf(stderr, "%s gatednorm_failed rows=%u err=%d\n", SPARK_QWEN36_MODULE_TAG, rows, (int)error);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchLinear(stream,&weights->output,slot->gated_bf16,slot->delta_bf16,rows);
	if ( error != cudaSuccess && rows >= 32u )
		fprintf(stderr, "%s gdn_output_failed rows=%u err=%d\n", SPARK_QWEN36_MODULE_TAG, rows, (int)error);
	{
		uint64_t spin_start = state->profile_enabled != 0u ? SparkQwen36ProfileNow() : 0ull;
		if ( error == cudaSuccess && SparkQwen36ModuleTpReduceDelta(state,slot,rows) != SPARK_STATUS_OK )
			error = cudaErrorUnknown;
		if ( state->profile_enabled != 0u )
			state->profile_gdn_spin_nanos += SparkQwen36ProfileNow() - spin_start;
	}
	return(SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,error,"gdn_layer"));
}

// Device row-control pointers for one attention pass. Main layers bind the
// slot arrays from row zero; an MTP draft step binds them offset to its one
// staged draft row, so the SAME attention path serves both.
typedef struct SparkQwen36AttnRowsView
{
	const uint32_t *slot_mapping;
	const uint64_t *row_positions;
	const uint32_t *row_lane_indices;
	const uint32_t *context_lengths;
} SparkQwen36AttnRowsView;

static SparkStatus SparkQwen36ModuleRunAttnLayer(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot, const SparkQwen36KvBlockTableView *table, const SparkQwen36AttnLayerWeights *weights, uint32_t ordinal, const SparkQwen36AttnRowsView *rows_view, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	error = SparkQwen36LaunchLinear(stream,&weights->query,slot->normalized_bf16,slot->q_fused_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchLinear(stream,&weights->key,slot->normalized_bf16,slot->k_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchLinear(stream,&weights->value,slot->normalized_bf16,slot->v_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchAttnPrepare(stream,slot->q_fused_bf16,slot->k_bf16,slot->v_bf16,weights,state->kv_cache_bf16,rows_view->slot_mapping,rows_view->row_positions,rows,ordinal,state->cache_layer_stride,state->cache_block_stride,SPARK_QWEN36_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchAttnDecode(stream,slot->q_fused_bf16,state->kv_cache_bf16,table,rows_view->row_lane_indices,rows_view->context_lengths,slot->head_out_bf16,rows,ordinal,state->cache_layer_stride,state->cache_block_stride);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchLinear(stream,&weights->output,slot->head_out_bf16,slot->delta_bf16,rows);
	{
		uint64_t spin_start = state->profile_enabled != 0u ? SparkQwen36ProfileNow() : 0ull;
		if ( error == cudaSuccess && SparkQwen36ModuleTpReduceDelta(state,slot,rows) != SPARK_STATUS_OK )
			error = cudaErrorUnknown;
		if ( state->profile_enabled != 0u )
			state->profile_attn_spin_nanos += SparkQwen36ProfileNow() - spin_start;
	}
	return(SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,error,"attn_layer"));
}

static SparkStatus SparkQwen36ModuleRunFfn(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot, const void *mlp_norm_bf16, const SparkQwen36FfnLayerWeights *weights, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	const char *ffn_gate_env;
	error = SparkQwen36LaunchFusedResidualRmsNorm(stream,slot->hidden_bf16,slot->delta_bf16,mlp_norm_bf16,slot->normalized_bf16,rows,SPARK_QWEN36_MODEL_HIDDEN_DIMENSION,SPARK_QWEN36_MODEL_RMS_NORM_EPSILON);
	ffn_gate_env = getenv("SPARK_QWEN36_SMALL_BATCH_GEMM");
	if ( error == cudaSuccess && (ffn_gate_env == 0 || strcmp(ffn_gate_env, "0") != 0) &&
		rows >= 5u && rows <= SPARK_QWEN36_SMALL_BATCH_MAX_ROWS &&
		weights->gate.weight_format == SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 &&
		weights->up.weight_format == SPARK_QWEN36_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 &&
		weights->gate.input_dimension == weights->up.input_dimension &&
		weights->gate.output_dimension == weights->up.output_dimension &&
		(weights->gate.input_dimension % SPARK_QWEN36_SMALL_BATCH_K_CHUNK) == 0u &&
		(weights->gate.output_dimension % SPARK_QWEN36_SMALL_BATCH_TILE_N) == 0u )
	{
		/* fused gate+up+swiglu: both projections stream once and the product
		 * lands directly in ffn_up_bf16, bit-identical to the three kernels */
		error = SparkQwen36LaunchFfnGateUp(stream,weights->gate.weight_payload,weights->up.weight_payload,slot->normalized_bf16,slot->ffn_up_bf16,rows,weights->gate.input_dimension,weights->gate.output_dimension);
	}
	else
	{
		if ( error == cudaSuccess )
			error = SparkQwen36LaunchLinear(stream,&weights->gate,slot->normalized_bf16,slot->ffn_gate_bf16,rows);
		if ( error == cudaSuccess )
			error = SparkQwen36LaunchLinear(stream,&weights->up,slot->normalized_bf16,slot->ffn_up_bf16,rows);
		if ( error == cudaSuccess )
			error = SparkQwen36LaunchSwiGlu(stream,slot->ffn_gate_bf16,slot->ffn_up_bf16,rows,state->tp.ffn_intermediate);
	}
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchLinear(stream,&weights->down,slot->ffn_up_bf16,slot->delta_bf16,rows);
	{
		uint64_t spin_start = state->profile_enabled != 0u ? SparkQwen36ProfileNow() : 0ull;
		if ( error == cudaSuccess && SparkQwen36ModuleTpReduceDelta(state,slot,rows) != SPARK_STATUS_OK )
			error = cudaErrorUnknown;
		if ( state->profile_enabled != 0u )
			state->profile_ffn_spin_nanos += SparkQwen36ProfileNow() - spin_start;
	}
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchResidualAdd(stream,slot->hidden_bf16,slot->delta_bf16,rows,SPARK_QWEN36_MODEL_HIDDEN_DIMENSION);
	return(SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,error,"ffn"));
}

static SparkStatus SparkQwen36ModuleRunLayer(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot, const SparkQwen36KvBlockTableView *table, const SparkQwen36PrefillFrameView *prefill, uint32_t layer, uint32_t rows)
{
	SparkQwen36AttnRowsView rows_view;
	SparkStatus status;
	cudaError_t error = SparkQwen36LaunchRmsNorm((cudaStream_t)slot->cuda_stream,slot->hidden_bf16,state->attention_norm_by_layer[layer],slot->normalized_bf16,rows,SPARK_QWEN36_MODEL_HIDDEN_DIMENSION,SPARK_QWEN36_MODEL_RMS_NORM_EPSILON);
	rows_view.slot_mapping = slot->slot_mapping;
	rows_view.row_positions = slot->row_positions;
	rows_view.row_lane_indices = slot->row_lane_indices;
	rows_view.context_lengths = slot->context_lengths;
	status = SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,error,"attention_norm");
	if ( status == SPARK_STATUS_OK )
		status = SPARK_QWEN36_MODEL_LAYER_IS_GDN(layer) != 0u ? SparkQwen36ModuleRunGdnLayer(state,slot,prefill,layer,rows) : SparkQwen36ModuleRunAttnLayer(state,slot,table,&state->attn_by_layer[layer],state->attn_ordinal_by_layer[layer],&rows_view,rows);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen36ModuleRunFfn(state,slot,state->mlp_norm_by_layer[layer],&state->ffn_by_layer[layer],rows);
	return(status);
}

static SparkStatus SparkQwen36ModuleValidateDecodeView(
    SparkQwen36ModuleState *state,
    const SparkModelDriverFrame *frame,
    const SparkQwen36ResidentDecodeStageFrameContext *context)
{
    const SparkQwen36DecodeBatchView *batch;
    uint32_t row;

    batch = context->decode_batch;
    if (batch == 0 ||
        batch->abi_version !=
            SPARK_QWEN36_RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION ||
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
                SPARK_QWEN36_MODEL_MAXIMUM_CONTEXT_TOKENS)
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
static SparkStatus SparkQwen36ModuleValidatePrefillView(
    SparkQwen36ModuleState *state,
    const SparkModelDriverFrame *frame,
    const SparkQwen36ResidentDecodeStageFrameContext *context)
{
    const SparkQwen36PrefillFrameView *view;

    view = context->prefill_frame;
    if (view == 0 ||
        view->abi_version !=
            SPARK_QWEN36_RESIDENT_DECODE_STAGE_PREFILL_FRAME_VIEW_ABI_VERSION ||
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
            SPARK_QWEN36_MODEL_MAXIMUM_CONTEXT_TOKENS) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkQwen36ModuleValidateSpeculation(
    SparkQwen36ModuleState *state,
    const SparkQwen36ResidentDecodeStageFrameContext *context)
{
    const SparkQwen36PrefillFrameView *prefill;
    const SparkQwen36MtpDraftView *draft;
    const SparkQwen36GdnSnapshotView *snapshot;
    const SparkQwen36DecodeBatchView *decode_batch;
    uint32_t drafted;
    uint32_t restore;
    uint32_t verify;
    uint32_t row;
    uint32_t matching_row_found;

    prefill = context->prefill_frame;
    draft = context->mtp_draft;
    snapshot = context->gdn_snapshot;
    decode_batch = context->decode_batch;
    verify = context->flags &
        SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_SPECULATIVE_VERIFY;
    restore = context->flags &
        SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_RESTORE_FIRST;
    drafted = context->flags &
        SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MTP_DRAFT_AFTER;

    if (verify != 0u || restore != 0u)
    {
        if (prefill == 0 || (verify != 0u && restore != 0u) ||
            (verify != 0u && prefill->base_position == 0u) ||
            snapshot == 0 ||
            snapshot->abi_version !=
                SPARK_QWEN36_RESIDENT_DECODE_STAGE_GDN_SNAPSHOT_VIEW_ABI_VERSION ||
            snapshot->descriptor_bytes < (uint32_t)sizeof(*snapshot) ||
            snapshot->reserved0 != 0u ||
            (state->gdn_layer_count != 0u &&
             (state->gdn_snapshot_slot_count == 0u ||
              snapshot->snapshot_index >= state->gdn_snapshot_slot_count)))
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
            SPARK_QWEN36_RESIDENT_DECODE_STAGE_MTP_DRAFT_VIEW_ABI_VERSION ||
        draft->descriptor_bytes < (uint32_t)sizeof(*draft) ||
        draft->draft_token_count == 0u ||
        draft->draft_token_count >
            SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS ||
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

static SparkStatus SparkQwen36ModuleValidateFrame(
    SparkQwen36ModuleState *state,
    const SparkModelDriverFrame *frame,
    const SparkQwen36ResidentDecodeStageFrameContext **context_out)
{
    const uint32_t known_frame_flags = SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL;
    const uint32_t known_context_flags =
        SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE |
        SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW |
        SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT |
        SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT |
        SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW |
        SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MTP_DRAFT_AFTER |
        SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_SPECULATIVE_VERIFY |
        SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_RESTORE_FIRST |
        SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DSPARK_DRAFT_AFTER;
    const SparkQwen36ResidentDecodeStageFrameContext *context;
    const SparkQwen36KvBlockTableView *block_table;
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

    context = (const SparkQwen36ResidentDecodeStageFrameContext *)frame->user_context;
    if (context->abi_version !=
            SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION ||
        context->descriptor_bytes < (uint32_t)sizeof(*context) ||
        context->reserved0 != 0u ||
        (context->flags & ~known_context_flags) != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    mode = context->flags &
        (SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW |
         SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW);
    is_prefill = (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u
        ? 1u
        : 0u;
    if ((is_prefill != 0u &&
         mode != SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW) ||
        (is_prefill == 0u &&
         mode != SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    needs_hidden_input = state->stage_index > 0u ? 1u : 0u;
    needs_hidden_output = state->stage_index + 1u < state->stage_count ? 1u : 0u;
    if (((context->flags &
          SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT) != 0u) !=
            (needs_hidden_input != 0u) ||
        ((context->flags &
          SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT) != 0u) !=
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
             SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE) == 0u ||
            block_table == 0 ||
            block_table->abi_version !=
                SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TABLE_ABI_VERSION ||
            block_table->descriptor_bytes < (uint32_t)sizeof(*block_table) ||
            block_table->block_token_count !=
                SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS ||
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
              SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE) != 0u ||
             block_table != 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = is_prefill != 0u
        ? SparkQwen36ModuleValidatePrefillView(state, frame, context)
        : SparkQwen36ModuleValidateDecodeView(state, frame, context);
    if (status == SPARK_STATUS_OK)
    {
        status = SparkQwen36ModuleValidateSpeculation(state, context);
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
             SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_SPECULATIVE_VERIFY) == 0u
            ? 1u
            : row_count;
        if ((context->flags &
             SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MTP_DRAFT_AFTER) != 0u)
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
static SparkStatus SparkQwen36ModuleStagePosition(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot, const SparkQwen36KvBlockTableView *table, uint32_t lane, uint64_t position, uint32_t index)
{
	uint32_t block_ordinal,block;
	slot->host_row_lane_indices[index] = lane;
	slot->host_row_positions[index] = position;
	if ( state->attn_layer_count == 0u )
		return(SPARK_STATUS_OK);
	if ( position + 1u > (uint64_t)table->lane_stride * SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	block_ordinal = (uint32_t)(position / SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
	if ( lane >= table->lane_count || block_ordinal >= table->host_lane_physical_block_counts[lane] )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	block = table->host_physical_block_indices[((uint64_t)lane * table->lane_stride) + block_ordinal];
	if ( block == SPARK_QWEN36_RESIDENT_DECODE_STAGE_NO_BLOCK || block >= state->kv_block_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	slot->host_slot_mapping[index] = (block * SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS) + (uint32_t)(position % SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
	slot->host_context_lengths[index] = (uint32_t)(position + 1u);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen36ModuleStageRows(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot, const SparkQwen36ResidentDecodeStageFrameContext *context, uint8_t *lane_used)
{
	const SparkQwen36DecodeBatchView *batch = context->decode_batch;
	uint32_t row,lane;
	SparkStatus status;
	for (row = 0; row < batch->row_count; row++)
	{
		lane = batch->row_lane_indices[row];
		if ( lane >= state->max_active_sequence_count || lane_used[lane] != 0u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		lane_used[lane] = 1u;
		slot->host_row_cold[row] = state->lane_warm[lane] != 0u ? 0u : 1u;
		status = SparkQwen36ModuleStagePosition(state,slot,context->kv_block_table,lane,batch->row_positions[row],row);
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
static SparkStatus SparkQwen36ModulePrefillStage(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot, const SparkQwen36ResidentDecodeStageFrameContext *context)
{
	const SparkQwen36PrefillFrameView *view = context->prefill_frame;
	uint32_t index;
	SparkStatus status;
	for (index = 0; index < view->token_count; index++)
	{
		status = SparkQwen36ModuleStagePosition(state,slot,context->kv_block_table,view->lane_index,view->base_position + index,index);
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
static SparkStatus SparkQwen36ModuleStageMtpDraft(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot, const SparkQwen36ResidentDecodeStageFrameContext *context, const SparkQwen36PrefillFrameView *prefill, uint32_t rows)
{
	const SparkQwen36MtpDraftView *view = context->mtp_draft;
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
		status = SparkQwen36ModuleStagePosition(state,slot,context->kv_block_table,view->lane_index,view->base_position + draft,rows + draft);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen36ModuleUploadRows(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot, const SparkQwen36ResidentDecodeStageFrameContext *context, const SparkModelDriverFrame *frame, const SparkQwen36PrefillFrameView *prefill, uint32_t rows)
{
	uint32_t token_guard;
	uint32_t drafted = (context->flags & SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MTP_DRAFT_AFTER) != 0u ? 1u : 0u;
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
			if ( ((const uint32_t *)frame->buffers[0].address)[token_guard] >= SPARK_QWEN36_MODEL_OUTPUT_VOCAB_COUNT )
			{
				fprintf(stderr,"%s token_id_out_of_range row=%u\n",SPARK_QWEN36_MODULE_TAG,token_guard);
				return(SPARK_STATUS_INVALID_ARGUMENT);
			}
		error = cudaMemcpyAsync(slot->input_token_ids,frame->buffers[0].address,rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	}
	if ( error == cudaSuccess && drafted != 0u && state->owns_embedding == 0u )
		error = cudaMemcpyAsync(slot->input_token_ids,context->mtp_draft->row_token_ids,(prefill != 0 ? rows : 1u) * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	return(SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,error,"stage_upload"));
}

// A base-zero prefill claims the lane fresh: the chunk kernels read the
// resident state unconditionally, so a reused lane's stale delta state and
// conv tails are zeroed on the slot stream before the walk.
static SparkStatus SparkQwen36ModuleResetLaneState(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot, uint32_t lane)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint64_t state_bytes = state->gdn_pool.state_lane_stride_elements * sizeof(float);
	uint64_t tail_bytes = state->gdn_pool.conv_tail_lane_stride_elements * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES;
	cudaError_t error;
	if ( state->gdn_layer_count == 0u )
		return(SPARK_STATUS_OK);
	error = cudaMemsetAsync(state->gdn_pool.state_f32 + ((uint64_t)lane * state->gdn_pool.state_lane_stride_elements),0,state_bytes,stream);
	if ( error == cudaSuccess )
		error = cudaMemsetAsync((uint8_t *)state->gdn_pool.conv_tail_bf16 + ((uint64_t)lane * tail_bytes),0,tail_bytes,stream);
	return(SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,error,"lane_reset"));
}

// Only the GDN recurrence is destructive under speculation: attention and
// MTP K/V at rejected positions are simply overwritten when the position
// re-executes. A verify frame copies the lane's delta state and conv tails
// OUT to the runtime-assigned snapshot slot before the walk; the replay
// frame copies them back IN before re-advancing over the accepted tokens.
static SparkStatus SparkQwen36ModuleGdnSnapshot(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot, uint32_t lane, uint32_t snapshot_index, uint32_t restore)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint64_t state_bytes = state->gdn_pool.state_lane_stride_elements * sizeof(float);
	uint64_t tail_bytes = state->gdn_pool.conv_tail_lane_stride_elements * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES;
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
	return(SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,error,"gdn_snapshot"));
}

static SparkStatus SparkQwen36ModuleBeginHidden(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot, SparkQwen36ResidentDecodeStageFrameContext *context, uint32_t rows)
{
	SparkStatus status;
	cudaError_t error;
	if ( state->owns_embedding != 0u )
	{
		error = SparkQwen36LaunchEmbeddingGather((cudaStream_t)slot->cuda_stream,slot->input_token_ids,state->token_embedding_bf16,slot->hidden_bf16,rows);
		return(SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,error,"embedding"));
	}
	if ( context->hidden_input_post_receive_function == 0 || context->hidden_input_transport_session == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = context->hidden_input_post_receive_function(context->hidden_input_transport_session,&context->hidden_input_packet);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( context->hidden_input_packet.active_sequence_count != rows || context->hidden_input_packet.hidden_dimension != SPARK_QWEN36_MODEL_HIDDEN_DIMENSION || context->hidden_input_packet.hidden_bf16 == 0 )
		return(SPARK_STATUS_VALIDATION_FAILED);
	error = cudaMemcpyAsync(slot->hidden_bf16,context->hidden_input_packet.hidden_bf16,(uint64_t)rows * SPARK_QWEN36_MODEL_HIDDEN_DIMENSION * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES,cudaMemcpyDeviceToDevice,(cudaStream_t)slot->cuda_stream);
	return(SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,error,"hidden_receive"));
}

/*
 * Head emission. Decode samples every row. Prefill samples ONLY the final
 * position: the norm and the fused matvec+argmax run on one row addressed
 * at the frame's last hidden row, and exactly one token id lands in the
 * output buffer - a 512-row argmax over a 248320 vocabulary for positions
 * nothing reads would be pure waste.
 */
static cudaError_t SparkQwen36ModuleEmitHead(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot, SparkModelDriverFrame *frame, const SparkQwen36PrefillFrameView *prefill, uint32_t emit_all, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint32_t out_index = state->owns_embedding != 0u ? 1u : 0u,head_rows = prefill != 0 && emit_all == 0u ? 1u : rows;
	const void *head_hidden = head_rows == 1u && rows != 1u ? (const void *)((const uint8_t *)slot->hidden_bf16 + ((uint64_t)(rows - 1u) * SPARK_QWEN36_MODEL_HIDDEN_DIMENSION * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES)) : slot->hidden_bf16;
	cudaError_t error;
	error = SparkQwen36LaunchRmsNorm(stream,head_hidden,state->final_norm_weight_bf16,slot->normalized_bf16,head_rows,SPARK_QWEN36_MODEL_HIDDEN_DIMENSION,SPARK_QWEN36_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchHeadScreenedArgmaxScore(stream,slot->normalized_bf16,state->lm_head_weight_bf16,state->head_shadow_payload,state->head_shadow_scale,state->head_error_norm_f32,slot->head_logits_bf16,slot->head_candidate_ids_u32,slot->head_candidate_counts_u32,slot->output_token_ids,slot->head_scores_f32,state->tp_rank * state->tp.head_rows,head_rows,state->tp.head_rows);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchHeadMaxLocPack(stream,slot->head_scores_f32,slot->output_token_ids,slot->head_maxloc_u64,head_rows);
	{
		uint64_t spin_start = state->profile_enabled != 0u ? SparkQwen36ProfileNow() : 0ull;
		if ( error == cudaSuccess && SparkQwen36TpReduceU64Max(&state->tp,slot->head_maxloc_u64,head_rows,stream) != SPARK_STATUS_OK )
			error = cudaErrorUnknown;
		if ( state->profile_enabled != 0u )
		{
			if ( state->tp_degree <= 1u )
				(void)cudaStreamSynchronize(stream);
			state->profile_head_spin_nanos += SparkQwen36ProfileNow() - spin_start;
		}
	}
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchHeadMaxLocUnpack(stream,slot->head_maxloc_u64,slot->output_token_ids,head_rows);
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
static SparkStatus SparkQwen36ModuleRunMtpPackInput(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot, const uint32_t *token_src, const void *hnorm_src, uint32_t rows_p)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint64_t half_bytes = SPARK_QWEN36_MODEL_HIDDEN_BF16_BYTES,pack_pitch = 2u * half_bytes;
	cudaError_t error;
	error = SparkQwen36LaunchEmbeddingGather(stream,token_src,state->token_embedding_bf16,slot->gated_bf16,rows_p);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchRmsNorm(stream,slot->gated_bf16,state->mtp.embed_norm_weight_bf16,slot->normalized_bf16,rows_p,SPARK_QWEN36_MODEL_HIDDEN_DIMENSION,SPARK_QWEN36_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchRmsNorm(stream,hnorm_src,state->mtp.hidden_norm_weight_bf16,slot->delta_bf16,rows_p,SPARK_QWEN36_MODEL_HIDDEN_DIMENSION,SPARK_QWEN36_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = cudaMemcpy2DAsync(slot->qkv_bf16,pack_pitch,slot->normalized_bf16,half_bytes,half_bytes,rows_p,cudaMemcpyDeviceToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpy2DAsync((uint8_t *)slot->qkv_bf16 + half_bytes,pack_pitch,slot->delta_bf16,half_bytes,half_bytes,rows_p,cudaMemcpyDeviceToDevice,stream);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchLinear(stream,&state->mtp.fc,slot->qkv_bf16,slot->hidden_bf16,rows_p);
	return(SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,error,"mtp_pack"));
}

static SparkStatus SparkQwen36ModuleRunMtpDecoderPass(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot, const SparkQwen36KvBlockTableView *table, const SparkQwen36AttnRowsView *rows_view, uint32_t rows_p)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	SparkStatus status;
	cudaError_t error = SparkQwen36LaunchRmsNorm(stream,slot->hidden_bf16,state->mtp.attention_norm_weight_bf16,slot->normalized_bf16,rows_p,SPARK_QWEN36_MODEL_HIDDEN_DIMENSION,SPARK_QWEN36_MODEL_RMS_NORM_EPSILON);
	status = SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,error,"mtp_attn_norm");
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen36ModuleRunAttnLayer(state,slot,table,&state->mtp.attention,state->mtp_cache_ordinal,rows_view,rows_p);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,SparkQwen36LaunchResidualAdd(stream,slot->hidden_bf16,slot->delta_bf16,rows_p,SPARK_QWEN36_MODEL_HIDDEN_DIMENSION),"mtp_residual");
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen36ModuleRunFfn(state,slot,state->mtp.mlp_norm_weight_bf16,&state->mtp.ffn,rows_p);
	return(status);
}

static SparkStatus SparkQwen36ModuleRunMtpArgmaxRow(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot, uint32_t row, uint32_t draft_index)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	const void *row_hidden = (const uint8_t *)slot->hidden_bf16 + ((uint64_t)row * SPARK_QWEN36_MODEL_HIDDEN_BF16_BYTES);
	cudaError_t error;
	error = SparkQwen36LaunchRmsNorm(stream,row_hidden,state->mtp.final_norm_weight_bf16,slot->normalized_bf16,1u,SPARK_QWEN36_MODEL_HIDDEN_DIMENSION,SPARK_QWEN36_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess && state->tp_degree > 1u )
	{
		/* TP4: argmax over the rank's head shard, then the u64 maxloc
		 * collective picks the global winner across ranks. */
		error = SparkQwen36LaunchHeadScreenedArgmaxScore(stream,slot->normalized_bf16,state->lm_head_weight_bf16,state->head_shadow_payload,state->head_shadow_scale,state->head_error_norm_f32,slot->head_logits_bf16,slot->head_candidate_ids_u32,slot->head_candidate_counts_u32,slot->mtp_draft_ids + draft_index,slot->head_scores_f32,state->tp_rank * state->tp.head_rows,1u,state->tp.head_rows);
		if ( error == cudaSuccess )
			error = SparkQwen36LaunchHeadMaxLocPack(stream,slot->head_scores_f32,slot->mtp_draft_ids + draft_index,slot->head_maxloc_u64,1u);
		if ( error == cudaSuccess && SparkQwen36TpReduceU64Max(&state->tp,slot->head_maxloc_u64,1u,stream) != SPARK_STATUS_OK )
			error = cudaErrorUnknown;
		if ( error == cudaSuccess )
			error = SparkQwen36LaunchHeadMaxLocUnpack(stream,slot->head_maxloc_u64,slot->mtp_draft_ids + draft_index,1u);
	}
	else if ( error == cudaSuccess )
		error = SparkQwen36LaunchHeadScreenedArgmax(stream,slot->normalized_bf16,state->lm_head_weight_bf16,state->head_shadow_payload,state->head_shadow_scale,state->head_error_norm_f32,slot->head_logits_bf16,slot->head_candidate_ids_u32,slot->head_candidate_counts_u32,slot->mtp_draft_ids + draft_index,1u,SPARK_QWEN36_MODEL_OUTPUT_VOCAB_COUNT);
	return(SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,error,"mtp_argmax"));
}

/*
 * The seed pass batches the MTP decoder over the frame's own rows, feeding
 * committed token ids against committed backbone hiddens, which rewrites
 * the lane's MTP K/V at those positions and yields draft one at the final
 * row. Chain steps then extend one position each, embedding the previous
 * draft against the previous MTP hidden through the identical pass. The
 * frame's backbone hiddens are dead by now: the head consumed them.
 */
static SparkStatus SparkQwen36ModuleRunMtpDraftChain(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot, SparkQwen36ResidentDecodeStageFrameContext *context, SparkModelDriverFrame *frame, const SparkQwen36PrefillFrameView *prefill, uint32_t rows)
{
	const SparkQwen36MtpDraftView *view = context->mtp_draft;
	uint32_t verify = (context->flags & SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_SPECULATIVE_VERIFY) != 0u ? 1u : 0u;
	uint32_t head_rows = prefill != 0 && verify == 0u ? 1u : rows;
	uint32_t rows_p = prefill != 0 ? rows : 1u,row_base = prefill != 0 ? 0u : slot->mtp_seed_row,step;
	uint32_t out_index = state->owns_embedding != 0u ? 1u : 0u;
	const uint32_t *seed_ids = slot->input_token_ids + (state->owns_embedding != 0u ? row_base : 0u);
	const void *seed_hidden = (const uint8_t *)slot->hidden_bf16 + ((uint64_t)row_base * SPARK_QWEN36_MODEL_HIDDEN_BF16_BYTES);
	SparkQwen36AttnRowsView rows_view;
	SparkStatus status;
	rows_view.slot_mapping = slot->slot_mapping + row_base;
	rows_view.row_positions = slot->row_positions + row_base;
	rows_view.row_lane_indices = slot->row_lane_indices + row_base;
	rows_view.context_lengths = slot->context_lengths + row_base;
	status = SparkQwen36ModuleRunMtpPackInput(state,slot,seed_ids,seed_hidden,rows_p);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen36ModuleRunMtpDecoderPass(state,slot,context->kv_block_table,&rows_view,rows_p);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen36ModuleRunMtpArgmaxRow(state,slot,rows_p - 1u,0u);
	for (step = 1; status == SPARK_STATUS_OK && step < view->draft_token_count; step++)
	{
		rows_view.slot_mapping = slot->slot_mapping + rows + (step - 1u);
		rows_view.row_positions = slot->row_positions + rows + (step - 1u);
		rows_view.row_lane_indices = slot->row_lane_indices + rows + (step - 1u);
		rows_view.context_lengths = slot->context_lengths + rows + (step - 1u);
		status = SparkQwen36ModuleRunMtpPackInput(state,slot,slot->mtp_draft_ids + (step - 1u),step == 1u ? (const void *)((const uint8_t *)slot->hidden_bf16 + ((uint64_t)(rows_p - 1u) * SPARK_QWEN36_MODEL_HIDDEN_BF16_BYTES)) : slot->hidden_bf16,1u);
		if ( status == SPARK_STATUS_OK )
			status = SparkQwen36ModuleRunMtpDecoderPass(state,slot,context->kv_block_table,&rows_view,1u);
		if ( status == SPARK_STATUS_OK )
			status = SparkQwen36ModuleRunMtpArgmaxRow(state,slot,0u,step);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,cudaMemcpyAsync((uint8_t *)frame->buffers[out_index].address + ((uint64_t)head_rows * sizeof(uint32_t)),slot->mtp_draft_ids,view->draft_token_count * sizeof(uint32_t),cudaMemcpyDeviceToHost,(cudaStream_t)slot->cuda_stream),"mtp_emit");
	return(status);
}

static SparkStatus SparkQwen36ModuleFinish(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot, SparkQwen36ResidentDecodeStageFrameContext *context, SparkModelDriverFrame *frame, const SparkQwen36PrefillFrameView *prefill, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint32_t emit_all = (context->flags & SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_SPECULATIVE_VERIFY) != 0u ? 1u : 0u;
	SparkStatus status = SPARK_STATUS_OK;
	if ( state->owns_final_head != 0u )
		status = SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,SparkQwen36ModuleEmitHead(state,slot,frame,prefill,emit_all,rows),"head");
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u && (context->flags & SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MTP_DRAFT_AFTER) != 0u )
		status = SparkQwen36ModuleRunMtpDraftChain(state,slot,context,frame,prefill,rows);
	if ( state->owns_final_head == 0u )
	{
		if ( context->hidden_output_send_function == 0 || context->hidden_output_transport_session == 0 )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		context->hidden_output_packet.active_sequence_count = rows;
		context->hidden_output_packet.hidden_dimension = SPARK_QWEN36_MODEL_HIDDEN_DIMENSION;
		context->hidden_output_packet.bytes_per_sequence = SPARK_QWEN36_MODEL_HIDDEN_DIMENSION * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES;
		context->hidden_output_packet.hidden_bf16 = slot->hidden_bf16;
		context->hidden_output_packet.cuda_stream = stream;
		context->hidden_output_packet.sideband_payload = 0;
		context->hidden_output_packet.sideband_kind = 0u;
		context->hidden_output_packet.sideband_bytes_per_sequence = 0u;
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,cudaStreamSynchronize(stream),"sync");
	if ( status == SPARK_STATUS_OK && state->owns_final_head == 0u )
		status = context->hidden_output_send_function(context->hidden_output_transport_session,&context->hidden_output_packet);
	return(status);
}

static uint64_t SparkQwen36ModuleFingerprint(const void *bytes, uint64_t count, uint64_t basis)
{
	const uint8_t *data = (const uint8_t *)bytes;
	uint64_t hash = basis,index;
	for (index = 0; index < count; index++)
		hash = (hash ^ data[index]) * 1099511628211ull;
	return(hash);
}

static SparkStatus SparkQwen36ModuleOpenKvTier(SparkQwen36ModuleState *state)
{
	SparkQwen36StagePackHeader geometry;
	const char *provider = 0,*service = 0,*socket_path = 0;
	uint64_t pool_bytes = 0u,model_fp,layout_fp,layout_bits[3];
	uint32_t workers = 0u;
	SparkStatus status = SparkStageModuleEnvironmentText(SPARK_QWEN36_MODULE_TAG,"SPARK_QWEN36_STAGE_KV_STORE",&provider);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( strcmp(provider,"none") == 0 )
		return(SparkStageKvClientOpen(&state->kv_client,SPARK_QWEN36_MODULE_TAG,provider,0u,0u,0u,0u,0u,0,0,0u,0u));
	status = SparkStageModuleEnvironmentText(SPARK_QWEN36_MODULE_TAG,"SPARK_QWEN36_STAGE_KV_SERVICE",&service);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentText(SPARK_QWEN36_MODULE_TAG,"SPARK_QWEN36_STAGE_KV_SOCKET",&socket_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned64(SPARK_QWEN36_MODULE_TAG,"SPARK_QWEN36_STAGE_KV_POOL_BYTES",1u,1ull << 40u,&pool_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN36_MODULE_TAG,"SPARK_QWEN36_STAGE_KV_WORKERS",1u,64u,&workers);
	if ( status != SPARK_STATUS_OK )
		return(status);
	SparkQwen36StagePackExpectedGeometry(&geometry,state->first_layer_index,state->layer_count);
	geometry.tp_degree = state->tp_degree;
	geometry.tp_rank = state->tp_rank;
	model_fp = SparkQwen36ModuleFingerprint(&geometry,sizeof(geometry),14695981039346656037ull);
	layout_bits[0] = state->cache_layer_stride;
	layout_bits[1] = state->cache_block_stride;
	layout_bits[2] = SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
	layout_fp = SparkQwen36ModuleFingerprint(layout_bits,sizeof(layout_bits),model_fp);
	return(SparkStageKvClientOpen(&state->kv_client,SPARK_QWEN36_MODULE_TAG,provider,state->stage_index,state->first_layer_index,state->layer_count,model_fp,layout_fp,service,socket_path,pool_bytes,workers));
}

static SparkStatus SparkQwen36ModuleValidateLaneSequenceContinuity(
    SparkQwen36ModuleState *state,
    const SparkQwen36ResidentDecodeStageFrameContext *context,
    const SparkQwen36PrefillFrameView *prefill,
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
        uint32_t restore_first;

        current_sequence_id = state->lane_sequence_ids[prefill->lane_index];
        expected_position = state->lane_next_positions[prefill->lane_index];
        restore_first = (context->flags &
            SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_RESTORE_FIRST) != 0u
            ? 1u
            : 0u;
        if (current_sequence_id == prefill->sequence_id)
        {
            if ((restore_first == 0u && prefill->base_position != expected_position) ||
                (restore_first != 0u && prefill->base_position > expected_position))
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
        }
        else
        {
            if (prefill->base_position != 0u)
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

static void SparkQwen36ModuleCommitLaneSequenceContinuity(
    SparkQwen36ModuleState *state,
    const SparkQwen36ResidentDecodeStageFrameContext *context,
    const SparkQwen36PrefillFrameView *prefill)
{
    uint32_t row;

    if (prefill != 0)
    {
        state->lane_sequence_ids[prefill->lane_index] = prefill->sequence_id;
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
        state->lane_next_positions[lane] = context->decode_batch->row_positions[row] + 1u;
        state->lane_warm[lane] = 1u;
    }
}

static void SparkQwen36ModuleInvalidateLaneSequenceContinuity(
    SparkQwen36ModuleState *state,
    const SparkQwen36ResidentDecodeStageFrameContext *context,
    const SparkQwen36PrefillFrameView *prefill)
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


/* Placeholder forward: launched when a DSPARK_DRAFT_AFTER frame completes its
 * decode taps. Filled in with the projector -> 5-layer -> lm_head -> Markov
 * sequence once the parity harness is in place (acceptance bar = draft parity). */
static SparkStatus SparkQwen36ModuleRunDsparkBlockForward(
	SparkQwen36ModuleState *state,
	SparkQwen36ModuleSlot *slot,
	const SparkQwen36DsparkDraftView *view,
	uint32_t rows)
{
	SparkQwen36DsparkWeights *w = &state->dspark_weights;
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint8_t *scr = (uint8_t *)slot->dspark_scratch;
	const uint32_t B = SPARK_QWEN36_DSPARK_BLOCK_SIZE;
	const uint32_t H = SPARK_QWEN36_MODEL_HIDDEN_DIMENSION;
	/* scratch carve-out (bf16 elements, 2 bytes each) */
	uint16_t *context = (uint16_t *)scr;
	uint16_t *block_hidden = context + H;
	uint16_t *q = block_hidden + (uint64_t)B * H;
	uint16_t *k = q + (uint64_t)B * H;
	uint16_t *v = k + (uint64_t)(B + 1u) * 1024u;
	uint16_t *attn_out = v + (uint64_t)(B + 1u) * 1024u;
	uint16_t *norm = attn_out + (uint64_t)B * H;
	uint16_t *ffn = norm + (uint64_t)B * H;
	uint16_t *up = ffn + (uint64_t)B * SPARK_QWEN36_DSPARK_FFN_INTERMEDIATE;
	uint16_t *logits = up + (uint64_t)B * SPARK_QWEN36_DSPARK_FFN_INTERMEDIATE;
	SparkStatus status;
	cudaError_t error;
	uint32_t layer;
	(void)rows;
	if ( w->armed == 0u )
		return(SPARK_STATUS_OK);
	if ( view == 0 || view->draft_token_ids == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	/* debug: dump first taps to confirm capture fired */
	{
		uint16_t tdbg[8];
		cudaMemcpy(tdbg,slot->dspark_tap_buffer,sizeof(tdbg),cudaMemcpyDeviceToHost);
		fprintf(stderr,"dspark_tap_dbg %u,%u,%u,%u %u,%u,%u,%u\n",tdbg[0],tdbg[1],tdbg[2],tdbg[3],tdbg[4],tdbg[5],tdbg[6],tdbg[7]);
	}
	/* 1) context = hidden_norm(fc(cat(5 taps))): one [H] vector, shared by every layer. */
	error = SparkQwen36LaunchLinear(stream,&w->projector,slot->dspark_tap_buffer,q,1u);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchRmsNorm(stream,q,w->hidden_norm_bf16,context,1u,H,SPARK_QWEN36_MODEL_RMS_NORM_EPSILON);
	/* 2) block[0] = embed(C0); block[1..B-1] = embed(mask_token_id). */
	/* C0 is the COMMITTED token the target just emitted (frame_output_ids[0]),
	 * not the frame's input token (slot->input_token_ids); EmitHead writes it to
	 * slot->output_token_ids before the DSpark forward runs. */
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchEmbeddingGather(stream,slot->output_token_ids,state->token_embedding_bf16,block_hidden,1u);
	if ( error == cudaSuccess )
		error = SparkQwen36LaunchEmbeddingGather(stream,slot->dspark_mask_token_ids,state->token_embedding_bf16,block_hidden + H,B - 1u);
	status = SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,error,"dspark_head_init");
	for (layer = 0u; status == SPARK_STATUS_OK && layer < SPARK_QWEN36_DSPARK_LAYER_COUNT; layer++)
	{
		SparkQwen36DsparkLayerWeights *lw = &w->layer[layer];
		error = SparkQwen36LaunchRmsNorm(stream,block_hidden,lw->input_norm_bf16,norm,B,H,SPARK_QWEN36_MODEL_RMS_NORM_EPSILON);
		/* attention_conv.prepare: kernel_projection(norm) -> delta, side-0 conv */
		if ( error == cudaSuccess )
			error = SparkQwen36LaunchLinear(stream,&lw->conv_attn_proj,norm,slot->dspark_conv_delta,B);
		if ( error == cudaSuccess )
			error = SparkQwen36LaunchDsparkConv(stream,norm,slot->dspark_conv_delta,lw->conv_attn_base,slot->dspark_conv_out,B,SPARK_QWEN36_MODEL_HIDDEN_DIMENSION/SPARK_QWEN36_DSPARK_CONV_GROUP_SIZE,SPARK_QWEN36_DSPARK_CONV_GROUP_SIZE,0u);
		if ( error == cudaSuccess )
			error = SparkQwen36LaunchLinear(stream,&lw->q,slot->dspark_conv_out,q,B);
		if ( error == cudaSuccess )
			error = SparkQwen36LaunchLinear(stream,&lw->k,context,k,1);
		if ( error == cudaSuccess )
			error = SparkQwen36LaunchLinear(stream,&lw->k,slot->dspark_conv_out,k + 1024u,B);
		if ( error == cudaSuccess )
			error = SparkQwen36LaunchLinear(stream,&lw->v,context,v,1);
		if ( error == cudaSuccess )
			error = SparkQwen36LaunchLinear(stream,&lw->v,slot->dspark_conv_out,v + 1024u,B);
		if ( error == cudaSuccess )
			error = SparkQwen36LaunchDsparkAttn(stream,q,k,v,lw->q_norm_bf16,lw->k_norm_bf16,attn_out,B,view->base_position);
		if ( error == cudaSuccess )
			error = SparkQwen36LaunchLinear(stream,&lw->o,attn_out,q,B);
		/* attention_conv.finish: side-1 conv, then residual add 1 */
		if ( error == cudaSuccess )
			error = SparkQwen36LaunchDsparkConv(stream,q,slot->dspark_conv_delta,(const void*)((const uint8_t*)lw->conv_attn_base + (SPARK_QWEN36_DSPARK_CONV_KERNEL_SIZE * SPARK_QWEN36_MODEL_HIDDEN_DIMENSION) * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES),slot->dspark_conv_out,B,SPARK_QWEN36_MODEL_HIDDEN_DIMENSION/SPARK_QWEN36_DSPARK_CONV_GROUP_SIZE,SPARK_QWEN36_DSPARK_CONV_GROUP_SIZE,1u);
		if ( error == cudaSuccess )
			error = SparkQwen36LaunchResidualAdd(stream,block_hidden,slot->dspark_conv_out,B,H);
		if ( error == cudaSuccess )
			error = SparkQwen36LaunchRmsNorm(stream,block_hidden,lw->post_norm_bf16,norm,B,H,SPARK_QWEN36_MODEL_RMS_NORM_EPSILON);
		/* mlp_conv.prepare: side-0 conv */
		if ( error == cudaSuccess )
			error = SparkQwen36LaunchLinear(stream,&lw->conv_mlp_proj,norm,slot->dspark_conv_delta,B);
		if ( error == cudaSuccess )
			error = SparkQwen36LaunchDsparkConv(stream,norm,slot->dspark_conv_delta,lw->conv_mlp_base,slot->dspark_conv_out,B,SPARK_QWEN36_MODEL_HIDDEN_DIMENSION/SPARK_QWEN36_DSPARK_CONV_GROUP_SIZE,SPARK_QWEN36_DSPARK_CONV_GROUP_SIZE,0u);
		if ( error == cudaSuccess )
			error = SparkQwen36LaunchLinear(stream,&lw->gate,slot->dspark_conv_out,ffn,B);
		if ( error == cudaSuccess )
			error = SparkQwen36LaunchLinear(stream,&lw->up,slot->dspark_conv_out,up,B);
		if ( error == cudaSuccess )
			error = SparkQwen36LaunchSwiGlu(stream,ffn,up,B,SPARK_QWEN36_DSPARK_FFN_INTERMEDIATE);
		if ( error == cudaSuccess )
			error = SparkQwen36LaunchLinear(stream,&lw->down,up,q,B);
		/* mlp_conv.finish: side-1 conv, then residual add 2 */
		if ( error == cudaSuccess )
			error = SparkQwen36LaunchDsparkConv(stream,q,slot->dspark_conv_delta,(const void*)((const uint8_t*)lw->conv_mlp_base + (SPARK_QWEN36_DSPARK_CONV_KERNEL_SIZE * SPARK_QWEN36_MODEL_HIDDEN_DIMENSION) * SPARK_QWEN36_MODEL_BF16_ELEMENT_BYTES),slot->dspark_conv_out,B,SPARK_QWEN36_MODEL_HIDDEN_DIMENSION/SPARK_QWEN36_DSPARK_CONV_GROUP_SIZE,SPARK_QWEN36_DSPARK_CONV_GROUP_SIZE,1u);
		if ( error == cudaSuccess )
			error = SparkQwen36LaunchResidualAdd(stream,block_hidden,slot->dspark_conv_out,B,H);
		status = SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,error,"dspark_layer");
	}
	/* 2) final norm + shared lm_head -> logits (B x vocab), then Markov bias. */
	if ( status == SPARK_STATUS_OK )
	{
		SparkQwen36LinearView lm_head;
		memset(&lm_head,0,sizeof(lm_head));
		lm_head.abi_version = SPARK_QWEN36_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION;
		lm_head.weight_format = 0u; /* BF16 */
		lm_head.input_dimension = SPARK_QWEN36_MODEL_HIDDEN_DIMENSION;
		lm_head.output_dimension = SPARK_QWEN36_MODEL_OUTPUT_VOCAB_COUNT;
		lm_head.weight_payload = state->lm_head_weight_bf16;
		lm_head.weight_payload_bytes = (uint64_t)SPARK_QWEN36_MODEL_HIDDEN_DIMENSION * SPARK_QWEN36_MODEL_OUTPUT_VOCAB_COUNT * 2u;
		error = SparkQwen36LaunchRmsNorm(stream,block_hidden,w->final_norm_bf16,norm,B,H,SPARK_QWEN36_MODEL_RMS_NORM_EPSILON);
		if ( error == cudaSuccess )
			error = SparkQwen36LaunchLinear(stream,&lm_head,norm,logits,B);
		/* D2H the B x vocab base logits for the parity dump (the selector reads the
		 * final-norm hidden directly and computes top-16 without materializing logits). */
		if ( error == cudaSuccess )
			error = cudaMemcpyAsync(slot->dspark_logits_host,logits,(size_t)B * SPARK_QWEN36_MODEL_OUTPUT_VOCAB_COUNT * sizeof(uint16_t),cudaMemcpyDeviceToHost,stream);
		if ( error == cudaSuccess )
			error = cudaStreamSynchronize(stream);
		if ( error == cudaSuccess )
		{
			static int dspark_dump_done = 0;
			if ( dspark_dump_done == 0 )
			{
				dspark_dump_done = 1;
				FILE *df;
				uint16_t *taps_host = (uint16_t *)malloc((size_t)5u * H * 2u);
				cudaMemcpy(taps_host,slot->dspark_tap_buffer,(size_t)5u * H * 2u,cudaMemcpyDeviceToHost);
				df = fopen("/tmp/dspark_taps.bin","wb"); fwrite(taps_host,1,(size_t)5u * H * 2u,df); fclose(df); free(taps_host);
				uint32_t c0;
				cudaMemcpy(&c0,slot->output_token_ids,4u,cudaMemcpyDeviceToHost);
				df = fopen("/tmp/dspark_c0.bin","wb"); fwrite(&c0,1,4u,df); fclose(df);
				df = fopen("/tmp/dspark_base.bin","wb"); fwrite(slot->dspark_logits_host,1,(size_t)B * SPARK_QWEN36_MODEL_OUTPUT_VOCAB_COUNT * 2u,df); fclose(df);
				uint64_t base_pos = (uint64_t)view->base_position;
				df = fopen("/tmp/dspark_basepos.bin","wb"); fwrite(&base_pos,1,8u,df); fclose(df);
				fprintf(stderr,"dspark_dump c0=%u base_pos=%llu\n",c0,(unsigned long long)base_pos);
			}
		}
		/* DFlash2 candidate selector (W4 top-16 -> W3 gate/lattice/walk): draft over
		 * the (B-1) mask positions, anchor = the committed token (output_token_ids[0]). */
		if ( error == cudaSuccess )
			error = SparkQwen36LaunchDsparkHeadTopK(stream,&lm_head,norm + H,slot->dspark_selector_chunk_keys,slot->dspark_selector_candidate_ids,slot->dspark_selector_unary,0,B - 1u,0u,SPARK_QWEN36_DSPARK_SELECTOR_TOP_K);
		if ( error == cudaSuccess )
			error = SparkQwen36LaunchDsparkSelector(stream,norm + H,w->confidence.weight_payload,w->markov_w1.weight_payload,w->markov_w2.weight_payload,slot->dspark_selector_candidate_ids,slot->output_token_ids,slot->dspark_selector_unary,slot->dspark_selector_gate,slot->dspark_selector_edges,view->draft_token_ids,slot->dspark_selector_slots,1u,B - 1u,SPARK_QWEN36_DSPARK_SELECTOR_TOP_K,SPARK_QWEN36_DSPARK_SELECTOR_RANK,H);
		status = SparkStageModuleCudaStatus(SPARK_QWEN36_MODULE_TAG,error,"dspark_head");
	}
	return(status);
}

static SparkStatus SparkQwen36ModuleCaptureDsparkTap(
	SparkQwen36ModuleState *state,
	SparkQwen36ModuleSlot *slot,
	uint32_t layer)
{
	uint32_t tap_index;
	(void)state;
	switch ( layer )
	{
	case 5u: tap_index = 0u; break;
	case 19u: tap_index = 1u; break;
	case 33u: tap_index = 2u; break;
	case 47u: tap_index = 3u; break;
	case 61u: tap_index = 4u; break;
	default: return(SPARK_STATUS_OK);
	}
	return(SparkStageModuleCudaStatus(
		SPARK_QWEN36_MODULE_TAG,
		cudaMemcpyAsync(
			(uint8_t *)slot->dspark_tap_buffer + (uint64_t)tap_index * SPARK_QWEN36_MODEL_HIDDEN_BF16_BYTES,
			slot->hidden_bf16,
			SPARK_QWEN36_MODEL_HIDDEN_BF16_BYTES,
			cudaMemcpyDeviceToDevice,
			(cudaStream_t)slot->cuda_stream),
		"dspark_tap"));
}

static SparkStatus SparkQwen36ModuleRunFrame(SparkQwen36ModuleState *state, SparkQwen36ModuleSlot *slot, SparkQwen36ResidentDecodeStageFrameContext *context, SparkModelDriverFrame *frame, const SparkQwen36PrefillFrameView *prefill, uint32_t rows)
{
	uint64_t frame_start = state->profile_enabled != 0u ? SparkQwen36ProfileNow() : 0ull;
	SparkStatus status = SparkQwen36ModuleUploadRows(state,slot,context,frame,prefill,rows);
	uint32_t layer;
	if ( status == SPARK_STATUS_OK && (context->flags & SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_SPECULATIVE_VERIFY) != 0u )
		status = SparkQwen36ModuleGdnSnapshot(state,slot,prefill->lane_index,context->gdn_snapshot->snapshot_index,0u);
	if ( status == SPARK_STATUS_OK && (context->flags & SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_GDN_RESTORE_FIRST) != 0u )
		status = SparkQwen36ModuleGdnSnapshot(state,slot,prefill->lane_index,context->gdn_snapshot->snapshot_index,1u);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen36ModuleBeginHidden(state,slot,context,rows);
	for (layer = state->first_layer_index; status == SPARK_STATUS_OK && layer < state->first_layer_index + state->layer_count; layer++)
	{
		status = SparkQwen36ModuleRunLayer(state,slot,context->kv_block_table,prefill,layer,rows);
		if ( status == SPARK_STATUS_OK && (context->flags & SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DSPARK_DRAFT_AFTER) != 0u )
			status = SparkQwen36ModuleCaptureDsparkTap(state,slot,layer);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen36ModuleFinish(state,slot,context,frame,prefill,rows);
	if ( state->profile_enabled != 0u )
		SparkQwen36ProfilePrint(state, SparkQwen36ProfileNow() - frame_start);
	return(status);
}

SparkStatus SparkQwen36ResidentDecodeStageExecute(
    void *module_state,
    SparkModelDriverFrame *frame)
{
    SparkQwen36ModuleState *state;
    SparkQwen36ResidentDecodeStageFrameContext *context;
    const SparkQwen36PrefillFrameView *prefill;
    const uint32_t *claimed_lane_indices;
    uint32_t prefill_lane_index;
    uint8_t lane_requires_reset[
        SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
    uint8_t lane_used[
        SPARK_QWEN36_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
    SparkQwen36ModuleSlot *slot;
    uint32_t claimed_lane_count;
    uint32_t slot_index;
    uint32_t rows;
    uint32_t row;
    uint32_t lanes_claimed;
    SparkStatus status;

    state = (SparkQwen36ModuleState *)module_state;
    if (state == 0 || frame == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    context = 0;
    status = SparkQwen36ModuleValidateFrame(
        state,
        frame,
        (const SparkQwen36ResidentDecodeStageFrameContext **)&context);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "%s frame_validate_failed status=%d frame_flags=0x%x buffers=%u tps=%u new_tokens=%u active_slots=%u seq_pos=%" PRIu64 " program_id=%u exp_buffers=%u\n", SPARK_QWEN36_MODULE_TAG, (int)status, frame->flags, frame->buffer_count, frame->tokens_per_sequence, frame->new_token_count, frame->active_slot_count, frame->sequence_position, frame->program_id, state->owns_embedding + state->owns_final_head);
        atomic_fetch_add_explicit(
            &state->rejected_count,
            1u,
            memory_order_relaxed);
        return status;
    }

    prefill = (context->flags &
        SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW) != 0u
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
        status = SparkQwen36ModuleValidateLaneSequenceContinuity(
            state,
            context,
            prefill,
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
            status = SparkQwen36ModuleResetLaneState(
                state,
                slot,
                claimed_lane_indices[row]);
        }
    }
    if (status == SPARK_STATUS_OK && prefill != 0)
    {
        status = SparkQwen36ModulePrefillStage(state, slot, context);
    }
    else if (status == SPARK_STATUS_OK)
    {
        memset(lane_used, 0, sizeof(lane_used));
        status = SparkQwen36ModuleStageRows(state, slot, context, lane_used);
    }
    if (status == SPARK_STATUS_OK &&
        (context->flags &
         SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MTP_DRAFT_AFTER) != 0u)
    {
        status = SparkQwen36ModuleStageMtpDraft(
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
        status = SparkQwen36ModuleRunFrame(
            state,
            slot,
            context,
            frame,
            prefill,
            rows);
    }
    if (status == SPARK_STATUS_OK && (context->flags & SPARK_QWEN36_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DSPARK_DRAFT_AFTER) != 0u)
        status = SparkQwen36ModuleRunDsparkBlockForward(state,slot,context->dspark_draft,rows);
    if (status == SPARK_STATUS_OK)
    {
        SparkQwen36ModuleCommitLaneSequenceContinuity(
            state,
            context,
            prefill);
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
        SparkQwen36ModuleInvalidateLaneSequenceContinuity(
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

static void SparkQwen36AdmissionCost(
    void *context,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision)
{
    SparkQwen36ModuleState *state = (SparkQwen36ModuleState *)context;
    (void)state;
    decision->host_staging_bytes = (uint64_t)request->new_token_count *
        (sizeof(uint32_t) *
             (uint64_t)(state->owns_embedding + state->owns_final_head + 3u) +
         sizeof(uint64_t));
    decision->device_memcpy_bytes = decision->host_staging_bytes;
}

static SparkStatus SparkQwen36AdmissionKvPredicate(
    void *context,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision)
{
    (void)context;
    if ((request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u &&
        SparkModelDriverRangeFitsWithinCapacity(
            request->sequence_position,
            request->new_token_count,
            SPARK_QWEN36_MODEL_MAXIMUM_CONTEXT_TOKENS) == 0u)
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

SparkStatus SparkQwen36ResidentDecodeStageAdmit(
    void *module_state,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision)
{
    SparkQwen36ModuleState *state;
    SparkAdmissionPolicyTable table;
    uint32_t available_slot_count;
    SparkStatus status;

    state = (SparkQwen36ModuleState *)module_state;
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
    table.max_sequence_positions = SPARK_QWEN36_MODEL_MAXIMUM_CONTEXT_TOKENS;
    table.flags = SPARK_ADMISSION_POLICY_FLAG_PREFILL_SINGLE_SLOT |
        SPARK_ADMISSION_POLICY_FLAG_DECODE_EQUALS_SLOTS;
    table.predicate = SparkQwen36AdmissionKvPredicate;
    table.predicate_context = state;
    table.cost = SparkQwen36AdmissionCost;
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

SparkStatus SparkQwen36ResidentDecodeStageSnapshot(
    void *module_state,
    uint32_t program_id,
    SparkModelDriverRuntimeSnapshot *snapshot)
{
    SparkQwen36ModuleState *state;

    state = (SparkQwen36ModuleState *)module_state;
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
    snapshot->kv_token_capacity = (uint64_t)state->kv_block_count * SPARK_QWEN36_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
    return SPARK_STATUS_OK;
}

void SparkQwen36ResidentDecodeStageDestroy(void *module_state)
{
    SparkQwen36ModuleState *state;
    uint32_t slot_index;

    state = (SparkQwen36ModuleState *)module_state;
    if (state == 0)
    {
        return;
    }
    if (SparkStageModuleWaitForSlots(
            SPARK_QWEN36_MODULE_TAG,
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
    }
    SparkQwen36TpDestroy(&state->tp);
    if ( state->tp_stream != 0 )
        cudaStreamDestroy((cudaStream_t)state->tp_stream);
    if (state->dspark_weights.markov_w1_host != 0)
        free(state->dspark_weights.markov_w1_host);
    if (state->dspark_weights.markov_w2_host != 0)
        free(state->dspark_weights.markov_w2_host);
    SparkStageKvClientClose(&state->kv_client);
    SparkStageModuleLedgerRelease(&state->ledger);
    free(state);
}

SparkStatus SparkQwen36ResidentDecodeStageInitialize(
    const SparkFirmwareModuleConfiguration *configuration,
    const SparkFirmwareModuleHostServices *host_services,
    void **module_state)
{
    SparkQwen36ModuleState *state;
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
        SPARK_QWEN36_MODULE_TAG,
        "SPARK_QWEN36_ALLOW_UNQUALIFIED_EXECUTION",
        1u,
        1u,
        &allow_unqualified_execution);
    if (status != SPARK_STATUS_OK || allow_unqualified_execution != 1u)
    {
        return SPARK_STATUS_MODULE_NOT_VALIDATED;
    }

    state = (SparkQwen36ModuleState *)calloc(1u, sizeof(*state));
    if (state == 0)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    state->ledger.module_tag = SPARK_QWEN36_MODULE_TAG;
    {
        const char *profile_env = getenv("SPARK_QWEN36_PROFILE");
        state->profile_enabled = profile_env != 0 && strcmp(profile_env, "0") != 0 ? 1u : 0u;
    }
    atomic_init(&state->submitted_count, 0u);
    atomic_init(&state->completed_count, 0u);
    atomic_init(&state->rejected_count, 0u);
    atomic_init(&state->failed_count, 0u);
    atomic_init(&state->tokens_emitted, 0u);

    status = SparkQwen36ModuleConfigure(state);
    if (status == SPARK_STATUS_OK)
    {
        status = SparkStageModuleCudaStatus(
            SPARK_QWEN36_MODULE_TAG,
            SparkQwen36ConfigureCudaKernels(),
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
            SPARK_QWEN36_MODULE_TAG,
            "SPARK_QWEN36_STAGE_PACK_PATH",
            &pack_path);
    }
    if (status == SPARK_STATUS_OK)
    {
        SparkQwen36ModuleBuildOrdinals(state);
        status = SparkQwen36ModuleLoadPack(state, pack_path);
    }
    if (status == SPARK_STATUS_OK)
    {
        /* Optional: the DSpark drafter pack is loaded only when the env var
         * is set (spec-method dspark). getenv (not the required-text helper)
         * so a no-spec / MTP deploy without the var still initializes. */
        const char *dspark_path = getenv("SPARK_QWEN36_DSPARK_PACK_PATH");
        if ( dspark_path != 0 && dspark_path[0] != '\0' )
            status = SparkQwen36ModuleLoadDsparkPack(state, dspark_path);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkQwen36ModuleInitializeTp(state);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkQwen36ModuleAllocatePools(state);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkQwen36ModuleBuildHeadShadow(state);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkQwen36ModuleOpenKvTier(state);
    }
    for (slot_index = 0u;
         status == SPARK_STATUS_OK && slot_index < state->pipeline_slot_count;
         slot_index++)
    {
        status = SparkQwen36ModuleAllocateSlot(state, &state->slots[slot_index]);
    }
    if (status != SPARK_STATUS_OK)
    {
        SparkQwen36ResidentDecodeStageDestroy(state);
        return status;
    }

    fprintf(
        stderr,
        "%s ready stage=%u/%u slice=%u+%u tp=%u/%u gdn=%u attn=%u lanes=%u kv_blocks=%u device_gib=%.1f\n",
        SPARK_QWEN36_MODULE_TAG,
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
