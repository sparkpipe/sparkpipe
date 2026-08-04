/* Large stage packs exceed 2 GB: 64-bit file offsets are required. */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_dsv4_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "spark_dsv4_stagepack_format.h"

/*
 * DeepSeek V4 resident decode stage host module, PP-Nx native, one variant
 * per build through the -include'd model header.
 *
 * One process is one STAGE over a layer slice; the pack must declare that
 * slice and the computed tensor inventory exactly. The stage boundary
 * carries the FOUR hyper-connection streams (hc_mult x hidden per row);
 * stage zero expands the embedding, the head stage's sigmoid reduction is
 * the only collapse. Execute serves DECODE frames: one token per lane
 * across up to max_active lanes, every attention kind, both router paths,
 * the full mHC chain. Prefill advances one round-major wave at a time, so
 * all live sequences at one prompt step share the same CUDA launch while
 * preserving each sequence's state dependency. A causal bulk-prefill kernel
 * can replace the wavefront after separate qualification.
 * MTP execution remains refused until its pass lands; the MTP tensors load
 * and verify so the pack contract is already final.
 *
 * The hash router pins to token ids, which exist only where the embedding
 * lives: a slice that starts inside the hash range without owning the
 * embedding is refused at configuration, not discovered at runtime.
 *
 * Caches are dense per lane: window ring 128 slots always; compress
 * layers append max_seq/ratio compressed slots behind the window in the
 * same lane block (the reference's win offset); CSA layers keep an
 * indexer stream of max_seq/4 rotated 128-wide entries and both
 * compressor f32 states. The paged migration rides the family PP pass.
 */

#define SPARK_DSV4_MODULE_TAG "dsv4_stage"

typedef struct SparkDsv4LayerWeights
{
	const void *attn_norm_bf16;
	const void *ffn_norm_bf16;
	SparkDsv4AttnWeights attn;
	SparkDsv4CompressorWeights compressor;
	SparkDsv4IndexerWeights indexer;
	SparkDsv4MoeWeights moe;
	SparkDsv4HcWeights hc;
} SparkDsv4LayerWeights;

typedef struct SparkDsv4ModuleSlot
{
	void *cuda_stream;
	uint32_t *input_token_ids;
	uint32_t *output_token_ids;
	uint32_t *row_lane_indices;
	uint64_t *row_positions;
	uint64_t *row_emit_positions;
	uint64_t *row_emit_positions_hca;
	uint32_t *slot_counts;
	int32_t *topk_idxs;
	void *streams_bf16;
	void *residual_bf16;
	void *reduced_bf16;
	void *normalized_bf16;
	void *qr_bf16;
	void *q_bf16;
	void *kv_bf16;
	void *attn_out_bf16;
	void *o_ranks_bf16;
	void *delta_bf16;
	void *compress_kv_bf16;
	void *compress_score_bf16;
	float *compress_kv_f32;
	float *compress_score_f32;
	void *emit_bf16;
	uint32_t *emitted_u32;
	void *index_q_bf16;
	void *index_weights_bf16;
	float *index_weights_f32;
	float *index_scores_f32;
	float *mixes_f32;
	float *pre_f32;
	float *post_f32;
	float *comb_f32;
	float *moe_scores_f32;
	uint32_t *moe_indices_u32;
	float *moe_weights_f32;
	void *ffn_gate_bf16;
	void *ffn_up_bf16;
	void *ffn_delta_bf16;
	void *ffn_accum_bf16;
	uint32_t *grouped_rows_u32;
	uint32_t *grouped_weight_slots_u32;
	void *moe_slot_gate_bf16;
	void *moe_slot_up_bf16;
	void *moe_slot_out_bf16;
	void *head_logits_bf16;
	uint32_t *head_candidate_ids_u32;
	uint32_t *head_candidate_counts_u32;
	uint32_t *expert_offsets_u32;
	uint32_t *moe_inverse_u32;
	int32_t *host_topk_indices;
	uint32_t host_slot_counts[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
} SparkDsv4ModuleSlot;

typedef struct SparkDsv4ModuleState
{
	SparkStageModuleLedger ledger;
	uint32_t stage_count;
	uint32_t stage_index;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t owns_embedding;
	uint32_t owns_final_head;
	uint32_t resident_sequence_capacity;
	uint32_t pipeline_slot_count;
	uint32_t max_sequence_positions;
	uint32_t mtp_armed;
	uint32_t compress_layer_count;
	uint32_t csa_layer_count;
	uint32_t topk_column_count;
	uint32_t index_slot_capacity;
	void *execution_stream;
	float hc_head_scale_value;
	uint32_t layer_local_by_layer[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT];
	uint32_t compress_ordinal_by_layer[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT];
	uint32_t csa_ordinal_by_layer[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT];
	uint64_t layer_seen_bits[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT];
	uint64_t mtp_seen_bits;
	uint64_t global_seen_bits;
	SparkDsv4LayerWeights layers[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT];
	SparkDsv4LayerWeights mtp_layer;
	SparkDsv4MtpWeights mtp;
	const void *token_embedding_bf16;
	const void *final_norm_weight_bf16;
	const void *lm_head_weight_bf16;
	uint8_t *head_shadow_payload;
	uint8_t *head_shadow_scale;
	float *head_error_norm_f32;
	const float *hc_head_fn_f32;
	const float *hc_head_base_f32;
	const float *hc_head_scale_f32;
	float *base_freqs_f32;
	float *compress_freqs_f32;
	void *kv_cache_bf16;
	uint64_t cache_layer_lane_stride;
	uint64_t cache_lane_block_elements;
	void *index_cache_bf16;
	uint64_t index_lane_stride;
	float *compress_kv_state_f32;
	float *compress_score_state_f32;
	uint64_t compress_state_lane_stride;
	float *index_kv_state_f32;
	float *index_score_state_f32;
	uint64_t index_state_lane_stride;
	SparkDsv4ModuleSlot slots[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	atomic_uint slot_states[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	atomic_uint lane_states[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_sequence_ids[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_next_positions[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	atomic_ullong submitted_count;
	atomic_ullong completed_count;
	atomic_ullong rejected_count;
	atomic_ullong failed_count;
	atomic_ullong tokens_emitted;
} SparkDsv4ModuleState;

extern cudaError_t SparkDsv4ConfigureCudaKernels(void);
extern cudaError_t SparkDsv4LaunchRmsNorm(cudaStream_t stream, const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon);
extern cudaError_t SparkDsv4LaunchLinear(cudaStream_t stream, const SparkDsv4LinearView *view, const void *input_bf16, void *output_bf16, uint32_t row_count);
extern cudaError_t SparkDsv4LaunchStridedLinear(cudaStream_t stream, const SparkDsv4LinearView *view, const void *payload, const uint8_t *scale, const void *input_bf16, uint64_t input_row_stride, uint32_t input_offset, void *output_bf16, uint64_t output_row_stride, uint32_t output_offset, uint32_t row_count);
extern cudaError_t SparkDsv4LaunchEmbeddingGather(cudaStream_t stream, const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count, uint32_t hidden_dimension);
extern cudaError_t SparkDsv4LaunchHeadArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint32_t *token_ids, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count, uint32_t hidden_dimension);
extern cudaError_t SparkDsv4LaunchHeadShadowQuantize(cudaStream_t stream, const void *head_bf16, uint8_t *shadow_payload, uint8_t *shadow_scale, float *error_norm, uint32_t candidate_count, uint32_t hidden_dimension);
extern cudaError_t SparkDsv4LaunchHeadScreenedArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint8_t *shadow_payload, const uint8_t *shadow_scale, const float *error_norm, void *logits_bf16, uint32_t *candidate_ids, uint32_t *candidate_counts, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count, uint32_t hidden_dimension);
extern cudaError_t SparkDsv4LaunchQuantSim(cudaStream_t stream, void *data_bf16, uint32_t row_count, uint32_t row_stride, uint32_t width, uint32_t block, uint32_t fp4);
extern cudaError_t SparkDsv4LaunchRope(cudaStream_t stream, void *data_bf16, const float *freqs_f32, const uint64_t *row_positions, uint32_t row_count, uint32_t head_count, uint32_t head_dim, uint32_t rope_dim, uint32_t inverse);
extern cudaError_t SparkDsv4LaunchQueryHeadRms(cudaStream_t stream, void *data_bf16, uint32_t row_count, uint32_t head_count, uint32_t head_dim, float epsilon);
extern cudaError_t SparkDsv4LaunchHadamard(cudaStream_t stream, void *data_bf16, uint32_t vector_count, uint32_t width);
extern cudaError_t SparkDsv4LaunchSparseAttn(cudaStream_t stream, const void *q_bf16, const void *kv_cache_bf16, uint64_t lane_stride_elements, const uint32_t *row_lane_indices, const int32_t *topk_idxs, uint32_t topk, const float *sink_f32, float scale, void *out_bf16, uint32_t row_count, uint32_t head_count, uint32_t head_dim);
extern cudaError_t SparkDsv4LaunchWiden(cudaStream_t stream, const void *input_bf16, float *output_f32, uint32_t row_count, uint32_t width, float scale);
extern cudaError_t SparkDsv4LaunchApeAdd(cudaStream_t stream, float *score_f32, const float *ape_f32, const uint64_t *row_positions, uint32_t row_count, uint32_t ratio, uint32_t channels);
extern cudaError_t SparkDsv4LaunchCompressStep(cudaStream_t stream, const float *kv_f32, const float *score_f32, float *kv_state_f32, float *score_state_f32, uint64_t state_lane_stride, const uint32_t *row_lane_indices, const uint64_t *row_positions, uint32_t row_count, uint32_t ratio, uint32_t overlap, uint32_t width, void *emit_bf16, uint32_t *emitted);
extern cudaError_t SparkDsv4LaunchGateScores(cudaStream_t stream, const SparkDsv4LinearView *gate, const void *input_bf16, float *scores_f32, uint32_t row_count);
extern cudaError_t SparkDsv4LaunchGateSelect(cudaStream_t stream, const float *scores_f32, const float *bias_f32, const uint32_t *tid2eid_u32, const uint32_t *token_ids, uint32_t row_count, uint32_t expert_count, uint32_t topk, float route_scale, uint32_t *indices_u32, float *weights_f32);
extern cudaError_t SparkDsv4LaunchSwigluClamp(cudaStream_t stream, const void *gate_bf16, void *up_bf16, uint32_t row_count, uint32_t width, float limit, const float *row_weights_f32, const uint32_t *weight_map);
extern cudaError_t SparkDsv4LaunchGatherLinear(cudaStream_t stream, const SparkDsv4LinearView *view, const void *input_bf16, const uint32_t *input_row_map, void *output_bf16, uint32_t slot_count);
extern cudaError_t SparkDsv4LaunchValidateTid2Eid(cudaStream_t stream, const uint32_t *tid2eid, uint64_t entry_count, uint32_t *violation_flag);
extern cudaError_t SparkDsv4LaunchMoeGroup(cudaStream_t stream, const uint32_t *pair_expert_ids, uint32_t pair_count, uint32_t *expert_offsets, uint32_t *grouped_rows, uint32_t *grouped_weight_slots, uint32_t *inverse_map);
extern cudaError_t SparkDsv4LaunchExpertTileAll(cudaStream_t stream, const SparkDsv4LinearView *stacked, const void *input_bf16, const uint32_t *grouped_rows, const uint32_t *expert_offsets, void *output_bf16, uint32_t max_group_slots, uint64_t rows_per_expert, uint64_t columns);
extern cudaError_t SparkDsv4LaunchMoePairReduce(cudaStream_t stream, const void *slot_out_bf16, const uint32_t *inverse_map, void *accum_bf16, uint32_t row_count);
extern cudaError_t SparkDsv4LaunchExpertTile(cudaStream_t stream, const SparkDsv4LinearView *view, const void *input_bf16, const uint32_t *input_row_map, void *output_bf16, uint32_t slot_count);
extern cudaError_t SparkDsv4LaunchScatterAdd(cudaStream_t stream, void *destination_bf16, const void *source_bf16, const uint32_t *row_map, uint32_t slot_count, uint32_t width);
extern cudaError_t SparkDsv4LaunchAccumAdd(cudaStream_t stream, void *destination_bf16, const void *source_bf16, uint32_t row_count, uint32_t width);
extern cudaError_t SparkDsv4LaunchIndexerScore(cudaStream_t stream, const void *q_bf16, const void *kv_cache_bf16, uint64_t lane_stride_elements, const uint32_t *row_lane_indices, const uint32_t *slot_counts, const float *head_weights_f32, float *scores_f32, uint32_t row_count, uint32_t max_slots, uint32_t head_count, uint32_t head_dim);
extern cudaError_t SparkDsv4LaunchTopK(cudaStream_t stream, float *scores_f32, const uint32_t *slot_counts, uint32_t max_slots, uint32_t topk, int32_t offset, int32_t *indices_out, uint64_t out_row_stride, uint32_t row_count);
extern cudaError_t SparkDsv4LaunchHcMix(cudaStream_t stream, const void *streams_bf16, const float *fn_f32, float *mixes_f32, uint32_t row_count, uint32_t flat_dimension, uint32_t mix_rows, float epsilon);
extern cudaError_t SparkDsv4LaunchHcSplitSinkhorn(cudaStream_t stream, const float *mixes_f32, const float *scale3_f32, const float *base_f32, uint32_t row_count, uint32_t hc, uint32_t iterations, float epsilon, float *pre_f32, float *post_f32, float *comb_f32);
extern cudaError_t SparkDsv4LaunchHcPreReduce(cudaStream_t stream, const void *streams_bf16, const float *pre_f32, void *reduced_bf16, uint32_t row_count, uint32_t hc, uint32_t dimension);
extern cudaError_t SparkDsv4LaunchHcPost(cudaStream_t stream, const void *out_bf16, const void *residual_bf16, const float *post_f32, const float *comb_f32, void *streams_bf16, uint32_t row_count, uint32_t hc, uint32_t dimension);
extern cudaError_t SparkDsv4LaunchHcHeadReduce(cudaStream_t stream, const void *streams_bf16, const float *mixes_f32, float scale, const float *base_f32, float epsilon, void *reduced_bf16, uint32_t row_count, uint32_t hc, uint32_t dimension);

static SparkStatus SparkDsv4ModuleConfigure(
	SparkDsv4ModuleState *state,
	const SparkFirmwareModuleHostServices *host_services,
	const char **pack_path_out)
{
	const SparkDsv4ResidentDecodeStageNodeContext *context;
	if ( state == 0 || host_services == 0 || pack_path_out == 0 ||
		host_services->node_context == 0 || host_services->execution_stream == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	context = (const SparkDsv4ResidentDecodeStageNodeContext *)host_services->node_context;
	if ( context->abi_version != SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION || context->descriptor_bytes != SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( (context->flags & ~SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_KNOWN_FLAGS) != 0u || context->reserved0 != 0u || context->reserved1 != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (context->flags & SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_FLAG_ALLOW_UNQUALIFIED) == 0u )
		return(SPARK_STATUS_MODULE_NOT_VALIDATED);
	if ( context->stage_count == 0u || context->stage_count > SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT || context->stage_index >= context->stage_count || context->first_layer_index >= SPARK_DSV4_MODEL_LAYER_COUNT || context->layer_count == 0u || context->layer_count > SPARK_DSV4_MODEL_LAYER_COUNT - context->first_layer_index || context->resident_sequence_capacity == 0u || context->resident_sequence_capacity > SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT || context->pipeline_slot_count == 0u || context->pipeline_slot_count > SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT || context->max_sequence_positions < SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO || context->max_sequence_positions > SPARK_DSV4_MODEL_MAX_POSITIONS || context->stage_pack_path == 0 || context->stage_pack_path[0] == '\0' )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	state->stage_count = context->stage_count;
	state->stage_index = context->stage_index;
	state->first_layer_index = context->first_layer_index;
	state->layer_count = context->layer_count;
	state->resident_sequence_capacity = context->resident_sequence_capacity;
	state->pipeline_slot_count = context->pipeline_slot_count;
	state->max_sequence_positions = context->max_sequence_positions;
	state->execution_stream = host_services->execution_stream;
	state->mtp_armed = 0u;
	*pack_path_out = context->stage_pack_path;
	return(SPARK_STATUS_OK);
}

// The slice sanity beyond ranges: position agreement, and the hash-router
// pin - token ids exist only beside the embedding, so a slice that starts
// inside the hash range without layer zero cannot route and is refused.
static SparkStatus SparkDsv4ModuleValidateSlice(SparkDsv4ModuleState *state)
{
	if ( state->stage_index >= state->stage_count || state->first_layer_index + state->layer_count > SPARK_DSV4_MODEL_LAYER_COUNT )
	{
		fprintf(stderr,"%s config_slice_invalid stage=%u/%u slice=%u+%u\n",SPARK_DSV4_MODULE_TAG,state->stage_index,state->stage_count,state->first_layer_index,state->layer_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	state->owns_embedding = state->first_layer_index == 0u ? 1u : 0u;
	state->owns_final_head = state->first_layer_index + state->layer_count == SPARK_DSV4_MODEL_LAYER_COUNT ? 1u : 0u;
	if ( (state->stage_index == 0u) != (state->owns_embedding != 0u) || (state->stage_index + 1u == state->stage_count) != (state->owns_final_head != 0u) )
	{
		fprintf(stderr,"%s config_position_mismatch stage=%u/%u slice=%u+%u\n",SPARK_DSV4_MODULE_TAG,state->stage_index,state->stage_count,state->first_layer_index,state->layer_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( state->owns_embedding == 0u && state->first_layer_index < SPARK_DSV4_MODEL_HASH_ROUTED_LAYER_COUNT )
	{
		fprintf(stderr,"%s config_hash_layer_without_tokens slice=%u+%u\n",SPARK_DSV4_MODULE_TAG,state->first_layer_index,state->layer_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	return(SPARK_STATUS_OK);
}

// Per-stage ordinals: dense compress and CSA numbering inside the slice,
// and the topk column budget - the window plus the larger of the indexer
// top-k and the full compressed slot count an HCA layer attends.
static void SparkDsv4ModuleBuildOrdinals(SparkDsv4ModuleState *state)
{
	uint32_t layer,kind,hca_columns = state->max_sequence_positions / SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO;
	for (layer = 0; layer < SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LAYER_COUNT; layer++)
	{
		state->layer_local_by_layer[layer] = UINT32_MAX;
		state->compress_ordinal_by_layer[layer] = UINT32_MAX;
		state->csa_ordinal_by_layer[layer] = UINT32_MAX;
	}
	for (layer = state->first_layer_index; layer < state->first_layer_index + state->layer_count; layer++)
	{
		kind = SparkDsv4ModelLayerKind(layer);
		state->layer_local_by_layer[layer] = layer - state->first_layer_index;
		if ( kind != SPARK_DSV4_MODEL_LAYER_KIND_SWA )
		{
			state->compress_ordinal_by_layer[layer] = state->compress_layer_count++;
			state->layers[layer].compressor.ratio = kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA ? SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO : SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO;
			state->layers[layer].compressor.overlap = kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA ? SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR : 1u;
		}
		if ( kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA )
		{
			state->csa_ordinal_by_layer[layer] = state->csa_layer_count++;
			state->layers[layer].indexer.compressor.ratio = SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO;
			state->layers[layer].indexer.compressor.overlap = SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR;
		}
	}
	state->index_slot_capacity = state->max_sequence_positions / SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO;
	state->topk_column_count = SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS + (SPARK_DSV4_MODEL_INDEX_TOP_K > hca_columns ? SPARK_DSV4_MODEL_INDEX_TOP_K : hca_columns);
}

static void SparkDsv4ModuleFillLinearView(SparkDsv4LinearView *view, const SparkDsv4StagePackEntry *entry, void *payload, void *scale)
{
	view->abi_version = SPARK_DSV4_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION;
	view->weight_format = entry->weight_format;
	view->rows = entry->rows;
	view->columns = entry->columns;
	view->payload = payload;
	view->scale_e8m0 = (const uint8_t *)scale;
}

static SparkStatus SparkDsv4ModuleValidateEntry(SparkDsv4ModuleState *state, const SparkDsv4StagePackEntry *entry, uint64_t file_bytes, uint32_t *is_global)
{
	SparkDsv4StagePackTensorShape shape;
	uint64_t payload_bytes,scale_bytes;
	uint32_t global = entry->layer_index == SPARK_DSV4_STAGEPACK_GLOBAL_LAYER ? 1u : 0u;
	uint32_t in_slice = entry->layer_index == SPARK_DSV4_STAGEPACK_MTP_LAYER || (entry->layer_index >= state->first_layer_index && entry->layer_index < state->first_layer_index + state->layer_count) ? 1u : 0u;
	if ( entry->tensor_kind >= SPARK_DSV4_STAGEPACK_TENSOR_KIND_COUNT || (global == 0u && in_slice == 0u) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( SparkDsv4StagePackResolvedShape(entry->tensor_kind,global != 0u ? 0u : entry->layer_index,global,&shape) < 0 )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( shape.rows != entry->rows || shape.columns != entry->columns || shape.weight_format != entry->weight_format )
		return(SPARK_STATUS_VALIDATION_FAILED);
	payload_bytes = SparkDsv4StagePackPayloadBytes(entry->weight_format,entry->rows,entry->columns);
	scale_bytes = SparkDsv4StagePackScaleBytes(entry->weight_format,entry->rows,entry->columns);
	if ( entry->payload_offset + payload_bytes > file_bytes || (scale_bytes != 0u && (entry->scale_offset != entry->payload_offset + payload_bytes || entry->scale_offset + scale_bytes > file_bytes)) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	*is_global = global;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleBindGlobal(SparkDsv4ModuleState *state, const SparkDsv4StagePackEntry *entry, void *payload, void *scale)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_DSV4_STAGEPACK_TENSOR_EMBEDDING: state->token_embedding_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_FINAL_NORM: state->final_norm_weight_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_LM_HEAD: state->lm_head_weight_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_HEAD_FN: state->hc_head_fn_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_HEAD_BASE: state->hc_head_base_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_HEAD_SCALE: state->hc_head_scale_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_E_PROJ: SparkDsv4ModuleFillLinearView(&state->mtp.e_proj,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_H_PROJ: SparkDsv4ModuleFillLinearView(&state->mtp.h_proj,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_ENORM: state->mtp.enorm_weight_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_HNORM: state->mtp.hnorm_weight_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_FINAL_NORM: state->mtp.final_norm_weight_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_HC_HEAD_FN: state->mtp.hc_head_fn_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_HC_HEAD_BASE: state->mtp.hc_head_base_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_MTP_HC_HEAD_SCALE: state->mtp.hc_head_scale_f32 = (const float *)payload; break;
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	state->global_seen_bits |= 1ull << entry->tensor_kind;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleBindLayerAttn(SparkDsv4LayerWeights *layer, const SparkDsv4StagePackEntry *entry, void *payload, void *scale)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_DSV4_STAGEPACK_TENSOR_ATTN_SINK: layer->attn.sink_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_WQ_A: SparkDsv4ModuleFillLinearView(&layer->attn.wq_a,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_Q_NORM: layer->attn.q_norm_weight_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_WQ_B: SparkDsv4ModuleFillLinearView(&layer->attn.wq_b,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_WKV: SparkDsv4ModuleFillLinearView(&layer->attn.wkv,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_KV_NORM: layer->attn.kv_norm_weight_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_WO_A: SparkDsv4ModuleFillLinearView(&layer->attn.wo_a,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_WO_B: SparkDsv4ModuleFillLinearView(&layer->attn.wo_b,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_ATTN_NORM: layer->attn_norm_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_FFN_NORM: layer->ffn_norm_bf16 = payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_APE: layer->compressor.ape_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_WKV: SparkDsv4ModuleFillLinearView(&layer->compressor.wkv,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_WGATE: SparkDsv4ModuleFillLinearView(&layer->compressor.wgate,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_NORM: layer->compressor.norm_weight_bf16 = payload; break;
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleBindLayerIndexer(SparkDsv4LayerWeights *layer, const SparkDsv4StagePackEntry *entry, void *payload, void *scale)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WQ_B: SparkDsv4ModuleFillLinearView(&layer->indexer.wq_b,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WEIGHTS: SparkDsv4ModuleFillLinearView(&layer->indexer.weights_proj,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_INDEX_APE: layer->indexer.compressor.ape_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WKV: SparkDsv4ModuleFillLinearView(&layer->indexer.compressor.wkv,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WGATE: SparkDsv4ModuleFillLinearView(&layer->indexer.compressor.wgate,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_INDEX_NORM: layer->indexer.compressor.norm_weight_bf16 = payload; break;
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleBindLayerRest(SparkDsv4LayerWeights *layer, const SparkDsv4StagePackEntry *entry, void *payload, void *scale)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_ATTN_FN: layer->hc.attn_fn_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_FFN_FN: layer->hc.ffn_fn_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_ATTN_BASE: layer->hc.attn_base_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_FFN_BASE: layer->hc.ffn_base_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_ATTN_SCALE: layer->hc.attn_scale_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_HC_FFN_SCALE: layer->hc.ffn_scale_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_GATE_WEIGHT: SparkDsv4ModuleFillLinearView(&layer->moe.gate,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_GATE_BIAS: layer->moe.gate_bias_f32 = (const float *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_GATE_TID2EID: layer->moe.gate_tid2eid_u32 = (const uint32_t *)payload; break;
	case SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W1: SparkDsv4ModuleFillLinearView(&layer->moe.experts_w1,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W2: SparkDsv4ModuleFillLinearView(&layer->moe.experts_w2,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W3: SparkDsv4ModuleFillLinearView(&layer->moe.experts_w3,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_SHARED_W1: SparkDsv4ModuleFillLinearView(&layer->moe.shared_w1,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_SHARED_W2: SparkDsv4ModuleFillLinearView(&layer->moe.shared_w2,entry,payload,scale); break;
	case SPARK_DSV4_STAGEPACK_TENSOR_SHARED_W3: SparkDsv4ModuleFillLinearView(&layer->moe.shared_w3,entry,payload,scale); break;
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleBindLayer(SparkDsv4ModuleState *state, const SparkDsv4StagePackEntry *entry, void *payload, void *scale)
{
	SparkDsv4LayerWeights *layer = entry->layer_index == SPARK_DSV4_STAGEPACK_MTP_LAYER ? &state->mtp_layer : &state->layers[entry->layer_index];
	uint64_t *seen = entry->layer_index == SPARK_DSV4_STAGEPACK_MTP_LAYER ? &state->mtp_seen_bits : &state->layer_seen_bits[entry->layer_index];
	SparkStatus status;
	if ( entry->tensor_kind <= SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_NORM && entry->tensor_kind >= SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_APE )
		status = SparkDsv4ModuleBindLayerAttn(layer,entry,payload,scale);
	else if ( entry->tensor_kind <= SPARK_DSV4_STAGEPACK_TENSOR_FFN_NORM )
		status = SparkDsv4ModuleBindLayerAttn(layer,entry,payload,scale);
	else if ( entry->tensor_kind >= SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WQ_B && entry->tensor_kind <= SPARK_DSV4_STAGEPACK_TENSOR_INDEX_NORM )
		status = SparkDsv4ModuleBindLayerIndexer(layer,entry,payload,scale);
	else
		status = SparkDsv4ModuleBindLayerRest(layer,entry,payload,scale);
	if ( status != SPARK_STATUS_OK )
		return(status);
	*seen |= 1ull << entry->tensor_kind;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleLoadEntry(SparkDsv4ModuleState *state, FILE *file, const SparkDsv4StagePackEntry *entry, uint64_t file_bytes)
{
	uint64_t payload_bytes = SparkDsv4StagePackPayloadBytes(entry->weight_format,entry->rows,entry->columns);
	uint64_t scale_bytes = SparkDsv4StagePackScaleBytes(entry->weight_format,entry->rows,entry->columns);
	void *payload = 0,*scale = 0;
	uint32_t is_global = 0u;
	SparkStatus status = SparkDsv4ModuleValidateEntry(state,entry,file_bytes,&is_global);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s pack_entry_invalid kind=%u layer=%u\n",SPARK_DSV4_MODULE_TAG,entry->tensor_kind,entry->layer_index);
		return(status);
	}
	status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,entry->payload_offset,payload_bytes,&payload);
	if ( status == SPARK_STATUS_OK && scale_bytes != 0u )
		status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,entry->scale_offset,scale_bytes,&scale);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( is_global != 0u )
		return(SparkDsv4ModuleBindGlobal(state,entry,payload,scale));
	return(SparkDsv4ModuleBindLayer(state,entry,payload,scale));
}

// Coverage: every layer in the slice must have seen the exact kind set its
// attention class demands, the MTP layer its SWA score-routed set, and the
// globals the position-derived set - a missing tensor is a refused pack.
static uint64_t SparkDsv4ModuleExpectedLayerBits(uint32_t layer_index)
{
	uint32_t kind = SparkDsv4StagePackLayerKind(layer_index),tensor;
	uint64_t bits = 0u;
	for (tensor = SPARK_DSV4_STAGEPACK_TENSOR_ATTN_SINK; tensor <= SPARK_DSV4_STAGEPACK_TENSOR_SHARED_W3; tensor++)
		bits |= 1ull << tensor;
	bits &= ~(1ull << (SparkDsv4StagePackLayerIsHashRouted(layer_index) != 0u ? SPARK_DSV4_STAGEPACK_TENSOR_GATE_BIAS : SPARK_DSV4_STAGEPACK_TENSOR_GATE_TID2EID));
	if ( kind != SPARK_DSV4_MODEL_LAYER_KIND_SWA )
		bits |= (1ull << SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_APE) | (1ull << SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_WKV) | (1ull << SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_WGATE) | (1ull << SPARK_DSV4_STAGEPACK_TENSOR_COMPRESS_NORM);
	if ( kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA )
	{
		bits |= (1ull << SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WQ_B) | (1ull << SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WEIGHTS) | (1ull << SPARK_DSV4_STAGEPACK_TENSOR_INDEX_APE);
		bits |= (1ull << SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WKV) | (1ull << SPARK_DSV4_STAGEPACK_TENSOR_INDEX_WGATE) | (1ull << SPARK_DSV4_STAGEPACK_TENSOR_INDEX_NORM);
	}
	return(bits);
}

static SparkStatus SparkDsv4ModuleVerifyCoverage(SparkDsv4ModuleState *state)
{
	uint64_t expected_globals = 0u;
	uint32_t layer,tensor;
	for (layer = state->first_layer_index; layer < state->first_layer_index + state->layer_count; layer++)
		if ( state->layer_seen_bits[layer] != SparkDsv4ModuleExpectedLayerBits(layer) )
		{
			fprintf(stderr,"%s pack_layer_coverage layer=%u seen=%llx\n",SPARK_DSV4_MODULE_TAG,layer,(unsigned long long)state->layer_seen_bits[layer]);
			return(SPARK_STATUS_VALIDATION_FAILED);
		}
	if ( state->owns_embedding != 0u || state->owns_final_head != 0u )
		expected_globals |= 1ull << SPARK_DSV4_STAGEPACK_TENSOR_EMBEDDING;
	if ( state->owns_final_head != 0u )
	{
		for (tensor = SPARK_DSV4_STAGEPACK_TENSOR_FINAL_NORM; tensor <= SPARK_DSV4_STAGEPACK_TENSOR_MTP_HC_HEAD_SCALE; tensor++)
			expected_globals |= 1ull << tensor;
		if ( state->mtp_seen_bits != SparkDsv4ModuleExpectedLayerBits(SPARK_DSV4_STAGEPACK_MTP_LAYER) )
		{
			fprintf(stderr,"%s pack_mtp_coverage seen=%llx\n",SPARK_DSV4_MODULE_TAG,(unsigned long long)state->mtp_seen_bits);
			return(SPARK_STATUS_VALIDATION_FAILED);
		}
	}
	if ( state->global_seen_bits != expected_globals )
	{
		fprintf(stderr,"%s pack_global_coverage seen=%llx expected=%llx\n",SPARK_DSV4_MODULE_TAG,(unsigned long long)state->global_seen_bits,(unsigned long long)expected_globals);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleLoadPack(SparkDsv4ModuleState *state, const char *path)
{
	SparkDsv4StagePackHeader header,expected;
	SparkDsv4StagePackEntry *directory;
	FILE *file;
	SparkStatus status;
	int32_t compare;
	uint32_t index;
	file = fopen(path,"rb");
	if ( file == 0 )
	{
		fprintf(stderr,"%s pack_open_failed path=%s\n",SPARK_DSV4_MODULE_TAG,path);
		return(SPARK_STATUS_IO_ERROR);
	}
	status = SparkStageModulePackRead(SPARK_DSV4_MODULE_TAG,file,0u,&header,sizeof(header));
	if ( status == SPARK_STATUS_OK )
	{
		SparkDsv4StagePackExpectedGeometry(&expected,state->first_layer_index,state->layer_count);
		compare = SparkDsv4StagePackCompareGeometry(&header,&expected);
		if ( compare != 0 )
		{
			fprintf(stderr,"%s pack_geometry_mismatch field=%s\n",SPARK_DSV4_MODULE_TAG,SparkDsv4StagePackGeometryFieldName(compare));
			status = SPARK_STATUS_VALIDATION_FAILED;
		}
	}
	directory = status == SPARK_STATUS_OK ? (SparkDsv4StagePackEntry *)malloc((size_t)header.tensor_count * sizeof(SparkDsv4StagePackEntry)) : 0;
	if ( status == SPARK_STATUS_OK && directory == 0 )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModulePackRead(SPARK_DSV4_MODULE_TAG,file,header.directory_offset,directory,(uint64_t)header.tensor_count * sizeof(SparkDsv4StagePackEntry));
	for (index = 0; status == SPARK_STATUS_OK && index < header.tensor_count; index++)
		status = SparkDsv4ModuleLoadEntry(state,file,&directory[index],header.file_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleVerifyCoverage(state);
	free(directory);
	fclose(file);
	return(status);
}

// Host YaRN frequency table, the reference precompute arithmetic; the
// interpolation ramp engages only when original positions are declared.
static void SparkDsv4ModuleComputeFreqs(float *freqs, float base, uint32_t original, float factor)
{
	uint32_t rope_dim = SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,pair;
	float low,high,ramp,smooth,frequency;
	low = floorf((float)rope_dim * logf((float)original / ((float)SPARK_DSV4_MODEL_ROPE_BETA_FAST * 2.0f * 3.14159265f)) / (2.0f * logf(base)));
	high = ceilf((float)rope_dim * logf((float)original / ((float)SPARK_DSV4_MODEL_ROPE_BETA_SLOW * 2.0f * 3.14159265f)) / (2.0f * logf(base)));
	if ( low < 0.0f )
		low = 0.0f;
	if ( high > (float)(rope_dim - 1u) )
		high = (float)(rope_dim - 1u);
	if ( low == high )
		high += 0.001f;
	for (pair = 0; pair < rope_dim / 2u; pair++)
	{
		frequency = 1.0f / powf(base,(float)(2u * pair) / (float)rope_dim);
		if ( original != 0u )
		{
			ramp = ((float)pair - low) / (high - low);
			if ( ramp < 0.0f )
				ramp = 0.0f;
			if ( ramp > 1.0f )
				ramp = 1.0f;
			smooth = 1.0f - ramp;
			frequency = frequency / factor * (1.0f - smooth) + frequency * smooth;
		}
		freqs[pair] = frequency;
	}
}

static SparkStatus SparkDsv4ModuleUploadFreqs(SparkDsv4ModuleState *state)
{
	float host_freqs[SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION / 2u];
	cudaStream_t stream = (cudaStream_t)state->execution_stream;
	SparkStatus status;
	cudaError_t error;
	status = SparkStageModuleDeviceAllocate(&state->ledger,sizeof(host_freqs),(void **)&state->base_freqs_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,sizeof(host_freqs),(void **)&state->compress_freqs_f32);
	if ( status != SPARK_STATUS_OK )
		return(status);
	SparkDsv4ModuleComputeFreqs(host_freqs,SPARK_DSV4_MODEL_ATTN_ROPE_THETA,0u,(float)SPARK_DSV4_MODEL_ATTN_YARN_FACTOR);
	error = cudaMemcpyAsync(state->base_freqs_f32,host_freqs,sizeof(host_freqs),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
	{
		SparkDsv4ModuleComputeFreqs(host_freqs,SPARK_DSV4_MODEL_COMPRESS_ROPE_THETA,SPARK_DSV4_MODEL_ATTN_YARN_ORIGINAL_POSITIONS,(float)SPARK_DSV4_MODEL_ATTN_YARN_FACTOR);
		error = cudaMemcpyAsync(state->compress_freqs_f32,host_freqs,sizeof(host_freqs),cudaMemcpyHostToDevice,stream);
	}
	if ( error == cudaSuccess )
		error = cudaStreamSynchronize(stream);
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"freq_upload"));
}

/*
 * Cache pools. Every layer holds a lane block of window + compressed slots
 * of head_dim bf16 in one contiguous run - the reference's [win | stream]
 * layout, so the attention indices address one base. Compressor states are
 * sized for the WORST class on the stage (HCA's 128x512 doubled) and
 * strided uniformly per (layer ordinal, lane); the indexer keeps its own
 * rotated cache and small overlap state per CSA ordinal.
 */
// One-time MXFP4 shadow of the lm_head plus per-neuron certified error
// norms, the mimo25 screened-head pattern; head stage only, built
// synchronously at initialize.

/*
 * The hash routing tables come off the pack unchecked and feed the
 * device grouping kernel's shared histogram directly, so a corrupt or
 * mismatched table would write out of bounds. One init-time scan per
 * hash layer, blocking readback, hard failure on any out-of-range
 * entry.
 */
static SparkStatus SparkDsv4ModuleValidateHashTables(SparkDsv4ModuleState *state)
{
	uint32_t layer,*flag_device,flag_host = 0u;
	uint64_t entries = (uint64_t)SPARK_DSV4_MODEL_VOCAB_COUNT * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN;
	cudaStream_t stream = (cudaStream_t)state->execution_stream;
	SparkStatus status = SPARK_STATUS_OK;
	cudaError_t error;
	error = cudaMalloc((void **)&flag_device,sizeof(uint32_t));
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"tid2eid_flag"));
	error = cudaMemsetAsync(flag_device,0,sizeof(uint32_t),stream);
	for (layer = state->first_layer_index; error == cudaSuccess && layer < state->first_layer_index + state->layer_count; layer++)
		if ( state->layers[layer].moe.gate_tid2eid_u32 != 0 )
				error = SparkDsv4LaunchValidateTid2Eid(stream,state->layers[layer].moe.gate_tid2eid_u32,entries,flag_device);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(&flag_host,flag_device,sizeof(uint32_t),cudaMemcpyDeviceToHost,stream);
	if ( error == cudaSuccess )
		error = cudaStreamSynchronize(stream);
	cudaFree(flag_device);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"tid2eid_scan"));
	if ( flag_host != 0u )
	{
		fprintf(stderr,"%s hash_table_out_of_range\n",SPARK_DSV4_MODULE_TAG);
		status = SPARK_STATUS_VALIDATION_FAILED;
	}
	return(status);
}

static SparkStatus SparkDsv4ModuleBuildHeadShadow(SparkDsv4ModuleState *state)
{
	uint64_t vocab = SPARK_DSV4_MODEL_VOCAB_COUNT,dim = SPARK_DSV4_MODEL_HIDDEN_DIMENSION;
	cudaStream_t stream = (cudaStream_t)state->execution_stream;
	SparkStatus status;
	if ( state->owns_final_head == 0u )
		return(SPARK_STATUS_OK);
	status = SparkStageModuleDeviceAllocate(&state->ledger,(vocab * dim) / 2u,(void **)&state->head_shadow_payload);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(vocab * dim) / 32u,(void **)&state->head_shadow_scale);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,vocab * sizeof(float),(void **)&state->head_error_norm_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,SparkDsv4LaunchHeadShadowQuantize(stream,state->lm_head_weight_bf16,state->head_shadow_payload,state->head_shadow_scale,state->head_error_norm_f32,(uint32_t)vocab,(uint32_t)dim),"head_shadow_quantize");
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,cudaStreamSynchronize(stream),"head_shadow_sync");
	return(status);
}

// Post-pack validation and derived-weight construction, one call from
// the initialize chain.
static SparkStatus SparkDsv4ModuleFinalizeLoad(SparkDsv4ModuleState *state)
{
	SparkStatus status;
	status = SparkDsv4ModuleValidateHashTables(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleBuildHeadShadow(state);
	return(status);
}

static SparkStatus SparkDsv4ModuleAllocatePools(SparkDsv4ModuleState *state)
{
	uint64_t compressed_slots = state->max_sequence_positions / SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO;
	uint64_t lane_block = ((uint64_t)SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS + compressed_slots) * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION;
	uint64_t state_elements = (uint64_t)SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR * SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO * SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION;
	uint64_t index_state_elements = (uint64_t)SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR * SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO * SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR * SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION;
	uint64_t lanes = state->resident_sequence_capacity;
	SparkStatus status;
	state->cache_lane_block_elements = lane_block;
	state->cache_layer_lane_stride = lane_block;
	state->index_lane_stride = (uint64_t)state->index_slot_capacity * SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION;
	state->compress_state_lane_stride = state_elements;
	state->index_state_lane_stride = index_state_elements;
	status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,(uint64_t)state->layer_count * lanes * lane_block * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,&state->kv_cache_bf16);
	if ( status == SPARK_STATUS_OK && state->csa_layer_count != 0u )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,(uint64_t)state->csa_layer_count * lanes * state->index_lane_stride * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,&state->index_cache_bf16);
	if ( status == SPARK_STATUS_OK && state->compress_layer_count != 0u )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,(uint64_t)state->compress_layer_count * lanes * state_elements * sizeof(float),(void **)&state->compress_kv_state_f32);
	if ( status == SPARK_STATUS_OK && state->compress_layer_count != 0u )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,(uint64_t)state->compress_layer_count * lanes * state_elements * sizeof(float),(void **)&state->compress_score_state_f32);
	if ( status == SPARK_STATUS_OK && state->csa_layer_count != 0u )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,(uint64_t)state->csa_layer_count * lanes * index_state_elements * sizeof(float),(void **)&state->index_kv_state_f32);
	if ( status == SPARK_STATUS_OK && state->csa_layer_count != 0u )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,(uint64_t)state->csa_layer_count * lanes * index_state_elements * sizeof(float),(void **)&state->index_score_state_f32);
	return(status);
}

static SparkStatus SparkDsv4ModuleAllocateSlotSmall(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot)
{
	uint32_t rows = state->resident_sequence_capacity;
	SparkStatus status;

	slot->host_topk_indices = (int32_t *)calloc(
		(uint64_t)rows * state->topk_column_count,
		sizeof(*slot->host_topk_indices));
	if (slot->host_topk_indices == 0)
	{
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * sizeof(uint32_t),(void **)&slot->input_token_ids);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * sizeof(uint32_t),(void **)&slot->output_token_ids);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * sizeof(uint32_t),(void **)&slot->row_lane_indices);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * sizeof(uint64_t),(void **)&slot->row_positions);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * sizeof(uint64_t),(void **)&slot->row_emit_positions);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * sizeof(uint64_t),(void **)&slot->row_emit_positions_hca);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * sizeof(uint32_t),(void **)&slot->slot_counts);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * state->topk_column_count * sizeof(int32_t),(void **)&slot->topk_idxs);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * sizeof(uint32_t),(void **)&slot->emitted_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * SPARK_DSV4_MODEL_HC_MIX_ROWS * sizeof(float),(void **)&slot->mixes_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * SPARK_DSV4_MODEL_HC_STREAM_COUNT * sizeof(float),(void **)&slot->pre_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * SPARK_DSV4_MODEL_HC_STREAM_COUNT * sizeof(float),(void **)&slot->post_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * SPARK_DSV4_MODEL_HC_STREAM_COUNT * SPARK_DSV4_MODEL_HC_STREAM_COUNT * sizeof(float),(void **)&slot->comb_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT * sizeof(float),(void **)&slot->moe_scores_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)rows * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN * sizeof(float),(void **)&slot->moe_weights_f32);
	return(status);
}

static SparkStatus SparkDsv4ModuleAllocateSlotWide(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot)
{
	uint64_t rows = state->resident_sequence_capacity,dim = SPARK_DSV4_MODEL_HIDDEN_DIMENSION,bf16 = SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	uint64_t stream_bytes = rows * SPARK_DSV4_MODEL_HC_STREAM_COUNT * dim * bf16;
	uint64_t compress_channels = (uint64_t)SPARK_DSV4_MODEL_CSA_OVERLAP_FACTOR * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION;
	SparkStatus status;
	status = SparkStageModuleDeviceAllocate(&state->ledger,stream_bytes,&slot->streams_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,stream_bytes,&slot->residual_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * dim * bf16,&slot->reduced_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * dim * bf16,&slot->normalized_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_QUERY_LORA_RANK * bf16,&slot->qr_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_ATTN_QUERY_DIMENSION * bf16,&slot->q_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION * bf16,&slot->kv_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_ATTN_QUERY_DIMENSION * bf16,&slot->attn_out_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT * SPARK_DSV4_MODEL_OUTPUT_LORA_RANK * bf16,&slot->o_ranks_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * dim * bf16,&slot->delta_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * compress_channels * bf16,&slot->compress_kv_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * compress_channels * bf16,&slot->compress_score_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * compress_channels * sizeof(float),(void **)&slot->compress_kv_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * compress_channels * sizeof(float),(void **)&slot->compress_score_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION * bf16,&slot->emit_bf16);
	return(status);
}

static SparkStatus SparkDsv4ModuleAllocateSlotTail(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot)
{
	uint64_t rows = state->resident_sequence_capacity,dim = SPARK_DSV4_MODEL_HIDDEN_DIMENSION,bf16 = SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	SparkStatus status;
	status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_INDEX_DIMENSION * bf16,&slot->index_q_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_INDEX_HEAD_COUNT * bf16,&slot->index_weights_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_INDEX_HEAD_COUNT * sizeof(float),(void **)&slot->index_weights_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * (uint64_t)state->index_slot_capacity * sizeof(float),(void **)&slot->index_scores_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION * bf16,&slot->ffn_gate_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION * bf16,&slot->ffn_up_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * dim * bf16,&slot->ffn_delta_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * dim * bf16,&slot->ffn_accum_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN * sizeof(uint32_t),(void **)&slot->moe_indices_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN * sizeof(uint32_t),(void **)&slot->grouped_rows_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN * sizeof(uint32_t),(void **)&slot->grouped_weight_slots_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT + 1u) * sizeof(uint32_t),(void **)&slot->expert_offsets_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN * sizeof(uint32_t),(void **)&slot->moe_inverse_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN * SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION * bf16,&slot->moe_slot_gate_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN * SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION * bf16,&slot->moe_slot_up_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN * dim * bf16,&slot->moe_slot_out_bf16);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_MODEL_VOCAB_COUNT * bf16,&slot->head_logits_bf16);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_DSV4_RESIDENT_DECODE_STAGE_HEAD_SCREEN_CAP * sizeof(uint32_t),(void **)&slot->head_candidate_ids_u32);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->head_candidate_counts_u32);
	return(status);
}

static SparkStatus SparkDsv4ModuleAllocateSlot(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot)
{
	SparkStatus status = SparkDsv4ModuleAllocateSlotSmall(state,slot);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleAllocateSlotWide(state,slot);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleAllocateSlotTail(state,slot);
	return(status);
}

static SparkStatus SparkDsv4ModuleValidateFrameShape(
	const SparkDsv4ModuleState *state,
	const SparkModelDriverFrame *frame,
	uint32_t *is_prefill_out)
{
	const uint32_t known_flags = SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL;
	uint32_t is_prefill;
	if ( state == 0 || frame == 0 || is_prefill_out == 0 ||
		frame->program_id == 0u || frame->reserved != 0u ||
		frame->execution_stream == 0 || (frame->flags & ~known_flags) != 0u ||
		(frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID) != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	is_prefill = (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u ? 1u : 0u;
	if ( frame->active_slot_count == 0u ||
		frame->active_slot_count > state->resident_sequence_capacity ||
		frame->new_token_count == 0u ||
		frame->new_token_count > SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT ||
		(is_prefill == 0u && frame->new_token_count != frame->active_slot_count) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*is_prefill_out = is_prefill;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleValidateFrameContext(
	const SparkDsv4ModuleState *state,
	const SparkModelDriverFrame *frame,
	uint32_t is_prefill,
	const SparkDsv4ResidentDecodeStageFrameContext **context_out)
{
	const uint32_t known_flags =
		SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW |
		SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_BUFFER |
		SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_BUFFER |
		SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW |
		SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_BATCH_VIEW;
	const SparkDsv4ResidentDecodeStageFrameContext *context;
	uint32_t decode_view,prefill_view,needs_input,needs_output;
	if ( frame->buffer_count != state->owns_embedding + state->owns_final_head ||
		(frame->buffer_count != 0u && frame->buffers == 0) || frame->user_context == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	context = (const SparkDsv4ResidentDecodeStageFrameContext *)frame->user_context;
	if ( context->abi_version != SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION ||
		context->descriptor_bytes < (uint32_t)sizeof(*context) ||
		context->reserved0 != 0u || (context->flags & ~known_flags) != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	decode_view = (context->flags & SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW) != 0u;
	prefill_view = (context->flags & SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_BATCH_VIEW) != 0u;
	needs_input = state->stage_index > 0u ? 1u : 0u;
	needs_output = state->stage_index + 1u < state->stage_count ? 1u : 0u;
	if ( prefill_view != is_prefill || decode_view == is_prefill ||
		((context->flags & SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_BUFFER) != 0u) != (needs_input != 0u) ||
		((context->flags & SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_BUFFER) != 0u) != (needs_output != 0u) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*context_out = context;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleValidateDecodeView(
	const SparkModelDriverFrame *frame,
	const SparkDsv4DecodeBatchView *batch)
{
	if ( batch == 0 ||
		batch->abi_version != SPARK_DSV4_RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION ||
		batch->descriptor_bytes < (uint32_t)sizeof(*batch) || batch->reserved0 != 0u ||
		batch->row_count != frame->active_slot_count ||
		batch->row_count != frame->new_token_count || batch->row_lane_indices == 0 ||
		batch->row_positions == 0 || batch->row_sequence_ids == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleValidatePrefillView(
	const SparkModelDriverFrame *frame,
	const SparkDsv4PrefillBatchView *prefill)
{
	if ( prefill == 0 ||
		prefill->abi_version != SPARK_DSV4_RESIDENT_DECODE_STAGE_PREFILL_BATCH_VIEW_ABI_VERSION ||
		prefill->descriptor_bytes < (uint32_t)sizeof(*prefill) ||
		prefill->row_count != frame->new_token_count ||
		prefill->active_sequence_count != frame->active_slot_count ||
		prefill->token_ids == 0 || prefill->row_lane_indices == 0 ||
		prefill->row_positions == 0 || prefill->row_sequence_ids == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkDsv4ValidateRoundMajorPrefillRows(
		prefill->row_count,prefill->active_sequence_count,prefill->row_lane_indices));
}

static SparkStatus SparkDsv4ModuleValidateBoundaryBuffers(
	const SparkDsv4ModuleState *state,
	const SparkDsv4ResidentDecodeStageFrameContext *context,
	uint32_t row_count)
{
	uint32_t needs_input,needs_output;
	uint64_t bytes;
	needs_input = state->stage_index > 0u ? 1u : 0u;
	needs_output = state->stage_index + 1u < state->stage_count ? 1u : 0u;
	bytes = (uint64_t)row_count * SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	if ( (needs_input != 0u && (context->hidden_input_bf16 == 0 || context->hidden_input_bytes < bytes)) ||
		(needs_input == 0u && (context->hidden_input_bf16 != 0 || context->hidden_input_bytes != 0u)) ||
		(needs_output != 0u && (context->hidden_output_bf16 == 0 || context->hidden_output_bytes < bytes)) ||
		(needs_output == 0u && (context->hidden_output_bf16 != 0 || context->hidden_output_bytes != 0u)) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModuleValidateTokenBuffers(
	const SparkDsv4ModuleState *state,
	const SparkModelDriverFrame *frame,
	const SparkDsv4PrefillBatchView *prefill,
	uint32_t is_prefill,
	uint32_t row_count)
{
	uint32_t output_index;
	uint64_t bytes;
	SparkStatus status;
	bytes = (uint64_t)row_count * sizeof(uint32_t);
	if ( state->owns_embedding != 0u )
	{
		status = SparkModelDriverValidateBuffer(frame,0u,0u,SPARK_MODEL_DRIVER_BUFFER_FLAG_READ,bytes);
		if ( status != SPARK_STATUS_OK )
			return(status);
		if ( is_prefill != 0u && frame->buffers[0].address != prefill->token_ids )
			return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( state->owns_final_head == 0u )
		return(SPARK_STATUS_OK);
	output_index = state->owns_embedding != 0u ? 1u : 0u;
	return(SparkModelDriverValidateBuffer(frame,output_index,1u,
		SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE,bytes));
}

static SparkStatus SparkDsv4ModuleValidateFrame(
	SparkDsv4ModuleState *state,
	const SparkModelDriverFrame *frame,
	const SparkDsv4ResidentDecodeStageFrameContext **context_out)
{
	const SparkDsv4ResidentDecodeStageFrameContext *context;
	uint32_t is_prefill,row_count;
	SparkStatus status;
	status = SparkDsv4ModuleValidateFrameShape(state,frame,&is_prefill);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleValidateFrameContext(state,frame,is_prefill,&context);
	if ( status == SPARK_STATUS_OK && is_prefill != 0u )
		status = SparkDsv4ModuleValidatePrefillView(frame,context->prefill_batch);
	if ( status == SPARK_STATUS_OK && is_prefill == 0u )
		status = SparkDsv4ModuleValidateDecodeView(frame,context->decode_batch);
	if ( status != SPARK_STATUS_OK )
		return(status);
	row_count = is_prefill != 0u ? context->prefill_batch->row_count : context->decode_batch->row_count;
	status = SparkDsv4ModuleValidateBoundaryBuffers(state,context,row_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleValidateTokenBuffers(state,frame,context->prefill_batch,is_prefill,row_count);
	if ( status == SPARK_STATUS_OK )
		*context_out = context;
	return(status);
}

// Decode staging: distinct lanes, in-range positions, and every position
// below the dense cache bound - past it the frame refuses rather than
// truncating a stream the attention would then silently miss. Two emit
// position tables ride along, one per compress ratio: an emitted entry is
// roped at its GROUP START, position+1-ratio, and 4 and 128 differ.
static SparkStatus SparkDsv4ModuleValidateLaneSequenceContinuity(
    SparkDsv4ModuleState *state,
    const SparkDsv4DecodeBatchView *batch,
    uint8_t *lane_requires_reset)
{
    uint32_t row;

    if (state == 0 || batch == 0 || lane_requires_reset == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(lane_requires_reset, 0, batch->row_count * sizeof(*lane_requires_reset));
    for (row = 0u; row < batch->row_count; row++)
    {
        uint32_t lane;
        uint32_t previous_row;
        uint64_t sequence_id;
        uint64_t position;
        uint64_t current_sequence_id;

        lane = batch->row_lane_indices[row];
        sequence_id = batch->row_sequence_ids[row];
        position = batch->row_positions[row];
        for (previous_row = 0u; previous_row < row; previous_row++)
        {
            if (batch->row_sequence_ids[previous_row] == sequence_id)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
        }
        if (lane >= state->resident_sequence_capacity || sequence_id == 0u ||
            position >= state->max_sequence_positions)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
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

static cudaError_t SparkDsv4ModuleResetCompressLaneState(
	SparkDsv4ModuleState *state,
	cudaStream_t stream,
	uint32_t lane)
{
	uint32_t ordinal;
	uint64_t offset,bytes;
	cudaError_t error;
	error = cudaSuccess;
	bytes = state->compress_state_lane_stride * sizeof(float);
	for (ordinal=0u; error==cudaSuccess && ordinal<state->compress_layer_count; ordinal++)
	{
		offset = ((uint64_t)ordinal * state->resident_sequence_capacity + lane) *
			state->compress_state_lane_stride;
		error = cudaMemsetAsync(state->compress_kv_state_f32 + offset,0,(size_t)bytes,stream);
		if ( error == cudaSuccess )
			error = cudaMemsetAsync(state->compress_score_state_f32 + offset,0,(size_t)bytes,stream);
	}
	return(error);
}

static cudaError_t SparkDsv4ModuleResetIndexerLaneState(
	SparkDsv4ModuleState *state,
	cudaStream_t stream,
	uint32_t lane)
{
	uint32_t ordinal;
	uint64_t offset,bytes;
	cudaError_t error;
	error = cudaSuccess;
	bytes = state->index_state_lane_stride * sizeof(float);
	for (ordinal=0u; error==cudaSuccess && ordinal<state->csa_layer_count; ordinal++)
	{
		offset = ((uint64_t)ordinal * state->resident_sequence_capacity + lane) *
			state->index_state_lane_stride;
		error = cudaMemsetAsync(state->index_kv_state_f32 + offset,0,(size_t)bytes,stream);
		if ( error == cudaSuccess )
			error = cudaMemsetAsync(state->index_score_state_f32 + offset,0,(size_t)bytes,stream);
	}
	return(error);
}

static SparkStatus SparkDsv4ModuleResetLaneState(
	SparkDsv4ModuleState *state,
	SparkDsv4ModuleSlot *slot,
	uint32_t lane)
{
	cudaStream_t stream;
	cudaError_t error;
	if ( state == 0 || slot == 0 || lane >= state->resident_sequence_capacity )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	stream = (cudaStream_t)slot->cuda_stream;
	error = SparkDsv4ModuleResetCompressLaneState(state,stream,lane);
	if ( error == cudaSuccess )
		error = SparkDsv4ModuleResetIndexerLaneState(state,stream,lane);
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"lane_state_reset"));
}

static void SparkDsv4ModuleCommitLaneSequenceContinuity(
    SparkDsv4ModuleState *state,
    const SparkDsv4DecodeBatchView *batch)
{
    uint32_t row;

    for (row = 0u; row < batch->row_count; row++)
    {
        uint32_t lane;

        lane = batch->row_lane_indices[row];
        state->lane_sequence_ids[lane] = batch->row_sequence_ids[row];
        state->lane_next_positions[lane] = batch->row_positions[row] + 1u;
    }
}

static void SparkDsv4ModuleInvalidateLaneSequenceContinuity(
    SparkDsv4ModuleState *state,
    const SparkDsv4DecodeBatchView *batch)
{
    uint32_t row;

    for (row = 0u; row < batch->row_count; row++)
    {
        uint32_t lane;

        lane = batch->row_lane_indices[row];
        if (lane < state->resident_sequence_capacity)
        {
            state->lane_sequence_ids[lane] = 0u;
            state->lane_next_positions[lane] = 0u;
        }
    }
}

static SparkStatus SparkDsv4ModuleStageRows(SparkDsv4ModuleState *state, const SparkDsv4DecodeBatchView *batch, SparkDsv4ModuleSlot *slot)
{
	uint8_t lane_used[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint64_t emit_positions[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t emit_positions_hca[SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t row,lane;
	cudaError_t error;
	memset(lane_used,0,sizeof(lane_used));
	for (row = 0; row < batch->row_count; row++)
	{
		lane = batch->row_lane_indices[row];
		if ( lane >= state->resident_sequence_capacity || lane_used[lane] != 0u || batch->row_positions[row] >= state->max_sequence_positions )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		lane_used[lane] = 1u;
		emit_positions[row] = batch->row_positions[row] + 1u >= SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO ? batch->row_positions[row] + 1u - SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO : 0u;
		emit_positions_hca[row] = batch->row_positions[row] + 1u >= SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO ? batch->row_positions[row] + 1u - SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO : 0u;
	}
	error = cudaMemcpyAsync(slot->row_lane_indices,batch->row_lane_indices,(uint64_t)batch->row_count * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->row_positions,batch->row_positions,(uint64_t)batch->row_count * sizeof(uint64_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->row_emit_positions,emit_positions,(uint64_t)batch->row_count * sizeof(uint64_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->row_emit_positions_hca,emit_positions_hca,(uint64_t)batch->row_count * sizeof(uint64_t),cudaMemcpyHostToDevice,stream);
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"stage_rows"));
}

static SparkStatus SparkDsv4ModuleBeginStreams(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const SparkDsv4ResidentDecodeStageFrameContext *context, const uint32_t *token_ids, uint32_t rows, const void **streams_out)
{
	uint32_t token_guard;
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint64_t stream_bytes = (uint64_t)rows * SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	cudaError_t error;
	uint32_t copy;
	if ( streams_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*streams_out = 0;
	if ( state->owns_embedding != 0u )
	{
		if ( token_ids == 0 )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		for (token_guard = 0; token_guard < rows; token_guard++)
			if ( token_ids[token_guard] >= SPARK_DSV4_MODEL_VOCAB_COUNT )
			{
				fprintf(stderr,"%s token_id_out_of_range row=%u\n",SPARK_DSV4_MODULE_TAG,token_guard);
				return(SPARK_STATUS_INVALID_ARGUMENT);
			}
		error = cudaMemcpyAsync(slot->input_token_ids,token_ids,(uint64_t)rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchEmbeddingGather(stream,slot->input_token_ids,state->token_embedding_bf16,slot->reduced_bf16,rows,SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
		for (copy = 0; error == cudaSuccess && copy < SPARK_DSV4_MODEL_HC_STREAM_COUNT; copy++)
			error = cudaMemcpy2DAsync((uint8_t *)slot->streams_bf16 + (uint64_t)copy * SPARK_DSV4_MODEL_HIDDEN_DIMENSION * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,(uint64_t)SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,slot->reduced_bf16,(uint64_t)SPARK_DSV4_MODEL_HIDDEN_DIMENSION * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,(uint64_t)SPARK_DSV4_MODEL_HIDDEN_DIMENSION * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,rows,cudaMemcpyDeviceToDevice,stream);
		if ( error != cudaSuccess )
			return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"embedding_streams"));
		*streams_out = slot->streams_bf16;
		return(SPARK_STATUS_OK);
	}
	if ( context == 0 || context->hidden_input_bf16 == 0 || context->hidden_input_bytes < stream_bytes )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*streams_out = context->hidden_input_bf16;
	return(SPARK_STATUS_OK);
}

// One mHC boundary: mix, split with Sinkhorn, reduce - the residual copy
// is the caller's, since attention and ffn share this exactly.
static cudaError_t SparkDsv4ModuleHcEnter(SparkDsv4ModuleSlot *slot, const void *streams_bf16, const float *fn, const float *scale3, const float *base, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	if ( streams_bf16 == 0 )
		return(cudaErrorInvalidValue);
	error = cudaMemcpyAsync(slot->residual_bf16,streams_bf16,(uint64_t)rows * SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,cudaMemcpyDeviceToDevice,stream);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchHcMix(stream,streams_bf16,fn,slot->mixes_f32,rows,SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS,SPARK_DSV4_MODEL_HC_MIX_ROWS,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchHcSplitSinkhorn(stream,slot->mixes_f32,scale3,base,rows,SPARK_DSV4_MODEL_HC_STREAM_COUNT,SPARK_DSV4_MODEL_HC_SINKHORN_ITERATIONS,SPARK_DSV4_MODEL_HC_EPSILON,slot->pre_f32,slot->post_f32,slot->comb_f32);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchHcPreReduce(stream,streams_bf16,slot->pre_f32,slot->reduced_bf16,rows,SPARK_DSV4_MODEL_HC_STREAM_COUNT,SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
	return(error);
}

/*
 * The compressor for one decode token, attention side: wkv/wgate on the
 * normalized x, widen to fp32, ape by in-group position, the state step,
 * and for boundary rows the emitted slot gets norm, rope at the group
 * start position, the fp8 cache sim, and lands at position/ratio behind
 * the window - the host knows the boundary from the position arithmetic.
 */
static cudaError_t SparkDsv4ModuleRunCompressor(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const SparkDsv4CompressorWeights *weights, const SparkDsv4DecodeBatchView *batch, float *kv_state, float *score_state, uint64_t state_stride, void *cache_base, uint64_t cache_lane_stride, uint64_t cache_slot_offset, uint32_t cache_width, uint32_t rotate, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint32_t channels = (weights->overlap != 0u ? 2u : 1u) * cache_width,row;
	uint64_t ratio = weights->ratio,position,slot_index;
	cudaError_t error;
	error = SparkDsv4LaunchLinear(stream,&weights->wkv,slot->normalized_bf16,slot->compress_kv_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchLinear(stream,&weights->wgate,slot->normalized_bf16,slot->compress_score_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchWiden(stream,slot->compress_kv_bf16,slot->compress_kv_f32,rows,channels,1.0f);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchWiden(stream,slot->compress_score_bf16,slot->compress_score_f32,rows,channels,1.0f);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchApeAdd(stream,slot->compress_score_f32,weights->ape_f32,slot->row_positions,rows,(uint32_t)ratio,channels);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchCompressStep(stream,slot->compress_kv_f32,slot->compress_score_f32,kv_state,score_state,state_stride,slot->row_lane_indices,slot->row_positions,rows,(uint32_t)ratio,weights->overlap,cache_width,slot->emit_bf16,slot->emitted_u32);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRmsNorm(stream,slot->emit_bf16,weights->norm_weight_bf16,slot->emit_bf16,rows,cache_width,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRope(stream,slot->emit_bf16,state->compress_freqs_f32,weights->ratio == SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO ? slot->row_emit_positions_hca : slot->row_emit_positions,rows,1u,cache_width,SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,0u);
	if ( error == cudaSuccess && rotate != 0u )
	{
		error = SparkDsv4LaunchHadamard(stream,slot->emit_bf16,rows,cache_width);
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchQuantSim(stream,slot->emit_bf16,rows,cache_width,cache_width,SPARK_DSV4_MODEL_FP4_QUANT_BLOCK,1u);
	}
	if ( error == cudaSuccess && rotate == 0u )
		error = SparkDsv4LaunchQuantSim(stream,slot->emit_bf16,rows,cache_width,cache_width - SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,SPARK_DSV4_MODEL_KV_QUANT_BLOCK,0u);
	for (row = 0; error == cudaSuccess && row < rows; row++)
	{
		position = batch->row_positions[row];
		if ( (position + 1u) % ratio != 0u )
			continue;
		slot_index = cache_slot_offset + position / ratio;
		error = cudaMemcpyAsync((uint8_t *)cache_base + ((uint64_t)batch->row_lane_indices[row] * cache_lane_stride + slot_index * cache_width) * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,(const uint8_t *)slot->emit_bf16 + (uint64_t)row * cache_width * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,(uint64_t)cache_width * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,cudaMemcpyDeviceToDevice,stream);
	}
	return(error);
}

// Host-side attention index assembly: the window slots every kind attends
// (ring slots [0, min(pos+1, 128))), the full compressed run for HCA, and
// the CSA tail left for the device top-k. -1 pads the rest.
static void SparkDsv4ModuleHostTopkFill(SparkDsv4ModuleState *state, const SparkDsv4DecodeBatchView *batch, int32_t *host_idxs, uint32_t *host_counts, uint32_t layer_kind, uint32_t rows)
{
	uint32_t row,column,window,compressed;
	uint64_t position;
	for (row = 0; row < rows; row++)
	{
		position = batch->row_positions[row];
		window = position + 1u < SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS ? (uint32_t)(position + 1u) : SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS;
		compressed = layer_kind == SPARK_DSV4_MODEL_LAYER_KIND_HCA ? (uint32_t)((position + 1u) / SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO) : 0u;
		host_counts[row] = layer_kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA ? (uint32_t)((position + 1u) / SPARK_DSV4_MODEL_CSA_COMPRESS_RATIO) : 0u;
		for (column = 0; column < state->topk_column_count; column++)
		{
			if ( column < window )
				host_idxs[(uint64_t)row * state->topk_column_count + column] = (int32_t)column;
			else if ( layer_kind == SPARK_DSV4_MODEL_LAYER_KIND_HCA && column < window + compressed )
				host_idxs[(uint64_t)row * state->topk_column_count + column] = (int32_t)(SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS + column - window);
			else
				host_idxs[(uint64_t)row * state->topk_column_count + column] = -1;
		}
	}
}

static cudaError_t SparkDsv4ModuleRunIndexer(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const SparkDsv4LayerWeights *layer, const SparkDsv4DecodeBatchView *batch, uint32_t csa_ordinal, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint64_t lanes = state->resident_sequence_capacity;
	void *index_cache = (uint8_t *)state->index_cache_bf16 + (uint64_t)csa_ordinal * lanes * state->index_lane_stride * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	float *kv_state = state->index_kv_state_f32 + (uint64_t)csa_ordinal * lanes * state->index_state_lane_stride;
	float *score_state = state->index_score_state_f32 + (uint64_t)csa_ordinal * lanes * state->index_state_lane_stride;
	float weight_scale = 1.0f / sqrtf((float)SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION) / sqrtf((float)SPARK_DSV4_MODEL_INDEX_HEAD_COUNT);
	cudaError_t error;
	error = SparkDsv4LaunchLinear(stream,&layer->indexer.wq_b,slot->qr_bf16,slot->index_q_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRope(stream,slot->index_q_bf16,state->compress_freqs_f32,slot->row_positions,rows,SPARK_DSV4_MODEL_INDEX_HEAD_COUNT,SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION,SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,0u);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchHadamard(stream,slot->index_q_bf16,rows * SPARK_DSV4_MODEL_INDEX_HEAD_COUNT,SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchQuantSim(stream,slot->index_q_bf16,rows,SPARK_DSV4_MODEL_INDEX_DIMENSION,SPARK_DSV4_MODEL_INDEX_DIMENSION,SPARK_DSV4_MODEL_FP4_QUANT_BLOCK,1u);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchLinear(stream,&layer->indexer.weights_proj,slot->normalized_bf16,slot->index_weights_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchWiden(stream,slot->index_weights_bf16,slot->index_weights_f32,rows,SPARK_DSV4_MODEL_INDEX_HEAD_COUNT,weight_scale);
	if ( error == cudaSuccess )
		error = SparkDsv4ModuleRunCompressor(state,slot,&layer->indexer.compressor,batch,kv_state,score_state,state->index_state_lane_stride,index_cache,state->index_lane_stride,0u,SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION,1u,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchIndexerScore(stream,slot->index_q_bf16,index_cache,state->index_lane_stride,slot->row_lane_indices,slot->slot_counts,slot->index_weights_f32,slot->index_scores_f32,rows,state->index_slot_capacity,SPARK_DSV4_MODEL_INDEX_HEAD_COUNT,SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchTopK(stream,slot->index_scores_f32,slot->slot_counts,state->index_slot_capacity,SPARK_DSV4_MODEL_INDEX_TOP_K,(int32_t)SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS,slot->topk_idxs + SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS,state->topk_column_count,rows);
	return(error);
}

// Attention index staging: the host fills the window and HCA parts and
// the CSA valid-slot counts into its own buffer, then one upload each -
// the CSA tail stays device-written by the top-k kernel.
static cudaError_t SparkDsv4ModuleStageTopk(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const SparkDsv4DecodeBatchView *batch, uint32_t layer_kind, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	SparkDsv4ModuleHostTopkFill(state,batch,slot->host_topk_indices,slot->host_slot_counts,layer_kind,rows);
	error = cudaMemcpyAsync(slot->topk_idxs,slot->host_topk_indices,(uint64_t)rows * state->topk_column_count * sizeof(int32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->slot_counts,slot->host_slot_counts,(uint64_t)rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	return(error);
}

// One o-composition group: wo_a's block against the group's slice of the
// attention output, ranks landing at the group's offset - block-diagonal
// through the strided kernel.
static cudaError_t SparkDsv4ModuleRunOutputGroup(SparkDsv4ModuleSlot *slot, const SparkDsv4LayerWeights *layer, uint32_t group, uint32_t rows)
{
	SparkDsv4LinearView view = layer->attn.wo_a;
	uint64_t payload_bytes = SparkDsv4StagePackPayloadBytes(view.weight_format,SPARK_DSV4_MODEL_OUTPUT_LORA_RANK,SPARK_DSV4_MODEL_OUTPUT_GROUP_DIMENSION);
	uint64_t scale_bytes = SparkDsv4StagePackScaleBytes(view.weight_format,SPARK_DSV4_MODEL_OUTPUT_LORA_RANK,SPARK_DSV4_MODEL_OUTPUT_GROUP_DIMENSION);
	const uint8_t *payload = (const uint8_t *)view.payload + (uint64_t)group * payload_bytes;
	const uint8_t *scale = view.scale_e8m0 != 0 ? view.scale_e8m0 + (uint64_t)group * scale_bytes : 0;
	view.rows = SPARK_DSV4_MODEL_OUTPUT_LORA_RANK;
	view.columns = SPARK_DSV4_MODEL_OUTPUT_GROUP_DIMENSION;
	return(SparkDsv4LaunchStridedLinear((cudaStream_t)slot->cuda_stream,&view,payload,scale,slot->attn_out_bf16,SPARK_DSV4_MODEL_ATTN_QUERY_DIMENSION,group * SPARK_DSV4_MODEL_OUTPUT_GROUP_DIMENSION,slot->o_ranks_bf16,(uint64_t)SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT * SPARK_DSV4_MODEL_OUTPUT_LORA_RANK,group * SPARK_DSV4_MODEL_OUTPUT_LORA_RANK,rows));
}

static cudaError_t SparkDsv4ModuleRunAttention(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const SparkDsv4LayerWeights *layer, const SparkDsv4DecodeBatchView *batch, uint32_t layer_index, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint32_t kind = SparkDsv4ModelLayerKind(layer_index),local = state->layer_local_by_layer[layer_index],group,row;
	const float *freqs = kind == SPARK_DSV4_MODEL_LAYER_KIND_SWA ? state->base_freqs_f32 : state->compress_freqs_f32;
	void *cache = (uint8_t *)state->kv_cache_bf16 + (uint64_t)local * state->resident_sequence_capacity * state->cache_lane_block_elements * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
	uint64_t lane_stride = state->cache_lane_block_elements,position;
	cudaError_t error;
	error = SparkDsv4LaunchLinear(stream,&layer->attn.wq_a,slot->normalized_bf16,slot->delta_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRmsNorm(stream,slot->delta_bf16,layer->attn.q_norm_weight_bf16,slot->qr_bf16,rows,SPARK_DSV4_MODEL_QUERY_LORA_RANK,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchLinear(stream,&layer->attn.wq_b,slot->qr_bf16,slot->q_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchQueryHeadRms(stream,slot->q_bf16,rows,SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRope(stream,slot->q_bf16,freqs,slot->row_positions,rows,SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,0u);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchLinear(stream,&layer->attn.wkv,slot->normalized_bf16,slot->kv_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRmsNorm(stream,slot->kv_bf16,layer->attn.kv_norm_weight_bf16,slot->kv_bf16,rows,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRope(stream,slot->kv_bf16,freqs,slot->row_positions,rows,1u,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,0u);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchQuantSim(stream,slot->kv_bf16,rows,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION - SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,SPARK_DSV4_MODEL_KV_QUANT_BLOCK,0u);
	for (row = 0; error == cudaSuccess && row < rows; row++)
	{
		position = batch->row_positions[row];
		error = cudaMemcpyAsync((uint8_t *)cache + ((uint64_t)batch->row_lane_indices[row] * lane_stride + (position % SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS) * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION) * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,(const uint8_t *)slot->kv_bf16 + (uint64_t)row * SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES,cudaMemcpyDeviceToDevice,stream);
	}
	if ( error == cudaSuccess && kind != SPARK_DSV4_MODEL_LAYER_KIND_SWA )
		error = SparkDsv4ModuleRunCompressor(state,slot,&layer->compressor,batch,state->compress_kv_state_f32 + (uint64_t)state->compress_ordinal_by_layer[layer_index] * state->resident_sequence_capacity * state->compress_state_lane_stride,state->compress_score_state_f32 + (uint64_t)state->compress_ordinal_by_layer[layer_index] * state->resident_sequence_capacity * state->compress_state_lane_stride,state->compress_state_lane_stride,cache,lane_stride,SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,0u,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4ModuleStageTopk(state,slot,batch,kind,rows);
	if ( error == cudaSuccess && kind == SPARK_DSV4_MODEL_LAYER_KIND_CSA )
		error = SparkDsv4ModuleRunIndexer(state,slot,layer,batch,state->csa_ordinal_by_layer[layer_index],rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchSparseAttn(stream,slot->q_bf16,cache,lane_stride,slot->row_lane_indices,slot->topk_idxs,state->topk_column_count,layer->attn.sink_f32,1.0f / sqrtf((float)SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION),slot->attn_out_bf16,rows,SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRope(stream,slot->attn_out_bf16,freqs,slot->row_positions,rows,SPARK_DSV4_MODEL_ATTN_QUERY_HEAD_COUNT,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,1u);
	for (group = 0; error == cudaSuccess && group < SPARK_DSV4_MODEL_OUTPUT_GROUP_COUNT; group++)
		error = SparkDsv4ModuleRunOutputGroup(slot,layer,group,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchLinear(stream,&layer->attn.wo_b,slot->o_ranks_bf16,slot->delta_bf16,rows);
	return(error);
}

/*
 * Device-grouped routed MoE, mirrored from mimo25: one grouping kernel,
 * three all-expert tile launches with device-side counts, the clamped
 * swiglu with the routing weight folded at the intermediate running
 * dense over the grouped pairs, and the unweighted pair reduce
 * accumulating race-free through the inverse map. No host round trips,
 * no per-layer synchronize; the step is graph-capturable end to end.
 */
static SparkStatus SparkDsv4ModuleRunMoeRouted(SparkDsv4ModuleSlot *slot, const SparkDsv4MoeWeights *moe, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint32_t pair_count = rows * SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN;
	uint64_t inter = SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION,dim = SPARK_DSV4_MODEL_HIDDEN_DIMENSION;
	cudaError_t error;
	error = SparkDsv4LaunchMoeGroup(stream,slot->moe_indices_u32,pair_count,slot->expert_offsets_u32,slot->grouped_rows_u32,slot->grouped_weight_slots_u32,slot->moe_inverse_u32);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchExpertTileAll(stream,&moe->experts_w1,slot->normalized_bf16,slot->grouped_rows_u32,slot->expert_offsets_u32,slot->moe_slot_gate_bf16,pair_count,inter,dim);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchExpertTileAll(stream,&moe->experts_w3,slot->normalized_bf16,slot->grouped_rows_u32,slot->expert_offsets_u32,slot->moe_slot_up_bf16,pair_count,inter,dim);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchSwigluClamp(stream,slot->moe_slot_gate_bf16,slot->moe_slot_up_bf16,pair_count,SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION,SPARK_DSV4_MODEL_SWIGLU_LIMIT,slot->moe_weights_f32,slot->grouped_weight_slots_u32);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchExpertTileAll(stream,&moe->experts_w2,slot->moe_slot_up_bf16,0,slot->expert_offsets_u32,slot->moe_slot_out_bf16,pair_count,dim,inter);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchMoePairReduce(stream,slot->moe_slot_out_bf16,slot->moe_inverse_u32,slot->ffn_accum_bf16,rows);
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"moe_routed"));
}

static SparkStatus SparkDsv4ModuleRunMoe(SparkDsv4ModuleSlot *slot, const SparkDsv4LayerWeights *layer, uint32_t layer_index, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	const SparkDsv4MoeWeights *moe = &layer->moe;
	uint32_t hash = SparkDsv4StagePackLayerIsHashRouted(layer_index);
	cudaError_t error;
	error = SparkDsv4LaunchGateScores(stream,&moe->gate,slot->normalized_bf16,slot->moe_scores_f32,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchGateSelect(stream,slot->moe_scores_f32,hash != 0u ? 0 : moe->gate_bias_f32,hash != 0u ? moe->gate_tid2eid_u32 : 0,slot->input_token_ids,rows,SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT,SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN,SPARK_DSV4_MODEL_ROUTED_SCALING_FACTOR,slot->moe_indices_u32,slot->moe_weights_f32);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchLinear(stream,&moe->shared_w1,slot->normalized_bf16,slot->ffn_gate_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchLinear(stream,&moe->shared_w3,slot->normalized_bf16,slot->ffn_up_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchSwigluClamp(stream,slot->ffn_gate_bf16,slot->ffn_up_bf16,rows,SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION,SPARK_DSV4_MODEL_SWIGLU_LIMIT,0,0);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchLinear(stream,&moe->shared_w2,slot->ffn_up_bf16,slot->ffn_accum_bf16,rows);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"moe_shared"));
	return(SparkDsv4ModuleRunMoeRouted(slot,moe,rows));
}

static SparkStatus SparkDsv4ModuleRunLayer(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const SparkDsv4DecodeBatchView *batch, const void *input_streams_bf16, void *output_streams_bf16, uint32_t layer_index, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	const SparkDsv4LayerWeights *layer = &state->layers[layer_index];
	cudaError_t error;
	SparkStatus status;
	error = SparkDsv4ModuleHcEnter(slot,input_streams_bf16,layer->hc.attn_fn_f32,layer->hc.attn_scale_f32,layer->hc.attn_base_f32,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRmsNorm(stream,slot->reduced_bf16,layer->attn_norm_bf16,slot->normalized_bf16,rows,SPARK_DSV4_MODEL_HIDDEN_DIMENSION,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkDsv4ModuleRunAttention(state,slot,layer,batch,layer_index,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchHcPost(stream,slot->delta_bf16,slot->residual_bf16,slot->post_f32,slot->comb_f32,slot->streams_bf16,rows,SPARK_DSV4_MODEL_HC_STREAM_COUNT,SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"attn_side"));
	error = SparkDsv4ModuleHcEnter(slot,slot->streams_bf16,layer->hc.ffn_fn_f32,layer->hc.ffn_scale_f32,layer->hc.ffn_base_f32,rows);
	if ( error == cudaSuccess )
		error = SparkDsv4LaunchRmsNorm(stream,slot->reduced_bf16,layer->ffn_norm_bf16,slot->normalized_bf16,rows,SPARK_DSV4_MODEL_HIDDEN_DIMENSION,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"ffn_enter"));
	status = SparkDsv4ModuleRunMoe(slot,layer,layer_index,rows);
	if ( status != SPARK_STATUS_OK )
		return(status);
	error = SparkDsv4LaunchHcPost(stream,slot->ffn_accum_bf16,slot->residual_bf16,slot->post_f32,slot->comb_f32,output_streams_bf16,rows,SPARK_DSV4_MODEL_HC_STREAM_COUNT,SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
	return(SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"ffn_side"));
}

static SparkStatus SparkDsv4ModuleRunLayers(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, const SparkDsv4DecodeBatchView *batch, const void *input_streams_bf16, void *output_streams_bf16, uint32_t rows)
{
	uint32_t layer;
	const void *layer_input;
	void *layer_output;
	SparkStatus status = SPARK_STATUS_OK;
	layer_input = input_streams_bf16;
	for (layer = state->first_layer_index; status == SPARK_STATUS_OK && layer < state->first_layer_index + state->layer_count; layer++)
	{
		layer_output = layer + 1u == state->first_layer_index + state->layer_count ? output_streams_bf16 : slot->streams_bf16;
		status = SparkDsv4ModuleRunLayer(state,slot,batch,layer_input,layer_output,layer,rows);
		layer_input = layer_output;
	}
	return(status);
}

static SparkStatus SparkDsv4ModuleFinish(SparkDsv4ModuleState *state, SparkDsv4ModuleSlot *slot, uint32_t *output_tokens, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	SparkStatus status = SPARK_STATUS_OK;
	cudaError_t error = cudaSuccess;
	if ( state->owns_final_head != 0u )
	{
		error = SparkDsv4LaunchHcMix(stream,slot->streams_bf16,state->hc_head_fn_f32,slot->mixes_f32,rows,SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS,SPARK_DSV4_MODEL_HC_STREAM_COUNT,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchHcHeadReduce(stream,slot->streams_bf16,slot->mixes_f32,state->hc_head_scale_value,state->hc_head_base_f32,SPARK_DSV4_MODEL_HC_EPSILON,slot->reduced_bf16,rows,SPARK_DSV4_MODEL_HC_STREAM_COUNT,SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchRmsNorm(stream,slot->reduced_bf16,state->final_norm_weight_bf16,slot->normalized_bf16,rows,SPARK_DSV4_MODEL_HIDDEN_DIMENSION,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
		if ( error == cudaSuccess )
			error = SparkDsv4LaunchHeadScreenedArgmax(stream,slot->normalized_bf16,state->lm_head_weight_bf16,state->head_shadow_payload,state->head_shadow_scale,state->head_error_norm_f32,slot->head_logits_bf16,slot->head_candidate_ids_u32,slot->head_candidate_counts_u32,slot->output_token_ids,rows,SPARK_DSV4_MODEL_VOCAB_COUNT,SPARK_DSV4_MODEL_HIDDEN_DIMENSION);
		if ( error == cudaSuccess && output_tokens == 0 )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		if ( error == cudaSuccess )
			error = cudaMemcpyAsync(output_tokens,slot->output_token_ids,(uint64_t)rows * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream);
		status = SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,error,"head");
	}
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
		status = SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,cudaStreamSynchronize(stream),"sync");
	return(status);
}

static SparkStatus SparkDsv4ModuleExecuteDecodeBatch(
    SparkDsv4ModuleState *state,
    SparkModelDriverFrame *frame,
    SparkDsv4ResidentDecodeStageFrameContext *context,
    const SparkDsv4DecodeBatchView *batch,
    const uint32_t *token_ids,
    uint32_t *output_tokens)
{
    SparkDsv4ModuleSlot *slot;
    uint8_t lane_requires_reset[
        SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
    uint32_t slot_index;
    uint32_t rows;
    uint32_t row;
    uint32_t lanes_claimed;
    const void *input_streams_bf16;
    void *output_streams_bf16;
    SparkStatus status;

    if (state == 0 || frame == 0 || context == 0 || batch == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    rows = batch->row_count;
    lanes_claimed = 0u;
    status = SparkStageModuleIndexSetClaim(
        state->lane_states,
        state->resident_sequence_capacity,
        batch->row_lane_indices,
        rows);
    lanes_claimed = status == SPARK_STATUS_OK ? 1u : 0u;
    if (status == SPARK_STATUS_OK)
    {
        status = SparkDsv4ModuleValidateLaneSequenceContinuity(
            state,
            batch,
            lane_requires_reset);
    }
    if (status != SPARK_STATUS_OK)
    {
        if (lanes_claimed != 0u)
        {
            SparkStageModuleIndexSetRelease(
                state->lane_states,
                state->resident_sequence_capacity,
                batch->row_lane_indices,
                rows);
        }
        atomic_fetch_add_explicit(&state->rejected_count, 1u, memory_order_relaxed);
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
            state->resident_sequence_capacity,
            batch->row_lane_indices,
            rows);
        atomic_fetch_add_explicit(&state->rejected_count, 1u, memory_order_relaxed);
        return status;
    }

	atomic_fetch_add_explicit(&state->submitted_count, 1u, memory_order_relaxed);
	slot = &state->slots[slot_index];
	slot->cuda_stream = frame->execution_stream;
    for (row = 0u; status == SPARK_STATUS_OK && row < rows; row++)
    {
        if (lane_requires_reset[row] != 0u)
        {
            status = SparkDsv4ModuleResetLaneState(
                state,
                slot,
                batch->row_lane_indices[row]);
        }
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkDsv4ModuleStageRows(state, batch, slot);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkDsv4ModuleBeginStreams(
            state, slot, context, token_ids, rows, &input_streams_bf16);
    }
    if (status == SPARK_STATUS_OK)
    {
        output_streams_bf16 = state->owns_final_head != 0u ?
            slot->streams_bf16 : context->hidden_output_bf16;
        status = SparkDsv4ModuleRunLayers(
            state, slot, batch, input_streams_bf16,
            output_streams_bf16, rows);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkDsv4ModuleFinish(state, slot, output_tokens, rows);
    }
    if (status == SPARK_STATUS_OK)
    {
        SparkDsv4ModuleCommitLaneSequenceContinuity(state, batch);
        atomic_fetch_add_explicit(&state->completed_count, 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&state->tokens_emitted, rows, memory_order_relaxed);
    }
    else
    {
        SparkDsv4ModuleInvalidateLaneSequenceContinuity(state, batch);
        atomic_fetch_add_explicit(&state->failed_count, 1u, memory_order_relaxed);
    }

    SparkStageModuleIndexSetRelease(
        state->lane_states,
        state->resident_sequence_capacity,
        batch->row_lane_indices,
        rows);
    SparkStageModuleSlotRelease(state->slot_states, slot_index);
    return status;
}

static uint32_t SparkDsv4ModulePrefillWaveRowCount(
	const SparkDsv4PrefillBatchView *prefill,
	uint32_t first_row)
{
	return(SparkDsv4RoundMajorPrefillWaveRowCount(prefill->row_count,
		prefill->active_sequence_count,prefill->row_lane_indices,first_row));
}

static SparkStatus SparkDsv4ModuleExecutePrefillWave(
    SparkDsv4ModuleState *state,
    SparkModelDriverFrame *frame,
    SparkDsv4ResidentDecodeStageFrameContext *context,
    uint32_t first_row,
    uint32_t row_count,
    uint32_t *output_tokens)
{
    const SparkDsv4PrefillBatchView *prefill;
    SparkDsv4ResidentDecodeStageFrameContext one_context;
    SparkDsv4DecodeBatchView batch;
    uint64_t row_bytes;
    prefill = context->prefill_batch;
    row_bytes = (uint64_t)SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS * SPARK_DSV4_MODEL_BF16_ELEMENT_BYTES;
    memset(&batch,0,sizeof(batch));
    batch.abi_version = SPARK_DSV4_RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION;
    batch.descriptor_bytes = (uint32_t)sizeof(batch);
    batch.row_count = row_count;
    batch.row_lane_indices = prefill->row_lane_indices + first_row;
    batch.row_positions = prefill->row_positions + first_row;
    batch.row_sequence_ids = prefill->row_sequence_ids + first_row;
    one_context = *context;
    if (one_context.hidden_input_bf16 != 0)
    {
        one_context.hidden_input_bf16 = (const uint8_t *)context->hidden_input_bf16 + ((uint64_t)first_row * row_bytes);
        one_context.hidden_input_bytes = (uint64_t)row_count * row_bytes;
    }
    if (one_context.hidden_output_bf16 != 0)
    {
        one_context.hidden_output_bf16 = (uint8_t *)context->hidden_output_bf16 + ((uint64_t)first_row * row_bytes);
        one_context.hidden_output_bytes = (uint64_t)row_count * row_bytes;
    }
    return SparkDsv4ModuleExecuteDecodeBatch(state,frame,&one_context,&batch,state->owns_embedding != 0u ? prefill->token_ids + first_row : 0,output_tokens != 0 ? output_tokens + first_row : 0);
}

static SparkStatus SparkDsv4ModuleExecutePrefill(
    SparkDsv4ModuleState *state,
    SparkModelDriverFrame *frame,
    SparkDsv4ResidentDecodeStageFrameContext *context)
{
    const SparkDsv4PrefillBatchView *prefill;
    uint32_t *output_tokens;
    uint32_t output_buffer_index,row,wave_rows;
    SparkStatus status;
    if (state == 0 || frame == 0 || context == 0 || context->prefill_batch == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    prefill = context->prefill_batch;
    output_tokens = 0;
    if (state->owns_final_head != 0u)
    {
        output_buffer_index = state->owns_embedding != 0u ? 1u : 0u;
        output_tokens = (uint32_t *)frame->buffers[output_buffer_index].address;
    }
    status = SPARK_STATUS_OK;
	for (row=0u; status==SPARK_STATUS_OK && row<prefill->row_count; row+=wave_rows)
	{
		wave_rows = SparkDsv4ModulePrefillWaveRowCount(prefill,row);
		if ( wave_rows == 0u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		status = SparkDsv4ModuleExecutePrefillWave(state,frame,context,row,wave_rows,output_tokens);
        if (status != SPARK_STATUS_OK)
            return status;
    }
    return status;
}

SparkStatus SparkDsv4ResidentDecodeStageExecute(
    void *module_state,
    SparkModelDriverFrame *frame)
{
    SparkDsv4ModuleState *state;
    SparkDsv4ResidentDecodeStageFrameContext *context;
    SparkStatus status;

    state = (SparkDsv4ModuleState *)module_state;
    if (state == 0 || frame == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    context = 0;
	status = SparkDsv4ModuleValidateFrame(
        state,
        frame,
        (const SparkDsv4ResidentDecodeStageFrameContext **)&context);
	if (status != SPARK_STATUS_OK)
    {
        atomic_fetch_add_explicit(
            &state->rejected_count,
            1u,
            memory_order_relaxed);
		return status;
	}
	if (state->execution_stream != frame->execution_stream)
	{
		atomic_fetch_add_explicit(
			&state->rejected_count,
			1u,
			memory_order_relaxed);
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
    if ((frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u)
    {
        return SparkDsv4ModuleExecutePrefill(state, frame, context);
    }
    return SparkDsv4ModuleExecuteDecodeBatch(
        state,
        frame,
        context,
        context->decode_batch,
        state->owns_embedding != 0u ?
            (const uint32_t *)frame->buffers[0].address : 0,
        state->owns_final_head != 0u ?
            (uint32_t *)frame->buffers[state->owns_embedding != 0u ? 1u : 0u].address :
            0);
}

static void SparkDsv4ModuleRejectAdmission(
	SparkDsv4ModuleState *state,
	SparkModelDriverAdmissionDecision *decision,
	uint32_t reason)
{
	SparkStageModuleAdmissionDecisionReject(decision,reason);
	atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
}

static uint32_t SparkDsv4ModuleAdmissionShapeSupported(
	const SparkDsv4ModuleState *state,
	const SparkModelDriverAdmissionRequest *request,
	uint32_t is_prefill)
{
	const uint32_t known_flags = SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL;
	if ( (request->frame_flags & ~known_flags) != 0u ||
		(request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID) != 0u ||
		request->active_slot_count == 0u ||
		request->active_slot_count > state->resident_sequence_capacity ||
		request->new_token_count == 0u ||
		request->new_token_count > SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT ||
		(is_prefill == 0u && request->new_token_count != request->active_slot_count) )
		return(0u);
	return(1u);
}

static uint32_t SparkDsv4ModuleAdmissionFitsKv(
	const SparkDsv4ModuleState *state,
	const SparkModelDriverAdmissionRequest *request)
{
	if ( request->sequence_position >= (uint64_t)state->max_sequence_positions )
		return(0u);
	if ( request->active_slot_count == 1u &&
		request->new_token_count > (uint64_t)state->max_sequence_positions - request->sequence_position )
		return(0u);
	return(1u);
}

SparkStatus SparkDsv4ResidentDecodeStageAdmit(
	void *module_state,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	SparkDsv4ModuleState *state;
	uint32_t available,is_prefill;
	state = (SparkDsv4ModuleState *)module_state;
	if ( state == 0 || request == 0 || decision == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	available = SparkStageModuleSlotCountFree(state->slot_states,state->pipeline_slot_count);
	SparkStageModuleAdmissionDecisionInitialize(decision,available);
	if ( request->descriptor_bytes < (uint32_t)sizeof(*request) || request->program_id == 0u )
		return(SPARK_STATUS_ABI_MISMATCH);
	is_prefill = (request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u ? 1u : 0u;
	if ( SparkDsv4ModuleAdmissionShapeSupported(state,request,is_prefill) == 0u )
	{
		SparkDsv4ModuleRejectAdmission(state,decision,SPARK_MODEL_DRIVER_ADMISSION_REJECTED_UNSUPPORTED_SHAPE);
		return(SPARK_STATUS_OK);
	}
	if ( SparkDsv4ModuleAdmissionFitsKv(state,request) == 0u )
	{
		SparkDsv4ModuleRejectAdmission(state,decision,SPARK_MODEL_DRIVER_ADMISSION_REJECTED_KV_CAPACITY);
		return(SPARK_STATUS_OK);
	}
	if ( available == 0u )
	{
		SparkDsv4ModuleRejectAdmission(state,decision,SPARK_MODEL_DRIVER_ADMISSION_REJECTED_BUSY);
		return(SPARK_STATUS_OK);
	}
	SparkStageModuleAdmissionDecisionAccept(decision);
	decision->host_staging_bytes = (uint64_t)request->new_token_count *
		(sizeof(uint32_t) * (uint64_t)(state->owns_embedding + state->owns_final_head + 1u) + sizeof(uint64_t) * 3u);
	decision->device_memcpy_bytes = decision->host_staging_bytes;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkDsv4ResidentDecodeStageSnapshot(
    void *module_state,
    uint32_t program_id,
    SparkModelDriverRuntimeSnapshot *snapshot)
{
    SparkDsv4ModuleState *state;

    state = (SparkDsv4ModuleState *)module_state;
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
    snapshot->kv_token_capacity = (uint64_t)state->resident_sequence_capacity * state->max_sequence_positions;
    return SPARK_STATUS_OK;
}

void SparkDsv4ResidentDecodeStageDestroy(void *module_state)
{
    SparkDsv4ModuleState *state;
    uint32_t slot_index;

    state = (SparkDsv4ModuleState *)module_state;
    if (state == 0)
    {
        return;
    }
    if (SparkStageModuleWaitForSlots(
            SPARK_DSV4_MODULE_TAG,
            state->slot_states,
            state->pipeline_slot_count,
            SPARK_STAGE_MODULE_DESTROY_QUIESCE_TIMEOUT_NS) != SPARK_STATUS_OK)
    {
        return;
    }
	for (slot_index = 0u; slot_index < state->pipeline_slot_count; ++slot_index)
	{
		free(state->slots[slot_index].host_topk_indices);
	}
    SparkStageModuleLedgerRelease(&state->ledger);
    free(state);
}

static SparkStatus SparkDsv4ModuleAllocateState(
	const SparkFirmwareModuleHostServices *host_services,
	SparkDsv4ModuleState **state_out,
	const char **pack_path_out)
{
	SparkDsv4ModuleState *state;
	SparkStatus status;
	state = (SparkDsv4ModuleState *)calloc(1u,sizeof(*state));
	if ( state == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->ledger.module_tag = SPARK_DSV4_MODULE_TAG;
	atomic_init(&state->submitted_count,0u);
	atomic_init(&state->completed_count,0u);
	atomic_init(&state->rejected_count,0u);
	atomic_init(&state->failed_count,0u);
	atomic_init(&state->tokens_emitted,0u);
	status = SparkDsv4ModuleConfigure(state,host_services,pack_path_out);
	if ( status != SPARK_STATUS_OK )
	{
		free(state);
		return(status);
	}
	SparkStageModuleAtomicStateArrayInitialize(state->slot_states,state->pipeline_slot_count);
	SparkStageModuleAtomicStateArrayInitialize(state->lane_states,state->resident_sequence_capacity);
	*state_out = state;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ModulePrepareState(
	SparkDsv4ModuleState *state,
	const char *pack_path)
{
	uint32_t slot_index;
	SparkStatus status;
	status = SparkStageModuleCudaStatus(SPARK_DSV4_MODULE_TAG,
		SparkDsv4ConfigureCudaKernels(),"configure_cuda_kernels");
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleValidateSlice(state);
	if ( status == SPARK_STATUS_OK )
	{
		SparkDsv4ModuleBuildOrdinals(state);
		status = SparkDsv4ModuleLoadPack(state,pack_path);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleUploadFreqs(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleAllocatePools(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkDsv4ModuleFinalizeLoad(state);
	for (slot_index=0u; status==SPARK_STATUS_OK && slot_index<state->pipeline_slot_count; slot_index++)
		status = SparkDsv4ModuleAllocateSlot(state,&state->slots[slot_index]);
	return(status);
}

static void SparkDsv4ModuleReportReady(const SparkDsv4ModuleState *state)
{
	fprintf(stderr,
		"%s ready stage=%u/%u slice=%u+%u compress=%u csa=%u lanes=%u max_seq=%u device_gib=%.1f\n",
		SPARK_DSV4_MODULE_TAG,state->stage_index,state->stage_count,
		state->first_layer_index,state->layer_count,state->compress_layer_count,
		state->csa_layer_count,state->resident_sequence_capacity,
		state->max_sequence_positions,(double)state->ledger.device_bytes_resident /
		(1024.0 * 1024.0 * 1024.0));
}

SparkStatus SparkDsv4ResidentDecodeStageInitialize(
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services,
	void **module_state)
{
	SparkDsv4ModuleState *state;
	const char *pack_path;
	SparkStatus status;
	pack_path = 0;
	status = SparkFirmwareModuleValidateInitialization(configuration,host_services,module_state);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkDsv4ModuleAllocateState(host_services,&state,&pack_path);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkDsv4ModulePrepareState(state,pack_path);
	if ( status != SPARK_STATUS_OK )
	{
		SparkDsv4ResidentDecodeStageDestroy(state);
		return(status);
	}
	SparkDsv4ModuleReportReady(state);
	*module_state = state;
	return(SPARK_STATUS_OK);
}
