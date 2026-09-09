#define _FILE_OFFSET_BITS 64

#include <stdatomic.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include "sparkpipe/spark_tp_chain_ordinal.h"

#include "sparkpipe/spark_glm5_next_resident_decode_stage_firmware.h"

#define SPARK_GLM5_NEXT_PROBE_CONNECT_TIMEOUT_SCALE 4u
#define SPARK_GLM5_NEXT_PROBE_OPERATION_TIMEOUT_SCALE 8u
static int SparkGlm5NextProbeEnabled(void)
{
	static int probe_enabled = -1;
	if ( probe_enabled < 0 )
		probe_enabled = getenv("SPARK_GLM5_NEXT_PROBE") != 0 ? 1 : 0;
	return(probe_enabled);
}
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_admission.h"
#include "sparkpipe/spark_kv_model_table.h"
#include "sparkpipe/spark_glm5_next_kv_geometry.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "sparkpipe/spark_row_layout.h"
#include "sparkpipe/spark_weightd_attach.h"
#include "sparkpipe/spark_weightd_lazy_pack.h"
#include "sparkpipe/spark_weightd_lease.h"
#include "sparkpipe/spark_head_screen.h"
#include "sparkpipe/spark_speculation_policy.h"
#include "spark_glm5_next_resident_decode_stage_internal.h"
#include "spark_glm5_next_stagepack_format.h"

#ifndef GLM5_NEXT_EXPERT_WEIGHT_CODEC
#error "GLM5_NEXT_EXPERT_WEIGHT_CODEC must name the exact package expert codec"
#endif
#ifndef GLM5_NEXT_CONTRACT_SHA256
#error "GLM5_NEXT_CONTRACT_SHA256 must identify the exact model package contract"
#endif

#define SPARK_GLM5_NEXT_MODULE_TAG "glm5_next_stage"
#define SPARK_GLM5_NEXT_STAGEPACK_MAX_TENSOR_COUNT 2048u
#define SPARK_GLM5_NEXT_HEAD_TILE 1024u
#define SPARK_GLM5_NEXT_NO_INDEX_ORDINAL UINT32_MAX
#define SPARK_GLM5_NEXT_KV_ACCESS_ERROR_WORD_COUNT 6u

_Static_assert(SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT <= SPARK_WEIGHTD_WORK_QUEUE_CAPACITY,"completion worker must hold one job per occupied slot");

#if SPARK_GLM5_NEXT_MODEL_KV_SLOT_BYTES != \
	( SPARK_GLM5_NEXT_KV_ARENA_KV_HEAD_COUNT * \
	  SPARK_GLM5_NEXT_KV_ARENA_HEAD_DIM * \
	  SPARK_GLM5_NEXT_KV_BYTES_PER_SCALAR )
#error "glm5_next KV slot bytes and arena block geometry disagree"
#endif

typedef struct SparkGlm5NextPackRange
{
	uint64_t offset;
	uint64_t bytes;
} SparkGlm5NextPackRange;

typedef struct SparkGlm5NextModuleState SparkGlm5NextModuleState;

typedef struct SparkGlm5NextAsyncCompletion
{
	SparkGlm5NextModuleState *state;
	SparkModelDriverCompletionFunction completion_function;
	void *completion_context;
	uint32_t slot_index;
	uint32_t lane_count;
	uint32_t row_count;
	uint32_t lane_indices[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint8_t lane_bound[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_sequence_ids[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_next_positions[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t *output_token_destination;
	uint32_t burst_token_count;
	uint64_t mtp_cache_extra;
	uint32_t mtp_draft_tokens[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH];
	SparkModelDriverCompletion completion;
} SparkGlm5NextAsyncCompletion;

struct SparkGlm5NextModuleState
{
	SparkStageModuleLedger ledger;
	SparkWeightdWorker *completion_worker;
	SparkWeightdLazyPack *lazy_pack;
	_Atomic(void *) lazy_retained[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	uint32_t stage_count;
	uint32_t stage_index;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t expert_weight_codec;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint32_t tp_collective_disabled;
	uint32_t resident_sequence_capacity;
	uint32_t pipeline_slot_count;
	uint32_t max_sequence_positions;
	uint32_t execution_row_capacity;
	uint32_t decode_split_context_threshold;
	uint32_t pages_per_sequence;
	uint32_t page_count;
	uint32_t index_layer_count;
	uint32_t multiprocessor_count;
	uint32_t owns_embedding;
	uint32_t owns_final_head;
	void *execution_stream;
	char model_revision[SPARK_GLM5_NEXT_STAGEPACK_MODEL_REVISION_BYTES];
	SparkGlm5NextLayerWeights layers[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE];
	uint32_t index_ordinal_by_local_layer[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE];
	uint32_t kv_ordinal_by_local_layer[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE];
	uint32_t kda_ordinal_by_local_layer[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE];
	uint32_t kv_layer_count;
	uint32_t kda_layer_count;
	uint8_t *kda_state_pools;
	uint64_t kda_state_layer_stride_bytes;
	uint8_t *kda_window_pools;
	uint8_t *kda_q_window_pool;
	uint8_t *kda_k_window_pool;
	uint8_t *kda_v_window_pool;
	uint64_t kda_window_layer_stride_bytes;
	uint32_t *kda_state_index_device;
	uint32_t *kda_state_index_host;
	uint64_t layer_seen[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE];
	uint64_t global_seen;
	uint64_t mtp_seen;
	uint32_t pack_has_mtp;
	SparkGlm5NextLayerWeights mtp_layer;
	const void *mtp_eh_proj_bf16;
	const void *mtp_enorm_bf16;
	const void *mtp_hnorm_bf16;
	const void *mtp_shared_norm_bf16;
	uint32_t mtp_enabled;
	uint16_t *mtp_lane_hidden_bf16;
	uint8_t *mtp_lane_armed;
	uint64_t kda_replay_layer_bytes;
	const void *embedding_bf16;
	const void *final_norm_bf16;
	const void *lm_head_bf16;
	uint8_t *head_certified_fp8_payload;
	float *head_certified_fp8_scale_f32;
	float *head_certified_fp8_norm_f32;
	uint8_t *kv_cache;
	uint64_t kv_layer_stride_bytes;
	uint8_t *index_cache;
	uint64_t index_layer_stride_bytes;
	uint32_t *page_table;
	uint32_t *page_table_shadow;
	SparkKvCacheArena kv_arena;
	SparkKvPageCache kv_page_cache;
	SparkKvPageStore kv_page_store;
	SparkKvPageStore recurrent_store;
	uint8_t *recurrent_staging;
	uint64_t recurrent_page_bytes;
	SparkKvCacheBlock *kv_blocks;
	uint32_t *kv_resident_slot_logical_block_indices;
	SparkKvPageCacheEntry *kv_entries;
	SparkKvPageCacheSequence *kv_sequences;
	uint32_t *kv_hash_bucket_heads;
	uint32_t *kv_entry_indices_by_logical_page;
	uint8_t *kv_page_staging;
	uint32_t *kv_lane_logical_pages;
	uint32_t *kv_lane_physical_pages;
	SparkKvLaneTransaction *kv_lane_transactions;
	SparkKvLaneTransactions kv_transactions;
	pthread_mutex_t kv_mutex;
	uint32_t kv_mutex_initialized;
	uint64_t control_generation;
	uint64_t reset_generation;
	const char *kv_backing_directory;
	uint64_t kv_backing_maximum_bytes;
	char kv_backing_default[256];
	SparkGlm5NextExecutionSlot slots[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	SparkGlm5NextAsyncCompletion completions[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	atomic_uint slot_states[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	atomic_uint lane_states[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	atomic_uchar lane_bound[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	atomic_ullong lane_sequence_ids[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	atomic_ullong lane_next_positions[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	atomic_ullong submitted_count;
	atomic_ullong completed_count;
	atomic_ullong rejected_count;
	atomic_ullong failed_count;
	atomic_ullong host_callback_completion_count;
	SparkTpDeviceCollective tp_device_collective;
	SparkTpDeviceCollective tp_device_collective_hc;
	uint32_t tp_device_collective_hc_initialized;
	uint32_t tp_device_collective_initialized;
	SparkTpDeviceCollectiveCreditBinding tp_credit_bindings[SPARK_TP_DEVICE_COLLECTIVE_MAX_BINDING_COUNT];
	uint32_t tp_credit_binding_count;
	void *tp_credit_send_bf16;
	void *tp_credit_receive_bf16;
	void *tp_host_credit_send_bf16;
	void *tp_host_credit_receive_bf16;
	SparkTpDeviceCollectiveCreditBinding tp_hc_credit_bindings[SPARK_TP_DEVICE_COLLECTIVE_MAX_BINDING_COUNT];
	uint32_t tp_hc_credit_binding_count;
	void *tp_hc_credit_send_bf16;
	void *tp_hc_credit_receive_bf16;
	void *tp_hc_host_credit_send_bf16;
	void *tp_hc_host_credit_receive_bf16;
	atomic_ullong nccl_next_ordinal;
	atomic_ullong nccl_next_ordinal_hc;
};


static SparkStatus SparkGlm5NextModuleConfigure(
	SparkGlm5NextModuleState *state,
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services,
	const char **pack_path)
{
	const SparkGlm5NextResidentDecodeStageNodeContext *context;
	if ( state == 0 || configuration == 0 || host_services == 0 || pack_path == 0 || host_services->node_context == 0 || host_services->execution_stream == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	context = (const SparkGlm5NextResidentDecodeStageNodeContext *)host_services->node_context;
	if ( context->abi_version != SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION || context->descriptor_bytes != SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( SparkGlm5NextResidentDecodeStageSpanIsValid(context->stage_count,context->stage_index,context->first_layer_index,context->layer_count) == 0u || context->layer_count > SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE || context->expert_weight_codec != GLM5_NEXT_EXPERT_WEIGHT_CODEC || context->resident_sequence_capacity == 0u || context->resident_sequence_capacity > SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT || context->pipeline_slot_count == 0u || context->pipeline_slot_count > SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT || context->max_sequence_positions == 0u || context->max_sequence_positions > SPARK_GLM5_NEXT_MODEL_MAXIMUM_CONTEXT_TOKENS || context->execution_row_capacity == 0u || context->execution_row_capacity > SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT || context->decode_split_context_threshold > context->max_sequence_positions || context->tp_degree == 0u || context->tp_rank >= context->tp_degree || (context->flags & ~SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_KNOWN_FLAGS) != 0u || context->stage_pack_path == 0 || context->stage_pack_path[0] == '\0' || context->model_revision == 0 || context->model_revision[0] == '\0' || strlen(context->model_revision) >= sizeof(state->model_revision) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( SparkWeightCodecIsKnown(context->expert_weight_codec) == 0u || context->expert_weight_codec == SPARK_WEIGHT_CODEC_BF16 )
		return(SPARK_STATUS_UNSUPPORTED);
	if ( (context->flags & SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_FLAG_MTP) != 0u && (context->stage_index + 1u) != context->stage_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( context->tp_degree != 1u && (SPARK_GLM5_NEXT_MODEL_HEAD_COUNT % context->tp_degree != 0u || SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT % context->tp_degree != 0u || SPARK_GLM5_NEXT_MODEL_DENSE_INTERMEDIATE_DIMENSION % context->tp_degree != 0u || SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION % context->tp_degree != 0u) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->model_revision == 0 || strcmp(configuration->model_revision,context->model_revision) != 0 )
		return(SPARK_STATUS_SCHEMA_ERROR);
	state->stage_count = context->stage_count;
	state->stage_index = context->stage_index;
	state->first_layer_index = context->first_layer_index;
	state->layer_count = context->layer_count;
	state->expert_weight_codec = context->expert_weight_codec;
	state->tp_degree = context->tp_degree;
	state->tp_rank = context->tp_rank;
	state->kv_backing_directory = context->kv_backing_directory;
	state->kv_backing_maximum_bytes = context->kv_backing_maximum_bytes;
	state->tp_collective_disabled = context->tp_collective_identifier == 0u ? 1u : 0u;
	state->resident_sequence_capacity = context->resident_sequence_capacity;
	state->pipeline_slot_count = context->pipeline_slot_count;
	state->max_sequence_positions = context->max_sequence_positions;
	state->decode_split_context_threshold = context->decode_split_context_threshold;
	state->execution_row_capacity = context->execution_row_capacity;
	state->owns_embedding = context->stage_index == 0u ? 1u : 0u;
	state->owns_final_head = context->stage_index + 1u == context->stage_count ? 1u : 0u;
	state->mtp_enabled = (context->flags & SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_NODE_CONTEXT_FLAG_MTP) != 0u ? 1u : 0u;
	state->execution_stream = host_services->execution_stream;
	(void)snprintf(state->model_revision,sizeof(state->model_revision),"%s",context->model_revision);
	*pack_path = context->stage_pack_path;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextPackFileSize(FILE *file,uint64_t *bytes)
{
	off_t end;
	if ( file == 0 || bytes == 0 || fseeko(file,0,SEEK_END) != 0 )
		return(SPARK_STATUS_IO_ERROR);
	end = ftello(file);
	if ( end < 0 || fseeko(file,0,SEEK_SET) != 0 )
		return(SPARK_STATUS_IO_ERROR);
	*bytes = (uint64_t)end;
	return(SPARK_STATUS_OK);
}

static uint32_t SparkGlm5NextPackRangesOverlap(const SparkGlm5NextPackRange *left,const SparkGlm5NextPackRange *right)
{
	return(left->bytes != 0u && right->bytes != 0u && left->offset < right->offset + right->bytes && right->offset < left->offset + left->bytes ? 1u : 0u);
}

static SparkStatus SparkGlm5NextPackValidateHeader(
	const SparkGlm5NextModuleState *state,
	const SparkGlm5NextStagePackHeader *header,
	uint64_t file_bytes)
{
	uint64_t directory_bytes,directory_end;
	if ( state == 0 || header == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( header->magic != SPARK_GLM5_NEXT_STAGEPACK_MAGIC || header->format_version != SPARK_GLM5_NEXT_STAGEPACK_FORMAT_VERSION || header->header_bytes != SPARK_GLM5_NEXT_STAGEPACK_HEADER_BYTES || header->directory_entry_bytes != SPARK_GLM5_NEXT_STAGEPACK_ENTRY_BYTES || header->codec_abi_version != SPARK_WEIGHT_CODEC_ABI_VERSION )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( (header->flags & ~SPARK_GLM5_NEXT_STAGEPACK_KNOWN_FLAGS) != 0u )
		return(SPARK_STATUS_UNSUPPORTED);
	if ( SparkGlm5NextStagePackHeaderTpDegree(header) == 0u || SparkGlm5NextStagePackHeaderTpDegree(header) != state->tp_degree || SparkGlm5NextStagePackHeaderTpRank(header) != state->tp_rank )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( header->tensor_count == 0u || header->tensor_count > SPARK_GLM5_NEXT_STAGEPACK_MAX_TENSOR_COUNT || header->stage_count != state->stage_count || header->stage_index != state->stage_index || header->first_layer_index != state->first_layer_index || header->layer_count != state->layer_count || header->total_layer_count != SPARK_GLM5_NEXT_MODEL_LAYER_COUNT || header->hidden_dimension != SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION || header->vocab_count != SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT || header->routed_expert_count != SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( header->linear_weight_codec != SPARK_WEIGHT_CODEC_BF16 || header->expert_weight_codec != state->expert_weight_codec || header->kv_cache_codec != SPARK_WEIGHT_CODEC_BF16 )
		return(SPARK_STATUS_TARGET_MISMATCH);
	if ( header->file_bytes != file_bytes || header->directory_offset < header->header_bytes || header->directory_offset % SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES != 0u || header->tensor_count > UINT64_MAX / header->directory_entry_bytes )
		return(SPARK_STATUS_SCHEMA_ERROR);
	directory_bytes = (uint64_t)header->tensor_count * header->directory_entry_bytes;
	directory_end = header->directory_offset + directory_bytes;
	if ( directory_end < header->directory_offset || directory_end > file_bytes )
		return(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextPackValidateEntryGeometry(
	const SparkGlm5NextModuleState *state,
	const SparkGlm5NextStagePackHeader *header,
	const SparkGlm5NextStagePackEntry *entry,
	SparkGlm5NextStagePackTensorShape *shape)
{
	uint64_t payload_bytes,scale_bytes,directory_end;
	uint32_t local_layer;
	if ( SparkGlm5NextStagePackExpectedShape(entry->tensor_kind,entry->layer_index,state->expert_weight_codec,state->tp_degree,shape) < 0 )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( entry->layer_index != SPARK_GLM5_NEXT_STAGEPACK_GLOBAL_LAYER )
	{
		if ( entry->layer_index == SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX )
		{
			if ( (state->mtp_seen & (UINT64_C(1) << entry->tensor_kind)) != 0u )
				return(SPARK_STATUS_DUPLICATE);
		}
		else
		{
			if ( entry->layer_index < state->first_layer_index || entry->layer_index >= state->first_layer_index + state->layer_count )
				return(SPARK_STATUS_SCHEMA_ERROR);
			local_layer = entry->layer_index - state->first_layer_index;
			if ( (state->layer_seen[local_layer] & (UINT64_C(1) << entry->tensor_kind)) != 0u )
				return(SPARK_STATUS_DUPLICATE);
		}
	}
	else if ( (state->global_seen & (UINT64_C(1) << entry->tensor_kind)) != 0u )
		return(SPARK_STATUS_DUPLICATE);
	if ( entry->payload_type != shape->payload_type || entry->weight_codec != shape->weight_codec || entry->scale_encoding != shape->scale_encoding || entry->group_count != shape->group_count || entry->rows != shape->rows || entry->columns != shape->columns )
		return(SPARK_STATUS_SCHEMA_ERROR);
	payload_bytes = SparkGlm5NextStagePackExpectedPayloadBytes(shape);
	scale_bytes = SparkGlm5NextStagePackExpectedScaleBytes(shape);
	if ( payload_bytes == 0u || entry->payload_bytes != payload_bytes || entry->scale_bytes != scale_bytes )
		return(SPARK_STATUS_SCHEMA_ERROR);
	directory_end = header->directory_offset + ((uint64_t)header->tensor_count * header->directory_entry_bytes);
	if ( entry->payload_offset % SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES != 0u || entry->payload_offset > header->file_bytes || entry->payload_bytes > header->file_bytes - entry->payload_offset )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( entry->payload_offset < directory_end && header->directory_offset < entry->payload_offset + entry->payload_bytes )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( scale_bytes == 0u )
	{
		if ( entry->scale_offset != 0u )
			return(SPARK_STATUS_SCHEMA_ERROR);
	}
	else if ( entry->scale_offset % SPARK_GLM5_NEXT_STAGEPACK_ALIGNMENT_BYTES != 0u || entry->scale_offset > header->file_bytes || entry->scale_bytes > header->file_bytes - entry->scale_offset )
		return(SPARK_STATUS_SCHEMA_ERROR);
	else if ( entry->scale_offset < directory_end && header->directory_offset < entry->scale_offset + entry->scale_bytes )
		return(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextPackValidateRanges(
	const SparkGlm5NextStagePackEntry *entries,
	uint32_t entry_count)
{
	SparkGlm5NextPackRange left[2],right[2];
	uint32_t left_index,right_index,left_part,right_part;
	for (left_index=0u; left_index<entry_count; left_index++)
	{
		left[0].offset = entries[left_index].payload_offset;
		left[0].bytes = entries[left_index].payload_bytes;
		left[1].offset = entries[left_index].scale_offset;
		left[1].bytes = entries[left_index].scale_bytes;
		for (right_index=left_index + 1u; right_index<entry_count; right_index++)
		{
			right[0].offset = entries[right_index].payload_offset;
			right[0].bytes = entries[right_index].payload_bytes;
			right[1].offset = entries[right_index].scale_offset;
			right[1].bytes = entries[right_index].scale_bytes;
			for (left_part=0u; left_part<2u; left_part++)
				for (right_part=0u; right_part<2u; right_part++)
					if ( SparkGlm5NextPackRangesOverlap(&left[left_part],&right[right_part]) != 0u )
						return(SPARK_STATUS_SCHEMA_ERROR);
		}
		if ( SparkGlm5NextPackRangesOverlap(&left[0],&left[1]) != 0u )
			return(SPARK_STATUS_SCHEMA_ERROR);
	}
	return(SPARK_STATUS_OK);
}

static void SparkGlm5NextPackMarkSeen(
	SparkGlm5NextModuleState *state,
	const SparkGlm5NextStagePackEntry *entry)
{
	if ( entry->layer_index == SPARK_GLM5_NEXT_STAGEPACK_GLOBAL_LAYER )
		state->global_seen |= UINT64_C(1) << entry->tensor_kind;
	else if ( entry->layer_index == SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX )
		state->mtp_seen |= UINT64_C(1) << entry->tensor_kind;
	else
		state->layer_seen[entry->layer_index - state->first_layer_index] |= UINT64_C(1) << entry->tensor_kind;
}

static SparkStatus SparkGlm5NextPackAssignLayer(
	SparkGlm5NextModuleState *state,
	SparkGlm5NextLayerWeights *weights,
	const SparkGlm5NextStagePackEntry *entry,
	const void *payload,
	const void *scale)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ATTN_NORM: weights->attn_norm_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_Q_A: weights->q_a_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_Q_A_NORM: weights->q_a_norm_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_Q_B: weights->q_b_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KV_A: weights->kv_a_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KV_A_NORM: weights->kv_a_norm_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KV_B_KEY_TRANSPOSED: weights->kv_b_key_transposed_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KV_B_VALUE: weights->kv_b_value_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ATTN_OUTPUT: weights->attn_output_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_POST_ATTN_NORM: weights->post_attn_norm_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_Q: weights->index_q_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_K: weights->index_k_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_HEAD: weights->index_head_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_NORM_WEIGHT: weights->index_norm_weight_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_NORM_BIAS: weights->index_norm_bias_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_DENSE_GATE_UP: weights->dense_gate_up_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_DENSE_DOWN: weights->dense_down_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ROUTER: weights->router_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ROUTER_CORRECTION: weights->router_correction_f32 = (const float *)payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_UP_GATE: weights->expert_up_gate_payload = payload; weights->expert_up_gate_scale = scale; weights->expert_up_gate_payload_offset = entry->payload_offset; weights->expert_up_gate_scale_offset = entry->scale_offset; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_DOWN: weights->expert_down_payload = payload; weights->expert_down_scale = scale; weights->expert_down_payload_offset = entry->payload_offset; weights->expert_down_scale_offset = entry->scale_offset; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_SHARED_GATE_UP: weights->shared_gate_up_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_SHARED_DOWN: weights->shared_down_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_QKV_BETA: weights->kda_qkv_beta_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_DECAY_GATE_DOWN: weights->kda_decay_gate_down_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_DECAY_UP: weights->kda_decay_up_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_GATE_UP: weights->kda_gate_up_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_Q_CONV: weights->kda_q_conv_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_K_CONV: weights->kda_k_conv_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_V_CONV: weights->kda_v_conv_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_DECAY_BIAS: weights->kda_decay_bias_f32 = (const float *)payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_HEAD_LOG_SCALE: weights->kda_head_log_scale_f32 = (const float *)payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_OUT_NORM: weights->kda_out_norm_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KDA_OUT: weights->kda_out_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_ATTN_FN: weights->hc_attn_fn_f32 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_ATTN_BASE: weights->hc_attn_base_f32 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_ATTN_SCALE: weights->hc_attn_scale_f32 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_FFN_FN: weights->hc_ffn_fn_f32 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_FFN_BASE: weights->hc_ffn_base_f32 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_HC_FFN_SCALE: weights->hc_ffn_scale_f32 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_COMPRESS_APE: weights->index_compress_ape_f32 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_INDEX_COMPRESS_GATE: weights->index_compress_gate_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_MTP_EH_PROJ: state->mtp_eh_proj_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_MTP_ENORM: state->mtp_enorm_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_MTP_HNORM: state->mtp_hnorm_bf16 = payload; break;
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_MTP_SHARED_NORM: state->mtp_shared_norm_bf16 = payload; break;
	default: return(SPARK_STATUS_SCHEMA_ERROR);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextPackAssign(
	SparkGlm5NextModuleState *state,
	const SparkGlm5NextStagePackEntry *entry,
	const void *payload,
	const void *scale)
{
	if ( entry->layer_index == SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX )
		return(SparkGlm5NextPackAssignLayer(state,&state->mtp_layer,entry,payload,scale));
	if ( entry->layer_index != SPARK_GLM5_NEXT_STAGEPACK_GLOBAL_LAYER )
		return(SparkGlm5NextPackAssignLayer(state,&state->layers[entry->layer_index - state->first_layer_index],entry,payload,scale));
	switch ( entry->tensor_kind )
	{
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EMBEDDING: state->embedding_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_FINAL_NORM: state->final_norm_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_GLM5_NEXT_STAGEPACK_TENSOR_LM_HEAD: state->lm_head_bf16 = payload; return(SPARK_STATUS_OK);
	default: return(SPARK_STATUS_SCHEMA_ERROR);
	}
}

typedef struct SparkGlm5NextManifestContext
{
	const SparkGlm5NextStagePackEntry *entries;
	uint32_t count;
} SparkGlm5NextManifestContext;

static SparkStatus SparkGlm5NextManifestPlane(const SparkWeightdManifest *manifest,const SparkGlm5NextStagePackEntry *entry,uint32_t plane)
{
	const SparkWeightdRangeGroup *group;
	const SparkWeightdRange *range;
	uint64_t offset,bytes,per;
	uint32_t expert,index,kind;
	offset = plane == 0u ? entry->payload_offset : entry->scale_offset;
	bytes = plane == 0u ? entry->payload_bytes : entry->scale_bytes;
	if ( entry->group_count == 0u || bytes == 0u || bytes % entry->group_count != 0u )
		return(SPARK_STATUS_SCHEMA_ERROR);
	per = (bytes / entry->group_count);
	kind = ((entry->tensor_kind * 2u) + plane);
	for (expert=0u; expert<entry->group_count; expert++)
	{
		group = SparkWeightdManifestFind(manifest,entry->layer_index,expert);
		if ( group == 0 )
			return(SPARK_STATUS_SCHEMA_ERROR);
		for (index=0u; index<group->range_count; index++)
		{
			range = &manifest->ranges[group->first_range + index];
			if ( range->kind == kind )
				break;
		}
		if ( index == group->range_count || range->offset != (offset + ((uint64_t)expert * per)) || range->bytes != per )
			return(SPARK_STATUS_SCHEMA_ERROR);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextManifestCheck(const SparkWeightdManifest *manifest,void *opaque)
{
	const SparkGlm5NextManifestContext *context = (const SparkGlm5NextManifestContext *)opaque;
	const SparkGlm5NextStagePackEntry *entry;
	SparkStatus status;
	uint64_t expected = 0u;
	uint32_t index,plane;
	for (index=0u; index<context->count; index++)
	{
		entry = &context->entries[index];
		if ( entry->tensor_kind != SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_UP_GATE && entry->tensor_kind != SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_DOWN )
			continue;
		if ( entry->weight_codec != SPARK_WEIGHT_CODEC_FP8_E4M3 )
			return(SPARK_STATUS_UNSUPPORTED);
		for (plane=0u; plane<2u; plane++)
		{
			status = SparkGlm5NextManifestPlane(manifest,entry,plane);
			if ( status != SPARK_STATUS_OK )
				return(status);
			expected += entry->group_count;
		}
	}
	return(expected == manifest->range_count ? SPARK_STATUS_OK : SPARK_STATUS_SCHEMA_ERROR);
}

static SparkStatus SparkGlm5NextLazyOpen(SparkGlm5NextModuleState *state,const char *path,uint64_t bytes,const SparkGlm5NextStagePackEntry *entries,uint32_t count)
{
	SparkWeightdLazyAttachRequest request;
	SparkGlm5NextManifestContext context = {entries,count};
	SparkStatus status;
	const char *digest;
	uint64_t spine_budget;
	status = SparkWeightdAttachRequested();
	if ( status == SPARK_STATUS_BUSY )
		return(SPARK_STATUS_OK);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( state->mtp_enabled != 0u )
		return(SPARK_STATUS_UNSUPPORTED);
	memset(&request,0,sizeof(request));
	digest = getenv(SPARK_WEIGHTD_ATTACH_ENV_SHA256);
	if ( digest == 0 || strlen(digest) != 64u || strlen(path) >= sizeof(request.pack_path) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memcpy(request.identity.pack_sha256,digest,65u);
	(void)snprintf(request.identity.model,sizeof(request.identity.model),"%s",SPARK_GLM5_NEXT_MODULE_TAG);
	(void)snprintf(request.identity.revision,sizeof(request.identity.revision),"%s",state->model_revision);
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = bytes;
	request.identity.topology = state->tp_degree;
	memcpy(request.pack_path,path,strlen(path) + 1u);
	status = SparkStageModuleEnvironmentUnsigned64(SPARK_GLM5_NEXT_MODULE_TAG,"SPARK_WEIGHTD_EXPERT_POOL_BYTES",1u,UINT64_MAX,&request.expert_pool_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned64(SPARK_GLM5_NEXT_MODULE_TAG,"SPARK_WEIGHTD_SPINE_BUDGET_BYTES",1u,UINT64_MAX,&spine_budget);
	if ( status == SPARK_STATUS_OK )
		status = SparkWeightdLazyPackCreateChecked(getenv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET),&request,spine_budget,SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS,SparkGlm5NextManifestCheck,&context,&state->lazy_pack);
	return(status);
}

static SparkStatus SparkGlm5NextPackLoadEntry(
	SparkGlm5NextModuleState *state,
	FILE *file,
	const SparkGlm5NextStagePackEntry *entry)
{
	void *payload,*scale;
	SparkStatus status;
	payload = 0;
	scale = 0;
	if ( state->lazy_pack != 0 )
	{
		if ( entry->tensor_kind == SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_UP_GATE || entry->tensor_kind == SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_DOWN )
			return(SparkGlm5NextPackAssign(state,entry,0,0));
		status = SparkWeightdLazyPackSlice(state->lazy_pack,entry->payload_offset,entry->payload_bytes,(const void **)&payload);
		if ( status == SPARK_STATUS_OK && entry->scale_bytes != 0u )
			status = SparkWeightdLazyPackSlice(state->lazy_pack,entry->scale_offset,entry->scale_bytes,(const void **)&scale);
		if ( status == SPARK_STATUS_OK )
			status = SparkGlm5NextPackAssign(state,entry,payload,scale);
		return(status);
	}
	status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,entry->payload_offset,entry->payload_bytes,&payload);
	if ( status == SPARK_STATUS_OK && entry->scale_bytes != 0u )
		status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,entry->scale_offset,entry->scale_bytes,&scale);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextPackAssign(state,entry,payload,scale);
	return(status);
}

static uint64_t SparkGlm5NextExpectedLayerMask(
	const SparkGlm5NextModuleState *state,
	uint32_t layer_index)
{
	SparkGlm5NextStagePackTensorShape shape;
	uint64_t mask;
	uint32_t kind;
	mask = 0u;
	for (kind=SPARK_GLM5_NEXT_STAGEPACK_TENSOR_ATTN_NORM; kind<SPARK_GLM5_NEXT_STAGEPACK_TENSOR_KIND_COUNT; kind++)
		if ( SparkGlm5NextStagePackExpectedShape(kind,layer_index,state->expert_weight_codec,state->tp_degree,&shape) == 0 )
			mask |= UINT64_C(1) << kind;
	return(mask);
}

static uint64_t SparkGlm5NextExpectedGlobalMask(const SparkGlm5NextModuleState *state)
{
	uint64_t mask;
	mask = 0u;
	if ( state->owns_embedding != 0u )
		mask |= UINT64_C(1) << SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EMBEDDING;
	if ( state->owns_final_head != 0u )
		mask |= (UINT64_C(1) << SPARK_GLM5_NEXT_STAGEPACK_TENSOR_FINAL_NORM) | (UINT64_C(1) << SPARK_GLM5_NEXT_STAGEPACK_TENSOR_LM_HEAD);
	return(mask);
}

static SparkStatus SparkGlm5NextPackValidateInventory(const SparkGlm5NextModuleState *state)
{
	uint64_t expected_mtp;
	uint32_t local;
	if ( state->global_seen != SparkGlm5NextExpectedGlobalMask(state) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	expected_mtp = state->pack_has_mtp != 0u ?
		SparkGlm5NextExpectedLayerMask(state,SPARK_GLM5_NEXT_MODEL_MTP_LAYER_INDEX) : 0u;
	if ( state->mtp_seen != expected_mtp )
		return(SPARK_STATUS_SCHEMA_ERROR);
	for (local=0u; local<state->layer_count; local++)
		if ( state->layer_seen[local] != SparkGlm5NextExpectedLayerMask(state,state->first_layer_index + local) )
			return(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextPackLoad(
	SparkGlm5NextModuleState *state,
	const char *path)
{
	SparkGlm5NextStagePackHeader header;
	SparkGlm5NextStagePackEntry entries[SPARK_GLM5_NEXT_STAGEPACK_MAX_TENSOR_COUNT];
	SparkGlm5NextStagePackTensorShape shape;
	FILE *file;
	uint64_t file_bytes;
	uint32_t index;
	SparkStatus status;
	file = fopen(path,"rb");
	if ( file == 0 )
		return(SPARK_STATUS_NOT_FOUND);
	memset(&header,0,sizeof(header));
	memset(entries,0,sizeof(entries));
	status = SparkGlm5NextPackFileSize(file,&file_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModulePackRead(SPARK_GLM5_NEXT_MODULE_TAG,file,0u,&header,sizeof(header));
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextPackValidateHeader(state,&header,file_bytes);
	if ( status == SPARK_STATUS_OK )
		state->pack_has_mtp = (header.flags & SPARK_GLM5_NEXT_STAGEPACK_FLAG_MTP) != 0u ? 1u : 0u;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModulePackRead(SPARK_GLM5_NEXT_MODULE_TAG,file,header.directory_offset,entries,(uint64_t)header.tensor_count * sizeof(entries[0]));
	for (index=0u; status==SPARK_STATUS_OK && index<header.tensor_count; index++)
	{
		status = SparkGlm5NextPackValidateEntryGeometry(state,&header,&entries[index],&shape);
		if ( status == SPARK_STATUS_OK )
			SparkGlm5NextPackMarkSeen(state,&entries[index]);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextPackValidateRanges(entries,header.tensor_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextPackValidateInventory(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextLazyOpen(state,path,file_bytes,entries,header.tensor_count);
	for (index=0u; status==SPARK_STATUS_OK && index<header.tensor_count; index++)
		status = SparkGlm5NextPackLoadEntry(state,file,&entries[index]);
	if ( fclose(file) != 0 && status == SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	return(status);
}

static SparkStatus SparkGlm5NextAllocateBytes(
	SparkGlm5NextModuleState *state,
	uint64_t count,
	uint64_t width,
	uint64_t element_bytes,
	void **pointer)
{
	uint64_t bytes;
	if ( state == 0 || pointer == 0 || count == 0u || width == 0u || element_bytes == 0u || count > UINT64_MAX / width || count * width > UINT64_MAX / element_bytes )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	bytes = count * width * element_bytes;
	return(SparkStageModuleDeviceAllocate(&state->ledger,bytes,pointer));
}

static SparkStatus SparkGlm5NextAllocateRows(
	SparkGlm5NextModuleState *state,
	uint64_t rows,
	uint64_t columns,
	void **pointer)
{
	return(SparkGlm5NextAllocateBytes(state,rows,columns,sizeof(uint16_t),pointer));
}

static SparkStatus SparkGlm5NextAllocateSlotHost(SparkGlm5NextExecutionSlot *slot)
{
	uint32_t *cursor;
	uint64_t rows,words,bytes;
	cudaError_t error;
	if ( slot == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	rows = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT;
	words = (rows * 4u) + SPARK_GLM5_NEXT_KV_ACCESS_ERROR_WORD_COUNT + SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT + 1u;
	bytes = words * sizeof(uint32_t);
	error = cudaHostAlloc(&slot->host_staging,bytes,cudaHostAllocPortable);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"host_staging"));
	memset(slot->host_staging,0,bytes);
	cursor = (uint32_t *)slot->host_staging;
	slot->host_token_ids = cursor;
	cursor += rows;
	slot->host_resident_slots = cursor;
	cursor += rows;
	slot->host_positions = cursor;
	cursor += rows;
	slot->host_output_token_ids = cursor;
	cursor += rows;
	slot->host_kv_access_error = cursor;
	cursor += SPARK_GLM5_NEXT_KV_ACCESS_ERROR_WORD_COUNT;
	slot->host_group_row_offset = cursor;
	bytes = ((rows + 1u) * 2u + rows) * sizeof(uint32_t);
	error = cudaHostAlloc((void **)&slot->host_run_begin,bytes,cudaHostAllocPortable);
	if ( error != cudaSuccess )
	{
		(void)cudaFreeHost(slot->host_staging);
		slot->host_staging = 0;
		return(SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"host_run_staging"));
	}
	memset(slot->host_run_begin,0,bytes);
	cursor = (uint32_t *)slot->host_run_begin;
	cursor += rows + 1u;
	slot->host_run_state_index = cursor;
	error = cudaEventCreateWithFlags((cudaEvent_t *)&slot->route_ready_event,cudaEventDisableTiming);
	return(SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"route_ready_event"));
}

static void SparkGlm5NextReleaseSlotHost(SparkGlm5NextModuleState *state)
{
	uint32_t index;
	if ( state == 0 )
		return;
	for (index=0u; index<state->pipeline_slot_count; index++)
	{
		if ( state->slots[index].route_ready_event != 0 )
			(void)cudaEventDestroy((cudaEvent_t)state->slots[index].route_ready_event);
		state->slots[index].route_ready_event = 0;
		state->slots[index].route_recorded = 0u;
		if ( state->slots[index].host_staging != 0 )
			(void)cudaFreeHost(state->slots[index].host_staging);
		state->slots[index].host_staging = 0;
		if ( state->slots[index].host_run_begin != 0 )
			(void)cudaFreeHost(state->slots[index].host_run_begin);
		state->slots[index].host_run_begin = 0;
		state->slots[index].host_run_state_index = 0;
	}
}

static SparkStatus SparkGlm5NextAllocateSlotMetadata(
	SparkGlm5NextModuleState *state,
	SparkGlm5NextExecutionSlot *slot)
{
	SparkStatus status;
	status = SparkGlm5NextAllocateBytes(state,state->execution_row_capacity,1u,sizeof(uint32_t),(void **)&slot->token_ids);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,state->execution_row_capacity,1u,sizeof(uint32_t),(void **)&slot->resident_slots);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,state->execution_row_capacity,1u,sizeof(uint32_t),(void **)&slot->positions);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,state->execution_row_capacity + 1u,1u,sizeof(uint32_t),(void **)&slot->run_begin);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,state->execution_row_capacity,1u,sizeof(uint32_t),(void **)&slot->run_state_index);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,state->resident_sequence_capacity,1u,sizeof(uint32_t),(void **)&slot->context_lengths);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,2u,1u,sizeof(uint32_t),(void **)&slot->dense_row_offset);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,2u,1u,sizeof(uint32_t),(void **)&slot->dense_tile_prefix);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,1u,sizeof(uint32_t) * 6u,1u,&slot->kv_access_error);
	return(status);
}

static SparkStatus SparkGlm5NextAllocateSlotHidden(
	SparkGlm5NextModuleState *state,
	SparkGlm5NextExecutionSlot *slot)
{
	uint64_t rows;
	SparkStatus status;
	rows = state->execution_row_capacity;
	status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HC_MULT * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->hidden_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->residual_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->normed_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->hc_collapsed_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HC_MULT * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->hc_snapshot_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->hc_mean_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,SPARK_GLM5_NEXT_MODEL_HC_MIX_DIMENSION,sizeof(float),(void **)&slot->hc_mixes_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,SPARK_GLM5_NEXT_MODEL_HC_MULT,sizeof(float),(void **)&slot->hc_pre_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,SPARK_GLM5_NEXT_MODEL_HC_MULT,sizeof(float),(void **)&slot->hc_post_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,SPARK_GLM5_NEXT_MODEL_HC_MULT * SPARK_GLM5_NEXT_MODEL_HC_MULT,sizeof(float),(void **)&slot->hc_comb_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_QUERY_A_DIMENSION,(void **)&slot->q_compressed_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_QUERY_B_DIMENSION,(void **)&slot->q_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HEAD_COUNT * SPARK_GLM5_NEXT_MODEL_LATENT_DIMENSION,(void **)&slot->query_latent_bf16);
	slot->query_rope_bf16 = 0;
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_DSA_INDEX_QUERY_DIMENSION,(void **)&slot->index_query_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_DSA_INDEX_HEAD_DIMENSION,(void **)&slot->index_key_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_DSA_INDEX_HEAD_COUNT,(void **)&slot->index_head_weight_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_DSA_INDEX_HEAD_DIMENSION,(void **)&slot->index_gate_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION,(void **)&slot->index_packed_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K / SPARK_GLM5_NEXT_MODEL_INDEX_KPOOL,sizeof(uint32_t),(void **)&slot->selected_pools);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,2u * SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION + SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION + SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT,(void **)&slot->fused_qkvb_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,2u * SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK,(void **)&slot->fused_decay_gate_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK,(void **)&slot->kda_decay_latent_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK,(void **)&slot->kda_gate_latent_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT,(void **)&slot->kda_beta_logit);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION,(void **)&slot->kda_gate_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION,(void **)&slot->kda_decay_logit_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->kda_output_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,SPARK_GLM5_NEXT_MODEL_KDA_QKV_DIMENSION,sizeof(float),(void **)&slot->kda_retention);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT,sizeof(float),(void **)&slot->kda_write_gate);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_CACHE_TOKEN_ELEMENTS,(void **)&slot->kv_slot_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HEAD_COUNT * SPARK_GLM5_NEXT_MODEL_LATENT_DIMENSION,(void **)&slot->attention_latent_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HEAD_COUNT * SPARK_GLM5_NEXT_MODEL_VALUE_HEAD_DIMENSION,(void **)&slot->attention_value_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->attention_out_bf16);
	return(status);
}

static SparkStatus SparkGlm5NextAllocateSlotMlp(
	SparkGlm5NextModuleState *state,
	SparkGlm5NextExecutionSlot *slot)
{
	uint64_t rows,packed_rows;
	SparkStatus status;
	rows = state->execution_row_capacity;
	packed_rows = rows * SPARK_GLM5_NEXT_MODEL_MOE_TOP_K;
	status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_MOE_ROUTED_GATE_UP_DIMENSION,(void **)&slot->gate_up_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_MOE_TOP_K * SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION,(void **)&slot->intermediate_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,packed_rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->expert_out_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->shared_out_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT,sizeof(float),(void **)&slot->router_logits_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,state->max_sequence_positions / SPARK_GLM5_NEXT_MODEL_INDEX_KPOOL,sizeof(float),(void **)&slot->selection_scores_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_BLOCKS(rows,SPARK_GLM5_NEXT_MODEL_HEAD_COUNT / state->tp_degree),SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_FLOATS,sizeof(float),(void **)&slot->attention_split_partials_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,SPARK_GLM5_NEXT_MODEL_INDEX_OUTPUT_WIDTH,sizeof(uint32_t),(void **)&slot->selected_positions);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,packed_rows,1u,sizeof(uint32_t),(void **)&slot->route_expert);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,packed_rows,1u,sizeof(float),(void **)&slot->route_weight);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,packed_rows,1u,sizeof(uint32_t),(void **)&slot->route_source_token);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,packed_rows,1u,sizeof(uint32_t),(void **)&slot->route_packed_row);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT + 1u,1u,sizeof(uint32_t),(void **)&slot->group_row_offset);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT + 1u,1u,sizeof(uint32_t),(void **)&slot->group_tile_prefix_w1);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT + 1u,1u,sizeof(uint32_t),(void **)&slot->group_tile_prefix_w2);
	return(status);
}

static SparkStatus SparkGlm5NextAllocateSlotHead(
	SparkGlm5NextModuleState *state,
	SparkGlm5NextExecutionSlot *slot)
{
	uint64_t rows,tiles;
	SparkStatus status;
	rows = state->execution_row_capacity;
	tiles = SparkCeilDivU64(SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT,SPARK_GLM5_NEXT_HEAD_TILE);
	status = SparkGlm5NextAllocateBytes(state,rows,tiles,sizeof(float),(void **)&slot->head_candidate_score);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,tiles,sizeof(uint32_t),(void **)&slot->head_candidate_token);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,1u,sizeof(uint32_t),(void **)&slot->output_token);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,1u,sizeof(float),(void **)&slot->output_score);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,rows,1u,sizeof(uint64_t),(void **)&slot->head_maxloc_u64);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
	{
		uint64_t shard_rows = SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT / state->tp_degree;
		status = SparkGlm5NextAllocateBytes(state,1u,SparkHeadCertifiedFp8ScratchBytes(shard_rows,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION),1u,(void **)&slot->head_certified_scratch);
		if ( status == SPARK_STATUS_OK )
			status = SparkGlm5NextAllocateBytes(state,1u,SparkHeadCertifiedFp8CandidateBytes(shard_rows),1u,(void **)&slot->head_certified_candidates);
		if ( status == SPARK_STATUS_OK )
			status = SparkGlm5NextAllocateBytes(state,1u,1u,sizeof(uint32_t),(void **)&slot->head_screened_count);
	}
	return(status);
}

static SparkStatus SparkGlm5NextAllocateSlots(SparkGlm5NextModuleState *state)
{
	uint32_t index;
	SparkStatus status;
	status = SPARK_STATUS_OK;
	for (index=0u; status==SPARK_STATUS_OK && index<state->pipeline_slot_count; index++)
	{
		state->slots[index].stream = state->execution_stream;
		status = SparkGlm5NextAllocateSlotHost(&state->slots[index]);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateSlotMetadata(state,&state->slots[index]);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateSlotHidden(state,&state->slots[index]);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateSlotMlp(state,&state->slots[index]);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateSlotHead(state,&state->slots[index]);
	}
	return(status);
}

static SparkStatus SparkGlm5NextAllocateMtp(SparkGlm5NextModuleState *state)
{
	SparkGlm5NextKdaReplayLayout layout;
	uint64_t kv_pool_bytes,index_pool_bytes,replay_bytes,steps_bytes,conv_bytes;
	uint32_t index,rank_heads,step;
	SparkStatus status;
	if ( state->mtp_enabled == 0u )
		return(SPARK_STATUS_OK);
	if ( state->execution_row_capacity < SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH + 1u )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	rank_heads = SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT / state->tp_degree;
	layout = SparkGlm5NextKdaReplayLayoutFor(rank_heads,SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH + 1u);
	state->kda_replay_layer_bytes = layout.layer_bytes;
	kv_pool_bytes = (uint64_t)SPARK_GLM5_NEXT_MODEL_KV_PAGE_SLOTS * SPARK_GLM5_NEXT_MODEL_KV_SLOT_BYTES;
	index_pool_bytes = (uint64_t)SPARK_GLM5_NEXT_MODEL_KV_PAGE_SLOTS * SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u * SPARK_GLM5_NEXT_MODEL_DSA_LAYER_COUNT;
	replay_bytes = layout.layer_bytes * state->kda_layer_count;
	steps_bytes = (uint64_t)state->kda_layer_count * (SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH + 1u) * SPARK_GLM5_NEXT_MTP_REPLAY_STEP_BYTES;
	conv_bytes = (uint64_t)(SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH + 1u) * rank_heads * SPARK_GLM5_NEXT_MODEL_KDA_HEAD_KEY_DIMENSION * SPARK_GLM5_NEXT_MODEL_BF16_ELEMENT_BYTES;
	state->mtp_lane_armed = (uint8_t *)calloc(state->resident_sequence_capacity,1u);
	if ( state->mtp_lane_armed == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkGlm5NextAllocateBytes(state,state->resident_sequence_capacity,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,sizeof(uint16_t),(void **)&state->mtp_lane_hidden_bf16);
	for (index=0u; status==SPARK_STATUS_OK && index<state->pipeline_slot_count; index++)
	{
		SparkGlm5NextExecutionSlot *slot = &state->slots[index];
		status = SparkGlm5NextAllocateRows(state,1u,SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->mtp_hidden_bf16);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateRows(state,1u,2u * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,(void **)&slot->mtp_concat_bf16);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,1u,kv_pool_bytes,1u,(void **)&slot->mtp_kv_pool);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,1u,index_pool_bytes,1u,(void **)&slot->mtp_index_pool);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH,sizeof(uint32_t),1u,(void **)&slot->mtp_positions);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH,sizeof(uint32_t),1u,(void **)&slot->mtp_context);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,1u,sizeof(uint32_t),1u,(void **)&slot->mtp_page_table);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,1u,sizeof(uint32_t),1u,(void **)&slot->mtp_sequence);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,1u,sizeof(uint32_t),1u,(void **)&slot->mtp_committed);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,1u,steps_bytes,1u,&slot->mtp_replay_steps);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,1u,conv_bytes,1u,(void **)&slot->mtp_conv_scratch);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm5NextAllocateBytes(state,1u,replay_bytes,1u,(void **)&slot->kda_replay_pool);
		if ( status != SPARK_STATUS_OK )
			break;
		{
			uint32_t meta[2u * SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH];
			cudaError_t error;
			for ( step = 0u; step < SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH; ++step )
			{
				meta[step] = step;
				meta[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH + step] = step + 1u;
			}
			error = cudaMemcpy(slot->mtp_positions,meta,SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH * sizeof(uint32_t),cudaMemcpyHostToDevice);
			if ( error == cudaSuccess )
				error = cudaMemcpy(slot->mtp_context,meta + SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH,SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH * sizeof(uint32_t),cudaMemcpyHostToDevice);
			if ( error == cudaSuccess )
				error = cudaMemset(slot->mtp_page_table,0,sizeof(uint32_t));
			if ( error == cudaSuccess )
				error = cudaMemset(slot->mtp_sequence,0,sizeof(uint32_t));
			status = SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"mtp_meta_init");
		}
	}
	return(status);
}

static SparkStatus SparkGlm5NextBuildPageTable(SparkGlm5NextModuleState *state)
{
	uint64_t entries;
	SparkStatus status;
	cudaError_t error;
	state->pages_per_sequence = SparkCeilDivU32(state->max_sequence_positions,64u);
	if ( state->pages_per_sequence == 0u || state->resident_sequence_capacity > UINT32_MAX / state->pages_per_sequence )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->page_count = state->resident_sequence_capacity * state->pages_per_sequence;
	entries = state->page_count;
	state->page_table_shadow = (uint32_t *)malloc(entries * sizeof(uint32_t));
	if ( state->page_table_shadow == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	memset(state->page_table_shadow,0xff,entries * sizeof(uint32_t));
	status = SparkGlm5NextAllocateBytes(state,entries,1u,sizeof(uint32_t),(void **)&state->page_table);
	if ( status == SPARK_STATUS_OK )
	{
		error = cudaMemset(state->page_table,0xff,entries * sizeof(uint32_t));
		status = SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"page_table");
	}
	return(status);
}

static SparkStatus SparkGlm5NextDevicePageCopy(
	void *context,
	uint32_t direction,
	uintptr_t device_address,
	void *host_address,
	uint64_t bytes)
{
	SparkGlm5NextModuleState *state;
	cudaError_t error;
	state = (SparkGlm5NextModuleState *)context;
	if ( state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( direction == SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST )
		error = cudaMemcpy(host_address,(const void *)device_address,(size_t)bytes,cudaMemcpyDeviceToHost);
	else if ( direction == SPARK_KV_PAGE_STORE_COPY_HOST_TO_DEVICE )
		error = cudaMemcpy((void *)device_address,host_address,(size_t)bytes,cudaMemcpyHostToDevice);
	else
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"kv_page_copy"));
}

// Checkpoints pack all KDA layers, then all Q, K and V convolution layers.
// The caller owns the resident slot until the complete transfer succeeds.
static inline SparkStatus SparkGlm5NextRecurrentCopy(SparkGlm5NextModuleState *state,uint32_t direction,uint32_t slot,void *host,uint64_t bytes)
{
	SparkKvLayeredPageLayout layout;
	uint8_t *pools[4];
	uint64_t strides[4],payloads[4],total = 0u,offset = 0u;
	uint32_t part;
	SparkStatus status;
	if ( state == 0 || host == 0 || state->resident_sequence_capacity == 0u || state->kda_layer_count == 0u || slot >= state->resident_sequence_capacity )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( direction != SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST && direction != SPARK_KV_PAGE_STORE_COPY_HOST_TO_DEVICE )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	pools[0] = state->kda_state_pools;
	pools[1] = state->kda_q_window_pool;
	pools[2] = state->kda_k_window_pool;
	pools[3] = state->kda_v_window_pool;
	strides[0] = state->kda_state_layer_stride_bytes;
	strides[1] = strides[2] = strides[3] = state->kda_window_layer_stride_bytes;
	for (part=0u; part<4u; part++)
	{
		if ( pools[part] == 0 || strides[part] == 0u || strides[part] % state->resident_sequence_capacity != 0u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		if ( strides[part] > (UINTPTR_MAX - (uintptr_t)pools[part]) / state->kda_layer_count )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		payloads[part] = (strides[part] / state->resident_sequence_capacity) * state->kda_layer_count;
		if ( payloads[part] > UINT64_MAX - total )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		total += payloads[part];
	}
	if ( bytes != total || bytes > UINTPTR_MAX - (uintptr_t)host )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	layout.layer_count = state->kda_layer_count;
	layout.page_count = state->resident_sequence_capacity;
	for (part=0u; part<4u; part++)
	{
		layout.device_base = (uintptr_t)pools[part];
		layout.device_bytes = strides[part] * layout.layer_count;
		layout.layer_stride_bytes = strides[part];
		layout.layer_page_bytes = strides[part] / layout.page_count;
		status = SparkKvPageStoreCopyLayered(&layout,direction,slot,(uint8_t *)host + offset,payloads[part],SparkGlm5NextDevicePageCopy,state);
		if ( status != SPARK_STATUS_OK )
			return(status);
		offset += payloads[part];
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextPageCopy(
	void *context,
	uint32_t direction,
	uintptr_t device_address,
	void *host_address,
	uint64_t bytes)
{
	SparkGlm5NextModuleState *state;
	SparkKvLayeredPageLayout layout;
	uint64_t offset,packed_page_bytes;
	state = (SparkGlm5NextModuleState *)context;
	if ( state == 0 || state->page_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->index_layer_count != 0u && state->index_layer_stride_bytes > UINT64_MAX / state->index_layer_count )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	layout.device_base = (uintptr_t)state->kv_cache;
	layout.layer_stride_bytes = state->kv_layer_stride_bytes;
	layout.layer_count = state->kv_layer_count;
	layout.page_count = state->page_count;
	if ( device_address >= (uintptr_t)state->index_cache && device_address - (uintptr_t)state->index_cache < state->index_layer_stride_bytes * state->index_layer_count )
	{
		layout.device_base = (uintptr_t)state->index_cache;
		layout.layer_stride_bytes = state->index_layer_stride_bytes;
		layout.layer_count = state->index_layer_count;
	}
	if ( layout.layer_count == 0u || layout.layer_stride_bytes % layout.page_count != 0u || layout.layer_stride_bytes > UINT64_MAX / layout.layer_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	layout.device_bytes = layout.layer_stride_bytes * layout.layer_count;
	layout.layer_page_bytes = layout.layer_stride_bytes / layout.page_count;
	packed_page_bytes = layout.layer_page_bytes * layout.layer_count;
	if ( device_address < layout.device_base || device_address - layout.device_base >= layout.device_bytes || packed_page_bytes == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	// Arena block addresses name packed payloads. Translate their page index
	// to the native layer-major allocation before a device copy dereferences it.
	offset = device_address - layout.device_base;
	if ( offset % packed_page_bytes != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkKvPageStoreCopyLayered(&layout,direction,(uint32_t)(offset / packed_page_bytes),host_address,bytes,SparkGlm5NextDevicePageCopy,state));
}

static SparkStatus SparkGlm5NextBackingCapacity(SparkGlm5NextModuleState *state,uint64_t kv_page_bytes)
{
	uint64_t window_bytes,state_bytes,total;
	if ( state->page_count == 0u || state->resident_sequence_capacity == 0u || kv_page_bytes == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	state->recurrent_page_bytes = 0u;
	if ( state->kda_layer_count != 0u )
	{
		if ( state->kda_state_layer_stride_bytes % state->resident_sequence_capacity != 0u || state->kda_window_layer_stride_bytes % state->resident_sequence_capacity != 0u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		state_bytes = state->kda_state_layer_stride_bytes / state->resident_sequence_capacity;
		window_bytes = state->kda_window_layer_stride_bytes / state->resident_sequence_capacity;
		if ( state_bytes == 0u || window_bytes == 0u || window_bytes > (UINT64_MAX - state_bytes) / 3u )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		state_bytes += 3u * window_bytes;
		if ( state_bytes > (UINT64_MAX - kv_page_bytes) / state->kda_layer_count )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		state->recurrent_page_bytes = state_bytes * state->kda_layer_count;
	}
	total = kv_page_bytes + state->recurrent_page_bytes;
	if ( total > INT64_MAX / state->page_count || state->recurrent_page_bytes > SIZE_MAX / 2u )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	total *= state->page_count;
	if ( state->kv_backing_maximum_bytes != 0u && state->kv_backing_maximum_bytes < total )
	{
		fprintf(stderr,"GLM cache backing budget insufficient: need %llu bytes for %u pages, configured %llu\n",(unsigned long long)total,state->page_count,(unsigned long long)state->kv_backing_maximum_bytes);
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextRecurrentInitialize(SparkGlm5NextModuleState *state,const char *backing_path)
{
	SparkKvPageStoreConfiguration config = {0};
	SparkStatus status;
	if ( state->kda_layer_count == 0u )
		return(SPARK_STATUS_OK);
	if ( state->recurrent_page_bytes == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( cudaHostAlloc((void **)&state->recurrent_staging,2u * state->recurrent_page_bytes,cudaHostAllocPortable) != cudaSuccess )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	config.abi_version = SPARK_KV_PAGE_STORE_ABI_VERSION;
	config.descriptor_bytes = SPARK_KV_PAGE_STORE_CONFIGURATION_BYTES;
	config.flags = SPARK_KV_PAGE_STORE_FLAG_ANONYMOUS;
	config.logical_page_capacity = state->page_count;
	config.transfer_capacity = 1u;
	config.page_bytes = state->recurrent_page_bytes;
	config.maximum_backing_bytes = state->page_count * state->recurrent_page_bytes;
	config.backing_path = backing_path;
	// First half is caller-owned gather/scatter storage; the worker uses the second.
	config.staging_address = state->recurrent_staging + state->recurrent_page_bytes;
	config.staging_bytes = state->recurrent_page_bytes;
	status = SparkKvPageStoreInitialize(&state->recurrent_store,&config);
	if ( status == SPARK_STATUS_OK )
		status = SparkKvPageCacheAttachStateStore(&state->kv_page_cache,&state->recurrent_store);
	return(status);
}

static SparkStatus SparkGlm5NextKvInitialize(SparkGlm5NextModuleState *state)
{
	SparkKvModelTable table;
	uint64_t block_bytes,index_block_bytes,payload_bytes;
	uint64_t lane_page_entries;
	SparkStatus status;
	if ( pthread_mutex_init(&state->kv_mutex,0) != 0 )
		return(SPARK_STATUS_INTERNAL_ERROR);
	state->kv_mutex_initialized = 1u;
	if ( state->kv_layer_count == 0u )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	block_bytes = (uint64_t)SPARK_GLM5_NEXT_KV_BLOCK_TOKEN_COUNT *
		(uint64_t)state->kv_layer_count * SPARK_GLM5_NEXT_KV_ARENA_HEAD_DIM *
		SPARK_GLM5_NEXT_KV_BYTES_PER_SCALAR;
	index_block_bytes = (uint64_t)SPARK_GLM5_NEXT_KV_BLOCK_TOKEN_COUNT * state->index_layer_count * SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u;
	if ( index_block_bytes > UINT64_MAX - block_bytes )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	payload_bytes = block_bytes + index_block_bytes;
	status = SparkGlm5NextBackingCapacity(state,payload_bytes);
	if ( status != SPARK_STATUS_OK )
		return(status);
	lane_page_entries = (uint64_t)state->resident_sequence_capacity *
		state->pages_per_sequence;
	state->kv_blocks = (SparkKvCacheBlock *)calloc(state->page_count,sizeof(*state->kv_blocks));
	state->kv_resident_slot_logical_block_indices = (uint32_t *)calloc(state->page_count,sizeof(*state->kv_resident_slot_logical_block_indices));
	state->kv_entries = (SparkKvPageCacheEntry *)calloc(state->page_count,sizeof(*state->kv_entries));
	state->kv_sequences = (SparkKvPageCacheSequence *)calloc(state->resident_sequence_capacity,sizeof(*state->kv_sequences));
	state->kv_hash_bucket_heads = (uint32_t *)calloc(state->page_count,sizeof(*state->kv_hash_bucket_heads));
	state->kv_entry_indices_by_logical_page = (uint32_t *)calloc(state->page_count,sizeof(*state->kv_entry_indices_by_logical_page));
	state->kv_page_staging = (uint8_t *)malloc((size_t)payload_bytes);
	state->kv_lane_logical_pages = (uint32_t *)calloc((size_t)lane_page_entries,sizeof(*state->kv_lane_logical_pages));
	state->kv_lane_transactions = (SparkKvLaneTransaction *)calloc(state->resident_sequence_capacity,sizeof(*state->kv_lane_transactions));
	if ( cudaHostAlloc((void **)&state->kv_lane_physical_pages,lane_page_entries * sizeof(uint32_t),cudaHostAllocPortable) != cudaSuccess )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( state->kv_blocks == 0 || state->kv_resident_slot_logical_block_indices == 0 || state->kv_entries == 0 || state->kv_sequences == 0 || state->kv_hash_bucket_heads == 0 || state->kv_entry_indices_by_logical_page == 0 || state->kv_page_staging == 0 || state->kv_lane_logical_pages == 0 || state->kv_lane_transactions == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->kv_transactions.cache = &state->kv_page_cache;
	state->kv_transactions.lanes = state->kv_lane_transactions;
	state->kv_transactions.logical_pages = state->kv_lane_logical_pages;
	state->kv_transactions.physical_pages = state->kv_lane_physical_pages;
	state->kv_transactions.page_capacity = state->pages_per_sequence;

	memset(&table,0,sizeof(table));
	table.abi_version = SPARK_KV_MODEL_TABLE_ABI_VERSION;
	table.descriptor_bytes = SPARK_KV_MODEL_TABLE_BYTES;
	SparkGlm5NextKvFillCapacityRequest(&table.capacity_request);
	table.capacity_request.layer_count = state->kv_layer_count;
	table.capacity_request.index_key_layer_count = state->index_layer_count;
	table.capacity_request.index_key_dimension = SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION;
	table.capacity_request.index_key_bytes_per_scalar = 2u;

	table.arena_configuration.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	table.arena_configuration.descriptor_bytes = SPARK_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
	table.arena_configuration.logical_block_count = state->page_count;
	table.arena_configuration.block_token_count = SPARK_GLM5_NEXT_KV_BLOCK_TOKEN_COUNT;
	table.arena_configuration.resident_block_capacity = state->page_count;
	table.arena_configuration.layer_count = state->kv_layer_count;
	table.arena_configuration.kv_head_count = SPARK_GLM5_NEXT_KV_ARENA_KV_HEAD_COUNT;
	table.arena_configuration.head_dim = SPARK_GLM5_NEXT_KV_ARENA_HEAD_DIM;
	table.arena_configuration.bytes_per_scalar = SPARK_GLM5_NEXT_KV_BYTES_PER_SCALAR;
	table.arena_configuration.key_device_base = state->kv_cache;
	table.arena_configuration.value_device_base = state->index_cache;
	table.arena_configuration.value_block_stride_bytes = index_block_bytes;
	table.arena_configuration.blocks = state->kv_blocks;
	table.arena_configuration.resident_slot_logical_block_indices = state->kv_resident_slot_logical_block_indices;

	table.page_store_config.abi_version = SPARK_KV_PAGE_STORE_ABI_VERSION;
	table.page_store_config.descriptor_bytes = SPARK_KV_PAGE_STORE_CONFIGURATION_BYTES;
	table.page_store_config.flags = SPARK_KV_PAGE_STORE_FLAG_ANONYMOUS;
	table.page_store_config.logical_page_capacity = state->page_count;
	table.page_store_config.transfer_capacity = state->page_count < 2u ? state->page_count : 2u;
	table.page_store_config.page_bytes = payload_bytes;
	if ( state->kv_backing_directory != 0 && state->kv_backing_directory[0] != '\0' )
		table.page_store_config.backing_path = state->kv_backing_directory;
	else
	{
		(void)snprintf(state->kv_backing_default,sizeof(state->kv_backing_default),
			"/tmp/sparkpipe_glm5_next_kv_%s",state->model_revision);
		mkdir(state->kv_backing_default,0700);
		table.page_store_config.backing_path = state->kv_backing_default;
	}
	table.page_store_config.maximum_backing_bytes = state->page_count * payload_bytes;
	table.page_store_config.staging_address = state->kv_page_staging;
	table.page_store_config.staging_bytes = payload_bytes;
	table.page_store_config.copy_function = SparkGlm5NextPageCopy;
	table.page_store_config.copy_context = state;

	table.sequence_capacity = state->resident_sequence_capacity;
	table.entry_capacity = state->page_count;
	table.hash_bucket_count = state->page_count;
	table.entries = state->kv_entries;
	table.sequences = state->kv_sequences;
	table.hash_bucket_heads = state->kv_hash_bucket_heads;
	table.entry_indices_by_logical_page = state->kv_entry_indices_by_logical_page;
	table.model_id = "glm5_next";
	table.model_revision = state->model_revision;
	table.cache_layout_fingerprint = "kv-bf16-index-packed-layer-major-gather-v1";

	status = SparkKvBackendInitialize(&table,&state->kv_arena,&state->kv_page_cache,&state->kv_page_store);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( state->kv_arena.key_block_stride_bytes != block_bytes ||
		state->kv_arena.value_block_stride_bytes != index_block_bytes ||
		state->kv_arena.logical_block_count != state->page_count ||
		state->kv_layer_stride_bytes == 0u ||
		block_bytes != ( state->kv_layer_stride_bytes /
				(uint64_t)state->page_count ) *
			(uint64_t)state->kv_layer_count ||
		(uint64_t)state->page_count * block_bytes !=
			state->kv_layer_stride_bytes * (uint64_t)state->kv_layer_count )
		return(SPARK_STATUS_INTERNAL_ERROR);
	return(SparkGlm5NextRecurrentInitialize(state,table.page_store_config.backing_path));
}

static SparkStatus SparkGlm5NextAllocateCaches(SparkGlm5NextModuleState *state)
{
	uint64_t main_page_bytes,index_page_bytes;
	uint64_t main_total,index_total;
	uint32_t local;
	SparkStatus status;
	uint64_t kda_window_stride;
	uint64_t kda_total,window_total;
	state->index_layer_count = 0u;
	state->kv_layer_count = 0u;
	state->kda_layer_count = 0u;
	for (local=0u; local<state->layer_count; local++)
	{
		uint32_t layer = state->first_layer_index + local;
		state->index_ordinal_by_local_layer[local] = SPARK_GLM5_NEXT_NO_INDEX_ORDINAL;
		state->kv_ordinal_by_local_layer[local] = SPARK_GLM5_NEXT_NO_INDEX_ORDINAL;
		state->kda_ordinal_by_local_layer[local] = SPARK_GLM5_NEXT_NO_INDEX_ORDINAL;
		if ( layer < SPARK_GLM5_NEXT_MODEL_LAYER_COUNT )
		{
			if ( SparkGlm5NextStagePackLayerIsDsa(layer) != 0u )
			{
				state->kv_ordinal_by_local_layer[local] = state->kv_layer_count++;
				state->index_ordinal_by_local_layer[local] = state->index_layer_count++;
			}
			else if ( SparkGlm5NextStagePackLayerIsKda(layer) != 0u )
				state->kda_ordinal_by_local_layer[local] = state->kda_layer_count++;
		}
	}
	status = SparkGlm5NextBuildPageTable(state);
	main_page_bytes = (uint64_t)64u * SPARK_GLM5_NEXT_MODEL_KV_SLOT_BYTES;
	index_page_bytes = (uint64_t)64u *
		SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION * 2u;
	// The model constant includes all three windows; each pool owns one rank-local window.
	kda_window_stride = (uint64_t)state->resident_sequence_capacity *
		(SPARK_GLM5_NEXT_MODEL_KDA_CONV_WINDOW_BYTES_PER_LAYER / (3u * state->tp_degree));
	state->kv_layer_stride_bytes = (uint64_t)state->page_count * main_page_bytes;
	state->index_layer_stride_bytes = (uint64_t)state->page_count * index_page_bytes;
	state->kda_state_layer_stride_bytes = (uint64_t)state->resident_sequence_capacity *
		(SPARK_GLM5_NEXT_MODEL_KDA_STATE_BYTES_PER_LAYER / state->tp_degree);
	state->kda_window_layer_stride_bytes = kda_window_stride;
	if ( status != SPARK_STATUS_OK || state->kv_layer_stride_bytes == 0u )
		return(status == SPARK_STATUS_OK ? SPARK_STATUS_CAPACITY_EXCEEDED : status);
	main_total = state->kv_layer_stride_bytes * (uint64_t)state->kv_layer_count;
	status = SparkStageModuleDeviceAllocate(&state->ledger,main_total,(void **)&state->kv_cache);
	if ( status == SPARK_STATUS_OK && state->index_layer_count != 0u )
	{
		if ( state->index_layer_stride_bytes > UINT64_MAX / state->index_layer_count )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		index_total = state->index_layer_stride_bytes * state->index_layer_count;
		status = SparkStageModuleDeviceAllocate(&state->ledger,index_total,(void **)&state->index_cache);
	}
	kda_total = state->kda_state_layer_stride_bytes * (uint64_t)state->kda_layer_count;
	if ( status == SPARK_STATUS_OK && state->kda_layer_count != 0u )
		status = SparkStageModuleDeviceAllocate(&state->ledger,kda_total,(void **)&state->kda_state_pools);
	window_total = kda_window_stride * 3u * (uint64_t)state->kda_layer_count;
	if ( status == SPARK_STATUS_OK && state->kda_layer_count != 0u )
	{
		status = SparkStageModuleDeviceAllocate(&state->ledger,window_total,(void **)&state->kda_window_pools);
		if ( status == SPARK_STATUS_OK )
		{
			state->kda_q_window_pool = state->kda_window_pools;
			state->kda_k_window_pool = state->kda_q_window_pool + kda_window_stride * (uint64_t)state->kda_layer_count;
			state->kda_v_window_pool = state->kda_k_window_pool + kda_window_stride * (uint64_t)state->kda_layer_count;
		}
	}
	if ( status == SPARK_STATUS_OK && state->kda_layer_count != 0u )
	{
		uint32_t sequence;
		cudaError_t error;
		state->kda_state_index_host = (uint32_t *)malloc((size_t)state->resident_sequence_capacity * sizeof(uint32_t));
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)state->resident_sequence_capacity * sizeof(uint32_t),(void **)&state->kda_state_index_device);
		if ( status == SPARK_STATUS_OK && state->kda_state_index_host != 0 )
		{
			for (sequence=0u; sequence<state->resident_sequence_capacity; sequence++)
				state->kda_state_index_host[sequence] = sequence;
			error = cudaMemcpy(state->kda_state_index_device,state->kda_state_index_host,(size_t)state->resident_sequence_capacity * sizeof(uint32_t),cudaMemcpyHostToDevice);
			if ( error != cudaSuccess )
				status = SPARK_STATUS_INTERNAL_ERROR;
		}
		else if ( state->kda_state_index_host == 0 )
		{
			status = SPARK_STATUS_CAPACITY_EXCEEDED;
		}
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextKvInitialize(state);
	return(status);
}

static SparkStatus SparkGlm5NextAdmissionPredicate(
	void *context,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	SparkGlm5NextModuleState *state = (SparkGlm5NextModuleState *)context;
	SparkStatus status;
	uint32_t lane,slot;
	if ( state == 0 || request == 0 || decision == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( pthread_mutex_lock(&state->kv_mutex) != 0 )
		return(SPARK_STATUS_INTERNAL_ERROR);
	status = request->control_generation < state->control_generation ? SPARK_STATUS_VALIDATION_FAILED : SparkKvLaneTransactionsAdmit(&state->kv_transactions,request);
	if ( status == SPARK_STATUS_OK )
		state->control_generation = request->control_generation;
	if ( status == SPARK_STATUS_OK && (request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE) != 0u )
		for (lane=0u; lane<request->cache_lane_count; lane++)
		{
			slot = request->cache_lanes[lane].resident_sequence_slot;
			atomic_store_explicit(&state->lane_bound[slot],0u,memory_order_release);
			if ( state->mtp_lane_armed != 0 )
				state->mtp_lane_armed[slot] = 0u;
		}
	(void)pthread_mutex_unlock(&state->kv_mutex);
	if ( status != SPARK_STATUS_OK )
		return(status);
	decision->accepted = 1u;
	decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
	decision->driver_dispatch_slot = (uint32_t)(request->request_id % state->pipeline_slot_count);
	decision->driver_dispatch_generation = request->control_generation;
	decision->driver_dispatch_cookie0 = request->transaction_id;
	decision->driver_dispatch_cookie1 = request->submission_id;
	return(SPARK_STATUS_OK);
}

static uint32_t SparkGlm5NextRoundMajorWaveRows(
	const SparkGlm5NextModuleState *state,
	const SparkGlm5NextResidentDecodeStageBatchView *batch,
	uint32_t first_row)
{
	SparkStageModuleClaimedLaneContext lanes;
	if ( state == 0 || batch == 0 || batch->active_sequence_count == 0u )
		return(0u);
	lanes.index_states = state->lane_states;
	lanes.index_capacity = state->resident_sequence_capacity;
	return(SparkRowLayoutRoundMajorWaveRowCount(first_row,batch->row_count,batch->row_resident_slots,SparkStageModuleClaimedLaneOrdinal,&lanes));
}

static SparkStatus SparkGlm5NextValidateRoundMajor(
	const SparkGlm5NextModuleState *state,
	const SparkGlm5NextResidentDecodeStageBatchView *batch)
{
	uint32_t ordinals[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t counts[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t last_rows[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	SparkRowLayoutDirectLaneContext lanes;
	SparkStatus status;
	if ( state == 0 || batch == 0 || batch->row_count < batch->active_sequence_count || state->resident_sequence_capacity > SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkRowLayoutDirectLaneMapInitialize(&lanes,ordinals,state->resident_sequence_capacity,batch->row_resident_slots,batch->active_sequence_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkRowLayoutValidateRoundMajor(batch->row_count,batch->active_sequence_count,batch->row_resident_slots,SparkRowLayoutDirectLaneOrdinal,&lanes,counts,last_rows));
}

static uint32_t SparkGlm5NextPrefixRestorePending(const SparkKvLaneTransaction *owner)
{
	return(owner != 0 && (owner->mutation_flags & SPARK_KV_PAGE_CACHE_MUTATION_BOUND_SEQUENCE) != 0u && (owner->lane.flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX) != 0u && owner->lane.sequence_position != 0u);
}

static SparkStatus SparkGlm5NextLoadSequenceContinuity(const SparkGlm5NextModuleState *state,const SparkGlm5NextResidentDecodeStageBatchView *batch,uint8_t *bound,uint64_t *sequence_ids,uint64_t *next_positions)
{
	const SparkKvLaneTransaction *owner;
	uint32_t lane,slot;
	if ( state->kv_lane_transactions == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (lane=0u; lane<batch->active_sequence_count; lane++)
	{
		slot = batch->row_resident_slots[lane];
		if ( slot >= state->resident_sequence_capacity )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		bound[lane] = atomic_load_explicit(&state->lane_bound[slot],memory_order_acquire);
		sequence_ids[lane] = atomic_load_explicit(&state->lane_sequence_ids[slot],memory_order_acquire);
		next_positions[lane] = atomic_load_explicit(&state->lane_next_positions[slot],memory_order_acquire);
		owner = &state->kv_lane_transactions[slot];
		if ( SparkGlm5NextPrefixRestorePending(owner) != 0u )
		{
			if ( owner->phase != SPARK_KV_LANE_TRANSACTION_COMMITTED || owner->lane.sequence_id != batch->row_sequence_ids[lane] || owner->lane.sequence_position != batch->row_positions[lane] )
				return(SPARK_STATUS_VALIDATION_FAILED);
			bound[lane] = 1u;
			sequence_ids[lane] = owner->lane.sequence_id;
			next_positions[lane] = owner->lane.sequence_position;
		}
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextValidateSequenceContinuity(
	const SparkGlm5NextModuleState *state,
	const SparkGlm5NextResidentDecodeStageBatchView *batch,
	uint8_t *bound,
	uint64_t *sequence_ids,
	uint64_t *next_positions)
{
	uint8_t touched[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT] = {0u};
	uint64_t position,sequence;
	uint32_t lane,row,slot;
	SparkStatus status;
	status = SparkGlm5NextLoadSequenceContinuity(state,batch,bound,sequence_ids,next_positions);
	if ( status != SPARK_STATUS_OK )
		return(status);
	for (row=0u; row<batch->row_count; row++)
	{
		slot = batch->row_resident_slots[row];
		status = SparkStageModuleIndexClaimOrdinal(state->lane_states,state->resident_sequence_capacity,slot,&lane);
		position = batch->row_positions[row];
		sequence = batch->row_sequence_ids[row];
		if ( status != SPARK_STATUS_OK || lane >= batch->active_sequence_count || batch->row_resident_slots[lane] != slot || position >= state->max_sequence_positions )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		if ( position == 0u )
		{
			if ( touched[lane] != 0u )
				return(SPARK_STATUS_SCHEMA_ERROR);
			bound[lane] = 1u;
			sequence_ids[lane] = sequence;
			next_positions[lane] = 1u;
		}
		else
		{
			if ( bound[lane] == 0u || sequence_ids[lane] != sequence || next_positions[lane] != position )
				return(SPARK_STATUS_SCHEMA_ERROR);
			next_positions[lane] = position + 1u;
		}
		touched[lane] = 1u;
	}
	return(SPARK_STATUS_OK);
}

typedef struct SparkGlm5NextClaimedContinuityContext
{
	SparkGlm5NextModuleState *state;
	const SparkGlm5NextResidentDecodeStageBatchView *batch;
	uint8_t *bound;
	uint64_t *sequence_ids;
	uint64_t *next_positions;
} SparkGlm5NextClaimedContinuityContext;

static SparkStatus SparkGlm5NextPrepareClaimedContinuity(void *prepare_context)
{
	SparkGlm5NextClaimedContinuityContext *context;
	SparkStatus status;
	context = (SparkGlm5NextClaimedContinuityContext *)prepare_context;
	if ( pthread_mutex_lock(&context->state->kv_mutex) != 0 )
		return(SPARK_STATUS_INTERNAL_ERROR);
	status = SparkGlm5NextValidateSequenceContinuity(context->state,context->batch,context->bound,context->sequence_ids,context->next_positions);
	(void)pthread_mutex_unlock(&context->state->kv_mutex);
	return(status);
}

static SparkStatus SparkGlm5NextValidateFrameBuffers(
	const SparkGlm5NextModuleState *state,
	const SparkModelDriverFrame *frame,
	uint32_t row_count)
{
	const SparkModelDriverBuffer *buffer;
	if ( state->owns_final_head == 0u )
		return(frame->buffer_count == 0u ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
	if ( frame->buffer_count != 1u || frame->buffers == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	buffer = &frame->buffers[0];
	if ( buffer->flags != SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE || buffer->address == 0 || buffer->bytes < (uint64_t)row_count * sizeof(uint32_t) )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( state->mtp_enabled != 0u && row_count == 1u &&
		(frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) == 0u &&
		buffer->bytes < (uint64_t)(SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH + 1u) * sizeof(uint32_t) )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextValidateFrame(
	const SparkGlm5NextModuleState *state,
	const SparkModelDriverFrame *frame,
	const SparkGlm5NextResidentDecodeStageFrameContext **context_out)
{
	const SparkGlm5NextResidentDecodeStageFrameContext *context;
	const SparkGlm5NextResidentDecodeStageBatchView *batch;
	uint32_t expected_flags,prefill;
	uint64_t boundary_bytes,sideband_bytes;
	SparkStatus status;
	if ( state == 0 || frame == 0 || context_out == 0 || frame->user_context == 0 || frame->execution_stream != state->execution_stream || frame->completion_function == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	context = (const SparkGlm5NextResidentDecodeStageFrameContext *)frame->user_context;
	if ( context->abi_version != SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION || context->descriptor_bytes != sizeof(*context) || context->reserved0 != 0u || (context->flags & ~SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_KNOWN_FLAGS) != 0u || context->batch == 0 )
		return(SPARK_STATUS_ABI_MISMATCH);
	batch = context->batch;
	if ( batch->abi_version != SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_BATCH_VIEW_ABI_VERSION || batch->descriptor_bytes != sizeof(*batch) || batch->row_count == 0u || batch->row_count > SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT || batch->active_sequence_count == 0u || batch->active_sequence_count > state->resident_sequence_capacity || batch->row_resident_slots == 0 || batch->row_positions == 0 || batch->row_sequence_ids == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	prefill = (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u ? 1u : 0u;
	if ( prefill == 0u && batch->row_count != batch->active_sequence_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( frame->active_slot_count != batch->active_sequence_count || frame->new_token_count != batch->row_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->owns_embedding != 0u && batch->token_ids == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	expected_flags = prefill != 0u ? SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_PREFILL : 0u;
	expected_flags |= state->owns_embedding == 0u ? SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_INPUT : 0u;
	expected_flags |= state->owns_final_head == 0u ? SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_OUTPUT : 0u;
	expected_flags |= SparkGlm5NextResidentDecodeStageRequiresSidebandInput(state->stage_index) != 0u ? SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_SIDEBAND_INPUT : 0u;
	expected_flags |= SparkGlm5NextResidentDecodeStageRequiresSidebandOutput(state->stage_index) != 0u ? SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_FRAME_FLAG_SIDEBAND_OUTPUT : 0u;
	if ( context->flags != expected_flags )
		return(SPARK_STATUS_SCHEMA_ERROR);
	boundary_bytes = (uint64_t)batch->row_count * SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_COUNT * SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_BYTES;
	sideband_bytes = (uint64_t)batch->row_count * SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_DSA_SIDEBAND_BYTES_PER_ROW;
	if ( (state->owns_embedding == 0u && (context->hidden_input_bf16 == 0 || context->hidden_input_bytes < boundary_bytes)) || (state->owns_embedding != 0u && (context->hidden_input_bf16 != 0 || context->hidden_input_bytes != 0u)) || (state->owns_final_head == 0u && (context->hidden_output_bf16 == 0 || context->hidden_output_bytes < boundary_bytes)) || (state->owns_final_head != 0u && (context->hidden_output_bf16 != 0 || context->hidden_output_bytes != 0u)) )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( (SparkGlm5NextResidentDecodeStageRequiresSidebandInput(state->stage_index) != 0u && (context->sideband_input == 0 || context->sideband_input_bytes < sideband_bytes)) || (SparkGlm5NextResidentDecodeStageRequiresSidebandInput(state->stage_index) == 0u && (context->sideband_input != 0 || context->sideband_input_bytes != 0u)) || (SparkGlm5NextResidentDecodeStageRequiresSidebandOutput(state->stage_index) != 0u && (context->sideband_output == 0 || context->sideband_output_bytes < sideband_bytes)) || (SparkGlm5NextResidentDecodeStageRequiresSidebandOutput(state->stage_index) == 0u && (context->sideband_output != 0 || context->sideband_output_bytes != 0u)) )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkGlm5NextValidateRoundMajor(state,batch);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextValidateFrameBuffers(state,frame,batch->row_count);
	*context_out = status == SPARK_STATUS_OK ? context : 0;
	return(status);
}

#define SPARK_GLM5_NEXT_TP_COLLECTIVE_CREDITS_PER_SLOT 2u
#define SPARK_GLM5_NEXT_TP_CHAIN_OPERATIONS ((2u * SPARK_GLM5_NEXT_MODEL_LAYER_COUNT + 16u) * SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT)
#define SPARK_GLM5_NEXT_TP_COLLECTIVE_HC_PORT_STRIDE 512u
#define SPARK_GLM5_NEXT_TP_COLLECTIVE_D2A_MAX_PAYLOAD_BYTES 65536u

typedef enum SparkGlm5NextChainStage
{
	SPARK_GLM5_NEXT_CHAIN_STAGE_BEGIN = 0,
	SPARK_GLM5_NEXT_CHAIN_STAGE_ATTENTION,
	SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_ATTENTION,
	SPARK_GLM5_NEXT_CHAIN_STAGE_MLP,
	SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_MLP,
	SPARK_GLM5_NEXT_CHAIN_STAGE_HEAD,
	SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_HEAD,
	SPARK_GLM5_NEXT_CHAIN_STAGE_FINISH
} SparkGlm5NextChainStage;

typedef struct SparkGlm5NextTpChain
{
	SparkGlm5NextModuleState *state;
	SparkGlm5NextExecutionSlot *slot;
	uint32_t slot_index;
	SparkModelDriverFrame *frame;
	const SparkGlm5NextResidentDecodeStageFrameContext *context;
	const SparkGlm5NextResidentDecodeStageBatchView *batch;
	SparkGlm5NextCudaWave wave;
	uint32_t first_row;
	uint32_t wave_rows;
	uint32_t next_wave_row;
	uint32_t stage;
	uint32_t next_layer;
	uint32_t active;
	uint32_t spec_verify;
	uint32_t tp_op_index;
	uint32_t tp_hc_op_index;
	uint64_t expert_lease;
	uint32_t expert_lease_begun;
	uint32_t expert_lease_recorded;
	SparkStatus retained_status;
} SparkGlm5NextTpChain;

static void SparkGlm5NextTpChainAdvance(void *chain_context,SparkStatus status);
static void CUDART_CB SparkGlm5NextCompleteAsync(void *context);
static void CUDART_CB SparkGlm5NextMtpResolveHost(void *context);
static SparkStatus SparkGlm5NextEnqueueAsyncCompletion(
	SparkGlm5NextModuleState *state,
	SparkGlm5NextExecutionSlot *slot,
	uint32_t slot_index);

static void SparkGlm5NextBuildWave(SparkGlm5NextTpChain *chain)
{
	SparkGlm5NextModuleState *state;
	SparkGlm5NextExecutionSlot *slot;
	const SparkGlm5NextResidentDecodeStageFrameContext *context;
	SparkGlm5NextCudaWave *wave;
	uint32_t row,maximum_context;
	state = chain->state;
	slot = chain->slot;
	context = chain->context;
	wave = &chain->wave;
	maximum_context = 0u;
	for (row=0u; row<chain->wave_rows; row++)
		if ( slot->host_positions[chain->first_row + row] + 1u > maximum_context )
			maximum_context = slot->host_positions[chain->first_row + row] + 1u;
	memset(wave,0,sizeof(*wave));
	wave->stage_index = state->stage_index;
	wave->first_layer_index = state->first_layer_index;
	wave->layer_count = state->layer_count;
	wave->tp_degree = state->tp_degree;
	wave->tp_rank = state->tp_rank;
	wave->row_count = chain->wave_rows;
	wave->commit = chain->spec_verify != 0u ? 0u : 1u;
	wave->mtp_verify = chain->spec_verify;
	wave->mtp_draft_depth = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH;
	wave->mtp_layer_weights = state->pack_has_mtp != 0u ? &state->mtp_layer : 0;
	wave->mtp_eh_proj_bf16 = state->mtp_eh_proj_bf16;
	wave->mtp_enorm_bf16 = state->mtp_enorm_bf16;
	wave->mtp_hnorm_bf16 = state->mtp_hnorm_bf16;
	wave->mtp_shared_norm_bf16 = state->mtp_shared_norm_bf16;
	wave->kda_replay_layer_bytes = state->kda_replay_layer_bytes;
	wave->maximum_context = maximum_context;
	wave->resident_sequence_capacity = state->resident_sequence_capacity;
	wave->max_sequence_positions = state->max_sequence_positions;
	wave->execution_row_capacity = state->execution_row_capacity;
	wave->pages_per_sequence = state->pages_per_sequence;
	wave->owns_embedding = state->owns_embedding;
	wave->owns_final_head = state->owns_final_head;
	wave->sideband_input = SparkGlm5NextResidentDecodeStageRequiresSidebandInput(state->stage_index);
	wave->sideband_output = SparkGlm5NextResidentDecodeStageRequiresSidebandOutput(state->stage_index);
	wave->boundary_row_offset = chain->first_row;
	wave->sideband_row_offset = chain->first_row;
	wave->host_token_ids = state->owns_embedding != 0u ? slot->host_token_ids + chain->first_row : 0;
	wave->host_resident_slots = slot->host_resident_slots + chain->first_row;
	wave->host_positions = slot->host_positions + chain->first_row;
	wave->hidden_input_bf16 = context->hidden_input_bf16;
	wave->hidden_output_bf16 = context->hidden_output_bf16;
	wave->sideband_input_u32 = context->sideband_input;
	wave->sideband_output_u32 = context->sideband_output;
	wave->host_output_token_ids = state->owns_final_head != 0u ? slot->host_output_token_ids + chain->first_row : 0;
	wave->embedding_bf16 = state->embedding_bf16;
	wave->final_norm_bf16 = state->final_norm_bf16;
	wave->lm_head_bf16 = state->lm_head_bf16;
	wave->head_certified_fp8_payload = state->head_certified_fp8_payload;
	wave->head_certified_fp8_scale_f32 = state->head_certified_fp8_scale_f32;
	wave->head_certified_fp8_norm_f32 = state->head_certified_fp8_norm_f32;
	wave->layers = state->layers;
	wave->lazy_experts = state->lazy_pack != 0 ? 1u : 0u;
	wave->slot = slot;
	wave->kv_cache = state->kv_cache;
	wave->kv_layer_stride_bytes = state->kv_layer_stride_bytes;
	wave->index_cache = state->index_cache;
	wave->index_layer_stride_bytes = state->index_layer_stride_bytes;
	wave->index_ordinal_by_local_layer = state->index_ordinal_by_local_layer;
	wave->kda_ordinal_by_local_layer = state->kda_ordinal_by_local_layer;
	wave->kda_state_pools = state->kda_state_pools;
	wave->kda_state_layer_stride_bytes = state->kda_state_layer_stride_bytes;
	wave->kda_q_window_pool = state->kda_q_window_pool;
	wave->kda_k_window_pool = state->kda_k_window_pool;
	wave->kda_v_window_pool = state->kda_v_window_pool;
	wave->kda_window_layer_stride_bytes = state->kda_window_layer_stride_bytes;
	wave->kda_state_index = state->kda_state_index_device;
	wave->kda_layer_count = state->kda_layer_count;
	wave->page_table = state->page_table;
	wave->multiprocessor_count = state->multiprocessor_count;
	wave->decode_split_context_threshold = state->decode_split_context_threshold;
	wave->attention_split_partials_f32 = slot->attention_split_partials_f32;
	wave->attention_split_partial_blocks = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_BLOCKS(
		state->execution_row_capacity,SPARK_GLM5_NEXT_MODEL_HEAD_COUNT / state->tp_degree);
	{
		uint32_t run,row_of_run;
		slot->host_run_begin[0] = 0u;
		run = 0u;
		for (row=1u; row<chain->wave_rows; row++)
		{
			if ( slot->host_resident_slots[chain->first_row + row] !=
			     slot->host_resident_slots[chain->first_row + row - 1u] )
			{
				run++;
				slot->host_run_begin[run] = row;
			}
		}
		run++;
		slot->host_run_begin[run] = chain->wave_rows;
		for (row_of_run=0u; row_of_run<run; row_of_run++)
			slot->host_run_state_index[row_of_run] =
				slot->host_resident_slots[chain->first_row + slot->host_run_begin[row_of_run]];
		wave->run_count = run;
		wave->sequence_row_begin = slot->run_begin;
		wave->run_state_index = slot->run_state_index;
		wave->host_sequence_row_begin = slot->host_run_begin;
		wave->host_run_state_index = slot->host_run_state_index;
	}
}

static SparkStatus SparkGlm5NextModuleCombineBf16(
	void *combine_context,
	void *destination_device,
	const void *source_device,
	uint32_t active_sequence_count,
	uint32_t hidden_dimension,
	void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkGlm5NextLaunchAccumAdd((cudaStream_t)cuda_stream,destination_device,source_device,active_sequence_count,hidden_dimension);
	return(SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"tp_all_reduce_sum"));
}

static SparkStatus SparkGlm5NextModuleCombineDirectBf16(
	void *combine_context,
	void *destination_device,
	const void *const rank_devices[
		SPARK_TP_DEVICE_COLLECTIVE_DIRECT_ALL_TO_ALL_RANK_COUNT],
	uint32_t tp_rank,
	uint32_t active_sequence_count,
	uint32_t hidden_dimension,
	void *cuda_stream)
{
	uint32_t index;
	cudaError_t error;
	(void)combine_context;
	for (index=0u;
		index<SPARK_TP_DEVICE_COLLECTIVE_DIRECT_ALL_TO_ALL_RANK_COUNT;
		index++)
	{
		if (rank_devices[index] == 0 || index == tp_rank)
			continue;
		error = SparkGlm5NextLaunchAccumAdd((cudaStream_t)cuda_stream,
			destination_device,rank_devices[index],
			active_sequence_count,hidden_dimension);
		if (error != cudaSuccess)
			return(SparkStageModuleCudaStatus(
				SPARK_GLM5_NEXT_MODULE_TAG,error,"tp_d2d_all_reduce_sum"));
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextModuleCombineU64Max(
	void *combine_context,
	uint64_t *destination_device,
	const uint64_t *source_device,
	uint32_t element_count,
	void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkGlm5NextLaunchAccumU64Max((cudaStream_t)cuda_stream,destination_device,source_device,element_count);
	return(SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"tp_all_reduce_max_u64"));
}

static SparkStatus SparkGlm5NextModuleInitializeTpCollective(
	SparkGlm5NextModuleState *state,
	const SparkGlm5NextResidentDecodeStageNodeContext *context)
{
	SparkTpDeviceCollectiveConfig configuration,configuration_hc;
	uint32_t probe_connect_timeout_milli,probe_operation_timeout_milli;
	uint64_t credit_bytes,offset,total_bytes;
	uint32_t credit,hidden,memory_mode,route,route_count,hc_route_count;
	uint32_t d2a_route_count,tree_route_count;
	void *mapped_receive,*mapped_send;
	cudaError_t error;
	SparkStatus status;
	if ( state == 0 || context == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->tp_degree == 1u || state->tp_collective_disabled != 0u )
		return(SPARK_STATUS_OK);
	probe_connect_timeout_milli = context->tp_connect_timeout_milli;
	if ( SparkGlm5NextProbeEnabled() )
	{
		if ( probe_connect_timeout_milli >
			UINT32_MAX / SPARK_GLM5_NEXT_PROBE_CONNECT_TIMEOUT_SCALE )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		probe_connect_timeout_milli *= SPARK_GLM5_NEXT_PROBE_CONNECT_TIMEOUT_SCALE;
	}
	probe_operation_timeout_milli = context->tp_operation_timeout_milli;
	if ( SparkGlm5NextProbeEnabled() )
	{
		if ( probe_operation_timeout_milli >
			UINT32_MAX / SPARK_GLM5_NEXT_PROBE_OPERATION_TIMEOUT_SCALE )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		probe_operation_timeout_milli *= SPARK_GLM5_NEXT_PROBE_OPERATION_TIMEOUT_SCALE;
	}
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	configuration.backend_kind = context->tp_collective_backend_kind;
	configuration.tp_degree = state->tp_degree;
	configuration.tp_rank = state->tp_rank;
	configuration.operation_kind = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	configuration.credit_count = state->pipeline_slot_count * SPARK_GLM5_NEXT_TP_COLLECTIVE_CREDITS_PER_SLOT;
	configuration.local_hidden_dimension = SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION;
	configuration.max_active_sequence_count = state->execution_row_capacity;
	configuration.connect_timeout_milli = probe_connect_timeout_milli;
	configuration.operation_timeout_milli = probe_operation_timeout_milli;
	configuration.control_port_base = context->tp_collective_control_port_base;
	configuration.collective_identifier = context->tp_collective_identifier;
	configuration.backend_module_path = context->tp_collective_backend_module_path;
	configuration.registration_cuda_stream = state->execution_stream;
	status = SparkTpDeviceCollectiveApplyTopology(&context->tp_collective_topology,&configuration);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( configuration.backend_kind == SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT )
	{
		configuration.algorithm_mask |= SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_DIRECT_ALL_TO_ALL;
		configuration.direct_all_to_all_max_payload_bytes =
			SPARK_GLM5_NEXT_TP_COLLECTIVE_D2A_MAX_PAYLOAD_BYTES;
	}
	memset(&configuration_hc,0,sizeof(configuration_hc));
	configuration_hc.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	configuration_hc.backend_kind = context->tp_collective_backend_kind;
	configuration_hc.tp_degree = state->tp_degree;
	configuration_hc.tp_rank = state->tp_rank;
	configuration_hc.operation_kind = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	configuration_hc.credit_count = configuration.credit_count;
	configuration_hc.local_hidden_dimension =
		SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION * SPARK_GLM5_NEXT_MODEL_HC_MULT;
	configuration_hc.max_active_sequence_count = configuration.max_active_sequence_count;
	configuration_hc.connect_timeout_milli = probe_connect_timeout_milli;
	configuration_hc.operation_timeout_milli = probe_operation_timeout_milli;
	configuration_hc.control_port_base = context->tp_collective_control_port_base +
		SPARK_GLM5_NEXT_TP_COLLECTIVE_HC_PORT_STRIDE;
	configuration_hc.collective_identifier = context->tp_collective_identifier + 1u;
	configuration_hc.backend_module_path = context->tp_collective_backend_module_path;
	configuration_hc.registration_cuda_stream = state->execution_stream;
	status = SparkTpDeviceCollectiveApplyTopology(&context->tp_collective_topology,&configuration_hc);
	if ( status != SPARK_STATUS_OK )
		return(status);
	memcpy(configuration_hc.session_ports,context->tp_collective_session_ports_hc,
		sizeof(configuration_hc.session_ports));
	if ( configuration.backend_kind == SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT )
	{
		configuration.combine_bf16_function = SparkGlm5NextModuleCombineBf16;
		configuration.combine_u64_max_function = SparkGlm5NextModuleCombineU64Max;
		configuration.combine_tp4_bf16_function = SparkGlm5NextModuleCombineDirectBf16;
		configuration.combine_context = state;
		configuration_hc.combine_bf16_function = SparkGlm5NextModuleCombineBf16;
		configuration_hc.combine_tp4_bf16_function = SparkGlm5NextModuleCombineDirectBf16;
		configuration_hc.combine_context = state;
	}
	if ( configuration.connect_timeout_milli == 0u || configuration.operation_timeout_milli == 0u || configuration.control_port_base == 0u || configuration.collective_identifier == 0u || configuration.backend_module_path == 0 || configuration.local_host == 0 || configuration.backend_module_path[0] == '\0' || configuration.local_host[0] == '\0' )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration.backend_kind != SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT && configuration.backend_kind != SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkTpDeviceCollectiveProbeMemoryMode(configuration.backend_kind,configuration.backend_module_path,&memory_mode);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkTpDeviceCollectiveCreditBindingRouteCount(&configuration,&route_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkTpDeviceCollectiveCreditBindingRouteCount(&configuration_hc,&hc_route_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	d2a_route_count = (configuration.algorithm_mask & SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_DIRECT_ALL_TO_ALL) != 0u && configuration.direct_all_to_all_max_payload_bytes != 0u ? SPARK_TP_DEVICE_COLLECTIVE_DIRECT_ALL_TO_ALL_MAX_PEERS : 0u;
	tree_route_count = route_count - d2a_route_count;
	total_bytes = 0u;
	for (route=0u; route<route_count; route++)
	{
		hidden = configuration.local_hidden_dimension;
		credit_bytes = (uint64_t)configuration.max_active_sequence_count * hidden * SPARK_GLM5_NEXT_MODEL_BF16_ELEMENT_BYTES + SPARK_TP_DEVICE_COLLECTIVE_NONCE_BYTES;
		if ( credit_bytes == 0u || total_bytes > UINT64_MAX - credit_bytes * configuration.credit_count )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		total_bytes += credit_bytes * configuration.credit_count;
	}
	status = SPARK_STATUS_OK;
	if ( total_bytes != 0u )
		status = SparkStageModuleDeviceAllocate(&state->ledger,total_bytes,&state->tp_credit_send_bf16);
	if ( status == SPARK_STATUS_OK && total_bytes != 0u )
		status = SparkStageModuleDeviceAllocate(&state->ledger,total_bytes,&state->tp_credit_receive_bf16);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( total_bytes != 0u && memory_mode == SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST )
	{
		mapped_receive = 0;
		mapped_send = 0;
		error = cudaHostAlloc(&state->tp_host_credit_send_bf16,total_bytes,cudaHostAllocPortable | cudaHostAllocMapped);
		if ( error == cudaSuccess )
			error = cudaHostAlloc(&state->tp_host_credit_receive_bf16,total_bytes,cudaHostAllocPortable | cudaHostAllocMapped);
		if ( error == cudaSuccess )
			error = cudaHostGetDevicePointer(&mapped_send,state->tp_host_credit_send_bf16,0u);
		if ( error == cudaSuccess )
			error = cudaHostGetDevicePointer(&mapped_receive,state->tp_host_credit_receive_bf16,0u);
		if ( error != cudaSuccess )
		{
			if ( state->tp_host_credit_send_bf16 != 0 )
				(void)cudaFreeHost(state->tp_host_credit_send_bf16);
			if ( state->tp_host_credit_receive_bf16 != 0 )
				(void)cudaFreeHost(state->tp_host_credit_receive_bf16);
			state->tp_host_credit_send_bf16 = 0;
			state->tp_host_credit_receive_bf16 = 0;
			return(SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"tp_credit_alloc"));
		}
		state->tp_credit_send_bf16 = mapped_send;
		state->tp_credit_receive_bf16 = mapped_receive;
	}
	offset = 0u;
	state->tp_credit_binding_count = 0u;
	for (route=0u; route<route_count; route++)
	{
		hidden = configuration.local_hidden_dimension;
		credit_bytes = (uint64_t)configuration.max_active_sequence_count * hidden * SPARK_GLM5_NEXT_MODEL_BF16_ELEMENT_BYTES + SPARK_TP_DEVICE_COLLECTIVE_NONCE_BYTES;
		for (credit=0u; credit<configuration.credit_count; credit++)
		{
			SparkTpDeviceCollectiveCreditBinding *binding;
			if ( state->tp_credit_binding_count >= SPARK_TP_DEVICE_COLLECTIVE_MAX_BINDING_COUNT )
				return(SPARK_STATUS_CAPACITY_EXCEEDED);
			binding = &state->tp_credit_bindings[state->tp_credit_binding_count++];
			binding->step_index = route < tree_route_count ? route : route - tree_route_count;
			binding->credit_index = credit;
			binding->send_device = (uint8_t *)state->tp_credit_send_bf16 + offset;
			binding->receive_device = (uint8_t *)state->tp_credit_receive_bf16 + offset;
			binding->send_transport = memory_mode == SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST ? (uint8_t *)state->tp_host_credit_send_bf16 + offset : binding->send_device;
			binding->receive_transport = memory_mode == SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST ? (uint8_t *)state->tp_host_credit_receive_bf16 + offset : binding->receive_device;
			binding->flags = (memory_mode == SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST ? (SPARK_TP_DEVICE_COLLECTIVE_BINDING_SEND_MAPPED_ALIAS | SPARK_TP_DEVICE_COLLECTIVE_BINDING_RECEIVE_MAPPED_ALIAS) : 0u) | (route < tree_route_count ? 0u : SPARK_TP_DEVICE_COLLECTIVE_BINDING_DIRECT_ALL_TO_ALL);
			binding->reserved0 = 0u;
			offset += credit_bytes;
		}
	}
	if ( state->tp_credit_binding_count != 0u )
	{
		configuration.credit_bindings = state->tp_credit_bindings;
		configuration.credit_binding_count = state->tp_credit_binding_count;
	}
	status = SparkTpDeviceCollectiveCreate(&configuration,&state->tp_device_collective);
	if ( status != SPARK_STATUS_OK )
	{
		if ( state->tp_host_credit_send_bf16 != 0 )
			(void)cudaFreeHost(state->tp_host_credit_send_bf16);
		if ( state->tp_host_credit_receive_bf16 != 0 )
			(void)cudaFreeHost(state->tp_host_credit_receive_bf16);
		state->tp_host_credit_send_bf16 = 0;
		state->tp_host_credit_receive_bf16 = 0;
		return(status);
	}
	state->tp_device_collective_initialized = 1u;
	{
		uint32_t hc_credit_count = configuration_hc.credit_count;
		uint64_t hc_credit_bytes = (uint64_t)configuration_hc.max_active_sequence_count *
			configuration_hc.local_hidden_dimension * SPARK_GLM5_NEXT_MODEL_BF16_ELEMENT_BYTES + SPARK_TP_DEVICE_COLLECTIVE_NONCE_BYTES;
		uint64_t hc_total;
		void *hc_mapped_send,*hc_mapped_receive;
		hc_total = 0u;
		for (route=0u; route<hc_route_count; route++)
		{
			if ( hc_credit_bytes == 0u || hc_total > UINT64_MAX - hc_credit_bytes * hc_credit_count )
				return(SPARK_STATUS_CAPACITY_EXCEEDED);
			hc_total += hc_credit_bytes * hc_credit_count;
		}
		status = SPARK_STATUS_OK;
		if ( hc_total != 0u )
			status = SparkStageModuleDeviceAllocate(&state->ledger,hc_total,&state->tp_hc_credit_send_bf16);
		if ( status == SPARK_STATUS_OK && hc_total != 0u )
			status = SparkStageModuleDeviceAllocate(&state->ledger,hc_total,&state->tp_hc_credit_receive_bf16);
		if ( status != SPARK_STATUS_OK )
			return(status);
		if ( hc_total != 0u && memory_mode == SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST )
		{
			hc_mapped_receive = 0;
			hc_mapped_send = 0;
			error = cudaHostAlloc(&state->tp_hc_host_credit_send_bf16,hc_total,cudaHostAllocPortable | cudaHostAllocMapped);
			if ( error == cudaSuccess )
				error = cudaHostAlloc(&state->tp_hc_host_credit_receive_bf16,hc_total,cudaHostAllocPortable | cudaHostAllocMapped);
			if ( error == cudaSuccess )
				error = cudaHostGetDevicePointer(&hc_mapped_send,state->tp_hc_host_credit_send_bf16,0u);
			if ( error == cudaSuccess )
				error = cudaHostGetDevicePointer(&hc_mapped_receive,state->tp_hc_host_credit_receive_bf16,0u);
			if ( error != cudaSuccess )
			{
				if ( state->tp_hc_host_credit_send_bf16 != 0 )
					(void)cudaFreeHost(state->tp_hc_host_credit_send_bf16);
				if ( state->tp_hc_host_credit_receive_bf16 != 0 )
					(void)cudaFreeHost(state->tp_hc_host_credit_receive_bf16);
				state->tp_hc_host_credit_send_bf16 = 0;
				state->tp_hc_host_credit_receive_bf16 = 0;
				return(SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"tp_hc_credit_alloc"));
			}
			state->tp_hc_credit_send_bf16 = hc_mapped_send;
			state->tp_hc_credit_receive_bf16 = hc_mapped_receive;
		}
		offset = 0u;
		state->tp_hc_credit_binding_count = 0u;
		for (route=0u; route<hc_route_count; route++)
		{
			for (credit=0u; credit<hc_credit_count; credit++)
			{
				SparkTpDeviceCollectiveCreditBinding *binding;
				if ( state->tp_hc_credit_binding_count >= SPARK_TP_DEVICE_COLLECTIVE_MAX_BINDING_COUNT )
					return(SPARK_STATUS_CAPACITY_EXCEEDED);
				binding = &state->tp_hc_credit_bindings[state->tp_hc_credit_binding_count++];
				binding->step_index = route;
				binding->credit_index = credit;
				binding->send_device = (uint8_t *)state->tp_hc_credit_send_bf16 + offset;
				binding->receive_device = (uint8_t *)state->tp_hc_credit_receive_bf16 + offset;
				binding->send_transport = memory_mode == SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST ? (uint8_t *)state->tp_hc_host_credit_send_bf16 + offset : binding->send_device;
				binding->receive_transport = memory_mode == SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST ? (uint8_t *)state->tp_hc_host_credit_receive_bf16 + offset : binding->receive_device;
				binding->flags = memory_mode == SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST ? (SPARK_TP_DEVICE_COLLECTIVE_BINDING_SEND_MAPPED_ALIAS | SPARK_TP_DEVICE_COLLECTIVE_BINDING_RECEIVE_MAPPED_ALIAS) : 0u;
				binding->reserved0 = 0u;
				offset += hc_credit_bytes;
			}
		}
		if ( state->tp_hc_credit_binding_count != 0u )
		{
			configuration_hc.credit_bindings = state->tp_hc_credit_bindings;
			configuration_hc.credit_binding_count = state->tp_hc_credit_binding_count;
		}
		status = SparkTpDeviceCollectiveCreate(&configuration_hc,&state->tp_device_collective_hc);
		if ( status != SPARK_STATUS_OK )
		{
			if ( state->tp_hc_host_credit_send_bf16 != 0 )
				(void)cudaFreeHost(state->tp_hc_host_credit_send_bf16);
			if ( state->tp_hc_host_credit_receive_bf16 != 0 )
				(void)cudaFreeHost(state->tp_hc_host_credit_receive_bf16);
			state->tp_hc_host_credit_send_bf16 = 0;
			state->tp_hc_host_credit_receive_bf16 = 0;
			return(status);
		}
		state->tp_device_collective_hc_initialized = 1u;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextModuleReduceHiddenWide(SparkGlm5NextTpChain *chain,
	void *device_bf16,uint32_t hc_wide);

static SparkStatus SparkGlm5NextModuleReduceHidden(SparkGlm5NextTpChain *chain,void *device_bf16)
{
	return(SparkGlm5NextModuleReduceHiddenWide(chain,device_bf16,1u));
}

static SparkStatus SparkGlm5NextModuleReduceAttentionOut(SparkGlm5NextTpChain *chain,void *device_bf16)
{
	return(SparkGlm5NextModuleReduceHiddenWide(chain,device_bf16,0u));
}

static void SparkGlm5NextModuleTpCompletion(
	void *context,
	const SparkTpDeviceCollectiveCompletion *completion)
{
	SparkGlm5NextTpChain *chain;
	chain = (SparkGlm5NextTpChain *)context;
	if ( chain == 0 || chain->active == 0u || completion == 0 )
		return;
	SparkGlm5NextTpChainAdvance(chain,completion->status);
}

static SparkStatus SparkGlm5NextChainOrdinal(SparkGlm5NextTpChain *chain,uint32_t hc_wide,uint32_t operation,uint64_t *ordinal)
{
	SparkGlm5NextModuleState *state;
	state = chain->state;
	if ( state->tp_device_collective.backend_kind == SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL )
	{
		*ordinal = atomic_fetch_add_explicit(hc_wide != 0u ? &state->nccl_next_ordinal_hc : &state->nccl_next_ordinal,1u,memory_order_relaxed);
		return(SPARK_STATUS_OK);
	}
	return(SparkTpChainOrdinal(chain->frame->request_id,state->pipeline_slot_count,SPARK_GLM5_NEXT_TP_COLLECTIVE_CREDITS_PER_SLOT,SPARK_GLM5_NEXT_TP_CHAIN_OPERATIONS,operation,ordinal));
}

static SparkStatus SparkGlm5NextModuleReduceHiddenWide(SparkGlm5NextTpChain *chain,
	void *device_bf16,uint32_t hc_wide)
{
	uint32_t *op_index;
	SparkStatus ordinal_status;
	SparkGlm5NextModuleState *state;
	SparkTpDeviceCollectiveSubmission submission;
	SparkTpDeviceCollective *collective;
	uint64_t ordinal;
	state = chain->state;
	if ( state->tp_degree == 1u || state->tp_collective_disabled != 0u )
	{
		SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
		return(SPARK_STATUS_OK);
	}
	if ( state->tp_device_collective_initialized == 0u )
		return(SPARK_STATUS_INTERNAL_ERROR);
	if ( hc_wide != 0u )
	{
		if ( state->tp_device_collective_hc_initialized == 0u )
			return(SPARK_STATUS_INTERNAL_ERROR);
		collective = &state->tp_device_collective_hc;
		op_index = &chain->tp_hc_op_index;
	}
	else
	{
		collective = &state->tp_device_collective;
		op_index = &chain->tp_op_index;
	}
	ordinal_status = SparkGlm5NextChainOrdinal(chain,hc_wide,*op_index,&ordinal);
	if ( ordinal_status != SPARK_STATUS_OK )
		return(ordinal_status);
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = chain->slot_index;
	submission.active_sequence_count = chain->wave_rows;
	submission.flags = SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission.ordinal = ordinal;
	submission.local_device = device_bf16;
	submission.full_device = device_bf16;
	submission.cuda_stream = chain->slot->stream;
	submission.completion_function = SparkGlm5NextModuleTpCompletion;
	submission.completion_context = chain;
	{
		SparkStatus submit_status;
		*op_index += 1u;
		submit_status = SparkTpDeviceCollectiveEnqueue(collective,&submission,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16);
		if ( submit_status != SPARK_STATUS_OK )
			*op_index -= 1u;
		if ( submit_status != SPARK_STATUS_OK )
			fprintf(stderr,"G5N-DBG reduce submit -> %d (rows %u slot %u dev %p stream %p maxact %u)\n",
				(int)submit_status,(unsigned)chain->wave_rows,(unsigned)chain->slot_index,
				device_bf16,chain->slot->stream,
				(unsigned)state->tp_device_collective.max_active_sequence_count);
		return(submit_status);
	}
}

static SparkStatus SparkGlm5NextModuleReduceHeadMax(SparkGlm5NextTpChain *chain)
{
	SparkGlm5NextModuleState *state;
	SparkTpDeviceCollectiveSubmission submission;
	uint64_t ordinal;
	SparkStatus status;
	state = chain->state;
	if ( state->tp_degree == 1u || state->tp_collective_disabled != 0u )
	{
		SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
		return(SPARK_STATUS_OK);
	}
	if ( state->tp_device_collective_initialized == 0u )
		return(SPARK_STATUS_INTERNAL_ERROR);
	status = SparkGlm5NextChainOrdinal(chain,0u,chain->tp_op_index,&ordinal);
	if ( status != SPARK_STATUS_OK )
		return(status);
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = chain->slot_index;
	submission.active_sequence_count = chain->wave_rows;
	submission.flags = SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission.ordinal = ordinal;
	submission.local_device = chain->slot->head_maxloc_u64;
	submission.full_device = chain->slot->head_maxloc_u64;
	submission.cuda_stream = chain->slot->stream;
	submission.completion_function = SparkGlm5NextModuleTpCompletion;
	submission.completion_context = chain;
	chain->tp_op_index += 1u;
	status = SparkTpDeviceCollectiveEnqueue(&state->tp_device_collective,&submission,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64);
	if ( status != SPARK_STATUS_OK )
		chain->tp_op_index -= 1u;
	return(status);
}

static void SparkGlm5NextBuildMtpDraftWave(
	const SparkGlm5NextModuleState *state,
	SparkGlm5NextExecutionSlot *slot,
	SparkGlm5NextCudaWave *wave)
{
	memset(wave,0,sizeof(*wave));
	wave->tp_degree = state->tp_degree;
	wave->tp_rank = state->tp_rank;
	wave->owns_final_head = state->owns_final_head;
	wave->multiprocessor_count = state->multiprocessor_count;
	wave->embedding_bf16 = state->embedding_bf16;
	wave->lm_head_bf16 = state->lm_head_bf16;
	wave->mtp_layer_weights = &state->mtp_layer;
	wave->mtp_eh_proj_bf16 = state->mtp_eh_proj_bf16;
	wave->mtp_enorm_bf16 = state->mtp_enorm_bf16;
	wave->mtp_hnorm_bf16 = state->mtp_hnorm_bf16;
	wave->mtp_shared_norm_bf16 = state->mtp_shared_norm_bf16;
	wave->mtp_draft_depth = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH;
	wave->slot = slot;
}

static SparkStatus SparkGlm5NextMtpDriveDraft(
	SparkGlm5NextModuleState *state,
	SparkModelDriverFrame *frame,
	const SparkGlm5NextResidentDecodeStageBatchView *batch,
	SparkGlm5NextExecutionSlot *slot,
	SparkGlm5NextTpChain *chain)
{
	SparkGlm5NextAsyncCompletion *async;
	SparkGlm5NextCudaWave draft_wave;
	uint32_t lane,step;
	uint64_t position;
	if ( state->mtp_enabled == 0u ||
		(frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u ||
		batch->row_count != 1u || batch->active_sequence_count != 1u ||
		batch->token_ids == 0 || state->owns_embedding == 0u || state->owns_final_head == 0u )
		return(SPARK_STATUS_OK);
	lane = batch->row_resident_slots[0];
	position = batch->row_positions[0];
	if ( lane >= state->resident_sequence_capacity ||
		state->mtp_lane_armed[lane] == 0u ||
		position + SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH + 1u > state->max_sequence_positions )
		return(SPARK_STATUS_OK);
	async = &state->completions[chain->slot_index];
	SparkGlm5NextBuildMtpDraftWave(state,slot,&draft_wave);
	if ( SparkGlm5NextLaunchCudaMtpDraft(&draft_wave,0,
		state->mtp_lane_hidden_bf16 + (uint64_t)lane * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,
		batch->token_ids[0],async->mtp_draft_tokens) != 0 )
		return(SPARK_STATUS_INTERNAL_ERROR);
	for ( step = 1u; step <= SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH; ++step )
	{
		slot->host_token_ids[step] = async->mtp_draft_tokens[step - 1u];
		slot->host_positions[step] = (uint32_t)(position + step);
		slot->host_resident_slots[step] = lane;
	}
	chain->wave_rows = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH + 1u;
	chain->next_wave_row = chain->wave_rows;
	chain->spec_verify = 1u;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextMtpStashHidden(
	SparkGlm5NextModuleState *state,
	const SparkGlm5NextTpChain *chain)
{
	cudaError_t error;
	uint32_t row,lane;
	for ( row = 0u; row < chain->wave_rows; ++row )
	{
		lane = chain->slot->host_resident_slots[chain->first_row + row];
		if ( row + 1u < chain->wave_rows &&
			chain->slot->host_resident_slots[chain->first_row + row + 1u] == lane )
			continue;
		error = cudaMemcpyAsync(
			state->mtp_lane_hidden_bf16 + (uint64_t)lane * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,
			chain->slot->hc_mean_bf16 + (uint64_t)row * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,
			(uint64_t)SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t),
			cudaMemcpyDeviceToDevice,(cudaStream_t)chain->slot->stream);
		if ( error != cudaSuccess )
			return(SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"mtp_stash"));
		state->mtp_lane_armed[lane] = 1u;
	}
	return(SPARK_STATUS_OK);
}

static void CUDART_CB SparkGlm5NextMtpResolveHost(void *context)
{
	SparkGlm5NextTpChain *chain;
	SparkGlm5NextModuleState *state;
	SparkGlm5NextExecutionSlot *slot;
	SparkGlm5NextAsyncCompletion *async;
	SparkSpeculationPolicyVerifyResult result;
	SparkStatus status;
	cudaError_t error;
	uint32_t lane;
	int32_t launch;
	chain = (SparkGlm5NextTpChain *)context;
	if ( chain == 0 || chain->active == 0u )
		return;
	state = chain->state;
	slot = chain->slot;
	async = &state->completions[chain->slot_index];
	lane = async->lane_indices[0];
	status = SparkSpeculationPolicyResolveVerifierTokens(
		async->mtp_draft_tokens,SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH,
		slot->host_output_token_ids,SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MTP_DRAFT_DEPTH + 1u,
		SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT,&result);
	if ( status == SPARK_STATUS_OK &&
		result.committed_token_count != result.accepted_draft_token_count + 1u )
		status = SPARK_STATUS_INTERNAL_ERROR;
	error = cudaSuccess;
	launch = 0;
	if ( status == SPARK_STATUS_OK )
	{
		async->completion.accepted_token_count = result.committed_token_count;
		async->completion.tokens_per_sequence = result.committed_token_count;
		async->burst_token_count = result.committed_token_count;
		async->lane_next_positions[0] += result.accepted_draft_token_count;
		async->mtp_cache_extra = result.accepted_draft_token_count;
		error = cudaMemcpyAsync(
			state->mtp_lane_hidden_bf16 + (uint64_t)lane * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,
			slot->hc_mean_bf16 + (uint64_t)result.accepted_draft_token_count * SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION,
			(uint64_t)SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t),
			cudaMemcpyDeviceToDevice,(cudaStream_t)slot->stream);
		if ( error == cudaSuccess )
			launch = SparkGlm5NextLaunchCudaMtpCommit(&chain->wave,result.committed_token_count);
	}
	if ( status != SPARK_STATUS_OK || error != cudaSuccess || launch != 0 )
		async->completion.status = SPARK_STATUS_INTERNAL_ERROR;
	if ( SparkGlm5NextEnqueueAsyncCompletion(state,slot,chain->slot_index) != SPARK_STATUS_OK )
	{
		async->completion.status = SPARK_STATUS_INTERNAL_ERROR;
		SparkGlm5NextCompleteAsync(async);
	}
	chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_FINISH;
	chain->active = 0u;
	free(chain);
}

static void SparkGlm5NextTpChainFail(SparkGlm5NextTpChain *chain,SparkStatus status)
{
	SparkGlm5NextModuleState *state;
	SparkGlm5NextAsyncCompletion *async;
	state = chain->state;
	fprintf(stderr,"G5N-DBG chainfail: stage %u next_layer %u rows %u status %d\n",
		(unsigned)chain->stage,(unsigned)chain->next_layer,(unsigned)chain->wave_rows,(int)status);
	if ( cudaStreamSynchronize((cudaStream_t)chain->slot->stream) != cudaSuccess )
	{
		chain->retained_status = status;
		fprintf(stderr,"GLM chain drain failed; retaining slot %u and CUDA resources for teardown retry\n",chain->slot_index);
		atomic_store_explicit(&state->lazy_retained[chain->slot_index],chain,memory_order_release);
		return;
	}
	async = &state->completions[chain->slot_index];
	async->completion.status = status;
	chain->active = 0u;
	SparkGlm5NextCompleteAsync(async);
	free(chain);
}

static SparkStatus SparkGlm5NextLazyRelease(SparkGlm5NextTpChain *chain)
{
	SparkWeightdMap *map = chain->state->lazy_pack->map;
	SparkStatus status;
	if ( chain->expert_lease == 0u )
		return(SPARK_STATUS_OK);
	if ( chain->expert_lease_begun != 0u )
	{
		if ( chain->expert_lease_recorded == 0u )
		{
			status = SparkWeightdMapRecordCompletion(map,chain->expert_lease,(cudaStream_t)chain->slot->stream);
			if ( status != SPARK_STATUS_OK )
				return(status);
			chain->expert_lease_recorded = 1u;
			chain->wave.expert_lease_base = 0;
		}
		if ( cudaStreamSynchronize((cudaStream_t)chain->slot->stream) != cudaSuccess )
			return(SPARK_STATUS_IO_ERROR);
	}
	status = SparkWeightdMapRelease(map,chain->expert_lease,SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS);
	if ( status == SPARK_STATUS_OK )
	{
		chain->expert_lease = 0u;
		chain->expert_lease_begun = 0u;
		chain->expert_lease_recorded = 0u;
		chain->wave.expert_lease_base = 0;
	}
	return(status);
}

static SparkStatus SparkGlm5NextLazyRecoverLease(SparkGlm5NextModuleState *state,uint32_t slot,SparkGlm5NextTpChain **out)
{
	SparkGlm5NextTpChain *chain;
	SparkStatus status = SPARK_STATUS_OK;
	*out = 0;
	chain = atomic_exchange_explicit(&state->lazy_retained[slot],0,memory_order_acq_rel);
	if ( chain == 0 )
		return(SPARK_STATUS_NOT_FOUND);
	if ( chain->expert_lease != 0u )
		status = SparkGlm5NextLazyRelease(chain);
	if ( status != SPARK_STATUS_OK )
		atomic_store_explicit(&state->lazy_retained[slot],chain,memory_order_release);
	else
		*out = chain;
	return(status);
}

static void SparkGlm5NextLazyRetryRetained(void *context)
{
	SparkGlm5NextModuleState *state = (SparkGlm5NextModuleState *)context;
	SparkGlm5NextTpChain *chain;
	uint32_t slot;
	for (slot=0u; slot<state->pipeline_slot_count; slot++)
		if ( SparkGlm5NextLazyRecoverLease(state,slot,&chain) == SPARK_STATUS_OK )
			SparkGlm5NextTpChainFail(chain,chain->retained_status);
}

static void SparkGlm5NextTpChainReduceMlp(SparkGlm5NextTpChain *chain)
{
	SparkStatus status;
	chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_MLP;
	status = SparkGlm5NextModuleReduceAttentionOut(chain,chain->slot->attention_out_bf16);
	if ( status != SPARK_STATUS_OK )
		SparkGlm5NextTpChainFail(chain,status);
}

static SparkStatus SparkGlm5NextLazyExperts(SparkGlm5NextTpChain *chain)
{
	SparkWeightdExpertKey keys[SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT];
	SparkWeightdMap *map = chain->state->lazy_pack->map;
	SparkStatus status;
	void *address = 0;
	uint32_t count = 0u;
	if ( chain->slot->route_recorded == 0u || cudaEventSynchronize((cudaEvent_t)chain->slot->route_ready_event) != cudaSuccess )
		return(SPARK_STATUS_IO_ERROR);
	status = SparkWeightdRouteKeys(chain->wave.first_layer_index + chain->next_layer,chain->slot->host_group_row_offset,SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT,chain->wave.row_count * SPARK_GLM5_NEXT_MODEL_MOE_TOP_K,keys,SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT,&count);
	if ( status == SPARK_STATUS_OK )
		status = SparkWeightdMapAcquire(map,keys,count,&chain->expert_lease,SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS);
	if ( status == SPARK_STATUS_OK )
		status = SparkWeightdMapBeginUse(map,chain->expert_lease,&address);
	if ( status != SPARK_STATUS_OK )
		return(status);
	chain->expert_lease_begun = 1u;
	chain->wave.expert_lease_base = (const uint8_t *)address;
	chain->wave.expert_lease_local_layer = chain->next_layer;
	if ( SparkGlm5NextLaunchCudaLayerMlpExperts(&chain->wave,chain->next_layer) != 0 )
		return(SPARK_STATUS_INTERNAL_ERROR);
	return(SPARK_STATUS_OK);
}

static void SparkGlm5NextLazyWork(void *context)
{
	SparkGlm5NextTpChain *chain = (SparkGlm5NextTpChain *)context;
	SparkStatus status,cleanup;
	status = SparkGlm5NextLazyExperts(chain);
	cleanup = SparkGlm5NextLazyRelease(chain);
	if ( cleanup == SPARK_STATUS_IO_ERROR || cleanup == SPARK_STATUS_BUSY )
		cleanup = SparkGlm5NextLazyRelease(chain);
	if ( cleanup != SPARK_STATUS_OK )
	{
		chain->retained_status = status != SPARK_STATUS_OK ? status : cleanup;
		fprintf(stderr,"GLM expert cleanup failed: slot=%u lease=%llu status=%d; retaining slot and lease for teardown retry\n",chain->slot_index,(unsigned long long)chain->expert_lease,(int32_t)cleanup);
		atomic_store_explicit(&chain->state->lazy_retained[chain->slot_index],chain,memory_order_release);
		return;
	}
	if ( status != SPARK_STATUS_OK )
		SparkGlm5NextTpChainFail(chain,status);
	else
		SparkGlm5NextTpChainReduceMlp(chain);
}

static void SparkGlm5NextTpChainAdvance(void *chain_context,SparkStatus status)
{
	SparkGlm5NextTpChain *chain;
	SparkGlm5NextModuleState *state;
	SparkStatus launch_status;
	cudaError_t error;
	chain = (SparkGlm5NextTpChain *)chain_context;
	if ( chain == 0 || chain->active == 0u )
		return;
	state = chain->state;
	if ( status != SPARK_STATUS_OK )
	{
		SparkGlm5NextTpChainFail(chain,status);
		return;
	}
	switch ( chain->stage )
	{
	case SPARK_GLM5_NEXT_CHAIN_STAGE_BEGIN:
		SparkGlm5NextBuildWave(chain);
		if ( SparkGlm5NextLaunchCudaWaveBegin(&chain->wave) != 0 )
		{
			SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_ATTENTION;
		chain->next_layer = 0u;
		launch_status = SparkGlm5NextModuleReduceHidden(chain,chain->slot->hidden_bf16);
		if ( launch_status != SPARK_STATUS_OK )
			SparkGlm5NextTpChainFail(chain,launch_status);
		return;
	case SPARK_GLM5_NEXT_CHAIN_STAGE_ATTENTION:
		if ( SparkGlm5NextLaunchCudaLayerAttention(&chain->wave,chain->next_layer) != 0 )
		{
			SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_ATTENTION;
		launch_status = SparkGlm5NextModuleReduceAttentionOut(chain,chain->slot->attention_out_bf16);
		if ( launch_status != SPARK_STATUS_OK )
			SparkGlm5NextTpChainFail(chain,launch_status);
		return;
	case SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_ATTENTION:
		if ( SparkGlm5NextLaunchCudaLayerAttentionPost(&chain->wave,chain->next_layer) != 0 )
		{
			SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_MLP;
		SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
		return;
	case SPARK_GLM5_NEXT_CHAIN_STAGE_MLP:
		if ( state->lazy_pack != 0 && (chain->wave.first_layer_index + chain->next_layer) >= SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER )
		{
			if ( SparkGlm5NextLaunchCudaLayerMlpRoute(&chain->wave,chain->next_layer) != 0 )
				SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			else
			{
				launch_status = SparkWeightdWorkerSubmit(state->lazy_pack->worker,SparkGlm5NextLazyWork,chain);
				if ( launch_status != SPARK_STATUS_OK )
					SparkGlm5NextTpChainFail(chain,launch_status);
			}
			return;
		}
		if ( SparkGlm5NextLaunchCudaLayerMlp(&chain->wave,chain->next_layer) != 0 )
		{
			SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		SparkGlm5NextTpChainReduceMlp(chain);
		return;
	case SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_MLP:
		if ( SparkGlm5NextLaunchCudaLayerMlpPost(&chain->wave,chain->next_layer) != 0 )
		{
			SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->next_layer++;
		if ( chain->next_layer < chain->wave.layer_count )
		{
			chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_ATTENTION;
			SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
		}
		else
		{
			chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_HEAD;
			SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
		}
		return;
	case SPARK_GLM5_NEXT_CHAIN_STAGE_HEAD:
		if ( SparkGlm5NextLaunchCudaWaveHead(&chain->wave) != 0 )
		{
			SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_HEAD;
		launch_status = SparkGlm5NextModuleReduceHeadMax(chain);
		if ( launch_status != SPARK_STATUS_OK )
			SparkGlm5NextTpChainFail(chain,launch_status);
		return;
	case SPARK_GLM5_NEXT_CHAIN_STAGE_REDUCE_HEAD:
		error = SparkGlm5NextLaunchHeadMaxlocUnpack((cudaStream_t)chain->slot->stream,chain->slot->head_maxloc_u64,chain->slot->output_token,chain->wave_rows);
		if ( error == cudaSuccess && state->owns_final_head != 0u )
			error = cudaMemcpyAsync(chain->slot->host_output_token_ids + chain->first_row,chain->slot->output_token,(uint64_t)chain->wave_rows * sizeof(uint32_t),cudaMemcpyDeviceToHost,(cudaStream_t)chain->slot->stream);
		launch_status = SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"tp_head_unpack");
		if ( launch_status != SPARK_STATUS_OK )
		{
			SparkGlm5NextTpChainFail(chain,launch_status);
			return;
		}
		if ( chain->spec_verify != 0u )
		{
			error = cudaLaunchHostFunc((cudaStream_t)chain->slot->stream,SparkGlm5NextMtpResolveHost,chain);
			if ( error != cudaSuccess )
			{
				SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
				return;
			}
			return;
		}
		if ( state->mtp_enabled != 0u && state->owns_final_head != 0u )
		{
			launch_status = SparkGlm5NextMtpStashHidden(state,chain);
			if ( launch_status != SPARK_STATUS_OK )
			{
				SparkGlm5NextTpChainFail(chain,launch_status);
				return;
			}
		}
		if ( chain->next_wave_row < chain->batch->row_count )
		{
			uint32_t next_wave;
			next_wave = SparkGlm5NextRoundMajorWaveRows(chain->state,chain->batch,chain->next_wave_row);
			if ( next_wave == 0u )
			{
				SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INVALID_ARGUMENT);
				return;
			}
			chain->first_row = chain->next_wave_row;
			chain->wave_rows = next_wave;
			chain->next_wave_row += next_wave;
			chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_BEGIN;
			SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
			return;
		}
		launch_status = SparkGlm5NextEnqueueAsyncCompletion(state,chain->slot,chain->slot_index);
		if ( launch_status != SPARK_STATUS_OK )
		{
			SparkGlm5NextTpChainFail(chain,launch_status);
			return;
		}
		chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_FINISH;
		chain->active = 0u;
		free(chain);
		return;
	default:
		SparkGlm5NextTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
		return;
	}
}

static SparkStatus SparkGlm5NextStageHostBatch(
	const SparkGlm5NextModuleState *state,
	SparkGlm5NextExecutionSlot *slot,
	const SparkGlm5NextResidentDecodeStageBatchView *batch)
{
	uint32_t row;
	if ( state == 0 || slot == 0 || batch == 0 || batch->row_count > SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (row=0u; row<batch->row_count; row++)
	{
		if ( batch->row_positions[row] >= UINT32_MAX )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		slot->host_resident_slots[row] = batch->row_resident_slots[row];
		slot->host_positions[row] = (uint32_t)batch->row_positions[row];
		if ( state->owns_embedding != 0u )
			slot->host_token_ids[row] = batch->token_ids[row];
	}
	memset(slot->host_kv_access_error,0,SPARK_GLM5_NEXT_KV_ACCESS_ERROR_WORD_COUNT * sizeof(uint32_t));
	return(SPARK_STATUS_OK);
}

static void SparkGlm5NextPrepareAsyncCompletion(
	SparkGlm5NextModuleState *state,
	SparkModelDriverFrame *frame,
	const SparkGlm5NextResidentDecodeStageBatchView *batch,
	const uint8_t *lane_bound,
	const uint64_t *lane_sequence_ids,
	const uint64_t *lane_next_positions,
	uint32_t slot_index)
{
	SparkGlm5NextAsyncCompletion *async;
	uint32_t lane;
	async = &state->completions[slot_index];
	memset(async,0,sizeof(*async));
	async->state = state;
	async->completion_function = frame->completion_function;
	async->completion_context = frame->completion_context;
	async->slot_index = slot_index;
	async->lane_count = batch->active_sequence_count;
	async->row_count = batch->row_count;
	async->output_token_destination = state->owns_final_head != 0u ? (uint32_t *)frame->buffers[0].address : 0;
	for (lane=0u; lane<batch->active_sequence_count && lane<SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT; lane++)
	{
		async->lane_indices[lane] = batch->row_resident_slots[lane];
		async->lane_bound[lane] = lane_bound[lane];
		async->lane_sequence_ids[lane] = lane_sequence_ids[lane];
		async->lane_next_positions[lane] = lane_next_positions[lane];
	}
	async->completion.request_id = frame->request_id;
	async->completion.sequence_id = frame->sequence_id;
	async->completion.sequence_position = frame->sequence_position;
	async->completion.program_id = frame->program_id;
	async->completion.driver_dispatch_slot = frame->driver_dispatch_slot;
	async->completion.accepted_token_count = frame->new_token_count;
	async->completion.tokens_per_sequence = frame->tokens_per_sequence;
	async->completion.status = SPARK_STATUS_OK;
	async->completion.residency = frame->residency;
	async->completion.host_staging_bytes = (uint64_t)batch->row_count * sizeof(uint32_t) * (3u + state->owns_final_head);
	async->completion.device_memcpy_bytes = async->completion.host_staging_bytes;
}

static SparkStatus SparkGlm5NextCaptureRecurrent(SparkGlm5NextModuleState *state,uint32_t resident)
{
	SparkKvLaneTransaction *owner;
	SparkKvPageCacheSequence *sequence;
	uint32_t page;
	uint64_t generation;
	SparkStatus status;
	if ( state->kda_layer_count == 0u )
		return(SPARK_STATUS_OK);
	if ( resident >= state->resident_sequence_capacity || state->kv_lane_transactions == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	owner = &state->kv_lane_transactions[resident];
	if ( (owner->lane.flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH) == 0u )
		return(SPARK_STATUS_OK);
	if ( owner->phase != SPARK_KV_LANE_TRANSACTION_EXECUTING )
		return(SPARK_STATUS_VALIDATION_FAILED);
	sequence = &state->kv_transactions.cache->sequences[resident];
	page = sequence->mutable_logical_page_index;
	if ( page >= state->page_count || state->kv_blocks[page].residency_reference_count == 0u )
		return(SPARK_STATUS_VALIDATION_FAILED);
	generation = state->kv_blocks[page].generation;
	status = SparkGlm5NextRecurrentCopy(state,SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST,resident,state->recurrent_staging,state->recurrent_page_bytes);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkKvPageStoreWriteback(&state->recurrent_store,page,resident,generation,(uintptr_t)state->recurrent_staging,state->recurrent_page_bytes,0u,0u);
	while ( status == SPARK_STATUS_BUSY )
	{
		if ( SparkKvPageStoreWaitForTransfers(&state->recurrent_store) != SPARK_STATUS_OK )
			return(SPARK_STATUS_BUSY);
		status = SparkKvPageStoreWriteback(&state->recurrent_store,page,resident,generation,(uintptr_t)state->recurrent_staging,state->recurrent_page_bytes,0u,0u);
	}
	return(status);
}

static SparkStatus SparkGlm5NextFinishCacheLanes(SparkGlm5NextAsyncCompletion *async)
{
	SparkGlm5NextModuleState *state = async->state;
	SparkStatus result;
	uint32_t lane,resident;
	if ( pthread_mutex_lock(&state->kv_mutex) != 0 )
		return(SPARK_STATUS_INTERNAL_ERROR);
	result = async->completion.status;
	for (lane=0u; lane<async->lane_count && result==SPARK_STATUS_OK; lane++)
		result = SparkGlm5NextCaptureRecurrent(state,async->lane_indices[lane]);
	if ( result == SPARK_STATUS_BUSY )
	{
		(void)pthread_mutex_unlock(&state->kv_mutex);
		return(result);
	}
	result = SparkKvLaneTransactionsFinish(&state->kv_transactions,async->lane_indices,async->lane_count,result,(uint32_t)async->mtp_cache_extra);
	for (lane=0u; lane<async->lane_count; lane++)
	{
		resident = async->lane_indices[lane];
		atomic_store_explicit(&state->lane_bound[resident],result == SPARK_STATUS_OK ? async->lane_bound[lane] : 0u,memory_order_release);
		if ( result == SPARK_STATUS_OK )
		{
			atomic_store_explicit(&state->lane_sequence_ids[resident],async->lane_sequence_ids[lane],memory_order_release);
			atomic_store_explicit(&state->lane_next_positions[resident],async->lane_next_positions[lane],memory_order_release);
		}
		else
		{
			memset(state->page_table_shadow + (uint64_t)resident * state->pages_per_sequence,0xff,(uint64_t)state->pages_per_sequence * sizeof(uint32_t));
			if ( state->mtp_lane_armed != 0 )
				state->mtp_lane_armed[resident] = 0u;
		}
	}
	(void)pthread_mutex_unlock(&state->kv_mutex);
	return(result);
}

static void SparkGlm5NextCompleteOnWorker(void *context)
{
	SparkGlm5NextAsyncCompletion *async = (SparkGlm5NextAsyncCompletion *)context;
	SparkGlm5NextModuleState *state = async != 0 ? async->state : 0;
	SparkGlm5NextExecutionSlot *slot;
	SparkModelDriverCompletion completion;
	SparkModelDriverCompletionFunction complete;
	void *complete_context;
	if ( state == 0 || async->slot_index >= state->pipeline_slot_count )
		return;
	slot = &state->slots[async->slot_index];
	if ( slot->host_kv_access_error[0] != 0u )
	{
		fprintf(stderr,"GLM cache access failed: code %u row %u slot %u\n",slot->host_kv_access_error[0],slot->host_kv_access_error[2],async->slot_index);
		async->completion.status = SPARK_STATUS_INTERNAL_ERROR;
	}
	async->completion.status = SparkGlm5NextFinishCacheLanes(async);
	if ( async->completion.status == SPARK_STATUS_BUSY )
	{
		fprintf(stderr,"GLM checkpoint completion not quiescent; retaining lane and slot ownership\n");
		return;
	}
	if ( async->completion.status == SPARK_STATUS_OK )
	{
		if ( async->output_token_destination != 0 )
			memcpy(async->output_token_destination,slot->host_output_token_ids,(uint64_t)(async->burst_token_count != 0u ? async->burst_token_count : async->row_count) * sizeof(uint32_t));
		atomic_fetch_add_explicit(&state->completed_count,1u,memory_order_relaxed);
	}
	else
		atomic_fetch_add_explicit(&state->failed_count,1u,memory_order_relaxed);
	atomic_fetch_add_explicit(&state->host_callback_completion_count,1u,memory_order_relaxed);
	// Snapshot before releasing the slot: callback-driven reuse can overwrite async.
	completion = async->completion;
	complete = async->completion_function;
	complete_context = async->completion_context;
	SparkStageModuleIndexSetRelease(state->lane_states,state->resident_sequence_capacity,async->lane_indices,async->lane_count);
	SparkStageModuleSlotRelease(state->slot_states,async->slot_index);
	complete(complete_context,&completion);
}

static void CUDART_CB SparkGlm5NextCompleteAsync(void *context)
{
	SparkGlm5NextAsyncCompletion *async = (SparkGlm5NextAsyncCompletion *)context;
	SparkStatus status;
	if ( async == 0 || async->state == 0 )
		return;
	status = SparkWeightdWorkerSubmit(async->state->completion_worker,SparkGlm5NextCompleteOnWorker,async);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"GLM completion handoff failed: status %d; retaining lane and slot ownership\n",(int32_t)status);
}

static SparkStatus SparkGlm5NextEnqueueAsyncCompletion(
	SparkGlm5NextModuleState *state,
	SparkGlm5NextExecutionSlot *slot,
	uint32_t slot_index)
{
	cudaStream_t stream;
	cudaError_t error;
	stream = (cudaStream_t)slot->stream;
	error = cudaMemcpyAsync(slot->host_kv_access_error,slot->kv_access_error,SPARK_GLM5_NEXT_KV_ACCESS_ERROR_WORD_COUNT * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream);
	if ( error == cudaSuccess )
		error = cudaLaunchHostFunc(stream,SparkGlm5NextCompleteAsync,&state->completions[slot_index]);
	return(SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"async_completion"));
}

static SparkStatus SparkGlm5NextClaimCacheFrame(SparkGlm5NextModuleState *state,const SparkModelDriverFrame *frame,const SparkGlm5NextResidentDecodeStageBatchView *batch,const uint64_t *next_positions)
{
	uint32_t lane;
	SparkStatus status;
	if ( frame->cache_lanes == 0 || frame->cache_lane_count != batch->active_sequence_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID) == 0u || frame->driver_dispatch_slot != (uint32_t)(frame->request_id % state->pipeline_slot_count) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	for (lane=0u; lane<frame->cache_lane_count; lane++)
		if ( frame->cache_lanes[lane].resident_sequence_slot != batch->row_resident_slots[lane] || frame->cache_lanes[lane].sequence_id != batch->row_sequence_ids[lane] || frame->cache_lanes[lane].sequence_position != batch->row_positions[lane] || frame->cache_lanes[lane].context_token_count != next_positions[lane] )
			return(SPARK_STATUS_VALIDATION_FAILED);
	if ( pthread_mutex_lock(&state->kv_mutex) != 0 )
		return(SPARK_STATUS_INTERNAL_ERROR);
	status = SparkKvLaneTransactionsClaim(&state->kv_transactions,frame);
	(void)pthread_mutex_unlock(&state->kv_mutex);
	return(status);
}

static SparkStatus SparkGlm5NextUploadPageTables(SparkGlm5NextModuleState *state,const SparkGlm5NextAsyncCompletion *async,void *stream)
{
	uint32_t lane,resident;
	uint64_t offset,bytes;
	cudaError_t error;
	for (lane=0u; lane<async->lane_count; lane++)
	{
		resident = async->lane_indices[lane];
		offset = ((uint64_t)resident * state->pages_per_sequence);
		bytes = ((uint64_t)state->kv_lane_transactions[resident].page_count * sizeof(uint32_t));
		if ( memcmp(state->page_table_shadow + offset,state->kv_lane_physical_pages + offset,bytes) == 0 )
			continue;
		error = cudaMemcpyAsync(state->page_table + offset,state->kv_lane_physical_pages + offset,bytes,cudaMemcpyHostToDevice,(cudaStream_t)stream);
		if ( error != cudaSuccess )
			return(SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"page_table_update"));
		memcpy(state->page_table_shadow + offset,state->kv_lane_physical_pages + offset,bytes);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextRestoreRecurrent(SparkGlm5NextModuleState *state,uint32_t resident)
{
	const SparkKvLaneTransaction *owner = &state->kv_lane_transactions[resident];
	SparkKvPageCache *cache = state->kv_transactions.cache;
	uint32_t entry,page;
	uint64_t generation;
	SparkStatus status;
	if ( SparkGlm5NextPrefixRestorePending(owner) == 0u || state->kda_layer_count == 0u )
		return(SPARK_STATUS_OK);
	if ( owner->phase != SPARK_KV_LANE_TRANSACTION_EXECUTING )
		return(SPARK_STATUS_VALIDATION_FAILED);
	entry = cache->sequences[resident].terminal_entry_index;
	if ( entry >= cache->entry_capacity || cache->entries[entry].token_count != owner->lane.sequence_position )
		return(SPARK_STATUS_VALIDATION_FAILED);
	page = cache->entries[entry].logical_page_index;
	if ( page >= state->page_count || state->kv_blocks[page].residency_reference_count == 0u )
		return(SPARK_STATUS_VALIDATION_FAILED);
	generation = state->kv_blocks[page].generation;
	status = SparkKvPageStoreReadback(&state->recurrent_store,page,generation,(uintptr_t)state->recurrent_staging,state->recurrent_page_bytes);
	while ( status == SPARK_STATUS_BUSY )
	{
		if ( SparkKvPageStoreWaitForTransfers(&state->recurrent_store) != SPARK_STATUS_OK )
			return(SPARK_STATUS_BUSY);
		status = SparkKvPageStoreReadback(&state->recurrent_store,page,generation,(uintptr_t)state->recurrent_staging,state->recurrent_page_bytes);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextRecurrentCopy(state,SPARK_KV_PAGE_STORE_COPY_HOST_TO_DEVICE,resident,state->recurrent_staging,state->recurrent_page_bytes);
	return(status);
}

static SparkStatus SparkGlm5NextRestoreCacheLanes(SparkGlm5NextModuleState *state,const SparkGlm5NextAsyncCompletion *async)
{
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t lane;
	if ( pthread_mutex_lock(&state->kv_mutex) != 0 )
		return(SPARK_STATUS_INTERNAL_ERROR);
	for (lane=0u; lane<async->lane_count && status==SPARK_STATUS_OK; lane++)
		status = SparkGlm5NextRestoreRecurrent(state,async->lane_indices[lane]);
	(void)pthread_mutex_unlock(&state->kv_mutex);
	return(status);
}

static SparkStatus SparkGlm5NextStartClaimedBatch(SparkGlm5NextModuleState *state,SparkModelDriverFrame *frame,const SparkGlm5NextResidentDecodeStageFrameContext *context,uint32_t slot_index)
{
	SparkGlm5NextExecutionSlot *slot = &state->slots[slot_index];
	SparkGlm5NextTpChain *chain;
	SparkStatus status;
	cudaError_t error;
	chain = (SparkGlm5NextTpChain *)calloc(1u,sizeof(*chain));
	if ( chain == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkGlm5NextClaimCacheFrame(state,frame,context->batch,state->completions[slot_index].lane_next_positions);
	if ( status != SPARK_STATUS_OK )
	{
		free(chain);
		return(status);
	}
	chain->state = state;
	chain->slot = slot;
	chain->slot_index = slot_index;
	chain->frame = frame;
	chain->context = context;
	chain->batch = context->batch;
	chain->wave_rows = SparkGlm5NextRoundMajorWaveRows(state,context->batch,0u);
	chain->next_wave_row = chain->wave_rows;
	chain->stage = SPARK_GLM5_NEXT_CHAIN_STAGE_BEGIN;
	chain->active = 1u;
	atomic_fetch_add_explicit(&state->submitted_count,1u,memory_order_relaxed);
	status = SparkGlm5NextRestoreCacheLanes(state,&state->completions[slot_index]);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextUploadPageTables(state,&state->completions[slot_index],slot->stream);
	if ( status == SPARK_STATUS_OK )
	{
		error = cudaMemsetAsync(slot->kv_access_error,0,SPARK_GLM5_NEXT_KV_ACCESS_ERROR_WORD_COUNT * sizeof(uint32_t),(cudaStream_t)slot->stream);
		status = SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"kv_access_reset");
	}
	if ( status == SPARK_STATUS_OK && chain->wave_rows == 0u )
		status = SPARK_STATUS_INVALID_ARGUMENT;
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextMtpDriveDraft(state,frame,context->batch,slot,chain);
	if ( status != SPARK_STATUS_OK )
		SparkGlm5NextTpChainFail(chain,status);
	else
		SparkGlm5NextTpChainAdvance(chain,SPARK_STATUS_OK);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextExecuteBatch(SparkGlm5NextModuleState *state,SparkModelDriverFrame *frame,const SparkGlm5NextResidentDecodeStageFrameContext *context)
{
	const SparkGlm5NextResidentDecodeStageBatchView *batch = context->batch;
	SparkGlm5NextClaimedContinuityContext continuity;
	SparkGlm5NextExecutionSlot *slot;
	uint8_t simulated_bound[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t simulated_sequence[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t simulated_next[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t slot_index;
	uint64_t last_ordinal;
	SparkStatus status;
	status = SparkTpChainOrdinal(frame->request_id,state->pipeline_slot_count,SPARK_GLM5_NEXT_TP_COLLECTIVE_CREDITS_PER_SLOT,SPARK_GLM5_NEXT_TP_CHAIN_OPERATIONS,SPARK_GLM5_NEXT_TP_CHAIN_OPERATIONS - 1u,&last_ordinal);
	if ( status != SPARK_STATUS_OK )
		return(status);
	continuity.state = state;
	continuity.batch = batch;
	continuity.bound = simulated_bound;
	continuity.sequence_ids = simulated_sequence;
	continuity.next_positions = simulated_next;
	status = SparkStageModuleIndexSetClaimAndPrepare(state->lane_states,state->resident_sequence_capacity,batch->row_resident_slots,batch->active_sequence_count,SparkGlm5NextPrepareClaimedContinuity,&continuity);
	if ( status != SPARK_STATUS_OK )
		return(status);
	slot_index = (uint32_t)(frame->request_id % state->pipeline_slot_count);
	status = SparkStageModuleIndexSetClaim(state->slot_states,state->pipeline_slot_count,&slot_index,1u);
	if ( status == SPARK_STATUS_OK )
	{
		slot = &state->slots[slot_index];
		slot->stream = frame->execution_stream;
		status = SparkGlm5NextStageHostBatch(state,slot,batch);
		if ( status == SPARK_STATUS_OK )
		{
			SparkGlm5NextPrepareAsyncCompletion(state,frame,batch,simulated_bound,simulated_sequence,simulated_next,slot_index);
			status = SparkGlm5NextStartClaimedBatch(state,frame,context,slot_index);
		}
		if ( status != SPARK_STATUS_OK )
			SparkStageModuleSlotRelease(state->slot_states,slot_index);
	}
	if ( status != SPARK_STATUS_OK )
		SparkStageModuleIndexSetRelease(state->lane_states,state->resident_sequence_capacity,batch->row_resident_slots,batch->active_sequence_count);
	return(status);
}

SparkStatus SparkGlm5NextResidentDecodeStageExecute(
	void *module_state,
	SparkModelDriverFrame *frame)
{
	SparkGlm5NextModuleState *state;
	const SparkGlm5NextResidentDecodeStageFrameContext *context;
	SparkStatus status;
	state = (SparkGlm5NextModuleState *)module_state;
	context = 0;
	status = SparkGlm5NextValidateFrame(state,frame,&context);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"G5N-DBG execute: ValidateFrame -> %d\n",(int)status);
		if ( state != 0 )
			atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
		return(status);
	}
	status = SparkGlm5NextExecuteBatch(state,frame,context);
	if ( status != SPARK_STATUS_OK )
		atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
	return(status);
}

static void SparkGlm5NextAdmissionCost(
	void *context,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	(void)context;
	decision->host_staging_bytes = (uint64_t)request->new_token_count *
		(sizeof(uint32_t) * 3u + sizeof(uint64_t) * 2u);
	decision->device_memcpy_bytes = decision->host_staging_bytes;
}

static SparkStatus SparkGlm5NextResetExecutionState(SparkGlm5NextModuleState *state)
{
	cudaError_t error = cudaSuccess,drain;
	uint32_t lane;
	if ( state->kda_layer_count != 0u )
	{
		error = cudaMemsetAsync(state->kda_state_pools,0,state->kda_state_layer_stride_bytes * state->kda_layer_count,(cudaStream_t)state->execution_stream);
		if ( error == cudaSuccess )
			error = cudaMemsetAsync(state->kda_window_pools,0,state->kda_window_layer_stride_bytes * state->kda_layer_count * 3u,(cudaStream_t)state->execution_stream);
	}
	drain = cudaStreamSynchronize((cudaStream_t)state->execution_stream);
	if ( drain != cudaSuccess )
	{
		(void)SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,drain,"reset_stream_drain");
		return(SPARK_STATUS_PENDING);
	}
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,error,"cache_reset"));
	memset(state->page_table_shadow,0xff,(uint64_t)state->resident_sequence_capacity * state->pages_per_sequence * sizeof(uint32_t));
	for (lane=0u; lane<state->resident_sequence_capacity; lane++)
	{
		atomic_store_explicit(&state->lane_bound[lane],0u,memory_order_release);
		atomic_store_explicit(&state->lane_sequence_ids[lane],0u,memory_order_release);
		atomic_store_explicit(&state->lane_next_positions[lane],0u,memory_order_release);
		if ( state->mtp_lane_armed != 0 )
			state->mtp_lane_armed[lane] = 0u;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm5NextResetClaimed(SparkGlm5NextModuleState *state,uint64_t generation)
{
	SparkStatus status;
	if ( pthread_mutex_lock(&state->kv_mutex) != 0 )
		return(SPARK_STATUS_INTERNAL_ERROR);
	status = SPARK_STATUS_VALIDATION_FAILED;
	if ( generation >= state->control_generation && generation > state->reset_generation )
	{
		state->control_generation = generation;
		status = SparkKvLaneTransactionsReset(&state->kv_transactions);
		if ( status == SPARK_STATUS_OK )
			status = SparkGlm5NextResetExecutionState(state);
		if ( status == SPARK_STATUS_OK )
			state->reset_generation = generation;
	}
	(void)pthread_mutex_unlock(&state->kv_mutex);
	return(status);
}

static SparkStatus SparkGlm5NextReset(SparkGlm5NextModuleState *state,const SparkModelDriverAdmissionRequest *request)
{
	uint32_t slots[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	uint32_t lanes[SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t index;
	SparkStatus status;
	if ( SparkModelDriverAdmissionRequestIsValid(request) == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (index=0u; index<state->pipeline_slot_count; index++)
		slots[index] = index;
	for (index=0u; index<state->resident_sequence_capacity; index++)
		lanes[index] = index;
	status = SparkStageModuleIndexSetClaim(state->slot_states,state->pipeline_slot_count,slots,state->pipeline_slot_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkStageModuleIndexSetClaim(state->lane_states,state->resident_sequence_capacity,lanes,state->resident_sequence_capacity);
	if ( status == SPARK_STATUS_OK )
	{
		status = SparkGlm5NextResetClaimed(state,request->control_generation);
		if ( status == SPARK_STATUS_PENDING )
		{
			fprintf(stderr,"GLM reset stream not quiescent; retaining lane and slot ownership\n");
			return(status);
		}
		SparkStageModuleIndexSetRelease(state->lane_states,state->resident_sequence_capacity,lanes,state->resident_sequence_capacity);
	}
	SparkStageModuleIndexSetRelease(state->slot_states,state->pipeline_slot_count,slots,state->pipeline_slot_count);
	return(status);
}

SparkStatus SparkGlm5NextResidentDecodeStageAdmit(
	void *module_state,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	SparkGlm5NextModuleState *state;
	SparkAdmissionPolicyTable table;
	uint32_t available;
	SparkStatus status;
	state = (SparkGlm5NextModuleState *)module_state;
	if ( state == 0 || request == 0 || decision == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( request->admission_flags == SPARK_MODEL_DRIVER_ADMISSION_FLAG_RESET )
	{
		SparkModelDriverInitializeAdmissionDecision(decision);
		status = SparkGlm5NextReset(state,request);
		if ( status == SPARK_STATUS_OK )
		{
			decision->accepted = 1u;
			decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
		}
		return(status);
	}
	if ( (request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE) != 0u )
	{
		if ( SparkModelDriverAdmissionRequestIsValid(request) == 0u || request->new_token_count != 0u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		SparkModelDriverInitializeAdmissionDecision(decision);
		return(SparkGlm5NextAdmissionPredicate(state,request,decision));
	}
	available = request->request_id != 0u && atomic_load_explicit(&state->slot_states[request->request_id % state->pipeline_slot_count],memory_order_acquire) == SPARK_STAGE_MODULE_SLOT_FREE ? 1u : 0u;
	memset(&table,0,sizeof(table));
	table.abi_version = SPARK_ADMISSION_ABI_VERSION;
	table.descriptor_bytes = (uint32_t)sizeof(table);
	table.max_active_sequence_count = state->resident_sequence_capacity;
	table.max_input_row_count = SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT;
	table.max_sequence_positions = state->max_sequence_positions;
	table.flags = SPARK_ADMISSION_POLICY_FLAG_DECODE_EQUALS_SLOTS |
		SPARK_ADMISSION_POLICY_FLAG_ALLOW_DISPATCH_FLAG;
	table.predicate = SparkGlm5NextAdmissionPredicate;
	table.predicate_context = state;
	table.cost = SparkGlm5NextAdmissionCost;
	table.cost_context = state;
	status = SparkAdmissionEvaluateShape(&table,available,request,decision);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( decision->accepted == 0u )
		fprintf(stderr,"G5N-DBG admit: shape-rejected reason %u\n",
			(unsigned)decision->rejection_reason);
	if ( decision->accepted == 0u )
		atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
	return(status);
}

SparkStatus SparkGlm5NextResidentDecodeStageSnapshot(
	void *module_state,
	uint32_t program_id,
	SparkModelDriverRuntimeSnapshot *snapshot)
{
	SparkGlm5NextModuleState *state;
	uint32_t index,resident_count;
	state = (SparkGlm5NextModuleState *)module_state;
	if ( state == 0 || snapshot == 0 || program_id == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	SparkStageModuleRuntimeSnapshotInitialize(snapshot,program_id,state->slot_states,state->pipeline_slot_count);
	snapshot->submitted_count = atomic_load_explicit(&state->submitted_count,memory_order_relaxed);
	snapshot->completed_count = atomic_load_explicit(&state->completed_count,memory_order_relaxed);
	snapshot->rejected_count = atomic_load_explicit(&state->rejected_count,memory_order_relaxed);
	snapshot->host_callback_completion_count = atomic_load_explicit(&state->host_callback_completion_count,memory_order_relaxed);
	resident_count = 0u;
	for (index=0u; index<state->resident_sequence_capacity; index++)
		resident_count += atomic_load_explicit(&state->lane_bound[index],memory_order_acquire) != 0u ? 1u : 0u;
	snapshot->resident_sequence_count = resident_count;
	snapshot->kv_token_capacity = (uint64_t)state->resident_sequence_capacity * state->max_sequence_positions;
	return(SPARK_STATUS_OK);
}

static void SparkGlm5NextReleaseCaches(SparkGlm5NextModuleState *state)
{
	if ( state->recurrent_store.abi_version == SPARK_KV_PAGE_STORE_ABI_VERSION )
		SparkKvPageStoreDestroy(&state->recurrent_store);
	if ( state->recurrent_staging != 0 )
		(void)cudaFreeHost(state->recurrent_staging);
	if ( state->kv_page_store.abi_version == SPARK_KV_PAGE_STORE_ABI_VERSION )
		SparkKvPageStoreDestroy(&state->kv_page_store);
	free(state->kda_state_index_host);
	free(state->kv_blocks);
	free(state->kv_resident_slot_logical_block_indices);
	free(state->kv_entries);
	free(state->kv_sequences);
	free(state->kv_hash_bucket_heads);
	free(state->kv_entry_indices_by_logical_page);
	free(state->kv_page_staging);
	free(state->kv_lane_logical_pages);
	free(state->page_table_shadow);
	free(state->kv_lane_transactions);
	if ( state->kv_lane_physical_pages != 0 )
		(void)cudaFreeHost(state->kv_lane_physical_pages);
	if ( state->kv_mutex_initialized != 0u )
		(void)pthread_mutex_destroy(&state->kv_mutex);
}

void SparkGlm5NextResidentDecodeStageDestroy(void *module_state)
{
	SparkGlm5NextModuleState *state;
	uint32_t slot;
	state = (SparkGlm5NextModuleState *)module_state;
	if ( state == 0 )
		return;
	for (slot=0u; slot<state->pipeline_slot_count; slot++)
		if ( atomic_load_explicit(&state->lazy_retained[slot],memory_order_acquire) != 0 )
			break;
	if ( slot < state->pipeline_slot_count )
	{
		if ( state->lazy_pack == 0 || state->lazy_pack->worker == 0 || SparkWeightdWorkerSubmit(state->lazy_pack->worker,SparkGlm5NextLazyRetryRetained,state) != SPARK_STATUS_OK )
			return;
	}
	if ( SparkStageModuleWaitForSlots(SPARK_GLM5_NEXT_MODULE_TAG,state->slot_states,state->pipeline_slot_count,SPARK_STAGE_MODULE_DESTROY_QUIESCE_TIMEOUT_NS) != SPARK_STATUS_OK )
		return;
	if ( state->lazy_pack != 0 && state->lazy_pack->worker != 0 && SparkWeightdWorkerWaitIdle(state->lazy_pack->worker,SPARK_STAGE_MODULE_DESTROY_QUIESCE_TIMEOUT_NS) != SPARK_STATUS_OK )
		return;
	if ( state->completion_worker != 0 )
	{
		if ( SparkWeightdWorkerWaitIdle(state->completion_worker,SPARK_STAGE_MODULE_DESTROY_QUIESCE_TIMEOUT_NS) != SPARK_STATUS_OK || SparkWeightdWorkerDestroy(state->completion_worker) != SPARK_STATUS_OK )
			return;
		state->completion_worker = 0;
	}
	if ( SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,cudaStreamSynchronize((cudaStream_t)state->execution_stream),"destroy_stream_drain") != SPARK_STATUS_OK )
		return;
	if ( state->lazy_pack != 0 )
	{
		if ( SparkWeightdLazyPackDestroy(state->lazy_pack) != SPARK_STATUS_OK )
			return;
		state->lazy_pack = 0;
	}
	if ( state->tp_device_collective_hc_initialized != 0u )
		SparkTpDeviceCollectiveDestroy(&state->tp_device_collective_hc);
	if ( state->tp_hc_host_credit_send_bf16 != 0 )
		(void)cudaFreeHost(state->tp_hc_host_credit_send_bf16);
	if ( state->tp_hc_host_credit_receive_bf16 != 0 )
		(void)cudaFreeHost(state->tp_hc_host_credit_receive_bf16);
	state->tp_hc_host_credit_send_bf16 = 0;
	state->tp_hc_host_credit_receive_bf16 = 0;
	if ( state->tp_device_collective_initialized != 0u )
		SparkTpDeviceCollectiveDestroy(&state->tp_device_collective);
	SparkGlm5NextReleaseCaches(state);
	SparkGlm5NextReleaseSlotHost(state);
	SparkStageModuleLedgerRelease(&state->ledger);
	free(state->mtp_lane_armed);
	free(state);
}

static SparkStatus SparkGlm5NextBuildHeadShadow(SparkGlm5NextModuleState *state)
{
	uint64_t head_rows,dim;
	SparkStatus status;
	if ( state->owns_final_head == 0u || state->lm_head_bf16 == 0 )
		return(SPARK_STATUS_OK);
	head_rows = SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT / state->tp_degree;
	dim = SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION;
	status = SparkGlm5NextAllocateBytes(state,head_rows,dim,1u,(void **)&state->head_certified_fp8_payload);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextAllocateBytes(state,head_rows,dim / 32u,sizeof(float),(void **)&state->head_certified_fp8_scale_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextAllocateBytes(state,head_rows,dim / 32u,sizeof(float),(void **)&state->head_certified_fp8_norm_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,SparkGlm5NextLaunchHeadCertifiedQuantize(0,state->lm_head_bf16,state->head_certified_fp8_payload,state->head_certified_fp8_scale_f32,state->head_certified_fp8_norm_f32,(uint32_t)head_rows,(uint32_t)dim),"head_certified_quantize");
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_GLM5_NEXT_MODULE_TAG,cudaDeviceSynchronize(),"head_certified_sync");
	return(status);
}

static SparkStatus SparkGlm5NextInitializeState(
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services,
	SparkGlm5NextModuleState **state_out)
{
	SparkGlm5NextModuleState *state;
	const char *pack_path;
	SparkStatus status;
	uint32_t lane;
	state = (SparkGlm5NextModuleState *)calloc(1u,sizeof(*state));
	if ( state == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->ledger.module_tag = SPARK_GLM5_NEXT_MODULE_TAG;
	for (lane=0u; lane<SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT; lane++)
		atomic_init(&state->lazy_retained[lane],0);
	status = SparkGlm5NextModuleConfigure(state,configuration,host_services,&pack_path);
	if ( status == SPARK_STATUS_OK && SparkGlm5NextConfigureCudaModule(&state->multiprocessor_count) != 0 )
		status = SPARK_STATUS_TARGET_MISMATCH;
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextPackLoad(state,pack_path);
	if ( status == SPARK_STATUS_OK && state->mtp_enabled != 0u && state->pack_has_mtp == 0u )
	{
		fprintf(stderr,"G5N-DBG config: MTP flag set but the pack carries no layer-45 tensors\n");
		status = SPARK_STATUS_SCHEMA_ERROR;
	}
	if ( status == SPARK_STATUS_OK && state->mtp_enabled != 0u && state->tp_degree != 1u )
	{
		fprintf(stderr,"G5N-DBG config: MTP speculation requires tp_degree 1 in this revision\n");
		status = SPARK_STATUS_UNSUPPORTED;
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextAllocateCaches(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextAllocateSlots(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextAllocateMtp(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextModuleInitializeTpCollective(state,(const SparkGlm5NextResidentDecodeStageNodeContext *)host_services->node_context);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm5NextBuildHeadShadow(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkWeightdWorkerCreate(&state->completion_worker);
	if ( status != SPARK_STATUS_OK )
	{
		if ( state->lazy_pack != 0 && SparkWeightdLazyPackDestroy(state->lazy_pack) != SPARK_STATUS_OK )
		{
			fprintf(stderr,"GLM lazy initialization cleanup failed; retaining CUDA resources until process exit\n");
			return(status);
		}
		SparkGlm5NextReleaseCaches(state);
		SparkGlm5NextReleaseSlotHost(state);
		SparkStageModuleLedgerRelease(&state->ledger);
		free(state->mtp_lane_armed);
		free(state);
		return(status);
	}
	SparkStageModuleAtomicStateArrayInitialize(state->slot_states,state->pipeline_slot_count);
	SparkStageModuleAtomicStateArrayInitialize(state->lane_states,state->resident_sequence_capacity);
	for (lane=0u; lane<state->resident_sequence_capacity; lane++)
	{
		atomic_init(&state->lane_bound[lane],0u);
		atomic_init(&state->lane_sequence_ids[lane],0u);
		atomic_init(&state->lane_next_positions[lane],0u);
	}
	atomic_init(&state->submitted_count,0u);
	atomic_init(&state->completed_count,0u);
	atomic_init(&state->rejected_count,0u);
	atomic_init(&state->failed_count,0u);
	atomic_init(&state->host_callback_completion_count,0u);
	atomic_init(&state->nccl_next_ordinal,0u);
	atomic_init(&state->nccl_next_ordinal_hc,0u);
	*state_out = state;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkGlm5NextResidentDecodeStageInitialize(
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services,
	void **module_state)
{
	SparkGlm5NextModuleState *state;
	SparkStatus status;
	status = SparkFirmwareModuleValidateInitialization(configuration,host_services,module_state);
	if ( status != SPARK_STATUS_OK )
		return(status);
	state = 0;
	status = SparkGlm5NextInitializeState(configuration,host_services,&state);
	if ( status != SPARK_STATUS_OK )
		return(status);
	*module_state = state;
	return(SPARK_STATUS_OK);
}
