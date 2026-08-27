#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_glm52_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_admission.h"
#include "sparkpipe/spark_kv_model_table.h"
#include "sparkpipe/spark_glm52_kv_geometry.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "spark_glm52_resident_decode_stage_internal.h"
#include "spark_glm52_stagepack_format.h"

#ifndef GLM52_EXPERT_WEIGHT_CODEC
#error "GLM52_EXPERT_WEIGHT_CODEC must name the exact package expert codec"
#endif
#ifndef GLM52_CONTRACT_SHA256
#error "GLM52_CONTRACT_SHA256 must identify the exact model package contract"
#endif

#define SPARK_GLM52_MODULE_TAG "glm52_stage"
#define SPARK_GLM52_STAGEPACK_MAX_TENSOR_COUNT 2048u
#define SPARK_GLM52_HEAD_TILE 1024u
#define SPARK_GLM52_NO_INDEX_ORDINAL UINT32_MAX
#define SPARK_GLM52_KV_ACCESS_ERROR_WORD_COUNT 6u

typedef struct SparkGlm52PackRange
{
	uint64_t offset;
	uint64_t bytes;
} SparkGlm52PackRange;

typedef struct SparkGlm52ModuleState SparkGlm52ModuleState;

typedef struct SparkGlm52AsyncCompletion
{
	SparkGlm52ModuleState *state;
	SparkModelDriverCompletionFunction completion_function;
	void *completion_context;
	uint32_t slot_index;
	uint32_t lane_count;
	uint32_t row_count;
	uint32_t lane_indices[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint8_t lane_bound[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_sequence_ids[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_next_positions[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t *output_token_destination;
	SparkModelDriverCompletion completion;
} SparkGlm52AsyncCompletion;

struct SparkGlm52ModuleState
{
	SparkStageModuleLedger ledger;
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
	uint32_t pages_per_sequence;
	uint32_t page_count;
	uint32_t index_layer_count;
	uint32_t multiprocessor_count;
	uint32_t owns_embedding;
	uint32_t owns_final_head;
	void *execution_stream;
	char model_revision[SPARK_GLM52_STAGEPACK_MODEL_REVISION_BYTES];
	SparkGlm52LayerWeights layers[SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE];
	uint32_t index_ordinal_by_local_layer[SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE];
	uint64_t layer_seen[SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE];
	uint64_t global_seen;
	const void *embedding_bf16;
	const void *final_norm_bf16;
	const void *lm_head_bf16;
	uint8_t *kv_cache;
	uint64_t kv_layer_stride_bytes;
	uint8_t *index_cache;
	uint64_t index_layer_stride_bytes;
	uint32_t *page_table;
	SparkKvCacheArena kv_arena;
	SparkKvPageCache kv_page_cache;
	SparkKvPageStore kv_page_store;
	SparkKvCacheBlock *kv_blocks;
	uint32_t *kv_resident_slot_logical_block_indices;
	SparkKvPageCacheEntry *kv_entries;
	SparkKvPageCacheSequence *kv_sequences;
	uint32_t *kv_hash_bucket_heads;
	uint32_t *kv_entry_indices_by_logical_page;
	uint8_t *kv_page_staging;
	uint32_t *kv_lane_logical_pages;
	uint32_t *kv_lane_page_count;
	uint32_t *kv_lane_mutable_page;
	uint32_t *kv_lane_mutation_flags;
	SparkModelDriverCacheLane *kv_lane_cache_lanes;
	const char *kv_backing_directory;
	uint64_t kv_backing_maximum_bytes;
	char kv_backing_default[256];
	SparkGlm52ExecutionSlot slots[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	SparkGlm52AsyncCompletion completions[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	atomic_uint slot_states[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	atomic_uint lane_states[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	atomic_uchar lane_bound[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	atomic_ullong lane_sequence_ids[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	atomic_ullong lane_next_positions[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	atomic_ullong submitted_count;
	atomic_ullong completed_count;
	atomic_ullong rejected_count;
	atomic_ullong failed_count;
	atomic_ullong host_callback_completion_count;
	SparkTpDeviceCollective tp_device_collective;
	uint32_t tp_device_collective_initialized;
	SparkTpDeviceCollectiveCreditBinding tp_credit_bindings[SPARK_TP_DEVICE_COLLECTIVE_MAX_BINDING_COUNT];
	uint32_t tp_credit_binding_count;
	void *tp_credit_send_bf16;
	void *tp_credit_receive_bf16;
	void *tp_host_credit_send_bf16;
	void *tp_host_credit_receive_bf16;
	atomic_ullong tp_next_ordinal;
};

static uint32_t SparkGlm52BytesAreZero(const uint8_t *bytes,uint32_t count)
{
	uint32_t index;
	if ( bytes == 0 )
		return(1u);
	for (index=0u; index<count; index++)
		if ( bytes[index] != 0u )
			return(0u);
	return(1u);
}

static int32_t SparkGlm52ContractHash(uint8_t hash[SPARK_GLM52_STAGEPACK_SHA256_BYTES])
{
	const char *text;
	uint32_t index,high,low;
	text = GLM52_CONTRACT_SHA256;
	if ( strlen(text) != 2u * SPARK_GLM52_STAGEPACK_SHA256_BYTES )
		return(-1);
	for (index=0u; index<SPARK_GLM52_STAGEPACK_SHA256_BYTES; index++)
	{
		high = text[2u * index] >= '0' && text[2u * index] <= '9' ? (uint32_t)(text[2u * index] - '0') : text[2u * index] >= 'a' && text[2u * index] <= 'f' ? (uint32_t)(text[2u * index] - 'a' + 10) : UINT32_MAX;
		low = text[2u * index + 1u] >= '0' && text[2u * index + 1u] <= '9' ? (uint32_t)(text[2u * index + 1u] - '0') : text[2u * index + 1u] >= 'a' && text[2u * index + 1u] <= 'f' ? (uint32_t)(text[2u * index + 1u] - 'a' + 10) : UINT32_MAX;
		if ( high > 15u || low > 15u )
			return(-2);
		hash[index] = (uint8_t)((high << 4u) | low);
	}
	return(0);
}

static SparkStatus SparkGlm52ModuleConfigure(
	SparkGlm52ModuleState *state,
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services,
	const char **pack_path)
{
	const SparkGlm52ResidentDecodeStageNodeContext *context;
	if ( state == 0 || configuration == 0 || host_services == 0 || pack_path == 0 || host_services->node_context == 0 || host_services->execution_stream == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	context = (const SparkGlm52ResidentDecodeStageNodeContext *)host_services->node_context;
	if ( context->abi_version != SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION || context->descriptor_bytes != SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( context->stage_count != SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT || context->stage_index >= context->stage_count || context->first_layer_index != SparkGlm52ResidentDecodeStageFirstLayer(context->stage_index) || context->layer_count != SPARK_GLM52_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE || context->expert_weight_codec != GLM52_EXPERT_WEIGHT_CODEC || context->resident_sequence_capacity == 0u || context->resident_sequence_capacity > SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT || context->pipeline_slot_count == 0u || context->pipeline_slot_count > SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT || context->max_sequence_positions == 0u || context->max_sequence_positions > SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS || context->execution_row_capacity == 0u || context->execution_row_capacity > context->resident_sequence_capacity || context->tp_degree == 0u || context->tp_rank >= context->tp_degree || context->stage_pack_path == 0 || context->stage_pack_path[0] == '\0' || context->model_revision == 0 || context->model_revision[0] == '\0' || strlen(context->model_revision) >= sizeof(state->model_revision) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( SparkWeightCodecIsKnown(context->expert_weight_codec) == 0u || context->expert_weight_codec == SPARK_WEIGHT_CODEC_BF16 )
		return(SPARK_STATUS_UNSUPPORTED);
	if ( context->tp_degree != 1u && (SPARK_GLM52_MODEL_HEAD_COUNT % context->tp_degree != 0u || SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT % context->tp_degree != 0u || SPARK_GLM52_MODEL_DENSE_INTERMEDIATE_DIMENSION % context->tp_degree != 0u || SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION % context->tp_degree != 0u) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->model_revision == 0 || strcmp(configuration->model_revision,context->model_revision) != 0 )
		return(SPARK_STATUS_SCHEMA_ERROR);
	/* Lenient: existing serving configs may omit the backing fields. The
	 * page-store backing then falls back to a default path in
	 * SparkGlm52KvInitialize; a configured value is used as-is. */
	state->stage_index = context->stage_index;
	state->first_layer_index = context->first_layer_index;
	state->layer_count = context->layer_count;
	state->expert_weight_codec = context->expert_weight_codec;
	state->tp_degree = context->tp_degree;
	state->tp_rank = context->tp_rank;
	state->kv_backing_directory = context->kv_backing_directory;
	state->kv_backing_maximum_bytes = context->kv_backing_maximum_bytes;
	/* Identifier zero names a degraded single-rank bringup mode: the pack
	 * keeps its real tp geometry but no collective peers exist, so the chain
	 * runs with every reduce elided and the math is rank-local. Real
	 * deployments always set a non-zero identifier. */
	state->tp_collective_disabled = context->tp_collective_identifier == 0u ? 1u : 0u;
	state->resident_sequence_capacity = context->resident_sequence_capacity;
	state->pipeline_slot_count = context->pipeline_slot_count;
	state->max_sequence_positions = context->max_sequence_positions;
	state->execution_row_capacity = context->execution_row_capacity;
	state->owns_embedding = context->stage_index == 0u ? 1u : 0u;
	state->owns_final_head = context->stage_index + 1u == context->stage_count ? 1u : 0u;
	state->execution_stream = host_services->execution_stream;
	(void)snprintf(state->model_revision,sizeof(state->model_revision),"%s",context->model_revision);
	*pack_path = context->stage_pack_path;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52PackFileSize(FILE *file,uint64_t *bytes)
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

static uint32_t SparkGlm52PackRangesOverlap(const SparkGlm52PackRange *left,const SparkGlm52PackRange *right)
{
	return(left->bytes != 0u && right->bytes != 0u && left->offset < right->offset + right->bytes && right->offset < left->offset + left->bytes ? 1u : 0u);
}

static SparkStatus SparkGlm52PackValidateHeader(
	const SparkGlm52ModuleState *state,
	const SparkGlm52StagePackHeader *header,
	uint64_t file_bytes)
{
	uint8_t contract_sha256[SPARK_GLM52_STAGEPACK_SHA256_BYTES];
	uint64_t directory_bytes,directory_end;
	if ( state == 0 || header == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( header->magic != SPARK_GLM52_STAGEPACK_MAGIC || header->format_version != SPARK_GLM52_STAGEPACK_FORMAT_VERSION || header->header_bytes != SPARK_GLM52_STAGEPACK_HEADER_BYTES || header->directory_entry_bytes != SPARK_GLM52_STAGEPACK_ENTRY_BYTES || header->codec_abi_version != SPARK_WEIGHT_CODEC_ABI_VERSION )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( (header->flags & ~SPARK_GLM52_STAGEPACK_KNOWN_FLAGS) != 0u || (header->flags & SPARK_GLM52_STAGEPACK_FLAG_MTP) != 0u )
		return(SPARK_STATUS_UNSUPPORTED);
	if ( SparkGlm52StagePackHeaderTpDegree(header) == 0u || SparkGlm52StagePackHeaderTpDegree(header) != state->tp_degree || SparkGlm52StagePackHeaderTpRank(header) != state->tp_rank )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( header->tensor_count == 0u || header->tensor_count > SPARK_GLM52_STAGEPACK_MAX_TENSOR_COUNT || header->stage_count != SPARK_GLM52_RESIDENT_DECODE_STAGE_STAGE_COUNT || header->stage_index != state->stage_index || header->first_layer_index != state->first_layer_index || header->layer_count != state->layer_count || header->total_layer_count != SPARK_GLM52_MODEL_LAYER_COUNT || header->hidden_dimension != SPARK_GLM52_MODEL_HIDDEN_DIMENSION || header->vocab_count != SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT || header->routed_expert_count != SPARK_GLM52_MODEL_MOE_EXPERT_COUNT )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( header->linear_weight_codec != SPARK_WEIGHT_CODEC_BF16 || header->expert_weight_codec != state->expert_weight_codec || header->kv_cache_codec != SPARK_WEIGHT_CODEC_BF16 )
		return(SPARK_STATUS_TARGET_MISMATCH);
	if ( SparkGlm52ContractHash(contract_sha256) < 0 || header->model_revision[SPARK_GLM52_STAGEPACK_MODEL_REVISION_BYTES - 1u] != '\0' || strcmp(header->model_revision,state->model_revision) != 0 || memcmp(header->contract_sha256,contract_sha256,sizeof(contract_sha256)) != 0 || SparkGlm52BytesAreZero(header->source_config_sha256,sizeof(header->source_config_sha256)) != 0u || SparkGlm52BytesAreZero(header->pack_recipe_sha256,sizeof(header->pack_recipe_sha256)) != 0u )
		return(SPARK_STATUS_HASH_MISMATCH);
	if ( header->file_bytes != file_bytes || header->directory_offset < header->header_bytes || header->directory_offset % SPARK_GLM52_STAGEPACK_ALIGNMENT_BYTES != 0u || header->tensor_count > UINT64_MAX / header->directory_entry_bytes )
		return(SPARK_STATUS_SCHEMA_ERROR);
	directory_bytes = (uint64_t)header->tensor_count * header->directory_entry_bytes;
	directory_end = header->directory_offset + directory_bytes;
	if ( directory_end < header->directory_offset || directory_end > file_bytes )
		return(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52PackValidateEntryGeometry(
	const SparkGlm52ModuleState *state,
	const SparkGlm52StagePackHeader *header,
	const SparkGlm52StagePackEntry *entry,
	SparkGlm52StagePackTensorShape *shape)
{
	uint64_t payload_bytes,scale_bytes;
	uint32_t local_layer;
	if ( SparkGlm52StagePackExpectedShape(entry->tensor_kind,entry->layer_index,state->expert_weight_codec,state->tp_degree,shape) < 0 )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( entry->layer_index != SPARK_GLM52_STAGEPACK_GLOBAL_LAYER )
	{
		if ( entry->layer_index < state->first_layer_index || entry->layer_index >= state->first_layer_index + state->layer_count )
			return(SPARK_STATUS_SCHEMA_ERROR);
		local_layer = entry->layer_index - state->first_layer_index;
		if ( (state->layer_seen[local_layer] & (UINT64_C(1) << entry->tensor_kind)) != 0u )
			return(SPARK_STATUS_DUPLICATE);
	}
	else if ( (state->global_seen & (UINT64_C(1) << entry->tensor_kind)) != 0u )
		return(SPARK_STATUS_DUPLICATE);
	if ( entry->payload_type != shape->payload_type || entry->weight_codec != shape->weight_codec || entry->scale_encoding != shape->scale_encoding || entry->group_count != shape->group_count || entry->rows != shape->rows || entry->columns != shape->columns )
		return(SPARK_STATUS_SCHEMA_ERROR);
	payload_bytes = SparkGlm52StagePackExpectedPayloadBytes(shape);
	scale_bytes = SparkGlm52StagePackExpectedScaleBytes(shape);
	if ( payload_bytes == 0u || entry->payload_bytes != payload_bytes || entry->scale_bytes != scale_bytes )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( entry->payload_offset % SPARK_GLM52_STAGEPACK_ALIGNMENT_BYTES != 0u || entry->payload_offset < header->directory_offset + ((uint64_t)header->tensor_count * header->directory_entry_bytes) || entry->payload_offset > header->file_bytes || entry->payload_bytes > header->file_bytes - entry->payload_offset )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( scale_bytes == 0u )
	{
		if ( entry->scale_offset != 0u )
			return(SPARK_STATUS_SCHEMA_ERROR);
	}
	else if ( entry->scale_offset % SPARK_GLM52_STAGEPACK_ALIGNMENT_BYTES != 0u || entry->scale_offset < header->directory_offset + ((uint64_t)header->tensor_count * header->directory_entry_bytes) || entry->scale_offset > header->file_bytes || entry->scale_bytes > header->file_bytes - entry->scale_offset )
		return(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52PackValidateRanges(
	const SparkGlm52StagePackEntry *entries,
	uint32_t entry_count)
{
	SparkGlm52PackRange left[2],right[2];
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
					if ( SparkGlm52PackRangesOverlap(&left[left_part],&right[right_part]) != 0u )
						return(SPARK_STATUS_SCHEMA_ERROR);
		}
		if ( SparkGlm52PackRangesOverlap(&left[0],&left[1]) != 0u )
			return(SPARK_STATUS_SCHEMA_ERROR);
	}
	return(SPARK_STATUS_OK);
}

static void SparkGlm52PackMarkSeen(
	SparkGlm52ModuleState *state,
	const SparkGlm52StagePackEntry *entry)
{
	if ( entry->layer_index == SPARK_GLM52_STAGEPACK_GLOBAL_LAYER )
		state->global_seen |= UINT64_C(1) << entry->tensor_kind;
	else
		state->layer_seen[entry->layer_index - state->first_layer_index] |= UINT64_C(1) << entry->tensor_kind;
}

static SparkStatus SparkGlm52PackAssignLayer(
	SparkGlm52LayerWeights *weights,
	uint32_t tensor_kind,
	const void *payload,
	const void *scale)
{
	switch ( tensor_kind )
	{
	case SPARK_GLM52_STAGEPACK_TENSOR_ATTN_NORM: weights->attn_norm_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_Q_A: weights->q_a_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_Q_A_NORM: weights->q_a_norm_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_Q_B: weights->q_b_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_KV_A: weights->kv_a_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_KV_A_NORM: weights->kv_a_norm_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_KV_B_KEY_TRANSPOSED: weights->kv_b_key_transposed_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_KV_B_VALUE: weights->kv_b_value_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_ATTN_OUTPUT: weights->attn_output_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_POST_ATTN_NORM: weights->post_attn_norm_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_INDEX_Q: weights->index_q_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_INDEX_K: weights->index_k_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_INDEX_HEAD: weights->index_head_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_INDEX_NORM_WEIGHT: weights->index_norm_weight_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_INDEX_NORM_BIAS: weights->index_norm_bias_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_DENSE_GATE_UP: weights->dense_gate_up_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_DENSE_DOWN: weights->dense_down_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_ROUTER: weights->router_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_ROUTER_CORRECTION: weights->router_correction_f32 = (const float *)payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_UP_GATE: weights->expert_up_gate_payload = payload; weights->expert_up_gate_scale = scale; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_EXPERT_DOWN: weights->expert_down_payload = payload; weights->expert_down_scale = scale; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_SHARED_GATE_UP: weights->shared_gate_up_bf16 = payload; break;
	case SPARK_GLM52_STAGEPACK_TENSOR_SHARED_DOWN: weights->shared_down_bf16 = payload; break;
	default: return(SPARK_STATUS_SCHEMA_ERROR);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52PackAssign(
	SparkGlm52ModuleState *state,
	const SparkGlm52StagePackEntry *entry,
	const void *payload,
	const void *scale)
{
	if ( entry->layer_index != SPARK_GLM52_STAGEPACK_GLOBAL_LAYER )
		return(SparkGlm52PackAssignLayer(&state->layers[entry->layer_index - state->first_layer_index],entry->tensor_kind,payload,scale));
	switch ( entry->tensor_kind )
	{
	case SPARK_GLM52_STAGEPACK_TENSOR_EMBEDDING: state->embedding_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_GLM52_STAGEPACK_TENSOR_FINAL_NORM: state->final_norm_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_GLM52_STAGEPACK_TENSOR_LM_HEAD: state->lm_head_bf16 = payload; return(SPARK_STATUS_OK);
	default: return(SPARK_STATUS_SCHEMA_ERROR);
	}
}

static SparkStatus SparkGlm52PackLoadEntry(
	SparkGlm52ModuleState *state,
	FILE *file,
	const SparkGlm52StagePackEntry *entry)
{
	void *payload,*scale;
	SparkStatus status;
	payload = 0;
	scale = 0;
	status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,entry->payload_offset,entry->payload_bytes,&payload);
	if ( status == SPARK_STATUS_OK && entry->scale_bytes != 0u )
		status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,entry->scale_offset,entry->scale_bytes,&scale);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52PackAssign(state,entry,payload,scale);
	return(status);
}

static uint64_t SparkGlm52ExpectedLayerMask(
	const SparkGlm52ModuleState *state,
	uint32_t layer_index)
{
	SparkGlm52StagePackTensorShape shape;
	uint64_t mask;
	uint32_t kind;
	mask = 0u;
	for (kind=SPARK_GLM52_STAGEPACK_TENSOR_ATTN_NORM; kind<SPARK_GLM52_STAGEPACK_TENSOR_KIND_COUNT; kind++)
		if ( SparkGlm52StagePackExpectedShape(kind,layer_index,state->expert_weight_codec,state->tp_degree,&shape) == 0 )
			mask |= UINT64_C(1) << kind;
	return(mask);
}

static uint64_t SparkGlm52ExpectedGlobalMask(const SparkGlm52ModuleState *state)
{
	uint64_t mask;
	mask = 0u;
	if ( state->owns_embedding != 0u )
		mask |= UINT64_C(1) << SPARK_GLM52_STAGEPACK_TENSOR_EMBEDDING;
	if ( state->owns_final_head != 0u )
		mask |= (UINT64_C(1) << SPARK_GLM52_STAGEPACK_TENSOR_FINAL_NORM) | (UINT64_C(1) << SPARK_GLM52_STAGEPACK_TENSOR_LM_HEAD);
	return(mask);
}

static SparkStatus SparkGlm52PackValidateInventory(const SparkGlm52ModuleState *state)
{
	uint32_t local;
	if ( state->global_seen != SparkGlm52ExpectedGlobalMask(state) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	for (local=0u; local<state->layer_count; local++)
		if ( state->layer_seen[local] != SparkGlm52ExpectedLayerMask(state,state->first_layer_index + local) )
			return(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52PackLoad(
	SparkGlm52ModuleState *state,
	const char *path)
{
	SparkGlm52StagePackHeader header;
	SparkGlm52StagePackEntry entries[SPARK_GLM52_STAGEPACK_MAX_TENSOR_COUNT];
	SparkGlm52StagePackTensorShape shape;
	FILE *file;
	uint64_t file_bytes;
	uint32_t index;
	SparkStatus status;
	file = fopen(path,"rb");
	if ( file == 0 )
		return(SPARK_STATUS_NOT_FOUND);
	memset(&header,0,sizeof(header));
	memset(entries,0,sizeof(entries));
	status = SparkGlm52PackFileSize(file,&file_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModulePackRead(SPARK_GLM52_MODULE_TAG,file,0u,&header,sizeof(header));
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52PackValidateHeader(state,&header,file_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModulePackRead(SPARK_GLM52_MODULE_TAG,file,header.directory_offset,entries,(uint64_t)header.tensor_count * sizeof(entries[0]));
	for (index=0u; status==SPARK_STATUS_OK && index<header.tensor_count; index++)
	{
		status = SparkGlm52PackValidateEntryGeometry(state,&header,&entries[index],&shape);
		if ( status == SPARK_STATUS_OK )
			SparkGlm52PackMarkSeen(state,&entries[index]);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52PackValidateRanges(entries,header.tensor_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52PackValidateInventory(state);
	for (index=0u; status==SPARK_STATUS_OK && index<header.tensor_count; index++)
		status = SparkGlm52PackLoadEntry(state,file,&entries[index]);
	if ( fclose(file) != 0 && status == SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	return(status);
}

static SparkStatus SparkGlm52AllocateBytes(
	SparkGlm52ModuleState *state,
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

static SparkStatus SparkGlm52AllocateRows(
	SparkGlm52ModuleState *state,
	uint64_t rows,
	uint64_t columns,
	void **pointer)
{
	return(SparkGlm52AllocateBytes(state,rows,columns,sizeof(uint16_t),pointer));
}

static SparkStatus SparkGlm52AllocateSlotHost(SparkGlm52ExecutionSlot *slot)
{
	uint32_t *cursor;
	uint64_t rows,words,bytes;
	cudaError_t error;
	if ( slot == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	rows = SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT;
	words = (rows * 4u) + SPARK_GLM52_KV_ACCESS_ERROR_WORD_COUNT;
	bytes = words * sizeof(uint32_t);
	error = cudaHostAlloc(&slot->host_staging,bytes,cudaHostAllocPortable);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_GLM52_MODULE_TAG,error,"host_staging"));
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
	return(SPARK_STATUS_OK);
}

static void SparkGlm52ReleaseSlotHost(SparkGlm52ModuleState *state)
{
	uint32_t index;
	if ( state == 0 )
		return;
	for (index=0u; index<state->pipeline_slot_count; index++)
	{
		if ( state->slots[index].host_staging != 0 )
			(void)cudaFreeHost(state->slots[index].host_staging);
		state->slots[index].host_staging = 0;
	}
}

static SparkStatus SparkGlm52AllocateSlotMetadata(
	SparkGlm52ModuleState *state,
	SparkGlm52ExecutionSlot *slot)
{
	SparkStatus status;
	status = SparkGlm52AllocateBytes(state,state->execution_row_capacity,1u,sizeof(uint32_t),(void **)&slot->token_ids);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateBytes(state,state->execution_row_capacity,1u,sizeof(uint32_t),(void **)&slot->resident_slots);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateBytes(state,state->execution_row_capacity,1u,sizeof(uint32_t),(void **)&slot->positions);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateBytes(state,state->resident_sequence_capacity,1u,sizeof(uint32_t),(void **)&slot->context_lengths);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateBytes(state,2u,1u,sizeof(uint32_t),(void **)&slot->dense_row_offset);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateBytes(state,2u,1u,sizeof(uint32_t),(void **)&slot->dense_tile_prefix);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateBytes(state,1u,sizeof(uint32_t) * 6u,1u,&slot->kv_access_error);
	return(status);
}

static SparkStatus SparkGlm52AllocateSlotHidden(
	SparkGlm52ModuleState *state,
	SparkGlm52ExecutionSlot *slot)
{
	uint64_t rows;
	SparkStatus status;
	rows = state->execution_row_capacity;
	status = SparkGlm52AllocateRows(state,rows,SPARK_GLM52_MODEL_HIDDEN_DIMENSION,(void **)&slot->hidden_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateRows(state,rows,SPARK_GLM52_MODEL_HIDDEN_DIMENSION,(void **)&slot->residual_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateRows(state,rows,SPARK_GLM52_MODEL_HIDDEN_DIMENSION,(void **)&slot->normed_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateRows(state,rows,SPARK_GLM52_MODEL_QUERY_A_DIMENSION,(void **)&slot->q_compressed_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateRows(state,rows,SPARK_GLM52_MODEL_QUERY_B_DIMENSION,(void **)&slot->q_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateRows(state,rows,SPARK_GLM52_MODEL_HEAD_COUNT * SPARK_GLM52_MODEL_LATENT_DIMENSION,(void **)&slot->query_latent_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateRows(state,rows,SPARK_GLM52_MODEL_HEAD_COUNT * SPARK_GLM52_MODEL_ROPE_DIMENSION,(void **)&slot->query_rope_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateRows(state,rows,SPARK_GLM52_MODEL_DSA_INDEX_QUERY_DIMENSION,(void **)&slot->index_query_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateRows(state,rows,SPARK_GLM52_MODEL_DSA_INDEX_HEAD_DIMENSION,(void **)&slot->index_key_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateRows(state,rows,SPARK_GLM52_MODEL_DSA_INDEX_HEAD_COUNT,(void **)&slot->index_head_weight_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateRows(state,rows,SPARK_GLM52_MODEL_CACHE_TOKEN_ELEMENTS,(void **)&slot->kv_slot_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateRows(state,rows,SPARK_GLM52_MODEL_HEAD_COUNT * SPARK_GLM52_MODEL_LATENT_DIMENSION,(void **)&slot->attention_latent_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateRows(state,rows,SPARK_GLM52_MODEL_HEAD_COUNT * SPARK_GLM52_MODEL_VALUE_HEAD_DIMENSION,(void **)&slot->attention_value_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateRows(state,rows,SPARK_GLM52_MODEL_HIDDEN_DIMENSION,(void **)&slot->attention_out_bf16);
	return(status);
}

static SparkStatus SparkGlm52AllocateSlotMlp(
	SparkGlm52ModuleState *state,
	SparkGlm52ExecutionSlot *slot)
{
	uint64_t rows,packed_rows;
	SparkStatus status;
	rows = state->execution_row_capacity;
	packed_rows = rows * SPARK_GLM52_MODEL_MOE_TOP_K;
	status = SparkGlm52AllocateRows(state,rows,SPARK_GLM52_MODEL_MOE_ROUTED_GATE_UP_DIMENSION,(void **)&slot->gate_up_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateRows(state,rows,SPARK_GLM52_MODEL_MOE_TOP_K * SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION,(void **)&slot->intermediate_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateRows(state,packed_rows,SPARK_GLM52_MODEL_HIDDEN_DIMENSION,(void **)&slot->expert_out_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateRows(state,rows,SPARK_GLM52_MODEL_HIDDEN_DIMENSION,(void **)&slot->shared_out_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateBytes(state,rows,SPARK_GLM52_MODEL_MOE_EXPERT_COUNT,sizeof(float),(void **)&slot->router_logits_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateBytes(state,rows,state->max_sequence_positions,sizeof(float),(void **)&slot->selection_scores_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateBytes(state,rows,SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT,sizeof(uint32_t),(void **)&slot->selected_positions);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateBytes(state,packed_rows,1u,sizeof(uint32_t),(void **)&slot->route_expert);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateBytes(state,packed_rows,1u,sizeof(float),(void **)&slot->route_weight);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateBytes(state,packed_rows,1u,sizeof(uint32_t),(void **)&slot->route_source_token);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateBytes(state,packed_rows,1u,sizeof(uint32_t),(void **)&slot->route_packed_row);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateBytes(state,SPARK_GLM52_MODEL_MOE_EXPERT_COUNT + 1u,1u,sizeof(uint32_t),(void **)&slot->group_row_offset);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateBytes(state,SPARK_GLM52_MODEL_MOE_EXPERT_COUNT + 1u,1u,sizeof(uint32_t),(void **)&slot->group_tile_prefix_w1);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateBytes(state,SPARK_GLM52_MODEL_MOE_EXPERT_COUNT + 1u,1u,sizeof(uint32_t),(void **)&slot->group_tile_prefix_w2);
	return(status);
}

static SparkStatus SparkGlm52AllocateSlotHead(
	SparkGlm52ModuleState *state,
	SparkGlm52ExecutionSlot *slot)
{
	uint64_t rows,tiles;
	SparkStatus status;
	rows = state->execution_row_capacity;
	tiles = SparkCeilDivU64(SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT,SPARK_GLM52_HEAD_TILE);
	status = SparkGlm52AllocateBytes(state,rows,tiles,sizeof(float),(void **)&slot->head_candidate_score);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateBytes(state,rows,tiles,sizeof(uint32_t),(void **)&slot->head_candidate_token);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateBytes(state,rows,1u,sizeof(uint32_t),(void **)&slot->output_token);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateBytes(state,rows,1u,sizeof(float),(void **)&slot->output_score);
	if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateBytes(state,rows,1u,sizeof(uint64_t),(void **)&slot->head_maxloc_u64);
	return(status);
}

static SparkStatus SparkGlm52AllocateSlots(SparkGlm52ModuleState *state)
{
	uint32_t index;
	SparkStatus status;
	status = SPARK_STATUS_OK;
	for (index=0u; status==SPARK_STATUS_OK && index<state->pipeline_slot_count; index++)
	{
		state->slots[index].stream = state->execution_stream;
		status = SparkGlm52AllocateSlotHost(&state->slots[index]);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateSlotMetadata(state,&state->slots[index]);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateSlotHidden(state,&state->slots[index]);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateSlotMlp(state,&state->slots[index]);
		if ( status == SPARK_STATUS_OK ) status = SparkGlm52AllocateSlotHead(state,&state->slots[index]);
	}
	return(status);
}

static SparkStatus SparkGlm52BuildPageTable(SparkGlm52ModuleState *state)
{
	uint32_t *host_table;
	uint64_t entries,index;
	SparkStatus status;
	cudaError_t error;
	state->pages_per_sequence = SparkCeilDivU32(state->max_sequence_positions,64u);
	if ( state->pages_per_sequence == 0u || state->resident_sequence_capacity > UINT32_MAX / state->pages_per_sequence )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->page_count = state->resident_sequence_capacity * state->pages_per_sequence;
	entries = state->page_count;
	host_table = (uint32_t *)malloc(entries * sizeof(uint32_t));
	if ( host_table == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (index=0u; index<entries; index++)
		host_table[index] = (uint32_t)index;
	status = SparkGlm52AllocateBytes(state,entries,1u,sizeof(uint32_t),(void **)&state->page_table);
	if ( status == SPARK_STATUS_OK )
	{
		error = cudaMemcpy(state->page_table,host_table,entries * sizeof(uint32_t),cudaMemcpyHostToDevice);
		status = SparkStageModuleCudaStatus(SPARK_GLM52_MODULE_TAG,error,"page_table");
	}
	free(host_table);
	return(status);
}

static SparkStatus SparkGlm52PageCopy(
	void *context,
	uint32_t direction,
	uintptr_t device_address,
	void *host_address,
	uint64_t bytes)
{
	SparkGlm52ModuleState *state;
	cudaError_t error;
	state = (SparkGlm52ModuleState *)context;
	if ( state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	/* The page-store worker treats this callback as complete when it
	 * returns and immediately performs NVMe I/O or reuses its single
	 * staging buffer; an async copy on the execution stream would still
	 * be in flight and corrupt both directions. Complete the copy here. */
	if ( direction == SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST )
		error = cudaMemcpy(host_address,(const void *)device_address,(size_t)bytes,cudaMemcpyDeviceToHost);
	else if ( direction == SPARK_KV_PAGE_STORE_COPY_HOST_TO_DEVICE )
		error = cudaMemcpy((void *)device_address,host_address,(size_t)bytes,cudaMemcpyHostToDevice);
	else
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkStageModuleCudaStatus(SPARK_GLM52_MODULE_TAG,error,"kv_page_copy"));
}

static SparkStatus SparkGlm52KvInitialize(SparkGlm52ModuleState *state)
{
	SparkKvModelTable table;
	uint64_t block_bytes;
	uint64_t lane_page_entries;
	SparkStatus status;
	block_bytes = (uint64_t)SPARK_GLM52_KV_BLOCK_TOKEN_COUNT *
		(uint64_t)state->layer_count * SPARK_GLM52_KV_ARENA_HEAD_DIM *
		SPARK_GLM52_KV_BYTES_PER_SCALAR;
	lane_page_entries = (uint64_t)state->resident_sequence_capacity *
		state->pages_per_sequence;
	state->kv_blocks = (SparkKvCacheBlock *)calloc(state->page_count,sizeof(*state->kv_blocks));
	state->kv_resident_slot_logical_block_indices = (uint32_t *)calloc(state->page_count,sizeof(*state->kv_resident_slot_logical_block_indices));
	state->kv_entries = (SparkKvPageCacheEntry *)calloc(state->page_count,sizeof(*state->kv_entries));
	state->kv_sequences = (SparkKvPageCacheSequence *)calloc(state->resident_sequence_capacity,sizeof(*state->kv_sequences));
	state->kv_hash_bucket_heads = (uint32_t *)calloc(state->page_count,sizeof(*state->kv_hash_bucket_heads));
	state->kv_entry_indices_by_logical_page = (uint32_t *)calloc(state->page_count,sizeof(*state->kv_entry_indices_by_logical_page));
	state->kv_page_staging = (uint8_t *)malloc((size_t)block_bytes);
	state->kv_lane_logical_pages = (uint32_t *)calloc((size_t)lane_page_entries,sizeof(*state->kv_lane_logical_pages));
	state->kv_lane_page_count = (uint32_t *)calloc(state->resident_sequence_capacity,sizeof(*state->kv_lane_page_count));
	state->kv_lane_mutable_page = (uint32_t *)calloc(state->resident_sequence_capacity,sizeof(*state->kv_lane_mutable_page));
	state->kv_lane_mutation_flags = (uint32_t *)calloc(state->resident_sequence_capacity,sizeof(*state->kv_lane_mutation_flags));
	state->kv_lane_cache_lanes = (SparkModelDriverCacheLane *)calloc(state->resident_sequence_capacity,sizeof(*state->kv_lane_cache_lanes));
	if ( state->kv_blocks == 0 || state->kv_resident_slot_logical_block_indices == 0 || state->kv_entries == 0 || state->kv_sequences == 0 || state->kv_hash_bucket_heads == 0 || state->kv_entry_indices_by_logical_page == 0 || state->kv_page_staging == 0 || state->kv_lane_logical_pages == 0 || state->kv_lane_page_count == 0 || state->kv_lane_mutable_page == 0 || state->kv_lane_mutation_flags == 0 || state->kv_lane_cache_lanes == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);

	memset(&table,0,sizeof(table));
	table.abi_version = SPARK_KV_MODEL_TABLE_ABI_VERSION;
	table.descriptor_bytes = SPARK_KV_MODEL_TABLE_BYTES;
	SparkGlm52KvFillCapacityRequest(&table.capacity_request);

	table.arena_configuration.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	table.arena_configuration.descriptor_bytes = SPARK_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
	table.arena_configuration.logical_block_count = state->page_count;
	table.arena_configuration.block_token_count = SPARK_GLM52_KV_BLOCK_TOKEN_COUNT;
	table.arena_configuration.resident_block_capacity = state->page_count;
	table.arena_configuration.layer_count = state->layer_count;
	table.arena_configuration.kv_head_count = SPARK_GLM52_KV_ARENA_KV_HEAD_COUNT;
	table.arena_configuration.head_dim = SPARK_GLM52_KV_ARENA_HEAD_DIM;
	table.arena_configuration.bytes_per_scalar = SPARK_GLM52_KV_BYTES_PER_SCALAR;
	table.arena_configuration.key_device_base = state->kv_cache;
	table.arena_configuration.blocks = state->kv_blocks;
	table.arena_configuration.resident_slot_logical_block_indices = state->kv_resident_slot_logical_block_indices;

	table.page_store_config.abi_version = SPARK_KV_PAGE_STORE_ABI_VERSION;
	table.page_store_config.descriptor_bytes = SPARK_KV_PAGE_STORE_CONFIGURATION_BYTES;
	table.page_store_config.flags = SPARK_KV_PAGE_STORE_FLAG_ANONYMOUS;
	table.page_store_config.logical_page_capacity = state->page_count;
	table.page_store_config.transfer_capacity = 2u;
	table.page_store_config.page_bytes = block_bytes;
	if ( state->kv_backing_directory != 0 && state->kv_backing_directory[0] != '\0' )
		table.page_store_config.backing_path = state->kv_backing_directory;
	else
	{
		/* Fallback for serving configs that predate the backing fields: keep
		 * the store functional with a well-known default (the page store
		 * opens the path once; it does not retain the pointer). */
		(void)snprintf(state->kv_backing_default,sizeof(state->kv_backing_default),
			"/tmp/sparkpipe_glm52_kv_%s",state->model_revision);
		table.page_store_config.backing_path = state->kv_backing_default;
	}
	table.page_store_config.maximum_backing_bytes =
		state->kv_backing_maximum_bytes >= block_bytes
			? state->kv_backing_maximum_bytes
			: block_bytes;
	table.page_store_config.staging_address = state->kv_page_staging;
	table.page_store_config.staging_bytes = block_bytes;
	table.page_store_config.copy_function = SparkGlm52PageCopy;
	table.page_store_config.copy_context = state;

	table.sequence_capacity = state->resident_sequence_capacity;
	table.entry_capacity = state->page_count;
	table.hash_bucket_count = state->page_count;
	table.entries = state->kv_entries;
	table.sequences = state->kv_sequences;
	table.hash_bucket_heads = state->kv_hash_bucket_heads;
	table.entry_indices_by_logical_page = state->kv_entry_indices_by_logical_page;
	table.model_id = "glm52";
	table.model_revision = state->model_revision;
	table.cache_layout_fingerprint = "compressed-key-value-bf16-block-major";

	status = SparkKvBackendInitialize(&table,&state->kv_arena,&state->kv_page_cache,&state->kv_page_store);
	return(status);
}

static SparkStatus SparkGlm52AllocateCaches(SparkGlm52ModuleState *state)
{
	uint64_t main_page_bytes,index_page_bytes;
	uint64_t main_total,index_total;
	uint32_t local;
	SparkStatus status;
	state->index_layer_count = 0u;
	for (local=0u; local<state->layer_count; local++)
	{
		state->index_ordinal_by_local_layer[local] = SPARK_GLM52_NO_INDEX_ORDINAL;
		if ( SparkGlm52StagePackLayerHasFullIndexer(state->first_layer_index + local) != 0u )
			state->index_ordinal_by_local_layer[local] = state->index_layer_count++;
	}
	status = SparkGlm52BuildPageTable(state);
	main_page_bytes = 64u * (uint64_t)SPARK_GLM52_MODEL_CACHE_TOKEN_ELEMENTS * sizeof(uint16_t);
	index_page_bytes = 64u * (uint64_t)SPARK_GLM52_MODEL_DSA_INDEX_HEAD_DIMENSION * sizeof(uint16_t);
	/* Block-major pool: one 64-token block holds all layers contiguously; the
	 * per-layer pool base the kernel receives is kv_cache + layer *
	 * main_page_bytes, and the block stride (main_page_bytes * layer_count) is
	 * the shared arena's default. */
	state->kv_layer_stride_bytes = main_page_bytes;
	state->index_layer_stride_bytes = (uint64_t)state->page_count * index_page_bytes;
	if ( status != SPARK_STATUS_OK || state->kv_layer_stride_bytes == 0u || state->page_count > UINT64_MAX / (main_page_bytes * state->layer_count) )
		return(status == SPARK_STATUS_OK ? SPARK_STATUS_CAPACITY_EXCEEDED : status);
	main_total = (uint64_t)state->page_count * main_page_bytes * state->layer_count;
	status = SparkStageModuleDeviceAllocate(&state->ledger,main_total,(void **)&state->kv_cache);
	if ( status == SPARK_STATUS_OK && state->index_layer_count != 0u )
	{
		if ( state->index_layer_stride_bytes > UINT64_MAX / state->index_layer_count )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		index_total = state->index_layer_stride_bytes * state->index_layer_count;
		status = SparkStageModuleDeviceAllocate(&state->ledger,index_total,(void **)&state->index_cache);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52KvInitialize(state);
	return(status);
}

static SparkStatus SparkGlm52AdmissionPredicate(
	void *context,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	SparkGlm52ModuleState *state;
	const SparkModelDriverCacheLane *lane;
	uint32_t lane_index;
	uint32_t mutation_flags;
	SparkStatus status;
	state = (SparkGlm52ModuleState *)context;
	if ( state == 0 || request == 0 || decision == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	/* Remember each lane's full identity for the completion tail (CompleteLane
	 * / RollbackLaneTransaction run after the kernel writes KV, keyed by
	 * resident slot). */
	for (lane_index=0u; lane_index<request->cache_lane_count; lane_index++)
	{
		lane = &request->cache_lanes[lane_index];
		if ( lane->resident_sequence_slot < state->resident_sequence_capacity )
			state->kv_lane_cache_lanes[lane->resident_sequence_slot] = *lane;
	}
	if ( (request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE) != 0u )
	{
		for (lane_index=0u; lane_index<request->cache_lane_count; lane_index++)
		{
			lane = &request->cache_lanes[lane_index];
			status = SparkKvPageCacheReleaseLane(&state->kv_page_cache,lane->resident_sequence_slot,lane->sequence_id);
			if ( status != SPARK_STATUS_OK )
				return(status);
		}
		decision->accepted = 1u;
		decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
		return(SPARK_STATUS_OK);
	}
	if ( (request->admission_flags & SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE) != 0u )
	{
		for (lane_index=0u; lane_index<request->cache_lane_count; lane_index++)
		{
			lane = &request->cache_lanes[lane_index];
			status = SparkKvPageCachePrepareLane(&state->kv_page_cache,lane,state->kv_lane_logical_pages + (uint64_t)lane->resident_sequence_slot * state->pages_per_sequence,state->pages_per_sequence,&state->kv_lane_page_count[lane->resident_sequence_slot]);
			if ( status != SPARK_STATUS_OK )
				return(status);
		}
	}
	if ( (request->admission_flags & SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT) != 0u )
	{
		for (lane_index=0u; lane_index<request->cache_lane_count; lane_index++)
		{
			lane = &request->cache_lanes[lane_index];
			status = SparkKvPageCacheBeginLaneTransaction(&state->kv_page_cache,lane,&state->kv_lane_mutable_page[lane->resident_sequence_slot],&mutation_flags);
			state->kv_lane_mutation_flags[lane->resident_sequence_slot] = mutation_flags;
			if ( status != SPARK_STATUS_OK )
				return(status);
		}
	}
	if ( (request->admission_flags & SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT) != 0u )
	{
		for (lane_index=0u; lane_index<request->cache_lane_count; lane_index++)
		{
			lane = &request->cache_lanes[lane_index];
			status = SparkKvPageCacheRollbackLaneTransaction(&state->kv_page_cache,lane,state->kv_lane_mutation_flags[lane->resident_sequence_slot]);
			if ( status != SPARK_STATUS_OK )
				return(status);
		}
	}
	return(SPARK_STATUS_OK);
}

static uint32_t SparkGlm52RoundMajorWaveRows(
	const SparkGlm52ResidentDecodeStageBatchView *batch,
	uint32_t first_row)
{
	uint32_t lane,current,count,next;
	if ( batch == 0 || first_row >= batch->row_count || batch->active_sequence_count == 0u )
		return(0u);
	current = batch->active_sequence_count;
	for (lane=0u; lane<batch->active_sequence_count; lane++)
		if ( batch->row_resident_slots[lane] == batch->row_resident_slots[first_row] )
			current = lane;
	if ( current == batch->active_sequence_count )
		return(0u);
	count = 1u;
	while ( first_row + count < batch->row_count )
	{
		next = batch->active_sequence_count;
		for (lane=0u; lane<batch->active_sequence_count; lane++)
			if ( batch->row_resident_slots[lane] == batch->row_resident_slots[first_row + count] )
				next = lane;
		if ( next == batch->active_sequence_count || next <= current )
			break;
		current = next;
		count++;
	}
	return(count);
}

static SparkStatus SparkGlm52ValidateRoundMajor(
	const SparkGlm52ModuleState *state,
	const SparkGlm52ResidentDecodeStageBatchView *batch)
{
	uint32_t counts[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT] = {0u};
	uint32_t lane,row,index,maximum,wave;
	if ( batch->row_count < batch->active_sequence_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (lane=0u; lane<batch->active_sequence_count; lane++)
	{
		if ( batch->row_resident_slots[lane] >= state->resident_sequence_capacity )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		for (index=0u; index<lane; index++)
			if ( batch->row_resident_slots[index] == batch->row_resident_slots[lane] )
				return(SPARK_STATUS_DUPLICATE);
	}
	maximum = 0u;
	for (row=0u; row<batch->row_count; row++)
	{
		for (lane=0u; lane<batch->active_sequence_count && batch->row_resident_slots[lane]!=batch->row_resident_slots[row]; lane++)
			;
		if ( lane == batch->active_sequence_count )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		counts[lane]++;
		if ( counts[lane] > maximum )
			maximum = counts[lane];
	}
	row = 0u;
	for (wave=0u; wave<maximum; wave++)
		for (lane=0u; lane<batch->active_sequence_count; lane++)
			if ( counts[lane] > wave && (row >= batch->row_count || batch->row_resident_slots[row++] != batch->row_resident_slots[lane]) )
				return(SPARK_STATUS_INVALID_ARGUMENT);
	return(row == batch->row_count ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
}

static SparkStatus SparkGlm52ValidateSequenceContinuity(
	const SparkGlm52ModuleState *state,
	const SparkGlm52ResidentDecodeStageBatchView *batch,
	uint8_t *bound,
	uint64_t *sequence_ids,
	uint64_t *next_positions)
{
	uint8_t touched[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT] = {0u};
	uint64_t position,sequence;
	uint32_t lane,row,slot;
	SparkStatus status;
	for (lane=0u; lane<batch->active_sequence_count; lane++)
	{
		slot = batch->row_resident_slots[lane];
		if ( slot >= state->resident_sequence_capacity )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		bound[lane] = atomic_load_explicit(&state->lane_bound[slot],memory_order_acquire);
		sequence_ids[lane] = atomic_load_explicit(&state->lane_sequence_ids[slot],memory_order_acquire);
		next_positions[lane] = atomic_load_explicit(&state->lane_next_positions[slot],memory_order_acquire);
	}
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

typedef struct SparkGlm52ClaimedContinuityContext
{
	const SparkGlm52ModuleState *state;
	const SparkGlm52ResidentDecodeStageBatchView *batch;
	uint8_t *bound;
	uint64_t *sequence_ids;
	uint64_t *next_positions;
} SparkGlm52ClaimedContinuityContext;

static SparkStatus SparkGlm52PrepareClaimedContinuity(void *prepare_context)
{
	SparkGlm52ClaimedContinuityContext *context;
	context = (SparkGlm52ClaimedContinuityContext *)prepare_context;
	return(SparkGlm52ValidateSequenceContinuity(context->state,context->batch,context->bound,context->sequence_ids,context->next_positions));
}

static SparkStatus SparkGlm52ValidateFrameBuffers(
	const SparkGlm52ModuleState *state,
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
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlm52ValidateFrame(
	const SparkGlm52ModuleState *state,
	const SparkModelDriverFrame *frame,
	const SparkGlm52ResidentDecodeStageFrameContext **context_out)
{
	const SparkGlm52ResidentDecodeStageFrameContext *context;
	const SparkGlm52ResidentDecodeStageBatchView *batch;
	uint32_t expected_flags,prefill;
	uint64_t boundary_bytes,sideband_bytes;
	SparkStatus status;
	if ( state == 0 || frame == 0 || context_out == 0 || frame->user_context == 0 || frame->execution_stream != state->execution_stream || frame->completion_function == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	context = (const SparkGlm52ResidentDecodeStageFrameContext *)frame->user_context;
	if ( context->abi_version != SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION || context->descriptor_bytes != sizeof(*context) || context->reserved0 != 0u || (context->flags & ~SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_KNOWN_FLAGS) != 0u || context->batch == 0 )
		return(SPARK_STATUS_ABI_MISMATCH);
	batch = context->batch;
	if ( batch->abi_version != SPARK_GLM52_RESIDENT_DECODE_STAGE_BATCH_VIEW_ABI_VERSION || batch->descriptor_bytes != sizeof(*batch) || batch->row_count == 0u || batch->row_count > SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT || batch->active_sequence_count == 0u || batch->active_sequence_count > state->resident_sequence_capacity || batch->row_resident_slots == 0 || batch->row_positions == 0 || batch->row_sequence_ids == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	prefill = (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u ? 1u : 0u;
	if ( prefill == 0u && batch->row_count != batch->active_sequence_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( frame->active_slot_count != batch->active_sequence_count || frame->new_token_count != batch->row_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->owns_embedding != 0u && batch->token_ids == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	expected_flags = prefill != 0u ? SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_PREFILL : 0u;
	expected_flags |= state->owns_embedding == 0u ? SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_INPUT : 0u;
	expected_flags |= state->owns_final_head == 0u ? SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_OUTPUT : 0u;
	expected_flags |= SparkGlm52ResidentDecodeStageRequiresSidebandInput(state->stage_index) != 0u ? SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_SIDEBAND_INPUT : 0u;
	expected_flags |= SparkGlm52ResidentDecodeStageRequiresSidebandOutput(state->stage_index) != 0u ? SPARK_GLM52_RESIDENT_DECODE_STAGE_FRAME_FLAG_SIDEBAND_OUTPUT : 0u;
	if ( context->flags != expected_flags )
		return(SPARK_STATUS_SCHEMA_ERROR);
	boundary_bytes = (uint64_t)batch->row_count * SPARK_GLM52_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_COUNT * SPARK_GLM52_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_BYTES;
	sideband_bytes = (uint64_t)batch->row_count * SPARK_GLM52_RESIDENT_DECODE_STAGE_DSA_SIDEBAND_BYTES_PER_ROW;
	if ( (state->owns_embedding == 0u && (context->hidden_input_bf16 == 0 || context->hidden_input_bytes < boundary_bytes)) || (state->owns_embedding != 0u && (context->hidden_input_bf16 != 0 || context->hidden_input_bytes != 0u)) || (state->owns_final_head == 0u && (context->hidden_output_bf16 == 0 || context->hidden_output_bytes < boundary_bytes)) || (state->owns_final_head != 0u && (context->hidden_output_bf16 != 0 || context->hidden_output_bytes != 0u)) )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( (SparkGlm52ResidentDecodeStageRequiresSidebandInput(state->stage_index) != 0u && (context->sideband_input == 0 || context->sideband_input_bytes < sideband_bytes)) || (SparkGlm52ResidentDecodeStageRequiresSidebandInput(state->stage_index) == 0u && (context->sideband_input != 0 || context->sideband_input_bytes != 0u)) || (SparkGlm52ResidentDecodeStageRequiresSidebandOutput(state->stage_index) != 0u && (context->sideband_output == 0 || context->sideband_output_bytes < sideband_bytes)) || (SparkGlm52ResidentDecodeStageRequiresSidebandOutput(state->stage_index) == 0u && (context->sideband_output != 0 || context->sideband_output_bytes != 0u)) )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkGlm52ValidateRoundMajor(state,batch);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52ValidateFrameBuffers(state,frame,batch->row_count);
	*context_out = status == SPARK_STATUS_OK ? context : 0;
	return(status);
}

/*
 * TP8 execution chain. One CUDA chunk = one half-layer (attention or MLP);
 * between chunks the hidden stream is all-reduced across ranks through the
 * spark_tp_device_collective, whose stream-ordered completion resumes the
 * chain. tp_degree == 1 runs the identical chain with the reduce elided, so
 * the single-rank path exercises every chunk boundary.
 */
#define SPARK_GLM52_TP_COLLECTIVE_CREDITS_PER_SLOT 2u

typedef enum SparkGlm52ChainStage
{
	SPARK_GLM52_CHAIN_STAGE_BEGIN = 0,
	SPARK_GLM52_CHAIN_STAGE_ATTENTION,
	SPARK_GLM52_CHAIN_STAGE_REDUCE_ATTENTION,
	SPARK_GLM52_CHAIN_STAGE_MLP,
	SPARK_GLM52_CHAIN_STAGE_REDUCE_MLP,
	SPARK_GLM52_CHAIN_STAGE_HEAD,
	SPARK_GLM52_CHAIN_STAGE_REDUCE_HEAD,
	SPARK_GLM52_CHAIN_STAGE_FINISH
} SparkGlm52ChainStage;

typedef struct SparkGlm52TpChain
{
	SparkGlm52ModuleState *state;
	SparkGlm52ExecutionSlot *slot;
	uint32_t slot_index;
	SparkModelDriverFrame *frame;
	const SparkGlm52ResidentDecodeStageFrameContext *context;
	const SparkGlm52ResidentDecodeStageBatchView *batch;
	SparkGlm52CudaWave wave;
	uint32_t first_row;
	uint32_t wave_rows;
	uint32_t next_wave_row;
	uint32_t stage;
	uint32_t next_layer;
	uint32_t active;
} SparkGlm52TpChain;

static void SparkGlm52TpChainAdvance(void *chain_context,SparkStatus status);
static void CUDART_CB SparkGlm52CompleteAsync(void *context);
static SparkStatus SparkGlm52EnqueueAsyncCompletion(
	SparkGlm52ModuleState *state,
	SparkGlm52ExecutionSlot *slot,
	uint32_t slot_index);

static void SparkGlm52BuildWave(SparkGlm52TpChain *chain)
{
	SparkGlm52ModuleState *state;
	SparkGlm52ExecutionSlot *slot;
	const SparkGlm52ResidentDecodeStageFrameContext *context;
	SparkGlm52CudaWave *wave;
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
	wave->maximum_context = maximum_context;
	wave->resident_sequence_capacity = state->resident_sequence_capacity;
	wave->max_sequence_positions = state->max_sequence_positions;
	wave->pages_per_sequence = state->pages_per_sequence;
	wave->owns_embedding = state->owns_embedding;
	wave->owns_final_head = state->owns_final_head;
	wave->sideband_input = SparkGlm52ResidentDecodeStageRequiresSidebandInput(state->stage_index);
	wave->sideband_output = SparkGlm52ResidentDecodeStageRequiresSidebandOutput(state->stage_index);
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
	wave->layers = state->layers;
	wave->slot = slot;
	wave->kv_cache = state->kv_cache;
	wave->kv_layer_stride_bytes = state->kv_layer_stride_bytes;
	wave->index_cache = state->index_cache;
	wave->index_layer_stride_bytes = state->index_layer_stride_bytes;
	wave->index_ordinal_by_local_layer = state->index_ordinal_by_local_layer;
	wave->page_table = state->page_table;
	wave->multiprocessor_count = state->multiprocessor_count;
}

static SparkStatus SparkGlm52ModuleCombineBf16(
	void *combine_context,
	void *destination_device,
	const void *source_device,
	uint32_t active_sequence_count,
	uint32_t hidden_dimension,
	void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkGlm52LaunchAccumAdd((cudaStream_t)cuda_stream,destination_device,source_device,active_sequence_count,hidden_dimension);
	return(SparkStageModuleCudaStatus(SPARK_GLM52_MODULE_TAG,error,"tp_all_reduce_sum"));
}

static SparkStatus SparkGlm52ModuleCombineU64Max(
	void *combine_context,
	uint64_t *destination_device,
	const uint64_t *source_device,
	uint32_t element_count,
	void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkGlm52LaunchAccumU64Max((cudaStream_t)cuda_stream,destination_device,source_device,element_count);
	return(SparkStageModuleCudaStatus(SPARK_GLM52_MODULE_TAG,error,"tp_all_reduce_max_u64"));
}

static SparkStatus SparkGlm52ModuleInitializeTpCollective(
	SparkGlm52ModuleState *state,
	const SparkGlm52ResidentDecodeStageNodeContext *context)
{
	SparkTpDeviceCollectiveConfig configuration;
	uint64_t credit_bytes,offset,total_bytes;
	uint32_t credit,hidden,memory_mode,route,route_count;
	void *mapped_receive,*mapped_send;
	cudaError_t error;
	SparkStatus status;
	if ( state == 0 || context == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->tp_degree == 1u || state->tp_collective_disabled != 0u )
		return(SPARK_STATUS_OK);
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	configuration.backend_kind = context->tp_collective_backend_kind;
	configuration.tp_degree = state->tp_degree;
	configuration.tp_rank = state->tp_rank;
	configuration.operation_kind = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	configuration.credit_count = state->pipeline_slot_count * SPARK_GLM52_TP_COLLECTIVE_CREDITS_PER_SLOT;
	configuration.local_hidden_dimension = SPARK_GLM52_MODEL_HIDDEN_DIMENSION;
	/* The chain never reduces more rows than one execution wave, so the
	 * credit buffers are priced by execution_row_capacity, not the bucket's
	 * absolute input-row ceiling. */
	configuration.max_active_sequence_count = state->execution_row_capacity;
	configuration.connect_timeout_milli = context->tp_connect_timeout_milli;
	configuration.operation_timeout_milli = context->tp_operation_timeout_milli;
	configuration.control_port_base = context->tp_collective_control_port_base;
	configuration.collective_identifier = context->tp_collective_identifier;
	configuration.backend_module_path = context->tp_collective_backend_module_path;
	configuration.registration_cuda_stream = state->execution_stream;
	status = SparkTpDeviceCollectiveApplyTopology(&context->tp_collective_topology,&configuration);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( configuration.backend_kind == SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT )
	{
		configuration.combine_bf16_function = SparkGlm52ModuleCombineBf16;
		configuration.combine_u64_max_function = SparkGlm52ModuleCombineU64Max;
		configuration.combine_context = state;
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
	total_bytes = 0u;
	for (route=0u; route<route_count; route++)
	{
		hidden = configuration.local_hidden_dimension;
		credit_bytes = (uint64_t)configuration.max_active_sequence_count * hidden * SPARK_GLM52_MODEL_BF16_ELEMENT_BYTES;
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
			return(SparkStageModuleCudaStatus(SPARK_GLM52_MODULE_TAG,error,"tp_credit_alloc"));
		}
		state->tp_credit_send_bf16 = mapped_send;
		state->tp_credit_receive_bf16 = mapped_receive;
	}
	offset = 0u;
	state->tp_credit_binding_count = 0u;
	for (route=0u; route<route_count; route++)
	{
		hidden = configuration.local_hidden_dimension;
		credit_bytes = (uint64_t)configuration.max_active_sequence_count * hidden * SPARK_GLM52_MODEL_BF16_ELEMENT_BYTES;
		for (credit=0u; credit<configuration.credit_count; credit++)
		{
			SparkTpDeviceCollectiveCreditBinding *binding;
			if ( state->tp_credit_binding_count >= SPARK_TP_DEVICE_COLLECTIVE_MAX_BINDING_COUNT )
				return(SPARK_STATUS_CAPACITY_EXCEEDED);
			binding = &state->tp_credit_bindings[state->tp_credit_binding_count++];
			binding->step_index = route;
			binding->credit_index = credit;
			binding->send_device = (uint8_t *)state->tp_credit_send_bf16 + offset;
			binding->receive_device = (uint8_t *)state->tp_credit_receive_bf16 + offset;
			binding->send_transport = memory_mode == SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST ? (uint8_t *)state->tp_host_credit_send_bf16 + offset : binding->send_device;
			binding->receive_transport = memory_mode == SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST ? (uint8_t *)state->tp_host_credit_receive_bf16 + offset : binding->receive_device;
			binding->flags = memory_mode == SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST ? SPARK_TP_DEVICE_COLLECTIVE_BINDING_KNOWN_FLAGS : 0u;
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
	return(SPARK_STATUS_OK);
}

static void SparkGlm52ModuleTpCompletion(
	void *context,
	const SparkTpDeviceCollectiveCompletion *completion)
{
	SparkGlm52TpChain *chain;
	chain = (SparkGlm52TpChain *)context;
	if ( chain == 0 || chain->active == 0u || completion == 0 )
		return;
	SparkGlm52TpChainAdvance(chain,completion->status);
}

static SparkStatus SparkGlm52ModuleReduceHidden(SparkGlm52TpChain *chain,void *device_bf16)
{
	SparkGlm52ModuleState *state;
	SparkTpDeviceCollectiveSubmission submission;
	uint64_t ordinal;
	state = chain->state;
	if ( state->tp_degree == 1u || state->tp_collective_disabled != 0u )
	{
		SparkGlm52TpChainAdvance(chain,SPARK_STATUS_OK);
		return(SPARK_STATUS_OK);
	}
	if ( state->tp_device_collective_initialized == 0u )
		return(SPARK_STATUS_INTERNAL_ERROR);
	ordinal = atomic_fetch_add_explicit(&state->tp_next_ordinal,1u,memory_order_relaxed);
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
	submission.completion_function = SparkGlm52ModuleTpCompletion;
	submission.completion_context = chain;
	return(SparkTpDeviceCollectiveSubmitBf16(&state->tp_device_collective,&submission));
}

static SparkStatus SparkGlm52ModuleReduceHeadMax(SparkGlm52TpChain *chain)
{
	SparkGlm52ModuleState *state;
	SparkTpDeviceCollectiveSubmission submission;
	uint64_t ordinal;
	state = chain->state;
	if ( state->tp_degree == 1u || state->tp_collective_disabled != 0u )
	{
		SparkGlm52TpChainAdvance(chain,SPARK_STATUS_OK);
		return(SPARK_STATUS_OK);
	}
	if ( state->tp_device_collective_initialized == 0u )
		return(SPARK_STATUS_INTERNAL_ERROR);
	ordinal = atomic_fetch_add_explicit(&state->tp_next_ordinal,1u,memory_order_relaxed);
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
	submission.completion_function = SparkGlm52ModuleTpCompletion;
	submission.completion_context = chain;
	return(SparkTpDeviceCollectiveSubmitU64Max(&state->tp_device_collective,&submission));
}

static void SparkGlm52TpChainFail(SparkGlm52TpChain *chain,SparkStatus status)
{
	SparkGlm52ModuleState *state;
	SparkGlm52AsyncCompletion *async;
	state = chain->state;
	(void)cudaStreamSynchronize((cudaStream_t)chain->slot->stream);
	async = &state->completions[chain->slot_index];
	async->completion.status = status;
	chain->active = 0u;
	SparkGlm52CompleteAsync(async);
	free(chain);
}

static void SparkGlm52TpChainAdvance(void *chain_context,SparkStatus status)
{
	SparkGlm52TpChain *chain;
	SparkGlm52ModuleState *state;
	SparkStatus launch_status;
	cudaError_t error;
	chain = (SparkGlm52TpChain *)chain_context;
	if ( chain == 0 || chain->active == 0u )
		return;
	state = chain->state;
	if ( status != SPARK_STATUS_OK )
	{
		SparkGlm52TpChainFail(chain,status);
		return;
	}
	switch ( chain->stage )
	{
	case SPARK_GLM52_CHAIN_STAGE_BEGIN:
		SparkGlm52BuildWave(chain);
		if ( SparkGlm52LaunchCudaWaveBegin(&chain->wave) != 0 )
		{
			SparkGlm52TpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->stage = SPARK_GLM52_CHAIN_STAGE_ATTENTION;
		chain->next_layer = 0u;
		/* The embedding wrote the partial stream into hidden_bf16. */
		launch_status = SparkGlm52ModuleReduceHidden(chain,chain->slot->hidden_bf16);
		if ( launch_status != SPARK_STATUS_OK )
			SparkGlm52TpChainFail(chain,launch_status);
		return;
	case SPARK_GLM52_CHAIN_STAGE_ATTENTION:
		if ( SparkGlm52LaunchCudaLayerAttention(&chain->wave,chain->next_layer) != 0 )
		{
			SparkGlm52TpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->stage = SPARK_GLM52_CHAIN_STAGE_REDUCE_ATTENTION;
		/* The attention writes its partial output into attention_out_bf16,
		 * NOT hidden_bf16: the hidden buffer still holds the pre-attention
		 * stream and must not be reduced again. */
		launch_status = SparkGlm52ModuleReduceHidden(chain,chain->slot->attention_out_bf16);
		if ( launch_status != SPARK_STATUS_OK )
			SparkGlm52TpChainFail(chain,launch_status);
		return;
	case SPARK_GLM52_CHAIN_STAGE_REDUCE_ATTENTION:
		chain->stage = SPARK_GLM52_CHAIN_STAGE_MLP;
		SparkGlm52TpChainAdvance(chain,SPARK_STATUS_OK);
		return;
	case SPARK_GLM52_CHAIN_STAGE_MLP:
		if ( SparkGlm52LaunchCudaLayerMlp(&chain->wave,chain->next_layer) != 0 )
		{
			SparkGlm52TpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->stage = SPARK_GLM52_CHAIN_STAGE_REDUCE_MLP;
		/* The MLP finalize writes its partial stream into hidden_bf16. */
		launch_status = SparkGlm52ModuleReduceHidden(chain,chain->slot->hidden_bf16);
		if ( launch_status != SPARK_STATUS_OK )
			SparkGlm52TpChainFail(chain,launch_status);
		return;
	case SPARK_GLM52_CHAIN_STAGE_REDUCE_MLP:
		chain->next_layer++;
		if ( chain->next_layer < chain->wave.layer_count )
		{
			chain->stage = SPARK_GLM52_CHAIN_STAGE_ATTENTION;
			SparkGlm52TpChainAdvance(chain,SPARK_STATUS_OK);
		}
		else
		{
			chain->stage = SPARK_GLM52_CHAIN_STAGE_HEAD;
			SparkGlm52TpChainAdvance(chain,SPARK_STATUS_OK);
		}
		return;
	case SPARK_GLM52_CHAIN_STAGE_HEAD:
		if ( SparkGlm52LaunchCudaWaveHead(&chain->wave) != 0 )
		{
			SparkGlm52TpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->stage = SPARK_GLM52_CHAIN_STAGE_REDUCE_HEAD;
		launch_status = SparkGlm52ModuleReduceHeadMax(chain);
		if ( launch_status != SPARK_STATUS_OK )
			SparkGlm52TpChainFail(chain,launch_status);
		return;
	case SPARK_GLM52_CHAIN_STAGE_REDUCE_HEAD:
		error = SparkGlm52LaunchHeadMaxlocUnpack((cudaStream_t)chain->slot->stream,chain->slot->head_maxloc_u64,chain->slot->output_token,chain->wave_rows);
		if ( error == cudaSuccess && state->owns_final_head != 0u )
			error = cudaMemcpyAsync(chain->slot->host_output_token_ids + chain->first_row,chain->slot->output_token,(uint64_t)chain->wave_rows * sizeof(uint32_t),cudaMemcpyDeviceToHost,(cudaStream_t)chain->slot->stream);
		launch_status = SparkStageModuleCudaStatus(SPARK_GLM52_MODULE_TAG,error,"tp_head_unpack");
		if ( launch_status != SPARK_STATUS_OK )
		{
			SparkGlm52TpChainFail(chain,launch_status);
			return;
		}
		if ( chain->next_wave_row < chain->batch->row_count )
		{
			uint32_t next_wave;
			next_wave = SparkGlm52RoundMajorWaveRows(chain->batch,chain->next_wave_row);
			if ( next_wave == 0u )
			{
				SparkGlm52TpChainFail(chain,SPARK_STATUS_INVALID_ARGUMENT);
				return;
			}
			chain->first_row = chain->next_wave_row;
			chain->wave_rows = next_wave;
			chain->next_wave_row += next_wave;
			chain->stage = SPARK_GLM52_CHAIN_STAGE_BEGIN;
			SparkGlm52TpChainAdvance(chain,SPARK_STATUS_OK);
			return;
		}
		launch_status = SparkGlm52EnqueueAsyncCompletion(state,chain->slot,chain->slot_index);
		if ( launch_status != SPARK_STATUS_OK )
		{
			SparkGlm52TpChainFail(chain,launch_status);
			return;
		}
		chain->stage = SPARK_GLM52_CHAIN_STAGE_FINISH;
		chain->active = 0u;
		free(chain);
		return;
	default:
		SparkGlm52TpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
		return;
	}
}

static SparkStatus SparkGlm52StageHostBatch(
	const SparkGlm52ModuleState *state,
	SparkGlm52ExecutionSlot *slot,
	const SparkGlm52ResidentDecodeStageBatchView *batch)
{
	uint32_t row;
	if ( state == 0 || slot == 0 || batch == 0 || batch->row_count > SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT )
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
	memset(slot->host_kv_access_error,0,SPARK_GLM52_KV_ACCESS_ERROR_WORD_COUNT * sizeof(uint32_t));
	return(SPARK_STATUS_OK);
}

static void SparkGlm52PrepareAsyncCompletion(
	SparkGlm52ModuleState *state,
	SparkModelDriverFrame *frame,
	const SparkGlm52ResidentDecodeStageBatchView *batch,
	const uint8_t *lane_bound,
	const uint64_t *lane_sequence_ids,
	const uint64_t *lane_next_positions,
	uint32_t slot_index)
{
	SparkGlm52AsyncCompletion *async;
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
	// The compiled bucket is the hard ceiling at the copy, not just upstream
	// of it: SparkGlm52ValidateFrame rejects an active_sequence_count above
	// resident_sequence_capacity and ModuleConfigure bounds that capacity by
	// MAX_ACTIVE_SEQUENCE_COUNT, but a tight variant build (b8 lane tables)
	// prices a broken invariant as a heap overflow the compiler can see, so
	// the loop names the ceiling itself.
	for (lane=0u; lane<batch->active_sequence_count && lane<SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT; lane++)
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

static void CUDART_CB SparkGlm52CompleteAsync(void *context)
{
	SparkGlm52AsyncCompletion *async;
	SparkGlm52ModuleState *state;
	SparkGlm52ExecutionSlot *slot;
	uint32_t lane,resident;
	async = (SparkGlm52AsyncCompletion *)context;
	state = async != 0 ? async->state : 0;
	if ( state == 0 || async->slot_index >= state->pipeline_slot_count )
		return;
	slot = &state->slots[async->slot_index];
	if ( slot->host_kv_access_error[0] != 0u )
		async->completion.status = SPARK_STATUS_INTERNAL_ERROR;
	if ( async->completion.status == SPARK_STATUS_OK )
	{
		if ( async->output_token_destination != 0 )
			memcpy(async->output_token_destination,slot->host_output_token_ids,(uint64_t)async->row_count * sizeof(uint32_t));
		for (lane=0u; lane<async->lane_count; lane++)
		{
			resident = async->lane_indices[lane];
			atomic_store_explicit(&state->lane_bound[resident],async->lane_bound[lane],memory_order_release);
			atomic_store_explicit(&state->lane_sequence_ids[resident],async->lane_sequence_ids[lane],memory_order_release);
			atomic_store_explicit(&state->lane_next_positions[resident],async->lane_next_positions[lane],memory_order_release);
			if ( SparkKvPageCacheCompleteLane(&state->kv_page_cache,&state->kv_lane_cache_lanes[resident]) != SPARK_STATUS_OK )
				async->completion.status = SPARK_STATUS_INTERNAL_ERROR;
		}
		atomic_fetch_add_explicit(&state->completed_count,1u,memory_order_relaxed);
	}
	else
	{
		for (lane=0u; lane<async->lane_count; lane++)
		{
			resident = async->lane_indices[lane];
			if ( SparkKvPageCacheRollbackLaneTransaction(&state->kv_page_cache,&state->kv_lane_cache_lanes[resident],state->kv_lane_mutation_flags[resident]) != SPARK_STATUS_OK )
				async->completion.status = SPARK_STATUS_INTERNAL_ERROR;
			atomic_store_explicit(&state->lane_bound[resident],0u,memory_order_release);
		}
		atomic_fetch_add_explicit(&state->failed_count,1u,memory_order_relaxed);
	}
	atomic_fetch_add_explicit(&state->host_callback_completion_count,1u,memory_order_relaxed);
	SparkStageModuleCompleteAndReleaseClaims(async->completion_function,async->completion_context,&async->completion,state->lane_states,state->resident_sequence_capacity,async->lane_indices,async->lane_count,state->slot_states,async->slot_index);
}

static SparkStatus SparkGlm52EnqueueAsyncCompletion(
	SparkGlm52ModuleState *state,
	SparkGlm52ExecutionSlot *slot,
	uint32_t slot_index)
{
	cudaStream_t stream;
	cudaError_t error;
	stream = (cudaStream_t)slot->stream;
	error = cudaMemcpyAsync(slot->host_kv_access_error,slot->kv_access_error,SPARK_GLM52_KV_ACCESS_ERROR_WORD_COUNT * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream);
	if ( error == cudaSuccess )
		error = cudaLaunchHostFunc(stream,SparkGlm52CompleteAsync,&state->completions[slot_index]);
	return(SparkStageModuleCudaStatus(SPARK_GLM52_MODULE_TAG,error,"async_completion"));
}

static void SparkGlm52InvalidateClaimedLanes(
	SparkGlm52ModuleState *state,
	const uint32_t *lane_indices,
	uint32_t lane_count)
{
	uint32_t lane;
	for (lane=0u; lane<lane_count; lane++)
		atomic_store_explicit(&state->lane_bound[lane_indices[lane]],0u,memory_order_release);
}

static SparkStatus SparkGlm52ExecuteBatch(
	SparkGlm52ModuleState *state,
	SparkModelDriverFrame *frame,
	const SparkGlm52ResidentDecodeStageFrameContext *context)
{
	const SparkGlm52ResidentDecodeStageBatchView *batch;
	SparkGlm52ClaimedContinuityContext continuity;
	SparkGlm52ExecutionSlot *slot;
	SparkGlm52TpChain *chain;
	uint8_t simulated_bound[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t simulated_sequence[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t simulated_next[SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t slot_index,wave_rows;
	SparkStatus status;
	cudaError_t error;
	batch = context->batch;
	continuity.state = state;
	continuity.batch = batch;
	continuity.bound = simulated_bound;
	continuity.sequence_ids = simulated_sequence;
	continuity.next_positions = simulated_next;
	status = SparkStageModuleIndexSetClaimAndPrepare(state->lane_states,state->resident_sequence_capacity,batch->row_resident_slots,batch->active_sequence_count,SparkGlm52PrepareClaimedContinuity,&continuity);
	if ( status != SPARK_STATUS_OK )
		return(status);
	slot_index = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
	status = SparkStageModuleSlotClaim(state->slot_states,state->pipeline_slot_count,&slot_index);
	if ( status != SPARK_STATUS_OK )
	{
		SparkStageModuleIndexSetRelease(state->lane_states,state->resident_sequence_capacity,batch->row_resident_slots,batch->active_sequence_count);
		return(status);
	}
	slot = &state->slots[slot_index];
	slot->stream = frame->execution_stream;
	status = SparkGlm52StageHostBatch(state,slot,batch);
	if ( status != SPARK_STATUS_OK )
	{
		SparkStageModuleSlotRelease(state->slot_states,slot_index);
		SparkStageModuleIndexSetRelease(state->lane_states,state->resident_sequence_capacity,batch->row_resident_slots,batch->active_sequence_count);
		return(status);
	}
	SparkGlm52PrepareAsyncCompletion(state,frame,batch,simulated_bound,simulated_sequence,simulated_next,slot_index);
	atomic_fetch_add_explicit(&state->submitted_count,1u,memory_order_relaxed);
	error = cudaMemsetAsync(slot->kv_access_error,0,SPARK_GLM52_KV_ACCESS_ERROR_WORD_COUNT * sizeof(uint32_t),(cudaStream_t)slot->stream);
	status = SparkStageModuleCudaStatus(SPARK_GLM52_MODULE_TAG,error,"kv_access_reset");
	wave_rows = status == SPARK_STATUS_OK ? SparkGlm52RoundMajorWaveRows(batch,0u) : 0u;
	if ( status == SPARK_STATUS_OK && wave_rows == 0u )
		status = SPARK_STATUS_INVALID_ARGUMENT;
	if ( status == SPARK_STATUS_OK )
	{
		chain = (SparkGlm52TpChain *)calloc(1u,sizeof(*chain));
		if ( chain == 0 )
			status = SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	if ( status != SPARK_STATUS_OK )
	{
		(void)cudaStreamSynchronize((cudaStream_t)slot->stream);
		SparkGlm52InvalidateClaimedLanes(state,batch->row_resident_slots,batch->active_sequence_count);
		atomic_fetch_add_explicit(&state->failed_count,1u,memory_order_relaxed);
		SparkStageModuleSlotRelease(state->slot_states,slot_index);
		SparkStageModuleIndexSetRelease(state->lane_states,state->resident_sequence_capacity,batch->row_resident_slots,batch->active_sequence_count);
		return(status);
	}
	chain->state = state;
	chain->slot = slot;
	chain->slot_index = slot_index;
	chain->frame = frame;
	chain->context = context;
	chain->batch = batch;
	chain->first_row = 0u;
	chain->wave_rows = wave_rows;
	chain->next_wave_row = wave_rows;
	chain->stage = SPARK_GLM52_CHAIN_STAGE_BEGIN;
	chain->active = 1u;
	SparkGlm52TpChainAdvance(chain,SPARK_STATUS_OK);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkGlm52ResidentDecodeStageExecute(
	void *module_state,
	SparkModelDriverFrame *frame)
{
	SparkGlm52ModuleState *state;
	const SparkGlm52ResidentDecodeStageFrameContext *context;
	SparkStatus status;
	state = (SparkGlm52ModuleState *)module_state;
	context = 0;
	status = SparkGlm52ValidateFrame(state,frame,&context);
	if ( status != SPARK_STATUS_OK )
	{
		if ( state != 0 )
			atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
		return(status);
	}
	status = SparkGlm52ExecuteBatch(state,frame,context);
	if ( status != SPARK_STATUS_OK )
		atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
	return(status);
}

static void SparkGlm52AdmissionCost(
	void *context,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	(void)context;
	decision->host_staging_bytes = (uint64_t)request->new_token_count *
		(sizeof(uint32_t) * 3u + sizeof(uint64_t) * 2u);
	decision->device_memcpy_bytes = decision->host_staging_bytes;
}

SparkStatus SparkGlm52ResidentDecodeStageAdmit(
	void *module_state,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	SparkGlm52ModuleState *state;
	SparkAdmissionPolicyTable table;
	uint32_t available;
	SparkStatus status;
	state = (SparkGlm52ModuleState *)module_state;
	if ( state == 0 || request == 0 || decision == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	available = SparkStageModuleSlotCountFree(state->slot_states,state->pipeline_slot_count);
	memset(&table,0,sizeof(table));
	table.abi_version = SPARK_ADMISSION_ABI_VERSION;
	table.descriptor_bytes = (uint32_t)sizeof(table);
	table.max_active_sequence_count = state->resident_sequence_capacity;
	table.max_input_row_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT;
	table.max_sequence_positions = state->max_sequence_positions;
	table.flags = SPARK_ADMISSION_POLICY_FLAG_DECODE_EQUALS_SLOTS |
		SPARK_ADMISSION_POLICY_FLAG_ALLOW_DISPATCH_FLAG;
	table.predicate = SparkGlm52AdmissionPredicate;
	table.predicate_context = state;
	table.cost = SparkGlm52AdmissionCost;
	table.cost_context = state;
	status = SparkAdmissionEvaluateShape(&table,available,request,decision);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( decision->accepted == 0u )
		atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
	return(status);
}

SparkStatus SparkGlm52ResidentDecodeStageSnapshot(
	void *module_state,
	uint32_t program_id,
	SparkModelDriverRuntimeSnapshot *snapshot)
{
	SparkGlm52ModuleState *state;
	uint32_t index,resident_count;
	state = (SparkGlm52ModuleState *)module_state;
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

void SparkGlm52ResidentDecodeStageDestroy(void *module_state)
{
	SparkGlm52ModuleState *state;
	state = (SparkGlm52ModuleState *)module_state;
	if ( state == 0 )
		return;
	if ( SparkStageModuleWaitForSlots(SPARK_GLM52_MODULE_TAG,state->slot_states,state->pipeline_slot_count,SPARK_STAGE_MODULE_DESTROY_QUIESCE_TIMEOUT_NS) != SPARK_STATUS_OK )
		return;
	(void)cudaStreamSynchronize((cudaStream_t)state->execution_stream);
	if ( state->tp_device_collective_initialized != 0u )
		SparkTpDeviceCollectiveDestroy(&state->tp_device_collective);
	if ( state->kv_page_store.abi_version == SPARK_KV_PAGE_STORE_ABI_VERSION )
		SparkKvPageStoreDestroy(&state->kv_page_store);
	SparkGlm52ReleaseSlotHost(state);
	SparkStageModuleLedgerRelease(&state->ledger);
	free(state->kv_blocks);
	free(state->kv_resident_slot_logical_block_indices);
	free(state->kv_entries);
	free(state->kv_sequences);
	free(state->kv_hash_bucket_heads);
	free(state->kv_entry_indices_by_logical_page);
	free(state->kv_page_staging);
	free(state->kv_lane_logical_pages);
	free(state->kv_lane_page_count);
	free(state->kv_lane_mutable_page);
	free(state->kv_lane_mutation_flags);
	free(state->kv_lane_cache_lanes);
	free(state);
}

static SparkStatus SparkGlm52InitializeState(
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services,
	SparkGlm52ModuleState **state_out)
{
	SparkGlm52ModuleState *state;
	const char *pack_path;
	SparkStatus status;
	uint32_t lane;
	state = (SparkGlm52ModuleState *)calloc(1u,sizeof(*state));
	if ( state == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->ledger.module_tag = SPARK_GLM52_MODULE_TAG;
	status = SparkGlm52ModuleConfigure(state,configuration,host_services,&pack_path);
	if ( status == SPARK_STATUS_OK && SparkGlm52ConfigureCudaModule(&state->multiprocessor_count) != 0 )
		status = SPARK_STATUS_TARGET_MISMATCH;
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52PackLoad(state,pack_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52AllocateCaches(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52AllocateSlots(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlm52ModuleInitializeTpCollective(state,(const SparkGlm52ResidentDecodeStageNodeContext *)host_services->node_context);
	if ( status != SPARK_STATUS_OK )
	{
		SparkGlm52ReleaseSlotHost(state);
		SparkStageModuleLedgerRelease(&state->ledger);
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
	atomic_init(&state->tp_next_ordinal,0u);
	*state_out = state;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkGlm52ResidentDecodeStageInitialize(
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services,
	void **module_state)
{
	SparkGlm52ModuleState *state;
	SparkStatus status;
	status = SparkFirmwareModuleValidateInitialization(configuration,host_services,module_state);
	if ( status != SPARK_STATUS_OK )
		return(status);
	state = 0;
	status = SparkGlm52InitializeState(configuration,host_services,&state);
	if ( status != SPARK_STATUS_OK )
		return(status);
	*module_state = state;
	return(SPARK_STATUS_OK);
}
