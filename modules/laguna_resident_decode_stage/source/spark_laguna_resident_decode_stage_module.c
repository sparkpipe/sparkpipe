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

#include "sparkpipe/spark_laguna_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_error_site.h"

#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_admission.h"
#include "sparkpipe/spark_kv_model_table.h"
#include "sparkpipe/spark_laguna_kv_geometry.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "sparkpipe/spark_row_layout.h"
#include "sparkpipe/spark_weightd_attach.h"
#include "sparkpipe/spark_weightd_lazy_pack.h"
#include "sparkpipe/spark_weightd_lease.h"
#include "spark_laguna_resident_decode_stage_internal.h"
#include "spark_laguna_stagepack_format.h"

#ifndef LAGUNA_EXPERT_WEIGHT_CODEC
#error "LAGUNA_EXPERT_WEIGHT_CODEC must name the exact package expert codec"
#endif
#ifndef LAGUNA_CONTRACT_SHA256
#error "LAGUNA_CONTRACT_SHA256 must identify the exact model package contract"
#endif

#define SPARK_LAGUNA_MODULE_TAG "laguna_stage"
#define SPARK_LAGUNA_STAGEPACK_MAX_TENSOR_COUNT 2048u
#define SPARK_LAGUNA_HEAD_TILE 1024u
#define SPARK_LAGUNA_NO_INDEX_ORDINAL UINT32_MAX
#define SPARK_LAGUNA_KV_ACCESS_ERROR_WORD_COUNT 6u
#define LAGUNA_HEAD_TILE_COUNT 1024u
#define LAGUNA_MAX_QKV_ROWS \
	(SPARK_LAGUNA_MODEL_TP8_Q_HEAD_COUNT_SLIDING * SPARK_LAGUNA_MODEL_ATTENTION_HEAD_DIMENSION + \
	 2u * SPARK_LAGUNA_MODEL_TP8_KV_HEAD_COUNT * SPARK_LAGUNA_MODEL_ATTENTION_HEAD_DIMENSION)

_Static_assert(SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT <= SPARK_WEIGHTD_WORK_QUEUE_CAPACITY,"completion worker must hold one job per occupied slot");

#if SPARK_LAGUNA_MODEL_KV_SLOT_BYTES != \
	( SPARK_LAGUNA_KV_ARENA_KV_HEAD_COUNT * \
	  SPARK_LAGUNA_KV_ARENA_HEAD_DIM * \
	  SPARK_LAGUNA_KV_BYTES_PER_SCALAR )
#error "laguna KV slot bytes and arena block geometry disagree"
#endif

typedef struct SparkLagunaPackRange
{
	uint64_t offset;
	uint64_t bytes;
} SparkLagunaPackRange;

typedef struct SparkLagunaModuleState SparkLagunaModuleState;

typedef struct SparkLagunaAsyncCompletion
{
	SparkLagunaModuleState *state;
	SparkModelDriverCompletionFunction completion_function;
	void *completion_context;
	uint32_t slot_index;
	uint32_t lane_count;
	uint32_t row_count;
	uint32_t lane_indices[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint8_t lane_bound[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_sequence_ids[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_next_positions[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t *output_token_destination;
	uint32_t burst_token_count;
	SparkModelDriverCompletion completion;
} SparkLagunaAsyncCompletion;

struct SparkLagunaModuleState
{
	SparkStageModuleLedger ledger;
	SparkWeightdWorker *completion_worker;
	SparkWeightdLazyPack *lazy_pack;
	_Atomic(void *) lazy_retained[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
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
	uint32_t multiprocessor_count;
	uint32_t owns_embedding;
	uint32_t owns_final_head;
	void *execution_stream;
	float *yarn_inv_freq;
	char model_revision[SPARK_LAGUNA_STAGEPACK_MODEL_REVISION_BYTES];
	SparkLagunaLayerWeights layers[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE];
	uint64_t layer_seen[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE];
	uint64_t global_seen;
	uint32_t dflash_sections;
	uint64_t dflash_bytes;
	const void *embedding_bf16;
	const void *final_norm_bf16;
	const void *lm_head_bf16;
	uint8_t *kv_cache;
	uint64_t kv_layer_stride_bytes;
	uint32_t *page_table;
	uint32_t *page_table_shadow;
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
	SparkLagunaExecutionSlot slots[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	SparkLagunaAsyncCompletion completions[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	atomic_uint slot_states[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	atomic_uint lane_states[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	atomic_uchar lane_bound[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	atomic_ullong lane_sequence_ids[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	atomic_ullong lane_next_positions[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	atomic_ullong submitted_count;
	atomic_ullong completed_count;
	atomic_ullong rejected_count;
	atomic_ullong failed_count;
	atomic_ullong host_callback_completion_count;
	SparkTpDeviceCollective tp_device_collective;
	uint32_t tp_device_collective_initialized;
	atomic_ullong nccl_next_ordinal;
};


static SparkStatus SparkLagunaModuleConfigure(
	SparkLagunaModuleState *state,
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services,
	const char **pack_path)
{
	const SparkLagunaResidentDecodeStageNodeContext *context;
	if ( state == 0 || configuration == 0 || host_services == 0 || pack_path == 0 || host_services->node_context == 0 || host_services->execution_stream == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	context = (const SparkLagunaResidentDecodeStageNodeContext *)host_services->node_context;
	if ( context->abi_version != SPARK_LAGUNA_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION || context->descriptor_bytes != SPARK_LAGUNA_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES )
		SPARK_FAIL(SPARK_STATUS_ABI_MISMATCH);
	if ( SparkLagunaResidentDecodeStageSpanIsValid(context->stage_count,context->stage_index,context->first_layer_index,context->layer_count) == 0u || context->layer_count > SPARK_LAGUNA_RESIDENT_DECODE_STAGE_LAYERS_PER_STAGE || context->expert_weight_codec != LAGUNA_EXPERT_WEIGHT_CODEC || context->resident_sequence_capacity == 0u || context->resident_sequence_capacity > SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT || context->pipeline_slot_count == 0u || context->pipeline_slot_count > SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT || context->max_sequence_positions == 0u || context->max_sequence_positions > SPARK_LAGUNA_MODEL_MAXIMUM_CONTEXT_TOKENS || context->execution_row_capacity == 0u || context->execution_row_capacity > SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT || context->decode_split_context_threshold > context->max_sequence_positions || context->tp_degree == 0u || context->tp_rank >= context->tp_degree || (context->flags & ~SPARK_LAGUNA_RESIDENT_DECODE_STAGE_NODE_CONTEXT_KNOWN_FLAGS) != 0u || context->stage_pack_path == 0 || context->stage_pack_path[0] == '\0' || context->model_revision == 0 || context->model_revision[0] == '\0' || strlen(context->model_revision) >= sizeof(state->model_revision) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( SparkWeightCodecIsKnown(context->expert_weight_codec) == 0u )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	if ( context->tp_degree != SPARK_LAGUNA_MODEL_TENSOR_PARALLEL_DEGREE || context->tp_degree == 0u || SPARK_LAGUNA_MODEL_OUTPUT_VOCAB_COUNT % context->tp_degree != 0u || SPARK_LAGUNA_MODEL_DENSE_INTERMEDIATE_DIMENSION % context->tp_degree != 0u || SPARK_LAGUNA_MODEL_MOE_INTERMEDIATE_DIMENSION % context->tp_degree != 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->model_revision == 0 || strcmp(configuration->model_revision,context->model_revision) != 0 )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
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
	state->execution_stream = host_services->execution_stream;
	(void)snprintf(state->model_revision,sizeof(state->model_revision),"%s",context->model_revision);
	*pack_path = context->stage_pack_path;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkLagunaPackFileSize(FILE *file,uint64_t *bytes)
{
	off_t end;
	if ( file == 0 || bytes == 0 || fseeko(file,0,SEEK_END) != 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	end = ftello(file);
	if ( end < 0 || fseeko(file,0,SEEK_SET) != 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	*bytes = (uint64_t)end;
	return(SPARK_STATUS_OK);
}

static uint32_t SparkLagunaPackRangesOverlap(const SparkLagunaPackRange *left,const SparkLagunaPackRange *right)
{
	return(left->bytes != 0u && right->bytes != 0u && left->offset < right->offset + right->bytes && right->offset < left->offset + left->bytes ? 1u : 0u);
}

static SparkStatus SparkLagunaPackValidateHeader(
	const SparkLagunaModuleState *state,
	const SparkLagunaStagePackHeader *header,
	uint64_t file_bytes)
{
	uint64_t directory_bytes,directory_end;
	if ( state == 0 || header == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( header->magic != SPARK_LAGUNA_STAGEPACK_MAGIC || header->format_version != SPARK_LAGUNA_STAGEPACK_FORMAT_VERSION || header->header_bytes != SPARK_LAGUNA_STAGEPACK_HEADER_BYTES || header->directory_entry_bytes != SPARK_LAGUNA_STAGEPACK_ENTRY_BYTES || header->codec_abi_version != SPARK_WEIGHT_CODEC_ABI_VERSION )
		SPARK_FAIL(SPARK_STATUS_ABI_MISMATCH);
	if ( (header->flags & ~SPARK_LAGUNA_STAGEPACK_KNOWN_FLAGS) != 0u )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	if ( SparkLagunaStagePackHeaderTpDegree(header) == 0u || SparkLagunaStagePackHeaderTpDegree(header) != state->tp_degree || SparkLagunaStagePackHeaderTpRank(header) != state->tp_rank )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( header->tensor_count == 0u || header->tensor_count > SPARK_LAGUNA_STAGEPACK_MAX_TENSOR_COUNT || header->stage_count != state->stage_count || header->stage_index != state->stage_index || header->first_layer_index != state->first_layer_index || header->layer_count != state->layer_count || header->total_layer_count != SPARK_LAGUNA_MODEL_LAYER_COUNT || header->hidden_dimension != SPARK_LAGUNA_MODEL_HIDDEN_DIMENSION || header->vocab_count != SPARK_LAGUNA_MODEL_OUTPUT_VOCAB_COUNT || header->routed_expert_count != SPARK_LAGUNA_MODEL_MOE_EXPERT_COUNT )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( header->linear_weight_codec != SPARK_WEIGHT_CODEC_BF16 || header->expert_weight_codec != state->expert_weight_codec || header->kv_cache_codec != SPARK_WEIGHT_CODEC_BF16 )
		SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
	if ( header->file_bytes != file_bytes || header->directory_offset < header->header_bytes || header->directory_offset % SPARK_LAGUNA_STAGEPACK_ALIGNMENT_BYTES != 0u || header->tensor_count > UINT64_MAX / header->directory_entry_bytes )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	directory_bytes = (uint64_t)header->tensor_count * header->directory_entry_bytes;
	directory_end = header->directory_offset + directory_bytes;
	if ( directory_end < header->directory_offset || directory_end > file_bytes )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkLagunaPackValidateEntryGeometry(
	SparkLagunaModuleState *state,
	const SparkLagunaStagePackHeader *header,
	const SparkLagunaStagePackEntry *entry,
	SparkLagunaStagePackTensorShape *shape)
{
	uint64_t payload_bytes,scale_bytes,directory_end;
	uint32_t local_layer;
	if ( entry->tensor_kind >= SPARK_LAGUNA_STAGEPACK_TENSOR_KIND_COUNT )
	{
		if ( (header->flags & SPARK_LAGUNA_STAGEPACK_FLAG_DFLASH) == 0u || entry->layer_index >= SPARK_LAGUNA_MODEL_LAYER_COUNT )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		if ( entry->payload_offset % SPARK_LAGUNA_STAGEPACK_ALIGNMENT_BYTES != 0u || entry->payload_offset > header->file_bytes || entry->payload_bytes > header->file_bytes - entry->payload_offset )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		state->dflash_sections += 1u;
		state->dflash_bytes += entry->payload_bytes + entry->scale_bytes;
		return(SPARK_STATUS_OK);
	}
	if ( SparkLagunaStagePackExpectedShape(entry->tensor_kind,entry->layer_index,state->expert_weight_codec,state->tp_degree,shape) < 0 )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( entry->layer_index != SPARK_LAGUNA_STAGEPACK_GLOBAL_LAYER )
	{
		if ( entry->layer_index < state->first_layer_index || entry->layer_index >= state->first_layer_index + state->layer_count )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		local_layer = entry->layer_index - state->first_layer_index;
		if ( (state->layer_seen[local_layer] & (UINT64_C(1) << entry->tensor_kind)) != 0u )
			SPARK_FAIL(SPARK_STATUS_DUPLICATE);
	}
	else if ( (state->global_seen & (UINT64_C(1) << entry->tensor_kind)) != 0u )
		SPARK_FAIL(SPARK_STATUS_DUPLICATE);
	if ( entry->payload_type != shape->payload_type || entry->weight_codec != shape->weight_codec || entry->scale_encoding != shape->scale_encoding || entry->group_count != shape->group_count || entry->rows != shape->rows || entry->columns != shape->columns )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	payload_bytes = SparkLagunaStagePackExpectedPayloadBytes(shape);
	scale_bytes = SparkLagunaStagePackExpectedScaleBytes(shape);
	if ( payload_bytes == 0u || entry->payload_bytes != payload_bytes || entry->scale_bytes != scale_bytes )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	directory_end = header->directory_offset + ((uint64_t)header->tensor_count * header->directory_entry_bytes);
	if ( entry->payload_offset % SPARK_LAGUNA_STAGEPACK_ALIGNMENT_BYTES != 0u || entry->payload_offset > header->file_bytes || entry->payload_bytes > header->file_bytes - entry->payload_offset )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( entry->payload_offset < directory_end && header->directory_offset < entry->payload_offset + entry->payload_bytes )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	if ( scale_bytes == 0u )
	{
		if ( entry->scale_offset != 0u )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	else if ( entry->scale_offset % SPARK_LAGUNA_STAGEPACK_ALIGNMENT_BYTES != 0u || entry->scale_offset > header->file_bytes || entry->scale_bytes > header->file_bytes - entry->scale_offset )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	else if ( entry->scale_offset < directory_end && header->directory_offset < entry->scale_offset + entry->scale_bytes )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkLagunaPackValidateRanges(
	const SparkLagunaStagePackEntry *entries,
	uint32_t entry_count)
{
	SparkLagunaPackRange left[2],right[2];
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
					if ( SparkLagunaPackRangesOverlap(&left[left_part],&right[right_part]) != 0u )
						SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
		if ( SparkLagunaPackRangesOverlap(&left[0],&left[1]) != 0u )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	return(SPARK_STATUS_OK);
}

static void SparkLagunaPackMarkSeen(
	SparkLagunaModuleState *state,
	const SparkLagunaStagePackEntry *entry)
{
	if ( entry->layer_index == SPARK_LAGUNA_STAGEPACK_GLOBAL_LAYER )
		state->global_seen |= UINT64_C(1) << entry->tensor_kind;
	else
		state->layer_seen[entry->layer_index - state->first_layer_index] |= UINT64_C(1) << entry->tensor_kind;
}

static SparkStatus SparkLagunaPackAssignLayer(
	SparkLagunaLayerWeights *weights,
	const SparkLagunaStagePackEntry *entry,
	const void *payload,
	const void *scale)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_LAGUNA_STAGEPACK_TENSOR_ATTN_INPUT_NORM: weights->attn_norm_bf16 = payload; break;
	case SPARK_LAGUNA_STAGEPACK_TENSOR_FUSED_QKV: weights->fused_qkv_bf16 = payload; break;
	case SPARK_LAGUNA_STAGEPACK_TENSOR_Q_NORM: weights->q_norm_bf16 = payload; break;
	case SPARK_LAGUNA_STAGEPACK_TENSOR_K_NORM: weights->k_norm_bf16 = payload; break;
	case SPARK_LAGUNA_STAGEPACK_TENSOR_ATTN_OUTPUT: weights->attn_output_bf16 = payload; break;
	case SPARK_LAGUNA_STAGEPACK_TENSOR_ATTN_GATE: weights->attn_gate_bf16 = payload; break;
	case SPARK_LAGUNA_STAGEPACK_TENSOR_ATTN_POST_NORM: weights->post_attn_norm_bf16 = payload; break;
	case SPARK_LAGUNA_STAGEPACK_TENSOR_DENSE_GATE_UP: weights->dense_gate_up_bf16 = payload; break;
	case SPARK_LAGUNA_STAGEPACK_TENSOR_DENSE_DOWN: weights->dense_down_bf16 = payload; break;
	case SPARK_LAGUNA_STAGEPACK_TENSOR_ROUTER: weights->router_bf16 = payload; break;
	case SPARK_LAGUNA_STAGEPACK_TENSOR_ROUTER_CORRECTION: weights->router_correction_f32 = (const float *)payload; break;
	case SPARK_LAGUNA_STAGEPACK_TENSOR_EXPERT_GATE_UP: weights->expert_gate_up_payload = payload; weights->expert_gate_up_scale = scale; weights->expert_gate_up_payload_offset = entry->payload_offset; weights->expert_gate_up_scale_offset = entry->scale_offset; break;
	case SPARK_LAGUNA_STAGEPACK_TENSOR_EXPERT_DOWN: weights->expert_down_payload = payload; weights->expert_down_scale = scale; weights->expert_down_payload_offset = entry->payload_offset; weights->expert_down_scale_offset = entry->scale_offset; break;
	case SPARK_LAGUNA_STAGEPACK_TENSOR_SHARED_GATE_UP: weights->shared_gate_up_bf16 = payload; break;
	case SPARK_LAGUNA_STAGEPACK_TENSOR_SHARED_DOWN: weights->shared_down_bf16 = payload; break;
	default: SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkLagunaPackAssign(
	SparkLagunaModuleState *state,
	const SparkLagunaStagePackEntry *entry,
	const void *payload,
	const void *scale)
{
	if ( entry->layer_index != SPARK_LAGUNA_STAGEPACK_GLOBAL_LAYER )
		return(SparkLagunaPackAssignLayer(&state->layers[entry->layer_index - state->first_layer_index],entry,payload,scale));
	switch ( entry->tensor_kind )
	{
	case SPARK_LAGUNA_STAGEPACK_TENSOR_EMBEDDING: state->embedding_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_LAGUNA_STAGEPACK_TENSOR_FINAL_NORM: state->final_norm_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_LAGUNA_STAGEPACK_TENSOR_LM_HEAD: state->lm_head_bf16 = payload; return(SPARK_STATUS_OK);
	default: SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
}

typedef struct SparkLagunaManifestContext
{
	const SparkLagunaStagePackEntry *entries;
	uint32_t count;
} SparkLagunaManifestContext;

static SparkStatus SparkLagunaManifestPlane(const SparkWeightdManifest *manifest,const SparkLagunaStagePackEntry *entry,uint32_t plane)
{
	const SparkWeightdRangeGroup *group;
	const SparkWeightdRange *range;
	uint64_t offset,bytes,per;
	uint32_t expert,index,kind;
	offset = plane == 0u ? entry->payload_offset : entry->scale_offset;
	bytes = plane == 0u ? entry->payload_bytes : entry->scale_bytes;
	if ( entry->group_count == 0u || bytes == 0u || bytes % entry->group_count != 0u )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	per = (bytes / entry->group_count);
	kind = ((entry->tensor_kind * 2u) + plane);
	for (expert=0u; expert<entry->group_count; expert++)
	{
		group = SparkWeightdManifestFind(manifest,entry->layer_index,expert);
		if ( group == 0 )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		for (index=0u; index<group->range_count; index++)
		{
			range = &manifest->ranges[group->first_range + index];
			if ( range->kind == kind )
				break;
		}
		if ( index == group->range_count || range->offset != (offset + ((uint64_t)expert * per)) || range->bytes != per )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkLagunaManifestCheck(const SparkWeightdManifest *manifest,void *opaque)
{
	const SparkLagunaManifestContext *context = (const SparkLagunaManifestContext *)opaque;
	const SparkLagunaStagePackEntry *entry;
	SparkStatus status;
	uint64_t expected = 0u;
	uint32_t index,plane;
	for (index=0u; index<context->count; index++)
	{
		entry = &context->entries[index];
		if ( entry->tensor_kind != SPARK_LAGUNA_STAGEPACK_TENSOR_EXPERT_GATE_UP && entry->tensor_kind != SPARK_LAGUNA_STAGEPACK_TENSOR_EXPERT_DOWN )
			continue;
		if ( entry->weight_codec != SPARK_WEIGHT_CODEC_FP8_E4M3 )
			SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
		for (plane=0u; plane<2u; plane++)
		{
			status = SparkLagunaManifestPlane(manifest,entry,plane);
			if ( status != SPARK_STATUS_OK )
				return(status);
			expected += entry->group_count;
		}
	}
	return(expected == manifest->range_count ? SPARK_STATUS_OK : SPARK_STATUS_SCHEMA_ERROR);
}

static SparkStatus SparkLagunaLazyOpen(SparkLagunaModuleState *state,const char *path,uint64_t bytes,const SparkLagunaStagePackEntry *entries,uint32_t count)
{
	SparkWeightdLazyAttachRequest request;
	SparkLagunaManifestContext context = {entries,count};
	SparkStatus status;
	const char *digest;
	uint64_t spine_budget;
	status = SparkWeightdAttachRequested();
	if ( status == SPARK_STATUS_BUSY )
		return(SPARK_STATUS_OK);
	if ( status != SPARK_STATUS_OK )
		return(status);
	memset(&request,0,sizeof(request));
	digest = getenv(SPARK_WEIGHTD_ATTACH_ENV_SHA256);
	if ( digest == 0 || strlen(digest) != 64u || strlen(path) >= sizeof(request.pack_path) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memcpy(request.identity.pack_sha256,digest,65u);
	(void)snprintf(request.identity.model,sizeof(request.identity.model),"%s",SPARK_LAGUNA_MODULE_TAG);
	(void)snprintf(request.identity.revision,sizeof(request.identity.revision),"%s",state->model_revision);
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = bytes;
	request.identity.topology = state->tp_degree;
	memcpy(request.pack_path,path,strlen(path) + 1u);
	status = SparkStageModuleEnvironmentUnsigned64(SPARK_LAGUNA_MODULE_TAG,"SPARK_WEIGHTD_EXPERT_POOL_BYTES",1u,UINT64_MAX,&request.expert_pool_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned64(SPARK_LAGUNA_MODULE_TAG,"SPARK_WEIGHTD_SPINE_BUDGET_BYTES",1u,UINT64_MAX,&spine_budget);
	if ( status == SPARK_STATUS_OK )
		status = SparkWeightdLazyPackCreateChecked(getenv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET),&request,spine_budget,SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS,SparkLagunaManifestCheck,&context,&state->lazy_pack);
	return(status);
}

static SparkStatus SparkLagunaPackLoadEntry(
	SparkLagunaModuleState *state,
	FILE *file,
	const SparkLagunaStagePackEntry *entry)
{
	void *payload,*scale;
	SparkStatus status;
	payload = 0;
	scale = 0;
	if ( state->lazy_pack != 0 )
	{
		if ( entry->tensor_kind == SPARK_LAGUNA_STAGEPACK_TENSOR_EXPERT_GATE_UP || entry->tensor_kind == SPARK_LAGUNA_STAGEPACK_TENSOR_EXPERT_DOWN )
			return(SparkLagunaPackAssign(state,entry,0,0));
		status = SparkWeightdLazyPackSlice(state->lazy_pack,entry->payload_offset,entry->payload_bytes,(const void **)&payload);
		if ( status == SPARK_STATUS_OK && entry->scale_bytes != 0u )
			status = SparkWeightdLazyPackSlice(state->lazy_pack,entry->scale_offset,entry->scale_bytes,(const void **)&scale);
		if ( status == SPARK_STATUS_OK )
			status = SparkLagunaPackAssign(state,entry,payload,scale);
		return(status);
	}
	status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,entry->payload_offset,entry->payload_bytes,&payload);
	if ( status == SPARK_STATUS_OK && entry->scale_bytes != 0u )
		status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,entry->scale_offset,entry->scale_bytes,&scale);
	if ( status == SPARK_STATUS_OK )
		status = SparkLagunaPackAssign(state,entry,payload,scale);
	return(status);
}

static uint64_t SparkLagunaExpectedLayerMask(
	const SparkLagunaModuleState *state,
	uint32_t layer_index)
{
	SparkLagunaStagePackTensorShape shape;
	uint64_t mask;
	uint32_t kind;
	mask = 0u;
	for (kind=SPARK_LAGUNA_STAGEPACK_TENSOR_ATTN_INPUT_NORM; kind<SPARK_LAGUNA_STAGEPACK_TENSOR_KIND_COUNT; kind++)
		if ( SparkLagunaStagePackExpectedShape(kind,layer_index,state->expert_weight_codec,state->tp_degree,&shape) == 0 )
			mask |= UINT64_C(1) << kind;
	return(mask);
}

static uint64_t SparkLagunaExpectedGlobalMask(const SparkLagunaModuleState *state)
{
	uint64_t mask;
	mask = 0u;
	if ( state->owns_embedding != 0u )
		mask |= UINT64_C(1) << SPARK_LAGUNA_STAGEPACK_TENSOR_EMBEDDING;
	if ( state->owns_final_head != 0u )
		mask |= (UINT64_C(1) << SPARK_LAGUNA_STAGEPACK_TENSOR_FINAL_NORM) | (UINT64_C(1) << SPARK_LAGUNA_STAGEPACK_TENSOR_LM_HEAD);
	return(mask);
}

static SparkStatus SparkLagunaPackValidateInventory(const SparkLagunaModuleState *state)
{
	uint32_t local;
	if ( state->global_seen != SparkLagunaExpectedGlobalMask(state) )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	for (local=0u; local<state->layer_count; local++)
		if ( state->layer_seen[local] != SparkLagunaExpectedLayerMask(state,state->first_layer_index + local) )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkLagunaPackLoad(
	SparkLagunaModuleState *state,
	const char *path)
{
	SparkLagunaStagePackHeader header;
	SparkLagunaStagePackEntry entries[SPARK_LAGUNA_STAGEPACK_MAX_TENSOR_COUNT];
	SparkLagunaStagePackTensorShape shape;
	FILE *file;
	uint64_t file_bytes;
	uint32_t index;
	SparkStatus status;
	file = fopen(path,"rb");
	if ( file == 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	memset(&header,0,sizeof(header));
	memset(entries,0,sizeof(entries));
	status = SparkLagunaPackFileSize(file,&file_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModulePackRead(SPARK_LAGUNA_MODULE_TAG,file,0u,&header,sizeof(header));
	if ( status == SPARK_STATUS_OK )
		status = SparkLagunaPackValidateHeader(state,&header,file_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModulePackRead(SPARK_LAGUNA_MODULE_TAG,file,header.directory_offset,entries,(uint64_t)header.tensor_count * sizeof(entries[0]));
	for (index=0u; status==SPARK_STATUS_OK && index<header.tensor_count; index++)
	{
		status = SparkLagunaPackValidateEntryGeometry(state,&header,&entries[index],&shape);
		if ( status == SPARK_STATUS_OK )
			SparkLagunaPackMarkSeen(state,&entries[index]);
	}
	if ( status == SPARK_STATUS_OK && state->dflash_sections != 0u )
		fprintf(stderr,"laguna pack carries %u DFlash payload sections (%llu bytes); recorded and skipped\n",
			state->dflash_sections,(unsigned long long)state->dflash_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkLagunaPackValidateRanges(entries,header.tensor_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkLagunaPackValidateInventory(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkLagunaLazyOpen(state,path,file_bytes,entries,header.tensor_count);
	for (index=0u; status==SPARK_STATUS_OK && index<header.tensor_count; index++)
		status = SparkLagunaPackLoadEntry(state,file,&entries[index]);
	if ( fclose(file) != 0 && status == SPARK_STATUS_OK )
		status = SPARK_STATUS_IO_ERROR;
	return(status);
}

static SparkStatus SparkLagunaAllocateBytes(
	SparkLagunaModuleState *state,
	uint64_t count,
	uint64_t width,
	uint64_t element_bytes,
	void **pointer)
{
	uint64_t bytes;
	if ( state == 0 || pointer == 0 || count == 0u || width == 0u || element_bytes == 0u || count > UINT64_MAX / width || count * width > UINT64_MAX / element_bytes )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	bytes = count * width * element_bytes;
	return(SparkStageModuleDeviceAllocate(&state->ledger,bytes,pointer));
}

static SparkStatus SparkLagunaAllocateRows(
	SparkLagunaModuleState *state,
	uint64_t rows,
	uint64_t columns,
	void **pointer)
{
	return(SparkLagunaAllocateBytes(state,rows,columns,sizeof(uint16_t),pointer));
}

static SparkStatus SparkLagunaAllocateSlotHost(SparkLagunaExecutionSlot *slot)
{
	uint32_t *cursor;
	uint64_t rows,words,bytes;
	cudaError_t error;
	if ( slot == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	rows = SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT;
	words = (rows * 4u) + SPARK_LAGUNA_KV_ACCESS_ERROR_WORD_COUNT + SPARK_LAGUNA_MODEL_MOE_EXPERT_COUNT + 1u;
	bytes = words * sizeof(uint32_t);
	error = cudaHostAlloc(&slot->host_staging,bytes,cudaHostAllocPortable);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_LAGUNA_MODULE_TAG,error,"host_staging"));
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
	cursor += SPARK_LAGUNA_KV_ACCESS_ERROR_WORD_COUNT;
	slot->host_group_row_offset = cursor;
	error = cudaEventCreateWithFlags((cudaEvent_t *)&slot->route_ready_event,cudaEventDisableTiming);
	return(SparkStageModuleCudaStatus(SPARK_LAGUNA_MODULE_TAG,error,"route_ready_event"));
}

static void SparkLagunaReleaseSlotHost(SparkLagunaModuleState *state)
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
	}
}

static SparkStatus SparkLagunaAllocateSlotMetadata(
	SparkLagunaModuleState *state,
	SparkLagunaExecutionSlot *slot)
{
	SparkStatus status;
	status = SparkLagunaAllocateBytes(state,state->execution_row_capacity,1u,sizeof(uint32_t),(void **)&slot->token_ids);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateBytes(state,state->execution_row_capacity,1u,sizeof(uint32_t),(void **)&slot->resident_slots);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateBytes(state,state->execution_row_capacity,1u,sizeof(uint32_t),(void **)&slot->positions);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateBytes(state,state->resident_sequence_capacity,1u,sizeof(uint32_t),(void **)&slot->context_lengths);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateBytes(state,2u,1u,sizeof(uint32_t),(void **)&slot->dense_row_offset);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateBytes(state,2u,1u,sizeof(uint32_t),(void **)&slot->dense_tile_prefix);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateBytes(state,1u,sizeof(uint32_t) * 6u,1u,&slot->kv_access_error);
	return(status);
}

static SparkStatus SparkLagunaAllocateSlotHidden(
	SparkLagunaModuleState *state,
	SparkLagunaExecutionSlot *slot)
{
	uint64_t rows;
	SparkStatus status;
	rows = state->execution_row_capacity;
	status = SparkLagunaAllocateRows(state,rows,SPARK_LAGUNA_MODEL_HIDDEN_DIMENSION,(void **)&slot->hidden_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateRows(state,rows,SPARK_LAGUNA_MODEL_HIDDEN_DIMENSION,(void **)&slot->residual_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateRows(state,rows,SPARK_LAGUNA_MODEL_HIDDEN_DIMENSION,(void **)&slot->normed_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateRows(state,rows,LAGUNA_MAX_QKV_ROWS,(void **)&slot->qkv_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateRows(state,rows,SPARK_LAGUNA_MODEL_TP8_Q_HEAD_COUNT_SLIDING * SPARK_LAGUNA_MODEL_ATTENTION_HEAD_DIMENSION,(void **)&slot->q_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateRows(state,rows,SPARK_LAGUNA_MODEL_TP8_KV_HEAD_COUNT * SPARK_LAGUNA_MODEL_ATTENTION_HEAD_DIMENSION,(void **)&slot->k_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateRows(state,rows,SPARK_LAGUNA_MODEL_TP8_KV_HEAD_COUNT * SPARK_LAGUNA_MODEL_ATTENTION_HEAD_DIMENSION,(void **)&slot->v_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateRows(state,rows,SPARK_LAGUNA_MODEL_TP8_Q_HEAD_COUNT_SLIDING * SPARK_LAGUNA_MODEL_ATTENTION_HEAD_DIMENSION,(void **)&slot->attention_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateRows(state,rows,SPARK_LAGUNA_MODEL_HIDDEN_DIMENSION,(void **)&slot->attention_out_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateRows(state,rows,SPARK_LAGUNA_MODEL_TP8_Q_HEAD_COUNT_SLIDING,(void **)&slot->gate_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateBytes(state,rows,SPARK_LAGUNA_MODEL_SLIDING_WINDOW,sizeof(uint32_t),(void **)&slot->window_positions);
	return(status);
}

static SparkStatus SparkLagunaAllocateSlotMlp(
	SparkLagunaModuleState *state,
	SparkLagunaExecutionSlot *slot)
{
	uint64_t rows;
	SparkStatus status;
	rows = state->execution_row_capacity;
	status = SparkLagunaAllocateRows(state,rows,SPARK_LAGUNA_MODEL_MOE_ROUTED_GATE_UP_DIMENSION,(void **)&slot->gate_up_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateRows(state,rows,SPARK_LAGUNA_MODEL_MOE_INTERMEDIATE_DIMENSION,(void **)&slot->intermediate_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateRows(state,rows,SPARK_LAGUNA_MODEL_HIDDEN_DIMENSION,(void **)&slot->expert_out_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateRows(state,rows,SPARK_LAGUNA_MODEL_HIDDEN_DIMENSION,(void **)&slot->shared_out_bf16);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateBytes(state,rows,SPARK_LAGUNA_MODEL_MOE_EXPERT_COUNT,sizeof(float),(void **)&slot->router_logits_f32);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateBytes(state,rows * SPARK_LAGUNA_MODEL_MOE_TOP_K,1u,sizeof(uint32_t),(void **)&slot->route_expert);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateBytes(state,rows * SPARK_LAGUNA_MODEL_MOE_TOP_K,1u,sizeof(float),(void **)&slot->route_weight);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateBytes(state,rows * SPARK_LAGUNA_MODEL_MOE_TOP_K,1u,sizeof(uint32_t),(void **)&slot->route_source_token);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateBytes(state,rows * SPARK_LAGUNA_MODEL_MOE_TOP_K,1u,sizeof(uint32_t),(void **)&slot->route_packed_row);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateBytes(state,SPARK_LAGUNA_MODEL_MOE_EXPERT_COUNT + 1u,1u,sizeof(uint32_t),(void **)&slot->group_row_offset);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateBytes(state,2u * SPARK_LAGUNA_MODEL_MOE_EXPERT_COUNT,1u,sizeof(uint32_t),(void **)&slot->group_tile_prefix_w1);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateBytes(state,2u * SPARK_LAGUNA_MODEL_MOE_EXPERT_COUNT,1u,sizeof(uint32_t),(void **)&slot->group_tile_prefix_w2);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateBytes(state,rows,LAGUNA_HEAD_TILE_COUNT,sizeof(float),(void **)&slot->head_candidate_score);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateBytes(state,rows,LAGUNA_HEAD_TILE_COUNT,sizeof(uint32_t),(void **)&slot->head_candidate_token);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateBytes(state,rows,1u,sizeof(uint32_t),(void **)&slot->output_token);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateBytes(state,rows,1u,sizeof(float),(void **)&slot->output_score);
	if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateBytes(state,rows,1u,sizeof(uint64_t),(void **)&slot->head_maxloc_u64);
	return(status);
}

static SparkStatus SparkLagunaAllocateSlots(SparkLagunaModuleState *state)
{
	uint32_t index;
	SparkStatus status;
	status = SPARK_STATUS_OK;
	for (index=0u; status==SPARK_STATUS_OK && index<state->pipeline_slot_count; index++)
	{
		state->slots[index].stream = state->execution_stream;
		status = SparkLagunaAllocateSlotHost(&state->slots[index]);
		if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateSlotMetadata(state,&state->slots[index]);
		if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateSlotHidden(state,&state->slots[index]);
		if ( status == SPARK_STATUS_OK ) status = SparkLagunaAllocateSlotMlp(state,&state->slots[index]);
	}
	return(status);
}


static SparkStatus SparkLagunaBuildPageTable(SparkLagunaModuleState *state)
{
	uint64_t entries;
	SparkStatus status;
	cudaError_t error;
	state->pages_per_sequence = SparkCeilDivU32(state->max_sequence_positions,64u);
	if ( state->pages_per_sequence == 0u || state->resident_sequence_capacity > UINT32_MAX / state->pages_per_sequence )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->page_count = state->resident_sequence_capacity * state->pages_per_sequence;
	entries = state->page_count;
	state->page_table_shadow = (uint32_t *)malloc(entries * sizeof(uint32_t));
	if ( state->page_table_shadow == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	memset(state->page_table_shadow,0xff,entries * sizeof(uint32_t));
	status = SparkLagunaAllocateBytes(state,entries,1u,sizeof(uint32_t),(void **)&state->page_table);
	if ( status == SPARK_STATUS_OK )
	{
		error = cudaMemset(state->page_table,0xff,entries * sizeof(uint32_t));
		status = SparkStageModuleCudaStatus(SPARK_LAGUNA_MODULE_TAG,error,"page_table");
	}
	return(status);
}

static SparkStatus SparkLagunaDevicePageCopy(
	void *context,
	uint32_t direction,
	uintptr_t device_address,
	void *host_address,
	uint64_t bytes)
{
	SparkLagunaModuleState *state;
	cudaError_t error;
	state = (SparkLagunaModuleState *)context;
	if ( state == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( direction == SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST )
		error = cudaMemcpy(host_address,(const void *)device_address,(size_t)bytes,cudaMemcpyDeviceToHost);
	else if ( direction == SPARK_KV_PAGE_STORE_COPY_HOST_TO_DEVICE )
		error = cudaMemcpy((void *)device_address,host_address,(size_t)bytes,cudaMemcpyHostToDevice);
	else
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkStageModuleCudaStatus(SPARK_LAGUNA_MODULE_TAG,error,"kv_page_copy"));
}

static SparkStatus SparkLagunaPageCopy(
	void *context,
	uint32_t direction,
	uintptr_t device_address,
	void *host_address,
	uint64_t bytes)
{
	SparkLagunaModuleState *state;
	SparkKvLayeredPageLayout layout;
	uint64_t offset,packed_page_bytes;
	state = (SparkLagunaModuleState *)context;
	if ( state == 0 || state->page_count == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	layout.device_base = (uintptr_t)state->kv_cache;
	layout.layer_stride_bytes = state->kv_layer_stride_bytes;
	layout.layer_count = state->layer_count;
	layout.page_count = state->page_count;
	if ( layout.layer_count == 0u || layout.layer_stride_bytes % layout.page_count != 0u || layout.layer_stride_bytes > UINT64_MAX / layout.layer_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	layout.device_bytes = layout.layer_stride_bytes * layout.layer_count;
	layout.layer_page_bytes = layout.layer_stride_bytes / layout.page_count;
	packed_page_bytes = layout.layer_page_bytes * layout.layer_count;
	if ( device_address < layout.device_base || device_address - layout.device_base >= layout.device_bytes || packed_page_bytes == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	offset = device_address - layout.device_base;
	if ( offset % packed_page_bytes != 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkKvPageStoreCopyLayered(&layout,direction,(uint32_t)(offset / packed_page_bytes),host_address,bytes,SparkLagunaDevicePageCopy,state));
}

static SparkStatus SparkLagunaBackingCapacity(SparkLagunaModuleState *state,uint64_t kv_page_bytes)
{
	uint64_t total;
	if ( state->page_count == 0u || state->resident_sequence_capacity == 0u || kv_page_bytes == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	total = kv_page_bytes;
	if ( total > INT64_MAX / state->page_count )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	total *= state->page_count;
	if ( state->kv_backing_maximum_bytes != 0u && state->kv_backing_maximum_bytes < total )
	{
		fprintf(stderr,"laguna cache backing budget insufficient: need %llu bytes for %u pages, configured %llu\n",(unsigned long long)total,state->page_count,(unsigned long long)state->kv_backing_maximum_bytes);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	return(SPARK_STATUS_OK);
}


static SparkStatus SparkLagunaKvInitialize(SparkLagunaModuleState *state)
{
	SparkKvModelTable table;
	uint64_t block_bytes,payload_bytes;
	uint64_t lane_page_entries;
	SparkStatus status;
	if ( pthread_mutex_init(&state->kv_mutex,0) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	state->kv_mutex_initialized = 1u;
	if ( state->layer_count == 0u )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	block_bytes = (uint64_t)SPARK_LAGUNA_MODEL_KV_PAGE_SLOTS *
		(uint64_t)state->layer_count * SPARK_LAGUNA_MODEL_KV_SLOT_BYTES_TP8;
	payload_bytes = block_bytes;
	status = SparkLagunaBackingCapacity(state,payload_bytes);
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
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( state->kv_blocks == 0 || state->kv_resident_slot_logical_block_indices == 0 || state->kv_entries == 0 || state->kv_sequences == 0 || state->kv_hash_bucket_heads == 0 || state->kv_entry_indices_by_logical_page == 0 || state->kv_page_staging == 0 || state->kv_lane_logical_pages == 0 || state->kv_lane_transactions == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->kv_transactions.cache = &state->kv_page_cache;
	state->kv_transactions.lanes = state->kv_lane_transactions;
	state->kv_transactions.logical_pages = state->kv_lane_logical_pages;
	state->kv_transactions.physical_pages = state->kv_lane_physical_pages;
	state->kv_transactions.page_capacity = state->pages_per_sequence;

	memset(&table,0,sizeof(table));
	table.abi_version = SPARK_KV_MODEL_TABLE_ABI_VERSION;
	table.descriptor_bytes = SPARK_KV_MODEL_TABLE_BYTES;
	SparkLagunaKvFillCapacityRequest(&table.capacity_request);
	table.capacity_request.layer_count = state->layer_count;

	table.arena_configuration.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	table.arena_configuration.descriptor_bytes = SPARK_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
	table.arena_configuration.logical_block_count = state->page_count;
	table.arena_configuration.block_token_count = SPARK_LAGUNA_MODEL_KV_PAGE_SLOTS;
	table.arena_configuration.resident_block_capacity = state->page_count;
	table.arena_configuration.layer_count = state->layer_count;
	table.arena_configuration.kv_head_count = 1u;
	table.arena_configuration.head_dim =
		2u * SPARK_LAGUNA_MODEL_ATTENTION_HEAD_DIMENSION;
	table.arena_configuration.bytes_per_scalar = SPARK_LAGUNA_MODEL_KV_BITS / 8u;
	table.arena_configuration.key_device_base = state->kv_cache;
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
			"/tmp/sparkpipe_laguna_kv_%s",state->model_revision);
		mkdir(state->kv_backing_default,0700);
		table.page_store_config.backing_path = state->kv_backing_default;
	}
	table.page_store_config.maximum_backing_bytes = state->page_count * payload_bytes;
	table.page_store_config.staging_address = state->kv_page_staging;
	table.page_store_config.staging_bytes = payload_bytes;
	table.page_store_config.copy_function = SparkLagunaPageCopy;
	table.page_store_config.copy_context = state;

	table.sequence_capacity = state->resident_sequence_capacity;
	table.entry_capacity = state->page_count;
	table.hash_bucket_count = state->page_count;
	table.entries = state->kv_entries;
	table.sequences = state->kv_sequences;
	table.hash_bucket_heads = state->kv_hash_bucket_heads;
	table.entry_indices_by_logical_page = state->kv_entry_indices_by_logical_page;
	table.model_id = "laguna";
	table.model_revision = state->model_revision;
	table.cache_layout_fingerprint = "kv-bf16-full-gqa-layer-major-gather-v1";

	status = SparkKvBackendInitialize(&table,&state->kv_arena,&state->kv_page_cache,&state->kv_page_store);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( state->kv_arena.key_block_stride_bytes != block_bytes ||
		state->kv_arena.logical_block_count != state->page_count ||
		state->kv_layer_stride_bytes == 0u ||
		block_bytes != ( state->kv_layer_stride_bytes /
				(uint64_t)state->page_count ) *
			(uint64_t)state->layer_count ||
		(uint64_t)state->page_count * block_bytes !=
			state->kv_layer_stride_bytes * (uint64_t)state->layer_count )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkLagunaAllocateCaches(SparkLagunaModuleState *state)
{
	uint64_t main_page_bytes;
	uint64_t main_total;
	SparkStatus status;
	status = SparkLagunaBuildPageTable(state);
	main_page_bytes = (uint64_t)SPARK_LAGUNA_MODEL_KV_PAGE_SLOTS * SPARK_LAGUNA_MODEL_KV_SLOT_BYTES_TP8;
	state->kv_layer_stride_bytes = (uint64_t)state->page_count * main_page_bytes;
	if ( status != SPARK_STATUS_OK || state->kv_layer_stride_bytes == 0u )
		return(status == SPARK_STATUS_OK ? SPARK_STATUS_CAPACITY_EXCEEDED : status);
	main_total = state->kv_layer_stride_bytes * (uint64_t)state->layer_count;
	status = SparkStageModuleDeviceAllocate(&state->ledger,main_total,(void **)&state->kv_cache);
	if ( status == SPARK_STATUS_OK )
		status = SparkLagunaKvInitialize(state);
	return(status);
}

static SparkStatus SparkLagunaAdmissionPredicate(
	void *context,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	SparkLagunaModuleState *state = (SparkLagunaModuleState *)context;
	SparkStatus status;
	uint32_t lane,slot;
	if ( state == 0 || request == 0 || decision == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( pthread_mutex_lock(&state->kv_mutex) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = request->control_generation < state->control_generation ? SPARK_STATUS_VALIDATION_FAILED : SparkKvLaneTransactionsAdmit(&state->kv_transactions,request);
	if ( status == SPARK_STATUS_OK )
		state->control_generation = request->control_generation;
	if ( status == SPARK_STATUS_OK && (request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE) != 0u )
		for (lane=0u; lane<request->cache_lane_count; lane++)
		{
			slot = request->cache_lanes[lane].resident_sequence_slot;
			atomic_store_explicit(&state->lane_bound[slot],0u,memory_order_release);
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

static uint32_t SparkLagunaRoundMajorWaveRows(
	const SparkLagunaModuleState *state,
	const SparkLagunaResidentDecodeStageBatchView *batch,
	uint32_t first_row)
{
	SparkStageModuleClaimedLaneContext lanes;
	if ( state == 0 || batch == 0 || batch->active_sequence_count == 0u )
		return(0u);
	lanes.index_states = state->lane_states;
	lanes.index_capacity = state->resident_sequence_capacity;
	return(SparkRowLayoutRoundMajorWaveRowCount(first_row,batch->row_count,batch->row_resident_slots,SparkStageModuleClaimedLaneOrdinal,&lanes));
}

static SparkStatus SparkLagunaValidateRoundMajor(
	const SparkLagunaModuleState *state,
	const SparkLagunaResidentDecodeStageBatchView *batch)
{
	uint32_t ordinals[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t counts[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t last_rows[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	SparkRowLayoutDirectLaneContext lanes;
	SparkStatus status;
	if ( state == 0 || batch == 0 || batch->row_count < batch->active_sequence_count || state->resident_sequence_capacity > SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkRowLayoutDirectLaneMapInitialize(&lanes,ordinals,state->resident_sequence_capacity,batch->row_resident_slots,batch->active_sequence_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkRowLayoutValidateRoundMajor(batch->row_count,batch->active_sequence_count,batch->row_resident_slots,SparkRowLayoutDirectLaneOrdinal,&lanes,counts,last_rows));
}

static uint32_t SparkLagunaPrefixRestorePending(const SparkKvLaneTransaction *owner)
{
	return(owner != 0 && (owner->mutation_flags & SPARK_KV_PAGE_CACHE_MUTATION_BOUND_SEQUENCE) != 0u && (owner->lane.flags & SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX) != 0u && owner->lane.sequence_position != 0u);
}

static SparkStatus SparkLagunaLoadSequenceContinuity(const SparkLagunaModuleState *state,const SparkLagunaResidentDecodeStageBatchView *batch,uint8_t *bound,uint64_t *sequence_ids,uint64_t *next_positions)
{
	const SparkKvLaneTransaction *owner;
	uint32_t lane,slot;
	if ( state->kv_lane_transactions == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (lane=0u; lane<batch->active_sequence_count; lane++)
	{
		slot = batch->row_resident_slots[lane];
		if ( slot >= state->resident_sequence_capacity )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		bound[lane] = atomic_load_explicit(&state->lane_bound[slot],memory_order_acquire);
		sequence_ids[lane] = atomic_load_explicit(&state->lane_sequence_ids[slot],memory_order_acquire);
		next_positions[lane] = atomic_load_explicit(&state->lane_next_positions[slot],memory_order_acquire);
		owner = &state->kv_lane_transactions[slot];
		if ( SparkLagunaPrefixRestorePending(owner) != 0u )
		{
			if ( owner->phase != SPARK_KV_LANE_TRANSACTION_COMMITTED || owner->lane.sequence_id != batch->row_sequence_ids[lane] || owner->lane.sequence_position != batch->row_positions[lane] )
				SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
			bound[lane] = 1u;
			sequence_ids[lane] = owner->lane.sequence_id;
			next_positions[lane] = owner->lane.sequence_position;
		}
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkLagunaValidateSequenceContinuity(
	const SparkLagunaModuleState *state,
	const SparkLagunaResidentDecodeStageBatchView *batch,
	uint8_t *bound,
	uint64_t *sequence_ids,
	uint64_t *next_positions)
{
	uint8_t touched[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT] = {0u};
	uint64_t position,sequence;
	uint32_t lane,row,slot;
	SparkStatus status;
	status = SparkLagunaLoadSequenceContinuity(state,batch,bound,sequence_ids,next_positions);
	if ( status != SPARK_STATUS_OK )
		return(status);
	for (row=0u; row<batch->row_count; row++)
	{
		slot = batch->row_resident_slots[row];
		status = SparkStageModuleIndexClaimOrdinal(state->lane_states,state->resident_sequence_capacity,slot,&lane);
		position = batch->row_positions[row];
		sequence = batch->row_sequence_ids[row];
		if ( status != SPARK_STATUS_OK || lane >= batch->active_sequence_count || batch->row_resident_slots[lane] != slot || position >= state->max_sequence_positions )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		if ( position == 0u )
		{
			if ( touched[lane] != 0u )
				SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
			bound[lane] = 1u;
			sequence_ids[lane] = sequence;
			next_positions[lane] = 1u;
		}
		else
		{
			if ( bound[lane] == 0u || sequence_ids[lane] != sequence || next_positions[lane] != position )
				SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
			next_positions[lane] = position + 1u;
		}
		touched[lane] = 1u;
	}
	return(SPARK_STATUS_OK);
}

typedef struct SparkLagunaClaimedContinuityContext
{
	SparkLagunaModuleState *state;
	const SparkLagunaResidentDecodeStageBatchView *batch;
	uint8_t *bound;
	uint64_t *sequence_ids;
	uint64_t *next_positions;
} SparkLagunaClaimedContinuityContext;

static SparkStatus SparkLagunaPrepareClaimedContinuity(void *prepare_context)
{
	SparkLagunaClaimedContinuityContext *context;
	SparkStatus status;
	context = (SparkLagunaClaimedContinuityContext *)prepare_context;
	if ( pthread_mutex_lock(&context->state->kv_mutex) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = SparkLagunaValidateSequenceContinuity(context->state,context->batch,context->bound,context->sequence_ids,context->next_positions);
	(void)pthread_mutex_unlock(&context->state->kv_mutex);
	return(status);
}

static SparkStatus SparkLagunaValidateFrameBuffers(
	const SparkLagunaModuleState *state,
	const SparkModelDriverFrame *frame,
	uint32_t row_count)
{
	const SparkModelDriverBuffer *buffer;
	if ( state->owns_final_head == 0u )
		return(frame->buffer_count == 0u ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
	if ( frame->buffer_count != 1u || frame->buffers == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	buffer = &frame->buffers[0];
	if ( buffer->flags != SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE || buffer->address == 0 || buffer->bytes < (uint64_t)row_count * sizeof(uint32_t) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkLagunaValidateFrame(
	const SparkLagunaModuleState *state,
	const SparkModelDriverFrame *frame,
	const SparkLagunaResidentDecodeStageFrameContext **context_out)
{
	const SparkLagunaResidentDecodeStageFrameContext *context;
	const SparkLagunaResidentDecodeStageBatchView *batch;
	uint32_t expected_flags,prefill;
	uint64_t boundary_bytes;
	SparkStatus status;
	if ( state == 0 || frame == 0 || context_out == 0 || frame->user_context == 0 || frame->execution_stream != state->execution_stream || frame->completion_function == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	context = (const SparkLagunaResidentDecodeStageFrameContext *)frame->user_context;
	if ( context->abi_version != SPARK_LAGUNA_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION || context->descriptor_bytes != sizeof(*context) || context->reserved0 != 0u || (context->flags & ~SPARK_LAGUNA_RESIDENT_DECODE_STAGE_FRAME_KNOWN_FLAGS) != 0u || context->batch == 0 )
		SPARK_FAIL(SPARK_STATUS_ABI_MISMATCH);
	batch = context->batch;
	if ( batch->abi_version != SPARK_LAGUNA_RESIDENT_DECODE_STAGE_BATCH_VIEW_ABI_VERSION || batch->descriptor_bytes != sizeof(*batch) || batch->row_count == 0u || batch->row_count > SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT || batch->active_sequence_count == 0u || batch->active_sequence_count > state->resident_sequence_capacity || batch->row_resident_slots == 0 || batch->row_positions == 0 || batch->row_sequence_ids == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	prefill = (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u ? 1u : 0u;
	if ( prefill == 0u && batch->row_count != batch->active_sequence_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( frame->active_slot_count != batch->active_sequence_count || frame->new_token_count != batch->row_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->owns_embedding != 0u && batch->token_ids == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	expected_flags = prefill != 0u ? SPARK_LAGUNA_RESIDENT_DECODE_STAGE_FRAME_FLAG_PREFILL : 0u;
	expected_flags |= state->owns_embedding == 0u ? SPARK_LAGUNA_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_INPUT : 0u;
	expected_flags |= state->owns_final_head == 0u ? SPARK_LAGUNA_RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_OUTPUT : 0u;
	if ( context->flags != expected_flags )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	boundary_bytes = (uint64_t)batch->row_count * SPARK_LAGUNA_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_COUNT * SPARK_LAGUNA_RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_BYTES;
	if ( (state->owns_embedding == 0u && (context->hidden_input_bf16 == 0 || context->hidden_input_bytes < boundary_bytes)) || (state->owns_embedding != 0u && (context->hidden_input_bf16 != 0 || context->hidden_input_bytes != 0u)) || (state->owns_final_head == 0u && (context->hidden_output_bf16 == 0 || context->hidden_output_bytes < boundary_bytes)) || (state->owns_final_head != 0u && (context->hidden_output_bf16 != 0 || context->hidden_output_bytes != 0u)) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkLagunaValidateRoundMajor(state,batch);
	if ( status == SPARK_STATUS_OK )
		status = SparkLagunaValidateFrameBuffers(state,frame,batch->row_count);
	*context_out = status == SPARK_STATUS_OK ? context : 0;
	return(status);
}

#define SPARK_LAGUNA_TP_COLLECTIVE_CREDITS_PER_SLOT 2u
#define SPARK_LAGUNA_TP_CHAIN_OPERATIONS ((2u * SPARK_LAGUNA_MODEL_LAYER_COUNT + 16u) * SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT)
#define SPARK_LAGUNA_TP_COLLECTIVE_HC_PORT_STRIDE 512u
#define SPARK_LAGUNA_TP_COLLECTIVE_D2A_MAX_PAYLOAD_BYTES 65536u

typedef enum SparkLagunaChainStage
{
	SPARK_LAGUNA_CHAIN_STAGE_BEGIN = 0,
	SPARK_LAGUNA_CHAIN_STAGE_ATTENTION,
	SPARK_LAGUNA_CHAIN_STAGE_REDUCE_ATTENTION,
	SPARK_LAGUNA_CHAIN_STAGE_MLP,
	SPARK_LAGUNA_CHAIN_STAGE_REDUCE_MLP,
	SPARK_LAGUNA_CHAIN_STAGE_HEAD,
	SPARK_LAGUNA_CHAIN_STAGE_REDUCE_HEAD,
	SPARK_LAGUNA_CHAIN_STAGE_FINISH
} SparkLagunaChainStage;

typedef struct SparkLagunaTpChain
{
	SparkLagunaModuleState *state;
	SparkLagunaExecutionSlot *slot;
	uint32_t slot_index;
	SparkModelDriverFrame *frame;
	const SparkLagunaResidentDecodeStageFrameContext *context;
	const SparkLagunaResidentDecodeStageBatchView *batch;
	SparkLagunaCudaWave wave;
	uint32_t first_row;
	uint32_t wave_rows;
	uint32_t next_wave_row;
	uint32_t stage;
	uint32_t next_layer;
	uint32_t active;
	uint32_t tp_op_index;
	uint64_t expert_lease;
	uint32_t expert_lease_begun;
	uint32_t expert_lease_recorded;
	SparkStatus retained_status;
} SparkLagunaTpChain;

static void SparkLagunaTpChainAdvance(void *chain_context,SparkStatus status);
static void CUDART_CB SparkLagunaCompleteAsync(void *context);
static SparkStatus SparkLagunaEnqueueAsyncCompletion(
	SparkLagunaModuleState *state,
	SparkLagunaExecutionSlot *slot,
	uint32_t slot_index);

static void SparkLagunaBuildWave(SparkLagunaTpChain *chain)
{
	SparkLagunaModuleState *state;
	SparkLagunaExecutionSlot *slot;
	const SparkLagunaResidentDecodeStageFrameContext *context;
	SparkLagunaCudaWave *wave;
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
	wave->commit = 1u;
	wave->maximum_context = maximum_context;
	wave->resident_sequence_capacity = state->resident_sequence_capacity;
	wave->max_sequence_positions = state->max_sequence_positions;
	wave->execution_row_capacity = state->execution_row_capacity;
	wave->pages_per_sequence = state->pages_per_sequence;
	wave->owns_embedding = state->owns_embedding;
	wave->owns_final_head = state->owns_final_head;
	wave->boundary_row_offset = chain->first_row;
	wave->host_token_ids = state->owns_embedding != 0u ? slot->host_token_ids + chain->first_row : 0;
	wave->host_resident_slots = slot->host_resident_slots + chain->first_row;
	wave->host_positions = slot->host_positions + chain->first_row;
	wave->hidden_input_bf16 = context->hidden_input_bf16;
	wave->hidden_output_bf16 = context->hidden_output_bf16;
	wave->host_output_token_ids = state->owns_final_head != 0u ? slot->host_output_token_ids + chain->first_row : 0;
	wave->embedding_bf16 = state->embedding_bf16;
	wave->final_norm_bf16 = state->final_norm_bf16;
	wave->lm_head_bf16 = state->lm_head_bf16;
	wave->layers = state->layers;
	wave->lazy_experts = state->lazy_pack != 0 ? 1u : 0u;
	wave->slot = slot;
	wave->yarn_inv_freq = state->yarn_inv_freq;
	wave->kv_cache = state->kv_cache;
	wave->kv_layer_stride_bytes = state->kv_layer_stride_bytes;
	wave->page_table = state->page_table;
	wave->multiprocessor_count = state->multiprocessor_count;
	wave->decode_split_context_threshold = state->decode_split_context_threshold;
}

static SparkStatus SparkLagunaModuleCombineBf16(
	void *combine_context,
	void *destination_device,
	const void *source_device,
	uint32_t active_sequence_count,
	uint32_t hidden_dimension,
	void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkLagunaLaunchAccumAdd((cudaStream_t)cuda_stream,destination_device,source_device,active_sequence_count,hidden_dimension);
	return(SparkStageModuleCudaStatus(SPARK_LAGUNA_MODULE_TAG,error,"tp_all_reduce_sum"));
}

static SparkStatus SparkLagunaModuleCombineDirectBf16(
	void *combine_context,
	void *destination_device,
	const void *const rank_devices[
		SPARK_TP_DEVICE_COLLECTIVE_DIRECT_ALL_TO_ALL_RANK_COUNT],
	uint32_t tp_rank,
	uint32_t active_sequence_count,
	uint32_t hidden_dimension,
	void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkLagunaLaunchDirectSum((cudaStream_t)cuda_stream,destination_device,rank_devices,tp_rank,active_sequence_count,hidden_dimension);
	return(SparkStageModuleCudaStatus(SPARK_LAGUNA_MODULE_TAG,error,"tp_d2d_all_reduce_sum"));
}

static SparkStatus SparkLagunaModuleCombineU64Max(
	void *combine_context,
	uint64_t *destination_device,
	const uint64_t *source_device,
	uint32_t element_count,
	void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkLagunaLaunchAccumU64Max((cudaStream_t)cuda_stream,destination_device,source_device,element_count);
	return(SparkStageModuleCudaStatus(SPARK_LAGUNA_MODULE_TAG,error,"tp_all_reduce_max_u64"));
}

static SparkStatus SparkLagunaModuleInitializeTpCollective(
	SparkLagunaModuleState *state,
	const SparkLagunaResidentDecodeStageNodeContext *context)
{
	SparkTpDeviceCollectiveConfig configuration;
	uint32_t probe_connect_timeout_milli,probe_operation_timeout_milli;
	SparkStatus status;
	if ( state == 0 || context == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->tp_degree == 1u || state->tp_collective_disabled != 0u )
		return(SPARK_STATUS_OK);
	probe_connect_timeout_milli = context->tp_connect_timeout_milli;
	probe_operation_timeout_milli = context->tp_operation_timeout_milli;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	configuration.backend_kind = context->tp_collective_backend_kind;
	configuration.tp_degree = state->tp_degree;
	configuration.tp_rank = state->tp_rank;
	configuration.operation_kind = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	configuration.credit_count = state->pipeline_slot_count * SPARK_LAGUNA_TP_COLLECTIVE_CREDITS_PER_SLOT;
	configuration.local_hidden_dimension = SPARK_LAGUNA_MODEL_HIDDEN_DIMENSION;
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
			SPARK_LAGUNA_TP_COLLECTIVE_D2A_MAX_PAYLOAD_BYTES;
	}
	if ( configuration.backend_kind == SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT )
	{
		configuration.combine_bf16_function = SparkLagunaModuleCombineBf16;
		configuration.combine_u64_max_function = SparkLagunaModuleCombineU64Max;
		configuration.combine_tp4_bf16_function = SparkLagunaModuleCombineDirectBf16;
		configuration.combine_context = state;
	}
	if ( configuration.connect_timeout_milli == 0u || configuration.operation_timeout_milli == 0u || configuration.control_port_base == 0u || configuration.collective_identifier == 0u || configuration.backend_module_path == 0 || configuration.local_host == 0 || configuration.backend_module_path[0] == '\0' || configuration.local_host[0] == '\0' )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration.backend_kind != SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkTpDeviceCollectiveCreate(&configuration,&state->tp_device_collective);
	if ( status != SPARK_STATUS_OK )
		return(status);
	state->tp_device_collective_initialized = 1u;
	if ( state->lazy_pack != 0 &&
	     state->lazy_pack->attached.mesh_send_buffer_addr != 0 )
		status = SparkTpDeviceCollectivePrepareReceiveBf16(
		    &state->tp_device_collective,
		    (void *)(uintptr_t)state->lazy_pack->attached.mesh_send_buffer_addr,
		    0u,0u,0u,0u);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SPARK_STATUS_OK);
}

static void SparkLagunaModuleTpCompletion(
	void *context,
	const SparkTpDeviceCollectiveCompletion *completion)
{
	SparkLagunaTpChain *chain;
	chain = (SparkLagunaTpChain *)context;
	if ( chain == 0 || chain->active == 0u || completion == 0 )
		return;
	SparkLagunaTpChainAdvance(chain,completion->status);
}

static SparkStatus SparkLagunaChainOrdinal(SparkLagunaTpChain *chain,uint32_t operation,uint64_t *ordinal)
{
	SparkLagunaModuleState *state;
	state = chain->state;
	if ( state->tp_device_collective.backend_kind == SPARK_TP_DEVICE_COLLECTIVE_BACKEND_NCCL )
	{
		*ordinal = atomic_fetch_add_explicit(&state->nccl_next_ordinal,1u,memory_order_relaxed);
		return(SPARK_STATUS_OK);
	}
	return(SparkTpChainOrdinal(chain->frame->request_id,state->pipeline_slot_count,SPARK_LAGUNA_TP_COLLECTIVE_CREDITS_PER_SLOT,SPARK_LAGUNA_TP_CHAIN_OPERATIONS,operation,ordinal));
}

static SparkStatus SparkLagunaModuleReduceHidden(SparkLagunaTpChain *chain,void *device_bf16)
{
	uint32_t *op_index;
	SparkStatus ordinal_status;
	SparkLagunaModuleState *state;
	SparkTpDeviceCollectiveSubmission submission;
	uint64_t ordinal;
	state = chain->state;
	if ( state->tp_degree == 1u || state->tp_collective_disabled != 0u )
	{
		SparkLagunaTpChainAdvance(chain,SPARK_STATUS_OK);
		return(SPARK_STATUS_OK);
	}
	if ( state->tp_device_collective_initialized == 0u )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	op_index = &chain->tp_op_index;
	ordinal_status = SparkLagunaChainOrdinal(chain,*op_index,&ordinal);
	if ( ordinal_status != SPARK_STATUS_OK )
		return(ordinal_status);
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = chain->slot_index;
	submission.active_sequence_count = chain->wave_rows;
	submission.logical_sequence_count = chain->batch->active_sequence_count;
	submission.flags = SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission.ordinal = ordinal;
	submission.local_device = device_bf16;
	submission.full_device = device_bf16;
	submission.cuda_stream = chain->slot->stream;
	submission.completion_function = SparkLagunaModuleTpCompletion;
	submission.completion_context = chain;
	{
		SparkStatus submit_status;
		*op_index += 1u;
		submit_status = SparkTpDeviceCollectiveEnqueue(&state->tp_device_collective,&submission,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16);
		if ( submit_status != SPARK_STATUS_OK )
			*op_index -= 1u;
		return(submit_status);
	}
}

static SparkStatus SparkLagunaModuleReduceAttentionOut(SparkLagunaTpChain *chain,void *device_bf16)
{
	return(SparkLagunaModuleReduceHidden(chain,device_bf16));
}

static SparkStatus SparkLagunaModuleReduceHeadMax(SparkLagunaTpChain *chain)
{
	SparkLagunaModuleState *state;
	SparkTpDeviceCollectiveSubmission submission;
	uint64_t ordinal;
	SparkStatus status;
	state = chain->state;
	if ( state->tp_degree == 1u || state->tp_collective_disabled != 0u )
	{
		SparkLagunaTpChainAdvance(chain,SPARK_STATUS_OK);
		return(SPARK_STATUS_OK);
	}
	if ( state->tp_device_collective_initialized == 0u )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = SparkLagunaChainOrdinal(chain,chain->tp_op_index,&ordinal);
	if ( status != SPARK_STATUS_OK )
		return(status);
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = chain->slot_index;
	submission.active_sequence_count = chain->wave_rows;
	submission.logical_sequence_count = chain->batch->active_sequence_count;
	submission.flags = SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission.ordinal = ordinal;
	submission.local_device = chain->slot->head_maxloc_u64;
	submission.full_device = chain->slot->head_maxloc_u64;
	submission.cuda_stream = chain->slot->stream;
	submission.completion_function = SparkLagunaModuleTpCompletion;
	submission.completion_context = chain;
	chain->tp_op_index += 1u;
	status = SparkTpDeviceCollectiveEnqueue(&state->tp_device_collective,&submission,SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64);
	if ( status != SPARK_STATUS_OK )
		chain->tp_op_index -= 1u;
	return(status);
}





static void SparkLagunaTpChainFail(SparkLagunaTpChain *chain,SparkStatus status)
{
	SparkLagunaModuleState *state;
	SparkLagunaAsyncCompletion *async;
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
	SparkLagunaCompleteAsync(async);
	free(chain);
}

static SparkStatus SparkLagunaLazyRelease(SparkLagunaTpChain *chain)
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
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
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

static SparkStatus SparkLagunaLazyRecoverLease(SparkLagunaModuleState *state,uint32_t slot,SparkLagunaTpChain **out)
{
	SparkLagunaTpChain *chain;
	SparkStatus status = SPARK_STATUS_OK;
	*out = 0;
	chain = atomic_exchange_explicit(&state->lazy_retained[slot],0,memory_order_acq_rel);
	if ( chain == 0 )
		SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
	if ( chain->expert_lease != 0u )
		status = SparkLagunaLazyRelease(chain);
	if ( status != SPARK_STATUS_OK )
		atomic_store_explicit(&state->lazy_retained[slot],chain,memory_order_release);
	else
		*out = chain;
	return(status);
}

static void SparkLagunaLazyRetryRetained(void *context)
{
	SparkLagunaModuleState *state = (SparkLagunaModuleState *)context;
	SparkLagunaTpChain *chain;
	uint32_t slot;
	for (slot=0u; slot<state->pipeline_slot_count; slot++)
		if ( SparkLagunaLazyRecoverLease(state,slot,&chain) == SPARK_STATUS_OK )
			SparkLagunaTpChainFail(chain,chain->retained_status);
}

static void SparkLagunaTpChainReduceMlp(SparkLagunaTpChain *chain)
{
	SparkStatus status;
	chain->stage = SPARK_LAGUNA_CHAIN_STAGE_REDUCE_MLP;
	status = SparkLagunaModuleReduceAttentionOut(chain,chain->slot->attention_out_bf16);
	if ( status != SPARK_STATUS_OK )
		SparkLagunaTpChainFail(chain,status);
}

static SparkStatus SparkLagunaLazyExperts(SparkLagunaTpChain *chain)
{
	SparkWeightdExpertKey keys[SPARK_LAGUNA_MODEL_MOE_EXPERT_COUNT];
	SparkWeightdMap *map = chain->state->lazy_pack->map;
	SparkStatus status;
	void *address = 0;
	uint32_t count = 0u;
	if ( chain->slot->route_recorded == 0u || cudaEventSynchronize((cudaEvent_t)chain->slot->route_ready_event) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	status = SparkWeightdRouteKeys(chain->wave.first_layer_index + chain->next_layer,chain->slot->host_group_row_offset,SPARK_LAGUNA_MODEL_MOE_EXPERT_COUNT,chain->wave.row_count * SPARK_LAGUNA_MODEL_MOE_TOP_K,keys,SPARK_LAGUNA_MODEL_MOE_EXPERT_COUNT,&count);
	if ( status == SPARK_STATUS_OK )
		status = SparkWeightdMapAcquire(map,keys,count,&chain->expert_lease,SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS);
	if ( status == SPARK_STATUS_OK )
		status = SparkWeightdMapBeginUse(map,chain->expert_lease,&address);
	if ( status != SPARK_STATUS_OK )
		return(status);
	chain->expert_lease_begun = 1u;
	chain->wave.expert_lease_base = (const uint8_t *)address;
	chain->wave.expert_lease_local_layer = chain->next_layer;
	if ( SparkLagunaLaunchCudaLayerMlpExperts(&chain->wave,chain->next_layer) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	return(SPARK_STATUS_OK);
}

static void SparkLagunaLazyWork(void *context)
{
	SparkLagunaTpChain *chain = (SparkLagunaTpChain *)context;
	SparkStatus status,cleanup;
	status = SparkLagunaLazyExperts(chain);
	cleanup = SparkLagunaLazyRelease(chain);
	if ( cleanup == SPARK_STATUS_IO_ERROR || cleanup == SPARK_STATUS_BUSY )
		cleanup = SparkLagunaLazyRelease(chain);
	if ( cleanup != SPARK_STATUS_OK )
	{
		chain->retained_status = status != SPARK_STATUS_OK ? status : cleanup;
		fprintf(stderr,"GLM expert cleanup failed: slot=%u lease=%llu status=%d; retaining slot and lease for teardown retry\n",chain->slot_index,(unsigned long long)chain->expert_lease,(int32_t)cleanup);
		atomic_store_explicit(&chain->state->lazy_retained[chain->slot_index],chain,memory_order_release);
		return;
	}
	if ( status != SPARK_STATUS_OK )
		SparkLagunaTpChainFail(chain,status);
	else
		SparkLagunaTpChainReduceMlp(chain);
}

static void SparkLagunaTpChainAdvance(void *chain_context,SparkStatus status)
{
	SparkLagunaTpChain *chain;
	SparkLagunaModuleState *state;
	SparkStatus launch_status;
	cudaError_t error;
	chain = (SparkLagunaTpChain *)chain_context;
	if ( chain == 0 || chain->active == 0u )
		return;
	state = chain->state;
	if ( status != SPARK_STATUS_OK )
	{
		SparkLagunaTpChainFail(chain,status);
		return;
	}
	switch ( chain->stage )
	{
	case SPARK_LAGUNA_CHAIN_STAGE_BEGIN:
		SparkLagunaBuildWave(chain);
		if ( SparkLagunaLaunchCudaWaveBegin(&chain->wave) != 0 )
		{
			SparkLagunaTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->stage = SPARK_LAGUNA_CHAIN_STAGE_ATTENTION;
		chain->next_layer = 0u;
		launch_status = SparkLagunaModuleReduceHidden(chain,chain->slot->hidden_bf16);
		if ( launch_status != SPARK_STATUS_OK )
			SparkLagunaTpChainFail(chain,launch_status);
		return;
	case SPARK_LAGUNA_CHAIN_STAGE_ATTENTION:
		if ( SparkLagunaLaunchCudaLayerAttention(&chain->wave,chain->next_layer) != 0 )
		{
			SparkLagunaTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->stage = SPARK_LAGUNA_CHAIN_STAGE_REDUCE_ATTENTION;
		launch_status = SparkLagunaModuleReduceAttentionOut(chain,chain->slot->attention_out_bf16);
		if ( launch_status != SPARK_STATUS_OK )
			SparkLagunaTpChainFail(chain,launch_status);
		return;
	case SPARK_LAGUNA_CHAIN_STAGE_REDUCE_ATTENTION:
		if ( SparkLagunaLaunchCudaLayerAttentionPost(&chain->wave,chain->next_layer) != 0 )
		{
			SparkLagunaTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->stage = SPARK_LAGUNA_CHAIN_STAGE_MLP;
		SparkLagunaTpChainAdvance(chain,SPARK_STATUS_OK);
		return;
	case SPARK_LAGUNA_CHAIN_STAGE_MLP:
		if ( state->lazy_pack != 0 && (chain->wave.first_layer_index + chain->next_layer) >= SPARK_LAGUNA_MODEL_FIRST_ROUTED_LAYER )
		{
			if ( SparkLagunaLaunchCudaLayerMlpRoute(&chain->wave,chain->next_layer) != 0 )
				SparkLagunaTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			else
			{
				launch_status = SparkWeightdWorkerSubmit(state->lazy_pack->worker,SparkLagunaLazyWork,chain);
				if ( launch_status != SPARK_STATUS_OK )
					SparkLagunaTpChainFail(chain,launch_status);
			}
			return;
		}
		if ( SparkLagunaLaunchCudaLayerMlp(&chain->wave,chain->next_layer) != 0 )
		{
			SparkLagunaTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		SparkLagunaTpChainReduceMlp(chain);
		return;
	case SPARK_LAGUNA_CHAIN_STAGE_REDUCE_MLP:
		if ( SparkLagunaLaunchCudaLayerMlpPost(&chain->wave,chain->next_layer) != 0 )
		{
			SparkLagunaTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->next_layer++;
		if ( chain->next_layer < chain->wave.layer_count )
		{
			chain->stage = SPARK_LAGUNA_CHAIN_STAGE_ATTENTION;
			SparkLagunaTpChainAdvance(chain,SPARK_STATUS_OK);
		}
		else
		{
			chain->stage = SPARK_LAGUNA_CHAIN_STAGE_HEAD;
			SparkLagunaTpChainAdvance(chain,SPARK_STATUS_OK);
		}
		return;
	case SPARK_LAGUNA_CHAIN_STAGE_HEAD:
		if ( SparkLagunaLaunchCudaWaveHead(&chain->wave) != 0 )
		{
			SparkLagunaTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
			return;
		}
		chain->stage = SPARK_LAGUNA_CHAIN_STAGE_REDUCE_HEAD;
		launch_status = SparkLagunaModuleReduceHeadMax(chain);
		if ( launch_status != SPARK_STATUS_OK )
			SparkLagunaTpChainFail(chain,launch_status);
		return;
	case SPARK_LAGUNA_CHAIN_STAGE_REDUCE_HEAD:
		error = SparkLagunaLaunchHeadMaxlocUnpack((cudaStream_t)chain->slot->stream,chain->slot->head_maxloc_u64,chain->slot->output_token,chain->wave_rows);
		if ( error == cudaSuccess && state->owns_final_head != 0u )
			error = cudaMemcpyAsync(chain->slot->host_output_token_ids + chain->first_row,chain->slot->output_token,(uint64_t)chain->wave_rows * sizeof(uint32_t),cudaMemcpyDeviceToHost,(cudaStream_t)chain->slot->stream);
		launch_status = SparkStageModuleCudaStatus(SPARK_LAGUNA_MODULE_TAG,error,"tp_head_unpack");
		if ( launch_status != SPARK_STATUS_OK )
		{
			SparkLagunaTpChainFail(chain,launch_status);
			return;
		}
		if ( chain->next_wave_row < chain->batch->row_count )
		{
			uint32_t next_wave;
			next_wave = SparkLagunaRoundMajorWaveRows(chain->state,chain->batch,chain->next_wave_row);
			if ( next_wave == 0u )
			{
				SparkLagunaTpChainFail(chain,SPARK_STATUS_INVALID_ARGUMENT);
				return;
			}
			chain->first_row = chain->next_wave_row;
			chain->wave_rows = next_wave;
			chain->next_wave_row += next_wave;
			chain->stage = SPARK_LAGUNA_CHAIN_STAGE_BEGIN;
			SparkLagunaTpChainAdvance(chain,SPARK_STATUS_OK);
			return;
		}
		launch_status = SparkLagunaEnqueueAsyncCompletion(state,chain->slot,chain->slot_index);
		if ( launch_status != SPARK_STATUS_OK )
		{
			SparkLagunaTpChainFail(chain,launch_status);
			return;
		}
		chain->stage = SPARK_LAGUNA_CHAIN_STAGE_FINISH;
		chain->active = 0u;
		free(chain);
		return;
	default:
		SparkLagunaTpChainFail(chain,SPARK_STATUS_INTERNAL_ERROR);
		return;
	}
}

static SparkStatus SparkLagunaStageHostBatch(
	const SparkLagunaModuleState *state,
	SparkLagunaExecutionSlot *slot,
	const SparkLagunaResidentDecodeStageBatchView *batch)
{
	uint32_t row;
	if ( state == 0 || slot == 0 || batch == 0 || batch->row_count > SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (row=0u; row<batch->row_count; row++)
	{
		if ( batch->row_positions[row] >= UINT32_MAX )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		slot->host_resident_slots[row] = batch->row_resident_slots[row];
		slot->host_positions[row] = (uint32_t)batch->row_positions[row];
		if ( state->owns_embedding != 0u )
			slot->host_token_ids[row] = batch->token_ids[row];
	}
	memset(slot->host_kv_access_error,0,SPARK_LAGUNA_KV_ACCESS_ERROR_WORD_COUNT * sizeof(uint32_t));
	return(SPARK_STATUS_OK);
}

static void SparkLagunaPrepareAsyncCompletion(
	SparkLagunaModuleState *state,
	SparkModelDriverFrame *frame,
	const SparkLagunaResidentDecodeStageBatchView *batch,
	const uint8_t *lane_bound,
	const uint64_t *lane_sequence_ids,
	const uint64_t *lane_next_positions,
	uint32_t slot_index)
{
	SparkLagunaAsyncCompletion *async;
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
	for (lane=0u; lane<batch->active_sequence_count && lane<SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT; lane++)
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


static SparkStatus SparkLagunaFinishCacheLanes(SparkLagunaAsyncCompletion *async)
{
	SparkLagunaModuleState *state = async->state;
	SparkStatus result;
	uint32_t lane,resident;
	if ( pthread_mutex_lock(&state->kv_mutex) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	result = async->completion.status;
	result = SparkKvLaneTransactionsFinish(&state->kv_transactions,async->lane_indices,async->lane_count,result,0u);
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
		}
	}
	(void)pthread_mutex_unlock(&state->kv_mutex);
	return(result);
}

static void SparkLagunaCompleteOnWorker(void *context)
{
	SparkLagunaAsyncCompletion *async = (SparkLagunaAsyncCompletion *)context;
	SparkLagunaModuleState *state = async != 0 ? async->state : 0;
	SparkLagunaExecutionSlot *slot;
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
	async->completion.status = SparkLagunaFinishCacheLanes(async);
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

static void CUDART_CB SparkLagunaCompleteAsync(void *context)
{
	SparkLagunaAsyncCompletion *async = (SparkLagunaAsyncCompletion *)context;
	SparkStatus status;
	if ( async == 0 || async->state == 0 )
		return;
	status = SparkWeightdWorkerSubmit(async->state->completion_worker,SparkLagunaCompleteOnWorker,async);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"GLM completion handoff failed: status %d; retaining lane and slot ownership\n",(int32_t)status);
}

static SparkStatus SparkLagunaEnqueueAsyncCompletion(
	SparkLagunaModuleState *state,
	SparkLagunaExecutionSlot *slot,
	uint32_t slot_index)
{
	cudaStream_t stream;
	cudaError_t error;
	stream = (cudaStream_t)slot->stream;
	error = cudaMemcpyAsync(slot->host_kv_access_error,slot->kv_access_error,SPARK_LAGUNA_KV_ACCESS_ERROR_WORD_COUNT * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream);
	if ( error == cudaSuccess )
		error = cudaLaunchHostFunc(stream,SparkLagunaCompleteAsync,&state->completions[slot_index]);
	return(SparkStageModuleCudaStatus(SPARK_LAGUNA_MODULE_TAG,error,"async_completion"));
}

static SparkStatus SparkLagunaClaimCacheFrame(SparkLagunaModuleState *state,const SparkModelDriverFrame *frame,const SparkLagunaResidentDecodeStageBatchView *batch,const uint64_t *next_positions)
{
	uint32_t lane;
	SparkStatus status;
	if ( frame->cache_lanes == 0 || frame->cache_lane_count != batch->active_sequence_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID) == 0u || frame->driver_dispatch_slot != (uint32_t)(frame->request_id % state->pipeline_slot_count) )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	for (lane=0u; lane<frame->cache_lane_count; lane++)
		if ( frame->cache_lanes[lane].resident_sequence_slot != batch->row_resident_slots[lane] || frame->cache_lanes[lane].sequence_id != batch->row_sequence_ids[lane] || frame->cache_lanes[lane].sequence_position != batch->row_positions[lane] || frame->cache_lanes[lane].context_token_count != next_positions[lane] )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	if ( pthread_mutex_lock(&state->kv_mutex) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = SparkKvLaneTransactionsClaim(&state->kv_transactions,frame);
	(void)pthread_mutex_unlock(&state->kv_mutex);
	return(status);
}

static SparkStatus SparkLagunaUploadPageTables(SparkLagunaModuleState *state,const SparkLagunaAsyncCompletion *async,void *stream)
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
			return(SparkStageModuleCudaStatus(SPARK_LAGUNA_MODULE_TAG,error,"page_table_update"));
		memcpy(state->page_table_shadow + offset,state->kv_lane_physical_pages + offset,bytes);
	}
	return(SPARK_STATUS_OK);
}



static SparkStatus SparkLagunaStartClaimedBatch(SparkLagunaModuleState *state,SparkModelDriverFrame *frame,const SparkLagunaResidentDecodeStageFrameContext *context,uint32_t slot_index)
{
	SparkLagunaExecutionSlot *slot = &state->slots[slot_index];
	SparkLagunaTpChain *chain;
	SparkStatus status;
	cudaError_t error;
	chain = (SparkLagunaTpChain *)calloc(1u,sizeof(*chain));
	if ( chain == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkLagunaClaimCacheFrame(state,frame,context->batch,state->completions[slot_index].lane_next_positions);
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
	chain->wave_rows = SparkLagunaRoundMajorWaveRows(state,context->batch,0u);
	chain->next_wave_row = chain->wave_rows;
	chain->stage = SPARK_LAGUNA_CHAIN_STAGE_BEGIN;
	chain->active = 1u;
	atomic_fetch_add_explicit(&state->submitted_count,1u,memory_order_relaxed);
	status = SparkLagunaUploadPageTables(state,&state->completions[slot_index],slot->stream);
	if ( status == SPARK_STATUS_OK )
	{
		error = cudaMemsetAsync(slot->kv_access_error,0,SPARK_LAGUNA_KV_ACCESS_ERROR_WORD_COUNT * sizeof(uint32_t),(cudaStream_t)slot->stream);
		status = SparkStageModuleCudaStatus(SPARK_LAGUNA_MODULE_TAG,error,"kv_access_reset");
	}
	if ( status == SPARK_STATUS_OK && chain->wave_rows == 0u )
		status = SPARK_STATUS_INVALID_ARGUMENT;
	if ( status != SPARK_STATUS_OK )
		SparkLagunaTpChainFail(chain,status);
	else
		SparkLagunaTpChainAdvance(chain,SPARK_STATUS_OK);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkLagunaExecuteBatch(SparkLagunaModuleState *state,SparkModelDriverFrame *frame,const SparkLagunaResidentDecodeStageFrameContext *context)
{
	const SparkLagunaResidentDecodeStageBatchView *batch = context->batch;
	SparkLagunaClaimedContinuityContext continuity;
	SparkLagunaExecutionSlot *slot;
	uint8_t simulated_bound[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t simulated_sequence[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t simulated_next[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t slot_index;
	uint64_t last_ordinal;
	SparkStatus status;
	status = SparkTpChainOrdinal(frame->request_id,state->pipeline_slot_count,SPARK_LAGUNA_TP_COLLECTIVE_CREDITS_PER_SLOT,SPARK_LAGUNA_TP_CHAIN_OPERATIONS,SPARK_LAGUNA_TP_CHAIN_OPERATIONS - 1u,&last_ordinal);
	if ( status != SPARK_STATUS_OK )
		return(status);
	continuity.state = state;
	continuity.batch = batch;
	continuity.bound = simulated_bound;
	continuity.sequence_ids = simulated_sequence;
	continuity.next_positions = simulated_next;
	status = SparkStageModuleIndexSetClaimAndPrepare(state->lane_states,state->resident_sequence_capacity,batch->row_resident_slots,batch->active_sequence_count,SparkLagunaPrepareClaimedContinuity,&continuity);
	if ( status != SPARK_STATUS_OK )
		return(status);
	slot_index = (uint32_t)(frame->request_id % state->pipeline_slot_count);
	status = SparkStageModuleIndexSetClaim(state->slot_states,state->pipeline_slot_count,&slot_index,1u);
	if ( status == SPARK_STATUS_OK )
	{
		slot = &state->slots[slot_index];
		slot->stream = frame->execution_stream;
		status = SparkLagunaStageHostBatch(state,slot,batch);
		if ( status == SPARK_STATUS_OK )
		{
			SparkLagunaPrepareAsyncCompletion(state,frame,batch,simulated_bound,simulated_sequence,simulated_next,slot_index);
			status = SparkLagunaStartClaimedBatch(state,frame,context,slot_index);
		}
		if ( status != SPARK_STATUS_OK )
			SparkStageModuleSlotRelease(state->slot_states,slot_index);
	}
	if ( status != SPARK_STATUS_OK )
		SparkStageModuleIndexSetRelease(state->lane_states,state->resident_sequence_capacity,batch->row_resident_slots,batch->active_sequence_count);
	return(status);
}

SparkStatus SparkLagunaResidentDecodeStageExecute(
	void *module_state,
	SparkModelDriverFrame *frame)
{
	SparkLagunaModuleState *state;
	const SparkLagunaResidentDecodeStageFrameContext *context;
	SparkStatus status;
	state = (SparkLagunaModuleState *)module_state;
	context = 0;
	status = SparkLagunaValidateFrame(state,frame,&context);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"G5N-DBG execute: ValidateFrame -> %d\n",(int)status);
		if ( state != 0 )
			atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
		return(status);
	}
	status = SparkLagunaExecuteBatch(state,frame,context);
	if ( status != SPARK_STATUS_OK )
		atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
	return(status);
}

static void SparkLagunaAdmissionCost(
	void *context,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	(void)context;
	decision->host_staging_bytes = (uint64_t)request->new_token_count *
		(sizeof(uint32_t) * 3u + sizeof(uint64_t) * 2u);
	decision->device_memcpy_bytes = decision->host_staging_bytes;
}

static SparkStatus SparkLagunaResetExecutionState(SparkLagunaModuleState *state)
{
	cudaError_t drain;
	uint32_t lane;
	drain = cudaStreamSynchronize((cudaStream_t)state->execution_stream);
	if ( drain != cudaSuccess )
	{
		(void)SparkStageModuleCudaStatus(SPARK_LAGUNA_MODULE_TAG,drain,"reset_stream_drain");
		SPARK_FAIL(SPARK_STATUS_PENDING);
	}
	memset(state->page_table_shadow,0xff,(uint64_t)state->resident_sequence_capacity * state->pages_per_sequence * sizeof(uint32_t));
	for (lane=0u; lane<state->resident_sequence_capacity; lane++)
	{
		atomic_store_explicit(&state->lane_bound[lane],0u,memory_order_release);
		atomic_store_explicit(&state->lane_sequence_ids[lane],0u,memory_order_release);
		atomic_store_explicit(&state->lane_next_positions[lane],0u,memory_order_release);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkLagunaResetClaimed(SparkLagunaModuleState *state,uint64_t generation)
{
	SparkStatus status;
	if ( pthread_mutex_lock(&state->kv_mutex) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = SPARK_STATUS_VALIDATION_FAILED;
	if ( generation >= state->control_generation && generation > state->reset_generation )
	{
		state->control_generation = generation;
		status = SparkKvLaneTransactionsReset(&state->kv_transactions);
		if ( status == SPARK_STATUS_OK )
			status = SparkLagunaResetExecutionState(state);
		if ( status == SPARK_STATUS_OK )
			state->reset_generation = generation;
	}
	(void)pthread_mutex_unlock(&state->kv_mutex);
	return(status);
}

static SparkStatus SparkLagunaReset(SparkLagunaModuleState *state,const SparkModelDriverAdmissionRequest *request)
{
	uint32_t slots[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	uint32_t lanes[SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t index;
	SparkStatus status;
	if ( SparkModelDriverAdmissionRequestIsValid(request) == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
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
		status = SparkLagunaResetClaimed(state,request->control_generation);
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

SparkStatus SparkLagunaResidentDecodeStageAdmit(
	void *module_state,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	SparkLagunaModuleState *state;
	SparkAdmissionPolicyTable table;
	uint32_t available;
	SparkStatus status;
	state = (SparkLagunaModuleState *)module_state;
	if ( state == 0 || request == 0 || decision == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( request->admission_flags == SPARK_MODEL_DRIVER_ADMISSION_FLAG_RESET )
	{
		SparkModelDriverInitializeAdmissionDecision(decision);
		status = SparkLagunaReset(state,request);
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
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		SparkModelDriverInitializeAdmissionDecision(decision);
		return(SparkLagunaAdmissionPredicate(state,request,decision));
	}
	available = request->request_id != 0u && atomic_load_explicit(&state->slot_states[request->request_id % state->pipeline_slot_count],memory_order_acquire) == SPARK_STAGE_MODULE_SLOT_FREE ? 1u : 0u;
	memset(&table,0,sizeof(table));
	table.abi_version = SPARK_ADMISSION_ABI_VERSION;
	table.descriptor_bytes = (uint32_t)sizeof(table);
	table.max_active_sequence_count = state->resident_sequence_capacity;
	table.max_input_row_count = SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT;
	table.max_sequence_positions = state->max_sequence_positions;
	table.flags = SPARK_ADMISSION_POLICY_FLAG_DECODE_EQUALS_SLOTS |
		SPARK_ADMISSION_POLICY_FLAG_ALLOW_DISPATCH_FLAG;
	table.predicate = SparkLagunaAdmissionPredicate;
	table.predicate_context = state;
	table.cost = SparkLagunaAdmissionCost;
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

SparkStatus SparkLagunaResidentDecodeStageSnapshot(
	void *module_state,
	uint32_t program_id,
	SparkModelDriverRuntimeSnapshot *snapshot)
{
	SparkLagunaModuleState *state;
	uint32_t index,resident_count;
	state = (SparkLagunaModuleState *)module_state;
	if ( state == 0 || snapshot == 0 || program_id == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
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

static void SparkLagunaReleaseCaches(SparkLagunaModuleState *state)
{
	if ( state->kv_page_store.abi_version == SPARK_KV_PAGE_STORE_ABI_VERSION )
		SparkKvPageStoreDestroy(&state->kv_page_store);
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

void SparkLagunaResidentDecodeStageDestroy(void *module_state)
{
	SparkLagunaModuleState *state;
	uint32_t slot;
	state = (SparkLagunaModuleState *)module_state;
	if ( state == 0 )
		return;
	for (slot=0u; slot<state->pipeline_slot_count; slot++)
		if ( atomic_load_explicit(&state->lazy_retained[slot],memory_order_acquire) != 0 )
			break;
	if ( slot < state->pipeline_slot_count )
	{
		if ( state->lazy_pack == 0 || state->lazy_pack->worker == 0 || SparkWeightdWorkerSubmit(state->lazy_pack->worker,SparkLagunaLazyRetryRetained,state) != SPARK_STATUS_OK )
			return;
	}
	if ( SparkStageModuleWaitForSlots(SPARK_LAGUNA_MODULE_TAG,state->slot_states,state->pipeline_slot_count,SPARK_STAGE_MODULE_DESTROY_QUIESCE_TIMEOUT_NS) != SPARK_STATUS_OK )
		return;
	if ( state->lazy_pack != 0 && state->lazy_pack->worker != 0 && SparkWeightdWorkerWaitIdle(state->lazy_pack->worker,SPARK_STAGE_MODULE_DESTROY_QUIESCE_TIMEOUT_NS) != SPARK_STATUS_OK )
		return;
	if ( state->completion_worker != 0 )
	{
		if ( SparkWeightdWorkerWaitIdle(state->completion_worker,SPARK_STAGE_MODULE_DESTROY_QUIESCE_TIMEOUT_NS) != SPARK_STATUS_OK || SparkWeightdWorkerDestroy(state->completion_worker) != SPARK_STATUS_OK )
			return;
		state->completion_worker = 0;
	}
	if ( SparkStageModuleCudaStatus(SPARK_LAGUNA_MODULE_TAG,cudaStreamSynchronize((cudaStream_t)state->execution_stream),"destroy_stream_drain") != SPARK_STATUS_OK )
		return;
	if ( state->lazy_pack != 0 )
	{
		if ( SparkWeightdLazyPackDestroy(state->lazy_pack) != SPARK_STATUS_OK )
			return;
		state->lazy_pack = 0;
	}
	if ( state->tp_device_collective_initialized != 0u )
		SparkTpDeviceCollectiveDestroy(&state->tp_device_collective);
	SparkLagunaReleaseCaches(state);
	SparkLagunaReleaseSlotHost(state);
	SparkStageModuleLedgerRelease(&state->ledger);
	free(state);
}


static SparkStatus SparkLagunaInitializeState(
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services,
	SparkLagunaModuleState **state_out)
{
	SparkLagunaModuleState *state;
	const char *pack_path;
	SparkStatus status;
	uint32_t lane;
	state = (SparkLagunaModuleState *)calloc(1u,sizeof(*state));
	if ( state == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->ledger.module_tag = SPARK_LAGUNA_MODULE_TAG;
	for (lane=0u; lane<SPARK_LAGUNA_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT; lane++)
		atomic_init(&state->lazy_retained[lane],0);
	status = SparkLagunaModuleConfigure(state,configuration,host_services,&pack_path);
	if ( status == SPARK_STATUS_OK && SparkLagunaConfigureCudaModule(&state->multiprocessor_count) != 0 )
		status = SPARK_STATUS_TARGET_MISMATCH;
	if ( status == SPARK_STATUS_OK )
		status = SparkLagunaPackLoad(state,pack_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkLagunaAllocateCaches(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkLagunaAllocateSlots(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkLagunaModuleInitializeTpCollective(state,(const SparkLagunaResidentDecodeStageNodeContext *)host_services->node_context);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,SPARK_LAGUNA_MODEL_ROPE_FULL_ROTARY_DIMENSION / 2u * sizeof(float),(void **)&state->yarn_inv_freq);
	if ( status == SPARK_STATUS_OK )
		status = SparkLagunaStageYarnTableUpload(state->yarn_inv_freq,state->execution_stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkWeightdWorkerCreate(&state->completion_worker);
	if ( status != SPARK_STATUS_OK )
	{
		if ( state->lazy_pack != 0 && SparkWeightdLazyPackDestroy(state->lazy_pack) != SPARK_STATUS_OK )
		{
			fprintf(stderr,"GLM lazy initialization cleanup failed; retaining CUDA resources until process exit\n");
			return(status);
		}
		SparkLagunaReleaseCaches(state);
		SparkLagunaReleaseSlotHost(state);
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
	atomic_init(&state->nccl_next_ordinal,0u);
	*state_out = state;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkLagunaResidentDecodeStageInitialize(
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services,
	void **module_state)
{
	SparkLagunaModuleState *state;
	SparkStatus status;
	status = SparkFirmwareModuleValidateInitialization(configuration,host_services,module_state);
	if ( status != SPARK_STATUS_OK )
		return(status);
	state = 0;
	status = SparkLagunaInitializeState(configuration,host_services,&state);
	if ( status != SPARK_STATUS_OK )
		return(status);
	*module_state = state;
	return(SPARK_STATUS_OK);
}
