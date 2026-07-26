/* Large stage packs exceed 2 GB: 64-bit file offsets and fseeko are required. */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>

#include "sparkpipe/spark_k3_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_model_driver_support.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "spark_k3_stagepack_format.h"

/*
 * K3 resident decode stage, host side.
 *
 * Owns everything that is resident: the stage pack's weights, the KDA
 * recurrent state pool, the paged MLA latent cache and the per-slot
 * activation buffers. Execute walks the layer stack for one dispatch and the
 * CUDA translation unit performs the work; this file never computes on
 * tensors, it only validates, places and sequences.
 *
 * The node owns exactly one qualified layer slice. The model frame context
 * carries any persistent sequence-lane identity and pipeline transport. The
 * generic dispatch ticket is never reused as a model lane.
 */

#define SPARK_K3_MODULE_TAG "k3_stage"
#define SPARK_K3_UNQUALIFIED_EXECUTION_ENVIRONMENT "K3_ALLOW_UNQUALIFIED_EXECUTION"
#define SPARK_K3_MODULE_MAX_MOE_INTERMEDIATE_ROW_ELEMENTS \
	((SPARK_K3_MODEL_MOE_TOP_K * SPARK_K3_MODEL_MOE_INTERMEDIATE_DIMENSION) > SPARK_K3_MODEL_DENSE_INTERMEDIATE_DIMENSION \
		? (SPARK_K3_MODEL_MOE_TOP_K * SPARK_K3_MODEL_MOE_INTERMEDIATE_DIMENSION) \
		: SPARK_K3_MODEL_DENSE_INTERMEDIATE_DIMENSION)

/*
 * Host staging owned by one pipeline slot. Concurrent Execute calls claim
 * distinct slots, so nothing here is shared; the device token count is a
 * per-slot single int32 the chunk kernel reads for partial-chunk masking.
 * Every buffer that sources an async H2D copy lives here for the life of the
 * module, so the copies stay correct even if these pages are later pinned.
 */
typedef struct SparkK3ModuleSlotStaging
{
	uint32_t *row_token_ids;
	uint32_t *row_slot_mapping;
	uint32_t *row_lane_indices;
	uint32_t *row_context_lengths;
	uint32_t *row_cold_flags;
	uint32_t *row_output_token_ids;
	int32_t sequence_token_count;
	int32_t *device_sequence_token_count;
} SparkK3ModuleSlotStaging;

typedef struct SparkK3ModuleState
{
	SparkK3ResidentDecodeStageNodeContext node_context;
	SparkFirmwareModuleHostServices host_services;
	SparkK3PipelineSlot pipeline_slots[SPARK_K3_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	SparkK3ModuleSlotStaging slot_staging[SPARK_K3_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	SparkK3AttnResSiteWeights attnres_sites[SPARK_K3_MODEL_LAYER_COUNT * SPARK_K3_RESIDENT_DECODE_STAGE_ATTNRES_SITES_PER_LAYER];
	SparkK3AttnResSiteWeights attnres_final_site;
	SparkK3KdaLayerWeights kda_weights[SPARK_K3_MODEL_LAYER_COUNT];
	SparkK3MlaLayerWeights mla_weights[SPARK_K3_MODEL_LAYER_COUNT];
	SparkK3MoeLayerWeights moe_weights[SPARK_K3_MODEL_LAYER_COUNT];
	const void *attention_norm_by_layer[SPARK_K3_MODEL_LAYER_COUNT];
	const void *mlp_norm_by_layer[SPARK_K3_MODEL_LAYER_COUNT];
	uint32_t kda_ordinal_by_layer[SPARK_K3_MODEL_LAYER_COUNT];
	uint32_t mla_ordinal_by_layer[SPARK_K3_MODEL_LAYER_COUNT];
	uint32_t kda_layer_count;
	uint32_t mla_layer_count;
	SparkK3MlaBlockTableView owned_block_table;
	uint32_t *host_block_indices;
	uint32_t *host_lane_block_counts;
	uint32_t lane_capacity;
	uint32_t row_capacity;
	uint32_t stage_count;
	uint32_t stage_index;
	uint32_t max_context_tokens;
	uint32_t blocks_per_lane;
	uint32_t scratch_block_index;
	uint32_t unqualified_execution_enabled;
	atomic_uint slot_states[SPARK_K3_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	atomic_uint lane_states[SPARK_K3_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_sequence_ids[SPARK_K3_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_next_positions[SPARK_K3_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	atomic_uint_fast64_t submitted_count;
	atomic_uint_fast64_t completed_count;
	atomic_uint_fast64_t rejected_count;
	atomic_uint_fast64_t failed_count;
	SparkStageModuleLedger ledger;
} SparkK3ModuleState;

typedef struct SparkK3ModuleTensorBinding
{
	SparkK3Mxfp4LinearView *view;
	const void **pointer_cell;
} SparkK3ModuleTensorBinding;

static int32_t SparkK3ModuleBindingGlobal(SparkK3ModuleState *state, uint32_t tensor_kind, SparkK3ModuleTensorBinding *binding)
{
	switch ( tensor_kind )
	{
	case SPARK_K3_STAGEPACK_TENSOR_EMBEDDING:
		binding->pointer_cell = &state->node_context.token_embedding_bf16;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_ATTNRES_FINAL_QUERY:
		binding->pointer_cell = &state->attnres_final_site.pseudo_query_bf16;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_ATTNRES_FINAL_NORM:
		binding->pointer_cell = &state->attnres_final_site.key_norm_weight_bf16;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_FINAL_NORM:
		binding->pointer_cell = &state->node_context.final_norm_weight_bf16;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_LM_HEAD_RESTRICTED:
		binding->pointer_cell = &state->node_context.restricted_lm_head_weight_bf16;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_RESTRICTED_TOKEN_IDS:
		binding->pointer_cell = (const void **)&state->node_context.restricted_token_ids;
		return(0);
	default:
		return(-1);
	}
}

static int32_t SparkK3ModuleBindingSite(SparkK3ModuleState *state, uint32_t tensor_kind, uint32_t layer_index, SparkK3ModuleTensorBinding *binding)
{
	SparkK3AttnResSiteWeights *attention_site = &state->attnres_sites[(layer_index * SPARK_K3_RESIDENT_DECODE_STAGE_ATTNRES_SITES_PER_LAYER) + SPARK_K3_RESIDENT_DECODE_STAGE_ATTNRES_ATTENTION_SITE];
	SparkK3AttnResSiteWeights *mlp_site = &state->attnres_sites[(layer_index * SPARK_K3_RESIDENT_DECODE_STAGE_ATTNRES_SITES_PER_LAYER) + SPARK_K3_RESIDENT_DECODE_STAGE_ATTNRES_MLP_SITE];
	switch ( tensor_kind )
	{
	case SPARK_K3_STAGEPACK_TENSOR_ATTNRES_ATTENTION_QUERY:
		binding->pointer_cell = &attention_site->pseudo_query_bf16;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_ATTNRES_ATTENTION_NORM:
		binding->pointer_cell = &attention_site->key_norm_weight_bf16;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_ATTNRES_MLP_QUERY:
		binding->pointer_cell = &mlp_site->pseudo_query_bf16;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_ATTNRES_MLP_NORM:
		binding->pointer_cell = &mlp_site->key_norm_weight_bf16;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_ATTENTION_NORM:
		binding->pointer_cell = &state->attention_norm_by_layer[layer_index];
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MLP_NORM:
		binding->pointer_cell = &state->mlp_norm_by_layer[layer_index];
		return(0);
	default:
		return(-1);
	}
}

static int32_t SparkK3ModuleBindingKda(SparkK3ModuleState *state, uint32_t tensor_kind, uint32_t layer_index, SparkK3ModuleTensorBinding *binding)
{
	SparkK3KdaLayerWeights *weights = &state->kda_weights[layer_index];
	switch ( tensor_kind )
	{
	case SPARK_K3_STAGEPACK_TENSOR_KDA_QUERY:
		binding->view = &weights->query;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_KDA_KEY:
		binding->view = &weights->key;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_KDA_VALUE:
		binding->view = &weights->value;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_KDA_DECAY_LOW:
		binding->view = &weights->decay_low;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_KDA_DECAY_HIGH:
		binding->view = &weights->decay_high;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_KDA_BETA:
		binding->view = &weights->beta;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_KDA_GATE_LOW:
		binding->view = &weights->output_gate_low;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_KDA_GATE_HIGH:
		binding->view = &weights->output_gate_high;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_KDA_OUTPUT:
		binding->view = &weights->output;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_KDA_HEAD_NORM:
		binding->pointer_cell = &weights->head_norm_weight_bf16;
		return(0);
	default:
		return(-1);
	}
}

static int32_t SparkK3ModuleBindingMla(SparkK3ModuleState *state, uint32_t tensor_kind, uint32_t layer_index, SparkK3ModuleTensorBinding *binding)
{
	SparkK3MlaLayerWeights *weights = &state->mla_weights[layer_index];
	switch ( tensor_kind )
	{
	case SPARK_K3_STAGEPACK_TENSOR_MLA_QUERY_A:
		binding->view = &weights->query_a;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MLA_QUERY_A_NORM:
		binding->pointer_cell = &weights->query_a_norm_weight_bf16;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MLA_QUERY_B:
		binding->view = &weights->query_b;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MLA_KV_A:
		binding->view = &weights->kv_a;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MLA_KV_A_NORM:
		binding->pointer_cell = &weights->kv_a_norm_weight_bf16;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MLA_KV_B:
		binding->view = &weights->kv_b;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MLA_HEAD_GATE:
		binding->view = &weights->head_gate;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MLA_OUTPUT:
		binding->view = &weights->output;
		return(0);
	default:
		return(-1);
	}
}

static int32_t SparkK3ModuleBindingMoe(SparkK3ModuleState *state, uint32_t tensor_kind, uint32_t layer_index, SparkK3ModuleTensorBinding *binding)
{
	SparkK3MoeLayerWeights *weights = &state->moe_weights[layer_index];
	switch ( tensor_kind )
	{
	case SPARK_K3_STAGEPACK_TENSOR_MOE_ROUTER:
		binding->pointer_cell = &weights->router_weight_bf16;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MOE_ROUTER_BIAS:
		binding->pointer_cell = (const void **)&weights->router_score_bias_f32;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MOE_SHARED_GATE:
		binding->view = &weights->shared_gate;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MOE_SHARED_UP:
		binding->view = &weights->shared_up;
		return(0);
	case SPARK_K3_STAGEPACK_TENSOR_MOE_SHARED_DOWN:
		binding->view = &weights->shared_down;
		return(0);
	default:
		return(-1);
	}
}

static int32_t SparkK3ModuleBindingForTensor(SparkK3ModuleState *state, uint32_t tensor_kind, uint32_t layer_index, SparkK3ModuleTensorBinding *binding)
{
	binding->view = 0;
	binding->pointer_cell = 0;
	if ( SparkK3ModuleBindingGlobal(state,tensor_kind,binding) == 0 )
		return(0);
	if ( layer_index >= SPARK_K3_MODEL_LAYER_COUNT )
		return(-1);
	if ( SparkK3ModuleBindingSite(state,tensor_kind,layer_index,binding) == 0 )
		return(0);
	if ( SparkK3ModuleBindingKda(state,tensor_kind,layer_index,binding) == 0 )
		return(0);
	if ( SparkK3ModuleBindingMla(state,tensor_kind,layer_index,binding) == 0 )
		return(0);
	return(SparkK3ModuleBindingMoe(state,tensor_kind,layer_index,binding));
}

static uint32_t SparkK3ModuleTensorIsExpertConcatenation(uint32_t tensor_kind)
{
	return(tensor_kind == SPARK_K3_STAGEPACK_TENSOR_MOE_EXPERT_GATE || tensor_kind == SPARK_K3_STAGEPACK_TENSOR_MOE_EXPERT_UP || tensor_kind == SPARK_K3_STAGEPACK_TENSOR_MOE_EXPERT_DOWN);
}

/*
 * An expert tensor is one allocation holding every expert's matrix back to
 * back, so a route only needs a base pointer and a stride. The kernels index
 * expert e at payload + e*payload_stride, which is why the stride is derived
 * from the per-expert shape rather than stored in the pack.
 */
static SparkStatus SparkK3ModuleBindExpertConcatenation(SparkK3ModuleState *state, const SparkK3StagePackEntry *entry, void *payload, void *scale)
{
	SparkK3MoeLayerWeights *weights = &state->moe_weights[entry->layer_index];
	uint32_t expert_rows = entry->rows / SPARK_K3_MODEL_MOE_EXPERT_COUNT;
	uint64_t payload_stride = SparkK3StagePackPayloadBytes(entry->weight_format,expert_rows,entry->columns);
	uint64_t scale_stride = SparkK3StagePackScaleBytes(entry->weight_format,expert_rows,entry->columns);
	weights->expert_count = SPARK_K3_MODEL_MOE_EXPERT_COUNT;
	weights->intermediate_dimension = SPARK_K3_MODEL_MOE_INTERMEDIATE_DIMENSION;
	weights->weight_format = entry->weight_format;
	if ( entry->tensor_kind == SPARK_K3_STAGEPACK_TENSOR_MOE_EXPERT_GATE )
	{
		weights->expert_gate_payload = payload;
		weights->expert_gate_scale_e8m0 = (const uint8_t *)scale;
		weights->expert_gate_payload_stride_bytes = payload_stride;
		weights->expert_gate_scale_stride_bytes = scale_stride;
		return(SPARK_STATUS_OK);
	}
	if ( entry->tensor_kind == SPARK_K3_STAGEPACK_TENSOR_MOE_EXPERT_UP )
	{
		weights->expert_up_payload = payload;
		weights->expert_up_scale_e8m0 = (const uint8_t *)scale;
		weights->expert_up_payload_stride_bytes = payload_stride;
		weights->expert_up_scale_stride_bytes = scale_stride;
		return(SPARK_STATUS_OK);
	}
	weights->expert_down_payload = payload;
	weights->expert_down_scale_e8m0 = (const uint8_t *)scale;
	weights->expert_down_payload_stride_bytes = payload_stride;
	weights->expert_down_scale_stride_bytes = scale_stride;
	return(SPARK_STATUS_OK);
}

/*
 * Validate one directory entry against the shape table before a byte of it is
 * trusted: the kind must exist, the layer must be in range and per-layer or
 * global as declared, the format must be the tensor's natural format or MXFP4
 * where quantization is allowed, the extents must match exactly, and the
 * payload and scale byte counts must be exactly what that shape implies.
 */
static SparkStatus SparkK3ModuleValidateEntry(const SparkK3ModuleState *state, const SparkK3StagePackEntry *entry, uint64_t file_bytes)
{
	SparkK3StagePackTensorShape shape;
	uint32_t is_global = (entry->layer_index == SPARK_K3_STAGEPACK_GLOBAL_LAYER);
	uint32_t layer_index = is_global ? 0u : entry->layer_index;
	uint64_t payload_bytes,scale_bytes;
	if ( SparkK3StagePackResolvedShape(entry->tensor_kind,layer_index,&shape) < 0 )
	{
		fprintf(stderr,"k3_stage pack_entry_unknown kind=%u layer=%u\n",entry->tensor_kind,entry->layer_index);
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( (shape.per_layer != 0u) == (is_global != 0u) )
	{
		fprintf(stderr,"k3_stage pack_entry_layer_scope kind=%u layer=%u per_layer=%u\n",entry->tensor_kind,entry->layer_index,shape.per_layer);
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( is_global == 0u && (entry->layer_index < state->node_context.first_layer_index || entry->layer_index >= state->node_context.first_layer_index + state->node_context.layer_count) )
	{
		fprintf(stderr,"k3_stage pack_entry_out_of_slice kind=%u layer=%u slice=%u+%u\n",entry->tensor_kind,entry->layer_index,state->node_context.first_layer_index,state->node_context.layer_count);
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( entry->rows != shape.rows || entry->columns != shape.columns )
	{
		fprintf(stderr,"k3_stage pack_entry_shape kind=%u layer=%u rows=%u expected=%u columns=%u expected=%u\n",entry->tensor_kind,entry->layer_index,entry->rows,shape.rows,entry->columns,shape.columns);
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( entry->weight_format != shape.natural_format && !(shape.quantizable != 0u && entry->weight_format == SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1) )
	{
		fprintf(stderr,"k3_stage pack_entry_format kind=%u layer=%u format=%u natural=%u quantizable=%u\n",entry->tensor_kind,entry->layer_index,entry->weight_format,shape.natural_format,shape.quantizable);
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( entry->weight_format == SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 && entry->scale_group_size != SPARK_K3_MODEL_MXFP4_GROUP_SIZE )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( entry->weight_format == SPARK_K3_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 && (entry->columns % SPARK_K3_MODEL_MXFP4_GROUP_SIZE) != 0u )
		return(SPARK_STATUS_SCHEMA_ERROR);
	payload_bytes = SparkK3StagePackPayloadBytes(entry->weight_format,entry->rows,entry->columns);
	scale_bytes = SparkK3StagePackScaleBytes(entry->weight_format,entry->rows,entry->columns);
	if ( entry->payload_bytes != payload_bytes || entry->scale_bytes != scale_bytes )
	{
		fprintf(stderr,"k3_stage pack_entry_bytes kind=%u layer=%u payload=%llu expected=%llu scale=%llu expected=%llu\n",entry->tensor_kind,entry->layer_index,(unsigned long long)entry->payload_bytes,(unsigned long long)payload_bytes,(unsigned long long)entry->scale_bytes,(unsigned long long)scale_bytes);
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( entry->payload_bytes > file_bytes || entry->payload_offset > file_bytes - entry->payload_bytes )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( scale_bytes != 0u && (entry->scale_bytes > file_bytes || entry->scale_offset > file_bytes - entry->scale_bytes) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

static void SparkK3ModuleFillLinearView(SparkK3Mxfp4LinearView *view, const SparkK3StagePackEntry *entry, void *payload, void *scale)
{
	view->abi_version = SPARK_K3_RESIDENT_DECODE_STAGE_MXFP4_LINEAR_VIEW_ABI_VERSION;
	view->weight_format = entry->weight_format;
	view->input_dimension = entry->columns;
	view->output_dimension = entry->rows;
	view->weight_payload = payload;
	view->weight_scale_e8m0 = (const uint8_t *)scale;
	view->weight_payload_bytes = entry->payload_bytes;
	view->weight_scale_bytes = entry->scale_bytes;
}

static SparkStatus SparkK3ModuleLoadTensor(SparkK3ModuleState *state, FILE *file, const SparkK3StagePackEntry *entry, uint64_t file_bytes)
{
	SparkK3ModuleTensorBinding binding;
	SparkStatus status;
	void *payload = 0;
	void *scale = 0;
	status = SparkK3ModuleValidateEntry(state,entry,file_bytes);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,entry->payload_offset,entry->payload_bytes,&payload);
	if ( status == SPARK_STATUS_OK && entry->scale_bytes != 0u )
		status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,entry->scale_offset,entry->scale_bytes,&scale);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( SparkK3ModuleTensorIsExpertConcatenation(entry->tensor_kind) != 0u )
		return(SparkK3ModuleBindExpertConcatenation(state,entry,payload,scale));
	if ( SparkK3ModuleBindingForTensor(state,entry->tensor_kind,entry->layer_index,&binding) < 0 )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( binding.view != 0 )
		SparkK3ModuleFillLinearView(binding.view,entry,payload,scale);
	else
		*binding.pointer_cell = payload;
	return(SPARK_STATUS_OK);
}

/*
 * Version 1 executes the whole stack on one node. A pack carrying a slice is
 * rejected rather than half-served: the hidden-state transport that a
 * pipelined K3 needs must also carry the AttnRes block array, which is a
 * protocol change tracked in DIFFERENCES.md, not something to fake here.
 */
/*
 * Load the pack: header first, then the geometry comparison, then every
 * directory entry. A geometry mismatch names the offending field and fails
 * the load; the driver never adapts to the pack.
 */
static SparkStatus SparkK3ModuleReadPackHeader(const SparkK3ModuleState *state, FILE *file, const char *path, SparkK3StagePackHeader *header)
{
	SparkK3StagePackHeader expected;
	SparkStatus status;
	int32_t mismatch;
	status = SparkStageModulePackRead(SPARK_K3_MODULE_TAG,file,0u,header,sizeof(*header));
	if ( status != SPARK_STATUS_OK )
		return(status);
	SparkK3StagePackExpectedGeometry(&expected,state->node_context.first_layer_index,state->node_context.layer_count,header->tensor_count);
	mismatch = SparkK3StagePackCompareGeometry(header,&expected);
	if ( mismatch != 0 )
	{
		fprintf(stderr,"k3_stage pack_geometry_mismatch field=%s path=%s\n",SparkK3StagePackGeometryFieldName(mismatch),path);
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( ((uint64_t)header->tensor_count * sizeof(SparkK3StagePackEntry)) > header->file_bytes || header->directory_offset > header->file_bytes - ((uint64_t)header->tensor_count * sizeof(SparkK3StagePackEntry)) )
	{
		fprintf(stderr,"k3_stage pack_directory_bounds offset=%llu tensors=%u file=%llu\n",(unsigned long long)header->directory_offset,header->tensor_count,(unsigned long long)header->file_bytes);
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkK3ModuleLoadPack(SparkK3ModuleState *state, const char *path)
{
	SparkK3StagePackHeader header;
	SparkK3StagePackEntry *directory;
	SparkStatus status;
	FILE *file;
	uint32_t index;
	file = fopen(path,"rb");
	if ( file == 0 )
	{
		fprintf(stderr,"k3_stage pack_open_failed path=%s\n",path);
		return(SPARK_STATUS_NOT_FOUND);
	}
	status = SparkK3ModuleReadPackHeader(state,file,path,&header);
	if ( status != SPARK_STATUS_OK )
	{
		fclose(file);
		return(status);
	}
	directory = (SparkK3StagePackEntry *)malloc((size_t)header.tensor_count * sizeof(SparkK3StagePackEntry));
	if ( directory == 0 )
	{
		fclose(file);
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	status = SparkStageModulePackRead(SPARK_K3_MODULE_TAG,file,header.directory_offset,directory,(uint64_t)header.tensor_count * sizeof(SparkK3StagePackEntry));
	for (index = 0; index < header.tensor_count && status == SPARK_STATUS_OK; index++)
		status = SparkK3ModuleLoadTensor(state,file,&directory[index],header.file_bytes);
	free(directory);
	fclose(file);
	if ( status != SPARK_STATUS_OK )
		return(status);
	fprintf(stderr,"k3_stage pack_loaded path=%s tensors=%u device_bytes=%llu\n",path,header.tensor_count,(unsigned long long)state->ledger.device_bytes_resident);
	return(SPARK_STATUS_OK);
}

/*
 * Every tensor the layer stack will dereference must have arrived. A pack
 * that is missing a tensor fails here rather than at the first launch that
 * reads a null pointer, and rather than running with a silently absent term.
 */
static SparkStatus SparkK3ModuleValidateLayerWeights(const SparkK3ModuleState *state, uint32_t layer_index)
{
	const SparkK3KdaLayerWeights *kda = &state->kda_weights[layer_index];
	const SparkK3MlaLayerWeights *mla = &state->mla_weights[layer_index];
	const SparkK3MoeLayerWeights *moe = &state->moe_weights[layer_index];
	uint32_t site_base = layer_index * SPARK_K3_RESIDENT_DECODE_STAGE_ATTNRES_SITES_PER_LAYER;
	if ( state->attention_norm_by_layer[layer_index] == 0 || state->mlp_norm_by_layer[layer_index] == 0 )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( state->attnres_sites[site_base].pseudo_query_bf16 == 0 || state->attnres_sites[site_base].key_norm_weight_bf16 == 0 )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( state->attnres_sites[site_base + 1u].pseudo_query_bf16 == 0 || state->attnres_sites[site_base + 1u].key_norm_weight_bf16 == 0 )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( SPARK_K3_MODEL_LAYER_IS_KDA(layer_index) != 0u )
	{
		if ( kda->query.weight_payload == 0 || kda->key.weight_payload == 0 || kda->value.weight_payload == 0 || kda->output.weight_payload == 0 )
			return(SPARK_STATUS_SCHEMA_ERROR);
		if ( kda->decay_low.weight_payload == 0 || kda->decay_high.weight_payload == 0 || kda->beta.weight_payload == 0 )
			return(SPARK_STATUS_SCHEMA_ERROR);
		if ( kda->output_gate_low.weight_payload == 0 || kda->output_gate_high.weight_payload == 0 || kda->head_norm_weight_bf16 == 0 )
			return(SPARK_STATUS_SCHEMA_ERROR);
	}
	else
	{
		if ( mla->query_a.weight_payload == 0 || mla->query_b.weight_payload == 0 || mla->kv_a.weight_payload == 0 || mla->kv_b.weight_payload == 0 )
			return(SPARK_STATUS_SCHEMA_ERROR);
		if ( mla->output.weight_payload == 0 || mla->head_gate.weight_payload == 0 || mla->query_a_norm_weight_bf16 == 0 || mla->kv_a_norm_weight_bf16 == 0 )
			return(SPARK_STATUS_SCHEMA_ERROR);
	}
	if ( moe->shared_gate.weight_payload == 0 || moe->shared_up.weight_payload == 0 || moe->shared_down.weight_payload == 0 )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( layer_index < SPARK_K3_MODEL_FIRST_ROUTED_LAYER )
		return(SPARK_STATUS_OK);
	if ( moe->router_weight_bf16 == 0 || moe->router_score_bias_f32 == 0 )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( moe->expert_gate_payload == 0 || moe->expert_up_payload == 0 || moe->expert_down_payload == 0 )
		return(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkK3ModuleValidateResidentWeights(const SparkK3ModuleState *state)
{
	SparkStatus status;
	uint32_t layer_index;
	if ( state->node_context.owns_embedding != 0u && state->node_context.token_embedding_bf16 == 0 )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( state->node_context.owns_final_head != 0u && (state->node_context.final_norm_weight_bf16 == 0 || state->node_context.restricted_lm_head_weight_bf16 == 0 || state->node_context.restricted_token_ids == 0 || state->attnres_final_site.pseudo_query_bf16 == 0 || state->attnres_final_site.key_norm_weight_bf16 == 0) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	for (layer_index = state->node_context.first_layer_index; layer_index < state->node_context.first_layer_index + state->node_context.layer_count; layer_index++)
	{
		status = SparkK3ModuleValidateLayerWeights(state,layer_index);
		if ( status != SPARK_STATUS_OK )
		{
			fprintf(stderr,"k3_stage weights_incomplete layer=%u kind=%s\n",layer_index,SPARK_K3_MODEL_LAYER_IS_KDA(layer_index) != 0u ? "kda" : "mla");
			return(status);
		}
	}
	return(SPARK_STATUS_OK);
}

// Attention alternates KDA and gated MLA on a fixed period; each kind numbers
// its own layers so the state pool and the latent cache stay dense.
static void SparkK3ModuleAssignLayerOrdinals(SparkK3ModuleState *state)
{
	uint32_t layer_index;
	state->kda_layer_count = 0u;
	state->mla_layer_count = 0u;
	for (layer_index = state->node_context.first_layer_index; layer_index < state->node_context.first_layer_index + state->node_context.layer_count; layer_index++)
	{
		if ( SPARK_K3_MODEL_LAYER_IS_KDA(layer_index) != 0u )
		{
			state->kda_ordinal_by_layer[layer_index] = state->kda_layer_count;
			state->kda_layer_count++;
		}
		else
		{
			state->mla_ordinal_by_layer[layer_index] = state->mla_layer_count;
			state->mla_layer_count++;
		}
	}
}

/*
 * The KDA state is the whole history: one dk by dv fp32 matrix per head, per
 * layer, per lane, carried across dispatches. state_valid_by_lane is the cold
 * flag the decode kernel reads to zero a lane's matrix on first touch instead
 * of paying a memset over the pool.
 */
static SparkStatus SparkK3ModuleAllocateKdaStatePool(SparkK3ModuleState *state)
{
	SparkK3KdaStatePool *pool = &state->node_context.kda_state_pool;
	SparkStatus status;
	uint64_t layer_stride = SPARK_K3_MODEL_KDA_STATE_ELEMENTS_PER_LAYER;
	uint64_t lane_stride = layer_stride * (uint64_t)state->kda_layer_count;
	void *pointer = 0;
	pool->abi_version = SPARK_K3_RESIDENT_DECODE_STAGE_KDA_STATE_POOL_ABI_VERSION;
	pool->lane_capacity = state->lane_capacity;
	pool->kda_layer_count = state->kda_layer_count;
	pool->lane_stride_elements = lane_stride;
	pool->layer_stride_elements = layer_stride;
	status = SparkStageModuleDeviceAllocate(&state->ledger,lane_stride * (uint64_t)state->lane_capacity * sizeof(float),&pointer);
	if ( status != SPARK_STATUS_OK )
		return(status);
	pool->state_f32 = (float *)pointer;
	status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,(uint64_t)state->row_capacity * sizeof(uint32_t),&pointer);
	if ( status != SPARK_STATUS_OK )
		return(status);
	pool->state_cold_by_row = (uint32_t *)pointer;
	fprintf(stderr,"k3_stage kda_state_pool lanes=%u layers=%u bytes=%llu\n",state->lane_capacity,state->kda_layer_count,(unsigned long long)(lane_stride * (uint64_t)state->lane_capacity * sizeof(float)));
	return(SPARK_STATUS_OK);
}

/*
 * The module owns the latent cache and hands each lane a contiguous run of
 * physical blocks, plus one scratch block that absorbs the writes of padded
 * prefill rows. A serving layer that wants shared prefixes supplies its own
 * table through the frame context instead.
 */
static SparkStatus SparkK3ModuleAllocateMlaCache(SparkK3ModuleState *state)
{
	SparkK3MlaBlockTableView *table = &state->owned_block_table;
	SparkStatus status;
	uint64_t cache_elements;
	uint32_t lane,block;
	void *pointer = 0;
	state->blocks_per_lane = (state->max_context_tokens + SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS - 1u) / SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS;
	state->node_context.mla_cache_block_count = (state->lane_capacity * state->blocks_per_lane) + 1u;
	state->scratch_block_index = state->node_context.mla_cache_block_count - 1u;
	cache_elements = (uint64_t)state->mla_layer_count * state->node_context.mla_cache_block_count * SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS * SPARK_K3_MODEL_MLA_CACHE_TOKEN_ELEMENTS;
	status = SparkStageModuleDeviceAllocate(&state->ledger,cache_elements * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status != SPARK_STATUS_OK )
		return(status);
	state->node_context.mla_cache_bf16 = pointer;
	state->host_block_indices = (uint32_t *)malloc((size_t)state->lane_capacity * state->blocks_per_lane * sizeof(uint32_t));
	state->host_lane_block_counts = (uint32_t *)malloc((size_t)state->lane_capacity * sizeof(uint32_t));
	if ( state->host_block_indices == 0 || state->host_lane_block_counts == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (lane = 0; lane < state->lane_capacity; lane++)
	{
		for (block = 0; block < state->blocks_per_lane; block++)
			state->host_block_indices[(lane * state->blocks_per_lane) + block] = (lane * state->blocks_per_lane) + block;
		state->host_lane_block_counts[lane] = state->blocks_per_lane;
	}
	table->abi_version = SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TABLE_ABI_VERSION;
	table->descriptor_bytes = (uint32_t)sizeof(SparkK3MlaBlockTableView);
	table->block_token_count = SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS;
	table->lane_count = state->lane_capacity;
	table->lane_stride = state->blocks_per_lane;
	table->lane_capacity = state->lane_capacity;
	table->host_physical_block_indices = state->host_block_indices;
	table->host_lane_physical_block_counts = state->host_lane_block_counts;
	status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)state->lane_capacity * state->blocks_per_lane * sizeof(uint32_t),&pointer);
	if ( status != SPARK_STATUS_OK )
		return(status);
	table->physical_block_indices = (const uint32_t *)pointer;
	status = SparkStageModuleCudaStatus(SPARK_K3_MODULE_TAG,cudaMemcpy(pointer,state->host_block_indices,(size_t)state->lane_capacity * state->blocks_per_lane * sizeof(uint32_t),cudaMemcpyHostToDevice),"block_table_h2d");
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)state->lane_capacity * sizeof(uint32_t),&pointer);
	if ( status != SPARK_STATUS_OK )
		return(status);
	table->lane_physical_block_counts = (const uint32_t *)pointer;
	status = SparkStageModuleCudaStatus(SPARK_K3_MODULE_TAG,cudaMemcpy(pointer,state->host_lane_block_counts,(size_t)state->lane_capacity * sizeof(uint32_t),cudaMemcpyHostToDevice),"block_counts_h2d");
	if ( status != SPARK_STATUS_OK )
		return(status);
	fprintf(stderr,"k3_stage mla_cache layers=%u blocks=%u tokens_per_lane=%u bytes=%llu\n",state->mla_layer_count,state->node_context.mla_cache_block_count,state->blocks_per_lane * SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS,(unsigned long long)(cache_elements * SPARK_K3_MODEL_BF16_ELEMENT_BYTES));
	return(SPARK_STATUS_OK);
}

/*
 * Per-slot activation buffers. Every buffer is sized for row_capacity rows,
 * which is also the representation stride the launchers compute from
 * row_capacity, so the two can never disagree.
 */
static SparkStatus SparkK3ModuleAllocateSlotBuffers(SparkK3ModuleState *state, SparkK3PipelineSlot *slot)
{
	uint64_t rows = state->row_capacity;
	SparkStatus status;
	void *pointer = 0;
	status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * sizeof(uint32_t),&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->input_token_ids = (const uint32_t *)pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * sizeof(uint32_t),&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->output_token_ids = (uint32_t *)pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * sizeof(uint32_t),&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->slot_mapping = (const uint32_t *)pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * sizeof(uint32_t),&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->lane_indices = (const uint32_t *)pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * sizeof(uint32_t),&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->context_lengths = (const uint32_t *)pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,(uint64_t)SPARK_K3_MODEL_ATTNRES_MAX_REPRESENTATIONS * rows * SPARK_K3_MODEL_HIDDEN_BF16_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->attnres_representations_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * SPARK_K3_MODEL_HIDDEN_BF16_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->mixed_hidden_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * SPARK_K3_MODEL_HIDDEN_BF16_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->normalized_hidden_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * SPARK_K3_MODEL_HIDDEN_BF16_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->attention_output_hidden_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * SPARK_K3_MODEL_HIDDEN_BF16_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->moe_output_hidden_bf16 = pointer;
	return(status);
}

static SparkStatus SparkK3ModuleAllocateSlotKdaBuffers(SparkK3ModuleState *state, SparkK3PipelineSlot *slot)
{
	uint64_t rows = state->row_capacity;
	SparkStatus status;
	void *pointer = 0;
	status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * SPARK_K3_MODEL_KDA_QK_DIMENSION * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->kda_query_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * SPARK_K3_MODEL_KDA_QK_DIMENSION * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->kda_key_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * SPARK_K3_MODEL_KDA_VALUE_DIMENSION * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->kda_value_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * SPARK_K3_MODEL_KDA_QK_DIMENSION * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->kda_log_decay_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * SPARK_K3_MODEL_KDA_HEAD_COUNT * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->kda_beta_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * SPARK_K3_MODEL_KDA_LOW_RANK_DIMENSION * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->kda_decay_low_rank_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * SPARK_K3_MODEL_KDA_LOW_RANK_DIMENSION * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->kda_gate_low_rank_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * SPARK_K3_MODEL_KDA_VALUE_DIMENSION * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->kda_core_output_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * SPARK_K3_MODEL_KDA_VALUE_DIMENSION * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->kda_gate_bf16 = pointer;
	return(status);
}

static SparkStatus SparkK3ModuleAllocateSlotMlaBuffers(SparkK3ModuleState *state, SparkK3PipelineSlot *slot)
{
	uint64_t rows = state->row_capacity;
	uint64_t latent_elements = (uint64_t)SPARK_K3_MODEL_MLA_HEAD_COUNT * SPARK_K3_MODEL_MLA_LATENT_DIMENSION;
	SparkStatus status;
	void *pointer = 0;
	status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * SPARK_K3_MODEL_MLA_QUERY_A_DIMENSION * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->mla_query_a_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * SPARK_K3_MODEL_MLA_QUERY_B_DIMENSION * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->mla_query_b_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * latent_elements * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->mla_query_latent_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * SPARK_K3_MODEL_MLA_KV_A_DIMENSION * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->mla_kv_a_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * latent_elements * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->mla_attention_latent_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * SPARK_K3_MODEL_MLA_ATTENTION_PROJECTION_DIMENSION * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->mla_head_output_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * SPARK_K3_MODEL_MLA_HEAD_COUNT * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->mla_head_gate_bf16 = pointer;
	return(status);
}

static SparkStatus SparkK3ModuleAllocateSlotMoeBuffers(SparkK3ModuleState *state, SparkK3PipelineSlot *slot)
{
	uint64_t rows = state->row_capacity;
	SparkStatus status;
	void *pointer = 0;
	status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * SPARK_K3_MODEL_MOE_TOP_K * sizeof(uint32_t),&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->moe_topk_expert_ids = (uint32_t *)pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * SPARK_K3_MODEL_MOE_TOP_K * sizeof(float),&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->moe_topk_weights_f32 = (float *)pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * SPARK_K3_MODEL_MOE_EXPERT_COUNT * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->moe_gate_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * SPARK_K3_MODULE_MAX_MOE_INTERMEDIATE_ROW_ELEMENTS * SPARK_K3_MODEL_BF16_ELEMENT_BYTES,&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->moe_intermediate_bf16 = pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,(SPARK_K3_MODEL_MOE_EXPERT_COUNT + 1u) * sizeof(uint32_t),&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->moe_expert_offsets = (uint32_t *)pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * SPARK_K3_MODEL_MOE_TOP_K * sizeof(uint32_t),&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->moe_grouped_rows = (uint32_t *)pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * SPARK_K3_MODEL_MOE_TOP_K * sizeof(uint32_t),&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->moe_grouped_weight_slots = (uint32_t *)pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * SPARK_K3_MODEL_MOE_TOP_K * sizeof(uint32_t),&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->moe_inverse_map = (uint32_t *)pointer;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,rows * SPARK_K3_MODEL_RESTRICTED_VOCAB_COUNT * sizeof(float),&pointer);
	if ( status == SPARK_STATUS_OK )
		slot->restricted_logits_f32 = (float *)pointer;
	return(status);
}

static SparkStatus SparkK3ModuleAllocatePipelineSlots(SparkK3ModuleState *state)
{
	SparkStatus status = SPARK_STATUS_OK;
	cudaStream_t stream;
	uint32_t index;
	for (index = 0; index < state->node_context.pipeline_slot_count && status == SPARK_STATUS_OK; index++)
	{
		status = SparkStageModuleCudaStatus(SPARK_K3_MODULE_TAG,cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"cudaStreamCreate");
		if ( status != SPARK_STATUS_OK )
			return(status);
		state->pipeline_slots[index].cuda_stream = stream;
		status = SparkK3ModuleAllocateSlotBuffers(state,&state->pipeline_slots[index]);
		if ( status == SPARK_STATUS_OK )
			status = SparkK3ModuleAllocateSlotKdaBuffers(state,&state->pipeline_slots[index]);
		if ( status == SPARK_STATUS_OK )
			status = SparkK3ModuleAllocateSlotMlaBuffers(state,&state->pipeline_slots[index]);
		if ( status == SPARK_STATUS_OK )
			status = SparkK3ModuleAllocateSlotMoeBuffers(state,&state->pipeline_slots[index]);
	}
	return(status);
}

static SparkStatus SparkK3ModuleAllocateSlotStaging(SparkK3ModuleState *state, SparkK3ModuleSlotStaging *staging)
{
	void *pointer = 0;
	SparkStatus status;
	staging->row_token_ids = (uint32_t *)malloc((size_t)state->row_capacity * sizeof(uint32_t));
	staging->row_slot_mapping = (uint32_t *)malloc((size_t)state->row_capacity * sizeof(uint32_t));
	staging->row_lane_indices = (uint32_t *)malloc((size_t)state->row_capacity * sizeof(uint32_t));
	staging->row_context_lengths = (uint32_t *)malloc((size_t)state->row_capacity * sizeof(uint32_t));
	staging->row_cold_flags = (uint32_t *)malloc((size_t)state->row_capacity * sizeof(uint32_t));
	staging->row_output_token_ids = (uint32_t *)malloc((size_t)state->row_capacity * sizeof(uint32_t));
	if ( staging->row_token_ids == 0 || staging->row_slot_mapping == 0 || staging->row_lane_indices == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( staging->row_context_lengths == 0 || staging->row_cold_flags == 0 || staging->row_output_token_ids == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,sizeof(int32_t),&pointer);
	if ( status != SPARK_STATUS_OK )
		return(status);
	staging->device_sequence_token_count = (int32_t *)pointer;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkK3ModuleAllocateHostStaging(SparkK3ModuleState *state)
{
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t index;
	for (index = 0; index < state->node_context.pipeline_slot_count && status == SPARK_STATUS_OK; index++)
		status = SparkK3ModuleAllocateSlotStaging(state,&state->slot_staging[index]);
	return(status);
}

static void SparkK3ModuleConfigureNodeContext(SparkK3ModuleState *state)
{
	SparkK3ResidentDecodeStageNodeContext *node = &state->node_context;
	node->abi_version = SPARK_K3_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION;
	node->row_capacity = state->row_capacity;
	node->max_prefill_tokens = SPARK_K3_MODEL_KDA_CHUNK_TOKENS;
	node->rms_norm_epsilon = SPARK_K3_MODEL_RMS_NORM_EPSILON;
	node->moe_routed_scaling_factor = SPARK_K3_MODEL_MOE_ROUTED_SCALING_FACTOR;
	node->moe_norm_topk_prob = SPARK_K3_MODEL_MOE_NORM_TOPK_PROB;
	node->enable_cuda_graph_replay = 0u;
	node->attnres_sites_by_layer = state->attnres_sites;
	node->attnres_final_site = &state->attnres_final_site;
	node->attention_norm_weights_by_layer_bf16 = state->attention_norm_by_layer;
	node->mlp_norm_weights_by_layer_bf16 = state->mlp_norm_by_layer;
	node->kda_weights_by_layer = state->kda_weights;
	node->mla_weights_by_layer = state->mla_weights;
	node->moe_weights_by_layer = state->moe_weights;
	node->restricted_vocab_count = SPARK_K3_MODEL_RESTRICTED_VOCAB_COUNT;
	node->pipeline_slots = state->pipeline_slots;
}

static SparkStatus SparkK3ModuleReadConfiguration(
    SparkK3ModuleState *state,
    const char **pack_path)
{
    SparkStatus status;
    uint32_t slots;

    status = SparkStageModuleEnvironmentUnsigned(
        SPARK_K3_MODULE_TAG,
        SPARK_K3_UNQUALIFIED_EXECUTION_ENVIRONMENT,
        1u,
        1u,
        &state->unqualified_execution_enabled);
    if (status != SPARK_STATUS_OK ||
        state->unqualified_execution_enabled != 1u)
    {
        fprintf(
            stderr,
            "k3_stage qualification_required set %s=1 only for controlled bring-up; "
            "this driver has no retained GPU qualification receipt\n",
            SPARK_K3_UNQUALIFIED_EXECUTION_ENVIRONMENT);
        return SPARK_STATUS_MODULE_NOT_VALIDATED;
    }

    status = SparkStageModuleEnvironmentText(
        SPARK_K3_MODULE_TAG,
        "K3_STAGE_PACK",
        pack_path);
    if (status == SPARK_STATUS_OK)
    {
        status = SparkStageModuleEnvironmentUnsigned(SPARK_K3_MODULE_TAG,
            "K3_MAX_LANES",
            1u,
            SPARK_K3_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,
            &state->lane_capacity);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkStageModuleEnvironmentUnsigned(SPARK_K3_MODULE_TAG,
            "K3_MAX_CONTEXT_TOKENS",
            SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS,
            SPARK_K3_MODEL_MAXIMUM_CONTEXT_TOKENS,
            &state->max_context_tokens);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkStageModuleEnvironmentUnsigned(SPARK_K3_MODULE_TAG,
            "K3_PIPELINE_SLOTS",
            1u,
            SPARK_K3_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,
            &slots);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkStageModuleEnvironmentUnsigned(SPARK_K3_MODULE_TAG,
            "K3_STAGE_COUNT",
            1u,
            SPARK_K3_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT,
            &state->stage_count);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkStageModuleEnvironmentUnsigned(SPARK_K3_MODULE_TAG,
            "K3_STAGE_INDEX",
            0u,
            SPARK_K3_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT - 1u,
            &state->stage_index);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkStageModuleEnvironmentUnsigned(SPARK_K3_MODULE_TAG,
            "K3_STAGE_FIRST_LAYER",
            0u,
            SPARK_K3_MODEL_LAYER_COUNT - 1u,
            &state->node_context.first_layer_index);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkStageModuleEnvironmentUnsigned(SPARK_K3_MODULE_TAG,
            "K3_STAGE_LAYER_COUNT",
            1u,
            SPARK_K3_MODEL_LAYER_COUNT,
            &state->node_context.layer_count);
    }
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    state->node_context.pipeline_slot_count = slots;
    state->row_capacity = state->lane_capacity > SPARK_K3_MODEL_KDA_CHUNK_TOKENS
        ? state->lane_capacity
        : SPARK_K3_MODEL_KDA_CHUNK_TOKENS;
    return SPARK_STATUS_OK;
}

// The stage's slice: ownership of the embedding and the head follows the
// slice edges, and the slice's place in the pipeline must agree with the
// stage index so a misdeployed ring fails at initialize, not at the first
// frame.
static SparkStatus SparkK3ModuleValidateSlice(SparkK3ModuleState *state)
{
	SparkK3ResidentDecodeStageNodeContext *node = &state->node_context;
	if ( state->stage_index >= state->stage_count || node->first_layer_index + node->layer_count > SPARK_K3_MODEL_LAYER_COUNT )
	{
		fprintf(stderr,"k3_stage config_slice_invalid stage=%u/%u slice=%u+%u\n",state->stage_index,state->stage_count,node->first_layer_index,node->layer_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	node->owns_embedding = node->first_layer_index == 0u ? 1u : 0u;
	node->owns_final_head = node->first_layer_index + node->layer_count == SPARK_K3_MODEL_LAYER_COUNT ? 1u : 0u;
	if ( (state->stage_index == 0u) != (node->owns_embedding != 0u) || (state->stage_index + 1u == state->stage_count) != (node->owns_final_head != 0u) )
	{
		fprintf(stderr,"k3_stage config_position_mismatch stage=%u/%u slice=%u+%u\n",state->stage_index,state->stage_count,node->first_layer_index,node->layer_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	return(SPARK_STATUS_OK);
}

// The AttnRes boundary entering layer L: every completed block (the
// embedding block among them) plus the running partial.
static uint32_t SparkK3ModuleBoundaryRepresentations(uint32_t layer_index)
{
	return(SPARK_K3_MODEL_ATTNRES_COMPLETED_BLOCKS_BEFORE_LAYER(layer_index) + 1u);
}

SparkStatus SparkK3ResidentDecodeStageInitialize(const SparkFirmwareModuleConfiguration *configuration, const SparkFirmwareModuleHostServices *host_services, void **module_state)
{
	SparkK3ModuleState *state;
	const char *pack_path = 0;
	SparkStatus status;
	status = SparkFirmwareModuleValidateInitialization(configuration,host_services,module_state);
	if ( status != SPARK_STATUS_OK )
		return(status);
    state = (SparkK3ModuleState *)calloc(1u, sizeof(*state));
    if (state == 0)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    state->host_services = *host_services;
    state->ledger.module_tag = SPARK_K3_MODULE_TAG;
    atomic_init(&state->submitted_count, 0u);
    atomic_init(&state->completed_count, 0u);
    atomic_init(&state->rejected_count, 0u);
    atomic_init(&state->failed_count, 0u);
	status = SparkK3ModuleReadConfiguration(state,&pack_path);
	if ( status == SPARK_STATUS_OK )
	{
		SparkStageModuleAtomicStateArrayInitialize(
			state->slot_states,
			state->node_context.pipeline_slot_count);
		SparkStageModuleAtomicStateArrayInitialize(
			state->lane_states,
			state->lane_capacity);
		status = SparkK3ModuleValidateSlice(state);
	}
	if (status == SPARK_STATUS_OK)
		status = SparkK3ConfigureCudaKernels();
	if ( status == SPARK_STATUS_OK )
	{
		SparkK3ModuleAssignLayerOrdinals(state);
		SparkK3ModuleConfigureNodeContext(state);
		status = SparkK3ModuleLoadPack(state,pack_path);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleValidateResidentWeights(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleAllocateKdaStatePool(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleAllocateMlaCache(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleAllocatePipelineSlots(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleAllocateHostStaging(state);
	if ( status != SPARK_STATUS_OK )
	{
		SparkK3ResidentDecodeStageDestroy(state);
		return(status);
	}
	fprintf(stderr,"k3_stage initialized stage=%u/%u slice=%u+%u lanes=%u rows=%u context=%u slots=%u boundary_in=%u boundary_out=%u device_bytes=%llu\n",state->stage_index,state->stage_count,state->node_context.first_layer_index,state->node_context.layer_count,state->lane_capacity,state->row_capacity,state->max_context_tokens,state->node_context.pipeline_slot_count,state->stage_index > 0u ? SparkK3ModuleBoundaryRepresentations(state->node_context.first_layer_index) : 0u,state->node_context.owns_final_head == 0u ? SparkK3ModuleBoundaryRepresentations(state->node_context.first_layer_index + state->node_context.layer_count) : 0u,(unsigned long long)state->ledger.device_bytes_resident);
	*module_state = state;
	return(SPARK_STATUS_OK);
}

/*
 * Row metadata for one dispatch. A decode row is one lane's next token, so
 * the row's cache slot follows that lane's own position. A prefill dispatch
 * puts many consecutive tokens of one lane in flight, and pads the batch to a
 * full KDA chunk: padded rows carry token id zero, park their latent writes
 * in the scratch block and attend only to themselves, so they compute
 * finite garbage that nothing reads instead of poisoning the cache.
 */
static SparkStatus SparkK3ModuleCacheSlotForPosition(
    const SparkK3ModuleState *state,
    const SparkK3MlaBlockTableView *block_table,
    uint32_t lane,
    uint32_t position,
    uint32_t *cache_slot)
{
    uint32_t logical_block;
    uint32_t physical_block;
    uint64_t table_index;
    uint64_t cache_slot_value;

    if (state == 0 || block_table == 0 || cache_slot == 0 ||
        block_table->host_physical_block_indices == 0 ||
        lane >= block_table->lane_count ||
        block_table->lane_stride == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    logical_block =
        position / SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS;
    if (logical_block >= block_table->lane_stride ||
        logical_block >= block_table->host_lane_physical_block_counts[lane])
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    table_index = (uint64_t)lane * block_table->lane_stride + logical_block;
    physical_block = block_table->host_physical_block_indices[table_index];
    if (physical_block == SPARK_K3_RESIDENT_DECODE_STAGE_NO_BLOCK ||
        physical_block >= state->node_context.mla_cache_block_count)
    {
        fprintf(
            stderr,
            "k3_stage block_table_invalid lane=%u logical_block=%u "
            "physical_block=%u cache_block_count=%u\n",
            lane,
            logical_block,
            physical_block,
            state->node_context.mla_cache_block_count);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    cache_slot_value =
        (uint64_t)physical_block *
            SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS +
        (position % SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS);
    if (cache_slot_value > UINT32_MAX)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    *cache_slot = (uint32_t)cache_slot_value;
    return SPARK_STATUS_OK;
}

/*
 * Wire token ids feed the embedding gather directly, so an id outside the
 * vocabulary is an out-of-bounds device read; it is rejected here, on the
 * host, before anything is uploaded.
 */
static SparkStatus SparkK3ModuleValidateWireTokens(const uint32_t *token_ids, uint32_t token_count)
{
	uint32_t row;
	for (row = 0; row < token_count; row++)
		if ( token_ids[row] >= SPARK_K3_MODEL_OUTPUT_VOCAB_COUNT )
		{
			fprintf(stderr,"k3_stage token_out_of_vocab row=%u token=%u vocab=%u\n",row,token_ids[row],SPARK_K3_MODEL_OUTPUT_VOCAB_COUNT);
			return(SPARK_STATUS_INVALID_ARGUMENT);
		}
	return(SPARK_STATUS_OK);
}

// Batched decode rows: one token for each of row_count DISTINCT lanes,
// every row's cache slot and context length from its own lane position.
static SparkStatus SparkK3ModuleFillDecodeBatchMetadata(
    SparkK3ModuleState *state,
    SparkK3ModuleSlotStaging *staging,
    const SparkK3DecodeBatchView *batch,
    const SparkK3MlaBlockTableView *block_table)
{
    uint32_t row;

    for (row = 0u; row < batch->row_count; row++)
    {
        uint32_t lane;
        uint32_t position;
        SparkStatus status;

        lane = batch->row_lane_indices[row];
        if (lane >= state->lane_capacity ||
            batch->row_positions[row] >= (uint64_t)state->max_context_tokens)
        {
            fprintf(
                stderr,
                "k3_stage decode_batch_row_invalid row=%u lane=%u "
                "position=%llu\n",
                row,
                lane,
                (unsigned long long)batch->row_positions[row]);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        position = (uint32_t)batch->row_positions[row];
        status = SparkK3ModuleCacheSlotForPosition(
            state,
            block_table,
            lane,
            position,
            &staging->row_slot_mapping[row]);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        staging->row_lane_indices[row] = lane;
        staging->row_context_lengths[row] = position + 1u;
        staging->row_cold_flags[row] = position == 0u ? 1u : 0u;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkK3ModuleFillRowMetadata(
    SparkK3ModuleState *state,
    SparkK3ModuleSlotStaging *staging,
    const SparkModelDriverFrame *frame,
    const SparkK3MlaBlockTableView *block_table,
    uint32_t lane,
    uint32_t token_count,
    uint32_t padded_rows)
{
    uint32_t base_position;
    uint32_t row;

    base_position = (uint32_t)frame->sequence_position;
    if ((uint64_t)base_position + token_count > state->max_context_tokens)
    {
        fprintf(
            stderr,
            "k3_stage context_exhausted lane=%u position=%u tokens=%u "
            "capacity=%u\n",
            lane,
            base_position,
            token_count,
            state->max_context_tokens);
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    for (row = 0u; row < padded_rows; row++)
    {
        staging->row_lane_indices[row] = lane;
        if (row < token_count)
        {
            SparkStatus status;
            uint32_t position;

            position = base_position + row;
            status = SparkK3ModuleCacheSlotForPosition(
                state,
                block_table,
                lane,
                position,
                &staging->row_slot_mapping[row]);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
            staging->row_context_lengths[row] = position + 1u;
        }
        else
        {
            uint64_t scratch_slot;

            scratch_slot =
                (uint64_t)state->scratch_block_index *
                    SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS +
                (row % SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS);
            if (scratch_slot > UINT32_MAX)
            {
                return SPARK_STATUS_CAPACITY_EXCEEDED;
            }
            staging->row_token_ids[row] = 0u;
            staging->row_slot_mapping[row] = (uint32_t)scratch_slot;
            staging->row_context_lengths[row] = 1u;
        }
    }
    return SPARK_STATUS_OK;
}


static SparkStatus SparkK3ModuleUploadRowMetadata(
    SparkK3ModuleSlotStaging *staging,
    const SparkK3PipelineSlot *slot,
    uint32_t padded_row_count,
    uint32_t token_count,
    uint32_t upload_token_ids,
    cudaStream_t stream)
{
    uint64_t row_bytes;
    SparkStatus status;

    if (staging == 0 || slot == 0 || padded_row_count == 0u ||
        token_count == 0u || token_count > padded_row_count ||
        upload_token_ids > 1u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    row_bytes = (uint64_t)padded_row_count * sizeof(uint32_t);
    staging->sequence_token_count = (int32_t)token_count;
    status = SPARK_STATUS_OK;
    if (upload_token_ids != 0u)
    {
        status = SparkStageModuleCudaStatus(
            SPARK_K3_MODULE_TAG,
            cudaMemcpyAsync(
                (void *)slot->input_token_ids,
                staging->row_token_ids,
                (size_t)row_bytes,
                cudaMemcpyHostToDevice,
                stream),
            "token_ids_h2d");
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkStageModuleCudaStatus(
            SPARK_K3_MODULE_TAG,
            cudaMemcpyAsync(
                (void *)slot->slot_mapping,
                staging->row_slot_mapping,
                (size_t)row_bytes,
                cudaMemcpyHostToDevice,
                stream),
            "slot_mapping_h2d");
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkStageModuleCudaStatus(
            SPARK_K3_MODULE_TAG,
            cudaMemcpyAsync(
                (void *)slot->lane_indices,
                staging->row_lane_indices,
                (size_t)row_bytes,
                cudaMemcpyHostToDevice,
                stream),
            "lane_indices_h2d");
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkStageModuleCudaStatus(
            SPARK_K3_MODULE_TAG,
            cudaMemcpyAsync(
                (void *)slot->context_lengths,
                staging->row_context_lengths,
                (size_t)row_bytes,
                cudaMemcpyHostToDevice,
                stream),
            "context_lengths_h2d");
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkStageModuleCudaStatus(
            SPARK_K3_MODULE_TAG,
            cudaMemcpyAsync(
                staging->device_sequence_token_count,
                &staging->sequence_token_count,
                sizeof(staging->sequence_token_count),
                cudaMemcpyHostToDevice,
                stream),
            "token_counts_h2d");
    }
    return status;
}


/*
 * A sequence at position zero has no history: its KDA state is stale from a
 * previous tenant of the lane and must be treated as zero on first touch. The
 * flag is per row because both KDA kernels read it per row.
 */
// Decode fills the per-row cold flags from the batch view; prefill marks
// its single sequence uniformly. Both paths upload from here.
static SparkStatus SparkK3ModuleUploadColdFlags(SparkK3ModuleState *state, SparkK3ModuleSlotStaging *staging, uint32_t padded_rows, cudaStream_t stream)
{
	SparkK3KdaStatePool *pool = &state->node_context.kda_state_pool;
	return(SparkStageModuleCudaStatus(SPARK_K3_MODULE_TAG,cudaMemcpyAsync(pool->state_cold_by_row,staging->row_cold_flags,(size_t)padded_rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream),"cold_flags_h2d"));
}

/*
 * The AttnRes boundary over the wire, in the representation array's own
 * layout: the sender's live prefix is boundary_representations x
 * representation_stride bf16 elements, the receiver copies it straight
 * into its own array. Per-row accounting rides sideband_bytes_per_sequence
 * so a slice mismatch between neighbors fails loudly.
 */
static SparkStatus SparkK3ModuleReceiveBoundary(SparkK3ModuleState *state, const SparkK3PipelineSlot *slot, SparkK3ResidentDecodeStageFrameContext *context, uint32_t padded_rows, cudaStream_t stream)
{
	uint32_t boundary = SparkK3ModuleBoundaryRepresentations(state->node_context.first_layer_index);
	uint64_t representation_stride = (uint64_t)state->row_capacity * SPARK_K3_MODEL_HIDDEN_DIMENSION;
	SparkHiddenTransportPacket *packet = &context->hidden_input_packet;
	SparkStatus status;
	status = context->hidden_input_post_receive_function(context->hidden_input_transport_session,packet);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( packet->active_sequence_count != padded_rows || packet->hidden_dimension != SPARK_K3_MODEL_HIDDEN_DIMENSION || packet->sideband_payload == 0 )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( packet->sideband_kind != SPARK_K3_RESIDENT_DECODE_STAGE_SIDEBAND_KIND_ATTNRES || packet->sideband_bytes_per_sequence != (uint64_t)boundary * SPARK_K3_MODEL_HIDDEN_DIMENSION * SPARK_K3_MODEL_BF16_ELEMENT_BYTES )
	{
		fprintf(stderr,"k3_stage boundary_mismatch kind=%u bytes=%llu expected_reps=%u\n",packet->sideband_kind,(unsigned long long)packet->sideband_bytes_per_sequence,boundary);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	return(SparkStageModuleCudaStatus(SPARK_K3_MODULE_TAG,cudaMemcpyAsync(slot->attnres_representations_bf16,packet->sideband_payload,(size_t)((uint64_t)boundary * representation_stride * SPARK_K3_MODEL_BF16_ELEMENT_BYTES),cudaMemcpyDeviceToDevice,stream),"boundary_d2d"));
}

static SparkStatus SparkK3ModuleSendBoundary(SparkK3ModuleState *state, const SparkK3PipelineSlot *slot, SparkK3ResidentDecodeStageFrameContext *context, uint32_t padded_rows, cudaStream_t stream)
{
	uint32_t boundary = SparkK3ModuleBoundaryRepresentations(state->node_context.first_layer_index + state->node_context.layer_count);
	uint64_t representation_stride = (uint64_t)state->row_capacity * SPARK_K3_MODEL_HIDDEN_DIMENSION;
	SparkHiddenTransportPacket *packet = &context->hidden_output_packet;
	packet->active_sequence_count = padded_rows;
	packet->hidden_dimension = SPARK_K3_MODEL_HIDDEN_DIMENSION;
	packet->bytes_per_sequence = SPARK_K3_MODEL_HIDDEN_DIMENSION * SPARK_K3_MODEL_BF16_ELEMENT_BYTES;
	packet->hidden_bf16 = (uint8_t *)slot->attnres_representations_bf16 + ((uint64_t)(boundary - 1u) * representation_stride * SPARK_K3_MODEL_BF16_ELEMENT_BYTES);
	packet->sideband_payload = slot->attnres_representations_bf16;
	packet->sideband_kind = SPARK_K3_RESIDENT_DECODE_STAGE_SIDEBAND_KIND_ATTNRES;
	packet->sideband_bytes_per_sequence = (uint64_t)boundary * SPARK_K3_MODEL_HIDDEN_DIMENSION * SPARK_K3_MODEL_BF16_ELEMENT_BYTES;
	packet->cuda_stream = stream;
	return(context->hidden_output_send_function(context->hidden_output_transport_session,packet));
}

/*
 * Every rowwise kernel in the layer stack runs at the padded width, so a
 * prefill's padded rows compute deterministic finite garbage end to end
 * instead of dragging stale device memory through the mixes; the true token
 * count matters only to the chunk kernel's masking (via the staged device
 * count) and to the host wire I/O in Execute.
 */
static SparkStatus SparkK3ModuleRunAttention(SparkK3ModuleState *state, const SparkK3PipelineSlot *slot, SparkK3ModuleSlotStaging *staging, const SparkK3MlaBlockTableView *block_table, uint32_t layer_index, uint32_t row_count, uint32_t is_prefill, uint32_t carry_state, cudaStream_t stream)
{
	const SparkK3ResidentDecodeStageNodeContext *node = &state->node_context;
	SparkStatus status;
	if ( SPARK_K3_MODEL_LAYER_IS_KDA(layer_index) == 0u )
	{
		if ( is_prefill != 0u )
			return(SparkK3LaunchMlaPrefill(node,slot,&state->mla_weights[layer_index],block_table,state->mla_ordinal_by_layer[layer_index],row_count,stream));
		return(SparkK3LaunchMlaDecode(node,slot,&state->mla_weights[layer_index],block_table,state->mla_ordinal_by_layer[layer_index],row_count,stream));
	}
	status = SparkK3LaunchKdaMaterialize(node,slot,&state->kda_weights[layer_index],row_count,stream);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( is_prefill != 0u )
		status = SparkK3LaunchKdaChunk(node,slot,state->kda_ordinal_by_layer[layer_index],1u,SPARK_K3_MODEL_KDA_CHUNK_TOKENS,staging->device_sequence_token_count,carry_state,1u,stream);
	else
		status = SparkK3LaunchKdaDecodeStep(node,slot,state->kda_ordinal_by_layer[layer_index],row_count,stream);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkK3LaunchKdaFinish(node,slot,&state->kda_weights[layer_index],row_count,stream));
}

static SparkStatus SparkK3ModuleRunMlp(SparkK3ModuleState *state, const SparkK3PipelineSlot *slot, uint32_t layer_index, uint32_t row_count, cudaStream_t stream)
{
	const SparkK3ResidentDecodeStageNodeContext *node = &state->node_context;
	SparkStatus status;
	if ( layer_index < SPARK_K3_MODEL_FIRST_ROUTED_LAYER )
		return(SparkK3LaunchDenseMlp(node,slot,&state->moe_weights[layer_index],row_count,stream));
	status = SparkK3LaunchMoeRoute(node,slot,&state->moe_weights[layer_index],row_count,stream);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkK3LaunchMoeExperts(node,slot,&state->moe_weights[layer_index],row_count,stream));
}

/*
 * One layer, in the order the published AttnRes forward() prescribes: mix the
 * completed blocks with the running partial, check the block boundary, run
 * attention on the normalized mixture, accumulate, mix again for the mlp
 * site, run the mlp, accumulate. Only the attention site can open a block.
 */
static SparkStatus SparkK3ModuleRunLayer(SparkK3ModuleState *state, const SparkK3PipelineSlot *slot, SparkK3ModuleSlotStaging *staging, const SparkK3MlaBlockTableView *block_table, uint32_t layer_index, uint32_t row_count, uint32_t is_prefill, uint32_t carry_state, cudaStream_t stream)
{
	const SparkK3ResidentDecodeStageNodeContext *node = &state->node_context;
	uint32_t site_base = layer_index * SPARK_K3_RESIDENT_DECODE_STAGE_ATTNRES_SITES_PER_LAYER;
	uint32_t completed = SPARK_K3_MODEL_ATTNRES_COMPLETED_BLOCKS_BEFORE_LAYER(layer_index);
	uint32_t opens = SPARK_K3_MODEL_ATTNRES_LAYER_OPENS_BLOCK(layer_index);
	SparkStatus status;
	status = SparkK3LaunchAttnResMix(node,slot,&state->attnres_sites[site_base + SPARK_K3_RESIDENT_DECODE_STAGE_ATTNRES_ATTENTION_SITE],completed + 1u,row_count,stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3LaunchRmsNorm(slot->mixed_hidden_bf16,state->attention_norm_by_layer[layer_index],slot->normalized_hidden_bf16,row_count,SPARK_K3_MODEL_HIDDEN_DIMENSION,node->rms_norm_epsilon,stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleRunAttention(state,slot,staging,block_table,layer_index,row_count,is_prefill,carry_state,stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3LaunchAttnResAccumulate(node,slot,slot->attention_output_hidden_bf16,opens,completed,row_count,stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3LaunchAttnResMix(node,slot,&state->attnres_sites[site_base + SPARK_K3_RESIDENT_DECODE_STAGE_ATTNRES_MLP_SITE],completed + opens + 1u,row_count,stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3LaunchRmsNorm(slot->mixed_hidden_bf16,state->mlp_norm_by_layer[layer_index],slot->normalized_hidden_bf16,row_count,SPARK_K3_MODEL_HIDDEN_DIMENSION,node->rms_norm_epsilon,stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3ModuleRunMlp(state,slot,layer_index,row_count,stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3LaunchAttnResAccumulate(node,slot,slot->moe_output_hidden_bf16,0u,completed + opens,row_count,stream);
	return(status);
}

static SparkStatus SparkK3ModuleRunStage(SparkK3ModuleState *state, const SparkK3PipelineSlot *slot, SparkK3ModuleSlotStaging *staging, const SparkK3MlaBlockTableView *block_table, uint32_t row_count, uint32_t is_prefill, uint32_t carry_state, cudaStream_t stream)
{
	const SparkK3ResidentDecodeStageNodeContext *node = &state->node_context;
	uint32_t final_completed = SPARK_K3_MODEL_ATTNRES_COMPLETED_BLOCKS_BEFORE_LAYER(SPARK_K3_MODEL_LAYER_COUNT - 1u) + SPARK_K3_MODEL_ATTNRES_LAYER_OPENS_BLOCK(SPARK_K3_MODEL_LAYER_COUNT - 1u);
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t layer_index;
	if ( node->owns_embedding != 0u )
		status = SparkK3LaunchEmbeddingGather(node,slot,row_count,stream);
	for (layer_index = node->first_layer_index; layer_index < node->first_layer_index + node->layer_count && status == SPARK_STATUS_OK; layer_index++)
		status = SparkK3ModuleRunLayer(state,slot,staging,block_table,layer_index,row_count,is_prefill,carry_state,stream);
	if ( node->owns_final_head == 0u )
		return(status);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3LaunchAttnResMix(node,slot,&state->attnres_final_site,final_completed + 1u,row_count,stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkK3LaunchRestrictedLogits(node,slot,row_count,stream);
	return(status);
}

/*
 * Frame contract for version 1.
 *
 * One frame is a synchronous stage dispatch. Persistent KDA/MLA ownership is
 * named only by the model-specific frame context: prefill_lane_index for a
 * single-sequence prefill, or the decode batch's row_lane_indices for decode.
 * The generic driver dispatch slot remains an execution-resource ticket and
 * is deliberately not reused as a model-state lane.
 */

static SparkStatus SparkK3ModuleValidateFrameShape(
    const SparkK3ModuleState *state,
    const SparkModelDriverFrame *frame,
    uint32_t *is_prefill,
    uint32_t *row_count)
{
    const uint32_t known_frame_flags =
        SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL;
    uint32_t expected_buffer_count;
    uint32_t output_buffer_index;
    uint32_t prefill;
    uint64_t token_bytes;
    SparkStatus status;

    if (state == 0 || frame == 0 || is_prefill == 0 || row_count == 0 ||
        frame->program_id == 0u || frame->reserved != 0u ||
        (frame->flags & ~known_frame_flags) != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    expected_buffer_count =
        state->node_context.owns_embedding +
        state->node_context.owns_final_head;
    if (frame->buffer_count != expected_buffer_count ||
        (expected_buffer_count != 0u && frame->buffers == 0))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    prefill = (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u
        ? 1u
        : 0u;
    if (frame->active_slot_count == 0u ||
        frame->active_slot_count > state->lane_capacity ||
        frame->new_token_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (prefill != 0u)
    {
        if (frame->sequence_id == 0u ||
            frame->active_slot_count != 1u ||
            frame->new_token_count > SPARK_K3_MODEL_KDA_CHUNK_TOKENS)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    else if (frame->new_token_count != frame->active_slot_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (prefill != 0u &&
        SparkModelDriverRangeFitsWithinCapacity(
            frame->sequence_position,
            frame->new_token_count,
            state->max_context_tokens) == 0u)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }

    token_bytes = (uint64_t)frame->new_token_count * sizeof(uint32_t);
    if (state->node_context.owns_embedding != 0u)
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
    if (state->node_context.owns_final_head != 0u)
    {
        output_buffer_index = state->node_context.owns_embedding != 0u
            ? 1u
            : 0u;
        status = SparkModelDriverValidateBuffer(
            frame,
            output_buffer_index,
            1u,
            SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE,
            token_bytes);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }

    *is_prefill = prefill;
    *row_count = frame->new_token_count;
    return SPARK_STATUS_OK;
}


/*
 * Resolve the frame context. Transport flags must mirror the stage's place
 * in the pipeline: every non-first stage receives the AttnRes boundary,
 * every non-last stage sends it. Decode frames carry the batch view; the
 * block table override stays optional on any stage that runs MLA.
 */
static SparkStatus SparkK3ModuleResolveFrameContext(
    SparkK3ModuleState *state,
    const SparkModelDriverFrame *frame,
    uint32_t is_prefill,
    SparkK3ResidentDecodeStageFrameContext **context_out,
    const SparkK3MlaBlockTableView **block_table)
{
    const uint32_t known_context_flags =
        SPARK_K3_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MLA_BLOCK_TABLE |
        SPARK_K3_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT |
        SPARK_K3_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT |
        SPARK_K3_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW;
    SparkK3ResidentDecodeStageFrameContext *context;
    uint32_t needs_input;
    uint32_t needs_output;
    uint32_t row_index;
    uint32_t previous_row_index;

    if (state == 0 || frame == 0 || context_out == 0 || block_table == 0 ||
        frame->user_context == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    *block_table = &state->owned_block_table;
    *context_out = 0;
    needs_input = state->stage_index > 0u ? 1u : 0u;
    needs_output = state->stage_index + 1u < state->stage_count ? 1u : 0u;
    context = (SparkK3ResidentDecodeStageFrameContext *)frame->user_context;

    if (context->abi_version !=
            SPARK_K3_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION ||
        context->descriptor_bytes < (uint32_t)sizeof(*context) ||
        context->reserved0 != 0u ||
        (context->flags & ~known_context_flags) != 0u ||
        context->logical_lane_count == 0u ||
        context->logical_lane_count > state->lane_capacity)
    {
        return SPARK_STATUS_ABI_MISMATCH;
    }
    if (((context->flags &
          SPARK_K3_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT) != 0u) !=
            (needs_input != 0u) ||
        ((context->flags &
          SPARK_K3_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT) != 0u) !=
            (needs_output != 0u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((needs_input != 0u &&
         (context->hidden_input_transport_session == 0 ||
          context->hidden_input_post_receive_function == 0)) ||
        (needs_input == 0u &&
         (context->hidden_input_transport_session != 0 ||
          context->hidden_input_post_receive_function != 0)) ||
        (needs_output != 0u &&
         (context->hidden_output_transport_session == 0 ||
          context->hidden_output_send_function == 0)) ||
        (needs_output == 0u &&
         (context->hidden_output_transport_session != 0 ||
          context->hidden_output_send_function != 0)))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (is_prefill != 0u)
    {
        if ((context->flags &
             SPARK_K3_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW) != 0u ||
            context->decode_batch != 0 ||
            context->prefill_lane_index >= context->logical_lane_count)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    else
    {
        const SparkK3DecodeBatchView *decode_batch;

        if ((context->flags &
             SPARK_K3_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW) == 0u ||
            context->decode_batch == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        decode_batch = context->decode_batch;
        if (decode_batch->abi_version !=
                SPARK_K3_RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION ||
            decode_batch->descriptor_bytes < (uint32_t)sizeof(*decode_batch) ||
            decode_batch->reserved0 != 0u)
        {
            return SPARK_STATUS_ABI_MISMATCH;
        }
        if (decode_batch->row_count != frame->new_token_count ||
            decode_batch->row_lane_indices == 0 ||
            decode_batch->row_positions == 0 ||
            decode_batch->row_sequence_ids == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        for (row_index = 0u; row_index < decode_batch->row_count; ++row_index)
        {
            if (decode_batch->row_lane_indices[row_index] >=
                    context->logical_lane_count ||
                decode_batch->row_positions[row_index] >=
                    state->max_context_tokens ||
                decode_batch->row_sequence_ids[row_index] == 0u)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            for (previous_row_index = 0u;
                 previous_row_index < row_index;
                 ++previous_row_index)
            {
                if (decode_batch->row_lane_indices[previous_row_index] ==
                        decode_batch->row_lane_indices[row_index] ||
                    decode_batch->row_sequence_ids[previous_row_index] ==
                        decode_batch->row_sequence_ids[row_index])
                {
                    return SPARK_STATUS_INVALID_ARGUMENT;
                }
            }
        }
    }

    if ((context->flags &
         SPARK_K3_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_MLA_BLOCK_TABLE) != 0u)
    {
        if (context->mla_block_table == 0 ||
            context->mla_block_table->abi_version !=
                SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TABLE_ABI_VERSION ||
            context->mla_block_table->descriptor_bytes <
                (uint32_t)sizeof(*context->mla_block_table) ||
            context->mla_block_table->block_token_count !=
                SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS ||
            context->mla_block_table->lane_count < context->logical_lane_count ||
            context->mla_block_table->lane_count > state->lane_capacity ||
            context->mla_block_table->lane_capacity <
                context->mla_block_table->lane_count ||
            context->mla_block_table->lane_stride == 0u ||
            context->mla_block_table->physical_block_indices == 0 ||
            context->mla_block_table->lane_physical_block_counts == 0 ||
            context->mla_block_table->host_physical_block_indices == 0 ||
            context->mla_block_table->host_lane_physical_block_counts == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        *block_table = context->mla_block_table;
    }
    else if (context->mla_block_table != 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    *context_out = context;
    return SPARK_STATUS_OK;
}

/*
 * The attend kernel walks the lane's block list up to the frame's last
 * context position; a table whose lane does not own that many blocks would
 * send the kernel past its list, so the coverage is proven on the host from
 * the table's own mirror before anything launches.
 */

static SparkStatus SparkK3ModuleValidateBlockCoverage(
    const SparkK3ModuleState *state,
    const SparkK3MlaBlockTableView *block_table,
    uint64_t sequence_position,
    uint32_t lane,
    uint32_t token_count)
{
    uint64_t covered_token_count;
    uint32_t block_count;
    uint32_t logical_block;

    if (state == 0 || block_table == 0 || token_count == 0u ||
        lane >= block_table->lane_count ||
        block_table->host_physical_block_indices == 0 ||
        block_table->host_lane_physical_block_counts == 0 ||
        block_table->lane_stride == 0u ||
        SparkModelDriverRangeFitsWithinCapacity(
            sequence_position,
            token_count,
            state->max_context_tokens) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    covered_token_count = sequence_position + token_count;
    block_count = (uint32_t)((covered_token_count +
        SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS - 1u) /
        SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS);
    if (block_count > block_table->lane_stride ||
        block_count > block_table->host_lane_physical_block_counts[lane])
    {
        fprintf(
            stderr,
            "k3_stage block_table_short lane=%u needed=%u owned=%u stride=%u\n",
            lane,
            block_count,
            block_table->host_lane_physical_block_counts[lane],
            block_table->lane_stride);
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    for (logical_block = 0u; logical_block < block_count; logical_block++)
    {
        uint64_t table_index;
        uint32_t physical_block;

        table_index = (uint64_t)lane * block_table->lane_stride + logical_block;
        physical_block = block_table->host_physical_block_indices[table_index];
        if (physical_block == SPARK_K3_RESIDENT_DECODE_STAGE_NO_BLOCK ||
            physical_block >= state->node_context.mla_cache_block_count ||
            physical_block == state->scratch_block_index)
        {
            fprintf(
                stderr,
                "k3_stage block_table_invalid lane=%u logical_block=%u "
                "physical_block=%u cache_block_count=%u\n",
                lane,
                logical_block,
                physical_block,
                state->node_context.mla_cache_block_count);
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    return SPARK_STATUS_OK;
}


// Claim a free pipeline slot by compare-and-swap, scanning once around the
// ring from an atomic hint. Every slot busy means the node is saturated and
// the frame bounces with BUSY rather than queueing.
static SparkStatus SparkK3ModuleClaimSlot(
    SparkK3ModuleState *state,
    uint32_t *slot_index)
{
    return SparkStageModuleSlotClaim(
        state->slot_states,
        state->node_context.pipeline_slot_count,
        slot_index);
}


static SparkStatus SparkK3ModuleStageDispatch(
    SparkK3ModuleState *state,
    const SparkK3PipelineSlot *slot,
    SparkK3ModuleSlotStaging *staging,
    SparkModelDriverFrame *frame,
    SparkK3ResidentDecodeStageFrameContext *context,
    const SparkK3MlaBlockTableView *block_table,
    uint32_t is_prefill,
    uint32_t prefill_lane_index,
    uint32_t row_count,
    uint32_t padded_row_count)
{
    cudaStream_t stream;
    SparkStatus status;
    uint32_t row_index;

    stream = (cudaStream_t)slot->cuda_stream;
    if (state->node_context.owns_embedding != 0u)
    {
        memcpy(
            staging->row_token_ids,
            frame->buffers[0].address,
            (size_t)row_count * sizeof(uint32_t));
    }
    if (is_prefill != 0u)
    {
        status = SparkK3ModuleFillRowMetadata(
            state,
            staging,
            frame,
            block_table,
            prefill_lane_index,
            row_count,
            padded_row_count);
        for (row_index = 0u;
             row_index < padded_row_count;
             row_index++)
        {
            staging->row_cold_flags[row_index] =
                frame->sequence_position == 0u ? 1u : 0u;
        }
    }
    else
    {
        status = SparkK3ModuleFillDecodeBatchMetadata(
            state,
            staging,
            context->decode_batch,
            block_table);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkK3ModuleUploadRowMetadata(
            staging,
            slot,
            padded_row_count,
            row_count,
            state->node_context.owns_embedding,
            stream);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkK3ModuleUploadColdFlags(
            state,
            staging,
            padded_row_count,
            stream);
    }
    if (status == SPARK_STATUS_OK && state->stage_index > 0u)
    {
        status = SparkK3ModuleReceiveBoundary(
            state,
            slot,
            context,
            padded_row_count,
            stream);
    }
    return status;
}



static SparkStatus SparkK3ModuleExecuteOnSlot(
    SparkK3ModuleState *state,
    const SparkK3PipelineSlot *slot,
    SparkK3ModuleSlotStaging *staging,
    SparkModelDriverFrame *frame,
    SparkK3ResidentDecodeStageFrameContext *context,
    const SparkK3MlaBlockTableView *block_table,
    uint32_t is_prefill,
    uint32_t row_count)
{
    cudaStream_t stream;
    SparkStatus status;
    uint32_t carry_state;
    uint32_t output_buffer_index;
    uint32_t padded_row_count;

    stream = (cudaStream_t)slot->cuda_stream;
    carry_state = frame->sequence_position != 0u ? 1u : 0u;
    padded_row_count = is_prefill != 0u
        ? SPARK_K3_MODEL_KDA_CHUNK_TOKENS
        : row_count;
    status = SparkK3ModuleStageDispatch(
        state,
        slot,
        staging,
        frame,
        context,
        block_table,
        is_prefill,
        context->prefill_lane_index,
        row_count,
        padded_row_count);
    if (status == SPARK_STATUS_OK)
    {
        status = SparkK3ModuleRunStage(
            state,
            slot,
            staging,
            block_table,
            padded_row_count,
            is_prefill,
            carry_state,
            stream);
    }
    if (status == SPARK_STATUS_OK && state->node_context.owns_final_head != 0u)
    {
        status = SparkStageModuleCudaStatus(
            SPARK_K3_MODULE_TAG,
            cudaMemcpyAsync(
                staging->row_output_token_ids,
                slot->output_token_ids,
                (size_t)row_count * sizeof(uint32_t),
                cudaMemcpyDeviceToHost,
                stream),
            "output_ids_d2h");
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkStageModuleCudaStatus(
            SPARK_K3_MODULE_TAG,
            cudaStreamSynchronize(stream),
            "cudaStreamSynchronize");
    }
    if (status == SPARK_STATUS_OK && state->node_context.owns_final_head == 0u)
    {
        status = SparkK3ModuleSendBoundary(
            state,
            slot,
            context,
            padded_row_count,
            stream);
    }
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    if (state->node_context.owns_final_head != 0u)
    {
        output_buffer_index = state->node_context.owns_embedding != 0u
            ? 1u
            : 0u;
        memcpy(
            frame->buffers[output_buffer_index].address,
            staging->row_output_token_ids,
            (size_t)row_count * sizeof(uint32_t));
    }
    return SPARK_STATUS_OK;
}


static SparkStatus SparkK3ModuleValidateLaneSequenceContinuity(
    SparkK3ModuleState *state,
    const SparkModelDriverFrame *frame,
    const SparkK3ResidentDecodeStageFrameContext *context,
    uint32_t is_prefill)
{
    uint32_t row;

    if (state == 0 || frame == 0 || context == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (is_prefill != 0u)
    {
        uint32_t lane;
        uint64_t current_sequence_id;

        lane = context->prefill_lane_index;
        current_sequence_id = state->lane_sequence_ids[lane];
        if (current_sequence_id == frame->sequence_id)
        {
            if (frame->sequence_position != state->lane_next_positions[lane])
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
        }
        else if (frame->sequence_position != 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
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
        else if (position != 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    return SPARK_STATUS_OK;
}

static void SparkK3ModuleCommitLaneSequenceContinuity(
    SparkK3ModuleState *state,
    const SparkModelDriverFrame *frame,
    const SparkK3ResidentDecodeStageFrameContext *context,
    uint32_t is_prefill)
{
    uint32_t row;

    if (is_prefill != 0u)
    {
        uint32_t lane;

        lane = context->prefill_lane_index;
        state->lane_sequence_ids[lane] = frame->sequence_id;
        state->lane_next_positions[lane] =
            frame->sequence_position + frame->new_token_count;
        return;
    }
    for (row = 0u; row < context->decode_batch->row_count; row++)
    {
        uint32_t lane;

        lane = context->decode_batch->row_lane_indices[row];
        state->lane_sequence_ids[lane] = context->decode_batch->row_sequence_ids[row];
        state->lane_next_positions[lane] =
            context->decode_batch->row_positions[row] + 1u;
    }
}

static void SparkK3ModuleInvalidateLaneSequenceContinuity(
    SparkK3ModuleState *state,
    const SparkK3ResidentDecodeStageFrameContext *context,
    uint32_t is_prefill)
{
    uint32_t row;

    if (is_prefill != 0u)
    {
        uint32_t lane;

        lane = context->prefill_lane_index;
        state->lane_sequence_ids[lane] = 0u;
        state->lane_next_positions[lane] = 0u;
        return;
    }
    for (row = 0u; row < context->decode_batch->row_count; row++)
    {
        uint32_t lane;

        lane = context->decode_batch->row_lane_indices[row];
        state->lane_sequence_ids[lane] = 0u;
        state->lane_next_positions[lane] = 0u;
    }
}

SparkStatus SparkK3ResidentDecodeStageExecute(
    void *module_state,
    SparkModelDriverFrame *frame)
{
    SparkK3ModuleState *state;
    SparkK3ResidentDecodeStageFrameContext *context;
    const SparkK3MlaBlockTableView *block_table;
    const uint32_t *claimed_lane_indices;
    SparkStatus status;
    uint32_t is_prefill;
    uint32_t rows;
    uint32_t slot_index;
    uint32_t prefill_lane_index;
    uint32_t claimed_lane_count;
    uint32_t slot_claimed;
    uint32_t lanes_claimed;

    state = (SparkK3ModuleState *)module_state;
    if (state == 0 || frame == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    context = 0;
    block_table = 0;
    claimed_lane_indices = 0;
    is_prefill = 0u;
    rows = 0u;
    slot_index = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
    prefill_lane_index = 0u;
    claimed_lane_count = 0u;
    slot_claimed = 0u;
    lanes_claimed = 0u;

    status = SparkK3ModuleValidateFrameShape(
        state,
        frame,
        &is_prefill,
        &rows);
    if (status == SPARK_STATUS_OK)
    {
        status = SparkK3ModuleResolveFrameContext(
            state,
            frame,
            is_prefill,
            &context,
            &block_table);
    }
    if (status == SPARK_STATUS_OK)
    {
        prefill_lane_index = context->prefill_lane_index;
        if (is_prefill != 0u)
        {
            claimed_lane_indices = &prefill_lane_index;
            claimed_lane_count = 1u;
            status = SparkK3ModuleValidateBlockCoverage(
                state,
                block_table,
                frame->sequence_position,
                prefill_lane_index,
                rows);
        }
        else
        {
            uint32_t row_index;

            claimed_lane_indices = context->decode_batch->row_lane_indices;
            claimed_lane_count = context->decode_batch->row_count;
            for (row_index = 0u;
                 row_index < context->decode_batch->row_count &&
                     status == SPARK_STATUS_OK;
                 ++row_index)
            {
                status = SparkK3ModuleValidateBlockCoverage(
                    state,
                    block_table,
                    context->decode_batch->row_positions[row_index],
                    context->decode_batch->row_lane_indices[row_index],
                    1u);
            }
        }
    }
    if (status == SPARK_STATUS_OK && state->node_context.owns_embedding != 0u)
    {
        status = SparkK3ModuleValidateWireTokens(
            (const uint32_t *)frame->buffers[0].address,
            rows);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkStageModuleIndexSetClaim(
            state->lane_states,
            state->lane_capacity,
            claimed_lane_indices,
            claimed_lane_count);
        lanes_claimed = status == SPARK_STATUS_OK ? 1u : 0u;
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkK3ModuleValidateLaneSequenceContinuity(
            state,
            frame,
            context,
            is_prefill);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkK3ModuleClaimSlot(state, &slot_index);
        slot_claimed = status == SPARK_STATUS_OK ? 1u : 0u;
    }
    if (status != SPARK_STATUS_OK)
    {
        if (lanes_claimed != 0u)
        {
            SparkStageModuleIndexSetRelease(
                state->lane_states,
                state->lane_capacity,
                claimed_lane_indices,
                claimed_lane_count);
        }
        atomic_fetch_add_explicit(
            &state->rejected_count,
            1u,
            memory_order_relaxed);
        if (status != SPARK_STATUS_BUSY)
        {
            fprintf(
                stderr,
                "k3_stage execute_reject status=%d flags=0x%08x tokens=%u "
                "position=%llu\n",
                (int32_t)status,
                frame->flags,
                frame->new_token_count,
                (unsigned long long)frame->sequence_position);
        }
        return status;
    }

    atomic_fetch_add_explicit(
        &state->submitted_count,
        1u,
        memory_order_relaxed);
    status = SparkK3ModuleExecuteOnSlot(
        state,
        &state->pipeline_slots[slot_index],
        &state->slot_staging[slot_index],
        frame,
        context,
        block_table,
        is_prefill,
        rows);
    if (status == SPARK_STATUS_OK)
    {
        SparkK3ModuleCommitLaneSequenceContinuity(
            state,
            frame,
            context,
            is_prefill);
        atomic_fetch_add_explicit(
            &state->completed_count,
            1u,
            memory_order_relaxed);
    }
    else
    {
        SparkK3ModuleInvalidateLaneSequenceContinuity(
            state,
            context,
            is_prefill);
        atomic_fetch_add_explicit(
            &state->failed_count,
            1u,
            memory_order_relaxed);
    }

    SparkStageModuleIndexSetRelease(
        state->lane_states,
        state->lane_capacity,
        claimed_lane_indices,
        claimed_lane_count);
    if (slot_claimed != 0u)
    {
        SparkStageModuleSlotRelease(state->slot_states, slot_index);
    }
    return status;
}

static uint32_t SparkK3ModuleAvailablePipelineSlotCount(
    const SparkK3ModuleState *state)
{
    return SparkStageModuleSlotAvailableCount(
        state->slot_states,
        state->node_context.pipeline_slot_count);
}

SparkStatus SparkK3ResidentDecodeStageAdmit(
    void *module_state,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision)
{
    SparkK3ModuleState *state;
    uint32_t available_slot_count;
    uint32_t is_prefill;

    state = (SparkK3ModuleState *)module_state;
    if (state == 0 || request == 0 || decision == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    available_slot_count = SparkK3ModuleAvailablePipelineSlotCount(state);
    SparkModelDriverInitializeAdmissionDecision(decision);
    decision->available_dispatch_slot_count = available_slot_count;
    decision->estimated_service_time_ns =
        state->node_context.estimated_service_time_ns;

    if (request->descriptor_bytes < (uint32_t)sizeof(*request) ||
        request->program_id == 0u)
    {
        return SPARK_STATUS_ABI_MISMATCH;
    }
    is_prefill = (request->frame_flags &
                  SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u
        ? 1u
        : 0u;
    if ((request->frame_flags &
         ~SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u ||
        request->active_slot_count == 0u ||
        request->active_slot_count > state->lane_capacity ||
        request->new_token_count == 0u)
    {
        atomic_fetch_add_explicit(
            &state->rejected_count,
            1u,
            memory_order_relaxed);
        return SparkModelDriverRejectAdmission(
            decision,
            SPARK_MODEL_DRIVER_ADMISSION_REJECTED_UNSUPPORTED_SHAPE,
            available_slot_count);
    }
    if (is_prefill != 0u &&
        (request->sequence_id == 0u ||
         SparkModelDriverRangeFitsWithinCapacity(
             request->sequence_position,
             request->new_token_count,
             state->max_context_tokens) == 0u))
    {
        atomic_fetch_add_explicit(
            &state->rejected_count,
            1u,
            memory_order_relaxed);
        return SparkModelDriverRejectAdmission(
            decision,
            SPARK_MODEL_DRIVER_ADMISSION_REJECTED_KV_CAPACITY,
            available_slot_count);
    }
    if ((is_prefill != 0u &&
         (request->active_slot_count != 1u ||
          request->new_token_count > SPARK_K3_MODEL_KDA_CHUNK_TOKENS)) ||
        (is_prefill == 0u &&
         request->new_token_count != request->active_slot_count))
    {
        atomic_fetch_add_explicit(
            &state->rejected_count,
            1u,
            memory_order_relaxed);
        return SparkModelDriverRejectAdmission(
            decision,
            SPARK_MODEL_DRIVER_ADMISSION_REJECTED_UNSUPPORTED_SHAPE,
            available_slot_count);
    }
    if (available_slot_count == 0u)
    {
        atomic_fetch_add_explicit(
            &state->rejected_count,
            1u,
            memory_order_relaxed);
        return SparkModelDriverRejectAdmission(
            decision,
            SPARK_MODEL_DRIVER_ADMISSION_REJECTED_BUSY,
            available_slot_count);
    }

    decision->host_staging_bytes =
        (uint64_t)request->new_token_count * sizeof(uint32_t);
    decision->device_memcpy_bytes =
        (uint64_t)request->new_token_count * sizeof(uint32_t) * 2u;
    decision->accepted = 1u;
    decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
    decision->driver_dispatch_slot = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
    return SPARK_STATUS_OK;
}

SparkStatus SparkK3ResidentDecodeStageSnapshot(
    void *module_state,
    uint32_t program_id,
    SparkModelDriverRuntimeSnapshot *snapshot)
{
    SparkK3ModuleState *state;
    uint32_t available_slot_count;

    state = (SparkK3ModuleState *)module_state;
    if (state == 0 || snapshot == 0 || program_id == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    available_slot_count = SparkK3ModuleAvailablePipelineSlotCount(state);
    SparkModelDriverInitializeRuntimeSnapshot(snapshot, program_id);
    snapshot->active_submission_count =
        state->node_context.pipeline_slot_count - available_slot_count;
    snapshot->available_dispatch_slot_count = available_slot_count;
    snapshot->submitted_count = atomic_load_explicit(
        &state->submitted_count,
        memory_order_relaxed);
    snapshot->completed_count = atomic_load_explicit(
        &state->completed_count,
        memory_order_relaxed);
    snapshot->rejected_count = atomic_load_explicit(
        &state->rejected_count,
        memory_order_relaxed);
    snapshot->kv_token_capacity =
        (uint64_t)state->lane_capacity *
        state->blocks_per_lane *
        SPARK_K3_RESIDENT_DECODE_STAGE_MLA_BLOCK_TOKENS;
    return SPARK_STATUS_OK;
}

void SparkK3ResidentDecodeStageDestroy(void *module_state)
{
    SparkK3ModuleState *state;
    uint32_t index;

    state = (SparkK3ModuleState *)module_state;
    if (state == 0)
    {
        return;
    }
    if (SparkStageModuleWaitForSlots(
            "k3_stage",
            state->slot_states,
            state->node_context.pipeline_slot_count,
            SPARK_STAGE_MODULE_DESTROY_QUIESCE_TIMEOUT_NS) != SPARK_STATUS_OK)
    {
        fprintf(
            stderr,
            "k3_stage destroy_refused active submissions did not quiesce; "
            "resident resources intentionally retained to prevent "
            "use-after-free\n");
        return;
    }

    for (index = 0u;
         index < state->node_context.pipeline_slot_count;
         ++index)
    {
        if (state->pipeline_slots[index].cuda_stream != 0)
        {
            (void)cudaStreamDestroy(
                (cudaStream_t)state->pipeline_slots[index].cuda_stream);
        }
    }
    SparkStageModuleLedgerRelease(&state->ledger);
    for (index = 0u;
         index < state->node_context.pipeline_slot_count;
         ++index)
    {
        free(state->slot_staging[index].row_token_ids);
        free(state->slot_staging[index].row_slot_mapping);
        free(state->slot_staging[index].row_lane_indices);
        free(state->slot_staging[index].row_context_lengths);
        free(state->slot_staging[index].row_cold_flags);
        free(state->slot_staging[index].row_output_token_ids);
    }
    free(state->host_block_indices);
    free(state->host_lane_block_counts);
    free(state);
}
