#define _FILE_OFFSET_BITS 64

#include <errno.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_hidden_transport.h"
#include "sparkpipe/spark_muse_glimmer_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_stage_kv_client.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "sparkpipe/spark_stage_module_lifecycle.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_muse_glimmer_work_control.h"
#include "spark_muse_glimmer_stagepack_format.h"

#define SPARK_MUSE_GLIMMER_MODULE_TAG "muse_stage"

#define SPARK_MUSE_GLIMMER_MODULE_TP_DEGREE_REPLICATED 1u
#define SPARK_MUSE_GLIMMER_MODULE_TP_RANK_REPLICATED 0u
#define SPARK_MUSE_GLIMMER_MODULE_TP_TIMEOUT_MILLI_DEFAULT 120000u

#define SPARK_MUSE_GLIMMER_MODULE_KV_STAGING_RECORDS 16u
#define SPARK_MUSE_GLIMMER_MODULE_KV_POLL_BOUND 10000u
#define SPARK_MUSE_GLIMMER_MODULE_KV_GDN_RECORD_PLACEHOLDER_BYTES 4096u
#define SPARK_MUSE_GLIMMER_MODULE_KV_MAX_BLOCKS_PER_LANE 4096u
#define SPARK_MUSE_GLIMMER_MODULE_HEAD_SCREEN_CAP 4096u
#define SPARK_MUSE_GLIMMER_MODULE_HEAD_SHADOW_GROUP 32u

typedef struct SparkMuseGlimmerFrameErrorShim
{
	uint32_t error_code;
	uint32_t access_kind;
	uint32_t row;
	uint32_t sequence;
	uint32_t position;
	uint32_t page;
} SparkMuseGlimmerFrameErrorShim;

typedef struct SparkMuseGlimmerKvViewShim
{
	void *pool;
	const uint32_t *page_table;
	uint32_t page_table_stride;
	uint32_t sequence_count;
	uint32_t pool_page_count;
	void *access_error;
} SparkMuseGlimmerKvViewShim;

typedef struct SparkMuseGlimmerModuleSlot
{
	void *cuda_stream;
	uint32_t *host_row_lane_indices;
	uint64_t *host_row_positions;
	uint32_t *host_row_positions_u32;
	uint32_t *host_row_cold;
	uint32_t *host_slot_mapping;
	uint32_t *host_context_lengths;
	uint32_t *input_token_ids;
	uint32_t *output_token_ids;
	uint32_t *local_token_ids;
	uint64_t *head_maxloc_u64;
	float *head_scores_f32;
	uint32_t *row_lane_indices;
	uint32_t *row_positions_u32;
	uint32_t *context_lengths;
	uint32_t *window_positions;
	uint32_t *row_cold;
	uint64_t *row_positions;
	void *hidden_bf16;
	void *residual_bf16;
	void *normalized_bf16;
	void *fused_qgkv_bf16;
	void *query_gate_bf16;
	void *query_bf16;
	void *attn_gate_bf16;
	void *key_bf16;
	void *value_bf16;
	void *head_out_bf16;
	void *delta_bf16;
	void *gate_up_bf16;
	void *intermediate_bf16;
} SparkMuseGlimmerModuleSlot;

typedef struct SparkMuseGlimmerModuleState
{
	SparkStageModuleLedger ledger;
	uint32_t multiprocessor_count;
	uint32_t tp_degree;
	uint32_t tp_rank;
	SparkTpDeviceCollective tp_device_collective;
	SparkTpDeviceCollectiveCreditBinding tp_credit_bindings[8u];
	uint32_t tp_credit_binding_count;
	uint32_t tp_collective_initialized;
	void *tp_collective_credit_send_bf16;
	void *tp_collective_credit_receive_bf16;
	void *tp_host_credit_send_bf16;
	void *tp_host_credit_receive_bf16;
	atomic_uint tp_completion_flag;
	atomic_ullong tp_next_ordinal;
	char tp_backend_path[SPARK_TP_DEVICE_COLLECTIVE_ROUTE_NAME_BYTES];
	char tp_hosts[SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE][SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES];
	char tp_local_host[SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES];
	uint16_t tp_session_ports[SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE][SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE];
	uint64_t tp_collective_identifier;
	uint32_t tp_control_port_base;
	uint32_t tp_connect_timeout_milli;
	uint32_t tp_operation_timeout_milli;
	uint32_t max_active_sequence_count;
	uint32_t pipeline_slot_count;
	atomic_uint slot_states[SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	uint32_t kv_block_count;
	uint32_t cache_layer_count;
	atomic_ullong submitted_count;
	atomic_ullong completed_count;
	atomic_ullong rejected_count;
	atomic_ullong failed_count;
	atomic_ullong tokens_emitted;
	void *kv_cache_bf16;
	uint64_t cache_layer_stride;
	uint64_t cache_block_stride;
	void *kv_access_error;
	SparkMuseGlimmerModuleSlot slots[SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	uint32_t stage_count;
	uint32_t stage_index;
	uint32_t allow_unqualified_execution;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t owns_embedding;
	uint32_t owns_final_head;
	uint32_t layer_seen_bits[SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	uint32_t global_seen_bits;
	SparkMuseGlimmerLayerWeights layers[SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	const void *token_embedding_bf16;
	const void *final_norm_weight_bf16;
	const void *lm_head_weight_bf16;
	SparkStageKvClient kv_client;
	SparkMuseGlimmerWorkControlKvState kv_work;
	SparkMuseGlimmerWorkControlKvPlanConfig kv_plan;
	uint32_t kv_tier_active;
	uint32_t kv_logical_page_capacity;
	uint32_t kv_physical_page_capacity;
	uint64_t kv_backing_maximum_bytes;
	uint32_t *kv_logical_to_slot;
	uint64_t kv_logical_to_slot_capacity;
	uint32_t kv_logical_stride;
	uint32_t *kv_table_indices_device;
	uint32_t *kv_table_counts_device;
	uint32_t *kv_table_indices_host;
	uint32_t *kv_slot_lane;
	uint32_t *kv_slot_logical;
	uint64_t *kv_slot_sequence;
	uint8_t *kv_slot_dirty;
	uint8_t *kv_slot_pinned;
	uint32_t *kv_slot_free_stack;
	uint32_t kv_slot_free_count;
	uint32_t kv_evict_cursor;
	void *kv_block_staging;
	void *kv_gdn_staging;
} SparkMuseGlimmerModuleState;

static SparkStatus SparkMuseGlimmerModuleConfigure(SparkMuseGlimmerModuleState *state)
{
	SparkStatus status;
	{
		status = SparkStageModuleEnvironmentUnsignedOrDefault(SPARK_MUSE_GLIMMER_MODULE_TAG,"SPARK_MUSE_GLIMMER_STAGE_TP_DEGREE",1u,SPARK_MUSE_GLIMMER_MODEL_ATTN_QUERY_HEAD_COUNT,SPARK_MUSE_GLIMMER_MODULE_TP_DEGREE_REPLICATED,&state->tp_degree);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleEnvironmentUnsignedOrDefault(SPARK_MUSE_GLIMMER_MODULE_TAG,"SPARK_MUSE_GLIMMER_STAGE_TP_RANK",0u,SPARK_MUSE_GLIMMER_MODEL_ATTN_QUERY_HEAD_COUNT - 1u,SPARK_MUSE_GLIMMER_MODULE_TP_RANK_REPLICATED,&state->tp_rank);
		if ( status != SPARK_STATUS_OK )
			return(status);
		if ( state->tp_rank >= state->tp_degree || state->tp_degree > SPARK_MUSE_GLIMMER_MODEL_ATTN_QUERY_HEAD_COUNT || (SPARK_MUSE_GLIMMER_MODEL_ATTN_QUERY_HEAD_COUNT % state->tp_degree) != 0u || (SPARK_MUSE_GLIMMER_MODEL_INTERMEDIATE_DIMENSION % state->tp_degree) != 0u || (SPARK_MUSE_GLIMMER_MODEL_OUTPUT_VOCAB_COUNT % state->tp_degree) != 0u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		state->tp_collective_identifier = 0u;
		state->tp_control_port_base = 0u;
		state->tp_connect_timeout_milli = SPARK_MUSE_GLIMMER_MODULE_TP_TIMEOUT_MILLI_DEFAULT;
		state->tp_operation_timeout_milli = SPARK_MUSE_GLIMMER_MODULE_TP_TIMEOUT_MILLI_DEFAULT;
		state->tp_backend_path[0] = '\0';
		state->tp_local_host[0] = '\0';
		for (uint32_t host_clear = 0u; host_clear < SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE; host_clear++)
			state->tp_hosts[host_clear][0] = '\0';
		if ( state->tp_degree > 1u )
		{
			const char *tp_backend;
			const char *tp_hosts;
			const char *tp_local_host;
			uint64_t tp_identifier;
			const char *scan;
			uint32_t host_index;
			status = SparkStageModuleEnvironmentText(SPARK_MUSE_GLIMMER_MODULE_TAG,"SPARK_MUSE_GLIMMER_STAGE_TP_BACKEND_PATH",&tp_backend);
			if ( status == SPARK_STATUS_OK )
				status = SparkStageModuleEnvironmentUnsigned64(SPARK_MUSE_GLIMMER_MODULE_TAG,"SPARK_MUSE_GLIMMER_STAGE_TP_IDENTIFIER",0u,UINT64_MAX,&tp_identifier);
			if ( status == SPARK_STATUS_OK )
				status = SparkStageModuleEnvironmentUnsigned(SPARK_MUSE_GLIMMER_MODULE_TAG,"SPARK_MUSE_GLIMMER_STAGE_TP_PORT_BASE",1u,65535u,&state->tp_control_port_base);
			if ( status == SPARK_STATUS_OK )
				status = SparkStageModuleEnvironmentText(SPARK_MUSE_GLIMMER_MODULE_TAG,"SPARK_MUSE_GLIMMER_STAGE_TP_HOSTS",&tp_hosts);
			if ( status == SPARK_STATUS_OK )
				status = SparkStageModuleEnvironmentText(SPARK_MUSE_GLIMMER_MODULE_TAG,"SPARK_MUSE_GLIMMER_STAGE_TP_LOCAL_HOST",&tp_local_host);
			if ( status == SPARK_STATUS_OK )
				status = SparkStageModuleEnvironmentUnsignedOrDefault(SPARK_MUSE_GLIMMER_MODULE_TAG,"SPARK_MUSE_GLIMMER_STAGE_TP_TIMEOUT_MS",1u,UINT32_MAX,SPARK_MUSE_GLIMMER_MODULE_TP_TIMEOUT_MILLI_DEFAULT,&state->tp_connect_timeout_milli);
			if ( status != SPARK_STATUS_OK )
				return(status);
			snprintf(state->tp_backend_path,sizeof(state->tp_backend_path),"%s",tp_backend);
			snprintf(state->tp_local_host,sizeof(state->tp_local_host),"%s",tp_local_host);
			state->tp_collective_identifier = tp_identifier;
			state->tp_operation_timeout_milli = state->tp_connect_timeout_milli;
			scan = tp_hosts;
			host_index = 0u;
			while ( *scan != '\0' && host_index < state->tp_degree )
			{
				const char *comma = strchr(scan,',');
				size_t length = comma != 0 ? (size_t)(comma - scan) : strlen(scan);
				if ( length >= SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES )
					return(SPARK_STATUS_INVALID_ARGUMENT);
				memcpy(state->tp_hosts[host_index],scan,length);
				state->tp_hosts[host_index][length] = '\0';
				host_index++;
				scan = comma != 0 ? comma + 1 : scan + length;
			}
			if ( host_index != state->tp_degree )
				return(SPARK_STATUS_INVALID_ARGUMENT);
			{
				const char *tp_session_ports;
				const char *cell_scan;
				uint32_t row_index,column_index;
				unsigned long cell_value;
				status = SparkStageModuleEnvironmentText(SPARK_MUSE_GLIMMER_MODULE_TAG,"SPARK_MUSE_GLIMMER_STAGE_TP_SESSION_PORTS",&tp_session_ports);
				if ( status != SPARK_STATUS_OK )
					return(status);
				cell_scan = tp_session_ports;
				for (row_index = 0u; row_index < state->tp_degree; row_index++)
					for (column_index = 0u; column_index < state->tp_degree; column_index++)
					{
						char *cell_end;
						errno = 0;
						cell_value = strtoul(cell_scan,&cell_end,10);
						if ( cell_end == cell_scan || errno != 0 ||
							cell_value > 65535u ||
							(row_index == column_index ? cell_value != 0u : cell_value == 0u) )
							return(SPARK_STATUS_INVALID_ARGUMENT);
						state->tp_session_ports[row_index][column_index] = (uint16_t)cell_value;
						cell_scan = cell_end;
						while ( *cell_scan == ',' )
							cell_scan++;
					}
			}
		}
	}
	status = SparkStageModuleEnvironmentUnsigned(SPARK_MUSE_GLIMMER_MODULE_TAG,"SPARK_MUSE_GLIMMER_STAGE_COUNT",1u,SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT,&state->stage_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_MUSE_GLIMMER_MODULE_TAG,"SPARK_MUSE_GLIMMER_STAGE_INDEX",0u,SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT - 1u,&state->stage_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_MUSE_GLIMMER_MODULE_TAG,"SPARK_MUSE_GLIMMER_STAGE_FIRST_LAYER",0u,SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_LAYER_COUNT - 1u,&state->first_layer_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_MUSE_GLIMMER_MODULE_TAG,"SPARK_MUSE_GLIMMER_STAGE_LAYER_COUNT",1u,SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_LAYER_COUNT,&state->layer_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_MUSE_GLIMMER_MODULE_TAG,"SPARK_MUSE_GLIMMER_STAGE_MAX_ACTIVE_SEQUENCES",1u,SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,&state->max_active_sequence_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_MUSE_GLIMMER_MODULE_TAG,"SPARK_MUSE_GLIMMER_STAGE_PIPELINE_SLOTS",1u,SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,&state->pipeline_slot_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_MUSE_GLIMMER_MODULE_TAG,"SPARK_MUSE_GLIMMER_STAGE_KV_BLOCKS",1u,1u << 20u,&state->kv_block_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( state->stage_index >= state->stage_count || state->first_layer_index + state->layer_count > SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_LAYER_COUNT )
	{
		fprintf(stderr,"%s config_slice_invalid stage=%u/%u slice=%u+%u\n",SPARK_MUSE_GLIMMER_MODULE_TAG,state->stage_index,state->stage_count,state->first_layer_index,state->layer_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	state->owns_embedding = state->first_layer_index == 0u ? 1u : 0u;
	state->owns_final_head = state->first_layer_index + state->layer_count == SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_LAYER_COUNT ? 1u : 0u;
	if ( (state->stage_index == 0u) != (state->owns_embedding != 0u) || (state->stage_index + 1u == state->stage_count) != (state->owns_final_head != 0u) )
	{
		fprintf(stderr,"%s config_position_mismatch stage=%u/%u slice=%u+%u\n",SPARK_MUSE_GLIMMER_MODULE_TAG,state->stage_index,state->stage_count,state->first_layer_index,state->layer_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkMuseGlimmerModuleBindGlobal(SparkMuseGlimmerModuleState *state, const SparkMuseGlimmerStagePackEntry *entry, void *payload)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_EMBEDDING:
		if ( state->owns_embedding == 0u && state->owns_final_head == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
		state->token_embedding_bf16 = payload;
		return(SPARK_STATUS_OK);
	case SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_FINAL_NORM:
		if ( state->owns_final_head == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
		state->final_norm_weight_bf16 = payload;
		return(SPARK_STATUS_OK);
	case SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_LM_HEAD:
		if ( state->owns_final_head == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
		state->lm_head_weight_bf16 = payload;
		return(SPARK_STATUS_OK);
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
}

static void SparkMuseGlimmerModuleFillLinearView(SparkMuseGlimmerLinearView *view, const SparkMuseGlimmerStagePackEntry *entry, void *payload, void *scale)
{
	view->abi_version = SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION;
	view->weight_format = entry->weight_format;
	view->input_dimension = entry->columns;
	view->output_dimension = entry->rows;
	view->weight_payload = payload;
	view->weight_scale_e8m0 = (const uint8_t *)scale;
	view->weight_payload_bytes = entry->payload_bytes;
	view->weight_scale_bytes = entry->scale_bytes;
}

static SparkStatus SparkMuseGlimmerModuleBindLayer(SparkMuseGlimmerModuleState *state, const SparkMuseGlimmerStagePackEntry *entry, void *payload, void *scale)
{
	uint32_t layer = entry->layer_index;
	SparkMuseGlimmerLayerWeights *weights = &state->layers[layer];
	switch ( entry->tensor_kind )
	{
	case SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_INPUT_NORM: weights->input_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_POST_ATTENTION_NORM: weights->post_attention_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_PRE_FFN_NORM: weights->pre_feedforward_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_POST_FFN_NORM: weights->post_feedforward_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_QGKV: SparkMuseGlimmerModuleFillLinearView(&weights->qgkv,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_ATTN_OUTPUT: SparkMuseGlimmerModuleFillLinearView(&weights->output,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_ATTN_GATE_UP: SparkMuseGlimmerModuleFillLinearView(&weights->gate_up,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_MLP_DOWN: SparkMuseGlimmerModuleFillLinearView(&weights->down,entry,payload,scale); return(SPARK_STATUS_OK);
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
}

static uint32_t SparkMuseGlimmerModuleExpectedGlobalBits(const SparkMuseGlimmerModuleState *state)
{
	uint32_t bits = 0u;
	if ( state->owns_embedding != 0u || state->owns_final_head != 0u )
		bits |= 1u << SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_EMBEDDING;
	if ( state->owns_final_head != 0u )
		bits |= (1u << SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_FINAL_NORM) | (1u << SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_LM_HEAD);
	return(bits);
}

static uint32_t SparkMuseGlimmerModuleExpectedLayerBits(const SparkMuseGlimmerModuleState *state, uint32_t layer)
{
	(void)state;
	(void)layer;
	return((1u << SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_INPUT_NORM) | (1u << SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_POST_ATTENTION_NORM) | (1u << SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_PRE_FFN_NORM) | (1u << SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_POST_FFN_NORM) | (1u << SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_QGKV) | (1u << SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_ATTN_OUTPUT) | (1u << SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_ATTN_GATE_UP) | (1u << SPARK_MUSE_GLIMMER_STAGEPACK_TENSOR_MLP_DOWN));
}

static SparkStatus SparkMuseGlimmerModuleValidateEntryPlacement(SparkMuseGlimmerModuleState *state, const SparkMuseGlimmerStagePackEntry *entry, uint64_t file_bytes, uint32_t *is_global)
{
	uint32_t global = entry->layer_index == SPARK_MUSE_GLIMMER_STAGEPACK_GLOBAL_LAYER ? 1u : 0u;
	if ( entry->payload_bytes != SparkMuseGlimmerStagePackPayloadBytes(entry->weight_format,entry->rows,entry->columns) || entry->scale_bytes != 0u )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->payload_offset > file_bytes || entry->payload_bytes > file_bytes - entry->payload_offset )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( global == 0u && (entry->layer_index < state->first_layer_index || entry->layer_index >= state->first_layer_index + state->layer_count) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	*is_global = global;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkMuseGlimmerModuleValidateEntry(SparkMuseGlimmerModuleState *state, const SparkMuseGlimmerStagePackEntry *entry, uint64_t file_bytes, uint32_t *is_global)
{
	SparkMuseGlimmerStagePackTensorShape shape;
	uint32_t global = entry->layer_index == SPARK_MUSE_GLIMMER_STAGEPACK_GLOBAL_LAYER ? 1u : 0u;
	if ( SparkMuseGlimmerStagePackResolvedShape(entry->tensor_kind,global != 0u ? 0u : entry->layer_index,global,state->tp_degree,&shape) != 0 || entry->rows != shape.rows || entry->columns != shape.columns )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->weight_format != shape.natural_format || entry->weight_format != SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 || entry->scale_group_size != 0u )
		return(SPARK_STATUS_VALIDATION_FAILED);
	return(SparkMuseGlimmerModuleValidateEntryPlacement(state,entry,file_bytes,is_global));
}

static SparkStatus SparkMuseGlimmerModuleLoadEntry(SparkMuseGlimmerModuleState *state, FILE *file, const SparkMuseGlimmerStagePackEntry *entry, uint64_t file_bytes)
{
	uint32_t is_global = 0u,bit;
	uint32_t *seen;
	void *payload = 0;
	SparkStatus status;
	status = SparkMuseGlimmerModuleValidateEntry(state,entry,file_bytes,&is_global);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s pack_entry_invalid kind=%u layer=%u\n",SPARK_MUSE_GLIMMER_MODULE_TAG,entry->tensor_kind,entry->layer_index);
		return(status);
	}
	bit = 1u << entry->tensor_kind;
	seen = is_global != 0u ? &state->global_seen_bits : &state->layer_seen_bits[entry->layer_index];
	if ( (*seen & bit) != 0u )
	{
		fprintf(stderr,"%s pack_entry_duplicate kind=%u layer=%u\n",SPARK_MUSE_GLIMMER_MODULE_TAG,entry->tensor_kind,entry->layer_index);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	*seen |= bit;
	status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,entry->payload_offset,entry->payload_bytes,&payload);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(is_global != 0u ? SparkMuseGlimmerModuleBindGlobal(state,entry,payload) : SparkMuseGlimmerModuleBindLayer(state,entry,payload,0));
}

static SparkStatus SparkMuseGlimmerModuleVerifyCoverage(SparkMuseGlimmerModuleState *state)
{
	uint32_t layer,expected_layer;
	if ( state->global_seen_bits != SparkMuseGlimmerModuleExpectedGlobalBits(state) )
	{
		fprintf(stderr,"%s pack_globals_incomplete seen=%08x expected=%08x\n",SPARK_MUSE_GLIMMER_MODULE_TAG,state->global_seen_bits,SparkMuseGlimmerModuleExpectedGlobalBits(state));
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	for (layer = state->first_layer_index; layer < state->first_layer_index + state->layer_count; layer++)
	{
		expected_layer = SparkMuseGlimmerModuleExpectedLayerBits(state,layer);
		if ( state->layer_seen_bits[layer] != expected_layer )
		{
			fprintf(stderr,"%s pack_layer_incomplete layer=%u seen=%08x expected=%08x\n",SPARK_MUSE_GLIMMER_MODULE_TAG,layer,state->layer_seen_bits[layer],expected_layer);
			return(SPARK_STATUS_VALIDATION_FAILED);
		}
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkMuseGlimmerModuleLoadPack(SparkMuseGlimmerModuleState *state, const char *path)
{
	SparkMuseGlimmerStagePackHeader header,expected;
	SparkMuseGlimmerStagePackEntry *directory;
	FILE *file;
	SparkStatus status;
	uint32_t index;
	file = fopen(path,"rb");
	if ( file == 0 )
	{
		fprintf(stderr,"%s pack_open_failed path=%s\n",SPARK_MUSE_GLIMMER_MODULE_TAG,path);
		return(SPARK_STATUS_IO_ERROR);
	}
	status = SparkStageModulePackRead(SPARK_MUSE_GLIMMER_MODULE_TAG,file,0u,&header,sizeof(header));
	if ( status == SPARK_STATUS_OK )
	{
		SparkMuseGlimmerStagePackExpectedGeometry(&expected,state->first_layer_index,state->layer_count);
		if ( SparkMuseGlimmerStagePackHeaderMatches(&header,&expected) == 0 || header.directory_offset != SPARK_MUSE_GLIMMER_STAGEPACK_HEADER_BYTES )
		{
			fprintf(stderr,"%s pack_geometry_mismatch\n",SPARK_MUSE_GLIMMER_MODULE_TAG);
			status = SPARK_STATUS_VALIDATION_FAILED;
		}
	}
	directory = status == SPARK_STATUS_OK ? (SparkMuseGlimmerStagePackEntry *)malloc((size_t)header.tensor_count * sizeof(SparkMuseGlimmerStagePackEntry)) : 0;
	if ( status == SPARK_STATUS_OK && directory == 0 )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModulePackRead(SPARK_MUSE_GLIMMER_MODULE_TAG,file,header.directory_offset,directory,(uint64_t)header.tensor_count * sizeof(SparkMuseGlimmerStagePackEntry));
	for (index = 0; status == SPARK_STATUS_OK && index < header.tensor_count; index++)
		status = SparkMuseGlimmerModuleLoadEntry(state,file,&directory[index],header.file_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkMuseGlimmerModuleVerifyCoverage(state);
	free(directory);
	fclose(file);
	return(status);
}

static SparkStatus SparkMuseGlimmerModuleAllocatePools(SparkMuseGlimmerModuleState *state);
static SparkStatus SparkMuseGlimmerModuleAllocateSlot(SparkMuseGlimmerModuleState *state, SparkMuseGlimmerModuleSlot *slot);
static SparkStatus SparkMuseGlimmerModuleAllocateSlotHostMirrors(SparkMuseGlimmerModuleState *state, SparkMuseGlimmerModuleSlot *slot);


static uint64_t SparkMuseGlimmerModuleFingerprint(const void *bytes, uint64_t count, uint64_t basis)
{
	const uint8_t *data = (const uint8_t *)bytes;
	uint64_t hash = basis,index;
	for (index = 0; index < count; index++)
		hash = (hash ^ data[index]) * 1099511628211ull;
	return(hash);
}

static SparkStatus SparkMuseGlimmerModuleOpenKvTier(SparkMuseGlimmerModuleState *state, const SparkFirmwareModuleHostServices *host_services)
{
	SparkMuseGlimmerStagePackHeader geometry;
	const char *provider = 0,*service = 0,*socket_path = 0;
	uint64_t pool_bytes = 0u,model_fp,layout_fp,layout_bits[3],block_record_bytes,staging_bytes;
	uint32_t workers = 0u,block_record_elements,index;
	SparkStatus status;
	static const char *none = "none";
	state->kv_tier_active = 0u;
	state->kv_logical_page_capacity = host_services->kv_logical_page_capacity;
	state->kv_physical_page_capacity = host_services->kv_physical_page_capacity;
	state->kv_backing_maximum_bytes = host_services->kv_backing_maximum_bytes;
	if ( host_services->kv_backing_directory == 0 && host_services->kv_backing_maximum_bytes != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	provider = getenv("SPARK_MUSE_GLIMMER_STAGE_KV_STORE");
	if ( provider == 0 )
		provider = none;
	if ( strcmp(provider,"none") == 0 )
		return(SparkStageKvClientOpen(&state->kv_client,SPARK_MUSE_GLIMMER_MODULE_TAG,provider,0u,0u,0u,0u,0u,0,0,0u,0u));
	status = SparkStageModuleEnvironmentText(SPARK_MUSE_GLIMMER_MODULE_TAG,"SPARK_MUSE_GLIMMER_STAGE_KV_SERVICE",&service);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentText(SPARK_MUSE_GLIMMER_MODULE_TAG,"SPARK_MUSE_GLIMMER_STAGE_KV_SOCKET",&socket_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned64(SPARK_MUSE_GLIMMER_MODULE_TAG,"SPARK_MUSE_GLIMMER_STAGE_KV_POOL_BYTES",1u,1ull << 40u,&pool_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_MUSE_GLIMMER_MODULE_TAG,"SPARK_MUSE_GLIMMER_STAGE_KV_WORKERS",1u,64u,&workers);
	if ( status != SPARK_STATUS_OK )
		return(status);
	SparkMuseGlimmerStagePackExpectedGeometry(&geometry,state->first_layer_index,state->layer_count);
	model_fp = SparkMuseGlimmerModuleFingerprint(&geometry,sizeof(geometry),14695981039346656037ull);
	
	block_record_elements = (uint64_t)SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS * 2ull * SPARK_MUSE_GLIMMER_MODEL_ATTN_LOCAL_KV_HEAD_COUNT(state->tp_degree) * SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION * state->layer_count;	layout_bits[0] = block_record_elements;
	layout_bits[1] = SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
	layout_bits[2] = state->kv_block_count;
	layout_fp = SparkMuseGlimmerModuleFingerprint(layout_bits,sizeof(layout_bits),model_fp);
	block_record_bytes = (uint64_t)block_record_elements * SPARK_MUSE_GLIMMER_MODEL_BF16_ELEMENT_BYTES;
	staging_bytes = block_record_bytes * SPARK_MUSE_GLIMMER_MODULE_KV_STAGING_RECORDS;
	if ( state->kv_physical_page_capacity != 0u && state->kv_block_count > state->kv_physical_page_capacity )
		state->kv_block_count = state->kv_physical_page_capacity;
	if ( state->kv_block_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	state->kv_plan.model_fingerprint = model_fp;
	state->kv_plan.cache_layout_fingerprint = layout_fp;
	state->kv_plan.rank_index = state->stage_index;
	state->kv_plan.block_record_bytes = (uint32_t)block_record_bytes;
	state->kv_plan.gdn_record_bytes = SPARK_MUSE_GLIMMER_MODULE_KV_GDN_RECORD_PLACEHOLDER_BYTES;
	state->kv_plan.lookahead_packet_count = 3u;
	state->kv_plan.physical_block_capacity = state->kv_block_count;
	state->kv_plan.allocated_physical_block_count = 0u;
	state->kv_plan.staging_block_capacity = SPARK_MUSE_GLIMMER_MODULE_KV_STAGING_RECORDS;
	status = SparkStageKvClientOpen(&state->kv_client,SPARK_MUSE_GLIMMER_MODULE_TAG,provider,state->stage_index,state->first_layer_index,state->layer_count,model_fp,layout_fp,service,socket_path,pool_bytes,workers);
	if ( status != SPARK_STATUS_OK )
		return(status);
	state->kv_slot_lane = (uint32_t *)malloc((size_t)state->kv_block_count * sizeof(uint32_t));
	state->kv_slot_logical = (uint32_t *)malloc((size_t)state->kv_block_count * sizeof(uint32_t));
	state->kv_slot_sequence = (uint64_t *)malloc((size_t)state->kv_block_count * sizeof(uint64_t));
	state->kv_slot_dirty = (uint8_t *)calloc((size_t)state->kv_block_count,sizeof(uint8_t));
	state->kv_slot_pinned = (uint8_t *)calloc((size_t)state->kv_block_count,sizeof(uint8_t));
	state->kv_slot_free_stack = (uint32_t *)malloc((size_t)state->kv_block_count * sizeof(uint32_t));
	state->kv_block_staging = malloc((size_t)staging_bytes);
	state->kv_gdn_staging = malloc(SPARK_MUSE_GLIMMER_MODULE_KV_GDN_RECORD_PLACEHOLDER_BYTES);
	if ( cudaMalloc((void **)&state->kv_table_indices_device,(size_t)SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT * SPARK_MUSE_GLIMMER_MODULE_KV_MAX_BLOCKS_PER_LANE * sizeof(uint32_t)) != cudaSuccess )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( cudaMalloc((void **)&state->kv_table_counts_device,(size_t)SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT * sizeof(uint32_t)) != cudaSuccess )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->kv_table_indices_host = (uint32_t *)malloc((size_t)SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT * SPARK_MUSE_GLIMMER_MODULE_KV_MAX_BLOCKS_PER_LANE * sizeof(uint32_t));
	if ( state->kv_slot_lane == 0 || state->kv_slot_logical == 0 || state->kv_slot_sequence == 0 || state->kv_slot_dirty == 0 || state->kv_slot_pinned == 0 || state->kv_slot_free_stack == 0 || state->kv_block_staging == 0 || state->kv_gdn_staging == 0 || state->kv_table_indices_host == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (index = 0u; index < state->kv_block_count; index++)
		state->kv_slot_free_stack[index] = index;
	state->kv_slot_free_count = state->kv_block_count;
	state->kv_evict_cursor = 0u;
	state->kv_tier_active = 1u;
	fprintf(stderr,"%s kv_tier_open provider=%s window=%u logical=%u physical=%u backing_bytes=%llu\n",SPARK_MUSE_GLIMMER_MODULE_TAG,provider,state->kv_block_count,state->kv_logical_page_capacity,state->kv_physical_page_capacity,(unsigned long long)state->kv_backing_maximum_bytes);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkMuseGlimmerModuleKvWaitBatch(SparkMuseGlimmerModuleState *state, SparkMuseGlimmerWorkControlKvBatchState *batch)
{
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t polls = 0u;
	struct timespec pause;
	pause.tv_sec = 0;
	pause.tv_nsec = 500000;
	while ( batch->state == SPARK_MUSE_GLIMMER_WORK_CONTROL_BATCH_SUBMITTED )
	{
		status = SparkMuseGlimmerWorkControlProgress(&state->kv_client,&state->kv_work);
		if ( status != SPARK_STATUS_OK )
			return(status);
		if ( batch->state == SPARK_MUSE_GLIMMER_WORK_CONTROL_BATCH_READY )
			break;
		if ( ++polls >= SPARK_MUSE_GLIMMER_MODULE_KV_POLL_BOUND )
		{
			fprintf(stderr,"%s kv_store_stall\n",SPARK_MUSE_GLIMMER_MODULE_TAG);
			return(SPARK_STATUS_IO_ERROR);
		}
		nanosleep(&pause,0);
	}
	if ( batch->state != SPARK_MUSE_GLIMMER_WORK_CONTROL_BATCH_READY || batch->status != SPARK_STATUS_OK )
		return(SPARK_STATUS_IO_ERROR);
	return(SparkMuseGlimmerWorkControlAcknowledge(batch));
}

static SparkStatus SparkMuseGlimmerModuleKvEvictSlot(SparkMuseGlimmerModuleState *state, uint32_t slot)
{
	SparkMuseGlimmerWorkControlKvBatchState *batch = &state->kv_work.evict;
	SparkKvStoreBlock blocks[1];
	uint32_t block_count = 0u,logical;
	uint64_t sequence_id;
	cudaError_t error;
	SparkStatus status;
	if ( state->kv_slot_dirty[slot] != 0u )
	{
		error = cudaMemcpy(state->kv_block_staging,(const uint8_t *)state->kv_cache_bf16 + (uint64_t)slot * state->kv_plan.block_record_bytes,(size_t)state->kv_plan.block_record_bytes,cudaMemcpyDeviceToHost);
		if ( error != cudaSuccess )
			return(SparkStageModuleCudaStatus(SPARK_MUSE_GLIMMER_MODULE_TAG,error,"kv_evict_copy"));
		logical = state->kv_slot_logical[slot];
		sequence_id = state->kv_slot_sequence[slot];
		status = SparkMuseGlimmerWorkControlBuildEvictBatch(&state->kv_plan,sequence_id,&logical,1u,0u,state->kv_block_staging,state->kv_gdn_staging,blocks,1u,&block_count);
		if ( status == SPARK_STATUS_OK )
			status = SparkMuseGlimmerWorkControlSubmit(&state->kv_client,batch,SPARK_KV_STORE_OPERATION_PUT,blocks,block_count,SPARK_MUSE_GLIMMER_WORK_CONTROL_RESTORE_PRIORITY_SPECULATIVE);
		if ( status == SPARK_STATUS_OK )
			status = SparkMuseGlimmerModuleKvWaitBatch(state,batch);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	if ( state->kv_logical_to_slot != 0 && state->kv_slot_lane[slot] != UINT32_MAX &&
		state->kv_slot_logical[slot] != UINT32_MAX && state->kv_logical_stride != 0u )
	{
		uint64_t evict_index = (uint64_t)state->kv_slot_lane[slot] * state->kv_logical_stride + state->kv_slot_logical[slot];
		if ( evict_index < state->kv_logical_to_slot_capacity &&
			state->kv_logical_to_slot[evict_index] == slot + 1u )
			state->kv_logical_to_slot[evict_index] = 0u;
	}
	state->kv_slot_dirty[slot] = 0u;
	state->kv_slot_pinned[slot] = 0u;
	state->kv_slot_lane[slot] = UINT32_MAX;
	state->kv_slot_logical[slot] = UINT32_MAX;
	state->kv_slot_sequence[slot] = 0u;
	state->kv_slot_free_stack[state->kv_slot_free_count++] = slot;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkMuseGlimmerModuleKvPrepareFrame(SparkMuseGlimmerModuleState *state, SparkMuseGlimmerModuleSlot *slot, SparkMuseGlimmerResidentDecodeStageFrameContext *context, SparkMuseGlimmerKvBlockTableView *table, uint32_t rows)
{
	SparkMuseGlimmerWorkControlKvBatchState *restore_batch = &state->kv_work.restore;
	SparkKvStoreBlock blocks[SPARK_MUSE_GLIMMER_MODULE_KV_STAGING_RECORDS];
	uint32_t packet_lane_counts[1],block_count,lanes_built;
	uint32_t lane_required[SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_sequence[SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t lane_list[SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t lane_count = 0u,row,lane_index,logical,slot_index;
	uint64_t logical_capacity;
	SparkStatus status;
	cudaError_t error;
	uint32_t uncommitted[SPARK_MUSE_GLIMMER_MODULE_KV_STAGING_RECORDS];
	uint32_t uncommitted_count = 0u;
	uint32_t unwind_index;
	SparkStatus fail_status;
	if ( state->kv_tier_active == 0u )
		return(SPARK_STATUS_OK);
	if ( context == 0 || context->decode_batch == 0 || context->decode_batch->row_sequence_ids == 0 || table == 0 || table->host_physical_block_indices == 0 || table->host_lane_physical_block_counts == 0 || table->physical_block_indices == 0 || table->lane_physical_block_counts == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	logical_capacity = (uint64_t)table->lane_count * table->lane_stride;
	if ( logical_capacity == 0u || table->lane_count > SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT || table->lane_stride > SPARK_MUSE_GLIMMER_MODULE_KV_MAX_BLOCKS_PER_LANE )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( logical_capacity > state->kv_logical_to_slot_capacity )
	{
		uint32_t *grown = (uint32_t *)realloc(state->kv_logical_to_slot,(size_t)logical_capacity * sizeof(uint32_t));
		if ( grown == 0 )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		memset(grown + state->kv_logical_to_slot_capacity,0,(size_t)(logical_capacity - state->kv_logical_to_slot_capacity) * sizeof(uint32_t));
		state->kv_logical_to_slot = grown;
		state->kv_logical_to_slot_capacity = logical_capacity;
		state->kv_logical_stride = table->lane_stride;
	}
	state->kv_logical_stride = table->lane_stride;
	for (row = 0u; row < rows; row++)
	{
		uint32_t lane = slot->host_row_lane_indices[row];
		uint32_t required_for_row = (slot->host_context_lengths[row] + SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS - 1u) / SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
		uint64_t sequence_id = context->decode_batch->row_sequence_ids[row];
		if ( lane >= table->lane_count || required_for_row > table->lane_stride || sequence_id == 0u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		for (lane_index = 0u; lane_index < lane_count; lane_index++)
			if ( lane_list[lane_index] == lane )
				break;
		if ( lane_index == lane_count )
		{
			lane_list[lane_count] = lane;
			lane_required[lane_count] = 0u;
			lane_sequence[lane_count] = sequence_id;
			lane_count++;
		}
		if ( required_for_row > lane_required[lane_index] )
			lane_required[lane_index] = required_for_row;
	}
	for (lane_index = 0u; lane_index < lane_count; lane_index++)
	{
		uint32_t lane = lane_list[lane_index];
		for (logical = 0u; logical < lane_required[lane_index]; logical++)
		{
			slot_index = state->kv_logical_to_slot[((uint64_t)lane * table->lane_stride) + logical];
			if ( slot_index != 0u )
				state->kv_slot_pinned[slot_index - 1u] = 1u;
		}
	}
	{
		SparkMuseGlimmerWorkControlPendingLane pending_lanes[SPARK_MUSE_GLIMMER_MODULE_KV_STAGING_RECORDS];
		uint32_t pending_slots[SPARK_MUSE_GLIMMER_MODULE_KV_STAGING_RECORDS];
		uint32_t pending_logical[SPARK_MUSE_GLIMMER_MODULE_KV_STAGING_RECORDS];
		uint64_t pending_lane_index[SPARK_MUSE_GLIMMER_MODULE_KV_STAGING_RECORDS];
		uint32_t batch_block_count = 0u,batch_index;
		memset(pending_lanes,0,sizeof(pending_lanes));
		for (lane_index = 0u; lane_index < lane_count; lane_index++)
		{
			uint32_t lane = lane_list[lane_index];
			for (logical = 0u; logical < lane_required[lane_index]; logical++)
			{
				uint32_t *residency = &state->kv_logical_to_slot[((uint64_t)lane * table->lane_stride) + logical];
				if ( *residency != 0u )
					continue;
				if ( state->kv_slot_free_count == 0u )
				{
					uint32_t scans = 0u;
					while ( state->kv_slot_pinned[state->kv_evict_cursor] != 0u )
					{
						state->kv_evict_cursor = (state->kv_evict_cursor + 1u) % state->kv_block_count;
						if ( ++scans > state->kv_block_count )
							{
								fail_status = SPARK_STATUS_CAPACITY_EXCEEDED;
								goto fail;
							}
					}
					status = SparkMuseGlimmerModuleKvEvictSlot(state,state->kv_evict_cursor);
					if ( status != SPARK_STATUS_OK )
						{
							fail_status = status;
							goto fail;
						}
				}
			slot_index = state->kv_slot_free_stack[--state->kv_slot_free_count];
			uncommitted[uncommitted_count++] = slot_index;
			pending_lanes[batch_block_count].sequence_id = lane_sequence[lane_index];
			pending_lanes[batch_block_count].nonresident_blocks = &pending_logical[batch_block_count];
			pending_lanes[batch_block_count].nonresident_block_count = 1u;
			pending_lanes[batch_block_count].gdn_nonresident = 0u;
			pending_logical[batch_block_count] = logical;
			pending_slots[batch_block_count] = slot_index;
			pending_lane_index[batch_block_count] = (uint64_t)lane_index;
			batch_block_count++;
			if ( batch_block_count == SPARK_MUSE_GLIMMER_MODULE_KV_STAGING_RECORDS )
			{
				packet_lane_counts[0] = batch_block_count;
				block_count = 0u;
				lanes_built = 0u;
				status = SparkMuseGlimmerWorkControlBuildRestoreBatch(&state->kv_plan,pending_lanes,batch_block_count,packet_lane_counts,1u,state->kv_block_staging,SPARK_MUSE_GLIMMER_MODULE_KV_STAGING_RECORDS,state->kv_gdn_staging,1u,blocks,SPARK_MUSE_GLIMMER_MODULE_KV_STAGING_RECORDS,&block_count,&lanes_built);
				if ( status == SPARK_STATUS_OK && lanes_built != batch_block_count )
					status = SPARK_STATUS_CAPACITY_EXCEEDED;
				if ( status == SPARK_STATUS_OK )
					status = SparkMuseGlimmerWorkControlSubmit(&state->kv_client,restore_batch,SPARK_KV_STORE_OPERATION_GET,blocks,block_count,SPARK_MUSE_GLIMMER_WORK_CONTROL_RESTORE_PRIORITY_IMMEDIATE);
				if ( status == SPARK_STATUS_OK )
					status = SparkMuseGlimmerModuleKvWaitBatch(state,restore_batch);
				if ( status != SPARK_STATUS_OK )
					{
						fail_status = status;
						goto fail;
					}
				for (batch_index = 0u; batch_index < batch_block_count; batch_index++)
				{
					error = cudaMemcpyAsync((uint8_t *)state->kv_cache_bf16 + (uint64_t)pending_slots[batch_index] * state->kv_plan.block_record_bytes,(const uint8_t *)state->kv_block_staging + ((uint64_t)batch_index * state->kv_plan.block_record_bytes),(size_t)state->kv_plan.block_record_bytes,cudaMemcpyHostToDevice,(cudaStream_t)slot->cuda_stream);
					if ( error != cudaSuccess )
						{
							fail_status = SparkStageModuleCudaStatus(SPARK_MUSE_GLIMMER_MODULE_TAG,error,"kv_restore_copy");
							goto fail;
						}
					state->kv_slot_lane[pending_slots[batch_index]] = lane_list[pending_lane_index[batch_index]];
					state->kv_slot_logical[pending_slots[batch_index]] = pending_logical[batch_index];
					state->kv_slot_sequence[pending_slots[batch_index]] = pending_lanes[batch_index].sequence_id;
					state->kv_slot_dirty[pending_slots[batch_index]] = 0u;
					state->kv_logical_to_slot[((uint64_t)lane_list[pending_lane_index[batch_index]] * table->lane_stride) + pending_logical[batch_index]] = pending_slots[batch_index] + 1u;
					for (unwind_index = 0u; unwind_index < uncommitted_count; unwind_index++)
						if ( uncommitted[unwind_index] == pending_slots[batch_index] )
						{
							uncommitted[unwind_index] = uncommitted[--uncommitted_count];
							break;
						}
				}
				batch_block_count = 0u;
			}
		}
	}
	if ( batch_block_count != 0u )
	{
			packet_lane_counts[0] = batch_block_count;
			block_count = 0u;
			lanes_built = 0u;
			status = SparkMuseGlimmerWorkControlBuildRestoreBatch(&state->kv_plan,pending_lanes,batch_block_count,packet_lane_counts,1u,state->kv_block_staging,SPARK_MUSE_GLIMMER_MODULE_KV_STAGING_RECORDS,state->kv_gdn_staging,1u,blocks,SPARK_MUSE_GLIMMER_MODULE_KV_STAGING_RECORDS,&block_count,&lanes_built);
			if ( status == SPARK_STATUS_OK && lanes_built != batch_block_count )
				status = SPARK_STATUS_CAPACITY_EXCEEDED;
			if ( status == SPARK_STATUS_OK )
				status = SparkMuseGlimmerWorkControlSubmit(&state->kv_client,restore_batch,SPARK_KV_STORE_OPERATION_GET,blocks,block_count,SPARK_MUSE_GLIMMER_WORK_CONTROL_RESTORE_PRIORITY_IMMEDIATE);
			if ( status == SPARK_STATUS_OK )
				status = SparkMuseGlimmerModuleKvWaitBatch(state,restore_batch);
			if ( status != SPARK_STATUS_OK )
				{
					fail_status = status;
					goto fail;
				}
			for (batch_index = 0u; batch_index < batch_block_count; batch_index++)
			{
				error = cudaMemcpyAsync((uint8_t *)state->kv_cache_bf16 + (uint64_t)pending_slots[batch_index] * state->kv_plan.block_record_bytes,(const uint8_t *)state->kv_block_staging + ((uint64_t)batch_index * state->kv_plan.block_record_bytes),(size_t)state->kv_plan.block_record_bytes,cudaMemcpyHostToDevice,(cudaStream_t)slot->cuda_stream);
				if ( error != cudaSuccess )
					{
						fail_status = SparkStageModuleCudaStatus(SPARK_MUSE_GLIMMER_MODULE_TAG,error,"kv_restore_copy");
						goto fail;
					}
				state->kv_slot_lane[pending_slots[batch_index]] = lane_list[pending_lane_index[batch_index]];
				state->kv_slot_logical[pending_slots[batch_index]] = pending_logical[batch_index];
				state->kv_slot_sequence[pending_slots[batch_index]] = pending_lanes[batch_index].sequence_id;
				state->kv_slot_dirty[pending_slots[batch_index]] = 0u;
				for (unwind_index = 0u; unwind_index < uncommitted_count; unwind_index++)
					if ( uncommitted[unwind_index] == pending_slots[batch_index] )
					{
						uncommitted[unwind_index] = uncommitted[--uncommitted_count];
						break;
					}
				state->kv_logical_to_slot[((uint64_t)lane_list[pending_lane_index[batch_index]] * table->lane_stride) + pending_logical[batch_index]] = pending_slots[batch_index] + 1u;
			}
		}
	}
	for (lane_index = 0u; lane_index < lane_count; lane_index++)
	{
		uint32_t lane = lane_list[lane_index];
		uint64_t lane_slice = (uint64_t)lane * table->lane_stride;
		memcpy(state->kv_table_indices_host + lane_slice,table->host_physical_block_indices + lane_slice,(size_t)table->lane_stride * sizeof(uint32_t));
		for (logical = 0u; logical < lane_required[lane_index]; logical++)
			state->kv_table_indices_host[lane_slice + logical] = state->kv_logical_to_slot[lane_slice + logical] - 1u;
		error = cudaMemcpyAsync((uint8_t *)state->kv_table_indices_device + (lane_slice * sizeof(uint32_t)),state->kv_table_indices_host + lane_slice,(size_t)table->lane_stride * sizeof(uint32_t),cudaMemcpyHostToDevice,(cudaStream_t)slot->cuda_stream);
		if ( error != cudaSuccess )
			{
				fail_status = SparkStageModuleCudaStatus(SPARK_MUSE_GLIMMER_MODULE_TAG,error,"kv_table_upload");
				goto fail;
			}
	}
	table->physical_block_indices = state->kv_table_indices_device;
	table->lane_physical_block_counts = state->kv_table_counts_device;
	error = cudaMemcpyAsync((void *)state->kv_table_counts_device,(const void *)table->host_lane_physical_block_counts,(size_t)table->lane_count * sizeof(uint32_t),cudaMemcpyHostToDevice,(cudaStream_t)slot->cuda_stream);
	if ( error != cudaSuccess )
		{
			fail_status = SparkStageModuleCudaStatus(SPARK_MUSE_GLIMMER_MODULE_TAG,error,"kv_table_upload");
			goto fail;
		}
	for (row = 0u; row < rows; row++)
	{
		uint32_t lane = slot->host_row_lane_indices[row];
		uint64_t position = slot->host_row_positions[row];
		slot_index = state->kv_logical_to_slot[((uint64_t)lane * table->lane_stride) + (uint32_t)(position / SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS)] - 1u;
		slot->host_slot_mapping[row] = slot_index * SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS + (uint32_t)(position % SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
	}
	if ( error != cudaSuccess )
		{
			fail_status = SparkStageModuleCudaStatus(SPARK_MUSE_GLIMMER_MODULE_TAG,error,"kv_slot_upload");
			goto fail;
		}
	for (lane_index = 0u; lane_index < lane_count; lane_index++)
	{
		uint32_t lane = lane_list[lane_index];
		for (logical = 0u; logical < lane_required[lane_index]; logical++)
		{
			slot_index = state->kv_logical_to_slot[((uint64_t)lane * table->lane_stride) + logical];
			if ( slot_index != 0u )
				state->kv_slot_pinned[slot_index - 1u] = 0u;
		}
	}
	return(SPARK_STATUS_OK);
fail:
	for (unwind_index = 0u; unwind_index < uncommitted_count; unwind_index++)
		state->kv_slot_free_stack[state->kv_slot_free_count++] = uncommitted[unwind_index];
	for (lane_index = 0u; lane_index < lane_count; lane_index++)
	{
		uint32_t fail_lane = lane_list[lane_index];
		for (logical = 0u; logical < lane_required[lane_index]; logical++)
		{
			slot_index = state->kv_logical_to_slot[((uint64_t)fail_lane * table->lane_stride) + logical];
			if ( slot_index != 0u )
				state->kv_slot_pinned[slot_index - 1u] = 0u;
		}
	}
	return(fail_status);
}

static void SparkMuseGlimmerModuleKvMarkWritten(SparkMuseGlimmerModuleState *state, SparkMuseGlimmerModuleSlot *slot, uint32_t rows)
{
	uint32_t row,slot_index;
	if ( state->kv_tier_active == 0u )
		return;
	for (row = 0u; row < rows; row++)
	{
		slot_index = slot->host_slot_mapping[row] / SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
		if ( slot_index < state->kv_block_count )
			state->kv_slot_dirty[slot_index] = 1u;
	}
}

extern cudaError_t SparkMuseGlimmerLaunchHeadArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint32_t *token_ids, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count);
extern cudaError_t SparkMuseGlimmerLaunchHeadShadowQuantize(cudaStream_t stream, const void *head_bf16, uint8_t *shadow_payload, uint8_t *shadow_scale, float *error_norm, uint32_t candidate_count, uint32_t hidden_dimension);
extern cudaError_t SparkMuseGlimmerLaunchHeadScreenedArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint8_t *shadow_payload, const uint8_t *shadow_scale, const float *error_norm, void *logits_bf16, uint32_t *candidate_ids, uint32_t *candidate_counts, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count);
extern cudaError_t SparkMuseGlimmerLaunchTpCombineAdd(cudaStream_t stream, void *destination_bf16, const void *source_bf16, uint32_t row_count, uint32_t width);


static SparkStatus SparkMuseGlimmerModuleTpCombineBf16(void *combine_context, void *destination_device, const void *source_device, uint32_t active_sequence_count, uint32_t hidden_dimension, void *cuda_stream)
{
	(void)combine_context;
	return(SparkStageModuleCudaStatus(SPARK_MUSE_GLIMMER_MODULE_TAG,SparkMuseGlimmerLaunchTpCombineAdd((cudaStream_t)cuda_stream,destination_device,source_device,active_sequence_count,hidden_dimension),"tp_combine"));
}

static void SparkMuseGlimmerModuleTpCompletion(void *context, const SparkTpDeviceCollectiveCompletion *completion)
{
	atomic_uint *flag = (atomic_uint *)context;
	atomic_store_explicit(flag,completion != 0 && completion->status == SPARK_STATUS_OK ? 1u : 2u,memory_order_release);
}

static SparkStatus SparkMuseGlimmerModuleInitializeTpCollective(SparkMuseGlimmerModuleState *state)
{
	SparkTpDeviceCollectiveConfig configuration;
	SparkTpDeviceCollectiveTopology topology;
	uint32_t credit,rank,route,route_count,memory_mode;
	uint64_t credit_bytes,total_bytes,offset;
	void *mapped_send,*mapped_receive;
	cudaError_t error;
	SparkStatus status;
	if ( state->tp_degree == 1u )
		return(SPARK_STATUS_OK);
	memset(&topology,0,sizeof(topology));
	topology.abi_version = SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_ABI_VERSION;
	topology.descriptor_bytes = SPARK_TP_DEVICE_COLLECTIVE_TOPOLOGY_BYTES;
	topology.rank_count = state->tp_degree;
	topology.algorithm_mask = SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_TREE;
	topology.rail_count = 0u;
	topology.direct_all_to_all_max_payload_bytes = 0u;
	topology.split_ring_min_payload_bytes = 0u;
	memcpy(topology.session_ports,state->tp_session_ports,
		sizeof(topology.session_ports));
	for (rank = 0u; rank < state->tp_degree; rank++)
		memcpy(topology.rank_hosts[rank],state->tp_hosts[rank],SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES);
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	configuration.backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
	configuration.tp_degree = state->tp_degree;
	configuration.tp_rank = state->tp_rank;
	configuration.operation_kind = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	configuration.credit_count = 8u;
	configuration.local_hidden_dimension = SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION;
	configuration.max_active_sequence_count = SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT;
	configuration.connect_timeout_milli = state->tp_connect_timeout_milli;
	configuration.operation_timeout_milli = state->tp_operation_timeout_milli;
	configuration.control_port_base = state->tp_control_port_base;
	configuration.collective_identifier = state->tp_collective_identifier;
	configuration.backend_module_path = state->tp_backend_path;
	configuration.local_host = state->tp_local_host;
	configuration.registration_cuda_stream = state->slots[0].cuda_stream;
	configuration.combine_bf16_function = SparkMuseGlimmerModuleTpCombineBf16;
	configuration.combine_context = state;
	status = SparkTpDeviceCollectiveApplyTopology(&topology,&configuration);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s tp_apply_topology_failed status=%d\n",SPARK_MUSE_GLIMMER_MODULE_TAG,(int)status);
		return(status);
	}
	status = SparkTpDeviceCollectiveCreditBindingRouteCount(&configuration,&route_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkTpDeviceCollectiveProbeMemoryMode(
		configuration.backend_kind,configuration.backend_module_path,
		&memory_mode);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s tp_probe_memory_mode_failed status=%d\n",SPARK_MUSE_GLIMMER_MODULE_TAG,(int)status);
		return(status);
	}
	credit_bytes = SparkTpDeviceCollectiveCreditBytes(configuration.max_active_sequence_count,configuration.local_hidden_dimension);
	total_bytes = credit_bytes * configuration.credit_count * route_count;
	status = SparkStageModuleDeviceAllocate(&state->ledger,total_bytes,&state->tp_collective_credit_send_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,total_bytes,&state->tp_collective_credit_receive_bf16);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( memory_mode == SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST )
	{
		mapped_send = 0;
		mapped_receive = 0;
		error = cudaHostAlloc(&state->tp_host_credit_send_bf16,total_bytes,cudaHostAllocPortable | cudaHostAllocMapped);
		if ( error == cudaSuccess )
			error = cudaHostAlloc(&state->tp_host_credit_receive_bf16,total_bytes,cudaHostAllocPortable | cudaHostAllocMapped);
		if ( error == cudaSuccess )
			error = cudaHostGetDevicePointer(&mapped_send,state->tp_host_credit_send_bf16,0u);
		if ( error == cudaSuccess )
			error = cudaHostGetDevicePointer(&mapped_receive,state->tp_host_credit_receive_bf16,0u);
		if ( error != cudaSuccess )
			return(SparkStageModuleCudaStatus(SPARK_MUSE_GLIMMER_MODULE_TAG,error,"tp_credit_mapped_alloc"));
		state->tp_collective_credit_send_bf16 = mapped_send;
		state->tp_collective_credit_receive_bf16 = mapped_receive;
	}
	offset = 0u;
	state->tp_credit_binding_count = 0u;
	for (route = 0u; route < route_count; route++)
		for (credit = 0u; credit < configuration.credit_count; credit++)
		{
			SparkTpDeviceCollectiveCreditBinding *binding = &state->tp_credit_bindings[state->tp_credit_binding_count++];
			binding->step_index = route;
			binding->credit_index = credit;
			binding->send_device = (uint8_t *)state->tp_collective_credit_send_bf16 + offset;
			binding->receive_device = (uint8_t *)state->tp_collective_credit_receive_bf16 + offset;
			binding->send_transport = memory_mode == SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST ? (uint8_t *)state->tp_host_credit_send_bf16 + offset : binding->send_device;
			binding->receive_transport = memory_mode == SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST ? (uint8_t *)state->tp_host_credit_receive_bf16 + offset : binding->receive_device;
			binding->flags = memory_mode == SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST ? SPARK_TP_DEVICE_COLLECTIVE_BINDING_KNOWN_FLAGS : 0u;
			binding->reserved0 = 0u;
			offset += credit_bytes;
		}
	configuration.credit_bindings = state->tp_credit_bindings;
	configuration.credit_binding_count = state->tp_credit_binding_count;
	status = SparkTpDeviceCollectiveCreate(&configuration,&state->tp_device_collective);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s tp_create_failed status=%d\n",SPARK_MUSE_GLIMMER_MODULE_TAG,(int)status);
		return(status);
	}
	state->tp_collective_initialized = 1u;
	fprintf(stderr,"%s tp_collective_open degree=%u rank=%u port_base=%u\n",SPARK_MUSE_GLIMMER_MODULE_TAG,state->tp_degree,state->tp_rank,state->tp_control_port_base);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkMuseGlimmerModuleTpAllReduceHidden(SparkMuseGlimmerModuleState *state, SparkMuseGlimmerModuleSlot *slot, void *device_bf16, uint32_t rows)
{
	SparkTpDeviceCollectiveSubmission submission;
	struct timespec pause;
	uint32_t polls,flag;
	SparkStatus status;
	if ( state->tp_degree == 1u )
		return(SPARK_STATUS_OK);
	if ( state->tp_collective_initialized == 0u )
		return(SPARK_STATUS_INTERNAL_ERROR);
	atomic_store_explicit(&state->tp_completion_flag,0u,memory_order_relaxed);
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = 0u;
	submission.active_sequence_count = rows;
	submission.logical_sequence_count = rows;
	submission.flags = SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission.ordinal = atomic_fetch_add_explicit(&state->tp_next_ordinal,1u,memory_order_relaxed);
	submission.local_device = device_bf16;
	submission.full_device = device_bf16;
	submission.cuda_stream = slot->cuda_stream;
	submission.completion_function = SparkMuseGlimmerModuleTpCompletion;
	submission.completion_context = &state->tp_completion_flag;
	status = SparkTpDeviceCollectiveSubmitBf16(&state->tp_device_collective,&submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	pause.tv_sec = 0u;
	pause.tv_nsec = 100000;
	for (polls = 0u; polls < 100000u; polls++)
	{
		flag = atomic_load_explicit(&state->tp_completion_flag,memory_order_acquire);
		if ( flag == 1u )
			return(SPARK_STATUS_OK);
		if ( flag == 2u )
			return(SPARK_STATUS_IO_ERROR);
		nanosleep(&pause,0);
	}
	fprintf(stderr,"%s tp_all_reduce_stall\n",SPARK_MUSE_GLIMMER_MODULE_TAG);
	return(SPARK_STATUS_IO_ERROR);
}

static SparkStatus SparkMuseGlimmerModuleTpMaxloc(SparkMuseGlimmerModuleState *state, SparkMuseGlimmerModuleSlot *slot, uint32_t rows)
{
	SparkTpDeviceCollectiveSubmission submission;
	struct timespec pause;
	uint32_t polls,flag;
	SparkStatus status;
	if ( state->tp_degree == 1u )
		return(SPARK_STATUS_OK);
	if ( state->tp_collective_initialized == 0u )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	atomic_store_explicit(&state->tp_completion_flag,0u,memory_order_relaxed);
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = 0u;
	submission.active_sequence_count = rows;
	submission.logical_sequence_count = rows;
	submission.flags = SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission.ordinal = atomic_fetch_add_explicit(&state->tp_next_ordinal,1u,memory_order_relaxed);
	submission.local_device = slot->head_maxloc_u64;
	submission.full_device = slot->head_maxloc_u64;
	submission.cuda_stream = slot->cuda_stream;
	submission.completion_function = SparkMuseGlimmerModuleTpCompletion;
	submission.completion_context = &state->tp_completion_flag;
	status = SparkTpDeviceCollectiveSubmitU64Max(&state->tp_device_collective,&submission);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	pause.tv_sec = 0u;
	pause.tv_nsec = 100000;
	for (polls = 0u; polls < 100000u; polls++)
	{
		flag = atomic_load_explicit(&state->tp_completion_flag,memory_order_acquire);
		if ( flag == 1u )
			return(SPARK_STATUS_OK);
		if ( flag == 2u )
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		nanosleep(&pause,0);
	}
	fprintf(stderr,"%s tp_maxloc_stall\n",SPARK_MUSE_GLIMMER_MODULE_TAG);
	SPARK_FAIL(SPARK_STATUS_IO_ERROR);
}

static SparkStatus SparkMuseGlimmerModuleExecuteFrame(void *module_state, SparkModelDriverFrame *frame);

static SparkStatus SparkMuseGlimmerModuleInitializeGate(void)
{
	uint32_t allow_unqualified_execution;
	allow_unqualified_execution = 0u;
	if ( SparkStageModuleEnvironmentUnsigned(SPARK_MUSE_GLIMMER_MODULE_TAG,"SPARK_MUSE_GLIMMER_ALLOW_UNQUALIFIED_EXECUTION",1u,1u,&allow_unqualified_execution) != SPARK_STATUS_OK || allow_unqualified_execution != 1u )
		return(SPARK_STATUS_MODULE_NOT_VALIDATED);
	return(SPARK_STATUS_OK);
}

static void SparkMuseGlimmerModuleDescribe(void *module_state, SparkStageModuleLifecycle *lifecycle)
{
	SparkMuseGlimmerModuleState *state = (SparkMuseGlimmerModuleState *)module_state;
	lifecycle->module_tag = SPARK_MUSE_GLIMMER_MODULE_TAG;
	lifecycle->ledger = &state->ledger;
	lifecycle->slot_states = state->slot_states;
	lifecycle->pipeline_slot_count = state->pipeline_slot_count;
	lifecycle->submitted_count = &state->submitted_count;
	lifecycle->completed_count = &state->completed_count;
	lifecycle->rejected_count = &state->rejected_count;
	lifecycle->failed_count = &state->failed_count;
	lifecycle->tokens_emitted = &state->tokens_emitted;
}

static SparkStatus SparkMuseGlimmerModulePrepare(
	void *module_state,
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services)
{
	SparkMuseGlimmerModuleState *state = (SparkMuseGlimmerModuleState *)module_state;
	const char *pack_path;
	SparkStatus status;
	(void)configuration;
	pack_path = 0;
	state->allow_unqualified_execution = 1u;
	atomic_init(&state->tp_completion_flag,0u);
	atomic_init(&state->tp_next_ordinal,0u);
	status = SparkMuseGlimmerModuleConfigure(state);
	if ( status == SPARK_STATUS_OK )
		SparkStageModuleAtomicStateArrayInitialize(state->slot_states,state->pipeline_slot_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentText(SPARK_MUSE_GLIMMER_MODULE_TAG,"SPARK_MUSE_GLIMMER_STAGE_PACK_PATH",&pack_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkMuseGlimmerModuleLoadPack(state,pack_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkMuseGlimmerModuleOpenKvTier(state,host_services);
	if ( status == SPARK_STATUS_OK )
	{
		int32_t sm_count = 0;
		cudaError_t attr = cudaDeviceGetAttribute(&sm_count,cudaDevAttrMultiProcessorCount,0);
		state->multiprocessor_count = attr == cudaSuccess && sm_count > 0 ? (uint32_t)sm_count : 1u;
		status = SparkMuseGlimmerModuleAllocatePools(state);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkMuseGlimmerModuleAllocateSlot(state,&state->slots[0]);
	if ( status == SPARK_STATUS_OK )
		status = SparkMuseGlimmerModuleAllocateSlotHostMirrors(state,&state->slots[0]);
	if ( status == SPARK_STATUS_OK && state->tp_degree > 1u )
		status = SparkMuseGlimmerModuleInitializeTpCollective(state);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"%s initialize_failed status=%d\n",SPARK_MUSE_GLIMMER_MODULE_TAG,(int)status);
	return(status);
}

static void SparkMuseGlimmerModuleReportReady(void *module_state)
{
	SparkMuseGlimmerModuleState *state = (SparkMuseGlimmerModuleState *)module_state;
	fprintf(stderr,"%s initialize ok slice=%u+%u tp=%u/%u owns_embedding=%u owns_head=%u\n",SPARK_MUSE_GLIMMER_MODULE_TAG,state->first_layer_index,state->layer_count,state->tp_rank,state->tp_degree,state->owns_embedding,state->owns_final_head);
}

static void SparkMuseGlimmerModuleStateTeardown(void *module_state)
{
	SparkMuseGlimmerModuleState *state = (SparkMuseGlimmerModuleState *)module_state;
	SparkStageKvClientClose(&state->kv_client);
	if ( state->tp_collective_initialized != 0u )
		SparkTpDeviceCollectiveDestroy(&state->tp_device_collective);
	free(state->kv_logical_to_slot);
	free(state->kv_slot_lane);
	free(state->kv_slot_logical);
	free(state->kv_slot_sequence);
	free(state->kv_slot_dirty);
	free(state->kv_slot_pinned);
	free(state->kv_slot_free_stack);
	free(state->kv_block_staging);
	free(state->kv_gdn_staging);
	free(state->kv_table_indices_host);
	if ( state->kv_table_indices_device != 0 )
		cudaFree(state->kv_table_indices_device);
	if ( state->kv_table_counts_device != 0 )
		cudaFree(state->kv_table_counts_device);
}

static SparkStatus SparkMuseGlimmerModuleAdmit(
	void *module_state,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	(void)module_state;
	(void)request;
	(void)decision;
	return(SPARK_STATUS_UNSUPPORTED);
}

static const SparkStageModuleLifecycleOps SparkMuseGlimmerModuleLifecycle =
{
	sizeof(SparkMuseGlimmerModuleState),
	SparkMuseGlimmerModuleInitializeGate,
	SparkMuseGlimmerModuleDescribe,
	SparkMuseGlimmerModulePrepare,
	SparkMuseGlimmerModuleReportReady,
	SparkMuseGlimmerModuleStateTeardown,
	SparkMuseGlimmerModuleExecuteFrame,
	SparkMuseGlimmerModuleAdmit,
	0
};

SparkStatus SparkMuseGlimmerResidentDecodeStageInitialize(
    const SparkFirmwareModuleConfiguration *configuration,
    const SparkFirmwareModuleHostServices *host_services,
    void **module_state)
{
	return(SparkStageModuleLifecycleInitialize(configuration,host_services,module_state,&SparkMuseGlimmerModuleLifecycle));
}

SparkStatus SparkMuseGlimmerResidentDecodeStageAdmit(
    void *module_state,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision)
{
	return(SparkStageModuleLifecycleAdmit(module_state,request,decision,&SparkMuseGlimmerModuleLifecycle));
}

SparkStatus SparkMuseGlimmerResidentDecodeStageExecute(
    void *module_state,
    SparkModelDriverFrame *frame)
{
	return(SparkStageModuleLifecycleExecute(module_state,frame,&SparkMuseGlimmerModuleLifecycle));
}

SparkStatus SparkMuseGlimmerResidentDecodeStageSnapshot(
    void *module_state,
    uint32_t program_id,
    SparkModelDriverRuntimeSnapshot *snapshot)
{
	(void)module_state;
	(void)program_id;
	(void)snapshot;
	return(SPARK_STATUS_UNSUPPORTED);
}

void SparkMuseGlimmerResidentDecodeStageDestroy(void *module_state)
{
	SparkStageModuleLifecycleDestroy(module_state,&SparkMuseGlimmerModuleLifecycle);
}


#define SPARK_MUSE_GLIMMER_MODULE_STAGED_ROW_CAPACITY \
	SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT



extern cudaError_t SparkMuseGlimmerLaunchEmbeddingGather(cudaStream_t stream, const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count, uint32_t tp_degree, uint32_t tp_rank);
extern cudaError_t SparkMuseGlimmerLaunchRmsNorm(cudaStream_t stream, const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon);
extern cudaError_t SparkMuseGlimmerLaunchCenteredRmsNorm(cudaStream_t stream, const void *input_bf16, const void *weight_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon);
extern cudaError_t SparkMuseGlimmerLaunchHeadRmsNorm(cudaStream_t stream, const void *input_bf16, const void *weight_bf16, void *output_bf16, uint32_t row_count, uint32_t head_count, uint32_t head_dimension, float epsilon, float head_multiply);
extern cudaError_t SparkMuseGlimmerLaunchCopyRows(cudaStream_t stream, const void *source_bf16, void *destination_bf16, uint32_t row_count, uint32_t dimension);
extern cudaError_t SparkMuseGlimmerLaunchAddRows(cudaStream_t stream, const void *a_bf16, const void *b_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension);
extern cudaError_t SparkMuseGlimmerLaunchLinear(cudaStream_t stream, const void *weight_bf16, const void *input_bf16, void *output_bf16, uint32_t row_count, uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessors);
extern cudaError_t SparkMuseGlimmerLaunchLinearScores(cudaStream_t stream, const void *weight_bf16, const void *input_bf16, float *scores_f32, uint32_t row_count, uint32_t input_dimension, uint32_t output_dimension, uint32_t multiprocessors);
extern cudaError_t SparkMuseGlimmerLaunchSplitQkv(cudaStream_t stream, const void *fused_bf16, void *query_gate_bf16, void *key_bf16, void *value_bf16, uint32_t row_count, uint32_t tp_degree);
extern cudaError_t SparkMuseGlimmerLaunchSplitQueryGate(cudaStream_t stream, const void *query_gate_bf16, void *query_bf16, void *gate_bf16, uint32_t row_count, uint32_t local_head_count);
extern cudaError_t SparkMuseGlimmerLaunchQkNorm(cudaStream_t stream, const void *input_bf16, void *output_bf16, uint32_t head_count, float head_multiply, uint32_t row_count);
extern cudaError_t SparkMuseGlimmerLaunchRope(cudaStream_t stream, void *rows_bf16, const uint32_t *positions, uint32_t head_count, uint32_t row_count);
extern cudaError_t SparkMuseGlimmerLaunchKvStore(cudaStream_t stream, const void *views, uint32_t layer_index, const void *key_bf16, const void *value_bf16, const uint32_t *sequence_of_row, const uint32_t *positions, uint32_t row_count, uint32_t local_kv_head_count);
extern cudaError_t SparkMuseGlimmerLaunchWindowPositions(cudaStream_t stream, const uint32_t *sequence_of_row, const uint32_t *context_lengths, const uint32_t *positions, uint32_t row_count, uint32_t *window_positions);
extern cudaError_t SparkMuseGlimmerLaunchAttentionDecode(cudaStream_t stream, const void *views, uint32_t layer_index, const void *query_bf16, const uint32_t *sequence_of_row, const uint32_t *context_lengths, const uint32_t *window_positions, const uint32_t *positions, void *head_out_bf16, uint32_t row_count, uint32_t local_head_count, uint32_t local_kv_head_count);
extern cudaError_t SparkMuseGlimmerLaunchOutputGate(cudaStream_t stream, void *head_out_bf16, const void *gate_bf16, uint32_t row_count, uint32_t local_query_dimension);
extern cudaError_t SparkMuseGlimmerLaunchSiluMul(cudaStream_t stream, const void *gate_up_bf16, void *intermediate_bf16, uint32_t row_count, uint32_t local_intermediate);
extern cudaError_t SparkMuseGlimmerLaunchResidualAdd(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, uint32_t row_count, uint32_t dimension);
extern cudaError_t SparkMuseGlimmerLaunchTpCombineAdd(cudaStream_t stream, void *destination_bf16, const void *source_bf16, uint32_t row_count, uint32_t width);
extern cudaError_t SparkMuseGlimmerLaunchHeadArgmaxPack(cudaStream_t stream, const float *scores_f32, uint32_t *local_token_ids, uint64_t *maxloc, uint32_t row_count, uint32_t candidate_count, uint32_t tp_degree, uint32_t tp_rank);
extern cudaError_t SparkMuseGlimmerLaunchHeadMaxlocUnpack(cudaStream_t stream, const uint64_t *maxloc, uint32_t *token_ids, uint32_t row_count);
#define SPARK_MUSE_GLIMMER_MODULE_HOST_ROW_CAPACITY \
	SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT

static SparkStatus SparkMuseGlimmerModuleRunLayer(SparkMuseGlimmerModuleState *state, SparkMuseGlimmerModuleSlot *slot, const SparkMuseGlimmerKvViewShim *views, uint32_t layer, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	const SparkMuseGlimmerLayerWeights *weights = &state->layers[layer];
	uint32_t tp = state->tp_degree;
	uint32_t local_q_heads = SPARK_MUSE_GLIMMER_MODEL_ATTN_LOCAL_QUERY_HEAD_COUNT(tp);
	uint32_t local_kv_heads = SPARK_MUSE_GLIMMER_MODEL_ATTN_LOCAL_KV_HEAD_COUNT(tp);
	uint32_t local_query_dimension = SPARK_MUSE_GLIMMER_MODEL_ATTN_LOCAL_QUERY_DIMENSION(tp);
	uint32_t local_intermediate = SPARK_MUSE_GLIMMER_MODEL_MLP_LOCAL_INTERMEDIATE(tp);
	uint32_t layer_ordinal = layer - state->first_layer_index;
	uint32_t full_attention = SPARK_MUSE_GLIMMER_MODEL_LAYER_IS_FULL_ATTENTION(layer);
	cudaError_t error;
	SparkStatus status;
	error = SparkMuseGlimmerLaunchCopyRows(stream,slot->hidden_bf16,slot->residual_bf16,rows,SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION);
	if ( error == cudaSuccess )
		error = SparkMuseGlimmerLaunchCenteredRmsNorm(stream,slot->hidden_bf16,weights->input_norm_weight_bf16,slot->normalized_bf16,rows,SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION,SPARK_MUSE_GLIMMER_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkMuseGlimmerLaunchLinear(stream,weights->qgkv.weight_payload,slot->normalized_bf16,slot->fused_qgkv_bf16,rows,SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION,SPARK_MUSE_GLIMMER_MODEL_QGKV_LOCAL_ROWS(tp),state->multiprocessor_count);
	if ( error == cudaSuccess )
		error = SparkMuseGlimmerLaunchSplitQkv(stream,slot->fused_qgkv_bf16,slot->query_gate_bf16,slot->key_bf16,slot->value_bf16,rows,tp);
	if ( error == cudaSuccess )
		error = SparkMuseGlimmerLaunchSplitQueryGate(stream,slot->query_gate_bf16,slot->query_bf16,slot->attn_gate_bf16,rows,local_q_heads);
	if ( error == cudaSuccess )
		error = SparkMuseGlimmerLaunchQkNorm(stream,slot->query_bf16,slot->query_bf16,local_q_heads,SPARK_MUSE_GLIMMER_MODEL_ATTN_QK_SCALE_FACTOR,rows);
	if ( error == cudaSuccess )
		error = SparkMuseGlimmerLaunchQkNorm(stream,slot->key_bf16,slot->key_bf16,local_kv_heads,1.0f,rows);
	if ( error == cudaSuccess && full_attention == 0u )
	{
		error = SparkMuseGlimmerLaunchRope(stream,slot->query_bf16,slot->row_positions_u32,local_q_heads,rows);
		if ( error == cudaSuccess )
			error = SparkMuseGlimmerLaunchRope(stream,slot->key_bf16,slot->row_positions_u32,local_kv_heads,rows);
	}
	if ( error == cudaSuccess )
		error = SparkMuseGlimmerLaunchKvStore(stream,views,layer_ordinal,slot->key_bf16,slot->value_bf16,slot->row_lane_indices,slot->row_positions_u32,rows,local_kv_heads);
	if ( error == cudaSuccess && full_attention == 0u )
	{
		error = SparkMuseGlimmerLaunchWindowPositions(stream,slot->row_lane_indices,slot->context_lengths,slot->row_positions_u32,rows,slot->window_positions);
		if ( error == cudaSuccess )
			error = SparkMuseGlimmerLaunchAttentionDecode(stream,views,layer_ordinal,slot->query_bf16,slot->row_lane_indices,slot->context_lengths,slot->window_positions,slot->row_positions_u32,slot->head_out_bf16,rows,local_q_heads,local_kv_heads);
	}
	if ( error == cudaSuccess && full_attention != 0u )
		error = SparkMuseGlimmerLaunchAttentionDecode(stream,views,layer_ordinal,slot->query_bf16,slot->row_lane_indices,slot->context_lengths,0,slot->row_positions_u32,slot->head_out_bf16,rows,local_q_heads,local_kv_heads);
	if ( error == cudaSuccess )
		error = SparkMuseGlimmerLaunchOutputGate(stream,slot->head_out_bf16,slot->attn_gate_bf16,rows,local_query_dimension);
	if ( error == cudaSuccess )
		error = SparkMuseGlimmerLaunchLinear(stream,weights->output.weight_payload,slot->head_out_bf16,slot->delta_bf16,rows,local_query_dimension,SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION,state->multiprocessor_count);
	status = SPARK_STATUS_OK;
	if ( error == cudaSuccess && state->tp_degree > 1u )
		status = SparkMuseGlimmerModuleTpAllReduceHidden(state,slot,slot->delta_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkMuseGlimmerLaunchCenteredRmsNorm(stream,slot->delta_bf16,weights->post_attention_norm_weight_bf16,slot->normalized_bf16,rows,SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION,SPARK_MUSE_GLIMMER_MODEL_POST_NORM_EPSILON);
	if ( status == SPARK_STATUS_OK && error == cudaSuccess )
		error = SparkMuseGlimmerLaunchAddRows(stream,slot->residual_bf16,slot->normalized_bf16,slot->hidden_bf16,rows,SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION);
	if ( status == SPARK_STATUS_OK && error == cudaSuccess )
		error = SparkMuseGlimmerLaunchCenteredRmsNorm(stream,slot->hidden_bf16,weights->pre_feedforward_norm_weight_bf16,slot->normalized_bf16,rows,SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION,SPARK_MUSE_GLIMMER_MODEL_RMS_NORM_EPSILON);
	if ( status == SPARK_STATUS_OK && error == cudaSuccess )
		error = SparkMuseGlimmerLaunchLinear(stream,weights->gate_up.weight_payload,slot->normalized_bf16,slot->gate_up_bf16,rows,SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION,2u * local_intermediate,state->multiprocessor_count);
	if ( status == SPARK_STATUS_OK && error == cudaSuccess )
		error = SparkMuseGlimmerLaunchSiluMul(stream,slot->gate_up_bf16,slot->intermediate_bf16,rows,local_intermediate);
	if ( status == SPARK_STATUS_OK && error == cudaSuccess )
		error = SparkMuseGlimmerLaunchLinear(stream,weights->down.weight_payload,slot->intermediate_bf16,slot->delta_bf16,rows,local_intermediate,SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION,state->multiprocessor_count);
	if ( status == SPARK_STATUS_OK && error == cudaSuccess && state->tp_degree > 1u )
		status = SparkMuseGlimmerModuleTpAllReduceHidden(state,slot,slot->delta_bf16,rows);
	if ( status == SPARK_STATUS_OK && error == cudaSuccess )
		error = SparkMuseGlimmerLaunchCenteredRmsNorm(stream,slot->delta_bf16,weights->post_feedforward_norm_weight_bf16,slot->normalized_bf16,rows,SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION,SPARK_MUSE_GLIMMER_MODEL_POST_NORM_EPSILON);
	if ( status == SPARK_STATUS_OK && error == cudaSuccess )
		error = SparkMuseGlimmerLaunchAddRows(stream,slot->residual_bf16,slot->normalized_bf16,slot->hidden_bf16,rows,SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	if ( error != cudaSuccess )
		SPARK_FAIL(SparkStageModuleCudaStatus(SPARK_MUSE_GLIMMER_MODULE_TAG,error,"layer"));
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkMuseGlimmerModuleAllocatePools(SparkMuseGlimmerModuleState *state)
{
	uint64_t local_cache_token_elements,cache_elements;
	SparkStatus status;
	state->cache_layer_count = state->layer_count;
	local_cache_token_elements = 2ull * SPARK_MUSE_GLIMMER_MODEL_ATTN_LOCAL_KV_HEAD_COUNT(state->tp_degree) * SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION;
	state->cache_layer_stride = (uint64_t)SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS * local_cache_token_elements;
	state->cache_block_stride = state->cache_layer_stride * state->cache_layer_count;
	cache_elements = state->cache_block_stride * state->kv_block_count;
	status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,cache_elements * SPARK_MUSE_GLIMMER_MODEL_BF16_ELEMENT_BYTES,&state->kv_cache_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,sizeof(SparkMuseGlimmerKvViewShim),(void **)&state->kv_access_error);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(size_t)SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT * SPARK_MUSE_GLIMMER_MODULE_KV_MAX_BLOCKS_PER_LANE * sizeof(uint32_t),(void **)&state->kv_table_indices_device);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(size_t)SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT * sizeof(uint32_t),(void **)&state->kv_table_counts_device);
	if ( status == SPARK_STATUS_OK )
	{
		state->kv_table_indices_host = (uint32_t *)malloc((size_t)SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT * SPARK_MUSE_GLIMMER_MODULE_KV_MAX_BLOCKS_PER_LANE * sizeof(uint32_t));
		if ( state->kv_table_indices_host == 0 )
			status = SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(SPARK_STATUS_OK);
}

extern uint32_t SparkMuseGlimmerKvViewBytes(void);
static SparkStatus SparkMuseGlimmerModuleAllocateSlot(SparkMuseGlimmerModuleState *state, SparkMuseGlimmerModuleSlot *slot);
static SparkStatus SparkMuseGlimmerModuleAllocateSlotHostMirrors(SparkMuseGlimmerModuleState *state, SparkMuseGlimmerModuleSlot *slot);

static SparkStatus SparkMuseGlimmerModuleAllocateSlot(SparkMuseGlimmerModuleState *state, SparkMuseGlimmerModuleSlot *slot)
{
	uint64_t rows = state->max_active_sequence_count;
	uint64_t hidden_bytes = rows * SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION * SPARK_MUSE_GLIMMER_MODEL_BF16_ELEMENT_BYTES;
	uint32_t tp = state->tp_degree;
	uint64_t qgkv_bytes = rows * SPARK_MUSE_GLIMMER_MODEL_QGKV_LOCAL_ROWS(tp) * SPARK_MUSE_GLIMMER_MODEL_BF16_ELEMENT_BYTES;
	uint64_t query_gate_bytes = rows * SPARK_MUSE_GLIMMER_MODEL_ATTN_LOCAL_QUERY_GATE_DIMENSION(tp) * SPARK_MUSE_GLIMMER_MODEL_BF16_ELEMENT_BYTES;
	uint64_t query_bytes = rows * SPARK_MUSE_GLIMMER_MODEL_ATTN_LOCAL_QUERY_DIMENSION(tp) * SPARK_MUSE_GLIMMER_MODEL_BF16_ELEMENT_BYTES;
	uint64_t kv_bytes = rows * SPARK_MUSE_GLIMMER_MODEL_ATTN_LOCAL_KV_DIMENSION(tp) * SPARK_MUSE_GLIMMER_MODEL_BF16_ELEMENT_BYTES;
	uint64_t gate_up_bytes = rows * 2ull * SPARK_MUSE_GLIMMER_MODEL_MLP_LOCAL_INTERMEDIATE(tp) * SPARK_MUSE_GLIMMER_MODEL_BF16_ELEMENT_BYTES;
	uint64_t intermediate_bytes = rows * SPARK_MUSE_GLIMMER_MODEL_MLP_LOCAL_INTERMEDIATE(tp) * SPARK_MUSE_GLIMMER_MODEL_BF16_ELEMENT_BYTES;
	SparkStatus status;
	cudaStream_t stream = 0;
	status = SparkStageModuleCudaStatus(SPARK_MUSE_GLIMMER_MODULE_TAG,cudaStreamCreate(&stream),"cudaStreamCreate");
	if ( status != SPARK_STATUS_OK )
		return(status);
	slot->cuda_stream = stream;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->input_token_ids);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->output_token_ids);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
	{
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->local_token_ids);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint64_t),(void **)&slot->head_maxloc_u64);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_MUSE_GLIMMER_MODEL_VOCAB_LOCAL_ROWS(tp) * sizeof(float),(void **)&slot->head_scores_f32);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->row_lane_indices);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->row_positions_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->context_lengths);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_MUSE_GLIMMER_MODEL_SLIDING_WINDOW * sizeof(uint32_t),(void **)&slot->window_positions);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->row_cold);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint64_t),(void **)&slot->row_positions);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,hidden_bytes,&slot->hidden_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,hidden_bytes,&slot->residual_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,hidden_bytes,&slot->normalized_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,hidden_bytes,&slot->delta_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,qgkv_bytes,&slot->fused_qgkv_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,query_gate_bytes,&slot->query_gate_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,query_bytes,&slot->query_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,query_bytes,&slot->attn_gate_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,kv_bytes,&slot->key_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,kv_bytes,&slot->value_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,query_bytes,&slot->head_out_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,gate_up_bytes,&slot->gate_up_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,intermediate_bytes,&slot->intermediate_bf16);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkMuseGlimmerModuleAllocateSlotHostMirrors(SparkMuseGlimmerModuleState *state, SparkMuseGlimmerModuleSlot *slot)
{
	uint64_t rows = state->max_active_sequence_count;
	SparkStatus status = SPARK_STATUS_OK;
	slot->host_row_lane_indices = (uint32_t *)malloc(SPARK_MUSE_GLIMMER_MODULE_HOST_ROW_CAPACITY * sizeof(uint32_t));
	slot->host_row_positions = (uint64_t *)malloc(SPARK_MUSE_GLIMMER_MODULE_HOST_ROW_CAPACITY * sizeof(uint64_t));
	slot->host_row_positions_u32 = (uint32_t *)malloc(SPARK_MUSE_GLIMMER_MODULE_HOST_ROW_CAPACITY * sizeof(uint32_t));
	slot->host_row_cold = (uint32_t *)malloc(rows * sizeof(uint32_t));
	slot->host_slot_mapping = (uint32_t *)malloc(SPARK_MUSE_GLIMMER_MODULE_HOST_ROW_CAPACITY * sizeof(uint32_t));
	slot->host_context_lengths = (uint32_t *)malloc(SPARK_MUSE_GLIMMER_MODULE_HOST_ROW_CAPACITY * sizeof(uint32_t));
	if ( slot->host_row_lane_indices == 0 || slot->host_row_positions == 0 ||
		slot->host_row_positions_u32 == 0 || slot->host_row_cold == 0 ||
		slot->host_slot_mapping == 0 || slot->host_context_lengths == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(status);
}

static SparkStatus SparkMuseGlimmerModuleUploadRows(SparkMuseGlimmerModuleState *state, SparkMuseGlimmerModuleSlot *slot, const SparkModelDriverFrame *frame, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	uint32_t token_guard;
	error = cudaMemcpyAsync(slot->row_lane_indices,slot->host_row_lane_indices,rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->row_positions,slot->host_row_positions,rows * sizeof(uint64_t),cudaMemcpyHostToDevice,stream);
	for (token_guard = 0; token_guard < rows; token_guard++)
		slot->host_row_positions_u32[token_guard] = (uint32_t)slot->host_row_positions[token_guard];
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->row_positions_u32,slot->host_row_positions_u32,rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->row_cold,slot->host_row_cold,rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->context_lengths,slot->host_context_lengths,rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess && state->owns_embedding != 0u )
	{
		if ( frame->buffer_count < 1u || frame->buffers == 0 || frame->buffers[0].address == 0 )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for (token_guard = 0; token_guard < rows; token_guard++)
			if ( ((const uint32_t *)frame->buffers[0].address)[token_guard] >= SPARK_MUSE_GLIMMER_MODEL_OUTPUT_VOCAB_COUNT )
			{
				fprintf(stderr,"%s token_id_out_of_range row=%u\n",SPARK_MUSE_GLIMMER_MODULE_TAG,token_guard);
				SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
			}
		error = cudaMemcpyAsync(slot->input_token_ids,frame->buffers[0].address,rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	}
	if ( error != cudaSuccess )
		SPARK_FAIL(SparkStageModuleCudaStatus(SPARK_MUSE_GLIMMER_MODULE_TAG,error,"stage_upload"));
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkMuseGlimmerModuleEmitHead(SparkMuseGlimmerModuleState *state, SparkMuseGlimmerModuleSlot *slot, SparkModelDriverFrame *frame, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint32_t tp = state->tp_degree;
	uint32_t vocab_local = SPARK_MUSE_GLIMMER_MODEL_VOCAB_LOCAL_ROWS(tp);
	uint32_t out_index = state->owns_embedding != 0u ? 1u : 0u;
	cudaError_t error;
	SparkStatus status;
	if ( frame->buffer_count <= out_index || frame->buffers == 0 || frame->buffers[out_index].address == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	error = SparkMuseGlimmerLaunchRmsNorm(stream,slot->hidden_bf16,state->final_norm_weight_bf16,slot->normalized_bf16,rows,SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION,SPARK_MUSE_GLIMMER_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkMuseGlimmerLaunchLinearScores(stream,state->lm_head_weight_bf16,slot->normalized_bf16,slot->head_scores_f32,rows,SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION,vocab_local,state->multiprocessor_count);
	if ( error == cudaSuccess )
		error = SparkMuseGlimmerLaunchHeadArgmaxPack(stream,slot->head_scores_f32,slot->local_token_ids,slot->head_maxloc_u64,rows,vocab_local,tp,state->tp_rank);
	status = SPARK_STATUS_OK;
	if ( error == cudaSuccess && tp > 1u )
		status = SparkMuseGlimmerModuleTpMaxloc(state,slot,rows);
	if ( status == SPARK_STATUS_OK && error == cudaSuccess )
		error = SparkMuseGlimmerLaunchHeadMaxlocUnpack(stream,slot->head_maxloc_u64,slot->output_token_ids,rows);
	if ( status == SPARK_STATUS_OK && error == cudaSuccess )
		error = cudaMemcpyAsync(frame->buffers[out_index].address,slot->output_token_ids,rows * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	if ( error != cudaSuccess )
		SPARK_FAIL(SparkStageModuleCudaStatus(SPARK_MUSE_GLIMMER_MODULE_TAG,error,"head_emit"));
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkMuseGlimmerModuleValidateFrameContext(SparkMuseGlimmerModuleState *state, const SparkMuseGlimmerResidentDecodeStageFrameContext *context)
{
	uint32_t wants_input,wants_output,has_input,has_output;
	wants_input = state->stage_index != 0u ? 1u : 0u;
	wants_output = state->stage_index + 1u < state->stage_count ? 1u : 0u;
	if ( context == 0 )
		return((wants_input == 0u && wants_output == 0u) || state->allow_unqualified_execution != 0u
			? SPARK_STATUS_OK
			: SPARK_STATUS_INVALID_ARGUMENT);
	if ( context->abi_version != SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION ||
		context->descriptor_bytes != sizeof(*context) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	has_input = (context->flags & SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT) != 0u ? 1u : 0u;
	has_output = (context->flags & SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT) != 0u ? 1u : 0u;
	if ( has_input != wants_input || has_output != wants_output )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( has_input != 0u && (context->hidden_input_transport_session == 0 || context->hidden_input_post_receive_function == 0) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( has_output != 0u && (context->hidden_output_transport_session == 0 || context->hidden_output_send_function == 0) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (context->flags & SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW) != 0u )
		return(SPARK_STATUS_UNSUPPORTED);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkMuseGlimmerModuleConsumeHiddenInput(SparkMuseGlimmerModuleSlot *slot, SparkMuseGlimmerResidentDecodeStageFrameContext *context, uint32_t rows)
{
	SparkHiddenTransportPacket *packet = &context->hidden_input_packet;
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	SparkStatus status;
	memset(packet,0,sizeof(*packet));
	status = context->hidden_input_post_receive_function(context->hidden_input_transport_session,packet);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( packet->hidden_bf16 == 0 || packet->active_sequence_count < rows ||
		packet->hidden_dimension != SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION ||
		packet->bytes_per_sequence < SPARK_MUSE_GLIMMER_MODEL_HIDDEN_BF16_BYTES )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	error = cudaMemcpyAsync(slot->hidden_bf16,packet->hidden_bf16,(uint64_t)rows * SPARK_MUSE_GLIMMER_MODEL_HIDDEN_BF16_BYTES,cudaMemcpyDeviceToDevice,stream);
	return(SparkStageModuleCudaStatus(SPARK_MUSE_GLIMMER_MODULE_TAG,error,"hidden_input"));
}

static SparkStatus SparkMuseGlimmerModuleEmitHiddenOutput(SparkMuseGlimmerModuleSlot *slot, SparkMuseGlimmerResidentDecodeStageFrameContext *context, uint32_t rows)
{
	SparkHiddenTransportPacket *packet = &context->hidden_output_packet;
	memset(packet,0,sizeof(*packet));
	packet->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
	packet->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_PACKET_BYTES;
	packet->flags = SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_BF16 | SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_DEVICE_POINTER;
	packet->active_sequence_count = rows;
	packet->hidden_dimension = SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION;
	packet->bytes_per_sequence = SPARK_MUSE_GLIMMER_MODEL_HIDDEN_BF16_BYTES;
	packet->hidden_bf16 = slot->hidden_bf16;
	packet->cuda_stream = slot->cuda_stream;
	return(context->hidden_output_send_function(context->hidden_output_transport_session,packet));
}

static SparkStatus SparkMuseGlimmerModuleRunDecode(SparkMuseGlimmerModuleState *state, SparkMuseGlimmerModuleSlot *slot, SparkModelDriverFrame *frame, SparkMuseGlimmerResidentDecodeStageFrameContext *context, uint32_t rows)
{
	SparkMuseGlimmerKvBlockTableView table;
	SparkMuseGlimmerKvViewShim views[SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	SparkMuseGlimmerFrameErrorShim error_record;
	uint32_t layer,row;
	uint32_t wants_input,wants_output;
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	SparkStatus status;
	cudaError_t error;
	if ( context != 0 && context->kv_block_table != 0 )
		table = *context->kv_block_table;
	else
	{
		memset(&table,0,sizeof(table));
		for (row = 0; row < state->max_active_sequence_count; row++)
			state->kv_table_indices_host[row] = row;
		error = cudaMemcpyAsync(state->kv_table_indices_device,state->kv_table_indices_host,(size_t)state->max_active_sequence_count * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
		if ( error != cudaSuccess )
			SPARK_FAIL(SparkStageModuleCudaStatus(SPARK_MUSE_GLIMMER_MODULE_TAG,error,"kv_identity_table"));
		table.abi_version = SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_KV_BLOCK_TABLE_ABI_VERSION;
		table.descriptor_bytes = sizeof(table);
		table.block_token_count = SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
		table.lane_count = state->max_active_sequence_count;
		table.lane_stride = 1u;
		table.lane_capacity = state->max_active_sequence_count;
		table.physical_block_indices = state->kv_table_indices_device;
		table.lane_physical_block_counts = state->kv_table_counts_device;
		table.host_physical_block_indices = state->kv_table_indices_host;
		table.host_lane_physical_block_counts = 0;
	}
	if ( SparkMuseGlimmerKvViewBytes() != sizeof(SparkMuseGlimmerKvViewShim) )
		SPARK_FAIL(SPARK_STATUS_ABI_MISMATCH);
	status = SparkMuseGlimmerModuleUploadRows(state,slot,frame,rows);
	if ( status == SPARK_STATUS_OK )
		status = SparkMuseGlimmerModuleKvPrepareFrame(state,slot,context,&table,rows);
	if ( status != SPARK_STATUS_OK )
		return(status);
	for (layer = 0; layer < state->layer_count; layer++)
	{
		views[layer].pool = (uint8_t *)state->kv_cache_bf16 + ((uint64_t)(state->first_layer_index + layer) * state->cache_layer_stride);
		views[layer].page_table = table.physical_block_indices;
		views[layer].page_table_stride = table.lane_stride;
		views[layer].sequence_count = table.lane_count;
		views[layer].pool_page_count = state->kv_block_count;
		views[layer].access_error = state->kv_access_error;
	}
	error = cudaMemsetAsync(state->kv_access_error,0,sizeof(error_record),stream);
	if ( error != cudaSuccess )
		SPARK_FAIL(SparkStageModuleCudaStatus(SPARK_MUSE_GLIMMER_MODULE_TAG,error,"kv_error_reset"));
	wants_input = context != 0 && (context->flags & SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT) != 0u ? 1u : 0u;
	if ( state->owns_embedding != 0u )
	{
		error = SparkMuseGlimmerLaunchEmbeddingGather(stream,slot->input_token_ids,state->token_embedding_bf16,slot->hidden_bf16,rows,state->tp_degree,state->tp_rank);
		if ( error == cudaSuccess && state->tp_degree > 1u )
			status = SparkMuseGlimmerModuleTpAllReduceHidden(state,slot,slot->hidden_bf16,rows);
		if ( status == SPARK_STATUS_OK && error == cudaSuccess )
			error = SparkMuseGlimmerLaunchHeadRmsNorm(stream,slot->hidden_bf16,0,slot->hidden_bf16,rows,1u,SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION,SPARK_MUSE_GLIMMER_MODEL_RMS_NORM_EPSILON,1.0f);
		if ( error != cudaSuccess )
			SPARK_FAIL(SparkStageModuleCudaStatus(SPARK_MUSE_GLIMMER_MODULE_TAG,error,"embedding"));
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	else if ( wants_input != 0u )
	{
		status = SparkMuseGlimmerModuleConsumeHiddenInput(slot,context,rows);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	for (layer = state->first_layer_index; layer < state->first_layer_index + state->layer_count; layer++)
	{
		status = SparkMuseGlimmerModuleRunLayer(state,slot,views,layer,rows);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	if ( state->owns_final_head != 0u )
	{
		status = SparkMuseGlimmerModuleEmitHead(state,slot,frame,rows);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	wants_output = context != 0 && (context->flags & SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT) != 0u ? 1u : 0u;
	if ( wants_output != 0u )
	{
		status = SparkMuseGlimmerModuleEmitHiddenOutput(slot,context,rows);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	error = cudaMemcpyAsync(&error_record,state->kv_access_error,sizeof(error_record),cudaMemcpyDeviceToHost,stream);
	error = cudaStreamSynchronize(stream);
	if ( error != cudaSuccess )
		SPARK_FAIL(SparkStageModuleCudaStatus(SPARK_MUSE_GLIMMER_MODULE_TAG,error,"stream_sync"));
	if ( error_record.error_code != 0u )
	{
		fprintf(stderr,"%s kv_access_failed code=%u row=%u sequence=%u position=%u page=%u\n",SPARK_MUSE_GLIMMER_MODULE_TAG,error_record.error_code,error_record.row,error_record.sequence,error_record.position,error_record.page);
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	SparkMuseGlimmerModuleKvMarkWritten(state,slot,rows);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkMuseGlimmerModuleExecuteFrame(
	void *module_state,
	SparkModelDriverFrame *frame)
{
	SparkMuseGlimmerModuleState *state = (SparkMuseGlimmerModuleState *)module_state;
	SparkMuseGlimmerResidentDecodeStageFrameContext *context;
	SparkMuseGlimmerModuleSlot *slot;
	uint32_t rows,row;
	SparkStatus status;
	if ( (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	context = (SparkMuseGlimmerResidentDecodeStageFrameContext *)frame->user_context;
	status = SparkMuseGlimmerModuleValidateFrameContext(state,context);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	rows = frame->active_slot_count;
	if ( rows == 0u || rows > state->max_active_sequence_count )
	{
		atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	slot = &state->slots[0];
	if ( slot->cuda_stream == 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	for (row = 0; row < rows; row++)
	{
		slot->host_row_lane_indices[row] = row;
		slot->host_row_positions[row] = frame->sequence_position;
		slot->host_row_cold[row] = 0u;
		slot->host_slot_mapping[row] = (frame->sequence_position + row) % SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
		slot->host_context_lengths[row] = (uint32_t)((frame->sequence_position + row) % SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS) + 1u;
	}
	atomic_fetch_add_explicit(&state->submitted_count,1u,memory_order_relaxed);
	status = SparkMuseGlimmerModuleRunDecode(state,slot,frame,context,rows);
	if ( status != SPARK_STATUS_OK )
		SPARK_FAIL(status);
	if ( status == SPARK_STATUS_OK )
	{
		atomic_fetch_add_explicit(&state->completed_count,1u,memory_order_relaxed);
		atomic_fetch_add_explicit(&state->tokens_emitted,rows,memory_order_relaxed);
	}
	else
		atomic_fetch_add_explicit(&state->failed_count,1u,memory_order_relaxed);
	return(status);
}
