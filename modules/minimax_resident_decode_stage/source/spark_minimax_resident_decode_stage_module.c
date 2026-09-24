#define _FILE_OFFSET_BITS 64

#include <errno.h>
#include "sparkpipe/spark_error_site.h"
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <stddef.h>
#include <time.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_admission.h"
#include "sparkpipe/spark_hidden_transport.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "sparkpipe/spark_stage_module_lifecycle.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_minimax_resident_decode_stage_firmware.h"
#include "spark_minimax_stagepack_format.h"
#define SPARK_FAMILY_CAMEL Minimax
#define SPARK_FAMILY_UPPER MINIMAX
#define SPARK_FAMILY_LOWER minimax

#include "sparkpipe/family/spark_family.h"

#define SPARK_MINIMAX_MODULE_TAG "minimax_stage"

#define SPARK_MINIMAX_MODULE_TP_DEGREE_DEFAULT 4u
#define SPARK_MINIMAX_MODULE_TP_RANK_DEFAULT 0u
#define SPARK_MINIMAX_MODULE_TP_TIMEOUT_MILLI_DEFAULT 120000u
#define SPARK_MINIMAX_STAGEPACK_ENTRY_ONE(kind) (1ull << (kind))
#define SPARK_MINIMAX_STAGEPACK_GLOBAL_BITS \
	(SPARK_MINIMAX_STAGEPACK_ENTRY_ONE(SPARK_MINIMAX_STAGEPACK_TENSOR_EMBEDDING) | \
	 SPARK_MINIMAX_STAGEPACK_ENTRY_ONE(SPARK_MINIMAX_STAGEPACK_TENSOR_FINAL_NORM) | \
	 SPARK_MINIMAX_STAGEPACK_ENTRY_ONE(SPARK_MINIMAX_STAGEPACK_TENSOR_LM_HEAD))

typedef struct SparkMinimaxModuleSlot
{
	uint32_t logical_sequence_count;
	void *cuda_stream;
	uint32_t *host_row_lane_indices;
	uint64_t *host_row_positions;
	uint32_t *host_slot_mapping;
	uint32_t *host_context_lengths;
	uint32_t *input_token_ids;
	uint32_t *output_token_ids;
	uint32_t *row_lane_indices;
	uint32_t *slot_mapping;
	uint32_t *context_lengths;
	uint64_t *row_positions;
	uint32_t *argmax_token_u32;
	uint64_t *argmax_reduce_u64;
	void *hidden_bf16;
	void *normalized_bf16;
	void *query_bf16;
	void *key_bf16;
	void *value_bf16;
	void *query_roped_bf16;
	void *attended_bf16;
	void *attn_output_bf16;
	void *mlp_gate_bf16;
	void *mlp_up_bf16;
	void *mlp_down_bf16;
} SparkMinimaxModuleSlot;

typedef struct SparkMinimaxModuleState
{
	SparkStageModuleLedger ledger;
	uint32_t multiprocessor_count;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint32_t tp_standalone;
	uint32_t tp_collective_initialized;
	SparkTpDeviceCollective tp_device_collective;
	SparkTpDeviceCollectiveCreditBinding tp_credit_bindings[8u];
	uint32_t tp_credit_binding_count;
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
	atomic_uint slot_states[SPARK_MINIMAX_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	atomic_ullong submitted_count;
	atomic_ullong completed_count;
	atomic_ullong rejected_count;
	atomic_ullong failed_count;
	atomic_ullong tokens_emitted;
	SparkMinimaxModuleSlot slots[SPARK_MINIMAX_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	uint32_t stage_count;
	uint32_t stage_index;
	uint32_t allow_unqualified_execution;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t owns_embedding;
	uint32_t owns_final_head;
	uint32_t local_query_head_count;
	uint32_t local_kv_head_count;
	uint32_t local_kv_dimension;
	uint32_t local_ffn_dimension;
	uint32_t local_vocab_rows;
	uint32_t local_vocab_base;
	uint64_t kv_layer_stride;
	uint64_t kv_block_stride;
	uint32_t kv_block_count;
	const void *token_embedding_bf16;
	const void *final_norm_weight_bf16;
	const void *lm_head_weight_bf16;
	const void *input_norm_by_layer[SPARK_MINIMAX_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	const void *post_attention_norm_by_layer[SPARK_MINIMAX_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	SparkMinimaxAttentionLayerWeights attention_by_layer[SPARK_MINIMAX_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	SparkMinimaxMlpLayerWeights mlp_by_layer[SPARK_MINIMAX_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	void *kv_cache_bf16;
	SparkMinimaxKvBlockTableView block_table;
	uint32_t *host_block_indices;
	uint32_t *device_block_indices;
	uint32_t *device_block_counts;
	uint32_t *free_blocks;
	uint32_t free_block_count;
	uint32_t *lane_block_counts;
	uint64_t *lane_context_tokens;
	uint32_t blocks_per_lane;
	uint64_t reset_generation;
} SparkMinimaxModuleState;

extern cudaError_t SparkMinimaxConfigureCudaKernels(void);
extern cudaError_t SparkMinimaxLaunchEmbeddingGather(cudaStream_t stream,const uint32_t *token_ids,const void *embedding_bf16,void *hidden_bf16,uint32_t row_count);
extern cudaError_t SparkMinimaxLaunchRmsNorm(cudaStream_t stream,const void *input_bf16,const void *gain_bf16,void *output_bf16,uint32_t row_count,uint32_t dimension,float epsilon);
extern cudaError_t SparkMinimaxLaunchKvCacheWrite(cudaStream_t stream,const void *key_bf16,const void *value_bf16,void *kv_cache_bf16,const uint32_t *physical_block_indices,const uint32_t *lane_block_counts,const uint32_t *row_lane_indices,const uint32_t *slot_mapping,uint32_t row_count,uint32_t local_kv_head_dimension,uint32_t lane_stride,uint64_t block_span,uint64_t layer_block_stride,uint32_t layer_index);
extern cudaError_t SparkMinimaxLaunchArgmaxResolve(cudaStream_t stream,const uint64_t *argmax_reduce_u64,uint32_t *token_ids,uint32_t row_count);
extern cudaError_t SparkMinimaxLaunchResidualAdd(cudaStream_t stream,void *hidden_bf16,const void *delta_bf16,uint32_t row_count,uint32_t dimension);
extern cudaError_t SparkMinimaxLaunchQueryKeyProjection(cudaStream_t stream,const SparkMinimaxLinearView *query,const SparkMinimaxLinearView *key,const void *input_bf16,void *query_bf16,void *key_bf16,uint32_t row_count);
extern cudaError_t SparkMinimaxLaunchValueProjection(cudaStream_t stream,const SparkMinimaxLinearView *value,const void *input_bf16,void *value_bf16,uint32_t row_count);
extern cudaError_t SparkMinimaxLaunchHeadNormRope(cudaStream_t stream,void *query_bf16,void *key_bf16,const void *query_norm_bf16,const void *key_norm_bf16,void *query_roped_bf16,const uint64_t *row_positions,uint32_t row_count,float epsilon,uint32_t local_query_head_count,uint32_t local_kv_head_count);
extern cudaError_t SparkMinimaxLaunchAttentionDecode(cudaStream_t stream,const void *query_roped_f32,void *kv_cache_bf16,const SparkMinimaxKvBlockTableView *table,const uint32_t *row_lane_indices,const uint32_t *slot_mapping,const uint32_t *context_lengths,void *attended_bf16,uint32_t row_count,uint32_t local_query_head_count,uint32_t local_kv_head_count,uint32_t local_kv_head_dimension,uint64_t block_span,uint64_t layer_block_stride,float epsilon,uint32_t tp_rank,uint32_t layer_index);
extern cudaError_t SparkMinimaxLaunchAttentionPrefill(cudaStream_t stream,const void *query_roped_f32,void *kv_cache_bf16,const SparkMinimaxKvBlockTableView *table,const uint32_t *row_lane_indices,const uint64_t *row_positions,const void *staged_key_bf16,const void *staged_value_bf16,void *attended_bf16,uint32_t row_count,uint32_t local_query_head_count,uint32_t local_kv_head_count,uint32_t local_kv_head_dimension,uint32_t lane_stride,uint64_t block_span,uint64_t layer_block_stride,float epsilon,uint32_t tp_rank,uint32_t layer_index,uint64_t base_position);
extern cudaError_t SparkMinimaxLaunchProjection(cudaStream_t stream,const SparkMinimaxLinearView *view,const void *input_bf16,void *output_bf16,uint32_t row_count);
extern cudaError_t SparkMinimaxLaunchSwiGlu(cudaStream_t stream,const void *gate_bf16,const void *up_bf16,void *activated_bf16,uint32_t row_count,uint32_t dimension);
extern cudaError_t SparkMinimaxLaunchVocabArgmax(cudaStream_t stream,const void *lm_head_bf16,const void *input_bf16,uint64_t *argmax_reduce_u64,uint32_t row_count,uint32_t local_vocab_rows,uint32_t input_dimension,uint32_t local_vocab_base);
extern cudaError_t SparkMinimaxLaunchTpCombineBf16(cudaStream_t stream,void *destination_device,const void *source_device,uint32_t element_count);
extern cudaError_t SparkMinimaxLaunchTpCombineU64Max(cudaStream_t stream,void *destination_device,const void *source_device,uint32_t element_count);
static SparkStatus SparkMinimaxModuleTpCombineBf16(void *combine_context,void *destination_device,const void *source_device,uint32_t active_sequence_count,uint32_t hidden_dimension,void *cuda_stream);
static SparkStatus SparkMinimaxModuleTpCombineU64Max(void *combine_context,uint64_t *destination_device,const uint64_t *source_device,uint32_t count,void *cuda_stream);

static void SparkMinimaxModuleFillLinearView(SparkMinimaxLinearView *view,const SparkMinimaxStagePackEntry *entry,void *payload)
{
	view->abi_version = SPARK_MINIMAX_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION;
	view->weight_format = entry->weight_format;
	view->input_dimension = entry->columns;
	view->output_dimension = entry->rows;
	view->row_base = 0u;
	view->weight_payload_bf16 = payload;
	view->weight_payload_bytes = entry->payload_bytes;
}

static SparkStatus SparkMinimaxModuleConfigure(SparkMinimaxModuleState *state)
{
	SparkStatus status;
	status = SparkStageModuleEnvironmentUnsignedOrDefault(SPARK_MINIMAX_MODULE_TAG,"SPARK_MINIMAX_TP_DEGREE",1u,SPARK_MINIMAX_RESIDENT_DECODE_STAGE_KV_HEAD_COUNT,SPARK_MINIMAX_MODULE_TP_DEGREE_DEFAULT,&state->tp_degree);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsignedOrDefault(SPARK_MINIMAX_MODULE_TAG,"SPARK_MINIMAX_TP_RANK",0u,SPARK_MINIMAX_RESIDENT_DECODE_STAGE_KV_HEAD_COUNT - 1u,SPARK_MINIMAX_MODULE_TP_RANK_DEFAULT,&state->tp_rank);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( state->tp_rank >= state->tp_degree )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (SPARK_MINIMAX_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT % state->tp_degree) != 0u ||
		(SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HEAD_COUNT % state->tp_degree) != 0u ||
		(SPARK_MINIMAX_RESIDENT_DECODE_STAGE_KV_HEAD_COUNT % state->tp_degree) != 0u ||
		(SPARK_MINIMAX_RESIDENT_DECODE_STAGE_FFN_INTERMEDIATE_DIMENSION % state->tp_degree) != 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkStageModuleEnvironmentUnsignedOrDefault(SPARK_MINIMAX_MODULE_TAG,"SPARK_MINIMAX_TP_STANDALONE",0u,1u,0u,&state->tp_standalone);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	state->local_query_head_count = SPARK_LLM_LOCAL_QUERY_HEAD_COUNT(state->tp_degree);
	state->local_kv_head_count = SPARK_LLM_LOCAL_KV_HEAD_COUNT(state->tp_degree);
	state->local_kv_dimension = SPARK_LLM_LOCAL_KV_DIMENSION(state->tp_degree);
	state->local_ffn_dimension = SPARK_LLM_LOCAL_FFN_DIMENSION(state->tp_degree);
	state->local_vocab_rows = SPARK_LLM_LOCAL_VOCAB_ROWS(state->tp_degree);
	state->local_vocab_base = state->tp_rank * state->local_vocab_rows;
	state->tp_collective_identifier = 0u;
	state->tp_control_port_base = 0u;
	state->tp_connect_timeout_milli = SPARK_MINIMAX_MODULE_TP_TIMEOUT_MILLI_DEFAULT;
	state->tp_operation_timeout_milli = SPARK_MINIMAX_MODULE_TP_TIMEOUT_MILLI_DEFAULT;
	state->tp_backend_path[0] = '\0';
	state->tp_local_host[0] = '\0';
	for (uint32_t host_clear = 0u; host_clear < SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE; host_clear++)
		state->tp_hosts[host_clear][0] = '\0';
	if ( state->tp_degree > 1u && state->tp_standalone == 0u )
	{
		const char *tp_backend;
		const char *tp_hosts;
		const char *tp_local_host;
		uint64_t tp_identifier;
		const char *scan;
		uint32_t host_index;
		status = SparkStageModuleEnvironmentText(SPARK_MINIMAX_MODULE_TAG,"SPARK_MINIMAX_STAGE_TP_BACKEND_PATH",&tp_backend);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleEnvironmentUnsigned64(SPARK_MINIMAX_MODULE_TAG,"SPARK_MINIMAX_STAGE_TP_IDENTIFIER",0u,UINT64_MAX,&tp_identifier);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleEnvironmentUnsigned(SPARK_MINIMAX_MODULE_TAG,"SPARK_MINIMAX_STAGE_TP_PORT_BASE",1u,65535u,&state->tp_control_port_base);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleEnvironmentText(SPARK_MINIMAX_MODULE_TAG,"SPARK_MINIMAX_STAGE_TP_HOSTS",&tp_hosts);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleEnvironmentText(SPARK_MINIMAX_MODULE_TAG,"SPARK_MINIMAX_STAGE_TP_LOCAL_HOST",&tp_local_host);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleEnvironmentUnsignedOrDefault(SPARK_MINIMAX_MODULE_TAG,"SPARK_MINIMAX_STAGE_TP_TIMEOUT_MS",1u,UINT32_MAX,SPARK_MINIMAX_MODULE_TP_TIMEOUT_MILLI_DEFAULT,&state->tp_connect_timeout_milli);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
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
				SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
			memcpy(state->tp_hosts[host_index],scan,length);
			state->tp_hosts[host_index][length] = '\0';
			host_index++;
			scan = comma != 0 ? comma + 1 : scan + length;
		}
		if ( host_index != state->tp_degree )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		{
			const char *tp_session_ports;
			const char *cell_scan;
			uint32_t row_index,column_index,parsed_count;
			unsigned long cell_value;
			status = SparkStageModuleEnvironmentText(SPARK_MINIMAX_MODULE_TAG,"SPARK_MINIMAX_STAGE_TP_SESSION_PORTS",&tp_session_ports);
			if ( status != SPARK_STATUS_OK )
				SPARK_RETURN(status);
			cell_scan = tp_session_ports;
			parsed_count = 0u;
			for (row_index = 0u; row_index < state->tp_degree; row_index++)
				for (column_index = 0u; column_index < state->tp_degree; column_index++)
				{
					char *cell_end;
					errno = 0;
					cell_value = strtoul(cell_scan,&cell_end,10);
					if ( cell_end == cell_scan || errno != 0 ||
						cell_value > 65535u ||
						(row_index == column_index ? cell_value != 0u : cell_value == 0u) )
						SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
					state->tp_session_ports[row_index][column_index] = (uint16_t)cell_value;
					parsed_count++;
					cell_scan = cell_end;
					while ( *cell_scan == ',' )
						cell_scan++;
				}
			if ( parsed_count != state->tp_degree * state->tp_degree )
				SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		}
	}
	status = SparkStageModuleEnvironmentUnsigned(SPARK_MINIMAX_MODULE_TAG,"SPARK_MINIMAX_STAGE_COUNT",1u,SPARK_LLM_MAX_STAGE_COUNT,&state->stage_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_MINIMAX_MODULE_TAG,"SPARK_MINIMAX_STAGE_INDEX",0u,SPARK_LLM_MAX_STAGE_COUNT - 1u,&state->stage_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_MINIMAX_MODULE_TAG,"SPARK_MINIMAX_STAGE_MAX_ACTIVE_SEQUENCES",1u,SPARK_MINIMAX_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,&state->max_active_sequence_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_MINIMAX_MODULE_TAG,"SPARK_MINIMAX_STAGE_PIPELINE_SLOTS",1u,SPARK_MINIMAX_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,&state->pipeline_slot_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_MINIMAX_MODULE_TAG,"SPARK_MINIMAX_STAGE_KV_BLOCKS",1u,1u << 20u,&state->kv_block_count);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( state->stage_index != 0u || state->stage_count != 1u )
	{
		fprintf(stderr,"%s config_stage_topology stage=%u/%u (the text-tower driver serves as one TP-sharded stage)\n",SPARK_MINIMAX_MODULE_TAG,state->stage_index,state->stage_count);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	state->first_layer_index = 0u;
	state->layer_count = SPARK_MINIMAX_RESIDENT_DECODE_STAGE_LAYER_COUNT;
	state->owns_embedding = 1u;
	state->owns_final_head = 1u;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkMinimaxModuleValidateEntry(SparkMinimaxModuleState *state,const SparkMinimaxStagePackEntry *entry,uint64_t file_bytes,uint32_t *is_global)
{
	SparkMinimaxStagePackTensorShape shape;
	uint64_t payload_bytes;
	uint32_t global = entry->layer_index == SPARK_MINIMAX_STAGEPACK_GLOBAL_LAYER ? 1u : 0u;
	(void)state;
	if ( entry->tensor_kind >= SPARK_MINIMAX_STAGEPACK_TENSOR_KIND_COUNT )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	if ( SparkMinimaxStagePackKindIsGlobal(entry->tensor_kind) != 0u )
	{
		if ( global == 0u )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	}
	else
	{
		if ( global != 0u || entry->layer_index >= SPARK_MINIMAX_RESIDENT_DECODE_STAGE_LAYER_COUNT )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	}
	SparkMinimaxStagePackNarrowShape(entry->tensor_kind,state->tp_degree,state->tp_rank,&shape);
	if ( entry->rows != shape.rows || entry->columns != shape.columns )
	{
		fprintf(stderr,"%s pack_entry_shape kind=%u layer=%u rows=%u/%u cols=%u/%u\n",SPARK_MINIMAX_MODULE_TAG,entry->tensor_kind,entry->layer_index,entry->rows,shape.rows,entry->columns,shape.columns);
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	}
	if ( entry->weight_format != SPARK_MINIMAX_STAGEPACK_WEIGHT_BF16 || entry->scale_group_size != 0u ||
		entry->scale_offset != 0u || entry->scale_bytes != 0u )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	payload_bytes = SparkMinimaxStagePackPayloadBytes(entry->rows,entry->columns);
	if ( entry->payload_bytes != payload_bytes )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->payload_offset > file_bytes || entry->payload_bytes > file_bytes - entry->payload_offset )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	if ( (entry->payload_offset % SPARK_MINIMAX_STAGEPACK_PAYLOAD_ALIGNMENT) != 0u )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	*is_global = global;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkMinimaxModuleBindGlobal(SparkMinimaxModuleState *state,const SparkMinimaxStagePackEntry *entry,void *payload)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_MINIMAX_STAGEPACK_TENSOR_EMBEDDING: state->token_embedding_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_STAGEPACK_TENSOR_FINAL_NORM: state->final_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_STAGEPACK_TENSOR_LM_HEAD: state->lm_head_weight_bf16 = payload; return(SPARK_STATUS_OK);
	default:
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	}
}

static SparkStatus SparkMinimaxModuleBindLayer(SparkMinimaxModuleState *state,const SparkMinimaxStagePackEntry *entry,void *payload)
{
	SparkMinimaxAttentionLayerWeights *attention = &state->attention_by_layer[entry->layer_index];
	SparkMinimaxMlpLayerWeights *mlp = &state->mlp_by_layer[entry->layer_index];
	switch ( entry->tensor_kind )
	{
	case SPARK_MINIMAX_STAGEPACK_TENSOR_INPUT_NORM: state->input_norm_by_layer[entry->layer_index] = payload; return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_STAGEPACK_TENSOR_POST_ATTENTION_NORM: state->post_attention_norm_by_layer[entry->layer_index] = payload; return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_STAGEPACK_TENSOR_Q_NORM: attention->query_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_STAGEPACK_TENSOR_K_NORM: attention->key_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_STAGEPACK_TENSOR_QUERY: SparkMinimaxModuleFillLinearView(&attention->query,entry,payload); return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_STAGEPACK_TENSOR_KEY: SparkMinimaxModuleFillLinearView(&attention->key,entry,payload); return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_STAGEPACK_TENSOR_VALUE: SparkMinimaxModuleFillLinearView(&attention->value,entry,payload); return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_STAGEPACK_TENSOR_OUTPUT: SparkMinimaxModuleFillLinearView(&attention->output,entry,payload); return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_STAGEPACK_TENSOR_FFN_GATE: SparkMinimaxModuleFillLinearView(&mlp->gate,entry,payload); return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_STAGEPACK_TENSOR_FFN_UP: SparkMinimaxModuleFillLinearView(&mlp->up,entry,payload); return(SPARK_STATUS_OK);
	case SPARK_MINIMAX_STAGEPACK_TENSOR_FFN_DOWN: SparkMinimaxModuleFillLinearView(&mlp->down,entry,payload); return(SPARK_STATUS_OK);
	default:
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	}
}

static void SparkMinimaxModuleExpectedGeometry(SparkMinimaxModuleState *state,SparkMinimaxStagePackHeader *header)
{
	memset(header,0,sizeof(*header));
	header->magic = SPARK_MINIMAX_STAGEPACK_MAGIC;
	header->format_version = SPARK_MINIMAX_STAGEPACK_FORMAT_VERSION;
	header->header_bytes = SPARK_MINIMAX_STAGEPACK_HEADER_BYTES;
	header->directory_entry_bytes = SPARK_MINIMAX_STAGEPACK_ENTRY_BYTES;
	header->tensor_count = SparkMinimaxStagePackExpectedTensorCount(SPARK_MINIMAX_RESIDENT_DECODE_STAGE_LAYER_COUNT);
	header->hidden_dimension = SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
	header->layer_count = SPARK_MINIMAX_RESIDENT_DECODE_STAGE_LAYER_COUNT;
	header->first_layer_index = 0u;
	header->total_layer_count = SPARK_MINIMAX_RESIDENT_DECODE_STAGE_LAYER_COUNT;
	header->query_head_count = SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HEAD_COUNT;
	header->kv_head_count = SPARK_MINIMAX_RESIDENT_DECODE_STAGE_KV_HEAD_COUNT;
	header->head_dimension = SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HEAD_DIMENSION;
	header->ffn_intermediate_dimension = SPARK_MINIMAX_RESIDENT_DECODE_STAGE_FFN_INTERMEDIATE_DIMENSION;
	header->output_vocab_count = SPARK_MINIMAX_RESIDENT_DECODE_STAGE_OUTPUT_VOCAB_COUNT;
	header->mtp_layer_count = 0u;
	header->tp_degree = state->tp_degree;
	header->tp_rank = state->tp_rank;
	header->directory_offset = SPARK_MINIMAX_STAGEPACK_HEADER_BYTES;
}

SparkStatus SparkMinimaxModuleLoadMappedPack(SparkMinimaxModuleState *state,const char *path,uint32_t resident,uint32_t *mapped_flag)
{
	SparkMinimaxStagePackHeader header,expected;
	SparkMinimaxStagePackEntry *directory;
	uint64_t *seen_layers;
	uint64_t seen_globals = 0u;
	uint64_t file_bytes;
	struct stat pack_info;
	FILE *file;
	void *mapping;
	SparkStatus status;
	uint32_t index;
	file = fopen(path,"rb");
	if ( file == 0 )
	{
		fprintf(stderr,"%s pack_open_failed path=%s\n",SPARK_MINIMAX_MODULE_TAG,path);
		return(SPARK_STATUS_IO_ERROR);
	}
	if ( fstat(fileno(file),&pack_info) != 0 )
	{
		fclose(file);
		return(SPARK_STATUS_IO_ERROR);
	}
	file_bytes = (uint64_t)pack_info.st_size;
	status = SparkStageModulePackRead(SPARK_MINIMAX_MODULE_TAG,file,0u,&header,sizeof(header));
	if ( status != SPARK_STATUS_OK )
	{
		fclose(file);
		return(status);
	}
	SparkMinimaxModuleExpectedGeometry(state,&expected);
	if ( memcmp(&header,&expected,offsetof(SparkMinimaxStagePackHeader,file_bytes)) != 0 ||
		header.directory_offset != expected.directory_offset ||
		header.file_bytes != file_bytes )
	{
		fprintf(stderr,"%s pack_geometry_mismatch path=%s magic=%u/%u version=%u/%u tensors=%u/%u hidden=%u/%u layers=%u/%u heads=%u/%u kv=%u/%u dim=%u/%u ffn=%u/%u vocab=%u/%u mtp=%u/%u tp=%u/%u rank=%u/%u dir=%llu bytes=%llu/%llu\n",
			SPARK_MINIMAX_MODULE_TAG,path,
			header.magic,expected.magic,
			header.format_version,expected.format_version,
			header.tensor_count,expected.tensor_count,
			header.hidden_dimension,expected.hidden_dimension,
			header.layer_count,expected.layer_count,
			header.query_head_count,expected.query_head_count,
			header.kv_head_count,expected.kv_head_count,
			header.head_dimension,expected.head_dimension,
			header.ffn_intermediate_dimension,expected.ffn_intermediate_dimension,
			header.output_vocab_count,expected.output_vocab_count,
			header.mtp_layer_count,expected.mtp_layer_count,
			header.tp_degree,expected.tp_degree,
			header.tp_rank,expected.tp_rank,
			(unsigned long long)header.directory_offset,
			(unsigned long long)header.file_bytes,(unsigned long long)file_bytes);
		fclose(file);
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	}
	directory = (SparkMinimaxStagePackEntry *)malloc((size_t)header.tensor_count * sizeof(*directory));
	seen_layers = (uint64_t *)calloc(SPARK_MINIMAX_RESIDENT_DECODE_STAGE_LAYER_COUNT,sizeof(*seen_layers));
	if ( directory == 0 || seen_layers == 0 )
	{
		free(directory);
		free(seen_layers);
		fclose(file);
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	status = SparkStageModulePackRead(SPARK_MINIMAX_MODULE_TAG,file,header.directory_offset,directory,(uint64_t)header.tensor_count * sizeof(*directory));
	for (index = 0; status == SPARK_STATUS_OK && index < header.tensor_count; index++)
	{
		const SparkMinimaxStagePackEntry *entry = &directory[index];
		uint64_t bit;
		uint32_t is_global = 0u;
		status = SparkMinimaxModuleValidateEntry(state,entry,file_bytes,&is_global);
		if ( status != SPARK_STATUS_OK )
		{
			fprintf(stderr,"%s pack_entry_invalid index=%u kind=%u layer=%u\n",SPARK_MINIMAX_MODULE_TAG,index,entry->tensor_kind,entry->layer_index);
			break;
		}
		bit = SPARK_MINIMAX_STAGEPACK_ENTRY_ONE(entry->tensor_kind);
		if ( is_global != 0u )
		{
			if ( (seen_globals & bit) != 0u )
			{
				fprintf(stderr,"%s pack_entry_duplicate kind=%u\n",SPARK_MINIMAX_MODULE_TAG,entry->tensor_kind);
				status = SPARK_STATUS_VALIDATION_FAILED;
				break;
			}
			seen_globals |= bit;
		}
		else
		{
			if ( (seen_layers[entry->layer_index] & bit) != 0u )
			{
				fprintf(stderr,"%s pack_entry_duplicate kind=%u layer=%u\n",SPARK_MINIMAX_MODULE_TAG,entry->tensor_kind,entry->layer_index);
				status = SPARK_STATUS_VALIDATION_FAILED;
				break;
			}
			seen_layers[entry->layer_index] |= bit;
		}
	}
	if ( status == SPARK_STATUS_OK )
	{
		uint64_t expected_layer_bits = 0u;
		for (uint32_t kind = SPARK_MINIMAX_STAGEPACK_TENSOR_INPUT_NORM; kind < SPARK_MINIMAX_STAGEPACK_TENSOR_KIND_COUNT; kind++)
			expected_layer_bits |= SPARK_MINIMAX_STAGEPACK_ENTRY_ONE(kind);
		for (uint32_t layer = 0u; layer < SPARK_MINIMAX_RESIDENT_DECODE_STAGE_LAYER_COUNT; layer++)
		{
			if ( seen_layers[layer] != expected_layer_bits )
			{
				fprintf(stderr,"%s pack_layer_incomplete layer=%u seen=%016llx expected=%016llx\n",SPARK_MINIMAX_MODULE_TAG,layer,(unsigned long long)seen_layers[layer],(unsigned long long)expected_layer_bits);
				status = SPARK_STATUS_VALIDATION_FAILED;
				break;
			}
		}
	}
	if ( status == SPARK_STATUS_OK && seen_globals != SPARK_MINIMAX_STAGEPACK_GLOBAL_BITS )
	{
		fprintf(stderr,"%s pack_globals_incomplete seen=%016llx expected=%016llx\n",SPARK_MINIMAX_MODULE_TAG,(unsigned long long)seen_globals,(unsigned long long)SPARK_MINIMAX_STAGEPACK_GLOBAL_BITS);
		status = SPARK_STATUS_VALIDATION_FAILED;
	}
	mapping = 0;
	if ( status == SPARK_STATUS_OK )
	{
		mapping = mmap(0,(size_t)file_bytes,PROT_READ,MAP_PRIVATE,fileno(file),0);
		if ( mapping == MAP_FAILED )
		{
			fprintf(stderr,"%s pack_map_failed path=%s\n",SPARK_MINIMAX_MODULE_TAG,path);
			status = SPARK_STATUS_IO_ERROR;
		}
	}
	for (index = 0; status == SPARK_STATUS_OK && index < header.tensor_count; index++)
	{
		const SparkMinimaxStagePackEntry *entry = &directory[index];
		void *payload = (uint8_t *)mapping + entry->payload_offset;
		if ( entry->layer_index == SPARK_MINIMAX_STAGEPACK_GLOBAL_LAYER )
			status = SparkMinimaxModuleBindGlobal(state,entry,payload);
		else
			status = SparkMinimaxModuleBindLayer(state,entry,payload);
	}
	if ( status != SPARK_STATUS_OK )
	{
		if ( mapping != 0 && mapping != MAP_FAILED )
			munmap(mapping,(size_t)file_bytes);
	}
	else if ( resident != 0u )
	{
		for (index = 0; status == SPARK_STATUS_OK && index < header.tensor_count; index++)
		{
			const SparkMinimaxStagePackEntry *entry = &directory[index];
			void *device_target = 0;
			void *host_source = (uint8_t *)mapping + entry->payload_offset;
			status = SparkStageModuleDeviceAllocate(&state->ledger,entry->payload_bytes,&device_target);
			if ( status != SPARK_STATUS_OK )
				break;
			status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,cudaMemcpy(device_target,host_source,(size_t)entry->payload_bytes,cudaMemcpyHostToDevice),"pack_upload");
			if ( status != SPARK_STATUS_OK )
				break;
			if ( entry->layer_index == SPARK_MINIMAX_STAGEPACK_GLOBAL_LAYER )
				status = SparkMinimaxModuleBindGlobal(state,entry,device_target);
			else
				status = SparkMinimaxModuleBindLayer(state,entry,device_target);
		}
		munmap(mapping,(size_t)file_bytes);
	}
	else
		*mapped_flag = 1u;
	free(directory);
	free(seen_layers);
	fclose(file);
	return(status);
}

void *SparkMinimaxModuleStateAllocate(void)
{
	return(calloc(1u,sizeof(SparkMinimaxModuleState)));
}

void SparkMinimaxModuleStateFree(void *state_pointer)
{
	free(state_pointer);
}

void SparkMinimaxModuleStateConfigureTp(void *state_pointer,uint32_t tp_degree,uint32_t tp_rank)
{
	SparkMinimaxModuleState *state = (SparkMinimaxModuleState *)state_pointer;
	state->tp_degree = tp_degree;
	state->tp_rank = tp_rank;
	state->local_query_head_count = SPARK_LLM_LOCAL_QUERY_HEAD_COUNT(tp_degree);
	state->local_kv_head_count = SPARK_LLM_LOCAL_KV_HEAD_COUNT(tp_degree);
	state->local_kv_dimension = SPARK_LLM_LOCAL_KV_DIMENSION(tp_degree);
	state->local_ffn_dimension = SPARK_LLM_LOCAL_FFN_DIMENSION(tp_degree);
	state->local_vocab_rows = SPARK_LLM_LOCAL_VOCAB_ROWS(tp_degree);
	state->local_vocab_base = tp_rank * state->local_vocab_rows;
	state->first_layer_index = 0u;
	state->layer_count = SPARK_MINIMAX_RESIDENT_DECODE_STAGE_LAYER_COUNT;
	state->owns_embedding = 1u;
	state->owns_final_head = 1u;
}

SparkStatus SparkMinimaxModuleValidationView(const SparkMinimaxModuleState *state,uint32_t kind,uint32_t layer,const void **payload,uint32_t *rows,uint32_t *columns)
{
	SparkMinimaxStagePackTensorShape shape;
	const void *source = 0;
	if ( state == 0 || payload == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	SparkMinimaxStagePackFullShape(kind,&shape);
	switch ( kind )
	{
	case SPARK_MINIMAX_STAGEPACK_TENSOR_EMBEDDING: source = state->token_embedding_bf16; break;
	case SPARK_MINIMAX_STAGEPACK_TENSOR_FINAL_NORM: source = state->final_norm_weight_bf16; break;
	case SPARK_MINIMAX_STAGEPACK_TENSOR_LM_HEAD: source = state->lm_head_weight_bf16; shape.rows = state->local_vocab_rows; break;
	case SPARK_MINIMAX_STAGEPACK_TENSOR_INPUT_NORM: source = state->input_norm_by_layer[layer]; break;
	case SPARK_MINIMAX_STAGEPACK_TENSOR_POST_ATTENTION_NORM: source = state->post_attention_norm_by_layer[layer]; break;
	case SPARK_MINIMAX_STAGEPACK_TENSOR_Q_NORM: source = state->attention_by_layer[layer].query_norm_weight_bf16; break;
	case SPARK_MINIMAX_STAGEPACK_TENSOR_K_NORM: source = state->attention_by_layer[layer].key_norm_weight_bf16; break;
	case SPARK_MINIMAX_STAGEPACK_TENSOR_QUERY: source = state->attention_by_layer[layer].query.weight_payload_bf16; break;
	case SPARK_MINIMAX_STAGEPACK_TENSOR_KEY: source = state->attention_by_layer[layer].key.weight_payload_bf16; break;
	case SPARK_MINIMAX_STAGEPACK_TENSOR_VALUE: source = state->attention_by_layer[layer].value.weight_payload_bf16; break;
	case SPARK_MINIMAX_STAGEPACK_TENSOR_OUTPUT: source = state->attention_by_layer[layer].output.weight_payload_bf16; break;
	case SPARK_MINIMAX_STAGEPACK_TENSOR_FFN_GATE: source = state->mlp_by_layer[layer].gate.weight_payload_bf16; break;
	case SPARK_MINIMAX_STAGEPACK_TENSOR_FFN_UP: source = state->mlp_by_layer[layer].up.weight_payload_bf16; break;
	case SPARK_MINIMAX_STAGEPACK_TENSOR_FFN_DOWN: source = state->mlp_by_layer[layer].down.weight_payload_bf16; break;
	default: return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( kind == SPARK_MINIMAX_STAGEPACK_TENSOR_QUERY || kind == SPARK_MINIMAX_STAGEPACK_TENSOR_KEY || kind == SPARK_MINIMAX_STAGEPACK_TENSOR_VALUE )
		shape.rows /= 1u;
	if ( source == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	SparkMinimaxStagePackNarrowShape(kind,state->tp_degree,state->tp_rank,&shape);
	if ( kind == SPARK_MINIMAX_STAGEPACK_TENSOR_LM_HEAD )
		shape.rows = state->local_vocab_rows;
	*payload = source;
	if ( rows != 0 )
		*rows = shape.rows;
	if ( columns != 0 )
		*columns = shape.columns;
	return(SPARK_STATUS_OK);
}

static void SparkMinimaxModuleDescribe(void *module_state,SparkStageModuleLifecycle *lifecycle)
{
	SparkMinimaxModuleState *state = (SparkMinimaxModuleState *)module_state;
	lifecycle->module_tag = SPARK_MINIMAX_MODULE_TAG;
	lifecycle->ledger = &state->ledger;
	lifecycle->slot_states = state->slot_states;
	lifecycle->pipeline_slot_count = state->pipeline_slot_count;
	lifecycle->submitted_count = &state->submitted_count;
	lifecycle->completed_count = &state->completed_count;
	lifecycle->rejected_count = &state->rejected_count;
	lifecycle->failed_count = &state->failed_count;
	lifecycle->tokens_emitted = &state->tokens_emitted;
}

#include "sparkpipe/family/module/spark_module_tp_submit_ordered.h"

static SparkStatus SparkMinimaxModuleInitializeTpCollective(SparkMinimaxModuleState *state)
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
	if ( state->tp_standalone != 0u )
	{
		fprintf(stderr,"%s tp_standalone degree=%u rank=%u (collective skipped; hidden and logits stay rank-partial)\n",
			SPARK_MINIMAX_MODULE_TAG,state->tp_degree,state->tp_rank);
		return(SPARK_STATUS_OK);
	}
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
	configuration.local_hidden_dimension = SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
	configuration.max_active_sequence_count = SPARK_MINIMAX_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT;
	configuration.connect_timeout_milli = state->tp_connect_timeout_milli;
	configuration.operation_timeout_milli = state->tp_operation_timeout_milli;
	configuration.control_port_base = state->tp_control_port_base;
	configuration.collective_identifier = state->tp_collective_identifier;
	configuration.backend_module_path = state->tp_backend_path;
	configuration.local_host = state->tp_local_host;
	configuration.registration_cuda_stream = state->slots[0].cuda_stream;
	configuration.combine_bf16_function = SparkMinimaxModuleTpCombineBf16;
	configuration.combine_u64_max_function = SparkMinimaxModuleTpCombineU64Max;
	configuration.combine_context = state;
	status = SparkTpDeviceCollectiveApplyTopology(&topology,&configuration);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s tp_apply_topology_failed status=%d\n",SPARK_MINIMAX_MODULE_TAG,(int)status);
		SPARK_RETURN(status);
	}
	status = SparkTpDeviceCollectiveCreditBindingRouteCount(&configuration,&route_count);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = SparkTpDeviceCollectiveProbeMemoryMode(
		configuration.backend_kind,configuration.backend_module_path,
		&memory_mode);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s tp_probe_memory_mode_failed status=%d\n",SPARK_MINIMAX_MODULE_TAG,(int)status);
		SPARK_RETURN(status);
	}
	credit_bytes = SparkTpDeviceCollectiveCreditBytes(configuration.max_active_sequence_count,configuration.local_hidden_dimension);
	total_bytes = credit_bytes * configuration.credit_count * route_count;
	status = SparkStageModuleDeviceAllocate(&state->ledger,total_bytes,&state->tp_collective_credit_send_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,total_bytes,&state->tp_collective_credit_receive_bf16);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
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
			return(SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,error,"tp_credit_mapped_alloc"));
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
		fprintf(stderr,"%s tp_create_failed status=%d\n",SPARK_MINIMAX_MODULE_TAG,(int)status);
		SPARK_RETURN(status);
	}
	state->tp_collective_initialized = 1u;
	status = SparkTpDeviceCollectiveAttachMesh(&state->tp_device_collective);
	if ( status != SPARK_STATUS_OK )
		return(status);
	fprintf(stderr,"%s tp_collective_open degree=%u rank=%u port_base=%u\n",SPARK_MINIMAX_MODULE_TAG,state->tp_degree,state->tp_rank,state->tp_control_port_base);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkMinimaxModuleTpCombineBf16(void *combine_context,void *destination_device,const void *source_device,uint32_t active_sequence_count,uint32_t hidden_dimension,void *cuda_stream)
{
	(void)combine_context;
	(void)hidden_dimension;
	return(SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchTpCombineBf16((cudaStream_t)cuda_stream,destination_device,source_device,active_sequence_count * SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION),"tp_combine_bf16"));
}

static SparkStatus SparkMinimaxModuleTpReduceArgmax(SparkMinimaxModuleState *state,SparkMinimaxModuleSlot *slot,uint64_t *device_u64,uint32_t count)
{
	if ( state->tp_degree == 1u || state->tp_standalone != 0u )
		return(SPARK_STATUS_OK);
	return(SparkMinimaxModuleTpSubmitOrdered(state,device_u64,count,slot,1u));
}

static SparkStatus SparkMinimaxModuleAllocateSlot(SparkMinimaxModuleState *state,SparkMinimaxModuleSlot *slot)
{
	uint64_t rows = state->max_active_sequence_count;
	uint64_t hidden_bytes = rows * SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HIDDEN_BF16_BYTES;
	uint64_t query_bytes = rows * state->local_query_head_count * SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HEAD_DIMENSION * SPARK_MINIMAX_RESIDENT_DECODE_STAGE_BF16_ELEMENT_BYTES;
	uint64_t query_roped_bytes = rows * state->local_query_head_count * SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HEAD_DIMENSION * sizeof(float);
	uint64_t kv_bytes = rows * state->local_kv_dimension * SPARK_MINIMAX_RESIDENT_DECODE_STAGE_BF16_ELEMENT_BYTES;
	uint64_t ffn_bytes = rows * state->local_ffn_dimension * SPARK_MINIMAX_RESIDENT_DECODE_STAGE_BF16_ELEMENT_BYTES;
	cudaStream_t stream = 0;
	SparkStatus status;
	status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,cudaStreamCreate(&stream),"cudaStreamCreate");
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	slot->cuda_stream = stream;
	status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->input_token_ids);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->output_token_ids);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->row_lane_indices);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->slot_mapping);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->context_lengths);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint64_t),(void **)&slot->row_positions);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint64_t),(void **)&slot->argmax_reduce_u64);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->argmax_token_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,hidden_bytes,&slot->hidden_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,hidden_bytes,&slot->normalized_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,query_bytes,&slot->query_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,kv_bytes,&slot->key_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,kv_bytes,&slot->value_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,query_roped_bytes,&slot->query_roped_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,query_bytes,&slot->attended_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,hidden_bytes,&slot->attn_output_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,ffn_bytes,&slot->mlp_gate_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,ffn_bytes,&slot->mlp_up_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,ffn_bytes,&slot->mlp_down_bf16);
	return(status);
}

static SparkStatus SparkMinimaxModuleAllocateSlotHostMirrors(SparkMinimaxModuleState *state,SparkMinimaxModuleSlot *slot)
{
	uint64_t rows = state->max_active_sequence_count;
	slot->host_row_lane_indices = (uint32_t *)malloc((size_t)rows * sizeof(uint32_t));
	slot->host_row_positions = (uint64_t *)malloc((size_t)rows * sizeof(uint64_t));
	slot->host_slot_mapping = (uint32_t *)malloc((size_t)rows * sizeof(uint32_t));
	slot->host_context_lengths = (uint32_t *)malloc((size_t)rows * sizeof(uint32_t));
	if ( slot->host_row_lane_indices == 0 || slot->host_row_positions == 0 ||
		slot->host_slot_mapping == 0 || slot->host_context_lengths == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkMinimaxModuleAllocatePools(SparkMinimaxModuleState *state)
{
	uint64_t lane_bytes,block_bytes,count_bytes;
	uint32_t block;
	state->blocks_per_lane = (SPARK_MINIMAX_RESIDENT_DECODE_STAGE_MAXIMUM_CONTEXT_TOKENS +
		SPARK_MINIMAX_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS - 1u) /
		SPARK_MINIMAX_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
	state->kv_layer_stride = (uint64_t)SPARK_MINIMAX_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS *
		2u * (uint64_t)state->local_kv_dimension * SPARK_MINIMAX_RESIDENT_DECODE_STAGE_BF16_ELEMENT_BYTES;
	state->kv_block_stride = state->kv_layer_stride / SPARK_MINIMAX_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
	block_bytes = state->kv_layer_stride * state->layer_count;
	lane_bytes = (uint64_t)state->blocks_per_lane * sizeof(uint32_t);
	count_bytes = (uint64_t)state->max_active_sequence_count * sizeof(uint32_t);
	state->host_block_indices = (uint32_t *)malloc((size_t)((uint64_t)state->max_active_sequence_count * lane_bytes));
	state->free_blocks = (uint32_t *)malloc((size_t)((uint64_t)state->kv_block_count * sizeof(uint32_t)));
	state->lane_block_counts = (uint32_t *)calloc(state->max_active_sequence_count,sizeof(uint32_t));
	state->lane_context_tokens = (uint64_t *)calloc(state->max_active_sequence_count,sizeof(uint64_t));
	if ( state->host_block_indices == 0 || state->free_blocks == 0 ||
		state->lane_block_counts == 0 || state->lane_context_tokens == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (block = 0u; block < state->kv_block_count; block++)
		state->free_blocks[block] = state->kv_block_count - 1u - block;
	state->free_block_count = state->kv_block_count;
	if ( SparkStageModuleDeviceAllocate(&state->ledger,block_bytes * state->kv_block_count,&state->kv_cache_bf16) != SPARK_STATUS_OK )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)state->max_active_sequence_count * lane_bytes,(void **)&state->device_block_indices) != SPARK_STATUS_OK )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( SparkStageModuleDeviceAllocate(&state->ledger,count_bytes,(void **)&state->device_block_counts) != SPARK_STATUS_OK )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->block_table.abi_version = SPARK_MINIMAX_RESIDENT_DECODE_STAGE_KV_BLOCK_TABLE_ABI_VERSION;
	state->block_table.descriptor_bytes = sizeof(state->block_table);
	state->block_table.block_token_count = SPARK_MINIMAX_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
	state->block_table.lane_count = state->max_active_sequence_count;
	state->block_table.lane_stride = state->blocks_per_lane;
	state->block_table.lane_capacity = state->max_active_sequence_count;
	state->block_table.physical_block_indices = state->device_block_indices;
	state->block_table.lane_physical_block_counts = state->device_block_counts;
	state->block_table.host_physical_block_indices = state->host_block_indices;
	state->block_table.host_lane_physical_block_counts = state->lane_block_counts;
	fprintf(stderr,"%s kv_cache blocks=%u layer_stride=%llu block_stride=%llu bytes=%llu\n",
		SPARK_MINIMAX_MODULE_TAG,state->kv_block_count,
		(unsigned long long)state->kv_layer_stride,
		(unsigned long long)state->kv_block_stride,
		(unsigned long long)(block_bytes * state->kv_block_count));
	return(SPARK_STATUS_OK);
}

static uint32_t SparkMinimaxBlockTableCoverRequired(uint64_t end_position)
{
	return((uint32_t)((end_position + SPARK_MINIMAX_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS - 1u) /
		SPARK_MINIMAX_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS));
}

static SparkStatus SparkMinimaxModuleCoverLane(SparkMinimaxModuleState *state,uint32_t lane,uint64_t end_position)
{
	uint32_t required,ordinal;
	required = SparkMinimaxBlockTableCoverRequired(end_position);
	if ( required > state->blocks_per_lane )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (ordinal = state->lane_block_counts[lane]; ordinal < required; ordinal++)
	{
		if ( state->free_block_count == 0u )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		state->host_block_indices[((uint64_t)lane * state->blocks_per_lane) + ordinal] = state->free_blocks[--state->free_block_count];
		state->lane_block_counts[lane] = ordinal + 1u;
	}
	state->lane_block_counts[lane] = required;
	return(SPARK_STATUS_OK);
}

static void SparkMinimaxModuleReleaseLane(SparkMinimaxModuleState *state,uint32_t lane)
{
	uint32_t ordinal;
	for (ordinal = 0u; ordinal < state->lane_block_counts[lane]; ordinal++)
		state->free_blocks[state->free_block_count++] = state->host_block_indices[((uint64_t)lane * state->blocks_per_lane) + ordinal];
	state->lane_block_counts[lane] = 0u;
	state->lane_context_tokens[lane] = 0u;
}

static SparkStatus SparkMinimaxModuleUploadBlockTable(SparkMinimaxModuleState *state,const SparkMinimaxResidentDecodeStageFrameContext *context,const SparkModelDriverFrame *frame)
{
	uint32_t lane;
	(void)context;
	(void)frame;
	{
		uint64_t lane_slice_bytes = (uint64_t)state->blocks_per_lane * sizeof(uint32_t);
		uint64_t counts_bytes = (uint64_t)state->max_active_sequence_count * sizeof(uint32_t);
		for (lane = 0u; lane < state->max_active_sequence_count; lane++)
			if ( state->lane_block_counts[lane] != 0u )
			{
				cudaError_t error = cudaMemcpyAsync((uint8_t *)state->device_block_indices + ((uint64_t)lane * lane_slice_bytes),
					(const uint8_t *)state->host_block_indices + ((uint64_t)lane * lane_slice_bytes),
					(size_t)lane_slice_bytes,cudaMemcpyHostToDevice,(cudaStream_t)state->slots[0].cuda_stream);
				if ( error != cudaSuccess )
					return(SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,error,"block_table_upload"));
			}
		
		if ( cudaMemcpyAsync(state->device_block_counts,state->lane_block_counts,(size_t)counts_bytes,
			cudaMemcpyHostToDevice,(cudaStream_t)state->slots[0].cuda_stream) != cudaSuccess )
			return(SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,cudaGetLastError(),"block_table_counts_upload"));
	}
	return(SPARK_STATUS_OK);
}

static void SparkMinimaxModuleFrameValidate(const SparkMinimaxResidentDecodeStageFrameContext *context,SparkStatus *status)
{
	if ( context == 0 || context->abi_version != SPARK_MINIMAX_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION ||
		context->descriptor_bytes < sizeof(*context) )
	{
		*status = SPARK_STATUS_INVALID_ARGUMENT;
		return;
	}
	*status = SPARK_STATUS_OK;
}

#include "sparkpipe/family/module/spark_module_tp_all_reduce_hidden.h"

static SparkStatus SparkMinimaxModuleRunDecode(SparkMinimaxModuleState *state,SparkMinimaxModuleSlot *slot,const SparkModelDriverFrame *frame,const SparkMinimaxResidentDecodeStageFrameContext *context,uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	const SparkMinimaxKvBlockTableView *table = context != 0 ? context->kv_block_table : 0;
	uint32_t layer;
	SparkStatus status;
	if ( rows != frame->active_slot_count || rows > state->max_active_sequence_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( table == 0 || table->physical_block_indices == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (uint32_t row = 0u; row < rows; row++)
	{
		uint32_t lane = slot->host_row_lane_indices[row];
		uint64_t position = slot->host_row_positions[row];
		if ( lane >= state->max_active_sequence_count )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		slot->host_slot_mapping[row] = (uint32_t)(position % SPARK_MINIMAX_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
		slot->host_context_lengths[row] = (uint32_t)(position % SPARK_MINIMAX_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS) + 1u;
	}
	status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,cudaMemcpyAsync(slot->input_token_ids,frame->buffers != 0 && frame->buffer_count != 0 ? frame->buffers[0].address : 0,(size_t)rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream),"decode_tokens_h2d");
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,cudaMemcpyAsync(slot->row_lane_indices,slot->host_row_lane_indices,(size_t)rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream),"decode_lanes_h2d");
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,cudaMemcpyAsync(slot->row_positions,slot->host_row_positions,(size_t)rows * sizeof(uint64_t),cudaMemcpyHostToDevice,stream),"decode_positions_h2d");
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,cudaMemcpyAsync(slot->slot_mapping,slot->host_slot_mapping,(size_t)rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream),"decode_slots_h2d");
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,cudaMemcpyAsync(slot->context_lengths,slot->host_context_lengths,(size_t)rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream),"decode_contexts_h2d");
	if ( status == SPARK_STATUS_OK )
		status = SparkMinimaxModuleUploadBlockTable(state,context,frame);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchEmbeddingGather(stream,slot->input_token_ids,state->token_embedding_bf16,slot->hidden_bf16,rows),"embedding_gather");
	for (layer = 0u; status == SPARK_STATUS_OK && layer < state->layer_count; layer++)
	{
		const SparkMinimaxAttentionLayerWeights *attention = &state->attention_by_layer[layer];
		const SparkMinimaxMlpLayerWeights *mlp = &state->mlp_by_layer[layer];
		status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchRmsNorm(stream,slot->hidden_bf16,state->input_norm_by_layer[layer],slot->normalized_bf16,rows,SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,SPARK_MINIMAX_RESIDENT_DECODE_STAGE_RMS_NORM_EPSILON),"input_norm");
		if ( status != SPARK_STATUS_OK )
			break;
		status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchQueryKeyProjection(stream,&attention->query,&attention->key,slot->normalized_bf16,slot->query_bf16,slot->key_bf16,rows),"qk_proj");
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchValueProjection(stream,&attention->value,slot->normalized_bf16,slot->value_bf16,rows),"v_proj");
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchHeadNormRope(stream,slot->query_bf16,slot->key_bf16,attention->query_norm_weight_bf16,attention->key_norm_weight_bf16,slot->query_roped_bf16,slot->row_positions,rows,SPARK_MINIMAX_RESIDENT_DECODE_STAGE_RMS_NORM_EPSILON,state->local_query_head_count,state->local_kv_head_count),"head_norm_rope");
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchKvCacheWrite(stream,slot->key_bf16,slot->value_bf16,state->kv_cache_bf16,table->physical_block_indices,table->lane_physical_block_counts,slot->row_lane_indices,slot->slot_mapping,rows,state->local_kv_dimension,state->blocks_per_lane,state->kv_layer_stride,state->kv_block_stride,layer),"kv_cache_write");
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchAttentionDecode(stream,slot->query_roped_bf16,state->kv_cache_bf16,table,slot->row_lane_indices,slot->slot_mapping,slot->context_lengths,slot->attended_bf16,rows,state->local_query_head_count,state->local_kv_head_count,state->local_kv_dimension,state->kv_layer_stride,state->kv_block_stride,SPARK_MINIMAX_RESIDENT_DECODE_STAGE_RMS_NORM_EPSILON,state->tp_rank,layer),"attn_decode");
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchProjection(stream,&attention->output,slot->attended_bf16,slot->attn_output_bf16,rows),"o_proj");
		if ( status == SPARK_STATUS_OK )
			status = SparkMinimaxModuleTpAllReduceHidden(state,slot,slot->attn_output_bf16,rows);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchResidualAdd(stream,slot->hidden_bf16,slot->attn_output_bf16,rows,SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION),"attn_residual");
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchRmsNorm(stream,slot->hidden_bf16,state->post_attention_norm_by_layer[layer],slot->normalized_bf16,rows,SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,SPARK_MINIMAX_RESIDENT_DECODE_STAGE_RMS_NORM_EPSILON),"post_attention_norm");
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchProjection(stream,&mlp->gate,slot->normalized_bf16,slot->mlp_gate_bf16,rows),"gate_proj");
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchProjection(stream,&mlp->up,slot->normalized_bf16,slot->mlp_up_bf16,rows),"up_proj");
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchSwiGlu(stream,slot->mlp_gate_bf16,slot->mlp_up_bf16,slot->mlp_down_bf16,rows,state->local_ffn_dimension),"swiglu");
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchProjection(stream,&mlp->down,slot->mlp_down_bf16,slot->attn_output_bf16,rows),"down_proj");
		if ( status == SPARK_STATUS_OK )
			status = SparkMinimaxModuleTpAllReduceHidden(state,slot,slot->attn_output_bf16,rows);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchResidualAdd(stream,slot->hidden_bf16,slot->attn_output_bf16,rows,SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION),"mlp_residual");
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchRmsNorm(stream,slot->hidden_bf16,state->final_norm_weight_bf16,slot->normalized_bf16,rows,SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,SPARK_MINIMAX_RESIDENT_DECODE_STAGE_RMS_NORM_EPSILON),"final_norm");
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchVocabArgmax(stream,state->lm_head_weight_bf16,slot->normalized_bf16,slot->argmax_reduce_u64,rows,state->local_vocab_rows,SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,state->local_vocab_base),"vocab_argmax");
	if ( status == SPARK_STATUS_OK )
		status = SparkMinimaxModuleTpReduceArgmax(state,slot,slot->argmax_reduce_u64,rows);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,cudaMemcpyAsync(slot->output_token_ids,slot->argmax_token_u32,(size_t)rows * sizeof(uint32_t),cudaMemcpyDeviceToDevice,stream),"argmax_copy");
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,cudaStreamSynchronize(stream),"decode_sync");
	if ( status == SPARK_STATUS_OK && frame->buffers != 0 )
	{
		for (uint32_t buffer_index = 0u; buffer_index < frame->buffer_count; buffer_index++)
		{
			SparkModelDriverBuffer *buffer = &frame->buffers[buffer_index];
			if ( (buffer->flags & SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE) != 0u )
				status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,cudaMemcpy(buffer->address,slot->argmax_token_u32,buffer->bytes < (size_t)rows * sizeof(uint32_t) ? buffer->bytes : (size_t)rows * sizeof(uint32_t),cudaMemcpyDeviceToHost),"tokens_d2h");
		}
	}
	return(status);
}

static SparkStatus SparkMinimaxModuleRunPrefill(SparkMinimaxModuleState *state,SparkMinimaxModuleSlot *slot,const SparkModelDriverFrame *frame,const SparkMinimaxResidentDecodeStageFrameContext *context,const SparkMinimaxPrefillFrameView *prefill)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	const SparkMinimaxKvBlockTableView *table = context != 0 ? context->kv_block_table : 0;
	const uint32_t rows = prefill->token_count;
	uint32_t layer;
	SparkStatus status;
	if ( rows == 0u || rows > state->max_active_sequence_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (uint32_t row = 0u; row < rows; row++)
	{
		uint64_t position = prefill->base_position + row;
		slot->host_slot_mapping[row] = (uint32_t)(position % SPARK_MINIMAX_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
		slot->host_context_lengths[row] = (uint32_t)(position % SPARK_MINIMAX_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS) + 1u;
		slot->host_row_positions[row] = position;
	}
	status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,cudaMemcpyAsync(slot->input_token_ids,prefill->row_token_ids,(size_t)rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream),"prefill_tokens_h2d");
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,cudaMemcpyAsync(slot->row_positions,slot->host_row_positions,(size_t)rows * sizeof(uint64_t),cudaMemcpyHostToDevice,stream),"prefill_positions_h2d");
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,cudaMemcpyAsync(slot->slot_mapping,slot->host_slot_mapping,(size_t)rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream),"prefill_slots_h2d");
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,cudaMemcpyAsync(slot->context_lengths,slot->host_context_lengths,(size_t)rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream),"prefill_contexts_h2d");
	if ( status == SPARK_STATUS_OK )
		status = SparkMinimaxModuleUploadBlockTable(state,context,frame);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchEmbeddingGather(stream,slot->input_token_ids,state->token_embedding_bf16,slot->hidden_bf16,rows),"prefill_embedding_gather");
	for (layer = 0u; status == SPARK_STATUS_OK && layer < state->layer_count; layer++)
	{
		const SparkMinimaxAttentionLayerWeights *attention = &state->attention_by_layer[layer];
		const SparkMinimaxMlpLayerWeights *mlp = &state->mlp_by_layer[layer];
		status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchRmsNorm(stream,slot->hidden_bf16,state->input_norm_by_layer[layer],slot->normalized_bf16,rows,SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,SPARK_MINIMAX_RESIDENT_DECODE_STAGE_RMS_NORM_EPSILON),"prefill_input_norm");
		if ( status != SPARK_STATUS_OK )
			break;
		status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchQueryKeyProjection(stream,&attention->query,&attention->key,slot->normalized_bf16,slot->query_bf16,slot->key_bf16,rows),"prefill_qk_proj");
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchValueProjection(stream,&attention->value,slot->normalized_bf16,slot->value_bf16,rows),"prefill_v_proj");
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchHeadNormRope(stream,slot->query_bf16,slot->key_bf16,attention->query_norm_weight_bf16,attention->key_norm_weight_bf16,slot->query_roped_bf16,slot->row_positions,rows,SPARK_MINIMAX_RESIDENT_DECODE_STAGE_RMS_NORM_EPSILON,state->local_query_head_count,state->local_kv_head_count),"prefill_head_norm_rope");
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchAttentionPrefill(stream,slot->query_roped_bf16,state->kv_cache_bf16,table,slot->row_lane_indices,slot->row_positions,slot->key_bf16,slot->value_bf16,slot->attended_bf16,rows,state->local_query_head_count,state->local_kv_head_count,state->local_kv_dimension,state->blocks_per_lane,state->kv_layer_stride,state->kv_block_stride,SPARK_MINIMAX_RESIDENT_DECODE_STAGE_RMS_NORM_EPSILON,state->tp_rank,layer,prefill->base_position),"prefill_attn");
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchKvCacheWrite(stream,slot->key_bf16,slot->value_bf16,state->kv_cache_bf16,table->physical_block_indices,table->lane_physical_block_counts,slot->row_lane_indices,slot->slot_mapping,rows,state->local_kv_dimension,state->blocks_per_lane,state->kv_layer_stride,state->kv_block_stride,layer),"prefill_kv_cache_write");
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchProjection(stream,&attention->output,slot->attended_bf16,slot->attn_output_bf16,rows),"prefill_o_proj");
		if ( status == SPARK_STATUS_OK )
			status = SparkMinimaxModuleTpAllReduceHidden(state,slot,slot->attn_output_bf16,rows);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchResidualAdd(stream,slot->hidden_bf16,slot->attn_output_bf16,rows,SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION),"prefill_attn_residual");
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchRmsNorm(stream,slot->hidden_bf16,state->post_attention_norm_by_layer[layer],slot->normalized_bf16,rows,SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,SPARK_MINIMAX_RESIDENT_DECODE_STAGE_RMS_NORM_EPSILON),"prefill_post_attention_norm");
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchProjection(stream,&mlp->gate,slot->normalized_bf16,slot->mlp_gate_bf16,rows),"prefill_gate_proj");
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchProjection(stream,&mlp->up,slot->normalized_bf16,slot->mlp_up_bf16,rows),"prefill_up_proj");
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchSwiGlu(stream,slot->mlp_gate_bf16,slot->mlp_up_bf16,slot->mlp_down_bf16,rows,state->local_ffn_dimension),"prefill_swiglu");
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchProjection(stream,&mlp->down,slot->mlp_down_bf16,slot->attn_output_bf16,rows),"prefill_down_proj");
		if ( status == SPARK_STATUS_OK )
			status = SparkMinimaxModuleTpAllReduceHidden(state,slot,slot->attn_output_bf16,rows);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchResidualAdd(stream,slot->hidden_bf16,slot->attn_output_bf16,rows,SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION),"prefill_mlp_residual");
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchRmsNorm(stream,slot->hidden_bf16,state->final_norm_weight_bf16,slot->normalized_bf16,rows,SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,SPARK_MINIMAX_RESIDENT_DECODE_STAGE_RMS_NORM_EPSILON),"prefill_final_norm");
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchVocabArgmax(stream,state->lm_head_weight_bf16,slot->normalized_bf16,slot->argmax_reduce_u64,rows,state->local_vocab_rows,SPARK_MINIMAX_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,state->local_vocab_base),"prefill_vocab_argmax");
	if ( status == SPARK_STATUS_OK )
		status = SparkMinimaxModuleTpReduceArgmax(state,slot,slot->argmax_reduce_u64,rows);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxLaunchArgmaxResolve(stream,slot->argmax_reduce_u64,slot->argmax_token_u32,rows),"prefill_argmax_resolve");
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,cudaStreamSynchronize(stream),"prefill_sync");
	if ( status == SPARK_STATUS_OK )
	{
		for (uint32_t buffer_index = 0u; buffer_index < frame->buffer_count; buffer_index++)
		{
			SparkModelDriverBuffer *buffer = &frame->buffers[buffer_index];
			if ( (buffer->flags & SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE) != 0u )
				status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,cudaMemcpy(buffer->address,slot->argmax_token_u32 + (rows - 1u),sizeof(uint32_t),cudaMemcpyDeviceToHost),"prefill_token_d2h");
		}
	}
	return(status);
}

static SparkStatus SparkMinimaxModuleExecuteFrame(void *module_state,SparkModelDriverFrame *frame)
{
	SparkMinimaxModuleState *state = (SparkMinimaxModuleState *)module_state;
	SparkMinimaxResidentDecodeStageFrameContext *context;
	SparkMinimaxModuleSlot *slot;
	uint32_t rows;
	SparkStatus status;
	context = (SparkMinimaxResidentDecodeStageFrameContext *)frame->user_context;
	if ( (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u )
	{
		const SparkMinimaxPrefillFrameView *prefill = context != 0 ? context->prefill_frame : 0;
		if ( prefill == 0 || prefill->token_count == 0u || prefill->token_count > state->max_active_sequence_count || prefill->lane_index >= state->max_active_sequence_count )
		{
			atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		}
		slot = &state->slots[0];
		slot->logical_sequence_count = frame->active_slot_count;
		status = SparkMinimaxModuleRunPrefill(state,slot,frame,context,prefill);
		if ( status == SPARK_STATUS_OK )
		{
			atomic_fetch_add_explicit(&state->completed_count,1u,memory_order_relaxed);
			atomic_fetch_add_explicit(&state->tokens_emitted,prefill->token_count,memory_order_relaxed);
		}
		else
			atomic_fetch_add_explicit(&state->failed_count,1u,memory_order_relaxed);
		SPARK_RETURN(status);
	}
	SparkMinimaxModuleFrameValidate(context,&status);
	if ( status != SPARK_STATUS_OK )
	{
		atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
		SPARK_RETURN(status);
	}
	if ( (context->flags & SPARK_MINIMAX_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW) == 0u ||
		context->decode_batch == 0 )
	{
		atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	rows = frame->active_slot_count;
	if ( rows == 0u || rows > state->max_active_sequence_count || context->decode_batch->row_count != rows )
	{
		atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	slot = &state->slots[0];
	slot->logical_sequence_count = frame->active_slot_count;
	if ( slot->cuda_stream == 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	for (uint32_t row = 0u; row < rows; row++)
	{
		uint32_t lane = context->decode_batch->row_lane_indices[row];
		if ( lane >= state->max_active_sequence_count )
		{
			atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		}
		slot->host_row_lane_indices[row] = lane;
		slot->host_row_positions[row] = context->decode_batch->row_positions[row];
		if ( slot->host_row_positions[row] == 0u && state->lane_context_tokens[lane] != 0u )
			SparkMinimaxModuleReleaseLane(state,lane);
		if ( slot->host_row_positions[row] + 1u > state->lane_context_tokens[lane] )
		{
			state->lane_context_tokens[lane] = slot->host_row_positions[row] + 1u;
			if ( SparkMinimaxModuleCoverLane(state,lane,state->lane_context_tokens[lane]) != SPARK_STATUS_OK )
			{
				atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
				SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
			}
		}
	}
	atomic_fetch_add_explicit(&state->submitted_count,1u,memory_order_relaxed);
	status = SparkMinimaxModuleRunDecode(state,slot,frame,context,rows);
	if ( status == SPARK_STATUS_OK )
	{
		atomic_fetch_add_explicit(&state->completed_count,1u,memory_order_relaxed);
		atomic_fetch_add_explicit(&state->tokens_emitted,rows,memory_order_relaxed);
	}
	else
		atomic_fetch_add_explicit(&state->failed_count,1u,memory_order_relaxed);
	SPARK_RETURN(status);
}

static SparkStatus SparkMinimaxModuleReset(
	SparkMinimaxModuleState *state,
	const SparkModelDriverAdmissionRequest *request)
{
	uint32_t slots[SPARK_MINIMAX_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	uint32_t lane;
	uint32_t retained;
	uint32_t index;
	cudaError_t drain;
	SparkStatus status;
	if ( SparkModelDriverAdmissionRequestIsValid(request) == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( request->control_generation <= state->reset_generation )
		return(SPARK_STATUS_OK);
	for (index=0u; index<state->pipeline_slot_count; index++)
		slots[index] = index;
	status = SparkStageModuleIndexSetClaim(state->slot_states,state->pipeline_slot_count,slots,state->pipeline_slot_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	for (index=0u; index<state->pipeline_slot_count; index++)
	{
		if ( state->slots[index].cuda_stream == 0 )
			continue;
		drain = cudaStreamSynchronize((cudaStream_t)state->slots[index].cuda_stream);
		if ( drain != cudaSuccess )
		{
			(void)SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,drain,"reset_stream_drain");
			return(SPARK_STATUS_PENDING);
		}
	}
	for (lane=0u; lane<state->max_active_sequence_count; lane++)
		SparkMinimaxModuleReleaseLane(state,lane);
	retained = 0u;
	for (lane=0u; lane<state->max_active_sequence_count; lane++)
		retained += state->lane_block_counts[lane];
	if ( retained != 0u || state->free_block_count != state->kv_block_count )
	{
		SparkStageModuleIndexSetRelease(state->slot_states,state->pipeline_slot_count,slots,state->pipeline_slot_count);
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	if ( cudaMemcpy(state->device_block_counts,state->lane_block_counts,
		(size_t)((uint64_t)state->max_active_sequence_count * sizeof(uint32_t)),
		cudaMemcpyHostToDevice) != cudaSuccess )
	{
		SparkStageModuleIndexSetRelease(state->slot_states,state->pipeline_slot_count,slots,state->pipeline_slot_count);
		return(SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,cudaGetLastError(),"reset_block_counts_upload"));
	}
	state->reset_generation = request->control_generation;
	SparkStageModuleIndexSetRelease(state->slot_states,state->pipeline_slot_count,slots,state->pipeline_slot_count);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkMinimaxModuleAdmit(
	void *module_state,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	SparkMinimaxModuleState *state = (SparkMinimaxModuleState *)module_state;
	SparkAdmissionPolicyTable table;
	uint32_t available_slot_count;
	SparkStatus status;
	if ( request->admission_flags == SPARK_MODEL_DRIVER_ADMISSION_FLAG_RESET )
	{
		SparkModelDriverInitializeAdmissionDecision(decision);
		status = SparkMinimaxModuleReset(state,request);
		if ( status == SPARK_STATUS_OK )
		{
			decision->accepted = 1u;
			decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
		}
		SPARK_RETURN(status);
	}
	available_slot_count = SparkStageModuleSlotCountFree(
		state->slot_states,
		state->pipeline_slot_count);
	memset(&table,0,sizeof(table));
	table.abi_version = SPARK_ADMISSION_ABI_VERSION;
	table.descriptor_bytes = (uint32_t)sizeof(table);
	table.max_active_sequence_count = state->max_active_sequence_count;
	table.max_input_row_count = state->max_active_sequence_count;
	table.max_sequence_positions = SPARK_MINIMAX_RESIDENT_DECODE_STAGE_MAXIMUM_CONTEXT_TOKENS;
	table.flags = SPARK_ADMISSION_POLICY_FLAG_PREFILL_SINGLE_SLOT |
		SPARK_ADMISSION_POLICY_FLAG_DECODE_EQUALS_SLOTS;
	table.predicate = 0;
	table.predicate_context = 0;
	table.cost = SparkStageModuleAdmissionCost;
	table.cost_context = state;
	status = SparkAdmissionEvaluateShape(&table,available_slot_count,request,decision);
	if (status != SPARK_STATUS_OK)
		SPARK_RETURN(status);
	if (decision->accepted == 0u)
		atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
	SPARK_RETURN(status);
}

static void SparkMinimaxModuleSnapshotExtend(
	void *module_state,
	SparkModelDriverRuntimeSnapshot *snapshot)
{
	SparkMinimaxModuleState *state = (SparkMinimaxModuleState *)module_state;
	snapshot->kv_token_capacity = (uint64_t)state->kv_block_count * SPARK_MINIMAX_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
}

static SparkStatus SparkMinimaxModuleStateTeardown(void *module_state)
{
	SparkMinimaxModuleState *state = (SparkMinimaxModuleState *)module_state;
	if ( state->tp_collective_initialized != 0u )
	{
		SparkTpDeviceCollectiveDestroy(&state->tp_device_collective);
		if ( state->tp_device_collective.implementation != 0 )
			return(SPARK_STATUS_BUSY);
		state->tp_collective_initialized = 0u;
	}
	free(state->host_block_indices);
	free(state->free_blocks);
	free(state->lane_block_counts);
	free(state->lane_context_tokens);
	SparkStageModuleLedgerRelease(&state->ledger);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkMinimaxModuleInitializeGate(void)
{
	uint32_t allow_unqualified_execution = 0u;
	if ( SparkStageModuleEnvironmentUnsigned(SPARK_MINIMAX_MODULE_TAG,"SPARK_MINIMAX_ALLOW_UNQUALIFIED_EXECUTION",1u,1u,&allow_unqualified_execution) != SPARK_STATUS_OK || allow_unqualified_execution != 1u )
		SPARK_FAIL(SPARK_STATUS_MODULE_NOT_VALIDATED);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkMinimaxModulePrepare(
	void *module_state,
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services)
{
	SparkMinimaxModuleState *state = (SparkMinimaxModuleState *)module_state;
	const char *pack_path;
	uint32_t mapped_unused = 0u;
	int32_t sm_count = 0;
	SparkStatus status;
	(void)configuration;
	(void)host_services;
	pack_path = 0;
	state->allow_unqualified_execution = 1u;
	atomic_init(&state->tp_completion_flag,0u);
	atomic_init(&state->tp_next_ordinal,0u);
	status = SparkMinimaxModuleConfigure(state);
	if ( status == SPARK_STATUS_OK )
		SparkStageModuleAtomicStateArrayInitialize(state->slot_states,state->pipeline_slot_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentText(SPARK_MINIMAX_MODULE_TAG,"SPARK_MINIMAX_STAGE_PACK_PATH",&pack_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkMinimaxModuleLoadMappedPack(state,pack_path,1u,&mapped_unused);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_MINIMAX_MODULE_TAG,SparkMinimaxConfigureCudaKernels(),"configure_cuda_kernels");
	if ( status == SPARK_STATUS_OK )
	{
		cudaError_t attr = cudaDeviceGetAttribute(&sm_count,cudaDevAttrMultiProcessorCount,0);
		state->multiprocessor_count = attr == cudaSuccess && sm_count > 0 ? (uint32_t)sm_count : 1u;
		status = SparkMinimaxModuleAllocatePools(state);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkMinimaxModuleAllocateSlot(state,&state->slots[0]);
	if ( status == SPARK_STATUS_OK )
		status = SparkMinimaxModuleAllocateSlotHostMirrors(state,&state->slots[0]);
	if ( status == SPARK_STATUS_OK )
		status = SparkMinimaxModuleInitializeTpCollective(state);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"%s initialize_failed status=%d\n",SPARK_MINIMAX_MODULE_TAG,(int)status);
	else
		fprintf(stderr,"%s initialize ok layers=%u tp=%u/%u local_heads=%u local_kv=%u local_ffn=%u vocab_rows=%u base=%u\n",
			SPARK_MINIMAX_MODULE_TAG,state->layer_count,state->tp_degree,state->tp_rank,
			state->local_query_head_count,state->local_kv_head_count,
			state->local_ffn_dimension,state->local_vocab_rows,state->local_vocab_base);
	SPARK_RETURN(status);
}

static void SparkMinimaxModuleReportReady(void *module_state)
{
	SparkMinimaxModuleState *state = (SparkMinimaxModuleState *)module_state;
	(void)state;
}

static const SparkStageModuleLifecycleOps SparkMinimaxModuleLifecycle =
{
	sizeof(SparkMinimaxModuleState),
	SparkMinimaxModuleInitializeGate,
	SparkMinimaxModuleDescribe,
	SparkMinimaxModulePrepare,
	SparkMinimaxModuleReportReady,
	SparkMinimaxModuleStateTeardown,
	SparkMinimaxModuleExecuteFrame,
	SparkMinimaxModuleAdmit,
	SparkMinimaxModuleSnapshotExtend
};

SPARK_STAGE_MODULE_LIFECYCLE_ENTRY_POINTS(
	SparkMinimaxResidentDecodeStage,
	&SparkMinimaxModuleLifecycle)
