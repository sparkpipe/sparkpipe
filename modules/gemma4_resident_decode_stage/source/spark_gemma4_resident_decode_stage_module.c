#define _FILE_OFFSET_BITS 64

#include <errno.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_admission.h"
#include "sparkpipe/spark_hidden_transport.h"
#include "sparkpipe/spark_gemma4_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "sparkpipe/spark_stage_module_lifecycle.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "spark_gemma4_stagepack_format.h"

#define SPARK_GEMMA4_MODULE_TAG "gemma4_stage"

#define SPARK_GEMMA4_MODULE_TP_DEGREE_REPLICATED 1u
#define SPARK_GEMMA4_MODULE_TP_RANK_REPLICATED 0u
#define SPARK_GEMMA4_MODULE_TP_TIMEOUT_MILLI_DEFAULT 120000u
#define SPARK_GEMMA4_MODULE_TP_MAX_DEGREE 16u
#define SPARK_FRAME_ERROR_WORDS 6u

typedef struct SparkGemma4ModuleSlot
{
	uint32_t logical_sequence_count;
	void *cuda_stream;
	uint32_t *host_row_lane_indices;
	uint64_t *host_row_positions;
	uint32_t *host_row_positions_u32;
	uint32_t *host_row_sequences_u32;
	uint32_t *host_slot_mapping;
	uint32_t *host_context_lengths;
	uint32_t *host_frame_error;
	uint32_t *input_token_ids;
	uint32_t *output_token_ids;
	uint32_t *row_lane_indices;
	uint32_t *slot_mapping;
	uint32_t *context_lengths;
	uint64_t *row_positions;
	uint32_t *row_positions_u32;
	uint32_t *row_sequences_u32;
	uint32_t *window_positions_u32;
	void *hidden_bf16;
	void *residual_bf16;
	void *normalized_bf16;
	void *sliding_query_bf16;
	void *sliding_kv_bf16;
	void *full_query_bf16;
	void *full_key_bf16;
	void *full_value_bf16;
	void *attn_head_output_bf16;
	void *attn_output_bf16;
	void *delta_bf16;
	void *mlp_gate_up_bf16;
	void *mlp_down_bf16;
	void *branch_bf16;
	uint32_t *moe_indices_u32;
	float *moe_scores_f32;
	float *moe_weights_f32;
	uint32_t *moe_inverse_u32;
	uint32_t *moe_grouped_rows_u32;
	uint32_t *moe_tile_prefix_w1_u32;
	uint32_t *moe_tile_prefix_w2_u32;
	uint32_t *moe_group_offset_u32;
	void *moe_gate_packed_bf16;
	void *moe_slot_out_bf16;
	void *argmax_score_f32;
	void *argmax_token_ids;
	uint64_t *head_maxloc_u64;
	uint32_t *frame_error;
} SparkGemma4ModuleSlot;

typedef struct SparkGemma4ModuleState
{
	SparkStageModuleLedger ledger;
	uint32_t multiprocessor_count;
	uint32_t tp_degree;
	uint32_t tp_rank;
	SparkTpDeviceCollective tp_device_collective;
	uint32_t tp_collective_initialized;
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
	atomic_uint slot_states[SPARK_GEMMA4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	uint32_t kv_block_count;
	atomic_ullong submitted_count;
	atomic_ullong completed_count;
	atomic_ullong rejected_count;
	atomic_ullong failed_count;
	atomic_ullong tokens_emitted;
	void *sliding_kv_pool_bf16;
	void *full_kv_pool_bf16;
	void *kv_access_error;
	uint64_t sliding_kv_pool_bytes;
	uint64_t full_kv_pool_bytes;
	uint32_t sliding_kv_heads_per_rank;
	uint32_t full_kv_heads_per_rank;
	SparkGemma4ModuleSlot slots[SPARK_GEMMA4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	uint32_t stage_count;
	uint32_t stage_index;
	uint32_t allow_unqualified_execution;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t owns_embedding;
	uint32_t owns_final_head;
	uint32_t tp_standalone;
	uint32_t tp_vocab_base;
	uint32_t tp_vocab_rows;
	uint32_t sliding_layer_count;
	uint32_t full_layer_count;
	uint32_t sliding_ordinal_by_layer[SPARK_GEMMA4_MODEL_LAYER_COUNT];
	uint32_t full_ordinal_by_layer[SPARK_GEMMA4_MODEL_LAYER_COUNT];
	uint64_t layer_seen_bits[SPARK_GEMMA4_MODEL_LAYER_COUNT];
	uint64_t global_seen_bits;
	uint64_t mtp_seen_bits;
	const void *token_embedding_bf16;
	const void *final_norm_weight_bf16;
	const void *full_rope_table_f32;
	const void *layer_input_norm_by_layer[SPARK_GEMMA4_MODEL_LAYER_COUNT];
	const void *layer_post_attention_norm_by_layer[SPARK_GEMMA4_MODEL_LAYER_COUNT];
	const void *layer_pre_feedforward_norm_by_layer[SPARK_GEMMA4_MODEL_LAYER_COUNT];
	const void *layer_post_feedforward_norm_by_layer[SPARK_GEMMA4_MODEL_LAYER_COUNT];
	SparkGemma4SlidingLayerWeights sliding_by_layer[SPARK_GEMMA4_MODEL_LAYER_COUNT];
	SparkGemma4FullLayerWeights full_by_layer[SPARK_GEMMA4_MODEL_LAYER_COUNT];
	SparkGemma4DenseMlpWeights mlp_by_layer[SPARK_GEMMA4_MODEL_LAYER_COUNT];
#if SPARK_GEMMA4_MODEL_MOE_BLOCK
	SparkGemma4MoeLayerWeights moe_by_layer[SPARK_GEMMA4_MODEL_LAYER_COUNT];
#endif
} SparkGemma4ModuleState;

static SparkStatus SparkGemma4ModuleConfigure(SparkGemma4ModuleState *state)
{
	SparkStatus status;
	{
		status = SparkStageModuleEnvironmentUnsignedOrDefault(SPARK_GEMMA4_MODULE_TAG,"SPARK_GEMMA4_TP_DEGREE",1u,SPARK_GEMMA4_MODULE_TP_MAX_DEGREE,SPARK_GEMMA4_MODULE_TP_DEGREE_REPLICATED,&state->tp_degree);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleEnvironmentUnsignedOrDefault(SPARK_GEMMA4_MODULE_TAG,"SPARK_GEMMA4_TP_RANK",0u,SPARK_GEMMA4_MODULE_TP_MAX_DEGREE - 1u,SPARK_GEMMA4_MODULE_TP_RANK_REPLICATED,&state->tp_rank);
		if ( status != SPARK_STATUS_OK )
			return(status);
		if ( state->tp_rank >= state->tp_degree )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( (SPARK_GEMMA4_MODEL_SLIDING_QUERY_HEAD_COUNT % state->tp_degree) != 0u || (SPARK_GEMMA4_MODEL_OUTPUT_VOCAB_COUNT % state->tp_degree) != 0u || (SPARK_GEMMA4_MODEL_DENSE_INTERMEDIATE_DIMENSION % state->tp_degree) != 0u )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
#if SPARK_GEMMA4_MODEL_MOE_BLOCK
		if ( (SPARK_GEMMA4_MODEL_ROUTED_EXPERT_COUNT % state->tp_degree) != 0u )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
#endif
		state->sliding_kv_heads_per_rank = SparkGemma4StagePackKvHeadsPerRank(SPARK_GEMMA4_MODEL_SLIDING_KV_HEAD_COUNT,state->tp_degree);
		state->full_kv_heads_per_rank = SparkGemma4StagePackKvHeadsPerRank(SPARK_GEMMA4_MODEL_FULL_KV_HEAD_COUNT,state->tp_degree);
		state->tp_standalone = getenv("SPARK_GEMMA4_TP_STANDALONE") != 0 ? 1u : 0u;
		state->tp_vocab_rows = SPARK_GEMMA4_MODEL_OUTPUT_VOCAB_COUNT / state->tp_degree;
		state->tp_vocab_base = state->tp_rank * state->tp_vocab_rows;
		state->tp_collective_identifier = 0u;
		state->tp_control_port_base = 0u;
		state->tp_connect_timeout_milli = SPARK_GEMMA4_MODULE_TP_TIMEOUT_MILLI_DEFAULT;
		state->tp_operation_timeout_milli = SPARK_GEMMA4_MODULE_TP_TIMEOUT_MILLI_DEFAULT;
		state->tp_backend_path[0] = '\0';
		state->tp_local_host[0] = '\0';
		{
			uint32_t host_clear;
			for (host_clear = 0u; host_clear < SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE; host_clear++)
				state->tp_hosts[host_clear][0] = '\0';
		}
		if ( state->tp_degree > 1u && state->tp_standalone == 0u )
		{
			const char *tp_backend;
			const char *tp_hosts;
			const char *tp_local_host;
			uint64_t tp_identifier;
			const char *scan;
			uint32_t host_index;
			status = SparkStageModuleEnvironmentText(SPARK_GEMMA4_MODULE_TAG,"SPARK_GEMMA4_STAGE_TP_BACKEND_PATH",&tp_backend);
			if ( status == SPARK_STATUS_OK )
				status = SparkStageModuleEnvironmentUnsigned64(SPARK_GEMMA4_MODULE_TAG,"SPARK_GEMMA4_STAGE_TP_IDENTIFIER",0u,UINT64_MAX,&tp_identifier);
			if ( status == SPARK_STATUS_OK )
				status = SparkStageModuleEnvironmentUnsigned(SPARK_GEMMA4_MODULE_TAG,"SPARK_GEMMA4_STAGE_TP_PORT_BASE",1u,65535u,&state->tp_control_port_base);
			if ( status == SPARK_STATUS_OK )
				status = SparkStageModuleEnvironmentText(SPARK_GEMMA4_MODULE_TAG,"SPARK_GEMMA4_STAGE_TP_HOSTS",&tp_hosts);
			if ( status == SPARK_STATUS_OK )
				status = SparkStageModuleEnvironmentText(SPARK_GEMMA4_MODULE_TAG,"SPARK_GEMMA4_STAGE_TP_LOCAL_HOST",&tp_local_host);
			if ( status == SPARK_STATUS_OK )
				status = SparkStageModuleEnvironmentUnsignedOrDefault(SPARK_GEMMA4_MODULE_TAG,"SPARK_GEMMA4_STAGE_TP_TIMEOUT_MS",1u,UINT32_MAX,SPARK_GEMMA4_MODULE_TP_TIMEOUT_MILLI_DEFAULT,&state->tp_connect_timeout_milli);
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
				status = SparkStageModuleEnvironmentText(SPARK_GEMMA4_MODULE_TAG,"SPARK_GEMMA4_STAGE_TP_SESSION_PORTS",&tp_session_ports);
				if ( status != SPARK_STATUS_OK )
					return(status);
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
	}
	status = SparkStageModuleEnvironmentUnsigned(SPARK_GEMMA4_MODULE_TAG,"SPARK_GEMMA4_STAGE_COUNT",1u,SPARK_GEMMA4_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT,&state->stage_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_GEMMA4_MODULE_TAG,"SPARK_GEMMA4_STAGE_INDEX",0u,SPARK_GEMMA4_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT - 1u,&state->stage_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_GEMMA4_MODULE_TAG,"SPARK_GEMMA4_STAGE_FIRST_LAYER",0u,SPARK_GEMMA4_MODEL_LAYER_COUNT - 1u,&state->first_layer_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_GEMMA4_MODULE_TAG,"SPARK_GEMMA4_STAGE_LAYER_COUNT",1u,SPARK_GEMMA4_MODEL_LAYER_COUNT,&state->layer_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_GEMMA4_MODULE_TAG,"SPARK_GEMMA4_STAGE_MAX_ACTIVE_SEQUENCES",1u,SPARK_GEMMA4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,&state->max_active_sequence_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_GEMMA4_MODULE_TAG,"SPARK_GEMMA4_STAGE_PIPELINE_SLOTS",1u,SPARK_GEMMA4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,&state->pipeline_slot_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_GEMMA4_MODULE_TAG,"SPARK_GEMMA4_STAGE_KV_BLOCKS",1u,1u << 20u,&state->kv_block_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( state->stage_index >= state->stage_count || state->first_layer_index + state->layer_count > SPARK_GEMMA4_MODEL_LAYER_COUNT )
	{
		fprintf(stderr,"%s config_slice_invalid stage=%u/%u slice=%u+%u\n",SPARK_GEMMA4_MODULE_TAG,state->stage_index,state->stage_count,state->first_layer_index,state->layer_count);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	state->owns_embedding = state->first_layer_index == 0u ? 1u : 0u;
	state->owns_final_head = state->first_layer_index + state->layer_count == SPARK_GEMMA4_MODEL_LAYER_COUNT ? 1u : 0u;
	if ( (state->stage_index == 0u) != (state->owns_embedding != 0u) || (state->stage_index + 1u == state->stage_count) != (state->owns_final_head != 0u) )
	{
		fprintf(stderr,"%s config_position_mismatch stage=%u/%u slice=%u+%u\n",SPARK_GEMMA4_MODULE_TAG,state->stage_index,state->stage_count,state->first_layer_index,state->layer_count);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	return(SPARK_STATUS_OK);
}

#define SPARK_PACK_LOAD_FN(name) SparkGemma4Module##name
#define SPARK_PACK_LOAD_TYPE(name) SparkGemma4##name
#define SPARK_PACK_LOAD_CONST(name) SPARK_GEMMA4_##name
#define SPARK_PACK_LOAD_LAYER_IS_GDN(layer) (SPARK_GEMMA4_MODEL_LAYER_IS_FULL(layer) == 0u)
#define SPARK_PACK_LOAD_SEEN_TYPE uint64_t
#define SPARK_PACK_LOAD_SEEN_ONE 1ull
#define SPARK_PACK_LOAD_SEEN_FORMAT "%016llx"
#define SPARK_PACK_LOAD_SEEN_ARG(value) ((unsigned long long)(value))
#define SPARK_PACK_LOAD_BYTES_MATCH(entry) \
	((entry)->payload_bytes == SparkGemma4StagePackPayloadBytes((entry)->weight_format,(entry)->rows,(entry)->columns) && (entry)->scale_bytes == SparkGemma4StagePackScaleBytes((entry)->weight_format,(entry)->rows,(entry)->columns))
#define SPARK_PACK_LOAD_EXPECT_GEOMETRY(state,expected) SparkGemma4StagePackExpectedGeometry((expected),(state)->first_layer_index,(state)->layer_count)
#define SPARK_PACK_LOAD_GEOMETRY_MISMATCH(state,header,expected) (SparkGemma4StagePackHeaderMatches((header),(expected)) != 0 || (header)->directory_offset != SPARK_GEMMA4_STAGEPACK_HEADER_BYTES)
#define SPARK_PACK_LOAD_LOG_GEOMETRY_MISMATCH(state,header,expected) \
	fprintf(stderr,"%s pack_geometry_mismatch code=%d slice=%u+%u tensor_count=%u/%u hidden=%u/%u layers=%u/%u first=%u/%u total=%u/%u experts=%u/%u dir=%llu\n", \
		SPARK_GEMMA4_MODULE_TAG, \
		SparkGemma4StagePackHeaderMatches((header),(expected)), \
		(state)->first_layer_index,(state)->layer_count, \
		(header)->tensor_count,(expected)->tensor_count, \
		(header)->hidden_dimension,(expected)->hidden_dimension, \
		(header)->layer_count,(expected)->layer_count, \
		(header)->first_layer_index,(expected)->first_layer_index, \
		(header)->total_layer_count,(expected)->total_layer_count, \
		(header)->routed_expert_count,(expected)->routed_expert_count, \
		(unsigned long long)(header)->directory_offset)
#define SPARK_PACK_LOAD_PREFLIGHT(state,file,header,status) do {} while (0)

#define SPARK_GEMMA4_RESIDENT_DECODE_STAGE_LAYER_COUNT SPARK_GEMMA4_MODEL_LAYER_COUNT
#define gdn_ordinal_by_layer sliding_ordinal_by_layer
#define attn_ordinal_by_layer full_ordinal_by_layer
#define gdn_layer_count sliding_layer_count
#define attn_layer_count full_layer_count

#include "sparkpipe/spark_pack_load_common.h"

#undef gdn_ordinal_by_layer
#undef attn_ordinal_by_layer
#undef gdn_layer_count
#undef attn_layer_count

static SparkStatus SparkGemma4ModuleValidateEntry(SparkGemma4ModuleState *state, const SparkGemma4StagePackEntry *entry, uint64_t file_bytes, uint32_t *is_global)
{
	SparkGemma4StagePackTensorShape shape;
	uint32_t global = entry->layer_index == SPARK_GEMMA4_STAGEPACK_GLOBAL_LAYER ? 1u : 0u;
	if ( SparkGemma4StagePackResolvedShape(entry->tensor_kind,global != 0u ? 0u : entry->layer_index,global,&shape) != 0 )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->weight_format != shape.natural_format )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->scale_group_size != 0u )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	{
		SparkGemma4StagePackTensorShape narrow = shape;
		SparkGemma4StagePackNarrowShape(&narrow,entry->tensor_kind,state->tp_degree,state->tp_rank);
		if ( entry->rows != narrow.rows || entry->columns != narrow.columns )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	}
	return(SparkGemma4ModuleValidateEntryPlacement(state,entry,file_bytes,is_global));
}

static SparkStatus SparkGemma4ModuleBindMoe(SparkGemma4ModuleState *state, const SparkGemma4StagePackEntry *entry, void *payload, void *scale)
{
#if SPARK_GEMMA4_MODEL_MOE_BLOCK
	uint32_t layer = entry->layer_index;
	(void)scale;
	switch ( entry->tensor_kind )
	{
	case SPARK_GEMMA4_STAGEPACK_TENSOR_ROUTER_PROJ: SparkGemma4ModuleFillLinearView(&state->moe_by_layer[layer].router_proj,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_PER_EXPERT_SCALE: state->moe_by_layer[layer].per_expert_scale_f32 = (const float *)payload; return(SPARK_STATUS_OK);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_EXPERT_GATE_UP: SparkGemma4ModuleFillLinearView(&state->moe_by_layer[layer].experts_gate_up,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_EXPERT_DOWN: SparkGemma4ModuleFillLinearView(&state->moe_by_layer[layer].experts_down,entry,payload,scale); return(SPARK_STATUS_OK);
	default:
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	}
#else
	(void)state;
	(void)entry;
	(void)payload;
	(void)scale;
	SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
#endif
}

static SparkStatus SparkGemma4ModuleBindMtp(SparkGemma4ModuleState *state, const SparkGemma4StagePackEntry *entry, void *payload, void *scale)
{
	(void)state;
	(void)entry;
	(void)payload;
	(void)scale;
	SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
}

static SparkStatus SparkGemma4ModuleBindGlobal(SparkGemma4ModuleState *state, const SparkGemma4StagePackEntry *entry, void *payload)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_GEMMA4_STAGEPACK_TENSOR_EMBEDDING:
		if ( state->owns_embedding == 0u )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
		state->token_embedding_bf16 = payload;
		return(SPARK_STATUS_OK);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_FINAL_NORM:
		if ( state->owns_final_head == 0u )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
		state->final_norm_weight_bf16 = payload;
		return(SPARK_STATUS_OK);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_ROPE_TABLE:
		state->full_rope_table_f32 = payload;
		return(SPARK_STATUS_OK);
	default:
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	}
}

static SparkStatus SparkGemma4ModuleBindLayer(SparkGemma4ModuleState *state, const SparkGemma4StagePackEntry *entry, void *payload, void *scale)
{
	uint32_t layer = entry->layer_index;
	SparkGemma4SlidingLayerWeights *sliding = &state->sliding_by_layer[layer];
	SparkGemma4FullLayerWeights *full = &state->full_by_layer[layer];
	switch ( entry->tensor_kind )
	{
	case SPARK_GEMMA4_STAGEPACK_TENSOR_LAYER_INPUT_NORM: state->layer_input_norm_by_layer[layer] = payload; return(SPARK_STATUS_OK);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_LAYER_POST_ATTENTION_NORM: state->layer_post_attention_norm_by_layer[layer] = payload; return(SPARK_STATUS_OK);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_LAYER_PRE_FEEDFORWARD_NORM: state->layer_pre_feedforward_norm_by_layer[layer] = payload; return(SPARK_STATUS_OK);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_LAYER_POST_FEEDFORWARD_NORM: state->layer_post_feedforward_norm_by_layer[layer] = payload; return(SPARK_STATUS_OK);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_MLP_GATE_UP: SparkGemma4ModuleFillLinearView(&state->mlp_by_layer[layer].gate_up,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_MLP_DOWN: SparkGemma4ModuleFillLinearView(&state->mlp_by_layer[layer].down,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_SLIDING_QUERY: SparkGemma4ModuleFillLinearView(&sliding->query,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_SLIDING_KV_FUSED: SparkGemma4ModuleFillLinearView(&sliding->kv_fused,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_SLIDING_OUTPUT: SparkGemma4ModuleFillLinearView(&sliding->output,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_SLIDING_QUERY_NORM: sliding->query_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_SLIDING_KEY_NORM: sliding->key_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_QUERY: SparkGemma4ModuleFillLinearView(&full->query,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_KEY: SparkGemma4ModuleFillLinearView(&full->key,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_OUTPUT: SparkGemma4ModuleFillLinearView(&full->output,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_QUERY_NORM: full->query_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_KEY_NORM: full->key_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	default:
		return(SparkGemma4ModuleBindMoe(state,entry,payload,scale));
	}
}

static uint64_t SparkGemma4ModuleExpectedGlobalBits(const SparkGemma4ModuleState *state)
{
	uint64_t bits = (1ull << SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_ROPE_TABLE);
	if ( state->owns_embedding != 0u )
		bits |= 1ull << SPARK_GEMMA4_STAGEPACK_TENSOR_EMBEDDING;
	if ( state->owns_final_head != 0u )
		bits |= 1ull << SPARK_GEMMA4_STAGEPACK_TENSOR_FINAL_NORM;
	return(bits);
}

static uint64_t SparkGemma4ModuleExpectedMtpBits(void)
{
	return(0ull);
}

static uint64_t SparkGemma4ModuleExpectedLayerBits(const SparkGemma4ModuleState *state, uint32_t layer)
{
	uint64_t bits = (1ull << SPARK_GEMMA4_STAGEPACK_TENSOR_LAYER_INPUT_NORM) | (1ull << SPARK_GEMMA4_STAGEPACK_TENSOR_LAYER_POST_ATTENTION_NORM) | (1ull << SPARK_GEMMA4_STAGEPACK_TENSOR_LAYER_PRE_FEEDFORWARD_NORM) | (1ull << SPARK_GEMMA4_STAGEPACK_TENSOR_LAYER_POST_FEEDFORWARD_NORM) | (1ull << SPARK_GEMMA4_STAGEPACK_TENSOR_MLP_GATE_UP) | (1ull << SPARK_GEMMA4_STAGEPACK_TENSOR_MLP_DOWN);
#if SPARK_GEMMA4_MODEL_MOE_BLOCK
	(void)state;
	bits |= (1ull << SPARK_GEMMA4_STAGEPACK_TENSOR_ROUTER_PROJ) | (1ull << SPARK_GEMMA4_STAGEPACK_TENSOR_PER_EXPERT_SCALE) | (1ull << SPARK_GEMMA4_STAGEPACK_TENSOR_EXPERT_GATE_UP) | (1ull << SPARK_GEMMA4_STAGEPACK_TENSOR_EXPERT_DOWN);
#else
	(void)state;
#endif
	if ( SPARK_GEMMA4_MODEL_LAYER_IS_FULL(layer) != 0u )
		bits |= (1ull << SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_QUERY) | (1ull << SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_KEY) | (1ull << SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_OUTPUT) | (1ull << SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_QUERY_NORM) | (1ull << SPARK_GEMMA4_STAGEPACK_TENSOR_FULL_KEY_NORM);
	else
		bits |= (1ull << SPARK_GEMMA4_STAGEPACK_TENSOR_SLIDING_QUERY) | (1ull << SPARK_GEMMA4_STAGEPACK_TENSOR_SLIDING_KV_FUSED) | (1ull << SPARK_GEMMA4_STAGEPACK_TENSOR_SLIDING_OUTPUT) | (1ull << SPARK_GEMMA4_STAGEPACK_TENSOR_SLIDING_QUERY_NORM) | (1ull << SPARK_GEMMA4_STAGEPACK_TENSOR_SLIDING_KEY_NORM);
	return(bits);
}

static SparkStatus SparkGemma4ModuleAllocatePools(SparkGemma4ModuleState *state);
static SparkStatus SparkGemma4ModuleAllocateSlot(SparkGemma4ModuleState *state, SparkGemma4ModuleSlot *slot);
static SparkStatus SparkGemma4ModuleAllocateSlotHostMirrors(SparkGemma4ModuleState *state, SparkGemma4ModuleSlot *slot);
static SparkStatus SparkGemma4ModuleExecuteFrame(void *module_state, SparkModelDriverFrame *frame);

extern cudaError_t SparkGemma4LaunchTpCombineAdd(cudaStream_t stream, void *destination_bf16, const void *source_bf16, uint32_t row_count, uint32_t width);
extern cudaError_t SparkGemma4LaunchTpCombineU64Max(cudaStream_t stream, uint64_t *destination, const uint64_t *source, uint32_t element_count);

static SparkStatus SparkGemma4ModuleOpenKvTier(SparkGemma4ModuleState *state, const SparkFirmwareModuleHostServices *host_services)
{
	const char *provider;
	(void)state;
	static const char *none = "none";
	(void)host_services;
	provider = getenv("SPARK_GEMMA4_STAGE_KV_STORE");
	if ( provider == 0 )
		provider = none;
	if ( strcmp(provider,"none") != 0 )
	{
		fprintf(stderr,"%s kv_provider_unsupported provider=%s (external kv tier lands with the real-pack arm)\n",SPARK_GEMMA4_MODULE_TAG,provider);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGemma4ModuleTpCombineBf16(void *combine_context, void *destination_device, const void *source_device, uint32_t active_sequence_count, uint32_t hidden_dimension, void *cuda_stream)
{
	(void)combine_context;
	return(SparkStageModuleCudaStatus(SPARK_GEMMA4_MODULE_TAG,SparkGemma4LaunchTpCombineAdd((cudaStream_t)cuda_stream,destination_device,source_device,active_sequence_count,hidden_dimension),"tp_combine"));
}

static SparkStatus SparkGemma4ModuleTpCombineU64Max(void *combine_context, uint64_t *destination_device, const uint64_t *source_device, uint32_t element_count, void *cuda_stream)
{
	(void)combine_context;
	return(SparkStageModuleCudaStatus(SPARK_GEMMA4_MODULE_TAG,SparkGemma4LaunchTpCombineU64Max((cudaStream_t)cuda_stream,destination_device,source_device,element_count),"tp_combine_u64_max"));
}

static void SparkGemma4ModuleTpCompletion(void *context, const SparkTpDeviceCollectiveCompletion *completion)
{
	atomic_uint *flag = (atomic_uint *)context;
	atomic_store_explicit(flag,completion != 0 && completion->status == SPARK_STATUS_OK ? 1u : 2u,memory_order_release);
}

static SparkStatus SparkGemma4ModuleInitializeTpCollective(SparkGemma4ModuleState *state)
{
	SparkTpDeviceCollectiveConfig configuration;
	SparkTpDeviceCollectiveTopology topology;
	uint32_t rank;
	SparkStatus status;
	if ( state->tp_degree == 1u )
		return(SPARK_STATUS_OK);
	if ( state->tp_standalone != 0u )
	{
		fprintf(stderr,"%s tp_standalone degree=%u rank=%u (collective skipped; embedding/head results stay rank-partial)\n",
			SPARK_GEMMA4_MODULE_TAG,state->tp_degree,state->tp_rank);
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
	configuration.local_hidden_dimension = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
	configuration.max_active_sequence_count = SPARK_GEMMA4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT;
	configuration.connect_timeout_milli = state->tp_connect_timeout_milli;
	configuration.operation_timeout_milli = state->tp_operation_timeout_milli;
	configuration.control_port_base = state->tp_control_port_base;
	configuration.collective_identifier = state->tp_collective_identifier;
	configuration.backend_module_path = state->tp_backend_path;
	configuration.local_host = state->tp_local_host;
	configuration.registration_cuda_stream = state->slots[0].cuda_stream;
	configuration.combine_bf16_function = SparkGemma4ModuleTpCombineBf16;
	configuration.combine_u64_max_function = SparkGemma4ModuleTpCombineU64Max;
	configuration.combine_context = state;
	status = SparkTpDeviceCollectiveApplyTopology(&topology,&configuration);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s tp_apply_topology_failed status=%d\n",SPARK_GEMMA4_MODULE_TAG,(int)status);
		return(status);
	}
	if ( configuration.connect_timeout_milli == 0u || configuration.operation_timeout_milli == 0u || configuration.collective_identifier == 0u || configuration.backend_kind != SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkTpDeviceCollectiveCreate(&configuration,&state->tp_device_collective);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s tp_create_failed status=%d\n",SPARK_GEMMA4_MODULE_TAG,(int)status);
		return(status);
	}
	state->tp_collective_initialized = 1u;
	fprintf(stderr,"%s tp_collective_open degree=%u rank=%u port_base=%u\n",SPARK_GEMMA4_MODULE_TAG,state->tp_degree,state->tp_rank,state->tp_control_port_base);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGemma4ModuleTpSubmitOrdered(SparkGemma4ModuleState *state, void *device_buffer, uint32_t count, SparkGemma4ModuleSlot *slot, uint32_t u64_max)
{
	SparkTpDeviceCollectiveSubmission submission;
	struct timespec pause;
	uint32_t polls,flag;
	SparkStatus status;
	if ( state->tp_degree == 1u || state->tp_standalone != 0u )
		return(SPARK_STATUS_OK);
	if ( state->tp_collective_initialized == 0u )
		return(SPARK_STATUS_INTERNAL_ERROR);
	atomic_store_explicit(&state->tp_completion_flag,0u,memory_order_relaxed);
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = 0u;
	submission.active_sequence_count = count;
	submission.logical_sequence_count = slot->logical_sequence_count;
	submission.flags = SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission.ordinal = atomic_fetch_add_explicit(&state->tp_next_ordinal,1u,memory_order_relaxed);
	submission.local_device = device_buffer;
	submission.full_device = device_buffer;
	submission.cuda_stream = slot->cuda_stream;
	submission.completion_function = SparkGemma4ModuleTpCompletion;
	submission.completion_context = &state->tp_completion_flag;
	status = u64_max != 0u
		? SparkTpDeviceCollectiveSubmitU64Max(&state->tp_device_collective,&submission)
		: SparkTpDeviceCollectiveSubmitBf16(&state->tp_device_collective,&submission);
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
	fprintf(stderr,"%s tp_all_reduce_stall\n",SPARK_GEMMA4_MODULE_TAG);
	return(SPARK_STATUS_IO_ERROR);
}

static SparkStatus SparkGemma4ModuleTpAllReduceHidden(SparkGemma4ModuleState *state, SparkGemma4ModuleSlot *slot, void *device_bf16, uint32_t rows)
{
	if ( state->tp_degree == 1u || state->tp_standalone != 0u )
		return(SPARK_STATUS_OK);
	return(SparkGemma4ModuleTpSubmitOrdered(state,device_bf16,rows,slot,0u));
}

static SparkStatus SparkGemma4ModuleTpReduceU64Max(SparkGemma4ModuleState *state, SparkGemma4ModuleSlot *slot, uint64_t *device_u64, uint32_t count)
{
	if ( state->tp_degree == 1u || state->tp_standalone != 0u )
		return(SPARK_STATUS_OK);
	return(SparkGemma4ModuleTpSubmitOrdered(state,device_u64,count,slot,1u));
}


static SparkStatus SparkGemma4ModuleInitializeGate(void)
{
	uint32_t allow_unqualified_execution;
	allow_unqualified_execution = 0u;
	if ( SparkStageModuleEnvironmentUnsigned(SPARK_GEMMA4_MODULE_TAG,"SPARK_GEMMA4_ALLOW_UNQUALIFIED_EXECUTION",1u,1u,&allow_unqualified_execution) != SPARK_STATUS_OK || allow_unqualified_execution != 1u )
		return(SPARK_STATUS_MODULE_NOT_VALIDATED);
	return(SPARK_STATUS_OK);
}

static void SparkGemma4ModuleDescribe(void *module_state, SparkStageModuleLifecycle *lifecycle)
{
	SparkGemma4ModuleState *state = (SparkGemma4ModuleState *)module_state;
	lifecycle->module_tag = SPARK_GEMMA4_MODULE_TAG;
	lifecycle->ledger = &state->ledger;
	lifecycle->slot_states = state->slot_states;
	lifecycle->pipeline_slot_count = state->pipeline_slot_count;
	lifecycle->submitted_count = &state->submitted_count;
	lifecycle->completed_count = &state->completed_count;
	lifecycle->rejected_count = &state->rejected_count;
	lifecycle->failed_count = &state->failed_count;
	lifecycle->tokens_emitted = &state->tokens_emitted;
}

extern cudaError_t SparkGemma4ConfigureCudaKernels(void);
extern cudaError_t SparkGemma4LaunchEmbeddingGatherShardedScaled(cudaStream_t stream, const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count, uint32_t vocab_base, uint32_t vocab_rows);
extern cudaError_t SparkGemma4LaunchRmsNorm(cudaStream_t stream, const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon);
extern cudaError_t SparkGemma4LaunchHeadRmsNorm(cudaStream_t stream, const void *input_bf16, const void *weight_or_null_bf16, void *output_bf16, uint32_t row_count, uint32_t heads, uint32_t head_dimension, float epsilon);
extern cudaError_t SparkGemma4LaunchLinear(cudaStream_t stream, const SparkGemma4LinearView *view, const void *input_bf16, void *output_bf16, uint32_t row_count);
extern cudaError_t SparkGemma4LaunchResidualAdd(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, uint32_t row_count, uint32_t dimension);
extern cudaError_t SparkGemma4LaunchBranchAdd(cudaStream_t stream, void *sum_bf16, const void *delta_bf16, uint32_t row_count, uint32_t dimension);
extern cudaError_t SparkGemma4LaunchSlidingRope(cudaStream_t stream, void *q_bf16, const uint32_t *positions, uint32_t row_count, uint32_t heads, uint32_t head_dimension, uint32_t rope_dimension, float theta);
extern cudaError_t SparkGemma4LaunchFullRope(cudaStream_t stream, void *q_bf16, const uint32_t *positions, const float *inv_freq_table, uint32_t row_count, uint32_t heads, uint32_t head_dimension, uint32_t rope_dimension);
extern cudaError_t SparkGemma4LaunchSlidingWindowPositions(cudaStream_t stream, const uint32_t *sequence_of_row, const uint32_t *context_lengths, const uint32_t *row_positions, uint32_t row_count, uint32_t *positions_out);
extern cudaError_t SparkGemma4LaunchKvStoreSliding(cudaStream_t stream, void *pool, const uint32_t *page_table, uint32_t page_table_stride, uint32_t sequence_count, uint32_t pool_page_count, void *access_error, const void *key_bf16, const void *value_bf16, const uint32_t *sequence_of_row, const uint32_t *positions, uint32_t row_count, uint32_t kv_heads);
extern cudaError_t SparkGemma4LaunchKvStoreFull(cudaStream_t stream, void *pool, const uint32_t *page_table, uint32_t page_table_stride, uint32_t sequence_count, uint32_t pool_page_count, void *access_error, const void *key_bf16, const void *value_bf16, const uint32_t *sequence_of_row, const uint32_t *positions, uint32_t row_count);
extern cudaError_t SparkGemma4LaunchAttentionDecodeSliding(cudaStream_t stream, void *pool, const uint32_t *page_table, uint32_t page_table_stride, uint32_t sequence_count, uint32_t pool_page_count, void *access_error, const void *query_bf16, const uint32_t *sequence_of_row, const uint32_t *context_lengths, const uint32_t *window_positions, uint32_t query_heads, void *output_bf16, uint32_t row_count, uint32_t kv_heads);
extern cudaError_t SparkGemma4LaunchAttentionDecodeFull(cudaStream_t stream, void *pool, const uint32_t *page_table, uint32_t page_table_stride, uint32_t sequence_count, uint32_t pool_page_count, void *access_error, const void *query_bf16, const uint32_t *sequence_of_row, const uint32_t *context_lengths, uint32_t query_heads, void *output_bf16, uint32_t row_count);
extern cudaError_t SparkGemma4LaunchGatedGelu(cudaStream_t stream, void *gate_up_bf16, uint32_t row_count, uint32_t intermediate);
extern cudaError_t SparkGemma4LaunchHeadMaxLocPack(cudaStream_t stream, const float *scores_f32, const uint32_t *token_ids_u32, uint64_t *keys_u64, uint32_t row_count);
extern cudaError_t SparkGemma4LaunchHeadMaxLocUnpack(cudaStream_t stream, const uint64_t *keys_u64, uint32_t *token_ids_u32, uint32_t row_count);
extern cudaError_t SparkGemma4LaunchHeadDirectArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, uint32_t *output_token_ids, float *output_scores, uint32_t candidate_offset, uint32_t row_count, uint32_t candidate_count);
#if SPARK_GEMMA4_MODEL_MOE_BLOCK
extern cudaError_t SparkGemma4LaunchRouterSoftmax(cudaStream_t stream, float *scores_f32, uint32_t row_count);
extern cudaError_t SparkGemma4LaunchRouterTopk(cudaStream_t stream, const float *scores_f32, uint32_t *indices_u32, float *weights_f32, uint32_t row_count);
extern cudaError_t SparkGemma4LaunchMoeRoute(cudaStream_t stream, const uint32_t *route_expert, uint32_t rows, uint32_t expert_width, uint32_t *group_row_offset, uint32_t *route_packed_row, uint32_t *route_source_token, uint32_t *group_tile_prefix_w1, uint32_t *group_tile_prefix_w2);
extern cudaError_t SparkGemma4LaunchGateScores(cudaStream_t stream, const SparkGemma4LinearView *gate, const void *input_bf16, float *scores_f32, uint32_t row_count);
extern cudaError_t SparkGemma4LaunchGroupedExpertLinear(cudaStream_t stream, const SparkGemma4LinearView *view, const void *input_bf16, const uint32_t *source_row_map, const uint32_t *group_row_offset, const uint32_t *group_tile_prefix, void *output_bf16, uint32_t source_row_count, uint32_t multiprocessor_count, uint32_t tp_degree, uint32_t tp_rank, uint32_t route_group_base, const void *frame_error);
extern cudaError_t SparkGemma4LaunchMoePairReduceOverwrite(cudaStream_t stream, const void *slot_out_bf16, const uint32_t *inverse_map, const float *pair_weights_f32, void *output_bf16, uint32_t row_count, uint32_t hidden_dimension);
#endif

static SparkStatus SparkGemma4ModulePrepare(
	void *module_state,
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services)
{
	SparkGemma4ModuleState *state = (SparkGemma4ModuleState *)module_state;
	const char *pack_path;
	SparkStatus status;
	(void)configuration;
	pack_path = 0;
	state->allow_unqualified_execution = 1u;
	atomic_init(&state->tp_completion_flag,0u);
	atomic_init(&state->tp_next_ordinal,0u);
	status = SparkGemma4ModuleConfigure(state);
	if ( status == SPARK_STATUS_OK )
		SparkStageModuleAtomicStateArrayInitialize(state->slot_states,state->pipeline_slot_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentText(SPARK_GEMMA4_MODULE_TAG,"SPARK_GEMMA4_STAGE_PACK_PATH",&pack_path);
	if ( status == SPARK_STATUS_OK )
	{
		SparkGemma4ModuleBuildOrdinals(state);
		status = SparkGemma4ModuleLoadPack(state,pack_path);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkGemma4ModuleOpenKvTier(state,host_services);
	if ( status == SPARK_STATUS_OK )
	{
		int32_t sm_count = 0;
		cudaError_t attr = cudaDeviceGetAttribute(&sm_count,cudaDevAttrMultiProcessorCount,0);
		state->multiprocessor_count = attr == cudaSuccess && sm_count > 0 ? (uint32_t)sm_count : 1u;
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_GEMMA4_MODULE_TAG,SparkGemma4ConfigureCudaKernels(),"configure_cuda_kernels");
	if ( status == SPARK_STATUS_OK )
		status = SparkGemma4ModuleAllocatePools(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkGemma4ModuleAllocateSlot(state,&state->slots[0]);
	if ( status == SPARK_STATUS_OK )
		status = SparkGemma4ModuleAllocateSlotHostMirrors(state,&state->slots[0]);
	if ( status == SPARK_STATUS_OK && state->tp_degree > 1u )
		status = SparkGemma4ModuleInitializeTpCollective(state);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"%s initialize_failed status=%d\n",SPARK_GEMMA4_MODULE_TAG,(int)status);
	return(status);
}

static void SparkGemma4ModuleReportReady(void *module_state)
{
	SparkGemma4ModuleState *state = (SparkGemma4ModuleState *)module_state;
	fprintf(stderr,"%s initialize ok slice=%u+%u sliding=%u full=%u tp=%u/%u owns_embedding=%u owns_head=%u\n",SPARK_GEMMA4_MODULE_TAG,state->first_layer_index,state->layer_count,state->sliding_layer_count,state->full_layer_count,state->tp_rank,state->tp_degree,state->owns_embedding,state->owns_final_head);
}

static void SparkGemma4AdmissionCost(
	void *context,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	SparkGemma4ModuleState *state = (SparkGemma4ModuleState *)context;
	decision->host_staging_bytes = (uint64_t)request->new_token_count *
		(sizeof(uint32_t) *
			 (uint64_t)(state->owns_embedding + state->owns_final_head + 3u) +
		 sizeof(uint64_t));
	decision->device_memcpy_bytes = decision->host_staging_bytes;
}

static SparkStatus SparkGemma4AdmissionKvPredicate(
	void *context,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	(void)context;
	if ((request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u &&
		SparkModelDriverRangeFitsWithinCapacity(
			request->sequence_position,
			request->new_token_count,
			SPARK_GEMMA4_MODEL_MAXIMUM_CONTEXT_TOKENS) == 0u)
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
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGemma4ModuleAdmit(
	void *module_state,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	SparkGemma4ModuleState *state;
	SparkAdmissionPolicyTable table;
	uint32_t available_slot_count;
	SparkStatus status;
	state = (SparkGemma4ModuleState *)module_state;
	available_slot_count = SparkStageModuleSlotCountFree(
		state->slot_states,
		state->pipeline_slot_count);
	memset(&table,0,sizeof(table));
	table.abi_version = SPARK_ADMISSION_ABI_VERSION;
	table.descriptor_bytes = (uint32_t)sizeof(table);
	table.max_active_sequence_count = state->max_active_sequence_count;
	table.max_input_row_count = state->max_active_sequence_count;
	table.max_sequence_positions = SPARK_GEMMA4_MODEL_MAXIMUM_CONTEXT_TOKENS;
	table.flags = SPARK_ADMISSION_POLICY_FLAG_PREFILL_SINGLE_SLOT |
		SPARK_ADMISSION_POLICY_FLAG_DECODE_EQUALS_SLOTS;
	table.predicate = SparkGemma4AdmissionKvPredicate;
	table.predicate_context = state;
	table.cost = SparkGemma4AdmissionCost;
	table.cost_context = state;
	status = SparkAdmissionEvaluateShape(&table,available_slot_count,request,decision);
	if (status != SPARK_STATUS_OK)
		return(status);
	if (decision->accepted == 0u)
		atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
	return(status);
}

static void SparkGemma4ModuleSnapshotExtend(
	void *module_state,
	SparkModelDriverRuntimeSnapshot *snapshot)
{
	SparkGemma4ModuleState *state = (SparkGemma4ModuleState *)module_state;
	snapshot->kv_token_capacity = (uint64_t)state->kv_block_count * SPARK_GEMMA4_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
}

static void SparkGemma4ModuleStateTeardown(void *module_state)
{
	SparkGemma4ModuleState *state = (SparkGemma4ModuleState *)module_state;
	uint32_t slot_index;
	if ( state->tp_collective_initialized != 0u )
		SparkTpDeviceCollectiveDestroy(&state->tp_device_collective);
	for (slot_index = 0u; slot_index < SPARK_GEMMA4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT; slot_index++)
	{
		free(state->slots[slot_index].host_row_lane_indices);
		free(state->slots[slot_index].host_row_positions);
		free(state->slots[slot_index].host_row_positions_u32);
		free(state->slots[slot_index].host_row_sequences_u32);
		free(state->slots[slot_index].host_slot_mapping);
		free(state->slots[slot_index].host_context_lengths);
		free(state->slots[slot_index].host_frame_error);
	}
}

static const SparkStageModuleLifecycleOps SparkGemma4ModuleLifecycle =
{
	sizeof(SparkGemma4ModuleState),
	SparkGemma4ModuleInitializeGate,
	SparkGemma4ModuleDescribe,
	SparkGemma4ModulePrepare,
	SparkGemma4ModuleReportReady,
	SparkGemma4ModuleStateTeardown,
	SparkGemma4ModuleExecuteFrame,
	SparkGemma4ModuleAdmit,
	SparkGemma4ModuleSnapshotExtend
};

SPARK_STAGE_MODULE_LIFECYCLE_ENTRY_POINTS(
	SparkGemma4ResidentDecodeStage,
	&SparkGemma4ModuleLifecycle)

#define SPARK_GEMMA4_MODULE_HOST_ROW_CAPACITY \
	(SPARK_GEMMA4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT)

static SparkStatus SparkGemma4ModuleAllocatePools(SparkGemma4ModuleState *state)
{
	SparkStatus status = SPARK_STATUS_OK;
	uint64_t sliding_slot_bytes = (uint64_t)state->sliding_kv_heads_per_rank * (SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION + SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION) * SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES;
	uint64_t full_slot_bytes = (uint64_t)state->full_kv_heads_per_rank * (SPARK_GEMMA4_MODEL_FULL_HEAD_DIMENSION + SPARK_GEMMA4_MODEL_FULL_HEAD_DIMENSION) * SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES;
	if ( sliding_slot_bytes % 16u != 0u || full_slot_bytes % 16u != 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	state->sliding_kv_pool_bytes = (uint64_t)state->kv_block_count * SPARK_GEMMA4_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS * sliding_slot_bytes;
	state->full_kv_pool_bytes = (uint64_t)state->kv_block_count * SPARK_GEMMA4_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS * full_slot_bytes;
	status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,state->sliding_kv_pool_bytes,&state->sliding_kv_pool_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,state->full_kv_pool_bytes,&state->full_kv_pool_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,SPARK_FRAME_ERROR_WORDS * sizeof(uint32_t),&state->kv_access_error);
	return(status);
}

static SparkStatus SparkGemma4ModuleAllocateSlot(SparkGemma4ModuleState *state, SparkGemma4ModuleSlot *slot)
{
	uint64_t rows = state->max_active_sequence_count;
	uint64_t hidden_bytes = rows * SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION * SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES;
	uint64_t local_hidden_bytes = rows * (SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION / state->tp_degree) * SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES;
	uint64_t sliding_query_bytes = rows * ((SPARK_GEMMA4_MODEL_SLIDING_QUERY_HEAD_COUNT / state->tp_degree) * SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION) * SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES;
	uint64_t sliding_kv_bytes = rows * 2u * (state->sliding_kv_heads_per_rank * SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION) * SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES;
	uint64_t full_query_bytes = rows * ((SPARK_GEMMA4_MODEL_FULL_QUERY_HEAD_COUNT / state->tp_degree) * SPARK_GEMMA4_MODEL_FULL_HEAD_DIMENSION) * SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES;
	uint64_t full_kv_bytes = rows * (state->full_kv_heads_per_rank * SPARK_GEMMA4_MODEL_FULL_HEAD_DIMENSION) * SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES;
	uint64_t mlp_gate_up_bytes = rows * 2u * (SPARK_GEMMA4_MODEL_DENSE_INTERMEDIATE_DIMENSION / state->tp_degree) * SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES;
	SparkStatus status;
	cudaStream_t stream = 0;
	status = SparkStageModuleCudaStatus(SPARK_GEMMA4_MODULE_TAG,cudaStreamCreate(&stream),"cudaStreamCreate");
	if ( status != SPARK_STATUS_OK )
		return(status);
	slot->cuda_stream = stream;
	status = SparkStageModuleDeviceAllocate(&state->ledger,SPARK_FRAME_ERROR_WORDS * sizeof(uint32_t),(void **)&slot->frame_error);
	if ( status == SPARK_STATUS_OK )
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
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->row_positions_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->row_sequences_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_GEMMA4_MODEL_SLIDING_WINDOW_TOKENS * sizeof(uint32_t),(void **)&slot->window_positions_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,hidden_bytes,&slot->hidden_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,hidden_bytes,&slot->normalized_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,sliding_query_bytes,&slot->sliding_query_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,sliding_kv_bytes,&slot->sliding_kv_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,full_query_bytes,&slot->full_query_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,full_kv_bytes,&slot->full_key_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,full_kv_bytes,&slot->full_value_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,sliding_query_bytes,&slot->attn_head_output_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,local_hidden_bytes,&slot->attn_output_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,local_hidden_bytes,&slot->delta_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,mlp_gate_up_bytes,&slot->mlp_gate_up_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,local_hidden_bytes,&slot->mlp_down_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,local_hidden_bytes,&slot->branch_bf16);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
	{
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(float),(void **)&slot->argmax_score_f32);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->argmax_token_ids);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint64_t),(void **)&slot->head_maxloc_u64);
	}
#if SPARK_GEMMA4_MODEL_MOE_BLOCK
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_GEMMA4_MODEL_ROUTED_EXPERT_COUNT * sizeof(float),(void **)&slot->moe_scores_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_GEMMA4_MODEL_EXPERTS_PER_TOKEN * sizeof(uint32_t),(void **)&slot->moe_indices_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_GEMMA4_MODEL_EXPERTS_PER_TOKEN * sizeof(float),(void **)&slot->moe_weights_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)(rows * SPARK_GEMMA4_MODEL_EXPERTS_PER_TOKEN + SPARK_GEMMA4_MODEL_ROUTED_EXPERT_COUNT + 2u) * sizeof(uint32_t),(void **)&slot->moe_inverse_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)(rows * SPARK_GEMMA4_MODEL_EXPERTS_PER_TOKEN + SPARK_GEMMA4_MODEL_ROUTED_EXPERT_COUNT + 2u) * sizeof(uint32_t),(void **)&slot->moe_grouped_rows_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)(SPARK_GEMMA4_MODEL_ROUTED_EXPERT_COUNT + 1u) * sizeof(uint32_t),(void **)&slot->moe_tile_prefix_w1_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)(SPARK_GEMMA4_MODEL_ROUTED_EXPERT_COUNT + 1u) * sizeof(uint32_t),(void **)&slot->moe_tile_prefix_w2_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)(SPARK_GEMMA4_MODEL_ROUTED_EXPERT_COUNT + 1u) * sizeof(uint32_t),(void **)&slot->moe_group_offset_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_GEMMA4_MODEL_EXPERTS_PER_TOKEN * 2u * SPARK_GEMMA4_MODEL_EXPERT_INTERMEDIATE_DIMENSION * SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES,&slot->moe_gate_packed_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_GEMMA4_MODEL_EXPERTS_PER_TOKEN * SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION * SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES,&slot->moe_slot_out_bf16);
#endif
	return(status);
}

static SparkStatus SparkGemma4ModuleAllocateSlotHostMirrors(SparkGemma4ModuleState *state, SparkGemma4ModuleSlot *slot)
{
	(void)state;
	slot->host_row_lane_indices = (uint32_t *)malloc(SPARK_GEMMA4_MODULE_HOST_ROW_CAPACITY * sizeof(uint32_t));
	slot->host_row_positions = (uint64_t *)malloc(SPARK_GEMMA4_MODULE_HOST_ROW_CAPACITY * sizeof(uint64_t));
	slot->host_row_positions_u32 = (uint32_t *)malloc(SPARK_GEMMA4_MODULE_HOST_ROW_CAPACITY * sizeof(uint32_t));
	slot->host_row_sequences_u32 = (uint32_t *)malloc(SPARK_GEMMA4_MODULE_HOST_ROW_CAPACITY * sizeof(uint32_t));
	slot->host_slot_mapping = (uint32_t *)malloc(SPARK_GEMMA4_MODULE_HOST_ROW_CAPACITY * sizeof(uint32_t));
	slot->host_context_lengths = (uint32_t *)malloc(SPARK_GEMMA4_MODULE_HOST_ROW_CAPACITY * sizeof(uint32_t));
	slot->host_frame_error = (uint32_t *)malloc(SPARK_FRAME_ERROR_WORDS * sizeof(uint32_t));
	if ( slot->host_row_lane_indices == 0 || slot->host_row_positions == 0 ||
		slot->host_row_positions_u32 == 0 || slot->host_row_sequences_u32 == 0 ||
		slot->host_slot_mapping == 0 || slot->host_context_lengths == 0 ||
		slot->host_frame_error == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGemma4ModuleRunAttentionBody(SparkGemma4ModuleState *state, SparkGemma4ModuleSlot *slot, const SparkGemma4ResidentDecodeStageFrameContext *context, uint32_t layer, uint32_t rows, uint32_t is_full)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	const SparkGemma4KvBlockTableView *table = context->kv_block_table;
	uint32_t kv_heads = is_full != 0u ? state->full_kv_heads_per_rank : state->sliding_kv_heads_per_rank;
	uint32_t query_heads = is_full != 0u ? SPARK_GEMMA4_MODEL_FULL_QUERY_HEAD_COUNT / state->tp_degree : SPARK_GEMMA4_MODEL_SLIDING_QUERY_HEAD_COUNT / state->tp_degree;
	uint32_t head_dimension = is_full != 0u ? SPARK_GEMMA4_MODEL_FULL_HEAD_DIMENSION : SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION;
	void *pool = is_full != 0u ? state->full_kv_pool_bf16 : state->sliding_kv_pool_bf16;
	const uint32_t *page_table = table != 0 ? table->physical_block_indices : 0;
	uint32_t page_table_stride = table != 0 ? table->lane_stride : 0u;
	uint32_t sequence_count = table != 0 ? table->lane_capacity : state->max_active_sequence_count;
	uint32_t pool_page_count = state->kv_block_count;
	SparkStatus status;
	if ( is_full != 0u )
	{
		const SparkGemma4FullLayerWeights *weights = &state->full_by_layer[layer];
		uint32_t kv_per_rank = state->full_kv_heads_per_rank;
		cudaError_t error = SparkGemma4LaunchLinear(stream,&weights->query,slot->normalized_bf16,slot->full_query_bf16,rows);
		if ( error == cudaSuccess )
			error = SparkGemma4LaunchHeadRmsNorm(stream,slot->full_query_bf16,weights->query_norm_weight_bf16,slot->full_query_bf16,rows,query_heads,head_dimension,SPARK_GEMMA4_MODEL_RMS_NORM_EPSILON);
		if ( error == cudaSuccess )
			error = SparkGemma4LaunchFullRope(stream,slot->full_query_bf16,slot->row_positions_u32,state->full_rope_table_f32,rows,query_heads,head_dimension,SPARK_GEMMA4_MODEL_FULL_ROPE_DIMENSION);
		if ( error == cudaSuccess )
			error = SparkGemma4LaunchLinear(stream,&weights->key,slot->normalized_bf16,slot->full_key_bf16,rows);
		if ( error == cudaSuccess )
			error = SparkGemma4LaunchHeadRmsNorm(stream,slot->full_key_bf16,0,slot->full_value_bf16,rows,kv_per_rank,head_dimension,SPARK_GEMMA4_MODEL_RMS_NORM_EPSILON);
		if ( error == cudaSuccess )
			error = SparkGemma4LaunchHeadRmsNorm(stream,slot->full_key_bf16,weights->key_norm_weight_bf16,slot->full_key_bf16,rows,kv_per_rank,head_dimension,SPARK_GEMMA4_MODEL_RMS_NORM_EPSILON);
		if ( error == cudaSuccess )
			error = SparkGemma4LaunchFullRope(stream,slot->full_key_bf16,slot->row_positions_u32,state->full_rope_table_f32,rows,kv_per_rank,head_dimension,SPARK_GEMMA4_MODEL_FULL_ROPE_DIMENSION);
		if ( error == cudaSuccess )
			error = SparkGemma4LaunchKvStoreFull(stream,pool,page_table,page_table_stride,sequence_count,pool_page_count,state->kv_access_error,slot->full_key_bf16,slot->full_value_bf16,slot->row_sequences_u32,slot->row_positions_u32,rows);
		if ( error == cudaSuccess )
			error = SparkGemma4LaunchAttentionDecodeFull(stream,pool,page_table,page_table_stride,sequence_count,pool_page_count,state->kv_access_error,slot->full_query_bf16,slot->row_sequences_u32,slot->context_lengths,query_heads,slot->attn_head_output_bf16,rows);
		if ( error == cudaSuccess )
			error = SparkGemma4LaunchLinear(stream,&weights->output,slot->attn_head_output_bf16,slot->attn_output_bf16,rows);
	}
	else
	{
		const SparkGemma4SlidingLayerWeights *weights = &state->sliding_by_layer[layer];
		uint32_t kv_per_rank = state->sliding_kv_heads_per_rank;
		uint64_t kv_half_bytes = (uint64_t)rows * kv_per_rank * head_dimension * SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES;
		void *value_half = (void *)((uint8_t *)slot->sliding_kv_bf16 + kv_half_bytes);
		cudaError_t error = SparkGemma4LaunchLinear(stream,&weights->query,slot->normalized_bf16,slot->sliding_query_bf16,rows);
		if ( error == cudaSuccess )
			error = SparkGemma4LaunchHeadRmsNorm(stream,slot->sliding_query_bf16,weights->query_norm_weight_bf16,slot->sliding_query_bf16,rows,query_heads,head_dimension,SPARK_GEMMA4_MODEL_RMS_NORM_EPSILON);
		if ( error == cudaSuccess )
			error = SparkGemma4LaunchSlidingRope(stream,slot->sliding_query_bf16,slot->row_positions_u32,rows,query_heads,head_dimension,SPARK_GEMMA4_MODEL_SLIDING_ROPE_DIMENSION,SPARK_GEMMA4_MODEL_SLIDING_ROPE_THETA);
		if ( error == cudaSuccess )
			error = SparkGemma4LaunchLinear(stream,&weights->kv_fused,slot->normalized_bf16,slot->sliding_kv_bf16,rows);
		if ( error == cudaSuccess )
			error = SparkGemma4LaunchHeadRmsNorm(stream,slot->sliding_kv_bf16,weights->key_norm_weight_bf16,slot->sliding_kv_bf16,rows,kv_per_rank,head_dimension,SPARK_GEMMA4_MODEL_RMS_NORM_EPSILON);
		if ( error == cudaSuccess )
			error = SparkGemma4LaunchHeadRmsNorm(stream,value_half,0,value_half,rows,kv_per_rank,head_dimension,SPARK_GEMMA4_MODEL_RMS_NORM_EPSILON);
		if ( error == cudaSuccess )
			error = SparkGemma4LaunchSlidingRope(stream,slot->sliding_kv_bf16,slot->row_positions_u32,rows,kv_per_rank,head_dimension,SPARK_GEMMA4_MODEL_SLIDING_ROPE_DIMENSION,SPARK_GEMMA4_MODEL_SLIDING_ROPE_THETA);
		if ( error == cudaSuccess )
			error = SparkGemma4LaunchSlidingWindowPositions(stream,slot->row_sequences_u32,slot->context_lengths,slot->row_positions_u32,rows,slot->window_positions_u32);
		if ( error == cudaSuccess )
			error = SparkGemma4LaunchKvStoreSliding(stream,pool,page_table,page_table_stride,sequence_count,pool_page_count,state->kv_access_error,slot->sliding_kv_bf16,value_half,slot->row_sequences_u32,slot->row_positions_u32,rows,kv_heads);
		if ( error == cudaSuccess )
			error = SparkGemma4LaunchAttentionDecodeSliding(stream,pool,page_table,page_table_stride,sequence_count,pool_page_count,state->kv_access_error,slot->sliding_query_bf16,slot->row_sequences_u32,slot->context_lengths,slot->window_positions_u32,query_heads,slot->attn_head_output_bf16,rows,kv_heads);
		if ( error == cudaSuccess )
			error = SparkGemma4LaunchLinear(stream,&weights->output,slot->attn_head_output_bf16,slot->attn_output_bf16,rows);
	}
	if ( state->tp_degree > 1u )
	{
		status = SparkGemma4ModuleTpAllReduceHidden(state,slot,slot->attn_output_bf16,rows);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	status = SparkStageModuleCudaStatus(SPARK_GEMMA4_MODULE_TAG,cudaGetLastError(),"attention");
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkStageModuleCudaStatus(SPARK_GEMMA4_MODULE_TAG,SparkGemma4LaunchRmsNorm(stream,slot->attn_output_bf16,state->layer_post_attention_norm_by_layer[layer],slot->delta_bf16,rows,SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION,SPARK_GEMMA4_MODEL_RMS_NORM_EPSILON),"post_attention_norm");
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkStageModuleCudaStatus(SPARK_GEMMA4_MODULE_TAG,SparkGemma4LaunchResidualAdd(stream,slot->hidden_bf16,slot->delta_bf16,rows,SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION),"attention_residual"));
}

static SparkStatus SparkGemma4ModuleRunFeedForward(SparkGemma4ModuleState *state, SparkGemma4ModuleSlot *slot, uint32_t layer, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	const SparkGemma4DenseMlpWeights *mlp = &state->mlp_by_layer[layer];
	uint32_t local_intermediate = SPARK_GEMMA4_MODEL_DENSE_INTERMEDIATE_DIMENSION / state->tp_degree;
	SparkStatus status;
	status = SparkStageModuleCudaStatus(SPARK_GEMMA4_MODULE_TAG,SparkGemma4LaunchRmsNorm(stream,slot->hidden_bf16,state->layer_pre_feedforward_norm_by_layer[layer],slot->normalized_bf16,rows,SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION,SPARK_GEMMA4_MODEL_RMS_NORM_EPSILON),"pre_ff_norm");
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkStageModuleCudaStatus(SPARK_GEMMA4_MODULE_TAG,SparkGemma4LaunchLinear(stream,&mlp->gate_up,slot->normalized_bf16,slot->mlp_gate_up_bf16,rows),"mlp_gate_up");
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkStageModuleCudaStatus(SPARK_GEMMA4_MODULE_TAG,SparkGemma4LaunchGatedGelu(stream,slot->mlp_gate_up_bf16,rows,local_intermediate),"mlp_gelu");
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkStageModuleCudaStatus(SPARK_GEMMA4_MODULE_TAG,SparkGemma4LaunchLinear(stream,&mlp->down,slot->mlp_gate_up_bf16,slot->mlp_down_bf16,rows),"mlp_down");
	if ( status != SPARK_STATUS_OK )
		return(status);
#if SPARK_GEMMA4_MODEL_MOE_BLOCK
	{
		const SparkGemma4MoeLayerWeights *moe = &state->moe_by_layer[layer];
		cudaError_t error = SparkGemma4LaunchHeadRmsNorm(stream,slot->hidden_bf16,0,slot->branch_bf16,rows,1u,SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION,SPARK_GEMMA4_MODEL_RMS_NORM_EPSILON);
		if ( error == cudaSuccess )
			error = SparkGemma4LaunchGateScores(stream,&moe->router_proj,slot->branch_bf16,slot->moe_scores_f32,rows);
		if ( error == cudaSuccess )
			error = SparkGemma4LaunchRouterSoftmax(stream,slot->moe_scores_f32,rows);
		if ( error == cudaSuccess )
			error = SparkGemma4LaunchRouterTopk(stream,slot->moe_scores_f32,slot->moe_indices_u32,slot->moe_weights_f32,rows);
		if ( error == cudaSuccess )
			error = SparkGemma4LaunchMoeRoute(stream,slot->moe_indices_u32,rows,SPARK_GEMMA4_MODEL_EXPERT_INTERMEDIATE_DIMENSION,slot->moe_group_offset_u32,slot->moe_inverse_u32,slot->moe_grouped_rows_u32,slot->moe_tile_prefix_w1_u32,slot->moe_tile_prefix_w2_u32);
		if ( error == cudaSuccess )
			error = SparkGemma4LaunchGroupedExpertLinear(stream,&moe->experts_gate_up,slot->normalized_bf16,slot->moe_grouped_rows_u32,slot->moe_group_offset_u32,slot->moe_tile_prefix_w1_u32,slot->moe_gate_packed_bf16,rows,state->multiprocessor_count,state->tp_degree,state->tp_rank,0u,slot->frame_error);
		if ( error == cudaSuccess )
			error = SparkGemma4LaunchGatedGelu(stream,slot->moe_gate_packed_bf16,rows * SPARK_GEMMA4_MODEL_EXPERTS_PER_TOKEN,SPARK_GEMMA4_MODEL_EXPERT_INTERMEDIATE_DIMENSION);
		if ( error == cudaSuccess )
			error = SparkGemma4LaunchGroupedExpertLinear(stream,&moe->experts_down,slot->moe_gate_packed_bf16,0,slot->moe_group_offset_u32,slot->moe_tile_prefix_w2_u32,slot->moe_slot_out_bf16,rows * SPARK_GEMMA4_MODEL_EXPERTS_PER_TOKEN,state->multiprocessor_count,state->tp_degree,state->tp_rank,0u,slot->frame_error);
		if ( error == cudaSuccess )
			error = SparkGemma4LaunchMoePairReduceOverwrite(stream,slot->moe_slot_out_bf16,slot->moe_inverse_u32,slot->moe_weights_f32,slot->branch_bf16,rows,SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION);
		if ( error == cudaSuccess )
			error = SparkGemma4LaunchBranchAdd(stream,slot->mlp_down_bf16,slot->branch_bf16,rows,SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION);
		if ( error != cudaSuccess )
			return(SparkStageModuleCudaStatus(SPARK_GEMMA4_MODULE_TAG,error,"moe"));
	}
#endif
	if ( state->tp_degree > 1u )
	{
		status = SparkGemma4ModuleTpAllReduceHidden(state,slot,slot->mlp_down_bf16,rows);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	status = SparkStageModuleCudaStatus(SPARK_GEMMA4_MODULE_TAG,SparkGemma4LaunchRmsNorm(stream,slot->mlp_down_bf16,state->layer_post_feedforward_norm_by_layer[layer],slot->delta_bf16,rows,SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION,SPARK_GEMMA4_MODEL_RMS_NORM_EPSILON),"post_ff_norm");
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkStageModuleCudaStatus(SPARK_GEMMA4_MODULE_TAG,SparkGemma4LaunchResidualAdd(stream,slot->hidden_bf16,slot->delta_bf16,rows,SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION),"ffn_residual"));
}

static SparkStatus SparkGemma4ModuleRunLayer(SparkGemma4ModuleState *state, SparkGemma4ModuleSlot *slot, const SparkGemma4ResidentDecodeStageFrameContext *context, uint32_t layer, uint32_t rows)
{
	SparkStatus status;
	status = SparkStageModuleCudaStatus(SPARK_GEMMA4_MODULE_TAG,SparkGemma4LaunchRmsNorm((cudaStream_t)slot->cuda_stream,slot->hidden_bf16,state->layer_input_norm_by_layer[layer],slot->normalized_bf16,rows,SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION,SPARK_GEMMA4_MODEL_RMS_NORM_EPSILON),"layer_input_norm");
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkGemma4ModuleRunAttentionBody(state,slot,context,layer,rows,SPARK_GEMMA4_MODEL_LAYER_IS_FULL(layer));
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkGemma4ModuleRunFeedForward(state,slot,layer,rows));
}

static SparkStatus SparkGemma4ModuleValidateFrameContext(SparkGemma4ModuleState *state, const SparkGemma4ResidentDecodeStageFrameContext *context)
{
	uint32_t wants_input,wants_output,has_input,has_output;
	wants_input = state->stage_index != 0u ? 1u : 0u;
	wants_output = state->stage_index + 1u < state->stage_count ? 1u : 0u;
	if ( context == 0 )
		return((wants_input == 0u && wants_output == 0u) || state->allow_unqualified_execution != 0u
			? SPARK_STATUS_OK
			: SPARK_STATUS_INVALID_ARGUMENT);
	if ( context->abi_version != SPARK_GEMMA4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION ||
		context->descriptor_bytes != sizeof(*context) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	has_input = (context->flags & SPARK_GEMMA4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT) != 0u ? 1u : 0u;
	has_output = (context->flags & SPARK_GEMMA4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT) != 0u ? 1u : 0u;
	if ( has_input != wants_input || has_output != wants_output )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( has_input != 0u && (context->hidden_input_transport_session == 0 || context->hidden_input_post_receive_function == 0) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( has_output != 0u && (context->hidden_output_transport_session == 0 || context->hidden_output_send_function == 0) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGemma4ModuleConsumeHiddenInput(SparkGemma4ModuleState *state, SparkGemma4ModuleSlot *slot, SparkGemma4ResidentDecodeStageFrameContext *context, uint32_t rows)
{
	SparkHiddenTransportPacket *packet = &context->hidden_input_packet;
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	SparkStatus status;
	(void)state;
	memset(packet,0,sizeof(*packet));
	status = context->hidden_input_post_receive_function(context->hidden_input_transport_session,packet);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( packet->hidden_bf16 == 0 || packet->active_sequence_count < rows ||
		packet->hidden_dimension != SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION ||
		packet->bytes_per_sequence < SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION * SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	error = cudaMemcpyAsync(slot->hidden_bf16,packet->hidden_bf16,(uint64_t)rows * SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION * SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES,cudaMemcpyDeviceToDevice,stream);
	return(SparkStageModuleCudaStatus(SPARK_GEMMA4_MODULE_TAG,error,"hidden_input"));
}

static SparkStatus SparkGemma4ModuleEmitHiddenOutput(SparkGemma4ModuleState *state, SparkGemma4ModuleSlot *slot, SparkGemma4ResidentDecodeStageFrameContext *context, uint32_t rows)
{
	SparkHiddenTransportPacket *packet = &context->hidden_output_packet;
	(void)state;
	memset(packet,0,sizeof(*packet));
	packet->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
	packet->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_PACKET_BYTES;
	packet->flags = SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_BF16 | SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_DEVICE_POINTER;
	packet->active_sequence_count = rows;
	packet->hidden_dimension = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
	packet->bytes_per_sequence = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION * SPARK_GEMMA4_MODEL_BF16_ELEMENT_BYTES;
	packet->hidden_bf16 = slot->hidden_bf16;
	packet->cuda_stream = slot->cuda_stream;
	return(context->hidden_output_send_function(context->hidden_output_transport_session,packet));
}

static SparkStatus SparkGemma4ModuleEmbedRows(SparkGemma4ModuleState *state, SparkGemma4ModuleSlot *slot, const void *token_ids_host, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	SparkStatus status;
	status = SparkStageModuleCudaStatus(SPARK_GEMMA4_MODULE_TAG,cudaMemcpyAsync(slot->input_token_ids,token_ids_host,rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream),"token_upload");
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkStageModuleCudaStatus(SPARK_GEMMA4_MODULE_TAG,SparkGemma4LaunchEmbeddingGatherShardedScaled(stream,slot->input_token_ids,state->token_embedding_bf16,slot->hidden_bf16,rows,state->tp_vocab_base,state->tp_vocab_rows),"embed");
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( state->tp_degree > 1u )
		return(SparkGemma4ModuleTpAllReduceHidden(state,slot,slot->hidden_bf16,rows));
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGemma4ModuleUploadRows(SparkGemma4ModuleState *state, SparkGemma4ModuleSlot *slot, const SparkModelDriverFrame *frame, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	uint32_t row;
	for (row = 0u; row < rows; row++)
	{
		slot->host_row_lane_indices[row] = row;
		slot->host_row_positions[row] = frame->sequence_position;
		slot->host_row_positions_u32[row] = (uint32_t)frame->sequence_position;
		slot->host_row_sequences_u32[row] = row;
		slot->host_context_lengths[row] = (uint32_t)frame->sequence_position + 1u;
	}
	error = cudaMemsetAsync(slot->frame_error,0,SPARK_FRAME_ERROR_WORDS * sizeof(uint32_t),stream);
	memset(slot->host_frame_error,0,SPARK_FRAME_ERROR_WORDS * sizeof(uint32_t));
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->row_lane_indices,slot->host_row_lane_indices,rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->row_positions,slot->host_row_positions,rows * sizeof(uint64_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->row_positions_u32,slot->host_row_positions_u32,rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->row_sequences_u32,slot->host_row_sequences_u32,rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->context_lengths,slot->host_context_lengths,rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess && state->owns_embedding != 0u )
	{
		if ( frame->buffer_count < 1u || frame->buffers == 0 || frame->buffers[0].address == 0 )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		{
			uint32_t token_guard;
			for (token_guard = 0; token_guard < rows; token_guard++)
				if ( ((const uint32_t *)frame->buffers[0].address)[token_guard] >= SPARK_GEMMA4_MODEL_OUTPUT_VOCAB_COUNT )
				{
					fprintf(stderr,"%s token_id_out_of_range row=%u\n",SPARK_GEMMA4_MODULE_TAG,token_guard);
					SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
				}
		}
	}
	return(SPARK_STATUS_OK);
}

static cudaError_t SparkGemma4ModuleEmitHead(SparkGemma4ModuleState *state, SparkGemma4ModuleSlot *slot, SparkModelDriverFrame *frame, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error = SparkGemma4LaunchRmsNorm(stream,slot->hidden_bf16,state->final_norm_weight_bf16,slot->normalized_bf16,rows,SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION,SPARK_GEMMA4_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkGemma4LaunchHeadDirectArgmax(stream,slot->normalized_bf16,state->token_embedding_bf16,slot->argmax_token_ids,slot->argmax_score_f32,state->tp_vocab_base,rows,state->tp_vocab_rows);
	if ( error == cudaSuccess )
		error = SparkGemma4LaunchHeadMaxLocPack(stream,slot->argmax_score_f32,slot->argmax_token_ids,slot->head_maxloc_u64,rows);
	if ( error == cudaSuccess && state->tp_degree > 1u )
	{
		SparkStatus status = SparkGemma4ModuleTpReduceU64Max(state,slot,slot->head_maxloc_u64,rows);
		if ( status != SPARK_STATUS_OK )
			return(cudaErrorInvalidValue);
	}
	if ( error == cudaSuccess )
		error = SparkGemma4LaunchHeadMaxLocUnpack(stream,slot->head_maxloc_u64,slot->argmax_token_ids,rows);
	if ( error == cudaSuccess && frame->buffer_count >= 2u && frame->buffers[1].address != 0 )
		error = cudaMemcpyAsync(frame->buffers[1].address,slot->argmax_token_ids,rows * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream);
	if ( error == cudaSuccess )
		error = cudaStreamSynchronize(stream);
	return(error);
}

static SparkStatus SparkGemma4ModuleRunDecode(SparkGemma4ModuleState *state, SparkGemma4ModuleSlot *slot, SparkModelDriverFrame *frame, SparkGemma4ResidentDecodeStageFrameContext *context, uint32_t rows)
{
	uint32_t layer;
	SparkStatus status;
	status = SparkGemma4ModuleUploadRows(state,slot,frame,rows);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( state->owns_embedding != 0u )
	{
		status = SparkGemma4ModuleEmbedRows(state,slot,frame->buffers[0].address,rows);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	for (layer = state->first_layer_index; layer < state->first_layer_index + state->layer_count; layer++)
	{
		status = SparkGemma4ModuleRunLayer(state,slot,context,layer,rows);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	if ( state->owns_final_head != 0u )
	{
		cudaError_t error = SparkGemma4ModuleEmitHead(state,slot,frame,rows);
		if ( error != cudaSuccess )
			return(SparkStageModuleCudaStatus(SPARK_GEMMA4_MODULE_TAG,error,"emit_head"));
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGemma4ModuleExecuteFrame(void *module_state, SparkModelDriverFrame *frame)
{
	SparkGemma4ModuleState *state = (SparkGemma4ModuleState *)module_state;
	SparkGemma4ResidentDecodeStageFrameContext *context;
	SparkGemma4ModuleSlot *slot;
	uint32_t rows;
	SparkStatus status;
	context = (SparkGemma4ResidentDecodeStageFrameContext *)frame->user_context;
	if ( (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u )
	{
		fprintf(stderr,"%s prefill_deferred (decode-only v1; prefill rides the real-pack arm)\n",SPARK_GEMMA4_MODULE_TAG);
		atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	status = SparkGemma4ModuleValidateFrameContext(state,context);
	if ( status != SPARK_STATUS_OK )
		return(status);
	rows = frame->active_slot_count;
	if ( rows == 0u || rows > state->max_active_sequence_count )
	{
		atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	slot = &state->slots[0];
	slot->logical_sequence_count = frame->active_slot_count;
	if ( slot->cuda_stream == 0 )
		return(SPARK_STATUS_INTERNAL_ERROR);
	if ( state->stage_index != 0u )
	{
		status = SparkGemma4ModuleConsumeHiddenInput(state,slot,context,rows);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	atomic_fetch_add_explicit(&state->submitted_count,1u,memory_order_relaxed);
	status = SparkGemma4ModuleRunDecode(state,slot,frame,context,rows);
	if ( status != SPARK_STATUS_OK )
	{
		atomic_fetch_add_explicit(&state->failed_count,1u,memory_order_relaxed);
		return(status);
	}
	if ( state->stage_index + 1u < state->stage_count )
	{
		status = SparkGemma4ModuleEmitHiddenOutput(state,slot,context,rows);
		if ( status != SPARK_STATUS_OK )
		{
			atomic_fetch_add_explicit(&state->failed_count,1u,memory_order_relaxed);
			return(status);
		}
	}
	atomic_fetch_add_explicit(&state->completed_count,1u,memory_order_relaxed);
	if ( state->owns_final_head != 0u )
		atomic_fetch_add_explicit(&state->tokens_emitted,(unsigned long long)rows,memory_order_relaxed);
	return(SPARK_STATUS_OK);
}
