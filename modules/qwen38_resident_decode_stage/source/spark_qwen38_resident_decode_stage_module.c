/* Qwen 3.8 Max resident decode stage: stage-pack load + validation, a lean
 * decode-only Execute (GDN/attention + routed FP8 MoE per layer, argmax
 * head on the final stage), and the firmware's first-class hidden-transport
 * contract for PP handoffs. Prefill, MTP and speculation fail closed with
 * SPARK_STATUS_UNSUPPORTED.
 */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_hidden_transport.h"
#include "sparkpipe/spark_qwen38_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_stage_kv_client.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_qwen38_work_control.h"
#include "spark_qwen38_stagepack_format.h"

#define SPARK_QWEN38_MODULE_TAG "qwen38_stage"

/* KV tier: the resident pool is a window; blocks beyond it live in the
 * pluggable KV store (local socket, network service, or a dedicated spark
 * ring running the store), keyed by (model fingerprint, layout fingerprint,
 * rank, sequence, logical block). Default provider "none" keeps the
 * all-resident behavior byte-identical to before the tier landed. */
#define SPARK_QWEN38_MODULE_KV_STAGING_RECORDS 16u
#define SPARK_QWEN38_MODULE_KV_POLL_BOUND 10000u
#define SPARK_QWEN38_MODULE_KV_GDN_RECORD_PLACEHOLDER_BYTES 4096u
#define SPARK_QWEN38_MODULE_KV_MAX_BLOCKS_PER_LANE 4096u
/* Mirrors SPARK_LM_HEAD_SCREEN_CAP / SPARK_LM_HEAD_SHADOW_GROUP from the
 * common CUDA header (module.c cannot include that header under cc). */
#define SPARK_QWEN38_MODULE_HEAD_SCREEN_CAP 4096u
#define SPARK_QWEN38_MODULE_HEAD_SHADOW_GROUP 32u

typedef struct SparkQwen38ModuleSlot
{
	void *cuda_stream;
	uint32_t *host_row_lane_indices;
	uint64_t *host_row_positions;
	uint32_t *host_row_cold;
	uint32_t *host_slot_mapping;
	uint32_t *host_context_lengths;
	uint32_t *input_token_ids;
	uint32_t *output_token_ids;
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
	float *moe_scores_f32;
	uint32_t *moe_indices_u32;
	float *moe_weights_f32;
	uint32_t *moe_inverse_u32;
	uint32_t *moe_grouped_rows_u32;
	uint32_t *moe_tile_prefix_w1_u32;
	uint32_t *moe_tile_prefix_w2_u32;
	void *moe_gate_packed_bf16;
	void *moe_slot_up_bf16;
	void *moe_slot_out_bf16;
	uint32_t *moe_group_offset_u32;
	void *shared_gate_bf16;
	void *shared_up_bf16;
	void *shared_down_bf16;
	float *chunk_qn_f32;
	float *chunk_kn_f32;
	float *chunk_cum_g_f32;
	float *chunk_decay_f32;
	float *chunk_attn_f32;
	float *chunk_w_f32;
	float *chunk_kg_f32;
	uint32_t *mtp_draft_ids;
	/* Screened head: coarse logits + screen candidates (the exact
	 * full-vocab matvec reads 4 GB of head weight PER ROW; the shadow
	 * path reads the 4-bit copy instead and rescores a certified
	 * candidate set - identical argmax, a fraction of the bytes). */
	void *head_logits_bf16;
	uint32_t *head_candidate_ids;
	uint32_t *head_candidate_counts;
} SparkQwen38ModuleSlot;

typedef struct SparkQwen38ModuleState
{
	SparkStageModuleLedger ledger;
	uint32_t multiprocessor_count;
	uint32_t tp_degree;
	uint32_t tp_rank;
	/* Measurement aids (timing probe bisect): skip one block of the layer
	 * to price the GDN/attention half vs the MoE half. Default 0. */
	uint32_t debug_skip_gdn;
	uint32_t debug_skip_moe;
	/* Tensor-parallel collective: one residual all-reduce per layer after
	 * the expert-sharded MoE. Env-driven (TP_BACKEND_PATH / TP_IDENTIFIER /
	 * TP_PORT_BASE / TP_HOSTS / TP_TIMEOUT_MS), mirrors the dsv4 wiring. */
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
	uint64_t tp_collective_identifier;
	uint32_t tp_control_port_base;
	uint32_t tp_connect_timeout_milli;
	uint32_t tp_operation_timeout_milli;
	uint32_t max_active_sequence_count;
	uint32_t pipeline_slot_count;
	uint32_t kv_block_count;
	uint32_t cache_layer_count;
	uint32_t mtp_cache_ordinal;
	atomic_ullong submitted_count;
	atomic_ullong completed_count;
	atomic_ullong rejected_count;
	atomic_ullong failed_count;
	atomic_ullong tokens_emitted;
	SparkQwen38GdnStatePool gdn_pool;
	void *kv_cache_bf16;
	uint64_t cache_layer_stride;
	uint64_t cache_block_stride;
	SparkQwen38ModuleSlot slots[SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	uint32_t stage_count;
	uint32_t stage_index;
	uint32_t allow_unqualified_execution;
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t owns_embedding;
	uint32_t owns_final_head;
	uint32_t gdn_layer_count;
	uint32_t attn_layer_count;
	uint32_t gdn_ordinal_by_layer[SPARK_QWEN38_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	uint32_t attn_ordinal_by_layer[SPARK_QWEN38_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	uint32_t layer_seen_bits[SPARK_QWEN38_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	uint32_t global_seen_bits;
	uint32_t mtp_seen_bits;
	SparkQwen38MtpWeights mtp;
	const void *token_embedding_bf16;
	const void *final_norm_weight_bf16;
	const void *lm_head_weight_bf16;
	/* One-time 4-bit shadow of the head weights + certified per-neuron
	 * error bounds for the screened argmax. */
	uint8_t *head_shadow_payload;
	uint8_t *head_shadow_scale;
	float *head_error_norm_f32;
	const void *attention_norm_by_layer[SPARK_QWEN38_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	const void *mlp_norm_by_layer[SPARK_QWEN38_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	SparkQwen38GdnLayerWeights gdn_by_layer[SPARK_QWEN38_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	SparkQwen38AttnLayerWeights attn_by_layer[SPARK_QWEN38_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	SparkQwen38MoeWeights moe_by_layer[SPARK_QWEN38_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	/* KV tier state. Logical blocks are (lane, logical_block) identities;
	 * kv_logical_to_slot[lane * lane_stride + logical] holds slot+1 while the
	 * block is resident, 0 while it lives in the store. Slots are pool
	 * windows; kv_slot_dirty marks a slot whose cached rows the store lacks. */
	SparkStageKvClient kv_client;
	SparkQwen38WorkControlKvState kv_work;
	SparkQwen38WorkControlKvPlanConfig kv_plan;
	uint32_t kv_tier_active;
	uint32_t kv_logical_page_capacity;
	uint32_t kv_physical_page_capacity;
	uint64_t kv_backing_maximum_bytes;
	uint32_t *kv_logical_to_slot;
	uint64_t kv_logical_to_slot_capacity;
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
} SparkQwen38ModuleState;

static SparkStatus SparkQwen38ModuleConfigure(SparkQwen38ModuleState *state)
{
	SparkStatus status;
	{
		/* Optional head-parallel TP geometry. Unset means the replicated
		 * layout (tp_degree 1). tp_degree > 1 needs the head-sliced
		 * projections AND the residual all-reduce, so the initialize path
		 * refuses it until the TP collective is wired (fail closed). */
		const char *tp_degree_text = getenv("SPARK_QWEN38_STAGE_TP_DEGREE");
		const char *tp_rank_text = getenv("SPARK_QWEN38_STAGE_TP_RANK");
		char *end = 0;
		unsigned long parsed = 1u;
		if ( tp_degree_text != 0 )
		{
			parsed = strtoul(tp_degree_text,&end,10);
			if ( end == tp_degree_text || parsed < 1u || parsed > SPARK_QWEN38_MODEL_ATTN_QUERY_HEAD_COUNT )
				return(SPARK_STATUS_INVALID_ARGUMENT);
		}
		state->tp_degree = (uint32_t)parsed;
		parsed = 0u;
		if ( tp_rank_text != 0 )
		{
			parsed = strtoul(tp_rank_text,&end,10);
			if ( end == tp_rank_text )
				return(SPARK_STATUS_INVALID_ARGUMENT);
		}
		state->tp_rank = (uint32_t)parsed;
		/* TP sharding (expert-sharded MoE, one residual all-reduce per
		 * layer). The attention/GDN head slicing and the KV-head split are
		 * the follow-on increments; the expert slice works for any degree
		 * that divides the 512 experts. tp>1 needs the collective env. */
		if ( state->tp_rank >= state->tp_degree || state->tp_degree > SPARK_QWEN38_MODEL_ROUTED_EXPERT_COUNT || (SPARK_QWEN38_MODEL_ROUTED_EXPERT_COUNT % state->tp_degree) != 0u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		state->tp_collective_identifier = 0u;
		state->tp_control_port_base = 0u;
		state->tp_connect_timeout_milli = 120000u;
		state->tp_operation_timeout_milli = 120000u;
		state->tp_backend_path[0] = '\0';
		state->tp_local_host[0] = '\0';
		for (parsed = 0u; parsed < SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE; parsed++)
			state->tp_hosts[parsed][0] = '\0';
		if ( state->tp_degree > 1u )
		{
			const char *tp_backend = getenv("SPARK_QWEN38_STAGE_TP_BACKEND_PATH");
			const char *tp_identifier = getenv("SPARK_QWEN38_STAGE_TP_IDENTIFIER");
			const char *tp_port_base = getenv("SPARK_QWEN38_STAGE_TP_PORT_BASE");
			const char *tp_hosts = getenv("SPARK_QWEN38_STAGE_TP_HOSTS");
			const char *tp_local_host = getenv("SPARK_QWEN38_STAGE_TP_LOCAL_HOST");
			const char *tp_timeout = getenv("SPARK_QWEN38_STAGE_TP_TIMEOUT_MS");
			const char *scan;
			uint32_t host_index,host_start;
			if ( tp_backend == 0 || tp_identifier == 0 || tp_port_base == 0 || tp_hosts == 0 || tp_local_host == 0 )
				return(SPARK_STATUS_INVALID_ARGUMENT);
			snprintf(state->tp_backend_path,sizeof(state->tp_backend_path),"%s",tp_backend);
			snprintf(state->tp_local_host,sizeof(state->tp_local_host),"%s",tp_local_host);
			state->tp_collective_identifier = strtoull(tp_identifier,0,10);
			state->tp_control_port_base = (uint32_t)strtoul(tp_port_base,0,10);
			if ( tp_timeout != 0 )
			{
				state->tp_connect_timeout_milli = (uint32_t)strtoul(tp_timeout,0,10);
				state->tp_operation_timeout_milli = state->tp_connect_timeout_milli;
			}
			/* Comma-separated peer host list, one per rank in rank order. */
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
			(void)host_start;
		}
	}
	state->debug_skip_gdn = getenv("SPARK_QWEN38_STAGE_DEBUG_SKIP_GDN") != 0 ? 1u : 0u;
	state->debug_skip_moe = getenv("SPARK_QWEN38_STAGE_DEBUG_SKIP_MOE") != 0 ? 1u : 0u;
	status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN38_MODULE_TAG,"SPARK_QWEN38_STAGE_COUNT",1u,SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT,&state->stage_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN38_MODULE_TAG,"SPARK_QWEN38_STAGE_INDEX",0u,SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT - 1u,&state->stage_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN38_MODULE_TAG,"SPARK_QWEN38_STAGE_FIRST_LAYER",0u,SPARK_QWEN38_RESIDENT_DECODE_STAGE_LAYER_COUNT - 1u,&state->first_layer_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN38_MODULE_TAG,"SPARK_QWEN38_STAGE_LAYER_COUNT",1u,SPARK_QWEN38_RESIDENT_DECODE_STAGE_LAYER_COUNT,&state->layer_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN38_MODULE_TAG,"SPARK_QWEN38_STAGE_MAX_ACTIVE_SEQUENCES",1u,SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,&state->max_active_sequence_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN38_MODULE_TAG,"SPARK_QWEN38_STAGE_PIPELINE_SLOTS",1u,SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,&state->pipeline_slot_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN38_MODULE_TAG,"SPARK_QWEN38_STAGE_KV_BLOCKS",1u,1u << 20u,&state->kv_block_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( state->stage_index >= state->stage_count || state->first_layer_index + state->layer_count > SPARK_QWEN38_RESIDENT_DECODE_STAGE_LAYER_COUNT )
	{
		fprintf(stderr,"%s config_slice_invalid stage=%u/%u slice=%u+%u\n",SPARK_QWEN38_MODULE_TAG,state->stage_index,state->stage_count,state->first_layer_index,state->layer_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	/* The grouped scalar expert path prices its row tiles at SPARK_LM_TILE
	 * (16) rows while the route build switches to 32-row tiles past 409
	 * sequences; a batch that crosses the boundary would silently skip rows
	 * in every expert group. Refuse it loudly until the MoE moves to the
	 * launch-planner GEMM (which shares one tile-M with the route build). */
	if ( state->max_active_sequence_count > 409u )
	{
		fprintf(stderr,"%s config_batch_too_wide max_active=%u (grouped scalar path supports at most 409 rows at a 16-row tile; see audit doc)\n",SPARK_QWEN38_MODULE_TAG,state->max_active_sequence_count);
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	state->owns_embedding = state->first_layer_index == 0u ? 1u : 0u;
	state->owns_final_head = state->first_layer_index + state->layer_count == SPARK_QWEN38_RESIDENT_DECODE_STAGE_LAYER_COUNT ? 1u : 0u;
	if ( (state->stage_index == 0u) != (state->owns_embedding != 0u) || (state->stage_index + 1u == state->stage_count) != (state->owns_final_head != 0u) )
	{
		fprintf(stderr,"%s config_position_mismatch stage=%u/%u slice=%u+%u\n",SPARK_QWEN38_MODULE_TAG,state->stage_index,state->stage_count,state->first_layer_index,state->layer_count);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	return(SPARK_STATUS_OK);
}

static void SparkQwen38ModuleBuildOrdinals(SparkQwen38ModuleState *state)
{
	uint32_t layer;
	for (layer = 0; layer < SPARK_QWEN38_RESIDENT_DECODE_STAGE_LAYER_COUNT; layer++)
	{
		state->gdn_ordinal_by_layer[layer] = UINT32_MAX;
		state->attn_ordinal_by_layer[layer] = UINT32_MAX;
	}
	for (layer = state->first_layer_index; layer < state->first_layer_index + state->layer_count; layer++)
	{
		if ( SPARK_QWEN38_MODEL_LAYER_IS_GDN(layer) != 0u )
			state->gdn_ordinal_by_layer[layer] = state->gdn_layer_count++;
		else
			state->attn_ordinal_by_layer[layer] = state->attn_layer_count++;
	}
}

static void SparkQwen38ModuleFillLinearView(SparkQwen38LinearView *view, const SparkQwen38StagePackEntry *entry, void *payload, void *scale)
{
	view->abi_version = SPARK_QWEN38_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION;
	view->weight_format = entry->weight_format;
	view->input_dimension = entry->columns;
	view->output_dimension = entry->rows;
	view->weight_payload = payload;
	view->weight_scale_e8m0 = (const uint8_t *)scale;
	view->weight_payload_bytes = entry->payload_bytes;
	view->weight_scale_bytes = entry->scale_bytes;
}

static SparkStatus SparkQwen38ModuleValidateEntry(SparkQwen38ModuleState *state, const SparkQwen38StagePackEntry *entry, uint64_t file_bytes, uint32_t *is_global)
{
	SparkQwen38StagePackTensorShape shape;
	uint32_t global = entry->layer_index == SPARK_QWEN38_STAGEPACK_GLOBAL_LAYER ? 1u : 0u;
	if ( SparkQwen38StagePackResolvedShape(entry->tensor_kind,global != 0u ? 0u : entry->layer_index,global,&shape) != 0 || entry->rows != shape.rows || entry->columns != shape.columns )
		return(SPARK_STATUS_VALIDATION_FAILED);
	/* Strict natural format, except the three routed-expert tensors may also
	 * arrive BF16 (synthesized test packs); anything else fails. */
	if ( entry->weight_format != shape.natural_format )
	{
		if ( (shape.natural_format != SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 && shape.natural_format != SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128) || entry->weight_format != SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 )
			return(SPARK_STATUS_VALIDATION_FAILED);
	}
	if ( SparkHybridStagePackScaleGroupSizeOk(SparkQwen38StagePackWeightClass(entry->weight_format),entry->scale_group_size) == 0u )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->payload_bytes != SparkQwen38StagePackPayloadBytes(entry->weight_format,entry->rows,entry->columns) || entry->scale_bytes != SparkQwen38StagePackScaleBytes(entry->weight_format,entry->rows,entry->columns) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->payload_offset > file_bytes || entry->payload_bytes > file_bytes - entry->payload_offset )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->scale_bytes != 0u && (entry->scale_offset > file_bytes || entry->scale_bytes > file_bytes - entry->scale_offset) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->layer_index == SPARK_QWEN38_STAGEPACK_MTP_LAYER || (global != 0u && (entry->tensor_kind >= SPARK_QWEN38_STAGEPACK_TENSOR_MTP_FC && entry->tensor_kind <= SPARK_QWEN38_STAGEPACK_TENSOR_MTP_FINAL_NORM)) )
	{
		if ( state->owns_final_head == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
	}
	else if ( global == 0u && (entry->layer_index < state->first_layer_index || entry->layer_index >= state->first_layer_index + state->layer_count) )
		return(SPARK_STATUS_VALIDATION_FAILED);
	*is_global = global;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen38ModuleBindMoe(SparkQwen38MoeWeights *moe, const SparkQwen38StagePackEntry *entry, void *payload, void *scale)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_QWEN38_STAGEPACK_TENSOR_MOE_GATE: SparkQwen38ModuleFillLinearView(&moe->gate,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_MOE_W1: SparkQwen38ModuleFillLinearView(&moe->experts_w1,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_MOE_W3: SparkQwen38ModuleFillLinearView(&moe->experts_w3,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_MOE_DOWN: SparkQwen38ModuleFillLinearView(&moe->experts_w2,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_MOE_SHARED_GATE: SparkQwen38ModuleFillLinearView(&moe->shared_gate,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_MOE_SHARED_UP: SparkQwen38ModuleFillLinearView(&moe->shared_up,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_MOE_SHARED_DOWN: SparkQwen38ModuleFillLinearView(&moe->shared_down,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_MOE_SHARED_GATE_WEIGHT: moe->shared_gate_weight_bf16 = payload; return(SPARK_STATUS_OK);
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
}

static SparkStatus SparkQwen38ModuleBindMtp(SparkQwen38ModuleState *state, const SparkQwen38StagePackEntry *entry, void *payload, void *scale)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_QWEN38_STAGEPACK_TENSOR_MTP_FC: SparkQwen38ModuleFillLinearView(&state->mtp.fc,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_ATTENTION_NORM: state->mtp.attention_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_MLP_NORM: state->mtp.mlp_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_QUERY: SparkQwen38ModuleFillLinearView(&state->mtp.attention.query,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_KEY: SparkQwen38ModuleFillLinearView(&state->mtp.attention.key,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_VALUE: SparkQwen38ModuleFillLinearView(&state->mtp.attention.value,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_OUTPUT: SparkQwen38ModuleFillLinearView(&state->mtp.attention.output,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_QUERY_NORM: state->mtp.attention.query_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_KEY_NORM: state->mtp.attention.key_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	default:
		return(SparkQwen38ModuleBindMoe(&state->mtp.moe,entry,payload,scale));
	}
}

static SparkStatus SparkQwen38ModuleBindGlobal(SparkQwen38ModuleState *state, const SparkQwen38StagePackEntry *entry, void *payload)
{
	switch ( entry->tensor_kind )
	{
	case SPARK_QWEN38_STAGEPACK_TENSOR_EMBEDDING:
		if ( state->owns_embedding == 0u && state->owns_final_head == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
		state->token_embedding_bf16 = payload;
		return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_FINAL_NORM:
		if ( state->owns_final_head == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
		state->final_norm_weight_bf16 = payload;
		return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_LM_HEAD:
		if ( state->owns_final_head == 0u )
			return(SPARK_STATUS_VALIDATION_FAILED);
		state->lm_head_weight_bf16 = payload;
		return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_MTP_EMBED_NORM: state->mtp.embed_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_MTP_HIDDEN_NORM: state->mtp.hidden_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_MTP_FINAL_NORM: state->mtp.final_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	default:
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
}

static SparkStatus SparkQwen38ModuleBindLayer(SparkQwen38ModuleState *state, const SparkQwen38StagePackEntry *entry, void *payload, void *scale)
{
	uint32_t layer = entry->layer_index;
	SparkStatus status;
	switch ( entry->tensor_kind )
	{
	case SPARK_QWEN38_STAGEPACK_TENSOR_ATTENTION_NORM: state->attention_norm_by_layer[layer] = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_MLP_NORM: state->mlp_norm_by_layer[layer] = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_GDN_QKV: SparkQwen38ModuleFillLinearView(&state->gdn_by_layer[layer].qkv,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_GDN_GATE: SparkQwen38ModuleFillLinearView(&state->gdn_by_layer[layer].gate,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_GDN_BETA: SparkQwen38ModuleFillLinearView(&state->gdn_by_layer[layer].beta,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_GDN_DECAY: SparkQwen38ModuleFillLinearView(&state->gdn_by_layer[layer].decay,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_GDN_OUTPUT: SparkQwen38ModuleFillLinearView(&state->gdn_by_layer[layer].output,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_GDN_CONV_WEIGHT: state->gdn_by_layer[layer].conv_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_GDN_A_LOG: state->gdn_by_layer[layer].a_log_f32 = (const float *)payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_GDN_DT_BIAS: state->gdn_by_layer[layer].dt_bias_f32 = (const float *)payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_GDN_NORM: state->gdn_by_layer[layer].gdn_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_QUERY: SparkQwen38ModuleFillLinearView(&state->attn_by_layer[layer].query,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_KEY: SparkQwen38ModuleFillLinearView(&state->attn_by_layer[layer].key,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_VALUE: SparkQwen38ModuleFillLinearView(&state->attn_by_layer[layer].value,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_OUTPUT: SparkQwen38ModuleFillLinearView(&state->attn_by_layer[layer].output,entry,payload,scale); return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_QUERY_NORM: state->attn_by_layer[layer].query_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_KEY_NORM: state->attn_by_layer[layer].key_norm_weight_bf16 = payload; return(SPARK_STATUS_OK);
	default:
		status = SparkQwen38ModuleBindMoe(&state->moe_by_layer[layer],entry,payload,scale);
		if ( status == SPARK_STATUS_OK )
			return(SPARK_STATUS_OK);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
}

static SparkStatus SparkQwen38ModuleLoadEntry(SparkQwen38ModuleState *state, FILE *file, const SparkQwen38StagePackEntry *entry, uint64_t file_bytes)
{
	SparkStatus status;
	uint32_t is_global = 0u,bit = 1u << entry->tensor_kind;
	uint32_t *seen;
	void *payload = 0,*scale = 0;
	status = SparkQwen38ModuleValidateEntry(state,entry,file_bytes,&is_global);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s pack_entry_invalid kind=%u layer=%u\n",SPARK_QWEN38_MODULE_TAG,entry->tensor_kind,entry->layer_index);
		return(status);
	}
	if ( entry->layer_index == SPARK_QWEN38_STAGEPACK_MTP_LAYER || (is_global != 0u && entry->tensor_kind >= SPARK_QWEN38_STAGEPACK_TENSOR_MTP_FC && entry->tensor_kind <= SPARK_QWEN38_STAGEPACK_TENSOR_MTP_FINAL_NORM) )
		seen = &state->mtp_seen_bits;
	else
		seen = is_global != 0u ? &state->global_seen_bits : &state->layer_seen_bits[entry->layer_index];
	if ( (*seen & bit) != 0u )
	{
		fprintf(stderr,"%s pack_entry_duplicate kind=%u layer=%u\n",SPARK_QWEN38_MODULE_TAG,entry->tensor_kind,entry->layer_index);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	*seen |= bit;
	status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,entry->payload_offset,entry->payload_bytes,&payload);
	if ( status == SPARK_STATUS_OK && entry->scale_bytes != 0u )
		status = SparkStageModuleLoadDeviceRegion(&state->ledger,file,entry->scale_offset,entry->scale_bytes,&scale);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( entry->layer_index == SPARK_QWEN38_STAGEPACK_MTP_LAYER || entry->tensor_kind == SPARK_QWEN38_STAGEPACK_TENSOR_MTP_FC )
		return(SparkQwen38ModuleBindMtp(state,entry,payload,scale));
	return(is_global != 0u ? SparkQwen38ModuleBindGlobal(state,entry,payload) : SparkQwen38ModuleBindLayer(state,entry,payload,scale));
}

static SparkStatus SparkQwen38ModuleVerifyCoverage(SparkQwen38ModuleState *state)
{
	uint32_t layer,expected_global = 0u,expected_layer;
	if ( state->owns_embedding != 0u || state->owns_final_head != 0u )
		expected_global |= 1u << SPARK_QWEN38_STAGEPACK_TENSOR_EMBEDDING;
	if ( state->owns_final_head != 0u )
		expected_global |= (1u << SPARK_QWEN38_STAGEPACK_TENSOR_FINAL_NORM) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_LM_HEAD);
	if ( state->owns_final_head != 0u )
	{
		uint32_t expected_mtp = (1u << SPARK_QWEN38_STAGEPACK_TENSOR_MTP_FC) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_MTP_EMBED_NORM) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_MTP_HIDDEN_NORM) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_MTP_FINAL_NORM) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_ATTENTION_NORM) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_MLP_NORM) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_MOE_GATE) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_MOE_W1) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_MOE_W3) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_MOE_DOWN) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_MOE_SHARED_GATE) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_MOE_SHARED_UP) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_MOE_SHARED_DOWN) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_MOE_SHARED_GATE_WEIGHT) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_QUERY) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_KEY) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_VALUE) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_OUTPUT) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_QUERY_NORM) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_KEY_NORM);
		if ( state->mtp_seen_bits != expected_mtp )
		{
			fprintf(stderr,"%s pack_mtp_incomplete seen=%08x expected=%08x\n",SPARK_QWEN38_MODULE_TAG,state->mtp_seen_bits,expected_mtp);
			return(SPARK_STATUS_VALIDATION_FAILED);
		}
	}
	if ( state->global_seen_bits != expected_global )
	{
		fprintf(stderr,"%s pack_globals_incomplete seen=%08x expected=%08x\n",SPARK_QWEN38_MODULE_TAG,state->global_seen_bits,expected_global);
		return(SPARK_STATUS_VALIDATION_FAILED);
	}
	for (layer = state->first_layer_index; layer < state->first_layer_index + state->layer_count; layer++)
	{
		expected_layer = (1u << SPARK_QWEN38_STAGEPACK_TENSOR_ATTENTION_NORM) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_MLP_NORM) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_MOE_GATE) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_MOE_W1) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_MOE_W3) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_MOE_DOWN) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_MOE_SHARED_GATE) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_MOE_SHARED_UP) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_MOE_SHARED_DOWN) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_MOE_SHARED_GATE_WEIGHT);
		if ( SPARK_QWEN38_MODEL_LAYER_IS_GDN(layer) != 0u )
			expected_layer |= (1u << SPARK_QWEN38_STAGEPACK_TENSOR_GDN_QKV) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_GDN_GATE) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_GDN_BETA) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_GDN_DECAY) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_GDN_OUTPUT) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_GDN_CONV_WEIGHT) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_GDN_A_LOG) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_GDN_DT_BIAS) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_GDN_NORM);
		else
			expected_layer |= (1u << SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_QUERY) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_KEY) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_VALUE) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_OUTPUT) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_QUERY_NORM) | (1u << SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_KEY_NORM);
		if ( state->layer_seen_bits[layer] != expected_layer )
		{
			fprintf(stderr,"%s pack_layer_incomplete layer=%u seen=%08x expected=%08x\n",SPARK_QWEN38_MODULE_TAG,layer,state->layer_seen_bits[layer],expected_layer);
			return(SPARK_STATUS_VALIDATION_FAILED);
		}
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen38ModuleLoadPack(SparkQwen38ModuleState *state, const char *path)
{
	SparkQwen38StagePackHeader header,expected;
	SparkQwen38StagePackEntry *directory;
	FILE *file;
	SparkStatus status;
	uint32_t index;
	file = fopen(path,"rb");
	if ( file == 0 )
	{
		fprintf(stderr,"%s pack_open_failed path=%s\n",SPARK_QWEN38_MODULE_TAG,path);
		return(SPARK_STATUS_IO_ERROR);
	}
	status = SparkStageModulePackRead(SPARK_QWEN38_MODULE_TAG,file,0u,&header,sizeof(header));
	if ( status == SPARK_STATUS_OK )
	{
		SparkQwen38StagePackExpectedGeometry(&expected,state->first_layer_index,state->layer_count);
		if ( SparkQwen38StagePackHeaderMatches(&header,&expected) != 0 || header.directory_offset != SPARK_QWEN38_STAGEPACK_HEADER_BYTES )
		{
			fprintf(stderr,"%s pack_geometry_mismatch\n",SPARK_QWEN38_MODULE_TAG);
			status = SPARK_STATUS_VALIDATION_FAILED;
		}
	}
	directory = status == SPARK_STATUS_OK ? (SparkQwen38StagePackEntry *)malloc((size_t)header.tensor_count * sizeof(SparkQwen38StagePackEntry)) : 0;
	if ( status == SPARK_STATUS_OK && directory == 0 )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModulePackRead(SPARK_QWEN38_MODULE_TAG,file,header.directory_offset,directory,(uint64_t)header.tensor_count * sizeof(SparkQwen38StagePackEntry));
	for (index = 0; status == SPARK_STATUS_OK && index < header.tensor_count; index++)
		status = SparkQwen38ModuleLoadEntry(state,file,&directory[index],header.file_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen38ModuleVerifyCoverage(state);
	free(directory);
	fclose(file);
	return(status);
}

void SparkQwen38ResidentDecodeStageDestroy(void *module_state);
static SparkStatus SparkQwen38ModuleAllocatePools(SparkQwen38ModuleState *state);
extern cudaError_t SparkQwen38ConfigureCudaKernels(void);
static SparkStatus SparkQwen38ModuleAllocateSlot(SparkQwen38ModuleState *state, SparkQwen38ModuleSlot *slot);
static SparkStatus SparkQwen38ModuleAllocateSlotHostMirrors(SparkQwen38ModuleState *state, SparkQwen38ModuleSlot *slot);

/* --------------------------------------------------------------------------
 * KV tier: the resident pool becomes a WINDOW over the logical block space.
 * Nonresident blocks page through the pluggable KV store (local socket,
 * network service address, or a dedicated spark ring running the store
 * service), keyed by (model fingerprint, layout fingerprint, rank, sequence,
 * logical block). Provider "none" keeps the all-resident behavior.
 * ------------------------------------------------------------------------*/

static uint64_t SparkQwen38ModuleFingerprint(const void *bytes, uint64_t count, uint64_t basis)
{
	const uint8_t *data = (const uint8_t *)bytes;
	uint64_t hash = basis,index;
	for (index = 0; index < count; index++)
		hash = (hash ^ data[index]) * 1099511628211ull;
	return(hash);
}

static SparkStatus SparkQwen38ModuleOpenKvTier(SparkQwen38ModuleState *state, const SparkFirmwareModuleHostServices *host_services)
{
	SparkQwen38StagePackHeader geometry;
	const char *provider = 0,*service = 0,*socket_path = 0;
	uint64_t pool_bytes = 0u,model_fp,layout_fp,layout_bits[3],block_record_bytes,staging_bytes;
	uint32_t workers = 0u,block_record_elements,index;
	SparkStatus status;
	static const char *none = "none";
	state->kv_tier_active = 0u;
	state->kv_logical_page_capacity = host_services->kv_logical_page_capacity;
	state->kv_physical_page_capacity = host_services->kv_physical_page_capacity;
	state->kv_backing_maximum_bytes = host_services->kv_backing_maximum_bytes;
	/* The ABI validator enforced (logical==0)==(physical==0) and
	 * physical<=logical; a byte cap with no backing path is refused here. */
	if ( host_services->kv_backing_directory == 0 && host_services->kv_backing_maximum_bytes != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	/* The store provider is optional; getenv stays quiet when it is unset
	 * instead of logging a config_missing on every storeless deployment. */
	provider = getenv("SPARK_QWEN38_STAGE_KV_STORE");
	if ( provider == 0 )
		provider = none;
	if ( strcmp(provider,"none") == 0 )
		return(SparkStageKvClientOpen(&state->kv_client,SPARK_QWEN38_MODULE_TAG,provider,0u,0u,0u,0u,0u,0,0,0u,0u));
	status = SparkStageModuleEnvironmentText(SPARK_QWEN38_MODULE_TAG,"SPARK_QWEN38_STAGE_KV_SERVICE",&service);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentText(SPARK_QWEN38_MODULE_TAG,"SPARK_QWEN38_STAGE_KV_SOCKET",&socket_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned64(SPARK_QWEN38_MODULE_TAG,"SPARK_QWEN38_STAGE_KV_POOL_BYTES",1u,1ull << 40u,&pool_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN38_MODULE_TAG,"SPARK_QWEN38_STAGE_KV_WORKERS",1u,64u,&workers);
	if ( status != SPARK_STATUS_OK )
		return(status);
	SparkQwen38StagePackExpectedGeometry(&geometry,state->first_layer_index,state->layer_count);
	model_fp = SparkQwen38ModuleFingerprint(&geometry,sizeof(geometry),14695981039346656037ull);
	block_record_elements = SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS * SPARK_QWEN38_MODEL_ATTN_CACHE_TOKEN_ELEMENTS * state->attn_layer_count;
	layout_bits[0] = block_record_elements;
	layout_bits[1] = SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
	layout_bits[2] = state->kv_block_count;
	layout_fp = SparkQwen38ModuleFingerprint(layout_bits,sizeof(layout_bits),model_fp);
	block_record_bytes = (uint64_t)block_record_elements * SPARK_QWEN38_MODEL_BF16_ELEMENT_BYTES;
	staging_bytes = block_record_bytes * SPARK_QWEN38_MODULE_KV_STAGING_RECORDS;
	/* The resident pool is a window: clamp the device pool to the physical
	 * page capacity the deployment declared, so the adapter's freelist can
	 * span the full logical space while the device holds the window. */
	if ( state->kv_physical_page_capacity != 0u && state->kv_block_count > state->kv_physical_page_capacity )
		state->kv_block_count = state->kv_physical_page_capacity;
	if ( state->kv_block_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	state->kv_plan.model_fingerprint = model_fp;
	state->kv_plan.cache_layout_fingerprint = layout_fp;
	state->kv_plan.rank_index = state->stage_index;
	state->kv_plan.block_record_bytes = (uint32_t)block_record_bytes;
	state->kv_plan.gdn_record_bytes = SPARK_QWEN38_MODULE_KV_GDN_RECORD_PLACEHOLDER_BYTES;
	state->kv_plan.lookahead_packet_count = 3u;
	state->kv_plan.physical_block_capacity = state->kv_block_count;
	state->kv_plan.allocated_physical_block_count = 0u;
	state->kv_plan.staging_block_capacity = SPARK_QWEN38_MODULE_KV_STAGING_RECORDS;
	status = SparkStageKvClientOpen(&state->kv_client,SPARK_QWEN38_MODULE_TAG,provider,state->stage_index,state->first_layer_index,state->layer_count,model_fp,layout_fp,service,socket_path,pool_bytes,workers);
	if ( status != SPARK_STATUS_OK )
		return(status);
	state->kv_slot_lane = (uint32_t *)malloc((size_t)state->kv_block_count * sizeof(uint32_t));
	state->kv_slot_logical = (uint32_t *)malloc((size_t)state->kv_block_count * sizeof(uint32_t));
	state->kv_slot_sequence = (uint64_t *)malloc((size_t)state->kv_block_count * sizeof(uint64_t));
	state->kv_slot_dirty = (uint8_t *)calloc((size_t)state->kv_block_count,sizeof(uint8_t));
	state->kv_slot_pinned = (uint8_t *)calloc((size_t)state->kv_block_count,sizeof(uint8_t));
	state->kv_slot_free_stack = (uint32_t *)malloc((size_t)state->kv_block_count * sizeof(uint32_t));
	state->kv_block_staging = malloc((size_t)staging_bytes);
	state->kv_gdn_staging = malloc(SPARK_QWEN38_MODULE_KV_GDN_RECORD_PLACEHOLDER_BYTES);
	/* Device-side rewritten block table, sized for the full logical space
	 * (max lanes x max blocks per lane); the module owns it so the const
	 * adapter mirror stays untouched. */
	if ( cudaMalloc((void **)&state->kv_table_indices_device,(size_t)SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT * SPARK_QWEN38_MODULE_KV_MAX_BLOCKS_PER_LANE * sizeof(uint32_t)) != cudaSuccess )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( cudaMalloc((void **)&state->kv_table_counts_device,(size_t)SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT * sizeof(uint32_t)) != cudaSuccess )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->kv_table_indices_host = (uint32_t *)malloc((size_t)SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT * SPARK_QWEN38_MODULE_KV_MAX_BLOCKS_PER_LANE * sizeof(uint32_t));
	if ( state->kv_slot_lane == 0 || state->kv_slot_logical == 0 || state->kv_slot_sequence == 0 || state->kv_slot_dirty == 0 || state->kv_slot_pinned == 0 || state->kv_slot_free_stack == 0 || state->kv_block_staging == 0 || state->kv_gdn_staging == 0 || state->kv_table_indices_host == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (index = 0u; index < state->kv_block_count; index++)
		state->kv_slot_free_stack[index] = index;
	state->kv_slot_free_count = state->kv_block_count;
	state->kv_evict_cursor = 0u;
	state->kv_tier_active = 1u;
	fprintf(stderr,"%s kv_tier_open provider=%s window=%u logical=%u physical=%u backing_bytes=%llu\n",SPARK_QWEN38_MODULE_TAG,provider,state->kv_block_count,state->kv_logical_page_capacity,state->kv_physical_page_capacity,(unsigned long long)state->kv_backing_maximum_bytes);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen38ModuleKvWaitBatch(SparkQwen38ModuleState *state, SparkQwen38WorkControlKvBatchState *batch)
{
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t polls = 0u;
	struct timespec pause;
	pause.tv_sec = 0;
	pause.tv_nsec = 500000;
	while ( batch->state == SPARK_QWEN38_WORK_CONTROL_BATCH_SUBMITTED )
	{
		status = SparkQwen38WorkControlProgress(&state->kv_client,&state->kv_work);
		if ( status != SPARK_STATUS_OK )
			return(status);
		if ( batch->state == SPARK_QWEN38_WORK_CONTROL_BATCH_READY )
			break;
		if ( ++polls >= SPARK_QWEN38_MODULE_KV_POLL_BOUND )
		{
			fprintf(stderr,"%s kv_store_stall\n",SPARK_QWEN38_MODULE_TAG);
			return(SPARK_STATUS_IO_ERROR);
		}
		nanosleep(&pause,0);
	}
	if ( batch->state != SPARK_QWEN38_WORK_CONTROL_BATCH_READY || batch->status != SPARK_STATUS_OK )
		return(SPARK_STATUS_IO_ERROR);
	return(SparkQwen38WorkControlAcknowledge(batch));
}

/* Write one dirty resident slot back to the store, leaving it free. */
static SparkStatus SparkQwen38ModuleKvEvictSlot(SparkQwen38ModuleState *state, uint32_t slot)
{
	SparkQwen38WorkControlKvBatchState *batch = &state->kv_work.evict;
	SparkKvStoreBlock blocks[1];
	uint32_t block_count = 0u,logical;
	uint64_t sequence_id;
	cudaError_t error;
	SparkStatus status;
	if ( state->kv_slot_dirty[slot] != 0u )
	{
		error = cudaMemcpy(state->kv_block_staging,(const uint8_t *)state->kv_cache_bf16 + (uint64_t)slot * state->kv_plan.block_record_bytes,(size_t)state->kv_plan.block_record_bytes,cudaMemcpyDeviceToHost);
		if ( error != cudaSuccess )
			return(SparkStageModuleCudaStatus(SPARK_QWEN38_MODULE_TAG,error,"kv_evict_copy"));
		logical = state->kv_slot_logical[slot];
		sequence_id = state->kv_slot_sequence[slot];
		status = SparkQwen38WorkControlBuildEvictBatch(&state->kv_plan,sequence_id,&logical,1u,0u,state->kv_block_staging,state->kv_gdn_staging,blocks,1u,&block_count);
		if ( status == SPARK_STATUS_OK )
			status = SparkQwen38WorkControlSubmit(&state->kv_client,batch,SPARK_KV_STORE_OPERATION_PUT,blocks,block_count,SPARK_QWEN38_WORK_CONTROL_RESTORE_PRIORITY_SPECULATIVE);
		if ( status == SPARK_STATUS_OK )
			status = SparkQwen38ModuleKvWaitBatch(state,batch);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	state->kv_slot_dirty[slot] = 0u;
	state->kv_slot_pinned[slot] = 0u;
	state->kv_slot_lane[slot] = UINT32_MAX;
	state->kv_slot_logical[slot] = UINT32_MAX;
	state->kv_slot_sequence[slot] = 0u;
	state->kv_slot_free_stack[state->kv_slot_free_count++] = slot;
	return(SPARK_STATUS_OK);
}

/* Make every block the frame's attention will read resident in the window,
 * rewrite the block table to window slots, and refresh the per-row slot
 * mappings. No-op when the tier is inactive. */
static SparkStatus SparkQwen38ModuleKvPrepareFrame(SparkQwen38ModuleState *state, SparkQwen38ModuleSlot *slot, SparkQwen38ResidentDecodeStageFrameContext *context, SparkQwen38KvBlockTableView *table, uint32_t rows)
{
	SparkQwen38WorkControlKvBatchState *restore_batch = &state->kv_work.restore;
	SparkKvStoreBlock blocks[SPARK_QWEN38_MODULE_KV_STAGING_RECORDS];
	uint32_t packet_lane_counts[1],block_count,lanes_built;
	uint32_t lane_required[SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_sequence[SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t lane_list[SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t lane_count = 0u,row,lane_index,logical,slot_index;
	uint64_t logical_capacity;
	SparkStatus status;
	cudaError_t error;
	if ( state->kv_tier_active == 0u )
		return(SPARK_STATUS_OK);
	if ( context == 0 || context->decode_batch == 0 || context->decode_batch->row_sequence_ids == 0 || table == 0 || table->host_physical_block_indices == 0 || table->host_lane_physical_block_counts == 0 || table->physical_block_indices == 0 || table->lane_physical_block_counts == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	logical_capacity = (uint64_t)table->lane_count * table->lane_stride;
	if ( logical_capacity == 0u || table->lane_count > SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT || table->lane_stride > SPARK_QWEN38_MODULE_KV_MAX_BLOCKS_PER_LANE )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( logical_capacity > state->kv_logical_to_slot_capacity )
	{
		uint32_t *grown = (uint32_t *)realloc(state->kv_logical_to_slot,(size_t)logical_capacity * sizeof(uint32_t));
		if ( grown == 0 )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		memset(grown + state->kv_logical_to_slot_capacity,0,(size_t)(logical_capacity - state->kv_logical_to_slot_capacity) * sizeof(uint32_t));
		state->kv_logical_to_slot = grown;
		state->kv_logical_to_slot_capacity = logical_capacity;
	}
	/* Pass 1: distinct lanes, required block counts, sequence ids, and pin
	 * every already-resident block the frame needs so eviction skips it. */
	for (row = 0u; row < rows; row++)
	{
		uint32_t lane = slot->host_row_lane_indices[row];
		uint32_t required_for_row = (slot->host_context_lengths[row] + SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS - 1u) / SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
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
	/* Pass 2: restore every nonresident block the frame needs, BATCHED.
	 * The per-block submit+wait of the first cut serialized one store
	 * round trip PER BLOCK - a 4096-block context restore was 4096
	 * latencies. Up to SPARK_QWEN38_MODULE_KV_STAGING_RECORDS blocks now
	 * ride one GET batch and one wait. */
	{
		SparkQwen38WorkControlPendingLane pending_lanes[SPARK_QWEN38_MODULE_KV_STAGING_RECORDS];
		uint32_t pending_slots[SPARK_QWEN38_MODULE_KV_STAGING_RECORDS];
		uint32_t pending_logical[SPARK_QWEN38_MODULE_KV_STAGING_RECORDS];
		uint64_t pending_lane_index[SPARK_QWEN38_MODULE_KV_STAGING_RECORDS];
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
					/* Round-robin eviction, skipping pinned (frame-needed) slots. */
					uint32_t scans = 0u;
					while ( state->kv_slot_pinned[state->kv_evict_cursor] != 0u )
					{
						state->kv_evict_cursor = (state->kv_evict_cursor + 1u) % state->kv_block_count;
						if ( ++scans > state->kv_block_count )
							return(SPARK_STATUS_CAPACITY_EXCEEDED);
					}
					status = SparkQwen38ModuleKvEvictSlot(state,state->kv_evict_cursor);
					if ( status != SPARK_STATUS_OK )
						return(status);
				}
			slot_index = state->kv_slot_free_stack[--state->kv_slot_free_count];
			pending_lanes[batch_block_count].sequence_id = lane_sequence[lane_index];
			pending_lanes[batch_block_count].nonresident_blocks = &pending_logical[batch_block_count];
			pending_lanes[batch_block_count].nonresident_block_count = 1u;
			pending_lanes[batch_block_count].gdn_nonresident = 0u;
			pending_logical[batch_block_count] = logical;
			pending_slots[batch_block_count] = slot_index;
			pending_lane_index[batch_block_count] = (uint64_t)lane_index;
			batch_block_count++;
			if ( batch_block_count == SPARK_QWEN38_MODULE_KV_STAGING_RECORDS )
			{
				packet_lane_counts[0] = batch_block_count;
				block_count = 0u;
				lanes_built = 0u;
				status = SparkQwen38WorkControlBuildRestoreBatch(&state->kv_plan,pending_lanes,batch_block_count,packet_lane_counts,1u,state->kv_block_staging,SPARK_QWEN38_MODULE_KV_STAGING_RECORDS,state->kv_gdn_staging,1u,blocks,SPARK_QWEN38_MODULE_KV_STAGING_RECORDS,&block_count,&lanes_built);
				if ( status == SPARK_STATUS_OK && lanes_built != batch_block_count )
					status = SPARK_STATUS_CAPACITY_EXCEEDED;
				if ( status == SPARK_STATUS_OK )
					status = SparkQwen38WorkControlSubmit(&state->kv_client,restore_batch,SPARK_KV_STORE_OPERATION_GET,blocks,block_count,SPARK_QWEN38_WORK_CONTROL_RESTORE_PRIORITY_IMMEDIATE);
				if ( status == SPARK_STATUS_OK )
					status = SparkQwen38ModuleKvWaitBatch(state,restore_batch);
				if ( status != SPARK_STATUS_OK )
					return(status);
				for (batch_index = 0u; batch_index < batch_block_count; batch_index++)
				{
					error = cudaMemcpyAsync((uint8_t *)state->kv_cache_bf16 + (uint64_t)pending_slots[batch_index] * state->kv_plan.block_record_bytes,(const uint8_t *)state->kv_block_staging + ((uint64_t)batch_index * state->kv_plan.block_record_bytes),(size_t)state->kv_plan.block_record_bytes,cudaMemcpyHostToDevice,(cudaStream_t)slot->cuda_stream);
					if ( error != cudaSuccess )
						return(SparkStageModuleCudaStatus(SPARK_QWEN38_MODULE_TAG,error,"kv_restore_copy"));
					state->kv_slot_lane[pending_slots[batch_index]] = lane_list[pending_lane_index[batch_index]];
					state->kv_slot_logical[pending_slots[batch_index]] = pending_logical[batch_index];
					state->kv_slot_sequence[pending_slots[batch_index]] = pending_lanes[batch_index].sequence_id;
					state->kv_slot_dirty[pending_slots[batch_index]] = 0u;
					state->kv_logical_to_slot[((uint64_t)lane_list[pending_lane_index[batch_index]] * table->lane_stride) + pending_logical[batch_index]] = pending_slots[batch_index] + 1u;
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
			status = SparkQwen38WorkControlBuildRestoreBatch(&state->kv_plan,pending_lanes,batch_block_count,packet_lane_counts,1u,state->kv_block_staging,SPARK_QWEN38_MODULE_KV_STAGING_RECORDS,state->kv_gdn_staging,1u,blocks,SPARK_QWEN38_MODULE_KV_STAGING_RECORDS,&block_count,&lanes_built);
			if ( status == SPARK_STATUS_OK && lanes_built != batch_block_count )
				status = SPARK_STATUS_CAPACITY_EXCEEDED;
			if ( status == SPARK_STATUS_OK )
				status = SparkQwen38WorkControlSubmit(&state->kv_client,restore_batch,SPARK_KV_STORE_OPERATION_GET,blocks,block_count,SPARK_QWEN38_WORK_CONTROL_RESTORE_PRIORITY_IMMEDIATE);
			if ( status == SPARK_STATUS_OK )
				status = SparkQwen38ModuleKvWaitBatch(state,restore_batch);
			if ( status != SPARK_STATUS_OK )
				return(status);
			for (batch_index = 0u; batch_index < batch_block_count; batch_index++)
			{
				error = cudaMemcpyAsync((uint8_t *)state->kv_cache_bf16 + (uint64_t)pending_slots[batch_index] * state->kv_plan.block_record_bytes,(const uint8_t *)state->kv_block_staging + ((uint64_t)batch_index * state->kv_plan.block_record_bytes),(size_t)state->kv_plan.block_record_bytes,cudaMemcpyHostToDevice,(cudaStream_t)slot->cuda_stream);
				if ( error != cudaSuccess )
					return(SparkStageModuleCudaStatus(SPARK_QWEN38_MODULE_TAG,error,"kv_restore_copy"));
				state->kv_slot_lane[pending_slots[batch_index]] = lane_list[pending_lane_index[batch_index]];
				state->kv_slot_logical[pending_slots[batch_index]] = pending_logical[batch_index];
				state->kv_slot_sequence[pending_slots[batch_index]] = pending_lanes[batch_index].sequence_id;
				state->kv_slot_dirty[pending_slots[batch_index]] = 0u;
				state->kv_logical_to_slot[((uint64_t)lane_list[pending_lane_index[batch_index]] * table->lane_stride) + pending_logical[batch_index]] = pending_slots[batch_index] + 1u;
			}
		}
	}
	/* Pass 3: build the rewritten table in the module's own buffers (the
	 * adapter's host mirror stays logical), point the frame's view at them,
	 * upload ONLY the touched lanes' slices (the full-table upload of the
	 * first cut moved 8 MB per frame; a lane slice is 16 KB), and refresh
	 * the row slot mappings ahead of the layer walk. */
	for (lane_index = 0u; lane_index < lane_count; lane_index++)
	{
		uint32_t lane = lane_list[lane_index];
		uint64_t lane_slice = (uint64_t)lane * table->lane_stride;
		memcpy(state->kv_table_indices_host + lane_slice,table->host_physical_block_indices + lane_slice,(size_t)table->lane_stride * sizeof(uint32_t));
		for (logical = 0u; logical < lane_required[lane_index]; logical++)
			state->kv_table_indices_host[lane_slice + logical] = state->kv_logical_to_slot[lane_slice + logical] - 1u;
		error = cudaMemcpyAsync((uint8_t *)state->kv_table_indices_device + (lane_slice * sizeof(uint32_t)),state->kv_table_indices_host + lane_slice,(size_t)table->lane_stride * sizeof(uint32_t),cudaMemcpyHostToDevice,(cudaStream_t)slot->cuda_stream);
		if ( error != cudaSuccess )
			return(SparkStageModuleCudaStatus(SPARK_QWEN38_MODULE_TAG,error,"kv_table_upload"));
	}
	table->physical_block_indices = state->kv_table_indices_device;
	table->lane_physical_block_counts = state->kv_table_counts_device;
	error = cudaMemcpyAsync((void *)state->kv_table_counts_device,(const void *)table->host_lane_physical_block_counts,(size_t)table->lane_count * sizeof(uint32_t),cudaMemcpyHostToDevice,(cudaStream_t)slot->cuda_stream);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_QWEN38_MODULE_TAG,error,"kv_table_upload"));
	for (row = 0u; row < rows; row++)
	{
		uint32_t lane = slot->host_row_lane_indices[row];
		uint64_t position = slot->host_row_positions[row];
		slot_index = state->kv_logical_to_slot[((uint64_t)lane * table->lane_stride) + (uint32_t)(position / SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS)] - 1u;
		slot->host_slot_mapping[row] = slot_index * SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS + (uint32_t)(position % SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
	}
	error = cudaMemcpyAsync(slot->slot_mapping,slot->host_slot_mapping,(size_t)rows * sizeof(uint32_t),cudaMemcpyHostToDevice,(cudaStream_t)slot->cuda_stream);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_QWEN38_MODULE_TAG,error,"kv_slot_upload"));
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
}

/* The frame just wrote K/V rows into window slots: mark them dirty so the
 * store receives them before the slots are reused. */
static void SparkQwen38ModuleKvMarkWritten(SparkQwen38ModuleState *state, SparkQwen38ModuleSlot *slot, uint32_t rows)
{
	uint32_t row,slot_index;
	if ( state->kv_tier_active == 0u )
		return;
	for (row = 0u; row < rows; row++)
	{
		slot_index = slot->host_slot_mapping[row] / SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
		if ( slot_index < state->kv_block_count )
			state->kv_slot_dirty[slot_index] = 1u;
	}
}

extern cudaError_t SparkQwen38LaunchHeadArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint32_t *token_ids, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count);
extern cudaError_t SparkQwen38LaunchHeadShadowQuantize(cudaStream_t stream, const void *head_bf16, uint8_t *shadow_payload, uint8_t *shadow_scale, float *error_norm, uint32_t candidate_count, uint32_t hidden_dimension);
extern cudaError_t SparkQwen38LaunchHeadScreenedArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint8_t *shadow_payload, const uint8_t *shadow_scale, const float *error_norm, void *logits_bf16, uint32_t *candidate_ids, uint32_t *candidate_counts, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count);
extern cudaError_t SparkQwen38LaunchTpCombineAdd(cudaStream_t stream, void *destination_bf16, const void *source_bf16, uint32_t row_count, uint32_t width);

/* --------------------------------------------------------------------------
 * Tensor-parallel collective: one residual all-reduce per layer. The
 * combine callback runs the elementwise add kernel; the module waits on an
 * atomic flag set by the completion callback (the device collective's own
 * progress thread drives the transfer phases).
 * ------------------------------------------------------------------------*/

static SparkStatus SparkQwen38ModuleTpCombineBf16(void *combine_context, void *destination_device, const void *source_device, uint32_t active_sequence_count, uint32_t hidden_dimension, void *cuda_stream)
{
	(void)combine_context;
	return(SparkStageModuleCudaStatus(SPARK_QWEN38_MODULE_TAG,SparkQwen38LaunchTpCombineAdd((cudaStream_t)cuda_stream,destination_device,source_device,active_sequence_count,hidden_dimension),"tp_combine"));
}

static void SparkQwen38ModuleTpCompletion(void *context, const SparkTpDeviceCollectiveCompletion *completion)
{
	atomic_uint *flag = (atomic_uint *)context;
	atomic_store_explicit(flag,completion != 0 && completion->status == SPARK_STATUS_OK ? 1u : 2u,memory_order_release);
}

static SparkStatus SparkQwen38ModuleInitializeTpCollective(SparkQwen38ModuleState *state)
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
	topology.algorithm_mask = SPARK_TP_DEVICE_COLLECTIVE_ALGORITHM_RECURSIVE_DOUBLING;
	topology.rail_count = 0u;
	topology.direct_all_to_all_max_payload_bytes = 0u;
	topology.split_ring_min_payload_bytes = 0u;
	for (rank = 0u; rank < state->tp_degree; rank++)
		memcpy(topology.rank_hosts[rank],state->tp_hosts[rank],SPARK_TP_DEVICE_COLLECTIVE_HOST_NAME_BYTES);
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	configuration.backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
	configuration.tp_degree = state->tp_degree;
	configuration.tp_rank = state->tp_rank;
	configuration.operation_kind = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	configuration.credit_count = 2u * state->pipeline_slot_count;
	configuration.local_hidden_dimension = SPARK_QWEN38_MODEL_HIDDEN_DIMENSION;
	/* The backend sizes its credit planes by this; use the static maximum
	 * (dsv4 does the same) so the .so contract is configuration-free. */
	configuration.max_active_sequence_count = SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT;
	configuration.connect_timeout_milli = state->tp_connect_timeout_milli;
	configuration.operation_timeout_milli = state->tp_operation_timeout_milli;
	configuration.control_port_base = state->tp_control_port_base;
	configuration.collective_identifier = state->tp_collective_identifier;
	configuration.backend_module_path = state->tp_backend_path;
	configuration.local_host = state->tp_local_host;
	configuration.registration_cuda_stream = state->slots[0].cuda_stream;
	configuration.combine_bf16_function = SparkQwen38ModuleTpCombineBf16;
	configuration.combine_context = state;
	status = SparkTpDeviceCollectiveApplyTopology(&topology,&configuration);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s tp_apply_topology_failed status=%d\n",SPARK_QWEN38_MODULE_TAG,(int)status);
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
		fprintf(stderr,"%s tp_probe_memory_mode_failed status=%d\n",SPARK_QWEN38_MODULE_TAG,(int)status);
		return(status);
	}
	credit_bytes = (uint64_t)SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT * SPARK_QWEN38_MODEL_HIDDEN_DIMENSION * SPARK_QWEN38_MODEL_BF16_ELEMENT_BYTES;
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
			return(SparkStageModuleCudaStatus(SPARK_QWEN38_MODULE_TAG,error,"tp_credit_mapped_alloc"));
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
		fprintf(stderr,"%s tp_create_failed status=%d\n",SPARK_QWEN38_MODULE_TAG,(int)status);
		return(status);
	}
	state->tp_collective_initialized = 1u;
	fprintf(stderr,"%s tp_collective_open degree=%u rank=%u port_base=%u\n",SPARK_QWEN38_MODULE_TAG,state->tp_degree,state->tp_rank,state->tp_control_port_base);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen38ModuleTpAllReduceHidden(SparkQwen38ModuleState *state, SparkQwen38ModuleSlot *slot, void *device_bf16, uint32_t rows)
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
	/* Stream-ordered: the backend orders its device work after the
	 * pair-reduce on the slot stream before the completion fires. */
	submission.flags = SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission.ordinal = atomic_fetch_add_explicit(&state->tp_next_ordinal,1u,memory_order_relaxed);
	submission.local_device = device_bf16;
	submission.full_device = device_bf16;
	submission.cuda_stream = slot->cuda_stream;
	submission.completion_function = SparkQwen38ModuleTpCompletion;
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
	fprintf(stderr,"%s tp_all_reduce_stall\n",SPARK_QWEN38_MODULE_TAG);
	return(SPARK_STATUS_IO_ERROR);
}

SparkStatus SparkQwen38ResidentDecodeStageInitialize(
    const SparkFirmwareModuleConfiguration *configuration,
    const SparkFirmwareModuleHostServices *host_services,
    void **module_state)
{
	SparkQwen38ModuleState *state;
	const char *pack_path;
	uint32_t allow_unqualified_execution;
	SparkStatus status;
	pack_path = 0;
	allow_unqualified_execution = 0u;
	status = SparkFirmwareModuleValidateInitialization(configuration,host_services,module_state);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkStageModuleEnvironmentUnsigned(SPARK_QWEN38_MODULE_TAG,"SPARK_QWEN38_ALLOW_UNQUALIFIED_EXECUTION",1u,1u,&allow_unqualified_execution);
	if ( status != SPARK_STATUS_OK || allow_unqualified_execution != 1u )
		return(SPARK_STATUS_MODULE_NOT_VALIDATED);
	state = (SparkQwen38ModuleState *)calloc(1u,sizeof(*state));
	if ( state == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->allow_unqualified_execution = allow_unqualified_execution;
	state->ledger.module_tag = SPARK_QWEN38_MODULE_TAG;
	atomic_init(&state->submitted_count,0u);
	atomic_init(&state->completed_count,0u);
	atomic_init(&state->tp_completion_flag,0u);
	atomic_init(&state->tp_next_ordinal,0u);
	atomic_init(&state->rejected_count,0u);
	atomic_init(&state->failed_count,0u);
	atomic_init(&state->tokens_emitted,0u);
	status = SparkQwen38ModuleConfigure(state);
	/* tp_degree > 1 runs the expert-sharded MoE with one residual
	 * all-reduce per layer (the collective opens right after the slot
	 * allocation); the head-parallel attention/GDN slicing is the next
	 * increment. tp=1 is the replicated path, byte-identical as before. */
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentText(SPARK_QWEN38_MODULE_TAG,"SPARK_QWEN38_STAGE_PACK_PATH",&pack_path);
	if ( status == SPARK_STATUS_OK )
	{
		SparkQwen38ModuleBuildOrdinals(state);
		status = SparkQwen38ModuleLoadPack(state,pack_path);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen38ModuleOpenKvTier(state,host_services);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_QWEN38_MODULE_TAG,SparkQwen38ConfigureCudaKernels(),"configure_cuda_kernels");
	if ( status == SPARK_STATUS_OK )
	{
		int32_t sm_count = 0;
		cudaError_t attr = cudaDeviceGetAttribute(&sm_count,cudaDevAttrMultiProcessorCount,0);
		state->multiprocessor_count = attr == cudaSuccess && sm_count > 0 ? (uint32_t)sm_count : 1u;
		status = SparkQwen38ModuleAllocatePools(state);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen38ModuleAllocateSlot(state,&state->slots[0]);
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen38ModuleAllocateSlotHostMirrors(state,&state->slots[0]);
	if ( status == SPARK_STATUS_OK && state->tp_degree > 1u )
		status = SparkQwen38ModuleInitializeTpCollective(state);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
	{
		/* One-time 4-bit shadow of the 248320 x 8192 head weights plus
		 * certified per-neuron error bounds: the screened argmax reads
		 * the shadow (1.02 GB) instead of the bf16 head (4.07 GB) per
		 * row and rescores a bounded candidate set exactly. */
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)SPARK_QWEN38_MODEL_OUTPUT_VOCAB_COUNT * SPARK_QWEN38_MODEL_HIDDEN_DIMENSION / 2u,(void **)&state->head_shadow_payload);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)SPARK_QWEN38_MODEL_OUTPUT_VOCAB_COUNT * (SPARK_QWEN38_MODEL_HIDDEN_DIMENSION / SPARK_QWEN38_MODULE_HEAD_SHADOW_GROUP),(void **)&state->head_shadow_scale);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,SPARK_QWEN38_MODEL_OUTPUT_VOCAB_COUNT * sizeof(float),(void **)&state->head_error_norm_f32);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(SPARK_QWEN38_MODULE_TAG,SparkQwen38LaunchHeadShadowQuantize((cudaStream_t)state->slots[0].cuda_stream,state->lm_head_weight_bf16,state->head_shadow_payload,state->head_shadow_scale,state->head_error_norm_f32,SPARK_QWEN38_MODEL_OUTPUT_VOCAB_COUNT,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION),"head_shadow_quantize");
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleCudaStatus(SPARK_QWEN38_MODULE_TAG,cudaStreamSynchronize((cudaStream_t)state->slots[0].cuda_stream),"head_shadow_sync");
	}
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s initialize_failed status=%d\n",SPARK_QWEN38_MODULE_TAG,(int)status);
		SparkQwen38ResidentDecodeStageDestroy(state);
		return(status);
	}
	*module_state = state;
	fprintf(stderr,"%s initialize ok slice=%u+%u gdn=%u attn=%u owns_embedding=%u owns_head=%u\n",SPARK_QWEN38_MODULE_TAG,state->first_layer_index,state->layer_count,state->gdn_layer_count,state->attn_layer_count,state->owns_embedding,state->owns_final_head);
	return(SPARK_STATUS_OK);
}



SparkStatus SparkQwen38ResidentDecodeStageAdmit(
    void *module_state,
    const SparkModelDriverAdmissionRequest *request,
    SparkModelDriverAdmissionDecision *decision)
{
	(void)module_state;
	(void)request;
	(void)decision;
	return(SPARK_STATUS_UNSUPPORTED);
}

SparkStatus SparkQwen38ResidentDecodeStageSnapshot(
    void *module_state,
    uint32_t program_id,
    SparkModelDriverRuntimeSnapshot *snapshot)
{
	(void)module_state;
	(void)program_id;
	(void)snapshot;
	return(SPARK_STATUS_UNSUPPORTED);
}

void SparkQwen38ResidentDecodeStageDestroy(void *module_state)
{
	SparkQwen38ModuleState *state = (SparkQwen38ModuleState *)module_state;
	if ( state == 0 )
		return;
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
	SparkStageModuleLedgerRelease(&state->ledger);
	free(state);
}

/* ---------------------------------------------------------------------------
 * Execution core: slot allocation and the per-layer runners.
 * -------------------------------------------------------------------------*/

#define SPARK_QWEN38_MODULE_STAGED_ROW_CAPACITY \
	(SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT + SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS)



extern cudaError_t SparkQwen38LaunchEmbeddingGather(cudaStream_t stream, const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count);
extern cudaError_t SparkQwen38LaunchRmsNorm(cudaStream_t stream, const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon);
extern cudaError_t SparkQwen38LaunchFusedResidualRmsNorm(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon);
extern cudaError_t SparkQwen38LaunchLinear(cudaStream_t stream, const SparkQwen38LinearView *view, const void *input_bf16, void *output_bf16, uint32_t row_count);
extern cudaError_t SparkQwen38LaunchConvUpdate(cudaStream_t stream, const void *qkv_bf16, const SparkQwen38GdnLayerWeights *weights, void *conv_out_bf16, const SparkQwen38GdnStatePool *pool, const uint32_t *row_lane_indices, uint32_t row_count, uint32_t gdn_layer_ordinal);
extern cudaError_t SparkQwen38LaunchDecayBeta(cudaStream_t stream, const void *decay_pre_bf16, const void *beta_pre_bf16, const SparkQwen38GdnLayerWeights *weights, float *log_decay_f32, float *beta_f32, uint32_t row_count);
extern cudaError_t SparkQwen38LaunchGdnStep(cudaStream_t stream, const void *conv_out_bf16, const float *log_decay_f32, const float *beta_f32, const SparkQwen38GdnStatePool *pool, void *core_out_bf16, const uint32_t *row_lane_indices, uint32_t row_count, uint32_t gdn_layer_ordinal);
extern cudaError_t SparkQwen38LaunchGatedNorm(cudaStream_t stream, const void *core_bf16, const void *z_bf16, const SparkQwen38GdnLayerWeights *weights, void *output_bf16, uint32_t row_count, float epsilon);
extern cudaError_t SparkQwen38LaunchAttnPrepare(cudaStream_t stream, void *q_fused_bf16, const void *k_bf16, const void *v_bf16, const SparkQwen38AttnLayerWeights *weights, void *kv_cache_bf16, const uint32_t *slot_mapping, const uint64_t *row_positions, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, float epsilon, uint32_t tp_degree, uint32_t tp_rank);
extern cudaError_t SparkQwen38LaunchAttnDecode(cudaStream_t stream, const void *q_fused_bf16, const void *kv_cache_bf16, const SparkQwen38KvBlockTableView *table, const uint32_t *row_lane_indices, const uint32_t *context_lengths, void *head_out_bf16, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, uint32_t tp_degree, uint32_t tp_rank);
extern cudaError_t SparkQwen38LaunchResidualAdd(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, uint32_t row_count, uint32_t dimension);
extern cudaError_t SparkQwen38LaunchGateScores(cudaStream_t stream, const SparkQwen38LinearView *gate, const void *input_bf16, float *scores_f32, uint32_t row_count);
extern cudaError_t SparkQwen38LaunchGateSelect(cudaStream_t stream, const float *scores_f32, const float *bias_f32, uint32_t row_count, uint32_t expert_count, uint32_t topk, float route_scale, uint32_t *indices_u32, float *weights_f32);
extern cudaError_t SparkQwen38LaunchMoeRoute(cudaStream_t stream, const uint32_t *route_expert, uint32_t rows, uint32_t expert_width, uint32_t *group_row_offset, uint32_t *route_packed_row, uint32_t *route_source_token, uint32_t *group_tile_prefix_w1, uint32_t *group_tile_prefix_w2);
extern cudaError_t SparkQwen38LaunchFusedExpertW13Act(cudaStream_t stream, const SparkQwen38LinearView *w1, const SparkQwen38LinearView *w3, const void *input_bf16, const uint32_t *route_source_token, const uint32_t *group_row_offset, uint32_t *group_tile_prefix, void *activated_bf16, uint32_t rows, uint32_t expert_width, float limit, uint32_t multiprocessor_count);
extern cudaError_t SparkQwen38LaunchExpertDown(cudaStream_t stream, const SparkQwen38LinearView *stacked, const void *input_bf16, const uint32_t *group_row_offset, uint32_t *group_tile_prefix, void *output_bf16, uint32_t rows, uint32_t expert_width, uint32_t hidden_dimension, uint32_t multiprocessor_count);
extern cudaError_t SparkQwen38LaunchMoePairReduce(cudaStream_t stream, const void *slot_out_bf16, const uint32_t *inverse_map, const float *pair_weights_f32, void *accum_bf16, uint32_t row_count, uint32_t hidden_dimension);
extern cudaError_t SparkQwen38LaunchMoePairReduceOverwrite(cudaStream_t stream, const void *slot_out_bf16, const uint32_t *inverse_map, const float *pair_weights_f32, void *output_bf16, uint32_t row_count, uint32_t hidden_dimension);
extern cudaError_t SparkQwen38LaunchGroupedExpertLinear(cudaStream_t stream, const SparkQwen38LinearView *view, const void *input_bf16, const uint32_t *source_row_map, const uint32_t *group_row_offset, const uint32_t *group_tile_prefix, void *output_bf16, uint32_t source_row_count, uint32_t multiprocessor_count, uint32_t tp_degree, uint32_t tp_rank);
extern cudaError_t SparkQwen38LaunchGroupedExpertTileLinear(cudaStream_t stream, const SparkQwen38LinearView *view, const void *input_bf16, const uint32_t *source_row_map, const uint32_t *group_row_offset, void *output_bf16, uint32_t source_row_count, uint32_t tp_degree, uint32_t tp_rank);
/* The MoE switches from the scalar grouped path to the tensor-core tile
 * path (SparkLmExpertTileAllKernel / SparkLmExpertTileAllMloopKernel) at
 * 8 rows. Measured on spark4: tile-at-1/2/4 LOSES (the 512-expert grid's
 * ~8K dead CTAs dominate a tiny batch), tile-at-8 wins 36.4 -> 21.7 ms,
 * tile-at-12 wins 54.5 -> 23.9 ms. The dense linears use the force-tile
 * path at every batch instead (no expert dimension, no dead CTAs). */
#define SPARK_QWEN38_MODULE_MOE_TILE_ROWS 8u
extern cudaError_t SparkQwen38LaunchSwiGlu(cudaStream_t stream, const void *gate_bf16, void *up_bf16, uint32_t row_count, uint32_t dimension);
extern cudaError_t SparkQwen38LaunchSharedGate(cudaStream_t stream, void *accum_bf16, const void *gate_weight_bf16, const void *gate_input_bf16, uint32_t row_count, uint32_t dimension);

static SparkStatus SparkQwen38ModuleAllocateSlot(SparkQwen38ModuleState *state, SparkQwen38ModuleSlot *slot)
{
	uint64_t rows = state->max_active_sequence_count;
	uint64_t hidden_bytes = rows * SPARK_QWEN38_MODEL_HIDDEN_DIMENSION * SPARK_QWEN38_MODEL_BF16_ELEMENT_BYTES;
	uint64_t expert_bytes = rows * SPARK_QWEN38_MODEL_EXPERT_INTERMEDIATE_DIMENSION * SPARK_QWEN38_MODEL_BF16_ELEMENT_BYTES;
	uint64_t moe_up_bytes = rows * SPARK_QWEN38_MODEL_EXPERTS_PER_TOKEN * SPARK_QWEN38_MODEL_EXPERT_INTERMEDIATE_DIMENSION * SPARK_QWEN38_MODEL_BF16_ELEMENT_BYTES;
	/* The w2 output holds rows * top_k packed rows at the HIDDEN width,
	 * not the expert width - the pair reduce reads it back at hidden. */
	uint64_t moe_down_bytes = rows * SPARK_QWEN38_MODEL_EXPERTS_PER_TOKEN * SPARK_QWEN38_MODEL_HIDDEN_DIMENSION * SPARK_QWEN38_MODEL_BF16_ELEMENT_BYTES;
	SparkStatus status;
	cudaStream_t stream = 0;
	status = SparkStageModuleCudaStatus(SPARK_QWEN38_MODULE_TAG,cudaStreamCreate(&stream),"cudaStreamCreate");
	if ( status != SPARK_STATUS_OK )
		return(status);
	slot->cuda_stream = stream;
	status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->input_token_ids);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->output_token_ids);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
	{
		/* Screened head: coarse logits at bf16 rows x vocab, screen
		 * candidates at rows x SPARK_LM_HEAD_SCREEN_CAP, counts. */
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN38_MODEL_OUTPUT_VOCAB_COUNT * SPARK_QWEN38_MODEL_BF16_ELEMENT_BYTES,(void **)&slot->head_logits_bf16);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN38_MODULE_HEAD_SCREEN_CAP * sizeof(uint32_t),(void **)&slot->head_candidate_ids);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->head_candidate_counts);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->row_lane_indices);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->slot_mapping);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->context_lengths);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint32_t),(void **)&slot->row_cold);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * sizeof(uint64_t),(void **)&slot->row_positions);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,hidden_bytes,&slot->hidden_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,hidden_bytes,&slot->normalized_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,hidden_bytes,&slot->delta_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN38_MODEL_GDN_CONV_CHANNELS * SPARK_QWEN38_MODEL_BF16_ELEMENT_BYTES,&slot->qkv_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN38_MODEL_GDN_CONV_CHANNELS * SPARK_QWEN38_MODEL_BF16_ELEMENT_BYTES,&slot->conv_out_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN38_MODEL_GDN_VALUE_DIMENSION * SPARK_QWEN38_MODEL_BF16_ELEMENT_BYTES,&slot->z_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN38_MODEL_GDN_VALUE_HEAD_COUNT * SPARK_QWEN38_MODEL_BF16_ELEMENT_BYTES,&slot->beta_pre_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN38_MODEL_GDN_VALUE_HEAD_COUNT * SPARK_QWEN38_MODEL_BF16_ELEMENT_BYTES,&slot->decay_pre_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN38_MODEL_GDN_VALUE_HEAD_COUNT * sizeof(float),(void **)&slot->log_decay_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN38_MODEL_GDN_VALUE_HEAD_COUNT * sizeof(float),(void **)&slot->beta_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN38_MODEL_GDN_VALUE_DIMENSION * SPARK_QWEN38_MODEL_BF16_ELEMENT_BYTES,&slot->core_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN38_MODEL_GDN_VALUE_DIMENSION * SPARK_QWEN38_MODEL_BF16_ELEMENT_BYTES,&slot->gated_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * 2u * SPARK_QWEN38_MODEL_ATTN_QUERY_DIMENSION * SPARK_QWEN38_MODEL_BF16_ELEMENT_BYTES,&slot->q_fused_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN38_MODEL_ATTN_KV_DIMENSION * SPARK_QWEN38_MODEL_BF16_ELEMENT_BYTES,&slot->k_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN38_MODEL_ATTN_KV_DIMENSION * SPARK_QWEN38_MODEL_BF16_ELEMENT_BYTES,&slot->v_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN38_MODEL_ATTN_QUERY_DIMENSION * SPARK_QWEN38_MODEL_BF16_ELEMENT_BYTES,&slot->head_out_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN38_MODEL_ROUTED_EXPERT_COUNT * sizeof(float),(void **)&slot->moe_scores_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN38_MODEL_EXPERTS_PER_TOKEN * sizeof(uint32_t),(void **)&slot->moe_indices_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,rows * SPARK_QWEN38_MODEL_EXPERTS_PER_TOKEN * sizeof(float),(void **)&slot->moe_weights_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)(rows * SPARK_QWEN38_MODEL_EXPERTS_PER_TOKEN + SPARK_QWEN38_MODEL_ROUTED_EXPERT_COUNT + 2u) * sizeof(uint32_t),(void **)&slot->moe_inverse_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)(rows * SPARK_QWEN38_MODEL_EXPERTS_PER_TOKEN + SPARK_QWEN38_MODEL_ROUTED_EXPERT_COUNT + 2u) * sizeof(uint32_t),(void **)&slot->moe_grouped_rows_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)(SPARK_QWEN38_MODEL_ROUTED_EXPERT_COUNT + 1u) * sizeof(uint32_t),(void **)&slot->moe_tile_prefix_w1_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)(SPARK_QWEN38_MODEL_ROUTED_EXPERT_COUNT + 1u) * sizeof(uint32_t),(void **)&slot->moe_tile_prefix_w2_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,(uint64_t)(SPARK_QWEN38_MODEL_ROUTED_EXPERT_COUNT + 1u) * sizeof(uint32_t),(void **)&slot->moe_group_offset_u32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,moe_up_bytes,&slot->moe_gate_packed_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,moe_up_bytes,&slot->moe_slot_up_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,moe_down_bytes,&slot->moe_slot_out_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,expert_bytes,&slot->shared_gate_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,expert_bytes,&slot->shared_up_bf16);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleDeviceAllocate(&state->ledger,hidden_bytes,&slot->shared_down_bf16);
	return(status);
}

static cudaError_t SparkQwen38ModuleRunGdnCoreDecode(SparkQwen38ModuleState *state, SparkQwen38ModuleSlot *slot, const SparkQwen38GdnLayerWeights *weights, uint32_t rows, uint32_t ordinal)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	SparkQwen38GdnStatePool pool = state->gdn_pool;
	pool.state_cold_by_row = slot->row_cold;
	error = SparkQwen38LaunchConvUpdate(stream,slot->qkv_bf16,weights,slot->conv_out_bf16,&pool,slot->row_lane_indices,rows,ordinal);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchDecayBeta(stream,slot->decay_pre_bf16,slot->beta_pre_bf16,weights,slot->log_decay_f32,slot->beta_f32,rows);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchGdnStep(stream,slot->conv_out_bf16,slot->log_decay_f32,slot->beta_f32,&pool,slot->core_bf16,slot->row_lane_indices,rows,ordinal);
	return(error);
}

static SparkStatus SparkQwen38ModuleRunGdnLayer(SparkQwen38ModuleState *state, SparkQwen38ModuleSlot *slot, uint32_t layer, uint32_t rows)
{
	const SparkQwen38GdnLayerWeights *weights = &state->gdn_by_layer[layer];
	uint32_t ordinal = state->gdn_ordinal_by_layer[layer];
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	error = SparkQwen38LaunchLinear(stream,&weights->qkv,slot->normalized_bf16,slot->qkv_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchLinear(stream,&weights->gate,slot->normalized_bf16,slot->z_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchLinear(stream,&weights->beta,slot->normalized_bf16,slot->beta_pre_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchLinear(stream,&weights->decay,slot->normalized_bf16,slot->decay_pre_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen38ModuleRunGdnCoreDecode(state,slot,weights,rows,ordinal);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchGatedNorm(stream,slot->core_bf16,slot->z_bf16,weights,slot->gated_bf16,rows,SPARK_QWEN38_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchLinear(stream,&weights->output,slot->gated_bf16,slot->delta_bf16,rows);
	return(SparkStageModuleCudaStatus(SPARK_QWEN38_MODULE_TAG,error,"gdn_layer"));
}

typedef struct SparkQwen38AttnRowsView
{
	const uint32_t *slot_mapping;
	const uint64_t *row_positions;
	const uint32_t *row_lane_indices;
	const uint32_t *context_lengths;
} SparkQwen38AttnRowsView;

static SparkStatus SparkQwen38ModuleRunAttnLayer(SparkQwen38ModuleState *state, SparkQwen38ModuleSlot *slot, const SparkQwen38KvBlockTableView *table, const SparkQwen38AttnLayerWeights *weights, uint32_t ordinal, const SparkQwen38AttnRowsView *rows_view, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	error = SparkQwen38LaunchLinear(stream,&weights->query,slot->normalized_bf16,slot->q_fused_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchLinear(stream,&weights->key,slot->normalized_bf16,slot->k_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchLinear(stream,&weights->value,slot->normalized_bf16,slot->v_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchAttnPrepare(stream,slot->q_fused_bf16,slot->k_bf16,slot->v_bf16,weights,state->kv_cache_bf16,rows_view->slot_mapping,rows_view->row_positions,rows,ordinal,state->cache_layer_stride,state->cache_block_stride,SPARK_QWEN38_MODEL_RMS_NORM_EPSILON,state->tp_degree,state->tp_rank);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchAttnDecode(stream,slot->q_fused_bf16,state->kv_cache_bf16,table,rows_view->row_lane_indices,rows_view->context_lengths,slot->head_out_bf16,rows,ordinal,state->cache_layer_stride,state->cache_block_stride,state->tp_degree,state->tp_rank);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchLinear(stream,&weights->output,slot->head_out_bf16,slot->delta_bf16,rows);
	return(SparkStageModuleCudaStatus(SPARK_QWEN38_MODULE_TAG,error,"attn_layer"));
}

/*
 * Routed MoE + shared expert, the layer's FFN side. The fused residual norm
 * already folded the attention/GDN delta into hidden, so delta_bf16 is
 * OVERWRITTEN by the routed expert mixture (weighted pair reduce from zero)
 * and then accumulates the shared expert output (scalar-gate-multiplied)
 * before the final residual add.
 */
static SparkStatus SparkQwen38ModuleRunMoe(SparkQwen38ModuleState *state, SparkQwen38ModuleSlot *slot, const void *mlp_norm_bf16, const SparkQwen38MoeWeights *weights, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	SparkStatus status;
	error = SparkQwen38LaunchFusedResidualRmsNorm(stream,slot->hidden_bf16,slot->delta_bf16,mlp_norm_bf16,slot->normalized_bf16,rows,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION,SPARK_QWEN38_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchGateScores(stream,&weights->gate,slot->normalized_bf16,slot->moe_scores_f32,rows);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchGateSelect(stream,slot->moe_scores_f32,0,rows,SPARK_QWEN38_MODEL_ROUTED_EXPERT_COUNT,SPARK_QWEN38_MODEL_EXPERTS_PER_TOKEN,1.0f,slot->moe_indices_u32,slot->moe_weights_f32);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchMoeRoute(stream,slot->moe_indices_u32,rows,SPARK_QWEN38_MODEL_EXPERT_INTERMEDIATE_DIMENSION,slot->moe_group_offset_u32,slot->moe_inverse_u32,slot->moe_grouped_rows_u32,slot->moe_tile_prefix_w1_u32,slot->moe_tile_prefix_w2_u32);
	if ( error == cudaSuccess )
	{
		if ( rows >= SPARK_QWEN38_MODULE_MOE_TILE_ROWS )
		{
			/* Tensor-core tile path: gridDim.z spans the experts, FP8
			 * decodes to BF16 fragments under wmma mma_sync. */
			error = SparkQwen38LaunchGroupedExpertTileLinear(stream,&weights->experts_w1,slot->normalized_bf16,slot->moe_grouped_rows_u32,slot->moe_group_offset_u32,slot->moe_gate_packed_bf16,rows,state->tp_degree,state->tp_rank);
			if ( error == cudaSuccess )
				error = SparkQwen38LaunchGroupedExpertTileLinear(stream,&weights->experts_w3,slot->normalized_bf16,slot->moe_grouped_rows_u32,slot->moe_group_offset_u32,slot->moe_slot_up_bf16,rows,state->tp_degree,state->tp_rank);
			if ( error == cudaSuccess )
				error = SparkQwen38LaunchSwiGlu(stream,slot->moe_gate_packed_bf16,slot->moe_slot_up_bf16,rows * SPARK_QWEN38_MODEL_EXPERTS_PER_TOKEN,SPARK_QWEN38_MODEL_EXPERT_INTERMEDIATE_DIMENSION);
			if ( error == cudaSuccess )
				error = SparkQwen38LaunchGroupedExpertTileLinear(stream,&weights->experts_w2,slot->moe_slot_up_bf16,0,slot->moe_group_offset_u32,slot->moe_slot_out_bf16,rows * SPARK_QWEN38_MODEL_EXPERTS_PER_TOKEN,state->tp_degree,state->tp_rank);
		}
		else
		{
			error = SparkQwen38LaunchGroupedExpertLinear(stream,&weights->experts_w1,slot->normalized_bf16,slot->moe_grouped_rows_u32,slot->moe_group_offset_u32,slot->moe_tile_prefix_w1_u32,slot->moe_gate_packed_bf16,rows,state->multiprocessor_count,state->tp_degree,state->tp_rank);
			if ( error == cudaSuccess )
				error = SparkQwen38LaunchGroupedExpertLinear(stream,&weights->experts_w3,slot->normalized_bf16,slot->moe_grouped_rows_u32,slot->moe_group_offset_u32,slot->moe_tile_prefix_w1_u32,slot->moe_slot_up_bf16,rows,state->multiprocessor_count,state->tp_degree,state->tp_rank);
			if ( error == cudaSuccess )
				error = SparkQwen38LaunchSwiGlu(stream,slot->moe_gate_packed_bf16,slot->moe_slot_up_bf16,rows * SPARK_QWEN38_MODEL_EXPERTS_PER_TOKEN,SPARK_QWEN38_MODEL_EXPERT_INTERMEDIATE_DIMENSION);
			if ( error == cudaSuccess )
				error = SparkQwen38LaunchGroupedExpertLinear(stream,&weights->experts_w2,slot->moe_slot_up_bf16,0,slot->moe_group_offset_u32,slot->moe_tile_prefix_w2_u32,slot->moe_slot_out_bf16,rows * SPARK_QWEN38_MODEL_EXPERTS_PER_TOKEN,state->multiprocessor_count,state->tp_degree,state->tp_rank);
		}
	}
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchMoePairReduceOverwrite(stream,slot->moe_slot_out_bf16,slot->moe_inverse_u32,slot->moe_weights_f32,slot->delta_bf16,rows,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION);
	if ( error == cudaSuccess && state->tp_degree > 1u )
	{
		/* Expert-sharded TP: the pair reduce holds only THIS rank's
		 * experts' contribution. Reduce the DELTA before the replicated
		 * shared expert and the residual add join it - reducing the hidden
		 * instead would double-count the shared expert and the base. */
		status = SparkQwen38ModuleTpAllReduceHidden(state,slot,slot->delta_bf16,rows);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchLinear(stream,&weights->shared_gate,slot->normalized_bf16,slot->shared_gate_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchLinear(stream,&weights->shared_up,slot->normalized_bf16,slot->shared_up_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchSwiGlu(stream,slot->shared_gate_bf16,slot->shared_up_bf16,rows,SPARK_QWEN38_MODEL_EXPERT_INTERMEDIATE_DIMENSION);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchLinear(stream,&weights->shared_down,slot->shared_up_bf16,slot->shared_down_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchSharedGate(stream,slot->shared_down_bf16,weights->shared_gate_weight_bf16,slot->normalized_bf16,rows,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchResidualAdd(stream,slot->delta_bf16,slot->shared_down_bf16,rows,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchResidualAdd(stream,slot->hidden_bf16,slot->delta_bf16,rows,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION);
	return(SparkStageModuleCudaStatus(SPARK_QWEN38_MODULE_TAG,error,"moe"));
}

static SparkStatus SparkQwen38ModuleRunLayer(SparkQwen38ModuleState *state, SparkQwen38ModuleSlot *slot, const SparkQwen38KvBlockTableView *table, uint32_t layer, uint32_t rows)
{
	SparkQwen38AttnRowsView rows_view;
	SparkStatus status;
	cudaError_t error = SparkQwen38LaunchRmsNorm((cudaStream_t)slot->cuda_stream,slot->hidden_bf16,state->attention_norm_by_layer[layer],slot->normalized_bf16,rows,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION,SPARK_QWEN38_MODEL_RMS_NORM_EPSILON);
	rows_view.slot_mapping = slot->slot_mapping;
	rows_view.row_positions = slot->row_positions;
	rows_view.row_lane_indices = slot->row_lane_indices;
	rows_view.context_lengths = slot->context_lengths;
	status = SparkStageModuleCudaStatus(SPARK_QWEN38_MODULE_TAG,error,"attention_norm");
	if ( status == SPARK_STATUS_OK && state->debug_skip_gdn == 0u )
		status = SPARK_QWEN38_MODEL_LAYER_IS_GDN(layer) != 0u ? SparkQwen38ModuleRunGdnLayer(state,slot,layer,rows) : SparkQwen38ModuleRunAttnLayer(state,slot,table,&state->attn_by_layer[layer],state->attn_ordinal_by_layer[layer],&rows_view,rows);
	if ( status == SPARK_STATUS_OK && state->debug_skip_moe == 0u )
		status = SparkQwen38ModuleRunMoe(state,slot,state->mlp_norm_by_layer[layer],&state->moe_by_layer[layer],rows);
	return(status);
}

/* ---------------------------------------------------------------------------
 * Pools, slot host mirrors, and the lean decode Execute path.
 * Decode-only for now: prefill, speculation and the KV tier land with the
 * serving adapter. The attention block table is a one-block-per-lane view
 * (valid for contexts up to KV_BLOCK_TOKENS); the adapter replaces it.
 * -------------------------------------------------------------------------*/

#define SPARK_QWEN38_MODULE_HOST_ROW_CAPACITY \
	(SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT + SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_MTP_DRAFT_TOKENS)

static SparkStatus SparkQwen38ModuleAllocatePools(SparkQwen38ModuleState *state)
{
	SparkStatus status = SPARK_STATUS_OK;
	uint64_t state_elements,tail_elements,cache_elements;
	state->gdn_pool.abi_version = SPARK_QWEN38_RESIDENT_DECODE_STAGE_GDN_STATE_POOL_ABI_VERSION;
	state->gdn_pool.lane_capacity = state->max_active_sequence_count;
	state->gdn_pool.gdn_layer_count = state->gdn_layer_count;
	state->gdn_pool.state_layer_stride_elements = (uint64_t)SPARK_QWEN38_MODEL_GDN_VALUE_HEAD_COUNT * SPARK_QWEN38_MODEL_GDN_HEAD_KEY_DIMENSION * SPARK_QWEN38_MODEL_GDN_HEAD_VALUE_DIMENSION;
	state->gdn_pool.state_lane_stride_elements = state->gdn_pool.state_layer_stride_elements * state->gdn_layer_count;
	state->gdn_pool.conv_tail_layer_stride_elements = (uint64_t)SPARK_QWEN38_MODEL_GDN_CONV_CHANNELS * (SPARK_QWEN38_MODEL_GDN_CONV_KERNEL - 1u);
	state->gdn_pool.conv_tail_lane_stride_elements = state->gdn_pool.conv_tail_layer_stride_elements * state->gdn_layer_count;
	if ( state->gdn_layer_count != 0u )
	{
		state_elements = state->gdn_pool.state_lane_stride_elements * state->max_active_sequence_count;
		tail_elements = state->gdn_pool.conv_tail_lane_stride_elements * state->max_active_sequence_count;
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,state_elements * sizeof(float),(void **)&state->gdn_pool.state_f32);
		if ( status == SPARK_STATUS_OK )
			status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,tail_elements * SPARK_QWEN38_MODEL_BF16_ELEMENT_BYTES,&state->gdn_pool.conv_tail_bf16);
	}
	state->mtp_cache_ordinal = state->attn_layer_count;
	state->cache_layer_count = state->attn_layer_count;
	if ( status == SPARK_STATUS_OK && state->cache_layer_count != 0u )
	{
		state->cache_layer_stride = (uint64_t)SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS * SPARK_QWEN38_MODEL_ATTN_CACHE_TOKEN_ELEMENTS;
		state->cache_block_stride = state->cache_layer_stride * state->cache_layer_count;
		cache_elements = state->cache_block_stride * state->kv_block_count;
		status = SparkStageModuleDeviceAllocateZeroed(&state->ledger,cache_elements * SPARK_QWEN38_MODEL_BF16_ELEMENT_BYTES,&state->kv_cache_bf16);
	}
	return(status);
}

static SparkStatus SparkQwen38ModuleAllocateSlotHostMirrors(SparkQwen38ModuleState *state, SparkQwen38ModuleSlot *slot)
{
	uint64_t rows = state->max_active_sequence_count;
	SparkStatus status = SPARK_STATUS_OK;
	slot->host_row_lane_indices = (uint32_t *)malloc(SPARK_QWEN38_MODULE_HOST_ROW_CAPACITY * sizeof(uint32_t));
	slot->host_row_positions = (uint64_t *)malloc(SPARK_QWEN38_MODULE_HOST_ROW_CAPACITY * sizeof(uint64_t));
	slot->host_row_cold = (uint32_t *)malloc(rows * sizeof(uint32_t));
	slot->host_slot_mapping = (uint32_t *)malloc(SPARK_QWEN38_MODULE_HOST_ROW_CAPACITY * sizeof(uint32_t));
	slot->host_context_lengths = (uint32_t *)malloc(SPARK_QWEN38_MODULE_HOST_ROW_CAPACITY * sizeof(uint32_t));
	if ( slot->host_row_lane_indices == 0 || slot->host_row_positions == 0 ||
		slot->host_row_cold == 0 || slot->host_slot_mapping == 0 ||
		slot->host_context_lengths == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(status);
}

static SparkStatus SparkQwen38ModuleUploadRows(SparkQwen38ModuleState *state, SparkQwen38ModuleSlot *slot, const SparkModelDriverFrame *frame, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	uint32_t token_guard;
	error = cudaMemcpyAsync(slot->row_lane_indices,slot->host_row_lane_indices,rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->row_positions,slot->host_row_positions,rows * sizeof(uint64_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(slot->row_cold,slot->host_row_cold,rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess && state->attn_layer_count != 0u )
		error = cudaMemcpyAsync(slot->slot_mapping,slot->host_slot_mapping,rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess && state->attn_layer_count != 0u )
		error = cudaMemcpyAsync(slot->context_lengths,slot->host_context_lengths,rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	if ( error == cudaSuccess && state->owns_embedding != 0u )
	{
		if ( frame->buffer_count < 1u || frame->buffers == 0 || frame->buffers[0].address == 0 )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		for (token_guard = 0; token_guard < rows; token_guard++)
			if ( ((const uint32_t *)frame->buffers[0].address)[token_guard] >= SPARK_QWEN38_MODEL_OUTPUT_VOCAB_COUNT )
			{
				fprintf(stderr,"%s token_id_out_of_range row=%u\n",SPARK_QWEN38_MODULE_TAG,token_guard);
				return(SPARK_STATUS_INVALID_ARGUMENT);
			}
		error = cudaMemcpyAsync(slot->input_token_ids,frame->buffers[0].address,rows * sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	}
	return(SparkStageModuleCudaStatus(SPARK_QWEN38_MODULE_TAG,error,"stage_upload"));
}

static cudaError_t SparkQwen38ModuleEmitHead(SparkQwen38ModuleState *state, SparkQwen38ModuleSlot *slot, SparkModelDriverFrame *frame, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	uint32_t out_index = state->owns_embedding != 0u ? 1u : 0u;
	cudaError_t error;
	if ( frame->buffer_count <= out_index || frame->buffers == 0 || frame->buffers[out_index].address == 0 )
		return(cudaErrorInvalidValue);
	error = SparkQwen38LaunchRmsNorm(stream,slot->hidden_bf16,state->final_norm_weight_bf16,slot->normalized_bf16,rows,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION,SPARK_QWEN38_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess && state->head_shadow_payload != 0 )
		error = SparkQwen38LaunchHeadScreenedArgmax(stream,slot->normalized_bf16,state->lm_head_weight_bf16,state->head_shadow_payload,state->head_shadow_scale,state->head_error_norm_f32,slot->head_logits_bf16,slot->head_candidate_ids,slot->head_candidate_counts,slot->output_token_ids,rows,SPARK_QWEN38_MODEL_OUTPUT_VOCAB_COUNT);
	else if ( error == cudaSuccess )
		error = SparkQwen38LaunchHeadArgmax(stream,slot->normalized_bf16,state->lm_head_weight_bf16,slot->input_token_ids,slot->output_token_ids,rows,SPARK_QWEN38_MODEL_OUTPUT_VOCAB_COUNT);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(frame->buffers[out_index].address,slot->output_token_ids,rows * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream);
	return(error);
}

/* The firmware's transport contract (firmware header, frame context): a
 * stage with stage_index > 0 requires HIDDEN_INPUT_TRANSPORT on every
 * frame and a stage with stage_index + 1 < stage_count requires
 * HIDDEN_OUTPUT_TRANSPORT; the module refuses a frame whose transport
 * flags disagree with its position, in either direction. The one escape is
 * the unqualified smoke path (context absent AND the env gate set), which
 * runs the stage on its own buffers. */
static SparkStatus SparkQwen38ModuleValidateFrameContext(SparkQwen38ModuleState *state, const SparkQwen38ResidentDecodeStageFrameContext *context)
{
	uint32_t wants_input,wants_output,has_input,has_output;
	wants_input = state->stage_index != 0u ? 1u : 0u;
	wants_output = state->stage_index + 1u < state->stage_count ? 1u : 0u;
	if ( context == 0 )
		return((wants_input == 0u && wants_output == 0u) || state->allow_unqualified_execution != 0u
			? SPARK_STATUS_OK
			: SPARK_STATUS_INVALID_ARGUMENT);
	if ( context->abi_version != SPARK_QWEN38_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION ||
		context->descriptor_bytes != sizeof(*context) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	has_input = (context->flags & SPARK_QWEN38_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT) != 0u ? 1u : 0u;
	has_output = (context->flags & SPARK_QWEN38_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT) != 0u ? 1u : 0u;
	if ( has_input != wants_input || has_output != wants_output )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( has_input != 0u && (context->hidden_input_transport_session == 0 || context->hidden_input_post_receive_function == 0) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( has_output != 0u && (context->hidden_output_transport_session == 0 || context->hidden_output_send_function == 0) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (context->flags & SPARK_QWEN38_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW) != 0u )
		return(SPARK_STATUS_UNSUPPORTED);
	return(SPARK_STATUS_OK);
}

/* Land the previous stage's hidden residual into the slot's hidden buffer. */
static SparkStatus SparkQwen38ModuleConsumeHiddenInput(SparkQwen38ModuleSlot *slot, SparkQwen38ResidentDecodeStageFrameContext *context, uint32_t rows)
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
		packet->hidden_dimension != SPARK_QWEN38_MODEL_HIDDEN_DIMENSION ||
		packet->bytes_per_sequence < SPARK_QWEN38_MODEL_HIDDEN_BF16_BYTES )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	error = cudaMemcpyAsync(slot->hidden_bf16,packet->hidden_bf16,(uint64_t)rows * SPARK_QWEN38_MODEL_HIDDEN_BF16_BYTES,cudaMemcpyDeviceToDevice,stream);
	return(SparkStageModuleCudaStatus(SPARK_QWEN38_MODULE_TAG,error,"hidden_input"));
}

/* Hand the slot's final hidden residual to the next stage. */
static SparkStatus SparkQwen38ModuleEmitHiddenOutput(SparkQwen38ModuleSlot *slot, SparkQwen38ResidentDecodeStageFrameContext *context, uint32_t rows)
{
	SparkHiddenTransportPacket *packet = &context->hidden_output_packet;
	memset(packet,0,sizeof(*packet));
	packet->abi_version = SPARK_HIDDEN_TRANSPORT_ABI_VERSION;
	packet->descriptor_bytes = SPARK_HIDDEN_TRANSPORT_PACKET_BYTES;
	packet->flags = SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_BF16 | SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_DEVICE_POINTER;
	packet->active_sequence_count = rows;
	packet->hidden_dimension = SPARK_QWEN38_MODEL_HIDDEN_DIMENSION;
	packet->bytes_per_sequence = SPARK_QWEN38_MODEL_HIDDEN_BF16_BYTES;
	packet->hidden_bf16 = slot->hidden_bf16;
	packet->cuda_stream = slot->cuda_stream;
	return(context->hidden_output_send_function(context->hidden_output_transport_session,packet));
}

static SparkStatus SparkQwen38ModuleRunDecode(SparkQwen38ModuleState *state, SparkQwen38ModuleSlot *slot, SparkModelDriverFrame *frame, SparkQwen38ResidentDecodeStageFrameContext *context, uint32_t rows)
{
	SparkQwen38KvBlockTableView table;
	uint32_t block_indices[SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t block_counts[SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t layer,row;
	uint32_t wants_input,wants_output;
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	SparkStatus status;
	cudaError_t error;
	/* The KV table comes from the frame context when the serving adapter
	 * provides one; the synthesized identity table is the smoke path. */
	if ( context != 0 && context->kv_block_table != 0 )
		table = *context->kv_block_table;
	else
	{
		memset(&table,0,sizeof(table));
		for (row = 0; row < rows; row++)
		{
			block_indices[row] = row;
			block_counts[row] = 1u;
		}
		table.abi_version = SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TABLE_ABI_VERSION;
		table.descriptor_bytes = sizeof(table);
		table.block_token_count = SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
		table.lane_count = state->max_active_sequence_count;
		table.lane_stride = 1u;
		table.lane_capacity = state->max_active_sequence_count;
		table.physical_block_indices = block_indices;
		table.lane_physical_block_counts = block_counts;
		table.host_physical_block_indices = block_indices;
		table.host_lane_physical_block_counts = block_counts;
	}
	status = SparkQwen38ModuleUploadRows(state,slot,frame,rows);
	if ( status == SPARK_STATUS_OK && state->tp_degree > 1u )
	{
		/* Zero the grouped expert output buffer so the rank-local pair
		 * reduce over ALL pairs sums only this rank's experts (the peers'
		 * rows live in their own buffers; the all-reduce completes the
		 * mixture). */
		error = cudaMemsetAsync(slot->moe_slot_out_bf16,0,(uint64_t)rows * SPARK_QWEN38_MODEL_EXPERTS_PER_TOKEN * SPARK_QWEN38_MODEL_HIDDEN_BF16_BYTES,stream);
		status = SparkStageModuleCudaStatus(SPARK_QWEN38_MODULE_TAG,error,"tp_slot_zero");
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkQwen38ModuleKvPrepareFrame(state,slot,context,&table,rows);
	if ( status != SPARK_STATUS_OK )
		return(status);
	wants_input = context != 0 && (context->flags & SPARK_QWEN38_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT) != 0u ? 1u : 0u;
	if ( state->owns_embedding != 0u )
	{
		error = SparkQwen38LaunchEmbeddingGather(stream,slot->input_token_ids,state->token_embedding_bf16,slot->hidden_bf16,rows);
		status = SparkStageModuleCudaStatus(SPARK_QWEN38_MODULE_TAG,error,"embedding");
	}
	else if ( wants_input != 0u )
		status = SparkQwen38ModuleConsumeHiddenInput(slot,context,rows);
	if ( status != SPARK_STATUS_OK )
		return(status);
	for (layer = state->first_layer_index; status == SPARK_STATUS_OK && layer < state->first_layer_index + state->layer_count; layer++)
		status = SparkQwen38ModuleRunLayer(state,slot,&table,layer,rows);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( state->owns_final_head != 0u )
		status = SparkStageModuleCudaStatus(SPARK_QWEN38_MODULE_TAG,SparkQwen38ModuleEmitHead(state,slot,frame,rows),"head_emit");
	wants_output = context != 0 && (context->flags & SPARK_QWEN38_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT) != 0u ? 1u : 0u;
	if ( status == SPARK_STATUS_OK && wants_output != 0u )
		status = SparkQwen38ModuleEmitHiddenOutput(slot,context,rows);
	error = cudaStreamSynchronize(stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_QWEN38_MODULE_TAG,error,"stream_sync");
	if ( status == SPARK_STATUS_OK )
		SparkQwen38ModuleKvMarkWritten(state,slot,rows);
	return(status);
}

SparkStatus SparkQwen38ResidentDecodeStageExecute(
    void *module_state,
    SparkModelDriverFrame *frame)
{
	SparkQwen38ModuleState *state = (SparkQwen38ModuleState *)module_state;
	SparkQwen38ResidentDecodeStageFrameContext *context;
	SparkQwen38ModuleSlot *slot;
	uint32_t rows,row;
	SparkStatus status;
	if ( state == 0 || frame == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u )
		return(SPARK_STATUS_UNSUPPORTED);
	context = (SparkQwen38ResidentDecodeStageFrameContext *)frame->user_context;
	status = SparkQwen38ModuleValidateFrameContext(state,context);
	if ( status != SPARK_STATUS_OK )
		return(status);
	rows = frame->active_slot_count;
	if ( rows == 0u || rows > state->max_active_sequence_count )
	{
		atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	slot = &state->slots[0];
	if ( slot->cuda_stream == 0 )
		return(SPARK_STATUS_INTERNAL_ERROR);
	for (row = 0; row < rows; row++)
	{
		slot->host_row_lane_indices[row] = row;
		slot->host_row_positions[row] = frame->sequence_position;
		slot->host_row_cold[row] = 0u;
		slot->host_slot_mapping[row] = (frame->sequence_position + row) % SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
		slot->host_context_lengths[row] = (uint32_t)((frame->sequence_position + row) % SPARK_QWEN38_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS) + 1u;
	}
	atomic_fetch_add_explicit(&state->submitted_count,1u,memory_order_relaxed);
	status = SparkQwen38ModuleRunDecode(state,slot,frame,context,rows);
	if ( status == SPARK_STATUS_OK )
	{
		atomic_fetch_add_explicit(&state->completed_count,1u,memory_order_relaxed);
		atomic_fetch_add_explicit(&state->tokens_emitted,rows,memory_order_relaxed);
	}
	else
		atomic_fetch_add_explicit(&state->failed_count,1u,memory_order_relaxed);
	return(status);
}
