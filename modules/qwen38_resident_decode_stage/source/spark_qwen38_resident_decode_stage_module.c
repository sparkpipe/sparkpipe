/* Qwen 3.8 Max resident decode stage - loader skeleton.
 *
 * This revision implements the complete stage-pack load and validation path
 * (geometry compare, per-entry shape/format checks, binding, coverage
 * verification) and the module entry points. Layer execution, pools and the
 * CUDA kernels land in the following revisions; Execute/Admit/Snapshot fail
 * closed with SPARK_STATUS_UNSUPPORTED until then, so this skeleton can
 * never serve a token.
 */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_hidden_transport.h"
#include "sparkpipe/spark_qwen38_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "spark_qwen38_stagepack_format.h"

#define SPARK_QWEN38_MODULE_TAG "qwen38_stage"

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
} SparkQwen38ModuleSlot;

typedef struct SparkQwen38ModuleState
{
	SparkStageModuleLedger ledger;
	uint32_t multiprocessor_count;
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
	const void *attention_norm_by_layer[SPARK_QWEN38_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	const void *mlp_norm_by_layer[SPARK_QWEN38_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	SparkQwen38GdnLayerWeights gdn_by_layer[SPARK_QWEN38_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	SparkQwen38AttnLayerWeights attn_by_layer[SPARK_QWEN38_RESIDENT_DECODE_STAGE_LAYER_COUNT];
	SparkQwen38MoeWeights moe_by_layer[SPARK_QWEN38_RESIDENT_DECODE_STAGE_LAYER_COUNT];
} SparkQwen38ModuleState;

static SparkStatus SparkQwen38ModuleConfigure(SparkQwen38ModuleState *state)
{
	SparkStatus status;
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
	if ( entry->weight_format == SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 ? entry->scale_group_size != SPARK_QWEN38_MODEL_MXFP4_GROUP_SIZE : (entry->weight_format == SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_FP8_E4M3_F32B128 ? entry->scale_group_size != 128u : entry->scale_group_size != 0u) )
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
	state->ledger.module_tag = SPARK_QWEN38_MODULE_TAG;
	atomic_init(&state->submitted_count,0u);
	atomic_init(&state->completed_count,0u);
	atomic_init(&state->rejected_count,0u);
	atomic_init(&state->failed_count,0u);
	atomic_init(&state->tokens_emitted,0u);
	status = SparkQwen38ModuleConfigure(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentText(SPARK_QWEN38_MODULE_TAG,"SPARK_QWEN38_STAGE_PACK_PATH",&pack_path);
	if ( status == SPARK_STATUS_OK )
	{
		SparkQwen38ModuleBuildOrdinals(state);
		status = SparkQwen38ModuleLoadPack(state,pack_path);
	}
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
extern cudaError_t SparkQwen38LaunchAttnPrepare(cudaStream_t stream, void *q_fused_bf16, const void *k_bf16, const void *v_bf16, const SparkQwen38AttnLayerWeights *weights, void *kv_cache_bf16, const uint32_t *slot_mapping, const uint64_t *row_positions, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride, float epsilon);
extern cudaError_t SparkQwen38LaunchAttnDecode(cudaStream_t stream, const void *q_fused_bf16, const void *kv_cache_bf16, const SparkQwen38KvBlockTableView *table, const uint32_t *row_lane_indices, const uint32_t *context_lengths, void *head_out_bf16, uint32_t row_count, uint32_t attn_layer_ordinal, uint64_t cache_layer_stride, uint64_t cache_block_stride);
extern cudaError_t SparkQwen38LaunchResidualAdd(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, uint32_t row_count, uint32_t dimension);
extern cudaError_t SparkQwen38LaunchHeadArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint32_t *token_ids, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count);
extern cudaError_t SparkQwen38LaunchGateScores(cudaStream_t stream, const SparkQwen38LinearView *gate, const void *input_bf16, float *scores_f32, uint32_t row_count);
extern cudaError_t SparkQwen38LaunchGateSelect(cudaStream_t stream, const float *scores_f32, const float *bias_f32, uint32_t row_count, uint32_t expert_count, uint32_t topk, float route_scale, uint32_t *indices_u32, float *weights_f32);
extern cudaError_t SparkQwen38LaunchMoeRoute(cudaStream_t stream, const uint32_t *route_expert, uint32_t rows, uint32_t expert_width, uint32_t *group_row_offset, uint32_t *route_packed_row, uint32_t *route_source_token, uint32_t *group_tile_prefix_w1, uint32_t *group_tile_prefix_w2);
extern cudaError_t SparkQwen38LaunchFusedExpertW13Act(cudaStream_t stream, const SparkQwen38LinearView *w1, const SparkQwen38LinearView *w3, const void *input_bf16, const uint32_t *route_source_token, const uint32_t *group_row_offset, uint32_t *group_tile_prefix, void *activated_bf16, uint32_t rows, uint32_t expert_width, float limit, uint32_t multiprocessor_count);
extern cudaError_t SparkQwen38LaunchExpertDown(cudaStream_t stream, const SparkQwen38LinearView *stacked, const void *input_bf16, const uint32_t *group_row_offset, uint32_t *group_tile_prefix, void *output_bf16, uint32_t rows, uint32_t expert_width, uint32_t hidden_dimension, uint32_t multiprocessor_count);
extern cudaError_t SparkQwen38LaunchMoePairReduce(cudaStream_t stream, const void *slot_out_bf16, const uint32_t *inverse_map, const float *pair_weights_f32, void *accum_bf16, uint32_t row_count, uint32_t hidden_dimension);
extern cudaError_t SparkQwen38LaunchGroupedExpertLinear(cudaStream_t stream, const SparkQwen38LinearView *view, const void *input_bf16, const uint32_t *source_row_map, const uint32_t *group_row_offset, const uint32_t *group_tile_prefix, void *output_bf16, uint32_t source_row_count, uint32_t multiprocessor_count);
extern cudaError_t SparkQwen38LaunchSwiGlu(cudaStream_t stream, const void *gate_bf16, void *up_bf16, uint32_t row_count, uint32_t dimension);
extern cudaError_t SparkQwen38LaunchSharedGate(cudaStream_t stream, void *accum_bf16, const void *gate_weight_bf16, uint32_t row_count, uint32_t dimension);

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
		error = SparkQwen38LaunchAttnPrepare(stream,slot->q_fused_bf16,slot->k_bf16,slot->v_bf16,weights,state->kv_cache_bf16,rows_view->slot_mapping,rows_view->row_positions,rows,ordinal,state->cache_layer_stride,state->cache_block_stride,SPARK_QWEN38_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchAttnDecode(stream,slot->q_fused_bf16,state->kv_cache_bf16,table,rows_view->row_lane_indices,rows_view->context_lengths,slot->head_out_bf16,rows,ordinal,state->cache_layer_stride,state->cache_block_stride);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchLinear(stream,&weights->output,slot->head_out_bf16,slot->delta_bf16,rows);
	return(SparkStageModuleCudaStatus(SPARK_QWEN38_MODULE_TAG,error,"attn_layer"));
}

/*
 * Routed MoE + shared expert, the layer's FFN side. delta_bf16 accumulates
 * the routed expert output (weighted pair reduce) and then the shared expert
 * output (gate-multiplied), and the fused residual norm folds the previous
 * residual plus delta into the next normalized input.
 */
static SparkStatus SparkQwen38ModuleRunMoe(SparkQwen38ModuleState *state, SparkQwen38ModuleSlot *slot, const void *mlp_norm_bf16, const SparkQwen38MoeWeights *weights, uint32_t rows)
{
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	error = SparkQwen38LaunchFusedResidualRmsNorm(stream,slot->hidden_bf16,slot->delta_bf16,mlp_norm_bf16,slot->normalized_bf16,rows,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION,SPARK_QWEN38_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchGateScores(stream,&weights->gate,slot->normalized_bf16,slot->moe_scores_f32,rows);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchGateSelect(stream,slot->moe_scores_f32,0,rows,SPARK_QWEN38_MODEL_ROUTED_EXPERT_COUNT,SPARK_QWEN38_MODEL_EXPERTS_PER_TOKEN,1.0f,slot->moe_indices_u32,slot->moe_weights_f32);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchMoeRoute(stream,slot->moe_indices_u32,rows,SPARK_QWEN38_MODEL_EXPERT_INTERMEDIATE_DIMENSION,slot->moe_group_offset_u32,slot->moe_inverse_u32,slot->moe_grouped_rows_u32,slot->moe_tile_prefix_w1_u32,slot->moe_tile_prefix_w2_u32);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchGroupedExpertLinear(stream,&weights->experts_w1,slot->normalized_bf16,slot->moe_grouped_rows_u32,slot->moe_group_offset_u32,slot->moe_tile_prefix_w1_u32,slot->moe_gate_packed_bf16,rows,state->multiprocessor_count);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchGroupedExpertLinear(stream,&weights->experts_w3,slot->normalized_bf16,slot->moe_grouped_rows_u32,slot->moe_group_offset_u32,slot->moe_tile_prefix_w1_u32,slot->moe_slot_up_bf16,rows,state->multiprocessor_count);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchSwiGlu(stream,slot->moe_gate_packed_bf16,slot->moe_slot_up_bf16,rows * SPARK_QWEN38_MODEL_EXPERTS_PER_TOKEN,SPARK_QWEN38_MODEL_EXPERT_INTERMEDIATE_DIMENSION);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchGroupedExpertLinear(stream,&weights->experts_w2,slot->moe_slot_up_bf16,0,slot->moe_group_offset_u32,slot->moe_tile_prefix_w2_u32,slot->moe_slot_out_bf16,rows * SPARK_QWEN38_MODEL_EXPERTS_PER_TOKEN,state->multiprocessor_count);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchMoePairReduce(stream,slot->moe_slot_out_bf16,slot->moe_inverse_u32,slot->moe_weights_f32,slot->delta_bf16,rows,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchLinear(stream,&weights->shared_gate,slot->normalized_bf16,slot->shared_gate_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchLinear(stream,&weights->shared_up,slot->normalized_bf16,slot->shared_up_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchSwiGlu(stream,slot->shared_gate_bf16,slot->shared_up_bf16,rows,SPARK_QWEN38_MODEL_EXPERT_INTERMEDIATE_DIMENSION);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchLinear(stream,&weights->shared_down,slot->shared_up_bf16,slot->shared_down_bf16,rows);
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchSharedGate(stream,slot->shared_down_bf16,weights->shared_gate_weight_bf16,rows,SPARK_QWEN38_MODEL_HIDDEN_DIMENSION);
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
	if ( status == SPARK_STATUS_OK )
		status = SPARK_QWEN38_MODEL_LAYER_IS_GDN(layer) != 0u ? SparkQwen38ModuleRunGdnLayer(state,slot,layer,rows) : SparkQwen38ModuleRunAttnLayer(state,slot,table,&state->attn_by_layer[layer],state->attn_ordinal_by_layer[layer],&rows_view,rows);
	if ( status == SPARK_STATUS_OK )
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
	if ( error == cudaSuccess )
		error = SparkQwen38LaunchHeadArgmax(stream,slot->normalized_bf16,state->lm_head_weight_bf16,slot->input_token_ids,slot->output_token_ids,rows,SPARK_QWEN38_MODEL_OUTPUT_VOCAB_COUNT);
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(frame->buffers[out_index].address,slot->output_token_ids,rows * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream);
	return(error);
}

static SparkStatus SparkQwen38ModuleRunDecode(SparkQwen38ModuleState *state, SparkQwen38ModuleSlot *slot, SparkModelDriverFrame *frame, uint32_t rows)
{
	SparkQwen38KvBlockTableView table;
	uint32_t block_indices[SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t block_counts[SPARK_QWEN38_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t layer,row;
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	SparkStatus status;
	cudaError_t error;
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
	status = SparkQwen38ModuleUploadRows(state,slot,frame,rows);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( state->owns_embedding != 0u )
	{
		error = SparkQwen38LaunchEmbeddingGather(stream,slot->input_token_ids,state->token_embedding_bf16,slot->hidden_bf16,rows);
		status = SparkStageModuleCudaStatus(SPARK_QWEN38_MODULE_TAG,error,"embedding");
	}
	if ( status != SPARK_STATUS_OK )
		return(status);
	for (layer = state->first_layer_index; status == SPARK_STATUS_OK && layer < state->first_layer_index + state->layer_count; layer++)
		status = SparkQwen38ModuleRunLayer(state,slot,&table,layer,rows);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( state->owns_final_head != 0u )
		status = SparkStageModuleCudaStatus(SPARK_QWEN38_MODULE_TAG,SparkQwen38ModuleEmitHead(state,slot,frame,rows),"head_emit");
	error = cudaStreamSynchronize(stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_QWEN38_MODULE_TAG,error,"stream_sync");
	return(status);
}

SparkStatus SparkQwen38ResidentDecodeStageExecute(
    void *module_state,
    SparkModelDriverFrame *frame)
{
	SparkQwen38ModuleState *state = (SparkQwen38ModuleState *)module_state;
	SparkQwen38ModuleSlot *slot;
	uint32_t rows,row;
	SparkStatus status;
	if ( state == 0 || frame == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u )
		return(SPARK_STATUS_UNSUPPORTED);
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
	status = SparkQwen38ModuleRunDecode(state,slot,frame,rows);
	if ( status == SPARK_STATUS_OK )
	{
		atomic_fetch_add_explicit(&state->completed_count,1u,memory_order_relaxed);
		atomic_fetch_add_explicit(&state->tokens_emitted,rows,memory_order_relaxed);
	}
	else
		atomic_fetch_add_explicit(&state->failed_count,1u,memory_order_relaxed);
	return(status);
}
