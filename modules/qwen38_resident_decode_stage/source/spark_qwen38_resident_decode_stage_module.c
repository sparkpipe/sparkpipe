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

typedef struct SparkQwen38MtpWeights
{
	SparkQwen38AttentionLayerWeights attention;
	SparkQwen38MoeWeights moe;
	const void *attention_norm_weight_bf16;
	const void *mlp_norm_weight_bf16;
	SparkQwen38LinearView fc;
	const void *embed_norm_weight_bf16;
	const void *hidden_norm_weight_bf16;
	const void *final_norm_weight_bf16;
} SparkQwen38MtpWeights;

typedef struct SparkQwen38ModuleState
{
	SparkStageModuleLedger ledger;
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
	SparkQwen38AttentionLayerWeights attn_by_layer[SPARK_QWEN38_RESIDENT_DECODE_STAGE_LAYER_COUNT];
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
		if ( shape.natural_format != SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 || entry->weight_format != SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 )
			return(SPARK_STATUS_VALIDATION_FAILED);
	}
	if ( entry->weight_format == SPARK_QWEN38_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_MXFP4_E2M1 ? entry->scale_group_size != SPARK_QWEN38_MODEL_MXFP4_GROUP_SIZE : entry->scale_group_size != 0u )
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
	case SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_QUERY_NORM: state->mtp.attention.query_norm_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_KEY_NORM: state->mtp.attention.key_norm_bf16 = payload; return(SPARK_STATUS_OK);
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
	case SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_QUERY_NORM: state->attn_by_layer[layer].query_norm_bf16 = payload; return(SPARK_STATUS_OK);
	case SPARK_QWEN38_STAGEPACK_TENSOR_ATTN_KEY_NORM: state->attn_by_layer[layer].key_norm_bf16 = payload; return(SPARK_STATUS_OK);
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
	status = SparkQwen38ModuleConfigure(state);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentText(SPARK_QWEN38_MODULE_TAG,"SPARK_QWEN38_STAGE_PACK_PATH",&pack_path);
	if ( status == SPARK_STATUS_OK )
	{
		SparkQwen38ModuleBuildOrdinals(state);
		status = SparkQwen38ModuleLoadPack(state,pack_path);
	}
	if ( status != SPARK_STATUS_OK )
	{
		SparkQwen38ResidentDecodeStageDestroy(state);
		return(status);
	}
	*module_state = state;
	fprintf(stderr,"%s initialize ok slice=%u+%u gdn=%u attn=%u owns_embedding=%u owns_head=%u\n",SPARK_QWEN38_MODULE_TAG,state->first_layer_index,state->layer_count,state->gdn_layer_count,state->attn_layer_count,state->owns_embedding,state->owns_final_head);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkQwen38ResidentDecodeStageExecute(
    void *module_state,
    SparkModelDriverFrame *frame)
{
	(void)module_state;
	(void)frame;
	/* Loader skeleton: no compute path yet. Fails closed. */
	return(SPARK_STATUS_UNSUPPORTED);
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
