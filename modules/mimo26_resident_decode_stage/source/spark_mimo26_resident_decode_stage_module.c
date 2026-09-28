#define _FILE_OFFSET_BITS 64

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_admission.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "sparkpipe/spark_stage_module_lifecycle.h"
#include "sparkpipe/spark_mimo26_resident_decode_stage_firmware.h"
#include "spark_mimo26_rank_engine.h"

#define SPARK_FAMILY_CAMEL Mimo26
#define SPARK_FAMILY_UPPER MIMO26
#define SPARK_FAMILY_LOWER mimo26
#include "sparkpipe/family/spark_family.h"

#define SPARK_MIMO26_MODULE_TAG "mimo26_stage"
#define SPARK_MIMO26_MODULE_WAIT_NS UINT64_C(120000000000)
#define SPARK_MIMO26_MODULE_TP_DEGREE 4u

typedef struct SparkMimo26ModuleState
{
	SparkStageModuleLedger ledger;
	uint32_t pipeline_slot_count;
	uint32_t max_active_sequence_count;
	uint32_t max_positions;
	uint32_t kv_block_count;
	uint32_t tp_rank;
	uint32_t graph;
	uint64_t reset_generation;
	atomic_uint slot_states[SPARK_MIMO26_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	atomic_ullong submitted_count;
	atomic_ullong completed_count;
	atomic_ullong rejected_count;
	atomic_ullong failed_count;
	atomic_ullong tokens_emitted;
	SparkMimo26RankEngine *engine;
	uint64_t lane_sequence[SPARK_MIMO26_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_next_position[SPARK_MIMO26_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	char pack_sha256[65];
} SparkMimo26ModuleState;

static SparkStatus SparkMimo26ModuleContinue(SparkMimo26ModuleState *state,uint32_t lane,uint64_t sequence,uint64_t position,uint64_t count)
{
	if ( position == 0u )
		state->lane_sequence[lane] = sequence;
	else if ( state->lane_sequence[lane] != sequence || state->lane_next_position[lane] != position )
	{
		fprintf(stderr,"%s continuity lane=%u sequence=%llu position=%llu held_sequence=%llu held_next=%llu: the lane KV does not hold this prefix (prefix reuse needs prefix_reuse false)\n",SPARK_MIMO26_MODULE_TAG,lane,(unsigned long long)sequence,(unsigned long long)position,(unsigned long long)state->lane_sequence[lane],(unsigned long long)state->lane_next_position[lane]);
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}
	state->lane_next_position[lane] = position + count;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkMimo26ModuleReadSha(char *sha256)
{
	const char *digest = 0;
	SparkStatus status = SparkStageModuleEnvironmentText(SPARK_MIMO26_MODULE_TAG,"SPARK_WEIGHTD_PACK_SHA256",&digest);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( strlen(digest) != 64u || strspn(digest,"0123456789abcdef") != 64u )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	memcpy(sha256,digest,65u);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkMimo26ModuleWriteTokens(SparkModelDriverFrame *frame,const uint32_t *tokens,uint32_t count)
{
	uint32_t index;
	for (index = 0u; frame->buffers != 0 && index < frame->buffer_count; index++)
	{
		SparkModelDriverBuffer *buffer = &frame->buffers[index];
		if ( (buffer->flags & SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE) == 0u )
			continue;
		if ( buffer->address == 0 || buffer->bytes < (uint64_t)count * sizeof(uint32_t) )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		memcpy(buffer->address,tokens,(size_t)count * sizeof(uint32_t));
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkMimo26ModuleRunPrefill(SparkMimo26ModuleState *state,SparkModelDriverFrame *frame,const SparkMimo26PrefillFrameView *prefill)
{
	const uint32_t *inputs;
	uint32_t row,next = 0u;
	SparkStatus status = SPARK_STATUS_OK;
	if ( frame->buffers == 0 || frame->buffer_count == 0u || (frame->buffers[0].flags & SPARK_MODEL_DRIVER_BUFFER_FLAG_READ) == 0u || frame->buffers[0].address == 0 || frame->buffers[0].bytes < (uint64_t)prefill->token_count * sizeof(uint32_t) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( prefill->lane_index >= state->max_active_sequence_count || prefill->base_position + prefill->token_count > state->max_positions )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	inputs = (const uint32_t *)frame->buffers[0].address;
	status = SparkMimo26ModuleContinue(state,prefill->lane_index,prefill->sequence_id,prefill->base_position,prefill->token_count);
	for (row = 0u; status == SPARK_STATUS_OK && row < prefill->token_count; row += SPARK_MIMO26_RANK_ENGINE_MAX_ROWS)
	{
		uint32_t lanes[SPARK_MIMO26_RANK_ENGINE_MAX_ROWS],positions[SPARK_MIMO26_RANK_ENGINE_MAX_ROWS],outputs[SPARK_MIMO26_RANK_ENGINE_MAX_ROWS],count,index;
		count = prefill->token_count - row < SPARK_MIMO26_RANK_ENGINE_MAX_ROWS ? prefill->token_count - row : SPARK_MIMO26_RANK_ENGINE_MAX_ROWS;
		for (index = 0u; index < count; index++)
		{
			lanes[index] = prefill->lane_index;
			positions[index] = (uint32_t)(prefill->base_position + row + index);
		}
		status = SparkMimo26RankEngineRows(state->engine,count,lanes,inputs + row,positions,outputs,0);
		next = outputs[count - 1u];
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkMimo26ModuleWriteTokens(frame,&next,1u);
	SPARK_RETURN(status);
}

static SparkStatus SparkMimo26ModuleRunDecode(SparkMimo26ModuleState *state,SparkModelDriverFrame *frame,const SparkMimo26DecodeBatchView *batch)
{
	uint32_t outputs[SPARK_MIMO26_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	const uint32_t *inputs;
	uint32_t row;
	SparkStatus status = SPARK_STATUS_OK;
	if ( frame->buffers == 0 || frame->buffer_count == 0u || (frame->buffers[0].flags & SPARK_MODEL_DRIVER_BUFFER_FLAG_READ) == 0u || frame->buffers[0].address == 0 || frame->buffers[0].bytes < (uint64_t)batch->row_count * sizeof(uint32_t) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	uint32_t positions[SPARK_MIMO26_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	inputs = (const uint32_t *)frame->buffers[0].address;
	for (row = 0u; status == SPARK_STATUS_OK && row < batch->row_count; row++)
	{
		if ( batch->row_lane_indices[row] >= state->max_active_sequence_count || batch->row_positions[row] >= state->max_positions || batch->row_sequence_ids == 0 )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		status = SparkMimo26ModuleContinue(state,batch->row_lane_indices[row],batch->row_sequence_ids[row],batch->row_positions[row],1u);
		positions[row] = (uint32_t)batch->row_positions[row];
	}
	for (row = 0u; status == SPARK_STATUS_OK && row < batch->row_count; row += SPARK_MIMO26_RANK_ENGINE_MAX_ROWS)
	{
		uint32_t count = batch->row_count - row < SPARK_MIMO26_RANK_ENGINE_MAX_ROWS ? batch->row_count - row : SPARK_MIMO26_RANK_ENGINE_MAX_ROWS;
		status = SparkMimo26RankEngineRows(state->engine,count,batch->row_lane_indices + row,inputs + row,positions + row,outputs + row,0);
	}
	if ( status == SPARK_STATUS_OK )
		status = SparkMimo26ModuleWriteTokens(frame,outputs,batch->row_count);
	SPARK_RETURN(status);
}

static SparkStatus SparkMimo26ModuleExecuteFrame(void *module_state,SparkModelDriverFrame *frame)
{
	SparkMimo26ModuleState *state = (SparkMimo26ModuleState *)module_state;
	const SparkMimo26ResidentDecodeStageFrameContext *context = (const SparkMimo26ResidentDecodeStageFrameContext *)frame->user_context;
	SparkStatus status;
	uint32_t tokens;
	if ( context == 0 || context->abi_version != SPARK_MIMO26_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION || context->descriptor_bytes < sizeof(*context) )
	{
		atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	atomic_fetch_add_explicit(&state->submitted_count,1u,memory_order_relaxed);
	if ( (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u )
	{
		if ( context->prefill_frame == 0 || context->prefill_frame->token_count == 0u )
			status = SPARK_STATUS_INVALID_ARGUMENT;
		else
			status = SparkMimo26ModuleRunPrefill(state,frame,context->prefill_frame);
		tokens = context->prefill_frame != 0 ? context->prefill_frame->token_count : 0u;
	}
	else if ( (context->flags & SPARK_MIMO26_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW) == 0u || context->decode_batch == 0 || context->decode_batch->row_count == 0u || context->decode_batch->row_count != frame->active_slot_count || context->decode_batch->row_count > state->max_active_sequence_count )
	{
		status = SPARK_STATUS_INVALID_ARGUMENT;
		tokens = 0u;
	}
	else
	{
		status = SparkMimo26ModuleRunDecode(state,frame,context->decode_batch);
		tokens = context->decode_batch->row_count;
	}
	if ( status == SPARK_STATUS_OK )
	{
		atomic_fetch_add_explicit(&state->completed_count,1u,memory_order_relaxed);
		atomic_fetch_add_explicit(&state->tokens_emitted,tokens,memory_order_relaxed);
	}
	else
		atomic_fetch_add_explicit(&state->failed_count,1u,memory_order_relaxed);
	SPARK_RETURN(status);
}

static SparkStatus SparkMimo26ModuleAdmit(void *module_state,const SparkModelDriverAdmissionRequest *request,SparkModelDriverAdmissionDecision *decision)
{
	SparkMimo26ModuleState *state = (SparkMimo26ModuleState *)module_state;
	SparkAdmissionPolicyTable table;
	SparkStatus status;
	if ( request->admission_flags == SPARK_MODEL_DRIVER_ADMISSION_FLAG_RESET )
	{
		SparkModelDriverInitializeAdmissionDecision(decision);
		if ( SparkModelDriverAdmissionRequestIsValid(request) == 0u )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		if ( request->control_generation > state->reset_generation )
		{
			state->reset_generation = request->control_generation;
			memset(state->lane_sequence,0,sizeof(state->lane_sequence));
			memset(state->lane_next_position,0,sizeof(state->lane_next_position));
		}
		decision->accepted = 1u;
		decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
		return(SPARK_STATUS_OK);
	}
	memset(&table,0,sizeof(table));
	table.abi_version = SPARK_ADMISSION_ABI_VERSION;
	table.descriptor_bytes = (uint32_t)sizeof(table);
	table.max_active_sequence_count = state->max_active_sequence_count;
	table.max_input_row_count = SPARK_MIMO26_RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT;
	table.max_sequence_positions = state->max_positions;
	table.flags = SPARK_ADMISSION_POLICY_FLAG_DECODE_EQUALS_SLOTS | SPARK_ADMISSION_POLICY_FLAG_ALLOW_DISPATCH_FLAG;
	table.cost = SparkStageModuleAdmissionCost;
	table.cost_context = state;
	status = SparkAdmissionEvaluateShape(&table,SparkStageModuleSlotCountFree(state->slot_states,state->pipeline_slot_count),request,decision);
	if ( status == SPARK_STATUS_OK && decision->accepted == 0u )
		atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
	SPARK_RETURN(status);
}

static SparkStatus SparkMimo26ModuleStateTeardown(void *module_state)
{
	SparkMimo26ModuleState *state = (SparkMimo26ModuleState *)module_state;
	SparkStatus status = SparkMimo26RankEngineDestroy(state->engine);
	state->engine = 0;
	SparkStageModuleLedgerRelease(&state->ledger);
	SPARK_RETURN(status);
}

static SparkStatus SparkMimo26ModulePrepare(void *module_state,const SparkFirmwareModuleConfiguration *configuration,const SparkFirmwareModuleHostServices *host_services)
{
	SparkMimo26ModuleState *state = (SparkMimo26ModuleState *)module_state;
	SparkMimo26RankEngineConfig engine;
	const char *pack_path = 0;
	uint64_t expert_pool_bytes = 0u,spine_budget_bytes = 0u;
	SparkStatus status;
	(void)configuration;
	(void)host_services;
	status = SparkStageModuleEnvironmentUnsigned(SPARK_MIMO26_MODULE_TAG,"SPARK_MIMO26_TP_RANK",0u,SPARK_MIMO26_MODULE_TP_DEGREE - 1u,&state->tp_rank);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_MIMO26_MODULE_TAG,"SPARK_MIMO26_STAGE_MAX_ACTIVE_SEQUENCES",1u,SPARK_MIMO26_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,&state->max_active_sequence_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_MIMO26_MODULE_TAG,"SPARK_MIMO26_STAGE_PIPELINE_SLOTS",1u,SPARK_MIMO26_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,&state->pipeline_slot_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_MIMO26_MODULE_TAG,"SPARK_MIMO26_STAGE_MAX_POSITIONS",SPARK_MIMO26_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS,SPARK_MIMO26_RESIDENT_DECODE_STAGE_MAXIMUM_CONTEXT_TOKENS,&state->max_positions);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned(SPARK_MIMO26_MODULE_TAG,"SPARK_MIMO26_STAGE_GRAPH",0u,1u,&state->graph);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned64(SPARK_MIMO26_MODULE_TAG,"SPARK_WEIGHTD_EXPERT_POOL_BYTES",1u,UINT64_MAX,&expert_pool_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentUnsigned64(SPARK_MIMO26_MODULE_TAG,"SPARK_WEIGHTD_SPINE_BUDGET_BYTES",1u,UINT64_MAX,&spine_budget_bytes);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleEnvironmentText(SPARK_MIMO26_MODULE_TAG,"SPARK_MIMO26_STAGE_PACK_PATH",&pack_path);
	if ( status == SPARK_STATUS_OK && (state->max_positions % SPARK_MIMO26_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS) != 0u )
		status = SPARK_STATUS_INVALID_ARGUMENT;
	if ( status == SPARK_STATUS_OK )
		status = SparkMimo26ModuleReadSha(state->pack_sha256);
	if ( status == SPARK_STATUS_OK )
	{
		SparkStageModuleAtomicStateArrayInitialize(state->slot_states,state->pipeline_slot_count);
		state->kv_block_count = state->max_active_sequence_count * (state->max_positions / SPARK_MIMO26_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
		memset(&engine,0,sizeof(engine));
		engine.rank = state->tp_rank;
		engine.lane_count = state->max_active_sequence_count;
		engine.max_positions = state->max_positions;
		engine.mode = state->graph != 0u ? SPARK_MIMO26_RANK_ENGINE_MODE_GRAPH : SPARK_MIMO26_RANK_ENGINE_MODE_EAGER;
		engine.expert_pool_bytes = expert_pool_bytes;
		engine.spine_budget_bytes = spine_budget_bytes;
		engine.wait_ns = SPARK_MIMO26_MODULE_WAIT_NS;
		engine.pack_path = pack_path;
		engine.pack_sha256 = state->pack_sha256;
		status = SparkMimo26RankEngineCreate(&engine,&state->engine);
	}
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"%s initialize_failed status=%d\n",SPARK_MIMO26_MODULE_TAG,(int)status);
	else
		fprintf(stderr,"%s initialize ok tp=%u/%u lanes=%u positions=%u graph=%u\n",SPARK_MIMO26_MODULE_TAG,state->tp_rank,SPARK_MIMO26_MODULE_TP_DEGREE,state->max_active_sequence_count,state->max_positions,state->graph);
	SPARK_RETURN(status);
}

static void SparkMimo26ModuleReportReady(void *module_state)
{
	(void)module_state;
}

#include "sparkpipe/family/module/spark_module_snapshot_extend.h"

#include "sparkpipe/family/module/spark_module_describe.h"

#include "sparkpipe/family/module/spark_module_initialize_gate.h"

static const SparkStageModuleLifecycleOps SparkMimo26ModuleLifecycle =
{
	sizeof(SparkMimo26ModuleState),
	SparkMimo26ModuleInitializeGate,
	SparkMimo26ModuleDescribe,
	SparkMimo26ModulePrepare,
	SparkMimo26ModuleReportReady,
	SparkMimo26ModuleStateTeardown,
	SparkMimo26ModuleExecuteFrame,
	SparkMimo26ModuleAdmit,
	SparkMimo26ModuleSnapshotExtend
};

SPARK_STAGE_MODULE_LIFECYCLE_ENTRY_POINTS(
	SparkMimo26ResidentDecodeStage,
	&SparkMimo26ModuleLifecycle)
