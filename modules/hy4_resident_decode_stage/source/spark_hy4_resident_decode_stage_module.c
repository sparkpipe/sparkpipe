#define _FILE_OFFSET_BITS 64

#include <errno.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_admission.h"
#include "sparkpipe/spark_hidden_transport.h"
#include "sparkpipe/spark_hy4_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "sparkpipe/spark_stage_module_lifecycle.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "spark_hy4_stagepack_format.h"

#define SPARK_HY4_MODULE_TAG "hy4_stage"

typedef struct SparkHy4ModuleState
{
	SparkStageModuleLedger ledger;
	atomic_uint slot_states[SPARK_HY4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT];
	atomic_ullong submitted_count;
	atomic_ullong completed_count;
	atomic_ullong rejected_count;
	atomic_ullong failed_count;
	atomic_ullong tokens_emitted;
	uint32_t max_active_sequence_count;
	uint32_t pipeline_slot_count;
	uint32_t tp_degree;
	uint32_t tp_rank;
} SparkHy4ModuleState;

static SparkStatus SparkHy4ModuleConfigure(void *module_state,
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services)
{
	SparkHy4ModuleState *state;
	SparkStatus status;
	uint32_t tp_degree;
	(void)host_services;
	state = (SparkHy4ModuleState *)module_state;
	if ( configuration->stage_name == 0 ||
	    configuration->model_id == 0 )
		return(SPARK_STATUS_SCHEMA_ERROR);
	(void)sizeof(SparkHy4ResidentDecodeStageNodeContext);
	state->max_active_sequence_count =
		SPARK_HY4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCES;
	state->pipeline_slot_count =
		SPARK_HY4_RESIDENT_DECODE_STAGE_PIPELINE_SLOT_COUNT;
	tp_degree = SPARK_HY4_MODEL_TP_RANKS;
	status = SparkStageModuleEnvironmentUnsignedOrDefault(
		SPARK_HY4_MODULE_TAG,"SPARK_HY4_TP_DEGREE",1u,
		SPARK_HY4_MODEL_TP_RANKS,SPARK_HY4_MODEL_TP_RANKS,&tp_degree);
	if ( status != SPARK_STATUS_OK )
		return(status);
	state->tp_degree = tp_degree;
	state->tp_rank = 0u;
	status = SparkStageModuleEnvironmentUnsignedOrDefault(
		SPARK_HY4_MODULE_TAG,"SPARK_HY4_TP_RANK",0u,tp_degree - 1u,0u,
		&state->tp_rank);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkStageModuleEnvironmentUnsignedOrDefault(
		SPARK_HY4_MODULE_TAG,"SPARK_HY4_STAGE_MAX_ACTIVE_SEQUENCES",
		1u,SPARK_HY4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCES,
		SPARK_HY4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCES,
		&state->max_active_sequence_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkStageModuleEnvironmentUnsignedOrDefault(
		SPARK_HY4_MODULE_TAG,"SPARK_HY4_STAGE_PIPELINE_SLOTS",1u,
		SPARK_HY4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,
		SPARK_HY4_RESIDENT_DECODE_STAGE_PIPELINE_SLOT_COUNT,
		&state->pipeline_slot_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	atomic_init(&state->submitted_count,0ull);
	atomic_init(&state->completed_count,0ull);
	atomic_init(&state->rejected_count,0ull);
	atomic_init(&state->failed_count,0ull);
	atomic_init(&state->tokens_emitted,0ull);
	SparkStageModuleAtomicStateArrayInitialize(state->slot_states,
		state->pipeline_slot_count);
	return(SPARK_STATUS_OK);
}

static void SparkHy4ModuleDescribe(void *module_state,
	SparkStageModuleLifecycle *lifecycle)
{
	SparkHy4ModuleState *state;
	state = (SparkHy4ModuleState *)module_state;
	lifecycle->module_tag = SPARK_HY4_MODULE_TAG;
	lifecycle->ledger = &state->ledger;
	lifecycle->slot_states = state->slot_states;
	lifecycle->pipeline_slot_count = state->pipeline_slot_count;
	lifecycle->submitted_count = &state->submitted_count;
	lifecycle->completed_count = &state->completed_count;
	lifecycle->rejected_count = &state->rejected_count;
	lifecycle->failed_count = &state->failed_count;
	lifecycle->tokens_emitted = &state->tokens_emitted;
}

static SparkStatus SparkHy4ModulePrepare(void *module_state,
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services)
{
	SparkHy4ModuleState *state;
	SparkStatus status;
	state = (SparkHy4ModuleState *)module_state;
	status = SparkHy4ModuleConfigure(module_state,configuration,
		host_services);
	if ( status != SPARK_STATUS_OK )
		return(status);
	fprintf(stderr,
		"%s ready tp=%u/%u slots=%u max_active=%u execute=UNSUPPORTED\n",
		SPARK_HY4_MODULE_TAG,state->tp_rank,state->tp_degree,
		state->pipeline_slot_count,state->max_active_sequence_count);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkHy4ModuleExecute(void *module_state,
	SparkModelDriverFrame *frame)
{
	(void)module_state;
	(void)frame;
	return(SPARK_STATUS_UNSUPPORTED);
}

static SparkStatus SparkHy4ModuleAdmit(void *module_state,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	SparkHy4ModuleState *state;
	(void)request;
	state = (SparkHy4ModuleState *)module_state;
	SparkStageModuleAdmissionDecisionInitialize(decision,
		state->max_active_sequence_count);
	SparkStageModuleAdmissionDecisionAccept(decision);
	return(SPARK_STATUS_OK);
}

static const SparkStageModuleLifecycleOps SparkHy4ModuleLifecycle =
{
	sizeof(SparkHy4ModuleState),
	0,
	SparkHy4ModuleDescribe,
	SparkHy4ModulePrepare,
	0,
	0,
	SparkHy4ModuleExecute,
	SparkHy4ModuleAdmit,
	0
};

SPARK_STAGE_MODULE_LIFECYCLE_ENTRY_POINTS(
	SparkHy4ResidentDecodeStage,
	&SparkHy4ModuleLifecycle)
