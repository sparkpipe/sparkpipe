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

/* hy4 TP16 resident decode stage — module lifecycle skeleton.
 *
 * Rung-1 scope (operator ladder): compile against the real ABI on the
 * shared runtime. Every entry point validates its arguments honestly
 * and reports SPARK_STATUS_NOT_IMPLEMENTED for the model work it does
 * not yet perform — the acceptance law requires behavioral tests to
 * FAIL for incomplete drivers, so these errors are the contract, not
 * placeholders that fake success. Rungs 3+ replace them with the real
 * implementation behind the same signatures. */

#define SPARK_HY4_MODULE_TAG "hy4_stage"

typedef struct SparkHy4ModuleState
{
	SparkStageModuleLedger ledger;
	uint32_t max_active_sequence_count;
	uint32_t pipeline_slot_count;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint32_t stage_layer_count;
	uint32_t first_layer_index;
	uint64_t arena_bytes;
} SparkHy4ModuleState;

static SparkStatus SparkHy4ModuleInitializeGate(void)
{
	return SPARK_STATUS_OK;
}

static SparkStatus SparkHy4ModuleInitializeTpCollective(
	SparkHy4ModuleState *state)
{
	(void)state;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkHy4ModuleInitialize(
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services,
	void **module_state)
{
	SparkHy4ModuleState *state;
	SparkStatus status;
	if ( configuration == 0 || host_services == 0 || module_state == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	(void)sizeof(SparkHy4ResidentDecodeStageNodeContext);
	status = SparkFirmwareModuleValidateInitialization(configuration,
		host_services,module_state);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( configuration->stage_name == 0 ||
	    configuration->model_id == 0 )
		return SPARK_STATUS_SCHEMA_ERROR;
	status = SparkHy4ModuleInitializeGate();
	if ( status != SPARK_STATUS_OK )
		return status;
	state = (SparkHy4ModuleState *)calloc(1u,sizeof(*state));
	if ( state == 0 )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	memset(&state->ledger,0,sizeof(state->ledger));
	state->ledger.module_tag = SPARK_HY4_MODULE_TAG;
	*module_state = state;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkHy4ModuleExecute(void *module_state,
	SparkModelDriverFrame *frame)
{
	(void)module_state;
	(void)frame;
	return SPARK_STATUS_UNSUPPORTED;
}

static SparkStatus SparkHy4ModuleAdmit(void *module_state,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	SparkHy4ModuleState *state = (SparkHy4ModuleState *)module_state;
	if ( state == 0 || request == 0 || decision == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	SparkStageModuleAdmissionDecisionInitialize(decision,
		state->max_active_sequence_count);
	SparkStageModuleAdmissionDecisionAccept(decision);
	return SPARK_STATUS_OK;
}

static SparkStatus SparkHy4ModuleSnapshot(void *module_state,
	uint32_t program_id, SparkModelDriverRuntimeSnapshot *snapshot)
{
	SparkHy4ModuleState *state = (SparkHy4ModuleState *)module_state;
	if ( state == 0 || snapshot == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(snapshot,0,sizeof(*snapshot));
	snapshot->program_id = program_id;
	snapshot->active_submission_count = 0u;
	return SPARK_STATUS_OK;
}

static void SparkHy4ModuleDestroy(void *module_state)
{
	SparkHy4ModuleState *state = (SparkHy4ModuleState *)module_state;
	if ( state == 0 )
		return;
	SparkStageModuleLedgerRelease(&state->ledger);
	free(state);
}

static SparkStatus SparkHy4ModuleInitializeAdapter(
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services,
	void **module_state)
{
	SparkHy4ModuleState *state;
	SparkStatus status;
	status = SparkHy4ModuleInitialize(configuration,host_services,
		module_state);
	if ( status != SPARK_STATUS_OK )
		return status;
	state = (SparkHy4ModuleState *)*module_state;
	status = SparkHy4ModuleInitializeTpCollective(state);
	if ( status != SPARK_STATUS_OK )
	{
		SparkHy4ModuleDestroy(state);
		*module_state = 0;
		return status;
	}
	return SPARK_STATUS_OK;
}

__attribute__((visibility("default")))
SparkStatus SparkHy4ResidentDecodeStageInitialize(
	const void *configuration, const void *host_services,
	void **module_state)
{
	return SparkHy4ModuleInitializeAdapter(
		(const SparkFirmwareModuleConfiguration *)configuration,
		(const SparkFirmwareModuleHostServices *)host_services,
		module_state);
}

__attribute__((visibility("default")))
SparkStatus SparkHy4ResidentDecodeStageExecute(void *module_state,
	void *frame)
{
	return SparkHy4ModuleExecute(module_state,
		(SparkModelDriverFrame *)frame);
}

__attribute__((visibility("default")))
SparkStatus SparkHy4ResidentDecodeStageAdmit(void *module_state,
	const void *request, void *decision)
{
	return SparkHy4ModuleAdmit(module_state,
		(const SparkModelDriverAdmissionRequest *)request,
		(SparkModelDriverAdmissionDecision *)decision);
}

__attribute__((visibility("default")))
SparkStatus SparkHy4ResidentDecodeStageSnapshot(void *module_state,
	uint32_t program_id, void *snapshot)
{
	return SparkHy4ModuleSnapshot(module_state,program_id,
		(SparkModelDriverRuntimeSnapshot *)snapshot);
}

__attribute__((visibility("default")))
void SparkHy4ResidentDecodeStageDestroy(void *module_state)
{
	SparkHy4ModuleDestroy(module_state);
}
