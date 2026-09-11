#define _FILE_OFFSET_BITS 64

#include <errno.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

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
#define SPARK_HY4_TP_COLLECTIVE_CREDITS_PER_SLOT 2u

extern cudaError_t SparkHy4LaunchAccumAddBf16(cudaStream_t stream,
	void *destination_bf16,const void *source_bf16,
	uint32_t row_count,uint32_t width);
extern cudaError_t SparkHy4LaunchAccumU64Max(cudaStream_t stream,
	uint64_t *destination,const uint64_t *source,
	uint32_t element_count);

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
	void *execution_stream;
	uint32_t tp_collective_disabled;
	SparkTpDeviceCollective tp_device_collective;
	uint32_t tp_device_collective_initialized;
} SparkHy4ModuleState;

static SparkStatus SparkHy4ModuleInitializeGate(void)
{
	return SPARK_STATUS_OK;
}

static SparkStatus SparkHy4ModuleCombineBf16(void *combine_context,
	void *destination_device,const void *source_device,
	uint32_t active_sequence_count,uint32_t hidden_dimension,
	void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkHy4LaunchAccumAddBf16((cudaStream_t)cuda_stream,
		destination_device,source_device,active_sequence_count,
		hidden_dimension);
	return SparkStageModuleCudaStatus(SPARK_HY4_MODULE_TAG,error,
		"tp_all_reduce_sum");
}

static SparkStatus SparkHy4ModuleCombineU64Max(void *combine_context,
	uint64_t *destination_device,const uint64_t *source_device,
	uint32_t element_count,void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkHy4LaunchAccumU64Max((cudaStream_t)cuda_stream,
		destination_device,source_device,element_count);
	return SparkStageModuleCudaStatus(SPARK_HY4_MODULE_TAG,error,
		"tp_all_reduce_max_u64");
}

static SparkStatus SparkHy4ModuleInitializeTpCollective(
	SparkHy4ModuleState *state,
	const SparkFirmwareModuleHostServices *host_services)
{
	const SparkHy4ResidentDecodeStageNodeContext *context;
	SparkTpDeviceCollectiveConfig configuration;
	SparkStatus status;
	if ( state == 0 || host_services == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	context = (const SparkHy4ResidentDecodeStageNodeContext *)
		host_services->node_context;
	if ( context == 0 )
		return SPARK_STATUS_OK;
	state->execution_stream = host_services->execution_stream;
	state->max_active_sequence_count = context->max_active_sequence_count;
	state->pipeline_slot_count = context->pipeline_slot_count;
	state->tp_degree = context->tp_degree;
	state->tp_rank = context->tp_rank;
	if ( context->tp_degree <= 1u ||
	    context->tp_collective_identifier == 0u )
	{
		state->tp_collective_disabled = 1u;
		return SPARK_STATUS_OK;
	}
	if ( context->tp_degree > SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE ||
	    context->tp_rank >= context->tp_degree ||
	    host_services->execution_stream == 0 ||
	    context->tp_collective_topology.rank_count !=
	    context->tp_degree ||
	    context->tp_collective_backend_kind !=
	    SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT ||
	    context->tp_collective_mesh_addr == 0u ||
	    context->tp_connect_timeout_milli == 0u ||
	    context->tp_operation_timeout_milli == 0u ||
	    context->tp_collective_backend_module_path == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	configuration.backend_kind = context->tp_collective_backend_kind;
	configuration.tp_degree = context->tp_degree;
	configuration.tp_rank = context->tp_rank;
	configuration.operation_kind =
		SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	configuration.credit_count = state->pipeline_slot_count *
		SPARK_HY4_TP_COLLECTIVE_CREDITS_PER_SLOT;
	configuration.local_hidden_dimension =
		SPARK_HY4_MODEL_HIDDEN_DIMENSION;
	configuration.max_active_sequence_count =
		state->max_active_sequence_count;
	configuration.connect_timeout_milli =
		context->tp_connect_timeout_milli;
	configuration.operation_timeout_milli =
		context->tp_operation_timeout_milli;
	configuration.control_port_base =
		context->tp_collective_control_port_base;
	configuration.collective_identifier =
		context->tp_collective_identifier;
	configuration.backend_module_path =
		context->tp_collective_backend_module_path;
	configuration.registration_cuda_stream = state->execution_stream;
	status = SparkTpDeviceCollectiveApplyTopology(
		&context->tp_collective_topology,&configuration);
	if ( status != SPARK_STATUS_OK )
		return status;
	configuration.combine_bf16_function = SparkHy4ModuleCombineBf16;
	configuration.combine_u64_max_function =
		SparkHy4ModuleCombineU64Max;
	configuration.combine_context = state;
	status = SparkTpDeviceCollectiveCreate(&configuration,
		&state->tp_device_collective);
	if ( status != SPARK_STATUS_OK )
		return status;
	state->tp_device_collective_initialized = 1u;
	status = SparkTpDeviceCollectivePrepareReceiveBf16(
		&state->tp_device_collective,
		(void *)(uintptr_t)context->tp_collective_mesh_addr,
		0u,0u,0u,0u);
	if ( status != SPARK_STATUS_OK )
		return status;
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
	if ( state->tp_device_collective_initialized != 0u )
	{
		state->tp_device_collective_initialized = 0u;
		SparkTpDeviceCollectiveDestroy(&state->tp_device_collective);
	}
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
	status = SparkHy4ModuleInitializeTpCollective(state,host_services);
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
