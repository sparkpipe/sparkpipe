#pragma once

static SparkStatus SPARK_FAMILY(ModuleInitializeTpCollective)(
	SPARK_FAMILY(ModuleState) *state,
	const SPARK_FAMILY(ResidentDecodeStageNodeContext) *context)
{
	SparkTpDeviceCollectiveConfig configuration;
	SparkStatus status;
	if ( state == 0 || context == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->tp_degree == 1u || SPARK_FAMILY_CONST(MODULE_TP_DISABLED)(state) )
		return(SPARK_STATUS_OK);
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	configuration.backend_kind = context->tp_collective_backend_kind;
	configuration.tp_degree = state->tp_degree;
	configuration.tp_rank = state->tp_rank;
	configuration.operation_kind = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	configuration.local_hidden_dimension = SPARK_FAMILY_CONST(MODEL_HIDDEN_DIMENSION);
	configuration.max_active_sequence_count = SPARK_FAMILY_CONST(MODULE_TP_ROW_CAPACITY)(state);
	configuration.operation_timeout_milli = context->tp_operation_timeout_milli;
	SparkTpMeshRegisterCommonCombines(&configuration);
	status = SparkTpDeviceCollectiveApplyTopology(&context->tp_collective_topology,&configuration);
	if ( status == SPARK_STATUS_OK )
		status = SparkTpDeviceCollectiveCreate(&configuration,&state->tp_device_collective);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	state->tp_device_collective_initialized = 1u;
	status = SparkTpDeviceCollectiveAttach(&state->tp_device_collective,SPARK_FAMILY_CONST(MODULE_TP_MESH_REGION)(state));
	SPARK_RETURN(status);
}
