#pragma once

static SparkStatus SPARK_FAMILY(ModuleInitializeTpCollective)(SPARK_FAMILY(ModuleState) *state)
{
	SparkTpDeviceCollectiveConfig configuration;
	SparkStatus status;
	if ( state->tp_degree == 1u )
		return(SPARK_STATUS_OK);
	if ( state->tp_standalone != 0u )
	{
		fprintf(stderr,"%s tp_collective_skipped standalone=1 degree=%u rank=%u\n",SPARK_FAMILY_CONST(MODULE_TAG),state->tp_degree,state->tp_rank);
		return(SPARK_STATUS_OK);
	}
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	configuration.backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
	configuration.tp_degree = state->tp_degree;
	configuration.tp_rank = state->tp_rank;
	configuration.operation_kind = SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16;
	configuration.local_hidden_dimension = SPARK_FAMILY_CONST(MODULE_TP_HIDDEN_DIMENSION);
	configuration.max_active_sequence_count = SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT);
	configuration.operation_timeout_milli = state->tp_operation_timeout_milli;
	SparkTpMeshRegisterCommonCombines(&configuration);
	status = SparkTpDeviceCollectiveCreate(&configuration,&state->tp_device_collective);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"%s tp_create_failed status=%d\n",SPARK_FAMILY_CONST(MODULE_TAG),(int)status);
		SPARK_RETURN(status);
	}
	state->tp_collective_initialized = 1u;
	status = SparkTpDeviceCollectiveAttach(&state->tp_device_collective,SPARK_FAMILY_CONST(MODULE_TP_MESH_REGION)(state));
	if ( status == SPARK_STATUS_OK )
		fprintf(stderr,"%s tp_collective_open degree=%u rank=%u\n",SPARK_FAMILY_CONST(MODULE_TAG),state->tp_degree,state->tp_rank);
	SPARK_RETURN(status);
}
