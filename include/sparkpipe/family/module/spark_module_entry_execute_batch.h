#pragma once

SparkStatus SPARK_FAMILY(ResidentDecodeStageExecute)(void *module_state,SparkModelDriverFrame *frame)
{
	SPARK_FAMILY(ModuleState) *state;
	const SPARK_FAMILY(ResidentDecodeStageFrameContext) *context;
	SparkStatus status;
	state = (SPARK_FAMILY(ModuleState) *)module_state;
	context = 0;
	status = SPARK_FAMILY(ValidateFrame)(state,frame,&context);
	if ( status != SPARK_STATUS_OK )
	{
		if ( state != 0 )
			atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
		return(status);
	}
	status = SPARK_FAMILY(ExecuteBatch)(state,frame,context);
	if ( status != SPARK_STATUS_OK )
		atomic_fetch_add_explicit(&state->rejected_count,1u,memory_order_relaxed);
	return(status);
}
