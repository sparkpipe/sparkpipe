#pragma once

SparkStatus SPARK_FAMILY(ResidentDecodeStageSnapshot)(void *module_state,uint32_t program_id,SparkModelDriverRuntimeSnapshot *snapshot)
{
	(void)module_state;
	(void)program_id;
	(void)snapshot;
	return(SPARK_STATUS_UNSUPPORTED);
}
