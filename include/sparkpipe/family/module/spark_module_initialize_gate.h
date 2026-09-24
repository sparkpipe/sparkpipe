#pragma once

static SparkStatus SPARK_FAMILY(ModuleInitializeGate)(void)
{
	uint32_t allow_unqualified_execution;
	allow_unqualified_execution = 0u;
	if ( SparkStageModuleEnvironmentUnsigned(SPARK_FAMILY_CONST(MODULE_TAG),"SPARK_" SPARK_FAMILY_STRING(SPARK_FAMILY_UPPER) "_ALLOW_UNQUALIFIED_EXECUTION",1u,1u,&allow_unqualified_execution) != SPARK_STATUS_OK || allow_unqualified_execution != 1u )
		SPARK_FAIL(SPARK_STATUS_MODULE_NOT_VALIDATED);
	return(SPARK_STATUS_OK);
}
