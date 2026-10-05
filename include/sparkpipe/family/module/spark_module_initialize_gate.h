#pragma once

static SparkStatus SPARK_FAMILY(ModuleInitializeGate)(const SparkFirmwareModuleConfiguration *configuration)
{
	const char *receipt;
	uint32_t index;
	receipt = configuration->validated_artifact_sha256;
	if ( receipt == 0 )
		SPARK_FAIL(SPARK_STATUS_MODULE_NOT_VALIDATED);
	for (index = 0u; index < 64u; index++)
	{
		if ( !((receipt[index] >= '0' && receipt[index] <= '9') || (receipt[index] >= 'a' && receipt[index] <= 'f')) )
			SPARK_FAIL(SPARK_STATUS_MODULE_NOT_VALIDATED);
	}
	if ( receipt[64] != '\0' )
		SPARK_FAIL(SPARK_STATUS_MODULE_NOT_VALIDATED);
	return(SPARK_STATUS_OK);
}
