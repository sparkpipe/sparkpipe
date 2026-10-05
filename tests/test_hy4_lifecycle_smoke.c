#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "sparkpipe/spark_hy4_resident_decode_stage_firmware.h"

#define TEST_PROGRAM_ID 7u

static uint32_t test_failures;

static void TestCheck(int condition,const char *what)
{
	if ( condition )
		return;
	fprintf(stderr,"FAIL %s\n",what);
	test_failures++;
}

static SparkFirmwareModuleConfiguration TestConfiguration(void)
{
	SparkFirmwareModuleConfiguration configuration;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_FIRMWARE_MODULE_ABI_VERSION;
	configuration.descriptor_bytes = (uint32_t)sizeof(configuration);
	configuration.model_id = "AngelSlim/Hy4-preview-GGUF";
	configuration.model_revision =
		"779242edccdedc2109a0b36b164263a88f015bfa";
	configuration.stage_name = "hy4.resident_decode_stage";
	return(configuration);
}

static SparkFirmwareModuleHostServices TestHostServices(void)
{
	SparkFirmwareModuleHostServices host_services;
	memset(&host_services,0,sizeof(host_services));
	host_services.abi_version =
		SPARK_FIRMWARE_MODULE_HOST_SERVICES_ABI_VERSION;
	host_services.descriptor_bytes = (uint32_t)sizeof(host_services);
	return(host_services);
}

static void TestNullArgumentsRejected(void)
{
	SparkFirmwareModuleConfiguration configuration = TestConfiguration();
	SparkFirmwareModuleHostServices host_services = TestHostServices();
	TestCheck(SparkHy4ResidentDecodeStageInitialize(0,0,0) ==
		SPARK_STATUS_INVALID_ARGUMENT,"initialize without arguments");
	TestCheck(SparkHy4ResidentDecodeStageInitialize(&configuration,
		&host_services,0) == SPARK_STATUS_INVALID_ARGUMENT,
		"initialize without a state pointer");
	TestCheck(SparkHy4ResidentDecodeStageExecute(0,0) ==
		SPARK_STATUS_INVALID_ARGUMENT,"execute without a state");
	TestCheck(SparkHy4ResidentDecodeStageAdmit(0,0,0) ==
		SPARK_STATUS_INVALID_ARGUMENT,"admit without a state");
	TestCheck(SparkHy4ResidentDecodeStageSnapshot(0,TEST_PROGRAM_ID,0) ==
		SPARK_STATUS_INVALID_ARGUMENT,"snapshot without a state");
	SparkHy4ResidentDecodeStageDestroy(0);
}

static void TestSchemaRejected(void)
{
	SparkFirmwareModuleConfiguration configuration = TestConfiguration();
	SparkFirmwareModuleHostServices host_services = TestHostServices();
	void *module_state;
	configuration.model_id = 0;
	module_state = 0;
	TestCheck(SparkHy4ResidentDecodeStageInitialize(&configuration,
		&host_services,&module_state) == SPARK_STATUS_SCHEMA_ERROR,
		"initialize without a model id is a schema error");
	TestCheck(module_state == 0,"schema error leaves no state");
	configuration = TestConfiguration();
	configuration.stage_name = 0;
	TestCheck(SparkHy4ResidentDecodeStageInitialize(&configuration,
		&host_services,&module_state) == SPARK_STATUS_SCHEMA_ERROR,
		"initialize without a stage name is a schema error");
	TestCheck(module_state == 0,"schema error leaves no state");
}

static void TestEnvironmentRangeEnforced(const char *name,
	const char *value)
{
	SparkFirmwareModuleConfiguration configuration = TestConfiguration();
	SparkFirmwareModuleHostServices host_services = TestHostServices();
	void *module_state;
	char what[160];
	module_state = 0;
	setenv(name,value,1);
	snprintf(what,sizeof(what),"%s=%s is refused as an invalid argument",
		name,value);
	TestCheck(SparkHy4ResidentDecodeStageInitialize(&configuration,
		&host_services,&module_state) == SPARK_STATUS_INVALID_ARGUMENT,
		what);
	TestCheck(module_state == 0,"refused configuration leaves no state");
	unsetenv(name);
}

static void TestValidConfigurationRefused(void)
{
	SparkFirmwareModuleConfiguration configuration = TestConfiguration();
	SparkFirmwareModuleHostServices host_services = TestHostServices();
	void *module_state = 0;
	TestCheck(SparkHy4ResidentDecodeStageInitialize(&configuration,
		&host_services,&module_state) == SPARK_STATUS_UNSUPPORTED,
		"a valid configuration is refused while the driver cannot execute the model");
	TestCheck(module_state == 0,"the refused driver leaves no state and no ready rank");
}

int main(void)
{
	TestNullArgumentsRejected();
	TestValidConfigurationRefused();
	TestSchemaRejected();
	TestEnvironmentRangeEnforced("SPARK_HY4_TP_RANK","99");
	TestEnvironmentRangeEnforced("SPARK_HY4_TP_DEGREE","0");
	TestEnvironmentRangeEnforced("SPARK_HY4_TP_DEGREE","17");
	TestEnvironmentRangeEnforced("SPARK_HY4_TP_DEGREE","4x");
	TestEnvironmentRangeEnforced("SPARK_HY4_STAGE_MAX_ACTIVE_SEQUENCES","0");
	TestEnvironmentRangeEnforced("SPARK_HY4_STAGE_PIPELINE_SLOTS","5");
	if ( test_failures != 0u )
	{
		fprintf(stderr,"hy4 lifecycle smoke: %u failures\n",test_failures);
		return(1);
	}
	printf("hy4 lifecycle smoke: OK\n");
	return(0);
}
