#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "sparkpipe/spark_hy4_resident_decode_stage_firmware.h"

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

static void TestStateWithoutPackIsNotServiceable(void *module_state)
{
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	SparkModelDriverFrame frame;
	SparkStatus status;
	memset(&request,0,sizeof(request));
	memset(&decision,0,sizeof(decision));
	status = SparkHy4ResidentDecodeStageAdmit(module_state,&request,
		&decision);
	TestCheck(status == SPARK_STATUS_ABI_MISMATCH,
		"admit refuses an all-zero request through the common admission "
		"policy (ABI_MISMATCH)");
	TestCheck(decision.accepted == 0u,
		"admit does not accept an all-zero request");
	memset(&frame,0,sizeof(frame));
	TestCheck(SparkHy4ResidentDecodeStageExecute(module_state,&frame) !=
		SPARK_STATUS_UNSUPPORTED,"execute is implemented");
}

int main(void)
{
	SparkFirmwareModuleConfiguration configuration = TestConfiguration();
	SparkFirmwareModuleHostServices host_services = TestHostServices();
	void *module_state;
	SparkStatus status;
	module_state = 0;
	status = SparkHy4ResidentDecodeStageInitialize(&configuration,
		&host_services,&module_state);
	TestCheck(status != SPARK_STATUS_OK,
		"initialize without a stage pack or .experts manifest fails "
		"explicitly");
	TestCheck(module_state == 0,"a refused initialize leaves no state");
	if ( status == SPARK_STATUS_OK && module_state != 0 )
	{
		TestStateWithoutPackIsNotServiceable(module_state);
		SparkHy4ResidentDecodeStageDestroy(module_state);
	}
	if ( test_failures != 0u )
	{
		fprintf(stderr,"hy4 driver acceptance: %u failures\n",
			test_failures);
		return(1);
	}
	printf("hy4 driver acceptance: OK\n");
	return(0);
}
