#include <assert.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "sparkpipe/spark_hy4_resident_decode_stage_firmware.h"

#define TEST_PROGRAM_ID 7u

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
	void *module_state;
	module_state = (void *)0x1;
	assert(SparkHy4ResidentDecodeStageInitialize(0,0,0) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkHy4ResidentDecodeStageInitialize(&configuration,
		&host_services,0) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkHy4ResidentDecodeStageExecute(0,0) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkHy4ResidentDecodeStageAdmit(0,0,0) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkHy4ResidentDecodeStageSnapshot(0,TEST_PROGRAM_ID,0) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	assert(module_state == (void *)0x1);
}

static void TestSchemaRejected(void)
{
	SparkFirmwareModuleConfiguration configuration = TestConfiguration();
	SparkFirmwareModuleHostServices host_services = TestHostServices();
	void *module_state;
	configuration.model_id = 0;
	module_state = 0;
	assert(SparkHy4ResidentDecodeStageInitialize(&configuration,
		&host_services,&module_state) == SPARK_STATUS_SCHEMA_ERROR);
	assert(module_state == 0);
}

static void TestLifecycleSmoke(void)
{
	SparkFirmwareModuleConfiguration configuration = TestConfiguration();
	SparkFirmwareModuleHostServices host_services = TestHostServices();
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	SparkModelDriverRuntimeSnapshot snapshot;
	SparkModelDriverFrame frame;
	void *module_state;
	SparkStatus status;
	module_state = 0;
	status = SparkHy4ResidentDecodeStageInitialize(&configuration,
		&host_services,&module_state);
	assert(status == SPARK_STATUS_OK);
	assert(module_state != 0);
	memset(&frame,0,sizeof(frame));
	assert(SparkHy4ResidentDecodeStageExecute(module_state,&frame) ==
		SPARK_STATUS_UNSUPPORTED);
	memset(&request,0,sizeof(request));
	memset(&decision,0,sizeof(decision));
	status = SparkHy4ResidentDecodeStageAdmit(module_state,&request,
		&decision);
	assert(status == SPARK_STATUS_OK);
	assert(decision.accepted == 1u);
	assert(decision.available_dispatch_slot_count ==
		SPARK_HY4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCES);
	assert(SparkHy4ResidentDecodeStageSnapshot(module_state,0u,
		&snapshot) == SPARK_STATUS_INVALID_ARGUMENT);
	memset(&snapshot,0,sizeof(snapshot));
	status = SparkHy4ResidentDecodeStageSnapshot(module_state,
		TEST_PROGRAM_ID,&snapshot);
	assert(status == SPARK_STATUS_OK);
	assert(snapshot.program_id == TEST_PROGRAM_ID);
	assert(snapshot.submitted_count == 0u);
	assert(snapshot.completed_count == 0u);
	assert(snapshot.rejected_count == 0u);
	assert(snapshot.active_submission_count == 0u);
	assert(snapshot.available_dispatch_slot_count ==
		SPARK_HY4_RESIDENT_DECODE_STAGE_PIPELINE_SLOT_COUNT);
	SparkHy4ResidentDecodeStageDestroy(module_state);
	SparkHy4ResidentDecodeStageDestroy(0);
}

static void TestEnvironmentRangeEnforced(const char *name,
	const char *value)
{
	SparkFirmwareModuleConfiguration configuration = TestConfiguration();
	SparkFirmwareModuleHostServices host_services = TestHostServices();
	void *module_state;
	module_state = 0;
	setenv(name,value,1);
	assert(SparkHy4ResidentDecodeStageInitialize(&configuration,
		&host_services,&module_state) != SPARK_STATUS_OK);
	assert(module_state == 0);
	unsetenv(name);
}

int main(void)
{
	TestNullArgumentsRejected();
	TestSchemaRejected();
	TestLifecycleSmoke();
	TestEnvironmentRangeEnforced("SPARK_HY4_TP_RANK","99");
	TestEnvironmentRangeEnforced("SPARK_HY4_TP_DEGREE","0");
	TestEnvironmentRangeEnforced("SPARK_HY4_TP_DEGREE","17");
	TestEnvironmentRangeEnforced("SPARK_HY4_STAGE_PIPELINE_SLOTS","5");
	printf("hy4 lifecycle smoke: OK\n");
	return(0);
}
