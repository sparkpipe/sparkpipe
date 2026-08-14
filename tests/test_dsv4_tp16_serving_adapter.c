#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "sparkpipe/spark_dsv4_model.h"
#include "sparkpipe/spark_model_serving_adapter.h"

#ifndef TEST_DSV4_TP16_ADAPTER_PATH
#define TEST_DSV4_TP16_ADAPTER_PATH ""
#endif
#ifndef TEST_DSV4_TP16_DRIVER_PATH
#define TEST_DSV4_TP16_DRIVER_PATH ""
#endif
#ifndef TEST_DSV4_TP16_CONFIG_PATH
#define TEST_DSV4_TP16_CONFIG_PATH ""
#endif

static void TestDsv4Tp16Completion(void *context,

	const SparkModelServingCompletion *completion)
{
	(void)context;
	(void)completion;
}

int main(void)
{
	SparkModelServingAdapterDynamicLibrary library;
	SparkModelServingAdapterConfiguration configuration;
	SparkModelServingAdapterSnapshot snapshot;
	void *adapter_state;
	char runtime_root[4096];
	SparkStatus status;
	assert(SparkModelServingAdapterLoadInterfaceFromSharedObject(
		TEST_DSV4_TP16_ADAPTER_PATH,
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PARALLEL_FANOUT |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFETCH |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DRIVER_OWNS_KV |
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_JIT_KV,
		&library) == SPARK_STATUS_OK);
	assert(library.adapter_interface.descriptor->stage_count == 16u);
	assert(library.adapter_interface.descriptor->max_inflight_submission_count == 16u);
	assert(library.adapter_interface.descriptor->minimum_efficient_submission_row_count == 1u);
	assert((library.adapter_interface.descriptor->capability_flags &
		SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT) == 0u);
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_CONFIGURATION_BYTES;
	configuration.rank_index = 15u;
	configuration.stage_index = 15u;
	configuration.runtime_limits.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	configuration.runtime_limits.descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	configuration.runtime_limits.max_inflight_submission_count = 1u;
	configuration.runtime_limits.max_active_sequence_count = 1u;
	configuration.runtime_limits.max_input_row_count = 1u;
	configuration.runtime_limits.resident_sequence_capacity = 128u;
	configuration.runtime_limits.kv_logical_page_capacity = 128u;
	configuration.runtime_limits.kv_physical_page_capacity = 128u;
	assert(getcwd(runtime_root,sizeof(runtime_root)) != 0);
	configuration.runtime_root = runtime_root;
	configuration.node_id = "spark15";
	configuration.node_target = SPARK_DSV4_MODEL_MODULE_TARGET;
	configuration.adapter_configuration_path = TEST_DSV4_TP16_CONFIG_PATH;
	configuration.driver_shared_object_path = TEST_DSV4_TP16_DRIVER_PATH;
	configuration.driver_program_name = "resident_decode";
	configuration.execution_stream = (void *)(uintptr_t)1u;
	configuration.completion_function = TestDsv4Tp16Completion;
	adapter_state = 0;
	status = library.adapter_interface.initialize(&configuration,&adapter_state);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"TP16 adapter initialize status=%d (%s)\n",(int)status,SparkStatusToString(status));
	assert(status == SPARK_STATUS_OK);
	assert(adapter_state != 0);
	assert(library.adapter_interface.snapshot(adapter_state,&snapshot) ==
		SPARK_STATUS_OK);
	assert(snapshot.kv_token_capacity == 130u);
	library.adapter_interface.destroy(adapter_state);
	SparkModelServingAdapterUnloadInterface(&library);
	return(0);
}
