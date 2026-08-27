// K3 serving adapter smoke: initialize the full adapter (config JSON ->
// runner -> pack/bind/dispatch) on a REAL rank pack with tp_degree 1 (no
// collective, so nothing binds a port), then destroy. Proves the residentd
// path works end to end before the fleet run.

#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_k3_serving_adapter.h"
#include "sparkpipe/spark_model_serving_adapter.h"

int main(int argc, char **argv)
{
	if ( argc < 2 )
	{
		printf("usage: k3_adapter_smoke <adapter_config.json>\n");
		return 2;
	}
	const SparkModelServingAdapterInterface *iface =
		SparkModelServingAdapterGetInterface();
	if ( iface == 0 || iface->initialize == 0 )
	{
		printf("INTERFACE FAIL\n");
		return 1;
	}
	printf("adapter id=%s stage_count=%u\n", iface->descriptor->adapter_id,
		iface->descriptor->stage_count);
	SparkModelServingAdapterConfiguration configuration;
	memset(&configuration, 0, sizeof(configuration));
	configuration.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	configuration.descriptor_bytes = (uint32_t)sizeof(configuration);
	configuration.rank_index = 0u;
	configuration.stage_index = 0u;
	configuration.adapter_configuration_path = argv[1];
	configuration.driver_shared_object_path = "lib/libk3_serving_adapter.so";
	configuration.driver_program_name = "k3";
	configuration.execution_stream = 0;
	void *state = 0;
	SparkStatus status = iface->initialize(&configuration, &state);
	if ( status != SPARK_STATUS_OK )
	{
		printf("INIT FAIL %d\n", (int)status);
		return 1;
	}
	SparkModelServingAdapterSnapshot snapshot;
	status = iface->snapshot(state, &snapshot);
	if ( status != SPARK_STATUS_OK )
	{
		printf("SNAPSHOT FAIL %d\n", (int)status);
		return 1;
	}
	printf("snapshot: submitted=%llu available=%u\n",
		(unsigned long long)snapshot.submitted_count,
		snapshot.available_submission_count);
	iface->destroy(state);
	printf("k3 adapter smoke PASS\n");
	return 0;
}
