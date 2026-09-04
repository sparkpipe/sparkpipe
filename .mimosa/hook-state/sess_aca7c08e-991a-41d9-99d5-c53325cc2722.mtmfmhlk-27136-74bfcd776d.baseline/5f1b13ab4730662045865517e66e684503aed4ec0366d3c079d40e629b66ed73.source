#pragma once

#include <stdint.h>

#include "sparkpipe/spark_model_resident_endpoint.h"
#include "sparkpipe/spark_model_serving_adapter.h"

typedef struct TestModelResidentDeploymentFixture
{
	const char *adapter_shared_object_path;
	const char *driver_shared_object_path;
	const char *driver_program_name;
	const char *transport_shared_object_path;
	const char *transport_mode;
	const char *node_target;
	const char *adapter_configuration_path;
	const char *kv_backing_directory;
	uint64_t kv_backing_maximum_bytes;
	const char *tokenizer_asset_path;
	const char *const *runtime_roots;
	const char *const *transport_hosts;
	const uint32_t *stage_indices;
	const SparkModelResidentEndpoint *control_endpoints;
	SparkModelServingRuntimeLimits runtime_limits;
	uint32_t control_port_base;
	uint32_t node_count;
	uint32_t coordinator_rank_index;
} TestModelResidentDeploymentFixture;

int32_t TestModelResidentDeploymentWrite(
	const char *path,
	const TestModelResidentDeploymentFixture *fixture);
