#include <assert.h>
#include <string.h>

#include "sparkpipe/spark_model_resident_deployment.h"

static void TestBuildDescriptor(
	SparkModelServingAdapterDescriptor *descriptor)
{
	memset(descriptor,0,sizeof(*descriptor));
	descriptor->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	descriptor->descriptor_bytes = SPARK_MODEL_SERVING_ADAPTER_DESCRIPTOR_BYTES;
	descriptor->capability_flags = SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_PREFILL | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_DECODE | SPARK_MODEL_SERVING_ADAPTER_CAPABILITY_HIDDEN_TRANSPORT;
	descriptor->stage_count = 3u;
	descriptor->layer_count = 6u;
	descriptor->boundary_format = SPARK_MODEL_SERVING_BOUNDARY_FORMAT_BF16;
	descriptor->boundary_element_count = 64u;
	descriptor->boundary_element_bytes = 2u;
	descriptor->linear_weight_codec = SPARK_WEIGHT_CODEC_BF16;
	descriptor->expert_weight_codec = SPARK_WEIGHT_CODEC_INT8;
	descriptor->kv_cache_codec = SPARK_WEIGHT_CODEC_BF16;
	descriptor->max_inflight_submission_count = 4u;
	descriptor->max_active_sequence_count = 4u;
	descriptor->max_input_row_count = 8u;
	descriptor->max_resident_sequence_count = 16u;
	descriptor->max_output_token_count = 4u;
	descriptor->adapter_id = "test.adapter";
	descriptor->model_id = "test/model";
	descriptor->model_revision = "revision";
	descriptor->driver_program_name = "resident_decode";
	descriptor->artifact_sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
	descriptor->stage_layer_counts[0] = 2u;
	descriptor->stage_layer_counts[1] = 2u;
	descriptor->stage_layer_counts[2] = 2u;
}

int main(void)
{
	SparkModelResidentDeployment deployment;
	SparkModelServingAdapterDescriptor descriptor;
	const SparkModelResidentDeploymentNode *node;
	char path[SPARK_MODEL_RESIDENT_DEPLOYMENT_PATH_BYTES];
	SparkModelResidentDeploymentReset(&deployment);
	assert(SparkModelResidentDeploymentLoad("tests/fixtures/model_resident_deployment.json",&deployment) == SPARK_STATUS_OK);
	assert(deployment.node_count == 3u);
	assert(deployment.runtime_limits.resident_sequence_capacity == 8u);
	assert(strcmp(deployment.driver_program_name,"resident_decode") == 0);
	node = SparkModelResidentDeploymentFindRank(&deployment,1u);
	assert(node != 0);
	assert(node->stage_index == 2u);
	assert(strcmp(node->transport_host,"spark1") == 0);
	assert(node->control_endpoint.kind == SPARK_MODEL_RESIDENT_ENDPOINT_KIND_UNIX);
	assert(strcmp(node->control_endpoint.unix_socket_path,"/tmp/test-model-resident-1.sock") == 0);
	assert(SparkModelResidentDeploymentResolvePath(node,deployment.driver_shared_object_path,path,sizeof(path)) == SPARK_STATUS_OK);
	assert(strcmp(path,"/tmp/test-runtime-1/build/test_modules/libdsv4_serving_driver_module.dylib") == 0);
	assert(SparkModelResidentDeploymentResolvePath(node,"../driver.so",path,sizeof(path)) == SPARK_STATUS_INVALID_ARGUMENT);
	node = SparkModelResidentDeploymentFindStage(&deployment,1u);
	assert(node != 0);
	assert(node->rank_index == 2u);
	TestBuildDescriptor(&descriptor);
	assert(SparkModelResidentDeploymentValidateForAdapter(&deployment,&descriptor) == SPARK_STATUS_OK);
	descriptor.driver_program_name = "other";
	assert(SparkModelResidentDeploymentValidateForAdapter(&deployment,&descriptor) == SPARK_STATUS_TARGET_MISMATCH);
	SparkModelResidentDeploymentDestroy(&deployment);
	assert(SparkModelResidentDeploymentLoad("tests/fixtures/model_resident_deployment_unknown.json",&deployment) == SPARK_STATUS_SCHEMA_ERROR);
	assert(SparkModelResidentDeploymentLoad("tests/fixtures/model_resident_deployment_duplicate_rank.json",&deployment) == SPARK_STATUS_SCHEMA_ERROR);
	assert(SparkModelResidentDeploymentLoad("examples/deployments/dsv4_flash_pp13_host_rdma.json",&deployment) == SPARK_STATUS_OK);
	assert(deployment.node_count == 13u);
	assert(strcmp(deployment.adapter_shared_object_path,"lib/model_serving_adapter.so") == 0);
	assert(strcmp(deployment.transport_shared_object_path,"lib/hidden_transport.so") == 0);
	node = SparkModelResidentDeploymentFindRank(&deployment,12u);
	assert(node != 0);
	assert(strcmp(node->transport_host,"sparkc") == 0);
	assert(strcmp(node->runtime_root,"/home/sparkc/sparkpipe_runtime/releases/dsv4-flash-ga-60d8d70770c6776ff598c94bb586a859a38244f1") == 0);
	SparkModelResidentDeploymentDestroy(&deployment);
	return(0);
}
