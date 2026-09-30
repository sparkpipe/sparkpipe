#pragma once

#include <stdint.h>

#include "sparkpipe/spark_model_resident_endpoint.h"
#include "sparkpipe/spark_model_serving_adapter.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_MODEL_RESIDENT_DEPLOYMENT_ABI_VERSION 3u
#define SPARK_MODEL_RESIDENT_DEPLOYMENT_MAX_EOS_TOKEN_COUNT 16u
#define SPARK_MODEL_RESIDENT_DEPLOYMENT_SCHEMA_VERSION 2u
#define SPARK_MODEL_RESIDENT_DEPLOYMENT_MAX_NODE_COUNT \
	SPARK_MODEL_SERVING_ADAPTER_MAX_STAGE_COUNT
#define SPARK_MODEL_RESIDENT_DEPLOYMENT_PATH_BYTES 4096u
#define SPARK_MODEL_RESIDENT_CHAT_TEMPLATE_MAX_STOP_MARKERS 8u

typedef struct SparkModelResidentChatTemplate
{
	uint32_t declared;
	uint32_t stop_marker_count;
	char *prefix;
	char *thinking_prefix;
	char *system;
	char *system_thinking;
	char *user;
	char *observation;
	char *assistant;
	char *assistant_thinking;
	char *assistant_suffix;
	char *turn_suffix;
	char *generation;
	char *generation_thinking;
	char *stop_markers[SPARK_MODEL_RESIDENT_CHAT_TEMPLATE_MAX_STOP_MARKERS];
	uint32_t control_marker_count;
	char *control_markers[SPARK_MODEL_RESIDENT_CHAT_TEMPLATE_MAX_STOP_MARKERS];
} SparkModelResidentChatTemplate;

typedef struct SparkModelResidentDeploymentNode
{
	uint32_t rank_index;
	uint32_t stage_index;
	char *runtime_root;
	char *node_target;
	char *transport_host;
	char *adapter_configuration_path;
	char *kv_backing_directory;
	uint64_t kv_backing_maximum_bytes;
	SparkModelResidentEndpoint control_endpoint;
} SparkModelResidentDeploymentNode;

typedef struct SparkModelResidentDeployment
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t schema_version;
	uint32_t node_count;
	uint32_t coordinator_rank_index;
	uint32_t reserved0;
	SparkModelServingRuntimeLimits runtime_limits;
	char *adapter_shared_object_path;
	char *driver_shared_object_path;
	char *driver_program_name;
	char *transport_shared_object_path;
	char *transport_mode;
	uint32_t transport_control_port_base;
	char *tokenizer_asset_path;
	uint32_t tokenizer_vocabulary_size;
	char *tokenizer_asset_sha256;
	char *weightd_socket_path;
	uint32_t eos_token_count;
	uint32_t eos_token_ids[SPARK_MODEL_RESIDENT_DEPLOYMENT_MAX_EOS_TOKEN_COUNT];
	uint32_t max_sequence_positions;
	uint32_t prefix_reuse_disabled;
	uint32_t reserved[1];
	SparkModelResidentChatTemplate chat_template;
	SparkModelResidentDeploymentNode nodes[
		SPARK_MODEL_RESIDENT_DEPLOYMENT_MAX_NODE_COUNT];
} SparkModelResidentDeployment;

#define SPARK_MODEL_RESIDENT_DEPLOYMENT_BYTES \
	((uint32_t)sizeof(SparkModelResidentDeployment))

void SparkModelResidentDeploymentReset(
	SparkModelResidentDeployment *deployment);
void SparkModelResidentDeploymentDestroy(
	SparkModelResidentDeployment *deployment);
SparkStatus SparkModelResidentDeploymentLoad(
	const char *path,
	SparkModelResidentDeployment *deployment);
SparkStatus SparkModelResidentDeploymentValidateForAdapter(
	const SparkModelResidentDeployment *deployment,
	const SparkModelServingAdapterDescriptor *descriptor);
const SparkModelResidentDeploymentNode *SparkModelResidentDeploymentFindRank(
	const SparkModelResidentDeployment *deployment,
	uint32_t rank_index);
const SparkModelResidentDeploymentNode *SparkModelResidentDeploymentFindStage(
	const SparkModelResidentDeployment *deployment,
	uint32_t stage_index);
SparkStatus SparkModelResidentDeploymentResolvePath(
	const SparkModelResidentDeploymentNode *node,
	const char *relative_path,
	char *resolved_path,
	uint32_t resolved_path_bytes);

#ifdef __cplusplus
}
#endif
