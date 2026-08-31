#include "sparkpipe/spark_model_resident_deployment.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "spark_filesystem.h"
#include "sparkpipe/spark_json.h"

static const char *const SparkModelResidentDeploymentRootMembers[] =
{
	"schema_version","coordinator_rank_index","adapter","driver","transport",
	"runtime_limits","nodes","tokenizer","weightd"
};
/* The root's required members. "tokenizer" is the one OPTIONAL member: the
 * sidecar asset reference. Deployment files without it stay valid (schema
 * version 2, additive), and consumers that never touch the sidecar see no
 * change. */
#define SPARK_MODEL_RESIDENT_DEPLOYMENT_ROOT_REQUIRED_MEMBER_COUNT 7u
static const char *const SparkModelResidentDeploymentTokenizerMembers[] =
{
	"path"
};
static const char *const SparkModelResidentDeploymentAdapterMembers[] =
{
	"shared_object_path"
};
static const char *const SparkModelResidentDeploymentDriverMembers[] =
{
	"shared_object_path","program_name"
};
static const char *const SparkModelResidentDeploymentTransportMembers[] =
{
	"shared_object_path","mode","control_port_base"
};
static const char *const SparkModelResidentDeploymentRuntimeMembers[] =
{
	"max_inflight_submissions","max_active_sequences","max_input_rows",
	"resident_sequence_capacity","kv_logical_page_capacity",
	"kv_physical_page_capacity"
};
static const char *const SparkModelResidentDeploymentNodeMembers[] =
{
	"rank_index","stage_index","runtime_root","node_target","transport_host",
	"adapter_configuration_path","kv_backing_directory",
	"kv_backing_maximum_bytes","control_endpoint"
};
static const char *const SparkModelResidentDeploymentUnixMembers[] =
{
	"kind","path"
};
static const char *const SparkModelResidentDeploymentTcpMembers[] =
{
	"kind","host","port"
};

static int32_t SparkModelResidentDeploymentMember(
	const SparkJsonDocument *document,
	int32_t object,
	const char *name)
{
	return(SparkJsonFindObjectMember(document,object,name));
}

static SparkStatus SparkModelResidentDeploymentObject(
	const SparkJsonDocument *document,
	int32_t parent,
	const char *name,
	int32_t *object)
{
	*object = SparkModelResidentDeploymentMember(document,parent,name);
	return(*object >= 0 && SparkJsonTokenIsType(document,*object,SPARK_JSON_TOKEN_OBJECT) ? SPARK_STATUS_OK : SPARK_STATUS_SCHEMA_ERROR);
}

static SparkStatus SparkModelResidentDeploymentString(
	const SparkJsonDocument *document,
	int32_t object,
	const char *name,
	char **value)
{
	int32_t token;
	token = SparkModelResidentDeploymentMember(document,object,name);
	return(token < 0 ? SPARK_STATUS_SCHEMA_ERROR : SparkJsonCopyString(document,token,value));
}

static SparkStatus SparkModelResidentDeploymentUnsigned(
	const SparkJsonDocument *document,
	int32_t object,
	const char *name,
	uint32_t *value)
{
	int32_t token;
	token = SparkModelResidentDeploymentMember(document,object,name);
	return(token < 0 ? SPARK_STATUS_SCHEMA_ERROR : SparkJsonGetUInt32(document,token,value));
}

static SparkStatus SparkModelResidentDeploymentUnsigned64(
	const SparkJsonDocument *document,
	int32_t object,
	const char *name,
	uint64_t *value)
{
	int32_t token;
	token = SparkModelResidentDeploymentMember(document,object,name);
	return(token < 0 ? SPARK_STATUS_SCHEMA_ERROR :
		SparkJsonGetUInt64(document,token,value));
}

static SparkStatus SparkModelResidentDeploymentNullableString(
	const SparkJsonDocument *document,
	int32_t object,
	const char *name,
	char **value)
{
	char *raw;
	uint32_t raw_bytes;
	int32_t token;
	SparkStatus status;
	if ( value == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*value = 0;
	token = SparkModelResidentDeploymentMember(document,object,name);
	if ( token < 0 )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( SparkJsonTokenIsType(document,token,SPARK_JSON_TOKEN_STRING) )
		return(SparkJsonCopyString(document,token,value));
	status = SparkJsonCopyRawValue(document,token,&raw,&raw_bytes);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = raw_bytes == 4u && strcmp(raw,"null") == 0 ?
		SPARK_STATUS_OK : SPARK_STATUS_SCHEMA_ERROR;
	free(raw);
	return(status);
}

static SparkStatus SparkModelResidentDeploymentParseEndpoint(
	const SparkJsonDocument *document,
	int32_t object,
	SparkModelResidentEndpoint *endpoint)
{
	char *kind,*value;
	SparkStatus status;
	memset(endpoint,0,sizeof(*endpoint));
	endpoint->abi_version = SPARK_MODEL_RESIDENT_ENDPOINT_ABI_VERSION;
	endpoint->descriptor_bytes = SPARK_MODEL_RESIDENT_ENDPOINT_BYTES;
	kind = 0;
	value = 0;
	status = SparkModelResidentDeploymentString(document,object,"kind",&kind);
	if ( status == SPARK_STATUS_OK && strcmp(kind,"unix") == 0 )
	{
		endpoint->kind = SPARK_MODEL_RESIDENT_ENDPOINT_KIND_UNIX;
		status = SparkJsonValidateObjectMembersExact(document,object,SparkModelResidentDeploymentUnixMembers,2u);
		if ( status == SPARK_STATUS_OK )
			status = SparkModelResidentDeploymentString(document,object,"path",&value);
		endpoint->unix_socket_path = value;
	}
	else if ( status == SPARK_STATUS_OK && strcmp(kind,"tcp") == 0 )
	{
		endpoint->kind = SPARK_MODEL_RESIDENT_ENDPOINT_KIND_TCP;
		status = SparkJsonValidateObjectMembersExact(document,object,SparkModelResidentDeploymentTcpMembers,3u);
		if ( status == SPARK_STATUS_OK )
			status = SparkModelResidentDeploymentString(document,object,"host",&value);
		endpoint->tcp_host = value;
		if ( status == SPARK_STATUS_OK )
			status = SparkModelResidentDeploymentUnsigned(document,object,"port",&endpoint->tcp_port);
	}
	else if ( status == SPARK_STATUS_OK )
		status = SPARK_STATUS_SCHEMA_ERROR;
	free(kind);
	return(status == SPARK_STATUS_OK ? SparkModelResidentEndpointValidate(endpoint) : status);
}

static SparkStatus SparkModelResidentDeploymentParseNode(
	const SparkJsonDocument *document,
	int32_t object,
	SparkModelResidentDeploymentNode *node)
{
	int32_t endpoint;
	SparkStatus status;
	if ( !SparkJsonTokenIsType(document,object,SPARK_JSON_TOKEN_OBJECT) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	status = SparkJsonValidateObjectMembersExact(document,object,SparkModelResidentDeploymentNodeMembers,9u);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentUnsigned(document,object,"rank_index",&node->rank_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentUnsigned(document,object,"stage_index",&node->stage_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentString(document,object,"runtime_root",&node->runtime_root);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentString(document,object,"node_target",&node->node_target);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentString(document,object,"transport_host",&node->transport_host);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentString(document,object,"adapter_configuration_path",&node->adapter_configuration_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentNullableString(document,object,
			"kv_backing_directory",&node->kv_backing_directory);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentUnsigned64(document,object,
			"kv_backing_maximum_bytes",&node->kv_backing_maximum_bytes);
	endpoint = status == SPARK_STATUS_OK ? SparkModelResidentDeploymentMember(document,object,"control_endpoint") : -1;
	if ( status == SPARK_STATUS_OK && !SparkJsonTokenIsType(document,endpoint,SPARK_JSON_TOKEN_OBJECT) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	return(status == SPARK_STATUS_OK ? SparkModelResidentDeploymentParseEndpoint(document,endpoint,&node->control_endpoint) : status);
}

static SparkStatus SparkModelResidentDeploymentParseRuntime(
	const SparkJsonDocument *document,
	int32_t object,
	SparkModelServingRuntimeLimits *limits)
{
	SparkStatus status;
	memset(limits,0,sizeof(*limits));
	limits->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	limits->descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	status = SparkJsonValidateObjectMembersExact(document,object,SparkModelResidentDeploymentRuntimeMembers,6u);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentUnsigned(document,object,"max_inflight_submissions",&limits->max_inflight_submission_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentUnsigned(document,object,"max_active_sequences",&limits->max_active_sequence_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentUnsigned(document,object,"max_input_rows",&limits->max_input_row_count);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentUnsigned(document,object,"resident_sequence_capacity",&limits->resident_sequence_capacity);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentUnsigned(document,object,
			"kv_logical_page_capacity",&limits->kv_logical_page_capacity);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentUnsigned(document,object,
			"kv_physical_page_capacity",&limits->kv_physical_page_capacity);
	return(status);
}

static SparkStatus SparkModelResidentDeploymentParseAdapter(
	const SparkJsonDocument *document,
	int32_t root,
	SparkModelResidentDeployment *deployment)
{
	int32_t object;
	SparkStatus status;
	status = SparkModelResidentDeploymentObject(document,root,"adapter",&object);
	if ( status == SPARK_STATUS_OK )
		status = SparkJsonValidateObjectMembersExact(document,object,SparkModelResidentDeploymentAdapterMembers,1u);
	return(status == SPARK_STATUS_OK ? SparkModelResidentDeploymentString(document,object,"shared_object_path",&deployment->adapter_shared_object_path) : status);
}

static SparkStatus SparkModelResidentDeploymentParseDriver(
	const SparkJsonDocument *document,
	int32_t root,
	SparkModelResidentDeployment *deployment)
{
	int32_t object;
	SparkStatus status;
	status = SparkModelResidentDeploymentObject(document,root,"driver",&object);
	if ( status == SPARK_STATUS_OK )
		status = SparkJsonValidateObjectMembersExact(document,object,SparkModelResidentDeploymentDriverMembers,2u);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentString(document,object,"shared_object_path",&deployment->driver_shared_object_path);
	return(status == SPARK_STATUS_OK ? SparkModelResidentDeploymentString(document,object,"program_name",&deployment->driver_program_name) : status);
}

static SparkStatus SparkModelResidentDeploymentParseTransport(
	const SparkJsonDocument *document,
	int32_t root,
	SparkModelResidentDeployment *deployment)
{
	int32_t object;
	SparkStatus status;
	status = SparkModelResidentDeploymentObject(document,root,"transport",&object);
	if ( status == SPARK_STATUS_OK )
		status = SparkJsonValidateObjectMembersExact(document,object,SparkModelResidentDeploymentTransportMembers,3u);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentString(document,object,"shared_object_path",&deployment->transport_shared_object_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentString(document,object,"mode",&deployment->transport_mode);
	return(status == SPARK_STATUS_OK ? SparkModelResidentDeploymentUnsigned(document,object,"control_port_base",&deployment->transport_control_port_base) : status);
}

/* Root members are the required seven plus the optional "tokenizer". The
 * shared exact-members validator pins the count, so the optional member gets
 * its own check: every key must be a known root member (no unknowns, no
 * duplicates) and every required member must be present exactly once. */
static int32_t SparkModelResidentDeploymentNextDirectChild(
	const SparkJsonDocument *document,
	int32_t parent,
	int32_t previous)
{
	int32_t token_index;
	token_index = previous + 1;
	while ( token_index >= 0 && (uint32_t)token_index < document->token_count )
	{
		if ( document->tokens[token_index].parent == parent )
			return(token_index);
		token_index++;
	}
	return(-1);
}

static SparkStatus SparkModelResidentDeploymentValidateRootMembers(
	const SparkJsonDocument *document,
	int32_t root)
{
	uint8_t seen[SPARK_MODEL_RESIDENT_DEPLOYMENT_ROOT_REQUIRED_MEMBER_COUNT];
	uint32_t required_seen_count;
	int32_t key_token_index;
	uint32_t child_index;
	required_seen_count = 0u;
	memset(seen,0,sizeof(seen));
	/* Direct children alternate key,value: keys sit at even offsets. */
	child_index = 0u;
	for ( key_token_index = SparkModelResidentDeploymentNextDirectChild(document,root,root);
		key_token_index >= 0;
		key_token_index = SparkModelResidentDeploymentNextDirectChild(document,root,key_token_index) )
	{
		uint32_t member_index;
		uint32_t match_count;
		if ( (child_index & 1u) != 0u )
		{
			/* Odd offset: this child is the previous key's value. */
			child_index++;
			continue;
		}
		if ( !SparkJsonTokenIsType(document,key_token_index,SPARK_JSON_TOKEN_STRING) )
			return(SPARK_STATUS_SCHEMA_ERROR);
		match_count = 0u;
		for ( member_index = 0u;
			member_index < sizeof(SparkModelResidentDeploymentRootMembers) /
				sizeof(SparkModelResidentDeploymentRootMembers[0]);
			member_index++ )
		{
			if ( SparkJsonStringEquals(document,key_token_index,SparkModelResidentDeploymentRootMembers[member_index]) )
			{
				match_count++;
				if ( member_index < SPARK_MODEL_RESIDENT_DEPLOYMENT_ROOT_REQUIRED_MEMBER_COUNT )
				{
					if ( seen[member_index] != 0u )
						return(SPARK_STATUS_SCHEMA_ERROR);
					seen[member_index] = 1u;
					required_seen_count++;
				}
				else if ( match_count > 1u )
					return(SPARK_STATUS_SCHEMA_ERROR);
			}
		}
		if ( match_count != 1u )
			return(SPARK_STATUS_SCHEMA_ERROR);
		child_index++;
	}
	return(required_seen_count == SPARK_MODEL_RESIDENT_DEPLOYMENT_ROOT_REQUIRED_MEMBER_COUNT ?
		SPARK_STATUS_OK : SPARK_STATUS_SCHEMA_ERROR);
}

static SparkStatus SparkModelResidentDeploymentParseTokenizer(
	const SparkJsonDocument *document,
	int32_t root,
	SparkModelResidentDeployment *deployment)
{
	int32_t object;
	SparkStatus status;
	object = SparkModelResidentDeploymentMember(document,root,"tokenizer");
	if ( object < 0 )
	{
		/* Absent: the deployment has no sidecar; token-id serving only. */
		deployment->tokenizer_asset_path = 0;
		return(SPARK_STATUS_OK);
	}
	if ( !SparkJsonTokenIsType(document,object,SPARK_JSON_TOKEN_OBJECT) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	status = SparkJsonValidateObjectMembersExact(document,object,SparkModelResidentDeploymentTokenizerMembers,1u);
	return(status == SPARK_STATUS_OK ? SparkModelResidentDeploymentString(document,object,"path",&deployment->tokenizer_asset_path) : status);
}

static SparkStatus SparkModelResidentDeploymentParseWeightd(
	const SparkJsonDocument *document,
	int32_t root,
	SparkModelResidentDeployment *deployment)
{
	static const char *const members[] = { "socket_path" };
	int32_t object;
	SparkStatus status;
	object = SparkModelResidentDeploymentMember(document,root,"weightd");
	if ( object < 0 )
	{
		/* Absent: residency off - the seam's direct-load path. */
		deployment->weightd_socket_path = 0;
		return(SPARK_STATUS_OK);
	}
	if ( !SparkJsonTokenIsType(document,object,SPARK_JSON_TOKEN_OBJECT) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	status = SparkJsonValidateObjectMembersExact(document,object,members,1u);
	return(status == SPARK_STATUS_OK ? SparkModelResidentDeploymentString(document,object,"socket_path",&deployment->weightd_socket_path) : status);
}

static SparkStatus SparkModelResidentDeploymentParseNodes(
	const SparkJsonDocument *document,
	int32_t root,
	SparkModelResidentDeployment *deployment)
{
	int32_t array,object;
	SparkStatus status;
	uint32_t index;
	array = SparkModelResidentDeploymentMember(document,root,"nodes");
	if ( !SparkJsonTokenIsType(document,array,SPARK_JSON_TOKEN_ARRAY) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	deployment->node_count = SparkJsonGetArrayElementCount(document,array);
	if ( deployment->node_count == 0u || deployment->node_count > SPARK_MODEL_RESIDENT_DEPLOYMENT_MAX_NODE_COUNT )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SPARK_STATUS_OK;
	for (index=0u; status==SPARK_STATUS_OK && index<deployment->node_count; index++)
	{
		object = SparkJsonGetArrayElement(document,array,index);
		status = SparkModelResidentDeploymentParseNode(document,object,&deployment->nodes[index]);
	}
	return(status);
}

static uint32_t SparkModelResidentDeploymentHasText(const char *text)
{
	return(text != 0 && text[0] != '\0' ? 1u : 0u);
}

static uint32_t SparkModelResidentDeploymentEndpointsEqual(
	const SparkModelResidentEndpoint *left,
	const SparkModelResidentEndpoint *right)
{
	if ( left->kind != right->kind )
		return(0u);
	if ( left->kind == SPARK_MODEL_RESIDENT_ENDPOINT_KIND_UNIX )
		return(strcmp(left->unix_socket_path,right->unix_socket_path) == 0 ? 1u : 0u);
	return(left->tcp_port == right->tcp_port && strcmp(left->tcp_host,right->tcp_host) == 0 ? 1u : 0u);
}

static SparkStatus SparkModelResidentDeploymentValidateStructure(
	const SparkModelResidentDeployment *deployment)
{
	uint8_t ranks[SPARK_MODEL_RESIDENT_DEPLOYMENT_MAX_NODE_COUNT];
	uint8_t stages[SPARK_MODEL_RESIDENT_DEPLOYMENT_MAX_NODE_COUNT];
	uint32_t index,previous;
	if ( deployment == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( deployment->abi_version != SPARK_MODEL_RESIDENT_DEPLOYMENT_ABI_VERSION || deployment->descriptor_bytes != SPARK_MODEL_RESIDENT_DEPLOYMENT_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( deployment->schema_version != SPARK_MODEL_RESIDENT_DEPLOYMENT_SCHEMA_VERSION || deployment->node_count == 0u || deployment->node_count > SPARK_MODEL_RESIDENT_DEPLOYMENT_MAX_NODE_COUNT || deployment->coordinator_rank_index >= deployment->node_count )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( !SparkPathIsNormalized(deployment->adapter_shared_object_path,false) || !SparkPathIsNormalized(deployment->driver_shared_object_path,false) || SparkModelResidentDeploymentHasText(deployment->driver_program_name) == 0u || !SparkPathIsNormalized(deployment->transport_shared_object_path,false) || SparkModelResidentDeploymentHasText(deployment->transport_mode) == 0u )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( deployment->tokenizer_asset_path != 0 && !SparkPathIsNormalized(deployment->tokenizer_asset_path,false) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	memset(ranks,0,sizeof(ranks));
	memset(stages,0,sizeof(stages));
	if ( deployment->runtime_limits.max_inflight_submission_count == 0u || deployment->runtime_limits.max_inflight_submission_count > SPARK_MODEL_SERVING_ADAPTER_MAX_INFLIGHT_SUBMISSION_COUNT || deployment->runtime_limits.max_active_sequence_count == 0u || deployment->runtime_limits.max_active_sequence_count > SPARK_MODEL_SERVING_ADAPTER_MAX_ACTIVE_SEQUENCE_COUNT || deployment->runtime_limits.max_input_row_count < deployment->runtime_limits.max_active_sequence_count || deployment->runtime_limits.max_input_row_count > SPARK_MODEL_SERVING_ADAPTER_MAX_INPUT_ROW_COUNT || deployment->runtime_limits.resident_sequence_capacity < deployment->runtime_limits.max_active_sequence_count || deployment->runtime_limits.resident_sequence_capacity > SPARK_MODEL_SERVING_ADAPTER_MAX_RESIDENT_SEQUENCE_COUNT )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (deployment->runtime_limits.kv_logical_page_capacity == 0u) !=
		(deployment->runtime_limits.kv_physical_page_capacity == 0u) ||
		(deployment->runtime_limits.kv_physical_page_capacity != 0u &&
		 (deployment->runtime_limits.kv_physical_page_capacity <
		  deployment->runtime_limits.max_active_sequence_count ||
		  deployment->runtime_limits.kv_logical_page_capacity <
		  deployment->runtime_limits.resident_sequence_capacity ||
		  deployment->runtime_limits.kv_physical_page_capacity >
		  deployment->runtime_limits.kv_logical_page_capacity)) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( strcmp(deployment->transport_mode,"host-rdma") != 0 && strcmp(deployment->transport_mode,"gpudirect-rdma") != 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( deployment->transport_control_port_base == 0u || deployment->transport_control_port_base > UINT16_MAX - (deployment->node_count - 1u) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (index=0u; index<deployment->node_count; index++)
	{
		const SparkModelResidentDeploymentNode *node;
		node = &deployment->nodes[index];
		if ( node->rank_index >= deployment->node_count || node->stage_index >= deployment->node_count || ranks[node->rank_index] != 0u || stages[node->stage_index] != 0u )
			return(SPARK_STATUS_SCHEMA_ERROR);
		if ( !SparkPathIsNormalized(node->runtime_root,true) || SparkModelResidentDeploymentHasText(node->node_target) == 0u || SparkModelResidentDeploymentHasText(node->transport_host) == 0u || !SparkPathIsNormalized(node->adapter_configuration_path,false) || (node->kv_backing_directory == 0 && node->kv_backing_maximum_bytes != 0u) || (node->kv_backing_directory != 0 && !SparkPathIsNormalized(node->kv_backing_directory,true)) || SparkModelResidentEndpointValidate(&node->control_endpoint) != SPARK_STATUS_OK )
			return(SPARK_STATUS_SCHEMA_ERROR);
		if ( strcmp(node->transport_host,"0.0.0.0") == 0 || strcmp(node->transport_host,"::") == 0 || strcmp(node->transport_host,"*") == 0 )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		if ( node->control_endpoint.kind == SPARK_MODEL_RESIDENT_ENDPOINT_KIND_TCP && (strcmp(node->control_endpoint.tcp_host,"0.0.0.0") == 0 || strcmp(node->control_endpoint.tcp_host,"::") == 0 || strcmp(node->control_endpoint.tcp_host,"*") == 0) )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		for (previous=0u; previous<index; previous++)
			if ( strcmp(node->transport_host,deployment->nodes[previous].transport_host) == 0 || SparkModelResidentDeploymentEndpointsEqual(&node->control_endpoint,&deployment->nodes[previous].control_endpoint) != 0u )
				return(SPARK_STATUS_SCHEMA_ERROR);
		ranks[node->rank_index] = 1u;
		stages[node->stage_index] = 1u;
	}
	return(SPARK_STATUS_OK);
}

void SparkModelResidentDeploymentReset(
	SparkModelResidentDeployment *deployment)
{
	if ( deployment == 0 )
		return;
	memset(deployment,0,sizeof(*deployment));
	deployment->abi_version = SPARK_MODEL_RESIDENT_DEPLOYMENT_ABI_VERSION;
	deployment->descriptor_bytes = SPARK_MODEL_RESIDENT_DEPLOYMENT_BYTES;
}

void SparkModelResidentDeploymentDestroy(
	SparkModelResidentDeployment *deployment)
{
	uint32_t index;
	if ( deployment == 0 )
		return;
	free(deployment->adapter_shared_object_path);
	free(deployment->driver_shared_object_path);
	free(deployment->driver_program_name);
	free(deployment->transport_shared_object_path);
	free(deployment->transport_mode);
	free(deployment->tokenizer_asset_path);
	free(deployment->weightd_socket_path);
	for (index=0u; index<deployment->node_count; index++)
	{
		free(deployment->nodes[index].runtime_root);
		free(deployment->nodes[index].node_target);
		free(deployment->nodes[index].transport_host);
		free(deployment->nodes[index].adapter_configuration_path);
		free(deployment->nodes[index].kv_backing_directory);
		free((void *)deployment->nodes[index].control_endpoint.unix_socket_path);
		free((void *)deployment->nodes[index].control_endpoint.tcp_host);
	}
	SparkModelResidentDeploymentReset(deployment);
}

SparkStatus SparkModelResidentDeploymentLoad(
	const char *path,
	SparkModelResidentDeployment *deployment)
{
	SparkJsonDocument document;
	int32_t root,runtime_object;
	SparkStatus status;
	if ( path == 0 || path[0] == '\0' || deployment == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	SparkModelResidentDeploymentReset(deployment);
	SparkJsonDocumentReset(&document);
	status = SparkJsonLoadFile(path,&document);
	root = status == SPARK_STATUS_OK ? SparkJsonGetRootToken(&document) : -1;
	if ( status == SPARK_STATUS_OK && !SparkJsonTokenIsType(&document,root,SPARK_JSON_TOKEN_OBJECT) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentValidateRootMembers(&document,root);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentUnsigned(&document,root,"schema_version",&deployment->schema_version);
	if ( status == SPARK_STATUS_OK && deployment->schema_version != SPARK_MODEL_RESIDENT_DEPLOYMENT_SCHEMA_VERSION )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentUnsigned(&document,root,"coordinator_rank_index",&deployment->coordinator_rank_index);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentParseAdapter(&document,root,deployment);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentParseDriver(&document,root,deployment);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentParseTransport(&document,root,deployment);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentParseTokenizer(&document,root,deployment);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentParseWeightd(&document,root,deployment);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentObject(&document,root,"runtime_limits",&runtime_object);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentParseRuntime(&document,runtime_object,&deployment->runtime_limits);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentParseNodes(&document,root,deployment);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelResidentDeploymentValidateStructure(deployment);
	SparkJsonDocumentDestroy(&document);
	if ( status != SPARK_STATUS_OK )
		SparkModelResidentDeploymentDestroy(deployment);
	return(status);
}

SparkStatus SparkModelResidentDeploymentValidateForAdapter(
	const SparkModelResidentDeployment *deployment,
	const SparkModelServingAdapterDescriptor *descriptor)
{
	SparkStatus status;
	if ( deployment == 0 || descriptor == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkModelResidentDeploymentValidateStructure(deployment);
	if ( status == SPARK_STATUS_OK )
		status = SparkModelServingAdapterValidateRuntimeLimits(descriptor,&deployment->runtime_limits);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( deployment->node_count != descriptor->stage_count || strcmp(deployment->driver_program_name,descriptor->driver_program_name) != 0 )
		return(SPARK_STATUS_TARGET_MISMATCH);
	return(SPARK_STATUS_OK);
}

const SparkModelResidentDeploymentNode *SparkModelResidentDeploymentFindRank(
	const SparkModelResidentDeployment *deployment,
	uint32_t rank_index)
{
	uint32_t index;
	if ( deployment == 0 || deployment->node_count > SPARK_MODEL_RESIDENT_DEPLOYMENT_MAX_NODE_COUNT )
		return(0);
	for (index=0u; index<deployment->node_count; index++)
		if ( deployment->nodes[index].rank_index == rank_index )
			return(&deployment->nodes[index]);
	return(0);
}

const SparkModelResidentDeploymentNode *SparkModelResidentDeploymentFindStage(
	const SparkModelResidentDeployment *deployment,
	uint32_t stage_index)
{
	uint32_t index;
	if ( deployment == 0 || deployment->node_count > SPARK_MODEL_RESIDENT_DEPLOYMENT_MAX_NODE_COUNT )
		return(0);
	for (index=0u; index<deployment->node_count; index++)
		if ( deployment->nodes[index].stage_index == stage_index )
			return(&deployment->nodes[index]);
	return(0);
}

SparkStatus SparkModelResidentDeploymentResolvePath(
	const SparkModelResidentDeploymentNode *node,
	const char *relative_path,
	char *resolved_path,
	uint32_t resolved_path_bytes)
{
	if ( node == 0 || resolved_path == 0 || resolved_path_bytes == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkResolveRuntimePath(node->runtime_root,relative_path,resolved_path,resolved_path_bytes));
}
