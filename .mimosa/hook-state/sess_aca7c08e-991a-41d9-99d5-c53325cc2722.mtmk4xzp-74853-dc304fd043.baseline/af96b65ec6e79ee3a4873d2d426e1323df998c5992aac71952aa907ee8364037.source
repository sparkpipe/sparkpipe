#include "sparkpipe/spark_model_resident_endpoint.h"

SparkStatus SparkModelResidentEndpointValidate(
	const SparkModelResidentEndpoint *endpoint)
{
	uint32_t unix_endpoint,tcp_endpoint;
	if ( endpoint == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( endpoint->abi_version != SPARK_MODEL_RESIDENT_ENDPOINT_ABI_VERSION || endpoint->descriptor_bytes != SPARK_MODEL_RESIDENT_ENDPOINT_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	unix_endpoint = endpoint->kind == SPARK_MODEL_RESIDENT_ENDPOINT_KIND_UNIX;
	tcp_endpoint = endpoint->kind == SPARK_MODEL_RESIDENT_ENDPOINT_KIND_TCP;
	if ( unix_endpoint != 0u && (endpoint->unix_socket_path == 0 || endpoint->unix_socket_path[0] == '\0' || endpoint->tcp_host != 0 || endpoint->tcp_port != 0u) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( tcp_endpoint != 0u && (endpoint->tcp_host == 0 || endpoint->tcp_host[0] == '\0' || endpoint->tcp_port == 0u || endpoint->tcp_port > UINT16_MAX || endpoint->unix_socket_path != 0) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (unix_endpoint | tcp_endpoint) == 0u || endpoint->reserved0 != 0u || endpoint->reserved1 != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}
