#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_MODEL_RESIDENT_ENDPOINT_ABI_VERSION 1u
#define SPARK_MODEL_RESIDENT_ENDPOINT_KIND_UNIX 1u
#define SPARK_MODEL_RESIDENT_ENDPOINT_KIND_TCP 2u

typedef struct SparkModelResidentEndpoint
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t kind;
	uint32_t tcp_port;
	const char *unix_socket_path;
	const char *tcp_host;
	uint32_t reserved0;
	uint32_t reserved1;
} SparkModelResidentEndpoint;

#define SPARK_MODEL_RESIDENT_ENDPOINT_BYTES \
	((uint32_t)sizeof(SparkModelResidentEndpoint))

SparkStatus SparkModelResidentEndpointValidate(
	const SparkModelResidentEndpoint *endpoint);

#ifdef __cplusplus
}
#endif
