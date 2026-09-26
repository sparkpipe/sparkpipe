#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "sparkpipe/spark_glm52_kv_geometry.h"

#ifndef VECTOR_KV_BYTES_PER_SCALAR
#error "VECTOR_KV_BYTES_PER_SCALAR must be injected from llm_defines.h"
#endif

static uint32_t failures;

#define CHECK_CONSTANT(actual,constant,label) \
	do { \
		if ( (uint32_t)(actual) != (uint32_t)(constant) ) \
		{ \
			printf("  FAIL kv %s: filler=%u constant=%u\n", \
				label,(unsigned)(actual),(unsigned)(constant)); \
			failures++; \
		} \
	} while (0)

#define CHECK_VECTOR(actual,vector,label) \
	do { \
		if ( (uint32_t)(actual) != (uint32_t)(vector) ) \
		{ \
			printf("  FAIL %s: header=%u llm_defines=%u\n", \
				label,(unsigned)(actual),(unsigned)(vector)); \
			failures++; \
		} \
	} while (0)

int main(void)
{
	SparkKvCacheCapacityRequest request;
	memset(&request,0xA5,sizeof(request));
	SparkGlmKvFillCapacityRequest(&request);
	CHECK_CONSTANT(request.abi_version,SPARK_KV_CACHE_ABI_VERSION,"abi_version");
	CHECK_CONSTANT(request.descriptor_bytes,SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES,"descriptor_bytes");
	CHECK_CONSTANT(request.layout,SPARK_GLM_KV_LAYOUT,"layout");
	CHECK_CONSTANT(request.layer_count,SPARK_GLM_KV_LAYER_COUNT,"layer_count");
	CHECK_CONSTANT(request.head_count,0u,"head_count");
	CHECK_CONSTANT(request.query_key_head_dimension,0u,"query_key_head_dimension");
	CHECK_CONSTANT(request.value_head_dimension,0u,"value_head_dimension");
	CHECK_CONSTANT(request.compressed_dimension,SPARK_GLM_KV_COMPRESSED_DIMENSION,"compressed_dimension");
	CHECK_CONSTANT(request.position_dimension,SPARK_GLM_KV_POSITION_DIMENSION,"position_dimension");
	CHECK_CONSTANT(request.compressed_dimension + request.position_dimension,SPARK_GLM52_MODEL_CACHE_TOKEN_ELEMENTS,"compressed plus position elements equal the cached row");
	CHECK_CONSTANT(request.bytes_per_scalar,SPARK_GLM_KV_BYTES_PER_SCALAR,"bytes_per_scalar");
	CHECK_CONSTANT(request.fp8_scale_block_size,SPARK_GLM_KV_FP8_SCALE_BLOCK_SIZE,"fp8_scale_block_size");
	CHECK_CONSTANT(request.index_key_layer_count,SPARK_GLM_KV_INDEX_KEY_LAYER_COUNT,"index_key_layer_count");
	CHECK_CONSTANT(request.index_key_dimension,SPARK_GLM_KV_INDEX_KEY_DIMENSION,"index_key_dimension");
	CHECK_CONSTANT(request.index_key_bytes_per_scalar,SPARK_GLM_KV_INDEX_KEY_BYTES_PER_SCALAR,"index_key_bytes_per_scalar");
	CHECK_VECTOR(SPARK_GLM_KV_BYTES_PER_SCALAR,VECTOR_KV_BYTES_PER_SCALAR,"bytes_per_scalar tracks llm_defines");
	if ( failures == 0u )
		printf("PASS common_kv_geometry filler matches the family constants\n");
	return failures == 0u ? 0 : 1;
}
