#pragma once

#include "sparkpipe/spark_kv_cache.h"

_Static_assert((SPARK_GLM_KV_BLOCK_TOKEN_COUNT &
	(SPARK_GLM_KV_BLOCK_TOKEN_COUNT - 1u)) == 0u &&
	SPARK_GLM_KV_BLOCK_TOKEN_COUNT != 0u,
	"SPARK_GLM_KV_BLOCK_TOKEN_COUNT must be a power of two for page math");
_Static_assert(SPARK_GLM_KV_BYTES_PER_SCALAR != 0u,
	"SPARK_GLM_KV_BYTES_PER_SCALAR must be nonzero");
_Static_assert(SPARK_GLM_KV_COMPRESSED_DIMENSION != 0u,
	"SPARK_GLM_KV_COMPRESSED_DIMENSION must be nonzero");
_Static_assert(SPARK_GLM_KV_ARENA_HEAD_DIMENSION ==
	SPARK_GLM_KV_COMPRESSED_DIMENSION,
	"SPARK_GLM_KV_ARENA_HEAD_DIMENSION must match the compressed row");
_Static_assert((SPARK_GLM_KV_INDEX_KEY_LAYER_COUNT == 0u &&
	SPARK_GLM_KV_INDEX_KEY_DIMENSION == 0u &&
	SPARK_GLM_KV_INDEX_KEY_BYTES_PER_SCALAR == 0u) ||
	(SPARK_GLM_KV_INDEX_KEY_LAYER_COUNT != 0u &&
	SPARK_GLM_KV_INDEX_KEY_DIMENSION != 0u &&
	SPARK_GLM_KV_INDEX_KEY_BYTES_PER_SCALAR != 0u),
	"the SPARK_GLM_KV_INDEX_KEY triple must be all zero or all nonzero");

static inline void SparkGlmKvFillCapacityRequest(
	SparkKvCacheCapacityRequest *request)
{
	request->abi_version = SPARK_KV_CACHE_ABI_VERSION;
	request->descriptor_bytes =
		SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	request->layout = SPARK_GLM_KV_LAYOUT;
	request->layer_count = SPARK_GLM_KV_LAYER_COUNT;
	request->head_count = 0u;
	request->query_key_head_dimension = 0u;
	request->value_head_dimension = 0u;
	request->compressed_dimension = SPARK_GLM_KV_COMPRESSED_DIMENSION;
	request->position_dimension = SPARK_GLM_KV_POSITION_DIMENSION;
	request->bytes_per_scalar = SPARK_GLM_KV_BYTES_PER_SCALAR;
	request->fp8_scale_block_size = SPARK_GLM_KV_FP8_SCALE_BLOCK_SIZE;
	request->index_key_layer_count = SPARK_GLM_KV_INDEX_KEY_LAYER_COUNT;
	request->index_key_dimension = SPARK_GLM_KV_INDEX_KEY_DIMENSION;
	request->index_key_bytes_per_scalar =
		SPARK_GLM_KV_INDEX_KEY_BYTES_PER_SCALAR;
}
