#ifndef SPARKPIPE_SPARK_LING_KV_GEOMETRY_H
#define SPARKPIPE_SPARK_LING_KV_GEOMETRY_H

#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_ling_model.h"


#define SPARK_LING_KV_LAYOUT SPARK_KV_CACHE_LAYOUT_COMPRESSED_KEY_VALUE
#define SPARK_LING_KV_LAYER_COUNT SPARK_LING_MODEL_MLA_LAYER_COUNT
#define SPARK_LING_KV_COMPRESSED_DIMENSION \
	SPARK_LING_MODEL_MLA_LATENT_DIMENSION
#define SPARK_LING_KV_POSITION_DIMENSION \
	SPARK_LING_MODEL_MLA_QK_ROPE_HEAD_DIMENSION
#define SPARK_LING_KV_BYTES_PER_SCALAR 2u
#define SPARK_LING_KV_FP8_SCALE_BLOCK_SIZE \
	SPARK_LING_MODEL_FP8_SCALE_BLOCK
#define SPARK_LING_KV_BLOCK_TOKEN_COUNT 64u
#define SPARK_LING_KV_ARENA_KV_HEAD_COUNT 1u
#define SPARK_LING_KV_ARENA_HEAD_DIM \
	SPARK_LING_MODEL_MLA_KV_A_DIMENSION
#define SPARK_LING_KV_INDEX_KEY_LAYER_COUNT 0u
#define SPARK_LING_KV_INDEX_KEY_DIMENSION 0u
#define SPARK_LING_KV_INDEX_KEY_BYTES_PER_SCALAR 0u

static inline void SparkLingKvFillCapacityRequest(
	SparkKvCacheCapacityRequest *request)
{
	request->abi_version = SPARK_KV_CACHE_ABI_VERSION;
	request->descriptor_bytes =
		SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	request->layout = SPARK_LING_KV_LAYOUT;
	request->layer_count = SPARK_LING_KV_LAYER_COUNT;
	request->head_count = 0u;
	request->query_key_head_dimension = 0u;
	request->value_head_dimension = 0u;
	request->compressed_dimension = SPARK_LING_KV_COMPRESSED_DIMENSION;
	request->position_dimension = SPARK_LING_KV_POSITION_DIMENSION;
	request->bytes_per_scalar = SPARK_LING_KV_BYTES_PER_SCALAR;
	request->fp8_scale_block_size = SPARK_LING_KV_FP8_SCALE_BLOCK_SIZE;
	request->index_key_layer_count = SPARK_LING_KV_INDEX_KEY_LAYER_COUNT;
	request->index_key_dimension = SPARK_LING_KV_INDEX_KEY_DIMENSION;
	request->index_key_bytes_per_scalar =
		SPARK_LING_KV_INDEX_KEY_BYTES_PER_SCALAR;
}

#endif
