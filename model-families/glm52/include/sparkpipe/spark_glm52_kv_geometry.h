#ifndef SPARKPIPE_SPARK_GLM52_KV_GEOMETRY_H
#define SPARKPIPE_SPARK_GLM52_KV_GEOMETRY_H

#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_glm52_model.h"

/* GLM-5.2 cache geometry for the common kv machinery, in the machinery's own
 * request vocabulary (mirrors spark_k3_kv_geometry.h). GLM52 caches the
 * compressed KV_A latent (latent 512 + rope 64 = 576) per token, BF16 (2
 * bytes/scalar); the FP8 scale block is 128. For the compressed layout the
 * capacity request fills compressed_dimension + position_dimension and leaves
 * head_count/query_key/value zero (the K3 MLA pattern). The DSA indexer is a
 * separate cache the resident stage still owns, so index-key fields stay zero. */

#define SPARK_GLM52_KV_LAYOUT SPARK_KV_CACHE_LAYOUT_COMPRESSED_KEY_VALUE
#define SPARK_GLM52_KV_LAYER_COUNT SPARK_GLM52_MODEL_LAYER_COUNT
#define SPARK_GLM52_KV_COMPRESSED_DIMENSION SPARK_GLM52_MODEL_KV_A_DIMENSION
#define SPARK_GLM52_KV_POSITION_DIMENSION SPARK_GLM52_MODEL_ROPE_DIMENSION
#define SPARK_GLM52_KV_BYTES_PER_SCALAR 2u
#define SPARK_GLM52_KV_FP8_SCALE_BLOCK_SIZE SPARK_GLM52_MODEL_FP8_SCALE_BLOCK
#define SPARK_GLM52_KV_BLOCK_TOKEN_COUNT 64u
/* Arena block geometry: one compressed KV_A row per token, 78 layers per block. */
#define SPARK_GLM52_KV_ARENA_KV_HEAD_COUNT 1u
#define SPARK_GLM52_KV_ARENA_HEAD_DIM SPARK_GLM52_MODEL_KV_A_DIMENSION
#define SPARK_GLM52_KV_INDEX_KEY_LAYER_COUNT 0u
#define SPARK_GLM52_KV_INDEX_KEY_DIMENSION 0u
#define SPARK_GLM52_KV_INDEX_KEY_BYTES_PER_SCALAR 0u

static inline void SparkGlm52KvFillCapacityRequest(
	SparkKvCacheCapacityRequest *request)
{
	request->abi_version = SPARK_KV_CACHE_ABI_VERSION;
	request->descriptor_bytes =
		SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	request->layout = SPARK_GLM52_KV_LAYOUT;
	request->layer_count = SPARK_GLM52_KV_LAYER_COUNT;
	request->head_count = 0u;
	request->query_key_head_dimension = 0u;
	request->value_head_dimension = 0u;
	request->compressed_dimension = SPARK_GLM52_KV_COMPRESSED_DIMENSION;
	request->position_dimension = SPARK_GLM52_KV_POSITION_DIMENSION;
	request->bytes_per_scalar = SPARK_GLM52_KV_BYTES_PER_SCALAR;
	request->fp8_scale_block_size = SPARK_GLM52_KV_FP8_SCALE_BLOCK_SIZE;
	request->index_key_layer_count = SPARK_GLM52_KV_INDEX_KEY_LAYER_COUNT;
	request->index_key_dimension = SPARK_GLM52_KV_INDEX_KEY_DIMENSION;
	request->index_key_bytes_per_scalar = SPARK_GLM52_KV_INDEX_KEY_BYTES_PER_SCALAR;
}

#endif
