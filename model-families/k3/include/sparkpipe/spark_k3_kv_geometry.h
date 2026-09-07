#ifndef SPARKPIPE_SPARK_K3_KV_GEOMETRY_H
#define SPARKPIPE_SPARK_K3_KV_GEOMETRY_H

#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_state_pool.h"


#define SPARK_K3_KV_LAYOUT SPARK_KV_CACHE_LAYOUT_COMPRESSED_KEY_VALUE
#define SPARK_K3_KV_MLA_LAYER_COUNT 24u
#define SPARK_K3_KV_KDA_LAYER_COUNT 69u
#define SPARK_K3_KV_LATENT_DIMENSION 512u
#define SPARK_K3_KV_ROPE_DIMENSION 64u
#define SPARK_K3_KV_BYTES_PER_SCALAR 2u
#define SPARK_K3_KV_INDEX_KEY_LAYER_COUNT 0u
#define SPARK_K3_KV_INDEX_KEY_DIMENSION 0u

#define SPARK_K3_KV_KDA_HEADS 96u
#define SPARK_K3_KV_KDA_KEY_DIM 128u
#define SPARK_K3_KV_KDA_VALUE_DIM 128u
#define SPARK_K3_KV_KDA_CONV_KERNEL 4u
#define SPARK_K3_KV_KDA_STATE_BYTES_PER_LAYER \
	((uint64_t)SPARK_K3_KV_KDA_HEADS * SPARK_K3_KV_KDA_KEY_DIM * \
	 SPARK_K3_KV_KDA_VALUE_DIM * 4u)
#define SPARK_K3_KV_KDA_CONV_BYTES_PER_LAYER \
	(((2ull * SPARK_K3_KV_KDA_HEADS * SPARK_K3_KV_KDA_KEY_DIM) + \
	  ((uint64_t)SPARK_K3_KV_KDA_HEADS * SPARK_K3_KV_KDA_VALUE_DIM)) * \
	 SPARK_K3_KV_KDA_CONV_KERNEL * 2u)
#define SPARK_K3_KV_KDA_SLOT_BYTES \
	(SPARK_K3_KV_KDA_LAYER_COUNT * \
	 (SPARK_K3_KV_KDA_STATE_BYTES_PER_LAYER + \
	  SPARK_K3_KV_KDA_CONV_BYTES_PER_LAYER))

static inline void SparkK3KvFillCapacityRequest(SparkKvCacheCapacityRequest *request)
{
	request->layout = SPARK_K3_KV_LAYOUT;
	request->layer_count = SPARK_K3_KV_MLA_LAYER_COUNT;
	request->head_count = 0u;
	request->query_key_head_dimension = 0u;
	request->value_head_dimension = 0u;
	request->compressed_dimension = SPARK_K3_KV_LATENT_DIMENSION;
	request->position_dimension = SPARK_K3_KV_ROPE_DIMENSION;
	request->bytes_per_scalar = SPARK_K3_KV_BYTES_PER_SCALAR;
	request->index_key_layer_count = SPARK_K3_KV_INDEX_KEY_LAYER_COUNT;
	request->index_key_dimension = SPARK_K3_KV_INDEX_KEY_DIMENSION;
	request->index_key_bytes_per_scalar = 0u;
}

#endif
