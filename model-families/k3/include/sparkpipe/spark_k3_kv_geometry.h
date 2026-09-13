#pragma once

#include <stdint.h>

#include "sparkpipe/spark_k3_llm_defines.h"
#include "sparkpipe/spark_kv_cache.h"

#define SPARK_K3_KV_LAYOUT SPARK_KV_CACHE_LAYOUT_COMPRESSED_KEY_VALUE
#define SPARK_K3_KV_MLA_LAYER_COUNT SPARK_K3_MODEL_MLA_LAYER_COUNT
#define SPARK_K3_KV_KDA_LAYER_COUNT SPARK_K3_MODEL_KDA_LAYER_COUNT
#define SPARK_K3_KV_LATENT_DIMENSION SPARK_K3_MODEL_MLA_LATENT_DIMENSION
#define SPARK_K3_KV_ROPE_DIMENSION SPARK_K3_MODEL_MLA_ROPE_DIMENSION
#define SPARK_K3_KV_INDEX_KEY_LAYER_COUNT 0u
#define SPARK_K3_KV_INDEX_KEY_DIMENSION 0u
#define SPARK_K3_KV_INDEX_KEY_BYTES_PER_SCALAR 0u
#define SPARK_K3_KV_KDA_HEADS SPARK_K3_MODEL_KDA_HEAD_COUNT
#define SPARK_K3_KV_KDA_KEY_DIM SPARK_K3_MODEL_KDA_HEAD_KEY_DIMENSION
#define SPARK_K3_KV_KDA_VALUE_DIM SPARK_K3_MODEL_KDA_HEAD_VALUE_DIMENSION
#define SPARK_K3_KV_KDA_CONV_KERNEL SPARK_K3_MODEL_KDA_CONV_KERNEL
#define SPARK_K3_KV_KDA_STATE_BYTES_PER_LAYER \
	SPARK_K3_MODEL_KDA_STATE_BYTES_PER_LAYER
#define SPARK_K3_KV_KDA_CONV_BYTES_PER_LAYER \
	SPARK_K3_MODEL_KDA_CONV_WINDOW_BYTES_PER_LAYER
#define SPARK_K3_KV_KDA_SLOT_BYTES SPARK_K3_MODEL_KDA_SLOT_BYTES

static inline void SparkK3KvFillCapacityRequest(SparkKvCacheCapacityRequest *request)
{
	request->abi_version = SPARK_KV_CACHE_ABI_VERSION;
	request->descriptor_bytes =
		SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
	request->layout = SPARK_K3_KV_LAYOUT;
	request->layer_count = SPARK_K3_KV_MLA_LAYER_COUNT;
	request->head_count = 0u;
	request->query_key_head_dimension = 0u;
	request->value_head_dimension = 0u;
	request->compressed_dimension = SPARK_K3_KV_LATENT_DIMENSION;
	request->position_dimension = SPARK_K3_KV_ROPE_DIMENSION;
	request->bytes_per_scalar = SPARK_K3_KV_BYTES_PER_SCALAR;
	request->fp8_scale_block_size = 0u;
	request->index_key_layer_count = SPARK_K3_KV_INDEX_KEY_LAYER_COUNT;
	request->index_key_dimension = SPARK_K3_KV_INDEX_KEY_DIMENSION;
	request->index_key_bytes_per_scalar =
		SPARK_K3_KV_INDEX_KEY_BYTES_PER_SCALAR;
}
