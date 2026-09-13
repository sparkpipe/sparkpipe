#pragma once

#include <stdint.h>

#include "sparkpipe/spark_k3_llm_defines.h"

#define SPARK_K3_MODEL_KV_POOL_TOKENS 4194304u
#define SPARK_K3_MODEL_RESTRICTED_VOCAB_COUNT 256u
#define SPARK_K3_MODEL_MAX_PREFILL_TOKENS_PER_DISPATCH 256u
#define SPARK_K3_MODEL_END_OF_TEXT_TOKEN_ID 163585u

#define SPARK_K3_MODEL_ATTNRES_COMPLETED_BLOCKS_BEFORE_LAYER(layer_index) \
	(1u + ((layer_index) / SPARK_K3_MODEL_ATTNRES_BLOCK_LAYERS))
#define SPARK_K3_MODEL_ATTNRES_LAYER_OPENS_BLOCK(layer_index) \
	(((((layer_index) + 1u) % SPARK_K3_MODEL_ATTNRES_BLOCK_LAYERS) == 0u) ? \
	 1u : 0u)

#define SPARK_K3_MODEL_MXFP8_GROUP_SIZE 32u
#define SPARK_K3_MODEL_MXFP4_PAYLOAD_BYTES(rows, columns) \
	(((uint64_t)(rows) * (uint64_t)(columns)) / 2u)
#define SPARK_K3_MODEL_MXFP4_SCALE_BYTES(rows, columns) \
	(((uint64_t)(rows) * (uint64_t)(columns)) / \
	 (uint64_t)SPARK_K3_MODEL_MXFP4_GROUP_SIZE)

#define SPARK_K3_MODEL_BF16_ELEMENT_BYTES ((uint32_t)sizeof(uint16_t))
#define SPARK_K3_MODEL_HIDDEN_BF16_BYTES \
	(SPARK_K3_MODEL_HIDDEN_DIMENSION * SPARK_K3_MODEL_BF16_ELEMENT_BYTES)
