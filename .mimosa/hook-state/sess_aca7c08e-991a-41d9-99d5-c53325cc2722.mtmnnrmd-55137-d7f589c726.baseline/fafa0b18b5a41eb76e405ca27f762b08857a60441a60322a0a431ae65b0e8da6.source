#ifndef SPARKPIPE_SPARK_K3_DSPARK_PACK_H
#define SPARKPIPE_SPARK_K3_DSPARK_PACK_H

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif


#define SPARK_K3_DSPARK_MAX_REFUSAL_BYTES 160u

typedef struct SparkK3DsparkPack
{
	int fd;
	uint8_t *mapping;
	uint64_t file_bytes;
	uint32_t tensor_count;
	uint8_t *entries;
	uint32_t hidden;
	uint32_t layer_count;
	uint32_t query_heads;
	uint32_t kv_heads;
	uint32_t head_dim;
	uint32_t ffn_dimension;
	uint32_t vocab;
	uint32_t block_size;
	uint32_t draft_token_count;
	uint32_t target_tap_layers[5];
	uint32_t markov_rank;
	uint32_t mask_token_id;
	uint32_t sliding_window;
	uint32_t flags;
	uint32_t confidence_input_dimension;
} SparkK3DsparkPack;

SparkStatus SparkK3DsparkPackBind(const char *path, SparkK3DsparkPack *pack,
	char *refusal, uint32_t refusal_bytes);

void SparkK3DsparkPackRelease(SparkK3DsparkPack *pack);

SparkStatus SparkK3DsparkPackPayload(const SparkK3DsparkPack *pack,
	uint32_t kind, uint32_t layer, const void **payload, uint64_t *bytes);

#ifdef __cplusplus
}
#endif

#endif
