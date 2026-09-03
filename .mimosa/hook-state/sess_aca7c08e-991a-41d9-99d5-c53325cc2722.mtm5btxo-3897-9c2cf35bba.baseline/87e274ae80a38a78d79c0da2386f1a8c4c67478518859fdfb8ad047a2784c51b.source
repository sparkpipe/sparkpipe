#ifndef SPARKPIPE_SPARK_K3_BIND_H
#define SPARKPIPE_SPARK_K3_BIND_H

#include <stdint.h>

#include "sparkpipe/spark_k3_pack_load.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif


#define SPARK_K3_BIND_MAX_NAMES 32u

typedef struct SparkK3BoundTensor
{
	const char *name;
	SparkK3PackEntry entry;
} SparkK3BoundTensor;

typedef struct SparkK3BoundLayer
{
	uint32_t layer_index;
	uint32_t layer_is_gdn;
	uint32_t layer_is_dense;
	uint32_t tensor_count;
	SparkK3BoundTensor tensors[SPARK_K3_BIND_MAX_NAMES];
} SparkK3BoundLayer;

SparkStatus SparkK3BindLayer(SparkK3Pack *pack, uint32_t layer_index,
	SparkK3BoundLayer *bound);

const SparkK3PackEntry *SparkK3BoundEntry(const SparkK3BoundLayer *bound,
	const char *name);
const void *SparkK3BoundPayload(const SparkK3Pack *pack,
	const SparkK3BoundLayer *bound, const char *name);

#ifdef __cplusplus
}
#endif

#endif
