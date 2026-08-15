#ifndef SPARKPIPE_SPARK_K3_BIND_H
#define SPARKPIPE_SPARK_K3_BIND_H

#include <stdint.h>

#include "sparkpipe/spark_k3_pack_load.h"
#include "sparkpipe/spark_status.h"

/*
 * Pack name -> layer weight binding for K3. The pack manifest is the single
 * source of truth for names; this file owns the per-layer-kind name tables
 * and fills a plain-C weight table the serving module maps onto the CUDA
 * K3LayerWeights struct (inference/llms/kimi_k3/slice.cuh). CUDA-free so
 * host gates and the serving tier share the same resolution.
 */

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

/* Fill the name table for one layer and resolve every entry against the pack. */
SparkStatus SparkK3BindLayer(SparkK3Pack *pack, uint32_t layer_index,
	SparkK3BoundLayer *bound);

/* Lookup helpers used by the serving module. */
const SparkK3PackEntry *SparkK3BoundEntry(const SparkK3BoundLayer *bound,
	const char *name);
const void *SparkK3BoundPayload(const SparkK3Pack *pack,
	const SparkK3BoundLayer *bound, const char *name);

#endif
