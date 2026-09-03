#ifndef SPARKPIPE_SPARK_K3_RESIDENT_DECODE_STAGE_MODULE_H
#define SPARKPIPE_SPARK_K3_RESIDENT_DECODE_STAGE_MODULE_H

#include <stdint.h>

#include "sparkpipe/spark_k3_bind.h"
#include "sparkpipe/spark_k3_pack_load.h"
#include "sparkpipe/spark_k3_pool_sizing.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif


#define SPARK_K3_MODULE_MAX_BOUND_LAYERS 93u
#define SPARK_K3_MODULE_DERIVE_SLICE UINT32_MAX

typedef struct SparkK3ModuleState
{
	SparkK3Pack pack;
	SparkK3PoolSizing sizing;
	uint32_t first_layer;
	uint32_t layer_count;
	uint32_t bound_count;
	SparkK3BoundLayer bound[SPARK_K3_MODULE_MAX_BOUND_LAYERS];
} SparkK3ModuleState;

SparkStatus SparkK3ModuleInitialize(SparkK3ModuleState *state,
	const char *pack_path, uint32_t first_layer, uint32_t layer_count);
void SparkK3ModuleDestroy(SparkK3ModuleState *state);

#ifdef __cplusplus
}
#endif

#endif
