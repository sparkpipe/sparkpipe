#pragma once

#include <stdint.h>

#define SPARK_L2_PREFETCH_BYTES_DEFAULT (12u << 20)
#define SPARK_L2_PREFETCH_BYTES_MAX (12u << 20)
#define SPARK_L2_PREFETCH_BYTES_STEP 65536u
#define SPARK_L2_PREFETCH_BLOCKS_DEFAULT 48u
#define SPARK_L2_PREFETCH_BLOCKS_MAX 192u

typedef struct SparkL2PrefetchShape
{
	uint32_t bytes;
	uint32_t blocks;
} SparkL2PrefetchShape;

static inline uint32_t SparkL2PrefetchShapeValid(const SparkL2PrefetchShape *shape)
{
	return(shape != 0 && shape->bytes != 0u && shape->bytes <= SPARK_L2_PREFETCH_BYTES_MAX &&
		shape->bytes % SPARK_L2_PREFETCH_BYTES_STEP == 0u && shape->blocks != 0u &&
		shape->blocks <= SPARK_L2_PREFETCH_BLOCKS_MAX ? 1u : 0u);
}
