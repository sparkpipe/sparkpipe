#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_weight_stream.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_PACK_WINDOW_MAX_DENSE_RANGES 4u
#define SPARK_PACK_WINDOW_MAX_DENSE_BUFFERS 4u
#define SPARK_PACK_WINDOW_MAX_EXPERT_SLOTS 64u

typedef SparkStatus (*SparkPackWindowRead)(void *context, uint64_t pack_offset, uint64_t bytes, uint8_t *destination);

typedef struct SparkPackWindowRange
{
	uint64_t offset;
	uint64_t bytes;
} SparkPackWindowRange;

typedef struct SparkPackWindowLayer
{
	SparkPackWindowRange dense[SPARK_PACK_WINDOW_MAX_DENSE_RANGES];
	uint32_t dense_count;
	uint32_t reserved0;
	SparkPackWindowRange experts;
} SparkPackWindowLayer;

typedef struct SparkPackWindowConfig
{
	SparkWeightStream *stream;
	uint8_t *region;
	uint64_t region_bytes;
	uint32_t rank;
	uint32_t ranks;
	uint32_t dense_buffers;
	uint32_t expert_slots;
	uint64_t dense_capacity;
	uint64_t expert_capacity;
	SparkPackWindowRead read_own;
	void *own_context;
	SparkPackWindowRead read_partner;
	void *partner_context;
} SparkPackWindowConfig;

typedef struct SparkPackWindow
{
	SparkPackWindowConfig config;
	uint64_t dense_base[SPARK_PACK_WINDOW_MAX_DENSE_BUFFERS];
	uint64_t expert_slot[SPARK_PACK_WINDOW_MAX_EXPERT_SLOTS];
	uint64_t own_staging;
	uint64_t partner_staging;
	uint64_t staged_layer_offset;
	uint32_t staged;
	uint32_t reserved0;
} SparkPackWindow;

uint64_t SparkPackWindowRegionBytes(uint32_t ranks, uint32_t dense_buffers, uint64_t dense_capacity, uint32_t expert_slots, uint64_t expert_capacity);
SparkStatus SparkPackWindowInitialize(SparkPackWindow *window, const SparkPackWindowConfig *config);
uint64_t SparkPackWindowDenseBytes(const SparkPackWindowLayer *layer);
SparkStatus SparkPackWindowDense(SparkPackWindow *window, const SparkPackWindowLayer *layer, uint32_t buffer, uint32_t tag, uint8_t **rank_base);
SparkStatus SparkPackWindowExpertStep(SparkPackWindow *window, const SparkPackWindowLayer *layer, uint32_t step, uint32_t slot, uint32_t tag, uint32_t *source,
	const uint8_t **block);

#ifdef __cplusplus
}
#endif
