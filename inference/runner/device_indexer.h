#ifndef SPARKPIPE_INFERENCE_RUNNER_DEVICE_INDEXER_H
#define SPARKPIPE_INFERENCE_RUNNER_DEVICE_INDEXER_H

#include <stdint.h>

#include "inference/runner/device_ops.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_INDEXER_NONE 0xffffffffu
#define SPARK_INDEXER_MAX_TOPK 2048u
#define SPARK_INDEXER_MAX_HEADS 128u
#define SPARK_INDEXER_MAX_HEAD_DIM 256u
#define SPARK_INDEXER_SCORE_ALIGN 256u

typedef enum SparkIndexerNorm
{
	SPARK_INDEXER_NORM_NONE = 0,
	SPARK_INDEXER_NORM_RMS = 1,
	SPARK_INDEXER_NORM_LAYER = 2
} SparkIndexerNorm;

typedef struct SparkIndexerShape
{
	uint32_t heads;
	uint32_t head_dimension;
	uint32_t key_norm;
	float key_norm_epsilon;
	float score_scale;
	uint32_t topk;
	uint32_t key_ratio;
	uint32_t query_stride;
	uint32_t weight_stride;
	uint32_t key_stride;
} SparkIndexerShape;

typedef struct SparkIndexerCache
{
	SparkTensor pool;
	SparkTensor page_table;
	uint32_t page_slots;
	uint32_t pages_per_sequence;
	uint32_t pool_pages;
	uint32_t sequences;
} SparkIndexerCache;

uint64_t SparkIndexerCacheBytes(const SparkIndexerShape *shape, uint32_t pool_pages, uint32_t page_slots);
uint32_t SparkIndexerScoreStride(uint32_t positions);
uint64_t SparkIndexerScoreBytes(uint32_t rows, uint32_t score_stride);
uint64_t SparkIndexerSelectedBytes(uint32_t rows, uint32_t topk);
uint64_t SparkIndexerAttentionScratchBytes(uint32_t rows, uint32_t heads, uint32_t value_dimension, uint32_t topk);
uint32_t SparkIndexerVisibleKeys(const SparkIndexerShape *shape, uint32_t position);

SparkStatus SparkIndexerStoreKeys(SparkDeviceOps *ops, SparkDeviceQueue *queue, const SparkIndexerCache *cache, SparkTensor keys,
	const SparkTensor *norm_weight, const SparkTensor *norm_bias, const SparkHeadRopeParams *rope, SparkTensor sequence_of_row,
	SparkTensor rope_positions, SparkTensor store_positions, const SparkIndexerShape *shape, uint32_t rows);
SparkStatus SparkIndexerScore(SparkDeviceOps *ops, SparkDeviceQueue *queue, const SparkIndexerCache *cache, SparkTensor query,
	SparkTensor head_weight, SparkTensor sequence_of_row, SparkTensor row_positions, SparkTensor scores, const SparkIndexerShape *shape,
	uint32_t rows, uint32_t score_stride);
SparkStatus SparkIndexerTopk(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor scores, SparkTensor selected, uint32_t rows,
	uint32_t score_stride, uint32_t topk);
SparkStatus SparkIndexerSelect(SparkDeviceOps *ops, SparkDeviceQueue *queue, const SparkIndexerCache *cache, SparkTensor query,
	SparkTensor head_weight, SparkTensor sequence_of_row, SparkTensor row_positions, SparkTensor scores, SparkTensor selected,
	const SparkIndexerShape *shape, uint32_t rows, uint32_t score_stride);
SparkStatus SparkIndexerAttention(SparkDeviceOps *ops, SparkDeviceQueue *queue, const SparkOpsPaged *paged, SparkTensor query, SparkTensor gate,
	const SparkTensor *sinks, SparkTensor selected, uint32_t topk, uint32_t key_ratio, SparkTensor partial, SparkTensor output,
	const SparkAttentionParams *params);

#ifdef __cplusplus
}
#endif

#endif
