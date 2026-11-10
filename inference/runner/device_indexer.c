#include "inference/runner/device_indexer.h"

#include <stdio.h>
#include <string.h>

static SparkStatus IndexerRefuse(const char *operation, const char *reason)
{
	fprintf(stderr,"device_indexer: %s refused: %s\n",operation,reason);
	return(SPARK_STATUS_INVALID_ARGUMENT);
}

static uint32_t IndexerRatio(const SparkIndexerShape *shape)
{
	return(shape->key_ratio != 0u ? shape->key_ratio : 1u);
}

static uint32_t IndexerShapeKnown(const SparkIndexerShape *shape)
{
	return(shape != 0 && shape->heads != 0u && shape->heads <= SPARK_INDEXER_MAX_HEADS && shape->head_dimension != 0u &&
		shape->head_dimension <= SPARK_INDEXER_MAX_HEAD_DIM && shape->key_norm <= SPARK_INDEXER_NORM_LAYER ? 1u : 0u);
}

static uint32_t IndexerCacheKnown(const SparkIndexerCache *cache)
{
	return(cache != 0 && cache->page_slots != 0u && cache->pages_per_sequence != 0u && cache->pool_pages != 0u ? 1u : 0u);
}

static void IndexerParams(const SparkIndexerShape *shape, const SparkIndexerCache *cache, uint32_t rows, uint32_t score_stride, SparkIndexerParams *params)
{
	memset(params,0,sizeof(*params));
	params->rows = rows;
	params->heads = shape->heads;
	params->head_dimension = shape->head_dimension;
	params->key_norm = shape->key_norm;
	params->key_norm_epsilon = shape->key_norm_epsilon;
	params->score_scale = shape->score_scale;
	params->topk = shape->topk;
	params->key_ratio = IndexerRatio(shape);
	params->query_stride = shape->query_stride != 0u ? shape->query_stride : shape->heads * shape->head_dimension;
	params->weight_stride = shape->weight_stride != 0u ? shape->weight_stride : shape->heads;
	params->key_stride = shape->key_stride != 0u ? shape->key_stride : shape->head_dimension;
	params->score_stride = score_stride;
	if ( cache != 0 )
	{
		params->page_slots = cache->page_slots;
		params->pages_per_sequence = cache->pages_per_sequence;
		params->pool_pages = cache->pool_pages;
	}
}

uint64_t SparkIndexerCacheBytes(const SparkIndexerShape *shape, uint32_t pool_pages, uint32_t page_slots)
{
	if ( IndexerShapeKnown(shape) == 0u )
		return(0u);
	return((uint64_t)pool_pages * page_slots * shape->head_dimension * sizeof(uint16_t));
}

uint32_t SparkIndexerScoreStride(uint32_t positions)
{
	const uint64_t stride = ((uint64_t)positions + SPARK_INDEXER_SCORE_ALIGN - 1u) / SPARK_INDEXER_SCORE_ALIGN * SPARK_INDEXER_SCORE_ALIGN;
	return(positions != 0u && stride <= 0xffffffffu ? (uint32_t)stride : 0u);
}

uint64_t SparkIndexerScoreBytes(uint32_t rows, uint32_t score_stride)
{
	return((uint64_t)rows * score_stride * sizeof(float));
}

uint64_t SparkIndexerSelectedBytes(uint32_t rows, uint32_t topk)
{
	return((uint64_t)rows * topk * sizeof(uint32_t));
}

uint64_t SparkIndexerAttentionScratchBytes(uint32_t rows, uint32_t heads, uint32_t value_dimension, uint32_t topk)
{
	return(SparkOpsAttentionGatherScratchBytes(rows,heads,value_dimension,topk));
}

uint32_t SparkIndexerVisibleKeys(const SparkIndexerShape *shape, uint32_t position)
{
	return(position == SPARK_INDEXER_NONE ? 0u : (position + 1u) / IndexerRatio(shape));
}

SparkStatus SparkIndexerStoreKeys(SparkDeviceOps *ops, SparkDeviceQueue *queue, const SparkIndexerCache *cache, SparkTensor keys,
	const SparkTensor *norm_weight, const SparkTensor *norm_bias, const SparkHeadRopeParams *rope, SparkTensor sequence_of_row,
	SparkTensor rope_positions, SparkTensor store_positions, const SparkIndexerShape *shape, uint32_t rows)
{
	SparkIndexerParams params;
	if ( IndexerShapeKnown(shape) == 0u || IndexerCacheKnown(cache) == 0u || rows == 0u )
		return(IndexerRefuse("store_keys","a known index shape, a configured cache and rows"));
	if ( rope != 0 && (rope->rope_dimension > shape->head_dimension || rope->rope_dimension % 2u != 0u) )
		return(IndexerRefuse("store_keys","the rope part is an even width inside the index key"));
	IndexerParams(shape,cache,rows,0u,&params);
	params.has_weight = norm_weight != 0 ? 1u : 0u;
	params.has_bias = norm_bias != 0 ? 1u : 0u;
	if ( rope != 0 )
	{
		params.rope_dimension = rope->rope_dimension;
		params.rope_rounding = rope->rope_rounding;
		params.rope_interleaved = rope->interleaved;
		params.rope_scale = rope->scale;
		memcpy(params.inverse_frequency,rope->inverse_frequency,sizeof(params.inverse_frequency));
	}
	return(SparkOpsIndexerStoreKeys(ops,queue,keys,norm_weight,norm_bias,sequence_of_row,rope_positions,store_positions,cache->page_table,cache->pool,&params));
}

SparkStatus SparkIndexerScore(SparkDeviceOps *ops, SparkDeviceQueue *queue, const SparkIndexerCache *cache, SparkTensor query,
	SparkTensor head_weight, SparkTensor sequence_of_row, SparkTensor row_positions, SparkTensor scores, const SparkIndexerShape *shape,
	uint32_t rows, uint32_t score_stride)
{
	SparkIndexerParams params;
	if ( IndexerShapeKnown(shape) == 0u || IndexerCacheKnown(cache) == 0u || rows == 0u || score_stride == 0u )
		return(IndexerRefuse("score","a known index shape, a configured cache, rows and a score stride"));
	IndexerParams(shape,cache,rows,score_stride,&params);
	return(SparkOpsIndexerScore(ops,queue,query,head_weight,cache->pool,cache->page_table,sequence_of_row,row_positions,scores,&params));
}

SparkStatus SparkIndexerTopk(SparkDeviceOps *ops, SparkDeviceQueue *queue, SparkTensor scores, SparkTensor selected, uint32_t rows,
	uint32_t score_stride, uint32_t topk)
{
	SparkIndexerParams params;
	if ( rows == 0u || score_stride == 0u || topk == 0u || topk > SPARK_INDEXER_MAX_TOPK )
		return(IndexerRefuse("topk","rows, a score stride and between 1 and 2048 selected keys"));
	memset(&params,0,sizeof(params));
	params.rows = rows;
	params.score_stride = score_stride;
	params.topk = topk;
	return(SparkOpsIndexerTopk(ops,queue,scores,selected,&params));
}

SparkStatus SparkIndexerSelect(SparkDeviceOps *ops, SparkDeviceQueue *queue, const SparkIndexerCache *cache, SparkTensor query,
	SparkTensor head_weight, SparkTensor sequence_of_row, SparkTensor row_positions, SparkTensor scores, SparkTensor selected,
	const SparkIndexerShape *shape, uint32_t rows, uint32_t score_stride)
{
	SparkStatus status = SparkIndexerScore(ops,queue,cache,query,head_weight,sequence_of_row,row_positions,scores,shape,rows,score_stride);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkIndexerTopk(ops,queue,scores,selected,rows,score_stride,shape->topk));
}

SparkStatus SparkIndexerAttention(SparkDeviceOps *ops, SparkDeviceQueue *queue, const SparkOpsPaged *paged, SparkTensor query, SparkTensor gate,
	const SparkTensor *sinks, SparkTensor selected, uint32_t topk, uint32_t key_ratio, SparkTensor partial, SparkTensor output,
	const SparkAttentionParams *params)
{
	SparkAttentionParams launch;
	if ( paged == 0 || params == 0 || topk == 0u || topk > SPARK_INDEXER_MAX_TOPK )
		return(IndexerRefuse("attention","a paged cache, attention parameters and between 1 and 2048 selected keys"));
	launch = *params;
	launch.gather_keys = topk;
	launch.gather_ratio = key_ratio != 0u ? key_ratio : 1u;
	return(SparkOpsAttentionGather(ops,queue,paged,query,gate,sinks,selected,partial,output,&launch));
}
