/* Head-split attention-projection staging: the rank's whole-head slice of
 * a full-width pack entry, staged host-side (no CUDA — the module's load
 * path uploads the returned span; tests run this file standalone).
 *
 *   KEY/VALUE [kv_heads*256, H]: cut along the OUTPUT rows — the rank's
 *   head block is a contiguous row span, one read.
 *   OUTPUT [H, Q]: cut along the INPUT columns — the rank owns
 *   q_heads/degree whole heads starting at rank*(q_heads/degree); strided
 *   row-by-row staging.
 *   QUERY stays full-width: its fused q|gate row layout is not
 *   head-planar, and the kernels already read only the rank's head range.
 *
 * The kv head base is SPARK_QWEN38_MAX_MODEL_ATTN_RANK_KV_HEAD_BASE, the
 * same replication arithmetic the merged #765 kernels use, so the staged
 * slice and the kernel's cache/head indexing always agree. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "spark_qwen38_max_attn_slice.h"

SparkStatus SparkQwen38MaxModuleStageAttnProjectionSlice(
	FILE *file,
	const SparkQwen38MaxStagePackEntry *entry,
	uint32_t tp_degree,
	uint32_t tp_rank,
	uint16_t **staging_out,
	SparkQwen38MaxStagePackEntry *sliced,
	uint64_t *staging_bytes_out)
{
	uint32_t heads_per_rank,local_rows = 0u,local_columns = 0u;
	uint32_t full_rows,full_columns;
	uint64_t column_base = 0u,source_offset = 0u,span_bytes = 0u;
	uint16_t *staging;
	uint32_t row_index;

	if ( file == 0 || entry == 0 || staging_out == 0 || sliced == 0 || staging_bytes_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*staging_out = 0;
	if ( entry->weight_format != SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16 )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( tp_degree == 0u || tp_degree > SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_HEAD_COUNT )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	heads_per_rank = SPARK_QWEN38_MAX_MODEL_ATTN_LOCAL_KV_HEAD_COUNT(tp_degree);
	full_rows = entry->rows;
	full_columns = entry->columns;
	if ( entry->payload_bytes != (uint64_t)full_rows * full_columns * 2u )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( entry->tensor_kind == SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_OUTPUT )
	{
		uint32_t local_q_heads = SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_HEAD_COUNT / tp_degree;
		column_base = (uint64_t)tp_rank * local_q_heads * SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION;
		local_columns = local_q_heads * SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION;
		local_rows = full_rows;
		if ( local_columns == 0u || local_columns > full_columns )
			return(SPARK_STATUS_VALIDATION_FAILED);
		span_bytes = (uint64_t)local_rows * local_columns * 2u;
		staging = (uint16_t *)malloc((size_t)span_bytes);
		if ( staging == 0 )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		for ( row_index = 0u; row_index < local_rows; row_index++ )
		{
			source_offset = entry->payload_offset +
				((uint64_t)row_index * full_columns + column_base) * 2u;
			if ( fseeko(file,(off_t)source_offset,SEEK_SET) != 0 ||
			     fread(staging + (uint64_t)row_index * local_columns,1u,(size_t)local_columns * 2u,file) != local_columns * 2u )
			{
				free(staging);
				fprintf(stderr,"q38max_attn_slice read_failed kind=%u row=%u\n",entry->tensor_kind,row_index);
				return(SPARK_STATUS_IO_ERROR);
			}
		}
	}
	else if ( entry->tensor_kind == SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_KEY ||
	          entry->tensor_kind == SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_VALUE )
	{
		/* replicating degrees (tp16 over 4 kv heads): LOCAL_KV_HEAD_COUNT
		 * and RANK_KV_HEAD_BASE already express the shared-head slice */
		local_rows = heads_per_rank * SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION;
		local_columns = full_columns;
		if ( local_rows == 0u || local_rows > full_rows )
			return(SPARK_STATUS_VALIDATION_FAILED);
		source_offset = entry->payload_offset +
			(uint64_t)SPARK_QWEN38_MAX_MODEL_ATTN_RANK_KV_HEAD_BASE(tp_degree,tp_rank)
				* SPARK_QWEN38_MAX_MODEL_ATTN_HEAD_DIMENSION * full_columns * 2u;
		span_bytes = (uint64_t)local_rows * local_columns * 2u;
		staging = (uint16_t *)malloc((size_t)span_bytes);
		if ( staging == 0 )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		if ( fseeko(file,(off_t)source_offset,SEEK_SET) != 0 ||
		     fread(staging,1u,(size_t)span_bytes,file) != span_bytes )
		{
			free(staging);
			fprintf(stderr,"q38max_attn_slice read_failed kind=%u span\n",entry->tensor_kind);
			return(SPARK_STATUS_IO_ERROR);
		}
	}
	else
		return(SPARK_STATUS_VALIDATION_FAILED);
	*sliced = *entry;
	sliced->rows = local_rows;
	sliced->columns = local_columns;
	sliced->payload_bytes = span_bytes;
	sliced->scale_bytes = 0;
	sliced->scale_offset = 0;
	*staging_out = staging;
	*staging_bytes_out = span_bytes;
	return(SPARK_STATUS_OK);
}
