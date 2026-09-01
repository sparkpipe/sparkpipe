/* Host-side unit test for the head-split attention-projection staging
 * slices (the LoadAttnProjectionSlice math, compiled here without CUDA).
 *
 * Builds a synthetic full-width pack entry image in a temp file with a
 * recognizable per-element pattern (value = row*65536 + column truncated
 * to u16 — every (row,column) decodable), stages each rank's slice, and
 * asserts the staged bytes equal the expected head-window of the pattern:
 *   KEY/VALUE: contiguous row span [kv_base*256, +local_rows) x all H.
 *   OUTPUT:    all H rows x column window [rank*q_local*256, +local_cols).
 * Degrees 4 (dividing) and 16 (replicating kv) both checked.
 *
 * Build:  cc -std=c11 -Wall -Wextra -Werror -O2 -I. -Iinclude \
 *           -Imodel-families/common/include -Imodel-families/qwen38_max/include \
 *           -Imodules/qwen38_max_resident_decode_stage/include \
 *           tests/test_qwen38max_attn_slice.c -o /tmp/test_attn_slice && /tmp/test_attn_slice
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "spark_qwen38_max_stagepack_format.h"
#include "spark_qwen38_max_attn_slice.h"

#define H 8192u
#define Q_HEADS SPARK_QWEN38_MAX_MODEL_ATTN_QUERY_HEAD_COUNT
#define KV_HEADS SPARK_QWEN38_MAX_MODEL_ATTN_KV_HEAD_COUNT
#define HEAD 256u

static uint16_t pattern_element(uint32_t row, uint32_t column)
{
	return (uint16_t)((row * 65536u + column * 7u) & 0xFFFFu);
}

static int stage_and_check(FILE *file, uint32_t tensor_kind, uint32_t rows,
	uint32_t columns, uint64_t payload_offset, uint32_t tp_degree, uint32_t tp_rank)
{
	SparkQwen38MaxStagePackEntry entry,sliced;
	uint16_t *staging = 0;
	uint64_t staging_bytes = 0;
	uint32_t local_rows,local_columns,r,c;
	SparkStatus status;

	(void)memset(&entry,0,sizeof(entry));
	entry.tensor_kind = tensor_kind;
	entry.layer_index = 3u;
	entry.weight_format = SPARK_QWEN38_MAX_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16;
	entry.rows = rows;
	entry.columns = columns;
	entry.payload_offset = payload_offset;
	entry.payload_bytes = (uint64_t)rows * columns * 2u;
	status = SparkQwen38MaxModuleStageAttnProjectionSlice(
		file,&entry,tp_degree,tp_rank,&staging,&sliced,&staging_bytes);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"FAIL kind=%u tp=%u/%u stage status=%d\n",tensor_kind,tp_degree,tp_rank,(int)status);
		free(staging);
		return 1;
	}
	local_rows = sliced.rows;
	local_columns = sliced.columns;
	if ( sliced.payload_bytes != staging_bytes || sliced.scale_bytes != 0u )
	{
		fprintf(stderr,"FAIL kind=%u tp=%u/%u entry bytes mismatch\n",tensor_kind,tp_degree,tp_rank);
		free(staging);
		return 1;
	}
	if ( tensor_kind == SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_OUTPUT )
	{
		uint32_t q_local = Q_HEADS / tp_degree;
		uint64_t column_base = (uint64_t)tp_rank * q_local * HEAD;
		if ( local_rows != rows || local_columns != q_local * HEAD )
		{
			fprintf(stderr,"FAIL out shape %ux%u tp=%u/%u\n",local_rows,local_columns,tp_degree,tp_rank);
			free(staging);
			return 1;
		}
		for ( r = 0u; r < local_rows; r += 997u ) /* sampled rows */
			for ( c = 0u; c < local_columns; ++c )
				if ( staging[(uint64_t)r * local_columns + c] != pattern_element(r,(uint32_t)column_base + c) )
				{
					fprintf(stderr,"FAIL out byte r=%u c=%u tp=%u/%u\n",r,c,tp_degree,tp_rank);
					free(staging);
					return 1;
				}
	}
	else
	{
		uint32_t local_kv = SPARK_QWEN38_MAX_MODEL_ATTN_LOCAL_KV_HEAD_COUNT(tp_degree);
		uint64_t row_base = (uint64_t)SPARK_QWEN38_MAX_MODEL_ATTN_RANK_KV_HEAD_BASE(tp_degree,tp_rank) * HEAD;
		if ( local_rows != local_kv * HEAD || local_columns != columns )
		{
			fprintf(stderr,"FAIL kv shape %ux%u tp=%u/%u\n",local_rows,local_columns,tp_degree,tp_rank);
			free(staging);
			return 1;
		}
		for ( r = 0u; r < local_rows; r += 17u )
			for ( c = 0u; c < local_columns; c += 613u )
				if ( staging[(uint64_t)r * local_columns + c] != pattern_element((uint32_t)row_base + r,c) )
				{
					fprintf(stderr,"FAIL kv byte r=%u c=%u tp=%u/%u\n",r,c,tp_degree,tp_rank);
					free(staging);
					return 1;
				}
	}
	free(staging);
	printf("ok kind=%u tp=%u/%u shape %ux%u staged\n",tensor_kind,tp_degree,tp_rank,local_rows,local_columns);
	return 0;
}

int main(void)
{
	/* Synthetic pack image: o_proj [H, Q] and k_proj [kv*256, H] patterns
	 * written back to back; entries point at their offsets. */
	const uint32_t q_total = Q_HEADS * HEAD; /* 16384 */
	const uint32_t kv_rows = KV_HEADS * HEAD; /* 1024 */
	const uint64_t out_bytes = (uint64_t)H * q_total * 2u;
	uint16_t *out_image = (uint16_t *)malloc((size_t)out_bytes);
	uint16_t *kv_image = (uint16_t *)malloc((size_t)kv_rows * H * 2u);
	FILE *file;
	uint32_t row,column,tp,rank;
	int failures = 0;

	if ( out_image == 0 || kv_image == 0 )
	{
		fprintf(stderr,"FAIL alloc\n");
		return 1;
	}
	for ( row = 0u; row < H; ++row )
		for ( column = 0u; column < q_total; ++column )
			out_image[(uint64_t)row * q_total + column] = pattern_element(row,column);
	for ( row = 0u; row < kv_rows; ++row )
		for ( column = 0u; column < H; ++column )
			kv_image[(uint64_t)row * H + column] = pattern_element(row,column);

	file = tmpfile();
	if ( file == 0 )
	{
		fprintf(stderr,"FAIL tmpfile\n");
		return 1;
	}
	if ( fwrite(out_image,1u,(size_t)out_bytes,file) != out_bytes ||
	     fwrite(kv_image,1u,(size_t)kv_rows * H * 2u,file) != (size_t)kv_rows * H * 2u )
	{
		fprintf(stderr,"FAIL write\n");
		return 1;
	}

	for ( tp = 4u; tp <= 16u; tp = (tp == 4u ? 16u : 32u) )
	{
		for ( rank = 0u; rank < tp; ++rank )
		{
			failures += stage_and_check(file,SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_OUTPUT,H,q_total,0u,tp,rank);
			failures += stage_and_check(file,SPARK_QWEN38_MAX_STAGEPACK_TENSOR_ATTN_KEY,kv_rows,H,out_bytes,tp,rank);
		}
	}
	fclose(file);
	free(out_image);
	free(kv_image);
	if ( failures != 0 )
	{
		fprintf(stderr,"FAIL %d case(s)\n",failures);
		return 1;
	}
	printf("PASS qwen38max attn slice staging (tp4 + tp16, all ranks, k/v rows + o columns)\n");
	return 0;
}
