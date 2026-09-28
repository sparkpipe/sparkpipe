#pragma once

#include <stdint.h>

#define SPARK_STATE_SPAN_ALIGN 16u

typedef struct SparkStateSpan
{
	uint8_t *base;
	uint64_t row_stride;
	uint64_t snapshot_offset;
	uint32_t row_bytes;
	uint32_t state_rows;
} SparkStateSpan;

static inline uint64_t SparkStateSpansLayout(SparkStateSpan *spans,uint32_t span_count,uint32_t rows_max,uint32_t *row_words_max)
{
	uint64_t offset = 0u;
	uint32_t index,words = 0u;
	for (index=0u; index<span_count; index++)
	{
		if ( spans[index].base == 0 || spans[index].row_bytes == 0u || spans[index].state_rows == 0u || ((uintptr_t)spans[index].base % SPARK_STATE_SPAN_ALIGN) != 0u || (spans[index].row_stride % SPARK_STATE_SPAN_ALIGN) != 0u || (spans[index].row_bytes % SPARK_STATE_SPAN_ALIGN) != 0u || spans[index].row_bytes > spans[index].row_stride )
			return(0u);
		spans[index].snapshot_offset = offset;
		offset += (uint64_t)rows_max * spans[index].row_bytes;
		words = spans[index].row_bytes / SPARK_STATE_SPAN_ALIGN > words ? spans[index].row_bytes / SPARK_STATE_SPAN_ALIGN : words;
	}
	if ( row_words_max != 0 )
		*row_words_max = words;
	return(offset);
}
