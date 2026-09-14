#pragma once

#include <stdint.h>

#define SPARK_ROPE_TABLE_ELEMENTS(rotary_dimension) ((rotary_dimension) / 2u)

typedef struct SparkRopeDomain
{
	uint32_t head_dimension;
	uint32_t rope_dimension;
	uint32_t rope_offset;
	float theta;
	float attention_scale;
	const float *inv_freq_table;
} SparkRopeDomain;

static inline void SparkRopeDomainInitTheta(SparkRopeDomain *domain, uint32_t head_dimension, uint32_t rope_dimension, float theta, float attention_scale)
{
	domain->head_dimension = head_dimension;
	domain->rope_dimension = rope_dimension;
	domain->rope_offset = head_dimension - rope_dimension;
	domain->theta = theta;
	domain->attention_scale = attention_scale;
	domain->inv_freq_table = 0;
}

static inline void SparkRopeDomainInitTable(SparkRopeDomain *domain, uint32_t head_dimension, uint32_t rope_dimension, const float *inv_freq_table, float table_base_theta, float attention_scale)
{
	domain->head_dimension = head_dimension;
	domain->rope_dimension = rope_dimension;
	domain->rope_offset = head_dimension - rope_dimension;
	domain->theta = table_base_theta;
	domain->attention_scale = attention_scale;
	domain->inv_freq_table = inv_freq_table;
}
