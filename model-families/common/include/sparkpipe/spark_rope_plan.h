#pragma once

#include <math.h>
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

typedef struct SparkRopePlanDomain
{
	float theta;
	float yarn_factor;
	float original_positions;
	float beta_fast;
	float beta_slow;
	float attention_factor;
	uint32_t rotary_dimension;
} SparkRopePlanDomain;

typedef struct SparkRopePlan
{
	SparkRopePlanDomain full;
	SparkRopePlanDomain sliding;
} SparkRopePlan;

static inline void SparkRopePlanBuildYarnInvFrequency(
	const SparkRopePlanDomain *domain,
	float *inv_freq)
{
	float low_exact,high_exact;
	float low,high;
	uint32_t half,index;
	if ( domain == 0 || inv_freq == 0 || domain->rotary_dimension < 2u ||
		domain->yarn_factor <= 1.0f )
		return;
	low_exact = ((float)domain->rotary_dimension *
		logf(domain->original_positions /
			(domain->beta_fast * 6.283185307179586f))) /
		(2.0f * logf(domain->theta));
	high_exact = ((float)domain->rotary_dimension *
		logf(domain->original_positions /
			(domain->beta_slow * 6.283185307179586f))) /
		(2.0f * logf(domain->theta));
	low = floorf(fmaxf(fminf(low_exact,(float)domain->rotary_dimension - 1.0f),0.0f));
	high = ceilf(fmaxf(fminf(high_exact,(float)domain->rotary_dimension - 1.0f),0.0f));
	half = domain->rotary_dimension / 2u;
	for (index = 0u; index < half; ++index)
	{
		float base = powf(domain->theta,-2.0f * (float)index / (float)domain->rotary_dimension);
		float ramp = ((float)index - low) / fmaxf(high - low,1e-6f);
		float blend = fminf(fmaxf(ramp,0.0f),1.0f);
		inv_freq[index] = (base * (1.0f - blend)) + ((base / domain->yarn_factor) * blend);
	}
}

static inline void SparkRopeYarnRampTable(float *table, uint32_t rope, float theta, float factor, float beta_fast, float beta_slow, uint32_t original_positions)
{
	const double pi = 3.14159265358979323846;
	double low = floor((double)rope * log((double)original_positions / ((double)beta_fast * 2.0 * pi)) / (2.0 * log((double)theta)));
	double high = ceil((double)rope * log((double)original_positions / ((double)beta_slow * 2.0 * pi)) / (2.0 * log((double)theta)));
	low = low > 0.0 ? low : 0.0;
	high = high < (double)(rope - 1u) ? high : (double)(rope - 1u);
	if ( low == high )
		high += 0.001;
	for ( uint32_t index = 0u; index < rope / 2u; index++ )
	{
		const float frequency = powf(theta, (float)(2u * index) / (float)rope);
		const float extrapolation = 1.0f / frequency, interpolation = 1.0f / (factor * frequency);
		float ramp = ((float)index - (float)low) / ((float)high - (float)low);
		ramp = ramp < 0.0f ? 0.0f : ramp > 1.0f ? 1.0f : ramp;
		const float extrapolation_factor = 1.0f - ramp;
		table[index] = interpolation * (1.0f - extrapolation_factor) + extrapolation * extrapolation_factor;
	}
}

static inline uint32_t SparkRopePlanYarnTableIsValid(
	const SparkRopePlanDomain *domain,
	const float *inv_freq)
{
	uint32_t half;
	if ( domain == 0 || inv_freq == 0 || domain->rotary_dimension < 2u )
		return(0u);
	if ( inv_freq[0] != 1.0f )
		return(0u);
	half = domain->rotary_dimension / 2u;
	if ( inv_freq[half - 1u] >= 1.0f )
		return(0u);
	return(1u);
}
