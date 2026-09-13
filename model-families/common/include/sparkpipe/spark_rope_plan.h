#pragma once

#include <math.h>
#include <stdint.h>

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
