#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_SAMPLING_MIN_TEMPERATURE 0.0001f
#define SPARK_SAMPLING_MAX_TEMPERATURE 2.0f
#define SPARK_SAMPLING_MAX_INVERSE_TEMPERATURE (1.0f / SPARK_SAMPLING_MIN_TEMPERATURE)
#define SPARK_SAMPLING_MAX_TOP_K 1048576u
#define SPARK_SAMPLING_MAX_TOP_LOGPROBS 20u
#define SPARK_SAMPLING_MAX_LOGPROBS (SPARK_SAMPLING_MAX_TOP_LOGPROBS + 1u)
#define SPARK_SAMPLING_NO_TOKEN 0xffffffffu

typedef struct SparkRowSampling
{
	uint64_t seed;
	float inverse_temperature;
	float nucleus;
	uint32_t top_k;
	uint32_t logprobs;
} SparkRowSampling;

typedef struct SparkSamplingLogprob
{
	uint32_t token;
	float logprob;
} SparkSamplingLogprob;

static inline uint32_t SparkSamplingTemperatureValid(float temperature)
{
	return temperature == 0.0f || (temperature >= SPARK_SAMPLING_MIN_TEMPERATURE && temperature <= SPARK_SAMPLING_MAX_TEMPERATURE) ? 1u : 0u;
}

static inline uint32_t SparkSamplingTopPValid(float top_p)
{
	return top_p > 0.0f && top_p <= 1.0f ? 1u : 0u;
}

static inline SparkRowSampling SparkSamplingRuleWith(float temperature, uint64_t seed, float top_p, uint32_t top_k, uint32_t logprobs)
{
	SparkRowSampling rule;
	rule.seed = temperature != 0.0f ? seed : 0u;
	rule.inverse_temperature = temperature != 0.0f ? 1.0f / temperature : 0.0f;
	rule.nucleus = temperature == 0.0f ? 0.0f : SparkSamplingTopPValid(top_p) == 0u ? -1.0f : top_p < 1.0f ? top_p : 0.0f;
	rule.top_k = temperature != 0.0f ? top_k : 0u;
	rule.logprobs = logprobs;
	return rule;
}

static inline SparkRowSampling SparkSamplingRule(float temperature, uint64_t seed)
{
	return SparkSamplingRuleWith(temperature,seed,1.0f,0u,0u);
}

static inline uint32_t SparkSamplingRuleValid(const SparkRowSampling *rule)
{
	if ( rule->inverse_temperature < 0.0f || rule->inverse_temperature > SPARK_SAMPLING_MAX_INVERSE_TEMPERATURE ||
		rule->nucleus < 0.0f || rule->nucleus >= 1.0f || rule->top_k > SPARK_SAMPLING_MAX_TOP_K || rule->logprobs > SPARK_SAMPLING_MAX_LOGPROBS )
		return 0u;
	if ( rule->inverse_temperature == 0.0f && (rule->seed != 0u || rule->nucleus != 0.0f || rule->top_k != 0u) )
		return 0u;
	return 1u;
}

static inline uint32_t SparkSamplingRuleTruncates(const SparkRowSampling *rule)
{
	return rule->inverse_temperature != 0.0f && (rule->nucleus != 0.0f || rule->top_k != 0u) ? 1u : 0u;
}

static inline uint32_t SparkSamplingRuleNeedsDistribution(const SparkRowSampling *rule)
{
	return rule->inverse_temperature != 0.0f || rule->logprobs != 0u ? 1u : 0u;
}

#ifdef __cplusplus
}
#endif
