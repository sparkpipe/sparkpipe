#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_SAMPLING_MIN_TEMPERATURE 0.0001f
#define SPARK_SAMPLING_MAX_TEMPERATURE 2.0f
#define SPARK_SAMPLING_MAX_INVERSE_TEMPERATURE (1.0f / SPARK_SAMPLING_MIN_TEMPERATURE)

typedef struct SparkRowSampling
{
	uint64_t seed;
	float inverse_temperature;
	uint32_t reserved;
} SparkRowSampling;

static inline uint32_t SparkSamplingTemperatureValid(float temperature)
{
	return temperature == 0.0f || (temperature >= SPARK_SAMPLING_MIN_TEMPERATURE && temperature <= SPARK_SAMPLING_MAX_TEMPERATURE) ? 1u : 0u;
}

static inline SparkRowSampling SparkSamplingRule(float temperature, uint64_t seed)
{
	SparkRowSampling rule;
	rule.seed = temperature != 0.0f ? seed : 0u;
	rule.inverse_temperature = temperature != 0.0f ? 1.0f / temperature : 0.0f;
	rule.reserved = 0u;
	return rule;
}

static inline uint32_t SparkSamplingRuleValid(const SparkRowSampling *rule)
{
	return rule->reserved == 0u && rule->inverse_temperature >= 0.0f && rule->inverse_temperature <= SPARK_SAMPLING_MAX_INVERSE_TEMPERATURE && (rule->inverse_temperature != 0.0f || rule->seed == 0u) ? 1u : 0u;
}

#ifdef __cplusplus
}
#endif
