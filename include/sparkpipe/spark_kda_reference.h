#pragma once

#include <stdint.h>

// Scalar reference for one KDA head. State is [key][value]; callers supply
// normalized/scaled query and key vectors and positive key/value dimensions.
// Retention is exp(log_gate), applied exactly once before the prediction.
// This independent host oracle is not a production device implementation.
static inline void SparkKdaReferenceHead(float *state,const float *query,const float *key,const float *value,const float *retention,float beta,uint32_t key_count,uint32_t value_count,float *output)
{
	uint32_t k,v;
	float prediction,correction,total;
	for (k=0u; k<key_count; k++)
		for (v=0u; v<value_count; v++)
			state[(uint64_t)k * value_count + v] *= retention[k];
	for (v=0u; v<value_count; v++)
	{
		prediction = 0.0f;
		for (k=0u; k<key_count; k++)
			prediction += state[(uint64_t)k * value_count + v] * key[k];
		correction = beta * (value[v] - prediction);
		total = 0.0f;
		for (k=0u; k<key_count; k++)
		{
			state[(uint64_t)k * value_count + v] += correction * key[k];
			total += state[(uint64_t)k * value_count + v] * query[k];
		}
		output[v] = total;
	}
}
