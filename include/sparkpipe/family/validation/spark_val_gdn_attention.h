#pragma once

static float SPARK_FAMILY(ValSilu)(float value)
{
	return(value / (1.0f + expf(-value)));
}

static void SPARK_FAMILY(ValL2Norm)(const float *input, float *output, uint32_t dimension)
{
	uint32_t element;
	float total = 0.0f;
	for (element = 0u; element < dimension; element++)
		total += input[element] * input[element];
	total = 1.0f / sqrtf(total + 1e-6f);
	for (element = 0u; element < dimension; element++)
		output[element] = input[element] * total;
}

static void SPARK_FAMILY(ValRope)(float *vector, uint32_t rope_dim, uint32_t position, float theta)
{
	uint32_t pair,half = rope_dim / 2u;
	float frequency,angle,cosine,sine,low,high;
	for (pair = 0u; pair < half; pair++)
	{
		frequency = powf(theta,-((float)(2u * pair) / (float)rope_dim));
		angle = (float)position * frequency;
		cosine = cosf(angle);
		sine = sinf(angle);
		low = vector[pair];
		high = vector[pair + half];
		vector[pair] = (low * cosine) - (high * sine);
		vector[pair + half] = (high * cosine) + (low * sine);
	}
}

static void SPARK_FAMILY(ValRmsNorm)(const float *input, const float *weight, float *output, uint32_t dimension, float epsilon)
{
	uint32_t element;
	float variance = 0.0f,inverse;
	for (element = 0u; element < dimension; element++)
		variance += input[element] * input[element];
	inverse = 1.0f / sqrtf((variance / (float)dimension) + epsilon);
	for (element = 0u; element < dimension; element++)
		output[element] = input[element] * inverse * weight[element];
}

static void SPARK_FAMILY(ValAttention)(const float *q_fused, const float *k_cache, const float *v_cache, const float *q_norm_weight, float *output, uint32_t group, uint32_t tokens, float epsilon)
{
	float qh[SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION)],scores[SPARK_FAMILY_CONST(VALIDATION_ATTN_TOKENS)],probability;
	float scale = 1.0f / sqrtf((float)SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION)),maximum,total;
	uint32_t head,element,token;
	for (head = 0u; head < group; head++)
	{
		const float *fused = q_fused + ((uint64_t)head * 2u * SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION));
		SPARK_FAMILY(ValRmsNorm)(fused,q_norm_weight,qh,SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION),epsilon);
		SPARK_FAMILY(ValRope)(qh,SPARK_FAMILY_CONST(MODEL_ATTN_ROPE_DIMENSION),tokens - 1u,SPARK_FAMILY_CONST(MODEL_ATTN_ROPE_THETA));
		maximum = -3.0e38f;
		for (token = 0u; token < tokens; token++)
		{
			probability = 0.0f;
			for (element = 0u; element < SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION); element++)
				probability += qh[element] * k_cache[((uint64_t)token * SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION)) + element];
			scores[token] = probability * scale;
			if (scores[token] > maximum)
				maximum = scores[token];
		}
		total = 0.0f;
		for (token = 0u; token < tokens; token++)
		{
			scores[token] = expf(scores[token] - maximum);
			total += scores[token];
		}
		for (element = 0u; element < SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION); element++)
		{
			probability = 0.0f;
			for (token = 0u; token < tokens; token++)
				probability += (scores[token] / total) * v_cache[((uint64_t)token * SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION)) + element];
			output[((uint64_t)head * SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION)) + element] =
				probability * (1.0f / (1.0f + expf(-fused[SPARK_FAMILY_CONST(MODEL_ATTN_HEAD_DIMENSION) + element])));
		}
	}
}

static int SPARK_FAMILY(ValCheckFinite)(const char *check, const uint16_t *hidden, uint64_t rows)
{
	uint64_t index,count = rows * SPARK_FAMILY_CONST(MODEL_HIDDEN_DIMENSION);
	for (index = 0u; index < count; index++)
		if (isfinite(SPARK_FAMILY(ValFromBf16)(hidden[index])) == 0)
			return(SPARK_FAMILY(ValFail)(check,"nonfinite"));
	return(0);
}
