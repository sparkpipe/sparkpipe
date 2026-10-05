#pragma once

static uint32_t SPARK_FAMILY(ValNext)(void)
{
	uint32_t value = SPARK_FAMILY(ValRandomState);
	value ^= value << 13;
	value ^= value >> 17;
	value ^= value << 5;
	SPARK_FAMILY(ValRandomState) = value;
	return(value);
}

static uint16_t SPARK_FAMILY(ValBf16)(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	uint32_t lsb = (bits >> 16) & 1u;
	uint32_t rounded = (bits + 0x7fffu + lsb) >> 16;
	return((uint16_t)(rounded & 0xffffu));
}

static void SPARK_FAMILY(ValFill)(uint16_t *packed,float *exact,uint64_t count,float scale)
{
	uint64_t index;
	for (index = 0u; index < count; index++)
	{
		float value = (((float)(int32_t)(SPARK_FAMILY(ValNext)() & 0xffffu) -
			32768.0f) / 32768.0f) * scale;
		exact[index] = value;
		packed[index] = SPARK_FAMILY(ValBf16)(value);
	}
}

static void SPARK_FAMILY(ValFillNorm)(uint16_t *packed,float *exact,uint64_t count)
{
	uint64_t index;
	for (index = 0u; index < count; index++)
	{
		exact[index] = 1.0f;
		packed[index] = SPARK_FAMILY(ValBf16)(1.0f);
	}
}

static float SPARK_FAMILY(ValE4m3Decode)(uint8_t code)
{
	int32_t sign = (code & 0x80u) != 0u ? -1 : 1;
	uint32_t exponent = (code >> 3) & 0xfu;
	uint32_t mantissa = code & 7u;
	float value;
	if ( exponent == 0u )
		return((float)sign * ((float)mantissa * (1.0f / 512.0f)));
	if ( exponent == 15u && mantissa == 7u )
		return((float)sign * NAN);
	value = (1.0f + ((float)mantissa) * 0.125f) *
		(float)(1u << (int32_t)(exponent - 7u < 31u ? exponent - 7u : 0u));
	if ( exponent < 7u )
		value = (1.0f + ((float)mantissa) * 0.125f) /
			(float)(1u << (7u - exponent));
	return((float)sign * value);
}

static float SPARK_FAMILY(ValBoundedDecay)(float logit,float bias,float head_log_scale,float lower_bound)
{
	return(expf(lower_bound * SPARK_FAMILY(ValSigmoid)(expf(head_log_scale) * (logit + bias))));
}

static void SPARK_FAMILY(ValRmsNorm)(float *row,const float *weight,uint32_t dimension,float epsilon)
{
	float sum = 0.0f;
	uint32_t index;
	for (index = 0u; index < dimension; index++)
		sum += row[index] * row[index];
	float inverse = 1.0f / sqrtf(sum / (float)dimension + epsilon);
	for (index = 0u; index < dimension; index++)
		row[index] = row[index] * inverse * weight[index];
}

static int SPARK_FAMILY(ValSelftestAssert)(int condition,const char *what)
{
	if (!condition)
	{
		printf("FAIL selftest: %s\n",what);
		return(1);
	}
	return(0);
}
