#pragma once

static float SPARK_FAMILY(ValE2m1Decode)(uint8_t nibble)
{
	static const float magnitudes[8] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};
	float magnitude = magnitudes[nibble & 7u];
	return((nibble & 8u) != 0u ? -magnitude : magnitude);
}

static uint64_t SPARK_FAMILY(ValPayloadRowBytes)(uint32_t codec,uint32_t columns)
{
	return(((uint64_t)columns * SPARK_FAMILY(ValCodecStoredBits)(codec) + 7u) / 8u);
}

static float SPARK_FAMILY(ValSigmoid)(float value)
{
	return(1.0f / (1.0f + expf(-value)));
}

static void *SPARK_FAMILY(ValAllocZeroed)(uint64_t bytes)
{
	void *pointer;
	if ( cudaMalloc(&pointer,bytes != 0u ? bytes : 16u) != cudaSuccess )
		return(0);
	if ( cudaMemset(pointer,0,bytes != 0u ? bytes : 16u) != cudaSuccess )
	{
		cudaFree(pointer);
		return(0);
	}
	return(pointer);
}
