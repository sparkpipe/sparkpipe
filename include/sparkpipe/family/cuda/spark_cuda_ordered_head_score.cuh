#pragma once

static __device__ __forceinline__ uint32_t SPARK_FAMILY(OrderedHeadScore)(float score)
{
	uint32_t bits;
	if ( isnan(score) )
		return(0u);
	if ( score == 0.0f )
		score = 0.0f;
	bits = __float_as_uint(score);
	return(bits ^ ((bits & UINT32_C(0x80000000)) != 0u ?
		UINT32_MAX : UINT32_C(0x80000000)));
}
