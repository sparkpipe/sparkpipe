#pragma once

#include "dtype.cuh"

__global__ void LmHcPostBf16Kernel(const uint16_t *out_bf16,const uint16_t *snapshot_bf16,const float *post_f32,const float *comb_f32,uint16_t *streams_bf16,uint32_t rows,uint32_t hc,uint32_t width)
{
	__shared__ float post[4],comb[16];
	uint32_t row = blockIdx.x,element,stream,source,index;
	float residual[4],out,value,contribution;
	if ( row >= rows || hc == 0u || hc > 4u )
		return;
	for (index=threadIdx.x; index<hc; index+=blockDim.x)
		post[index] = LmBf16ToFloat(LmFloatToBf16(post_f32[((uint64_t)row * hc) + index]));
	for (index=threadIdx.x; index<(hc * hc); index+=blockDim.x)
		comb[index] = LmBf16ToFloat(LmFloatToBf16(comb_f32[((uint64_t)row * hc * hc) + index]));
	__syncthreads();
	for (element=threadIdx.x; element<width; element+=blockDim.x)
	{
		out = LmBf16ToFloat(out_bf16[((uint64_t)row * width) + element]);
		for (source=0u; source<hc; source++)
			residual[source] = LmBf16ToFloat(snapshot_bf16[((((uint64_t)row * hc) + source) * width) + element]);
		for (stream=0u; stream<hc; stream++)
		{
			value = 0.0f;
			for (source=0u; source<hc; source++)
				value = fmaf(comb[(source * hc) + stream],residual[source],value);
			value = LmBf16ToFloat(LmFloatToBf16(value));
			contribution = LmBf16ToFloat(LmFloatToBf16(post[stream] * out));
			streams_bf16[((((uint64_t)row * hc) + stream) * width) + element] = LmFloatToBf16(contribution + value);
		}
	}
}
