#pragma once

/* Shared stage-module device helpers. These are model-independent CUDA
 * primitives every resident decode stage compiles into its own translation
 * unit - they live here so N families cannot grow N private copies of the
 * same body. Static inline kernels: no new link unit, each including TU
 * emits what it uses. Both launchers refuse zero counts and odd widths
 * (bf16 pairs), matching the bodies they replaced verbatim.
 *
 * NUMERIC DISCIPLINE - READ BEFORE MIGRATING ANY ACCUMULATOR ONTO THIS
 * HEADER: the pair store below TRUNCATES each float to its top 16 bits
 * (the discipline of the family this header was minted from). Other
 * families' elementwise accumulators store through SparkLmStoreBf16Pair
 * (spark_lm_kernels.cuh), which ROUNDS TO NEAREST EVEN via
 * __floats2bfloat162_rn. A host measurement over bf16 add pairs shows
 * ~38% of stored patterns differ between the two disciplines, so pointing
 * an RN-discipline family at AccumAddKernel would change its TP all-reduce
 * results. Unifying the disciplines requires an owner rounding-policy
 * ruling plus bitwise-receipt regeneration; until then only AccumU64Max
 * (integer, discipline-free) is provably safe to share fleet-wide.
 *
 * Bit-faithfulness note: bf16 pairs are reinterpreted through the float bit
 * pattern (bf16 lives in the top 16 bits of a float), never integer-
 * converted - identical arithmetic to the private kernels this header
 * replaces, within their store discipline, so TP all-reduce results are
 * unchanged for consumers of this header.
 */

#include <cuda_runtime.h>
#include <stdint.h>

#ifndef SPARK_STAGE_KERNEL_THREADS
#define SPARK_STAGE_KERNEL_THREADS 256u
#endif

namespace spark_stage_kernels
{

__device__ __forceinline__ float2 LoadBf16Pair(const void *base,uint64_t element)
{
	/* Little-endian pair layout: x occupies the LOW 16 bits, y the HIGH. */
	uint32_t packed = ((const uint32_t *)base)[element];
	float2 pair;
	pair.x = __int_as_float((int32_t)((packed & UINT32_C(0x0000ffff)) << 16u));
	pair.y = __int_as_float((int32_t)(packed & UINT32_C(0xffff0000)));
	return(pair);
}

__device__ __forceinline__ void StoreBf16Pair(void *base,uint64_t element,float x,float y)
{
	uint32_t packed = (__float_as_uint(y) & UINT32_C(0xffff0000)) |
		(__float_as_uint(x) >> 16u);
	((uint32_t *)base)[element] = packed;
}

inline __global__ void AccumAddKernel(
	void *destination_bf16,
	const void *source_bf16,
	uint32_t row_count,
	uint32_t width)
{
	uint32_t row = blockIdx.x,element;
	uint64_t offset = ((uint64_t)row * width) >> 1u;
	float2 destination_pair,source_pair;
	if ( row >= row_count )
		return;
	for (element=threadIdx.x; element<(width >> 1u); element+=blockDim.x)
	{
		destination_pair = LoadBf16Pair(destination_bf16,offset + element);
		source_pair = LoadBf16Pair(source_bf16,offset + element);
		StoreBf16Pair(destination_bf16,offset + element,destination_pair.x + source_pair.x,destination_pair.y + source_pair.y);
	}
}

inline __global__ void AccumU64MaxKernel(
	uint64_t *destination,
	const uint64_t *source,
	uint32_t element_count)
{
	uint32_t element;
	element = blockIdx.x * blockDim.x + threadIdx.x;
	if ( element < element_count && source[element] > destination[element] )
		destination[element] = source[element];
}

} /* namespace spark_stage_kernels */

/* Inline launchers (nvcc-compiled); family TUs expose them to host C
 * through the neutral extern "C" names prototyped in
 * runtime/stage_module_kernels.h. */
#ifdef __cplusplus
extern "C" {
#endif

inline cudaError_t SparkStageLaunchAccumAddInline(cudaStream_t stream,void *destination_bf16,const void *source_bf16,uint32_t row_count,uint32_t width)
{
	if ( destination_bf16 == 0 || source_bf16 == 0 || row_count == 0u || width == 0u || (width & 1u) != 0u )
		return(cudaErrorInvalidValue);
	spark_stage_kernels::AccumAddKernel<<<row_count,SPARK_STAGE_KERNEL_THREADS,0u,stream>>>(destination_bf16,source_bf16,row_count,width);
	return(cudaPeekAtLastError());
}

inline cudaError_t SparkStageLaunchAccumU64MaxInline(cudaStream_t stream,uint64_t *destination,const uint64_t *source,uint32_t element_count)
{
	if ( destination == 0 || source == 0 || element_count == 0u )
		return(cudaErrorInvalidValue);
	spark_stage_kernels::AccumU64MaxKernel<<<(element_count + SPARK_STAGE_KERNEL_THREADS - 1u) / SPARK_STAGE_KERNEL_THREADS,SPARK_STAGE_KERNEL_THREADS,0u,stream>>>(destination,source,element_count);
	return(cudaPeekAtLastError());
}

#ifdef __cplusplus
}
#endif
