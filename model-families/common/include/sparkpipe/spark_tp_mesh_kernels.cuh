#pragma once
#include <stdint.h>
#include <stddef.h>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define SPARK_TP_MESH_THREADS 256u

static __device__ __forceinline__ unsigned long long SparkTpMeshGlobalTimerNs()
{
	unsigned long long t;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
	return t;
}

__global__ void SparkGlm5NextMeshPublishKernel(
	volatile uint64_t *entry,
	unsigned long long *seq_cell,
	unsigned long long *round_seq,
	uint64_t bytes,
	uint64_t slot_index)
{
	unsigned long long sequence;
	if ( threadIdx.x != 0u || blockIdx.x != 0u )
		return;
	sequence = 1ull + atomicAdd((unsigned long long *)seq_cell,1ull);
	round_seq[0] = sequence;
	entry[2] = slot_index;
	entry[1] = bytes;
	__threadfence_system();
	entry[0] = sequence;
}

__global__ void SparkGlm5NextMeshGuardKernel(
    volatile unsigned long long *error_word,
    unsigned long long *output)
{
	if ( threadIdx.x != 0u || blockIdx.x != 0u )
		return;
	if ( *error_word != 0ull )
	{
		output[0] = 0xFFFFFFFFFFFFFFFFull;
		*error_word = 0ull;
		printf("MESH-GUARD-POISON\\n");
	}
}

__global__ void SparkGlm5NextMeshWaitKernel(
	volatile uint64_t *band_base,
	uint64_t slot_bytes,
	const unsigned long long *round_seq,
	uint64_t slots_per_rank,
	uint64_t ring,
	uint32_t rank,
	uint32_t degree,
	unsigned long long *error_word,
	unsigned long long deadline_ns)
{
	uint32_t peer;
	volatile uint64_t *end_word;
	uint64_t sequence;
	unsigned long long stop_at;
	if ( threadIdx.x != 0u || blockIdx.x != 0u )
		return;
	sequence = round_seq[0];
	stop_at = SparkTpMeshGlobalTimerNs() + deadline_ns;
	for ( peer = 0u; peer < degree - 1u; peer++ )
	{
		uint32_t peer_rank = peer < rank ? peer : peer + 1u;
		end_word = (volatile uint64_t *)
			((uint8_t *)band_base +
			((uint64_t)peer_rank * slots_per_rank +
				(ring & (slots_per_rank - 1ull))) * slot_bytes +
			slot_bytes - 8u);
		while ( *end_word < sequence )
		{
			if ( SparkTpMeshGlobalTimerNs() >= stop_at )
			{
				atomicExch((unsigned long long *)error_word,sequence);
				return;
			}
			__nanosleep(200u);
		}
	}
}


static __device__ __forceinline__ float2 SparkGlm5NextLoadBf16Pair(const void *base,uint64_t element)
{
	uint32_t packed = ((const uint32_t *)base)[element];
	float2 pair;
	pair.x = __int_as_float((int32_t)((packed & UINT32_C(0x0000ffff)) << 16u));
	pair.y = __int_as_float((int32_t)(packed & UINT32_C(0xffff0000)));
	return(pair);
}

static __device__ __forceinline__ void SparkGlm5NextStoreBf16Pair(void *base,uint64_t element,float x,float y)
{
	uint32_t packed = ((uint32_t)(__float_as_int(y) & 0xffff0000u)) |
	    (uint32_t)((__float_as_int(x) >> 16) & 0x0000ffffu);
	((uint32_t *)base)[element] = packed;
}

static __device__ __forceinline__ unsigned long long SparkGlm5NextGlobalTimerNs(void)
{
	unsigned long long ns;
	asm volatile("mov.u64 %%nsec, %%globaltimer;" : "=l"(ns));
	return ns;
}


struct SparkGlm5NextRankSources
{
	const void *pointer[16u];
};

static __global__ void SparkGlm5NextSumRanksF32Kernel(
    void *destination_bf16,
    SparkGlm5NextRankSources sources,
    uint32_t source_count,
    uint32_t pair_count)
{
	uint32_t pair;
	float2 acc,v;
	for (pair=threadIdx.x; pair<pair_count; pair+=blockDim.x)
	{
		uint32_t source;
		acc.x = 0.0f;
		acc.y = 0.0f;
		for ( source = 0u; source < source_count; source++ )
		{
			v = SparkGlm5NextLoadBf16Pair(sources.pointer[source],pair);
			acc.x += v.x;
			acc.y += v.y;
		}
		SparkGlm5NextStoreBf16Pair(destination_bf16,pair,acc.x,acc.y);
	}
}

extern "C" cudaError_t SparkGlm5NextLaunchSumRanksF32(cudaStream_t stream,
    void *destination,const void *const *sources,uint32_t source_count,
    uint32_t element_count)
{
	SparkGlm5NextRankSources by_value;
	uint32_t index;
	if ( destination == 0 || sources == 0 || source_count == 0u ||
	     source_count > 16u || element_count == 0u )
		return(cudaErrorInvalidValue);
	for ( index = 0u; index < source_count; index++ )
		by_value.pointer[index] = sources[index];
	{
		dim3 grid;
		uint32_t pairs = (element_count + 1u) / 2u;
		uint32_t rows = (pairs + 255u) / 256u;
		grid = dim3(rows < 1u ? 1u : rows,1u,1u);
		SparkGlm5NextSumRanksF32Kernel<<<grid,256u,0u,stream>>>(
		    destination,by_value,source_count,pairs);
	}
	return cudaPeekAtLastError();
}

static __global__ void SparkGlm5NextSeedF32Kernel(
    float *destination_f32,
    const void *source_a_bf16,
    const void *source_b_bf16,
    uint32_t pair_count)
{
	uint32_t pair;
	float2 a,b;
	for (pair=threadIdx.x; pair<pair_count; pair+=blockDim.x)
	{
		a = SparkGlm5NextLoadBf16Pair(source_a_bf16,pair);
		b = SparkGlm5NextLoadBf16Pair(source_b_bf16,pair);
		destination_f32[2u * pair] = a.x + b.x;
		destination_f32[2u * pair + 1u] = a.y + b.y;
	}
}

static __global__ void SparkGlm5NextAddF32Kernel(
    float *destination_f32,
    const void *source_bf16,
    uint32_t pair_count)
{
	uint32_t pair;
	float2 b;
	for (pair=threadIdx.x; pair<pair_count; pair+=blockDim.x)
	{
		b = SparkGlm5NextLoadBf16Pair(source_bf16,pair);
		destination_f32[2u * pair] += b.x;
		destination_f32[2u * pair + 1u] += b.y;
	}
}

static __global__ void SparkGlm5NextRoundF32Kernel(
    void *destination_bf16,
    const float *source_f32,
    uint32_t pair_count)
{
	uint32_t pair;
	float2 v;
	for (pair=threadIdx.x; pair<pair_count; pair+=blockDim.x)
	{
		v.x = source_f32[2u * pair];
		v.y = source_f32[2u * pair + 1u];
		SparkGlm5NextStoreBf16Pair(destination_bf16,pair,v.x,v.y);
	}
}

extern "C" cudaError_t SparkGlm5NextLaunchSeedF32(cudaStream_t stream,
    float *destination,const void *a,const void *b,uint32_t element_count)
{
	SparkGlm5NextSeedF32Kernel<<<1,SPARK_TP_MESH_THREADS,0u,stream>>>(
	    destination,a,b,(element_count + 1u) / 2u);
	return cudaPeekAtLastError();
}

extern "C" cudaError_t SparkGlm5NextLaunchAddF32(cudaStream_t stream,
    float *destination,const void *b,uint32_t element_count)
{
	SparkGlm5NextAddF32Kernel<<<1,SPARK_TP_MESH_THREADS,0u,stream>>>(
	    destination,b,(element_count + 1u) / 2u);
	return cudaPeekAtLastError();
}

extern "C" cudaError_t SparkGlm5NextLaunchRoundF32(cudaStream_t stream,
    void *destination,const float *source,uint32_t element_count)
{
	SparkGlm5NextRoundF32Kernel<<<1,SPARK_TP_MESH_THREADS,0u,stream>>>(
	    destination,source,(element_count + 1u) / 2u);
	return cudaPeekAtLastError();
}

static __global__ void SparkGlm5NextAccumAddKernel(
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
		destination_pair = SparkGlm5NextLoadBf16Pair(destination_bf16,offset + element);
		source_pair = SparkGlm5NextLoadBf16Pair(source_bf16,offset + element);
		SparkGlm5NextStoreBf16Pair(destination_bf16,offset + element,destination_pair.x + source_pair.x,destination_pair.y + source_pair.y);
	}
}

static __global__ void SparkGlm5NextAccumU64MaxKernel(
	uint64_t *destination,
	const uint64_t *source,
	uint32_t element_count)
{
	uint32_t element;
	element = blockIdx.x * blockDim.x + threadIdx.x;
	if ( element < element_count && source[element] > destination[element] )
		destination[element] = source[element];
}

extern "C" cudaError_t SparkGlm5NextLaunchAccumAdd(cudaStream_t stream,void *destination_bf16,const void *source_bf16,uint32_t row_count,uint32_t width)
{
	if ( destination_bf16 == 0 || source_bf16 == 0 || row_count == 0u || width == 0u || (width & 1u) != 0u )
		return(cudaErrorInvalidValue);
	SparkGlm5NextAccumAddKernel<<<row_count,256u,0u,stream>>>(destination_bf16,source_bf16,row_count,width);
	return(cudaPeekAtLastError());
}

extern "C" cudaError_t SparkGlm5NextLaunchAccumU64Max(cudaStream_t stream,uint64_t *destination,const uint64_t *source,uint32_t element_count)
{
	if ( destination == 0 || source == 0 || element_count == 0u )
		return(cudaErrorInvalidValue);
	SparkGlm5NextAccumU64MaxKernel<<<(element_count + 255u) / 256u,256u,0u,stream>>>(destination,source,element_count);
	return(cudaPeekAtLastError());
}

extern "C" cudaError_t SparkGlm5NextLaunchMeshGuard(cudaStream_t stream,
	volatile void *error_word,void *output)
{
	SparkGlm5NextMeshGuardKernel<<<1,32,0u,stream>>>(
		(volatile unsigned long long *)error_word,
		(unsigned long long *)output);
	return cudaPeekAtLastError();
}

extern "C" cudaError_t SparkGlm5NextLaunchMeshPublish(cudaStream_t stream,
	volatile void *entry,void *seq_cell,void *round_seq,uint64_t bytes,
	uint64_t slot_index)
{
	if ( entry == 0 || seq_cell == 0 || round_seq == 0 )
		return(cudaErrorInvalidValue);
	SparkGlm5NextMeshPublishKernel<<<1,32,0u,stream>>>(
		(volatile uint64_t *)entry,(unsigned long long *)seq_cell,
		(unsigned long long *)round_seq,bytes,slot_index);
	return(cudaPeekAtLastError());
}

extern "C" cudaError_t SparkGlm5NextLaunchMeshWait(cudaStream_t stream,
	volatile void *band_base,uint64_t slot_bytes,const void *round_seq,
	uint64_t slots_per_rank,uint64_t ring,uint32_t rank,uint32_t degree,
	void *error_word,unsigned long long deadline_ns)
{
	if ( band_base == 0 || round_seq == 0 || degree == 0u ||
	     error_word == 0 )
		return(cudaErrorInvalidValue);
	SparkGlm5NextMeshWaitKernel<<<1,32,0u,stream>>>(
		(volatile uint64_t *)band_base,slot_bytes,
		(const unsigned long long *)round_seq,slots_per_rank,ring,rank,
		degree,(unsigned long long *)error_word,deadline_ns);
	return(cudaPeekAtLastError());
}








#endif
