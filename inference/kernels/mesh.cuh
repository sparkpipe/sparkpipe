#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ unsigned long long LmMeshGlobalTimerNs()
{
	unsigned long long t;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
	return t;
}

__global__ inline void LmMeshPublishKernel(
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

__global__ inline void LmMeshGuardKernel(
	volatile unsigned long long *error_word,
	unsigned long long *output)
{
	if ( threadIdx.x != 0u || blockIdx.x != 0u )
		return;
	if ( *error_word != 0ull )
	{
		output[0] = 0xFFFFFFFFFFFFFFFFull;
		*error_word = 0ull;
		printf("MESH-GUARD-POISON\n");
	}
}

__global__ inline void LmMeshWaitKernel(
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
	stop_at = LmMeshGlobalTimerNs() + deadline_ns;
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
			if ( LmMeshGlobalTimerNs() >= stop_at )
			{
				atomicExch((unsigned long long *)error_word,sequence);
				return;
			}
			__nanosleep(200u);
		}
	}
}

static inline cudaError_t LmMeshLaunchPublish(cudaStream_t stream,
	volatile void *entry,void *seq_cell,void *round_seq,uint64_t bytes,
	uint64_t slot_index)
{
	if ( entry == 0 || seq_cell == 0 || round_seq == 0 )
		return(cudaErrorInvalidValue);
	LmMeshPublishKernel<<<1,32,0u,stream>>>(
		(volatile uint64_t *)entry,(unsigned long long *)seq_cell,
		(unsigned long long *)round_seq,bytes,slot_index);
	return(cudaPeekAtLastError());
}

static inline cudaError_t LmMeshLaunchGuard(cudaStream_t stream,
	volatile void *error_word,void *output)
{
	LmMeshGuardKernel<<<1,32,0u,stream>>>(
		(volatile unsigned long long *)error_word,
		(unsigned long long *)output);
	return(cudaPeekAtLastError());
}

static inline cudaError_t LmMeshLaunchWait(cudaStream_t stream,
	volatile void *band_base,uint64_t slot_bytes,const void *round_seq,
	uint64_t slots_per_rank,uint64_t ring,uint32_t rank,uint32_t degree,
	void *error_word,unsigned long long deadline_ns)
{
	if ( band_base == 0 || round_seq == 0 || degree == 0u ||
	     error_word == 0 )
		return(cudaErrorInvalidValue);
	LmMeshWaitKernel<<<1,32,0u,stream>>>(
		(volatile uint64_t *)band_base,slot_bytes,
		(const unsigned long long *)round_seq,slots_per_rank,ring,rank,
		degree,(unsigned long long *)error_word,deadline_ns);
	return(cudaPeekAtLastError());
}
