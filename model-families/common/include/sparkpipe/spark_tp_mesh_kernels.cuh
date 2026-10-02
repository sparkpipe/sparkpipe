#pragma once
#include <stdint.h>
#include <stddef.h>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#include <cuda.h>
#include "sparkpipe/spark_weightd.h"
#include <stdio.h>
#include "sparkpipe/spark_tp_mesh_round_control.h"
#define SPARK_TP_MESH_KERNELS_MARKER "SPARK-TP-MESH-KERNELS-V13-POLLED-GATES"
#define SPARK_TP_MESH_ERROR_PARITY_MISMATCH 0xFFFFFFFFFF000000ull
#define SPARK_TP_MESH_ERROR_CANCELLED 0xFFFFFFFFFE000000ull
#if defined(__CUDACC__)
__constant__ char SparkTpMeshKernelsBuildMarker[] =
    SPARK_TP_MESH_KERNELS_MARKER;
/* Host-side twin with __attribute__((used)): nvcc dead-strips the
 * unreferenced __constant__ into the compressed fatbin where strings
 * cannot see it, tripping the publish guard. The host pass (gcc)
 * compiles this file-scope copy into .rodata unconditionally, from the
 * SAME header and macro — a stale private copy still fails the check. */
__attribute__((used)) static const char SparkTpMeshKernelsBuildMarkerHost[] =
    SPARK_TP_MESH_KERNELS_MARKER;
#endif

#define SPARK_TP_MESH_THREADS 256u

static __device__ __forceinline__ unsigned long long SparkTpLdcvU64(
    const volatile void *address)
{
	unsigned long long value;
	asm volatile("ld.global.cv.u64 %0,[%1];"
	    : "=l"(value) : "l"(address) : "memory");
	return(value);
}

static __device__ __forceinline__ unsigned long long SparkTpLdAcquireU64(
    const volatile void *address)
{
	unsigned long long value;
	asm volatile("ld.acquire.sys.global.u64 %0,[%1];"
	    : "=l"(value) : "l"(address) : "memory");
	return(value);
}

static __device__ __forceinline__ unsigned long long SparkTpGlobalTimerNs(void)
{
	unsigned long long ns;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns));
	return ns;
}


__global__ void SparkTpMeshPublishKernel(
    volatile uint64_t *entry,
    unsigned long long *seq_cell,
    const unsigned long long *epoch_cell,
    unsigned long long *round_seq,
    uint64_t bytes,
    uint64_t slot_index,
    uint64_t slots_per_rank,
    volatile uint64_t *slot_tail,
    unsigned long long *error_word,uint32_t peer_mask)
{
	unsigned long long sequence;
	unsigned long long tag;
	uint64_t ring;
	if ( threadIdx.x != 0u || blockIdx.x != 0u )
		return;
	if ( SparkTpLdcvU64(error_word) != 0ull )
		return;
	sequence = 1ull + atomicAdd((unsigned long long *)seq_cell,1ull);
	ring = (sequence - 1ull) & (slots_per_rank - 1ull);
	if ( (slot_index & (slots_per_rank - 1ull)) != ring )
	{
		atomicExch((unsigned long long *)error_word,
		    SPARK_TP_MESH_ERROR_PARITY_MISMATCH | sequence);
		return;
	}
	tag = (epoch_cell[0] << 32ull) | (sequence & 0xffffffffull);
	round_seq[0] = tag;
	entry[2] = slot_index;
	entry[1] = bytes;
	entry[3] = peer_mask;
	__threadfence_system();
	*slot_tail = tag;
	__threadfence_system();
	entry[0] = tag;
}

__global__ void SparkTpMeshSeqPadKernel(
    unsigned long long *seq_cell)
{
	if ( threadIdx.x != 0u || blockIdx.x != 0u )
		return;
	(void)atomicAdd((unsigned long long *)seq_cell,1ull);
}

__global__ void SparkTpMeshGuardKernel(
    volatile unsigned long long *error_word,
    unsigned long long *output)
{
	if ( threadIdx.x != 0u || blockIdx.x != 0u )
		return;
	if ( *error_word != 0ull )
	{
		output[0] = 0xFFFFFFFFFFFFFFFFull;
		printf("MESH-GUARD-POISON\\n");
	}
}

#define SPARK_TP_MESH_WAIT_SLEEP_NS 200u
#define SPARK_TP_MESH_WAIT_SPINS_BEFORE_SLEEP 1024ull

__global__ void SparkTpMeshWaitKernel(
    volatile uint64_t *band_base,
    uint64_t slot_bytes,
    const unsigned long long *round_seq,
    uint64_t slots_per_rank,
    uint32_t rank,
    uint32_t degree,
    unsigned long long *error_word,
    unsigned long long deadline_ns,
    unsigned long long *diag_word,
    volatile uint64_t *cancel_cell,
    const unsigned long long *cancel_expected,
    unsigned long long *arrival_ring)
{
	uint32_t peer;
	volatile uint64_t *end_word;
	uint64_t sequence;
	uint64_t expected_cancel;
	uint64_t ring;
	unsigned long long stop_at;
	if ( threadIdx.x != 0u || blockIdx.x != 0u )
		return;
	expected_cancel = cancel_expected != 0 ? SparkTpLdcvU64(cancel_expected) : 0u;
	sequence = SparkTpLdcvU64(round_seq);
	ring = (sequence - 1ull) & (slots_per_rank - 1ull);
	stop_at = SparkTpGlobalTimerNs() + deadline_ns;
	if ( cancel_cell != 0 && cancel_expected != 0 &&
	     SparkTpLdcvU64(cancel_cell) != expected_cancel )
	{
		atomicExch((unsigned long long *)error_word,
		    SPARK_TP_MESH_ERROR_CANCELLED | sequence);
		return;
	}
	{
		unsigned long long spins = 0ull;
		unsigned long long spin_cap = deadline_ns / 200ull;
		if ( spin_cap < 1000000ull )
			spin_cap = 1000000ull;
		for ( peer = 0u; peer < degree - 1u; peer++ )
		{
			uint32_t peer_rank = peer < rank ? peer : peer + 1u;
			uint64_t peer_epoch;
			end_word = (volatile uint64_t *)
				((uint8_t *)band_base +
				((uint64_t)peer_rank * slots_per_rank +
					ring) * slot_bytes +
				slot_bytes - 8u);
			while ( (peer_epoch = SparkTpLdcvU64(end_word) >>
			             32ull) != (sequence >> 32ull) ||
			        SparkTpLdcvU64(end_word) < sequence )
			{
				if ( SparkTpLdcvU64(error_word) != 0ull )
					return;
				if ( cancel_cell != 0 && cancel_expected != 0 &&
				     SparkTpLdcvU64(cancel_cell) !=
				     expected_cancel )
				{
					atomicExch((unsigned long long *)error_word,
					    SPARK_TP_MESH_ERROR_CANCELLED | sequence);
					return;
				}
				spins++;
				if ( (spins & 255ull) == 0ull &&
				     ( spins >= spin_cap ||
				       SparkTpGlobalTimerNs() >= stop_at ) )
				{
					unsigned long long off = (unsigned long long)
						((uint8_t *)end_word - (uint8_t *)band_base);
					unsigned long long got = SparkTpLdcvU64(end_word);
					atomicExch((unsigned long long *)diag_word,
						((unsigned long long)peer_rank << 56ull) |
						((ring & 0xffull) << 48ull) |
						((off / slot_bytes) << 32ull) |
						((sequence & 0xffffull) << 16ull) |
						(got & 0xffffull));
					atomicExch((unsigned long long *)error_word,sequence);
					return;
				}
				if ( spins > SPARK_TP_MESH_WAIT_SPINS_BEFORE_SLEEP )
					__nanosleep(SPARK_TP_MESH_WAIT_SLEEP_NS);
			}
		}
	}
	if ( arrival_ring != 0 && threadIdx.x == 0u && blockIdx.x == 0u )
		arrival_ring[sequence & 255ull] = SparkTpGlobalTimerNs();
}


static __device__ __forceinline__ float2 SparkTpLoadBf16Pair(const void *base,uint64_t element)
{
	uint32_t packed = ((const uint32_t *)base)[element];
	float2 pair;
	pair.x = __int_as_float((int32_t)((packed & UINT32_C(0x0000ffff)) << 16u));
	pair.y = __int_as_float((int32_t)(packed & UINT32_C(0xffff0000)));
	return(pair);
}

static __device__ __forceinline__ void SparkTpStoreBf16Pair(void *base,uint64_t element,float x,float y)
{
	uint32_t packed = ((uint32_t)(__float_as_int(y) & 0xffff0000u)) |
	    (uint32_t)((__float_as_int(x) >> 16) & 0x0000ffffu);
	((uint32_t *)base)[element] = packed;
}

static_assert(sizeof(SparkTpMeshRoundControl) ==
    SPARK_TP_MESH_ROUND_CONTROL_BYTES,"round control layout drift");

__global__ void SparkTpMeshRoundLoopKernel(
	volatile uint64_t *band_base,
	uint64_t slot_bytes,
	uint64_t slots_per_rank,
	volatile uint64_t *entry,
	volatile uint64_t *shipped_cell,
	volatile uint64_t *cancel_cell,
	SparkTpMeshRoundControl *control,
	uint32_t rank,
	uint32_t degree,
	const void *local_device,
	void *full_device,
	uint64_t bytes)
{
	__shared__ uint32_t s_active;
	__shared__ uint32_t s_decision;
	__shared__ uint64_t s_slot;
	__shared__ uint64_t s_parity_slot;
	__shared__ uint64_t s_tag;
	uint32_t tid = threadIdx.x;
	uint32_t nthreads = blockDim.x;
	uint64_t cursor = 0ull;
	uint64_t prev_tag = 0ull;
	uint64_t stop_at = 0ull;
	uint64_t spin_cap;
	uint64_t expected_cancel = control->cancel_expected;
	if ( tid == 0u )
	{
		uint64_t now = SparkTpGlobalTimerNs();
		cursor = control->slot_cursor;
		prev_tag = control->round_seq;
		stop_at = now + control->deadline_ns;
		spin_cap = control->deadline_ns / 200ull;
		if ( spin_cap < 1000000ull )
			spin_cap = 1000000ull;
	}
	for ( ;; )
	{
		if ( tid == 0u )
		{
			s_active = control->rounds_done < control->rounds_total ? 1u : 0u;
			s_decision = SPARK_TP_MESH_ROUND_LOOP_DECISION_GO;
		}
		__syncthreads();
		if ( s_active == 0u )
			break;
		if ( tid == 0u && prev_tag != 0ull )
		{
			uint64_t spins = 0ull;
			while ( *shipped_cell != prev_tag )
			{
				if ( *cancel_cell != expected_cancel )
				{
					s_decision = SPARK_TP_MESH_ROUND_LOOP_DECISION_CANCEL;
					break;
				}
				spins++;
				if ( (spins & 4095ull) == 0ull &&
				     ( spins >= spin_cap ||
				       SparkTpGlobalTimerNs() >= stop_at ) )
				{
					control->diag_word = (0xa5ull << 56ull) |
					    ((prev_tag & 0xffffull) << 16ull) |
					    (*shipped_cell & 0xffffull);
					control->error_word = prev_tag;
					printf("MESH-ROUNDLOOP-TIMEOUT rank=%u phase=ship-ack want=%llu got=%llu\\n",
						rank,(unsigned long long)prev_tag,(unsigned long long)*shipped_cell);
					s_decision = SPARK_TP_MESH_ROUND_LOOP_DECISION_TIMEOUT;
					break;
				}
				__nanosleep(200u);
			}
		}
		__syncthreads();
		if ( s_decision != SPARK_TP_MESH_ROUND_LOOP_DECISION_GO )
			break;
		if ( tid == 0u )
		{
			s_slot = ((uint64_t)rank * slots_per_rank +
			    (cursor & (slots_per_rank - 1ull))) * slot_bytes;
			s_parity_slot =
			    (cursor & (slots_per_rank - 1ull)) * slot_bytes;
			s_tag = (control->epoch << 32ull) |
			    ((control->seq + 1ull) & 0xffffffffull);
		}
		__syncthreads();
		{
			volatile uint64_t *destination = (volatile uint64_t *)
			    ((uint8_t *)band_base + s_slot);
			const uint64_t *source = (const uint64_t *)local_device;
			uint64_t quads = bytes >> 3ull;
			uint64_t quad;
			for ( quad = tid; quad < quads; quad += nthreads )
				destination[quad] = source[quad];
            if ( tid == 0u )
                for ( uint64_t tail = quads * 8u; tail < bytes; tail++ )
                    ((volatile uint8_t *)destination)[tail] = ((const uint8_t *)source)[tail];
			__threadfence_system();
		}
		__syncthreads();
		if ( tid == 0u )
		{
			volatile uint64_t *tail = (volatile uint64_t *)
			    ((uint8_t *)band_base + s_slot + slot_bytes - 8ull);
			entry[2] = s_slot / slot_bytes;
			entry[1] = bytes;
			entry[3] = ((1u << degree) - 1u) & ~(1u << rank);
			control->seq = control->seq + 1ull;
			control->round_seq = s_tag;
			__threadfence_system();
			*tail = s_tag;
			__threadfence_system();
			entry[0] = s_tag;
		}
		if ( tid == 0u )
		{
			uint32_t peer;
			uint64_t spins = 0ull;
			for ( peer = 0u; peer < degree - 1u; peer++ )
			{
				uint32_t peer_rank = peer < rank ? peer : peer + 1u;
				volatile uint64_t *end_word = (volatile uint64_t *)
				    ((uint8_t *)band_base +
				    (((uint64_t)peer_rank * slots_per_rank +
				      (cursor & (slots_per_rank - 1ull))) *
					slot_bytes) + slot_bytes - 8ull);
				while ( (*end_word >> 32ull) != (s_tag >> 32ull) ||
				        *end_word < s_tag )
				{
					if ( *cancel_cell != expected_cancel )
					{
						s_decision =
						    SPARK_TP_MESH_ROUND_LOOP_DECISION_CANCEL;
						break;
					}
					spins++;
					if ( (spins & 4095ull) == 0ull &&
					     ( spins >= spin_cap ||
					       SparkTpGlobalTimerNs() >= stop_at ) )
					{
						control->diag_word =
						    ((unsigned long long)peer_rank << 56ull) |
						    ((cursor & 0xffull) << 48ull) |
						    ((unsigned long long)
							(s_slot / slot_bytes) << 32ull) |
						    ((s_tag & 0xffffull) << 16ull) |
						    (*end_word & 0xffffull);
						control->error_word = s_tag;
						printf("MESH-ROUNDLOOP-TIMEOUT rank=%u phase=peer-wait peer=%u want=%u got=%u\\n",
						    rank,peer_rank,
						    (uint32_t)s_tag,
						    (uint32_t)*end_word);
						s_decision =
						    SPARK_TP_MESH_ROUND_LOOP_DECISION_TIMEOUT;
						break;
					}
					__nanosleep(200u);
				}
				if ( s_decision != SPARK_TP_MESH_ROUND_LOOP_DECISION_GO )
					break;
			}
		}
		__syncthreads();
		if ( s_decision != SPARK_TP_MESH_ROUND_LOOP_DECISION_GO )
			break;
		__threadfence();
		{
			uint64_t pairs = bytes >> 2ull;
			uint64_t pair;
			uint32_t source;
			for ( pair = tid; pair < pairs; pair += nthreads )
			{
				float acc_x = 0.0f;
				float acc_y = 0.0f;
				for ( source = 0u; source < degree; source++ )
				{
					float2 part = SparkTpLoadBf16Pair(
					    (const void *)((uint8_t *)band_base +
					    (uint64_t)source * slots_per_rank *
					        slot_bytes + s_parity_slot),pair);
					acc_x += part.x;
					acc_y += part.y;
				}
				SparkTpStoreBf16Pair(full_device,pair,acc_x,acc_y);
			}
		}
		__syncthreads();
		if ( tid == 0u )
		{
			cursor = cursor + 1ull;
			prev_tag = s_tag;
			control->slot_cursor = cursor;
			control->rounds_done = control->rounds_done + 1ull;
		}
		__syncthreads();
	}
}

static __device__ bool SparkTpMeshTreeWait(
    const volatile uint64_t *cell,uint64_t tag,
    const volatile uint64_t *cancel,SparkTpMeshRoundControl *control,
    uint64_t deadline,uint64_t expected_cancel)
{
    for (;;)
    {
        if ( SparkTpLdcvU64(&control->error_word) != 0u ) return false;
        if ( SparkTpLdcvU64(cancel) != expected_cancel )
        {
            control->error_word = SPARK_TP_MESH_ERROR_CANCELLED | tag;
            return false;
        }
        if ( tag == 0u || SparkTpLdcvU64(cell) == tag ) return true;
        if ( SparkTpGlobalTimerNs() >= deadline )
        {
            control->error_word = tag;
            control->diag_word = SparkTpLdcvU64(cell);
            return false;
        }
        __nanosleep(200u);
    }
}

static __device__ void SparkTpMeshTreeSeed(
    void *scratch,const void *local,uint64_t elements,uint64_t begin,uint64_t count,
    uint32_t operation,uint32_t rank,uint32_t degree)
{
    for ( uint64_t i = threadIdx.x; i < count; i += blockDim.x )
    {
        uint64_t index = begin + i;
        if ( operation == 1u )
            ((float *)scratch)[i] = __uint_as_float((uint32_t)((const uint16_t *)local)[index] << 16u);
        else if ( operation == 2u )
            ((uint64_t *)scratch)[i] = ((const uint64_t *)local)[index];
        else
        {
            uint32_t owner = (uint32_t)(index / (elements / degree));
            uint64_t source = index % (elements / degree);
            ((uint16_t *)scratch)[i] = owner == rank ? ((const uint16_t *)local)[source] : 0u;
        }
    }
}

static __device__ void SparkTpMeshTreeFold(
    void *scratch,const void *source,uint64_t count,uint32_t operation,uint32_t reduce)
{
    for ( uint64_t i = threadIdx.x; i < count; i += blockDim.x )
    {
        if ( operation == 1u )
            ((float *)scratch)[i] = reduce != 0u ?
                ((float *)scratch)[i] + ((const volatile float *)source)[i] : ((const volatile float *)source)[i];
        else if ( operation == 2u )
        {
            uint64_t value = ((const volatile uint64_t *)source)[i];
            if ( reduce == 0u || value > ((uint64_t *)scratch)[i] ) ((uint64_t *)scratch)[i] = value;
        }
        else
            ((uint16_t *)scratch)[i] = reduce != 0u ?
                ((uint16_t *)scratch)[i] | ((const volatile uint16_t *)source)[i] : ((const volatile uint16_t *)source)[i];
    }
}

static __device__ void SparkTpMeshTreeFinish(
    void *output,const void *scratch,uint64_t begin,uint64_t count,uint32_t operation)
{
    for ( uint64_t i = threadIdx.x; i < count; i += blockDim.x )
    {
        if ( operation == 1u ) ((uint16_t *)output)[begin + i] = (uint16_t)(__float_as_uint(((const float *)scratch)[i]) >> 16u);
        else if ( operation == 2u ) ((uint64_t *)output)[begin + i] = ((const uint64_t *)scratch)[i];
        else ((uint16_t *)output)[begin + i] = ((const uint16_t *)scratch)[i];
    }
}

__global__ void SparkTpMeshTreeKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    volatile uint64_t *entry,const volatile uint64_t *shipped,
    const volatile uint64_t *cancel,SparkTpMeshRoundControl *control,
    uint32_t rank,uint32_t degree,const void *local,void *output,void *scratch,
    uint64_t elements,uint32_t operation,uint32_t rounds,
    uint64_t timeout_ns)
{
    __shared__ uint32_t ready;
    __shared__ uint64_t tag;
    uint32_t tid = threadIdx.x;
    uint32_t levels = SparkTpMeshTreeLevels(degree);
    uint32_t width = operation == 2u ? 8u : operation == 1u ? 4u : 2u;
    uint64_t capacity = (slot_bytes - 16u) / width;
    uint64_t deadline = SparkTpGlobalTimerNs() + timeout_ns;
    uint64_t expected_cancel = control->cancel_expected;
    for ( uint32_t round = 0u; round < rounds; round++ )
    {
        for ( uint64_t begin = 0u; begin < elements; begin += capacity )
        {
            uint64_t count = elements - begin < capacity ? elements - begin : capacity;
            SparkTpMeshTreeSeed(scratch,local,elements,begin,count,operation,rank,degree);
            __syncthreads();
            for ( uint32_t phase = 0u; phase < 2u * levels; phase++ )
            {
                uint32_t route = SparkTpMeshTreeRoute(rank,degree,phase);
                uint32_t send = route >> 16u;
                uint32_t receive = route & 0xffffu;
                if ( tid == 0u )
                {
                    ready = control->seq < UINT32_MAX && control->error_word == 0u;
                    if ( ready == 0u ) control->error_word = UINT64_MAX;
                    tag = (control->epoch << 32u) | (control->seq + 1u);
                    if ( send != 0u && ready != 0u )
                        ready = SparkTpMeshTreeWait(shipped,control->round_seq,cancel,control,deadline,expected_cancel);
                }
                __syncthreads();
                if ( ready == 0u ) return;
                uint64_t ring = (tag - 1u) & (slots_per_rank - 1u);
                uint64_t own_slot = ((uint64_t)rank * slots_per_rank + ring) * slot_bytes;
                if ( send != 0u )
                {
                    for ( uint64_t i = tid; i < count * width; i += blockDim.x )
                        ((volatile uint8_t *)band)[own_slot + i] = ((const uint8_t *)scratch)[i];
                    __threadfence_system();
                    __syncthreads();
                    if ( tid == 0u )
                    {
                        entry[1] = count * width;
                        entry[2] = own_slot / slot_bytes;
                        entry[3] = 1u << (send - 1u);
                        control->round_seq = tag;
                        __threadfence_system();
                        *(volatile uint64_t *)(band + own_slot + slot_bytes - 8u) = tag;
                        __threadfence_system();
                        entry[0] = tag;
                    }
                }
                if ( receive != 0u )
                {
                    uint8_t *source = band + ((uint64_t)(receive - 1u) * slots_per_rank + ring) * slot_bytes;
                    if ( tid == 0u )
                        ready = SparkTpMeshTreeWait((const volatile uint64_t *)(source + slot_bytes - 8u),tag,cancel,control,deadline,expected_cancel);
                    __syncthreads();
                    if ( ready == 0u ) return;
                    __threadfence_system();
                    SparkTpMeshTreeFold(scratch,source,count,operation,phase < levels);

                }
                __syncthreads();
                if ( tid == 0u ) control->seq++;
                __syncthreads();
            }
            SparkTpMeshTreeFinish(output,scratch,begin,count,operation);
            __syncthreads();
        }
        if ( tid == 0u )
        {
            control->slot_cursor = control->seq;
            control->rounds_done++;
        }
        __syncthreads();
    }
}

extern "C" cudaError_t SparkTpLaunchMeshTree(cudaStream_t stream,
    void *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile void *entry,
    const volatile void *shipped,const volatile void *cancel,void *round_control,
    uint32_t rank,uint32_t degree,const void *local,void *output,void *scratch,
    uint64_t elements,uint32_t operation,uint32_t rounds,uint64_t timeout_ns)
{
    SparkTpMeshTreeKernel<<<1,SPARK_TP_MESH_THREADS,0u,stream>>>(
        (uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,
        (const volatile uint64_t *)shipped,(const volatile uint64_t *)cancel,
        (SparkTpMeshRoundControl *)round_control,rank,degree,local,output,scratch,
        elements,operation,rounds,timeout_ns);
    return cudaPeekAtLastError();
}

extern "C" cudaError_t SparkTpLaunchMeshRoundLoop(cudaStream_t stream,
	volatile void *band_base,uint64_t slot_bytes,uint64_t slots_per_rank,
	volatile void *entry,void *shipped_cell,volatile void *cancel_cell,
	void *round_control,uint32_t rank,uint32_t degree,
	const void *local_device,void *full_device,uint64_t bytes)
{
	SparkTpMeshRoundLoopKernel<<<1,SPARK_TP_MESH_THREADS,0u,stream>>>(
		(volatile uint64_t *)band_base,slot_bytes,slots_per_rank,
		(volatile uint64_t *)entry,(volatile uint64_t *)shipped_cell,
		(volatile uint64_t *)cancel_cell,
		(SparkTpMeshRoundControl *)round_control,rank,degree,
		local_device,full_device,bytes);
	return cudaPeekAtLastError();
}


struct SparkTpRankSources
{
	const void *pointer[16u];
};

static __global__ void SparkTpSumRanksF32Kernel(
    void *destination_bf16,
    SparkTpRankSources sources,
    uint32_t source_count,
    uint32_t pair_count)
{
	uint32_t pair;
	float2 acc,v;
	for (pair=blockIdx.x*blockDim.x+threadIdx.x; pair<pair_count; pair+=blockDim.x*gridDim.x)
	{
		uint32_t source;
		acc.x = 0.0f;
		acc.y = 0.0f;
		for ( source = 0u; source < source_count; source++ )
		{
			v = SparkTpLoadBf16Pair(sources.pointer[source],pair);
			acc.x += v.x;
			acc.y += v.y;
		}
		SparkTpStoreBf16Pair(destination_bf16,pair,acc.x,acc.y);
	}
}

extern "C" cudaError_t SparkTpLaunchSumRanksF32(cudaStream_t stream,
    void *destination,const void *const *sources,uint32_t source_count,
    uint32_t element_count)
{
	SparkTpRankSources by_value;
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
		SparkTpSumRanksF32Kernel<<<grid,256u,0u,stream>>>(
		    destination,by_value,source_count,pairs);
	}
	return cudaPeekAtLastError();
}

static __global__ void SparkTpSeedF32Kernel(
    float *destination_f32,
    const void *source_a_bf16,
    const void *source_b_bf16,
    uint32_t pair_count)
{
	uint32_t pair;
	float2 a,b;
	for (pair=threadIdx.x; pair<pair_count; pair+=blockDim.x)
	{
		a = SparkTpLoadBf16Pair(source_a_bf16,pair);
		b = SparkTpLoadBf16Pair(source_b_bf16,pair);
		destination_f32[2u * pair] = a.x + b.x;
		destination_f32[2u * pair + 1u] = a.y + b.y;
	}
}

static __global__ void SparkTpAddF32Kernel(
    float *destination_f32,
    const void *source_bf16,
    uint32_t pair_count)
{
	uint32_t pair;
	float2 b;
	for (pair=threadIdx.x; pair<pair_count; pair+=blockDim.x)
	{
		b = SparkTpLoadBf16Pair(source_bf16,pair);
		destination_f32[2u * pair] += b.x;
		destination_f32[2u * pair + 1u] += b.y;
	}
}

static __global__ void SparkTpRoundF32Kernel(
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
		SparkTpStoreBf16Pair(destination_bf16,pair,v.x,v.y);
	}
}

extern "C" cudaError_t SparkTpLaunchSeedF32(cudaStream_t stream,
    float *destination,const void *a,const void *b,uint32_t element_count)
{
	SparkTpSeedF32Kernel<<<1,SPARK_TP_MESH_THREADS,0u,stream>>>(
	    destination,a,b,(element_count + 1u) / 2u);
	return cudaPeekAtLastError();
}

extern "C" cudaError_t SparkTpLaunchAddF32(cudaStream_t stream,
    float *destination,const void *b,uint32_t element_count)
{
	SparkTpAddF32Kernel<<<1,SPARK_TP_MESH_THREADS,0u,stream>>>(
	    destination,b,(element_count + 1u) / 2u);
	return cudaPeekAtLastError();
}

extern "C" cudaError_t SparkTpLaunchRoundF32(cudaStream_t stream,
    void *destination,const float *source,uint32_t element_count)
{
	SparkTpRoundF32Kernel<<<1,SPARK_TP_MESH_THREADS,0u,stream>>>(
	    destination,source,(element_count + 1u) / 2u);
	return cudaPeekAtLastError();
}

static __global__ void SparkTpAccumAddKernel(
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
		destination_pair = SparkTpLoadBf16Pair(destination_bf16,offset + element);
		source_pair = SparkTpLoadBf16Pair(source_bf16,offset + element);
		SparkTpStoreBf16Pair(destination_bf16,offset + element,destination_pair.x + source_pair.x,destination_pair.y + source_pair.y);
	}
}

static __global__ void SparkTpAccumU64MaxKernel(
	uint64_t *destination,
	const uint64_t *source,
	uint32_t element_count)
{
	uint32_t element;
	element = blockIdx.x * blockDim.x + threadIdx.x;
	if ( element < element_count && source[element] > destination[element] )
		destination[element] = source[element];
}

extern "C" cudaError_t SparkTpLaunchAccumAdd(cudaStream_t stream,void *destination_bf16,const void *source_bf16,uint32_t row_count,uint32_t width)
{
	if ( destination_bf16 == 0 || source_bf16 == 0 || row_count == 0u || width == 0u || (width & 1u) != 0u )
		return(cudaErrorInvalidValue);
	SparkTpAccumAddKernel<<<row_count,256u,0u,stream>>>(destination_bf16,source_bf16,row_count,width);
	return(cudaPeekAtLastError());
}

extern "C" cudaError_t SparkTpLaunchAccumU64Max(cudaStream_t stream,uint64_t *destination,const uint64_t *source,uint32_t element_count)
{
	if ( destination == 0 || source == 0 || element_count == 0u )
		return(cudaErrorInvalidValue);
	SparkTpAccumU64MaxKernel<<<(element_count + 255u) / 256u,256u,0u,stream>>>(destination,source,element_count);
	return(cudaPeekAtLastError());
}

extern "C" cudaError_t SparkTpLaunchMeshGuard(cudaStream_t stream,
	volatile void *error_word,void *output)
{
	SparkTpMeshGuardKernel<<<1,32,0u,stream>>>(
		(volatile unsigned long long *)error_word,
		(unsigned long long *)output);
	return cudaPeekAtLastError();
}

__global__ void SparkTpMeshCopyDownKernel(
    volatile uint8_t *destination,
    const uint8_t *source,
    uint64_t bytes,
    const volatile uint64_t *shipped,
    SparkTpMeshRoundControl *control,
    const volatile uint64_t *cancel,
    uint64_t timeout_ns)
{
    __shared__ uint32_t ready;
    if ( threadIdx.x == 0u )
    {
        uint64_t previous = control->round_seq;
        uint64_t expected_cancel = control->cancel_expected;
        uint64_t deadline = SparkTpGlobalTimerNs() + timeout_ns;
        ready = 1u;
        while ( previous != 0ull && SparkTpLdcvU64(shipped) != previous )
        {
            if ( SparkTpLdcvU64(&control->error_word) != 0ull ||
                 SparkTpLdcvU64(cancel) != expected_cancel )
            {
                atomicExch((unsigned long long *)&control->error_word,
                    SPARK_TP_MESH_ERROR_CANCELLED | previous);
                ready = 0u;
                break;
            }
            if ( SparkTpGlobalTimerNs() >= deadline )
            {
                atomicExch((unsigned long long *)&control->error_word,previous);
                ready = 0u;
                break;
            }
            __nanosleep(200u);
        }
        if ( SparkTpLdcvU64(cancel) != expected_cancel )
            atomicExch((unsigned long long *)&control->error_word,
                SPARK_TP_MESH_ERROR_CANCELLED | previous);
        if ( SparkTpLdcvU64(&control->error_word) != 0ull )
            ready = 0u;
    }
    __syncthreads();
    if ( ready == 0u )
        return;
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t quads = bytes / 8u;
    if ( i < quads )
        ((volatile uint64_t *)destination)[i] = ((const uint64_t *)source)[i];
    if ( i == quads )
        for ( uint64_t tail = quads * 8u; tail < bytes; tail++ )
            destination[tail] = source[tail];
}

extern "C" cudaError_t SparkTpLaunchMeshCopyDown(
    cudaStream_t stream,void *destination,const void *source,
    uint64_t bytes,const volatile void *shipped,void *round_control,
    const volatile void *cancel,uint64_t timeout_ns)
{
    uint64_t quads = bytes / 8u + 1u;
    if ( destination == 0 || source == 0 || bytes == 0u || shipped == 0 ||
         round_control == 0 || cancel == 0 || timeout_ns == 0u )
        return(cudaErrorInvalidValue);
    SparkTpMeshCopyDownKernel<<<(quads + 255u) / 256u,256u,0u,stream>>>(
        (volatile uint8_t *)destination,(const uint8_t *)source,bytes,
        (const volatile uint64_t *)shipped,(SparkTpMeshRoundControl *)round_control,
        (const volatile uint64_t *)cancel,timeout_ns);
    return(cudaPeekAtLastError());
}

extern "C" cudaError_t SparkTpLaunchMeshPublish(cudaStream_t stream,
    volatile void *entry,void *seq_cell,const void *epoch_cell,
    void *round_seq,uint64_t bytes,
    uint64_t slot_index,uint64_t slots_per_rank,volatile void *slot_tail,
    void *error_word,uint32_t peer_mask)
{
	SparkTpMeshPublishKernel<<<1,32,0u,stream>>>(
		(volatile uint64_t *)entry,(unsigned long long *)seq_cell,
		(const unsigned long long *)epoch_cell,
		(unsigned long long *)round_seq,bytes,slot_index,slots_per_rank,
		(volatile uint64_t *)slot_tail,
		(unsigned long long *)error_word,peer_mask);
	return cudaPeekAtLastError();
}

extern "C" cudaError_t SparkTpLaunchMeshSeqPad(cudaStream_t stream,
    void *seq_cell)
{
	SparkTpMeshSeqPadKernel<<<1,32,0u,stream>>>(
		(unsigned long long *)seq_cell);
	return cudaPeekAtLastError();
}

extern "C" cudaError_t SparkTpLaunchMeshWait(cudaStream_t stream,
    volatile void *band_base,uint64_t slot_bytes,const void *round_seq,
    uint64_t slots_per_rank,uint32_t rank,uint32_t degree,
    void *error_word,unsigned long long deadline_ns,void *diag_word,
    volatile void *cancel_cell,const void *cancel_expected,
    void *arrival_ring)
{
	SparkTpMeshWaitKernel<<<1,32,0u,stream>>>(
		(volatile uint64_t *)band_base,slot_bytes,
		(const unsigned long long *)round_seq,slots_per_rank,rank,
		degree,(unsigned long long *)error_word,deadline_ns,
		(unsigned long long *)diag_word,
		(volatile uint64_t *)cancel_cell,
		(const unsigned long long *)cancel_expected,
		(unsigned long long *)arrival_ring);
	return cudaPeekAtLastError();
}








#define SPARK_TP_MESH_PUBLISH_GUARD 1u
#define SPARK_TP_MESH_PUBLISH_REQUEST 2u
#define SPARK_TP_MESH_PUBLISH_SHIP 4u

static __device__ __forceinline__ uint64_t *SparkTpMeshPipeSeq(SparkTpMeshRoundControl *control,uint32_t pipe)
{
    return pipe != 0u ? &control->second_seq : &control->seq;
}

static __device__ __forceinline__ uint64_t *SparkTpMeshPipeRoundSeq(SparkTpMeshRoundControl *control,uint32_t pipe)
{
    return pipe != 0u ? &control->second_round_seq : &control->round_seq;
}

static __device__ void SparkTpMeshGateRequest(
    volatile SparkWeightdMeshWaitRequest *gate,SparkTpMeshRoundControl *control,
    uint64_t kind,uint64_t peer_mask,uint64_t phase_offset,uint64_t timeout_ns,uint32_t pipe)
{
    uint64_t previous = SparkTpLdcvU64(&gate->request_id);
    uint64_t seq = *SparkTpMeshPipeSeq(control,pipe);
    if ( previous == UINT64_MAX )
    {
        control->error_word = UINT64_MAX;
        return;
    }
    if ( seq > UINT32_MAX - phase_offset ) control->error_word = UINT64_MAX;
    uint64_t tag = kind == SPARK_WEIGHTD_MESH_WAIT_SHIPPED ? *SparkTpMeshPipeRoundSeq(control,pipe) :
        (control->epoch << 32u) | ((seq + phase_offset) & UINT32_MAX);
    control->wait_started_ns = SparkTpGlobalTimerNs();
    gate->ready = 0u;
    gate->kind = kind;
    gate->tag = tag;
    gate->peer_mask = peer_mask;
    gate->cancel_expected = control->cancel_expected;
    gate->timeout_ns = timeout_ns;
    gate->version = SPARK_WEIGHTD_MESH_WAIT_VERSION;
    gate->upstream_error = control->error_word;
    __threadfence_system();
    gate->request_id = previous + 1u;
    __threadfence_system();
}

static __device__ void SparkTpMeshGateAwait(
    const volatile SparkWeightdMeshWaitRequest *gate,SparkTpMeshRoundControl *control)
{
    uint64_t started = SparkTpGlobalTimerNs(),limit = SparkTpLdcvU64(&gate->timeout_ns);
    limit = limit > UINT64_MAX / 2u ? UINT64_MAX : 2u * limit;
    while ( SparkTpLdAcquireU64(&gate->ready) != 1u )
        if ( SparkTpGlobalTimerNs() - started > limit )
        {
            if ( control->error_word == 0u )
            {
                control->error_word = UINT64_MAX;
                control->diag_word = SparkTpLdcvU64(&gate->request_id);
            }
            return;
        }
}

static __device__ void SparkTpMeshGateGuard(
    const volatile SparkWeightdMeshWaitRequest *gate,SparkTpMeshRoundControl *control)
{
    uint64_t error = SparkTpLdcvU64(&gate->error);
    if ( control->error_word == 0u )
    {
        uint64_t elapsed = SparkTpGlobalTimerNs() - control->wait_started_ns;
        if ( gate->kind == SPARK_WEIGHTD_MESH_WAIT_SHIPPED ) control->source_wait_ns += elapsed;
        else control->peer_wait_ns += elapsed;
    }
    control->math_started_ns = UINT64_MAX;
    control->math_finished_ns = 0u;
    control->math_blocks_done = 0u;
    if ( control->error_word == 0u && error != 0u )
    {
        control->error_word = error;
        control->diag_word = SparkTpLdcvU64(&gate->diag);
    }
}

static __global__ void SparkTpMeshHardwareGuardKernel(
    const volatile SparkWeightdMeshWaitRequest *gate,SparkTpMeshRoundControl *control)
{
    if ( threadIdx.x != 0u || blockIdx.x != 0u ) return;
    SparkTpMeshGateAwait(gate,control);
    SparkTpMeshGateGuard(gate,control);
}

static __global__ void SparkTpMeshHardwareWaitKernel(
    volatile SparkWeightdMeshWaitRequest *gate,SparkTpMeshRoundControl *control,
    uint64_t kind,uint64_t peer_mask,uint64_t phase_offset,uint64_t timeout_ns,uint32_t pipe)
{
    if ( threadIdx.x != 0u || blockIdx.x != 0u ) return;
    SparkTpMeshGateRequest(gate,control,kind,peer_mask,phase_offset,timeout_ns,pipe);
    SparkTpMeshGateAwait(gate,control);
    SparkTpMeshGateGuard(gate,control);
}

static __device__ void SparkTpMeshPublishSlot(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    volatile uint64_t *entry,SparkTpMeshRoundControl *control,uint32_t rank,
    const void *source,uint64_t bytes,uint64_t offset,uint64_t doorbell_bytes,uint64_t route,uint32_t phase_offset,
    uint32_t advance,uint64_t started,uint32_t pipe)
{
    uint64_t sequence = *SparkTpMeshPipeSeq(control,pipe) + phase_offset;
    uint64_t tag = (control->epoch << 32u) | sequence;
    uint64_t slot = (uint64_t)rank * slots_per_rank + ((sequence - 1u) & (slots_per_rank - 1u));
    volatile uint8_t *destination = band + slot * slot_bytes;
    volatile uint8_t *payload = destination + offset;
    for ( uint64_t i = threadIdx.x; i < bytes / sizeof(uint64_t); i += blockDim.x )
        ((volatile uint64_t *)payload)[i] = ((const uint64_t *)source)[i];
    if ( threadIdx.x == 0u )
        for ( uint64_t i = bytes & ~UINT64_C(7); i < bytes; i++ )
            payload[i] = ((const uint8_t *)source)[i];
    __threadfence_system();
    __syncthreads();
    if ( threadIdx.x == 0u )
    {
        entry[1] = doorbell_bytes;
        entry[2] = slot;
        entry[3] = route;
        *SparkTpMeshPipeRoundSeq(control,pipe) = tag;
        *SparkTpMeshPipeSeq(control,pipe) += advance;
        __threadfence_system();
        *(volatile uint64_t *)(destination + slot_bytes - 8u) = tag;
        __threadfence_system();
        entry[0] = tag;
        control->copy_ns += SparkTpGlobalTimerNs() - started;
    }
}

static __global__ void SparkTpMeshHardwarePublishKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    volatile uint64_t *entry,SparkTpMeshRoundControl *control,uint32_t rank,
    const void *source,uint64_t bytes,uint64_t offset,uint64_t doorbell_bytes,uint64_t route,uint32_t phase_offset,
    uint32_t advance,volatile SparkWeightdMeshWaitRequest *gate,uint32_t mode,uint64_t peer_mask,uint64_t timeout_ns,uint32_t pipe)
{
    __shared__ uint64_t started;
    __shared__ uint64_t failed;
    if ( threadIdx.x == 0u )
    {
        if ( (mode & SPARK_TP_MESH_PUBLISH_SHIP) != 0u )
        {
            SparkTpMeshGateRequest(gate,control,SPARK_WEIGHTD_MESH_WAIT_SHIPPED,0u,0u,timeout_ns,pipe);
            SparkTpMeshGateAwait(gate,control);
        }
        if ( (mode & SPARK_TP_MESH_PUBLISH_GUARD) != 0u ) SparkTpMeshGateGuard(gate,control);
        started = SparkTpGlobalTimerNs();
        if ( *SparkTpMeshPipeSeq(control,pipe) > UINT32_MAX - phase_offset ) control->error_word = UINT64_MAX;
        failed = control->error_word;
    }
    __syncthreads();
    if ( failed == 0u )
        SparkTpMeshPublishSlot(band,slot_bytes,slots_per_rank,entry,control,rank,source,bytes,offset,doorbell_bytes,route,phase_offset,advance,started,pipe);
    if ( threadIdx.x == 0u && (mode & SPARK_TP_MESH_PUBLISH_REQUEST) != 0u )
        SparkTpMeshGateRequest(gate,control,SPARK_WEIGHTD_MESH_WAIT_PEERS,peer_mask,0u,timeout_ns,pipe);
}

static __device__ void SparkTpMeshHardwareDirectElement(
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,uint64_t ring,
    uint32_t degree,void *output,uint64_t local_elements,uint64_t begin,uint64_t count,
    uint32_t operation,uint64_t i,uint64_t packed_first)
{
    const volatile uint16_t *half;
    const volatile uint64_t *wide;
    uint64_t maximum,value,peer,j;
    float sum;
    if ( operation == 0u || operation == SPARK_TP_MESH_OPERATION_SLICE_GATHER )
    {
        peer = operation == 0u ? i / count : i / local_elements;
        j = operation == 0u ? i % count : i;
        half = (const volatile uint16_t *)(band + (peer * slots_per_rank + ring) * slot_bytes);
        ((uint16_t *)output)[(operation == 0u ? peer * local_elements : 0u) + begin + j] =
            half[packed_first != UINT64_MAX && operation == SPARK_TP_MESH_OPERATION_SLICE_GATHER ? j - peer * local_elements : j];
        return;
    }
    maximum = 0u;
    sum = 0.0f;
    for ( peer = 0u; peer < degree; peer++ )
    {
        wide = (const volatile uint64_t *)(band + (peer * slots_per_rank + ring) * slot_bytes);
        half = (const volatile uint16_t *)wide;
        value = operation == 2u ? wide[i] : 0u;
        maximum = value > maximum ? value : maximum;
        if ( operation != 2u )
            sum += __uint_as_float((uint32_t)half[packed_first != UINT64_MAX ? i - packed_first : i] << 16u);
    }
    if ( operation == 2u )
        ((uint64_t *)output)[begin + i] = maximum;
    else
        ((uint16_t *)output)[begin + i] = (uint16_t)(__float_as_uint(sum) >> 16u);
}

static __global__ void SparkTpMeshHardwareDirectKernel(
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    SparkTpMeshRoundControl *control,uint32_t degree,void *output,
    uint64_t local_elements,uint64_t begin,uint64_t count,uint32_t operation,uint64_t first,uint64_t span,uint32_t last,uint64_t packed_first,uint32_t pipe)
{
    uint64_t ring,i;
    if ( control->error_word != 0u ) return;
    if ( threadIdx.x == 0u )
        (void)atomicMin((unsigned long long *)&control->math_started_ns,SparkTpGlobalTimerNs());
    ring = (*SparkTpMeshPipeRoundSeq(control,pipe) - 1u) & (slots_per_rank - 1u);
    for ( i = first + (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < first + span; i += (uint64_t)blockDim.x * gridDim.x )
        SparkTpMeshHardwareDirectElement(band,slot_bytes,slots_per_rank,ring,degree,output,local_elements,begin,count,operation,i,packed_first);
    __syncthreads();
    if ( threadIdx.x == 0u )
    {
        (void)atomicMax((unsigned long long *)&control->math_finished_ns,SparkTpGlobalTimerNs());
        __threadfence();
        if ( atomicAdd((unsigned long long *)&control->math_blocks_done,1ull) + 1ull == gridDim.x )
        {
            control->combine_ns += SparkTpLdcvU64(&control->math_finished_ns) -
                SparkTpLdcvU64(&control->math_started_ns);
            control->slot_cursor = control->seq;
            if ( last != 0u )
                control->rounds_done++;
        }
    }
}

static __global__ void SparkTpMeshAllToAllPackKernel(
    uint16_t *scratch,const uint16_t *local,uint32_t degree,
    uint64_t per_peer_elements,uint64_t begin,uint64_t count,uint64_t slice_elements)
{
    uint64_t i,peer,j;
    for ( i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < (uint64_t)degree * slice_elements; i += (uint64_t)blockDim.x * gridDim.x )
    {
        peer = i / slice_elements;
        j = i % slice_elements;
        scratch[i] = j < count ? local[peer * per_peer_elements + begin + j] : (uint16_t)0u;
    }
}

static __global__ void SparkTpMeshAllToAllStageKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,const SparkTpMeshRoundControl *control,
    uint16_t *staging,uint64_t staging_slot_elements,const uint16_t *local,uint32_t degree,uint32_t rank,
    uint64_t per_peer_elements,uint64_t begin,uint64_t count)
{
    uint64_t i,peer,j;
    uint16_t *self;
    if ( control->error_word != 0u ) return;
    self = (uint16_t *)(band + ((uint64_t)rank * slots_per_rank + (control->seq & (slots_per_rank - 1u))) * slot_bytes);
    for ( i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < (uint64_t)degree * count; i += (uint64_t)blockDim.x * gridDim.x )
    {
        peer = i / count;
        j = i % count;
        if ( peer == rank )
            self[j] = local[peer * per_peer_elements + begin + j];
        else
            staging[peer * staging_slot_elements + j] = local[peer * per_peer_elements + begin + j];
    }
    __threadfence_system();
}

static __global__ void SparkTpMeshRsagStageKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,const SparkTpMeshRoundControl *control,
    uint16_t *staging,uint64_t staging_slot_elements,const uint16_t *local,uint32_t degree,uint32_t rank,
    uint64_t count,uint64_t slice,uint32_t pipe)
{
    uint64_t i,peer,j;
    uint16_t *self;
    if ( control->error_word != 0u ) return;
    self = (uint16_t *)(band + ((uint64_t)rank * slots_per_rank + ((pipe != 0u ? control->second_seq : control->seq) & (slots_per_rank - 1u))) * slot_bytes);
    for ( i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (uint64_t)blockDim.x * gridDim.x )
    {
        peer = i / slice;
        j = i % slice;
        if ( peer == rank )
            self[j] = local[i];
        else
            staging[peer * staging_slot_elements + j] = local[i];
    }
    __threadfence_system();
}

static __global__ void SparkTpMeshHardwareAllToAllKernel(
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    SparkTpMeshRoundControl *control,uint32_t degree,uint32_t rank,void *output,
    uint64_t per_peer_elements,uint64_t begin,uint64_t count,uint64_t slice_elements,uint32_t last)
{
    const volatile uint16_t *half;
    uint64_t ring,i,peer,j;
    if ( control->error_word != 0u ) return;
    if ( threadIdx.x == 0u )
        (void)atomicMin((unsigned long long *)&control->math_started_ns,SparkTpGlobalTimerNs());
    ring = (control->round_seq - 1u) & (slots_per_rank - 1u);
    for ( i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < (uint64_t)degree * count; i += (uint64_t)blockDim.x * gridDim.x )
    {
        peer = i / count;
        j = i % count;
        half = (const volatile uint16_t *)(band + (peer * slots_per_rank + ring) * slot_bytes);
        ((uint16_t *)output)[peer * per_peer_elements + begin + j] = half[(uint64_t)rank * slice_elements + j];
    }
    __syncthreads();
    if ( threadIdx.x == 0u )
    {
        (void)atomicMax((unsigned long long *)&control->math_finished_ns,SparkTpGlobalTimerNs());
        __threadfence();
        if ( atomicAdd((unsigned long long *)&control->math_blocks_done,1ull) + 1ull == gridDim.x )
        {
            control->combine_ns += SparkTpLdcvU64(&control->math_finished_ns) -
                SparkTpLdcvU64(&control->math_started_ns);
            control->slot_cursor = control->seq;
            if ( last != 0u )
                control->rounds_done++;
        }
    }
}

static __global__ void SparkTpMeshHardwareSeedKernel(
    SparkTpMeshRoundControl *control,void *scratch,const void *local,
    uint64_t elements,uint64_t begin,uint64_t count,uint32_t operation,
    uint32_t rank,uint32_t degree,uint32_t phases)
{
    __shared__ uint64_t started;
    if ( threadIdx.x == 0u )
    {
        started = SparkTpGlobalTimerNs();
        if ( control->seq > UINT32_MAX - phases ) control->error_word = UINT64_MAX;
    }
    __syncthreads();
    if ( control->error_word != 0u ) return;
    SparkTpMeshTreeSeed(scratch,local,elements,begin,count,operation,rank,degree);
    __syncthreads();
    if ( threadIdx.x == 0u ) control->combine_ns += SparkTpGlobalTimerNs() - started;
}

static __global__ void SparkTpMeshHardwareFoldKernel(
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    SparkTpMeshRoundControl *control,void *scratch,uint64_t count,
    uint32_t operation,uint32_t peer,uint32_t phase,uint32_t levels)
{
    __shared__ uint64_t started;
    if ( threadIdx.x == 0u ) started = SparkTpGlobalTimerNs();
    __syncthreads();
    if ( control->error_word != 0u ) return;
    uint64_t ring = (control->seq + phase) & (slots_per_rank - 1u);
    const void *source = band + ((uint64_t)peer * slots_per_rank + ring) * slot_bytes;
    SparkTpMeshTreeFold(scratch,source,count,operation,phase < levels);
    __syncthreads();
    if ( threadIdx.x == 0u ) control->combine_ns += SparkTpGlobalTimerNs() - started;
}

static __global__ void SparkTpMeshHardwareFinishKernel(
    SparkTpMeshRoundControl *control,void *output,const void *scratch,
    uint64_t begin,uint64_t count,uint32_t operation,uint32_t phases,uint32_t last)
{
    __shared__ uint64_t started;
    if ( threadIdx.x == 0u ) started = SparkTpGlobalTimerNs();
    __syncthreads();
    if ( control->error_word != 0u ) return;
    SparkTpMeshTreeFinish(output,scratch,begin,count,operation);
    __syncthreads();
    if ( threadIdx.x == 0u )
    {
        control->combine_ns += SparkTpGlobalTimerNs() - started;
        control->seq += phases;
        control->slot_cursor = control->seq;
        if ( last != 0u ) control->rounds_done++;
    }
}

static cudaError_t SparkTpMeshDriverStatus(CUresult status,const char *phase)
{
    if ( status == CUDA_SUCCESS ) return cudaSuccess;
    const char *name = 0;
    (void)cuGetErrorName(status,&name);
    fprintf(stderr,"MESH-HARDWARE-FAIL phase=%s driver=%d (%s)\n",phase,(int)status,name != 0 ? name : "unknown");
    return status == CUDA_ERROR_NOT_SUPPORTED ? cudaErrorNotSupported : cudaErrorUnknown;
}

static __global__ void SparkTpMeshPairPackKernel(const SparkTpMeshRoundControl *control,const uint16_t *source,uint8_t *staging,
    uint64_t begin,uint64_t count,uint64_t slice,uint32_t parity,uint32_t degree);
static __global__ void SparkTpMeshPairStageKernel(uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,const SparkTpMeshRoundControl *control,
    uint16_t *staging,uint64_t staging_slot_elements,const uint16_t *local,uint8_t *pair_band,uint64_t begin,uint64_t count,uint64_t slice,
    uint32_t rank,uint32_t degree);
static __global__ void SparkTpMeshPairCombineKernel(const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,const SparkTpMeshRoundControl *control,
    uint16_t *output,uint64_t begin,uint64_t count,uint64_t slice,uint32_t rank,uint32_t degree);
static __global__ void SparkTpMeshPairGroupGatherKernel(const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,const SparkTpMeshRoundControl *control,
    uint16_t *output,uint64_t begin,uint64_t count,uint64_t slice,uint32_t rank,uint32_t degree);
static __global__ void SparkTpMeshPairFinishKernel(uint8_t *pair_band,uint64_t slot_bytes,SparkTpMeshRoundControl *control,
    uint16_t *output,uint64_t begin,uint64_t count,uint64_t slice,uint32_t rank,uint32_t degree,uint32_t last);

extern "C" cudaError_t SparkTpMeshHardwarePrepare(void *host,void **device_out)
{
    if ( host == 0 || device_out == 0 ) return cudaErrorInvalidValue;
    *device_out = 0;
    cudaError_t status;
    const void *kernels[] = {
        (const void *)SparkTpMeshHardwareWaitKernel,
        (const void *)SparkTpMeshHardwareGuardKernel,
        (const void *)SparkTpMeshHardwarePublishKernel,
        (const void *)SparkTpMeshHardwareDirectKernel,
        (const void *)SparkTpMeshHardwareSeedKernel,
        (const void *)SparkTpMeshHardwareFoldKernel,
        (const void *)SparkTpMeshHardwareFinishKernel,
        (const void *)SparkTpMeshAllToAllPackKernel,
        (const void *)SparkTpMeshAllToAllStageKernel,
        (const void *)SparkTpMeshRsagStageKernel,
        (const void *)SparkTpMeshHardwareAllToAllKernel,
        (const void *)SparkTpMeshPairPackKernel,
        (const void *)SparkTpMeshPairStageKernel,
        (const void *)SparkTpMeshPairCombineKernel,
        (const void *)SparkTpMeshPairGroupGatherKernel,
        (const void *)SparkTpMeshPairFinishKernel
    };
    cudaFuncAttributes attributes;
    for ( uint32_t i = 0u; i < sizeof(kernels) / sizeof(kernels[0]); i++ )
    {
        status = cudaFuncGetAttributes(&attributes,kernels[i]);
        if ( status != cudaSuccess )
        {
            fprintf(stderr,"MESH-HARDWARE-FAIL phase=preload kernel=%u cuda=%d (%s)\n",
                i,(int)status,cudaGetErrorString(status));
            return status;
        }
    }
    status = cudaHostGetDevicePointer(device_out,host,0u);
    if ( status == cudaSuccess )
        return status;
    /* The shared weightd's mesh region cannot always be host-registered
     * (RDMA-registered shmem pages; lane-0 fleet reproduction), in which
     * case no mapped alias exists. The mesh kernels and the
     * stream-wait path address this region through the host virtual address
     * directly (cache-coherent GB10), so fall back to the identity mapping
     * instead of failing collective initialization. */
    fprintf(stderr,"MESH-DEVICE-ALIAS-IDENTITY ptr=%p cuda=%d (%s) coherent-host-path\n",
        host,(int)status,cudaGetErrorString(status));
    *device_out = host;
    return cudaSuccess;
}

static cudaError_t SparkTpMeshHardwarePeerGuard(cudaStream_t stream,SparkWeightdMeshWaitRequest *gate,SparkTpMeshRoundControl *control)
{
    SparkTpMeshHardwareGuardKernel<<<1,1,0u,stream>>>(gate,control);
    return cudaPeekAtLastError();
}

static cudaError_t SparkTpMeshHardwareWait(cudaStream_t stream,
    SparkWeightdMeshWaitRequest *gate,SparkTpMeshRoundControl *control,
    uint64_t kind,uint64_t peer_mask,uint64_t phase_offset,uint64_t timeout_ns,uint32_t pipe)
{
    SparkTpMeshHardwareWaitKernel<<<1,1,0u,stream>>>(
        gate,control,kind,peer_mask,phase_offset,timeout_ns,pipe);
    return cudaPeekAtLastError();
}

static cudaError_t SparkTpMeshHardwareExchange(cudaStream_t stream,uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,const uint8_t *source,uint64_t bytes,uint64_t offset,uint64_t doorbell_bytes,SparkWeightdMeshRoute route,uint64_t timeout_ns)
{
    cudaError_t status;
    SparkTpMeshHardwarePublishKernel<<<1,SPARK_TP_MESH_THREADS,0u,stream>>>(band,slot_bytes,slots_per_rank,entry,control,rank,source,bytes,offset,doorbell_bytes,route.word,1u,1u,
        request,SPARK_TP_MESH_PUBLISH_SHIP | SPARK_TP_MESH_PUBLISH_GUARD | SPARK_TP_MESH_PUBLISH_REQUEST,route.fields.peer_mask,timeout_ns,0u);
    status = cudaPeekAtLastError();
    return status == cudaSuccess ? SparkTpMeshHardwarePeerGuard(stream,request,control) : status;
}

static cudaError_t SparkTpMeshHardwareCombinePacked(cudaStream_t stream,const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,SparkTpMeshRoundControl *control,uint32_t degree,void *output,uint64_t local_elements,uint64_t begin,uint64_t count,uint32_t operation,uint64_t first,uint64_t span,uint32_t last,uint64_t packed_first,uint32_t pipe)
{
    uint32_t blocks;
    blocks = (uint32_t)((span + SPARK_TP_MESH_THREADS - 1u) / SPARK_TP_MESH_THREADS);
    SparkTpMeshHardwareDirectKernel<<<blocks != 0u ? blocks : 1u,SPARK_TP_MESH_THREADS,0u,stream>>>(band,slot_bytes,slots_per_rank,control,degree,output,local_elements,begin,count,operation,first,span,last,packed_first,pipe);
    return cudaPeekAtLastError();
}

static cudaError_t SparkTpMeshHardwareCombine(cudaStream_t stream,const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,SparkTpMeshRoundControl *control,uint32_t degree,void *output,uint64_t local_elements,uint64_t begin,uint64_t count,uint32_t operation,uint64_t first,uint64_t span,uint32_t last)
{
    uint32_t blocks;
    blocks = (uint32_t)((span + SPARK_TP_MESH_THREADS - 1u) / SPARK_TP_MESH_THREADS);
    SparkTpMeshHardwareDirectKernel<<<blocks != 0u ? blocks : 1u,SPARK_TP_MESH_THREADS,0u,stream>>>(band,slot_bytes,slots_per_rank,control,degree,output,local_elements,begin,count,operation,first,span,last,UINT64_MAX,0u);
    return cudaPeekAtLastError();
}

static cudaError_t SparkTpMeshHardwareDirectChunk(cudaStream_t stream,uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,const uint8_t *local,void *output,uint64_t local_elements,uint64_t begin,uint64_t count,uint32_t operation,uint64_t timeout_ns)
{
    SparkWeightdMeshRoute route = {0};
    uint32_t width;
    cudaError_t status;
    width = operation == 2u ? 8u : 2u;
    route.fields.peer_mask = ((1u << degree) - 1u) & ~(1u << rank);
    status = SparkTpMeshHardwareExchange(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,local + begin * width,count * width,0u,count * width,route,timeout_ns);
    if ( status != cudaSuccess )
        return status;
    return SparkTpMeshHardwareCombine(stream,band,slot_bytes,slots_per_rank,control,degree,output,local_elements,begin,count,operation,0u,operation == 0u ? count * degree : count,begin + count == local_elements);
}

static cudaError_t SparkTpMeshHardwareRsagChunk(cudaStream_t stream,uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,const uint8_t *local,void *output,uint64_t local_elements,uint64_t begin,uint64_t count,uint64_t timeout_ns)
{
    SparkWeightdMeshRoute route = {0};
    uint64_t slice,first,span;
    cudaError_t status;
    slice = SparkTpMeshRsagSlice(count,degree);
    first = (uint64_t)rank * slice < count ? (uint64_t)rank * slice : count;
    span = count - first < slice ? count - first : slice;
    route.fields.peer_mask = ((1u << degree) - 1u) & ~(1u << rank);
    route.fields.slice_bytes = slice * sizeof(uint16_t);
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_SCATTER;
    status = SparkTpMeshHardwareExchange(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,local + begin * sizeof(uint16_t),count * sizeof(uint16_t),0u,count * sizeof(uint16_t),route,timeout_ns);
    if ( status == cudaSuccess )
        status = SparkTpMeshHardwareCombine(stream,band,slot_bytes,slots_per_rank,control,degree,output,local_elements,begin,count,1u,first,span,0u);
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_GATHER;
    if ( status == cudaSuccess )
        status = SparkTpMeshHardwareExchange(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,(const uint8_t *)output + (begin + first) * sizeof(uint16_t),span * sizeof(uint16_t),first * sizeof(uint16_t),count * sizeof(uint16_t),route,timeout_ns);
    if ( status != cudaSuccess )
        return status;
    return SparkTpMeshHardwareCombine(stream,band,slot_bytes,slots_per_rank,control,degree,output,slice,begin,count,SPARK_TP_MESH_OPERATION_SLICE_GATHER,0u,count,begin + count == local_elements);
}

typedef struct SparkTpMeshPipeBand
{
    uint8_t *band;
    volatile uint64_t *entry;
    SparkWeightdMeshWaitRequest *gate;
    uint8_t *staging;
    uint32_t pipe;
} SparkTpMeshPipeBand;

static uint64_t SparkTpMeshPeerRsagFirst(uint64_t count,uint32_t rank,uint32_t degree,uint64_t *span)
{
    uint64_t slice = SparkTpMeshRsagSlice(count,degree),first;
    first = (uint64_t)rank * slice < count ? (uint64_t)rank * slice : count;
    *span = count - first < slice ? count - first : slice;
    return(first);
}

static SparkWeightdMeshRoute SparkTpMeshPeerRsagRoute(uint32_t rank,uint32_t degree,uint32_t mode,uint64_t slice_bytes)
{
    SparkWeightdMeshRoute route = {0};
    route.fields.peer_mask = ((1u << degree) - 1u) & ~(1u << rank);
    route.fields.mode = mode;
    route.fields.slice_bytes = slice_bytes;
    return(route);
}

static cudaError_t SparkTpMeshHardwarePeerRsagIssue(cudaStream_t stream,const SparkTpMeshPipeBand *view,uint64_t slot_bytes,uint64_t slots_per_rank,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,const uint8_t *local,uint64_t begin,uint64_t count,uint64_t timeout_ns)
{
    uint64_t slice = SparkTpMeshRsagSlice(count,degree),staging_slot_elements = SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES / sizeof(uint16_t);
    SparkWeightdMeshRoute route = SparkTpMeshPeerRsagRoute(rank,degree,SPARK_WEIGHTD_MESH_ROUTE_PEER,slice * sizeof(uint16_t));
    uint32_t blocks;
    cudaError_t status;
    if ( view->staging == 0 || slice > staging_slot_elements )
        return cudaErrorInvalidValue;
    status = SparkTpMeshHardwareWait(stream,view->gate,control,SPARK_WEIGHTD_MESH_WAIT_SHIPPED,0u,0u,timeout_ns,view->pipe);
    if ( status != cudaSuccess )
        return status;
    blocks = (uint32_t)((count + SPARK_TP_MESH_THREADS - 1u) / SPARK_TP_MESH_THREADS);
    SparkTpMeshRsagStageKernel<<<blocks != 0u ? blocks : 1u,SPARK_TP_MESH_THREADS,0u,stream>>>(view->band,slot_bytes,slots_per_rank,control,(uint16_t *)view->staging,staging_slot_elements,(const uint16_t *)local + begin,degree,rank,count,slice,view->pipe);
    status = cudaPeekAtLastError();
    if ( status != cudaSuccess )
        return status;
    SparkTpMeshHardwarePublishKernel<<<1,SPARK_TP_MESH_THREADS,0u,stream>>>(view->band,slot_bytes,slots_per_rank,view->entry,control,rank,local,0u,0u,route.fields.slice_bytes,route.word,1u,1u,
        view->gate,SPARK_TP_MESH_PUBLISH_REQUEST,route.fields.peer_mask,timeout_ns,view->pipe);
    return cudaPeekAtLastError();
}

static cudaError_t SparkTpMeshHardwarePeerRsagReduce(cudaStream_t stream,const SparkTpMeshPipeBand *view,uint64_t slot_bytes,uint64_t slots_per_rank,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,void *output,uint64_t local_elements,uint64_t begin,uint64_t count,uint64_t timeout_ns)
{
    SparkWeightdMeshRoute route = SparkTpMeshPeerRsagRoute(rank,degree,SPARK_WEIGHTD_MESH_ROUTE_FULL,0u);
    uint64_t span,first = SparkTpMeshPeerRsagFirst(count,rank,degree,&span),gather_bytes;
    cudaError_t status = SparkTpMeshHardwarePeerGuard(stream,view->gate,control);
    if ( status == cudaSuccess )
        status = SparkTpMeshHardwareCombinePacked(stream,view->band,slot_bytes,slots_per_rank,control,degree,output,local_elements,begin,count,1u,first,span,0u,first,view->pipe);
    if ( status != cudaSuccess )
        return status;
    gather_bytes = (span * sizeof(uint16_t) + 7u) & ~UINT64_C(7);
    SparkTpMeshHardwarePublishKernel<<<1,SPARK_TP_MESH_THREADS,0u,stream>>>(view->band,slot_bytes,slots_per_rank,view->entry,control,rank,(const uint8_t *)output + (begin + first) * sizeof(uint16_t),span * sizeof(uint16_t),0u,gather_bytes != 0u ? gather_bytes : 8u,route.word,1u,1u,
        view->gate,SPARK_TP_MESH_PUBLISH_SHIP | SPARK_TP_MESH_PUBLISH_GUARD | SPARK_TP_MESH_PUBLISH_REQUEST,route.fields.peer_mask,timeout_ns,view->pipe);
    return cudaPeekAtLastError();
}

static cudaError_t SparkTpMeshHardwarePeerRsagGather(cudaStream_t stream,const SparkTpMeshPipeBand *view,uint64_t slot_bytes,uint64_t slots_per_rank,SparkTpMeshRoundControl *control,uint32_t degree,void *output,uint64_t local_elements,uint64_t begin,uint64_t count)
{
    cudaError_t status = SparkTpMeshHardwarePeerGuard(stream,view->gate,control);
    if ( status != cudaSuccess )
        return status;
    return SparkTpMeshHardwareCombinePacked(stream,view->band,slot_bytes,slots_per_rank,control,degree,output,SparkTpMeshRsagSlice(count,degree),begin,count,SPARK_TP_MESH_OPERATION_SLICE_GATHER,0u,count,begin + count == local_elements,0u,view->pipe);
}

static cudaError_t SparkTpMeshHardwarePeerRsagChunk(cudaStream_t stream,const SparkTpMeshPipeBand *view,uint64_t slot_bytes,uint64_t slots_per_rank,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,const uint8_t *local,void *output,uint64_t local_elements,uint64_t begin,uint64_t count,uint64_t timeout_ns)
{
    cudaError_t status = SparkTpMeshHardwarePeerRsagIssue(stream,view,slot_bytes,slots_per_rank,control,rank,degree,local,begin,count,timeout_ns);
    if ( status == cudaSuccess )
        status = SparkTpMeshHardwarePeerRsagReduce(stream,view,slot_bytes,slots_per_rank,control,rank,degree,output,local_elements,begin,count,timeout_ns);
    if ( status != cudaSuccess )
        return status;
    return SparkTpMeshHardwarePeerRsagGather(stream,view,slot_bytes,slots_per_rank,control,degree,output,local_elements,begin,count);
}

static cudaError_t SparkTpMeshHardwarePipelinedRsagRound(cudaStream_t stream,const SparkTpMeshPipeBand *views,uint64_t slot_bytes,uint64_t slots_per_rank,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,const uint8_t *local,void *output,uint64_t local_elements,uint64_t timeout_ns)
{
    uint64_t per = SparkTpMeshPipelineChunkElements(local_elements,degree,SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES);
    uint64_t chunks = (local_elements + per - 1u) / per,chunk[2],count;
    uint32_t reduced[2],live[2],pipe;
    cudaError_t status = cudaSuccess;
    for (pipe=0u; pipe<2u; pipe++)
    {
        chunk[pipe] = pipe;
        reduced[pipe] = 0u;
        live[pipe] = pipe < chunks;
        count = local_elements - pipe * per < per ? local_elements - pipe * per : per;
        if ( live[pipe] != 0u && status == cudaSuccess )
            status = SparkTpMeshHardwarePeerRsagIssue(stream,&views[pipe],slot_bytes,slots_per_rank,control,rank,degree,local,pipe * per,count,timeout_ns);
    }
    while ( status == cudaSuccess && (live[0] != 0u || live[1] != 0u) )
        for (pipe=0u; pipe<2u && status==cudaSuccess; pipe++)
        {
            uint64_t begin = chunk[pipe] * per;
            if ( live[pipe] == 0u )
                continue;
            count = local_elements - begin < per ? local_elements - begin : per;
            if ( reduced[pipe] == 0u )
            {
                status = SparkTpMeshHardwarePeerRsagReduce(stream,&views[pipe],slot_bytes,slots_per_rank,control,rank,degree,output,local_elements,begin,count,timeout_ns);
                reduced[pipe] = 1u;
                continue;
            }
            status = SparkTpMeshHardwarePeerRsagGather(stream,&views[pipe],slot_bytes,slots_per_rank,control,degree,output,local_elements,begin,count);
            reduced[pipe] = 0u;
            chunk[pipe] += 2u;
            live[pipe] = chunk[pipe] < chunks;
            if ( live[pipe] != 0u && status == cudaSuccess )
            {
                begin = chunk[pipe] * per;
                count = local_elements - begin < per ? local_elements - begin : per;
                status = SparkTpMeshHardwarePeerRsagIssue(stream,&views[pipe],slot_bytes,slots_per_rank,control,rank,degree,local,begin,count,timeout_ns);
            }
        }
    return status;
}

static __device__ __forceinline__ uint64_t SparkTpMeshPairSpan(uint64_t count,uint64_t slice,uint32_t index,uint64_t *first)
{
    uint64_t begin = (uint64_t)index * slice < count ? (uint64_t)index * slice : count;
    *first = begin;
    return count - begin < slice ? count - begin : slice;
}

static __device__ __forceinline__ uint8_t *SparkTpMeshPairLanding(uint8_t *band,uint64_t slot_bytes,uint32_t rank,uint64_t round_seq)
{
    uint64_t ring = (round_seq - 1u) & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u);
    (void)slot_bytes;
    return band + SPARK_WEIGHTD_MESH_BULK_LANDING_OFFSET(rank ^ 1u,ring);
}

static __global__ void SparkTpMeshPairPackKernel(const SparkTpMeshRoundControl *control,const uint16_t *source,uint8_t *staging,
    uint64_t begin,uint64_t count,uint64_t slice,uint32_t parity,uint32_t degree)
{
    uint64_t i,group,t,first,span;
    uint16_t *destination;
    if ( control->error_word != 0u ) return;
    destination = (uint16_t *)(staging + (control->second_seq & (SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u)) * SPARK_WEIGHTD_MESH_BULK_BYTES);
    for ( i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < (uint64_t)(degree / 2u) * slice; i += (uint64_t)blockDim.x * gridDim.x )
    {
        group = i / slice;
        t = i % slice;
        span = SparkTpMeshPairSpan(count,slice,parity + 2u * (uint32_t)group,&first);
        if ( t < span )
            destination[i] = source[begin + first + t];
    }
    __threadfence_system();
}

static __global__ void SparkTpMeshPairStageKernel(uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,const SparkTpMeshRoundControl *control,
    uint16_t *staging,uint64_t staging_slot_elements,const uint16_t *local,uint8_t *pair_band,uint64_t begin,uint64_t count,uint64_t slice,
    uint32_t rank,uint32_t degree)
{
    const uint16_t *landing;
    uint16_t *self,*target;
    uint64_t i,group,t,first,span;
    uint32_t peer;
    if ( control->error_word != 0u ) return;
    landing = (const uint16_t *)SparkTpMeshPairLanding(pair_band,slot_bytes,rank,control->second_round_seq);
    self = (uint16_t *)(band + ((uint64_t)rank * slots_per_rank + (control->seq & (slots_per_rank - 1u))) * slot_bytes);
    for ( i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < (uint64_t)(degree / 2u) * slice; i += (uint64_t)blockDim.x * gridDim.x )
    {
        group = i / slice;
        t = i % slice;
        peer = (rank & 1u) + 2u * (uint32_t)group;
        span = SparkTpMeshPairSpan(count,slice,peer,&first);
        if ( t >= span )
            continue;
        target = peer == rank ? self : staging + (uint64_t)peer * staging_slot_elements;
        target[t] = local[begin + first + t];
        target[slice + t] = landing[i];
    }
    __threadfence_system();
}

static __global__ void SparkTpMeshPairCombineKernel(const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,const SparkTpMeshRoundControl *control,
    uint16_t *output,uint64_t begin,uint64_t count,uint64_t slice,uint32_t rank,uint32_t degree)
{
    const volatile uint16_t *half;
    uint64_t ring,t,first,span;
    uint32_t pair,sender;
    float sum,head,tail;
    if ( control->error_word != 0u ) return;
    ring = (control->round_seq - 1u) & (slots_per_rank - 1u);
    span = SparkTpMeshPairSpan(count,slice,rank,&first);
    for ( t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; t < span; t += (uint64_t)blockDim.x * gridDim.x )
    {
        sum = 0.0f;
        for ( pair = 0u; pair < degree / 2u; pair++ )
        {
            sender = 2u * pair + (rank & 1u);
            half = (const volatile uint16_t *)(band + ((uint64_t)sender * slots_per_rank + ring) * slot_bytes);
            head = __uint_as_float((uint32_t)half[t] << 16u);
            tail = __uint_as_float((uint32_t)half[slice + t] << 16u);
            if ( (rank & 1u) == 0u ) { sum += head; sum += tail; }
            else { sum += tail; sum += head; }
        }
        output[begin + first + t] = (uint16_t)(__float_as_uint(sum) >> 16u);
    }
}

static __global__ void SparkTpMeshPairGroupGatherKernel(const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,const SparkTpMeshRoundControl *control,
    uint16_t *output,uint64_t begin,uint64_t count,uint64_t slice,uint32_t rank,uint32_t degree)
{
    const volatile uint16_t *half;
    uint64_t ring,i,group,t,first,span;
    uint32_t peer;
    if ( control->error_word != 0u ) return;
    ring = (control->round_seq - 1u) & (slots_per_rank - 1u);
    for ( i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < (uint64_t)(degree / 2u) * slice; i += (uint64_t)blockDim.x * gridDim.x )
    {
        group = i / slice;
        t = i % slice;
        peer = (rank & 1u) + 2u * (uint32_t)group;
        span = SparkTpMeshPairSpan(count,slice,peer,&first);
        if ( peer == rank || t >= span )
            continue;
        half = (const volatile uint16_t *)(band + ((uint64_t)peer * slots_per_rank + ring) * slot_bytes);
        output[begin + first + t] = half[t];
    }
}

static __global__ void SparkTpMeshPairFinishKernel(uint8_t *pair_band,uint64_t slot_bytes,SparkTpMeshRoundControl *control,
    uint16_t *output,uint64_t begin,uint64_t count,uint64_t slice,uint32_t rank,uint32_t degree,uint32_t last)
{
    const volatile uint16_t *landing;
    uint64_t i,group,t,first,span;
    if ( control->error_word != 0u ) return;
    landing = (const volatile uint16_t *)SparkTpMeshPairLanding(pair_band,slot_bytes,rank,control->second_round_seq);
    for ( i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < (uint64_t)(degree / 2u) * slice; i += (uint64_t)blockDim.x * gridDim.x )
    {
        group = i / slice;
        t = i % slice;
        span = SparkTpMeshPairSpan(count,slice,((rank & 1u) ^ 1u) + 2u * (uint32_t)group,&first);
        if ( t < span )
            output[begin + first + t] = landing[i];
    }
    __syncthreads();
    if ( threadIdx.x == 0u )
    {
        __threadfence();
        if ( atomicAdd((unsigned long long *)&control->math_blocks_done,1ull) + 1ull == gridDim.x )
        {
            control->math_blocks_done = 0u;
            control->slot_cursor = control->seq;
            if ( last != 0u )
                control->rounds_done++;
        }
    }
}

static uint32_t SparkTpMeshPairBlocks(uint64_t elements)
{
    uint64_t blocks = (elements + SPARK_TP_MESH_THREADS - 1u) / SPARK_TP_MESH_THREADS;
    return blocks == 0u ? 1u : blocks > 4096u ? 4096u : (uint32_t)blocks;
}

static uint64_t SparkTpMeshPairGroupMask(uint32_t rank,uint32_t degree)
{
    uint64_t mask = 0u;
    for ( uint32_t peer = rank & 1u; peer < degree; peer += 2u )
        if ( peer != rank ) mask |= UINT64_C(1) << peer;
    return mask;
}

static cudaError_t SparkTpMeshPairIssueBulk(cudaStream_t stream,const SparkTpMeshPipeBand *pair,uint64_t slot_bytes,uint64_t slots_per_rank,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    const uint16_t *source,uint64_t begin,uint64_t count,uint64_t slice,uint32_t parity,uint64_t timeout_ns)
{
    SparkWeightdMeshRoute route = SparkTpMeshPeerRsagRoute(rank,degree,SPARK_WEIGHTD_MESH_ROUTE_PEER,(uint64_t)(degree / 2u) * slice * sizeof(uint16_t));
    cudaError_t status;
    route.fields.peer_mask = UINT64_C(1) << (rank ^ 1u);
    route.fields.reserved = SPARK_WEIGHTD_MESH_ROUTE_FLAG_BULK;
    status = SparkTpMeshHardwareWait(stream,pair->gate,control,SPARK_WEIGHTD_MESH_WAIT_SHIPPED,0u,0u,timeout_ns,1u);
    if ( status != cudaSuccess )
        return status;
    SparkTpMeshPairPackKernel<<<SparkTpMeshPairBlocks((uint64_t)(degree / 2u) * slice),SPARK_TP_MESH_THREADS,0u,stream>>>(control,source,pair->staging,begin,count,slice,parity,degree);
    status = cudaPeekAtLastError();
    if ( status != cudaSuccess )
        return status;
    SparkTpMeshHardwarePublishKernel<<<1,SPARK_TP_MESH_THREADS,0u,stream>>>(pair->band,slot_bytes,slots_per_rank,pair->entry,control,rank,source,0u,0u,route.fields.slice_bytes,route.word,1u,1u,
        pair->gate,SPARK_TP_MESH_PUBLISH_REQUEST,route.fields.peer_mask,timeout_ns,1u);
    return cudaPeekAtLastError();
}

static cudaError_t SparkTpMeshPairIssueScatter(cudaStream_t stream,const SparkTpMeshPipeBand *views,uint64_t slot_bytes,uint64_t slots_per_rank,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    const uint16_t *local,uint64_t begin,uint64_t count,uint64_t slice,uint64_t timeout_ns)
{
    SparkWeightdMeshRoute route = SparkTpMeshPeerRsagRoute(rank,degree,SPARK_WEIGHTD_MESH_ROUTE_PEER,2u * slice * sizeof(uint16_t));
    cudaError_t status;
    route.fields.peer_mask = SparkTpMeshPairGroupMask(rank,degree);
    status = SparkTpMeshHardwareWait(stream,views[0].gate,control,SPARK_WEIGHTD_MESH_WAIT_SHIPPED,0u,0u,timeout_ns,0u);
    if ( status != cudaSuccess )
        return status;
    SparkTpMeshPairStageKernel<<<SparkTpMeshPairBlocks((uint64_t)(degree / 2u) * slice),SPARK_TP_MESH_THREADS,0u,stream>>>(views[0].band,slot_bytes,slots_per_rank,control,
        (uint16_t *)views[0].staging,SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES / sizeof(uint16_t),local,views[1].band,begin,count,slice,rank,degree);
    status = cudaPeekAtLastError();
    if ( status != cudaSuccess )
        return status;
    SparkTpMeshHardwarePublishKernel<<<1,SPARK_TP_MESH_THREADS,0u,stream>>>(views[0].band,slot_bytes,slots_per_rank,views[0].entry,control,rank,local,0u,0u,route.fields.slice_bytes,route.word,1u,1u,
        views[0].gate,SPARK_TP_MESH_PUBLISH_REQUEST,route.fields.peer_mask,timeout_ns,0u);
    return cudaPeekAtLastError();
}

static cudaError_t SparkTpMeshPairReduce(cudaStream_t stream,const SparkTpMeshPipeBand *views,uint64_t slot_bytes,uint64_t slots_per_rank,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    uint16_t *output,uint64_t begin,uint64_t count,uint64_t slice,uint64_t timeout_ns)
{
    SparkWeightdMeshRoute route = SparkTpMeshPeerRsagRoute(rank,degree,SPARK_WEIGHTD_MESH_ROUTE_FULL,0u);
    uint64_t first = (uint64_t)rank * slice < count ? (uint64_t)rank * slice : count,span = count - first < slice ? count - first : slice;
    uint64_t gather_bytes = (span * sizeof(uint16_t) + 7u) & ~UINT64_C(7);
    cudaError_t status = SparkTpMeshHardwarePeerGuard(stream,views[0].gate,control);
    route.fields.peer_mask = SparkTpMeshPairGroupMask(rank,degree);
    if ( status != cudaSuccess )
        return status;
    SparkTpMeshPairCombineKernel<<<SparkTpMeshPairBlocks(span),SPARK_TP_MESH_THREADS,0u,stream>>>(views[0].band,slot_bytes,slots_per_rank,control,output,begin,count,slice,rank,degree);
    status = cudaPeekAtLastError();
    if ( status != cudaSuccess )
        return status;
    SparkTpMeshHardwarePublishKernel<<<1,SPARK_TP_MESH_THREADS,0u,stream>>>(views[0].band,slot_bytes,slots_per_rank,views[0].entry,control,rank,(const uint8_t *)(output + begin + first),span * sizeof(uint16_t),0u,gather_bytes != 0u ? gather_bytes : 8u,route.word,1u,1u,
        views[0].gate,SPARK_TP_MESH_PUBLISH_SHIP | SPARK_TP_MESH_PUBLISH_GUARD | SPARK_TP_MESH_PUBLISH_REQUEST,route.fields.peer_mask,timeout_ns,0u);
    return cudaPeekAtLastError();
}

static cudaError_t SparkTpMeshHardwarePairRound(cudaStream_t stream,const SparkTpMeshPipeBand *views,uint64_t slot_bytes,uint64_t slots_per_rank,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,
    const uint8_t *local,void *output,uint64_t local_elements,uint64_t timeout_ns)
{
    const uint16_t *input = (const uint16_t *)local;
    uint16_t *result = (uint16_t *)output;
    uint64_t per = SparkTpMeshPairChunkElements(local_elements,degree,SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES),chunks = (local_elements + per - 1u) / per,chunk,begin,count,slice,next_begin,next_count,next_slice;
    uint32_t blocks;
    cudaError_t status;
    if ( views[1].band == 0 || views[1].staging == 0 || (degree & 1u) != 0u || degree < 4u )
        return cudaErrorInvalidValue;
    count = local_elements < per ? local_elements : per;
    slice = SparkTpMeshRsagSlice(count,degree);
    status = SparkTpMeshPairIssueBulk(stream,&views[1],slot_bytes,slots_per_rank,control,rank,degree,input,0u,count,slice,(rank & 1u) ^ 1u,timeout_ns);
    if ( status == cudaSuccess )
        status = SparkTpMeshHardwarePeerGuard(stream,views[1].gate,control);
    if ( status == cudaSuccess )
        status = SparkTpMeshPairIssueScatter(stream,views,slot_bytes,slots_per_rank,control,rank,degree,input,0u,count,slice,timeout_ns);
    for ( chunk = 0u; chunk < chunks && status == cudaSuccess; chunk++ )
    {
        begin = chunk * per;
        count = local_elements - begin < per ? local_elements - begin : per;
        slice = SparkTpMeshRsagSlice(count,degree);
        next_begin = begin + count;
        next_count = chunk + 1u < chunks ? (local_elements - next_begin < per ? local_elements - next_begin : per) : 0u;
        next_slice = next_count != 0u ? SparkTpMeshRsagSlice(next_count,degree) : 0u;
        if ( next_count != 0u )
            status = SparkTpMeshPairIssueBulk(stream,&views[1],slot_bytes,slots_per_rank,control,rank,degree,input,next_begin,next_count,next_slice,(rank & 1u) ^ 1u,timeout_ns);
        if ( status == cudaSuccess )
            status = SparkTpMeshPairReduce(stream,views,slot_bytes,slots_per_rank,control,rank,degree,result,begin,count,slice,timeout_ns);
        if ( status == cudaSuccess && next_count != 0u )
            status = SparkTpMeshHardwarePeerGuard(stream,views[1].gate,control);
        if ( status == cudaSuccess )
            status = SparkTpMeshHardwarePeerGuard(stream,views[0].gate,control);
        if ( status == cudaSuccess )
        {
            SparkTpMeshPairGroupGatherKernel<<<SparkTpMeshPairBlocks((uint64_t)(degree / 2u) * slice),SPARK_TP_MESH_THREADS,0u,stream>>>(views[0].band,slot_bytes,slots_per_rank,control,result,begin,count,slice,rank,degree);
            status = cudaPeekAtLastError();
        }
        if ( status == cudaSuccess && next_count != 0u )
            status = SparkTpMeshPairIssueScatter(stream,views,slot_bytes,slots_per_rank,control,rank,degree,input,next_begin,next_count,next_slice,timeout_ns);
        if ( status == cudaSuccess )
            status = SparkTpMeshPairIssueBulk(stream,&views[1],slot_bytes,slots_per_rank,control,rank,degree,result,begin,count,slice,rank & 1u,timeout_ns);
        if ( status == cudaSuccess )
            status = SparkTpMeshHardwarePeerGuard(stream,views[1].gate,control);
        if ( status != cudaSuccess )
            break;
        blocks = SparkTpMeshPairBlocks((uint64_t)(degree / 2u) * slice);
        SparkTpMeshPairFinishKernel<<<blocks,SPARK_TP_MESH_THREADS,0u,stream>>>(views[1].band,slot_bytes,control,result,begin,count,slice,rank,degree,chunk + 1u == chunks);
        status = cudaPeekAtLastError();
    }
    return status;
}

static cudaError_t SparkTpMeshHardwareDirectRound(cudaStream_t stream,uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,const uint8_t *local,void *output,uint64_t elements,uint32_t operation,uint32_t slice_routes,uint8_t *staging,const SparkTpMeshSecondBand *second,uint64_t timeout_ns)
{
    uint64_t local_elements,capacity,begin,count;
    uint32_t rsag,peer;
    cudaError_t status = cudaSuccess;
    SparkTpMeshPipeBand views[2] = {
        {band,entry,request,staging,0u},
        {second != 0 ? (uint8_t *)second->band : 0,second != 0 ? (volatile uint64_t *)second->entry : 0,
         second != 0 ? (SparkWeightdMeshWaitRequest *)second->gate : 0,second != 0 ? (uint8_t *)second->staging : 0,1u}};
    local_elements = SparkTpMeshDirectLocalElements(elements,degree,operation);
    rsag = SparkTpMeshDirectPhasesPerChunk(elements,degree,operation,slice_routes) == 2u;
    peer = rsag != 0u && (slice_routes & SPARK_TP_MESH_ROUTES_PEER) != 0u;
    if ( peer != 0u && second != 0 && (slice_routes & SPARK_TP_MESH_ROUTES_PAIR) != 0u && (degree & 1u) == 0u && degree >= 4u )
        return SparkTpMeshHardwarePairRound(stream,views,slot_bytes,slots_per_rank,control,rank,degree,local,output,local_elements,timeout_ns);
    if ( peer != 0u && second != 0 && local_elements > SparkTpMeshDirectPeerCapacity(degree,SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES) )
        return SparkTpMeshHardwarePipelinedRsagRound(stream,views,slot_bytes,slots_per_rank,control,rank,degree,local,output,local_elements,timeout_ns);
    capacity = peer != 0u ? SparkTpMeshDirectPeerCapacity(degree,SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES) : SparkTpMeshDirectCapacity(slot_bytes,operation);
    for (begin=0u; begin<local_elements && status==cudaSuccess; begin+=count)
    {
        count = local_elements - begin < capacity ? local_elements - begin : capacity;
        if ( peer != 0u )
        {
            status = SparkTpMeshHardwarePeerRsagChunk(stream,&views[0],slot_bytes,slots_per_rank,control,rank,degree,local,output,local_elements,begin,count,timeout_ns);
            continue;
        }
        status = rsag != 0u ? SparkTpMeshHardwareRsagChunk(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,local,output,local_elements,begin,count,timeout_ns) : SparkTpMeshHardwareDirectChunk(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,local,output,local_elements,begin,count,operation,timeout_ns);
    }
    return status;
}

static cudaError_t SparkTpMeshHardwarePeerAllToAllRound(cudaStream_t stream,uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,const uint8_t *local,void *output,uint8_t *staging,uint64_t per_peer_elements,uint64_t timeout_ns)
{
    SparkWeightdMeshRoute route = {0};
    uint64_t slice = SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES / sizeof(uint16_t),begin,count;
    uint32_t blocks;
    cudaError_t status = cudaSuccess;
    if ( staging == 0 || slice * sizeof(uint16_t) > slot_bytes - SPARK_WEIGHTD_MESH_SLOT_TRAILER_BYTES )
        return cudaErrorInvalidValue;
    route.fields.peer_mask = ((1u << degree) - 1u) & ~(1u << rank);
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_PEER;
    for (begin=0u; begin<per_peer_elements && status==cudaSuccess; begin+=count)
    {
        count = per_peer_elements - begin < slice ? per_peer_elements - begin : slice;
        route.fields.slice_bytes = ((count * sizeof(uint16_t) + 7u) & ~UINT64_C(7));
        status = SparkTpMeshHardwareWait(stream,request,control,SPARK_WEIGHTD_MESH_WAIT_SHIPPED,0u,0u,timeout_ns,0u);
        if ( status != cudaSuccess )
            return status;
        blocks = (uint32_t)(((uint64_t)degree * count + SPARK_TP_MESH_THREADS - 1u) / SPARK_TP_MESH_THREADS);
        SparkTpMeshAllToAllStageKernel<<<blocks,SPARK_TP_MESH_THREADS,0u,stream>>>(band,slot_bytes,slots_per_rank,control,(uint16_t *)staging,slice,(const uint16_t *)local,degree,rank,per_peer_elements,begin,count);
        status = cudaPeekAtLastError();
        if ( status != cudaSuccess )
            return status;
        SparkTpMeshHardwarePublishKernel<<<1,SPARK_TP_MESH_THREADS,0u,stream>>>(band,slot_bytes,slots_per_rank,entry,control,rank,local,0u,0u,route.fields.slice_bytes,route.word,1u,1u,
            request,SPARK_TP_MESH_PUBLISH_REQUEST,route.fields.peer_mask,timeout_ns,0u);
        status = cudaPeekAtLastError();
        if ( status == cudaSuccess )
            status = SparkTpMeshHardwarePeerGuard(stream,request,control);
        if ( status != cudaSuccess )
            return status;
        blocks = (uint32_t)(((uint64_t)degree * count + SPARK_TP_MESH_THREADS - 1u) / SPARK_TP_MESH_THREADS);
        SparkTpMeshHardwareAllToAllKernel<<<blocks,SPARK_TP_MESH_THREADS,0u,stream>>>(band,slot_bytes,slots_per_rank,control,degree,rank,output,per_peer_elements,begin,count,0u,begin + count == per_peer_elements);
        status = cudaPeekAtLastError();
    }
    return status;
}

static cudaError_t SparkTpMeshHardwareAllToAllRound(cudaStream_t stream,uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,const uint8_t *local,void *output,void *scratch,uint64_t per_peer_elements,uint64_t timeout_ns)
{
    SparkWeightdMeshRoute route = {0};
    uint64_t slice,begin,count,span;
    uint32_t blocks;
    cudaError_t status = cudaSuccess;
    slice = SparkTpMeshAllToAllSliceElements(slot_bytes,degree);
    if ( slice == 0u || slice * sizeof(uint16_t) >= (1u << 24u) )
        return cudaErrorInvalidValue;
    route.fields.peer_mask = ((1u << degree) - 1u) & ~(1u << rank);
    route.fields.slice_bytes = slice * sizeof(uint16_t);
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_SCATTER;
    for (begin=0u; begin<per_peer_elements && status==cudaSuccess; begin+=count)
    {
        count = per_peer_elements - begin < slice ? per_peer_elements - begin : slice;
        span = (uint64_t)degree * slice;
        blocks = (uint32_t)((span + SPARK_TP_MESH_THREADS - 1u) / SPARK_TP_MESH_THREADS);
        SparkTpMeshAllToAllPackKernel<<<blocks,SPARK_TP_MESH_THREADS,0u,stream>>>((uint16_t *)scratch,(const uint16_t *)local,degree,per_peer_elements,begin,count,slice);
        status = cudaPeekAtLastError();
        if ( status == cudaSuccess )
            status = SparkTpMeshHardwareExchange(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,(const uint8_t *)scratch,span * sizeof(uint16_t),0u,span * sizeof(uint16_t),route,timeout_ns);
        if ( status != cudaSuccess )
            return status;
        blocks = (uint32_t)(((uint64_t)degree * count + SPARK_TP_MESH_THREADS - 1u) / SPARK_TP_MESH_THREADS);
        SparkTpMeshHardwareAllToAllKernel<<<blocks,SPARK_TP_MESH_THREADS,0u,stream>>>(band,slot_bytes,slots_per_rank,control,degree,rank,output,per_peer_elements,begin,count,slice,begin + count == per_peer_elements);
        status = cudaPeekAtLastError();
    }
    return status;
}

extern "C" cudaError_t SparkTpLaunchMeshHardware(cudaStream_t stream,
    void *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile void *entry,
    void *gate,void *round_control,uint32_t rank,uint32_t degree,
    const void *local,void *output,void *scratch,uint64_t elements,
    uint32_t operation,uint32_t rounds,uint32_t logical_rows,uint32_t slice_routes,void *staging,
    const SparkTpMeshSecondBand *second,uint64_t timeout_ns)
{
    if ( band == 0 || entry == 0 || gate == 0 || round_control == 0 || local == 0 ||
         output == 0 || scratch == 0 || elements == 0u || rounds == 0u || logical_rows == 0u ||
         degree < 2u || degree > 16u || rank >= degree || operation > SPARK_TP_MESH_OPERATION_ALL_TO_ALL ||
         slot_bytes <= 16u || slots_per_rank != 2u || timeout_ns == 0u ||
         (operation == 0u && (elements % degree != 0u || local == output)) ||
         (operation == SPARK_TP_MESH_OPERATION_ALL_TO_ALL && (local == output || slice_routes == 0u)) ||
         ((slice_routes & SPARK_TP_MESH_ROUTES_PEER) != 0u && staging == 0) ||
         (second != 0 && (second->band == 0 || second->entry == 0 || second->gate == 0 || second->band == band ||
             ((slice_routes & SPARK_TP_MESH_ROUTES_PEER) != 0u && second->staging == 0))) )
        return cudaErrorInvalidValue;
    if ( operation == SPARK_TP_MESH_OPERATION_ALL_TO_ALL )
    {
        for ( uint32_t round = 0u; round < rounds; round++ )
        {
            cudaError_t status = (slice_routes & SPARK_TP_MESH_ROUTES_PEER) != 0u ?
                SparkTpMeshHardwarePeerAllToAllRound(stream,(uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,(SparkWeightdMeshWaitRequest *)gate,(SparkTpMeshRoundControl *)round_control,rank,degree,(const uint8_t *)local,output,(uint8_t *)staging,elements,timeout_ns) :
                SparkTpMeshHardwareAllToAllRound(stream,(uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,(SparkWeightdMeshWaitRequest *)gate,(SparkTpMeshRoundControl *)round_control,rank,degree,(const uint8_t *)local,output,scratch,elements,timeout_ns);
            if ( status != cudaSuccess ) return status;
        }
        return cudaSuccess;
    }
    SparkTpMeshRoundControl *control = (SparkTpMeshRoundControl *)round_control;
    SparkWeightdMeshWaitRequest *request = (SparkWeightdMeshWaitRequest *)gate;
    uint32_t levels = SparkTpMeshTreeLevels(degree);
    uint32_t width = operation == 2u ? 8u : operation == 1u ? 4u : 2u;
    uint64_t capacity = (slot_bytes - 16u) / width;
    if ( capacity == 0u ) return cudaErrorInvalidValue;
    if ( SparkTpMeshDirectCapacity(slot_bytes,operation) == 0u ) return cudaErrorInvalidValue;
    for ( uint32_t round = 0u; round < rounds; round++ )
    {
        cudaError_t status;
        if ( logical_rows == 1u )
        {
            status = SparkTpMeshHardwareDirectRound(stream,(uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,request,control,rank,degree,(const uint8_t *)local,output,elements,operation,slice_routes,(uint8_t *)staging,second,timeout_ns);
            if ( status != cudaSuccess ) return status;
            continue;
        }
        for ( uint64_t begin = 0u; begin < elements; begin += capacity )
        {
            uint64_t count = elements - begin < capacity ? elements - begin : capacity;
            SparkTpMeshHardwareSeedKernel<<<1,SPARK_TP_MESH_THREADS,0u,stream>>>(
                control,scratch,local,elements,begin,count,operation,rank,degree,2u * levels);
            status = cudaPeekAtLastError();
            if ( status != cudaSuccess ) return status;
            for ( uint32_t phase = 0u; phase < 2u * levels; phase++ )
            {
                uint32_t route = SparkTpMeshTreeRoute(rank,degree,phase);
                uint32_t send = route >> 16u,receive = route & 0xffffu;
                if ( send != 0u )
                {
                    status = SparkTpMeshHardwareWait(stream,request,control,
                        SPARK_WEIGHTD_MESH_WAIT_SHIPPED,0u,0u,timeout_ns,0u);
                    if ( status != cudaSuccess ) return status;
                    SparkTpMeshHardwarePublishKernel<<<1,SPARK_TP_MESH_THREADS,0u,stream>>>(
                        (uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,control,
                        rank,scratch,count * width,0u,count * width,1u << (send - 1u),phase + 1u,0u,0,0u,0u,0u,0u);
                }
                else if ( receive != 0u )
                {
                    status = SparkTpMeshHardwareWait(stream,request,control,
                        SPARK_WEIGHTD_MESH_WAIT_PEERS,1u << (receive - 1u),phase + 1u,timeout_ns,0u);
                    if ( status != cudaSuccess ) return status;
                    SparkTpMeshHardwareFoldKernel<<<1,SPARK_TP_MESH_THREADS,0u,stream>>>(
                        (const uint8_t *)band,slot_bytes,slots_per_rank,control,scratch,count,
                        operation,receive - 1u,phase,levels);
                }
                status = cudaPeekAtLastError();
                if ( status != cudaSuccess ) return status;
            }
            SparkTpMeshHardwareFinishKernel<<<1,SPARK_TP_MESH_THREADS,0u,stream>>>(
                control,output,scratch,begin,count,operation,2u * levels,begin + count == elements);
            status = cudaPeekAtLastError();
            if ( status != cudaSuccess ) return status;
        }
    }
    return cudaSuccess;
}

#endif
