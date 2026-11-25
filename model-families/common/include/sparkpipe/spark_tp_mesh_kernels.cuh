#pragma once
#include <stdint.h>
#include <stddef.h>
#include "sparkpipe/spark_gpu.h"

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include "sparkpipe/spark_weightd.h"
#include <stdio.h>
#include "sparkpipe/spark_tp_mesh_round_control.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/mma.cuh"
#include "inference/kernels/weight_prefetch.cuh"
#include "runtime/launch.h"
#define SPARK_TP_MESH_KERNELS_MARKER "SPARK-TP-MESH-KERNELS-V17-HALF-SLOTS"
#define SPARK_TP_MESH_ERROR_PARITY_MISMATCH 0xFFFFFFFFFF000000ull
#define SPARK_TP_MESH_ERROR_CANCELLED 0xFFFFFFFFFE000000ull
#if defined(__CUDACC__)
__constant__ char SparkTpMeshKernelsBuildMarker[] =
    SPARK_TP_MESH_KERNELS_MARKER;
__attribute__((used)) static const char SparkTpMeshKernelsBuildMarkerHost[] =
    SPARK_TP_MESH_KERNELS_MARKER;
#endif

#define SPARK_TP_MESH_THREADS 256u
#define SPARK_TP_MESH_COPY_BLOCKS 192u
#define SPARK_TP_MESH_DIRECT_THREADS 128u
#define SPARK_TP_MESH_PIECE_RESIDENCY 4u
#define SPARK_TP_MESH_DEPENDENT_ELEMENTS 65536u
#define SPARK_TP_MESH_SMALL_MESSAGE_ELEMENTS 131072u
#define SPARK_TP_MESH_EARLY_ELEMENTS 16384u
#define SPARK_TP_MESH_FUSED_GATHER_ELEMENTS 131072u
#define SPARK_TP_MESH_FUSED_STAGE_ELEMENTS 131072u
#define SPARK_TP_MESH_EARLY_BLOCKS 4u
#define SPARK_TP_MESH_PREFETCH_THREADS 32u
#define SPARK_TP_MESH_WIDE_PUBLISH_BYTES 65536u
#define SPARK_TP_MESH_WIDE_PUBLISH_UNIT (SPARK_TP_MESH_THREADS * 16u * 4u)

struct SparkTpU256 { uint4 lo,hi; };

static __device__ __forceinline__ unsigned long long SparkTpLdcvU64(const volatile void *address)
{ unsigned long long v; asm volatile("ld.global.cv.u64 %0,[%1];" : "=l"(v) : "l"(address) : "memory"); return(v); }
static __device__ __forceinline__ unsigned long long SparkTpLdAcquireU64(const volatile void *address)
{ unsigned long long v; asm volatile("ld.acquire.sys.global.u64 %0,[%1];" : "=l"(v) : "l"(address) : "memory"); return(v); }
static __device__ __forceinline__ unsigned long long SparkTpGlobalTimerNs(void)
{ unsigned long long ns; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns) :: "memory"); return ns; }
static __device__ __forceinline__ uint4 SparkTpLdCvU128(const volatile void *address)
{ uint4 v; asm volatile("ld.global.cv.v4.u32 {%0,%1,%2,%3},[%4];" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(address) : "memory"); return(v); }
static __device__ __forceinline__ void SparkTpStVolatileU128(volatile void *address,uint4 v)
{ asm volatile("st.volatile.global.v4.u32 [%0],{%1,%2,%3,%4};" :: "l"(address),"r"(v.x),"r"(v.y),"r"(v.z),"r"(v.w) : "memory"); }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
static __device__ __forceinline__ SparkTpU256 SparkTpLdCvU256(const volatile void *a)
{ SparkTpU256 v; asm volatile("ld.global.cv.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];" : "=r"(v.lo.x),"=r"(v.lo.y),"=r"(v.lo.z),"=r"(v.lo.w),"=r"(v.hi.x),"=r"(v.hi.y),"=r"(v.hi.z),"=r"(v.hi.w) : "l"(a) : "memory"); return(v); }
static __device__ __forceinline__ SparkTpU256 SparkTpLdNcU256(const void *a)
{ SparkTpU256 v; asm volatile("ld.global.nc.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];" : "=r"(v.lo.x),"=r"(v.lo.y),"=r"(v.lo.z),"=r"(v.lo.w),"=r"(v.hi.x),"=r"(v.hi.y),"=r"(v.hi.z),"=r"(v.hi.w) : "l"(a) : "memory"); return(v); }
static __device__ __forceinline__ void SparkTpStU256(void *a,SparkTpU256 v)
{ asm volatile("st.global.v8.u32 [%0],{%1,%2,%3,%4,%5,%6,%7,%8};" :: "l"(a),"r"(v.lo.x),"r"(v.lo.y),"r"(v.lo.z),"r"(v.lo.w),"r"(v.hi.x),"r"(v.hi.y),"r"(v.hi.z),"r"(v.hi.w) : "memory"); }
static __device__ __forceinline__ void SparkTpStVolatileU256(volatile void *a,SparkTpU256 v)
{ asm volatile("st.volatile.global.v8.u32 [%0],{%1,%2,%3,%4,%5,%6,%7,%8};" :: "l"(a),"r"(v.lo.x),"r"(v.lo.y),"r"(v.lo.z),"r"(v.lo.w),"r"(v.hi.x),"r"(v.hi.y),"r"(v.hi.z),"r"(v.hi.w) : "memory"); }
#else
static __device__ __forceinline__ SparkTpU256 SparkTpLdCvU256(const volatile void *a)
{ SparkTpU256 v; v.lo = SparkTpLdCvU128(a); v.hi = SparkTpLdCvU128((const volatile uint8_t *)a + 16u); return(v); }
static __device__ __forceinline__ SparkTpU256 SparkTpLdNcU256(const void *a)
{ SparkTpU256 v; v.lo = ((const uint4 *)a)[0]; v.hi = ((const uint4 *)a)[1]; return(v); }
static __device__ __forceinline__ void SparkTpStU256(void *a,SparkTpU256 v) { ((uint4 *)a)[0] = v.lo; ((uint4 *)a)[1] = v.hi; }
static __device__ __forceinline__ void SparkTpStVolatileU256(volatile void *a,SparkTpU256 v)
{ SparkTpStVolatileU128(a,v.lo); SparkTpStVolatileU128((volatile uint8_t *)a + 16u,v.hi); }
#endif
static __device__ __forceinline__ uint32_t SparkTpMeshAligned16(const volatile void *a,const volatile void *b) { return(((((uintptr_t)a) | ((uintptr_t)b)) & 15u) == 0u ? 1u : 0u); }
static __device__ __forceinline__ uint32_t SparkTpMeshAligned32(const volatile void *a,const volatile void *b) { return(((((uintptr_t)a) | ((uintptr_t)b)) & 31u) == 0u ? 1u : 0u); }

static __device__ __forceinline__ void SparkTpMeshCopyHalves(uint16_t *destination,const uint16_t *source,uint64_t count,uint64_t lane,uint64_t lanes)
{
	uint64_t i,done = 0u;
	if ( SparkTpMeshAligned32(destination,source) != 0u ) { uint64_t octs = count / 16u; for ( i = lane; i < octs; i += lanes ) SparkTpStU256(destination + i * 16u,SparkTpLdNcU256(source + i * 16u)); done = octs * 16u; }
	if ( SparkTpMeshAligned16(destination + done,source + done) != 0u ) { uint64_t quads = (count - done) / 8u; for ( i = lane; i < quads; i += lanes ) ((uint4 *)(destination + done))[i] = ((const uint4 *)(source + done))[i]; done += quads * 8u; }
	for ( i = done + lane; i < count; i += lanes ) destination[i] = source[i];
}

static __device__ __forceinline__ void SparkTpMeshCopyHalvesFromBand(uint16_t *destination,const volatile uint16_t *source,uint64_t count,uint64_t lane,uint64_t lanes)
{
	uint64_t i,done = 0u;
	if ( SparkTpMeshAligned32(destination,source) != 0u ) { uint64_t octs = count / 16u; for ( i = lane; i < octs; i += lanes ) SparkTpStU256(destination + i * 16u,SparkTpLdCvU256(source + i * 16u)); done = octs * 16u; }
	if ( SparkTpMeshAligned16(destination + done,source + done) != 0u ) { uint64_t quads = (count - done) / 8u; for ( i = lane; i < quads; i += lanes ) ((uint4 *)(destination + done))[i] = SparkTpLdCvU128(source + done + i * 8u); done += quads * 8u; }
	for ( i = done + lane; i < count; i += lanes ) destination[i] = source[i];
}

static __device__ __forceinline__ void SparkTpMeshCopyBytesVolatile(volatile uint8_t *destination,const void *source,uint64_t bytes,uint64_t lane,uint64_t lanes)
{
	const uint8_t *src = (const uint8_t *)source;
	uint64_t i,done = 0u;
	if ( SparkTpMeshAligned32(destination,src) != 0u ) { uint64_t octs = bytes / 32u; for ( i = lane; i < octs; i += lanes ) SparkTpStVolatileU256(destination + i * 32u,SparkTpLdNcU256(src + i * 32u)); done = octs * 32u; }
	if ( SparkTpMeshAligned16(destination + done,src + done) != 0u ) { uint64_t quads = (bytes - done) / 16u; for ( i = lane; i < quads; i += lanes ) SparkTpStVolatileU128(destination + done + i * 16u,((const uint4 *)(src + done))[i]); done += quads * 16u; }
	if ( (((uintptr_t)(destination + done) | (uintptr_t)(src + done)) & 7u) == 0u ) { uint64_t words = (bytes - done) / 8u; for ( i = lane; i < words; i += lanes ) ((volatile uint64_t *)(destination + done))[i] = ((const uint64_t *)(src + done))[i]; done += words * 8u; }
	for ( i = done + lane; i < bytes; i += lanes ) destination[i] = src[i];
}

static inline uint32_t SparkTpMeshMultiprocessors(void)
{
	static int multiprocessors = 0;
	int device = 0,count = 0;
	if ( multiprocessors == 0 && SparkGpuGetDevice(&device) == SPARK_GPU_SUCCESS &&
		SparkGpuDeviceGetAttribute(&count,SPARK_GPU_ATTRIBUTE_MULTIPROCESSOR_COUNT,device) == SPARK_GPU_SUCCESS && count > 0 )
		multiprocessors = count;
	return(multiprocessors > 0 ? (uint32_t)multiprocessors : 1u);
}

static inline uint32_t SparkTpMeshDirectBlocks(uint64_t elements,uint32_t peers)
{
	uint64_t needed = (elements / 8u + SPARK_TP_MESH_DIRECT_THREADS - 1u) / SPARK_TP_MESH_DIRECT_THREADS;
	uint64_t limit = SPARK_TP_MESH_PIECE_RESIDENCY * SparkTpMeshMultiprocessors() / (peers != 0u ? peers : 1u);
	limit = limit != 0u ? limit : 1u;
	return((uint32_t)(needed == 0u ? 1u : needed < limit ? needed : limit));
}

static inline uint32_t SparkTpMeshPieceBlocks(uint64_t elements,uint32_t peers) { return(SparkTpMeshDirectBlocks(elements,peers)); }

static inline uint32_t SparkTpMeshCopyBlocks(uint64_t elements,uint32_t peers)
{
	uint64_t needed = (elements + SPARK_TP_MESH_THREADS - 1u) / SPARK_TP_MESH_THREADS;
	uint64_t limit = SPARK_TP_MESH_COPY_BLOCKS / (peers != 0u ? peers : 1u);
	limit = limit != 0u ? limit : 1u;
	return((uint32_t)(needed == 0u ? 1u : needed < limit ? needed : limit));
}

static inline uint32_t SparkTpMeshCombineFused(uint32_t operation,uint64_t count,uint64_t span,uint32_t degree)
{
	if ( operation == SPARK_TP_MESH_OPERATION_ALL_GATHER || operation == SPARK_TP_MESH_OPERATION_SLICE_GATHER )
		return((operation == SPARK_TP_MESH_OPERATION_ALL_GATHER ? count * degree : count) <= SPARK_TP_MESH_FUSED_GATHER_ELEMENTS ? 1u : 0u);
	return(span <= SPARK_TP_MESH_EARLY_ELEMENTS ? 1u : 0u);
}

static inline dim3 SparkTpMeshFusedGrid(uint32_t operation,uint64_t span,uint32_t degree)
{
	if ( operation == SPARK_TP_MESH_OPERATION_ALL_GATHER || operation == SPARK_TP_MESH_OPERATION_SLICE_GATHER )
	{
		uint32_t limit = span > 32768u ? 8u : SPARK_TP_MESH_EARLY_BLOCKS;
		return(dim3(1u,degree < limit ? degree : limit));
	}
	uint64_t needed = (span / 8u + SPARK_TP_MESH_THREADS - 1u) / SPARK_TP_MESH_THREADS;
	return(dim3((uint32_t)(needed == 0u ? 1u : needed < SPARK_TP_MESH_EARLY_BLOCKS ? needed : SPARK_TP_MESH_EARLY_BLOCKS),1u));
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

static __device__ __forceinline__ void SparkTpMeshFaultRecord(SparkTpMeshRoundControl *control, uint64_t phase, uint64_t peer, uint64_t want,
	uint64_t seen)
{
	atomicAdd((unsigned long long *)&control->fault_hits,1ull);
	if ( atomicCAS((unsigned long long *)&control->fault_phase,0ull,(unsigned long long)phase) == 0ull )
	{
		control->fault_peer = peer;
		control->fault_want = want;
		control->fault_seen = seen;
	}
}

__global__ void SparkTpMeshGuardKernel(
    volatile unsigned long long *error_word,
    unsigned long long *output)
{
	if ( threadIdx.x != 0u || blockIdx.x != 0u )
		return;
	if ( *error_word != 0ull )
	{
		SparkTpMeshRoundControl *control = (SparkTpMeshRoundControl *)((uintptr_t)error_word - offsetof(SparkTpMeshRoundControl,error_word));
		output[0] = 0xFFFFFFFFFFFFFFFFull;
		atomicAdd((unsigned long long *)&control->guard_poisons,1ull);
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
			end_word = (volatile uint64_t *)
				((uint8_t *)band_base +
				((uint64_t)peer_rank * slots_per_rank +
					ring) * slot_bytes +
				slot_bytes - 8u);
			while ( (SparkTpLdcvU64(end_word) >> 32ull) != (sequence >> 32ull) ||
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

static __device__ __forceinline__ uint16_t SparkTpFloatToBf16(float value)
{
	return(__bfloat16_as_ushort(__float2bfloat16_rn(value)));
}

static __device__ __forceinline__ void SparkTpStoreBf16Pair(void *base,uint64_t element,float x,float y)
{
	((uint32_t *)base)[element] = LmPackBf16Pair(x,y);
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
					const uint64_t shipped = *shipped_cell;
					control->diag_word = (0xa5ull << 56ull) |
					    ((prev_tag & 0xffffull) << 16ull) |
					    (shipped & 0xffffull);
					SparkTpMeshFaultRecord(control,SPARK_TP_MESH_FAULT_SHIP_ACK,rank,prev_tag,shipped);
					control->error_word = prev_tag;
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
						const uint64_t ended = *end_word;
						control->diag_word =
						    ((unsigned long long)peer_rank << 56ull) |
						    ((cursor & 0xffull) << 48ull) |
						    ((unsigned long long)
							(s_slot / slot_bytes) << 32ull) |
						    ((s_tag & 0xffffull) << 16ull) |
						    (ended & 0xffffull);
						SparkTpMeshFaultRecord(control,SPARK_TP_MESH_FAULT_PEER_WAIT,peer_rank,(uint32_t)s_tag,(uint32_t)ended);
						control->error_word = s_tag;
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
        if ( operation == 1u ) ((uint16_t *)output)[begin + i] = SparkTpFloatToBf16(((const float *)scratch)[i]);
        else if ( operation == 2u ) ((uint64_t *)output)[begin + i] = ((const uint64_t *)scratch)[i];
        else ((uint16_t *)output)[begin + i] = ((const uint16_t *)scratch)[i];
    }
}

static __device__ __forceinline__ uint32_t SparkTpMeshDeviceTreeLevels(uint32_t degree) { uint32_t l = 0u; while ( (1u << l) < degree ) l++; return l; }
static __device__ __forceinline__ uint32_t SparkTpMeshDeviceTreeRoute(uint32_t rank,uint32_t degree,uint32_t phase)
{
    uint32_t levels = SparkTpMeshDeviceTreeLevels(degree),reduce = phase < levels,step = 1u << (reduce != 0u ? phase : 2u * levels - phase - 1u),residue = rank % (2u * step);
    uint32_t peer = residue == 0u && rank + step < degree ? rank + step : residue == step ? rank - step : UINT32_MAX;
    return peer == UINT32_MAX ? 0u : ((residue == step) == (reduce != 0u) ? (peer + 1u) << 16u : peer + 1u);
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
    uint32_t levels = SparkTpMeshDeviceTreeLevels(degree);
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
                uint32_t route = SparkTpMeshDeviceTreeRoute(rank,degree,phase);
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

extern "C" SparkGpuError SparkTpLaunchMeshTree(SparkGpuStream stream,
    void *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile void *entry,
    const volatile void *shipped,const volatile void *cancel,void *round_control,
    uint32_t rank,uint32_t degree,const void *local,void *output,void *scratch,
    uint64_t elements,uint32_t operation,uint32_t rounds,uint64_t timeout_ns)
{
    LM_LAUNCH((SparkTpMeshTreeKernel),1,SPARK_TP_MESH_THREADS,0u,stream,
        (uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,
        (const volatile uint64_t *)shipped,(const volatile uint64_t *)cancel,
        (SparkTpMeshRoundControl *)round_control,rank,degree,local,output,scratch,
        elements,operation,rounds,timeout_ns);
    return SparkGpuPeekAtLastError();
}

extern "C" SparkGpuError SparkTpLaunchMeshRoundLoop(SparkGpuStream stream,
	volatile void *band_base,uint64_t slot_bytes,uint64_t slots_per_rank,
	volatile void *entry,void *shipped_cell,volatile void *cancel_cell,
	void *round_control,uint32_t rank,uint32_t degree,
	const void *local_device,void *full_device,uint64_t bytes)
{
	LM_LAUNCH((SparkTpMeshRoundLoopKernel),1,SPARK_TP_MESH_THREADS,0u,stream,
		(volatile uint64_t *)band_base,slot_bytes,slots_per_rank,
		(volatile uint64_t *)entry,(volatile uint64_t *)shipped_cell,
		(volatile uint64_t *)cancel_cell,
		(SparkTpMeshRoundControl *)round_control,rank,degree,
		local_device,full_device,bytes);
	return SparkGpuPeekAtLastError();
}


struct SparkTpRankSources
{
	const void *pointer[16u];
};

static __global__ void SparkTpSumRanksF32Kernel(
    void *destination_bf16,SparkTpRankSources sources,uint32_t source_count,uint32_t pair_count)
{
	for ( uint32_t pair = blockIdx.x * blockDim.x + threadIdx.x; pair < pair_count; pair += blockDim.x * gridDim.x )
	{
		float2 acc = {0.0f,0.0f};
#pragma unroll
		for ( uint32_t source = 0u; source < 16u; source++ )
			if ( source < source_count )
			{
				float2 v = SparkTpLoadBf16Pair(sources.pointer[source],pair);
				acc.x += v.x;
				acc.y += v.y;
			}
		SparkTpStoreBf16Pair(destination_bf16,pair,acc.x,acc.y);
	}
}

extern "C" SparkGpuError SparkTpLaunchSumRanksF32(SparkGpuStream stream,
    void *destination,const void *const *sources,uint32_t source_count,
    uint32_t element_count)
{
	SparkTpRankSources by_value;
	uint32_t index;
	if ( destination == 0 || sources == 0 || source_count == 0u ||
	     source_count > 16u || element_count == 0u )
		return(SPARK_GPU_ERROR_INVALID_VALUE);
	for ( index = 0u; index < source_count; index++ )
		by_value.pointer[index] = sources[index];
	{
		dim3 grid;
		uint32_t pairs = (element_count + 1u) / 2u;
		uint32_t rows = (pairs + 255u) / 256u;
		grid = dim3(rows < 1u ? 1u : rows,1u,1u);
		LM_LAUNCH((SparkTpSumRanksF32Kernel),grid,256u,0u,stream,
		    destination,by_value,source_count,pairs);
	}
	return SparkGpuPeekAtLastError();
}

static __global__ void SparkTpGatherRanksKernel(
    uint16_t *destination,SparkTpRankSources sources,uint32_t source_count,uint32_t elements_per_rank)
{
	uint64_t index,total = (uint64_t)source_count * elements_per_rank;
	for ( index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < total; index += (uint64_t)blockDim.x * gridDim.x )
	{
		uint32_t rank = (uint32_t)(index / elements_per_rank);
		const uint16_t *src = 0;
#pragma unroll
		for ( uint32_t s = 0u; s < 16u; s++ )
			if ( s == rank ) src = (const uint16_t *)sources.pointer[s];
		destination[index] = src[index % elements_per_rank];
	}
}

extern "C" SparkGpuError SparkTpLaunchGatherRanks(SparkGpuStream stream,
    void *destination,const void *const *sources,uint32_t source_count,
    uint32_t elements_per_rank)
{
	SparkTpRankSources by_value;
	uint64_t blocks;
	uint32_t index;
	if ( destination == 0 || sources == 0 || source_count == 0u ||
	     source_count > 16u || elements_per_rank == 0u )
		return(SPARK_GPU_ERROR_INVALID_VALUE);
	for ( index = 0u; index < source_count; index++ )
	{
		if ( sources[index] == 0 )
			return(SPARK_GPU_ERROR_INVALID_VALUE);
		by_value.pointer[index] = sources[index];
	}
	blocks = ((uint64_t)source_count * elements_per_rank + 255u) / 256u;
	LM_LAUNCH((SparkTpGatherRanksKernel),(uint32_t)(blocks < 65535u ? blocks : 65535u),256u,0u,stream,
	    (uint16_t *)destination,by_value,source_count,elements_per_rank);
	return SparkGpuPeekAtLastError();
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

extern "C" SparkGpuError SparkTpLaunchAccumAdd(SparkGpuStream stream,void *destination_bf16,const void *source_bf16,uint32_t row_count,uint32_t width)
{
	if ( destination_bf16 == 0 || source_bf16 == 0 || row_count == 0u || width == 0u || (width & 1u) != 0u )
		return(SPARK_GPU_ERROR_INVALID_VALUE);
	LM_LAUNCH((SparkTpAccumAddKernel),row_count,256u,0u,stream,destination_bf16,source_bf16,row_count,width);
	return(SparkGpuPeekAtLastError());
}

extern "C" SparkGpuError SparkTpLaunchAccumU64Max(SparkGpuStream stream,uint64_t *destination,const uint64_t *source,uint32_t element_count)
{
	if ( destination == 0 || source == 0 || element_count == 0u )
		return(SPARK_GPU_ERROR_INVALID_VALUE);
	LM_LAUNCH((SparkTpAccumU64MaxKernel),(element_count + 255u) / 256u,256u,0u,stream,destination,source,element_count);
	return(SparkGpuPeekAtLastError());
}

extern "C" SparkGpuError SparkTpLaunchMeshGuard(SparkGpuStream stream,
	volatile void *error_word,void *output)
{
	LM_LAUNCH((SparkTpMeshGuardKernel),1,32,0u,stream,
		(volatile unsigned long long *)error_word,
		(unsigned long long *)output);
	return SparkGpuPeekAtLastError();
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
            if ( SparkTpLdcvU64(&control->error_word) != 0ull )
            {
                ready = 0u;
                break;
            }
            if ( SparkTpLdcvU64(cancel) != expected_cancel )
            {
                atomicCAS((unsigned long long *)&control->error_word,0ull,
                    SPARK_TP_MESH_ERROR_CANCELLED | previous);
                ready = 0u;
                break;
            }
            if ( SparkTpGlobalTimerNs() >= deadline )
            {
                atomicCAS((unsigned long long *)&control->error_word,0ull,previous);
                ready = 0u;
                break;
            }
            __nanosleep(200u);
        }
        if ( SparkTpLdcvU64(cancel) != expected_cancel )
            atomicCAS((unsigned long long *)&control->error_word,0ull,
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

extern "C" SparkGpuError SparkTpLaunchMeshCopyDown(
    SparkGpuStream stream,void *destination,const void *source,
    uint64_t bytes,const volatile void *shipped,void *round_control,
    const volatile void *cancel,uint64_t timeout_ns)
{
    uint64_t quads = bytes / 8u + 1u;
    if ( destination == 0 || source == 0 || bytes == 0u || shipped == 0 ||
         round_control == 0 || cancel == 0 || timeout_ns == 0u )
        return(SPARK_GPU_ERROR_INVALID_VALUE);
    LM_LAUNCH((SparkTpMeshCopyDownKernel),(quads + 255u) / 256u,256u,0u,stream,
        (volatile uint8_t *)destination,(const uint8_t *)source,bytes,
        (const volatile uint64_t *)shipped,(SparkTpMeshRoundControl *)round_control,
        (const volatile uint64_t *)cancel,timeout_ns);
    return(SparkGpuPeekAtLastError());
}

extern "C" SparkGpuError SparkTpLaunchMeshPublish(SparkGpuStream stream,
    volatile void *entry,void *seq_cell,const void *epoch_cell,
    void *round_seq,uint64_t bytes,
    uint64_t slot_index,uint64_t slots_per_rank,volatile void *slot_tail,
    void *error_word,uint32_t peer_mask)
{
	LM_LAUNCH((SparkTpMeshPublishKernel),1,32,0u,stream,
		(volatile uint64_t *)entry,(unsigned long long *)seq_cell,
		(const unsigned long long *)epoch_cell,
		(unsigned long long *)round_seq,bytes,slot_index,slots_per_rank,
		(volatile uint64_t *)slot_tail,
		(unsigned long long *)error_word,peer_mask);
	return SparkGpuPeekAtLastError();
}

extern "C" SparkGpuError SparkTpLaunchMeshSeqPad(SparkGpuStream stream,
    void *seq_cell)
{
	LM_LAUNCH((SparkTpMeshSeqPadKernel),1,32,0u,stream,
		(unsigned long long *)seq_cell);
	return SparkGpuPeekAtLastError();
}

extern "C" SparkGpuError SparkTpLaunchMeshWait(SparkGpuStream stream,
    volatile void *band_base,uint64_t slot_bytes,const void *round_seq,
    uint64_t slots_per_rank,uint32_t rank,uint32_t degree,
    void *error_word,unsigned long long deadline_ns,void *diag_word,
    volatile void *cancel_cell,const void *cancel_expected,
    void *arrival_ring)
{
	LM_LAUNCH((SparkTpMeshWaitKernel),1,32,0u,stream,
		(volatile uint64_t *)band_base,slot_bytes,
		(const unsigned long long *)round_seq,slots_per_rank,rank,
		degree,(unsigned long long *)error_word,deadline_ns,
		(unsigned long long *)diag_word,
		(volatile uint64_t *)cancel_cell,
		(const unsigned long long *)cancel_expected,
		(unsigned long long *)arrival_ring);
	return SparkGpuPeekAtLastError();
}








#define SPARK_TP_MESH_PUBLISH_GUARD 1u
#define SPARK_TP_MESH_PUBLISH_REQUEST 2u
#define SPARK_TP_MESH_PUBLISH_SHIP 4u
#define SPARK_TP_MESH_PUBLISH_ARRIVAL 8u
#define SPARK_TP_MESH_PUBLISH_HALF 16u
#define SPARK_TP_MESH_PUBLISH_SHIP_PRIOR 32u
#define SPARK_TP_MESH_VIEW_HALF 1u
#define SPARK_TP_MESH_VIEW_LAG 2u
#define SPARK_TP_MESH_SHIP_POLL_NS 4000000ull

static __device__ void SparkTpMeshGateRequest(
    volatile SparkWeightdMeshWaitRequest *gate,SparkTpMeshRoundControl *control,
    uint64_t kind,uint64_t peer_mask,uint64_t phase_offset,uint64_t timeout_ns)
{
    uint64_t previous = SparkTpLdcvU64(&gate->request_id);
    if ( previous == UINT64_MAX )
    {
        control->error_word = UINT64_MAX;
        return;
    }
    if ( control->seq > UINT32_MAX - phase_offset ) control->error_word = UINT64_MAX;
    uint64_t tag = kind == SPARK_WEIGHTD_MESH_WAIT_SHIPPED ? control->round_seq :
        (control->epoch << 32u) | ((control->seq + phase_offset) & UINT32_MAX);
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
    __shared__ volatile uint64_t s_started;
    uint64_t limit = SparkTpLdcvU64(&gate->timeout_ns);
    s_started = SparkTpGlobalTimerNs();
    limit = limit > UINT64_MAX / 2u ? UINT64_MAX : 2u * limit;
    while ( SparkTpLdAcquireU64(&gate->ready) != 1u )
    {
        const uint64_t base = s_started,now = SparkTpGlobalTimerNs();
        if ( now > base && now - base > limit )
        {
            if ( control->error_word == 0u )
            {
                control->error_word = UINT64_MAX;
                control->diag_word = SparkTpLdcvU64(&gate->request_id);
            }
            return;
        }
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
    control->math_started_ns = 0u;
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

static __device__ __forceinline__ uint64_t SparkTpMeshArrivalPoll(
    const volatile uint64_t *tail,const volatile uint64_t *cancel,uint64_t expected,uint64_t tag,uint64_t started,uint64_t timeout_ns,
    uint32_t lane,uint32_t missing,uint64_t &landed,uint64_t &diag)
{
    __shared__ volatile uint64_t s_poll_started;
    uint64_t seen = 0u,fault = 0u;
    uint32_t first;
    if ( lane == 0u ) s_poll_started = started;
    for (;;)
    {
        if ( missing != 0u && (seen = SparkTpLdAcquireU64(tail)) == tag )
        {
            missing = 0u;
            landed = SparkTpGlobalTimerNs();
        }
        if ( __all_sync(0xffffffffu,missing == 0u) ) break;
        if ( lane == 0u )
        {
            const uint64_t base = s_poll_started,now = SparkTpGlobalTimerNs();
            if ( cancel != 0 && (diag = SparkTpLdAcquireU64(cancel)) != expected ) fault = SPARK_WEIGHTD_MESH_WAIT_ERROR_CANCELLED | tag;
            else if ( now > base && now - base > timeout_ns ) fault = tag;
        }
        fault = __shfl_sync(0xffffffffu,fault,0);
        if ( fault == 0u ) continue;
        first = __ffs(__ballot_sync(0xffffffffu,missing != 0u)) - 1u;
        seen = __shfl_sync(0xffffffffu,seen,first);
        if ( fault == tag ) diag = seen;
        break;
    }
    return fault;
}

static __device__ __forceinline__ void SparkTpMeshArrivalRecord(
    SparkTpMeshRoundControl *control,uint32_t lane,uint64_t base,uint64_t wait_end,uint64_t landed)
{
    if ( lane < SPARK_TP_MESH_ROUND_CONTROL_PEERS && landed != UINT64_MAX && landed > base )
        control->peer_arrival_ns[lane] += landed - base;
    for ( uint32_t step = 16u; step != 0u; step >>= 1u )
    {
        uint64_t other = __shfl_xor_sync(0xffffffffu,landed,step);
        landed = other < landed ? other : landed;
    }
    if ( lane == 0u )
    {
        control->peer_wait_ns += wait_end - base;
        control->first_arrival_ns += landed > base && landed != UINT64_MAX ? landed - base : 0u;
    }
}

static __device__ __forceinline__ uint32_t SparkTpMeshArrivalWait(
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,uint64_t timeout_ns,
    LmWeightPrefetchGroup *prefetch,uint32_t resume,uint32_t view,uint32_t reset_math,
    uint64_t &base_out,uint64_t &wait_end_out,uint64_t &landed_out)
{
    __shared__ uint32_t wait_over;
    __shared__ uint32_t failed;
    __shared__ volatile uint64_t s_arr_start;
    const uint32_t lane = threadIdx.x;
    base_out = 0u; wait_end_out = 0u; landed_out = UINT64_MAX;
    if ( lane == 0u ) { wait_over = 0u; failed = SparkTpLdcvU64(&control->error_word) != 0u ? 1u : 0u; s_arr_start = SparkTpGlobalTimerNs(); }
    __syncthreads();
    if ( failed != 0u ) return 0u;
    if ( lane < 32u )
    {
        const uint64_t tag = control->round_seq - ((view & SPARK_TP_MESH_VIEW_LAG) != 0u ? 1u : 0u);
        const uint64_t ring = (tag - 1u) & (slots_per_rank - 1u);
        const volatile uint64_t *tail = (const volatile uint64_t *)(band + ((uint64_t)lane * slots_per_rank + ring) * slot_bytes +
            ((view & SPARK_TP_MESH_VIEW_HALF) != 0u ? SPARK_WEIGHTD_MESH_TAIL_OFFSET(SPARK_WEIGHTD_MESH_SEQUENCE_HALF(tag)) : slot_bytes - 8u));
        const volatile uint64_t *cancel = (const volatile uint64_t *)(uintptr_t)control->cancel_cell;
        const uint64_t expected = control->cancel_expected,started = s_arr_start;
        uint64_t diag = 0u,fault;
        base_out = view != 0u ? started : control->wait_started_ns;
        fault = SparkTpMeshArrivalPoll(tail,cancel,expected,tag,started,timeout_ns,lane,lane < degree && lane != rank ? 1u : 0u,landed_out,diag);
        if ( lane == 0u ) { *(volatile uint32_t *)&wait_over = 1u; wait_end_out = SparkTpGlobalTimerNs(); }
        if ( reset_math != 0u ) LmDependentRelease();
        if ( fault != 0u && lane == 0u )
        {
            if ( control->error_word == 0u ) { control->error_word = fault; control->diag_word = diag; __threadfence(); }
            failed = 1u;
        }
        if ( reset_math != 0u )
        {
            if ( fault == 0u ) SparkTpMeshArrivalRecord(control,lane,base_out,wait_end_out,landed_out);
            if ( lane == 0u ) { control->math_started_ns = 0u; control->math_finished_ns = 0u; control->math_blocks_done = 0u; }
        }
    }
    else if ( prefetch != 0 && lane < 32u + SPARK_TP_MESH_PREFETCH_THREADS )
    {
        const volatile uint32_t *stop = &wait_over;
        const uint32_t spans = prefetch->count;
        const uint64_t extent = LmWeightPrefetchExtent(prefetch,spans),stride = (uint64_t)SPARK_TP_MESH_PREFETCH_THREADS * LM_WEIGHT_PREFETCH_PIECE_BYTES;
        uint64_t position = resume != 0u ? prefetch->progress : 0u;
        for ( ; position < extent && *stop == 0u; position += stride )
            LmWeightPrefetchPiece(prefetch,spans,position + (uint64_t)(lane - 32u) * LM_WEIGHT_PREFETCH_PIECE_BYTES);
        if ( lane == 32u ) prefetch->progress = position < extent ? position : extent;
    }
    __syncthreads();
    return failed == 0u ? 1u : 0u;
}

static __global__ void SparkTpMeshHardwareArrivalKernel(
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,uint64_t timeout_ns,
    LmWeightPrefetchGroup *prefetch,uint32_t resume,uint32_t view)
{
    uint64_t base,wait_end,landed;
    if ( blockIdx.x != 0u ) return;
    (void)SparkTpMeshArrivalWait(band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns,prefetch,resume,view,1u,base,wait_end,landed);
}

static __device__ __forceinline__ uint32_t SparkTpMeshShippedAccepts(const SparkTpMeshRoundControl *control,uint64_t shipped,uint32_t prior)
{
    return control->round_seq == 0u || shipped == control->round_seq ||
        (prior != 0u && control->round_seq_prior != 0u && shipped == control->round_seq_prior) ? 1u : 0u;
}

static __device__ uint32_t SparkTpMeshAlreadyShipped(const SparkTpMeshRoundControl *control,uint32_t prior)
{
    const volatile uint64_t *shipped = (const volatile uint64_t *)(uintptr_t)control->shipped_cell;
    return shipped != 0 && SparkTpMeshShippedAccepts(control,control->round_seq == 0u ? 0u : SparkTpLdAcquireU64(shipped),prior) != 0u ? 1u : 0u;
}

static __device__ uint32_t SparkTpMeshShipPoll(SparkTpMeshRoundControl *control)
{
    __shared__ volatile uint64_t s_started;
    const volatile uint64_t *shipped = (const volatile uint64_t *)(uintptr_t)control->shipped_cell;
    s_started = SparkTpGlobalTimerNs();
    if ( shipped == 0 ) return 0u;
    for (;;)
    {
        const uint64_t base = s_started,now = SparkTpGlobalTimerNs();
        if ( SparkTpMeshShippedAccepts(control,SparkTpLdAcquireU64(shipped),1u) != 0u )
        {
            if ( now > base ) control->source_wait_ns += now - base;
            return 1u;
        }
        if ( now > base && now - base > SPARK_TP_MESH_SHIP_POLL_NS ) return 0u;
    }
}

static __global__ void SparkTpMeshHardwareWaitKernel(
    volatile SparkWeightdMeshWaitRequest *gate,SparkTpMeshRoundControl *control,
    uint64_t kind,uint64_t peer_mask,uint64_t phase_offset,uint64_t timeout_ns)
{
    if ( threadIdx.x != 0u || blockIdx.x != 0u ) return;
    if ( kind == SPARK_WEIGHTD_MESH_WAIT_SHIPPED && SparkTpMeshAlreadyShipped(control,0u) != 0u ) return;
    SparkTpMeshGateRequest(gate,control,kind,peer_mask,phase_offset,timeout_ns);
    SparkTpMeshGateAwait(gate,control);
    SparkTpMeshGateGuard(gate,control);
}

static __device__ void SparkTpMeshShipGate(volatile SparkWeightdMeshWaitRequest *gate,SparkTpMeshRoundControl *control,uint32_t mode,uint64_t timeout_ns)
{
    const uint32_t prior = (mode & SPARK_TP_MESH_PUBLISH_SHIP_PRIOR) != 0u ? 1u : 0u;
    if ( (mode & SPARK_TP_MESH_PUBLISH_SHIP) != 0u && SparkTpMeshAlreadyShipped(control,prior) != 0u )
        return;
    if ( (mode & SPARK_TP_MESH_PUBLISH_SHIP) != 0u && prior != 0u && SparkTpMeshShipPoll(control) != 0u )
        return;
    if ( (mode & SPARK_TP_MESH_PUBLISH_SHIP) != 0u )
    {
        SparkTpMeshGateRequest(gate,control,SPARK_WEIGHTD_MESH_WAIT_SHIPPED,0u,0u,timeout_ns);
        SparkTpMeshGateAwait(gate,control);
    }
    if ( (mode & SPARK_TP_MESH_PUBLISH_GUARD) != 0u ) SparkTpMeshGateGuard(gate,control);
}

static __global__ void SparkTpMeshHardwarePublishKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    volatile uint64_t *entry,SparkTpMeshRoundControl *control,uint32_t rank,
    const void *source,uint64_t bytes,uint64_t offset,uint64_t doorbell_bytes,uint64_t route,uint32_t phase_offset,
    uint32_t advance,volatile SparkWeightdMeshWaitRequest *gate,uint32_t mode,uint64_t peer_mask,uint64_t timeout_ns)
{
    __shared__ uint32_t ready;
    const uint64_t started = SparkTpGlobalTimerNs();
    const uint32_t failed = control->error_word != 0u || control->seq > UINT32_MAX - phase_offset ? 1u : 0u;
    const uint64_t sequence = control->seq + phase_offset;
    const uint64_t tag = (control->epoch << 32u) | sequence;
    const uint64_t slot = (uint64_t)rank * slots_per_rank + ((sequence - 1u) & (slots_per_rank - 1u));
    const uint32_t half = (mode & SPARK_TP_MESH_PUBLISH_HALF) != 0u ? SPARK_WEIGHTD_MESH_SEQUENCE_HALF(sequence) : SPARK_WEIGHTD_MESH_ROUTE_HALF_WHOLE;
    const uint32_t prior = (mode & SPARK_TP_MESH_PUBLISH_SHIP_PRIOR) != 0u ? 1u : 0u;
    volatile uint8_t *destination = band + slot * slot_bytes;
    volatile uint8_t *payload = destination + SPARK_WEIGHTD_MESH_HALF_OFFSET(half) + offset;
    volatile uint64_t *tail = (volatile uint64_t *)(destination + (half != SPARK_WEIGHTD_MESH_ROUTE_HALF_WHOLE ? SPARK_WEIGHTD_MESH_TAIL_OFFSET(half) : slot_bytes - 8u));
    const volatile uint64_t *shipped_cell = (const volatile uint64_t *)(uintptr_t)control->shipped_cell;
    const uint64_t previous = control->round_seq;
    SparkWeightdMeshRoute routed;
    uint64_t shipped = 0u;
    routed.word = route;
    routed.fields.half = half;
    route = routed.word;
    if ( half != SPARK_WEIGHTD_MESH_ROUTE_HALF_WHOLE )
        entry += SPARK_WEIGHTD_MESH_DOORBELL_CELL_WORDS(sequence);
    if ( threadIdx.x == 0u && (mode & SPARK_TP_MESH_PUBLISH_SHIP) != 0u && shipped_cell != 0 && previous != 0u )
        shipped = SparkTpLdcvU64(shipped_cell);
    if ( failed == 0u && bytes != 0u )
        SparkTpMeshCopyBytesVolatile(payload,source,bytes,threadIdx.x,blockDim.x);
    if ( threadIdx.x == 0u )
    {
        ready = (mode & SPARK_TP_MESH_PUBLISH_SHIP) == 0u || (shipped_cell != 0 && SparkTpMeshShippedAccepts(control,shipped,prior) != 0u) ? 1u : 0u;
        if ( failed == 0u )
        {
            *tail = tag;
            if ( ready != 0u ) { entry[1] = doorbell_bytes; entry[2] = slot; entry[3] = route; }
        }
    }
    __threadfence_system();
    __syncthreads();
    if ( threadIdx.x != 0u ) return;
    if ( ready == 0u && !(shipped_cell != 0 && SparkTpMeshShippedAccepts(control,SparkTpLdAcquireU64(shipped_cell),prior) != 0u) &&
         !(prior != 0u && SparkTpMeshShipPoll(control) != 0u) )
    {
        SparkTpMeshGateRequest(gate,control,SPARK_WEIGHTD_MESH_WAIT_SHIPPED,0u,0u,timeout_ns);
        SparkTpMeshGateAwait(gate,control);
        if ( (mode & SPARK_TP_MESH_PUBLISH_GUARD) != 0u ) SparkTpMeshGateGuard(gate,control);
    }
    if ( failed != 0u && control->error_word == 0u ) control->error_word = UINT64_MAX;
    if ( control->error_word == 0u )
    {
        if ( ready == 0u ) { entry[1] = doorbell_bytes; entry[2] = slot; entry[3] = route; __threadfence_system(); }
        control->round_seq_prior = control->round_seq;
        control->round_seq = tag;
        control->seq += advance;
        entry[0] = tag;
        control->copy_ns += SparkTpGlobalTimerNs() - started;
    }
    if ( (mode & SPARK_TP_MESH_PUBLISH_REQUEST) != 0u )
        SparkTpMeshGateRequest(gate,control,SPARK_WEIGHTD_MESH_WAIT_PEERS,peer_mask,0u,timeout_ns);
    if ( (mode & SPARK_TP_MESH_PUBLISH_ARRIVAL) != 0u )
        control->wait_started_ns = SparkTpGlobalTimerNs();
    __threadfence();
    LmDependentRelease();
}

static __global__ void SparkTpMeshHardwarePublishGateKernel(
    volatile SparkWeightdMeshWaitRequest *gate,SparkTpMeshRoundControl *control,uint32_t mode,uint64_t timeout_ns)
{
    if ( threadIdx.x != 0u || blockIdx.x != 0u ) return;
    SparkTpMeshShipGate(gate,control,mode,timeout_ns);
    LmDependentRelease();
}

static __global__ void SparkTpMeshHardwarePublishWideKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    volatile uint64_t *entry,SparkTpMeshRoundControl *control,uint32_t rank,
    const void *source,uint64_t bytes,uint64_t offset,uint64_t doorbell_bytes,uint64_t route,uint32_t phase_offset,
    uint32_t advance,volatile SparkWeightdMeshWaitRequest *gate,uint32_t mode,uint64_t peer_mask,uint64_t timeout_ns)
{
    __shared__ uint32_t last;
    uint64_t sequence,tag,slot;
    volatile uint8_t *destination,*payload;
    uint32_t failed,half;
    SparkWeightdMeshRoute routed;
    if ( threadIdx.x == 0u )
        (void)atomicCAS((unsigned long long *)&control->publish_started_ns,0ull,SparkTpGlobalTimerNs());
    failed = control->error_word != 0u || control->seq > UINT32_MAX - phase_offset ? 1u : 0u;
    sequence = control->seq + phase_offset;
    tag = (control->epoch << 32u) | sequence;
    slot = (uint64_t)rank * slots_per_rank + ((sequence - 1u) & (slots_per_rank - 1u));
    half = (mode & SPARK_TP_MESH_PUBLISH_HALF) != 0u ? SPARK_WEIGHTD_MESH_SEQUENCE_HALF(sequence) : SPARK_WEIGHTD_MESH_ROUTE_HALF_WHOLE;
    destination = band + slot * slot_bytes;
    payload = destination + SPARK_WEIGHTD_MESH_HALF_OFFSET(half) + offset;
    if ( failed == 0u && bytes != 0u )
        SparkTpMeshCopyBytesVolatile(payload,source,bytes,(uint64_t)blockIdx.x * blockDim.x + threadIdx.x,(uint64_t)blockDim.x * gridDim.x);
    __threadfence_system();
    __syncthreads();
    if ( threadIdx.x == 0u )
        last = atomicAdd((unsigned long long *)&control->publish_blocks_done,1ull) + 1ull == gridDim.x ? 1u : 0u;
    __syncthreads();
    if ( last == 0u || threadIdx.x != 0u ) return;
    control->publish_blocks_done = 0u;
    if ( control->error_word == 0u && control->seq > UINT32_MAX - phase_offset ) control->error_word = UINT64_MAX;
    if ( failed == 0u )
    {
        routed.word = route;
        routed.fields.half = half;
        if ( half != SPARK_WEIGHTD_MESH_ROUTE_HALF_WHOLE )
            entry += SPARK_WEIGHTD_MESH_DOORBELL_CELL_WORDS(sequence);
        entry[1] = doorbell_bytes;
        entry[2] = slot;
        entry[3] = routed.word;
        control->round_seq_prior = control->round_seq;
        control->round_seq = tag;
        control->seq += advance;
        __threadfence_system();
        *(volatile uint64_t *)(destination + (half != SPARK_WEIGHTD_MESH_ROUTE_HALF_WHOLE ? SPARK_WEIGHTD_MESH_TAIL_OFFSET(half) : slot_bytes - 8u)) = tag;
        __threadfence_system();
        entry[0] = tag;
        control->copy_ns += SparkTpGlobalTimerNs() - control->publish_started_ns;
    }
    control->publish_started_ns = 0u;
    if ( (mode & SPARK_TP_MESH_PUBLISH_REQUEST) != 0u )
        SparkTpMeshGateRequest(gate,control,SPARK_WEIGHTD_MESH_WAIT_PEERS,peer_mask,0u,timeout_ns);
    if ( (mode & SPARK_TP_MESH_PUBLISH_ARRIVAL) != 0u )
        control->wait_started_ns = SparkTpGlobalTimerNs();
    LmDependentRelease();
}

static SparkGpuError SparkTpMeshHardwarePublish(SparkGpuStream stream,
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    volatile uint64_t *entry,SparkTpMeshRoundControl *control,uint32_t rank,
    const void *source,uint64_t bytes,uint64_t offset,uint64_t doorbell_bytes,uint64_t route,
    SparkWeightdMeshWaitRequest *gate,uint32_t mode,uint64_t peer_mask,uint64_t timeout_ns)
{
    uint64_t blocks;
    if ( bytes < SPARK_TP_MESH_WIDE_PUBLISH_BYTES )
    {
        LM_LAUNCH((SparkTpMeshHardwarePublishKernel),1,SPARK_TP_MESH_THREADS,0u,stream,band,slot_bytes,slots_per_rank,entry,control,rank,source,bytes,offset,doorbell_bytes,route,1u,1u,
            gate,mode,peer_mask,timeout_ns);
        return SparkGpuPeekAtLastError();
    }
    if ( (mode & (SPARK_TP_MESH_PUBLISH_SHIP | SPARK_TP_MESH_PUBLISH_GUARD)) != 0u )
    {
        LM_LAUNCH((SparkTpMeshHardwarePublishGateKernel),1,1,0u,stream,gate,control,mode,timeout_ns);
        SparkGpuError status = SparkGpuPeekAtLastError();
        if ( status != SPARK_GPU_SUCCESS )
            return status;
    }
    blocks = (bytes + SPARK_TP_MESH_WIDE_PUBLISH_UNIT - 1u) / SPARK_TP_MESH_WIDE_PUBLISH_UNIT;
    blocks = blocks < SPARK_TP_MESH_COPY_BLOCKS ? blocks : SPARK_TP_MESH_COPY_BLOCKS;
    LM_LAUNCH((SparkTpMeshHardwarePublishWideKernel),(uint32_t)blocks,SPARK_TP_MESH_THREADS,0u,stream,band,slot_bytes,slots_per_rank,entry,control,rank,source,bytes,offset,doorbell_bytes,route,1u,1u,
        gate,mode & (SPARK_TP_MESH_PUBLISH_REQUEST | SPARK_TP_MESH_PUBLISH_ARRIVAL | SPARK_TP_MESH_PUBLISH_HALF),peer_mask,timeout_ns);
    return SparkGpuPeekAtLastError();
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
        ((uint16_t *)output)[begin + i] = SparkTpFloatToBf16(sum);
}

static __device__ __forceinline__ void SparkTpAccumU128Bf16(float sum[8],uint4 v)
{
    sum[0] += __uint_as_float(v.x << 16u); sum[1] += __uint_as_float(v.x & 0xffff0000u);
    sum[2] += __uint_as_float(v.y << 16u); sum[3] += __uint_as_float(v.y & 0xffff0000u);
    sum[4] += __uint_as_float(v.z << 16u); sum[5] += __uint_as_float(v.z & 0xffff0000u);
    sum[6] += __uint_as_float(v.w << 16u); sum[7] += __uint_as_float(v.w & 0xffff0000u);
}

static __device__ __forceinline__ uint4 SparkTpPackU128Bf16(const float sum[8])
{
    return make_uint4(LmPackBf16Pair(sum[0],sum[1]),LmPackBf16Pair(sum[2],sum[3]),LmPackBf16Pair(sum[4],sum[5]),LmPackBf16Pair(sum[6],sum[7]));
}

static __device__ __forceinline__ void SparkTpAccumU256Bf16(float sum[16],SparkTpU256 v)
{
    SparkTpAccumU128Bf16(sum,v.lo);
    SparkTpAccumU128Bf16(sum + 8u,v.hi);
}

static __device__ __forceinline__ SparkTpU256 SparkTpPackU256Bf16(const float sum[16])
{
    SparkTpU256 out;
    out.lo = SparkTpPackU128Bf16(sum);
    out.hi = SparkTpPackU128Bf16(sum + 8u);
    return out;
}

static __device__ __forceinline__ SparkTpU256 SparkTpMeshHardwareSumOctValue(
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,uint64_t ring,uint32_t degree,uint64_t source_index)
{
    float sum[16] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
    const uint64_t stride = slots_per_rank * slot_bytes;
    const uint8_t *base = band + ring * slot_bytes + source_index * sizeof(uint16_t);
    for ( uint32_t peer = 0u; peer < degree; peer += 4u )
    {
        SparkTpU256 v[4];
        const uint32_t n = degree - peer < 4u ? degree - peer : 4u;
#pragma unroll
        for ( uint32_t j = 0u; j < 4u; j++ ) if ( j < n ) v[j] = SparkTpLdCvU256(base + (uint64_t)(peer + j) * stride);
#pragma unroll
        for ( uint32_t j = 0u; j < 4u; j++ ) if ( j < n ) SparkTpAccumU256Bf16(sum,v[j]);
    }
    return SparkTpPackU256Bf16(sum);
}

static __device__ __forceinline__ uint4 SparkTpMeshHardwareSumQuadValue(
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,uint64_t ring,uint32_t degree,uint64_t source_index)
{
    float sum[8] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
    const uint64_t stride = slots_per_rank * slot_bytes;
    const uint8_t *base = band + ring * slot_bytes + source_index * sizeof(uint16_t);
    for ( uint32_t peer = 0u; peer < degree; peer += 8u )
    {
        uint4 v[8];
        const uint32_t n = degree - peer < 8u ? degree - peer : 8u;
#pragma unroll
        for ( uint32_t j = 0u; j < 8u; j++ ) if ( j < n ) v[j] = SparkTpLdCvU128(base + (uint64_t)(peer + j) * stride);
#pragma unroll
        for ( uint32_t j = 0u; j < 8u; j++ ) if ( j < n ) SparkTpAccumU128Bf16(sum,v[j]);
    }
    return SparkTpPackU128Bf16(sum);
}

static __device__ void SparkTpMeshHardwareCombineDone(SparkTpMeshRoundControl *control,uint32_t last)
{
    __syncthreads();
    if ( threadIdx.x == 0u )
    {
        (void)atomicMax((unsigned long long *)&control->math_finished_ns,SparkTpGlobalTimerNs());
        __threadfence();
        if ( atomicAdd((unsigned long long *)&control->math_blocks_done,1ull) + 1ull == (unsigned long long)gridDim.x * gridDim.y )
        {
            control->combine_ns += SparkTpLdcvU64(&control->math_finished_ns) - SparkTpLdcvU64(&control->math_started_ns);
            control->math_started_ns = 0u; control->math_finished_ns = 0u; control->math_blocks_done = 0u;
            control->slot_cursor = control->seq;
            if ( last != 0u ) control->rounds_done++;
        }
    }
}

static __device__ __forceinline__ void SparkTpMeshHardwareCombineSettle(SparkTpMeshRoundControl *control,uint32_t last,uint64_t started)
{
    if ( threadIdx.x == 0u ) (void)atomicCAS((unsigned long long *)&control->math_started_ns,0ull,started);
    SparkTpMeshHardwareCombineDone(control,last);
    LmDependentRelease();
}

static __device__ __forceinline__ void SparkTpMeshHardwareDirectBody(
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,uint64_t ring,
    uint32_t rank,uint32_t degree,void *output,uint64_t local_elements,uint64_t begin,uint64_t count,
    uint32_t operation,uint64_t first,uint64_t span,uint64_t packed_first,uint32_t skip_peer,volatile uint16_t *next_dst)
{
    const uint64_t lane = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x,lanes = (uint64_t)blockDim.x * gridDim.x;
    if ( operation == SPARK_TP_MESH_OPERATION_PAIR_GATHER )
    {
        const uint64_t peer = blockIdx.y == 0u ? rank : rank ^ 1u;
        SparkTpMeshCopyHalvesFromBand((uint16_t *)output + peer * local_elements + begin,
            (const volatile uint16_t *)(band + (peer * slots_per_rank + ring) * slot_bytes),count,lane,lanes);
        return;
    }
    if ( operation == SPARK_TP_MESH_OPERATION_ALL_GATHER || operation == SPARK_TP_MESH_OPERATION_SLICE_GATHER )
    {
        if ( gridDim.x == 1u && gridDim.y < degree )
        {
            const uint32_t nwarps = blockDim.x >> 5u,warp = threadIdx.x >> 5u,wlane = threadIdx.x & 31u;
            const uint32_t pcount = (degree - blockIdx.y + gridDim.y - 1u) / gridDim.y;
            const uint32_t wpp = nwarps >= pcount ? nwarps / pcount : 1u,active = wpp * pcount;
            for ( uint32_t lp = warp / wpp; lp < pcount && warp < active; lp += nwarps / wpp )
            {
                const uint64_t peer = blockIdx.y + (uint64_t)lp * gridDim.y;
                const uint64_t start = operation == SPARK_TP_MESH_OPERATION_ALL_GATHER ? 0u : (peer * local_elements < count ? peer * local_elements : count);
                const uint64_t length = operation == SPARK_TP_MESH_OPERATION_ALL_GATHER ? count : (count - start < local_elements ? count - start : local_elements);
                if ( peer != skip_peer && length != 0u )
                    SparkTpMeshCopyHalvesFromBand(
                        (uint16_t *)output + (operation == SPARK_TP_MESH_OPERATION_ALL_GATHER ? peer * local_elements : start) + begin,
                        (const volatile uint16_t *)(band + (peer * slots_per_rank + ring) * slot_bytes) +
                            (operation == SPARK_TP_MESH_OPERATION_SLICE_GATHER && packed_first == UINT64_MAX ? start : 0u),
                        length,(uint64_t)(warp % wpp) * 32u + wlane,(uint64_t)wpp * 32u);
            }
            return;
        }
        for ( uint64_t peer = blockIdx.y; peer < degree; peer += gridDim.y )
        {
            const uint64_t start = operation == SPARK_TP_MESH_OPERATION_ALL_GATHER ? 0u : (peer * local_elements < count ? peer * local_elements : count);
            const uint64_t length = operation == SPARK_TP_MESH_OPERATION_ALL_GATHER ? count : (count - start < local_elements ? count - start : local_elements);
            if ( peer != skip_peer && length != 0u )
                SparkTpMeshCopyHalvesFromBand(
                    (uint16_t *)output + (operation == SPARK_TP_MESH_OPERATION_ALL_GATHER ? peer * local_elements : start) + begin,
                    (const volatile uint16_t *)(band + (peer * slots_per_rank + ring) * slot_bytes) +
                        (operation == SPARK_TP_MESH_OPERATION_SLICE_GATHER && packed_first == UINT64_MAX ? start : 0u),length,lane,lanes);
        }
        return;
    }
    const uint64_t base = packed_first != UINT64_MAX ? first - packed_first : first;
    uint16_t *out16 = (uint16_t *)output + begin + first;
    uint64_t octs = operation == 1u && ((uintptr_t)band & 31u) == 0u && (slot_bytes & 31u) == 0u &&
        ((uintptr_t)out16 & 31u) == 0u && (base & 15u) == 0u && (next_dst == 0 || ((uintptr_t)next_dst & 31u) == 0u) ? span / 16u : 0u;
    uint64_t done = octs * 16u,i;
    for ( i = lane; i < octs; i += lanes )
    {
        SparkTpU256 v = SparkTpMeshHardwareSumOctValue(band,slot_bytes,slots_per_rank,ring,degree,base + i * 16u);
        SparkTpStU256(out16 + i * 16u,v);
        if ( next_dst != 0 ) SparkTpStVolatileU256(next_dst + i * 16u,v);
    }
    uint64_t quads = operation == 1u && ((uintptr_t)band & 15u) == 0u && (slot_bytes & 15u) == 0u &&
        ((uintptr_t)(out16 + done) & 15u) == 0u && ((base + done) & 7u) == 0u && (next_dst == 0 || ((uintptr_t)(next_dst + done) & 15u) == 0u) ? (span - done) / 8u : 0u;
    for ( i = lane; i < quads; i += lanes )
    {
        uint4 v = SparkTpMeshHardwareSumQuadValue(band,slot_bytes,slots_per_rank,ring,degree,base + done + i * 8u);
        ((uint4 *)(out16 + done))[i] = v;
        if ( next_dst != 0 ) SparkTpStVolatileU128(next_dst + done + i * 8u,v);
    }
    done += quads * 8u;
    for ( i = first + done + lane; i < first + span; i += lanes )
    {
        SparkTpMeshHardwareDirectElement(band,slot_bytes,slots_per_rank,ring,degree,output,local_elements,begin,count,operation,i,packed_first);
        if ( next_dst != 0 ) next_dst[i - first] = out16[i - first];
    }
}

static __device__ __forceinline__ void SparkTpMeshHardwareDirectRun(
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    SparkTpMeshRoundControl *control,uint32_t degree,void *output,
    uint64_t local_elements,uint64_t begin,uint64_t count,uint32_t operation,uint64_t first,uint64_t span,uint32_t last,uint64_t packed_first,uint32_t skip_peer,
    uint32_t rank)
{
    LmDependentWait();
    if ( control->error_word != 0u ) return;
    const uint64_t started = SparkTpGlobalTimerNs(),ring = (control->round_seq - 1u) & (slots_per_rank - 1u);
    SparkTpMeshHardwareDirectBody(band,slot_bytes,slots_per_rank,ring,rank,degree,output,local_elements,begin,count,operation,first,span,packed_first,skip_peer,0);
    SparkTpMeshHardwareCombineSettle(control,last,started);
}

static __global__ void SparkTpMeshHardwareFusedDirectKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,uint64_t timeout_ns,
    LmWeightPrefetchGroup *prefetch,uint32_t resume,void *output,
    uint64_t local_elements,uint64_t begin,uint64_t count,uint32_t operation,uint64_t first,uint64_t span,
    uint32_t last,uint64_t packed_first,uint32_t skip_peer,
    volatile uint64_t *next_entry,volatile SparkWeightdMeshWaitRequest *next_gate,
    uint64_t next_offset,uint64_t next_doorbell_bytes,uint64_t next_route)
{
    __shared__ uint32_t is_last;
    __shared__ uint32_t is_leader;
    __shared__ uint32_t s_go;
    uint64_t arr_base = 0u,arr_end = 0u,arr_landed = UINT64_MAX;
    const uint32_t tid = threadIdx.x;
    const uint64_t total_blocks = (uint64_t)gridDim.x * gridDim.y;
    uint64_t started = 0u;
    LmDependentWait();
    if ( total_blocks == 1u )
    {
        if ( SparkTpMeshArrivalWait(band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns,prefetch,resume,0u,0u,arr_base,arr_end,arr_landed) == 0u ) return;
        if ( tid == 0u ) { is_leader = 1u; started = SparkTpGlobalTimerNs(); }
    }
    else
    {
        if ( tid == 0u )
            is_leader = SparkTpLdcvU64(&control->error_word) != 0u ? 2u :
                atomicCAS((unsigned long long *)&control->math_finished_ns,0ull,1ull) == 0ull ? 1u : 0u;
        __syncthreads();
        if ( is_leader == 2u ) return;
        if ( is_leader != 0u )
        {
            if ( SparkTpMeshArrivalWait(band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns,prefetch,resume,0u,0u,arr_base,arr_end,arr_landed) == 0u )
            {
                if ( tid == 0u ) { control->math_finished_ns = 0u; __threadfence(); }
                return;
            }
            if ( tid == 0u ) { started = SparkTpGlobalTimerNs() | 1ull; control->math_started_ns = started; __threadfence(); }
        }
        else
        {
            if ( tid == 0u )
            {
                uint32_t go = 0u,spins = 0u;
                for (;;)
                {
                    if ( SparkTpLdAcquireU64(&control->math_started_ns) != 0u ) { go = 1u; break; }
                    if ( SparkTpLdAcquireU64(&control->error_word) != 0u ) { go = 0u; break; }
                    if ( ++spins > 256u ) __nanosleep(200u);
                }
                s_go = go;
            }
            __syncthreads();
            if ( s_go == 0u ) return;
        }
    }
    const uint64_t seq = control->seq,ring = (control->round_seq - 1u) & (slots_per_rank - 1u);
    const uint32_t next_overflow = next_entry != 0 && seq > UINT32_MAX - 1u ? 1u : 0u;
    const uint64_t next_seq = seq + 1u,next_slot = (uint64_t)rank * slots_per_rank + ((next_seq - 1u) & (slots_per_rank - 1u));
    volatile uint16_t *next_dst = next_entry != 0 && next_overflow == 0u ? (volatile uint16_t *)(band + next_slot * slot_bytes + next_offset) : 0;
    SparkTpMeshHardwareDirectBody(band,slot_bytes,slots_per_rank,ring,rank,degree,output,local_elements,begin,count,operation,first,span,packed_first,skip_peer,next_dst);
    if ( next_dst != 0 ) __threadfence_system();
    __syncthreads();
    if ( total_blocks == 1u )
    {
        if ( tid == 0u ) is_last = 1u;
    }
    else if ( tid == 0u )
        is_last = atomicAdd((unsigned long long *)&control->math_blocks_done,1ull) + 1ull == total_blocks ? 1u : 0u;
    __syncthreads();
    if ( is_last != 0u && tid == 0u )
    {
        const uint64_t begin_ns = total_blocks == 1u ? started : SparkTpLdcvU64(&control->math_started_ns);
        if ( total_blocks > 1u ) { control->math_blocks_done = 0u; control->math_started_ns = 0u; control->math_finished_ns = 0u; }
        control->combine_ns += SparkTpGlobalTimerNs() - begin_ns;
        if ( next_entry != 0 )
        {
            if ( next_overflow != 0u )
            {
                if ( control->error_word == 0u ) control->error_word = UINT64_MAX;
            }
            else
            {
                const uint64_t next_tag = (control->epoch << 32u) | next_seq;
                *(volatile uint64_t *)(band + next_slot * slot_bytes + slot_bytes - 8u) = next_tag;
                SparkTpMeshShipGate(next_gate,control,SPARK_TP_MESH_PUBLISH_SHIP | SPARK_TP_MESH_PUBLISH_GUARD,timeout_ns);
                if ( control->error_word == 0u )
                {
                    next_entry[1] = next_doorbell_bytes; next_entry[2] = next_slot; next_entry[3] = next_route;
                    __threadfence_system();
                    control->round_seq_prior = control->round_seq; control->round_seq = next_tag;
                    control->seq = next_seq; control->slot_cursor = next_seq;
                    next_entry[0] = next_tag;
                    control->wait_started_ns = SparkTpGlobalTimerNs();
                }
            }
        }
        else
        {
            control->slot_cursor = seq;
            if ( last != 0u ) control->rounds_done++;
        }
    }
    LmDependentRelease();
    if ( is_leader != 0u && tid < 32u )
        SparkTpMeshArrivalRecord(control,tid,arr_base,arr_end,arr_landed);
}

static __global__ void SparkTpMeshHardwareDirectKernel(
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    SparkTpMeshRoundControl *control,uint32_t degree,void *output,
    uint64_t local_elements,uint64_t begin,uint64_t count,uint32_t operation,uint64_t first,uint64_t span,uint32_t last,uint64_t packed_first,uint32_t skip_peer,uint32_t rank)
{
    SparkTpMeshHardwareDirectRun(band,slot_bytes,slots_per_rank,control,degree,output,local_elements,begin,count,operation,first,span,last,packed_first,skip_peer,rank);
}

static __global__ __launch_bounds__(SPARK_TP_MESH_DIRECT_THREADS,4) void SparkTpMeshHardwareDirectWideKernel(
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    SparkTpMeshRoundControl *control,uint32_t degree,void *output,
    uint64_t local_elements,uint64_t begin,uint64_t count,uint32_t operation,uint64_t first,uint64_t span,uint32_t last,uint64_t packed_first,uint32_t skip_peer,uint32_t rank)
{
    SparkTpMeshHardwareDirectRun(band,slot_bytes,slots_per_rank,control,degree,output,local_elements,begin,count,operation,first,span,last,packed_first,skip_peer,rank);
}

static __global__ void SparkTpMeshPairFirstStageKernel(
    const SparkTpMeshRoundControl *control,uint16_t *staging,uint64_t staging_slot_elements,const uint16_t *local,uint32_t rank,
    uint64_t segment,uint64_t count)
{
    const uint64_t target = 2u * (uint64_t)blockIdx.y + ((rank & 1u) ^ 1u);
    if ( control->error_word != 0u ) return;
    SparkTpMeshCopyHalves(staging + target * staging_slot_elements + SPARK_WEIGHTD_MESH_STAGING_HALF_OFFSET(control->seq) / sizeof(uint16_t),
        local + target * segment,count,(uint64_t)blockIdx.x * blockDim.x + threadIdx.x,(uint64_t)blockDim.x * gridDim.x);
    __threadfence_system();
}

static __global__ void SparkTpMeshPairFirstPairKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,const SparkTpMeshRoundControl *control,
    uint16_t *staging,uint64_t staging_slot_elements,const uint16_t *local,uint32_t rank,uint64_t segment,uint64_t count,uint32_t view)
{
    const uint64_t tag = control->round_seq - ((view & SPARK_TP_MESH_VIEW_LAG) != 0u ? 1u : 0u),next = tag + 2u,target = 2u * (uint64_t)blockIdx.y + (rank & 1u);
    const volatile uint16_t *received = (const volatile uint16_t *)(band + ((target ^ 1u) * slots_per_rank + ((tag - 1u) & (slots_per_rank - 1u))) * slot_bytes +
        SPARK_WEIGHTD_MESH_HALF_OFFSET(SPARK_WEIGHTD_MESH_SEQUENCE_HALF(tag)));
    float *destination = target == rank ?
        (float *)(band + ((uint64_t)rank * slots_per_rank + ((next - 1u) & (slots_per_rank - 1u))) * slot_bytes + SPARK_WEIGHTD_MESH_HALF_OFFSET(SPARK_WEIGHTD_MESH_SEQUENCE_HALF(next))) :
        (float *)((uint8_t *)(staging + target * staging_slot_elements) + SPARK_WEIGHTD_MESH_STAGING_HALF_OFFSET(next - 1u));
    const uint16_t *own = local + target * segment;
    uint64_t i,done = 0u,lane = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x,lanes = (uint64_t)blockDim.x * gridDim.x;
    if ( control->error_word != 0u ) return;
    if ( SparkTpMeshAligned16(own,received) != 0u && (((uintptr_t)destination & 31u) == 0u) )
    {
        uint64_t quads = count / 8u;
        for ( i = lane; i < quads; i += lanes )
        {
            float s[8] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
            SparkTpAccumU128Bf16(s,((const uint4 *)own)[i]);
            SparkTpAccumU128Bf16(s,SparkTpLdCvU128(received + i * 8u));
            SparkTpU256 out = {make_uint4(__float_as_uint(s[0]),__float_as_uint(s[1]),__float_as_uint(s[2]),__float_as_uint(s[3])),
                make_uint4(__float_as_uint(s[4]),__float_as_uint(s[5]),__float_as_uint(s[6]),__float_as_uint(s[7]))};
            SparkTpStU256(destination + i * 8u,out);
        }
        done = quads * 8u;
    }
    for ( i = done + lane; i < count; i += lanes ) destination[i] = __uint_as_float((uint32_t)own[i] << 16u) + __uint_as_float((uint32_t)received[i] << 16u);
    __threadfence_system();
}

static __global__ __launch_bounds__(SPARK_TP_MESH_DIRECT_THREADS,4) void SparkTpMeshPairFirstGroupKernel(
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,SparkTpMeshRoundControl *control,uint32_t degree,uint16_t *output,
    uint64_t count,uint32_t last,uint32_t rank,uint32_t view)
{
    const uint64_t tag = control->round_seq - ((view & SPARK_TP_MESH_VIEW_LAG) != 0u ? 1u : 0u),ring = (tag - 1u) & (slots_per_rank - 1u);
    const uint64_t half = SPARK_WEIGHTD_MESH_HALF_OFFSET(SPARK_WEIGHTD_MESH_SEQUENCE_HALF(tag)),started = SparkTpGlobalTimerNs(),stride = slots_per_rank * slot_bytes;
    uint64_t i,done = 0u,lane = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x,lanes = (uint64_t)blockDim.x * gridDim.x;
    LmDependentWait();
    if ( control->error_word != 0u ) return;
    if ( ((uintptr_t)output & 15u) == 0u && ((uintptr_t)(band + half) & 31u) == 0u && (slot_bytes & 31u) == 0u )
    {
        const uint64_t quads = count / 8u;
        const uint8_t *base = band + ring * slot_bytes + half;
        for ( i = lane; i < quads; i += lanes )
        {
            float sum[8] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
            for ( uint32_t peer = rank & 1u; peer < degree; peer += 4u )
            {
                SparkTpU256 v0 = SparkTpLdCvU256(base + (uint64_t)peer * stride + i * 32u),v1;
                if ( peer + 2u < degree ) v1 = SparkTpLdCvU256(base + (uint64_t)(peer + 2u) * stride + i * 32u);
                sum[0] += __uint_as_float(v0.lo.x); sum[1] += __uint_as_float(v0.lo.y); sum[2] += __uint_as_float(v0.lo.z); sum[3] += __uint_as_float(v0.lo.w);
                sum[4] += __uint_as_float(v0.hi.x); sum[5] += __uint_as_float(v0.hi.y); sum[6] += __uint_as_float(v0.hi.z); sum[7] += __uint_as_float(v0.hi.w);
                if ( peer + 2u < degree ) { sum[0] += __uint_as_float(v1.lo.x); sum[1] += __uint_as_float(v1.lo.y); sum[2] += __uint_as_float(v1.lo.z); sum[3] += __uint_as_float(v1.lo.w); sum[4] += __uint_as_float(v1.hi.x); sum[5] += __uint_as_float(v1.hi.y); sum[6] += __uint_as_float(v1.hi.z); sum[7] += __uint_as_float(v1.hi.w); }
            }
            ((uint4 *)output)[i] = SparkTpPackU128Bf16(sum);
        }
        done = quads * 8u;
    }
    for ( i = done + lane; i < count; i += lanes )
    {
        float sum = 0.0f;
        for ( uint32_t peer = rank & 1u; peer < degree; peer += 2u ) sum += ((const volatile float *)(band + ((uint64_t)peer * slots_per_rank + ring) * slot_bytes + half))[i];
        output[i] = SparkTpFloatToBf16(sum);
    }
    SparkTpMeshHardwareCombineSettle(control,last,started);
}

static __global__ void SparkTpMeshPairFirstGatherStageKernel(
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,const SparkTpMeshRoundControl *control,
    uint16_t *staging,uint64_t staging_slot_elements,const uint16_t *group_source,const uint16_t *pair_source,uint32_t rank,
    uint64_t group,uint64_t pair,uint32_t view)
{
    const uint64_t target = blockIdx.y,sender = target ^ 1u;
    const uint64_t tag = control->round_seq - ((view & SPARK_TP_MESH_VIEW_LAG) != 0u ? 1u : 0u);
    uint16_t *destination = staging + target * staging_slot_elements + SPARK_WEIGHTD_MESH_STAGING_HALF_OFFSET(control->seq) / sizeof(uint16_t);
    const uint64_t lane = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x,lanes = (uint64_t)blockDim.x * gridDim.x;
    if ( control->error_word != 0u || target == rank ) return;
    if ( ((target ^ rank) & 1u) == 0u )
        SparkTpMeshCopyHalves(destination,group_source,group,lane,lanes);
    else if ( sender == rank )
        SparkTpMeshCopyHalves(destination,pair_source,pair,lane,lanes);
    else
        SparkTpMeshCopyHalvesFromBand(destination,(const volatile uint16_t *)(band + (sender * slots_per_rank + ((tag - 1u) & (slots_per_rank - 1u))) * slot_bytes +
            SPARK_WEIGHTD_MESH_HALF_OFFSET(SPARK_WEIGHTD_MESH_SEQUENCE_HALF(tag))),pair,lane,lanes);
    __threadfence_system();
}

static __global__ __launch_bounds__(SPARK_TP_MESH_DIRECT_THREADS,4) void SparkTpMeshPairFirstGatherKernel(
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,SparkTpMeshRoundControl *control,uint16_t *output,
    const uint16_t *local,uint64_t local_elements,uint64_t round,uint64_t chunks,uint32_t last,uint32_t rank,uint32_t view)
{
    const uint64_t width = SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES / sizeof(uint16_t);
    const uint64_t tag = control->round_seq - ((view & SPARK_TP_MESH_VIEW_LAG) != 0u ? 1u : 0u);
    const uint64_t peer = blockIdx.y,same = ((peer ^ rank) & 1u) == 0u ? 1u : 0u;
    const uint64_t started = SparkTpGlobalTimerNs();
    const uint64_t lane = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x,lanes = (uint64_t)blockDim.x * gridDim.x;
    LmDependentWait();
    if ( control->error_word != 0u ) return;
    if ( same != 0u ? round < chunks : round >= 2u && round < chunks + 2u )
    {
        const uint64_t begin = (same != 0u ? round : round - 2u) * width;
        const uint64_t count = local_elements - begin < width ? local_elements - begin : width;
        if ( peer == rank )
            SparkTpMeshCopyHalves(output + peer * local_elements + begin,local + begin,count,lane,lanes);
        else
            SparkTpMeshCopyHalvesFromBand(output + peer * local_elements + begin,
                (const volatile uint16_t *)(band + (peer * slots_per_rank + ((tag - 1u) & (slots_per_rank - 1u))) * slot_bytes +
                    SPARK_WEIGHTD_MESH_HALF_OFFSET(SPARK_WEIGHTD_MESH_SEQUENCE_HALF(tag))),count,lane,lanes);
    }
    SparkTpMeshHardwareCombineSettle(control,last,started);
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
    uint64_t peer = blockIdx.y;
    uint16_t *destination;
    if ( control->error_word != 0u ) return;
    destination = peer == rank ?
        (uint16_t *)(band + ((uint64_t)rank * slots_per_rank + (control->seq & (slots_per_rank - 1u))) * slot_bytes) :
        staging + peer * staging_slot_elements;
    SparkTpMeshCopyHalves(destination,local + peer * per_peer_elements + begin,count,
        (uint64_t)blockIdx.x * blockDim.x + threadIdx.x,(uint64_t)blockDim.x * gridDim.x);
    __threadfence_system();
}

static __global__ void SparkTpMeshRsagStageKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,SparkTpMeshRoundControl *control,
    uint16_t *staging,uint64_t staging_slot_elements,const uint16_t *local,uint32_t rank,
    uint64_t count,uint64_t slice,uint64_t doorbell_bytes,uint64_t route)
{
    __shared__ uint32_t last;
    LmDependentWait();
    if ( threadIdx.x == 0u )
        (void)atomicCAS((unsigned long long *)&control->publish_started_ns,0ull,SparkTpGlobalTimerNs());
    const uint32_t failed = control->error_word != 0u || control->seq > UINT32_MAX - 1u ? 1u : 0u;
    const uint64_t sequence = control->seq + 1u,slot = (uint64_t)rank * slots_per_rank + ((sequence - 1u) & (slots_per_rank - 1u));
    if ( failed == 0u )
    {
        const uint64_t peer = blockIdx.y;
        const uint64_t first = peer * slice < count ? peer * slice : count;
        const uint64_t length = count - first < slice ? count - first : slice;
        uint16_t *destination = peer == rank ? (uint16_t *)(band + slot * slot_bytes) : staging + peer * staging_slot_elements;
        SparkTpMeshCopyHalves(destination,local + first,length,(uint64_t)blockIdx.x * blockDim.x + threadIdx.x,(uint64_t)blockDim.x * gridDim.x);
    }
    __threadfence_system();
    __syncthreads();
    if ( threadIdx.x == 0u )
        last = atomicAdd((unsigned long long *)&control->publish_blocks_done,1ull) + 1ull == (unsigned long long)gridDim.x * gridDim.y ? 1u : 0u;
    __syncthreads();
    if ( last == 0u || threadIdx.x != 0u ) return;
    control->publish_blocks_done = 0u;
    if ( control->error_word == 0u && control->seq > UINT32_MAX - 1u ) control->error_word = UINT64_MAX;
    if ( failed == 0u )
    {
        const uint64_t tag = (control->epoch << 32u) | sequence;
        entry[1] = doorbell_bytes; entry[2] = slot; entry[3] = route;
        control->round_seq_prior = control->round_seq; control->round_seq = tag; control->seq = sequence;
        __threadfence_system();
        *(volatile uint64_t *)(band + slot * slot_bytes + slot_bytes - 8u) = tag;
        __threadfence_system();
        entry[0] = tag;
        control->copy_ns += SparkTpGlobalTimerNs() - control->publish_started_ns;
    }
    control->publish_started_ns = 0u;
    control->wait_started_ns = SparkTpGlobalTimerNs();
    LmDependentRelease();
}

static __global__ void SparkTpMeshRsagStagePublishKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,
    volatile SparkWeightdMeshWaitRequest *gate,SparkTpMeshRoundControl *control,
    uint16_t *staging,uint64_t staging_slot_elements,const uint16_t *local,uint32_t rank,
    uint64_t count,uint64_t slice,uint64_t doorbell_bytes,uint64_t route,uint64_t timeout_ns)
{
    const uint64_t started = threadIdx.x == 0u ? SparkTpGlobalTimerNs() : 0u;
    if ( threadIdx.x == 0u )
        SparkTpMeshShipGate(gate,control,SPARK_TP_MESH_PUBLISH_SHIP | SPARK_TP_MESH_PUBLISH_GUARD,timeout_ns);
    __syncthreads();
    const uint32_t failed = control->error_word != 0u || control->seq > UINT32_MAX - 1u ? 1u : 0u;
    const uint64_t sequence = control->seq + 1u,slot = (uint64_t)rank * slots_per_rank + ((sequence - 1u) & (slots_per_rank - 1u));
    if ( failed == 0u )
    {
        const uint64_t peer = threadIdx.x >> 5u;
        const uint64_t first = peer * slice < count ? peer * slice : count;
        const uint64_t length = count - first < slice ? count - first : slice;
        uint16_t *destination = peer == rank ? (uint16_t *)(band + slot * slot_bytes) : staging + peer * staging_slot_elements;
        SparkTpMeshCopyHalves(destination,local + first,length,threadIdx.x & 31u,32u);
    }
    __threadfence_system();
    __syncthreads();
    if ( threadIdx.x == 0u )
    {
        if ( control->error_word == 0u && control->seq > UINT32_MAX - 1u ) control->error_word = UINT64_MAX;
        if ( failed == 0u )
        {
            const uint64_t tag = (control->epoch << 32u) | sequence;
            entry[1] = doorbell_bytes; entry[2] = slot; entry[3] = route;
            control->round_seq_prior = control->round_seq; control->round_seq = tag; control->seq = sequence;
            *(volatile uint64_t *)(band + slot * slot_bytes + slot_bytes - 8u) = tag;
            __threadfence_system();
            entry[0] = tag;
            control->copy_ns += SparkTpGlobalTimerNs() - started;
        }
        control->wait_started_ns = SparkTpGlobalTimerNs();
        __threadfence();
        LmDependentRelease();
    }
}

static __global__ void SparkTpMeshScatterStageKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,SparkTpMeshRoundControl *control,
    uint16_t *staging,uint64_t staging_slot_elements,const uint16_t *local,uint32_t rank,
    uint64_t segment,uint64_t count,uint64_t doorbell_bytes,uint64_t route)
{
    __shared__ uint32_t last;
    LmDependentWait();
    if ( threadIdx.x == 0u )
        (void)atomicCAS((unsigned long long *)&control->publish_started_ns,0ull,SparkTpGlobalTimerNs());
    const uint32_t failed = control->error_word != 0u || control->seq > UINT32_MAX - 1u ? 1u : 0u;
    const uint64_t sequence = control->seq + 1u,slot = (uint64_t)rank * slots_per_rank + ((sequence - 1u) & (slots_per_rank - 1u));
    if ( failed == 0u )
    {
        const uint64_t peer = blockIdx.y;
        uint16_t *destination = peer == rank ? (uint16_t *)(band + slot * slot_bytes) : staging + peer * staging_slot_elements;
        SparkTpMeshCopyHalves(destination,local + peer * segment,count,
            (uint64_t)blockIdx.x * blockDim.x + threadIdx.x,(uint64_t)blockDim.x * gridDim.x);
    }
    __threadfence_system();
    __syncthreads();
    if ( threadIdx.x == 0u )
        last = atomicAdd((unsigned long long *)&control->publish_blocks_done,1ull) + 1ull == (unsigned long long)gridDim.x * gridDim.y ? 1u : 0u;
    __syncthreads();
    if ( last == 0u || threadIdx.x != 0u ) return;
    control->publish_blocks_done = 0u;
    if ( control->error_word == 0u && control->seq > UINT32_MAX - 1u ) control->error_word = UINT64_MAX;
    if ( failed == 0u )
    {
        const uint64_t tag = (control->epoch << 32u) | sequence;
        entry[1] = doorbell_bytes; entry[2] = slot; entry[3] = route;
        control->round_seq_prior = control->round_seq; control->round_seq = tag; control->seq = sequence;
        __threadfence_system();
        *(volatile uint64_t *)(band + slot * slot_bytes + slot_bytes - 8u) = tag;
        __threadfence_system();
        entry[0] = tag;
        control->copy_ns += SparkTpGlobalTimerNs() - control->publish_started_ns;
    }
    control->publish_started_ns = 0u;
    control->wait_started_ns = SparkTpGlobalTimerNs();
    LmDependentRelease();
}

#define SPARK_TP_MESH_PIECE_PUBLISH_GATHER 1u
#define SPARK_TP_MESH_PIECE_PUBLISH_SCATTER 2u
#define SPARK_TP_MESH_PIECE_COMBINE_GATHER 1u
#define SPARK_TP_MESH_PIECE_COMBINE_SUM 2u

typedef struct SparkTpMeshPiece
{
    uint8_t *band;
    volatile uint64_t *entry;
    volatile SparkWeightdMeshWaitRequest *gate;
    SparkTpMeshRoundControl *control;
    const uint16_t *source;
    uint16_t *staging,*output;
    uint64_t slot_bytes,slots_per_rank,source_stride,source_count,route,doorbell_bytes,output_stride,begin,count,timeout_ns;
    uint32_t rank,degree,publish,combine,lag,last,ship_prior,ring_direct;
} SparkTpMeshPiece;

static __global__ __launch_bounds__(SPARK_TP_MESH_DIRECT_THREADS,SPARK_TP_MESH_PIECE_RESIDENCY) void SparkTpMeshHardwarePieceStageKernel(SparkTpMeshPiece piece)
{
    __shared__ uint32_t s_last;
    SparkTpMeshRoundControl *control = piece.control;
    const uint32_t tid = threadIdx.x;
    LmDependentWait();
    if ( SparkTpLdcvU64(&control->error_word) != 0u ) return;
    if ( tid == 0u )
        (void)atomicCAS((unsigned long long *)&control->publish_started_ns,0ull,SparkTpGlobalTimerNs() | 1ull);
    const uint32_t failed = control->seq > UINT32_MAX - 1u ? 1u : 0u;
    const uint64_t mask = piece.slots_per_rank - 1u,sequence = control->seq + 1u;
    const uint64_t slot = (uint64_t)piece.rank * piece.slots_per_rank + ((sequence - 1u) & mask);
    const uint64_t lane = (uint64_t)blockIdx.x * blockDim.x + tid,lanes = (uint64_t)blockDim.x * gridDim.x;
    if ( failed == 0u )
    {
        uint16_t *own = (uint16_t *)(piece.band + slot * piece.slot_bytes + SPARK_WEIGHTD_MESH_HALF_OFFSET(SPARK_WEIGHTD_MESH_SEQUENCE_HALF(sequence)));
        if ( piece.publish == SPARK_TP_MESH_PIECE_PUBLISH_GATHER )
            SparkTpMeshCopyHalves(own,piece.source,piece.source_count,lane,lanes);
        else
            SparkTpMeshCopyHalves(blockIdx.y == piece.rank ? own :
                piece.staging + (uint64_t)blockIdx.y * (SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES / sizeof(uint16_t)) +
                    SPARK_WEIGHTD_MESH_STAGING_HALF_OFFSET(sequence - 1u) / sizeof(uint16_t),
                piece.source + (uint64_t)blockIdx.y * piece.source_stride,piece.source_count,lane,lanes);
        __threadfence_system();
    }
    __syncthreads();
    if ( tid == 0u )
        s_last = atomicAdd((unsigned long long *)&control->publish_blocks_done,1ull) + 1ull == (unsigned long long)gridDim.x * gridDim.y ? 1u : 0u;
    __syncthreads();
    if ( s_last == 0u || tid != 0u ) return;
    control->publish_blocks_done = 0u;
    if ( failed != 0u && control->error_word == 0u ) control->error_word = UINT64_MAX;
    if ( failed == 0u && control->error_word == 0u )
    {
        const uint64_t pub_start = SparkTpLdcvU64(&control->publish_started_ns);
        const uint64_t pub_end = SparkTpGlobalTimerNs();
        if ( pub_end > pub_start ) control->copy_ns += pub_end - pub_start;
        if ( piece.ring_direct != 0u )
        {
            const uint64_t tag = (control->epoch << 32u) | sequence;
            const uint32_t half = SPARK_WEIGHTD_MESH_SEQUENCE_HALF(sequence);
            volatile uint64_t *cell = piece.entry + SPARK_WEIGHTD_MESH_DOORBELL_CELL_WORDS(sequence);
            SparkWeightdMeshRoute routed;
            routed.word = piece.route;
            routed.fields.half = half;
            cell[1] = piece.doorbell_bytes;
            cell[2] = slot;
            cell[3] = routed.word;
            control->round_seq_prior = control->round_seq;
            control->round_seq = tag;
            control->seq = sequence;
            *(volatile uint64_t *)(piece.band + slot * piece.slot_bytes + SPARK_WEIGHTD_MESH_TAIL_OFFSET(half)) = tag;
            __threadfence_system();
            cell[0] = tag;
            control->wait_started_ns = SparkTpGlobalTimerNs();
        }
    }
    control->publish_started_ns = 0u;
}

static __global__ void SparkTpMeshHardwarePieceArriveRingKernel(
    uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    volatile uint64_t *entry,SparkTpMeshRoundControl *control,
    uint32_t rank,uint32_t degree,uint64_t timeout_ns,
    uint32_t arrive_lag,uint32_t ring_publish,uint64_t doorbell_bytes,uint64_t route)
{
    __shared__ volatile uint64_t s_start;
    const uint32_t lane = threadIdx.x;
    uint64_t tag,started,landed = UINT64_MAX,diag = 0u,other,fault;
    uint32_t step;
    if ( blockIdx.x != 0u || SparkTpLdcvU64(&control->error_word) != 0u ) return;
    tag = control->round_seq - arrive_lag;
    if ( lane == 0u ) s_start = SparkTpGlobalTimerNs();
    __syncwarp();
    started = s_start;
    fault = SparkTpMeshArrivalPoll((const volatile uint64_t *)(band +
            ((uint64_t)lane * slots_per_rank + ((tag - 1u) & (slots_per_rank - 1u))) * slot_bytes +
            SPARK_WEIGHTD_MESH_TAIL_OFFSET(SPARK_WEIGHTD_MESH_SEQUENCE_HALF(tag))),
        (const volatile uint64_t *)(uintptr_t)control->cancel_cell,control->cancel_expected,tag,started,timeout_ns,
        lane,lane < degree && lane != rank ? 1u : 0u,landed,diag);
    started = s_start;
    if ( fault == 0u && lane < SPARK_TP_MESH_ROUND_CONTROL_PEERS && landed != UINT64_MAX && landed > started )
        control->peer_arrival_ns[lane] += landed - started;
    for ( step = 16u; step != 0u; step >>= 1u )
    {
        other = __shfl_xor_sync(0xffffffffu,landed,step);
        landed = other < landed ? other : landed;
    }
    if ( lane != 0u ) return;
    if ( fault != 0u )
    {
        if ( control->error_word == 0u )
        {
            control->error_word = fault;
            control->diag_word = diag;
            __threadfence();
        }
        LmDependentRelease();
        return;
    }
    {
        const uint64_t end_ns = SparkTpGlobalTimerNs();
        if ( end_ns > started ) control->peer_wait_ns += end_ns - started;
    }
    control->first_arrival_ns += landed > started && landed != UINT64_MAX ? landed - started : 0u;
    if ( ring_publish != 0u )
    {
        if ( control->seq > UINT32_MAX - 1u )
            control->error_word = UINT64_MAX;
        else
        {
            const uint64_t sequence = control->seq + 1u;
            const uint64_t next_tag = (control->epoch << 32u) | sequence;
            const uint64_t slot = (uint64_t)rank * slots_per_rank + ((sequence - 1u) & (slots_per_rank - 1u));
            const uint32_t half = SPARK_WEIGHTD_MESH_SEQUENCE_HALF(sequence);
            volatile uint64_t *cell = entry + SPARK_WEIGHTD_MESH_DOORBELL_CELL_WORDS(sequence);
            SparkWeightdMeshRoute routed;
            routed.word = route;
            routed.fields.half = half;
            cell[1] = doorbell_bytes;
            cell[2] = slot;
            cell[3] = routed.word;
            control->round_seq_prior = control->round_seq;
            control->round_seq = next_tag;
            control->seq = sequence;
            *(volatile uint64_t *)(band + slot * slot_bytes + SPARK_WEIGHTD_MESH_TAIL_OFFSET(half)) = next_tag;
            __threadfence_system();
            cell[0] = next_tag;
            control->wait_started_ns = SparkTpGlobalTimerNs();
        }
    }
    control->math_started_ns = 0u;
    control->math_finished_ns = 0u;
    control->math_blocks_done = 0u;
    __threadfence();
    LmDependentRelease();
}

static __global__ __launch_bounds__(SPARK_TP_MESH_DIRECT_THREADS,SPARK_TP_MESH_PIECE_RESIDENCY) void SparkTpMeshHardwarePieceCombineKernel(SparkTpMeshPiece piece)
{
    SparkTpMeshRoundControl *control = piece.control;
    const uint64_t mask = piece.slots_per_rank - 1u;
    const uint64_t lane = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x,lanes = (uint64_t)blockDim.x * gridDim.x;
    uint64_t i;
    LmDependentWait();
    if ( SparkTpLdcvU64(&control->error_word) != 0u ) return;
    const uint64_t started = SparkTpGlobalTimerNs();
    const uint64_t tag = control->round_seq - piece.lag,ring = (tag - 1u) & mask;
    const uint8_t *halves = piece.band + SPARK_WEIGHTD_MESH_HALF_OFFSET(SPARK_WEIGHTD_MESH_SEQUENCE_HALF(tag));
    if ( piece.combine == SPARK_TP_MESH_PIECE_COMBINE_GATHER )
        SparkTpMeshCopyHalvesFromBand(piece.output + (uint64_t)blockIdx.y * piece.output_stride + piece.begin,
            (const volatile uint16_t *)(halves + ((uint64_t)blockIdx.y * piece.slots_per_rank + ring) * piece.slot_bytes),piece.count,lane,lanes);
    else
    {
        uint16_t *out16 = piece.output + piece.begin;
        uint64_t octs = ((uintptr_t)out16 & 31u) == 0u && ((uintptr_t)halves & 31u) == 0u && (piece.slot_bytes & 31u) == 0u ? piece.count / 16u : 0u;
        uint64_t done = octs * 16u;
        for ( i = lane; i < octs; i += lanes )
            SparkTpStU256(out16 + i * 16u,SparkTpMeshHardwareSumOctValue(halves,piece.slot_bytes,piece.slots_per_rank,ring,piece.degree,i * 16u));
        uint64_t quads = ((uintptr_t)(out16 + done) & 15u) == 0u && ((uintptr_t)halves & 15u) == 0u && (piece.slot_bytes & 15u) == 0u ? (piece.count - done) / 8u : 0u;
        for ( i = lane; i < quads; i += lanes )
            ((uint4 *)(out16 + done))[i] = SparkTpMeshHardwareSumQuadValue(halves,piece.slot_bytes,piece.slots_per_rank,ring,piece.degree,done + i * 8u);
        done += quads * 8u;
        for ( i = done + lane; i < piece.count; i += lanes )
            SparkTpMeshHardwareDirectElement(halves,piece.slot_bytes,piece.slots_per_rank,ring,piece.degree,piece.output,0u,piece.begin,piece.count,1u,i,0u);
    }
    SparkTpMeshHardwareCombineSettle(control,piece.last,started);
}

static __global__ void SparkTpMeshHardwarePieceStreamGatherKernel(SparkTpMeshPiece piece)
{
    __shared__ uint32_t s_go;
    __shared__ volatile uint64_t s_start;
    SparkTpMeshRoundControl *control = piece.control;
    const uint32_t peer = blockIdx.y,stripe = blockIdx.x,tid = threadIdx.x;
    const uint64_t lane = (uint64_t)stripe * blockDim.x + tid,lanes = (uint64_t)blockDim.x * gridDim.x;
    if ( SparkTpLdcvU64(&control->error_word) != 0u ) return;
    const uint64_t tag = control->round_seq - piece.lag;
    const uint64_t ring = (tag - 1u) & (piece.slots_per_rank - 1u);
    const uint32_t half = SPARK_WEIGHTD_MESH_SEQUENCE_HALF(tag);
    const uint8_t *halves = piece.band + SPARK_WEIGHTD_MESH_HALF_OFFSET(half);
    const uint64_t tail_off = SPARK_WEIGHTD_MESH_TAIL_OFFSET(half);
    if ( peer == piece.rank )
    {
        SparkTpMeshCopyHalvesFromBand(piece.output + (uint64_t)peer * piece.output_stride + piece.begin,
            (const volatile uint16_t *)(halves + ((uint64_t)peer * piece.slots_per_rank + ring) * piece.slot_bytes),
            piece.count,lane,lanes);
        if ( stripe == 0u )
        {
            __syncthreads();
            if ( tid < 32u )
            {
                if ( tid == 0u ) s_start = SparkTpGlobalTimerNs();
                __syncwarp();
                const uint64_t started = s_start;
                uint64_t landed = UINT64_MAX,diag = 0u,other,fault;
                fault = SparkTpMeshArrivalPoll((const volatile uint64_t *)(piece.band +
                        ((uint64_t)tid * piece.slots_per_rank + ring) * piece.slot_bytes + tail_off),
                    (const volatile uint64_t *)(uintptr_t)control->cancel_cell,control->cancel_expected,tag,started,piece.timeout_ns,
                    tid,tid < piece.degree && tid != piece.rank ? 1u : 0u,landed,diag);
                const uint64_t base = s_start;
                if ( fault == 0u && tid < SPARK_TP_MESH_ROUND_CONTROL_PEERS && landed != UINT64_MAX && landed > base )
                    control->peer_arrival_ns[tid] += landed - base;
                for ( uint32_t step = 16u; step != 0u; step >>= 1u )
                {
                    other = __shfl_xor_sync(0xffffffffu,landed,step);
                    landed = other < landed ? other : landed;
                }
                if ( tid == 0u )
                {
                    const uint64_t arr_end = SparkTpGlobalTimerNs();
                    if ( fault != 0u )
                    {
                        if ( control->error_word == 0u ) { control->error_word = fault; control->diag_word = diag; __threadfence(); }
                    }
                    else
                    {
                        if ( arr_end > base ) control->peer_wait_ns += arr_end - base;
                        control->first_arrival_ns += landed > base && landed != UINT64_MAX ? landed - base : 0u;
                        control->math_started_ns = arr_end;
                        __threadfence();
                    }
                }
            }
        }
    }
    else
    {
        if ( tid == 0u )
        {
            const volatile uint64_t *tail = (const volatile uint64_t *)(piece.band +
                ((uint64_t)peer * piece.slots_per_rank + ring) * piece.slot_bytes + tail_off);
            const volatile uint64_t *cancel = (const volatile uint64_t *)(uintptr_t)control->cancel_cell;
            const uint64_t expected = control->cancel_expected;
            uint32_t go = 0u;
            s_start = SparkTpGlobalTimerNs();
            for (;;)
            {
                const uint64_t base = s_start,now = SparkTpGlobalTimerNs();
                if ( SparkTpLdAcquireU64(tail) == tag ) { go = 1u; break; }
                if ( SparkTpLdcvU64(&control->error_word) != 0u ) break;
                if ( cancel != 0 && SparkTpLdAcquireU64(cancel) != expected ) break;
                if ( now > base && now - base > piece.timeout_ns ) break;
            }
            s_go = go;
        }
        __syncthreads();
        if ( s_go != 0u )
            SparkTpMeshCopyHalvesFromBand(piece.output + (uint64_t)peer * piece.output_stride + piece.begin,
                (const volatile uint16_t *)(halves + ((uint64_t)peer * piece.slots_per_rank + ring) * piece.slot_bytes),
                piece.count,lane,lanes);
    }
    __syncthreads();
    if ( tid == 0u &&
         atomicAdd((unsigned long long *)&control->math_blocks_done,1ull) + 1ull == (unsigned long long)gridDim.x * gridDim.y )
    {
        const uint64_t end_ns = SparkTpGlobalTimerNs();
        const uint64_t start_ns = SparkTpLdcvU64(&control->math_started_ns);
        control->math_blocks_done = 0u;
        control->math_started_ns = 0u;
        control->math_finished_ns = 0u;
        if ( SparkTpLdcvU64(&control->error_word) == 0u )
        {
            if ( start_ns != 0u && end_ns > start_ns ) control->combine_ns += end_ns - start_ns;
            control->slot_cursor = control->seq;
            if ( piece.last != 0u ) control->rounds_done++;
        }
    }
}

static __global__ void SparkTpMeshHardwareAllToAllKernel(
    const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    SparkTpMeshRoundControl *control,uint32_t degree,uint32_t rank,void *output,
    uint64_t per_peer_elements,uint64_t begin,uint64_t count,uint64_t slice_elements,uint32_t last)
{
    uint64_t ring,peer = blockIdx.y;
    if ( control->error_word != 0u ) return;
    if ( threadIdx.x == 0u )
        (void)atomicCAS((unsigned long long *)&control->math_started_ns,0ull,SparkTpGlobalTimerNs());
    ring = (control->round_seq - 1u) & (slots_per_rank - 1u);
    SparkTpMeshCopyHalvesFromBand((uint16_t *)output + peer * per_peer_elements + begin,
        (const volatile uint16_t *)(band + (peer * slots_per_rank + ring) * slot_bytes) + (uint64_t)rank * slice_elements,
        count,(uint64_t)blockIdx.x * blockDim.x + threadIdx.x,(uint64_t)blockDim.x * gridDim.x);
    SparkTpMeshHardwareCombineDone(control,last);
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

static SparkGpuError SparkTpMeshDriverStatus(CUresult status,const char *phase)
{
    if ( status == CUDA_SUCCESS ) return SPARK_GPU_SUCCESS;
    const char *name = 0;
    (void)cuGetErrorName(status,&name);
    fprintf(stderr,"MESH-HARDWARE-FAIL phase=%s driver=%d (%s)\n",phase,(int)status,name != 0 ? name : "unknown");
    return status == CUDA_ERROR_NOT_SUPPORTED ? SPARK_GPU_ERROR_NOT_SUPPORTED : SPARK_GPU_ERROR_UNKNOWN;
}

extern "C" SparkGpuError SparkTpMeshHardwarePrepare(void *host,void **device_out)
{
    if ( host == 0 || device_out == 0 ) return SPARK_GPU_ERROR_INVALID_VALUE;
    *device_out = 0;
    SparkGpuError status;
    const void *kernels[] = {
        (const void *)SparkTpMeshHardwareWaitKernel,
        (const void *)SparkTpMeshHardwareGuardKernel,
        (const void *)SparkTpMeshHardwarePublishKernel,
        (const void *)SparkTpMeshHardwarePublishGateKernel,
        (const void *)SparkTpMeshHardwarePublishWideKernel,
        (const void *)SparkTpMeshHardwareDirectKernel,
        (const void *)SparkTpMeshHardwareDirectWideKernel,
        (const void *)SparkTpMeshHardwareFusedDirectKernel,
        (const void *)SparkTpMeshHardwarePieceStageKernel,
        (const void *)SparkTpMeshHardwarePieceArriveRingKernel,
        (const void *)SparkTpMeshHardwarePieceCombineKernel,
        (const void *)SparkTpMeshHardwarePieceStreamGatherKernel,
        (const void *)SparkTpMeshHardwareArrivalKernel,
        (const void *)SparkTpMeshHardwareSeedKernel,
        (const void *)SparkTpMeshHardwareFoldKernel,
        (const void *)SparkTpMeshHardwareFinishKernel,
        (const void *)SparkTpMeshAllToAllPackKernel,
        (const void *)SparkTpMeshAllToAllStageKernel,
        (const void *)SparkTpMeshRsagStageKernel,
        (const void *)SparkTpMeshRsagStagePublishKernel,
        (const void *)SparkTpMeshScatterStageKernel,
        (const void *)SparkTpMeshPairFirstStageKernel,
        (const void *)SparkTpMeshPairFirstPairKernel,
        (const void *)SparkTpMeshPairFirstGroupKernel,
        (const void *)SparkTpMeshPairFirstGatherStageKernel,
        (const void *)SparkTpMeshPairFirstGatherKernel,
        (const void *)SparkTpMeshHardwareAllToAllKernel
    };
    SparkGpuFunctionAttributes attributes;
    for ( uint32_t i = 0u; i < sizeof(kernels) / sizeof(kernels[0]); i++ )
    {
        status = SparkGpuFuncGetAttributes(&attributes,kernels[i]);
        if ( status != SPARK_GPU_SUCCESS )
        {
            fprintf(stderr,"MESH-HARDWARE-FAIL phase=preload kernel=%u cuda=%d (%s)\n",
                i,(int)status,SparkGpuGetErrorString(status));
            return status;
        }
    }
    status = SparkGpuHostGetDevicePointer(device_out,host,0u);
    if ( status == SPARK_GPU_SUCCESS )
        return status;
    {
        int device = 0,pageable = 0,host_tables = 0;
        if ( SparkGpuGetDevice(&device) != SPARK_GPU_SUCCESS ||
             SparkGpuDeviceGetAttribute(&pageable,SPARK_GPU_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,device) != SPARK_GPU_SUCCESS ||
             SparkGpuDeviceGetAttribute(&host_tables,SPARK_GPU_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,device) != SPARK_GPU_SUCCESS ||
             pageable != 1 || host_tables != 1 )
        {
            fprintf(stderr,"MESH-DEVICE-ALIAS-FAIL ptr=%p cuda=%d (%s) pageable_access=%d host_page_tables=%d\n",
                host,(int)status,SparkGpuGetErrorString(status),pageable,host_tables);
            return status;
        }
    }
    (void)SparkGpuGetLastError();
    fprintf(stderr,"MESH-DEVICE-ALIAS-IDENTITY ptr=%p cuda=%d (%s) coherent-host-path pageable_access=1 host_page_tables=1\n",
        host,(int)status,SparkGpuGetErrorString(status));
    *device_out = host;
    return SPARK_GPU_SUCCESS;
}

static SparkGpuError SparkTpMeshHardwarePeerGuard(SparkGpuStream stream,SparkWeightdMeshWaitRequest *gate,SparkTpMeshRoundControl *control)
{
    LM_LAUNCH((SparkTpMeshHardwareGuardKernel),1,1,0u,stream,gate,control);
    return SparkGpuPeekAtLastError();
}

static SparkGpuError SparkTpMeshHardwareArrivalView(SparkGpuStream stream,const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,uint64_t timeout_ns,uint32_t view)
{
    uint32_t resume;
    LmWeightPrefetchGroup *prefetch = LmWeightPrefetchSlot(stream,&resume);
    LM_LAUNCH((SparkTpMeshHardwareArrivalKernel),1,prefetch != 0 ? 32u + SPARK_TP_MESH_PREFETCH_THREADS : 32u,0u,stream,band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns,prefetch,resume,view);
    return SparkGpuPeekAtLastError();
}

static SparkGpuError SparkTpMeshHardwareArrival(SparkGpuStream stream,const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,uint64_t timeout_ns)
{
    return SparkTpMeshHardwareArrivalView(stream,band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns,0u);
}

static SparkGpuError SparkTpMeshHardwareWait(SparkGpuStream stream,
    SparkWeightdMeshWaitRequest *gate,SparkTpMeshRoundControl *control,
    uint64_t kind,uint64_t peer_mask,uint64_t phase_offset,uint64_t timeout_ns)
{
    LM_LAUNCH((SparkTpMeshHardwareWaitKernel),1,1,0u,stream,gate,control,kind,peer_mask,phase_offset,timeout_ns);
    return SparkGpuPeekAtLastError();
}

static SparkGpuError SparkTpMeshHardwareExchange(SparkGpuStream stream,uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,const uint8_t *source,uint64_t bytes,uint64_t offset,uint64_t doorbell_bytes,SparkWeightdMeshRoute route,uint64_t timeout_ns)
{
    SparkGpuError status = SparkTpMeshHardwarePublish(stream,band,slot_bytes,slots_per_rank,entry,control,rank,source,bytes,offset,doorbell_bytes,route.word,
        request,SPARK_TP_MESH_PUBLISH_SHIP | SPARK_TP_MESH_PUBLISH_GUARD | SPARK_TP_MESH_PUBLISH_REQUEST,route.fields.peer_mask,timeout_ns);
    return status == SPARK_GPU_SUCCESS ? SparkTpMeshHardwarePeerGuard(stream,request,control) : status;
}

static SparkGpuError SparkTpMeshHardwareCombinePacked(SparkGpuStream stream,const uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,void *output,uint64_t local_elements,uint64_t begin,uint64_t count,uint32_t operation,uint64_t first,uint64_t span,uint32_t last,uint64_t packed_first,uint32_t skip_peer)
{
    dim3 grid;
    uint64_t work;
    if ( (operation == SPARK_TP_MESH_OPERATION_ALL_GATHER ? count * degree : count) <= SPARK_TP_MESH_SMALL_MESSAGE_ELEMENTS )
    {
        grid = operation == SPARK_TP_MESH_OPERATION_ALL_GATHER ? dim3(SparkTpMeshCopyBlocks(count,degree),degree) :
            operation == SPARK_TP_MESH_OPERATION_SLICE_GATHER ? dim3(SparkTpMeshCopyBlocks(local_elements,degree),degree) :
            dim3(SparkTpMeshCopyBlocks(span,1u),1u);
        return LM_LAUNCH_DEPENDENT((SparkTpMeshHardwareDirectKernel),grid,SPARK_TP_MESH_THREADS,0u,stream,band,slot_bytes,slots_per_rank,control,degree,output,local_elements,begin,count,operation,first,span,last,packed_first,skip_peer,rank);
    }
    work = operation == SPARK_TP_MESH_OPERATION_ALL_GATHER ? count * degree :
        operation == SPARK_TP_MESH_OPERATION_SLICE_GATHER ? local_elements * degree : span;
    grid = operation == SPARK_TP_MESH_OPERATION_ALL_GATHER ? dim3(SparkTpMeshDirectBlocks(count,degree),degree) :
        operation == SPARK_TP_MESH_OPERATION_SLICE_GATHER ? dim3(SparkTpMeshDirectBlocks(local_elements,degree),degree) :
        dim3(SparkTpMeshDirectBlocks(span,1u),1u);
    if ( work <= SPARK_TP_MESH_DEPENDENT_ELEMENTS )
        return LM_LAUNCH_DEPENDENT((SparkTpMeshHardwareDirectWideKernel),grid,SPARK_TP_MESH_DIRECT_THREADS,0u,stream,band,slot_bytes,slots_per_rank,control,degree,output,local_elements,begin,count,operation,first,span,last,packed_first,skip_peer,rank);
    LM_LAUNCH((SparkTpMeshHardwareDirectWideKernel),grid,SPARK_TP_MESH_DIRECT_THREADS,0u,stream,band,slot_bytes,slots_per_rank,control,degree,output,local_elements,begin,count,operation,first,span,last,packed_first,skip_peer,rank);
    return SparkGpuPeekAtLastError();
}

static SparkGpuError SparkTpMeshHardwareCompleteCombine(SparkGpuStream stream,uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,
    SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,uint64_t timeout_ns,void *output,
    uint64_t local_elements,uint64_t begin,uint64_t count,uint32_t operation,uint64_t first,uint64_t span,
    uint32_t last,uint64_t packed_first,uint32_t skip_peer,
    volatile uint64_t *next_entry,SparkWeightdMeshWaitRequest *next_gate,
    uint64_t next_offset,uint64_t next_doorbell_bytes,uint64_t next_route,uint64_t next_peer_mask)
{
    SparkGpuError status;
    if ( SparkTpMeshCombineFused(operation,count,span,degree) != 0u )
    {
        uint32_t resume;
        LmWeightPrefetchGroup *prefetch = LmWeightPrefetchSlot(stream,&resume);
        return LM_LAUNCH_DEPENDENT((SparkTpMeshHardwareFusedDirectKernel),SparkTpMeshFusedGrid(operation,span,degree),SPARK_TP_MESH_THREADS,0u,stream,band,slot_bytes,slots_per_rank,
            control,rank,degree,timeout_ns,prefetch,resume,output,local_elements,begin,count,operation,first,span,last,packed_first,skip_peer,
            next_entry,next_gate,next_offset,next_doorbell_bytes,next_route);
    }
    status = SparkTpMeshHardwareArrival(stream,band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns);
    if ( status == SPARK_GPU_SUCCESS )
        status = SparkTpMeshHardwareCombinePacked(stream,band,slot_bytes,slots_per_rank,control,rank,degree,output,local_elements,begin,count,operation,first,span,last,packed_first,skip_peer);
    if ( status == SPARK_GPU_SUCCESS && next_entry != 0 )
        status = SparkTpMeshHardwarePublish(stream,band,slot_bytes,slots_per_rank,next_entry,control,rank,(const uint8_t *)output + (begin + first) * sizeof(uint16_t),
            span * sizeof(uint16_t),next_offset,next_doorbell_bytes,next_route,next_gate,
            SPARK_TP_MESH_PUBLISH_SHIP | SPARK_TP_MESH_PUBLISH_GUARD | SPARK_TP_MESH_PUBLISH_ARRIVAL,next_peer_mask,timeout_ns);
    return status;
}

static SparkGpuError SparkTpMeshHardwareDirectChunk(SparkGpuStream stream,uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,const uint8_t *local,void *output,uint64_t local_elements,uint64_t begin,uint64_t count,uint32_t operation,uint64_t timeout_ns,uint32_t stage)
{
    SparkWeightdMeshRoute route = {0};
    uint32_t width = operation == 2u ? 8u : 2u;
    uint64_t span = operation == 0u ? count * degree : count;
    SparkGpuError status = SPARK_GPU_SUCCESS;
    route.fields.peer_mask = ((1u << degree) - 1u) & ~(1u << rank);
    if ( stage != SPARK_TP_MESH_STAGE_COMPLETE )
        status = SparkTpMeshHardwarePublish(stream,band,slot_bytes,slots_per_rank,entry,control,rank,local + begin * width,count * width,0u,count * width,route.word,
            request,SPARK_TP_MESH_PUBLISH_SHIP | SPARK_TP_MESH_PUBLISH_GUARD | SPARK_TP_MESH_PUBLISH_ARRIVAL,route.fields.peer_mask,timeout_ns);
    if ( status != SPARK_GPU_SUCCESS || stage == SPARK_TP_MESH_STAGE_PUBLISH ) return status;
    return SparkTpMeshHardwareCompleteCombine(stream,band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns,output,
        local_elements,begin,count,operation,0u,span,begin + count == local_elements,UINT64_MAX,UINT32_MAX,0,0,0u,0u,0u,0u);
}

static SparkGpuError SparkTpMeshHardwarePairChunk(SparkGpuStream stream,uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,const uint8_t *local,void *output,uint64_t local_elements,uint64_t begin,uint64_t count,uint64_t timeout_ns)
{
    SparkWeightdMeshRoute route = {0};
    SparkGpuError status;
    route.fields.peer_mask = ((1u << degree) - 1u) & ~(1u << rank);
    route.fields.pair = 1u;
    status = SparkTpMeshHardwarePublish(stream,band,slot_bytes,slots_per_rank,entry,control,rank,local + begin * sizeof(uint16_t),count * sizeof(uint16_t),0u,
        count * sizeof(uint16_t),route.word,request,SPARK_TP_MESH_PUBLISH_SHIP | SPARK_TP_MESH_PUBLISH_GUARD | SPARK_TP_MESH_PUBLISH_ARRIVAL,route.fields.peer_mask,timeout_ns);
    if ( status == SPARK_GPU_SUCCESS )
        status = SparkTpMeshHardwareArrival(stream,band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns);
    if ( status != SPARK_GPU_SUCCESS ) return status;
    LM_LAUNCH((SparkTpMeshHardwareDirectWideKernel),dim3(SparkTpMeshDirectBlocks(count,2u),2u),SPARK_TP_MESH_DIRECT_THREADS,0u,stream,band,slot_bytes,slots_per_rank,control,
        degree,output,local_elements,begin,count,SPARK_TP_MESH_OPERATION_PAIR_GATHER,0u,count,begin + count == local_elements,UINT64_MAX,UINT32_MAX,rank);
    return SparkGpuPeekAtLastError();
}

static SparkGpuError SparkTpMeshHardwareRsagChunk(SparkGpuStream stream,uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,const uint8_t *local,void *output,uint64_t local_elements,uint64_t begin,uint64_t count,uint64_t timeout_ns,uint32_t stage)
{
    SparkWeightdMeshRoute route = {0};
    uint64_t slice = SparkTpMeshRsagSlice(count,degree);
    uint64_t first = (uint64_t)rank * slice < count ? (uint64_t)rank * slice : count;
    uint64_t span = count - first < slice ? count - first : slice;
    SparkGpuError status = SPARK_GPU_SUCCESS;
    route.fields.peer_mask = ((1u << degree) - 1u) & ~(1u << rank);
    route.fields.slice_bytes = slice * sizeof(uint16_t);
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_SCATTER;
    if ( stage != SPARK_TP_MESH_STAGE_COMPLETE )
        status = SparkTpMeshHardwarePublish(stream,band,slot_bytes,slots_per_rank,entry,control,rank,local + begin * sizeof(uint16_t),count * sizeof(uint16_t),0u,count * sizeof(uint16_t),route.word,
            request,SPARK_TP_MESH_PUBLISH_SHIP | SPARK_TP_MESH_PUBLISH_GUARD | SPARK_TP_MESH_PUBLISH_ARRIVAL,route.fields.peer_mask,timeout_ns);
    if ( status != SPARK_GPU_SUCCESS || stage == SPARK_TP_MESH_STAGE_PUBLISH ) return status;
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_GATHER;
    status = SparkTpMeshHardwareCompleteCombine(stream,band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns,output,local_elements,begin,count,1u,first,span,0u,UINT64_MAX,UINT32_MAX,
        entry,request,first * sizeof(uint16_t),count * sizeof(uint16_t),route.word,route.fields.peer_mask);
    if ( status != SPARK_GPU_SUCCESS ) return status;
    return SparkTpMeshHardwareCompleteCombine(stream,band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns,output,slice,begin,count,SPARK_TP_MESH_OPERATION_SLICE_GATHER,0u,count,begin + count == local_elements,UINT64_MAX,rank,0,0,0u,0u,0u,0u);
}

static SparkGpuError SparkTpMeshHardwarePeerRsagChunk(SparkGpuStream stream,uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,const uint8_t *local,void *output,uint8_t *staging,uint64_t local_elements,uint64_t begin,uint64_t count,uint64_t timeout_ns,uint32_t stage)
{
    SparkWeightdMeshRoute route = {0};
    uint64_t slice = SparkTpMeshRsagSlice(count,degree),staging_slot_elements = SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES / sizeof(uint16_t);
    uint64_t first,span,gather_bytes;
    SparkGpuError status;
    if ( staging == 0 || slice > staging_slot_elements ) return SPARK_GPU_ERROR_INVALID_VALUE;
    first = (uint64_t)rank * slice < count ? (uint64_t)rank * slice : count;
    span = count - first < slice ? count - first : slice;
    route.fields.peer_mask = ((1u << degree) - 1u) & ~(1u << rank);
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_PEER;
    route.fields.slice_bytes = slice * sizeof(uint16_t);
    if ( stage != SPARK_TP_MESH_STAGE_COMPLETE )
    {
        if ( count <= SPARK_TP_MESH_FUSED_STAGE_ELEMENTS )
        {
            LM_LAUNCH((SparkTpMeshRsagStagePublishKernel),1u,degree * 32u,0u,stream,band,slot_bytes,slots_per_rank,entry,request,control,
                (uint16_t *)staging,staging_slot_elements,(const uint16_t *)local + begin,rank,count,slice,route.fields.slice_bytes,route.word,timeout_ns);
            status = SparkGpuPeekAtLastError();
        }
        else
        {
            status = SparkTpMeshHardwareWait(stream,request,control,SPARK_WEIGHTD_MESH_WAIT_SHIPPED,0u,0u,timeout_ns);
            if ( status == SPARK_GPU_SUCCESS )
            {
                LM_LAUNCH((SparkTpMeshRsagStageKernel),dim3(SparkTpMeshCopyBlocks(slice,degree),degree),SPARK_TP_MESH_THREADS,0u,stream,band,slot_bytes,slots_per_rank,entry,control,
                    (uint16_t *)staging,staging_slot_elements,(const uint16_t *)local + begin,rank,count,slice,route.fields.slice_bytes,route.word);
                status = SparkGpuPeekAtLastError();
            }
        }
        if ( status != SPARK_GPU_SUCCESS || stage == SPARK_TP_MESH_STAGE_PUBLISH ) return status;
    }
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_FULL;
    route.fields.slice_bytes = 0u;
    gather_bytes = (span * sizeof(uint16_t) + 7u) & ~UINT64_C(7);
    status = SparkTpMeshHardwareCompleteCombine(stream,band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns,output,local_elements,begin,count,1u,first,span,0u,first,UINT32_MAX,
        entry,request,0u,gather_bytes != 0u ? gather_bytes : 8u,route.word,route.fields.peer_mask);
    if ( status != SPARK_GPU_SUCCESS ) return status;
    return SparkTpMeshHardwareCompleteCombine(stream,band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns,output,slice,begin,count,SPARK_TP_MESH_OPERATION_SLICE_GATHER,0u,count,begin + count == local_elements,0u,rank,0,0,0u,0u,0u,0u);
}

static SparkGpuError SparkTpMeshHardwarePeerScatterChunk(SparkGpuStream stream,uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,const uint8_t *local,void *output,uint8_t *staging,uint64_t segment,uint64_t begin,uint64_t count,uint64_t timeout_ns)
{
    SparkWeightdMeshRoute route = {0};
    const uint64_t staging_slot_elements = SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES / sizeof(uint16_t);
    const uint64_t slice_bytes = count * sizeof(uint16_t);
    SparkGpuError status;
    if ( staging == 0 || count == 0u || count > staging_slot_elements || (count % 4u) != 0u )
        return SPARK_GPU_ERROR_INVALID_VALUE;
    route.fields.peer_mask = ((1u << degree) - 1u) & ~(1u << rank);
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_PEER;
    route.fields.slice_bytes = slice_bytes;
    LM_LAUNCH((SparkTpMeshHardwarePublishGateKernel),1u,1u,0u,stream,request,control,
        SPARK_TP_MESH_PUBLISH_SHIP | SPARK_TP_MESH_PUBLISH_GUARD,timeout_ns);
    status = SparkGpuPeekAtLastError();
    if ( status != SPARK_GPU_SUCCESS ) return status;
    status = LM_LAUNCH_DEPENDENT((SparkTpMeshScatterStageKernel),dim3(SparkTpMeshCopyBlocks(count,degree),degree),SPARK_TP_MESH_THREADS,0u,stream,
        band,slot_bytes,slots_per_rank,entry,control,(uint16_t *)staging,staging_slot_elements,(const uint16_t *)local + begin,rank,segment,count,slice_bytes,route.word);
    if ( status != SPARK_GPU_SUCCESS ) return status;
    if ( SparkTpMeshCombineFused(1u,count,count,degree) != 0u )
        return SparkTpMeshHardwareCompleteCombine(stream,band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns,output,segment,begin,count,1u,0u,count,begin + count == segment,0u,UINT32_MAX,0,0,0u,0u,0u,0u);
    status = SparkTpMeshHardwareArrival(stream,band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns);
    if ( status != SPARK_GPU_SUCCESS ) return status;
    return LM_LAUNCH_DEPENDENT((SparkTpMeshHardwareDirectWideKernel),SparkTpMeshDirectBlocks(count,1u),SPARK_TP_MESH_DIRECT_THREADS,0u,stream,band,slot_bytes,slots_per_rank,control,degree,output,segment,begin,count,1u,0u,count,begin + count == segment,0u,UINT32_MAX,rank);
}

static SparkGpuError SparkTpMeshHardwarePipelineRound(SparkGpuStream stream,uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,const uint8_t *local,void *output,uint8_t *staging,uint64_t elements,uint32_t publish,uint32_t combine,uint64_t timeout_ns)
{
    const uint64_t width = SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES / sizeof(uint16_t);
    const uint64_t chunks = SparkTpMeshPipelineChunks(elements,SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES);
    SparkWeightdMeshRoute route = {0};
    SparkTpMeshPiece piece = {0};
    uint64_t launch;
    if ( slot_bytes != SPARK_WEIGHTD_MESH_SLOT_BYTES || slots_per_rank != SPARK_WEIGHTD_MESH_SLOTS_PER_RANK || chunks < 3u ||
         (publish == SPARK_TP_MESH_PIECE_PUBLISH_SCATTER && staging == 0) )
        return SPARK_GPU_ERROR_INVALID_VALUE;
    route.fields.peer_mask = ((1u << degree) - 1u) & ~(1u << rank);
    if ( publish == SPARK_TP_MESH_PIECE_PUBLISH_SCATTER )
        route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_PEER;
    piece.band = band;
    piece.entry = entry;
    piece.gate = request;
    piece.control = control;
    piece.staging = (uint16_t *)staging;
    piece.output = (uint16_t *)output;
    piece.slot_bytes = slot_bytes;
    piece.slots_per_rank = slots_per_rank;
    piece.source_stride = elements;
    piece.output_stride = elements;
    piece.timeout_ns = timeout_ns;
    piece.rank = rank;
    piece.degree = degree;
    for ( launch = 0u; launch < chunks + SPARK_WEIGHTD_MESH_PIPELINE_DEPTH; launch++ )
    {
        const uint64_t taken = SparkTpMeshPipelineTaken(launch,SPARK_WEIGHTD_MESH_PIPELINE_DEPTH);
        const uint32_t has_publish = launch < chunks ? 1u : 0u;
        const uint32_t has_combine = taken != SPARK_TP_MESH_PIPELINE_NONE ? 1u : 0u;
        piece.publish = has_publish != 0u ? publish : 0u;
        piece.combine = has_combine != 0u ? combine : 0u;
        if ( has_publish != 0u )
        {
            const uint64_t begin = launch * width,count = elements - begin < width ? elements - begin : width;
            const dim3 stage_grid = publish == SPARK_TP_MESH_PIECE_PUBLISH_GATHER ?
                dim3(SparkTpMeshDirectBlocks(count,1u),1u) :
                dim3(SparkTpMeshPieceBlocks(count,degree),degree);
            const uint32_t ship_mode = SPARK_TP_MESH_PUBLISH_SHIP | SPARK_TP_MESH_PUBLISH_GUARD |
                (launch != 0u ? SPARK_TP_MESH_PUBLISH_SHIP_PRIOR : 0u);
            route.fields.slice_bytes = publish == SPARK_TP_MESH_PIECE_PUBLISH_SCATTER ? (count * sizeof(uint16_t) + 7u) & ~UINT64_C(7) : 0u;
            piece.source = (const uint16_t *)local + begin;
            piece.source_count = count;
            piece.route = route.word;
            piece.doorbell_bytes = publish == SPARK_TP_MESH_PIECE_PUBLISH_SCATTER ? route.fields.slice_bytes : count * sizeof(uint16_t);
            piece.ship_prior = launch != 0u ? 1u : 0u;
            piece.ring_direct = has_combine == 0u ? 1u : 0u;
            LM_LAUNCH((SparkTpMeshHardwarePublishGateKernel),1u,1u,0u,stream,request,control,ship_mode,timeout_ns);
            if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
                return SparkGpuPeekAtLastError();
            SparkGpuError status = LM_LAUNCH_DEPENDENT((SparkTpMeshHardwarePieceStageKernel),stage_grid,SPARK_TP_MESH_DIRECT_THREADS,0u,stream,piece);
            if ( status != SPARK_GPU_SUCCESS )
                return status;
        }
        if ( has_combine != 0u )
        {
            const uint32_t arrive_lag = (uint32_t)((has_publish != 0u ? launch : chunks) - 1u - taken);
            piece.begin = taken * width;
            piece.count = elements - piece.begin < width ? elements - piece.begin : width;
            piece.lag = (uint32_t)((has_publish != 0u ? launch : chunks - 1u) - taken);
            piece.last = taken + 1u == chunks ? 1u : 0u;
            if ( has_publish == 0u && combine == SPARK_TP_MESH_PIECE_COMBINE_GATHER )
            {
                const dim3 gather_grid(SparkTpMeshCopyBlocks(piece.count,degree),degree);
                LM_LAUNCH((SparkTpMeshHardwarePieceStreamGatherKernel),gather_grid,SPARK_TP_MESH_THREADS,0u,stream,piece);
                if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
                    return SparkGpuPeekAtLastError();
            }
            else
            {
                const dim3 combine_grid = combine == SPARK_TP_MESH_PIECE_COMBINE_GATHER ?
                    dim3(SparkTpMeshPieceBlocks(piece.count,degree),degree) :
                    dim3(SparkTpMeshDirectBlocks(piece.count,1u),1u);
                LM_LAUNCH((SparkTpMeshHardwarePieceArriveRingKernel),1u,32u,0u,stream,band,slot_bytes,slots_per_rank,
                    entry,control,rank,degree,timeout_ns,arrive_lag,has_publish,piece.doorbell_bytes,piece.route);
                if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
                    return SparkGpuPeekAtLastError();
                SparkGpuError status = LM_LAUNCH_DEPENDENT((SparkTpMeshHardwarePieceCombineKernel),combine_grid,SPARK_TP_MESH_DIRECT_THREADS,0u,stream,piece);
                if ( status != SPARK_GPU_SUCCESS )
                    return status;
            }
        }
    }
    route.fields.slice_bytes = 0u;
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_FULL;
    LM_LAUNCH((SparkTpMeshHardwarePublishKernel),1,SPARK_TP_MESH_THREADS,0u,stream,band,slot_bytes,slots_per_rank,entry,control,rank,local,
        SPARK_TP_MESH_PIPELINE_MARKER_BYTES,0u,SPARK_TP_MESH_PIPELINE_MARKER_BYTES,route.word,1u,1u,request,
        SPARK_TP_MESH_PUBLISH_SHIP | SPARK_TP_MESH_PUBLISH_GUARD | SPARK_TP_MESH_PUBLISH_HALF,route.fields.peer_mask,timeout_ns);
    if ( SparkGpuPeekAtLastError() != SPARK_GPU_SUCCESS )
        return SparkGpuPeekAtLastError();
    return SparkTpMeshHardwareArrivalView(stream,band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns,SPARK_TP_MESH_VIEW_HALF);
}

static SparkGpuError SparkTpMeshHardwarePairFirstPublish(SparkGpuStream stream,uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,const uint8_t *local,uint8_t *staging,uint64_t segment,uint64_t round,uint64_t chunks,uint64_t timeout_ns)
{
    const uint64_t width = SparkTpMeshPairFirstScatterWidth(SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES);
    const uint64_t pair = round < chunks ? (segment - round * width < width ? segment - round * width : width) : 0u;
    const uint64_t group = round >= 2u ? (segment - (round - 2u) * width < width ? segment - (round - 2u) * width : width) : 0u;
    SparkWeightdMeshRoute route = {0};
    SparkGpuError status;
    route.fields.peer_mask = ((1u << degree) - 1u) & ~(1u << rank);
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_PEER;
    route.fields.pair_first = 1u;
    LM_LAUNCH((SparkTpMeshHardwarePublishGateKernel),1,1,0u,stream,request,control,
        SPARK_TP_MESH_PUBLISH_SHIP | SPARK_TP_MESH_PUBLISH_GUARD | (round != 0u ? SPARK_TP_MESH_PUBLISH_SHIP_PRIOR : 0u),timeout_ns);
    status = SparkGpuPeekAtLastError();
    if ( status == SPARK_GPU_SUCCESS && pair != 0u )
    {
        LM_LAUNCH((SparkTpMeshPairFirstStageKernel),dim3(SparkTpMeshCopyBlocks(pair,degree / 2u),degree / 2u),SPARK_TP_MESH_THREADS,0u,stream,control,(uint16_t *)staging,
            SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES / sizeof(uint16_t),(const uint16_t *)local + round * width,rank,segment,pair);
        status = SparkGpuPeekAtLastError();
    }
    if ( status != SPARK_GPU_SUCCESS )
        return status;
    LM_LAUNCH((SparkTpMeshHardwarePublishKernel),1,SPARK_TP_MESH_THREADS,0u,stream,band,slot_bytes,slots_per_rank,entry,control,rank,local,0u,0u,
        SPARK_WEIGHTD_MESH_PAIR_FIRST_BYTES(group * sizeof(float),pair * sizeof(uint16_t)),route.word,1u,1u,
        request,SPARK_TP_MESH_PUBLISH_HALF | SPARK_TP_MESH_PUBLISH_ARRIVAL,route.fields.peer_mask,timeout_ns);
    return SparkGpuPeekAtLastError();
}

static SparkGpuError SparkTpMeshHardwarePairFirstScatterRound(SparkGpuStream stream,uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,const uint8_t *local,void *output,uint8_t *staging,uint64_t segment,uint64_t timeout_ns)
{
    const uint64_t width = SparkTpMeshPairFirstScatterWidth(SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES);
    const uint64_t chunks = SparkTpMeshPairFirstScatterChunks(segment,SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES),rounds = chunks + 2u;
    uint64_t published,round;
    SparkGpuError status = SPARK_GPU_SUCCESS;
    if ( slot_bytes != SPARK_WEIGHTD_MESH_SLOT_BYTES || slots_per_rank != SPARK_WEIGHTD_MESH_SLOTS_PER_RANK || chunks < 3u || staging == 0 ||
         degree < 4u || (degree & 1u) != 0u )
        return SPARK_GPU_ERROR_INVALID_VALUE;
    for ( published = 0u; published < SPARK_TP_MESH_PIPELINE_LEAD && status == SPARK_GPU_SUCCESS; published++ )
        status = SparkTpMeshHardwarePairFirstPublish(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,local,staging,segment,published,chunks,timeout_ns);
    for ( round = 0u; round < rounds && status == SPARK_GPU_SUCCESS; round++ )
    {
        uint64_t lag = published - 1u - round;
        uint32_t view = SPARK_TP_MESH_VIEW_HALF | (lag != 0u ? SPARK_TP_MESH_VIEW_LAG : 0u);
        if ( lag > 1u ) return SPARK_GPU_ERROR_INVALID_VALUE;
        status = SparkTpMeshHardwareArrivalView(stream,band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns,view);
        if ( status == SPARK_GPU_SUCCESS && SparkTpMeshPipelineEarly(round,rounds) != SPARK_TP_MESH_PIPELINE_NONE )
        {
            status = SparkTpMeshHardwarePairFirstPublish(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,local,staging,segment,
                SparkTpMeshPipelineEarly(round,rounds),chunks,timeout_ns);
            published++;
            lag = published - 1u - round;
            view = SPARK_TP_MESH_VIEW_HALF | (lag != 0u ? SPARK_TP_MESH_VIEW_LAG : 0u);
        }
        if ( status == SPARK_GPU_SUCCESS && round < chunks )
        {
            const uint64_t count = segment - round * width < width ? segment - round * width : width;
            LM_LAUNCH((SparkTpMeshHardwarePublishGateKernel),1,1,0u,stream,request,control,SPARK_TP_MESH_PUBLISH_SHIP | SPARK_TP_MESH_PUBLISH_GUARD | SPARK_TP_MESH_PUBLISH_SHIP_PRIOR,timeout_ns);
            LM_LAUNCH((SparkTpMeshPairFirstPairKernel),dim3(SparkTpMeshCopyBlocks(count,degree / 2u),degree / 2u),SPARK_TP_MESH_THREADS,0u,stream,band,slot_bytes,slots_per_rank,control,
                (uint16_t *)staging,SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES / sizeof(uint16_t),(const uint16_t *)local + round * width,rank,segment,count,view);
            status = SparkGpuPeekAtLastError();
        }
        if ( status == SPARK_GPU_SUCCESS && round >= 2u )
        {
            const uint64_t begin = (round - 2u) * width,count = segment - begin < width ? segment - begin : width;
            LM_LAUNCH((SparkTpMeshPairFirstGroupKernel),SparkTpMeshDirectBlocks(count,1u),SPARK_TP_MESH_DIRECT_THREADS,0u,stream,band,slot_bytes,slots_per_rank,control,degree,
                (uint16_t *)output + begin,count,round + 1u == rounds ? 1u : 0u,rank,view);
            status = SparkGpuPeekAtLastError();
        }
        if ( status == SPARK_GPU_SUCCESS && SparkTpMeshPipelineLate(round,rounds) != SPARK_TP_MESH_PIPELINE_NONE )
        {
            status = SparkTpMeshHardwarePairFirstPublish(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,local,staging,segment,
                SparkTpMeshPipelineLate(round,rounds),chunks,timeout_ns);
            published++;
        }
    }
    if ( status != SPARK_GPU_SUCCESS )
        return status;
    return published == rounds ? SPARK_GPU_SUCCESS : SPARK_GPU_ERROR_INVALID_VALUE;
}

static SparkGpuError SparkTpMeshHardwarePairFirstGatherPublish(SparkGpuStream stream,uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,const uint8_t *local,uint8_t *staging,uint64_t local_elements,uint64_t round,uint64_t chunks,uint32_t view,uint64_t timeout_ns)
{
    const uint64_t width = SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES / sizeof(uint16_t);
    const uint64_t group = round < chunks ? (local_elements - round * width < width ? local_elements - round * width : width) : round >= chunks + 2u ? 4u : 0u;
    const uint64_t pair = round >= 2u && round < chunks + 2u ? (local_elements - (round - 2u) * width < width ? local_elements - (round - 2u) * width : width) : 0u;
    SparkWeightdMeshRoute route = {0};
    SparkGpuError status;
    route.fields.peer_mask = ((1u << degree) - 1u) & ~(1u << rank);
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_PEER;
    route.fields.pair_first = 1u;
    LM_LAUNCH((SparkTpMeshHardwarePublishGateKernel),1,1,0u,stream,request,control,
        SPARK_TP_MESH_PUBLISH_SHIP | SPARK_TP_MESH_PUBLISH_GUARD | (round != 0u ? SPARK_TP_MESH_PUBLISH_SHIP_PRIOR : 0u),timeout_ns);
    status = SparkGpuPeekAtLastError();
    if ( status == SPARK_GPU_SUCCESS )
    {
        LM_LAUNCH((SparkTpMeshPairFirstGatherStageKernel),dim3(SparkTpMeshCopyBlocks(width,degree),degree),SPARK_TP_MESH_THREADS,0u,stream,band,slot_bytes,slots_per_rank,control,
            (uint16_t *)staging,SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES / sizeof(uint16_t),(const uint16_t *)local + (round < chunks ? round * width : 0u),
            (const uint16_t *)local + (pair != 0u ? (round - 2u) * width : 0u),rank,group,pair,view);
        status = SparkGpuPeekAtLastError();
    }
    if ( status != SPARK_GPU_SUCCESS )
        return status;
    LM_LAUNCH((SparkTpMeshHardwarePublishKernel),1,SPARK_TP_MESH_THREADS,0u,stream,band,slot_bytes,slots_per_rank,entry,control,rank,local,0u,0u,
        SPARK_WEIGHTD_MESH_PAIR_FIRST_BYTES(group * sizeof(uint16_t),pair * sizeof(uint16_t)),route.word,1u,1u,
        request,SPARK_TP_MESH_PUBLISH_HALF | SPARK_TP_MESH_PUBLISH_ARRIVAL,route.fields.peer_mask,timeout_ns);
    return SparkGpuPeekAtLastError();
}

static SparkGpuError SparkTpMeshHardwarePairFirstGatherRound(SparkGpuStream stream,uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,const uint8_t *local,void *output,uint8_t *staging,uint64_t local_elements,uint64_t timeout_ns)
{
    const uint64_t width = SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES / sizeof(uint16_t);
    const uint64_t chunks = SparkTpMeshPipelineChunks(local_elements,SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES),rounds = chunks + SPARK_TP_MESH_PAIR_FIRST_GATHER_EXTRA_ROUNDS;
    uint64_t published,round;
    SparkGpuError status = SPARK_GPU_SUCCESS;
    if ( slot_bytes != SPARK_WEIGHTD_MESH_SLOT_BYTES || slots_per_rank != SPARK_WEIGHTD_MESH_SLOTS_PER_RANK || chunks < 3u || staging == 0 ||
         degree < 4u || (degree & 1u) != 0u || local_elements % 4u != 0u )
        return SPARK_GPU_ERROR_INVALID_VALUE;
    for ( published = 0u; published < SPARK_TP_MESH_PIPELINE_LEAD && status == SPARK_GPU_SUCCESS; published++ )
        status = SparkTpMeshHardwarePairFirstGatherPublish(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,local,staging,local_elements,published,chunks,0u,timeout_ns);
    for ( round = 0u; round < rounds && status == SPARK_GPU_SUCCESS; round++ )
    {
        uint64_t lag = published - 1u - round;
        uint32_t view = SPARK_TP_MESH_VIEW_HALF | (lag != 0u ? SPARK_TP_MESH_VIEW_LAG : 0u);
        if ( lag > 1u ) return SPARK_GPU_ERROR_INVALID_VALUE;
        status = SparkTpMeshHardwareArrivalView(stream,band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns,view);
        if ( status == SPARK_GPU_SUCCESS && SparkTpMeshPipelineEarly(round,rounds) != SPARK_TP_MESH_PIPELINE_NONE )
        {
            status = SparkTpMeshHardwarePairFirstGatherPublish(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,local,staging,local_elements,
                SparkTpMeshPipelineEarly(round,rounds),chunks,view,timeout_ns);
            published++;
            lag = published - 1u - round;
            view = SPARK_TP_MESH_VIEW_HALF | (lag != 0u ? SPARK_TP_MESH_VIEW_LAG : 0u);
        }
        if ( status == SPARK_GPU_SUCCESS )
        {
            LM_LAUNCH((SparkTpMeshPairFirstGatherKernel),dim3(SparkTpMeshDirectBlocks(width,degree),degree),SPARK_TP_MESH_DIRECT_THREADS,0u,stream,band,slot_bytes,slots_per_rank,control,
                (uint16_t *)output,(const uint16_t *)local,local_elements,round,chunks,round + 1u == rounds ? 1u : 0u,rank,view);
            status = SparkGpuPeekAtLastError();
        }
        if ( status == SPARK_GPU_SUCCESS && SparkTpMeshPipelineLate(round,rounds) != SPARK_TP_MESH_PIPELINE_NONE )
        {
            status = SparkTpMeshHardwarePairFirstGatherPublish(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,local,staging,local_elements,
                SparkTpMeshPipelineLate(round,rounds),chunks,view,timeout_ns);
            published++;
        }
    }
    if ( status != SPARK_GPU_SUCCESS )
        return status;
    return published == rounds ? SPARK_GPU_SUCCESS : SPARK_GPU_ERROR_INVALID_VALUE;
}

static SparkGpuError SparkTpMeshHardwareScatterRound(SparkGpuStream stream,uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,const uint8_t *local,void *output,uint8_t *staging,uint64_t elements,uint32_t slice_routes,uint64_t timeout_ns)
{
    const uint64_t segment = elements / degree,capacity = SparkTpMeshScatterCapacity(SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES);
    uint64_t begin,count;
    SparkGpuError status = SPARK_GPU_SUCCESS;
    if ( (slice_routes & SPARK_TP_MESH_ROUTES_PAIR_FIRST) != 0u )
        return SparkTpMeshScatterPairFirst(elements,degree,SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES) != 0u ?
            SparkTpMeshHardwarePairFirstScatterRound(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,local,output,staging,segment,timeout_ns) :
            SPARK_GPU_ERROR_INVALID_VALUE;
    if ( SparkTpMeshScatterPipelined(elements,degree,SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES) != 0u )
        return SparkTpMeshHardwarePipelineRound(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,local,output,staging,segment,
            SPARK_TP_MESH_PIECE_PUBLISH_SCATTER,SPARK_TP_MESH_PIECE_COMBINE_SUM,timeout_ns);
    for (begin=0u; begin<segment && status==SPARK_GPU_SUCCESS; begin+=count)
    {
        count = segment - begin < capacity ? segment - begin : capacity;
        status = SparkTpMeshHardwarePeerScatterChunk(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,local,output,staging,segment,begin,count,timeout_ns);
    }
    return status;
}

static SparkGpuError SparkTpMeshHardwareDirectRound(SparkGpuStream stream,uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,const uint8_t *local,void *output,uint64_t elements,uint32_t operation,uint32_t slice_routes,uint8_t *staging,uint64_t timeout_ns)
{
    uint64_t local_elements,capacity,begin,count;
    uint32_t rsag,peer;
    SparkGpuError status = SPARK_GPU_SUCCESS;
    local_elements = SparkTpMeshDirectLocalElements(elements,degree,operation);
    if ( operation == SPARK_TP_MESH_OPERATION_ALL_GATHER && (slice_routes & SPARK_TP_MESH_ROUTES_PAIR_FIRST) != 0u )
        return SparkTpMeshGatherPairFirst(elements,degree,slot_bytes,SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES) != 0u ?
            SparkTpMeshHardwarePairFirstGatherRound(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,local,output,staging,local_elements,timeout_ns) :
            SPARK_GPU_ERROR_INVALID_VALUE;
    if ( operation == SPARK_TP_MESH_OPERATION_ALL_GATHER && (slice_routes & SPARK_TP_MESH_ROUTES_PAIR) != 0u )
    {
        capacity = SparkTpMeshDirectCapacity(slot_bytes,operation);
        for (begin=0u; begin<local_elements && status==SPARK_GPU_SUCCESS; begin+=count)
        {
            count = local_elements - begin < capacity ? local_elements - begin : capacity;
            status = SparkTpMeshHardwarePairChunk(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,local,output,local_elements,begin,count,timeout_ns);
        }
        return status;
    }
    if ( SparkTpMeshDirectPipelined(elements,degree,operation,slot_bytes) != 0u )
        return SparkTpMeshHardwarePipelineRound(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,local,output,staging,local_elements,
            SPARK_TP_MESH_PIECE_PUBLISH_GATHER,SPARK_TP_MESH_PIECE_COMBINE_GATHER,timeout_ns);
    rsag = SparkTpMeshDirectPhasesPerChunk(elements,degree,operation,slice_routes) == 2u;
    peer = rsag != 0u && (slice_routes & SPARK_TP_MESH_ROUTES_PEER) != 0u;
    capacity = peer != 0u ? SparkTpMeshDirectPeerCapacity(degree,SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES) : SparkTpMeshDirectCapacity(slot_bytes,operation);
    for (begin=0u; begin<local_elements && status==SPARK_GPU_SUCCESS; begin+=count)
    {
        count = local_elements - begin < capacity ? local_elements - begin : capacity;
        if ( peer != 0u )
        {
            status = SparkTpMeshHardwarePeerRsagChunk(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,local,output,staging,local_elements,begin,count,timeout_ns,SPARK_TP_MESH_STAGE_ALL);
            continue;
        }
        status = rsag != 0u ? SparkTpMeshHardwareRsagChunk(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,local,output,local_elements,begin,count,timeout_ns,SPARK_TP_MESH_STAGE_ALL) : SparkTpMeshHardwareDirectChunk(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,local,output,local_elements,begin,count,operation,timeout_ns,SPARK_TP_MESH_STAGE_ALL);
    }
    return status;
}

static SparkGpuError SparkTpMeshHardwarePeerAllToAllRound(SparkGpuStream stream,uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,const uint8_t *local,void *output,uint8_t *staging,uint64_t per_peer_elements,uint64_t timeout_ns)
{
    SparkWeightdMeshRoute route = {0};
    uint64_t slice = SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES / sizeof(uint16_t),begin,count;
    SparkGpuError status = SPARK_GPU_SUCCESS;
    if ( staging == 0 || slice * sizeof(uint16_t) > slot_bytes - SPARK_WEIGHTD_MESH_SLOT_TRAILER_BYTES )
        return SPARK_GPU_ERROR_INVALID_VALUE;
    route.fields.peer_mask = ((1u << degree) - 1u) & ~(1u << rank);
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_PEER;
    if ( SparkTpMeshAllToAllPipelined(per_peer_elements,SPARK_WEIGHTD_MESH_HALF_PAYLOAD_BYTES) != 0u )
        return SparkTpMeshHardwarePipelineRound(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,degree,local,output,staging,per_peer_elements,
            SPARK_TP_MESH_PIECE_PUBLISH_SCATTER,SPARK_TP_MESH_PIECE_COMBINE_GATHER,timeout_ns);
    for (begin=0u; begin<per_peer_elements && status==SPARK_GPU_SUCCESS; begin+=count)
    {
        count = per_peer_elements - begin < slice ? per_peer_elements - begin : slice;
        route.fields.slice_bytes = ((count * sizeof(uint16_t) + 7u) & ~UINT64_C(7));
        status = SparkTpMeshHardwareWait(stream,request,control,SPARK_WEIGHTD_MESH_WAIT_SHIPPED,0u,0u,timeout_ns);
        if ( status != SPARK_GPU_SUCCESS )
            return status;
        LM_LAUNCH((SparkTpMeshAllToAllStageKernel),dim3(SparkTpMeshCopyBlocks(count,degree),degree),SPARK_TP_MESH_THREADS,0u,stream,band,slot_bytes,slots_per_rank,control,(uint16_t *)staging,slice,(const uint16_t *)local,degree,rank,per_peer_elements,begin,count);
        status = SparkGpuPeekAtLastError();
        if ( status != SPARK_GPU_SUCCESS )
            return status;
        LM_LAUNCH((SparkTpMeshHardwarePublishKernel),1,SPARK_TP_MESH_THREADS,0u,stream,band,slot_bytes,slots_per_rank,entry,control,rank,local,0u,0u,route.fields.slice_bytes,route.word,1u,1u,
            request,SPARK_TP_MESH_PUBLISH_ARRIVAL,route.fields.peer_mask,timeout_ns);
        status = SparkGpuPeekAtLastError();
        if ( status == SPARK_GPU_SUCCESS )
            status = SparkTpMeshHardwareArrival(stream,band,slot_bytes,slots_per_rank,control,rank,degree,timeout_ns);
        if ( status != SPARK_GPU_SUCCESS )
            return status;
        LM_LAUNCH((SparkTpMeshHardwareAllToAllKernel),dim3(SparkTpMeshCopyBlocks(count,degree),degree),SPARK_TP_MESH_THREADS,0u,stream,band,slot_bytes,slots_per_rank,control,degree,rank,output,per_peer_elements,begin,count,0u,begin + count == per_peer_elements);
        status = SparkGpuPeekAtLastError();
    }
    return status;
}

static SparkGpuError SparkTpMeshHardwareAllToAllRound(SparkGpuStream stream,uint8_t *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile uint64_t *entry,SparkWeightdMeshWaitRequest *request,SparkTpMeshRoundControl *control,uint32_t rank,uint32_t degree,const uint8_t *local,void *output,void *scratch,uint64_t per_peer_elements,uint64_t timeout_ns)
{
    SparkWeightdMeshRoute route = {0};
    uint64_t slice,begin,count,span;
    uint32_t blocks;
    SparkGpuError status = SPARK_GPU_SUCCESS;
    slice = SparkTpMeshAllToAllSliceElements(slot_bytes,degree);
    if ( slice == 0u || slice * sizeof(uint16_t) >= (1u << 24u) )
        return SPARK_GPU_ERROR_INVALID_VALUE;
    route.fields.peer_mask = ((1u << degree) - 1u) & ~(1u << rank);
    route.fields.slice_bytes = slice * sizeof(uint16_t);
    route.fields.mode = SPARK_WEIGHTD_MESH_ROUTE_SCATTER;
    for (begin=0u; begin<per_peer_elements && status==SPARK_GPU_SUCCESS; begin+=count)
    {
        count = per_peer_elements - begin < slice ? per_peer_elements - begin : slice;
        span = (uint64_t)degree * slice;
        blocks = (uint32_t)((span + SPARK_TP_MESH_THREADS - 1u) / SPARK_TP_MESH_THREADS);
        LM_LAUNCH((SparkTpMeshAllToAllPackKernel),blocks,SPARK_TP_MESH_THREADS,0u,stream,(uint16_t *)scratch,(const uint16_t *)local,degree,per_peer_elements,begin,count,slice);
        status = SparkGpuPeekAtLastError();
        if ( status == SPARK_GPU_SUCCESS )
            status = SparkTpMeshHardwareExchange(stream,band,slot_bytes,slots_per_rank,entry,request,control,rank,(const uint8_t *)scratch,span * sizeof(uint16_t),0u,span * sizeof(uint16_t),route,timeout_ns);
        if ( status != SPARK_GPU_SUCCESS )
            return status;
        LM_LAUNCH((SparkTpMeshHardwareAllToAllKernel),dim3(SparkTpMeshCopyBlocks(count,degree),degree),SPARK_TP_MESH_THREADS,0u,stream,band,slot_bytes,slots_per_rank,control,degree,rank,output,per_peer_elements,begin,count,slice,begin + count == per_peer_elements);
        status = SparkGpuPeekAtLastError();
    }
    return status;
}

extern "C" SparkGpuError SparkTpLaunchMeshHardware(SparkGpuStream stream,
    void *band,uint64_t slot_bytes,uint64_t slots_per_rank,volatile void *entry,
    void *gate,void *round_control,uint32_t rank,uint32_t degree,
    const void *local,void *output,void *scratch,uint64_t elements,
    uint32_t operation,uint32_t rounds,uint32_t logical_rows,uint32_t slice_routes,void *staging,uint64_t timeout_ns,uint32_t stage)
{
    if ( band == 0 || entry == 0 || gate == 0 || round_control == 0 || local == 0 ||
         output == 0 || scratch == 0 || elements == 0u || rounds == 0u || logical_rows == 0u ||
         degree < 2u || degree > 16u || rank >= degree || operation > SPARK_TP_MESH_OPERATION_REDUCE_SCATTER ||
         slot_bytes <= 16u || slots_per_rank != SPARK_WEIGHTD_MESH_SLOTS_PER_RANK || timeout_ns == 0u ||
         (operation == 0u && (elements % degree != 0u || local == output)) ||
         (operation == SPARK_TP_MESH_OPERATION_ALL_TO_ALL && (local == output || slice_routes == 0u)) ||
         (operation == SPARK_TP_MESH_OPERATION_REDUCE_SCATTER && (local == output || (slice_routes & SPARK_TP_MESH_ROUTES_PEER) == 0u ||
            elements % (4u * degree) != 0u || stage != SPARK_TP_MESH_STAGE_ALL)) ||
         ((slice_routes & SPARK_TP_MESH_ROUTES_PEER) != 0u && staging == 0) )
        return SPARK_GPU_ERROR_INVALID_VALUE;
    if ( stage != SPARK_TP_MESH_STAGE_ALL )
    {
        if ( stage > SPARK_TP_MESH_STAGE_COMPLETE || SparkTpMeshDirectStaged(slot_bytes,SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES,degree,elements,operation,rounds,logical_rows,slice_routes) == 0u )
            return SPARK_GPU_ERROR_INVALID_VALUE;
        uint64_t local_elements = SparkTpMeshDirectLocalElements(elements,degree,operation);
        uint32_t rsag = SparkTpMeshDirectPhasesPerChunk(elements,degree,operation,slice_routes) == 2u ? 1u : 0u;
        if ( rsag != 0u && (slice_routes & SPARK_TP_MESH_ROUTES_PEER) != 0u )
            return SparkTpMeshHardwarePeerRsagChunk(stream,(uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,(SparkWeightdMeshWaitRequest *)gate,
                (SparkTpMeshRoundControl *)round_control,rank,degree,(const uint8_t *)local,output,(uint8_t *)staging,local_elements,0u,local_elements,timeout_ns,stage);
        if ( rsag != 0u )
            return SparkTpMeshHardwareRsagChunk(stream,(uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,(SparkWeightdMeshWaitRequest *)gate,
                (SparkTpMeshRoundControl *)round_control,rank,degree,(const uint8_t *)local,output,local_elements,0u,local_elements,timeout_ns,stage);
        return SparkTpMeshHardwareDirectChunk(stream,(uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,(SparkWeightdMeshWaitRequest *)gate,
            (SparkTpMeshRoundControl *)round_control,rank,degree,(const uint8_t *)local,output,local_elements,0u,local_elements,operation,timeout_ns,stage);
    }
    if ( operation == SPARK_TP_MESH_OPERATION_REDUCE_SCATTER )
    {
        for ( uint32_t round = 0u; round < rounds; round++ )
        {
            SparkGpuError status = SparkTpMeshHardwareScatterRound(stream,(uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,(SparkWeightdMeshWaitRequest *)gate,
                (SparkTpMeshRoundControl *)round_control,rank,degree,(const uint8_t *)local,output,(uint8_t *)staging,elements,slice_routes,timeout_ns);
            if ( status != SPARK_GPU_SUCCESS ) return status;
        }
        return SPARK_GPU_SUCCESS;
    }
    if ( operation == SPARK_TP_MESH_OPERATION_ALL_TO_ALL )
    {
        for ( uint32_t round = 0u; round < rounds; round++ )
        {
            SparkGpuError status = (slice_routes & SPARK_TP_MESH_ROUTES_PEER) != 0u ?
                SparkTpMeshHardwarePeerAllToAllRound(stream,(uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,(SparkWeightdMeshWaitRequest *)gate,(SparkTpMeshRoundControl *)round_control,rank,degree,(const uint8_t *)local,output,(uint8_t *)staging,elements,timeout_ns) :
                SparkTpMeshHardwareAllToAllRound(stream,(uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,(SparkWeightdMeshWaitRequest *)gate,(SparkTpMeshRoundControl *)round_control,rank,degree,(const uint8_t *)local,output,scratch,elements,timeout_ns);
            if ( status != SPARK_GPU_SUCCESS ) return status;
        }
        return SPARK_GPU_SUCCESS;
    }
    SparkTpMeshRoundControl *control = (SparkTpMeshRoundControl *)round_control;
    SparkWeightdMeshWaitRequest *request = (SparkWeightdMeshWaitRequest *)gate;
    uint32_t levels = SparkTpMeshTreeLevels(degree);
    uint32_t width = operation == 2u ? 8u : operation == 1u ? 4u : 2u;
    uint64_t capacity = (slot_bytes - 16u) / width;
    if ( capacity == 0u ) return SPARK_GPU_ERROR_INVALID_VALUE;
    if ( SparkTpMeshDirectCapacity(slot_bytes,operation) == 0u ) return SPARK_GPU_ERROR_INVALID_VALUE;
    for ( uint32_t round = 0u; round < rounds; round++ )
    {
        SparkGpuError status;
        if ( logical_rows == 1u )
        {
            status = SparkTpMeshHardwareDirectRound(stream,(uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,request,control,rank,degree,(const uint8_t *)local,output,elements,operation,slice_routes,(uint8_t *)staging,timeout_ns);
            if ( status != SPARK_GPU_SUCCESS ) return status;
            continue;
        }
        for ( uint64_t begin = 0u; begin < elements; begin += capacity )
        {
            uint64_t count = elements - begin < capacity ? elements - begin : capacity;
            LM_LAUNCH((SparkTpMeshHardwareSeedKernel),1,SPARK_TP_MESH_THREADS,0u,stream,
                control,scratch,local,elements,begin,count,operation,rank,degree,2u * levels);
            status = SparkGpuPeekAtLastError();
            if ( status != SPARK_GPU_SUCCESS ) return status;
            for ( uint32_t phase = 0u; phase < 2u * levels; phase++ )
            {
                uint32_t route = SparkTpMeshTreeRoute(rank,degree,phase);
                uint32_t send = route >> 16u,receive = route & 0xffffu;
                if ( send != 0u )
                {
                    status = SparkTpMeshHardwareWait(stream,request,control,
                        SPARK_WEIGHTD_MESH_WAIT_SHIPPED,0u,0u,timeout_ns);
                    if ( status != SPARK_GPU_SUCCESS ) return status;
                    LM_LAUNCH((SparkTpMeshHardwarePublishKernel),1,SPARK_TP_MESH_THREADS,0u,stream,
                        (uint8_t *)band,slot_bytes,slots_per_rank,(volatile uint64_t *)entry,control,
                        rank,scratch,count * width,0u,count * width,1u << (send - 1u),phase + 1u,0u,0,0u,0u,0u);
                }
                else if ( receive != 0u )
                {
                    status = SparkTpMeshHardwareWait(stream,request,control,
                        SPARK_WEIGHTD_MESH_WAIT_PEERS,1u << (receive - 1u),phase + 1u,timeout_ns);
                    if ( status != SPARK_GPU_SUCCESS ) return status;
                    LM_LAUNCH((SparkTpMeshHardwareFoldKernel),1,SPARK_TP_MESH_THREADS,0u,stream,
                        (const uint8_t *)band,slot_bytes,slots_per_rank,control,scratch,count,
                        operation,receive - 1u,phase,levels);
                }
                status = SparkGpuPeekAtLastError();
                if ( status != SPARK_GPU_SUCCESS ) return status;
            }
            LM_LAUNCH((SparkTpMeshHardwareFinishKernel),1,SPARK_TP_MESH_THREADS,0u,stream,
                control,output,scratch,begin,count,operation,2u * levels,begin + count == elements);
            status = SparkGpuPeekAtLastError();
            if ( status != SPARK_GPU_SUCCESS ) return status;
        }
    }
    return SPARK_GPU_SUCCESS;
}

extern "C" void SparkTpDeviceCollectiveRegisterLaunchers(const SparkTpMeshLaunchers *launchers) __attribute__((weak));

static const SparkTpMeshLaunchers SparkTpMeshKernelLaunchers =
{
    SPARK_TP_MESH_LAUNCHERS_ABI_VERSION,
    (uint32_t)sizeof(SparkTpMeshLaunchers),
    reinterpret_cast<SparkTpMeshCopyDownFunction>(SparkTpLaunchMeshCopyDown),
    reinterpret_cast<SparkTpMeshPublishFunction>(SparkTpLaunchMeshPublish),
    reinterpret_cast<SparkTpMeshTreeFunction>(SparkTpLaunchMeshTree),
    reinterpret_cast<SparkTpMeshHardwarePrepareFunction>(SparkTpMeshHardwarePrepare),
    reinterpret_cast<SparkTpMeshHardwareFunction>(SparkTpLaunchMeshHardware),
    reinterpret_cast<SparkTpMeshSeqPadFunction>(SparkTpLaunchMeshSeqPad),
    reinterpret_cast<SparkTpMeshGuardFunction>(SparkTpLaunchMeshGuard),
    reinterpret_cast<SparkTpMeshWaitFunction>(SparkTpLaunchMeshWait),
    reinterpret_cast<SparkTpMeshRoundLoopFunction>(SparkTpLaunchMeshRoundLoop)
};

__attribute__((constructor)) static void SparkTpMeshKernelLaunchersInstall(void)
{
    if ( SparkTpDeviceCollectiveRegisterLaunchers != 0 )
        SparkTpDeviceCollectiveRegisterLaunchers(&SparkTpMeshKernelLaunchers);
}

#endif
