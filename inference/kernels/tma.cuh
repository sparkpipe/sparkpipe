#pragma once


#include <cuda_runtime.h>
#include <stdint.h>

#define LM_TMA_ALIGNMENT_BYTES 128u

static __device__ __forceinline__ uint32_t LmTmaSharedAddress(const void *shared_pointer)
{
	return(static_cast<uint32_t>(__cvta_generic_to_shared(const_cast<void *>(shared_pointer))));
}

static __device__ __forceinline__ uint32_t LmTmaBoxBytes(uint32_t rows, uint32_t columns, uint32_t element_bytes)
{
	return(rows * columns * element_bytes);
}


static __device__ __forceinline__ void LmMbarrierInit(uint64_t *barrier, uint32_t arrive_count)
{
	asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
		:: "r"(LmTmaSharedAddress(barrier)), "r"(arrive_count));
}

static __device__ __forceinline__ void LmMbarrierInvalidate(uint64_t *barrier)
{
	asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
		:: "r"(LmTmaSharedAddress(barrier)));
}

static __device__ __forceinline__ void LmMbarrierInitFence(void)
{
	asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

static __device__ __forceinline__ void LmMbarrierArriveExpect(uint64_t *barrier, uint32_t transaction_bytes)
{
	uint64_t state;
	asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;\n"
		: "=l"(state) : "r"(LmTmaSharedAddress(barrier)), "r"(transaction_bytes));
}

static __device__ __forceinline__ void LmMbarrierArrive(uint64_t *barrier)
{
	uint64_t state;
	asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];\n"
		: "=l"(state) : "r"(LmTmaSharedAddress(barrier)));
}

static __device__ __forceinline__ bool LmMbarrierTryWait(uint64_t *barrier, uint32_t phase)
{
	uint32_t ready;
	asm volatile("{\n\t.reg .pred P;\n\tmbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\n\tselp.b32 %0, 1, 0, P;\n\t}\n"
		: "=r"(ready) : "r"(LmTmaSharedAddress(barrier)), "r"(phase));
	return(ready != 0u);
}

static __device__ __forceinline__ void LmMbarrierWait(uint64_t *barrier, uint32_t phase)
{
	while ( LmMbarrierTryWait(barrier,phase) == false )
		;
}

static __device__ __forceinline__ void LmTmaLoad2d(void *shared_destination, const void *tensor_map, uint64_t *barrier, int32_t coordinate_0, int32_t coordinate_1)
{
	asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
		:: "r"(LmTmaSharedAddress(shared_destination)), "l"(tensor_map),
		   "r"(LmTmaSharedAddress(barrier)), "r"(coordinate_0), "r"(coordinate_1)
		: "memory");
}

static __device__ __forceinline__ void LmTmaLoad3d(void *shared_destination, const void *tensor_map, uint64_t *barrier, int32_t coordinate_0, int32_t coordinate_1, int32_t coordinate_2)
{
	asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];\n"
		:: "r"(LmTmaSharedAddress(shared_destination)), "l"(tensor_map),
		   "r"(LmTmaSharedAddress(barrier)), "r"(coordinate_0), "r"(coordinate_1), "r"(coordinate_2)
		: "memory");
}

static __device__ __forceinline__ void LmTmaLoadBulk1d(void *shared_destination, const void *global_source, uint64_t *barrier, uint32_t bytes)
{
	asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
		:: "r"(LmTmaSharedAddress(shared_destination)), "l"(global_source),
		   "r"(bytes), "r"(LmTmaSharedAddress(barrier))
		: "memory");
}

static __device__ __forceinline__ void LmTmaStore2d(const void *tensor_map, const void *shared_source, int32_t coordinate_0, int32_t coordinate_1)
{
	asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];\n"
		:: "l"(tensor_map), "r"(LmTmaSharedAddress(shared_source)), "r"(coordinate_0), "r"(coordinate_1)
		: "memory");
}

static __device__ __forceinline__ void LmTmaStoreCommit(void)
{
	asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<uint32_t KEEP>
static __device__ __forceinline__ void LmTmaStoreWait(void)
{
	asm volatile("cp.async.bulk.wait_group.read %0;\n" :: "n"(KEEP) : "memory");
}

static __device__ __forceinline__ void LmTmaStoreFence(void)
{
	asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
