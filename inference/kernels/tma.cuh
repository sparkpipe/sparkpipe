#pragma once

// Tensor-memory-accelerator staging and mbarrier pipeline synchronisation.
//
// lm/lm_dtype.cuh issues cp.async, which is a per-thread instruction:
// every thread in the CTA computes an address and issues its own transfer, and
// the pipeline is tracked by commit-group depth. That works, but it burns issue
// slots proportional to tile bytes and it cannot express a bounded 2D box.
//
// TMA moves a whole tile with ONE instruction from ONE thread, addressed by
// tensor coordinates rather than by a linear pointer, and completion is tracked
// by an mbarrier transaction count rather than by group depth. That is the
// structure CUTLASS's SM120 collective uses - see
// third_party/flashinfer/3rdparty/cutlass/include/cutlass/gemm/collective/
// sm120_mma_array_tma_blockwise_scaling.hpp, which static_asserts its
// GmemTiledCopy to SM90_TMA_LOAD and builds its MainloopPipeline from
// PipelineTmaAsync<Stages>. Matching that structure is the point of this file.
//
// Every PTX form below assembles against the shipping target; see
// tests/test_ptx_capability_gate.py, which fails the build if one stops.
//
// Renamed from spark_lm_tma.cuh into lm/ with the rest of the rewrite. The
// content is unchanged and was already assembler-verified; only the prefix
// moved.
//
// There is no cp.async fallback here on purpose. A fallback would silently turn
// a one-instruction tile fetch into a per-thread address computation loop and
// nothing would report the change. lm/lm_dtype.cuh remains available as
// an explicit, separately selected staging path.
//
// LmTmaLoadBulk1d is NOT that fallback. It is the same mbarrier-transaction
// machinery with a linear address instead of a tensor map, and it exists for
// exactly one job: the indirect-A grouped GEMM (route.cuh's consumer contract),
// where each staged A row comes from an index the affine box coordinates cannot
// express. Completion is still complete_tx against the stage barrier, so the
// pipeline's arrive/expect/parity protocol cannot tell the two paths apart.

#include <cuda_runtime.h>
#include <stdint.h>

// A TMA box is addressed in elements, but the transaction count an mbarrier
// expects is in bytes, and the two are only consistent if the caller derives
// one from the other. LmTmaBoxBytes is the single place that conversion
// happens.
#define LM_TMA_ALIGNMENT_BYTES 128u

static __device__ __forceinline__ uint32_t LmTmaSharedAddress(const void *shared_pointer)
{
	return(static_cast<uint32_t>(__cvta_generic_to_shared(const_cast<void *>(shared_pointer))));
}

static __device__ __forceinline__ uint32_t LmTmaBoxBytes(uint32_t rows, uint32_t columns, uint32_t element_bytes)
{
	return(rows * columns * element_bytes);
}

// The caller elects the single CTA producer with threadIdx.x == 0.
// Warp-scoped elect.sync is intentionally not used: it elects once per warp,
// which duplicates TMA transactions and barrier arrivals in a multi-warp CTA.

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

// An mbarrier is initialised by one thread and read by all of them, and the
// initialising store is in the generic proxy while the TMA completion write is
// in the async proxy. Without this fence the two are not ordered and a consumer
// can observe an uninitialised barrier.
static __device__ __forceinline__ void LmMbarrierInitFence(void)
{
	asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// Arrive and declare how many bytes this phase will receive. The barrier flips
// phase when the arrive count is met AND the byte count has landed, so the
// declared total must equal the sum of every box issued into this stage or the
// pipeline deadlocks. Callers derive it from LmTmaBoxBytes rather than
// writing a literal.
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

// try_wait returns a predicate rather than blocking, so the spin lives in C++
// where it needs no PTX label. Inline-asm labels collide when a function is
// inlined more than once in a translation unit; returning the predicate avoids
// that class of failure entirely.
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

// Load a 2D box. The tensor map is a CUtensorMap built on the host by
// cuTensorMapEncodeTiled and passed as a __grid_constant__ parameter; it encodes
// the global base, the element type, the box shape and the swizzle, so none of
// those appear here. Coordinates are in elements and are bounds-checked by the
// hardware, which zero-fills out-of-range elements - that is what makes a ragged
// group tail safe with no branch and no separate epilogue kernel.
static __device__ __forceinline__ void LmTmaLoad2d(void *shared_destination, const void *tensor_map, uint64_t *barrier, int32_t coordinate_0, int32_t coordinate_1)
{
	asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
		:: "r"(LmTmaSharedAddress(shared_destination)), "l"(tensor_map),
		   "r"(LmTmaSharedAddress(barrier)), "r"(coordinate_0), "r"(coordinate_1)
		: "memory");
}

// Load a 3D box. Expert-major weights are exactly this: coordinate 2 selects the
// expert, so one descriptor covers all 256 of them and the grouped dispatch
// never rebuilds a tensor map per group.
static __device__ __forceinline__ void LmTmaLoad3d(void *shared_destination, const void *tensor_map, uint64_t *barrier, int32_t coordinate_0, int32_t coordinate_1, int32_t coordinate_2)
{
	asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];\n"
		:: "r"(LmTmaSharedAddress(shared_destination)), "l"(tensor_map),
		   "r"(LmTmaSharedAddress(barrier)), "r"(coordinate_0), "r"(coordinate_1), "r"(coordinate_2)
		: "memory");
}

// Bulk-copy BYTES from a linear global address into shared, completion counted
// against the same mbarrier transaction total a tensor box would use. This is
// the staging primitive for rows a tensor map cannot address: an indirect A
// row lives at base + index[p] * pitch, and the 16-byte swizzle chunk is the
// largest span whose destination is not permuted by the row's own swizzle, so
// one of these moves one chunk. Unlike the tensor forms there is no hardware
// bounds check and no zero fill - the caller clamps every index into a live
// row, because a wild one faults instead of padding.
static __device__ __forceinline__ void LmTmaLoadBulk1d(void *shared_destination, const void *global_source, uint64_t *barrier, uint32_t bytes)
{
	asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
		:: "r"(LmTmaSharedAddress(shared_destination)), "l"(global_source),
		   "r"(bytes), "r"(LmTmaSharedAddress(barrier))
		: "memory");
}

// Shared -> global for the epilogue. Completion is tracked by bulk group depth,
// not by an mbarrier, because nothing waits on the store except the next reuse
// of the staging buffer.
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

// Retire all but KEEP outstanding store groups. .read is the weaker and cheaper
// form: it only guarantees the source shared memory is readable again, which is
// the actual requirement before overwriting a staging buffer.
template<uint32_t KEEP>
static __device__ __forceinline__ void LmTmaStoreWait(void)
{
	asm volatile("cp.async.bulk.wait_group.read %0;\n" :: "n"(KEEP) : "memory");
}

// Writes into shared memory that a TMA store will read happen in the generic
// proxy; the store reads in the async proxy. This orders the two.
static __device__ __forceinline__ void LmTmaStoreFence(void)
{
	asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
