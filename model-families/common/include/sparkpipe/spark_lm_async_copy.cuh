#pragma once

// Asynchronous global -> shared staging.
//
// Nothing in this codebase issues cp.async. Every staging loop is a synchronous
// global load into a register followed by a shared store, which stalls the warp
// on memory latency and burns registers holding data that is only in transit.
// cp.async moves bytes global -> shared without landing in registers and without
// blocking the issuing warp, so a tile can be fetched for step N+1 while step N
// computes.
//
// Availability is not assumed. Every PTX form below is assembled against the
// shipping target by tests/test_ptx_capability_gate.py on each `make test`; if
// the target stops accepting one, that test fails rather than this header
// silently compiling to something slower or not at all.
//
// There is deliberately no pre-Ampere fallback. A fallback would silently turn
// an async pipeline into a synchronous one and the caller would never know why
// the numbers moved. Compiling this for a target without cp.async is a build
// error, which is the correct outcome.

#include <cuda_runtime.h>
#include <stdint.h>

// cp.async requires the shared destination to be aligned to the transfer width,
// and 16-byte transfers are the only width that reaches full bandwidth.
#define SPARK_LM_ASYNC_COPY_WIDEST_BYTES 16u

// A .cg copy bypasses L1 and is the right choice for tiles that are streamed
// once. A .ca copy caches in L1 and is the right choice for tiles that several
// CTAs will read. Callers pick; there is no default that silently guesses.
enum SparkLmAsyncCopyCache
{
	SPARK_LM_ASYNC_COPY_CACHE_ALL = 0,
	SPARK_LM_ASYNC_COPY_CACHE_GLOBAL = 1
};

static __device__ __forceinline__ uint32_t SparkLmAsyncSharedAddress(void *shared_destination)
{
	return(static_cast<uint32_t>(__cvta_generic_to_shared(shared_destination)));
}

// One in-flight copy of BYTES from global to shared. BYTES must be 4, 8 or 16;
// the .cg cache policy exists only at 16, which the static_assert enforces
// rather than leaving ptxas to reject it later with a less obvious message.
template<uint32_t BYTES, SparkLmAsyncCopyCache CACHE>
static __device__ __forceinline__ void SparkLmAsyncCopy(void *shared_destination, const void *global_source)
{
	static_assert(BYTES == 4u || BYTES == 8u || BYTES == 16u, "cp.async transfers 4, 8 or 16 bytes");
	static_assert(CACHE == SPARK_LM_ASYNC_COPY_CACHE_ALL || BYTES == SPARK_LM_ASYNC_COPY_WIDEST_BYTES,
		"the .cg cache policy exists only for 16-byte transfers");
	// if constexpr, not if: a plain branch emits BOTH asm strings into the PTX
	// and only optimisation removes the dead one. At -O0 that emits a .cg copy
	// with a 4-byte size, which ptxas rejects outright.
	if constexpr ( CACHE == SPARK_LM_ASYNC_COPY_CACHE_GLOBAL )
		asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
			:: "r"(SparkLmAsyncSharedAddress(shared_destination)), "l"(global_source), "n"(BYTES));
	else
		asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n"
			:: "r"(SparkLmAsyncSharedAddress(shared_destination)), "l"(global_source), "n"(BYTES));
}

// The same copy, bounded. Bytes beyond `source_bytes` are zero-filled by the
// hardware instead of read, which is what makes a ragged tail safe without a
// branch per thread and without a separate epilogue kernel.
template<uint32_t BYTES, SparkLmAsyncCopyCache CACHE>
static __device__ __forceinline__ void SparkLmAsyncCopyBounded(void *shared_destination, const void *global_source, uint32_t source_bytes)
{
	static_assert(BYTES == 4u || BYTES == 8u || BYTES == 16u, "cp.async transfers 4, 8 or 16 bytes");
	static_assert(CACHE == SPARK_LM_ASYNC_COPY_CACHE_ALL || BYTES == SPARK_LM_ASYNC_COPY_WIDEST_BYTES,
		"the .cg cache policy exists only for 16-byte transfers");
	if constexpr ( CACHE == SPARK_LM_ASYNC_COPY_CACHE_GLOBAL )
		asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n"
			:: "r"(SparkLmAsyncSharedAddress(shared_destination)), "l"(global_source), "n"(BYTES), "r"(source_bytes));
	else
		asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n"
			:: "r"(SparkLmAsyncSharedAddress(shared_destination)), "l"(global_source), "n"(BYTES), "r"(source_bytes));
}

// Close the current batch of copies. Everything issued since the last commit
// becomes one group that can be waited on as a unit.
static __device__ __forceinline__ void SparkLmAsyncCommit(void)
{
	asm volatile("cp.async.commit_group;\n" ::: "memory");
}

// Wait until at most KEEP_IN_FLIGHT groups remain outstanding. A double-buffered
// loop waits with KEEP_IN_FLIGHT = 1, which lets the next tile keep flying while
// the current one is consumed.
template<uint32_t KEEP_IN_FLIGHT>
static __device__ __forceinline__ void SparkLmAsyncWait(void)
{
	asm volatile("cp.async.wait_group %0;\n" :: "n"(KEEP_IN_FLIGHT) : "memory");
}

static __device__ __forceinline__ void SparkLmAsyncWaitAll(void)
{
	asm volatile("cp.async.wait_all;\n" ::: "memory");
}

// Stage a contiguous byte range with the whole CTA, 16 bytes per thread per
// step. `byte_count` need not be a multiple of the transfer width or of the
// thread count: the tail is zero-filled by the bounded form above.
//
// This is the shape every hand-rolled staging loop in this codebase is already
// doing synchronously. Callers issue this, then SparkLmAsyncCommit, then compute
// on the previous tile before waiting.
template<SparkLmAsyncCopyCache CACHE>
static __device__ __forceinline__ void SparkLmAsyncStageBytes(void *shared_destination, const void *global_source, uint32_t byte_count, uint32_t thread_index, uint32_t thread_count)
{
	uint8_t *destination;
	const uint8_t *source;
	uint32_t offset, remaining;

	destination = static_cast<uint8_t *>(shared_destination);
	source = static_cast<const uint8_t *>(global_source);
	for (offset = thread_index * SPARK_LM_ASYNC_COPY_WIDEST_BYTES;
		offset < byte_count;
		offset += thread_count * SPARK_LM_ASYNC_COPY_WIDEST_BYTES)
	{
		remaining = byte_count - offset;
		if ( remaining >= SPARK_LM_ASYNC_COPY_WIDEST_BYTES )
			SparkLmAsyncCopy<SPARK_LM_ASYNC_COPY_WIDEST_BYTES, CACHE>(destination + offset, source + offset);
		else
			SparkLmAsyncCopyBounded<SPARK_LM_ASYNC_COPY_WIDEST_BYTES, CACHE>(destination + offset, source + offset, remaining);
	}
}
