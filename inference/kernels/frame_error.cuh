#pragma once

// THE PER-FRAME ERROR RECORD - the fail-frame, never kill-the-context path.
//
// A kernel that cannot honor its inputs (a corrupt route map, an out-of-range
// sparse index, an oversized payload window, an unmapped KV page) records the
// FIRST failure here and returns a bounded result. It never traps: a device-
// side trap terminates the whole CUDA context, so one corrupt frame would
// take down every resident model, every request, and every lane sharing the
// GPU. The frame is the failure boundary, not the context.
//
// THE LIFECYCLE is owned by the driver, not the kernel:
//
//   1. the driver zeroes the slot before the frame (cudaMemsetAsync on the
//      frame's stream, or a plain memset of the host mirror);
//   2. kernels report through LmFrameErrorReport - first writer wins, so the
//      record holds the FIRST failure and no thread spins;
//   3. the driver copies the slot back at the end of the frame and checks
//      error_code != LM_FRAME_ERROR_NONE - a non-zero code fails the frame
//      (SPARK_STATUS_INTERNAL_ERROR) with the record's fields for diagnosis.
//
// The record is six words so a KV page failure and a route-map failure share
// one layout and one host check. The field names name the KV case (the first
// consumer); every other reporter documents its own packing in a comment at
// the report site. glm5_next's execution slots carry exactly this record as
// kv_access_error and are the reference wiring.
//
// THIS HEADER COMPILES EVERYWHERE by the scale.cuh contract: under nvcc it
// is host+device; under a plain host compiler (the CPU shim, C test TUs) the
// space macros collapse and the atomics degrade to plain reads and writes,
// which is the sequential schedule a correct kernel must be valid under
// anyway.

#include <stdint.h>

#if !defined(__CUDACC__) && !defined(__forceinline__)
#define LM_FRAME_ERROR_HOST_FALLBACK 1
#define __forceinline__ inline
#define __device__
#define __host__
#endif

typedef enum LmFrameErrorCode
{
	LM_FRAME_ERROR_NONE = 0u,
	LM_FRAME_ERROR_INVALID_VIEW = 1u,
	LM_FRAME_ERROR_SEQUENCE_OUT_OF_RANGE = 2u,
	LM_FRAME_ERROR_PAGE_TABLE_OUT_OF_RANGE = 3u,
	LM_FRAME_ERROR_PAGE_UNMAPPED = 4u,
	LM_FRAME_ERROR_POOL_PAGE_OUT_OF_RANGE = 5u,
	LM_FRAME_ERROR_INVALID_GQA_GEOMETRY = 6u,
	// Route-map corruption (K1): a packed row's source token is past the
	// activation tensor's row count. Report fields: row = packed row,
	// sequence = the out-of-range source row, position = row_base,
	// page = source_row_count.
	LM_FRAME_ERROR_ROUTE_MAP_OUT_OF_RANGE = 7u,
	// Sparse-selection corruption (K2): a top-k block/position index is past
	// the table or pool bound that gives it meaning. Report fields: row =
	// attention row, sequence = the raw index, position = the derived
	// position, page = the bound it violated.
	LM_FRAME_ERROR_SPARSE_INDEX_OUT_OF_RANGE = 8u,
	// Payload window corruption (K4): a compressed payload's declared span
	// does not fit the fixed shared-memory window that stages it. Report
	// fields: row = chunk ordinal, sequence = the declared span bytes,
	// position = the window capacity, page = 0.
	LM_FRAME_ERROR_PAYLOAD_WINDOW_OUT_OF_RANGE = 9u
}
LmFrameErrorCode;

// The two-phase publish protocol: claim with RECORDING through a CAS so
// exactly one thread fills the record, then release the real code with a
// system-scope exchange so a host reading after the launch sees the code and
// not a half-filled struct.
#define LM_FRAME_ERROR_RECORDING 0xffffffffu

#define LM_FRAME_ERROR_WORDS 6u

typedef struct LmFrameError
{
	uint32_t error_code;
	uint32_t access_kind;
	uint32_t row;
	uint32_t sequence;
	uint32_t position;
	uint32_t page;
}
LmFrameError;

static __host__ __device__ __forceinline__ void LmFrameErrorReset(
	LmFrameError *error)
{
	if ( error == 0 )
		return;
	error->error_code = LM_FRAME_ERROR_NONE;
	error->access_kind = 0u;
	error->row = 0xffffffffu;
	error->sequence = 0xffffffffu;
	error->position = 0xffffffffu;
	error->page = 0xffffffffu;
}

static __device__ __forceinline__ uint32_t LmFrameErrorCompareExchange(
	uint32_t *address,
	uint32_t expected,
	uint32_t desired)
{
#if defined(__CUDA_ARCH__)
	return(atomicCAS(address,expected,desired));
#else
	uint32_t previous = *address;
	if ( previous == expected )
		*address = desired;
	return(previous);
#endif
}

static __device__ __forceinline__ void LmFrameErrorPublish(
	uint32_t *address,
	uint32_t value)
{
#if defined(__CUDA_ARCH__)
	__threadfence_system();
	atomicExch(address,value);
#else
	*address = value;
#endif
}

// Report one frame failure. Null-safe: a kernel launched by a driver that has
// not wired a slot yet still gets the bounded fallback behavior, it just
// cannot name the failure - wiring the slot is what makes the frame fail
// loud instead of silent, and every live corruption site's driver carries one.
static __device__ __forceinline__ void LmFrameErrorReport(
	LmFrameError *error,
	uint32_t error_code,
	uint32_t access_kind,
	uint32_t row,
	uint32_t sequence,
	uint32_t position,
	uint32_t page)
{
	if ( error == 0 )
		return;
	if ( LmFrameErrorCompareExchange(
		&error->error_code,
		LM_FRAME_ERROR_NONE,
		LM_FRAME_ERROR_RECORDING) != LM_FRAME_ERROR_NONE )
		return;
	error->access_kind = access_kind;
	error->row = row;
	error->sequence = sequence;
	error->position = position;
	error->page = page;
	LmFrameErrorPublish(&error->error_code,error_code);
}

#if defined(LM_FRAME_ERROR_HOST_FALLBACK)
#undef __forceinline__
#undef __device__
#undef __host__
#undef LM_FRAME_ERROR_HOST_FALLBACK
#endif
