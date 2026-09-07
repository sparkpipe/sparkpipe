#pragma once


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
	LM_FRAME_ERROR_ROUTE_MAP_OUT_OF_RANGE = 7u,
	LM_FRAME_ERROR_SPARSE_INDEX_OUT_OF_RANGE = 8u,
	LM_FRAME_ERROR_PAYLOAD_WINDOW_OUT_OF_RANGE = 9u
}
LmFrameErrorCode;

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
