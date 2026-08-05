#pragma once

// cuTensorMapEncodeTiled. The host half of the TMA descriptor.
//
// kernels/tensor_map.cuh computes the geometry with stdint alone; this is the
// driver translation over an already-validated plan. The split is deliberate:
// the arithmetic is where the silent errors live - NVFP4 K extents halving, the
// swizzle span varying with the stored width, the rank-1 stride convention - and
// it should be checkable on any host with no CUDA at all.
//
// cuTensorMapEncodeTiled is a DRIVER API entry point. It needs cuda.h and a
// current context, not just the runtime API, so the link line gains -lcuda.

#include "inference/kernels/tensor_map.cuh"
#include <cuda.h>
#include <stdint.h>
#include <string.h>

#define LM_TM_ENCODE_OK 0
#define LM_TM_ENCODE_ERR_PLAN (-21)
#define LM_TM_ENCODE_ERR_NULL (-22)
#define LM_TM_ENCODE_ERR_DRIVER (-23)
#define LM_TM_ENCODE_ERR_SWIZZLE (-24)
#define LM_TM_ENCODE_ERR_ALIGNMENT (-25)

// The descriptor swizzle must be the one the kernel's chunk xor implements.
// spark_lm_tensor_map.h fixes the span at 128 bytes and the kernel
// static_asserts its chunk count against the shared contract header, so this
// only has to reject a plan that arrived with a different span rather than
// choose one.
static int32_t LmTensorMapSwizzleEnum(uint32_t swizzle_bytes, CUtensorMapSwizzle *out)
{
	if ( out == 0 )
		return(LM_TM_ENCODE_ERR_NULL);
	if ( swizzle_bytes == 0u )
	{
		*out = CU_TENSOR_MAP_SWIZZLE_NONE;
		return(LM_TM_ENCODE_OK);
	}
	if ( swizzle_bytes == 128u )
	{
		*out = CU_TENSOR_MAP_SWIZZLE_128B;
		return(LM_TM_ENCODE_OK);
	}
	if ( swizzle_bytes == 64u )
	{
		*out = CU_TENSOR_MAP_SWIZZLE_64B;
		return(LM_TM_ENCODE_OK);
	}
	if ( swizzle_bytes == 32u )
	{
		*out = CU_TENSOR_MAP_SWIZZLE_32B;
		return(LM_TM_ENCODE_OK);
	}
	return(LM_TM_ENCODE_ERR_SWIZZLE);
}

// Encode one descriptor from a built plan.
//
// BF16, FP8, and FP4 are described as raw CU_TENSOR_MAP_DATA_TYPE_UINT8 bytes. There is
// no 4-bit data type, so an NVFP4 tensor is a byte tensor of half the K extent
// and the plan has already halved every K-axis figure. Passing element counts
// here instead of the plan's byte counts is the mistake this signature is
// shaped to prevent: it takes a plan, never raw dimensions.
//
// globalStrides carries rank-1 entries and excludes the innermost axis, so
// plan->global_stride_bytes[0] is the row pitch and [1] is the expert stride.
// The driver reads exactly rank-1 of them.
static int32_t LmTensorMapEncode(CUtensorMap *tensor_map, const LmTensorMapPlan *plan, void *global_address)
{
	CUtensorMapSwizzle swizzle;
	CUresult driver_status;
	int32_t status;
	if ( tensor_map == 0 || plan == 0 || global_address == 0 )
		return(LM_TM_ENCODE_ERR_NULL);
	if ( ((uintptr_t)tensor_map % 64u) != 0u )
		return(LM_TM_ENCODE_ERR_ALIGNMENT);
	if ( plan->rank < 2u || plan->rank > LM_TM_MAX_RANK )
		return(LM_TM_ENCODE_ERR_PLAN);
	status = LmTensorMapSwizzleEnum(plan->swizzle_bytes,&swizzle);
	if ( status != LM_TM_ENCODE_OK )
		return(status);
	memset(tensor_map,0,sizeof(*tensor_map));
	driver_status = cuTensorMapEncodeTiled(
		tensor_map,
		CU_TENSOR_MAP_DATA_TYPE_UINT8,
		plan->rank,
		global_address,
		plan->global_dimension,
		plan->global_stride_bytes,
		plan->box_dimension,
		plan->element_stride,
		CU_TENSOR_MAP_INTERLEAVE_NONE,
		swizzle,
		CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
		// A grouped GEMM's last M tile is ragged whenever a group's row count is
		// not a multiple of TILE_M, which at decode is almost always. Zero-fill
		// is what makes that safe without a branch or an epilogue kernel: the
		// padded rows contribute zero to the accumulator.
		CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
	if ( driver_status != CUDA_SUCCESS )
		return(LM_TM_ENCODE_ERR_DRIVER);
	return(LM_TM_ENCODE_OK);
}

// Build and encode in one step, which is how callers should use it - there is
// no legitimate reason to encode a plan that was not just built and checked.
static int32_t LmTensorMapPrepare(CUtensorMap *tensor_map, const LmTensorMapRequest *request)
{
	LmTensorMapPlan plan;
	int32_t status;
	status = LmTensorMapPlanBuild(request,&plan);
	if ( status != LM_TM_OK )
		return(status);
	return(LmTensorMapEncode(tensor_map,&plan,(void *)request->global_address));
}
