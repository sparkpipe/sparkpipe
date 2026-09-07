#pragma once


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
		CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
	if ( driver_status != CUDA_SUCCESS )
		return(LM_TM_ENCODE_ERR_DRIVER);
	return(LM_TM_ENCODE_OK);
}

static int32_t LmTensorMapPrepare(CUtensorMap *tensor_map, const LmTensorMapRequest *request)
{
	LmTensorMapPlan plan;
	int32_t status;
	status = LmTensorMapPlanBuild(request,&plan);
	if ( status != LM_TM_OK )
		return(status);
	return(LmTensorMapEncode(tensor_map,&plan,(void *)request->global_address));
}
