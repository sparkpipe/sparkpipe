#pragma once
#include <stdint.h>
typedef enum { CUDA_SUCCESS=0, CUDA_ERROR_INVALID_VALUE=1 } CUresult;
typedef struct { uint64_t opaque[16]; } CUtensorMap;
typedef enum { CU_TENSOR_MAP_DATA_TYPE_UINT8=0 } CUtensorMapDataType;
typedef enum { CU_TENSOR_MAP_INTERLEAVE_NONE=0 } CUtensorMapInterleave;
typedef enum { CU_TENSOR_MAP_SWIZZLE_NONE=0, CU_TENSOR_MAP_SWIZZLE_32B, CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_SWIZZLE_128B } CUtensorMapSwizzle;
typedef enum { CU_TENSOR_MAP_L2_PROMOTION_NONE=0, CU_TENSOR_MAP_L2_PROMOTION_L2_128B } CUtensorMapL2promotion;
typedef enum { CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE=0 } CUtensorMapFloatOOBfill;
CUresult cuTensorMapEncodeTiled(CUtensorMap*,CUtensorMapDataType,uint32_t,void*,
  const uint64_t*,const uint64_t*,const uint32_t*,const uint32_t*,
  CUtensorMapInterleave,CUtensorMapSwizzle,CUtensorMapL2promotion,CUtensorMapFloatOOBfill);
