#pragma once

#include <stdint.h>

typedef void *CUstream;
typedef uintptr_t CUdeviceptr;
typedef int32_t CUresult;

#define CUDA_SUCCESS 0
#define CU_STREAM_WAIT_VALUE_GEQ 0x0u
#define CU_STREAM_WRITE_VALUE_DEFAULT 0x0u

CUresult cuStreamWaitValue32(CUstream stream,CUdeviceptr address,
	uint32_t value,uint32_t flags);
CUresult cuStreamWriteValue32(CUstream stream,CUdeviceptr address,
	uint32_t value,uint32_t flags);
