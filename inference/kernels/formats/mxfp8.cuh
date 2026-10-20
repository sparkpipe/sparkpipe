#pragma once


#include "inference/kernels/formats/fp8.cuh"

#define LM_MXFP8_GROUP 32u

struct LmMxfp8 : LmFp8
{
	static constexpr uint32_t kScaleGroup = LM_MXFP8_GROUP;
};
