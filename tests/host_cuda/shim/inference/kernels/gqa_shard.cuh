#pragma once

#include "inference/kernels/kv_shard.cuh"

static inline uint32_t LmGqaShardWidth(uint32_t heads, uint32_t value_dim, uint32_t degree)
{
	return(heads * value_dim / degree);
}

static inline uint32_t LmGqaShardHeadSlots(uint32_t heads, uint32_t value_dim, uint32_t degree)
{
	return((LmGqaShardWidth(heads,value_dim,degree) + value_dim - 1u) / value_dim + 1u);
}

static inline uint32_t LmGqaShardRecordFloats(uint32_t heads, uint32_t value_dim, uint32_t degree)
{
	return(LmGqaShardWidth(heads,value_dim,degree) + 2u * LmGqaShardHeadSlots(heads,value_dim,degree));
}

template<class Geometry, class Pages, uint32_t KV_HEADS, uint32_t HEAD_DIM, uint32_t VALUE_DIM>
static inline cudaError_t LmGqaShardPartialLaunch(Pages, const uint16_t *, uint32_t, const uint32_t *, const uint32_t *, const uint32_t *, float, float *,
	uint64_t, uint32_t, cudaStream_t, const uint32_t * = 0, uint32_t = 0u, uint32_t = 1u, uint32_t = 0u)
{
	return(cudaErrorNotSupported);
}

template<uint32_t VALUE_DIM>
static inline cudaError_t LmGqaShardMergeLaunch(const float *, uint64_t, uint32_t, uint32_t, uint32_t, uint16_t *, uint32_t, cudaStream_t, const float * = 0)
{
	return(cudaErrorNotSupported);
}
