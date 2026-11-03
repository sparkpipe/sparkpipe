#pragma once

#include <stdint.h>

#include "sparkpipe/spark_dsv41_flash_model.h"
#include "sparkpipe/spark_stage_runner_model.h"

enum D41Part : uint32_t
{
	D41_ATTENTION_PART = 0u,
	D41_FFN_PART = 1u
};

static inline uint32_t D41Ratio(uint32_t layer)
{
	return layer < 2u ? 0u : layer < 20u ? 2u : 1u;
}

static inline uint32_t D41KvSource(uint32_t layer)
{
	return layer == 2u || layer == 8u || layer == 14u || layer == 20u;
}

static inline uint32_t D41IndexSource(uint32_t layer)
{
	return D41KvSource(layer) || layer == 24u || layer == 28u || layer == 32u || layer == 36u;
}

static inline uint32_t D41Owner(uint32_t layer)
{
	return layer < 8u ? 2u : layer < 14u ? 8u : layer < 20u ? 14u : 20u;
}

static inline uint32_t D41CacheIndex(uint32_t layer)
{
	const uint32_t owner = D41Owner(layer);
	return owner == 2u ? 0u : owner == 8u ? 1u : owner == 14u ? 2u : 3u;
}

static inline int32_t D41EngramIndex(uint32_t layer)
{
	return layer == 1u ? 0 : layer == 14u ? 1 : -1;
}

template<class B>
struct D41Schedule
{
	using State = typename B::State;
	using Frame = typename B::Frame;

	static SparkStatus Layer(State *state, Frame &frame, uint32_t index)
	{
		const int32_t engram = D41EngramIndex(index);
		SparkStatus status = SPARK_STATUS_OK;
		if ( engram >= 0 )
			status = B::Engram(state, frame, (uint32_t)engram);
		if ( status == SPARK_STATUS_OK )
			status = B::HyperMix(state, frame, index, D41_ATTENTION_PART);
		if ( status != SPARK_STATUS_OK )
			return status;
		status = B::CollapseNorm(state, frame, index, D41_ATTENTION_PART);
		if ( status != SPARK_STATUS_OK )
			return status;
		status = B::Attention(state, frame, index);
		if ( status == SPARK_STATUS_OK )
			status = B::AttentionReduce(state, frame, index);
		if ( status != SPARK_STATUS_OK )
			return status;
		status = B::HyperPost(state, frame, D41_ATTENTION_PART);
		if ( status == SPARK_STATUS_OK )
			status = B::HyperMix(state, frame, index, D41_FFN_PART);
		if ( status != SPARK_STATUS_OK )
			return status;
		status = B::CollapseNorm(state, frame, index, D41_FFN_PART);
		if ( status != SPARK_STATUS_OK )
			return status;
		status = B::Moe(state, frame, index);
		if ( status != SPARK_STATUS_OK )
			return status;
		status = B::HyperPost(state, frame, D41_FFN_PART);
		if ( status == SPARK_STATUS_OK )
			status = B::Tap(state, frame, index);
		if ( status != SPARK_STATUS_OK )
			return status;
		return B::Finish(state, frame, index);
	}

	static SparkStatus Step(State *state, const SparkStageRunnerStep *step, void *queue)
	{
		Frame frame;
		SparkStatus status = B::Begin(state, step, queue, &frame);
		if ( status != SPARK_STATUS_OK )
			return status;
		for ( uint32_t index = 0u; index < SPARK_DSV41_FLASH_MODEL_LAYER_COUNT && status == SPARK_STATUS_OK; ++index )
		{
			status = B::LayerRows(state, frame, index);
			if ( status == SPARK_STATUS_OK && B::HasRows(frame) )
				status = Layer(state, frame, index);
			B::LayerEnd(state, frame, index, status);
		}
		return B::End(state, frame, status);
	}
};
