#pragma once

#include <stdint.h>

#include "sparkpipe/spark_stage_runner_model.h"

#define Q3_EPSILON 1e-6f
#define Q3_PACK_ALIGNMENT 128u

template<class G>
struct Qwen3Layout
{
	static constexpr uint32_t kQWidth = G::kHeads * G::kHeadDim;
	static constexpr uint32_t kKvWidth = G::kKvHeads * G::kHeadDim;
	static constexpr uint32_t kQkvWidth = kQWidth + 2u * kKvWidth;
	static_assert(G::kHeads % G::kKvHeads == 0u && G::kHeadDim % 2u == 0u, "query heads group evenly over the key heads and a head is whole rotary pairs");
};

template<class G, class B>
struct Qwen3Schedule
{
	using L = Qwen3Layout<G>;
	using State = typename B::State;
	using Frame = typename B::Frame;

	static SparkStatus Begin(State *state, const SparkStageRunnerStep *step, void *queue, Frame *frame)
	{
		return B::Begin(state, step, queue, frame);
	}

	static SparkStatus Layer(State *state, const Frame &frame, uint32_t index)
	{
		const typename B::Layer &layer = B::LayerOf(state, index);
		SparkStatus status = index == 0u ? B::RmsNorm(state, frame, layer.input_norm) : B::AddRmsNorm(state, frame, layer.input_norm);
		if ( status == SPARK_STATUS_OK )
			status = B::Attention(state, frame, layer);
		if ( status == SPARK_STATUS_OK )
			status = B::AllReduce(state, frame);
		if ( status != SPARK_STATUS_OK )
			return status;
		status = B::AddRmsNorm(state, frame, layer.post_norm);
		if ( status == SPARK_STATUS_OK )
			status = B::Ffn(state, frame, layer);
		if ( status == SPARK_STATUS_OK )
			status = B::AllReduce(state, frame);
		if ( status == SPARK_STATUS_OK )
			status = B::LayerDone(state, frame, index);
		return status;
	}

	static SparkStatus End(State *state, const Frame &frame, SparkStatus status)
	{
		if ( status != SPARK_STATUS_OK )
			return status;
		status = B::AddRows(state, frame);
		if ( status == SPARK_STATUS_OK )
			status = B::Check(state, frame);
		return status;
	}

	static SparkStatus Step(State *state, const SparkStageRunnerStep *step, void *queue)
	{
		Frame frame;
		SparkStatus status = Begin(state, step, queue, &frame);
		if ( status != SPARK_STATUS_OK )
			return status;
		for ( uint32_t index = 0u; index < G::kLayers && status == SPARK_STATUS_OK; ++index )
			status = Layer(state, frame, index);
		return End(state, frame, status);
	}
};
