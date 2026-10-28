#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm5_next_model.h"
#include "sparkpipe/spark_stage_runner_model.h"

enum Glm5NextPart : uint32_t
{
	GLM5_NEXT_ATTENTION_PART = 0u,
	GLM5_NEXT_FFN_PART = 1u
};

struct Glm5NextLayout
{
	static constexpr bool Kda(uint32_t layer)
	{
		return SPARK_GLM5_NEXT_MODEL_LAYER_IS_KDA(layer);
	}
	static constexpr bool Routed(uint32_t layer)
	{
		return layer >= SPARK_GLM5_NEXT_MODEL_FIRST_ROUTED_LAYER;
	}
	static constexpr uint32_t kLayers = SPARK_GLM5_NEXT_MODEL_LAYER_COUNT;
};

template<class G, class B>
struct Glm5NextSchedule
{
	using L = Glm5NextLayout;
	using State = typename B::State;
	using Frame = typename B::Frame;

	static SparkStatus Step(State *state, const SparkStageRunnerStep *step, void *queue)
	{
		Frame frame;
		SparkStatus status = B::Begin(state, step, queue, &frame);
		if ( status == SPARK_STATUS_OK )
			status = B::HyperBegin(state, frame);
		for ( uint32_t layer = 0u; layer < L::kLayers && status == SPARK_STATUS_OK; ++layer )
		{
			B::BindLayer(state, frame, layer);
			status = B::HyperNorm(state, frame, layer, GLM5_NEXT_ATTENTION_PART);
			if ( status == SPARK_STATUS_OK )
				status = L::Kda(layer) ? B::Kda(state, frame, layer) : B::Attention(state, frame, layer);
			if ( status == SPARK_STATUS_OK )
				status = B::AllReduce(state, frame);
			if ( status == SPARK_STATUS_OK )
				status = B::HyperPost(state, frame, layer, GLM5_NEXT_ATTENTION_PART);
			if ( status == SPARK_STATUS_OK )
				status = B::HyperNorm(state, frame, layer, GLM5_NEXT_FFN_PART);
			if ( status == SPARK_STATUS_OK )
				status = L::Routed(layer) ? B::RoutedMlp(state, frame, layer) : B::DenseMlp(state, frame, layer);
			if ( status == SPARK_STATUS_OK )
				status = B::AllReduce(state, frame);
			if ( status == SPARK_STATUS_OK )
				status = B::HyperPost(state, frame, layer, GLM5_NEXT_FFN_PART);
		}
		if ( status != SPARK_STATUS_OK )
			return status;
		return B::HyperEnd(state, frame);
	}
};
