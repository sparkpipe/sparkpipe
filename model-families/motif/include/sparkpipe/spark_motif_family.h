#pragma once

#include <stdint.h>

#include "sparkpipe/spark_motif_model.h"
#include "sparkpipe/spark_stage_runner_model.h"

enum MotifPart : uint32_t
{
	MOTIF_ATTENTION_PART = 0u,
	MOTIF_FFN_PART = 1u
};

struct MotifLayout
{
	static constexpr bool Window(uint32_t layer)
	{
		return SPARK_MOTIF_MODEL_LAYER_IS_WINDOW(layer);
	}
	static constexpr bool Routed(uint32_t layer)
	{
		return layer >= SPARK_MOTIF_MODEL_FIRST_ROUTED_LAYER;
	}
	static constexpr uint32_t kLayers = SPARK_MOTIF_MODEL_LAYER_COUNT;
};

template<class G, class B>
struct MotifSchedule
{
	using L = MotifLayout;
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
			status = B::HyperNorm(state, frame, layer, MOTIF_ATTENTION_PART);
			if ( status == SPARK_STATUS_OK )
				status = L::Window(layer) ? B::WindowAttention(state, frame, layer) : B::FullAttention(state, frame, layer);
			if ( status == SPARK_STATUS_OK )
				status = B::AllReduce(state, frame);
			if ( status == SPARK_STATUS_OK )
				status = B::HyperPost(state, frame, layer, MOTIF_ATTENTION_PART);
			if ( status == SPARK_STATUS_OK )
				status = B::HyperNorm(state, frame, layer, MOTIF_FFN_PART);
			if ( status == SPARK_STATUS_OK )
				status = L::Routed(layer) ? B::RoutedMlp(state, frame, layer) : B::DenseMlp(state, frame, layer);
			if ( status == SPARK_STATUS_OK )
				status = B::AllReduce(state, frame);
			if ( status == SPARK_STATUS_OK )
				status = B::HyperPost(state, frame, layer, MOTIF_FFN_PART);
		}
		if ( status != SPARK_STATUS_OK )
			return status;
		return B::HyperEnd(state, frame);
	}
};
