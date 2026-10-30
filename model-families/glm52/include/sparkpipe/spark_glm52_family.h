#pragma once

#include <stdint.h>

#include "sparkpipe/spark_stage_runner_model.h"

template<class G>
struct GlmLayout
{
	static constexpr bool Routed(uint32_t layer)
	{
		return layer >= G::kFirstRouted;
	}
};

template<class G, class B>
struct GlmSchedule
{
	using L = GlmLayout<G>;
	using State = typename B::State;
	using Frame = typename B::Frame;

	static SparkStatus Layers(State *state, Frame &frame)
	{
		SparkStatus status = SPARK_STATUS_OK;
		for ( uint32_t layer = 0u; layer < G::kLayers && status == SPARK_STATUS_OK; layer++ )
		{
			status = B::Attention(state, frame, layer);
			if ( status == SPARK_STATUS_OK )
				status = B::AttentionResidual(state, frame, layer);
			if ( status == SPARK_STATUS_OK )
				status = L::Routed(layer) ? B::RoutedMlp(state, frame, layer) : B::DenseMlp(state, frame, layer);
			if ( status == SPARK_STATUS_OK )
				status = B::MlpResidual(state, frame, layer);
			if ( status == SPARK_STATUS_OK )
				status = B::LayerDone(state, frame, layer);
		}
		return status;
	}

	static SparkStatus Step(State *state, const SparkStageRunnerStep *step, void *queue)
	{
		Frame frame;
		bool more = false;
		SparkStatus status = B::Begin(state, step, queue, &frame);
		if ( status == SPARK_STATUS_OK )
			status = B::NextWave(state, frame, &more);
		while ( status == SPARK_STATUS_OK && more )
		{
			status = B::WaveBegin(state, frame);
			if ( status == SPARK_STATUS_OK )
				status = Layers(state, frame);
			if ( status == SPARK_STATUS_OK )
				status = B::WaveEnd(state, frame);
			if ( status == SPARK_STATUS_OK )
				status = B::NextWave(state, frame, &more);
		}
		if ( status != SPARK_STATUS_OK )
			return status;
		return B::End(state, frame);
	}
};
