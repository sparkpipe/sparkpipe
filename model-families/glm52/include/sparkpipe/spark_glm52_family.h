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

	static SparkStatus Begin(State *state, const SparkStageRunnerStep *step, void *queue, Frame *frame)
	{
		return B::Begin(state, step, queue, frame);
	}

	static SparkStatus Wave(State *state, Frame &frame, uint32_t *more)
	{
		bool next = false;
		SparkStatus status = B::NextWave(state, frame, &next);
		*more = status == SPARK_STATUS_OK && next ? 1u : 0u;
		if ( *more == 0u )
			return status;
		return B::WaveBegin(state, frame);
	}

	static SparkStatus Layer(State *state, Frame &frame, uint32_t layer)
	{
		SparkStatus status = B::Attention(state, frame, layer);
		if ( status == SPARK_STATUS_OK )
			status = B::AttentionResidual(state, frame, layer);
		if ( status == SPARK_STATUS_OK )
			status = L::Routed(layer) ? B::RoutedMlp(state, frame, layer) : B::DenseMlp(state, frame, layer);
		if ( status == SPARK_STATUS_OK )
			status = B::MlpResidual(state, frame, layer);
		if ( status == SPARK_STATUS_OK )
			status = B::LayerDone(state, frame, layer);
		return status;
	}

	static SparkStatus WaveEnd(State *state, Frame &frame)
	{
		return B::WaveEnd(state, frame);
	}

	static SparkStatus End(State *state, Frame &frame, SparkStatus status)
	{
		return status == SPARK_STATUS_OK ? B::End(state, frame) : status;
	}

	static SparkStatus Step(State *state, const SparkStageRunnerStep *step, void *queue)
	{
		Frame frame;
		uint32_t more = 0u;
		SparkStatus status = Begin(state, step, queue, &frame);
		if ( status != SPARK_STATUS_OK )
			return status;
		for ( status = Wave(state, frame, &more); status == SPARK_STATUS_OK && more != 0u; status = Wave(state, frame, &more) )
		{
			for ( uint32_t layer = 0u; layer < G::kLayers && status == SPARK_STATUS_OK; layer++ )
				status = Layer(state, frame, layer);
			if ( status == SPARK_STATUS_OK )
				status = WaveEnd(state, frame);
			if ( status != SPARK_STATUS_OK )
				break;
		}
		return End(state, frame, status);
	}
};
