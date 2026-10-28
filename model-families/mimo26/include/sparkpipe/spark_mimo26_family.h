#pragma once

#include <stdint.h>

#include "sparkpipe/spark_stage_runner_model.h"

template<class G>
struct MimoLayout
{
	static bool Full(uint32_t index)
	{
		return G::FullLayer(index);
	}
	static constexpr uint32_t kRankHeads = G::kHeads / G::kTpDegree;
	static constexpr uint32_t kQuery = G::kHeads * G::kHeadDim;
	static constexpr uint32_t kSlidingKey = G::kSlidingKvHeads * G::kHeadDim;
	static constexpr uint32_t kSlidingValue = G::kSlidingKvHeads * G::kValueDim;
	static constexpr uint32_t kSlidingFused = kQuery + kSlidingKey + kSlidingValue;
	static constexpr uint32_t kFullKey = G::kFullKvHeads * G::kHeadDim;
	static constexpr uint32_t kFullValue = G::kFullKvHeads * G::kValueDim;
	static constexpr uint32_t kFullFused = kQuery + kFullKey + kFullValue;
	static constexpr uint32_t kMaxFused = kSlidingFused > kFullFused ? kSlidingFused : kFullFused;
	static constexpr uint32_t kMaxKey = kSlidingKey > kFullKey ? kSlidingKey : kFullKey;
	static constexpr uint32_t kMaxValue = kSlidingValue > kFullValue ? kSlidingValue : kFullValue;
	static constexpr uint32_t kRankWidth = kRankHeads * G::kValueDim;
	static constexpr uint32_t kSlidingLayers = G::kLayers - G::kFullLayers;
	static constexpr uint32_t kDenseRows = G::kDenseInter / G::kTpDegree;
	static_assert(G::kHeads % G::kTpDegree == 0u && G::kLocalExperts * G::kTpDegree == G::kExperts, "heads and experts split evenly over the ranks");
	static_assert(kSlidingFused % G::kTpDegree == 0u && kFullFused % G::kTpDegree == 0u, "the fused q|k|v rows split evenly over the ranks");
	static_assert(G::kRopeDim % 2u == 0u && G::kRopeDim <= G::kHeadDim, "the rotary part is whole pairs inside a head");
};

template<class G, class B>
struct MimoSchedule
{
	using L = MimoLayout<G>;
	using State = typename B::State;
	using Frame = typename B::Frame;

	static SparkStatus Step(State *state, const SparkStageRunnerStep *step, void *queue)
	{
		Frame frame;
		SparkStatus status = B::Begin(state, step, queue, &frame);
		for ( uint32_t index = 0u; index < G::kLayers && status == SPARK_STATUS_OK; ++index )
		{
			const typename B::Layer &layer = B::LayerOf(state, index);
			status = B::RmsNorm(state, frame, layer.input_norm);
			if ( status == SPARK_STATUS_OK )
				status = L::Full(index) ? B::template Attention<true>(state, frame, layer) : B::template Attention<false>(state, frame, layer);
			if ( status == SPARK_STATUS_OK )
				status = B::AllReduce(state, frame);
			if ( status == SPARK_STATUS_OK )
				status = B::AddResidual(state, frame);
			if ( status != SPARK_STATUS_OK )
				break;
			status = B::RmsNorm(state, frame, layer.post_attn_norm);
			if ( status == SPARK_STATUS_OK )
				status = index < G::kDenseLayers ? B::Dense(state, frame, layer) : B::Experts(state, frame, layer);
			if ( status == SPARK_STATUS_OK )
				status = B::AllReduce(state, frame);
			if ( status == SPARK_STATUS_OK )
				status = B::AddResidual(state, frame);
			if ( status == SPARK_STATUS_OK )
				status = B::LayerDone(state, frame, index);
		}
		return status;
	}
};
