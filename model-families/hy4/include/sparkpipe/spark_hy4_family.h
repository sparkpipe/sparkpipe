#pragma once

#include <stdint.h>

#include "sparkpipe/spark_stage_runner_model.h"

#define HY_PACK_ALIGNMENT 128u

enum HyHyperPart : uint32_t
{
	HY_ATTENTION_PART = 0u,
	HY_FFN_PART = 1u
};

template<class G>
struct HyLayout
{
	static constexpr uint32_t kRankHeads = G::kHeads / G::kTpDegree;
	static constexpr uint32_t kQkDim = G::kNope + G::kRope;
	static constexpr uint32_t kSlot = G::kLatent + G::kRope;
	static constexpr uint32_t kResidual = G::kStreams * G::kHidden;
	static constexpr uint32_t kIndexWidth = G::kIndexHeads * G::kIndexDim;
	static constexpr uint32_t kCandidateSplit = (G::kIndexTopk * 4u + G::kHidden - 1u) / G::kHidden;
	static_assert(G::kHeads % G::kTpDegree == 0u && G::kExperts % G::kTpDegree == 0u, "heads and experts split evenly over the ranks");
	static_assert(G::kIndexTopk % kCandidateSplit == 0u, "the candidate rows split evenly into collective rows");
};

template<class G, class B>
struct HySchedule
{
	using L = HyLayout<G>;
	using State = typename B::State;
	using Frame = typename B::Frame;

	static SparkStatus Step(State *state, const SparkStageRunnerStep *step, void *queue)
	{
		Frame frame;
		SparkStatus status = B::Begin(state, step, queue, &frame);
		if ( status != SPARK_STATUS_OK )
			return status;
		status = B::HyperBegin(state, frame);
		for ( uint32_t index = 0u; index < G::kLayers && status == SPARK_STATUS_OK; ++index )
		{
			const typename B::Layer &layer = B::LayerOf(state, index);
			status = B::HyperNorm(state, frame, layer, HY_ATTENTION_PART);
			if ( status == SPARK_STATUS_OK )
				status = B::Attention(state, frame, layer, index);
			if ( status == SPARK_STATUS_OK )
				status = B::AllReduce(state, frame);
			if ( status != SPARK_STATUS_OK )
				break;
			status = B::HyperPost(state, frame);
			if ( status == SPARK_STATUS_OK )
				status = B::HyperNorm(state, frame, layer, HY_FFN_PART);
			if ( status == SPARK_STATUS_OK )
				status = index < G::kDenseLayers ? B::Dense(state, frame, layer) : B::Experts(state, frame, layer);
			if ( status == SPARK_STATUS_OK )
				status = B::AllReduce(state, frame);
			if ( status != SPARK_STATUS_OK )
				break;
			status = B::HyperPost(state, frame);
			if ( status == SPARK_STATUS_OK )
				status = B::Check(state, frame);
			if ( status == SPARK_STATUS_OK )
				status = B::LayerDone(state, frame, index);
		}
		if ( status != SPARK_STATUS_OK )
			return status;
		return B::HyperEnd(state, frame);
	}
};
