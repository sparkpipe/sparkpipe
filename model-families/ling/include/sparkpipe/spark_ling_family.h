#pragma once

#include <stdint.h>

#include "sparkpipe/spark_stage_runner_model.h"

#define LING_ATTENTION_MLA 0u
#define LING_ATTENTION_GQA 1u
#define LG_EPSILON 1e-6f
#define LG_PACK_ALIGNMENT 128u

struct LingDefaults
{
	static constexpr uint32_t kKdaRank = 0u;
	static constexpr uint32_t kMlaHeads = 0u;
	static constexpr uint32_t kNope = 0u;
	static constexpr uint32_t kRope = 0u;
	static constexpr uint32_t kValueDim = 0u;
	static constexpr uint32_t kLatent = 0u;
	static constexpr float kRopeTheta = 0.0f;
	static constexpr uint32_t kGqaHeads = 0u;
	static constexpr uint32_t kGqaKvHeads = 0u;
	static constexpr uint32_t kGqaHeadDim = 0u;
	static constexpr uint32_t kVerifyRows = 0u;
	static float RoutedLimit(uint32_t layer) { (void)layer; return 0.0f; }
	static float SharedLimit(uint32_t layer) { (void)layer; return 0.0f; }
};

template<class G>
struct LingLayout
{
	static constexpr bool Attention(uint32_t index)
	{
		return index % G::kPeriod == G::kPhase;
	}
	static constexpr bool kGqa = G::kAttention == LING_ATTENTION_GQA;
	static constexpr uint32_t kAttnLayers = G::kLayers / G::kPeriod;
	static constexpr uint32_t kKdaLayers = G::kLayers - kAttnLayers;
	static constexpr uint32_t kKdaRankHeads = G::kKdaHeads / G::kTpDegree;
	static constexpr uint32_t kKdaWidth = kKdaRankHeads * G::kKdaDim;
	static constexpr uint32_t kKdaChannels = 3u * kKdaWidth;
	static constexpr uint32_t kMlaRankHeads = G::kMlaHeads / G::kTpDegree;
	static constexpr uint32_t kQkDim = G::kNope + G::kRope;
	static constexpr uint32_t kGqaWidth = G::kGqaHeads * G::kGqaHeadDim;
	static constexpr uint32_t kGqaKvWidth = G::kGqaKvHeads * G::kGqaHeadDim;
	static constexpr uint32_t kGqaFused = kGqaWidth + 2u * kGqaKvWidth;
	static constexpr uint32_t kGqaRankFused = kGqaFused / G::kTpDegree;
	static constexpr uint32_t kGqaRankWidth = kGqaWidth / G::kTpDegree;
	static constexpr uint32_t kSlot = kGqa ? 2u * kGqaKvWidth : G::kLatent + G::kRope;
	static_assert(G::kKdaHeads % G::kTpDegree == 0u && G::kMlaHeads % G::kTpDegree == 0u && G::kExperts % G::kTpDegree == 0u,
		"heads and experts split evenly over the ranks");
};

template<class G, class B>
struct LingSchedule
{
	using L = LingLayout<G>;
	using State = typename B::State;
	using Frame = typename B::Frame;

	static SparkStatus Step(State *state, const SparkStageRunnerStep *step, void *queue)
	{
		Frame frame;
		SparkStatus status = B::Begin(state, step, queue, &frame);
		if ( status != SPARK_STATUS_OK )
			return status;
		for ( uint32_t index = 0u; index < G::kLayers && status == SPARK_STATUS_OK; ++index )
		{
			const typename B::Layer &layer = B::LayerOf(state, index);
			status = index == 0u ? B::RmsNorm(state, frame, layer.input_norm) : B::AddRmsNorm(state, frame, layer.input_norm);
			if ( status == SPARK_STATUS_OK )
				status = L::Attention(index) ? B::Attention(state, frame, layer) : B::Kda(state, frame, layer);
			if ( status == SPARK_STATUS_OK )
				status = B::AllReduce(state, frame);
			if ( status != SPARK_STATUS_OK )
				break;
			status = B::AddRmsNorm(state, frame, layer.post_norm);
			if ( status == SPARK_STATUS_OK )
				status = index < G::kDenseLayers ? B::Dense(state, frame, layer) : B::Experts(state, frame, layer, index);
			if ( status == SPARK_STATUS_OK )
				status = B::AllReduce(state, frame);
			if ( status == SPARK_STATUS_OK )
				status = B::LayerDone(state, frame, index);
		}
		if ( status != SPARK_STATUS_OK )
			return status;
		status = B::AddRows(state, frame);
		if ( status == SPARK_STATUS_OK )
			status = B::Check(state, frame);
		return status;
	}
};
