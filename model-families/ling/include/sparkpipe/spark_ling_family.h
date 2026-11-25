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
	static constexpr uint32_t kQLora = 0u;
	static constexpr float kRopeTheta = 0.0f;
	static constexpr uint32_t kGqaHeads = 0u;
	static constexpr uint32_t kGqaKvHeads = 0u;
	static constexpr uint32_t kGqaHeadDim = 0u;
	static constexpr uint32_t kVerifyRows = 0u;
	static float RoutedLimit(uint32_t layer) { (void)layer; return 0.0f; }
	static float SharedLimit(uint32_t layer) { (void)layer; return 0.0f; }
};

template<uint32_t TP>
struct Ling31FlashGeometry : LingDefaults
{
	static constexpr uint32_t kTpDegree = TP;
	static constexpr uint32_t kHidden = 4096u;
	static constexpr uint32_t kLayers = 72u;
	static constexpr uint32_t kVocab = 157184u;
	static constexpr uint32_t kPeriod = 8u;
	static constexpr uint32_t kPhase = 7u;
	static constexpr uint32_t kDenseLayers = 4u;
	static constexpr uint32_t kKdaHeads = 64u;
	static constexpr uint32_t kKdaDim = 128u;
	static constexpr uint32_t kConv = 4u;
	static constexpr float kKdaLowerBound = -5.0f;
	static constexpr uint32_t kMlaHeads = 64u;
	static constexpr uint32_t kNope = 128u;
	static constexpr uint32_t kRope = 64u;
	static constexpr uint32_t kValueDim = 128u;
	static constexpr uint32_t kLatent = 512u;
	static constexpr uint32_t kQLora = 1536u;
	static constexpr float kRopeTheta = 6000000.0f;
	static constexpr uint32_t kExperts = 512u;
	static constexpr uint32_t kLocalExperts = kExperts / kTpDegree;
	static constexpr uint32_t kTopK = 8u;
	static constexpr uint32_t kGroups = 8u;
	static constexpr uint32_t kTopkGroups = 4u;
	static constexpr uint32_t kMoeInter = 1280u;
	static constexpr uint32_t kSharedInter = 1280u;
	static constexpr uint32_t kSharedRows = (kSharedInter / TP + 63u) / 64u * 64u;
	static constexpr uint32_t kVerifyRows = 256u;
	static constexpr float kRouteScale = 2.5f;
	static constexpr uint32_t kPageSlots = 64u;
	static float RoutedLimit(uint32_t layer) { return layer >= 65u ? 10.0f : 0.0f; }
	static float SharedLimit(uint32_t layer) { return layer >= 64u ? 10.0f : 0.0f; }
	static const char *Tag(void) { return "sparkpipe_ling31"; }
	static const char *Name(void) { return "ling-3.1-flash"; }
	static const char *ReportTag(void) { return "LING31-STATE"; }
	static constexpr uint32_t kAttention = LING_ATTENTION_MLA;
	static constexpr float kNormEpsilon = LG_EPSILON;
	static constexpr bool kNormRoundFirst = true;
	static constexpr bool kKdaDoubledBeta = false;
	static constexpr bool kKdaConvRoundedSum = false;
	static constexpr bool kKdaNormRoundOnce = true;
	static constexpr bool kFfnSiluRounded = true;
	static constexpr bool kExpertWeightAfterDown = true;
	static constexpr bool kTpPartialsF32 = false;
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

	static SparkStatus Begin(State *state, const SparkStageRunnerStep *step, void *queue, Frame *frame)
	{
		return B::Begin(state, step, queue, frame);
	}

	static SparkStatus Layer(State *state, const Frame &frame, uint32_t index)
	{
		const typename B::Layer &layer = B::LayerOf(state, index);
		SparkStatus status = index == 0u ? B::RmsNorm(state, frame, layer.input_norm) : B::AddRmsNorm(state, frame, layer.input_norm);
		if ( status == SPARK_STATUS_OK )
			status = L::Attention(index) ? B::Attention(state, frame, layer) : B::Kda(state, frame, layer);
		if ( status == SPARK_STATUS_OK )
			status = B::AllReduce(state, frame);
		if ( status != SPARK_STATUS_OK )
			return status;
		status = B::AddRmsNorm(state, frame, layer.post_norm);
		if ( status == SPARK_STATUS_OK )
			status = index < G::kDenseLayers ? B::Dense(state, frame, layer) : B::Experts(state, frame, layer, index);
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
