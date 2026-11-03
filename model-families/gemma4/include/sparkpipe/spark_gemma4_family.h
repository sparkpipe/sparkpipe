#pragma once

#include <math.h>
#include <stdint.h>

#include "sparkpipe/spark_stage_runner_model.h"

#define GEMMA_PACK_ALIGNMENT 128u

struct GemmaDefaults
{
	static constexpr bool kEmbedNorm = false;
	static constexpr bool kOutputGate = false;
	static constexpr bool kUntiedHead = false;
	static constexpr float kLogitScale = 0.0f;
	static constexpr bool kMoe = false;
	static constexpr uint32_t kExperts = 0u;
	static constexpr uint32_t kLocalExperts = 0u;
	static constexpr uint32_t kTopK = 0u;
	static constexpr uint32_t kMoeInter = 0u;
	static constexpr uint32_t kFullRopeDim = 0u;
	static constexpr float kFullRopeScale = 1.0f;
	static constexpr float kYarnFactor = 0.0f;
	static constexpr float kYarnOriginal = 0.0f;
	static constexpr float kYarnBetaFast = 0.0f;
	static constexpr float kYarnBetaSlow = 0.0f;
	static constexpr bool kHeadGate = false;
	static constexpr uint32_t kFirstMoeLayer = 0u;
	static constexpr uint32_t kEmbedAliasFirst = 0u;
	static constexpr uint32_t kEmbedAliasCount = 0u;
	static constexpr uint32_t kEmbedAliasRow = 0u;
	static constexpr bool kLayerTable = false;
};

template<class G>
struct GemmaLayout
{
	static constexpr bool Full(uint32_t index)
	{
		if constexpr ( G::kLayerTable )
			return G::LayerWindow(index) == 0u;
		else
			return index % G::kPeriod == G::kPhase;
	}
	static constexpr uint32_t Window(uint32_t index)
	{
		if constexpr ( G::kLayerTable )
			return G::LayerWindow(index);
		else
			return Full(index) ? 0u : G::kWindow;
	}
	static constexpr float SwigluLimit(uint32_t index)
	{
		if constexpr ( G::kLayerTable )
			return G::LayerSwigluLimit(index);
		else
			return INFINITY;
	}
	static constexpr uint32_t CountFull(void)
	{
		uint32_t count = 0u;
		for ( uint32_t index = 0u; index < G::kLayers; ++index )
			count += Full(index) ? 1u : 0u;
		return count;
	}
	static constexpr uint32_t kFullLayers = CountFull();
	static_assert(G::kLayerTable || kFullLayers == G::kLayers / G::kPeriod, "a periodic layout has one full layer per period");
	static constexpr uint32_t kSlidingLayers = G::kLayers - kFullLayers;
	static constexpr uint32_t kSlidingQ = G::kSlidingHeads * G::kSlidingDim;
	static constexpr uint32_t kSlidingKv = G::kSlidingKvHeads * G::kSlidingDim;
	static constexpr uint32_t kSlidingFused = kSlidingQ + 2u * kSlidingKv;
	static constexpr uint32_t kFullQ = G::kFullHeads * G::kFullDim;
	static constexpr uint32_t kFullKv = G::kFullKvHeads * G::kFullDim;
	static constexpr uint32_t kFullFused = kFullQ + (G::kKeyIsValue ? 1u : 2u) * kFullKv;
	static constexpr uint32_t kMaxFused = kSlidingFused > kFullFused ? kSlidingFused : kFullFused;
	static constexpr uint32_t kMaxQ = kSlidingQ > kFullQ ? kSlidingQ : kFullQ;
	static constexpr uint32_t kMaxKv = kSlidingKv > kFullKv ? kSlidingKv : kFullKv;
	static constexpr uint32_t kFullRopeWidth = G::kFullRopeDim != 0u ? G::kFullRopeDim : G::kFullDim;
	static constexpr uint32_t kSlidingGateRows = (G::kSlidingHeads / G::kTpDegree + 1u) & ~1u;
	static constexpr uint32_t kFullGateRows = (G::kFullHeads / G::kTpDegree + 1u) & ~1u;
	static constexpr uint32_t kMaxGateRows = kSlidingGateRows > kFullGateRows ? kSlidingGateRows : kFullGateRows;
	static_assert(!G::kHeadGate || (G::kSlidingHeads % G::kTpDegree == 0u && G::kFullHeads % G::kTpDegree == 0u),
		"a per-head gate needs whole heads on every rank");
	static_assert(kFullRopeWidth % 2u == 0u && kFullRopeWidth <= G::kFullDim, "the full-layer rope rotates an even prefix of the head");
	static_assert(G::kYarnFactor <= 1.0f || (G::kYarnOriginal > 0.0f && G::kYarnBetaFast > 0.0f && G::kYarnBetaSlow > 0.0f),
		"a geometry that turns YaRN on chooses its betas and original positions");
};

enum GemmaBranch : uint32_t
{
	GEMMA_ATTENTION_BRANCH = 0u,
	GEMMA_EXPERT_BRANCH = 1u
};

template<class G, class B>
struct GemmaSchedule
{
	using L = GemmaLayout<G>;
	using State = typename B::State;
	using Frame = typename B::Frame;

	static SparkStatus Attention(State *state, const Frame &frame, uint32_t index)
	{
		const typename B::Layer &layer = B::LayerOf(state, index);
		return L::Full(index) ? B::template Attention<true>(state, frame, layer) : B::template Attention<false>(state, frame, layer);
	}

	static SparkStatus PreNormLayers(State *state, const Frame &frame)
	{
		SparkStatus status = B::RmsNorm(state, frame, B::LayerOf(state, 0u).input_norm);
		for ( uint32_t index = 0u; index < G::kLayers && status == SPARK_STATUS_OK; ++index )
		{
			const typename B::Layer &layer = B::LayerOf(state, index);
			const bool moe = G::kMoe && index >= G::kFirstMoeLayer;
			const GemmaBranch ffn = moe ? GEMMA_EXPERT_BRANCH : GEMMA_ATTENTION_BRANCH;
			status = Attention(state, frame, index);
			if ( status == SPARK_STATUS_OK )
				status = B::AllReduce(state, frame, GEMMA_ATTENTION_BRANCH, 1u);
			if ( status != SPARK_STATUS_OK )
				break;
			status = B::AddRmsNorm(state, frame, GEMMA_ATTENTION_BRANCH, layer.pre_ff_norm);
			if ( status == SPARK_STATUS_OK )
				status = B::Mlp(state, frame, layer);
			if constexpr ( G::kMoe && G::kSigmoidRouter )
				if ( status == SPARK_STATUS_OK && moe )
					status = B::SigmoidExperts(state, frame, layer);
			if ( status == SPARK_STATUS_OK )
				status = B::AllReduce(state, frame, ffn, 1u);
			if ( status != SPARK_STATUS_OK )
				break;
			if ( index + 1u < G::kLayers )
				status = B::AddRmsNorm(state, frame, ffn, B::LayerOf(state, index + 1u).input_norm);
			else
				status = B::AddRows(state, frame, ffn);
			if ( status == SPARK_STATUS_OK )
				status = B::Check(state, frame);
			if ( status == SPARK_STATUS_OK )
				status = B::LayerDone(state, frame, index);
		}
		return status;
	}

	static SparkStatus Step(State *state, const SparkStageRunnerStep *step, void *queue)
	{
		Frame frame;
		SparkStatus status = B::Begin(state, step, queue, &frame);
		if ( status != SPARK_STATUS_OK )
			return status;
		if constexpr ( G::kEmbedNorm )
			status = B::EmbedNorm(state, frame);
		else if constexpr ( G::kEmbedScale != 1.0f )
			status = B::EmbedScale(state, frame, G::kEmbedScale);
		if ( status != SPARK_STATUS_OK )
			return status;
		if constexpr ( G::kPreNorm )
			return PreNormLayers(state, frame);
		for ( uint32_t index = 0u; index < G::kLayers && status == SPARK_STATUS_OK; ++index )
		{
			const typename B::Layer &layer = B::LayerOf(state, index);
			status = B::RmsNorm(state, frame, layer.input_norm);
			if ( status == SPARK_STATUS_OK )
				status = Attention(state, frame, index);
			if ( status == SPARK_STATUS_OK )
				status = B::AllReduce(state, frame, GEMMA_ATTENTION_BRANCH, 1u);
			if ( status != SPARK_STATUS_OK )
				break;
			status = B::NormAdd(state, frame, layer.post_attn_norm);
			if ( status == SPARK_STATUS_OK )
				status = B::RmsNorm(state, frame, layer.pre_ff_norm);
			if ( status == SPARK_STATUS_OK )
				status = B::Mlp(state, frame, layer);
			if constexpr ( G::kMoe )
				if ( status == SPARK_STATUS_OK )
					status = B::Experts(state, frame, layer);
			if ( status == SPARK_STATUS_OK )
				status = B::AllReduce(state, frame, GEMMA_ATTENTION_BRANCH, G::kMoe ? 2u : 1u);
			if ( status != SPARK_STATUS_OK )
				break;
			if constexpr ( G::kMoe )
				status = B::BranchNormAdd(state, frame, layer.post_ff_norm_1, layer.post_ff_norm_2, layer.post_ff_norm, layer.layer_scalar);
			else if constexpr ( G::kLayerScalar )
				status = B::NormAddScaled(state, frame, layer.post_ff_norm, layer.layer_scalar);
			else
				status = B::NormAdd(state, frame, layer.post_ff_norm);
			if ( status == SPARK_STATUS_OK )
				status = B::Check(state, frame);
			if ( status == SPARK_STATUS_OK )
				status = B::LayerDone(state, frame, index);
		}
		return status;
	}
};
