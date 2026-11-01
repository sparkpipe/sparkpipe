#pragma once

#include <stdint.h>

#include "sparkpipe/spark_stage_runner_model.h"

#define QH_KEY_DIM 128u
#define QH_VALUE_DIM 128u
#define QH_CONV 4u
#define QH_EPSILON 1e-6f
#define QH_PACK_ALIGNMENT 128u

struct QwenHybridDefaults
{
	static constexpr uint32_t kStreams = 1u;
	static constexpr uint32_t kHcLowRank = 0u;
	static constexpr uint32_t kPleLayer = 0xffffffffu;
	static constexpr uint32_t kPleOrders = 0u;
	static constexpr uint32_t kPleHeadsPerOrder = 0u;
	static constexpr uint32_t kPleHeadDim = 0u;
	static constexpr uint32_t kPleConv = 1u;
	static constexpr uint32_t kPleDilation = 1u;
	static constexpr uint32_t kPleEos = 0u;
	static constexpr uint32_t kIndexHeads = 0u;
	static constexpr uint32_t kIndexDim = 0u;
	static constexpr uint32_t kIndexBudget = 0u;
	static constexpr uint32_t kIndexRatio = 1u;
};

template<class G>
struct QwenHybridLayout
{
	static constexpr bool Attention(uint32_t index)
	{
		return index % G::kPeriod == G::kPhase;
	}
	static constexpr uint32_t kGdnLayers = G::kLayers - G::kLayers / G::kPeriod;
	static constexpr uint32_t kAttnLayers = G::kLayers / G::kPeriod;
	static constexpr uint32_t kQWidth = G::kHeads * G::kHeadDim;
	static constexpr uint32_t kKvWidth = G::kKvHeads * G::kHeadDim;
	static constexpr uint32_t kQkvWidth = 2u * kQWidth + 2u * kKvWidth;
	static constexpr bool kHyper = G::kStreams > 1u;
	static constexpr uint32_t kResidual = G::kStreams * G::kHidden;
	static constexpr bool kPle = G::kPleLayer < G::kLayers;
	static constexpr uint32_t kPleColumns = G::kPleOrders * G::kPleHeadsPerOrder;
	static constexpr uint32_t kPleSpan = (G::kPleConv - 1u) * G::kPleDilation + 1u;
	static constexpr uint32_t kPleHistory = 4u;
	static constexpr bool kIndexer = G::kIndexHeads > 0u;
	static constexpr uint32_t kIndexWidth = (G::kIndexHeads + 1u) * G::kIndexDim;
	static constexpr uint32_t kIndexBlocks = G::kIndexBudget / G::kIndexRatio;
	static constexpr uint32_t kIndexTail = G::kIndexRatio - 1u;
};

template<class G, class B>
struct QwenHybridSchedule
{
	using L = QwenHybridLayout<G>;
	using State = typename B::State;
	using Frame = typename B::Frame;

	static SparkStatus Mixer(State *state, const Frame &frame, uint32_t index)
	{
		const typename B::Layer &layer = B::LayerOf(state, index);
		return L::Attention(index) ? B::Attention(state, frame, layer) : B::Gdn(state, frame, layer);
	}

	static SparkStatus HyperLayers(State *state, const Frame &frame)
	{
		SparkStatus status = B::HyperBegin(state, frame);
		for ( uint32_t index = 0u; index < G::kLayers && status == SPARK_STATUS_OK; ++index )
		{
			const typename B::Layer &layer = B::LayerOf(state, index);
			if constexpr ( L::kPle )
				if ( index == G::kPleLayer )
					status = B::Ple(state, frame);
			for ( uint32_t part = 0u; part < 2u && status == SPARK_STATUS_OK; ++part )
			{
				status = B::HyperPre(state, frame, layer, part);
				if ( status == SPARK_STATUS_OK )
					status = part == 1u ? B::Ffn(state, frame, layer) : Mixer(state, frame, index);
				if ( status == SPARK_STATUS_OK )
					status = B::AllReduce(state, frame);
				if ( status == SPARK_STATUS_OK )
					status = B::HyperPost(state, frame);
			}
			if ( status == SPARK_STATUS_OK )
				status = B::LayerDone(state, frame, index);
		}
		if ( status != SPARK_STATUS_OK )
			return status;
		return B::HyperEnd(state, frame);
	}

	static SparkStatus PlainLayers(State *state, const Frame &frame)
	{
		SparkStatus status = SPARK_STATUS_OK;
		for ( uint32_t index = 0u; index < G::kLayers && status == SPARK_STATUS_OK; ++index )
		{
			const typename B::Layer &layer = B::LayerOf(state, index);
			status = index == 0u ? B::RmsNorm(state, frame, layer.input_norm) : B::AddRmsNorm(state, frame, layer.input_norm);
			if ( status == SPARK_STATUS_OK )
				status = Mixer(state, frame, index);
			if ( status == SPARK_STATUS_OK )
				status = B::AllReduce(state, frame);
			if ( status != SPARK_STATUS_OK )
				break;
			status = B::AddRmsNorm(state, frame, layer.post_norm);
			if ( status == SPARK_STATUS_OK )
				status = B::Ffn(state, frame, layer);
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

	static SparkStatus Step(State *state, const SparkStageRunnerStep *step, void *queue)
	{
		Frame frame;
		const SparkStatus status = B::Begin(state, step, queue, &frame);
		if ( status != SPARK_STATUS_OK )
			return status;
		if constexpr ( L::kHyper )
			return HyperLayers(state, frame);
		else
			return PlainLayers(state, frame);
	}
};
