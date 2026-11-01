#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inference/runner/device_layers.h"
#include "sparkpipe/spark_ling_device_stage.h"
#include "sparkpipe/spark_ling_family.h"
#include "sparkpipe/spark_ling3_stage_model.h"
#include "sparkpipe/spark_solar_open2_tp16_stage_model.h"

#define LD_SLICES 40u

template<class P>
struct LingDevice
{
	using G = typename P::Geometry;
	using L = LingLayout<G>;

	struct Layer
	{
		SparkTensor input_norm;
		SparkTensor post_norm;
		SparkDeviceChannelDeltaWeights delta;
		SparkDeviceLatentAttentionWeights attention;
		SparkDeviceAttentionWeights gqa;
		SparkDeviceFfnWeights ffn;
		SparkDeviceFfnWeights shared;
		SparkDeviceExpertsWeights experts;
		uint32_t index;
		uint32_t recurrent_index;
		uint32_t kv_index;
	};

	struct State
	{
		SparkDeviceStage stage;
		SparkDeviceStageShape shape;
		SparkDeviceChannelDeltaShape delta_shape;
		SparkDeviceLatentAttentionShape attention_shape;
		SparkDeviceLatentAttentionRope rope;
		SparkDeviceAttentionShape gqa_shape;
		SparkDeviceAttentionRope gqa_rope;
		SparkDeviceExpertsShape experts_shape;
		Layer layers[G::kLayers];
		uint32_t ffn;
		SparkTensor normed;
		SparkTensor partial;
		SparkTensor shared_out;
		SparkDeviceChannelDeltaScratch delta;
		SparkDeviceLatentAttentionScratch attention;
		SparkDeviceAttentionScratch gqa;
		SparkDeviceFfnScratch ffn_scratch;
		SparkDeviceExpertsScratch experts;
	};

	struct Frame
	{
		SparkDeviceStepTensors t;
		SparkDeviceQueue *queue;
	};

	static uint64_t Max(uint64_t a, uint64_t b)
	{
		return a > b ? a : b;
	}

	static void Shapes(State *state)
	{
		SparkDeviceStageShape *shape = &state->shape;
		SparkDeviceChannelDeltaShape *delta = &state->delta_shape;
		SparkDeviceLatentAttentionShape *attention = &state->attention_shape;
		SparkDeviceExpertsShape *experts = &state->experts_shape;
		memset(shape, 0, sizeof(*shape));
		shape->tag = P::kTag;
		shape->hidden = G::kHidden;
		shape->vocab = G::kVocab;
		shape->layers = G::kLayers;
		shape->kv_layers = L::kAttnLayers;
		shape->kv_heads = L::kGqa ? G::kGqaKvHeads : 1u;
		shape->kv_head_dimension = L::kGqa ? G::kGqaHeadDim : (G::kLatent + G::kRope + G::kLatent) / 2u;
		shape->pack_alignment = LG_PACK_ALIGNMENT;
		shape->epsilon = G::kNormEpsilon;
		shape->projection = P::kProjection;
		memset(delta, 0, sizeof(*delta));
		delta->heads = L::kKdaRankHeads;
		delta->dimension = G::kKdaDim;
		delta->conv_taps = G::kConv;
		delta->low_rank = G::kKdaRank;
		delta->epsilon = G::kNormEpsilon;
		delta->l2_epsilon = 1e-6f;
		delta->minimum_log_decay = G::kKdaLowerBound;
		delta->beta_mode = G::kKdaDoubledBeta ? SPARK_DELTA_BETA_DOUBLED : SPARK_DELTA_BETA_SIGMOID;
		memset(attention, 0, sizeof(*attention));
		attention->heads = G::kMlaHeads;
		attention->nope = G::kNope;
		attention->rope = G::kRope;
		attention->latent = G::kLatent;
		attention->value_dimension = G::kValueDim;
		attention->rope_theta = G::kRopeTheta;
		attention->epsilon = G::kNormEpsilon;
		attention->scale = 1.0f / sqrtf((float)L::kQkDim);
		attention->head_gate = 1u;
		memset(&state->gqa_shape, 0, sizeof(state->gqa_shape));
		state->gqa_shape.heads = G::kGqaHeads;
		state->gqa_shape.kv_heads = G::kGqaKvHeads;
		state->gqa_shape.head_dimension = G::kGqaHeadDim;
		state->gqa_shape.epsilon = G::kNormEpsilon;
		state->gqa_shape.scale = L::kGqa ? 1.0f / sqrtf((float)G::kGqaHeadDim) : 0.0f;
		state->gqa_shape.qk_plain = 1u;
		state->gqa_shape.output_gate = SPARK_ATTENTION_GATE_SIGMOID;
		state->gqa_shape.gate_projection = 1u;
		state->gqa_shape.tensor_prefix = "gqa";
		memset(experts, 0, sizeof(*experts));
		experts->experts = G::kExperts;
		experts->top_k = G::kTopK;
		experts->intermediate = G::kMoeInter;
		experts->activation = SPARK_GATED_SILU_CLAMP;
		experts->epsilon = G::kNormEpsilon;
		experts->routed_scale = G::kRouteScale;
		experts->groups = G::kGroups;
		experts->topk_groups = G::kTopkGroups;
		experts->router = "router";
		experts->tensor_parallel = G::kLocalExperts == G::kExperts ? 1u : 0u;
	}

	static uint32_t ScratchSlices(State *state, SparkDeviceStageSlice *slices)
	{
		const uint64_t rows = state->stage.max_rows, hidden = G::kHidden, width = L::kKdaWidth, heads = L::kMlaRankHeads;
		const uint64_t slot = G::kLatent + G::kRope, ffn = Max(state->ffn, G::kSharedRows);
		SparkTensor *experts[] = { &state->experts.routed, &state->experts.logits, &state->experts.experts, &state->experts.weights,
			&state->experts.normed, &state->experts.gate_up, &state->experts.act, &state->experts.out, &state->experts.sorted };
		const uint64_t fused = SparkDeviceAttentionFusedWidth(&state->gqa_shape), query = SparkDeviceAttentionQueryWidth(&state->gqa_shape);
		const uint64_t kv = SparkDeviceAttentionKvWidth(&state->gqa_shape);
		const SparkDeviceStageSlice latent[] =
		{
			{ rows * heads * L::kQkDim * 2u, &state->attention.query },
			{ rows * slot * 2u, &state->attention.kv },
			{ rows * heads * 2u, &state->attention.gate },
			{ rows * heads * slot * 2u, &state->attention.packed },
			{ SparkOpsAttentionScratchBytes(state->stage.max_rows, L::kMlaRankHeads, G::kLatent, state->stage.max_positions), &state->attention.records },
			{ rows * heads * G::kLatent * 2u, &state->attention.latent },
			{ rows * heads * G::kValueDim * 2u, &state->attention.value }
		};
		const SparkDeviceStageSlice grouped[] =
		{
			{ rows * fused * 2u, &state->gqa.qkv },
			{ rows * query * 2u, &state->gqa.query },
			{ rows * kv * 2u, &state->gqa.key },
			{ rows * kv * 2u, &state->gqa.value },
			{ SparkOpsAttentionScratchBytes(state->stage.max_rows, G::kGqaHeads, G::kGqaHeadDim, state->stage.max_positions), &state->gqa.records },
			{ rows * query * 2u, &state->gqa.gate },
			{ rows * query * 2u, &state->gqa.gated },
			{ rows * fused * 2u, &state->gqa.qkv_local },
			{ rows * fused * 2u, &state->gqa.qkv_gathered }
		};
		const SparkDeviceStageSlice plan[] =
		{
			{ rows * hidden * 2u, &state->normed },
			{ rows * hidden * 2u, &state->partial },
			{ rows * hidden * 2u, &state->shared_out },
			{ rows * 3u * width * 2u, &state->delta.qkv },
			{ rows * L::kKdaRankHeads * 2u, &state->delta.beta },
			{ rows * Max(G::kKdaRank, 64u) * 2u, &state->delta.low },
			{ rows * width * 2u, &state->delta.decay },
			{ rows * width * 2u, &state->delta.gate },
			{ SparkOpsChannelDeltaScratchBytes(state->stage.max_rows, L::kKdaRankHeads, G::kKdaDim), &state->delta.delta },
			{ rows * width * 2u, &state->delta.core },
			{ rows * width * 2u, &state->delta.gated },
			{ rows * 2u * ffn * 2u, &state->ffn_scratch.gate_up },
			{ rows * ffn * 2u, &state->ffn_scratch.act }
		};
		uint32_t count = 0u;
		for ( uint32_t index = 0u; index < sizeof(plan) / sizeof(plan[0]); index++ )
			slices[count++] = plan[index];
		if constexpr ( L::kGqa )
			for ( uint32_t index = 0u; index < sizeof(grouped) / sizeof(grouped[0]); index++ )
				slices[count++] = grouped[index];
		else
			for ( uint32_t index = 0u; index < sizeof(latent) / sizeof(latent[0]); index++ )
				slices[count++] = latent[index];
		for ( uint32_t index = 0u; index < sizeof(experts) / sizeof(experts[0]); index++ )
		{
			slices[count].bytes = SparkDeviceExpertsScratchBytes(&state->stage, &state->experts_shape, index);
			slices[count++].target = experts[index];
		}
		return count;
	}

	static SparkStatus Open(const SparkStageRunnerModelOpen *request, void **out, SparkStageRunnerModelGeometry *geometry)
	{
		SparkDeviceStageSlice slices[LD_SLICES];
		State *state = (State *)calloc(1u, sizeof(State));
		SparkStatus status;
		*out = state;
		if ( state == 0 )
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		Shapes(state);
		const SparkDeviceStageRecurrentPlan pools[] = { { L::kKdaLayers, SparkDeviceChannelDeltaStateBytes(&state->delta_shape) },
			{ L::kKdaLayers, SparkDeviceChannelDeltaWindowBytes(&state->delta_shape) } };
		const SparkDeviceStageConfigValue config[] =
		{
			{ "hidden", G::kHidden }, { "layers", G::kLayers }, { "vocab", G::kVocab }, { "vocab_rows", G::kVocab / G::kTpDegree },
			{ "experts", G::kExperts }, { "local_experts", G::kLocalExperts }, { "top_k", G::kTopK }, { "shared_rows", G::kSharedRows }
		};
		status = SparkDeviceStageOpen(&state->stage, &state->shape, request);
		if ( status == SPARK_STATUS_OK && SparkNamedPackConfigU32(&state->stage.pack, "ffn_rows", &state->ffn) != SPARK_STATUS_OK )
			status = SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_PARSE_ERROR, "the pack configuration lacks ffn_rows");
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageExpectConfig(&state->stage, config, sizeof(config) / sizeof(config[0]));
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageReserve(&state->stage, SparkDeviceStagePlan(0, slices, ScratchSlices(state, slices)), pools,
				SPARK_DEVICE_GATED_DELTA_POOLS, request->configuration->state_budget_bytes);
		if ( status != SPARK_STATUS_OK )
			return status;
		if constexpr ( L::kGqa )
			SparkDeviceAttentionRopePrepare(&state->gqa_shape, &state->gqa_rope);
		else
			SparkDeviceLatentAttentionRopePrepare(&state->attention_shape, L::kMlaRankHeads, &state->rope);
		return SparkDeviceStageGeometry(&state->stage, geometry);
	}

	static SparkStatus BindLayer(State *state, const SparkStageRunnerDeviceWeights *weights, uint32_t index, uint32_t *recurrent, uint32_t *kv)
	{
		Layer *layer = &state->layers[index];
		SparkStatus status;
		layer->index = index;
		status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "input_norm", "f32", G::kHidden * sizeof(float), &layer->input_norm);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "post_norm", "f32", G::kHidden * sizeof(float), &layer->post_norm);
		if ( status == SPARK_STATUS_OK && L::Attention(index) )
		{
			layer->kv_index = (*kv)++;
			if constexpr ( L::kGqa )
				status = SparkDeviceAttentionBind(&state->stage, weights, index, &state->gqa_shape, &layer->gqa);
			else
				status = SparkDeviceLatentAttentionBind(&state->stage, weights, index, &state->attention_shape, &layer->attention);
		}
		else if ( status == SPARK_STATUS_OK )
		{
			layer->recurrent_index = (*recurrent)++;
			status = SparkDeviceChannelDeltaBind(&state->stage, weights, index, &state->delta_shape, &layer->delta);
		}
		if ( status == SPARK_STATUS_OK && index < G::kDenseLayers )
			return SparkDeviceFfnBind(&state->stage, weights, index, state->ffn, &layer->ffn);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceSigmoidExpertsBind(&state->stage, weights, index, &state->experts_shape, &layer->experts);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceFfnBindNamed(&state->stage, weights, index, "shared_gate_up", "shared_down", G::kSharedRows, &layer->shared);
		return status;
	}

	static SparkStatus BindDevice(void *model, const SparkStageRunnerDeviceWeights *weights, SparkStageRunnerModelGeometry *geometry)
	{
		State *state = (State *)model;
		SparkDeviceStageSlice slices[LD_SLICES];
		SparkStatus status = SPARK_STATUS_OK;
		uint32_t recurrent = 0u, kv = 0u;
		(void)geometry;
		if ( weights == 0 || weights->pack_address == 0u || weights->pack_bytes != state->stage.pack_bytes )
			return SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_INVALID_ARGUMENT, "bind_device needs the device mapping of the whole rank pack");
		for ( uint32_t index = 0u; index < G::kLayers && status == SPARK_STATUS_OK; index++ )
			status = BindLayer(state, weights, index, &recurrent, &kv);
		if ( status != SPARK_STATUS_OK )
			return status;
		return SparkDeviceStageAllocate(&state->stage, slices, ScratchSlices(state, slices));
	}

	static const Layer &LayerOf(State *state, uint32_t index)
	{
		return state->layers[index];
	}

	static SparkStatus Begin(State *state, const SparkStageRunnerStep *step, void *queue, Frame *frame)
	{
		frame->queue = (SparkDeviceQueue *)queue;
		return SparkDeviceStageStepTables(&state->stage, step, &frame->t);
	}

	static SparkStatus RmsNorm(State *state, const Frame &frame, SparkTensor weight)
	{
		return SparkDeviceResidualNorm(&state->stage, frame.queue, &frame.t, 0, weight, state->normed);
	}

	static SparkStatus AddRmsNorm(State *state, const Frame &frame, SparkTensor weight)
	{
		const SparkTensor addend = state->partial;
		return SparkDeviceResidualNorm(&state->stage, frame.queue, &frame.t, &addend, weight, state->normed);
	}

	static SparkStatus Attention(State *state, const Frame &frame, const Layer &layer)
	{
		if constexpr ( L::kGqa )
			return SparkDeviceAttention(&state->stage, frame.queue, &state->gqa_shape, &state->gqa_rope, &frame.t, &layer.gqa, layer.kv_index, &state->gqa,
				state->normed, state->partial);
		return SparkDeviceLatentAttention(&state->stage, frame.queue, &state->attention_shape, &state->rope, &frame.t, &layer.attention, layer.kv_index,
			&state->attention, state->normed, state->partial);
	}

	static SparkStatus Kda(State *state, const Frame &frame, const Layer &layer)
	{
		return SparkDeviceChannelDelta(&state->stage, frame.queue, &state->delta_shape, &frame.t, &layer.delta, layer.recurrent_index, &state->delta,
			state->normed, state->partial);
	}

	static SparkStatus AllReduce(State *state, const Frame &frame)
	{
		return SparkDeviceStageAllReduce(&state->stage, frame.queue, state->partial, frame.t.rows);
	}

	static SparkStatus Dense(State *state, const Frame &frame, const Layer &layer)
	{
		return SparkDeviceGatedFfn(&state->stage, frame.queue, state->ffn, SPARK_GATED_SILU, 0.0f, &frame.t, &layer.ffn, &state->ffn_scratch, state->normed,
			state->partial);
	}

	static SparkStatus Experts(State *state, const Frame &frame, const Layer &layer, uint32_t index)
	{
		const float routed = G::RoutedLimit(index), shared = G::SharedLimit(index);
		SparkDeviceExpertsShape shape = state->experts_shape;
		shape.limit = routed > 0.0f ? routed : INFINITY;
		SparkStatus status = SparkDeviceGatedFfn(&state->stage, frame.queue, G::kSharedRows, SPARK_GATED_SILU_CLAMP, shared > 0.0f ? shared : INFINITY,
			&frame.t, &layer.shared, &state->ffn_scratch, state->normed, state->shared_out);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceSigmoidExperts(&state->stage, frame.queue, &shape, &frame.t, &layer.experts, &state->experts, state->normed,
				state->shared_out, state->partial);
		return status;
	}

	static SparkStatus LayerDone(State *state, const Frame &frame, uint32_t index)
	{
		SparkDeviceStageLayerDone(&state->stage, index);
		return SparkDeviceStageDumpRows(&state->stage, frame.queue, &frame.t, frame.t.hidden, G::kHidden, index);
	}

	static SparkStatus AddRows(State *state, const Frame &frame)
	{
		return SparkOpsAddRows(state->stage.ops, frame.queue, frame.t.hidden, state->partial, frame.t.rows, G::kHidden, G::kHidden, G::kHidden);
	}

	static SparkStatus Check(State *state, const Frame &frame)
	{
		(void)state;
		(void)frame;
		return SPARK_STATUS_OK;
	}

	static SparkStatus Step(void *model, const SparkStageRunnerStep *step, void *queue)
	{
		return LingSchedule<G, LingDevice>::Step((State *)model, step, queue);
	}

	static const SparkStageRunnerModelInterface *Interface(void)
	{
		static const SparkStageRunnerModelInterface interface_table =
		{
			SPARK_STAGE_RUNNER_MODEL_ABI_VERSION,
			P::kTag,
			P::kName,
			"official",
			Open,
			SparkDeviceStageManifestCheck,
			SparkDeviceStageBindWeightd,
			SparkDeviceStageClose,
			Step,
			SparkDeviceStageSideband,
			SparkDeviceStageAttachKv,
			SparkDeviceStageRecurrentCopy,
			SparkDeviceStageResetSlot,
			SparkDeviceStageProbeBuffers,
			SparkDeviceStageReport,
			0,
			SparkDeviceStageLane,
			0,
			0,
			BindDevice,
			0,
			0
		};
		return &interface_table;
	}
};

template<uint32_t TP>
struct Ling3FlashDevice
{
	using Geometry = Ling3Geometry<TP>;
	static constexpr const char *kTag = "sparkpipe_ling3_device";
	static constexpr const char *kName = "ling-3.0-flash";
	static constexpr uint32_t kProjection = SPARK_PROJECTION_GEMV;
};

template<uint32_t TP>
struct Ling3FinDevice
{
	using Geometry = Ling3Geometry<TP>;
	static constexpr const char *kTag = "sparkpipe_ling3_fin_device";
	static constexpr const char *kName = "ling-3.0-flash-fin";
	static constexpr uint32_t kProjection = SPARK_PROJECTION_GEMV;
};

template<uint32_t TP>
struct SolarOpen2Device
{
	using Geometry = SolarOpen2Geometry<TP>;
	static constexpr const char *kTag = "sparkpipe_solar_open2_device";
	static constexpr const char *kName = "solar-open-2";
	static constexpr uint32_t kProjection = SPARK_PROJECTION_GEMV;
};

template<template<uint32_t> class P>
static const SparkStageRunnerModelInterface *LingDeviceAt(uint32_t tp_degree)
{
	switch ( tp_degree )
	{
	case 1u:
		return LingDevice<P<1u>>::Interface();
	case 2u:
		return LingDevice<P<2u>>::Interface();
	case 4u:
		return LingDevice<P<4u>>::Interface();
	case 8u:
		return LingDevice<P<8u>>::Interface();
	default:
		return 0;
	}
}

const SparkStageRunnerModelInterface *SparkLing3FlashDeviceStageModelAt(uint32_t tp_degree)
{
	return LingDeviceAt<Ling3FlashDevice>(tp_degree);
}

const SparkStageRunnerModelInterface *SparkLing3FinDeviceStageModelAt(uint32_t tp_degree)
{
	return LingDeviceAt<Ling3FinDevice>(tp_degree);
}

const SparkStageRunnerModelInterface *SparkSolarOpen2DeviceStageModelAt(uint32_t tp_degree)
{
	return LingDeviceAt<SolarOpen2Device>(tp_degree);
}
