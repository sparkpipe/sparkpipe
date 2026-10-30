#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inference/runner/device_layers.h"
#include "sparkpipe/spark_gemma4_26b_stage_model.h"
#include "sparkpipe/spark_gemma4_31b_stage_model.h"
#include "sparkpipe/spark_gemma4_device_stage.h"
#include "sparkpipe/spark_gemma4_family.h"
#include "sparkpipe/spark_laguna_tp16_stage_model.h"
#include "sparkpipe/spark_muse_tp16_stage_model.h"

#define GD_SLICES 24u

template<class P>
struct GemmaDevice
{
	using G = typename P::Geometry;
	using L = GemmaLayout<G>;

	static_assert(!G::kRouterFloat && !G::kLayerTable && (!G::kEmbedNorm || G::kEmbedScale == 1.0f),
		"the device layers do not cover this Gemma variant yet");

	struct Layer
	{
		SparkTensor input_norm;
		SparkTensor post_attn_norm;
		SparkTensor pre_ff_norm;
		SparkTensor post_ff_norm;
		SparkTensor post_ff_norm_1;
		SparkTensor post_ff_norm_2;
		SparkTensor layer_scalar;
		SparkDeviceAttentionWeights attention;
		SparkDeviceFfnWeights ffn_weights;
		SparkDeviceExpertsWeights experts;
		uint32_t index;
		uint32_t ffn;
	};

	struct State
	{
		SparkDeviceStage stage;
		SparkDeviceStageShape shape;
		SparkDeviceAttentionShape sliding;
		SparkDeviceAttentionShape full;
		SparkDeviceExpertsShape experts_shape;
		uint32_t ffn;
		Layer layers[G::kLayers];
		SparkTensor normed;
		SparkTensor partial;
		SparkTensor expert_out;
		SparkDeviceAttentionScratch attention;
		SparkDeviceFfnScratch ffn_scratch;
		SparkDeviceExpertsScratch experts;
		SparkDeviceAttentionRope sliding_rope;
		SparkDeviceAttentionRope full_rope;
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
		const bool sliding_slot = L::kSlidingKv >= L::kFullKv;
		SparkDeviceStageShape *shape = &state->shape;
		SparkDeviceAttentionShape *sliding = &state->sliding, *full = &state->full;
		SparkDeviceExpertsShape *experts = &state->experts_shape;
		memset(shape, 0, sizeof(*shape));
		shape->tag = P::kTag;
		shape->hidden = G::kHidden;
		shape->vocab = G::kVocab;
		shape->layers = G::kLayers;
		shape->kv_layers = G::kLayers;
		shape->kv_heads = sliding_slot ? G::kSlidingKvHeads : G::kFullKvHeads;
		shape->kv_head_dimension = sliding_slot ? G::kSlidingDim : G::kFullDim;
		shape->pack_alignment = GEMMA_PACK_ALIGNMENT;
		shape->epsilon = G::kNormEpsilon;
		shape->post_epsilon = G::kPostEpsilon;
		shape->tied_head = G::kUntiedHead ? 0u : 1u;
		shape->logit_softcap = G::kLogitSoftcap;
		shape->logit_scale = G::kLogitScale;
		shape->embed_alias_first = G::kEmbedAliasFirst;
		shape->embed_alias_count = G::kEmbedAliasCount;
		shape->embed_alias_row = G::kEmbedAliasRow;
		shape->projection = P::kProjection;
		memset(sliding, 0, sizeof(*sliding));
		sliding->heads = G::kSlidingHeads;
		sliding->kv_heads = G::kSlidingKvHeads;
		sliding->head_dimension = G::kSlidingDim;
		sliding->rope_dimension = G::kSlidingDim;
		sliding->rope_theta = G::kSlidingTheta;
		sliding->epsilon = G::kNormEpsilon;
		sliding->scale = G::kAttentionScale;
		sliding->value_norm = G::kValueNorm ? 1u : 0u;
		sliding->output_gate = G::kOutputGate ? SPARK_ATTENTION_GATE_SIGMOID_BF16 : 0u;
		sliding->gate_projection = G::kOutputGate ? 1u : 0u;
		sliding->qk_weightless = G::kQkNormWeighted ? 0u : 1u;
		sliding->query_multiply = G::kQueryMultiply == 1.0f ? 0.0f : G::kQueryMultiply;
		sliding->window = G::kWindow;
		sliding->rope_rounded_terms = 1u;
		memset(full, 0, sizeof(*full));
		full->heads = G::kFullHeads;
		full->kv_heads = G::kFullKvHeads;
		full->head_dimension = G::kFullDim;
		full->rope_dimension = G::kFullRope ? L::kFullRopeWidth : 0u;
		full->rope_theta = G::kFullTheta;
		full->epsilon = G::kNormEpsilon;
		full->scale = G::kAttentionScale;
		full->key_is_value = G::kKeyIsValue ? 1u : 0u;
		full->value_norm = G::kValueNorm ? 1u : 0u;
		full->output_gate = sliding->output_gate;
		full->gate_projection = sliding->gate_projection;
		full->qk_weightless = sliding->qk_weightless;
		full->query_multiply = sliding->query_multiply;
		full->rope_pairs = G::kFullRopePairs;
		full->rope_rounded_terms = 1u;
		full->rope_scale = G::kFullRopeScale;
		full->yarn_factor = G::kYarnFactor;
		full->yarn_original = G::kYarnOriginal;
		full->yarn_beta_fast = G::kYarnBetaFast;
		full->yarn_beta_slow = G::kYarnBetaSlow;
		sliding->head_gate = full->head_gate = G::kHeadGate ? 1u : 0u;
		memset(experts, 0, sizeof(*experts));
		experts->experts = G::kExperts;
		experts->top_k = G::kTopK;
		experts->intermediate = G::kMoeInter;
		experts->activation = G::kGeluMlp ? SPARK_GATED_GELU_TANH : SPARK_GATED_SILU;
		experts->epsilon = G::kNormEpsilon;
		if constexpr ( G::kSigmoidRouter )
			experts->routed_scale = G::kRoutedScale;
		experts->round_logits = G::kRouterFloat ? 0u : 1u;
	}

	static uint32_t ScratchSlices(State *state, SparkDeviceStageSlice *slices)
	{
		const uint64_t rows = state->stage.max_rows, hidden = G::kHidden;
		const uint64_t fused = Max(SparkDeviceAttentionFusedWidth(&state->sliding), SparkDeviceAttentionFusedWidth(&state->full));
		const uint64_t query = Max(SparkDeviceAttentionQueryWidth(&state->sliding), SparkDeviceAttentionQueryWidth(&state->full));
		const uint64_t kv = Max(SparkDeviceAttentionKvWidth(&state->sliding), SparkDeviceAttentionKvWidth(&state->full));
		SparkTensor *experts[] = { &state->experts.routed, &state->experts.logits, &state->experts.experts, &state->experts.weights,
			&state->experts.normed, &state->experts.gate_up, &state->experts.act, &state->experts.out, &state->experts.sorted };
		const SparkDeviceStageSlice plan[] =
		{
			{ rows * hidden * 2u, &state->normed },
			{ rows * hidden * 2u, &state->partial },
			{ rows * hidden * 2u, &state->expert_out },
			{ rows * fused * 2u, &state->attention.qkv },
			{ rows * query * 2u, &state->attention.query },
			{ rows * kv * 2u, &state->attention.key },
			{ rows * kv * 2u, &state->attention.value },
			{ Max(SparkOpsAttentionScratchBytes(state->stage.max_rows, state->full.heads, state->full.head_dimension, state->stage.max_positions),
				SparkOpsAttentionScratchBytes(state->stage.max_rows, state->sliding.heads, state->sliding.head_dimension, state->stage.max_positions)),
				&state->attention.records },
			{ rows * Max(query, Max(SparkDeviceAttentionHeadGateRows(&state->sliding, G::kTpDegree), SparkDeviceAttentionHeadGateRows(&state->full,
				G::kTpDegree))) * 2u, &state->attention.gate },
			{ rows * query * 2u, &state->attention.gated },
			{ rows * fused / G::kTpDegree * 2u, &state->attention.qkv_local },
			{ rows * fused * 2u, &state->attention.qkv_gathered },
			{ rows * 2u * state->ffn * 2u, &state->ffn_scratch.gate_up },
			{ rows * state->ffn * 2u, &state->ffn_scratch.act }
		};
		uint32_t count = 0u;
		for ( uint32_t index = 0u; index < sizeof(plan) / sizeof(plan[0]); index++ )
			if ( (G::kMoe || plan[index].target != &state->expert_out) && (G::kOutputGate || G::kHeadGate || plan[index].target != &state->attention.gate) &&
				(G::kTpDegree > 1u || (plan[index].target != &state->attention.qkv_local && plan[index].target != &state->attention.qkv_gathered)) )
				slices[count++] = plan[index];
		if constexpr ( G::kMoe )
			for ( uint32_t index = 0u; index < sizeof(experts) / sizeof(experts[0]); index++ )
			{
				slices[count].bytes = SparkDeviceExpertsScratchBytes(&state->stage, &state->experts_shape, index);
				slices[count++].target = experts[index];
			}
		return count;
	}

	static SparkStatus Open(const SparkStageRunnerModelOpen *request, void **out, SparkStageRunnerModelGeometry *geometry)
	{
		SparkDeviceStageSlice slices[GD_SLICES];
		State *state = (State *)calloc(1u, sizeof(State));
		SparkStatus status;
		*out = state;
		if ( state == 0 )
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		Shapes(state);
		status = SparkDeviceStageOpen(&state->stage, &state->shape, request);
		if ( status == SPARK_STATUS_OK && SparkNamedPackConfigU32(&state->stage.pack, "ffn_rows", &state->ffn) != SPARK_STATUS_OK )
			status = SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_PARSE_ERROR, "the pack configuration lacks ffn_rows");
		const SparkDeviceStageConfigValue config[] =
		{
			{ "hidden", G::kHidden }, { "layers", G::kLayers }, { "vocab", G::kVocab }, { "vocab_rows", G::kVocab / G::kTpDegree },
			{ "full_period", G::kPeriod }, { "sliding_window", G::kWindow }, { "experts", G::kExperts }, { "top_k", G::kTopK }
		};
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageExpectConfig(&state->stage, config, G::kMoe ? 8u : 6u);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageReserve(&state->stage, SparkDeviceStagePlan(0, slices, ScratchSlices(state, slices)), 0, 0u,
				request->configuration->state_budget_bytes);
		if ( status != SPARK_STATUS_OK )
			return status;
		SparkDeviceAttentionRopePrepare(&state->sliding, &state->sliding_rope);
		SparkDeviceAttentionRopePrepare(&state->full, &state->full_rope);
		return SparkDeviceStageGeometry(&state->stage, geometry);
	}

	static SparkStatus BindLayer(State *state, const SparkStageRunnerDeviceWeights *weights, uint32_t index)
	{
		static const char *const norms[] = { "input_norm", "pre_ff_norm", "post_attn_norm", "post_ff_norm", "post_ff_norm_1", "post_ff_norm_2" };
		Layer *layer = &state->layers[index];
		SparkTensor *targets[] = { &layer->input_norm, &layer->pre_ff_norm, &layer->post_attn_norm, &layer->post_ff_norm, &layer->post_ff_norm_1,
			&layer->post_ff_norm_2 };
		const bool moe = G::kMoe && index >= G::kFirstMoeLayer;
		const uint32_t norm_count = G::kPreNorm ? 2u : G::kMoe ? 6u : 4u;
		char key[64];
		uint64_t offset = 0u, bytes = 0u;
		uint32_t gate_up_rows = 0u;
		SparkStatus status = SPARK_STATUS_OK;
		layer->index = index;
		for ( uint32_t norm = 0u; norm < norm_count && status == SPARK_STATUS_OK; norm++ )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, norms[norm], "f32", G::kHidden * sizeof(float), targets[norm]);
		snprintf(key, sizeof(key), "layers.%u.ffn_gate_up", index);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageEntry(&state->stage, key, "bf16", &offset, &bytes, &gate_up_rows);
		layer->ffn = gate_up_rows / 2u;
		if ( status == SPARK_STATUS_OK && (gate_up_rows % 2u != 0u || layer->ffn > state->ffn) )
			status = SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_SCHEMA_ERROR, "a layer's MLP is wider than the pack's ffn_rows");
		if ( G::kLayerScalar && status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "layer_scalar", "f32", sizeof(float), &layer->layer_scalar);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceAttentionBind(&state->stage, weights, index, L::Full(index) ? &state->full : &state->sliding, &layer->attention);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceFfnBind(&state->stage, weights, index, layer->ffn, &layer->ffn_weights);
		if ( moe && status == SPARK_STATUS_OK )
			status = G::kSigmoidRouter ? SparkDeviceSigmoidExpertsBind(&state->stage, weights, index, &state->experts_shape, &layer->experts) :
				SparkDeviceExpertsBind(&state->stage, weights, index, &state->experts_shape, &layer->experts);
		return status;
	}

	static SparkStatus BindDevice(void *model, const SparkStageRunnerDeviceWeights *weights, SparkStageRunnerModelGeometry *geometry)
	{
		State *state = (State *)model;
		SparkDeviceStageSlice slices[GD_SLICES];
		SparkStatus status = SPARK_STATUS_OK;
		(void)geometry;
		if ( weights == 0 || weights->pack_address == 0u || weights->pack_bytes != state->stage.pack_bytes )
			return SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_INVALID_ARGUMENT, "bind_device needs the device mapping of the whole rank pack");
		for ( uint32_t index = 0u; index < G::kLayers && status == SPARK_STATUS_OK; index++ )
			status = BindLayer(state, weights, index);
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

	static SparkStatus EmbedNorm(State *state, const Frame &frame)
	{
		SparkNormParams params;
		memset(&params, 0, sizeof(params));
		params.rows = frame.t.rows;
		params.dimension = params.input_stride = params.output_stride = G::kHidden;
		params.epsilon = G::kNormEpsilon;
		params.weightless = 1u;
		return SparkOpsRmsNormResidual(state->stage.ops, frame.queue, frame.t.hidden, 0, frame.t.hidden, frame.t.hidden, &params);
	}

	static SparkStatus EmbedScale(State *state, const Frame &frame, float scale)
	{
		return SparkOpsScaleRows(state->stage.ops, frame.queue, frame.t.hidden, (uint64_t)frame.t.rows * G::kHidden, scale);
	}

	static SparkStatus RmsNorm(State *state, const Frame &frame, SparkTensor weight)
	{
		return SparkDeviceResidualNorm(&state->stage, frame.queue, &frame.t, 0, weight, state->normed);
	}

	static SparkStatus NormAdd(State *state, const Frame &frame, SparkTensor weight)
	{
		return SparkDeviceBranchNorm(&state->stage, frame.queue, &frame.t, state->partial, 0, 0, 0, weight, 0);
	}

	static SparkStatus NormAddScaled(State *state, const Frame &frame, SparkTensor weight, const SparkTensor &scalar)
	{
		return SparkDeviceBranchNorm(&state->stage, frame.queue, &frame.t, state->partial, 0, 0, 0, weight, &scalar);
	}

	static SparkStatus BranchNormAdd(State *state, const Frame &frame, const SparkTensor &first_norm, const SparkTensor &second_norm, SparkTensor norm,
		const SparkTensor &scalar)
	{
		return SparkDeviceBranchNorm(&state->stage, frame.queue, &frame.t, state->partial, &state->expert_out, &first_norm, &second_norm, norm, &scalar);
	}

	template<bool FULL>
	static SparkStatus Attention(State *state, const Frame &frame, const Layer &layer)
	{
		return SparkDeviceAttention(&state->stage, frame.queue, FULL ? &state->full : &state->sliding, FULL ? &state->full_rope : &state->sliding_rope,
			&frame.t, &layer.attention, layer.index, &state->attention, state->normed, state->partial);
	}

	static SparkStatus Mlp(State *state, const Frame &frame, const Layer &layer)
	{
		return SparkDeviceGatedFfn(&state->stage, frame.queue, layer.ffn, G::kGeluMlp ? SPARK_GATED_GELU_TANH : SPARK_GATED_SILU, 0.0f, &frame.t,
			&layer.ffn_weights, &state->ffn_scratch, state->normed, state->partial);
	}

	static SparkTensor Branch(State *state, GemmaBranch branch)
	{
		return branch == GEMMA_EXPERT_BRANCH ? state->expert_out : state->partial;
	}

	static SparkStatus AddRmsNorm(State *state, const Frame &frame, GemmaBranch branch, SparkTensor weight)
	{
		const SparkTensor addend = Branch(state, branch);
		return SparkDeviceResidualNorm(&state->stage, frame.queue, &frame.t, &addend, weight, state->normed);
	}

	static SparkStatus AddRows(State *state, const Frame &frame, GemmaBranch branch)
	{
		return SparkOpsAddRows(state->stage.ops, frame.queue, frame.t.hidden, Branch(state, branch), frame.t.rows, G::kHidden, G::kHidden, G::kHidden);
	}

	static SparkStatus SigmoidExperts(State *state, const Frame &frame, const Layer &layer)
	{
		return SparkDeviceSigmoidExperts(&state->stage, frame.queue, &state->experts_shape, &frame.t, &layer.experts, &state->experts, state->normed,
			state->partial, state->expert_out);
	}

	static SparkStatus Experts(State *state, const Frame &frame, const Layer &layer)
	{
		return SparkDeviceExperts(&state->stage, frame.queue, &state->experts_shape, &frame.t, layer.index, &layer.experts, &state->experts,
			state->expert_out);
	}

	static SparkStatus AllReduce(State *state, const Frame &frame, GemmaBranch branch, uint32_t blocks)
	{
		SparkStatus status = SparkDeviceStageAllReduce(&state->stage, frame.queue, Branch(state, branch), frame.t.rows);
		if ( status == SPARK_STATUS_OK && blocks > 1u )
			status = SparkDeviceStageAllReduce(&state->stage, frame.queue, state->expert_out, frame.t.rows);
		return status;
	}

	static SparkStatus Check(State *state, const Frame &frame)
	{
		(void)state;
		(void)frame;
		return SPARK_STATUS_OK;
	}

	static SparkStatus LayerDone(State *state, const Frame &frame, uint32_t index)
	{
		(void)frame;
		SparkDeviceStageLayerDone(&state->stage, index);
		return SPARK_STATUS_OK;
	}

	static SparkStatus Step(void *model, const SparkStageRunnerStep *step, void *queue)
	{
		return GemmaSchedule<G, GemmaDevice>::Step((State *)model, step, queue);
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
			0
		};
		return &interface_table;
	}
};

template<uint32_t TP>
struct Gemma4_26bDevice
{
	using Geometry = Gemma4_26bGeometry<TP>;
	static constexpr const char *kTag = "sparkpipe_gemma4_26b_device";
	static constexpr const char *kName = "gemma4-26b-a4b";
	static constexpr uint32_t kProjection = SPARK_PROJECTION_GEMV;
};

template<uint32_t TP>
struct Gemma4_31bDevice
{
	using Geometry = Gemma4_31bGeometry<TP>;
	static constexpr const char *kTag = "sparkpipe_gemma4_31b_device";
	static constexpr const char *kName = "gemma4-31b";
	static constexpr uint32_t kProjection = SPARK_PROJECTION_ROW_TILES;
};

template<uint32_t TP>
struct MuseGlimmerDevice
{
	using Geometry = MuseGlimmerGeometry<TP>;
	static constexpr const char *kTag = "sparkpipe_muse_glimmer_device";
	static constexpr const char *kName = "muse-glimmer-30b";
	static constexpr uint32_t kProjection = SPARK_PROJECTION_ROW_TILES;
};

template<uint32_t TP>
struct LagunaDevice
{
	using Geometry = LagunaGeometry<TP>;
	static constexpr const char *kTag = "sparkpipe_laguna_device";
	static constexpr const char *kName = "laguna-s-2.1";
	static constexpr uint32_t kProjection = SPARK_PROJECTION_GEMV;
};

template<template<uint32_t> class P>
static const SparkStageRunnerModelInterface *GemmaDeviceAt(uint32_t tp_degree)
{
	switch ( tp_degree )
	{
	case 1u:
		return GemmaDevice<P<1u>>::Interface();
	case 2u:
		return GemmaDevice<P<2u>>::Interface();
	case 4u:
		return GemmaDevice<P<4u>>::Interface();
	case 8u:
		return GemmaDevice<P<8u>>::Interface();
	default:
		return 0;
	}
}

const SparkStageRunnerModelInterface *SparkGemma4_26bDeviceStageModel(void)
{
	return GemmaDeviceAt<Gemma4_26bDevice>(1u);
}

const SparkStageRunnerModelInterface *SparkGemma4_31bDeviceStageModel(void)
{
	return GemmaDeviceAt<Gemma4_31bDevice>(1u);
}

const SparkStageRunnerModelInterface *SparkMuseGlimmerDeviceStageModel(void)
{
	return GemmaDeviceAt<MuseGlimmerDevice>(1u);
}

const SparkStageRunnerModelInterface *SparkGemma4_26bDeviceStageModelAt(uint32_t tp_degree)
{
	return GemmaDeviceAt<Gemma4_26bDevice>(tp_degree);
}

const SparkStageRunnerModelInterface *SparkGemma4_31bDeviceStageModelAt(uint32_t tp_degree)
{
	return GemmaDeviceAt<Gemma4_31bDevice>(tp_degree);
}

const SparkStageRunnerModelInterface *SparkMuseGlimmerDeviceStageModelAt(uint32_t tp_degree)
{
	return GemmaDeviceAt<MuseGlimmerDevice>(tp_degree);
}

const SparkStageRunnerModelInterface *SparkLagunaDeviceStageModelAt(uint32_t tp_degree)
{
	return GemmaDeviceAt<LagunaDevice>(tp_degree);
}
