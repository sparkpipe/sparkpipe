#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inference/runner/device_layers.h"
#include "sparkpipe/spark_hy4_device_stage.h"
#include "sparkpipe/spark_hy4_family.h"
#include "sparkpipe/spark_hy4_tp16_stage_model.h"

#define HD_SLICES 32u
#define HD_MIXES 8u

template<class P>
struct HyDevice
{
	using G = typename P::Geometry;
	using L = HyLayout<G>;

	struct Hyper
	{
		SparkTensor fn;
		SparkTensor base;
		SparkTensor scale;
	};

	struct Layer
	{
		Hyper hyper[2];
		SparkTensor input_norm;
		SparkTensor post_norm;
		SparkDeviceLatentAttentionWeights attention;
		SparkDeviceFfnWeights ffn;
		SparkDeviceFfnWeights shared;
		SparkTensor router;
		SparkTensor router_bias;
		SparkDeviceExpertMatrix gate;
		SparkDeviceExpertMatrix up;
		SparkDeviceExpertMatrix down;
		uint32_t index;
	};

	struct State
	{
		SparkDeviceStage stage;
		SparkDeviceStageShape shape;
		SparkDeviceLatentAttentionShape attention_shape;
		SparkDeviceLatentAttentionRope rope;
		Layer layers[G::kLayers];
		Hyper head;
		uint32_t ffn_rows;
		uint32_t shared_rows;
		SparkTensor streams;
		SparkTensor post;
		SparkTensor mixed;
		SparkTensor normed;
		SparkTensor partial;
		SparkTensor shared_out;
		SparkTensor router_logits;
		SparkDeviceLatentAttentionScratch attention;
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
		SparkDeviceLatentAttentionShape *attention = &state->attention_shape;
		memset(shape, 0, sizeof(*shape));
		shape->tag = P::kTag;
		shape->hidden = G::kHidden;
		shape->vocab = G::kVocab;
		shape->layers = G::kLayers;
		shape->kv_layers = G::kLayers;
		shape->kv_heads = 1u;
		shape->kv_head_dimension = (L::kSlot + G::kLatent) / 2u;
		shape->pack_alignment = HY_PACK_ALIGNMENT;
		shape->epsilon = G::kEpsilon;
		shape->projection = P::kProjection;
		memset(attention, 0, sizeof(*attention));
		attention->heads = G::kHeads;
		attention->nope = G::kNope;
		attention->rope = G::kRope;
		attention->latent = G::kLatent;
		attention->value_dimension = G::kValueDim;
		attention->rope_theta = G::kRopeTheta;
		attention->epsilon = G::kEpsilon;
		attention->scale = 1.0f / sqrtf((float)L::kQkDim);
		attention->head_gate = SPARK_LATENT_GATE_ELEMENT;
		attention->query_lora = G::kQLora;
		attention->norm_bf16 = 1u;
		attention->sinks = 1u;
		attention->tensor_prefix = "";
	}

	static uint32_t ScratchSlices(State *state, SparkDeviceStageSlice *slices)
	{
		const uint64_t rows = state->stage.max_rows, hidden = G::kHidden, routes = rows * G::kTopK, heads = L::kRankHeads;
		const uint64_t ffn = Max(state->ffn_rows, state->shared_rows);
		const SparkDeviceStageSlice plan[] =
		{
			{ rows * L::kResidual * 2u, &state->streams },
			{ rows * G::kStreams * sizeof(float), &state->post },
			{ rows * hidden * 2u, &state->mixed },
			{ rows * hidden * 2u, &state->normed },
			{ rows * hidden * 2u, &state->partial },
			{ rows * hidden * 2u, &state->shared_out },
			{ rows * G::kExperts * sizeof(float), &state->router_logits },
			{ rows * G::kQLora * 2u, &state->attention.query_lora },
			{ rows * heads * L::kQkDim * 2u, &state->attention.query },
			{ rows * L::kSlot * 2u, &state->attention.kv },
			{ rows * heads * G::kValueDim * 2u, &state->attention.gate },
			{ rows * heads * L::kSlot * 2u, &state->attention.packed },
			{ SparkOpsAttentionScratchBytes(state->stage.max_rows, (uint32_t)heads, G::kLatent, state->stage.max_positions), &state->attention.records },
			{ rows * heads * G::kLatent * 2u, &state->attention.latent },
			{ rows * heads * G::kValueDim * 2u, &state->attention.value },
			{ rows * 2u * ffn * 2u, &state->ffn_scratch.gate_up },
			{ rows * ffn * 2u, &state->ffn_scratch.act },
			{ routes * sizeof(uint32_t), &state->experts.experts },
			{ routes * sizeof(float), &state->experts.weights },
			{ routes * 2u * G::kMoeInter * 2u, &state->experts.gate_up },
			{ routes * G::kMoeInter * 2u, &state->experts.act },
			{ routes * hidden * 2u, &state->experts.out },
			{ (routes + 1u) * hidden * 2u, &state->experts.sorted },
			{ Max(Max(2u * ffn, (uint64_t)G::kQLora), Max((uint64_t)heads * L::kQkDim, Max((uint64_t)L::kSlot, hidden))) * Max(hidden, (uint64_t)G::kQLora) * 2u,
				&state->stage.dequant }
		};
		const uint32_t count = sizeof(plan) / sizeof(plan[0]);
		for ( uint32_t index = 0u; index < count; index++ )
			slices[index] = plan[index];
		return count;
	}

	static SparkStatus Open(const SparkStageRunnerModelOpen *request, void **out, SparkStageRunnerModelGeometry *geometry)
	{
		SparkDeviceStageSlice slices[HD_SLICES];
		State *state = (State *)calloc(1u, sizeof(State));
		SparkStatus status;
		*out = state;
		if ( state == 0 )
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		Shapes(state);
		status = SparkDeviceStageOpen(&state->stage, &state->shape, request);
		if ( status == SPARK_STATUS_OK && (SparkNamedPackConfigU32(&state->stage.pack, "ffn_rows", &state->ffn_rows) != SPARK_STATUS_OK ||
			SparkNamedPackConfigU32(&state->stage.pack, "shared_rows", &state->shared_rows) != SPARK_STATUS_OK) )
			status = SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_PARSE_ERROR, "the pack configuration lacks ffn_rows or shared_rows");
		const SparkDeviceStageConfigValue config[] =
		{
			{ "hidden", G::kHidden }, { "layers", G::kLayers }, { "vocab", G::kVocab }, { "vocab_rows", G::kVocab / G::kTpDegree },
			{ "experts", G::kExperts }, { "local_experts", G::kLocalExperts }, { "top_k", G::kTopK }
		};
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageExpectConfig(&state->stage, config, sizeof(config) / sizeof(config[0]));
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageReserve(&state->stage, SparkDeviceStagePlan(0, slices, ScratchSlices(state, slices)), 0, 0u,
				request->configuration->state_budget_bytes);
		if ( status != SPARK_STATUS_OK )
			return status;
		SparkDeviceLatentAttentionRopePrepare(&state->attention_shape, L::kRankHeads, &state->rope);
		return SparkDeviceStageGeometry(&state->stage, geometry);
	}

	static SparkStatus BindHyper(State *state, const SparkStageRunnerDeviceWeights *weights, const char *prefix, uint32_t mixes, uint32_t scales, Hyper *hyper)
	{
		char name[64];
		snprintf(name, sizeof(name), "%s_fn", prefix);
		SparkStatus status = SparkDeviceStageTensor(&state->stage, weights, name, "f32", (uint64_t)mixes * L::kResidual * sizeof(float), &hyper->fn);
		snprintf(name, sizeof(name), "%s_base", prefix);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageTensor(&state->stage, weights, name, "f32", (uint64_t)mixes * sizeof(float), &hyper->base);
		snprintf(name, sizeof(name), "%s_scale", prefix);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageTensor(&state->stage, weights, name, "f32", (uint64_t)scales * sizeof(float), &hyper->scale);
		return status;
	}

	static SparkStatus BindExpert(State *state, const SparkStageRunnerDeviceWeights *weights, uint32_t index, const char *name, uint64_t rows,
		uint64_t columns, SparkDeviceExpertMatrix *matrix)
	{
		char scale_name[64];
		SparkStatus status = SparkDeviceStageLayerTensor(&state->stage, weights, index, name, "fp8", G::kLocalExperts * rows * columns, &matrix->values);
		matrix->codec = SPARK_WEIGHT_CODEC_FP8_E4M3;
		matrix->scale_group = SPARK_MXFP8_GROUP;
		snprintf(scale_name, sizeof(scale_name), "%s_scale", name);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, scale_name, "u8", G::kLocalExperts * rows * (columns / SPARK_MXFP8_GROUP),
				&matrix->scale);
		if ( status == SPARK_STATUS_OK )
			SparkDeviceStageNoteRouted(&state->stage, matrix->values.bytes + matrix->scale.bytes, G::kLocalExperts, G::kTopK);
		return status;
	}

	static SparkStatus BindLayer(State *state, const SparkStageRunnerDeviceWeights *weights, uint32_t index)
	{
		Layer *layer = &state->layers[index];
		char prefix[48];
		SparkStatus status = SPARK_STATUS_OK;
		layer->index = index;
		snprintf(prefix, sizeof(prefix), "layers.%u.attn_hc", index);
		status = BindHyper(state, weights, prefix, HD_MIXES, 2u, &layer->hyper[HY_ATTENTION_PART]);
		snprintf(prefix, sizeof(prefix), "layers.%u.mlp_hc", index);
		if ( status == SPARK_STATUS_OK )
			status = BindHyper(state, weights, prefix, HD_MIXES, 2u, &layer->hyper[HY_FFN_PART]);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "input_norm", "bf16", G::kHidden * 2u, &layer->input_norm);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "post_norm", "bf16", G::kHidden * 2u, &layer->post_norm);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceLatentAttentionBind(&state->stage, weights, index, &state->attention_shape, &layer->attention);
		if ( status == SPARK_STATUS_OK && index < G::kDenseLayers )
			return SparkDeviceFfnBind(&state->stage, weights, index, state->ffn_rows, &layer->ffn);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "router", "bf16", (uint64_t)G::kExperts * G::kHidden * 2u, &layer->router);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "router_bias", "f32", (uint64_t)G::kExperts * sizeof(float), &layer->router_bias);
		if ( status == SPARK_STATUS_OK )
			status = BindExpert(state, weights, index, "moe_w1", G::kMoeInter, G::kHidden, &layer->gate);
		if ( status == SPARK_STATUS_OK )
			status = BindExpert(state, weights, index, "moe_w3", G::kMoeInter, G::kHidden, &layer->up);
		if ( status == SPARK_STATUS_OK )
			status = BindExpert(state, weights, index, "moe_w2", G::kHidden, G::kMoeInter, &layer->down);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceFfnBindNamed(&state->stage, weights, index, "shared_gate_up", "shared_down", state->shared_rows, &layer->shared);
		return status;
	}

	static SparkStatus BindDevice(void *model, const SparkStageRunnerDeviceWeights *weights, SparkStageRunnerModelGeometry *geometry)
	{
		State *state = (State *)model;
		SparkDeviceStageSlice slices[HD_SLICES];
		SparkStatus status;
		(void)geometry;
		if ( weights == 0 || weights->pack_address == 0u || weights->pack_bytes != state->stage.pack_bytes )
			return SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_INVALID_ARGUMENT, "bind_device needs the device mapping of the whole rank pack");
		status = BindHyper(state, weights, "head_hc", G::kStreams, 1u, &state->head);
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

	static SparkHcParams HcParams(const Frame &frame, uint32_t mixes)
	{
		SparkHcParams params;
		memset(&params, 0, sizeof(params));
		params.rows = frame.t.rows;
		params.streams = G::kStreams;
		params.width = G::kHidden;
		params.mixes = mixes;
		params.norm_epsilon = G::kEpsilon;
		params.hc_epsilon = G::kHcEpsilon;
		params.post_magnitude = G::kHcMagnitude;
		return params;
	}

	static SparkStatus Begin(State *state, const SparkStageRunnerStep *step, void *queue, Frame *frame)
	{
		frame->queue = (SparkDeviceQueue *)queue;
		SparkStatus status = SparkDeviceStageStepTables(&state->stage, step, &frame->t);
		for ( uint32_t row = 0u; status == SPARK_STATUS_OK && row < frame->t.rows; row++ )
			if ( frame->t.host_positions == 0 || frame->t.host_positions[row] >= G::kIndexTopk )
				status = SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_CAPACITY_EXCEEDED,
					"the indexer is not built: a row may see at most the indexer's top-k keys, where attention is dense");
		return status;
	}

	static SparkStatus HyperBegin(State *state, const Frame &frame)
	{
		SparkHyperParams params;
		memset(&params, 0, sizeof(params));
		params.rows = frame.t.rows;
		params.streams = G::kStreams;
		params.width = G::kHidden;
		return SparkOpsHyperExpand(state->stage.ops, frame.queue, frame.t.hidden, state->streams, &params);
	}

	static SparkStatus HyperNorm(State *state, const Frame &frame, const Layer &layer, HyHyperPart part)
	{
		const Hyper &hyper = layer.hyper[part];
		const SparkHcParams params = HcParams(frame, HD_MIXES);
		SparkNormParams norm;
		SparkStatus status = SparkOpsHcMix(state->stage.ops, frame.queue, state->streams, hyper.fn, hyper.base, hyper.scale, state->post, state->mixed, &params);
		memset(&norm, 0, sizeof(norm));
		norm.rows = frame.t.rows;
		norm.dimension = norm.input_stride = norm.output_stride = G::kHidden;
		norm.epsilon = G::kEpsilon;
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsRmsNormBf16(state->stage.ops, frame.queue, state->mixed, part == HY_ATTENTION_PART ? layer.input_norm : layer.post_norm,
				state->normed, &norm);
		return status;
	}

	static SparkStatus Attention(State *state, const Frame &frame, const Layer &layer, uint32_t index)
	{
		return SparkDeviceLatentAttention(&state->stage, frame.queue, &state->attention_shape, &state->rope, &frame.t, &layer.attention, index,
			&state->attention, state->normed, state->partial);
	}

	static SparkStatus AllReduce(State *state, const Frame &frame)
	{
		return SparkDeviceStageAllReduce(&state->stage, frame.queue, state->partial, frame.t.rows);
	}

	static SparkStatus HyperPost(State *state, const Frame &frame)
	{
		const SparkHcParams params = HcParams(frame, HD_MIXES);
		return SparkOpsHcPost(state->stage.ops, frame.queue, state->streams, state->post, state->partial, &params);
	}

	static SparkStatus Dense(State *state, const Frame &frame, const Layer &layer)
	{
		return SparkDeviceGatedFfn(&state->stage, frame.queue, state->ffn_rows, SPARK_GATED_SILU, 0.0f, &frame.t, &layer.ffn, &state->ffn_scratch,
			state->normed, state->partial);
	}

	static SparkStatus Experts(State *state, const Frame &frame, const Layer &layer)
	{
		const uint32_t rows = frame.t.rows;
		const SparkDeviceRoutedFfnShape ffn = { G::kLocalExperts, G::kTopK, G::kMoeInter, SPARK_GATED_SILU_GATE_LIMIT, 1u, G::kSwigluLimit, 0.0f, 0u };
		SparkGemvParams router;
		SparkRouteParams route;
		SparkStatus status;
		memset(&router, 0, sizeof(router));
		router.rows = rows;
		router.input_dimension = router.input_stride = G::kHidden;
		router.output_dimension = router.output_stride = G::kExperts;
		router.output_f32 = 1u;
		router.projection = state->stage.shape->projection;
		router.row_exact = state->stage.step_exact;
		status = SparkOpsLinear(state->stage.ops, frame.queue, state->normed, layer.router, state->router_logits, &router);
		memset(&route, 0, sizeof(route));
		route.rows = rows;
		route.experts = G::kExperts;
		route.top_k = G::kTopK;
		route.width = G::kHidden;
		route.logit_stride = G::kExperts;
		route.scale = G::kRouteScale;
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsRouteSigmoid(state->stage.ops, frame.queue, state->router_logits, layer.router_bias, state->experts.experts, state->experts.weights,
				&route);
		route.local_experts = G::kLocalExperts;
		route.first_expert = state->stage.tp_rank * G::kLocalExperts;
		if ( status == SPARK_STATUS_OK && G::kTpDegree > 1u )
			status = SparkOpsRouteLocal(state->stage.ops, frame.queue, state->experts.experts, &route);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceRoutedFfn(&state->stage, frame.queue, &ffn, rows, &layer.gate, &layer.up, &layer.down, &state->experts, state->normed);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceGatedFfn(&state->stage, frame.queue, state->shared_rows, SPARK_GATED_SILU, 0.0f, &frame.t, &layer.shared, &state->ffn_scratch,
				state->normed, state->shared_out);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsRouteCombineShared(state->stage.ops, frame.queue, state->experts.out, state->shared_out, state->partial, &route);
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
		SparkDeviceStageLayerDone(&state->stage, index);
		return SparkDeviceStageDumpRows(&state->stage, frame.queue, &frame.t, state->streams, L::kResidual, index);
	}

	static SparkStatus HyperEnd(State *state, const Frame &frame)
	{
		const SparkHcParams params = HcParams(frame, G::kStreams);
		return SparkOpsHcMix(state->stage.ops, frame.queue, state->streams, state->head.fn, state->head.base, state->head.scale, state->post, frame.t.hidden,
			&params);
	}

	static SparkStatus Step(void *model, const SparkStageRunnerStep *step, void *queue)
	{
		return HySchedule<G, HyDevice>::Step((State *)model, step, queue);
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
struct Hy4Device
{
	using Geometry = Hy4Geometry<TP>;
	static constexpr const char *kTag = "sparkpipe_hy4_device";
	static constexpr const char *kName = "hy4-preview";
	static constexpr uint32_t kProjection = SPARK_PROJECTION_GEMV;
};

template<template<uint32_t> class P>
static const SparkStageRunnerModelInterface *HyDeviceAt(uint32_t tp_degree)
{
	switch ( tp_degree )
	{
	case 1u:
		return HyDevice<P<1u>>::Interface();
	case 2u:
		return HyDevice<P<2u>>::Interface();
	case 4u:
		return HyDevice<P<4u>>::Interface();
	case 8u:
		return HyDevice<P<8u>>::Interface();
	default:
		return 0;
	}
}

const SparkStageRunnerModelInterface *SparkHy4DeviceStageModelAt(uint32_t tp_degree)
{
	return HyDeviceAt<Hy4Device>(tp_degree);
}
