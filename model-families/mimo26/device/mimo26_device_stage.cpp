#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inference/runner/device_layers.h"
#include "sparkpipe/spark_mimo26_device_stage.h"
#include "sparkpipe/spark_mimo26_family.h"
#include "sparkpipe/spark_mimo26_flash_tp16_stage_model.h"

#define MD_SLICES 24u

template<class P>
struct MimoDevice
{
	using G = typename P::Geometry;
	using L = MimoLayout<G>;

	static_assert(G::kTpDegree == 1u, "a device stage is one whole rank");

	struct Layer
	{
		SparkTensor input_norm;
		SparkTensor post_attn_norm;
		SparkDeviceAttentionWeights attention;
		SparkDeviceFfnWeights ffn;
		SparkTensor router;
		SparkTensor router_bias;
		SparkTensor moe_w1;
		SparkTensor moe_w1_scale;
		SparkTensor moe_w3;
		SparkTensor moe_w3_scale;
		SparkTensor moe_w2;
		SparkTensor moe_w2_scale;
		uint32_t index;
	};

	struct State
	{
		SparkDeviceStage stage;
		SparkDeviceStageShape shape;
		SparkDeviceAttentionShape sliding;
		SparkDeviceAttentionShape full;
		Layer layers[G::kLayers];
		SparkTensor normed;
		SparkTensor partial;
		SparkDeviceAttentionScratch attention;
		SparkDeviceFfnScratch ffn_scratch;
		SparkTensor router_logits;
		SparkTensor route_experts;
		SparkTensor route_weights;
		SparkTensor expert_gate_up;
		SparkTensor expert_act;
		SparkTensor expert_out;
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

	static void AttentionShape(SparkDeviceAttentionShape *attention, bool full)
	{
		memset(attention, 0, sizeof(*attention));
		attention->heads = G::kHeads;
		attention->kv_heads = full ? G::kFullKvHeads : G::kSlidingKvHeads;
		attention->head_dimension = G::kHeadDim;
		attention->value_dimension = G::kValueDim;
		attention->rope_dimension = G::kRopeDim;
		attention->rope_theta = full ? G::kFullTheta : G::kSlidingTheta;
		attention->epsilon = G::kEpsilon;
		attention->scale = G::kAttentionScale;
		attention->qk_plain = 1u;
		attention->value_scale = G::kValueScale;
		attention->sinks = full ? 0u : 1u;
		attention->window = full ? 0u : G::kWindow;
	}

	static void Shapes(State *state)
	{
		SparkDeviceStageShape *shape = &state->shape;
		const uint32_t kv_heads = G::kSlidingKvHeads > G::kFullKvHeads ? G::kSlidingKvHeads : G::kFullKvHeads;
		memset(shape, 0, sizeof(*shape));
		shape->tag = P::kTag;
		shape->hidden = G::kHidden;
		shape->vocab = G::kVocab;
		shape->layers = G::kLayers;
		shape->kv_layers = G::kLayers;
		shape->kv_heads = kv_heads;
		shape->kv_head_dimension = (G::kHeadDim + G::kValueDim) / 2u;
		shape->pack_alignment = MM_PACK_ALIGNMENT;
		shape->epsilon = G::kEpsilon;
		shape->tied_head = 0u;
		shape->projection = P::kProjection;
		AttentionShape(&state->sliding, false);
		AttentionShape(&state->full, true);
	}

	static uint32_t ScratchSlices(State *state, SparkDeviceStageSlice *slices)
	{
		const uint64_t rows = state->stage.max_rows, hidden = G::kHidden, routes = rows * G::kTopK;
		const uint64_t fused = Max(SparkDeviceAttentionFusedWidth(&state->sliding), SparkDeviceAttentionFusedWidth(&state->full));
		const uint64_t key = Max(SparkDeviceAttentionKvWidth(&state->sliding), SparkDeviceAttentionKvWidth(&state->full));
		const uint64_t value = Max(SparkDeviceAttentionValueWidth(&state->sliding), SparkDeviceAttentionValueWidth(&state->full));
		const uint64_t output = SparkDeviceAttentionOutputWidth(&state->full);
		const SparkDeviceStageSlice plan[] =
		{
			{ rows * hidden * 2u, &state->normed },
			{ rows * hidden * 2u, &state->partial },
			{ rows * fused * 2u, &state->attention.qkv },
			{ rows * SparkDeviceAttentionQueryWidth(&state->full) * 2u, &state->attention.query },
			{ rows * key * 2u, &state->attention.key },
			{ rows * value * 2u, &state->attention.value },
			{ SparkOpsAttentionScratchBytes(state->stage.max_rows, G::kHeads, G::kValueDim, state->stage.max_positions), &state->attention.records },
			{ rows * output * 2u, &state->attention.gated },
			{ rows * 2u * L::kDenseRows * 2u, &state->ffn_scratch.gate_up },
			{ rows * L::kDenseRows * 2u, &state->ffn_scratch.act },
			{ rows * G::kExperts * sizeof(float), &state->router_logits },
			{ routes * sizeof(uint32_t), &state->route_experts },
			{ routes * sizeof(float), &state->route_weights },
			{ routes * 2u * G::kMoeInter * 2u, &state->expert_gate_up },
			{ routes * G::kMoeInter * 2u, &state->expert_act },
			{ routes * hidden * 2u, &state->expert_out },
			{ Max(fused, 2u * L::kDenseRows) * hidden * 2u, &state->stage.dequant }
		};
		const uint32_t count = sizeof(plan) / sizeof(plan[0]);
		for ( uint32_t index = 0u; index < count; index++ )
			slices[index] = plan[index];
		return count;
	}

	static SparkStatus Open(const SparkStageRunnerModelOpen *request, void **out, SparkStageRunnerModelGeometry *geometry)
	{
		SparkDeviceStageSlice slices[MD_SLICES];
		State *state = (State *)calloc(1u, sizeof(State));
		SparkStatus status;
		*out = state;
		if ( state == 0 )
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		Shapes(state);
		const SparkDeviceStageConfigValue config[] =
		{
			{ "hidden", G::kHidden }, { "layers", G::kLayers }, { "vocab_rows", G::kVocab }, { "ffn_rows", L::kDenseRows },
			{ "sliding_window", G::kWindow }, { "experts", G::kExperts }, { "top_k", G::kTopK }
		};
		status = SparkDeviceStageOpen(&state->stage, &state->shape, request);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageExpectConfig(&state->stage, config, sizeof(config) / sizeof(config[0]));
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageReserve(&state->stage, SparkDeviceStagePlan(0, slices, ScratchSlices(state, slices)), 0, 0u,
				request->configuration->state_budget_bytes);
		if ( status != SPARK_STATUS_OK )
			return status;
		SparkDeviceAttentionRopePrepare(&state->sliding, &state->sliding_rope);
		SparkDeviceAttentionRopePrepare(&state->full, &state->full_rope);
		return SparkDeviceStageGeometry(&state->stage, geometry);
	}

	static SparkStatus BindExpert(State *state, const SparkStageRunnerDeviceWeights *weights, uint32_t index, const char *name, uint64_t rows,
		uint64_t columns, SparkTensor *values, SparkTensor *scale)
	{
		char scale_name[64];
		SparkStatus status = SparkDeviceStageLayerTensor(&state->stage, weights, index, name, "mxfp4_e2m1", G::kExperts * rows * (columns / 2u), values);
		snprintf(scale_name, sizeof(scale_name), "%s.scale", name);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, scale_name, "e8m0", G::kExperts * rows * (columns / SPARK_MXFP4_GROUP), scale);
		if ( status == SPARK_STATUS_OK )
			SparkDeviceStageNoteRouted(&state->stage, values->bytes + scale->bytes, G::kExperts, G::kTopK);
		return status;
	}

	static SparkStatus BindLayer(State *state, const SparkStageRunnerDeviceWeights *weights, uint32_t index)
	{
		Layer *layer = &state->layers[index];
		SparkStatus status;
		layer->index = index;
		status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "input_norm", "bf16", G::kHidden * 2u, &layer->input_norm);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "post_attn_norm", "bf16", G::kHidden * 2u, &layer->post_attn_norm);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceAttentionBind(&state->stage, weights, index, L::Full(index) ? &state->full : &state->sliding, &layer->attention);
		if ( status == SPARK_STATUS_OK && index < G::kDenseLayers )
			return SparkDeviceFfnBind(&state->stage, weights, index, L::kDenseRows, &layer->ffn);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "router", "bf16", (uint64_t)G::kExperts * G::kHidden * 2u, &layer->router);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "router_bias", "f32", (uint64_t)G::kExperts * sizeof(float), &layer->router_bias);
		if ( status == SPARK_STATUS_OK )
			status = BindExpert(state, weights, index, "moe_w1", G::kMoeInter, G::kHidden, &layer->moe_w1, &layer->moe_w1_scale);
		if ( status == SPARK_STATUS_OK )
			status = BindExpert(state, weights, index, "moe_w3", G::kMoeInter, G::kHidden, &layer->moe_w3, &layer->moe_w3_scale);
		if ( status == SPARK_STATUS_OK )
			status = BindExpert(state, weights, index, "moe_w2", G::kHidden, G::kMoeInter, &layer->moe_w2, &layer->moe_w2_scale);
		return status;
	}

	static SparkStatus BindDevice(void *model, const SparkStageRunnerDeviceWeights *weights, SparkStageRunnerModelGeometry *geometry)
	{
		State *state = (State *)model;
		SparkDeviceStageSlice slices[MD_SLICES];
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

	static SparkStatus RmsNorm(State *state, const Frame &frame, SparkTensor weight)
	{
		SparkNormParams params;
		memset(&params, 0, sizeof(params));
		params.rows = frame.t.rows;
		params.dimension = params.input_stride = params.output_stride = G::kHidden;
		params.epsilon = G::kEpsilon;
		return SparkOpsRmsNormBf16(state->stage.ops, frame.queue, frame.t.hidden, weight, state->normed, &params);
	}

	template<bool FULL>
	static SparkStatus Attention(State *state, const Frame &frame, const Layer &layer)
	{
		return SparkDeviceAttention(&state->stage, frame.queue, FULL ? &state->full : &state->sliding, FULL ? &state->full_rope : &state->sliding_rope,
			&frame.t, &layer.attention, layer.index, &state->attention, state->normed, state->partial);
	}

	static SparkStatus AllReduce(State *state, const Frame &frame)
	{
		(void)state;
		(void)frame;
		return SPARK_STATUS_OK;
	}

	static SparkStatus AddResidual(State *state, const Frame &frame)
	{
		return SparkOpsAddRows(state->stage.ops, frame.queue, frame.t.hidden, state->partial, frame.t.rows, G::kHidden, G::kHidden, G::kHidden);
	}

	static SparkStatus Dense(State *state, const Frame &frame, const Layer &layer)
	{
		return SparkDeviceGatedFfn(&state->stage, frame.queue, L::kDenseRows, SPARK_GATED_SILU_ROUNDED, &frame.t, &layer.ffn, &state->ffn_scratch,
			state->normed, state->partial);
	}

	static SparkStatus Experts(State *state, const Frame &frame, const Layer &layer)
	{
		const uint32_t rows = frame.t.rows, routes = rows * G::kTopK, inter = G::kMoeInter;
		SparkGemvParams router, up, down;
		SparkRouteParams route;
		SparkSiluMulParams act;
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
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsRouteSigmoid(state->stage.ops, frame.queue, state->router_logits, layer.router_bias, state->route_experts,
				state->route_weights, &route);
		memset(&up, 0, sizeof(up));
		up.rows = routes;
		up.input_dimension = up.input_stride = G::kHidden;
		up.output_dimension = inter;
		up.output_stride = 2u * inter;
		up.route_divisor = G::kTopK;
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsLinearRoutedMxfp4(state->stage.ops, frame.queue, state->normed, layer.moe_w1, layer.moe_w1_scale, state->expert_gate_up,
				state->route_experts, G::kExperts, &up);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsLinearRoutedMxfp4(state->stage.ops, frame.queue, state->normed, layer.moe_w3, layer.moe_w3_scale,
				SparkTensorSlice(state->expert_gate_up, (uint64_t)inter * 2u, state->expert_gate_up.bytes - (uint64_t)inter * 2u), state->route_experts,
				G::kExperts, &up);
		memset(&act, 0, sizeof(act));
		act.rows = routes;
		act.dimension = inter;
		act.input_stride = 2u * inter;
		act.output_stride = inter;
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsGatedMul(state->stage.ops, frame.queue, state->expert_gate_up, state->expert_act, &act, SPARK_GATED_SILU_ROUNDED);
		memset(&down, 0, sizeof(down));
		down.rows = routes;
		down.input_dimension = down.input_stride = inter;
		down.output_dimension = down.output_stride = G::kHidden;
		down.route_divisor = 1u;
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsLinearRoutedMxfp4(state->stage.ops, frame.queue, state->expert_act, layer.moe_w2, layer.moe_w2_scale, state->expert_out,
				state->route_experts, G::kExperts, &down);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsRouteCombineWeighted(state->stage.ops, frame.queue, state->expert_out, state->route_weights, state->partial, &route);
		return status;
	}

	static SparkStatus LayerDone(State *state, const Frame &frame, uint32_t index)
	{
		SparkDeviceStageLayerDone(&state->stage, index);
		return SparkDeviceStageDumpRows(&state->stage, frame.queue, &frame.t, frame.t.hidden, G::kHidden, index);
	}

	static SparkStatus Step(void *model, const SparkStageRunnerStep *step, void *queue)
	{
		return MimoSchedule<G, MimoDevice>::Step((State *)model, step, queue);
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

struct Mimo26FlashDevice
{
	using Geometry = Mimo26FlashGeometry<1u>;
	static constexpr const char *kTag = "sparkpipe_mimo26_flash_device";
	static constexpr const char *kName = "mimo-v2.6-flash";
	static constexpr uint32_t kProjection = SPARK_PROJECTION_ROW_TILES;
};

const SparkStageRunnerModelInterface *SparkMimo26FlashDeviceStageModel(void)
{
	return MimoDevice<Mimo26FlashDevice>::Interface();
}
