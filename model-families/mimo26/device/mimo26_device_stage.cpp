#include "inference/runner/device_stage_cxx.h"
#include "sparkpipe/spark_mimo26_device_stage.h"
#include "sparkpipe/spark_mimo26_family.h"
#include "sparkpipe/spark_mimo26_flash_tp16_stage_model.h"
#include "sparkpipe/spark_mimo26_pro_tp16_stage_model.h"

template<class P>
struct MimoDevice : SparkDeviceStageModel<MimoDevice<P>>
{
	using Base = SparkDeviceStageModel<MimoDevice<P>>;
	using G = typename P::Geometry;
	using L = MimoLayout<G>;

	static constexpr bool kHasKeepRows = true;
	static constexpr bool kTreeVerify = true;
	static constexpr uint32_t kMaxSlices = 24u;
	static constexpr const char *Tag(void) { return P::kTag; }
	static constexpr const char *Name(void) { return P::kName; }

	struct Layer
	{
		SparkTensor input_norm;
		SparkTensor post_attn_norm;
		SparkDeviceAttentionWeights attention;
		SparkDeviceFfnWeights ffn;
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
		SparkDeviceAttentionShape sliding;
		SparkDeviceAttentionShape full;
		Layer layers[G::kLayers];
		SparkTensor normed;
		SparkTensor partial;
		SparkDeviceAttentionScratch attention;
		SparkDeviceFfnScratch ffn_scratch;
		SparkTensor router_logits;
		SparkDeviceExpertsScratch experts;
		SparkDeviceAttentionRope sliding_rope;
		SparkDeviceAttentionRope full_rope;
	};

	struct Frame
	{
		SparkDeviceStepTensors t;
		SparkDeviceQueue *queue;
	};

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
		attention->rope_rounding = SPARK_ROPE_ROUNDING_TERMS;
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
		const uint64_t fused = SparkDeviceStageMax(SparkDeviceAttentionFusedWidth(&state->sliding), SparkDeviceAttentionFusedWidth(&state->full));
		const uint64_t key = SparkDeviceStageMax(SparkDeviceAttentionKvWidth(&state->sliding), SparkDeviceAttentionKvWidth(&state->full));
		const uint64_t value = SparkDeviceStageMax(SparkDeviceAttentionValueWidth(&state->sliding), SparkDeviceAttentionValueWidth(&state->full));
		const uint64_t output = SparkDeviceAttentionOutputWidth(&state->full);
		const SparkDeviceStageSlice plan[] =
		{
			{ rows * hidden * 2u, &state->normed },
			{ rows * hidden * 2u, &state->partial },
			{ rows * fused * 2u, &state->attention.qkv },
			{ G::kTpDegree > 1u ? rows * fused / G::kTpDegree * 2u : 0u, &state->attention.qkv_local },
			{ G::kTpDegree > 1u ? rows * fused * 2u : 0u, &state->attention.qkv_gathered },
			{ rows * SparkDeviceAttentionQueryWidth(&state->full) * 2u, &state->attention.query },
			{ rows * key * 2u, &state->attention.key },
			{ rows * value * 2u, &state->attention.value },
			{ SparkOpsAttentionScratchBytes(state->stage.max_rows, G::kHeads, G::kValueDim, state->stage.max_positions), &state->attention.records },
			{ rows * output * 2u, &state->attention.gated },
			{ rows * 2u * L::kDenseRows * 2u, &state->ffn_scratch.gate_up },
			{ rows * L::kDenseRows * 2u, &state->ffn_scratch.act },
			{ rows * G::kExperts * sizeof(float), &state->router_logits },
			{ SparkDeviceStageMax(fused, 2u * L::kDenseRows) * hidden * 2u, &state->stage.dequant }
		};
		uint32_t count = SparkDeviceStageAppendSlices(slices, 0u, plan);
		return SparkDeviceStageAppendRoutedScratch(&state->experts, routes, G::kMoeInter, hidden, slices, count);
	}

	static SparkStatus Configure(State *state, const SparkStageRunnerModelOpen *request)
	{
		(void)request;
		const SparkDeviceStageConfigValue config[] =
		{
			{ "hidden", G::kHidden }, { "layers", G::kLayers }, { "vocab", G::kVocab }, { "vocab_rows", G::kVocab / G::kTpDegree },
			{ "ffn_rows", L::kDenseRows }, { "sliding_window", G::kWindow }, { "experts", G::kExperts }, { "local_experts", G::kLocalExperts },
			{ "top_k", G::kTopK }
		};
		return SparkDeviceStageExpectConfig(&state->stage, config, sizeof(config) / sizeof(config[0]));
	}

	static SparkStatus AfterReserve(State *state, const SparkStageRunnerModelOpen *request)
	{
		(void)request;
		SparkDeviceAttentionRopePrepare(&state->sliding, &state->sliding_rope);
		SparkDeviceAttentionRopePrepare(&state->full, &state->full_rope);
		return SPARK_STATUS_OK;
	}

	static SparkStatus BindLayer(State *state, const SparkStageRunnerDeviceWeights *weights, uint32_t index)
	{
		Layer *layer = &state->layers[index];
		layer->index = index;
		SparkStatus status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "input_norm", "bf16", G::kHidden * 2u, &layer->input_norm);
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
			status = SparkDeviceStageBindExpertMxfp4(&state->stage, weights, index, "moe_w1", ".", G::kLocalExperts, G::kTopK, G::kMoeInter, G::kHidden, SPARK_MXFP4_GROUP, &layer->gate);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageBindExpertMxfp4(&state->stage, weights, index, "moe_w3", ".", G::kLocalExperts, G::kTopK, G::kMoeInter, G::kHidden, SPARK_MXFP4_GROUP, &layer->up);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageBindExpertMxfp4(&state->stage, weights, index, "moe_w2", ".", G::kLocalExperts, G::kTopK, G::kHidden, G::kMoeInter, SPARK_MXFP4_GROUP, &layer->down);
		return status;
	}

	static SparkStatus BindWeights(State *state, const SparkStageRunnerDeviceWeights *weights)
	{
		SparkStatus status = SPARK_STATUS_OK;
		for ( uint32_t index = 0u; index < G::kLayers && status == SPARK_STATUS_OK; index++ )
			status = BindLayer(state, weights, index);
		return status;
	}

	static SparkStatus RmsNorm(State *state, const Frame &frame, SparkTensor weight)
	{
		const SparkNormParams params = SparkDeviceStageNormParams(frame.t.rows, G::kHidden, G::kEpsilon);
		return SparkOpsRmsNormBf16(state->stage.ops, frame.queue, frame.t.hidden, weight, state->normed, &params);
	}

	template<bool FULL>
	static SparkStatus Attention(State *state, const Frame &frame, const Layer &layer)
	{
		return SparkDeviceAttention(&state->stage, frame.queue, FULL ? &state->full : &state->sliding, FULL ? &state->full_rope : &state->sliding_rope,
			&frame.t, &layer.attention, layer.index, &state->attention, state->normed, state->partial);
	}

	static SparkStatus AddResidual(State *state, const Frame &frame)
	{
		return Base::AddRows(state, frame);
	}

	static SparkStatus Dense(State *state, const Frame &frame, const Layer &layer)
	{
		return SparkDeviceGatedFfn(&state->stage, frame.queue, L::kDenseRows, SPARK_GATED_SILU_ROUNDED, 0.0f, &frame.t, &layer.ffn, &state->ffn_scratch,
			state->normed, state->partial);
	}

	static SparkStatus Experts(State *state, const Frame &frame, const Layer &layer)
	{
		const uint32_t rows = frame.t.rows;
		const SparkDeviceRoutedFfnShape ffn = { G::kLocalExperts, G::kTopK, G::kMoeInter, SPARK_GATED_SILU_ROUNDED, 0u, 0.0f, 0.0f, 0u };
		SparkRouteParams route;
		SparkStatus status = SparkDeviceStageRouteSigmoid(&state->stage, frame.queue, state->normed, layer.router, layer.router_bias, state->router_logits,
			&state->experts, rows, G::kHidden, G::kExperts, G::kLocalExperts, G::kTopK, G::kTpDegree, 0.0f, 0u, 0u, 0u, 0u, G::kHidden, &route);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceRoutedFfn(&state->stage, frame.queue, &ffn, rows, &layer.gate, &layer.up, &layer.down, &state->experts, state->normed);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsRouteCombineWeighted(state->stage.ops, frame.queue, state->experts.out, state->experts.weights, state->partial, &route);
		return status;
	}

	static SparkStatus Step(void *model, const SparkStageRunnerStep *step, void *queue)
	{
		return MimoSchedule<G, MimoDevice>::Step((State *)model, step, queue);
	}

	static SparkStatus KeepRows(void *model, const SparkStageRunnerStep *step, const uint32_t *kept, void *queue)
	{
		State *state = (State *)model;
		return SparkDeviceStageKeepAttentionKv(&state->stage, queue, step, kept, G::kLayers,
			[state](uint32_t index) -> const SparkDeviceAttentionShape & { return L::Full(index) ? state->full : state->sliding; });
	}
};

template<uint32_t TP>
struct Mimo26FlashDevice
{
	using Geometry = Mimo26FlashGeometry<TP>;
	static constexpr const char *kTag = "sparkpipe_mimo26_flash_device";
	static constexpr const char *kName = "mimo-v2.6-flash";
	static constexpr uint32_t kProjection = SPARK_PROJECTION_ROW_TILES;
};

template<uint32_t TP>
struct Mimo26ProDevice
{
	using Geometry = Mimo26ProGeometry<TP>;
	static constexpr const char *kTag = "sparkpipe_mimo26_pro_device";
	static constexpr const char *kName = "mimo-v2.6-pro";
	static constexpr uint32_t kProjection = SPARK_PROJECTION_ROW_TILES;
};

const SparkStageRunnerModelInterface *SparkMimo26FlashDeviceStageModelAt(uint32_t tp_degree)
{
	return SparkDeviceStageAt<MimoDevice, Mimo26FlashDevice>(tp_degree);
}

const SparkStageRunnerModelInterface *SparkMimo26ProDeviceStageModelAt(uint32_t tp_degree)
{
	return SparkDeviceStageAt<MimoDevice, Mimo26ProDevice>(tp_degree);
}
