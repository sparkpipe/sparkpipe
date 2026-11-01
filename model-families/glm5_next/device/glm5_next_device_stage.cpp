#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inference/runner/device_layers.h"
#include "sparkpipe/spark_glm5_next_device_stage.h"
#include "sparkpipe/spark_glm5_next_family.h"
#include "sparkpipe/spark_glm53_flash_tp16_stage_model.h"

#define GD_SLICES 40u
#define GD_ALIGNMENT 128u

template<class P>
struct Glm5NextDevice
{
	using G = typename P::Geometry;
	using L = Glm5NextLayout;

	static constexpr uint32_t kTp = G::kTpDegree;
	static constexpr uint32_t kHidden = SPARK_GLM5_NEXT_MODEL_HIDDEN_DIMENSION;
	static constexpr uint32_t kLayers = SPARK_GLM5_NEXT_MODEL_LAYER_COUNT;
	static constexpr uint32_t kVocab = SPARK_GLM5_NEXT_MODEL_OUTPUT_VOCAB_COUNT;
	static constexpr uint32_t kKdaLayers = SPARK_GLM5_NEXT_MODEL_KDA_LAYER_COUNT;
	static constexpr uint32_t kMlaLayers = SPARK_GLM5_NEXT_MODEL_DSA_LAYER_COUNT;
	static constexpr uint32_t kKdaHeads = SPARK_GLM5_NEXT_MODEL_KDA_HEAD_COUNT / kTp;
	static constexpr uint32_t kKdaDim = SPARK_GLM5_NEXT_MODEL_KDA_HEAD_KEY_DIMENSION;
	static constexpr uint32_t kKdaWidth = kKdaHeads * kKdaDim;
	static constexpr uint32_t kKdaRank = SPARK_GLM5_NEXT_MODEL_KDA_LOW_RANK_GATE_BOTTLENECK;
	static constexpr uint32_t kMlaHeads = SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT / kTp;
	static constexpr uint32_t kNope = SPARK_GLM5_NEXT_MODEL_MLA_QK_NOPE_HEAD_DIMENSION;
	static constexpr uint32_t kLatent = SPARK_GLM5_NEXT_MODEL_MLA_LATENT_DIMENSION;
	static constexpr uint32_t kValueDim = SPARK_GLM5_NEXT_MODEL_MLA_VALUE_HEAD_DIMENSION;
	static constexpr uint32_t kQueryLora = SPARK_GLM5_NEXT_MODEL_MLA_QUERY_A_DIMENSION;
	static constexpr uint32_t kExperts = SPARK_GLM5_NEXT_MODEL_MOE_EXPERT_COUNT;
	static constexpr uint32_t kTopK = SPARK_GLM5_NEXT_MODEL_MOE_TOP_K;
	static constexpr uint32_t kMoeRows = SPARK_GLM5_NEXT_MODEL_MOE_INTERMEDIATE_DIMENSION / kTp;
	static constexpr uint32_t kDenseRows = SPARK_GLM5_NEXT_MODEL_DENSE_INTERMEDIATE_DIMENSION / kTp;
	static constexpr uint32_t kStreams = SPARK_GLM5_NEXT_MODEL_HC_MULT;
	static constexpr uint32_t kResidual = kStreams * kHidden;
	static constexpr uint32_t kDenseKeys = SPARK_GLM5_NEXT_MODEL_INDEX_TOP_K;
	static constexpr uint32_t kScaleBlock = SPARK_GLM5_NEXT_MODEL_FP8_SCALE_BLOCK;
	static constexpr float kEpsilon = SPARK_GLM5_NEXT_MODEL_RMS_NORM_EPSILON;

	struct Hyper
	{
		SparkTensor fn;
		SparkTensor base;
		SparkTensor scale;
	};

	struct Layer
	{
		Hyper hyper[2];
		SparkTensor norm[2];
		SparkDeviceChannelDeltaWeights delta;
		SparkDeviceLatentAttentionWeights attention;
		SparkDeviceFfnWeights ffn;
		SparkDeviceFfnWeights shared;
		SparkTensor router;
		SparkTensor router_bias;
		SparkDeviceExpertMatrix gate;
		SparkDeviceExpertMatrix up;
		SparkDeviceExpertMatrix down;
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
		Layer layers[kLayers];
		SparkTensor final_norm;
		SparkTensor streams;
		SparkTensor post;
		SparkTensor comb;
		SparkTensor mixed;
		SparkTensor normed;
		SparkTensor partial;
		SparkTensor shared_out;
		SparkTensor router_logits;
		SparkDeviceChannelDeltaScratch delta;
		SparkDeviceLatentAttentionScratch attention;
		SparkDeviceFfnScratch ffn_scratch;
		SparkDeviceExpertsScratch experts;
	};

	struct Frame
	{
		SparkDeviceStepTensors t;
		SparkDeviceQueue *queue;
		const Layer *layer;
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
		memset(shape, 0, sizeof(*shape));
		shape->tag = P::kTag;
		shape->hidden = kHidden;
		shape->vocab = kVocab;
		shape->layers = kLayers;
		shape->kv_layers = kMlaLayers;
		shape->kv_heads = 1u;
		shape->kv_head_dimension = kLatent;
		shape->pack_alignment = GD_ALIGNMENT;
		shape->epsilon = kEpsilon;
		shape->projection = P::kProjection;
		shape->prenormed_head = 1u;
		memset(delta, 0, sizeof(*delta));
		delta->heads = kKdaHeads;
		delta->dimension = kKdaDim;
		delta->conv_taps = SPARK_GLM5_NEXT_MODEL_KDA_CONV_KERNEL;
		delta->low_rank = kKdaRank;
		delta->epsilon = kEpsilon;
		delta->l2_epsilon = 1e-6f;
		delta->minimum_log_decay = SPARK_GLM5_NEXT_MODEL_KDA_GATE_LOWER_BOUND;
		delta->beta_mode = SPARK_DELTA_BETA_ROUNDED;
		delta->norm_f32 = 1u;
		memset(attention, 0, sizeof(*attention));
		attention->heads = SPARK_GLM5_NEXT_MODEL_MLA_HEAD_COUNT;
		attention->nope = kNope;
		attention->rope = SPARK_GLM5_NEXT_MODEL_MLA_QK_ROPE_HEAD_DIMENSION;
		attention->latent = kLatent;
		attention->value_dimension = kValueDim;
		attention->rope_theta = 1.0f;
		attention->epsilon = kEpsilon;
		attention->scale = SPARK_GLM5_NEXT_MODEL_MLA_QK_SCALE;
		attention->query_lora = kQueryLora;
		attention->norm_bf16 = 1u;
	}

	static uint32_t ScratchSlices(State *state, SparkDeviceStageSlice *slices)
	{
		const uint64_t rows = state->stage.max_rows, hidden = kHidden, routes = rows * kTopK, heads = kMlaHeads, width = kKdaWidth;
		const uint64_t ffn = Max(kDenseRows, kMoeRows);
		const SparkDeviceStageSlice plan[] =
		{
			{ rows * kResidual * 2u, &state->streams },
			{ rows * kStreams * sizeof(float), &state->post },
			{ rows * kStreams * kStreams * sizeof(float), &state->comb },
			{ rows * hidden * 2u, &state->mixed },
			{ rows * hidden * 2u, &state->normed },
			{ rows * hidden * 2u, &state->partial },
			{ rows * hidden * 2u, &state->shared_out },
			{ rows * kExperts * sizeof(float), &state->router_logits },
			{ rows * 3u * width * 2u, &state->delta.qkv },
			{ rows * kKdaHeads * 2u, &state->delta.beta },
			{ rows * Max(kKdaRank, 64u) * 2u, &state->delta.low },
			{ rows * width * 2u, &state->delta.decay },
			{ rows * width * 2u, &state->delta.gate },
			{ SparkOpsChannelDeltaScratchBytes(state->stage.max_rows, kKdaHeads, kKdaDim), &state->delta.delta },
			{ rows * width * 2u, &state->delta.core },
			{ rows * width * 2u, &state->delta.gated },
			{ rows * kQueryLora * 2u, &state->attention.query_lora },
			{ rows * heads * kNope * 2u, &state->attention.query },
			{ rows * kLatent * 2u, &state->attention.kv },
			{ rows * heads * kLatent * 2u, &state->attention.packed },
			{ SparkOpsAttentionScratchBytes(state->stage.max_rows, kMlaHeads, kLatent, state->stage.max_positions), &state->attention.records },
			{ rows * heads * kLatent * 2u, &state->attention.latent },
			{ rows * heads * kValueDim * 2u, &state->attention.value },
			{ rows * 2u * ffn * 2u, &state->ffn_scratch.gate_up },
			{ rows * ffn * 2u, &state->ffn_scratch.act },
			{ routes * sizeof(uint32_t), &state->experts.experts },
			{ routes * sizeof(float), &state->experts.weights },
			{ routes * 2u * kMoeRows * 2u, &state->experts.gate_up },
			{ routes * kMoeRows * 2u, &state->experts.act },
			{ routes * hidden * 2u, &state->experts.out },
			{ (routes + 1u) * hidden * 2u, &state->experts.sorted },
			{ Max(2u * ffn, Max((uint64_t)kQueryLora, Max(heads * kNope, Max(3u * width, hidden)))) * Max(hidden, (uint64_t)kQueryLora) * 2u,
				&state->stage.dequant }
		};
		const uint32_t count = sizeof(plan) / sizeof(plan[0]);
		for ( uint32_t index = 0u; index < count; index++ )
			slices[index] = plan[index];
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
		const SparkDeviceStageRecurrentPlan pools[] = { { kKdaLayers, SparkDeviceChannelDeltaStateBytes(&state->delta_shape) },
			{ kKdaLayers, SparkDeviceChannelDeltaWindowBytes(&state->delta_shape) } };
		const SparkDeviceStageConfigValue config[] =
		{
			{ "hidden", kHidden }, { "layers", kLayers }, { "vocab", kVocab }, { "vocab_rows", kVocab / kTp }, { "experts", kExperts },
			{ "local_experts", kExperts }, { "top_k", kTopK }, { "ffn_rows", kDenseRows }, { "shared_rows", kMoeRows }, { "moe_rows", kMoeRows }
		};
		status = SparkDeviceStageOpen(&state->stage, &state->shape, request);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageExpectConfig(&state->stage, config, sizeof(config) / sizeof(config[0]));
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageReserve(&state->stage, SparkDeviceStagePlan(0, slices, ScratchSlices(state, slices)), pools,
				SPARK_DEVICE_GATED_DELTA_POOLS, request->configuration->state_budget_bytes);
		if ( status != SPARK_STATUS_OK )
			return status;
		SparkDeviceLatentAttentionRopePrepare(&state->attention_shape, kMlaHeads, &state->rope);
		return SparkDeviceStageGeometry(&state->stage, geometry);
	}

	static SparkStatus BindHyper(State *state, const SparkStageRunnerDeviceWeights *weights, uint32_t index, const char *prefix, Hyper *hyper)
	{
		char name[48];
		snprintf(name, sizeof(name), "%s_fn", prefix);
		SparkStatus status = SparkDeviceStageLayerTensor(&state->stage, weights, index, name, "f32",
			(uint64_t)SPARK_HC_COMB_MIXES * kResidual * sizeof(float), &hyper->fn);
		snprintf(name, sizeof(name), "%s_base", prefix);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, name, "f32", SPARK_HC_COMB_MIXES * sizeof(float), &hyper->base);
		snprintf(name, sizeof(name), "%s_scale", prefix);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, name, "f32", SPARK_GLM5_NEXT_MODEL_HC_SCALE_COUNT * sizeof(float),
				&hyper->scale);
		return status;
	}

	static SparkStatus BindExpert(State *state, const SparkStageRunnerDeviceWeights *weights, uint32_t index, const char *name, uint64_t rows,
		uint64_t columns, SparkDeviceExpertMatrix *matrix)
	{
		char scale_name[48];
		SparkStatus status = SparkDeviceStageLayerTensor(&state->stage, weights, index, name, "fp8", kExperts * rows * columns, &matrix->values);
		matrix->codec = SPARK_WEIGHT_CODEC_FP8_E4M3;
		matrix->scale_group = 0u;
		snprintf(scale_name, sizeof(scale_name), "%s_scale", name);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, scale_name, "f32", kExperts * rows * (columns / kScaleBlock) * sizeof(float),
				&matrix->scale);
		if ( status == SPARK_STATUS_OK )
			SparkDeviceStageNoteRouted(&state->stage, matrix->values.bytes + matrix->scale.bytes, kExperts, kTopK);
		return status;
	}

	static SparkStatus BindLayerWeights(State *state, const SparkStageRunnerDeviceWeights *weights, uint32_t index, uint32_t *recurrent, uint32_t *kv)
	{
		Layer *layer = &state->layers[index];
		SparkStatus status = BindHyper(state, weights, index, "attn_hc", &layer->hyper[GLM5_NEXT_ATTENTION_PART]);
		if ( status == SPARK_STATUS_OK )
			status = BindHyper(state, weights, index, "ffn_hc", &layer->hyper[GLM5_NEXT_FFN_PART]);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "attn_norm", "bf16", kHidden * 2u, &layer->norm[GLM5_NEXT_ATTENTION_PART]);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "post_norm", "bf16", kHidden * 2u, &layer->norm[GLM5_NEXT_FFN_PART]);
		if ( status == SPARK_STATUS_OK && L::Kda(index) )
		{
			layer->recurrent_index = (*recurrent)++;
			status = SparkDeviceChannelDeltaBind(&state->stage, weights, index, &state->delta_shape, &layer->delta);
		}
		else if ( status == SPARK_STATUS_OK )
		{
			layer->kv_index = (*kv)++;
			status = SparkDeviceLatentAttentionBind(&state->stage, weights, index, &state->attention_shape, &layer->attention);
		}
		if ( status == SPARK_STATUS_OK && !L::Routed(index) )
			return SparkDeviceFfnBind(&state->stage, weights, index, kDenseRows, &layer->ffn);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "router", "bf16", (uint64_t)kExperts * kHidden * 2u, &layer->router);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "router_bias", "f32", (uint64_t)kExperts * sizeof(float), &layer->router_bias);
		if ( status == SPARK_STATUS_OK )
			status = BindExpert(state, weights, index, "moe_w1", kMoeRows, kHidden, &layer->gate);
		if ( status == SPARK_STATUS_OK )
			status = BindExpert(state, weights, index, "moe_w3", kMoeRows, kHidden, &layer->up);
		if ( status == SPARK_STATUS_OK )
			status = BindExpert(state, weights, index, "moe_w2", kHidden, kMoeRows, &layer->down);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceFfnBindNamed(&state->stage, weights, index, "shared_gate_up", "shared_down", kMoeRows, &layer->shared);
		return status;
	}

	static SparkStatus BindDevice(void *model, const SparkStageRunnerDeviceWeights *weights, SparkStageRunnerModelGeometry *geometry)
	{
		State *state = (State *)model;
		SparkDeviceStageSlice slices[GD_SLICES];
		uint32_t recurrent = 0u, kv = 0u;
		SparkStatus status;
		(void)geometry;
		if ( weights == 0 || weights->pack_address == 0u || weights->pack_bytes != state->stage.pack_bytes )
			return SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_INVALID_ARGUMENT, "bind_device needs the device mapping of the whole rank pack");
		status = SparkDeviceStageTensor(&state->stage, weights, "final_norm", "bf16", kHidden * 2u, &state->final_norm);
		for ( uint32_t index = 0u; index < kLayers && status == SPARK_STATUS_OK; index++ )
			status = BindLayerWeights(state, weights, index, &recurrent, &kv);
		if ( status != SPARK_STATUS_OK )
			return status;
		return SparkDeviceStageAllocate(&state->stage, slices, ScratchSlices(state, slices));
	}

	static SparkHcParams HcParams(const Frame &frame)
	{
		SparkHcParams params;
		memset(&params, 0, sizeof(params));
		params.rows = frame.t.rows;
		params.streams = kStreams;
		params.width = kHidden;
		params.mixes = SPARK_HC_COMB_MIXES;
		params.norm_epsilon = kEpsilon;
		params.hc_epsilon = SPARK_GLM5_NEXT_MODEL_HC_EPSILON;
		params.post_magnitude = 2.0f;
		params.iterations = SPARK_GLM5_NEXT_MODEL_HC_SINKHORN_ITERATIONS;
		return params;
	}

	static SparkNormParams NormParams(const Frame &frame)
	{
		SparkNormParams norm;
		memset(&norm, 0, sizeof(norm));
		norm.rows = frame.t.rows;
		norm.dimension = norm.input_stride = norm.output_stride = kHidden;
		norm.epsilon = kEpsilon;
		return norm;
	}

	static SparkStatus Begin(State *state, const SparkStageRunnerStep *step, void *queue, Frame *frame)
	{
		frame->queue = (SparkDeviceQueue *)queue;
		frame->layer = 0;
		SparkStatus status = SparkDeviceStageStepTables(&state->stage, step, &frame->t);
		for ( uint32_t row = 0u; status == SPARK_STATUS_OK && row < frame->t.rows; row++ )
			if ( frame->t.host_positions == 0 || frame->t.host_positions[row] >= kDenseKeys )
				status = SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_CAPACITY_EXCEEDED,
					"the sparse-attention indexer is not built: a row may see at most the indexer's selected keys, where attention is dense");
		return status;
	}

	static SparkStatus HyperBegin(State *state, Frame &frame)
	{
		SparkHyperParams params;
		memset(&params, 0, sizeof(params));
		params.rows = frame.t.rows;
		params.streams = kStreams;
		params.width = kHidden;
		return SparkOpsHyperExpand(state->stage.ops, frame.queue, frame.t.hidden, state->streams, &params);
	}

	static void BindLayer(State *state, Frame &frame, uint32_t layer)
	{
		frame.layer = &state->layers[layer];
	}

	static SparkStatus HyperNorm(State *state, Frame &frame, uint32_t layer, Glm5NextPart part)
	{
		const Hyper &hyper = frame.layer->hyper[part];
		const SparkHcParams params = HcParams(frame);
		const SparkNormParams norm = NormParams(frame);
		(void)layer;
		SparkStatus status = SparkOpsHcSite(state->stage.ops, frame.queue, state->streams, hyper.fn, hyper.base, hyper.scale, state->post, state->comb,
			state->mixed, &params);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsRmsNormBf16(state->stage.ops, frame.queue, state->mixed, frame.layer->norm[part], state->normed, &norm);
		return status;
	}

	static SparkStatus Kda(State *state, Frame &frame, uint32_t layer)
	{
		(void)layer;
		return SparkDeviceChannelDelta(&state->stage, frame.queue, &state->delta_shape, &frame.t, &frame.layer->delta, frame.layer->recurrent_index,
			&state->delta, state->normed, state->partial);
	}

	static SparkStatus Attention(State *state, Frame &frame, uint32_t layer)
	{
		(void)layer;
		return SparkDeviceLatentAttention(&state->stage, frame.queue, &state->attention_shape, &state->rope, &frame.t, &frame.layer->attention,
			frame.layer->kv_index, &state->attention, state->normed, state->partial);
	}

	static SparkStatus AllReduce(State *state, Frame &frame)
	{
		return SparkDeviceStageAllReduce(&state->stage, frame.queue, state->partial, frame.t.rows);
	}

	static SparkStatus HyperPost(State *state, Frame &frame, uint32_t layer, Glm5NextPart part)
	{
		const SparkHcParams params = HcParams(frame);
		SparkStatus status = SparkOpsHcPostComb(state->stage.ops, frame.queue, state->streams, state->post, state->comb, state->partial, &params);
		if ( status != SPARK_STATUS_OK || part != GLM5_NEXT_FFN_PART )
			return status;
		SparkDeviceStageLayerDone(&state->stage, layer);
		return SparkDeviceStageDumpRows(&state->stage, frame.queue, &frame.t, state->streams, kResidual, layer);
	}

	static SparkStatus DenseMlp(State *state, Frame &frame, uint32_t layer)
	{
		(void)layer;
		return SparkDeviceGatedFfn(&state->stage, frame.queue, kDenseRows, SPARK_GATED_SILU_GATE_LIMIT, SPARK_GLM5_NEXT_MODEL_SWIGLU_LIMIT, &frame.t,
			&frame.layer->ffn, &state->ffn_scratch, state->normed, state->partial);
	}

	static SparkStatus RoutedMlp(State *state, Frame &frame, uint32_t layer)
	{
		const Layer &weights = *frame.layer;
		const uint32_t rows = frame.t.rows;
		const SparkDeviceRoutedFfnShape ffn = { kExperts, kTopK, kMoeRows, SPARK_GATED_SILU_GATE_LIMIT, 0u, SPARK_GLM5_NEXT_MODEL_SWIGLU_LIMIT, 0.0f, 0u };
		SparkGemvParams router;
		SparkRouteParams route;
		SparkStatus status;
		(void)layer;
		memset(&router, 0, sizeof(router));
		router.rows = rows;
		router.input_dimension = router.input_stride = kHidden;
		router.output_dimension = router.output_stride = kExperts;
		router.output_f32 = 1u;
		router.projection = state->stage.shape->projection;
		router.row_exact = state->stage.step_exact;
		status = SparkOpsLinear(state->stage.ops, frame.queue, state->normed, weights.router, state->router_logits, &router);
		memset(&route, 0, sizeof(route));
		route.rows = rows;
		route.experts = kExperts;
		route.top_k = kTopK;
		route.width = kHidden;
		route.logit_stride = kExperts;
		route.scale = SPARK_GLM5_NEXT_MODEL_MOE_ROUTED_SCALING_FACTOR;
		route.tree_sum = 1u;
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsRouteSigmoid(state->stage.ops, frame.queue, state->router_logits, weights.router_bias, state->experts.experts,
				state->experts.weights, &route);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceRoutedFfn(&state->stage, frame.queue, &ffn, rows, &weights.gate, &weights.up, &weights.down, &state->experts, state->normed);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsRouteCombineWeighted(state->stage.ops, frame.queue, state->experts.out, state->experts.weights, state->partial, &route);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceGatedFfn(&state->stage, frame.queue, kMoeRows, SPARK_GATED_SILU_GATE_LIMIT, SPARK_GLM5_NEXT_MODEL_SWIGLU_LIMIT, &frame.t,
				&weights.shared, &state->ffn_scratch, state->normed, state->shared_out);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsAddRows(state->stage.ops, frame.queue, state->partial, state->shared_out, rows, kHidden, kHidden, kHidden);
		return status;
	}

	static SparkStatus HyperEnd(State *state, Frame &frame)
	{
		const SparkHcParams params = HcParams(frame);
		const SparkNormParams norm = NormParams(frame);
		SparkStatus status = SparkOpsStreamMean(state->stage.ops, frame.queue, state->streams, state->mixed, &params);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsRmsNormBf16(state->stage.ops, frame.queue, state->mixed, state->final_norm, frame.t.hidden, &norm);
		return status;
	}

	static SparkStatus Step(void *model, const SparkStageRunnerStep *step, void *queue)
	{
		return Glm5NextSchedule<G, Glm5NextDevice>::Step((State *)model, step, queue);
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
struct Glm53FlashDevice
{
	using Geometry = Glm53FlashGeometry<TP>;
	static constexpr const char *kTag = "sparkpipe_glm53_flash_device";
	static constexpr const char *kName = "glm-5.3-flash";
	static constexpr uint32_t kProjection = SPARK_PROJECTION_GEMV;
};

template<template<uint32_t> class P>
static const SparkStageRunnerModelInterface *Glm5NextDeviceAt(uint32_t tp_degree)
{
	switch ( tp_degree )
	{
	case 1u:
		return Glm5NextDevice<P<1u>>::Interface();
	case 2u:
		return Glm5NextDevice<P<2u>>::Interface();
	case 4u:
		return Glm5NextDevice<P<4u>>::Interface();
	case 8u:
		return Glm5NextDevice<P<8u>>::Interface();
	default:
		return 0;
	}
}

const SparkStageRunnerModelInterface *SparkGlm53FlashDeviceStageModelAt(uint32_t tp_degree)
{
	return Glm5NextDeviceAt<Glm53FlashDevice>(tp_degree);
}
