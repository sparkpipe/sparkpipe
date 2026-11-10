#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inference/runner/device_layers.h"
#include "sparkpipe/spark_motif_device_stage.h"
#include "sparkpipe/spark_motif_family.h"
#include "sparkpipe/spark_motif3_tp16_stage_model.h"

#define MD_SLICES 40u

template<class P>
struct MotifDevice
{
	using G = typename P::Geometry;
	using L = MotifLayout;

	static constexpr uint32_t kTp = G::kTpDegree;
	static constexpr uint32_t kHidden = SPARK_MOTIF_MODEL_HIDDEN_DIMENSION;
	static constexpr uint32_t kLayers = SPARK_MOTIF_MODEL_LAYER_COUNT;
	static constexpr uint32_t kVocab = SPARK_MOTIF_MODEL_VOCABULARY_SIZE;
	static constexpr uint32_t kGroup = SPARK_MOTIF_MODEL_HEAD_GROUP_SIZE;
	static constexpr uint32_t kGroups = SPARK_MOTIF_MODEL_NOISE_HEAD_COUNT / kTp;
	static constexpr uint32_t kHeads = kGroups * kGroup;
	static constexpr uint32_t kSignals = kGroups * (kGroup - 1u);
	static constexpr uint32_t kLambdas = SPARK_MOTIF_MODEL_SIGNAL_HEAD_COUNT;
	static constexpr uint32_t kNope = SPARK_MOTIF_MODEL_NOPE_DIMENSION;
	static constexpr uint32_t kRope = SPARK_MOTIF_MODEL_ROPE_DIMENSION;
	static constexpr uint32_t kLatent = SPARK_MOTIF_MODEL_KV_LORA_RANK;
	static constexpr uint32_t kSlot = SPARK_MOTIF_MODEL_LATENT_ROW_DIMENSION;
	static constexpr uint32_t kValueDim = SPARK_MOTIF_MODEL_VALUE_DIMENSION;
	static constexpr uint32_t kQueryLora = SPARK_MOTIF_MODEL_QUERY_LORA_RANK;
	static constexpr uint32_t kExperts = SPARK_MOTIF_MODEL_MOE_EXPERT_COUNT;
	static constexpr uint32_t kLocalExperts = kExperts / kTp;
	static constexpr uint32_t kTopK = SPARK_MOTIF_MODEL_MOE_TOP_K;
	static constexpr uint32_t kExpertRows = SPARK_MOTIF_MODEL_MOE_EXPERT_INTERMEDIATE;
	static constexpr uint32_t kDenseRows = SPARK_MOTIF_MODEL_DENSE_INTERMEDIATE;
	static constexpr uint32_t kStreams = SPARK_MOTIF_MODEL_HC_STREAMS;
	static constexpr uint32_t kResidual = kStreams * kHidden;
	static constexpr float kEpsilon = SPARK_MOTIF_MODEL_RMS_NORM_EPSILON;

	static_assert(SPARK_MOTIF_MODEL_NOISE_HEAD_COUNT % kTp == 0u && kExperts % kTp == 0u, "whole head groups and experts per rank");
	static_assert(kStreams == SPARK_HC_MAX_STREAMS && SPARK_MOTIF_MODEL_HC_MIX_DIMENSION == SPARK_HC_COMB_MIXES, "four mHC streams with 24 mixes");

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
		SparkDeviceLatentAttentionWeights attention;
		SparkDeviceFfnWeights ffn;
		SparkTensor ffn_poly;
		SparkTensor router;
		SparkTensor expert_bias;
		SparkDeviceExpertMatrix gate_up;
		SparkDeviceExpertMatrix down;
		SparkTensor expert_poly;
		SparkDeviceFfnWeights shared;
		SparkTensor shared_poly;
		uint32_t owner;
	};

	struct State
	{
		SparkDeviceStage stage;
		SparkDeviceStageShape shape;
		SparkDeviceLatentAttentionShape window_shape;
		SparkDeviceLatentAttentionShape full_shape;
		SparkDeviceLatentAttentionRope window_rope;
		SparkDeviceLatentAttentionRope full_rope;
		Layer layers[kLayers];
		SparkTensor head_norm;
		SparkTensor streams;
		SparkTensor post;
		SparkTensor comb;
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
		const Layer *layer;
	};

	static uint64_t Max(uint64_t a, uint64_t b)
	{
		return a > b ? a : b;
	}

	static void AttentionShape(SparkDeviceLatentAttentionShape *attention, uint32_t window)
	{
		memset(attention, 0, sizeof(*attention));
		attention->heads = SPARK_MOTIF_MODEL_QUERY_HEAD_COUNT;
		attention->nope = kNope;
		attention->rope = kRope;
		attention->latent = kLatent;
		attention->value_dimension = kValueDim;
		attention->epsilon = kEpsilon;
		attention->query_lora = kQueryLora;
		attention->tensor_prefix = "attn_";
		attention->rope_half = 1u;
		attention->differential = kGroup;
		attention->window = window;
		if ( window != 0u )
		{
			attention->rope_theta = SPARK_MOTIF_MODEL_WINDOW_ROPE_THETA;
			attention->scale = SPARK_MOTIF_MODEL_WINDOW_SOFTMAX_SCALE;
			return;
		}
		attention->rope_theta = SPARK_MOTIF_MODEL_ROPE_THETA;
		attention->scale = SPARK_MOTIF_MODEL_FULL_SOFTMAX_SCALE;
		attention->rope_factor = SPARK_MOTIF_MODEL_YARN_FACTOR;
		attention->rope_beta_fast = SPARK_MOTIF_MODEL_YARN_BETA_FAST;
		attention->rope_beta_slow = SPARK_MOTIF_MODEL_YARN_BETA_SLOW;
		attention->rope_original = SPARK_MOTIF_MODEL_YARN_ORIGINAL_CONTEXT;
	}

	static void Shapes(State *state)
	{
		SparkDeviceStageShape *shape = &state->shape;
		memset(shape, 0, sizeof(*shape));
		shape->tag = P::kTag;
		shape->hidden = kHidden;
		shape->vocab = kVocab;
		shape->layers = kLayers;
		shape->kv_layers = kLayers;
		shape->kv_heads = 1u;
		shape->kv_head_dimension = (kSlot + kLatent) / 2u;
		shape->pack_alignment = G::kPackAlignment;
		shape->epsilon = kEpsilon;
		shape->projection = SPARK_PROJECTION_GEMV;
		shape->prenormed_head = 1u;
		AttentionShape(&state->window_shape, SPARK_MOTIF_MODEL_WINDOW_TOKENS);
		AttentionShape(&state->full_shape, 0u);
	}

	static uint32_t ScratchSlices(State *state, SparkDeviceStageSlice *slices)
	{
		const uint64_t rows = state->stage.max_rows, hidden = kHidden, routes = rows * kTopK, heads = kHeads;
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
			{ rows * kQueryLora * 2u, &state->attention.query_lora },
			{ rows * heads * (kNope + kRope) * 2u, &state->attention.query },
			{ rows * kSlot * 2u, &state->attention.kv },
			{ rows * kSignals * kValueDim * 2u, &state->attention.gate },
			{ rows * heads * kSlot * 2u, &state->attention.packed },
			{ SparkOpsAttentionScratchBytes(state->stage.max_rows, kHeads, kLatent, state->stage.max_positions), &state->attention.records },
			{ rows * heads * kLatent * 2u, &state->attention.latent },
			{ rows * heads * kValueDim * 2u, &state->attention.value },
			{ rows * kLambdas * 2u, &state->attention.lambda },
			{ rows * kSignals * kValueDim * 2u, &state->attention.differential },
			{ rows * 2u * Max(kDenseRows, kExpertRows) * 2u, &state->ffn_scratch.gate_up },
			{ rows * Max(kDenseRows, kExpertRows) * 2u, &state->ffn_scratch.act },
			{ routes * sizeof(uint32_t), &state->experts.experts },
			{ routes * sizeof(float), &state->experts.weights },
			{ routes * 2u * kExpertRows * 2u, &state->experts.gate_up },
			{ routes * kExpertRows * 2u, &state->experts.act },
			{ routes * hidden * 2u, &state->experts.out },
			{ (routes + 1u) * hidden * 2u, &state->experts.sorted }
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
			{ "hidden", kHidden }, { "layers", kLayers }, { "vocab_rows", kVocab / kTp }, { "head_groups", kGroups }, { "experts", kExperts },
			{ "local_experts", kLocalExperts }
		};
		status = SparkDeviceStageOpen(&state->stage, &state->shape, request);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageExpectConfig(&state->stage, config, sizeof(config) / sizeof(config[0]));
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageReserve(&state->stage, SparkDeviceStagePlan(0, slices, ScratchSlices(state, slices)), 0, 0u,
				request->configuration->state_budget_bytes);
		if ( status != SPARK_STATUS_OK )
			return status;
		SparkDeviceLatentAttentionRopePrepare(&state->window_shape, kHeads, &state->window_rope);
		SparkDeviceLatentAttentionRopePrepare(&state->full_shape, kHeads, &state->full_rope);
		return SparkDeviceStageGeometry(&state->stage, geometry);
	}

	static SparkStatus BindHyper(State *state, const SparkStageRunnerDeviceWeights *weights, uint32_t index, const char *prefix, Hyper *hyper)
	{
		char name[48];
		snprintf(name, sizeof(name), "%sfn", prefix);
		SparkStatus status = SparkDeviceStageLayerTensor(&state->stage, weights, index, name, "f32", (uint64_t)SPARK_HC_COMB_MIXES * kResidual * sizeof(float),
			&hyper->fn);
		snprintf(name, sizeof(name), "%sbase", prefix);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, name, "f32", SPARK_HC_COMB_MIXES * sizeof(float), &hyper->base);
		snprintf(name, sizeof(name), "%sscale", prefix);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, name, "f32", 3u * sizeof(float), &hyper->scale);
		return status;
	}

	static SparkStatus BindOwned(State *state, const SparkStageRunnerDeviceWeights *weights, uint32_t index, const char *prefix, uint32_t rows,
		SparkDeviceFfnWeights *ffn, SparkTensor *poly)
	{
		char gate_up[48], down[48], params[48];
		snprintf(gate_up, sizeof(gate_up), "%s_gate_up", prefix);
		snprintf(down, sizeof(down), "%s_down", prefix);
		snprintf(params, sizeof(params), "%s_poly", prefix);
		SparkStatus status = SparkDeviceFfnBindNamed(&state->stage, weights, index, gate_up, down, rows, ffn);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, params, "f32", 4u * sizeof(float), poly);
		return status;
	}

	static SparkStatus BindExperts(State *state, const SparkStageRunnerDeviceWeights *weights, uint32_t index, Layer *layer)
	{
		SparkStatus status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "router_proj", "bf16", (uint64_t)kExperts * kHidden * 2u, &layer->router);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "expert_bias", "f32", (uint64_t)kExperts * sizeof(float), &layer->expert_bias);
		layer->gate_up.codec = layer->down.codec = SPARK_WEIGHT_CODEC_BF16;
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "moe_w13", "bf16", (uint64_t)kLocalExperts * 2u * kExpertRows * kHidden * 2u,
				&layer->gate_up.values);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "moe_w2", "bf16", (uint64_t)kLocalExperts * kHidden * kExpertRows * 2u,
				&layer->down.values);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "moe_poly", "f32", (uint64_t)kLocalExperts * 4u * sizeof(float),
				&layer->expert_poly);
		if ( status == SPARK_STATUS_OK )
			SparkDeviceStageNoteRouted(&state->stage, layer->gate_up.values.bytes + layer->down.values.bytes, kLocalExperts, kTopK);
		return status;
	}

	static SparkStatus BindLayerWeights(State *state, const SparkStageRunnerDeviceWeights *weights, uint32_t index)
	{
		Layer *layer = &state->layers[index];
		layer->owner = index % kTp == state->stage.tp_rank;
		SparkStatus status = BindHyper(state, weights, index, "hc_attn_", &layer->hyper[MOTIF_ATTENTION_PART]);
		if ( status == SPARK_STATUS_OK )
			status = BindHyper(state, weights, index, "hc_ffn_", &layer->hyper[MOTIF_FFN_PART]);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "attn_norm", "f32", kHidden * sizeof(float), &layer->norm[MOTIF_ATTENTION_PART]);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "ffn_norm", "f32", kHidden * sizeof(float), &layer->norm[MOTIF_FFN_PART]);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceLatentAttentionBind(&state->stage, weights, index, L::Window(index) ? &state->window_shape : &state->full_shape,
				&layer->attention);
		if ( status == SPARK_STATUS_OK && !L::Routed(index) )
			return layer->owner != 0u ? BindOwned(state, weights, index, "ffn", kDenseRows, &layer->ffn, &layer->ffn_poly) : SPARK_STATUS_OK;
		if ( status == SPARK_STATUS_OK )
			status = BindExperts(state, weights, index, layer);
		if ( status == SPARK_STATUS_OK && layer->owner != 0u )
			status = BindOwned(state, weights, index, "shared", kExpertRows, &layer->shared, &layer->shared_poly);
		return status;
	}

	static SparkStatus BindDevice(void *model, const SparkStageRunnerDeviceWeights *weights, SparkStageRunnerModelGeometry *geometry)
	{
		State *state = (State *)model;
		SparkDeviceStageSlice slices[MD_SLICES];
		SparkStatus status;
		(void)geometry;
		if ( weights == 0 || weights->pack_address == 0u || weights->pack_bytes != state->stage.pack_bytes )
			return SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_INVALID_ARGUMENT, "bind_device needs the device mapping of the whole rank pack");
		status = SparkDeviceStageTensor(&state->stage, weights, "head_norm", "f32", kHidden * sizeof(float), &state->head_norm);
		for ( uint32_t index = 0u; index < kLayers && status == SPARK_STATUS_OK; index++ )
			status = BindLayerWeights(state, weights, index);
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
		params.norm_epsilon = SPARK_MOTIF_MODEL_HC_NORM_EPSILON;
		params.post_magnitude = SPARK_MOTIF_MODEL_HC_POST_COEFFICIENT;
		params.iterations = SPARK_MOTIF_MODEL_HC_SINKHORN_ITERATIONS;
		params.gate_clamp = SPARK_MOTIF_MODEL_HC_GATE_CLAMP;
		params.residual_clamp = SPARK_MOTIF_MODEL_HC_RESIDUAL_CLAMP;
		params.sum_floor = SPARK_MOTIF_MODEL_HC_SUM_FLOOR;
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

	static SparkPolyNormParams PolyParams(uint32_t rows, uint32_t width)
	{
		SparkPolyNormParams poly;
		memset(&poly, 0, sizeof(poly));
		poly.rows = rows;
		poly.width = width;
		poly.input_stride = 2u * width;
		poly.output_stride = width;
		poly.param_count = 1u;
		poly.clamp = SPARK_MOTIF_MODEL_HIDDEN_CLAMP;
		poly.bias_clamp = SPARK_MOTIF_MODEL_POLYNORM_ROUTED_BIAS_CLAMP;
		poly.output_scale = SPARK_MOTIF_MODEL_POLYNORM_OUTPUT_SCALE;
		poly.epsilon = SPARK_MOTIF_MODEL_POLYNORM_EPSILON;
		return poly;
	}

	static SparkStatus Begin(State *state, const SparkStageRunnerStep *step, void *queue, Frame *frame)
	{
		frame->queue = (SparkDeviceQueue *)queue;
		frame->layer = 0;
		return SparkDeviceStageStepTables(&state->stage, step, &frame->t);
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

	static SparkStatus HyperNorm(State *state, Frame &frame, uint32_t layer, uint32_t part)
	{
		const Hyper &hyper = frame.layer->hyper[part];
		const SparkHcParams params = HcParams(frame);
		const SparkNormParams norm = NormParams(frame);
		(void)layer;
		SparkStatus status = SparkOpsHcSiteClamped(state->stage.ops, frame.queue, state->streams, hyper.fn, hyper.base, hyper.scale, state->post, state->comb,
			state->mixed, &params);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsRmsNormResidual(state->stage.ops, frame.queue, state->mixed, 0, frame.layer->norm[part], state->normed, &norm);
		return status;
	}

	static SparkStatus Attention(State *state, Frame &frame, uint32_t layer, bool window)
	{
		return SparkDeviceLatentAttention(&state->stage, frame.queue, window ? &state->window_shape : &state->full_shape,
			window ? &state->window_rope : &state->full_rope, &frame.t, &frame.layer->attention, layer, &state->attention, state->normed, state->partial);
	}

	static SparkStatus WindowAttention(State *state, Frame &frame, uint32_t layer)
	{
		return Attention(state, frame, layer, true);
	}

	static SparkStatus FullAttention(State *state, Frame &frame, uint32_t layer)
	{
		return Attention(state, frame, layer, false);
	}

	static SparkStatus AllReduce(State *state, Frame &frame)
	{
		return SparkDeviceStageAllReduce(&state->stage, frame.queue, state->partial, frame.t.rows);
	}

	static SparkStatus HyperPost(State *state, Frame &frame, uint32_t layer, uint32_t part)
	{
		const SparkHcParams params = HcParams(frame);
		SparkStatus status = SparkOpsHcPostCombF32(state->stage.ops, frame.queue, state->streams, state->post, state->comb, state->partial, &params);
		if ( status != SPARK_STATUS_OK || part != MOTIF_FFN_PART )
			return status;
		return SparkDeviceStageDumpRows(&state->stage, frame.queue, &frame.t, state->streams, kResidual, layer);
	}

	static SparkStatus OwnedMlp(State *state, Frame &frame, const SparkDeviceFfnWeights &ffn, SparkTensor poly, uint32_t rows_out, SparkTensor output)
	{
		const uint32_t rows = frame.t.rows;
		const SparkPolyNormParams params = PolyParams(rows, rows_out);
		SparkStatus status = SparkDeviceLinearWeight(&state->stage, frame.queue, state->normed, kHidden, &ffn.gate_up, state->ffn_scratch.gate_up, 2u * rows_out,
			rows, kHidden, 2u * rows_out);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsPolyNormMul(state->stage.ops, frame.queue, state->ffn_scratch.gate_up, poly, 0, state->ffn_scratch.act, &params);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceLinearWeight(&state->stage, frame.queue, state->ffn_scratch.act, rows_out, &ffn.down, output, kHidden, rows, rows_out, kHidden);
		return status;
	}

	static SparkStatus DenseMlp(State *state, Frame &frame, uint32_t layer)
	{
		const Layer &weights = *frame.layer;
		SparkStatus status = weights.owner != 0u ? OwnedMlp(state, frame, weights.ffn, weights.ffn_poly, kDenseRows, state->partial) :
			SparkDeviceFill(frame.queue, state->partial.range, 0u, (uint64_t)frame.t.rows * kHidden * 2u);
		SparkDeviceStageLayerDone(&state->stage, layer);
		return status;
	}

	static SparkStatus RoutedMlp(State *state, Frame &frame, uint32_t layer)
	{
		const Layer &weights = *frame.layer;
		const uint32_t rows = frame.t.rows;
		SparkDeviceRoutedFfnShape ffn;
		SparkDeviceRoutedPoly poly;
		SparkGemvParams router;
		SparkRouteParams route;
		SparkStatus status;
		memset(&ffn, 0, sizeof(ffn));
		ffn.experts = kLocalExperts;
		ffn.top_k = kTopK;
		ffn.intermediate = kExpertRows;
		poly.params = weights.expert_poly;
		poly.clamp = SPARK_MOTIF_MODEL_HIDDEN_CLAMP;
		poly.bias_clamp = SPARK_MOTIF_MODEL_POLYNORM_ROUTED_BIAS_CLAMP;
		poly.scale = SPARK_MOTIF_MODEL_POLYNORM_OUTPUT_SCALE;
		poly.epsilon = SPARK_MOTIF_MODEL_POLYNORM_EPSILON;
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
		route.scale = SPARK_MOTIF_MODEL_MOE_ROUTED_SCALE;
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsRouteSigmoid(state->stage.ops, frame.queue, state->router_logits, weights.expert_bias, state->experts.experts,
				state->experts.weights, &route);
		route.local_experts = kLocalExperts;
		route.first_expert = state->stage.tp_rank * kLocalExperts;
		if ( status == SPARK_STATUS_OK && kTp > 1u )
			status = SparkOpsRouteLocal(state->stage.ops, frame.queue, state->experts.experts, &route);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceRoutedFfnPoly(&state->stage, frame.queue, &ffn, rows, &weights.gate_up, 0, &weights.down, &poly, &state->experts,
				state->normed);
		if ( status == SPARK_STATUS_OK )
			status = weights.owner != 0u ? OwnedMlp(state, frame, weights.shared, weights.shared_poly, kExpertRows, state->shared_out) :
				SparkDeviceFill(frame.queue, state->shared_out.range, 0u, (uint64_t)rows * kHidden * 2u);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsRouteCombineWeightedShared(state->stage.ops, frame.queue, state->experts.out, state->experts.weights, state->shared_out,
				state->partial, &route);
		SparkDeviceStageLayerDone(&state->stage, layer);
		return status;
	}

	static SparkStatus HyperEnd(State *state, Frame &frame)
	{
		const SparkHcParams params = HcParams(frame);
		const SparkNormParams norm = NormParams(frame);
		SparkStatus status = SparkOpsStreamMean(state->stage.ops, frame.queue, state->streams, state->mixed, &params);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsRmsNormResidual(state->stage.ops, frame.queue, state->mixed, 0, state->head_norm, frame.t.hidden, &norm);
		return status;
	}

	static SparkStatus Step(void *model, const SparkStageRunnerStep *step, void *queue)
	{
		return MotifSchedule<G, MotifDevice>::Step((State *)model, step, queue);
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
struct Motif3Device
{
	using Geometry = Motif3Geometry<TP>;
	static constexpr const char *kTag = "sparkpipe_motif3_device";
	static constexpr const char *kName = "motif-3";
};

const SparkStageRunnerModelInterface *SparkMotif3DeviceStageModelAt(uint32_t tp_degree)
{
	switch ( tp_degree )
	{
	case 1u:
		return MotifDevice<Motif3Device<1u>>::Interface();
	case 2u:
		return MotifDevice<Motif3Device<2u>>::Interface();
	case 4u:
		return MotifDevice<Motif3Device<4u>>::Interface();
	case 8u:
		return MotifDevice<Motif3Device<8u>>::Interface();
	default:
		return 0;
	}
}
