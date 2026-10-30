#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inference/runner/device_layers.h"
#include "sparkpipe/spark_qwen38_27b_stage_model.h"
#include "sparkpipe/spark_qwen38_device_stage.h"
#include "sparkpipe/spark_qwen38_family.h"
#include "sparkpipe/spark_qwen38_flash_stage_model.h"

#define QD_SLICES 48u
#define QD_PLE_WINDOW_POOL 2u
#define QD_PLE_HISTORY_POOL 3u
#define QD_DENSE_INDEX_POSITIONS(G) ((G::kIndexBudget / G::kIndexRatio + 1u) * G::kIndexRatio - 1u)

template<class P>
struct QwenDevice
{
	using G = typename P::Geometry;
	using L = QwenHybridLayout<G>;

	static_assert(G::kTpDegree == 1u, "a device stage is one whole rank");
	static_assert(!G::kRankHeads, "a device stage holds every head");
	static_assert(L::kHyper == L::kPle && L::kHyper == G::kMoe, "the device layers cover the plain and the hyper-connection Qwen variants");

	struct Hyper
	{
		SparkTensor norm;
		SparkTensor down;
		SparkTensor up;
		SparkTensor inject;
	};

	struct Layer
	{
		SparkTensor input_norm;
		SparkTensor post_norm;
		SparkDeviceFfnWeights ffn;
		SparkDeviceGatedDeltaWeights delta;
		SparkDeviceAttentionWeights attention;
		Hyper hyper[2];
		SparkTensor router;
		SparkTensor shared_gate;
		SparkTensor shared_gate_up;
		SparkTensor shared_down;
		SparkDeviceExpertMatrix gate;
		SparkDeviceExpertMatrix up;
		SparkDeviceExpertMatrix down;
		uint32_t recurrent_index;
		uint32_t kv_index;
	};

	struct PleWeights
	{
		SparkTensor key;
		SparkTensor value;
		SparkTensor norm_key;
		SparkTensor norm_query;
		SparkTensor norm_conv;
		SparkTensor conv;
		SparkTensor multipliers;
		SparkTensor primes;
		SparkTensor offsets;
		SparkTensor table;
		SparkTensor table_scale;
		uint64_t table_rows;
	};

	struct State
	{
		SparkDeviceStage stage;
		SparkDeviceStageShape shape;
		SparkDeviceGatedDeltaShape delta_shape;
		SparkDeviceAttentionShape attention_shape;
		uint32_t ffn;
		uint32_t fp8;
		uint32_t shared_rows;
		Layer layers[G::kLayers];
		Hyper final_hyper;
		PleWeights ple;
		SparkTensor normed;
		SparkTensor partial;
		SparkDeviceGatedDeltaScratch delta;
		SparkDeviceAttentionScratch attention;
		SparkDeviceFfnScratch ffn_scratch;
		SparkDeviceAttentionRope rope;
		SparkTensor residual;
		SparkTensor hc_normed;
		SparkTensor hc_low;
		SparkTensor hc_up;
		SparkTensor hc_inject;
		SparkTensor router_logits;
		SparkDeviceExpertsScratch experts;
		SparkTensor shared_gate_up;
		SparkTensor shared_act;
		SparkTensor shared_out;
		SparkTensor shared_logit;
		SparkTensor ple_ids;
		SparkTensor ple_embed;
		SparkTensor ple_key;
		SparkTensor ple_value;
		SparkTensor ple_query;
		SparkTensor ple_gated;
		SparkTensor ple_conv_in;
		SparkTensor ple_conv_out;
		SparkTensor ple_keep;
	};

	struct Frame
	{
		SparkDeviceStepTensors t;
		SparkDeviceQueue *queue;
	};

	static void Shapes(State *state)
	{
		SparkDeviceStageShape *shape = &state->shape;
		SparkDeviceGatedDeltaShape *delta = &state->delta_shape;
		SparkDeviceAttentionShape *attention = &state->attention_shape;
		memset(shape, 0, sizeof(*shape));
		shape->tag = P::kTag;
		shape->hidden = G::kHidden;
		shape->vocab = G::kVocab;
		shape->layers = G::kLayers;
		shape->kv_layers = L::kAttnLayers;
		shape->kv_heads = G::kKvHeads;
		shape->kv_head_dimension = G::kHeadDim;
		shape->pack_alignment = QH_PACK_ALIGNMENT;
		shape->epsilon = QH_EPSILON;
		shape->projection = P::kProjection;
		shape->prenormed_head = L::kHyper ? 1u : 0u;
		memset(delta, 0, sizeof(*delta));
		delta->key_heads = G::kGdnKeyHeads;
		delta->value_heads = G::kGdnValueHeads;
		delta->key_dimension = QH_KEY_DIM;
		delta->value_dimension = QH_VALUE_DIM;
		delta->conv_taps = QH_CONV;
		delta->epsilon = QH_EPSILON;
		delta->sigmoid_gate = G::kGdnSigmoidGate ? 1u : 0u;
		memset(attention, 0, sizeof(*attention));
		attention->heads = G::kHeads;
		attention->kv_heads = G::kKvHeads;
		attention->head_dimension = G::kHeadDim;
		attention->rope_dimension = G::kRopeDim;
		attention->rope_theta = G::kRopeTheta;
		attention->epsilon = QH_EPSILON;
		attention->scale = (float)(1.0 / sqrt((double)G::kHeadDim));
		attention->output_gate = SPARK_ATTENTION_GATE_SIGMOID;
	}

	static uint64_t Max(uint64_t a, uint64_t b)
	{
		return a > b ? a : b;
	}

	static uint32_t MixWidth(const State *state)
	{
		const uint32_t delta = SparkDeviceGatedDeltaMix(&state->delta_shape), attention = SparkDeviceAttentionFusedWidth(&state->attention_shape);
		return delta > attention ? delta : attention;
	}

	static uint32_t GatedWidth(const State *state)
	{
		const uint32_t delta = SparkDeviceGatedDeltaValues(&state->delta_shape), attention = SparkDeviceAttentionQueryWidth(&state->attention_shape);
		return delta > attention ? delta : attention;
	}

	static uint32_t ScratchSlices(State *state, SparkDeviceStageSlice *slices)
	{
		const uint64_t rows = state->stage.max_rows, hidden = G::kHidden, routes = G::kMoe ? rows * G::kTopK : 0u;
		const uint64_t residual = L::kHyper ? rows * L::kResidual * 2u : 0u, ple = L::kPle ? residual : 0u;
		const SparkDeviceGatedDeltaShape *delta = &state->delta_shape;
		const SparkDeviceAttentionShape *attention = &state->attention_shape;
		const SparkDeviceStageSlice plan[] =
		{
			{ rows * hidden * 2u, &state->normed },
			{ rows * hidden * 2u, &state->partial },
			{ rows * MixWidth(state) * 2u, &state->delta.mix },
			{ rows * SparkDeviceGatedDeltaGates(delta) * 2u, &state->delta.gates },
			{ SparkOpsDeltaScratchBytes(state->stage.max_rows, delta->key_heads, delta->value_heads, delta->key_dimension), &state->delta.delta },
			{ rows * SparkDeviceGatedDeltaValues(delta) * 2u, &state->delta.core },
			{ rows * GatedWidth(state) * 2u, &state->delta.gated },
			{ rows * SparkDeviceAttentionQueryWidth(attention) * 2u, &state->attention.query },
			{ rows * SparkDeviceAttentionKvWidth(attention) * 2u, &state->attention.key },
			{ SparkOpsAttentionScratchBytes(state->stage.max_rows, attention->heads, attention->head_dimension, state->stage.max_positions),
				&state->attention.records },
			{ rows * 2u * state->ffn * 2u, &state->ffn_scratch.gate_up },
			{ rows * state->ffn * 2u, &state->ffn_scratch.act },
			{ SparkDeviceGatedDeltaKeepBytes(delta, state->stage.verify_rows, L::kGdnLayers), &state->delta.keep },
			{ state->fp8 != 0u ? Max(Max(2u * (uint64_t)state->ffn, MixWidth(state)), Max(GatedWidth(state), state->ffn)) * hidden * 2u : 0u,
				&state->stage.dequant },
			{ residual, &state->residual },
			{ residual, &state->hc_normed },
			{ L::kHyper ? rows * G::kHcLowRank * 2u : 0u, &state->hc_low },
			{ residual, &state->hc_up },
			{ L::kHyper ? rows * G::kStreams * sizeof(float) : 0u, &state->hc_inject },
			{ G::kMoe ? rows * G::kExperts * 2u : 0u, &state->router_logits },
			{ routes * sizeof(uint32_t), &state->experts.experts },
			{ routes * sizeof(float), &state->experts.weights },
			{ routes * 2u * G::kMoeInter * 2u, &state->experts.gate_up },
			{ routes * G::kMoeInter * 2u, &state->experts.act },
			{ routes * hidden * 2u, &state->experts.out },
			{ G::kMoe ? (routes + 1u) * hidden * 2u : 0u, &state->experts.sorted },
			{ rows * 2u * state->shared_rows * 2u, &state->shared_gate_up },
			{ rows * state->shared_rows * 2u, &state->shared_act },
			{ G::kMoe ? rows * hidden * 2u : 0u, &state->shared_out },
			{ G::kMoe ? rows * 2u : 0u, &state->shared_logit },
			{ L::kPle ? rows * L::kPleColumns * sizeof(int64_t) : 0u, &state->ple_ids },
			{ L::kPle ? rows * L::kPleColumns * G::kPleHeadDim * 2u : 0u, &state->ple_embed },
			{ ple, &state->ple_key },
			{ L::kPle ? rows * hidden * 2u : 0u, &state->ple_value },
			{ ple, &state->ple_query },
			{ ple, &state->ple_gated },
			{ ple, &state->ple_conv_in },
			{ ple, &state->ple_conv_out },
			{ L::kPle ? (uint64_t)state->stage.verify_rows * L::kResidual * 2u : 0u, &state->ple_keep }
		};
		const uint32_t count = sizeof(plan) / sizeof(plan[0]);
		for ( uint32_t index = 0u; index < count; index++ )
			slices[index] = plan[index];
		return count;
	}

	static SparkStatus Open(const SparkStageRunnerModelOpen *request, void **out, SparkStageRunnerModelGeometry *geometry)
	{
		SparkDeviceStageSlice slices[QD_SLICES];
		State *state = (State *)calloc(1u, sizeof(State));
		SparkStatus status;
		*out = state;
		if ( state == 0 )
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		Shapes(state);
		const SparkDeviceStageRecurrentPlan pools[] = { { L::kGdnLayers, SparkDeviceGatedDeltaStateBytes(&state->delta_shape) },
			{ L::kGdnLayers, SparkDeviceGatedDeltaWindowBytes(&state->delta_shape) },
			{ 1u, (uint64_t)L::kResidual * SPARK_CONV_SPAN(G::kPleConv, G::kPleDilation) * 2u }, { 1u, L::kPleHistory * sizeof(int32_t) } };
		const SparkDeviceStageConfigValue config[] = { { "hidden", G::kHidden }, { "layers", G::kLayers }, { "vocab_rows", G::kVocab } };
		const SparkDeviceStageConfigValue hyper[] = { { "experts", G::kExperts }, { "local_experts", G::kExperts }, { "top_k", G::kTopK },
			{ "moe_inter", G::kMoeInter }, { "hc_streams", G::kStreams }, { "hc_lowrank", G::kHcLowRank }, { "index_heads", G::kIndexHeads },
			{ "index_dim", G::kIndexDim }, { "index_budget", G::kIndexBudget }, { "index_ratio", G::kIndexRatio }, { "ple_table_first", 0u } };
		status = SparkDeviceStageOpen(&state->stage, &state->shape, request);
		if constexpr ( G::kMoe )
		{
			if ( status == SPARK_STATUS_OK && SparkNamedPackConfigU32(&state->stage.pack, "shared_rows", &state->shared_rows) != SPARK_STATUS_OK )
				status = SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_PARSE_ERROR, "the pack configuration lacks shared_rows");
			if ( status == SPARK_STATUS_OK )
				status = SparkDeviceStageExpectConfig(&state->stage, hyper, sizeof(hyper) / sizeof(hyper[0]));
		}
		else
		{
			if ( status == SPARK_STATUS_OK && SparkNamedPackConfigU32(&state->stage.pack, "ffn_rows", &state->ffn) != SPARK_STATUS_OK )
				status = SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_PARSE_ERROR, "the pack configuration lacks ffn_rows");
			if ( status == SPARK_STATUS_OK )
			{
				SparkNamedPackEntry entry;
				state->fp8 = SparkNamedPackEntryLoad(&state->stage.pack, "layers.0.ffn_gate_up", &entry) == SPARK_STATUS_OK && strcmp(entry.kind, "fp8_e4m3") == 0 ? 1u : 0u;
			}
		}
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageExpectConfig(&state->stage, config, sizeof(config) / sizeof(config[0]));
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageReserve(&state->stage, SparkDeviceStagePlan(0, slices, ScratchSlices(state, slices)), pools,
				L::kPle ? 4u : SPARK_DEVICE_GATED_DELTA_POOLS, request->configuration->state_budget_bytes);
		if ( status != SPARK_STATUS_OK )
			return status;
		SparkDeviceAttentionRopePrepare(&state->attention_shape, &state->rope);
		return SparkDeviceStageGeometry(&state->stage, geometry);
	}

	static void HyperName(char *name, size_t bytes, uint32_t index, const char *prefix, const char *part)
	{
		if ( index == G::kLayers )
			snprintf(name, bytes, "%s%s", prefix, part);
		else
			snprintf(name, bytes, "layers.%u.%s%s", index, prefix, part);
	}

	static SparkStatus BindHyper(State *state, const SparkStageRunnerDeviceWeights *weights, uint32_t index, const char *prefix, uint32_t inject,
		Hyper *bound)
	{
		char name[64];
		const uint64_t width = L::kResidual;
		HyperName(name, sizeof(name), index, prefix, "norm");
		SparkStatus status = SparkDeviceStageTensor(&state->stage, weights, name, "f32", width * sizeof(float), &bound->norm);
		HyperName(name, sizeof(name), index, prefix, "down");
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageTensor(&state->stage, weights, name, "bf16", (uint64_t)G::kHcLowRank * width * 2u, &bound->down);
		HyperName(name, sizeof(name), index, prefix, "up");
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageTensor(&state->stage, weights, name, "bf16", width * G::kHcLowRank * 2u, &bound->up);
		HyperName(name, sizeof(name), index, prefix, "inject");
		if ( status == SPARK_STATUS_OK && inject != 0u )
			status = SparkDeviceStageTensor(&state->stage, weights, name, "bf16", (uint64_t)G::kStreams * width * 2u, &bound->inject);
		return status;
	}

	static SparkStatus BindNvfp4(State *state, const SparkStageRunnerDeviceWeights *weights, uint32_t index, const char *name, uint64_t rows, uint64_t columns,
		SparkDeviceExpertMatrix *bound)
	{
		char scale_name[64];
		const uint64_t experts = G::kExperts, plane = experts * rows * (columns / SPARK_NVFP4_GROUP);
		SparkTensor scale;
		SparkStatus status = SparkDeviceStageLayerTensor(&state->stage, weights, index, name, "nvfp4_e2m1", experts * rows * (columns / 2u), &bound->values);
		bound->codec = SPARK_WEIGHT_CODEC_NVFP4_E2M1;
		snprintf(scale_name, sizeof(scale_name), "%s.scale", name);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, scale_name, "ue4m3_f32_global", experts * sizeof(float) + plane, &scale);
		if ( status == SPARK_STATUS_OK )
		{
			bound->global = SparkTensorSlice(scale, 0u, experts * sizeof(float));
			bound->scale = SparkTensorSlice(scale, experts * sizeof(float), plane);
			SparkDeviceStageNoteRouted(&state->stage, bound->values.bytes + scale.bytes, G::kExperts, G::kTopK);
		}
		return status;
	}

	static SparkStatus BindMoe(State *state, const SparkStageRunnerDeviceWeights *weights, uint32_t index, Layer *layer)
	{
		const uint64_t hidden = G::kHidden, shared = state->shared_rows;
		SparkStatus status = BindHyper(state, weights, index, "attn_hc_", 1u, &layer->hyper[0]);
		if ( status == SPARK_STATUS_OK )
			status = BindHyper(state, weights, index, "mlp_hc_", 1u, &layer->hyper[1]);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "moe_router", "bf16", (uint64_t)G::kExperts * hidden * 2u, &layer->router);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "moe_shared_gate", "bf16", hidden * 2u, &layer->shared_gate);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "moe_shared_gate_up", "bf16", 2u * shared * hidden * 2u, &layer->shared_gate_up);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "moe_shared_down", "bf16", hidden * shared * 2u, &layer->shared_down);
		if ( status == SPARK_STATUS_OK )
			status = BindNvfp4(state, weights, index, "moe_w1", G::kMoeInter, hidden, &layer->gate);
		if ( status == SPARK_STATUS_OK )
			status = BindNvfp4(state, weights, index, "moe_w3", G::kMoeInter, hidden, &layer->up);
		if ( status == SPARK_STATUS_OK )
			status = BindNvfp4(state, weights, index, "moe_w2", hidden, G::kMoeInter, &layer->down);
		return status;
	}

	static SparkStatus BindPle(State *state, const SparkStageRunnerDeviceWeights *weights)
	{
		const uint32_t index = G::kPleLayer;
		const uint64_t width = L::kResidual, hidden = G::kHidden;
		PleWeights *ple = &state->ple;
		uint64_t offset = 0u, bytes = 0u;
		char name[64];
		SparkStatus status;
		snprintf(name, sizeof(name), "layers.%u.ple_table", index);
		status = SparkDeviceStageEntry(&state->stage, name, "fp8_e4m3", &offset, &bytes, 0);
		ple->table_rows = bytes / G::kPleHeadDim;
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "ple_table", "fp8_e4m3", ple->table_rows * G::kPleHeadDim, &ple->table);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "ple_table_scale", "f32", sizeof(float), &ple->table_scale);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "ple_key", "bf16", width * hidden * 2u, &ple->key);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "ple_value", "bf16", hidden * hidden * 2u, &ple->value);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "ple_norm_key", "f32", width * sizeof(float), &ple->norm_key);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "ple_norm_query", "f32", width * sizeof(float), &ple->norm_query);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "ple_norm_conv", "f32", width * sizeof(float), &ple->norm_conv);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "ple_conv", "bf16", width * G::kPleConv * 2u, &ple->conv);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "ple_multipliers", "i64", (G::kPleOrders + 1u) * sizeof(int64_t), &ple->multipliers);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "ple_vocab_sizes", "i64", L::kPleColumns * sizeof(int64_t), &ple->primes);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "ple_offsets", "i64", L::kPleColumns * sizeof(int64_t), &ple->offsets);
		if ( status == SPARK_STATUS_OK )
			status = BindHyper(state, weights, G::kLayers, "final_hc_", 0u, &state->final_hyper);
		return status;
	}

	static SparkStatus BindLayer(State *state, const SparkStageRunnerDeviceWeights *weights, uint32_t index)
	{
		Layer *layer = &state->layers[index];
		SparkStatus status;
		if constexpr ( G::kMoe )
			status = BindMoe(state, weights, index, layer);
		else
		{
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "input_norm", "f32", G::kHidden * 4u, &layer->input_norm);
			if ( status == SPARK_STATUS_OK )
				status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "post_norm", "f32", G::kHidden * 4u, &layer->post_norm);
			if ( status == SPARK_STATUS_OK )
				status = SparkDeviceFfnBind(&state->stage, weights, index, state->ffn, &layer->ffn);
		}
		if ( status != SPARK_STATUS_OK )
			return status;
		if ( L::Attention(index) )
			return SparkDeviceAttentionBind(&state->stage, weights, index, &state->attention_shape, &layer->attention);
		return SparkDeviceGatedDeltaBind(&state->stage, weights, index, &state->delta_shape, &layer->delta);
	}

	static SparkStatus BindDevice(void *model, const SparkStageRunnerDeviceWeights *weights, SparkStageRunnerModelGeometry *geometry)
	{
		State *state = (State *)model;
		SparkDeviceStageSlice slices[QD_SLICES];
		uint32_t recurrent = 0u, attention = 0u;
		SparkStatus status = SPARK_STATUS_OK;
		(void)geometry;
		if ( weights == 0 || weights->pack_address == 0u || weights->pack_bytes != state->stage.pack_bytes )
			return SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_INVALID_ARGUMENT, "bind_device needs the device mapping of the whole rank pack");
		for ( uint32_t index = 0u; index < G::kLayers && status == SPARK_STATUS_OK; index++ )
		{
			status = BindLayer(state, weights, index);
			if ( L::Attention(index) )
				state->layers[index].kv_index = attention++;
			else
				state->layers[index].recurrent_index = recurrent++;
		}
		if constexpr ( L::kPle )
			if ( status == SPARK_STATUS_OK )
				status = BindPle(state, weights);
		if ( status != SPARK_STATUS_OK )
			return status;
		status = SparkDeviceStageAllocate(&state->stage, slices, ScratchSlices(state, slices));
		state->attention.qkv = state->delta.mix;
		state->attention.gated = state->delta.gated;
		return status;
	}

	static const Layer &LayerOf(State *state, uint32_t index)
	{
		return state->layers[index];
	}

	static SparkStatus Begin(State *state, const SparkStageRunnerStep *step, void *queue, Frame *frame)
	{
		frame->queue = (SparkDeviceQueue *)queue;
		const SparkStatus status = SparkDeviceStageStepTables(&state->stage, step, &frame->t);
		if ( status == SPARK_STATUS_OK && G::kMoe && frame->t.route_override_width != 0u )
			return SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_UNSUPPORTED, "route overrides are not built for these experts");
		if ( status == SPARK_STATUS_OK && L::kPle && frame->t.token_ids.bytes == 0u )
			return SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_INVALID_ARGUMENT, "the n-gram embedding needs the step's token ids");
		if constexpr ( L::kIndexer )
			for ( uint32_t row = 0u; row < frame->t.rows && status == SPARK_STATUS_OK; row++ )
				if ( frame->t.host_positions[row] >= QD_DENSE_INDEX_POSITIONS(G) )
					return SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_UNSUPPORTED,
						"the device indexer is not built: positions past its dense range need it");
		return status;
	}

	static SparkStatus RmsNorm(State *state, const Frame &frame, SparkTensor weight)
	{
		return SparkDeviceResidualNorm(&state->stage, frame.queue, &frame.t, 0, weight, state->normed);
	}

	static SparkStatus AddRmsNorm(State *state, const Frame &frame, SparkTensor weight)
	{
		return SparkDeviceResidualNorm(&state->stage, frame.queue, &frame.t, &state->partial, weight, state->normed);
	}

	static SparkStatus Attention(State *state, const Frame &frame, const Layer &layer)
	{
		return SparkDeviceAttention(&state->stage, frame.queue, &state->attention_shape, &state->rope, &frame.t, &layer.attention, layer.kv_index,
			&state->attention, state->normed, state->partial);
	}

	static SparkStatus Gdn(State *state, const Frame &frame, const Layer &layer)
	{
		return SparkDeviceGatedDelta(&state->stage, frame.queue, &state->delta_shape, &frame.t, &layer.delta, layer.recurrent_index, &state->delta,
			state->normed, state->partial);
	}

	static SparkHyperParams HyperShape(uint32_t rows, uint32_t streams, uint32_t width)
	{
		SparkHyperParams params;
		memset(&params, 0, sizeof(params));
		params.rows = rows;
		params.streams = streams;
		params.width = width;
		params.epsilon = QH_EPSILON;
		return params;
	}

	static SparkStatus HyperMix(State *state, const Frame &frame, const Hyper &weights, SparkTensor mixed, uint32_t inject)
	{
		const uint32_t rows = frame.t.rows;
		const SparkHyperParams streams = HyperShape(rows, G::kStreams, G::kHidden);
		SparkHyperParams low = HyperShape(rows, 1u, G::kHcLowRank);
		low.divisor = (float)G::kStreams;
		SparkStatus status = SparkOpsGroupRmsNorm(state->stage.ops, frame.queue, state->residual, weights.norm, state->hc_normed, &streams);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceLinear(&state->stage, frame.queue, state->hc_normed, L::kResidual, weights.down, state->hc_low, G::kHcLowRank, rows,
				L::kResidual, G::kHcLowRank);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsHyperGate(state->stage.ops, frame.queue, state->hc_low, &low);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceLinear(&state->stage, frame.queue, state->hc_low, G::kHcLowRank, weights.up, state->hc_up, L::kResidual, rows,
				G::kHcLowRank, L::kResidual);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsHyperMix(state->stage.ops, frame.queue, state->hc_up, state->hc_normed, mixed, &streams);
		if ( status == SPARK_STATUS_OK && inject != 0u )
			status = SparkOpsHyperInjectWeights(state->stage.ops, frame.queue, state->hc_normed, weights.inject, state->hc_inject, &streams);
		return status;
	}

	static SparkStatus HyperBegin(State *state, const Frame &frame)
	{
		const SparkHyperParams streams = HyperShape(frame.t.rows, G::kStreams, G::kHidden);
		return SparkOpsHyperExpand(state->stage.ops, frame.queue, frame.t.hidden, state->residual, &streams);
	}

	static SparkStatus HyperPre(State *state, const Frame &frame, const Layer &layer, uint32_t part)
	{
		return HyperMix(state, frame, layer.hyper[part], state->normed, 1u);
	}

	static SparkStatus HyperPost(State *state, const Frame &frame)
	{
		const SparkHyperParams streams = HyperShape(frame.t.rows, G::kStreams, G::kHidden);
		return SparkOpsHyperInject(state->stage.ops, frame.queue, state->residual, state->partial, state->hc_inject, &streams);
	}

	static SparkStatus HyperEnd(State *state, const Frame &frame)
	{
		return HyperMix(state, frame, state->final_hyper, frame.t.hidden, 0u);
	}

	static SparkNgramParams NgramShape(const State *state, uint32_t rows, const SparkOpsSequences *sequences)
	{
		SparkNgramParams ngram;
		memset(&ngram, 0, sizeof(ngram));
		ngram.rows = rows;
		ngram.sequences = sequences->sequences;
		ngram.orders = G::kPleOrders;
		ngram.heads = G::kPleHeadsPerOrder;
		ngram.pad_token = (int32_t)G::kPleEos;
		ngram.block_token = (int32_t)G::kPleEos;
		ngram.block_from = 1u;
		ngram.history_stride = L::kPleHistory;
		ngram.head_dimension = G::kPleHeadDim;
		ngram.commit = sequences->commit;
		ngram.table_rows = state->ple.table_rows;
		return ngram;
	}

	static SparkConvParams PleConvShape(const SparkOpsSequences *sequences)
	{
		SparkConvParams conv;
		memset(&conv, 0, sizeof(conv));
		conv.channels = L::kResidual;
		conv.row_stride = L::kResidual;
		conv.sequences = sequences->sequences;
		conv.kernel_taps = G::kPleConv;
		conv.dilation = G::kPleDilation;
		conv.commit = sequences->commit;
		return conv;
	}

	static SparkStatus Ple(State *state, const Frame &frame)
	{
		const uint32_t rows = frame.t.rows;
		const PleWeights &ple = state->ple;
		const SparkHyperParams streams = HyperShape(rows, G::kStreams, G::kHidden);
		const SparkNgramParams ngram = NgramShape(state, rows, &frame.t.sequences);
		const SparkConvParams conv = PleConvShape(&frame.t.sequences);
		const SparkTensor history = SparkDeviceStageRecurrent(&state->stage, QD_PLE_HISTORY_POOL, 0u);
		SparkStatus status = SparkOpsNgramHash(state->stage.ops, frame.queue, frame.t.token_ids, frame.t.positions, &frame.t.sequences, history,
			ple.multipliers, ple.primes, ple.offsets, state->ple_ids, &ngram);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsNgramGatherFp8(state->stage.ops, frame.queue, state->ple_ids, ple.table, ple.table_scale, state->ple_embed, &ngram);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceLinear(&state->stage, frame.queue, state->ple_embed, G::kHidden, ple.key, state->ple_key, L::kResidual, rows, G::kHidden,
				L::kResidual);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceLinear(&state->stage, frame.queue, state->ple_embed, G::kHidden, ple.value, state->ple_value, G::kHidden, rows, G::kHidden,
				G::kHidden);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsGroupRmsNorm(state->stage.ops, frame.queue, state->ple_key, ple.norm_key, state->ple_key, &streams);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsGroupRmsNorm(state->stage.ops, frame.queue, state->residual, ple.norm_query, state->ple_query, &streams);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsHyperKeyGate(state->stage.ops, frame.queue, state->ple_key, state->ple_query, state->ple_value, state->ple_gated, &streams);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsGroupRmsNorm(state->stage.ops, frame.queue, state->ple_gated, ple.norm_conv, state->ple_conv_in, &streams);
		if ( status == SPARK_STATUS_OK && frame.t.sequences.commit == 0u )
			status = SparkOpsGather(state->stage.ops, frame.queue, state->ple_conv_in, frame.t.row_identity, state->ple_keep, rows, L::kResidual, rows);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsCausalConv(state->stage.ops, frame.queue, SparkDeviceStageRecurrent(&state->stage, QD_PLE_WINDOW_POOL, 0u), &frame.t.sequences,
				state->ple_conv_in, ple.conv, state->ple_conv_out, &conv);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsHyperAdd(state->stage.ops, frame.queue, state->residual, state->ple_gated, state->ple_conv_out, &streams);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsNgramHistory(state->stage.ops, frame.queue, frame.t.token_ids, 0, &frame.t.sequences, history, &ngram);
		return status;
	}

	static SparkStatus Moe(State *state, const Frame &frame, const Layer &layer)
	{
		const uint32_t rows = frame.t.rows, hidden = G::kHidden, shared = state->shared_rows;
		SparkDeviceOps *ops = state->stage.ops;
		const SparkDeviceRoutedFfnShape ffn = { G::kExperts, G::kTopK, G::kMoeInter, SPARK_GATED_SILU, 1u, 0.0f };
		SparkRouteParams route;
		SparkSiluMulParams act;
		memset(&route, 0, sizeof(route));
		route.rows = rows;
		route.experts = G::kExperts;
		route.top_k = G::kTopK;
		route.width = hidden;
		route.logit_stride = G::kExperts;
		SparkStatus status = SparkDeviceLinear(&state->stage, frame.queue, state->normed, hidden, layer.router, state->router_logits, G::kExperts, rows, hidden,
			G::kExperts);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsRouteTopk(ops, frame.queue, state->router_logits, SparkDeviceStageUnitScale(&state->stage), 0, state->experts.experts,
				state->experts.weights, &route);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceRoutedFfn(&state->stage, frame.queue, &ffn, rows, &layer.gate, &layer.up, &layer.down, &state->experts, state->normed);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceLinear(&state->stage, frame.queue, state->normed, hidden, layer.shared_gate_up, state->shared_gate_up, 2u * shared, rows, hidden,
				2u * shared);
		memset(&act, 0, sizeof(act));
		act.rows = rows;
		act.dimension = shared;
		act.input_stride = 2u * shared;
		act.output_stride = shared;
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsGatedMul(ops, frame.queue, state->shared_gate_up, state->shared_act, &act, SPARK_GATED_SILU);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceLinear(&state->stage, frame.queue, state->shared_act, shared, layer.shared_down, state->shared_out, hidden, rows, shared, hidden);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceLinear(&state->stage, frame.queue, state->normed, hidden, layer.shared_gate, state->shared_logit, 1u, rows, hidden, 1u);
		route.logit_stride = 1u;
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsSigmoidGateRows(ops, frame.queue, state->shared_out, state->shared_logit, &route);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsRouteCombineShared(ops, frame.queue, state->experts.out, state->shared_out, state->partial, &route);
		return status;
	}

	static SparkStatus Ffn(State *state, const Frame &frame, const Layer &layer)
	{
		if constexpr ( G::kMoe )
			return Moe(state, frame, layer);
		else
			return SparkDeviceGatedFfn(&state->stage, frame.queue, state->ffn, SPARK_GATED_SILU, 0.0f, &frame.t, &layer.ffn, &state->ffn_scratch, state->normed,
				state->partial);
	}

	static SparkStatus AllReduce(State *state, const Frame &frame)
	{
		(void)state;
		(void)frame;
		return SPARK_STATUS_OK;
	}

	static SparkStatus LayerDone(State *state, const Frame &frame, uint32_t index)
	{
		SparkDeviceStageLayerDone(&state->stage, index);
		if constexpr ( L::kHyper )
			return SparkDeviceStageDumpRows(&state->stage, frame.queue, &frame.t, state->residual, L::kResidual, index);
		else
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
		return QwenHybridSchedule<G, QwenDevice>::Step((State *)model, step, queue);
	}

	static SparkStatus KeepPle(State *state, SparkDeviceQueue *queue, const SparkDeviceStepTensors *frame, const SparkDeviceKeepTensors *keep)
	{
		const SparkNgramParams ngram = NgramShape(state, keep->rows, &keep->sequences);
		const SparkConvParams conv = PleConvShape(&keep->sequences);
		SparkStatus status = SparkOpsGather(state->stage.ops, queue, state->ple_keep, keep->kept_rows, state->ple_conv_in, keep->rows, L::kResidual,
			keep->frame_rows);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsCausalConv(state->stage.ops, queue, SparkDeviceStageRecurrent(&state->stage, QD_PLE_WINDOW_POOL, 0u), &keep->sequences,
				state->ple_conv_in, state->ple.conv, state->ple_conv_out, &conv);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsNgramHistory(state->stage.ops, queue, frame->token_ids, &keep->kept_rows, &keep->sequences,
				SparkDeviceStageRecurrent(&state->stage, QD_PLE_HISTORY_POOL, 0u), &ngram);
		return status;
	}

	static SparkStatus KeepRows(void *model, const SparkStageRunnerStep *step, const uint32_t *kept, void *queue)
	{
		State *state = (State *)model;
		SparkDeviceKeepTensors keep;
		SparkDeviceStepTensors frame;
		SparkStatus status = SPARK_STATUS_OK;
		if constexpr ( L::kPle )
			status = SparkDeviceStageStepTables(&state->stage, step, &frame);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageKeepTables(&state->stage, step, kept, &keep);
		for ( uint32_t index = 0u; index < G::kLayers && status == SPARK_STATUS_OK; index++ )
			if ( !L::Attention(index) )
				status = SparkDeviceGatedDeltaKeep(&state->stage, (SparkDeviceQueue *)queue, &state->delta_shape, &keep, &state->layers[index].delta,
					state->layers[index].recurrent_index, &state->delta);
		if constexpr ( L::kPle )
			if ( status == SPARK_STATUS_OK && keep.rows != 0u )
				status = KeepPle(state, (SparkDeviceQueue *)queue, &frame, &keep);
		return status;
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
			KeepRows,
			0,
			BindDevice,
			0
		};
		return &interface_table;
	}
};

struct Qwen38_27bDevice
{
	using Geometry = Qwen38_27bGeometry<1u>;
	static constexpr const char *kTag = "sparkpipe_qwen38_27b_device";
	static constexpr const char *kName = "qwen3.8-27b";
	static constexpr uint32_t kProjection = SPARK_PROJECTION_ROW_TILES;
};

const SparkStageRunnerModelInterface *SparkQwen38_27bDeviceStageModel(void)
{
	return QwenDevice<Qwen38_27bDevice>::Interface();
}

struct Qwen38FlashDevice
{
	using Geometry = Qwen38FlashGeometry<1u>;
	static constexpr const char *kTag = "sparkpipe_qwen38_flash_device";
	static constexpr const char *kName = "qwen3.8-flash";
	static constexpr uint32_t kProjection = SPARK_PROJECTION_ROW_TILES;
};

const SparkStageRunnerModelInterface *SparkQwen38FlashDeviceStageModel(void)
{
	return QwenDevice<Qwen38FlashDevice>::Interface();
}
