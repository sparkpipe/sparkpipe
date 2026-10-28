#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "inference/runner/device_layers.h"
#include "sparkpipe/spark_qwen38_27b_stage_model.h"
#include "sparkpipe/spark_qwen38_device_stage.h"
#include "sparkpipe/spark_qwen38_family.h"

#define QD_SLICES 16u

template<class P>
struct QwenDevice
{
	using G = typename P::Geometry;
	using L = QwenHybridLayout<G>;

	static_assert(G::kTpDegree == 1u, "a device stage is one whole rank");
	static_assert(!L::kHyper && !L::kPle && !L::kIndexer && !G::kMoe && !G::kRankHeads && !G::kGdnSigmoidGate,
		"the device layers do not cover this Qwen variant yet");

	struct Layer
	{
		SparkTensor input_norm;
		SparkTensor post_norm;
		SparkDeviceFfnWeights ffn;
		SparkDeviceGatedDeltaWeights delta;
		SparkDeviceAttentionWeights attention;
		uint32_t recurrent_index;
		uint32_t kv_index;
	};

	struct State
	{
		SparkDeviceStage stage;
		SparkDeviceStageShape shape;
		SparkDeviceGatedDeltaShape delta_shape;
		SparkDeviceAttentionShape attention_shape;
		uint32_t ffn;
		Layer layers[G::kLayers];
		SparkTensor normed;
		SparkTensor partial;
		SparkDeviceGatedDeltaScratch delta;
		SparkDeviceAttentionScratch attention;
		SparkDeviceFfnScratch ffn_scratch;
		SparkDeviceAttentionRope rope;
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
		memset(delta, 0, sizeof(*delta));
		delta->key_heads = G::kGdnKeyHeads;
		delta->value_heads = G::kGdnValueHeads;
		delta->key_dimension = QH_KEY_DIM;
		delta->value_dimension = QH_VALUE_DIM;
		delta->conv_taps = QH_CONV;
		delta->epsilon = QH_EPSILON;
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
		const uint64_t rows = state->stage.max_rows, hidden = G::kHidden;
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
			{ SparkDeviceGatedDeltaKeepBytes(delta, state->stage.verify_rows, L::kGdnLayers), &state->delta.keep }
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
		const uint64_t widths[SPARK_DEVICE_GATED_DELTA_POOLS] = { SparkDeviceGatedDeltaStateBytes(&state->delta_shape),
			SparkDeviceGatedDeltaWindowBytes(&state->delta_shape) };
		const SparkDeviceStageConfigValue config[] = { { "hidden", G::kHidden }, { "layers", G::kLayers }, { "vocab_rows", G::kVocab } };
		status = SparkDeviceStageOpen(&state->stage, &state->shape, request);
		if ( status == SPARK_STATUS_OK && SparkNamedPackConfigU32(&state->stage.pack, "ffn_rows", &state->ffn) != SPARK_STATUS_OK )
			status = SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_PARSE_ERROR, "the pack configuration lacks ffn_rows");
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageExpectConfig(&state->stage, config, sizeof(config) / sizeof(config[0]));
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageReserve(&state->stage, SparkDeviceStagePlan(0, slices, ScratchSlices(state, slices)), L::kGdnLayers, widths,
				SPARK_DEVICE_GATED_DELTA_POOLS, request->configuration->state_budget_bytes);
		if ( status != SPARK_STATUS_OK )
			return status;
		SparkDeviceAttentionRopePrepare(&state->attention_shape, &state->rope);
		return SparkDeviceStageGeometry(&state->stage, geometry);
	}

	static SparkStatus BindLayer(State *state, const SparkStageRunnerDeviceWeights *weights, uint32_t index)
	{
		Layer *layer = &state->layers[index];
		SparkStatus status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "input_norm", "f32", G::kHidden * 4u, &layer->input_norm);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "post_norm", "f32", G::kHidden * 4u, &layer->post_norm);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceFfnBind(&state->stage, weights, index, state->ffn, &layer->ffn);
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
		return SparkDeviceStageStepTables(&state->stage, step, &frame->t);
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

	static SparkStatus Ffn(State *state, const Frame &frame, const Layer &layer)
	{
		return SparkDeviceGatedFfn(&state->stage, frame.queue, state->ffn, SPARK_GATED_SILU, &frame.t, &layer.ffn, &state->ffn_scratch, state->normed,
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
		(void)frame;
		SparkDeviceStageLayerDone(&state->stage, index);
		return SPARK_STATUS_OK;
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

	static SparkStatus KeepRows(void *model, const SparkStageRunnerStep *step, const uint32_t *kept, void *queue)
	{
		State *state = (State *)model;
		SparkDeviceKeepTensors keep;
		SparkStatus status = SparkDeviceStageKeepTables(&state->stage, step, kept, &keep);
		for ( uint32_t index = 0u; index < G::kLayers && status == SPARK_STATUS_OK; index++ )
			if ( !L::Attention(index) )
				status = SparkDeviceGatedDeltaKeep(&state->stage, (SparkDeviceQueue *)queue, &state->delta_shape, &keep, &state->layers[index].delta,
					state->layers[index].recurrent_index, &state->delta);
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
