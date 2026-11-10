#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inference/runner/device_layers.h"
#include "inference/runner/device_stage_runner.h"
#include "sparkpipe/spark_glm52_device_stage.h"
#include "sparkpipe/spark_glm52_family.h"
#include "sparkpipe/spark_glm52_model.h"
#include "sparkpipe/spark_glm52_stagepack.h"
#include "sparkpipe/spark_axk2_tp16_stage_model.h"

#define GF_SLICES 40u
#define GF_ALIGNMENT 128u
#define GF_INDEX_ROWS 16u

template<class P>
struct Glm52Device
{
	using G = typename P::Geometry;

	static constexpr uint32_t kTp = G::kTpDegree;
	static constexpr uint32_t kHidden = G::kHidden;
	static constexpr uint32_t kLayers = G::kLayers;
	static constexpr uint32_t kVocab = G::kVocab;
	static constexpr uint32_t kHeads = G::kHeads / kTp;
	static constexpr uint32_t kNope = G::kNope;
	static constexpr uint32_t kRope = G::kRope;
	static constexpr uint32_t kQkDim = kNope + kRope;
	static constexpr uint32_t kQueryHead = kQkDim + G::kQueryGate;
	static constexpr uint32_t kLatent = G::kLatent;
	static constexpr uint32_t kSlot = kLatent + kRope;
	static constexpr uint32_t kValueDim = G::kValue;
	static constexpr uint32_t kQueryLora = G::kQueryA;
	static constexpr uint32_t kQueryInput = G::kQueryConcat ? 2u * G::kQueryA : G::kQueryA;
	static constexpr uint32_t kExperts = G::kExperts;
	static constexpr uint32_t kTopK = G::kTopK;
	static constexpr uint32_t kMoeRows = G::kExpertInter / kTp;
	static constexpr uint32_t kDenseRows = G::kDenseInter / kTp;
	static constexpr uint32_t kDenseKeys = G::kSelected;
	static constexpr uint32_t kGateRank = G::kNormGateRank;
	static constexpr uint32_t kScaleBlock = 128u;
	static constexpr float kEpsilon = G::kRmsEpsilon;
	static constexpr uint32_t kIndexHeads = G::kIndexHeads;
	static constexpr uint32_t kIndexDim = G::kIndexDim;
	static constexpr uint32_t kIndexQuery = kIndexHeads * kIndexDim;
	static_assert(kDenseKeys <= SPARK_INDEXER_MAX_TOPK && kIndexHeads <= SPARK_INDEXER_MAX_HEADS && kIndexDim <= SPARK_INDEXER_MAX_HEAD_DIM,
		"the indexer's heads, width and top-k fit the shared indexer");
	static_assert(kRope <= kIndexDim, "the indexer ropes the leading part of its heads");

	struct Layer
	{
		SparkTensor input_norm;
		SparkTensor post_norm;
		SparkDeviceLowRankGateWeights input_gate;
		SparkDeviceLowRankGateWeights post_gate;
		SparkDeviceLatentAttentionWeights attention;
		SparkDeviceFfnWeights ffn;
		SparkDeviceFfnWeights shared;
		SparkTensor router;
		SparkTensor router_bias;
		SparkDeviceExpertMatrix gate;
		SparkDeviceExpertMatrix up;
		SparkDeviceExpertMatrix down;
		SparkDeviceWeight index_query;
		SparkDeviceWeight index_key;
		SparkDeviceWeight index_head;
		SparkTensor index_norm;
		SparkTensor index_norm_bias;
	};

	struct State
	{
		SparkDeviceStage stage;
		SparkDeviceStageShape shape;
		SparkDeviceLatentAttentionShape attention_shape;
		SparkDeviceLatentAttentionRope rope;
		Layer layers[kLayers];
		SparkTensor head_norm;
		SparkTensor normed;
		SparkTensor partial;
		SparkTensor shared_out;
		SparkTensor router_logits;
		SparkDeviceLowRankGateScratch gate;
		SparkDeviceLatentAttentionScratch attention;
		SparkDeviceFfnScratch ffn_scratch;
		SparkDeviceExpertsScratch experts;
		SparkIndexerShape indexer;
		SparkHeadRopeParams index_query_rope;
		SparkHeadRopeParams index_key_rope;
		uint32_t index_ordinal[kLayers];
		uint32_t index_layers;
		uint32_t index_rows;
		uint32_t score_stride;
		SparkTensor index_query;
		SparkTensor index_key;
		SparkTensor index_weight;
		SparkTensor index_scores;
		SparkTensor selected;
	};

	struct Frame
	{
		SparkDeviceStepTensors t;
		SparkDeviceQueue *queue;
		uint32_t waves;
		bool sparse;
	};

	struct Select
	{
		State *state;
		Frame *frame;
		uint32_t layer;
	};

	static uint64_t Max(uint64_t a, uint64_t b)
	{
		return a > b ? a : b;
	}

	static void IndexerShape(State *state)
	{
		SparkIndexerShape *indexer = &state->indexer;
		memset(indexer, 0, sizeof(*indexer));
		indexer->heads = kIndexHeads;
		indexer->head_dimension = kIndexDim;
		indexer->key_norm = SPARK_INDEXER_NORM_LAYER;
		indexer->key_norm_epsilon = G::kIndexEpsilon;
		indexer->score_scale = G::kIndexScale / sqrtf((float)kIndexHeads);
		indexer->topk = kDenseKeys;
		indexer->key_ratio = 1u;
		indexer->query_stride = kIndexQuery;
		indexer->weight_stride = kIndexHeads;
		indexer->key_stride = kIndexDim;
		state->index_layers = 0u;
		for ( uint32_t layer = 0u; layer < kLayers; layer++ )
		{
			state->index_ordinal[layer] = G::Indexer(layer) ? state->index_layers : SPARK_INDEXER_NONE;
			state->index_layers += G::Indexer(layer) ? 1u : 0u;
		}
	}

	static void IndexerRope(State *state)
	{
		SparkHeadRopeParams *query = &state->index_query_rope, *key = &state->index_key_rope;
		*query = state->rope.query;
		query->heads = kIndexHeads;
		query->head_dimension = kRope;
		query->input_stride = query->output_stride = kIndexQuery;
		query->input_offset = 0u;
		query->input_head_stride = query->output_head_stride = kIndexDim;
		query->rope_dimension = kRope;
		query->interleaved = G::kIndexRopeHalf ? 0u : 1u;
		*key = *query;
		key->heads = 1u;
		key->input_stride = key->output_stride = kIndexDim;
	}

	static void Shapes(State *state)
	{
		SparkDeviceStageShape *shape = &state->shape;
		SparkDeviceLatentAttentionShape *attention = &state->attention_shape;
		IndexerShape(state);
		memset(shape, 0, sizeof(*shape));
		shape->tag = P::kTag;
		shape->hidden = kHidden;
		shape->vocab = kVocab;
		shape->layers = kLayers;
		shape->kv_layers = kLayers;
		shape->kv_heads = 1u;
		shape->kv_head_dimension = (kSlot + kLatent) / 2u;
		shape->pack_alignment = GF_ALIGNMENT;
		shape->epsilon = kEpsilon;
		shape->projection = P::kProjection;
		shape->prenormed_head = 1u;
		shape->norm_round_first = G::kNormRoundFirst ? 1u : 0u;
		shape->index_layers = state->index_layers;
		shape->index_page_bytes = SparkIndexerCacheBytes(&state->indexer, 1u, SPARK_DEVICE_RUNNER_PAGE_SLOTS);
		memset(attention, 0, sizeof(*attention));
		attention->heads = G::kHeads;
		attention->nope = kNope;
		attention->rope = kRope;
		attention->latent = kLatent;
		attention->value_dimension = kValueDim;
		attention->rope_theta = G::kRopeTheta;
		attention->rope_rounding = G::kRopeRoundedTerms ? SPARK_ROPE_ROUNDING_TERMS : SPARK_ROPE_ROUNDING_FP32;
		attention->epsilon = kEpsilon;
		attention->scale = G::kQkScale;
		attention->query_lora = kQueryLora;
		attention->query_concat = G::kQueryConcat ? 1u : 0u;
		attention->query_gate = G::kQueryGate;
		attention->rope_factor = G::kRopeFactor > 1.0f ? G::kRopeFactor : 0.0f;
		attention->rope_beta_fast = G::kRopeBetaFast;
		attention->rope_beta_slow = G::kRopeBetaSlow;
		attention->rope_original = G::kRopeOriginalPositions;
	}

	static uint32_t ScratchSlices(State *state, SparkDeviceStageSlice *slices)
	{
		const uint64_t rows = state->stage.max_rows, hidden = kHidden, routes = rows * kTopK, heads = kHeads;
		const uint64_t ffn = Max(kDenseRows, kMoeRows);
		const uint64_t records = Max(SparkOpsAttentionScratchBytes(state->stage.max_rows, kHeads, kLatent, state->stage.max_positions),
			SparkIndexerAttentionScratchBytes(state->stage.max_rows, kHeads, kLatent, kDenseKeys));
		const SparkDeviceStageSlice plan[] =
		{
			{ rows * hidden * 2u, &state->normed },
			{ rows * hidden * 2u, &state->partial },
			{ rows * hidden * 2u, &state->shared_out },
			{ rows * kExperts * sizeof(float), &state->router_logits },
			{ rows * Max(kGateRank, 8u) * 2u, &state->gate.bottleneck },
			{ rows * hidden * 2u, &state->gate.gate },
			{ rows * kQueryInput * 2u, &state->attention.query_lora },
			{ rows * heads * kQueryHead * 2u, &state->attention.query },
			{ rows * kSlot * 2u, &state->attention.kv },
			{ rows * heads * kSlot * 2u, &state->attention.packed },
			{ records, &state->attention.records },
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
			{ rows * kIndexQuery * 2u, &state->index_query },
			{ rows * kIndexDim * 2u, &state->index_key },
			{ rows * kIndexHeads * 2u, &state->index_weight },
			{ SparkIndexerScoreBytes(state->index_rows, state->score_stride), &state->index_scores },
			{ SparkIndexerSelectedBytes(state->stage.max_rows, kDenseKeys), &state->selected },
			{ Max(2u * ffn, Max((uint64_t)kQueryLora, Max(Max(heads * kQueryHead, (uint64_t)kIndexQuery), Max((uint64_t)kSlot, hidden)))) *
				Max(hidden, (uint64_t)kQueryInput) * 2u, &state->stage.dequant }
		};
		const uint32_t count = sizeof(plan) / sizeof(plan[0]);
		for ( uint32_t index = 0u; index < count; index++ )
			slices[index] = plan[index];
		return count;
	}

	static SparkStatus Open(const SparkStageRunnerModelOpen *request, void **out, SparkStageRunnerModelGeometry *geometry)
	{
		SparkDeviceStageSlice slices[GF_SLICES];
		State *state = (State *)calloc(1u, sizeof(State));
		SparkStatus status;
		*out = state;
		if ( state == 0 )
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		Shapes(state);
		const SparkDeviceStageConfigValue config[] =
		{
			{ "hidden", kHidden }, { "layers", kLayers }, { "vocab", kVocab }, { "vocab_rows", kVocab / kTp }, { "experts", kExperts },
			{ "local_experts", kExperts }, { "top_k", kTopK }, { "ffn_rows", kDenseRows }, { "shared_rows", kMoeRows }, { "moe_rows", kMoeRows }
		};
		status = SparkDeviceStageOpen(&state->stage, &state->shape, request);
		if ( status == SPARK_STATUS_OK && state->shape.index_page_bytes == 0u )
			status = SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_UNSUPPORTED, "the indexer's key cache needs a page size");
		if ( status == SPARK_STATUS_OK )
		{
			state->index_rows = state->stage.max_rows < GF_INDEX_ROWS ? state->stage.max_rows : GF_INDEX_ROWS;
			state->score_stride = SparkIndexerScoreStride(state->stage.max_positions);
			status = SparkDeviceStageExpectConfig(&state->stage, config, sizeof(config) / sizeof(config[0]));
		}
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageReserve(&state->stage, SparkDeviceStagePlan(0, slices, ScratchSlices(state, slices)), 0, 0u,
				request->configuration->state_budget_bytes);
		if ( status != SPARK_STATUS_OK )
			return status;
		SparkDeviceLatentAttentionRopePrepare(&state->attention_shape, kHeads, &state->rope);
		IndexerRope(state);
		return SparkDeviceStageGeometry(&state->stage, geometry);
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

	static SparkStatus BindIndexer(State *state, const SparkStageRunnerDeviceWeights *weights, uint32_t index)
	{
		Layer *layer = &state->layers[index];
		SparkStatus status = SparkDeviceStageLayerWeight(&state->stage, weights, index, "index_q", kIndexQuery, kQueryLora, &layer->index_query);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerWeight(&state->stage, weights, index, "index_k", kIndexDim, kHidden, &layer->index_key);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerWeight(&state->stage, weights, index, "index_head", kIndexHeads, kHidden, &layer->index_head);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "index_norm", "f32", kIndexDim * sizeof(float), &layer->index_norm);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "index_norm_bias", "f32", kIndexDim * sizeof(float), &layer->index_norm_bias);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageNoteKvRead(&state->stage, 0u, state->shape.index_page_bytes / SPARK_DEVICE_RUNNER_PAGE_SLOTS);
		return status;
	}

	static SparkStatus BindLayer(State *state, const SparkStageRunnerDeviceWeights *weights, uint32_t index)
	{
		Layer *layer = &state->layers[index];
		SparkStatus status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "input_norm", "f32", kHidden * sizeof(float), &layer->input_norm);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "post_norm", "f32", kHidden * sizeof(float), &layer->post_norm);
		if ( status == SPARK_STATUS_OK && kGateRank != 0u )
			status = SparkDeviceLowRankGateBind(&state->stage, weights, index, "input_gate", kGateRank, &layer->input_gate);
		if ( status == SPARK_STATUS_OK && kGateRank != 0u && GlmLayout<G>::Routed(index) )
			status = SparkDeviceLowRankGateBind(&state->stage, weights, index, "post_gate", kGateRank, &layer->post_gate);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceLatentAttentionBind(&state->stage, weights, index, &state->attention_shape, &layer->attention);
		if ( status == SPARK_STATUS_OK && G::Indexer(index) )
			status = BindIndexer(state, weights, index);
		if ( status == SPARK_STATUS_OK && !GlmLayout<G>::Routed(index) )
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
		SparkDeviceStageSlice slices[GF_SLICES];
		SparkStatus status = SPARK_STATUS_OK;
		(void)geometry;
		if ( weights == 0 || weights->pack_address == 0u || weights->pack_bytes != state->stage.pack_bytes )
			return SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_INVALID_ARGUMENT, "bind_device needs the device mapping of the whole rank pack");
		for ( uint32_t index = 0u; index < kLayers && status == SPARK_STATUS_OK; index++ )
			status = BindLayer(state, weights, index);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageTensor(&state->stage, weights, "head_norm", "f32", kHidden * sizeof(float), &state->head_norm);
		if ( status != SPARK_STATUS_OK )
			return status;
		return SparkDeviceStageAllocate(&state->stage, slices, ScratchSlices(state, slices));
	}

	static SparkStatus Begin(State *state, const SparkStageRunnerStep *step, void *queue, Frame *frame)
	{
		frame->queue = (SparkDeviceQueue *)queue;
		frame->waves = 0u;
		frame->sparse = false;
		SparkStatus status = SparkDeviceStageStepTables(&state->stage, step, &frame->t);
		for ( uint32_t row = 0u; status == SPARK_STATUS_OK && row < frame->t.rows; row++ )
			frame->sparse = frame->sparse || SparkIndexerVisibleKeys(&state->indexer, frame->t.host_positions[row]) > kDenseKeys;
		if ( status == SPARK_STATUS_OK && frame->sparse && frame->t.tree != 0u )
			status = SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_UNSUPPORTED,
				"a tree step sees its keys through its ancestors, which the indexer's selection does not follow");
		return status;
	}

	static SparkStatus NextWave(State *state, Frame &frame, bool *more)
	{
		(void)state;
		*more = frame.waves++ == 0u;
		return SPARK_STATUS_OK;
	}

	static SparkStatus WaveBegin(State *state, Frame &frame)
	{
		(void)state;
		(void)frame;
		return SPARK_STATUS_OK;
	}

	static SparkStatus IndexKeys(State *state, Frame &frame, uint32_t layer)
	{
		const Layer &weights = state->layers[layer];
		const SparkIndexerCache cache = SparkDeviceStageIndexCache(&state->stage, state->index_ordinal[layer]);
		const uint32_t rows = frame.t.rows;
		SparkHeadRopeParams rope = state->index_key_rope;
		SparkStatus status = SparkDeviceLinearWeight(&state->stage, frame.queue, state->normed, kHidden, &weights.index_key, state->index_key, kIndexDim, rows,
			kHidden, kIndexDim);
		rope.rows = rows;
		if ( status == SPARK_STATUS_OK )
			status = SparkIndexerStoreKeys(state->stage.ops, frame.queue, &cache, state->index_key, &weights.index_norm, &weights.index_norm_bias, &rope,
				frame.t.sequence_of_row, frame.t.positions, frame.t.tree != 0u ? frame.t.slots : frame.t.positions, &state->indexer, rows);
		return status;
	}

	static SparkStatus IndexSelect(State *state, Frame &frame, uint32_t layer, SparkTensor query_input, uint32_t query_input_stride)
	{
		const Layer &weights = state->layers[layer];
		const SparkIndexerCache cache = SparkDeviceStageIndexCache(&state->stage, state->index_ordinal[layer]);
		const uint32_t rows = frame.t.rows;
		SparkHeadRopeParams rope = state->index_query_rope;
		SparkStatus status = SparkDeviceLinearWeight(&state->stage, frame.queue, query_input, query_input_stride, &weights.index_query, state->index_query,
			kIndexQuery, rows, kQueryLora, kIndexQuery);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceLinearWeight(&state->stage, frame.queue, state->normed, kHidden, &weights.index_head, state->index_weight, kIndexHeads, rows,
				kHidden, kIndexHeads);
		rope.rows = rows;
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsHeadRmsNormRope(state->stage.ops, frame.queue, state->index_query, state->index_query, frame.t.positions, state->index_query,
				&rope);
		for ( uint32_t first = 0u; status == SPARK_STATUS_OK && first < rows; first += state->index_rows )
		{
			const uint32_t count = rows - first < state->index_rows ? rows - first : state->index_rows;
			status = SparkIndexerSelect(state->stage.ops, frame.queue, &cache,
				SparkTensorSlice(state->index_query, (uint64_t)first * kIndexQuery * 2u, (uint64_t)count * kIndexQuery * 2u),
				SparkTensorSlice(state->index_weight, (uint64_t)first * kIndexHeads * 2u, (uint64_t)count * kIndexHeads * 2u),
				SparkTensorSlice(frame.t.sequence_of_row, (uint64_t)first * sizeof(uint32_t), (uint64_t)count * sizeof(uint32_t)),
				SparkTensorSlice(frame.t.positions, (uint64_t)first * sizeof(uint32_t), (uint64_t)count * sizeof(uint32_t)), state->index_scores,
				SparkTensorSlice(state->selected, SparkIndexerSelectedBytes(first, kDenseKeys), SparkIndexerSelectedBytes(count, kDenseKeys)),
				&state->indexer, count, state->score_stride);
		}
		return status;
	}

	static SparkStatus IndexLayer(void *context, SparkDeviceQueue *queue, SparkTensor query_input, uint32_t query_input_stride)
	{
		const Select *select = (const Select *)context;
		State *state = select->state;
		Frame &frame = *select->frame;
		SparkStatus status = SPARK_STATUS_OK;
		(void)queue;
		if ( G::Indexer(select->layer) )
			status = IndexKeys(state, frame, select->layer);
		if ( status == SPARK_STATUS_OK && G::Indexer(select->layer) && frame.sparse )
			status = IndexSelect(state, frame, select->layer, query_input, query_input_stride);
		return status;
	}

	static SparkStatus Attention(State *state, Frame &frame, uint32_t layer)
	{
		const SparkTensor addend = state->partial;
		Select select = { state, &frame, layer };
		SparkDeviceLatentSelection selection;
		SparkStatus status = SparkDeviceResidualNorm(&state->stage, frame.queue, &frame.t, layer == 0u ? 0 : &addend, state->layers[layer].input_norm,
			state->normed);
		if ( status == SPARK_STATUS_OK && kGateRank != 0u )
			status = SparkDeviceLowRankGate(&state->stage, frame.queue, frame.t.rows, kGateRank, &state->layers[layer].input_gate, &state->gate, state->normed);
		memset(&selection, 0, sizeof(selection));
		selection.select = IndexLayer;
		selection.context = &select;
		selection.sparse = frame.sparse ? 1u : 0u;
		selection.selected = state->selected;
		selection.topk = kDenseKeys;
		selection.key_ratio = 1u;
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceLatentAttentionSelected(&state->stage, frame.queue, &state->attention_shape, &state->rope, &frame.t,
				&state->layers[layer].attention, layer, &state->attention, &selection, state->normed, state->partial);
		return status;
	}

	static SparkStatus AttentionResidual(State *state, Frame &frame, uint32_t layer)
	{
		(void)layer;
		return SparkDeviceStageAllReduce(&state->stage, frame.queue, state->partial, frame.t.rows);
	}

	static SparkStatus PostNorm(State *state, Frame &frame, uint32_t layer)
	{
		const SparkTensor addend = state->partial;
		SparkStatus status = SparkDeviceResidualNorm(&state->stage, frame.queue, &frame.t, &addend, state->layers[layer].post_norm, state->normed);
		if ( status == SPARK_STATUS_OK && kGateRank != 0u && GlmLayout<G>::Routed(layer) )
			status = SparkDeviceLowRankGate(&state->stage, frame.queue, frame.t.rows, kGateRank, &state->layers[layer].post_gate, &state->gate, state->normed);
		return status;
	}

	static SparkStatus DenseMlp(State *state, Frame &frame, uint32_t layer)
	{
		SparkStatus status = PostNorm(state, frame, layer);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceGatedFfn(&state->stage, frame.queue, kDenseRows, SPARK_GATED_SILU, 0.0f, &frame.t, &state->layers[layer].ffn,
				&state->ffn_scratch, state->normed, state->partial);
		return status;
	}

	static SparkStatus RoutedMlp(State *state, Frame &frame, uint32_t layer)
	{
		const Layer &weights = state->layers[layer];
		const uint32_t rows = frame.t.rows;
		const SparkDeviceRoutedFfnShape ffn = { kExperts, kTopK, kMoeRows, SPARK_GATED_SILU, 0u, 0.0f, 0.0f, 0u };
		SparkGemvParams router;
		SparkRouteParams route;
		SparkStatus status = PostNorm(state, frame, layer);
		memset(&router, 0, sizeof(router));
		router.rows = rows;
		router.input_dimension = router.input_stride = kHidden;
		router.output_dimension = router.output_stride = kExperts;
		router.output_f32 = 1u;
		router.projection = state->stage.shape->projection;
		router.row_exact = state->stage.step_exact;
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsLinear(state->stage.ops, frame.queue, state->normed, weights.router, state->router_logits, &router);
		memset(&route, 0, sizeof(route));
		route.rows = rows;
		route.experts = kExperts;
		route.top_k = kTopK;
		route.width = kHidden;
		route.logit_stride = kExperts;
		route.scale = G::kRoutedScale;
		route.tree_sum = 1u;
		route.groups = G::kExpertGroups;
		route.topk_groups = G::kTopGroups;
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsRouteSigmoid(state->stage.ops, frame.queue, state->router_logits, weights.router_bias, state->experts.experts,
				state->experts.weights, &route);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceRoutedFfn(&state->stage, frame.queue, &ffn, rows, &weights.gate, &weights.up, &weights.down, &state->experts, state->normed);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsRouteCombineWeighted(state->stage.ops, frame.queue, state->experts.out, state->experts.weights, state->partial, &route);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceGatedFfn(&state->stage, frame.queue, kMoeRows, SPARK_GATED_SILU, 0.0f, &frame.t, &weights.shared, &state->ffn_scratch,
				state->normed, state->shared_out);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsAddRows(state->stage.ops, frame.queue, state->partial, state->shared_out, rows, kHidden, kHidden, kHidden);
		return status;
	}

	static SparkStatus MlpResidual(State *state, Frame &frame, uint32_t layer)
	{
		(void)layer;
		return SparkDeviceStageAllReduce(&state->stage, frame.queue, state->partial, frame.t.rows);
	}

	static SparkStatus LayerDone(State *state, Frame &frame, uint32_t layer)
	{
		SparkDeviceStageLayerDone(&state->stage, layer);
		return SparkDeviceStageDumpRows(&state->stage, frame.queue, &frame.t, frame.t.hidden, kHidden, layer);
	}

	static SparkStatus WaveEnd(State *state, Frame &frame)
	{
		SparkNormParams norm;
		memset(&norm, 0, sizeof(norm));
		norm.rows = frame.t.rows;
		norm.dimension = kHidden;
		norm.input_stride = norm.output_stride = kHidden;
		norm.has_addend = 1u;
		norm.epsilon = kEpsilon;
		norm.round_first = state->stage.shape->norm_round_first;
		return SparkOpsRmsNormResidual(state->stage.ops, frame.queue, frame.t.hidden, &state->partial, state->head_norm, frame.t.hidden, &norm);
	}

	static SparkStatus End(State *state, Frame &frame)
	{
		(void)state;
		(void)frame;
		return SPARK_STATUS_OK;
	}

	static SparkStatus Step(void *model, const SparkStageRunnerStep *step, void *queue)
	{
		return GlmSchedule<G, Glm52Device>::Step((State *)model, step, queue);
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
			0,
			0,
			0,
			0,
			0,
			0,
			0u,
			0u
		};
		return &interface_table;
	}
};

template<uint32_t TP>
struct Glm53FullDevice
{
	struct Geometry
	{
		static constexpr uint32_t kTpDegree = TP;
		static constexpr uint32_t kHidden = SPARK_GLM52_MODEL_HIDDEN_DIMENSION;
		static constexpr uint32_t kLayers = SPARK_GLM52_MODEL_LAYER_COUNT;
		static constexpr uint32_t kFirstRouted = SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER;
		static constexpr uint32_t kVocab = SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT;
		static constexpr uint32_t kHeads = SPARK_GLM52_MODEL_HEAD_COUNT;
		static constexpr uint32_t kLatent = SPARK_GLM52_MODEL_LATENT_DIMENSION;
		static constexpr uint32_t kRope = SPARK_GLM52_MODEL_ROPE_DIMENSION;
		static constexpr uint32_t kNope = SPARK_GLM52_MODEL_QK_NOPE_HEAD_DIMENSION;
		static constexpr uint32_t kValue = SPARK_GLM52_MODEL_VALUE_HEAD_DIMENSION;
		static constexpr uint32_t kQueryA = SPARK_GLM52_MODEL_QUERY_A_DIMENSION;
		static constexpr uint32_t kSelected = SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT;
		static constexpr uint32_t kExperts = SPARK_GLM52_MODEL_MOE_EXPERT_COUNT;
		static constexpr uint32_t kTopK = SPARK_GLM52_MODEL_MOE_TOP_K;
		static constexpr uint32_t kExpertInter = SPARK_GLM52_MODEL_MOE_INTERMEDIATE_DIMENSION;
		static constexpr uint32_t kDenseInter = SPARK_GLM52_MODEL_DENSE_INTERMEDIATE_DIMENSION;
		static constexpr float kRmsEpsilon = SPARK_GLM52_MODEL_RMS_NORM_EPSILON;
		static constexpr bool kNormRoundFirst = true;
		static constexpr bool kRopeRoundedTerms = true;
		static constexpr float kRopeTheta = SPARK_GLM52_MODEL_ROPE_THETA;
		static constexpr float kQkScale = SPARK_GLM52_MODEL_QK_SCALE;
		static constexpr float kRoutedScale = SPARK_GLM52_MODEL_MOE_ROUTED_SCALING_FACTOR;
		static constexpr uint32_t kQueryGate = 0u;
		static constexpr bool kQueryConcat = false;
		static constexpr uint32_t kNormGateRank = 0u;
		static constexpr uint32_t kExpertGroups = 1u;
		static constexpr uint32_t kTopGroups = 1u;
		static constexpr float kRopeFactor = 1.0f;
		static constexpr float kRopeBetaFast = 0.0f;
		static constexpr float kRopeBetaSlow = 0.0f;
		static constexpr uint32_t kRopeOriginalPositions = 0u;
		static constexpr uint32_t kIndexHeads = SPARK_GLM52_MODEL_DSA_INDEX_HEAD_COUNT;
		static constexpr uint32_t kIndexDim = SPARK_GLM52_MODEL_DSA_INDEX_HEAD_DIMENSION;
		static constexpr float kIndexEpsilon = SPARK_GLM52_MODEL_DSA_INDEX_NORM_EPSILON;
		static constexpr float kIndexScale = SPARK_GLM52_MODEL_DSA_INDEX_SOFTMAX_SCALE;
		static constexpr bool kIndexRopeHalf = SPARK_GLM52_MODEL_DSA_ROPE_INTERLEAVE == 0u;
		static bool Indexer(uint32_t layer)
		{
			return SparkGlm52StagePackLayerHasFullIndexer(layer) != 0u;
		}
	};
	static constexpr const char *kTag = "sparkpipe_glm53_full_device";
	static constexpr const char *kName = "glm-5.3-full";
	static constexpr uint32_t kProjection = SPARK_PROJECTION_GEMV;
};

template<uint32_t TP>
struct Axk2Device
{
	struct Geometry : Axk2Dimensions
	{
		static constexpr uint32_t kTpDegree = TP;
		static constexpr uint32_t kIndexHeads = 64u;
		static constexpr uint32_t kIndexDim = SPARK_AXK2_TP16_STAGE_INDEX_DIM;
		static constexpr float kIndexEpsilon = 1e-06f;
		static constexpr float kIndexScale = 0.0883883476f;
		static constexpr bool kIndexRopeHalf = true;
		static bool Indexer(uint32_t layer)
		{
			return layer < kLayers;
		}
	};
	static constexpr const char *kTag = "sparkpipe_axk2_device";
	static constexpr const char *kName = "ax-k2";
	static constexpr uint32_t kProjection = SPARK_PROJECTION_GEMV;
};

template<template<uint32_t> class P>
static const SparkStageRunnerModelInterface *Glm52DeviceAt(uint32_t tp_degree)
{
	switch ( tp_degree )
	{
	case 1u:
		return Glm52Device<P<1u>>::Interface();
	case 2u:
		return Glm52Device<P<2u>>::Interface();
	case 4u:
		return Glm52Device<P<4u>>::Interface();
	case 8u:
		return Glm52Device<P<8u>>::Interface();
	default:
		return 0;
	}
}

const SparkStageRunnerModelInterface *SparkGlm53FullDeviceStageModelAt(uint32_t tp_degree)
{
	return Glm52DeviceAt<Glm53FullDevice>(tp_degree);
}

const SparkStageRunnerModelInterface *SparkAxk2DeviceStageModelAt(uint32_t tp_degree)
{
	return Glm52DeviceAt<Axk2Device>(tp_degree);
}
