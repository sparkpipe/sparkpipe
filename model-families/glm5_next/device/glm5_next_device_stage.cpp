#include <math.h>

#include "inference/runner/device_indexer.h"
#include "inference/runner/device_stage_cxx.h"
#include "inference/runner/device_stage_runner.h"
#include "sparkpipe/spark_glm5_next_device_stage.h"
#include "sparkpipe/spark_glm5_next_family.h"
#include "sparkpipe/spark_glm53_flash_tp16_stage_model.h"

#define GD_ALIGNMENT 128u

template<class P>
struct Glm5NextDevice : SparkDeviceStageModel<Glm5NextDevice<P>>
{
	using G = typename P::Geometry;
	using L = Glm5NextLayout;

	static constexpr uint32_t kMaxSlices = 49u;
	static constexpr const char *Tag(void) { return P::kTag; }
	static constexpr const char *Name(void) { return P::kName; }

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
	static constexpr uint32_t kIndexHeads = SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_COUNT;
	static constexpr uint32_t kIndexDim = SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_DIMENSION;
	static constexpr uint32_t kIndexQuery = SPARK_GLM5_NEXT_MODEL_INDEX_QUERY_DIMENSION;
	static constexpr uint32_t kPool = SPARK_GLM5_NEXT_MODEL_INDEX_KPOOL;
	static constexpr uint32_t kSelectedPools = SPARK_GLM5_NEXT_MODEL_INDEX_POOL_SELECT_COUNT;
	static constexpr uint32_t kExpandedKeys = SPARK_GLM5_NEXT_MODEL_INDEX_OUTPUT_WIDTH;
	static constexpr uint32_t kScaleBlock = SPARK_GLM5_NEXT_MODEL_FP8_SCALE_BLOCK;
	static constexpr float kEpsilon = SPARK_GLM5_NEXT_MODEL_RMS_NORM_EPSILON;

	struct Indexer
	{
		SparkDeviceWeight query;
		SparkDeviceWeight key;
		SparkDeviceWeight gate;
		SparkDeviceWeight head;
		SparkTensor norm;
		SparkTensor norm_bias;
		SparkTensor ape;
	};

	struct IndexScratch
	{
		SparkTensor query;
		SparkTensor key;
		SparkTensor gate;
		SparkTensor head;
		SparkTensor scores;
		SparkTensor pools;
		SparkTensor selected;
		SparkTensor partial;
	};

	struct Layer
	{
		SparkDeviceHyper hyper[2];
		SparkTensor norm[2];
		SparkDeviceChannelDeltaWeights delta;
		SparkDeviceLatentAttentionWeights attention;
		Indexer indexer;
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
		SparkIndexerShape index_shape;
		uint32_t score_stride;
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
		IndexScratch index;
		SparkDeviceFfnScratch ffn_scratch;
		SparkDeviceExpertsScratch experts;
	};

	struct Frame
	{
		SparkDeviceStepTensors t;
		SparkDeviceQueue *queue;
		const Layer *layer;
		uint32_t sparse_rows;
	};

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
		memset(&state->index_shape, 0, sizeof(state->index_shape));
		state->index_shape.heads = kIndexHeads;
		state->index_shape.head_dimension = kIndexDim;
		state->index_shape.key_norm = SPARK_INDEXER_NORM_LAYER;
		state->index_shape.key_norm_epsilon = SPARK_GLM5_NEXT_MODEL_INDEX_NORM_EPSILON;
		state->index_shape.score_scale = SPARK_GLM5_NEXT_MODEL_INDEX_SOFTMAX_SCALE;
		state->index_shape.topk = kSelectedPools;
		state->index_shape.key_ratio = kPool;
		state->index_shape.query_stride = kIndexQuery;
		state->index_shape.weight_stride = kIndexHeads;
		state->index_shape.weight_scale = SPARK_GLM5_NEXT_MODEL_INDEX_HEAD_WEIGHT_SCALE;
		state->index_shape.key_stride = kIndexDim;
		state->index_shape.gate_stride = kIndexDim;
		shape->index_layers = kMlaLayers;
		shape->index_page_bytes = (uint64_t)SPARK_DEVICE_RUNNER_PAGE_SLOTS * SparkIndexerSlotPitch(&state->index_shape, 1u) * sizeof(uint16_t);
	}

	static uint32_t ScratchSlices(State *state, SparkDeviceStageSlice *slices)
	{
		const uint64_t rows = state->stage.max_rows, hidden = kHidden, routes = rows * kTopK, heads = kMlaHeads, width = kKdaWidth;
		const uint64_t ffn = SparkDeviceStageMax(kDenseRows, kMoeRows);
		state->score_stride = SparkIndexerScoreStride(state->stage.max_positions / kPool);
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
			{ rows * heads * kNope * 2u, &state->attention.query },
			{ rows * kLatent * 2u, &state->attention.kv },
			{ rows * heads * kLatent * 2u, &state->attention.packed },
			{ SparkOpsAttentionScratchBytes(state->stage.max_rows, kMlaHeads, kLatent, state->stage.max_positions), &state->attention.records },
			{ rows * heads * kLatent * 2u, &state->attention.latent },
			{ rows * heads * kValueDim * 2u, &state->attention.value },
			{ rows * kIndexQuery * 2u, &state->index.query },
			{ rows * kIndexDim * 2u, &state->index.key },
			{ rows * kIndexDim * 2u, &state->index.gate },
			{ rows * kIndexHeads * 2u, &state->index.head },
			{ SparkIndexerScoreBytes(state->stage.max_rows, state->score_stride), &state->index.scores },
			{ SparkIndexerSelectedBytes(state->stage.max_rows, kSelectedPools), &state->index.pools },
			{ SparkIndexerSelectedBytes(state->stage.max_rows, kExpandedKeys), &state->index.selected },
			{ SparkIndexerAttentionScratchBytes(state->stage.max_rows, kMlaHeads, kLatent, kExpandedKeys), &state->index.partial },
			{ rows * 2u * ffn * 2u, &state->ffn_scratch.gate_up },
			{ rows * ffn * 2u, &state->ffn_scratch.act },
			{ SparkDeviceStageMax(2u * ffn, SparkDeviceStageMax((uint64_t)kQueryLora, SparkDeviceStageMax(heads * kNope, SparkDeviceStageMax(3u * width, hidden)))) *
				SparkDeviceStageMax(hidden, (uint64_t)kQueryLora) * 2u, &state->stage.dequant }
		};
		uint32_t count = SparkDeviceStageAppendSlices(slices, 0u, plan);
		count = SparkDeviceStageAppendChannelDeltaScratch(&state->stage, width, kKdaHeads, kKdaDim, SparkDeviceStageMax(kKdaRank, 64u),
			&state->delta, slices, count);
		return SparkDeviceStageAppendRoutedScratch(&state->experts, routes, kMoeRows, hidden, slices, count);
	}

	static uint32_t RecurrentPools(const State *state, SparkDeviceStageRecurrentPlan *pools)
	{
		pools[0] = { kKdaLayers, SparkDeviceChannelDeltaStateBytes(&state->delta_shape) };
		pools[1] = { kKdaLayers, SparkDeviceChannelDeltaWindowBytes(&state->delta_shape) };
		return SPARK_DEVICE_GATED_DELTA_POOLS;
	}

	static SparkStatus Configure(State *state, const SparkStageRunnerModelOpen *request)
	{
		(void)request;
		const SparkDeviceStageConfigValue config[] =
		{
			{ "hidden", kHidden }, { "layers", kLayers }, { "vocab", kVocab }, { "vocab_rows", kVocab / kTp }, { "experts", kExperts },
			{ "local_experts", kExperts }, { "top_k", kTopK }, { "ffn_rows", kDenseRows }, { "shared_rows", kMoeRows }, { "moe_rows", kMoeRows }
		};
		return SparkDeviceStageExpectConfig(&state->stage, config, sizeof(config) / sizeof(config[0]));
	}

	static SparkStatus AfterReserve(State *state, const SparkStageRunnerModelOpen *request)
	{
		(void)request;
		if ( SparkIndexerExpandWidth(&state->index_shape) != kExpandedKeys )
			return SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_INVALID_ARGUMENT, "the expanded selection holds every selected pool and the open pool");
		SparkDeviceLatentAttentionRopePrepare(&state->attention_shape, kMlaHeads, &state->rope);
		return SPARK_STATUS_OK;
	}

	static SparkStatus BindIndexer(State *state, const SparkStageRunnerDeviceWeights *weights, uint32_t index, Indexer *indexer)
	{
		SparkStatus status = SparkDeviceStageLayerWeight(&state->stage, weights, index, "index_q", kIndexQuery, kQueryLora, &indexer->query);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerWeight(&state->stage, weights, index, "index_k", kIndexDim, kHidden, &indexer->key);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerWeight(&state->stage, weights, index, "index_gate", kIndexDim, kHidden, &indexer->gate);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerWeight(&state->stage, weights, index, "index_head", kIndexHeads, kHidden, &indexer->head);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "index_norm", "f32", kIndexDim * sizeof(float), &indexer->norm);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "index_norm_bias", "f32", kIndexDim * sizeof(float), &indexer->norm_bias);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "index_ape", "f32", (uint64_t)kPool * kIndexDim * sizeof(float),
				&indexer->ape);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageNoteKvRead(&state->stage, 0u, state->stage.shape->index_page_bytes / SPARK_DEVICE_RUNNER_PAGE_SLOTS);
		return status;
	}

	static SparkStatus BindLayerWeights(State *state, const SparkStageRunnerDeviceWeights *weights, uint32_t index, uint32_t *recurrent, uint32_t *kv)
	{
		Layer *layer = &state->layers[index];
		SparkStatus status = SparkDeviceStageBindLayerHyper(&state->stage, weights, index, "attn_hc", "_", SPARK_HC_COMB_MIXES, kResidual,
			SPARK_GLM5_NEXT_MODEL_HC_SCALE_COUNT, &layer->hyper[GLM5_NEXT_ATTENTION_PART]);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageBindLayerHyper(&state->stage, weights, index, "ffn_hc", "_", SPARK_HC_COMB_MIXES, kResidual,
				SPARK_GLM5_NEXT_MODEL_HC_SCALE_COUNT, &layer->hyper[GLM5_NEXT_FFN_PART]);
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
			if ( status == SPARK_STATUS_OK )
				status = BindIndexer(state, weights, index, &layer->indexer);
		}
		if ( status == SPARK_STATUS_OK && !L::Routed(index) )
			return SparkDeviceFfnBind(&state->stage, weights, index, kDenseRows, &layer->ffn);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "router", "bf16", (uint64_t)kExperts * kHidden * 2u, &layer->router);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "router_bias", "f32", (uint64_t)kExperts * sizeof(float), &layer->router_bias);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageBindExpertFp8(&state->stage, weights, index, "moe_w1", kExperts, kTopK, kMoeRows, kHidden, kScaleBlock, &layer->gate);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageBindExpertFp8(&state->stage, weights, index, "moe_w3", kExperts, kTopK, kMoeRows, kHidden, kScaleBlock, &layer->up);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageBindExpertFp8(&state->stage, weights, index, "moe_w2", kExperts, kTopK, kHidden, kMoeRows, kScaleBlock, &layer->down);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceFfnBindNamed(&state->stage, weights, index, "shared_gate_up", "shared_down", kMoeRows, &layer->shared);
		return status;
	}

	static SparkStatus BindWeights(State *state, const SparkStageRunnerDeviceWeights *weights)
	{
		uint32_t recurrent = 0u, kv = 0u;
		SparkStatus status = SparkDeviceStageTensor(&state->stage, weights, "final_norm", "bf16", kHidden * 2u, &state->final_norm);
		for ( uint32_t index = 0u; index < kLayers && status == SPARK_STATUS_OK; index++ )
			status = BindLayerWeights(state, weights, index, &recurrent, &kv);
		return status;
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

	static SparkStatus Begin(State *state, const SparkStageRunnerStep *step, void *queue, Frame *frame)
	{
		frame->queue = (SparkDeviceQueue *)queue;
		frame->layer = 0;
		SparkStatus status = SparkDeviceStageStepTables(&state->stage, step, &frame->t);
		frame->sparse_rows = 0u;
		if ( status == SPARK_STATUS_OK && frame->t.host_positions == 0 )
			status = SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_INVALID_ARGUMENT, "the indexer needs the step's host positions");
		for ( uint32_t row = 0u; status == SPARK_STATUS_OK && row < frame->t.rows; row++ )
			frame->sparse_rows += Sparse(frame->t.host_positions[row]) ? 1u : 0u;
		if ( status == SPARK_STATUS_OK && frame->sparse_rows != 0u && frame->t.tree != 0u )
			status = SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_INVALID_ARGUMENT, "a tree frame attends densely: every row must see at most the indexer's selected keys");
		return status;
	}

	static SparkStatus HyperBegin(State *state, Frame &frame)
	{
		return SparkDeviceStageHyperExpand(&state->stage, frame.queue, frame.t.hidden, state->streams, frame.t.rows, kStreams, kHidden);
	}

	static void BindLayer(State *state, Frame &frame, uint32_t layer) { frame.layer = &state->layers[layer]; }

	static SparkStatus HyperNorm(State *state, Frame &frame, uint32_t layer, Glm5NextPart part)
	{
		const SparkDeviceHyper &hyper = frame.layer->hyper[part];
		const SparkHcParams params = HcParams(frame);
		const SparkNormParams norm = SparkDeviceStageNormParams(frame.t.rows, kHidden, kEpsilon);
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

	static bool Sparse(uint32_t position) { return position >= kDenseKeys; }

	static SparkIndexerCache IndexCache(const State *state, const Frame &frame)
	{
		SparkIndexerCache cache = SparkDeviceStageIndexCache(&state->stage, frame.layer->kv_index);
		cache.slot_pitch = SparkIndexerSlotPitch(&state->index_shape, 1u);
		return cache;
	}

	static SparkStatus IndexKeys(State *state, Frame &frame, const SparkIndexerCache &cache)
	{
		const Indexer &indexer = frame.layer->indexer;
		const uint32_t rows = frame.t.rows;
		SparkStatus status = SparkDeviceLinearWeight(&state->stage, frame.queue, state->normed, kHidden, &indexer.key, state->index.key, kIndexDim, rows,
			kHidden, kIndexDim);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceLinearWeight(&state->stage, frame.queue, state->normed, kHidden, &indexer.gate, state->index.gate, kIndexDim, rows, kHidden,
				kIndexDim);
		if ( status == SPARK_STATUS_OK )
			status = SparkIndexerStoreGatedKeys(state->stage.ops, frame.queue, &cache, state->index.key, state->index.gate, &indexer.norm, &indexer.norm_bias, 0,
				frame.t.sequence_of_row, frame.t.positions, frame.t.positions, &state->index_shape, rows);
		return status;
	}

	static SparkStatus IndexSelect(State *state, Frame &frame, const SparkIndexerCache &cache)
	{
		const Indexer &indexer = frame.layer->indexer;
		const uint32_t rows = frame.t.rows;
		SparkStatus status = SparkDeviceLinearWeight(&state->stage, frame.queue, state->attention.query_lora, kQueryLora, &indexer.query, state->index.query,
			kIndexQuery, rows, kQueryLora, kIndexQuery);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceLinearWeight(&state->stage, frame.queue, state->normed, kHidden, &indexer.head, state->index.head, kIndexHeads, rows, kHidden,
				kIndexHeads);
		if ( status == SPARK_STATUS_OK )
			status = SparkIndexerScorePooled(state->stage.ops, frame.queue, &cache, state->index.query, state->index.head, indexer.ape, frame.t.sequence_of_row,
				frame.t.positions, state->index.scores, &state->index_shape, rows, state->score_stride);
		if ( status == SPARK_STATUS_OK )
			status = SparkIndexerTopk(state->stage.ops, frame.queue, state->index.scores, state->index.pools, rows, state->score_stride, kSelectedPools);
		if ( status == SPARK_STATUS_OK )
			status = SparkIndexerExpandPools(state->stage.ops, frame.queue, state->index.pools, frame.t.positions, state->index.selected, &state->index_shape,
				rows, kExpandedKeys);
		return status;
	}

	static SparkStatus AttendSelected(State *state, Frame &frame, uint32_t first, uint32_t count)
	{
		SparkDeviceLatentSelection selection;
		memset(&selection, 0, sizeof(selection));
		selection.sparse = 1u;
		selection.selected = state->index.selected;
		selection.topk = kExpandedKeys;
		selection.key_ratio = 1u;
		return SparkDeviceLatentAttentionAttendSelected(&state->stage, frame.queue, &state->attention_shape, &frame.t, &frame.layer->attention,
			frame.layer->kv_index, &state->attention, &selection, state->index.partial, first, count);
	}

	static SparkStatus Attend(State *state, Frame &frame)
	{
		SparkStatus status = SPARK_STATUS_OK;
		for ( uint32_t first = 0u, row = 1u; status == SPARK_STATUS_OK && row <= frame.t.rows; row++ )
		{
			const bool sparse = Sparse(frame.t.host_positions[first]);
			if ( row < frame.t.rows && Sparse(frame.t.host_positions[row]) == sparse )
				continue;
			status = sparse ? AttendSelected(state, frame, first, row - first) :
				SparkDeviceLatentAttentionAttend(&state->stage, frame.queue, &state->attention_shape, &frame.t, &frame.layer->attention, frame.layer->kv_index,
					&state->attention, first, row - first);
			first = row;
		}
		return status;
	}

	static SparkStatus Attention(State *state, Frame &frame, uint32_t layer)
	{
		const SparkIndexerCache cache = IndexCache(state, frame);
		(void)layer;
		SparkStatus status = SparkDeviceLatentAttentionProject(&state->stage, frame.queue, &state->attention_shape, &state->rope, &frame.t,
			&frame.layer->attention, frame.layer->kv_index, &state->attention, state->normed);
		if ( status == SPARK_STATUS_OK )
			status = IndexKeys(state, frame, cache);
		if ( status == SPARK_STATUS_OK && frame.sparse_rows != 0u )
			status = IndexSelect(state, frame, cache);
		if ( status == SPARK_STATUS_OK )
			status = frame.sparse_rows == 0u ? SparkDeviceLatentAttentionAttend(&state->stage, frame.queue, &state->attention_shape, &frame.t,
				&frame.layer->attention, frame.layer->kv_index, &state->attention, 0u, frame.t.rows) : Attend(state, frame);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceLatentAttentionOutput(&state->stage, frame.queue, &state->attention_shape, &frame.t, &frame.layer->attention, &state->attention,
				state->partial);
		return status;
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
		SparkRouteParams route;
		(void)layer;
		SparkStatus status = SparkDeviceStageRouteSigmoid(&state->stage, frame.queue, state->normed, weights.router, weights.router_bias, state->router_logits,
			&state->experts, rows, kHidden, kExperts, kExperts, kTopK, kTp, SPARK_GLM5_NEXT_MODEL_MOE_ROUTED_SCALING_FACTOR, 1u, 0u, 0u, 0u, kHidden, &route);
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
		const SparkNormParams norm = SparkDeviceStageNormParams(frame.t.rows, kHidden, kEpsilon);
		SparkStatus status = SparkOpsStreamMean(state->stage.ops, frame.queue, state->streams, state->mixed, &params);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsRmsNormBf16(state->stage.ops, frame.queue, state->mixed, state->final_norm, frame.t.hidden, &norm);
		return status;
	}

	static SparkStatus Step(void *model, const SparkStageRunnerStep *step, void *queue)
	{
		return Glm5NextSchedule<G, Glm5NextDevice>::Step((State *)model, step, queue);
	}
};

template<uint32_t TP>
struct Glm53FlashDevice
{
	using Geometry = Glm53FlashGeometry<TP, SPARK_WEIGHT_CODEC_FP8_E4M3>;
	static constexpr const char *kTag = "sparkpipe_glm53_flash_device";
	static constexpr const char *kName = "glm-5.3-flash";
	static constexpr uint32_t kProjection = SPARK_PROJECTION_GEMV;
};

const SparkStageRunnerModelInterface *SparkGlm53FlashDeviceStageModelAt(uint32_t tp_degree)
{
	return SparkDeviceStageAt<Glm5NextDevice, Glm53FlashDevice>(tp_degree);
}
