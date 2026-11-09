#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inference/runner/device_layers.h"
#include "sparkpipe/spark_k3_device_stage.h"
#include "sparkpipe/spark_k3_family.h"

#define KD_SLICES 48u
#define KD_ALIGNMENT 128u

template<uint32_t TP>
struct K3Device
{
	using L = K3Layout;

	static constexpr int32_t kOk = SPARK_STATUS_OK;
	static constexpr int32_t kShapeError = SPARK_STATUS_INVALID_ARGUMENT;
	static constexpr bool kShardedMla = false;
	static constexpr bool kExpertCells = false;
	static constexpr bool kLazyExperts = false;

	static constexpr uint32_t kHidden = K3_HIDDEN;
	static constexpr uint32_t kKdaHeads = K3_KDA_HEADS / TP;
	static constexpr uint32_t kKdaWidth = kKdaHeads * K3_KDA_KEY_DIM;
	static constexpr uint32_t kMlaHeads = K3_MLA_HEADS / TP;
	static constexpr uint32_t kQkDim = K3_QK_NOPE_DIM + K3_QK_UNROTATED_DIM;
	static constexpr uint32_t kSlot = K3_KV_LORA_RANK + K3_QK_UNROTATED_DIM;
	static constexpr uint32_t kLatent = K3_ROUTED_EXPERT_HIDDEN;
	static constexpr uint32_t kLatentSlice = kLatent / TP;
	static constexpr uint32_t kMoeRows = K3_EXPERT_INTERMEDIATE / TP;
	static constexpr uint32_t kSharedRows = K3_SHARED_EXPERTS * K3_EXPERT_INTERMEDIATE / TP;
	static constexpr uint32_t kDenseRows = K3_DENSE_INTERMEDIATE / TP;
	static constexpr uint32_t kDecayRank = 128u;
	static constexpr uint32_t kBanks = K3_LAYERS / K3_ATTNRES_BLOCK_SIZE + 1u;

	struct Layer
	{
		SparkTensor attn_norm;
		SparkTensor mlp_norm;
		SparkTensor attnres_attn;
		SparkTensor attnres_mlp;
		SparkDeviceChannelDeltaWeights delta;
		SparkDeviceLatentAttentionWeights attention;
		SparkDeviceFfnWeights ffn;
		SparkDeviceFfnWeights shared;
		SparkTensor router;
		SparkTensor router_bias;
		SparkDeviceWeight routed_down;
		SparkTensor routed_norm;
		SparkDeviceWeight routed_up;
		SparkDeviceExpertMatrix gate;
		SparkDeviceExpertMatrix up;
		SparkDeviceExpertMatrix down;
		uint32_t index;
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
		Layer layers[K3_LAYERS];
		SparkTensor attnres_out;
		SparkTensor bank;
		SparkTensor partial;
		SparkTensor scores;
		SparkTensor normed;
		SparkTensor branch;
		SparkTensor shared_out;
		SparkTensor router_logits;
		SparkTensor latent;
		SparkTensor latent_local;
		SparkTensor latent_lanes;
		SparkTensor latent_out;
		SparkTensor latent_normed;
		SparkDeviceChannelDeltaScratch delta;
		SparkDeviceLatentAttentionScratch attention;
		SparkDeviceFfnScratch ffn_scratch;
		SparkDeviceExpertsScratch experts;
	};

	struct Context
	{
		State *state;
		SparkDeviceStepTensors t;
		SparkDeviceQueue *queue;
		const Layer *layer;
		int32_t status;
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
		shape->tag = "sparkpipe_k3_device";
		shape->hidden = kHidden;
		shape->vocab = K3_VOCAB;
		shape->layers = K3_LAYERS;
		shape->kv_layers = K3_MLA_LAYER_COUNT;
		shape->kv_heads = 1u;
		shape->kv_head_dimension = (kSlot + K3_KV_LORA_RANK) / 2u;
		shape->pack_alignment = KD_ALIGNMENT;
		shape->epsilon = K3_RMS_EPSILON;
		shape->projection = SPARK_PROJECTION_GEMV;
		memset(delta, 0, sizeof(*delta));
		delta->heads = kKdaHeads;
		delta->dimension = K3_KDA_KEY_DIM;
		delta->conv_taps = K3_KDA_CONV_KERNEL;
		delta->low_rank = kDecayRank;
		delta->epsilon = K3_RMS_EPSILON;
		delta->l2_epsilon = 1e-6f;
		delta->minimum_log_decay = K3_KDA_GATE_LOWER_BOUND;
		delta->beta_mode = SPARK_DELTA_BETA_SIGMOID;
		delta->norm_f32 = 1u;
		delta->norm_round = 1u;
		delta->conv_f32 = 1u;
		delta->gate_full_rank = K3_KDA_FULL_RANK_GATE;
		delta->prenorm_epsilon = K3_RMS_EPSILON;
		memset(attention, 0, sizeof(*attention));
		attention->heads = K3_MLA_HEADS;
		attention->nope = K3_QK_NOPE_DIM;
		attention->rope = K3_QK_UNROTATED_DIM;
		attention->latent = K3_KV_LORA_RANK;
		attention->value_dimension = K3_V_HEAD_DIM;
		attention->rope_theta = 1.0;
		attention->epsilon = K3_LORA_RMS_EPSILON;
		attention->scale = K3_MLA_QK_SCALE;
		attention->head_gate = SPARK_LATENT_GATE_ELEMENT;
		attention->query_lora = K3_Q_LORA_RANK;
		attention->rope_unrotated = 1u;
		attention->gate_unrounded = 1u;
	}

	static uint32_t ScratchSlices(State *state, SparkDeviceStageSlice *slices)
	{
		const uint64_t rows = state->stage.max_rows, hidden = kHidden, routes = rows * K3_TOP_K, heads = kMlaHeads, width = kKdaWidth;
		const uint64_t ffn = Max(kDenseRows, kSharedRows);
		const SparkDeviceStageSlice plan[] =
		{
			{ (uint64_t)kBanks * rows * hidden * 2u, &state->bank },
			{ rows * hidden * 2u, &state->partial },
			{ rows * 16u * sizeof(float), &state->scores },
			{ rows * hidden * 2u, &state->normed },
			{ rows * hidden * 2u, &state->branch },
			{ rows * hidden * 2u, &state->shared_out },
			{ rows * K3_EXPERTS * sizeof(float), &state->router_logits },
			{ rows * kLatent * 2u, &state->latent },
			{ rows * kLatentSlice * 2u, &state->latent_local },
			{ (uint64_t)TP * rows * kLatentSlice * 2u, &state->latent_lanes },
			{ rows * kLatent * 2u, &state->latent_out },
			{ rows * kLatent * 2u, &state->latent_normed },
			{ rows * 3u * width * 2u, &state->delta.qkv },
			{ rows * kKdaHeads * 2u, &state->delta.beta },
			{ rows * kDecayRank * 2u, &state->delta.low },
			{ rows * width * 2u, &state->delta.decay },
			{ rows * width * 2u, &state->delta.gate },
			{ SparkOpsChannelDeltaScratchBytes(state->stage.max_rows, kKdaHeads, K3_KDA_KEY_DIM), &state->delta.delta },
			{ rows * width * 2u, &state->delta.core },
			{ rows * width * 2u, &state->delta.gated },
			{ rows * K3_Q_LORA_RANK * 2u, &state->attention.query_lora },
			{ rows * heads * kQkDim * 2u, &state->attention.query },
			{ rows * kSlot * 2u, &state->attention.kv },
			{ rows * heads * K3_V_HEAD_DIM * 2u, &state->attention.gate },
			{ rows * heads * kSlot * 2u, &state->attention.packed },
			{ SparkOpsAttentionScratchBytes(state->stage.max_rows, kMlaHeads, K3_KV_LORA_RANK, state->stage.max_positions), &state->attention.records },
			{ rows * heads * K3_KV_LORA_RANK * 2u, &state->attention.latent },
			{ rows * heads * K3_V_HEAD_DIM * 2u, &state->attention.value },
			{ rows * 2u * ffn * 2u, &state->ffn_scratch.gate_up },
			{ rows * ffn * 2u, &state->ffn_scratch.act },
			{ routes * sizeof(uint32_t), &state->experts.experts },
			{ routes * sizeof(float), &state->experts.weights },
			{ routes * 2u * kMoeRows * 2u, &state->experts.gate_up },
			{ routes * kMoeRows * 2u, &state->experts.act },
			{ routes * kLatent * 2u, &state->experts.out },
			{ (routes + 1u) * kLatent * 2u, &state->experts.sorted },
			{ Max(2u * ffn, Max(3u * width, hidden)) * hidden * 2u, &state->stage.dequant }
		};
		const uint32_t count = sizeof(plan) / sizeof(plan[0]);
		for ( uint32_t index = 0u; index < count; index++ )
			slices[index] = plan[index];
		return count;
	}

	static SparkStatus Open(const SparkStageRunnerModelOpen *request, void **out, SparkStageRunnerModelGeometry *geometry)
	{
		SparkDeviceStageSlice slices[KD_SLICES];
		State *state = (State *)calloc(1u, sizeof(State));
		SparkStatus status;
		*out = state;
		if ( state == 0 )
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		Shapes(state);
		const SparkDeviceStageRecurrentPlan pools[] = { { K3_KDA_LAYER_COUNT, SparkDeviceChannelDeltaStateBytes(&state->delta_shape) },
			{ K3_KDA_LAYER_COUNT, SparkDeviceChannelDeltaWindowBytes(&state->delta_shape) } };
		const SparkDeviceStageConfigValue config[] =
		{
			{ "hidden", kHidden }, { "layers", K3_LAYERS }, { "vocab", K3_VOCAB }, { "vocab_rows", K3_VOCAB / TP }, { "experts", K3_EXPERTS },
			{ "local_experts", K3_EXPERTS }, { "top_k", K3_TOP_K }, { "ffn_rows", kDenseRows }, { "shared_rows", kSharedRows }, { "moe_rows", kMoeRows },
			{ "latent_rows", kLatentSlice }
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

	static SparkStatus BindExpert(State *state, const SparkStageRunnerDeviceWeights *weights, uint32_t index, const char *name, uint64_t rows,
		uint64_t columns, SparkDeviceExpertMatrix *matrix)
	{
		char scale_name[48];
		SparkStatus status = SparkDeviceStageLayerTensor(&state->stage, weights, index, name, "mxfp4_e2m1", K3_EXPERTS * rows * columns / 2u, &matrix->values);
		matrix->codec = SPARK_WEIGHT_CODEC_MXFP4_E2M1;
		snprintf(scale_name, sizeof(scale_name), "%s_scale", name);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, scale_name, "e8m0", K3_EXPERTS * rows * (columns / K3_MXFP4_GROUP),
				&matrix->scale);
		if ( status == SPARK_STATUS_OK )
			SparkDeviceStageNoteRouted(&state->stage, matrix->values.bytes + matrix->scale.bytes, K3_EXPERTS, K3_TOP_K);
		return status;
	}

	static SparkStatus BindLayer(State *state, const SparkStageRunnerDeviceWeights *weights, uint32_t index, uint32_t *recurrent, uint32_t *kv)
	{
		Layer *layer = &state->layers[index];
		SparkStatus status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "attn_norm", "f32", kHidden * sizeof(float), &layer->attn_norm);
		layer->index = index;
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "mlp_norm", "f32", kHidden * sizeof(float), &layer->mlp_norm);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "attnres_attn", "bf16", kHidden * 2u, &layer->attnres_attn);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "attnres_mlp", "bf16", kHidden * 2u, &layer->attnres_mlp);
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
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "router", "bf16", (uint64_t)K3_EXPERTS * kHidden * 2u, &layer->router);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "router_bias", "f32", (uint64_t)K3_EXPERTS * sizeof(float), &layer->router_bias);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerWeight(&state->stage, weights, index, "routed_down", kLatentSlice, kHidden, &layer->routed_down);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerTensor(&state->stage, weights, index, "routed_norm", "f32", kLatent * sizeof(float), &layer->routed_norm);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceStageLayerWeight(&state->stage, weights, index, "routed_up", kHidden, kLatentSlice, &layer->routed_up);
		if ( status == SPARK_STATUS_OK )
			status = BindExpert(state, weights, index, "moe_w1", kMoeRows, kLatent, &layer->gate);
		if ( status == SPARK_STATUS_OK )
			status = BindExpert(state, weights, index, "moe_w3", kMoeRows, kLatent, &layer->up);
		if ( status == SPARK_STATUS_OK )
			status = BindExpert(state, weights, index, "moe_w2", kLatent, kMoeRows, &layer->down);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceFfnBindNamed(&state->stage, weights, index, "shared_gate_up", "shared_down", kSharedRows, &layer->shared);
		return status;
	}

	static SparkStatus BindDevice(void *model, const SparkStageRunnerDeviceWeights *weights, SparkStageRunnerModelGeometry *geometry)
	{
		State *state = (State *)model;
		SparkDeviceStageSlice slices[KD_SLICES];
		uint32_t recurrent = 0u, kv = 0u;
		SparkStatus status;
		(void)geometry;
		if ( weights == 0 || weights->pack_address == 0u || weights->pack_bytes != state->stage.pack_bytes )
			return SparkDeviceStageRefuse(&state->stage, SPARK_STATUS_INVALID_ARGUMENT, "bind_device needs the device mapping of the whole rank pack");
		status = SparkDeviceStageTensor(&state->stage, weights, "attnres_out", "bf16", kHidden * 2u, &state->attnres_out);
		for ( uint32_t index = 0u; index < K3_LAYERS && status == SPARK_STATUS_OK; index++ )
			status = BindLayer(state, weights, index, &recurrent, &kv);
		if ( status != SPARK_STATUS_OK )
			return status;
		return SparkDeviceStageAllocate(&state->stage, slices, ScratchSlices(state, slices));
	}

	static int32_t Note(Context &c, SparkStatus status)
	{
		if ( c.status == kOk && status != SPARK_STATUS_OK )
			c.status = (int32_t)status;
		return (int32_t)status;
	}

	static SparkTensor Block(SparkTensor tensor, uint32_t rows, uint32_t width, uint32_t first)
	{
		return SparkTensorSlice(tensor, (uint64_t)first * rows * width * sizeof(uint16_t), (uint64_t)rows * width * sizeof(uint16_t));
	}

	static SparkStatus Copy(Context &c, SparkTensor from, SparkTensor to)
	{
		const uint64_t bytes = (uint64_t)c.t.rows * kHidden * sizeof(uint16_t);
		return SparkDeviceCopy(c.queue, SparkTensorSlice(to, 0u, bytes).range, SparkTensorSlice(from, 0u, bytes).range, bytes);
	}

	static SparkStatus Norm(Context &c, SparkTensor weight, uint32_t width, float epsilon, SparkTensor input, SparkTensor output)
	{
		SparkNormParams norm;
		memset(&norm, 0, sizeof(norm));
		norm.rows = c.t.rows;
		norm.dimension = norm.input_stride = norm.output_stride = width;
		norm.epsilon = epsilon;
		return SparkOpsRmsNormResidual(c.state->stage.ops, c.queue, input, 0, weight, output, &norm);
	}

	static uint32_t Rows(const Context &c, uint32_t lane) { (void)lane; return c.t.rows; }
	static uint32_t Sequences(const Context &c) { return c.t.sequences.sequences; }
	static uint32_t PackedRows(const Context &c) { return c.t.rows * K3_TOP_K; }
	static bool HasCollective(const Context &c) { (void)c; return true; }
	static bool HasLazy(const Context &c) { (void)c; return false; }
	static uint32_t TpSharded(const Context &c, uint32_t lane) { (void)c; (void)lane; return TP > 1u ? 1u : 0u; }
	static bool KvSharded(const Context &c, uint32_t lane) { (void)c; (void)lane; return false; }
	static bool ExpertCells(const Context &c, uint32_t lane) { (void)c; (void)lane; return false; }

	static int32_t Failure(uint32_t layer, const char *phase, int32_t status)
	{
		fprintf(stderr, "k3 device stage: %s failed at layer %u (status %d)\n", phase, layer, (int)status);
		return status;
	}

	static void Bind(Context &c, uint32_t lane, uint32_t offset, uint32_t layer)
	{
		(void)lane;
		(void)offset;
		c.layer = &c.state->layers[layer];
	}

	static void AttnRes(Context &c, uint32_t lane, uint32_t site, uint32_t sources, uint32_t stream)
	{
		State *state = c.state;
		const SparkTensor weight = site == K3_ATTNRES_ATTENTION ? c.layer->attnres_attn : site == K3_ATTNRES_MLP ? c.layer->attnres_mlp : state->attnres_out;
		SparkAttnResParams params;
		(void)lane;
		(void)stream;
		memset(&params, 0, sizeof(params));
		params.rows = c.t.rows;
		params.sources = sources > K3_ATTNRES_MAX_SOURCES ? K3_ATTNRES_MAX_SOURCES : sources;
		params.dimension = kHidden;
		params.score_stride = 16u;
		params.epsilon = K3_RMS_EPSILON;
		if ( c.status == kOk )
			Note(c, SparkOpsAttnResScore(state->stage.ops, c.queue, state->bank, state->partial, weight, state->scores, &params));
		if ( c.status == kOk )
			Note(c, SparkOpsAttnResMix(state->stage.ops, c.queue, state->bank, state->partial, state->scores, c.t.hidden, &params));
	}

	static void PartialSet(Context &c, uint32_t lane, uint32_t stream)
	{
		(void)lane;
		(void)stream;
		if ( c.status == kOk )
			Note(c, Copy(c, c.t.hidden, c.state->partial));
	}

	static void PartialSetUnsharded(Context &c, uint32_t lane, uint32_t stream)
	{
		(void)c;
		(void)lane;
		(void)stream;
	}

	static void BankStore(Context &c, uint32_t lane, uint32_t bank, uint32_t stream)
	{
		(void)lane;
		(void)stream;
		if ( c.status == kOk )
			Note(c, Copy(c, c.state->partial, Block(c.state->bank, c.t.rows, kHidden, bank)));
	}

	static int32_t Kda(Context &c, uint32_t lane, uint32_t sequences, uint32_t accumulate, uint32_t stream)
	{
		State *state = c.state;
		(void)lane;
		(void)sequences;
		(void)accumulate;
		(void)stream;
		SparkStatus status = SparkDeviceResidualNorm(&state->stage, c.queue, &c.t, 0, c.layer->attn_norm, state->normed);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceChannelDelta(&state->stage, c.queue, &state->delta_shape, &c.t, &c.layer->delta, c.layer->recurrent_index, &state->delta,
				state->normed, state->branch);
		return Note(c, status);
	}

	static int32_t MlaLocal(Context &c, uint32_t lane, uint32_t accumulate, uint32_t stream)
	{
		State *state = c.state;
		(void)lane;
		(void)accumulate;
		(void)stream;
		SparkStatus status = SparkDeviceResidualNorm(&state->stage, c.queue, &c.t, 0, c.layer->attn_norm, state->normed);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceLatentAttention(&state->stage, c.queue, &state->attention_shape, &state->rope, &c.t, &c.layer->attention, c.layer->kv_index,
				&state->attention, state->normed, state->branch);
		return Note(c, status);
	}

	static int32_t MlpPrefetchRound(Context &c, uint32_t layer)
	{
		State *state = c.state;
		SparkStatus status = SparkDeviceStageAllReduce(&state->stage, c.queue, state->branch, c.t.rows);
		if ( status == SPARK_STATUS_OK )
			status = L::Boundary(layer) ? Copy(c, state->branch, state->partial) :
				SparkOpsAddRows(state->stage.ops, c.queue, state->partial, state->branch, c.t.rows, kHidden, kHidden, kHidden);
		return Note(c, status);
	}

	static int32_t Dense(Context &c, uint32_t lane, uint32_t stream)
	{
		State *state = c.state;
		(void)lane;
		(void)stream;
		SparkStatus status = SparkDeviceResidualNorm(&state->stage, c.queue, &c.t, 0, c.layer->mlp_norm, state->normed);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceGatedFfnLimits(&state->stage, c.queue, kDenseRows, SPARK_GATED_SITU, K3_SITU_BETA, K3_SITU_LINEAR_BETA, &c.t, &c.layer->ffn,
				&state->ffn_scratch, state->normed, state->branch);
		return Note(c, status);
	}

	static SparkStatus RouteAndDown(Context &c)
	{
		State *state = c.state;
		const Layer &layer = *c.layer;
		SparkGemvParams router;
		SparkRouteParams route;
		SparkStatus status = SparkDeviceResidualNorm(&state->stage, c.queue, &c.t, 0, layer.mlp_norm, state->normed);
		memset(&router, 0, sizeof(router));
		router.rows = c.t.rows;
		router.input_dimension = router.input_stride = kHidden;
		router.output_dimension = router.output_stride = K3_EXPERTS;
		router.output_f32 = 1u;
		router.projection = state->stage.shape->projection;
		router.row_exact = state->stage.step_exact;
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsLinear(state->stage.ops, c.queue, state->normed, layer.router, state->router_logits, &router);
		memset(&route, 0, sizeof(route));
		route.rows = c.t.rows;
		route.experts = K3_EXPERTS;
		route.top_k = K3_TOP_K;
		route.width = kLatent;
		route.logit_stride = K3_EXPERTS;
		route.scale = K3_ROUTED_SCALE;
		route.tree_sum = 1u;
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsRouteSigmoid(state->stage.ops, c.queue, state->router_logits, layer.router_bias, state->experts.experts, state->experts.weights,
				&route);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceLinearWeight(&state->stage, c.queue, state->normed, kHidden, &layer.routed_down, TP > 1u ? state->latent_local : state->latent,
				kLatentSlice, c.t.rows, kHidden, kLatentSlice);
		return status;
	}

	static SparkStatus ExpertsAndShared(Context &c)
	{
		State *state = c.state;
		const Layer &layer = *c.layer;
		SparkDeviceRoutedFfnShape ffn;
		SparkRouteParams route;
		memset(&ffn, 0, sizeof(ffn));
		ffn.experts = K3_EXPERTS;
		ffn.top_k = K3_TOP_K;
		ffn.intermediate = kMoeRows;
		ffn.activation = SPARK_GATED_SITU;
		ffn.limit = K3_SITU_BETA;
		ffn.up_limit = K3_SITU_LINEAR_BETA;
		ffn.width = kLatent;
		memset(&route, 0, sizeof(route));
		route.rows = c.t.rows;
		route.top_k = K3_TOP_K;
		route.width = kLatent;
		SparkStatus status = SparkDeviceRoutedFfn(&state->stage, c.queue, &ffn, c.t.rows, &layer.gate, &layer.up, &layer.down, &state->experts, state->latent);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsRouteCombineWeighted(state->stage.ops, c.queue, state->experts.out, state->experts.weights, state->latent_out, &route);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceGatedFfnLimits(&state->stage, c.queue, kSharedRows, SPARK_GATED_SITU, K3_SITU_BETA, K3_SITU_LINEAR_BETA, &c.t, &layer.shared,
				&state->ffn_scratch, state->normed, state->shared_out);
		return status;
	}

	static SparkStatus LatentUp(Context &c)
	{
		State *state = c.state;
		const Layer &layer = *c.layer;
		const uint32_t first = state->stage.tp_rank * kLatentSlice;
		SparkStatus status = Norm(c, layer.routed_norm, kLatent, K3_RMS_EPSILON, state->latent_out, state->latent_normed);
		if ( status == SPARK_STATUS_OK )
			status = SparkDeviceLinearWeight(&state->stage, c.queue, SparkTensorSlice(state->latent_normed, (uint64_t)first * sizeof(uint16_t),
				state->latent_normed.bytes - (uint64_t)first * sizeof(uint16_t)), kLatent, &layer.routed_up, state->branch, kHidden, c.t.rows, kLatentSlice,
				kHidden);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsAddRows(state->stage.ops, c.queue, state->branch, state->shared_out, c.t.rows, kHidden, kHidden, kHidden);
		return status;
	}

	static int32_t LatentMoe(Context &c, uint32_t lane, uint32_t rows, uint32_t packed_rows, uint32_t phase, uint32_t stream)
	{
		SparkStatus status;
		(void)lane;
		(void)rows;
		(void)packed_rows;
		(void)stream;
		if ( phase == 0u )
			status = RouteAndDown(c);
		else if ( phase == 1u )
		{
			status = ExpertsAndShared(c);
			if ( status == SPARK_STATUS_OK && TP == 1u )
				status = LatentUp(c);
		}
		else
			status = LatentUp(c);
		return Note(c, status);
	}

	static void Collective(Context &c, uint32_t lane, uint32_t stream, uint32_t layer, uint32_t phase)
	{
		State *state = c.state;
		(void)lane;
		(void)stream;
		(void)layer;
		if ( TP == 1u || c.status != kOk )
			return;
		if ( phase == 2u )
			Note(c, SparkDeviceStageGatherRows(&state->stage, c.queue, state->latent_local, state->latent_lanes, state->latent, c.t.rows, kLatentSlice));
		else if ( phase == 3u )
			Note(c, SparkDeviceStageAllReduceWidth(&state->stage, c.queue, state->latent_out, c.t.rows, kLatent));
	}

	static int32_t AttentionPrefetchRound(Context &c, uint32_t offset, uint32_t layer, uint32_t layer_count)
	{
		State *state = c.state;
		(void)offset;
		(void)layer;
		(void)layer_count;
		SparkStatus status = SparkDeviceStageAllReduce(&state->stage, c.queue, state->branch, c.t.rows);
		if ( status == SPARK_STATUS_OK )
			status = SparkOpsAddRows(state->stage.ops, c.queue, state->partial, state->branch, c.t.rows, kHidden, kHidden, kHidden);
		return Note(c, status);
	}

	static void Release(Context &c, uint32_t layer)
	{
		SparkDeviceStageLayerDone(&c.state->stage, layer);
		if ( c.status == kOk )
			Note(c, SparkDeviceStageDumpRows(&c.state->stage, c.queue, &c.t, c.state->partial, kHidden, layer));
	}

	static void Tap(Context &c, uint32_t lane, uint32_t layer, uint32_t row_offset, uint32_t stream)
	{
		(void)c;
		(void)lane;
		(void)layer;
		(void)row_offset;
		(void)stream;
	}

	static void CopyRows(Context &c, uint32_t lane, uint32_t stream)
	{
		(void)lane;
		(void)stream;
		if ( c.status == kOk )
			Note(c, Copy(c, c.state->partial, c.t.hidden));
	}

	static SparkStatus Step(void *model, const SparkStageRunnerStep *step, void *queue)
	{
		State *state = (State *)model;
		Context c;
		memset(&c, 0, sizeof(c));
		c.state = state;
		c.queue = (SparkDeviceQueue *)queue;
		c.status = kOk;
		SparkStatus status = SparkDeviceStageStepTables(&state->stage, step, &c.t);
		if ( status != SPARK_STATUS_OK )
			return status;
		const int32_t result = K3Schedule<K3Device>::Slice(c, 0u, K3_LAYERS);
		if ( c.status != kOk )
			return (SparkStatus)c.status;
		return result == kOk ? SPARK_STATUS_OK : (SparkStatus)result;
	}

	static const SparkStageRunnerModelInterface *Interface(void)
	{
		static const SparkStageRunnerModelInterface interface_table =
		{
			SPARK_STAGE_RUNNER_MODEL_ABI_VERSION,
			"sparkpipe_k3_device",
			"kimi-k3",
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

const SparkStageRunnerModelInterface *SparkK3DeviceStageModelAt(uint32_t tp_degree)
{
	switch ( tp_degree )
	{
	case 1u:
		return K3Device<1u>::Interface();
	case 2u:
		return K3Device<2u>::Interface();
	case 4u:
		return K3Device<4u>::Interface();
	case 8u:
		return K3Device<8u>::Interface();
	default:
		return 0;
	}
}
