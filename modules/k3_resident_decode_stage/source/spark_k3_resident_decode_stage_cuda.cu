
#include <cstddef>
#include <cstdio>
#include <cstring>

#include "sparkpipe/spark_k3_resident_decode_stage_cuda.h"
#include "sparkpipe/spark_k3_kv_shard.h"
#include "sparkpipe/spark_k3_weightd_include.h"

#include "sparkpipe/spark_tp_mesh_kernels.cuh"

extern "C" int32_t K3StageSlice(const void *layer_weights, const void *slice_state,
	void *layer_buffers, uint32_t first_layer, uint32_t layer_count, uint32_t rows,
	uint32_t sequences, uint32_t commit, uint32_t packed_rows, uint32_t context,
	uint32_t multiprocessors, void *stream);
#define K3_DISPATCH_PAIR_IDENTITY 16u

extern "C" int32_t K3StageSlicePair(const void *layer_weights, const void *slice_state,
	void *buffers_a, void *buffers_b, uint32_t first_layer, uint32_t layer_count, uint32_t rows_a,
	uint32_t rows_b, uint32_t commit, uint32_t multiprocessors, void *stream);

#define WF(field, name) { offsetof(K3LayerWeights, field), name }
static const struct SparkK3WeightBind
{
	size_t offset;
	const char *name;
} k3_weight_binds[] =
{
	WF(attn_norm_weight, "attn_norm_weight"),
	WF(mlp_norm_weight, "mlp_norm_weight"),
	WF(kda_qkv_beta_weight, "kda_qkv_beta_weight"),
	WF(kda_decay_down_weight, "kda_decay_down_weight"),
	WF(kda_q_conv_weight, "kda_q_conv_weight"),
	WF(kda_k_conv_weight, "kda_k_conv_weight"),
	WF(kda_v_conv_weight, "kda_v_conv_weight"),
	WF(kda_decay_up_weight, "kda_decay_up_weight"),
	WF(kda_decay_bias, "kda_decay_bias"),
	WF(kda_head_log_scale, "kda_head_log_scale"),
	WF(kda_gate_weight, "kda_gate_weight"),
	WF(kda_out_norm_weight, "kda_out_norm_weight"),
	WF(kda_out_weight, "kda_out_weight"),
	WF(mla_q_down_weight, "mla_q_down_weight"),
	WF(mla_q_norm_weight, "mla_q_norm_weight"),
	WF(mla_q_up_weight, "mla_q_up_weight"),
	WF(mla_kv_a_weight, "mla_kv_a_weight"),
	WF(mla_kv_a_norm_weight, "mla_kv_a_norm_weight"),
	WF(mla_kv_b_value_weight, "mla_kv_b_value_weight"),
	WF(mla_gate_weight, "mla_gate_weight"),
	WF(mla_out_weight, "mla_out_weight"),
	WF(router_weight, "router_weight"),
	WF(router_bias, "router_bias"),
	WF(routed_down_weight, "routed_down_weight"),
	WF(routed_up_weight, "routed_up_weight"),
	WF(routed_norm_weight, "routed_norm_weight"),
	WF(expert_w1_weight, "expert_w1_weight"),
	WF(expert_w2_weight, "expert_w2_weight"),
	WF(shared_w1_weight, "shared_w1_weight"),
	WF(shared_w2_weight, "shared_w2_weight"),
	WF(dense_gate_up_weight, "dense_gate_up_weight"),
	WF(dense_down_weight, "dense_down_weight"),
	WF(attnres_attn_weight, "attnres_attn_weight"),
	WF(attnres_mlp_weight, "attnres_mlp_weight"),
};
#undef WF

static const char *const k3_required_every[] =
{
	"attn_norm_weight", "mlp_norm_weight",
	"attnres_attn_weight", "attnres_mlp_weight",
};
static const char *const k3_required_kda[] =
{
	"kda_qkv_beta_weight", "kda_decay_down_weight", "kda_q_conv_weight",
	"kda_k_conv_weight", "kda_v_conv_weight", "kda_decay_up_weight",
	"kda_decay_bias", "kda_head_log_scale", "kda_gate_weight",
	"kda_out_norm_weight", "kda_out_weight",
};
static const char *const k3_required_mla[] =
{
	"mla_q_down_weight", "mla_q_norm_weight", "mla_q_up_weight",
	"mla_kv_a_weight", "mla_kv_a_norm_weight", "mla_kv_b_value_weight",
	"mla_gate_weight", "mla_out_weight",
};
static const char *const k3_required_moe[] =
{
	"router_weight", "router_bias", "routed_down_weight", "routed_up_weight",
	"routed_norm_weight", "expert_w1_weight", "expert_w2_weight",
	"shared_w1_weight", "shared_w2_weight",
};
static const char *const k3_required_dense[] =
{
	"dense_gate_up_weight", "dense_down_weight",
};

static int32_t k3_require(const SparkK3BoundLayer *bound,
	const char *const *names, uint32_t count)
{
	for ( uint32_t i = 0u; i < count; ++i )
		if ( SparkK3BoundEntry(bound, names[i]) == 0 )
			return SPARK_K3_DISPATCH_ERR_BIND;
	return SPARK_K3_DISPATCH_OK;
}

static int32_t k3_require_whole_head_table(const SparkK3BoundLayer *bound,
	const char *name, uint64_t bytes)
{
	const SparkK3PackEntry *entry = SparkK3BoundEntry(bound, name);
	if ( entry != 0 && entry->bytes == bytes )
		return SPARK_K3_DISPATCH_OK;
	fprintf(stderr, "sparkpipe_k3: %s holds %llu bytes, the per-head table "
		"must carry all %u heads (%llu bytes)\n", name,
		(unsigned long long)(entry != 0 ? entry->bytes : 0u), K3_KDA_HEADS,
		(unsigned long long)bytes);
	return SPARK_K3_DISPATCH_ERR_BIND;
}

static int32_t k3_require_f32_table(const SparkK3BoundLayer *bound,
	const char *name, uint32_t count)
{
	const SparkK3PackEntry *entry = SparkK3BoundEntry(bound, name);
	if ( entry != 0 && entry->kind == SPARK_K3_PACK_KIND_F32 &&
		entry->bytes == (uint64_t)count * sizeof(float) )
		return SPARK_K3_DISPATCH_OK;
	fprintf(stderr, "sparkpipe_k3: %s must be %u F32 values per layer\n",
		name, count);
	return SPARK_K3_DISPATCH_ERR_BIND;
}

static uint8_t *k3_carve(SparkK3Dispatch *d, size_t *offset, size_t bytes)
{
	*offset = (*offset + 15u) & ~(size_t)15u;
	uint8_t *p = d->scratch + *offset;
	*offset += bytes;
	return p;
}

static uint32_t K3Max(uint32_t a, uint32_t b)
{
	return a > b ? a : b;
}

static void K3ScratchWidthsFor(const K3LayerBuffers *b, SparkK3ScratchWidths *w)
{
	const uint32_t kda_heads = K3_RANK_DIM(b, kda_heads_rank, K3_KDA_HEADS);
	const uint32_t mla_heads = K3_RANK_DIM(b, mla_heads_rank, K3_MLA_HEADS);
	const uint32_t w1 = K3_RANK_DIM(b, expert_w1_output, K3_EXPERT_INTERMEDIATE * 2u);
	const uint32_t slice = K3_RANK_DIM(b, routed_down_rows, K3_ROUTED_EXPERT_HIDDEN);
	const uint32_t ranks = K3_ROUTED_EXPERT_HIDDEN / slice;
	const uint32_t router = 2u * (K3_EXPERTS / ranks);
	const uint32_t cells = K3_EXPERT_CELLS(b) ? 1u : 0u;
	w->qkvb = K3_RANK_DIM(b, kda_qkvb_rows, K3_KDA_QKVB_FUSED_ROWS);
	w->query = K3Max(kda_heads * K3_KDA_KEY_DIM, K3_RANK_DIM(b, mla_q_up_rows, K3_MLA_Q_DIM));
	w->key = kda_heads * K3_KDA_KEY_DIM;
	w->value = K3Max(K3Max(kda_heads * K3_KDA_VALUE_DIM, mla_heads * K3_V_HEAD_DIM),
		K3_RANK_DIM(b, mla_gate_rows, K3_MLA_OUT_DIM));
	w->gate = K3Max(K3_RANK_DIM(b, kda_gate_rows, K3_KDA_V_DIM), K3_RANK_DIM(b, mla_gate_rows, K3_MLA_OUT_DIM));
	w->decay = K3_RANK_DIM(b, kda_decay_up_rows, K3_KDA_QK_DIM);
	w->latent = cells != 0u
		? K3Max(K3Max(K3_KDA_KEY_DIM, K3_Q_LORA_RANK + K3_MLA_KV_A_DIM), K3Max(slice + router, K3_ROUTED_EXPERT_HIDDEN))
		: K3_TOP_K * K3_ROUTED_EXPERT_HIDDEN;
	w->attention_out = K3Max(K3Max(K3_HIDDEN, mla_heads * K3_KV_LORA_RANK), kda_heads * K3_KDA_VALUE_DIM);
	w->gate_up = K3Max(K3_TOP_K * K3Max(w1, K3_ROUTED_EXPERT_HIDDEN),
		K3Max(K3_RANK_DIM(b, shared_w1_rows, K3_SHARED_INTERMEDIATE * 2u), K3_RANK_DIM(b, dense_gate_up_rows, K3_DENSE_INTERMEDIATE * 2u)));
	w->intermediate = K3Max(K3_TOP_K * (w1 / 2u),
		K3Max(K3_RANK_DIM(b, shared_w2_input, K3_SHARED_INTERMEDIATE), K3_RANK_DIM(b, dense_down_input, K3_DENSE_INTERMEDIATE)));
	w->retention = kda_heads * K3_KDA_KEY_DIM;
	w->shared_mid = K3_RANK_DIM(b, shared_w2_input, K3_SHARED_INTERMEDIATE);
	w->fused = cells != 0u
		? K3Max(K3_HIDDEN, K3Max((slice + router) * ranks, K3_Q_LORA_RANK + K3_MLA_KV_A_DIM))
		: K3_TOP_K * K3_EXPERT_INTERMEDIATE * 2u;
}

static uint32_t K3ScratchFits(const SparkK3ScratchWidths *needs, const SparkK3ScratchWidths *capacity)
{
	static const char *const names[] = {"qkvb", "query", "key", "value", "gate", "decay", "latent",
		"attention_out", "gate_up", "intermediate", "retention", "shared_mid", "fused"};
	const uint32_t *need = (const uint32_t *)needs, *have = (const uint32_t *)capacity;
	uint32_t index, fits = 1u;
	static_assert(sizeof(SparkK3ScratchWidths) == sizeof(names) / sizeof(names[0]) * sizeof(uint32_t),
		"every scratch width has a name");
	for ( index = 0u; index < sizeof(names) / sizeof(names[0]); ++index )
		if ( need[index] > have[index] )
		{
			fprintf(stderr, "sparkpipe_k3: bound weights need %u %s elements per row, scratch holds %u\n",
				need[index], names[index], have[index]);
			fits = 0u;
		}
	return fits;
}

static int32_t K3PackRankDims(SparkK3Pack *pack, uint32_t first_layer, uint32_t layer_count, K3LayerBuffers *b)
{
	int32_t status = SPARK_K3_DISPATCH_OK;
	char dim_name[96];
	SparkK3SliceKindLayers kinds;
	SparkK3PackEntry dim_entry;
	SparkK3SliceKindLayersFor(first_layer, layer_count, &kinds);
#define K3_FILL_RANK(layer, field_name, field_ptr, axis, rank) \
	do { \
		if ( status != SPARK_K3_DISPATCH_OK || (layer) == SPARK_K3_SLICE_NO_LAYER ) \
			break; \
		snprintf(dim_name, sizeof(dim_name), "model.layers.%u.%s", \
			(uint32_t)(layer), field_name); \
		if ( SparkK3PackLoadEntry(pack, dim_name, &dim_entry) != 0 || \
			dim_entry.shape_count < (rank) || dim_entry.shape[axis] == 0u ) \
		{ \
			fprintf(stderr, "sparkpipe_k3: rank dimension missing %s\n", dim_name); \
			status = SPARK_K3_DISPATCH_ERR_BIND; \
			break; \
		} \
		*(field_ptr) = (uint32_t)dim_entry.shape[axis]; \
	} while ( 0 )
	K3_FILL_RANK(kinds.kda, "kda_qkv_beta_weight", &b->kda_qkvb_rows, 0u, 2u);
	K3_FILL_RANK(kinds.kda, "kda_gate_weight", &b->kda_gate_rows, 0u, 2u);
	K3_FILL_RANK(kinds.kda, "kda_decay_up_weight", &b->kda_decay_up_rows, 0u, 2u);
	K3_FILL_RANK(kinds.kda, "kda_out_weight", &b->kda_out_input, 1u, 2u);
	K3_FILL_RANK(kinds.mla, "mla_q_up_weight", &b->mla_q_up_rows, 0u, 2u);
	K3_FILL_RANK(kinds.mla, "mla_gate_weight", &b->mla_gate_rows, 0u, 2u);
	K3_FILL_RANK(kinds.mla, "mla_out_weight", &b->mla_out_input, 1u, 2u);
	K3_FILL_RANK(kinds.routed, "routed_down_weight", &b->routed_down_rows, 0u, 2u);
	K3_FILL_RANK(kinds.routed, "routed_up_weight", &b->routed_up_input, 1u, 2u);
	K3_FILL_RANK(kinds.routed, "expert_w1_weight", &b->expert_w1_output, 1u, 2u);
	K3_FILL_RANK(kinds.routed, "shared_w1_weight", &b->shared_w1_rows, 0u, 2u);
	K3_FILL_RANK(kinds.routed, "shared_w2_weight", &b->shared_w2_input, 1u, 2u);
	K3_FILL_RANK(kinds.routed, "expert_w2_weight", &b->expert_w2_input, 2u, 3u);
	K3_FILL_RANK(kinds.dense, "dense_gate_up_weight", &b->dense_gate_up_rows, 0u, 2u);
	K3_FILL_RANK(kinds.dense, "dense_down_weight", &b->dense_down_input, 1u, 2u);
	if ( status != SPARK_K3_DISPATCH_OK )
		return status;
	if ( b->kda_qkvb_rows != 0u )
		b->kda_heads_rank = b->kda_qkvb_rows / 385u;
	if ( b->mla_gate_rows != 0u )
		b->mla_heads_rank = b->mla_gate_rows / K3_V_HEAD_DIM;
#undef K3_FILL_RANK
	return status;
}

int32_t SparkK3DispatchScratchWidths(SparkK3Pack *pack, uint32_t first_layer, uint32_t layer_count,
	uint32_t tp_degree, SparkK3ScratchWidths *widths)
{
	K3LayerBuffers *dims;
	int32_t status;
	if ( pack == 0 || widths == 0 || layer_count == 0u )
		return SPARK_K3_DISPATCH_ERR_ARGUMENT;
	dims = new K3LayerBuffers;
	memset(dims, 0, sizeof(*dims));
	dims->tp_sharded = tp_degree > 1u ? 1u : 0u;
	status = K3PackRankDims(pack, first_layer, layer_count, dims);
	if ( status == SPARK_K3_DISPATCH_OK )
		K3ScratchWidthsFor(dims, widths);
	delete dims;
	return status;
}

int32_t SparkK3DispatchCreate(SparkK3Dispatch *d, const SparkK3PoolSizing *sizing,
	uint32_t sequences, uint32_t max_rows, uint32_t kv_pages_per_view,
	uint64_t kv_page_bytes, uint32_t tp_degree, uint32_t tp_rank, const SparkK3ScratchWidths *widths, int device)
{
	SparkK3KdaRankLayout layout;
	SparkK3RankStateBytes state_plan;
	if ( d == 0 || sizing == 0 || widths == 0 || sequences == 0u || max_rows == 0u ||
		sizing->layer_count == 0u ||
		SparkK3KdaRankLayoutFor(tp_degree, &layout) == 0u ||
		SparkK3RankStateBytesFor(sizing->kda_layer_count,
			sizing->mla_layer_count, sequences, tp_degree,
			kv_pages_per_view, kv_page_bytes, &state_plan) == 0u )
		return SPARK_K3_DISPATCH_ERR_ARGUMENT;
	if ( tp_degree > 1u && (tp_rank >= tp_degree || SparkK3KvShardFits(tp_degree, max_rows) == 0u) )
	{
		fprintf(stderr, "sparkpipe_k3: the MLA context split cannot run at tp_degree %u rank %u with %u rows\n",
			tp_degree, tp_rank, max_rows);
		return SPARK_K3_DISPATCH_ERR_ARGUMENT;
	}
	memset(d, 0, sizeof(*d));
	d->tp_degree = tp_degree;
	d->kda_rank_heads = layout.heads;
	d->state_bytes = state_plan;
	cudaSetDevice(device);
	d->first_layer = sizing->first_layer;
	d->layer_count = sizing->layer_count;
	d->kda_count = sizing->kda_layer_count;
	d->mla_count = sizing->mla_layer_count;
	d->sequences = sequences;
	d->max_rows = max_rows;
	d->routes_capacity = max_rows * K3_TOP_K;
	d->kv_pages_per_view = kv_pages_per_view;
	d->kv_page_bytes = kv_page_bytes;
	d->widths = *widths;
	d->device = device;

	const uint64_t state_bytes = (uint64_t)d->kda_count * sequences * layout.state_slot_bytes;
	const uint64_t qk_bytes = (uint64_t)d->kda_count * sequences * layout.qk_window_slot_bytes;
	const uint64_t v_bytes = (uint64_t)d->kda_count * sequences * layout.v_window_slot_bytes;
	if ( state_bytes + 2u * qk_bytes + v_bytes !=
		state_plan.kda_state + state_plan.kda_windows )
		return SPARK_K3_DISPATCH_ERR_ARGUMENT;
	if ( cudaMalloc(&d->kda_state_pool, state_bytes) != cudaSuccess ||
		cudaMalloc(&d->kda_q_window_pool, qk_bytes) != cudaSuccess ||
		cudaMalloc(&d->kda_k_window_pool, qk_bytes) != cudaSuccess ||
		cudaMalloc(&d->kda_v_window_pool, v_bytes) != cudaSuccess )
		{ SparkK3DispatchDestroy(d); return SPARK_K3_DISPATCH_ERR_CUDA; }
	cudaMemset(d->kda_state_pool, 0, state_bytes);
	cudaMemset(d->kda_q_window_pool, 0, qk_bytes);
	cudaMemset(d->kda_k_window_pool, 0, qk_bytes);
	cudaMemset(d->kda_v_window_pool, 0, v_bytes);

	if ( cudaMalloc(&d->access_error, (size_t)(d->mla_count != 0u ? d->mla_count : 1u) * sizeof(LmKvAccessError)) != cudaSuccess )
		{ SparkK3DispatchDestroy(d); return SPARK_K3_DISPATCH_ERR_CUDA; }
	cudaMemset(d->access_error, 0, (size_t)(d->mla_count != 0u ? d->mla_count : 1u) * sizeof(LmKvAccessError));
	d->mla_cache = new LmKvView[d->mla_count != 0u ? d->mla_count : 1u];
	for ( uint32_t i = 0u; i < d->mla_count; ++i )
	{
		memset(&d->mla_cache[i], 0, sizeof(d->mla_cache[i]));
		d->mla_cache[i].access_error = d->access_error + i;
	}

	d->weights = new K3LayerWeights[d->layer_count];
	memset(d->weights, 0, (size_t)d->layer_count * sizeof(K3LayerWeights));
	d->slice_state = new K3SliceState;
	memset(d->slice_state, 0, sizeof(*d->slice_state));
	d->buffers = new K3LayerBuffers;
	d->buffers_host = d->buffers;
	memset(d->buffers_host, 0, sizeof(*d->buffers_host));

	K3SliceState *st = d->slice_state;
	st->kda_state = d->kda_state_pool;
	st->kda_q_window = d->kda_q_window_pool;
	st->kda_k_window = d->kda_k_window_pool;
	st->kda_v_window = d->kda_v_window_pool;
	st->mla_cache = d->mla_cache;
	st->sequences = sequences;
	st->kda_rank_heads = layout.heads;
	st->kda_state_bf16 = 0u;
	st->first_mla_index = d->first_layer / 4u;
	st->first_kda_index = d->first_layer - (d->first_layer / 4u);

	size_t off = 0u;
	const SparkK3ScratchWidths *w = &d->widths;
	const size_t rows = max_rows;
	const size_t query_elements = tp_degree > 1u && SparkK3KvShardQueryStride(max_rows, tp_degree) > rows * w->query
		? (size_t)SparkK3KvShardQueryStride(max_rows, tp_degree) : rows * w->query;
	const size_t sizes[] = {
		rows * K3_HIDDEN * 2u, rows * K3_HIDDEN * 2u, rows * w->qkvb * 2u,
		rows * K3_KDA_DECAY_GATE_DOWN_FUSED_ROWS * 2u, rows * K3_KDA_KEY_DIM * 2u, query_elements * 2u,
		rows * w->key * 2u, rows * w->value * 2u, rows * w->gate * 2u, rows * w->decay * 2u,
		rows * w->latent * 2u, rows * K3_MLA_KV_A_DIM * 2u, rows * w->attention_out * 2u,
		rows * K3_HIDDEN * 2u, (size_t)K3_ATTNRES_MAX_SOURCES * rows * K3_HIDDEN * 2u, rows * K3_HIDDEN * 2u,
		rows * K3_KDA_HEADS * 2u, rows * K3_KDA_HEADS * 4u, rows * w->gate_up * 2u, rows * w->intermediate * 2u,
		rows * w->retention * 4u, rows * K3_EXPERTS * 4u, rows * K3_ATTNRES_MAX_SOURCES * 4u,
		rows * K3_ROUTED_EXPERT_HIDDEN * 2u, rows * w->shared_mid * 2u};
	const size_t count = sizeof(sizes) / sizeof(sizes[0]);
	size_t index = 0u;
	d->scratch_bytes = 0u;
	for ( size_t item = 0u; item < count; ++item )
		d->scratch_bytes += (sizes[item] + 15u) & ~(size_t)15u;
	if ( tp_degree > 1u )
		d->scratch_bytes += (size_t)SparkK3KvShardScratchBytes(max_rows, tp_degree) + 48u;
	d->scratch_bytes += 256u;
	if ( cudaMalloc(&d->scratch, d->scratch_bytes) != cudaSuccess )
		{ SparkK3DispatchDestroy(d); return SPARK_K3_DISPATCH_ERR_CUDA; }
	cudaMemset(d->scratch, 0, d->scratch_bytes);
	K3LayerBuffers *b = d->buffers_host;
	b->hidden_bf16 = (uint16_t *)k3_carve(d, &off, sizes[index++]);
	b->normed_bf16 = (uint16_t *)k3_carve(d, &off, sizes[index++]);
	b->fused_qkvb_bf16 = (uint16_t *)k3_carve(d, &off, sizes[index++]);
	b->fused_decay_gate_bf16 = (uint16_t *)k3_carve(d, &off, sizes[index++]);
	b->gate_latent_bf16 = (uint16_t *)k3_carve(d, &off, sizes[index++]);
	b->query_bf16 = (uint16_t *)k3_carve(d, &off, sizes[index++]);
	b->key_bf16 = (uint16_t *)k3_carve(d, &off, sizes[index++]);
	b->value_bf16 = (uint16_t *)k3_carve(d, &off, sizes[index++]);
	b->gate_bf16 = (uint16_t *)k3_carve(d, &off, sizes[index++]);
	b->decay_logit_bf16 = (uint16_t *)k3_carve(d, &off, sizes[index++]);
	b->latent_bf16 = (uint16_t *)k3_carve(d, &off, sizes[index++]);
	b->kv_slot_bf16 = (uint16_t *)k3_carve(d, &off, sizes[index++]);
	b->attention_out_bf16 = (uint16_t *)k3_carve(d, &off, sizes[index++]);
	b->shared_out_bf16 = (uint16_t *)k3_carve(d, &off, sizes[index++]);
	b->attnres_bank_bf16 = (uint16_t *)k3_carve(d, &off, sizes[index++]);
	b->attnres_partial_bf16 = (uint16_t *)k3_carve(d, &off, sizes[index++]);
	b->kda_beta_logit = (uint16_t *)k3_carve(d, &off, sizes[index++]);
	b->kda_write_gate_out = (float *)k3_carve(d, &off, sizes[index++]);
	b->gate_up_bf16 = (uint16_t *)k3_carve(d, &off, sizes[index++]);
	b->intermediate_bf16 = (uint16_t *)k3_carve(d, &off, sizes[index++]);
	b->kda_retention = (float *)k3_carve(d, &off, sizes[index++]);
	b->router_logits = (float *)k3_carve(d, &off, sizes[index++]);
	b->attnres_score_f32 = (float *)k3_carve(d, &off, sizes[index++]);
	b->latent_full_bf16 = (uint16_t *)k3_carve(d, &off, sizes[index++]);
	b->shared_mid_bf16 = (uint16_t *)k3_carve(d, &off, sizes[index++]);
	if ( index != count )
		{ SparkK3DispatchDestroy(d); return SPARK_K3_DISPATCH_ERR_ARGUMENT; }
	if ( tp_degree > 1u )
	{
		b->kv_shard = SparkK3KvShardContext(tp_rank, tp_degree);
		b->shard_query_gathered_bf16 = (uint16_t *)k3_carve(d, &off,
			(size_t)tp_degree * SparkK3KvShardQueryStride(max_rows, tp_degree) * sizeof(uint16_t));
		b->shard_partials_f32 = (float *)k3_carve(d, &off,
			(size_t)tp_degree * SparkK3KvShardPartialStride(max_rows, tp_degree) * sizeof(float));
		b->shard_partials_received_f32 = (float *)k3_carve(d, &off,
			(size_t)tp_degree * SparkK3KvShardPartialStride(max_rows, tp_degree) * sizeof(float));
		b->shard_gather_send = (uint8_t *)b->shard_partials_f32;
		b->shard_gather_keys = (const uint8_t *)b->shard_query_gathered_bf16;
		if ( cudaMalloc(&d->gather_plan, (size_t)3u * sequences * sizeof(uint32_t)) != cudaSuccess )
			{ SparkK3DispatchDestroy(d); return SPARK_K3_DISPATCH_ERR_CUDA; }
		cudaMemset(d->gather_plan, 0, (size_t)3u * sequences * sizeof(uint32_t));
		d->pair_buffers = new K3LayerBuffers;
		memset(d->pair_buffers, 0, sizeof(*d->pair_buffers));
		if ( cudaMalloc(&d->pair_arrays, ((size_t)K3_DISPATCH_PAIR_IDENTITY + max_rows) * sizeof(uint32_t)) != cudaSuccess )
			{ SparkK3DispatchDestroy(d); return SPARK_K3_DISPATCH_ERR_CUDA; }
		cudaMemset(d->pair_arrays, 0, ((size_t)K3_DISPATCH_PAIR_IDENTITY + max_rows) * sizeof(uint32_t));
	}
	return SPARK_K3_DISPATCH_OK;
}

int32_t SparkK3DispatchAttachKv(SparkK3Dispatch *d, uint8_t *pool, uint64_t layer_stride_bytes,
	const uint32_t *page_table, uint32_t page_table_stride, uint32_t pool_page_count, uint32_t sequence_count)
{
	if ( d == 0 || d->mla_cache == 0 || (d->mla_count != 0u && (pool == 0 || page_table == 0 || layer_stride_bytes == 0u ||
		page_table_stride == 0u || pool_page_count == 0u || sequence_count == 0u || sequence_count > d->sequences)) )
		return SPARK_K3_DISPATCH_ERR_ARGUMENT;
	d->kv_pool = pool;
	d->page_table = (uint32_t *)page_table;
	for ( uint32_t i = 0u; i < d->mla_count; ++i )
	{
		d->mla_cache[i].pool = pool + (size_t)i * layer_stride_bytes;
		d->mla_cache[i].page_table = page_table;
		d->mla_cache[i].page_table_stride = page_table_stride;
		d->mla_cache[i].sequence_count = sequence_count;
		d->mla_cache[i].pool_page_count = pool_page_count;
	}
	d->kv_attached = 1u;
	return SPARK_K3_DISPATCH_OK;
}

int32_t SparkK3DispatchResetSlot(SparkK3Dispatch *d, uint32_t slot,
	uint32_t tp_degree, cudaStream_t stream)
{
	SparkK3SlotResetSpan spans[SPARK_K3_SLOT_RESET_SPANS_MAX];
	uint8_t *pools[SPARK_K3_SLOT_POOLS];
	uint32_t count;
	if ( d == 0 || d->kda_state_pool == 0 )
		return SPARK_K3_DISPATCH_ERR_ARGUMENT;
	count = SparkK3SlotResetSpans(d->kda_count, d->sequences, tp_degree, slot,
		spans, SPARK_K3_SLOT_RESET_SPANS_MAX);
	if ( count != d->kda_count * SPARK_K3_SLOT_POOLS )
		return SPARK_K3_DISPATCH_ERR_ARGUMENT;
	pools[SPARK_K3_SLOT_POOL_STATE] = d->kda_state_pool;
	pools[SPARK_K3_SLOT_POOL_Q_WINDOW] = (uint8_t *)d->kda_q_window_pool;
	pools[SPARK_K3_SLOT_POOL_K_WINDOW] = (uint8_t *)d->kda_k_window_pool;
	pools[SPARK_K3_SLOT_POOL_V_WINDOW] = (uint8_t *)d->kda_v_window_pool;
	for ( uint32_t i = 0u; i < count; ++i )
		if ( cudaMemsetAsync(pools[spans[i].pool] + spans[i].offset, 0,
				spans[i].bytes, stream) != cudaSuccess )
			return SPARK_K3_DISPATCH_ERR_CUDA;
	return SPARK_K3_DISPATCH_OK;
}

void SparkK3DispatchDestroy(SparkK3Dispatch *d)
{
	if ( d == 0 )
		return;
	cudaFree(d->kda_state_pool); cudaFree(d->kda_q_window_pool);
	cudaFree(d->kda_k_window_pool); cudaFree(d->kda_v_window_pool);
	cudaFree(d->access_error);
	cudaFree(d->scratch);
	cudaFree(d->gather_plan);
	cudaFree(d->pair_arrays);
	delete d->pair_buffers;
	delete[] d->mla_cache;
	delete[] d->weights;
	delete d->slice_state;
	delete d->buffers;
	memset(d, 0, sizeof(*d));
}

int32_t SparkK3DispatchBindWeights(SparkK3Dispatch *d, SparkK3Pack *pack,
	SparkK3BoundLayer *bounds, uint32_t layer_count,
	SparkWeightdLazyPack *lazy)
{
	if ( d == 0 || pack == 0 || bounds == 0 || lazy == 0 ||
		layer_count != d->layer_count )
		return SPARK_K3_DISPATCH_ERR_ARGUMENT;
	K3LayerWeights *host = new K3LayerWeights[layer_count];
	memset(host, 0, (size_t)layer_count * sizeof(K3LayerWeights));
	int32_t status = SPARK_K3_DISPATCH_OK;
	for ( uint32_t off = 0u; off < layer_count && status == SPARK_K3_DISPATCH_OK; ++off )
	{
		SparkK3BoundLayer *bound = &bounds[off];
		K3LayerWeights *w = &host[off];
		for ( uint32_t i = 0u; i < (uint32_t)(sizeof(k3_weight_binds) / sizeof(k3_weight_binds[0])); ++i )
		{
			const char *name = k3_weight_binds[i].name;
			size_t name_len = strlen(name);
			int32_t is_expert = name_len >= 16u &&
				strcmp(name + name_len - 16u, "expert_w1_weight") == 0;
			const void *payload = 0;
			if ( !is_expert && name_len >= 16u )
				is_expert = strcmp(name + name_len - 16u,
					"expert_w2_weight") == 0;
			if ( is_expert )
				continue;
			const SparkK3PackEntry *entry =
				SparkK3BoundEntry(bound, name);
			if ( entry == 0 )
				continue;
			status = SparkWeightdLazyPackSlice(lazy,
				pack->payload_base + entry->payload_offset,
				entry->bytes, &payload);
			if ( status != SPARK_K3_DISPATCH_OK )
				return status;
			if ( payload != 0 )
				*(const void **)((char *)w + k3_weight_binds[i].offset) = payload;
		}
		w->expert_interleave = bound->layer_is_dense ? 0u : 1u;
		w->expert_tile_k = 128u;
		if ( w->expert_interleave != 0u )
		{
			char w1_name[128];
			uint32_t tile_k = 0u;
			snprintf(w1_name, sizeof(w1_name), "model.layers.%u.expert_w1_weight",
				d->first_layer + off);
			if ( SparkK3PackLoadInterleaveTileK(pack, w1_name, &tile_k) == SPARK_STATUS_OK &&
				(tile_k == 128u || tile_k == 32u) )
				w->expert_tile_k = tile_k;
		}
		status = k3_require(bound, k3_required_every,
			(uint32_t)(sizeof(k3_required_every) / sizeof(k3_required_every[0])));
		if ( status != SPARK_K3_DISPATCH_OK )
			break;
		if ( SparkK3LayerIsMla(d->first_layer + off) )
			status = k3_require(bound, k3_required_mla,
				(uint32_t)(sizeof(k3_required_mla) / sizeof(k3_required_mla[0])));
		else
		{
			status = k3_require(bound, k3_required_kda,
				(uint32_t)(sizeof(k3_required_kda) / sizeof(k3_required_kda[0])));
			if ( status == SPARK_K3_DISPATCH_OK )
				status = k3_require_whole_head_table(bound, "kda_decay_bias",
					(uint64_t)K3_KDA_QK_DIM * sizeof(float));
			if ( status == SPARK_K3_DISPATCH_OK )
				status = k3_require_whole_head_table(bound, "kda_head_log_scale",
					(uint64_t)K3_KDA_HEADS * sizeof(float));
		}
		if ( status != SPARK_K3_DISPATCH_OK )
			break;
		if ( bound->layer_is_dense )
			status = k3_require(bound, k3_required_dense,
				(uint32_t)(sizeof(k3_required_dense) / sizeof(k3_required_dense[0])));
		else
		{
			status = k3_require(bound, k3_required_moe,
				(uint32_t)(sizeof(k3_required_moe) / sizeof(k3_required_moe[0])));
			if ( status == SPARK_K3_DISPATCH_OK )
				status = k3_require_f32_table(bound, "router_bias", K3_EXPERTS);
		}
		if ( status != SPARK_K3_DISPATCH_OK )
			break;
	}
	if ( status == SPARK_K3_DISPATCH_OK )
	{
		SparkK3ScratchWidths needs;
		status = K3PackRankDims(pack, d->first_layer, layer_count, d->buffers);
		if ( status != SPARK_K3_DISPATCH_OK )
		{
			delete[] host;
			return status;
		}
		if ( (d->buffers->kda_qkvb_rows != 0u ?
				d->buffers->kda_heads_rank : (uint32_t)K3_KDA_HEADS) !=
			d->kda_rank_heads )
		{
			fprintf(stderr, "sparkpipe_k3: pack KDA heads per rank %u != "
				"state pool heads per rank %u (tp_degree %u)\n",
				d->buffers->kda_heads_rank, d->kda_rank_heads, d->tp_degree);
			delete[] host;
			return SPARK_K3_DISPATCH_ERR_BIND;
		}
		K3ScratchWidthsFor(d->buffers, &needs);
		if ( K3ScratchFits(&needs, &d->widths) == 0u )
		{
			delete[] host;
			return SPARK_K3_DISPATCH_ERR_BIND;
		}
		SparkK3PackEntry entry;
		if ( SparkK3PackLoadEntry(pack, "model.attnres_out_weight", &entry) == 0 )
		{
			const void *slice = 0;
			status = SparkWeightdLazyPackSlice(lazy,
				pack->payload_base + entry.payload_offset,
				entry.bytes, &slice);
			if ( status != SPARK_K3_DISPATCH_OK )
			{
				delete[] host;
				return status;
			}
			d->buffers_host->attnres_out_weight = slice;
		}
		else
			d->buffers_host->attnres_out_weight = 0;
		d->buffers_host->router_bias = 0;
		memcpy(d->weights, host, (size_t)layer_count * sizeof(K3LayerWeights));
	}
	delete[] host;
	return status;
}

int32_t SparkK3DispatchShardRows(SparkK3Dispatch *d, uint32_t rows)
{
	K3LayerBuffers *b;
	if ( d == 0 || d->buffers_host == 0 || rows == 0u || rows > d->max_rows )
		return SPARK_K3_DISPATCH_ERR_ARGUMENT;
	b = d->buffers_host;
	if ( b->kv_shard.degree < 2u )
		return SPARK_K3_DISPATCH_OK;
	b->shard_query_rank_stride = SparkK3KvShardQueryStride(rows, b->kv_shard.degree);
	b->shard_partial_rank_stride = SparkK3KvShardPartialStride(rows, b->kv_shard.degree);
	return SPARK_K3_DISPATCH_OK;
}

__global__ static void K3DispatchPairKernel(uint32_t *arrays, uint32_t rows_a, uint32_t rows_b)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index == 0u )
	{
		arrays[0] = 0u; arrays[1] = rows_a; arrays[2] = 0u; arrays[3] = rows_b;
		arrays[4] = 0u; arrays[5] = rows_a; arrays[6] = 0u; arrays[7] = rows_b;
	}
	if ( index < rows_a + rows_b )
		arrays[K3_DISPATCH_PAIR_IDENTITY + index] = index;
}

static void K3DispatchPairView(const SparkK3Dispatch *d, const K3LayerBuffers *a, K3LayerBuffers *b, uint32_t rows_a)
{
	const SparkK3ScratchWidths *w = &d->widths;
	const size_t r = rows_a;
	*b = *a;
	b->hidden_bf16 += r * K3_HIDDEN;
	b->normed_bf16 += r * K3_HIDDEN;
	b->fused_qkvb_bf16 += r * w->qkvb;
	b->fused_decay_gate_bf16 += r * K3_KDA_DECAY_GATE_DOWN_FUSED_ROWS;
	b->gate_latent_bf16 += r * K3_KDA_KEY_DIM;
	b->query_bf16 += r * w->query;
	b->key_bf16 += r * w->key;
	b->value_bf16 += r * w->value;
	b->gate_bf16 += r * w->gate;
	b->decay_logit_bf16 += r * w->decay;
	b->latent_bf16 += r * w->latent;
	b->kv_slot_bf16 += r * K3_MLA_KV_A_DIM;
	b->attention_out_bf16 += r * w->attention_out;
	b->shared_out_bf16 += r * K3_HIDDEN;
	b->attnres_bank_bf16 += (size_t)K3_ATTNRES_MAX_SOURCES * r * K3_HIDDEN;
	b->attnres_partial_bf16 += r * K3_HIDDEN;
	b->kda_beta_logit += r * K3_KDA_HEADS;
	b->kda_write_gate_out += r * K3_KDA_HEADS;
	b->gate_up_bf16 += r * w->gate_up;
	b->intermediate_bf16 += r * w->intermediate;
	b->kda_retention += r * w->retention;
	b->router_logits += r * K3_EXPERTS;
	b->attnres_score_f32 += r * K3_ATTNRES_MAX_SOURCES;
	b->latent_full_bf16 += r * K3_ROUTED_EXPERT_HIDDEN;
	b->shared_mid_bf16 += r * w->shared_mid;
	b->positions = a->positions + r;
	b->sequence_of_row = a->sequence_of_row + r;
	b->sequence_row_begin = d->pair_arrays + 2u;
	b->sequence_row_indices = d->pair_arrays + K3_DISPATCH_PAIR_IDENTITY;
	b->dense_row_offset = d->pair_arrays + 6u;
	b->dense_tile_prefix = d->pair_arrays + 10u;
}

__global__ static void K3DispatchGatherPlanKernel(uint32_t *plan, uint32_t capacity, uint32_t sequence, uint32_t context)
{
	plan[0] = sequence;
	plan[capacity + sequence] = 0u;
	plan[2u * capacity + sequence] = context;
}

static int32_t K3DispatchGatherPlan(SparkK3Dispatch *d, K3LayerBuffers *b, uint32_t rows, uint32_t sequence,
	uint32_t context, cudaStream_t stream)
{
	const uint32_t degree = b->kv_shard.degree;
	const uint64_t unit = (uint64_t)K3_HIDDEN * sizeof(uint16_t);
	const uint32_t most = SparkKvShardGatherKeys(b->kv_shard, context);
	const uint64_t units = ((uint64_t)most * K3_MLA_KV_A_DIM * sizeof(uint16_t) + unit - 1u) / unit;
	const uint64_t query_units = SparkK3KvShardQuerySequences(rows, degree);
	const uint64_t partial_units = SparkK3KvShardPartialSequences(rows, degree);
	uint64_t gather_rounds, scatter_rounds;
	b->shard_gather = 0u;
	if ( d->gather_plan == 0 || sequence >= d->sequences || most == 0u ||
		units * K3_HIDDEN > SparkK3KvShardQueryStride(d->max_rows, degree) ||
		units * unit > (uint64_t)degree * SparkK3KvShardPartialStride(d->max_rows, degree) * sizeof(float) )
		return 0;
	scatter_rounds = SparkTpMeshDirectChunks(query_units * K3_HIDDEN * degree, degree, SPARK_TP_MESH_OPERATION_ALL_GATHER, SPARK_WEIGHTD_MESH_SLOT_BYTES) +
		SparkTpMeshAllToAllChunks(partial_units * K3_HIDDEN, degree, SPARK_WEIGHTD_MESH_SLOT_BYTES);
	gather_rounds = SparkTpMeshDirectChunks(units * K3_HIDDEN * degree, degree, SPARK_TP_MESH_OPERATION_ALL_GATHER, SPARK_WEIGHTD_MESH_SLOT_BYTES);
	if ( SparkKvShardExchangeCostNs(gather_rounds, (uint64_t)(degree - 1u) * units * unit) >=
		SparkKvShardExchangeCostNs(scatter_rounds, (uint64_t)(degree - 1u) * (query_units + partial_units) * unit) )
		return 0;
	K3DispatchGatherPlanKernel<<<1u, 1u, 0, stream>>>(d->gather_plan, d->sequences, sequence, context);
	if ( cudaPeekAtLastError() != cudaSuccess )
		return -1;
	b->shard_gather = 1u;
	b->shard_gather_most_keys = most;
	b->shard_gather_list = d->gather_plan;
	b->shard_gather_offset = d->gather_plan + d->sequences;
	b->shard_gather_context = d->gather_plan + 2u * d->sequences;
	b->shard_gather_stride = units * unit;
	return 1;
}

int32_t SparkK3DispatchStep(SparkK3Dispatch *d, const SparkK3StepInput *in,
	uint32_t rows, uint32_t sequences, uint32_t commit, uint32_t packed_rows,
	uint32_t context, uint32_t multiprocessors, cudaStream_t stream)
{
	if ( d == 0 || in == 0 || rows == 0u || rows > d->max_rows ||
		sequences == 0u || sequences > d->sequences ||
		SparkK3DispatchShardRows(d, rows) != SPARK_K3_DISPATCH_OK )
		return SPARK_K3_DISPATCH_ERR_ARGUMENT;
	cudaSetDevice(d->device);
	K3LayerBuffers *b = d->buffers_host;
	b->hidden_bf16 = (uint16_t *)in->hidden_in;
	b->positions = in->positions;
	b->context_length = in->context_length;
	b->sequence_of_row = in->sequence_of_row;
	b->sequence_row_begin = in->sequence_row_begin;
	b->sequence_row_indices = in->sequence_row_indices;
	b->kda_state_index = in->kda_state_index;
	b->route_expert = in->route_expert;
	b->route_packed_row = in->route_packed_row;
	b->route_source_token = in->route_source_token;
	b->route_weight = in->route_weight;
	b->group_row_offset = in->group_row_offset;
	b->group_tile_prefix_w1 = in->group_tile_prefix_w1;
	b->group_tile_prefix_w2 = in->group_tile_prefix_w2;
	b->dense_row_offset = in->dense_row_offset;
	b->dense_tile_prefix = in->dense_tile_prefix;
	b->head_candidate_score = in->head_candidate_score;
	b->head_candidate_token = in->head_candidate_token;
	b->output_token = in->output_token;
	b->output_score = in->output_score;
	b->shard_gather = 0u;
	if ( b->kv_shard.degree > 1u && rows > LM_SKINNY_ROWS_WIDE )
	{
		const int32_t gather = in->gather_context != 0u && sequences == 1u
			? K3DispatchGatherPlan(d, b, rows, in->gather_sequence, in->gather_context, stream) : 0;
		if ( gather < 0 )
			return SPARK_K3_DISPATCH_ERR_CUDA;
		if ( gather > 0 )
			d->mla_gather_waves++;
		else
			d->mla_scatter_waves++;
	}
	if ( b->shard_gather != 0u && in->pair_rows != 0u && sequences == 1u && d->pair_buffers != 0 &&
		d->pair_arrays != 0 && d->widths.latent == K3_ROUTED_EXPERT_HIDDEN && in->pair_rows < rows )
	{
		const uint32_t rows_a = in->pair_rows, rows_b = rows - in->pair_rows;
		const uint32_t *row_begin = b->sequence_row_begin, *row_indices = b->sequence_row_indices;
		uint32_t *dense_offset = b->dense_row_offset, *dense_tiles = b->dense_tile_prefix;
		int32_t status;
		K3DispatchPairKernel<<<(rows + 255u) / 256u, 256u, 0, stream>>>(d->pair_arrays, rows_a, rows_b);
		if ( cudaPeekAtLastError() != cudaSuccess )
			return SPARK_K3_DISPATCH_ERR_CUDA;
		K3DispatchPairView(d, b, d->pair_buffers, rows_a);
		if ( K3PairReady(d->slice_state, b, d->pair_buffers, rows_a, rows_b) != 0u )
		{
			b->sequence_row_begin = d->pair_arrays;
			b->sequence_row_indices = d->pair_arrays + K3_DISPATCH_PAIR_IDENTITY;
			b->dense_row_offset = d->pair_arrays + 4u;
			b->dense_tile_prefix = d->pair_arrays + 8u;
			d->pair_rows[0] = rows_a;
			d->pair_rows[1] = rows_b;
			d->pair_active = 1u;
			status = K3StageSlicePair(d->weights, d->slice_state, d->buffers, d->pair_buffers, d->first_layer,
				d->layer_count, rows_a, rows_b, commit, multiprocessors, (void *)stream);
			d->pair_active = 0u;
			b->sequence_row_begin = row_begin;
			b->sequence_row_indices = row_indices;
			b->dense_row_offset = dense_offset;
			b->dense_tile_prefix = dense_tiles;
			if ( status == LM_LAUNCH_OK )
				d->pair_waves++;
			return status;
		}
	}
	return(K3StageSlice(d->weights, d->slice_state, d->buffers, d->first_layer,
		d->layer_count, rows, sequences, commit, packed_rows, context,
		multiprocessors, (void *)stream));
}
