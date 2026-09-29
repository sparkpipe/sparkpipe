
#include <cstddef>
#include <cstdio>
#include <cstring>

#include "sparkpipe/spark_k3_resident_decode_stage_cuda.h"
#include "sparkpipe/spark_k3_weightd_include.h"

/* The weightd-mesh device kernels (extern "C" launchers named for the GLM
 * pattern that first validated them). ring/transport/tp_device_collective.c
 * — pulled into every serving adapter whose runner creates a device
 * collective — calls these, so each family's CUDA TU must instantiate them
 * into its own .so or dlopen fails closed on the undefined symbol (the
 * compile gates never dlopen; first resident launch is what catches it).
 * Same include as the glm5_next/gemma4/ling modules. */
#include "sparkpipe/spark_tp_mesh_kernels.cuh"

extern "C" int32_t K3StageSlice(const void *layer_weights, const void *slice_state,
	void *layer_buffers, uint32_t first_layer, uint32_t layer_count, uint32_t rows,
	uint32_t sequences, uint32_t commit, uint32_t packed_rows, uint32_t context,
	uint32_t multiprocessors, void *stream);

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
	"router_weight", "routed_down_weight", "routed_up_weight",
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

static uint8_t *k3_carve(SparkK3Dispatch *d, size_t *offset, size_t bytes)
{
	*offset = (*offset + 15u) & ~(size_t)15u;
	uint8_t *p = d->scratch + *offset;
	*offset += bytes;
	return p;
}

int32_t SparkK3DispatchCreate(SparkK3Dispatch *d, const SparkK3PoolSizing *sizing,
	uint32_t sequences, uint32_t max_rows, uint32_t kv_pages_per_view,
	uint64_t kv_page_bytes, uint32_t tp_degree, int device)
{
	SparkK3KdaRankLayout layout;
	SparkK3RankStateBytes state_plan;
	if ( d == 0 || sizing == 0 || sequences == 0u || max_rows == 0u ||
		sizing->layer_count == 0u ||
		SparkK3KdaRankLayoutFor(tp_degree, &layout) == 0u ||
		SparkK3RankStateBytesFor(sizing->kda_layer_count,
			sizing->mla_layer_count, sequences, tp_degree,
			kv_pages_per_view, kv_page_bytes, &state_plan) == 0u )
		return SPARK_K3_DISPATCH_ERR_ARGUMENT;
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

	uint64_t kv_total = (uint64_t)d->mla_count * kv_pages_per_view * kv_page_bytes;
	if ( cudaMalloc(&d->kv_pool, kv_total) != cudaSuccess ||
		cudaMalloc(&d->page_table, (size_t)d->mla_count * kv_pages_per_view * 4u) != cudaSuccess ||
		cudaMalloc(&d->access_error, (size_t)d->mla_count * sizeof(LmKvAccessError)) != cudaSuccess )
		{ SparkK3DispatchDestroy(d); return SPARK_K3_DISPATCH_ERR_CUDA; }
	cudaMemset(d->kv_pool, 0, kv_total);
	cudaMemset(d->access_error, 0, (size_t)d->mla_count * sizeof(LmKvAccessError));
	d->mla_cache = new LmKvView[d->mla_count];
	for ( uint32_t i = 0u; i < d->mla_count; ++i )
	{
		memset(&d->mla_cache[i], 0, sizeof(d->mla_cache[i]));
		d->mla_cache[i].pool = d->kv_pool + (size_t)i * kv_pages_per_view * kv_page_bytes;
		d->mla_cache[i].page_table = d->page_table + (size_t)i * kv_pages_per_view;
		d->mla_cache[i].page_table_stride = 1u;
		d->mla_cache[i].sequence_count = sequences;
		d->mla_cache[i].pool_page_count = kv_pages_per_view;
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
	d->scratch_bytes = 0u;
	d->scratch_bytes += (size_t)max_rows * K3_HIDDEN * 2u;
	d->scratch_bytes += (size_t)max_rows * K3_HIDDEN * 2u;
	d->scratch_bytes += (size_t)max_rows * K3_KDA_QKVB_FUSED_ROWS * 2u;
	d->scratch_bytes += (size_t)max_rows * K3_KDA_DECAY_GATE_DOWN_FUSED_ROWS * 2u;
	d->scratch_bytes += (size_t)max_rows * K3_KDA_KEY_DIM * 2u;
	d->scratch_bytes += (size_t)max_rows * K3_MLA_Q_DIM * 2u;
	d->scratch_bytes += (size_t)max_rows * K3_KDA_QK_DIM * 2u;
	d->scratch_bytes += (size_t)max_rows * K3_KDA_V_DIM * 2u;
	d->scratch_bytes += (size_t)max_rows * K3_KDA_V_DIM * 2u;
	d->scratch_bytes += (size_t)max_rows * K3_KDA_QK_DIM * 2u;
	d->scratch_bytes += (size_t)d->routes_capacity * K3_ROUTED_EXPERT_HIDDEN * 2u;
	d->scratch_bytes += (size_t)max_rows * K3_MLA_KV_A_DIM * 2u;
	d->scratch_bytes += (size_t)max_rows * K3_MLA_LATENT_OUT_DIM * 2u;
	d->scratch_bytes += (size_t)max_rows * K3_HIDDEN * 2u;
	d->scratch_bytes += (size_t)K3_ATTNRES_MAX_SOURCES * max_rows * K3_HIDDEN * 2u;
	d->scratch_bytes += (size_t)max_rows * K3_HIDDEN * 2u;
	d->scratch_bytes += (size_t)max_rows * K3_KDA_HEADS * 2u;
	d->scratch_bytes += (size_t)max_rows * K3_KDA_HEADS * 4u;
	d->scratch_bytes += (size_t)d->routes_capacity * K3_SHARED_INTERMEDIATE * 2u * 2u;
	d->scratch_bytes += (size_t)d->routes_capacity * K3_SHARED_INTERMEDIATE * 2u;
	d->scratch_bytes += (size_t)max_rows * K3_KDA_HEADS * K3_KDA_KEY_DIM * 4u;
	d->scratch_bytes += (size_t)max_rows * K3_EXPERTS * 4u;
	d->scratch_bytes += 256u;
	if ( cudaMalloc(&d->scratch, d->scratch_bytes) != cudaSuccess )
		{ SparkK3DispatchDestroy(d); return SPARK_K3_DISPATCH_ERR_CUDA; }
	cudaMemset(d->scratch, 0, d->scratch_bytes);
	K3LayerBuffers *b = d->buffers_host;
	b->hidden_bf16 = (uint16_t *)k3_carve(d, &off, (size_t)max_rows * K3_HIDDEN * 2u);
	b->normed_bf16 = (uint16_t *)k3_carve(d, &off, (size_t)max_rows * K3_HIDDEN * 2u);
	b->fused_qkvb_bf16 = (uint16_t *)k3_carve(d, &off, (size_t)max_rows * K3_KDA_QKVB_FUSED_ROWS * 2u);
	b->fused_decay_gate_bf16 = (uint16_t *)k3_carve(d, &off, (size_t)max_rows * K3_KDA_DECAY_GATE_DOWN_FUSED_ROWS * 2u);
	b->gate_latent_bf16 = (uint16_t *)k3_carve(d, &off, (size_t)max_rows * K3_KDA_KEY_DIM * 2u);
	b->query_bf16 = (uint16_t *)k3_carve(d, &off, (size_t)max_rows * K3_MLA_Q_DIM * 2u);
	b->key_bf16 = (uint16_t *)k3_carve(d, &off, (size_t)max_rows * K3_KDA_QK_DIM * 2u);
	b->value_bf16 = (uint16_t *)k3_carve(d, &off, (size_t)max_rows * K3_KDA_V_DIM * 2u);
	b->gate_bf16 = (uint16_t *)k3_carve(d, &off, (size_t)max_rows * K3_KDA_V_DIM * 2u);
	b->decay_logit_bf16 = (uint16_t *)k3_carve(d, &off, (size_t)max_rows * K3_KDA_QK_DIM * 2u);
	b->latent_bf16 = (uint16_t *)k3_carve(d, &off, (size_t)d->routes_capacity * K3_ROUTED_EXPERT_HIDDEN * 2u);
	b->kv_slot_bf16 = (uint16_t *)k3_carve(d, &off, (size_t)max_rows * K3_MLA_KV_A_DIM * 2u);
	b->attention_out_bf16 = (uint16_t *)k3_carve(d, &off, (size_t)max_rows * K3_MLA_LATENT_OUT_DIM * 2u);
	b->shared_out_bf16 = (uint16_t *)k3_carve(d, &off, (size_t)max_rows * K3_HIDDEN * 2u);
	b->attnres_bank_bf16 = (uint16_t *)k3_carve(d, &off, (size_t)K3_ATTNRES_MAX_SOURCES * max_rows * K3_HIDDEN * 2u);
	b->attnres_partial_bf16 = (uint16_t *)k3_carve(d, &off, (size_t)max_rows * K3_HIDDEN * 2u);
	b->kda_beta_logit = (uint16_t *)k3_carve(d, &off, (size_t)max_rows * K3_KDA_HEADS * 2u);
	b->kda_write_gate_out = (float *)k3_carve(d, &off, (size_t)max_rows * K3_KDA_HEADS * 4u);
	b->gate_up_bf16 = (uint16_t *)k3_carve(d, &off, (size_t)d->routes_capacity * K3_SHARED_INTERMEDIATE * 2u * 2u);
	b->intermediate_bf16 = (uint16_t *)k3_carve(d, &off, (size_t)d->routes_capacity * K3_SHARED_INTERMEDIATE * 2u);
	b->kda_retention = (float *)k3_carve(d, &off, (size_t)max_rows * K3_KDA_HEADS * K3_KDA_KEY_DIM * 4u);
	b->router_logits = (float *)k3_carve(d, &off, (size_t)max_rows * K3_EXPERTS * 4u);
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
	cudaFree(d->kv_pool); cudaFree(d->page_table); cudaFree(d->access_error);
	cudaFree(d->scratch);
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
			status = k3_require(bound, k3_required_kda,
				(uint32_t)(sizeof(k3_required_kda) / sizeof(k3_required_kda[0])));
		if ( status != SPARK_K3_DISPATCH_OK )
			break;
		if ( bound->layer_is_dense )
			status = k3_require(bound, k3_required_dense,
				(uint32_t)(sizeof(k3_required_dense) / sizeof(k3_required_dense[0])));
		else
			status = k3_require(bound, k3_required_moe,
				(uint32_t)(sizeof(k3_required_moe) / sizeof(k3_required_moe[0])));
		if ( status != SPARK_K3_DISPATCH_OK )
			break;
	}
	if ( status == SPARK_K3_DISPATCH_OK )
	{
		char dim_name[96];
		SparkK3SliceKindLayers kinds;
		SparkK3PackEntry dim_entry;
		SparkK3SliceKindLayersFor(d->first_layer, layer_count, &kinds);
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
		K3_FILL_RANK(kinds.kda, "kda_qkv_beta_weight", &d->buffers->kda_qkvb_rows, 0u, 2u);
		K3_FILL_RANK(kinds.kda, "kda_gate_weight", &d->buffers->kda_gate_rows, 0u, 2u);
		K3_FILL_RANK(kinds.kda, "kda_decay_up_weight", &d->buffers->kda_decay_up_rows, 0u, 2u);
		K3_FILL_RANK(kinds.kda, "kda_out_weight", &d->buffers->kda_out_input, 1u, 2u);
		K3_FILL_RANK(kinds.mla, "mla_q_up_weight", &d->buffers->mla_q_up_rows, 0u, 2u);
		K3_FILL_RANK(kinds.mla, "mla_gate_weight", &d->buffers->mla_gate_rows, 0u, 2u);
		K3_FILL_RANK(kinds.mla, "mla_out_weight", &d->buffers->mla_out_input, 1u, 2u);
		K3_FILL_RANK(kinds.routed, "routed_down_weight", &d->buffers->routed_down_rows, 0u, 2u);
		K3_FILL_RANK(kinds.routed, "routed_up_weight", &d->buffers->routed_up_input, 1u, 2u);
		K3_FILL_RANK(kinds.routed, "expert_w1_weight", &d->buffers->expert_w1_output, 1u, 2u);
		K3_FILL_RANK(kinds.routed, "shared_w1_weight", &d->buffers->shared_w1_rows, 0u, 2u);
		K3_FILL_RANK(kinds.routed, "shared_w2_weight", &d->buffers->shared_w2_input, 1u, 2u);
		K3_FILL_RANK(kinds.routed, "expert_w2_weight", &d->buffers->expert_w2_input, 2u, 3u);
		K3_FILL_RANK(kinds.dense, "dense_gate_up_weight", &d->buffers->dense_gate_up_rows, 0u, 2u);
		K3_FILL_RANK(kinds.dense, "dense_down_weight", &d->buffers->dense_down_input, 1u, 2u);
		if ( status != SPARK_K3_DISPATCH_OK )
		{
			delete[] host;
			return status;
		}
		if ( d->buffers->kda_qkvb_rows != 0u )
			d->buffers->kda_heads_rank = d->buffers->kda_qkvb_rows / 385u;
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
		if ( d->buffers->mla_gate_rows != 0u )
			d->buffers->mla_heads_rank = d->buffers->mla_gate_rows / K3_V_HEAD_DIM;
#undef K3_FILL_RANK
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
		for ( uint32_t off = 0u; off < layer_count; ++off )
		{
			const SparkK3PackEntry *bias_entry =
				SparkK3BoundEntry(&bounds[off], "router_bias");
			const void *bias = 0;
			if ( bias_entry == 0 )
				continue;
			status = SparkWeightdLazyPackSlice(lazy,
				pack->payload_base + bias_entry->payload_offset,
				bias_entry->bytes, &bias);
			if ( status != SPARK_K3_DISPATCH_OK )
				break;
			d->buffers_host->router_bias = (const float *)bias;
			break;
		}
		if ( status != SPARK_K3_DISPATCH_OK )
		{
			delete[] host;
			return status;
		}
		memcpy(d->weights, host, (size_t)layer_count * sizeof(K3LayerWeights));
	}
	delete[] host;
	return status;
}

int32_t SparkK3DispatchStep(SparkK3Dispatch *d, const SparkK3StepInput *in,
	uint32_t rows, uint32_t sequences, uint32_t commit, uint32_t packed_rows,
	uint32_t context, uint32_t multiprocessors, cudaStream_t stream)
{
	if ( d == 0 || in == 0 || rows == 0u || rows > d->max_rows ||
		sequences == 0u || sequences > d->sequences )
		return SPARK_K3_DISPATCH_ERR_ARGUMENT;
	cudaSetDevice(d->device);
	K3LayerBuffers *b = d->buffers_host;
	b->hidden_bf16 = (uint16_t *)in->hidden_in;
	b->positions = in->positions;
	b->context_length = in->context_length;
	b->sequence_of_row = in->sequence_of_row;
	b->sequence_row_begin = in->sequence_row_begin;
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
	return(K3StageSlice(d->weights, d->slice_state, d->buffers, d->first_layer,
		d->layer_count, rows, sequences, commit, packed_rows, context,
		multiprocessors, (void *)stream));
}
