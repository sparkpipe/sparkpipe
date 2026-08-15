// CUDA dispatch for the K3 resident decode stage. See the header for the
// ownership layout; this file is the only K3 serving-tier code that touches
// CUDA allocation and launch. The weight table is a mechanical copy of
// slice.cuh's K3LayerWeights fields onto the binder's name tables, and the
// pool carve mirrors tests/host_cuda/k3_slice_host.cu exactly, so the drift
// gate in tests/ holds the two allocators together.

#include <cstddef>
#include <cstring>

#include "sparkpipe/spark_k3_resident_decode_stage_cuda.h"

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

/* The names a layer of each kind MUST resolve. The binder's per-kind tables
 * are the single source of truth for what the pack carries; this list is the
 * same set, grouped, and a hole in it fails the bind loudly. */
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

static int32_t k3_require(const SparkK3Pack *pack, const SparkK3BoundLayer *bound,
	const char *const *names, uint32_t count)
{
	for ( uint32_t i = 0u; i < count; ++i )
		if ( SparkK3BoundPayload(pack, bound, names[i]) == 0 )
			return SPARK_K3_DISPATCH_ERR_BIND;
	return SPARK_K3_DISPATCH_OK;
}

int32_t SparkK3DispatchRegisterPack(SparkK3Pack *pack)
{
	cudaError_t err = cudaHostRegister((void *)pack->mapping, pack->file_bytes,
		cudaHostRegisterDefault);
	return(err == cudaSuccess ? SPARK_K3_DISPATCH_OK : SPARK_K3_DISPATCH_ERR_REGISTER);
}

/* One aligned carve out of the scratch blob. 16-byte alignment keeps every
 * uint16/float/uint32 array happy. */
static uint8_t *k3_carve(SparkK3Dispatch *d, size_t *offset, size_t bytes)
{
	*offset = (*offset + 15u) & ~(size_t)15u;
	uint8_t *p = d->scratch + *offset;
	*offset += bytes;
	return p;
}

int32_t SparkK3DispatchCreate(SparkK3Dispatch *d, const SparkK3PoolSizing *sizing,
	uint32_t sequences, uint32_t max_rows, uint32_t kv_pages_per_view,
	uint64_t kv_page_bytes, int device)
{
	if ( d == 0 || sizing == 0 || sequences == 0u || max_rows == 0u ||
		sizing->layer_count == 0u )
		return SPARK_K3_DISPATCH_ERR_ARGUMENT;
	memset(d, 0, sizeof(*d));
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

	/* Recurrent pools, per the host-test carve the slice strides against. */
	const uint64_t qk_window = (uint64_t)K3_KDA_QK_DIM * K3_KDA_CONV_KERNEL;
	const uint64_t v_window = (uint64_t)K3_KDA_V_DIM * K3_KDA_CONV_KERNEL;
	const uint64_t state_bytes = (uint64_t)d->kda_count * sequences * K3_KDA_STATE_SLOT_BYTES;
	const uint64_t qk_bytes = (uint64_t)d->kda_count * sequences * qk_window * 2u;
	const uint64_t v_bytes = (uint64_t)d->kda_count * sequences * v_window * 2u;
	if ( cudaMalloc(&d->kda_state_pool, state_bytes) != cudaSuccess ||
		cudaMalloc(&d->kda_q_window_pool, qk_bytes) != cudaSuccess ||
		cudaMalloc(&d->kda_k_window_pool, qk_bytes) != cudaSuccess ||
		cudaMalloc(&d->kda_v_window_pool, v_bytes) != cudaSuccess )
		{ SparkK3DispatchDestroy(d); return SPARK_K3_DISPATCH_ERR_CUDA; }
	cudaMemset(d->kda_state_pool, 0, state_bytes);

	/* MLA caches: the view structs are HOST memory (K3BindLayerState copies
	 * one by value into the buffers on the host); the pools, page tables
	 * and error slots they point at are device memory. */
	uint64_t kv_total = (uint64_t)d->mla_count * kv_pages_per_view * kv_page_bytes;
	if ( cudaMalloc(&d->kv_pool, kv_total) != cudaSuccess ||
		cudaMalloc(&d->page_table, (size_t)d->mla_count * kv_pages_per_view * 4u) != cudaSuccess ||
		cudaMalloc(&d->access_error, (size_t)d->mla_count * sizeof(LmKvAccessError)) != cudaSuccess )
		{ SparkK3DispatchDestroy(d); return SPARK_K3_DISPATCH_ERR_CUDA; }
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

	/* THE LAUNCH STRUCTS ARE HOST MEMORY. K3LaunchSlice runs its per-layer
	 * loop ON THE HOST (K3BindLayer/K3BindLayerState mutate the buffers
	 * struct between launches and only the individual pointer FIELDS are
	 * passed to kernels as arguments), so weights, slice_state and buffers
	 * are plain host structs whose members are device pointers. A
	 * device-allocated copy of any of them would be dereferenced by host
	 * code and fault. */
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
	st->kda_state_bf16 = 0u;

	/* Scratch blob, carved per the host test's static arrays. */
	size_t off = 0u;
	d->scratch_bytes = 0u;
	d->scratch_bytes += (size_t)max_rows * K3_HIDDEN * 2u;            /* hidden */
	d->scratch_bytes += (size_t)max_rows * K3_HIDDEN * 2u;            /* normed */
	d->scratch_bytes += (size_t)max_rows * K3_KDA_QKVB_FUSED_ROWS * 2u;
	d->scratch_bytes += (size_t)max_rows * K3_KDA_DECAY_GATE_DOWN_FUSED_ROWS * 2u;
	d->scratch_bytes += (size_t)max_rows * K3_KDA_KEY_DIM * 2u;       /* gate_latent */
	d->scratch_bytes += (size_t)max_rows * K3_MLA_Q_DIM * 2u;         /* query */
	d->scratch_bytes += (size_t)max_rows * K3_KDA_QK_DIM * 2u;        /* key */
	d->scratch_bytes += (size_t)max_rows * K3_KDA_V_DIM * 2u;         /* value */
	d->scratch_bytes += (size_t)max_rows * K3_KDA_V_DIM * 2u;         /* gate */
	d->scratch_bytes += (size_t)max_rows * K3_KDA_QK_DIM * 2u;        /* decay_logit */
	d->scratch_bytes += (size_t)d->routes_capacity * K3_ROUTED_EXPERT_HIDDEN * 2u;
	d->scratch_bytes += (size_t)max_rows * K3_MLA_KV_A_DIM * 2u;      /* kv_slot */
	d->scratch_bytes += (size_t)max_rows * K3_MLA_LATENT_OUT_DIM * 2u;/* attention_out */
	d->scratch_bytes += (size_t)max_rows * K3_HIDDEN * 2u;            /* shared_out */
	d->scratch_bytes += (size_t)K3_ATTNRES_MAX_SOURCES * max_rows * K3_HIDDEN * 2u;
	d->scratch_bytes += (size_t)max_rows * K3_HIDDEN * 2u;            /* partial */
	d->scratch_bytes += (size_t)max_rows * K3_KDA_HEADS * 2u;         /* beta_logit */
	d->scratch_bytes += (size_t)max_rows * K3_KDA_HEADS * 4u;         /* write_gate */
	d->scratch_bytes += (size_t)d->routes_capacity * K3_SHARED_INTERMEDIATE * 2u * 2u;
	d->scratch_bytes += (size_t)d->routes_capacity * K3_SHARED_INTERMEDIATE * 2u;
	d->scratch_bytes += (size_t)max_rows * K3_KDA_HEADS * K3_KDA_KEY_DIM * 4u;
	d->scratch_bytes += (size_t)max_rows * K3_EXPERTS * 4u;           /* router_logits */
	d->scratch_bytes += 256u;
	if ( cudaMalloc(&d->scratch, d->scratch_bytes) != cudaSuccess )
		{ SparkK3DispatchDestroy(d); return SPARK_K3_DISPATCH_ERR_CUDA; }
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
	SparkK3BoundLayer *bounds, uint32_t layer_count)
{
	if ( d == 0 || pack == 0 || bounds == 0 || layer_count != d->layer_count )
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
			const void *payload = SparkK3BoundPayload(pack, bound, k3_weight_binds[i].name);
			if ( payload != 0 )
				*(const void **)((char *)w + k3_weight_binds[i].offset) = payload;
		}
		w->expert_interleave = bound->layer_is_dense ? 0u : 1u;
		status = k3_require(pack, bound, k3_required_every,
			(uint32_t)(sizeof(k3_required_every) / sizeof(k3_required_every[0])));
		if ( status != SPARK_K3_DISPATCH_OK )
			break;
		if ( SparkK3LayerIsMla(d->first_layer + off) )
			status = k3_require(pack, bound, k3_required_mla,
				(uint32_t)(sizeof(k3_required_mla) / sizeof(k3_required_mla[0])));
		else
			status = k3_require(pack, bound, k3_required_kda,
				(uint32_t)(sizeof(k3_required_kda) / sizeof(k3_required_kda[0])));
		if ( status != SPARK_K3_DISPATCH_OK )
			break;
		if ( bound->layer_is_dense )
			status = k3_require(pack, bound, k3_required_dense,
				(uint32_t)(sizeof(k3_required_dense) / sizeof(k3_required_dense[0])));
		else
			status = k3_require(pack, bound, k3_required_moe,
				(uint32_t)(sizeof(k3_required_moe) / sizeof(k3_required_moe[0])));
		if ( status != SPARK_K3_DISPATCH_OK )
			break;
	}
	if ( status == SPARK_K3_DISPATCH_OK )
	{
		/* Model-level fields the slice loop never sets. */
		SparkK3PackEntry entry;
		if ( SparkK3PackLoadEntry(pack, "model.attnres_out_weight", &entry) == 0 )
			d->buffers_host->attnres_out_weight = SparkK3PackPayload(pack, &entry);
		else
			d->buffers_host->attnres_out_weight = 0;
		d->buffers_host->router_bias = 0;
		for ( uint32_t off = 0u; off < layer_count; ++off )
		{
			const void *bias = SparkK3BoundPayload(pack, &bounds[off], "router_bias");
			if ( bias != 0 )
			{
				d->buffers_host->router_bias = (const float *)bias;
				break;
			}
		}
		/* The launch structs are host memory (see Create); the bound table
		 * copies straight over and the loop reads it on the host. */
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
	/* The host struct IS the launch struct: the slice loop mutates it on the
	 * host between launches and the kernels receive individual pointer
	 * fields as arguments, so there is no device copy of it. */
	return(K3StageSlice(d->weights, d->slice_state, d->buffers, d->first_layer,
		d->layer_count, rows, sequences, commit, packed_rows, context,
		multiprocessors, (void *)stream));
}
