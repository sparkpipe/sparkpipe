// Run the real K3 slice loop on a CPU: fourteen layers, two rows, the AttnRes
// stream carried between them by the code that ships.
//
// The per-layer harness proved every kernel and one whole layer; the wiring
// audit then found five defects in the LOOP - retrieval joined to a residual,
// a dead MLP-side retrieval, a partial that missed the attention output, a
// partial that never reset, a missing output retrieval - and one more in the
// state binding, where every KDA layer shared one slot and every MLA layer one
// cache. All of them live between layers, which is exactly the span no other
// harness executes.
//
// Layers 0..13 cover both block boundaries (0 and 12), three MLA layers
// (3, 7, 11), the dense layer 0 and the MoE everywhere else. The GEMM is the
// recorder - output 0.125 * call index, constant across rows - so the whole
// partial/bank trajectory is closed-form arithmetic the checker recomputes.
// Everything else is the shipping kernel. The slice is called one layer at a
// time so the stream can be observed between layers; the loop-carried state
// lives in the buffers and pools, so the sequencing is identical to one call.

#include "tests/host_cuda/lm_host_cuda.cuh"

#include <stdio.h>
#include <vector>

LmHostDim3 blockIdx, threadIdx, blockDim, gridDim;

uint32_t lm_topk_shared[LM_HOST_SHARED_BYTES / sizeof(uint32_t)];
float lm_norm_shared[LM_HOST_SHARED_BYTES / sizeof(float)];
float state_s[LM_HOST_SHARED_BYTES / sizeof(float)];
float lm_fused_shared[LM_HOST_SHARED_BYTES / sizeof(float)];
float lm_quant_shared[LM_HOST_SHARED_BYTES / sizeof(float)];

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/tile.cuh"
#include "inference/kernels/mma.cuh"
#undef LM_WARP_LANES
#define LM_WARP_LANES LM_HOST_WARP_LANES

#include "runtime/gemm.cuh"
std::vector<LmRecordedGemm> lm_recorded_gemms;

// kv.cuh guards its store kernel on __CUDACC__; scope the macro to that one
// include exactly as k3_layer_host.cu does, so dtype.cuh never sees it.
#define __CUDACC__ 1
#include "inference/kernels/kv.cuh"
#undef __CUDACC__

struct LmHostRecorderFormat
{
	static constexpr uint32_t kScaleGroup = 32u;
	static constexpr uint32_t kTileK = 128u;
	static constexpr uint32_t kStoredBits = 4u;
	static constexpr float kMax = 6.0f;
	static __device__ __forceinline__ uint8_t Encode(float value)
	{
		float clamped = value > kMax ? kMax : (value < -kMax ? -kMax : value);
		return((uint8_t)(((int)clamped) & 15));
	}
};
#include "inference/llms/kimi_k3/slice.cuh"

#define ROWS 2u
#define ROUTES (ROWS * K3_TOP_K)
#define SLICE_LAYERS 14u
#define SLICE_KDA 11u
#define SLICE_MLA 3u
#define KV_POOL_BYTES (ROWS * K3GlobalKv::kPageBytes)

static uint16_t hidden[ROWS * K3_HIDDEN], normed[ROWS * K3_HIDDEN];
static uint16_t attention_out[ROWS * K3_MLA_LATENT_OUT_DIM];
static uint16_t shared_out[ROWS * K3_HIDDEN];
static uint16_t query[ROWS * K3_MLA_Q_DIM], key[ROWS * K3_KDA_QK_DIM];
static uint16_t value[ROWS * K3_KDA_V_DIM], gate[ROWS * K3_KDA_V_DIM];
static uint16_t decay_logit[ROWS * K3_KDA_QK_DIM];
static uint16_t latent[ROUTES * K3_ROUTED_EXPERT_HIDDEN];
// The pack-V2 wide scratch the fused q|k|v|beta GEMM lands in. decay_down is
// standalone now (docs/K3_GATE_RECONCILIATION.md) and projects into latent;
// the recorder never reads weight CONTENT - pointer identity is what proves
// the bind propagated the tensors - so the weight stand-ins are one row each.
static uint16_t fused_qkvb[ROWS * K3_KDA_QKVB_FUSED_ROWS];
static uint16_t qkvb_weight[K3_HIDDEN], decay_down_weight[K3_HIDDEN];
static uint16_t aux_slab[K3_DSPARK_AUX_LAYER_COUNT * ROWS * K3_HIDDEN];
static uint16_t aux_reference[ROWS * K3_HIDDEN];
static uint16_t kv_slot[ROWS * K3_MLA_KV_A_DIM];
static uint16_t gate_up[ROUTES * K3_SHARED_INTERMEDIATE * 2u];
static uint16_t intermediate[ROUTES * K3_SHARED_INTERMEDIATE];
static uint16_t bank[K3_ATTNRES_MAX_SOURCES * ROWS * K3_HIDDEN];
static uint16_t partial[ROWS * K3_HIDDEN];
static uint16_t beta_logit[ROWS * K3_KDA_HEADS];
static float write_gate[ROWS * K3_KDA_HEADS];
static float retention[ROWS * K3_KDA_HEADS * K3_KDA_KEY_DIM];
static float router_logits[ROWS * K3_EXPERTS], router_bias[K3_EXPERTS];
static float route_weight[ROUTES];
static uint32_t route_expert[ROUTES], route_packed[ROUTES], route_source[ROUTES];
static uint32_t group_offsets[K3_EXPERTS + 1u], group_tiles[K3_EXPERTS + 1u];
static uint32_t group_tiles_down[K3_EXPERTS + 1u];
static uint32_t dense_offsets[2], dense_tiles[2];
static uint16_t ones_weight[K3_HIDDEN];
static float ones_f32[K3_KDA_VALUE_DIM];
static uint16_t attn_query_weight[K3_HIDDEN], mlp_query_weight[K3_HIDDEN];
static float conv_weight[K3_KDA_V_DIM * K3_KDA_CONV_KERNEL];
static uint16_t head_project_weight[K3_MLA_HEADS * K3_V_HEAD_DIM * K3_KV_LORA_RANK];
static float decay_bias[K3_KDA_QK_DIM], head_log_scale[K3_KDA_HEADS];
static uint8_t kda_state[SLICE_KDA * ROWS * K3_KDA_STATE_SLOT_BYTES];
static uint16_t q_window[SLICE_KDA * ROWS * K3_KDA_QK_DIM * K3_KDA_CONV_KERNEL];
static uint16_t k_window[SLICE_KDA * ROWS * K3_KDA_QK_DIM * K3_KDA_CONV_KERNEL];
static uint16_t v_window[SLICE_KDA * ROWS * K3_KDA_V_DIM * K3_KDA_CONV_KERNEL];
static uint8_t kv_pool[SLICE_MLA][KV_POOL_BYTES];
static uint32_t page_table[SLICE_MLA][ROWS];
static LmKvView cache_views[SLICE_MLA];
static LmKvAccessError cache_errors[SLICE_MLA];
static uint32_t state_index[ROWS], sequence_of_row[ROWS];
static uint32_t positions[ROWS], context_length[ROWS];

static uint32_t seed = 24680u;
static float NextRandom(void)
{
	seed = (seed * 1664525u) + 1013904223u;
	return (float)((seed >> 8) & 0xffffu) / 32768.0f - 1.0f;
}

static void Emit(const char *tag, const uint16_t *values, uint32_t count)
{
	uint32_t index;
	for (index = 0u; index < count; ++index)
		printf("%s %.9g\n", tag, (double)LmBf16ToFloat(values[index]));
}

static const char *GemmName(const void *output)
{
	return output == (void *)hidden ? "hidden"
		: output == (void *)shared_out ? "shared_out"
		: output == (void *)attention_out ? "attention_out"
		: output == (void *)latent ? "latent"
		: output == (void *)fused_qkvb ? "fused_qkvb"
		: output == (void *)query ? "query"
		: output == (void *)key ? "key"
		: output == (void *)value ? "value"
		: output == (void *)gate ? "gate"
		: output == (void *)decay_logit ? "decay"
		: output == (void *)beta_logit ? "beta"
		: output == (void *)kv_slot ? "kv_slot"
		: output == (void *)gate_up ? "gate_up"
		: output == (void *)router_logits ? "router"
		: "other";
}

// The weight a recorded GEMM consumed, named the same way. The fused KDA
// tensor is the pack-V2 bind contract; a KDA layer recording any other
// weight against normed_bf16 except the standalone decay_down and the
// full-rank gate is the six-launch block come back.
static const char *WeightName(const void *weight)
{
	return weight == (const void *)qkvb_weight ? "qkvb"
		: weight == (const void *)decay_down_weight ? "decay_down"
		: "other";
}

int main(void)
{
	static K3LayerWeights weights[SLICE_LAYERS];
	static K3LayerBuffers b;
	static K3SliceState state;
	uint32_t index, layer, view, logged = 0u;
	int32_t status;
	memset(&b, 0, sizeof(b));
	for (index = 0u; index < K3_HIDDEN; ++index)
	{
		ones_weight[index] = LmFloatToBf16(1.0f);
		attn_query_weight[index] = LmFloatToBf16(NextRandom() * 0.05f);
		mlp_query_weight[index] = LmFloatToBf16(NextRandom() * 0.05f);
	}
	for (index = 0u; index < K3_KDA_VALUE_DIM; ++index)
		ones_f32[index] = 1.0f;
	for (index = 0u; index < ROWS * K3_HIDDEN; ++index)
		hidden[index] = LmFloatToBf16(0.01f * (float)(index % 17) + 0.02f);
	for (index = 0u; index < ROUTES; ++index)
	{
		route_weight[index] = 1.0f / (float)K3_TOP_K;
	}
	for (index = 0u; index < ROWS; ++index)
	{
		state_index[index] = index;
		sequence_of_row[index] = index;
		positions[index] = 0u;
		context_length[index] = 1u;
	}
	for (view = 0u; view < SLICE_MLA; ++view)
	{
		page_table[view][0] = 0u;
		page_table[view][1] = 1u;
		LmKvAccessErrorReset(&cache_errors[view]);
		cache_views[view].pool = kv_pool[view];
		cache_views[view].page_table = page_table[view];
		cache_views[view].page_table_stride = 1u;
		cache_views[view].sequence_count = ROWS;
		cache_views[view].pool_page_count = ROWS;
		cache_views[view].access_error = &cache_errors[view];
	}
	for (layer = 0u; layer < SLICE_LAYERS; ++layer)
	{
		K3LayerWeights *w = &weights[layer];
		memset(w, 0, sizeof(*w));
		w->attn_norm_weight = ones_weight; w->mlp_norm_weight = ones_weight;
		w->routed_norm_weight = ones_weight;
		w->mla_q_norm_weight = ones_weight; w->mla_kv_a_norm_weight = ones_weight;
		w->kda_out_norm_weight = ones_f32;
		w->kda_qkv_beta_weight = qkvb_weight;
		w->kda_decay_down_weight = decay_down_weight;
		w->kda_q_conv_weight = conv_weight; w->kda_k_conv_weight = conv_weight;
		w->kda_v_conv_weight = conv_weight;
		w->kda_decay_bias = decay_bias; w->kda_head_log_scale = head_log_scale;
		w->mla_kv_b_value_weight = head_project_weight;
		w->attnres_attn_weight = attn_query_weight;
		w->attnres_mlp_weight = mlp_query_weight;
	}
	b.hidden_bf16 = hidden; b.normed_bf16 = normed;
	b.attention_out_bf16 = attention_out; b.shared_out_bf16 = shared_out;
	b.fused_qkvb_bf16 = fused_qkvb;
	b.query_bf16 = query; b.key_bf16 = key; b.value_bf16 = value;
	b.gate_bf16 = gate; b.decay_logit_bf16 = decay_logit;
	b.latent_bf16 = latent; b.kv_slot_bf16 = kv_slot;
	b.gate_up_bf16 = gate_up; b.intermediate_bf16 = intermediate;
	b.attnres_bank_bf16 = bank; b.attnres_partial_bf16 = partial;
	b.attnres_out_weight = attn_query_weight;
	b.kda_beta_logit = beta_logit; b.kda_write_gate_out = write_gate;
	b.kda_retention = retention;
	b.router_logits = router_logits; b.router_bias = router_bias;
	b.route_expert = route_expert; b.route_weight = route_weight;
	b.route_packed_row = route_packed; b.route_source_token = route_source;
	b.group_row_offset = group_offsets; b.group_tile_prefix_w1 = group_tiles;
	b.group_tile_prefix_w2 = group_tiles_down;
	dense_offsets[0] = 0u; dense_offsets[1] = ROWS;
	b.dense_row_offset = dense_offsets; b.dense_tile_prefix = dense_tiles;
	b.kda_state_index = state_index; b.sequence_of_row = sequence_of_row;
	b.positions = positions; b.context_length = context_length;
	state.kda_state = kda_state;
	state.kda_q_window = q_window; state.kda_k_window = k_window;
	state.kda_v_window = v_window;
	state.mla_cache = cache_views;
	state.sequences = ROWS;
	printf("rows %u layers %u\n", ROWS, SLICE_LAYERS);
	Emit("embedding", hidden, ROWS * K3_HIDDEN);
	Emit("attnw", attn_query_weight, K3_HIDDEN);
	Emit("mlpw", mlp_query_weight, K3_HIDDEN);
	// VERIFY THEN FOLD MUST LAND WHERE COMMIT WOULD HAVE. Layer 0 is KDA: run
	// its two rows as one COMMITTED run of one row (the truth), then from zero
	// state run both rows with commit off - slabs filling, state untouched -
	// and fold accepted = 1. The folded state, and all three conv windows,
	// must equal the truth byte for byte; the kda gate already proved the
	// kernels equivalent, so any difference here is the plumbing lying.
	{
		static uint16_t rq[2u * K3_KDA_QK_DIM], rk[2u * K3_KDA_QK_DIM];
		static uint16_t rv[2u * K3_KDA_V_DIM];
		static float replay_retention[2u * K3_KDA_QK_DIM];
		static float replay_write_gate[2u * K3_KDA_HEADS];
		static uint8_t truth_state[sizeof(kda_state)];
		static uint16_t truth_q[sizeof(q_window) / 2u], truth_k[sizeof(k_window) / 2u];
		static uint16_t truth_v[sizeof(v_window) / 2u];
		static uint32_t vbegin[2] = { 0u, 0u }, vcount[1] = { 1u };
		static uint16_t hidden_saved[2u * K3_HIDDEN];
		uint32_t mismatch = 0u; uint64_t byte;
		// The recorder derives every GEMM's output from the global call index,
		// so the truth run and the verify run must count from the same zero or
		// they are different models by construction. And the slice advances the
		// stream through hidden, which the main loop below treats as the
		// embedding - saved here, restored on the way out.
		memcpy(hidden_saved, hidden, sizeof(hidden_saved));
		lm_recorded_gemms.clear();
		state.verify_rows = 2u;
		state.dspark_aux = aux_slab; state.aux_rows = ROWS;
		b.sequence_row_begin = 0;
		status = K3LaunchSlice<LmHostRecorderFormat,K3GlobalKv>(
			&weights[0], &state, &b, 0u, 1u, 1u, 1u, 1u, ROUTES, 1u, 1u, 0);
		if ( status != LM_LAUNCH_OK ) { printf("FAIL truth run %d\n", status); return 1; }
		memcpy(truth_state, kda_state, sizeof(truth_state));
		memcpy(truth_q, q_window, sizeof(truth_q));
		memcpy(truth_k, k_window, sizeof(truth_k));
		memcpy(truth_v, v_window, sizeof(truth_v));
		memset(kda_state, 0, sizeof(kda_state));
		memset(q_window, 0, sizeof(q_window));
		memset(k_window, 0, sizeof(k_window));
		memset(v_window, 0, sizeof(v_window));
		state.replay_conv_q = rq; state.replay_conv_k = rk; state.replay_conv_v = rv;
		state.replay_retention = replay_retention;
		state.replay_write_gate = replay_write_gate;
		static uint32_t two_run[2] = { 0u, 2u };
		memcpy(hidden, hidden_saved, sizeof(hidden_saved));
		lm_recorded_gemms.clear();
		b.sequence_row_begin = two_run;
		status = K3LaunchSlice<LmHostRecorderFormat,K3GlobalKv>(
			&weights[0], &state, &b, 0u, 1u, 2u, 1u, 0u, ROUTES, 1u, 1u, 0);
		if ( status != LM_LAUNCH_OK ) { printf("FAIL verify run %d\n", status); return 1; }
		for (byte = 0u; byte < sizeof(kda_state); ++byte)
			if ( kda_state[byte] != 0u ) ++mismatch;
		printf("verify_untouched %u\n", mismatch);
		status = K3FoldAccepted<LmHostRecorderFormat>(
			&weights[0], &state, &b, 0u, 1u, 1u, vbegin, vcount, 2u, 1u, 0);
		if ( status != LM_LAUNCH_OK ) { printf("FAIL fold %d\n", status); return 1; }
		mismatch = 0u;
		for (byte = 0u; byte < sizeof(kda_state); ++byte)
			if ( kda_state[byte] != truth_state[byte] ) ++mismatch;
		for (byte = 0u; byte < sizeof(truth_q); ++byte)
			if ( q_window[byte] != truth_q[byte] || k_window[byte] != truth_k[byte]
				|| v_window[byte] != truth_v[byte] ) ++mismatch;
		printf("fold_mismatch %u\n", mismatch);
		state.replay_conv_q = 0; state.replay_conv_k = 0; state.replay_conv_v = 0;
		state.replay_retention = 0; state.replay_write_gate = 0;
		memset(kda_state, 0, sizeof(kda_state));
		memset(q_window, 0, sizeof(q_window));
		memset(k_window, 0, sizeof(k_window));
		memset(v_window, 0, sizeof(v_window));
		b.sequence_row_begin = 0;
		memcpy(hidden, hidden_saved, sizeof(hidden_saved));
		lm_recorded_gemms.clear();
	}
	for (layer = 0u; layer < SLICE_LAYERS; ++layer)
	{
		status = K3LaunchSlice<LmHostRecorderFormat,K3GlobalKv>(
			&weights[layer], &state, &b, layer, 1u, ROWS, ROWS, 1u, ROUTES, 1u, 1u, 0);
		if ( status != LM_LAUNCH_OK )
		{
			printf("FAIL layer %u status %d\n", layer, (int)status);
			return 1;
		}
		if ( layer == 7u )
			memcpy(aux_reference, partial, sizeof(aux_reference));
		printf("layer %u state_offset %llu cache %u\n", layer,
			(unsigned long long)((const uint8_t *)b.kda_state_pool - kda_state),
			K3_LAYER_KIND(layer) == LM_LAYER_LATENT
				? (uint32_t)(b.cache.pool == kv_pool[layer / 4u] ? layer / 4u : 999u)
				: 998u);
		for (; logged < lm_recorded_gemms.size(); ++logged)
			printf("gemm %u layer %u dest %s wgt %s\n", logged + 1u, layer,
				GemmName(lm_recorded_gemms[logged].output),
				WeightName(lm_recorded_gemms[logged].weight));
		Emit("partial", partial, ROWS * K3_HIDDEN);
		Emit("stream", hidden, ROWS * K3_HIDDEN);
		Emit("normed", normed, ROWS * K3_HIDDEN);
		Emit("attnout", attention_out, ROWS * K3_HIDDEN);
	}
	Emit("bank0", bank, ROWS * K3_HIDDEN);
	Emit("bank1", bank + (1u * ROWS * K3_HIDDEN), ROWS * K3_HIDDEN);
	printf("done\n");
	{
		uint32_t mismatch = 0u; uint64_t byte;
		for (byte = 0u; byte < (uint64_t)ROWS * K3_HIDDEN; ++byte)
			if ( aux_slab[byte] != aux_reference[byte] )
				++mismatch;
		printf("aux_mismatch %u\n", mismatch);
	}
	return 0;
}
