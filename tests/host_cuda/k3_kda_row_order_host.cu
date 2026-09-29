#include "tests/host_cuda/lm_host_cuda.cuh"

#include <stdio.h>
#include <string.h>
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

#define __CUDACC__ 1
#include "inference/kernels/kv.cuh"
#undef __CUDACC__

#include "inference/llms/kimi_k3/layer.cuh"

#define TP_DEGREE 16u
#define RANK_HEADS (K3_KDA_HEADS / TP_DEGREE)
#define RANK_QK (RANK_HEADS * K3_KDA_KEY_DIM)
#define RANK_V (RANK_HEADS * K3_KDA_VALUE_DIM)
#define FUSED (RANK_HEADS * (2u * K3_KDA_KEY_DIM + K3_KDA_VALUE_DIM + 1u))
#define SEQUENCES 2u
#define STEPS 2u
#define ROWS (SEQUENCES * STEPS)
#define SLOT_BYTES ((size_t)RANK_HEADS * K3_KDA_KEY_DIM * K3_KDA_VALUE_DIM * sizeof(float))
static uint16_t hidden[ROWS * K3_HIDDEN], normed[ROWS * K3_HIDDEN];
static uint16_t norm_weight[K3_HIDDEN], weight_token[K3_HIDDEN];
static uint16_t fused_qkvb[ROWS * FUSED];
static uint16_t latent[ROWS * K3_KDA_KEY_DIM];
static uint16_t gate[ROWS * RANK_V];
static uint16_t query[ROWS * RANK_QK], key[ROWS * RANK_QK], value[ROWS * RANK_V];
static uint16_t beta_logit[ROWS * RANK_HEADS];
static uint16_t decay_logit[ROWS * RANK_QK];
static uint16_t attention_out[ROWS * RANK_V];
static float conv_weight[K3_KDA_V_DIM * K3_KDA_CONV_KERNEL];
static float out_norm_weight[K3_KDA_VALUE_DIM];
static float decay_bias[K3_KDA_QK_DIM], head_log_scale[K3_KDA_HEADS];
static float retention[ROWS * RANK_QK], write_gate[ROWS * RANK_HEADS];
static uint16_t q_window[SEQUENCES * RANK_QK * K3_KDA_CONV_KERNEL];
static uint16_t k_window[SEQUENCES * RANK_QK * K3_KDA_CONV_KERNEL];
static uint16_t v_window[SEQUENCES * RANK_V * K3_KDA_CONV_KERNEL];
static uint8_t state_pool[SEQUENCES * SLOT_BYTES];
static uint32_t state_index[SEQUENCES] = { 1u, 0u };
static uint32_t row_begin[SEQUENCES + 1u] = { 0u, STEPS, ROWS };
static uint32_t interleaved[ROWS];
static uint32_t dense_offsets[2], dense_tiles[2];
static uint16_t grouped_out[ROWS * RANK_V];
static uint8_t grouped_state[sizeof(state_pool)];
static uint16_t grouped_q[sizeof(q_window) / 2u], grouped_k[sizeof(k_window) / 2u], grouped_v[sizeof(v_window) / 2u];

static void Bind(K3LayerBuffers *b)
{
	memset(b, 0, sizeof(*b));
	b->hidden_bf16 = hidden; b->normed_bf16 = normed;
	b->attn_norm_weight = norm_weight;
	b->kda_qkv_beta_weight = weight_token;
	b->kda_decay_down_weight = weight_token;
	b->kda_decay_up_weight = weight_token;
	b->kda_gate_weight = weight_token;
	b->kda_out_weight = weight_token;
	b->kda_q_conv_weight = conv_weight;
	b->kda_k_conv_weight = conv_weight;
	b->kda_v_conv_weight = conv_weight;
	b->kda_decay_bias = decay_bias;
	b->kda_head_log_scale = head_log_scale;
	b->kda_out_norm_weight = out_norm_weight;
	b->fused_qkvb_bf16 = fused_qkvb;
	b->latent_bf16 = latent; b->gate_bf16 = gate;
	b->query_bf16 = query; b->key_bf16 = key; b->value_bf16 = value;
	b->kda_beta_logit = beta_logit;
	b->decay_logit_bf16 = decay_logit;
	b->attention_out_bf16 = attention_out;
	b->kda_retention = retention;
	b->kda_write_gate_out = write_gate;
	b->kda_q_window = q_window; b->kda_k_window = k_window;
	b->kda_v_window = v_window;
	b->kda_state_pool = state_pool;
	b->kda_state_index = state_index;
	b->sequence_row_begin = row_begin;
	b->dense_row_offset = dense_offsets;
	b->dense_tile_prefix = dense_tiles;
	b->tp_sharded = 1u;
	b->tp_rank = 0u;
	b->kda_heads_rank = RANK_HEADS;
	b->kda_qkvb_rows = FUSED;
	b->kda_gate_rows = RANK_V;
	b->kda_decay_up_rows = RANK_QK;
	b->kda_out_input = RANK_V;
}

static float Value(uint32_t sequence, uint32_t step, uint32_t index, uint32_t salt)
{
	return(0.02f * (float)((index * 7u + sequence * 13u + step * 29u + salt * 3u) % 41u) - 0.4f);
}

static void Fill(uint32_t grouped)
{
	uint32_t sequence, step, index;
	for (sequence = 0u; sequence < SEQUENCES; ++sequence)
		for (step = 0u; step < STEPS; ++step)
		{
			uint32_t row = grouped != 0u ? sequence * STEPS + step : step * SEQUENCES + sequence;
			for (index = 0u; index < K3_HIDDEN; ++index)
				hidden[row * K3_HIDDEN + index] = LmFloatToBf16(Value(sequence, step, index, 1u));
			for (index = 0u; index < FUSED; ++index)
				fused_qkvb[row * FUSED + index] = LmFloatToBf16(Value(sequence, step, index, 2u));
			for (index = 0u; index < RANK_QK; ++index)
				decay_logit[row * RANK_QK + index] = LmFloatToBf16(Value(sequence, step, index, 3u));
			for (index = 0u; index < RANK_V; ++index)
				gate[row * RANK_V + index] = LmFloatToBf16(Value(sequence, step, index, 4u));
		}
	memset(state_pool, 0, sizeof(state_pool));
	memset(q_window, 0, sizeof(q_window));
	memset(k_window, 0, sizeof(k_window));
	memset(v_window, 0, sizeof(v_window));
	memset(attention_out, 0, sizeof(attention_out));
}

static int32_t Run(K3LayerBuffers *b, const uint32_t *row_indices)
{
	lm_recorded_gemms.clear();
	b->sequence_row_indices = row_indices;
	return(K3LayerKda<LmBf16Format>(b, ROWS, SEQUENCES, 1u, (uint16_t *)0, 1u, 0));
}

static uint32_t Mismatch(void)
{
	uint32_t sequence, step, index, count = 0u;
	for (sequence = 0u; sequence < SEQUENCES; ++sequence)
		for (step = 0u; step < STEPS; ++step)
			for (index = 0u; index < RANK_V; ++index)
				if ( attention_out[(step * SEQUENCES + sequence) * RANK_V + index] !=
					grouped_out[(sequence * STEPS + step) * RANK_V + index] )
					++count;
	count += memcmp(state_pool, grouped_state, sizeof(state_pool)) != 0 ? 1u : 0u;
	count += memcmp(q_window, grouped_q, sizeof(q_window)) != 0 ? 1u : 0u;
	count += memcmp(k_window, grouped_k, sizeof(k_window)) != 0 ? 1u : 0u;
	count += memcmp(v_window, grouped_v, sizeof(v_window)) != 0 ? 1u : 0u;
	return(count);
}

int main(void)
{
	static K3LayerBuffers b;
	uint32_t index, nonzero = 0u;
	int32_t status;
	for (index = 0u; index < K3_HIDDEN; ++index)
		norm_weight[index] = LmFloatToBf16(1.0f);
	for (index = 0u; index < K3_KDA_V_DIM * K3_KDA_CONV_KERNEL; ++index)
		conv_weight[index] = 0.25f + 0.01f * (float)(index % 7u);
	for (index = 0u; index < K3_KDA_VALUE_DIM; ++index)
		out_norm_weight[index] = 1.0f;
	for (index = 0u; index < K3_KDA_QK_DIM; ++index)
		decay_bias[index] = -3.0f + 0.05f * (float)((index * 37u) % 101u);
	for (index = 0u; index < K3_KDA_HEADS; ++index)
		head_log_scale[index] = -1.5f + 0.03f * (float)index;
	for (index = 0u; index < ROWS; ++index)
		interleaved[(index % STEPS) + (index / STEPS) * STEPS] = (index % STEPS) * SEQUENCES + index / STEPS;
	Bind(&b);
	Fill(1u);
	status = Run(&b, 0);
	if ( status != LM_LAUNCH_OK )
	{
		printf("FAIL grouped run status %d\n", (int)status);
		return 1;
	}
	memcpy(grouped_out, attention_out, sizeof(grouped_out));
	memcpy(grouped_state, state_pool, sizeof(grouped_state));
	memcpy(grouped_q, q_window, sizeof(grouped_q));
	memcpy(grouped_k, k_window, sizeof(grouped_k));
	memcpy(grouped_v, v_window, sizeof(grouped_v));
	for (index = 0u; index < sizeof(grouped_state); ++index)
		nonzero += grouped_state[index] != 0u ? 1u : 0u;
	printf("grouped_state_nonzero %u\n", nonzero);
	Fill(0u);
	status = Run(&b, interleaved);
	printf("interleaved_indexed status %d mismatch %u\n", (int)status, Mismatch());
	Fill(0u);
	status = Run(&b, 0);
	printf("interleaved_unindexed status %d mismatch %u\n", (int)status, Mismatch());
	printf("done\n");
	return 0;
}
