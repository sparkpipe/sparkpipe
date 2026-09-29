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

#define TP_DEGREE 4u
#define RANK_HEADS (K3_KDA_HEADS / TP_DEGREE)
#define STEPS 3u
#define RANK_CONV (RANK_HEADS * K3_KDA_KEY_DIM * K3_KDA_CONV_KERNEL)

static uint16_t hidden[K3_HIDDEN], normed[K3_HIDDEN];
static uint16_t norm_weight[K3_HIDDEN], weight_token[K3_HIDDEN];
static uint16_t fused_qkvb[K3_KDA_QKVB_FUSED_ROWS];
static uint16_t latent[K3_KDA_KEY_DIM];
static uint16_t gate[K3_KDA_V_DIM];
static uint16_t query[K3_KDA_QK_DIM], key[K3_KDA_QK_DIM], value[K3_KDA_V_DIM];
static uint16_t beta_logit[K3_KDA_HEADS];
static uint16_t decay_logit[K3_KDA_QK_DIM];
static uint16_t attention_out[K3_KDA_V_DIM];
static float conv_weight[K3_KDA_V_DIM * K3_KDA_CONV_KERNEL];
static float out_norm_weight[K3_KDA_VALUE_DIM];
static float decay_bias[K3_KDA_QK_DIM], head_log_scale[K3_KDA_HEADS];
static float retention[K3_KDA_QK_DIM], write_gate[K3_KDA_HEADS];
static uint16_t q_window[K3_KDA_QK_DIM * K3_KDA_CONV_KERNEL];
static uint16_t k_window[K3_KDA_QK_DIM * K3_KDA_CONV_KERNEL];
static uint16_t v_window[K3_KDA_V_DIM * K3_KDA_CONV_KERNEL];
static uint8_t state_pool[K3_KDA_STATE_SLOT_BYTES];
static uint32_t state_index[1], dense_offsets[2], dense_tiles[2];

static float full_retention[K3_KDA_QK_DIM];
static uint16_t full_out[K3_KDA_V_DIM];
static uint8_t full_state[K3_KDA_STATE_SLOT_BYTES];

static void Bind(K3LayerBuffers *b, uint32_t sharded, uint32_t rank)
{
	memset(b, 0, sizeof(*b));
	b->hidden_bf16 = hidden; b->normed_bf16 = normed;
	b->attn_norm_weight = norm_weight;
	b->kda_qkv_beta_weight = weight_token;
	b->kda_decay_down_weight = weight_token;
	b->kda_decay_up_weight = weight_token;
	b->kda_gate_weight = weight_token;
	b->kda_out_weight = weight_token;
	b->kda_q_conv_weight = conv_weight + (sharded != 0u ? rank * RANK_CONV : 0u);
	b->kda_k_conv_weight = conv_weight + (sharded != 0u ? rank * RANK_CONV : 0u);
	b->kda_v_conv_weight = conv_weight + (sharded != 0u ? rank * RANK_CONV : 0u);
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
	b->dense_row_offset = dense_offsets;
	b->dense_tile_prefix = dense_tiles;
	b->tp_sharded = sharded;
	b->tp_rank = rank;
	if ( sharded != 0u )
	{
		b->kda_heads_rank = RANK_HEADS;
		b->kda_qkvb_rows = RANK_HEADS * (2u * K3_KDA_KEY_DIM + K3_KDA_VALUE_DIM + 1u);
		b->kda_gate_rows = RANK_HEADS * K3_KDA_VALUE_DIM;
		b->kda_decay_up_rows = RANK_HEADS * K3_KDA_KEY_DIM;
		b->kda_out_input = RANK_HEADS * K3_KDA_VALUE_DIM;
	}
}

static int32_t Run(K3LayerBuffers *b)
{
	uint32_t step;
	int32_t status = LM_LAUNCH_OK;
	memset(state_pool, 0, sizeof(state_pool));
	memset(q_window, 0, sizeof(q_window));
	memset(k_window, 0, sizeof(k_window));
	memset(v_window, 0, sizeof(v_window));
	for (step = 0u; step < STEPS && status == LM_LAUNCH_OK; ++step)
	{
		lm_recorded_gemms.clear();
		status = K3LayerKda<LmBf16Format>(b, 1u, 1u, 1u, (uint16_t *)0, 1u, 0);
	}
	return(status);
}

int main(void)
{
	static K3LayerBuffers b;
	uint32_t index, rank, head, distinct = 0u;
	int32_t status;
	for (index = 0u; index < K3_HIDDEN; ++index)
	{
		norm_weight[index] = LmFloatToBf16(1.0f);
		hidden[index] = LmFloatToBf16(0.01f * (float)(index % 13u) + 0.02f);
	}
	for (index = 0u; index < K3_KDA_V_DIM * K3_KDA_CONV_KERNEL; ++index)
		conv_weight[index] = 0.25f + 0.01f * (float)(index % 7u);
	for (index = 0u; index < K3_KDA_VALUE_DIM; ++index)
		out_norm_weight[index] = 1.0f;
	for (index = 0u; index < K3_KDA_QK_DIM; ++index)
		decay_bias[index] = -3.0f + 0.05f * (float)((index * 37u) % 101u);
	for (head = 0u; head < K3_KDA_HEADS; ++head)
		head_log_scale[head] = -1.5f + 0.03f * (float)head;
	Bind(&b, 0u, 0u);
	status = Run(&b);
	if ( status != LM_LAUNCH_OK )
	{
		printf("FAIL full-width run status %d\n", (int)status);
		return 1;
	}
	memcpy(full_retention, retention, sizeof(full_retention));
	memcpy(full_out, attention_out, sizeof(full_out));
	memcpy(full_state, state_pool, sizeof(full_state));
	for (head = 1u; head < K3_KDA_HEADS; ++head)
		if ( full_retention[head * K3_KDA_KEY_DIM] != full_retention[0] )
			++distinct;
	printf("distinct_heads %u\n", distinct);
	for (rank = 0u; rank < TP_DEGREE; ++rank)
	{
		uint32_t retention_mismatch = 0u, state_mismatch = 0u, out_mismatch = 0u;
		const size_t head_state = (size_t)K3_KDA_KEY_DIM * K3_KDA_VALUE_DIM * sizeof(float);
		const size_t first = (size_t)rank * RANK_HEADS;
		Bind(&b, 1u, rank);
		status = Run(&b);
		if ( status != LM_LAUNCH_OK )
		{
			printf("FAIL rank %u status %d\n", rank, (int)status);
			return 1;
		}
		for (index = 0u; index < RANK_HEADS * K3_KDA_KEY_DIM; ++index)
			if ( retention[index] != full_retention[first * K3_KDA_KEY_DIM + index] )
				++retention_mismatch;
		for (index = 0u; index < RANK_HEADS * K3_KDA_VALUE_DIM; ++index)
			if ( attention_out[index] != full_out[first * K3_KDA_VALUE_DIM + index] )
				++out_mismatch;
		if ( memcmp(state_pool, full_state + first * head_state, RANK_HEADS * head_state) != 0 )
			state_mismatch = 1u;
		printf("rank %u retention_mismatch %u state_mismatch %u out_mismatch %u\n",
			rank, retention_mismatch, state_mismatch, out_mismatch);
	}
	Bind(&b, 1u, TP_DEGREE);
	status = Run(&b);
	printf("rank_past_heads_status %d\n", (int)status);
	printf("done\n");
	return 0;
}
