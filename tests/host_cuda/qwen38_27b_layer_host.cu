// Run a whole Qwen 3.6 layer - attention, recurrent and MLP - on a CPU and
// check where its data went.
//
// gqa_host proves the two GQA kernels compute the right numbers. This proves
// the DRIVER wires them: that the fused projection reaches the convolution,
// the convolved row reaches the split, the split reaches the cache slot in the
// contracted [K|V] layout, the delta rule's 48 value-head states all advance
// inside the pool, and nothing writes past either pool. Those are the defects
// this driver actually had - a cache stored from a buffer nothing wrote, an
// attention kernel of the wrong class, a state slot half the kernel's stride,
// and a convolution over one of the three tensors the reference convolves -
// and every one of them passes a per-kernel test.
//
// The GEMM is the recorder reached through the include path: it writes a
// known constant per call, so every downstream tensor holds a computable
// value, and it logs shapes so the projections' widths are checked rather
// than assumed. Every other kernel is the one that ships.
//
// Two sequences everywhere: one sequence makes every slot index the identity
// and cannot see cross-sequence addressing.

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

// One thread, so every kernel the layer launches runs the whole of its work
// loop in order - the schedule the shim guarantees the house loop shape is
// valid under. layer.cuh's default is 256 for the device.
#define QWEN38_27B_LAYER_THREADS 1u
#include "inference/llms/qwen_3_6/layer.cuh"

#define ROWS 2u
#define CANARY_BYTES 65536u
#define CANARY_BYTE 0xAA

static uint16_t hidden[ROWS * QWEN38_27B_HIDDEN];
static uint16_t residual[ROWS * QWEN38_27B_HIDDEN];
static uint16_t normed[ROWS * QWEN38_27B_HIDDEN];
static uint16_t fused_qkv[ROWS * QWEN38_27B_ATTN_QKV_DIM];
static uint16_t query_gate[ROWS * QWEN38_27B_ATTN_QG_DIM];
static uint16_t query[ROWS * QWEN38_27B_Q_DIM];
static uint16_t attn_gate[ROWS * QWEN38_27B_Q_DIM];
static uint16_t key[ROWS * QWEN38_27B_GDN_QK_DIM];
static uint16_t value[ROWS * QWEN38_27B_GDN_V_DIM];
static uint16_t expanded_q[ROWS * QWEN38_27B_GDN_V_DIM];
static uint16_t expanded_k[ROWS * QWEN38_27B_GDN_V_DIM];
static uint16_t attention_out[ROWS * QWEN38_27B_Q_DIM];
static uint16_t gate_up[ROWS * QWEN38_27B_FFN_INTERMEDIATE * 2u];
static uint16_t intermediate[ROWS * QWEN38_27B_FFN_INTERMEDIATE];
static uint16_t norm_weight[QWEN38_27B_HIDDEN];
static uint16_t conv_weight[QWEN38_27B_GDN_QKV_DIM * QWEN38_27B_GDN_CONV_KERNEL];
static uint16_t beta_logit[ROWS * QWEN38_27B_GDN_VALUE_HEADS];
static uint16_t decay_logit[ROWS * QWEN38_27B_GDN_VALUE_HEADS];
static float a_log[QWEN38_27B_GDN_VALUE_HEADS];
static float dt_bias[QWEN38_27B_GDN_VALUE_HEADS];
static float forget_gate[ROWS * QWEN38_27B_GDN_VALUE_HEADS * QWEN38_27B_GDN_KEY_DIM];
static float write_gate[ROWS * QWEN38_27B_GDN_VALUE_HEADS];
static uint32_t state_index[ROWS];
static uint32_t sequence_of_row[ROWS];
static uint32_t context_length[ROWS];
static uint32_t positions[ROWS];
static uint32_t dense_offsets[ROWS + 1u];
static uint32_t dense_tiles[2];
static uint8_t state_pool[ROWS * QWEN38_27B_GDN_STATE_BYTES + CANARY_BYTES];
static uint16_t conv_window[ROWS * QWEN38_27B_GDN_QKV_DIM * QWEN38_27B_GDN_CONV_KERNEL
	+ CANARY_BYTES / 2u];
static uint8_t kv_pool[2u * Qwen38_27bFullKv::kPageBytes];
static uint32_t page_table[ROWS];
static LmKvAccessError kv_access_error;

static uint32_t CanaryIntact(const uint8_t *canary)
{
	uint32_t index;
	for (index = 0u; index < CANARY_BYTES; ++index)
		if ( canary[index] != CANARY_BYTE )
			return(0u);
	return(1u);
}

int main(void)
{
	static Qwen38_27bLayerBuffers b;
	uint32_t index,head;
	float expected,maxdiff;
	memset(&b, 0, sizeof(b));
	LmKvAccessErrorReset(&kv_access_error);
	for (index = 0u; index < QWEN38_27B_HIDDEN; ++index)
		norm_weight[index] = LmFloatToBf16(1.0f);
	for (index = 0u; index < ROWS * QWEN38_27B_HIDDEN; ++index)
		hidden[index] = LmFloatToBf16(0.01f * (float)(index % 17u));
	for (index = 0u; index < QWEN38_27B_GDN_QKV_DIM * QWEN38_27B_GDN_CONV_KERNEL; ++index)
		conv_weight[index] = LmFloatToBf16(0.25f);
	// The gate mapping's per-head tensors: log-scale zero and bias zero, so
	// the retention is exp(-softplus(logit)) and beta sigmoid(logit) of
	// whatever the GEMM recorder wrote - values python computes closed-form.
	for (index = 0u; index < QWEN38_27B_GDN_VALUE_HEADS; ++index)
	{
		a_log[index] = 0.0f;
		dt_bias[index] = 0.0f;
	}
	memset(state_pool, 0, ROWS * QWEN38_27B_GDN_STATE_BYTES);
	memset(state_pool + ROWS * QWEN38_27B_GDN_STATE_BYTES, CANARY_BYTE, CANARY_BYTES);
	memset(conv_window, 0,
		ROWS * QWEN38_27B_GDN_QKV_DIM * QWEN38_27B_GDN_CONV_KERNEL * sizeof(uint16_t));
	memset(conv_window + ROWS * QWEN38_27B_GDN_QKV_DIM * QWEN38_27B_GDN_CONV_KERNEL,
		CANARY_BYTE, CANARY_BYTES);
	// A poisoned cache: bf16 0xFFFF is NaN, so a store that does not happen is
	// visible in the slot check rather than hiding behind zeroed memory.
	memset(kv_pool, 0xff, sizeof(kv_pool));
	state_index[0] = 0u; state_index[1] = 1u;
	sequence_of_row[0] = 0u; sequence_of_row[1] = 1u;
	context_length[0] = 1u; context_length[1] = 1u;
	// Position zero: rope is the identity there, so the slot check compares
	// the projection's values rather than their rotation.
	positions[0] = 0u; positions[1] = 0u;
	// The pages are swapped, so addressing by sequence rather than by the
	// table stores into the other sequence's slot.
	page_table[0] = 1u; page_table[1] = 0u;
	dense_offsets[0] = 0u; dense_offsets[1] = ROWS; dense_offsets[2] = ROWS * 2u;

	b.hidden_bf16 = hidden; b.residual_bf16 = residual;
	b.normed_bf16 = normed; b.fused_qkv_bf16 = fused_qkv;
	b.query_gate_bf16 = query_gate; b.query_bf16 = query;
	b.attn_gate_bf16 = attn_gate;
	b.key_bf16 = key; b.value_bf16 = value;
	b.gdn_query_expanded_bf16 = expanded_q;
	b.gdn_key_expanded_bf16 = expanded_k;
	b.attention_out_bf16 = attention_out;
	b.gate_up_bf16 = gate_up; b.intermediate_bf16 = intermediate;
	b.attn_norm_weight = norm_weight; b.mlp_norm_weight = norm_weight;
	b.gdn_conv_weight = conv_weight;
	b.gdn_beta_weight = conv_weight; b.gdn_decay_weight = conv_weight;
	b.gdn_a_log = a_log; b.gdn_dt_bias = dt_bias;
	b.gdn_beta_logit_bf16 = beta_logit; b.gdn_decay_logit_bf16 = decay_logit;
	b.gdn_state_pool = state_pool; b.gdn_conv_window = conv_window;
	b.gdn_state_index = state_index;
	b.gdn_forget_gate = forget_gate; b.gdn_write_gate = write_gate;
	b.cache.pool = kv_pool; b.cache.page_table = page_table;
	b.cache.page_table_stride = 1u; b.cache.sequence_count = ROWS;
	b.cache.pool_page_count = 2u; b.cache.access_error = &kv_access_error;
	b.sequence_of_row = sequence_of_row;
	b.context_length = context_length; b.positions = positions;
	b.dense_row_offset = dense_offsets; b.dense_tile_prefix = dense_tiles;

	// -- full attention layer -------------------------------------------------
	lm_recorded_gemms.clear();
	if ( Qwen38_27bLayerAttention<LmBf16Format,Qwen38_27bFullKv>(&b,ROWS,1u,1u,0) != LM_LAUNCH_OK )
	{
		printf("attn_layer_status FAIL\n");
		return(1);
	}
	printf("attn_gemms %u\n", (unsigned)lm_recorded_gemms.size());
	for (index = 0u; index < lm_recorded_gemms.size(); ++index)
		printf("attn_gemm in %u out %u\n",
			lm_recorded_gemms[index].input_dimension,
			lm_recorded_gemms[index].output_dimension);
	// The slot of sequence 0, position 0: K then V, the projection's 0.125
	// throughout, anything else poisoned or a layout slip.
	expected = LmBf16ToFloat(LmFloatToBf16(0.125f));
	{
		const uint16_t *slot = (const uint16_t *)(kv_pool
			+ ((uint64_t)page_table[0] * Qwen38_27bFullKv::kPageBytes));
		maxdiff = 0.0f;
		for (index = 0u; index < QWEN38_27B_KV_HEADS * QWEN38_27B_HEAD_DIM * 2u; ++index)
		{
			float diff = LmBf16ToFloat(slot[index]) - expected;
			if ( diff < 0.0f ) diff = -diff;
			if ( diff > maxdiff ) maxdiff = diff;
		}
		printf("slot_kv_maxdiff %.9g\n", (double)maxdiff);
	}
	// The layer's own artifacts, attended again directly: the output gemm
	// recorder overwrote attention_out, so the attention result is recomputed
	// from the cache the layer filled and the query the split left behind.
	{
		static uint16_t direct_out[ROWS * QWEN38_27B_Q_DIM];
		LM_HOST_LAUNCH(dim3(ROWS,QWEN38_27B_ATTN_HEADS),
			(LmGqaAttentionDecodeKernel<Qwen38_27bFullKv,1u,QWEN38_27B_KV_HEADS,QWEN38_27B_HEAD_DIM,QWEN38_27B_HEAD_DIM>(
				query, b.cache, sequence_of_row, context_length,
				0, 0u, QWEN38_27B_ATTN_HEADS, QWEN38_27B_QK_SCALE, direct_out, 0)));
		maxdiff = 0.0f;
		for (index = 0u; index < ROWS * QWEN38_27B_Q_DIM; ++index)
		{
			float diff = LmBf16ToFloat(direct_out[index]) - expected;
			if ( diff < 0.0f ) diff = -diff;
			if ( diff > maxdiff ) maxdiff = diff;
		}
		printf("attn_maxdiff %.9g\n", (double)maxdiff);
	}
	// The gate half of the fused query projection, de-interleaved: the GEMM
	// recorder's constant throughout, anything else a split-layout slip.
	printf("attn_gate_c0 %.9g\n", (double)LmBf16ToFloat(attn_gate[0]));
	printf("attn_query_c0 %.9g\n", (double)LmBf16ToFloat(query[0]));

	// -- recurrent layer ------------------------------------------------------
	lm_recorded_gemms.clear();
	if ( Qwen38_27bLayerLinear<LmBf16Format>(&b,ROWS,1u,0) != LM_LAUNCH_OK )
	{
		printf("gdn_layer_status FAIL\n");
		return(1);
	}
	printf("gdn_gemms %u\n", (unsigned)lm_recorded_gemms.size());
	for (index = 0u; index < lm_recorded_gemms.size(); ++index)
		printf("gdn_gemm in %u out %u\n",
			lm_recorded_gemms[index].input_dimension,
			lm_recorded_gemms[index].output_dimension);
	// The convolved key as the reference's input: python computes the state
	// from exactly this value, so swish and bf16 rounding appear once.
	printf("conv_c0 %.9g\n", (double)LmBf16ToFloat(key[0]));
	// The produced gates, sequence 0 head 0: beta is sigmoid of the beta
	// GEMM's recorded output, the retention exp(-softplus(decay logit)) with
	// the harness's zeroed A_log and dt_bias. Python holds the closed forms.
	printf("gdn_beta %.9g\n", (double)write_gate[0]);
	printf("gdn_retention %.9g\n", (double)forget_gate[0]);
	// Head 0 and head 47 of sequence 0's state: the first and last of the 48
	// value-head slices. A 16-slice recurrence leaves head 47 at zero.
	{
		const float *state = (const float *)state_pool;
		for (head = 0u; head < QWEN38_27B_GDN_VALUE_HEADS; head += (QWEN38_27B_GDN_VALUE_HEADS - 1u))
			for (index = 0u; index < 8u; ++index)
				printf("state_h%u %.9g\n", head,
					(double)state[(head * QWEN38_27B_GDN_KEY_DIM * QWEN38_27B_GDN_VALUE_DIM) + index]);
	}
	// The committed window: all 10240 channels of both sequences must hold
	// the admitted token at the last tap. A conv over the key's 2048 channels
	// leaves the value channels' window at zero.
	{
		float lowest = 1e30f,highest = -1e30f;
		for (index = 0u;
			index < ROWS * QWEN38_27B_GDN_QKV_DIM * QWEN38_27B_GDN_CONV_KERNEL;
			index += QWEN38_27B_GDN_CONV_KERNEL)
		{
			float tap = LmBf16ToFloat(conv_window[index + (QWEN38_27B_GDN_CONV_KERNEL - 1u)]);
			if ( tap < lowest ) lowest = tap;
			if ( tap > highest ) highest = tap;
		}
		printf("window_tap3_min %.9g\nwindow_tap3_max %.9g\n",
			(double)lowest, (double)highest);
	}
	printf("canary_state %u\n", CanaryIntact(state_pool + ROWS * QWEN38_27B_GDN_STATE_BYTES));
	printf("canary_window %u\n", CanaryIntact(
		(uint8_t *)(conv_window + ROWS * QWEN38_27B_GDN_QKV_DIM * QWEN38_27B_GDN_CONV_KERNEL)));

	// -- head expansion, with values that discriminate ------------------------
	// Uniform projections cannot see a wrong group mapping: every head is the
	// same number. Give head h the value h + 1 and check each expanded head
	// against source head h / 3.
	{
		uint32_t mismatch = 0u,h,g;
		for (h = 0u; h < QWEN38_27B_GDN_KEY_HEADS; ++h)
			for (index = 0u; index < QWEN38_27B_GDN_KEY_DIM; ++index)
				key[(h * QWEN38_27B_GDN_KEY_DIM) + index] = LmFloatToBf16((float)(h + 1u));
		LM_HOST_LAUNCH(dim3(1u),
			(LmExpandHeadsKernel<1u>(key, expanded_k,
				QWEN38_27B_GDN_KEY_HEADS, QWEN38_27B_GDN_KEY_DIM,
				QWEN38_27B_GDN_VALUE_PER_KEY, 1u)));
		for (h = 0u; h < QWEN38_27B_GDN_VALUE_HEADS; ++h)
		{
			g = h / QWEN38_27B_GDN_VALUE_PER_KEY;
			for (index = 0u; index < QWEN38_27B_GDN_KEY_DIM; ++index)
				if ( LmBf16ToFloat(expanded_k[(h * QWEN38_27B_GDN_KEY_DIM) + index])
					!= (float)(g + 1u) )
					++mismatch;
		}
		printf("expand_mismatch %u\n", mismatch);
	}

	// -- dense MLP --------------------------------------------------------------
	lm_recorded_gemms.clear();
	if ( Qwen38_27bLayerDenseMlp<LmBf16Format>(&b,ROWS,1u,0) != LM_LAUNCH_OK )
	{
		printf("mlp_layer_status FAIL\n");
		return(1);
	}
	printf("mlp_gemms %u\n", (unsigned)lm_recorded_gemms.size());
	for (index = 0u; index < lm_recorded_gemms.size(); ++index)
		printf("mlp_gemm in %u out %u\n",
			lm_recorded_gemms[index].input_dimension,
			lm_recorded_gemms[index].output_dimension);
	{
		float peak = 0.0f;
		for (index = 0u; index < ROWS * QWEN38_27B_FFN_INTERMEDIATE; ++index)
		{
			float v = LmBf16ToFloat(intermediate[index]);
			if ( v < 0.0f ) v = -v;
			if ( v > peak ) peak = v;
		}
		printf("mlp_intermediate_max %.9g\n", (double)peak);
	}
	return(0);
}
