#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define SPARK_K3_SERVING_INDEXER 0
#include "model-families/k3/source/k3_resident_decode_stage_cuda.cu"
#include "model-families/k3/source/k3_stage.cuh"

#define ODD_TP 16u
#define ODD_HEADS (K3_KDA_HEADS / ODD_TP)
#define ODD_QK (ODD_HEADS * K3_KDA_KEY_DIM)
#define ODD_V (ODD_HEADS * K3_KDA_VALUE_DIM)
#define ODD_QKVB (2u * ODD_QK + ODD_V + ODD_HEADS)
#define ODD_MLA_Q_UP (K3_MLA_Q_DIM / ODD_TP)
#define ODD_SLICE (K3_ROUTED_EXPERT_HIDDEN / ODD_TP)
#define ODD_W1_OUT (2u * K3_EXPERT_INTERMEDIATE / ODD_TP)
#define ODD_CHANNELS (K3_EXPERT_INTERMEDIATE / ODD_TP)
#define ODD_SHARED_W1 (2u * K3_SHARED_INTERMEDIATE / ODD_TP)
#define ODD_SHARED_W2 (K3_SHARED_INTERMEDIATE / ODD_TP)
#define ODD_DENSE_W1 (2u * K3_DENSE_INTERMEDIATE / ODD_TP)
#define ODD_DENSE_W2 (K3_DENSE_INTERMEDIATE / ODD_TP)

#define ODD_CUDA(call) do { cudaError_t e = (call); if ( e != cudaSuccess ) { printf("FAIL cuda %s:%d: %s\n", #call, __LINE__, cudaGetErrorString(e)); exit(1); } } while ( 0 )

static int odd_failures;

static void OddExpect(const char *what, int condition)
{
	if ( condition != 0 )
		return;
	printf("FAIL %s\n", what);
	odd_failures++;
}

extern "C" SparkStatus SparkWeightdLazyPackSlice(const SparkWeightdLazyPack *, uint64_t, uint64_t, const void **)
{
	return SPARK_STATUS_UNSUPPORTED;
}

extern "C" int32_t K3StageSlice(const void *, const void *, void *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, void *)
{
	return LM_LAUNCH_OK;
}

extern "C" int32_t K3StageFold(const void *, const void *, void *, uint32_t, uint32_t, uint32_t, const uint32_t *, const uint32_t *, const uint32_t *, uint32_t, uint32_t, void *)
{
	return LM_LAUNCH_OK;
}

extern "C" int32_t K3StageSlicePair(const void *, const void *, void *, void *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, void *)
{
	return LM_LAUNCH_OK;
}

template<class T>
static T *OddAlloc(uint64_t count)
{
	T *p = 0;
	ODD_CUDA(cudaMalloc((void **)&p, count * sizeof(T) + 256u));
	ODD_CUDA(cudaMemset(p, 0, count * sizeof(T) + 256u));
	return p;
}

__global__ static void OddFillBf16(uint16_t *p, uint64_t n, uint32_t salt, float scale)
{
	for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x)
	{
		uint32_t x = (uint32_t)(i * 2654435761u) ^ (salt * 0x9e3779b9u);
		x ^= x >> 13u; x *= 0x5bd1e995u; x ^= x >> 15u;
		p[i] = LmFloatToBf16(scale * (((float)(x & 0xffffu) - 32768.0f) / 32768.0f));
	}
}

__global__ static void OddFillF32(float *p, uint64_t n, uint32_t salt, float scale)
{
	for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x)
	{
		uint32_t x = (uint32_t)(i * 2654435761u) ^ (salt * 0x9e3779b9u);
		x ^= x >> 13u; x *= 0x5bd1e995u; x ^= x >> 15u;
		p[i] = scale * (((float)(x & 0xffffu) - 32768.0f) / 32768.0f);
	}
}

static uint16_t *OddBf16(uint64_t n, uint32_t salt, float scale = 0.25f)
{
	uint16_t *p = OddAlloc<uint16_t>(n);
	OddFillBf16<<<256, 256>>>(p, n, salt, scale);
	return p;
}

static float *OddF32(uint64_t n, uint32_t salt, float scale = 0.25f)
{
	float *p = OddAlloc<float>(n);
	OddFillF32<<<64, 256>>>(p, n, salt, scale);
	return p;
}

static void CheckAligned16(const char *label, const void *ptr)
{
	if ( ((uintptr_t)ptr & 15u) != 0u )
	{
		printf("FAIL %s not 16-byte aligned (%p)\n", label, ptr);
		odd_failures++;
	}
}

static void CheckPairRowsInvariants(void)
{
	const uint32_t degrees[] = {2u, 4u, 8u, 16u};
	for (uint32_t di = 0u; di < 4u; ++di)
	{
		const uint32_t deg = degrees[di], att = 1024u;
		for (uint32_t rows = 2u * K3_RUNNER_PAIR_MIN_ROWS; rows <= 2048u; ++rows)
		{
			const uint32_t ra = K3RunnerPairRows(rows, deg, att), rb = rows - ra;
			if ( ra == 0u || rb == 0u || ra > att || rb > att || (ra & 7u) != 0u || (deg == 16u && (ra & 15u) != 0u) )
			{
				printf("FAIL K3RunnerPairRows(rows=%u, deg=%u, att=%u) -> %u + %u\n", rows, deg, att, ra, rb);
				odd_failures++;
				return;
			}
		}
	}
	printf("PASS K3RunnerPairRows invariants for all rows 512..2048 across TP2/4/8/16\n");
}

int main(void)
{
	cudaDeviceProp prop;
	ODD_CUDA(cudaGetDeviceProperties(&prop, 0));
	const uint32_t mps = (uint32_t)prop.multiProcessorCount;
	cudaStream_t stream;
	ODD_CUDA(cudaStreamCreate(&stream));

	CheckPairRowsInvariants();

	SparkK3PoolSizing sizing;
	memset(&sizing, 0, sizeof(sizing));
	sizing.first_layer = 0u;
	sizing.layer_count = 4u;
	sizing.kda_layer_count = 3u;
	sizing.mla_layer_count = 1u;

	K3LayerBuffers dims;
	memset(&dims, 0, sizeof(dims));
	dims.tp_sharded = 1u;
	dims.kda_heads_rank = ODD_HEADS;
	dims.mla_heads_rank = ODD_HEADS;
	dims.kda_qkvb_rows = ODD_QKVB;
	dims.kda_gate_rows = ODD_V;
	dims.kda_decay_up_rows = ODD_QK;
	dims.kda_out_input = ODD_V;
	dims.mla_q_up_rows = ODD_MLA_Q_UP;
	dims.mla_gate_rows = ODD_V;
	dims.mla_out_input = ODD_V;
	dims.routed_down_rows = ODD_SLICE;
	dims.routed_up_input = ODD_SLICE;
	dims.expert_w1_output = ODD_W1_OUT;
	dims.expert_w2_input = ODD_CHANNELS;
	dims.shared_w1_rows = ODD_SHARED_W1;
	dims.shared_w2_input = ODD_SHARED_W2;
	dims.dense_gate_up_rows = ODD_DENSE_W1;
	dims.dense_down_input = ODD_DENSE_W2;

	SparkK3ScratchWidths widths;
	K3ScratchWidthsFor(&dims, &widths);

	SparkK3Dispatch dispatch;
	const uint32_t max_rows = 2048u, attention_rows = 1024u;
	const uint32_t pages_per_seq = 64u;
	const int32_t cstat = SparkK3DispatchCreate(&dispatch, &sizing, 1u, max_rows, attention_rows, pages_per_seq,
		K3GlobalKv::kPageBytes / ODD_TP, ODD_TP, 0u, &widths, 0);
	OddExpect("SparkK3DispatchCreate succeeds", cstat == SPARK_K3_DISPATCH_OK);
	if ( cstat != SPARK_K3_DISPATCH_OK )
		return 1;

	uint8_t *kv_pool = OddAlloc<uint8_t>((uint64_t)pages_per_seq * (K3GlobalKv::kPageBytes / ODD_TP));
	uint32_t *page_table = OddAlloc<uint32_t>(pages_per_seq);
	std::vector<uint32_t> pt_h(pages_per_seq);
	for (uint32_t i = 0u; i < pages_per_seq; ++i) pt_h[i] = i;
	ODD_CUDA(cudaMemcpy(page_table, pt_h.data(), pages_per_seq * sizeof(uint32_t), cudaMemcpyHostToDevice));
	OddExpect("SparkK3DispatchAttachKv succeeds",
		SparkK3DispatchAttachKv(&dispatch, kv_pool, (uint64_t)pages_per_seq * (K3GlobalKv::kPageBytes / ODD_TP),
			0, 0u, page_table, pages_per_seq, pages_per_seq, 1u) == SPARK_K3_DISPATCH_OK);

	K3LayerBuffers *a = dispatch.buffers_host;
	a->tp_sharded = 1u;
	a->tp_rank = 0u;
	a->kda_heads_rank = ODD_HEADS;
	a->mla_heads_rank = ODD_HEADS;
	a->kda_qkvb_rows = ODD_QKVB;
	a->kda_gate_rows = ODD_V;
	a->kda_decay_up_rows = ODD_QK;
	a->kda_out_input = ODD_V;
	a->mla_q_up_rows = ODD_MLA_Q_UP;
	a->mla_gate_rows = ODD_V;
	a->mla_out_input = ODD_V;
	a->routed_down_rows = ODD_SLICE;
	a->routed_up_input = ODD_SLICE;
	a->expert_w1_output = ODD_W1_OUT;
	a->expert_w2_input = ODD_CHANNELS;
	a->shared_w1_rows = ODD_SHARED_W1;
	a->shared_w2_input = ODD_SHARED_W2;
	a->dense_gate_up_rows = ODD_DENSE_W1;
	a->dense_down_input = ODD_DENSE_W2;
	a->expert_interleave = 1u;
	a->expert_tile_k = 32u;

	a->attn_norm_weight = OddBf16(K3_HIDDEN, 1, 1.0f);
	a->mlp_norm_weight = OddBf16(K3_HIDDEN, 2, 1.0f);
	a->kda_qkv_beta_weight = OddBf16((uint64_t)ODD_QKVB * K3_HIDDEN, 3);
	a->kda_decay_down_weight = OddBf16((uint64_t)K3_KDA_KEY_DIM * K3_HIDDEN, 4);
	a->kda_q_conv_weight = OddF32((uint64_t)ODD_QK * K3_KDA_CONV_KERNEL, 5);
	a->kda_k_conv_weight = OddF32((uint64_t)ODD_QK * K3_KDA_CONV_KERNEL, 6);
	a->kda_v_conv_weight = OddF32((uint64_t)ODD_V * K3_KDA_CONV_KERNEL, 7);
	a->kda_decay_up_weight = OddBf16((uint64_t)ODD_QK * K3_KDA_KEY_DIM, 8);
	a->kda_decay_bias = OddF32(K3_KDA_QK_DIM, 9);
	a->kda_head_log_scale = OddF32(K3_KDA_HEADS, 10);
	a->kda_gate_weight = OddBf16((uint64_t)ODD_V * K3_HIDDEN, 11);
	a->kda_out_norm_weight = OddF32(K3_KDA_VALUE_DIM, 12, 1.0f);
	a->kda_out_weight = OddBf16((uint64_t)K3_HIDDEN * ODD_V, 13);
	a->mla_q_down_weight = OddBf16((uint64_t)K3_Q_LORA_RANK * K3_HIDDEN, 14);
	a->mla_q_norm_weight = OddBf16(K3_Q_LORA_RANK, 15, 1.0f);
	a->mla_q_up_weight = OddBf16((uint64_t)ODD_MLA_Q_UP * K3_Q_LORA_RANK, 16);
	a->mla_kv_a_weight = OddBf16((uint64_t)K3_MLA_KV_A_DIM * K3_HIDDEN, 17);
	a->mla_kv_a_norm_weight = OddBf16(K3_KV_LORA_RANK, 18, 1.0f);
	a->dense_gate_up_weight = OddBf16((uint64_t)ODD_DENSE_W1 * K3_HIDDEN, 19);
	a->dense_down_weight = OddBf16((uint64_t)K3_HIDDEN * ODD_DENSE_W2, 20);
	a->router_weight = OddBf16((uint64_t)K3_EXPERTS * K3_HIDDEN, 21);
	a->routed_down_weight = OddBf16((uint64_t)ODD_SLICE * K3_HIDDEN, 22);
	a->routed_up_weight = OddBf16((uint64_t)K3_HIDDEN * ODD_SLICE, 23);
	a->routed_norm_weight = OddBf16(K3_ROUTED_EXPERT_HIDDEN, 24, 1.0f);
	a->shared_w1_weight = OddBf16((uint64_t)ODD_SHARED_W1 * K3_HIDDEN, 25);
	a->shared_w2_weight = OddBf16((uint64_t)K3_HIDDEN * ODD_SHARED_W2, 26);

	uint16_t *hidden_init = OddBf16((uint64_t)max_rows * K3_HIDDEN, 40);
	uint32_t *seq_of_row = OddAlloc<uint32_t>(max_rows);
	uint32_t *positions = OddAlloc<uint32_t>(max_rows);
	uint32_t *state_idx = OddAlloc<uint32_t>(1u);
	std::vector<uint32_t> pos_h(max_rows);
	for (uint32_t i = 0u; i < max_rows; ++i) pos_h[i] = i;
	ODD_CUDA(cudaMemcpy(positions, pos_h.data(), max_rows * sizeof(uint32_t), cudaMemcpyHostToDevice));
	a->sequence_of_row = seq_of_row;
	a->positions = positions;
	a->kda_state_index = state_idx;
	a->kda_state_pool = dispatch.kda_state_pool;
	a->kda_q_window = dispatch.kda_q_window_pool;
	a->kda_k_window = dispatch.kda_k_window_pool;
	a->kda_v_window = dispatch.kda_v_window_pool;
	a->cache = dispatch.mla_cache[0];
	a->cache_shard.pages = a->cache;
	a->cache_shard.shard = a->kv_shard;

	struct Case { uint32_t rows; uint32_t rows_a; const char *name; };
	const Case cases[] = {
		{1703u, 851u, "live-crash raw odd split 1703 = 851 + 852"},
		{1703u, K3RunnerPairRows(1703u, ODD_TP, attention_rows), "aligned split 1703"},
		{1705u, 852u, "odd total 1705 = 852 + 853"},
		{1705u, K3RunnerPairRows(1705u, ODD_TP, attention_rows), "aligned split 1705"},
		{513u, 257u, "raw odd split 513 = 257 + 256"},
		{65u, 33u, "small odd split 65 = 33 + 32"},
		{33u, 17u, "small odd split 33 = 17 + 16"},
	};

	for (uint32_t ci = 0u; ci < sizeof(cases) / sizeof(cases[0]); ++ci)
	{
		const uint32_t rows = cases[ci].rows, ra = cases[ci].rows_a, rb = rows - ra;
		K3LayerBuffers b;
		ODD_CUDA(cudaMemcpy(a->hidden_bf16, hidden_init, (uint64_t)rows * K3_HIDDEN * 2u, cudaMemcpyDeviceToDevice));
		ODD_CUDA(cudaMemset(a->kda_state_pool, 0, K3_KDA_RANK_STATE_SLOT_BYTES(ODD_HEADS, 0u)));
		ODD_CUDA(cudaMemset(a->kda_q_window, 0, (uint64_t)ODD_QK * K3_KDA_CONV_KERNEL * 2u));
		ODD_CUDA(cudaMemset(a->kda_k_window, 0, (uint64_t)ODD_QK * K3_KDA_CONV_KERNEL * 2u));
		ODD_CUDA(cudaMemset(a->kda_v_window, 0, (uint64_t)ODD_V * K3_KDA_CONV_KERNEL * 2u));
		LM_LAUNCH((K3DispatchPairKernel), (rows + 255u) / 256u, 256u, 0, stream, dispatch.pair_arrays, ra, rb);
		K3DispatchPairView(&dispatch, a, &b, ra);
		a->sequence_row_begin = dispatch.pair_arrays;
		a->sequence_row_indices = dispatch.pair_arrays + K3_DISPATCH_PAIR_IDENTITY;
		a->dense_row_offset = dispatch.pair_arrays + 4u;
		a->dense_tile_prefix = dispatch.pair_arrays + 8u;

		CheckAligned16("b.fused_qkvb_bf16", b.fused_qkvb_bf16);
		CheckAligned16("b.query_bf16", b.query_bf16);
		CheckAligned16("b.key_bf16", b.key_bf16);
		CheckAligned16("b.value_bf16", b.value_bf16);
		CheckAligned16("b.gate_bf16", b.gate_bf16);
		CheckAligned16("b.decay_logit_bf16", b.decay_logit_bf16);
		CheckAligned16("b.latent_bf16", b.latent_bf16);
		CheckAligned16("b.kv_slot_bf16", b.kv_slot_bf16);
		CheckAligned16("b.attention_out_bf16", b.attention_out_bf16);
		CheckAligned16("b.gate_up_bf16", b.gate_up_bf16);
		CheckAligned16("b.intermediate_bf16", b.intermediate_bf16);
		CheckAligned16("b.shared_mid_bf16", b.shared_mid_bf16);

		int32_t s = K3LayerKda<LmMxfp4>(a, ra, 1u, 1u, (uint16_t *)0, mps, stream);
		if ( s == LM_LAUNCH_OK )
			s = K3LayerKda<LmMxfp4>(&b, rb, 1u, 1u, (uint16_t *)0, mps, stream);
		if ( s == LM_LAUNCH_OK )
			s = K3LayerMlaQuery<LmMxfp4, K3GlobalKv>(a, ra, mps, stream);
		if ( s == LM_LAUNCH_OK )
			s = K3LayerMlaQuery<LmMxfp4, K3GlobalKv>(&b, rb, mps, stream);
		if ( s == LM_LAUNCH_OK )
			s = K3LayerDenseMlp<LmMxfp4>(a, ra, mps, stream);
		if ( s == LM_LAUNCH_OK )
			s = K3LayerDenseMlp<LmMxfp4>(&b, rb, mps, stream);
		if ( s == LM_LAUNCH_OK )
			s = K3LayerLatentMoe<LmMxfp4>(a, ra, ra * K3_TOP_K, mps, stream, 0u);
		if ( s == LM_LAUNCH_OK )
			s = K3LayerLatentMoe<LmMxfp4>(&b, rb, rb * K3_TOP_K, mps, stream, 0u);
		if ( s == LM_LAUNCH_OK )
			s = K3LayerSharedUp<LmMxfp4>(a, ra, a->shared_mid_bf16, mps, stream);
		if ( s == LM_LAUNCH_OK )
			s = K3LayerSharedUp<LmMxfp4>(&b, rb, b.shared_mid_bf16, mps, stream);
		if ( s == LM_LAUNCH_OK )
			s = K3LayerSharedDown<LmMxfp4>(a, ra, a->shared_mid_bf16, mps, stream);
		if ( s == LM_LAUNCH_OK )
			s = K3LayerSharedDown<LmMxfp4>(&b, rb, b.shared_mid_bf16, mps, stream);
		if ( s == LM_LAUNCH_OK )
			s = K3LayerLatentMoe<LmMxfp4>(a, ra, ra * K3_TOP_K, mps, stream, 2u);
		if ( s == LM_LAUNCH_OK )
			s = K3LayerLatentMoe<LmMxfp4>(&b, rb, rb * K3_TOP_K, mps, stream, 2u);
		ODD_CUDA(cudaStreamSynchronize(stream));
		OddExpect(cases[ci].name, s == LM_LAUNCH_OK);
		if ( s == LM_LAUNCH_OK )
			printf("PASS %s (%u + %u)\n", cases[ci].name, ra, rb);
	}

	for (uint32_t rows : {65u, 513u, 1703u, 1705u})
	{
		const uint32_t ra = rows >= 2u * K3_RUNNER_PAIR_MIN_ROWS ? K3RunnerPairRows(rows, ODD_TP, attention_rows) : (rows / 2u + 1u), rb = rows - ra;
		K3LayerBuffers b;
		uint32_t *single_rb = OddAlloc<uint32_t>(2u);
		uint32_t hrb[2] = {0u, rows};
		ODD_CUDA(cudaMemcpy(single_rb, hrb, sizeof(hrb), cudaMemcpyHostToDevice));

		ODD_CUDA(cudaMemcpy(a->hidden_bf16, hidden_init, (uint64_t)rows * K3_HIDDEN * 2u, cudaMemcpyDeviceToDevice));
		ODD_CUDA(cudaMemset(a->kda_state_pool, 0, K3_KDA_RANK_STATE_SLOT_BYTES(ODD_HEADS, 0u)));
		ODD_CUDA(cudaMemset(a->kda_q_window, 0, (uint64_t)ODD_QK * K3_KDA_CONV_KERNEL * 2u));
		ODD_CUDA(cudaMemset(a->kda_k_window, 0, (uint64_t)ODD_QK * K3_KDA_CONV_KERNEL * 2u));
		ODD_CUDA(cudaMemset(a->kda_v_window, 0, (uint64_t)ODD_V * K3_KDA_CONV_KERNEL * 2u));
		a->sequence_row_begin = single_rb;
		a->sequence_row_indices = 0;
		a->dense_row_offset = single_rb;
		OddExpect("single-lane odd KDA succeeds", K3LayerKda<LmMxfp4>(a, rows, 1u, 1u, (uint16_t *)0, mps, stream) == LM_LAUNCH_OK);
		ODD_CUDA(cudaStreamSynchronize(stream));
		std::vector<uint16_t> ref((size_t)rows * K3_HIDDEN);
		ODD_CUDA(cudaMemcpy(ref.data(), a->hidden_bf16, ref.size() * 2u, cudaMemcpyDeviceToHost));

		ODD_CUDA(cudaMemcpy(a->hidden_bf16, hidden_init, (uint64_t)rows * K3_HIDDEN * 2u, cudaMemcpyDeviceToDevice));
		ODD_CUDA(cudaMemset(a->kda_state_pool, 0, K3_KDA_RANK_STATE_SLOT_BYTES(ODD_HEADS, 0u)));
		ODD_CUDA(cudaMemset(a->kda_q_window, 0, (uint64_t)ODD_QK * K3_KDA_CONV_KERNEL * 2u));
		ODD_CUDA(cudaMemset(a->kda_k_window, 0, (uint64_t)ODD_QK * K3_KDA_CONV_KERNEL * 2u));
		ODD_CUDA(cudaMemset(a->kda_v_window, 0, (uint64_t)ODD_V * K3_KDA_CONV_KERNEL * 2u));
		LM_LAUNCH((K3DispatchPairKernel), (rows + 255u) / 256u, 256u, 0, stream, dispatch.pair_arrays, ra, rb);
		K3DispatchPairView(&dispatch, a, &b, ra);
		a->sequence_row_begin = dispatch.pair_arrays;
		a->sequence_row_indices = dispatch.pair_arrays + K3_DISPATCH_PAIR_IDENTITY;
		a->dense_row_offset = dispatch.pair_arrays + 4u;
		a->dense_tile_prefix = dispatch.pair_arrays + 8u;
		OddExpect("paired lane A odd KDA succeeds", K3LayerKda<LmMxfp4>(a, ra, 1u, 1u, (uint16_t *)0, mps, stream) == LM_LAUNCH_OK);
		OddExpect("paired lane B odd KDA succeeds", K3LayerKda<LmMxfp4>(&b, rb, 1u, 1u, (uint16_t *)0, mps, stream) == LM_LAUNCH_OK);
		ODD_CUDA(cudaStreamSynchronize(stream));
		std::vector<uint16_t> got((size_t)rows * K3_HIDDEN);
		ODD_CUDA(cudaMemcpy(got.data(), a->hidden_bf16, got.size() * 2u, cudaMemcpyDeviceToHost));

		uint32_t diffs = 0u;
		for (size_t i = 0; i < ref.size(); ++i)
			if ( ref[i] != got[i] )
				diffs++;
		printf("parity rows=%u (%u + %u): diffs=%u / %zu\n", rows, ra, rb, diffs, ref.size());
		OddExpect("paired odd-row KDA matches single-lane KDA bit-exact", diffs == 0u);
		ODD_CUDA(cudaFree(single_rb));
	}

	{
		const uint32_t rows = 65u, rank = 3u;
		uint16_t *lat_in = OddBf16((uint64_t)rows * K3_ROUTED_EXPERT_HIDDEN, 70);
		uint16_t *lat_copy = OddAlloc<uint16_t>((uint64_t)rows * K3_ROUTED_EXPERT_HIDDEN);
		uint16_t *w_norm = OddBf16(K3_ROUTED_EXPERT_HIDDEN, 71, 1.0f);
		uint16_t *out_ref = OddAlloc<uint16_t>((uint64_t)rows * ODD_SLICE);
		uint16_t *out_opt = OddAlloc<uint16_t>((uint64_t)rows * ODD_SLICE);
		ODD_CUDA(cudaMemcpy(lat_copy, lat_in, (uint64_t)rows * K3_ROUTED_EXPERT_HIDDEN * 2u, cudaMemcpyDeviceToDevice));
		LM_LAUNCH((LmFusedResidualRmsNormKernel<K3_LAYER_THREADS,uint16_t,true>), rows, K3_LAYER_THREADS, (K3_ROUTED_EXPERT_HIDDEN + 8u) * sizeof(float), stream,
			lat_copy, 0, w_norm, 0, lat_copy, K3_ROUTED_EXPERT_HIDDEN, K3_ROUTED_EXPERT_HIDDEN, K3_RMS_EPSILON);
		LM_LAUNCH((LmSliceColumnsKernel<K3_LAYER_THREADS>), dim3(LmVecBlocks<K3_LAYER_THREADS>(ODD_SLICE), rows), K3_LAYER_THREADS, 0, stream,
			lat_copy, out_ref, K3_ROUTED_EXPERT_HIDDEN, rank * ODD_SLICE, ODD_SLICE);
		ODD_CUDA(K3Skl2Launch(K3SKL2_k3_rms_norm_slice_256, rows, K3_LAYER_THREADS, 0u, stream,
			lat_in, w_norm, out_opt, (uint32_t)K3_ROUTED_EXPERT_HIDDEN, rank * ODD_SLICE, ODD_SLICE, (float)K3_RMS_EPSILON));
		ODD_CUDA(cudaStreamSynchronize(stream));
		std::vector<uint16_t> href((size_t)rows * ODD_SLICE), hopt((size_t)rows * ODD_SLICE);
		ODD_CUDA(cudaMemcpy(href.data(), out_ref, href.size() * 2u, cudaMemcpyDeviceToHost));
		ODD_CUDA(cudaMemcpy(hopt.data(), out_opt, hopt.size() * 2u, cudaMemcpyDeviceToHost));
		uint32_t diffs = 0u;
		for (size_t i = 0; i < href.size(); ++i)
			if ( href[i] != hopt[i] ) diffs++;
		printf("parity K3RmsNormSliceKernel rows=%u: diffs=%u / %zu\n", rows, diffs, href.size());
		OddExpect("K3RmsNormSliceKernel matches unfused bit-exact", diffs == 0u);
		ODD_CUDA(cudaFree(lat_in)); ODD_CUDA(cudaFree(lat_copy)); ODD_CUDA(cudaFree(w_norm));
		ODD_CUDA(cudaFree(out_ref)); ODD_CUDA(cudaFree(out_opt));
	}

	SparkK3DispatchDestroy(&dispatch);
	ODD_CUDA(cudaStreamDestroy(stream));
	if ( odd_failures != 0 )
	{
		printf("FAIL k3 odd-row prefill: %d failures\n", odd_failures);
		return 1;
	}
	printf("PASS k3 odd-row prefill: all odd row splits (1703, 1705, 513, 65, 33) align and match bit-exact\n");
	return 0;
}
