#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "sparkpipe/spark_dsv41_flash_stage_model.h"
#include "dsv41_flash_stage.cuh"

#define CHECK(c) do { cudaError_t e = (c); if ( e != cudaSuccess ) { fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while ( 0 )
#define REQ(c, m) do { if ( !(c) ) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, m); exit(1); } } while ( 0 )

static float Bf16ToF(uint16_t b)
{
	uint32_t u = (uint32_t)b << 16u;
	float f;
	memcpy(&f, &u, 4u);
	return f;
}

static __global__ void FillKernel(uint8_t *dst, uint64_t bytes, uint64_t seed, uint32_t kind)
{
	for ( uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < bytes; i += (uint64_t)gridDim.x * blockDim.x )
	{
		uint64_t h = (seed ^ (i >> (kind == 2u ? 2u : kind == 1u ? 1u : 0u))) + 0x9e3779b97f4a7c15ull;
		h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ull;
		h = (h ^ (h >> 27)) * 0x94d049bb133111ebull;
		h ^= h >> 31;
		if ( kind == 1u )
		{
			uint16_t v = (uint16_t)((h & 0x8000u) | ((120u + (uint32_t)((h >> 16) % 7u)) << 7) | ((h >> 24) & 0x7fu));
			dst[i] = (uint8_t)(v >> (8u * (i & 1u)));
		}
		else if ( kind == 2u )
		{
			uint32_t v = (uint32_t)((h & 0x80000000u) | ((120u + (uint32_t)((h >> 32) % 6u)) << 23) | ((h >> 8) & 0x7fffffu));
			dst[i] = (uint8_t)(v >> (8u * (i & 3u)));
		}
		else if ( kind == 3u )
			dst[i] = (uint8_t)((h & 0x3eu) | 0x20u);
		else if ( kind == 4u )
			dst[i] = (uint8_t)(124u + (h & 3u));
		else
			dst[i] = (i % 272u) >= 256u ? (uint8_t)(124u + (h & 3u)) : (uint8_t)(h & 0xffu);
	}
}

static void *AllocFilled(uint64_t bytes, uint32_t kind, uint64_t seed)
{
	void *p = 0;
	CHECK(cudaMalloc(&p, bytes));
	FillKernel<<<256, 256>>>((uint8_t *)p, bytes, seed, kind);
	return p;
}

static D41Fp8Linear MakeLin(uint32_t out_d, uint32_t in_d, uint64_t seed)
{
	D41Fp8Linear l;
	memset(&l, 0, sizeof(l));
	l.weight = (const uint8_t *)AllocFilled((uint64_t)out_d * in_d, 3u, seed);
	l.scale = (const uint8_t *)AllocFilled((uint64_t)out_d * (in_d / D41_FP8_BLOCK), 4u, seed + 1u);
	l.wide = 0;
	return l;
}

struct MockRoundCtx { uint32_t op_count; };

static SparkStatus MockRound(void *ctx, void *stream_v, uint32_t op, uint32_t stage, uint32_t rows, uint32_t row_elems, const void *local, void *full)
{
	MockRoundCtx *m = (MockRoundCtx *)ctx;
	cudaStream_t s = (cudaStream_t)stream_v;
	if ( stage == SPARK_STAGE_RUNNER_ROUND_FINISH )
		return SPARK_STATUS_OK;
	m->op_count++;
	if ( row_elems == 0u )
		row_elems = D41_HIDDEN;
	const uint64_t bytes = (uint64_t)rows * row_elems * sizeof(uint16_t);
	if ( op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER )
	{
		for ( uint32_t r = 0u; r < D41_TP; ++r )
			CHECK(cudaMemcpyAsync((uint8_t *)full + (uint64_t)r * bytes, local, bytes, cudaMemcpyDeviceToDevice, s));
	}
	else if ( op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL )
	{
		CHECK(cudaMemsetAsync(full, 0, (uint64_t)D41_TP * bytes, s));
		for ( uint32_t r = 0u; r < D41_TP; ++r )
		{
			const uint8_t *src = (const uint8_t *)local + (uint64_t)r * bytes;
			uint8_t *dst = (uint8_t *)full + (uint64_t)r * bytes;
			uint32_t sf, sw;
			D41SharedSlice(r, &sf, &sw);
			const uint64_t rbase = (uint64_t)rows * D41_MOE_SHARED_ELEMENTS * 2u;
			CHECK(cudaMemcpyAsync(dst + rbase, src + rbase, bytes - rbase, cudaMemcpyDeviceToDevice, s));
			for ( uint32_t rr = 0u; rr < rows; ++rr )
			{
				CHECK(cudaMemcpyAsync(dst + (uint64_t)rr * sw, src + (uint64_t)rr * D41_SHARED_SLICE_MAX, sw, cudaMemcpyDeviceToDevice, s));
				CHECK(cudaMemcpyAsync(dst + (uint64_t)rows * sw + (uint64_t)rr * (sw / 32u),
					src + (uint64_t)rows * D41_SHARED_SLICE_MAX + (uint64_t)rr * (D41_SHARED_SLICE_MAX / 32u), sw / 32u, cudaMemcpyDeviceToDevice, s));
			}
		}
	}
	else if ( full != local )
		CHECK(cudaMemcpyAsync(full, local, bytes, cudaMemcpyDeviceToDevice, s));
	return SPARK_STATUS_OK;
}

static D41Layer MakeLayer(uint64_t seed, uint32_t sh_w, const uint8_t *w13, const uint8_t *w2)
{
	D41Layer L;
	memset(&L, 0, sizeof(L));
	L.attn_norm = (const uint16_t *)AllocFilled(D41_HIDDEN * 2u, 1u, seed + 1u);
	L.ffn_norm = (const uint16_t *)AllocFilled(D41_HIDDEN * 2u, 1u, seed + 2u);
	L.hc_attn_fn = (const float *)AllocFilled((uint64_t)LmHcSinkhornShape<D41K_HC>::kMix * D41K_HC * D41_HIDDEN * 4u, 2u, seed + 3u);
	L.hc_attn_base = (const float *)AllocFilled(LmHcSinkhornShape<D41K_HC>::kMix * 4u, 2u, seed + 4u);
	L.hc_attn_scale = (const float *)AllocFilled(4u * 4u, 2u, seed + 5u);
	L.hc_ffn_fn = (const float *)AllocFilled((uint64_t)LmHcSinkhornShape<D41K_HC>::kMix * D41K_HC * D41_HIDDEN * 4u, 2u, seed + 6u);
	L.hc_ffn_base = (const float *)AllocFilled(LmHcSinkhornShape<D41K_HC>::kMix * 4u, 2u, seed + 7u);
	L.hc_ffn_scale = (const float *)AllocFilled(4u * 4u, 2u, seed + 8u);
	L.q_a = MakeLin(D41_QLORA, D41_HIDDEN, seed + 10u);
	L.q_b = MakeLin(D41_LOCAL_HEADS * D41_HEAD_DIM, D41_QLORA, seed + 12u);
	L.kv_a = MakeLin(D41_HEAD_DIM, D41_HIDDEN, seed + 14u);
	L.q_norm = (const uint16_t *)AllocFilled(D41_QLORA * 2u, 1u, seed + 16u);
	L.kv_norm = (const uint16_t *)AllocFilled(D41_HEAD_DIM * 2u, 1u, seed + 17u);
	L.sink = (const float *)AllocFilled(D41_LOCAL_HEADS * 4u, 2u, seed + 18u);
	for ( uint32_t g = 0u; g < D41_LOCAL_GROUPS; ++g )
		L.o_a[g] = MakeLin(D41_GROUP_ROWS, D41_GROUP_WIDTH, seed + 20u + 2u * g);
	L.o_b = MakeLin(D41_HIDDEN, D41_WO_A_ROWS, seed + 30u);
	L.c_wkv = (const uint16_t *)AllocFilled((uint64_t)D41_HEAD_DIM * D41_HIDDEN * 2u, 1u, seed + 32u);
	L.c_wgate = (const uint16_t *)AllocFilled((uint64_t)D41_HEAD_DIM * D41_HIDDEN * 2u, 1u, seed + 33u);
	L.c_norm = (const uint16_t *)AllocFilled(D41_HEAD_DIM * 2u, 1u, seed + 34u);
	L.idx_q_b = MakeLin(D41_LOCAL_INDEX_HEADS * D41_INDEX_DIM, D41_QLORA, seed + 36u);
	L.idx_wk = (const uint16_t *)AllocFilled((uint64_t)D41_INDEX_DIM * D41_HEAD_DIM * 2u, 1u, seed + 38u);
	L.idx_kn = (const uint16_t *)AllocFilled(D41_INDEX_DIM * 2u, 1u, seed + 39u);
	L.idx_wp = (const uint16_t *)AllocFilled((uint64_t)D41_LOCAL_INDEX_HEADS * D41_HIDDEN * 2u, 1u, seed + 40u);
	L.router = (const uint16_t *)AllocFilled((uint64_t)D41_EXPERTS * D41_HIDDEN * 2u, 1u, seed + 42u);
	L.router_bias = (const float *)AllocFilled(D41_EXPERTS * 4u, 2u, seed + 43u);
	L.sh_w1 = MakeLin(sh_w, D41_HIDDEN, seed + 44u);
	L.sh_w3 = MakeLin(sh_w, D41_HIDDEN, seed + 46u);
	L.sh_w2 = MakeLin(D41_HIDDEN, sh_w, seed + 48u);
	for ( uint32_t s = 0u; s < D41_TP; ++s )
	{
		uint32_t f, w;
		D41SharedSlice(s, &f, &w);
		L.sh_w2_part[s] = MakeLin(D41_HIDDEN_SLICE, w, seed + 50u + 2u * s);
	}
	L.e_w13_cells = w13;
	L.e_w2_cells = w2;
	return L;
}

static uint32_t *DevU32(const std::vector<uint32_t> &v)
{
	uint32_t *p = 0;
	CHECK(cudaMalloc(&p, v.size() * 4u));
	CHECK(cudaMemcpy(p, v.data(), v.size() * 4u, cudaMemcpyHostToDevice));
	return p;
}

static uint32_t Overlaps(const void *a, uint64_t ab, const void *b, uint64_t bb)
{
	const uintptr_t a0 = (uintptr_t)a, a1 = a0 + ab, b0 = (uintptr_t)b, b1 = b0 + bb;
	return a0 < b1 && b0 < a1 ? 1u : 0u;
}

static void CheckExact(const char *label, const uint16_t *got, const uint16_t *ref, size_t count)
{
	size_t diffs = 0u;
	float mad = 0.0f;
	for ( size_t i = 0u; i < count; ++i )
		if ( got[i] != ref[i] )
		{
			diffs++;
			mad = fmaxf(mad, fabsf(Bf16ToF(got[i]) - Bf16ToF(ref[i])));
		}
	if ( diffs != 0u )
	{
		fprintf(stderr, "MISMATCH %s diffs=%zu/%zu mad=%.6g\n", label, diffs, count, mad);
		exit(1);
	}
}

int main(void)
{
	int sms = 0;
	CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
	D41ModelState *st = (D41ModelState *)calloc(1u, sizeof(D41ModelState));
	MockRoundCtx rctx = { 0u };
	SparkStageRunnerServices svc;
	memset(&svc, 0, sizeof(svc));
	svc.context = &rctx;
	svc.round = MockRound;
	st->services = &svc;
	st->tp_degree = D41_TP;
	st->tp_rank = 0u;
	st->max_rows = 64u;
	st->slots = 8u;
	st->kv_attached = 1u;
	st->context_slots = 8u;
	st->max_positions = 512u;
	st->moe_whole = 0u;
	D41SharedSlice(0u, &st->shared_first, &st->shared_width);
	st->shard.degree = D41_KV_TP;
	st->shard.rank = D41_KV_RANK(st);
	st->shard.grain = D41_GRAIN;
	st->local_score_stride = 512u;
	st->select_rows = 64u;
	st->global_score_stride = 512u;
	st->topk_entries = 2u * 64u * LmTopkExactCandidateEntries(512u, D41_INDEX_TOPK, LM_TOPK_EXACT_CHUNK);
	st->ring_bytes = (uint64_t)D41_LAYERS * D41_RING_ROWS * D41_HEAD_DIM * 2u;
	st->compressor_bytes = 3u * 2u * D41_HEAD_DIM * 4u;
	st->history_bytes = 4u * 4u;
	st->recurrent_bytes = (st->ring_bytes + st->compressor_bytes + st->history_bytes + 255u) & ~255ull;

	const uint64_t abytes = D41Cuda::Plan(st, 0, 0u);
	uint8_t *arena = 0;
	CHECK(cudaMalloc(&arena, abytes));
	CHECK(cudaMemset(arena, 0, abytes));
	REQ(D41Cuda::Plan(st, arena, 0u) == abytes, "Plan");
	st->arena = arena;
	st->arena_bytes = abytes;

	const uint64_t moe_send_bytes = (uint64_t)D41_TP * st->max_rows * D41_MOE_ROW_ELEMENTS * 2u;
	const uint64_t q_local_bytes = (uint64_t)st->max_rows * D41_LOCAL_HEADS * D41_HEAD_DIM * 2u;
	const uint64_t q_gathered_bytes = (uint64_t)D41_TP * st->max_rows * D41_INDEX_PACK * 2u;
	const uint64_t idx_q_bytes = (uint64_t)st->max_rows * D41_INDEX_HEADS * D41_INDEX_DIM * 2u;
	const uint64_t idx_w_bytes = (uint64_t)st->max_rows * D41_INDEX_HEADS * 4u;
	const uint64_t lscores_bytes = (uint64_t)st->max_rows * st->local_score_stride * 4u;
	const uint64_t sel_bytes = (uint64_t)st->max_rows * D41_INDEX_TOPK * 4u;
	REQ(Overlaps(st->moe_send, moe_send_bytes, st->q_local, q_local_bytes) == 0u, "moe_send vs q_local");
	REQ(Overlaps(st->moe_send, moe_send_bytes, st->q_gathered, q_gathered_bytes) == 0u, "moe_send vs q_gathered");
	REQ(Overlaps(st->moe_send, moe_send_bytes, st->idx_query, idx_q_bytes) == 0u, "moe_send vs idx_query");
	REQ(Overlaps(st->moe_send, moe_send_bytes, st->idx_w_full, idx_w_bytes) == 0u, "moe_send vs idx_w_full");
	REQ(Overlaps(st->moe_send, moe_send_bytes, st->local_scores, lscores_bytes) == 0u, "moe_send vs local_scores");
	REQ(Overlaps(st->moe_send, moe_send_bytes, st->selected, sel_bytes) == 0u, "moe_send vs selected");

	st->recurrent = (uint8_t *)AllocFilled((uint64_t)8u * st->recurrent_bytes, 1u, 0x111u);
	std::vector<float> hfreq(D41K_ROPE_PAIRS * 2u);
	for ( uint32_t i = 0u; i < D41K_ROPE_PAIRS; ++i )
	{
		hfreq[i] = powf(10000.0f, -2.0f * (float)i / (float)D41_ROPE);
		hfreq[D41K_ROPE_PAIRS + i] = powf(100000.0f, -2.0f * (float)i / (float)D41_ROPE);
	}
	float *dfreq = 0;
	CHECK(cudaMalloc(&dfreq, hfreq.size() * 4u));
	CHECK(cudaMemcpy(dfreq, hfreq.data(), hfreq.size() * 4u, cudaMemcpyHostToDevice));
	st->full_frequency = dfreq;
	st->compress_frequency = dfreq + D41K_ROPE_PAIRS;

	LmKvAccessError *kverr = 0;
	CHECK(cudaMalloc(&kverr, sizeof(*kverr)));
	CHECK(cudaMemset(kverr, 0, sizeof(*kverr)));
	uint8_t *kv_pools[4] = {}, *kv_pools_init[4] = {};
	size_t kv_pool_sz[4] = {};
	for ( uint32_t c = 0u; c < 4u; ++c )
	{
		const uint32_t pslots = c < 3u ? D41Ratio2Kv::kPageSlots : D41Ratio1Kv::kPageSlots;
		const uint32_t pages = 4u;
		std::vector<uint32_t> pt(8u * pages);
		for ( uint32_t s = 0u; s < 8u * pages; ++s )
			pt[s] = s;
		kv_pool_sz[c] = (size_t)8u * pages * pslots * D41_SLOT_BYTES;
		kv_pools[c] = (uint8_t *)AllocFilled(kv_pool_sz[c], 1u, 0x500u + c);
		kv_pools_init[c] = (uint8_t *)AllocFilled(kv_pool_sz[c], 1u, 0x500u + c);
		LmKvShardView v;
		memset(&v, 0, sizeof(v));
		REQ(LmKvViewInitialize(&v.pages, kv_pools[c], DevU32(pt), pages, 8u, 8u * pages, kverr) == 0, "kv");
		v.shard = st->shard;
		st->latent_views[c] = v;
		st->index_views[c] = v;
	}

	const uint64_t c13 = (uint64_t)(D41_HIDDEN / 32u) * (2u * D41_EXPERT_SLICE / 16u) * 272u;
	const uint64_t c2 = (uint64_t)(D41_MOE_INTER / 32u) * (D41_HIDDEN_SLICE / 16u) * 272u;
	const uint8_t *w13 = (const uint8_t *)AllocFilled(c13 * D41_EXPERTS, 5u, 0x600u);
	const uint8_t *w2 = (const uint8_t *)AllocFilled(c2 * D41_EXPERTS, 5u, 0x601u);
	const uint32_t test_layers[4] = { 0u, 1u, 2u, 20u };
	for ( uint32_t li = 0u; li < 4u; ++li )
		st->layers[test_layers[li]] = MakeLayer(0x1000u + li * 100u, st->shared_width, w13, w2);

	const uint32_t N = 12u;
	uint16_t *init_streams = (uint16_t *)AllocFilled((uint64_t)N * D41K_HC * D41_HIDDEN * 2u, 1u, 0x777u);
	uint8_t *init_rec = (uint8_t *)AllocFilled((uint64_t)8u * st->recurrent_bytes, 1u, 0x888u);
	float *init_pre = (float *)AllocFilled((uint64_t)N * D41K_HC * 4u, 2u, 0x999u);
	CHECK(cudaDeviceSynchronize());

	auto reset_state = [&]()
	{
		CHECK(cudaMemcpy(st->recurrent, init_rec, (uint64_t)8u * st->recurrent_bytes, cudaMemcpyDeviceToDevice));
		for ( uint32_t c = 0u; c < 4u; ++c )
			CHECK(cudaMemcpy(kv_pools[c], kv_pools_init[c], kv_pool_sz[c], cudaMemcpyDeviceToDevice));
	};

	auto run_layers = [&](SparkStageRunnerStep &step, const uint32_t *layers, uint32_t n_layers)
	{
		D41Cuda::Frame fr;
		memset(&fr, 0, sizeof(fr));
		fr.step = &step;
		fr.current = &step;
		fr.streams = st->streams;
		fr.pre_head = st->pre_head;
		fr.selected = st->selected;
		for ( uint32_t li = 0u; li < n_layers; ++li )
		{
			const uint32_t L = layers[li];
			st->compress_done = 0u;
			st->mix_collapsed = 0u;
			st->mx_valid = 0u;
			st->post_mode = 0u;
			REQ(D41Cuda::HyperMix(st, fr, L, D41_ATTENTION_PART) == SPARK_STATUS_OK, "hmA");
			REQ(D41Cuda::CollapseNorm(st, fr, L, D41_ATTENTION_PART) == SPARK_STATUS_OK, "cnA");
			REQ(D41AttentionWhole(st, &st->layers[L], L, &step, 0) == SPARK_STATUS_OK, "attn");
			REQ(D41Cuda::HyperPost(st, fr, D41_ATTENTION_PART) == SPARK_STATUS_OK, "hpA");
			REQ(D41Cuda::HyperMix(st, fr, L, D41_FFN_PART) == SPARK_STATUS_OK, "hmF");
			REQ(D41Cuda::CollapseNorm(st, fr, L, D41_FFN_PART) == SPARK_STATUS_OK, "cnF");
			REQ(D41Moe(st, &st->layers[L], &step, 0) == SPARK_STATUS_OK, "moe");
			REQ(D41Cuda::HyperPost(st, fr, D41_FFN_PART) == SPARK_STATUS_OK, "hpF");
		}
	};

	auto prep_step = [&](SparkStageRunnerStep &step, uint32_t nr, const std::vector<uint32_t> &pos, uint32_t ctx_len, uint32_t commit, uint32_t tree,
		uint32_t row_inv, const std::vector<uint32_t> *par, uint32_t *rfault)
	{
		memset(&step, 0, sizeof(step));
		step.rows = nr;
		step.sequences = 1u;
		step.commit = commit;
		step.tree = tree;
		step.multiprocessors = (uint32_t)sms;
		std::vector<uint32_t> len(8u, 1u), seq(nr, 0u), beg = { 0u, nr }, idx(nr), rec(8u);
		len[0] = ctx_len;
		for ( uint32_t s = 0u; s < 8u; ++s )
			rec[s] = s;
		for ( uint32_t r = 0u; r < nr; ++r )
			idx[r] = r;
		step.positions = DevU32(pos);
		step.context_length = DevU32(len);
		step.sequence_of_row = DevU32(seq);
		step.sequence_row_begin = DevU32(beg);
		step.sequence_row_indices = DevU32(idx);
		step.recurrent_index = DevU32(rec);
		step.route_fault = rfault;
		if ( tree != 0u )
		{
			std::vector<uint32_t> frk(nr, 0xffffffffu), off(nr, 0u), slt = pos;
			step.parent_rows = DevU32(*par);
			step.fork_positions = DevU32(frk);
			step.slot_offsets = DevU32(off);
			step.slots = DevU32(slt);
			CHECK(cudaMemcpy(st->fork_ratio2, frk.data(), nr * 4u, cudaMemcpyHostToDevice));
			CHECK(cudaMemcpy(st->offset_ratio2, off.data(), nr * 4u, cudaMemcpyHostToDevice));
		}
		st->row_invariant = row_inv;
		st->own_rows = 0u;
		st->step_score_stride = 512u;
		st->step_ratio2_stride = 512u;
		LM_LAUNCH((LmRowLanesKernel), 1u, 256u, 0, (SparkGpuStream)0, step.sequence_row_begin, step.sequence_row_indices, st->row_lane, st->row_ordinal);
		D41CompressedEmitPositionLaunch(1u, 256u, (SparkGpuStream)0, step.positions, st->bound_ratio2, nr, 2u);
		D41CompressedEmitPositionLaunch(1u, 256u, (SparkGpuStream)0, step.positions, st->bound_ratio1, nr, 1u);
		D41CompressedLengthLaunch(1u, 256u, (SparkGpuStream)0, step.context_length, st->length_ratio2, 8u, 2u);
		D41CompressedLengthLaunch(1u, 256u, (SparkGpuStream)0, step.context_length, st->length_ratio1, 8u, 1u);
		D41RopeTableLaunch(nr, D41K_ROPE_PAIRS, (SparkGpuStream)0, step.positions, st->full_frequency, st->rope_sc_full, st->rope_sc_compress, nr);
	};

	auto prep_batch_decode = [&](SparkStageRunnerStep &step, uint32_t nr, const std::vector<uint32_t> &pos, uint32_t *rfault)
	{
		memset(&step, 0, sizeof(step));
		step.rows = nr;
		step.sequences = nr;
		step.commit = 1u;
		step.tree = 0u;
		step.multiprocessors = (uint32_t)sms;
		std::vector<uint32_t> len(8u, 1u), seq(nr), beg(nr + 1u), idx(nr), rec(8u);
		for ( uint32_t s = 0u; s < 8u; ++s )
			rec[s] = s;
		for ( uint32_t r = 0u; r < nr; ++r )
		{
			len[r] = pos[r] + 1u;
			seq[r] = r;
			beg[r] = r;
			idx[r] = r;
		}
		beg[nr] = nr;
		step.positions = DevU32(pos);
		step.context_length = DevU32(len);
		step.sequence_of_row = DevU32(seq);
		step.sequence_row_begin = DevU32(beg);
		step.sequence_row_indices = DevU32(idx);
		step.recurrent_index = DevU32(rec);
		step.route_fault = rfault;
		st->row_invariant = 0u;
		st->own_rows = 0u;
		st->step_score_stride = 512u;
		st->step_ratio2_stride = 512u;
		LM_LAUNCH((LmRowLanesKernel), nr, 256u, 0, (SparkGpuStream)0, step.sequence_row_begin, step.sequence_row_indices, st->row_lane, st->row_ordinal);
		D41CompressedEmitPositionLaunch(1u, 256u, (SparkGpuStream)0, step.positions, st->bound_ratio2, nr, 2u);
		D41CompressedEmitPositionLaunch(1u, 256u, (SparkGpuStream)0, step.positions, st->bound_ratio1, nr, 1u);
		D41CompressedLengthLaunch(1u, 256u, (SparkGpuStream)0, step.context_length, st->length_ratio2, 8u, 2u);
		D41CompressedLengthLaunch(1u, 256u, (SparkGpuStream)0, step.context_length, st->length_ratio1, 8u, 1u);
		D41RopeTableLaunch(nr, D41K_ROPE_PAIRS, (SparkGpuStream)0, step.positions, st->full_frequency, st->rope_sc_full, st->rope_sc_compress, nr);
	};

	auto free_step = [&](SparkStageRunnerStep &step)
	{
		cudaFree((void *)step.positions);
		cudaFree((void *)step.context_length);
		cudaFree((void *)step.sequence_of_row);
		cudaFree((void *)step.sequence_row_begin);
		cudaFree((void *)step.sequence_row_indices);
		cudaFree((void *)step.recurrent_index);
		if ( step.tree != 0u )
		{
			cudaFree((void *)step.parent_rows);
			cudaFree((void *)step.fork_positions);
			cudaFree((void *)step.slot_offsets);
			cudaFree((void *)step.slots);
		}
	};

	uint32_t *rfault = DevU32({ 0u });

	{
		reset_state();
		const uint32_t nr = 6u;
		std::vector<uint32_t> pos(nr);
		for ( uint32_t r = 0u; r < nr; ++r )
			pos[r] = 200u + r;
		SparkStageRunnerStep step;
		prep_step(step, nr, pos, 200u + nr, 0u, 0u, 1u, 0, rfault);
		const uint64_t qbytes = (uint64_t)nr * D41_LOCAL_HEADS * D41_HEAD_DIM * 2u;
		FillKernel<<<256, 256>>>((uint8_t *)st->q_local, qbytes, 0xabcdu, 1u);
		FillKernel<<<256, 256>>>((uint8_t *)st->q_gathered, (uint64_t)D41_TP * nr * D41_INDEX_PACK * 2u, 0xbcdeu, 1u);
		CHECK(cudaDeviceSynchronize());
		std::vector<uint16_t> q_before(qbytes / 2u), q_after(qbytes / 2u);
		CHECK(cudaMemcpy(q_before.data(), st->q_local, qbytes, cudaMemcpyDeviceToHost));
		REQ(D41SelectWhole<D41Ratio2Kv>(st, 2u, &step, st->bound_ratio2, st->length_ratio2, 0) == SPARK_STATUS_OK, "SelectWhole");
		CHECK(cudaDeviceSynchronize());
		CHECK(cudaMemcpy(q_after.data(), st->q_local, qbytes, cudaMemcpyDeviceToHost));
		CheckExact("D41SelectWhole q_local non-overlap", q_after.data(), q_before.data(), q_before.size());
		free_step(step);
	}

	auto run_seq_order = [&](uint32_t P, const std::vector<uint32_t> &row_ids, const uint32_t *layers, uint32_t n_layers, std::vector<uint16_t> &out)
	{
		reset_state();
		out.resize((size_t)row_ids.size() * D41K_HC * D41_HIDDEN);
		for ( uint32_t k = 0u; k < (uint32_t)row_ids.size(); ++k )
		{
			const uint32_t rid = row_ids[k];
			CHECK(cudaMemcpy(st->streams, init_streams + (uint64_t)rid * D41K_HC * D41_HIDDEN, D41K_HC * D41_HIDDEN * 2u, cudaMemcpyDeviceToDevice));
			CHECK(cudaMemcpy(st->pre_init, init_pre + (uint64_t)rid * D41K_HC, D41K_HC * 4u, cudaMemcpyDeviceToDevice));
			SparkStageRunnerStep step;
			prep_step(step, 1u, { P + k }, P + k + 1u, 1u, 0u, 0u, 0, rfault);
			run_layers(step, layers, n_layers);
			CHECK(cudaDeviceSynchronize());
			CHECK(cudaMemcpy(out.data() + (size_t)k * D41K_HC * D41_HIDDEN, st->streams, D41K_HC * D41_HIDDEN * 2u, cudaMemcpyDeviceToHost));
			free_step(step);
		}
	};

	for ( uint32_t P : { 40u, 125u, 200u } )
	{
		std::vector<uint32_t> ids12 = { 0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u };
		std::vector<uint16_t> ref12;
		run_seq_order(P, ids12, test_layers, 4u, ref12);

		struct Case { const char *name; uint32_t rows; uint32_t tree; };
		const Case cases[] = {
			{ "linear_r6", 6u, 0u },
			{ "linear_r8", 8u, 0u },
			{ "linear_r10", 10u, 0u },
			{ "linear_r12", 12u, 0u },
			{ "tree_r6", 6u, 1u },
			{ "tree_r10", 10u, 1u },
			{ "tree_r12", 12u, 1u }
		};
		for ( const Case &c : cases )
		{
			reset_state();
			CHECK(cudaMemcpy(st->streams, init_streams, (uint64_t)c.rows * D41K_HC * D41_HIDDEN * 2u, cudaMemcpyDeviceToDevice));
			CHECK(cudaMemcpy(st->pre_init, init_pre, (uint64_t)c.rows * D41K_HC * 4u, cudaMemcpyDeviceToDevice));
			std::vector<uint32_t> pos(c.rows), par(c.rows);
			for ( uint32_t r = 0u; r < c.rows; ++r )
			{
				pos[r] = P + r;
				par[r] = r == 0u ? D41K_NO_ROW : r - 1u;
			}
			SparkStageRunnerStep step;
			prep_step(step, c.rows, pos, P + c.rows, 0u, c.tree, 1u, &par, rfault);
			run_layers(step, test_layers, 4u);
			CHECK(cudaDeviceSynchronize());
			std::vector<uint16_t> got((size_t)c.rows * D41K_HC * D41_HIDDEN);
			CHECK(cudaMemcpy(got.data(), st->streams, got.size() * 2u, cudaMemcpyDeviceToHost));
			free_step(step);
			char label[64];
			snprintf(label, sizeof(label), "P=%u %s", P, c.name);
			CheckExact(label, got.data(), ref12.data(), got.size());
		}

		{
			reset_state();
			const uint32_t b8 = 8u;
			CHECK(cudaMemcpy(st->streams, init_streams, (uint64_t)b8 * D41K_HC * D41_HIDDEN * 2u, cudaMemcpyDeviceToDevice));
			CHECK(cudaMemcpy(st->pre_init, init_pre, (uint64_t)b8 * D41K_HC * 4u, cudaMemcpyDeviceToDevice));
			const std::vector<uint32_t> pos = { P, P + 3u, P + 11u, P + 17u, P + 29u, P + 43u, P + 59u, P + 71u };
			SparkStageRunnerStep step;
			prep_batch_decode(step, b8, pos, rfault);
			REQ(st->row_invariant == 0u, "5c row_invariant must be 0");
			run_layers(step, test_layers, 4u);
			CHECK(cudaDeviceSynchronize());
			std::vector<uint16_t> got_lane0(D41K_HC * D41_HIDDEN);
			CHECK(cudaMemcpy(got_lane0.data(), st->streams, got_lane0.size() * 2u, cudaMemcpyDeviceToHost));
			free_step(step);
			char label[64];
			snprintf(label, sizeof(label), "5c P=%u B1==B8_lane0", P);
			CheckExact(label, got_lane0.data(), ref12.data(), got_lane0.size());
		}
	}

	{
		const uint32_t P = 125u, layer0 = 0u;
		std::vector<uint16_t> ref_a, ref_b, ref_c;
		run_seq_order(P, { 0u, 1u, 2u }, &layer0, 1u, ref_a);
		run_seq_order(P, { 0u, 3u, 4u }, &layer0, 1u, ref_b);
		run_seq_order(P, { 0u, 1u, 5u }, &layer0, 1u, ref_c);

		reset_state();
		CHECK(cudaMemcpy(st->streams, init_streams, (uint64_t)6u * D41K_HC * D41_HIDDEN * 2u, cudaMemcpyDeviceToDevice));
		CHECK(cudaMemcpy(st->pre_init, init_pre, (uint64_t)6u * D41K_HC * 4u, cudaMemcpyDeviceToDevice));
		const std::vector<uint32_t> pos = { P, P + 1u, P + 2u, P + 1u, P + 2u, P + 2u };
		const std::vector<uint32_t> par = { D41K_NO_ROW, 0u, 1u, 0u, 3u, 1u };
		SparkStageRunnerStep step;
		prep_step(step, 6u, pos, P + 3u, 0u, 1u, 1u, &par, rfault);
		run_layers(step, &layer0, 1u);
		CHECK(cudaDeviceSynchronize());
		std::vector<uint16_t> got((size_t)6u * D41K_HC * D41_HIDDEN);
		CHECK(cudaMemcpy(got.data(), st->streams, got.size() * 2u, cudaMemcpyDeviceToHost));
		free_step(step);
		const size_t row_el = D41K_HC * D41_HIDDEN;
		CheckExact("branch_tree row0", got.data() + 0u * row_el, ref_a.data() + 0u * row_el, row_el);
		CheckExact("branch_tree row1", got.data() + 1u * row_el, ref_a.data() + 1u * row_el, row_el);
		CheckExact("branch_tree row2", got.data() + 2u * row_el, ref_a.data() + 2u * row_el, row_el);
		CheckExact("branch_tree row3", got.data() + 3u * row_el, ref_b.data() + 1u * row_el, row_el);
		CheckExact("branch_tree row4", got.data() + 4u * row_el, ref_b.data() + 2u * row_el, row_el);
		CheckExact("branch_tree row5", got.data() + 5u * row_el, ref_c.data() + 2u * row_el, row_el);
	}

	{
		const uint32_t P = 125u, nr = 6u, layer0 = 0u;
		std::vector<uint16_t> ref_out, got_out((size_t)nr * D41K_HC * D41_HIDDEN);
		std::vector<uint16_t> ref_ring(D41_RING_ROWS * D41_HEAD_DIM), got_ring(D41_RING_ROWS * D41_HEAD_DIM);
		run_seq_order(P, { 0u, 1u, 2u, 3u, 4u, 5u }, &layer0, 1u, ref_out);
		CHECK(cudaMemcpy(ref_ring.data(), st->recurrent, ref_ring.size() * 2u, cudaMemcpyDeviceToHost));

		reset_state();
		CHECK(cudaMemcpy(st->streams, init_streams, (uint64_t)nr * D41K_HC * D41_HIDDEN * 2u, cudaMemcpyDeviceToDevice));
		CHECK(cudaMemcpy(st->pre_init, init_pre, (uint64_t)nr * D41K_HC * 4u, cudaMemcpyDeviceToDevice));
		std::vector<uint32_t> pos(nr);
		for ( uint32_t r = 0u; r < nr; ++r )
			pos[r] = P + r;
		SparkStageRunnerStep step;
		prep_step(step, nr, pos, P + nr, 1u, 0u, 1u, 0, rfault);
		run_layers(step, &layer0, 1u);
		CHECK(cudaDeviceSynchronize());
		CHECK(cudaMemcpy(got_out.data(), st->streams, got_out.size() * 2u, cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(got_ring.data(), st->recurrent, got_ring.size() * 2u, cudaMemcpyDeviceToHost));
		free_step(step);
		CheckExact("commit_r6 out", got_out.data(), ref_out.data(), got_out.size());
		CheckExact("commit_r6 ring", got_ring.data(), ref_ring.data(), got_ring.size());
	}

	{
		const uint32_t wave_layers[3] = { 0u, 2u, 20u };
		const uint32_t seq_counts[3] = { 12u, 24u, 20u };
		const uint32_t seq_slots[3] = { 2u, 5u, 1u };
		const uint32_t total_rows = seq_counts[0] + seq_counts[1] + seq_counts[2];
		uint16_t *init_hidden = (uint16_t *)AllocFilled((uint64_t)total_rows * D41_HIDDEN * 2u, 1u, 0x4321u);
		uint16_t *work_hidden = 0;
		CHECK(cudaMalloc(&work_hidden, (uint64_t)total_rows * D41_HIDDEN * 2u));
		std::vector<uint32_t> h_pos(total_rows), h_seq(total_rows), h_tok(total_rows), h_len(8u, 1u), h_beg = { 0u, 12u, 36u, 56u }, h_idx(total_rows), h_rec = { 5u, 1u, 2u };
		uint32_t off = 0u;
		for ( uint32_t s = 0u; s < 3u; ++s )
		{
			h_len[seq_slots[s]] = seq_counts[s];
			for ( uint32_t k = 0u; k < seq_counts[s]; ++k )
			{
				h_pos[off + k] = k;
				h_seq[off + k] = seq_slots[s];
				h_tok[off + k] = 100u + off + k;
				h_idx[off + k] = total_rows - 1u - (off + k);
			}
			off += seq_counts[s];
		}
		auto run_schedule_waves = [&](SparkStageRunnerStep &step, uint32_t rows, uint32_t src_off, std::vector<uint16_t> &hidden_out, std::vector<uint8_t> &rec_out)
		{
			reset_state();
			CHECK(cudaMemcpy(work_hidden, init_hidden + (uint64_t)src_off * D41_HIDDEN, (uint64_t)rows * D41_HIDDEN * 2u, cudaMemcpyDeviceToDevice));
			step.hidden_bf16 = work_hidden;
			D41Cuda::Frame fr;
			memset(&fr, 0, sizeof(fr));
			REQ(D41Cuda::Begin(st, &step, 0, &fr) == SPARK_STATUS_OK, "WaveBegin");
			uint32_t more = 1u;
			while ( true )
			{
				REQ(D41Cuda::Wave(st, fr, &more) == SPARK_STATUS_OK, "WaveNext");
				if ( more == 0u )
					break;
				for ( uint32_t li = 0u; li < 3u; ++li )
					REQ(D41Schedule<D41Cuda>::Layer(st, fr, wave_layers[li]) == SPARK_STATUS_OK, "WaveLayer");
				REQ(D41Cuda::WaveEnd(st, fr) == SPARK_STATUS_OK, "WaveEnd");
			}
			REQ(D41Cuda::End(st, fr, SPARK_STATUS_OK) == SPARK_STATUS_OK, "WaveStepEnd");
			CHECK(cudaDeviceSynchronize());
			hidden_out.resize((size_t)rows * D41_HIDDEN);
			CHECK(cudaMemcpy(hidden_out.data(), work_hidden, hidden_out.size() * 2u, cudaMemcpyDeviceToHost));
			rec_out.resize((size_t)8u * st->recurrent_bytes);
			CHECK(cudaMemcpy(rec_out.data(), st->recurrent, rec_out.size(), cudaMemcpyDeviceToHost));
		};
		SparkStageRunnerStep packed;
		memset(&packed, 0, sizeof(packed));
		packed.rows = total_rows;
		packed.sequences = 3u;
		packed.commit = 1u;
		packed.multiprocessors = (uint32_t)sms;
		packed.positions = DevU32(h_pos);
		packed.context_length = DevU32(h_len);
		packed.sequence_of_row = DevU32(h_seq);
		packed.sequence_row_begin = DevU32(h_beg);
		packed.sequence_row_indices = DevU32(h_idx);
		packed.recurrent_index = DevU32(h_rec);
		packed.token_ids = DevU32(h_tok);
		packed.route_fault = rfault;
		packed.host_positions = h_pos.data();
		packed.host_sequence_of_row = h_seq.data();
		REQ(D41Cuda::GraphKey(st, &packed) == 0u, "TP4 packed prefill disables graph key");
		std::vector<uint16_t> packed_hidden;
		std::vector<uint8_t> packed_rec;
		run_schedule_waves(packed, total_rows, 0u, packed_hidden, packed_rec);
		uint32_t base = 0u;
		for ( uint32_t s = 0u; s < 3u; ++s )
		{
			const uint32_t cnt = seq_counts[s], slot = seq_slots[s];
			std::vector<uint32_t> sp(cnt), ss(cnt, slot), st_tok(cnt), sl(8u, 1u), sb = { 0u, cnt }, si(cnt), sr = { slot };
			sl[slot] = cnt;
			for ( uint32_t k = 0u; k < cnt; ++k )
			{
				sp[k] = k;
				st_tok[k] = h_tok[base + k];
				si[k] = k;
			}
			SparkStageRunnerStep single;
			memset(&single, 0, sizeof(single));
			single.rows = cnt;
			single.sequences = 1u;
			single.commit = 1u;
			single.multiprocessors = (uint32_t)sms;
			single.positions = DevU32(sp);
			single.context_length = DevU32(sl);
			single.sequence_of_row = DevU32(ss);
			single.sequence_row_begin = DevU32(sb);
			single.sequence_row_indices = DevU32(si);
			single.recurrent_index = DevU32(sr);
			single.token_ids = DevU32(st_tok);
			single.route_fault = rfault;
			single.host_positions = sp.data();
			single.host_sequence_of_row = ss.data();
			std::vector<uint16_t> single_hidden;
			std::vector<uint8_t> single_rec;
			run_schedule_waves(single, cnt, base, single_hidden, single_rec);
			char label[96];
			snprintf(label, sizeof(label), "TP4 5c packed prefill wave seq%u(rows=%u,slot=%u)==B1 hidden", s, cnt, slot);
			CheckExact(label, packed_hidden.data() + (size_t)base * D41_HIDDEN, single_hidden.data(), (size_t)cnt * D41_HIDDEN);
			REQ(memcmp(packed_rec.data() + (size_t)slot * st->recurrent_bytes, single_rec.data() + (size_t)slot * st->recurrent_bytes, st->recurrent_bytes) == 0,
				"TP4 5c packed prefill recurrent exact");
			cudaFree((void *)single.positions);
			cudaFree((void *)single.context_length);
			cudaFree((void *)single.sequence_of_row);
			cudaFree((void *)single.sequence_row_begin);
			cudaFree((void *)single.sequence_row_indices);
			cudaFree((void *)single.recurrent_index);
			cudaFree((void *)single.token_ids);
			base += cnt;
		}
		cudaFree((void *)packed.positions);
		cudaFree((void *)packed.context_length);
		cudaFree((void *)packed.sequence_of_row);
		cudaFree((void *)packed.sequence_row_begin);
		cudaFree((void *)packed.sequence_row_indices);
		cudaFree((void *)packed.recurrent_index);
		cudaFree((void *)packed.token_ids);
		cudaFree(work_hidden);
		cudaFree(init_hidden);
		printf("PASS TP4 5c packed multi-sequence prefill wave (12+24+20=56 rows, non-identity slots/order == B1 hidden+recurrent: 0 bit diffs)\n");
	}

	cudaFree(rfault);
	printf("PASS dsv41 tp4 verify exact (P=40/125/200 linear r6/r8/r10/r12, tree r6/r10/r12, 5c B1==B8_lane0 row_inv=0, branching tree, commit r6 ring, select non-overlap)\n");
	return 0;
}
