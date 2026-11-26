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

static uint16_t FToBf16(float f)
{
	uint32_t u;
	memcpy(&u, &f, 4u);
	return (uint16_t)((u + 0x7fffu + ((u >> 16u) & 1u)) >> 16u);
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

static D41Fp8Linear SliceCols(const D41Fp8Linear &full, uint32_t out_d, uint32_t full_in, uint32_t first_col, uint32_t width)
{
	D41Fp8Linear s;
	memset(&s, 0, sizeof(s));
	uint8_t *w = 0, *sc = 0;
	CHECK(cudaMalloc(&w, (uint64_t)out_d * width));
	CHECK(cudaMalloc(&sc, (uint64_t)out_d * (width / D41_FP8_BLOCK)));
	CHECK(cudaMemcpy2D(w, width, full.weight + first_col, full_in, width, out_d, cudaMemcpyDeviceToDevice));
	CHECK(cudaMemcpy2D(sc, width / D41_FP8_BLOCK, full.scale + first_col / D41_FP8_BLOCK, full_in / D41_FP8_BLOCK,
		width / D41_FP8_BLOCK, out_d, cudaMemcpyDeviceToDevice));
	s.weight = w;
	s.scale = sc;
	return s;
}

static D41Fp8Linear SliceRows(const D41Fp8Linear &full, uint32_t first_row, uint32_t width, uint32_t in_d)
{
	D41Fp8Linear s;
	memset(&s, 0, sizeof(s));
	s.weight = full.weight + (uint64_t)first_row * in_d;
	s.scale = full.scale + (uint64_t)first_row * (in_d / D41_FP8_BLOCK);
	return s;
}

struct RoundRec
{
	uint32_t op;
	uint32_t rows;
	uint32_t row_elems;
	uint64_t payload_bytes;
	uint64_t wire_egress_bytes;
	int capture_only;
	std::vector<uint8_t> captured;
	const uint8_t *inject_ptr;
	uint64_t inject_bytes;
};

static SparkStatus MockRound(void *ctx, void *stream_v, uint32_t op, uint32_t stage, uint32_t rows, uint32_t row_elems, const void *local, void *full)
{
	RoundRec *rec = (RoundRec *)ctx;
	cudaStream_t s = (cudaStream_t)stream_v;
	if ( stage == SPARK_STAGE_RUNNER_ROUND_FINISH )
		return SPARK_STATUS_OK;
	if ( row_elems == 0u )
		row_elems = D41_HIDDEN;
	const uint64_t bytes = (uint64_t)rows * row_elems * sizeof(uint16_t);
	rec->op = op;
	rec->rows = rows;
	rec->row_elems = row_elems;
	rec->payload_bytes = bytes;
	if ( op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER )
		rec->wire_egress_bytes = (uint64_t)(D41_TP - 1u) * bytes;
	else if ( op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16 )
		rec->wire_egress_bytes = (uint64_t)rows * row_elems < 10240ull ?
			(uint64_t)(D41_TP - 1u) * bytes :
			2u * (uint64_t)(D41_TP - 1u) * (bytes / D41_TP);
	else if ( op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_PAIR_GATHER )
		rec->wire_egress_bytes = bytes;
	else
		rec->wire_egress_bytes = bytes;
	if ( rec->capture_only != 0 )
	{
		rec->captured.resize((size_t)bytes);
		CHECK(cudaMemcpyAsync(rec->captured.data(), local, (size_t)bytes, cudaMemcpyDeviceToHost, s));
		CHECK(cudaStreamSynchronize(s));
		return SPARK_STATUS_OK;
	}
	if ( rec->inject_ptr != 0 )
	{
		CHECK(cudaMemcpyAsync(full, rec->inject_ptr, (size_t)rec->inject_bytes, cudaMemcpyHostToDevice, s));
		return SPARK_STATUS_OK;
	}
	if ( op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER || op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_PAIR_GATHER )
	{
		const uint32_t n = op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_PAIR_GATHER ? 2u : D41_TP;
		for ( uint32_t r = 0u; r < n; ++r )
			CHECK(cudaMemcpyAsync((uint8_t *)full + (uint64_t)r * bytes, local, (size_t)bytes, cudaMemcpyDeviceToDevice, s));
	}
	else if ( full != local )
		CHECK(cudaMemcpyAsync(full, local, (size_t)bytes, cudaMemcpyDeviceToDevice, s));
	return SPARK_STATUS_OK;
}

static uint32_t *DevU32(const std::vector<uint32_t> &v)
{
	uint32_t *p = 0;
	CHECK(cudaMalloc(&p, v.size() * 4u));
	CHECK(cudaMemcpy(p, v.data(), v.size() * 4u, cudaMemcpyHostToDevice));
	return p;
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

static D41Layer MakeAttnMoeLayer(uint64_t seed)
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
	L.q_b = MakeLin(D41_HEADS * D41_HEAD_DIM, D41_QLORA, seed + 12u);
	L.kv_a = MakeLin(D41_HEAD_DIM, D41_HIDDEN, seed + 14u);
	L.q_norm = (const uint16_t *)AllocFilled(D41_QLORA * 2u, 1u, seed + 16u);
	L.kv_norm = (const uint16_t *)AllocFilled(D41_HEAD_DIM * 2u, 1u, seed + 17u);
	L.sink = (const float *)AllocFilled(D41_HEADS * 4u, 2u, seed + 18u);
	for ( uint32_t g = 0u; g < D41_OGROUPS; ++g )
		L.o_a[g] = MakeLin(D41_OLORA, D41_GROUP_WIDTH, seed + 20u + 2u * g);
	L.o_b = MakeLin(D41_HIDDEN, D41_WO_A_ROWS, seed + 40u);
	L.c_wkv = (const uint16_t *)AllocFilled((uint64_t)D41_HEAD_DIM * D41_HIDDEN * 2u, 1u, seed + 42u);
	L.c_wgate = (const uint16_t *)AllocFilled((uint64_t)D41_HEAD_DIM * D41_HIDDEN * 2u, 1u, seed + 43u);
	L.c_norm = (const uint16_t *)AllocFilled(D41_HEAD_DIM * 2u, 1u, seed + 44u);
	L.idx_q_b = MakeLin(D41_LOCAL_INDEX_HEADS * D41_INDEX_DIM, D41_QLORA, seed + 46u);
	L.idx_wk = (const uint16_t *)AllocFilled((uint64_t)D41_INDEX_DIM * D41_HEAD_DIM * 2u, 1u, seed + 48u);
	L.idx_kn = (const uint16_t *)AllocFilled(D41_INDEX_DIM * 2u, 1u, seed + 49u);
	L.idx_wp = (const uint16_t *)AllocFilled((uint64_t)D41_LOCAL_INDEX_HEADS * D41_HIDDEN * 2u, 1u, seed + 50u);
	L.router = (const uint16_t *)AllocFilled((uint64_t)D41_EXPERTS * D41_HIDDEN * 2u, 1u, seed + 52u);
	L.router_bias = (const float *)AllocFilled(D41_EXPERTS * 4u, 2u, seed + 53u);
	L.sh_w1 = MakeLin(D41_MOE_INTER, D41_HIDDEN, seed + 54u);
	L.sh_w3 = MakeLin(D41_MOE_INTER, D41_HIDDEN, seed + 56u);
	L.sh_w2 = MakeLin(D41_HIDDEN, D41_MOE_INTER, seed + 58u);
	return L;
}

int main(void)
{
	static_assert(D41_TP == 16u && D41_ROWS == 1 && D41_WHOLE == 1, "compiled for ROWS16");
	int sms = 0;
	CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));

	D41ModelState *st = (D41ModelState *)calloc(1u, sizeof(D41ModelState));
	RoundRec rctx;
	memset(&rctx, 0, sizeof(rctx));
	SparkStageRunnerServices svc;
	memset(&svc, 0, sizeof(svc));
	svc.context = &rctx;
	svc.round = MockRound;
	st->services = &svc;
	st->tp_degree = D41_TP;
	st->tp_rank = 0u;
	st->max_rows = 512u;
	st->slots = 8u;
	st->kv_attached = 1u;
	st->context_slots = 8u;
	st->max_positions = 1024u;
	st->moe_whole = 1u;
	D41SharedSlice(0u, &st->shared_first, &st->shared_width);
	st->shard.degree = D41_KV_TP;
	st->shard.rank = 0u;
	st->shard.grain = D41_GRAIN;
	st->local_score_stride = 1024u;
	st->select_rows = 512u;
	st->global_score_stride = 1024u;
	st->topk_entries = 2u * 512u * LmTopkExactCandidateEntries(1024u, D41_INDEX_TOPK, LM_TOPK_EXACT_CHUNK);
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

	uint8_t *init_rec = (uint8_t *)AllocFilled((uint64_t)8u * st->recurrent_bytes, 1u, 0x111u);
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
		const uint32_t pages = 8u;
		std::vector<uint32_t> pt(8u * pages);
		for ( uint32_t s = 0u; s < 8u * pages; ++s ) pt[s] = s;
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

	const uint64_t c13 = (uint64_t)(D41_HIDDEN / 32u) * (2u * D41_MOE_INTER / 16u) * 272u;
	const uint64_t c2 = (uint64_t)(D41_MOE_INTER / 32u) * (D41_HIDDEN / 16u) * 272u;
	const uint8_t *all_w13 = (const uint8_t *)AllocFilled(c13 * D41_EXPERTS, 5u, 0x600u);
	const uint8_t *all_w2 = (const uint8_t *)AllocFilled(c2 * D41_EXPERTS, 5u, 0x601u);

	D41Layer &L0 = st->layers[0];
	D41Layer &L2 = st->layers[2];
	L0 = MakeAttnMoeLayer(0x1000u);
	L2 = MakeAttnMoeLayer(0x2000u);

	const uint64_t table = (uint64_t)D41_LAYERS * D41_EXPERTS;
	std::vector<uint8_t> h_owner(table, 0u), h_backup(D41_BACKUP_COPIES * table, 255u);
	std::vector<std::vector<uint8_t>> h_group_rank(D41_TP, std::vector<uint8_t>(table, 255u));
	std::vector<uint8_t *> egroup_rank(D41_TP);
	std::vector<uint8_t *> bk_w13_rank(D41_TP), bk_w2_rank(D41_TP);
	for ( uint32_t layer = 0u; layer < D41_LAYERS; ++layer )
	{
		const uint32_t copies = D41BackupCopies(layer);
		for ( uint32_t e = 0u; e < D41_EXPERTS; ++e )
		{
			const uint32_t ow = e / D41_LOCAL_EXPERTS, sl = e % D41_LOCAL_EXPERTS;
			h_owner[(uint64_t)layer * D41_EXPERTS + e] = (uint8_t)ow;
			h_group_rank[ow][(uint64_t)layer * D41_EXPERTS + e] = (uint8_t)sl;
			for ( uint32_t cp = 0u; cp < copies; ++cp )
				h_backup[((uint64_t)layer * D41_BACKUP_COPIES + cp) * D41_EXPERTS + e] = (uint8_t)D41BackupRank(cp, ow, sl);
		}
		uint32_t placed[D41_BACKUP_COPIES][D41_TP];
		memset(placed, 0, sizeof(placed));
		for ( uint32_t cp = 0u; cp < copies; ++cp )
			for ( uint32_t e = 0u; e < D41_EXPERTS; ++e )
			{
				const uint32_t holder = h_backup[((uint64_t)layer * D41_BACKUP_COPIES + cp) * D41_EXPERTS + e];
				h_group_rank[holder][(uint64_t)layer * D41_EXPERTS + e] = (uint8_t)(D41_LOCAL_EXPERTS + cp * D41_LOCAL_EXPERTS + placed[cp][holder]);
				placed[cp][holder]++;
			}
	}
	uint8_t *d_owner = 0, *d_backup = 0;
	CHECK(cudaMalloc(&d_owner, table));
	CHECK(cudaMemcpy(d_owner, h_owner.data(), table, cudaMemcpyHostToDevice));
	CHECK(cudaMalloc(&d_backup, D41_BACKUP_COPIES * table));
	CHECK(cudaMemcpy(d_backup, h_backup.data(), D41_BACKUP_COPIES * table, cudaMemcpyHostToDevice));
	st->expert_owner = d_owner;
	st->expert_backup = d_backup;

	std::vector<D41Fp8Linear> sh_w1_s(D41_TP), sh_w3_s(D41_TP), sh_w2_s(D41_TP);
	std::vector<D41Fp8Linear> sh2_w1_s(D41_TP), sh2_w3_s(D41_TP), sh2_w2_s(D41_TP);
	for ( uint32_t r = 0u; r < D41_TP; ++r )
	{
		uint32_t f, w;
		D41SharedPart(r, D41_TP, D41_MOE_INTER / D41_SHARED_GRANULE, D41_SHARED_GRANULE, &f, &w);
		sh_w1_s[r] = SliceRows(L0.sh_w1, f, w, D41_HIDDEN);
		sh_w3_s[r] = SliceRows(L0.sh_w3, f, w, D41_HIDDEN);
		sh_w2_s[r] = SliceCols(L0.sh_w2, D41_HIDDEN, D41_MOE_INTER, f, w);
		sh2_w1_s[r] = SliceRows(L2.sh_w1, f, w, D41_HIDDEN);
		sh2_w3_s[r] = SliceRows(L2.sh_w3, f, w, D41_HIDDEN);
		sh2_w2_s[r] = SliceCols(L2.sh_w2, D41_HIDDEN, D41_MOE_INTER, f, w);
		CHECK(cudaMalloc(&egroup_rank[r], table));
		CHECK(cudaMemcpy(egroup_rank[r], h_group_rank[r].data(), table, cudaMemcpyHostToDevice));
		const uint32_t bk_cnt = D41_BACKUP_COPIES * D41_LOCAL_EXPERTS;
		CHECK(cudaMalloc(&bk_w13_rank[r], (uint64_t)bk_cnt * c13));
		CHECK(cudaMalloc(&bk_w2_rank[r], (uint64_t)bk_cnt * c2));
		for ( uint32_t ex = 0u; ex < D41_EXPERTS; ++ex )
		{
			const uint8_t g = h_group_rank[r][ex];
			if ( g >= D41_LOCAL_EXPERTS && g < D41_HELD_EXPERTS )
			{
				const uint32_t slot = g - D41_LOCAL_EXPERTS;
				CHECK(cudaMemcpy(bk_w13_rank[r] + (uint64_t)slot * c13, all_w13 + (uint64_t)ex * c13, c13, cudaMemcpyDeviceToDevice));
				CHECK(cudaMemcpy(bk_w2_rank[r] + (uint64_t)slot * c2, all_w2 + (uint64_t)ex * c2, c2, cudaMemcpyDeviceToDevice));
			}
		}
	}
	st->backup_cells = bk_w13_rank[0];
	st->backup_filled = (1ull << 0u) | (1ull << 2u);

	uint16_t *init_streams = (uint16_t *)AllocFilled((uint64_t)256u * D41K_HC * D41_HIDDEN * 2u, 1u, 0x777u);
	float *init_pre = (float *)AllocFilled((uint64_t)256u * D41K_HC * 4u, 2u, 0x999u);
	uint32_t *rfault = DevU32({ 0u });
	float *f32_ref_buf = 0;
	CHECK(cudaMalloc(&f32_ref_buf, (uint64_t)256u * D41_HIDDEN * sizeof(float)));
	CHECK(cudaDeviceSynchronize());

	auto reset_kv = [&]()
	{
		CHECK(cudaMemcpy(st->recurrent, init_rec, (uint64_t)8u * st->recurrent_bytes, cudaMemcpyDeviceToDevice));
		for ( uint32_t c = 0u; c < 4u; ++c )
			CHECK(cudaMemcpy(kv_pools[c], kv_pools_init[c], kv_pool_sz[c], cudaMemcpyDeviceToDevice));
	};

	auto run_moe_16rank = [&](uint32_t nr, uint32_t row_inv, std::vector<uint16_t> &streams_out, std::vector<float> &moe_sum_out,
		std::vector<float> &f32_ref_sum_out, RoundRec &last_rec, double &rank0_us)
	{
		std::vector<std::vector<uint8_t>> rank_payloads(D41_TP);
		RoundRec r0_rec;
		memset(&r0_rec, 0, sizeof(r0_rec));
		SparkStageRunnerStep step;
		memset(&step, 0, sizeof(step));
		step.rows = nr;
		step.sequences = row_inv != 0u ? 1u : nr;
		step.commit = row_inv != 0u ? 0u : 1u;
		step.multiprocessors = (uint32_t)sms;
		step.route_fault = rfault;
		D41Cuda::Frame fr;
		memset(&fr, 0, sizeof(fr));
		fr.step = &step;
		fr.current = &step;
		fr.streams = st->streams;
		fr.pre_head = st->pre_head;

		const size_t elems = (size_t)nr * D41_HIDDEN;
		f32_ref_sum_out.assign(elems, 0.0f);
		std::vector<float> rank_f32(elems);
		const uint32_t groups0 = D41_LOCAL_EXPERTS + D41BackupCopies(0u) * D41_LOCAL_EXPERTS;

		for ( uint32_t r = 0u; r < D41_TP; ++r )
		{
			st->tp_rank = r;
			st->row_invariant = row_inv;
			st->own_rows = 0u;
			st->mix_collapsed = 0u;
			st->mx_valid = 0u;
			st->post_mode = 0u;
			st->expert_group = egroup_rank[r];
			L0.e_w13_cells = all_w13 + (uint64_t)r * D41_LOCAL_EXPERTS * c13;
			L0.e_w2_cells = all_w2 + (uint64_t)r * D41_LOCAL_EXPERTS * c2;
			L0.e_w13_backup = bk_w13_rank[r];
			L0.e_w2_backup = bk_w2_rank[r];
			L0.sh_w1_split = sh_w1_s[r];
			L0.sh_w3_split = sh_w3_s[r];
			L0.sh_w2_split = sh_w2_s[r];
			CHECK(cudaMemcpy(st->streams, init_streams, (uint64_t)nr * D41K_HC * D41_HIDDEN * 2u, cudaMemcpyDeviceToDevice));
			CHECK(cudaMemcpy(st->pre_attn, init_pre, (uint64_t)nr * D41K_HC * 4u, cudaMemcpyDeviceToDevice));
			CHECK(cudaMemcpy(st->pre_head, init_pre, (uint64_t)nr * D41K_HC * 4u, cudaMemcpyDeviceToDevice));
			rctx = RoundRec{};
			rctx.capture_only = 1;
			REQ(D41Cuda::HyperMix(st, fr, 0u, D41_FFN_PART) == SPARK_STATUS_OK, "hmF");
			REQ(D41Cuda::CollapseNorm(st, fr, 0u, D41_FFN_PART) == SPARK_STATUS_OK, "cnF");
			REQ(D41Moe(st, &L0, &step, 0) == SPARK_STATUS_OK, "moe");
			rank_payloads[r] = rctx.captured;
			if ( r == 0u ) r0_rec = rctx;
			if ( D41SharedSplitStep(st, nr) != 0u )
			{
				D41MoeFinalizeF32Kernel<<<dim3(D41Blocks(D41_HIDDEN / 8u), nr), D41_THREADS>>>(st->expert_slice, st->route_packed_row,
					st->route_expert, (const float *)st->shared_out, f32_ref_buf, D41_TOPK, groups0, D41_HIDDEN);
				CHECK(cudaDeviceSynchronize());
				CHECK(cudaMemcpy(rank_f32.data(), f32_ref_buf, elems * sizeof(float), cudaMemcpyDeviceToHost));
				for ( size_t i = 0u; i < elems; ++i ) f32_ref_sum_out[i] += rank_f32[i];
			}
			else
			{
				const uint16_t *bp = (const uint16_t *)rank_payloads[r].data();
				for ( size_t i = 0u; i < elems; ++i ) f32_ref_sum_out[i] += Bf16ToF(bp[i]);
			}
		}
		last_rec = r0_rec;

		moe_sum_out.assign(elems, 0.0f);
		std::vector<uint8_t> combined;
		if ( r0_rec.op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16 )
		{
			combined.resize(elems * 2u);
			uint16_t *out16 = (uint16_t *)combined.data();
			for ( size_t i = 0u; i < elems; ++i )
			{
				float acc = 0.0f;
				for ( uint32_t r = 0u; r < D41_TP; ++r )
					acc += Bf16ToF(((const uint16_t *)rank_payloads[r].data())[i]);
				out16[i] = FToBf16(acc);
				moe_sum_out[i] = Bf16ToF(out16[i]);
			}
		}
		else
			REQ(false, "ROWS16 MoE must use ALL_REDUCE_SUM_BF16");

		st->tp_rank = 0u;
		st->row_invariant = row_inv;
		st->own_rows = 0u;
		st->mix_collapsed = 0u;
		st->mx_valid = 0u;
		st->post_mode = 0u;
		st->expert_group = egroup_rank[0];
		L0.e_w13_cells = all_w13;
		L0.e_w2_cells = all_w2;
		L0.e_w13_backup = bk_w13_rank[0];
		L0.e_w2_backup = bk_w2_rank[0];
		L0.sh_w1_split = sh_w1_s[0];
		L0.sh_w3_split = sh_w3_s[0];
		L0.sh_w2_split = sh_w2_s[0];
		CHECK(cudaMemcpy(st->streams, init_streams, (uint64_t)nr * D41K_HC * D41_HIDDEN * 2u, cudaMemcpyDeviceToDevice));
		CHECK(cudaMemcpy(st->pre_attn, init_pre, (uint64_t)nr * D41K_HC * 4u, cudaMemcpyDeviceToDevice));
		CHECK(cudaMemcpy(st->pre_head, init_pre, (uint64_t)nr * D41K_HC * 4u, cudaMemcpyDeviceToDevice));
		rctx = RoundRec{};
		rctx.inject_ptr = combined.data();
		rctx.inject_bytes = combined.size();
		REQ(D41Cuda::HyperMix(st, fr, 0u, D41_FFN_PART) == SPARK_STATUS_OK, "hmF2");
		REQ(D41Cuda::CollapseNorm(st, fr, 0u, D41_FFN_PART) == SPARK_STATUS_OK, "cnF2");
		REQ(D41Moe(st, &L0, &step, 0) == SPARK_STATUS_OK, "moe2");
		REQ(D41Cuda::HyperPost(st, fr, D41_FFN_PART) == SPARK_STATUS_OK, "hpF2");
		CHECK(cudaDeviceSynchronize());
		streams_out.resize((size_t)nr * D41K_HC * D41_HIDDEN);
		CHECK(cudaMemcpy(streams_out.data(), st->streams, streams_out.size() * 2u, cudaMemcpyDeviceToHost));

		cudaEvent_t ev0, ev1;
		CHECK(cudaEventCreate(&ev0));
		CHECK(cudaEventCreate(&ev1));
		rctx = RoundRec{};
		for ( int w = 0; w < 5; ++w )
		{
			st->mix_collapsed = 0u; st->mx_valid = 0u; st->post_mode = 0u;
			D41Cuda::HyperMix(st, fr, 0u, D41_FFN_PART);
			D41Cuda::CollapseNorm(st, fr, 0u, D41_FFN_PART);
			D41Moe(st, &L0, &step, 0);
			D41Cuda::HyperPost(st, fr, D41_FFN_PART);
		}
		CHECK(cudaEventRecord(ev0));
		const int reps = 20;
		for ( int rep = 0; rep < reps; ++rep )
		{
			st->mix_collapsed = 0u; st->mx_valid = 0u; st->post_mode = 0u;
			D41Cuda::HyperMix(st, fr, 0u, D41_FFN_PART);
			D41Cuda::CollapseNorm(st, fr, 0u, D41_FFN_PART);
			D41Moe(st, &L0, &step, 0);
			D41Cuda::HyperPost(st, fr, D41_FFN_PART);
		}
		CHECK(cudaEventRecord(ev1));
		CHECK(cudaEventSynchronize(ev1));
		float ms = 0.0f;
		CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
		rank0_us = (double)ms * 1000.0 / (double)reps;
		cudaEventDestroy(ev0);
		cudaEventDestroy(ev1);
	};

	const size_t row_el = D41K_HC * D41_HIDDEN;
	std::vector<std::vector<uint16_t>> b1_lanes(12u);
	for ( uint32_t lane = 0u; lane < 12u; ++lane )
	{
		uint16_t *saved_init = init_streams;
		float *saved_pre = init_pre;
		init_streams = saved_init + (uint64_t)lane * row_el;
		init_pre = saved_pre + (uint64_t)lane * D41K_HC;
		std::vector<float> m_sum, f32_sum;
		RoundRec rec;
		double us = 0.0;
		run_moe_16rank(1u, 0u, b1_lanes[lane], m_sum, f32_sum, rec, us);
		REQ(rec.op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, "B1 uses AR_BF16");
		init_streams = saved_init;
		init_pre = saved_pre;
	}

	for ( uint32_t nr : { 4u, 8u } )
	{
		std::vector<uint16_t> out;
		std::vector<float> m_sum, f32_sum;
		RoundRec rec;
		double us = 0.0;
		run_moe_16rank(nr, 0u, out, m_sum, f32_sum, rec, us);
		REQ(rec.op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, "B4/B8 uses AR_BF16");
		for ( uint32_t lane = 0u; lane < nr; ++lane )
		{
			char label[64];
			snprintf(label, sizeof(label), "ROWS16 balanced MoE 5c B1==B%u_lane%u", nr, lane);
			CheckExact(label, out.data() + (size_t)lane * row_el, b1_lanes[lane].data(), row_el);
		}
	}
	for ( uint32_t vr : { 6u, 8u, 10u, 12u } )
	{
		std::vector<uint16_t> out;
		std::vector<float> m_sum, f32_sum;
		RoundRec rec;
		double us = 0.0;
		run_moe_16rank(vr, 1u, out, m_sum, f32_sum, rec, us);
		REQ(rec.op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, "ver uses AR_BF16");
		for ( uint32_t lane = 0u; lane < vr; ++lane )
		{
			char label[64];
			snprintf(label, sizeof(label), "ROWS16 balanced MoE ver%u_row%u==B1", vr, lane);
			CheckExact(label, out.data() + (size_t)lane * row_el, b1_lanes[lane].data(), row_el);
		}
	}
	printf("PASS ROWS16 balanced MoE 5c (ALL lanes B4[0..3], B8[0..7], ver6/8/10/12 == B1: 0 bit diffs)\n");

	auto run_full_layer_rank0 = [&](uint32_t layer_idx, uint32_t nr, const std::vector<uint32_t> &pos, const std::vector<uint32_t> &seq_ids,
		uint32_t commit, uint32_t row_inv, const uint16_t *in_st, const float *in_pr, std::vector<uint16_t> &out)
	{
		reset_kv();
		D41Layer &L = st->layers[layer_idx];
		SparkStageRunnerStep step;
		memset(&step, 0, sizeof(step));
		step.rows = nr;
		step.sequences = row_inv != 0u ? 1u : nr;
		step.commit = commit;
		step.multiprocessors = (uint32_t)sms;
		std::vector<uint32_t> len(8u, 1u), beg(step.sequences + 1u), idx(nr), rec(8u);
		for ( uint32_t s = 0u; s < 8u; ++s ) rec[s] = s;
		if ( row_inv != 0u )
		{
			len[seq_ids[0]] = pos[nr - 1u] + 1u;
			rec[0] = seq_ids[0];
			beg = { 0u, nr };
			for ( uint32_t i = 0u; i < nr; ++i ) idx[i] = i;
		}
		else
		{
			for ( uint32_t i = 0u; i < nr; ++i )
			{
				len[seq_ids[i]] = pos[i] + 1u;
				rec[i] = seq_ids[i];
				beg[i] = i;
				idx[i] = i;
			}
			beg[nr] = nr;
		}
		step.positions = DevU32(pos);
		step.context_length = DevU32(len);
		step.sequence_of_row = DevU32(seq_ids);
		step.sequence_row_begin = DevU32(beg);
		step.sequence_row_indices = DevU32(idx);
		step.recurrent_index = DevU32(rec);
		step.route_fault = rfault;
		st->tp_rank = 0u;
		st->row_invariant = row_inv;
		st->own_rows = 0u;
		st->step_score_stride = 1024u;
		st->step_ratio2_stride = 1024u;
		st->compress_done = 0u;
		st->mix_collapsed = 0u;
		st->mx_valid = 0u;
		st->post_mode = 0u;
		st->expert_group = egroup_rank[0];
		L.e_w13_cells = all_w13;
		L.e_w2_cells = all_w2;
		L.e_w13_backup = bk_w13_rank[0];
		L.e_w2_backup = bk_w2_rank[0];
		L.sh_w1_split = layer_idx == 0u ? sh_w1_s[0] : sh2_w1_s[0];
		L.sh_w3_split = layer_idx == 0u ? sh_w3_s[0] : sh2_w3_s[0];
		L.sh_w2_split = layer_idx == 0u ? sh_w2_s[0] : sh2_w2_s[0];
		LM_LAUNCH((LmRowLanesKernel), step.sequences, 256u, 0, (SparkGpuStream)0, step.sequence_row_begin, step.sequence_row_indices, st->row_lane, st->row_ordinal);
		LM_LAUNCH((D41CompressedEmitPositionKernel), 1u, 256u, 0, (SparkGpuStream)0, step.positions, st->bound_ratio2, nr, 2u);
		LM_LAUNCH((D41CompressedEmitPositionKernel), 1u, 256u, 0, (SparkGpuStream)0, step.positions, st->bound_ratio1, nr, 1u);
		LM_LAUNCH((D41CompressedLengthKernel), 1u, 256u, 0, (SparkGpuStream)0, step.context_length, st->length_ratio2, 8u, 2u);
		LM_LAUNCH((D41CompressedLengthKernel), 1u, 256u, 0, (SparkGpuStream)0, step.context_length, st->length_ratio1, 8u, 1u);
		LM_LAUNCH((D41RopeTableKernel), nr, D41K_ROPE_PAIRS, 0, (SparkGpuStream)0, step.positions, st->full_frequency, st->rope_sc_full, st->rope_sc_compress, nr);
		CHECK(cudaMemcpy(st->streams, in_st, (uint64_t)nr * row_el * 2u, cudaMemcpyDeviceToDevice));
		CHECK(cudaMemcpy(st->pre_init, in_pr, (uint64_t)nr * D41K_HC * 4u, cudaMemcpyDeviceToDevice));
		CHECK(cudaMemcpy(st->pre_attn, in_pr, (uint64_t)nr * D41K_HC * 4u, cudaMemcpyDeviceToDevice));
		CHECK(cudaMemcpy(st->pre_head, in_pr, (uint64_t)nr * D41K_HC * 4u, cudaMemcpyDeviceToDevice));
		D41Cuda::Frame fr;
		memset(&fr, 0, sizeof(fr));
		fr.step = &step;
		fr.current = &step;
		fr.streams = st->streams;
		fr.pre_head = st->pre_head;
		fr.selected = st->selected;
		rctx = RoundRec{};
		REQ(D41Cuda::HyperMix(st, fr, layer_idx, D41_ATTENTION_PART) == SPARK_STATUS_OK, "hmA");
		REQ(D41Cuda::CollapseNorm(st, fr, layer_idx, D41_ATTENTION_PART) == SPARK_STATUS_OK, "cnA");
		REQ(D41AttentionSpread(st, &L, layer_idx, &step, 0) == SPARK_STATUS_OK, "attnSpread");
		REQ(D41Cuda::HyperPost(st, fr, D41_ATTENTION_PART) == SPARK_STATUS_OK, "hpA");
		REQ(D41Cuda::HyperMix(st, fr, layer_idx, D41_FFN_PART) == SPARK_STATUS_OK, "hmF");
		REQ(D41Cuda::CollapseNorm(st, fr, layer_idx, D41_FFN_PART) == SPARK_STATUS_OK, "cnF");
		REQ(D41Moe(st, &L, &step, 0) == SPARK_STATUS_OK, "moeF");
		REQ(D41Cuda::HyperPost(st, fr, D41_FFN_PART) == SPARK_STATUS_OK, "hpF");
		CHECK(cudaDeviceSynchronize());
		out.resize((size_t)nr * row_el);
		CHECK(cudaMemcpy(out.data(), st->streams, out.size() * 2u, cudaMemcpyDeviceToHost));
		cudaFree((void *)step.positions);
		cudaFree((void *)step.context_length);
		cudaFree((void *)step.sequence_of_row);
		cudaFree((void *)step.sequence_row_begin);
		cudaFree((void *)step.sequence_row_indices);
		cudaFree((void *)step.recurrent_index);
	};

	for ( uint32_t layer_idx : { 0u, 2u } )
		for ( uint32_t P : { 200u, 640u } )
		{
			const std::vector<uint32_t> pos8 = { P, P + 3u, P + 11u, P + 17u, P + 29u, P + 43u, P + 59u, P + 71u };
			const std::vector<uint32_t> seq8 = { 0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u };
			std::vector<uint16_t> b4_out, b8_out;
			run_full_layer_rank0(layer_idx, 4u, { pos8.begin(), pos8.begin() + 4 }, { seq8.begin(), seq8.begin() + 4 }, 1u, 0u, init_streams, init_pre, b4_out);
			run_full_layer_rank0(layer_idx, 8u, pos8, seq8, 1u, 0u, init_streams, init_pre, b8_out);
			for ( uint32_t lane = 0u; lane < 8u; ++lane )
			{
				std::vector<uint16_t> ref1;
				run_full_layer_rank0(layer_idx, 1u, { pos8[lane] }, { seq8[lane] }, 1u, 0u,
					init_streams + (uint64_t)lane * row_el, init_pre + (uint64_t)lane * D41K_HC, ref1);
				char label[96];
				if ( lane < 4u )
				{
					snprintf(label, sizeof(label), "ROWS16 full L%u P=%u B1==B4_lane%u", layer_idx, P, lane);
					CheckExact(label, b4_out.data() + (size_t)lane * row_el, ref1.data(), row_el);
				}
				snprintf(label, sizeof(label), "ROWS16 full L%u P=%u B1==B8_lane%u", layer_idx, P, lane);
				CheckExact(label, b8_out.data() + (size_t)lane * row_el, ref1.data(), row_el);
			}
		}
	printf("PASS ROWS16 full layer (AttnSpread + balanced MoE) 5c L0/L2 P=200/640 B4[0..3] & B8[0..7] == B1: 0 bit diffs\n");

	const uint32_t batch_cases[] = { 1u, 4u, 8u, 12u, 16u, 32u, 48u, 64u, 256u };
	for ( uint32_t nr : batch_cases )
	{
		std::vector<uint16_t> new_out;
		std::vector<float> new_moe, ref_moe;
		RoundRec rec_new;
		double t_new = 0.0;
		run_moe_16rank(nr, 0u, new_out, new_moe, ref_moe, rec_new, t_new);
		REQ(rec_new.op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, "ROWS16 MoE must use ALL_REDUCE_SUM_BF16");
		const uint64_t main_wire = (uint64_t)(D41_TP - 1u) * (uint64_t)nr * D41_HIDDEN * (nr <= 64u ? 4u : 2u);
		double mdot = 0.0, mna = 0.0, mnb = 0.0, mdiff2 = 0.0;
		float mref_max = 0.0f, mnew_max = 0.0f;
		for ( size_t i = 0u; i < new_moe.size(); ++i )
		{
			const float a = new_moe[i], b = ref_moe[i];
			mnew_max = fmaxf(mnew_max, fabsf(a));
			mref_max = fmaxf(mref_max, fabsf(b));
			mdot += (double)a * (double)b;
			mna += (double)a * (double)a;
			mnb += (double)b * (double)b;
			mdiff2 += (double)(a - b) * (double)(a - b);
		}
		REQ(mref_max > 0.1f && mnew_max > 0.1f, "non-zero MoE output");
		const double moe_cos = mdot / (sqrt(mna) * sqrt(mnb));
		const double moe_rel_l2 = sqrt(mdiff2 / mnb);
		REQ(moe_cos > 0.9999 && moe_rel_l2 < 0.01, "numerical parity vs FP32 reference");
		printf("ROWS16 B=%-3u op=AR_BF16 wire_old=%7.2f KB -> wire_new=%6.2f KB (%4.1fx cut) | 40L_wire_100G: %6.2f ms -> %5.2f ms | gpu_1L: %6.1f us | moe_cos=%.7f rel_l2=%.5f\n",
			nr, (double)main_wire / 1024.0, (double)rec_new.wire_egress_bytes / 1024.0,
			(double)main_wire / (double)rec_new.wire_egress_bytes,
			(double)main_wire * 40.0 / (10.5e9) * 1000.0,
			(double)rec_new.wire_egress_bytes * 40.0 / (10.5e9) * 1000.0,
			t_new, moe_cos, moe_rel_l2);
	}
	printf("PASS dsv41 rows16 moe exact\n");
	return 0;
}
