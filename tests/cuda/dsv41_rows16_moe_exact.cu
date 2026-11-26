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
		rec->wire_egress_bytes = 2u * (uint64_t)(D41_TP - 1u) * (bytes / D41_TP);
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
		for ( uint32_t r = 0u; r < D41_TP; ++r )
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
	st->max_positions = 512u;
	st->moe_whole = 1u;
	D41SharedSlice(0u, &st->shared_first, &st->shared_width);
	st->shard.degree = D41_KV_TP;
	st->shard.rank = 0u;
	st->shard.grain = D41_GRAIN;
	st->local_score_stride = 512u;
	st->select_rows = 512u;
	st->global_score_stride = 512u;
	st->topk_entries = 2u * 512u * LmTopkExactCandidateEntries(512u, D41_INDEX_TOPK, LM_TOPK_EXACT_CHUNK);
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
	for ( uint32_t c = 0u; c < 4u; ++c )
	{
		const uint32_t pslots = c < 3u ? D41Ratio2Kv::kPageSlots : D41Ratio1Kv::kPageSlots;
		const uint32_t pages = 4u;
		std::vector<uint32_t> pt(8u * pages);
		for ( uint32_t s = 0u; s < 8u * pages; ++s ) pt[s] = s;
		uint8_t *pool = (uint8_t *)AllocFilled((size_t)8u * pages * pslots * D41_SLOT_BYTES, 1u, 0x500u + c);
		LmKvShardView v;
		memset(&v, 0, sizeof(v));
		REQ(LmKvViewInitialize(&v.pages, pool, DevU32(pt), pages, 8u, 8u * pages, kverr) == 0, "kv");
		v.shard = st->shard;
		st->latent_views[c] = v;
		st->index_views[c] = v;
	}

	const uint64_t c13 = (uint64_t)(D41_HIDDEN / 32u) * (2u * D41_MOE_INTER / 16u) * 272u;
	const uint64_t c2 = (uint64_t)(D41_MOE_INTER / 32u) * (D41_HIDDEN / 16u) * 272u;
	const uint8_t *all_w13 = (const uint8_t *)AllocFilled(c13 * D41_EXPERTS, 5u, 0x600u);
	const uint8_t *all_w2 = (const uint8_t *)AllocFilled(c2 * D41_EXPERTS, 5u, 0x601u);

	D41Layer &L0 = st->layers[0];
	memset(&L0, 0, sizeof(L0));
	L0.attn_norm = (const uint16_t *)AllocFilled(D41_HIDDEN * 2u, 1u, 0x1001u);
	L0.ffn_norm = (const uint16_t *)AllocFilled(D41_HIDDEN * 2u, 1u, 0x1002u);
	L0.hc_attn_fn = (const float *)AllocFilled((uint64_t)LmHcSinkhornShape<D41K_HC>::kMix * D41K_HC * D41_HIDDEN * 4u, 2u, 0x1003u);
	L0.hc_attn_base = (const float *)AllocFilled(LmHcSinkhornShape<D41K_HC>::kMix * 4u, 2u, 0x1004u);
	L0.hc_attn_scale = (const float *)AllocFilled(4u * 4u, 2u, 0x1005u);
	L0.hc_ffn_fn = (const float *)AllocFilled((uint64_t)LmHcSinkhornShape<D41K_HC>::kMix * D41K_HC * D41_HIDDEN * 4u, 2u, 0x1006u);
	L0.hc_ffn_base = (const float *)AllocFilled(LmHcSinkhornShape<D41K_HC>::kMix * 4u, 2u, 0x1007u);
	L0.hc_ffn_scale = (const float *)AllocFilled(4u * 4u, 2u, 0x1008u);
	L0.router = (const uint16_t *)AllocFilled((uint64_t)D41_EXPERTS * D41_HIDDEN * 2u, 1u, 0x1042u);
	L0.router_bias = (const float *)AllocFilled(D41_EXPERTS * 4u, 2u, 0x1043u);
	L0.sh_w1 = MakeLin(D41_MOE_INTER, D41_HIDDEN, 0x1044u);
	L0.sh_w3 = MakeLin(D41_MOE_INTER, D41_HIDDEN, 0x1046u);
	L0.sh_w2 = MakeLin(D41_HIDDEN, D41_MOE_INTER, 0x1048u);

	std::vector<D41Fp8Linear> sh_w1_s(D41_TP), sh_w3_s(D41_TP), sh_w2_s(D41_TP);
	std::vector<uint8_t *> egroup_rank(D41_TP);
	for ( uint32_t r = 0u; r < D41_TP; ++r )
	{
		uint32_t f, w;
		D41SharedPart(r, D41_TP, D41_MOE_INTER / D41_SHARED_GRANULE, D41_SHARED_GRANULE, &f, &w);
		sh_w1_s[r] = SliceRows(L0.sh_w1, f, w, D41_HIDDEN);
		sh_w3_s[r] = SliceRows(L0.sh_w3, f, w, D41_HIDDEN);
		sh_w2_s[r] = SliceCols(L0.sh_w2, D41_HIDDEN, D41_MOE_INTER, f, w);
		std::vector<uint8_t> gmap(D41_EXPERTS, 255u);
		for ( uint32_t e = 0u; e < D41_LOCAL_EXPERTS; ++e )
			gmap[r * D41_LOCAL_EXPERTS + e] = (uint8_t)e;
		CHECK(cudaMalloc(&egroup_rank[r], D41_EXPERTS));
		CHECK(cudaMemcpy(egroup_rank[r], gmap.data(), D41_EXPERTS, cudaMemcpyHostToDevice));
	}

	uint16_t *init_streams = (uint16_t *)AllocFilled((uint64_t)256u * D41K_HC * D41_HIDDEN * 2u, 1u, 0x777u);
	float *init_pre = (float *)AllocFilled((uint64_t)256u * D41K_HC * 4u, 2u, 0x999u);
	uint32_t *rfault = DevU32({ 0u });
	CHECK(cudaDeviceSynchronize());

	auto run_moe_16rank = [&](uint32_t nr, uint32_t row_inv, uint32_t force_split, std::vector<uint16_t> &streams_out, std::vector<float> &moe_sum_out, RoundRec &last_rec, double &rank0_us)
	{
		std::vector<std::vector<uint8_t>> rank_payloads(D41_TP);
		RoundRec r0_rec;
		memset(&r0_rec, 0, sizeof(r0_rec));
		uint32_t saved_row_inv = row_inv;
		std::vector<uint32_t> pos(nr, 100u), len(8u, 101u), seq(nr, 0u), beg = { 0u, nr }, idx(nr), rec(8u, 0u);
		for ( uint32_t i = 0u; i < nr; ++i ) idx[i] = i;
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

		for ( uint32_t r = 0u; r < D41_TP; ++r )
		{
			st->tp_rank = r;
			st->row_invariant = force_split != 0u ? 1u : saved_row_inv;
			st->own_rows = 0u;
			st->mix_collapsed = 0u;
			st->mx_valid = 0u;
			st->post_mode = 0u;
			st->expert_group = egroup_rank[r];
			L0.e_w13_cells = all_w13 + (uint64_t)r * D41_LOCAL_EXPERTS * c13;
			L0.e_w2_cells = all_w2 + (uint64_t)r * D41_LOCAL_EXPERTS * c2;
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
		}
		last_rec = r0_rec;

		const size_t elems = (size_t)nr * D41_HIDDEN;
		moe_sum_out.assign(elems, 0.0f);
		std::vector<uint8_t> combined;
		if ( r0_rec.op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER )
		{
			combined.resize((size_t)D41_TP * r0_rec.payload_bytes);
			for ( uint32_t r = 0u; r < D41_TP; ++r )
			{
				memcpy(combined.data() + (size_t)r * r0_rec.payload_bytes, rank_payloads[r].data(), (size_t)r0_rec.payload_bytes);
				if ( r0_rec.payload_bytes == elems * 4u )
				{
					const float *fp = (const float *)rank_payloads[r].data();
					for ( size_t i = 0u; i < elems; ++i ) moe_sum_out[i] += fp[i];
				}
				else
				{
					const uint16_t *bp = (const uint16_t *)rank_payloads[r].data();
					for ( size_t i = 0u; i < elems; ++i ) moe_sum_out[i] += Bf16ToF(bp[i]);
				}
			}
		}
		else if ( r0_rec.op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16 )
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
			REQ(false, "unexpected collective op");

		st->tp_rank = 0u;
		st->row_invariant = force_split != 0u ? 1u : saved_row_inv;
		st->own_rows = 0u;
		st->mix_collapsed = 0u;
		st->mx_valid = 0u;
		st->post_mode = 0u;
		st->expert_group = egroup_rank[0];
		L0.e_w13_cells = all_w13;
		L0.e_w2_cells = all_w2;
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
		const int reps = 30;
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

	std::vector<uint16_t> b1_out, b8_out, ver_out;
	std::vector<float> b1_moe, b8_moe, ver_moe;
	RoundRec rec1, rec8, rec_v;
	double t1 = 0.0, t8 = 0.0, tv = 0.0;
	run_moe_16rank(1u, 0u, 0u, b1_out, b1_moe, rec1, t1);
	run_moe_16rank(8u, 0u, 0u, b8_out, b8_moe, rec8, t8);
	REQ(rec1.op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, "B1 must stay FP32 AG");
	REQ(rec8.op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, "B8 must stay FP32 AG for 5c");
	float b1_moe_max = 0.0f;
	for ( float v : b1_moe ) b1_moe_max = fmaxf(b1_moe_max, fabsf(v));
	REQ(b1_moe_max > 0.1f, "non-zero B1 MoE output");
	CheckExact("ROWS16 5c B1==B8_lane0", b8_out.data(), b1_out.data(), D41K_HC * D41_HIDDEN);
	const uint32_t ver_cases[] = { 6u, 8u, 10u, 12u };
	for ( uint32_t vr : ver_cases )
	{
		run_moe_16rank(vr, 1u, 0u, ver_out, ver_moe, rec_v, tv);
		REQ(rec_v.op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER, "ver row_inv=1 must stay FP32 AG");
		CheckExact("ROWS16 ver_row0==B1", ver_out.data(), b1_out.data(), D41K_HC * D41_HIDDEN);
	}
	printf("PASS ROWS16 5c (B1==B8_lane0: 0 bit diffs) & ver6/8/10/12_row0==B1 (0 bit diffs, moe_max=%.3f)\n", b1_moe_max);

	const uint32_t batch_cases[] = { 12u, 16u, 32u, 48u, 64u, 256u };
	for ( uint32_t nr : batch_cases )
	{
		std::vector<uint16_t> new_out, ref_out;
		std::vector<float> new_moe, ref_moe;
		RoundRec rec_new, rec_ref;
		double t_new = 0.0, t_ref = 0.0;
		run_moe_16rank(nr, 0u, 0u, new_out, new_moe, rec_new, t_new);
		run_moe_16rank(nr, 0u, 1u, ref_out, ref_moe, rec_ref, t_ref);
		REQ(rec_new.op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16, "B>=12 must use ALL_REDUCE_SUM_BF16");
		const uint64_t expected_wire = 2u * (uint64_t)(D41_TP - 1u) * ((uint64_t)nr * D41_HIDDEN * 2u / D41_TP);
		REQ(rec_new.wire_egress_bytes == expected_wire, "expected reduced wire");
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
		double sdot = 0.0, sna = 0.0, snb = 0.0;
		float s_max_abs = 0.0f;
		for ( size_t i = 0u; i < new_out.size(); ++i )
		{
			const float a = Bf16ToF(new_out[i]), b = Bf16ToF(ref_out[i]);
			s_max_abs = fmaxf(s_max_abs, fabsf(a - b));
			sdot += (double)a * (double)b;
			sna += (double)a * (double)a;
			snb += (double)b * (double)b;
		}
		const double stream_cos = sdot / (sqrt(sna) * sqrt(snb));
		REQ(moe_cos > 0.999 && moe_rel_l2 < 0.05 && stream_cos > 0.999, "numerical parity vs FP32 reference");
		printf("ROWS16 B=%-3u op=AR_BF16 wire_main=%7.2f KB -> wire_new=%6.2f KB (%4.1fx cut) | 40L_wire_100G: %6.2f ms -> %5.2f ms | gpu_1L: ref=%6.1f us new=%6.1f us | moe_cos=%.7f rel_l2=%.5f stream_cos=%.7f\n",
			nr, (double)main_wire / 1024.0, (double)rec_new.wire_egress_bytes / 1024.0,
			(double)main_wire / (double)rec_new.wire_egress_bytes,
			(double)main_wire * 40.0 / (10.5e9) * 1000.0,
			(double)rec_new.wire_egress_bytes * 40.0 / (10.5e9) * 1000.0,
			t_ref, t_new, moe_cos, moe_rel_l2, stream_cos);
	}
	printf("PASS dsv41 rows16 moe exact\n");
	return 0;
}
