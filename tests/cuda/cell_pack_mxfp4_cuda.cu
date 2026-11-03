#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <functional>

#include "runtime/gemm.cuh"
#include "inference/kernels/route.cuh"
#include "inference/kernels/skinny.cuh"
#include "inference/kernels/cell_mma.cuh"
#include "inference/kernels/cell_pack.cuh"
#include "inference/kernels/weight_codec.cuh"
#include "inference/kernels/moe_local.cuh"
#include "inference/kernels/formats/mxfp4.cuh"

#define EXPERTS 128u
#define LOCAL 8u
#define TOPK 6u
#define HIDDEN 1024u
#define INTER 768u

#define CHECK(call) do { cudaError_t e = (call); if ( e != cudaSuccess ) { printf("FAIL cuda %s: %s\n", #call, cudaGetErrorString(e)); exit(1); } } while (0)
#define LAUNCH(call) do { int32_t s_ = (call); if ( s_ != LM_LAUNCH_OK ) { printf("FAIL launch %s: %d\n", #call, s_); exit(1); } } while (0)

static uint64_t Hash(uint64_t v)
{
	v = v * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
	v = (v ^ (v >> 30u)) * 0xbf58476d1ce4e5b9ull;
	v = (v ^ (v >> 27u)) * 0x94d049bb133111ebull;
	return v ^ (v >> 31u);
}

static uint16_t Bf16(float f)
{
	uint32_t u;
	memcpy(&u, &f, 4);
	return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

static float FromBf16(uint16_t h)
{
	uint32_t u = (uint32_t)h << 16;
	float f;
	memcpy(&f, &u, 4);
	return f;
}

struct Matrix
{
	uint8_t *payload, *scale, *cells;
	uint32_t n, k;
};

static void MakeMatrix(Matrix *m, uint32_t n, uint32_t k, uint64_t seed)
{
	const uint64_t pbytes = (uint64_t)LOCAL * n * k / 2u, sbytes = (uint64_t)LOCAL * n * k / 32u;
	std::vector<uint8_t> p(pbytes), s(sbytes);
	for ( uint64_t i = 0; i < pbytes; ++i )
		p[i] = (uint8_t)Hash(seed + i);
	for ( uint64_t i = 0; i < sbytes; ++i )
		s[i] = (uint8_t)(118u + Hash(seed * 7u + i) % 9u);
	m->n = n;
	m->k = k;
	CHECK(cudaMalloc(&m->payload, pbytes));
	CHECK(cudaMalloc(&m->scale, sbytes));
	CHECK(cudaMalloc(&m->cells, LmCellPackBytes(LOCAL, n, k)));
	CHECK(cudaMemcpy(m->payload, p.data(), pbytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(m->scale, s.data(), sbytes, cudaMemcpyHostToDevice));
	LAUNCH(LmCellPackMxfp4(m->payload, m->scale, m->cells, LOCAL, n, k, 0));
	CHECK(cudaDeviceSynchronize());
}

static int failures;

static void Compare(const char *label, const uint16_t *a, const uint16_t *b, uint64_t elements)
{
	std::vector<uint16_t> x(elements), y(elements);
	CHECK(cudaMemcpy(x.data(), a, elements * 2u, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(y.data(), b, elements * 2u, cudaMemcpyDeviceToHost));
	double max_abs = 0.0, max_ref = 0.0, sum_sq = 0.0, ref_sq = 0.0;
	uint64_t same = 0;
	for ( uint64_t i = 0; i < elements; ++i )
	{
		const double u = FromBf16(x[i]), v = FromBf16(y[i]);
		same += x[i] == y[i];
		max_abs = fmax(max_abs, fabs(u - v));
		max_ref = fmax(max_ref, fabs(u));
		sum_sq += (u - v) * (u - v);
		ref_sq += u * u;
	}
	const double rel = ref_sq > 0 ? sqrt(sum_sq / ref_sq) : 0.0;
	const bool pass = elements > 0u && max_ref > 0.0 && rel < 1e-4;
	failures += pass ? 0 : 1;
	printf("%s %s elements=%llu bit_equal=%.6f max_abs=%.4g max_ref=%.4g rel_l2=%.3g\n", pass ? "PASS" : "FAIL", label,
		(unsigned long long)elements, (double)same / (double)elements, max_abs, max_ref, rel);
}

static void Segmented(const char *label, const Matrix *m, uint32_t parts, const uint16_t *activation, const uint32_t *offsets, const uint32_t *source,
	uint32_t routes, uint32_t packed, uint64_t rows, uint16_t *whole, uint16_t *split)
{
	const uint64_t tile_bytes = (uint64_t)(m->n / LM_SKINNY_CELL_NEURONS) * LM_SKINNY_CELL_ROWS * LM_SKINNY_CHUNK_BYTES;
	const uint64_t group_bytes = (uint64_t)(m->k / LM_SKINNY_CELL_TILE_K) * tile_bytes, part_bytes = group_bytes / parts;
	std::vector<uint8_t *> owned(parts);
	std::vector<uint16_t> x(rows * m->n), y(rows * m->n);
	const void *pieces[LM_CELL_MMA_MAX_SEGMENTS];
	uint64_t differ = 0u;
	for ( uint32_t p = 0; p < parts; ++p )
	{
		CHECK(cudaMalloc(&owned[p], part_bytes * LOCAL));
		CHECK(cudaMemcpy2D(owned[p], part_bytes, (const uint8_t *)m->cells + p * part_bytes, group_bytes, part_bytes, LOCAL, cudaMemcpyDeviceToDevice));
		pieces[p] = owned[p];
	}
	CHECK(cudaMemset(whole, 0, rows * m->n * 2u));
	CHECK(cudaMemset(split, 0xff, rows * m->n * 2u));
	LAUNCH(LmCellMmaExperts(m->cells, activation, whole, offsets, source, LOCAL, routes, packed, m->k, m->n, 32u, 0));
	LAUNCH(LmCellMmaExpertsSegmented(pieces, parts, activation, split, offsets, source, LOCAL, routes, packed, m->k, m->n, 32u, 0));
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaMemcpy(x.data(), whole, x.size() * 2u, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(y.data(), split, y.size() * 2u, cudaMemcpyDeviceToHost));
	for ( uint64_t i = 0; i < x.size(); ++i )
		differ += x[i] != y[i] ? 1u : 0u;
	failures += differ == 0u ? 0 : 1;
	printf("%s %s: %u K segments in separate buffers equal one packed weight, %llu of %llu outputs differ\n", differ == 0u ? "PASS" : "FAIL", label, parts,
		(unsigned long long)differ, (unsigned long long)x.size());
	for ( uint32_t p = 0; p < parts; ++p )
		cudaFree(owned[p]);
}

static void Columns(const char *label, const Matrix *m, uint32_t parts, const uint16_t *activation, const uint32_t *offsets, const uint32_t *source,
	uint32_t routes, uint32_t packed, uint64_t rows, uint16_t *whole, uint16_t *split)
{
	const uint64_t cell_bytes = (uint64_t)LM_SKINNY_CELL_ROWS * LM_SKINNY_CHUNK_BYTES, cells = m->n / LM_SKINNY_CELL_NEURONS, part_cells = cells / parts;
	const uint64_t lines = (uint64_t)LOCAL * (m->k / LM_SKINNY_CELL_TILE_K);
	std::vector<uint8_t *> owned(parts);
	std::vector<uint16_t> x(rows * m->n), y(rows * m->n);
	const void *pieces[LM_CELL_MMA_MAX_SEGMENTS];
	uint64_t differ = 0u;
	for ( uint32_t p = 0; p < parts; ++p )
	{
		CHECK(cudaMalloc(&owned[p], lines * part_cells * cell_bytes));
		CHECK(cudaMemcpy2D(owned[p], part_cells * cell_bytes, (const uint8_t *)m->cells + p * part_cells * cell_bytes, cells * cell_bytes, part_cells * cell_bytes, lines,
			cudaMemcpyDeviceToDevice));
		pieces[p] = owned[p];
	}
	CHECK(cudaMemset(whole, 0, rows * m->n * 2u));
	CHECK(cudaMemset(split, 0xff, rows * m->n * 2u));
	LAUNCH(LmCellMmaExperts(m->cells, activation, whole, offsets, source, LOCAL, routes, packed, m->k, m->n, 32u, 0));
	LAUNCH(LmCellMmaExpertsColumns(pieces, parts, activation, split, offsets, source, LOCAL, routes, packed, m->k, m->n, 32u, 0));
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaMemcpy(x.data(), whole, x.size() * 2u, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(y.data(), split, y.size() * 2u, cudaMemcpyDeviceToHost));
	for ( uint64_t i = 0; i < x.size(); ++i )
		differ += x[i] != y[i] ? 1u : 0u;
	failures += differ == 0u ? 0 : 1;
	printf("%s %s: %u output column segments in separate buffers equal one packed weight, %llu of %llu outputs differ\n", differ == 0u ? "PASS" : "FAIL", label, parts,
		(unsigned long long)differ, (unsigned long long)x.size());
	for ( uint32_t p = 0; p < parts; ++p )
		cudaFree(owned[p]);
}

static void Run(uint32_t tokens)
{
	const uint32_t routes = tokens * TOPK;
	cudaDeviceProp prop;
	CHECK(cudaGetDeviceProperties(&prop, 0));
	const uint32_t sms = (uint32_t)prop.multiProcessorCount;
	Matrix w1, w2;
	MakeMatrix(&w1, INTER, HIDDEN, 11u);
	MakeMatrix(&w2, HIDDEN, INTER, 23u);
	std::vector<uint32_t> experts(routes);
	for ( uint32_t t = 0; t < tokens; ++t )
		for ( uint32_t j = 0; j < TOPK; ++j )
			experts[t * TOPK + j] = (uint32_t)((Hash(1000u + t * 31u + j) % (EXPERTS / TOPK)) * TOPK + j) % EXPERTS;
	if ( tokens <= 8u )
		experts[0] = 3u;
	std::vector<uint16_t> act((uint64_t)tokens * HIDDEN), inter((uint64_t)routes * INTER);
	for ( uint64_t i = 0; i < act.size(); ++i )
		act[i] = Bf16((float)((int64_t)(Hash(5u + i) % 2001u) - 1000) / 400.0f);
	for ( uint64_t i = 0; i < inter.size(); ++i )
		inter[i] = Bf16((float)((int64_t)(Hash(9u + i) % 2001u) - 1000) / 400.0f);
	uint32_t *route_expert, *route_global, *offsets, *packed_row, *source, *prefix_up, *prefix_down;
	float *route_weight;
	uint16_t *d_act, *d_inter, *out_a, *out_b;
	CHECK(cudaMalloc(&route_expert, routes * 4u));
	CHECK(cudaMalloc(&route_global, routes * 4u));
	CHECK(cudaMalloc(&route_weight, routes * 4u));
	CHECK(cudaMalloc(&offsets, 4096u * 4u));
	CHECK(cudaMalloc(&packed_row, routes * 4u));
	CHECK(cudaMalloc(&source, routes * 4u));
	CHECK(cudaMalloc(&prefix_up, 4096u * 4u));
	CHECK(cudaMalloc(&prefix_down, 4096u * 4u));
	CHECK(cudaMalloc(&d_act, act.size() * 2u));
	CHECK(cudaMalloc(&d_inter, inter.size() * 2u));
	CHECK(cudaMalloc(&out_a, (uint64_t)routes * HIDDEN * 2u));
	CHECK(cudaMalloc(&out_b, (uint64_t)routes * HIDDEN * 2u));
	CHECK(cudaMemcpy(route_expert, experts.data(), routes * 4u, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_act, act.data(), act.size() * 2u, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_inter, inter.data(), inter.size() * 2u, cudaMemcpyHostToDevice));
	CHECK(cudaMemset(route_weight, 0, routes * 4u));
	LmRouteLocalKernel<<<(routes + 255u) / 256u, 256u>>>(route_expert, route_weight, route_global, routes, 0u, LOCAL);
	LAUNCH((LmRouteBuild<256u, LOCAL + 1u>(route_expert, tokens, routes, TOPK, offsets, packed_row, source, INTER, HIDDEN, 128u,
		prefix_up, prefix_down, 0)));
	CHECK(cudaDeviceSynchronize());
	std::vector<uint32_t> off(LOCAL + 1u);
	CHECK(cudaMemcpy(off.data(), offsets, (LOCAL + 1u) * 4u, cudaMemcpyDeviceToHost));
	uint32_t peak = 0;
	for ( uint32_t g = 0; g < LOCAL; ++g )
		peak = off[g + 1] - off[g] > peak ? off[g + 1] - off[g] : peak;
	const uint32_t local_rows = off[LOCAL];
	const LmScaleTensor s1 = LmWeightCodecScaleTensor<SPARK_WEIGHT_CODEC_MXFP4_E2M1>(w1.scale, LOCAL, INTER, HIDDEN);
	const LmScaleTensor s2 = LmWeightCodecScaleTensor<SPARK_WEIGHT_CODEC_MXFP4_E2M1>(w2.scale, LOCAL, HIDDEN, INTER);
	auto up_current = [&](uint16_t *out) -> int32_t {
		if ( tokens <= 16u )
			return LmSkinnyGroupedExperts<LmMxfp4>(w1.payload, s1, d_act, out, offsets, source, LOCAL, routes, 0u, HIDDEN, INTER, 0);
		LmGemmArguments gemm;
		memset(&gemm, 0, sizeof(gemm));
		gemm.scale_a = LmScaleTensorNone();
		gemm.scale_b = s1;
		gemm.prefix_built = 1u;
		gemm.group_row_offset = offsets;
		gemm.group_tile_prefix = prefix_up;
		gemm.source_row_map = source;
		gemm.source_row_count = tokens;
		gemm.output_bf16 = out;
		return LmGemmWeightOnlyIndirectLaunch<LmMxfp4,128u,2u,8u>(&gemm, d_act, w1.payload, routes, tokens, TOPK, LOCAL, HIDDEN, INTER, sms, 0);
	};
	auto down_current = [&](uint16_t *out) -> int32_t {
		if ( tokens <= 16u )
			return LmSkinnyGroupedExperts<LmMxfp4>(w2.payload, s2, d_inter, out, offsets, source, LOCAL, routes, 1u, INTER, HIDDEN, 0);
		LmGemmArguments gemm;
		memset(&gemm, 0, sizeof(gemm));
		gemm.scale_a = LmScaleTensorNone();
		gemm.scale_b = s2;
		gemm.prefix_built = 1u;
		gemm.group_row_offset = offsets;
		gemm.group_tile_prefix = prefix_down;
		gemm.output_bf16 = out;
		return LmGemmWeightOnlyLaunch<LmMxfp4,128u,2u,8u>(&gemm, d_inter, w2.payload, routes, tokens, TOPK, LOCAL, INTER, HIDDEN, sms, true, 0);
	};
	auto up_cells = [&](uint16_t *out) -> int32_t {
		if ( tokens <= 16u )
			return LmSkinnyCellExperts(w1.cells, d_act, out, offsets, source, LOCAL, routes, peak, 0u, HIDDEN, INTER, 32u, 0);
		return LmCellMmaExperts(w1.cells, d_act, out, offsets, source, LOCAL, routes, 0u, HIDDEN, INTER, 32u, 0);
	};
	auto down_cells = [&](uint16_t *out) -> int32_t {
		if ( tokens <= 16u )
			return LmSkinnyCellExperts(w2.cells, d_inter, out, offsets, 0, LOCAL, routes, peak, 1u, INTER, HIDDEN, 32u, 0);
		return LmCellMmaExperts(w2.cells, d_inter, out, offsets, 0, LOCAL, routes, 1u, INTER, HIDDEN, 32u, 0);
	};
	struct Arm { const char *name; std::function<int32_t(uint16_t *)> run; uint16_t *out; uint32_t n; };
	Arm arms[4] = {{"up_current", up_current, out_a, INTER}, {"up_cells", up_cells, out_b, INTER},
		{"down_current", down_current, out_a, HIDDEN}, {"down_cells", down_cells, out_b, HIDDEN}};
	for ( uint32_t a = 0; a < 4u; ++a )
	{
		CHECK(cudaMemset(arms[a].out, 0, (uint64_t)routes * arms[a].n * 2u));
		LAUNCH(arms[a].run(arms[a].out));
		CHECK(cudaDeviceSynchronize());
		if ( a % 2u == 1u )
			Compare(a == 1u ? "up cells vs verbatim" : "down cells vs verbatim", arms[a - 1].out, arms[a].out, (uint64_t)local_rows * arms[a].n);
	}
	if ( tokens > 16u )
	{
		Segmented("up", &w1, 4u, d_act, offsets, source, routes, 0u, local_rows, out_a, out_b);
		Segmented("down", &w2, 4u, d_inter, offsets, 0, routes, 1u, local_rows, out_a, out_b);
		Columns("up", &w1, 2u, d_act, offsets, source, routes, 0u, local_rows, out_a, out_b);
		Columns("down", &w2, 4u, d_inter, offsets, 0, routes, 1u, local_rows, out_a, out_b);
	}
	cudaFree(w1.payload); cudaFree(w1.scale); cudaFree(w1.cells); cudaFree(w2.payload); cudaFree(w2.scale); cudaFree(w2.cells);
	cudaFree(route_expert); cudaFree(route_global); cudaFree(route_weight); cudaFree(offsets); cudaFree(packed_row); cudaFree(source);
	cudaFree(prefix_up); cudaFree(prefix_down); cudaFree(d_act); cudaFree(d_inter); cudaFree(out_a); cudaFree(out_b);
}

int main(void)
{
	const uint32_t rows[] = {1u, 8u, 16u, 64u, 512u, 2731u, 4096u};
	for ( uint32_t i = 0; i < sizeof(rows) / sizeof(rows[0]); ++i )
		Run(rows[i]);
	printf("%s cell_pack_mxfp4 failures=%d\n", failures == 0 ? "PASS" : "FAIL", failures);
	return failures == 0 ? 0 : 1;
}
