#include "modules/solar_open2_tp16_resident_decode_stage/source/spark_solar_open2_tp16_stage_model.cu"
#include "modules/ling3_resident_decode_stage/source/spark_ling3_stage_model.cu"
#include "sparkpipe/spark_model_serving_adapter.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "tests/cuda/verify_frames_common.cuh"

static constexpr uint32_t kBenchMaxRows = 512u;
static constexpr uint32_t kBenchSlots = 2u;
static constexpr uint32_t kBenchPages = 128u;
static constexpr uint32_t kBenchWarmup = 8u;
static constexpr uint32_t kBenchIters = 32u;

struct SolarOpen2Bench16 : SolarOpen2Geometry<16u>
{
	static constexpr uint32_t kLayers = 4u;
	static constexpr uint32_t kFullLayers = 48u;
	static constexpr uint32_t kFullAttnLayers = 12u;
	static constexpr uint32_t kFullKdaLayers = 36u;
	static constexpr bool kFp8Weights = false;
	static const char *Tag(void) { return "solar-open-2-tp16"; }
};

struct Ling3Bench16 : Ling3Geometry<16u>
{
	static constexpr uint32_t kLayers = 6u;
	static constexpr uint32_t kDenseLayers = 0u;
	static constexpr uint32_t kFullLayers = 42u;
	static constexpr uint32_t kFullAttnLayers = 7u;
	static constexpr uint32_t kFullKdaLayers = 35u;
	static constexpr bool kFp8Weights = false;
	static const char *Tag(void) { return "ling-3.0-flash-tp16"; }
};

struct Ling32AirBench16 : Ling3Bench16
{
	static constexpr bool kFp8Weights = true;
	static const char *Tag(void) { return "ling-3.2-air-fp8-tp16"; }
};

struct BenchBus
{
	uint32_t hidden;
	uint32_t tp_degree;
};

static SparkStatus BenchMockRound(void *context, void *stream_ptr, uint32_t op, uint32_t stage, uint32_t rows, uint32_t row_elements,
	const void *local, void *full)
{
	const BenchBus *bus = (const BenchBus *)context;
	cudaStream_t stream = (cudaStream_t)stream_ptr;
	const uint64_t block = (uint64_t)rows * (row_elements != 0u ? row_elements : bus->hidden) * sizeof(uint16_t);
	if ( stage != SPARK_STAGE_RUNNER_ROUND_ALL )
		return SPARK_STATUS_UNSUPPORTED;
	if ( op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16 )
	{
		if ( local == full )
			return SPARK_STATUS_OK;
		return cudaMemcpyAsync(full, local, block, cudaMemcpyDeviceToDevice, stream) == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
	}
	if ( op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER )
	{
		for ( uint32_t r = 0u; r < bus->tp_degree; ++r )
			if ( cudaMemcpyAsync((uint8_t *)full + (uint64_t)r * block, local, block, cudaMemcpyDeviceToDevice, stream) != cudaSuccess )
				return SPARK_STATUS_IO_ERROR;
		return SPARK_STATUS_OK;
	}
	if ( op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL )
	{
		return cudaMemcpyAsync(full, local, (uint64_t)bus->tp_degree * block, cudaMemcpyDeviceToDevice, stream) == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
	}
	return SPARK_STATUS_UNSUPPORTED;
}

static LingWeight BenchBf16Weight(uint64_t rows, uint64_t cols, std::vector<void *> *allocs)
{
	LingWeight w;
	w.weight = Matrix(rows, cols);
	w.scale = 0;
	allocs->push_back((void *)w.weight);
	return w;
}

static LingWeight BenchFp8Weight(uint64_t rows, uint64_t cols, std::vector<void *> *allocs)
{
	const uint64_t payload = rows * cols, scales = rows * (cols / LG_FP8_BLOCK);
	const float base = 1.0f / sqrtf((float)cols);
	std::vector<uint8_t> wh(payload);
	std::vector<float> sh(scales);
	const uint32_t seed = vf_seed++;
	for ( uint64_t i = 0u; i < payload; ++i )
		wh[i] = E4m3(Uniform(seed, (uint32_t)i));
	for ( uint64_t i = 0u; i < scales; ++i )
		sh[i] = base * (0.75f + 0.5f * ((float)(Mix(seed + (uint32_t)i) & 255u) / 255.0f));
	LingWeight w;
	w.weight = Device(wh.data(), wh.size());
	w.scale = (const float *)Device(sh.data(), sh.size() * sizeof(float));
	allocs->push_back((void *)w.weight);
	allocs->push_back((void *)w.scale);
	return w;
}

template<class G>
struct BenchRig
{
	using M = LingModel<G>;
	using S = LingShape<G>;
	using State = LingState<G>;
	static constexpr uint64_t kRankKv = (uint64_t)S::kAttnLayers * kBenchSlots * kBenchPages * (S::Kv::kPageBytes / G::kTpDegree);

	State *state;
	BenchBus bus;
	SparkStageRunnerServices services;
	cudaStream_t stream;
	uint32_t multiprocessors;
	uint8_t *kv;
	uint32_t *page_table, *positions, *context, *sequence_of_row, *row_begin, *row_indices, *recurrent;
	uint16_t *hidden, *input_save;
	std::vector<void *> allocs;
	uint32_t kda_layer_idx, attn_layer_idx;

	LingWeight Proj(uint64_t r, uint64_t c)
	{
		return G::kFp8Weights ? BenchFp8Weight(r, c, &allocs) : BenchBf16Weight(r, c, &allocs);
	}

	const uint16_t *Mat(uint64_t r, uint64_t c)
	{
		const uint16_t *p = Matrix(r, c);
		allocs.push_back((void *)p);
		return p;
	}

	const float *Nrm(uint64_t c)
	{
		const float *p = Norm(c);
		allocs.push_back((void *)p);
		return p;
	}

	template<class T>
	const T *Ten(uint64_t n, float sc, float off)
	{
		const T *p = Tensor<T>(n, sc, off);
		allocs.push_back((void *)p);
		return p;
	}

	int Open(void)
	{
		std::vector<uint32_t> table(kBenchSlots * kBenchPages);
		state = 0;
		stream = 0;
		kv = 0;
		page_table = positions = context = sequence_of_row = row_begin = row_indices = recurrent = 0;
		hidden = input_save = 0;
		bus.hidden = G::kHidden;
		bus.tp_degree = G::kTpDegree;
		memset(&services, 0, sizeof(services));
		services.context = &bus;
		services.round = BenchMockRound;
		state = (State *)calloc(1u, sizeof(State));
		state->services = &services;
		state->tp_degree = G::kTpDegree;
		state->tp_rank = 0u;
		state->max_rows = kBenchMaxRows;
		state->slots = kBenchSlots;
		state->ffn_rows = 256u;
		state->expert_first = 0u;
		M::Shape(state);
		state->arena_bytes = M::Plan(state, 0, 0u);
		if ( cudaDeviceGetAttribute((int *)&multiprocessors, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess ||
			cudaStreamCreate(&stream) != cudaSuccess ||
			cudaMalloc((void **)&state->arena, state->arena_bytes) != cudaSuccess ||
			cudaMemset(state->arena, 0, state->arena_bytes) != cudaSuccess ||
			cudaMalloc((void **)&kv, kRankKv) != cudaSuccess ||
			cudaMemset(kv, 0, kRankKv) != cudaSuccess ||
			cudaMalloc((void **)&page_table, table.size() * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&positions, kBenchMaxRows * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&sequence_of_row, kBenchMaxRows * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&row_indices, kBenchMaxRows * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&row_begin, (kBenchSlots + 1u) * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&recurrent, kBenchSlots * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&context, kBenchSlots * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&hidden, (uint64_t)kBenchMaxRows * G::kHidden * sizeof(uint16_t)) != cudaSuccess ||
			cudaMalloc((void **)&input_save, (uint64_t)kBenchMaxRows * G::kHidden * sizeof(uint16_t)) != cudaSuccess )
			return 0;
		M::Plan(state, (uint8_t *)state->arena, 0u);
		M::Arrange(state);
		for ( uint32_t i = 0u; i < table.size(); ++i )
			table[i] = i;
		if ( cudaMemcpy(page_table, table.data(), table.size() * sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess )
			return 0;
		attn_layer_idx = G::kPhase;
		kda_layer_idx = G::kPhase == 0u ? 1u : 0u;
		const float *in_norm = Nrm(G::kHidden);
		const float *post_norm = Nrm(G::kHidden);
		const uint16_t *router = Mat(G::kExperts, G::kHidden);
		const float *ebias = Ten<float>(G::kExperts, 0.05f, 0.0f);
		const LingWeight w1 = Proj((uint64_t)G::kLocalExperts * G::kMoeInter, G::kHidden);
		const LingWeight w3 = Proj((uint64_t)G::kLocalExperts * G::kMoeInter, G::kHidden);
		const LingWeight w2 = Proj((uint64_t)G::kLocalExperts * G::kHidden, G::kMoeInter);
		const uint16_t *sh_gu = Mat(2u * G::kSharedRows, G::kHidden);
		const uint16_t *sh_dn = Mat(G::kHidden, G::kSharedRows);
		LingLayer *kda_l = &state->layers[kda_layer_idx];
		kda_l->input_norm = in_norm;
		kda_l->post_norm = post_norm;
		kda_l->kda_qkv = Proj(S::kKdaChannels, G::kHidden);
		if constexpr ( G::kKdaRank != 0u )
		{
			kda_l->kda_f_a = BenchBf16Weight(G::kKdaRank, G::kHidden, &allocs);
			kda_l->kda_g_a = BenchBf16Weight(G::kKdaRank, G::kHidden, &allocs);
		}
		kda_l->kda_f = Proj(S::kKdaWidth, G::kKdaRank != 0u ? G::kKdaRank : G::kHidden);
		kda_l->kda_g = Proj(S::kKdaWidth, G::kKdaRank != 0u ? G::kKdaRank : G::kHidden);
		kda_l->kda_beta = Mat(S::kKdaRankHeads, G::kHidden);
		kda_l->kda_conv = Ten<uint16_t>((uint64_t)S::kKdaChannels * G::kConv, 0.5f, 0.0f);
		kda_l->kda_a_log = Ten<float>(S::kKdaRankHeads, 0.5f, 0.0f);
		kda_l->kda_dt_bias = Ten<float>(S::kKdaWidth, 0.5f, 0.0f);
		kda_l->kda_norm = Ten<uint16_t>(G::kKdaDim, 0.2f, 1.0f);
		kda_l->kda_out = Proj(G::kHidden, S::kKdaWidth);
		kda_l->router = router;
		kda_l->expert_bias = ebias;
		kda_l->moe_w1 = w1;
		kda_l->moe_w3 = w3;
		kda_l->moe_w2 = w2;
		kda_l->shared_gate_up = sh_gu;
		kda_l->shared_down = sh_dn;
		LingLayer *attn_l = &state->layers[attn_layer_idx];
		attn_l->input_norm = in_norm;
		attn_l->post_norm = post_norm;
		if constexpr ( S::kGqa )
		{
			attn_l->gqa_qkv = Mat(S::kGqaRankFused, G::kHidden);
			attn_l->gqa_gate = Mat(S::kGqaRankWidth, G::kHidden);
			attn_l->gqa_out = Mat(G::kHidden, S::kGqaRankWidth);
		}
		else
		{
			attn_l->mla_q = Proj((uint64_t)S::kMlaRankHeads * S::kQkDim, G::kHidden);
			attn_l->mla_kv_a = Proj(S::kSlot, G::kHidden);
			attn_l->mla_kv_norm = Nrm(G::kLatent);
			attn_l->mla_uk = Ten<uint16_t>((uint64_t)S::kMlaRankHeads * G::kLatent * G::kNope, 1.7f / sqrtf((float)G::kNope), 0.0f);
			attn_l->mla_uv = Ten<uint16_t>((uint64_t)S::kMlaRankHeads * G::kValueDim * G::kLatent, 1.7f / sqrtf((float)G::kLatent), 0.0f);
			attn_l->mla_gate = Mat(S::kMlaRankHeads, G::kHidden);
			attn_l->mla_out = Proj(G::kHidden, (uint64_t)S::kMlaRankHeads * G::kValueDim);
		}
		attn_l->router = router;
		attn_l->expert_bias = ebias;
		attn_l->moe_w1 = w1;
		attn_l->moe_w3 = w3;
		attn_l->moe_w2 = w2;
		attn_l->shared_gate_up = sh_gu;
		attn_l->shared_down = sh_dn;
		SparkStageRunnerKv attach;
		memset(&attach, 0, sizeof(attach));
		attach.pool = kv;
		attach.layer_stride_bytes = (uint64_t)kBenchSlots * kBenchPages * (S::Kv::kPageBytes / G::kTpDegree);
		attach.layer_page_bytes = S::Kv::kPageBytes;
		attach.layer_count = S::kAttnLayers;
		attach.page_table = page_table;
		attach.page_table_stride = kBenchPages;
		attach.pool_page_count = kBenchSlots * kBenchPages;
		attach.sequence_count = kBenchSlots;
		attach.context_shard.degree = G::kTpDegree;
		attach.context_shard.rank = 0u;
		attach.context_shard.grain = 1u;
		if ( M::AttachKv(state, &attach) != SPARK_STATUS_OK )
			return 0;
		std::vector<uint16_t> host_in((uint64_t)kBenchMaxRows * G::kHidden);
		for ( uint32_t r = 0u; r < kBenchMaxRows; ++r )
			for ( uint32_t c = 0u; c < G::kHidden; ++c )
				host_in[(uint64_t)r * G::kHidden + c] = Bf16(Uniform(7919u + r * 31u, c));
		if ( cudaMemcpy(input_save, host_in.data(), host_in.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess )
			return 0;
		std::vector<uint16_t> kv_init(kRankKv / sizeof(uint16_t));
		for ( uint64_t i = 0u; i < kv_init.size(); ++i )
			kv_init[i] = Bf16(0.15f * Uniform(12345u, (uint32_t)i));
		if ( cudaMemcpy(kv, kv_init.data(), kRankKv, cudaMemcpyHostToDevice) != cudaSuccess )
			return 0;
		return cudaDeviceSynchronize() == cudaSuccess;
	}

	void BuildStep(SparkStageRunnerStep *step, uint32_t rows, uint32_t ctx_pos, uint32_t commit, uint32_t last_rows_only, uint32_t gather_ctx)
	{
		std::vector<uint32_t> pos(rows), seq(rows, 0u), idx(rows);
		uint32_t rbeg[3] = { 0u, rows, rows };
		uint32_t rec[2] = { 0u, 1u };
		uint32_t ctx[2] = { ctx_pos + rows, 64u };
		if ( last_rows_only == 0u && rows > 1u && commit != 0u )
		{
			pos[0] = ctx_pos;
			seq[0] = 0u;
			idx[0] = 0u;
			for ( uint32_t r = 1u; r < rows; ++r )
			{
				pos[r] = r - 1u;
				seq[r] = 1u;
				idx[r] = r;
			}
			rbeg[0] = 0u;
			rbeg[1] = 1u;
			rbeg[2] = rows;
			ctx[0] = ctx_pos + 1u;
			ctx[1] = rows + 16u;
		}
		else
		{
			for ( uint32_t r = 0u; r < rows; ++r )
			{
				pos[r] = ctx_pos + r;
				seq[r] = 0u;
				idx[r] = r;
			}
		}
		cudaMemcpy(positions, pos.data(), rows * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(sequence_of_row, seq.data(), rows * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(row_indices, idx.data(), rows * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(row_begin, rbeg, sizeof(rbeg), cudaMemcpyHostToDevice);
		cudaMemcpy(recurrent, rec, sizeof(rec), cudaMemcpyHostToDevice);
		cudaMemcpy(context, ctx, sizeof(ctx), cudaMemcpyHostToDevice);
		memset(step, 0, sizeof(*step));
		step->hidden_bf16 = hidden;
		step->positions = positions;
		step->context_length = context;
		step->sequence_of_row = sequence_of_row;
		step->sequence_row_begin = row_begin;
		step->sequence_row_indices = row_indices;
		step->recurrent_index = recurrent;
		step->rows = rows;
		step->sequences = (last_rows_only == 0u && rows > 1u && commit != 0u) ? 2u : 1u;
		step->commit = commit;
		step->last_rows_only = last_rows_only;
		step->gather_context = gather_ctx;
		step->context = kBenchPages * G::kPageSlots;
		step->multiprocessors = multiprocessors;
		step->route_fault = VfRouteFault();
	}

	SparkStatus RunSubBlock(const SparkStageRunnerStep *step, uint32_t kind)
	{
		typename M::Frame frame = { step, stream };
		if ( kind == 0u )
		{
			const LingLayer &l = state->layers[kda_layer_idx];
			SparkStatus st = M::RmsNorm(state, frame, l.input_norm);
			if ( st == SPARK_STATUS_OK )
				st = M::Kda(state, frame, l);
			if ( st == SPARK_STATUS_OK )
				st = M::AddRows(state, frame);
			return st;
		}
		if ( kind == 1u )
		{
			const LingLayer &l = state->layers[attn_layer_idx];
			SparkStatus st = M::RmsNorm(state, frame, l.input_norm);
			if ( st == SPARK_STATUS_OK )
				st = M::Attention(state, frame, l);
			if ( st == SPARK_STATUS_OK )
				st = M::AddRows(state, frame);
			return st;
		}
		const LingLayer &l = state->layers[kda_layer_idx];
		SparkStatus st = M::RmsNorm(state, frame, l.post_norm);
		if ( st == SPARK_STATUS_OK )
			st = M::Experts(state, frame, l, kda_layer_idx);
		if ( st == SPARK_STATUS_OK )
			st = M::AddRows(state, frame);
		return st;
	}

	void ResetRecurrent(void)
	{
		cudaMemset(state->kda_state, 0, (uint64_t)S::kKdaLayers * kBenchSlots * state->kda_state_bytes);
		cudaMemset(state->kda_window, 0, (uint64_t)S::kKdaLayers * kBenchSlots * state->kda_window_bytes);
	}

	void Check5cInvariance(void)
	{
		static const uint32_t cases[][2] = { { 4u, 1u }, { 8u, 1u }, { 16u, 1u }, { 12u, 0u }, { 64u, 0u } };
		const char *sub_names[3] = { "KDA", S::kGqa ? "GQA" : "MLA", "MoE" };
		const uint32_t ctx_pos = 512u;
		for ( uint32_t kind = 0u; kind < 3u; ++kind )
		{
			SparkStageRunnerStep b1_step;
			std::vector<uint16_t> b1_out(G::kHidden);
			ResetRecurrent();
			cudaMemcpy(hidden, input_save, (uint64_t)G::kHidden * sizeof(uint16_t), cudaMemcpyDeviceToDevice);
			BuildStep(&b1_step, 1u, ctx_pos, 1u, 0u, 0u);
			const SparkStatus b1_st = RunSubBlock(&b1_step, kind);
			const cudaError_t b1_err = cudaStreamSynchronize(stream);
			Require(b1_st == SPARK_STATUS_OK && b1_err == cudaSuccess && cudaGetLastError() == cudaSuccess, G::Tag(), sub_names[kind], 1u, (uint32_t)b1_err, 0u);
			cudaMemcpy(b1_out.data(), hidden, (uint64_t)G::kHidden * sizeof(uint16_t), cudaMemcpyDeviceToHost);
			Require(Finite(b1_out) != 0u, G::Tag(), sub_names[kind], 1u, 999u, 0u);
			for ( uint32_t ci = 0u; ci < 5u; ++ci )
			{
				const uint32_t rows = cases[ci][0], commit = cases[ci][1];
				SparkStageRunnerStep step;
				std::vector<uint16_t> row0(G::kHidden);
				ResetRecurrent();
				cudaMemcpy(hidden, input_save, (uint64_t)rows * G::kHidden * sizeof(uint16_t), cudaMemcpyDeviceToDevice);
				BuildStep(&step, rows, ctx_pos, commit, 0u, 0u);
				const SparkStatus st = RunSubBlock(&step, kind);
				const cudaError_t cerr = cudaStreamSynchronize(stream);
				Require(st == SPARK_STATUS_OK && cerr == cudaSuccess && cudaGetLastError() == cudaSuccess, G::Tag(), sub_names[kind], rows, commit, (uint32_t)cerr);
				cudaMemcpy(row0.data(), hidden, (uint64_t)G::kHidden * sizeof(uint16_t), cudaMemcpyDeviceToHost);
				Require(memcmp(row0.data(), b1_out.data(), (uint64_t)G::kHidden * sizeof(uint16_t)) == 0,
					G::Tag(), sub_names[kind], rows, commit, FirstDifference(row0, b1_out, G::kHidden));
			}
		}
		printf("ok %s 0-ULP 5c batch invariance: B1 == B4 == B8 == B16 == ver12 == ver64 across KDA, %s, MoE\n",
			G::Tag(), S::kGqa ? "GQA" : "MLA");
	}

	float TimeSubBlock(const SparkStageRunnerStep *step, uint32_t kind)
	{
		cudaEvent_t t0, t1;
		cudaEventCreate(&t0);
		cudaEventCreate(&t1);
		SparkStatus st = SPARK_STATUS_OK;
		for ( uint32_t w = 0u; w < kBenchWarmup; ++w )
		{
			cudaMemcpyAsync(hidden, input_save, (uint64_t)step->rows * G::kHidden * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream);
			if ( st == SPARK_STATUS_OK )
				st = RunSubBlock(step, kind);
		}
		const cudaError_t werr = cudaStreamSynchronize(stream);
		Require(st == SPARK_STATUS_OK && werr == cudaSuccess && cudaGetLastError() == cudaSuccess, G::Tag(), "warmup ok", step->rows, kind, (uint32_t)werr);
		cudaEventRecord(t0, stream);
		for ( uint32_t it = 0u; it < kBenchIters; ++it )
			if ( st == SPARK_STATUS_OK )
				st = RunSubBlock(step, kind);
		cudaEventRecord(t1, stream);
		const cudaError_t rerr = cudaEventSynchronize(t1);
		Require(st == SPARK_STATUS_OK && rerr == cudaSuccess && cudaGetLastError() == cudaSuccess, G::Tag(), "bench ok", step->rows, kind, (uint32_t)rerr);
		float ms = 0.0f;
		cudaEventElapsedTime(&ms, t0, t1);
		cudaEventDestroy(t0);
		cudaEventDestroy(t1);
		return ms / (float)kBenchIters;
	}

	void BenchmarkAll(void)
	{
		struct Case { const char *name; uint32_t rows; uint32_t ctx_pos; uint32_t commit; uint32_t prefill; uint32_t gather_ctx; };
		static const Case cases[] =
		{
			{ "B1_short", 1u, 256u, 1u, 0u, 0u },
			{ "B1_long_4k", 1u, 4096u, 1u, 0u, 0u },
			{ "ver12", 12u, 512u, 0u, 0u, 0u },
			{ "B4", 4u, 512u, 1u, 0u, 0u },
			{ "B8", 8u, 512u, 1u, 0u, 0u },
			{ "B16", 16u, 512u, 1u, 0u, 0u },
			{ "B32", 32u, 512u, 1u, 0u, 0u },
			{ "B64", 64u, 512u, 1u, 0u, 0u },
			{ "warm_128_5718", 128u, 5718u, 1u, 1u, 5846u },
			{ "cold_512_0", 512u, 0u, 1u, 1u, 512u },
			{ "cold_512_4096", 512u, 4096u, 1u, 1u, 4608u }
		};
		printf("\n=== BENCH %s (TP=%u, hidden=%u, layers=%u [%u KDA + %u %s]) ===\n",
			G::Tag(), G::kTpDegree, G::kHidden, G::kFullLayers, G::kFullKdaLayers, G::kFullAttnLayers, S::kGqa ? "GQA" : "MLA");
		printf("%-16s %8s %8s %8s %10s %10s\n", "case", "KDA_ms", "Attn_ms", "MoE_ms", "Model_ms", "tok/s");
		for ( uint32_t ci = 0u; ci < sizeof(cases) / sizeof(cases[0]); ++ci )
		{
			const Case &c = cases[ci];
			SparkStageRunnerStep step;
			ResetRecurrent();
			BuildStep(&step, c.rows, c.ctx_pos, c.commit, c.prefill, c.gather_ctx);
			const float kda_ms = TimeSubBlock(&step, 0u);
			const float attn_ms = TimeSubBlock(&step, 1u);
			const float moe_ms = TimeSubBlock(&step, 2u);
			const float total_ms = (float)G::kFullKdaLayers * kda_ms + (float)G::kFullAttnLayers * attn_ms + (float)G::kFullLayers * moe_ms;
			const float tps = total_ms > 0.0f ? ((float)c.rows * 1000.0f / total_ms) : 0.0f;
			printf("%-16s %8.3f %8.3f %8.3f %10.2f %10.1f\n", c.name, kda_ms, attn_ms, moe_ms, total_ms, tps);
		}
	}

	void Close(void)
	{
		if ( state != 0 )
		{
			cudaFree(state->arena);
			free(state);
		}
		for ( void *p : allocs )
			cudaFree(p);
		cudaFree(kv);
		cudaFree(page_table);
		cudaFree(positions);
		cudaFree(sequence_of_row);
		cudaFree(row_indices);
		cudaFree(row_begin);
		cudaFree(recurrent);
		cudaFree(context);
		cudaFree(hidden);
		cudaFree(input_save);
		if ( stream != 0 )
			cudaStreamDestroy(stream);
		cudaDeviceSynchronize();
	}
};

template<class G>
static void RunGeometry(void)
{
	BenchRig<G> rig;
	if ( rig.Open() == 0 )
	{
		Require(0, G::Tag(), "bench rig opens", 0u, 0u, 0u);
		return;
	}
	rig.Check5cInvariance();
	rig.BenchmarkAll();
	rig.Close();
}

struct SolarOpen2ChunkExact1 : SolarOpen2Geometry<16u>
{
	static constexpr uint32_t kTpDegree = 1u;
	static constexpr uint32_t kHidden = 1536u;
	static constexpr uint32_t kLayers = 4u;
	static constexpr uint32_t kGqaHeads = 8u;
	static constexpr uint32_t kGqaKvHeads = 1u;
	static constexpr uint32_t kKdaHeads = 4u;
	static constexpr uint32_t kExperts = 16u;
	static constexpr uint32_t kLocalExperts = 16u;
	static constexpr uint32_t kMoeInter = 80u;
	static constexpr uint32_t kSharedRows = 128u;
	static constexpr uint32_t kFullLayers = 4u;
	static constexpr uint32_t kFullAttnLayers = 1u;
	static constexpr uint32_t kFullKdaLayers = 3u;
	static constexpr bool kFp8Weights = false;
	static const char *Tag(void) { return "solar-open-2-chunk512-tp1"; }
};

static float BenchRelL2(const std::vector<uint16_t> &a, const std::vector<uint16_t> &b, uint32_t row, uint32_t width)
{
	double diff = 0.0, norm = 0.0;
	for ( uint32_t c = 0u; c < width; ++c )
	{
		const uint32_t ua = (uint32_t)a[(uint64_t)row * width + c] << 16u;
		const uint32_t ub = (uint32_t)b[(uint64_t)row * width + c] << 16u;
		float fa = 0.0f, fb = 0.0f;
		memcpy(&fa, &ua, sizeof(float));
		memcpy(&fb, &ub, sizeof(float));
		const double d = (double)fa - (double)fb;
		diff += d * d;
		norm += (double)fb * (double)fb;
	}
	return norm > 0.0 ? (float)sqrt(diff / norm) : (float)sqrt(diff);
}

static void CheckGqaChunk512Exact(void)
{
	using G = SolarOpen2ChunkExact1;
	BenchRig<G> rig;
	if ( rig.Open() == 0 )
	{
		Require(0, G::Tag(), "chunk512 rig opens", 0u, 0u, 0u);
		return;
	}
	const uint64_t row_bytes = (uint64_t)G::kHidden * sizeof(uint16_t);
	std::vector<uint8_t> kv_384(rig.kRankKv), kv_512(rig.kRankKv);
	std::vector<uint16_t> out_a(256u * G::kHidden), out_b1(128u * G::kHidden), out_b2(128u * G::kHidden), out_ref(128u * G::kHidden);
	SparkStageRunnerStep step;
	cudaMemset(rig.kv, 0, rig.kRankKv);
	cudaMemcpy(rig.hidden, rig.input_save, 384u * row_bytes, cudaMemcpyDeviceToDevice);
	rig.BuildStep(&step, 384u, 0u, 1u, 1u, 384u);
	Require(rig.RunSubBlock(&step, 1u) == SPARK_STATUS_OK && cudaStreamSynchronize(rig.stream) == cudaSuccess, G::Tag(), "prefix 0..383", 0u, 0u, 0u);
	cudaMemcpy(kv_384.data(), rig.kv, rig.kRankKv, cudaMemcpyDeviceToHost);

	cudaMemcpy(rig.hidden, rig.input_save + (uint64_t)128u * G::kHidden, 256u * row_bytes, cudaMemcpyDeviceToDevice);
	rig.BuildStep(&step, 256u, 384u, 1u, 1u, 640u);
	Require(rig.RunSubBlock(&step, 1u) == SPARK_STATUS_OK && cudaStreamSynchronize(rig.stream) == cudaSuccess, G::Tag(), "chunk2 384..639", 0u, 0u, 0u);
	cudaMemcpy(out_a.data(), rig.hidden, 256u * row_bytes, cudaMemcpyDeviceToHost);

	cudaMemcpy(rig.kv, kv_384.data(), rig.kRankKv, cudaMemcpyHostToDevice);
	cudaMemcpy(rig.hidden, rig.input_save + (uint64_t)128u * G::kHidden, 128u * row_bytes, cudaMemcpyDeviceToDevice);
	rig.BuildStep(&step, 128u, 384u, 1u, 1u, 512u);
	Require(rig.RunSubBlock(&step, 1u) == SPARK_STATUS_OK && cudaStreamSynchronize(rig.stream) == cudaSuccess, G::Tag(), "chunk1 384..511", 0u, 0u, 0u);
	cudaMemcpy(out_b1.data(), rig.hidden, 128u * row_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(kv_512.data(), rig.kv, rig.kRankKv, cudaMemcpyDeviceToHost);
	Require(memcmp(out_b1.data(), out_a.data(), 128u * row_bytes) == 0, G::Tag(),
		"1-chunk vs 2-chunk 0-ULP identical for pos<512", FirstDifference(out_b1, out_a, G::kHidden), 0u, 0u);

	cudaMemcpy(rig.hidden, rig.input_save + (uint64_t)256u * G::kHidden, 128u * row_bytes, cudaMemcpyDeviceToDevice);
	rig.BuildStep(&step, 128u, 512u, 1u, 1u, 640u);
	Require(rig.RunSubBlock(&step, 1u) == SPARK_STATUS_OK && cudaStreamSynchronize(rig.stream) == cudaSuccess, G::Tag(), "warm 512..639", 0u, 0u, 0u);
	cudaMemcpy(out_b2.data(), rig.hidden, 128u * row_bytes, cudaMemcpyDeviceToHost);
	std::vector<uint16_t> out_a_tail(out_a.begin() + (ptrdiff_t)(128u * G::kHidden), out_a.end());
	Require(memcmp(out_b2.data(), out_a_tail.data(), 128u * row_bytes) == 0, G::Tag(),
		"warm 512..639 equals chunk2 tail 0-ULP", FirstDifference(out_b2, out_a_tail, G::kHidden), 0u, 0u);

	cudaMemcpy(rig.kv, kv_512.data(), rig.kRankKv, cudaMemcpyHostToDevice);
	cudaMemcpy(rig.hidden, rig.input_save + (uint64_t)256u * G::kHidden, 128u * row_bytes, cudaMemcpyDeviceToDevice);
	rig.BuildStep(&step, 128u, 512u, 1u, 1u, 0u);
	Require(rig.RunSubBlock(&step, 1u) == SPARK_STATUS_OK && cudaStreamSynchronize(rig.stream) == cudaSuccess, G::Tag(), "scatter ref 512..639", 0u, 0u, 0u);
	cudaMemcpy(out_ref.data(), rig.hidden, 128u * row_bytes, cudaMemcpyDeviceToHost);
	for ( uint32_t r = 0u; r < 128u; ++r )
	{
		const float d = BenchRelL2(out_b2, out_ref, r, G::kHidden);
		Require(d < 0.01f, G::Tag(), "2-chunk GQA matches scatter ref (row, rel x 1e6)", r, (uint32_t)(d * 1.0e6f), 0u);
	}
	printf("ok %s 0-ULP Gate GX across KV_CHUNK=512 boundary (1-chunk==2-chunk, warm==cold, relL2<0.01 vs scatter)\n", G::Tag());
	rig.Close();
}

int main(void)
{
	CheckGqaChunk512Exact();
	RunGeometry<SolarOpen2Bench16>();
	RunGeometry<Ling3Bench16>();
	RunGeometry<Ling32AirBench16>();
	if ( vf_failures != 0 )
	{
		printf("FAIL ling blackwell bench+exact: %d failures\n", vf_failures);
		return 1;
	}
	printf("\nPASS ling blackwell bench+exact: 0-ULP 5c verified and TP16 per-rank timings collected\n");
	return 0;
}
