#include "modules/gemma4_31b_resident_decode_stage/source/spark_gemma4_31b_stage_model.cu"
#include "modules/gemma4_26b_resident_decode_stage/source/spark_gemma4_26b_stage_model.cu"
#include "modules/muse_glimmer_tp16_resident_decode_stage/source/spark_muse_tp16_stage_model.cu"
#include "modules/laguna_tp16_resident_decode_stage/source/spark_laguna_tp16_stage_model.cu"
#include "modules/kexaone2_tp16_resident_decode_stage/source/spark_kexaone2_tp16_stage_model.cu"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "tests/cuda/verify_frames_common.cuh"

#define GB_MAX_ROWS 512u
#define GB_SLOTS 64u
#define GB_PAGES 16u
#define GB_GRAIN 256u
#define GB_PREFIX 64u

struct Bench31b : Gemma4_31bGeometry<16u>
{
	static constexpr uint32_t kLayers = 2u;
	static constexpr uint32_t kPeriod = 2u;
	static constexpr uint32_t kPhase = 1u;
	static constexpr uint32_t kBenchFfn = 1344u;
	static constexpr uint32_t kRealLayers = 60u;
	static const char *Tag(void) { return "gemma-4-31b"; }
};

struct Bench26b : Gemma4_26bGeometry<16u>
{
	static constexpr uint32_t kLayers = 2u;
	static constexpr uint32_t kPeriod = 2u;
	static constexpr uint32_t kPhase = 1u;
	static constexpr uint32_t kBenchFfn = 128u;
	static constexpr uint32_t kRealLayers = 30u;
	static const char *Tag(void) { return "gemma-4-26b-a4b"; }
};

struct BenchKexaone : Kexaone2Geometry<16u>
{
	static constexpr uint32_t kLayers = 2u;
	static constexpr uint32_t kPeriod = 2u;
	static constexpr uint32_t kPhase = 1u;
	static constexpr uint32_t kFirstMoeLayer = 0u;
	static constexpr uint32_t kBenchFfn = 128u;
	static constexpr uint32_t kRealLayers = 78u;
	static constexpr uint32_t LayerWindow(uint32_t index) { return index == 1u ? 0u : kWindow; }
	static constexpr float LayerSwigluLimit(uint32_t index) { return index == 1u ? 7.0f : INFINITY; }
	static const char *Tag(void) { return "k-exaone-2.0"; }
};

struct BenchLaguna : LagunaGeometry<16u>
{
	static constexpr uint32_t kLayers = 2u;
	static constexpr uint32_t kPeriod = 2u;
	static constexpr uint32_t kPhase = 1u;
	static constexpr uint32_t kFirstMoeLayer = 0u;
	static constexpr uint32_t kBenchFfn = 64u;
	static constexpr uint32_t kRealLayers = 48u;
	static const char *Tag(void) { return "laguna-s-2.1"; }
};

struct BenchMuse : MuseGlimmerGeometry<16u>
{
	static constexpr uint32_t kLayers = 2u;
	static constexpr uint32_t kPeriod = 2u;
	static constexpr uint32_t kPhase = 1u;
	static constexpr uint32_t kBenchFfn = 1248u;
	static constexpr uint32_t kRealLayers = 52u;
	static const char *Tag(void) { return "muse-glimmer-30b"; }
};

__global__ static void BenchFillBf16Kernel(uint16_t *dst, uint64_t count, uint32_t seed, float scale, float offset)
{
	const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t step = (uint64_t)gridDim.x * blockDim.x;
	for ( uint64_t i = idx; i < count; i += step )
	{
		const float v = offset + scale * Uniform(seed, (uint32_t)i);
		dst[i] = Bf16(v);
	}
}

__global__ static void BenchFillF32Kernel(float *dst, uint64_t count, uint32_t seed, float scale, float offset)
{
	const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t step = (uint64_t)gridDim.x * blockDim.x;
	for ( uint64_t i = idx; i < count; i += step )
		dst[i] = offset + scale * Uniform(seed, (uint32_t)i);
}

__global__ static void BenchReplicateKernel(const uint4 *src, uint4 *dst, uint64_t slice_vec, uint32_t degree)
{
	const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t step = (uint64_t)gridDim.x * blockDim.x;
	for ( uint64_t i = idx; i < slice_vec; i += step )
	{
		const uint4 v = src[i];
		for ( uint32_t r = 0u; r < degree; ++r )
			dst[r * slice_vec + i] = v;
	}
}

typedef struct BenchBus
{
	uint32_t hidden;
	uint32_t tp_degree;
} BenchBus;

static SparkStatus BenchRound(void *context, void *stream, uint32_t operation, uint32_t stage, uint32_t rows, uint32_t row_elements,
	const void *local, void *full)
{
	const BenchBus *bus = (const BenchBus *)context;
	if ( stage != SPARK_STAGE_RUNNER_ROUND_ALL )
		return SPARK_STATUS_UNSUPPORTED;
	if ( operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16 || local == full )
		return SPARK_STATUS_OK;
	const uint64_t slice_bytes = (uint64_t)rows * (row_elements != 0u ? row_elements : bus->hidden) * sizeof(uint16_t);
	if ( slice_bytes == 0u )
		return SPARK_STATUS_OK;
	if ( (slice_bytes & 15u) == 0u )
	{
		const uint64_t vecs = slice_bytes >> 4u;
		const uint32_t blocks = (uint32_t)((vecs + 255u) / 256u < 256u ? (vecs + 255u) / 256u : 256u);
		BenchReplicateKernel<<<blocks, 256u, 0, (cudaStream_t)stream>>>((const uint4 *)local, (uint4 *)full, vecs, bus->tp_degree);
		return cudaPeekAtLastError() == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
	}
	for ( uint32_t r = 0u; r < bus->tp_degree; ++r )
		if ( cudaMemcpyAsync((uint8_t *)full + r * slice_bytes, local, slice_bytes, cudaMemcpyDeviceToDevice, (cudaStream_t)stream) != cudaSuccess )
			return SPARK_STATUS_IO_ERROR;
	return SPARK_STATUS_OK;
}

template<class G>
struct BenchRig
{
	using M = GemmaModel<G>;
	using S = GemmaShape<G>;
	using State = GemmaState<G>;
	static constexpr uint64_t kSlidingRegion = (uint64_t)S::kSlidingLayers * GB_SLOTS * GB_PAGES * S::SlidingKvGeometry::kPageBytes;
	static constexpr uint64_t kFullRegion = (uint64_t)S::kFullLayers * GB_SLOTS * GB_PAGES * S::FullKvGeometry::kPageBytes;
	State *state;
	BenchBus bus;
	SparkStageRunnerServices services;
	cudaStream_t stream;
	uint32_t multiprocessors;
	uint32_t seed;
	uint8_t *kv;
	uint32_t *page_table, *positions, *context, *sequence_of_row;
	uint16_t *hidden;
	uint32_t host_context[GB_SLOTS];
	std::vector<void *> allocs;

	const uint16_t *GpuBf16(uint64_t count, float scale, float offset)
	{
		uint16_t *ptr = 0;
		if ( cudaMalloc(&ptr, count * sizeof(uint16_t)) != cudaSuccess )
		{
			printf("FAIL %s: cudaMalloc bf16 %llu bytes\n", G::Tag(), (unsigned long long)(count * sizeof(uint16_t)));
			exit(1);
		}
		allocs.push_back(ptr);
		const uint32_t blocks = (uint32_t)((count + 255u) / 256u < 512u ? (count + 255u) / 256u : 512u);
		BenchFillBf16Kernel<<<blocks, 256u, 0, stream>>>(ptr, count, seed++, scale, offset);
		return ptr;
	}

	const float *GpuF32(uint64_t count, float scale, float offset)
	{
		float *ptr = 0;
		if ( cudaMalloc(&ptr, count * sizeof(float)) != cudaSuccess )
		{
			printf("FAIL %s: cudaMalloc f32 %llu bytes\n", G::Tag(), (unsigned long long)(count * sizeof(float)));
			exit(1);
		}
		allocs.push_back(ptr);
		const uint32_t blocks = (uint32_t)((count + 255u) / 256u < 256u ? (count + 255u) / 256u : 256u);
		BenchFillF32Kernel<<<blocks, 256u, 0, stream>>>(ptr, count, seed++, scale, offset);
		return ptr;
	}

	const uint16_t *GpuMatrix(uint64_t rows, uint64_t cols)
	{
		return GpuBf16(rows * cols, 1.5f / sqrtf((float)cols), 0.0f);
	}

	void InitLayer(GemmaLayer *layer, uint32_t index)
	{
		const bool full = S::Full(index);
		const uint32_t fused = (full ? S::kFullFused : S::kSlidingFused) / G::kTpDegree;
		const uint32_t q = (full ? S::kFullQ : S::kSlidingQ) / G::kTpDegree;
		const uint32_t dim = full ? G::kFullDim : G::kSlidingDim;
		layer->qkv = GpuMatrix(fused, G::kHidden);
		layer->out = GpuMatrix(G::kHidden, q);
		layer->gate_up = GpuMatrix(2u * G::kBenchFfn, G::kHidden);
		layer->down = GpuMatrix(G::kHidden, G::kBenchFfn);
		layer->ffn_rows = G::kBenchFfn;
		layer->input_norm = GpuF32(G::kHidden, 0.2f, 1.0f);
		layer->pre_ff_norm = GpuF32(G::kHidden, 0.2f, 1.0f);
		if ( !G::kPreNorm )
		{
			layer->post_attn_norm = GpuF32(G::kHidden, 0.2f, 1.0f);
			layer->post_ff_norm = GpuF32(G::kHidden, 0.2f, 1.0f);
		}
		if ( G::kLayerScalar )
			layer->layer_scalar = GpuF32(1u, 0.1f, 0.8f);
		if ( G::kQkNormWeighted )
		{
			layer->q_norm = GpuF32(dim, 0.2f, 1.0f);
			layer->k_norm = GpuF32(dim, 0.2f, 1.0f);
		}
		if ( G::kOutputGate )
			layer->gate = GpuMatrix(q, G::kHidden);
		if ( G::kHeadGate )
			layer->gate = GpuMatrix(full ? S::kFullGateRows : S::kSlidingGateRows, G::kHidden);
		if ( G::kMoe )
		{
			layer->router_proj = GpuMatrix(G::kExperts, G::kHidden);
			layer->moe_w1 = GpuMatrix((uint64_t)G::kLocalExperts * G::kMoeInter, G::kHidden);
			layer->moe_w3 = GpuMatrix((uint64_t)G::kLocalExperts * G::kMoeInter, G::kHidden);
			layer->moe_w2 = GpuBf16((uint64_t)G::kLocalExperts * G::kHidden * G::kMoeInter, 1.5f / sqrtf((float)G::kMoeInter), 0.0f);
			if ( G::kSigmoidRouter )
				layer->expert_bias = GpuF32(G::kExperts, 0.05f, 0.0f);
			else
			{
				layer->router_scale = GpuF32(G::kHidden, 0.2f, 1.0f);
				layer->expert_scale = GpuF32(G::kExperts, 0.2f, 1.0f);
				layer->post_ff_norm_1 = GpuF32(G::kHidden, 0.2f, 1.0f);
				layer->pre_ff_norm_2 = GpuF32(G::kHidden, 0.2f, 1.0f);
				layer->post_ff_norm_2 = GpuF32(G::kHidden, 0.2f, 1.0f);
			}
		}
	}

	int Open(void)
	{
		std::vector<uint32_t> table(GB_SLOTS * GB_PAGES);
		SparkStageRunnerKv attach;
		state = 0;
		stream = 0;
		kv = 0;
		page_table = positions = context = sequence_of_row = 0;
		hidden = 0;
		seed = 101u;
		memset(host_context, 0, sizeof(host_context));
		bus.hidden = G::kHidden;
		bus.tp_degree = G::kTpDegree;
		memset(&services, 0, sizeof(services));
		services.context = &bus;
		services.round = BenchRound;
		state = (State *)calloc(1u, sizeof(State));
		state->services = &services;
		state->tp_degree = G::kTpDegree;
		state->tp_rank = 0u;
		state->max_rows = GB_MAX_ROWS;
		state->slots = GB_SLOTS;
		state->ffn_rows = G::kBenchFfn;
		if ( cudaStreamCreate(&stream) != cudaSuccess ||
			cudaDeviceGetAttribute((int *)&multiprocessors, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess ||
			M::Size(state) != SPARK_STATUS_OK )
			return 0;
		for ( uint32_t i = 0u; i < G::kLayers; ++i )
			InitLayer(&state->layers[i], i);
		if ( cudaStreamSynchronize(stream) != cudaSuccess ||
			M::Allocate(state) != SPARK_STATUS_OK ||
			cudaMalloc((void **)&kv, kSlidingRegion + kFullRegion) != cudaSuccess ||
			cudaMemset(kv, 0, kSlidingRegion + kFullRegion) != cudaSuccess ||
			cudaMalloc((void **)&page_table, table.size() * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&positions, GB_MAX_ROWS * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&sequence_of_row, GB_MAX_ROWS * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&context, GB_SLOTS * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&hidden, (uint64_t)GB_MAX_ROWS * G::kHidden * sizeof(uint16_t)) != cudaSuccess )
			return 0;
		for ( uint32_t i = 0u; i < table.size(); ++i )
			table[i] = i;
		if ( cudaMemcpy(page_table, table.data(), table.size() * sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess )
			return 0;
		memset(&attach, 0, sizeof(attach));
		attach.pool = kv;
		attach.layer_stride_bytes = (uint64_t)GB_SLOTS * GB_PAGES * S::SlidingKvGeometry::kPageBytes;
		attach.layer_page_bytes = S::SlidingKvGeometry::kPageBytes;
		attach.layer_count = S::kSlidingLayers;
		attach.page_table = page_table;
		attach.page_table_stride = GB_PAGES;
		attach.pool_page_count = GB_SLOTS * GB_PAGES;
		attach.sequence_count = GB_SLOTS;
		attach.context_shard.degree = G::kTpDegree;
		attach.context_shard.rank = 0u;
		attach.context_shard.grain = GB_GRAIN;
		attach.second_pool = kv + kSlidingRegion;
		attach.second_layer_stride_bytes = (uint64_t)GB_SLOTS * GB_PAGES * S::FullKvGeometry::kPageBytes;
		attach.second_layer_page_bytes = S::FullKvGeometry::kPageBytes;
		attach.second_layer_count = S::kFullLayers;
		return M::AttachKv(state, &attach) == SPARK_STATUS_OK;
	}

	void FillInput(const VfRow &row, uint16_t *dst) const
	{
		const uint32_t s = Mix(row.sequence * 7919u + row.variant * 104729u) ^ row.position;
		for ( uint32_t i = 0u; i < G::kHidden; ++i )
			dst[i] = Bf16(Uniform(s, i));
	}

	int Run(const VfRow *rows, uint32_t count, uint32_t commit, std::vector<uint16_t> *output, uint32_t last_rows_only = 0u, uint32_t gather_context = 0u)
	{
		std::vector<uint16_t> input((uint64_t)count * G::kHidden);
		std::vector<uint32_t> pos(count), seq(count);
		uint8_t seen[GB_SLOTS];
		memset(seen, 0, sizeof(seen));
		uint32_t sequences = 0u;
		SparkStageRunnerStep step;
		for ( uint32_t r = 0u; r < count; ++r )
		{
			FillInput(rows[r], input.data() + (uint64_t)r * G::kHidden);
			pos[r] = rows[r].position;
			seq[r] = rows[r].sequence;
			if ( seen[rows[r].sequence] == 0u )
				sequences++;
			seen[rows[r].sequence] = 1u;
			host_context[rows[r].sequence] = rows[r].position + 1u;
		}
		memset(&step, 0, sizeof(step));
		step.hidden_bf16 = hidden;
		step.positions = positions;
		step.context_length = context;
		step.sequence_of_row = sequence_of_row;
		step.rows = count;
		step.sequences = sequences;
		step.commit = commit;
		step.last_rows_only = last_rows_only;
		step.gather_context = gather_context;
		step.host_positions = pos.data();
		step.host_sequence_of_row = seq.data();
		step.context = GB_PAGES * G::kPageSlots;
		step.multiprocessors = multiprocessors;
		step.route_fault = VfRouteFault();
		if ( cudaMemcpyAsync(hidden, input.data(), input.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			cudaMemcpyAsync(positions, pos.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			cudaMemcpyAsync(sequence_of_row, seq.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			cudaMemcpyAsync(context, host_context, sizeof(host_context), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			M::Step(state, &step, stream) != SPARK_STATUS_OK || cudaStreamSynchronize(stream) != cudaSuccess )
			return 0;
		if ( output != 0 )
		{
			const uint64_t off = output->size();
			output->resize(off + input.size());
			if ( cudaMemcpy(output->data() + off, hidden, input.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess )
				return 0;
		}
		return 1;
	}

	double TimeStep(const VfRow *rows, uint32_t count, uint32_t commit, uint32_t last_rows_only, uint32_t gather_context,
		uint32_t warmup, uint32_t iters)
	{
		std::vector<uint16_t> input((uint64_t)count * G::kHidden);
		std::vector<uint32_t> pos(count), seq(count);
		uint8_t seen[GB_SLOTS];
		memset(seen, 0, sizeof(seen));
		uint32_t sequences = 0u;
		for ( uint32_t r = 0u; r < count; ++r )
		{
			FillInput(rows[r], input.data() + (uint64_t)r * G::kHidden);
			pos[r] = rows[r].position;
			seq[r] = rows[r].sequence;
			if ( seen[rows[r].sequence] == 0u )
				sequences++;
			seen[rows[r].sequence] = 1u;
			host_context[rows[r].sequence] = rows[r].position + 1u;
		}
		uint16_t *saved_in = 0;
		if ( cudaMalloc(&saved_in, input.size() * sizeof(uint16_t)) != cudaSuccess ||
			cudaMemcpy(saved_in, input.data(), input.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
			cudaMemcpy(positions, pos.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess ||
			cudaMemcpy(sequence_of_row, seq.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess ||
			cudaMemcpy(context, host_context, sizeof(host_context), cudaMemcpyHostToDevice) != cudaSuccess )
		{
			cudaFree(saved_in);
			return -1.0;
		}
		SparkStageRunnerStep step;
		memset(&step, 0, sizeof(step));
		step.hidden_bf16 = hidden;
		step.positions = positions;
		step.context_length = context;
		step.sequence_of_row = sequence_of_row;
		step.rows = count;
		step.sequences = sequences;
		step.commit = commit;
		step.last_rows_only = last_rows_only;
		step.gather_context = gather_context;
		step.host_positions = pos.data();
		step.host_sequence_of_row = seq.data();
		step.context = GB_PAGES * G::kPageSlots;
		step.multiprocessors = multiprocessors;
		step.route_fault = VfRouteFault();
		for ( uint32_t w = 0u; w < warmup; ++w )
		{
			cudaMemcpyAsync(hidden, saved_in, input.size() * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream);
			if ( M::Step(state, &step, stream) != SPARK_STATUS_OK )
			{
				cudaFree(saved_in);
				return -1.0;
			}
		}
		cudaStreamSynchronize(stream);
		cudaEvent_t t0, t1;
		cudaEventCreate(&t0);
		cudaEventCreate(&t1);
		cudaEventRecord(t0, stream);
		for ( uint32_t it = 0u; it < iters; ++it )
		{
			cudaMemcpyAsync(hidden, saved_in, input.size() * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream);
			if ( M::Step(state, &step, stream) != SPARK_STATUS_OK )
			{
				cudaEventDestroy(t0);
				cudaEventDestroy(t1);
				cudaFree(saved_in);
				return -1.0;
			}
		}
		cudaEventRecord(t1, stream);
		cudaEventSynchronize(t1);
		float ms = 0.0f;
		cudaEventElapsedTime(&ms, t0, t1);
		cudaEventDestroy(t0);
		cudaEventDestroy(t1);
		cudaFree(saved_in);
		return (double)ms * 1000.0 / (double)iters;
	}

	std::vector<uint8_t> SaveKv(void) const
	{
		std::vector<uint8_t> copy(kSlidingRegion + kFullRegion);
		if ( cudaMemcpy(copy.data(), kv, copy.size(), cudaMemcpyDeviceToHost) != cudaSuccess )
			copy.clear();
		return copy;
	}

	int LoadKv(const std::vector<uint8_t> &copy)
	{
		return copy.size() == kSlidingRegion + kFullRegion && cudaMemcpy(kv, copy.data(), copy.size(), cudaMemcpyHostToDevice) == cudaSuccess;
	}

	uint32_t KvClean(void) const
	{
		LmKvAccessError err, zero;
		memset(&zero, 0, sizeof(zero));
		return cudaMemcpy(&err, state->kv_error, sizeof(err), cudaMemcpyDeviceToHost) == cudaSuccess && memcmp(&err, &zero, sizeof(err)) == 0 ? 1u : 0u;
	}

	void Close(void)
	{
		if ( state != 0 )
			M::Close(state);
		for ( void *p : allocs )
			cudaFree(p);
		allocs.clear();
		cudaFree(kv);
		cudaFree(page_table);
		cudaFree(positions);
		cudaFree(sequence_of_row);
		cudaFree(context);
		cudaFree(hidden);
		if ( stream != 0 )
			cudaStreamDestroy(stream);
	}
};

template<class G>
static void VerifyModel(BenchRig<G> &rig)
{
	const char *model = G::Tag();
	std::vector<VfRow> prefix_rows(GB_PREFIX);
	for ( uint32_t p = 0u; p < GB_PREFIX; ++p )
		prefix_rows[p] = { 0u, p, 0u };
	Require(cudaMemset(rig.kv, 0, rig.kSlidingRegion + rig.kFullRegion) == cudaSuccess, model, "KV reset", 0u, 0u, 0u);
	memset(rig.host_context, 0, sizeof(rig.host_context));
	Require(rig.Run(prefix_rows.data(), GB_PREFIX, 1u, 0, 1u, GB_PREFIX) != 0, model, "prefix prefill", GB_PREFIX, 0u, 0u);
	const std::vector<uint8_t> cached = rig.SaveKv();

	static const uint32_t verify_frames[] = { 6u, 8u, 12u, 16u, 32u, 64u };
	for ( uint32_t f : verify_frames )
	{
		std::vector<uint16_t> seq_out, ver_out;
		std::vector<VfRow> vrows(f);
		rig.LoadKv(cached);
		memset(rig.host_context, 0, sizeof(rig.host_context));
		rig.host_context[0] = GB_PREFIX;
		for ( uint32_t r = 0u; r < f; ++r )
		{
			vrows[r] = { 0u, GB_PREFIX + r, 0u };
			Require(rig.Run(&vrows[r], 1u, 1u, &seq_out) != 0, model, "1-row decode", f, r, 0u);
		}
		rig.LoadKv(cached);
		memset(rig.host_context, 0, sizeof(rig.host_context));
		rig.host_context[0] = GB_PREFIX;
		Require(rig.Run(vrows.data(), f, 0u, &ver_out) != 0, model, "verify frame", f, 0u, 0u);
		Require(Finite(ver_out) != 0u, model, "verify finite", f, 0u, 0u);
		Require(FirstDifference(ver_out, seq_out, G::kHidden) == UINT32_MAX, model,
			"0-ULP 5c verify==B1 (frame, first diff row)", f, FirstDifference(ver_out, seq_out, G::kHidden), 0u);
	}

	static const uint32_t batch_sizes[] = { 4u, 8u };
	for ( uint32_t bs : batch_sizes )
	{
		std::vector<uint16_t> b1_out, bs_out;
		std::vector<VfRow> brows(bs);
		VfRow lane0 = { 0u, GB_PREFIX, 5u };
		rig.LoadKv(cached);
		memset(rig.host_context, 0, sizeof(rig.host_context));
		rig.host_context[0] = GB_PREFIX;
		Require(rig.Run(&lane0, 1u, 1u, &b1_out) != 0, model, "B1 reference decode", bs, 0u, 0u);
		rig.LoadKv(cached);
		memset(rig.host_context, 0, sizeof(rig.host_context));
		rig.host_context[0] = GB_PREFIX;
		brows[0] = lane0;
		for ( uint32_t r = 1u; r < bs; ++r )
			brows[r] = { r, 0u, 20u + r };
		Require(rig.Run(brows.data(), bs, 1u, &bs_out) != 0, model, "batched decode", bs, 0u, 0u);
		Require(memcmp(b1_out.data(), bs_out.data(), G::kHidden * sizeof(uint16_t)) == 0, model,
			"0-ULP 5c batch lane0 == B1", bs, 0u, 0u);
	}

	static const uint32_t deltas[] = { 1u, 8u, 12u, 16u, 64u, 128u };
	std::vector<VfRow> all_pf(GB_PREFIX + 128u);
	for ( uint32_t p = 0u; p < GB_PREFIX + 128u; ++p )
		all_pf[p] = { 0u, p, 0u };
	for ( uint32_t d : deltas )
	{
		const uint32_t total = GB_PREFIX + d;
		std::vector<uint16_t> cold_out, warm_out;
		Require(cudaMemset(rig.kv, 0, rig.kSlidingRegion + rig.kFullRegion) == cudaSuccess, model, "cold KV reset", d, 0u, 0u);
		memset(rig.host_context, 0, sizeof(rig.host_context));
		Require(rig.Run(all_pf.data(), total, 1u, &cold_out, 1u, total) != 0, model, "cold prefill", total, d, 0u);
		std::vector<uint16_t> cold_tail(cold_out.begin() + (ptrdiff_t)((uint64_t)GB_PREFIX * G::kHidden), cold_out.end());
		Require(rig.LoadKv(cached) != 0, model, "warm KV restore", d, 0u, 0u);
		memset(rig.host_context, 0, sizeof(rig.host_context));
		rig.host_context[0] = GB_PREFIX;
		Require(rig.Run(all_pf.data() + GB_PREFIX, d, 1u, &warm_out, 1u, total) != 0, model, "warm delta prefill", d, 0u, 0u);
		Require(FirstDifference(warm_out, cold_tail, G::kHidden) == UINT32_MAX, model,
			"0-ULP warm delta == cold prefill (delta, first diff row)", d, FirstDifference(warm_out, cold_tail, G::kHidden), 0u);
	}
	Require(rig.KvClean() != 0u, model, "no KV access errors", 0u, 0u, 0u);
	printf("ok %s [TP=16 real geometry]: 0-ULP 5c (B1==B4==B8==ver6/8/12/16/32/64) + warm==cold (d=1..128)\n", model);
}

template<class G>
static void BenchModel(BenchRig<G> &rig)
{
	const char *model = G::Tag();
	const double scale_full = (double)G::kRealLayers / (double)G::kLayers;
	std::vector<VfRow> seed_rows(128u);
	for ( uint32_t s = 0u; s < GB_SLOTS; ++s )
	{
		for ( uint32_t p = 0u; p < 128u; ++p )
			seed_rows[p] = { s, p, s };
		rig.Run(seed_rows.data(), 128u, 1u, 0, 1u, 128u);
	}
	printf("=== BENCH %s (TP=16 per-rank, 2L=1S+1F, proj %uL, SMs=%u) ===\n", model, G::kRealLayers, rig.multiprocessors);
	auto report = [&](const char *label, uint32_t rows, double us_2l)
	{
		const double us_per_layer = us_2l / (double)G::kLayers;
		const double full_ms = (us_2l * scale_full) / 1000.0;
		const double tok_s = (rows * 1000.0) / full_ms;
		printf("  %-12s rows=%3u  2L=%8.1f us  /L=%7.1f us  proj_%uL=%7.2f ms  proj_tok/s=%9.1f\n",
			label, rows, us_2l, us_per_layer, G::kRealLayers, full_ms, tok_s);
	};

	VfRow b1 = { 0u, 128u, 1u };
	report("B1", 1u, rig.TimeStep(&b1, 1u, 1u, 0u, 0u, 5u, 20u));

	static const uint32_t vlist[] = { 6u, 8u, 12u, 16u };
	for ( uint32_t v : vlist )
	{
		std::vector<VfRow> vr(v);
		for ( uint32_t r = 0u; r < v; ++r )
			vr[r] = { 0u, 128u + r, r };
		char name[32];
		snprintf(name, sizeof(name), "ver%u", v);
		report(name, v, rig.TimeStep(vr.data(), v, 0u, 0u, 0u, 4u, 12u));
	}

	static const uint32_t blist[] = { 4u, 8u, 16u, 32u, 64u };
	for ( uint32_t b : blist )
	{
		std::vector<VfRow> br(b);
		for ( uint32_t r = 0u; r < b; ++r )
			br[r] = { r, 128u, r };
		char name[32];
		snprintf(name, sizeof(name), "B%u", b);
		report(name, b, rig.TimeStep(br.data(), b, 1u, 0u, 0u, 4u, 12u));
	}

	static const uint32_t plist[] = { 128u, 512u };
	for ( uint32_t pf : plist )
	{
		std::vector<VfRow> pr(pf);
		for ( uint32_t r = 0u; r < pf; ++r )
			pr[r] = { 0u, r, 0u };
		char name[32];
		snprintf(name, sizeof(name), "PF%u_cold", pf);
		report(name, pf, rig.TimeStep(pr.data(), pf, 1u, 1u, pf, 3u, 8u));
	}

	{
		std::vector<VfRow> wr(128u);
		for ( uint32_t r = 0u; r < 128u; ++r )
			wr[r] = { 0u, 128u + r, 0u };
		report("PF128_warm", 128u, rig.TimeStep(wr.data(), 128u, 1u, 1u, 256u, 3u, 8u));
	}
}

template<class G>
static void RunOne(bool do_verify, bool do_bench, const char *filter)
{
	if ( filter != 0 && strcmp(filter, G::Tag()) != 0 )
		return;
	BenchRig<G> rig;
	if ( rig.Open() == 0 )
	{
		Require(0, G::Tag(), "BenchRig opens", 0u, 0u, 0u);
		rig.Close();
		return;
	}
	if ( do_verify )
		VerifyModel(rig);
	if ( do_bench )
		BenchModel(rig);
	rig.Close();
}

int main(int argc, char **argv)
{
	bool do_verify = true, do_bench = true;
	const char *filter = 0;
	for ( int i = 1; i < argc; ++i )
	{
		if ( strcmp(argv[i], "--verify-only") == 0 )
			do_bench = false;
		else if ( strcmp(argv[i], "--bench-only") == 0 )
			do_verify = false;
		else if ( strcmp(argv[i], "--model") == 0 && i + 1 < argc )
			filter = argv[++i];
	}
	RunOne<Bench31b>(do_verify, do_bench, filter);
	RunOne<Bench26b>(do_verify, do_bench, filter);
	RunOne<BenchKexaone>(do_verify, do_bench, filter);
	RunOne<BenchLaguna>(do_verify, do_bench, filter);
	RunOne<BenchMuse>(do_verify, do_bench, filter);
	if ( vf_failures != 0 )
	{
		printf("FAIL gemma4 blackwell bench: %d failures\n", vf_failures);
		return 1;
	}
	printf("PASS gemma4 blackwell bench\n");
	return 0;
}
