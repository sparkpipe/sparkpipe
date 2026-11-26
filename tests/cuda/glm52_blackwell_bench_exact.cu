#define GLM53_FULL_CONTRACT_SHA256 "bench-exact"
#define GLM53_FULL_MODEL_REVISION "bench-exact"
#define GLM53_FULL_LINEAR_CODEC SPARK_WEIGHT_CODEC_FP8_E4M3
#define GLM53_FULL_EXPERT_CODEC SPARK_WEIGHT_CODEC_FP8_E4M3
#define AXK2_CONTRACT_SHA256 "bench-exact"
#define AXK2_MODEL_REVISION "bench-exact"
#include "modules/glm53_full_tp16_resident_decode_stage/source/spark_glm53_full_tp16_stage_model.cu"
#include "modules/axk2_tp16_resident_decode_stage/source/spark_axk2_tp16_stage_model.cu"
#include "sparkpipe/spark_model_serving_adapter.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "tests/cuda/verify_frames_common.cuh"

static constexpr uint32_t kBenchMaxRows = 2048u;
static constexpr uint32_t kBenchSlots = 64u;
static constexpr uint32_t kBenchPages = 128u;
static constexpr uint32_t kBenchWarmup = 6u;
static constexpr uint32_t kBenchIters = 24u;

struct Glm53Bench16Fp8 : Glm52Dimensions<SPARK_WEIGHT_CODEC_FP8_E4M3, SPARK_WEIGHT_CODEC_FP8_E4M3>
{
	static constexpr uint32_t kTpDegree = 16u;
	static constexpr uint32_t kLayers = 4u;
	static constexpr uint32_t kFirstRouted = 1u;
	static constexpr uint32_t kIndexLayers = 2u;
	static constexpr uint32_t kFullLayers = 78u;
	static constexpr uint32_t kFullDenseLayers = 3u;
	static constexpr uint32_t kFullRoutedLayers = 75u;
	static constexpr uint32_t kFullIndexLayers = 21u;
	static uint32_t FullIndexer(uint32_t layer) { return (layer == 0u || layer == 2u) ? 1u : 0u; }
	static const char *Tag(void) { return "glm-5.3-fp8-tp16"; }
	static const char *ReportTag(void) { return "GLM53-FP8-BENCH"; }
	static const char *WeightdModel(void) { return "glm52_stage"; }
	static const char *Revision(void) { return "bench-exact"; }
	static const char *Contract(void) { return "bench-exact"; }
};

struct Glm53Bench16Bf16 : Glm52Dimensions<SPARK_WEIGHT_CODEC_BF16, SPARK_WEIGHT_CODEC_BF16>
{
	static constexpr uint32_t kTpDegree = 16u;
	static constexpr uint32_t kLayers = 4u;
	static constexpr uint32_t kFirstRouted = 1u;
	static constexpr uint32_t kIndexLayers = 2u;
	static constexpr uint32_t kFullLayers = 78u;
	static constexpr uint32_t kFullDenseLayers = 3u;
	static constexpr uint32_t kFullRoutedLayers = 75u;
	static constexpr uint32_t kFullIndexLayers = 21u;
	static uint32_t FullIndexer(uint32_t layer) { return (layer == 0u || layer == 2u) ? 1u : 0u; }
	static const char *Tag(void) { return "glm-5.3-bf16-tp16"; }
	static const char *ReportTag(void) { return "GLM53-BF16-BENCH"; }
	static const char *WeightdModel(void) { return "glm52_stage"; }
	static const char *Revision(void) { return "bench-exact"; }
	static const char *Contract(void) { return "bench-exact"; }
};

struct Axk2Bench16Fp8 : Axk2Geometry
{
	static constexpr uint32_t kLayers = 4u;
	static constexpr uint32_t kFirstRouted = 1u;
	static constexpr uint32_t kIndexLayers = 4u;
	static constexpr uint32_t kFullLayers = 61u;
	static constexpr uint32_t kFullDenseLayers = 1u;
	static constexpr uint32_t kFullRoutedLayers = 60u;
	static constexpr uint32_t kFullIndexLayers = 61u;
	static uint32_t FullIndexer(uint32_t layer) { return layer < kLayers ? 1u : 0u; }
	static const char *Tag(void) { return "a.x-k2-fp8-tp16"; }
};

struct GlmBenchBus
{
	uint32_t hidden;
	uint32_t tp_degree;
};

static SparkStatus GlmBenchRound(void *context, void *stream_ptr, uint32_t op, uint32_t stage, uint32_t rows, uint32_t row_elements,
	const void *local, void *full)
{
	const GlmBenchBus *bus = (const GlmBenchBus *)context;
	cudaStream_t stream = (cudaStream_t)stream_ptr;
	const uint64_t block = (uint64_t)rows * (row_elements != 0u ? row_elements : bus->hidden) * sizeof(uint16_t);
	if ( stage != SPARK_STAGE_RUNNER_ROUND_ALL )
		return SPARK_STATUS_UNSUPPORTED;
	if ( op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16 || op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64 )
	{
		if ( local == full )
			return SPARK_STATUS_OK;
		return cudaMemcpyAsync(full, local, block, cudaMemcpyDeviceToDevice, stream) == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
	}
	if ( op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_REDUCE_SCATTER_SUM_BF16 )
	{
		return cudaMemcpyAsync(full, local, block / bus->tp_degree, cudaMemcpyDeviceToDevice, stream) == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
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

static const uint8_t *BenchFp8Block(uint64_t groups, uint64_t rows, uint64_t columns, uint64_t *payload_bytes, std::vector<void *> *allocs)
{
	const uint64_t blocks = (columns + 127u) / 128u, payload = groups * rows * columns;
	std::vector<uint8_t> host(payload + groups * rows * blocks * sizeof(float));
	std::vector<float> scales(groups * rows * blocks);
	const uint32_t seed = vf_seed++;
	for ( uint64_t i = 0u; i < payload; ++i )
		host[i] = E4m3(Uniform(seed, (uint32_t)i));
	const float base = 1.7f / sqrtf((float)columns);
	for ( uint64_t i = 0u; i < scales.size(); ++i )
		scales[i] = (0.75f + 0.25f * Uniform(seed + 1u, (uint32_t)i)) * base;
	memcpy(host.data() + payload, scales.data(), scales.size() * sizeof(float));
	*payload_bytes = payload;
	const uint8_t *ptr = Device(host.data(), host.size());
	allocs->push_back((void *)ptr);
	return ptr;
}

template<class G>
static GlmLinearWeight BenchLinear(uint64_t rows, uint64_t columns, std::vector<void *> *allocs)
{
	GlmLinearWeight w;
	if constexpr ( GlmLinearScaled<G> )
	{
		uint64_t payload = 0u;
		w.weight = BenchFp8Block(1u, rows, columns, &payload, allocs);
		w.scale = (const float *)(w.weight + payload);
	}
	else
	{
		w.weight = (const uint8_t *)Matrix(rows, columns);
		w.scale = 0;
		allocs->push_back((void *)w.weight);
	}
	return w;
}

template<class G>
static const uint8_t *BenchExpertBlock(uint64_t groups, uint64_t rows, uint64_t columns, uint64_t *scale_offset, std::vector<void *> *allocs)
{
	if constexpr ( GlmExpertScaled<G> )
		return BenchFp8Block(groups, rows, columns, scale_offset, allocs);
	*scale_offset = 0u;
	const uint8_t *ptr = (const uint8_t *)Matrix(groups * rows, columns);
	allocs->push_back((void *)ptr);
	return ptr;
}

template<class G>
struct GlmBenchRig
{
	using M = GlmModel<G>;
	using S = GlmShape<G>;
	using State = typename M::State;
	static constexpr uint64_t kPool = (uint64_t)kBenchSlots * kBenchPages;
	static constexpr uint64_t kLatentRegion = kPool * (GlmKv<S>::kPageBytes / G::kTpDegree);
	static constexpr uint64_t kIndexRegion = (uint64_t)G::kIndexLayers * kPool * (GlmIndexKv<S>::kPageBytes / G::kTpDegree);

	State *state;
	GlmBenchBus bus;
	SparkStageRunnerServices services;
	cudaStream_t stream;
	uint32_t multiprocessors;
	uint8_t *kv;
	uint32_t *page_table, *positions, *context, *sequence_of_row;
	uint16_t *hidden, *input_save;
	const uint8_t *experts_up, *experts_down;
	uint64_t experts_up_scale, experts_down_scale;
	std::vector<void *> allocs;
	std::vector<uint32_t> host_pos, host_seq;

	static SparkStatus Weights(void *ctx, uint32_t layer, const uint32_t *group_row_offset, uint32_t routed_rows, const void **w1, const void **w2)
	{
		const GlmBenchRig *rig = (const GlmBenchRig *)ctx;
		(void)layer;
		(void)group_row_offset;
		(void)routed_rows;
		*w1 = rig->experts_up;
		*w2 = rig->experts_down;
		return SPARK_STATUS_OK;
	}

	static SparkStatus Round(void *ctx, void *stream_ptr, uint32_t op, uint32_t stage, uint32_t rows, uint32_t row_elements,
		const void *local, void *full)
	{
		return GlmBenchRound(&((GlmBenchRig *)ctx)->bus, stream_ptr, op, stage, rows, row_elements, local, full);
	}

	template<class T>
	const T *Ten(uint64_t n, float sc, float off)
	{
		const T *p = Tensor<T>(n, sc, off);
		allocs.push_back((void *)p);
		return p;
	}

	const uint16_t *Mat(uint64_t r, uint64_t c)
	{
		const uint16_t *p = Matrix(r, c);
		allocs.push_back((void *)p);
		return p;
	}

	int Open(void)
	{
		std::vector<uint32_t> table(kBenchSlots * kBenchPages);
		SparkStageRunnerKv attach;
		state = 0; stream = 0; kv = 0; page_table = positions = context = sequence_of_row = 0; hidden = input_save = 0;
		experts_up = experts_down = 0; experts_up_scale = experts_down_scale = 0u;
		bus.hidden = G::kHidden;
		bus.tp_degree = G::kTpDegree;
		memset(&services, 0, sizeof(services));
		services.context = this;
		services.round = Round;
		services.expert_weights = Weights;
		state = (State *)calloc(1u, sizeof(State));
		state->services = &services;
		state->tp_degree = G::kTpDegree;
		state->tp_rank = 0u;
		state->heads = G::kHeads / G::kTpDegree;
		state->max_rows = kBenchMaxRows;
		state->scatter_rows = GLM_SCATTER_ROWS;
		state->slots = kBenchSlots;
		state->max_positions = kBenchPages * GLM_PAGE_SLOTS;
		state->shard.degree = G::kTpDegree;
		state->shard.rank = 0u;
		state->shard.grain = 1u;
		if ( M::Size(state) != SPARK_STATUS_OK )
			return 0;
		const uint32_t heads = state->heads;
		const uint32_t dense_inter = G::kDenseInter / G::kTpDegree;
		const uint32_t expert_inter = G::kExpertInter / G::kTpDegree;
		const uint16_t *norm_h = Ten<uint16_t>(G::kHidden, 0.2f, 1.0f);
		const GlmLinearWeight q_a = BenchLinear<G>(G::kQueryA, G::kHidden, &allocs);
		const uint16_t *q_a_norm = Ten<uint16_t>(G::kQueryA, 0.2f, 1.0f);
		const GlmLinearWeight q_b = BenchLinear<G>((uint64_t)heads * S::kQueryHead, S::kQueryInput, &allocs);
		const GlmLinearWeight kv_a = BenchLinear<G>(S::kSlot, G::kHidden, &allocs);
		const uint16_t *kv_a_norm = Ten<uint16_t>(G::kLatent, 0.2f, 1.0f);
		const uint16_t *kv_b_key = Ten<uint16_t>((uint64_t)heads * G::kLatent * G::kNope, 1.7f / sqrtf((float)G::kNope), 0.0f);
		const uint16_t *kv_b_value = Ten<uint16_t>((uint64_t)heads * G::kValue * G::kLatent, 1.7f / sqrtf((float)G::kLatent), 0.0f);
		const GlmLinearWeight o_proj = BenchLinear<G>(G::kHidden, (uint64_t)heads * G::kValue, &allocs);
		const GlmLinearWeight index_q = BenchLinear<G>(S::kIndexQuery, G::kQueryA, &allocs);
		const GlmLinearWeight index_k = BenchLinear<G>(G::kIndexDim, G::kHidden, &allocs);
		const uint16_t *index_head = Mat(G::kIndexHeads, G::kHidden);
		const uint16_t *index_nw = Ten<uint16_t>(G::kIndexDim, 0.2f, 1.0f);
		const uint16_t *index_nb = Ten<uint16_t>(G::kIndexDim, 0.1f, 0.0f);
		const GlmLinearWeight dense_gu = BenchLinear<G>(2u * dense_inter, G::kHidden, &allocs);
		const GlmLinearWeight dense_dn = BenchLinear<G>(G::kHidden, dense_inter, &allocs);
		const uint16_t *router = Mat(G::kExperts, G::kHidden);
		const float *router_bias = Ten<float>(G::kExperts, 0.05f, 0.0f);
		const GlmLinearWeight sh_gu = BenchLinear<G>(2u * expert_inter, G::kHidden, &allocs);
		const GlmLinearWeight sh_dn = BenchLinear<G>(G::kHidden, expert_inter, &allocs);
		experts_up = BenchExpertBlock<G>(G::kExperts, 2u * expert_inter, G::kHidden, &experts_up_scale, &allocs);
		experts_down = BenchExpertBlock<G>(G::kExperts, G::kHidden, expert_inter, &experts_down_scale, &allocs);
		const uint16_t *gd = G::kNormGateRank != 0u ? Mat(G::kNormGateRank, G::kHidden) : 0;
		const uint16_t *gu = G::kNormGateRank != 0u ? Ten<uint16_t>((uint64_t)G::kHidden * G::kNormGateRank, 0.3f, 0.0f) : 0;
		for ( uint32_t l = 0u; l < G::kLayers; ++l )
		{
			GlmLayer *layer = &state->layers[l];
			layer->attn_norm = norm_h;
			layer->q_a = q_a;
			layer->q_a_norm = q_a_norm;
			layer->q_b = q_b;
			layer->kv_a = kv_a;
			layer->kv_a_norm = kv_a_norm;
			layer->kv_b_key = kv_b_key;
			layer->kv_b_value = kv_b_value;
			layer->o_proj = o_proj;
			layer->post_norm = norm_h;
			if ( G::FullIndexer(l) != 0u )
			{
				layer->index_q = index_q;
				layer->index_k = index_k;
				layer->index_head = index_head;
				layer->index_norm_weight = index_nw;
				layer->index_norm_bias = index_nb;
			}
			if constexpr ( G::kNormGateRank != 0u )
			{
				layer->attn_gate_down = gd;
				layer->attn_gate_up = gu;
				layer->post_gate_down = gd;
				layer->post_gate_up = gu;
			}
			if ( l < G::kFirstRouted )
			{
				layer->dense_gate_up = dense_gu;
				layer->dense_down = dense_dn;
			}
			else
			{
				layer->router = router;
				layer->router_bias = router_bias;
				layer->shared_gate_up = sh_gu;
				layer->shared_down = sh_dn;
				layer->up_gate_scale = experts_up_scale;
				layer->down_scale = experts_down_scale;
				state->w1_offset[l] = 0u;
				state->w2_offset[l] = 0u;
			}
		}
		state->final_norm = norm_h;
		if ( M::Allocate(state) != SPARK_STATUS_OK || cudaStreamCreate(&stream) != cudaSuccess ||
			cudaDeviceGetAttribute((int *)&multiprocessors, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess ||
			cudaMalloc((void **)&kv, kLatentRegion + kIndexRegion) != cudaSuccess ||
			cudaMalloc((void **)&page_table, table.size() * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&positions, kBenchMaxRows * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&sequence_of_row, kBenchMaxRows * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&context, kBenchSlots * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&hidden, (uint64_t)kBenchMaxRows * G::kHidden * sizeof(uint16_t)) != cudaSuccess ||
			cudaMalloc((void **)&input_save, (uint64_t)kBenchMaxRows * G::kHidden * sizeof(uint16_t)) != cudaSuccess )
			return 0;
		for ( uint32_t i = 0u; i < table.size(); ++i )
			table[i] = i;
		std::vector<uint16_t> host_in((uint64_t)kBenchMaxRows * G::kHidden);
		for ( uint32_t r = 0u; r < kBenchMaxRows; ++r )
			for ( uint32_t c = 0u; c < G::kHidden; ++c )
				host_in[(uint64_t)r * G::kHidden + c] = Bf16(Uniform(7919u + r * 31u, c));
		std::vector<uint16_t> kv_init((kLatentRegion + kIndexRegion) / sizeof(uint16_t));
		for ( uint64_t i = 0u; i < kv_init.size(); ++i )
			kv_init[i] = Bf16(0.15f * Uniform(12345u, (uint32_t)i));
		if ( cudaMemcpy(page_table, table.data(), table.size() * sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess ||
			cudaMemcpy(input_save, host_in.data(), host_in.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
			cudaMemcpy(kv, kv_init.data(), kLatentRegion + kIndexRegion, cudaMemcpyHostToDevice) != cudaSuccess )
			return 0;
		memset(&attach, 0, sizeof(attach));
		attach.pool = kv;
		attach.layer_page_bytes = (uint64_t)GLM_PAGE_SLOTS * S::kSlotBytes;
		attach.layer_stride_bytes = attach.layer_page_bytes / G::kTpDegree;
		attach.layer_count = S::kLayers;
		attach.page_table = page_table;
		attach.page_table_stride = kBenchPages;
		attach.pool_page_count = (uint32_t)kPool;
		attach.sequence_count = kBenchSlots;
		attach.context_shard = state->shard;
		attach.second_pool = kv + kLatentRegion;
		attach.second_layer_stride_bytes = kPool * (GlmIndexKv<S>::kPageBytes / G::kTpDegree);
		attach.second_layer_page_bytes = GlmIndexKv<S>::kPageBytes;
		attach.second_layer_count = G::kIndexLayers;
		return M::AttachKv(state, &attach) == SPARK_STATUS_OK && cudaDeviceSynchronize() == cudaSuccess;
	}

	void BuildStep(SparkStageRunnerStep *step, uint32_t rows, uint32_t ctx_pos, uint32_t commit, uint32_t prefill)
	{
		host_pos.resize(rows);
		host_seq.resize(rows);
		std::vector<uint32_t> ctx(kBenchSlots, 64u);
		uint32_t seqs = 1u;
		if ( prefill != 0u )
		{
			for ( uint32_t r = 0u; r < rows; ++r )
			{
				host_pos[r] = ctx_pos + r;
				host_seq[r] = 0u;
			}
			ctx[0] = ctx_pos + rows;
		}
		else if ( commit == 0u )
		{
			for ( uint32_t r = 0u; r < rows; ++r )
			{
				host_pos[r] = ctx_pos + r;
				host_seq[r] = 0u;
			}
			ctx[0] = ctx_pos + rows;
		}
		else
		{
			seqs = rows;
			host_pos[0] = ctx_pos;
			host_seq[0] = 0u;
			ctx[0] = ctx_pos + 1u;
			for ( uint32_t r = 1u; r < rows; ++r )
			{
				host_pos[r] = 64u + r;
				host_seq[r] = r;
				ctx[r] = host_pos[r] + 1u;
			}
		}
		cudaMemcpy(positions, host_pos.data(), rows * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(sequence_of_row, host_seq.data(), rows * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(context, ctx.data(), ctx.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
		memset(step, 0, sizeof(*step));
		step->hidden_bf16 = hidden;
		step->positions = positions;
		step->context_length = context;
		step->sequence_of_row = sequence_of_row;
		step->rows = rows;
		step->sequences = seqs;
		step->commit = commit;
		step->last_rows_only = prefill;
		step->context = kBenchPages * GLM_PAGE_SLOTS;
		step->multiprocessors = multiprocessors;
		step->route_fault = VfRouteFault();
		step->host_positions = host_pos.data();
		step->host_sequence_of_row = host_seq.data();
		step->graph_key = M::GraphKey(state, step);
	}

	void Check5cInvariance(void)
	{
		static const uint32_t ctx_cases[] = { 256u, 4096u };
		static const uint32_t batch_cases[][2] = { { 4u, 1u }, { 8u, 1u }, { 12u, 0u }, { 64u, 0u } };
		for ( uint32_t ctx : ctx_cases )
		{
			SparkStageRunnerStep b1_step;
			std::vector<uint16_t> b1_out(G::kHidden);
			cudaMemcpy(hidden, input_save, (uint64_t)G::kHidden * sizeof(uint16_t), cudaMemcpyDeviceToDevice);
			BuildStep(&b1_step, 1u, ctx, 1u, 0u);
			const SparkStatus b1_st = M::Step(state, &b1_step, stream);
			const cudaError_t b1_err = cudaStreamSynchronize(stream);
			Require(b1_st == SPARK_STATUS_OK && b1_err == cudaSuccess && cudaGetLastError() == cudaSuccess, G::Tag(), "5c B1 step ok", ctx, (uint32_t)b1_err, 0u);
			cudaMemcpy(b1_out.data(), hidden, (uint64_t)G::kHidden * sizeof(uint16_t), cudaMemcpyDeviceToHost);
			Require(Finite(b1_out) != 0u, G::Tag(), "5c B1 finite", ctx, 1u, 0u);
			for ( uint32_t ci = 0u; ci < 4u; ++ci )
			{
				const uint32_t rows = batch_cases[ci][0], commit = batch_cases[ci][1];
				SparkStageRunnerStep step;
				std::vector<uint16_t> row0(G::kHidden);
				cudaMemcpy(hidden, input_save, (uint64_t)rows * G::kHidden * sizeof(uint16_t), cudaMemcpyDeviceToDevice);
				BuildStep(&step, rows, ctx, commit, 0u);
				const SparkStatus st = M::Step(state, &step, stream);
				const cudaError_t cerr = cudaStreamSynchronize(stream);
				Require(st == SPARK_STATUS_OK && cerr == cudaSuccess && cudaGetLastError() == cudaSuccess, G::Tag(), "5c step ok", ctx, rows, (uint32_t)cerr);
				cudaMemcpy(row0.data(), hidden, (uint64_t)G::kHidden * sizeof(uint16_t), cudaMemcpyDeviceToHost);
				Require(memcmp(row0.data(), b1_out.data(), (uint64_t)G::kHidden * sizeof(uint16_t)) == 0,
					G::Tag(), "5c row0 == B1 bit for bit (ctx, rows, first_diff)", ctx, rows, FirstDifference(row0, b1_out, G::kHidden));
			}
		}
		Require(M::CheckErrors(state) == SPARK_STATUS_OK, G::Tag(), "5c KV clean", 0u, 0u, 0u);
		printf("ok %s 0-ULP 5c batch invariance: B1 == B4 == B8 == ver12 == ver64 at ctx=256 (dense) & ctx=4096 (sparse DSA)\n", G::Tag());
	}

	float TimeSubBlock(const SparkStageRunnerStep *step, uint32_t kind)
	{
		typename M::Frame frame;
		bool more = false;
		cudaEvent_t t0, t1;
		cudaEventCreate(&t0);
		cudaEventCreate(&t1);
		cudaMemcpy(hidden, input_save, (uint64_t)step->rows * G::kHidden * sizeof(uint16_t), cudaMemcpyDeviceToDevice);
		cudaMemcpy(state->normed, input_save, (uint64_t)step->rows * G::kHidden * sizeof(uint16_t), cudaMemcpyDeviceToDevice);
		cudaMemcpy(state->attention_out, input_save, (uint64_t)step->rows * G::kHidden * sizeof(uint16_t), cudaMemcpyDeviceToDevice);
		SparkStatus st = M::Begin(state, step, (void *)stream, &frame);
		if ( st == SPARK_STATUS_OK )
			st = M::NextWave(state, frame, &more);
		if ( st == SPARK_STATUS_OK && more )
			st = M::WaveBegin(state, frame);
		if ( st == SPARK_STATUS_OK && more )
			M::BindLayer(state, &frame.wave, (kind == 0u || kind == 2u) ? 0u : 1u, frame.hidden, frame.positions, frame.slots, &frame.b);
		auto run_one = [&]() -> SparkStatus
		{
			if ( kind == 0u )
				return M::Attention(state, frame, 0u);
			if ( kind == 1u )
				return M::Attention(state, frame, 1u);
			if ( kind == 2u )
				return M::DenseMlp(state, frame, 0u);
			return M::RoutedMlp(state, frame, 1u);
		};
		for ( uint32_t w = 0u; w < kBenchWarmup && st == SPARK_STATUS_OK; ++w )
			st = run_one();
		const cudaError_t werr = cudaStreamSynchronize(stream);
		Require(st == SPARK_STATUS_OK && werr == cudaSuccess && cudaGetLastError() == cudaSuccess, G::Tag(), "warmup ok", step->rows, kind, (uint32_t)werr);
		cudaEventRecord(t0, stream);
		for ( uint32_t it = 0u; it < kBenchIters && st == SPARK_STATUS_OK; ++it )
			st = run_one();
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
		struct Case { const char *name; uint32_t rows; uint32_t ctx_pos; uint32_t commit; uint32_t prefill; };
		static const Case cases[] =
		{
			{ "B1_short_256", 1u, 256u, 1u, 0u },
			{ "B1_long_4k", 1u, 4096u, 1u, 0u },
			{ "ver12", 12u, 512u, 0u, 0u },
			{ "B4", 4u, 512u, 1u, 0u },
			{ "B8", 8u, 512u, 1u, 0u },
			{ "B16", 16u, 512u, 1u, 0u },
			{ "B32", 32u, 512u, 1u, 0u },
			{ "B64", 64u, 512u, 1u, 0u },
			{ "prefill_512_d", 512u, 0u, 1u, 1u },
			{ "prefill_2048_d", 2048u, 0u, 1u, 1u },
			{ "prefill_2048_8k", 2048u, 6144u, 1u, 1u }
		};
		printf("\n=== BENCH %s (TP=%u, hidden=%u, full=%uL [%u idx, %u dense, %u routed]) ===\n",
			G::Tag(), G::kTpDegree, G::kHidden, G::kFullLayers, G::kFullIndexLayers, G::kFullDenseLayers, G::kFullRoutedLayers);
		printf("%-16s %8s %8s %8s %8s %10s %10s\n", "case", "AttnIdx", "AttnNoI", "DenseFF", "MoE_ms", "Model_ms", "tok/s");
		for ( uint32_t ci = 0u; ci < sizeof(cases) / sizeof(cases[0]); ++ci )
		{
			const Case &c = cases[ci];
			SparkStageRunnerStep step;
			BuildStep(&step, c.rows, c.ctx_pos, c.commit, c.prefill);
			const float a_idx = TimeSubBlock(&step, 0u);
			const float a_noi = (G::kFullIndexLayers < G::kFullLayers) ? TimeSubBlock(&step, 1u) : a_idx;
			const float d_ff = TimeSubBlock(&step, 2u);
			const float moe = TimeSubBlock(&step, 3u);
			const float total_ms = (float)G::kFullIndexLayers * a_idx + (float)(G::kFullLayers - G::kFullIndexLayers) * a_noi +
				(float)G::kFullDenseLayers * d_ff + (float)G::kFullRoutedLayers * moe;
			const float tps = total_ms > 0.0f ? ((float)c.rows * 1000.0f / total_ms) : 0.0f;
			printf("%-16s %8.3f %8.3f %8.3f %8.3f %10.2f %10.1f\n", c.name, a_idx, a_noi, d_ff, moe, total_ms, tps);
		}
	}

	void Close(void)
	{
		if ( state != 0 )
		{
			M::Release(state);
			cudaFree(state->arena);
			free(state);
		}
		for ( void *p : allocs )
			cudaFree(p);
		cudaFree(kv);
		cudaFree(page_table);
		cudaFree(positions);
		cudaFree(sequence_of_row);
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
	GlmBenchRig<G> rig;
	if ( rig.Open() == 0 )
	{
		Require(0, G::Tag(), "bench rig opens", 0u, 0u, 0u);
		return;
	}
	rig.Check5cInvariance();
	rig.BenchmarkAll();
	rig.Close();
}

int main(void)
{
	RunGeometry<Glm53Bench16Fp8>();
	RunGeometry<Glm53Bench16Bf16>();
	RunGeometry<Axk2Bench16Fp8>();
	if ( vf_failures != 0 )
	{
		printf("FAIL glm52 blackwell bench+exact: %d failures\n", vf_failures);
		return 1;
	}
	printf("\nPASS glm52 blackwell bench+exact: 0-ULP 5c verified and TP16 per-rank timings collected\n");
	return 0;
}
