#define GLM53_FULL_CONTRACT_SHA256 "tap-plan"
#define GLM53_FULL_MODEL_REVISION "tap-plan"
#define GLM53_FULL_LINEAR_CODEC SPARK_WEIGHT_CODEC_FP8_E4M3
#define GLM53_FULL_EXPERT_CODEC SPARK_WEIGHT_CODEC_FP8_E4M3
#define AXK2_CONTRACT_SHA256 "tap-plan"
#define AXK2_MODEL_REVISION "tap-plan"
#define GLM53_FLASH_EXPERT_CODEC SPARK_WEIGHT_CODEC_FP8_E4M3

#include "modules/glm53_full_tp16_resident_decode_stage/source/spark_glm53_full_tp16_stage_model.cu"
#include "modules/axk2_tp16_resident_decode_stage/source/spark_axk2_tp16_stage_model.cu"
#include "modules/glm53_flash_tp16_resident_decode_stage/source/spark_glm53_flash_tp16_stage_model.cu"
#include "modules/kexaone2_tp16_resident_decode_stage/source/spark_kexaone2_tp16_stage_model.cu"
#include "modules/gemma4_31b_resident_decode_stage/source/spark_gemma4_31b_stage_model.cu"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "tests/cuda/verify_frames_common.cuh"

#define TP_SENTINEL 0x5aa5u
#define TP_SLOTS 2u
#define TP_PAGES 8u
#define TP_FFN 256u
#define TP_GLM_MAX_ROWS 256u
#define TP_GEMMA_MAX_ROWS 32u
#define TP_G5N_DEGREE SPARK_GLM53_FLASH_TP16_STAGE_TP_DEGREE
#define TP_G5N_ROWS 8u
#define TP_G5N_PAGES 4u
#define TP_G5N_CONTEXT (TP_G5N_PAGES * G5N_PAGE_SLOTS)

struct TapGlm53 : Glm53FullGeometry
{
	static constexpr uint32_t kTpDegree = 1u;
	static constexpr uint32_t kIndexLayers = 2u;
	static constexpr uint32_t kHidden = 1024u;
	static constexpr uint32_t kLayers = 3u;
	static constexpr uint32_t kFirstRouted = 1u;
	static constexpr uint32_t kHeads = 4u;
	static constexpr uint32_t kQueryA = 256u;
	static constexpr uint32_t kSelected = 64u;
	static constexpr uint32_t kExperts = 16u;
	static constexpr uint32_t kExpertInter = 256u;
	static constexpr uint32_t kDenseInter = 512u;
	static uint32_t FullIndexer(uint32_t layer) { return layer != 1u ? 1u : 0u; }
	static const char *Tag(void) { return "glm-5.3"; }
};

struct TapAxk2 : Axk2Geometry
{
	static constexpr uint32_t kTpDegree = 1u;
	static constexpr uint32_t kIndexLayers = 3u;
	static constexpr uint32_t kHidden = 1024u;
	static constexpr uint32_t kLayers = 3u;
	static constexpr uint32_t kHeads = 4u;
	static constexpr uint32_t kQueryA = 256u;
	static constexpr uint32_t kSelected = 64u;
	static constexpr uint32_t kIndexHeads = 32u;
	static constexpr uint32_t kExperts = 32u;
	static constexpr uint32_t kExpertInter = 256u;
	static constexpr uint32_t kDenseInter = 512u;
	static uint32_t FullIndexer(uint32_t layer) { return layer < kLayers ? 1u : 0u; }
	static const char *Tag(void) { return "a.x-k2"; }
};

struct TapKexaone : Kexaone2Geometry<SPARK_KEXAONE2_TP16_STAGE_TP_DEGREE>
{
	static constexpr uint32_t kTpDegree = 1u;
	static constexpr uint32_t kHidden = 512u;
	static constexpr uint32_t kLayers = 4u;
	static constexpr uint32_t kSlidingHeads = 8u;
	static constexpr uint32_t kFullHeads = 8u;
	static constexpr uint32_t kSlidingKvHeads = 1u;
	static constexpr uint32_t kFullKvHeads = 1u;
	static constexpr uint32_t kExperts = 16u;
	static constexpr uint32_t kLocalExperts = 16u;
	static constexpr uint32_t kMoeInter = 128u;
	static constexpr float LayerSwigluLimit(uint32_t index) { return index == 3u ? 7.0f : INFINITY; }
	static const char *Tag(void) { return "k-exaone-2.0"; }
};

struct TapGemma31b : Gemma4_31bGeometry<SPARK_GEMMA4_31B_STAGE_TP_DEGREE>
{
	static constexpr uint32_t kTpDegree = 1u;
	static constexpr uint32_t kHidden = 512u;
	static constexpr uint32_t kLayers = 4u;
	static constexpr uint32_t kSlidingHeads = 4u;
	static constexpr uint32_t kFullHeads = 4u;
	static constexpr uint32_t kSlidingKvHeads = 2u;
	static constexpr uint32_t kFullKvHeads = 1u;
	static constexpr uint32_t kWindow = 24u;
	static const char *Tag(void) { return "gemma4-31b"; }
};

struct TapGlm5NextGeometry
{
	static constexpr uint32_t kTpDegree = TP_G5N_DEGREE;
	static constexpr uint32_t kExpertCodec = SPARK_WEIGHT_CODEC_FP8_E4M3;
	static constexpr uint32_t kVerifyRows = TP_G5N_ROWS;
	static const char *Tag(void) { return "glm-5.3-flash"; }
	static const char *WeightdModel(void) { return "glm5_next_stage"; }
	static const char *WeightdRevision(void) { return "tap-plan"; }
	static const char *ReportTag(void) { return "GLM5-NEXT-TAP"; }
};

static const uint8_t *Fp8Block(uint64_t groups, uint64_t rows, uint64_t columns, uint64_t *payload_bytes)
{
	const uint64_t blocks = (columns + 127u) / 128u, payload = groups * rows * columns;
	std::vector<uint8_t> host(payload + groups * rows * blocks * sizeof(float));
	std::vector<float> scales(groups * rows * blocks);
	const uint32_t seed = vf_seed++;
	for ( uint64_t index = 0u; index < payload; ++index )
		host[index] = E4m3(Uniform(seed, (uint32_t)index));
	for ( uint64_t index = 0u; index < scales.size(); ++index )
		scales[index] = (0.75f + 0.25f * Uniform(seed + 1u, (uint32_t)index)) * 1.7f / sqrtf((float)columns);
	memcpy(host.data() + payload, scales.data(), scales.size() * sizeof(float));
	*payload_bytes = payload;
	return Device(host.data(), host.size());
}

static GlmLinearWeight GlmFp8(uint64_t rows, uint64_t columns)
{
	uint64_t payload = 0u;
	GlmLinearWeight weight;
	weight.weight = Fp8Block(1u, rows, columns, &payload);
	weight.scale = (const float *)(weight.weight + payload);
	return weight;
}

static void FillSentinel(uint16_t *device, uint64_t count)
{
	std::vector<uint16_t> host(count, TP_SENTINEL);
	Require(cudaMemcpy(device, host.data(), count * sizeof(uint16_t), cudaMemcpyHostToDevice) == cudaSuccess, "sentinel", "upload", 0u, 0u, 0u);
}

static void CheckSentinelSlice(const char *tag, const char *what, const std::vector<uint16_t> &slab, uint64_t begin, uint64_t count)
{
	for ( uint64_t index = 0u; index < count; ++index )
		if ( slab[begin + index] != TP_SENTINEL )
		{
			Require(0, tag, what, (uint32_t)index, slab[begin + index], TP_SENTINEL);
			return;
		}
}

template<class M>
static void CheckTapPlanContract(const char *tag, typename M::State *state, uint32_t layers, uint16_t *slab, uint64_t stride)
{
	SparkStageRunnerTapPlan plan;
	memset(&plan, 0, sizeof(plan));
	Require(M::Interface()->tap_plan != 0, tag, "interface tap_plan", 0u, 0u, 0u);
	Require(M::TapPlan(0, &plan) == SPARK_STATUS_INVALID_ARGUMENT, tag, "null model", 0u, 0u, 0u);
	Require(M::TapPlan(state, 0) == SPARK_STATUS_INVALID_ARGUMENT, tag, "null plan", 0u, 0u, 0u);

	plan.entry_count = SPARK_STAGE_RUNNER_TAP_MAX_ENTRIES + 1u;
	Require(M::TapPlan(state, &plan) == SPARK_STATUS_INVALID_ARGUMENT, tag, "too many entries", 0u, 0u, 0u);

	memset(&plan, 0, sizeof(plan));
	plan.entry_count = 1u;
	plan.entries[0].form = SPARK_STAGE_RUNNER_TAP_FORM_OUTPUT;
	plan.entries[0].layer = layers;
	plan.capture = 1u;
	plan.slab = slab;
	plan.row_stride_elements = stride;
	Require(M::TapPlan(state, &plan) == SPARK_STATUS_UNSUPPORTED, tag, "out-of-range layer", layers, 0u, 0u);
	Require(state->tap_slab == 0, tag, "slab unchanged after invalid layer", 0u, 0u, 0u);

	plan.entries[0].form = 99u;
	plan.entries[0].layer = 0u;
	Require(M::TapPlan(state, &plan) == SPARK_STATUS_UNSUPPORTED, tag, "invalid form", 99u, 0u, 0u);

	plan.entry_count = 2u;
	plan.entries[0].form = SPARK_STAGE_RUNNER_TAP_FORM_OUTPUT;
	plan.entries[0].layer = 0u;
	plan.entries[1].form = SPARK_STAGE_RUNNER_TAP_FORM_FINAL;
	plan.entries[1].layer = 0u;
	plan.capture = 0u;
	plan.slab = slab;
	plan.row_stride_elements = stride;
	Require(M::TapPlan(state, &plan) == SPARK_STATUS_OK, tag, "capture=0 accepted", 0u, 0u, 0u);
	Require(state->tap_slab == 0, tag, "capture=0 clears tap_slab", 0u, 0u, 0u);
}

template<class G>
struct GlmTapRig
{
	using M = GlmModel<G>;
	using S = GlmShape<G>;
	using State = typename M::State;
	static constexpr uint64_t kPool = (uint64_t)TP_SLOTS * TP_PAGES;
	static constexpr uint64_t kLatentRegion = kPool * GlmKv<S>::kPageBytes;
	static constexpr uint64_t kIndexRegion = (uint64_t)G::kIndexLayers * kPool * GlmIndexKv<S>::kPageBytes;
	State *state;
	VfBus bus;
	SparkStageRunnerServices services;
	cudaStream_t stream;
	uint32_t multiprocessors;
	uint8_t *kv;
	uint32_t *page_table, *positions, *context, *sequence_of_row;
	uint16_t *hidden;
	const uint8_t *experts_up[S::kLayers], *experts_down[S::kLayers];
	uint32_t saw_ordered;
	uint32_t saw_row_sharded;
	uint32_t saw_waves;
	std::vector<uint16_t> ref_layers[S::kLayers];

	static SparkStatus Weights(void *context_pointer, uint32_t layer, const uint32_t *group_row_offset, uint32_t routed_rows, const void **w1, const void **w2)
	{
		const GlmTapRig *rig = (const GlmTapRig *)context_pointer;
		(void)group_row_offset;
		(void)routed_rows;
		*w1 = rig->experts_up[layer];
		*w2 = rig->experts_down[layer];
		return SPARK_STATUS_OK;
	}

	static SparkStatus Round(void *context_pointer, void *stream_pointer, uint32_t operation, uint32_t stage, uint32_t rows, uint32_t row_elements,
		const void *local, void *full)
	{
		return VfRound(&((GlmTapRig *)context_pointer)->bus, stream_pointer, operation, stage, rows, row_elements, local, full);
	}

	void RecordLayerRef(const typename M::Frame &frame, uint32_t layer)
	{
		const GlmWave *wave = &frame.wave;
		const uint32_t rows = wave->rows, sharded = M::RowSharded(wave);
		const uint64_t count = (uint64_t)rows * G::kHidden;
		std::vector<uint16_t> res(count), hid(count);
		Require(cudaStreamSynchronize(stream) == cudaSuccess, G::Tag(), "ref sync", layer, 0u, 0u);
		Require(cudaMemcpy(res.data(), state->residual, count * sizeof(uint16_t), cudaMemcpyDeviceToHost) == cudaSuccess, G::Tag(), "ref residual copy",
			layer, 0u, 0u);
		if ( sharded == 0u )
			Require(cudaMemcpy(hid.data(), frame.hidden, count * sizeof(uint16_t), cudaMemcpyDeviceToHost) == cudaSuccess, G::Tag(), "ref hidden copy",
				layer, 0u, 0u);
		for ( uint32_t row = 0u; row < rows; ++row )
		{
			const uint32_t dst_row = wave->ordered != 0u ? state->host_order[wave->first + row] : (wave->first + row);
			for ( uint32_t col = 0u; col < G::kHidden; ++col )
			{
				const uint64_t src = (uint64_t)row * G::kHidden + col;
				const uint64_t dst = (uint64_t)dst_row * G::kHidden + col;
				ref_layers[layer][dst] = sharded != 0u ? res[src] : Bf16(VfFloat(res[src]) + VfFloat(hid[src]));
			}
		}
	}

	void Layer(uint32_t index)
	{
		GlmLayer *layer = &state->layers[index];
		const uint32_t heads = state->heads;
		layer->attn_norm = Tensor<uint16_t>(G::kHidden, 0.2f, 1.0f);
		layer->q_a = GlmFp8(G::kQueryA, G::kHidden);
		layer->q_a_norm = Tensor<uint16_t>(G::kQueryA, 0.2f, 1.0f);
		layer->q_b = GlmFp8((uint64_t)heads * S::kQueryHead, S::kQueryInput);
		layer->kv_a = GlmFp8(S::kSlot, G::kHidden);
		layer->kv_a_norm = Tensor<uint16_t>(G::kLatent, 0.2f, 1.0f);
		layer->kv_b_key = Tensor<uint16_t>((uint64_t)heads * G::kLatent * G::kNope, 1.7f / sqrtf((float)G::kNope), 0.0f);
		layer->kv_b_value = Tensor<uint16_t>((uint64_t)heads * G::kValue * G::kLatent, 1.7f / sqrtf((float)G::kLatent), 0.0f);
		layer->o_proj = GlmFp8(G::kHidden, (uint64_t)heads * G::kValue);
		layer->post_norm = Tensor<uint16_t>(G::kHidden, 0.2f, 1.0f);
		if ( G::FullIndexer(index) != 0u )
		{
			layer->index_q = GlmFp8(S::kIndexQuery, G::kQueryA);
			layer->index_k = GlmFp8(G::kIndexDim, G::kHidden);
			layer->index_head = Matrix(G::kIndexHeads, G::kHidden);
			layer->index_norm_weight = Tensor<uint16_t>(G::kIndexDim, 0.2f, 1.0f);
			layer->index_norm_bias = Tensor<uint16_t>(G::kIndexDim, 0.1f, 0.0f);
		}
		if ( G::kNormGateRank != 0u )
		{
			layer->attn_gate_down = Matrix(G::kNormGateRank, G::kHidden);
			layer->attn_gate_up = Tensor<uint16_t>((uint64_t)G::kHidden * G::kNormGateRank, 0.3f, 0.0f);
			layer->post_gate_down = Matrix(G::kNormGateRank, G::kHidden);
			layer->post_gate_up = Tensor<uint16_t>((uint64_t)G::kHidden * G::kNormGateRank, 0.3f, 0.0f);
		}
		if ( index < G::kFirstRouted )
		{
			layer->dense_gate_up = GlmFp8(2u * G::kDenseInter, G::kHidden);
			layer->dense_down = GlmFp8(G::kHidden, G::kDenseInter);
			return;
		}
		layer->router = Matrix(G::kExperts, G::kHidden);
		layer->router_bias = Tensor<float>(G::kExperts, 0.05f, 0.0f);
		layer->shared_gate_up = GlmFp8(2u * G::kExpertInter, G::kHidden);
		layer->shared_down = GlmFp8(G::kHidden, G::kExpertInter);
		experts_up[index] = Fp8Block(G::kExperts, 2u * G::kExpertInter, G::kHidden, &layer->up_gate_scale);
		experts_down[index] = Fp8Block(G::kExperts, G::kHidden, G::kExpertInter, &layer->down_scale);
		state->w1_offset[index] = 0u;
		state->w2_offset[index] = 0u;
	}

	int Open(void)
	{
		std::vector<uint32_t> table(TP_SLOTS * TP_PAGES);
		SparkStageRunnerKv attach;
		bus = {};
		bus.hidden = G::kHidden;
		memset(&services, 0, sizeof(services));
		services.context = this;
		services.round = Round;
		services.expert_weights = Weights;
		state = (State *)calloc(1u, sizeof(State));
		for ( uint32_t layer = 0u; layer < S::kLayers; ++layer )
			state->tap_output[layer] = -1;
		state->services = &services;
		state->tp_degree = 1u;
		state->heads = G::kHeads;
		state->max_rows = TP_GLM_MAX_ROWS;
		state->scatter_rows = TP_GLM_MAX_ROWS;
		state->slots = TP_SLOTS;
		state->max_positions = TP_PAGES * GLM_PAGE_SLOTS;
		state->shard.degree = 1u;
		state->shard.grain = 1u;
		if ( M::Size(state) != SPARK_STATUS_OK )
			return 0;
		for ( uint32_t index = 0u; index < G::kLayers; ++index )
			Layer(index);
		state->final_norm = Tensor<uint16_t>(G::kHidden, 0.2f, 1.0f);
		if ( M::Allocate(state) != SPARK_STATUS_OK || cudaStreamCreate(&stream) != cudaSuccess ||
			cudaDeviceGetAttribute((int *)&multiprocessors, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess ||
			cudaMalloc((void **)&kv, kLatentRegion + kIndexRegion) != cudaSuccess || cudaMemset(kv, 0, kLatentRegion + kIndexRegion) != cudaSuccess ||
			cudaMalloc((void **)&page_table, table.size() * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&positions, TP_GLM_MAX_ROWS * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&sequence_of_row, TP_GLM_MAX_ROWS * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&context, TP_SLOTS * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&hidden, (uint64_t)TP_GLM_MAX_ROWS * G::kHidden * sizeof(uint16_t)) != cudaSuccess )
			return 0;
		for ( uint32_t index = 0u; index < table.size(); ++index )
			table[index] = index;
		if ( cudaMemcpy(page_table, table.data(), table.size() * sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess )
			return 0;
		memset(&attach, 0, sizeof(attach));
		attach.pool = kv;
		attach.layer_page_bytes = (uint64_t)GLM_PAGE_SLOTS * S::kSlotBytes;
		attach.layer_stride_bytes = attach.layer_page_bytes;
		attach.layer_count = S::kLayers;
		attach.page_table = page_table;
		attach.page_table_stride = TP_PAGES;
		attach.pool_page_count = (uint32_t)kPool;
		attach.sequence_count = TP_SLOTS;
		attach.context_shard = state->shard;
		attach.second_pool = kv + kLatentRegion;
		attach.second_layer_stride_bytes = kPool * GlmIndexKv<S>::kPageBytes;
		attach.second_layer_page_bytes = GlmIndexKv<S>::kPageBytes;
		attach.second_layer_count = G::kIndexLayers;
		return M::AttachKv(state, &attach) == SPARK_STATUS_OK;
	}

	void RunStep(const std::vector<uint16_t> &input, const std::vector<uint32_t> &pos, const std::vector<uint32_t> &seq, uint32_t commit,
		uint32_t dispatch_row_offset, uint32_t record_ref, std::vector<uint16_t> *out)
	{
		const uint32_t rows = (uint32_t)pos.size();
		uint32_t host_context[TP_SLOTS] = { 0u, 0u }, seen[TP_SLOTS] = { 0u, 0u }, sequences = 0u;
		SparkStageRunnerStep step;
		for ( uint32_t row = 0u; row < rows; ++row )
		{
			if ( seen[seq[row]] == 0u )
				sequences++;
			seen[seq[row]] = 1u;
			if ( pos[row] + 1u > host_context[seq[row]] )
				host_context[seq[row]] = pos[row] + 1u;
		}
		memset(&step, 0, sizeof(step));
		step.hidden_bf16 = hidden;
		step.positions = positions;
		step.context_length = context;
		step.sequence_of_row = sequence_of_row;
		step.rows = rows;
		step.sequences = sequences;
		step.commit = commit;
		step.dispatch_row_offset = dispatch_row_offset;
		step.context = state->max_positions;
		step.multiprocessors = multiprocessors;
		step.route_fault = VfRouteFault();
		step.host_positions = pos.data();
		step.host_sequence_of_row = seq.data();
		step.graph_key = M::GraphKey(state, &step);
		saw_ordered = 0u;
		saw_row_sharded = 0u;
		saw_waves = 0u;
		if ( record_ref != 0u )
			for ( uint32_t layer = 0u; layer < S::kLayers; ++layer )
				ref_layers[layer].assign((uint64_t)rows * G::kHidden, 0u);
		Require(cudaMemsetAsync(kv, 0, kLatentRegion + kIndexRegion, stream) == cudaSuccess, G::Tag(), "kv reset", 0u, 0u, 0u);
		Require(cudaMemcpyAsync(hidden, input.data(), input.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, stream) == cudaSuccess, G::Tag(), "in copy", 0u, 0u, 0u);
		Require(cudaMemcpyAsync(positions, pos.data(), rows * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) == cudaSuccess, G::Tag(), "pos copy", 0u, 0u, 0u);
		Require(cudaMemcpyAsync(sequence_of_row, seq.data(), rows * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) == cudaSuccess, G::Tag(), "seq copy", 0u, 0u, 0u);
		Require(cudaMemcpyAsync(context, host_context, sizeof(host_context), cudaMemcpyHostToDevice, stream) == cudaSuccess, G::Tag(), "ctx copy", 0u, 0u, 0u);
		typename M::Frame frame;
		uint32_t more = 0u;
		SparkStatus status = M::Schedule::Begin(state, &step, stream, &frame);
		Require(status == SPARK_STATUS_OK, G::Tag(), "step begin", rows, commit, 0u);
		for ( status = M::Schedule::Wave(state, frame, &more); status == SPARK_STATUS_OK && more != 0u; status = M::Schedule::Wave(state, frame, &more) )
		{
			saw_waves++;
			if ( frame.wave.ordered != 0u )
				saw_ordered = 1u;
			if ( M::RowSharded(&frame.wave) != 0u )
				saw_row_sharded = 1u;
			for ( uint32_t layer = 0u; layer < S::kLayers && status == SPARK_STATUS_OK; ++layer )
			{
				status = M::Schedule::Layer(state, frame, layer);
				if ( status == SPARK_STATUS_OK && record_ref != 0u )
					RecordLayerRef(frame, layer);
			}
			if ( status == SPARK_STATUS_OK )
				status = M::Schedule::WaveEnd(state, frame);
		}
		status = M::Schedule::End(state, frame, status);
		Require(status == SPARK_STATUS_OK, G::Tag(), "step", rows, commit, 0u);
		Require(cudaStreamSynchronize(stream) == cudaSuccess, G::Tag(), "step sync", rows, commit, 0u);
		if ( out != 0 )
		{
			out->resize(input.size());
			Require(cudaMemcpy(out->data(), hidden, input.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) == cudaSuccess, G::Tag(), "out copy", 0u, 0u, 0u);
		}
	}
};

template<class G>
static void TestGlmFamily(void)
{
	GlmTapRig<G> rig;
	Require(rig.Open(), G::Tag(), "open", 0u, 0u, 0u);
	const uint32_t total_slab_rows = TP_GLM_MAX_ROWS + 4u;
	const uint32_t max_entries = 3u;
	const uint64_t stride = (uint64_t)max_entries * G::kHidden;
	uint16_t *tap_slab = 0;
	Require(cudaMalloc((void **)&tap_slab, (size_t)total_slab_rows * stride * sizeof(uint16_t)) == cudaSuccess, G::Tag(), "tap slab alloc", 0u, 0u, 0u);

	CheckTapPlanContract<GlmModel<G>>(G::Tag(), rig.state, G::kLayers, tap_slab, stride);

	{
		const uint32_t rows = 4u, offset = 2u;
		std::vector<uint16_t> input((uint64_t)rows * G::kHidden), out_tapped, out_plain;
		std::vector<uint32_t> pos = { 0u, 1u, 2u, 3u }, seq(rows, 0u);
		for ( uint64_t index = 0u; index < input.size(); ++index )
			input[index] = Bf16(Uniform(101u, (uint32_t)index));
		SparkStageRunnerTapPlan plan;
		memset(&plan, 0, sizeof(plan));
		plan.entry_count = 3u;
		plan.entries[0] = { SPARK_STAGE_RUNNER_TAP_FORM_OUTPUT, 0u };
		plan.entries[1] = { SPARK_STAGE_RUNNER_TAP_FORM_OUTPUT, 2u };
		plan.entries[2] = { SPARK_STAGE_RUNNER_TAP_FORM_FINAL, 0u };
		plan.capture = 1u;
		plan.row_capacity = total_slab_rows;
		plan.slab = tap_slab;
		plan.row_stride_elements = stride;
		FillSentinel(tap_slab, (uint64_t)total_slab_rows * stride);
		Require(GlmModel<G>::TapPlan(rig.state, &plan) == SPARK_STATUS_OK, G::Tag(), "unordered plan", 0u, 0u, 0u);
		rig.RunStep(input, pos, seq, 0u, offset, 1u, &out_tapped);
		std::vector<uint16_t> host_slab((uint64_t)total_slab_rows * stride);
		Require(cudaMemcpy(host_slab.data(), tap_slab, host_slab.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) == cudaSuccess, G::Tag(), "slab read", 0u, 0u, 0u);
		CheckSentinelSlice(G::Tag(), "prefix untouched", host_slab, 0u, (uint64_t)offset * stride);
		for ( uint32_t row = 0u; row < rows; ++row )
		{
			const uint64_t base = (uint64_t)(offset + row) * stride;
			for ( uint32_t col = 0u; col < G::kHidden; ++col )
			{
				if ( host_slab[base + col] != rig.ref_layers[0][row * G::kHidden + col] )
					{ Require(0, G::Tag(), "unordered o0 mismatch", row, col, 0u); break; }
				if ( host_slab[base + G::kHidden + col] != rig.ref_layers[2][row * G::kHidden + col] )
					{ Require(0, G::Tag(), "unordered o2 mismatch", row, col, 0u); break; }
			}
			CheckSentinelSlice(G::Tag(), "final entry untouched by model", host_slab, base + 2u * G::kHidden, G::kHidden);
		}
		plan.capture = 0u;
		FillSentinel(tap_slab, (uint64_t)total_slab_rows * stride);
		Require(GlmModel<G>::TapPlan(rig.state, &plan) == SPARK_STATUS_OK, G::Tag(), "disable capture", 0u, 0u, 0u);
		rig.RunStep(input, pos, seq, 0u, offset, 0u, &out_plain);
		Require(out_tapped == out_plain, G::Tag(), "taps do not change hidden output", 0u, 0u, 0u);
		Require(cudaMemcpy(host_slab.data(), tap_slab, host_slab.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) == cudaSuccess, G::Tag(), "slab read 2", 0u, 0u, 0u);
		CheckSentinelSlice(G::Tag(), "capture=0 leaves slab untouched", host_slab, 0u, (uint64_t)total_slab_rows * stride);
	}

	{
		const uint32_t rows = 6u;
		std::vector<uint16_t> input((uint64_t)rows * G::kHidden);
		std::vector<uint32_t> pos = { 0u, 0u, 1u, 1u, 2u, 2u }, seq = { 1u, 0u, 1u, 0u, 1u, 0u };
		for ( uint64_t index = 0u; index < input.size(); ++index )
			input[index] = Bf16(Uniform(202u, (uint32_t)index));
		SparkStageRunnerTapPlan plan;
		memset(&plan, 0, sizeof(plan));
		plan.entry_count = 3u;
		plan.entries[0] = { SPARK_STAGE_RUNNER_TAP_FORM_OUTPUT, 1u };
		plan.entries[1] = { SPARK_STAGE_RUNNER_TAP_FORM_OUTPUT, 2u };
		plan.entries[2] = { SPARK_STAGE_RUNNER_TAP_FORM_OUTPUT, 0u };
		plan.capture = 1u;
		plan.row_capacity = total_slab_rows;
		plan.slab = tap_slab;
		plan.row_stride_elements = stride;
		FillSentinel(tap_slab, (uint64_t)total_slab_rows * stride);
		Require(GlmModel<G>::TapPlan(rig.state, &plan) == SPARK_STATUS_OK, G::Tag(), "ordered plan", 0u, 0u, 0u);
		rig.RunStep(input, pos, seq, 1u, 0u, 1u, 0);
		Require(rig.saw_ordered == 1u && rig.saw_row_sharded == 0u && rig.saw_waves == 2u, G::Tag(), "ordered multi-wave", rig.saw_ordered, rig.saw_row_sharded,
			rig.saw_waves);
		std::vector<uint16_t> host_slab((uint64_t)rows * stride);
		Require(cudaMemcpy(host_slab.data(), tap_slab, host_slab.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) == cudaSuccess, G::Tag(), "ordered read", 0u, 0u, 0u);
		const uint32_t layers[3] = { 1u, 2u, 0u };
		for ( uint32_t row = 0u; row < rows; ++row )
			for ( uint32_t entry = 0u; entry < 3u; ++entry )
				for ( uint32_t col = 0u; col < G::kHidden; ++col )
					if ( host_slab[(uint64_t)row * stride + (uint64_t)entry * G::kHidden + col] != rig.ref_layers[layers[entry]][(uint64_t)row * G::kHidden + col] )
						{ Require(0, G::Tag(), "ordered scatter mismatch", row, layers[entry], col); break; }
	}

	{
		const uint32_t rows = TP_GLM_MAX_ROWS;
		std::vector<uint16_t> input((uint64_t)rows * G::kHidden);
		std::vector<uint32_t> pos(rows), seq(rows, 0u);
		for ( uint32_t row = 0u; row < rows; ++row )
			pos[row] = 64u + row;
		for ( uint64_t index = 0u; index < input.size(); ++index )
			input[index] = Bf16(Uniform(303u, (uint32_t)index));
		SparkStageRunnerTapPlan plan;
		memset(&plan, 0, sizeof(plan));
		plan.entry_count = 2u;
		plan.entries[0] = { SPARK_STAGE_RUNNER_TAP_FORM_OUTPUT, 0u };
		plan.entries[1] = { SPARK_STAGE_RUNNER_TAP_FORM_OUTPUT, 2u };
		plan.capture = 1u;
		plan.row_capacity = total_slab_rows;
		plan.slab = tap_slab;
		plan.row_stride_elements = 2u * G::kHidden;
		FillSentinel(tap_slab, (uint64_t)rows * 2u * G::kHidden);
		Require(GlmModel<G>::TapPlan(rig.state, &plan) == SPARK_STATUS_OK, G::Tag(), "row-sharded plan", 0u, 0u, 0u);
		rig.RunStep(input, pos, seq, 1u, 0u, 1u, 0);
		Require(rig.saw_row_sharded == 1u && rig.saw_waves == 1u, G::Tag(), "row-sharded wave", rig.saw_row_sharded, rig.saw_waves, 0u);
		std::vector<uint16_t> host_slab((uint64_t)rows * 2u * G::kHidden);
		Require(cudaMemcpy(host_slab.data(), tap_slab, host_slab.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) == cudaSuccess, G::Tag(), "row-sharded read", 0u, 0u, 0u);
		const uint32_t layers[2] = { 0u, 2u };
		for ( uint32_t row = 0u; row < rows; ++row )
			for ( uint32_t entry = 0u; entry < 2u; ++entry )
				for ( uint32_t col = 0u; col < G::kHidden; ++col )
					if ( host_slab[(uint64_t)row * 2u * G::kHidden + (uint64_t)entry * G::kHidden + col] != rig.ref_layers[layers[entry]][(uint64_t)row * G::kHidden + col] )
						{ Require(0, G::Tag(), "row-sharded tap mismatch", row, layers[entry], col); break; }
	}
	cudaFree(tap_slab);
	printf("PASS tap plan %s (unordered, ordered, row-sharded)\n", G::Tag());
}

template<class G>
struct GemmaTapRig
{
	using M = GemmaModel<G>;
	using S = GemmaShape<G>;
	using State = GemmaState<G>;
	static constexpr uint64_t kSlidingRegion = (uint64_t)S::kSlidingLayers * TP_SLOTS * TP_PAGES * S::SlidingKvGeometry::kPageBytes;
	static constexpr uint64_t kFullRegion = (uint64_t)S::kFullLayers * TP_SLOTS * TP_PAGES * S::FullKvGeometry::kPageBytes;
	State *state;
	VfBus bus;
	SparkStageRunnerServices services;
	cudaStream_t stream;
	uint32_t multiprocessors;
	uint8_t *kv;
	uint32_t *page_table, *positions, *context, *sequence_of_row;
	uint16_t *hidden;
	uint32_t capture_ref;
	uint32_t active_rows;
	std::vector<uint16_t> ref_layers[G::kLayers];

	static SparkStatus Round(void *context_pointer, void *stream_pointer, uint32_t operation, uint32_t stage, uint32_t rows, uint32_t row_elements,
		const void *local, void *full)
	{
		return VfRound(&((GemmaTapRig *)context_pointer)->bus, stream_pointer, operation, stage, rows, row_elements, local, full);
	}

	static void OnLayerDone(void *context_pointer, uint32_t layer)
	{
		GemmaTapRig *rig = (GemmaTapRig *)context_pointer;
		if ( rig->capture_ref == 0u || layer >= G::kLayers )
			return;
		const uint64_t count = (uint64_t)rig->active_rows * G::kHidden;
		rig->ref_layers[layer].resize(count);
		Require(cudaStreamSynchronize(rig->stream) == cudaSuccess, G::Tag(), "gemma ref sync", layer, 0u, 0u);
		Require(cudaMemcpy(rig->ref_layers[layer].data(), rig->hidden, count * sizeof(uint16_t), cudaMemcpyDeviceToHost) == cudaSuccess, G::Tag(),
			"gemma ref copy", layer, 0u, 0u);
	}

	void Layer(GemmaLayer *layer, uint32_t index)
	{
		const bool full = S::Full(index);
		const uint32_t fused = full ? S::kFullFused : S::kSlidingFused, q = full ? S::kFullQ : S::kSlidingQ, dim = full ? G::kFullDim : G::kSlidingDim;
		layer->qkv = Matrix(fused, G::kHidden);
		layer->out = Matrix(G::kHidden, q);
		layer->gate_up = Matrix(2u * TP_FFN, G::kHidden);
		layer->down = Matrix(G::kHidden, TP_FFN);
		layer->ffn_rows = TP_FFN;
		layer->input_norm = Norm(G::kHidden);
		layer->pre_ff_norm = Norm(G::kHidden);
		if ( !G::kPreNorm )
		{
			layer->post_attn_norm = Norm(G::kHidden);
			layer->post_ff_norm = Norm(G::kHidden);
		}
		if ( G::kLayerScalar )
			layer->layer_scalar = Tensor<float>(1u, 0.1f, 0.8f);
		if ( G::kQkNormWeighted )
		{
			layer->q_norm = Norm(dim);
			layer->k_norm = Norm(dim);
		}
		if ( G::kOutputGate )
			layer->gate = Matrix(q, G::kHidden);
		if ( G::kHeadGate )
			layer->gate = Matrix(full ? S::kFullGateRows : S::kSlidingGateRows, G::kHidden);
		if ( G::kMoe && (G::kSigmoidRouter ? index >= G::kFirstMoeLayer : true) )
		{
			layer->router_proj = Matrix(G::kExperts, G::kHidden);
			layer->moe_w1 = Matrix((uint64_t)G::kLocalExperts * G::kMoeInter, G::kHidden);
			layer->moe_w3 = Matrix((uint64_t)G::kLocalExperts * G::kMoeInter, G::kHidden);
			layer->moe_w2 = Tensor<uint16_t>((uint64_t)G::kLocalExperts * G::kHidden * G::kMoeInter, 1.7f / sqrtf((float)G::kMoeInter), 0.0f);
		}
		if ( G::kMoe && G::kSigmoidRouter && index >= G::kFirstMoeLayer )
			layer->expert_bias = Tensor<float>(G::kExperts, 0.05f, 0.0f);
		if ( G::kMoe && !G::kSigmoidRouter )
		{
			layer->router_scale = Norm(G::kHidden);
			layer->expert_scale = Norm(G::kExperts);
			layer->post_ff_norm_1 = Norm(G::kHidden);
			layer->pre_ff_norm_2 = Norm(G::kHidden);
			layer->post_ff_norm_2 = Norm(G::kHidden);
		}
	}

	int Open(void)
	{
		std::vector<uint32_t> table(TP_SLOTS * TP_PAGES);
		SparkStageRunnerKv attach;
		bus = {};
		bus.hidden = G::kHidden;
		memset(&services, 0, sizeof(services));
		services.context = this;
		services.round = Round;
		services.layer_done = OnLayerDone;
		state = (State *)calloc(1u, sizeof(State));
		if ( M::Prepare(state, 0) != SPARK_STATUS_OK )
			return 0;
		state->services = &services;
		state->tp_degree = 1u;
		state->max_rows = TP_GEMMA_MAX_ROWS;
		state->slots = TP_SLOTS;
		state->ffn_rows = TP_FFN;
		if ( M::Size(state) != SPARK_STATUS_OK )
			return 0;
		for ( uint32_t index = 0u; index < G::kLayers; ++index )
			Layer(&state->layers[index], index);
		if ( M::Allocate(state) != SPARK_STATUS_OK || cudaStreamCreate(&stream) != cudaSuccess ||
			cudaDeviceGetAttribute((int *)&multiprocessors, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess ||
			cudaMalloc((void **)&kv, kSlidingRegion + kFullRegion) != cudaSuccess || cudaMemset(kv, 0, kSlidingRegion + kFullRegion) != cudaSuccess ||
			cudaMalloc((void **)&page_table, table.size() * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&positions, TP_GEMMA_MAX_ROWS * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&sequence_of_row, TP_GEMMA_MAX_ROWS * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&context, TP_SLOTS * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&hidden, (uint64_t)TP_GEMMA_MAX_ROWS * G::kHidden * sizeof(uint16_t)) != cudaSuccess )
			return 0;
		for ( uint32_t index = 0u; index < table.size(); ++index )
			table[index] = index;
		if ( cudaMemcpy(page_table, table.data(), table.size() * sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess )
			return 0;
		memset(&attach, 0, sizeof(attach));
		attach.pool = kv;
		attach.layer_stride_bytes = (uint64_t)TP_SLOTS * TP_PAGES * S::SlidingKvGeometry::kPageBytes;
		attach.layer_page_bytes = S::SlidingKvGeometry::kPageBytes;
		attach.layer_count = S::kSlidingLayers;
		attach.page_table = page_table;
		attach.page_table_stride = TP_PAGES;
		attach.pool_page_count = TP_SLOTS * TP_PAGES;
		attach.sequence_count = TP_SLOTS;
		attach.context_shard.degree = 1u;
		attach.context_shard.grain = 1u;
		attach.second_pool = kv + kSlidingRegion;
		attach.second_layer_stride_bytes = (uint64_t)TP_SLOTS * TP_PAGES * S::FullKvGeometry::kPageBytes;
		attach.second_layer_page_bytes = S::FullKvGeometry::kPageBytes;
		attach.second_layer_count = S::kFullLayers;
		return M::AttachKv(state, &attach) == SPARK_STATUS_OK;
	}

	void RunStep(const std::vector<uint16_t> &input, uint32_t rows, uint32_t dispatch_row_offset, uint32_t record_ref, std::vector<uint16_t> *out)
	{
		std::vector<uint32_t> pos(rows), seq(rows, 0u);
		uint32_t host_context[TP_SLOTS] = { rows, 0u };
		SparkStageRunnerStep step;
		for ( uint32_t row = 0u; row < rows; ++row )
			pos[row] = row;
		memset(&step, 0, sizeof(step));
		step.hidden_bf16 = hidden;
		step.positions = positions;
		step.context_length = context;
		step.sequence_of_row = sequence_of_row;
		step.rows = rows;
		step.sequences = 1u;
		step.commit = 1u;
		step.dispatch_row_offset = dispatch_row_offset;
		step.context = TP_PAGES * G::kPageSlots;
		step.multiprocessors = multiprocessors;
		step.route_fault = VfRouteFault();
		active_rows = rows;
		capture_ref = record_ref;
		Require(cudaMemsetAsync(kv, 0, kSlidingRegion + kFullRegion, stream) == cudaSuccess, G::Tag(), "gemma kv reset", 0u, 0u, 0u);
		Require(cudaMemcpyAsync(hidden, input.data(), input.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, stream) == cudaSuccess, G::Tag(), "gemma in copy", 0u, 0u, 0u);
		Require(cudaMemcpyAsync(positions, pos.data(), rows * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) == cudaSuccess, G::Tag(), "gemma pos copy", 0u, 0u, 0u);
		Require(cudaMemcpyAsync(sequence_of_row, seq.data(), rows * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) == cudaSuccess, G::Tag(), "gemma seq copy", 0u, 0u, 0u);
		Require(cudaMemcpyAsync(context, host_context, sizeof(host_context), cudaMemcpyHostToDevice, stream) == cudaSuccess, G::Tag(), "gemma ctx copy", 0u, 0u, 0u);
		Require(M::Step(state, &step, stream) == SPARK_STATUS_OK, G::Tag(), "gemma step", rows, 0u, 0u);
		Require(cudaStreamSynchronize(stream) == cudaSuccess, G::Tag(), "gemma sync", rows, 0u, 0u);
		capture_ref = 0u;
		if ( out != 0 )
		{
			out->resize(input.size());
			Require(cudaMemcpy(out->data(), hidden, input.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) == cudaSuccess, G::Tag(), "gemma out copy", 0u, 0u, 0u);
		}
	}
};

template<class G>
static void TestGemmaFamily(void)
{
	GemmaTapRig<G> rig;
	Require(rig.Open(), G::Tag(), "open", 0u, 0u, 0u);
	for ( uint32_t layer = 0u; layer < G::kLayers; ++layer )
		Require(rig.state->tap_output[layer] == -1, G::Tag(), "tap_output init -1", layer, 0u, 0u);
	const uint32_t rows = 5u, offset = 3u, slab_rows = rows + offset + 2u, entries = 4u;
	const uint64_t stride = (uint64_t)entries * G::kHidden;
	uint16_t *tap_slab = 0;
	Require(cudaMalloc((void **)&tap_slab, (size_t)slab_rows * stride * sizeof(uint16_t)) == cudaSuccess, G::Tag(), "slab alloc", 0u, 0u, 0u);
	CheckTapPlanContract<GemmaModel<G>>(G::Tag(), rig.state, G::kLayers, tap_slab, stride);

	std::vector<uint16_t> input((uint64_t)rows * G::kHidden), out_tapped, out_plain;
	for ( uint64_t index = 0u; index < input.size(); ++index )
		input[index] = Bf16(Uniform(404u, (uint32_t)index));
	SparkStageRunnerTapPlan plan;
	memset(&plan, 0, sizeof(plan));
	plan.entry_count = entries;
	plan.entries[0] = { SPARK_STAGE_RUNNER_TAP_FORM_OUTPUT, 0u };
	plan.entries[1] = { SPARK_STAGE_RUNNER_TAP_FORM_OUTPUT, 2u };
	plan.entries[2] = { SPARK_STAGE_RUNNER_TAP_FORM_OUTPUT, 3u };
	plan.entries[3] = { SPARK_STAGE_RUNNER_TAP_FORM_FINAL, 0u };
	plan.capture = 1u;
	plan.row_capacity = slab_rows;
	plan.slab = tap_slab;
	plan.row_stride_elements = stride;
	FillSentinel(tap_slab, (uint64_t)slab_rows * stride);
	Require(GemmaModel<G>::TapPlan(rig.state, &plan) == SPARK_STATUS_OK, G::Tag(), "tap plan", 0u, 0u, 0u);
	rig.RunStep(input, rows, offset, 1u, &out_tapped);

	std::vector<uint16_t> host_slab((uint64_t)slab_rows * stride);
	Require(cudaMemcpy(host_slab.data(), tap_slab, host_slab.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) == cudaSuccess, G::Tag(), "slab read", 0u, 0u, 0u);
	CheckSentinelSlice(G::Tag(), "prefix untouched", host_slab, 0u, (uint64_t)offset * stride);
	const uint32_t layers[3] = { 0u, 2u, 3u };
	for ( uint32_t row = 0u; row < rows; ++row )
	{
		const uint64_t base = (uint64_t)(offset + row) * stride;
		for ( uint32_t entry = 0u; entry < 3u; ++entry )
			for ( uint32_t col = 0u; col < G::kHidden; ++col )
				if ( host_slab[base + (uint64_t)entry * G::kHidden + col] != rig.ref_layers[layers[entry]][(uint64_t)row * G::kHidden + col] )
					{ Require(0, G::Tag(), "layer tap mismatch", row, layers[entry], col); break; }
		CheckSentinelSlice(G::Tag(), "final entry untouched", host_slab, base + 3u * G::kHidden, G::kHidden);
	}
	plan.capture = 0u;
	FillSentinel(tap_slab, (uint64_t)slab_rows * stride);
	Require(GemmaModel<G>::TapPlan(rig.state, &plan) == SPARK_STATUS_OK, G::Tag(), "disable capture", 0u, 0u, 0u);
	rig.RunStep(input, rows, offset, 0u, &out_plain);
	Require(out_tapped == out_plain, G::Tag(), "taps preserve output", 0u, 0u, 0u);
	Require(cudaMemcpy(host_slab.data(), tap_slab, host_slab.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) == cudaSuccess, G::Tag(), "slab read 2", 0u, 0u, 0u);
	CheckSentinelSlice(G::Tag(), "capture=0 leaves slab untouched", host_slab, 0u, (uint64_t)slab_rows * stride);
	cudaFree(tap_slab);
	printf("PASS tap plan %s\n", G::Tag());
}

struct Glm5NextTapRig
{
	using Stage = Glm5NextStage<TapGlm5NextGeometry>;
	Glm5NextModelState *state;
	SparkStageRunnerServices services;
	cudaStream_t stream;
	uint32_t sms;
	const void *expert_w1;
	const void *expert_w2;
	uint8_t *latent_pool;
	uint8_t *index_pool;
	uint64_t latent_bytes;
	uint64_t index_bytes;
	uint32_t *meta;
	uint16_t *hidden;
	uint32_t capture_ref;
	uint32_t active_rows;
	std::vector<uint16_t> ref_layers[G5N_LAYERS];

	static SparkStatus Round(void *context, void *stream_void, uint32_t operation, uint32_t stage, uint32_t rows, uint32_t row_elements,
		const void *local, void *full)
	{
		const uint64_t block = (uint64_t)rows * (row_elements != 0u ? row_elements : G5N_HIDDEN) * sizeof(uint16_t);
		(void)context;
		(void)stage;
		if ( operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16 )
			return local == full || cudaMemcpyAsync(full, local, block, cudaMemcpyDeviceToDevice, (cudaStream_t)stream_void) == cudaSuccess ?
				SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
		if ( operation != SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER && operation != SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL )
			return SPARK_STATUS_UNSUPPORTED;
		for ( uint32_t source = 0u; source < TP_G5N_DEGREE; ++source )
			if ( cudaMemcpyAsync((uint8_t *)full + source * block, local, block, cudaMemcpyDeviceToDevice, (cudaStream_t)stream_void) != cudaSuccess )
				return SPARK_STATUS_IO_ERROR;
		return SPARK_STATUS_OK;
	}

	static void OnFail(void *context, SparkStatus status)
	{
		(void)context;
		Require(0, "glm-5.3-flash", "service fail", (uint32_t)status, 0u, 0u);
	}

	static SparkStatus Experts(void *context, uint32_t layer, const uint32_t *offsets, uint32_t routed_rows, const void **w1, const void **w2)
	{
		const Glm5NextTapRig *rig = (const Glm5NextTapRig *)context;
		(void)layer;
		(void)offsets;
		(void)routed_rows;
		*w1 = rig->expert_w1;
		*w2 = rig->expert_w2;
		return SPARK_STATUS_OK;
	}

	static void OnLayerDone(void *context, uint32_t layer)
	{
		(void)context;
		(void)layer;
	}

	static void OnRowState(void *context, uint32_t layer, uint32_t phase)
	{
		Glm5NextTapRig *rig = (Glm5NextTapRig *)context;
		if ( rig->capture_ref == 0u || phase != SPARK_STAGE_RUNNER_ROW_STATE_READ || layer == 0u || layer >= G5N_LAYERS )
			return;
		const uint32_t prev = layer - 1u, rows = rig->active_rows;
		std::vector<uint16_t> streams((uint64_t)rows * G5N_HC * G5N_HIDDEN);
		Require(cudaStreamSynchronize(rig->stream) == cudaSuccess, "glm-5.3-flash", "row_state sync", prev, 0u, 0u);
		Require(cudaMemcpy(streams.data(), rig->state->streams, streams.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) == cudaSuccess, "glm-5.3-flash",
			"streams copy", prev, 0u, 0u);
		rig->ref_layers[prev].resize((uint64_t)rows * G5N_HIDDEN);
		for ( uint32_t row = 0u; row < rows; ++row )
			for ( uint32_t col = 0u; col < G5N_HIDDEN; ++col )
			{
				float total = 0.0f;
				for ( uint32_t s = 0u; s < G5N_HC; ++s )
					total += VfFloat(streams[((uint64_t)row * G5N_HC + s) * G5N_HIDDEN + col]);
				rig->ref_layers[prev][(uint64_t)row * G5N_HIDDEN + col] = Bf16(total / (float)G5N_HC);
			}
	}

	static const void *WeightTensor(uint32_t kind, Glm5NextPackShape *shape)
	{
		static const void *made[GLM5_NEXT_T_COUNT];
		static Glm5NextPackShape shapes[GLM5_NEXT_T_COUNT];
		Require(Glm5NextPackShapeOf(kind, shape) != 0u, "glm-5.3-flash", "shape", kind, 0u, 0u);
		if ( shape->split == GLM5_NEXT_SPLIT_ROWS )
			shape->rows /= TP_G5N_DEGREE;
		if ( shape->split == GLM5_NEXT_SPLIT_COLUMNS )
			shape->columns /= TP_G5N_DEGREE;
		if ( made[kind] != 0 )
		{
			*shape = shapes[kind];
			return made[kind];
		}
		const uint64_t count = (uint64_t)shape->groups * shape->rows * shape->columns;
		float offset = 0.0f, scale = 1.7f / sqrtf((float)shape->columns);
		if ( shape->rows == 1u && (kind == GLM5_NEXT_T_ATTN_NORM || kind == GLM5_NEXT_T_Q_A_NORM || kind == GLM5_NEXT_T_KV_A_NORM ||
			kind == GLM5_NEXT_T_POST_ATTN_NORM || kind == GLM5_NEXT_T_INDEX_NORM_WEIGHT || kind == GLM5_NEXT_T_KDA_OUT_NORM ||
			kind == GLM5_NEXT_T_FINAL_NORM) )
			offset = 1.0f, scale = 0.1f;
		if ( kind == GLM5_NEXT_T_INDEX_NORM_BIAS || kind == GLM5_NEXT_T_ROUTER_CORRECTION || kind == GLM5_NEXT_T_INDEX_COMPRESS_APE )
			scale = 0.1f;
		if ( kind == GLM5_NEXT_T_KDA_Q_CONV || kind == GLM5_NEXT_T_KDA_K_CONV || kind == GLM5_NEXT_T_KDA_V_CONV || kind == GLM5_NEXT_T_KDA_DECAY_BIAS )
			scale = 0.5f;
		if ( kind == GLM5_NEXT_T_KDA_HEAD_LOG_SCALE )
			offset = -0.5f, scale = 0.3f;
		if ( kind == GLM5_NEXT_T_HC_ATTN_FN || kind == GLM5_NEXT_T_HC_FFN_FN )
			scale = 0.02f;
		if ( kind == GLM5_NEXT_T_HC_ATTN_BASE || kind == GLM5_NEXT_T_HC_FFN_BASE )
			scale = 0.5f;
		if ( kind == GLM5_NEXT_T_HC_ATTN_SCALE || kind == GLM5_NEXT_T_HC_FFN_SCALE )
			offset = 0.5f, scale = 0.25f;
		const void *pointer = shape->payload_type == GLM5_NEXT_PAYLOAD_BF16 ? (const void *)Tensor<uint16_t>(count, scale, offset) :
			(const void *)Tensor<float>(count, scale, offset);
		made[kind] = pointer;
		shapes[kind] = *shape;
		return pointer;
	}

	static const void *ExpertPlane(uint32_t kind, Glm5NextLayerWeights *w)
	{
		Glm5NextPackShape shape;
		uint64_t payload = 0u;
		Require(Glm5NextPackShapeOf(kind, &shape) != 0u, "glm-5.3-flash", "expert shape", kind, 0u, 0u);
		if ( shape.split == GLM5_NEXT_SPLIT_ROWS )
			shape.rows /= TP_G5N_DEGREE;
		if ( shape.split == GLM5_NEXT_SPLIT_COLUMNS )
			shape.columns /= TP_G5N_DEGREE;
		const uint8_t *plane = Fp8Block(shape.groups, shape.rows, shape.columns, &payload);
		for ( uint32_t layer = 0u; layer < G5N_LAYERS; ++layer )
			if ( kind == GLM5_NEXT_T_EXPERT_UP_GATE )
				w[layer].expert_w1_offset = 0u, w[layer].expert_w1_scale_offset = payload;
			else
				w[layer].expert_w2_offset = 0u, w[layer].expert_w2_scale_offset = payload;
		return plane;
	}

	void Open(void)
	{
		Glm5NextPackShape shape;
		SparkStageRunnerKv kv;
		uint32_t *table = 0;
		state = (Glm5NextModelState *)calloc(1u, sizeof(Glm5NextModelState));
		memset(&services, 0, sizeof(services));
		services.context = this;
		services.round = Round;
		services.fail = OnFail;
		services.expert_weights = Experts;
		services.layer_done = OnLayerDone;
		services.row_state = OnRowState;
		Require(cudaStreamCreate(&stream) == cudaSuccess, "glm-5.3-flash", "stream", 0u, 0u, 0u);
		Require(cudaDeviceGetAttribute((int *)&sms, cudaDevAttrMultiProcessorCount, 0) == cudaSuccess, "glm-5.3-flash", "sms", 0u, 0u, 0u);
		state->services = &services;
		state->tp_degree = TP_G5N_DEGREE;
		state->tp_rank = 0u;
		state->max_rows = TP_G5N_ROWS;
		state->slots = TP_SLOTS;
		state->context_capacity = TP_G5N_CONTEXT;
		state->collective_capacity = 2u * TP_G5N_ROWS + 1u;
		state->kda_state_slot_bytes = (uint64_t)Glm5NextRankHeads(G5N_KDA_HEADS, TP_G5N_DEGREE) * G5N_KDA_KEY * G5N_KDA_VALUE * sizeof(float);
		state->kda_window_slot_bytes = (uint64_t)Glm5NextRankHeads(G5N_KDA_HEADS, TP_G5N_DEGREE) * G5N_KDA_KEY * G5N_KDA_CONV * sizeof(uint16_t);
		state->verify_rows = TP_G5N_ROWS;
		for ( uint32_t layer = 0u; layer < G5N_LAYERS; ++layer )
		{
			state->tap_output[layer] = -1;
			state->layers[layer].kda_ordinal = Glm5NextKdaOrdinal(layer);
			state->layers[layer].dsa_ordinal = Glm5NextDsaOrdinal(layer);
			for ( uint32_t kind = GLM5_NEXT_T_ATTN_NORM; kind < GLM5_NEXT_T_MTP_EH_PROJ; ++kind )
				if ( kind != GLM5_NEXT_T_EXPERT_UP_GATE && kind != GLM5_NEXT_T_EXPERT_DOWN && Glm5NextPackKindAllowed(kind, layer) != 0u )
					Require(Glm5NextPlaceTensor(&state->layers[layer], kind, WeightTensor(kind, &shape)) == SPARK_STATUS_OK, "glm-5.3-flash", "place",
						layer, kind, 0u);
		}
		state->final_norm = WeightTensor(GLM5_NEXT_T_FINAL_NORM, &shape);
		expert_w1 = ExpertPlane(GLM5_NEXT_T_EXPERT_UP_GATE, state->layers);
		expert_w2 = ExpertPlane(GLM5_NEXT_T_EXPERT_DOWN, state->layers);
		Require(Glm5NextAllocate(state, 0u) == SPARK_STATUS_OK, "glm-5.3-flash", "alloc", 0u, 0u, 0u);
		latent_bytes = (uint64_t)TP_SLOTS * TP_G5N_PAGES * Glm5NextKv::kPageBytes / TP_G5N_DEGREE;
		index_bytes = (uint64_t)TP_SLOTS * TP_G5N_PAGES * Glm5NextIndexKv::kPageBytes / TP_G5N_DEGREE;
		Require(cudaMalloc((void **)&latent_pool, latent_bytes * G5N_DSA_LAYERS) == cudaSuccess &&
			cudaMemset(latent_pool, 0, latent_bytes * G5N_DSA_LAYERS) == cudaSuccess, "glm-5.3-flash", "latent pool", 0u, 0u, 0u);
		Require(cudaMalloc((void **)&index_pool, index_bytes * G5N_DSA_LAYERS) == cudaSuccess &&
			cudaMemset(index_pool, 0, index_bytes * G5N_DSA_LAYERS) == cudaSuccess, "glm-5.3-flash", "index pool", 0u, 0u, 0u);
		std::vector<uint32_t> pages(TP_SLOTS * TP_G5N_PAGES);
		for ( uint32_t page = 0u; page < pages.size(); ++page )
			pages[page] = page;
		Require(cudaMalloc((void **)&table, pages.size() * sizeof(uint32_t)) == cudaSuccess &&
			cudaMemcpy(table, pages.data(), pages.size() * sizeof(uint32_t), cudaMemcpyHostToDevice) == cudaSuccess, "glm-5.3-flash", "pages", 0u, 0u, 0u);
		memset(&kv, 0, sizeof(kv));
		kv.pool = latent_pool;
		kv.layer_stride_bytes = latent_bytes;
		kv.layer_page_bytes = Glm5NextKv::kPageBytes;
		kv.layer_count = G5N_DSA_LAYERS;
		kv.page_table = table;
		kv.page_table_stride = TP_G5N_PAGES;
		kv.pool_page_count = TP_SLOTS * TP_G5N_PAGES;
		kv.sequence_count = TP_SLOTS;
		kv.context_shard = SparkGlm5NextKvShardLatent(0u, TP_G5N_DEGREE);
		kv.second_pool = index_pool;
		kv.second_layer_stride_bytes = index_bytes;
		kv.second_layer_page_bytes = Glm5NextIndexKv::kPageBytes;
		kv.second_layer_count = G5N_DSA_LAYERS;
		Require(Stage::AttachKv(state, &kv) == SPARK_STATUS_OK, "glm-5.3-flash", "attach kv", 0u, 0u, 0u);
		Require(cudaMalloc((void **)&meta, 64u * sizeof(uint32_t)) == cudaSuccess, "glm-5.3-flash", "meta", 0u, 0u, 0u);
		Require(cudaMalloc((void **)&hidden, (uint64_t)TP_G5N_ROWS * G5N_HIDDEN * sizeof(uint16_t)) == cudaSuccess, "glm-5.3-flash", "hidden", 0u, 0u, 0u);
	}

	void RunStep(const std::vector<uint16_t> &input, uint32_t rows, uint32_t dispatch_row_offset, uint32_t record_ref, std::vector<uint16_t> *out)
	{
		uint32_t host_meta[64] = {};
		const uint32_t seq_off = 0u, pos_off = 16u, ctx_off = 32u, rec_off = 36u, beg_off = 40u, idx_off = 44u;
		for ( uint32_t row = 0u; row < rows; ++row )
		{
			host_meta[seq_off + row] = 0u;
			host_meta[pos_off + row] = row;
			host_meta[idx_off + row] = row;
		}
		host_meta[ctx_off] = rows;
		host_meta[rec_off] = 0u;
		host_meta[beg_off] = 0u;
		host_meta[beg_off + 1u] = rows;
		SparkStageRunnerStep step;
		memset(&step, 0, sizeof(step));
		step.hidden_bf16 = hidden;
		step.sequence_of_row = meta + seq_off;
		step.positions = meta + pos_off;
		step.context_length = meta + ctx_off;
		step.recurrent_index = meta + rec_off;
		step.sequence_row_begin = meta + beg_off;
		step.sequence_row_indices = meta + idx_off;
		step.rows = rows;
		step.sequences = 1u;
		step.commit = 1u;
		step.dispatch_row_offset = dispatch_row_offset;
		step.context = TP_G5N_CONTEXT;
		step.multiprocessors = sms;
		step.route_fault = VfRouteFault();
		active_rows = rows;
		capture_ref = record_ref;
		Require(Stage::ResetSlot(state, 0u, stream) == SPARK_STATUS_OK, "glm-5.3-flash", "reset slot", 0u, 0u, 0u);
		Require(cudaMemsetAsync(latent_pool, 0, latent_bytes * G5N_DSA_LAYERS, stream) == cudaSuccess &&
			cudaMemsetAsync(index_pool, 0, index_bytes * G5N_DSA_LAYERS, stream) == cudaSuccess, "glm-5.3-flash", "kv reset", 0u, 0u, 0u);
		Require(cudaMemcpyAsync(meta, host_meta, sizeof(host_meta), cudaMemcpyHostToDevice, stream) == cudaSuccess &&
			cudaMemcpyAsync(hidden, input.data(), input.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, stream) == cudaSuccess, "glm-5.3-flash", "upload", 0u, 0u, 0u);
		Require(Stage::Step(state, &step, stream) == SPARK_STATUS_OK, "glm-5.3-flash", "step", rows, 0u, 0u);
		Require(cudaStreamSynchronize(stream) == cudaSuccess, "glm-5.3-flash", "sync", rows, 0u, 0u);
		capture_ref = 0u;
		if ( record_ref != 0u )
		{
			ref_layers[G5N_LAYERS - 1u].resize((uint64_t)rows * G5N_HIDDEN);
			Require(cudaMemcpy(ref_layers[G5N_LAYERS - 1u].data(), state->streams_mean, (uint64_t)rows * G5N_HIDDEN * sizeof(uint16_t),
				cudaMemcpyDeviceToHost) == cudaSuccess, "glm-5.3-flash", "streams_mean copy", 0u, 0u, 0u);
		}
		if ( out != 0 )
		{
			out->resize(input.size());
			Require(cudaMemcpy(out->data(), hidden, input.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) == cudaSuccess, "glm-5.3-flash", "out copy", 0u, 0u, 0u);
		}
	}
};

static void TestGlm5Next(void)
{
	Glm5NextTapRig rig;
	rig.Open();
	const uint32_t rows = 3u, offset = 2u, slab_rows = rows + offset + 2u, entries = 8u;
	const uint64_t stride = (uint64_t)entries * G5N_HIDDEN;
	uint16_t *tap_slab = 0;
	Require(cudaMalloc((void **)&tap_slab, (size_t)slab_rows * stride * sizeof(uint16_t)) == cudaSuccess, "glm-5.3-flash", "slab alloc", 0u, 0u, 0u);
	CheckTapPlanContract<Glm5NextTapRig::Stage>("glm-5.3-flash", rig.state, G5N_LAYERS, tap_slab, stride);

	std::vector<uint16_t> input((uint64_t)rows * G5N_HIDDEN), out_tapped, out_plain;
	for ( uint64_t index = 0u; index < input.size(); ++index )
		input[index] = Bf16(Uniform(505u, (uint32_t)index));
	const uint16_t tapped_layers[7] = { 0u, 5u, 14u, 24u, 33u, 42u, G5N_LAYERS - 1u };
	SparkStageRunnerTapPlan plan;
	memset(&plan, 0, sizeof(plan));
	plan.entry_count = entries;
	for ( uint32_t index = 0u; index < 7u; ++index )
		plan.entries[index] = { SPARK_STAGE_RUNNER_TAP_FORM_OUTPUT, tapped_layers[index] };
	plan.entries[7] = { SPARK_STAGE_RUNNER_TAP_FORM_FINAL, 0u };
	plan.capture = 1u;
	plan.row_capacity = slab_rows;
	plan.slab = tap_slab;
	plan.row_stride_elements = stride;
	FillSentinel(tap_slab, (uint64_t)slab_rows * stride);
	Require(Glm5NextTapRig::Stage::TapPlan(rig.state, &plan) == SPARK_STATUS_OK, "glm-5.3-flash", "tap plan", 0u, 0u, 0u);
	rig.RunStep(input, rows, offset, 1u, &out_tapped);

	std::vector<uint16_t> host_slab((uint64_t)slab_rows * stride);
	Require(cudaMemcpy(host_slab.data(), tap_slab, host_slab.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) == cudaSuccess, "glm-5.3-flash", "slab read", 0u, 0u, 0u);
	CheckSentinelSlice("glm-5.3-flash", "prefix untouched", host_slab, 0u, (uint64_t)offset * stride);
	for ( uint32_t row = 0u; row < rows; ++row )
	{
		const uint64_t base = (uint64_t)(offset + row) * stride;
		for ( uint32_t entry = 0u; entry < 7u; ++entry )
			for ( uint32_t col = 0u; col < G5N_HIDDEN; ++col )
				if ( host_slab[base + (uint64_t)entry * G5N_HIDDEN + col] != rig.ref_layers[tapped_layers[entry]][(uint64_t)row * G5N_HIDDEN + col] )
					{ Require(0, "glm-5.3-flash", "layer tap mismatch", row, tapped_layers[entry], col); break; }
		CheckSentinelSlice("glm-5.3-flash", "final entry untouched", host_slab, base + 7u * G5N_HIDDEN, G5N_HIDDEN);
	}
	plan.capture = 0u;
	FillSentinel(tap_slab, (uint64_t)slab_rows * stride);
	Require(Glm5NextTapRig::Stage::TapPlan(rig.state, &plan) == SPARK_STATUS_OK, "glm-5.3-flash", "disable capture", 0u, 0u, 0u);
	rig.RunStep(input, rows, offset, 0u, &out_plain);
	Require(out_tapped == out_plain, "glm-5.3-flash", "taps preserve output", 0u, 0u, 0u);
	Require(cudaMemcpy(host_slab.data(), tap_slab, host_slab.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) == cudaSuccess, "glm-5.3-flash", "slab read 2", 0u, 0u, 0u);
	CheckSentinelSlice("glm-5.3-flash", "capture=0 leaves slab untouched", host_slab, 0u, (uint64_t)slab_rows * stride);
	cudaFree(tap_slab);
	printf("PASS tap plan glm-5.3-flash\n");
}

int main(void)
{
	Require(SparkGlm53FullTp16StageModel()->tap_plan != 0, "interfaces", "glm53_full tap_plan", 0u, 0u, 0u);
	Require(SparkAxk2Tp16StageModel()->tap_plan != 0, "interfaces", "axk2 tap_plan", 0u, 0u, 0u);
	Require(SparkGlm53FlashTp16StageModel()->tap_plan != 0, "interfaces", "glm53_flash tap_plan", 0u, 0u, 0u);
	Require(SparkKexaone2Tp16StageModel()->tap_plan != 0, "interfaces", "kexaone2 tap_plan", 0u, 0u, 0u);
	TestGlmFamily<TapGlm53>();
	TestGlmFamily<TapAxk2>();
	TestGemmaFamily<TapKexaone>();
	TestGemmaFamily<TapGemma31b>();
	TestGlm5Next();
	if ( vf_failures != 0 )
	{
		printf("FAIL four-family tap plan (%d failures)\n", vf_failures);
		return 1;
	}
	printf("PASS four-family tap plan\n");
	return 0;
}
