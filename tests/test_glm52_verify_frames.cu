#define GLM53_FULL_CONTRACT_SHA256 "verify-frames"
#define GLM53_FULL_MODEL_REVISION "verify-frames"
#define AXK2_CONTRACT_SHA256 "verify-frames"
#define AXK2_MODEL_REVISION "verify-frames"
#include "modules/glm53_full_tp16_resident_decode_stage/source/spark_glm53_full_tp16_stage_model.cu"
#include "modules/axk2_tp16_resident_decode_stage/source/spark_axk2_tp16_stage_model.cu"
#include "sparkpipe/spark_model_serving_adapter.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "tests/cuda/verify_frames_common.cuh"

#define VF_MAX_FRAME SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE
#define VF_SLOTS 2u
#define VF_PAGES 8u
#define VF_PREFIX 160u
#define VF_OTHER_PREFIX 100u

struct SmallGlm53Full : Glm53FullGeometry
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
	static const char *Tag(void) { return "glm-5.3-full"; }
};

struct SmallAxk2 : Axk2Geometry
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

static uint8_t E4m3(float value)
{
	const float magnitude = fabsf(value);
	uint8_t best = 0u;
	float error = INFINITY;
	for ( uint32_t code = 0u; code < 0x7fu; ++code )
	{
		const uint32_t exponent = code >> 3u, mantissa = code & 7u;
		const float decoded = exponent == 0u ? ldexpf((float)mantissa / 8.0f, -6) : ldexpf(1.0f + (float)mantissa / 8.0f, (int)exponent - 7);
		if ( fabsf(decoded - magnitude) < error )
		{
			error = fabsf(decoded - magnitude);
			best = (uint8_t)code;
		}
	}
	return (uint8_t)(best | (value < 0.0f ? 0x80u : 0u));
}

static uint8_t *Device(const void *host, uint64_t bytes)
{
	void *device = 0;
	if ( cudaMalloc(&device, bytes) != cudaSuccess || cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess )
	{
		printf("FAIL weight allocation of %llu bytes\n", (unsigned long long)bytes);
		exit(1);
	}
	return (uint8_t *)device;
}

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

static GlmLinearWeight Fp8(uint64_t rows, uint64_t columns)
{
	uint64_t payload = 0u;
	GlmLinearWeight weight;
	weight.weight = Fp8Block(1u, rows, columns, &payload);
	weight.scale = (const float *)(weight.weight + payload);
	return weight;
}

template<class G>
struct VfRig
{
	using M = GlmModel<G>;
	using S = GlmShape<G>;
	using State = typename M::State;
	static constexpr uint64_t kPool = (uint64_t)VF_SLOTS * VF_PAGES;
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
	uint32_t host_context[VF_SLOTS];
	const uint8_t *experts_up[S::kLayers], *experts_down[S::kLayers];

	static SparkStatus Weights(void *context_pointer, uint32_t layer, const uint32_t *group_row_offset, uint32_t routed_rows, const void **w1, const void **w2)
	{
		const VfRig *rig = (const VfRig *)context_pointer;
		(void)group_row_offset;
		(void)routed_rows;
		*w1 = rig->experts_up[layer];
		*w2 = rig->experts_down[layer];
		return SPARK_STATUS_OK;
	}

	static SparkStatus Round(void *context_pointer, void *stream_pointer, uint32_t operation, uint32_t stage, uint32_t rows, uint32_t row_elements,
		const void *local, void *full)
	{
		return VfRound(&((VfRig *)context_pointer)->bus, stream_pointer, operation, stage, rows, row_elements, local, full);
	}

	void Layer(uint32_t index)
	{
		GlmLayer *layer = &state->layers[index];
		const uint32_t heads = state->heads;
		layer->attn_norm = Tensor<uint16_t>(G::kHidden, 0.2f, 1.0f);
		layer->q_a = Fp8(G::kQueryA, G::kHidden);
		layer->q_a_norm = Tensor<uint16_t>(G::kQueryA, 0.2f, 1.0f);
		layer->q_b = Fp8((uint64_t)heads * S::kQueryHead, S::kQueryInput);
		layer->kv_a = Fp8(S::kSlot, G::kHidden);
		layer->kv_a_norm = Tensor<uint16_t>(G::kLatent, 0.2f, 1.0f);
		layer->kv_b_key = Tensor<uint16_t>((uint64_t)heads * G::kLatent * G::kNope, 1.7f / sqrtf((float)G::kNope), 0.0f);
		layer->kv_b_value = Tensor<uint16_t>((uint64_t)heads * G::kValue * G::kLatent, 1.7f / sqrtf((float)G::kLatent), 0.0f);
		layer->o_proj = Fp8(G::kHidden, (uint64_t)heads * G::kValue);
		layer->post_norm = Tensor<uint16_t>(G::kHidden, 0.2f, 1.0f);
		if ( G::FullIndexer(index) != 0u )
		{
			layer->index_q = Fp8(S::kIndexQuery, G::kQueryA);
			layer->index_k = Fp8(G::kIndexDim, G::kHidden);
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
			layer->dense_gate_up = Fp8(2u * G::kDenseInter, G::kHidden);
			layer->dense_down = Fp8(G::kHidden, G::kDenseInter);
			return;
		}
		layer->router = Matrix(G::kExperts, G::kHidden);
		layer->router_bias = Tensor<float>(G::kExperts, 0.05f, 0.0f);
		layer->shared_gate_up = Fp8(2u * G::kExpertInter, G::kHidden);
		layer->shared_down = Fp8(G::kHidden, G::kExpertInter);
		experts_up[index] = Fp8Block(G::kExperts, 2u * G::kExpertInter, G::kHidden, &layer->up_gate_scale);
		experts_down[index] = Fp8Block(G::kExperts, G::kHidden, G::kExpertInter, &layer->down_scale);
		state->w1_offset[index] = 0u;
		state->w2_offset[index] = 0u;
	}

	int Open(void)
	{
		std::vector<uint32_t> table(VF_SLOTS * VF_PAGES);
		SparkStageRunnerKv attach;
		memset(this, 0, sizeof(*this));
		bus.hidden = G::kHidden;
		services.context = this;
		services.round = Round;
		services.expert_weights = Weights;
		state = (State *)calloc(1u, sizeof(State));
		state->services = &services;
		state->tp_degree = 1u;
		state->heads = G::kHeads;
		state->max_rows = VF_MAX_FRAME;
		state->scatter_rows = VF_MAX_FRAME;
		state->slots = VF_SLOTS;
		state->max_positions = VF_PAGES * GLM_PAGE_SLOTS;
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
			cudaMalloc((void **)&positions, VF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&sequence_of_row, VF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&context, VF_SLOTS * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&hidden, (uint64_t)VF_MAX_FRAME * G::kHidden * sizeof(uint16_t)) != cudaSuccess )
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
		attach.page_table_stride = VF_PAGES;
		attach.pool_page_count = (uint32_t)kPool;
		attach.sequence_count = VF_SLOTS;
		attach.context_shard = state->shard;
		attach.second_pool = kv + kLatentRegion;
		attach.second_layer_stride_bytes = kPool * GlmIndexKv<S>::kPageBytes;
		attach.second_layer_page_bytes = GlmIndexKv<S>::kPageBytes;
		attach.second_layer_count = G::kIndexLayers;
		return M::AttachKv(state, &attach) == SPARK_STATUS_OK;
	}

	void Input(const VfRow &row, uint16_t *target) const
	{
		const uint32_t seed = Mix(row.sequence * 7919u + row.variant * 104729u) ^ row.position;
		for ( uint32_t index = 0u; index < G::kHidden; ++index )
			target[index] = Bf16(Uniform(seed, index));
	}

	int Run(const VfRow *rows, uint32_t count, uint32_t commit, std::vector<uint16_t> *output)
	{
		std::vector<uint16_t> input((uint64_t)count * G::kHidden);
		std::vector<uint32_t> position(count), sequence(count);
		uint32_t seen[VF_SLOTS] = { 0u, 0u }, sequences = 0u;
		SparkStageRunnerStep step;
		for ( uint32_t row = 0u; row < count; ++row )
		{
			Input(rows[row], input.data() + (uint64_t)row * G::kHidden);
			position[row] = rows[row].position;
			sequence[row] = rows[row].sequence;
			sequences += seen[rows[row].sequence] == 0u ? 1u : 0u;
			seen[rows[row].sequence] = 1u;
			host_context[rows[row].sequence] = rows[row].position + 1u;
		}
		memset(&step, 0, sizeof(step));
		step.hidden_bf16 = hidden;
		step.positions = positions;
		step.context_length = context;
		step.sequence_of_row = sequence_of_row;
		step.rows = count;
		step.sequences = sequences;
		step.commit = commit;
		step.context = state->max_positions;
		step.multiprocessors = multiprocessors;
		step.host_positions = position.data();
		step.host_sequence_of_row = sequence.data();
		step.graph_key = M::GraphKey(state, &step);
		if ( cudaMemcpyAsync(hidden, input.data(), input.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			cudaMemcpyAsync(positions, position.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			cudaMemcpyAsync(sequence_of_row, sequence.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			cudaMemcpyAsync(context, host_context, sizeof(host_context), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			M::Step(state, &step, stream) != SPARK_STATUS_OK || cudaStreamSynchronize(stream) != cudaSuccess )
			return 0;
		if ( output != 0 )
		{
			const uint64_t offset = output->size();
			output->resize(offset + input.size());
			if ( cudaMemcpy(output->data() + offset, hidden, input.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess )
				return 0;
		}
		return 1;
	}

	int Decode(uint32_t sequence, uint32_t position, uint32_t variant, std::vector<uint16_t> *output)
	{
		const VfRow row = { sequence, position, variant };
		return Run(&row, 1u, 1u, output);
	}

	std::vector<uint8_t> Save(void) const
	{
		std::vector<uint8_t> copy(kLatentRegion + kIndexRegion);
		if ( cudaMemcpy(copy.data(), kv, copy.size(), cudaMemcpyDeviceToHost) != cudaSuccess )
			copy.clear();
		return copy;
	}

	int Load(const std::vector<uint8_t> &copy)
	{
		return copy.size() == kLatentRegion + kIndexRegion && cudaMemcpy(kv, copy.data(), copy.size(), cudaMemcpyHostToDevice) == cudaSuccess;
	}

	static uint32_t KvMatches(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b, uint32_t limit)
	{
		if ( a.size() != b.size() )
			return 0u;
		for ( uint32_t sequence = 0u; sequence < VF_SLOTS; ++sequence )
			for ( uint32_t position = 0u; position < VF_PAGES * GLM_PAGE_SLOTS; ++position )
			{
				const uint64_t page = (uint64_t)sequence * VF_PAGES + position / GLM_PAGE_SLOTS, slot = position % GLM_PAGE_SLOTS;
				if ( sequence == 0u && position >= limit )
					continue;
				for ( uint32_t layer = 0u; layer < S::kLayers; ++layer )
				{
					const uint64_t offset = (uint64_t)layer * GLM_PAGE_SLOTS * S::kSlotBytes + page * GlmKv<S>::kPageBytes + slot * S::kSlotBytes;
					if ( memcmp(a.data() + offset, b.data() + offset, S::kSlotBytes) != 0 )
						return 0u;
				}
				for ( uint32_t layer = 0u; layer < G::kIndexLayers; ++layer )
				{
					const uint64_t offset = kLatentRegion + ((uint64_t)layer * kPool + page) * GlmIndexKv<S>::kPageBytes + slot * GlmIndexKv<S>::kSlotBytes;
					if ( memcmp(a.data() + offset, b.data() + offset, GlmIndexKv<S>::kSlotBytes) != 0 )
						return 0u;
				}
			}
		return 1u;
	}

	uint32_t KvClean(void) const
	{
		return M::Check(state) == SPARK_STATUS_OK ? 1u : 0u;
	}
};

template<class G>
static void Check(void)
{
	static const uint32_t frames[] = { 1u, 2u, 7u, 16u, 17u, 40u, VF_MAX_FRAME };
	VfRig<G> rig;
	std::vector<VfRow> rows;
	std::vector<uint8_t> prefix;
	const char *model = G::Tag();
	uint32_t checked = 0u;
	if ( rig.Open() == 0 )
	{
		Require(0, model, "the synthetic stage opens, binds and attaches its KV pools", 0u, 0u, 0u);
		return;
	}
	for ( uint32_t sequence = 0u; sequence < VF_SLOTS; ++sequence )
	{
		rows.clear();
		for ( uint32_t position = 0u; position < (sequence == 0u ? VF_PREFIX : VF_OTHER_PREFIX); ++position )
			rows.push_back({ sequence, position, 0u });
		Require(rig.Run(rows.data(), (uint32_t)rows.size(), 0u, 0) != 0, model, "the prefix fill runs", sequence, 0u, 0u);
	}
	prefix = rig.Save();
	for ( uint32_t frame : frames )
	{
		std::vector<uint16_t> decoded, verified;
		std::vector<std::vector<uint8_t> > before;
		std::vector<uint32_t> kept;
		std::vector<uint8_t> after_decode, after_verify;
		Kept(frame, &kept);
		rig.Load(prefix);
		for ( uint32_t row = 0u; row < frame; ++row )
		{
			for ( uint32_t k : kept )
				if ( k == row )
					before.push_back(rig.Save());
			Require(rig.Decode(0u, VF_PREFIX + row, 0u, &decoded) != 0, model, "one-row decode runs", frame, row, 0u);
		}
		before.push_back(rig.Save());
		after_decode = before.back();
		rig.Load(prefix);
		rows.clear();
		for ( uint32_t row = 0u; row < frame; ++row )
			rows.push_back({ 0u, VF_PREFIX + row, 0u });
		Require(rig.Run(rows.data(), frame, 0u, &verified) != 0, model, "the verify frame runs", frame, 0u, 0u);
		after_verify = rig.Save();
		Require(Finite(verified) != 0u, model, "verify rows are finite and not constant", frame, 0u, 0u);
		Require(FirstDifference(verified, decoded, G::kHidden) == UINT32_MAX, model,
			"every verify row equals its one-row decode bit for bit (frame rows, first differing row)", frame, FirstDifference(verified, decoded, G::kHidden), 0u);
		Require(rig.KvMatches(after_verify, after_decode, UINT32_MAX) != 0u, model,
			"the KV the verify frame leaves equals the KV of the one-row decodes at every position", frame, 0u, 0u);
		for ( uint32_t index = 0u; index < kept.size(); ++index )
		{
			const uint32_t k = kept[index], limit = VF_PREFIX + k + 2u;
			std::vector<uint16_t> speculated, plain;
			rig.Load(after_verify);
			Require(rig.Decode(0u, VF_PREFIX + k, 1u, &speculated) != 0, model, "the decode after a rejected tail runs", frame, k, 0u);
			Require(rig.Decode(0u, VF_PREFIX + k + 1u, 2u, &speculated) != 0, model, "the second decode after a rejected tail runs", frame, k, 0u);
			const std::vector<uint8_t> speculated_kv = rig.Save();
			rig.Load(before[index]);
			Require(rig.Decode(0u, VF_PREFIX + k, 1u, &plain) != 0 && rig.Decode(0u, VF_PREFIX + k + 1u, 2u, &plain) != 0, model,
				"the decodes of the run that never saw the draft run", frame, k, 0u);
			Require(FirstDifference(speculated, plain, G::kHidden) == UINT32_MAX, model,
				"after keeping k rows the next two decodes equal a run that never saw the draft (frame rows, k, first differing step)", frame, k,
				FirstDifference(speculated, plain, G::kHidden));
			Require(rig.KvMatches(speculated_kv, rig.Save(), limit) != 0u, model,
				"after keeping k rows the KV up to the next decode equals a run that never saw the draft (frame rows, k)", frame, k, 0u);
			checked++;
		}
	}
	{
		std::vector<uint16_t> decoded, verified;
		std::vector<uint8_t> after_decode;
		const uint32_t first = 7u, second = 16u;
		rig.Load(prefix);
		rows.clear();
		for ( uint32_t row = 0u; row < first; ++row )
		{
			rows.push_back({ 0u, VF_PREFIX + row, 0u });
			Require(rig.Decode(0u, VF_PREFIX + row, 0u, &decoded) != 0, model, "one-row decode of the first sequence runs", row, 0u, 0u);
		}
		for ( uint32_t row = 0u; row < second; ++row )
		{
			rows.push_back({ 1u, VF_OTHER_PREFIX + row, 0u });
			Require(rig.Decode(1u, VF_OTHER_PREFIX + row, 0u, &decoded) != 0, model, "one-row decode of the second sequence runs", row, 0u, 0u);
		}
		after_decode = rig.Save();
		rig.Load(prefix);
		Require(rig.Run(rows.data(), first + second, 0u, &verified) != 0, model, "a verify frame over two sequences runs", first, second, 0u);
		Require(FirstDifference(verified, decoded, G::kHidden) == UINT32_MAX, model,
			"every row of a two-sequence verify frame equals its one-row decode (first differing row)", FirstDifference(verified, decoded, G::kHidden), 0u, 0u);
		Require(rig.KvMatches(rig.Save(), after_decode, UINT32_MAX) != 0u, model, "a two-sequence verify frame leaves the decodes' KV", 0u, 0u, 0u);
	}
	Require(rig.KvClean() != 0u, model, "no step reported a KV access error", 0u, 0u, 0u);
	printf("%s %s: frames of 1-%u rows, %u rejection points\n", vf_failures == 0 ? "ok" : "checked", model, VF_MAX_FRAME, checked);
}

int main(void)
{
	Check<SmallGlm53Full>();
	Check<SmallAxk2>();
	if ( vf_failures != 0 )
	{
		printf("FAIL glm52 verify frames: %d failures\n", vf_failures);
		return 1;
	}
	printf("PASS glm52 verify frames: every row of every verify frame equals one-row decode bit for bit, and a rejected tail leaves no trace\n");
	return 0;
}
