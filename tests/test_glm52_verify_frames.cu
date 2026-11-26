#define GLM53_FULL_CONTRACT_SHA256 "verify-frames"
#define GLM53_FULL_MODEL_REVISION "verify-frames"
#define AXK2_CONTRACT_SHA256 "verify-frames"
#define AXK2_MODEL_REVISION "verify-frames"
#include "modules/glm53_full_tp16_resident_decode_stage/source/spark_glm53_full_tp16_stage_model.cu"
#include "modules/axk2_tp16_resident_decode_stage/source/spark_axk2_tp16_stage_model.cu"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_speculation_tree_frame.h"

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

struct SmallGlm53FullInter128 : SmallGlm53Full
{
	static constexpr uint32_t kExpertInter = 128u;
	static const char *Tag(void) { return "glm-5.3-full-inter128"; }
};

struct SmallAxk2Inter128 : SmallAxk2
{
	static constexpr uint32_t kExpertInter = 128u;
	static const char *Tag(void) { return "a.x-k2-inter128"; }
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

static GlmLinearWeight Fp8(uint64_t rows, uint64_t columns)
{
	uint64_t payload = 0u;
	GlmLinearWeight weight;
	weight.weight = Fp8Block(1u, rows, columns, &payload);
	weight.scale = (const float *)(weight.weight + payload);
	return weight;
}

static GlmLinearWeight Bf16Weight(uint64_t rows, uint64_t columns)
{
	GlmLinearWeight weight;
	weight.weight = (const uint8_t *)Matrix(rows, columns);
	weight.scale = 0;
	return weight;
}

template<class G>
static GlmLinearWeight Linear(uint64_t rows, uint64_t columns)
{
	return GlmLinearScaled<G> ? Fp8(rows, columns) : Bf16Weight(rows, columns);
}

template<class G>
static const uint8_t *ExpertBlock(uint64_t groups, uint64_t rows, uint64_t columns, uint64_t *scale_offset)
{
	if ( GlmExpertScaled<G> )
		return Fp8Block(groups, rows, columns, scale_offset);
	*scale_offset = 0u;
	return (const uint8_t *)Matrix(groups * rows, columns);
}

template<class G, uint32_t MAX_ROWS = VF_MAX_FRAME, uint32_t PAGES = VF_PAGES, uint32_t SLOTS = VF_SLOTS>
struct VfRig
{
	using M = GlmModel<G>;
	using S = GlmShape<G>;
	using State = typename M::State;
	static constexpr uint64_t kPool = (uint64_t)SLOTS * PAGES;
	static constexpr uint64_t kLatentRegion = kPool * GlmKv<S>::kPageBytes;
	static constexpr uint64_t kIndexRegion = (uint64_t)G::kIndexLayers * kPool * GlmIndexKv<S>::kPageBytes;
	State *state;
	VfBus bus;
	SparkStageRunnerServices services;
	cudaStream_t stream;
	uint32_t multiprocessors;
	uint8_t *kv;
	uint32_t *page_table, *positions, *context, *sequence_of_row;
	uint32_t *tree_slots, *forks, *offsets, *row_indices, *kept;
	uint16_t *hidden;
	uint32_t host_context[SLOTS];
	SparkStageRunnerStep last;
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
		layer->q_a = Linear<G>(G::kQueryA, G::kHidden);
		layer->q_a_norm = Tensor<uint16_t>(G::kQueryA, 0.2f, 1.0f);
		layer->q_b = Linear<G>((uint64_t)heads * S::kQueryHead, S::kQueryInput);
		layer->kv_a = Linear<G>(S::kSlot, G::kHidden);
		layer->kv_a_norm = Tensor<uint16_t>(G::kLatent, 0.2f, 1.0f);
		layer->kv_b_key = Tensor<uint16_t>((uint64_t)heads * G::kLatent * G::kNope, 1.7f / sqrtf((float)G::kNope), 0.0f);
		layer->kv_b_value = Tensor<uint16_t>((uint64_t)heads * G::kValue * G::kLatent, 1.7f / sqrtf((float)G::kLatent), 0.0f);
		layer->o_proj = Linear<G>(G::kHidden, (uint64_t)heads * G::kValue);
		layer->post_norm = Tensor<uint16_t>(G::kHidden, 0.2f, 1.0f);
		if ( G::FullIndexer(index) != 0u )
		{
			layer->index_q = Linear<G>(S::kIndexQuery, G::kQueryA);
			layer->index_k = Linear<G>(G::kIndexDim, G::kHidden);
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
			layer->dense_gate_up = Linear<G>(2u * G::kDenseInter, G::kHidden);
			layer->dense_down = Linear<G>(G::kHidden, G::kDenseInter);
			return;
		}
		layer->router = Matrix(G::kExperts, G::kHidden);
		layer->router_bias = Tensor<float>(G::kExperts, 0.05f, 0.0f);
		layer->shared_gate_up = Linear<G>(2u * G::kExpertInter, G::kHidden);
		layer->shared_down = Linear<G>(G::kHidden, G::kExpertInter);
		experts_up[index] = ExpertBlock<G>(G::kExperts, 2u * G::kExpertInter, G::kHidden, &layer->up_gate_scale);
		experts_down[index] = ExpertBlock<G>(G::kExperts, G::kHidden, G::kExpertInter, &layer->down_scale);
		state->w1_offset[index] = 0u;
		state->w2_offset[index] = 0u;
	}

	int Open(void)
	{
		std::vector<uint32_t> table(SLOTS * PAGES);
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
		state->max_rows = MAX_ROWS;
		state->scatter_rows = MAX_ROWS < GLM_SCATTER_ROWS ? MAX_ROWS : GLM_SCATTER_ROWS;
		state->slots = SLOTS;
		state->max_positions = PAGES * GLM_PAGE_SLOTS;
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
			cudaMalloc((void **)&positions, MAX_ROWS * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&sequence_of_row, MAX_ROWS * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&tree_slots, MAX_ROWS * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&forks, MAX_ROWS * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&offsets, MAX_ROWS * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&row_indices, MAX_ROWS * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&kept, SLOTS * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&context, SLOTS * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&hidden, (uint64_t)MAX_ROWS * G::kHidden * sizeof(uint16_t)) != cudaSuccess )
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
		attach.page_table_stride = PAGES;
		attach.pool_page_count = (uint32_t)kPool;
		attach.sequence_count = SLOTS;
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

	int Run(const VfRow *rows, uint32_t count, uint32_t commit, std::vector<uint16_t> *output, uint32_t last_rows_only = 0u)
	{
		std::vector<uint16_t> input((uint64_t)count * G::kHidden);
		std::vector<uint32_t> position(count), sequence(count);
		uint32_t seen[SLOTS] = {}, sequences = 0u;
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
		step.last_rows_only = last_rows_only;
		step.context = state->max_positions;
		step.multiprocessors = multiprocessors;
		step.route_fault = VfRouteFault();
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

	int UploadTree(const VfRow *rows, const uint32_t *parent_rows, uint32_t count, uint32_t slot_grain, SparkStageRunnerStep *step_out)
	{
		std::vector<uint16_t> input((uint64_t)count * G::kHidden);
		std::vector<uint32_t> position(count), sequence(count, rows[0].sequence), slot_table(count), fork_table(count), offset_table(count), tok(count);
		SparkSpeculationTreeLayout layout;
		SparkStageRunnerStep step;
		for ( uint32_t row = 0u; row < count; ++row )
			tok[row] = rows[row].variant;
		if ( count > VF_MAX_FRAME || SparkSpeculationTreeLayoutBuild(parent_rows, tok.data(), count, rows[0].position, slot_grain, &layout) != SPARK_STATUS_OK )
			return 0;
		for ( uint32_t row = 0u; row < count; ++row )
		{
			Input(rows[row], input.data() + (uint64_t)row * G::kHidden);
			position[row] = layout.anchor_position + layout.row[row].depth;
			if ( position[row] != rows[row].position )
				return 0;
			slot_table[row] = layout.row[row].slot;
			fork_table[row] = layout.row[row].fork_position;
			offset_table[row] = layout.row[row].slot_offset;
		}
		host_context[rows[0].sequence] = layout.context;
		memset(&step, 0, sizeof(step));
		step.hidden_bf16 = hidden;
		step.positions = positions;
		step.context_length = context;
		step.sequence_of_row = sequence_of_row;
		step.sequence_row_indices = row_indices;
		step.rows = count;
		step.sequences = 1u;
		step.commit = 0u;
		step.context = state->max_positions;
		step.multiprocessors = multiprocessors;
		step.route_fault = VfRouteFault();
		step.tree = 1u;
		step.slots = tree_slots;
		step.fork_positions = forks;
		step.slot_offsets = offsets;
		step.graph_key = M::GraphKey(state, &step);
		last = step;
		*step_out = step;
		return cudaMemcpyAsync(hidden, input.data(), input.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, stream) == cudaSuccess &&
			cudaMemcpyAsync(positions, position.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) == cudaSuccess &&
			cudaMemcpyAsync(sequence_of_row, sequence.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) == cudaSuccess &&
			cudaMemcpyAsync(context, host_context, sizeof(host_context), cudaMemcpyHostToDevice, stream) == cudaSuccess &&
			cudaMemcpyAsync(tree_slots, slot_table.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) == cudaSuccess &&
			cudaMemcpyAsync(forks, fork_table.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) == cudaSuccess &&
			cudaMemcpyAsync(offsets, offset_table.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) == cudaSuccess ? 1 : 0;
	}

	int RunTree(const VfRow *rows, const uint32_t *parent_rows, uint32_t count, uint32_t slot_grain, std::vector<uint16_t> *output)
	{
		SparkStageRunnerStep step;
		if ( UploadTree(rows, parent_rows, count, slot_grain, &step) == 0 || M::Step(state, &step, stream) != SPARK_STATUS_OK || cudaStreamSynchronize(stream) != cudaSuccess )
			return 0;
		if ( output != 0 )
		{
			const uint64_t offset = output->size(), elems = (uint64_t)count * G::kHidden;
			output->resize(offset + elems);
			if ( cudaMemcpy(output->data() + offset, hidden, elems * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess )
				return 0;
		}
		return 1;
	}

	int KeepPath(const uint32_t *path, uint32_t length)
	{
		return cudaMemcpy(row_indices, path, length * sizeof(uint32_t), cudaMemcpyHostToDevice) == cudaSuccess &&
			cudaMemcpy(kept, &length, sizeof(uint32_t), cudaMemcpyHostToDevice) == cudaSuccess &&
			M::KeepRows(state, &last, kept, stream) == SPARK_STATUS_OK && cudaStreamSynchronize(stream) == cudaSuccess ? 1 : 0;
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
		for ( uint32_t sequence = 0u; sequence < SLOTS; ++sequence )
			for ( uint32_t position = 0u; position < PAGES * GLM_PAGE_SLOTS; ++position )
			{
				const uint64_t page = (uint64_t)sequence * PAGES + position / GLM_PAGE_SLOTS, slot = position % GLM_PAGE_SLOTS;
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
		return M::CheckErrors(state) == SPARK_STATUS_OK ? 1u : 0u;
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
	{
		static const uint32_t kTreeRows = 8u;
		static const uint32_t parents_a[kTreeRows] = { SPARK_SPECULATION_TREE_FRAME_NO_ROW, 0u, 1u, 2u, 3u, 1u, 5u, 0u };
		static const uint32_t depths_a[kTreeRows] = { 0u, 1u, 2u, 3u, 4u, 2u, 3u, 1u };
		static const uint32_t variants_a[kTreeRows] = { 10u, 11u, 12u, 13u, 14u, 25u, 26u, 37u };
		static const uint32_t parents_b[kTreeRows] = { SPARK_SPECULATION_TREE_FRAME_NO_ROW, 0u, 1u, 2u, 2u, 4u, 0u, 6u };
		static const uint32_t depths_b[kTreeRows] = { 0u, 1u, 2u, 3u, 3u, 4u, 1u, 2u };
		static const uint32_t paths[3][5] = { { 0u, 1u, 2u, 3u, 4u }, { 0u, 1u, 5u, 6u, 0u }, { 0u, 7u, 0u, 0u, 0u } };
		static const uint32_t lengths[3] = { 5u, 4u, 2u };
		VfRow tree_a[kTreeRows], tree_b[kTreeRows];
		std::vector<uint16_t> tree_out, seq_out[3], eager_b;
		std::vector<uint16_t> graph_a((uint64_t)kTreeRows * G::kHidden), graph_b((uint64_t)kTreeRows * G::kHidden);
		std::vector<uint8_t> after_tree;
		SparkStageRunnerStep tree_step, step_a, step_b;
		cudaGraph_t graph = 0;
		cudaGraphExec_t exec = 0;
		for ( uint32_t row = 0u; row < kTreeRows; ++row )
		{
			tree_a[row] = { 0u, VF_PREFIX + depths_a[row], variants_a[row] };
			tree_b[row] = { 0u, VF_PREFIX + depths_b[row], 70u + row };
		}
		for ( uint32_t path = 0u; path < 3u; ++path )
		{
			rig.Load(prefix);
			for ( uint32_t k = 0u; k < lengths[path]; ++k )
				Require(rig.Decode(0u, VF_PREFIX + k, variants_a[paths[path][k]], &seq_out[path]) != 0, model, "tree path sequential decode runs", path, k, 0u);
		}
		rig.Load(prefix);
		Require(rig.RunTree(tree_a, parents_a, kTreeRows, 16u, &tree_out) != 0, model, "multi-branch tree verify runs", 0u, 0u, 0u);
		after_tree = rig.Save();
		tree_step = rig.last;
		Require(tree_step.graph_key == 2u, model, "tree verify graph_key is 2", tree_step.graph_key, 0u, 0u);
		Require(Finite(tree_out) != 0u, model, "tree verify rows are finite and not constant", 0u, 0u, 0u);
		for ( uint32_t path = 0u; path < 3u; ++path )
			for ( uint32_t k = 0u; k < lengths[path]; ++k )
				Require(memcmp(tree_out.data() + (uint64_t)paths[path][k] * G::kHidden, seq_out[path].data() + (uint64_t)k * G::kHidden,
						G::kHidden * sizeof(uint16_t)) == 0, model, "every tree node equals its sequential decode bit for bit (path, depth, row)",
					path, k, paths[path][k]);
		Require(memcmp(tree_out.data() + (uint64_t)5u * G::kHidden, tree_out.data() + (uint64_t)2u * G::kHidden, G::kHidden * sizeof(uint16_t)) != 0,
			model, "branch siblings at the same depth differ", 0u, 0u, 0u);
		for ( uint32_t path = 0u; path < 3u; ++path )
		{
			const uint32_t length = lengths[path], limit = VF_PREFIX + length + 2u;
			std::vector<uint16_t> plain, speculated, unused;
			std::vector<uint8_t> reference, plain_kv, speculated_kv;
			rig.Load(prefix);
			for ( uint32_t k = 0u; k < length; ++k )
				Require(rig.Decode(0u, VF_PREFIX + k, variants_a[paths[path][k]], &unused) != 0, model, "tree keep reference decode runs", path, k, 0u);
			reference = rig.Save();
			Require(rig.Decode(0u, VF_PREFIX + length, 1u, &plain) != 0 && rig.Decode(0u, VF_PREFIX + length + 1u, 2u, &plain) != 0,
				model, "plain follow-up decodes run", path, 0u, 0u);
			plain_kv = rig.Save();
			rig.Load(after_tree);
			rig.last = tree_step;
			Require(rig.KeepPath(paths[path], length) != 0, model, "KeepRows relocates accepted tree path KV", path, 0u, 0u);
			Require(rig.KvMatches(rig.Save(), reference, VF_PREFIX + length) != 0u, model,
				"relocated KV matches sequential decode bit for bit", path, 0u, 0u);
			Require(rig.Decode(0u, VF_PREFIX + length, 1u, &speculated) != 0 && rig.Decode(0u, VF_PREFIX + length + 1u, 2u, &speculated) != 0,
				model, "follow-up decodes after tree KeepRows run", path, 0u, 0u);
			speculated_kv = rig.Save();
			Require(FirstDifference(speculated, plain, G::kHidden) == UINT32_MAX, model,
				"follow-up decodes after KeepRows equal plain run bit for bit (path, first differing step)", path, FirstDifference(speculated, plain, G::kHidden), 0u);
			Require(rig.KvMatches(speculated_kv, plain_kv, limit) != 0u, model,
				"follow-up KV after KeepRows equals plain run bit for bit", path, 0u, 0u);
		}
		rig.Load(prefix);
		Require(rig.RunTree(tree_b, parents_b, kTreeRows, 16u, &eager_b) != 0, model, "eager tree B runs for graph check", 0u, 0u, 0u);
		rig.Load(prefix);
		Require(rig.UploadTree(tree_a, parents_a, kTreeRows, 16u, &step_a) != 0 &&
			cudaStreamSynchronize(rig.stream) == cudaSuccess &&
			cudaStreamBeginCapture(rig.stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess &&
			GlmModel<G>::Step(rig.state, &step_a, rig.stream) == SPARK_STATUS_OK &&
			cudaStreamEndCapture(rig.stream, &graph) == cudaSuccess &&
			cudaGraphInstantiate(&exec, graph, 0, 0, 0) == cudaSuccess, model, "tree verify CUDA graph captures cleanly", 0u, 0u, 0u);
		Require(cudaGraphLaunch(exec, rig.stream) == cudaSuccess && cudaStreamSynchronize(rig.stream) == cudaSuccess &&
			cudaMemcpy(graph_a.data(), rig.hidden, graph_a.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) == cudaSuccess,
			model, "tree verify CUDA graph launches for tree A", 0u, 0u, 0u);
		Require(FirstDifference(graph_a, tree_out, G::kHidden) == UINT32_MAX, model,
			"tree verify CUDA graph A equals eager bit for bit", FirstDifference(graph_a, tree_out, G::kHidden), 0u, 0u);
		rig.Load(prefix);
		Require(rig.UploadTree(tree_b, parents_b, kTreeRows, 16u, &step_b) != 0 &&
			cudaGraphLaunch(exec, rig.stream) == cudaSuccess && cudaStreamSynchronize(rig.stream) == cudaSuccess &&
			cudaMemcpy(graph_b.data(), rig.hidden, graph_b.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) == cudaSuccess,
			model, "tree verify CUDA graph replays tree B with 0 recaptures", 0u, 0u, 0u);
		Require(FirstDifference(graph_b, eager_b, G::kHidden) == UINT32_MAX, model,
			"tree verify CUDA graph B equals eager bit for bit", FirstDifference(graph_b, eager_b, G::kHidden), 0u, 0u);
		cudaGraphExecDestroy(exec);
		cudaGraphDestroy(graph);
	}
	Require(rig.KvClean() != 0u, model, "no step reported a KV access error", 0u, 0u, 0u);
	printf("%s %s: frames of 1-%u rows, %u rejection points\n", vf_failures == 0 ? "ok" : "checked", model, VF_MAX_FRAME, checked);
}

template<class G>
static void CheckPrefixExact(void)
{
	static const uint32_t deltas[] = { 1u, 4u, 8u, 12u, 16u, 23u, 64u, 128u };
	static constexpr uint32_t kCached = 9u * GLM_PAGE_SLOTS;
	static constexpr uint32_t kMaxTotal = kCached + 128u;
	VfRig<G, 768u, 16u> rig;
	const char *model = G::Tag();
	std::vector<VfRow> all_rows(kMaxTotal);
	std::vector<uint8_t> cached_kv;
	if ( rig.Open() == 0 )
	{
		Require(0, model, "prefix-exact synthetic stage opens", 0u, 0u, 0u);
		return;
	}
	for ( uint32_t position = 0u; position < kMaxTotal; ++position )
		all_rows[position] = { 0u, position, 0u };
	Require(rig.Run(all_rows.data(), kCached, 1u, 0, 1u) != 0, model, "cached prefix prefill runs", kCached, 0u, 0u);
	cached_kv = rig.Save();
	for ( uint32_t delta : deltas )
	{
		const uint32_t total = kCached + delta;
		std::vector<uint16_t> cold_out, warm_out;
		std::vector<uint8_t> cold_kv;
		Require(cudaMemset(rig.kv, 0, rig.kLatentRegion + rig.kIndexRegion) == cudaSuccess, model, "prefix-exact KV reset succeeds", delta, 0u, 0u);
		Require(rig.Run(all_rows.data(), total, 1u, &cold_out, 1u) != 0, model, "cold prefill runs", total, delta, 0u);
		cold_kv = rig.Save();
		Require(Finite(cold_out) != 0u, model, "cold prefill rows are finite", total, delta, 0u);
		std::vector<uint16_t> cold_tail(cold_out.begin() + (ptrdiff_t)((uint64_t)kCached * G::kHidden), cold_out.end());
		Require(rig.Load(cached_kv) != 0, model, "cached KV restore succeeds", delta, 0u, 0u);
		Require(rig.Run(all_rows.data() + kCached, delta, 1u, &warm_out, 1u) != 0, model, "warm delta prefill runs", kCached, delta, 0u);
		Require(FirstDifference(warm_out, cold_tail, G::kHidden) == UINT32_MAX, model,
			"warm delta prefill equals cold prefill bit for bit (delta, first differing row)", delta, FirstDifference(warm_out, cold_tail, G::kHidden), 0u);
		Require(rig.KvMatches(rig.Save(), cold_kv, total) != 0u, model,
			"warm delta prefill leaves the same KV as cold prefill bit for bit (delta)", delta, 0u, 0u);
	}
	Require(rig.KvClean() != 0u, model, "prefix-exact steps reported no KV access error", 0u, 0u, 0u);
	printf("%s %s prefix-exact: cached=%u vs warm deltas 1..128\n", vf_failures == 0 ? "ok" : "checked", model, kCached);
}

template<class G>
static void Check5cBatchInvariance(void)
{
	static constexpr uint32_t kSlots = 8u;
	static const uint32_t ctx_list[] = { 32u, 96u };
	static const uint32_t batches[] = { 2u, 4u, 8u };
	VfRig<G, 128u, 4u, kSlots> rig;
	const char *model = G::Tag();
	if ( rig.Open() == 0 )
	{
		Require(0, model, "5c batch invariance rig opens", 0u, 0u, 0u);
		return;
	}
	for ( uint32_t seq = 0u; seq < kSlots; ++seq )
	{
		const uint32_t fill = seq == 0u ? 96u : (20u + seq * 3u);
		std::vector<VfRow> init_rows(fill);
		for ( uint32_t p = 0u; p < fill; ++p )
			init_rows[p] = { seq, p, 0u };
		Require(rig.Run(init_rows.data(), fill, 0u, 0) != 0, model, "5c prefix fill runs", seq, fill, 0u);
	}
	const std::vector<uint8_t> saved = rig.Save();
	for ( uint32_t ctx : ctx_list )
	{
		std::vector<uint16_t> b1_out;
		Require(rig.Load(saved) != 0, model, "5c KV restore succeeds", ctx, 1u, 0u);
		Require(rig.Decode(0u, ctx, 7u, &b1_out) != 0, model, "5c B1 decode runs", ctx, 1u, 0u);
		Require(Finite(b1_out) != 0u, model, "5c B1 output is finite", ctx, 1u, 0u);
		for ( uint32_t bs : batches )
		{
			std::vector<VfRow> batch(bs);
			std::vector<uint16_t> out;
			batch[0] = { 0u, ctx, 7u };
			for ( uint32_t r = 1u; r < bs; ++r )
				batch[r] = { r, 20u + r * 3u, 11u + r };
			Require(rig.Load(saved) != 0, model, "5c KV restore succeeds for batch", ctx, bs, 0u);
			Require(rig.Run(batch.data(), bs, 1u, &out, 0u) != 0, model, "5c batch decode runs", ctx, bs, 0u);
			std::vector<uint16_t> row0(out.begin(), out.begin() + G::kHidden);
			Require(FirstDifference(row0, b1_out, G::kHidden) == UINT32_MAX, model,
				"5c batch decode row0 equals B1 bit for bit (ctx, batch, first diff)", ctx, bs, FirstDifference(row0, b1_out, G::kHidden));
		}
	}
	Require(rig.KvClean() != 0u, model, "5c batch invariance reported no KV access error", 0u, 0u, 0u);
	printf("%s %s 5c batch invariance: B1 == B2 == B4 == B8 (short & sparse ctx)\n", vf_failures == 0 ? "ok" : "checked", model);
}

int main(void)
{
	Check<SmallGlm53Full>();
	Check<SmallAxk2>();
	Check<SmallGlm53FullInter128>();
	Check<SmallAxk2Inter128>();
	Check5cBatchInvariance<SmallGlm53Full>();
	Check5cBatchInvariance<SmallAxk2>();
	Check5cBatchInvariance<SmallGlm53FullInter128>();
	Check5cBatchInvariance<SmallAxk2Inter128>();
	CheckPrefixExact<SmallGlm53FullInter128>();
	CheckPrefixExact<SmallAxk2Inter128>();
	CheckPrefixExact<SmallAxk2>();
	if ( vf_failures != 0 )
	{
		printf("FAIL glm52 verify frames: %d failures\n", vf_failures);
		return 1;
	}
	printf("PASS glm52 verify frames: every row of every verify frame equals one-row decode bit for bit, and a rejected tail leaves no trace\n");
	return 0;
}
