#include "modules/solar_open2_tp16_resident_decode_stage/source/spark_solar_open2_tp16_stage_model.cu"
#include "modules/ling3_resident_decode_stage/source/spark_ling3_stage_model.cu"
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
#define VF_FFN 256u
#define VF_FILL_ROWS 16u
#define VF_TREE_ROWS 8u
#define VF_TREE_TOLERANCE 0.05f

static_assert(SolarOpen2Geometry<SPARK_SOLAR_OPEN2_TP16_STAGE_TP_DEGREE>::kVerifyRows == VF_MAX_FRAME && Ling3Geometry<SPARK_LING3_STAGE_TP_DEGREE>::kVerifyRows == VF_MAX_FRAME,
	"the test frames reach the verify rows Solar Open 2 and Ling 3.0 serve");

struct SmallSolar : SolarOpen2Geometry<SPARK_SOLAR_OPEN2_TP16_STAGE_TP_DEGREE>
{
	static constexpr uint32_t kTpDegree = 1u;
	static constexpr uint32_t kHidden = 1536u;
	static constexpr uint32_t kLayers = 4u;
	static constexpr uint32_t kKdaHeads = 4u;
	static constexpr uint32_t kGqaHeads = 8u;
	static constexpr uint32_t kGqaKvHeads = 1u;
	static constexpr uint32_t kExperts = 16u;
	static constexpr uint32_t kLocalExperts = 16u;
	static const char *Tag(void) { return "solar-open-2"; }
};

struct SmallLing3 : Ling3Geometry<SPARK_LING3_STAGE_TP_DEGREE>
{
	static constexpr uint32_t kTpDegree = 1u;
	static constexpr uint32_t kHidden = 2304u;
	static constexpr uint32_t kLayers = 6u;
	static constexpr uint32_t kKdaHeads = 2u;
	static constexpr uint32_t kMlaHeads = 2u;
	static constexpr uint32_t kExperts = 32u;
	static constexpr uint32_t kLocalExperts = 16u;
	static constexpr uint32_t kMoeInter = 128u;
	static float RoutedLimit(uint32_t layer) { return layer == 4u ? 4.0f : 0.0f; }
	static float SharedLimit(uint32_t layer) { return layer == 3u ? 5.0f : 0.0f; }
	static const char *Tag(void) { return "ling-3.0-flash"; }
};

template<class G>
struct VfRig
{
	using M = LingModel<G>;
	using S = LingShape<G>;
	using State = LingState<G>;
	static constexpr uint64_t kKvRegion = (uint64_t)S::kAttnLayers * VF_SLOTS * VF_PAGES * S::Kv::kPageBytes;
	State *state;
	VfBus bus;
	SparkStageRunnerServices services;
	cudaStream_t stream;
	uint32_t multiprocessors;
	uint8_t *kv;
	uint32_t *page_table, *positions, *context, *sequence_of_row, *row_begin, *row_indices, *recurrent, *kept;
	uint32_t *slots, *forks, *offsets, *parents, *loads, *saves, *path_begin;
	uint16_t *hidden;
	uint32_t host_context[VF_SLOTS];
	uint32_t host_slots[VF_MAX_FRAME], host_parents[VF_MAX_FRAME];
	SparkStageRunnerStep last;

	uint64_t StateBytes(void) const
	{
		return (uint64_t)S::kKdaLayers * VF_SLOTS * (state->kda_state_bytes + state->kda_window_bytes);
	}

	void Layer(LingLayer *layer, uint32_t index)
	{
		layer->input_norm = Norm(G::kHidden);
		layer->post_norm = Norm(G::kHidden);
		if ( index % G::kPeriod == G::kPhase )
		{
			if constexpr ( S::kGqa )
			{
				layer->gqa_qkv = Matrix(S::kGqaRankFused, G::kHidden);
				layer->gqa_gate = Matrix(S::kGqaRankWidth, G::kHidden);
				layer->gqa_out = Matrix(G::kHidden, S::kGqaRankWidth);
			}
			else
			{
				layer->mla_q = Matrix(S::kMlaRankHeads * S::kQkDim, G::kHidden);
				layer->mla_kv_a = Matrix(S::kSlot, G::kHidden);
				layer->mla_kv_norm = Norm(G::kLatent);
				layer->mla_uk = Tensor<uint16_t>((uint64_t)S::kMlaRankHeads * G::kLatent * G::kNope, 1.7f / sqrtf((float)G::kNope), 0.0f);
				layer->mla_uv = Tensor<uint16_t>((uint64_t)S::kMlaRankHeads * G::kValueDim * G::kLatent, 1.7f / sqrtf((float)G::kLatent), 0.0f);
				layer->mla_gate = Matrix(S::kMlaRankHeads, G::kHidden);
				layer->mla_out = Matrix(G::kHidden, S::kMlaRankHeads * G::kValueDim);
			}
		}
		else
		{
			layer->kda_qkv = Matrix(S::kKdaChannels, G::kHidden);
			if ( G::kKdaRank != 0u )
			{
				layer->kda_f_a = Matrix(G::kKdaRank, G::kHidden);
				layer->kda_g_a = Matrix(G::kKdaRank, G::kHidden);
			}
			layer->kda_f = Matrix(S::kKdaWidth, G::kKdaRank != 0u ? G::kKdaRank : G::kHidden);
			layer->kda_g = Matrix(S::kKdaWidth, G::kKdaRank != 0u ? G::kKdaRank : G::kHidden);
			layer->kda_beta = Matrix(S::kKdaRankHeads, G::kHidden);
			layer->kda_conv = Tensor<uint16_t>((uint64_t)S::kKdaChannels * G::kConv, 0.5f, 0.0f);
			layer->kda_a_log = Tensor<float>(S::kKdaRankHeads, 0.5f, 0.0f);
			layer->kda_dt_bias = Tensor<float>(S::kKdaWidth, 0.5f, 0.0f);
			layer->kda_norm = Tensor<uint16_t>(G::kKdaDim, 0.2f, 1.0f);
			layer->kda_out = Matrix(G::kHidden, S::kKdaWidth);
		}
		if ( index < G::kDenseLayers )
		{
			layer->ffn_gate_up = Matrix(2u * VF_FFN, G::kHidden);
			layer->ffn_down = Matrix(G::kHidden, VF_FFN);
		}
		else
		{
			layer->router = Matrix(G::kExperts, G::kHidden);
			layer->expert_bias = Tensor<float>(G::kExperts, 0.05f, 0.0f);
			layer->moe_w1 = Matrix((uint64_t)G::kLocalExperts * G::kMoeInter, G::kHidden);
			layer->moe_w3 = Matrix((uint64_t)G::kLocalExperts * G::kMoeInter, G::kHidden);
			layer->moe_w2 = Tensor<uint16_t>((uint64_t)G::kLocalExperts * G::kHidden * G::kMoeInter, 1.7f / sqrtf((float)G::kMoeInter), 0.0f);
			layer->shared_gate_up = Matrix(2u * G::kSharedRows, G::kHidden);
			layer->shared_down = Matrix(G::kHidden, G::kSharedRows);
		}
	}

	int Open(void)
	{
		std::vector<uint32_t> table(VF_SLOTS * VF_PAGES);
		SparkStageRunnerKv attach;
		memset(this, 0, sizeof(*this));
		bus.hidden = G::kHidden;
		services.context = &bus;
		services.round = VfRound;
		state = (State *)calloc(1u, sizeof(State));
		state->services = &services;
		state->tp_degree = 1u;
		state->max_rows = VF_MAX_FRAME;
		state->slots = VF_SLOTS;
		state->ffn_rows = VF_FFN;
		if ( M::Size(state) != SPARK_STATUS_OK )
			return 0;
		for ( uint32_t index = 0u; index < G::kLayers; ++index )
			Layer(&state->layers[index], index);
		if ( M::Allocate(state) != SPARK_STATUS_OK || cudaStreamCreate(&stream) != cudaSuccess ||
			cudaDeviceGetAttribute((int *)&multiprocessors, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess ||
			cudaMalloc((void **)&kv, kKvRegion) != cudaSuccess || cudaMemset(kv, 0, kKvRegion) != cudaSuccess ||
			cudaMalloc((void **)&page_table, table.size() * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&positions, VF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&sequence_of_row, VF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&row_indices, VF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&row_begin, (VF_SLOTS + 1u) * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&recurrent, VF_SLOTS * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&kept, VF_SLOTS * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&context, VF_SLOTS * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&slots, VF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&forks, VF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&offsets, VF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&parents, VF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&loads, VF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&saves, VF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&path_begin, 2u * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&hidden, (uint64_t)VF_MAX_FRAME * G::kHidden * sizeof(uint16_t)) != cudaSuccess )
			return 0;
		for ( uint32_t index = 0u; index < table.size(); ++index )
			table[index] = index;
		if ( cudaMemcpy(page_table, table.data(), table.size() * sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess )
			return 0;
		memset(&attach, 0, sizeof(attach));
		attach.pool = kv;
		attach.layer_stride_bytes = (uint64_t)VF_SLOTS * VF_PAGES * S::Kv::kPageBytes;
		attach.layer_page_bytes = S::Kv::kPageBytes;
		attach.layer_count = S::kAttnLayers;
		attach.page_table = page_table;
		attach.page_table_stride = VF_PAGES;
		attach.pool_page_count = VF_SLOTS * VF_PAGES;
		attach.sequence_count = VF_SLOTS;
		attach.context_shard.degree = 1u;
		attach.context_shard.grain = 1u;
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
		std::vector<uint32_t> position(count), sequence(count), order(count), begin(1u, 0u), slots;
		SparkStageRunnerStep step;
		for ( uint32_t row = 0u; row < count; ++row )
		{
			Input(rows[row], input.data() + (uint64_t)row * G::kHidden);
			position[row] = rows[row].position;
			sequence[row] = rows[row].sequence;
			order[row] = row;
			if ( slots.empty() || slots.back() != rows[row].sequence )
			{
				if ( !slots.empty() )
					begin.push_back(row);
				slots.push_back(rows[row].sequence);
			}
			host_context[rows[row].sequence] = rows[row].position + 1u;
		}
		begin.push_back(count);
		memset(&step, 0, sizeof(step));
		step.hidden_bf16 = hidden;
		step.positions = positions;
		step.context_length = context;
		step.sequence_of_row = sequence_of_row;
		step.sequence_row_begin = row_begin;
		step.sequence_row_indices = row_indices;
		step.recurrent_index = recurrent;
		step.rows = count;
		step.sequences = (uint32_t)slots.size();
		step.commit = commit;
		step.context = VF_PAGES * G::kPageSlots;
		step.multiprocessors = multiprocessors;
		step.route_fault = VfRouteFault();
		last = step;
		if ( cudaMemcpyAsync(hidden, input.data(), input.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			cudaMemcpyAsync(positions, position.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			cudaMemcpyAsync(sequence_of_row, sequence.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			cudaMemcpyAsync(row_indices, order.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			cudaMemcpyAsync(row_begin, begin.data(), begin.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			cudaMemcpyAsync(recurrent, slots.data(), slots.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
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

	int Keep(const uint32_t *counts)
	{
		return cudaMemcpy(kept, counts, last.sequences * sizeof(uint32_t), cudaMemcpyHostToDevice) == cudaSuccess &&
			M::KeepRows(state, &last, kept, stream) == SPARK_STATUS_OK && cudaStreamSynchronize(stream) == cudaSuccess;
	}

	int RunTree(const VfRow *rows, const uint32_t *parent_rows, uint32_t count, std::vector<uint16_t> *output)
	{
		std::vector<uint16_t> input((uint64_t)count * G::kHidden);
		std::vector<uint32_t> position(count), sequence(count, rows[0].sequence), order(count), begin(2u), tokens(count), slot_table(count), fork_table(count);
		std::vector<uint32_t> offset_table(count), load_table(count), save_table(count, 0u);
		SparkSpeculationTreeLayout layout;
		SparkStageRunnerStep step;
		for ( uint32_t row = 0u; row < count; ++row )
			tokens[row] = rows[row].variant;
		if ( count > VF_MAX_FRAME || SparkSpeculationTreeLayoutBuild(parent_rows, tokens.data(), count, rows[0].position, 1u, &layout) != SPARK_STATUS_OK )
			return 0;
		for ( uint32_t row = 0u; row < count; ++row )
		{
			Input(rows[row], input.data() + (uint64_t)row * G::kHidden);
			position[row] = layout.anchor_position + layout.row[row].depth;
			if ( position[row] != rows[row].position )
				return 0;
			order[row] = row;
			slot_table[row] = layout.row[row].slot;
			fork_table[row] = layout.row[row].fork_position;
			offset_table[row] = layout.row[row].slot_offset;
			load_table[row] = row != 0u && layout.row[row].branch != layout.row[parent_rows[row]].branch ? layout.row[row].branch : 0u;
			host_slots[row] = layout.row[row].slot;
			host_parents[row] = parent_rows[row];
		}
		for ( uint32_t row = 1u; row < count; ++row )
			if ( load_table[row] != 0u )
				save_table[parent_rows[row]] |= 1u << (load_table[row] - 1u);
		begin[0] = 0u;
		begin[1] = count;
		host_context[rows[0].sequence] = layout.context;
		memset(&step, 0, sizeof(step));
		step.hidden_bf16 = hidden;
		step.positions = positions;
		step.context_length = context;
		step.sequence_of_row = sequence_of_row;
		step.sequence_row_begin = row_begin;
		step.sequence_row_indices = row_indices;
		step.recurrent_index = recurrent;
		step.rows = count;
		step.sequences = 1u;
		step.commit = 0u;
		step.context = VF_PAGES * G::kPageSlots;
		step.multiprocessors = multiprocessors;
		step.route_fault = VfRouteFault();
		step.tree = 1u;
		step.slots = slots;
		step.fork_positions = forks;
		step.slot_offsets = offsets;
		step.parent_rows = parents;
		step.state_loads = loads;
		step.state_saves = saves;
		step.host_slots = host_slots;
		step.host_parent_rows = host_parents;
		last = step;
		if ( cudaMemcpyAsync(hidden, input.data(), input.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			cudaMemcpyAsync(positions, position.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			cudaMemcpyAsync(sequence_of_row, sequence.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			cudaMemcpyAsync(row_indices, order.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			cudaMemcpyAsync(row_begin, begin.data(), begin.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			cudaMemcpyAsync(recurrent, sequence.data(), sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			cudaMemcpyAsync(context, host_context, sizeof(host_context), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			cudaMemcpyAsync(slots, slot_table.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			cudaMemcpyAsync(forks, fork_table.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			cudaMemcpyAsync(offsets, offset_table.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			cudaMemcpyAsync(parents, parent_rows, count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			cudaMemcpyAsync(loads, load_table.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			cudaMemcpyAsync(saves, save_table.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			M::Step(state, &step, stream) != SPARK_STATUS_OK || cudaStreamSynchronize(stream) != cudaSuccess )
		{
			(void)cudaStreamSynchronize(stream);
			return 0;
		}
		if ( output != 0 )
		{
			const uint64_t offset = output->size();
			output->resize(offset + input.size());
			if ( cudaMemcpy(output->data() + offset, hidden, input.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess )
				return 0;
		}
		return 1;
	}

	int KeepPath(const uint32_t *path, uint32_t length)
	{
		const uint32_t begin[2] = { 0u, length };
		if ( cudaMemcpy(row_indices, path, length * sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess ||
			cudaMemcpy(path_begin, begin, sizeof(begin), cudaMemcpyHostToDevice) != cudaSuccess ||
			cudaMemcpy(kept, &length, sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess )
			return 0;
		last.sequence_row_indices = row_indices;
		last.sequence_row_begin = path_begin;
		return M::KeepRows(state, &last, kept, stream) == SPARK_STATUS_OK && cudaStreamSynchronize(stream) == cudaSuccess;
	}

	float RecurrentDistance(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b) const
	{
		const uint64_t states = (uint64_t)S::kKdaLayers * VF_SLOTS * state->kda_state_bytes;
		const float *fa = (const float *)(a.data() + kKvRegion), *fb = (const float *)(b.data() + kKvRegion);
		const uint16_t *wa = (const uint16_t *)(a.data() + kKvRegion + states), *wb = (const uint16_t *)(b.data() + kKvRegion + states);
		float worst = 0.0f, largest = 1.0e-6f;
		if ( a.size() != b.size() )
			return 1.0e30f;
		for ( uint64_t index = 0u; index < states / sizeof(float); ++index )
		{
			worst = fabsf(fa[index] - fb[index]) > worst ? fabsf(fa[index] - fb[index]) : worst;
			largest = fabsf(fa[index]) > largest ? fabsf(fa[index]) : largest;
		}
		for ( uint64_t index = 0u; index < (StateBytes() - states) / sizeof(uint16_t); ++index )
		{
			const float difference = fabsf(VfFloat(wa[index]) - VfFloat(wb[index]));
			worst = difference > worst ? difference : worst;
			largest = fabsf(VfFloat(wa[index])) > largest ? fabsf(VfFloat(wa[index])) : largest;
		}
		return worst / largest;
	}

	int Decode(uint32_t sequence, uint32_t position, uint32_t variant, std::vector<uint16_t> *output)
	{
		const VfRow row = { sequence, position, variant };
		return Run(&row, 1u, 1u, output);
	}

	std::vector<uint8_t> Save(void) const
	{
		std::vector<uint8_t> copy(kKvRegion + StateBytes());
		const uint64_t states = (uint64_t)S::kKdaLayers * VF_SLOTS * state->kda_state_bytes;
		if ( cudaMemcpy(copy.data(), kv, kKvRegion, cudaMemcpyDeviceToHost) != cudaSuccess ||
			cudaMemcpy(copy.data() + kKvRegion, state->kda_state, states, cudaMemcpyDeviceToHost) != cudaSuccess ||
			cudaMemcpy(copy.data() + kKvRegion + states, state->kda_window, StateBytes() - states, cudaMemcpyDeviceToHost) != cudaSuccess )
			copy.clear();
		return copy;
	}

	int Load(const std::vector<uint8_t> &copy)
	{
		const uint64_t states = (uint64_t)S::kKdaLayers * VF_SLOTS * state->kda_state_bytes;
		return copy.size() == kKvRegion + StateBytes() && cudaMemcpy(kv, copy.data(), kKvRegion, cudaMemcpyHostToDevice) == cudaSuccess &&
			cudaMemcpy(state->kda_state, copy.data() + kKvRegion, states, cudaMemcpyHostToDevice) == cudaSuccess &&
			cudaMemcpy(state->kda_window, copy.data() + kKvRegion + states, StateBytes() - states, cudaMemcpyHostToDevice) == cudaSuccess;
	}

	uint32_t RecurrentMatches(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b) const
	{
		return a.size() == b.size() && memcmp(a.data() + kKvRegion, b.data() + kKvRegion, StateBytes()) == 0 ? 1u : 0u;
	}

	static uint32_t KvMatches(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b, uint32_t limit)
	{
		using Kv = typename S::Kv;
		if ( a.size() != b.size() )
			return 0u;
		for ( uint32_t layer = 0u; layer < S::kAttnLayers; ++layer )
			for ( uint32_t sequence = 0u; sequence < VF_SLOTS; ++sequence )
				for ( uint32_t position = 0u; position < VF_PAGES * Kv::kPageSlots; ++position )
				{
					const uint64_t offset = ((uint64_t)layer * VF_SLOTS * VF_PAGES + sequence * VF_PAGES + position / Kv::kPageSlots) * Kv::kPageBytes +
						(uint64_t)(position % Kv::kPageSlots) * Kv::kSlotBytes;
					if ( (sequence != 0u || position < limit) && memcmp(a.data() + offset, b.data() + offset, Kv::kSlotBytes) != 0 )
						return 0u;
				}
		return 1u;
	}

	uint32_t KvClean(void) const
	{
		LmKvAccessError error, zero;
		memset(&zero, 0, sizeof(zero));
		return cudaMemcpy(&error, state->kv_error, sizeof(error), cudaMemcpyDeviceToHost) == cudaSuccess && memcmp(&error, &zero, sizeof(error)) == 0 ? 1u : 0u;
	}

	void Close(void)
	{
		M::Close(state);
		cudaFree(kv);
		cudaFree(page_table);
		cudaFree(positions);
		cudaFree(sequence_of_row);
		cudaFree(row_indices);
		cudaFree(row_begin);
		cudaFree(recurrent);
		cudaFree(kept);
		cudaFree(context);
		cudaFree(slots);
		cudaFree(forks);
		cudaFree(offsets);
		cudaFree(parents);
		cudaFree(loads);
		cudaFree(saves);
		cudaFree(path_begin);
		cudaFree(hidden);
		cudaStreamDestroy(stream);
	}
};

static float RowDistance(const std::vector<uint16_t> &a, uint32_t a_row, const std::vector<uint16_t> &b, uint32_t b_row, uint32_t width)
{
	float worst = 0.0f;
	for ( uint32_t index = 0u; index < width; ++index )
	{
		const float difference = fabsf(VfFloat(a[(uint64_t)a_row * width + index]) - VfFloat(b[(uint64_t)b_row * width + index]));
		worst = difference > worst ? difference : worst;
	}
	return worst;
}

template<class G>
static void CheckTree(VfRig<G> &rig, const std::vector<uint8_t> &prefix)
{
	using S = LingShape<G>;
	static const uint32_t parents[VF_TREE_ROWS] = { SPARK_SPECULATION_TREE_FRAME_NO_ROW, 0u, 1u, 2u, 3u, 1u, 5u, 0u };
	static const uint32_t depths[VF_TREE_ROWS] = { 0u, 1u, 2u, 3u, 4u, 2u, 3u, 1u };
	static const uint32_t variants[VF_TREE_ROWS] = { 10u, 11u, 12u, 13u, 14u, 25u, 26u, 37u };
	static const uint32_t paths[3][5] = { { 0u, 1u, 2u, 3u, 4u }, { 0u, 1u, 5u, 6u, 0u }, { 0u, 7u, 0u, 0u, 0u } };
	static const uint32_t lengths[3] = { 5u, 4u, 2u };
	const char *model = G::Tag();
	VfRow tree_rows[VF_TREE_ROWS];
	std::vector<uint16_t> tree_out, chain_out[3];
	std::vector<uint8_t> after_tree;
	SparkStageRunnerStep tree_step;
	for ( uint32_t row = 0u; row < VF_TREE_ROWS; ++row )
		tree_rows[row] = { 0u, VF_PREFIX + depths[row], variants[row] };
	if constexpr ( !S::kGqa )
	{
		rig.Load(prefix);
		Require(rig.RunTree(tree_rows, parents, VF_TREE_ROWS, &tree_out) == 0, model, "a tree frame on the latent attention stage is refused", 0u, 0u, 0u);
		return;
	}
	for ( uint32_t path = 0u; path < 3u; ++path )
	{
		VfRow chain_rows[5];
		for ( uint32_t k = 0u; k < lengths[path]; ++k )
			chain_rows[k] = tree_rows[paths[path][k]];
		rig.Load(prefix);
		Require(rig.Run(chain_rows, lengths[path], 0u, &chain_out[path]) != 0, model, "a root path verifies as a chain frame", path, 0u, 0u);
	}
	rig.Load(prefix);
	Require(rig.RunTree(tree_rows, parents, VF_TREE_ROWS, &tree_out) != 0, model, "the tree frame runs", 0u, 0u, 0u);
	after_tree = rig.Save();
	tree_step = rig.last;
	Require(Finite(tree_out) != 0u, model, "tree rows are finite and not constant", 0u, 0u, 0u);
	Require(memcmp(tree_out.data(), chain_out[0].data(), (uint64_t)lengths[0] * G::kHidden * sizeof(uint16_t)) == 0, model,
		"the trunk rows of a tree frame equal the trunk verified as a chain bit for bit (first differing row)", FirstDifference(tree_out, chain_out[0], G::kHidden), 0u, 0u);
	for ( uint32_t path = 1u; path < 3u; ++path )
		for ( uint32_t k = 1u; k < lengths[path]; ++k )
		{
			const float distance = RowDistance(tree_out, paths[path][k], chain_out[path], k, G::kHidden);
			Require(distance < VF_TREE_TOLERANCE, model, "a branch row equals its root path verified as a chain up to attention summation order (path, row, distance x 1e6)", path,
				paths[path][k], (uint32_t)(distance * 1.0e6f));
		}
	Require(RowDistance(tree_out, 5u, tree_out, 2u, G::kHidden) >= VF_TREE_TOLERANCE, model, "siblings at one position differ (distance x 1e6)",
		(uint32_t)(RowDistance(tree_out, 5u, tree_out, 2u, G::kHidden) * 1.0e6f), 0u, 0u);
	Require(rig.RecurrentMatches(after_tree, prefix) != 0u, model, "a tree frame leaves the recurrent state untouched", 0u, 0u, 0u);
	for ( uint32_t path = 0u; path < 3u; ++path )
	{
		const uint32_t length = lengths[path];
		std::vector<uint16_t> plain, speculated, unused;
		std::vector<uint8_t> reference, after_keep;
		rig.Load(prefix);
		for ( uint32_t k = 0u; k < length; ++k )
			Require(rig.Decode(0u, VF_PREFIX + k, variants[paths[path][k]], &unused) != 0, model, "one-row decode of a path row runs", path, k, 0u);
		reference = rig.Save();
		Require(rig.Decode(0u, VF_PREFIX + length, 1u, &plain) != 0 && rig.Decode(0u, VF_PREFIX + length + 1u, 2u, &plain) != 0, model,
			"the decodes of the run that never saw the tree run", path, 0u, 0u);
		rig.Load(after_tree);
		rig.last = tree_step;
		Require(rig.KeepPath(paths[path], length) != 0, model, "keep_rows folds the accepted path and relocates its KV", path, 0u, 0u);
		after_keep = rig.Save();
		if constexpr ( S::kAttnLayers == 1u && G::kPhase == 0u )
			Require(rig.KvMatches(after_keep, reference, VF_PREFIX + length) != 0u, model,
				"after keeping a path the KV at the kept positions equals the path decoded one row at a time (path)", path, 0u, 0u);
		if ( path == 0u )
			Require(rig.RecurrentMatches(after_keep, reference) != 0u, model, "after keeping the trunk the recurrent state equals the one-row decodes bit for bit", 0u, 0u, 0u);
		else
			Require(rig.RecurrentDistance(after_keep, reference) < VF_TREE_TOLERANCE, model,
				"after keeping a branch the recurrent state equals the one-row decodes up to attention summation order (path, relative distance x 1e6)", path,
				(uint32_t)(rig.RecurrentDistance(after_keep, reference) * 1.0e6f), 0u);
		Require(rig.Decode(0u, VF_PREFIX + length, 1u, &speculated) != 0 && rig.Decode(0u, VF_PREFIX + length + 1u, 2u, &speculated) != 0, model,
			"the decodes after an accepted path run", path, 0u, 0u);
		if ( path == 0u )
			Require(FirstDifference(speculated, plain, G::kHidden) == UINT32_MAX, model,
				"after keeping the trunk the next two decodes equal a run that never saw the tree (first differing step)", FirstDifference(speculated, plain, G::kHidden), 0u, 0u);
		else
			for ( uint32_t step = 0u; step < 2u; ++step )
				Require(RowDistance(speculated, step, plain, step, G::kHidden) < VF_TREE_TOLERANCE, model,
					"after keeping a branch the next two decodes equal a run that never saw the tree up to attention summation order (path, step, distance x 1e6)", path, step,
					(uint32_t)(RowDistance(speculated, step, plain, step, G::kHidden) * 1.0e6f));
	}
}

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
		Require(0, model, "the synthetic stage opens, binds and attaches its KV pool", 0u, 0u, 0u);
		return;
	}
	for ( uint32_t sequence = 0u; sequence < VF_SLOTS; ++sequence )
		for ( uint32_t first = 0u; first < (sequence == 0u ? VF_PREFIX : VF_OTHER_PREFIX); first += VF_FILL_ROWS )
		{
			rows.clear();
			for ( uint32_t position = first; position < first + VF_FILL_ROWS; ++position )
				rows.push_back({ sequence, position, 0u });
			Require(rig.Run(rows.data(), (uint32_t)rows.size(), 1u, 0) != 0, model, "the prefix fill runs", sequence, first, 0u);
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
		const SparkStageRunnerStep frame_step = rig.last;
		Require(Finite(verified) != 0u, model, "verify rows are finite and not constant", frame, 0u, 0u);
		Require(FirstDifference(verified, decoded, G::kHidden) == UINT32_MAX, model,
			"every verify row equals its one-row decode bit for bit (frame rows, first differing row)", frame, FirstDifference(verified, decoded, G::kHidden), 0u);
		Require(rig.KvMatches(after_verify, after_decode, UINT32_MAX) != 0u, model,
			"the KV the verify frame leaves equals the KV of the one-row decodes at every position", frame, 0u, 0u);
		Require(rig.RecurrentMatches(after_verify, prefix) != 0u, model, "a verify frame leaves the recurrent state untouched", frame, 0u, 0u);
		for ( uint32_t index = 0u; index < kept.size(); ++index )
		{
			const uint32_t k = kept[index], limit = VF_PREFIX + k + 2u, counts[1] = { k };
			std::vector<uint16_t> speculated, plain;
			rig.Load(after_verify);
			rig.last = frame_step;
			Require(rig.Keep(counts) != 0, model, "keep_rows folds the kept rows", frame, k, 0u);
			Require(rig.RecurrentMatches(rig.Save(), before[index]) != 0u, model,
				"after keeping k rows the recurrent state equals k one-row decodes (frame rows, k)", frame, k, 0u);
			Require(rig.Decode(0u, VF_PREFIX + k, 1u, &speculated) != 0, model, "the decode after a rejected tail runs", frame, k, 0u);
			Require(rig.Decode(0u, VF_PREFIX + k + 1u, 2u, &speculated) != 0, model, "the second decode after a rejected tail runs", frame, k, 0u);
			const std::vector<uint8_t> speculated_kv = rig.Save();
			rig.Load(before[index]);
			Require(rig.Decode(0u, VF_PREFIX + k, 1u, &plain) != 0 && rig.Decode(0u, VF_PREFIX + k + 1u, 2u, &plain) != 0, model,
				"the decodes of the run that never saw the draft run", frame, k, 0u);
			Require(FirstDifference(speculated, plain, G::kHidden) == UINT32_MAX, model,
				"after keeping k rows the next two decodes equal a run that never saw the draft (frame rows, k, first differing step)", frame, k,
				FirstDifference(speculated, plain, G::kHidden));
			Require(rig.KvMatches(speculated_kv, rig.Save(), limit) != 0u && rig.RecurrentMatches(speculated_kv, rig.Save()) != 0u, model,
				"after keeping k rows the KV and recurrent state equal a run that never saw the draft (frame rows, k)", frame, k, 0u);
			checked++;
		}
	}
	{
		std::vector<uint16_t> decoded, verified;
		std::vector<uint8_t> after_decode;
		const uint32_t first = 7u, second = 16u, counts[2] = { first, second };
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
		Require(rig.Run(rows.data(), first + second, 0u, &verified) != 0 && rig.Keep(counts) != 0, model, "a verify frame over two sequences runs and keeps",
			first, second, 0u);
		Require(FirstDifference(verified, decoded, G::kHidden) == UINT32_MAX, model,
			"every row of a two-sequence verify frame equals its one-row decode (first differing row)", FirstDifference(verified, decoded, G::kHidden), 0u, 0u);
		Require(rig.KvMatches(rig.Save(), after_decode, UINT32_MAX) != 0u && rig.RecurrentMatches(rig.Save(), after_decode) != 0u, model,
			"a two-sequence verify frame, every row kept, leaves the decodes' KV and recurrent state", 0u, 0u, 0u);
	}
	CheckTree<G>(rig, prefix);
	Require(rig.KvClean() != 0u, model, "no step reported a KV access error", 0u, 0u, 0u);
	printf("%s %s: frames of 1-%u rows, %u rejection points, a tree frame with two branches\n", vf_failures == 0 ? "ok" : "checked", model, VF_MAX_FRAME, checked);
	rig.Close();
}

int main(void)
{
	Check<SmallSolar>();
	Check<SmallLing3>();
	if ( vf_failures != 0 )
	{
		printf("FAIL ling verify frames: %d failures\n", vf_failures);
		return 1;
	}
	printf("PASS ling verify frames: every row of every verify frame equals one-row decode bit for bit, keep_rows restores the recurrent state of the kept rows, a rejected tail leaves no trace, a tree frame's trunk keeps its chain bits and its branches verify and fold as their own chains\n");
	return 0;
}
