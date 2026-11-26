#include "modules/solar_open2_tp16_resident_decode_stage/source/spark_solar_open2_tp16_stage_model.cu"
#include "modules/ling3_resident_decode_stage/source/spark_ling3_stage_model.cu"
#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_speculation_tree_frame.h"

#include <pthread.h>
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
static_assert(LingShape<Ling31FlashGeometry<16u>>::kAttnLayers == 9u && LingShape<Ling31FlashGeometry<16u>>::kKdaLayers == 63u,
	"Ling 3.1 Flash has 9 MLA and 63 KDA layers over TP16");

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
	static constexpr bool kFp8Weights = false;
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
	static constexpr bool kFp8Weights = false;
	static float RoutedLimit(uint32_t layer) { return layer == 4u ? 4.0f : 0.0f; }
	static float SharedLimit(uint32_t layer) { return layer == 3u ? 5.0f : 0.0f; }
	static const char *Tag(void) { return "ling-3.0-flash"; }
};

struct SmallLing32Fp8 : SmallLing3
{
	static constexpr bool kFp8Weights = true;
	static const char *Tag(void) { return "ling-3.2-air"; }
};

struct SmallLing31Flash : Ling31FlashGeometry<16u>
{
	static constexpr uint32_t kTpDegree = 1u;
	static constexpr uint32_t kHidden = 2304u;
	static constexpr uint32_t kLayers = 8u;
	static constexpr uint32_t kDenseLayers = 1u;
	static constexpr uint32_t kKdaHeads = 2u;
	static constexpr uint32_t kMlaHeads = 2u;
	static constexpr uint32_t kQLora = 256u;
	static constexpr uint32_t kExperts = 32u;
	static constexpr uint32_t kLocalExperts = 16u;
	static constexpr uint32_t kMoeInter = 128u;
	static constexpr uint32_t kSharedInter = 64u;
	static constexpr uint32_t kSharedRows = 64u;
	static constexpr bool kFp8Weights = true;
	static float RoutedLimit(uint32_t layer) { return layer == 6u ? 10.0f : 0.0f; }
	static float SharedLimit(uint32_t layer) { return layer == 5u ? 10.0f : 0.0f; }
	static const char *Tag(void) { return "ling-3.1-flash"; }
};

static LingWeight Bf16Weight(uint64_t rows, uint64_t columns)
{
	LingWeight weight;
	weight.weight = Matrix(rows, columns);
	weight.scale = 0;
	return weight;
}

static LingWeight Fp8Weight(uint64_t rows, uint64_t columns)
{
	const uint64_t payload = rows * columns, scales = rows * (columns / LG_FP8_BLOCK);
	const float base = 1.0f / sqrtf((float)columns);
	std::vector<uint8_t> w_host(payload);
	std::vector<float> s_host(scales);
	const uint32_t seed = vf_seed++;
	LingWeight weight;
	for ( uint64_t index = 0u; index < payload; ++index )
		w_host[index] = E4m3(Uniform(seed, (uint32_t)index));
	for ( uint64_t index = 0u; index < scales; ++index )
		s_host[index] = base * (0.75f + 0.5f * ((float)(Mix(seed + (uint32_t)index) & 255u) / 255.0f));
	weight.weight = Device(w_host.data(), w_host.size());
	weight.scale = (const float *)Device(s_host.data(), s_host.size() * sizeof(float));
	return weight;
}

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

	static LingWeight Proj(uint64_t rows, uint64_t columns)
	{
		return G::kFp8Weights ? Fp8Weight(rows, columns) : Bf16Weight(rows, columns);
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
				if constexpr ( G::kQLora != 0u )
				{
					layer->mla_q_a = Proj(G::kQLora, G::kHidden);
					layer->mla_q_norm = Norm(G::kQLora);
					layer->mla_q_b = Proj((uint64_t)S::kMlaRankHeads * S::kQkDim, G::kQLora);
				}
				else
					layer->mla_q = Proj((uint64_t)S::kMlaRankHeads * S::kQkDim, G::kHidden);
				layer->mla_kv_a = Proj(S::kSlot, G::kHidden);
				layer->mla_kv_norm = Norm(G::kLatent);
				layer->mla_uk = Tensor<uint16_t>((uint64_t)S::kMlaRankHeads * G::kLatent * G::kNope, 1.7f / sqrtf((float)G::kNope), 0.0f);
				layer->mla_uv = Tensor<uint16_t>((uint64_t)S::kMlaRankHeads * G::kValueDim * G::kLatent, 1.7f / sqrtf((float)G::kLatent), 0.0f);
				layer->mla_gate = Matrix(S::kMlaRankHeads, G::kHidden);
				layer->mla_out = Proj(G::kHidden, (uint64_t)S::kMlaRankHeads * G::kValueDim);
			}
		}
		else
		{
			layer->kda_qkv = Proj(S::kKdaChannels, G::kHidden);
			if ( G::kKdaRank != 0u )
			{
				layer->kda_f_a = Bf16Weight(G::kKdaRank, G::kHidden);
				layer->kda_g_a = Bf16Weight(G::kKdaRank, G::kHidden);
			}
			layer->kda_f = Proj(S::kKdaWidth, G::kKdaRank != 0u ? G::kKdaRank : G::kHidden);
			layer->kda_g = Proj(S::kKdaWidth, G::kKdaRank != 0u ? G::kKdaRank : G::kHidden);
			layer->kda_beta = Matrix(S::kKdaRankHeads, G::kHidden);
			layer->kda_conv = Tensor<uint16_t>((uint64_t)S::kKdaChannels * G::kConv, 0.5f, 0.0f);
			layer->kda_a_log = Tensor<float>(S::kKdaRankHeads, 0.5f, 0.0f);
			layer->kda_dt_bias = Tensor<float>(S::kKdaWidth, 0.5f, 0.0f);
			layer->kda_norm = Tensor<uint16_t>(G::kKdaDim, 0.2f, 1.0f);
			layer->kda_out = Proj(G::kHidden, S::kKdaWidth);
		}
		if ( index < G::kDenseLayers )
		{
			layer->ffn_gate_up = Proj(2u * VF_FFN, G::kHidden);
			layer->ffn_down = Proj(G::kHidden, VF_FFN);
		}
		else
		{
			layer->router = Matrix(G::kExperts, G::kHidden);
			layer->expert_bias = Tensor<float>(G::kExperts, 0.05f, 0.0f);
			layer->moe_w1 = Proj((uint64_t)G::kLocalExperts * G::kMoeInter, G::kHidden);
			layer->moe_w3 = Proj((uint64_t)G::kLocalExperts * G::kMoeInter, G::kHidden);
			layer->moe_w2 = Proj((uint64_t)G::kLocalExperts * G::kHidden, G::kMoeInter);
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

	int Run(const VfRow *rows, uint32_t count, uint32_t commit, std::vector<uint16_t> *output, uint32_t last_rows_only = 0u, uint32_t gather_context = 0u)
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
		step.last_rows_only = last_rows_only;
		step.gather_context = gather_context;
		step.host_positions = position.data();
		step.host_sequence_of_row = sequence.data();
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
		if ( cudaMemcpy(&error, state->kv_error, sizeof(error), cudaMemcpyDeviceToHost) != cudaSuccess )
			return 0u;
		if ( memcmp(&error, &zero, sizeof(error)) != 0 )
			printf("kv access error: code %u access %u row %u sequence %u position %u page %u\n", error.error_code, error.access_kind, error.row, error.sequence,
				error.position, error.page);
		return memcmp(&error, &zero, sizeof(error)) == 0 ? 1u : 0u;
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

static float RelativeRowDistance(const std::vector<uint16_t> &a, uint32_t ra, const std::vector<uint16_t> &b, uint32_t rb, uint32_t width)
{
	double diff = 0.0, norm = 0.0;
	for ( uint32_t c = 0u; c < width; ++c )
	{
		const uint32_t ua = (uint32_t)a[(uint64_t)ra * width + c] << 16u;
		const uint32_t ub = (uint32_t)b[(uint64_t)rb * width + c] << 16u;
		float fa = 0.0f, fb = 0.0f;
		memcpy(&fa, &ua, sizeof(float));
		memcpy(&fb, &ub, sizeof(float));
		const double d = (double)fa - (double)fb;
		diff += d * d;
		norm += (double)fb * (double)fb;
	}
	return norm > 0.0 ? (float)sqrt(diff / norm) : (float)sqrt(diff);
}

template<class G>
static void CheckPrefixExact(void)
{
	static const uint32_t deltas[] = { 1u, 4u, 8u, 12u, 16u, 23u, 64u, 128u };
	static constexpr uint32_t kCached = 2u * G::kPageSlots;
	static constexpr uint32_t kMaxTotal = kCached + 128u;
	VfRig<G> rig;
	const char *model = G::Tag();
	std::vector<VfRow> all_rows(kMaxTotal);
	std::vector<uint8_t> cached_state, zero_state;
	if ( rig.Open() == 0 )
	{
		Require(0, model, "prefix-exact synthetic stage opens", 0u, 0u, 0u);
		return;
	}
	zero_state = rig.Save();
	for ( uint32_t position = 0u; position < kMaxTotal; ++position )
		all_rows[position] = { 0u, position, 0u };
	rig.Load(zero_state);
	Require(rig.Run(all_rows.data(), kCached, 1u, 0, 1u, kCached) != 0, model, "cached prefix prefill runs", kCached, 0u, 0u);
	cached_state = rig.Save();
	for ( uint32_t delta : deltas )
	{
		const uint32_t total = kCached + delta;
		std::vector<uint16_t> cold_out, warm_out;
		std::vector<uint8_t> cold_state;
		rig.Load(zero_state);
		Require(rig.Run(all_rows.data(), total, 1u, &cold_out, 1u, total) != 0, model, "cold prefill runs", total, delta, 0u);
		cold_state = rig.Save();
		Require(Finite(cold_out) != 0u, model, "cold prefill rows are finite", total, delta, 0u);
		std::vector<uint16_t> cold_tail(cold_out.begin() + (ptrdiff_t)((uint64_t)kCached * G::kHidden), cold_out.end());
		rig.Load(cached_state);
		Require(rig.Run(all_rows.data() + kCached, delta, 1u, &warm_out, 1u, total) != 0, model, "warm delta prefill runs", kCached, delta, 0u);
		Require(FirstDifference(warm_out, cold_tail, G::kHidden) == UINT32_MAX, model,
			"warm delta prefill equals cold prefill bit for bit (delta, first differing row)", delta, FirstDifference(warm_out, cold_tail, G::kHidden), 0u);
		const std::vector<uint8_t> warm_state = rig.Save();
		Require(rig.KvMatches(warm_state, cold_state, total) != 0u && rig.RecurrentMatches(warm_state, cold_state) != 0u, model,
			"warm delta prefill leaves the same KV and recurrent state as cold prefill bit for bit (delta)", delta, 0u, 0u);
	}
	{
		std::vector<uint16_t> ref_out, pref_out;
		const uint32_t check_len = 72u;
		rig.Load(zero_state);
		Require(rig.Run(all_rows.data(), check_len, 1u, &ref_out, 1u, 0u) != 0, model, "reference scatter prefill runs", check_len, 0u, 0u);
		rig.Load(zero_state);
		Require(rig.Run(all_rows.data(), check_len, 1u, &pref_out, 1u, check_len) != 0, model, "KV-gather local prefill runs", check_len, 0u, 0u);
		uint32_t moe_flips = 0u;
		for ( uint32_t p = 0u; p < check_len; ++p )
		{
			const float dist = RelativeRowDistance(pref_out, p, ref_out, p, G::kHidden);
			if ( dist >= 0.02f )
				moe_flips++;
		}
		Require(moe_flips <= 2u, model, "KV-gather local prefill matches reference scatter prefill modulo synthetic router ties (flips)", moe_flips, 0u, 0u);
	}
	{
		static const uint32_t b8_peer_lens[] = { 1u, 9u, 20u };
		std::vector<uint16_t> b1_pf, bn_pf;
		const uint32_t lane0_len = 50u;
		rig.Load(zero_state);
		Require(rig.Run(all_rows.data(), lane0_len, 1u, &b1_pf, 1u, lane0_len) != 0, model, "B1 prefill for 5c runs", lane0_len, 0u, 0u);
		for ( uint32_t peer_len : b8_peer_lens )
		{
			std::vector<VfRow> multi;
			for ( uint32_t p = 0u; p < lane0_len; ++p )
				multi.push_back({ 0u, p, 0u });
			for ( uint32_t s = 1u; s < VF_SLOTS; ++s )
				for ( uint32_t p = 0u; p < peer_len; ++p )
					multi.push_back({ s, p, 10u + s });
			rig.Load(zero_state);
			Require(rig.Run(multi.data(), (uint32_t)multi.size(), 1u, &bn_pf, 1u, 0u) != 0, model, "multi-seq prefill runs", peer_len, (uint32_t)multi.size(), 0u);
			std::vector<uint16_t> lane0_out(bn_pf.begin(), bn_pf.begin() + (ptrdiff_t)((uint64_t)lane0_len * G::kHidden));
			Require(FirstDifference(lane0_out, b1_pf, G::kHidden) == UINT32_MAX, model,
				"multi-seq prefill lane0 equals B1 prefill bit for bit (peer_len, first differing row)", peer_len, FirstDifference(lane0_out, b1_pf, G::kHidden), 0u);
		}
		std::vector<uint16_t> b1_warm_delta, b8_warm_delta;
		const uint32_t warm_delta = 5u;
		rig.Load(cached_state);
		Require(rig.Run(all_rows.data() + kCached, warm_delta, 1u, &b1_warm_delta, 1u, kCached + warm_delta) != 0, model,
			"B1 warm delta prefill for 5c runs", warm_delta, 0u, 0u);
		std::vector<VfRow> warm_multi;
		for ( uint32_t p = 0u; p < warm_delta; ++p )
			warm_multi.push_back({ 0u, kCached + p, 0u });
		for ( uint32_t s = 1u; s < VF_SLOTS; ++s )
			warm_multi.push_back({ s, 0u, 30u + s });
		rig.Load(cached_state);
		Require(rig.Run(warm_multi.data(), (uint32_t)warm_multi.size(), 1u, &b8_warm_delta, 1u, 0u) != 0, model,
			"multi-seq warm delta prefill runs", (uint32_t)warm_multi.size(), 0u, 0u);
		std::vector<uint16_t> warm_lane0(b8_warm_delta.begin(), b8_warm_delta.begin() + (ptrdiff_t)((uint64_t)warm_delta * G::kHidden));
		Require(FirstDifference(warm_lane0, b1_warm_delta, G::kHidden) == UINT32_MAX, model,
			"multi-seq warm prefill lane0 equals B1 warm prefill bit for bit (first differing row)", FirstDifference(warm_lane0, b1_warm_delta, G::kHidden), 0u, 0u);
	}
	{
		static const uint32_t batches[] = { 4u, 8u, 16u };
		std::vector<uint16_t> b1_out, bn_out;
		rig.Load(cached_state);
		VfRow b1_row = { 0u, kCached, 5u };
		Require(rig.Run(&b1_row, 1u, 1u, &b1_out, 0u, 0u) != 0, model, "B1 decode runs", 0u, 0u, 0u);
		for ( uint32_t bs : batches )
		{
			std::vector<VfRow> bn_rows(bs);
			bn_rows[0] = b1_row;
			for ( uint32_t r = 1u; r < bs; ++r )
				bn_rows[r] = { 1u, r - 1u, 20u + r };
			rig.Load(cached_state);
			Require(rig.Run(bn_rows.data(), bs, 1u, &bn_out, 0u, 0u) != 0, model, "batched decode runs", bs, 0u, 0u);
			Require(memcmp(b1_out.data(), bn_out.data(), G::kHidden * sizeof(uint16_t)) == 0, model,
				"batched decode lane0 equals B1 decode bit for bit (batch)", bs, 0u, 0u);
		}
	}
	Require(rig.KvClean() != 0u, model, "prefix-exact steps reported no KV access error", 0u, 0u, 0u);
	printf("%s %s prefix-exact + B1==B4==B8==B16: cached=%u vs warm deltas 1..128\n", vf_failures == 0 ? "ok" : "checked", model, kCached);
	rig.Close();
}

struct SmallLing31Tp4 : Ling31FlashGeometry<4u>
{
	static constexpr uint32_t kHidden = 2304u;
	static constexpr uint32_t kLayers = 8u;
	static constexpr uint32_t kDenseLayers = 8u;
	static constexpr uint32_t kKdaHeads = 8u;
	static constexpr uint32_t kMlaHeads = 8u;
	static constexpr uint32_t kQLora = 256u;
	static constexpr uint32_t kExperts = 32u;
	static constexpr uint32_t kLocalExperts = 8u;
	static constexpr uint32_t kMoeInter = 128u;
	static constexpr uint32_t kSharedInter = 256u;
	static constexpr uint32_t kSharedRows = 64u;
	static constexpr bool kFp8Weights = true;
	static const char *Tag(void) { return "ling-3.1-flash-tp4"; }
};

struct LingTp4Barrier
{
	pthread_mutex_t mutex;
	pthread_cond_t cond;
	uint32_t parties, waiting, generation, aborted;
};

static int LingTp4Wait(LingTp4Barrier *b)
{
	pthread_mutex_lock(&b->mutex);
	const uint32_t gen = b->generation;
	if ( b->aborted == 0u && ++b->waiting == b->parties )
	{
		b->waiting = 0u;
		b->generation++;
		pthread_cond_broadcast(&b->cond);
	}
	else
		while ( b->aborted == 0u && gen == b->generation )
			pthread_cond_wait(&b->cond, &b->mutex);
	const int ok = b->aborted == 0u;
	pthread_mutex_unlock(&b->mutex);
	return ok;
}

__global__ static void LingTp4SumKernel(const uint8_t *staging, uint64_t pitch, uint32_t ranks, uint16_t *out, uint64_t count)
{
	for ( uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < count; idx += (uint64_t)gridDim.x * blockDim.x )
	{
		float total = 0.0f;
		for ( uint32_t r = 0u; r < ranks; ++r )
			total += LmBf16ToFloat(((const uint16_t *)(staging + (uint64_t)r * pitch))[idx]);
		out[idx] = LmFloatToBf16(total);
	}
}

struct LingTp4Rig
{
	using G = SmallLing31Tp4;
	using M = LingModel<G>;
	using S = LingShape<G>;
	using State = LingState<G>;
	static constexpr uint32_t kRanks = 4u;
	static constexpr uint64_t kRankKv = (uint64_t)VF_SLOTS * VF_PAGES * (S::Kv::kPageBytes / kRanks);

	struct Rank
	{
		LingTp4Rig *rig;
		uint32_t index;
		State *state;
		SparkStageRunnerServices services;
		cudaStream_t stream;
		uint16_t *hidden;
		uint32_t *positions, *context, *sequence_of_row;
		SparkStageRunnerStep step;
		SparkStatus status;
	};

	Rank ranks[kRanks];
	LingTp4Barrier barrier;
	uint8_t *kv, *staging;
	uint64_t staging_bytes;
	uint32_t *page_table;
	uint32_t multiprocessors;
	uint32_t host_context[VF_SLOTS];

	static SparkStatus Round(void *context, void *stream_ptr, uint32_t op, uint32_t stage, uint32_t rows, uint32_t row_elements,
		const void *local, void *full)
	{
		Rank *rank = (Rank *)context;
		LingTp4Rig *rig = rank->rig;
		cudaStream_t stream = (cudaStream_t)stream_ptr;
		const uint64_t block = (uint64_t)rows * (row_elements != 0u ? row_elements : G::kHidden) * sizeof(uint16_t);
		uint8_t *slot = rig->staging + (uint64_t)rank->index * rig->staging_bytes;
		if ( stage != SPARK_STAGE_RUNNER_ROUND_ALL )
			return SPARK_STATUS_UNSUPPORTED;
		const uint64_t copy = op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL ? (uint64_t)kRanks * block : block;
		if ( copy > rig->staging_bytes || cudaMemcpyAsync(slot, local, copy, cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
			cudaStreamSynchronize(stream) != cudaSuccess )
			return SPARK_STATUS_IO_ERROR;
		if ( LingTp4Wait(&rig->barrier) == 0 )
			return SPARK_STATUS_IO_ERROR;
		if ( op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER )
		{
			for ( uint32_t s = 0u; s < kRanks; ++s )
				if ( cudaMemcpyAsync((uint8_t *)full + (uint64_t)s * block, rig->staging + (uint64_t)s * rig->staging_bytes, block, cudaMemcpyDeviceToDevice, stream) != cudaSuccess )
					return SPARK_STATUS_IO_ERROR;
		}
		else if ( op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL )
		{
			for ( uint32_t s = 0u; s < kRanks; ++s )
				if ( cudaMemcpyAsync((uint8_t *)full + (uint64_t)s * block, rig->staging + (uint64_t)s * rig->staging_bytes + (uint64_t)rank->index * block, block, cudaMemcpyDeviceToDevice, stream) != cudaSuccess )
					return SPARK_STATUS_IO_ERROR;
		}
		else if ( op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16 )
		{
			const uint64_t elems = block / sizeof(uint16_t);
			LingTp4SumKernel<<<(uint32_t)((elems + 255u) / 256u), 256u, 0, stream>>>(rig->staging, rig->staging_bytes, kRanks, (uint16_t *)full, elems);
		}
		if ( cudaStreamSynchronize(stream) != cudaSuccess || LingTp4Wait(&rig->barrier) == 0 )
			return SPARK_STATUS_IO_ERROR;
		return SPARK_STATUS_OK;
	}

	static void *Thread(void *arg)
	{
		Rank *rank = (Rank *)arg;
		typename M::Frame frame = { &rank->step, rank->stream };
		const LingLayer &layer = rank->state->layers[G::kPhase];
		SparkStatus st = M::RmsNorm(rank->state, frame, layer.input_norm);
		if ( st == SPARK_STATUS_OK )
			st = M::Attention(rank->state, frame, layer);
		if ( st == SPARK_STATUS_OK )
			st = M::AllReduce(rank->state, frame);
		if ( st == SPARK_STATUS_OK )
			st = M::AddRows(rank->state, frame);
		if ( st == SPARK_STATUS_OK && cudaStreamSynchronize(rank->stream) != cudaSuccess )
			st = SPARK_STATUS_IO_ERROR;
		rank->status = st;
		return 0;
	}

	int Open(void)
	{
		std::vector<uint32_t> table(VF_SLOTS * VF_PAGES);
		memset(this, 0, sizeof(*this));
		barrier.parties = kRanks;
		if ( pthread_mutex_init(&barrier.mutex, 0) != 0 || pthread_cond_init(&barrier.cond, 0) != 0 ||
			cudaDeviceGetAttribute((int *)&multiprocessors, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess ||
			cudaMalloc((void **)&kv, kRanks * kRankKv) != cudaSuccess || cudaMemset(kv, 0, kRanks * kRankKv) != cudaSuccess ||
			cudaMalloc((void **)&page_table, table.size() * sizeof(uint32_t)) != cudaSuccess )
			return 0;
		for ( uint32_t i = 0u; i < table.size(); ++i )
			table[i] = i;
		if ( cudaMemcpy(page_table, table.data(), table.size() * sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess )
			return 0;
		const float *in_norm = Norm(G::kHidden);
		const LingWeight q_a = Fp8Weight(G::kQLora, G::kHidden);
		const float *q_norm = Norm(G::kQLora);
		const LingWeight kv_a = Fp8Weight(S::kSlot, G::kHidden);
		const float *kv_norm = Norm(G::kLatent);
		for ( uint32_t r = 0u; r < kRanks; ++r )
		{
			Rank *rk = &ranks[r];
			rk->rig = this;
			rk->index = r;
			rk->services.context = rk;
			rk->services.round = Round;
			rk->state = (State *)calloc(1u, sizeof(State));
			rk->state->tp_degree = kRanks;
			rk->state->tp_rank = r;
			rk->state->max_rows = VF_MAX_FRAME;
			rk->state->slots = VF_SLOTS;
			rk->state->ffn_rows = VF_FFN;
			rk->state->expert_first = r * G::kLocalExperts;
			rk->state->services = &rk->services;
			M::Shape(rk->state);
			rk->state->arena_bytes = M::Plan(rk->state, 0, 0u);
			if ( cudaStreamCreate(&rk->stream) != cudaSuccess ||
				cudaMalloc((void **)&rk->state->arena, rk->state->arena_bytes) != cudaSuccess ||
				cudaMemset(rk->state->arena, 0, rk->state->arena_bytes) != cudaSuccess ||
				cudaMalloc((void **)&rk->hidden, (uint64_t)VF_MAX_FRAME * G::kHidden * sizeof(uint16_t)) != cudaSuccess ||
				cudaMalloc((void **)&rk->positions, VF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
				cudaMalloc((void **)&rk->sequence_of_row, VF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
				cudaMalloc((void **)&rk->context, VF_SLOTS * sizeof(uint32_t)) != cudaSuccess )
				return 0;
			M::Plan(rk->state, (uint8_t *)rk->state->arena, 0u);
			M::Arrange(rk->state);
			LingLayer *ly = &rk->state->layers[G::kPhase];
			ly->input_norm = in_norm;
			ly->mla_q_a = q_a;
			ly->mla_q_norm = q_norm;
			ly->mla_q_b = Fp8Weight((uint64_t)S::kMlaRankHeads * S::kQkDim, G::kQLora);
			ly->mla_kv_a = kv_a;
			ly->mla_kv_norm = kv_norm;
			ly->mla_uk = Tensor<uint16_t>((uint64_t)S::kMlaRankHeads * G::kLatent * G::kNope, 1.7f / sqrtf((float)G::kNope), 0.0f);
			ly->mla_uv = Tensor<uint16_t>((uint64_t)S::kMlaRankHeads * G::kValueDim * G::kLatent, 1.7f / sqrtf((float)G::kLatent), 0.0f);
			ly->mla_gate = Matrix(S::kMlaRankHeads, G::kHidden);
			ly->mla_out = Fp8Weight(G::kHidden, (uint64_t)S::kMlaRankHeads * G::kValueDim);
			SparkStageRunnerKv attach;
			memset(&attach, 0, sizeof(attach));
			attach.pool = kv + (uint64_t)r * kRankKv;
			attach.layer_stride_bytes = kRankKv;
			attach.layer_page_bytes = S::Kv::kPageBytes;
			attach.layer_count = 1u;
			attach.page_table = page_table;
			attach.page_table_stride = VF_PAGES;
			attach.pool_page_count = VF_SLOTS * VF_PAGES;
			attach.sequence_count = VF_SLOTS;
			attach.context_shard.degree = kRanks;
			attach.context_shard.rank = r;
			attach.context_shard.grain = 1u;
			if ( M::AttachKv(rk->state, &attach) != SPARK_STATUS_OK )
				return 0;
		}
		staging_bytes = (uint64_t)kRanks * ranks[0].state->partial_stride * sizeof(float);
		const uint64_t ag_cap = (uint64_t)VF_MAX_FRAME * S::kMlaRankHeads * S::kSlot * sizeof(uint16_t);
		if ( ag_cap > staging_bytes )
			staging_bytes = ag_cap;
		return cudaMalloc((void **)&staging, kRanks * staging_bytes) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
	}

	int Run(const VfRow *rows, uint32_t count, uint32_t gather_ctx, std::vector<uint16_t> *out)
	{
		std::vector<uint16_t> input((uint64_t)count * G::kHidden), res(input.size()), other(input.size());
		std::vector<uint32_t> pos(count), seq(count, 0u);
		uint32_t seq_count = 0u, prev_seq = UINT32_MAX;
		pthread_t threads[kRanks];
		for ( uint32_t i = 0u; i < count; ++i )
		{
			const uint32_t seed = Mix(rows[i].sequence * 7919u + rows[i].variant * 104729u) ^ rows[i].position;
			for ( uint32_t c = 0u; c < G::kHidden; ++c )
				input[(uint64_t)i * G::kHidden + c] = Bf16(Uniform(seed, c));
			pos[i] = rows[i].position;
			seq[i] = rows[i].sequence;
			if ( rows[i].sequence != prev_seq )
			{
				prev_seq = rows[i].sequence;
				seq_count++;
			}
			host_context[rows[i].sequence] = rows[i].position + 1u;
		}
		for ( uint32_t r = 0u; r < kRanks; ++r )
		{
			Rank *rk = &ranks[r];
			memset(&rk->step, 0, sizeof(rk->step));
			rk->step.hidden_bf16 = rk->hidden;
			rk->step.positions = rk->positions;
			rk->step.context_length = rk->context;
			rk->step.sequence_of_row = rk->sequence_of_row;
			rk->step.rows = count;
			rk->step.sequences = seq_count;
			rk->step.commit = 1u;
			rk->step.last_rows_only = 1u;
			rk->step.gather_context = gather_ctx;
			rk->step.host_positions = pos.data();
			rk->step.host_sequence_of_row = seq.data();
			rk->step.context = VF_PAGES * G::kPageSlots;
			rk->step.multiprocessors = multiprocessors;
			rk->step.route_fault = VfRouteFault();
			if ( cudaMemcpyAsync(rk->hidden, input.data(), input.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, rk->stream) != cudaSuccess ||
				cudaMemcpyAsync(rk->positions, pos.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice, rk->stream) != cudaSuccess ||
				cudaMemcpyAsync(rk->sequence_of_row, seq.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice, rk->stream) != cudaSuccess ||
				cudaMemcpyAsync(rk->context, host_context, sizeof(host_context), cudaMemcpyHostToDevice, rk->stream) != cudaSuccess )
				return 0;
		}
		barrier.waiting = 0u;
		barrier.aborted = 0u;
		for ( uint32_t r = 0u; r < kRanks; ++r )
			if ( pthread_create(&threads[r], 0, Thread, &ranks[r]) != 0 )
				return 0;
		int ok = 1;
		for ( uint32_t r = 0u; r < kRanks; ++r )
		{
			pthread_join(threads[r], 0);
			ok = ok && ranks[r].status == SPARK_STATUS_OK;
		}
		if ( !ok || cudaMemcpy(res.data(), ranks[0].hidden, res.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess )
			return 0;
		for ( uint32_t r = 1u; r < kRanks; ++r )
		{
			if ( cudaMemcpy(other.data(), ranks[r].hidden, other.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess )
				return 0;
			Require(memcmp(other.data(), res.data(), other.size() * sizeof(uint16_t)) == 0, G::Tag(), "all TP4 ranks agree on MLA output (rank)", r, 0u, 0u);
		}
		if ( out != 0 )
			*out = res;
		return 1;
	}

	void Close(void)
	{
		for ( uint32_t r = 0u; r < kRanks; ++r )
		{
			if ( ranks[r].state != 0 )
			{
				cudaFree(ranks[r].state->arena);
				free(ranks[r].state);
			}
			cudaFree(ranks[r].hidden);
			cudaFree(ranks[r].positions);
			cudaFree(ranks[r].sequence_of_row);
			cudaFree(ranks[r].context);
			if ( ranks[r].stream != 0 )
				cudaStreamDestroy(ranks[r].stream);
		}
		cudaFree(staging);
		cudaFree(page_table);
		cudaFree(kv);
		pthread_cond_destroy(&barrier.cond);
		pthread_mutex_destroy(&barrier.mutex);
	}
};

static void CheckMlaTp4(void)
{
	static const uint32_t deltas[] = { 1u, 4u, 8u, 16u, 23u, 64u, 128u };
	static constexpr uint32_t kCached = 2u * SmallLing31Tp4::kPageSlots;
	static constexpr uint32_t kMaxTotal = kCached + 128u;
	const char *model = SmallLing31Tp4::Tag();
	LingTp4Rig rig;
	if ( rig.Open() == 0 )
	{
		Require(0, model, "TP4 MLA synthetic stage opens", 0u, 0u, 0u);
		return;
	}
	std::vector<VfRow> all_rows(kMaxTotal);
	for ( uint32_t p = 0u; p < kMaxTotal; ++p )
		all_rows[p] = { 0u, p, 0u };
	std::vector<uint8_t> cached_kv(rig.kRanks * rig.kRankKv), cold_kv(rig.kRanks * rig.kRankKv), warm_kv(rig.kRanks * rig.kRankKv);
	Require(rig.Run(all_rows.data(), kCached, kCached, 0) != 0, model, "TP4 cached prefix runs", kCached, 0u, 0u);
	Require(cudaMemcpy(cached_kv.data(), rig.kv, cached_kv.size(), cudaMemcpyDeviceToHost) == cudaSuccess, model, "TP4 save cached KV", 0u, 0u, 0u);
	for ( uint32_t delta : deltas )
	{
		const uint32_t total = kCached + delta;
		std::vector<uint16_t> cold_out, warm_out;
		Require(cudaMemset(rig.kv, 0, cached_kv.size()) == cudaSuccess, model, "TP4 reset KV", delta, 0u, 0u);
		Require(rig.Run(all_rows.data(), total, total, &cold_out) != 0, model, "TP4 cold MLA prefill runs", total, delta, 0u);
		Require(cudaMemcpy(cold_kv.data(), rig.kv, cold_kv.size(), cudaMemcpyDeviceToHost) == cudaSuccess, model, "TP4 save cold KV", delta, 0u, 0u);
		std::vector<uint16_t> cold_tail(cold_out.begin() + (ptrdiff_t)((uint64_t)kCached * SmallLing31Tp4::kHidden), cold_out.end());
		Require(cudaMemcpy(rig.kv, cached_kv.data(), cached_kv.size(), cudaMemcpyHostToDevice) == cudaSuccess, model, "TP4 restore cached KV", delta, 0u, 0u);
		Require(rig.Run(all_rows.data() + kCached, delta, total, &warm_out) != 0, model, "TP4 warm MLA delta prefill runs", kCached, delta, 0u);
		Require(cudaMemcpy(warm_kv.data(), rig.kv, warm_kv.size(), cudaMemcpyDeviceToHost) == cudaSuccess, model, "TP4 save warm KV", delta, 0u, 0u);
		Require(FirstDifference(warm_out, cold_tail, SmallLing31Tp4::kHidden) == UINT32_MAX, model,
			"TP4 warm MLA prefill equals cold bit for bit (delta, first differing row)", delta, FirstDifference(warm_out, cold_tail, SmallLing31Tp4::kHidden), 0u);
		Require(memcmp(warm_kv.data(), cold_kv.data(), warm_kv.size()) == 0, model,
			"TP4 warm MLA leaves identical sharded KV on all 4 ranks (delta)", delta, 0u, 0u);
	}
	{
		std::vector<uint16_t> ref_out, gather_out;
		const uint32_t delta = 64u, total = kCached + delta;
		Require(cudaMemcpy(rig.kv, cached_kv.data(), cached_kv.size(), cudaMemcpyHostToDevice) == cudaSuccess, model, "TP4 accuracy restore 1", 0u, 0u, 0u);
		Require(rig.Run(all_rows.data() + kCached, delta, 0u, &ref_out) != 0, model, "TP4 reference scatter MLA runs", delta, 0u, 0u);
		Require(cudaMemcpy(rig.kv, cached_kv.data(), cached_kv.size(), cudaMemcpyHostToDevice) == cudaSuccess, model, "TP4 accuracy restore 2", 0u, 0u, 0u);
		Require(rig.Run(all_rows.data() + kCached, delta, total, &gather_out) != 0, model, "TP4 KV-gather warm MLA runs", delta, 0u, 0u);
		for ( uint32_t p = 0u; p < delta; ++p )
		{
			const float dist = RelativeRowDistance(gather_out, p, ref_out, p, SmallLing31Tp4::kHidden);
			Require(dist < 0.01f, model, "TP4 warm KV-gather MLA matches reference scatter MLA (pos, rel x 1e6)", p, (uint32_t)(dist * 1.0e6f), 0u);
		}
	}
	printf("%s %s multi-rank MLA KV-gather prefix-exact + accuracy: 4 ranks, cached=%u vs warm deltas 1..128\n", vf_failures == 0 ? "ok" : "checked", model, kCached);
	rig.Close();
}

struct SmallSolarTp4 : SolarOpen2Geometry<4u>
{
	static constexpr uint32_t kHidden = 1536u;
	static constexpr uint32_t kLayers = 4u;
	static constexpr uint32_t kKdaHeads = 16u;
	static constexpr uint32_t kGqaHeads = 16u;
	static constexpr uint32_t kGqaKvHeads = 2u;
	static constexpr uint32_t kExperts = 16u;
	static constexpr uint32_t kLocalExperts = 16u;
	static constexpr uint32_t kMoeInter = 80u;
	static constexpr uint32_t kSharedRows = 128u;
	static const char *Tag(void) { return "solar-open-2-tp4"; }
};

struct LingGqaTp4Rig
{
	using G = SmallSolarTp4;
	using M = LingModel<G>;
	using S = LingShape<G>;
	using State = LingState<G>;
	static constexpr uint32_t kRanks = 4u;
	static constexpr uint64_t kRankKv = (uint64_t)VF_SLOTS * VF_PAGES * (S::Kv::kPageBytes / kRanks);

	struct Rank
	{
		LingGqaTp4Rig *rig;
		uint32_t index;
		State *state;
		SparkStageRunnerServices services;
		cudaStream_t stream;
		uint16_t *hidden;
		uint32_t *positions, *context, *sequence_of_row;
		SparkStageRunnerStep step;
		SparkStatus status;
	};

	Rank ranks[kRanks];
	LingTp4Barrier barrier;
	uint8_t *kv, *staging;
	uint64_t staging_bytes;
	uint32_t *page_table;
	uint32_t multiprocessors;
	uint32_t host_context[VF_SLOTS];

	static SparkStatus Round(void *context, void *stream_ptr, uint32_t op, uint32_t stage, uint32_t rows, uint32_t row_elements,
		const void *local, void *full)
	{
		Rank *rank = (Rank *)context;
		LingGqaTp4Rig *rig = rank->rig;
		cudaStream_t stream = (cudaStream_t)stream_ptr;
		const uint64_t block = (uint64_t)rows * (row_elements != 0u ? row_elements : G::kHidden) * sizeof(uint16_t);
		uint8_t *slot = rig->staging + (uint64_t)rank->index * rig->staging_bytes;
		if ( stage != SPARK_STAGE_RUNNER_ROUND_ALL )
			return SPARK_STATUS_UNSUPPORTED;
		const uint64_t copy = op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL ? (uint64_t)kRanks * block : block;
		if ( copy > rig->staging_bytes || cudaMemcpyAsync(slot, local, copy, cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
			cudaStreamSynchronize(stream) != cudaSuccess )
			return SPARK_STATUS_IO_ERROR;
		if ( LingTp4Wait(&rig->barrier) == 0 )
			return SPARK_STATUS_IO_ERROR;
		if ( op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER )
		{
			for ( uint32_t s = 0u; s < kRanks; ++s )
				if ( cudaMemcpyAsync((uint8_t *)full + (uint64_t)s * block, rig->staging + (uint64_t)s * rig->staging_bytes, block, cudaMemcpyDeviceToDevice, stream) != cudaSuccess )
					return SPARK_STATUS_IO_ERROR;
		}
		else if ( op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL )
		{
			for ( uint32_t s = 0u; s < kRanks; ++s )
				if ( cudaMemcpyAsync((uint8_t *)full + (uint64_t)s * block, rig->staging + (uint64_t)s * rig->staging_bytes + (uint64_t)rank->index * block, block, cudaMemcpyDeviceToDevice, stream) != cudaSuccess )
					return SPARK_STATUS_IO_ERROR;
		}
		else if ( op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16 )
		{
			const uint64_t elems = block / sizeof(uint16_t);
			LingTp4SumKernel<<<(uint32_t)((elems + 255u) / 256u), 256u, 0, stream>>>(rig->staging, rig->staging_bytes, kRanks, (uint16_t *)full, elems);
		}
		if ( cudaStreamSynchronize(stream) != cudaSuccess || LingTp4Wait(&rig->barrier) == 0 )
			return SPARK_STATUS_IO_ERROR;
		return SPARK_STATUS_OK;
	}

	static void *Thread(void *arg)
	{
		Rank *rank = (Rank *)arg;
		typename M::Frame frame = { &rank->step, rank->stream };
		const LingLayer &layer = rank->state->layers[G::kPhase];
		SparkStatus st = M::RmsNorm(rank->state, frame, layer.input_norm);
		if ( st == SPARK_STATUS_OK )
			st = M::Attention(rank->state, frame, layer);
		if ( st == SPARK_STATUS_OK )
			st = M::AllReduce(rank->state, frame);
		if ( st == SPARK_STATUS_OK )
			st = M::AddRows(rank->state, frame);
		if ( st == SPARK_STATUS_OK && cudaStreamSynchronize(rank->stream) != cudaSuccess )
			st = SPARK_STATUS_IO_ERROR;
		rank->status = st;
		return 0;
	}

	int Open(void)
	{
		std::vector<uint32_t> table(VF_SLOTS * VF_PAGES);
		memset(this, 0, sizeof(*this));
		barrier.parties = kRanks;
		if ( pthread_mutex_init(&barrier.mutex, 0) != 0 || pthread_cond_init(&barrier.cond, 0) != 0 ||
			cudaDeviceGetAttribute((int *)&multiprocessors, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess ||
			cudaMalloc((void **)&kv, kRanks * kRankKv) != cudaSuccess || cudaMemset(kv, 0, kRanks * kRankKv) != cudaSuccess ||
			cudaMalloc((void **)&page_table, table.size() * sizeof(uint32_t)) != cudaSuccess )
			return 0;
		for ( uint32_t i = 0u; i < table.size(); ++i )
			table[i] = i;
		if ( cudaMemcpy(page_table, table.data(), table.size() * sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess )
			return 0;
		const float *in_norm = Norm(G::kHidden);
		for ( uint32_t r = 0u; r < kRanks; ++r )
		{
			Rank *rk = &ranks[r];
			rk->rig = this;
			rk->index = r;
			rk->services.context = rk;
			rk->services.round = Round;
			rk->state = (State *)calloc(1u, sizeof(State));
			rk->state->tp_degree = kRanks;
			rk->state->tp_rank = r;
			rk->state->max_rows = VF_MAX_FRAME;
			rk->state->slots = VF_SLOTS;
			rk->state->ffn_rows = VF_FFN;
			rk->state->expert_first = 0u;
			rk->state->services = &rk->services;
			M::Shape(rk->state);
			rk->state->arena_bytes = M::Plan(rk->state, 0, 0u);
			if ( cudaStreamCreate(&rk->stream) != cudaSuccess ||
				cudaMalloc((void **)&rk->state->arena, rk->state->arena_bytes) != cudaSuccess ||
				cudaMemset(rk->state->arena, 0, rk->state->arena_bytes) != cudaSuccess ||
				cudaMalloc((void **)&rk->hidden, (uint64_t)VF_MAX_FRAME * G::kHidden * sizeof(uint16_t)) != cudaSuccess ||
				cudaMalloc((void **)&rk->positions, VF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
				cudaMalloc((void **)&rk->sequence_of_row, VF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
				cudaMalloc((void **)&rk->context, VF_SLOTS * sizeof(uint32_t)) != cudaSuccess )
				return 0;
			M::Plan(rk->state, (uint8_t *)rk->state->arena, 0u);
			M::Arrange(rk->state);
			LingLayer *ly = &rk->state->layers[G::kPhase];
			ly->input_norm = in_norm;
			ly->gqa_qkv = Matrix(S::kGqaRankFused, G::kHidden);
			ly->gqa_gate = Matrix(S::kGqaRankWidth, G::kHidden);
			ly->gqa_out = Matrix(G::kHidden, S::kGqaRankWidth);
			SparkStageRunnerKv attach;
			memset(&attach, 0, sizeof(attach));
			attach.pool = kv + (uint64_t)r * kRankKv;
			attach.layer_stride_bytes = kRankKv;
			attach.layer_page_bytes = S::Kv::kPageBytes;
			attach.layer_count = 1u;
			attach.page_table = page_table;
			attach.page_table_stride = VF_PAGES;
			attach.pool_page_count = VF_SLOTS * VF_PAGES;
			attach.sequence_count = VF_SLOTS;
			attach.context_shard.degree = kRanks;
			attach.context_shard.rank = r;
			attach.context_shard.grain = 1u;
			if ( M::AttachKv(rk->state, &attach) != SPARK_STATUS_OK )
				return 0;
		}
		staging_bytes = (uint64_t)kRanks * ranks[0].state->partial_stride * sizeof(float);
		const uint64_t ag_cap = (uint64_t)VF_MAX_FRAME * S::kGqaRankFused * sizeof(uint16_t);
		if ( ag_cap > staging_bytes )
			staging_bytes = ag_cap;
		return cudaMalloc((void **)&staging, kRanks * staging_bytes) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
	}

	int Run(const VfRow *rows, uint32_t count, uint32_t gather_ctx, std::vector<uint16_t> *out)
	{
		std::vector<uint16_t> input((uint64_t)count * G::kHidden), res(input.size()), other(input.size());
		std::vector<uint32_t> pos(count), seq(count, 0u);
		uint32_t seq_count = 0u, prev_seq = UINT32_MAX;
		pthread_t threads[kRanks];
		for ( uint32_t i = 0u; i < count; ++i )
		{
			const uint32_t seed = Mix(rows[i].sequence * 7919u + rows[i].variant * 104729u) ^ rows[i].position;
			for ( uint32_t c = 0u; c < G::kHidden; ++c )
				input[(uint64_t)i * G::kHidden + c] = Bf16(Uniform(seed, c));
			pos[i] = rows[i].position;
			seq[i] = rows[i].sequence;
			if ( rows[i].sequence != prev_seq )
			{
				prev_seq = rows[i].sequence;
				seq_count++;
			}
			host_context[rows[i].sequence] = rows[i].position + 1u;
		}
		for ( uint32_t r = 0u; r < kRanks; ++r )
		{
			Rank *rk = &ranks[r];
			memset(&rk->step, 0, sizeof(rk->step));
			rk->step.hidden_bf16 = rk->hidden;
			rk->step.positions = rk->positions;
			rk->step.context_length = rk->context;
			rk->step.sequence_of_row = rk->sequence_of_row;
			rk->step.rows = count;
			rk->step.sequences = seq_count;
			rk->step.commit = 1u;
			rk->step.last_rows_only = 1u;
			rk->step.gather_context = gather_ctx;
			rk->step.host_positions = pos.data();
			rk->step.host_sequence_of_row = seq.data();
			rk->step.context = VF_PAGES * G::kPageSlots;
			rk->step.multiprocessors = multiprocessors;
			rk->step.route_fault = VfRouteFault();
			if ( cudaMemcpyAsync(rk->hidden, input.data(), input.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, rk->stream) != cudaSuccess ||
				cudaMemcpyAsync(rk->positions, pos.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice, rk->stream) != cudaSuccess ||
				cudaMemcpyAsync(rk->sequence_of_row, seq.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice, rk->stream) != cudaSuccess ||
				cudaMemcpyAsync(rk->context, host_context, sizeof(host_context), cudaMemcpyHostToDevice, rk->stream) != cudaSuccess )
				return 0;
		}
		barrier.waiting = 0u;
		barrier.aborted = 0u;
		for ( uint32_t r = 0u; r < kRanks; ++r )
			if ( pthread_create(&threads[r], 0, Thread, &ranks[r]) != 0 )
				return 0;
		int ok = 1;
		for ( uint32_t r = 0u; r < kRanks; ++r )
		{
			pthread_join(threads[r], 0);
			ok = ok && ranks[r].status == SPARK_STATUS_OK;
		}
		if ( !ok || cudaMemcpy(res.data(), ranks[0].hidden, res.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess )
			return 0;
		for ( uint32_t r = 1u; r < kRanks; ++r )
		{
			if ( cudaMemcpy(other.data(), ranks[r].hidden, other.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess )
				return 0;
			Require(memcmp(other.data(), res.data(), other.size() * sizeof(uint16_t)) == 0, G::Tag(), "all TP4 ranks agree on GQA output (rank)", r, 0u, 0u);
		}
		if ( out != 0 )
			*out = res;
		return 1;
	}

	void Close(void)
	{
		for ( uint32_t r = 0u; r < kRanks; ++r )
		{
			if ( ranks[r].state != 0 )
			{
				cudaFree(ranks[r].state->arena);
				free(ranks[r].state);
			}
			cudaFree(ranks[r].hidden);
			cudaFree(ranks[r].positions);
			cudaFree(ranks[r].sequence_of_row);
			cudaFree(ranks[r].context);
			if ( ranks[r].stream != 0 )
				cudaStreamDestroy(ranks[r].stream);
		}
		cudaFree(staging);
		cudaFree(page_table);
		cudaFree(kv);
		pthread_cond_destroy(&barrier.cond);
		pthread_mutex_destroy(&barrier.mutex);
	}
};

static void CheckGqaTp4(void)
{
	static const uint32_t deltas[] = { 1u, 4u, 8u, 16u, 23u, 64u, 128u };
	static constexpr uint32_t kCached = 2u * SmallSolarTp4::kPageSlots;
	static constexpr uint32_t kMaxTotal = kCached + 128u;
	const char *model = SmallSolarTp4::Tag();
	LingGqaTp4Rig rig;
	if ( rig.Open() == 0 )
	{
		Require(0, model, "TP4 GQA synthetic stage opens", 0u, 0u, 0u);
		return;
	}
	std::vector<VfRow> all_rows(kMaxTotal);
	for ( uint32_t p = 0u; p < kMaxTotal; ++p )
		all_rows[p] = { 0u, p, 0u };
	std::vector<uint8_t> cached_kv(rig.kRanks * rig.kRankKv), cold_kv(rig.kRanks * rig.kRankKv), warm_kv(rig.kRanks * rig.kRankKv);
	Require(rig.Run(all_rows.data(), kCached, kCached, 0) != 0, model, "TP4 GQA cached prefix runs", kCached, 0u, 0u);
	Require(cudaMemcpy(cached_kv.data(), rig.kv, cached_kv.size(), cudaMemcpyDeviceToHost) == cudaSuccess, model, "TP4 GQA save cached KV", 0u, 0u, 0u);
	for ( uint32_t delta : deltas )
	{
		const uint32_t total = kCached + delta;
		std::vector<uint16_t> cold_out, warm_out;
		Require(cudaMemset(rig.kv, 0, cached_kv.size()) == cudaSuccess, model, "TP4 GQA reset KV", delta, 0u, 0u);
		Require(rig.Run(all_rows.data(), total, total, &cold_out) != 0, model, "TP4 cold GQA prefill runs", total, delta, 0u);
		Require(cudaMemcpy(cold_kv.data(), rig.kv, cold_kv.size(), cudaMemcpyDeviceToHost) == cudaSuccess, model, "TP4 GQA save cold KV", delta, 0u, 0u);
		std::vector<uint16_t> cold_tail(cold_out.begin() + (ptrdiff_t)((uint64_t)kCached * SmallSolarTp4::kHidden), cold_out.end());
		Require(cudaMemcpy(rig.kv, cached_kv.data(), cached_kv.size(), cudaMemcpyHostToDevice) == cudaSuccess, model, "TP4 GQA restore cached KV", delta, 0u, 0u);
		Require(rig.Run(all_rows.data() + kCached, delta, total, &warm_out) != 0, model, "TP4 warm GQA delta prefill runs", kCached, delta, 0u);
		Require(cudaMemcpy(warm_kv.data(), rig.kv, warm_kv.size(), cudaMemcpyDeviceToHost) == cudaSuccess, model, "TP4 GQA save warm KV", delta, 0u, 0u);
		Require(FirstDifference(warm_out, cold_tail, SmallSolarTp4::kHidden) == UINT32_MAX, model,
			"TP4 warm GQA prefill equals cold bit for bit (delta, first differing row)", delta, FirstDifference(warm_out, cold_tail, SmallSolarTp4::kHidden), 0u);
		Require(memcmp(warm_kv.data(), cold_kv.data(), warm_kv.size()) == 0, model,
			"TP4 warm GQA leaves identical sharded KV on all 4 ranks (delta)", delta, 0u, 0u);
	}
	{
		std::vector<uint16_t> ref_out, gather_out;
		const uint32_t delta = 64u, total = kCached + delta;
		Require(cudaMemcpy(rig.kv, cached_kv.data(), cached_kv.size(), cudaMemcpyHostToDevice) == cudaSuccess, model, "TP4 GQA accuracy restore 1", 0u, 0u, 0u);
		Require(rig.Run(all_rows.data() + kCached, delta, 0u, &ref_out) != 0, model, "TP4 reference scatter GQA runs", delta, 0u, 0u);
		Require(cudaMemcpy(rig.kv, cached_kv.data(), cached_kv.size(), cudaMemcpyHostToDevice) == cudaSuccess, model, "TP4 GQA accuracy restore 2", 0u, 0u, 0u);
		Require(rig.Run(all_rows.data() + kCached, delta, total, &gather_out) != 0, model, "TP4 KV-gather warm GQA runs", delta, 0u, 0u);
		for ( uint32_t p = 0u; p < delta; ++p )
		{
			const float dist = RelativeRowDistance(gather_out, p, ref_out, p, SmallSolarTp4::kHidden);
			Require(dist < 0.01f, model, "TP4 warm KV-gather GQA matches reference scatter GQA (pos, rel x 1e6)", p, (uint32_t)(dist * 1.0e6f), 0u);
		}
	}
	{
		std::vector<uint16_t> b1_warm, b2_warm;
		const uint32_t delta = 16u, total = kCached + delta;
		std::vector<VfRow> multi;
		for ( uint32_t p = 0u; p < delta; ++p )
			multi.push_back({ 0u, kCached + p, 0u });
		for ( uint32_t p = 0u; p < 24u; ++p )
			multi.push_back({ 1u, p, 7u });
		Require(cudaMemcpy(rig.kv, cached_kv.data(), cached_kv.size(), cudaMemcpyHostToDevice) == cudaSuccess, model, "TP4 multi-seq GQA restore 1", 0u, 0u, 0u);
		Require(rig.Run(all_rows.data() + kCached, delta, total, &b1_warm) != 0, model, "TP4 B1 warm GQA runs", delta, 0u, 0u);
		Require(cudaMemcpy(rig.kv, cached_kv.data(), cached_kv.size(), cudaMemcpyHostToDevice) == cudaSuccess, model, "TP4 multi-seq GQA restore 2", 0u, 0u, 0u);
		Require(rig.Run(multi.data(), (uint32_t)multi.size(), 0u, &b2_warm) != 0, model, "TP4 multi-seq warm GQA runs", (uint32_t)multi.size(), 0u, 0u);
		std::vector<uint16_t> lane0(b2_warm.begin(), b2_warm.begin() + (ptrdiff_t)((uint64_t)delta * SmallSolarTp4::kHidden));
		Require(FirstDifference(lane0, b1_warm, SmallSolarTp4::kHidden) == UINT32_MAX, model,
			"TP4 multi-seq warm GQA lane0 equals B1 warm GQA bit for bit (first differing row)", FirstDifference(lane0, b1_warm, SmallSolarTp4::kHidden), 0u, 0u);
	}
	printf("%s %s multi-rank GQA KV-gather prefix-exact + multi-seq 5c + accuracy: 4 ranks, cached=%u vs warm deltas 1..128\n", vf_failures == 0 ? "ok" : "checked", model, kCached);
	rig.Close();
}

int main(void)
{
	Check<SmallSolar>();
	Check<SmallLing3>();
	Check<SmallLing32Fp8>();
	Check<SmallLing31Flash>();
	CheckPrefixExact<SmallSolar>();
	CheckPrefixExact<SmallLing3>();
	CheckPrefixExact<SmallLing32Fp8>();
	CheckPrefixExact<SmallLing31Flash>();
	CheckMlaTp4();
	CheckGqaTp4();
	if ( vf_failures != 0 )
	{
		printf("FAIL ling verify frames: %d failures\n", vf_failures);
		return 1;
	}
	printf("PASS ling verify frames: every row of every verify frame equals one-row decode bit for bit, keep_rows restores the recurrent state of the kept rows, a rejected tail leaves no trace, a tree frame's trunk keeps its chain bits and its branches verify and fold as their own chains\n");
	return 0;
}
