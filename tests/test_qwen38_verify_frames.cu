#include "modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_stage_model.cu"
#include "modules/qwen38_max_resident_decode_stage/source/spark_qwen38_max_stage_model.cu"
#include "modules/qwen38_flash_resident_decode_stage/source/spark_qwen38_flash_stage_model.cu"
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
#define VF_FILL_ROWS 16u
#define VF_FFN 512u

struct Small27b : Qwen38_27bGeometry<16u>
{
	static constexpr uint32_t kTpDegree = 1u;
	static constexpr uint32_t kHidden = 1024u;
	static constexpr uint32_t kLayers = 4u;
	static constexpr uint32_t kGdnKeyHeads = 2u;
	static constexpr uint32_t kGdnValueHeads = 4u;
	static constexpr uint32_t kHeads = 4u;
	static constexpr uint32_t kKvHeads = 1u;
	static const char *Tag(void) { return "qwen3.8-27b"; }
};

struct SmallMax : Qwen38MaxGeometry<SPARK_QWEN38_MAX_STAGE_TP_DEGREE>
{
	static constexpr uint32_t kTpDegree = 1u;
	static constexpr uint32_t kHidden = 1024u;
	static constexpr uint32_t kLayers = 4u;
	static constexpr uint32_t kGdnKeyHeads = 2u;
	static constexpr uint32_t kGdnValueHeads = 4u;
	static constexpr uint32_t kHeads = 4u;
	static constexpr uint32_t kKvHeads = 1u;
	static constexpr uint32_t kExperts = 32u;
	static constexpr uint32_t kLocalExperts = 32u;
	static constexpr uint32_t kMoeInter = 256u;
	static constexpr uint32_t kSharedRows = 256u;
	static const char *Tag(void) { return "qwen3.8-max"; }
};

struct SmallFlash : Qwen38FlashGeometry<SPARK_QWEN38_FLASH_STAGE_TP_DEGREE, QWEN38_FLASH_EXPERT_CODEC, QWEN38_FLASH_PLE_TABLE_CODEC>
{
	static constexpr uint32_t kTpDegree = 1u;
	static constexpr uint32_t kHidden = 1280u;
	static constexpr uint32_t kLayers = 4u;
	static constexpr uint32_t kGdnKeyHeads = 2u;
	static constexpr uint32_t kGdnValueHeads = 4u;
	static constexpr uint32_t kHeads = 4u;
	static constexpr uint32_t kKvHeads = 1u;
	static constexpr uint32_t kExperts = 32u;
	static constexpr uint32_t kLocalExperts = 32u;
	static constexpr uint32_t kMoeInter = 128u;
	static constexpr uint32_t kHcLowRank = 128u;
	static constexpr uint32_t kPleHeadDim = 80u;
	static constexpr uint32_t kIndexBudget = 64u;
	static const char *Tag(void) { return "qwen3.8-flash"; }
};

static_assert(QwenHybridShape<SmallFlash>::kPleColumns * SmallFlash::kPleHeadDim == SmallFlash::kHidden, "the n-gram heads tile the hidden width");
static_assert(VF_PREFIX / SmallFlash::kIndexRatio > QwenHybridShape<SmallFlash>::kIndexBlocks, "the frames start where the indexer selects blocks");

static void Need(int condition, const char *model, const char *what, uint32_t a, uint32_t b, uint32_t c)
{
	if ( condition )
		return;
	printf("FAIL %s: %s (%u %u %u)\n", model, what, a, b, c);
	exit(1);
}

static uint8_t *Bytes(uint64_t count, uint32_t kind)
{
	std::vector<uint8_t> host(count);
	void *device = 0;
	const uint32_t seed = vf_seed++;
	for ( uint64_t index = 0u; index < count; ++index )
	{
		const uint32_t z = Mix(seed * 0x9e3779b9u + (uint32_t)index);
		host[index] = kind == 0u ? (uint8_t)(((z & 1u) << 7u) | ((3u + (z >> 1u) % 8u) << 3u) | ((z >> 8u) & 7u)) :
			kind == 1u ? (uint8_t)z : (uint8_t)(0x30u + (z >> 4u) % 16u);
	}
	Need(cudaMalloc(&device, count) == cudaSuccess && cudaMemcpy(device, host.data(), count, cudaMemcpyHostToDevice) == cudaSuccess,
		"weights", "byte tensor allocates", (uint32_t)count, 0u, 0u);
	return (uint8_t *)device;
}

template<class G>
static void Experts(uint64_t rows, uint64_t columns, const uint8_t **payload, const uint8_t **scale)
{
	if constexpr ( G::kExpertCodec == SPARK_WEIGHT_CODEC_BF16 )
	{
		*payload = (const uint8_t *)Matrix((uint64_t)G::kLocalExperts * rows, columns);
		*scale = 0;
		return;
	}
	const uint64_t scale_bytes = SparkWeightCodecScaleBytes(SPARK_WEIGHT_CODEC_NVFP4_E2M1, G::kLocalExperts, rows, columns);
	uint8_t *scales = Bytes(scale_bytes, 2u);
	const float *global = Tensor<float>(G::kLocalExperts, 0.0f, 0.5f / sqrtf((float)columns));
	Need(cudaMemcpy(scales, global, G::kLocalExperts * sizeof(float), cudaMemcpyDeviceToDevice) == cudaSuccess, "weights", "expert scales copy", 0u, 0u, 0u);
	*payload = Bytes(SparkWeightCodecPayloadBytes(SPARK_WEIGHT_CODEC_NVFP4_E2M1, G::kLocalExperts * rows, columns), 1u);
	*scale = scales;
}

template<class G>
struct VfRig
{
	using M = QwenHybridModel<G>;
	using S = QwenHybridShape<G>;
	using State = QwenHybridState<G>;
	static constexpr uint64_t kKvRegion = (uint64_t)S::kAttnLayers * VF_SLOTS * VF_PAGES * S::Kv::kPageBytes;
	static constexpr uint64_t kIndexRegion = S::kIndexer ? (uint64_t)S::kAttnLayers * VF_SLOTS * VF_PAGES * S::IndexKv::kPageBytes : 0u;
	State *state;
	VfBus bus;
	SparkStageRunnerServices services;
	cudaStream_t stream;
	uint32_t multiprocessors;
	uint8_t *kv, *index_kv, *recurrent_buffer;
	uint32_t *page_table, *positions, *context, *sequence_of_row, *row_begin, *row_indices, *recurrent, *kept, *tokens;
	uint16_t *hidden;
	uint32_t host_context[VF_SLOTS];
	uint32_t linear_bytes;
	uint64_t recurrent_bytes;
	SparkStageRunnerStep last;
	std::vector<uint32_t> frame_tables[7];

	void Linear(uint64_t rows, uint64_t columns, const uint8_t **weight, const float **scale)
	{
		if ( linear_bytes == 2u )
		{
			*weight = (const uint8_t *)Matrix(rows, columns);
			*scale = 0;
			return;
		}
		*weight = Bytes(rows * columns, 0u);
		*scale = Tensor<float>(rows * ((columns + QH_SCALE_K - 1u) / QH_SCALE_K), 0.3f / sqrtf((float)columns), 0.6f / sqrtf((float)columns));
	}

	void Layer(QwenHybridLayer *layer, uint32_t index)
	{
		if ( !S::kHyper )
		{
			layer->input_norm = Norm(G::kHidden);
			layer->post_norm = Norm(G::kHidden);
		}
		for ( uint32_t part = 0u; part < 2u && S::kHyper; ++part )
		{
			layer->hc_norm[part] = Norm(S::kResidual);
			layer->hc_down[part] = Matrix(G::kHcLowRank, S::kResidual);
			layer->hc_up[part] = Matrix(S::kResidual, G::kHcLowRank);
			layer->hc_inject[part] = Tensor<uint16_t>((uint64_t)G::kStreams * S::kResidual, 0.5f / sqrtf((float)S::kResidual), 0.0f);
		}
		if ( index % G::kPeriod == G::kPhase )
		{
			Linear(S::kQkvWidth, G::kHidden, &layer->mix_weight, &layer->mix_scale);
			Linear(G::kHidden, S::kQWidth, &layer->out_weight, &layer->out_scale);
			layer->q_norm = Norm(G::kHeadDim);
			layer->k_norm = Norm(G::kHeadDim);
			if ( S::kIndexer )
			{
				layer->index_qk = Matrix(S::kIndexWidth, G::kHidden);
				layer->index_q_norm = Norm(G::kIndexDim);
				layer->index_k_norm = Norm(G::kIndexDim);
			}
		}
		else
		{
			Linear(2u * state->gdn_qk + 2u * state->gdn_v, G::kHidden, &layer->mix_weight, &layer->mix_scale);
			Linear(G::kHidden, state->gdn_v, &layer->out_weight, &layer->out_scale);
			layer->gdn_ba = Matrix(2u * state->gdn_heads, G::kHidden);
			layer->gdn_conv = Tensor<uint16_t>((uint64_t)state->gdn_channels * QH_CONV, 0.5f, 0.0f);
			layer->gdn_norm = Tensor<uint16_t>(QH_VALUE_DIM, 0.2f, 1.0f);
			layer->gdn_a_log = Tensor<float>(state->gdn_heads, 0.5f, 0.0f);
			layer->gdn_dt_bias = Tensor<float>(state->gdn_heads, 0.5f, 0.0f);
		}
		if constexpr ( G::kMoe )
		{
			layer->moe_router = Matrix(G::kExperts, G::kHidden);
			Experts<G>(G::kMoeInter, G::kHidden, &layer->moe_w1, &layer->moe_w1_scale);
			Experts<G>(G::kMoeInter, G::kHidden, &layer->moe_w3, &layer->moe_w3_scale);
			Experts<G>(G::kHidden, G::kMoeInter, &layer->moe_w2, &layer->moe_w2_scale);
			layer->moe_shared_gate_up = Matrix(2u * G::kSharedRows, G::kHidden);
			layer->moe_shared_down = Matrix(G::kHidden, G::kSharedRows);
			layer->moe_shared_gate = Tensor<uint16_t>(G::kHidden, 1.0f / sqrtf((float)G::kHidden), 0.0f);
		}
		else
		{
			Linear(2u * state->ffn_rows, G::kHidden, &layer->ffn_gate_up, &layer->ffn_gate_up_scale);
			Linear(G::kHidden, state->ffn_rows, &layer->ffn_down, &layer->ffn_down_scale);
		}
	}

	void Ple(void)
	{
		std::vector<int64_t> multipliers(G::kPleOrders + 1u), primes(S::kPleColumns), offsets(S::kPleColumns);
		int64_t rows = 0;
		void *device[3] = { 0, 0, 0 };
		for ( uint32_t order = 0u; order <= G::kPleOrders; ++order )
			multipliers[order] = 1000003 + 7919 * (int64_t)order;
		for ( uint32_t column = 0u; column < S::kPleColumns; ++column )
		{
			primes[column] = 61 + 6 * (int64_t)column;
			offsets[column] = rows;
			rows += primes[column];
		}
		state->ple_table_first = 0u;
		state->ple_table_rows = (uint32_t)rows;
		state->ple_key_weight = Matrix(S::kResidual, G::kHidden);
		state->ple_value_weight = Matrix(G::kHidden, G::kHidden);
		state->ple_norm_key = Norm(S::kResidual);
		state->ple_norm_query = Norm(S::kResidual);
		state->ple_norm_conv = Norm(S::kResidual);
		state->ple_conv_weight = Tensor<uint16_t>((uint64_t)S::kResidual * G::kPleConv, 0.5f, 0.0f);
		if constexpr ( G::kPleTableCodec == SPARK_WEIGHT_CODEC_BF16 )
		{
			state->ple_table = (const uint8_t *)Tensor<uint16_t>((uint64_t)rows * G::kPleHeadDim, 0.5f, 0.0f);
			state->ple_table_scale = 0;
		}
		else
		{
			state->ple_table = Bytes((uint64_t)rows * G::kPleHeadDim, 0u);
			state->ple_table_scale = Tensor<float>(1u, 0.0f, 0.5f);
		}
		Need(cudaMalloc(&device[0], multipliers.size() * sizeof(int64_t)) == cudaSuccess &&
			cudaMalloc(&device[1], primes.size() * sizeof(int64_t)) == cudaSuccess && cudaMalloc(&device[2], offsets.size() * sizeof(int64_t)) == cudaSuccess &&
			cudaMemcpy(device[0], multipliers.data(), multipliers.size() * sizeof(int64_t), cudaMemcpyHostToDevice) == cudaSuccess &&
			cudaMemcpy(device[1], primes.data(), primes.size() * sizeof(int64_t), cudaMemcpyHostToDevice) == cudaSuccess &&
			cudaMemcpy(device[2], offsets.data(), offsets.size() * sizeof(int64_t), cudaMemcpyHostToDevice) == cudaSuccess, G::Tag(),
			"the n-gram tables load", 0u, 0u, 0u);
		state->ple_multipliers = (const int64_t *)device[0];
		state->ple_vocab_sizes = (const int64_t *)device[1];
		state->ple_offsets = (const int64_t *)device[2];
	}

	int Open(uint32_t bytes)
	{
		std::vector<uint32_t> table(VF_SLOTS * VF_PAGES);
		SparkStageRunnerKv attach;
		memset(this, 0, sizeof(*this));
		linear_bytes = bytes;
		bus.hidden = G::kHidden;
		services.context = &bus;
		services.round = VfRound;
		state = (State *)calloc(1u, sizeof(State));
		state->services = &services;
		state->tp_degree = 1u;
		state->max_rows = VF_MAX_FRAME;
		state->slots = VF_SLOTS;
		state->ffn_rows = VF_FFN;
		state->linear_bytes = bytes;
		state->linear_kind = bytes == 1u ? "fp8_e4m3" : "bf16";
		state->max_positions = VF_PAGES * G::kPageSlots;
		if ( M::Size(state) != SPARK_STATUS_OK )
			return 0;
		for ( uint32_t index = 0u; index < G::kLayers; ++index )
			Layer(&state->layers[index], index);
		if constexpr ( S::kHyper )
		{
			state->final_hc_norm = Norm(S::kResidual);
			state->final_hc_down = Matrix(G::kHcLowRank, S::kResidual);
			state->final_hc_up = Matrix(S::kResidual, G::kHcLowRank);
		}
		if constexpr ( S::kPle )
			Ple();
		recurrent_bytes = M::RecurrentBytes(state);
		if ( M::Allocate(state) != SPARK_STATUS_OK || cudaStreamCreate(&stream) != cudaSuccess ||
			cudaDeviceGetAttribute((int *)&multiprocessors, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess ||
			cudaMalloc((void **)&kv, kKvRegion) != cudaSuccess || cudaMemset(kv, 0, kKvRegion) != cudaSuccess ||
			(S::kIndexer && (cudaMalloc((void **)&index_kv, kIndexRegion) != cudaSuccess || cudaMemset(index_kv, 0, kIndexRegion) != cudaSuccess)) ||
			cudaMalloc((void **)&recurrent_buffer, VF_SLOTS * recurrent_bytes) != cudaSuccess ||
			cudaMalloc((void **)&page_table, table.size() * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&positions, VF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&tokens, VF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&sequence_of_row, VF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&row_indices, VF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&row_begin, (VF_SLOTS + 1u) * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&recurrent, VF_SLOTS * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&kept, VF_SLOTS * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&context, VF_SLOTS * sizeof(uint32_t)) != cudaSuccess ||
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
		if ( S::kIndexer )
		{
			attach.second_pool = index_kv;
			attach.second_layer_stride_bytes = (uint64_t)VF_SLOTS * VF_PAGES * S::IndexKv::kPageBytes;
			attach.second_layer_page_bytes = S::IndexKv::kPageBytes;
			attach.second_layer_count = S::kAttnLayers;
		}
		return M::AttachKv(state, &attach) == SPARK_STATUS_OK;
	}

	void Input(const VfRow &row, uint16_t *target, uint32_t *token) const
	{
		const uint32_t seed = Mix(row.sequence * 7919u + row.variant * 104729u) ^ row.position;
		for ( uint32_t index = 0u; index < G::kHidden; ++index )
			target[index] = Bf16(Uniform(seed, index));
		*token = row.position % 37u == 5u ? G::kPleEos : Mix(seed) % 4093u + 1u;
	}

	int Run(const VfRow *rows, uint32_t count, uint32_t commit, std::vector<uint16_t> *output)
	{
		std::vector<uint16_t> input((uint64_t)count * G::kHidden);
		std::vector<uint32_t> position(count), sequence(count), order(count), token(count), begin(1u, 0u), slots;
		SparkStageRunnerStep step;
		for ( uint32_t row = 0u; row < count; ++row )
		{
			Input(rows[row], input.data() + (uint64_t)row * G::kHidden, &token[row]);
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
		step.token_ids = tokens;
		step.rows = count;
		step.sequences = (uint32_t)slots.size();
		step.commit = commit;
		step.context = VF_PAGES * G::kPageSlots;
		step.multiprocessors = multiprocessors;
		step.route_fault = VfRouteFault();
		last = step;
		if ( commit == 0u )
		{
			frame_tables[0] = position;
			frame_tables[1] = token;
			frame_tables[2] = sequence;
			frame_tables[3] = order;
			frame_tables[4] = begin;
			frame_tables[5] = slots;
			frame_tables[6].assign(host_context, host_context + VF_SLOTS);
		}
		if ( cudaMemcpyAsync(hidden, input.data(), input.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			cudaMemcpyAsync(positions, position.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			cudaMemcpyAsync(tokens, token.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
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
			Need(Finite(std::vector<uint16_t>(output->begin() + offset, output->end())) != 0u, G::Tag(), "every step output is finite and not constant",
				count, commit, 0u);
		}
		return 1;
	}

	int Keep(const uint32_t *counts)
	{
		uint32_t *targets[7] = { positions, tokens, sequence_of_row, row_indices, row_begin, recurrent, context };
		for ( uint32_t table = 0u; table < 7u; ++table )
			if ( cudaMemcpy(targets[table], frame_tables[table].data(), frame_tables[table].size() * sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess )
				return 0;
		return cudaMemcpy(kept, counts, last.sequences * sizeof(uint32_t), cudaMemcpyHostToDevice) == cudaSuccess &&
			M::KeepRows(state, &last, kept, stream) == SPARK_STATUS_OK && cudaStreamSynchronize(stream) == cudaSuccess;
	}

	int Decode(uint32_t sequence, uint32_t position, uint32_t variant, std::vector<uint16_t> *output)
	{
		const VfRow row = { sequence, position, variant };
		return Run(&row, 1u, 1u, output);
	}

	std::vector<uint8_t> Save(void) const
	{
		std::vector<uint8_t> copy(kKvRegion + kIndexRegion + VF_SLOTS * recurrent_bytes);
		for ( uint32_t slot = 0u; slot < VF_SLOTS; ++slot )
			if ( M::RecurrentCopy(state, 1u, slot, recurrent_buffer + slot * recurrent_bytes, recurrent_bytes, 0) != SPARK_STATUS_OK )
				copy.clear();
		if ( copy.empty() || cudaMemcpy(copy.data(), kv, kKvRegion, cudaMemcpyDeviceToHost) != cudaSuccess ||
			(S::kIndexer && cudaMemcpy(copy.data() + kKvRegion, index_kv, kIndexRegion, cudaMemcpyDeviceToHost) != cudaSuccess) ||
			cudaMemcpy(copy.data() + kKvRegion + kIndexRegion, recurrent_buffer, VF_SLOTS * recurrent_bytes, cudaMemcpyDeviceToHost) != cudaSuccess )
			copy.clear();
		return copy;
	}

	int Load(const std::vector<uint8_t> &copy)
	{
		if ( copy.size() != kKvRegion + kIndexRegion + VF_SLOTS * recurrent_bytes || cudaMemcpy(kv, copy.data(), kKvRegion, cudaMemcpyHostToDevice) != cudaSuccess ||
			(S::kIndexer && cudaMemcpy(index_kv, copy.data() + kKvRegion, kIndexRegion, cudaMemcpyHostToDevice) != cudaSuccess) ||
			cudaMemcpy(recurrent_buffer, copy.data() + kKvRegion + kIndexRegion, VF_SLOTS * recurrent_bytes, cudaMemcpyHostToDevice) != cudaSuccess )
			return 0;
		for ( uint32_t slot = 0u; slot < VF_SLOTS; ++slot )
			if ( M::RecurrentCopy(state, 0u, slot, recurrent_buffer + slot * recurrent_bytes, recurrent_bytes, 0) != SPARK_STATUS_OK )
				return 0;
		return 1;
	}

	uint32_t RecurrentMatches(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b) const
	{
		const uint64_t offset = kKvRegion + kIndexRegion;
		return a.size() == b.size() && memcmp(a.data() + offset, b.data() + offset, VF_SLOTS * recurrent_bytes) == 0 ? 1u : 0u;
	}

	uint32_t RecurrentDifference(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b) const
	{
		const uint64_t offset = kKvRegion + kIndexRegion;
		for ( uint64_t index = 0u; index < VF_SLOTS * recurrent_bytes && a.size() == b.size(); ++index )
			if ( a[offset + index] != b[offset + index] )
				return (uint32_t)index;
		return UINT32_MAX;
	}

	template<class Kv>
	static uint32_t RegionMatches(const uint8_t *a, const uint8_t *b, uint32_t layers, uint32_t limit)
	{
		for ( uint32_t layer = 0u; layer < layers; ++layer )
			for ( uint32_t sequence = 0u; sequence < VF_SLOTS; ++sequence )
				for ( uint32_t slot = 0u; slot < VF_PAGES * Kv::kPageSlots; ++slot )
				{
					const uint64_t offset = ((uint64_t)layer * VF_SLOTS * VF_PAGES + sequence * VF_PAGES + slot / Kv::kPageSlots) * Kv::kPageBytes +
						(uint64_t)(slot % Kv::kPageSlots) * Kv::kSlotBytes;
					if ( (sequence != 0u || slot < limit) && memcmp(a + offset, b + offset, Kv::kSlotBytes) != 0 )
						return 0u;
				}
		return 1u;
	}

	static uint32_t KvMatches(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b, uint32_t limit)
	{
		if ( a.size() != b.size() || RegionMatches<typename S::Kv>(a.data(), b.data(), S::kAttnLayers, limit) == 0u )
			return 0u;
		return S::kIndexer ? RegionMatches<typename S::IndexKv>(a.data() + kKvRegion, b.data() + kKvRegion, S::kAttnLayers,
			limit == UINT32_MAX ? UINT32_MAX : limit / G::kIndexRatio) : 1u;
	}

	uint32_t KvClean(void) const
	{
		LmKvAccessError error, zero;
		memset(&zero, 0, sizeof(zero));
		return cudaMemcpy(&error, state->kv_error, sizeof(error), cudaMemcpyDeviceToHost) == cudaSuccess && memcmp(&error, &zero, sizeof(error)) == 0 ? 1u : 0u;
	}
};

template<class G>
static void Check(uint32_t linear_bytes, const char *model)
{
	static const uint32_t frames[] = { 1u, 2u, 7u, 16u, 17u, 40u, VF_MAX_FRAME };
	VfRig<G> rig;
	std::vector<VfRow> rows;
	std::vector<uint8_t> prefix;
	uint32_t checked = 0u;
	Need(rig.Open(linear_bytes) != 0, model, "the synthetic stage sizes, allocates and attaches its KV pools", 0u, 0u, 0u);
	Need(QwenHybridModel<G>::Interface()->keep_rows != 0, model, "the stage interface carries keep_rows", 0u, 0u, 0u);
	for ( uint32_t sequence = 0u; sequence < VF_SLOTS; ++sequence )
		for ( uint32_t first = 0u; first < (sequence == 0u ? VF_PREFIX : VF_OTHER_PREFIX); first += VF_FILL_ROWS )
		{
			rows.clear();
			for ( uint32_t position = first; position < first + VF_FILL_ROWS; ++position )
				rows.push_back({ sequence, position, 0u });
			Need(rig.Run(rows.data(), (uint32_t)rows.size(), 1u, 0) != 0, model, "the prefix fill runs", sequence, first, 0u);
		}
	prefix = rig.Save();
	Need(!prefix.empty(), model, "the prefix state saves", 0u, 0u, 0u);
	for ( uint32_t frame : frames )
	{
		std::vector<uint16_t> decoded, verified;
		std::vector<std::vector<uint8_t> > before;
		std::vector<uint32_t> kept;
		std::vector<uint8_t> after_decode, after_verify;
		Kept(frame, &kept);
		Need(rig.Load(prefix) != 0, model, "the prefix state loads", frame, 0u, 0u);
		for ( uint32_t row = 0u; row < frame; ++row )
		{
			for ( uint32_t k : kept )
				if ( k == row )
					before.push_back(rig.Save());
			Need(rig.Decode(0u, VF_PREFIX + row, 0u, &decoded) != 0, model, "one-row decode runs", frame, row, 0u);
		}
		before.push_back(rig.Save());
		after_decode = before.back();
		Need(rig.Load(prefix) != 0, model, "the prefix state loads", frame, 0u, 0u);
		rows.clear();
		for ( uint32_t row = 0u; row < frame; ++row )
			rows.push_back({ 0u, VF_PREFIX + row, 0u });
		Need(rig.Run(rows.data(), frame, 0u, &verified) != 0, model, "the verify frame runs", frame, 0u, 0u);
		after_verify = rig.Save();
		const SparkStageRunnerStep frame_step = rig.last;
		Need(FirstDifference(verified, decoded, G::kHidden) == UINT32_MAX, model,
			"every verify row equals its one-row decode bit for bit (frame rows, first differing row)", frame, FirstDifference(verified, decoded, G::kHidden), 0u);
		Need(rig.KvMatches(after_verify, after_decode, UINT32_MAX) != 0u, model,
			"the attention and index KV the verify frame leaves equals the KV of the one-row decodes", frame, 0u, 0u);
		Need(rig.RecurrentMatches(after_verify, prefix) != 0u, model, "a verify frame leaves every recurrent part untouched (frame rows, first differing byte)",
			frame, rig.RecurrentDifference(after_verify, prefix), 0u);
		for ( uint32_t index = 0u; index < kept.size(); ++index )
		{
			const uint32_t k = kept[index], limit = VF_PREFIX + k + 2u, counts[1] = { k };
			std::vector<uint16_t> speculated, plain;
			Need(rig.Load(after_verify) != 0, model, "the verified state loads", frame, k, 0u);
			rig.last = frame_step;
			Need(rig.Keep(counts) != 0, model, "keep_rows folds the kept rows", frame, k, 0u);
			const std::vector<uint8_t> kept_state = rig.Save();
			Need(rig.RecurrentMatches(kept_state, before[index]) != 0u, model,
				"after keeping k rows every recurrent part equals k one-row decodes (frame rows, k, first differing state byte)", frame, k,
				rig.RecurrentDifference(kept_state, before[index]));
			Need(rig.Decode(0u, VF_PREFIX + k, 1u, &speculated) != 0 && rig.Decode(0u, VF_PREFIX + k + 1u, 2u, &speculated) != 0, model,
				"the decodes after a rejected tail run", frame, k, 0u);
			const std::vector<uint8_t> speculated_state = rig.Save();
			Need(rig.Load(before[index]) != 0, model, "the never-drafted state loads", frame, k, 0u);
			Need(rig.Decode(0u, VF_PREFIX + k, 1u, &plain) != 0 && rig.Decode(0u, VF_PREFIX + k + 1u, 2u, &plain) != 0, model,
				"the decodes of the run that never saw the draft run", frame, k, 0u);
			Need(FirstDifference(speculated, plain, G::kHidden) == UINT32_MAX, model,
				"after keeping k rows the next two decodes equal a run that never saw the draft (frame rows, k, first differing step)", frame, k,
				FirstDifference(speculated, plain, G::kHidden));
			const std::vector<uint8_t> plain_state = rig.Save();
			Need(rig.KvMatches(speculated_state, plain_state, limit) != 0u && rig.RecurrentMatches(speculated_state, plain_state) != 0u, model,
				"after keeping k rows the KV and recurrent state equal a run that never saw the draft (frame rows, k)", frame, k, 0u);
			checked++;
		}
		printf("  %s: %3u-row frame equals its one-row decodes; keeps of %zu row counts restore the state\n", model, frame, kept.size());
	}
	{
		std::vector<uint16_t> decoded, verified;
		std::vector<uint8_t> after_decode;
		const uint32_t first = 7u, second = 16u, counts[2] = { first, second };
		Need(rig.Load(prefix) != 0, model, "the prefix state loads", 0u, 0u, 0u);
		rows.clear();
		for ( uint32_t row = 0u; row < first; ++row )
		{
			rows.push_back({ 0u, VF_PREFIX + row, 0u });
			Need(rig.Decode(0u, VF_PREFIX + row, 0u, &decoded) != 0, model, "one-row decode of the first sequence runs", row, 0u, 0u);
		}
		for ( uint32_t row = 0u; row < second; ++row )
		{
			rows.push_back({ 1u, VF_OTHER_PREFIX + row, 0u });
			Need(rig.Decode(1u, VF_OTHER_PREFIX + row, 0u, &decoded) != 0, model, "one-row decode of the second sequence runs", row, 0u, 0u);
		}
		after_decode = rig.Save();
		Need(rig.Load(prefix) != 0, model, "the prefix state loads", 0u, 0u, 0u);
		Need(rig.Run(rows.data(), first + second, 0u, &verified) != 0 && rig.Keep(counts) != 0, model, "a verify frame over two sequences runs and keeps",
			first, second, 0u);
		Need(FirstDifference(verified, decoded, G::kHidden) == UINT32_MAX, model,
			"every row of a two-sequence verify frame equals its one-row decode (first differing row)", FirstDifference(verified, decoded, G::kHidden), 0u, 0u);
		const std::vector<uint8_t> kept_state = rig.Save();
		Need(rig.KvMatches(kept_state, after_decode, UINT32_MAX) != 0u && rig.RecurrentMatches(kept_state, after_decode) != 0u, model,
			"a two-sequence verify frame, every row kept, leaves the decodes' KV and recurrent state", 0u, 0u, 0u);
	}
	Need(rig.KvClean() != 0u, model, "no step reported a KV access error", 0u, 0u, 0u);
	printf("ok %s: frames of 1-%u rows, %u rejection points, %llu recurrent bytes per slot\n", model, VF_MAX_FRAME, checked,
		(unsigned long long)rig.recurrent_bytes);
}

int main(void)
{
	Check<Small27b>(1u, "qwen3.8-27b (fp8 linear)");
	Check<Small27b>(2u, "qwen3.8-27b (bf16 linear)");
	Check<SmallMax>(2u, "qwen3.8-max");
	Check<SmallFlash>(2u, "qwen3.8-flash");
	printf("PASS qwen38 verify frames: every row of every verify frame equals one-row decode bit for bit, keep_rows restores the gated delta-net, "
		"n-gram and index-tail state of the kept rows, and a rejected tail leaves no trace\n");
	return 0;
}
