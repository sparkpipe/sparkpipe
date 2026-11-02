#include "modules/solar_open2_tp16_resident_decode_stage/source/spark_solar_open2_tp16_stage_model.cu"
#include "modules/ling3_resident_decode_stage/source/spark_ling3_stage_model.cu"
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
#define VF_FFN 256u
#define VF_FILL_ROWS 16u

static_assert(SolarOpen2Geometry::kVerifyRows == VF_MAX_FRAME && Ling3Geometry::kVerifyRows == VF_MAX_FRAME,
	"the test frames reach the verify rows Solar Open 2 and Ling 3.0 serve");

struct SmallSolar : SolarOpen2Geometry
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

struct SmallLing3 : Ling3Geometry
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
	uint16_t *hidden;
	uint32_t host_context[VF_SLOTS];
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
		cudaFree(hidden);
		cudaStreamDestroy(stream);
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
	Require(rig.KvClean() != 0u, model, "no step reported a KV access error", 0u, 0u, 0u);
	printf("%s %s: frames of 1-%u rows, %u rejection points\n", vf_failures == 0 ? "ok" : "checked", model, VF_MAX_FRAME, checked);
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
	printf("PASS ling verify frames: every row of every verify frame equals one-row decode bit for bit, keep_rows restores the recurrent state of the kept rows, and a rejected tail leaves no trace\n");
	return 0;
}
