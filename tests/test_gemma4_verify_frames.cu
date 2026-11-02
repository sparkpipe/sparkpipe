#include "modules/gemma4_31b_resident_decode_stage/source/spark_gemma4_31b_stage_model.cu"
#include "modules/gemma4_26b_resident_decode_stage/source/spark_gemma4_26b_stage_model.cu"
#include "modules/muse_glimmer_tp16_resident_decode_stage/source/spark_muse_tp16_stage_model.cu"
#include "modules/laguna_tp16_resident_decode_stage/source/spark_laguna_tp16_stage_model.cu"
#include "modules/kexaone2_tp16_resident_decode_stage/source/spark_kexaone2_tp16_stage_model.cu"

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

struct Small31b : Gemma4_31bGeometry
{
	static constexpr uint32_t kTpDegree = 1u;
	static constexpr uint32_t kHidden = 512u;
	static constexpr uint32_t kLayers = 6u;
	static constexpr uint32_t kSlidingHeads = 4u;
	static constexpr uint32_t kFullHeads = 4u;
	static constexpr uint32_t kSlidingKvHeads = 2u;
	static constexpr uint32_t kFullKvHeads = 1u;
	static constexpr uint32_t kWindow = 24u;
	static const char *Tag(void) { return "gemma4-31b"; }
};

struct Small26b : Gemma4_26bGeometry
{
	static constexpr uint32_t kTpDegree = 1u;
	static constexpr uint32_t kHidden = 512u;
	static constexpr uint32_t kLayers = 6u;
	static constexpr uint32_t kSlidingHeads = 4u;
	static constexpr uint32_t kFullHeads = 4u;
	static constexpr uint32_t kSlidingKvHeads = 2u;
	static constexpr uint32_t kFullKvHeads = 1u;
	static constexpr uint32_t kWindow = 24u;
	static constexpr uint32_t kExperts = 16u;
	static constexpr uint32_t kLocalExperts = 8u;
	static constexpr uint32_t kMoeInter = 128u;
	static const char *Tag(void) { return "gemma4-26b"; }
};

struct SmallMuse : MuseGlimmerGeometry
{
	static constexpr uint32_t kTpDegree = 1u;
	static constexpr uint32_t kHidden = 512u;
	static constexpr uint32_t kLayers = 4u;
	static constexpr uint32_t kSlidingHeads = 16u;
	static constexpr uint32_t kFullHeads = 16u;
	static constexpr uint32_t kSlidingKvHeads = 1u;
	static constexpr uint32_t kFullKvHeads = 1u;
	static constexpr uint32_t kWindow = 24u;
	static const char *Tag(void) { return "muse-glimmer"; }
};

struct SmallLaguna : LagunaGeometry
{
	static constexpr uint32_t kTpDegree = 1u;
	static constexpr uint32_t kHidden = 512u;
	static constexpr uint32_t kLayers = 4u;
	static constexpr uint32_t kSlidingHeads = 10u;
	static constexpr uint32_t kFullHeads = 6u;
	static constexpr uint32_t kSlidingKvHeads = 1u;
	static constexpr uint32_t kFullKvHeads = 1u;
	static constexpr uint32_t kWindow = 24u;
	static constexpr uint32_t kExperts = 32u;
	static constexpr uint32_t kLocalExperts = 16u;
	static constexpr uint32_t kMoeInter = 128u;
	static const char *Tag(void) { return "laguna"; }
};

struct SmallKexaone : Kexaone2Geometry
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
	static const char *Tag(void) { return "k-exaone"; }
};

template<class G>
struct VfRig
{
	using M = GemmaModel<G>;
	using S = GemmaShape<G>;
	using State = GemmaState<G>;
	static constexpr uint64_t kSlidingRegion = (uint64_t)S::kSlidingLayers * VF_SLOTS * VF_PAGES * S::SlidingKvGeometry::kPageBytes;
	static constexpr uint64_t kFullRegion = (uint64_t)S::kFullLayers * VF_SLOTS * VF_PAGES * S::FullKvGeometry::kPageBytes;
	State *state;
	VfBus bus;
	SparkStageRunnerServices services;
	cudaStream_t stream;
	uint32_t multiprocessors;
	uint8_t *kv;
	uint32_t *page_table, *positions, *context, *sequence_of_row;
	uint16_t *hidden;
	uint32_t host_context[VF_SLOTS];

	void Layer(GemmaLayer *layer, uint32_t index)
	{
		const bool full = S::Full(index);
		const uint32_t fused = full ? S::kFullFused : S::kSlidingFused, q = full ? S::kFullQ : S::kSlidingQ, dim = full ? G::kFullDim : G::kSlidingDim;
		layer->qkv = Matrix(fused, G::kHidden);
		layer->out = Matrix(G::kHidden, q);
		layer->gate_up = Matrix(2u * VF_FFN, G::kHidden);
		layer->down = Matrix(G::kHidden, VF_FFN);
		layer->ffn_rows = VF_FFN;
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
			cudaMalloc((void **)&kv, kSlidingRegion + kFullRegion) != cudaSuccess || cudaMemset(kv, 0, kSlidingRegion + kFullRegion) != cudaSuccess ||
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
		attach.layer_stride_bytes = (uint64_t)VF_SLOTS * VF_PAGES * S::SlidingKvGeometry::kPageBytes;
		attach.layer_page_bytes = S::SlidingKvGeometry::kPageBytes;
		attach.layer_count = S::kSlidingLayers;
		attach.page_table = page_table;
		attach.page_table_stride = VF_PAGES;
		attach.pool_page_count = VF_SLOTS * VF_PAGES;
		attach.sequence_count = VF_SLOTS;
		attach.context_shard.degree = 1u;
		attach.context_shard.grain = 1u;
		attach.second_pool = kv + kSlidingRegion;
		attach.second_layer_stride_bytes = (uint64_t)VF_SLOTS * VF_PAGES * S::FullKvGeometry::kPageBytes;
		attach.second_layer_page_bytes = S::FullKvGeometry::kPageBytes;
		attach.second_layer_count = S::kFullLayers;
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
			if ( seen[rows[row].sequence] == 0u )
				sequences++;
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
		step.context = VF_PAGES * G::kPageSlots;
		step.multiprocessors = multiprocessors;
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
		std::vector<uint8_t> copy(kSlidingRegion + kFullRegion);
		if ( cudaMemcpy(copy.data(), kv, copy.size(), cudaMemcpyDeviceToHost) != cudaSuccess )
			copy.clear();
		return copy;
	}

	int Load(const std::vector<uint8_t> &copy)
	{
		return copy.size() == kSlidingRegion + kFullRegion && cudaMemcpy(kv, copy.data(), copy.size(), cudaMemcpyHostToDevice) == cudaSuccess;
	}

	template<class Kv>
	static uint32_t RegionMatches(const uint8_t *a, const uint8_t *b, uint32_t layers, uint32_t limit)
	{
		for ( uint32_t layer = 0u; layer < layers; ++layer )
			for ( uint32_t sequence = 0u; sequence < VF_SLOTS; ++sequence )
				for ( uint32_t position = 0u; position < VF_PAGES * Kv::kPageSlots; ++position )
				{
					const uint64_t offset = ((uint64_t)layer * VF_SLOTS * VF_PAGES + sequence * VF_PAGES + position / Kv::kPageSlots) * Kv::kPageBytes +
						(uint64_t)(position % Kv::kPageSlots) * Kv::kSlotBytes;
					if ( sequence == 0u && position >= limit )
						continue;
					if ( memcmp(a + offset, b + offset, Kv::kSlotBytes) != 0 )
						return 0u;
				}
		return 1u;
	}

	static uint32_t KvMatches(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b, uint32_t limit)
	{
		return a.size() == b.size() && RegionMatches<typename S::SlidingKvGeometry>(a.data(), b.data(), S::kSlidingLayers, limit) != 0u &&
			RegionMatches<typename S::FullKvGeometry>(a.data() + kSlidingRegion, b.data() + kSlidingRegion, S::kFullLayers, limit) != 0u ? 1u : 0u;
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
		Require(0, model, "the synthetic stage opens, binds and attaches its KV pools", 0u, 0u, 0u);
		return;
	}
	for ( uint32_t sequence = 0u; sequence < VF_SLOTS; ++sequence )
	{
		rows.clear();
		for ( uint32_t position = 0u; position < (sequence == 0u ? VF_PREFIX : VF_OTHER_PREFIX); ++position )
			rows.push_back({ sequence, position, 0u });
		Require(rig.Run(rows.data(), (uint32_t)rows.size(), 1u, 0) != 0, model, "the prefix prefill runs", sequence, 0u, 0u);
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
	rig.Close();
}

int main(void)
{
	Check<Small31b>();
	Check<Small26b>();
	Check<SmallMuse>();
	Check<SmallLaguna>();
	Check<SmallKexaone>();
	if ( vf_failures != 0 )
	{
		printf("FAIL gemma4 verify frames: %d failures\n", vf_failures);
		return 1;
	}
	printf("PASS gemma4 verify frames: every row of every verify frame equals one-row decode bit for bit, and a rejected tail leaves no trace\n");
	return 0;
}
