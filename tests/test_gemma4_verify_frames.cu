#include "modules/gemma4_31b_resident_decode_stage/source/spark_gemma4_31b_stage_model.cu"
#include "modules/gemma4_26b_resident_decode_stage/source/spark_gemma4_26b_stage_model.cu"
#include "modules/muse_glimmer_tp16_resident_decode_stage/source/spark_muse_tp16_stage_model.cu"
#include "modules/laguna_tp16_resident_decode_stage/source/spark_laguna_tp16_stage_model.cu"
#include "modules/kexaone2_tp16_resident_decode_stage/source/spark_kexaone2_tp16_stage_model.cu"

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

struct Small31b : Gemma4_31bGeometry<SPARK_GEMMA4_31B_STAGE_TP_DEGREE>
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

struct Small26b : Gemma4_26bGeometry<SPARK_GEMMA4_26B_STAGE_TP_DEGREE>
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

struct SmallMuse : MuseGlimmerGeometry<SPARK_MUSE_TP16_STAGE_TP_DEGREE>
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

struct SmallLaguna : LagunaGeometry<SPARK_LAGUNA_TP16_STAGE_TP_DEGREE>
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

struct SmallKexaone : Kexaone2Geometry<SPARK_KEXAONE2_TP16_STAGE_TP_DEGREE>
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
	uint32_t *tree_slots, *forks, *offsets, *row_indices, *kept;
	uint16_t *hidden;
	uint32_t host_context[VF_SLOTS];
	SparkStageRunnerStep last;

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
			cudaMalloc((void **)&tree_slots, VF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&forks, VF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&offsets, VF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&row_indices, VF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
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

	int Run(const VfRow *rows, uint32_t count, uint32_t commit, std::vector<uint16_t> *output, uint32_t last_rows_only = 0u, uint32_t gather_context = 0u)
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
		step.last_rows_only = last_rows_only;
		step.gather_context = gather_context;
		step.host_positions = position.data();
		step.host_sequence_of_row = sequence.data();
		step.context = VF_PAGES * G::kPageSlots;
		step.multiprocessors = multiprocessors;
		step.route_fault = VfRouteFault();
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
		step.context = VF_PAGES * G::kPageSlots;
		step.multiprocessors = multiprocessors;
		step.route_fault = VfRouteFault();
		step.tree = 1u;
		step.slots = tree_slots;
		step.fork_positions = forks;
		step.slot_offsets = offsets;
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
		cudaFree(tree_slots);
		cudaFree(forks);
		cudaFree(offsets);
		cudaFree(row_indices);
		cudaFree(kept);
		cudaFree(context);
		cudaFree(hidden);
		cudaStreamDestroy(stream);
	}
};

static float RelativeRowDistance(const std::vector<uint16_t> &a, uint32_t a_row, const std::vector<uint16_t> &b, uint32_t b_row, uint32_t width)
{
	float worst = 0.0f, scale = 1.0f;
	for ( uint32_t index = 0u; index < width; ++index )
	{
		const float va = VfFloat(a[(uint64_t)a_row * width + index]), vb = VfFloat(b[(uint64_t)b_row * width + index]);
		const float diff = fabsf(va - vb), mag = fabsf(vb);
		worst = diff > worst ? diff : worst;
		scale = mag > scale ? mag : scale;
	}
	return worst / scale;
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
		Require(Finite(tree_out) != 0u, model, "tree verify rows are finite and not constant", 0u, 0u, 0u);
		for ( uint32_t path = 0u; path < 3u; ++path )
			for ( uint32_t k = 0u; k < lengths[path]; ++k )
			{
				if ( path == 0u )
					Require(memcmp(tree_out.data() + (uint64_t)paths[path][k] * G::kHidden, seq_out[path].data() + (uint64_t)k * G::kHidden,
							G::kHidden * sizeof(uint16_t)) == 0, model, "trunk tree node equals its sequential decode bit for bit (path, depth, row)",
						path, k, paths[path][k]);
				else
				{
					const float dist = RelativeRowDistance(tree_out, paths[path][k], seq_out[path], k, G::kHidden);
					Require(dist < 0.02f, model, "branch tree node matches sequential decode up to tiled GQA summation order (path, row, rel x 1e6)",
						path, paths[path][k], (uint32_t)(dist * 1.0e6f));
				}
			}
		Require(RelativeRowDistance(tree_out, 5u, tree_out, 2u, G::kHidden) >= 0.1f,
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
			if ( path == 0u )
				Require(rig.KvMatches(rig.Save(), reference, VF_PREFIX + length) != 0u, model,
					"relocated trunk KV matches sequential decode bit for bit", path, 0u, 0u);
			else
			{
				const std::vector<uint8_t> kept_kv = rig.Save();
				if constexpr ( !GemmaShape<G>::Full(0u) )
					Require(VfRig<G>::template RegionMatches<typename GemmaShape<G>::SlidingKvGeometry>(kept_kv.data(), reference.data(), 1u, VF_PREFIX + length) != 0u,
						model, "layer-0 relocated branch KV matches sequential decode bit for bit", path, 0u, 0u);
				else
					Require(VfRig<G>::template RegionMatches<typename GemmaShape<G>::FullKvGeometry>(kept_kv.data() + VfRig<G>::kSlidingRegion,
							reference.data() + VfRig<G>::kSlidingRegion, 1u, VF_PREFIX + length) != 0u,
						model, "layer-0 relocated branch KV matches sequential decode bit for bit", path, 0u, 0u);
			}
			Require(rig.Decode(0u, VF_PREFIX + length, 1u, &speculated) != 0 && rig.Decode(0u, VF_PREFIX + length + 1u, 2u, &speculated) != 0,
				model, "follow-up decodes after tree KeepRows run", path, 0u, 0u);
			speculated_kv = rig.Save();
			if ( path == 0u )
			{
				Require(FirstDifference(speculated, plain, G::kHidden) == UINT32_MAX, model,
					"follow-up decodes after trunk KeepRows equal plain run bit for bit (path, first differing step)", path, FirstDifference(speculated, plain, G::kHidden), 0u);
				Require(rig.KvMatches(speculated_kv, plain_kv, limit) != 0u, model,
					"follow-up KV after trunk KeepRows equals plain run bit for bit", path, 0u, 0u);
			}
			else
				for ( uint32_t s = 0u; s < 2u; ++s )
				{
					const float dist = RelativeRowDistance(speculated, s, plain, s, G::kHidden);
					Require(dist < 0.02f, model,
						"follow-up decodes after branch KeepRows match plain run up to tiled GQA summation order (path, step, rel x 1e6)",
						path, s, (uint32_t)(dist * 1.0e6f));
				}
		}
		rig.Load(prefix);
		Require(rig.RunTree(tree_b, parents_b, kTreeRows, 16u, &eager_b) != 0, model, "eager tree B runs for graph check", 0u, 0u, 0u);
		rig.Load(prefix);
		Require(rig.UploadTree(tree_a, parents_a, kTreeRows, 16u, &step_a) != 0 &&
			cudaStreamSynchronize(rig.stream) == cudaSuccess &&
			cudaStreamBeginCapture(rig.stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess &&
			GemmaModel<G>::Step(rig.state, &step_a, rig.stream) == SPARK_STATUS_OK &&
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
	rig.Close();
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
	std::vector<uint8_t> cached_kv;
	if ( rig.Open() == 0 )
	{
		Require(0, model, "prefix-exact synthetic stage opens", 0u, 0u, 0u);
		return;
	}
	for ( uint32_t position = 0u; position < kMaxTotal; ++position )
		all_rows[position] = { 0u, position, 0u };
	Require(rig.Run(all_rows.data(), kCached, 1u, 0, 1u, kCached) != 0, model, "cached prefix prefill runs", kCached, 0u, 0u);
	cached_kv = rig.Save();
	for ( uint32_t delta : deltas )
	{
		const uint32_t total = kCached + delta;
		std::vector<uint16_t> cold_out, warm_out;
		std::vector<uint8_t> cold_kv;
		Require(cudaMemset(rig.kv, 0, rig.kSlidingRegion + rig.kFullRegion) == cudaSuccess, model, "prefix-exact KV reset succeeds", delta, 0u, 0u);
		Require(rig.Run(all_rows.data(), total, 1u, &cold_out, 1u, total) != 0, model, "cold prefill runs", total, delta, 0u);
		cold_kv = rig.Save();
		Require(Finite(cold_out) != 0u, model, "cold prefill rows are finite", total, delta, 0u);
		std::vector<uint16_t> cold_tail(cold_out.begin() + (ptrdiff_t)((uint64_t)kCached * G::kHidden), cold_out.end());
		Require(rig.Load(cached_kv) != 0, model, "cached KV restore succeeds", delta, 0u, 0u);
		Require(rig.Run(all_rows.data() + kCached, delta, 1u, &warm_out, 1u, total) != 0, model, "warm delta prefill runs", kCached, delta, 0u);
		Require(FirstDifference(warm_out, cold_tail, G::kHidden) == UINT32_MAX, model,
			"warm delta prefill equals cold prefill bit for bit (delta, first differing row)", delta, FirstDifference(warm_out, cold_tail, G::kHidden), 0u);
		Require(rig.KvMatches(rig.Save(), cold_kv, total) != 0u, model,
			"warm delta prefill leaves the same KV as cold prefill bit for bit (delta)", delta, 0u, 0u);
	}
	{
		std::vector<uint16_t> ref_out, pref_out;
		const uint32_t check_len = 72u;
		Require(cudaMemset(rig.kv, 0, rig.kSlidingRegion + rig.kFullRegion) == cudaSuccess, model, "prefill accuracy KV reset succeeds", 0u, 0u, 0u);
		Require(rig.Run(all_rows.data(), check_len, 1u, &ref_out, 1u, 0u) != 0, model, "reference GQA prefill runs", check_len, 0u, 0u);
		Require(cudaMemset(rig.kv, 0, rig.kSlidingRegion + rig.kFullRegion) == cudaSuccess, model, "prefill accuracy KV reset 2 succeeds", 0u, 0u, 0u);
		Require(rig.Run(all_rows.data(), check_len, 1u, &pref_out, 1u, check_len) != 0, model, "local-head prefill run succeeds", check_len, 0u, 0u);
		uint32_t moe_flips = 0u;
		for ( uint32_t p = 0u; p < check_len; ++p )
		{
			const float dist = RelativeRowDistance(pref_out, p, ref_out, p, G::kHidden);
			if constexpr ( G::kMoe )
			{
				if ( dist >= 0.02f )
					moe_flips++;
			}
			else
				Require(dist < 0.02f, model, "local-head prefill matches reference GQA prefill (pos, rel x 1e6)", p, (uint32_t)(dist * 1.0e6f), 0u);
		}
		if constexpr ( G::kMoe )
			Require(moe_flips <= 2u, model, "local-head prefill matches reference GQA prefill modulo synthetic router ties (flips)", moe_flips, 0u, 0u);
	}
	{
		std::vector<uint16_t> b1_out, b8_out;
		VfRow b8_rows[8];
		Require(rig.Load(cached_kv) != 0, model, "B1==B8 KV restore succeeds", 0u, 0u, 0u);
		VfRow b1_row = { 0u, kCached, 5u };
		Require(rig.Run(&b1_row, 1u, 1u, &b1_out, 0u, 0u) != 0, model, "B1 decode runs", 0u, 0u, 0u);
		Require(rig.Load(cached_kv) != 0, model, "B8 KV restore succeeds", 0u, 0u, 0u);
		b8_rows[0] = b1_row;
		for ( uint32_t r = 1u; r < 8u; ++r )
			b8_rows[r] = { 1u, r - 1u, 20u + r };
		Require(rig.Run(b8_rows, 8u, 1u, &b8_out, 0u, 0u) != 0, model, "B8 decode runs", 0u, 0u, 0u);
		Require(memcmp(b1_out.data(), b8_out.data(), G::kHidden * sizeof(uint16_t)) == 0, model,
			"B8 lane0 equals B1 decode bit for bit", 0u, 0u, 0u);
	}
	{
		const uint32_t len0 = 42u, len1 = 36u;
		std::vector<VfRow> solo_pf(len0), mix_pf, warm_seed(kCached);
		std::vector<uint16_t> solo_pf_out, solo_dec_out, mix_pf_out, mix_dec_out;
		for ( uint32_t r = 0u; r < len0; ++r )
		{
			solo_pf[r] = { 0u, r, 61u };
			mix_pf.push_back(solo_pf[r]);
		}
		for ( uint32_t r = 0u; r < kCached; ++r )
			warm_seed[r] = { 1u, r, 55u };
		for ( uint32_t r = 0u; r < len1; ++r )
			mix_pf.push_back({ 1u, kCached + r, 73u });
		VfRow solo_dec = { 0u, len0, 91u };
		VfRow mix_dec[2] = { solo_dec, { 1u, kCached + len1, 103u } };
		Require(cudaMemset(rig.kv, 0, rig.kSlidingRegion + rig.kFullRegion) == cudaSuccess, model, "RAIS solo KV reset succeeds", 0u, 0u, 0u);
		memset(rig.host_context, 0, sizeof(rig.host_context));
		Require(rig.Run(solo_pf.data(), len0, 1u, &solo_pf_out, 1u, len0) != 0, model, "RAIS solo prefill wave runs", len0, 0u, 0u);
		Require(rig.Run(&solo_dec, 1u, 1u, &solo_dec_out, 0u, 0u) != 0, model, "RAIS solo post-prefill decode runs", len0, 0u, 0u);
		Require(cudaMemset(rig.kv, 0, rig.kSlidingRegion + rig.kFullRegion) == cudaSuccess, model, "RAIS mix KV reset succeeds", 0u, 0u, 0u);
		memset(rig.host_context, 0, sizeof(rig.host_context));
		Require(rig.Run(warm_seed.data(), kCached, 1u, 0, 1u, kCached) != 0, model, "RAIS warm lane1 seed runs", kCached, 0u, 0u);
		Require(rig.Run(mix_pf.data(), (uint32_t)mix_pf.size(), 1u, &mix_pf_out, 1u, 0u) != 0, model, "RAIS mixed cold+warm prefill wave runs", (uint32_t)mix_pf.size(), 0u, 0u);
		Require(rig.Run(mix_dec, 2u, 1u, &mix_dec_out, 0u, 0u) != 0, model, "RAIS mixed post-prefill decode runs", 2u, 0u, 0u);
		Require(FirstDifference(solo_pf_out, std::vector<uint16_t>(mix_pf_out.begin(), mix_pf_out.begin() + (ptrdiff_t)solo_pf_out.size()), G::kHidden) == UINT32_MAX,
			model, "RAIS mixed prefill wave lane0 equals solo prefill wave bit for bit (first diff row)",
			FirstDifference(solo_pf_out, std::vector<uint16_t>(mix_pf_out.begin(), mix_pf_out.begin() + (ptrdiff_t)solo_pf_out.size()), G::kHidden), 0u, 0u);
		Require(memcmp(solo_dec_out.data(), mix_dec_out.data(), G::kHidden * sizeof(uint16_t)) == 0, model,
			"RAIS mixed post-prefill decode lane0 equals B1 decode bit for bit", 0u, 0u, 0u);
	}
	Require(rig.KvClean() != 0u, model, "prefix-exact steps reported no KV access error", 0u, 0u, 0u);
	printf("%s %s prefix-exact + B1==B8: cached=%u vs warm deltas 1..128\n", vf_failures == 0 ? "ok" : "checked", model, kCached);
	rig.Close();
}

static void CheckMuseTp16SplitMultiInvariance(void)
{
	constexpr uint32_t K = 6656u, N0 = 288u, N1 = 256u, B8 = 8u;
	cudaStream_t stream = 0;
	uint16_t *w0 = 0, *w1 = 0, *act = 0, *out0_b1 = 0, *out1_b1 = 0, *out0_b8 = 0, *out1_b8 = 0, *ref0 = 0, *ref1 = 0;
	std::vector<uint16_t> h_w0((size_t)N0 * K), h_w1((size_t)N1 * K), h_act((size_t)B8 * K);
	std::vector<uint16_t> h_o0_b1(N0), h_o1_b1(N1), h_o0_b8(N0), h_o1_b8(N1), h_r0(N0), h_r1(N1);
	for ( size_t i = 0; i < h_w0.size(); ++i ) h_w0[i] = Bf16(Uniform(11u, (uint32_t)i) * 0.05f);
	for ( size_t i = 0; i < h_w1.size(); ++i ) h_w1[i] = Bf16(Uniform(23u, (uint32_t)i) * 0.05f);
	for ( size_t i = 0; i < h_act.size(); ++i ) h_act[i] = Bf16(Uniform(37u, (uint32_t)i) * 0.5f);
	Require(cudaStreamCreate(&stream) == cudaSuccess &&
		cudaMalloc(&w0, h_w0.size() * sizeof(uint16_t)) == cudaSuccess &&
		cudaMalloc(&w1, h_w1.size() * sizeof(uint16_t)) == cudaSuccess &&
		cudaMalloc(&act, h_act.size() * sizeof(uint16_t)) == cudaSuccess &&
		cudaMalloc(&out0_b1, N0 * sizeof(uint16_t)) == cudaSuccess &&
		cudaMalloc(&out1_b1, N1 * sizeof(uint16_t)) == cudaSuccess &&
		cudaMalloc(&out0_b8, (size_t)B8 * N0 * sizeof(uint16_t)) == cudaSuccess &&
		cudaMalloc(&out1_b8, (size_t)B8 * N1 * sizeof(uint16_t)) == cudaSuccess &&
		cudaMalloc(&ref0, N0 * sizeof(uint16_t)) == cudaSuccess &&
		cudaMalloc(&ref1, N1 * sizeof(uint16_t)) == cudaSuccess &&
		cudaMemcpy(w0, h_w0.data(), h_w0.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) == cudaSuccess &&
		cudaMemcpy(w1, h_w1.data(), h_w1.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) == cudaSuccess &&
		cudaMemcpy(act, h_act.data(), h_act.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) == cudaSuccess,
		"muse-tp16-split-multi", "buffers allocate", 0u, 0u, 0u);
	const LmSkinnyDenseTarget t_b1[2] =
	{
		{ w0, out0_b1, (float *)0, N0, N0, 0u, LmScaleTensorNone() },
		{ w1, out1_b1, (float *)0, N1, N1, 0u, LmScaleTensorNone() }
	};
	const LmSkinnyDenseTarget t_b8[2] =
	{
		{ w0, out0_b8, (float *)0, N0, N0, 0u, LmScaleTensorNone() },
		{ w1, out1_b8, (float *)0, N1, N1, 0u, LmScaleTensorNone() }
	};
	Require(LmSkinnyDenseMulti<LmBf16Format>(t_b1, 2u, act, 1u, K, stream, 0u, 2u) == LM_LAUNCH_OK &&
		LmSkinnyDenseMulti<LmBf16Format>(t_b8, 2u, act, B8, K, stream, 0u, 2u) == LM_LAUNCH_OK &&
		LmDenseRows<LmBf16Format>(w0, LmScaleTensorNone(), act, ref0, (float *)0, 1u, K, N0, 0u, 48u, stream) == LM_LAUNCH_OK &&
		LmDenseRows<LmBf16Format>(w1, LmScaleTensorNone(), act, ref1, (float *)0, 1u, K, N1, 0u, 48u, stream) == LM_LAUNCH_OK &&
		cudaStreamSynchronize(stream) == cudaSuccess &&
		cudaMemcpy(h_o0_b1.data(), out0_b1, N0 * sizeof(uint16_t), cudaMemcpyDeviceToHost) == cudaSuccess &&
		cudaMemcpy(h_o1_b1.data(), out1_b1, N1 * sizeof(uint16_t), cudaMemcpyDeviceToHost) == cudaSuccess &&
		cudaMemcpy(h_o0_b8.data(), out0_b8, N0 * sizeof(uint16_t), cudaMemcpyDeviceToHost) == cudaSuccess &&
		cudaMemcpy(h_o1_b8.data(), out1_b8, N1 * sizeof(uint16_t), cudaMemcpyDeviceToHost) == cudaSuccess &&
		cudaMemcpy(h_r0.data(), ref0, N0 * sizeof(uint16_t), cudaMemcpyDeviceToHost) == cudaSuccess &&
		cudaMemcpy(h_r1.data(), ref1, N1 * sizeof(uint16_t), cudaMemcpyDeviceToHost) == cudaSuccess,
		"muse-tp16-split-multi", "Split-K multi projections run", 0u, 0u, 0u);
	Require(memcmp(h_o0_b1.data(), h_o0_b8.data(), N0 * sizeof(uint16_t)) == 0 &&
		memcmp(h_o1_b1.data(), h_o1_b8.data(), N1 * sizeof(uint16_t)) == 0 &&
		memcmp(h_o0_b1.data(), h_r0.data(), N0 * sizeof(uint16_t)) == 0 &&
		memcmp(h_o1_b1.data(), h_r1.data(), N1 * sizeof(uint16_t)) == 0,
		"muse-tp16-split-multi", "Muse TP16 qkv(288)+gate(256) Split-K B1 == B8_lane0 == single-target bit-exact", 0u, 0u, 0u);
	printf("ok muse-tp16-split-multi: K=%u qkv=%u+gate=%u Split-K B1==B8 bit-exact\n", K, N0, N1);
	cudaFree(w0); cudaFree(w1); cudaFree(act); cudaFree(out0_b1); cudaFree(out1_b1);
	cudaFree(out0_b8); cudaFree(out1_b8); cudaFree(ref0); cudaFree(ref1);
	cudaStreamDestroy(stream);
}

int main(void)
{
	CheckMuseTp16SplitMultiInvariance();
	Check<Small31b>();
	Check<Small26b>();
	Check<SmallMuse>();
	Check<SmallLaguna>();
	Check<SmallKexaone>();
	CheckPrefixExact<Small31b>();
	CheckPrefixExact<Small26b>();
	CheckPrefixExact<SmallMuse>();
	CheckPrefixExact<SmallLaguna>();
	CheckPrefixExact<SmallKexaone>();
	if ( vf_failures != 0 )
	{
		printf("FAIL gemma4 verify frames: %d failures\n", vf_failures);
		return 1;
	}
	printf("PASS gemma4 verify frames: every row of every verify frame equals one-row decode bit for bit, and a rejected tail leaves no trace\n");
	return 0;
}
