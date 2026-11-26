#include "modules/motif3_tp16_resident_decode_stage/source/spark_motif3_tp16_stage_model.cu"

#include "sparkpipe/spark_model_serving_adapter.h"
#include "sparkpipe/spark_speculation_tree_frame.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "tests/cuda/verify_frames_common.cuh"

#define MVF_MAX_FRAME SPARK_MODEL_SERVING_ADAPTER_MAX_VERIFY_TOKENS_PER_SEQUENCE
#define MVF_SLOTS 8u
#define MVF_PAGES 8u
#define MVF_PREFIX 160u
#define MVF_OTHER_PREFIX 100u

struct MotifTestGeom
{
	static constexpr uint32_t kTpDegree = 16u;
	static constexpr uint32_t kPackAlignment = 128u;
	static const char *Tag(void) { return "motif3-test"; }
	static const char *WeightdModel(void) { return "motif3_test"; }
	static const char *WeightdRevision(void) { return "test"; }
	static const char *ReportTag(void) { return "MOTIF3-TEST"; }
};

struct MotifTestRig
{
	using G = MotifTestGeom;
	using M = MotifStage<G>;
	using State = MotifState<G>;
	static constexpr uint32_t kTestLayers = 6u;
	static constexpr uint64_t kKvRegion = (uint64_t)MVF_SLOTS * MVF_PAGES * MotifKv::kPageBytes;
	State *state;
	VfBus bus;
	SparkStageRunnerServices services;
	cudaStream_t stream;
	uint32_t multiprocessors;
	uint8_t *kv;
	uint32_t *page_table, *positions, *context, *sequence_of_row;
	uint32_t *tree_slots, *forks, *offsets, *row_indices, *kept;
	uint16_t *hidden;
	uint32_t host_context[MVF_SLOTS];
	SparkStageRunnerStep last;

	int Open(void)
	{
		std::vector<uint32_t> table(MVF_SLOTS * MVF_PAGES);
		SparkStageRunnerKv attach;
		memset(this, 0, sizeof(*this));
		bus.hidden = MO_HIDDEN;
		services.context = &bus;
		services.round = VfRound;
		state = (State *)calloc(1u, sizeof(State));
		state->services = &services;
		state->tp_degree = 1u;
		state->tp_rank = 0u;
		state->max_rows = MVF_MAX_FRAME;
		state->slots = MVF_SLOTS;
		state->expert_first = 0u;
		if ( M::Size(state) != SPARK_STATUS_OK )
			return 0;
		const float *w_norm = Norm(MO_HIDDEN);
		const float *w_hc_fn = Tensor<float>((uint64_t)LM_HC_SITE_MIX * LM_HC_SITE_STREAMS * MO_HIDDEN, 0.02f, 0.0f);
		const float *w_hc_sc = Tensor<float>(3u, 0.1f, 0.05f);
		const float *w_hc_ba = Tensor<float>(LM_HC_SITE_MIX, 0.1f, 0.0f);
		const uint16_t *w_qa = Matrix(MO_QA, MO_HIDDEN);
		const float *w_qn = Norm(MO_QA);
		const uint16_t *w_kva = Matrix(MO_SLOT, MO_HIDDEN);
		const float *w_kvn = Norm(MO_LATENT);
		const uint16_t *w_lam = Matrix(MO_GROUPS * MO_SIGNAL, MO_HIDDEN);
		const uint16_t *w_qb = Matrix(State::kHeads * MO_HEAD, MO_QA);
		const uint16_t *w_gate = Matrix(State::kSignal * MO_VALUE, MO_QA);
		const uint16_t *w_out = Matrix(MO_HIDDEN, State::kSignal * MO_VALUE);
		const uint16_t *w_uk = Matrix((uint64_t)State::kHeads * MO_LATENT, MO_NOPE);
		const uint16_t *w_uv = Matrix((uint64_t)State::kHeads * MO_VALUE, MO_LATENT);
		const uint16_t *w_dgu = Matrix(2u * MO_DENSE_INTER, MO_HIDDEN);
		const uint16_t *w_ddn = Matrix(MO_HIDDEN, MO_DENSE_INTER);
		const float *w_dpoly = Tensor<float>(LM_POLY_NORM_PARAMS, 0.2f, 0.1f);
		const uint16_t *w_router = Matrix(MO_EXPERTS, MO_HIDDEN);
		const float *w_ebias = Tensor<float>(MO_EXPERTS, 0.05f, 0.0f);
		const uint16_t *w_m13 = Matrix((uint64_t)State::kLocalExperts * 2u * MO_EXPERT_INTER, MO_HIDDEN);
		const uint16_t *w_m2 = Matrix((uint64_t)State::kLocalExperts * MO_HIDDEN, MO_EXPERT_INTER);
		const float *w_mpoly = Tensor<float>((uint64_t)State::kLocalExperts * LM_POLY_NORM_PARAMS, 0.2f, 0.1f);
		const uint16_t *w_sgu = Matrix(2u * MO_EXPERT_INTER, MO_HIDDEN);
		const uint16_t *w_sdn = Matrix(MO_HIDDEN, MO_EXPERT_INTER);
		const float *w_spoly = Tensor<float>(LM_POLY_NORM_PARAMS, 0.2f, 0.1f);
		for ( uint32_t i = 0u; i < MO_LAYERS; ++i )
		{
			MotifLayer *l = &state->layers[i];
			l->attn_norm = w_norm;
			l->ffn_norm = w_norm;
			l->hc_attn_fn = w_hc_fn;
			l->hc_attn_scale = w_hc_sc;
			l->hc_attn_base = w_hc_ba;
			l->hc_ffn_fn = w_hc_fn;
			l->hc_ffn_scale = w_hc_sc;
			l->hc_ffn_base = w_hc_ba;
			l->q_a = w_qa;
			l->q_norm = w_qn;
			l->kv_a = w_kva;
			l->kv_norm = w_kvn;
			l->lambda = w_lam;
			l->q_b = w_qb;
			l->gate = w_gate;
			l->out = w_out;
			l->uk = w_uk;
			l->uv = w_uv;
			l->ffn_gate_up = w_dgu;
			l->ffn_down = w_ddn;
			l->ffn_poly = w_dpoly;
			l->router = w_router;
			l->expert_bias = w_ebias;
			l->moe_w13 = w_m13;
			l->moe_w2 = w_m2;
			l->moe_poly = w_mpoly;
			l->shared_gate_up = w_sgu;
			l->shared_down = w_sdn;
			l->shared_poly = w_spoly;
			l->window = MotifLayout::Window(i) ? 1u : 0u;
			l->owner = 1u;
		}
		if ( M::Allocate(state) != SPARK_STATUS_OK || cudaStreamCreate(&stream) != cudaSuccess ||
			cudaDeviceGetAttribute((int *)&multiprocessors, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess ||
			cudaMalloc((void **)&kv, kKvRegion * kTestLayers) != cudaSuccess || cudaMemset(kv, 0, kKvRegion * kTestLayers) != cudaSuccess ||
			cudaMalloc((void **)&page_table, table.size() * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&positions, MVF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&sequence_of_row, MVF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&tree_slots, MVF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&forks, MVF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&offsets, MVF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&row_indices, MVF_MAX_FRAME * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&kept, MVF_SLOTS * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&context, MVF_SLOTS * sizeof(uint32_t)) != cudaSuccess ||
			cudaMalloc((void **)&hidden, (uint64_t)MVF_MAX_FRAME * MO_HIDDEN * sizeof(uint16_t)) != cudaSuccess )
			return 0;
		for ( uint32_t index = 0u; index < table.size(); ++index )
			table[index] = index;
		if ( cudaMemcpy(page_table, table.data(), table.size() * sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess )
			return 0;
		memset(&attach, 0, sizeof(attach));
		attach.pool = kv;
		attach.layer_stride_bytes = 0u;
		attach.layer_page_bytes = MotifKv::kPageBytes;
		attach.layer_count = MO_LAYERS;
		attach.page_table = page_table;
		attach.page_table_stride = MVF_PAGES;
		attach.pool_page_count = MVF_SLOTS * MVF_PAGES;
		attach.sequence_count = MVF_SLOTS;
		attach.context_shard.degree = 1u;
		attach.context_shard.grain = 1u;
		if ( M::AttachKv(state, &attach) != SPARK_STATUS_OK )
			return 0;
		for ( uint32_t layer = 0u; layer < kTestLayers; ++layer )
			state->views[layer].pages.pool = kv + (uint64_t)layer * kKvRegion;
		return 1;
	}

	void Input(const VfRow &row, uint16_t *target) const
	{
		const uint32_t seed = Mix(row.sequence * 7919u + row.variant * 104729u) ^ row.position;
		for ( uint32_t index = 0u; index < MO_HIDDEN; ++index )
			target[index] = Bf16(Uniform(seed, index));
	}

	int StepLayers(const SparkStageRunnerStep *step)
	{
		typename M::Frame frame;
		SparkStatus status = M::Begin(state, step, stream, &frame);
		if ( status == SPARK_STATUS_OK )
			status = M::HyperBegin(state, frame);
		for ( uint32_t layer = 0u; layer < kTestLayers && status == SPARK_STATUS_OK; ++layer )
		{
			M::BindLayer(state, frame, layer);
			status = M::HyperNorm(state, frame, layer, MOTIF_ATTENTION_PART);
			if ( status == SPARK_STATUS_OK )
				status = MotifLayout::Window(layer) ? M::WindowAttention(state, frame, layer) : M::FullAttention(state, frame, layer);
			if ( status == SPARK_STATUS_OK )
				status = M::AllReduce(state, frame);
			if ( status == SPARK_STATUS_OK )
				status = M::HyperPost(state, frame, layer, MOTIF_ATTENTION_PART);
			if ( status == SPARK_STATUS_OK )
				status = M::HyperNorm(state, frame, layer, MOTIF_FFN_PART);
			if ( status == SPARK_STATUS_OK )
				status = MotifLayout::Routed(layer) ? M::RoutedMlp(state, frame, layer) : M::DenseMlp(state, frame, layer);
			if ( status == SPARK_STATUS_OK )
				status = M::AllReduce(state, frame);
			if ( status == SPARK_STATUS_OK )
				status = M::HyperPost(state, frame, layer, MOTIF_FFN_PART);
		}
		if ( status == SPARK_STATUS_OK )
			status = M::HyperEnd(state, frame);
		return status == SPARK_STATUS_OK ? 1 : 0;
	}

	int Run(const VfRow *rows, uint32_t count, uint32_t commit, std::vector<uint16_t> *output, uint32_t last_rows_only = 0u, uint32_t gather_context = 0u)
	{
		std::vector<uint16_t> input((uint64_t)count * MO_HIDDEN);
		std::vector<uint32_t> position(count), sequence(count);
		uint32_t seen[MVF_SLOTS] = {}, sequences = 0u;
		SparkStageRunnerStep step;
		for ( uint32_t row = 0u; row < count; ++row )
		{
			Input(rows[row], input.data() + (uint64_t)row * MO_HIDDEN);
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
		step.context = MVF_PAGES * MotifKv::kPageSlots;
		step.multiprocessors = multiprocessors;
		step.route_fault = VfRouteFault();
		if ( cudaMemcpyAsync(hidden, input.data(), input.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			cudaMemcpyAsync(positions, position.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			cudaMemcpyAsync(sequence_of_row, sequence.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			cudaMemcpyAsync(context, host_context, sizeof(host_context), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			StepLayers(&step) == 0 || cudaStreamSynchronize(stream) != cudaSuccess )
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

	std::vector<uint8_t> Save(void) const
	{
		std::vector<uint8_t> copy(kKvRegion * kTestLayers);
		if ( cudaMemcpy(copy.data(), kv, copy.size(), cudaMemcpyDeviceToHost) != cudaSuccess )
			copy.clear();
		return copy;
	}

	int Load(const std::vector<uint8_t> &copy)
	{
		return copy.size() == kKvRegion * kTestLayers && cudaMemcpy(kv, copy.data(), copy.size(), cudaMemcpyHostToDevice) == cudaSuccess;
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

int main(void)
{
	static const uint32_t frames[] = { 1u, 2u, 7u, 16u, 17u, 40u };
	MotifTestRig rig;
	Require(rig.Open() != 0, "motif3", "synthetic stage opens and attaches KV", 0u, 0u, 0u);
	std::vector<VfRow> rows;
	for ( uint32_t seq = 0u; seq < MVF_SLOTS; ++seq )
	{
		rows.clear();
		const uint32_t len = seq == 0u ? MVF_PREFIX : MVF_OTHER_PREFIX;
		for ( uint32_t pos = 0u; pos < len; ++pos )
			rows.push_back({ seq, pos, 0u });
		Require(rig.Run(rows.data(), (uint32_t)rows.size(), 1u, 0) != 0, "motif3", "prefix prefill runs", seq, 0u, 0u);
	}
	const std::vector<uint8_t> prefix = rig.Save();

	for ( uint32_t frame : frames )
	{
		std::vector<uint16_t> decoded, verified;
		rig.Load(prefix);
		for ( uint32_t row = 0u; row < frame; ++row )
		{
			const VfRow r = { 0u, MVF_PREFIX + row, 1u };
			Require(rig.Run(&r, 1u, 1u, &decoded) != 0, "motif3", "1-row decode runs", frame, row, 0u);
		}
		rig.Load(prefix);
		rows.clear();
		for ( uint32_t row = 0u; row < frame; ++row )
			rows.push_back({ 0u, MVF_PREFIX + row, 1u });
		Require(rig.Run(rows.data(), frame, 0u, &verified) != 0, "motif3", "verify frame runs", frame, 0u, 0u);
		Require(decoded == verified, "motif3", "verify frame equals 1-row decode bit for bit", frame, 0u, 0u);
	}
	printf("ok motif3: verify frames 1..40 match 1-row decode 0-ULP bit for bit\n");

	{
		std::vector<uint16_t> out_b1, out_b8;
		rig.Load(prefix);
		const VfRow r0 = { 0u, MVF_PREFIX, 7u };
		Require(rig.Run(&r0, 1u, 1u, &out_b1) != 0, "motif3", "B1 decode runs", 0u, 0u, 0u);
		rig.Load(prefix);
		rows.clear();
		rows.push_back(r0);
		for ( uint32_t s = 1u; s < 8u; ++s )
			rows.push_back({ s, MVF_OTHER_PREFIX, 10u + s });
		Require(rig.Run(rows.data(), 8u, 1u, &out_b8) != 0, "motif3", "B8 decode runs", 0u, 0u, 0u);
		const int bit_eq = memcmp(out_b1.data(), out_b8.data(), MO_HIDDEN * sizeof(uint16_t)) == 0;
		Require(bit_eq, "motif3", "5c B1 == B8_lane0 is 0 bit diffs", 0u, 0u, 0u);
		printf("ok motif3: 5c B1 == B8_lane0 = 0 bit diffs\n");
	}

	{
		const uint32_t cached = 128u;
		static const uint32_t deltas[] = { 1u, 2u, 7u, 16u, 32u, 64u, 128u };
		std::vector<uint16_t> cold_full, ref_decode;
		std::vector<uint8_t> zero_kv(MotifTestRig::kKvRegion * MotifTestRig::kTestLayers, 0u);
		rig.Load(zero_kv);
		rows.clear();
		for ( uint32_t p = 0u; p < cached + 128u; ++p )
			rows.push_back({ 0u, p, 3u });
		Require(rig.Run(rows.data(), cached + 128u, 1u, &cold_full, 1u, cached + 128u) != 0, "motif3", "cold unified prefill 256 runs", 0u, 0u, 0u);

		rig.Load(zero_kv);
		for ( uint32_t p = 0u; p < cached + 16u; ++p )
		{
			const VfRow r = { 0u, p, 3u };
			Require(rig.Run(&r, 1u, 1u, &ref_decode) != 0, "motif3", "scalar reference decode runs", p, 0u, 0u);
		}
		const float rel = RelativeRowDistance(cold_full, cached + 15u, ref_decode, cached + 15u, MO_HIDDEN);
		Require(rel < 0.05f, "motif3", "local MMA prefill matches scalar reference", (uint32_t)(rel * 10000.0f), 0u, 0u);

		rig.Load(zero_kv);
		rows.clear();
		for ( uint32_t p = 0u; p < cached; ++p )
			rows.push_back({ 0u, p, 3u });
		Require(rig.Run(rows.data(), cached, 1u, 0, 1u, cached) != 0, "motif3", "cached 128 prefix runs", 0u, 0u, 0u);
		const std::vector<uint8_t> snap128 = rig.Save();
		for ( uint32_t delta : deltas )
		{
			std::vector<uint16_t> warm;
			rig.Load(snap128);
			rows.clear();
			for ( uint32_t i = 0u; i < delta; ++i )
				rows.push_back({ 0u, cached + i, 3u });
			Require(rig.Run(rows.data(), delta, 1u, &warm, 1u, cached + delta) != 0, "motif3", "warm unified prefill runs", delta, 0u, 0u);
			const int gx_eq = memcmp(cold_full.data() + (uint64_t)(cached + delta - 1u) * MO_HIDDEN,
				warm.data() + (uint64_t)(delta - 1u) * MO_HIDDEN, MO_HIDDEN * sizeof(uint16_t)) == 0;
			Require(gx_eq, "motif3", "Gate GX warm prefill matches cold prefill 0 bit diffs", delta, 0u, 0u);
		}
		printf("ok motif3: Gate GX cold(256) == cached(128)+warm(1..128) = 0 bit diffs (rel_ref=%.4f)\n", rel);
	}

	{
		const uint32_t degree = 16u, grain = 4u, win_begin = 32u, old_ctx = 160u;
		const uint32_t prior_keys = old_ctx - win_begin;
		const uint32_t b0 = win_begin / grain, b1 = (old_ctx + grain - 1u) / grain;
		const uint32_t max_peer_keys = ((b1 - b0 + degree - 1u) / degree) * grain;
		const uint64_t slot_bytes = (uint64_t)MO_SLOT * sizeof(uint16_t);
		const uint64_t unit_bytes = (uint64_t)MO_HIDDEN * sizeof(uint16_t);
		const uint32_t ag_units = (uint32_t)(((uint64_t)max_peer_keys * slot_bytes + unit_bytes - 1u) / unit_bytes);
		const uint64_t ag_stride_bytes = (uint64_t)ag_units * unit_bytes;
		const uint64_t shard_page_bytes = MotifKv::kPageBytes / degree;
		uint8_t *pools[16];
		uint32_t *seq0 = 0;
		uint16_t *d_direct = 0, *d_unpacked = 0, *d_send = 0;
		uint8_t *d_gathered = 0;
		cudaMalloc((void **)&seq0, sizeof(uint32_t));
		cudaMemset(seq0, 0, sizeof(uint32_t));
		cudaMalloc((void **)&d_direct, prior_keys * slot_bytes);
		cudaMalloc((void **)&d_unpacked, prior_keys * slot_bytes);
		cudaMalloc((void **)&d_send, ag_stride_bytes);
		cudaMalloc((void **)&d_gathered, (uint64_t)degree * ag_stride_bytes);
		cudaMemset(d_direct, 0, prior_keys * slot_bytes);
		cudaMemset(d_unpacked, 0, prior_keys * slot_bytes);
		cudaMemset(d_gathered, 0, (uint64_t)degree * ag_stride_bytes);
		std::vector<uint8_t> full_kv = rig.Save();
		for ( uint32_t r = 0u; r < degree; ++r )
		{
			std::vector<uint8_t> r_pool(MVF_PAGES * shard_page_bytes, 0u);
			const SparkKvShard sh = { degree, r, grain };
			for ( uint32_t p = 0u; p < old_ctx; ++p )
			{
				if ( SparkKvShardOwns(sh, p) != 0u )
				{
					const uint32_t lp = SparkKvShardLocalKeys(sh, p);
					const uint64_t src_off = (uint64_t)(p / MotifKv::kPageSlots) * MotifKv::kPageBytes + (uint64_t)(p % MotifKv::kPageSlots) * slot_bytes;
					const uint64_t dst_off = (uint64_t)(lp / (MotifKv::kPageSlots / degree)) * shard_page_bytes + (uint64_t)(lp % (MotifKv::kPageSlots / degree)) * slot_bytes;
					memcpy(r_pool.data() + dst_off, full_kv.data() + src_off, slot_bytes);
				}
			}
			cudaMalloc((void **)&pools[r], r_pool.size());
			cudaMemcpy(pools[r], r_pool.data(), r_pool.size(), cudaMemcpyHostToDevice);
			LmKvShardView rv;
			LmKvShardViewInitialize<MotifKv>(&rv, pools[r], rig.page_table, MVF_PAGES, MVF_SLOTS, MVF_SLOTS * MVF_PAGES, rig.state->kv_error, sh);
			cudaMemsetAsync(d_send, 0, ag_stride_bytes, rig.stream);
			(void)LM_LAUNCH_DEPENDENT((MotifPriorKvPackKernel), max_peer_keys, 128u, 0, rig.stream, rv, seq0, win_begin, old_ctx, b0, max_peer_keys, d_send);
			cudaMemcpyAsync(d_gathered + (uint64_t)r * ag_stride_bytes, d_send, ag_stride_bytes, cudaMemcpyDeviceToDevice, rig.stream);
		}
		const SparkKvShard sh0 = { degree, 0u, grain };
		(void)LM_LAUNCH_DEPENDENT((MotifPriorKvUnpackKernel), dim3(max_peer_keys, degree), 128u, 0, rig.stream,
			sh0, d_gathered, ag_stride_bytes, win_begin, old_ctx, b0, max_peer_keys, d_unpacked);
		(void)LM_LAUNCH_DEPENDENT((MotifPriorKvDirectKernel), prior_keys, 128u, 0, rig.stream,
			rig.state->views[0], seq0, win_begin, prior_keys, d_direct);
		cudaStreamSynchronize(rig.stream);
		std::vector<uint8_t> h_dir(prior_keys * slot_bytes), h_unp(prior_keys * slot_bytes);
		cudaMemcpy(h_dir.data(), d_direct, h_dir.size(), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_unp.data(), d_unpacked, h_unp.size(), cudaMemcpyDeviceToHost);
		Require(h_dir == h_unp, "motif3", "TP16 prior KV pack/ALL_GATHER/unpack equals direct 0 bit diffs", 0u, 0u, 0u);
		for ( uint32_t r = 0u; r < degree; ++r )
			cudaFree(pools[r]);
		cudaFree(seq0);
		cudaFree(d_direct);
		cudaFree(d_unpacked);
		cudaFree(d_send);
		cudaFree(d_gathered);
		printf("ok motif3: TP16 prior KV pack/ALL_GATHER/unpack = 0 bit diffs vs direct\n");
	}

	rig.Close();
	printf("PASS motif3 verify frames + 5c + Gate GX\n");
	return 0;
}
