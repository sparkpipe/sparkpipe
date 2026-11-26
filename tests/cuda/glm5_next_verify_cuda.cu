#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "sparkpipe/spark_glm53_flash_tp16_stage_model.h"
#include "sparkpipe/spark_speculation_tree_frame.h"
#include "glm5_next_stage.cuh"

#define REQUIRE(condition) do { if ( !(condition) ) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #condition); exit(1); } } while ( 0 )
#define CUDA(call) REQUIRE((call) == cudaSuccess)

struct ProbeGeometry
{
	static constexpr uint32_t kTpDegree = SPARK_GLM53_FLASH_TP16_STAGE_TP_DEGREE;
	static constexpr uint32_t kExpertCodec = SPARK_WEIGHT_CODEC_FP8_E4M3;
	static constexpr uint32_t kVerifyRows = SPARK_GLM53_FLASH_TP16_VERIFY_ROWS;
	static const char *Tag(void) { return "glm5_next_verify_probe"; }
	static const char *WeightdModel(void) { return "glm5_next_stage"; }
	static const char *WeightdRevision(void) { return "probe"; }
	static const char *ReportTag(void) { return "GLM5-NEXT-VERIFY-PROBE"; }
};

typedef Glm5NextStage<ProbeGeometry> Stage;

#define PROBE_DEGREE ProbeGeometry::kTpDegree
#define PROBE_ROWS ProbeGeometry::kVerifyRows
#define PROBE_PAGES 48u
#define PROBE_SEQUENCES 2u
#define PROBE_SLOT 1u
#define PROBE_CONTEXT (PROBE_PAGES * G5N_PAGE_SLOTS)
#define PROBE_PREFIX 2304u
#define PROBE_BOUNDARY (G5N_SELECTED - 8u)
#define PROBE_POSITIONS (PROBE_PREFIX + PROBE_ROWS + 32u)
#define PROBE_ALIGN 256u

static_assert(PROBE_PREFIX > G5N_SELECTED && PROBE_POSITIONS <= PROBE_CONTEXT && PROBE_BOUNDARY + 16u > G5N_SELECTED,
	"the long frames start past the index top-k so the selector drops pools, a 16-row frame crosses into that regime, all inside the capacity");

static __host__ __device__ uint64_t Mix(uint64_t z)
{
	z = z * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
	z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
	z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
	return z ^ (z >> 31u);
}

__global__ void ProbeFillBf16(uint16_t *out, uint64_t count, uint64_t seed, float offset, float scale)
{
	for ( uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (uint64_t)gridDim.x * blockDim.x )
	{
		const float u = (float)(Mix(seed * 0x100000001ull + i) & 0xffffffu) / 8388608.0f - 1.0f;
		out[i] = LmFloatToBf16(offset + scale * u);
	}
}

__global__ void ProbeFillF32(float *out, uint64_t count, uint64_t seed, float offset, float scale)
{
	for ( uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (uint64_t)gridDim.x * blockDim.x )
		out[i] = offset + scale * ((float)(Mix(seed * 0x100000001ull + i) & 0xffffffu) / 8388608.0f - 1.0f);
}

__global__ void ProbeFillFp8(uint8_t *out, uint64_t count, uint64_t seed)
{
	for ( uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (uint64_t)gridDim.x * blockDim.x )
	{
		const uint64_t z = Mix(seed * 0x100000001ull + i);
		out[i] = (uint8_t)(((z & 1u) << 7u) | ((3u + (z >> 1u) % 8u) << 3u) | ((z >> 8u) & 7u));
	}
}

template<class Geometry>
__global__ void ProbeDumpKernel(LmKvShardView view, uint32_t slot_bytes, uint8_t *out)
{
	const uint32_t position = blockIdx.x;
	const uint8_t *slot = SparkKvShardOwns(view.shard, position) != 0u ? LmKvShardSlotRequired<Geometry>(view, 0u, position, 0u, LM_KV_ACCESS_READ) : 0;
	for ( uint32_t i = threadIdx.x; i < slot_bytes; i += blockDim.x )
		out[(uint64_t)position * slot_bytes + i] = slot != 0 ? slot[i] : 0u;
}

static void FillBf16(uint16_t *out, uint64_t count, uint64_t seed, float offset, float scale)
{
	ProbeFillBf16<<<1024, 256>>>(out, count, seed, offset, scale);
	CUDA(cudaGetLastError());
}

static void FillF32(float *out, uint64_t count, uint64_t seed, float offset, float scale)
{
	ProbeFillF32<<<1024, 256>>>(out, count, seed, offset, scale);
	CUDA(cudaGetLastError());
}

static void *Device(uint64_t bytes)
{
	void *pointer = 0;
	CUDA(cudaMalloc(&pointer, bytes));
	CUDA(cudaMemset(pointer, 0, bytes));
	return pointer;
}

static uint64_t Align(uint64_t bytes)
{
	return (bytes + PROBE_ALIGN - 1u) / PROBE_ALIGN * PROBE_ALIGN;
}

static std::vector<uint8_t> Host(const void *device, uint64_t bytes)
{
	std::vector<uint8_t> host(bytes);
	CUDA(cudaDeviceSynchronize());
	CUDA(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost));
	return host;
}

struct Probe
{
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
	uint64_t recurrent_bytes;
	uint16_t *inputs;
	uint16_t *drafts;
	uint16_t *variants;
	uint8_t *dump;
};

struct Frame
{
	SparkStageRunnerStep step;
	uint32_t *metadata;
	uint32_t *tree_meta;
	uint16_t *hidden;
};

static SparkStatus ProbeRound(void *context, void *stream, uint32_t operation, uint32_t stage, uint32_t rows, uint32_t row_elements,
	const void *local, void *full)
{
	const uint64_t block = (uint64_t)rows * (row_elements != 0u ? row_elements : G5N_HIDDEN) * sizeof(uint16_t);
	(void)context;
	(void)stage;
	if ( operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16 )
		return local == full || cudaMemcpyAsync(full, local, block, cudaMemcpyDeviceToDevice, (cudaStream_t)stream) == cudaSuccess ?
			SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
	if ( operation != SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER && operation != SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL )
		return SPARK_STATUS_UNSUPPORTED;
	for ( uint32_t source = 0u; source < PROBE_DEGREE; ++source )
		if ( cudaMemcpyAsync((uint8_t *)full + source * block, local, block, cudaMemcpyDeviceToDevice, (cudaStream_t)stream) != cudaSuccess )
			return SPARK_STATUS_IO_ERROR;
	return SPARK_STATUS_OK;
}

static void ProbeFail(void *context, SparkStatus status)
{
	(void)context;
	fprintf(stderr, "FAIL the stage reported a service failure %d\n", (int)status);
	exit(1);
}

static SparkStatus ProbeExperts(void *context, uint32_t layer, const uint32_t *offsets, uint32_t routed_rows, const void **w1, const void **w2)
{
	const Probe *probe = (const Probe *)context;
	(void)layer;
	(void)offsets;
	(void)routed_rows;
	*w1 = probe->expert_w1;
	*w2 = probe->expert_w2;
	return SPARK_STATUS_OK;
}

static void ProbeLayerDone(void *context, uint32_t layer)
{
	(void)context;
	(void)layer;
}

static float LinearScale(const Glm5NextPackShape *shape)
{
	return 1.7f / sqrtf((float)shape->columns);
}

static const void *ProbeTensor(uint32_t kind, Glm5NextPackShape *shape)
{
	static const void *made[GLM5_NEXT_T_COUNT];
	static Glm5NextPackShape shapes[GLM5_NEXT_T_COUNT];
	REQUIRE(Glm5NextPackShapeOf(kind, shape) != 0u);
	if ( shape->split == GLM5_NEXT_SPLIT_ROWS )
		shape->rows /= PROBE_DEGREE;
	if ( shape->split == GLM5_NEXT_SPLIT_COLUMNS )
		shape->columns /= PROBE_DEGREE;
	if ( made[kind] != 0 )
	{
		*shape = shapes[kind];
		return made[kind];
	}
	const uint64_t count = (uint64_t)shape->groups * shape->rows * shape->columns;
	const uint64_t seed = 100u + kind;
	float offset = 0.0f, scale = LinearScale(shape);
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
	void *pointer = Device(Align(count * (shape->payload_type == GLM5_NEXT_PAYLOAD_BF16 ? 2u : 4u)));
	if ( shape->payload_type == GLM5_NEXT_PAYLOAD_BF16 )
		FillBf16((uint16_t *)pointer, count, seed, offset, scale);
	else
		FillF32((float *)pointer, count, seed, offset, scale);
	made[kind] = pointer;
	shapes[kind] = *shape;
	return pointer;
}

static const void *ProbeExpertPlane(uint32_t kind, Glm5NextLayerWeights *w)
{
	Glm5NextPackShape shape;
	REQUIRE(Glm5NextPackShapeOf(kind, &shape) != 0u);
	if ( shape.split == GLM5_NEXT_SPLIT_ROWS )
		shape.rows /= PROBE_DEGREE;
	if ( shape.split == GLM5_NEXT_SPLIT_COLUMNS )
		shape.columns /= PROBE_DEGREE;
	const uint64_t payload = Align((uint64_t)shape.groups * shape.rows * shape.columns);
	const uint64_t scales = (uint64_t)shape.groups * shape.rows * ((shape.columns + LmFp8::kScaleGroup - 1u) / LmFp8::kScaleGroup);
	uint8_t *plane = (uint8_t *)Device(payload + Align(scales * 4u));
	ProbeFillFp8<<<1024, 256>>>(plane, (uint64_t)shape.groups * shape.rows * shape.columns, 300u + kind);
	CUDA(cudaGetLastError());
	FillF32((float *)(plane + payload), scales, 400u + kind, 0.6f / sqrtf((float)shape.columns), 0.3f / sqrtf((float)shape.columns));
	for ( uint32_t layer = 0u; layer < G5N_LAYERS; ++layer )
		if ( kind == GLM5_NEXT_T_EXPERT_UP_GATE )
			w[layer].expert_w1_offset = 0u, w[layer].expert_w1_scale_offset = payload;
		else
			w[layer].expert_w2_offset = 0u, w[layer].expert_w2_scale_offset = payload;
	return plane;
}

static void ProbeOpen(Probe *probe)
{
	Glm5NextModelState *state = (Glm5NextModelState *)calloc(1u, sizeof(Glm5NextModelState));
	Glm5NextPackShape shape;
	SparkStageRunnerKv kv;
	uint32_t *table;
	int device = 0;
	REQUIRE(state != 0);
	memset(&probe->services, 0, sizeof(probe->services));
	probe->services.context = probe;
	probe->services.round = ProbeRound;
	probe->services.fail = ProbeFail;
	probe->services.expert_weights = ProbeExperts;
	probe->services.layer_done = ProbeLayerDone;
	CUDA(cudaStreamCreateWithFlags(&probe->stream, cudaStreamNonBlocking));
	CUDA(cudaGetDevice(&device));
	CUDA(cudaDeviceGetAttribute((int *)&probe->sms, cudaDevAttrMultiProcessorCount, device));
	state->services = &probe->services;
	state->tp_degree = PROBE_DEGREE;
	state->tp_rank = 0u;
	state->max_rows = PROBE_ROWS;
	state->slots = PROBE_SEQUENCES;
	state->context_capacity = PROBE_CONTEXT;
	state->collective_capacity = 2u * PROBE_ROWS + 1u;
	state->kda_state_slot_bytes = (uint64_t)Glm5NextRankHeads(G5N_KDA_HEADS, PROBE_DEGREE) * G5N_KDA_KEY * G5N_KDA_VALUE * sizeof(float);
	state->kda_window_slot_bytes = (uint64_t)Glm5NextRankHeads(G5N_KDA_HEADS, PROBE_DEGREE) * G5N_KDA_KEY * G5N_KDA_CONV * sizeof(uint16_t);
	state->verify_rows = ProbeGeometry::kVerifyRows;
	REQUIRE(SparkGlm5NextKvShardFits(PROBE_DEGREE, PROBE_ROWS) != 0u &&
		SparkGlm5NextIndexCpGatherSequences(PROBE_ROWS, Glm5NextLocalStride(state, PROBE_CONTEXT)) <= state->collective_capacity &&
		SparkGlm5NextKvShardPartialSequences(PROBE_ROWS, PROBE_DEGREE, UINT32_MAX) <= state->collective_capacity &&
		SparkGlm5NextKvShardQuerySequences(PROBE_ROWS, PROBE_DEGREE) <= state->collective_capacity);
	for ( uint32_t layer = 0u; layer < G5N_LAYERS; ++layer )
	{
		state->layers[layer].kda_ordinal = Glm5NextKdaOrdinal(layer);
		state->layers[layer].dsa_ordinal = Glm5NextDsaOrdinal(layer);
		for ( uint32_t kind = GLM5_NEXT_T_ATTN_NORM; kind < GLM5_NEXT_T_MTP_EH_PROJ; ++kind )
			if ( kind != GLM5_NEXT_T_EXPERT_UP_GATE && kind != GLM5_NEXT_T_EXPERT_DOWN && Glm5NextPackKindAllowed(kind, layer) != 0u )
				REQUIRE(Glm5NextPlaceTensor(&state->layers[layer], kind, ProbeTensor(kind, &shape)) == SPARK_STATUS_OK);
	}
	state->final_norm = ProbeTensor(GLM5_NEXT_T_FINAL_NORM, &shape);
	probe->expert_w1 = ProbeExpertPlane(GLM5_NEXT_T_EXPERT_UP_GATE, state->layers);
	probe->expert_w2 = ProbeExpertPlane(GLM5_NEXT_T_EXPERT_DOWN, state->layers);
	REQUIRE(Glm5NextAllocate(state, 0u) == SPARK_STATUS_OK);
	probe->latent_bytes = (uint64_t)PROBE_SEQUENCES * PROBE_PAGES * Glm5NextKv::kPageBytes / PROBE_DEGREE;
	probe->index_bytes = (uint64_t)PROBE_SEQUENCES * PROBE_PAGES * Glm5NextIndexKv::kPageBytes / PROBE_DEGREE;
	probe->latent_pool = (uint8_t *)Device(probe->latent_bytes * G5N_DSA_LAYERS);
	probe->index_pool = (uint8_t *)Device(probe->index_bytes * G5N_DSA_LAYERS);
	std::vector<uint32_t> pages(PROBE_SEQUENCES * PROBE_PAGES);
	for ( uint32_t page = 0u; page < pages.size(); ++page )
		pages[page] = page;
	table = (uint32_t *)Device(pages.size() * sizeof(uint32_t));
	CUDA(cudaMemcpy(table, pages.data(), pages.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
	CUDA(cudaDeviceSynchronize());
	memset(&kv, 0, sizeof(kv));
	kv.pool = probe->latent_pool;
	kv.layer_stride_bytes = probe->latent_bytes;
	kv.layer_page_bytes = Glm5NextKv::kPageBytes;
	kv.layer_count = G5N_DSA_LAYERS;
	kv.page_table = table;
	kv.page_table_stride = PROBE_PAGES;
	kv.pool_page_count = PROBE_SEQUENCES * PROBE_PAGES;
	kv.sequence_count = PROBE_SEQUENCES;
	kv.context_shard = SparkGlm5NextKvShardLatent(0u, PROBE_DEGREE);
	kv.second_pool = probe->index_pool;
	kv.second_layer_stride_bytes = probe->index_bytes;
	kv.second_layer_page_bytes = Glm5NextIndexKv::kPageBytes;
	kv.second_layer_count = G5N_DSA_LAYERS;
	REQUIRE(Stage::AttachKv(state, &kv) == SPARK_STATUS_OK);
	probe->recurrent_bytes = (uint64_t)G5N_KDA_LAYERS * (state->kda_state_slot_bytes + 3u * state->kda_window_slot_bytes);
	probe->inputs = (uint16_t *)Device((uint64_t)PROBE_POSITIONS * G5N_HIDDEN * 2u);
	probe->drafts = (uint16_t *)Device((uint64_t)PROBE_POSITIONS * G5N_HIDDEN * 2u);
	probe->variants = (uint16_t *)Device(32u * (uint64_t)G5N_HIDDEN * 2u);
	FillBf16(probe->inputs, (uint64_t)PROBE_POSITIONS * G5N_HIDDEN, 1u, 0.0f, 1.7f);
	FillBf16(probe->drafts, (uint64_t)PROBE_POSITIONS * G5N_HIDDEN, 2u, 0.0f, 1.7f);
	FillBf16(probe->variants, 32u * (uint64_t)G5N_HIDDEN, 3u, 0.0f, 1.7f);
	probe->dump = (uint8_t *)Device((uint64_t)PROBE_POSITIONS * Glm5NextKv::kSlotBytes);
	probe->state = state;
}

static void FrameOpen(Frame *frame, const Probe *probe)
{
	frame->metadata = (uint32_t *)Device((4u * PROBE_ROWS + 8u) * sizeof(uint32_t));
	frame->tree_meta = (uint32_t *)Device(6u * PROBE_ROWS * sizeof(uint32_t));
	frame->hidden = (uint16_t *)Device((uint64_t)PROBE_ROWS * G5N_HIDDEN * 2u);
	memset(&frame->step, 0, sizeof(frame->step));
	frame->step.hidden_bf16 = frame->hidden;
	frame->step.positions = frame->metadata;
	frame->step.sequence_of_row = frame->metadata + PROBE_ROWS;
	frame->step.sequence_row_indices = frame->metadata + 2u * PROBE_ROWS;
	frame->step.context_length = frame->metadata + 3u * PROBE_ROWS;
	frame->step.sequence_row_begin = frame->metadata + 3u * PROBE_ROWS + 2u;
	frame->step.recurrent_index = frame->metadata + 3u * PROBE_ROWS + 4u;
	frame->step.sequences = 1u;
	frame->step.context = PROBE_CONTEXT;
	frame->step.multiprocessors = probe->sms;
	CUDA(cudaMalloc((void **)&frame->step.route_fault, sizeof(uint32_t)));
	CUDA(cudaMemset(frame->step.route_fault, 0, sizeof(uint32_t)));
	frame->step.every_row_hidden = 1u;
}

static uint32_t Finite(const std::vector<uint8_t> &rows)
{
	for ( uint64_t i = 0u; i + 1u < rows.size(); i += 2u )
		if ( ((((uint32_t)rows[i + 1u] << 8u) | rows[i]) & 0x7f80u) == 0x7f80u )
			return 0u;
	return 1u;
}

static void Run(Probe *probe, Frame *frame, uint32_t first, uint32_t rows, uint32_t commit, uint32_t gather_context, uint32_t draft_from)
{
	std::vector<uint32_t> metadata(4u * PROBE_ROWS + 8u, 0u);
	for ( uint32_t row = 0u; row < rows; ++row )
	{
		metadata[row] = first + row;
		metadata[2u * PROBE_ROWS + row] = row;
	}
	metadata[3u * PROBE_ROWS] = first + rows;
	metadata[3u * PROBE_ROWS + 3u] = rows;
	metadata[3u * PROBE_ROWS + 4u] = PROBE_SLOT;
	CUDA(cudaMemcpyAsync(frame->metadata, metadata.data(), metadata.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, probe->stream));
	for ( uint32_t row = 0u; row < rows; ++row )
		CUDA(cudaMemcpyAsync(frame->hidden + (uint64_t)row * G5N_HIDDEN, (row < draft_from ? probe->inputs : probe->drafts) +
			(uint64_t)(first + row) * G5N_HIDDEN, G5N_HIDDEN * 2u, cudaMemcpyDeviceToDevice, probe->stream));
	frame->step.rows = rows;
	frame->step.commit = commit;
	frame->step.gather_context = gather_context;
	REQUIRE(Stage::Step(probe->state, &frame->step, probe->stream) == SPARK_STATUS_OK);
	CUDA(cudaStreamSynchronize(probe->stream));
	CUDA(cudaGetLastError());
	REQUIRE(Finite(Host(frame->hidden, (uint64_t)rows * G5N_HIDDEN * 2u)) != 0u);
}

static void Keep(Probe *probe, Frame *frame, uint32_t kept)
{
	uint32_t *device = frame->metadata + 3u * PROBE_ROWS + 6u;
	CUDA(cudaMemcpyAsync(device, &kept, sizeof(kept), cudaMemcpyHostToDevice, probe->stream));
	REQUIRE(Stage::KeepRows(probe->state, &frame->step, device, probe->stream) == SPARK_STATUS_OK);
	CUDA(cudaStreamSynchronize(probe->stream));
	CUDA(cudaGetLastError());
}

static void Recurrent(Probe *probe, uint32_t to_buffer, void *buffer)
{
	REQUIRE(Stage::RecurrentCopy(probe->state, to_buffer, PROBE_SLOT, buffer, probe->recurrent_bytes, probe->stream) == SPARK_STATUS_OK);
	CUDA(cudaStreamSynchronize(probe->stream));
}

static std::vector<uint8_t> RecurrentHost(Probe *probe, void *scratch)
{
	Recurrent(probe, 1u, scratch);
	return Host(scratch, probe->recurrent_bytes);
}

static std::vector<uint8_t> KvPrefix(Probe *probe, uint32_t positions)
{
	std::vector<uint8_t> all;
	for ( uint32_t layer = 0u; layer < G5N_DSA_LAYERS; ++layer )
	{
		ProbeDumpKernel<Glm5NextKv><<<positions, 128, 0, probe->stream>>>(probe->state->latent_views[layer], Glm5NextKv::kSlotBytes, probe->dump);
		CUDA(cudaGetLastError());
		std::vector<uint8_t> latent = Host(probe->dump, (uint64_t)positions * Glm5NextKv::kSlotBytes);
		ProbeDumpKernel<Glm5NextIndexKv><<<positions, 128, 0, probe->stream>>>(probe->state->index_views[layer], Glm5NextIndexKv::kSlotBytes, probe->dump);
		CUDA(cudaGetLastError());
		std::vector<uint8_t> index = Host(probe->dump, (uint64_t)positions * Glm5NextIndexKv::kSlotBytes);
		all.insert(all.end(), latent.begin(), latent.end());
		all.insert(all.end(), index.begin(), index.end());
	}
	return all;
}

static uint32_t Differ(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b)
{
	return a.size() != b.size() || memcmp(a.data(), b.data(), a.size()) != 0 ? 1u : 0u;
}

static void Mismatch(uint32_t differ, const char *what, uint32_t rows, uint32_t first, uint32_t kept)
{
	if ( differ == 0u )
		return;
	printf("FAIL glm5_next verify: %u-row frame at position %u, %u kept rows: %s\n", rows, first, kept, what);
	exit(1);
}

static void Sweep(Probe *probe, Frame *verify, Frame *decode, void *start, void *scratch, void *current, uint32_t first, uint32_t rows)
{
	const uint64_t row_bytes = (uint64_t)G5N_HIDDEN * 2u;
	Recurrent(probe, 0u, start);
	Run(probe, verify, first, rows, 0u, first + rows, rows);
	const std::vector<uint8_t> frame = Host(verify->hidden, rows * row_bytes);
	const std::vector<uint8_t> kv = KvPrefix(probe, first + rows);
	Mismatch(Differ(RecurrentHost(probe, scratch), Host(start, probe->recurrent_bytes)), "the verify step itself moved the recurrent state", rows, first, 0u);
	for ( uint32_t kept = 0u; kept <= rows; ++kept )
	{
		Recurrent(probe, 1u, current);
		Recurrent(probe, 0u, start);
		Keep(probe, verify, kept);
		Mismatch(Differ(RecurrentHost(probe, scratch), Host(current, probe->recurrent_bytes)),
			"keep_rows leaves recurrent state that the same number of one-row decode steps did not", rows, first, kept);
		Recurrent(probe, 0u, current);
		if ( kept == rows )
			break;
		Run(probe, decode, first + kept, 1u, 1u, 0u, 1u);
		Mismatch(memcmp(Host(decode->hidden, row_bytes).data(), frame.data() + kept * row_bytes, row_bytes) != 0 ? 1u : 0u,
			"the frame row after the kept rows differs from its one-row decode step", rows, first, kept);
	}
	Mismatch(Differ(KvPrefix(probe, first + rows), kv), "the latent or index cache rows the frame wrote differ from those of one-row steps", rows, first,
		rows);
	printf("%3u-row verify frame at position %u: every row and every keep of 0..%u rows equal the one-row decode steps\n", rows, first, rows);
	Recurrent(probe, 0u, start);
}

static void Rejection(Probe *probe, Frame *verify, Frame *decode, void *start, void *scratch, uint32_t rows, uint32_t kept)
{
	const uint64_t row_bytes = (uint64_t)G5N_HIDDEN * 2u;
	const uint32_t after = 3u;
	std::vector<uint8_t> drafted, plain;
	Recurrent(probe, 0u, start);
	Run(probe, verify, PROBE_PREFIX, rows, 0u, PROBE_PREFIX + rows, kept);
	Keep(probe, verify, kept);
	for ( uint32_t step = 0u; step < after; ++step )
	{
		Run(probe, decode, PROBE_PREFIX + kept + step, 1u, 1u, 0u, 1u);
		std::vector<uint8_t> row = Host(decode->hidden, row_bytes);
		drafted.insert(drafted.end(), row.begin(), row.end());
	}
	const std::vector<uint8_t> drafted_state = RecurrentHost(probe, scratch);
	const std::vector<uint8_t> drafted_kv = KvPrefix(probe, PROBE_PREFIX + kept + after);
	Recurrent(probe, 0u, start);
	for ( uint32_t step = 0u; step < kept + after; ++step )
	{
		Run(probe, decode, PROBE_PREFIX + step, 1u, 1u, 0u, 1u);
		if ( step < kept )
			continue;
		std::vector<uint8_t> row = Host(decode->hidden, row_bytes);
		plain.insert(plain.end(), row.begin(), row.end());
	}
	Mismatch(Differ(drafted, plain), "the decode steps after the rejected tail differ from a run that never saw the drafts", rows, PROBE_PREFIX, kept);
	Mismatch(Differ(RecurrentHost(probe, scratch), drafted_state), "the recurrent state after the rejected tail differs from a run that never saw the drafts",
		rows, PROBE_PREFIX, kept);
	Mismatch(Differ(KvPrefix(probe, PROBE_PREFIX + kept + after), drafted_kv),
		"the cache rows up to the last decoded position differ from a run that never saw the drafts", rows, PROBE_PREFIX, kept);
	printf("rejection: a %u-row frame keeping %u rows, then %u decode steps, equal a run that never saw the drafts\n", rows, kept, after);
	Recurrent(probe, 0u, start);
}

static void RunVariant(Probe *probe, Frame *frame, uint32_t position, uint32_t variant)
{
	std::vector<uint32_t> metadata(4u * PROBE_ROWS + 8u, 0u);
	metadata[0] = position;
	metadata[2u * PROBE_ROWS] = 0u;
	metadata[3u * PROBE_ROWS] = position + 1u;
	metadata[3u * PROBE_ROWS + 3u] = 1u;
	metadata[3u * PROBE_ROWS + 4u] = PROBE_SLOT;
	CUDA(cudaMemcpyAsync(frame->metadata, metadata.data(), metadata.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, probe->stream));
	CUDA(cudaMemcpyAsync(frame->hidden, probe->variants + (uint64_t)variant * G5N_HIDDEN, G5N_HIDDEN * 2u, cudaMemcpyDeviceToDevice, probe->stream));
	frame->step.rows = 1u;
	frame->step.commit = 1u;
	frame->step.gather_context = 0u;
	frame->step.tree = 0u;
	frame->step.slots = 0;
	frame->step.fork_positions = 0;
	frame->step.slot_offsets = 0;
	frame->step.parent_rows = 0;
	frame->step.state_loads = 0;
	frame->step.state_saves = 0;
	REQUIRE(Stage::Step(probe->state, &frame->step, probe->stream) == SPARK_STATUS_OK);
	CUDA(cudaStreamSynchronize(probe->stream));
	CUDA(cudaGetLastError());
	REQUIRE(Finite(Host(frame->hidden, (uint64_t)G5N_HIDDEN * 2u)) != 0u);
}

static void UploadTree(Probe *probe, Frame *frame, uint32_t first, const uint32_t *parents, const uint32_t *variants, uint32_t rows)
{
	std::vector<uint32_t> metadata(4u * PROBE_ROWS + 8u, 0u);
	std::vector<uint32_t> tree_meta(6u * PROBE_ROWS, 0u);
	SparkSpeculationTreeLayout layout;
	const uint32_t slot_grain = G5N_KPOOL * PROBE_DEGREE;
	REQUIRE(SparkSpeculationTreeLayoutBuild(parents, variants, rows, first, slot_grain, &layout) == SPARK_STATUS_OK);
	for ( uint32_t row = 0u; row < rows; ++row )
	{
		metadata[row] = layout.anchor_position + layout.row[row].depth;
		metadata[2u * PROBE_ROWS + row] = row;
		tree_meta[row] = layout.row[row].slot;
		tree_meta[PROBE_ROWS + row] = layout.row[row].fork_position;
		tree_meta[2u * PROBE_ROWS + row] = layout.row[row].slot_offset;
		tree_meta[3u * PROBE_ROWS + row] = parents[row];
		tree_meta[4u * PROBE_ROWS + row] = row != 0u && layout.row[row].branch != layout.row[parents[row]].branch ? layout.row[row].branch : 0u;
		CUDA(cudaMemcpyAsync(frame->hidden + (uint64_t)row * G5N_HIDDEN, probe->variants + (uint64_t)variants[row] * G5N_HIDDEN,
			G5N_HIDDEN * 2u, cudaMemcpyDeviceToDevice, probe->stream));
	}
	for ( uint32_t row = 1u; row < rows; ++row )
		if ( tree_meta[4u * PROBE_ROWS + row] != 0u )
			tree_meta[5u * PROBE_ROWS + parents[row]] |= 1u << (tree_meta[4u * PROBE_ROWS + row] - 1u);
	metadata[3u * PROBE_ROWS] = layout.context;
	metadata[3u * PROBE_ROWS + 3u] = rows;
	metadata[3u * PROBE_ROWS + 4u] = PROBE_SLOT;
	CUDA(cudaMemcpyAsync(frame->metadata, metadata.data(), metadata.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, probe->stream));
	CUDA(cudaMemcpyAsync(frame->tree_meta, tree_meta.data(), tree_meta.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, probe->stream));
	frame->step.rows = rows;
	frame->step.commit = 0u;
	frame->step.gather_context = layout.context;
	frame->step.tree = 1u;
	frame->step.slots = frame->tree_meta;
	frame->step.fork_positions = frame->tree_meta + PROBE_ROWS;
	frame->step.slot_offsets = frame->tree_meta + 2u * PROBE_ROWS;
	frame->step.parent_rows = frame->tree_meta + 3u * PROBE_ROWS;
	frame->step.state_loads = frame->tree_meta + 4u * PROBE_ROWS;
	frame->step.state_saves = frame->tree_meta + 5u * PROBE_ROWS;
}

static void KeepPath(Probe *probe, Frame *frame, const uint32_t *path, uint32_t length)
{
	uint32_t begin[2] = { 0u, length };
	uint32_t *kept_device = frame->metadata + 3u * PROBE_ROWS + 6u;
	CUDA(cudaMemcpyAsync(frame->metadata + 2u * PROBE_ROWS, path, length * sizeof(uint32_t), cudaMemcpyHostToDevice, probe->stream));
	CUDA(cudaMemcpyAsync(frame->metadata + 3u * PROBE_ROWS + 2u, begin, sizeof(begin), cudaMemcpyHostToDevice, probe->stream));
	CUDA(cudaMemcpyAsync(kept_device, &length, sizeof(length), cudaMemcpyHostToDevice, probe->stream));
	REQUIRE(Stage::KeepRows(probe->state, &frame->step, kept_device, probe->stream) == SPARK_STATUS_OK);
	CUDA(cudaStreamSynchronize(probe->stream));
	CUDA(cudaGetLastError());
}

static void TreeSweep(Probe *probe, Frame *verify, Frame *decode, void *start, void *scratch, uint32_t first, uint32_t test_graph)
{
	static const uint32_t kTreeRows = 8u;
	static const uint32_t parents_a[kTreeRows] = { SPARK_SPECULATION_TREE_FRAME_NO_ROW, 0u, 1u, 2u, 3u, 1u, 5u, 0u };
	static const uint32_t variants_a[kTreeRows] = { 0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u };
	static const uint32_t parents_b[kTreeRows] = { SPARK_SPECULATION_TREE_FRAME_NO_ROW, 0u, 1u, 2u, 2u, 4u, 0u, 6u };
	static const uint32_t variants_b[kTreeRows] = { 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u };
	static const uint32_t paths[3][5] = { { 0u, 1u, 2u, 3u, 4u }, { 0u, 1u, 5u, 6u, 0u }, { 0u, 7u, 0u, 0u, 0u } };
	static const uint32_t lengths[3] = { 5u, 4u, 2u };
	const uint64_t row_bytes = (uint64_t)G5N_HIDDEN * 2u;
	const std::vector<uint8_t> prefix_latent = Host(probe->latent_pool, probe->latent_bytes * G5N_DSA_LAYERS);
	const std::vector<uint8_t> prefix_index = Host(probe->index_pool, probe->index_bytes * G5N_DSA_LAYERS);
	std::vector<uint8_t> seq_out[3];
	for ( uint32_t path = 0u; path < 3u; ++path )
	{
		CUDA(cudaMemcpy(probe->latent_pool, prefix_latent.data(), prefix_latent.size(), cudaMemcpyHostToDevice));
		CUDA(cudaMemcpy(probe->index_pool, prefix_index.data(), prefix_index.size(), cudaMemcpyHostToDevice));
		Recurrent(probe, 0u, start);
		for ( uint32_t k = 0u; k < lengths[path]; ++k )
		{
			RunVariant(probe, decode, first + k, variants_a[paths[path][k]]);
			std::vector<uint8_t> row = Host(decode->hidden, row_bytes);
			seq_out[path].insert(seq_out[path].end(), row.begin(), row.end());
		}
	}
	CUDA(cudaMemcpy(probe->latent_pool, prefix_latent.data(), prefix_latent.size(), cudaMemcpyHostToDevice));
	CUDA(cudaMemcpy(probe->index_pool, prefix_index.data(), prefix_index.size(), cudaMemcpyHostToDevice));
	Recurrent(probe, 0u, start);
	UploadTree(probe, verify, first, parents_a, variants_a, kTreeRows);
	REQUIRE(Stage::Step(probe->state, &verify->step, probe->stream) == SPARK_STATUS_OK);
	CUDA(cudaStreamSynchronize(probe->stream));
	CUDA(cudaGetLastError());
	const std::vector<uint8_t> tree_out = Host(verify->hidden, kTreeRows * row_bytes);
	REQUIRE(Finite(tree_out) != 0u);
	Mismatch(Differ(RecurrentHost(probe, scratch), Host(start, probe->recurrent_bytes)), "tree verify moved recurrent state", kTreeRows, first, 0u);
	const std::vector<uint8_t> after_tree_latent = Host(probe->latent_pool, probe->latent_bytes * G5N_DSA_LAYERS);
	const std::vector<uint8_t> after_tree_index = Host(probe->index_pool, probe->index_bytes * G5N_DSA_LAYERS);
	for ( uint32_t path = 0u; path < 3u; ++path )
		for ( uint32_t k = 0u; k < lengths[path]; ++k )
			Mismatch(memcmp(tree_out.data() + (uint64_t)paths[path][k] * row_bytes, seq_out[path].data() + (uint64_t)k * row_bytes, row_bytes) != 0 ? 1u : 0u,
				"tree node differs from its sequential one-row decode", kTreeRows, first, paths[path][k]);
	REQUIRE(memcmp(tree_out.data() + 5u * row_bytes, tree_out.data() + 2u * row_bytes, row_bytes) != 0);
	for ( uint32_t path = 0u; path < 3u; ++path )
	{
		const uint32_t length = lengths[path];
		std::vector<uint8_t> plain, speculated;
		CUDA(cudaMemcpy(probe->latent_pool, prefix_latent.data(), prefix_latent.size(), cudaMemcpyHostToDevice));
		CUDA(cudaMemcpy(probe->index_pool, prefix_index.data(), prefix_index.size(), cudaMemcpyHostToDevice));
		Recurrent(probe, 0u, start);
		for ( uint32_t k = 0u; k < length; ++k )
			RunVariant(probe, decode, first + k, variants_a[paths[path][k]]);
		const std::vector<uint8_t> ref_rec = RecurrentHost(probe, scratch);
		const std::vector<uint8_t> ref_kv = KvPrefix(probe, first + length);
		for ( uint32_t step = 0u; step < 2u; ++step )
		{
			RunVariant(probe, decode, first + length + step, 16u + step);
			std::vector<uint8_t> row = Host(decode->hidden, row_bytes);
			plain.insert(plain.end(), row.begin(), row.end());
		}
		const std::vector<uint8_t> plain_rec = RecurrentHost(probe, scratch);
		const std::vector<uint8_t> plain_kv = KvPrefix(probe, first + length + 2u);
		CUDA(cudaMemcpy(probe->latent_pool, after_tree_latent.data(), after_tree_latent.size(), cudaMemcpyHostToDevice));
		CUDA(cudaMemcpy(probe->index_pool, after_tree_index.data(), after_tree_index.size(), cudaMemcpyHostToDevice));
		Recurrent(probe, 0u, start);
		KeepPath(probe, verify, paths[path], length);
		Mismatch(Differ(RecurrentHost(probe, scratch), ref_rec), "KeepRows tree path recurrent state differs", kTreeRows, first, path);
		Mismatch(Differ(KvPrefix(probe, first + length), ref_kv), "KeepRows tree path relocated KV differs", kTreeRows, first, path);
		for ( uint32_t step = 0u; step < 2u; ++step )
		{
			RunVariant(probe, decode, first + length + step, 16u + step);
			std::vector<uint8_t> row = Host(decode->hidden, row_bytes);
			speculated.insert(speculated.end(), row.begin(), row.end());
		}
		Mismatch(Differ(speculated, plain), "follow-up decodes after tree KeepRows differ", kTreeRows, first, path);
		Mismatch(Differ(RecurrentHost(probe, scratch), plain_rec), "follow-up recurrent state after tree KeepRows differs", kTreeRows, first, path);
		Mismatch(Differ(KvPrefix(probe, first + length + 2u), plain_kv), "follow-up KV after tree KeepRows differs", kTreeRows, first, path);
	}
	if ( test_graph != 0u )
	{
		cudaGraph_t graph = 0;
		cudaGraphExec_t exec = 0;
		CUDA(cudaMemcpy(probe->latent_pool, prefix_latent.data(), prefix_latent.size(), cudaMemcpyHostToDevice));
		CUDA(cudaMemcpy(probe->index_pool, prefix_index.data(), prefix_index.size(), cudaMemcpyHostToDevice));
		Recurrent(probe, 0u, start);
		UploadTree(probe, verify, first, parents_b, variants_b, kTreeRows);
		REQUIRE(Stage::Step(probe->state, &verify->step, probe->stream) == SPARK_STATUS_OK);
		CUDA(cudaStreamSynchronize(probe->stream));
		const std::vector<uint8_t> eager_b = Host(verify->hidden, kTreeRows * row_bytes);
		CUDA(cudaMemcpy(probe->latent_pool, prefix_latent.data(), prefix_latent.size(), cudaMemcpyHostToDevice));
		CUDA(cudaMemcpy(probe->index_pool, prefix_index.data(), prefix_index.size(), cudaMemcpyHostToDevice));
		Recurrent(probe, 0u, start);
		UploadTree(probe, verify, first, parents_a, variants_a, kTreeRows);
		CUDA(cudaStreamSynchronize(probe->stream));
		CUDA(cudaStreamBeginCapture(probe->stream, cudaStreamCaptureModeThreadLocal));
		REQUIRE(Stage::Step(probe->state, &verify->step, probe->stream) == SPARK_STATUS_OK);
		CUDA(cudaStreamEndCapture(probe->stream, &graph));
		CUDA(cudaGraphInstantiate(&exec, graph, 0, 0, 0));
		CUDA(cudaGraphLaunch(exec, probe->stream));
		CUDA(cudaStreamSynchronize(probe->stream));
		Mismatch(Differ(Host(verify->hidden, kTreeRows * row_bytes), tree_out), "tree verify CUDA graph A differs from eager", kTreeRows, first, 0u);
		CUDA(cudaMemcpy(probe->latent_pool, prefix_latent.data(), prefix_latent.size(), cudaMemcpyHostToDevice));
		CUDA(cudaMemcpy(probe->index_pool, prefix_index.data(), prefix_index.size(), cudaMemcpyHostToDevice));
		Recurrent(probe, 0u, start);
		UploadTree(probe, verify, first, parents_b, variants_b, kTreeRows);
		CUDA(cudaGraphLaunch(exec, probe->stream));
		CUDA(cudaStreamSynchronize(probe->stream));
		Mismatch(Differ(Host(verify->hidden, kTreeRows * row_bytes), eager_b), "tree verify CUDA graph B replay differs from eager", kTreeRows, first, 1u);
		CUDA(cudaGraphExecDestroy(exec));
		CUDA(cudaGraphDestroy(graph));
	}
	CUDA(cudaMemcpy(probe->latent_pool, prefix_latent.data(), prefix_latent.size(), cudaMemcpyHostToDevice));
	CUDA(cudaMemcpy(probe->index_pool, prefix_index.data(), prefix_index.size(), cudaMemcpyHostToDevice));
	Recurrent(probe, 0u, start);
	verify->step.tree = 0u;
	verify->step.slots = 0;
	verify->step.fork_positions = 0;
	verify->step.slot_offsets = 0;
	verify->step.parent_rows = 0;
	verify->step.state_loads = 0;
	verify->step.state_saves = 0;
	printf("tree verify at position %u: 8-node multi-branch tree, all 3 KeepRows paths + 2 post-decodes%s bit-exact\n", first,
		test_graph != 0u ? ", and 2-topology CUDA graph replay" : "");
}

static void LargeTreeCheck(Probe *probe, Frame *verify, Frame *decode, void *start, uint32_t first)
{
	static const uint32_t kRows = 128u, kTrunk = 65u;
	const uint64_t row_bytes = (uint64_t)G5N_HIDDEN * 2u;
	const std::vector<uint8_t> prefix_latent = Host(probe->latent_pool, probe->latent_bytes * G5N_DSA_LAYERS);
	const std::vector<uint8_t> prefix_index = Host(probe->index_pool, probe->index_bytes * G5N_DSA_LAYERS);
	std::vector<uint32_t> parents(kRows), variants(kRows);
	parents[0] = SPARK_SPECULATION_TREE_FRAME_NO_ROW;
	variants[0] = 0u;
	for ( uint32_t row = 1u; row < kTrunk; ++row )
	{
		parents[row] = row - 1u;
		variants[row] = row % 16u;
	}
	parents[kTrunk] = 0u;
	variants[kTrunk] = 17u;
	for ( uint32_t row = kTrunk + 1u; row < kRows; ++row )
	{
		parents[row] = row - 1u;
		variants[row] = 16u + (row % 16u);
	}
	CUDA(cudaMemcpy(probe->latent_pool, prefix_latent.data(), prefix_latent.size(), cudaMemcpyHostToDevice));
	CUDA(cudaMemcpy(probe->index_pool, prefix_index.data(), prefix_index.size(), cudaMemcpyHostToDevice));
	Recurrent(probe, 0u, start);
	UploadTree(probe, verify, first, parents.data(), variants.data(), kRows);
	REQUIRE(Stage::Step(probe->state, &verify->step, probe->stream) == SPARK_STATUS_OK);
	CUDA(cudaStreamSynchronize(probe->stream));
	CUDA(cudaGetLastError());
	const std::vector<uint8_t> tree_out = Host(verify->hidden, kRows * row_bytes);
	REQUIRE(Finite(tree_out) != 0u);
	CUDA(cudaMemcpy(probe->latent_pool, prefix_latent.data(), prefix_latent.size(), cudaMemcpyHostToDevice));
	CUDA(cudaMemcpy(probe->index_pool, prefix_index.data(), prefix_index.size(), cudaMemcpyHostToDevice));
	Recurrent(probe, 0u, start);
	RunVariant(probe, decode, first, variants[0]);
	Mismatch(memcmp(Host(decode->hidden, row_bytes).data(), tree_out.data(), row_bytes) != 0 ? 1u : 0u,
		"128-row tree root differs from sequential decode", kRows, first, 0u);
	for ( uint32_t step = 0u; step < 3u; ++step )
	{
		const uint32_t row = kTrunk + step;
		RunVariant(probe, decode, first + 1u + step, variants[row]);
		Mismatch(memcmp(Host(decode->hidden, row_bytes).data(), tree_out.data() + (uint64_t)row * row_bytes, row_bytes) != 0 ? 1u : 0u,
			"128-row tree branch row differs from sequential decode", kRows, first, row);
	}
	CUDA(cudaMemcpy(probe->latent_pool, prefix_latent.data(), prefix_latent.size(), cudaMemcpyHostToDevice));
	CUDA(cudaMemcpy(probe->index_pool, prefix_index.data(), prefix_index.size(), cudaMemcpyHostToDevice));
	Recurrent(probe, 0u, start);
	verify->step.tree = 0u;
	verify->step.slots = 0;
	verify->step.fork_positions = 0;
	verify->step.slot_offsets = 0;
	verify->step.parent_rows = 0;
	verify->step.state_loads = 0;
	verify->step.state_saves = 0;
	printf("large tree verify at position %u: 128-row multi-branch tree bit-exact\n", first);
}

static void Prefill(Probe *probe, Frame *frame, uint32_t first, uint32_t end)
{
	for ( ; first < end; first += PROBE_ROWS )
	{
		const uint32_t rows = end - first < PROBE_ROWS ? end - first : PROBE_ROWS;
		Run(probe, frame, first, rows, 1u, first + rows, rows);
	}
}

struct Tp16Barrier
{
	pthread_mutex_t lock;
	pthread_cond_t cond;
	uint32_t parties, waiting, phase, failed;
};

static uint32_t Tp16Wait(Tp16Barrier *b)
{
	pthread_mutex_lock(&b->lock);
	const uint32_t phase = b->phase;
	if ( ++b->waiting == b->parties )
	{
		b->waiting = 0u;
		b->phase++;
		pthread_cond_broadcast(&b->cond);
	}
	else
		while ( b->phase == phase && b->failed == 0u )
			pthread_cond_wait(&b->cond, &b->lock);
	const uint32_t ok = b->failed == 0u ? 1u : 0u;
	pthread_mutex_unlock(&b->lock);
	return ok;
}

__global__ void Tp16SumBf16Kernel(const uint8_t *staging, uint64_t stride, uint32_t ranks, uint16_t *out, uint64_t count)
{
	for ( uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (uint64_t)gridDim.x * blockDim.x )
	{
		float total = 0.0f;
		for ( uint32_t r = 0u; r < ranks; ++r )
			total += LmBf16ToFloat(((const uint16_t *)(staging + (uint64_t)r * stride))[i]);
		out[i] = LmFloatToBf16(total);
	}
}

struct Tp16Rig
{
	static constexpr uint32_t kRanks = PROBE_DEGREE;
	struct Rank
	{
		Tp16Rig *rig;
		uint32_t index;
		Glm5NextModelState *state;
		SparkStageRunnerServices services;
		cudaStream_t stream;
		uint8_t *latent_pool;
		uint8_t *index_pool;
		uint16_t *hidden;
		uint32_t *metadata;
		uint32_t *route_fault;
		SparkStageRunnerStep step;
		uint32_t single_dsa;
		SparkStatus status;
	};
	Rank ranks[kRanks];
	Tp16Barrier barrier;
	uint8_t *staging;
	uint64_t staging_bytes;
	uint64_t latent_bytes;
	uint64_t index_bytes;
	uint32_t *page_table;
	const void *expert_w1;
	const void *expert_w2;
	uint32_t sms;

	static SparkStatus Round(void *context, void *stream_ptr, uint32_t op, uint32_t stage, uint32_t rows, uint32_t row_elements,
		const void *local, void *full)
	{
		Rank *rank = (Rank *)context;
		Tp16Rig *rig = rank->rig;
		cudaStream_t stream = (cudaStream_t)stream_ptr;
		const uint64_t block = (uint64_t)rows * (row_elements != 0u ? row_elements : G5N_HIDDEN) * sizeof(uint16_t);
		uint8_t *slot = rig->staging + (uint64_t)rank->index * rig->staging_bytes;
		const uint64_t copy = op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL ? (uint64_t)kRanks * block : block;
		if ( stage != SPARK_STAGE_RUNNER_ROUND_ALL )
			return SPARK_STATUS_UNSUPPORTED;
		if ( copy > rig->staging_bytes || cudaMemcpyAsync(slot, local, copy, cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
			cudaStreamSynchronize(stream) != cudaSuccess )
			return SPARK_STATUS_IO_ERROR;
		if ( Tp16Wait(&rig->barrier) == 0u )
			return SPARK_STATUS_IO_ERROR;
		if ( op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER )
		{
			for ( uint32_t s = 0u; s < kRanks; ++s )
				if ( cudaMemcpyAsync((uint8_t *)full + (uint64_t)s * block, rig->staging + (uint64_t)s * rig->staging_bytes, block,
						cudaMemcpyDeviceToDevice, stream) != cudaSuccess )
					return SPARK_STATUS_IO_ERROR;
		}
		else if ( op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_TO_ALL )
		{
			for ( uint32_t s = 0u; s < kRanks; ++s )
				if ( cudaMemcpyAsync((uint8_t *)full + (uint64_t)s * block,
						rig->staging + (uint64_t)s * rig->staging_bytes + (uint64_t)rank->index * block, block,
						cudaMemcpyDeviceToDevice, stream) != cudaSuccess )
					return SPARK_STATUS_IO_ERROR;
		}
		else if ( op == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16 )
		{
			const uint64_t elems = block / sizeof(uint16_t);
			Tp16SumBf16Kernel<<<(uint32_t)((elems + 255u) / 256u), 256u, 0, stream>>>(rig->staging, rig->staging_bytes, kRanks, (uint16_t *)full, elems);
		}
		else
			return SPARK_STATUS_UNSUPPORTED;
		if ( cudaStreamSynchronize(stream) != cudaSuccess || Tp16Wait(&rig->barrier) == 0u )
			return SPARK_STATUS_IO_ERROR;
		return SPARK_STATUS_OK;
	}

	static SparkStatus Experts(void *context, uint32_t layer, const uint32_t *offsets, uint32_t routed_rows, const void **w1, const void **w2)
	{
		const Rank *rank = (const Rank *)context;
		(void)layer;
		(void)offsets;
		(void)routed_rows;
		*w1 = rank->rig->expert_w1;
		*w2 = rank->rig->expert_w2;
		return SPARK_STATUS_OK;
	}

	static void *Thread(void *arg)
	{
		Rank *rank = (Rank *)arg;
		SparkStatus st;
		if ( rank->single_dsa != 0u )
		{
			Stage::Frame frame;
			st = Stage::Begin(rank->state, &rank->step, rank->stream, &frame);
			if ( st == SPARK_STATUS_OK )
			{
				Stage::BindLayer(rank->state, frame, 3u);
				frame.b.prenormed = 1u;
				if ( cudaMemcpyAsync(frame.b.normed_bf16, rank->step.hidden_bf16, (uint64_t)rank->step.rows * G5N_HIDDEN * 2u,
						cudaMemcpyDeviceToDevice, rank->stream) != cudaSuccess )
					st = SPARK_STATUS_IO_ERROR;
			}
			if ( st == SPARK_STATUS_OK )
				st = Stage::Attention(rank->state, frame, 3u);
			if ( st == SPARK_STATUS_OK )
				st = Stage::AllReduce(rank->state, frame);
			if ( st == SPARK_STATUS_OK && cudaMemcpyAsync(rank->step.hidden_bf16, frame.b.attention_out_bf16,
					(uint64_t)rank->step.rows * G5N_HIDDEN * 2u, cudaMemcpyDeviceToDevice, rank->stream) != cudaSuccess )
				st = SPARK_STATUS_IO_ERROR;
		}
		else
			st = Stage::Step(rank->state, &rank->step, rank->stream);
		if ( st == SPARK_STATUS_OK && cudaStreamSynchronize(rank->stream) != cudaSuccess )
			st = SPARK_STATUS_IO_ERROR;
		if ( st != SPARK_STATUS_OK )
		{
			pthread_mutex_lock(&rank->rig->barrier.lock);
			rank->rig->barrier.failed = 1u;
			pthread_cond_broadcast(&rank->rig->barrier.cond);
			pthread_mutex_unlock(&rank->rig->barrier.lock);
		}
		rank->status = st;
		return 0;
	}

	void Open(const Probe *probe)
	{
		memset(this, 0, sizeof(*this));
		barrier.parties = kRanks;
		pthread_mutex_init(&barrier.lock, 0);
		pthread_cond_init(&barrier.cond, 0);
		sms = probe->sms;
		expert_w1 = probe->expert_w1;
		expert_w2 = probe->expert_w2;
		latent_bytes = probe->latent_bytes;
		index_bytes = probe->index_bytes;
		std::vector<uint32_t> pages(PROBE_SEQUENCES * PROBE_PAGES);
		for ( uint32_t page = 0u; page < pages.size(); ++page )
			pages[page] = page;
		page_table = (uint32_t *)Device(pages.size() * sizeof(uint32_t));
		CUDA(cudaMemcpy(page_table, pages.data(), pages.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
		const uint64_t partial_bytes = (uint64_t)kRanks * SparkGlm5NextKvShardPartialStrideCapacity(kRanks, PROBE_ROWS) * sizeof(float);
		const uint64_t query_bytes = (uint64_t)kRanks * (SparkGlm5NextKvShardQueryStride(PROBE_ROWS, kRanks) +
			(uint64_t)SparkGlm5NextIndexCpGatherSequences(PROBE_ROWS, Glm5NextLocalStride(probe->state, PROBE_CONTEXT)) *
			SPARK_GLM5_NEXT_INDEX_CP_SEQUENCE_FLOATS * 2u) * sizeof(uint16_t);
		staging_bytes = partial_bytes > query_bytes ? partial_bytes : query_bytes;
		staging = (uint8_t *)Device((uint64_t)kRanks * staging_bytes);
		for ( uint32_t r = 0u; r < kRanks; ++r )
		{
			Rank *rank = &ranks[r];
			Glm5NextModelState *st = (Glm5NextModelState *)calloc(1u, sizeof(Glm5NextModelState));
			SparkStageRunnerKv kv;
			REQUIRE(st != 0);
			rank->rig = this;
			rank->index = r;
			rank->state = st;
			rank->services.context = rank;
			rank->services.round = Round;
			rank->services.fail = ProbeFail;
			rank->services.expert_weights = Experts;
			rank->services.layer_done = ProbeLayerDone;
			CUDA(cudaStreamCreateWithFlags(&rank->stream, cudaStreamNonBlocking));
			st->services = &rank->services;
			st->tp_degree = kRanks;
			st->tp_rank = r;
			st->max_rows = PROBE_ROWS;
			st->slots = PROBE_SEQUENCES;
			st->context_capacity = PROBE_CONTEXT;
			st->collective_capacity = 2u * PROBE_ROWS + 1u;
			st->kda_state_slot_bytes = probe->state->kda_state_slot_bytes;
			st->kda_window_slot_bytes = probe->state->kda_window_slot_bytes;
			st->verify_rows = ProbeGeometry::kVerifyRows;
			memcpy(st->layers, probe->state->layers, sizeof(st->layers));
			st->final_norm = probe->state->final_norm;
			REQUIRE(Glm5NextAllocate(st, 0u) == SPARK_STATUS_OK);
			rank->latent_pool = (uint8_t *)Device(latent_bytes * G5N_DSA_LAYERS);
			rank->index_pool = (uint8_t *)Device(index_bytes * G5N_DSA_LAYERS);
			memset(&kv, 0, sizeof(kv));
			kv.pool = rank->latent_pool;
			kv.layer_stride_bytes = latent_bytes;
			kv.layer_page_bytes = Glm5NextKv::kPageBytes;
			kv.layer_count = G5N_DSA_LAYERS;
			kv.page_table = page_table;
			kv.page_table_stride = PROBE_PAGES;
			kv.pool_page_count = PROBE_SEQUENCES * PROBE_PAGES;
			kv.sequence_count = PROBE_SEQUENCES;
			kv.context_shard = SparkGlm5NextKvShardLatent(r, kRanks);
			kv.second_pool = rank->index_pool;
			kv.second_layer_stride_bytes = index_bytes;
			kv.second_layer_page_bytes = Glm5NextIndexKv::kPageBytes;
			kv.second_layer_count = G5N_DSA_LAYERS;
			REQUIRE(Stage::AttachKv(st, &kv) == SPARK_STATUS_OK);
			rank->metadata = (uint32_t *)Device((4u * PROBE_ROWS + 8u) * sizeof(uint32_t));
			rank->hidden = (uint16_t *)Device((uint64_t)PROBE_ROWS * G5N_HIDDEN * 2u);
			rank->route_fault = (uint32_t *)Device(sizeof(uint32_t));
		}
	}

	void Reset(void)
	{
		for ( uint32_t r = 0u; r < kRanks; ++r )
		{
			CUDA(cudaMemsetAsync(ranks[r].latent_pool, 0, latent_bytes * G5N_DSA_LAYERS, ranks[r].stream));
			CUDA(cudaMemsetAsync(ranks[r].index_pool, 0, index_bytes * G5N_DSA_LAYERS, ranks[r].stream));
			CUDA(cudaMemsetAsync(ranks[r].state->kda_state, 0, (uint64_t)G5N_KDA_LAYERS * PROBE_SEQUENCES * ranks[r].state->kda_state_slot_bytes, ranks[r].stream));
			CUDA(cudaMemsetAsync(ranks[r].state->kda_windows, 0, 3u * (uint64_t)G5N_KDA_LAYERS * PROBE_SEQUENCES * ranks[r].state->kda_window_slot_bytes, ranks[r].stream));
			CUDA(cudaStreamSynchronize(ranks[r].stream));
		}
		CUDA(cudaDeviceSynchronize());
	}

	void RunAll(const uint16_t *inputs, uint32_t first, uint32_t rows, uint32_t gather_context, uint32_t last_only, uint32_t single_dsa)
	{
		std::vector<uint32_t> meta(4u * PROBE_ROWS + 8u, 0u);
		pthread_t threads[kRanks];
		for ( uint32_t row = 0u; row < rows; ++row )
		{
			meta[row] = first + row;
			meta[2u * PROBE_ROWS + row] = row;
		}
		meta[3u * PROBE_ROWS] = first + rows;
		meta[3u * PROBE_ROWS + 3u] = rows;
		meta[3u * PROBE_ROWS + 4u] = PROBE_SLOT;
		barrier.failed = 0u;
		for ( uint32_t r = 0u; r < kRanks; ++r )
		{
			Rank *rank = &ranks[r];
			CUDA(cudaMemcpyAsync(rank->metadata, meta.data(), meta.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, rank->stream));
			CUDA(cudaMemcpyAsync(rank->hidden, inputs + (uint64_t)first * G5N_HIDDEN, (uint64_t)rows * G5N_HIDDEN * 2u,
				cudaMemcpyDeviceToDevice, rank->stream));
			memset(&rank->step, 0, sizeof(rank->step));
			rank->step.hidden_bf16 = rank->hidden;
			rank->step.positions = rank->metadata;
			rank->step.sequence_of_row = rank->metadata + PROBE_ROWS;
			rank->step.sequence_row_indices = rank->metadata + 2u * PROBE_ROWS;
			rank->step.context_length = rank->metadata + 3u * PROBE_ROWS;
			rank->step.sequence_row_begin = rank->metadata + 3u * PROBE_ROWS + 2u;
			rank->step.recurrent_index = rank->metadata + 3u * PROBE_ROWS + 4u;
			rank->step.rows = rows;
			rank->step.sequences = 1u;
			rank->step.commit = 1u;
			rank->step.last_rows_only = last_only;
			rank->step.every_row_hidden = 1u;
			rank->step.context = PROBE_CONTEXT;
			rank->step.gather_context = gather_context;
			rank->step.multiprocessors = sms;
			rank->step.route_fault = rank->route_fault;
			rank->single_dsa = single_dsa;
			CUDA(cudaStreamSynchronize(rank->stream));
		}
		for ( uint32_t r = 0u; r < kRanks; ++r )
			REQUIRE(pthread_create(&threads[r], 0, Thread, &ranks[r]) == 0);
		for ( uint32_t r = 0u; r < kRanks; ++r )
		{
			REQUIRE(pthread_join(threads[r], 0) == 0);
			REQUIRE(ranks[r].status == SPARK_STATUS_OK);
		}
		REQUIRE(Finite(Host(ranks[0].hidden, (uint64_t)rows * G5N_HIDDEN * 2u)) != 0u);
	}

	void Close(void)
	{
		for ( uint32_t r = 0u; r < kRanks; ++r )
		{
			Glm5NextFree(ranks[r].state);
			free(ranks[r].state);
			CUDA(cudaFree(ranks[r].latent_pool));
			CUDA(cudaFree(ranks[r].index_pool));
			CUDA(cudaFree(ranks[r].metadata));
			CUDA(cudaFree(ranks[r].hidden));
			CUDA(cudaFree(ranks[r].route_fault));
			CUDA(cudaStreamDestroy(ranks[r].stream));
		}
		CUDA(cudaFree(staging));
		CUDA(cudaFree(page_table));
		pthread_mutex_destroy(&barrier.lock);
		pthread_cond_destroy(&barrier.cond);
	}
};

static void Tp16GateGxAndParityCheck(const Probe *probe)
{
	static const uint32_t kDeltas[] = { 1u, 4u, 8u, 12u, 16u, 23u, 64u, 128u };
	static constexpr uint32_t kCached = 128u;
	const uint64_t row_bytes = (uint64_t)G5N_HIDDEN * 2u;
	Tp16Rig rig;
	rig.Open(probe);
	rig.Reset();
	rig.RunAll(probe->inputs, 0u, kCached, kCached, 1u, 0u);
	std::vector<std::vector<uint8_t>> cached_lat(Tp16Rig::kRanks), cached_idx(Tp16Rig::kRanks), cached_kda(Tp16Rig::kRanks), cached_win(Tp16Rig::kRanks);
	const uint64_t kda_bytes = (uint64_t)G5N_KDA_LAYERS * PROBE_SEQUENCES * rig.ranks[0].state->kda_state_slot_bytes;
	const uint64_t win_bytes = 3u * (uint64_t)G5N_KDA_LAYERS * PROBE_SEQUENCES * rig.ranks[0].state->kda_window_slot_bytes;
	for ( uint32_t r = 0u; r < Tp16Rig::kRanks; ++r )
	{
		cached_lat[r] = Host(rig.ranks[r].latent_pool, rig.latent_bytes * G5N_DSA_LAYERS);
		cached_idx[r] = Host(rig.ranks[r].index_pool, rig.index_bytes * G5N_DSA_LAYERS);
		cached_kda[r] = Host(rig.ranks[r].state->kda_state, kda_bytes);
		cached_win[r] = Host(rig.ranks[r].state->kda_windows, win_bytes);
	}
	for ( uint32_t delta : kDeltas )
	{
		const uint32_t total = kCached + delta;
		rig.Reset();
		rig.RunAll(probe->inputs, 0u, total, total, 1u, 0u);
		const std::vector<uint8_t> cold_all = Host(rig.ranks[0].hidden, (uint64_t)total * row_bytes);
		const std::vector<uint8_t> cold_tail(cold_all.begin() + (ptrdiff_t)((uint64_t)kCached * row_bytes), cold_all.end());
		const std::vector<uint8_t> cold_lat0 = Host(rig.ranks[0].latent_pool, rig.latent_bytes * G5N_DSA_LAYERS);
		const std::vector<uint8_t> cold_kda0 = Host(rig.ranks[0].state->kda_state, kda_bytes);
		for ( uint32_t r = 0u; r < Tp16Rig::kRanks; ++r )
		{
			CUDA(cudaMemcpyAsync(rig.ranks[r].latent_pool, cached_lat[r].data(), cached_lat[r].size(), cudaMemcpyHostToDevice, rig.ranks[r].stream));
			CUDA(cudaMemcpyAsync(rig.ranks[r].index_pool, cached_idx[r].data(), cached_idx[r].size(), cudaMemcpyHostToDevice, rig.ranks[r].stream));
			CUDA(cudaMemcpyAsync(rig.ranks[r].state->kda_state, cached_kda[r].data(), kda_bytes, cudaMemcpyHostToDevice, rig.ranks[r].stream));
			CUDA(cudaMemcpyAsync(rig.ranks[r].state->kda_windows, cached_win[r].data(), win_bytes, cudaMemcpyHostToDevice, rig.ranks[r].stream));
			CUDA(cudaStreamSynchronize(rig.ranks[r].stream));
		}
		rig.RunAll(probe->inputs, kCached, delta, total, 1u, 0u);
		const std::vector<uint8_t> warm_tail = Host(rig.ranks[0].hidden, (uint64_t)delta * row_bytes);
		Mismatch(Differ(warm_tail, cold_tail), "TP16 Gate GX warm delta prefill differs from cold prefill", delta, kCached, total);
		Mismatch(Differ(Host(rig.ranks[0].latent_pool, rig.latent_bytes * G5N_DSA_LAYERS), cold_lat0), "TP16 Gate GX warm KV differs from cold", delta, kCached, total);
		Mismatch(Differ(Host(rig.ranks[0].state->kda_state, kda_bytes), cold_kda0), "TP16 Gate GX warm KDA state differs from cold", delta, kCached, total);
	}
	printf("TP16 Gate GX prefix-exact: 16 ranks, 45 layers, cached=%u vs warm deltas 1..128 bit-exact (0 diffs in hidden, KV, and KDA state)\n", kCached);
	const auto compare_rows = [&](const std::vector<uint8_t> &a, const std::vector<uint8_t> &b, uint32_t check_rows, const char *label)
	{
		double dot = 0.0, na = 0.0, nb = 0.0, max_abs = 0.0;
		const uint16_t *pa = (const uint16_t *)a.data(), *pb = (const uint16_t *)b.data();
		const uint64_t count = (uint64_t)check_rows * G5N_HIDDEN;
		for ( uint64_t i = 0u; i < count; ++i )
		{
			const uint32_t ua = (uint32_t)pa[i] << 16u, ub = (uint32_t)pb[i] << 16u;
			float fa = 0.0f, fb = 0.0f;
			memcpy(&fa, &ua, sizeof(fa));
			memcpy(&fb, &ub, sizeof(fb));
			const double va = fa, vb = fb, d = fabs(va - vb);
			if ( d > max_abs )
				max_abs = d;
			dot += va * vb;
			na += va * va;
			nb += vb * vb;
		}
		const double cos_sim = dot / sqrt(na * nb);
		printf("TP16 gather parity (%s): cos=%.6f max_abs=%.4f\n", label, cos_sim, max_abs);
		REQUIRE(cos_sim > 0.9995);
	};
	rig.Reset();
	rig.RunAll(probe->inputs, 0u, 64u, 64u, 0u, 1u);
	const std::vector<uint8_t> dense_scatter = Host(rig.ranks[0].hidden, 64u * row_bytes);
	rig.Reset();
	rig.RunAll(probe->inputs, 0u, 64u, 64u, 1u, 1u);
	const std::vector<uint8_t> dense_gather = Host(rig.ranks[0].hidden, 64u * row_bytes);
	compare_rows(dense_gather, dense_scatter, 64u, "dense_cold_64");
	for ( uint32_t pos = 64u; pos < PROBE_PREFIX; pos += PROBE_ROWS )
	{
		const uint32_t chunk = PROBE_PREFIX - pos < PROBE_ROWS ? PROBE_PREFIX - pos : PROBE_ROWS;
		rig.RunAll(probe->inputs, pos, chunk, pos + chunk, 1u, 1u);
	}
	for ( uint32_t r = 0u; r < Tp16Rig::kRanks; ++r )
	{
		cached_lat[r] = Host(rig.ranks[r].latent_pool, rig.latent_bytes * G5N_DSA_LAYERS);
		cached_idx[r] = Host(rig.ranks[r].index_pool, rig.index_bytes * G5N_DSA_LAYERS);
	}
	rig.RunAll(probe->inputs, PROBE_PREFIX, 64u, PROBE_PREFIX + 64u, 0u, 1u);
	const std::vector<uint8_t> sparse_scatter = Host(rig.ranks[0].hidden, 64u * row_bytes);
	for ( uint32_t r = 0u; r < Tp16Rig::kRanks; ++r )
	{
		CUDA(cudaMemcpyAsync(rig.ranks[r].latent_pool, cached_lat[r].data(), cached_lat[r].size(), cudaMemcpyHostToDevice, rig.ranks[r].stream));
		CUDA(cudaMemcpyAsync(rig.ranks[r].index_pool, cached_idx[r].data(), cached_idx[r].size(), cudaMemcpyHostToDevice, rig.ranks[r].stream));
		CUDA(cudaStreamSynchronize(rig.ranks[r].stream));
	}
	rig.RunAll(probe->inputs, PROBE_PREFIX, 64u, PROBE_PREFIX + 64u, 1u, 1u);
	const std::vector<uint8_t> sparse_gather = Host(rig.ranks[0].hidden, 64u * row_bytes);
	compare_rows(sparse_gather, sparse_scatter, 64u, "sparse_warm_2304+64");
	rig.Close();
}

int main(void)
{
	static const uint32_t kFrames[] = {1u, 2u, 7u, 16u, PROBE_ROWS};
	Probe probe;
	Frame verify, decode;
	memset(&probe, 0, sizeof(probe));
	ProbeOpen(&probe);
	FrameOpen(&verify, &probe);
	FrameOpen(&decode, &probe);
	void *start = Device(probe.recurrent_bytes), *scratch = Device(probe.recurrent_bytes), *current = Device(probe.recurrent_bytes);
	Tp16GateGxAndParityCheck(&probe);
	Prefill(&probe, &decode, 0u, PROBE_BOUNDARY);
	Sweep(&probe, &verify, &decode, start, scratch, current, PROBE_BOUNDARY, 16u);
	Prefill(&probe, &decode, PROBE_BOUNDARY, PROBE_PREFIX);
	for ( uint32_t rows : kFrames )
		Sweep(&probe, &verify, &decode, start, scratch, current, PROBE_PREFIX, rows);
	Rejection(&probe, &verify, &decode, start, scratch, 16u, 5u);
	Rejection(&probe, &verify, &decode, start, scratch, PROBE_ROWS, 0u);
	TreeSweep(&probe, &verify, &decode, start, scratch, PROBE_PREFIX, 1u);
	LargeTreeCheck(&probe, &verify, &decode, start, PROBE_PREFIX);
	Prefill(&probe, &decode, PROBE_PREFIX, PROBE_PREFIX + 14u);
	Recurrent(&probe, 1u, start);
	TreeSweep(&probe, &verify, &decode, start, scratch, PROBE_PREFIX + 14u, 0u);
	printf("PASS glm5_next verify: frames of 1, 2, 7, 16 and %u rows, and a 16-row frame entering the top-k regime, equal one-row decode steps "
		"bit for bit in every row, in the recurrent state left by every keep and in the cache rows; a rejected tail leaves no trace\n", PROBE_ROWS);
	return 0;
}
