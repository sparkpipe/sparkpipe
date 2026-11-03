#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "sparkpipe/spark_glm53_flash_tp16_stage_model.h"
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
	uint8_t *dump;
};

struct Frame
{
	SparkStageRunnerStep step;
	uint32_t *metadata;
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
	FillBf16(probe->inputs, (uint64_t)PROBE_POSITIONS * G5N_HIDDEN, 1u, 0.0f, 1.7f);
	FillBf16(probe->drafts, (uint64_t)PROBE_POSITIONS * G5N_HIDDEN, 2u, 0.0f, 1.7f);
	probe->dump = (uint8_t *)Device((uint64_t)PROBE_POSITIONS * Glm5NextKv::kSlotBytes);
	probe->state = state;
}

static void FrameOpen(Frame *frame, const Probe *probe)
{
	frame->metadata = (uint32_t *)Device((4u * PROBE_ROWS + 8u) * sizeof(uint32_t));
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

static void Prefill(Probe *probe, Frame *frame, uint32_t first, uint32_t end)
{
	for ( ; first < end; first += PROBE_ROWS )
	{
		const uint32_t rows = end - first < PROBE_ROWS ? end - first : PROBE_ROWS;
		Run(probe, frame, first, rows, 1u, first + rows, rows);
	}
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
	Prefill(&probe, &decode, 0u, PROBE_BOUNDARY);
	Sweep(&probe, &verify, &decode, start, scratch, current, PROBE_BOUNDARY, 16u);
	Prefill(&probe, &decode, PROBE_BOUNDARY, PROBE_PREFIX);
	for ( uint32_t rows : kFrames )
		Sweep(&probe, &verify, &decode, start, scratch, current, PROBE_PREFIX, rows);
	Rejection(&probe, &verify, &decode, start, scratch, 16u, 5u);
	Rejection(&probe, &verify, &decode, start, scratch, PROBE_ROWS, 0u);
	printf("PASS glm5_next verify: frames of 1, 2, 7, 16 and %u rows, and a 16-row frame entering the top-k regime, equal one-row decode steps "
		"bit for bit in every row, in the recurrent state left by every keep and in the cache rows; a rejected tail leaves no trace\n", PROBE_ROWS);
	return 0;
}
