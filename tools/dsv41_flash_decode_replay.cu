#include <algorithm>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "sparkpipe/spark_dsv41_flash_stage_model.h"
#include "dsv41_flash_stage.cuh"

#define REPLAY_SLOTS 16u
#define REPLAY_MAX_ROWS 16u
#define REPLAY_MAX_POSITIONS 8192u
#define REPLAY_SITES 2u
#define REPLAY_SPANS 8u
#define REPLAY_PIECE_BYTES 16384u
#define REPLAY_PREFETCH_THREADS 32u
#define REPLAY_FILL_FP8 1u
#define REPLAY_FILL_SCALE 2u
#define REPLAY_FILL_BF16 3u
#define REPLAY_FILL_F32 4u
#define REPLAY_FILL_CELLS 5u
#define REPLAY_CELL_BYTES (LM_SKINNY_CELL_ROWS * LM_SKINNY_CHUNK_BYTES)
#define REPLAY_CELL_CODES (LM_SKINNY_CELL_NEURONS * LM_SKINNY_CHUNK_BYTES)
#define REPLAY_CHAIN 48u
#define REPLAY_COPIES 16u

typedef struct ReplayRounds
{
	uint32_t layer, failures;
	uint64_t hold_ns[D41_LAYERS][REPLAY_SITES];
	uint4 *staging;
} ReplayRounds;

typedef struct ReplaySpans
{
	uint64_t address[REPLAY_SPANS];
	uint64_t bytes[REPLAY_SPANS];
	uint64_t budget;
	uint32_t count;
} ReplaySpans;

typedef struct ReplayFork
{
	uint32_t enabled, threads;
	uint64_t budget;
	SparkGpuStream side;
	SparkGpuEvent before, after;
} ReplayFork;

typedef struct ReplayOptions
{
	const char *mode, *holds;
	uint32_t rows, steps, warmup, position, prefetch_threads;
	uint64_t prefetch_bytes;
} ReplayOptions;

static void ReplayRequire(int valid, const char *what)
{
	if ( valid != 0 )
		return;
	fprintf(stderr, "REPLAY-FAIL %s\n", what);
	exit(1);
}

static void ReplayCuda(cudaError_t error, const char *what)
{
	if ( error == cudaSuccess )
		return;
	fprintf(stderr, "REPLAY-FAIL %s: %s\n", what, cudaGetErrorString(error));
	exit(1);
}

static __device__ __forceinline__ uint64_t ReplayClock(void)
{
	uint64_t ns;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns));
	return ns;
}

static __device__ __forceinline__ uint64_t ReplayMix(uint64_t value)
{
	value += 0x9e3779b97f4a7c15ull;
	value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
	value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
	return value ^ (value >> 31);
}

static __device__ __forceinline__ uint8_t ReplayByte(uint64_t index, uint64_t seed, uint32_t kind)
{
	if ( kind == REPLAY_FILL_FP8 )
	{
		const uint64_t random = ReplayMix(seed ^ index);
		return (uint8_t)((random & 0x80u) | ((3u + (uint32_t)((random >> 8) % 6u)) << 3) | ((random >> 16) & 7u));
	}
	if ( kind == REPLAY_FILL_SCALE )
		return (uint8_t)(121u + ReplayMix(seed ^ index) % 7u);
	if ( kind == REPLAY_FILL_CELLS )
		return index % REPLAY_CELL_BYTES >= REPLAY_CELL_CODES ? (uint8_t)(121u + ReplayMix(seed ^ index) % 7u) : (uint8_t)ReplayMix(seed ^ index);
	if ( kind == REPLAY_FILL_BF16 )
	{
		const uint64_t random = ReplayMix(seed ^ (index >> 1));
		const uint32_t value = (uint32_t)(random & 0x8000u) | ((120u + (uint32_t)((random >> 16) % 8u)) << 7) | (uint32_t)((random >> 24) & 0x7fu);
		return (uint8_t)(value >> (8u * (uint32_t)(index & 1u)));
	}
	const uint64_t random = ReplayMix(seed ^ (index >> 2));
	const uint32_t value = (uint32_t)(random & 0x80000000u) | ((118u + (uint32_t)((random >> 32) % 9u)) << 23) | (uint32_t)((random >> 8) & 0x7fffffu);
	return (uint8_t)(value >> (8u * (uint32_t)(index & 3u)));
}

static __global__ void ReplayFillKernel(uint8_t *data, uint64_t bytes, uint64_t seed, uint32_t kind)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < bytes; index += (uint64_t)gridDim.x * blockDim.x )
		data[index] = ReplayByte(index, seed, kind);
}

static __global__ void ReplayCopyKernel(const uint4 *__restrict__ source, uint4 *__restrict__ target, uint64_t vectors)
{
	for ( uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; index < vectors; index += (uint64_t)gridDim.x * blockDim.x )
		target[index] = source[index];
}

static __global__ void ReplayArrivalKernel(uint64_t hold_ns, LmWeightPrefetchGroup *prefetch, uint32_t resume)
{
	__shared__ uint32_t wait_over;
	const uint32_t lane = threadIdx.x;
	if ( lane == 0u )
		wait_over = 0u;
	__syncthreads();
	if ( lane >= 32u )
	{
		const volatile uint32_t *stop = &wait_over;
		const uint32_t spans = prefetch->count;
		const uint64_t extent = LmWeightPrefetchExtent(prefetch, spans), stride = (uint64_t)(blockDim.x - 32u) * LM_WEIGHT_PREFETCH_PIECE_BYTES;
		uint64_t position = resume != 0u ? prefetch->progress : 0u;
		for ( ; position < extent && *stop == 0u; position += stride )
			LmWeightPrefetchPiece(prefetch, spans, position + (uint64_t)(lane - 32u) * LM_WEIGHT_PREFETCH_PIECE_BYTES);
		if ( lane == 32u )
			prefetch->progress = position < extent ? position : extent;
		return;
	}
	const uint64_t started = ReplayClock();
	while ( ReplayClock() - started < hold_ns )
		;
	if ( lane == 0u )
		*(volatile uint32_t *)&wait_over = 1u;
}

static __global__ void ReplayPrefetchKernel(const __grid_constant__ ReplaySpans spans)
{
	uint64_t taken = 0u;
	for ( uint32_t span = 0u; span < spans.count && taken < spans.budget; ++span )
	{
		const uint64_t bytes = spans.bytes[span] < spans.budget - taken ? spans.bytes[span] : spans.budget - taken;
		for ( uint64_t offset = (uint64_t)threadIdx.x * REPLAY_PIECE_BYTES; offset < bytes; offset += (uint64_t)blockDim.x * REPLAY_PIECE_BYTES )
			LmPrefetchL2((const void *)(uintptr_t)(spans.address[span] + offset), (uint32_t)(bytes - offset < REPLAY_PIECE_BYTES ? bytes - offset : REPLAY_PIECE_BYTES));
		taken += bytes;
	}
}

static __global__ void ReplayMarkKernel(uint32_t mark, uint32_t *sink)
{
	if ( threadIdx.x == 0u && mark == 0xffffffffu )
		*sink = mark;
}

static __global__ void ReplayTinyKernel(const uint4 *__restrict__ source, uint4 *__restrict__ target, uint32_t vectors)
{
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < vectors; index += gridDim.x * blockDim.x )
		target[index] = source[index];
}

static __global__ void ReplayTinyReleaseKernel(const uint4 *__restrict__ source, uint4 *__restrict__ target, uint32_t vectors)
{
	LmDependentRelease();
	for ( uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < vectors; index += gridDim.x * blockDim.x )
		target[index] = source[index];
}

template<bool LOAD_FIRST>
__global__ __launch_bounds__(256) void ReplayStreamKernel(const uint4 *__restrict__ weight, uint64_t vectors, const uint4 *__restrict__ activation,
	uint4 *__restrict__ sink)
{
	const uint64_t first = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x, step = (uint64_t)gridDim.x * blockDim.x;
	uint4 head = make_uint4(0u, 0u, 0u, 0u);
	if ( LOAD_FIRST && first < vectors )
		head = __ldcs(weight + first);
	LmDependentWait();
	if ( !LOAD_FIRST && first < vectors )
		head = __ldcs(weight + first);
	uint32_t folded = head.x ^ head.y ^ head.z ^ head.w ^ activation[threadIdx.x % 64u].x;
	for ( uint64_t index = first + step; index < vectors; index += step )
	{
		const uint4 value = __ldcs(weight + index);
		folded ^= value.x ^ value.y ^ value.z ^ value.w;
	}
	if ( folded == 0x5eed5eedu )
		sink[0] = make_uint4(folded, 0u, 0u, 0u);
}

static void *ReplayAllocate(uint64_t bytes)
{
	void *pointer = 0;
	ReplayCuda(cudaMalloc(&pointer, bytes), "allocate");
	return pointer;
}

static void ReplayFill(void *pointer, uint64_t bytes, uint32_t kind, uint64_t seed)
{
	LM_LAUNCH((ReplayFillKernel), 1024u, 256u, 0u, (SparkGpuStream)0, (uint8_t *)pointer, bytes, seed, kind);
	ReplayCuda(cudaGetLastError(), "fill");
}

static void *ReplayFilled(uint64_t bytes, uint32_t kind, uint64_t seed)
{
	void *pointer = ReplayAllocate(bytes);
	ReplayFill(pointer, bytes, kind, seed);
	return pointer;
}

static D41Fp8Linear ReplayLinear(uint32_t outputs, uint32_t inputs, uint64_t seed)
{
	D41Fp8Linear linear;
	memset(&linear, 0, sizeof(linear));
	linear.weight = (const uint8_t *)ReplayFilled((uint64_t)outputs * inputs, REPLAY_FILL_FP8, seed);
	linear.scale = (const uint8_t *)ReplayFilled((uint64_t)outputs * (inputs / LM_MXFP8_GROUP), REPLAY_FILL_SCALE, seed + 1u);
	return linear;
}

static D41Layer ReplayLayer(uint32_t slot, uint32_t shared_width, const uint8_t *cells13, const uint8_t *cells2)
{
	const uint64_t seed = 0x1000u + 0x100u * (uint64_t)slot;
	const uint64_t fn = (uint64_t)LmHcSinkhornShape<D41K_HC>::kMix * D41K_HC * D41_HIDDEN * sizeof(float);
	D41Layer layer;
	memset(&layer, 0, sizeof(layer));
	layer.attn_norm = (const uint16_t *)ReplayFilled(D41_HIDDEN * 2u, REPLAY_FILL_BF16, seed + 1u);
	layer.ffn_norm = (const uint16_t *)ReplayFilled(D41_HIDDEN * 2u, REPLAY_FILL_BF16, seed + 2u);
	layer.q_norm = (const uint16_t *)ReplayFilled(D41_QLORA * 2u, REPLAY_FILL_BF16, seed + 3u);
	layer.kv_norm = (const uint16_t *)ReplayFilled(D41_HEAD_DIM * 2u, REPLAY_FILL_BF16, seed + 4u);
	layer.q_a = ReplayLinear(D41_QLORA, D41_HIDDEN, seed + 10u);
	layer.kv_a = ReplayLinear(D41_HEAD_DIM, D41_HIDDEN, seed + 12u);
	layer.q_b = ReplayLinear(D41_LOCAL_HEADS * D41_HEAD_DIM, D41_QLORA, seed + 14u);
	for ( uint32_t group = 0u; group < D41_LOCAL_GROUPS; ++group )
		layer.o_a[group] = ReplayLinear(D41_GROUP_ROWS, D41_GROUP_WIDTH, seed + 16u + 2u * group);
	layer.o_b = ReplayLinear(D41_HIDDEN, D41_WO_A_ROWS, seed + 30u);
	layer.sink = (const float *)ReplayFilled(D41_HEADS * sizeof(float), REPLAY_FILL_F32, seed + 32u);
	layer.group_sink = layer.sink;
	layer.hc_attn_fn = (const float *)ReplayFilled(fn, REPLAY_FILL_F32, seed + 33u);
	layer.hc_attn_base = (const float *)ReplayFilled(LmHcSinkhornShape<D41K_HC>::kMix * sizeof(float), REPLAY_FILL_F32, seed + 34u);
	layer.hc_attn_scale = (const float *)ReplayFilled(4u * sizeof(float), REPLAY_FILL_F32, seed + 35u);
	layer.hc_ffn_fn = (const float *)ReplayFilled(fn, REPLAY_FILL_F32, seed + 36u);
	layer.hc_ffn_base = (const float *)ReplayFilled(LmHcSinkhornShape<D41K_HC>::kMix * sizeof(float), REPLAY_FILL_F32, seed + 37u);
	layer.hc_ffn_scale = (const float *)ReplayFilled(4u * sizeof(float), REPLAY_FILL_F32, seed + 38u);
	layer.router = (const uint16_t *)ReplayFilled((uint64_t)D41_EXPERTS * D41_HIDDEN * 2u, REPLAY_FILL_BF16, seed + 39u);
	layer.router_bias = (const float *)ReplayFilled(D41_EXPERTS * sizeof(float), REPLAY_FILL_F32, seed + 40u);
	layer.e_w13_cells = cells13;
	layer.e_w2_cells = cells2;
	layer.sh_w1 = ReplayLinear(shared_width, D41_HIDDEN, seed + 41u);
	layer.sh_w3 = ReplayLinear(shared_width, D41_HIDDEN, seed + 43u);
	layer.sh_w2 = ReplayLinear(D41_HIDDEN, shared_width, seed + 45u);
	return layer;
}

static uint32_t ReplayIndex(uint32_t slot)
{
	if ( D41EngramIndex(slot) >= 0 && !D41KvSource(slot) )
		return slot == 1u ? 0u : slot + 1u;
	if ( D41KvSource(slot) || D41IndexSource(slot) || D41EngramIndex(slot) >= 0 )
		return slot + 1u;
	return slot;
}

static SparkStatus ReplayRound(void *context, void *stream_pointer, uint32_t operation, uint32_t stage, uint32_t rows, uint32_t row_elements,
	const void *local, void *full)
{
	ReplayRounds *rounds = (ReplayRounds *)context;
	const SparkGpuStream stream = (SparkGpuStream)stream_pointer;
	uint32_t site, resume = 0u;
	uint64_t vectors;
	(void)stage;
	if ( operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16 && local == full )
	{
		site = 0u;
		vectors = (uint64_t)rows * D41_HIDDEN * 2u / 16u;
	}
	else if ( operation == SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER && row_elements == D41_HIDDEN )
	{
		site = 1u;
		vectors = (uint64_t)rows * row_elements * 2u / 16u;
	}
	else
	{
		fprintf(stderr, "REPLAY-FAIL layer %u asked for collective %u (%u rows of %u); the replay stands in for the attention reduce and the MoE gather only\n",
			rounds->layer, operation, rows, row_elements);
		rounds->failures++;
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	}
	LmWeightPrefetchGroup *group;
	LM_LAUNCH((ReplayCopyKernel), 5u, 256u, 0u, stream, (const uint4 *)local, rounds->staging, vectors);
	group = LmWeightPrefetchSlot(stream, &resume);
	LM_LAUNCH((ReplayArrivalKernel), 1u, group != 0 ? 32u + REPLAY_PREFETCH_THREADS : 32u, 0u, stream, rounds->hold_ns[rounds->layer][site], group, resume);
	LM_LAUNCH((ReplayCopyKernel), 4u, 256u, 0u, stream, (const uint4 *)rounds->staging, (uint4 *)full, vectors);
	return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static void ReplayHolds(ReplayRounds *rounds, const char *path, uint32_t rows)
{
	const uint32_t want = rows <= 2u ? 1u : 8u;
	uint32_t file_rows, layer, filled = 0u;
	char site[32];
	unsigned long long hold;
	FILE *file = fopen(path, "r");
	ReplayRequire(file != 0, "the holds file opens");
	while ( fscanf(file, "%u %u %31s %llu", &file_rows, &layer, site, &hold) == 4 )
	{
		if ( file_rows != want || layer >= D41_LAYERS )
			continue;
		rounds->hold_ns[layer][strcmp(site, "moe") == 0 ? 1u : 0u] = hold;
		filled++;
	}
	fclose(file);
	ReplayRequire(filled == D41_LAYERS * REPLAY_SITES, "the holds file gives both sites of every layer for the row count");
}

static SparkStatus ReplayAttention(D41ModelState *state, const D41Layer *layer, uint32_t index, const SparkStageRunnerStep *step, SparkGpuStream stream,
	const ReplayFork *fork)
{
	const uint32_t rows = step->rows, sms = step->multiprocessors, ratio = D41Ratio(index), cache = D41CacheIndex(index);
	const float *frequency = ratio > 0u ? state->compress_frequency : state->full_frequency;
	const uint32_t *bound = ratio == 2u ? state->bound_ratio2 : state->bound_ratio1;
	const uint32_t *length = ratio == 2u ? state->length_ratio2 : state->length_ratio1;
	const uint32_t local_width = D41_LOCAL_HEADS * D41_HEAD_DIM;
	const uint64_t stride = D41DestinationStride(rows, D41_LOCAL_HEADS), query_stride = (uint64_t)rows * local_width;
	const float qk_scale = 1.0f / sqrtf((float)D41_HEAD_DIM);
	uint16_t *ring = (uint16_t *)state->recurrent + (uint64_t)index * D41_RING_ROWS * D41_HEAD_DIM;
	if ( D41LinearPair(&layer->q_a, &layer->kv_a, state->xq, state->q_latent, state->kv, rows, D41_HIDDEN, D41_QLORA, D41_HEAD_DIM, sms,
		state->row_invariant, &state->mx, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41RmsNormQdq(state->q_latent, layer->q_norm, 0, state->qrq, rows, D41_QLORA, stream);
	LM_LAUNCH((D41KvNormRopeQdqKernel<D41_THREADS>), rows, D41_THREADS, (D41_HEAD_DIM + 8u) * sizeof(float), stream, state->kv, layer->kv_norm,
		step->positions, frequency, state->window_kv, D41_HEAD_DIM, D41_HEAD_DIM - D41_ROPE, D41_EPS);
	if ( D41Linear(&layer->q_b, state->qrq, state->q_local, rows, D41_QLORA, local_width, sms, state->row_invariant, &state->mx, stream) != LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	D41Rope(state->q_local, rows, D41_LOCAL_HEADS, D41_HEAD_DIM, D41_HEAD_DIM - D41_ROPE, step->positions, 0, frequency, 0u, 0, stream);
	if ( ratio > 0u )
	{
		if ( (cache < 3u ? LmLatentShardAttendHeadsLaunch<D41Ratio2Kv,LmKvShardView,D41_HEAD_DIM,0u>(state->latent_views[cache], state->q_local,
				query_stride, D41_LOCAL_HEADS, D41_LOCAL_HEADS, step->sequence_of_row, length, bound, state->selected, D41_INDEX_TOPK, D41_INDEX_TOPK,
				qk_scale, state->records_received, stride, D41_LOCAL_HEADS, 1u, rows, stream) :
			LmLatentShardAttendHeadsLaunch<D41Ratio1Kv,LmKvShardView,D41_HEAD_DIM,0u>(state->latent_views[cache], state->q_local,
				query_stride, D41_LOCAL_HEADS, D41_LOCAL_HEADS, step->sequence_of_row, length, bound, state->selected, D41_INDEX_TOPK, D41_INDEX_TOPK,
				qk_scale, state->records_received, stride, D41_LOCAL_HEADS, 1u, rows, stream)) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	else
		D41InitRecords(state->records_received, stride, rows, D41_LOCAL_HEADS, 1u, stream);
	if ( fork->enabled != 0u )
	{
		ReplaySpans spans;
		memset(&spans, 0, sizeof(spans));
		for ( uint32_t group = 0u; group < D41_LOCAL_GROUPS; ++group )
		{
			spans.address[spans.count] = (uint64_t)(uintptr_t)layer->o_a[group].weight;
			spans.bytes[spans.count++] = (uint64_t)D41_GROUP_ROWS * D41_GROUP_WIDTH;
			spans.address[spans.count] = (uint64_t)(uintptr_t)layer->o_a[group].scale;
			spans.bytes[spans.count++] = (uint64_t)D41_GROUP_ROWS * (D41_GROUP_WIDTH / LM_MXFP8_GROUP);
		}
		spans.address[spans.count] = (uint64_t)(uintptr_t)layer->o_b.weight;
		spans.bytes[spans.count++] = (uint64_t)D41_HIDDEN * D41_WO_A_ROWS;
		spans.address[spans.count] = (uint64_t)(uintptr_t)layer->o_b.scale;
		spans.bytes[spans.count++] = (uint64_t)D41_HIDDEN * (D41_WO_A_ROWS / LM_MXFP8_GROUP);
		spans.budget = fork->budget;
		if ( SparkGpuEventRecord(fork->before, stream) != SPARK_GPU_SUCCESS || SparkGpuStreamWaitEvent(fork->side, fork->before, 0u) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LM_LAUNCH((ReplayPrefetchKernel), 1u, fork->threads, 0u, fork->side, spans);
		if ( SparkGpuEventRecord(fork->after, fork->side) != SPARK_GPU_SUCCESS )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	LM_LAUNCH((D41WindowFoldWholeKernel), dim3(rows, D41_LOCAL_HEADS), D41K_THREADS, 0, stream, state->records_received, D41_LOCAL_HEADS, state->q_local,
		state->window_kv, ring, step->positions, state->row_lane, state->row_ordinal, step->sequence_row_begin, step->sequence_row_indices,
		step->recurrent_index, state->recurrent_bytes / sizeof(uint16_t), D41_WINDOW, qk_scale);
	LM_LAUNCH((D41WindowRingStoreKernel), rows, D41_THREADS, 0, stream, state->window_kv, ring, step->positions, state->row_lane, state->row_ordinal,
		step->sequence_row_begin, step->recurrent_index, state->recurrent_bytes / sizeof(uint16_t), D41_RING_ROWS, D41_WINDOW, 1u, 0u, rows,
		step->commit, (const uint32_t *)0);
	LM_LAUNCH((D41MergeSinkKernel), dim3(rows, D41_LOCAL_HEADS), D41_THREADS, 0, stream, state->records_received, stride, 1u, D41_LOCAL_HEADS, layer->sink,
		state->attn_merged);
	D41Rope(state->attn_merged, rows, D41_LOCAL_HEADS, D41_HEAD_DIM, D41_HEAD_DIM - D41_ROPE, step->positions, 0, frequency, 1u, 0, stream);
	for ( uint32_t group = 0u; group < D41_LOCAL_GROUPS; ++group )
	{
		LM_LAUNCH((D41ColumnsKernel), dim3(D41Blocks(D41_GROUP_WIDTH), rows), D41_THREADS, 0, stream, state->attn_merged + (uint64_t)group * D41_GROUP_WIDTH,
			local_width, state->group_input, D41_GROUP_WIDTH, D41_GROUP_WIDTH);
		if ( D41QdqLinear(&layer->o_a[group], state->group_input, state->wo_a_group, rows, D41_GROUP_WIDTH, D41_GROUP_ROWS, sms, state->row_invariant,
			&state->mx, stream) != LM_LAUNCH_OK )
			SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
		LM_LAUNCH((D41ColumnsKernel), dim3(D41Blocks(D41_GROUP_ROWS), rows), D41_THREADS, 0, stream, state->wo_a_group, D41_GROUP_ROWS,
			state->wo_a_out + (uint64_t)group * D41_GROUP_ROWS, D41_WO_A_ROWS, D41_GROUP_ROWS);
	}
	if ( D41QdqLinear(&layer->o_b, state->wo_a_out, state->partial, rows, D41_WO_A_ROWS, D41_HIDDEN, sms, state->row_invariant, &state->mx, stream) !=
		LM_LAUNCH_OK )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	if ( fork->enabled != 0u && SparkGpuStreamWaitEvent(stream, fork->after, 0u) != SPARK_GPU_SUCCESS )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	state->owner_reduced = 0u;
	return D41Launched();
}

static SparkStatus ReplayLayerBody(D41ModelState *state, D41Cuda::Frame &frame, uint32_t index, const ReplayFork *fork)
{
	SparkStatus status;
	if ( fork == 0 )
		return D41Schedule<D41Cuda>::Body(state, frame, index);
	status = D41Cuda::HyperMix(state, frame, index, D41_ATTENTION_PART);
	if ( status == SPARK_STATUS_OK )
		status = D41Cuda::CollapseNorm(state, frame, index, D41_ATTENTION_PART);
	if ( status == SPARK_STATUS_OK )
		status = ReplayAttention(state, &state->layers[index], index, frame.current, frame.stream, fork);
	if ( status == SPARK_STATUS_OK )
		status = D41Cuda::AttentionReduce(state, frame, index);
	if ( status == SPARK_STATUS_OK )
		status = D41Cuda::HyperPost(state, frame, D41_ATTENTION_PART);
	if ( status == SPARK_STATUS_OK )
		status = D41Cuda::HyperMix(state, frame, index, D41_FFN_PART);
	if ( status == SPARK_STATUS_OK )
		status = D41Cuda::CollapseNorm(state, frame, index, D41_FFN_PART);
	if ( status == SPARK_STATUS_OK )
		status = D41Cuda::Moe(state, frame, index);
	if ( status == SPARK_STATUS_OK )
		status = D41Cuda::HyperPost(state, frame, D41_FFN_PART);
	if ( status == SPARK_STATUS_OK )
		status = D41Cuda::Tap(state, frame, index);
	return status == SPARK_STATUS_OK ? D41Cuda::Finish(state, frame, index) : status;
}

typedef struct ReplayWorld
{
	D41ModelState *state;
	D41Layer weights[D41_LAYERS];
	SparkStageRunnerServices services;
	ReplayRounds rounds;
	SparkStageRunnerStep step;
	LmWeightPrefetchPlan plan;
	uint32_t *mark_sink;
	uint16_t *initial_streams;
} ReplayWorld;

static uint32_t *ReplayDeviceWords(const std::vector<uint32_t> &words)
{
	uint32_t *device = (uint32_t *)ReplayAllocate((uint64_t)(words.empty() ? 1u : words.size()) * sizeof(uint32_t));
	if ( !words.empty() )
		ReplayCuda(cudaMemcpy(device, words.data(), words.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), "copy words");
	return device;
}

static void ReplaySetup(ReplayWorld *world, const ReplayOptions *options, uint32_t multiprocessors)
{
	D41ModelState *state = (D41ModelState *)calloc(1u, sizeof(D41ModelState));
	const uint32_t rows = options->rows;
	uint64_t arena_bytes;
	uint8_t *arena;
	ReplayRequire(state != 0, "the model state allocates");
	world->state = state;
	memset(&world->rounds, 0, sizeof(world->rounds));
	memset(&world->services, 0, sizeof(world->services));
	world->services.context = &world->rounds;
	world->services.round = ReplayRound;
	state->services = &world->services;
	state->tp_degree = D41_TP;
	state->tp_rank = 0u;
	state->max_rows = REPLAY_MAX_ROWS;
	state->slots = REPLAY_SLOTS;
	state->kv_attached = 1u;
	state->context_slots = REPLAY_SLOTS;
	state->max_positions = REPLAY_MAX_POSITIONS;
	state->moe_whole = 1u;
	D41SharedSlice(state->tp_rank, &state->shared_first, &state->shared_width);
	state->shard.degree = D41_KV_TP;
	state->shard.rank = D41_KV_RANK(state);
	state->shard.grain = D41_GRAIN;
	state->local_score_stride = ((SparkKvShardGatherKeys(state->shard, state->max_positions) + 63u) / 64u) * 64u;
	state->select_rows = state->max_rows;
	state->global_score_stride = ((state->max_positions + 63u) / 64u) * 64u;
	state->candidate_blocks = state->max_positions > D41_CANDIDATE_ROWS ? 1u : 0u;
	ReplayRequire(state->candidate_blocks == 0u, "the replay's context needs no candidate blocks");
	state->topk_entries = 2u * (uint64_t)state->select_rows * LmTopkExactCandidateEntries(state->local_score_stride, D41_INDEX_TOPK, LM_TOPK_EXACT_CHUNK);
	state->ring_bytes = (uint64_t)D41_LAYERS * D41_RING_ROWS * D41_HEAD_DIM * sizeof(uint16_t);
	state->compressor_bytes = 3u * 2u * D41_HEAD_DIM * sizeof(float);
	state->history_bytes = 4u * sizeof(int32_t);
	state->recurrent_bytes = (state->ring_bytes + state->compressor_bytes + state->history_bytes + 255u) & ~255ull;
	arena_bytes = D41Cuda::Plan(state, 0, 0u);
	arena = (uint8_t *)ReplayAllocate(arena_bytes);
	ReplayCuda(cudaMemset(arena, 0, arena_bytes), "clear the arena");
	ReplayRequire(D41Cuda::Plan(state, arena, 0u) == arena_bytes, "the arena plan places every buffer");
	state->arena = arena;
	state->arena_bytes = arena_bytes;
	ReplayFill(state->streams, (uint64_t)REPLAY_MAX_ROWS * D41K_HC * D41_HIDDEN * 2u, REPLAY_FILL_BF16, 0x77u);
	ReplayFill(state->recurrent, (uint64_t)REPLAY_SLOTS * state->recurrent_bytes, REPLAY_FILL_BF16, 0x78u);
	ReplayRequire(D41Cuda::Upload(state) == SPARK_STATUS_OK, "the stage's upload runs");
	state->full_frequency = (float *)ReplayAllocate(2u * D41K_ROPE_PAIRS * sizeof(float));
	state->compress_frequency = state->full_frequency + D41K_ROPE_PAIRS;
	ReplayRequire(D41Frequencies(state->full_frequency) == SPARK_STATUS_OK, "the rotary frequencies build");
	{
		std::vector<uint8_t> groups((uint64_t)D41_LAYERS * D41_EXPERTS);
		for ( uint64_t index = 0u; index < groups.size(); ++index )
			groups[index] = (uint8_t)(index % D41_EXPERTS < D41_LOCAL_EXPERTS ? index % D41_EXPERTS : 0xffu);
		state->expert_group = (uint8_t *)ReplayAllocate(groups.size());
		ReplayCuda(cudaMemcpy(state->expert_group, groups.data(), groups.size(), cudaMemcpyHostToDevice), "copy the expert placement");
	}
	{
		LmKvAccessError *error = (LmKvAccessError *)ReplayAllocate(sizeof(LmKvAccessError));
		ReplayCuda(cudaMemset(error, 0, sizeof(LmKvAccessError)), "clear the KV error");
		state->kv_error = error;
		for ( uint32_t cache = 0u; cache < 4u; ++cache )
		{
			const uint32_t ratio = cache < 3u ? 2u : 1u, page_slots = D41_PAGE_TOKENS / ratio;
			const uint32_t pages = (REPLAY_MAX_POSITIONS / ratio + page_slots - 1u) / page_slots;
			std::vector<uint32_t> table((uint64_t)REPLAY_SLOTS * pages);
			for ( uint64_t index = 0u; index < table.size(); ++index )
				table[index] = (uint32_t)index;
			uint8_t *pool = (uint8_t *)ReplayFilled((uint64_t)REPLAY_SLOTS * pages * page_slots * D41_SLOT_BYTES, REPLAY_FILL_BF16, 0x500u + cache);
			LmKvShardView view;
			memset(&view, 0, sizeof(view));
			ReplayRequire(LmKvViewInitialize(&view.pages, pool, ReplayDeviceWords(table), pages, REPLAY_SLOTS, REPLAY_SLOTS * pages, error) == 0,
				"the compressed KV view builds");
			view.shard = state->shard;
			state->latent_views[cache] = view;
		}
	}
	{
		std::vector<uint32_t> selected((uint64_t)REPLAY_MAX_ROWS * D41_INDEX_TOPK);
		for ( uint64_t index = 0u; index < selected.size(); ++index )
			selected[index] = (uint32_t)(2u * (index % D41_INDEX_TOPK));
		ReplayCuda(cudaMemcpy(state->selected, selected.data(), selected.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), "copy the selection");
	}
	{
		const uint64_t cells13 = (uint64_t)(D41_HIDDEN / LM_SKINNY_CELL_TILE_K) * (2u * D41_MOE_INTER / LM_SKINNY_CELL_NEURONS) * REPLAY_CELL_BYTES;
		const uint64_t cells2 = (uint64_t)(D41_MOE_INTER / LM_SKINNY_CELL_TILE_K) * (D41_HIDDEN / LM_SKINNY_CELL_NEURONS) * REPLAY_CELL_BYTES;
		const uint8_t *pool13 = (const uint8_t *)ReplayFilled(cells13 * D41_LOCAL_EXPERTS, REPLAY_FILL_CELLS, 0x600u);
		const uint8_t *pool2 = (const uint8_t *)ReplayFilled(cells2 * D41_LOCAL_EXPERTS, REPLAY_FILL_CELLS, 0x601u);
		for ( uint32_t slot = 0u; slot < D41_LAYERS; ++slot )
			world->weights[slot] = ReplayLayer(slot, state->shared_width, pool13, pool2);
	}
	world->rounds.staging = (uint4 *)ReplayAllocate((uint64_t)REPLAY_MAX_ROWS * D41_HIDDEN * 2u);
	{
		std::vector<uint32_t> positions(rows), lengths(REPLAY_SLOTS, 1u), of_row(rows), begin(rows + 1u), indices(rows), recurrent(REPLAY_SLOTS);
		for ( uint32_t row = 0u; row < rows; ++row )
		{
			positions[row] = options->position + row;
			lengths[row] = positions[row] + 1u;
			of_row[row] = row;
			begin[row] = row;
			indices[row] = row;
		}
		begin[rows] = rows;
		for ( uint32_t slot = 0u; slot < REPLAY_SLOTS; ++slot )
			recurrent[slot] = slot;
		memset(&world->step, 0, sizeof(world->step));
		world->step.rows = rows;
		world->step.sequences = rows;
		world->step.commit = 1u;
		world->step.multiprocessors = multiprocessors;
		world->step.positions = ReplayDeviceWords(positions);
		world->step.context_length = ReplayDeviceWords(lengths);
		world->step.sequence_of_row = ReplayDeviceWords(of_row);
		world->step.sequence_row_begin = ReplayDeviceWords(begin);
		world->step.sequence_row_indices = ReplayDeviceWords(indices);
		world->step.recurrent_index = ReplayDeviceWords(recurrent);
		world->step.token_ids = ReplayDeviceWords(std::vector<uint32_t>(rows, 7u));
		world->step.route_fault = ReplayDeviceWords(std::vector<uint32_t>(1u, 0u));
		world->step.hidden_bf16 = (uint16_t *)ReplayFilled((uint64_t)REPLAY_MAX_ROWS * D41_HIDDEN * 2u, REPLAY_FILL_BF16, 0x700u);
		LM_LAUNCH((LmRowLanesKernel), rows, 256u, 0, (SparkGpuStream)0, world->step.sequence_row_begin, world->step.sequence_row_indices, state->row_lane,
			state->row_ordinal);
		LM_LAUNCH((D41CompressedEmitPositionKernel), 1u, 256u, 0, (SparkGpuStream)0, world->step.positions, state->bound_ratio2, rows, 2u);
		LM_LAUNCH((D41CompressedEmitPositionKernel), 1u, 256u, 0, (SparkGpuStream)0, world->step.positions, state->bound_ratio1, rows, 1u);
		LM_LAUNCH((D41CompressedLengthKernel), 1u, 256u, 0, (SparkGpuStream)0, world->step.context_length, state->length_ratio2, REPLAY_SLOTS, 2u);
		LM_LAUNCH((D41CompressedLengthKernel), 1u, 256u, 0, (SparkGpuStream)0, world->step.context_length, state->length_ratio1, REPLAY_SLOTS, 1u);
	}
	state->row_invariant = world->step.commit == 0u ? 1u : 0u;
	memset(&world->plan, 0, sizeof(world->plan));
	world->plan.host = (LmWeightPrefetchGroup *)calloc(LM_WEIGHT_PREFETCH_GROUPS, sizeof(LmWeightPrefetchGroup));
	world->plan.device = (LmWeightPrefetchGroup *)ReplayAllocate((uint64_t)LM_WEIGHT_PREFETCH_GROUPS * sizeof(LmWeightPrefetchGroup));
	world->plan.capacity = LM_WEIGHT_PREFETCH_GROUPS;
	ReplayRequire(world->plan.host != 0, "the prefetch plan allocates");
	world->mark_sink = (uint32_t *)ReplayAllocate(sizeof(uint32_t));
	world->initial_streams = (uint16_t *)ReplayAllocate((uint64_t)REPLAY_MAX_ROWS * D41K_HC * D41_HIDDEN * 2u);
	ReplayCuda(cudaMemcpy(world->initial_streams, state->streams, (uint64_t)REPLAY_MAX_ROWS * D41K_HC * D41_HIDDEN * 2u, cudaMemcpyDeviceToDevice),
		"keep the first streams");
	if ( options->holds != 0 )
		ReplayHolds(&world->rounds, options->holds, rows);
	ReplayCuda(cudaDeviceSynchronize(), "setup");
}

static cudaGraphExec_t ReplayCapture(ReplayWorld *world, SparkGpuStream stream, const ReplayFork *fork)
{
	D41ModelState *state = world->state;
	cudaGraph_t graph = 0;
	cudaGraphExec_t exec = 0;
	D41Cuda::Frame frame;
	SparkStatus status = SPARK_STATUS_OK;
	memset(&frame, 0, sizeof(frame));
	frame.step = &world->step;
	frame.stream = stream;
	frame.tail = world->step;
	frame.current = &world->step;
	frame.streams = state->streams;
	frame.pre_head = state->pre_head;
	frame.selected = state->selected;
	frame.block_mask = state->block_mask;
	LmWeightPrefetchArm(&world->plan, stream);
	ReplayCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
	LM_LAUNCH((ReplayCopyKernel), 8u, 256u, 0u, stream, (const uint4 *)world->initial_streams, (uint4 *)state->streams,
		(uint64_t)world->step.rows * D41K_HC * D41_HIDDEN * 2u / 16u);
	for ( uint32_t slot = 0u; slot < D41_LAYERS && status == SPARK_STATUS_OK; ++slot )
	{
		const uint32_t index = ReplayIndex(slot);
		world->rounds.layer = slot;
		state->layers[index] = world->weights[slot];
		state->compress_done = 0u;
		status = ReplayLayerBody(state, frame, index, fork);
	}
	ReplayCuda(cudaStreamEndCapture(stream, &graph), "end capture");
	ReplayCuda(LmWeightPrefetchDisarm(&world->plan, status == SPARK_STATUS_OK ? 1u : 0u), "publish the prefetch groups");
	ReplayRequire(status == SPARK_STATUS_OK && world->rounds.failures == 0u, "every layer records");
	ReplayCuda(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
	ReplayCuda(cudaGraphUpload(exec, stream), "upload");
	ReplayCuda(cudaGraphDestroy(graph), "destroy graph");
	return exec;
}

static void ReplayRun(ReplayWorld *world, cudaGraphExec_t exec, SparkGpuStream stream, const ReplayOptions *options, const char *label)
{
	std::vector<float> spans;
	cudaEvent_t start, stop;
	ReplayCuda(cudaEventCreate(&start), "event");
	ReplayCuda(cudaEventCreate(&stop), "event");
	for ( uint32_t round = 0u; round < options->warmup + options->steps; ++round )
	{
		float milliseconds = 0.0f;
		LM_LAUNCH((ReplayMarkKernel), 1u, 32u, 0u, stream, round, world->mark_sink);
		ReplayCuda(cudaEventRecord(start, stream), "record");
		ReplayCuda(cudaGraphLaunch(exec, stream), "launch");
		ReplayCuda(cudaEventRecord(stop, stream), "record");
		ReplayCuda(cudaEventSynchronize(stop), "step");
		ReplayCuda(cudaEventElapsedTime(&milliseconds, start, stop), "elapsed");
		if ( round >= options->warmup )
			spans.push_back(milliseconds);
	}
	std::vector<float> sorted(spans);
	std::sort(sorted.begin(), sorted.end());
	printf("REPLAY %s rows=%u steps=%zu step_ms_median=%.3f step_ms_min=%.3f\n", label, options->rows, sorted.size(), sorted[sorted.size() / 2u], sorted[0]);
	ReplayCuda(cudaEventDestroy(start), "event");
	ReplayCuda(cudaEventDestroy(stop), "event");
}

static void ReplayCheckFinite(ReplayWorld *world)
{
	std::vector<uint16_t> streams((uint64_t)world->step.rows * D41K_HC * D41_HIDDEN);
	std::vector<uint32_t> fault(1u);
	uint64_t bad = 0u, digest = 0xcbf29ce484222325ull;
	ReplayCuda(cudaMemcpy(streams.data(), world->state->streams, streams.size() * 2u, cudaMemcpyDeviceToHost), "read the streams");
	ReplayCuda(cudaMemcpy(fault.data(), world->step.route_fault, sizeof(uint32_t), cudaMemcpyDeviceToHost), "read the route fault");
	for ( uint64_t index = 0u; index < streams.size(); ++index )
	{
		bad += (streams[index] & 0x7f80u) == 0x7f80u ? 1u : 0u;
		digest = (digest ^ streams[index]) * 0x100000001b3ull;
	}
	printf("REPLAY finite streams_nonfinite=%llu route_fault=%u streams_digest=%016llx\n", (unsigned long long)bad, fault[0], (unsigned long long)digest);
}

static void ReplayBoundaries(const ReplayOptions *options, SparkGpuStream stream, uint32_t multiprocessors)
{
	static const char *names[6] = { "copy-then-copy", "qdq-then-gemv", "copy-then-gemv", "release-then-gemv", "release-then-stream-wait-first",
		"release-then-stream-load-first" };
	const uint64_t weight_bytes = (uint64_t)D41_HIDDEN * D41_WO_A_ROWS;
	const uint32_t tiny_blocks = 128u, tiny_vectors = tiny_blocks * 256u;
	std::vector<D41Fp8Linear> copies(REPLAY_COPIES);
	uint4 *tiny_source = (uint4 *)ReplayFilled((uint64_t)tiny_vectors * 16u, REPLAY_FILL_BF16, 0x800u);
	uint4 *tiny_target = (uint4 *)ReplayAllocate((uint64_t)tiny_vectors * 16u);
	uint16_t *activation = (uint16_t *)ReplayFilled((uint64_t)REPLAY_MAX_ROWS * D41_WO_A_ROWS * 2u, REPLAY_FILL_BF16, 0x801u);
	uint16_t *output = (uint16_t *)ReplayAllocate((uint64_t)REPLAY_MAX_ROWS * D41_HIDDEN * 2u);
	uint4 *sink = (uint4 *)ReplayAllocate(16u);
	for ( uint32_t copy = 0u; copy < REPLAY_COPIES; ++copy )
		copies[copy] = ReplayLinear(D41_HIDDEN, D41_WO_A_ROWS, 0x900u + 4u * copy);
	for ( uint32_t chain = 0u; chain < 6u; ++chain )
	{
		cudaGraph_t graph = 0;
		cudaGraphExec_t exec = 0;
		ReplayCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
		for ( uint32_t link = 0u; link < REPLAY_CHAIN; ++link )
		{
			const D41Fp8Linear *weight = &copies[link % REPLAY_COPIES];
			if ( chain == 1u )
				D41Fp8Qdq(activation, (uint64_t)options->rows * D41_WO_A_ROWS, stream);
			else if ( chain <= 2u )
				LM_LAUNCH((ReplayTinyKernel), tiny_blocks, 256u, 0u, stream, (const uint4 *)tiny_source, tiny_target, tiny_vectors);
			else
				LM_LAUNCH((ReplayTinyReleaseKernel), tiny_blocks, 256u, 0u, stream, (const uint4 *)tiny_source, tiny_target, tiny_vectors);
			if ( chain == 0u )
				LM_LAUNCH((ReplayTinyKernel), tiny_blocks, 256u, 0u, stream, (const uint4 *)tiny_source, tiny_target, tiny_vectors);
			else if ( chain <= 3u )
				ReplayRequire(D41Linear(weight, activation, output, options->rows, D41_WO_A_ROWS, D41_HIDDEN, multiprocessors, 0u, (const D41Mx *)0, stream) ==
					LM_LAUNCH_OK, "the GEMV launches");
			else if ( chain == 4u )
				LM_LAUNCH_DEPENDENT((ReplayStreamKernel<false>), 2u * multiprocessors, 256u, 0u, stream, (const uint4 *)weight->weight, weight_bytes / 16u,
					(const uint4 *)activation, sink);
			else
				LM_LAUNCH_DEPENDENT((ReplayStreamKernel<true>), 2u * multiprocessors, 256u, 0u, stream, (const uint4 *)weight->weight, weight_bytes / 16u,
					(const uint4 *)activation, sink);
		}
		ReplayCuda(cudaStreamEndCapture(stream, &graph), "end capture");
		ReplayCuda(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
		for ( uint32_t round = 0u; round < options->warmup + options->steps; ++round )
		{
			LM_LAUNCH((ReplayMarkKernel), 1u, 32u, 0u, stream, 1000u * (chain + 1u) + round, (uint32_t *)sink);
			ReplayCuda(cudaGraphLaunch(exec, stream), "launch");
		}
		ReplayCuda(cudaStreamSynchronize(stream), "chain");
		printf("REPLAY boundary chain=%u name=%s links=%u rounds=%u\n", chain, names[chain], REPLAY_CHAIN, options->warmup + options->steps);
		ReplayCuda(cudaGraphExecDestroy(exec), "destroy exec");
		ReplayCuda(cudaGraphDestroy(graph), "destroy graph");
	}
}

static void ReplayUsage(void)
{
	fprintf(stderr, "usage: dsv41_flash_decode_replay --mode replay|prefetch|boundary [--rows R] [--steps N] [--warmup W] [--position P] "
		"[--holds FILE] [--prefetch-mib M] [--prefetch-threads T]\n");
	exit(2);
}

int main(int argc, char **argv)
{
	ReplayOptions options;
	SparkGpuStream stream = 0;
	int device = 0, multiprocessors = 0;
	memset(&options, 0, sizeof(options));
	options.mode = "replay";
	options.rows = 8u;
	options.steps = 50u;
	options.warmup = 5u;
	options.position = 160u;
	options.prefetch_bytes = 16ull << 20;
	options.prefetch_threads = REPLAY_PREFETCH_THREADS;
	for ( int index = 1; index < argc; ++index )
	{
		const char *flag = argv[index], *value = index + 1 < argc ? argv[index + 1] : 0;
		if ( value == 0 )
			ReplayUsage();
		if ( strcmp(flag, "--mode") == 0 )
			options.mode = value;
		else if ( strcmp(flag, "--rows") == 0 )
			options.rows = (uint32_t)strtoul(value, 0, 10);
		else if ( strcmp(flag, "--steps") == 0 )
			options.steps = (uint32_t)strtoul(value, 0, 10);
		else if ( strcmp(flag, "--warmup") == 0 )
			options.warmup = (uint32_t)strtoul(value, 0, 10);
		else if ( strcmp(flag, "--position") == 0 )
			options.position = (uint32_t)strtoul(value, 0, 10);
		else if ( strcmp(flag, "--holds") == 0 )
			options.holds = value;
		else if ( strcmp(flag, "--prefetch-mib") == 0 )
			options.prefetch_bytes = (uint64_t)strtoull(value, 0, 10) << 20;
		else if ( strcmp(flag, "--prefetch-threads") == 0 )
			options.prefetch_threads = (uint32_t)strtoul(value, 0, 10);
		else
			ReplayUsage();
		++index;
	}
	ReplayRequire(options.rows >= 1u && options.rows <= REPLAY_MAX_ROWS && options.steps >= 1u, "rows 1 to 16 and at least one step");
	ReplayRequire(options.position + options.rows < REPLAY_MAX_POSITIONS && options.position >= D41_WINDOW, "the position leaves a full window and room");
	ReplayCuda(cudaSetDevice(0), "device");
	ReplayCuda(cudaGetDevice(&device), "device");
	ReplayCuda(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device), "multiprocessors");
	ReplayCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
	if ( strcmp(options.mode, "boundary") == 0 )
	{
		ReplayBoundaries(&options, stream, (uint32_t)multiprocessors);
		return 0;
	}
	ReplayRequire(strcmp(options.mode, "replay") == 0 || strcmp(options.mode, "prefetch") == 0, "a known mode");
	ReplayRequire(options.holds != 0, "a holds file");
	static ReplayWorld world;
	ReplaySetup(&world, &options, (uint32_t)multiprocessors);
	if ( strcmp(options.mode, "replay") == 0 )
	{
		cudaGraphExec_t exec = ReplayCapture(&world, stream, (const ReplayFork *)0);
		ReplayRun(&world, exec, stream, &options, "production");
		ReplayCheckFinite(&world);
		return 0;
	}
	ReplayFork fork;
	memset(&fork, 0, sizeof(fork));
	ReplayCuda(cudaStreamCreateWithFlags(&fork.side, cudaStreamNonBlocking), "side stream");
	ReplayCuda(cudaEventCreateWithFlags(&fork.before, cudaEventDisableTiming), "event");
	ReplayCuda(cudaEventCreateWithFlags(&fork.after, cudaEventDisableTiming), "event");
	fork.budget = options.prefetch_bytes;
	fork.threads = options.prefetch_threads;
	ReplayRequire(fork.threads >= 1u && fork.threads <= 1024u, "1 to 1024 prefetch threads");
	fork.enabled = 0u;
	cudaGraphExec_t plain = ReplayCapture(&world, stream, &fork);
	fork.enabled = 1u;
	cudaGraphExec_t fetched = ReplayCapture(&world, stream, &fork);
	ReplayRun(&world, plain, stream, &options, "attention-copy");
	ReplayRun(&world, fetched, stream, &options, "attention-prefetch");
	ReplayCheckFinite(&world);
	return 0;
}
