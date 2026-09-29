
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>

#include "sparkpipe/spark_head_screen.h"
#include "sparkpipe/spark_k3_resident_decode_stage_cuda.h"
#include "sparkpipe/spark_k3_resident_decode_stage_module.h"
#include "sparkpipe/spark_k3_resident_decode_stage_runner.h"
#include "sparkpipe/spark_k3_weightd_include.h"
#include "sparkpipe/spark_error_site.h"
#include "inference/llms/kimi_k3/layer.cuh"

typedef struct SparkK3RunnerState SparkK3RunnerState;

__global__ static void K3RunnerDenseOffsetsKernel(uint32_t *offsets, uint32_t rows)
{
	if ( threadIdx.x == 0u )
	{
		offsets[0] = 0u;
		offsets[1] = rows;
	}
}

typedef struct SparkK3RunnerTpContext SparkK3RunnerTpContext;

#define K3_RUNNER_TP_CONTEXT_POOL_DEPTH (2u * K3_LAYERS)

static_assert(K3_RUNNER_TP_CONTEXT_POOL_DEPTH >= 2u * K3_LAYERS,
	"tp context pool must cover both per-layer collectives");

/* Width contract (cold16): the shared device collective sizes EVERY
 * ALL_REDUCE as active_sequence_count x the COLLECTIVE's
 * local_hidden_dimension - never a per-submission element count - and
 * submission.reserved0 must stay zero. k3 therefore rides TWO width-matched
 * collectives: the hidden collective (band 0, width K3_HIDDEN) carries the
 * in-place embedding reduce, the per-layer attention/hidden segment, the
 * shared segment (a second stream-ordered op) and the width-independent
 * ALL_REDUCE_MAX_U64 head argmax; the wide collective (band 1, width
 * K3_TOP_K x 2 x K3_EXPERT_INTERMEDIATE) carries the fused gate_up reduce
 * whose per-row extent is not an integer multiple of K3_HIDDEN. */
#define K3_RUNNER_GATE_UP_WIDTH (K3_TOP_K * (K3_EXPERT_INTERMEDIATE * 2u))

typedef struct SparkK3RunnerTpContext
{
	SparkK3RunnerTpContext *pool_next;
	SparkK3RunnerState *owner;
	K3LayerBuffers *buffers;
	uint16_t *fused;
	cudaStream_t stream;
	uint32_t rows;
	uint32_t boundary;
	uint32_t segments;
	uint32_t phase;
	uint32_t gate_up_elements;
} SparkK3RunnerTpContext;

static SparkK3RunnerTpContext *K3RunnerTpContextAcquire(
	SparkK3RunnerState *state);
static void K3RunnerTpContextRelease(SparkK3RunnerTpContext *context);
static void K3RunnerTpCompletion(void *context,
	const SparkTpDeviceCollectiveCompletion *completion);

__global__ static void K3RunnerFusedPackKernel(const uint16_t *attention,
	const uint16_t *hidden,const uint16_t *shared,const uint16_t *gate_up,
	uint16_t *fused,uint32_t rows,uint32_t phase,uint32_t segments,
	uint32_t gate_up_elements)
{
	uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint32_t elements = rows * K3_HIDDEN;
	if ( phase == 2u )
	{
		if ( i >= gate_up_elements )
			return;
		fused[i] = gate_up[i];
		return;
	}
	if ( phase == 3u )
	{
		if ( i >= gate_up_elements )
			return;
		fused[i] = shared[i];
		return;
	}
	if ( i >= elements )
		return;
	if ( phase == 0u )
		fused[i] = attention[i];
	else
	{
		fused[i] = hidden[i];
		if ( segments == 2u )
			fused[elements + i] = shared[i];
	}
}

#define K3_RUNNER_GATHER_SOURCES_MAX 16u

typedef struct K3RunnerRankSources
{
	const uint16_t *rank[K3_RUNNER_GATHER_SOURCES_MAX];
} K3RunnerRankSources;

__global__ static void K3RunnerCombineTp4TreeKernel(K3RunnerRankSources sources,
	uint16_t *destination,uint32_t rows,uint32_t hidden_dimension)
{
	uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint32_t elements = rows * hidden_dimension;
	if ( i >= elements )
		return;
	float a = LmBf16ToFloat(sources.rank[0][i]) + LmBf16ToFloat(sources.rank[1][i]);
	float b = LmBf16ToFloat(sources.rank[2][i]) + LmBf16ToFloat(sources.rank[3][i]);
	destination[i] = LmFloatToBf16(a + b);
}

__global__ static void K3RunnerCombinePairKernel(const uint16_t *source,
	uint16_t *destination,uint32_t elements)
{
	uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if ( i >= elements )
		return;
	destination[i] = LmFloatToBf16(LmBf16ToFloat(destination[i]) + LmBf16ToFloat(source[i]));
}

__global__ static void K3RunnerGatherStripesKernel(K3RunnerRankSources sources,
	uint16_t *destination,uint32_t rank_count,uint32_t elements_per_rank)
{
	uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint32_t total = rank_count * elements_per_rank;
	if ( i >= total )
		return;
	destination[i] = sources.rank[i / elements_per_rank][i % elements_per_rank];
}

static SparkStatus K3RunnerCombineGatherBf16(void *combine_context,
	void *destination_device,const void *const *source_devices,
	uint32_t source_count,uint32_t active_sequence_count,
	uint32_t hidden_dimension,void *cuda_stream)
{
	K3RunnerRankSources sources;
	uint32_t elements_per_rank = active_sequence_count * hidden_dimension;
	(void)combine_context;
	if ( destination_device == 0 || source_devices == 0 || source_count == 0u ||
		source_count > K3_RUNNER_GATHER_SOURCES_MAX || elements_per_rank == 0u )
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(&sources, 0, sizeof(sources));
	for ( uint32_t r = 0u; r < source_count; ++r )
	{
		if ( source_devices[r] == 0 )
			return SPARK_STATUS_INVALID_ARGUMENT;
		sources.rank[r] = (const uint16_t *)source_devices[r];
	}
	K3RunnerGatherStripesKernel<<<(source_count * elements_per_rank + 255u) / 256u,
		256u, 0, (cudaStream_t)cuda_stream>>>(
		sources,(uint16_t *)destination_device,source_count,elements_per_rank);
	return cudaGetLastError() == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus K3RunnerCombineBf16(void *combine_context,
	void *destination_device,const void *source_device,
	uint32_t active_sequence_count,uint32_t hidden_dimension,void *cuda_stream)
{
	uint32_t elements = active_sequence_count * hidden_dimension;
	(void)combine_context;
	if ( destination_device == 0 || source_device == 0 || elements == 0u )
		return SPARK_STATUS_INVALID_ARGUMENT;
	K3RunnerCombinePairKernel<<<(elements + 255u) / 256u,
		256u, 0, (cudaStream_t)cuda_stream>>>(
		(const uint16_t *)source_device,(uint16_t *)destination_device,elements);
	return cudaGetLastError() == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus K3RunnerCombineTp4Bf16(void *combine_context,
	void *destination_device,const void *const rank_devices[4],uint32_t tp_rank,
	uint32_t active_sequence_count,uint32_t hidden_dimension,void *cuda_stream)
{
	K3RunnerRankSources sources;
	uint32_t elements = active_sequence_count * hidden_dimension;
	(void)combine_context;
	if ( destination_device == 0 || rank_devices == 0 || tp_rank >= 4u || elements == 0u )
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(&sources, 0, sizeof(sources));
	for ( uint32_t r = 0u; r < 4u; ++r )
	{
		if ( rank_devices[r] == 0 )
			return SPARK_STATUS_INVALID_ARGUMENT;
		sources.rank[r] = (const uint16_t *)rank_devices[r];
	}
	K3RunnerCombineTp4TreeKernel<<<(elements + 255u) / 256u,
		256u, 0, (cudaStream_t)cuda_stream>>>(
		sources,(uint16_t *)destination_device,active_sequence_count,hidden_dimension);
	return cudaGetLastError() == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

__global__ static void K3RunnerCombineU64MaxKernel(const uint64_t *source,
	uint64_t *destination,uint32_t elements)
{
	uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if ( i >= elements )
		return;
	if ( source[i] > destination[i] )
		destination[i] = source[i];
}

static SparkStatus K3RunnerCombineU64Max(void *combine_context,
	uint64_t *destination_device,const uint64_t *source_device,
	uint32_t element_count,void *cuda_stream)
{
	(void)combine_context;
	if ( destination_device == 0 || source_device == 0 || element_count == 0u )
		return SPARK_STATUS_INVALID_ARGUMENT;
	K3RunnerCombineU64MaxKernel<<<(element_count + 255u) / 256u,
		256u, 0, (cudaStream_t)cuda_stream>>>(
		source_device,destination_device,element_count);
	return cudaGetLastError() == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

enum
{
	SPARK_K3_LEASE_ACQUIRED = 1u,
	SPARK_K3_LEASE_BEGUN = 2u,
	SPARK_K3_LEASE_RECORDED = 3u
};

typedef struct SparkK3RunnerState
{
	SparkK3ModuleState module;
	SparkK3Dispatch dispatch;
	SparkTpCollective collective;
	int collective_created;
	SparkTpDeviceCollective device_collective;
	int device_collective_created;
	SparkTpDeviceCollective device_collective_wide;
	int device_collective_wide_created;
	SparkWeightdLazyPack *lazy_pack;
	uint64_t lease_identifier;
	uint32_t lease_phase;
	void *lease_address;
	uint32_t *group_offset_host;
	uint64_t layer_w1_offset[K3_LAYERS];
	uint64_t layer_w2_offset[K3_LAYERS];
	uint32_t lease_tensor_base;
	uint32_t tp_rank;
	uint16_t *fused_device;
	uint32_t fused_rows;
	uint64_t tp_next_ordinal;
	uint64_t tp_next_ordinal_wide;
	SparkK3RunnerTpContext *tp_context_free_head;
	SparkK3RunnerTpContext tp_context_pool[K3_RUNNER_TP_CONTEXT_POOL_DEPTH];
	uint32_t tp_context_overflow;
	uint32_t tp_collective_failed;
	uint32_t copy_failed;
	uint32_t rows;
	uint32_t logical_sequence_count;
	const uint16_t *embed_weight;
	const uint16_t *head_norm_weight;
	const uint16_t *head_weight;
	uint8_t *head_certified_fp8_payload;
	float *head_certified_fp8_scale_f32;
	float *head_certified_fp8_norm_f32;
	void *head_certified_scratch;
	uint32_t *head_certified_candidates;
	uint32_t *head_screened_count;
	uint32_t vocab;
	uint32_t vocab_slice_rows;
	uint16_t *staging_values;
	uint16_t *staging_scratch;
	uint32_t staging_capacity;
	uint32_t fused_capacity;
	float *head_slots_host;
	float *head_slots_device;
	uint64_t *head_maxloc;
	uint32_t head_slots_capacity;
	uint32_t *route_expert;
	uint32_t *route_packed_row;
	uint32_t *route_source_token;
	float *route_weight;
	uint32_t *group_row_offset;
	uint32_t *group_tile_prefix_w1;
	uint32_t *group_tile_prefix_w2;
	uint32_t *dense_row_offset;
	uint32_t *dense_tile_prefix;
	uint32_t *head_candidate_token;
	float *head_candidate_score;
	uint32_t head_tiles;
	uint32_t *output_token;
	float *output_score;
	uint32_t *output_token_host;
	float *output_score_host;
	uint32_t *positions;
	uint32_t *token_ids_device;
	uint32_t *context_length;
	uint32_t *sequence_of_row;
	uint32_t *kda_state_index;
	/* M3 lazy-stray accounting (opt-in: SPARK_K3_STRAY_WSET names the
	 * preload head .wset; every routed (layer, expert) selection the
	 * runner demands is checked against it and the receipt prints at
	 * destroy — the measured stray rate to compare with the manifest's
	 * pinned head_selection_coverage). */
	uint8_t *stray_head_bits;
	uint8_t *stray_seen_bits;
	uint64_t stray_selections;
	uint64_t stray_count;
	uint32_t stray_head_keys;
	cudaStream_t stream;
	uint32_t max_rows;
	uint32_t max_context;
	uint32_t multiprocessors;
	uint64_t kv_page_bytes;
} SparkK3RunnerState;

/* Bit index over the full-model (layer, expert) key space; the head .wset
 * holds full-model pairs (the census basis), of which only this rank's
 * PP stage can ever route. */
#define K3_STRAY_BIT_INDEX(layer, expert) \
	((uint64_t)(layer) * K3_EXPERTS + (uint64_t)(expert))
#define K3_STRAY_BIT_BYTES \
	((K3_STRAY_BIT_INDEX(K3_LAYERS, 0u) + 7u) / 8u)

static void SparkK3RunnerStrayAccount(
	SparkK3RunnerState *state,
	const SparkWeightdExpertKey *keys, uint32_t count)
{
	uint32_t index;
	if ( state == 0 || state->stray_head_bits == 0 || keys == 0 )
		return;
	for ( index = 0u; index < count; ++index )
	{
		uint64_t bit;
		uint8_t mask;
		if ( keys[index].layer >= K3_LAYERS ||
			keys[index].expert >= K3_EXPERTS )
			continue;
		bit = K3_STRAY_BIT_INDEX(keys[index].layer, keys[index].expert);
		mask = (uint8_t)(1u << (bit & 7u));
		state->stray_selections++;
		if ( (state->stray_head_bits[bit >> 3] & mask) == 0u )
		{
			state->stray_count++;
			state->stray_seen_bits[bit >> 3] |= mask;
		}
	}
}

static void SparkK3RunnerStrayLoad(SparkK3RunnerState *state)
{
	const char *path = getenv("SPARK_K3_STRAY_WSET");
	FILE *input;
	uint32_t pair[2];
	uint32_t loaded = 0u;
	if ( path == 0 || path[0] == '\0' )
		return;
	state->stray_head_bits = (uint8_t *)calloc(K3_STRAY_BIT_BYTES, 1u);
	state->stray_seen_bits = (uint8_t *)calloc(K3_STRAY_BIT_BYTES, 1u);
	if ( state->stray_head_bits == 0 || state->stray_seen_bits == 0 )
	{
		free(state->stray_head_bits);
		free(state->stray_seen_bits);
		state->stray_head_bits = 0;
		state->stray_seen_bits = 0;
		fprintf(stderr, "sparkpipe_k3: stray accounting disabled "
			"(allocation failed)\n");
		return;
	}
	input = fopen(path, "rb");
	if ( input == 0 )
	{
		fprintf(stderr, "sparkpipe_k3: stray accounting disabled "
			"(cannot open SPARK_K3_STRAY_WSET=%s)\n", path);
		free(state->stray_head_bits);
		free(state->stray_seen_bits);
		state->stray_head_bits = 0;
		state->stray_seen_bits = 0;
		return;
	}
	while ( fread(pair, 1u, sizeof(pair), input) == sizeof(pair) )
	{
		uint64_t bit;
		if ( pair[0] >= K3_LAYERS || pair[1] >= K3_EXPERTS )
			continue;
		bit = K3_STRAY_BIT_INDEX(pair[0], pair[1]);
		state->stray_head_bits[bit >> 3] |= (uint8_t)(1u << (bit & 7u));
		loaded++;
	}
	(void)fclose(input);
	state->stray_head_keys = loaded;
	fprintf(stderr, "sparkpipe_k3: stray accounting armed head_keys=%u "
		"wset=%s\n", loaded, path);
}

static void SparkK3RunnerStrayReport(const SparkK3RunnerState *state)
{
	uint64_t unique = 0u;
	uint32_t index;
	if ( state == 0 || state->stray_head_bits == 0 )
		return;
	if ( state->stray_seen_bits != 0 )
		for ( index = 0u; index < K3_STRAY_BIT_BYTES; ++index )
		{
			uint8_t word = state->stray_seen_bits[index];
			while ( word != 0u )
			{
				unique += word & 1u;
				word >>= 1;
			}
		}
	fprintf(stderr, "K3-STRAY-RECEIPT selections=%llu strays=%llu "
		"stray_rate=%.4f unique_stray_pairs=%llu head_keys=%u\n",
		(unsigned long long)state->stray_selections,
		(unsigned long long)state->stray_count,
		state->stray_selections != 0u ?
			(double)state->stray_count / (double)state->stray_selections : 0.0,
		(unsigned long long)unique, state->stray_head_keys);
}

static SparkK3RunnerTpContext *K3RunnerTpContextAcquire(
	SparkK3RunnerState *state)
{
	SparkK3RunnerTpContext *context = state->tp_context_free_head;
	if ( context == 0 )
	{
		state->tp_context_overflow = 1u;
		return 0;
	}
	state->tp_context_free_head = context->pool_next;
	return context;
}

static void K3RunnerTpContextRelease(SparkK3RunnerTpContext *context)
{
	SparkK3RunnerState *state = context->owner;
	context->pool_next = state->tp_context_free_head;
	state->tp_context_free_head = context;
}

static void K3RunnerTpCompletion(void *context,
	const SparkTpDeviceCollectiveCompletion *completion)
{
	SparkK3RunnerTpContext *tp = (SparkK3RunnerTpContext *)context;
	K3LayerBuffers *b = tp->buffers;
	uint32_t rows = tp->rows;
	uint32_t elements = rows * K3_HIDDEN;
	uint16_t *fused = tp->fused;
	if ( completion != 0 && completion->status != SPARK_STATUS_OK )
		tp->owner->tp_collective_failed = 1u;
	if ( tp->phase == 2u )
	{
		if ( cudaMemcpyAsync(b->gate_up_bf16, fused,
			(uint64_t)tp->gate_up_elements * sizeof(*b->gate_up_bf16),
			cudaMemcpyDeviceToDevice, tp->stream) != cudaSuccess )
			tp->owner->copy_failed = 1u;
		K3RunnerTpContextRelease(tp);
		return;
	}
	if ( tp->phase == 3u )
	{
		if ( cudaMemcpyAsync(b->shared_out_bf16, fused,
			(uint64_t)tp->gate_up_elements * sizeof(*b->shared_out_bf16),
			cudaMemcpyDeviceToDevice, tp->stream) != cudaSuccess )
			tp->owner->copy_failed = 1u;
		K3RunnerTpContextRelease(tp);
		return;
	}
	if ( tp->phase == 0u )
	{
		if ( tp->boundary != 0u )
			K3PartialSet(b, fused, rows, tp->stream);
		else
			K3PartialAdd(b, fused, rows, tp->stream);
	}
	else
	{
		K3PartialAdd(b, fused, rows, tp->stream);
		if ( tp->segments == 2u )
			K3PartialAdd(b, fused + elements, rows, tp->stream);
	}
	K3RunnerTpContextRelease(tp);
}

static int32_t K3RunnerLaunchSliceDirect(SparkK3RunnerState *state,
	SparkK3StepInput *in, uint32_t rows, uint32_t sequences,
	uint32_t packed_rows, cudaStream_t stream)
{
	K3RunnerDenseOffsetsKernel<<<1u, 1u, 0, stream>>>(state->dense_row_offset, rows);
	return SparkK3DispatchStep(&state->dispatch, in, rows, sequences, 1u,
		packed_rows, state->max_context, state->multiprocessors, stream);
}

#define K3_RUNNER_PP_STAGE_COUNT 4u
#define K3_RUNNER_PP_STAGE_BASE_LAYERS (K3_LAYERS / K3_RUNNER_PP_STAGE_COUNT)
#define K3_RUNNER_PP_STAGE_REMAINDER (K3_LAYERS % K3_RUNNER_PP_STAGE_COUNT)
#define K3_RUNNER_PP_STAGE_LAYERS(stage_index) \
	(K3_RUNNER_PP_STAGE_BASE_LAYERS + \
	((stage_index) < K3_RUNNER_PP_STAGE_REMAINDER ? 1u : 0u))
#define K3_RUNNER_PP_STAGE_FIRST(stage_index) \
	((stage_index) * K3_RUNNER_PP_STAGE_BASE_LAYERS + \
	((stage_index) < K3_RUNNER_PP_STAGE_REMAINDER ? \
	(stage_index) : K3_RUNNER_PP_STAGE_REMAINDER))

static_assert(K3_RUNNER_PP_STAGE_FIRST(K3_RUNNER_PP_STAGE_COUNT - 1u) +
	K3_RUNNER_PP_STAGE_LAYERS(K3_RUNNER_PP_STAGE_COUNT - 1u) ==
	K3_LAYERS,
	"pp stage bounds must tile the k3 layer stack");

static_assert(K3_LAYERS == SPARK_K3_MODULE_TOTAL_LAYERS,
	"k3 kernel layer count must equal the module layer total");
static uint32_t K3RunnerFirstLayer(uint32_t stage_index)
{
	return(SPARK_K3_PP_STAGE_FIRST(stage_index % SPARK_K3_PP_STAGE_COUNT));
}

static uint32_t K3RunnerLayerCount(uint32_t stage_index)
{
	return(SPARK_K3_PP_STAGE_LAYERS(stage_index % SPARK_K3_PP_STAGE_COUNT));
}

static void K3RunnerEmbedCompletion(void *context,
	const SparkTpDeviceCollectiveCompletion *completion)
{
	(void)context;
	(void)completion;
}

static cudaError_t K3RunnerCopy(void *destination, const void *source,
	uint64_t bytes, cudaStream_t stream)
{
	cudaError_t error = cudaMemcpyAsync(destination, source, (size_t)bytes,
		cudaMemcpyDefault, stream);
	if ( error != cudaSuccess )
		return error;
	return cudaStreamSynchronize(stream);
}

static void K3RunnerHostAllReduce(SparkK3RunnerState *state, uint64_t elements)
{
	if ( SparkTpCollectiveAllReduceSumBf16(&state->collective,
		state->staging_values, elements, state->staging_scratch) != SPARK_STATUS_OK )
		state->tp_collective_failed = 1u;
}

static SparkStatus K3RunnerTakeFailure(SparkK3RunnerState *state)
{
	SparkStatus status = SPARK_STATUS_OK;
	if ( state->copy_failed != 0u )
	{
		fprintf(stderr, "sparkpipe_k3: a stream-ordered copy failed\n");
		status = SPARK_STATUS_IO_ERROR;
	}
	else if ( state->tp_collective_failed != 0u )
	{
		fprintf(stderr, "sparkpipe_k3: a tensor-parallel collective failed\n");
		status = SPARK_STATUS_INTERNAL_ERROR;
	}
	state->copy_failed = 0u;
	state->tp_collective_failed = 0u;
	return status;
}

static SparkStatus K3RunnerReduceBf16(SparkK3RunnerState *state, cudaStream_t stream,
	const uint16_t *device_values, uint32_t rows)
{
	SparkStatus status;
	uint32_t elements = rows * K3_HIDDEN;
	if ( state->device_collective_created != 0 )
	{
		/* Hidden collective (band 0, width K3_HIDDEN): rows x K3_HIDDEN
		 * elements in place; reserved0 must stay zero (the transport sizes
		 * the reduce from the collective width, not the submission). */
		SparkTpDeviceCollectiveSubmission submission;
		memset(&submission, 0, sizeof(submission));
		submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
		submission.descriptor_bytes = sizeof(submission);
		submission.slot_index = 0u;
		submission.active_sequence_count = rows;
		submission.logical_sequence_count = state->logical_sequence_count;
		submission.flags =
			SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
		submission.ordinal = state->tp_next_ordinal++;
		submission.local_device = device_values;
		submission.full_device = (void *)device_values;
		submission.cuda_stream = stream;
		submission.completion_function = K3RunnerEmbedCompletion;
		submission.completion_context = 0;
		return SparkTpDeviceCollectiveEnqueue(&state->device_collective,
			&submission,
			SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16);
	}
	if ( state->collective_created == 0 )
		return SPARK_STATUS_OK;
	cudaError_t error = cudaStreamSynchronize(stream);
	if ( error != cudaSuccess )
		return SPARK_STATUS_INTERNAL_ERROR;
	error = K3RunnerCopy(state->staging_values, device_values,
		(uint64_t)elements * sizeof(uint16_t), stream);
	if ( error != cudaSuccess )
		return SPARK_STATUS_INTERNAL_ERROR;
	status = SparkTpCollectiveAllReduceSumBf16(&state->collective,
		state->staging_values, elements, state->staging_scratch);
	if ( status != SPARK_STATUS_OK )
		return status;
	error = K3RunnerCopy((void *)device_values, state->staging_values,
		(uint64_t)elements * sizeof(uint16_t), stream);
	if ( error != cudaSuccess )
		return SPARK_STATUS_INTERNAL_ERROR;
	return SPARK_STATUS_OK;
}

static void K3RunnerLayerCollective(void *context, void *stream_void,
	uint32_t layer, uint32_t phase)
{
	SparkK3RunnerState *state = (SparkK3RunnerState *)context;
	K3LayerBuffers *b = state->dispatch.buffers;
	cudaStream_t stream = (cudaStream_t)stream_void;
	uint32_t rows = state->rows;
	uint32_t boundary = (layer % K3_ATTNRES_BLOCK_SIZE) == 0u;
	uint32_t elements = rows * K3_HIDDEN;
	uint32_t segments = phase == 0u ? 1u : (layer == 0u ? 1u : 2u);
	uint16_t *phase0_source =
		(K3_LAYER_KIND(layer) == LM_LAYER_RECURRENT)
			? b->hidden_bf16 : b->attention_out_bf16;
	if ( b->tp_sharded == 0u )
		return;
	if ( phase == 2u || phase == 3u )
	{
		const uint32_t gate_up_elements = phase == 2u
			? rows * K3_TOP_K * (K3_EXPERT_INTERMEDIATE * 2u)
			: rows * K3_ROUTED_EXPERT_HIDDEN;
		uint16_t *reduce_values = phase == 2u ? b->gate_up_bf16 : b->shared_out_bf16;
		if ( state->device_collective_wide_created != 0 )
		{
			/* Wide collective (band 1, width K3_RUNNER_GATE_UP_WIDTH):
			 * rows x 98304 elements exactly, ordinal chain of its own. */
			K3RunnerFusedPackKernel<<<(gate_up_elements + 255u) / 256u,
				256u, 0, stream>>>(
				0, 0, reduce_values, reduce_values, state->fused_device,
				rows, phase, 1u, gate_up_elements);
			SparkK3RunnerTpContext *completion_context =
				K3RunnerTpContextAcquire(state);
			if ( completion_context == 0 )
				return;
			completion_context->owner = state;
			completion_context->fused = state->fused_device;
			completion_context->buffers = b;
			completion_context->stream = stream;
			completion_context->rows = rows;
			completion_context->boundary = 0u;
			completion_context->segments = 1u;
			completion_context->phase = phase;
			completion_context->gate_up_elements = gate_up_elements;
			SparkTpDeviceCollectiveSubmission submission;
			memset(&submission, 0, sizeof(submission));
			submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
			submission.descriptor_bytes = sizeof(submission);
			submission.slot_index = 0u;
			submission.active_sequence_count = rows;
			submission.logical_sequence_count = state->logical_sequence_count;
			submission.flags =
				SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
			submission.ordinal = state->tp_next_ordinal_wide++;
			submission.local_device = state->fused_device;
			submission.full_device = state->fused_device;
			submission.cuda_stream = stream;
			submission.completion_function = K3RunnerEmbedCompletion;
			submission.completion_context = 0;
			if ( SparkTpDeviceCollectiveEnqueue(&state->device_collective_wide,
				&submission,
				SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16) != SPARK_STATUS_OK )
				state->tp_collective_failed = 1u;
			K3RunnerTpCompletion(completion_context, 0);
			return;
		}
		if ( state->collective_created != 0 )
		{
			cudaStreamSynchronize(stream);
			if ( K3RunnerCopy(state->staging_values, reduce_values,
				(uint64_t)gate_up_elements * sizeof(uint16_t), stream) != cudaSuccess )
				state->copy_failed = 1u;
			K3RunnerHostAllReduce(state, gate_up_elements);
			if ( K3RunnerCopy(reduce_values, state->staging_values,
				(uint64_t)gate_up_elements * sizeof(uint16_t), stream) != cudaSuccess )
				state->copy_failed = 1u;
		}
		return;
	}
	if ( state->device_collective_created != 0 )
	{
		K3RunnerFusedPackKernel<<<(elements + 255u) / 256u, 256u, 0, stream>>>(
			phase0_source,b->hidden_bf16,b->shared_out_bf16,b->gate_up_bf16,
			state->fused_device,rows,phase,segments,0u);
		/* Hidden collective (band 0, width K3_HIDDEN): one stream-ordered
		 * op per segment (attention/hidden at fused[0], shared at
		 * fused[elements]) instead of one wide op - the collective's
		 * width is the row extent, so a 2xhidden reduce cannot ride it.
		 * Completions fire in ordinal order, reproducing the previous
		 * single-completion PartialAdd sequence exactly. */
		{
			const uint32_t op_count =
				(phase == 1u && segments == 2u) ? 2u : 1u;
			for ( uint32_t op = 0u; op < op_count; ++op )
			{
				uint16_t *segment = state->fused_device +
					(op == 0u ? 0u : elements);
				SparkK3RunnerTpContext *completion_context =
					K3RunnerTpContextAcquire(state);
				if ( completion_context == 0 )
					return;
				completion_context->owner = state;
				completion_context->fused = segment;
				completion_context->buffers = b;
				completion_context->stream = stream;
				completion_context->rows = rows;
				completion_context->boundary = (op == 0u) ? boundary : 0u;
				completion_context->segments = 1u;
				completion_context->phase = phase;
				completion_context->gate_up_elements = 0u;
				SparkTpDeviceCollectiveSubmission submission;
				memset(&submission, 0, sizeof(submission));
				submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
				submission.descriptor_bytes = sizeof(submission);
				submission.slot_index = 0u;
				submission.active_sequence_count = rows;
				submission.logical_sequence_count =
					state->logical_sequence_count;
				submission.flags =
					SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
				submission.ordinal = state->tp_next_ordinal++;
				submission.local_device = segment;
				submission.full_device = segment;
				submission.cuda_stream = stream;
				submission.completion_function = K3RunnerEmbedCompletion;
				submission.completion_context = 0;
				if ( SparkTpDeviceCollectiveEnqueue(&state->device_collective,
					&submission,
					SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16) != SPARK_STATUS_OK )
					state->tp_collective_failed = 1u;
				K3RunnerTpCompletion(completion_context, 0);
			}
		}
		return;
	}
	if ( state->collective_created != 0 )
	{
		cudaStreamSynchronize(stream);
		if ( phase == 0u )
		{
			if ( K3RunnerCopy(state->staging_values, phase0_source,
				(uint64_t)elements * sizeof(uint16_t), stream) != cudaSuccess )
				state->copy_failed = 1u;
			K3RunnerHostAllReduce(state, elements);
			if ( K3RunnerCopy(phase0_source, state->staging_values,
				(uint64_t)elements * sizeof(uint16_t), stream) != cudaSuccess )
				state->copy_failed = 1u;
			if ( boundary != 0u )
				K3PartialSet(b, phase0_source, rows, stream);
			else
				K3PartialAdd(b, phase0_source, rows, stream);
		}
		else
		{
			if ( K3RunnerCopy(state->staging_values, b->hidden_bf16,
				(uint64_t)elements * sizeof(uint16_t), stream) != cudaSuccess )
				state->copy_failed = 1u;
			if ( segments == 2u )
				if ( K3RunnerCopy(state->staging_values + elements, b->shared_out_bf16,
					(uint64_t)elements * sizeof(uint16_t), stream) != cudaSuccess )
					state->copy_failed = 1u;
			K3RunnerHostAllReduce(state, (uint64_t)segments * elements);
			if ( K3RunnerCopy(b->hidden_bf16, state->staging_values,
				(uint64_t)elements * sizeof(uint16_t), stream) != cudaSuccess )
				state->copy_failed = 1u;
			if ( segments == 2u )
				if ( K3RunnerCopy(b->shared_out_bf16, state->staging_values + elements,
					(uint64_t)elements * sizeof(uint16_t), stream) != cudaSuccess )
					state->copy_failed = 1u;
			K3PartialAdd(b, b->hidden_bf16, rows, stream);
			if ( segments == 2u )
				K3PartialAdd(b, b->shared_out_bf16, rows, stream);
		}
	}
}
typedef struct SparkK3ManifestCheckContext
{
	SparkK3Pack *pack;
} SparkK3ManifestCheckContext;

static SparkStatus SparkK3ManifestCheck(const SparkWeightdManifest *manifest,
	void *context)
{
	SparkK3ManifestCheckContext *check = (SparkK3ManifestCheckContext *)context;
	SparkK3Pack *pack = check->pack;
	uint32_t group_index = 0u;
	uint32_t layer;
	if ( manifest->group_count == 0u ||
		manifest->range_count < manifest->group_count )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	for ( layer = pack->config.first_layer;
		layer < pack->config.first_layer + pack->config.layers;
		++layer )
	{
		char name[SPARK_K3_PACK_MAX_NAME_BYTES];
		SparkK3PackEntry w1;
		SparkK3PackEntry w2;
		int have_w1;
		int have_w2;
		uint32_t expert;
		snprintf(name, sizeof(name), "model.layers.%u.expert_w1_weight",
			layer);
		have_w1 = SparkK3PackLoadEntry(pack, name, &w1) == SPARK_STATUS_OK;
		snprintf(name, sizeof(name), "model.layers.%u.expert_w2_weight",
			layer);
		have_w2 = SparkK3PackLoadEntry(pack, name, &w2) == SPARK_STATUS_OK;
		if ( have_w1 != have_w2 )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		if ( !have_w1 )
			continue;
		/* per-expert divisibility per tensor only: k3's w1/w2 expert
		 * geometries differ (intermediate 6144 vs 3072 halves the w2
		 * span), so demanding equal per-expert bytes rejects the real
		 * deployed manifests — measured: w1 2924544 vs w2 1462272
		 * bytes/expert on every rank pack (first-launch find #7, the
		 * lazy-attach manifest check). Each range is still validated
		 * against ITS tensor's span below. */
		if ( w1.bytes % pack->config.experts != 0u ||
			w2.bytes % pack->config.experts != 0u )
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		for ( expert = 0u; expert < pack->config.experts; ++expert )
		{
			const SparkWeightdRangeGroup *group =
				&manifest->groups[group_index + expert];
			const SparkWeightdRange *range;
			uint32_t r;
			if ( group->layer != layer || group->expert != expert ||
				group->range_count != 2u )
				SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
			for ( r = 0u; r < 2u; ++r )
			{
				const SparkK3PackEntry *tensor =
					r == 0u ? &w1 : &w2;
				uint64_t expert_bytes =
					tensor->bytes / pack->config.experts;
				uint64_t expected = pack->payload_base +
					tensor->payload_offset +
					(uint64_t)expert * expert_bytes;
				range = &manifest->ranges[group->first_range + r];
				if ( range->offset != expected ||
					range->bytes != expert_bytes ||
					range->kind != r ||
					range->layer != layer ||
					range->expert != expert )
					SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
			}
		}
		group_index += pack->config.experts;
	}
	if ( group_index != manifest->group_count )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	return SPARK_STATUS_OK;
}

static SparkStatus SparkK3RunnerReleaseLease(SparkK3RunnerState *state)
{
	SparkStatus status;
	if ( state->lease_identifier == 0u )
		return(state->lease_phase == 0u ? SPARK_STATUS_OK : SPARK_STATUS_VALIDATION_FAILED);
	if ( state->lazy_pack == 0 || state->lazy_pack->map == 0 ||
		state->lease_phase < SPARK_K3_LEASE_ACQUIRED || state->lease_phase > SPARK_K3_LEASE_RECORDED )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( state->lease_phase == SPARK_K3_LEASE_BEGUN )
	{
		status = SparkWeightdMapRecordCompletion(state->lazy_pack->map,state->lease_identifier,state->stream);
		if ( status != SPARK_STATUS_OK )
			return(status);
		state->lease_phase = SPARK_K3_LEASE_RECORDED;
	}
	status = SparkWeightdMapRelease(state->lazy_pack->map,state->lease_identifier,SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS);
	if ( status == SPARK_STATUS_OK )
	{
		state->lease_identifier = 0u;
		state->lease_phase = 0u;
		state->lease_address = 0;
	}
	return(status);
}

static SparkStatus K3RunnerLeaseTensorBase(SparkK3RunnerState *state,
	uint32_t layer, SparkWeightdExpertKey *keys, uint32_t *count)
{
	if ( state->lease_tensor_base == 0u || (*count != 0u && keys[0].expert == 0u) )
		return SPARK_STATUS_OK;
	if ( *count >= SPARK_WEIGHTD_LEASE_GROUPS_MAX )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	memmove(&keys[1], &keys[0], (size_t)*count * sizeof(keys[0]));
	keys[0].layer = layer;
	keys[0].expert = 0u;
	(*count)++;
	return SPARK_STATUS_OK;
}

static int32_t SparkK3RunnerLazyAcquire(void *context, uint32_t layer,
	void *buffers_void)
{
	SparkK3RunnerState *state = (SparkK3RunnerState *)context;
	K3LayerBuffers *buffers = (K3LayerBuffers *)buffers_void;
	SparkWeightdMap *map;
	SparkWeightdExpertKey keys[SPARK_WEIGHTD_LEASE_GROUPS_MAX];
	uint32_t count = 0u;
	cudaError_t error;
	SparkStatus status;
	if ( state == 0 || state->lazy_pack == 0 || buffers == 0 ||
		layer >= K3_LAYERS )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	map = state->lazy_pack->map;
	if ( map == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	error = cudaStreamSynchronize(state->stream);
	if ( error != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	status = SparkK3RunnerReleaseLease(state);
	if ( status != SPARK_STATUS_OK )
		return status;
	error = cudaMemcpy(state->group_offset_host, buffers->group_row_offset,
		(K3_EXPERTS + 1u) * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	if ( error != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	status = SparkWeightdRouteKeys(layer, state->group_offset_host,
		K3_EXPERTS, state->rows * K3_TOP_K, keys,
		SPARK_WEIGHTD_LEASE_GROUPS_MAX, &count);
	if ( status != SPARK_STATUS_OK )
		return status;
	status = K3RunnerLeaseTensorBase(state, layer, keys, &count);
	if ( status != SPARK_STATUS_OK )
		return status;
	SparkK3RunnerStrayAccount(state, keys, count);
	status = SparkWeightdMapAcquire(map, keys, count,
		&state->lease_identifier, SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS);
	if ( state->lease_identifier != 0u )
		state->lease_phase = SPARK_K3_LEASE_ACQUIRED;
	if ( status != SPARK_STATUS_OK )
	{
		(void)SparkK3RunnerReleaseLease(state);
		SPARK_FAIL(status);
	}
	if ( state->lease_identifier == 0u )
		return SPARK_STATUS_VALIDATION_FAILED;
	status = SparkWeightdMapBeginUse(map,
		state->lease_identifier, &state->lease_address);
	if ( status != SPARK_STATUS_OK )
		return status;
	state->lease_phase = SPARK_K3_LEASE_BEGUN;
	buffers->expert_w1_weight = (const uint8_t *)state->lease_address +
		state->layer_w1_offset[layer];
	buffers->expert_w2_weight = (const uint8_t *)state->lease_address +
		state->layer_w2_offset[layer];
	return LM_LAUNCH_OK;
}

static void SparkK3RunnerLazyRelease(void *context, uint32_t layer)
{
	SparkK3RunnerState *state = (SparkK3RunnerState *)context;
	SparkStatus status;
	(void)layer;
	if ( state == 0 )
		return;
	status = SparkK3RunnerReleaseLease(state);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"sparkpipe_k3: lease release failed status=%d (retained for recovery)\n",(int)status);
}

static SparkStatus K3RunnerEnvUnsigned64(const char *name, uint64_t minimum,
	uint64_t maximum, uint64_t *value)
{
	const char *text;
	char *end;
	unsigned long long parsed;
	if ( value == 0 || minimum > maximum )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	text = getenv(name);
	if ( text == 0 || text[0] == '\0' )
	{
		fprintf(stderr, "sparkpipe_k3: config_missing name=%s\n", name);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	errno = 0;
	parsed = strtoull(text, &end, 10);
	if ( errno != 0 || end == text || *end != '\0' )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	if ( (uint64_t)parsed < minimum || (uint64_t)parsed > maximum )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	*value = (uint64_t)parsed;
	return SPARK_STATUS_OK;
}

static SparkStatus K3RunnerCreateDispatch(SparkK3RunnerState *state,
	const SparkK3StageRunnerConfiguration *configuration)
{
	uint64_t budget = 0u, planned;
	SparkK3RankStateBytes state_plan;
	memset(&state_plan, 0, sizeof(state_plan));
	if ( K3RunnerEnvUnsigned64("SPARK_K3_STATE_BUDGET_BYTES", 1u, UINT64_MAX,
		&budget) != SPARK_STATUS_OK )
	{
		fprintf(stderr, "sparkpipe_k3: SPARK_K3_STATE_BUDGET_BYTES is required"
			" (per-rank KDA state + windows + MLA KV + dispatch scratch)\n");
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( SparkK3RankStateBytesFor(state->module.sizing.kda_layer_count,
			state->module.sizing.mla_layer_count,
			configuration->max_active_sequence_count, configuration->tp_degree,
			configuration->kv_pages_per_sequence, state->kv_page_bytes,
			&state_plan) == 0u || state_plan.total > budget )
	{
		fprintf(stderr, "sparkpipe_k3: rank state %llu exceeds"
			" SPARK_K3_STATE_BUDGET_BYTES %llu or tp_degree %u is invalid"
			" (refused before allocation)\n",
			(unsigned long long)state_plan.total, (unsigned long long)budget,
			configuration->tp_degree);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	if ( SparkK3DispatchCreate(&state->dispatch, &state->module.sizing,
		configuration->max_active_sequence_count,
		configuration->max_input_row_count,
		configuration->kv_pages_per_sequence,
		state->kv_page_bytes, configuration->tp_degree, 0) != SPARK_K3_DISPATCH_OK )
	{
		fprintf(stderr, "sparkpipe_k3: dispatch create failed tp_degree=%u\n",
			configuration->tp_degree);
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	planned = state->dispatch.state_bytes.total + state->dispatch.scratch_bytes;
	fprintf(stderr, "sparkpipe_k3: rank state planned=%llu budget=%llu"
		" kda_state=%llu kda_windows=%llu mla_kv=%llu scratch=%llu"
		" sequences=%u kda_heads_per_rank=%u\n",
		(unsigned long long)planned, (unsigned long long)budget,
		(unsigned long long)state->dispatch.state_bytes.kda_state,
		(unsigned long long)state->dispatch.state_bytes.kda_windows,
		(unsigned long long)state->dispatch.state_bytes.mla_kv,
		(unsigned long long)state->dispatch.scratch_bytes,
		configuration->max_active_sequence_count,
		state->dispatch.kda_rank_heads);
	if ( planned > budget )
	{
		fprintf(stderr, "sparkpipe_k3: rank state %llu exceeds"
			" SPARK_K3_STATE_BUDGET_BYTES %llu (fail-closed)\n",
			(unsigned long long)planned, (unsigned long long)budget);
		SparkK3DispatchDestroy(&state->dispatch);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	return SPARK_STATUS_OK;
}

static SparkStatus K3RunnerSeedIndices(SparkK3RunnerState *state,
	const SparkK3StageRunnerConfiguration *configuration)
{
	const uint32_t zero = 0u, one = 1u;
	if ( configuration->resident_sequence_capacity > configuration->max_active_sequence_count )
	{
		fprintf(stderr, "sparkpipe_k3: resident_sequence_capacity %u exceeds the %u KDA state slots\n",
			configuration->resident_sequence_capacity, configuration->max_active_sequence_count);
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	if ( K3RunnerCopy(state->positions, &zero, sizeof(uint32_t), state->stream) != cudaSuccess ||
		K3RunnerCopy(state->context_length, &one, sizeof(uint32_t), state->stream) != cudaSuccess ||
		K3RunnerCopy(state->sequence_of_row, &zero, sizeof(uint32_t), state->stream) != cudaSuccess ||
		K3RunnerCopy(state->kda_state_index, &zero, sizeof(uint32_t), state->stream) != cudaSuccess )
		return SPARK_STATUS_IO_ERROR;
	return SPARK_STATUS_OK;
}

SparkStatus SparkK3StageRunnerInitialize(
	SparkK3StageRunner *runner,
	const SparkK3StageRunnerConfiguration *configuration)
{
	SparkK3RunnerState *state;
	SparkStatus status;
	SparkK3PackEntry entry;
	uint32_t routes;
	if ( runner == 0 || configuration == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ( configuration->abi_version != SPARK_K3_STAGE_RUNNER_ABI_VERSION ||
		configuration->stage_index >= configuration->stage_count ||
		(configuration->stage_count != 1u &&
			configuration->stage_count != 4u) ||
		configuration->rank_pack_path == 0 ||
		configuration->max_active_sequence_count == 0u ||
		configuration->max_input_row_count == 0u )
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(runner, 0, sizeof(*runner));
	state = new SparkK3RunnerState;
	memset(state, 0, sizeof(*state));
	SparkK3RunnerStrayLoad(state);
	{
		uint32_t pool_index;
		for (pool_index = 0u;
			pool_index < K3_RUNNER_TP_CONTEXT_POOL_DEPTH;
			++pool_index)
		{
			state->tp_context_pool[pool_index].owner = state;
			state->tp_context_pool[pool_index].pool_next =
				state->tp_context_free_head;
			state->tp_context_free_head =
				&state->tp_context_pool[pool_index];
		}
	}
	runner->abi_version = SPARK_K3_STAGE_RUNNER_ABI_VERSION;
	runner->descriptor_bytes = SPARK_K3_STAGE_RUNNER_BYTES;
	runner->flags = configuration->flags;
	runner->stage_index = configuration->stage_index;
	runner->stage_count = configuration->stage_count;
	runner->tp_degree = configuration->tp_degree;
	runner->tp_rank = configuration->tp_rank;
	runner->owns_embedding = configuration->stage_index == 0u ? 1u : 0u;
	runner->owns_final_head = configuration->stage_index + 1u == configuration->stage_count ? 1u : 0u;
	runner->private_state = state;
	state->tp_rank = configuration->tp_rank;
	state->stream = (cudaStream_t)configuration->execution_stream;
	state->max_rows = configuration->max_input_row_count;
	state->max_context = configuration->resident_sequence_capacity;
	state->multiprocessors = configuration->multiprocessors;
	if ( configuration->kv_page_bytes == 0u )
		state->kv_page_bytes = K3GlobalKv::kPageBytes;
	else
		state->kv_page_bytes = configuration->kv_page_bytes;
	runner->stats.abi_version = SPARK_K3_STAGE_RUNNER_ABI_VERSION;
	runner->stats.descriptor_bytes = (uint32_t)sizeof(SparkK3StageRunnerStats);
	{
		uint32_t first_layer = configuration->stage_count == 4u ?
			K3RunnerFirstLayer(configuration->stage_index) :
			SPARK_K3_MODULE_DERIVE_SLICE;
		uint32_t layer_count = configuration->stage_count == 4u ?
			K3RunnerLayerCount(configuration->stage_index) :
			SPARK_K3_MODULE_DERIVE_SLICE;
		status = SparkK3ModuleInitialize(&state->module,
			configuration->rank_pack_path, first_layer, layer_count);
	}
	if ( status != SPARK_STATUS_OK )
		{ runner->private_state = 0; delete state; return status; }
	status = K3RunnerCreateDispatch(state, configuration);
	if ( status != SPARK_STATUS_OK )
		{ SparkK3ModuleDestroy(&state->module); runner->private_state = 0; delete state; return status; }
	status = SparkWeightdAttachRequested();
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr, "sparkpipe_k3: weightd attach not granted"
			" (SPARK_WEIGHTD_SOCKET unset or attach switch invalid),"
			" status=%d (fail-closed, no direct load)\n", (int)status);
		SparkK3DispatchDestroy(&state->dispatch);
		SparkK3ModuleDestroy(&state->module);
		runner->private_state = 0;
		delete state;
		SPARK_FAIL(status);
	}
	{
		SparkWeightdLazyAttachRequest request;
		SparkK3ManifestCheckContext check;
		const char *digest = getenv(SPARK_WEIGHTD_ATTACH_ENV_SHA256);
		uint64_t expert_pool = 0u, spine_budget = 0u;
		memset(&request, 0, sizeof(request));
		if ( digest == 0 || strlen(digest) != 64u ||
			strlen(configuration->rank_pack_path) >=
			sizeof(request.pack_path) )
		{
			fprintf(stderr, "sparkpipe_k3: lazy load requires"
				" SPARK_WEIGHTD_PACK_SHA256 (64 hex chars)\n");
			SparkK3DispatchDestroy(&state->dispatch);
			SparkK3ModuleDestroy(&state->module);
			runner->private_state = 0;
			delete state;
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		}
		if ( K3RunnerEnvUnsigned64("SPARK_WEIGHTD_EXPERT_POOL_BYTES",
			1u, UINT64_MAX, &expert_pool) != SPARK_STATUS_OK ||
			K3RunnerEnvUnsigned64("SPARK_WEIGHTD_SPINE_BUDGET_BYTES",
			1u, UINT64_MAX, &spine_budget) != SPARK_STATUS_OK )
		{
			fprintf(stderr, "sparkpipe_k3: lazy load requires the weightd"
				" pool and spine budget envs\n");
			SparkK3DispatchDestroy(&state->dispatch);
			SparkK3ModuleDestroy(&state->module);
			runner->private_state = 0;
			delete state;
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		}
		memcpy(request.identity.pack_sha256, digest, 64u);
		snprintf(request.identity.model, sizeof(request.identity.model), "kimi-k3");
		snprintf(request.identity.revision, sizeof(request.identity.revision), "mxfp4");
		request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
		/* the daemon's size-mismatch contract: identity.arena_bytes must
		 * equal the pack file size (WDATTACH rejects anything else with
		 * INVALID_ARGUMENT - measured against the release-shared weightd;
		 * the campaign-era daemon tolerated the old pool-sized value) */
		request.identity.arena_bytes = state->module.pack.file_bytes;
		request.identity.topology = configuration->tp_degree;
		memcpy(request.pack_path, configuration->rank_pack_path,
			strlen(configuration->rank_pack_path) + 1u);
		request.expert_pool_bytes = expert_pool;
		check.pack = &state->module.pack;
		status = SparkWeightdLazyPackCreateChecked(
			getenv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET), &request, spine_budget,
			SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS, SparkK3ManifestCheck,
			&check, &state->lazy_pack);
		if ( status != SPARK_STATUS_OK )
		{
			fprintf(stderr, "sparkpipe_k3: lazy startup failed status=%d"
				" (fail-closed, no eager fallback)\n", (int)status);
			SparkK3DispatchDestroy(&state->dispatch);
			SparkK3ModuleDestroy(&state->module);
			runner->private_state = 0;
			delete state;
			SPARK_FAIL(status);
		}
	}
	if ( SparkK3DispatchBindWeights(&state->dispatch,&state->module.pack,
			state->module.bound,state->module.bound_count,
			state->lazy_pack) != SPARK_K3_DISPATCH_OK )
		{ fprintf(stderr, "sparkpipe_k3: weight bind failed\n"); SparkK3DispatchDestroy(&state->dispatch); SparkK3ModuleDestroy(&state->module); runner->private_state = 0; delete state; return SPARK_STATUS_INTERNAL_ERROR; }
	state->vocab = state->module.pack.config.vocab;
	{
		uint32_t routed;
		state->group_offset_host =
			(uint32_t *)malloc((K3_EXPERTS + 1u) * 4u);
		if ( state->group_offset_host == 0 )
		{
			SparkK3DispatchDestroy(&state->dispatch);
			SparkK3ModuleDestroy(&state->module);
			runner->private_state = 0;
			delete state;
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		/* the slice bound is config.layers (this rank pack's stage
		 * slice), NOT config.total_layers (the whole model): a stage-0
		 * pack holds layers 0..23 of 93, and layer 0 is dense (no
		 * expert tensors). Skip dense layers like the manifest check
		 * does; every layer that HAS experts inside the slice must
		 * load both spans (first-launch find #8). */
		{
			uint32_t routed_layers = 0u;
			for ( routed = state->module.pack.config.first_layer;
				routed < state->module.pack.config.first_layer +
					state->module.pack.config.layers; ++routed )
			{
				SparkK3PackEntry w1;
				SparkK3PackEntry w2;
				int have_w1;
				int have_w2;
				char name[SPARK_K3_PACK_MAX_NAME_BYTES];
				snprintf(name, sizeof(name),
					"model.layers.%u.expert_w1_weight", routed);
				have_w1 = SparkK3PackLoadEntry(&state->module.pack,
					name, &w1) == SPARK_STATUS_OK;
				snprintf(name, sizeof(name),
					"model.layers.%u.expert_w2_weight", routed);
				have_w2 = SparkK3PackLoadEntry(&state->module.pack,
					name, &w2) == SPARK_STATUS_OK;
				if ( have_w1 != have_w2 )
					break;
				if ( !have_w1 )
					continue;
				state->layer_w1_offset[routed] =
					state->module.pack.payload_base + w1.payload_offset;
				state->layer_w2_offset[routed] =
					state->module.pack.payload_base + w2.payload_offset;
				routed_layers++;
			}
			if ( routed != state->module.pack.config.first_layer +
				state->module.pack.config.layers || routed_layers == 0u )
			{
				SparkK3DispatchDestroy(&state->dispatch);
				SparkK3ModuleDestroy(&state->module);
				runner->private_state = 0;
				delete state;
				SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
			}
		}
	}
	state->vocab_slice_rows = state->vocab;
	{
		uint64_t embed_rows = 0u, head_rows = 0u;
		if ( runner->owns_embedding != 0u &&
			SparkK3PackLoadEntry(&state->module.pack,"model.embed_tokens.weight",&entry) == 0 &&
			entry.shape_count >= 1u )
			embed_rows = entry.shape[0];
		if ( runner->owns_final_head != 0u &&
			SparkK3PackLoadEntry(&state->module.pack,"lm_head.weight",&entry) == 0 &&
			entry.shape_count >= 1u )
			head_rows = entry.shape[0];
		if ( (runner->owns_embedding != 0u && (embed_rows == 0u || embed_rows > state->vocab)) ||
			(runner->owns_final_head != 0u && (head_rows == 0u || head_rows > state->vocab)) ||
			(embed_rows != 0u && head_rows != 0u && embed_rows != head_rows) )
		{
			fprintf(stderr, "sparkpipe_k3: vocab shard rows embed=%llu head=%llu vocab=%u do not agree\n",
				(unsigned long long)embed_rows, (unsigned long long)head_rows, state->vocab);
			SparkK3DispatchDestroy(&state->dispatch);
			SparkK3ModuleDestroy(&state->module);
			runner->private_state = 0;
			delete state;
			SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
		}
		if ( head_rows != 0u )
			state->vocab_slice_rows = (uint32_t)head_rows;
		else if ( embed_rows != 0u )
			state->vocab_slice_rows = (uint32_t)embed_rows;
	}
	state->dispatch.buffers->tp_sharded = configuration->tp_degree > 1u ? 1u : 0u;
	state->lease_tensor_base = (uint32_t)(state->dispatch.buffers->routed_down_rows %
		K3_LAYER_TILE_N != 0u);
	state->dispatch.buffers->tp_rank = configuration->tp_rank;
	state->dispatch.slice_state->layer_collective = K3RunnerLayerCollective;
	state->dispatch.slice_state->collective_context = state;
	state->dispatch.slice_state->lazy_context = state;
	state->dispatch.slice_state->lazy_acquire =
		SparkK3RunnerLazyAcquire;
	state->dispatch.slice_state->lazy_release =
		SparkK3RunnerLazyRelease;
	if ( configuration->tp_degree > 1u )
	{
		if ( configuration->tp_collective == 0 &&
			configuration->device_collective == 0 )
			{ SparkK3DispatchDestroy(&state->dispatch); SparkK3ModuleDestroy(&state->module); runner->private_state = 0; delete state; return SPARK_STATUS_INVALID_ARGUMENT; }
		if ( configuration->tp_collective != 0 )
		{
			fprintf(stderr, "sparkpipe_k3: creating host collective tp=%u rank=%u port=%u\n",
				configuration->tp_degree, configuration->tp_rank,
				configuration->tp_collective->listen_port);
			status = SparkTpCollectiveCreate(configuration->tp_collective,&state->collective);
			fprintf(stderr, "sparkpipe_k3: host collective create -> %d\n", (int)status);
			if ( status != SPARK_STATUS_OK )
				{ SparkK3DispatchDestroy(&state->dispatch); SparkK3ModuleDestroy(&state->module); runner->private_state = 0; delete state; return status; }
			state->collective_created = 1;
		}
	}
	if ( configuration->layer_collective_override != 0 )
	{
		state->dispatch.slice_state->layer_collective =
			configuration->layer_collective_override;
		state->dispatch.slice_state->collective_context =
			configuration->layer_collective_context;
	}
	if ( configuration->device_collective != 0 )
	{
		/* Width contract (cold16): the hidden collective must be exactly
		 * K3_HIDDEN wide - every ALL_REDUCE moves rows x this width from
		 * the submitted device pointer, so a mismatch is memory corruption
		 * or a silent wrong-extent reduce, not a slowdown. */
		if ( configuration->device_collective->local_hidden_dimension !=
			K3_HIDDEN )
		{
			fprintf(stderr, "sparkpipe_k3: device collective width %u != "
				"K3_HIDDEN %u (cold16 width contract)\n",
				configuration->device_collective->local_hidden_dimension,
				K3_HIDDEN);
			{ SparkK3StageRunnerDestroy(runner); return SPARK_STATUS_INVALID_ARGUMENT; }
		}
		state->fused_rows = configuration->max_input_row_count;
		cudaMalloc(&state->fused_device,
			(uint64_t)state->fused_rows * K3_TOP_K *
			(K3_EXPERT_INTERMEDIATE * 2u) * 2u);
		SparkTpDeviceCollectiveConfig device_config =
			*configuration->device_collective;
		if ( device_config.backend_kind ==
			SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT )
		{
			device_config.combine_bf16_function = K3RunnerCombineBf16;
			device_config.combine_tp4_bf16_function = K3RunnerCombineTp4Bf16;
			device_config.combine_u64_max_function = K3RunnerCombineU64Max;
			device_config.combine_gather_bf16_function = K3RunnerCombineGatherBf16;
			device_config.combine_context = state;
		}
		status = SparkTpDeviceCollectiveCreate(&device_config,
			&state->device_collective);
		if ( status != SPARK_STATUS_OK )
			{ SparkK3StageRunnerDestroy(runner); return status; }
		state->device_collective_created = 1;
		if ( state->lazy_pack != 0 &&
			state->lazy_pack->attached.mesh_send_buffer_addr != 0 )
			status = SparkTpDeviceCollectivePrepareReceiveBf16(
				&state->device_collective,
				(void *)(uintptr_t)state->lazy_pack->attached.mesh_send_buffer_addr,
				0u,0u,0u,0u);
		if ( status != SPARK_STATUS_OK )
			{ SparkK3StageRunnerDestroy(runner); return status; }
	}
	if ( configuration->device_collective_wide != 0 )
	{
		if ( configuration->device_collective_wide->local_hidden_dimension !=
			K3_RUNNER_GATE_UP_WIDTH )
		{
			fprintf(stderr, "sparkpipe_k3: wide device collective width %u != "
				"gate_up %u (cold16 width contract)\n",
				configuration->device_collective_wide->local_hidden_dimension,
				K3_RUNNER_GATE_UP_WIDTH);
			{ SparkK3StageRunnerDestroy(runner); return SPARK_STATUS_INVALID_ARGUMENT; }
		}
		SparkTpDeviceCollectiveConfig wide_config =
			*configuration->device_collective_wide;
		if ( wide_config.backend_kind ==
			SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT )
		{
			wide_config.combine_bf16_function = K3RunnerCombineBf16;
			wide_config.combine_tp4_bf16_function = K3RunnerCombineTp4Bf16;
			wide_config.combine_u64_max_function = K3RunnerCombineU64Max;
			wide_config.combine_gather_bf16_function = K3RunnerCombineGatherBf16;
			wide_config.combine_context = state;
		}
		status = SparkTpDeviceCollectiveCreate(&wide_config,
			&state->device_collective_wide);
		if ( status != SPARK_STATUS_OK )
			{ SparkK3StageRunnerDestroy(runner); return status; }
		state->device_collective_wide_created = 1;
		if ( state->lazy_pack != 0 &&
			state->lazy_pack->attached.mesh_send_buffer_addr != 0 )
			status = SparkTpDeviceCollectivePrepareReceiveBf16(
				&state->device_collective_wide,
				(void *)(uintptr_t)state->lazy_pack->attached.mesh_send_buffer_addr,
				0u,0u,0u,0u);
		if ( status != SPARK_STATUS_OK )
			{ SparkK3StageRunnerDestroy(runner); return status; }
	}
	if ( runner->owns_embedding != 0u )
	{
		const void *embed_slice = 0;
		if ( SparkK3PackLoadEntry(&state->module.pack,
			"model.embed_tokens.weight",&entry) != SPARK_STATUS_OK )
			{ SparkK3StageRunnerDestroy(runner); SPARK_FAIL(SPARK_STATUS_PARSE_ERROR); }
		if ( SparkWeightdLazyPackSlice(state->lazy_pack,
			state->module.pack.payload_base + entry.payload_offset,
			entry.bytes, &embed_slice) != SPARK_STATUS_OK )
			{ SparkK3StageRunnerDestroy(runner); SPARK_FAIL(SPARK_STATUS_PARSE_ERROR); }
		state->embed_weight = (const uint16_t *)embed_slice;
	}
	if ( runner->owns_final_head != 0u )
	{
		const void *norm_slice = 0;
		const void *head_slice = 0;
		if ( SparkK3PackLoadEntry(&state->module.pack,"model.norm.weight",&entry) != SPARK_STATUS_OK ||
			SparkWeightdLazyPackSlice(state->lazy_pack,
			state->module.pack.payload_base + entry.payload_offset,
			entry.bytes, &norm_slice) != SPARK_STATUS_OK )
			{ SparkK3StageRunnerDestroy(runner); SPARK_FAIL(SPARK_STATUS_PARSE_ERROR); }
		state->head_norm_weight = (const uint16_t *)norm_slice;
		if ( SparkK3PackLoadEntry(&state->module.pack,"lm_head.weight",&entry) != SPARK_STATUS_OK ||
			SparkWeightdLazyPackSlice(state->lazy_pack,
			state->module.pack.payload_base + entry.payload_offset,
			entry.bytes, &head_slice) != SPARK_STATUS_OK )
			{ SparkK3StageRunnerDestroy(runner); SPARK_FAIL(SPARK_STATUS_PARSE_ERROR); }
		state->head_weight = (const uint16_t *)head_slice;
	{
		uint64_t shard_rows = (uint64_t)state->vocab_slice_rows;
		uint64_t dim = K3_HIDDEN;
		if (cudaMalloc(&state->head_certified_fp8_payload, shard_rows * dim) == cudaSuccess &&
			cudaMalloc(&state->head_certified_fp8_scale_f32, shard_rows * (dim / 32u) * sizeof(float)) == cudaSuccess &&
			cudaMalloc(&state->head_certified_fp8_norm_f32, shard_rows * (dim / 32u) * sizeof(float)) == cudaSuccess &&
			cudaMalloc(&state->head_certified_scratch, SparkHeadCertifiedFp8ScratchBytes(shard_rows,dim)) == cudaSuccess &&
			cudaMalloc(&state->head_certified_candidates, SparkHeadCertifiedFp8CandidateBytes(shard_rows)) == cudaSuccess &&
			cudaMalloc(&state->head_screened_count, sizeof(uint32_t)) == cudaSuccess &&
			SparkLmHostLaunchHeadCertifiedFp8Quantize(0,state->head_weight,
				state->head_certified_fp8_payload,state->head_certified_fp8_scale_f32,
				state->head_certified_fp8_norm_f32,(uint32_t)shard_rows,(uint32_t)dim) == cudaSuccess)
		{
			cudaError_t quantized = cudaDeviceSynchronize();
			if ( quantized != cudaSuccess )
			{
				fprintf(stderr, "sparkpipe_k3: certified head quantize failed cuda=%d rows=%llu\n",
					(int)quantized, (unsigned long long)shard_rows);
				SparkK3StageRunnerDestroy(runner);
				SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
			}
		}
		else
		{
			cudaFree(state->head_certified_fp8_payload);
			cudaFree(state->head_certified_fp8_scale_f32);
			cudaFree(state->head_certified_fp8_norm_f32);
			cudaFree(state->head_certified_scratch);
			cudaFree(state->head_certified_candidates);
			cudaFree(state->head_screened_count);
			state->head_certified_fp8_payload = 0;
		}
	}
	}
	state->staging_capacity = configuration->max_input_row_count *
		K3_TOP_K * (K3_EXPERT_INTERMEDIATE * 2u);
	state->staging_values = new uint16_t[state->staging_capacity];
	state->staging_scratch = new uint16_t[state->staging_capacity];
	state->fused_capacity = state->staging_capacity;
	state->head_slots_capacity = configuration->max_input_row_count * 2u * configuration->tp_degree;
	state->head_slots_host = new float[state->head_slots_capacity];
	cudaMalloc(&state->head_slots_device,(uint64_t)state->head_slots_capacity * 4u);
	cudaMalloc(&state->head_maxloc,
		(uint64_t)configuration->max_input_row_count * sizeof(uint64_t));
	routes = configuration->max_input_row_count * K3_TOP_K;
	cudaMalloc(&state->route_expert,(uint64_t)routes * 4u);
	cudaMalloc(&state->route_packed_row,(uint64_t)routes * 4u);
	cudaMalloc(&state->route_source_token,(uint64_t)routes * 4u);
	cudaMalloc(&state->route_weight,(uint64_t)routes * 4u);
	cudaMalloc(&state->group_row_offset,(uint64_t)(K3_EXPERTS + 1u) * 4u);
	cudaMalloc(&state->group_tile_prefix_w1,(uint64_t)(K3_EXPERTS + 1u) * 4u);
	cudaMalloc(&state->group_tile_prefix_w2,(uint64_t)(K3_EXPERTS + 1u) * 4u);
	cudaMalloc(&state->dense_row_offset, 8u);
	cudaMalloc(&state->dense_tile_prefix, 8u);
	state->head_tiles = (state->vocab_slice_rows + K3_HEAD_TILE - 1u) / K3_HEAD_TILE;
	cudaMalloc(&state->head_candidate_token,
		(uint64_t)configuration->max_input_row_count * state->head_tiles * 4u);
	cudaMalloc(&state->head_candidate_score,
		(uint64_t)configuration->max_input_row_count * state->head_tiles * 4u);
	cudaMalloc(&state->output_token,
		(uint64_t)configuration->max_input_row_count * 4u);
	cudaMalloc(&state->output_score,
		(uint64_t)configuration->max_input_row_count * 4u);
	cudaMalloc(&state->positions, 4u);
	if ( cudaMalloc(&state->token_ids_device,
		(uint64_t)configuration->max_input_row_count * sizeof(*state->token_ids_device)) != cudaSuccess )
		{ SparkK3StageRunnerDestroy(runner); SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED); }
	cudaMalloc(&state->context_length, 4u);
	cudaMalloc(&state->sequence_of_row, 4u);
	cudaMalloc(&state->kda_state_index, 4u);
	status = K3RunnerSeedIndices(state, configuration);
	if ( status != SPARK_STATUS_OK )
		{ SparkK3StageRunnerDestroy(runner); SPARK_FAIL(status); }
	state->output_token_host = new uint32_t[configuration->max_input_row_count];
	state->output_score_host = new float[configuration->max_input_row_count];
	return SPARK_STATUS_OK;
}

static SparkStatus K3RunnerHeadExchange(SparkK3RunnerState *state,
	uint32_t rows, uint32_t tp_degree, uint32_t tp_rank,
	uint32_t *output_token_ids, float *output_scores)
{
	uint32_t slots = rows * 2u * tp_degree;
	SparkStatus status;
	uint32_t row, rank_slot, best, r;
	if ( tp_degree <= 1u )
		return SPARK_STATUS_OK;
	memset(state->head_slots_host, 0, (uint64_t)slots * 4u);
	for ( row = 0u; row < rows; ++row )
	{
		rank_slot = tp_rank * 2u;
		state->head_slots_host[row * 2u * tp_degree + rank_slot] = output_scores[row];
		state->head_slots_host[row * 2u * tp_degree + rank_slot + 1u] = (float)output_token_ids[row];
	}
	status = SparkTpCollectiveAllReduceSumF32(&state->collective,
		state->head_slots_host, slots, (float *)state->staging_values);
	if ( status != SPARK_STATUS_OK )
		return status;
	for ( row = 0u; row < rows; ++row )
	{
		best = 0u;
		for ( r = 1u; r < tp_degree; ++r )
		{
			float bs = state->head_slots_host[row * 2u * tp_degree + best * 2u];
			float rs = state->head_slots_host[row * 2u * tp_degree + r * 2u];
			if ( rs > bs )
				best = r;
		}
		output_token_ids[row] = (uint32_t)state->head_slots_host[row * 2u * tp_degree + best * 2u + 1u];
		output_scores[row] = state->head_slots_host[row * 2u * tp_degree + best * 2u];
	}
	return SPARK_STATUS_OK;
}

static_assert(SPARK_K3_RESIDUAL_BANK_BYTES_PER_ROW == K3_ATTNRES_BANK_BYTES,
	"the pipeline residual bank sideband carries every attention-residual slot");

static SparkStatus K3RunnerChainBegin(SparkK3RunnerState *state, uint64_t request_id)
{
	const uint64_t key = request_id & SPARK_TP_DEVICE_COLLECTIVE_CHAIN_ID_MASK;
	SparkStatus status = SPARK_STATUS_OK;
	if ( state->device_collective_created != 0 )
		status = SparkTpDeviceCollectiveChainKey(&state->device_collective, key);
	if ( status == SPARK_STATUS_OK && state->device_collective_wide_created != 0 )
		status = SparkTpDeviceCollectiveChainKey(&state->device_collective_wide, key);
	return status;
}

static SparkStatus K3RunnerChainEnd(SparkK3RunnerState *state, cudaStream_t stream)
{
	SparkStatus status = SPARK_STATUS_OK;
	if ( state->device_collective_created != 0 )
		status = SparkTpDeviceCollectiveEndChain(&state->device_collective, stream);
	if ( status == SPARK_STATUS_OK && state->device_collective_wide_created != 0 )
		status = SparkTpDeviceCollectiveEndChain(&state->device_collective_wide, stream);
	return status;
}

SparkStatus SparkK3StageRunnerSubmit(
	SparkK3StageRunner *runner,
	const SparkK3StageRunnerDispatch *dispatch)
{
	SparkK3RunnerState *state;
	SparkK3StepInput in;
	K3LayerBuffers *b;
	cudaStream_t stream;
	uint32_t rows, sequences, packed_rows;
	uint32_t dense_offsets[2];
	int32_t status;
	SparkStatus exchange_status;
	SparkModelDriverCompletion completion;
	uint32_t *host_tokens;
	float *host_scores;
	uint32_t i;
	if ( runner == 0 || dispatch == 0 || runner->private_state == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	state = (SparkK3RunnerState *)runner->private_state;
	rows = dispatch->row_count;
	if ( rows == 0u || rows > state->max_rows || dispatch->active_sequence_count == 0u || dispatch->active_sequence_count > rows ||
		(runner->owns_embedding != 0u && dispatch->token_ids == 0) )
		return SPARK_STATUS_INVALID_ARGUMENT;
	stream = state->stream;
	exchange_status = K3RunnerChainBegin(state, dispatch->request_id);
	if ( exchange_status != SPARK_STATUS_OK )
		return exchange_status;
	state->rows = rows;
	state->logical_sequence_count = dispatch->active_sequence_count;
	b = state->dispatch.buffers;
	sequences = dispatch->active_sequence_count;
	packed_rows = rows * K3_TOP_K;
	memset(&in, 0, sizeof(in));
	if ( runner->owns_embedding != 0u )
	{
		if ( cudaMemcpyAsync(state->token_ids_device, dispatch->token_ids,
			(uint64_t)rows * sizeof(*state->token_ids_device),
			cudaMemcpyDefault, stream) != cudaSuccess ||
			cudaStreamSynchronize(stream) != cudaSuccess )
			return SPARK_STATUS_IO_ERROR;
		status = K3Embedding(state->embed_weight, state->token_ids_device,
			b->hidden_bf16, rows, runner->tp_rank * state->vocab_slice_rows,
			state->vocab_slice_rows, stream);
		if ( status != LM_LAUNCH_OK )
			return SPARK_STATUS_INTERNAL_ERROR;
		if ( K3RunnerReduceBf16(state, stream, b->hidden_bf16, rows) != SPARK_STATUS_OK )
			return SPARK_STATUS_INTERNAL_ERROR;
	}
	else
	{
		if ( dispatch->hidden_input_bf16 == 0 )
			return SPARK_STATUS_INVALID_ARGUMENT;
		if ( K3RunnerCopy(b->hidden_bf16, dispatch->hidden_input_bf16,
			(uint64_t)rows * K3_HIDDEN * sizeof(uint16_t), stream) != cudaSuccess )
			state->copy_failed = 1u;
		if ( dispatch->residual_bank_input != 0 )
		{
			if ( dispatch->residual_bank_input_bytes < (uint64_t)rows * SPARK_K3_RESIDUAL_BANK_BYTES_PER_ROW )
				return SPARK_STATUS_INVALID_ARGUMENT;
			if ( K3RunnerCopy(b->attnres_bank_bf16, dispatch->residual_bank_input,
				(uint64_t)rows * SPARK_K3_RESIDUAL_BANK_BYTES_PER_ROW, stream) != cudaSuccess )
				state->copy_failed = 1u;
		}
		if ( K3RunnerCopy(b->attnres_partial_bf16, b->hidden_bf16,
			(uint64_t)rows * K3_HIDDEN * sizeof(*b->hidden_bf16), stream) != cudaSuccess )
			state->copy_failed = 1u;
	}
	in.hidden_in = b->hidden_bf16;
	in.positions = dispatch->positions;
	in.context_length = dispatch->context_length;
	in.sequence_of_row = dispatch->sequence_of_row;
	in.sequence_row_begin = dispatch->sequence_row_begin;
	in.kda_state_index = dispatch->kda_state_index;
	in.route_expert = state->route_expert;
	in.route_packed_row = state->route_packed_row;
	in.route_source_token = state->route_source_token;
	in.route_weight = state->route_weight;
	in.group_row_offset = state->group_row_offset;
	in.group_tile_prefix_w1 = state->group_tile_prefix_w1;
	in.group_tile_prefix_w2 = state->group_tile_prefix_w2;
	in.dense_row_offset = state->dense_row_offset;
	in.dense_tile_prefix = state->dense_tile_prefix;
	in.head_candidate_score = state->head_candidate_score;
	in.head_candidate_token = state->head_candidate_token;
	in.output_token = state->output_token;
	in.output_score = state->output_score;
	(void)dense_offsets;
	status = K3RunnerLaunchSliceDirect(state, &in, rows, sequences,
		packed_rows, stream);
	if ( status != SPARK_K3_DISPATCH_OK )
		{ fprintf(stderr, "sparkpipe_k3: slice dispatch failed %d\n", status); return SPARK_STATUS_INTERNAL_ERROR; }
	if ( runner->owns_final_head != 0u && state->head_weight != 0 )
	{
		if ( rows == 1u && state->head_certified_fp8_payload != 0 )
			status = K3HeadCertifiedB1(b, state->head_norm_weight, state->head_weight, state->head_certified_fp8_payload, state->head_certified_fp8_scale_f32, state->head_certified_fp8_norm_f32, state->head_certified_scratch, state->head_certified_candidates, state->head_screened_count, state->tp_rank * state->vocab_slice_rows, state->vocab_slice_rows, stream);
		else
			status = K3HeadRankSlice(b, state->head_norm_weight, state->head_weight,
				state->tp_rank * state->vocab_slice_rows, state->vocab_slice_rows, rows, stream);
		if ( status != LM_LAUNCH_OK )
			{ fprintf(stderr, "sparkpipe_k3: final head launch failed %d\n", status); return SPARK_STATUS_INTERNAL_ERROR; }
		if ( state->device_collective_created != 0 )
		{
			SparkTpDeviceCollectiveSubmission submission;
			if ( K3HeadMaxlocPack(state->output_score, state->output_token,
				state->head_maxloc, rows, stream) != LM_LAUNCH_OK )
				{ fprintf(stderr, "sparkpipe_k3: head maxloc pack failed\n"); return SPARK_STATUS_INTERNAL_ERROR; }
			memset(&submission, 0, sizeof(submission));
			submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
			submission.descriptor_bytes = sizeof(submission);
			submission.slot_index = 0u;
			submission.active_sequence_count = rows;
			submission.logical_sequence_count = state->logical_sequence_count;
			submission.flags =
				SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
			submission.ordinal = state->tp_next_ordinal++;
			submission.local_device = state->head_maxloc;
			submission.full_device = state->head_maxloc;
			submission.cuda_stream = stream;
			submission.completion_function = K3RunnerEmbedCompletion;
			submission.completion_context = 0;
			if ( SparkTpDeviceCollectiveEnqueue(&state->device_collective,
					&submission,
					SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64) !=
				SPARK_STATUS_OK )
				{ fprintf(stderr, "sparkpipe_k3: head argmax collective enqueue failed\n"); return SPARK_STATUS_INTERNAL_ERROR; }
			if ( K3HeadMaxlocUnpack(state->head_maxloc, state->output_token,
				state->output_score, rows, stream) != LM_LAUNCH_OK )
				{ fprintf(stderr, "sparkpipe_k3: head maxloc unpack failed\n"); return SPARK_STATUS_INTERNAL_ERROR; }
			cudaStreamSynchronize(stream);
			if ( K3RunnerCopy(state->output_token_host, state->output_token,
				(uint64_t)rows * sizeof(uint32_t), stream) != cudaSuccess )
				state->copy_failed = 1u;
			if ( K3RunnerCopy(state->output_score_host, state->output_score,
				(uint64_t)rows * sizeof(float), stream) != cudaSuccess )
				state->copy_failed = 1u;
			if ( dispatch->output_token_ids != 0 )
				if ( K3RunnerCopy(dispatch->output_token_ids, state->output_token,
					(uint64_t)rows * sizeof(uint32_t), stream) != cudaSuccess )
					state->copy_failed = 1u;
			if ( dispatch->output_scores != 0 )
				if ( K3RunnerCopy(dispatch->output_scores, state->output_score,
					(uint64_t)rows * sizeof(float), stream) != cudaSuccess )
					state->copy_failed = 1u;
		}
		else
		{
			cudaStreamSynchronize(stream);
			if ( K3RunnerCopy(state->output_token_host, state->output_token,
				(uint64_t)rows * sizeof(uint32_t), stream) != cudaSuccess )
				state->copy_failed = 1u;
			if ( K3RunnerCopy(state->output_score_host, state->output_score,
				(uint64_t)rows * sizeof(float), stream) != cudaSuccess )
				state->copy_failed = 1u;
			exchange_status = K3RunnerHeadExchange(state, rows, runner->tp_degree,
				runner->tp_rank, state->output_token_host, state->output_score_host);
			if ( exchange_status != SPARK_STATUS_OK )
				return exchange_status;
			if ( dispatch->output_token_ids != 0 )
				if ( K3RunnerCopy(dispatch->output_token_ids, state->output_token_host,
					(uint64_t)rows * sizeof(uint32_t), stream) != cudaSuccess )
					state->copy_failed = 1u;
			if ( dispatch->output_scores != 0 )
				if ( K3RunnerCopy(dispatch->output_scores, state->output_score_host,
					(uint64_t)rows * sizeof(float), stream) != cudaSuccess )
					state->copy_failed = 1u;
		}
	}
	else if ( dispatch->hidden_output_bf16 != 0 )
	{
		if ( K3RunnerCopy(dispatch->hidden_output_bf16, b->hidden_bf16,
			(uint64_t)rows * K3_HIDDEN * sizeof(uint16_t), stream) != cudaSuccess )
			state->copy_failed = 1u;
		if ( dispatch->residual_bank_output != 0 )
		{
			if ( dispatch->residual_bank_output_bytes < (uint64_t)rows * SPARK_K3_RESIDUAL_BANK_BYTES_PER_ROW )
				return SPARK_STATUS_INVALID_ARGUMENT;
			if ( K3RunnerCopy(dispatch->residual_bank_output, b->attnres_bank_bf16,
				(uint64_t)rows * SPARK_K3_RESIDUAL_BANK_BYTES_PER_ROW, stream) != cudaSuccess )
				state->copy_failed = 1u;
		}
	}
	exchange_status = K3RunnerTakeFailure(state);
	if ( exchange_status != SPARK_STATUS_OK )
		return exchange_status;
	if ( state->tp_context_overflow != 0u )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	exchange_status = K3RunnerChainEnd(state, stream);
	if ( exchange_status != SPARK_STATUS_OK )
		return exchange_status;
	runner->stats.submitted_count++;
	runner->stats.completed_count++;
	if ( dispatch->completion_function != 0 )
	{
		if ( runner->owns_final_head == 0u )
		{
			memset(state->output_token_host, 0, (uint64_t)rows * 4u);
		}
		memset(&completion, 0, sizeof(completion));
		completion.request_id = dispatch->request_id;
		completion.sequence_id = dispatch->sequence_id;
		completion.sequence_position = dispatch->sequence_position;
		completion.accepted_token_count = rows;
		completion.token_count = rows;
		completion.tokens_per_sequence = 1u;
		for ( i = 0u; i < rows && i < SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY; ++i )
			completion.token_ids[i] = state->output_token_host[i];
		completion.status = SPARK_STATUS_OK;
		dispatch->completion_function(dispatch->completion_context, &completion);
	}
	return SPARK_STATUS_OK;
}

SparkStatus SparkK3StageRunnerResetSlots(
	SparkK3StageRunner *runner,
	const uint32_t *slots,
	uint32_t count)
{
	SparkK3RunnerState *state;
	if ( runner == 0 || runner->private_state == 0 || (count != 0u && slots == 0) )
		return SPARK_STATUS_INVALID_ARGUMENT;
	state = (SparkK3RunnerState *)runner->private_state;
	for ( uint32_t i = 0u; i < count; ++i )
	{
		int32_t status = SparkK3DispatchResetSlot(&state->dispatch, slots[i],
			runner->tp_degree, state->stream);
		if ( status == SPARK_K3_DISPATCH_ERR_ARGUMENT )
			return SPARK_STATUS_INVALID_ARGUMENT;
		if ( status != SPARK_K3_DISPATCH_OK )
			return SPARK_STATUS_IO_ERROR;
	}
	if ( cudaStreamSynchronize(state->stream) != cudaSuccess )
		return SPARK_STATUS_IO_ERROR;
	return SPARK_STATUS_OK;
}

SparkStatus SparkK3StageRunnerGetStats(
	const SparkK3StageRunner *runner,
	SparkK3StageRunnerStats *stats_out)
{
	if ( runner == 0 || stats_out == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	*stats_out = runner->stats;
	return SPARK_STATUS_OK;
}

const void *SparkK3StageRunnerProbeBuffers(const SparkK3StageRunner *runner)
{
	if ( runner == 0 || runner->private_state == 0 )
		return 0;
	return ((SparkK3RunnerState *)runner->private_state)->dispatch.buffers;
}

void SparkK3StageRunnerDestroy(SparkK3StageRunner *runner)
{
	SparkK3RunnerState *state;
	if ( runner == 0 || runner->private_state == 0 )
		return;
	state = (SparkK3RunnerState *)runner->private_state;
	/* the M3 stray receipt prints on every teardown path, including
	 * partial ones (early returns below) — the counters are the point */
	SparkK3RunnerStrayReport(state);
	free(state->stray_head_bits);
	free(state->stray_seen_bits);
	state->stray_head_bits = 0;
	state->stray_seen_bits = 0;
	if ( state->device_collective_created != 0 )
	{
		SparkTpDeviceCollectiveDestroy(&state->device_collective);
		if ( state->device_collective.implementation != 0 )
			return;
		state->device_collective_created = 0;
	}
	if ( state->device_collective_wide_created != 0 )
	{
		SparkTpDeviceCollectiveDestroy(&state->device_collective_wide);
		if ( state->device_collective_wide.implementation != 0 )
			return;
		state->device_collective_wide_created = 0;
	}
	if ( SparkK3RunnerReleaseLease(state) != SPARK_STATUS_OK )
		return;
	if ( state->lazy_pack != 0 )
	{
		if ( SparkWeightdLazyPackDestroy(state->lazy_pack) != SPARK_STATUS_OK )
			return;
		state->lazy_pack = 0;
	}
	if ( state->collective_created != 0 )
		SparkTpCollectiveDestroy(&state->collective);
	cudaFree(state->fused_device);
	SparkK3DispatchDestroy(&state->dispatch);
	SparkK3ModuleDestroy(&state->module);
	delete[] state->staging_values;
	free(state->group_offset_host);
	delete[] state->staging_scratch;
	delete[] state->head_slots_host;
	delete[] state->output_token_host;
	delete[] state->output_score_host;
	cudaFree(state->head_slots_device);
	cudaFree(state->head_maxloc);
	cudaFree(state->route_expert);
	cudaFree(state->route_packed_row);
	cudaFree(state->route_source_token);
	cudaFree(state->route_weight);
	cudaFree(state->group_row_offset);
	cudaFree(state->group_tile_prefix_w1);
	cudaFree(state->group_tile_prefix_w2);
	cudaFree(state->dense_row_offset);
	cudaFree(state->dense_tile_prefix);
	cudaFree(state->head_candidate_token);
	cudaFree(state->head_candidate_score);
	cudaFree(state->output_token);
	cudaFree(state->output_score);
	cudaFree(state->positions);
	cudaFree(state->token_ids_device);
	cudaFree(state->context_length);
	cudaFree(state->sequence_of_row);
	cudaFree(state->kda_state_index);
	delete state;
	runner->private_state = 0;
}

extern "C" int32_t K3StageSliceHalf(const void *layer_weights, const void *slice_state, void *layer_buffers, uint32_t layer, uint32_t phase, uint32_t rows, uint32_t sequences, uint32_t commit, uint32_t packed_rows, uint32_t context, uint32_t multiprocessors, void *stream);

SparkStatus SparkK3StageRunnerStepHalf(SparkK3StageRunner *runner, uint32_t layer, uint32_t phase,
	const void *hidden_input_bf16, const void *partial_input_bf16, void *partial_output_bf16)
{
	SparkK3RunnerState *state;
	SparkK3Dispatch *d;
	K3LayerBuffers *b;
	cudaStream_t stream;
	uint32_t rows, sequences, packed_rows;
	int32_t status;
	if ( runner == 0 || runner->private_state == 0 || partial_output_bf16 == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	state = (SparkK3RunnerState *)runner->private_state;
	d = &state->dispatch;
	if ( layer < d->first_layer || layer >= d->first_layer + d->layer_count )
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ( phase == 0u && hidden_input_bf16 == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	b = d->buffers;
	if ( phase == 2u && b->tp_sharded != 0u )
		return SPARK_STATUS_UNSUPPORTED;
	stream = state->stream;
	state->logical_sequence_count = 1u;
	rows = 1u;
	sequences = 1u;
	packed_rows = rows * K3_TOP_K;
	if ( phase == 0u && K3RunnerCopy(b->hidden_bf16, hidden_input_bf16,
			(uint64_t)rows * K3_HIDDEN * sizeof(uint16_t), stream) != cudaSuccess )
		state->copy_failed = 1u;
	if ( phase == 2u )
	{
		if ( K3RunnerCopy(b->gate_up_bf16, partial_input_bf16,
			(uint64_t)packed_rows * (2u * K3_EXPERT_INTERMEDIATE) * sizeof(uint16_t), stream) != cudaSuccess )
			state->copy_failed = 1u;
	}
	else if ( partial_input_bf16 != 0 &&
		K3RunnerCopy(b->attnres_partial_bf16, partial_input_bf16,
			(uint64_t)rows * K3_HIDDEN * sizeof(uint16_t), stream) != cudaSuccess )
			state->copy_failed = 1u;
	b->dense_row_offset = state->dense_row_offset;
	K3RunnerDenseOffsetsKernel<<<1u, 1u, 0, stream>>>(state->dense_row_offset, rows);
	b->dense_tile_prefix = state->dense_tile_prefix;
	b->group_row_offset = state->group_row_offset;
	b->group_tile_prefix_w1 = state->group_tile_prefix_w1;
	b->group_tile_prefix_w2 = state->group_tile_prefix_w2;
	b->route_expert = state->route_expert;
	b->route_packed_row = state->route_packed_row;
	b->route_source_token = state->route_source_token;
	b->route_weight = state->route_weight;
	b->sequence_row_begin = 0;
	b->positions = state->positions;
	b->context_length = state->context_length;
	b->sequence_of_row = state->sequence_of_row;
	b->kda_state_index = state->kda_state_index;
	if ( state->copy_failed != 0u )
		{ state->copy_failed = 0u; return SPARK_STATUS_IO_ERROR; }
	status = K3StageSliceHalf(d->weights + (layer - d->first_layer), d->slice_state, d->buffers,
		layer, phase, rows, sequences, 1u, packed_rows, rows, state->multiprocessors, stream);
	if ( status != LM_LAUNCH_OK )
	{
		fprintf(stderr, "sparkpipe_k3: half step layer %u phase %u -> %d\n",
			layer, phase, status);
		return SPARK_STATUS_INTERNAL_ERROR;
	}
	{
		SparkStatus failure = K3RunnerTakeFailure(state);
		if ( failure != SPARK_STATUS_OK )
			return failure;
	}
	if ( phase == 1u && layer >= K3_FIRST_ROUTED_LAYER )
	{
		if ( K3RunnerCopy(partial_output_bf16, b->gate_up_bf16,
			(uint64_t)packed_rows * (2u * K3_EXPERT_INTERMEDIATE) * sizeof(uint16_t), stream) != cudaSuccess )
			return SPARK_STATUS_IO_ERROR;
		return SPARK_STATUS_OK;
	}
	const uint16_t *phase0_source = (phase == 0u &&
		K3_LAYER_KIND(layer) == LM_LAYER_LATENT)
		? b->attention_out_bf16 : b->hidden_bf16;
	if ( K3RunnerCopy(partial_output_bf16, phase0_source,
		(uint64_t)rows * K3_HIDDEN * sizeof(uint16_t), stream) != cudaSuccess )
		return SPARK_STATUS_IO_ERROR;
	if ( phase == 2u )
		LM_LAUNCH((LmAddRowsKernel<K3_LAYER_THREADS>),
			dim3((K3_HIDDEN + K3_LAYER_THREADS - 1u) / K3_LAYER_THREADS,rows),
			K3_LAYER_THREADS, 0, stream,
			(uint16_t *)partial_output_bf16,b->shared_out_bf16,
			(uint16_t *)partial_output_bf16,rows,K3_HIDDEN);
	return SPARK_STATUS_OK;
}
