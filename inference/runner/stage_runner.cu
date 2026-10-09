
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <ctime>

#include "sparkpipe/spark_head_screen.h"
#include "sparkpipe/spark_stage_runner.h"
#include "sparkpipe/spark_stage_runner_model.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_cxx.h"
#include "sparkpipe/spark_weightd_manifest.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_tp_mesh_register.h"
#include "inference/kernels/sample.cuh"
#include "inference/kernels/stage_head.cuh"

#define STAGE_RUNNER_SAMPLE_THREADS 1024u
#define STAGE_RUNNER_DISTRIBUTION_HEAD_ROWS 4u
#define STAGE_RUNNER_GRAPH_ROWS 16u
#define STAGE_RUNNER_WIDE_ROWS 16u

typedef struct SparkStageRunnerState SparkStageRunnerState;

#define STAGE_RUNNER_GATHER_SOURCES_MAX 16u

typedef struct StageRunnerRankSources
{
	const uint16_t *rank[STAGE_RUNNER_GATHER_SOURCES_MAX];
} StageRunnerRankSources;

__global__ static void StageRunnerCombineTp4TreeKernel(StageRunnerRankSources sources,
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

__global__ static void StageRunnerCombinePairKernel(const uint16_t *source,
	uint16_t *destination,uint32_t elements)
{
	uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if ( i >= elements )
		return;
	destination[i] = LmFloatToBf16(LmBf16ToFloat(destination[i]) + LmBf16ToFloat(source[i]));
}

__global__ static void StageRunnerGatherStripesKernel(StageRunnerRankSources sources,
	uint16_t *destination,uint32_t rank_count,uint32_t elements_per_rank)
{
	uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
	uint32_t total = rank_count * elements_per_rank;
	if ( i >= total )
		return;
	destination[i] = sources.rank[i / elements_per_rank][i % elements_per_rank];
}

static SparkStatus StageRunnerCombineGatherBf16(void *combine_context,
	void *destination_device,const void *const *source_devices,
	uint32_t source_count,uint32_t active_sequence_count,
	uint32_t hidden_dimension,void *cuda_stream)
{
	StageRunnerRankSources sources;
	uint32_t elements_per_rank = active_sequence_count * hidden_dimension;
	(void)combine_context;
	if ( destination_device == 0 || source_devices == 0 || source_count == 0u ||
		source_count > STAGE_RUNNER_GATHER_SOURCES_MAX || elements_per_rank == 0u )
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(&sources, 0, sizeof(sources));
	for ( uint32_t r = 0u; r < source_count; ++r )
	{
		if ( source_devices[r] == 0 )
			return SPARK_STATUS_INVALID_ARGUMENT;
		sources.rank[r] = (const uint16_t *)source_devices[r];
	}
	StageRunnerGatherStripesKernel<<<(source_count * elements_per_rank + 255u) / 256u,
		256u, 0, (cudaStream_t)cuda_stream>>>(
		sources,(uint16_t *)destination_device,source_count,elements_per_rank);
	return cudaGetLastError() == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus StageRunnerCombineSumRanksF32(void *combine_context,
	void *destination_device,const void *const *source_devices,
	uint32_t source_count,uint32_t active_sequence_count,
	uint32_t hidden_dimension,void *cuda_stream)
{
	uint64_t elements = (uint64_t)active_sequence_count * hidden_dimension;
	(void)combine_context;
	if ( destination_device == 0 || source_devices == 0 || source_count == 0u ||
		source_count > STAGE_RUNNER_GATHER_SOURCES_MAX || elements == 0u ||
		elements > UINT32_MAX )
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SparkTpLaunchSumRanksF32((cudaStream_t)cuda_stream,
		destination_device,source_devices,source_count,(uint32_t)elements) ==
		cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus StageRunnerCombineBf16(void *combine_context,
	void *destination_device,const void *source_device,
	uint32_t active_sequence_count,uint32_t hidden_dimension,void *cuda_stream)
{
	uint32_t elements = active_sequence_count * hidden_dimension;
	(void)combine_context;
	if ( destination_device == 0 || source_device == 0 || elements == 0u )
		return SPARK_STATUS_INVALID_ARGUMENT;
	StageRunnerCombinePairKernel<<<(elements + 255u) / 256u,
		256u, 0, (cudaStream_t)cuda_stream>>>(
		(const uint16_t *)source_device,(uint16_t *)destination_device,elements);
	return cudaGetLastError() == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus StageRunnerCombineTp4Bf16(void *combine_context,
	void *destination_device,const void *const rank_devices[4],uint32_t tp_rank,
	uint32_t active_sequence_count,uint32_t hidden_dimension,void *cuda_stream)
{
	StageRunnerRankSources sources;
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
	StageRunnerCombineTp4TreeKernel<<<(elements + 255u) / 256u,
		256u, 0, (cudaStream_t)cuda_stream>>>(
		sources,(uint16_t *)destination_device,active_sequence_count,hidden_dimension);
	return cudaGetLastError() == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

__global__ static void StageRunnerCombineU64MaxKernel(const uint64_t *source,
	uint64_t *destination,uint32_t elements)
{
	uint32_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if ( i >= elements )
		return;
	if ( source[i] > destination[i] )
		destination[i] = source[i];
}

static SparkStatus StageRunnerCombineU64Max(void *combine_context,
	uint64_t *destination_device,const uint64_t *source_device,
	uint32_t element_count,void *cuda_stream)
{
	(void)combine_context;
	if ( destination_device == 0 || source_device == 0 || element_count == 0u )
		return SPARK_STATUS_INVALID_ARGUMENT;
	StageRunnerCombineU64MaxKernel<<<(element_count + 255u) / 256u,
		256u, 0, (cudaStream_t)cuda_stream>>>(
		source_device,destination_device,element_count);
	return cudaGetLastError() == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

enum
{
	SPARK_STAGE_LEASE_ACQUIRED = 1u,
	SPARK_STAGE_LEASE_BEGUN = 2u,
	SPARK_STAGE_LEASE_RECORDED = 3u
};

typedef struct SparkStageRunnerState
{
	const SparkStageRunnerModelInterface *model_interface;
	void *model;
	SparkStageRunnerModelGeometry geometry;
	SparkStageRunnerServices services;
	SparkTpDeviceCollective device_collective;
	int device_collective_created;
	uint32_t device_collective_deferred;
	SparkWeightdLazyPack *lazy_pack;
	uint32_t lease_slots;
	uint64_t *lease_identifier;
	uint32_t *lease_phase;
	SparkWeightdExpertKey *lease_keys;
	void *lease_address;
	const void *resident_base;
	uint32_t resident;
	uint32_t *group_offset_host;
	uint32_t tp_rank;
	uint32_t tp_degree;
	uint64_t tp_next_ordinal;
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
	uint64_t *head_maxloc;
	uint32_t head_last_rows;
	uint16_t *hidden;
	uint16_t *head_normed;
	uint16_t *head_hidden;
	uint32_t *head_token;
	float *head_score;
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
	uint32_t *recurrent_index;
	uint8_t *stray_head_bits;
	uint8_t *stray_seen_bits;
	uint64_t stray_selections;
	uint64_t stray_count;
	uint32_t stray_head_keys;
	uint64_t stray_bit_bytes;
	cudaStream_t stream;
	uint32_t max_rows;
	uint32_t max_context;
	uint32_t multiprocessors;
	uint64_t kv_page_bytes;
	uint32_t distribution_capacity;
	uint32_t distribution_sub_rows;
	uint32_t distribution_chunk_rows;
	uint32_t *distribution_rows;
	uint32_t *distribution_positions;
	SparkRowSampling *distribution_rules;
	SparkSamplingLogprob *distribution_logprobs;
	uint16_t *distribution_hidden;
	uint16_t *distribution_normed;
	float *distribution_logits;
	float *distribution_gathered;
	cudaGraphExec_t graph_exec[STAGE_RUNNER_GRAPH_ROWS];
	SparkStageRunnerStep graph_input[STAGE_RUNNER_GRAPH_ROWS];
	uint8_t graph_refused[STAGE_RUNNER_GRAPH_ROWS];
	uint64_t graph_launches;
	uint64_t timing_steps;
	uint64_t timing_graph_steps;
	uint64_t timing_submit_ns;
	uint64_t phase_ns[4];
	uint32_t phase_waves;
	uint32_t mark_count;
	uint32_t kv_attached;
	cudaEvent_t *layer_event;
	uint64_t *layer_host_ns;
	double *layer_gpu_ms;
	double *layer_enqueue_ms;
	double *layer_lag_ms;
	double layer_span_ms[2];
	uint32_t layer_events_created;
	uint32_t layer_marks;
	uint32_t layer_mark_last;
	uint32_t layer_waves;
	uint64_t timing_graph_ns;
	SparkTpDeviceCollectiveHardwareTiming timing_wait;
} SparkStageRunnerState;

static SparkStatus StageRunnerAllocateDistribution(SparkStageRunnerState *state,
	const SparkStageRunner *runner, const SparkStageRunnerConfiguration *configuration);
static SparkStatus StageRunnerSeedIndices(SparkStageRunnerState *state,
	const SparkStageRunnerConfiguration *configuration);

static SparkStatus StageRunnerPrepareOutputs(SparkStageRunnerState *state,
	const SparkStageRunner *runner, const SparkStageRunnerConfiguration *configuration)
{
	SparkStatus status = StageRunnerSeedIndices(state, configuration);
	return status != SPARK_STATUS_OK ? status : StageRunnerAllocateDistribution(state, runner, configuration);
}

static uint64_t StageRunnerStrayBit(const SparkStageRunnerState *state, uint32_t layer, uint32_t expert)
{
	return (uint64_t)layer * state->geometry.experts + (uint64_t)expert;
}

static void SparkStageRunnerStrayAccount(
	SparkStageRunnerState *state,
	const SparkWeightdExpertKey *keys, uint32_t count)
{
	uint32_t index;
	if ( state == 0 || state->stray_head_bits == 0 || keys == 0 )
		return;
	for ( index = 0u; index < count; ++index )
	{
		uint64_t bit;
		uint8_t mask;
		if ( keys[index].layer >= state->geometry.total_layers ||
			keys[index].expert >= state->geometry.experts )
			continue;
		bit = StageRunnerStrayBit(state, keys[index].layer, keys[index].expert);
		mask = (uint8_t)(1u << (bit & 7u));
		state->stray_selections++;
		if ( (state->stray_head_bits[bit >> 3] & mask) == 0u )
		{
			state->stray_count++;
			state->stray_seen_bits[bit >> 3] |= mask;
		}
	}
}

static SparkStatus SparkStageRunnerStrayLoad(SparkStageRunnerState *state, const char *path)
{
	FILE *input;
	uint32_t pair[2];
	uint32_t loaded = 0u;
	if ( path == 0 || path[0] == '\0' )
		return SPARK_STATUS_OK;
	if ( state->geometry.experts == 0u )
	{
		fprintf(stderr, "sparkpipe_stage_runner: a stray working set needs a routed-expert model\n");
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	state->stray_bit_bytes = (StageRunnerStrayBit(state, state->geometry.total_layers, 0u) + 7u) / 8u;
	state->stray_head_bits = (uint8_t *)calloc(state->stray_bit_bytes, 1u);
	state->stray_seen_bits = (uint8_t *)calloc(state->stray_bit_bytes, 1u);
	input = state->stray_head_bits != 0 && state->stray_seen_bits != 0 ? fopen(path, "rb") : 0;
	if ( input == 0 )
	{
		fprintf(stderr, "sparkpipe_stage_runner: stray working set %s cannot be loaded\n", path);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	while ( fread(pair, 1u, sizeof(pair), input) == sizeof(pair) )
	{
		uint64_t bit;
		if ( pair[0] >= state->geometry.total_layers || pair[1] >= state->geometry.experts )
			continue;
		bit = StageRunnerStrayBit(state, pair[0], pair[1]);
		state->stray_head_bits[bit >> 3] |= (uint8_t)(1u << (bit & 7u));
		loaded++;
	}
	(void)fclose(input);
	state->stray_head_keys = loaded;
	fprintf(stderr, "sparkpipe_stage_runner: stray accounting armed head_keys=%u wset=%s\n", loaded, path);
	return SPARK_STATUS_OK;
}

static void SparkStageRunnerStrayReport(const SparkStageRunnerState *state)
{
	uint64_t unique = 0u;
	uint32_t index;
	if ( state == 0 || state->stray_head_bits == 0 )
		return;
	if ( state->stray_seen_bits != 0 )
		for ( index = 0u; index < state->stray_bit_bytes; ++index )
		{
			uint8_t word = state->stray_seen_bits[index];
			while ( word != 0u )
			{
				unique += word & 1u;
				word >>= 1;
			}
		}
	fprintf(stderr, "STAGE-STRAY-RECEIPT selections=%llu strays=%llu "
		"stray_rate=%.4f unique_stray_pairs=%llu head_keys=%u\n",
		(unsigned long long)state->stray_selections,
		(unsigned long long)state->stray_count,
		state->stray_selections != 0u ?
			(double)state->stray_count / (double)state->stray_selections : 0.0,
		(unsigned long long)unique, state->stray_head_keys);
}

static void StageRunnerEmbedCompletion(void *context,
	const SparkTpDeviceCollectiveCompletion *completion)
{
	(void)context;
	(void)completion;
}

static cudaError_t StageRunnerCopy(void *destination, const void *source,
	uint64_t bytes, cudaStream_t stream)
{
	cudaError_t error = cudaMemcpyAsync(destination, source, (size_t)bytes,
		cudaMemcpyDefault, stream);
	if ( error != cudaSuccess )
		return error;
	return cudaStreamSynchronize(stream);
}

static void StageRunnerSubmissionInit(SparkTpDeviceCollectiveSubmission *submission,
	uint32_t deferred, const SparkStageRunnerState *state, cudaStream_t stream,
	uint32_t active, const void *local, void *full, uint64_t ordinal)
{
	memset(submission, 0, sizeof(*submission));
	submission->abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission->descriptor_bytes = sizeof(*submission);
	submission->active_sequence_count = active;
	submission->logical_sequence_count = state->logical_sequence_count;
	submission->flags = SPARK_TP_DEVICE_COLLECTIVE_SUBMISSION_STREAM_ORDERED_COMPLETION;
	submission->ordinal = ordinal;
	submission->local_device = local;
	submission->full_device = full;
	submission->cuda_stream = stream;
	submission->completion_function = deferred != 0u ? 0 : StageRunnerEmbedCompletion;
}

static SparkStatus StageRunnerVerifyCollectives(SparkStageRunnerState *state, cudaStream_t stream)
{
	SparkStatus status = SPARK_STATUS_OK;
	if ( state->device_collective_created != 0 )
		status = SparkTpDeviceCollectiveVerifyDeferred(&state->device_collective, stream);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr, "sparkpipe_stage_runner: deferred collective rounds failed status=%d\n", (int)status);
	return status;
}

static SparkStatus StageRunnerTakeFailure(SparkStageRunnerState *state)
{
	SparkStatus status = SPARK_STATUS_OK;
	if ( state->copy_failed != 0u )
	{
		fprintf(stderr, "sparkpipe_stage_runner: a stream-ordered copy failed\n");
		status = SPARK_STATUS_IO_ERROR;
	}
	else if ( state->tp_collective_failed != 0u )
	{
		fprintf(stderr, "sparkpipe_stage_runner: a tensor-parallel collective failed\n");
		status = SPARK_STATUS_INTERNAL_ERROR;
	}
	state->copy_failed = 0u;
	state->tp_collective_failed = 0u;
	return status;
}

static SparkStatus StageRunnerReduceBf16(SparkStageRunnerState *state, cudaStream_t stream,
	const uint16_t *device_values, uint32_t rows)
{
	SparkTpDeviceCollectiveSubmission submission;
	if ( state->device_collective_created == 0 )
		return state->tp_degree > 1u ? SPARK_STATUS_INTERNAL_ERROR : SPARK_STATUS_OK;
	StageRunnerSubmissionInit(&submission, state->device_collective_deferred, state, stream,
		rows, device_values, (void *)device_values, state->tp_next_ordinal++);
	return SparkTpDeviceCollectiveEnqueue(&state->device_collective, &submission,
		SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16);
}

static SparkStatus StageRunnerRoundService(void *context, void *stream_void, uint32_t operation, uint32_t stage,
	uint32_t rows, uint32_t row_elements, const void *local, void *full)
{
	SparkStageRunnerState *state = (SparkStageRunnerState *)context;
	SparkTpDeviceCollectiveSubmission submission;
	SparkStatus status;
	if ( state->device_collective_created == 0 )
	{
		state->tp_collective_failed = 1u;
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	if ( stage == SPARK_STAGE_RUNNER_ROUND_FINISH )
		status = SparkTpDeviceCollectiveFinish(&state->device_collective);
	else
	{
		StageRunnerSubmissionInit(&submission, state->device_collective_deferred, state, (cudaStream_t)stream_void,
			rows, local, full, state->tp_next_ordinal++);
		submission.row_elements = row_elements;
		status = stage == SPARK_STAGE_RUNNER_ROUND_BEGIN
			? SparkTpDeviceCollectiveBegin(&state->device_collective, &submission, operation)
			: SparkTpDeviceCollectiveEnqueue(&state->device_collective, &submission, operation);
	}
	if ( status != SPARK_STATUS_OK )
		state->tp_collective_failed = 1u;
	return status;
}

static uint32_t StageRunnerPublishedService(void *context)
{
	SparkStageRunnerState *state = (SparkStageRunnerState *)context;
	return state->device_collective_created != 0 ? SparkTpDeviceCollectivePublished(&state->device_collective) : 0u;
}

static void StageRunnerFailService(void *context, SparkStatus status)
{
	SparkStageRunnerState *state = (SparkStageRunnerState *)context;
	if ( status == SPARK_STATUS_IO_ERROR )
		state->copy_failed = 1u;
	else
		state->tp_collective_failed = 1u;
}

static SparkStatus SparkStageRunnerReleaseOneLease(SparkStageRunnerState *state, uint32_t index)
{
	SparkStatus status;
	if ( state->lease_identifier[index] == 0u )
		return(state->lease_phase[index] == 0u ? SPARK_STATUS_OK : SPARK_STATUS_VALIDATION_FAILED);
	if ( state->lazy_pack == 0 || state->lazy_pack->map == 0 ||
		state->lease_phase[index] < SPARK_STAGE_LEASE_ACQUIRED || state->lease_phase[index] > SPARK_STAGE_LEASE_RECORDED )
		return(SPARK_STATUS_VALIDATION_FAILED);
	if ( state->lease_phase[index] == SPARK_STAGE_LEASE_BEGUN )
	{
		status = SparkWeightdMapRecordCompletion(state->lazy_pack->map,state->lease_identifier[index],state->stream);
		if ( status != SPARK_STATUS_OK )
			return(status);
		state->lease_phase[index] = SPARK_STAGE_LEASE_RECORDED;
	}
	status = SparkWeightdMapRelease(state->lazy_pack->map,state->lease_identifier[index],SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS);
	if ( status == SPARK_STATUS_OK )
	{
		state->lease_identifier[index] = 0u;
		state->lease_phase[index] = 0u;
	}
	return(status);
}

static SparkStatus SparkStageRunnerReleaseLease(SparkStageRunnerState *state)
{
	SparkStatus status,result = SPARK_STATUS_OK;
	uint32_t index;
	for (index=0u; index<state->lease_slots; index++)
	{
		status = SparkStageRunnerReleaseOneLease(state, index);
		if ( status != SPARK_STATUS_OK && result == SPARK_STATUS_OK )
			result = status;
	}
	if ( result == SPARK_STATUS_OK )
		state->lease_address = 0;
	return(result);
}

static SparkStatus StageRunnerLeaseTensorBase(SparkStageRunnerState *state,
	uint32_t layer, SparkWeightdExpertKey *keys, uint32_t *count)
{
	if ( state->geometry.lease_tensor_base == 0u || (*count != 0u && keys[0].expert == 0u) )
		return SPARK_STATUS_OK;
	if ( *count >= state->geometry.experts + 1u )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	memmove(&keys[1], &keys[0], (size_t)*count * sizeof(keys[0]));
	keys[0].layer = layer;
	keys[0].expert = 0u;
	(*count)++;
	return SPARK_STATUS_OK;
}

static SparkStatus StageRunnerExpertWeightsService(void *context, uint32_t layer,
	const uint32_t *group_row_offset_device, uint32_t routed_rows, const void **w1, const void **w2)
{
	SparkStageRunnerState *state = (SparkStageRunnerState *)context;
	SparkWeightdMap *map;
	SparkWeightdExpertKey *keys;
	uint32_t count = 0u,first,chunk,index;
	cudaError_t error;
	SparkStatus status;
	if ( state == 0 || state->lazy_pack == 0 || w1 == 0 || w2 == 0 || group_row_offset_device == 0 ||
		state->geometry.experts == 0u || layer >= state->geometry.total_layers )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	map = state->lazy_pack->map;
	if ( map == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->resident != 0u )
	{
		*w1 = (const uint8_t *)state->resident_base + state->geometry.expert_w1_offset[layer];
		*w2 = (const uint8_t *)state->resident_base + state->geometry.expert_w2_offset[layer];
		return SPARK_STATUS_OK;
	}
	error = cudaStreamSynchronize(state->stream);
	if ( error != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	status = SparkStageRunnerReleaseLease(state);
	if ( status != SPARK_STATUS_OK )
		return status;
	error = cudaMemcpy(state->group_offset_host, group_row_offset_device,
		(state->geometry.experts + 1u) * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	if ( error != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	keys = state->lease_keys;
	status = SparkWeightdRouteKeys(layer, state->group_offset_host,
		state->geometry.experts, routed_rows, keys, state->geometry.experts, &count);
	if ( status != SPARK_STATUS_OK )
		return status;
	status = StageRunnerLeaseTensorBase(state, layer, keys, &count);
	if ( status != SPARK_STATUS_OK )
		return status;
	SparkStageRunnerStrayAccount(state, keys, count);
	for (first=0u,index=0u; first<count; first+=chunk,index++)
	{
		chunk = count - first < SPARK_WEIGHTD_LEASE_GROUPS_MAX ? count - first : SPARK_WEIGHTD_LEASE_GROUPS_MAX;
		if ( index >= state->lease_slots )
		{
			(void)SparkStageRunnerReleaseLease(state);
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		status = SparkWeightdMapAcquire(map, keys + first, chunk,
			&state->lease_identifier[index], SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS);
		if ( state->lease_identifier[index] != 0u )
			state->lease_phase[index] = SPARK_STAGE_LEASE_ACQUIRED;
		if ( status == SPARK_STATUS_OK && state->lease_identifier[index] == 0u )
			status = SPARK_STATUS_VALIDATION_FAILED;
		if ( status == SPARK_STATUS_OK )
			status = SparkWeightdMapBeginUse(map, state->lease_identifier[index], &state->lease_address);
		if ( status == SPARK_STATUS_OK )
			state->lease_phase[index] = SPARK_STAGE_LEASE_BEGUN;
		if ( status != SPARK_STATUS_OK )
		{
			(void)SparkStageRunnerReleaseLease(state);
			SPARK_FAIL(status);
		}
	}
	*w1 = (const uint8_t *)state->lease_address + state->geometry.expert_w1_offset[layer];
	*w2 = (const uint8_t *)state->lease_address + state->geometry.expert_w2_offset[layer];
	return SPARK_STATUS_OK;
}

static uint64_t StageRunnerNowNs(void);

static void StageRunnerLayerMark(SparkStageRunnerState *state, uint32_t index)
{
	if ( state->layer_marks == 0u || index >= state->mark_count )
		return;
	if ( cudaEventRecord(state->layer_event[index], state->stream) != cudaSuccess )
	{
		state->layer_marks = 0u;
		return;
	}
	state->layer_host_ns[index] = StageRunnerNowNs();
	state->layer_mark_last = index;
}

static void StageRunnerLayerDoneService(void *context, uint32_t layer)
{
	SparkStageRunnerState *state = (SparkStageRunnerState *)context;
	SparkStatus status;
	if ( state != 0 && layer < state->geometry.total_layers )
		StageRunnerLayerMark(state, layer + 1u);
	if ( state == 0 || state->resident != 0u || state->lease_slots == 0u )
		return;
	status = SparkStageRunnerReleaseLease(state);
	if ( status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY )
		fprintf(stderr,"sparkpipe_stage_runner: lease release failed status=%d (retained for recovery)\n",(int)status);
}

static SparkStatus StageRunnerSeedIndices(SparkStageRunnerState *state,
	const SparkStageRunnerConfiguration *configuration)
{
	const uint32_t zero = 0u, one = 1u;
	if ( configuration->resident_sequence_capacity > configuration->max_active_sequence_count )
	{
		fprintf(stderr, "sparkpipe_stage_runner: resident_sequence_capacity %u exceeds the %u recurrent state slots\n",
			configuration->resident_sequence_capacity, configuration->max_active_sequence_count);
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	if ( StageRunnerCopy(state->positions, &zero, sizeof(uint32_t), state->stream) != cudaSuccess ||
		StageRunnerCopy(state->context_length, &one, sizeof(uint32_t), state->stream) != cudaSuccess ||
		StageRunnerCopy(state->sequence_of_row, &zero, sizeof(uint32_t), state->stream) != cudaSuccess ||
		StageRunnerCopy(state->recurrent_index, &zero, sizeof(uint32_t), state->stream) != cudaSuccess )
		return SPARK_STATUS_IO_ERROR;
	return SPARK_STATUS_OK;
}

static SparkStatus StageRunnerManifestCheck(const SparkWeightdManifest *manifest, void *context)
{
	SparkStageRunnerState *state = (SparkStageRunnerState *)context;
	return state->model_interface->manifest_check != 0 ? state->model_interface->manifest_check(state->model, manifest) : SPARK_STATUS_OK;
}

static SparkStatus StageRunnerSlice(SparkStageRunnerState *state, uint64_t offset, uint64_t bytes, const uint16_t **out)
{
	const void *slice = 0;
	if ( bytes == 0u || SparkWeightdLazyPackSlice(state->lazy_pack, offset, bytes, &slice) != SPARK_STATUS_OK || slice == 0 )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	*out = (const uint16_t *)slice;
	return SPARK_STATUS_OK;
}

static SparkStatus StageRunnerAttachWeights(SparkStageRunnerState *state, const SparkStageRunnerConfiguration *configuration)
{
	SparkWeightdLazyAttachRequest request;
	SparkStatus status = SparkWeightdAttachRequested();
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr, "sparkpipe_stage_runner: weightd attach not granted status=%d (fail-closed, no direct load)\n", (int)status);
		SPARK_FAIL(status);
	}
	memset(&request, 0, sizeof(request));
	if ( strlen(configuration->weights.pack_sha256) != SPARK_STAGE_RUNNER_DIGEST_HEX ||
		strlen(configuration->rank_pack_path) >= sizeof(request.pack_path) ||
		configuration->weights.socket_path == 0 || configuration->weights.expert_pool_bytes == 0u ||
		configuration->weights.spine_budget_bytes == 0u )
	{
		fprintf(stderr, "sparkpipe_stage_runner: weightd attach needs the pack digest, socket, expert pool and spine budget\n");
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	memcpy(request.identity.pack_sha256, configuration->weights.pack_sha256, SPARK_STAGE_RUNNER_DIGEST_HEX);
	snprintf(request.identity.model, sizeof(request.identity.model), "%s", state->model_interface->weightd_model);
	snprintf(request.identity.revision, sizeof(request.identity.revision), "%s", state->model_interface->weightd_revision);
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = state->geometry.pack_bytes;
	request.identity.topology = configuration->tp_degree;
	memcpy(request.pack_path, configuration->rank_pack_path, strlen(configuration->rank_pack_path) + 1u);
	request.expert_pool_bytes = configuration->weights.expert_pool_bytes;
	status = SparkWeightdLazyPackCreateChecked(configuration->weights.socket_path, &request,
		configuration->weights.spine_budget_bytes, SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS,
		StageRunnerManifestCheck, state, &state->lazy_pack);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr, "sparkpipe_stage_runner: lazy startup failed status=%d (fail-closed, no eager fallback)\n", (int)status);
	return status;
}

static SparkStatus StageRunnerCreateCollective(SparkStageRunnerState *state, const SparkStageRunnerConfiguration *configuration)
{
	SparkTpDeviceCollectiveConfig device_config;
	SparkStatus status;
	if ( configuration->device_collective == 0 )
		return configuration->tp_degree > 1u ? SPARK_STATUS_INVALID_ARGUMENT : SPARK_STATUS_OK;
	if ( configuration->device_collective->local_hidden_dimension != state->geometry.hidden )
	{
		fprintf(stderr, "sparkpipe_stage_runner: device collective width %u != model hidden %u\n",
			configuration->device_collective->local_hidden_dimension, state->geometry.hidden);
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	}
	device_config = *configuration->device_collective;
	if ( device_config.backend_kind == SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT )
	{
		device_config.combine_bf16_function = StageRunnerCombineBf16;
		device_config.combine_tp4_bf16_function = StageRunnerCombineTp4Bf16;
		device_config.combine_u64_max_function = StageRunnerCombineU64Max;
		device_config.combine_gather_bf16_function = StageRunnerCombineGatherBf16;
		device_config.combine_fused_bf16_function = StageRunnerCombineSumRanksF32;
		device_config.combine_context = state;
	}
	status = SparkTpDeviceCollectiveCreate(&device_config, &state->device_collective);
	if ( status != SPARK_STATUS_OK )
		return status;
	state->device_collective_created = 1;
	state->device_collective_deferred = device_config.wait_mode == SPARK_TP_DEVICE_COLLECTIVE_WAIT_HARDWARE ? 1u : 0u;
	if ( state->lazy_pack != 0 && state->lazy_pack->attached.mesh_send_buffer_addr != 0 )
		status = SparkTpDeviceCollectivePrepareReceiveBf16(&state->device_collective,
			(void *)(uintptr_t)state->lazy_pack->attached.mesh_send_buffer_addr, 0u,0u,0u,0u);
	return status;
}

static SparkStatus StageRunnerCreateHead(SparkStageRunnerState *state, const SparkStageRunner *runner)
{
	const uint64_t shard_rows = state->vocab_slice_rows, dim = state->geometry.hidden;
	if ( runner->owns_embedding != 0u &&
		StageRunnerSlice(state, state->geometry.embed_offset, state->geometry.embed_bytes, &state->embed_weight) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	if ( runner->owns_final_head == 0u )
		return SPARK_STATUS_OK;
	if ( StageRunnerSlice(state, state->geometry.head_norm_offset, state->geometry.head_norm_bytes, &state->head_norm_weight) != SPARK_STATUS_OK ||
		StageRunnerSlice(state, state->geometry.head_offset, state->geometry.head_bytes, &state->head_weight) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	if ( cudaMalloc(&state->head_certified_fp8_payload, shard_rows * dim) != cudaSuccess ||
		cudaMalloc(&state->head_certified_fp8_scale_f32, shard_rows * (dim / 32u) * sizeof(float)) != cudaSuccess ||
		cudaMalloc(&state->head_certified_fp8_norm_f32, shard_rows * (dim / 32u) * sizeof(float)) != cudaSuccess ||
		cudaMalloc(&state->head_certified_scratch, SparkHeadCertifiedFp8ScratchBytes(shard_rows,dim) * state->max_rows) != cudaSuccess ||
		cudaMalloc(&state->head_certified_candidates, SparkHeadCertifiedFp8CandidateBytes(shard_rows) * state->max_rows) != cudaSuccess ||
		cudaMalloc(&state->head_screened_count, sizeof(uint32_t) * state->max_rows) != cudaSuccess )
	{
		fprintf(stderr, "sparkpipe_stage_runner: certified head buffers for %llu rows do not fit\n", (unsigned long long)shard_rows);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	if ( SparkLmHostLaunchHeadCertifiedFp8Quantize(0,state->head_weight,state->head_certified_fp8_payload,
			state->head_certified_fp8_scale_f32,state->head_certified_fp8_norm_f32,(uint32_t)shard_rows,(uint32_t)dim) != cudaSuccess ||
		cudaDeviceSynchronize() != cudaSuccess )
	{
		fprintf(stderr, "sparkpipe_stage_runner: certified head quantize failed rows=%llu\n", (unsigned long long)shard_rows);
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	}
	return SPARK_STATUS_OK;
}

static SparkStatus StageRunnerCreateBuffers(SparkStageRunnerState *state, const SparkStageRunner *runner,
	const SparkStageRunnerConfiguration *configuration)
{
	const uint64_t rows = configuration->max_input_row_count, hidden = state->geometry.hidden;
	SparkStatus status;
	state->head_tiles = (state->vocab_slice_rows + LM_STAGE_HEAD_TILE - 1u) / LM_STAGE_HEAD_TILE;
	if ( cudaMalloc(&state->hidden, rows * hidden * sizeof(uint16_t)) != cudaSuccess ||
		cudaMalloc(&state->head_normed, rows * hidden * sizeof(uint16_t)) != cudaSuccess ||
		cudaMalloc(&state->head_maxloc, rows * sizeof(uint64_t)) != cudaSuccess ||
		cudaMalloc(&state->head_candidate_token, rows * state->head_tiles * 4u) != cudaSuccess ||
		cudaMalloc(&state->head_candidate_score, rows * state->head_tiles * 4u) != cudaSuccess ||
		cudaMalloc(&state->output_token, rows * 4u) != cudaSuccess ||
		cudaMalloc(&state->output_score, rows * 4u) != cudaSuccess ||
		cudaMalloc(&state->head_hidden, rows * hidden * sizeof(uint16_t)) != cudaSuccess ||
		cudaMalloc(&state->head_token, rows * 4u) != cudaSuccess ||
		cudaMalloc(&state->head_score, rows * 4u) != cudaSuccess ||
		cudaMalloc(&state->positions, 4u) != cudaSuccess ||
		cudaMalloc(&state->token_ids_device, rows * sizeof(*state->token_ids_device)) != cudaSuccess ||
		cudaMalloc(&state->context_length, 4u) != cudaSuccess ||
		cudaMalloc(&state->sequence_of_row, 4u) != cudaSuccess ||
		cudaMalloc(&state->recurrent_index, 4u) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = StageRunnerPrepareOutputs(state, runner, configuration);
	if ( status != SPARK_STATUS_OK )
		return status;
	state->output_token_host = new uint32_t[rows];
	state->output_score_host = new float[rows];
	state->mark_count = state->geometry.total_layers + 1u;
	state->layer_event = (cudaEvent_t *)calloc(state->mark_count, sizeof(cudaEvent_t));
	state->layer_host_ns = (uint64_t *)calloc(state->mark_count, sizeof(uint64_t));
	state->layer_gpu_ms = (double *)calloc(state->mark_count, sizeof(double));
	state->layer_enqueue_ms = (double *)calloc(state->mark_count, sizeof(double));
	state->layer_lag_ms = (double *)calloc(state->mark_count, sizeof(double));
	if ( state->layer_event == 0 || state->layer_host_ns == 0 || state->layer_gpu_ms == 0 ||
		state->layer_enqueue_ms == 0 || state->layer_lag_ms == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	return SPARK_STATUS_OK;
}

static SparkStatus StageRunnerCreateLeases(SparkStageRunnerState *state)
{
	if ( state->geometry.experts == 0u )
		return SPARK_STATUS_OK;
	if ( state->geometry.expert_w1_offset == 0 || state->geometry.expert_w2_offset == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	state->lease_slots = (state->geometry.experts + 1u + SPARK_WEIGHTD_LEASE_GROUPS_MAX - 1u) / SPARK_WEIGHTD_LEASE_GROUPS_MAX;
	state->lease_identifier = (uint64_t *)calloc(state->lease_slots, sizeof(uint64_t));
	state->lease_phase = (uint32_t *)calloc(state->lease_slots, sizeof(uint32_t));
	state->lease_keys = (SparkWeightdExpertKey *)calloc(state->geometry.experts + 1u, sizeof(SparkWeightdExpertKey));
	state->group_offset_host = (uint32_t *)malloc((state->geometry.experts + 1u) * sizeof(uint32_t));
	if ( state->lease_identifier == 0 || state->lease_phase == 0 || state->lease_keys == 0 || state->group_offset_host == 0 )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	return SPARK_STATUS_OK;
}

SparkStatus SparkStageRunnerInitialize(SparkStageRunner *runner, const SparkStageRunnerConfiguration *configuration,
	const SparkStageRunnerModelInterface *model)
{
	SparkStageRunnerState *state;
	SparkStageRunnerModelOpen request;
	SparkStatus status;
	if ( runner == 0 || configuration == 0 || model == 0 || model->abi_version != SPARK_STAGE_RUNNER_MODEL_ABI_VERSION ||
		model->open == 0 || model->bind == 0 || model->close == 0 || model->step == 0 || model->weightd_model == 0 ||
		model->weightd_revision == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ( configuration->abi_version != SPARK_STAGE_RUNNER_ABI_VERSION || configuration->stage_count == 0u ||
		configuration->stage_index >= configuration->stage_count || configuration->rank_pack_path == 0 ||
		configuration->max_active_sequence_count == 0u || configuration->max_input_row_count == 0u ||
		(configuration->tp_degree > 1u && configuration->device_collective == 0) )
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(runner, 0, sizeof(*runner));
	state = new SparkStageRunnerState;
	memset(state, 0, sizeof(*state));
	runner->abi_version = SPARK_STAGE_RUNNER_ABI_VERSION;
	runner->descriptor_bytes = SPARK_STAGE_RUNNER_BYTES;
	runner->flags = configuration->flags;
	runner->stage_index = configuration->stage_index;
	runner->stage_count = configuration->stage_count;
	runner->tp_degree = configuration->tp_degree;
	runner->tp_rank = configuration->tp_rank;
	runner->owns_embedding = configuration->stage_index == 0u ? 1u : 0u;
	runner->owns_final_head = configuration->stage_index + 1u == configuration->stage_count ? 1u : 0u;
	runner->private_state = state;
	runner->stats.abi_version = SPARK_STAGE_RUNNER_ABI_VERSION;
	runner->stats.descriptor_bytes = (uint32_t)sizeof(SparkStageRunnerStats);
	state->model_interface = model;
	state->tp_rank = configuration->tp_rank;
	state->tp_degree = configuration->tp_degree;
	state->stream = (cudaStream_t)configuration->execution_stream;
	state->max_rows = configuration->max_input_row_count;
	state->max_context = configuration->resident_sequence_capacity;
	state->multiprocessors = configuration->multiprocessors;
	state->services.context = state;
	state->services.round = StageRunnerRoundService;
	state->services.published = StageRunnerPublishedService;
	state->services.fail = StageRunnerFailService;
	state->services.expert_weights = StageRunnerExpertWeightsService;
	state->services.layer_done = StageRunnerLayerDoneService;
	request.configuration = configuration;
	request.services = &state->services;
	request.stream = configuration->execution_stream;
	status = model->open(&request, &state->model, &state->geometry);
	if ( status == SPARK_STATUS_OK && (state->geometry.hidden == 0u || state->geometry.vocab == 0u ||
		state->geometry.total_layers == 0u || state->geometry.hidden % 8u != 0u) )
		status = SPARK_STATUS_INVALID_ARGUMENT;
	if ( status == SPARK_STATUS_OK )
		status = SparkStageRunnerStrayLoad(state, configuration->stray_working_set_path);
	if ( status == SPARK_STATUS_OK )
		status = StageRunnerAttachWeights(state, configuration);
	if ( status == SPARK_STATUS_OK )
	{
		status = model->bind(state->model, state->lazy_pack, &state->geometry);
		if ( status != SPARK_STATUS_OK )
			fprintf(stderr, "sparkpipe_stage_runner: %s weight bind failed status=%d\n", model->tag, (int)status);
	}
	if ( status == SPARK_STATUS_OK )
		status = StageRunnerCreateLeases(state);
	if ( status == SPARK_STATUS_OK )
	{
		state->vocab = state->geometry.vocab;
		state->vocab_slice_rows = state->geometry.head_rows != 0u ? state->geometry.head_rows :
			state->geometry.embed_rows != 0u ? state->geometry.embed_rows : state->vocab;
		if ( (runner->owns_embedding != 0u && (state->geometry.embed_rows == 0u || state->geometry.embed_rows > state->vocab)) ||
			(runner->owns_final_head != 0u && (state->geometry.head_rows == 0u || state->geometry.head_rows > state->vocab)) ||
			(state->geometry.embed_rows != 0u && state->geometry.head_rows != 0u && state->geometry.embed_rows != state->geometry.head_rows) )
		{
			fprintf(stderr, "sparkpipe_stage_runner: vocab shard rows embed=%u head=%u vocab=%u do not agree\n",
				state->geometry.embed_rows, state->geometry.head_rows, state->vocab);
			status = SPARK_STATUS_PARSE_ERROR;
		}
	}
	if ( status == SPARK_STATUS_OK )
		status = StageRunnerCreateCollective(state, configuration);
	if ( status == SPARK_STATUS_OK )
		status = StageRunnerCreateHead(state, runner);
	if ( status == SPARK_STATUS_OK )
		status = StageRunnerCreateBuffers(state, runner, configuration);
	if ( status != SPARK_STATUS_OK )
	{
		SparkStageRunnerDestroy(runner);
		SPARK_FAIL(status);
	}
	return SPARK_STATUS_OK;
}

static SparkStatus StageRunnerAllocateDistribution(SparkStageRunnerState *state,
	const SparkStageRunner *runner, const SparkStageRunnerConfiguration *configuration)
{
	const uint64_t rows = configuration->max_input_row_count, slice = state->vocab_slice_rows, hidden = state->geometry.hidden;
	uint64_t full;
	uint32_t elements, sub_rows;
	state->distribution_capacity = 0u;
	if ( runner->owns_final_head == 0u || state->head_weight == 0 )
		return SPARK_STATUS_OK;
	if ( slice == 0u || (uint64_t)state->vocab > slice * runner->tp_degree )
	{
		fprintf(stderr, "sparkpipe_stage_runner: distribution refused: vocab %u exceeds %u ranks of %llu head rows\n",
			state->vocab, runner->tp_degree, (unsigned long long)slice);
		return SPARK_STATUS_SCHEMA_ERROR;
	}
	state->distribution_sub_rows = 1u;
	state->distribution_chunk_rows = (uint32_t)rows;
	if ( runner->tp_degree > 1u && state->device_collective_created != 0 )
	{
		elements = 2u * (uint32_t)slice;
		for ( sub_rows = 1u; sub_rows <= elements && (elements % sub_rows != 0u || elements / sub_rows > hidden); ++sub_rows )
			;
		state->distribution_sub_rows = sub_rows;
		state->distribution_chunk_rows = sub_rows <= elements ? (uint32_t)rows / sub_rows : 0u;
		if ( state->distribution_chunk_rows == 0u )
		{
			fprintf(stderr, "sparkpipe_stage_runner: distribution refused: a %llu-token logit row needs %u collective rows and %llu are configured\n",
				(unsigned long long)slice, sub_rows, (unsigned long long)rows);
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		}
	}
	full = (uint64_t)runner->tp_degree * state->distribution_chunk_rows * slice;
	if ( cudaMalloc(&state->distribution_rows, rows * sizeof(uint32_t)) != cudaSuccess ||
		cudaMalloc(&state->distribution_positions, rows * sizeof(uint32_t)) != cudaSuccess ||
		cudaMalloc(&state->distribution_rules, rows * sizeof(SparkRowSampling)) != cudaSuccess ||
		cudaMalloc(&state->distribution_logprobs, rows * SPARK_SAMPLING_MAX_LOGPROBS * sizeof(SparkSamplingLogprob)) != cudaSuccess ||
		cudaMalloc(&state->distribution_hidden, rows * hidden * sizeof(uint16_t)) != cudaSuccess ||
		cudaMalloc(&state->distribution_normed, rows * hidden * sizeof(uint16_t)) != cudaSuccess ||
		cudaMalloc(&state->distribution_logits, rows * slice * sizeof(float)) != cudaSuccess ||
		cudaMalloc(&state->distribution_gathered, full * sizeof(float)) != cudaSuccess )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	state->distribution_capacity = (uint32_t)rows;
	return SPARK_STATUS_OK;
}

static SparkStatus StageRunnerValidateDistribution(const SparkStageRunnerState *state,
	const SparkStageRunner *runner, const SparkStageRunnerDispatch *dispatch)
{
	uint32_t entry, other, logprobs = 0u;
	if ( dispatch->distribution_count == 0u )
		return SPARK_STATUS_OK;
	if ( runner->owns_final_head == 0u || state->distribution_capacity == 0u ||
		dispatch->distribution_count > dispatch->active_sequence_count ||
		dispatch->distribution_rows == 0 || dispatch->distribution_positions == 0 || dispatch->distribution_rules == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	for ( entry = 0u; entry < dispatch->distribution_count; ++entry )
	{
		if ( dispatch->distribution_rows[entry] >= dispatch->row_count ||
			SparkSamplingRuleValid(&dispatch->distribution_rules[entry]) == 0u ||
			SparkSamplingRuleNeedsDistribution(&dispatch->distribution_rules[entry]) == 0u )
			return SPARK_STATUS_INVALID_ARGUMENT;
		for ( other = 0u; other < entry; ++other )
			if ( dispatch->distribution_rows[other] == dispatch->distribution_rows[entry] )
				return SPARK_STATUS_INVALID_ARGUMENT;
		logprobs |= dispatch->distribution_rules[entry].logprobs;
	}
	return logprobs != 0u && dispatch->distribution_logprobs == 0 ? SPARK_STATUS_INVALID_ARGUMENT : SPARK_STATUS_OK;
}

static SparkStatus StageRunnerSampleDistribution(SparkStageRunnerState *state,
	const float *logits, uint64_t shard_stride, uint32_t row_stride, uint32_t shard_tokens,
	uint32_t first, uint32_t rows, cudaStream_t stream)
{
	LmSampleVocab vocab;
	LmSampleRows sample;
	vocab.logits = logits;
	vocab.shard_stride = shard_stride;
	vocab.row_stride = row_stride;
	vocab.shard_tokens = shard_tokens;
	vocab.vocabulary = state->vocab;
	sample.rules = state->distribution_rules + first;
	sample.positions = state->distribution_positions + first;
	sample.source_rows = state->distribution_rows + first;
	sample.greedy_tokens = state->output_token;
	sample.logprob_rows = 0;
	sample.token_out = state->output_token;
	sample.logit_out = 0;
	sample.logprobs_out = state->distribution_logprobs + (uint64_t)first * SPARK_SAMPLING_MAX_LOGPROBS;
	sample.rows = rows;
	LM_LAUNCH((LmSampleRowsKernel<STAGE_RUNNER_SAMPLE_THREADS>), rows, STAGE_RUNNER_SAMPLE_THREADS, 0, stream, vocab, sample);
	return cudaPeekAtLastError() == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus StageRunnerDistribution(SparkStageRunnerState *state,
	const SparkStageRunner *runner, const SparkStageRunnerDispatch *dispatch, cudaStream_t stream)
{
	const uint32_t count = dispatch->distribution_count, slice = state->vocab_slice_rows, tp = runner->tp_degree;
	const uint32_t hidden = state->geometry.hidden;
	uint32_t first, rows;
	SparkStatus status = SPARK_STATUS_OK;
	if ( count == 0u )
		return SPARK_STATUS_OK;
	if ( cudaMemcpyAsync(state->distribution_rows, dispatch->distribution_rows, (uint64_t)count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
		cudaMemcpyAsync(state->distribution_positions, dispatch->distribution_positions, (uint64_t)count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
		cudaMemcpyAsync(state->distribution_rules, dispatch->distribution_rules, (uint64_t)count * sizeof(SparkRowSampling), cudaMemcpyHostToDevice, stream) != cudaSuccess )
		return SPARK_STATUS_IO_ERROR;
	if ( tp > 1u && state->device_collective_created == 0 &&
		cudaMemcpyAsync(state->output_token, state->output_token_host, (uint64_t)dispatch->row_count * sizeof(uint32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess )
		return SPARK_STATUS_IO_ERROR;
	LmStageRowsGatherKernel<<<count, LM_STAGE_HEAD_THREADS, 0, stream>>>(state->distribution_rows, state->hidden, state->distribution_hidden, hidden);
	LM_LAUNCH((LmFusedResidualRmsNormKernel<LM_STAGE_HEAD_THREADS,uint16_t>), count, LM_STAGE_HEAD_THREADS, (hidden + 8u) * sizeof(float), stream,
		state->distribution_hidden, 0, state->head_norm_weight, 0, state->distribution_normed, hidden, hidden, state->geometry.rms_epsilon);
	LM_LAUNCH((LmHeadLogitsRowsKernel<LM_STAGE_HEAD_THREADS,LM_STAGE_HEAD_TILE,STAGE_RUNNER_DISTRIBUTION_HEAD_ROWS>),
		dim3((slice + LM_STAGE_HEAD_TILE - 1u) / LM_STAGE_HEAD_TILE, (count + STAGE_RUNNER_DISTRIBUTION_HEAD_ROWS - 1u) / STAGE_RUNNER_DISTRIBUTION_HEAD_ROWS),
		LM_STAGE_HEAD_THREADS, 0, stream, state->distribution_normed, state->head_weight, state->distribution_logits, count, hidden, slice, slice);
	if ( cudaPeekAtLastError() != cudaSuccess )
		return SPARK_STATUS_INTERNAL_ERROR;
	if ( tp == 1u )
		status = StageRunnerSampleDistribution(state, state->distribution_logits, (uint64_t)count * slice, slice, slice, 0u, count, stream);
	else if ( state->device_collective_created != 0 )
		for ( first = 0u; status == SPARK_STATUS_OK && first < count; first += rows )
		{
			SparkTpDeviceCollectiveSubmission submission;
			rows = count - first < state->distribution_chunk_rows ? count - first : state->distribution_chunk_rows;
			StageRunnerSubmissionInit(&submission, state->device_collective_deferred, state, stream,
				rows * state->distribution_sub_rows, state->distribution_logits + (uint64_t)first * slice,
				state->distribution_gathered, state->tp_next_ordinal++);
			submission.row_elements = 2u * slice / state->distribution_sub_rows;
			status = SparkTpDeviceCollectiveEnqueue(&state->device_collective, &submission, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_GATHER);
			if ( status == SPARK_STATUS_OK )
				status = StageRunnerSampleDistribution(state, state->distribution_gathered, (uint64_t)rows * slice, slice, slice, first, rows, stream);
		}
	else
		status = SPARK_STATUS_INTERNAL_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = StageRunnerVerifyCollectives(state, stream);
	if ( status != SPARK_STATUS_OK )
		return status;
	if ( StageRunnerCopy(state->output_token_host, state->output_token, (uint64_t)dispatch->row_count * sizeof(uint32_t), stream) != cudaSuccess ||
		(dispatch->output_token_ids != 0 && StageRunnerCopy(dispatch->output_token_ids, state->output_token, (uint64_t)dispatch->row_count * sizeof(uint32_t), stream) != cudaSuccess) ||
		(dispatch->distribution_logprobs != 0 && StageRunnerCopy(dispatch->distribution_logprobs, state->distribution_logprobs, (uint64_t)count * SPARK_SAMPLING_MAX_LOGPROBS * sizeof(SparkSamplingLogprob), stream) != cudaSuccess) )
		state->copy_failed = 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus StageRunnerCheckResident(SparkStageRunnerState *state)
{
	const void *pool = 0;
	uint32_t resident = 0u;
	SparkStatus status;
	if ( state->lazy_pack == 0 || state->lazy_pack->map == 0 || state->resident != 0u )
		return SPARK_STATUS_OK;
	status = SparkWeightdMapResident(state->lazy_pack->map, SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS, &resident);
	if ( status == SPARK_STATUS_OK && resident != 0u )
		status = SparkWeightdMapPool(state->lazy_pack->map, &pool);
	if ( status != SPARK_STATUS_OK || resident == 0u )
		return status;
	if ( cudaStreamSynchronize(state->stream) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	status = SparkStageRunnerReleaseLease(state);
	if ( status != SPARK_STATUS_OK )
		return status;
	state->resident_base = pool;
	state->resident = 1u;
	fprintf(stderr, "STAGE-RESIDENT every expert group is present in a fixed pool: per-layer expert leases are off\n");
	return SPARK_STATUS_OK;
}

static SparkStatus StageRunnerChainBegin(SparkStageRunnerState *state, uint64_t request_id)
{
	const uint64_t key = request_id & SPARK_TP_DEVICE_COLLECTIVE_CHAIN_ID_MASK;
	SparkStatus status = StageRunnerVerifyCollectives(state, state->stream);
	if ( status == SPARK_STATUS_OK && state->device_collective_created != 0 )
		status = SparkTpDeviceCollectiveChainKey(&state->device_collective, key);
	return status;
}

static SparkStatus StageRunnerChainEnd(SparkStageRunnerState *state, cudaStream_t stream)
{
	SparkStatus status = StageRunnerVerifyCollectives(state, stream);
	if ( status == SPARK_STATUS_OK && state->device_collective_created != 0 )
		status = SparkTpDeviceCollectiveEndChain(&state->device_collective, stream);
	return status;
}

static uint64_t StageRunnerNowNs(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC, &now);
	return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}

static void StageRunnerPhases(SparkStageRunnerState *state, uint32_t rank, const uint64_t *marks)
{
	uint32_t phase;
	for ( phase = 0u; phase < 4u; ++phase )
		state->phase_ns[phase] += marks[phase + 1u] - marks[phase];
	if ( ++state->phase_waves < 16u )
		return;
	fprintf(stderr, "STAGE-SUBMIT-PHASES rank=%u waves=%u setup_ms=%.1f enqueue_ms=%.1f outputs_ms=%.1f chain_end_ms=%.1f\n",
		rank, state->phase_waves, (double)state->phase_ns[0] / state->phase_waves / 1e6, (double)state->phase_ns[1] / state->phase_waves / 1e6,
		(double)state->phase_ns[2] / state->phase_waves / 1e6, (double)state->phase_ns[3] / state->phase_waves / 1e6);
	memset(state->phase_ns, 0, sizeof(state->phase_ns));
	state->phase_waves = 0u;
}

static uint32_t StageRunnerLayerMarksBegin(SparkStageRunnerState *state)
{
	uint32_t index;
	state->layer_marks = 0u;
	for ( ; state->layer_events_created < state->mark_count; ++state->layer_events_created )
		if ( cudaEventCreate(&state->layer_event[state->layer_events_created]) != cudaSuccess )
			return 0u;
	state->layer_marks = 1u;
	state->layer_mark_last = 0u;
	for ( index = 0u; index < state->mark_count; ++index )
		state->layer_host_ns[index] = 0u;
	StageRunnerLayerMark(state, 0u);
	return state->layer_marks;
}

static void StageRunnerLayerMarksEnd(SparkStageRunnerState *state, uint32_t rank)
{
	const uint32_t last = state->layer_mark_last;
	uint32_t index, order, top[4] = {0u, 0u, 0u, 0u}, starved = 0u;
	float gpu, since;
	state->layer_marks = 0u;
	if ( last == 0u || cudaEventSynchronize(state->layer_event[last]) != cudaSuccess )
		return;
	for ( index = 1u; index <= last; ++index )
	{
		if ( state->layer_host_ns[index] == 0u ||
			cudaEventElapsedTime(&gpu, state->layer_event[index - 1u], state->layer_event[index]) != cudaSuccess ||
			cudaEventElapsedTime(&since, state->layer_event[0], state->layer_event[index]) != cudaSuccess )
			return;
		state->layer_gpu_ms[index] += gpu;
		state->layer_enqueue_ms[index] += (double)(state->layer_host_ns[index] - state->layer_host_ns[index - 1u]) / 1e6;
		state->layer_lag_ms[index] += since - (double)(state->layer_host_ns[index] - state->layer_host_ns[0]) / 1e6;
	}
	state->layer_span_ms[0] += since;
	state->layer_span_ms[1] += (double)(state->layer_host_ns[last] - state->layer_host_ns[0]) / 1e6;
	if ( ++state->layer_waves < 16u )
		return;
	for ( index = 1u; index <= last; ++index )
	{
		for ( order = 0u; order < 4u; ++order )
			if ( top[order] == 0u || state->layer_gpu_ms[index] > state->layer_gpu_ms[top[order]] )
			{
				memmove(top + order + 1u, top + order, (3u - order) * sizeof(top[0]));
				top[order] = index;
				break;
			}
		if ( starved == 0u || state->layer_lag_ms[index] < state->layer_lag_ms[starved] )
			starved = index;
	}
	fprintf(stderr, "STAGE-LAYER-MARKS rank=%u waves=%u layers=%u gpu_span_ms=%.1f host_span_ms=%.1f starved=%u:%.1f top",
		rank, state->layer_waves, last, state->layer_span_ms[0] / state->layer_waves, state->layer_span_ms[1] / state->layer_waves,
		starved - 1u, state->layer_lag_ms[starved] / state->layer_waves);
	for ( order = 0u; order < 4u && top[order] != 0u; ++order )
		fprintf(stderr, " %u:%.1f/%.1f/%.1f", top[order] - 1u, state->layer_gpu_ms[top[order]] / state->layer_waves,
			state->layer_enqueue_ms[top[order]] / state->layer_waves, state->layer_lag_ms[top[order]] / state->layer_waves);
	fprintf(stderr, "\nSTAGE-LAYER-GPU rank=%u ms", rank);
	for ( index = 1u; index <= last; ++index )
		fprintf(stderr, "%c%.0f", index == 1u ? '=' : ',', state->layer_gpu_ms[index] / state->layer_waves);
	fprintf(stderr, "\nSTAGE-LAYER-LAG rank=%u ms", rank);
	for ( index = 1u; index <= last; ++index )
		fprintf(stderr, "%c%.0f", index == 1u ? '=' : ',', state->layer_lag_ms[index] / state->layer_waves);
	fprintf(stderr, "\n");
	memset(state->layer_gpu_ms, 0, (size_t)state->mark_count * sizeof(double));
	memset(state->layer_enqueue_ms, 0, (size_t)state->mark_count * sizeof(double));
	memset(state->layer_lag_ms, 0, (size_t)state->mark_count * sizeof(double));
	memset(state->layer_span_ms, 0, sizeof(state->layer_span_ms));
	state->layer_waves = 0u;
}

static SparkStatus StageRunnerEmbed(SparkStageRunnerState *state, const SparkStageRunner *runner, uint32_t rows, cudaStream_t stream)
{
	if ( LmStageEmbedding(state->embed_weight, state->token_ids_device, state->hidden, rows, state->geometry.hidden,
		runner->tp_rank * state->vocab_slice_rows, state->vocab_slice_rows, stream) != LM_LAUNCH_OK )
		return SPARK_STATUS_INTERNAL_ERROR;
	if ( StageRunnerReduceBf16(state, stream, state->hidden, rows) != SPARK_STATUS_OK )
		return SPARK_STATUS_INTERNAL_ERROR;
	return SPARK_STATUS_OK;
}

static SparkStatus StageRunnerHeadRows(SparkStageRunnerState *state, const uint16_t *hidden, uint32_t *token, float *score,
	uint32_t rows, cudaStream_t stream)
{
	SparkTpDeviceCollectiveSubmission submission;
	LmStageHeadRows head;
	LmStageHeadCertified certified;
	int32_t status;
	head.hidden_bf16 = hidden;
	head.normed_bf16 = state->head_normed;
	head.norm_weight = state->head_norm_weight;
	head.head_weight = state->head_weight;
	head.hidden = state->geometry.hidden;
	head.epsilon = state->geometry.rms_epsilon;
	head.norm_f32 = state->geometry.head_norm_f32;
	head.vocab_slice_rows = state->vocab_slice_rows;
	head.rank_offset = state->tp_rank * state->vocab_slice_rows;
	head.candidate_score = state->head_candidate_score;
	head.candidate_token = state->head_candidate_token;
	head.output_token = token;
	head.output_score = score;
	certified.payload = state->head_certified_fp8_payload;
	certified.scale = state->head_certified_fp8_scale_f32;
	certified.norm = state->head_certified_fp8_norm_f32;
	certified.scratch = state->head_certified_scratch;
	certified.candidates = state->head_certified_candidates;
	certified.screened = state->head_screened_count;
	status = LmStageHeadCertifiedSlice(head, certified, rows, stream);
	if ( status != LM_LAUNCH_OK )
		{ fprintf(stderr, "sparkpipe_stage_runner: final head launch failed %d\n", status); return SPARK_STATUS_INTERNAL_ERROR; }
	if ( state->device_collective_created == 0 )
		return SPARK_STATUS_OK;
	if ( LmStageMaxlocPack(score, token, state->head_maxloc, rows, stream) != LM_LAUNCH_OK )
		{ fprintf(stderr, "sparkpipe_stage_runner: head maxloc pack failed\n"); return SPARK_STATUS_INTERNAL_ERROR; }
	StageRunnerSubmissionInit(&submission, state->device_collective_deferred, state, stream,
		rows, state->head_maxloc, state->head_maxloc, state->tp_next_ordinal++);
	if ( SparkTpDeviceCollectiveEnqueue(&state->device_collective, &submission,
			SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_MAX_U64) != SPARK_STATUS_OK )
		{ fprintf(stderr, "sparkpipe_stage_runner: head argmax collective enqueue failed\n"); return SPARK_STATUS_INTERNAL_ERROR; }
	if ( LmStageMaxlocUnpack(state->head_maxloc, token, score, rows, stream) != LM_LAUNCH_OK )
		{ fprintf(stderr, "sparkpipe_stage_runner: head maxloc unpack failed\n"); return SPARK_STATUS_INTERNAL_ERROR; }
	return SPARK_STATUS_OK;
}

static SparkStatus StageRunnerHeadArgmax(SparkStageRunnerState *state, const SparkStageRunnerStep *in, cudaStream_t stream)
{
	SparkStatus status;
	const uint32_t rows = in->rows, sequences = in->sequences;
	if ( state->head_last_rows == 0u || sequences >= rows || in->sequence_row_begin == 0 || in->sequence_row_indices == 0 )
		return StageRunnerHeadRows(state, state->hidden, state->output_token, state->output_score, rows, stream);
	LmStageLastRowsGatherKernel<<<sequences, LM_STAGE_HEAD_THREADS, 0, stream>>>(in->sequence_row_begin,
		in->sequence_row_indices, state->hidden, state->head_hidden, state->geometry.hidden);
	if ( cudaPeekAtLastError() != cudaSuccess )
		return SPARK_STATUS_INTERNAL_ERROR;
	status = StageRunnerHeadRows(state, state->head_hidden, state->head_token, state->head_score, sequences, stream);
	if ( status != SPARK_STATUS_OK )
		return status;
	LmStageLastRowsScatterKernel<<<(sequences + LM_STAGE_HEAD_THREADS - 1u) / LM_STAGE_HEAD_THREADS, LM_STAGE_HEAD_THREADS, 0, stream>>>(
		in->sequence_row_begin, in->sequence_row_indices, state->head_token, state->head_score,
		state->output_token, state->output_score, sequences);
	return cudaPeekAtLastError() == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus StageRunnerStepBody(SparkStageRunnerState *state, const SparkStageRunner *runner,
	const SparkStageRunnerStep *in, cudaStream_t stream)
{
	SparkStatus status = runner->owns_embedding != 0u ? StageRunnerEmbed(state, runner, in->rows, stream) : SPARK_STATUS_OK;
	if ( status != SPARK_STATUS_OK )
		return status;
	status = state->model_interface->step(state->model, in, stream);
	if ( status != SPARK_STATUS_OK )
		{ fprintf(stderr, "sparkpipe_stage_runner: %s step failed status=%d\n", state->model_interface->tag, (int)status); return status; }
	if ( runner->owns_final_head != 0u && state->head_weight != 0 )
		return StageRunnerHeadArgmax(state, in, stream);
	return SPARK_STATUS_OK;
}

static uint32_t StageRunnerGraphEligible(const SparkStageRunnerState *state, const SparkStageRunner *runner,
	uint32_t rows, uint32_t sequences)
{
	return (state->resident != 0u || state->geometry.experts == 0u) && state->device_collective_created != 0 &&
		state->device_collective_deferred != 0u && runner->owns_embedding != 0u &&
		runner->owns_final_head != 0u && state->head_weight != 0 && rows == sequences &&
		rows <= STAGE_RUNNER_GRAPH_ROWS && state->graph_refused[rows - 1u] == 0u ? 1u : 0u;
}

static void StageRunnerGraphDrop(SparkStageRunnerState *state)
{
	for ( uint32_t index = 0u; index < STAGE_RUNNER_GRAPH_ROWS; ++index )
	{
		if ( state->graph_exec[index] != 0 )
			(void)cudaGraphExecDestroy(state->graph_exec[index]);
		state->graph_exec[index] = 0;
		state->graph_refused[index] = 0u;
	}
}

static void StageRunnerGraphCapture(SparkStageRunnerState *state, const SparkStageRunner *runner,
	const SparkStageRunnerStep *in, cudaStream_t stream)
{
	const uint64_t started = StageRunnerNowNs();
	const uint32_t rows = in->rows;
	cudaGraph_t graph = 0;
	cudaGraphExec_t exec = 0;
	cudaError_t error = cudaSuccess;
	size_t nodes = 0u;
	SparkStatus status = SparkTpDeviceCollectiveArmCapture(&state->device_collective);
	if ( status == SPARK_STATUS_OK )
	{
		error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
		if ( error == cudaSuccess )
		{
			status = StageRunnerStepBody(state, runner, in, stream);
			error = cudaStreamEndCapture(stream, &graph);
		}
	}
	(void)SparkTpDeviceCollectiveDisarmCapture(&state->device_collective);
	if ( status == SPARK_STATUS_OK && error == cudaSuccess && graph != 0 )
		error = cudaGraphGetNodes(graph, 0, &nodes);
	if ( status == SPARK_STATUS_OK && error == cudaSuccess && graph != 0 )
		error = cudaGraphInstantiate(&exec, graph, 0);
	if ( status == SPARK_STATUS_OK && error == cudaSuccess && exec != 0 )
		error = cudaGraphUpload(exec, stream);
	if ( graph != 0 )
		(void)cudaGraphDestroy(graph);
	if ( status != SPARK_STATUS_OK || error != cudaSuccess || exec == 0 )
	{
		fprintf(stderr, "STAGE-GRAPH-CAPTURE-FAILED rows=%u status=%d cuda=%s: this row count runs eager\n",
			rows, (int)status, cudaGetErrorString(error));
		if ( exec != 0 )
			(void)cudaGraphExecDestroy(exec);
		(void)cudaGetLastError();
		state->graph_refused[rows - 1u] = 1u;
		state->copy_failed = 0u;
		state->tp_collective_failed = 0u;
		return;
	}
	state->graph_exec[rows - 1u] = exec;
	state->graph_input[rows - 1u] = *in;
	fprintf(stderr, "STAGE-GRAPH-CAPTURE rows=%u nodes=%llu capture_ms=%.1f\n", rows,
		(unsigned long long)nodes, (double)(StageRunnerNowNs() - started) / 1e6);
}

static SparkStatus StageRunnerGraphLaunch(SparkStageRunnerState *state, cudaGraphExec_t exec, cudaStream_t stream)
{
	uint64_t collective_error = 0ull;
	SparkStatus status = SparkTpDeviceCollectiveGraphPreLaunch(&state->device_collective, stream);
	if ( status == SPARK_STATUS_OK )
		status = SparkTpDeviceCollectiveGraphCancelSeed(&state->device_collective, stream);
	const uint64_t started = StageRunnerNowNs();
	if ( status == SPARK_STATUS_OK && cudaGraphLaunch(exec, stream) != cudaSuccess )
		status = SPARK_STATUS_IO_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = SparkTpDeviceCollectiveGraphSettle(&state->device_collective, stream, &collective_error);
	state->timing_graph_ns += StageRunnerNowNs() - started;
	state->timing_graph_steps++;
	if ( status == SPARK_STATUS_OK && collective_error != 0ull )
		status = SPARK_STATUS_INTERNAL_ERROR;
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr, "STAGE-GRAPH-FAILED status=%d collective_error=%llu\n", (int)status, (unsigned long long)collective_error);
		return status;
	}
	state->graph_launches++;
	return SPARK_STATUS_OK;
}

static void StageRunnerTiming(SparkStageRunnerState *state, uint32_t rank, uint64_t submit_ns)
{
	SparkTpDeviceCollectiveHardwareTiming step;
	uint64_t steps;
	state->timing_submit_ns += submit_ns;
	state->timing_steps++;
	if ( state->device_collective_created != 0 &&
		SparkTpDeviceCollectiveHardwareStats(&state->device_collective, &step) == SPARK_STATUS_OK )
	{
		state->timing_wait.source_wait_ns += step.source_wait_ns;
		state->timing_wait.peer_wait_ns += step.peer_wait_ns;
		state->timing_wait.copy_ns += step.copy_ns;
		state->timing_wait.combine_ns += step.combine_ns;
	}
	if ( state->timing_steps < 64u )
		return;
	steps = state->timing_steps;
	fprintf(stderr, "STAGE-STEP-TIMING rank=%u steps=%llu graph_steps=%llu submit_us=%llu graph_us=%llu source_wait_us=%llu peer_wait_us=%llu copy_us=%llu combine_us=%llu\n",
		rank, (unsigned long long)steps, (unsigned long long)state->timing_graph_steps,
		(unsigned long long)(state->timing_submit_ns / steps / 1000u),
		(unsigned long long)(state->timing_graph_steps != 0u ? state->timing_graph_ns / state->timing_graph_steps / 1000u : 0u),
		(unsigned long long)(state->timing_wait.source_wait_ns / steps / 1000u),
		(unsigned long long)(state->timing_wait.peer_wait_ns / steps / 1000u),
		(unsigned long long)(state->timing_wait.copy_ns / steps / 1000u),
		(unsigned long long)(state->timing_wait.combine_ns / steps / 1000u));
	if ( state->model_interface->report != 0 )
		state->model_interface->report(state->model, rank);
	state->timing_steps = 0u;
	state->timing_graph_steps = 0u;
	state->timing_submit_ns = 0u;
	state->timing_graph_ns = 0u;
	memset(&state->timing_wait, 0, sizeof(state->timing_wait));
}

static SparkStatus StageRunnerStep(SparkStageRunnerState *state, const SparkStageRunner *runner,
	const SparkStageRunnerStep *in, cudaStream_t stream)
{
	const uint32_t rows = in->rows;
	if ( StageRunnerGraphEligible(state, runner, rows, in->sequences) != 0u )
	{
		if ( state->graph_exec[rows - 1u] != 0 && memcmp(&state->graph_input[rows - 1u], in, sizeof(*in)) != 0 )
		{
			(void)cudaGraphExecDestroy(state->graph_exec[rows - 1u]);
			state->graph_exec[rows - 1u] = 0;
		}
		if ( state->graph_exec[rows - 1u] == 0 )
			StageRunnerGraphCapture(state, runner, in, stream);
		if ( state->graph_exec[rows - 1u] != 0 )
			return StageRunnerGraphLaunch(state, state->graph_exec[rows - 1u], stream);
	}
	return StageRunnerStepBody(state, runner, in, stream);
}

static SparkStatus StageRunnerStepOutputs(SparkStageRunnerState *state, const SparkStageRunner *runner,
	const SparkStageRunnerDispatch *current, cudaStream_t stream, uint32_t rows)
{
	SparkStatus exchange_status;
	if ( runner->owns_final_head != 0u && state->head_weight != 0 )
	{
		if ( state->device_collective_created != 0 )
		{
			exchange_status = StageRunnerVerifyCollectives(state, stream);
			if ( exchange_status != SPARK_STATUS_OK )
				return exchange_status;
			if ( StageRunnerCopy(state->output_token_host, state->output_token, (uint64_t)rows * sizeof(uint32_t), stream) != cudaSuccess ||
				StageRunnerCopy(state->output_score_host, state->output_score, (uint64_t)rows * sizeof(float), stream) != cudaSuccess ||
				(current->output_token_ids != 0 && StageRunnerCopy(current->output_token_ids, state->output_token, (uint64_t)rows * sizeof(uint32_t), stream) != cudaSuccess) ||
				(current->output_scores != 0 && StageRunnerCopy(current->output_scores, state->output_score, (uint64_t)rows * sizeof(float), stream) != cudaSuccess) )
				state->copy_failed = 1u;
		}
		else
		{
			if ( cudaStreamSynchronize(stream) != cudaSuccess ||
				StageRunnerCopy(state->output_token_host, state->output_token, (uint64_t)rows * sizeof(uint32_t), stream) != cudaSuccess ||
				StageRunnerCopy(state->output_score_host, state->output_score, (uint64_t)rows * sizeof(float), stream) != cudaSuccess )
				state->copy_failed = 1u;
			if ( runner->tp_degree > 1u )
				return SPARK_STATUS_INTERNAL_ERROR;
			if ( (current->output_token_ids != 0 && StageRunnerCopy(current->output_token_ids, state->output_token_host, (uint64_t)rows * sizeof(uint32_t), stream) != cudaSuccess) ||
				(current->output_scores != 0 && StageRunnerCopy(current->output_scores, state->output_score_host, (uint64_t)rows * sizeof(float), stream) != cudaSuccess) )
				state->copy_failed = 1u;
		}
		return StageRunnerDistribution(state, runner, current, stream);
	}
	if ( current->hidden_output_bf16 == 0 )
		return SPARK_STATUS_OK;
	exchange_status = StageRunnerVerifyCollectives(state, stream);
	if ( exchange_status != SPARK_STATUS_OK )
		return exchange_status;
	if ( StageRunnerCopy(current->hidden_output_bf16, state->hidden, (uint64_t)rows * state->geometry.hidden * sizeof(uint16_t), stream) != cudaSuccess )
		state->copy_failed = 1u;
	if ( current->sideband_output == 0 )
		return SPARK_STATUS_OK;
	if ( state->model_interface->sideband == 0 || current->sideband_output_bytes < (uint64_t)rows * state->geometry.sideband_bytes_per_row )
		return SPARK_STATUS_INVALID_ARGUMENT;
	return state->model_interface->sideband(state->model, 1u, state->hidden, current->sideband_output, (uint64_t)rows * state->geometry.sideband_bytes_per_row, rows, stream);
}

#define STAGE_RUNNER_CHAIN_LANES 16u

__global__ static void StageRunnerChainAdvanceKernel(uint32_t *token_ids, const uint32_t *tokens, uint32_t *positions,
	uint32_t *context_length, const uint32_t *sequence_of_row, uint32_t rows)
{
	const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
	if ( row >= rows )
		return;
	token_ids[row] = tokens[row];
	positions[row] += 1u;
	context_length[sequence_of_row[row]] = positions[row] + 1u;
}

static SparkStatus StageRunnerStageInput(SparkStageRunnerState *state, const SparkStageRunnerDispatch *dispatch, uint32_t rows, cudaStream_t stream)
{
	if ( dispatch->hidden_input_bf16 == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ( StageRunnerCopy(state->hidden, dispatch->hidden_input_bf16, (uint64_t)rows * state->geometry.hidden * sizeof(uint16_t), stream) != cudaSuccess )
		state->copy_failed = 1u;
	if ( state->model_interface->sideband == 0 )
		return dispatch->sideband_input == 0 ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT;
	if ( dispatch->sideband_input != 0 && dispatch->sideband_input_bytes < (uint64_t)rows * state->geometry.sideband_bytes_per_row )
		return SPARK_STATUS_INVALID_ARGUMENT;
	return state->model_interface->sideband(state->model, 0u, state->hidden, (void *)dispatch->sideband_input,
		dispatch->sideband_input != 0 ? (uint64_t)rows * state->geometry.sideband_bytes_per_row : 0u, rows, stream);
}

SparkStatus SparkStageRunnerSubmit(SparkStageRunner *runner, const SparkStageRunnerDispatch *dispatch)
{
	SparkStageRunnerState *state;
	SparkStageRunnerStep in;
	cudaStream_t stream;
	uint32_t rows;
	SparkStatus exchange_status;
	SparkModelDriverCompletion completion;
	uint32_t i, step, steps;
	const uint64_t submit_started = StageRunnerNowNs();
	uint64_t phase_marks[5];
	if ( runner == 0 || dispatch == 0 || runner->private_state == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	state = (SparkStageRunnerState *)runner->private_state;
	rows = dispatch->row_count;
	steps = dispatch->chain_steps > 1u ? dispatch->chain_steps : 1u;
	if ( steps > 1u && (runner->owns_embedding == 0u || runner->owns_final_head == 0u || state->head_weight == 0 ||
		rows != dispatch->active_sequence_count || (uint64_t)rows * steps > state->max_rows ||
		dispatch->distribution_count > STAGE_RUNNER_CHAIN_LANES || dispatch->positions == 0 || dispatch->context_length == 0) )
	{
		fprintf(stderr, "sparkpipe_stage_runner: decode chain refused: steps=%u rows=%u lanes=%u distributions=%u\n",
			steps, rows, dispatch->active_sequence_count, dispatch->distribution_count);
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	if ( rows == 0u || rows > state->max_rows || dispatch->active_sequence_count == 0u || dispatch->active_sequence_count > rows ||
		(runner->owns_embedding != 0u && dispatch->token_ids == 0) )
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ( (dispatch->flags & ~SPARK_STAGE_RUNNER_DISPATCH_KNOWN_FLAGS) != 0u )
	{
		fprintf(stderr, "sparkpipe_stage_runner: submit refused: unknown dispatch flags 0x%x\n", dispatch->flags);
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	if ( state->geometry.kv_layer_count != 0u && state->kv_attached == 0u )
	{
		fprintf(stderr, "sparkpipe_stage_runner: submit refused: the KV binding pool is not attached\n");
		return SPARK_STATUS_INVALID_ARGUMENT;
	}
	exchange_status = StageRunnerValidateDistribution(state, runner, dispatch);
	if ( exchange_status != SPARK_STATUS_OK )
		return exchange_status;
	stream = state->stream;
	exchange_status = StageRunnerCheckResident(state);
	if ( exchange_status != SPARK_STATUS_OK )
		return exchange_status;
	exchange_status = StageRunnerChainBegin(state, dispatch->request_id);
	if ( exchange_status != SPARK_STATUS_OK )
		return exchange_status;
	state->rows = rows;
	state->head_last_rows = (dispatch->flags & SPARK_STAGE_RUNNER_DISPATCH_FLAG_PREFILL) != 0u ? 1u : 0u;
	state->logical_sequence_count = dispatch->active_sequence_count;
	if ( runner->owns_embedding != 0u )
	{
		if ( cudaMemcpyAsync(state->token_ids_device, dispatch->token_ids, (uint64_t)rows * sizeof(*state->token_ids_device),
			cudaMemcpyDefault, stream) != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess )
			return SPARK_STATUS_IO_ERROR;
	}
	else
	{
		exchange_status = StageRunnerStageInput(state, dispatch, rows, stream);
		if ( exchange_status != SPARK_STATUS_OK )
			return exchange_status;
	}
	memset(&in, 0, sizeof(in));
	in.hidden_bf16 = state->hidden;
	in.positions = dispatch->positions;
	in.context_length = dispatch->context_length;
	in.sequence_of_row = dispatch->sequence_of_row;
	in.sequence_row_begin = dispatch->sequence_row_begin;
	in.sequence_row_indices = dispatch->sequence_row_indices;
	in.recurrent_index = dispatch->recurrent_index;
	in.gather_context = (dispatch->flags & SPARK_STAGE_RUNNER_DISPATCH_FLAG_PREFILL) != 0u && dispatch->active_sequence_count == 1u
		? dispatch->gather_context : 0u;
	in.gather_sequence = in.gather_context != 0u ? dispatch->gather_sequence : 0u;
	in.rows = rows;
	in.sequences = dispatch->active_sequence_count;
	in.commit = 1u;
	in.context = state->max_context;
	in.multiprocessors = state->multiprocessors;
	in.last_rows_only = state->head_last_rows;
	phase_marks[0] = submit_started;
	if ( steps == 1u && rows > STAGE_RUNNER_WIDE_ROWS )
		(void)StageRunnerLayerMarksBegin(state);
	phase_marks[1] = StageRunnerNowNs();
	for ( step = 0u; step < steps; ++step )
	{
		SparkStageRunnerDispatch current = *dispatch;
		uint32_t chain_positions[STAGE_RUNNER_CHAIN_LANES];
		if ( step != 0u )
		{
			StageRunnerChainAdvanceKernel<<<(rows + 63u) / 64u, 64u, 0, stream>>>(state->token_ids_device,
				state->output_token, (uint32_t *)dispatch->positions, (uint32_t *)dispatch->context_length,
				dispatch->sequence_of_row, rows);
			if ( cudaPeekAtLastError() != cudaSuccess )
				return SPARK_STATUS_INTERNAL_ERROR;
			if ( dispatch->output_token_ids != 0 )
				current.output_token_ids = dispatch->output_token_ids + (uint64_t)step * rows;
			if ( dispatch->output_scores != 0 )
				current.output_scores = dispatch->output_scores + (uint64_t)step * rows;
			for ( i = 0u; i < dispatch->distribution_count; ++i )
				chain_positions[i] = dispatch->distribution_positions[i] + step;
			if ( dispatch->distribution_count != 0u )
				current.distribution_positions = chain_positions;
		}
		exchange_status = StageRunnerStep(state, runner, &in, stream);
		phase_marks[2] = StageRunnerNowNs();
		if ( exchange_status == SPARK_STATUS_OK )
			exchange_status = StageRunnerStepOutputs(state, runner, &current, stream, rows);
		phase_marks[3] = StageRunnerNowNs();
		if ( state->layer_marks != 0u && exchange_status == SPARK_STATUS_OK )
			StageRunnerLayerMarksEnd(state, runner->tp_rank);
		state->layer_marks = 0u;
		if ( exchange_status != SPARK_STATUS_OK )
			return exchange_status;
	}
	exchange_status = StageRunnerTakeFailure(state);
	if ( exchange_status != SPARK_STATUS_OK )
		return exchange_status;
	StageRunnerTiming(state, runner->tp_rank, StageRunnerNowNs() - submit_started);
	exchange_status = StageRunnerChainEnd(state, stream);
	phase_marks[4] = StageRunnerNowNs();
	if ( exchange_status == SPARK_STATUS_OK && rows > STAGE_RUNNER_WIDE_ROWS )
		StageRunnerPhases(state, runner->tp_rank, phase_marks);
	if ( exchange_status != SPARK_STATUS_OK )
		return exchange_status;
	runner->stats.submitted_count++;
	runner->stats.completed_count++;
	if ( dispatch->completion_function != 0 )
	{
		if ( runner->owns_final_head == 0u )
			memset(state->output_token_host, 0, (uint64_t)rows * 4u);
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

SparkStatus SparkStageRunnerResetSlots(SparkStageRunner *runner, const uint32_t *slots, uint32_t count)
{
	SparkStageRunnerState *state;
	SparkStatus status;
	if ( runner == 0 || runner->private_state == 0 || (count != 0u && slots == 0) )
		return SPARK_STATUS_INVALID_ARGUMENT;
	state = (SparkStageRunnerState *)runner->private_state;
	if ( count != 0u && state->model_interface->reset_slot == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	for ( uint32_t i = 0u; i < count; ++i )
	{
		status = state->model_interface->reset_slot(state->model, slots[i], state->stream);
		if ( status != SPARK_STATUS_OK )
			return status;
	}
	return cudaStreamSynchronize(state->stream) == cudaSuccess ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
}

SparkStatus SparkStageRunnerGetStats(const SparkStageRunner *runner, SparkStageRunnerStats *stats_out)
{
	if ( runner == 0 || stats_out == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	*stats_out = runner->stats;
	return SPARK_STATUS_OK;
}

uint32_t SparkStageRunnerKvLayerCount(const SparkStageRunner *runner)
{
	if ( runner == 0 || runner->private_state == 0 )
		return 0u;
	return ((const SparkStageRunnerState *)runner->private_state)->geometry.kv_layer_count;
}

SparkStatus SparkStageRunnerAttachKv(SparkStageRunner *runner, const SparkStageRunnerKv *kv)
{
	SparkStageRunnerState *state;
	SparkStatus status;
	if ( runner == 0 || runner->private_state == 0 || kv == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	state = (SparkStageRunnerState *)runner->private_state;
	if ( state->model_interface->attach_kv == 0 || kv->layer_count != state->geometry.kv_layer_count ||
		kv->layer_page_bytes != state->geometry.kv_layer_page_bytes )
	{
		fprintf(stderr, "sparkpipe_stage_runner: KV attach refused: binding has %u layers of %llu-byte pages, the slice needs %u of %llu\n",
			kv->layer_count, (unsigned long long)kv->layer_page_bytes, state->geometry.kv_layer_count,
			(unsigned long long)state->geometry.kv_layer_page_bytes);
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	}
	{
		const SparkKvShard want = state->geometry.kv_shard;
		const uint32_t sharded = want.degree > 1u ? 1u : 0u;
		if ( sharded != (kv->context_shard.degree > 1u ? 1u : 0u) || (sharded != 0u &&
			(kv->context_shard.degree != want.degree || kv->context_shard.rank != want.rank || kv->context_shard.grain != want.grain)) )
		{
			fprintf(stderr, "sparkpipe_stage_runner: KV attach refused: binding splits context %u/%u grain %u, the slice needs %u/%u grain %u\n",
				kv->context_shard.rank, kv->context_shard.degree, kv->context_shard.grain,
				sharded != 0u ? want.rank : 0u, sharded != 0u ? want.degree : 1u, sharded != 0u ? want.grain : 0u);
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
		}
	}
	StageRunnerGraphDrop(state);
	status = state->model_interface->attach_kv(state->model, kv);
	if ( status == SPARK_STATUS_OK )
		state->kv_attached = 1u;
	return status;
}

uint64_t SparkStageRunnerRecurrentBytes(const SparkStageRunner *runner)
{
	if ( runner == 0 || runner->private_state == 0 )
		return 0u;
	return ((const SparkStageRunnerState *)runner->private_state)->geometry.recurrent_bytes;
}

SparkStatus SparkStageRunnerRecurrentCopy(SparkStageRunner *runner, uint32_t to_buffer, uint32_t slot, void *buffer, uint64_t bytes, void *stream)
{
	SparkStageRunnerState *state;
	if ( runner == 0 || runner->private_state == 0 || buffer == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	state = (SparkStageRunnerState *)runner->private_state;
	if ( state->model_interface->recurrent_copy == 0 || bytes != state->geometry.recurrent_bytes )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	return state->model_interface->recurrent_copy(state->model, to_buffer, slot, buffer, bytes, stream);
}

SparkStatus SparkStageRunnerPackIdentity(const SparkStageRunner *runner, uint8_t *digest, uint32_t digest_bytes)
{
	const SparkStageRunnerState *state;
	if ( runner == 0 || runner->private_state == 0 || digest == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	state = (const SparkStageRunnerState *)runner->private_state;
	if ( state->lazy_pack == 0 || digest_bytes != sizeof(state->lazy_pack->pack_sha256) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	memcpy(digest, state->lazy_pack->pack_sha256, digest_bytes);
	return SPARK_STATUS_OK;
}

const void *SparkStageRunnerProbeBuffers(const SparkStageRunner *runner)
{
	const SparkStageRunnerState *state;
	if ( runner == 0 || runner->private_state == 0 )
		return 0;
	state = (const SparkStageRunnerState *)runner->private_state;
	return state->model_interface->probe_buffers != 0 ? state->model_interface->probe_buffers(state->model) : 0;
}

void *SparkStageRunnerModel(const SparkStageRunner *runner)
{
	if ( runner == 0 || runner->private_state == 0 )
		return 0;
	return ((const SparkStageRunnerState *)runner->private_state)->model;
}

void SparkStageRunnerDestroy(SparkStageRunner *runner)
{
	SparkStageRunnerState *state;
	if ( runner == 0 || runner->private_state == 0 )
		return;
	state = (SparkStageRunnerState *)runner->private_state;
	StageRunnerGraphDrop(state);
	SparkStageRunnerStrayReport(state);
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
	if ( state->lease_slots != 0u && SparkStageRunnerReleaseLease(state) != SPARK_STATUS_OK )
		return;
	if ( state->lazy_pack != 0 )
	{
		if ( SparkWeightdLazyPackDestroy(state->lazy_pack) != SPARK_STATUS_OK )
			return;
		state->lazy_pack = 0;
	}
	if ( state->model != 0 )
		state->model_interface->close(state->model);
	state->model = 0;
	while ( state->layer_events_created != 0u )
		(void)cudaEventDestroy(state->layer_event[--state->layer_events_created]);
	free(state->layer_event);
	free(state->layer_host_ns);
	free(state->layer_gpu_ms);
	free(state->layer_enqueue_ms);
	free(state->layer_lag_ms);
	free(state->lease_identifier);
	free(state->lease_phase);
	free(state->lease_keys);
	free(state->group_offset_host);
	cudaFree(state->distribution_rows);
	cudaFree(state->distribution_positions);
	cudaFree(state->distribution_rules);
	cudaFree(state->distribution_logprobs);
	cudaFree(state->distribution_hidden);
	cudaFree(state->distribution_normed);
	cudaFree(state->distribution_logits);
	cudaFree(state->distribution_gathered);
	delete[] state->output_token_host;
	delete[] state->output_score_host;
	cudaFree(state->head_certified_fp8_payload);
	cudaFree(state->head_certified_fp8_scale_f32);
	cudaFree(state->head_certified_fp8_norm_f32);
	cudaFree(state->head_certified_scratch);
	cudaFree(state->head_certified_candidates);
	cudaFree(state->head_screened_count);
	cudaFree(state->hidden);
	cudaFree(state->head_normed);
	cudaFree(state->head_maxloc);
	cudaFree(state->head_hidden);
	cudaFree(state->head_token);
	cudaFree(state->head_score);
	cudaFree(state->head_candidate_token);
	cudaFree(state->head_candidate_score);
	cudaFree(state->output_token);
	cudaFree(state->output_score);
	cudaFree(state->positions);
	cudaFree(state->token_ids_device);
	cudaFree(state->context_length);
	cudaFree(state->sequence_of_row);
	cudaFree(state->recurrent_index);
	delete state;
	runner->private_state = 0;
}
