#include <cuda_runtime.h>

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_dsv4_model.h"
#include "sparkpipe/spark_dsv4_resident_decode_stage_firmware.h"

typedef struct SparkStageModuleCudaFork
{
	cudaStream_t auxiliary_streams[2u];
	cudaEvent_t fork_event;
	cudaEvent_t milestone_event;
	cudaEvent_t join_events[2u];
} SparkStageModuleCudaFork;

extern "C" SparkStatus SparkStageModuleCudaForkInitialize(const char *module_tag, SparkStageModuleCudaFork *fork);
extern "C" cudaError_t SparkStageModuleCudaForkBegin(SparkStageModuleCudaFork *fork, cudaStream_t primary_stream, uint32_t branch_count);
extern "C" cudaError_t SparkStageModuleCudaForkJoin(SparkStageModuleCudaFork *fork, cudaStream_t primary_stream, uint32_t branch_count);
extern "C" void SparkStageModuleCudaForkDestroy(SparkStageModuleCudaFork *fork);

#define SPARK_DSV4_VALIDATION_ROW_COUNT \
	(SPARK_BATCH_BUCKET < 8u ? SPARK_BATCH_BUCKET : 8u)
#define SPARK_DSV4_VALIDATION_REFERENCE_FIXTURE_ROW_COUNT 128u
#define SPARK_DSV4_VALIDATION_REFERENCE_ROW_COUNT \
	(SPARK_BATCH_BUCKET < SPARK_DSV4_VALIDATION_REFERENCE_FIXTURE_ROW_COUNT ? \
	SPARK_BATCH_BUCKET : SPARK_DSV4_VALIDATION_REFERENCE_FIXTURE_ROW_COUNT)
#define SPARK_DSV4_VALIDATION_REFERENCE_MAX_RELATIVE_L2 0.02
#define SPARK_DSV4_VALIDATION_REFERENCE_MIN_COSINE 0.999
#define SPARK_DSV4_VALIDATION_REFERENCE_MAX_ROW_RELATIVE_L2 0.075
#define SPARK_DSV4_VALIDATION_REFERENCE_MAX_SCALED_ROW_RELATIVE_L2 0.02
#define SPARK_DSV4_VALIDATION_REFERENCE_MIN_ROW_COSINE 0.995
#define SPARK_DSV4_VALIDATION_REFERENCE_MAX_ABSOLUTE 1.5
#define SPARK_DSV4_VALIDATION_HEAD_ROW_COUNT 8u
#define SPARK_DSV4_VALIDATION_HEAD_HIDDEN_DIMENSION 64u
#define SPARK_DSV4_VALIDATION_HEAD_SCREENED_VOCAB 257u
#define SPARK_DSV4_VALIDATION_HEAD_OVERFLOW_VOCAB (SPARK_DSV4_RESIDENT_DECODE_STAGE_HEAD_SCREEN_CAP + 17u)
#define SPARK_DSV4_VALIDATION_GATE_WEIGHT_FORMAT_BF16 0u
#define SPARK_DSV4_VALIDATION_GATE_W13_TILES \
	(SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION / 32u)
#define SPARK_DSV4_VALIDATION_GATE_W2_TILES \
	(SPARK_DSV4_MODEL_HIDDEN_DIMENSION / 128u)
#define SPARK_DSV4_VALIDATION_PRECOLLECTIVE_COLUMNS \
	(SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS + SPARK_DSV4_MODEL_INDEX_TOP_K)
#define SPARK_DSV4_VALIDATION_PRECOLLECTIVE_INDEX_CAPACITY 1024u
#define SPARK_DSV4_VALIDATION_PRECOLLECTIVE_SHADOW_BYTES 8192u
#define SPARK_DSV4_VALIDATION_POST_QUERY_HEAD_COUNT 16u
#define SPARK_DSV4_VALIDATION_POST_QUERY_ELEMENTS \
	(SPARK_DSV4_VALIDATION_POST_QUERY_HEAD_COUNT * \
	SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION)
#define SPARK_DSV4_VALIDATION_POST_INDEX_ELEMENTS \
	SPARK_DSV4_MODEL_INDEX_DIMENSION
#define SPARK_DSV4_VALIDATION_POST_ELEMENT_CAPACITY \
	(SPARK_DSV4_VALIDATION_POST_QUERY_ELEMENTS > \
	SPARK_DSV4_VALIDATION_POST_INDEX_ELEMENTS ? \
	SPARK_DSV4_VALIDATION_POST_QUERY_ELEMENTS : \
	SPARK_DSV4_VALIDATION_POST_INDEX_ELEMENTS)
#define SPARK_DSV4_VALIDATION_POST_FUSION_COUNT 3u
#define SPARK_DSV4_VALIDATION_POST_FREQUENCY_COUNT 2u

extern "C" cudaError_t SparkDsv4LaunchHeadShadowQuantize(cudaStream_t stream, const void *head_bf16, uint8_t *shadow_payload, uint8_t *shadow_scale, float *error_norm, uint32_t candidate_count, uint32_t hidden_dimension);
extern "C" cudaError_t SparkDsv4LaunchHeadScreenedArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint8_t *shadow_payload, const uint8_t *shadow_scale, const float *error_norm, void *logits_bf16, uint32_t *candidate_ids, uint32_t *candidate_counts, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count, uint32_t hidden_dimension);
extern "C" cudaError_t SparkDsv4LaunchHeadArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint32_t *token_ids, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count, uint32_t hidden_dimension);
extern "C" cudaError_t SparkDsv4LaunchGateRoute(cudaStream_t stream, const SparkDsv4LinearView *gate, const void *input_bf16, float *scores_f32, const float *bias_f32, const uint32_t *tid2eid_u32, const uint32_t *token_ids, uint32_t row_count, uint32_t expert_count, uint32_t topk, float route_scale, uint32_t *indices_u32, float *weights_f32, uint32_t expert_width, uint32_t *group_row_offset, uint32_t *route_packed_row, uint32_t *route_source_token, uint32_t *group_tile_prefix_w1, uint32_t *group_tile_prefix_w2);
extern "C" cudaError_t SparkDsv4LaunchLinear(cudaStream_t stream, const SparkDsv4LinearView *view, const void *input_bf16, void *output_bf16, uint32_t row_count);
extern "C" cudaError_t SparkDsv4LaunchWiden(cudaStream_t stream, const void *input_bf16, float *output_f32, uint32_t row_count, uint32_t width, float scale);
extern "C" cudaError_t SparkDsv4LaunchBuildAttentionIndices(cudaStream_t stream, const uint64_t *row_positions, int32_t *indices, uint32_t *slot_counts, uint32_t *attention_slot_counts, uint32_t row_count, uint32_t column_count, uint32_t index_slot_capacity, uint32_t layer_kind);
extern "C" cudaError_t SparkDsv4LaunchRmsNorm(cudaStream_t stream, const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon);
extern "C" cudaError_t SparkDsv4LaunchQuantSim(cudaStream_t stream, void *data_bf16, uint32_t row_count, uint32_t row_stride, uint32_t width, uint32_t block, uint32_t fp4);
extern "C" cudaError_t SparkDsv4LaunchRope(cudaStream_t stream, void *data_bf16, const float *freqs_f32, const uint64_t *row_positions, uint32_t row_count, uint32_t head_count, uint32_t head_dim, uint32_t rope_dim, uint32_t inverse);
extern "C" cudaError_t SparkDsv4LaunchQueryHeadRms(cudaStream_t stream, void *data_bf16, uint32_t row_count, uint32_t head_count, uint32_t head_dim, float epsilon);
extern "C" cudaError_t SparkDsv4LaunchQueryHeadRmsRope(cudaStream_t stream, void *data_bf16, const float *freqs_f32, const uint64_t *row_positions, uint32_t row_count, uint32_t head_count, uint32_t head_dim, uint32_t rope_dim, float epsilon);
extern "C" cudaError_t SparkDsv4LaunchKvPost(cudaStream_t stream, void *data_bf16, const void *gain_bf16, const float *freqs_f32, const uint64_t *row_positions, uint32_t row_count, uint32_t head_dim, uint32_t rope_dim, uint32_t quant_width, uint32_t quant_block, float epsilon);
extern "C" cudaError_t SparkDsv4LaunchIndexerPost(cudaStream_t stream, void *data_bf16, const float *freqs_f32, const uint64_t *row_positions, uint32_t row_count, uint32_t head_count, uint32_t head_dim, uint32_t rope_dim, uint32_t quant_block);
extern "C" cudaError_t SparkDsv4LaunchHadamard(cudaStream_t stream, void *data_bf16, uint32_t vector_count, uint32_t width);

typedef struct SparkDsv4ValidationCapture
{
	uint16_t hidden[SPARK_DSV4_VALIDATION_ROW_COUNT * SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS];
	uint32_t output_token_ids[SPARK_DSV4_VALIDATION_ROW_COUNT];
	uint32_t output_count;
	uint32_t nonzero_count;
	uint32_t completion_count;
	SparkModelDriverCompletion completion;
} SparkDsv4ValidationCapture;

typedef struct SparkDsv4ValidationFrame
{
	SparkDsv4DecodeBatchView batch;
	SparkDsv4ResidentDecodeStageFrameContext context;
	SparkModelDriverBuffer buffers[2];
	SparkModelDriverFrame frame;
	void *hidden_input_bf16;
	void *hidden_output_bf16;
	uint32_t token_ids[SPARK_DSV4_VALIDATION_ROW_COUNT];
	uint32_t lanes[SPARK_DSV4_VALIDATION_ROW_COUNT];
	uint64_t positions[SPARK_DSV4_VALIDATION_ROW_COUNT];
	uint64_t sequence_ids[SPARK_DSV4_VALIDATION_ROW_COUNT];
} SparkDsv4ValidationFrame;

typedef struct SparkDsv4ValidationHeadBuffers
{
	void *hidden_bf16;
	void *head_bf16;
	uint8_t *shadow_payload;
	uint8_t *shadow_scale;
	float *error_norm;
	void *logits_bf16;
	uint32_t *candidate_ids;
	uint32_t *candidate_counts;
	uint32_t *screened_output;
	uint32_t *reference_output;
} SparkDsv4ValidationHeadBuffers;

typedef struct SparkDsv4ValidationGateRouteBuffers
{
	void *weight_bf16;
	void *input_bf16;
	float *scores_f32;
	float *bias_f32;
	float *weights_f32;
	uint32_t *tid2eid_u32;
	uint32_t *token_ids;
	uint32_t *indices_u32;
	uint32_t *group_row_offset;
	uint32_t *route_packed_row;
	uint32_t *route_source_token;
	uint32_t *group_tile_prefix_w1;
	uint32_t *group_tile_prefix_w2;
} SparkDsv4ValidationGateRouteBuffers;

typedef struct SparkDsv4ValidationPrecollectiveBuffers
{
	void *input_bf16;
	void *weight_bf16;
	void *control_weight_bf16;
	void *candidate_weight_bf16;
	float *control_weight_f32;
	float *candidate_weight_f32;
	uint64_t *row_position;
	int32_t *control_indices;
	int32_t *candidate_indices;
	uint32_t *control_slot_count;
	uint32_t *candidate_slot_count;
	uint32_t *control_attention_count;
	uint32_t *candidate_attention_count;
	uint8_t *collective_shadow;
	SparkStageModuleCudaFork fork;
} SparkDsv4ValidationPrecollectiveBuffers;

typedef struct SparkDsv4ValidationPostBuffers
{
	void *control_bf16;
	void *candidate_bf16;
	void *gain_bf16;
	float *freqs_f32;
	uint64_t *position_u64;
	uint16_t *input;
	uint16_t *control;
	uint16_t *candidate;
	uint16_t *gain;
} SparkDsv4ValidationPostBuffers;

typedef struct SparkDsv4ValidationReferenceFrame
{
	SparkDsv4PrefillBatchView batch;
	SparkDsv4ResidentDecodeStageFrameContext context;
	SparkModelDriverBuffer buffers[1];
	SparkModelDriverFrame frame;
	void *hidden_output_bf16;
	uint32_t token_ids[SPARK_DSV4_VALIDATION_REFERENCE_ROW_COUNT];
	uint32_t lanes[SPARK_DSV4_VALIDATION_REFERENCE_ROW_COUNT];
	uint64_t positions[SPARK_DSV4_VALIDATION_REFERENCE_ROW_COUNT];
	uint64_t sequence_ids[SPARK_DSV4_VALIDATION_REFERENCE_ROW_COUNT];
	uint32_t completion_count;
	SparkModelDriverCompletion completion;
} SparkDsv4ValidationReferenceFrame;

typedef struct SparkDsv4ValidationReferenceMetrics
{
	double actual_l2;
	double difference_l2;
	double dot;
	double reference_l2;
	double relative_l2;
	double cosine;
	double worst_row_relative_l2;
	double worst_row_scaled_relative_l2;
	double worst_row_cosine;
	double worst_row_difference_l2;
	double worst_row_reference_l2;
	double maximum_absolute;
	uint64_t nonfinite;
	uint32_t worst_row_relative_l2_index;
	uint32_t worst_row_scaled_relative_l2_index;
	uint32_t worst_row_cosine_index;
} SparkDsv4ValidationReferenceMetrics;

typedef struct SparkDsv4ValidationMode
{
	const char *reference_path;
	const char *reference_token_path;
	uint32_t active_slot_count;
	uint32_t new_token_count;
	uint32_t frame_flags;
	int32_t use_reference;
} SparkDsv4ValidationMode;

static int SparkDsv4ValidationRequire(int condition,const char *message)
{
	if ( condition != 0 )
		return(0);
	fprintf(stderr,"dsv4_validation failure=%s\n",message);
	return(1);
}

static int32_t SparkDsv4ValidationRequireCuda(cudaError_t error,const char *message)
{
	if ( error == cudaSuccess )
		return(0);
	fprintf(stderr,"dsv4_validation failure=%s cuda=%s\n",message,cudaGetErrorString(error));
	return(1);
}

static void SparkDsv4ValidationHeadDestroy(SparkDsv4ValidationHeadBuffers *buffers)
{
	if ( buffers->hidden_bf16 != 0 )
		cudaFree(buffers->hidden_bf16);
	if ( buffers->head_bf16 != 0 )
		cudaFree(buffers->head_bf16);
	if ( buffers->shadow_payload != 0 )
		cudaFree(buffers->shadow_payload);
	if ( buffers->shadow_scale != 0 )
		cudaFree(buffers->shadow_scale);
	if ( buffers->error_norm != 0 )
		cudaFree(buffers->error_norm);
	if ( buffers->logits_bf16 != 0 )
		cudaFree(buffers->logits_bf16);
	if ( buffers->candidate_ids != 0 )
		cudaFree(buffers->candidate_ids);
	if ( buffers->candidate_counts != 0 )
		cudaFree(buffers->candidate_counts);
	if ( buffers->screened_output != 0 )
		cudaFree(buffers->screened_output);
	if ( buffers->reference_output != 0 )
		cudaFree(buffers->reference_output);
	memset(buffers,0,sizeof(*buffers));
}

static int SparkDsv4ValidationHeadAllocate(SparkDsv4ValidationHeadBuffers *buffers)
{
	uint64_t head_elements = (uint64_t)SPARK_DSV4_VALIDATION_HEAD_OVERFLOW_VOCAB * SPARK_DSV4_VALIDATION_HEAD_HIDDEN_DIMENSION;
	cudaError_t error;
	memset(buffers,0,sizeof(*buffers));
	error = cudaMalloc(&buffers->hidden_bf16,SPARK_DSV4_VALIDATION_HEAD_ROW_COUNT * SPARK_DSV4_VALIDATION_HEAD_HIDDEN_DIMENSION * sizeof(uint16_t));
	if ( error == cudaSuccess ) error = cudaMalloc(&buffers->head_bf16,head_elements * sizeof(uint16_t));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&buffers->shadow_payload,head_elements / 2u);
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&buffers->shadow_scale,SPARK_DSV4_VALIDATION_HEAD_OVERFLOW_VOCAB * (SPARK_DSV4_VALIDATION_HEAD_HIDDEN_DIMENSION / 32u));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&buffers->error_norm,SPARK_DSV4_VALIDATION_HEAD_OVERFLOW_VOCAB * sizeof(float));
	if ( error == cudaSuccess ) error = cudaMalloc(&buffers->logits_bf16,SPARK_DSV4_VALIDATION_HEAD_ROW_COUNT * SPARK_DSV4_VALIDATION_HEAD_OVERFLOW_VOCAB * sizeof(uint16_t));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&buffers->candidate_ids,SPARK_DSV4_VALIDATION_HEAD_ROW_COUNT * SPARK_DSV4_RESIDENT_DECODE_STAGE_HEAD_SCREEN_CAP * sizeof(uint32_t));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&buffers->candidate_counts,SPARK_DSV4_VALIDATION_HEAD_ROW_COUNT * sizeof(uint32_t));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&buffers->screened_output,SPARK_DSV4_VALIDATION_HEAD_ROW_COUNT * sizeof(uint32_t));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&buffers->reference_output,SPARK_DSV4_VALIDATION_HEAD_ROW_COUNT * sizeof(uint32_t));
	return(error == cudaSuccess ? 0 : 1);
}

static uint16_t SparkDsv4ValidationHeadRandomBf16(uint32_t *state)
{
	uint16_t value;
	*state = (*state * UINT32_C(1664525)) + UINT32_C(1013904223);
	value = (uint16_t)(UINT16_C(0x3c00) + ((*state >> 8u) & UINT32_C(0x03ff)));
	return((uint16_t)(value | ((*state & 1u) != 0u ? UINT16_C(0x8000) : 0u)));
}

static void SparkDsv4ValidationHeadInputs(uint16_t *hidden, uint16_t *head, uint32_t *expected, uint32_t candidate_count, uint32_t random)
{
	uint64_t element;
	uint32_t far,primary,row,state = UINT32_C(0x5a17c9e3) ^ (random * UINT32_C(0x9e3779b9));
	memset(hidden,0,SPARK_DSV4_VALIDATION_HEAD_ROW_COUNT * SPARK_DSV4_VALIDATION_HEAD_HIDDEN_DIMENSION * sizeof(uint16_t));
	memset(head,0,(uint64_t)candidate_count * SPARK_DSV4_VALIDATION_HEAD_HIDDEN_DIMENSION * sizeof(uint16_t));
	if ( random != 0u )
	{
		for (element=0u; element<SPARK_DSV4_VALIDATION_HEAD_ROW_COUNT * SPARK_DSV4_VALIDATION_HEAD_HIDDEN_DIMENSION; element++) hidden[element] = SparkDsv4ValidationHeadRandomBf16(&state);
		for (element=0u; element<(uint64_t)candidate_count * SPARK_DSV4_VALIDATION_HEAD_HIDDEN_DIMENSION; element++) head[element] = SparkDsv4ValidationHeadRandomBf16(&state);
		return;
	}
	for (row=0u; row<SPARK_DSV4_VALIDATION_HEAD_ROW_COUNT; row++)
	{
		primary = 17u + row;
		far = candidate_count - 1u - row;
		hidden[(row * SPARK_DSV4_VALIDATION_HEAD_HIDDEN_DIMENSION) + row] = UINT16_C(0x3f80);
		head[((uint64_t)primary * SPARK_DSV4_VALIDATION_HEAD_HIDDEN_DIMENSION) + row] = UINT16_C(0x4100);
		head[((uint64_t)far * SPARK_DSV4_VALIDATION_HEAD_HIDDEN_DIMENSION) + row] = (row & 1u) == 0u ? UINT16_C(0x4100) : UINT16_C(0x4140);
		expected[row] = (row & 1u) == 0u ? primary : far;
	}
}

static uint32_t SparkDsv4ValidationHeadExpectedCount(uint32_t row, uint32_t row_count, uint32_t candidate_count, uint32_t overflow)
{
	if ( row_count == 1u )
		return(UINT32_MAX);
	if ( overflow != 0u )
		return(candidate_count);
	return((row & 1u) == 0u ? 2u : 1u);
}

static int SparkDsv4ValidationHeadRunCase(SparkDsv4ValidationHeadBuffers *buffers, uint16_t *head, float *errors, uint32_t row_count, uint32_t candidate_count, uint32_t overflow, uint32_t random)
{
	uint16_t hidden[SPARK_DSV4_VALIDATION_HEAD_ROW_COUNT * SPARK_DSV4_VALIDATION_HEAD_HIDDEN_DIMENSION];
	uint32_t counts[SPARK_DSV4_VALIDATION_HEAD_ROW_COUNT],expected[SPARK_DSV4_VALIDATION_HEAD_ROW_COUNT],reference[SPARK_DSV4_VALIDATION_HEAD_ROW_COUNT],screened[SPARK_DSV4_VALIDATION_HEAD_ROW_COUNT];
	uint32_t index,row;
	cudaError_t error;
	SparkDsv4ValidationHeadInputs(hidden,head,expected,candidate_count,random);
	error = cudaMemcpy(buffers->hidden_bf16,hidden,sizeof(hidden),cudaMemcpyHostToDevice);
	if ( error == cudaSuccess ) error = cudaMemcpy(buffers->head_bf16,head,(uint64_t)candidate_count * SPARK_DSV4_VALIDATION_HEAD_HIDDEN_DIMENSION * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if ( error == cudaSuccess ) error = SparkDsv4LaunchHeadShadowQuantize(cudaStreamPerThread,buffers->head_bf16,buffers->shadow_payload,buffers->shadow_scale,buffers->error_norm,candidate_count,SPARK_DSV4_VALIDATION_HEAD_HIDDEN_DIMENSION);
	if ( overflow != 0u )
	{
		for (index=0u; index<candidate_count; index++) errors[index] = 1.0e6f;
		if ( error == cudaSuccess ) error = cudaMemcpyAsync(buffers->error_norm,errors,(uint64_t)candidate_count * sizeof(float),cudaMemcpyHostToDevice,cudaStreamPerThread);
	}
	if ( error == cudaSuccess ) error = cudaMemsetAsync(buffers->screened_output,0xff,SPARK_DSV4_VALIDATION_HEAD_ROW_COUNT * sizeof(uint32_t),cudaStreamPerThread);
	if ( error == cudaSuccess ) error = cudaMemsetAsync(buffers->reference_output,0xff,SPARK_DSV4_VALIDATION_HEAD_ROW_COUNT * sizeof(uint32_t),cudaStreamPerThread);
	if ( error == cudaSuccess ) error = SparkDsv4LaunchHeadScreenedArgmax(cudaStreamPerThread,buffers->hidden_bf16,buffers->head_bf16,buffers->shadow_payload,buffers->shadow_scale,buffers->error_norm,buffers->logits_bf16,buffers->candidate_ids,buffers->candidate_counts,buffers->screened_output,row_count,candidate_count,SPARK_DSV4_VALIDATION_HEAD_HIDDEN_DIMENSION);
	if ( error == cudaSuccess ) error = cudaMemcpyAsync(counts,buffers->candidate_counts,sizeof(counts),cudaMemcpyDeviceToHost,cudaStreamPerThread);
	if ( error == cudaSuccess ) error = cudaStreamSynchronize(cudaStreamPerThread);
	if ( error == cudaSuccess && overflow != 0u && row_count > 1u )
	{
		for (row=0u; row<row_count && error == cudaSuccess; row++)
			error = SparkDsv4LaunchHeadScreenedArgmax(cudaStreamPerThread,((const uint16_t *)buffers->hidden_bf16) + ((uint64_t)row * SPARK_DSV4_VALIDATION_HEAD_HIDDEN_DIMENSION),buffers->head_bf16,buffers->shadow_payload,buffers->shadow_scale,buffers->error_norm,buffers->logits_bf16,buffers->candidate_ids,buffers->candidate_counts,buffers->reference_output + row,1u,candidate_count,SPARK_DSV4_VALIDATION_HEAD_HIDDEN_DIMENSION);
	}
	else if ( error == cudaSuccess )
		error = SparkDsv4LaunchHeadArgmax(cudaStreamPerThread,buffers->hidden_bf16,buffers->head_bf16,0,buffers->reference_output,row_count,candidate_count,SPARK_DSV4_VALIDATION_HEAD_HIDDEN_DIMENSION);
	if ( error == cudaSuccess ) error = cudaStreamSynchronize(cudaStreamPerThread);
	if ( error == cudaSuccess ) error = cudaMemcpy(reference,buffers->reference_output,sizeof(reference),cudaMemcpyDeviceToHost);
	if ( error == cudaSuccess ) error = cudaMemcpy(screened,buffers->screened_output,sizeof(screened),cudaMemcpyDeviceToHost);
	if ( SparkDsv4ValidationRequireCuda(error,"head_cuda") != 0 ) return(1);
	for (row=0u; row<row_count; row++)
	{
		if ( SparkDsv4ValidationRequire(counts[row] == SparkDsv4ValidationHeadExpectedCount(row,row_count,candidate_count,overflow),"head_screen_branch") != 0 ) return(1);
		if ( random == 0u && SparkDsv4ValidationRequire(screened[row] == expected[row],"head_expected_tokens") != 0 ) return(1);
		if ( SparkDsv4ValidationRequire(screened[row] == reference[row],"head_reference_parity") != 0 ) return(1);
	}
	return(0);
}

static int SparkDsv4ValidationHead(void)
{
	SparkDsv4ValidationHeadBuffers buffers;
	uint16_t *head;
	float *errors;
	int result;
	uint32_t seed;
	memset(&buffers,0,sizeof(buffers));
	head = (uint16_t *)calloc((uint64_t)SPARK_DSV4_VALIDATION_HEAD_OVERFLOW_VOCAB * SPARK_DSV4_VALIDATION_HEAD_HIDDEN_DIMENSION,sizeof(uint16_t));
	errors = (float *)calloc(SPARK_DSV4_VALIDATION_HEAD_OVERFLOW_VOCAB,sizeof(float));
	result = head == 0 || errors == 0 || SparkDsv4ValidationHeadAllocate(&buffers) != 0;
	if ( result == 0 ) result = SparkDsv4ValidationHeadRunCase(&buffers,head,errors,SPARK_DSV4_VALIDATION_HEAD_ROW_COUNT,SPARK_DSV4_VALIDATION_HEAD_SCREENED_VOCAB,0u,0u);
	if ( result == 0 ) result = SparkDsv4ValidationHeadRunCase(&buffers,head,errors,1u,SPARK_DSV4_VALIDATION_HEAD_OVERFLOW_VOCAB,1u,0u);
	if ( result == 0 ) result = SparkDsv4ValidationHeadRunCase(&buffers,head,errors,SPARK_DSV4_VALIDATION_HEAD_ROW_COUNT,SPARK_DSV4_VALIDATION_HEAD_OVERFLOW_VOCAB,1u,0u);
	for (seed=1u; seed<=4u && result == 0; seed++) result = SparkDsv4ValidationHeadRunCase(&buffers,head,errors,SPARK_DSV4_VALIDATION_HEAD_ROW_COUNT,SPARK_DSV4_VALIDATION_HEAD_OVERFLOW_VOCAB,1u,seed);
	SparkDsv4ValidationHeadDestroy(&buffers);
	free(errors);
	free(head);
	return(result);
}

static void SparkDsv4ValidationGateRouteDestroy(
	SparkDsv4ValidationGateRouteBuffers *buffers)
{
	if ( buffers->weight_bf16 != 0 ) cudaFree(buffers->weight_bf16);
	if ( buffers->input_bf16 != 0 ) cudaFree(buffers->input_bf16);
	if ( buffers->scores_f32 != 0 ) cudaFree(buffers->scores_f32);
	if ( buffers->bias_f32 != 0 ) cudaFree(buffers->bias_f32);
	if ( buffers->weights_f32 != 0 ) cudaFree(buffers->weights_f32);
	if ( buffers->tid2eid_u32 != 0 ) cudaFree(buffers->tid2eid_u32);
	if ( buffers->token_ids != 0 ) cudaFree(buffers->token_ids);
	if ( buffers->indices_u32 != 0 ) cudaFree(buffers->indices_u32);
	if ( buffers->group_row_offset != 0 ) cudaFree(buffers->group_row_offset);
	if ( buffers->route_packed_row != 0 ) cudaFree(buffers->route_packed_row);
	if ( buffers->route_source_token != 0 ) cudaFree(buffers->route_source_token);
	if ( buffers->group_tile_prefix_w1 != 0 ) cudaFree(buffers->group_tile_prefix_w1);
	if ( buffers->group_tile_prefix_w2 != 0 ) cudaFree(buffers->group_tile_prefix_w2);
	memset(buffers,0,sizeof(*buffers));
}

static int SparkDsv4ValidationGateRouteAllocate(
	SparkDsv4ValidationGateRouteBuffers *buffers)
{
	uint64_t gate_elements = (uint64_t)SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT *
		SPARK_DSV4_MODEL_HIDDEN_DIMENSION;
	uint32_t group_count = SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT + 1u;
	uint32_t topk = SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN;
	cudaError_t error;
	memset(buffers,0,sizeof(*buffers));
	error = cudaMalloc(&buffers->weight_bf16,gate_elements * sizeof(uint16_t));
	if ( error == cudaSuccess ) error = cudaMalloc(&buffers->input_bf16,SPARK_DSV4_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&buffers->scores_f32,SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT * sizeof(float));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&buffers->bias_f32,SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT * sizeof(float));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&buffers->weights_f32,topk * sizeof(float));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&buffers->tid2eid_u32,topk * sizeof(uint32_t));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&buffers->token_ids,sizeof(uint32_t));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&buffers->indices_u32,topk * sizeof(uint32_t));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&buffers->group_row_offset,group_count * sizeof(uint32_t));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&buffers->route_packed_row,topk * sizeof(uint32_t));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&buffers->route_source_token,topk * sizeof(uint32_t));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&buffers->group_tile_prefix_w1,group_count * sizeof(uint32_t));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&buffers->group_tile_prefix_w2,group_count * sizeof(uint32_t));
	if ( error == cudaSuccess ) error = cudaMemset(buffers->weight_bf16,0,gate_elements * sizeof(uint16_t));
	if ( error == cudaSuccess ) error = cudaMemset(buffers->input_bf16,0,SPARK_DSV4_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t));
	return(error == cudaSuccess ? 0 : 1);
}

static uint32_t SparkDsv4ValidationGateRouteContains(
	const uint32_t *expected,uint32_t expert)
{
	uint32_t rank;
	for (rank=0u; rank<SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN; rank++)
		if ( expected[rank] == expert ) return(1u);
	return(0u);
}

static int SparkDsv4ValidationGateRouteRead(
	const SparkDsv4ValidationGateRouteBuffers *buffers,
	float *scores,float *weights,uint32_t *indices,uint32_t *offsets,
	uint32_t *packed,uint32_t *source,uint32_t *prefix_w1,
	uint32_t *prefix_w2)
{
	uint32_t experts = SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT;
	uint32_t groups = experts + 1u,topk = SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN;
	cudaError_t error = cudaMemcpy(scores,buffers->scores_f32,experts * sizeof(float),cudaMemcpyDeviceToHost);
	if ( error == cudaSuccess ) error = cudaMemcpy(weights,buffers->weights_f32,topk * sizeof(float),cudaMemcpyDeviceToHost);
	if ( error == cudaSuccess ) error = cudaMemcpy(indices,buffers->indices_u32,topk * sizeof(uint32_t),cudaMemcpyDeviceToHost);
	if ( error == cudaSuccess ) error = cudaMemcpy(offsets,buffers->group_row_offset,groups * sizeof(uint32_t),cudaMemcpyDeviceToHost);
	if ( error == cudaSuccess ) error = cudaMemcpy(packed,buffers->route_packed_row,topk * sizeof(uint32_t),cudaMemcpyDeviceToHost);
	if ( error == cudaSuccess ) error = cudaMemcpy(source,buffers->route_source_token,topk * sizeof(uint32_t),cudaMemcpyDeviceToHost);
	if ( error == cudaSuccess ) error = cudaMemcpy(prefix_w1,buffers->group_tile_prefix_w1,groups * sizeof(uint32_t),cudaMemcpyDeviceToHost);
	if ( error == cudaSuccess ) error = cudaMemcpy(prefix_w2,buffers->group_tile_prefix_w2,groups * sizeof(uint32_t),cudaMemcpyDeviceToHost);
	return(SparkDsv4ValidationRequireCuda(error,"gate_route_read"));
}

static int SparkDsv4ValidationGateRouteCheckSelection(
	const float *scores,const float *weights,const uint32_t *indices,
	const uint32_t *expected)
{
	uint32_t expert,rank;
	float expected_weight;
	if ( SparkDsv4ValidationRequire(isfinite(scores[0]) && scores[0] > 0.0f,"gate_route_score_finite") != 0 ) return(1);
	for (expert=1u; expert<SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT; expert++)
		if ( SparkDsv4ValidationRequire(scores[expert] == scores[0],"gate_route_score_exact") != 0 ) return(1);
	for (rank=0u; rank<SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN; rank++)
	{
		if ( SparkDsv4ValidationRequire(indices[rank] == expected[rank],"gate_route_selection") != 0 ) return(1);
		expected_weight = SPARK_DSV4_MODEL_ROUTED_SCALING_FACTOR /
			(float)SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN;
		if ( SparkDsv4ValidationRequire(fabsf(weights[rank] - expected_weight) <= 1.0e-6f,"gate_route_weight") != 0 ) return(1);
	}
	return(0);
}

static int SparkDsv4ValidationGateRouteCheckLayout(
	const uint32_t *offsets,const uint32_t *packed,const uint32_t *source,
	const uint32_t *prefix_w1,const uint32_t *prefix_w2,
	const uint32_t *expected)
{
	uint32_t expert,held = 0u,lower,rank;
	for (expert=0u; expert<SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT; expert++)
	{
		if ( SparkDsv4ValidationRequire(offsets[expert] == held,"gate_route_group_offset") != 0 ) return(1);
		if ( SparkDsv4ValidationRequire(prefix_w1[expert] == held * SPARK_DSV4_VALIDATION_GATE_W13_TILES,"gate_route_w13_prefix") != 0 ) return(1);
		if ( SparkDsv4ValidationRequire(prefix_w2[expert] == held * SPARK_DSV4_VALIDATION_GATE_W2_TILES,"gate_route_w2_prefix") != 0 ) return(1);
		held += SparkDsv4ValidationGateRouteContains(expected,expert);
	}
	if ( SparkDsv4ValidationRequire(offsets[expert] == held && prefix_w1[expert] == held * SPARK_DSV4_VALIDATION_GATE_W13_TILES && prefix_w2[expert] == held * SPARK_DSV4_VALIDATION_GATE_W2_TILES,"gate_route_terminal_prefix") != 0 ) return(1);
	for (rank=0u; rank<SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN; rank++)
	{
		lower = 0u;
		for (expert=0u; expert<SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN; expert++) lower += expected[expert] < expected[rank] ? 1u : 0u;
		if ( SparkDsv4ValidationRequire(packed[rank] == lower,"gate_route_packed_row") != 0 ) return(1);
		if ( SparkDsv4ValidationRequire(source[rank] == 0u,"gate_route_source_token") != 0 ) return(1);
	}
	return(0);
}

static int SparkDsv4ValidationGateRouteRunCase(
	SparkDsv4ValidationGateRouteBuffers *buffers,uint32_t hash_routed)
{
	static const uint32_t score_expected[SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN] = {2u,4u,1u,3u,7u,8u};
	static const uint32_t hash_expected[SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN] = {9u,2u,200u,1u,4u,3u};
	float bias[SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT] = {0.0f},scores[SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT],weights[SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN];
	uint32_t indices[SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN],offsets[SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT + 1u],packed[SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN],source[SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN],prefix_w1[SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT + 1u],prefix_w2[SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT + 1u],token = 0u;
	const uint32_t *expected = hash_routed != 0u ? hash_expected : score_expected;
	SparkDsv4LinearView gate = {SPARK_DSV4_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION,SPARK_DSV4_VALIDATION_GATE_WEIGHT_FORMAT_BF16,SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT,SPARK_DSV4_MODEL_HIDDEN_DIMENSION,buffers->weight_bf16,0};
	cudaError_t error;
	bias[2] = 6.0f; bias[4] = 5.0f; bias[1] = 4.0f; bias[3] = 4.0f; bias[7] = 3.0f; bias[8] = 3.0f;
	error = cudaMemcpy(buffers->bias_f32,bias,sizeof(bias),cudaMemcpyHostToDevice);
	if ( error == cudaSuccess ) error = cudaMemcpy(buffers->tid2eid_u32,hash_expected,sizeof(hash_expected),cudaMemcpyHostToDevice);
	if ( error == cudaSuccess ) error = cudaMemcpy(buffers->token_ids,&token,sizeof(token),cudaMemcpyHostToDevice);
	if ( error == cudaSuccess ) error = SparkDsv4LaunchGateRoute(cudaStreamPerThread,&gate,buffers->input_bf16,buffers->scores_f32,hash_routed != 0u ? 0 : buffers->bias_f32,hash_routed != 0u ? buffers->tid2eid_u32 : 0,buffers->token_ids,1u,SPARK_DSV4_MODEL_ROUTED_EXPERT_COUNT,SPARK_DSV4_MODEL_EXPERTS_PER_TOKEN,SPARK_DSV4_MODEL_ROUTED_SCALING_FACTOR,buffers->indices_u32,buffers->weights_f32,SPARK_DSV4_MODEL_EXPERT_INTERMEDIATE_DIMENSION,buffers->group_row_offset,buffers->route_packed_row,buffers->route_source_token,buffers->group_tile_prefix_w1,buffers->group_tile_prefix_w2);
	if ( error == cudaSuccess ) error = cudaStreamSynchronize(cudaStreamPerThread);
	if ( SparkDsv4ValidationRequireCuda(error,"gate_route_launch") != 0 ) return(1);
	if ( SparkDsv4ValidationGateRouteRead(buffers,scores,weights,indices,offsets,packed,source,prefix_w1,prefix_w2) != 0 ) return(1);
	if ( SparkDsv4ValidationGateRouteCheckSelection(scores,weights,indices,expected) != 0 ) return(1);
	return(SparkDsv4ValidationGateRouteCheckLayout(offsets,packed,source,prefix_w1,prefix_w2,expected));
}

static int SparkDsv4ValidationGateRouteB1(void)
{
	SparkDsv4ValidationGateRouteBuffers buffers;
	int result;
	memset(&buffers,0,sizeof(buffers));
	result = SparkDsv4ValidationGateRouteAllocate(&buffers);
	if ( result == 0 ) result = SparkDsv4ValidationGateRouteRunCase(&buffers,0u);
	if ( result == 0 ) result = SparkDsv4ValidationGateRouteRunCase(&buffers,1u);
	SparkDsv4ValidationGateRouteDestroy(&buffers);
	return(result);
}

static void SparkDsv4ValidationPrecollectiveDestroy(
	SparkDsv4ValidationPrecollectiveBuffers *buffers)
{
	SparkStageModuleCudaForkDestroy(&buffers->fork);
	if ( buffers->input_bf16 != 0 ) cudaFree(buffers->input_bf16);
	if ( buffers->weight_bf16 != 0 ) cudaFree(buffers->weight_bf16);
	if ( buffers->control_weight_bf16 != 0 ) cudaFree(buffers->control_weight_bf16);
	if ( buffers->candidate_weight_bf16 != 0 ) cudaFree(buffers->candidate_weight_bf16);
	if ( buffers->control_weight_f32 != 0 ) cudaFree(buffers->control_weight_f32);
	if ( buffers->candidate_weight_f32 != 0 ) cudaFree(buffers->candidate_weight_f32);
	if ( buffers->row_position != 0 ) cudaFree(buffers->row_position);
	if ( buffers->control_indices != 0 ) cudaFree(buffers->control_indices);
	if ( buffers->candidate_indices != 0 ) cudaFree(buffers->candidate_indices);
	if ( buffers->control_slot_count != 0 ) cudaFree(buffers->control_slot_count);
	if ( buffers->candidate_slot_count != 0 ) cudaFree(buffers->candidate_slot_count);
	if ( buffers->control_attention_count != 0 ) cudaFree(buffers->control_attention_count);
	if ( buffers->candidate_attention_count != 0 ) cudaFree(buffers->candidate_attention_count);
	if ( buffers->collective_shadow != 0 ) cudaFree(buffers->collective_shadow);
	memset(buffers,0,sizeof(*buffers));
}

static cudaError_t SparkDsv4ValidationPrecollectiveDeviceAllocate(
	SparkDsv4ValidationPrecollectiveBuffers *buffers)
{
	uint64_t weight_bytes = (uint64_t)SPARK_DSV4_MODEL_INDEX_HEAD_COUNT *
		SPARK_DSV4_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t);
	uint64_t index_bytes = (uint64_t)SPARK_DSV4_VALIDATION_PRECOLLECTIVE_COLUMNS *
		sizeof(int32_t);
	cudaError_t error = cudaMalloc(&buffers->input_bf16,
		SPARK_DSV4_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t));
	if ( error == cudaSuccess ) error = cudaMalloc(&buffers->weight_bf16,weight_bytes);
	if ( error == cudaSuccess ) error = cudaMalloc(&buffers->control_weight_bf16,SPARK_DSV4_MODEL_INDEX_HEAD_COUNT * sizeof(uint16_t));
	if ( error == cudaSuccess ) error = cudaMalloc(&buffers->candidate_weight_bf16,SPARK_DSV4_MODEL_INDEX_HEAD_COUNT * sizeof(uint16_t));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&buffers->control_weight_f32,SPARK_DSV4_MODEL_INDEX_HEAD_COUNT * sizeof(float));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&buffers->candidate_weight_f32,SPARK_DSV4_MODEL_INDEX_HEAD_COUNT * sizeof(float));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&buffers->row_position,sizeof(uint64_t));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&buffers->control_indices,index_bytes);
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&buffers->candidate_indices,index_bytes);
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&buffers->control_slot_count,sizeof(uint32_t));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&buffers->candidate_slot_count,sizeof(uint32_t));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&buffers->control_attention_count,sizeof(uint32_t));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&buffers->candidate_attention_count,sizeof(uint32_t));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&buffers->collective_shadow,SPARK_DSV4_VALIDATION_PRECOLLECTIVE_SHADOW_BYTES);
	return(error);
}

static int SparkDsv4ValidationPrecollectiveAllocate(
	SparkDsv4ValidationPrecollectiveBuffers *buffers)
{
	uint64_t weight_count = (uint64_t)SPARK_DSV4_MODEL_INDEX_HEAD_COUNT *
		SPARK_DSV4_MODEL_HIDDEN_DIMENSION;
	uint16_t *input = (uint16_t *)calloc(SPARK_DSV4_MODEL_HIDDEN_DIMENSION,sizeof(uint16_t));
	uint16_t *weight = (uint16_t *)calloc(weight_count,sizeof(uint16_t));
	uint64_t position = 4095u,index;
	cudaError_t error;
	SparkStatus status = SPARK_STATUS_OK;
	memset(buffers,0,sizeof(*buffers));
	if ( input == 0 || weight == 0 )
	{
		free(weight);
		free(input);
		return(1);
	}
	for (index=0u; index<SPARK_DSV4_MODEL_HIDDEN_DIMENSION; index++) input[index] = 0x3f80u;
	for (index=0u; index<weight_count; index++) weight[index] = 0x3a80u;
	error = SparkDsv4ValidationPrecollectiveDeviceAllocate(buffers);
	if ( error == cudaSuccess ) error = cudaMemcpy(buffers->input_bf16,input,SPARK_DSV4_MODEL_HIDDEN_DIMENSION * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if ( error == cudaSuccess ) error = cudaMemcpy(buffers->weight_bf16,weight,weight_count * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if ( error == cudaSuccess ) error = cudaMemcpy(buffers->row_position,&position,sizeof(position),cudaMemcpyHostToDevice);
	if ( error == cudaSuccess ) status = SparkStageModuleCudaForkInitialize("dsv4_validation",&buffers->fork);
	free(weight);
	free(input);
	if ( SparkDsv4ValidationRequireCuda(error,"projection_precollective_allocate") != 0 ) return(1);
	return(SparkDsv4ValidationRequire(status == SPARK_STATUS_OK,"projection_precollective_fork_initialize"));
}

static cudaError_t SparkDsv4ValidationPrecollectiveLaunch(
	const SparkDsv4ValidationPrecollectiveBuffers *buffers,cudaStream_t stream,
	void *weight_bf16,float *weight_f32,int32_t *indices,
	uint32_t *slot_count,uint32_t *attention_count)
{
	SparkDsv4LinearView projection = {SPARK_DSV4_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION,SPARK_DSV4_VALIDATION_GATE_WEIGHT_FORMAT_BF16,SPARK_DSV4_MODEL_INDEX_HEAD_COUNT,SPARK_DSV4_MODEL_HIDDEN_DIMENSION,buffers->weight_bf16,0};
	float scale = 1.0f / sqrtf((float)SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION) /
		sqrtf((float)SPARK_DSV4_MODEL_INDEX_HEAD_COUNT);
	cudaError_t error = SparkDsv4LaunchLinear(stream,&projection,buffers->input_bf16,
		weight_bf16,1u);
	if ( error == cudaSuccess ) error = SparkDsv4LaunchWiden(stream,weight_bf16,
		weight_f32,1u,SPARK_DSV4_MODEL_INDEX_HEAD_COUNT,scale);
	if ( error == cudaSuccess ) error = SparkDsv4LaunchBuildAttentionIndices(stream,
		buffers->row_position,indices,slot_count,attention_count,1u,
		SPARK_DSV4_VALIDATION_PRECOLLECTIVE_COLUMNS,
		SPARK_DSV4_VALIDATION_PRECOLLECTIVE_INDEX_CAPACITY,
		SPARK_DSV4_MODEL_LAYER_KIND_CSA);
	return(error);
}

static int SparkDsv4ValidationPrecollectiveRunControl(
	const SparkDsv4ValidationPrecollectiveBuffers *buffers)
{
	cudaError_t error = SparkDsv4ValidationPrecollectiveLaunch(buffers,
		cudaStreamPerThread,buffers->control_weight_bf16,
		buffers->control_weight_f32,buffers->control_indices,
		buffers->control_slot_count,buffers->control_attention_count);
	if ( error == cudaSuccess ) error = cudaStreamSynchronize(cudaStreamPerThread);
	return(SparkDsv4ValidationRequireCuda(error,"projection_precollective_control"));
}

static int SparkDsv4ValidationPrecollectiveRunCandidate(
	SparkDsv4ValidationPrecollectiveBuffers *buffers)
{
	cudaError_t error,join_error;
	uint32_t begun = 0u;
	error = SparkStageModuleCudaForkBegin(&buffers->fork,cudaStreamPerThread,1u);
	if ( error == cudaSuccess ) begun = 1u;
	if ( error == cudaSuccess ) error = SparkDsv4ValidationPrecollectiveLaunch(buffers,
		buffers->fork.auxiliary_streams[0],buffers->candidate_weight_bf16,
		buffers->candidate_weight_f32,buffers->candidate_indices,
		buffers->candidate_slot_count,buffers->candidate_attention_count);
	if ( error == cudaSuccess ) error = cudaMemsetAsync(buffers->collective_shadow,
		0xa5,SPARK_DSV4_VALIDATION_PRECOLLECTIVE_SHADOW_BYTES,cudaStreamPerThread);
	if ( begun != 0u )
	{
		join_error = SparkStageModuleCudaForkJoin(&buffers->fork,cudaStreamPerThread,1u);
		if ( error == cudaSuccess ) error = join_error;
	}
	if ( error == cudaSuccess ) error = cudaStreamSynchronize(cudaStreamPerThread);
	return(SparkDsv4ValidationRequireCuda(error,"projection_precollective_candidate"));
}

static int SparkDsv4ValidationPrecollectiveCheck(
	const SparkDsv4ValidationPrecollectiveBuffers *buffers)
{
	uint16_t control_bf16[SPARK_DSV4_MODEL_INDEX_HEAD_COUNT],candidate_bf16[SPARK_DSV4_MODEL_INDEX_HEAD_COUNT];
	float control_f32[SPARK_DSV4_MODEL_INDEX_HEAD_COUNT],candidate_f32[SPARK_DSV4_MODEL_INDEX_HEAD_COUNT];
	int32_t control_indices[SPARK_DSV4_VALIDATION_PRECOLLECTIVE_COLUMNS],candidate_indices[SPARK_DSV4_VALIDATION_PRECOLLECTIVE_COLUMNS];
	uint32_t control_slot,candidate_slot,control_attention,candidate_attention;
	cudaError_t error = cudaMemcpy(control_bf16,buffers->control_weight_bf16,sizeof(control_bf16),cudaMemcpyDeviceToHost);
	if ( error == cudaSuccess ) error = cudaMemcpy(candidate_bf16,buffers->candidate_weight_bf16,sizeof(candidate_bf16),cudaMemcpyDeviceToHost);
	if ( error == cudaSuccess ) error = cudaMemcpy(control_f32,buffers->control_weight_f32,sizeof(control_f32),cudaMemcpyDeviceToHost);
	if ( error == cudaSuccess ) error = cudaMemcpy(candidate_f32,buffers->candidate_weight_f32,sizeof(candidate_f32),cudaMemcpyDeviceToHost);
	if ( error == cudaSuccess ) error = cudaMemcpy(control_indices,buffers->control_indices,sizeof(control_indices),cudaMemcpyDeviceToHost);
	if ( error == cudaSuccess ) error = cudaMemcpy(candidate_indices,buffers->candidate_indices,sizeof(candidate_indices),cudaMemcpyDeviceToHost);
	if ( error == cudaSuccess ) error = cudaMemcpy(&control_slot,buffers->control_slot_count,sizeof(control_slot),cudaMemcpyDeviceToHost);
	if ( error == cudaSuccess ) error = cudaMemcpy(&candidate_slot,buffers->candidate_slot_count,sizeof(candidate_slot),cudaMemcpyDeviceToHost);
	if ( error == cudaSuccess ) error = cudaMemcpy(&control_attention,buffers->control_attention_count,sizeof(control_attention),cudaMemcpyDeviceToHost);
	if ( error == cudaSuccess ) error = cudaMemcpy(&candidate_attention,buffers->candidate_attention_count,sizeof(candidate_attention),cudaMemcpyDeviceToHost);
	if ( SparkDsv4ValidationRequireCuda(error,"projection_precollective_read") != 0 ) return(1);
	if ( SparkDsv4ValidationRequire(memcmp(control_bf16,candidate_bf16,sizeof(control_bf16)) == 0,"projection_precollective_bf16_exact") != 0 ) return(1);
	if ( SparkDsv4ValidationRequire(memcmp(control_f32,candidate_f32,sizeof(control_f32)) == 0,"projection_precollective_f32_exact") != 0 ) return(1);
	if ( SparkDsv4ValidationRequire(memcmp(control_indices,candidate_indices,sizeof(control_indices)) == 0,"projection_precollective_indices_exact") != 0 ) return(1);
	if ( SparkDsv4ValidationRequire(control_slot == candidate_slot && control_attention == candidate_attention,"projection_precollective_counts_exact") != 0 ) return(1);
	if ( SparkDsv4ValidationRequire(control_bf16[0] != 0u && isfinite(control_f32[0]) && control_f32[0] > 0.0f,"projection_precollective_nontrivial") != 0 ) return(1);
	return(SparkDsv4ValidationRequire(control_slot == SPARK_DSV4_VALIDATION_PRECOLLECTIVE_INDEX_CAPACITY && control_attention == SPARK_DSV4_VALIDATION_PRECOLLECTIVE_COLUMNS,"projection_precollective_shape"));
}

static int SparkDsv4ValidationProjectionPrecollective(void)
{
	SparkDsv4ValidationPrecollectiveBuffers buffers;
	int result;
	memset(&buffers,0,sizeof(buffers));
	result = SparkDsv4ValidationPrecollectiveAllocate(&buffers);
	if ( result == 0 ) result = SparkDsv4ValidationPrecollectiveRunControl(&buffers);
	if ( result == 0 ) result = SparkDsv4ValidationPrecollectiveRunCandidate(&buffers);
	if ( result == 0 ) result = SparkDsv4ValidationPrecollectiveCheck(&buffers);
	SparkDsv4ValidationPrecollectiveDestroy(&buffers);
	return(result);
}

static void SparkDsv4ValidationPostDestroy(
	SparkDsv4ValidationPostBuffers *buffers)
{
	if ( buffers->control_bf16 != 0 ) cudaFree(buffers->control_bf16);
	if ( buffers->candidate_bf16 != 0 ) cudaFree(buffers->candidate_bf16);
	if ( buffers->gain_bf16 != 0 ) cudaFree(buffers->gain_bf16);
	if ( buffers->freqs_f32 != 0 ) cudaFree(buffers->freqs_f32);
	if ( buffers->position_u64 != 0 ) cudaFree(buffers->position_u64);
	free(buffers->gain);
	free(buffers->candidate);
	free(buffers->control);
	free(buffers->input);
	memset(buffers,0,sizeof(*buffers));
}

static int SparkDsv4ValidationPostAllocate(
	SparkDsv4ValidationPostBuffers *buffers)
{
	uint64_t bytes = (uint64_t)SPARK_DSV4_VALIDATION_POST_ELEMENT_CAPACITY *
		sizeof(uint16_t);
	cudaError_t error;
	memset(buffers,0,sizeof(*buffers));
	buffers->input = (uint16_t *)calloc(SPARK_DSV4_VALIDATION_POST_ELEMENT_CAPACITY,sizeof(uint16_t));
	buffers->control = (uint16_t *)calloc(SPARK_DSV4_VALIDATION_POST_ELEMENT_CAPACITY,sizeof(uint16_t));
	buffers->candidate = (uint16_t *)calloc(SPARK_DSV4_VALIDATION_POST_ELEMENT_CAPACITY,sizeof(uint16_t));
	buffers->gain = (uint16_t *)calloc(SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,sizeof(uint16_t));
	if ( buffers->input == 0 || buffers->control == 0 || buffers->candidate == 0 || buffers->gain == 0 ) return(1);
	error = cudaMalloc(&buffers->control_bf16,bytes);
	if ( error == cudaSuccess ) error = cudaMalloc(&buffers->candidate_bf16,bytes);
	if ( error == cudaSuccess ) error = cudaMalloc(&buffers->gain_bf16,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION * sizeof(uint16_t));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&buffers->freqs_f32,(SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION / 2u) * sizeof(float));
	if ( error == cudaSuccess ) error = cudaMalloc((void **)&buffers->position_u64,sizeof(uint64_t));
	return(error == cudaSuccess ? 0 : 1);
}

static uint32_t SparkDsv4ValidationPostRandom(uint32_t *state)
{
	uint32_t value = *state;
	value ^= value << 13u;
	value ^= value >> 17u;
	value ^= value << 5u;
	*state = value;
	return(value);
}

static uint16_t SparkDsv4ValidationPostFloatToBf16(float value)
{
	uint32_t bias,bits;
	memcpy(&bits,&value,sizeof(bits));
	bias = UINT32_C(0x00007fff) + ((bits >> 16u) & 1u);
	return((uint16_t)((bits + bias) >> 16u));
}

static void SparkDsv4ValidationPostFillInput(uint16_t *input,
	uint32_t elements,uint32_t seed)
{
	uint32_t index,random,state = seed * UINT32_C(0x9e3779b9) + 1u;
	float value;
	for (index=0u; index<elements; index++)
	{
		random = SparkDsv4ValidationPostRandom(&state);
		if ( seed % 4u == 0u )
		{
			value = (float)(int32_t)(random & UINT32_C(0x0000ffff)) -
				32768.0f;
			input[index] = SparkDsv4ValidationPostFloatToBf16(value / 4096.0f);
		}
		else if ( seed % 4u == 1u )
			input[index] = (uint16_t)(((random >> 16u) & UINT32_C(0x8000)) |
				(((118u + (random % 17u)) & 0xffu) << 7u) |
				((random >> 8u) & UINT32_C(0x007f)));
		else if ( seed % 4u == 2u )
		{
			value = (float)(int32_t)(random & UINT32_C(0x000fffff)) -
				524288.0f;
			input[index] = SparkDsv4ValidationPostFloatToBf16(value / 65536.0f);
		}
		else
		{
			value = index % SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION <
				SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION -
				SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION ?
				(float)((int32_t)(random & 255u) - 128) / 32.0f :
				(float)((int32_t)(random & 65535u) - 32768) / 2048.0f;
			input[index] = SparkDsv4ValidationPostFloatToBf16(value);
		}
	}
}

static void SparkDsv4ValidationPostFillGain(uint16_t *gain,uint32_t seed)
{
	uint32_t index,state = seed * UINT32_C(0x85ebca6b) + 3u;
	float value;
	for (index=0u; index<SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION; index++)
	{
		value = 0.5f + (float)(SparkDsv4ValidationPostRandom(&state) & 1023u) /
			1024.0f;
		gain[index] = SparkDsv4ValidationPostFloatToBf16(value);
	}
}

static void SparkDsv4ValidationPostComputeFreqs(float *freqs,
	uint32_t frequency_mode)
{
	float base = frequency_mode == 0u ? SPARK_DSV4_MODEL_ATTN_ROPE_THETA :
		SPARK_DSV4_MODEL_COMPRESS_ROPE_THETA;
	uint32_t original = frequency_mode == 0u ? 0u :
		SPARK_DSV4_MODEL_ATTN_YARN_ORIGINAL_POSITIONS;
	float factor = (float)SPARK_DSV4_MODEL_ATTN_YARN_FACTOR;
	uint32_t rope_dim = SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,pair;
	float frequency,high,low,ramp,smooth;
	low = floorf((float)rope_dim * logf((float)original /
		((float)SPARK_DSV4_MODEL_ROPE_BETA_FAST * 2.0f * 3.14159265f)) /
		(2.0f * logf(base)));
	high = ceilf((float)rope_dim * logf((float)original /
		((float)SPARK_DSV4_MODEL_ROPE_BETA_SLOW * 2.0f * 3.14159265f)) /
		(2.0f * logf(base)));
	if ( low < 0.0f ) low = 0.0f;
	if ( high > (float)(rope_dim - 1u) ) high = (float)(rope_dim - 1u);
	if ( low == high ) high += 0.001f;
	for (pair=0u; pair<rope_dim / 2u; pair++)
	{
		frequency = 1.0f / powf(base,(float)(2u * pair) / (float)rope_dim);
		if ( original != 0u )
		{
			ramp = ((float)pair - low) / (high - low);
			if ( ramp < 0.0f ) ramp = 0.0f;
			if ( ramp > 1.0f ) ramp = 1.0f;
			smooth = 1.0f - ramp;
			frequency = frequency / factor * (1.0f - smooth) +
				frequency * smooth;
		}
		freqs[pair] = frequency;
	}
}

static int SparkDsv4ValidationPostLoad(
	SparkDsv4ValidationPostBuffers *buffers,uint32_t elements,uint32_t seed,
	uint32_t frequency_mode,uint64_t position)
{
	float freqs[SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION / 2u];
	cudaError_t error;
	SparkDsv4ValidationPostFillInput(buffers->input,elements,seed);
	SparkDsv4ValidationPostFillGain(buffers->gain,seed);
	SparkDsv4ValidationPostComputeFreqs(freqs,frequency_mode);
	error = cudaMemcpy(buffers->control_bf16,buffers->input,
		elements * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if ( error == cudaSuccess ) error = cudaMemcpy(buffers->candidate_bf16,buffers->input,elements * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if ( error == cudaSuccess ) error = cudaMemcpy(buffers->gain_bf16,buffers->gain,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION * sizeof(uint16_t),cudaMemcpyHostToDevice);
	if ( error == cudaSuccess ) error = cudaMemcpy(buffers->freqs_f32,freqs,sizeof(freqs),cudaMemcpyHostToDevice);
	if ( error == cudaSuccess ) error = cudaMemcpy(buffers->position_u64,&position,sizeof(position),cudaMemcpyHostToDevice);
	return(SparkDsv4ValidationRequireCuda(error,"post_fusion_load"));
}

static int SparkDsv4ValidationPostExact(
	SparkDsv4ValidationPostBuffers *buffers,uint32_t elements,
	const char *message)
{
	uint64_t bytes = (uint64_t)elements * sizeof(uint16_t);
	uint32_t index,nonzero = 0u;
	cudaError_t error = cudaStreamSynchronize(cudaStreamPerThread);
	if ( error == cudaSuccess ) error = cudaMemcpy(buffers->control,buffers->control_bf16,bytes,cudaMemcpyDeviceToHost);
	if ( error == cudaSuccess ) error = cudaMemcpy(buffers->candidate,buffers->candidate_bf16,bytes,cudaMemcpyDeviceToHost);
	if ( SparkDsv4ValidationRequireCuda(error,"post_fusion_read") != 0 ) return(1);
	if ( memcmp(buffers->control,buffers->candidate,bytes) != 0 )
	{
		for (index=0u; index<elements; index++)
			if ( buffers->control[index] != buffers->candidate[index] )
			{
				fprintf(stderr,"%s mismatch_index=%u control=%04x candidate=%04x\n",message,index,(unsigned)buffers->control[index],(unsigned)buffers->candidate[index]);
				break;
			}
		return(SparkDsv4ValidationRequire(0,message));
	}
	for (index=0u; index<elements; index++)
		nonzero += buffers->control[index] != 0u ? 1u : 0u;
	return(SparkDsv4ValidationRequire(nonzero != 0u,"post_fusion_nontrivial"));
}

static int SparkDsv4ValidationQueryPostCase(
	SparkDsv4ValidationPostBuffers *buffers,uint32_t seed,
	uint32_t frequency_mode,uint64_t position)
{
	uint32_t elements = SPARK_DSV4_VALIDATION_POST_QUERY_ELEMENTS;
	cudaError_t error;
	if ( SparkDsv4ValidationPostLoad(buffers,elements,seed,frequency_mode,
		position) != 0 ) return(1);
	error = SparkDsv4LaunchQueryHeadRms(cudaStreamPerThread,buffers->control_bf16,1u,SPARK_DSV4_VALIDATION_POST_QUERY_HEAD_COUNT,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess ) error = SparkDsv4LaunchRope(cudaStreamPerThread,buffers->control_bf16,buffers->freqs_f32,buffers->position_u64,1u,SPARK_DSV4_VALIDATION_POST_QUERY_HEAD_COUNT,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,0u);
	if ( error == cudaSuccess ) error = SparkDsv4LaunchQueryHeadRmsRope(cudaStreamPerThread,buffers->candidate_bf16,buffers->freqs_f32,buffers->position_u64,1u,SPARK_DSV4_VALIDATION_POST_QUERY_HEAD_COUNT,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( SparkDsv4ValidationRequireCuda(error,"query_post_fusion_launch") != 0 ) return(1);
	return(SparkDsv4ValidationPostExact(buffers,elements,"query_post_fusion_exact"));
}

static int SparkDsv4ValidationKvPostCase(
	SparkDsv4ValidationPostBuffers *buffers,uint32_t seed,
	uint32_t frequency_mode,uint64_t position)
{
	uint32_t elements = SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION;
	cudaError_t error;
	if ( SparkDsv4ValidationPostLoad(buffers,elements,seed,frequency_mode,
		position) != 0 ) return(1);
	error = SparkDsv4LaunchRmsNorm(cudaStreamPerThread,buffers->control_bf16,buffers->gain_bf16,buffers->control_bf16,1u,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( error == cudaSuccess ) error = SparkDsv4LaunchRope(cudaStreamPerThread,buffers->control_bf16,buffers->freqs_f32,buffers->position_u64,1u,1u,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,0u);
	if ( error == cudaSuccess ) error = SparkDsv4LaunchQuantSim(cudaStreamPerThread,buffers->control_bf16,1u,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION - SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,SPARK_DSV4_MODEL_KV_QUANT_BLOCK,0u);
	if ( error == cudaSuccess ) error = SparkDsv4LaunchKvPost(cudaStreamPerThread,buffers->candidate_bf16,buffers->gain_bf16,buffers->freqs_f32,buffers->position_u64,1u,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION,SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,SPARK_DSV4_MODEL_ATTN_HEAD_DIMENSION - SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,SPARK_DSV4_MODEL_KV_QUANT_BLOCK,SPARK_DSV4_MODEL_RMS_NORM_EPSILON);
	if ( SparkDsv4ValidationRequireCuda(error,"kv_post_fusion_launch") != 0 ) return(1);
	return(SparkDsv4ValidationPostExact(buffers,elements,"kv_post_fusion_exact"));
}

static int SparkDsv4ValidationIndexerPostCase(
	SparkDsv4ValidationPostBuffers *buffers,uint32_t seed,
	uint32_t frequency_mode,uint64_t position)
{
	uint32_t elements = SPARK_DSV4_VALIDATION_POST_INDEX_ELEMENTS;
	cudaError_t error;
	if ( SparkDsv4ValidationPostLoad(buffers,elements,seed,frequency_mode,
		position) != 0 ) return(1);
	error = SparkDsv4LaunchRope(cudaStreamPerThread,buffers->control_bf16,buffers->freqs_f32,buffers->position_u64,1u,SPARK_DSV4_MODEL_INDEX_HEAD_COUNT,SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION,SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,0u);
	if ( error == cudaSuccess ) error = SparkDsv4LaunchHadamard(cudaStreamPerThread,buffers->control_bf16,SPARK_DSV4_MODEL_INDEX_HEAD_COUNT,SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION);
	if ( error == cudaSuccess ) error = SparkDsv4LaunchQuantSim(cudaStreamPerThread,buffers->control_bf16,1u,SPARK_DSV4_MODEL_INDEX_DIMENSION,SPARK_DSV4_MODEL_INDEX_DIMENSION,SPARK_DSV4_MODEL_FP4_QUANT_BLOCK,1u);
	if ( error == cudaSuccess ) error = SparkDsv4LaunchIndexerPost(cudaStreamPerThread,buffers->candidate_bf16,buffers->freqs_f32,buffers->position_u64,1u,SPARK_DSV4_MODEL_INDEX_HEAD_COUNT,SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION,SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION,SPARK_DSV4_MODEL_FP4_QUANT_BLOCK);
	if ( SparkDsv4ValidationRequireCuda(error,"indexer_post_fusion_launch") != 0 ) return(1);
	return(SparkDsv4ValidationPostExact(buffers,elements,"indexer_post_fusion_exact"));
}

static int SparkDsv4ValidationPostFusions(void)
{
	static const uint32_t seeds[] = {1u,9u,53u};
	static const uint64_t positions[] = {0u,1u,127u,128u,255u,256u,4095u,
		4096u,65535u,65536u,262143u,262144u,1048575u};
	SparkDsv4ValidationPostBuffers buffers;
	uint32_t frequency,fusion,position,seed;
	int result;
	memset(&buffers,0,sizeof(buffers));
	result = SparkDsv4ValidationPostAllocate(&buffers);
	for (fusion=0u; fusion<SPARK_DSV4_VALIDATION_POST_FUSION_COUNT &&
		result==0; fusion++)
	{
		for (frequency=0u; frequency<SPARK_DSV4_VALIDATION_POST_FREQUENCY_COUNT &&
			result==0; frequency++)
		{
			for (position=0u; position<sizeof(positions)/sizeof(positions[0]) &&
				result==0; position++)
			{
				for (seed=0u; seed<sizeof(seeds)/sizeof(seeds[0]) && result==0;
					seed++)
				{
					if ( fusion == 0u ) result = SparkDsv4ValidationQueryPostCase(
						&buffers,seeds[seed],frequency,positions[position]);
					else if ( fusion == 1u ) result = SparkDsv4ValidationKvPostCase(
						&buffers,seeds[seed],frequency,positions[position]);
					else result = SparkDsv4ValidationIndexerPostCase(&buffers,
						seeds[seed],frequency,positions[position]);
				}
			}
		}
	}
	SparkDsv4ValidationPostDestroy(&buffers);
	return(result);
}

static int SparkDsv4ValidationReadUnsigned(
	const char *name,
	uint32_t minimum,
	uint32_t maximum,
	uint32_t *value)
{
	const char *text;
	char *end;
	unsigned long parsed;
	text = getenv(name);
	if ( value == 0 )
	{
		fprintf(stderr,"dsv4_validation field=%s destination=null\n",name);
		return(1);
	}
	if ( text == 0 || text[0] == '\0' )
	{
		fprintf(stderr,"dsv4_validation field=%s value=missing range=%u..%u\n",name,minimum,maximum);
		return(1);
	}
	errno = 0;
	end = 0;
	parsed = strtoul(text,&end,10);
	if ( errno != 0 || end == text || end[0] != '\0' || parsed < minimum || parsed > maximum )
	{
		fprintf(stderr,"dsv4_validation field=%s value=%s range=%u..%u\n",name,text,minimum,maximum);
		return(1);
	}
	*value = (uint32_t)parsed;
	return(0);
}

static int SparkDsv4ValidationLoadNodeContext(
	SparkDsv4ResidentDecodeStageNodeContext *context,
	uint32_t *logical_page_capacity,
	uint32_t *physical_page_capacity)
{
	uint32_t mtp_layer_count,cuda_graph_count;
	memset(context,0,sizeof(*context));
	context->abi_version = SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION;
	context->descriptor_bytes = SPARK_DSV4_RESIDENT_DECODE_STAGE_NODE_CONTEXT_BYTES;
	context->tp_degree = 1u;
	context->tp_rank = 0u;
	if ( SparkDsv4ValidationReadUnsigned("SPARK_DSV4_STAGE_COUNT",1u,SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT,&context->stage_count) != 0 ||
		SparkDsv4ValidationReadUnsigned("SPARK_DSV4_STAGE_INDEX",0u,SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_STAGE_COUNT - 1u,&context->stage_index) != 0 ||
		SparkDsv4ValidationReadUnsigned("SPARK_DSV4_STAGE_FIRST_LAYER",0u,SPARK_DSV4_MODEL_LAYER_COUNT - 1u,&context->first_layer_index) != 0 ||
		SparkDsv4ValidationReadUnsigned("SPARK_DSV4_STAGE_LAYER_COUNT",1u,SPARK_DSV4_MODEL_LAYER_COUNT,&context->layer_count) != 0 ||
		SparkDsv4ValidationReadUnsigned("SPARK_DSV4_STAGE_MAX_ACTIVE_SEQUENCES",SPARK_DSV4_VALIDATION_ROW_COUNT,SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT,&context->resident_sequence_capacity) != 0 ||
		SparkDsv4ValidationReadUnsigned("SPARK_DSV4_STAGE_PIPELINE_SLOTS",1u,SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT,&context->pipeline_slot_count) != 0 ||
		SparkDsv4ValidationReadUnsigned("SPARK_DSV4_STAGE_MAX_SEQ",SPARK_DSV4_MODEL_HCA_COMPRESS_RATIO,SPARK_DSV4_MODEL_MAX_POSITIONS,&context->max_sequence_positions) != 0 ||
		SparkDsv4ValidationReadUnsigned("SPARK_DSV4_STAGE_LOGICAL_PAGES",1u,SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_LOGICAL_PAGE_COUNT,logical_page_capacity) != 0 ||
		SparkDsv4ValidationReadUnsigned("SPARK_DSV4_STAGE_PHYSICAL_PAGES",1u,SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_PHYSICAL_PAGE_COUNT,physical_page_capacity) != 0 ||
		SparkDsv4ValidationReadUnsigned("SPARK_DSV4_STAGE_MTP",0u,0u,&mtp_layer_count) != 0 ||
		SparkDsv4ValidationReadUnsigned("SPARK_DSV4_STAGE_GRAPHS",0u,SPARK_DSV4_RESIDENT_DECODE_STAGE_MAX_GRAPH_COUNT,&cuda_graph_count) != 0 )
		return(1);
	context->cuda_graph_count = cuda_graph_count;
	context->linear_weight_codec = SPARK_DSV4_MODEL_NON_EXPERT_WEIGHT_CODEC;
	context->expert_weight_codec = SPARK_DSV4_MODEL_EXPERT_WEIGHT_CODEC;
	context->kv_cache_codec = SPARK_DSV4_MODEL_KV_CACHE_CODEC;
	context->stage_pack_path = getenv("SPARK_DSV4_STAGE_PACK_PATH");
	if ( context->stage_index >= context->stage_count || context->first_layer_index + context->layer_count > SPARK_DSV4_MODEL_LAYER_COUNT || context->stage_pack_path == 0 || context->stage_pack_path[0] == '\0' )
		return(1);
	return(0);
}

static void SparkDsv4ValidationInitializeConfiguration(
	SparkFirmwareModuleConfiguration *configuration,
	SparkFirmwareModuleHostServices *host_services,
	SparkDsv4ResidentDecodeStageNodeContext *node_context,
	uint32_t logical_page_capacity,
	uint32_t physical_page_capacity)
{
	memset(configuration,0,sizeof(*configuration));
	configuration->abi_version = SPARK_FIRMWARE_MODULE_ABI_VERSION;
	configuration->descriptor_bytes = sizeof(*configuration);
	configuration->model_id = SPARK_DSV4_MODEL_ID;
	configuration->model_revision = SPARK_DSV4_MODEL_SOURCE_REVISION;
	configuration->stage_name = "dsv4_resident_decode_stage";
	configuration->program_name = "resident_decode";
	configuration->operation_name = "dsv4_resident_decode_stage";
	configuration->configuration_json = "{}";
	configuration->configuration_json_bytes = 2u;
	memset(host_services,0,sizeof(*host_services));
	host_services->abi_version = SPARK_FIRMWARE_MODULE_HOST_SERVICES_ABI_VERSION;
	host_services->descriptor_bytes = sizeof(*host_services);
	host_services->node_id = "spark-dsv4-validator";
	host_services->node_target = SPARK_DSV4_MODEL_MODULE_TARGET;
	host_services->execution_stream = (void *)cudaStreamPerThread;
	host_services->node_context = node_context;
	host_services->kv_logical_page_capacity = logical_page_capacity;
	host_services->kv_physical_page_capacity = physical_page_capacity;
}

static void SparkDsv4ValidationCompletion(
	void *completion_context,
	const SparkModelDriverCompletion *completion)
{
	SparkDsv4ValidationCapture *capture;
	capture = (SparkDsv4ValidationCapture *)completion_context;
	if ( capture == 0 || completion == 0 )
		return;
	capture->completion = *completion;
	capture->completion_count++;
}

static void SparkDsv4ValidationReferenceCompletion(
	void *completion_context,
	const SparkModelDriverCompletion *completion)
{
	SparkDsv4ValidationReferenceFrame *frame;
	frame = (SparkDsv4ValidationReferenceFrame *)completion_context;
	if ( frame == 0 || completion == 0 )
		return;
	frame->completion = *completion;
	frame->completion_count++;
}

static float SparkDsv4ValidationBf16ToFloat(uint16_t value)
{
	uint32_t bits;
	float converted;
	bits = (uint32_t)value << 16u;
	memcpy(&converted,&bits,sizeof(converted));
	return(converted);
}

static int SparkDsv4ValidationReadPrefix(const char *path,void *destination,uint64_t bytes,uint64_t file_bytes)
{
	FILE *file;
	uint64_t read_bytes;
	long actual_bytes;
	file = fopen(path,"rb");
	if ( file == 0 )
		return(1);
	actual_bytes = fseek(file,0,SEEK_END) == 0 ? ftell(file) : -1;
	if ( actual_bytes < 0 || (uint64_t)actual_bytes != file_bytes || fseek(file,0,SEEK_SET) != 0 )
	{
		fclose(file);
		fprintf(stderr,"dsv4_validation reference_size path=%s bytes=%ld expected=%llu\n",path,actual_bytes,(unsigned long long)file_bytes);
		return(1);
	}
	read_bytes = fread(destination,1u,(size_t)bytes,file);
	fclose(file);
	if ( read_bytes != bytes )
	{
		fprintf(stderr,"dsv4_validation reference_read path=%s bytes=%llu expected=%llu\n",path,(unsigned long long)read_bytes,(unsigned long long)bytes);
		return(1);
	}
	return(0);
}

static void SparkDsv4ValidationAccumulateReferenceRow(SparkDsv4ValidationReferenceMetrics *metrics,const uint16_t *actual,const uint16_t *reference,uint32_t row,double *row_difference_l2,double *row_reference_l2)
{
	double actual_l2 = 0.0,difference_l2 = 0.0,dot = 0.0,reference_l2 = 0.0;
	double cosine,relative_l2;
	float difference,got,want;
	uint64_t index;
	for (index=0u; index<SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS; index++)
	{
		got = SparkDsv4ValidationBf16ToFloat(actual[index]);
		want = SparkDsv4ValidationBf16ToFloat(reference[index]);
		difference = got - want;
		metrics->nonfinite += isfinite(got) == 0 || isfinite(want) == 0 ? 1u : 0u;
		actual_l2 += (double)got * got;
		reference_l2 += (double)want * want;
		difference_l2 += (double)difference * difference;
		dot += (double)got * want;
		if ( fabs((double)difference) > metrics->maximum_absolute )
			metrics->maximum_absolute = fabs((double)difference);
	}
	metrics->actual_l2 += actual_l2;
	metrics->reference_l2 += reference_l2;
	metrics->difference_l2 += difference_l2;
	metrics->dot += dot;
	*row_difference_l2 = difference_l2;
	*row_reference_l2 = reference_l2;
	relative_l2 = reference_l2 > 0.0 ? sqrt(difference_l2 / reference_l2) : INFINITY;
	cosine = actual_l2 > 0.0 && reference_l2 > 0.0 ? dot / sqrt(actual_l2 * reference_l2) : 0.0;
	if ( relative_l2 > metrics->worst_row_relative_l2 )
	{
		metrics->worst_row_relative_l2 = relative_l2;
		metrics->worst_row_difference_l2 = sqrt(difference_l2);
		metrics->worst_row_reference_l2 = sqrt(reference_l2);
		metrics->worst_row_relative_l2_index = row;
	}
	if ( cosine < metrics->worst_row_cosine )
	{
		metrics->worst_row_cosine = cosine;
		metrics->worst_row_cosine_index = row;
	}
}

static int SparkDsv4ValidationReferenceThresholds(const SparkDsv4ValidationReferenceMetrics *metrics)
{
	if ( SparkDsv4ValidationRequire(metrics->nonfinite == 0u,"reference_finite") != 0 )
		return(1);
	if ( SparkDsv4ValidationRequire(metrics->relative_l2 <= SPARK_DSV4_VALIDATION_REFERENCE_MAX_RELATIVE_L2,"reference_relative_l2") != 0 )
		return(1);
	if ( SparkDsv4ValidationRequire(metrics->cosine >= SPARK_DSV4_VALIDATION_REFERENCE_MIN_COSINE,"reference_cosine") != 0 )
		return(1);
	if ( SparkDsv4ValidationRequire(metrics->worst_row_relative_l2 <= SPARK_DSV4_VALIDATION_REFERENCE_MAX_ROW_RELATIVE_L2,"reference_row_relative_l2") != 0 )
		return(1);
	if ( SparkDsv4ValidationRequire(metrics->worst_row_scaled_relative_l2 <= SPARK_DSV4_VALIDATION_REFERENCE_MAX_SCALED_ROW_RELATIVE_L2,"reference_row_scaled_relative_l2") != 0 )
		return(1);
	if ( SparkDsv4ValidationRequire(metrics->worst_row_cosine >= SPARK_DSV4_VALIDATION_REFERENCE_MIN_ROW_COSINE,"reference_row_cosine") != 0 )
		return(1);
	return(SparkDsv4ValidationRequire(metrics->maximum_absolute <= SPARK_DSV4_VALIDATION_REFERENCE_MAX_ABSOLUTE,"reference_maximum_absolute"));
}

static int SparkDsv4ValidationCompareReference(const char *path,const uint16_t *actual)
{
	const uint64_t elements = (uint64_t)SPARK_DSV4_VALIDATION_REFERENCE_ROW_COUNT * SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS;
	const uint64_t fixture_elements = (uint64_t)SPARK_DSV4_VALIDATION_REFERENCE_FIXTURE_ROW_COUNT * SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS;
	SparkDsv4ValidationReferenceMetrics metrics;
	double row_difference_l2[SPARK_DSV4_VALIDATION_REFERENCE_ROW_COUNT];
	double row_reference_l2[SPARK_DSV4_VALIDATION_REFERENCE_ROW_COUNT];
	double reference_row_floor,scaled_relative_l2;
	uint16_t *reference;
	uint32_t row;
	reference = (uint16_t *)calloc((size_t)elements,sizeof(uint16_t));
	if ( reference == 0 || SparkDsv4ValidationReadPrefix(path,reference,elements * sizeof(uint16_t),fixture_elements * sizeof(uint16_t)) != 0 )
	{
		free(reference);
		return(1);
	}
	memset(&metrics,0,sizeof(metrics));
	metrics.worst_row_cosine = 1.0;
	for (row=0u; row<SPARK_DSV4_VALIDATION_REFERENCE_ROW_COUNT; row++)
		SparkDsv4ValidationAccumulateReferenceRow(&metrics,actual + (uint64_t)row * SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS,reference + (uint64_t)row * SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS,row,&row_difference_l2[row],&row_reference_l2[row]);
	metrics.relative_l2 = metrics.reference_l2 > 0.0 ? sqrt(metrics.difference_l2 / metrics.reference_l2) : INFINITY;
	metrics.cosine = metrics.actual_l2 > 0.0 && metrics.reference_l2 > 0.0 ? metrics.dot / sqrt(metrics.actual_l2 * metrics.reference_l2) : 0.0;
	// Keep low-energy rows from inflating the localized relative-error guard.
	reference_row_floor = metrics.reference_l2 / SPARK_DSV4_VALIDATION_REFERENCE_ROW_COUNT;
	for (row=0u; row<SPARK_DSV4_VALIDATION_REFERENCE_ROW_COUNT; row++)
	{
		scaled_relative_l2 = reference_row_floor > 0.0 ? sqrt(row_difference_l2[row] / fmax(row_reference_l2[row],reference_row_floor)) : INFINITY;
		if ( scaled_relative_l2 > metrics.worst_row_scaled_relative_l2 )
		{
			metrics.worst_row_scaled_relative_l2 = scaled_relative_l2;
			metrics.worst_row_scaled_relative_l2_index = row;
		}
	}
	printf("dsv4_validation reference path=%s elements=%llu relative_l2=%.9g cosine=%.9g worst_row_relative_l2=%.9g worst_row_relative_l2_index=%u worst_row_scaled_relative_l2=%.9g worst_row_scaled_relative_l2_index=%u worst_row_difference_l2=%.9g worst_row_reference_l2=%.9g worst_row_cosine=%.9g worst_row_cosine_index=%u max_abs=%.9g nonfinite=%llu\n",path,(unsigned long long)elements,metrics.relative_l2,metrics.cosine,metrics.worst_row_relative_l2,metrics.worst_row_relative_l2_index,metrics.worst_row_scaled_relative_l2,metrics.worst_row_scaled_relative_l2_index,metrics.worst_row_difference_l2,metrics.worst_row_reference_l2,metrics.worst_row_cosine,metrics.worst_row_cosine_index,metrics.maximum_absolute,(unsigned long long)metrics.nonfinite);
	free(reference);
	return(SparkDsv4ValidationReferenceThresholds(&metrics));
}

static int SparkDsv4ValidationBuildReferenceRows(SparkDsv4ValidationReferenceFrame *frame,const char *token_path)
{
	const uint64_t fixture_bytes = (uint64_t)SPARK_DSV4_VALIDATION_REFERENCE_FIXTURE_ROW_COUNT * sizeof(uint32_t);
	uint32_t row;
	if ( SparkDsv4ValidationReadPrefix(token_path,frame->token_ids,sizeof(frame->token_ids),fixture_bytes) != 0 )
		return(1);
	for (row=0u; row<SPARK_DSV4_VALIDATION_REFERENCE_ROW_COUNT; row++)
	{
		if ( frame->token_ids[row] >= SPARK_DSV4_MODEL_VOCAB_COUNT )
			return(1);
		frame->lanes[row] = 0u;
		frame->positions[row] = row;
		frame->sequence_ids[row] = 1u;
	}
	return(0);
}

static void SparkDsv4ValidationBuildReferenceBatch(SparkDsv4ValidationReferenceFrame *frame)
{
	frame->batch.abi_version = SPARK_DSV4_RESIDENT_DECODE_STAGE_PREFILL_BATCH_VIEW_ABI_VERSION;
	frame->batch.descriptor_bytes = sizeof(frame->batch);
	frame->batch.row_count = SPARK_DSV4_VALIDATION_REFERENCE_ROW_COUNT;
	frame->batch.active_sequence_count = 1u;
	frame->batch.token_ids = frame->token_ids;
	frame->batch.row_lane_indices = frame->lanes;
	frame->batch.row_positions = frame->positions;
	frame->batch.row_sequence_ids = frame->sequence_ids;
}

static void SparkDsv4ValidationBuildReferenceContext(SparkDsv4ValidationReferenceFrame *frame,uint64_t hidden_bytes)
{
	frame->context.abi_version = SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION;
	frame->context.descriptor_bytes = sizeof(frame->context);
	frame->context.flags = SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_BATCH_VIEW | SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_BUFFER;
	frame->context.submission_id = 1u;
	frame->context.control_generation = 1u;
	frame->context.transaction_id = 1u;
	frame->context.dispatch_generation = 1u;
	frame->context.request_generation = 1u;
	frame->context.step_generation = 1u;
	frame->context.prefill_batch = &frame->batch;
	frame->context.hidden_output_bf16 = frame->hidden_output_bf16;
	frame->context.hidden_output_bytes = hidden_bytes;
	frame->buffers[0].flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_READ;
	frame->buffers[0].address = frame->token_ids;
	frame->buffers[0].bytes = sizeof(frame->token_ids);
}

static void SparkDsv4ValidationBuildReferenceDriverFrame(SparkDsv4ValidationReferenceFrame *frame)
{
	frame->frame.flags = SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL;
	frame->frame.request_id = 1u;
	frame->frame.sequence_id = 1u;
	frame->frame.active_slot_count = 1u;
	frame->frame.new_token_count = SPARK_DSV4_VALIDATION_REFERENCE_ROW_COUNT;
	frame->frame.tokens_per_sequence = 1u;
	frame->frame.program_id = 1u;
	frame->frame.execution_stream = (void *)cudaStreamPerThread;
	frame->frame.buffers = frame->buffers;
	frame->frame.buffer_count = 1u;
	frame->frame.user_context = &frame->context;
	frame->frame.completion_function = SparkDsv4ValidationReferenceCompletion;
	frame->frame.completion_context = frame;
}

static int SparkDsv4ValidationBuildReferenceFrame(SparkDsv4ValidationReferenceFrame *frame,const char *token_path)
{
	const uint64_t hidden_bytes = (uint64_t)SPARK_DSV4_VALIDATION_REFERENCE_ROW_COUNT * SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS * sizeof(uint16_t);
	cudaError_t error;
	memset(frame,0,sizeof(*frame));
	if ( SparkDsv4ValidationBuildReferenceRows(frame,token_path) != 0 )
		return(1);
	error = cudaMalloc(&frame->hidden_output_bf16,hidden_bytes);
	if ( error == cudaSuccess )
		error = cudaMemsetAsync(frame->hidden_output_bf16,0xff,hidden_bytes,cudaStreamPerThread);
	if ( error != cudaSuccess )
	{
		if ( frame->hidden_output_bf16 != 0 )
			cudaFree(frame->hidden_output_bf16);
		frame->hidden_output_bf16 = 0;
		return(1);
	}
	SparkDsv4ValidationBuildReferenceBatch(frame);
	SparkDsv4ValidationBuildReferenceContext(frame,hidden_bytes);
	SparkDsv4ValidationBuildReferenceDriverFrame(frame);
	return(0);
}

static int SparkDsv4ValidationRunReference(void *module_state,const SparkDsv4ResidentDecodeStageNodeContext *node_context,const char *token_path,const char *output_path)
{
	const uint64_t elements = (uint64_t)SPARK_DSV4_VALIDATION_REFERENCE_ROW_COUNT * SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS;
	SparkDsv4ValidationReferenceFrame frame;
	uint16_t *actual;
	cudaError_t error;
	SparkStatus status;
	if ( node_context->stage_index != 0u || node_context->first_layer_index != 0u || node_context->layer_count != 3u )
		return(SparkDsv4ValidationRequire(0,"reference_stage0_slice"));
	actual = (uint16_t *)calloc((size_t)elements,sizeof(uint16_t));
	if ( actual == 0 || SparkDsv4ValidationBuildReferenceFrame(&frame,token_path) != 0 )
	{
		free(actual);
		return(1);
	}
	status = SparkDsv4ResidentDecodeStageExecute(module_state,&frame.frame);
	error = status == SPARK_STATUS_OK ? cudaStreamSynchronize(cudaStreamPerThread) : cudaErrorUnknown;
	if ( error == cudaSuccess )
		error = cudaMemcpy(actual,frame.hidden_output_bf16,elements * sizeof(uint16_t),cudaMemcpyDeviceToHost);
	if ( frame.hidden_output_bf16 != 0 )
		cudaFree(frame.hidden_output_bf16);
	if ( status != SPARK_STATUS_OK || error != cudaSuccess || frame.completion_count != 1u || frame.completion.status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"dsv4_validation reference_execute=%s cuda=%s completion=%u status=%s\n",SparkStatusToString(status),cudaGetErrorString(error),frame.completion_count,SparkStatusToString(frame.completion.status));
		free(actual);
		return(1);
	}
	status = SparkDsv4ValidationCompareReference(output_path,actual) == 0 ? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
	free(actual);
	return(status == SPARK_STATUS_OK ? 0 : 1);
}

static void SparkDsv4ValidationDestroyFrame(SparkDsv4ValidationFrame *frame)
{
	if ( frame->hidden_input_bf16 != 0 )
		cudaFree(frame->hidden_input_bf16);
	if ( frame->hidden_output_bf16 != 0 )
		cudaFree(frame->hidden_output_bf16);
	memset(frame,0,sizeof(*frame));
}

static int SparkDsv4ValidationInitializeBoundaries(
	const SparkDsv4ResidentDecodeStageNodeContext *node_context,
	SparkDsv4ValidationCapture *capture,
	SparkDsv4ValidationFrame *frame)
{
	uint32_t element;
	uint64_t hidden_bytes;
	cudaError_t error;
	hidden_bytes = sizeof(capture->hidden);
	if ( node_context->stage_index != 0u )
	{
		for (element=0u; element<SPARK_DSV4_VALIDATION_ROW_COUNT * SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS; element++)
			capture->hidden[element] = UINT16_C(0x3f80);
		error = cudaMalloc(&frame->hidden_input_bf16,hidden_bytes);
		if ( error == cudaSuccess )
			error = cudaMemcpy(frame->hidden_input_bf16,capture->hidden,hidden_bytes,cudaMemcpyHostToDevice);
		if ( error != cudaSuccess )
			return(1);
	}
	if ( node_context->stage_index + 1u < node_context->stage_count )
	{
		error = cudaMalloc(&frame->hidden_output_bf16,hidden_bytes);
		if ( error != cudaSuccess )
			return(1);
	}
	return(0);
}

static void SparkDsv4ValidationBuildFrame(
	const SparkDsv4ResidentDecodeStageNodeContext *node_context,
	SparkDsv4ValidationCapture *capture,
	SparkDsv4ValidationFrame *frame)
{
	uint32_t buffer_count,row;
	uint64_t hidden_bytes;
	hidden_bytes = sizeof(capture->hidden);
	for (row=0u; row<SPARK_DSV4_VALIDATION_ROW_COUNT; row++)
	{
		frame->token_ids[row] = 10397u + row;
		frame->lanes[row] = row;
		frame->positions[row] = 0u;
		frame->sequence_ids[row] = (uint64_t)row + 1u;
	}
	frame->batch.abi_version = SPARK_DSV4_RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION;
	frame->batch.descriptor_bytes = sizeof(frame->batch);
	frame->batch.row_count = SPARK_DSV4_VALIDATION_ROW_COUNT;
	frame->batch.row_lane_indices = frame->lanes;
	frame->batch.row_positions = frame->positions;
	frame->batch.row_sequence_ids = frame->sequence_ids;
	frame->context.abi_version = SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION;
	frame->context.descriptor_bytes = sizeof(frame->context);
	frame->context.flags = SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW;
	frame->context.submission_id = 1u;
	frame->context.control_generation = 1u;
	frame->context.transaction_id = 1u;
	frame->context.dispatch_generation = 1u;
	frame->context.request_generation = 1u;
	frame->context.step_generation = 1u;
	frame->context.decode_batch = &frame->batch;
	frame->context.hidden_input_bf16 = frame->hidden_input_bf16;
	frame->context.hidden_input_bytes = frame->hidden_input_bf16 != 0 ? hidden_bytes : 0u;
	frame->context.hidden_output_bf16 = frame->hidden_output_bf16;
	frame->context.hidden_output_bytes = frame->hidden_output_bf16 != 0 ? hidden_bytes : 0u;
	if ( frame->hidden_input_bf16 != 0 )
		frame->context.flags |= SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_BUFFER;
	if ( frame->hidden_output_bf16 != 0 )
		frame->context.flags |= SPARK_DSV4_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_BUFFER;
	buffer_count = 0u;
	frame->buffers[buffer_count].flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_READ;
	frame->buffers[buffer_count].address = frame->token_ids;
	frame->buffers[buffer_count++].bytes = sizeof(frame->token_ids);
	if ( node_context->first_layer_index + node_context->layer_count == SPARK_DSV4_MODEL_LAYER_COUNT )
	{
		frame->buffers[buffer_count].slot = 1u;
		frame->buffers[buffer_count].flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE;
		frame->buffers[buffer_count].address = capture->output_token_ids;
		frame->buffers[buffer_count++].bytes = sizeof(capture->output_token_ids);
	}
	frame->frame.request_id = 1u;
	frame->frame.sequence_id = 1u;
	frame->frame.active_slot_count = SPARK_DSV4_VALIDATION_ROW_COUNT;
	frame->frame.new_token_count = SPARK_DSV4_VALIDATION_ROW_COUNT;
	frame->frame.tokens_per_sequence = 1u;
	frame->frame.program_id = 1u;
	frame->frame.execution_stream = (void *)cudaStreamPerThread;
	frame->frame.buffers = buffer_count != 0u ? frame->buffers : 0;
	frame->frame.buffer_count = buffer_count;
	frame->frame.user_context = &frame->context;
	frame->frame.completion_function = SparkDsv4ValidationCompletion;
	frame->frame.completion_context = capture;
}

static int SparkDsv4ValidationRunFrame(
	void *module_state,
	const SparkDsv4ResidentDecodeStageNodeContext *node_context,
	SparkDsv4ValidationCapture *capture)
{
	SparkDsv4ValidationFrame frame;
	uint32_t element;
	cudaError_t error;
	SparkStatus status;
	memset(&frame,0,sizeof(frame));
	memset(capture,0,sizeof(*capture));
	for (element=0u; element<SPARK_DSV4_VALIDATION_ROW_COUNT; element++)
		capture->output_token_ids[element] = UINT32_MAX;
	if ( SparkDsv4ValidationInitializeBoundaries(node_context,capture,&frame) != 0 )
	{
		SparkDsv4ValidationDestroyFrame(&frame);
		return(1);
	}
	SparkDsv4ValidationBuildFrame(node_context,capture,&frame);
	status = SparkDsv4ResidentDecodeStageExecute(module_state,&frame.frame);
	error = status == SPARK_STATUS_OK ? cudaStreamSynchronize(cudaStreamPerThread) : cudaErrorUnknown;
	if ( status == SPARK_STATUS_OK && error == cudaSuccess && frame.hidden_output_bf16 != 0 )
		error = cudaMemcpy(capture->hidden,frame.hidden_output_bf16,sizeof(capture->hidden),cudaMemcpyDeviceToHost);
	if ( error == cudaSuccess && frame.hidden_output_bf16 != 0 )
		for (element=0u; element<SPARK_DSV4_VALIDATION_ROW_COUNT * SPARK_DSV4_MODEL_BOUNDARY_STREAM_ELEMENTS; element++)
			capture->nonzero_count += capture->hidden[element] != 0u ? 1u : 0u;
	capture->output_count = error == cudaSuccess ? SPARK_DSV4_VALIDATION_ROW_COUNT : 0u;
	SparkDsv4ValidationDestroyFrame(&frame);
	if ( status != SPARK_STATUS_OK || error != cudaSuccess )
	{
		fprintf(stderr,"dsv4_validation execute=%s cuda=%s\n",SparkStatusToString(status),cudaGetErrorString(error));
		return(1);
	}
	if ( SparkDsv4ValidationRequire(capture->completion_count == 1u && capture->completion.status == SPARK_STATUS_OK,"completion") != 0 || SparkDsv4ValidationRequire(capture->output_count == SPARK_DSV4_VALIDATION_ROW_COUNT,"output_count") != 0 )
		return(1);
	if ( node_context->stage_index + 1u < node_context->stage_count )
		return(SparkDsv4ValidationRequire(capture->nonzero_count > 0u,"hidden_output_nonzero"));
	for (element=0u; element<SPARK_DSV4_VALIDATION_ROW_COUNT; element++)
		if ( SparkDsv4ValidationRequire(capture->output_token_ids[element] < SPARK_DSV4_MODEL_VOCAB_COUNT,"output_token_range") != 0 )
			return(1);
	return(0);
}

static int SparkDsv4ValidationAdmit(void *module_state,uint32_t active_slot_count,uint32_t new_token_count,uint32_t frame_flags)
{
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	SparkStatus status;
	memset(&request,0,sizeof(request));
	request.descriptor_bytes = sizeof(request);
	request.program_id = 1u;
	request.request_id = 1u;
	request.sequence_id = 1u;
	request.active_slot_count = active_slot_count;
	request.new_token_count = new_token_count;
	request.frame_flags = frame_flags;
	memset(&decision,0,sizeof(decision));
	decision.descriptor_bytes = sizeof(decision);
	status = SparkDsv4ResidentDecodeStageAdmit(module_state,&request,&decision);
	if ( status != SPARK_STATUS_OK || decision.accepted == 0u )
	{
		fprintf(stderr,"dsv4_validation admit=%s accepted=%u\n",SparkStatusToString(status),decision.accepted);
		return(1);
	}
	return(0);
}

static int SparkDsv4ValidationLoadMode(SparkDsv4ValidationMode *mode)
{
	int32_t has_output,has_tokens;
	memset(mode,0,sizeof(*mode));
	mode->reference_path = getenv("SPARK_DSV4_REFERENCE_OUTPUT_PATH");
	mode->reference_token_path = getenv("SPARK_DSV4_REFERENCE_TOKEN_PATH");
	has_output = mode->reference_path != 0 && mode->reference_path[0] != '\0';
	has_tokens = mode->reference_token_path != 0 && mode->reference_token_path[0] != '\0';
	if ( has_output != has_tokens )
	{
		fprintf(stderr,"dsv4_validation reference_configuration=invalid\n");
		return(1);
	}
	mode->use_reference = has_output;
	mode->active_slot_count = has_output != 0 ? 1u : SPARK_DSV4_VALIDATION_ROW_COUNT;
	mode->new_token_count = has_output != 0 ? SPARK_DSV4_VALIDATION_REFERENCE_ROW_COUNT : SPARK_DSV4_VALIDATION_ROW_COUNT;
	mode->frame_flags = has_output != 0 ? SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL : 0u;
	return(0);
}

static int SparkDsv4ValidationRequireMode(const SparkDsv4ResidentDecodeStageNodeContext *node_context,const SparkDsv4ValidationMode *mode)
{
	int32_t exact_reference_slice;
	exact_reference_slice = node_context->stage_index == 0u && node_context->first_layer_index == 0u && node_context->layer_count == 3u;
	if ( mode->use_reference != exact_reference_slice )
	{
		fprintf(stderr,"dsv4_validation reference_mode=invalid stage=%u slice=%u+%u enabled=%d\n",node_context->stage_index,node_context->first_layer_index,node_context->layer_count,mode->use_reference);
		return(1);
	}
	return(0);
}

static int SparkDsv4ValidationRunMode(void *module_state,const SparkDsv4ResidentDecodeStageNodeContext *node_context,SparkDsv4ValidationCapture *capture,const SparkDsv4ValidationMode *mode)
{
	if ( SparkDsv4ValidationAdmit(module_state,mode->active_slot_count,mode->new_token_count,mode->frame_flags) != 0 )
		return(1);
	if ( mode->use_reference != 0 )
		return(SparkDsv4ValidationRunReference(module_state,node_context,mode->reference_token_path,mode->reference_path));
	return(SparkDsv4ValidationRunFrame(module_state,node_context,capture));
}

static int SparkDsv4ValidationSnapshot(void *module_state)
{
	SparkModelDriverRuntimeSnapshot snapshot;
	SparkStatus status;
	memset(&snapshot,0,sizeof(snapshot));
	snapshot.descriptor_bytes = sizeof(snapshot);
	status = SparkDsv4ResidentDecodeStageSnapshot(module_state,1u,&snapshot);
	if ( status != SPARK_STATUS_OK || snapshot.submitted_count != 1u || snapshot.completed_count != 1u || snapshot.active_submission_count != 0u )
	{
		fprintf(stderr,"dsv4_validation snapshot=%s submitted=%llu completed=%llu active=%u\n",SparkStatusToString(status),(unsigned long long)snapshot.submitted_count,(unsigned long long)snapshot.completed_count,snapshot.active_submission_count);
		return(1);
	}
	return(0);
}

static void SparkDsv4ValidationPrintPass(const char *configuration_hash,const SparkDsv4ResidentDecodeStageNodeContext *node_context,const SparkDsv4ValidationCapture *capture,const SparkDsv4ValidationMode *mode)
{
	if ( mode->use_reference != 0 )
		printf("dsv4_validation PASS config=%s stage=%u slice=%u+%u reference_rows=%u\n",configuration_hash,node_context->stage_index,node_context->first_layer_index,node_context->layer_count,(unsigned)SPARK_DSV4_VALIDATION_REFERENCE_ROW_COUNT);
	else
		printf("dsv4_validation PASS config=%s stage=%u slice=%u+%u rows=%u nonzero_hidden=%u output_token=%u\n",configuration_hash,node_context->stage_index,node_context->first_layer_index,node_context->layer_count,(unsigned)SPARK_DSV4_VALIDATION_ROW_COUNT,capture->nonzero_count,capture->output_token_ids[0]);
}

int main(int argument_count,char **arguments)
{
	SparkFirmwareModuleConfiguration configuration;
	SparkFirmwareModuleHostServices host_services;
	SparkDsv4ResidentDecodeStageNodeContext node_context;
	SparkDsv4ValidationCapture capture;
	SparkDsv4ValidationMode mode;
	void *module_state;
	int32_t snapshot_result;
	uint32_t logical_page_capacity,physical_page_capacity;
	SparkStatus status;
	if ( argument_count != 2 )
	{
		fprintf(stderr,"usage: %s VALIDATION_CONFIGURATION_SHA256\n",arguments[0]);
		return(2);
	}
	if ( SparkDsv4ValidationHead() != 0 )
		return(1);
	if ( SparkDsv4ValidationGateRouteB1() != 0 )
		return(1);
	if ( SparkDsv4ValidationProjectionPrecollective() != 0 )
		return(1);
	if ( SparkDsv4ValidationPostFusions() != 0 )
		return(1);
	if ( SparkDsv4ValidationLoadNodeContext(&node_context,
		&logical_page_capacity,&physical_page_capacity) != 0 )
	{
		fprintf(stderr,"dsv4_validation configuration=invalid\n");
		return(1);
	}
	SparkDsv4ValidationInitializeConfiguration(&configuration,&host_services,
		&node_context,logical_page_capacity,physical_page_capacity);
	if ( SparkDsv4ValidationLoadMode(&mode) != 0 || SparkDsv4ValidationRequireMode(&node_context,&mode) != 0 )
		return(1);
	module_state = 0;
	status = SparkDsv4ResidentDecodeStageInitialize(&configuration,&host_services,&module_state);
	if ( status != SPARK_STATUS_OK || module_state == 0 )
	{
		fprintf(stderr,"dsv4_validation initialize=%s\n",SparkStatusToString(status));
		return(1);
	}
	if ( SparkDsv4ValidationRunMode(module_state,&node_context,&capture,&mode) != 0 )
	{
		SparkDsv4ResidentDecodeStageDestroy(module_state);
		return(1);
	}
	snapshot_result = SparkDsv4ValidationSnapshot(module_state);
	SparkDsv4ResidentDecodeStageDestroy(module_state);
	if ( snapshot_result != 0 )
		return(1);
	SparkDsv4ValidationPrintPass(arguments[1],&node_context,&capture,&mode);
	return(0);
}
