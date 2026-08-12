#include <cuda_runtime.h>

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_dsv4_model.h"
#include "sparkpipe/spark_dsv4_resident_decode_stage_firmware.h"

#define SPARK_DSV4_VALIDATION_ROW_COUNT \
	(SPARK_BATCH_BUCKET < 8u ? SPARK_BATCH_BUCKET : 8u)
#define SPARK_DSV4_VALIDATION_REFERENCE_ROW_COUNT 128u
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

extern "C" cudaError_t SparkDsv4LaunchHeadShadowQuantize(cudaStream_t stream, const void *head_bf16, uint8_t *shadow_payload, uint8_t *shadow_scale, float *error_norm, uint32_t candidate_count, uint32_t hidden_dimension);
extern "C" cudaError_t SparkDsv4LaunchHeadScreenedArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint8_t *shadow_payload, const uint8_t *shadow_scale, const float *error_norm, void *logits_bf16, uint32_t *candidate_ids, uint32_t *candidate_counts, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count, uint32_t hidden_dimension);
extern "C" cudaError_t SparkDsv4LaunchHeadArgmax(cudaStream_t stream, const void *hidden_bf16, const void *head_weight_bf16, const uint32_t *token_ids, uint32_t *output_token_ids, uint32_t row_count, uint32_t candidate_count, uint32_t hidden_dimension);

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
	if ( SparkDsv4ValidationRequire(error == cudaSuccess,"head_cuda") != 0 ) return(1);
	for (row=0u; row<row_count; row++)
	{
		if ( SparkDsv4ValidationRequire(counts[row] == (overflow != 0u ? candidate_count : ((row & 1u) == 0u ? 2u : 1u)),"head_screen_branch") != 0 ) return(1);
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
	if ( result == 0 ) result = SparkDsv4ValidationHeadRunCase(&buffers,head,errors,SPARK_DSV4_VALIDATION_HEAD_ROW_COUNT - 1u,SPARK_DSV4_VALIDATION_HEAD_OVERFLOW_VOCAB,1u,0u);
	if ( result == 0 ) result = SparkDsv4ValidationHeadRunCase(&buffers,head,errors,SPARK_DSV4_VALIDATION_HEAD_ROW_COUNT,SPARK_DSV4_VALIDATION_HEAD_OVERFLOW_VOCAB,1u,0u);
	for (seed=1u; seed<=4u && result == 0; seed++) result = SparkDsv4ValidationHeadRunCase(&buffers,head,errors,SPARK_DSV4_VALIDATION_HEAD_ROW_COUNT,SPARK_DSV4_VALIDATION_HEAD_OVERFLOW_VOCAB,1u,seed);
	SparkDsv4ValidationHeadDestroy(&buffers);
	free(errors);
	free(head);
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
	if ( text == 0 || text[0] == '\0' || value == 0 )
		return(1);
	errno = 0;
	end = 0;
	parsed = strtoul(text,&end,10);
	if ( errno != 0 || end == text || end[0] != '\0' || parsed < minimum || parsed > maximum )
		return(1);
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

static int SparkDsv4ValidationReadExact(const char *path,void *destination,uint64_t bytes)
{
	FILE *file;
	uint64_t read_bytes;
	int32_t extra;
	file = fopen(path,"rb");
	if ( file == 0 )
		return(1);
	read_bytes = fread(destination,1u,(size_t)bytes,file);
	extra = fgetc(file);
	fclose(file);
	if ( read_bytes != bytes || extra != EOF )
	{
		fprintf(stderr,"dsv4_validation reference_read path=%s bytes=%llu expected=%llu extra=%d\n",path,(unsigned long long)read_bytes,(unsigned long long)bytes,extra);
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
	SparkDsv4ValidationReferenceMetrics metrics;
	double row_difference_l2[SPARK_DSV4_VALIDATION_REFERENCE_ROW_COUNT];
	double row_reference_l2[SPARK_DSV4_VALIDATION_REFERENCE_ROW_COUNT];
	double reference_row_floor,scaled_relative_l2;
	uint16_t *reference;
	uint32_t row;
	reference = (uint16_t *)calloc((size_t)elements,sizeof(uint16_t));
	if ( reference == 0 || SparkDsv4ValidationReadExact(path,reference,elements * sizeof(uint16_t)) != 0 )
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
	uint32_t row;
	if ( SparkDsv4ValidationReadExact(token_path,frame->token_ids,sizeof(frame->token_ids)) != 0 )
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
