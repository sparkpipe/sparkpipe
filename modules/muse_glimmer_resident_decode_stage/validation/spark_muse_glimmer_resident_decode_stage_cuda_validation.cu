#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_muse_glimmer_model.h"
#include "sparkpipe/spark_muse_glimmer_resident_decode_stage_firmware.h"
#include "sparkpipe/spark_model_driver_support.h"


#define SPARK_MUSE_GLIMMER_VALIDATION_ROWS 2u
#define SPARK_MUSE_GLIMMER_VALIDATION_STEPS 8u
#ifndef SPARK_MUSE_GLIMMER_STAGE_MAX_ACTIVE_SEQUENCES
#define SPARK_MUSE_GLIMMER_STAGE_MAX_ACTIVE_SEQUENCES 8u
#endif
#define SPARK_MUSE_GLIMMER_VALIDATION_KV_LANES SPARK_MUSE_GLIMMER_STAGE_MAX_ACTIVE_SEQUENCES
#define SPARK_MUSE_GLIMMER_VALIDATION_LOCAL_HEADS 2u
#define SPARK_MUSE_GLIMMER_VALIDATION_WINDOW_CONTEXT 2049u

extern "C" cudaError_t SparkMuseGlimmerLaunchEmbeddingGather(cudaStream_t stream, const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count, uint32_t tp_degree, uint32_t tp_rank);
extern "C" cudaError_t SparkMuseGlimmerLaunchCenteredRmsNorm(cudaStream_t stream, const void *input_bf16, const void *weight_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon);
extern "C" cudaError_t SparkMuseGlimmerLaunchHeadRmsNorm(cudaStream_t stream, const void *input_bf16, const void *weight_bf16, void *output_bf16, uint32_t row_count, uint32_t head_count, uint32_t head_dimension, float epsilon, float head_multiply);
extern "C" cudaError_t SparkMuseGlimmerLaunchKvStore(cudaStream_t stream, const void *views, uint32_t layer_index, const void *key_bf16, const void *value_bf16, const uint32_t *sequence_of_row, const uint32_t *positions, uint32_t row_count, uint32_t local_kv_head_count);
extern "C" cudaError_t SparkMuseGlimmerLaunchWindowPositions(cudaStream_t stream, const uint32_t *sequence_of_row, const uint32_t *context_lengths, const uint32_t *positions, uint32_t row_count, uint32_t *window_positions);
extern "C" cudaError_t SparkMuseGlimmerLaunchAttentionDecode(cudaStream_t stream, const void *views, uint32_t layer_index, const void *query_bf16, const uint32_t *sequence_of_row, const uint32_t *context_lengths, const uint32_t *window_positions, const uint32_t *positions, void *head_out_bf16, uint32_t row_count, uint32_t local_head_count, uint32_t local_kv_head_count);
extern "C" cudaError_t SparkMuseGlimmerLaunchOutputGate(cudaStream_t stream, void *head_out_bf16, const void *gate_bf16, uint32_t row_count, uint32_t local_query_dimension);
extern "C" cudaError_t SparkMuseGlimmerLaunchSiluMul(cudaStream_t stream, const void *gate_up_bf16, void *intermediate_bf16, uint32_t row_count, uint32_t local_intermediate);
extern "C" uint32_t SparkMuseGlimmerKvViewBytes(void);
extern "C" SparkStatus SparkMuseGlimmerResidentDecodeStageInitialize(const SparkFirmwareModuleConfiguration *configuration, const SparkFirmwareModuleHostServices *host_services, void **module_state);
extern "C" SparkStatus SparkMuseGlimmerResidentDecodeStageExecute(void *module_state, SparkModelDriverFrame *frame);
extern "C" SparkStatus SparkMuseGlimmerResidentDecodeStageAdmit(void *module_state, const SparkModelDriverAdmissionRequest *request, SparkModelDriverAdmissionDecision *decision);
extern "C" SparkStatus SparkMuseGlimmerResidentDecodeStageSnapshot(void *module_state, uint32_t program_id, SparkModelDriverRuntimeSnapshot *snapshot);
extern "C" void SparkMuseGlimmerResidentDecodeStageDestroy(void *module_state);

typedef struct SparkMuseGlimmerKvViewShim
{
	void *pool;
	const uint32_t *page_table;
	uint32_t page_table_stride;
	uint32_t sequence_count;
	uint32_t pool_page_count;
	void *access_error;
} SparkMuseGlimmerKvViewShim;

typedef struct SparkMuseGlimmerFrameErrorShim
{
	uint32_t error_code;
	uint32_t access_kind;
	uint32_t row;
	uint32_t sequence;
	uint32_t position;
	uint32_t page;
} SparkMuseGlimmerFrameErrorShim;

static uint32_t SparkMuseGlimmerValRandomState = 0x12345678u;

static uint32_t SparkMuseGlimmerValNext(void)
{
	uint32_t state = SparkMuseGlimmerValRandomState;
	state ^= state << 13;
	state ^= state >> 17;
	state ^= state << 5;
	SparkMuseGlimmerValRandomState = state;
	return(state);
}

static float SparkMuseGlimmerValUniform(float scale)
{
	return(((float)(SparkMuseGlimmerValNext() & 0xffffu) - 32768.0f) / 32768.0f) * scale;
}

static uint16_t SparkMuseGlimmerValBf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	return((uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u));
}

static float SparkMuseGlimmerValFromBf16(uint16_t value)
{
	uint32_t bits = ((uint32_t)value) << 16u;
	float result;
	memcpy(&result,&bits,sizeof(result));
	return(result);
}

static int SparkMuseGlimmerValFail(const char *check, const char *detail)
{
	fprintf(stderr,"muse_glimmer_validation failure=%s detail=%s\n",check,detail);
	return(1);
}

static int SparkMuseGlimmerValCuda(cudaError_t error, const char *check)
{
	if ( error != cudaSuccess )
	{
		fprintf(stderr,"muse_glimmer_validation failure=%s cuda=%s\n",check,cudaGetErrorString(error));
		return(1);
	}
	return(0);
}

static void SparkMuseGlimmerValReferenceCenteredNorm(const uint16_t *input, const uint16_t *weight, uint16_t *output, uint32_t dimension, float epsilon)
{
	uint32_t index;
	float total = 0.0f,scale;
	for (index = 0; index < dimension; index++)
		total += SparkMuseGlimmerValFromBf16(input[index]) * SparkMuseGlimmerValFromBf16(input[index]);
	scale = 1.0f / sqrtf(total / (float)dimension + epsilon);
	for (index = 0; index < dimension; index++)
		output[index] = SparkMuseGlimmerValBf16(SparkMuseGlimmerValFromBf16(input[index]) * scale * (1.0f + SparkMuseGlimmerValFromBf16(weight[index])));
}

static void SparkMuseGlimmerValReferenceHeadNorm(const uint16_t *input, uint16_t *output, uint32_t head_dimension, float epsilon, float multiply)
{
	uint32_t index;
	float total = 0.0f,scale;
	for (index = 0; index < head_dimension; index++)
		total += SparkMuseGlimmerValFromBf16(input[index]) * SparkMuseGlimmerValFromBf16(input[index]);
	scale = 1.0f / sqrtf(total / (float)head_dimension + epsilon);
	for (index = 0; index < head_dimension; index++)
		output[index] = SparkMuseGlimmerValBf16(SparkMuseGlimmerValBf16(SparkMuseGlimmerValFromBf16(input[index]) * scale) * multiply);
}

static void SparkMuseGlimmerValReferenceDecode(const uint16_t *query, const uint16_t *pool, const uint32_t *selected, uint32_t selected_count, uint16_t *output, uint64_t slot_elements)
{
	uint32_t head,element,step;
	for (head = 0; head < SPARK_MUSE_GLIMMER_VALIDATION_LOCAL_HEADS; head++)
	{
		const uint16_t *query_head = query + (uint64_t)head * SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION;
		float scores[4096];
		uint32_t count = 0u,seen = 0u;
		float maximum = -3.0e38f,total = 0.0f;
		float accumulator[SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION];
		for (step = 0; step < selected_count; step++)
		{
			uint32_t position = selected[step];
			const uint16_t *key;
			float score = 0.0f;
			if ( position == 0xffffffffu )
				continue;
			key = pool + (uint64_t)position * slot_elements;
			for (element = 0; element < SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION; element++)
				score += SparkMuseGlimmerValFromBf16(query_head[element]) * SparkMuseGlimmerValFromBf16(key[element]);
			scores[count++] = score * (1.0f / sqrtf((float)SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION));
			if ( scores[count - 1u] > maximum )
				maximum = scores[count - 1u];
		}
		for (element = 0; element < count; element++)
		{
			scores[element] = expf(scores[element] - maximum);
			total += scores[element];
		}
		for (element = 0; element < SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION; element++)
			accumulator[element] = 0.0f;
		for (step = 0; step < selected_count; step++)
		{
			uint32_t position = selected[step];
			const uint16_t *value;
			if ( position == 0xffffffffu )
				continue;
			value = pool + (uint64_t)position * slot_elements + (uint64_t)SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION;
			for (element = 0; element < SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION; element++)
				accumulator[element] += scores[seen] * SparkMuseGlimmerValFromBf16(value[element]);
			seen++;
		}
		for (element = 0; element < SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION; element++)
			output[(uint64_t)head * SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION + element] = SparkMuseGlimmerValBf16(accumulator[element] / total);
	}
}

static int SparkMuseGlimmerValCompareEqual(const char *check, const uint16_t *actual, const uint16_t *expected, uint64_t count)
{
	uint64_t index;
	for (index = 0; index < count; index++)
		if ( actual[index] != expected[index] )
		{
			fprintf(stderr,"muse_glimmer_validation failure=%s detail=bitwise_mismatch index=%llu actual=%04x expected=%04x\n",check,(unsigned long long)index,actual[index],expected[index]);
			return(1);
		}
	return(0);
}

static int SparkMuseGlimmerValCompareNearby(const char *check, const uint16_t *actual, const uint16_t *expected, uint64_t count, uint32_t allowed_ulp)
{
	uint64_t index,violations = 0u,worst = 0u;
	uint32_t worst_distance = 0u;
	for (index = 0; index < count; index++)
	{
		uint16_t a = actual[index],e = expected[index];
		uint16_t magnitude_a = a & 0x7fffu,magnitude_e = e & 0x7fffu;
		uint32_t distance;
		if ( a == e )
			continue;
		distance = magnitude_a > magnitude_e ? magnitude_a - magnitude_e : magnitude_e - magnitude_a;
		if ( distance > worst_distance )
		{
			worst_distance = distance;
			worst = index;
		}
		if ( distance > allowed_ulp )
			violations++;
	}
	if ( violations != 0u )
	{
		fprintf(stderr,"muse_glimmer_validation failure=%s detail=beyond_ulp violations=%llu worst_index=%llu worst_distance=%u actual=%04x expected=%04x\n",check,(unsigned long long)violations,(unsigned long long)worst,worst_distance,actual[worst],expected[worst]);
		return(1);
	}
	printf("muse_glimmer_validation check=%s tolerance=ulp%u worst_distance=%u\n",check,allowed_ulp,worst_distance);
	return(0);
}

static int SparkMuseGlimmerValCheckNorms(void)
{
	uint16_t *input,*weight,*output,*reference,*head_input,*head_output,*head_reference;
	const uint32_t rows = SPARK_MUSE_GLIMMER_VALIDATION_ROWS;
	uint32_t row;
	cudaError_t error;
	int failures = 0;
	error = cudaMallocManaged((void **)&input,(size_t)rows * SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION * 2u,cudaMemAttachGlobal);
	error = cudaMallocManaged((void **)&weight,SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION * 2u,cudaMemAttachGlobal);
	error = cudaMallocManaged((void **)&output,(size_t)rows * SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION * 2u,cudaMemAttachGlobal);
	error = cudaMallocManaged((void **)&reference,(size_t)rows * SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION * 2u,cudaMemAttachGlobal);
	if ( SparkMuseGlimmerValCuda(error,"norm_alloc") != 0 )
		return(1);
	for (row = 0; row < rows; row++)
		for (uint32_t index = 0; index < SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION; index++)
		{
			input[row * SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION + index] = SparkMuseGlimmerValBf16(SparkMuseGlimmerValUniform(1.0f));
			if ( row == 0 )
				weight[index] = SparkMuseGlimmerValBf16(SparkMuseGlimmerValUniform(0.125f));
		}
	error = SparkMuseGlimmerLaunchCenteredRmsNorm(0,input,weight,output,rows,SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION,SPARK_MUSE_GLIMMER_MODEL_RMS_NORM_EPSILON);
	if ( SparkMuseGlimmerValCuda(error,"centered_norm_launch") != 0 )
		return(1);
	if ( SparkMuseGlimmerValCuda(cudaDeviceSynchronize(),"centered_norm_sync") != 0 )
		return(1);
	for (row = 0; row < rows; row++)
		SparkMuseGlimmerValReferenceCenteredNorm(input + (size_t)row * SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION,weight,reference + (size_t)row * SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION,SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION,SPARK_MUSE_GLIMMER_MODEL_RMS_NORM_EPSILON);
	failures += SparkMuseGlimmerValCompareNearby("centered_norm",output,reference,(uint64_t)rows * SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION,1u);
	error = cudaMallocManaged((void **)&head_input,SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION * 2u,cudaMemAttachGlobal);
	error = cudaMallocManaged((void **)&head_output,SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION * 2u,cudaMemAttachGlobal);
	error = cudaMallocManaged((void **)&head_reference,SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION * 2u,cudaMemAttachGlobal);
	for (uint32_t index = 0; index < SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION; index++)
		head_input[index] = SparkMuseGlimmerValBf16(SparkMuseGlimmerValUniform(1.0f));
	error = SparkMuseGlimmerLaunchHeadRmsNorm(0,head_input,0,head_output,1u,1u,SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION,SPARK_MUSE_GLIMMER_MODEL_ATTN_QK_NORM_EPSILON,SPARK_MUSE_GLIMMER_MODEL_ATTN_QK_SCALE_FACTOR);
	if ( SparkMuseGlimmerValCuda(error,"head_norm_launch") != 0 )
		return(1);
	if ( SparkMuseGlimmerValCuda(cudaDeviceSynchronize(),"head_norm_sync") != 0 )
		return(1);
	SparkMuseGlimmerValReferenceHeadNorm(head_input,head_reference,SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION,SPARK_MUSE_GLIMMER_MODEL_ATTN_QK_NORM_EPSILON,SPARK_MUSE_GLIMMER_MODEL_ATTN_QK_SCALE_FACTOR);
	failures += SparkMuseGlimmerValCompareNearby("qk_norm_3_87",head_output,head_reference,SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION,1u);
	cudaFree(input);
	cudaFree(weight);
	cudaFree(output);
	cudaFree(reference);
	cudaFree(head_input);
	cudaFree(head_output);
	cudaFree(head_reference);
	if ( failures == 0 )
		printf("muse_glimmer_validation check=centered_norm_and_qk_norm rows=%u bitwise=exact\n",rows);
	return(failures);
}

static int SparkMuseGlimmerValCheckWindowWalk(void)
{
	const uint32_t context = SPARK_MUSE_GLIMMER_VALIDATION_WINDOW_CONTEXT;
	const uint64_t slot_elements = 2ull * SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION;
	uint16_t *pool_host,*query,*head_out,*reference_out,*key,*value;
	uint32_t *positions,*context_lengths,*window,*sequence,*all_positions,*page_table;
	const uint32_t page_count = (context + SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS - 1u) / SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
	SparkMuseGlimmerKvViewShim view;
	SparkMuseGlimmerFrameErrorShim *error_record;
	uint32_t position = context - 1u;
	uint64_t pool_elements = (uint64_t)context * slot_elements;
	cudaError_t error;
	int failures = 0;
	error = cudaMallocManaged((void **)&pool_host,pool_elements * 2u,cudaMemAttachGlobal);
	if ( SparkMuseGlimmerValCuda(error,"window_alloc") != 0 )
		return(1);
	error = cudaMallocManaged((void **)&error_record,sizeof(*error_record),cudaMemAttachGlobal);
	error = cudaMallocManaged((void **)&page_table,page_count * sizeof(uint32_t),cudaMemAttachGlobal);
	error = cudaMallocManaged((void **)&query,SPARK_MUSE_GLIMMER_VALIDATION_LOCAL_HEADS * SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION * 2u,cudaMemAttachGlobal);
	error = cudaMallocManaged((void **)&head_out,SPARK_MUSE_GLIMMER_VALIDATION_LOCAL_HEADS * SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION * 2u,cudaMemAttachGlobal);
	error = cudaMallocManaged((void **)&reference_out,SPARK_MUSE_GLIMMER_VALIDATION_LOCAL_HEADS * SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION * 2u,cudaMemAttachGlobal);
	error = cudaMallocManaged((void **)&positions,sizeof(uint32_t),cudaMemAttachGlobal);
	error = cudaMallocManaged((void **)&context_lengths,sizeof(uint32_t),cudaMemAttachGlobal);
	error = cudaMallocManaged((void **)&sequence,sizeof(uint32_t),cudaMemAttachGlobal);
	error = cudaMallocManaged((void **)&window,SPARK_MUSE_GLIMMER_MODEL_SLIDING_WINDOW * sizeof(uint32_t),cudaMemAttachGlobal);
	error = cudaMallocManaged((void **)&all_positions,(size_t)context * sizeof(uint32_t),cudaMemAttachGlobal);
	error = cudaMallocManaged((void **)&key,SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION * 2u,cudaMemAttachGlobal);
	error = cudaMallocManaged((void **)&value,SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION * 2u,cudaMemAttachGlobal);
	for (uint64_t index = 0; index < pool_elements; index++)
		pool_host[index] = SparkMuseGlimmerValBf16(SparkMuseGlimmerValUniform(0.5f));
	for (uint32_t index = 0; index < SPARK_MUSE_GLIMMER_VALIDATION_LOCAL_HEADS * SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION; index++)
		query[index] = SparkMuseGlimmerValBf16(SparkMuseGlimmerValUniform(0.5f));
	for (uint32_t index = 0; index < context; index++)
		all_positions[index] = index;
	*positions = position;
	*context_lengths = context;
	*sequence = 0u;
	error_record->error_code = 0u;
	for (uint32_t page = 0; page < page_count; page++)
		page_table[page] = page;
	view.pool = pool_host;
	view.page_table = page_table;
	view.page_table_stride = page_count;
	view.sequence_count = 1u;
	view.pool_page_count = page_count;
	view.access_error = error_record;
	for (uint32_t element = 0; element < SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION; element++)
	{
		key[element] = pool_host[((uint64_t)position * slot_elements) + element];
		value[element] = pool_host[((uint64_t)position * slot_elements) + SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION + element];
	}
	error = SparkMuseGlimmerLaunchKvStore(0,&view,0u,key,value,sequence,positions,1u,1u);
	if ( SparkMuseGlimmerValCuda(error,"window_store_launch") != 0 )
		return(1);
	error = SparkMuseGlimmerLaunchWindowPositions(0,sequence,context_lengths,positions,1u,window);
	if ( SparkMuseGlimmerValCuda(error,"window_build_launch") != 0 )
		return(1);
	if ( SparkMuseGlimmerValCuda(cudaDeviceSynchronize(),"window_sync") != 0 )
		return(1);
	if ( window[0] != position + 1u - SPARK_MUSE_GLIMMER_MODEL_SLIDING_WINDOW )
		return(SparkMuseGlimmerValFail("window_walk","first_selected_position_wrong"));
	if ( window[SPARK_MUSE_GLIMMER_MODEL_SLIDING_WINDOW - 1u] != position )
		return(SparkMuseGlimmerValFail("window_walk","last_selected_position_wrong"));
	error = SparkMuseGlimmerLaunchAttentionDecode(0,&view,0u,query,sequence,context_lengths,window,positions,head_out,1u,SPARK_MUSE_GLIMMER_VALIDATION_LOCAL_HEADS,1u);
	if ( SparkMuseGlimmerValCuda(error,"window_decode_launch") != 0 )
		return(1);
	if ( SparkMuseGlimmerValCuda(cudaDeviceSynchronize(),"window_decode_sync") != 0 )
		return(1);
	SparkMuseGlimmerValReferenceDecode(query,pool_host,window,SPARK_MUSE_GLIMMER_MODEL_SLIDING_WINDOW,reference_out,slot_elements);
	failures += SparkMuseGlimmerValCompareNearby("window_decode",head_out,reference_out,SPARK_MUSE_GLIMMER_VALIDATION_LOCAL_HEADS * SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION,2u);
	error = SparkMuseGlimmerLaunchAttentionDecode(0,&view,0u,query,sequence,context_lengths,0,positions,head_out,1u,SPARK_MUSE_GLIMMER_VALIDATION_LOCAL_HEADS,1u);
	if ( SparkMuseGlimmerValCuda(error,"full_decode_launch") != 0 )
		return(1);
	if ( SparkMuseGlimmerValCuda(cudaDeviceSynchronize(),"full_decode_sync") != 0 )
		return(1);
	SparkMuseGlimmerValReferenceDecode(query,pool_host,all_positions,context,reference_out,slot_elements);
	failures += SparkMuseGlimmerValCompareNearby("full_decode",head_out,reference_out,SPARK_MUSE_GLIMMER_VALIDATION_LOCAL_HEADS * SPARK_MUSE_GLIMMER_MODEL_ATTN_HEAD_DIMENSION,2u);
	if ( error_record->error_code != 0u )
		return(SparkMuseGlimmerValFail("kv_access","reported_failure"));
	cudaFree(pool_host);
	cudaFree(query);
	cudaFree(head_out);
	cudaFree(reference_out);
	cudaFree(positions);
	cudaFree(context_lengths);
	cudaFree(sequence);
	cudaFree(window);
	cudaFree(all_positions);
	cudaFree(page_table);
	cudaFree(key);
	cudaFree(value);
	cudaFree(error_record);
	if ( failures == 0 )
		printf("muse_glimmer_validation check=window_walk context=%u window=%u boundary_drop=verified decode=bitwise\n",context,SPARK_MUSE_GLIMMER_MODEL_SLIDING_WINDOW);
	return(failures);
}

static int SparkMuseGlimmerValCheckGateAndSilu(void)
{
	uint16_t *head_out,*gate,*gate_up,*intermediate,*reference;
	const uint32_t local_query = SPARK_MUSE_GLIMMER_MODEL_ATTN_LOCAL_QUERY_DIMENSION(16u);
	const uint32_t local_intermediate = SPARK_MUSE_GLIMMER_MODEL_MLP_LOCAL_INTERMEDIATE(16u);
	cudaError_t error;
	int failures = 0;
	error = cudaMallocManaged((void **)&head_out,local_query * 2u,cudaMemAttachGlobal);
	error = cudaMallocManaged((void **)&gate,local_query * 2u,cudaMemAttachGlobal);
	error = cudaMallocManaged((void **)&gate_up,2u * local_intermediate * 2u,cudaMemAttachGlobal);
	error = cudaMallocManaged((void **)&intermediate,local_intermediate * 2u,cudaMemAttachGlobal);
	error = cudaMallocManaged((void **)&reference,local_intermediate * 2u,cudaMemAttachGlobal);
	if ( SparkMuseGlimmerValCuda(error,"gate_silu_alloc") != 0 )
		return(1);
	for (uint32_t index = 0; index < local_query; index++)
	{
		head_out[index] = SparkMuseGlimmerValBf16(SparkMuseGlimmerValUniform(1.0f));
		gate[index] = SparkMuseGlimmerValBf16(SparkMuseGlimmerValUniform(1.0f));
	}
	for (uint32_t index = 0; index < 2u * local_intermediate; index++)
		gate_up[index] = SparkMuseGlimmerValBf16(SparkMuseGlimmerValUniform(1.0f));
	if ( SparkMuseGlimmerValCuda(SparkMuseGlimmerLaunchOutputGate(0,head_out,gate,1u,local_query),"gate_launch") != 0 )
		return(1);
	if ( SparkMuseGlimmerValCuda(SparkMuseGlimmerLaunchSiluMul(0,gate_up,intermediate,1u,local_intermediate),"silu_launch") != 0 )
		return(1);
	if ( SparkMuseGlimmerValCuda(cudaDeviceSynchronize(),"gate_silu_sync") != 0 )
		return(1);
	for (uint32_t index = 0; index < local_intermediate; index++)
	{
		float gate_value = SparkMuseGlimmerValFromBf16(gate_up[index]);
		float up_value = SparkMuseGlimmerValFromBf16(gate_up[local_intermediate + index]);
		reference[index] = SparkMuseGlimmerValBf16(up_value * (gate_value / (1.0f + expf(-gate_value))));
	}
	failures += SparkMuseGlimmerValCompareEqual("silu_mul",intermediate,reference,local_intermediate);
	cudaFree(head_out);
	cudaFree(gate);
	cudaFree(gate_up);
	cudaFree(intermediate);
	cudaFree(reference);
	if ( failures == 0 )
		printf("muse_glimmer_validation check=output_gate_and_silu_mul bitwise=exact\n");
	return(failures);
}

typedef struct SparkMuseGlimmerValCapture
{
	uint16_t hidden[SPARK_MUSE_GLIMMER_MODEL_HIDDEN_DIMENSION];
	uint32_t received;
} SparkMuseGlimmerValCapture;

static SparkStatus SparkMuseGlimmerValCaptureSend(SparkHiddenTransportSession *session, const SparkHiddenTransportPacket *packet)
{
	SparkMuseGlimmerValCapture *capture = (SparkMuseGlimmerValCapture *)session;
	if ( packet->hidden_bf16 == 0 || packet->active_sequence_count < 1u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (packet->flags & SPARK_HIDDEN_TRANSPORT_PACKET_FLAG_DEVICE_POINTER) != 0u )
	{
		cudaError_t error = cudaMemcpy(capture->hidden,packet->hidden_bf16,SPARK_MUSE_GLIMMER_MODEL_HIDDEN_BF16_BYTES,cudaMemcpyDeviceToHost);
		if ( error != cudaSuccess )
			return(SPARK_STATUS_IO_ERROR);
	}
	else
		memcpy(capture->hidden,packet->hidden_bf16,SPARK_MUSE_GLIMMER_MODEL_HIDDEN_BF16_BYTES);
	capture->received++;
	return(SPARK_STATUS_OK);
}

typedef struct SparkMuseGlimmerValModule
{
	void *state;
	uint32_t token_ids[SPARK_MUSE_GLIMMER_VALIDATION_KV_LANES];
	uint32_t output_token_ids[SPARK_MUSE_GLIMMER_VALIDATION_KV_LANES];
	uint32_t head_stage;
	uint32_t host_blocks[SPARK_MUSE_GLIMMER_VALIDATION_KV_LANES];
	uint32_t host_counts[SPARK_MUSE_GLIMMER_VALIDATION_KV_LANES];
	uint32_t *device_blocks;
	uint32_t *device_counts;
	SparkMuseGlimmerKvBlockTableView table;
	SparkMuseGlimmerResidentDecodeStageFrameContext context;
	SparkModelDriverBuffer buffers[2];
	SparkModelDriverFrame frame;
	SparkMuseGlimmerValCapture capture;
} SparkMuseGlimmerValModule;

static int SparkMuseGlimmerValModuleInitialize(SparkMuseGlimmerValModule *module)
{
	SparkFirmwareModuleConfiguration configuration;
	SparkFirmwareModuleHostServices host_services;
	SparkStatus status;
	const char *stage_count_text;
	uint32_t lane;
	cudaError_t error;
	memset(module,0,sizeof(*module));
	stage_count_text = getenv("SPARK_MUSE_GLIMMER_STAGE_COUNT");
	module->head_stage = stage_count_text != 0 && strcmp(stage_count_text,"1") == 0 ? 1u : 0u;
	for (lane = 0u; lane < SPARK_MUSE_GLIMMER_VALIDATION_KV_LANES; lane++)
	{
		module->host_blocks[lane] = lane;
		module->host_counts[lane] = 1u;
	}
	error = cudaMalloc((void **)&module->device_blocks,sizeof(module->host_blocks));
	if ( error == cudaSuccess )
		error = cudaMemcpy(module->device_blocks,module->host_blocks,sizeof(module->host_blocks),cudaMemcpyHostToDevice);
	if ( error == cudaSuccess )
		error = cudaMalloc((void **)&module->device_counts,sizeof(module->host_counts));
	if ( error == cudaSuccess )
		error = cudaMemcpy(module->device_counts,module->host_counts,sizeof(module->host_counts),cudaMemcpyHostToDevice);
	if ( SparkMuseGlimmerValCuda(error,"module_table_alloc") != 0 )
		return(1);
	module->table.abi_version = SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_KV_BLOCK_TABLE_ABI_VERSION;
	module->table.descriptor_bytes = sizeof(module->table);
	module->table.block_token_count = SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS;
	module->table.lane_count = SPARK_MUSE_GLIMMER_VALIDATION_KV_LANES;
	module->table.lane_stride = 1u;
	module->table.lane_capacity = SPARK_MUSE_GLIMMER_VALIDATION_KV_LANES;
	module->table.physical_block_indices = module->device_blocks;
	module->table.lane_physical_block_counts = module->device_counts;
	module->table.host_physical_block_indices = module->host_blocks;
	module->table.host_lane_physical_block_counts = module->host_counts;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_FIRMWARE_MODULE_ABI_VERSION;
	configuration.descriptor_bytes = sizeof(configuration);
	configuration.model_id = "meta-models/Muse-Glimmer-30B";
	configuration.model_revision = "validation";
	configuration.stage_name = "muse_glimmer_resident_decode_stage";
	configuration.program_name = "resident_decode";
	configuration.operation_name = "muse_glimmer_resident_decode_stage";
	configuration.configuration_json = "{}";
	configuration.configuration_json_bytes = 2u;
	memset(&host_services,0,sizeof(host_services));
	host_services.abi_version = SPARK_FIRMWARE_MODULE_HOST_SERVICES_ABI_VERSION;
	host_services.descriptor_bytes = sizeof(host_services);
	host_services.node_id = "spark-muse-validator";
	host_services.node_target = "cuda.sm121.muse_glimmer.resident_decode_stage.bf16";
	host_services.execution_stream = (void *)cudaStreamPerThread;
	status = SparkMuseGlimmerResidentDecodeStageInitialize(&configuration,&host_services,&module->state);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"muse_glimmer_validation failure=module_initialize status=%d\n",(int)status);
		return(1);
	}
	return(0);
}

static int SparkMuseGlimmerValModuleExecute(SparkMuseGlimmerValModule *module, uint32_t rows, uint32_t position)
{
	SparkStatus status;
	memset(&module->context,0,sizeof(module->context));
	memset(&module->frame,0,sizeof(module->frame));
	module->context.abi_version = SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION;
	module->context.descriptor_bytes = sizeof(module->context);
	module->context.flags =
		SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE |
		SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW |
		(module->head_stage == 0u
			? SPARK_MUSE_GLIMMER_RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT
			: 0u);
	module->context.kv_block_table = &module->table;
	module->context.hidden_output_transport_session = module->head_stage != 0u ? 0 : (SparkHiddenTransportSession *)&module->capture;
	module->context.hidden_output_send_function = module->head_stage != 0u ? 0 : SparkMuseGlimmerValCaptureSend;
	module->frame.program_id = 1u;
	module->frame.tokens_per_sequence = 1u;
	module->frame.request_id = 1u;
	module->frame.sequence_id = 1u;
	module->frame.sequence_position = position;
	module->frame.active_slot_count = rows;
	module->frame.new_token_count = rows;
	module->frame.execution_stream = (void *)cudaStreamPerThread;
	module->frame.buffers = module->buffers;
	module->frame.buffer_count = module->head_stage != 0u ? 2u : 1u;
	module->frame.user_context = &module->context;
	memset(module->buffers,0,sizeof(module->buffers));
	module->buffers[0].flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_READ;
	module->buffers[0].address = module->token_ids;
	module->buffers[0].bytes = rows * sizeof(uint32_t);
	if ( module->head_stage != 0u )
	{
		module->buffers[1].slot = 1u;
		module->buffers[1].flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE;
		module->buffers[1].address = module->output_token_ids;
		module->buffers[1].bytes = sizeof(module->output_token_ids);
	}
	status = SparkMuseGlimmerResidentDecodeStageExecute(module->state,&module->frame);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"muse_glimmer_validation failure=module_execute rows=%u position=%u status=%d\n",rows,position,(int)status);
		return(1);
	}
	return(0);
}

static int SparkMuseGlimmerValCheckModule(void)
{
	SparkMuseGlimmerValModule module;
	uint32_t run,step,row;
	static uint32_t first_tokens[SPARK_MUSE_GLIMMER_VALIDATION_STEPS];
	static uint32_t second_tokens[SPARK_MUSE_GLIMMER_VALIDATION_STEPS];
	if ( SparkMuseGlimmerValModuleInitialize(&module) != 0 )
		return(1);
	for (run = 0; run < 2u; run++)
	{
		for (row = 0; row < SPARK_MUSE_GLIMMER_VALIDATION_KV_LANES; row++)
			module.token_ids[row] = 200000u + ((run * 7u + row * 3u) % 97u);
		for (step = 0; step < SPARK_MUSE_GLIMMER_VALIDATION_STEPS; step++)
		{
			module.capture.received = 0u;
			if ( SparkMuseGlimmerValModuleExecute(&module,1u,step) != 0 )
				return(1);
			if ( run == 0u )
				first_tokens[step] = module.output_token_ids[0];
			else
				second_tokens[step] = module.output_token_ids[0];
		}
	}
	for (step = 0; step < SPARK_MUSE_GLIMMER_VALIDATION_STEPS; step++)
	{
		if ( first_tokens[step] != second_tokens[step] )
			return(SparkMuseGlimmerValFail("module_determinism","token_streams_diverge"));
		if ( first_tokens[step] >= SPARK_MUSE_GLIMMER_MODEL_OUTPUT_VOCAB_COUNT )
			return(SparkMuseGlimmerValFail("module_tokens","token_out_of_range"));
	}
	printf("muse_glimmer_validation check=module_decode steps=%u deterministic=exact tokens=",SPARK_MUSE_GLIMMER_VALIDATION_STEPS);
	for (step = 0; step < SPARK_MUSE_GLIMMER_VALIDATION_STEPS; step++)
		printf("%u ",first_tokens[step]);
	printf("\n");
	SparkMuseGlimmerResidentDecodeStageDestroy(module.state);
	return(0);
}

int main(int argc, char **argv)
{
	int failures = 0;
	uint32_t view_bytes;
	if ( argc != 2 )
	{
		fprintf(stderr,"usage: %s CONFIGURATION_SHA256\n",argv[0]);
		return(2);
	}
	printf("muse_glimmer_validation configuration_sha256=%s\n",argv[1]);
	view_bytes = SparkMuseGlimmerKvViewBytes();
	if ( view_bytes != sizeof(SparkMuseGlimmerKvViewShim) )
		return(SparkMuseGlimmerValFail("kv_view_abi","size_mismatch"));
	failures += SparkMuseGlimmerValCheckNorms();
	failures += SparkMuseGlimmerValCheckWindowWalk();
	failures += SparkMuseGlimmerValCheckGateAndSilu();
	failures += SparkMuseGlimmerValCheckModule();
	if ( failures != 0 )
	{
		fprintf(stderr,"muse_glimmer_validation FAIL failures=%d\n",failures);
		return(1);
	}
	printf("muse_glimmer_validation PASS\n");
	return(0);
}
