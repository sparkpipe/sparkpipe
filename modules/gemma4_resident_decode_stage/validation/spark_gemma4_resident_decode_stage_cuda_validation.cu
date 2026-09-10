#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_gemma4_model.h"
#include "sparkpipe/spark_gemma4_resident_decode_stage_firmware.h"
#include "inference/kernels/kv.cuh"


#define SPARK_GEMMA4_VAL_ROWS 4u
#define SPARK_GEMMA4_VAL_SLIDING_CONTEXT 1030u
#define SPARK_GEMMA4_VAL_FULL_CONTEXT 1200u
#define SPARK_GEMMA4_VAL_WINDOW SPARK_GEMMA4_MODEL_SLIDING_WINDOW_TOKENS
#define SPARK_GEMMA4_VAL_PAGE_SLOTS SPARK_GEMMA4_RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS
#define SPARK_GEMMA4_VAL_KV_PAGES 32u
#define SPARK_GEMMA4_VAL_POOL_TOKENS (SPARK_GEMMA4_VAL_PAGE_SLOTS * SPARK_GEMMA4_VAL_KV_PAGES)
#define SPARK_GEMMA4_VAL_CHAIN_ROWS 2u
#define SPARK_GEMMA4_VAL_CHAIN_BASE 2000u

typedef LmKvGeometry<SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION * 4u,SPARK_GEMMA4_VAL_PAGE_SLOTS,true> SparkGemma4ValSlidingGeometry1;
typedef LmKvGeometry<SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION * 8u,SPARK_GEMMA4_VAL_PAGE_SLOTS,true> SparkGemma4ValSlidingGeometry2;
typedef LmKvGeometry<SPARK_GEMMA4_MODEL_FULL_HEAD_DIMENSION * 4u,SPARK_GEMMA4_VAL_PAGE_SLOTS,true> SparkGemma4ValFullGeometry;

extern "C" cudaError_t SparkGemma4ConfigureCudaKernels(void);
extern "C" cudaError_t SparkGemma4LaunchEmbeddingGatherShardedScaled(cudaStream_t stream, const uint32_t *token_ids, const void *embedding_bf16, void *hidden_bf16, uint32_t row_count, uint32_t vocab_base, uint32_t vocab_rows);
extern "C" cudaError_t SparkGemma4LaunchFusedResidualRmsNorm(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon);
extern "C" cudaError_t SparkGemma4LaunchRmsNorm(cudaStream_t stream, const void *input_bf16, const void *gain_bf16, void *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon);
extern "C" cudaError_t SparkGemma4LaunchHeadRmsNorm(cudaStream_t stream, const void *input_bf16, const void *weight_or_null_bf16, void *output_bf16, uint32_t row_count, uint32_t heads, uint32_t head_dimension, float epsilon);
extern "C" cudaError_t SparkGemma4LaunchLinear(cudaStream_t stream, const SparkGemma4LinearView *view, const void *input_bf16, void *output_bf16, uint32_t row_count);
extern "C" cudaError_t SparkGemma4LaunchResidualAdd(cudaStream_t stream, void *hidden_bf16, const void *delta_bf16, uint32_t row_count, uint32_t dimension);
extern "C" cudaError_t SparkGemma4LaunchBranchAdd(cudaStream_t stream, void *sum_bf16, const void *delta_bf16, uint32_t row_count, uint32_t dimension);
extern "C" cudaError_t SparkGemma4LaunchGatedGelu(cudaStream_t stream, void *gate_up_bf16, uint32_t row_count, uint32_t intermediate);
extern "C" cudaError_t SparkGemma4LaunchSlidingRope(cudaStream_t stream, void *q_bf16, const uint32_t *positions, uint32_t row_count, uint32_t heads, uint32_t head_dimension, uint32_t rope_dimension, float theta);
extern "C" cudaError_t SparkGemma4LaunchFullRope(cudaStream_t stream, void *q_bf16, const uint32_t *positions, const float *inv_freq_table, uint32_t row_count, uint32_t heads, uint32_t head_dimension, uint32_t rope_dimension);
extern "C" cudaError_t SparkGemma4LaunchSlidingWindowPositions(cudaStream_t stream, const uint32_t *sequence_of_row, const uint32_t *context_lengths, const uint32_t *row_positions, uint32_t row_count, uint32_t *positions_out);
extern "C" cudaError_t SparkGemma4LaunchKvStoreSliding(cudaStream_t stream, void *pool, const uint32_t *page_table, uint32_t page_table_stride, uint32_t sequence_count, uint32_t pool_page_count, void *access_error, const void *key_bf16, const void *value_bf16, const uint32_t *sequence_of_row, const uint32_t *positions, uint32_t row_count, uint32_t kv_heads);
extern "C" cudaError_t SparkGemma4LaunchKvStoreFull(cudaStream_t stream, void *pool, const uint32_t *page_table, uint32_t page_table_stride, uint32_t sequence_count, uint32_t pool_page_count, void *access_error, const void *key_bf16, const void *value_bf16, const uint32_t *sequence_of_row, const uint32_t *positions, uint32_t row_count);
extern "C" cudaError_t SparkGemma4LaunchAttentionDecodeSliding(cudaStream_t stream, void *pool, const uint32_t *page_table, uint32_t page_table_stride, uint32_t sequence_count, uint32_t pool_page_count, void *access_error, const void *query_bf16, const uint32_t *sequence_of_row, const uint32_t *context_lengths, const uint32_t *window_positions, uint32_t query_heads, void *output_bf16, uint32_t row_count, uint32_t kv_heads);
extern "C" cudaError_t SparkGemma4LaunchAttentionDecodeFull(cudaStream_t stream, void *pool, const uint32_t *page_table, uint32_t page_table_stride, uint32_t sequence_count, uint32_t pool_page_count, void *access_error, const void *query_bf16, const uint32_t *sequence_of_row, const uint32_t *context_lengths, uint32_t query_heads, void *output_bf16, uint32_t row_count);
extern "C" cudaError_t SparkGemma4LaunchTpCombineAdd(cudaStream_t stream, void *destination_bf16, const void *source_bf16, uint32_t row_count, uint32_t width);
extern "C" cudaError_t SparkGemma4LaunchHeadMaxLocPack(cudaStream_t stream, const float *scores_f32, const uint32_t *token_ids_u32, uint64_t *keys_u64, uint32_t row_count);
extern "C" cudaError_t SparkGemma4LaunchHeadMaxLocUnpack(cudaStream_t stream, const uint64_t *keys_u64, uint32_t *token_ids_u32, uint32_t row_count);
extern "C" cudaError_t SparkGemma4LaunchTpCombineU64Max(cudaStream_t stream, uint64_t *destination, const uint64_t *source, uint32_t element_count);
#if SPARK_GEMMA4_MODEL_MOE_BLOCK
extern "C" cudaError_t SparkGemma4LaunchRouterSoftmax(cudaStream_t stream, float *scores_f32, uint32_t row_count);
extern "C" cudaError_t SparkGemma4LaunchRouterTopk(cudaStream_t stream, const float *scores_f32, uint32_t *indices_u32, float *weights_f32, uint32_t row_count);
#endif

static uint32_t gemma4_val_sites;
static uint32_t SparkGemma4ValRandomState;

static uint32_t SparkGemma4ValNext(void)
{
	SparkGemma4ValRandomState = SparkGemma4ValRandomState * 1664525u + 1013904223u;
	return(SparkGemma4ValRandomState >> 8u);
}

static float SparkGemma4ValUniform(float scale)
{
	return(((float)(int32_t)(SparkGemma4ValNext() & 0xffffu) - 32768.0f) * scale / 32768.0f);
}

static uint16_t SparkGemma4ValBf16(float value)
{
	uint32_t bits;
	uint16_t high;
	uint16_t low;
	uint16_t round_up;
	memcpy(&bits,&value,sizeof(bits));
	high = (uint16_t)((bits >> 16) & 0xFFFFu);
	low = (uint16_t)(bits & 0xFFFFu);
	round_up = (uint16_t)((low > 0x8000u) || ((low == 0x8000u) && ((high & 1u) == 1u)));
	high = (uint16_t)(high + round_up);
	return (uint16_t)(high & 0xFFFFu);
}

static float SparkGemma4ValFromBf16(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float converted;
	memcpy(&converted,&bits,sizeof(converted));
	return(converted);
}

static void SparkGemma4ValFillBf16(uint16_t *packed, uint64_t count, float scale)
{
	uint64_t index;
	for (index = 0u; index < count; index++)
		packed[index] = SparkGemma4ValBf16(SparkGemma4ValUniform(scale));
}

static int SparkGemma4ValFail(const char *check, const char *detail)
{
	fprintf(stderr,"gemma4_validation failure=%s detail=%s\n",check,detail);
	return(1);
}

static int SparkGemma4ValCuda(cudaError_t error, const char *check)
{
	if (error == cudaSuccess)
		return(0);
	fprintf(stderr,"gemma4_validation failure=%s cuda=%s\n",check,cudaGetErrorString(error));
	return(1);
}

typedef struct SparkGemma4ValMetrics
{
	double difference_l2;
	double reference_l2;
	double actual_l2;
	double dot;
	double maximum_absolute;
	uint64_t count;
} SparkGemma4ValMetrics;

static void SparkGemma4ValMeasure(SparkGemma4ValMetrics *metrics, const float *actual, const float *reference, uint64_t count)
{
	uint64_t index;
	double difference;
	memset(metrics,0,sizeof(*metrics));
	metrics->count = count;
	for (index = 0u; index < count; index++)
	{
		difference = (double)actual[index] - (double)reference[index];
		metrics->difference_l2 += difference * difference;
		metrics->reference_l2 += (double)reference[index] * (double)reference[index];
		metrics->actual_l2 += (double)actual[index] * (double)actual[index];
		metrics->dot += (double)actual[index] * (double)reference[index];
		if (fabs((double)actual[index] - (double)reference[index]) > metrics->maximum_absolute)
			metrics->maximum_absolute = fabs((double)actual[index] - (double)reference[index]);
	}
}

static int SparkGemma4ValReport(const char *check, const SparkGemma4ValMetrics *metrics, double max_relative_l2, double minimum_cosine)
{
	double relative_l2 = metrics->reference_l2 > 0.0
		? sqrt(metrics->difference_l2 / metrics->reference_l2) : INFINITY;
	double cosine = metrics->actual_l2 > 0.0 && metrics->reference_l2 > 0.0
		? metrics->dot / sqrt(metrics->actual_l2 * metrics->reference_l2) : 0.0;
	printf("gemma4_validation check=%s elements=%llu relative_l2=%.9g cosine=%.9g max_abs=%.9g\n",
		check,(unsigned long long)metrics->count,relative_l2,cosine,metrics->maximum_absolute);
	if (isfinite(relative_l2) == 0 || relative_l2 > max_relative_l2)
		return(SparkGemma4ValFail(check,"relative_l2"));
	if (cosine < minimum_cosine)
		return(SparkGemma4ValFail(check,"cosine"));
	gemma4_val_sites++;
	return(0);
}

static int SparkGemma4ValCompareBf16(const char *check, const uint16_t *actual, const uint16_t *expected, uint64_t count)
{
	float *actual_f = (float *)malloc(count * sizeof(float));
	float *expected_f = (float *)malloc(count * sizeof(float));
	uint64_t index;
	int result;
	SparkGemma4ValMetrics metrics;
	if (actual_f == 0 || expected_f == 0)
		return(SparkGemma4ValFail(check,"host_alloc"));
	for (index = 0u; index < count; index++)
	{
		actual_f[index] = SparkGemma4ValFromBf16(actual[index]);
		expected_f[index] = SparkGemma4ValFromBf16(expected[index]);
	}
	SparkGemma4ValMeasure(&metrics,actual_f,expected_f,count);
	result = SparkGemma4ValReport(check,&metrics,5e-3,0.999);
	free(actual_f);
	free(expected_f);
	return(result);
}

static int SparkGemma4ValCompareAgainstFloat(const char *check, const uint16_t *actual, const float *expected, uint64_t count)
{
	float *actual_f = (float *)malloc(count * sizeof(float));
	uint64_t index;
	int result;
	SparkGemma4ValMetrics metrics;
	if (actual_f == 0)
		return(SparkGemma4ValFail(check,"host_alloc"));
	for (index = 0u; index < count; index++)
		actual_f[index] = SparkGemma4ValFromBf16(actual[index]);
	SparkGemma4ValMeasure(&metrics,actual_f,expected,count);
	result = SparkGemma4ValReport(check,&metrics,5e-3,0.999);
	free(actual_f);
	return(result);
}

static cudaError_t SparkGemma4ValCopyUp(void *device, const void *host, uint64_t bytes)
{
	return(cudaMemcpy(device,host,bytes,cudaMemcpyHostToDevice));
}

static cudaError_t SparkGemma4ValCopyDown(void *host, const void *device, uint64_t bytes)
{
	return(cudaMemcpy(host,device,bytes,cudaMemcpyDeviceToHost));
}

static cudaError_t SparkGemma4ValSync(void)
{
	return(cudaStreamSynchronize(cudaStreamPerThread));
}

static void SparkGemma4ValMirrorRms(const uint16_t *input, const uint16_t *weight, uint16_t *output, uint32_t rows, uint32_t dimension)
{
	uint32_t row,element;
	for (row = 0u; row < rows; row++)
	{
		float variance = 0.0f,inverse;
		const uint16_t *src = input + ((uint64_t)row * dimension);
		uint16_t *dst = output + ((uint64_t)row * dimension);
		for (element = 0u; element < dimension; element++)
			variance += SparkGemma4ValFromBf16(src[element]) * SparkGemma4ValFromBf16(src[element]);
		inverse = 1.0f / sqrtf((variance / (float)dimension) + SPARK_GEMMA4_MODEL_RMS_NORM_EPSILON);
		for (element = 0u; element < dimension; element++)
		{
			float value = SparkGemma4ValFromBf16(src[element]) * inverse;
			if (weight != 0)
				value *= SparkGemma4ValFromBf16(weight[element]);
			dst[element] = SparkGemma4ValBf16(value);
		}
	}
}

static void SparkGemma4ValMirrorHeadRms(const uint16_t *input, const uint16_t *weight, uint16_t *output, uint32_t rows, uint32_t heads, uint32_t dimension)
{
	uint32_t row,head,element;
	for (row = 0u; row < rows; row++)
		for (head = 0u; head < heads; head++)
		{
			float variance = 0.0f,scale;
			const uint16_t *src = input + (((uint64_t)row * heads + head) * dimension);
			uint16_t *dst = output + (((uint64_t)row * heads + head) * dimension);
			for (element = 0u; element < dimension; element++)
				variance += SparkGemma4ValFromBf16(src[element]) * SparkGemma4ValFromBf16(src[element]);
			scale = 1.0f / sqrtf((variance / (float)dimension) + SPARK_GEMMA4_MODEL_RMS_NORM_EPSILON);
			for (element = 0u; element < dimension; element++)
			{
				float value = SparkGemma4ValFromBf16(src[element]) * scale;
				if (weight != 0)
					value *= SparkGemma4ValFromBf16(weight[element]);
				dst[element] = SparkGemma4ValBf16(value);
			}
		}
}

static void SparkGemma4ValMirrorLinear(const uint16_t *weight, const uint16_t *input, uint16_t *output, uint32_t rows, uint32_t input_dimension, uint32_t output_dimension)
{
	uint32_t row,out_index,element;
	for (row = 0u; row < rows; row++)
		for (out_index = 0u; out_index < output_dimension; out_index++)
		{
			float total = 0.0f;
			for (element = 0u; element < input_dimension; element++)
				total += SparkGemma4ValFromBf16(input[((uint64_t)row * input_dimension) + element])
					* SparkGemma4ValFromBf16(weight[((uint64_t)out_index * input_dimension) + element]);
			output[((uint64_t)row * output_dimension) + out_index] = SparkGemma4ValBf16(total);
		}
}

static float SparkGemma4ValGelu(float value)
{
	float cube = value * value * value;
	return(0.5f * value * (1.0f + tanhf(0.7978845608028654f * (value + 0.044715f * cube))));
}

static int SparkGemma4ValCheckSelf(void)
{
	if (SparkGemma4ValBf16(sqrtf(5376.0f)) != SparkGemma4ValBf16(73.5f)
		|| SparkGemma4ValBf16(sqrtf(2816.0f)) != SparkGemma4ValBf16(53.0f))
		return(SparkGemma4ValFail("self.embed_scale","bf16_sqrt"));
	if (SPARK_GEMMA4_MODEL_QK_SCALE != 1.0f || SPARK_GEMMA4_MODEL_RMS_NORM_EPSILON != 1e-06f)
		return(SparkGemma4ValFail("self.constants","qk_scale_epsilon"));
	if (SPARK_GEMMA4_MODEL_SLIDING_WINDOW_TOKENS != 1024u || SPARK_GEMMA4_MODEL_FULL_ROPE_TABLE_ELEMENTS != 256u)
		return(SparkGemma4ValFail("self.constants","window_table"));
	printf("gemma4_validation check=self_constants arm=%s embed_scale=%.1f qk_scale=%.1f window=%u PASS\n",
		SPARK_GEMMA4_MODEL_MODULE_ID,(double)SPARK_GEMMA4_MODEL_EMBED_SCALE,(double)SPARK_GEMMA4_MODEL_QK_SCALE,SPARK_GEMMA4_MODEL_SLIDING_WINDOW_TOKENS);
	gemma4_val_sites++;
	return(0);
}

static int SparkGemma4ValCheckEmbedding(void)
{
	const uint32_t rows = SPARK_GEMMA4_VAL_ROWS;
	const uint32_t vocab_base = 1000u;
	const uint32_t vocab_rows = 4096u;
	const uint64_t table_bytes = (uint64_t)vocab_rows * SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION * 2u;
	const uint64_t hidden_bytes = (uint64_t)rows * SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION * 2u;
	uint16_t *table = (uint16_t *)malloc(table_bytes);
	uint16_t *expected = (uint16_t *)malloc(hidden_bytes);
	uint16_t *actual = (uint16_t *)malloc(hidden_bytes);
	uint32_t tokens[SPARK_GEMMA4_VAL_ROWS] = {1000u,3000u,999u,5095u};
	uint32_t row,element;
	void *table_device = 0,*hidden_device = 0,*tokens_device = 0;
	cudaError_t error;
	if (table == 0 || expected == 0 || actual == 0)
		return(SparkGemma4ValFail("embedding_gather","host_alloc"));
	SparkGemma4ValRandomState = 101u;
	SparkGemma4ValFillBf16(table,(uint64_t)vocab_rows * SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION,0.25f);
	for (row = 0u; row < rows; row++)
		for (element = 0u; element < SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION; element++)
		{
			uint16_t source = (tokens[row] >= vocab_base && tokens[row] < vocab_base + vocab_rows)
				? table[((uint64_t)(tokens[row] - vocab_base) * SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION) + element] : 0u;
			expected[((uint64_t)row * SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION) + element] =
				SparkGemma4ValBf16(SparkGemma4ValFromBf16(source) * SPARK_GEMMA4_MODEL_EMBED_SCALE);
		}
	error = cudaMalloc(&table_device,table_bytes);
	if (error == cudaSuccess) error = cudaMalloc(&hidden_device,hidden_bytes);
	if (error == cudaSuccess) error = cudaMalloc(&tokens_device,sizeof(tokens));
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(table_device,table,table_bytes);
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(tokens_device,tokens,sizeof(tokens));
	if (error == cudaSuccess)
		error = SparkGemma4LaunchEmbeddingGatherShardedScaled(cudaStreamPerThread,(const uint32_t *)tokens_device,table_device,hidden_device,rows,vocab_base,vocab_rows);
	if (error == cudaSuccess) error = SparkGemma4ValSync();
	if (error == cudaSuccess) error = SparkGemma4ValCopyDown(actual,hidden_device,hidden_bytes);
	if (SparkGemma4ValCuda(error,"embedding_gather") != 0)
		return(1);
	cudaFree(table_device);
	cudaFree(hidden_device);
	cudaFree(tokens_device);
	if (memcmp(actual,expected,hidden_bytes) != 0)
		return(SparkGemma4ValFail("embedding_gather_scaled","bitwise"));
	printf("gemma4_validation check=embedding_gather_scaled rows=%u embed_scale=%.1f bit_exact=1\n",rows,(double)SPARK_GEMMA4_MODEL_EMBED_SCALE);
	gemma4_val_sites++;
	free(table);
	free(expected);
	free(actual);
	return(0);
}

static int SparkGemma4ValCheckNorms(void)
{
	const uint32_t rows = SPARK_GEMMA4_VAL_ROWS;
	const uint32_t dimension = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
	const uint64_t bytes = (uint64_t)rows * dimension * 2u;
	uint16_t *input = (uint16_t *)malloc(bytes);
	uint16_t *gain = (uint16_t *)malloc((uint64_t)dimension * 2u);
	uint16_t *delta = (uint16_t *)malloc(bytes);
	uint16_t *expected_normed = (uint16_t *)malloc(bytes);
	uint16_t *expected_hidden = (uint16_t *)malloc(bytes);
	uint16_t *actual = (uint16_t *)malloc(bytes);
	uint32_t row,element;
	void *input_device = 0,*gain_device = 0,*delta_device = 0,*hidden_device = 0,*output_device = 0;
	cudaError_t error;
	if (input == 0 || gain == 0 || delta == 0 || expected_normed == 0 || expected_hidden == 0 || actual == 0)
		return(SparkGemma4ValFail("norms","host_alloc"));
	SparkGemma4ValRandomState = 211u;
	SparkGemma4ValFillBf16(input,(uint64_t)rows * dimension,0.5f);
	SparkGemma4ValFillBf16(gain,dimension,1.0f);
	SparkGemma4ValFillBf16(delta,(uint64_t)rows * dimension,0.25f);
	SparkGemma4ValMirrorRms(input,gain,expected_normed,rows,dimension);
	for (row = 0u; row < rows; row++)
		for (element = 0u; element < dimension; element++)
		{
			uint64_t at = ((uint64_t)row * dimension) + element;
			expected_hidden[at] = SparkGemma4ValBf16(SparkGemma4ValFromBf16(input[at]) + SparkGemma4ValFromBf16(delta[at]));
		}
	error = cudaMalloc(&input_device,bytes);
	if (error == cudaSuccess) error = cudaMalloc(&gain_device,(uint64_t)dimension * 2u);
	if (error == cudaSuccess) error = cudaMalloc(&delta_device,bytes);
	if (error == cudaSuccess) error = cudaMalloc(&hidden_device,bytes);
	if (error == cudaSuccess) error = cudaMalloc(&output_device,bytes);
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(input_device,input,bytes);
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(gain_device,gain,(uint64_t)dimension * 2u);
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(delta_device,delta,bytes);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchRmsNorm(cudaStreamPerThread,input_device,gain_device,output_device,rows,dimension,SPARK_GEMMA4_MODEL_RMS_NORM_EPSILON);
	if (error == cudaSuccess) error = SparkGemma4ValSync();
	if (error == cudaSuccess) error = SparkGemma4ValCopyDown(actual,output_device,bytes);
	if (error == cudaSuccess && SparkGemma4ValCompareBf16("rms_norm",actual,expected_normed,(uint64_t)rows * dimension) != 0)
		return(1);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchFusedResidualRmsNorm(cudaStreamPerThread,hidden_device,delta_device,gain_device,output_device,rows,dimension,SPARK_GEMMA4_MODEL_RMS_NORM_EPSILON);
	if (error == cudaSuccess) error = SparkGemma4ValSync();
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(hidden_device,input,bytes);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchFusedResidualRmsNorm(cudaStreamPerThread,hidden_device,delta_device,gain_device,output_device,rows,dimension,SPARK_GEMMA4_MODEL_RMS_NORM_EPSILON);
	if (error == cudaSuccess) error = SparkGemma4ValSync();
	if (error == cudaSuccess) error = SparkGemma4ValCopyDown(actual,hidden_device,bytes);
	if (error == cudaSuccess && memcmp(actual,expected_hidden,bytes) != 0)
		return(SparkGemma4ValFail("fused_residual_hidden","bitwise"));
	if (error == cudaSuccess) error = SparkGemma4ValCopyDown(actual,output_device,bytes);
	SparkGemma4ValMirrorRms(expected_hidden,gain,expected_normed,rows,dimension);
	if (error == cudaSuccess && SparkGemma4ValCompareBf16("fused_residual_normed",actual,expected_normed,(uint64_t)rows * dimension) != 0)
		return(1);
	printf("gemma4_validation check=fused_residual_hidden bit_exact=1\n");
	gemma4_val_sites++;
	cudaFree(input_device);
	cudaFree(gain_device);
	cudaFree(delta_device);
	cudaFree(hidden_device);
	cudaFree(output_device);
	free(input);
	free(gain);
	free(delta);
	free(expected_normed);
	free(expected_hidden);
	free(actual);
	return(SparkGemma4ValCuda(error,"norms"));
}

static int SparkGemma4ValCheckHeadNorms(void)
{
	const uint32_t rows = 2u;
	const uint32_t heads = SPARK_GEMMA4_MODEL_SLIDING_QUERY_HEAD_COUNT;
	const uint32_t dimension = SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION;
	const uint64_t elements = (uint64_t)rows * heads * dimension;
	uint16_t *input = (uint16_t *)malloc(elements * 2u);
	uint16_t *weight = (uint16_t *)malloc((uint64_t)dimension * 2u);
	uint16_t *expected = (uint16_t *)malloc(elements * 2u);
	uint16_t *actual = (uint16_t *)malloc(elements * 2u);
	void *input_device = 0,*weight_device = 0,*output_device = 0;
	cudaError_t error;
	if (input == 0 || weight == 0 || expected == 0 || actual == 0)
		return(SparkGemma4ValFail("head_norms","host_alloc"));
	SparkGemma4ValRandomState = 307u;
	SparkGemma4ValFillBf16(input,elements,0.5f);
	SparkGemma4ValFillBf16(weight,dimension,1.0f);
	error = cudaMalloc(&input_device,elements * 2u);
	if (error == cudaSuccess) error = cudaMalloc(&weight_device,(uint64_t)dimension * 2u);
	if (error == cudaSuccess) error = cudaMalloc(&output_device,elements * 2u);
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(input_device,input,elements * 2u);
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(weight_device,weight,(uint64_t)dimension * 2u);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchHeadRmsNorm(cudaStreamPerThread,input_device,weight_device,output_device,rows,heads,dimension,SPARK_GEMMA4_MODEL_RMS_NORM_EPSILON);
	if (error == cudaSuccess) error = SparkGemma4ValSync();
	if (error == cudaSuccess) error = SparkGemma4ValCopyDown(actual,output_device,elements * 2u);
	SparkGemma4ValMirrorHeadRms(input,weight,expected,rows,heads,dimension);
	if (error == cudaSuccess && SparkGemma4ValCompareBf16("head_norm_weighted_qk",actual,expected,elements) != 0)
		return(1);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchHeadRmsNorm(cudaStreamPerThread,input_device,0,output_device,rows,heads,dimension,SPARK_GEMMA4_MODEL_RMS_NORM_EPSILON);
	if (error == cudaSuccess) error = SparkGemma4ValSync();
	if (error == cudaSuccess) error = SparkGemma4ValCopyDown(actual,output_device,elements * 2u);
	SparkGemma4ValMirrorHeadRms(input,0,expected,rows,heads,dimension);
	if (error == cudaSuccess && SparkGemma4ValCompareBf16("head_norm_scale_free_v_router",actual,expected,elements) != 0)
		return(1);
	cudaFree(input_device);
	cudaFree(weight_device);
	cudaFree(output_device);
	free(input);
	free(weight);
	free(expected);
	free(actual);
	return(SparkGemma4ValCuda(error,"head_norms"));
}

static int SparkGemma4ValLinearPhase(const char *tag, uint32_t rows, uint32_t in_dimension, uint32_t out_dimension, uint16_t *weight, uint16_t *input, uint16_t *expected, uint16_t *actual)
{
	SparkGemma4LinearView view;
	void *weight_device = 0,*input_device = 0,*output_device = 0;
	cudaError_t error;
	memset(&view,0,sizeof(view));
	view.abi_version = SPARK_GEMMA4_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION;
	view.weight_format = SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16;
	view.input_dimension = in_dimension;
	view.output_dimension = out_dimension;
	error = cudaMalloc(&weight_device,(uint64_t)out_dimension * in_dimension * 2u);
	if (error == cudaSuccess)
	{
		view.weight_payload = weight_device;
		view.weight_payload_bytes = (uint64_t)out_dimension * in_dimension * 2u;
	}
	if (error == cudaSuccess) error = cudaMalloc(&input_device,(uint64_t)rows * in_dimension * 2u);
	if (error == cudaSuccess) error = cudaMalloc(&output_device,(uint64_t)rows * out_dimension * 2u);
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(weight_device,weight,(uint64_t)out_dimension * in_dimension * 2u);
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(input_device,input,(uint64_t)rows * in_dimension * 2u);
	if (error == cudaSuccess) error = SparkGemma4LaunchLinear(cudaStreamPerThread,&view,input_device,output_device,rows);
	if (error == cudaSuccess) error = SparkGemma4ValSync();
	if (error == cudaSuccess) error = SparkGemma4ValCopyDown(actual,output_device,(uint64_t)rows * out_dimension * 2u);
	if (error != cudaSuccess)
	{
		cudaFree(weight_device);
		cudaFree(input_device);
		cudaFree(output_device);
		return(SparkGemma4ValCuda(error,tag));
	}
	cudaFree(weight_device);
	cudaFree(input_device);
	cudaFree(output_device);
	SparkGemma4ValMirrorLinear(weight,input,expected,rows,in_dimension,out_dimension);
	if (memcmp(actual,expected,(uint64_t)rows * out_dimension * 2u) == 0)
		printf("gemma4_validation check=%s rows=%u in=%u out=%u bit_exact=1\n",tag,rows,in_dimension,out_dimension);
	return(SparkGemma4ValCompareBf16(tag,actual,expected,(uint64_t)rows * out_dimension));
}

static int SparkGemma4ValCheckLinear(void)
{
	const uint32_t max_rows = 64u;
	const uint32_t max_in = 512u;
	const uint32_t max_out = 256u;
	uint16_t *weight = (uint16_t *)malloc((uint64_t)max_out * max_in * 2u);
	uint16_t *input = (uint16_t *)malloc((uint64_t)max_rows * max_in * 2u);
	uint16_t *expected = (uint16_t *)malloc((uint64_t)max_rows * max_out * 2u);
	uint16_t *actual = (uint16_t *)malloc((uint64_t)max_rows * max_out * 2u);
	int result;
	if (weight == 0 || input == 0 || expected == 0 || actual == 0)
		return(SparkGemma4ValFail("linear","host_alloc"));
	SparkGemma4ValRandomState = 401u;
	SparkGemma4ValFillBf16(weight,(uint64_t)max_out * max_in,0.08f);
	SparkGemma4ValFillBf16(input,(uint64_t)max_rows * max_in,0.5f);
	result = SparkGemma4ValLinearPhase("linear_tile16",16u,512u,256u,weight,input,expected,actual);
	if (result == 0)
		result = SparkGemma4ValLinearPhase("linear_mloop",64u,512u,256u,weight,input,expected,actual);
	if (result == 0)
		result = SparkGemma4ValLinearPhase("linear_scalar",4u,96u,64u,weight,input,expected,actual);
	free(weight);
	free(input);
	free(expected);
	free(actual);
	return(result);
}

static int SparkGemma4ValCheckAdds(void)
{
	const uint32_t rows = SPARK_GEMMA4_VAL_ROWS;
	const uint32_t dimension = 512u;
	const uint64_t elements = (uint64_t)rows * dimension;
	uint16_t *left = (uint16_t *)malloc(elements * 2u);
	uint16_t *right = (uint16_t *)malloc(elements * 2u);
	uint16_t *expected = (uint16_t *)malloc(elements * 2u);
	uint16_t *actual = (uint16_t *)malloc(elements * 2u);
	float scores[SPARK_GEMMA4_VAL_ROWS];
	uint32_t tokens[SPARK_GEMMA4_VAL_ROWS] = {7u,9u,3u,1u};
	uint64_t keys[SPARK_GEMMA4_VAL_ROWS];
	uint32_t row,i,j;
	void *left_device = 0,*right_device = 0,*scores_device = 0,*tokens_device = 0,*keys_device = 0,*u64_device = 0,*u64_source_device = 0;
	cudaError_t error;
	if (left == 0 || right == 0 || expected == 0 || actual == 0)
		return(SparkGemma4ValFail("adds","host_alloc"));
	SparkGemma4ValRandomState = 509u;
	SparkGemma4ValFillBf16(left,elements,1.0f);
	SparkGemma4ValFillBf16(right,elements,1.0f);
	for (row = 0u; row < rows; row++)
		for (i = 0u; i < dimension; i++)
			expected[((uint64_t)row * dimension) + i] = SparkGemma4ValBf16(
				SparkGemma4ValFromBf16(left[((uint64_t)row * dimension) + i]) + SparkGemma4ValFromBf16(right[((uint64_t)row * dimension) + i]));
	error = cudaMalloc(&left_device,elements * 2u);
	if (error == cudaSuccess) error = cudaMalloc(&right_device,elements * 2u);
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(left_device,left,elements * 2u);
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(right_device,right,elements * 2u);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchResidualAdd(cudaStreamPerThread,left_device,right_device,rows,dimension);
	if (error == cudaSuccess) error = SparkGemma4ValSync();
	if (error == cudaSuccess) error = SparkGemma4ValCopyDown(actual,left_device,elements * 2u);
	if (error == cudaSuccess && memcmp(actual,expected,elements * 2u) != 0)
		return(SparkGemma4ValFail("residual_add","bitwise"));
	if (error == cudaSuccess)
		error = SparkGemma4LaunchBranchAdd(cudaStreamPerThread,left_device,right_device,rows,dimension);
	if (error == cudaSuccess) error = SparkGemma4ValSync();
	if (error == cudaSuccess) error = SparkGemma4ValCopyDown(actual,left_device,elements * 2u);
	for (row = 0u; row < rows; row++)
		for (i = 0u; i < dimension; i++)
			expected[((uint64_t)row * dimension) + i] = SparkGemma4ValBf16(
				SparkGemma4ValFromBf16(expected[((uint64_t)row * dimension) + i]) + SparkGemma4ValFromBf16(right[((uint64_t)row * dimension) + i]));
	if (error == cudaSuccess && memcmp(actual,expected,elements * 2u) != 0)
		return(SparkGemma4ValFail("branch_add","bitwise"));
	printf("gemma4_validation check=residual_branch_adds rows=%u bit_exact=1\n",rows);
	gemma4_val_sites++;
	SparkGemma4ValRandomState = 613u;
	for (row = 0u; row < rows; row++)
		scores[row] = (row == 0u || row == 2u) ? 0.5f : SparkGemma4ValUniform(1.0f);
	error = cudaMalloc(&scores_device,sizeof(scores));
	if (error == cudaSuccess) error = cudaMalloc(&tokens_device,sizeof(tokens));
	if (error == cudaSuccess) error = cudaMalloc(&keys_device,sizeof(keys));
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(scores_device,scores,sizeof(scores));
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(tokens_device,tokens,sizeof(tokens));
	if (error == cudaSuccess)
		error = SparkGemma4LaunchHeadMaxLocPack(cudaStreamPerThread,(const float *)scores_device,(const uint32_t *)tokens_device,(uint64_t *)keys_device,rows);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchHeadMaxLocUnpack(cudaStreamPerThread,(const uint64_t *)keys_device,(uint32_t *)tokens_device,rows);
	if (error == cudaSuccess) error = SparkGemma4ValSync();
	if (error == cudaSuccess) error = SparkGemma4ValCopyDown(tokens,tokens_device,sizeof(tokens));
	if (error == cudaSuccess) error = SparkGemma4ValCopyDown(keys,keys_device,sizeof(keys));
	if (SparkGemma4ValCuda(error,"adds") != 0)
		return(1);
	if (tokens[0] != 7u || tokens[1] != 9u || tokens[2] != 3u || tokens[3] != 1u)
		return(SparkGemma4ValFail("head_maxloc_unpack","roundtrip"));
	{
		float sorted_scores[SPARK_GEMMA4_VAL_ROWS];
		uint32_t sorted_tokens[SPARK_GEMMA4_VAL_ROWS];
		for (i = 0u; i < rows; i++)
		{
			sorted_scores[i] = scores[i];
			sorted_tokens[i] = tokens[i];
		}
		for (i = 0u; i < rows; i++)
			for (j = i + 1u; j < rows; j++)
				if (keys[j] > keys[i])
				{
					uint64_t key_swap = keys[i];
					float score_swap = sorted_scores[i];
					uint32_t token_swap = sorted_tokens[i];
					keys[i] = keys[j];
					keys[j] = key_swap;
					sorted_scores[i] = sorted_scores[j];
					sorted_scores[j] = score_swap;
					sorted_tokens[i] = sorted_tokens[j];
					sorted_tokens[j] = token_swap;
				}
		for (i = 0u; i < rows; i++)
			for (j = i + 1u; j < rows; j++)
				if (sorted_scores[j] > sorted_scores[i]
					|| (sorted_scores[j] == sorted_scores[i] && sorted_tokens[j] < sorted_tokens[i]))
					return(SparkGemma4ValFail("head_maxloc_order","score_desc_token_asc"));
		printf("gemma4_validation check=head_maxloc_order tie_token=%u PASS\n",sorted_tokens[0]);
	}
	gemma4_val_sites++;
	{
		uint64_t u64_left[SPARK_GEMMA4_VAL_ROWS],u64_right[SPARK_GEMMA4_VAL_ROWS],u64_expected[SPARK_GEMMA4_VAL_ROWS];
		for (row = 0u; row < rows; row++)
		{
			u64_left[row] = 0x9e3779b97f4a7c15ull * (row + 1u);
			u64_right[row] = 0xbf58476d1ce4e5b9ull * (row + 3u);
			u64_expected[row] = u64_left[row] > u64_right[row] ? u64_left[row] : u64_right[row];
		}
		error = cudaMalloc(&u64_device,sizeof(u64_left));
		if (error == cudaSuccess) error = cudaMalloc(&u64_source_device,sizeof(u64_right));
		if (error == cudaSuccess) error = SparkGemma4ValCopyUp(u64_device,u64_left,sizeof(u64_left));
		if (error == cudaSuccess) error = SparkGemma4ValCopyUp(u64_source_device,u64_right,sizeof(u64_right));
		if (error == cudaSuccess)
			error = SparkGemma4LaunchTpCombineU64Max(cudaStreamPerThread,(uint64_t *)u64_device,(const uint64_t *)u64_source_device,rows);
		if (error == cudaSuccess) error = SparkGemma4ValSync();
		if (error == cudaSuccess) error = SparkGemma4ValCopyDown(u64_left,u64_device,sizeof(u64_left));
		if (SparkGemma4ValCuda(error,"adds") != 0)
			return(1);
		if (memcmp(u64_left,u64_expected,sizeof(u64_left)) != 0)
			return(SparkGemma4ValFail("tp_combine_u64_max","bitwise"));
	}
	printf("gemma4_validation check=tp_combine_u64_max bit_exact=1\n");
	gemma4_val_sites++;
	cudaFree(left_device);
	cudaFree(right_device);
	cudaFree(scores_device);
	cudaFree(tokens_device);
	cudaFree(keys_device);
	cudaFree(u64_device);
	cudaFree(u64_source_device);
	free(left);
	free(right);
	free(expected);
	free(actual);
	return(0);
}

static int SparkGemma4ValCheckGatedGelu(void)
{
	const uint32_t rows = 2u;
	const uint32_t intermediate = 1024u;
	const uint64_t elements = (uint64_t)rows * intermediate * 2u;
	uint16_t *gate_up = (uint16_t *)malloc(elements * 2u);
	uint16_t *gate_up_keep = (uint16_t *)malloc(elements * 2u);
	uint16_t *expected = (uint16_t *)malloc((uint64_t)rows * intermediate * 2u);
	uint16_t *actual = (uint16_t *)malloc((uint64_t)rows * intermediate * 2u);
	uint32_t row,element;
	void *gate_up_device = 0;
	cudaError_t error;
	if (gate_up == 0 || gate_up_keep == 0 || expected == 0 || actual == 0)
		return(SparkGemma4ValFail("gated_gelu","host_alloc"));
	SparkGemma4ValRandomState = 701u;
	SparkGemma4ValFillBf16(gate_up,elements,1.0f);
	memcpy(gate_up_keep,gate_up,elements * 2u);
	for (row = 0u; row < rows; row++)
		for (element = 0u; element < intermediate; element++)
		{
			uint64_t base = ((uint64_t)row * intermediate * 2u) + element;
			expected[((uint64_t)row * intermediate) + element] = SparkGemma4ValBf16(
				SparkGemma4ValGelu(SparkGemma4ValFromBf16(gate_up_keep[base])) * SparkGemma4ValFromBf16(gate_up_keep[base + intermediate]));
		}
	error = cudaMalloc(&gate_up_device,elements * 2u);
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(gate_up_device,gate_up,elements * 2u);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchGatedGelu(cudaStreamPerThread,gate_up_device,rows,intermediate);
	if (error == cudaSuccess) error = SparkGemma4ValSync();
	if (error == cudaSuccess) error = SparkGemma4ValCopyDown(gate_up,gate_up_device,elements * 2u);
	if (SparkGemma4ValCuda(error,"gated_gelu") != 0)
		return(1);
	for (row = 0u; row < rows; row++)
		for (element = 0u; element < intermediate; element++)
			actual[((uint64_t)row * intermediate) + element] =
				gate_up[((uint64_t)row * intermediate * 2u) + element];
	if (SparkGemma4ValCompareBf16("gated_gelu_tanh",actual,expected,(uint64_t)rows * intermediate) != 0)
		return(1);
	cudaFree(gate_up_device);
	free(gate_up);
	free(gate_up_keep);
	free(expected);
	free(actual);
	return(0);
}

static int SparkGemma4ValCheckRope(void)
{
	const uint32_t rows = 3u;
	const uint32_t heads = SPARK_GEMMA4_MODEL_SLIDING_QUERY_HEAD_COUNT;
	const uint32_t dimension = SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION;
	const uint32_t full_dimension = SPARK_GEMMA4_MODEL_FULL_HEAD_DIMENSION;
	const uint32_t table_elements = SPARK_GEMMA4_MODEL_FULL_ROPE_TABLE_ELEMENTS;
	const uint32_t rotated_pairs = 64u;
	const uint32_t positions[3] = {0u,5u,33u};
	uint16_t *sliding = (uint16_t *)malloc((uint64_t)rows * heads * dimension * 2u);
	uint16_t *sliding_keep = (uint16_t *)malloc((uint64_t)rows * heads * dimension * 2u);
	uint16_t *full = (uint16_t *)malloc((uint64_t)rows * heads * full_dimension * 2u);
	uint16_t *full_keep = (uint16_t *)malloc((uint64_t)rows * heads * full_dimension * 2u);
	float table[SPARK_GEMMA4_MODEL_FULL_ROPE_TABLE_ELEMENTS];
	uint32_t positions_host[3];
	uint32_t row,head,pair,element;
	void *sliding_device = 0,*full_device = 0,*positions_device = 0,*table_device = 0;
	cudaError_t error;
	if (sliding == 0 || sliding_keep == 0 || full == 0 || full_keep == 0)
		return(SparkGemma4ValFail("rope","host_alloc"));
	SparkGemma4ValRandomState = 809u;
	SparkGemma4ValFillBf16(sliding,(uint64_t)rows * heads * dimension,0.5f);
	SparkGemma4ValFillBf16(full,(uint64_t)rows * heads * full_dimension,0.5f);
	memcpy(sliding_keep,sliding,(uint64_t)rows * heads * dimension * 2u);
	memcpy(full_keep,full,(uint64_t)rows * heads * full_dimension * 2u);
	for (element = 0u; element < table_elements; element++)
		table[element] = element < rotated_pairs
			? 1.0f / powf(SPARK_GEMMA4_MODEL_FULL_ROPE_BASE,(float)(2u * element) / (float)full_dimension) : 0.0f;
	for (row = 0u; row < rows; row++)
		positions_host[row] = positions[row];
	error = cudaMalloc(&sliding_device,(uint64_t)rows * heads * dimension * 2u);
	if (error == cudaSuccess) error = cudaMalloc(&full_device,(uint64_t)rows * heads * full_dimension * 2u);
	if (error == cudaSuccess) error = cudaMalloc(&positions_device,sizeof(positions_host));
	if (error == cudaSuccess) error = cudaMalloc(&table_device,sizeof(table));
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(sliding_device,sliding,(uint64_t)rows * heads * dimension * 2u);
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(full_device,full,(uint64_t)rows * heads * full_dimension * 2u);
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(positions_device,positions_host,sizeof(positions_host));
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(table_device,table,sizeof(table));
	if (error == cudaSuccess)
		error = SparkGemma4LaunchSlidingRope(cudaStreamPerThread,sliding_device,(const uint32_t *)positions_device,rows,heads,dimension,dimension,SPARK_GEMMA4_MODEL_SLIDING_ROPE_THETA);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchFullRope(cudaStreamPerThread,full_device,(const uint32_t *)positions_device,(const float *)table_device,rows,heads,full_dimension,full_dimension);
	if (error == cudaSuccess) error = SparkGemma4ValSync();
	if (error == cudaSuccess) error = SparkGemma4ValCopyDown(sliding,sliding_device,(uint64_t)rows * heads * dimension * 2u);
	if (error == cudaSuccess) error = SparkGemma4ValCopyDown(full,full_device,(uint64_t)rows * heads * full_dimension * 2u);
	if (SparkGemma4ValCuda(error,"rope") != 0)
		return(1);
	{
		float *sliding_expected = (float *)malloc((uint64_t)rows * heads * dimension * sizeof(float));
		float *full_expected = (float *)malloc((uint64_t)rows * heads * full_dimension * sizeof(float));
		if (sliding_expected == 0 || full_expected == 0)
			return(SparkGemma4ValFail("rope","mirror_alloc"));
		for (row = 0u; row < rows; row++)
			for (head = 0u; head < heads; head++)
			{
				uint64_t base = (((uint64_t)row * heads) + head) * dimension;
				for (pair = 0u; pair < dimension / 2u; pair++)
				{
					float angle = (float)positions[row] * powf(SPARK_GEMMA4_MODEL_SLIDING_ROPE_THETA,-(float)(2u * pair) / (float)dimension);
					float cosine = cosf(angle);
					float sine = sinf(angle);
					float low = SparkGemma4ValFromBf16(sliding_keep[base + pair]);
					float high = SparkGemma4ValFromBf16(sliding_keep[base + (dimension / 2u) + pair]);
					sliding_expected[base + pair] = SparkGemma4ValFromBf16(SparkGemma4ValBf16((low * cosine) - (high * sine)));
					sliding_expected[base + (dimension / 2u) + pair] = SparkGemma4ValFromBf16(SparkGemma4ValBf16((high * cosine) + (low * sine)));
				}
			}
		if (SparkGemma4ValCompareAgainstFloat("rope_sliding_theta_1e4",sliding,sliding_expected,(uint64_t)rows * heads * dimension) != 0)
			return(1);
		free(sliding_expected);
		for (row = 0u; row < rows; row++)
			for (head = 0u; head < heads; head++)
			{
				uint64_t base = (((uint64_t)row * heads) + head) * full_dimension;
				for (pair = 0u; pair < full_dimension / 2u; pair++)
				{
					float angle = (float)positions[row] * table[pair];
					float cosine = cosf(angle);
					float sine = sinf(angle);
					float low = SparkGemma4ValFromBf16(full_keep[base + pair]);
					float high = SparkGemma4ValFromBf16(full_keep[base + (full_dimension / 2u) + pair]);
					full_expected[base + pair] = SparkGemma4ValFromBf16(SparkGemma4ValBf16((low * cosine) - (high * sine)));
					full_expected[base + (full_dimension / 2u) + pair] = SparkGemma4ValFromBf16(SparkGemma4ValBf16((high * cosine) + (low * sine)));
				}
			}
		if (SparkGemma4ValCompareAgainstFloat("rope_full_inv_freq_table",full,full_expected,(uint64_t)rows * heads * full_dimension) != 0)
			return(1);
		free(full_expected);
	}
	for (row = 0u; row < rows; row++)
		for (head = 0u; head < heads; head++)
		{
			uint64_t base = (((uint64_t)row * heads) + head) * full_dimension;
			for (pair = rotated_pairs; pair < full_dimension / 2u; pair++)
				if (full[base + pair] != full_keep[base + pair]
					|| full[base + (full_dimension / 2u) + pair] != full_keep[base + (full_dimension / 2u) + pair])
					return(SparkGemma4ValFail("rope_full_identity_region","bitwise"));
		}
	for (element = 0u; element < (uint64_t)heads * full_dimension; element++)
		if (full[element] != full_keep[element])
			return(SparkGemma4ValFail("rope_full_position_zero_identity","bitwise"));
	printf("gemma4_validation check=rope_full_identity_region rotated_pairs=%u table=%u bit_exact=1\n",rotated_pairs,table_elements);
	gemma4_val_sites++;
	cudaFree(sliding_device);
	cudaFree(full_device);
	cudaFree(positions_device);
	cudaFree(table_device);
	free(sliding);
	free(sliding_keep);
	free(full);
	free(full_keep);
	return(0);
}

static int SparkGemma4ValCheckWindow(void)
{
	const uint32_t cases = 4u;
	const uint32_t contexts[4] = {5u,1024u,1030u,2048u};
	uint32_t expected[SPARK_GEMMA4_VAL_WINDOW];
	uint32_t actual[SPARK_GEMMA4_VAL_WINDOW];
	uint32_t sequence[1] = {0u};
	uint32_t row_position[1];
	uint32_t case_index,index,selected,start;
	void *sequence_device = 0,*context_device = 0,*row_position_device = 0,*window_device = 0;
	cudaError_t error;
	error = cudaMalloc(&sequence_device,sizeof(sequence));
	if (error == cudaSuccess) error = cudaMalloc(&context_device,sizeof(uint32_t));
	if (error == cudaSuccess) error = cudaMalloc(&row_position_device,sizeof(uint32_t));
	if (error == cudaSuccess) error = cudaMalloc(&window_device,sizeof(actual));
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(sequence_device,sequence,sizeof(sequence));
	if (SparkGemma4ValCuda(error,"window") != 0)
		return(1);
	for (case_index = 0u; case_index < cases; case_index++)
	{
		row_position[0] = contexts[case_index] - 1u;
		selected = contexts[case_index] < SPARK_GEMMA4_VAL_WINDOW ? contexts[case_index] : SPARK_GEMMA4_VAL_WINDOW;
		start = contexts[case_index] - selected;
		for (index = 0u; index < SPARK_GEMMA4_VAL_WINDOW; index++)
			expected[index] = index < selected ? start + index : 0xffffffffu;
		error = SparkGemma4ValCopyUp(context_device,&contexts[case_index],sizeof(uint32_t));
		if (error == cudaSuccess) error = SparkGemma4ValCopyUp(row_position_device,row_position,sizeof(uint32_t));
		if (error == cudaSuccess)
			error = SparkGemma4LaunchSlidingWindowPositions(cudaStreamPerThread,(const uint32_t *)sequence_device,(const uint32_t *)context_device,(const uint32_t *)row_position_device,1u,(uint32_t *)window_device);
		if (error == cudaSuccess) error = SparkGemma4ValSync();
		if (error == cudaSuccess) error = SparkGemma4ValCopyDown(actual,window_device,sizeof(actual));
		if (SparkGemma4ValCuda(error,"window") != 0)
			return(1);
		if (memcmp(actual,expected,sizeof(actual)) != 0)
			return(SparkGemma4ValFail("sliding_window_positions","bitwise"));
		printf("gemma4_validation check=sliding_window_positions context=%u start=%u selected=%u bit_exact=1\n",
			contexts[case_index],start,selected);
	}
	gemma4_val_sites++;
	cudaFree(sequence_device);
	cudaFree(context_device);
	cudaFree(row_position_device);
	cudaFree(window_device);
	return(0);
}

typedef struct SparkGemma4ValKv
{
	uint32_t kv_heads;
	uint32_t head_dimension;
	uint32_t tokens;
	uint32_t page_count;
	uint32_t slot_bytes;
	uint32_t page_bytes;
	void *pool;
	uint32_t *page_table;
	void *access_error;
	uint16_t *key_host;
	uint16_t *value_host;
	uint32_t sequence_host[1];
	uint32_t context_host[1];
	uint32_t *positions_host;
	uint32_t *sequence_rows_host;
	void *key_device;
	void *value_device;
	void *sequence_rows_device;
	void *sequence_device;
	void *context_device;
	void *positions_device;
} SparkGemma4ValKv;

static cudaError_t SparkGemma4ValKvSetup(SparkGemma4ValKv *kv, uint32_t kv_heads, uint32_t head_dimension, uint32_t tokens)
{
	uint32_t page_count = (tokens + SPARK_GEMMA4_VAL_PAGE_SLOTS - 1u) / SPARK_GEMMA4_VAL_PAGE_SLOTS;
	uint32_t *page_host;
	uint64_t pool_bytes;
	uint32_t index;
	cudaError_t error;
	kv->kv_heads = kv_heads;
	kv->head_dimension = head_dimension;
	kv->tokens = tokens;
	kv->page_count = (tokens + SPARK_GEMMA4_VAL_PAGE_SLOTS - 1u) / SPARK_GEMMA4_VAL_PAGE_SLOTS;
	kv->slot_bytes = kv_heads * (head_dimension + head_dimension) * 2u;
	kv->page_bytes = kv->slot_bytes * SPARK_GEMMA4_VAL_PAGE_SLOTS;
	kv->sequence_host[0] = 0u;
	kv->context_host[0] = tokens;
	pool_bytes = (uint64_t)page_count * kv->page_bytes;
	page_host = (uint32_t *)malloc((uint64_t)page_count * sizeof(uint32_t));
	kv->key_host = (uint16_t *)malloc((uint64_t)tokens * kv_heads * head_dimension * 2u);
	kv->value_host = (uint16_t *)malloc((uint64_t)tokens * kv_heads * head_dimension * 2u);
	kv->positions_host = (uint32_t *)malloc((uint64_t)tokens * sizeof(uint32_t));
	if (page_host == 0 || kv->key_host == 0 || kv->value_host == 0 || kv->positions_host == 0)
	{
		SparkGemma4ValFail("kv_setup","host_alloc");
		return(cudaErrorInvalidValue);
	}
	for (index = 0u; index < page_count; index++)
		page_host[index] = index;
	SparkGemma4ValRandomState = 1123u;
	SparkGemma4ValFillBf16(kv->key_host,(uint64_t)tokens * kv_heads * head_dimension,0.5f);
	SparkGemma4ValFillBf16(kv->value_host,(uint64_t)tokens * kv_heads * head_dimension,0.5f);
	for (index = 0u; index < tokens; index++)
	{
		kv->positions_host[index] = index;
		kv->sequence_rows_host[index] = 0u;
	}
	error = cudaMalloc(&kv->pool,pool_bytes);
	if (error == cudaSuccess) error = cudaMemset(kv->pool,0xAB,pool_bytes);
	if (error == cudaSuccess) error = cudaMalloc((void **)&kv->page_table,(uint64_t)page_count * sizeof(uint32_t));
	if (error == cudaSuccess) error = cudaMallocManaged(&kv->access_error,sizeof(LmFrameError),cudaMemAttachGlobal);
	if (error == cudaSuccess)
		memset(kv->access_error,0,sizeof(LmFrameError));
	if (error == cudaSuccess) error = cudaMalloc(&kv->key_device,(uint64_t)tokens * kv_heads * head_dimension * 2u);
	if (error == cudaSuccess) error = cudaMalloc(&kv->value_device,(uint64_t)tokens * kv_heads * head_dimension * 2u);
	if (error == cudaSuccess) error = cudaMalloc(&kv->sequence_rows_device,(uint64_t)tokens * sizeof(uint32_t));
	if (error == cudaSuccess) error = cudaMalloc(&kv->sequence_device,sizeof(uint32_t));
	if (error == cudaSuccess) error = cudaMalloc(&kv->context_device,sizeof(uint32_t));
	if (error == cudaSuccess) error = cudaMalloc(&kv->positions_device,(uint64_t)tokens * sizeof(uint32_t));
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(kv->page_table,page_host,(uint64_t)page_count * sizeof(uint32_t));
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(kv->key_device,kv->key_host,(uint64_t)tokens * kv_heads * head_dimension * 2u);
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(kv->value_device,kv->value_host,(uint64_t)tokens * kv_heads * head_dimension * 2u);
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(kv->sequence_rows_device,kv->sequence_rows_host,(uint64_t)tokens * sizeof(uint32_t));
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(kv->sequence_device,kv->sequence_host,sizeof(kv->sequence_host));
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(kv->context_device,kv->context_host,sizeof(kv->context_host));
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(kv->positions_device,kv->positions_host,(uint64_t)tokens * sizeof(uint32_t));
	free(page_host);
	return(error);
}

static void SparkGemma4ValKvTeardown(SparkGemma4ValKv *kv)
{
	cudaFree(kv->pool);
	cudaFree(kv->page_table);
	cudaFree(kv->access_error);
	cudaFree(kv->key_device);
	cudaFree(kv->value_device);
	cudaFree(kv->sequence_device);
	cudaFree(kv->context_device);
	cudaFree(kv->positions_device);
	cudaFree(kv->sequence_rows_device);
	free(kv->key_host);
	free(kv->value_host);
	free(kv->positions_host);
	free(kv->sequence_rows_host);
}

static int SparkGemma4ValKvCheckStored(SparkGemma4ValKv *kv, uint32_t position)
{
	uint16_t *stored = (uint16_t *)malloc(kv->slot_bytes);
	uint32_t page = position / SPARK_GEMMA4_VAL_PAGE_SLOTS;
	uint32_t slot = position % SPARK_GEMMA4_VAL_PAGE_SLOTS;
	uint64_t offset = ((uint64_t)page * kv->page_bytes) + ((uint64_t)slot * kv->slot_bytes);
	uint32_t index;
	uint32_t stored_count = kv->kv_heads * kv->head_dimension;
	cudaError_t error;
	if (stored == 0)
		return(SparkGemma4ValFail("kv_store_bitwise","host_alloc"));
	error = SparkGemma4ValCopyDown(stored,((uint8_t *)kv->pool) + offset,kv->slot_bytes);
	if (SparkGemma4ValCuda(error,"kv_store_bitwise") != 0)
		return(1);
	for (index = 0u; index < stored_count; index++)
		if (stored[index] != kv->key_host[((uint64_t)position * stored_count) + index])
			return(SparkGemma4ValFail("kv_store_bitwise","key"));
	for (index = 0u; index < stored_count; index++)
		if (stored[stored_count + index] != kv->value_host[((uint64_t)position * stored_count) + index])
			return(SparkGemma4ValFail("kv_store_bitwise","value"));
	free(stored);
	return(0);
}

static int SparkGemma4ValKvCheckErrorClear(SparkGemma4ValKv *kv, const char *check)
{
	LmFrameError host_error;
	if (SparkGemma4ValCopyDown(&host_error,kv->access_error,sizeof(host_error)) != cudaSuccess)
		return(1);
	if (host_error.error_code != 0u)
	{
		fprintf(stderr,"gemma4_validation failure=%s access_error code=%u row=%u seq=%u pos=%u page=%u\n",
			check,host_error.error_code,host_error.row,host_error.sequence,host_error.position,host_error.page);
		return(SparkGemma4ValFail(check,"access_error_raised"));
	}
	return(0);
}

static void SparkGemma4ValMirrorDecode(const SparkGemma4ValKv *kv, const uint16_t *query, const uint32_t *window, uint32_t window_count, uint32_t query_heads, uint16_t *output)
{
	uint32_t row,head,step,element;
	uint32_t group = query_heads / kv->kv_heads;
	for (row = 0u; row < SPARK_GEMMA4_VAL_ROWS; row++)
		for (head = 0u; head < query_heads; head++)
		{
			uint32_t kv_head = head / group;
			float scores[SPARK_GEMMA4_VAL_WINDOW];
			float maximum = -INFINITY;
			float total = 0.0f;
			uint64_t query_base = (((uint64_t)row * query_heads) + head) * kv->head_dimension;
			for (step = 0u; step < window_count; step++)
			{
				uint32_t position = window[step];
				uint64_t slot_base = (uint64_t)position * kv->kv_heads * kv->head_dimension;
				float score = 0.0f;
				if (position == 0xffffffffu)
					continue;
				for (element = 0u; element < kv->head_dimension; element++)
					score += SparkGemma4ValFromBf16(query[query_base + element])
						* SparkGemma4ValFromBf16(kv->key_host[slot_base + ((uint64_t)kv_head * kv->head_dimension) + element]);
				scores[step] = score * SPARK_GEMMA4_MODEL_QK_SCALE;
				if (scores[step] > maximum)
					maximum = scores[step];
			}
			for (step = 0u; step < window_count; step++)
			{
				if (window[step] == 0xffffffffu)
				{
					scores[step] = 0.0f;
					continue;
				}
				scores[step] = expf(scores[step] - maximum);
				total += scores[step];
			}
			for (element = 0u; element < kv->head_dimension; element++)
			{
				float accumulator = 0.0f;
				for (step = 0u; step < window_count; step++)
				{
					uint32_t position = window[step];
					uint64_t slot_base;
					if (position == 0xffffffffu)
						continue;
					slot_base = (uint64_t)position * kv->kv_heads * kv->head_dimension;
					accumulator += (scores[step] / total)
						* SparkGemma4ValFromBf16(kv->value_host[slot_base + ((uint64_t)kv_head * kv->head_dimension) + element]);
				}
				output[query_base + element] = SparkGemma4ValBf16(accumulator);
			}
		}
}

static int SparkGemma4ValCheckKvSliding(void)
{
	const uint32_t query_heads = SPARK_GEMMA4_MODEL_SLIDING_QUERY_HEAD_COUNT;
	const uint32_t context = SPARK_GEMMA4_VAL_SLIDING_CONTEXT;
	const uint64_t count = (uint64_t)SPARK_GEMMA4_VAL_ROWS * query_heads * SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION;
	uint16_t *query = (uint16_t *)malloc(count * 2u);
	uint16_t *expected = (uint16_t *)malloc(count * 2u);
	uint16_t *actual = (uint16_t *)malloc(count * 2u);
	uint16_t *actual_rerun = (uint16_t *)malloc(count * 2u);
	uint32_t window[SPARK_GEMMA4_VAL_WINDOW];
	uint32_t row_position[1] = {context - 1u};
	uint32_t window_count = context < SPARK_GEMMA4_VAL_WINDOW ? context : SPARK_GEMMA4_VAL_WINDOW;
	void *query_device = 0,*output_device = 0,*row_position_device = 0,*window_device = 0;
	SparkGemma4ValKv kv;
	cudaError_t error;
	SparkGemma4ValMetrics metrics;
	float *actual_f;
	float *expected_f;
	uint64_t element;
	if (query == 0 || expected == 0 || actual == 0 || actual_rerun == 0)
		return(SparkGemma4ValFail("kv_sliding","host_alloc"));
	SparkGemma4ValRandomState = 3389u;
	SparkGemma4ValFillBf16(query,count,1.0f);
	error = SparkGemma4ValKvSetup(&kv,1u,SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION,context);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchKvStoreSliding(cudaStreamPerThread,kv.pool,kv.page_table,kv.page_count,1u,kv.page_count,kv.access_error,kv.key_device,kv.value_device,(const uint32_t *)kv.sequence_device,(const uint32_t *)kv.positions_device,context,1u);
	if (error == cudaSuccess) error = SparkGemma4ValSync();
	if (error == cudaSuccess && SparkGemma4ValKvCheckErrorClear(&kv,"kv_sliding") != 0)
		return(1);
	if (error == cudaSuccess && SparkGemma4ValKvCheckStored(&kv,0u) != 0)
		return(1);
	if (error == cudaSuccess && SparkGemma4ValKvCheckStored(&kv,63u) != 0)
		return(1);
	if (error == cudaSuccess && SparkGemma4ValKvCheckStored(&kv,1029u) != 0)
		return(1);
	if (error == cudaSuccess && SparkGemma4ValKvCheckErrorClear(&kv,"kv_sliding") != 0)
		return(1);
	if (SparkGemma4ValCuda(error,"kv_sliding") != 0)
		return(1);
	error = cudaMalloc(&query_device,count * 2u);
	if (error == cudaSuccess) error = cudaMalloc(&output_device,count * 2u);
	if (error == cudaSuccess) error = cudaMalloc(&row_position_device,sizeof(uint32_t));
	if (error == cudaSuccess) error = cudaMalloc(&window_device,sizeof(window));
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(query_device,query,count * 2u);
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(row_position_device,row_position,sizeof(uint32_t));
	if (error == cudaSuccess)
		error = SparkGemma4LaunchSlidingWindowPositions(cudaStreamPerThread,(const uint32_t *)kv.sequence_device,(const uint32_t *)kv.context_device,(const uint32_t *)row_position_device,1u,(uint32_t *)window_device);
	if (error == cudaSuccess) error = SparkGemma4ValCopyDown(window,window_device,sizeof(window));
	if (error == cudaSuccess)
		error = SparkGemma4LaunchAttentionDecodeSliding(cudaStreamPerThread,kv.pool,kv.page_table,kv.page_count,1u,kv.page_count,kv.access_error,query_device,(const uint32_t *)kv.sequence_device,(const uint32_t *)kv.context_device,(const uint32_t *)window_device,query_heads,output_device,SPARK_GEMMA4_VAL_ROWS,1u);
	if (error == cudaSuccess) error = SparkGemma4ValSync();
	if (error == cudaSuccess) error = SparkGemma4ValCopyDown(actual,output_device,count * 2u);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchAttentionDecodeSliding(cudaStreamPerThread,kv.pool,kv.page_table,kv.page_count,1u,kv.page_count,kv.access_error,query_device,(const uint32_t *)kv.sequence_device,(const uint32_t *)kv.context_device,(const uint32_t *)window_device,query_heads,output_device,SPARK_GEMMA4_VAL_ROWS,1u);
	if (error == cudaSuccess) error = SparkGemma4ValSync();
	if (error == cudaSuccess) error = SparkGemma4ValCopyDown(actual_rerun,output_device,count * 2u);
	if (SparkGemma4ValCuda(error,"kv_sliding") != 0)
		return(1);
	SparkGemma4ValMirrorDecode(&kv,query,window,window_count,query_heads,expected);
	if (memcmp(actual,actual_rerun,count * 2u) != 0)
		return(SparkGemma4ValFail("kv_sliding_determinism","rerun_bit_exact"));
	actual_f = (float *)malloc(count * sizeof(float));
	expected_f = (float *)malloc(count * sizeof(float));
	if (actual_f == 0 || expected_f == 0)
		return(SparkGemma4ValFail("kv_sliding","mirror_alloc"));
	for (element = 0u; element < count; element++)
	{
		actual_f[element] = SparkGemma4ValFromBf16(actual[element]);
		expected_f[element] = SparkGemma4ValFromBf16(expected[element]);
	}
	SparkGemma4ValMeasure(&metrics,actual_f,expected_f,count);
	if (SparkGemma4ValReport("kv_sliding_store_decode_window1024",&metrics,5e-3,0.999) != 0)
		return(1);
	printf("gemma4_validation check=kv_sliding_determinism bit_exact=1 store_bitwise=1 window=%u\n",window_count);
	gemma4_val_sites++;
	free(actual_f);
	free(expected_f);
	SparkGemma4ValKvTeardown(&kv);
	error = SparkGemma4ValKvSetup(&kv,2u,SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION,context);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchKvStoreSliding(cudaStreamPerThread,kv.pool,kv.page_table,kv.page_count,1u,kv.page_count,kv.access_error,kv.key_device,kv.value_device,(const uint32_t *)kv.sequence_device,(const uint32_t *)kv.positions_device,context,2u);
	if (error == cudaSuccess) error = SparkGemma4ValSync();
	if (error == cudaSuccess && SparkGemma4ValKvCheckStored(&kv,1029u) != 0)
		return(1);
	if (error == cudaSuccess && SparkGemma4ValKvCheckErrorClear(&kv,"kv_sliding_g2") != 0)
		return(1);
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(query_device,query,count * 2u);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchAttentionDecodeSliding(cudaStreamPerThread,kv.pool,kv.page_table,kv.page_count,1u,kv.page_count,kv.access_error,query_device,(const uint32_t *)kv.sequence_device,(const uint32_t *)kv.context_device,(const uint32_t *)window_device,query_heads,output_device,SPARK_GEMMA4_VAL_ROWS,2u);
	if (error == cudaSuccess) error = SparkGemma4ValSync();
	if (error == cudaSuccess) error = SparkGemma4ValCopyDown(actual,output_device,count * 2u);
	if (SparkGemma4ValCuda(error,"kv_sliding_g2") != 0)
		return(1);
	SparkGemma4ValMirrorDecode(&kv,query,window,window_count,query_heads,expected);
	actual_f = (float *)malloc(count * sizeof(float));
	expected_f = (float *)malloc(count * sizeof(float));
	if (actual_f == 0 || expected_f == 0)
		return(SparkGemma4ValFail("kv_sliding_g2","mirror_alloc"));
	for (element = 0u; element < count; element++)
	{
		actual_f[element] = SparkGemma4ValFromBf16(actual[element]);
		expected_f[element] = SparkGemma4ValFromBf16(expected[element]);
	}
	SparkGemma4ValMeasure(&metrics,actual_f,expected_f,count);
	if (SparkGemma4ValReport("kv_sliding_two_heads_geometry2",&metrics,5e-3,0.999) != 0)
		return(1);
	free(actual_f);
	free(expected_f);
	SparkGemma4ValKvTeardown(&kv);
	cudaFree(query_device);
	cudaFree(output_device);
	cudaFree(row_position_device);
	cudaFree(window_device);
	free(query);
	free(expected);
	free(actual);
	free(actual_rerun);
	return(0);
}

static int SparkGemma4ValCheckKvFull(void)
{
	const uint32_t query_heads = SPARK_GEMMA4_MODEL_FULL_QUERY_HEAD_COUNT;
	const uint32_t dimension = SPARK_GEMMA4_MODEL_FULL_HEAD_DIMENSION;
	const uint32_t kv_heads = 1u;
	const uint32_t context = SPARK_GEMMA4_VAL_FULL_CONTEXT;
	const uint64_t query_count = (uint64_t)SPARK_GEMMA4_VAL_ROWS * query_heads * dimension;
	const uint64_t row_count = (uint64_t)SPARK_GEMMA4_VAL_ROWS * kv_heads * dimension;
	uint16_t *kraw = (uint16_t *)malloc(row_count * 2u);
	uint16_t *v_norm = (uint16_t *)malloc(row_count * 2u);
	uint16_t *k_rope = (uint16_t *)malloc(row_count * 2u);
	uint16_t *query = (uint16_t *)malloc(query_count * 2u);
	uint16_t *expected = (uint16_t *)malloc(query_count * 2u);
	uint16_t *actual = (uint16_t *)malloc(query_count * 2u);
	uint32_t positions[SPARK_GEMMA4_VAL_ROWS];
	uint32_t row;
	uint64_t changed = 0u,element;
	void *kraw_device = 0,*v_device = 0,*k_device = 0,*query_device = 0,*output_device = 0,*positions_device = 0,*table_device = 0;
	float table[SPARK_GEMMA4_MODEL_FULL_ROPE_TABLE_ELEMENTS];
	uint32_t table_element;
	SparkGemma4ValKv kv;
	cudaError_t error;
	SparkGemma4ValMetrics metrics;
	float *actual_f;
	float *expected_f;
	if (kraw == 0 || v_norm == 0 || k_rope == 0 || query == 0 || expected == 0 || actual == 0)
		return(SparkGemma4ValFail("kv_full","host_alloc"));
	for (table_element = 0u; table_element < SPARK_GEMMA4_MODEL_FULL_ROPE_TABLE_ELEMENTS; table_element++)
		table[table_element] = table_element < 64u
			? 1.0f / powf(SPARK_GEMMA4_MODEL_FULL_ROPE_BASE,(float)(2u * table_element) / (float)dimension) : 0.0f;
	SparkGemma4ValRandomState = 4507u;
	SparkGemma4ValFillBf16(kraw,row_count,0.5f);
	SparkGemma4ValFillBf16(query,query_count,0.5f);
	for (row = 0u; row < SPARK_GEMMA4_VAL_ROWS; row++)
		positions[row] = context - SPARK_GEMMA4_VAL_ROWS + row;
	error = SparkGemma4ValKvSetup(&kv,1u,dimension,context);
	if (error == cudaSuccess) error = cudaMalloc(&kraw_device,row_count * 2u);
	if (error == cudaSuccess) error = cudaMalloc(&v_device,row_count * 2u);
	if (error == cudaSuccess) error = cudaMalloc(&k_device,row_count * 2u);
	if (error == cudaSuccess) error = cudaMalloc(&query_device,query_count * 2u);
	if (error == cudaSuccess) error = cudaMalloc(&output_device,query_count * 2u);
	if (error == cudaSuccess) error = cudaMalloc(&positions_device,sizeof(positions));
	if (error == cudaSuccess) error = cudaMalloc(&table_device,sizeof(table));
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(kraw_device,kraw,row_count * 2u);
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(query_device,query,query_count * 2u);
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(positions_device,positions,sizeof(positions));
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(table_device,table,sizeof(table));
	if (error == cudaSuccess)
		error = SparkGemma4LaunchKvStoreFull(cudaStreamPerThread,kv.pool,kv.page_table,kv.page_count,1u,kv.page_count,kv.access_error,kv.key_device,kv.value_device,(const uint32_t *)kv.sequence_device,(const uint32_t *)kv.positions_device,context);
	if (error == cudaSuccess) error = SparkGemma4ValSync();
	if (SparkGemma4ValCuda(error,"kv_full") != 0)
		return(1);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchHeadRmsNorm(cudaStreamPerThread,kraw_device,0,v_device,SPARK_GEMMA4_VAL_ROWS,kv_heads,dimension,SPARK_GEMMA4_MODEL_RMS_NORM_EPSILON);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchHeadRmsNorm(cudaStreamPerThread,kraw_device,kraw_device,k_device,SPARK_GEMMA4_VAL_ROWS,kv_heads,dimension,SPARK_GEMMA4_MODEL_RMS_NORM_EPSILON);
	if (error == cudaSuccess) error = SparkGemma4ValSync();
	if (error == cudaSuccess)
		error = SparkGemma4LaunchFullRope(cudaStreamPerThread,k_device,(const uint32_t *)positions_device,(const float *)table_device,SPARK_GEMMA4_VAL_ROWS,kv_heads,dimension,dimension);
	if (error == cudaSuccess) error = SparkGemma4ValSync();
	if (error == cudaSuccess) error = SparkGemma4ValCopyDown(k_rope,k_device,row_count * 2u);
	if (error == cudaSuccess) error = SparkGemma4ValCopyDown(v_norm,v_device,row_count * 2u);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchKvStoreFull(cudaStreamPerThread,kv.pool,kv.page_table,kv.page_count,1u,kv.page_count,kv.access_error,k_device,v_device,(const uint32_t *)kv.sequence_device,(const uint32_t *)positions_device,SPARK_GEMMA4_VAL_ROWS);
	if (error == cudaSuccess) error = SparkGemma4ValSync();
	if (error == cudaSuccess)
		error = SparkGemma4LaunchAttentionDecodeFull(cudaStreamPerThread,kv.pool,kv.page_table,kv.page_count,1u,kv.page_count,kv.access_error,query_device,(const uint32_t *)kv.sequence_device,(const uint32_t *)kv.context_device,query_heads,output_device,SPARK_GEMMA4_VAL_ROWS);
	if (error == cudaSuccess) error = SparkGemma4ValSync();
	if (error == cudaSuccess) error = SparkGemma4ValCopyDown(actual,output_device,query_count * 2u);
	if (SparkGemma4ValCuda(error,"kv_full") != 0)
		return(1);
	for (row = 0u; row < SPARK_GEMMA4_VAL_ROWS; row++)
	{
		uint64_t target = (uint64_t)positions[row] * kv_heads * dimension;
		uint64_t source = (uint64_t)row * kv_heads * dimension;
		memcpy(kv.key_host + target,k_rope + source,(uint64_t)kv_heads * dimension * 2u);
		memcpy(kv.value_host + target,v_norm + source,(uint64_t)kv_heads * dimension * 2u);
	}
	SparkGemma4ValMirrorDecode(&kv,query,kv.positions_host,context,query_heads,expected);
	for (element = 0u; element < row_count; element++)
		if (k_rope[element] != kraw[element])
			changed++;
	if (changed == 0u)
		return(SparkGemma4ValFail("kv_full_rope_applied","rotation_coverage"));
	actual_f = (float *)malloc(query_count * sizeof(float));
	expected_f = (float *)malloc(query_count * sizeof(float));
	if (actual_f == 0 || expected_f == 0)
		return(SparkGemma4ValFail("kv_full","mirror_alloc"));
	for (element = 0u; element < query_count; element++)
	{
		actual_f[element] = SparkGemma4ValFromBf16(actual[element]);
		expected_f[element] = SparkGemma4ValFromBf16(expected[element]);
	}
	SparkGemma4ValMeasure(&metrics,actual_f,expected_f,query_count);
	if (SparkGemma4ValReport("kv_full_keqv_store_decode",&metrics,5e-3,0.999) != 0)
		return(1);
	if (SparkGemma4ValKvCheckErrorClear(&kv,"kv_full") != 0)
		return(1);
	printf("gemma4_validation check=kv_full_keqv query_heads=%u kv_heads_rank=%u context=%u PASS\n",query_heads,kv_heads,context);
	gemma4_val_sites++;
	free(actual_f);
	free(expected_f);
	cudaFree(kraw_device);
	cudaFree(v_device);
	cudaFree(k_device);
	cudaFree(query_device);
	cudaFree(output_device);
	cudaFree(positions_device);
	cudaFree(table_device);
	SparkGemma4ValKvTeardown(&kv);
	free(kraw);
	free(v_norm);
	free(k_rope);
	free(query);
	free(expected);
	free(actual);
	return(0);
}

#if SPARK_GEMMA4_MODEL_MOE_BLOCK
static int SparkGemma4ValCheckRouter(void)
{
	const uint32_t rows = 4u;
	const uint32_t experts = SPARK_GEMMA4_MODEL_ROUTED_EXPERT_COUNT;
	const uint32_t top = SPARK_GEMMA4_MODEL_EXPERTS_PER_TOKEN;
	float logits[rows * experts];
	float probs[rows * experts];
	uint32_t indices[rows * top];
	float weights[rows * top];
	uint32_t row,k,candidate,e;
	void *scores_device = 0,*indices_device = 0,*weights_device = 0;
	cudaError_t error;
	SparkGemma4ValRandomState = 5801u;
	for (e = 0u; e < rows * experts; e++)
		logits[e] = SparkGemma4ValUniform(2.0f);
	for (e = 0u; e < experts; e++)
	{
		logits[e] = 0.75f;
		logits[experts + e] = 0.0f;
	}
	error = cudaMalloc(&scores_device,sizeof(logits));
	if (error == cudaSuccess) error = cudaMalloc(&indices_device,sizeof(indices));
	if (error == cudaSuccess) error = cudaMalloc(&weights_device,sizeof(weights));
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(scores_device,logits,sizeof(logits));
	if (error == cudaSuccess)
		error = SparkGemma4LaunchRouterSoftmax(cudaStreamPerThread,(float *)scores_device,rows);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchRouterTopk(cudaStreamPerThread,(const float *)scores_device,(uint32_t *)indices_device,(float *)weights_device,rows);
	if (error == cudaSuccess) error = SparkGemma4ValSync();
	if (error == cudaSuccess) error = SparkGemma4ValCopyDown(probs,scores_device,sizeof(probs));
	if (error == cudaSuccess) error = SparkGemma4ValCopyDown(indices,indices_device,sizeof(indices));
	if (error == cudaSuccess) error = SparkGemma4ValCopyDown(weights,weights_device,sizeof(weights));
	if (SparkGemma4ValCuda(error,"router") != 0)
		return(1);
	for (row = 0u; row < rows; row++)
	{
		float total = 0.0f;
		float sum = 0.0f;
		uint32_t taken[SPARK_GEMMA4_MODEL_EXPERTS_PER_TOKEN];
		uint32_t taken_count = 0u;
		for (e = 0u; e < experts; e++)
			total += probs[(row * experts) + e];
		if (fabsf(total - 1.0f) > 1e-4f)
			return(SparkGemma4ValFail("router_softmax_sum","unit"));
		for (k = 0u; k < top; k++)
		{
			float best_value = -1.0f;
			int64_t best = -1;
			uint32_t used;
			for (candidate = 0u; candidate < experts; candidate++)
			{
				used = 0u;
				for (e = 0u; e < taken_count; e++)
					if (taken[e] == candidate)
						used = 1u;
				if (used != 0u)
					continue;
				if (probs[(row * experts) + candidate] > best_value)
				{
					best_value = probs[(row * experts) + candidate];
					best = (int64_t)candidate;
				}
			}
			taken[taken_count++] = (uint32_t)best;
			sum += probs[(row * experts) + best];
			if (indices[(row * top) + k] != (uint32_t)best)
				return(SparkGemma4ValFail("router_topk_lowest_index_ties","indices"));
		}
		for (k = 0u; k < top; k++)
		{
			float expected_weight = probs[(row * experts) + taken[k]] / sum;
			if (fabsf(weights[(row * top) + k] - expected_weight) > 1e-5f)
				return(SparkGemma4ValFail("router_topk_renormalised","weights"));
		}
		if (row == 1u)
		{
			for (e = 0u; e < experts; e++)
				if (fabsf(probs[(row * experts) + e] - (1.0f / (float)experts)) > 1e-6f)
					return(SparkGemma4ValFail("router_zero_residual_uniform","1/128"));
			for (k = 0u; k < top; k++)
				if (indices[k] != k)
					return(SparkGemma4ValFail("router_zero_top8_indices","0..7"));
		}
	}
	printf("gemma4_validation check=router_softmax_topk rows=%u experts=%u top=%u ties=lowest_index zero_uniform=1/%u PASS\n",
		rows,experts,top,experts);
	gemma4_val_sites++;
	cudaFree(scores_device);
	cudaFree(indices_device);
	cudaFree(weights_device);
	return(0);
}
#endif

typedef struct SparkGemma4ValChain
{
	uint32_t hidden;
	uint32_t query_out;
	uint32_t kv_out;
	uint32_t kv_half;
	uint32_t kv_heads;
	uint32_t head_dimension;
	uint32_t intermediate;
	uint16_t *h0;
	uint16_t *input_ln;
	uint16_t *query_weight;
	uint16_t *query_norm;
	uint16_t *kv_weight;
	uint16_t *key_norm;
	uint16_t *output_weight;
	uint16_t *post_attention;
	uint16_t *gate_up_weight;
	uint16_t *down_weight;
	uint16_t *post_feedforward;
	uint16_t *expected_hidden;
	uint16_t *expected_normed;
	uint16_t *actual_hidden;
	uint16_t *actual_normed;
	uint16_t *mirror_query;
	uint16_t *mirror_dec;
	uint16_t *expected_dec;
	uint32_t positions[SPARK_GEMMA4_VAL_CHAIN_ROWS];
	uint32_t row_position[1];
	uint32_t context[1];
	void *h_device;
	void *normed_device;
	void *query_device;
	void *kv_device;
	void *att_device;
	void *gu_device;
	void *mlp_device;
	void *gain_device;
	void *positions_device;
	void *query_norm_device;
	void *key_norm_device;
	void *window_device;
	void *kv_pool;
	uint32_t kv_page_count;
	uint32_t *kv_page_table;
	void *kv_access_error;
	void *kv_sequence_device;
	void *kv_context_device;
	void *kv_row_position_device;
} SparkGemma4ValChain;

static void SparkGemma4ValChainWeightsFill(SparkGemma4ValChain *chain)
{
	SparkGemma4ValRandomState = 6151u;
	SparkGemma4ValFillBf16(chain->h0,(uint64_t)SPARK_GEMMA4_VAL_CHAIN_ROWS * chain->hidden,0.25f);
	SparkGemma4ValFillBf16(chain->input_ln,chain->hidden,1.0f);
	SparkGemma4ValFillBf16(chain->query_weight,(uint64_t)chain->query_out * chain->hidden,0.05f);
	SparkGemma4ValFillBf16(chain->query_norm,chain->head_dimension,1.0f);
	SparkGemma4ValFillBf16(chain->kv_weight,(uint64_t)chain->kv_out * chain->hidden,0.05f);
	SparkGemma4ValFillBf16(chain->key_norm,chain->head_dimension,1.0f);
	SparkGemma4ValFillBf16(chain->output_weight,(uint64_t)chain->hidden * chain->query_out,0.05f);
	SparkGemma4ValFillBf16(chain->post_attention,chain->hidden,1.0f);
	SparkGemma4ValFillBf16(chain->gate_up_weight,(uint64_t)chain->intermediate * 2u * chain->hidden,0.05f);
	SparkGemma4ValFillBf16(chain->down_weight,(uint64_t)chain->hidden * chain->intermediate,0.05f);
	SparkGemma4ValFillBf16(chain->post_feedforward,chain->hidden,1.0f);
	chain->positions[0] = SPARK_GEMMA4_VAL_CHAIN_BASE;
	chain->positions[1] = SPARK_GEMMA4_VAL_CHAIN_BASE + 1u;
	chain->row_position[0] = SPARK_GEMMA4_VAL_CHAIN_BASE + 1u;
	chain->context[0] = SPARK_GEMMA4_VAL_CHAIN_BASE + SPARK_GEMMA4_VAL_CHAIN_ROWS;
}

static cudaError_t SparkGemma4ValChainAlloc(SparkGemma4ValChain *chain)
{
	const uint64_t hidden_bytes = (uint64_t)SPARK_GEMMA4_VAL_CHAIN_ROWS * chain->hidden * 2u;
	cudaError_t error;
	chain->h0 = (uint16_t *)malloc(hidden_bytes);
	chain->input_ln = (uint16_t *)malloc((uint64_t)chain->hidden * 2u);
	chain->query_weight = (uint16_t *)malloc((uint64_t)chain->query_out * chain->hidden * 2u);
	chain->query_norm = (uint16_t *)malloc((uint64_t)chain->head_dimension * 2u);
	chain->kv_weight = (uint16_t *)malloc((uint64_t)chain->kv_out * chain->hidden * 2u);
	chain->key_norm = (uint16_t *)malloc((uint64_t)chain->head_dimension * 2u);
	chain->output_weight = (uint16_t *)malloc((uint64_t)chain->hidden * chain->query_out * 2u);
	chain->post_attention = (uint16_t *)malloc((uint64_t)chain->hidden * 2u);
	chain->gate_up_weight = (uint16_t *)malloc((uint64_t)chain->intermediate * 2u * chain->hidden * 2u);
	chain->down_weight = (uint16_t *)malloc((uint64_t)chain->hidden * chain->intermediate * 2u);
	chain->post_feedforward = (uint16_t *)malloc((uint64_t)chain->hidden * 2u);
	chain->expected_hidden = (uint16_t *)malloc(hidden_bytes);
	chain->expected_normed = (uint16_t *)malloc(hidden_bytes);
	chain->actual_hidden = (uint16_t *)malloc(hidden_bytes);
	chain->actual_normed = (uint16_t *)malloc(hidden_bytes);
	chain->mirror_query = (uint16_t *)malloc((uint64_t)SPARK_GEMMA4_VAL_CHAIN_ROWS * chain->query_out * 2u);
	chain->mirror_dec = (uint16_t *)malloc((uint64_t)SPARK_GEMMA4_VAL_CHAIN_ROWS * chain->query_out * 2u);
	chain->expected_dec = (uint16_t *)malloc((uint64_t)SPARK_GEMMA4_VAL_CHAIN_ROWS * chain->query_out * 2u);
	if (chain->h0 == 0 || chain->input_ln == 0 || chain->query_weight == 0 || chain->query_norm == 0
		|| chain->kv_weight == 0 || chain->key_norm == 0 || chain->output_weight == 0 || chain->post_attention == 0
		|| chain->gate_up_weight == 0 || chain->down_weight == 0 || chain->post_feedforward == 0
		|| chain->expected_hidden == 0 || chain->expected_normed == 0 || chain->actual_hidden == 0
		|| chain->actual_normed == 0 || chain->mirror_query == 0 || chain->mirror_dec == 0 || chain->expected_dec == 0)
	{
		SparkGemma4ValFail("chain_sliding","host_alloc");
		return(cudaErrorInvalidValue);
	}
	error = cudaMalloc(&chain->h_device,hidden_bytes);
	if (error == cudaSuccess) error = cudaMalloc(&chain->normed_device,hidden_bytes);
	if (error == cudaSuccess) error = cudaMalloc(&chain->query_device,(uint64_t)SPARK_GEMMA4_VAL_CHAIN_ROWS * chain->query_out * 2u);
	if (error == cudaSuccess) error = cudaMalloc(&chain->kv_device,(uint64_t)SPARK_GEMMA4_VAL_CHAIN_ROWS * chain->kv_out * 2u);
	if (error == cudaSuccess) error = cudaMalloc(&chain->att_device,hidden_bytes);
	if (error == cudaSuccess) error = cudaMalloc(&chain->gu_device,(uint64_t)SPARK_GEMMA4_VAL_CHAIN_ROWS * chain->intermediate * 2u * 2u);
	if (error == cudaSuccess) error = cudaMalloc(&chain->mlp_device,hidden_bytes);
	if (error == cudaSuccess) error = cudaMalloc(&chain->gain_device,(uint64_t)chain->hidden * 2u);
	if (error == cudaSuccess) error = cudaMalloc(&chain->positions_device,sizeof(chain->positions));
	if (error == cudaSuccess) error = cudaMalloc(&chain->query_norm_device,(uint64_t)chain->head_dimension * 2u);
	if (error == cudaSuccess) error = cudaMalloc(&chain->key_norm_device,(uint64_t)chain->head_dimension * 2u);
	if (error != cudaSuccess)
		SparkGemma4ValCuda(error,"chain_sliding");
	return(error);
}

static cudaError_t SparkGemma4ValChainView(SparkGemma4LinearView *view, const void *payload, uint32_t input_dimension, uint32_t output_dimension)
{
	memset(view,0,sizeof(*view));
	view->abi_version = SPARK_GEMMA4_RESIDENT_DECODE_STAGE_LINEAR_VIEW_ABI_VERSION;
	view->weight_format = SPARK_GEMMA4_RESIDENT_DECODE_STAGE_WEIGHT_FORMAT_BF16;
	view->input_dimension = input_dimension;
	view->output_dimension = output_dimension;
	view->weight_payload = payload;
	return(cudaSuccess);
}

static cudaError_t SparkGemma4ValChainDeviceStages(SparkGemma4ValChain *chain)
{
	SparkGemma4LinearView view;
	cudaError_t error;
	error = SparkGemma4LaunchRmsNorm(cudaStreamPerThread,chain->h_device,chain->gain_device,chain->normed_device,SPARK_GEMMA4_VAL_CHAIN_ROWS,chain->hidden,SPARK_GEMMA4_MODEL_RMS_NORM_EPSILON);
	if (error == cudaSuccess)
		error = SparkGemma4ValChainView(&view,chain->query_weight,chain->hidden,chain->query_out);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchLinear(cudaStreamPerThread,&view,chain->normed_device,chain->query_device,SPARK_GEMMA4_VAL_CHAIN_ROWS);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchHeadRmsNorm(cudaStreamPerThread,chain->query_device,chain->query_norm_device,chain->query_device,SPARK_GEMMA4_VAL_CHAIN_ROWS,SPARK_GEMMA4_MODEL_SLIDING_QUERY_HEAD_COUNT,chain->head_dimension,SPARK_GEMMA4_MODEL_RMS_NORM_EPSILON);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchSlidingRope(cudaStreamPerThread,chain->query_device,(const uint32_t *)chain->positions_device,SPARK_GEMMA4_VAL_CHAIN_ROWS,SPARK_GEMMA4_MODEL_SLIDING_QUERY_HEAD_COUNT,chain->head_dimension,chain->head_dimension,SPARK_GEMMA4_MODEL_SLIDING_ROPE_THETA);
	if (error == cudaSuccess)
		error = SparkGemma4ValChainView(&view,chain->kv_weight,chain->hidden,chain->kv_out);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchLinear(cudaStreamPerThread,&view,chain->normed_device,chain->kv_device,SPARK_GEMMA4_VAL_CHAIN_ROWS);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchHeadRmsNorm(cudaStreamPerThread,chain->kv_device,chain->key_norm_device,chain->kv_device,SPARK_GEMMA4_VAL_CHAIN_ROWS,chain->kv_heads,chain->head_dimension,SPARK_GEMMA4_MODEL_RMS_NORM_EPSILON);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchHeadRmsNorm(cudaStreamPerThread,((uint16_t *)chain->kv_device) + chain->kv_half,0,((uint16_t *)chain->kv_device) + chain->kv_half,SPARK_GEMMA4_VAL_CHAIN_ROWS,chain->kv_heads,chain->head_dimension,SPARK_GEMMA4_MODEL_RMS_NORM_EPSILON);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchSlidingRope(cudaStreamPerThread,chain->kv_device,(const uint32_t *)chain->positions_device,SPARK_GEMMA4_VAL_CHAIN_ROWS,chain->kv_heads,chain->head_dimension,chain->head_dimension,SPARK_GEMMA4_MODEL_SLIDING_ROPE_THETA);
	return(error);
}

static cudaError_t SparkGemma4ValChainDeviceTail(SparkGemma4ValChain *chain)
{
	SparkGemma4LinearView view;
	void *value_half = ((uint16_t *)chain->kv_device) + chain->kv_half;
	cudaError_t error;
	error = SparkGemma4LaunchKvStoreSliding(cudaStreamPerThread,chain->kv_pool,chain->kv_page_table,chain->kv_page_count,1u,chain->kv_page_count,chain->kv_access_error,chain->kv_device,value_half,(const uint32_t *)chain->kv_sequence_device,(const uint32_t *)chain->positions_device,SPARK_GEMMA4_VAL_CHAIN_ROWS,chain->kv_heads);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchAttentionDecodeSliding(cudaStreamPerThread,chain->kv_pool,chain->kv_page_table,chain->kv_page_count,1u,chain->kv_page_count,chain->kv_access_error,chain->query_device,(const uint32_t *)chain->kv_sequence_device,(const uint32_t *)chain->kv_context_device,(const uint32_t *)chain->window_device,SPARK_GEMMA4_MODEL_SLIDING_QUERY_HEAD_COUNT,chain->query_device,SPARK_GEMMA4_VAL_CHAIN_ROWS,chain->kv_heads);
	if (error == cudaSuccess)
		error = SparkGemma4ValChainView(&view,chain->output_weight,chain->query_out,chain->hidden);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchLinear(cudaStreamPerThread,&view,chain->query_device,chain->att_device,SPARK_GEMMA4_VAL_CHAIN_ROWS);
	if (error == cudaSuccess)
		error = SparkGemma4ValCopyUp(chain->gain_device,chain->post_attention,(uint64_t)chain->hidden * 2u);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchFusedResidualRmsNorm(cudaStreamPerThread,chain->h_device,chain->att_device,chain->gain_device,chain->normed_device,SPARK_GEMMA4_VAL_CHAIN_ROWS,chain->hidden,SPARK_GEMMA4_MODEL_RMS_NORM_EPSILON);
	if (error == cudaSuccess)
		error = SparkGemma4ValChainView(&view,chain->gate_up_weight,chain->hidden,chain->intermediate * 2u);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchLinear(cudaStreamPerThread,&view,chain->normed_device,chain->gu_device,SPARK_GEMMA4_VAL_CHAIN_ROWS);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchGatedGelu(cudaStreamPerThread,chain->gu_device,SPARK_GEMMA4_VAL_CHAIN_ROWS,chain->intermediate);
	if (error == cudaSuccess)
		error = SparkGemma4ValChainView(&view,chain->down_weight,chain->intermediate,chain->hidden);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchLinear(cudaStreamPerThread,&view,chain->gu_device,chain->mlp_device,SPARK_GEMMA4_VAL_CHAIN_ROWS);
	if (error == cudaSuccess)
		error = SparkGemma4ValCopyUp(chain->gain_device,chain->post_feedforward,(uint64_t)chain->hidden * 2u);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchFusedResidualRmsNorm(cudaStreamPerThread,chain->h_device,chain->mlp_device,chain->gain_device,chain->normed_device,SPARK_GEMMA4_VAL_CHAIN_ROWS,chain->hidden,SPARK_GEMMA4_MODEL_RMS_NORM_EPSILON);
	return(error);
}

static void SparkGemma4ValChainMirrorRope(const SparkGemma4ValChain *chain, uint16_t *rows, uint32_t heads)
{
	uint32_t row,head,pair;
	for (row = 0u; row < SPARK_GEMMA4_VAL_CHAIN_ROWS; row++)
		for (head = 0u; head < heads; head++)
		{
			uint64_t base = (((uint64_t)row * heads) + head) * chain->head_dimension;
			for (pair = 0u; pair < chain->head_dimension / 2u; pair++)
			{
				float angle = (float)chain->positions[row] * powf(SPARK_GEMMA4_MODEL_SLIDING_ROPE_THETA,-(float)(2u * pair) / (float)chain->head_dimension);
				float cosine = cosf(angle);
				float sine = sinf(angle);
				float low = SparkGemma4ValFromBf16(rows[base + pair]);
				float high = SparkGemma4ValFromBf16(rows[base + (chain->head_dimension / 2u) + pair]);
				rows[base + pair] = SparkGemma4ValBf16((low * cosine) - (high * sine));
				rows[base + (chain->head_dimension / 2u) + pair] = SparkGemma4ValBf16((high * cosine) + (low * sine));
			}
		}
}

static void SparkGemma4ValChainMirror(SparkGemma4ValChain *chain, SparkGemma4ValKv *kv)
{
	const uint32_t rows = SPARK_GEMMA4_VAL_CHAIN_ROWS;
	uint16_t *normed = (uint16_t *)malloc((uint64_t)rows * chain->hidden * 2u);
	uint16_t *kv_raw = (uint16_t *)malloc((uint64_t)rows * chain->kv_out * 2u);
	uint16_t *k_half = (uint16_t *)malloc((uint64_t)rows * chain->kv_half * 2u);
	uint16_t *v_half = (uint16_t *)malloc((uint64_t)rows * chain->kv_half * 2u);
	uint16_t *attended = (uint16_t *)malloc((uint64_t)rows * chain->hidden * 2u);
	uint16_t *gate_up = (uint16_t *)malloc((uint64_t)rows * chain->intermediate * 2u * 2u);
	uint16_t *hidden = (uint16_t *)malloc((uint64_t)rows * chain->hidden * 2u);
	uint32_t window[SPARK_GEMMA4_VAL_WINDOW];
	uint32_t row,element,selected,start;
	memcpy(hidden,chain->h0,(uint64_t)rows * chain->hidden * 2u);
	SparkGemma4ValMirrorRms(hidden,chain->input_ln,normed,rows,chain->hidden);
	SparkGemma4ValMirrorLinear(chain->query_weight,normed,chain->mirror_query,rows,chain->hidden,chain->query_out);
	SparkGemma4ValMirrorHeadRms(chain->mirror_query,chain->query_norm,chain->mirror_query,rows,SPARK_GEMMA4_MODEL_SLIDING_QUERY_HEAD_COUNT,chain->head_dimension);
	SparkGemma4ValChainMirrorRope(chain,chain->mirror_query,SPARK_GEMMA4_MODEL_SLIDING_QUERY_HEAD_COUNT);
	SparkGemma4ValMirrorLinear(chain->kv_weight,normed,kv_raw,rows,chain->hidden,chain->kv_out);
	for (row = 0u; row < rows; row++)
	{
		memcpy(k_half + ((uint64_t)row * chain->kv_half),kv_raw + ((uint64_t)row * chain->kv_out),chain->kv_half * 2u);
		memcpy(v_half + ((uint64_t)row * chain->kv_half),kv_raw + ((uint64_t)row * chain->kv_out) + chain->kv_half,chain->kv_half * 2u);
	}
	SparkGemma4ValMirrorHeadRms(k_half,chain->key_norm,k_half,rows,chain->kv_heads,chain->head_dimension);
	SparkGemma4ValMirrorHeadRms(v_half,0,v_half,rows,chain->kv_heads,chain->head_dimension);
	SparkGemma4ValChainMirrorRope(chain,k_half,chain->kv_heads);
	for (row = 0u; row < rows; row++)
	{
		uint64_t target = (uint64_t)chain->positions[row] * chain->kv_half;
		memcpy(kv->key_host + target,k_half + ((uint64_t)row * chain->kv_half),chain->kv_half * 2u);
		memcpy(kv->value_host + target,v_half + ((uint64_t)row * chain->kv_half),chain->kv_half * 2u);
	}
	selected = chain->context[0] < SPARK_GEMMA4_VAL_WINDOW ? chain->context[0] : SPARK_GEMMA4_VAL_WINDOW;
	start = chain->context[0] - selected;
	for (element = 0u; element < SPARK_GEMMA4_VAL_WINDOW; element++)
		window[element] = element < selected ? start + element : 0xffffffffu;
	SparkGemma4ValMirrorDecode(kv,chain->mirror_query,window,SPARK_GEMMA4_VAL_WINDOW,SPARK_GEMMA4_MODEL_SLIDING_QUERY_HEAD_COUNT,chain->expected_dec);
	SparkGemma4ValMirrorLinear(chain->output_weight,chain->expected_dec,attended,rows,chain->query_out,chain->hidden);
	for (row = 0u; row < rows; row++)
		for (element = 0u; element < chain->hidden; element++)
		{
			uint64_t at = ((uint64_t)row * chain->hidden) + element;
			hidden[at] = SparkGemma4ValBf16(SparkGemma4ValFromBf16(hidden[at]) + SparkGemma4ValFromBf16(attended[at]));
		}
	SparkGemma4ValMirrorRms(hidden,chain->post_attention,normed,rows,chain->hidden);
	SparkGemma4ValMirrorLinear(chain->gate_up_weight,normed,gate_up,rows,chain->hidden,chain->intermediate * 2u);
	for (row = 0u; row < rows; row++)
		for (element = 0u; element < chain->intermediate; element++)
		{
			uint64_t base = ((uint64_t)row * chain->intermediate * 2u) + element;
			gate_up[base] = SparkGemma4ValBf16(
				SparkGemma4ValGelu(SparkGemma4ValFromBf16(gate_up[base])) * SparkGemma4ValFromBf16(gate_up[base + chain->intermediate]));
		}
	SparkGemma4ValMirrorLinear(chain->down_weight,gate_up,attended,rows,chain->intermediate,chain->hidden);
	for (row = 0u; row < rows; row++)
		for (element = 0u; element < chain->hidden; element++)
		{
			uint64_t at = ((uint64_t)row * chain->hidden) + element;
			hidden[at] = SparkGemma4ValBf16(SparkGemma4ValFromBf16(hidden[at]) + SparkGemma4ValFromBf16(attended[at]));
		}
	SparkGemma4ValMirrorRms(hidden,chain->post_feedforward,normed,rows,chain->hidden);
	memcpy(chain->expected_hidden,hidden,(uint64_t)rows * chain->hidden * 2u);
	memcpy(chain->expected_normed,normed,(uint64_t)rows * chain->hidden * 2u);
	free(normed);
	free(kv_raw);
	free(k_half);
	free(v_half);
	free(attended);
	free(gate_up);
	free(hidden);
}

static void SparkGemma4ValChainFree(SparkGemma4ValChain *chain)
{
	free(chain->h0);
	free(chain->input_ln);
	free(chain->query_weight);
	free(chain->query_norm);
	free(chain->kv_weight);
	free(chain->key_norm);
	free(chain->output_weight);
	free(chain->post_attention);
	free(chain->gate_up_weight);
	free(chain->down_weight);
	free(chain->post_feedforward);
	free(chain->expected_hidden);
	free(chain->expected_normed);
	free(chain->actual_hidden);
	free(chain->actual_normed);
	free(chain->mirror_query);
	free(chain->mirror_dec);
	free(chain->expected_dec);
}

static int SparkGemma4ValCheckChainSliding(void)
{
	const uint32_t hidden = SPARK_GEMMA4_MODEL_HIDDEN_DIMENSION;
	SparkGemma4ValChain chain;
	SparkGemma4ValKv kv;
	uint32_t row_position = SPARK_GEMMA4_VAL_CHAIN_BASE + SPARK_GEMMA4_VAL_CHAIN_ROWS - 1u;
	uint32_t sequence = 0u;
	cudaError_t error;
	SparkGemma4ValMetrics metrics;
	float *actual_f;
	float *expected_f;
	uint64_t element,count;
	memset(&chain,0,sizeof(chain));
	chain.hidden = hidden;
	chain.query_out = SPARK_GEMMA4_MODEL_SLIDING_QUERY_DIMENSION;
	chain.kv_out = SPARK_GEMMA4_MODEL_SLIDING_KV_DIMENSION;
	chain.kv_half = SPARK_GEMMA4_MODEL_SLIDING_KV_DIMENSION / 2u;
	chain.kv_heads = SPARK_GEMMA4_MODEL_SLIDING_KV_HEAD_COUNT / SPARK_GEMMA4_MODEL_SLIDING_KV_HEAD_COUNT;
	chain.head_dimension = SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION;
	chain.intermediate = SPARK_GEMMA4_MODEL_DENSE_INTERMEDIATE_DIMENSION;
	error = SparkGemma4ValKvSetup(&kv,1u,SPARK_GEMMA4_MODEL_SLIDING_HEAD_DIMENSION,SPARK_GEMMA4_VAL_POOL_TOKENS);
	if (error == cudaSuccess)
		error = SparkGemma4LaunchKvStoreSliding(cudaStreamPerThread,kv.pool,kv.page_table,kv.page_count,1u,kv.page_count,kv.access_error,kv.key_device,kv.value_device,(const uint32_t *)kv.sequence_rows_device,(const uint32_t *)kv.positions_device,SPARK_GEMMA4_VAL_POOL_TOKENS,1u);
	if (error == cudaSuccess) error = SparkGemma4ValChainAlloc(&chain);
	if (error == cudaSuccess) SparkGemma4ValChainWeightsFill(&chain);
	chain.kv_pool = kv.pool;
	chain.kv_page_count = kv.page_count;
	chain.kv_page_table = kv.page_table;
	chain.kv_access_error = kv.access_error;
	chain.kv_sequence_device = kv.sequence_rows_device;
	chain.kv_context_device = kv.context_device;
	error = cudaMalloc(&chain.window_device,SPARK_GEMMA4_VAL_WINDOW * sizeof(uint32_t));
	if (error == cudaSuccess) error = cudaMalloc(&chain.kv_row_position_device,sizeof(uint32_t));
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(kv.context_device,chain.context,sizeof(uint32_t));
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(chain.kv_row_position_device,&row_position,sizeof(uint32_t));
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(chain.kv_sequence_device,&sequence,sizeof(uint32_t));
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(chain.h_device,chain.h0,(uint64_t)SPARK_GEMMA4_VAL_CHAIN_ROWS * hidden * 2u);
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(chain.gain_device,chain.input_ln,(uint64_t)hidden * 2u);
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(chain.positions_device,chain.positions,sizeof(chain.positions));
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(chain.query_norm_device,chain.query_norm,(uint64_t)chain.head_dimension * 2u);
	if (error == cudaSuccess) error = SparkGemma4ValCopyUp(chain.key_norm_device,chain.key_norm,(uint64_t)chain.head_dimension * 2u);
	if (error == cudaSuccess) error = SparkGemma4ValSync();
	if (SparkGemma4ValCuda(error,"chain_sliding") != 0)
		return(1);
	SparkGemma4ValChainMirror(&chain,&kv);
	error = SparkGemma4LaunchSlidingWindowPositions(cudaStreamPerThread,(const uint32_t *)chain.kv_sequence_device,(const uint32_t *)chain.kv_context_device,(const uint32_t *)chain.kv_row_position_device,1u,(uint32_t *)chain.window_device);
	if (error == cudaSuccess) error = SparkGemma4ValChainDeviceStages(&chain);
	if (error == cudaSuccess) error = SparkGemma4ValSync();
	if (error == cudaSuccess) error = SparkGemma4ValChainDeviceTail(&chain);
	if (error == cudaSuccess) error = SparkGemma4ValSync();
	if (error == cudaSuccess) error = SparkGemma4ValCopyDown(chain.actual_hidden,chain.h_device,(uint64_t)SPARK_GEMMA4_VAL_CHAIN_ROWS * hidden * 2u);
	if (error == cudaSuccess) error = SparkGemma4ValCopyDown(chain.actual_normed,chain.normed_device,(uint64_t)SPARK_GEMMA4_VAL_CHAIN_ROWS * hidden * 2u);
	if (error == cudaSuccess) error = SparkGemma4ValCopyDown(chain.mirror_dec,chain.query_device,(uint64_t)SPARK_GEMMA4_VAL_CHAIN_ROWS * chain.query_out * 2u);
	if (SparkGemma4ValCuda(error,"chain_sliding") != 0)
		return(1);
	if (SparkGemma4ValKvCheckErrorClear(&kv,"chain_sliding") != 0)
		return(1);
	count = (uint64_t)SPARK_GEMMA4_VAL_CHAIN_ROWS * chain.query_out;
	actual_f = (float *)malloc(count * sizeof(float));
	expected_f = (float *)malloc(count * sizeof(float));
	if (actual_f == 0 || expected_f == 0)
		return(SparkGemma4ValFail("chain_sliding","mirror_alloc"));
	for (element = 0u; element < count; element++)
	{
		actual_f[element] = SparkGemma4ValFromBf16(chain.mirror_dec[element]);
		expected_f[element] = SparkGemma4ValFromBf16(chain.expected_dec[element]);
	}
	SparkGemma4ValMeasure(&metrics,actual_f,expected_f,count);
	if (SparkGemma4ValReport("chain_sliding_attention_dataflow",&metrics,5e-3,0.999) != 0)
		return(1);
	free(actual_f);
	free(expected_f);
	count = (uint64_t)SPARK_GEMMA4_VAL_CHAIN_ROWS * hidden;
	if (SparkGemma4ValCompareBf16("chain_sliding_hidden",chain.actual_hidden,chain.expected_hidden,count) != 0)
		return(1);
	if (SparkGemma4ValCompareBf16("chain_sliding_normed",chain.actual_normed,chain.expected_normed,count) != 0)
		return(1);
	{
		uint16_t *rerun_hidden = (uint16_t *)malloc((uint64_t)SPARK_GEMMA4_VAL_CHAIN_ROWS * hidden * 2u);
		uint16_t *rerun_normed = (uint16_t *)malloc((uint64_t)SPARK_GEMMA4_VAL_CHAIN_ROWS * hidden * 2u);
		if (rerun_hidden == 0 || rerun_normed == 0)
			return(SparkGemma4ValFail("chain_determinism","host_alloc"));
		error = SparkGemma4ValCopyUp(chain.h_device,chain.h0,(uint64_t)SPARK_GEMMA4_VAL_CHAIN_ROWS * hidden * 2u);
		if (error == cudaSuccess) error = SparkGemma4ValChainDeviceStages(&chain);
		if (error == cudaSuccess) error = SparkGemma4ValSync();
		if (error == cudaSuccess) error = SparkGemma4ValChainDeviceTail(&chain);
		if (error == cudaSuccess) error = SparkGemma4ValSync();
		if (error == cudaSuccess) error = SparkGemma4ValCopyDown(rerun_hidden,chain.h_device,(uint64_t)SPARK_GEMMA4_VAL_CHAIN_ROWS * hidden * 2u);
		if (error == cudaSuccess) error = SparkGemma4ValCopyDown(rerun_normed,chain.normed_device,(uint64_t)SPARK_GEMMA4_VAL_CHAIN_ROWS * hidden * 2u);
		if (SparkGemma4ValCuda(error,"chain_determinism") != 0)
			return(1);
		if (memcmp(rerun_hidden,chain.actual_hidden,(uint64_t)SPARK_GEMMA4_VAL_CHAIN_ROWS * hidden * 2u) != 0
			|| memcmp(rerun_normed,chain.actual_normed,(uint64_t)SPARK_GEMMA4_VAL_CHAIN_ROWS * hidden * 2u) != 0)
			return(SparkGemma4ValFail("chain_determinism","rerun_bit_exact"));
		free(rerun_hidden);
		free(rerun_normed);
	}
	printf("gemma4_validation check=chain_determinism bit_exact=1\n");
	gemma4_val_sites++;
	cudaFree(chain.h_device);
	cudaFree(chain.normed_device);
	cudaFree(chain.query_device);
	cudaFree(chain.kv_device);
	cudaFree(chain.att_device);
	cudaFree(chain.gu_device);
	cudaFree(chain.mlp_device);
	cudaFree(chain.gain_device);
	cudaFree(chain.positions_device);
	cudaFree(chain.query_norm_device);
	cudaFree(chain.key_norm_device);
	cudaFree(chain.window_device);
	cudaFree(chain.kv_row_position_device);
	SparkGemma4ValChainFree(&chain);
	SparkGemma4ValKvTeardown(&kv);
	return(0);
}

int main(int argc, char **argv)
{
	int result = 0;
	if (argc != 2 || strlen(argv[1]) != 64u)
	{
		fprintf(stderr,"usage: %s VALIDATION_CONFIGURATION_SHA256\n",argv[0]);
		return(2);
	}
	if (SparkGemma4ValCuda(SparkGemma4ConfigureCudaKernels(),"configure") != 0)
		return(1);
	if (result == 0) result = SparkGemma4ValCheckSelf();
	if (result == 0) result = SparkGemma4ValCheckEmbedding();
	if (result == 0) result = SparkGemma4ValCheckNorms();
	if (result == 0) result = SparkGemma4ValCheckHeadNorms();
	if (result == 0) result = SparkGemma4ValCheckLinear();
	if (result == 0) result = SparkGemma4ValCheckAdds();
	if (result == 0) result = SparkGemma4ValCheckGatedGelu();
	if (result == 0) result = SparkGemma4ValCheckRope();
	if (result == 0) result = SparkGemma4ValCheckWindow();
	if (result == 0) result = SparkGemma4ValCheckKvSliding();
	if (result == 0) result = SparkGemma4ValCheckKvFull();
#if SPARK_GEMMA4_MODEL_MOE_BLOCK
	if (result == 0) result = SparkGemma4ValCheckRouter();
#endif
	if (result == 0) result = SparkGemma4ValCheckChainSliding();
	if (result == 0)
		printf("gemma4_validation PASS arm=%s sites=%u\n",SPARK_GEMMA4_MODEL_MODULE_ID,gemma4_val_sites);
	return(result);
}
