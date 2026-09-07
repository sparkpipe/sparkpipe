
#include "tests/host_cuda/lm_host_cuda.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

LmHostDim3 blockIdx, threadIdx, blockDim, gridDim;

static float shared_input[32768];

#ifndef LM_HOST_CUDA_WMMA_STUB
#define LM_HOST_CUDA_WMMA_STUB
namespace nvcuda
{
namespace wmma
{
struct matrix_a {};
struct matrix_b {};
struct accumulator {};
struct row_major {};
struct col_major {};
enum { mem_row_major = 0 };
template <typename Use, int M, int N, int K, typename Element,
	typename Layout = void>
struct fragment { unsigned storage[8]; };
template <typename Fragment, typename Value> static inline void fill_fragment(Fragment &, Value) {}
template <typename Fragment, typename Pointer, typename Stride> static inline void load_matrix_sync(Fragment &, Pointer, Stride) {}
template <typename Pointer, typename Fragment, typename Stride, typename Layout> static inline void store_matrix_sync(Pointer, Fragment &, Stride, Layout) {}
template <typename Accum, typename A, typename B, typename C> static inline void mma_sync(Accum &, A &, B &, C &) {}
}
}
#endif

static inline cudaError_t cudaMemsetAsync(void *, int, size_t, cudaStream_t) { return cudaSuccess; }
static inline cudaError_t cudaMalloc(void **, size_t) { return cudaErrorInvalidValue; }

#include "inference/kernels/activation.cuh"
#include "sparkpipe/spark_lm_kernels.cuh"

static uint32_t lm_batched_seed = 0x9e3779b9u;

static uint32_t lm_batched_next(void)
{
	lm_batched_seed = lm_batched_seed * 1664525u + 1013904223u;
	return lm_batched_seed;
}

static void lm_fill_words(uint32_t *words, uint32_t word_count)
{
	uint32_t index;
	for (index = 0u; index < word_count; index++)
		words[index] = lm_batched_next();
}

static void lm_fill_f32_scales(float *scales, uint32_t count)
{
	uint32_t index,bits;
	for (index = 0u; index < count; index++)
	{
		bits = 0x3f800000u | (lm_batched_next() & 0x007fff00u);
		memcpy(&scales[index],&bits,sizeof(float));
	}
}

static void lm_fill_e8m0_scales(uint8_t *codes, uint32_t count)
{
	uint32_t index;
	for (index = 0u; index < count; index++)
		codes[index] = (uint8_t)(120u + (lm_batched_next() % 15u));
}

static void lm_fill_bf16(uint16_t *values, uint32_t count)
{
	uint32_t index,bits;
	for (index = 0u; index < count; index++)
	{
		bits = lm_batched_next();
		values[index] = (uint16_t)(((bits & 1u) << 15u) | ((118u + ((bits >> 1u) % 5u)) << 7u) | ((bits >> 3u) & 0x7fu));
	}
}

#define MAX_TEST_ROWS 15u
#define MAX_TEST_DIM 2048u
#define MAX_TEST_K 8192u

static uint16_t input_bf16[MAX_TEST_ROWS * MAX_TEST_K];
static uint16_t reference_output[MAX_TEST_ROWS * MAX_TEST_DIM];
static uint16_t batched_output[MAX_TEST_ROWS * MAX_TEST_DIM];
static uint16_t single_output[MAX_TEST_ROWS * MAX_TEST_DIM];

static uint32_t payload_words[3u * 1024u * 1024u];
static float scale_f32[(MAX_TEST_K / 128u + 1u) * (MAX_TEST_K / 128u + 1u)];
static uint8_t scale_u8[(MAX_TEST_DIM / 32u + 1u) * MAX_TEST_DIM];

static uint32_t failures = 0u;

static uint64_t lm_payload_words(uint32_t weight_format, uint32_t input_dimension, uint32_t output_dimension)
{
	uint64_t elements = (uint64_t)input_dimension * output_dimension;
	if ( weight_format == SPARK_LM_WEIGHT_FORMAT_BF16 )
		return elements / 2u;
	if ( weight_format == SPARK_LM_WEIGHT_FORMAT_MXFP4_E2M1 )
		return elements / 8u;
	return elements / 4u;
}

static uint64_t lm_scale_u8_words(uint32_t weight_format, uint32_t input_dimension, uint32_t output_dimension)
{
	if ( weight_format == SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_E8M0B128 )
		return (uint64_t)output_dimension * (input_dimension / 128u);
	return (uint64_t)output_dimension * (input_dimension / 32u);
}

static void run_case(uint32_t weight_format, uint32_t row_count, uint32_t input_dimension, uint32_t output_dimension, int scalar_warps, const char *label)
{
	uint64_t words = lm_payload_words(weight_format,input_dimension,output_dimension);
	uint64_t scale_f32_count = weight_format == SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_F32B128 ?
		(uint64_t)(output_dimension / 128u + 1u) * (input_dimension / 128u + 1u) : 1u;
	uint64_t scale_u8_count = lm_scale_u8_words(weight_format,input_dimension,output_dimension);
	uint32_t row,offset,reference_row;
	const void *weight_scale = weight_format == SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_F32B128 ? (const void *)scale_f32 : (const void *)scale_u8;

	if ( words > sizeof(payload_words) / sizeof(uint32_t) ||
		scale_f32_count > sizeof(scale_f32) / sizeof(float) ||
		scale_u8_count > sizeof(scale_u8) ||
		(uint64_t)row_count * input_dimension > sizeof(input_bf16) / sizeof(uint16_t) )
	{
		printf("FAIL %s: fixture too small (K=%u N=%u)\n",label,
			(unsigned)input_dimension,(unsigned)output_dimension);
		failures++;
		return;
	}
	lm_fill_words(payload_words,(uint32_t)words);
	lm_fill_f32_scales(scale_f32,(uint32_t)scale_f32_count);
	lm_fill_e8m0_scales(scale_u8,(uint32_t)scale_u8_count);
	lm_fill_bf16(input_bf16,row_count * input_dimension);
	memset(reference_output,0xCD,sizeof(reference_output));
	memset(batched_output,0xCD,sizeof(batched_output));

	if ( scalar_warps == 8 )
	{
		LM_HOST_LAUNCH(dim3(row_count,(output_dimension + 7u) / 8u),
			(SparkLmLinearKernel<32u,SPARK_ACTIVATION_CODEC_NONE,8u>(
				weight_format,(const void *)payload_words,weight_scale,
				input_bf16,reference_output,row_count,input_dimension,
				output_dimension)));
		LM_HOST_LAUNCH(dim3((output_dimension + 7u) / 8u),
			(SparkLmBatchedLinearKernel<32u,SPARK_ACTIVATION_CODEC_NONE,
				SPARK_LM_TILE,8u>(weight_format,
				(const void *)payload_words,weight_scale,input_bf16,
				batched_output,row_count,input_dimension,output_dimension)));
	}
	else
	{
		LM_HOST_LAUNCH(dim3(row_count,(output_dimension + 31u) / 32u),
			(SparkLmLinearKernel<32u,SPARK_ACTIVATION_CODEC_NONE,32u>(
				weight_format,(const void *)payload_words,weight_scale,
				input_bf16,reference_output,row_count,input_dimension,
				output_dimension)));
		LM_HOST_LAUNCH(dim3((output_dimension + 31u) / 32u),
			(SparkLmBatchedLinearKernel<32u,SPARK_ACTIVATION_CODEC_NONE,
				SPARK_LM_TILE,32u>(weight_format,
				(const void *)payload_words,weight_scale,input_bf16,
				batched_output,row_count,input_dimension,output_dimension)));
	}

	for (row = 0u; row < row_count; row++)
	{
		for (offset = 0u; offset < output_dimension; offset++)
		{
			uint32_t slot = (uint32_t)(row * output_dimension) + offset;
			if ( reference_output[slot] == batched_output[slot] )
				continue;
			if ( (input_dimension & 1u) != 0u )
			{
				memset(single_output,0xCD,sizeof(single_output));
				for (reference_row = 0u; reference_row < row_count; reference_row++)
					LM_HOST_LAUNCH(dim3(1u,1u),
						(SparkLmLinearKernel<32u,SPARK_ACTIVATION_CODEC_NONE,8u>(
							weight_format,(const void *)payload_words,weight_scale,
							input_bf16 + ((uint64_t)reference_row * input_dimension),
							single_output + ((uint64_t)reference_row * output_dimension),
							1u,input_dimension,output_dimension)));
				if ( single_output[slot] == batched_output[slot] )
					continue;
			}
			printf("FAIL %s rows=%u: row %u neuron %u scalar=0x%04x batched=0x%04x\n",
				label,(unsigned)row_count,(unsigned)row,(unsigned)offset,
				(unsigned)reference_output[slot],
				(unsigned)batched_output[slot]);
			failures++;
			return;
		}
	}
	printf("PASS %s rows=%u K=%u N=%u warps=%d bit-exact\n",label,
		(unsigned)row_count,(unsigned)input_dimension,
		(unsigned)output_dimension,scalar_warps);
}

static void run_case_single(uint32_t weight_format, uint32_t input_dimension, uint32_t output_dimension, const char *label)
{
	uint64_t words = lm_payload_words(weight_format,input_dimension,output_dimension);
	uint64_t scale_f32_count = weight_format == SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_F32B128 ?
		(uint64_t)(output_dimension / 128u + 1u) * (input_dimension / 128u + 1u) : 1u;
	uint64_t scale_u8_count = lm_scale_u8_words(weight_format,input_dimension,output_dimension);
	const void *weight_scale = weight_format == SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_F32B128 ? (const void *)scale_f32 : (const void *)scale_u8;

	if ( words > sizeof(payload_words) / sizeof(uint32_t) ||
		(uint64_t)input_dimension > sizeof(input_bf16) / sizeof(uint16_t) )
	{
		printf("FAIL %s: fixture too small\n",label);
		failures++;
		return;
	}
	lm_fill_words(payload_words,(uint32_t)words);
	lm_fill_f32_scales(scale_f32,(uint32_t)scale_f32_count);
	lm_fill_e8m0_scales(scale_u8,(uint32_t)scale_u8_count);
	lm_fill_bf16(input_bf16,input_dimension);
	memset(reference_output,0xCD,sizeof(reference_output));
	memset(single_output,0xCD,sizeof(single_output));

	LM_HOST_LAUNCH(dim3(1u,(output_dimension + 7u) / 8u),
		(SparkLmLinearKernel<32u,SPARK_ACTIVATION_CODEC_NONE,8u>(
			weight_format,(const void *)payload_words,weight_scale,
			input_bf16,reference_output,1u,input_dimension,output_dimension)));
	LM_HOST_LAUNCH(dim3((output_dimension + 7u) / 8u),
		(SparkLmBatchedLinearKernel<32u,SPARK_ACTIVATION_CODEC_NONE,
			SPARK_LM_TILE,8u>(weight_format,(const void *)payload_words,
			weight_scale,input_bf16,single_output,1u,input_dimension,
			output_dimension)));
	if ( memcmp(reference_output,single_output,sizeof(uint16_t) * output_dimension) != 0 )
	{
		printf("FAIL %s single-row pin\n",label);
		failures++;
		return;
	}
	printf("PASS %s B=1 pin bit-exact\n",label);
}


int main(void)
{
	static const uint32_t rows_per_case[] = {1u,2u,3u,4u,5u,8u,15u};
	uint32_t case_index,row_index;

	for (case_index = 0u; case_index < sizeof(rows_per_case) / sizeof(rows_per_case[0]); case_index++)
	{
		row_index = rows_per_case[case_index];
		run_case(SPARK_LM_WEIGHT_FORMAT_BF16,row_index,128u,20u,8,"bf16-narrow");
		run_case(SPARK_LM_WEIGHT_FORMAT_BF16,row_index,513u,8u,8,"bf16-odd-k");
		run_case(SPARK_LM_WEIGHT_FORMAT_BF16,row_index,1152u,40u,8,"bf16-multichunk");
		run_case(SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_E8M0B128,row_index,1152u,64u,8,"fp8-e8m0-27b-shape");
		run_case(SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_E8M0B128,row_index,5120u,2048u,8,"fp8-e8m0-27b-wide");
		run_case(SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_F32B128,row_index,512u,512u,8,"fp8-f32b128");
		run_case(SPARK_LM_WEIGHT_FORMAT_FP8_E4M3,row_index,1024u,128u,8,"fp8-group32");
		run_case(SPARK_LM_WEIGHT_FORMAT_MXFP4_E2M1,row_index,1024u,256u,8,"mxfp4-shadow-8w");
		run_case(SPARK_LM_WEIGHT_FORMAT_MXFP4_E2M1,row_index,5120u,512u,32,"mxfp4-shadow-32w");
	}

	run_case_single(SPARK_LM_WEIGHT_FORMAT_BF16,256u,64u,"bf16");
	run_case_single(SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_E8M0B128,512u,128u,"fp8-e8m0");
	run_case_single(SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_F32B128,512u,512u,"fp8-f32b128");
	run_case_single(SPARK_LM_WEIGHT_FORMAT_FP8_E4M3,512u,128u,"fp8");
	run_case_single(SPARK_LM_WEIGHT_FORMAT_MXFP4_E2M1,512u,256u,"mxfp4");

	if ( failures != 0u )
	{
		printf("FAILED: %u case(s)\n",(unsigned)failures);
		return 1;
	}
	printf("spark_lm_batched_host: all cases bit-exact\n");
	return 0;
}
