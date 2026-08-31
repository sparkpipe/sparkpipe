// Bit-exactness oracle for the small-batch batched linear kernel.
//
// SparkLmBatchedLinearKernel (model-families/common/include/sparkpipe/
// spark_lm_kernels.cuh) carries rows 2..15 of a decode batch on ONE weight
// stream, where the scalar GEMV re-read the matrix once per row. Its claim
// is stronger than "agrees within family tolerance": every row keeps the
// scalar kernel's own FMA chain, so this oracle diffs it against
// SparkLmLinearKernel - the kernel that owns B=1 - EXACTLY, on the same
// weights, for every weight format the shared gate dispatches.
//
// Both kernels run here under the THREADS==1 harness, which is the proof the
// kernels stayed host-compilable. The static __shared__ tile bodies and the
// grouped/native kernels in the header parse but are never called. The
// activation-codec path is not exercised: LmActivationFp8QdqFloatRow's base
// stride is (blockDim.x/32)*128, which is zero under the one-thread harness,
// so a codec case would spin rather than lie - and the codec is not part of
// any small-B call site this kernel serves (the 27B FP8 gate and the head
// shadow both pass ACTIVATION_CODEC_NONE).

#include "tests/host_cuda/lm_host_cuda.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

LmHostDim3 blockIdx, threadIdx, blockDim, gridDim;

// Dynamic shared backing: both linear kernels declare
// extern __shared__ float shared_input[]. 32768 floats covers the scalar
// kernel at K=8192 and the batched kernel at 15 rows (15*512).
static float shared_input[32768];

// The shared header's tensor-tile kernels need nvcuda::wmma to PARSE on the
// host compiler. These declarations are never invoked: the oracle calls only
// the two scalar-arithmetic kernels, and any call would fail to link - which
// is the loud failure the harness prefers.
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
// mem_row_major is an enumerator in the real header - a VALUE here, unlike
// row_major/col_major which only ever appear as fragment layout types.
enum { mem_row_major = 0 };
// Every fragment<...> in the header uses the same six-argument shape
// (use, M, N, K, element type, layout), so the stub is that exact arity - a
// variadic pack cannot bind the mixed type-and-integer argument list.
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

// The launcher bodies reference the runtime API directly. They are never
// CALLED under the harness (kernels run through LM_HOST_LAUNCH), but the
// declarations must exist for the bodies to compile.
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
		// Sane positive f32 scales in [1, 2).
		bits = 0x3f800000u | (lm_batched_next() & 0x007fff00u);
		memcpy(&scales[index],&bits,sizeof(float));
	}
}

static void lm_fill_e8m0_scales(uint8_t *codes, uint32_t count)
{
	uint32_t index;
	for (index = 0u; index < count; index++)
		// E8M0 codes in [120, 134] - powers of two in [2^-7, 2^7), never 0xff.
		codes[index] = (uint8_t)(120u + (lm_batched_next() % 15u));
}

static void lm_fill_bf16(uint16_t *values, uint32_t count)
{
	uint32_t index,bits;
	for (index = 0u; index < count; index++)
	{
		// Small-magnitude bf16 activations: exponent 118..122, random mantissa.
		bits = lm_batched_next();
		values[index] = (uint16_t)(((bits & 1u) << 15u) | ((118u + ((bits >> 1u) % 5u)) << 7u) | ((bits >> 3u) & 0x7fu));
	}
}

// Packed weight words per (K, N) for each format: BF16 is 2 bytes per
// element, FP8 is 1, MXFP4 is 0.5. The oracle sizes and fills by the LARGEST
// format's count so one fill per case feeds every format's read pattern.
#define MAX_TEST_ROWS 15u
#define MAX_TEST_DIM 2048u
#define MAX_TEST_K 8192u

static uint16_t input_bf16[MAX_TEST_ROWS * MAX_TEST_K];
static uint16_t reference_output[MAX_TEST_ROWS * MAX_TEST_DIM];
static uint16_t batched_output[MAX_TEST_ROWS * MAX_TEST_DIM];
static uint16_t single_output[MAX_TEST_ROWS * MAX_TEST_DIM];

// 3M words covers the largest case below (FP8 at 5120 x 2048 = 2.6M words).
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

// Scale bytes the kernel's own indexing can touch: E8M0B128 keeps one code
// per row per 128-K group, the group-32 formats keep one per row per 32.
static uint64_t lm_scale_u8_words(uint32_t weight_format, uint32_t input_dimension, uint32_t output_dimension)
{
	if ( weight_format == SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_E8M0B128 )
		return (uint64_t)output_dimension * (input_dimension / 128u);
	return (uint64_t)output_dimension * (input_dimension / 32u);
}

// Run one case: diff the batched kernel against the scalar kernel it must
// match bit for bit, at the call-site CTA geometry (scalar_warps).
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
				/* Odd K makes row r's element start r*K pair-MISALIGNED, which
				 * the scalar kernel's uint32 pair loads silently misread (a
				 * pre-existing defect of the multi-row scalar route this kernel
				 * replaces). The alignment-valid invocation of the same kernel
				 * is one row per launch on an offset pointer - byte offset
				 * r*K*2 is always even - so the reference here is the per-row
				 * form, and the batched kernel must still match it exactly. */
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

// The row-1 pin: the batched kernel at B=1 must equal the scalar kernel's
// B=1 answer too, so nothing about the route change can move B1 numerics.
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
		// BF16, exact-width single chunk, partial final CTA (20 % 8 != 0).
		run_case(SPARK_LM_WEIGHT_FORMAT_BF16,row_index,128u,20u,8,"bf16-narrow");
		// BF16, odd K - the scalar tail element, chunked (512 + 1).
		run_case(SPARK_LM_WEIGHT_FORMAT_BF16,row_index,513u,8u,8,"bf16-odd-k");
		// BF16, multi-chunk with a partial trailing chunk (512+512+128).
		run_case(SPARK_LM_WEIGHT_FORMAT_BF16,row_index,1152u,40u,8,"bf16-multichunk");
		// FP8 E8M0B128 - the qwen38_27b decode pack format (the measured
		// 8.31 flat spot). Multi-chunk K, CTA not full.
		run_case(SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_E8M0B128,row_index,1152u,64u,8,"fp8-e8m0-27b-shape");
		// FP8 E8M0B128 at the real 27B projection shape.
		run_case(SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_E8M0B128,row_index,5120u,2048u,8,"fp8-e8m0-27b-wide");
		// FP8 F32B128 - the MiMo 2-D tile scales.
		run_case(SPARK_LM_WEIGHT_FORMAT_FP8_E4M3_F32B128,row_index,512u,512u,8,"fp8-f32b128");
		// FP8 E8M0 group scales (format 4).
		run_case(SPARK_LM_WEIGHT_FORMAT_FP8_E4M3,row_index,1024u,128u,8,"fp8-group32");
		// MXFP4 - the head shadow projection format, 8-warp CTAs.
		run_case(SPARK_LM_WEIGHT_FORMAT_MXFP4_E2M1,row_index,1024u,256u,8,"mxfp4-shadow-8w");
		// MXFP4 at the head call site's constants: 32 warps, 1024 threads.
		run_case(SPARK_LM_WEIGHT_FORMAT_MXFP4_E2M1,row_index,5120u,512u,32,"mxfp4-shadow-32w");
	}

	// The B1 pin per format at one shape each.
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
