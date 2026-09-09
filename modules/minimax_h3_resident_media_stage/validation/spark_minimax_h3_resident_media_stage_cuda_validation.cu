#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_minimax_h3_model.h"
#include "sparkpipe/spark_minimax_h3_scheduler.h"
#include "sparkpipe/spark_numerical_metrics.h"

#define SPARK_H3_REF_EMBED 1
#include "spark_minimax_h3_reference.c"

extern "C" cudaError_t SparkMinimaxH3Rope3d(cudaStream_t stream, const void *input_bf16,
	const float *cos_angles, const float *sin_angles, uint32_t rows, uint32_t heads,
	uint32_t head_dim, uint32_t rope_dim, void *output_bf16);
extern "C" cudaError_t SparkMinimaxH3DenseAttention(cudaStream_t stream,
	const void *queries_bf16, const void *keys_bf16, const void *values_bf16,
	uint32_t seq, uint32_t heads, uint32_t head_dim, void *output_bf16);
extern "C" cudaError_t SparkMinimaxH3AdaLNAffine(cudaStream_t stream,
	const void *input_bf16, const void *scale_bf16, const void *shift_bf16,
	uint32_t rows, uint32_t width, void *output_bf16);
extern "C" cudaError_t SparkMinimaxH3GateResidual(cudaStream_t stream,
	const void *value_bf16, const void *gate_bf16, const void *residual_bf16,
	uint32_t rows, uint32_t width, void *output_bf16);
extern "C" cudaError_t SparkMinimaxH3SiluMul(cudaStream_t stream,
	const void *gate_up_bf16, uint32_t rows, uint32_t ffn, void *output_bf16);
extern "C" cudaError_t SparkMinimaxH3Conv2d3x3Reflect(cudaStream_t stream,
	const void *input_bf16, const void *weight_bf16, const void *bias_bf16,
	uint32_t in_channels, uint32_t out_channels, uint32_t height, uint32_t width,
	void *output_bf16);
extern "C" cudaError_t SparkMinimaxH3Conv1dDilated(cudaStream_t stream,
	const void *input_bf16, const void *weight_bf16, const void *bias_bf16,
	uint32_t channels, uint32_t length, uint32_t dilation, void *output_bf16);
extern "C" cudaError_t SparkMinimaxH3Snake(cudaStream_t stream, const void *input_bf16,
	const void *alpha_bf16, uint32_t channels, uint32_t length, void *output_bf16);
extern "C" int32_t SparkMinimaxH3SchedulerBuildSigmas(float shift,
	uint32_t sigma_point_count, float *sigmas_out, uint32_t *sigma_count_out);

#define SPARK_MINIMAX_H3_VAL_MAX_RELATIVE_L2 2e-2
#define SPARK_MINIMAX_H3_VAL_MIN_COSINE 0.9999

#define SPARK_MINIMAX_H3_VAL_CONV1D_C 8u
#define SPARK_MINIMAX_H3_VAL_CONV1D_L 32u

static uint32_t SparkMinimaxH3ValFailures;

static uint16_t SparkMinimaxH3ValToBf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	bits += 0x8000u;
	return((uint16_t)(bits >> 16u));
}

static float SparkMinimaxH3ValFromBf16(uint16_t packed)
{
	uint32_t bits = (uint32_t)packed << 16u;
	float value;
	memcpy(&value,&bits,sizeof(value));
	return(value);
}

static void SparkMinimaxH3ValFill(uint16_t *packed, float *exact, uint64_t count,
	float scale, uint64_t *state)
{
	uint64_t index;
	for (index=0u; index<count; index++)
	{
		float uniform;
		*state = *state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
		uniform = (float)(*state >> 11u) / 1000000.0f - 1.0f;
		packed[index] = SparkMinimaxH3ValToBf16(uniform * scale);
		exact[index] = SparkMinimaxH3ValFromBf16(packed[index]);
	}
}

static int SparkMinimaxH3ValCompare(const char *check, const float *actual,
	const float *reference, uint64_t count)
{
	SparkNumericalMetrics metrics = SparkNumericalMeasureF32(actual,reference,count);
	uint32_t within = SparkNumericalMetricsWithin(&metrics,
		SPARK_MINIMAX_H3_VAL_MAX_RELATIVE_L2,SPARK_MINIMAX_H3_VAL_MIN_COSINE);
	printf("%-24s rel_l2=%.6f cosine=%.8f max_abs=%.6f %s\n",check,
		metrics.relative_l2,metrics.cosine,metrics.max_absolute,
		within != 0u ? "OK" : "FAIL");
	if ( within == 0u )
	{
		SparkMinimaxH3ValFailures++;
		return(1);
	}
	return(0);
}

static int SparkMinimaxH3ValCuda(cudaError_t error, const char *check)
{
	if ( error == cudaSuccess )
		return(0);
	printf("%-24s cuda=%s FAIL\n",check,cudaGetErrorString(error));
	SparkMinimaxH3ValFailures++;
	return(1);
}

static void *SparkMinimaxH3ValDevice(uint64_t bytes)
{
	void *pointer;
	if ( cudaMalloc(&pointer,(size_t)bytes) != cudaSuccess )
		return(0);
	return(pointer);
}

static float *SparkMinimaxH3ValReadBf16(void *device_buffer, uint64_t count)
{
	float *host = (float *)malloc(count * 4u);
	uint16_t *packed;
	uint64_t index;
	if ( host == 0 )
		return(0);
	packed = (uint16_t *)malloc(count * 2u);
	if ( packed == 0 )
	{
		free(host);
		return(0);
	}
	if ( cudaMemcpy(packed,device_buffer,count * 2u,cudaMemcpyDeviceToHost) !=
		cudaSuccess )
	{
		free(host);
		free(packed);
		return(0);
	}
	for (index=0u; index<count; index++)
		host[index] = SparkMinimaxH3ValFromBf16(packed[index]);
	free(packed);
	return(host);
}

int main(int argc, char **argv)
{
	uint64_t rng = UINT64_C(0x68333253504b5350);
	const uint32_t seq = SPARK_H3_REF_SEQ;
	const uint32_t heads = SPARK_H3_REF_HEADS;
	const uint32_t head_dim = SPARK_H3_REF_HEAD_DIM;
	const uint32_t rope_dim = SPARK_H3_REF_ROPE_DIM;
	const uint32_t hidden = SPARK_H3_REF_HIDDEN;
	const uint32_t ffn = SPARK_H3_REF_FFN;
	const uint32_t conv_c = SPARK_H3_REF_CONV_C;
	const uint32_t conv_h = SPARK_H3_REF_CONV_H;
	const uint32_t conv_w = SPARK_H3_REF_CONV_W;
	const uint64_t slots = (uint64_t)seq * heads * head_dim;
	const uint64_t rows = (uint64_t)seq * hidden;
	const uint64_t conv_count = (uint64_t)conv_c * conv_h * conv_w;
	const uint64_t conv1d_count = (uint64_t)SPARK_MINIMAX_H3_VAL_CONV1D_C *
		SPARK_MINIMAX_H3_VAL_CONV1D_L;
	uint16_t *packed_a = (uint16_t *)malloc(rows * 2u);
	uint16_t *packed_b = (uint16_t *)malloc(rows * 2u);
	float *exact_a = (float *)malloc(rows * 4u);
	float *exact_b = (float *)malloc(rows * 4u);
	cudaStream_t stream = 0;
	if ( argc != 2 )
	{
		printf("usage: %s VALIDATION_CONFIGURATION_SHA256\n",argv[0]);
		return(2);
	}
	if ( packed_a == 0 || packed_b == 0 || exact_a == 0 || exact_b == 0 )
	{
		printf("host allocation failure\n");
		return(2);
	}
	printf("minimax_h3 resident media stage V2 gate; config %s\n",argv[1]);
	printf("geometry: seq=%u heads=%u head_dim=%u rope_dim=%u hidden=%u ffn=%u "
		"conv2d=%ux%ux%u conv1d=%ux%u gate=rel_l2<=%g cosine>=%g\n",seq,heads,
		head_dim,rope_dim,hidden,ffn,conv_c,conv_h,conv_w,
		SPARK_MINIMAX_H3_VAL_CONV1D_C,SPARK_MINIMAX_H3_VAL_CONV1D_L,
		SPARK_MINIMAX_H3_VAL_MAX_RELATIVE_L2,SPARK_MINIMAX_H3_VAL_MIN_COSINE);

	{
		double positions[SPARK_H3_REF_SEQ * 3u];
		float cos_angles[SPARK_H3_REF_SEQ * SPARK_H3_REF_ROPE_DIM];
		float sin_angles[SPARK_H3_REF_SEQ * SPARK_H3_REF_ROPE_DIM];
		float *rotated = (float *)malloc(slots * 4u);
		float *device_values = 0;
		void *cos_device = SparkMinimaxH3ValDevice(
			(uint64_t)seq * rope_dim * 4u);
		void *sin_device = SparkMinimaxH3ValDevice(
			(uint64_t)seq * rope_dim * 4u);
		void *q_in = SparkMinimaxH3ValDevice(slots * 2u);
		void *q_out = SparkMinimaxH3ValDevice(slots * 2u);
		uint32_t row;
		if ( rotated == 0 || cos_device == 0 || sin_device == 0 || q_in == 0 ||
			q_out == 0 )
		{
			printf("allocation failure (rope3d)\n");
			return(2);
		}
		for (row=0u; row<seq; row++)
		{
			positions[row * 3u] = (double)(row % 7u);
			positions[row * 3u + 1u] = (double)(row % 3u);
			positions[row * 3u + 2u] = (double)(row % 5u);
		}
		SparkH3RefRopeAngles(positions,seq,SPARK_MINIMAX_H3_DIT_ROPE_THETA,
			SPARK_H3_REF_ROPE_FREQ_DIM,cos_angles,sin_angles);
		SparkMinimaxH3ValFill(packed_a,exact_a,slots,1.0f,&rng);
		SparkH3RefApplyRope(exact_a,cos_angles,sin_angles,seq,heads,head_dim,
			rope_dim,rotated);
		cudaMemcpy(cos_device,cos_angles,(size_t)seq * rope_dim * 4u,
			cudaMemcpyHostToDevice);
		cudaMemcpy(sin_device,sin_angles,(size_t)seq * rope_dim * 4u,
			cudaMemcpyHostToDevice);
		cudaMemcpy(q_in,packed_a,slots * 2u,cudaMemcpyHostToDevice);
		if ( SparkMinimaxH3ValCuda(SparkMinimaxH3Rope3d(stream,q_in,
			(const float *)cos_device,(const float *)sin_device,seq,heads,
			head_dim,rope_dim,q_out),"rope3d") == 0 )
		{
			device_values = SparkMinimaxH3ValReadBf16(q_out,slots);
			if ( device_values == 0 )
			{
				printf("rope3d readback FAIL\n");
				SparkMinimaxH3ValFailures++;
			}
			else
				SparkMinimaxH3ValCompare("rope3d",device_values,rotated,slots);
		}
		free(device_values);
		cudaFree(cos_device);
		cudaFree(sin_device);
		cudaFree(q_in);
		cudaFree(q_out);
		free(rotated);
	}

	{
		float *attended = (float *)malloc(slots * 4u);
		float *device_values = 0;
		void *q_in = SparkMinimaxH3ValDevice(slots * 2u);
		void *k_in = SparkMinimaxH3ValDevice(slots * 2u);
		void *v_in = SparkMinimaxH3ValDevice(slots * 2u);
		void *out = SparkMinimaxH3ValDevice(slots * 2u);
		if ( attended == 0 || q_in == 0 || k_in == 0 || v_in == 0 || out == 0 )
		{
			printf("allocation failure (attention)\n");
			return(2);
		}
		SparkMinimaxH3ValFill(packed_a,exact_a,slots,1.0f,&rng);
		SparkMinimaxH3ValFill(packed_b,exact_b,slots,1.0f,&rng);
		SparkH3RefDenseAttention(exact_a,exact_a,exact_b,seq,heads,head_dim,attended);
		cudaMemcpy(q_in,packed_a,slots * 2u,cudaMemcpyHostToDevice);
		cudaMemcpy(k_in,packed_a,slots * 2u,cudaMemcpyHostToDevice);
		cudaMemcpy(v_in,packed_b,slots * 2u,cudaMemcpyHostToDevice);
		if ( SparkMinimaxH3ValCuda(SparkMinimaxH3DenseAttention(stream,q_in,k_in,
			v_in,seq,heads,head_dim,out),"dense_attention") == 0 )
		{
			device_values = SparkMinimaxH3ValReadBf16(out,slots);
			if ( device_values == 0 )
			{
				printf("attention readback FAIL\n");
				SparkMinimaxH3ValFailures++;
			}
			else
				SparkMinimaxH3ValCompare("dense_attention",device_values,attended,slots);
		}
		free(device_values);
		cudaFree(q_in);
		cudaFree(k_in);
		cudaFree(v_in);
		cudaFree(out);
		free(attended);
	}

	{
		float *affine = (float *)malloc(rows * 4u);
		float *gated = (float *)malloc(rows * 4u);
		float *scale_exact = (float *)malloc(hidden * 4u);
		float *shift_exact = (float *)malloc(hidden * 4u);
		float *gate_exact = (float *)malloc(hidden * 4u);
		uint16_t *scale_packed = (uint16_t *)malloc(hidden * 2u);
		uint16_t *shift_packed = (uint16_t *)malloc(hidden * 2u);
		uint16_t *gate_packed = (uint16_t *)malloc(hidden * 2u);
		float *device_values = 0;
		void *x_in = SparkMinimaxH3ValDevice(rows * 2u);
		void *res_in = SparkMinimaxH3ValDevice(rows * 2u);
		void *scale_in = SparkMinimaxH3ValDevice(hidden * 2u);
		void *shift_in = SparkMinimaxH3ValDevice(hidden * 2u);
		void *gate_in = SparkMinimaxH3ValDevice(hidden * 2u);
		void *out = SparkMinimaxH3ValDevice(rows * 2u);
		if ( affine == 0 || gated == 0 || scale_exact == 0 || shift_exact == 0 ||
			gate_exact == 0 || scale_packed == 0 || shift_packed == 0 ||
			gate_packed == 0 || x_in == 0 || res_in == 0 || scale_in == 0 ||
			shift_in == 0 || gate_in == 0 || out == 0 )
		{
			printf("allocation failure (adaln)\n");
			return(2);
		}
		SparkMinimaxH3ValFill(packed_a,exact_a,rows,1.0f,&rng);
		SparkMinimaxH3ValFill(packed_b,exact_b,rows,1.0f,&rng);
		SparkMinimaxH3ValFill(scale_packed,scale_exact,hidden,0.25f,&rng);
		SparkMinimaxH3ValFill(shift_packed,shift_exact,hidden,0.25f,&rng);
		SparkMinimaxH3ValFill(gate_packed,gate_exact,hidden,0.25f,&rng);
		SparkH3RefAdaLNAffine(exact_a,scale_exact,shift_exact,seq,hidden,affine);
		SparkH3RefGateResidual(affine,gate_exact,exact_b,seq,hidden,gated);
		cudaMemcpy(x_in,packed_a,rows * 2u,cudaMemcpyHostToDevice);
		cudaMemcpy(res_in,packed_b,rows * 2u,cudaMemcpyHostToDevice);
		cudaMemcpy(scale_in,scale_packed,hidden * 2u,cudaMemcpyHostToDevice);
		cudaMemcpy(shift_in,shift_packed,hidden * 2u,cudaMemcpyHostToDevice);
		cudaMemcpy(gate_in,gate_packed,hidden * 2u,cudaMemcpyHostToDevice);
		if ( SparkMinimaxH3ValCuda(SparkMinimaxH3AdaLNAffine(stream,x_in,scale_in,
			shift_in,seq,hidden,out),"adaln_affine") == 0 )
		{
			device_values = SparkMinimaxH3ValReadBf16(out,rows);
			if ( device_values == 0 || SparkMinimaxH3ValCompare("adaln_affine",
				device_values,affine,rows) != 0 )
			{
				if ( device_values == 0 )
				{
					printf("adaln readback FAIL\n");
					SparkMinimaxH3ValFailures++;
				}
			}
			free(device_values);
			if ( SparkMinimaxH3ValCuda(SparkMinimaxH3GateResidual(stream,out,
				gate_in,res_in,seq,hidden,out),"gate_residual") == 0 )
			{
				device_values = SparkMinimaxH3ValReadBf16(out,rows);
				if ( device_values == 0 )
				{
					printf("gate readback FAIL\n");
					SparkMinimaxH3ValFailures++;
				}
				else
					SparkMinimaxH3ValCompare("gate_residual",device_values,gated,rows);
			}
		}
		free(device_values);
		cudaFree(x_in);
		cudaFree(res_in);
		cudaFree(scale_in);
		cudaFree(shift_in);
		cudaFree(gate_in);
		cudaFree(out);
		free(affine);
		free(gated);
		free(scale_exact);
		free(shift_exact);
		free(gate_exact);
		free(scale_packed);
		free(shift_packed);
		free(gate_packed);
	}

	{
		float *activated = (float *)malloc((uint64_t)seq * ffn * 4u);
		float *gate_up_exact = (float *)malloc((uint64_t)seq * ffn * 2u * 4u);
		uint16_t *gate_up_packed = (uint16_t *)malloc((uint64_t)seq * ffn * 2u * 2u);
		float *device_values = 0;
		void *in = SparkMinimaxH3ValDevice((uint64_t)seq * ffn * 2u * 2u);
		void *out = SparkMinimaxH3ValDevice((uint64_t)seq * ffn * 2u);
		if ( activated == 0 || gate_up_exact == 0 || gate_up_packed == 0 ||
			in == 0 || out == 0 )
		{
			printf("allocation failure (silu_mul)\n");
			return(2);
		}
		SparkMinimaxH3ValFill(gate_up_packed,gate_up_exact,(uint64_t)seq * ffn * 2u,
			1.0f,&rng);
		SparkH3RefSiluMul(gate_up_exact,seq,ffn,activated);
		cudaMemcpy(in,gate_up_packed,(uint64_t)seq * ffn * 2u * 2u,
			cudaMemcpyHostToDevice);
		if ( SparkMinimaxH3ValCuda(SparkMinimaxH3SiluMul(stream,in,seq,ffn,out),
			"silu_mul") == 0 )
		{
			device_values = SparkMinimaxH3ValReadBf16(out,(uint64_t)seq * ffn);
			if ( device_values == 0 )
			{
				printf("silu_mul readback FAIL\n");
				SparkMinimaxH3ValFailures++;
			}
			else
				SparkMinimaxH3ValCompare("silu_mul",device_values,activated,
					(uint64_t)seq * ffn);
		}
		free(device_values);
		cudaFree(in);
		cudaFree(out);
		free(activated);
		free(gate_up_exact);
		free(gate_up_packed);
	}

	{
		float *conv_out = (float *)malloc(conv_count * 4u);
		float *weight_exact = (float *)malloc((uint64_t)conv_c * conv_c * 9u * 4u);
		float *bias_exact = (float *)malloc(conv_c * 4u);
		uint16_t *weight_packed = (uint16_t *)malloc((uint64_t)conv_c * conv_c * 9u * 2u);
		uint16_t *bias_packed = (uint16_t *)malloc(conv_c * 2u);
		float *device_values = 0;
		void *in = SparkMinimaxH3ValDevice(conv_count * 2u);
		void *weight = SparkMinimaxH3ValDevice((uint64_t)conv_c * conv_c * 9u * 2u);
		void *bias = SparkMinimaxH3ValDevice(conv_c * 2u);
		void *out = SparkMinimaxH3ValDevice(conv_count * 2u);
		if ( conv_out == 0 || weight_exact == 0 || bias_exact == 0 ||
			weight_packed == 0 || bias_packed == 0 || in == 0 || weight == 0 ||
			bias == 0 || out == 0 )
		{
			printf("allocation failure (conv2d)\n");
			return(2);
		}
		SparkMinimaxH3ValFill(packed_a,exact_a,conv_count,1.0f,&rng);
		SparkMinimaxH3ValFill(weight_packed,weight_exact,(uint64_t)conv_c * conv_c * 9u,
			0.25f,&rng);
		SparkMinimaxH3ValFill(bias_packed,bias_exact,conv_c,0.25f,&rng);
		SparkH3RefConv2d3x3Reflect(exact_a,weight_exact,bias_exact,conv_c,conv_c,
			conv_h,conv_w,conv_out);
		cudaMemcpy(in,packed_a,conv_count * 2u,cudaMemcpyHostToDevice);
		cudaMemcpy(weight,weight_packed,(uint64_t)conv_c * conv_c * 9u * 2u,
			cudaMemcpyHostToDevice);
		cudaMemcpy(bias,bias_packed,conv_c * 2u,cudaMemcpyHostToDevice);
		if ( SparkMinimaxH3ValCuda(SparkMinimaxH3Conv2d3x3Reflect(stream,in,weight,
			bias,conv_c,conv_c,conv_h,conv_w,out),"conv2d3x3_reflect") == 0 )
		{
			device_values = SparkMinimaxH3ValReadBf16(out,conv_count);
			if ( device_values == 0 )
			{
				printf("conv2d readback FAIL\n");
				SparkMinimaxH3ValFailures++;
			}
			else
				SparkMinimaxH3ValCompare("conv2d3x3_reflect",device_values,conv_out,
					conv_count);
		}
		free(device_values);
		cudaFree(in);
		cudaFree(weight);
		cudaFree(bias);
		cudaFree(out);
		free(conv_out);
		free(weight_exact);
		free(bias_exact);
		free(weight_packed);
		free(bias_packed);
	}

	{
		uint32_t dilation;
		for (dilation=1u; dilation<=3u; dilation+=2u)
		{
			char label[32];
			float *conv_out = (float *)malloc(conv1d_count * 4u);
			float *weight_exact = (float *)malloc(
				(uint64_t)SPARK_MINIMAX_H3_VAL_CONV1D_C * 3u * 4u);
			float *bias_exact = (float *)malloc(SPARK_MINIMAX_H3_VAL_CONV1D_C * 4u);
			uint16_t *weight_packed = (uint16_t *)malloc(
				(uint64_t)SPARK_MINIMAX_H3_VAL_CONV1D_C * 3u * 2u);
			uint16_t *bias_packed = (uint16_t *)malloc(
				SPARK_MINIMAX_H3_VAL_CONV1D_C * 2u);
			float *device_values = 0;
			void *in = SparkMinimaxH3ValDevice(conv1d_count * 2u);
			void *weight = SparkMinimaxH3ValDevice(
				(uint64_t)SPARK_MINIMAX_H3_VAL_CONV1D_C * 3u * 2u);
			void *bias = SparkMinimaxH3ValDevice(SPARK_MINIMAX_H3_VAL_CONV1D_C * 2u);
			void *out = SparkMinimaxH3ValDevice(conv1d_count * 2u);
			if ( conv_out == 0 || weight_exact == 0 || bias_exact == 0 ||
				weight_packed == 0 || bias_packed == 0 || in == 0 || weight == 0 ||
				bias == 0 || out == 0 )
			{
				printf("allocation failure (conv1d)\n");
				return(2);
			}
			SparkMinimaxH3ValFill(packed_a,exact_a,conv1d_count,1.0f,&rng);
			SparkMinimaxH3ValFill(weight_packed,weight_exact,
				(uint64_t)SPARK_MINIMAX_H3_VAL_CONV1D_C * 3u,0.25f,&rng);
			SparkMinimaxH3ValFill(bias_packed,bias_exact,
				SPARK_MINIMAX_H3_VAL_CONV1D_C,0.25f,&rng);
			SparkH3RefConv1dDilated(exact_a,weight_exact,bias_exact,
				SPARK_MINIMAX_H3_VAL_CONV1D_C,SPARK_MINIMAX_H3_VAL_CONV1D_L,
				dilation,conv_out);
			snprintf(label,sizeof(label),"conv1d_dil%u",dilation);
			cudaMemcpy(in,packed_a,conv1d_count * 2u,cudaMemcpyHostToDevice);
			cudaMemcpy(weight,weight_packed,
				(uint64_t)SPARK_MINIMAX_H3_VAL_CONV1D_C * 3u * 2u,
				cudaMemcpyHostToDevice);
			cudaMemcpy(bias,bias_packed,SPARK_MINIMAX_H3_VAL_CONV1D_C * 2u,
				cudaMemcpyHostToDevice);
			if ( SparkMinimaxH3ValCuda(SparkMinimaxH3Conv1dDilated(stream,in,weight,
				bias,SPARK_MINIMAX_H3_VAL_CONV1D_C,SPARK_MINIMAX_H3_VAL_CONV1D_L,
				dilation,out),label) == 0 )
			{
				device_values = SparkMinimaxH3ValReadBf16(out,conv1d_count);
				if ( device_values == 0 )
				{
					printf("conv1d readback FAIL\n");
					SparkMinimaxH3ValFailures++;
				}
				else
					SparkMinimaxH3ValCompare(label,device_values,conv_out,conv1d_count);
			}
			free(device_values);
			cudaFree(in);
			cudaFree(weight);
			cudaFree(bias);
			cudaFree(out);
			free(conv_out);
			free(weight_exact);
			free(bias_exact);
			free(weight_packed);
			free(bias_packed);
		}
	}

	{
		float *snaked = (float *)malloc(conv1d_count * 4u);
		float *alpha_exact = (float *)malloc(SPARK_MINIMAX_H3_VAL_CONV1D_C * 4u);
		uint16_t *alpha_packed = (uint16_t *)malloc(SPARK_MINIMAX_H3_VAL_CONV1D_C * 2u);
		float *device_values = 0;
		void *in = SparkMinimaxH3ValDevice(conv1d_count * 2u);
		void *alpha = SparkMinimaxH3ValDevice(SPARK_MINIMAX_H3_VAL_CONV1D_C * 2u);
		void *out = SparkMinimaxH3ValDevice(conv1d_count * 2u);
		if ( snaked == 0 || alpha_exact == 0 || alpha_packed == 0 || in == 0 ||
			alpha == 0 || out == 0 )
		{
			printf("allocation failure (snake)\n");
			return(2);
		}
		SparkMinimaxH3ValFill(packed_a,exact_a,conv1d_count,2.0f,&rng);
		SparkMinimaxH3ValFill(alpha_packed,alpha_exact,SPARK_MINIMAX_H3_VAL_CONV1D_C,
			1.0f,&rng);
		SparkH3RefSnake(exact_a,alpha_exact,SPARK_MINIMAX_H3_VAL_CONV1D_C,
			SPARK_MINIMAX_H3_VAL_CONV1D_L,snaked);
		cudaMemcpy(in,packed_a,conv1d_count * 2u,cudaMemcpyHostToDevice);
		cudaMemcpy(alpha,alpha_packed,SPARK_MINIMAX_H3_VAL_CONV1D_C * 2u,
			cudaMemcpyHostToDevice);
		if ( SparkMinimaxH3ValCuda(SparkMinimaxH3Snake(stream,in,alpha,
			SPARK_MINIMAX_H3_VAL_CONV1D_C,SPARK_MINIMAX_H3_VAL_CONV1D_L,out),
			"snake") == 0 )
		{
			device_values = SparkMinimaxH3ValReadBf16(out,conv1d_count);
			if ( device_values == 0 )
			{
				printf("snake readback FAIL\n");
				SparkMinimaxH3ValFailures++;
			}
			else
				SparkMinimaxH3ValCompare("snake",device_values,snaked,conv1d_count);
		}
		free(device_values);
		cudaFree(in);
		cudaFree(alpha);
		cudaFree(out);
		free(snaked);
		free(alpha_exact);
		free(alpha_packed);
	}

	{
		float oracle_sigmas[SPARK_H3_REF_SIGMA_POINTS];
		float module_sigmas[SPARK_H3_REF_SIGMA_POINTS];
		uint32_t oracle_count = 0u;
		uint32_t module_count = 0u;
		float shifts[2u];
		uint32_t shift_index;
		int32_t status;
		shifts[0] = SPARK_MINIMAX_H3_SCHEDULER_SHIFT;
		shifts[1] = SPARK_MINIMAX_H3_SCHEDULER_AUDIO_SHIFT;
		for (shift_index=0u; shift_index<2u; shift_index++)
		{
			SparkH3RefSchedulerBuildSigmas(shifts[shift_index],
				SPARK_H3_REF_SIGMA_POINTS,oracle_sigmas,&oracle_count);
			status = SparkMinimaxH3SchedulerBuildSigmas(shifts[shift_index],
				SPARK_H3_REF_SIGMA_POINTS,module_sigmas,&module_count);
			if ( status != 0 || module_count != oracle_count ||
				memcmp(module_sigmas,oracle_sigmas,
				(size_t)oracle_count * sizeof(float)) != 0 )
			{
				printf("scheduler shift=%.1f mismatch module=%u oracle=%u status=%d "
					"FAIL\n",(double)shifts[shift_index],module_count,oracle_count,
					(int)status);
				SparkMinimaxH3ValFailures++;
			}
			else
			{
				printf("scheduler shift=%.1f            bit-identical %u sigmas OK\n",
					(double)shifts[shift_index],oracle_count);
			}
		}
	}

	if ( cudaDeviceSynchronize() != cudaSuccess )
	{
		printf("device synchronize failed: %s FAIL\n",
			cudaGetErrorString(cudaGetLastError()));
		SparkMinimaxH3ValFailures++;
	}
	free(packed_a);
	free(packed_b);
	free(exact_a);
	free(exact_b);
	if ( SparkMinimaxH3ValFailures != 0u )
	{
		printf("V2 gate FAIL (%u failures)\n",SparkMinimaxH3ValFailures);
		return(1);
	}
	printf("V2 gate OK: all kernel comparisons within gate\n");
	return(0);
}
