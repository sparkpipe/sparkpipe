#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_minimax_h3_model.h"
#include "sparkpipe/spark_numerical_metrics.h"

extern "C" cudaError_t SparkMinimaxH3Rope3d(cudaStream_t stream, const void *input_bf16,
	const float *cos_angles, const float *sin_angles, uint32_t rows, uint32_t heads,
	uint32_t head_dim, uint32_t rope_dim, void *output_bf16);
extern "C" cudaError_t SparkMinimaxH3DenseAttention(cudaStream_t stream,
	const void *queries_bf16, const void *keys_bf16, const void *values_bf16,
	uint32_t seq, uint32_t heads, uint32_t head_dim, void *output_bf16);
extern "C" cudaError_t SparkMinimaxH3RmsNorm(cudaStream_t stream, const void *input_bf16,
	const void *weight_bf16, uint32_t rows, uint32_t width, float epsilon,
	void *output_bf16);
extern "C" cudaError_t SparkMinimaxH3Gemm(cudaStream_t stream, const void *activations_bf16,
	const void *weights_bf16, uint32_t rows, uint32_t width, uint32_t depth,
	void *segments_f32, void *output_bf16);
extern "C" cudaError_t SparkMinimaxH3GemmSegment(cudaStream_t stream,
	const void *activations_bf16, const void *weights_bf16, uint32_t rows,
	uint32_t width, uint32_t depth, uint32_t segment_index, float *partials_f32);
extern "C" cudaError_t SparkMinimaxH3GemmCombinePartials(cudaStream_t stream,
	const float *partials_f32, uint64_t count, void *output_bf16);
extern "C" cudaError_t SparkMinimaxH3AdaLNIndexed(cudaStream_t stream,
	const void *input_bf16, const void *scale_rows_bf16, const void *shift_rows_bf16,
	const uint32_t *row_of, uint32_t rows, uint32_t width, void *output_bf16);
extern "C" cudaError_t SparkMinimaxH3GateResidualIndexed(cudaStream_t stream,
	const void *value_bf16, const void *gate_rows_bf16, const void *residual_bf16,
	const uint32_t *row_of, uint32_t rows, uint32_t width, void *output_bf16);
extern "C" cudaError_t SparkMinimaxH3SiluMul(cudaStream_t stream,
	const void *gate_up_bf16, uint32_t rows, uint32_t ffn, void *output_bf16);

#define SPARK_MINIMAX_H3_V3_SEQ 13u
#define SPARK_MINIMAX_H3_V3_HIDDEN 5376u
#define SPARK_MINIMAX_H3_V3_HEADS 56u
#define SPARK_MINIMAX_H3_V3_HEAD_DIM 128u
#define SPARK_MINIMAX_H3_V3_QKV 7168u
#define SPARK_MINIMAX_H3_V3_FFN 14336u
#define SPARK_MINIMAX_H3_V3_FFN_FUSED 28672u
#define SPARK_MINIMAX_H3_V3_ROPE 96u
#define SPARK_MINIMAX_H3_V3_MOD_ROWS 6u
#define SPARK_MINIMAX_H3_V3_MAX_REL 1e-2
#define SPARK_MINIMAX_H3_V3_MIN_COSINE 0.9999

static uint32_t SparkMinimaxH3V3Failures;

static void *SparkMinimaxH3V3ReadFile(const char *directory, const char *name,
	uint64_t bytes)
{
	char path[1024];
	FILE *file;
	void *buffer = malloc((size_t)bytes);
	if ( buffer == 0 )
	{
		printf("host alloc failure for %s\n",name);
		exit(2);
	}
	snprintf(path,sizeof(path),"%s/%s",directory,name);
	file = fopen(path,"rb");
	if ( file == 0 )
	{
		printf("missing fixture file %s\n",path);
		exit(2);
	}
	if ( fread(buffer,1,(size_t)bytes,file) != bytes )
	{
		printf("short read on %s\n",path);
		exit(2);
	}
	fclose(file);
	return(buffer);
}

static void *SparkMinimaxH3V3DeviceUpload(const void *host, uint64_t bytes)
{
	void *device = 0;
	if ( cudaMalloc(&device,(size_t)bytes) != cudaSuccess )
	{
		printf("cudaMalloc failure\n");
		exit(2);
	}
	if ( cudaMemcpy(device,host,(size_t)bytes,cudaMemcpyHostToDevice) != cudaSuccess )
	{
		printf("cudaMemcpy failure\n");
		exit(2);
	}
	return(device);
}

static void SparkMinimaxH3V3Compare(const char *check, const float *actual,
	const float *reference, uint64_t count)
{
	SparkNumericalMetrics metrics = SparkNumericalMeasureF32(actual,reference,count);
	uint32_t within = SparkNumericalMetricsWithin(&metrics,
		SPARK_MINIMAX_H3_V3_MAX_REL,SPARK_MINIMAX_H3_V3_MIN_COSINE);
	printf("%-16s rel_l2=%.6f cosine=%.8f max_abs=%.6f %s\n",check,
		metrics.relative_l2,metrics.cosine,metrics.max_absolute,
		within != 0u ? "OK" : "FAIL");
	if ( within == 0u )
		SparkMinimaxH3V3Failures++;
}

static void *SparkMinimaxH3V3ReadBf16(const void *device, uint64_t count)
{
	uint16_t *packed = (uint16_t *)malloc(count * 2u);
	float *host = (float *)malloc(count * 4u);
	uint64_t index;
	if ( packed == 0 || host == 0 )
		exit(2);
	if ( cudaMemcpy(packed,device,count * 2u,cudaMemcpyDeviceToHost) != cudaSuccess )
		exit(2);
	for (index=0u; index<count; index++)
	{
		uint32_t bits = (uint32_t)packed[index] << 16u;
		memcpy(&host[index],&bits,sizeof(bits));
	}
	free(packed);
	return(host);
}

struct SparkMinimaxH3V3Weights
{
	void *query,*key,*value,*output_proj,*norm_q,*norm_k,*gate_up,*down,*norm1,*norm2;
};

static void SparkMinimaxH3V3LoadWeight(void **device, const char *weights_directory,
	const char *manifest, const char *name, uint32_t rows, uint32_t columns)
{
	char line[512];
	char entry_name[256];
	char file[256];
	uint64_t entry_rows,entry_columns;
	int32_t found = 0;
	FILE *fp = fopen(manifest,"r");
	if ( fp == 0 )
	{
		printf("missing weight manifest %s\n",manifest);
		exit(2);
	}
	while ( fgets(line,sizeof(line),fp) != 0 )
	{
		if ( sscanf(line,"%255s %llu %llu %255s",entry_name,
			(unsigned long long *)&entry_rows,(unsigned long long *)&entry_columns,
			file) == 4 && strcmp(entry_name,name) == 0 )
		{
			found = 1;
			break;
		}
	}
	fclose(fp);
	if ( found == 0 || entry_rows != rows || entry_columns != columns )
	{
		printf("weight %s missing or misshaped in manifest (%u,%u)\n",name,rows,columns);
		exit(2);
	}
	{
		void *host = SparkMinimaxH3V3ReadFile(weights_directory,file,
			(uint64_t)rows * columns * 2u);
		uint64_t element,nonfinite = 0;
		float maximum = 0.0f;
		uint16_t *packed = (uint16_t *)host;
		for (element=0u; element<(uint64_t)rows * columns; element++)
		{
			uint32_t bits = (uint32_t)packed[element] << 16u;
			float value;
			memcpy(&value,&bits,sizeof(value));
			if ( !isfinite(value) )
				nonfinite++;
			else if ( fabsf(value) > maximum )
				maximum = fabsf(value);
		}
		printf("weight %-52s rows=%u cols=%u max=%.6g nonfinite=%llu\n",name,
			rows,columns,(double)maximum,(unsigned long long)nonfinite);
		*device = SparkMinimaxH3V3DeviceUpload(host,(uint64_t)rows * columns * 2u);
		free(host);
	}
}

static void SparkMinimaxH3V3LoadBlock(struct SparkMinimaxH3V3Weights *weights,
	const char *weights_directory, const char *manifest, uint32_t block)
{
	char name[256];
	#define SPARK_MINIMAX_H3_V3_LOAD(field,label,rows,columns) \
		snprintf(name,sizeof(name),"transformer_blocks.%u.%s",block,label); \
		SparkMinimaxH3V3LoadWeight(&weights->field,weights_directory,manifest,name, \
			rows,columns)
	SPARK_MINIMAX_H3_V3_LOAD(query,"attn.to_q.weight",
		SPARK_MINIMAX_H3_V3_QKV,SPARK_MINIMAX_H3_V3_HIDDEN);
	SPARK_MINIMAX_H3_V3_LOAD(key,"attn.to_k.weight",
		SPARK_MINIMAX_H3_V3_QKV,SPARK_MINIMAX_H3_V3_HIDDEN);
	SPARK_MINIMAX_H3_V3_LOAD(value,"attn.to_v.weight",
		SPARK_MINIMAX_H3_V3_QKV,SPARK_MINIMAX_H3_V3_HIDDEN);
	SPARK_MINIMAX_H3_V3_LOAD(output_proj,"attn.to_out.0.weight",
		SPARK_MINIMAX_H3_V3_HIDDEN,SPARK_MINIMAX_H3_V3_QKV);
	SPARK_MINIMAX_H3_V3_LOAD(norm_q,"attn.norm_q.weight",1u,
		SPARK_MINIMAX_H3_V3_HEAD_DIM);
	SPARK_MINIMAX_H3_V3_LOAD(norm_k,"attn.norm_k.weight",1u,
		SPARK_MINIMAX_H3_V3_HEAD_DIM);
	SPARK_MINIMAX_H3_V3_LOAD(gate_up,"ff.net.0.proj.weight",
		SPARK_MINIMAX_H3_V3_FFN_FUSED,SPARK_MINIMAX_H3_V3_HIDDEN);
	SPARK_MINIMAX_H3_V3_LOAD(down,"ff.net.2.weight",
		SPARK_MINIMAX_H3_V3_HIDDEN,SPARK_MINIMAX_H3_V3_FFN);
	SPARK_MINIMAX_H3_V3_LOAD(norm1,"norm1.weight",1u,SPARK_MINIMAX_H3_V3_HIDDEN);
	SPARK_MINIMAX_H3_V3_LOAD(norm2,"norm2.weight",1u,SPARK_MINIMAX_H3_V3_HIDDEN);
	#undef SPARK_MINIMAX_H3_V3_LOAD
}

struct SparkMinimaxH3V3Scratch
{
	void *normed,*adaln,*q,*k,*v,*q_rope,*k_rope,*attn_raw,*attn_out,
		*ffn_fused,*ffn_mid,*ffn_out,*segments;
};

static cudaError_t SparkMinimaxH3V3Gemm(cudaStream_t stream, const void *a,
	const void *w, uint32_t rows, uint32_t width, uint32_t depth, void *segments,
	void *out)
{
	return(SparkMinimaxH3Gemm(stream,a,w,rows,width,depth,segments,out));
}

static cudaError_t SparkMinimaxH3V3BlockForward(cudaStream_t stream,
	const struct SparkMinimaxH3V3Weights *weights, const void *input_bf16,
	const void *scale_msa, const void *shift_msa, const void *gate_msa,
	const void *scale_mlp, const void *shift_mlp, const void *gate_mlp,
	const float *rope_cos, const float *rope_sin, const uint32_t *row_of,
	struct SparkMinimaxH3V3Scratch *scratch, void *result_bf16)
{
	const char *stage = "start";
	cudaError_t error;
	stage = "call SparkMinimaxH3RmsNorm #1";
	error = SparkMinimaxH3RmsNorm(stream,input_bf16,weights->norm1,
		SPARK_MINIMAX_H3_V3_SEQ,SPARK_MINIMAX_H3_V3_HIDDEN,
		SPARK_MINIMAX_H3_DIT_NORM_EPSILON,scratch->normed);
	if ( error != cudaSuccess )
	{
		printf("kernel failure at %s: %s\n",stage,cudaGetErrorString(error));
		return(error);
	}
	stage = "call SparkMinimaxH3AdaLNIndexed #2";
	error = SparkMinimaxH3AdaLNIndexed(stream,scratch->normed,scale_msa,shift_msa,
		row_of,SPARK_MINIMAX_H3_V3_SEQ,SPARK_MINIMAX_H3_V3_HIDDEN,scratch->adaln);
	if ( error != cudaSuccess )
	{
		printf("kernel failure at %s: %s\n",stage,cudaGetErrorString(error));
		return(error);
	}
	stage = "call SparkMinimaxH3Gemm #3";
	error = SparkMinimaxH3Gemm(stream,scratch->adaln,weights->query,
		SPARK_MINIMAX_H3_V3_SEQ,SPARK_MINIMAX_H3_V3_QKV,
		SPARK_MINIMAX_H3_V3_HIDDEN,scratch->segments,scratch->q);
	if ( error != cudaSuccess )
	{
		printf("kernel failure at %s: %s\n",stage,cudaGetErrorString(error));
		return(error);
	}
	stage = "call SparkMinimaxH3RmsNorm #4";
	error = SparkMinimaxH3RmsNorm(stream,scratch->q,weights->norm_q,
		SPARK_MINIMAX_H3_V3_SEQ * SPARK_MINIMAX_H3_V3_HEADS,
		SPARK_MINIMAX_H3_V3_HEAD_DIM,SPARK_MINIMAX_H3_DIT_QK_NORM_EPSILON,
		scratch->q);
	if ( error != cudaSuccess )
	{
		printf("kernel failure at %s: %s\n",stage,cudaGetErrorString(error));
		return(error);
	}
	stage = "call SparkMinimaxH3Rope3d #5";
	error = SparkMinimaxH3Rope3d(stream,scratch->q,rope_cos,rope_sin,
		SPARK_MINIMAX_H3_V3_SEQ,SPARK_MINIMAX_H3_V3_HEADS,
		SPARK_MINIMAX_H3_V3_HEAD_DIM,SPARK_MINIMAX_H3_V3_ROPE,scratch->q_rope);
	if ( error != cudaSuccess )
	{
		printf("kernel failure at %s: %s\n",stage,cudaGetErrorString(error));
		return(error);
	}
	stage = "call SparkMinimaxH3Gemm #6";
	error = SparkMinimaxH3Gemm(stream,scratch->adaln,weights->key,
		SPARK_MINIMAX_H3_V3_SEQ,SPARK_MINIMAX_H3_V3_QKV,
		SPARK_MINIMAX_H3_V3_HIDDEN,scratch->segments,scratch->k);
	if ( error != cudaSuccess )
	{
		printf("kernel failure at %s: %s\n",stage,cudaGetErrorString(error));
		return(error);
	}
	stage = "call SparkMinimaxH3RmsNorm #7";
	error = SparkMinimaxH3RmsNorm(stream,scratch->k,weights->norm_k,
		SPARK_MINIMAX_H3_V3_SEQ * SPARK_MINIMAX_H3_V3_HEADS,
		SPARK_MINIMAX_H3_V3_HEAD_DIM,SPARK_MINIMAX_H3_DIT_QK_NORM_EPSILON,
		scratch->k);
	if ( error != cudaSuccess )
	{
		printf("kernel failure at %s: %s\n",stage,cudaGetErrorString(error));
		return(error);
	}
	stage = "call SparkMinimaxH3Rope3d #8";
	error = SparkMinimaxH3Rope3d(stream,scratch->k,rope_cos,rope_sin,
		SPARK_MINIMAX_H3_V3_SEQ,SPARK_MINIMAX_H3_V3_HEADS,
		SPARK_MINIMAX_H3_V3_HEAD_DIM,SPARK_MINIMAX_H3_V3_ROPE,scratch->k_rope);
	if ( error != cudaSuccess )
	{
		printf("kernel failure at %s: %s\n",stage,cudaGetErrorString(error));
		return(error);
	}
	stage = "call SparkMinimaxH3Gemm #9";
	error = SparkMinimaxH3Gemm(stream,scratch->adaln,weights->value,
		SPARK_MINIMAX_H3_V3_SEQ,SPARK_MINIMAX_H3_V3_QKV,
		SPARK_MINIMAX_H3_V3_HIDDEN,scratch->segments,scratch->v);
	if ( error != cudaSuccess )
	{
		printf("kernel failure at %s: %s\n",stage,cudaGetErrorString(error));
		return(error);
	}
	stage = "call SparkMinimaxH3DenseAttention #10";
	error = SparkMinimaxH3DenseAttention(stream,scratch->q_rope,scratch->k_rope,
		scratch->v,SPARK_MINIMAX_H3_V3_SEQ,SPARK_MINIMAX_H3_V3_HEADS,
		SPARK_MINIMAX_H3_V3_HEAD_DIM,scratch->attn_raw);
	if ( error != cudaSuccess )
	{
		printf("kernel failure at %s: %s\n",stage,cudaGetErrorString(error));
		return(error);
	}
	stage = "call SparkMinimaxH3Gemm #11";
	error = SparkMinimaxH3Gemm(stream,scratch->attn_raw,weights->output_proj,
		SPARK_MINIMAX_H3_V3_SEQ,SPARK_MINIMAX_H3_V3_HIDDEN,
		SPARK_MINIMAX_H3_V3_QKV,scratch->segments,scratch->attn_out);
	if ( error != cudaSuccess )
	{
		printf("kernel failure at %s: %s\n",stage,cudaGetErrorString(error));
		return(error);
	}
	stage = "call SparkMinimaxH3GateResidualIndexed #12";
	error = SparkMinimaxH3GateResidualIndexed(stream,scratch->attn_out,gate_msa,
		input_bf16,row_of,SPARK_MINIMAX_H3_V3_SEQ,SPARK_MINIMAX_H3_V3_HIDDEN,
		scratch->normed);
	if ( error != cudaSuccess )
	{
		printf("kernel failure at %s: %s\n",stage,cudaGetErrorString(error));
		return(error);
	}
	stage = "call SparkMinimaxH3RmsNorm #13";
	error = SparkMinimaxH3RmsNorm(stream,scratch->normed,weights->norm2,
		SPARK_MINIMAX_H3_V3_SEQ,SPARK_MINIMAX_H3_V3_HIDDEN,
		SPARK_MINIMAX_H3_DIT_NORM_EPSILON,scratch->ffn_out);
	if ( error != cudaSuccess )
	{
		printf("kernel failure at %s: %s\n",stage,cudaGetErrorString(error));
		return(error);
	}
	stage = "call SparkMinimaxH3AdaLNIndexed #14";
	error = SparkMinimaxH3AdaLNIndexed(stream,scratch->ffn_out,scale_mlp,shift_mlp,
		row_of,SPARK_MINIMAX_H3_V3_SEQ,SPARK_MINIMAX_H3_V3_HIDDEN,scratch->ffn_out);
	if ( error != cudaSuccess )
	{
		printf("kernel failure at %s: %s\n",stage,cudaGetErrorString(error));
		return(error);
	}
	stage = "call SparkMinimaxH3Gemm #15";
	error = SparkMinimaxH3Gemm(stream,scratch->ffn_out,weights->gate_up,
		SPARK_MINIMAX_H3_V3_SEQ,SPARK_MINIMAX_H3_V3_FFN_FUSED,
		SPARK_MINIMAX_H3_V3_HIDDEN,scratch->segments,scratch->ffn_fused);
	if ( error != cudaSuccess )
	{
		printf("kernel failure at %s: %s\n",stage,cudaGetErrorString(error));
		return(error);
	}
	stage = "call SparkMinimaxH3SiluMul #16";
	error = SparkMinimaxH3SiluMul(stream,scratch->ffn_fused,
		SPARK_MINIMAX_H3_V3_SEQ,SPARK_MINIMAX_H3_V3_FFN,scratch->ffn_mid);
	if ( error != cudaSuccess )
	{
		printf("kernel failure at %s: %s\n",stage,cudaGetErrorString(error));
		return(error);
	}
	stage = "call SparkMinimaxH3Gemm #17";
	error = SparkMinimaxH3Gemm(stream,scratch->ffn_mid,weights->down,
		SPARK_MINIMAX_H3_V3_SEQ,SPARK_MINIMAX_H3_V3_HIDDEN,
		SPARK_MINIMAX_H3_V3_FFN,scratch->segments,scratch->ffn_out);
	if ( error != cudaSuccess )
	{
		printf("kernel failure at %s: %s\n",stage,cudaGetErrorString(error));
		return(error);
	}
	return(SparkMinimaxH3GateResidualIndexed(stream,scratch->ffn_out,gate_mlp,
		input_bf16,row_of,SPARK_MINIMAX_H3_V3_SEQ,SPARK_MINIMAX_H3_V3_HIDDEN,
		result_bf16));
}

int main(int argc, char **argv)
{
	uint64_t rows_bytes = (uint64_t)SPARK_MINIMAX_H3_V3_SEQ * SPARK_MINIMAX_H3_V3_HIDDEN;
	uint32_t row_of[SPARK_MINIMAX_H3_V3_SEQ];
	int64_t *timestep_host;
	int64_t *tags_host;
	uint32_t index;
	cudaStream_t stream = 0;
	if ( argc != 4 )
	{
		printf("usage: %s FIXTURE_DIR WEIGHTS_DIR WEIGHT_MANIFEST\n",argv[0]);
		return(2);
	}
	timestep_host = (int64_t *)SparkMinimaxH3V3ReadFile(argv[1],
		"timestep_indices__13.i64",SPARK_MINIMAX_H3_V3_SEQ * 8u);
	tags_host = (int64_t *)SparkMinimaxH3V3ReadFile(argv[1],
		"token_tags__13.i64",SPARK_MINIMAX_H3_V3_SEQ * 8u);
	for (index=0u; index<SPARK_MINIMAX_H3_V3_SEQ; index++)
		row_of[index] = (uint32_t)(timestep_host[index] * 3 + tags_host[index]);
	free(timestep_host);
	free(tags_host);
	{
		uint16_t *packed_h = (uint16_t *)SparkMinimaxH3V3ReadFile(argv[1],
			"packed_h__1x13x5376.u16",rows_bytes * 2u);
		uint16_t *mods_scale_msa = (uint16_t *)SparkMinimaxH3V3ReadFile(argv[1],
			"mod_scale_msa__6x5376.u16",(uint64_t)SPARK_MINIMAX_H3_V3_MOD_ROWS *
			SPARK_MINIMAX_H3_V3_HIDDEN * 2u);
		uint16_t *mods_shift_msa = (uint16_t *)SparkMinimaxH3V3ReadFile(argv[1],
			"mod_shift_msa__6x5376.u16",(uint64_t)SPARK_MINIMAX_H3_V3_MOD_ROWS *
			SPARK_MINIMAX_H3_V3_HIDDEN * 2u);
		uint16_t *mods_gate_msa = (uint16_t *)SparkMinimaxH3V3ReadFile(argv[1],
			"mod_gate_msa__6x5376.u16",(uint64_t)SPARK_MINIMAX_H3_V3_MOD_ROWS *
			SPARK_MINIMAX_H3_V3_HIDDEN * 2u);
		uint16_t *mods_scale_mlp = (uint16_t *)SparkMinimaxH3V3ReadFile(argv[1],
			"mod_scale_mlp__6x5376.u16",(uint64_t)SPARK_MINIMAX_H3_V3_MOD_ROWS *
			SPARK_MINIMAX_H3_V3_HIDDEN * 2u);
		uint16_t *mods_shift_mlp = (uint16_t *)SparkMinimaxH3V3ReadFile(argv[1],
			"mod_shift_mlp__6x5376.u16",(uint64_t)SPARK_MINIMAX_H3_V3_MOD_ROWS *
			SPARK_MINIMAX_H3_V3_HIDDEN * 2u);
		uint16_t *mods_gate_mlp = (uint16_t *)SparkMinimaxH3V3ReadFile(argv[1],
			"mod_gate_mlp__6x5376.u16",(uint64_t)SPARK_MINIMAX_H3_V3_MOD_ROWS *
			SPARK_MINIMAX_H3_V3_HIDDEN * 2u);
		float *rope_cos = (float *)SparkMinimaxH3V3ReadFile(argv[1],
			"rope_cos__13x96.f32",(uint64_t)SPARK_MINIMAX_H3_V3_SEQ *
			SPARK_MINIMAX_H3_V3_ROPE * 4u);
		float *rope_sin = (float *)SparkMinimaxH3V3ReadFile(argv[1],
			"rope_sin__13x96.f32",(uint64_t)SPARK_MINIMAX_H3_V3_SEQ *
			SPARK_MINIMAX_H3_V3_ROPE * 4u);
		uint16_t *block0_expected = (uint16_t *)SparkMinimaxH3V3ReadFile(argv[1],
			"block0_out__1x13x5376.u16",rows_bytes * 2u);
		uint16_t *block1_expected = (uint16_t *)SparkMinimaxH3V3ReadFile(argv[1],
			"block1_out__1x13x5376.u16",rows_bytes * 2u);
		uint16_t *first_run = 0;
		float *block0_reference = (float *)malloc(rows_bytes * 4u);
		float *block1_reference = (float *)malloc(rows_bytes * 4u);
		uint16_t *second_run;
		struct SparkMinimaxH3V3Weights block0,block1;
		void *device_h = SparkMinimaxH3V3DeviceUpload(packed_h,rows_bytes * 2u);
		void *device_h_saved = 0;
		void *device_result = 0;
		void *device_partials = 0;
		void *device_tp1 = 0;
		void *device_tp4 = 0;
		float *device_cos = (float *)SparkMinimaxH3V3DeviceUpload(rope_cos,
			(uint64_t)SPARK_MINIMAX_H3_V3_SEQ * SPARK_MINIMAX_H3_V3_ROPE * 4u);
		float *device_sin = (float *)SparkMinimaxH3V3DeviceUpload(rope_sin,
			(uint64_t)SPARK_MINIMAX_H3_V3_SEQ * SPARK_MINIMAX_H3_V3_ROPE * 4u);
		uint32_t *device_row_of = (uint32_t *)SparkMinimaxH3V3DeviceUpload(row_of,
			sizeof(row_of));
		void *device_scale_msa = SparkMinimaxH3V3DeviceUpload(mods_scale_msa,
			(uint64_t)SPARK_MINIMAX_H3_V3_MOD_ROWS * SPARK_MINIMAX_H3_V3_HIDDEN * 2u);
		void *device_shift_msa = SparkMinimaxH3V3DeviceUpload(mods_shift_msa,
			(uint64_t)SPARK_MINIMAX_H3_V3_MOD_ROWS * SPARK_MINIMAX_H3_V3_HIDDEN * 2u);
		void *device_gate_msa = SparkMinimaxH3V3DeviceUpload(mods_gate_msa,
			(uint64_t)SPARK_MINIMAX_H3_V3_MOD_ROWS * SPARK_MINIMAX_H3_V3_HIDDEN * 2u);
		void *device_scale_mlp = SparkMinimaxH3V3DeviceUpload(mods_scale_mlp,
			(uint64_t)SPARK_MINIMAX_H3_V3_MOD_ROWS * SPARK_MINIMAX_H3_V3_HIDDEN * 2u);
		void *device_shift_mlp = SparkMinimaxH3V3DeviceUpload(mods_shift_mlp,
			(uint64_t)SPARK_MINIMAX_H3_V3_MOD_ROWS * SPARK_MINIMAX_H3_V3_HIDDEN * 2u);
		void *device_gate_mlp = SparkMinimaxH3V3DeviceUpload(mods_gate_mlp,
			(uint64_t)SPARK_MINIMAX_H3_V3_MOD_ROWS * SPARK_MINIMAX_H3_V3_HIDDEN * 2u);
		cudaError_t error;
		uint64_t count;
		printf("minimax_h3 V3 gate: real-weight DiT blocks 0 and 1 vs anchor fixture\n");
		for (index=0u; index<rows_bytes; index++)
		{
			uint32_t bits0 = (uint32_t)block0_expected[index] << 16u;
			uint32_t bits1 = (uint32_t)block1_expected[index] << 16u;
			memcpy(&block0_reference[index],&bits0,sizeof(bits0));
			memcpy(&block1_reference[index],&bits1,sizeof(bits1));
		}
		struct SparkMinimaxH3V3Scratch scratch = {0};
		cudaMalloc(&scratch.normed,rows_bytes * 2u);
		cudaMalloc(&scratch.adaln,rows_bytes * 2u);
		cudaMalloc(&scratch.q,(uint64_t)SPARK_MINIMAX_H3_V3_SEQ *
			SPARK_MINIMAX_H3_V3_QKV * 2u);
		cudaMalloc(&scratch.k,(uint64_t)SPARK_MINIMAX_H3_V3_SEQ *
			SPARK_MINIMAX_H3_V3_QKV * 2u);
		cudaMalloc(&scratch.v,(uint64_t)SPARK_MINIMAX_H3_V3_SEQ *
			SPARK_MINIMAX_H3_V3_QKV * 2u);
		cudaMalloc(&scratch.q_rope,(uint64_t)SPARK_MINIMAX_H3_V3_SEQ *
			SPARK_MINIMAX_H3_V3_QKV * 2u);
		cudaMalloc(&scratch.k_rope,(uint64_t)SPARK_MINIMAX_H3_V3_SEQ *
			SPARK_MINIMAX_H3_V3_QKV * 2u);
		cudaMalloc(&scratch.attn_raw,(uint64_t)SPARK_MINIMAX_H3_V3_SEQ *
			SPARK_MINIMAX_H3_V3_QKV * 2u);
		cudaMalloc(&scratch.attn_out,rows_bytes * 2u);
		cudaMalloc(&scratch.ffn_fused,(uint64_t)SPARK_MINIMAX_H3_V3_SEQ *
			SPARK_MINIMAX_H3_V3_FFN_FUSED * 2u);
		cudaMalloc(&scratch.ffn_mid,(uint64_t)SPARK_MINIMAX_H3_V3_SEQ *
			SPARK_MINIMAX_H3_V3_FFN * 2u);
		cudaMalloc(&scratch.ffn_out,rows_bytes * 2u);
		cudaMalloc(&device_h_saved,rows_bytes * 2u);
		cudaMalloc(&device_result,rows_bytes * 2u);
		cudaMalloc(&scratch.segments,(uint64_t)SPARK_MINIMAX_H3_V3_SEQ *
			SPARK_MINIMAX_H3_V3_FFN_FUSED * 4u * 4u);
		cudaMemcpy(device_h_saved,device_h,rows_bytes * 2u,cudaMemcpyDeviceToDevice);
		SparkMinimaxH3V3LoadBlock(&block0,argv[2],argv[3],0u);
		SparkMinimaxH3V3LoadBlock(&block1,argv[2],argv[3],1u);
		error = SparkMinimaxH3V3BlockForward(stream,&block0,device_h,
			device_scale_msa,device_shift_msa,device_gate_msa,device_scale_mlp,
			device_shift_mlp,device_gate_mlp,device_cos,device_sin,device_row_of,
			&scratch,device_result);
		if ( error != cudaSuccess )
		{
			printf("block0 forward cuda error: %s FAIL\n",cudaGetErrorString(error));
			return(1);
		}
		{
			float *actual = (float *)SparkMinimaxH3V3ReadBf16(device_result,rows_bytes);
			SparkMinimaxH3V3Compare("block0",actual,block0_reference,rows_bytes);
			free(actual);
		}
		first_run = (uint16_t *)malloc(rows_bytes * 2u);
		cudaMemcpy(first_run,device_result,rows_bytes * 2u,cudaMemcpyDeviceToHost);
		error = SparkMinimaxH3V3BlockForward(stream,&block1,device_result,
			device_scale_msa,device_shift_msa,device_gate_msa,device_scale_mlp,
			device_shift_mlp,device_gate_mlp,device_cos,device_sin,device_row_of,
			&scratch,device_result);
		if ( error != cudaSuccess )
		{
			printf("block1 forward cuda error: %s FAIL\n",cudaGetErrorString(error));
			return(1);
		}
		{
			float *actual = (float *)SparkMinimaxH3V3ReadBf16(device_result,rows_bytes);
			SparkMinimaxH3V3Compare("block1",actual,block1_reference,rows_bytes);
			free(actual);
		}
		cudaMemcpy(device_h,device_h_saved,rows_bytes * 2u,cudaMemcpyDeviceToDevice);
		error = SparkMinimaxH3V3BlockForward(stream,&block0,device_h,
			device_scale_msa,device_shift_msa,device_gate_msa,device_scale_mlp,
			device_shift_mlp,device_gate_mlp,device_cos,device_sin,device_row_of,
			&scratch,device_result);
		if ( error != cudaSuccess )
		{
			printf("block0 rerun cuda error: %s FAIL\n",cudaGetErrorString(error));
			return(1);
		}
		second_run = (uint16_t *)malloc(rows_bytes * 2u);
		cudaMemcpy(second_run,device_result,rows_bytes * 2u,cudaMemcpyDeviceToHost);
		printf("%-16s block0 rerun bit-identical %s\n","determinism",
			memcmp(first_run,second_run,rows_bytes * 2u) == 0 ? "OK" : "FAIL");
		if ( memcmp(first_run,second_run,rows_bytes * 2u) != 0 )
			SparkMinimaxH3V3Failures++;
		free(first_run);
		free(second_run);
		count = (uint64_t)SPARK_MINIMAX_H3_V3_SEQ * SPARK_MINIMAX_H3_V3_HIDDEN;
		cudaMalloc(&device_partials,count * 4u * 4u);
		cudaMalloc(&device_tp1,rows_bytes * 2u);
		cudaMalloc(&device_tp4,rows_bytes * 2u);
		{
			const void *ffn_input = scratch.ffn_mid;
			cudaMemset(scratch.ffn_mid,0,(uint64_t)SPARK_MINIMAX_H3_V3_SEQ *
				SPARK_MINIMAX_H3_V3_FFN * 2u);
			cudaMemcpy(scratch.ffn_mid,device_h,rows_bytes * 2u,
				cudaMemcpyDeviceToDevice);
			for (index=0u; index<4u; index++)
				SparkMinimaxH3GemmSegment(stream,ffn_input,block1.down,
					SPARK_MINIMAX_H3_V3_SEQ,SPARK_MINIMAX_H3_V3_HIDDEN,
					SPARK_MINIMAX_H3_V3_FFN,index,(float *)device_partials);
			SparkMinimaxH3GemmCombinePartials(stream,(const float *)device_partials,
				count,device_tp4);
			SparkMinimaxH3Gemm(stream,ffn_input,block1.down,
				SPARK_MINIMAX_H3_V3_SEQ,SPARK_MINIMAX_H3_V3_HIDDEN,
				SPARK_MINIMAX_H3_V3_FFN,scratch.segments,device_tp1);
		}
		{
			uint16_t *tp1 = (uint16_t *)malloc(rows_bytes * 2u);
			uint16_t *tp4 = (uint16_t *)malloc(rows_bytes * 2u);
			cudaMemcpy(tp1,device_tp1,rows_bytes * 2u,cudaMemcpyDeviceToHost);
			cudaMemcpy(tp4,device_tp4,rows_bytes * 2u,cudaMemcpyDeviceToHost);
			printf("%-16s down-proj TP4 partials vs TP1 %s\n","tp4_vs_tp1",
				memcmp(tp1,tp4,rows_bytes * 2u) == 0 ? "OK" : "FAIL");
			if ( memcmp(tp1,tp4,rows_bytes * 2u) != 0 )
				SparkMinimaxH3V3Failures++;
			free(tp1);
			free(tp4);
		}
		if ( cudaDeviceSynchronize() != cudaSuccess )
		{
			printf("device synchronize failed FAIL\n");
			SparkMinimaxH3V3Failures++;
		}
		if ( SparkMinimaxH3V3Failures != 0u )
		{
			printf("V3 gate FAIL (%u failures)\n",SparkMinimaxH3V3Failures);
			return(1);
		}
		printf("V3 gate OK: real-weight blocks within rel<=%g, deterministic rerun, "
			"TP4-vs-TP1 bit-identical\n",(double)SPARK_MINIMAX_H3_V3_MAX_REL);
	}
	return(0);
}
