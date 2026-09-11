#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_minimax_h3_model.h"
#include "sparkpipe/spark_minimax_h3_scheduler.h"

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

#define SPARK_MINIMAX_H3_V5_SEQ 13u
#define SPARK_MINIMAX_H3_V5_HIDDEN 5376u
#define SPARK_MINIMAX_H3_V5_HEADS 56u
#define SPARK_MINIMAX_H3_V5_HEAD_DIM 128u
#define SPARK_MINIMAX_H3_V5_QKV 7168u
#define SPARK_MINIMAX_H3_V5_FFN 14336u
#define SPARK_MINIMAX_H3_V5_FFN_FUSED 28672u
#define SPARK_MINIMAX_H3_V5_ROPE 96u
#define SPARK_MINIMAX_H3_V5_MOD_ROWS 6u
#define SPARK_MINIMAX_H3_V5_STEPS 2u
#define SPARK_MINIMAX_H3_V5_SEGMENTS 4u
#define SPARK_MINIMAX_H3_V5_ELEMENTS \
	((uint64_t)SPARK_MINIMAX_H3_V5_SEQ * SPARK_MINIMAX_H3_V5_HIDDEN)
#define SPARK_MINIMAX_H3_V5_LCG_SEED_DEFAULT 1099511628211ull

static uint32_t SparkMinimaxH3V5Failures;

static void *SparkMinimaxH3V5ReadFile(const char *directory, const char *name,
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

static void *SparkMinimaxH3V5DeviceUpload(const void *host, uint64_t bytes)
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

static float SparkMinimaxH3V5Bf16Element(uint16_t bits)
{
	uint32_t wide = (uint32_t)bits << 16u;
	float value;
	memcpy(&value,&wide,sizeof(value));
	return(value);
}

static uint16_t SparkMinimaxH3V5FloatBf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	return(uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

struct SparkMinimaxH3V5Weights
{
	void *query,*key,*value,*output_proj,*norm_q,*norm_k,*gate_up,*down,*norm1,*norm2;
};

static void SparkMinimaxH3V5LoadWeight(void **device, const char *weights_directory,
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
		void *host = SparkMinimaxH3V5ReadFile(weights_directory,file,
			(uint64_t)rows * columns * 2u);
		*device = SparkMinimaxH3V5DeviceUpload(host,(uint64_t)rows * columns * 2u);
		free(host);
	}
}

static void SparkMinimaxH3V5LoadBlock(struct SparkMinimaxH3V5Weights *weights,
	const char *weights_directory, const char *manifest, uint32_t block)
{
	char name[256];
	#define SPARK_MINIMAX_H3_V5_LOAD(field,label,rows,columns) \
		snprintf(name,sizeof(name),"transformer_blocks.%u.%s",block,label); \
		SparkMinimaxH3V5LoadWeight(&weights->field,weights_directory,manifest,name, \
			rows,columns)
	SPARK_MINIMAX_H3_V5_LOAD(query,"attn.to_q.weight",
		SPARK_MINIMAX_H3_V5_QKV,SPARK_MINIMAX_H3_V5_HIDDEN);
	SPARK_MINIMAX_H3_V5_LOAD(key,"attn.to_k.weight",
		SPARK_MINIMAX_H3_V5_QKV,SPARK_MINIMAX_H3_V5_HIDDEN);
	SPARK_MINIMAX_H3_V5_LOAD(value,"attn.to_v.weight",
		SPARK_MINIMAX_H3_V5_QKV,SPARK_MINIMAX_H3_V5_HIDDEN);
	SPARK_MINIMAX_H3_V5_LOAD(output_proj,"attn.to_out.0.weight",
		SPARK_MINIMAX_H3_V5_HIDDEN,SPARK_MINIMAX_H3_V5_QKV);
	SPARK_MINIMAX_H3_V5_LOAD(norm_q,"attn.norm_q.weight",1u,
		SPARK_MINIMAX_H3_V5_HEAD_DIM);
	SPARK_MINIMAX_H3_V5_LOAD(norm_k,"attn.norm_k.weight",1u,
		SPARK_MINIMAX_H3_V5_HEAD_DIM);
	SPARK_MINIMAX_H3_V5_LOAD(gate_up,"ff.net.0.proj.weight",
		SPARK_MINIMAX_H3_V5_FFN_FUSED,SPARK_MINIMAX_H3_V5_HIDDEN);
	SPARK_MINIMAX_H3_V5_LOAD(down,"ff.net.2.weight",
		SPARK_MINIMAX_H3_V5_HIDDEN,SPARK_MINIMAX_H3_V5_FFN);
	SPARK_MINIMAX_H3_V5_LOAD(norm1,"norm1.weight",1u,SPARK_MINIMAX_H3_V5_HIDDEN);
	SPARK_MINIMAX_H3_V5_LOAD(norm2,"norm2.weight",1u,SPARK_MINIMAX_H3_V5_HIDDEN);
	#undef SPARK_MINIMAX_H3_V5_LOAD
}

struct SparkMinimaxH3V5Scratch
{
	void *normed,*adaln,*q,*k,*v,*q_rope,*k_rope,*attn_raw,*attn_out,
		*ffn_fused,*ffn_mid,*ffn_out,*segments,*partials,*row_of;
};

static cudaError_t SparkMinimaxH3V5GemmDispatch(cudaStream_t stream, uint32_t tp_mode,
	const void *a, const void *w, uint32_t rows, uint32_t width, uint32_t depth,
	struct SparkMinimaxH3V5Scratch *scratch, void *out)
{
	cudaError_t error;
	uint32_t index;
	if ( tp_mode == 0u )
		return(SparkMinimaxH3Gemm(stream,a,w,rows,width,depth,scratch->segments,out));
	for (index=0u; index<SPARK_MINIMAX_H3_V5_SEGMENTS; index++)
	{
		error = SparkMinimaxH3GemmSegment(stream,a,w,rows,width,depth,index,
			(float *)scratch->partials);
		if ( error != cudaSuccess )
			return(error);
	}
	return(SparkMinimaxH3GemmCombinePartials(stream,(const float *)scratch->partials,
		(uint64_t)rows * width,out));
}

static cudaError_t SparkMinimaxH3V5BlockForward(cudaStream_t stream,
	const struct SparkMinimaxH3V5Weights *weights, const void *input_bf16,
	const void *scale_msa, const void *shift_msa, const void *gate_msa,
	const void *scale_mlp, const void *shift_mlp, const void *gate_mlp,
	const float *rope_cos, const float *rope_sin, const uint32_t *row_of,
	uint32_t tp_mode, struct SparkMinimaxH3V5Scratch *scratch, void *result_bf16)
{
	cudaError_t error;
	error = SparkMinimaxH3RmsNorm(stream,input_bf16,weights->norm1,
		SPARK_MINIMAX_H3_V5_SEQ,SPARK_MINIMAX_H3_V5_HIDDEN,
		SPARK_MINIMAX_H3_DIT_NORM_EPSILON,scratch->normed);
	if ( error != cudaSuccess )
		return(error);
	error = SparkMinimaxH3AdaLNIndexed(stream,scratch->normed,scale_msa,shift_msa,
		row_of,SPARK_MINIMAX_H3_V5_SEQ,SPARK_MINIMAX_H3_V5_HIDDEN,scratch->adaln);
	if ( error != cudaSuccess )
		return(error);
	error = SparkMinimaxH3V5GemmDispatch(stream,tp_mode,scratch->adaln,weights->query,
		SPARK_MINIMAX_H3_V5_SEQ,SPARK_MINIMAX_H3_V5_QKV,
		SPARK_MINIMAX_H3_V5_HIDDEN,scratch,scratch->q);
	if ( error != cudaSuccess )
		return(error);
	error = SparkMinimaxH3RmsNorm(stream,scratch->q,weights->norm_q,
		SPARK_MINIMAX_H3_V5_SEQ * SPARK_MINIMAX_H3_V5_HEADS,
		SPARK_MINIMAX_H3_V5_HEAD_DIM,SPARK_MINIMAX_H3_DIT_QK_NORM_EPSILON,
		scratch->q);
	if ( error != cudaSuccess )
		return(error);
	error = SparkMinimaxH3Rope3d(stream,scratch->q,rope_cos,rope_sin,
		SPARK_MINIMAX_H3_V5_SEQ,SPARK_MINIMAX_H3_V5_HEADS,
		SPARK_MINIMAX_H3_V5_HEAD_DIM,SPARK_MINIMAX_H3_V5_ROPE,scratch->q_rope);
	if ( error != cudaSuccess )
		return(error);
	error = SparkMinimaxH3V5GemmDispatch(stream,tp_mode,scratch->adaln,weights->key,
		SPARK_MINIMAX_H3_V5_SEQ,SPARK_MINIMAX_H3_V5_QKV,
		SPARK_MINIMAX_H3_V5_HIDDEN,scratch,scratch->k);
	if ( error != cudaSuccess )
		return(error);
	error = SparkMinimaxH3RmsNorm(stream,scratch->k,weights->norm_k,
		SPARK_MINIMAX_H3_V5_SEQ * SPARK_MINIMAX_H3_V5_HEADS,
		SPARK_MINIMAX_H3_V5_HEAD_DIM,SPARK_MINIMAX_H3_DIT_QK_NORM_EPSILON,
		scratch->k);
	if ( error != cudaSuccess )
		return(error);
	error = SparkMinimaxH3Rope3d(stream,scratch->k,rope_cos,rope_sin,
		SPARK_MINIMAX_H3_V5_SEQ,SPARK_MINIMAX_H3_V5_HEADS,
		SPARK_MINIMAX_H3_V5_HEAD_DIM,SPARK_MINIMAX_H3_V5_ROPE,scratch->k_rope);
	if ( error != cudaSuccess )
		return(error);
	error = SparkMinimaxH3V5GemmDispatch(stream,tp_mode,scratch->adaln,weights->value,
		SPARK_MINIMAX_H3_V5_SEQ,SPARK_MINIMAX_H3_V5_QKV,
		SPARK_MINIMAX_H3_V5_HIDDEN,scratch,scratch->v);
	if ( error != cudaSuccess )
		return(error);
	error = SparkMinimaxH3DenseAttention(stream,scratch->q_rope,scratch->k_rope,
		scratch->v,SPARK_MINIMAX_H3_V5_SEQ,SPARK_MINIMAX_H3_V5_HEADS,
		SPARK_MINIMAX_H3_V5_HEAD_DIM,scratch->attn_raw);
	if ( error != cudaSuccess )
		return(error);
	error = SparkMinimaxH3V5GemmDispatch(stream,tp_mode,scratch->attn_raw,
		weights->output_proj,SPARK_MINIMAX_H3_V5_SEQ,SPARK_MINIMAX_H3_V5_HIDDEN,
		SPARK_MINIMAX_H3_V5_QKV,scratch,scratch->attn_out);
	if ( error != cudaSuccess )
		return(error);
	error = SparkMinimaxH3GateResidualIndexed(stream,scratch->attn_out,gate_msa,
		input_bf16,row_of,SPARK_MINIMAX_H3_V5_SEQ,SPARK_MINIMAX_H3_V5_HIDDEN,
		scratch->normed);
	if ( error != cudaSuccess )
		return(error);
	error = SparkMinimaxH3RmsNorm(stream,scratch->normed,weights->norm2,
		SPARK_MINIMAX_H3_V5_SEQ,SPARK_MINIMAX_H3_V5_HIDDEN,
		SPARK_MINIMAX_H3_DIT_NORM_EPSILON,scratch->ffn_out);
	if ( error != cudaSuccess )
		return(error);
	error = SparkMinimaxH3AdaLNIndexed(stream,scratch->ffn_out,scale_mlp,shift_mlp,
		row_of,SPARK_MINIMAX_H3_V5_SEQ,SPARK_MINIMAX_H3_V5_HIDDEN,scratch->ffn_out);
	if ( error != cudaSuccess )
		return(error);
	error = SparkMinimaxH3V5GemmDispatch(stream,tp_mode,scratch->ffn_out,
		weights->gate_up,SPARK_MINIMAX_H3_V5_SEQ,SPARK_MINIMAX_H3_V5_FFN_FUSED,
		SPARK_MINIMAX_H3_V5_HIDDEN,scratch,scratch->ffn_fused);
	if ( error != cudaSuccess )
		return(error);
	error = SparkMinimaxH3SiluMul(stream,scratch->ffn_fused,
		SPARK_MINIMAX_H3_V5_SEQ,SPARK_MINIMAX_H3_V5_FFN,scratch->ffn_mid);
	if ( error != cudaSuccess )
		return(error);
	error = SparkMinimaxH3V5GemmDispatch(stream,tp_mode,scratch->ffn_mid,weights->down,
		SPARK_MINIMAX_H3_V5_SEQ,SPARK_MINIMAX_H3_V5_HIDDEN,
		SPARK_MINIMAX_H3_V5_FFN,scratch,scratch->ffn_out);
	if ( error != cudaSuccess )
		return(error);
	return(SparkMinimaxH3GateResidualIndexed(stream,scratch->ffn_out,gate_mlp,
		scratch->normed,row_of,SPARK_MINIMAX_H3_V5_SEQ,SPARK_MINIMAX_H3_V5_HIDDEN,
		result_bf16));
}

static void SparkMinimaxH3V5SeedLatents(uint64_t seed, float *latents)
{
	uint64_t state = seed;
	uint64_t index;
	for (index=0u; index<SPARK_MINIMAX_H3_V5_ELEMENTS; index++)
	{
		state = state * 6364136223846793005ull + 1442695040888963407ull;
		latents[index] = ((float)(uint32_t)(state >> 33u) / 4294967296.0f) * 2.0f - 1.0f;
	}
}

static int32_t SparkMinimaxH3V5RunLoop(const struct SparkMinimaxH3V5Weights *block0,
	const struct SparkMinimaxH3V5Weights *block1, const void *device_scale_msa,
	const void *device_shift_msa, const void *device_gate_msa,
	const void *device_scale_mlp, const void *device_shift_mlp,
	const void *device_gate_mlp, const float *device_cos, const float *device_sin,
	const int64_t *tags_host, const float *timesteps_host, uint64_t seed,
	uint32_t tp_mode, struct SparkMinimaxH3V5Scratch *scratch, void *device_h,
	void *device_result, uint16_t *latents_out)
{
	cudaStream_t stream = 0;
	uint32_t row_of[SPARK_MINIMAX_H3_V5_SEQ];
	uint32_t step;
	uint64_t index;
	float sample[SPARK_MINIMAX_H3_V5_ELEMENTS];
	SparkMinimaxH3V5SeedLatents(seed,sample);
	for (step=0u; step<SPARK_MINIMAX_H3_V5_STEPS; step++)
	{
		cudaError_t error;
		uint16_t *velocity;
		float sigma,sigma_next,timestep;
		for (index=0u; index<SPARK_MINIMAX_H3_V5_SEQ; index++)
			row_of[index] = step * 3u + (uint32_t)tags_host[index];
		if ( cudaMemcpy(scratch->row_of,row_of,sizeof(row_of),
			cudaMemcpyHostToDevice) != cudaSuccess )
			return(1);
		if ( cudaMemcpy(device_h,sample,SPARK_MINIMAX_H3_V5_ELEMENTS * 2u,
			cudaMemcpyHostToDevice) != cudaSuccess )
			return(1);
		error = SparkMinimaxH3V5BlockForward(stream,block0,device_h,
			device_scale_msa,device_shift_msa,device_gate_msa,device_scale_mlp,
			device_shift_mlp,device_gate_mlp,device_cos,device_sin,
			(const uint32_t *)scratch->row_of,tp_mode,scratch,device_result);
		if ( error != cudaSuccess )
			return(1);
		error = SparkMinimaxH3V5BlockForward(stream,block1,device_result,
			device_scale_msa,device_shift_msa,device_gate_msa,device_scale_mlp,
			device_shift_mlp,device_gate_mlp,device_cos,device_sin,
			(const uint32_t *)scratch->row_of,tp_mode,scratch,device_result);
		if ( error != cudaSuccess )
			return(1);
		if ( cudaStreamSynchronize(stream) != cudaSuccess )
			return(1);
		velocity = (uint16_t *)malloc(SPARK_MINIMAX_H3_V5_ELEMENTS * 2u);
		if ( velocity == 0 )
			return(1);
		if ( cudaMemcpy(velocity,device_result,SPARK_MINIMAX_H3_V5_ELEMENTS * 2u,
			cudaMemcpyDeviceToHost) != cudaSuccess )
			return(1);
		timestep = timesteps_host[step];
		sigma = 1.0f - timestep;
		sigma_next = step + 1u < SPARK_MINIMAX_H3_V5_STEPS ?
			1.0f - timesteps_host[step + 1u] : 0.0f;
		for (index=0u; index<SPARK_MINIMAX_H3_V5_ELEMENTS; index++)
		{
			float next;
			SparkMinimaxH3SchedulerStepElement(timestep,sigma,sigma_next,sample[index],
				SparkMinimaxH3V5Bf16Element(velocity[index]),&next);
			sample[index] = next;
			latents_out[step * SPARK_MINIMAX_H3_V5_ELEMENTS + index] =
				SparkMinimaxH3V5FloatBf16(next);
		}
		free(velocity);
	}
	return(0);
}

int main(int argc, char **argv)
{
	uint64_t seed = SPARK_MINIMAX_H3_V5_LCG_SEED_DEFAULT;
	int64_t *tags_host;
	float *timesteps_host;
	uint16_t *mods_scale_msa,*mods_shift_msa,*mods_gate_msa;
	uint16_t *mods_scale_mlp,*mods_shift_mlp,*mods_gate_mlp;
	float *rope_cos,*rope_sin;
	struct SparkMinimaxH3V5Weights block0,block1;
	struct SparkMinimaxH3V5Scratch scratch;
	static uint16_t runs[4u][SPARK_MINIMAX_H3_V5_STEPS][SPARK_MINIMAX_H3_V5_ELEMENTS];
	uint32_t step,run;
	if ( argc != 4 && argc != 5 )
	{
		printf("usage: %s FIXTURE_DIR WEIGHTS_DIR WEIGHT_MANIFEST [SEED]\n",argv[0]);
		return(2);
	}
	if ( argc == 5 )
		seed = strtoull(argv[4],0,0);
	tags_host = (int64_t *)SparkMinimaxH3V5ReadFile(argv[1],
		"token_tags__13.i64",SPARK_MINIMAX_H3_V5_SEQ * 8u);
	timesteps_host = (float *)SparkMinimaxH3V5ReadFile(argv[1],
		"timesteps__2.f32",SPARK_MINIMAX_H3_V5_STEPS * 4u);
	mods_scale_msa = (uint16_t *)SparkMinimaxH3V5ReadFile(argv[1],
		"mod_scale_msa__6x5376.u16",(uint64_t)SPARK_MINIMAX_H3_V5_MOD_ROWS *
		SPARK_MINIMAX_H3_V5_HIDDEN * 2u);
	mods_shift_msa = (uint16_t *)SparkMinimaxH3V5ReadFile(argv[1],
		"mod_shift_msa__6x5376.u16",(uint64_t)SPARK_MINIMAX_H3_V5_MOD_ROWS *
		SPARK_MINIMAX_H3_V5_HIDDEN * 2u);
	mods_gate_msa = (uint16_t *)SparkMinimaxH3V5ReadFile(argv[1],
		"mod_gate_msa__6x5376.u16",(uint64_t)SPARK_MINIMAX_H3_V5_MOD_ROWS *
		SPARK_MINIMAX_H3_V5_HIDDEN * 2u);
	mods_scale_mlp = (uint16_t *)SparkMinimaxH3V5ReadFile(argv[1],
		"mod_scale_mlp__6x5376.u16",(uint64_t)SPARK_MINIMAX_H3_V5_MOD_ROWS *
		SPARK_MINIMAX_H3_V5_HIDDEN * 2u);
	mods_shift_mlp = (uint16_t *)SparkMinimaxH3V5ReadFile(argv[1],
		"mod_shift_mlp__6x5376.u16",(uint64_t)SPARK_MINIMAX_H3_V5_MOD_ROWS *
		SPARK_MINIMAX_H3_V5_HIDDEN * 2u);
	mods_gate_mlp = (uint16_t *)SparkMinimaxH3V5ReadFile(argv[1],
		"mod_gate_mlp__6x5376.u16",(uint64_t)SPARK_MINIMAX_H3_V5_MOD_ROWS *
		SPARK_MINIMAX_H3_V5_HIDDEN * 2u);
	rope_cos = (float *)SparkMinimaxH3V5ReadFile(argv[1],
		"rope_cos__13x96.f32",(uint64_t)SPARK_MINIMAX_H3_V5_SEQ *
		SPARK_MINIMAX_H3_V5_ROPE * 4u);
	rope_sin = (float *)SparkMinimaxH3V5ReadFile(argv[1],
		"rope_sin__13x96.f32",(uint64_t)SPARK_MINIMAX_H3_V5_SEQ *
		SPARK_MINIMAX_H3_V5_ROPE * 4u);
	printf("minimax_h3 V5 gate: mini-DiT determinism (2 steps x blocks 0,1, "
		"real weights, seed %llu)\n",(unsigned long long)seed);
	SparkMinimaxH3V5LoadBlock(&block0,argv[2],argv[3],0u);
	SparkMinimaxH3V5LoadBlock(&block1,argv[2],argv[3],1u);
	cudaMalloc(&scratch.normed,SPARK_MINIMAX_H3_V5_ELEMENTS * 2u);
	cudaMalloc(&scratch.adaln,SPARK_MINIMAX_H3_V5_ELEMENTS * 2u);
	cudaMalloc(&scratch.q,(uint64_t)SPARK_MINIMAX_H3_V5_SEQ *
		SPARK_MINIMAX_H3_V5_QKV * 2u);
	cudaMalloc(&scratch.k,(uint64_t)SPARK_MINIMAX_H3_V5_SEQ *
		SPARK_MINIMAX_H3_V5_QKV * 2u);
	cudaMalloc(&scratch.v,(uint64_t)SPARK_MINIMAX_H3_V5_SEQ *
		SPARK_MINIMAX_H3_V5_QKV * 2u);
	cudaMalloc(&scratch.q_rope,(uint64_t)SPARK_MINIMAX_H3_V5_SEQ *
		SPARK_MINIMAX_H3_V5_QKV * 2u);
	cudaMalloc(&scratch.k_rope,(uint64_t)SPARK_MINIMAX_H3_V5_SEQ *
		SPARK_MINIMAX_H3_V5_QKV * 2u);
	cudaMalloc(&scratch.attn_raw,(uint64_t)SPARK_MINIMAX_H3_V5_SEQ *
		SPARK_MINIMAX_H3_V5_QKV * 2u);
	cudaMalloc(&scratch.attn_out,SPARK_MINIMAX_H3_V5_ELEMENTS * 2u);
	cudaMalloc(&scratch.ffn_fused,(uint64_t)SPARK_MINIMAX_H3_V5_SEQ *
		SPARK_MINIMAX_H3_V5_FFN_FUSED * 2u);
	cudaMalloc(&scratch.ffn_mid,(uint64_t)SPARK_MINIMAX_H3_V5_SEQ *
		SPARK_MINIMAX_H3_V5_FFN * 2u);
	cudaMalloc(&scratch.ffn_out,SPARK_MINIMAX_H3_V5_ELEMENTS * 2u);
	cudaMalloc(&scratch.segments,(uint64_t)SPARK_MINIMAX_H3_V5_SEQ *
		SPARK_MINIMAX_H3_V5_FFN_FUSED * 4u * 4u);
	cudaMalloc(&scratch.partials,SPARK_MINIMAX_H3_V5_ELEMENTS * 4u *
		SPARK_MINIMAX_H3_V5_SEGMENTS);
	cudaMalloc(&scratch.row_of,SPARK_MINIMAX_H3_V5_SEQ * 4u);
	{
		void *device_h = 0;
		void *device_result = 0;
		void *device_scale_msa = SparkMinimaxH3V5DeviceUpload(mods_scale_msa,
			(uint64_t)SPARK_MINIMAX_H3_V5_MOD_ROWS * SPARK_MINIMAX_H3_V5_HIDDEN * 2u);
		void *device_shift_msa = SparkMinimaxH3V5DeviceUpload(mods_shift_msa,
			(uint64_t)SPARK_MINIMAX_H3_V5_MOD_ROWS * SPARK_MINIMAX_H3_V5_HIDDEN * 2u);
		void *device_gate_msa = SparkMinimaxH3V5DeviceUpload(mods_gate_msa,
			(uint64_t)SPARK_MINIMAX_H3_V5_MOD_ROWS * SPARK_MINIMAX_H3_V5_HIDDEN * 2u);
		void *device_scale_mlp = SparkMinimaxH3V5DeviceUpload(mods_scale_mlp,
			(uint64_t)SPARK_MINIMAX_H3_V5_MOD_ROWS * SPARK_MINIMAX_H3_V5_HIDDEN * 2u);
		void *device_shift_mlp = SparkMinimaxH3V5DeviceUpload(mods_shift_mlp,
			(uint64_t)SPARK_MINIMAX_H3_V5_MOD_ROWS * SPARK_MINIMAX_H3_V5_HIDDEN * 2u);
		void *device_gate_mlp = SparkMinimaxH3V5DeviceUpload(mods_gate_mlp,
			(uint64_t)SPARK_MINIMAX_H3_V5_MOD_ROWS * SPARK_MINIMAX_H3_V5_HIDDEN * 2u);
		float *device_cos = (float *)SparkMinimaxH3V5DeviceUpload(rope_cos,
			(uint64_t)SPARK_MINIMAX_H3_V5_SEQ * SPARK_MINIMAX_H3_V5_ROPE * 4u);
		float *device_sin = (float *)SparkMinimaxH3V5DeviceUpload(rope_sin,
			(uint64_t)SPARK_MINIMAX_H3_V5_SEQ * SPARK_MINIMAX_H3_V5_ROPE * 4u);
		cudaMalloc(&device_h,SPARK_MINIMAX_H3_V5_ELEMENTS * 2u);
		cudaMalloc(&device_result,SPARK_MINIMAX_H3_V5_ELEMENTS * 2u);
		for (run=0u; run<4u; run++)
		{
			if ( SparkMinimaxH3V5RunLoop(&block0,&block1,device_scale_msa,
				device_shift_msa,device_gate_msa,device_scale_mlp,device_shift_mlp,
				device_gate_mlp,device_cos,device_sin,tags_host,timesteps_host,seed,
				run < 2u ? 0u : 1u,&scratch,device_h,device_result,
				(uint16_t *)runs[run]) != 0 )
			{
				printf("loop run %u cuda failure FAIL\n",run);
				return(1);
			}
		}
		for (step=0u; step<SPARK_MINIMAX_H3_V5_STEPS; step++)
		{
			int32_t same_tp1 = memcmp(runs[0u][step],runs[1u][step],
				SPARK_MINIMAX_H3_V5_ELEMENTS * 2u) == 0;
			int32_t same_tp4 = memcmp(runs[2u][step],runs[3u][step],
				SPARK_MINIMAX_H3_V5_ELEMENTS * 2u) == 0;
			int32_t tp_cross = memcmp(runs[0u][step],runs[2u][step],
				SPARK_MINIMAX_H3_V5_ELEMENTS * 2u) == 0;
			printf("step %u same-seed rerun TP1 %s TP4 %s TP4-vs-TP1 %s\n",step,
				same_tp1 != 0 ? "bit-identical OK" : "FAIL",
				same_tp4 != 0 ? "bit-identical OK" : "FAIL",
				tp_cross != 0 ? "bit-identical OK" : "FAIL");
			if ( same_tp1 == 0 || same_tp4 == 0 || tp_cross == 0 )
				SparkMinimaxH3V5Failures++;
		}
	}
	if ( cudaDeviceSynchronize() != cudaSuccess )
	{
		printf("device synchronize failed FAIL\n");
		SparkMinimaxH3V5Failures++;
	}
	if ( SparkMinimaxH3V5Failures != 0u )
	{
		printf("V5 gate FAIL (%u failures)\n",SparkMinimaxH3V5Failures);
		return(1);
	}
	printf("V5 gate OK: same-seed latents bit-identical across reruns and "
		"TP4-segmented vs TP1 GEMM partition at mini-DiT scale\n");
	return(0);
}
