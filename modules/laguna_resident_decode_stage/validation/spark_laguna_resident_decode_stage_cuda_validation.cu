#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "modules/laguna_resident_decode_stage/source/cuda/layer.cuh"
#include "../source/spark_laguna_resident_decode_stage_internal.h"

#define VALIDATION_HEADS LAGUNA_Q_HEADS_FULL
#define VALIDATION_WINDOW LAGUNA_WINDOW
#define VALIDATION_CONTEXT 4u
#define VALIDATION_THREADS 256u

static uint32_t validation_seed = 20260909u;

static float ValidationNextRandom(void)
{
	validation_seed = (validation_seed * 1664525u) + 1013904223u;
	return((float)((validation_seed >> 8) & 0xffffu) / 32768.0f - 1.0f);
}

static int ValidationDumpF32(const char *path,const float *host,uint32_t count)
{
	FILE *file = fopen(path,"wb");
	uint32_t index;
	if ( file == 0 )
		return(-1);
	for (index=0u; index<count; index++)
		fprintf(file,"%.9g\n",(double)host[index]);
	fclose(file);
	return(0);
}

static int ValidationDumpU32(const char *path,const uint32_t *host,uint32_t count)
{
	FILE *file = fopen(path,"wb");
	uint32_t index;
	if ( file == 0 )
		return(-1);
	for (index=0u; index<count; index++)
		fprintf(file,"%u\n",host[index]);
	fclose(file);
	return(0);
}

int main(int argc,char **argv)
{
	const char *dump_prefix = argc > 1 ? argv[1] : "laguna_v0";
	float host_yarn[SPARK_LAGUNA_MODEL_ROPE_FULL_ROTARY_DIMENSION / 2u];
	float host_yarn_second[SPARK_LAGUNA_MODEL_ROPE_FULL_ROTARY_DIMENSION / 2u];
	float host_q[VALIDATION_HEADS * LAGUNA_HEAD_DIM];
	float host_norm_expected[VALIDATION_HEADS * LAGUNA_HEAD_DIM];
	uint16_t *device_q;
	uint16_t *device_normed;
	float *device_yarn;
	uint32_t *device_positions;
	uint32_t host_positions[1];
	uint32_t host_window[VALIDATION_WINDOW];
	uint32_t *device_window;
	uint32_t sequence_of_row[1];
	uint32_t context_length[1];
	uint32_t row_position[1];
	uint32_t *device_sequence;
	uint32_t *device_context;
	uint32_t *device_row_position;
	float host_gate[VALIDATION_HEADS];
	uint16_t *device_gate;
	uint32_t head,index;
	double max_qk_norm_error = 0.0;
	uint32_t window_failures = 0;
	char path[512];
	uint32_t multiprocessors = 0u;
	if ( SparkLagunaConfigureCudaModule(&multiprocessors) == LM_LAUNCH_OK )
		fprintf(stderr,"laguna validation: device with %u SMs\n",multiprocessors);
	else
		fprintf(stderr,"laguna validation: no sm_121 device; host checks only\n");
	LagunaBuildYarnInvFrequency(host_yarn,
		SPARK_LAGUNA_MODEL_ROPE_FULL_ROTARY_DIMENSION,
		SPARK_LAGUNA_MODEL_ROPE_FULL_THETA,
		SPARK_LAGUNA_MODEL_ROPE_FULL_FACTOR,
		SPARK_LAGUNA_MODEL_ROPE_FULL_ORIGINAL_POSITIONS,
		SPARK_LAGUNA_MODEL_ROPE_FULL_BETA_FAST,
		SPARK_LAGUNA_MODEL_ROPE_FULL_BETA_SLOW);
	LagunaBuildYarnInvFrequency(host_yarn_second,
		SPARK_LAGUNA_MODEL_ROPE_FULL_ROTARY_DIMENSION,
		SPARK_LAGUNA_MODEL_ROPE_FULL_THETA,
		SPARK_LAGUNA_MODEL_ROPE_FULL_FACTOR,
		SPARK_LAGUNA_MODEL_ROPE_FULL_ORIGINAL_POSITIONS,
		SPARK_LAGUNA_MODEL_ROPE_FULL_BETA_FAST,
		SPARK_LAGUNA_MODEL_ROPE_FULL_BETA_SLOW);
	if ( memcmp(host_yarn,host_yarn_second,sizeof(host_yarn)) != 0 )
	{
		fprintf(stderr,"laguna validation: yarn table not deterministic\n");
		return(1);
	}
	snprintf(path,sizeof(path),"%s_yarn_table.txt",dump_prefix);
	if ( ValidationDumpF32(path,host_yarn,(uint32_t)(sizeof(host_yarn) / sizeof(host_yarn[0]))) != 0 )
		return(2);
	if ( host_yarn[0] <= 1.0f )
	{
		fprintf(stderr,"laguna validation: yarn inv_freq[0] must exceed 1 (theta^0 blended toward theta^-0)\n");
		return(1);
	}
	if ( host_yarn[SPARK_LAGUNA_MODEL_ROPE_FULL_ROTARY_DIMENSION / 2u - 1u] >= 1.0f )
	{
		fprintf(stderr,"laguna validation: yarn inv_freq tail must be below the unblended base\n");
		return(1);
	}
	for (head=0u; head<VALIDATION_HEADS*LAGUNA_HEAD_DIM; head++)
		host_q[head] = ValidationNextRandom();
	for (head=0u; head<VALIDATION_HEADS*LAGUNA_HEAD_DIM; head++)
	{
		double sum = 0.0;
		uint32_t within = head % LAGUNA_HEAD_DIM;
		uint32_t start = head - within;
		uint32_t element;
		for (element=0u; element<LAGUNA_HEAD_DIM; element++)
			sum += (double)host_q[start + element] * host_q[start + element];
		host_norm_expected[head] = (float)((double)host_q[head] /
			sqrt(sum / (double)LAGUNA_HEAD_DIM + (double)LAGUNA_RMS_EPSILON));
	}
	if ( cudaMalloc((void **)&device_q,sizeof(host_q)) != cudaSuccess ||
		cudaMalloc((void **)&device_normed,sizeof(host_q)) != cudaSuccess ||
		cudaMalloc((void **)&device_yarn,sizeof(host_yarn)) != cudaSuccess ||
		cudaMalloc((void **)&device_positions,sizeof(host_positions)) != cudaSuccess ||
		cudaMalloc((void **)&device_window,sizeof(host_window)) != cudaSuccess ||
		cudaMalloc((void **)&device_sequence,sizeof(sequence_of_row)) != cudaSuccess ||
		cudaMalloc((void **)&device_context,sizeof(context_length)) != cudaSuccess ||
		cudaMalloc((void **)&device_row_position,sizeof(row_position)) != cudaSuccess ||
		cudaMalloc((void **)&device_gate,sizeof(host_gate)) != cudaSuccess )
	{
		fprintf(stderr,"laguna validation: allocation failed\n");
		return(2);
	}
	for (head=0u; head<VALIDATION_HEADS; head++)
		host_gate[head] = ValidationNextRandom();
	host_positions[0] = 3u;
	sequence_of_row[0] = 0u;
	context_length[0] = VALIDATION_CONTEXT;
	row_position[0] = 3u;
	if ( cudaMemcpy(device_q,host_q,sizeof(host_q),cudaMemcpyHostToDevice) != cudaSuccess ||
		cudaMemcpy(device_yarn,host_yarn,sizeof(host_yarn),cudaMemcpyHostToDevice) != cudaSuccess ||
		cudaMemcpy(device_positions,host_positions,sizeof(host_positions),cudaMemcpyHostToDevice) != cudaSuccess ||
		cudaMemcpy(device_sequence,sequence_of_row,sizeof(sequence_of_row),cudaMemcpyHostToDevice) != cudaSuccess ||
		cudaMemcpy(device_context,context_length,sizeof(context_length),cudaMemcpyHostToDevice) != cudaSuccess ||
		cudaMemcpy(device_row_position,row_position,sizeof(row_position),cudaMemcpyHostToDevice) != cudaSuccess ||
		cudaMemcpy(device_gate,host_gate,sizeof(host_gate),cudaMemcpyHostToDevice) != cudaSuccess )
	{
		fprintf(stderr,"laguna validation: upload failed\n");
		return(2);
	}
	LmBuildSlidingWindowPositionsKernel<VALIDATION_THREADS><<<1u,VALIDATION_THREADS>>>(
		device_sequence,device_context,device_row_position,1u,LAGUNA_WINDOW,device_window);
	if ( cudaDeviceSynchronize() != cudaSuccess )
	{
		fprintf(stderr,"laguna validation: window kernel failed\n");
		return(2);
	}
	if ( cudaMemcpy(host_window,device_window,sizeof(host_window),cudaMemcpyDeviceToHost) != cudaSuccess )
		return(2);
	for (index=0u; index<VALIDATION_WINDOW; index++)
	{
		uint32_t expected = index < VALIDATION_CONTEXT ? index : 0xffffffffu;
		if ( host_window[index] != expected )
			window_failures++;
	}
	if ( window_failures != 0u )
	{
		fprintf(stderr,"laguna validation: window positions wrong at %u slots\n",window_failures);
		return(1);
	}
	snprintf(path,sizeof(path),"%s_window_positions.txt",dump_prefix);
	ValidationDumpU32(path,host_window,VALIDATION_WINDOW);
	snprintf(path,sizeof(path),"%s_q_inputs.txt",dump_prefix);
	ValidationDumpF32(path,host_q,(uint32_t)(sizeof(host_q) / sizeof(host_q[0])));
	snprintf(path,sizeof(path),"%s_qk_norm_expected.txt",dump_prefix);
	ValidationDumpF32(path,host_norm_expected,(uint32_t)(sizeof(host_norm_expected) / sizeof(host_norm_expected[0])));
	{
		double worst = 0.0;
		for (head=0u; head<VALIDATION_HEADS*LAGUNA_HEAD_DIM; head++)
		{
			double delta = fabs((double)host_norm_expected[head] - (double)host_q[head]);
			if ( delta > worst )
				worst = delta;
		}
		max_qk_norm_error = worst;
	}
	printf("laguna validation: yarn table dumped, window positions exact, qk-norm reference spread %.6g (diagnostic only)\n",
		max_qk_norm_error);
	return(0);
}
