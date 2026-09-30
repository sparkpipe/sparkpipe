#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "common/common_glm_cuda_tree/spark_glm_cuda_unity.cu"
#include "sparkpipe/spark_row_bucket.cuh"
#include "sparkpipe/spark_tp_chain_graph.h"

#define RB_MAX_ROWS SPARK_TP_CHAIN_GRAPH_MAX_ROWS

static int rb_failures;

#define RB_CHECK(condition,...) do { if ( !(condition) ) { fprintf(stderr,"FAIL %s:%d ",__FILE__,__LINE__); fprintf(stderr,__VA_ARGS__); fprintf(stderr,"\n"); rb_failures++; } } while (0)
#define RB_CUDA(call) do { cudaError_t rb_error = (call); if ( rb_error != cudaSuccess ) { fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(rb_error)); exit(2); } } while (0)

static uint16_t RbValue(uint64_t key)
{
	uint64_t z = key * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull;
	float value;
	uint32_t bits;
	z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
	z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
	z ^= z >> 31u;
	value = (float)((int32_t)((z >> 40u) & 0xfffu) - 2048) / 16384.0f;
	memcpy(&bits,&value,4u);
	return (uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

static void RbPadIndices(void)
{
	uint32_t host[3][RB_MAX_ROWS],back[3][RB_MAX_ROWS],*device[3];
	uint32_t array,row,rows,bucket;
	for (array=0u; array<3u; array++)
		RB_CUDA(cudaMalloc((void **)&device[array],sizeof(host[array])));
	for (rows=1u; rows<=RB_MAX_ROWS; rows++)
	{
		bucket = SparkTpChainGraphBucketRows(rows,RB_MAX_ROWS);
		for (array=0u; array<3u; array++)
		{
			for (row=0u; row<RB_MAX_ROWS; row++)
				host[array][row] = row < rows ? 1000u * array + 7u * row + 1u : 0xdeadbeefu;
			RB_CUDA(cudaMemcpy(device[array],host[array],sizeof(host[array]),cudaMemcpyHostToDevice));
		}
		RB_CUDA(SparkRowBucketPad(0,device[0],device[1],(rows & 1u) != 0u ? device[2] : 0,rows,bucket));
		for (array=0u; array<3u; array++)
		{
			RB_CUDA(cudaMemcpy(back[array],device[array],sizeof(back[array]),cudaMemcpyDeviceToHost));
			for (row=0u; row<RB_MAX_ROWS; row++)
			{
				uint32_t expected = row < rows ? host[array][row] : row < bucket && (array < 2u || (rows & 1u) != 0u) ? host[array][rows - 1u] : 0xdeadbeefu;
				RB_CHECK(back[array][row] == expected,"pad rows=%u bucket=%u array=%u row=%u got=%u want=%u",rows,bucket,array,row,back[array][row],expected);
			}
		}
	}
	RB_CHECK(SparkRowBucketPad(0,device[0],0,0,5u,4u) == cudaErrorInvalidValue,"a bucket below the rows is refused");
	for (array=0u; array<3u; array++)
		RB_CUDA(cudaFree(device[array]));
}

static void RbLinearRows(uint32_t input_dimension,uint32_t output_dimension,uint32_t multiprocessors)
{
	static const uint32_t cases[] = {1u,2u,3u,5u,7u,8u,9u,12u,16u,17u,31u,33u,47u,64u,65u,100u,129u,200u,255u};
	std::vector<uint16_t> weight((size_t)output_dimension * input_dimension);
	std::vector<uint16_t> activation((size_t)RB_MAX_ROWS * input_dimension),padded((size_t)RB_MAX_ROWS * input_dimension);
	std::vector<uint16_t> exact((size_t)RB_MAX_ROWS * output_dimension),bucketed((size_t)RB_MAX_ROWS * output_dimension);
	uint16_t *device_weight,*device_activation,*device_output;
	uint32_t *row_offset,*tile_prefix,host_offset[2];
	uint64_t index;
	uint32_t item,rows,bucket,row,column,mismatches,pad_mismatches;
	for (index=0u; index<weight.size(); index++)
		weight[index] = RbValue(index * 3u + input_dimension);
	for (index=0u; index<activation.size(); index++)
		activation[index] = RbValue(index * 5u + 11u + output_dimension);
	RB_CUDA(cudaMalloc((void **)&device_weight,weight.size() * 2u));
	RB_CUDA(cudaMalloc((void **)&device_activation,activation.size() * 2u));
	RB_CUDA(cudaMalloc((void **)&device_output,exact.size() * 2u));
	RB_CUDA(cudaMalloc((void **)&row_offset,2u * sizeof(uint32_t)));
	RB_CUDA(cudaMalloc((void **)&tile_prefix,2u * sizeof(uint32_t)));
	RB_CUDA(cudaMemcpy(device_weight,weight.data(),weight.size() * 2u,cudaMemcpyHostToDevice));
	for (item=0u; item<sizeof(cases) / sizeof(cases[0]); item++)
	{
		rows = cases[item];
		bucket = SparkTpChainGraphBucketRows(rows,RB_MAX_ROWS);
		memcpy(padded.data(),activation.data(),(size_t)rows * input_dimension * 2u);
		for (row=rows; row<bucket; row++)
			memcpy(&padded[(size_t)row * input_dimension],&activation[(size_t)(rows - 1u) * input_dimension],(size_t)input_dimension * 2u);
		host_offset[0] = 0u;
		host_offset[1] = rows;
		RB_CUDA(cudaMemcpy(row_offset,host_offset,sizeof(host_offset),cudaMemcpyHostToDevice));
		RB_CUDA(cudaMemcpy(device_activation,activation.data(),(size_t)rows * input_dimension * 2u,cudaMemcpyHostToDevice));
		RB_CUDA(cudaMemset(device_output,0xff,exact.size() * 2u));
		RB_CHECK(GlmLaunchBf16Linear(device_activation,device_weight,device_output,row_offset,tile_prefix,rows,input_dimension,output_dimension,output_dimension,0u,multiprocessors,0) == LM_LAUNCH_OK,"linear rows=%u",rows);
		RB_CUDA(cudaDeviceSynchronize());
		RB_CUDA(cudaMemcpy(exact.data(),device_output,exact.size() * 2u,cudaMemcpyDeviceToHost));
		host_offset[1] = bucket;
		RB_CUDA(cudaMemcpy(row_offset,host_offset,sizeof(host_offset),cudaMemcpyHostToDevice));
		RB_CUDA(cudaMemcpy(device_activation,padded.data(),(size_t)bucket * input_dimension * 2u,cudaMemcpyHostToDevice));
		RB_CUDA(cudaMemset(device_output,0xff,bucketed.size() * 2u));
		RB_CHECK(GlmLaunchBf16Linear(device_activation,device_weight,device_output,row_offset,tile_prefix,bucket,input_dimension,output_dimension,output_dimension,0u,multiprocessors,0) == LM_LAUNCH_OK,"linear bucket=%u",bucket);
		RB_CUDA(cudaDeviceSynchronize());
		RB_CUDA(cudaMemcpy(bucketed.data(),device_output,bucketed.size() * 2u,cudaMemcpyDeviceToHost));
		mismatches = 0u;
		pad_mismatches = 0u;
		for (row=0u; row<bucket; row++)
			for (column=0u; column<output_dimension; column++)
			{
				uint16_t got = bucketed[(size_t)row * output_dimension + column];
				uint16_t want = exact[(size_t)(row < rows ? row : rows - 1u) * output_dimension + column];
				if ( got != want )
				{
					if ( row < rows )
						mismatches++;
					else
						pad_mismatches++;
				}
			}
		RB_CHECK(mismatches == 0u && pad_mismatches == 0u,"linear %ux%u rows=%u bucket=%u: %u real and %u pad outputs differ from the unpadded rows",input_dimension,output_dimension,rows,bucket,mismatches,pad_mismatches);
		printf("linear %ux%u rows=%u bucket=%u %s\n",input_dimension,output_dimension,rows,bucket,mismatches == 0u && pad_mismatches == 0u ? "bit-identical" : "DIFFERS");
	}
	RB_CUDA(cudaFree(device_weight));
	RB_CUDA(cudaFree(device_activation));
	RB_CUDA(cudaFree(device_output));
	RB_CUDA(cudaFree(row_offset));
	RB_CUDA(cudaFree(tile_prefix));
}

int main(void)
{
	int multiprocessors = 0;
	RB_CUDA(cudaDeviceGetAttribute(&multiprocessors,cudaDevAttrMultiProcessorCount,0));
	RbPadIndices();
	RbLinearRows(GLM_HIDDEN,2048u,(uint32_t)multiprocessors);
	RbLinearRows(2048u,GLM_HIDDEN,(uint32_t)multiprocessors);
	if ( rb_failures != 0 )
	{
		fprintf(stderr,"glm52 row bucket: %d failures\n",rb_failures);
		return 1;
	}
	printf("glm52 row bucket: padded rows leave every real row bit-identical\n");
	return 0;
}
