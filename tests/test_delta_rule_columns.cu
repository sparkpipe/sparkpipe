#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "inference/kernels/linear_attn.cuh"

#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { fprintf(stderr,"FAIL line=%d cuda=%s call=%s\n",__LINE__,cudaGetErrorString(e),#call); exit(1); } } while(0)
#define REQUIRE(test) do { if(!(test)) { fprintf(stderr,"FAIL line=%d test=%s\n",__LINE__,#test); exit(1); } } while(0)

#define DR_THREADS 256u
#define DR_KEY 128u
#define DR_VALUE 128u

static uint32_t random_state = 20260928u;

static uint32_t Random()
{
	random_state ^= random_state << 13u;
	random_state ^= random_state >> 17u;
	random_state ^= random_state << 5u;
	return random_state;
}

static float Signed()
{
	return ((int32_t)(Random()%2049u)-1024)/1024.0f;
}

static uint16_t Bf16(float value)
{
	uint32_t bits;
	memcpy(&bits,&value,sizeof(bits));
	return (uint16_t)((bits+0x7fffu+((bits>>16u)&1u))>>16u);
}

template<class T> static T *Upload(const std::vector<T> &host)
{
	T *device;
	CUDA(cudaMalloc(&device,host.size()*sizeof(T)));
	CUDA(cudaMemcpy(device,host.data(),host.size()*sizeof(T),cudaMemcpyHostToDevice));
	return device;
}

template<class T> static std::vector<T> Download(const T *device,size_t count)
{
	std::vector<T> host(count);
	CUDA(cudaMemcpy(host.data(),device,count*sizeof(T),cudaMemcpyDeviceToHost));
	return host;
}

typedef struct DeltaCase
{
	uint32_t sequences,heads,rows_per_sequence,commit;
}
DeltaCase;

template<uint32_t COLUMNS>
static void Run(const DeltaCase *shape,const std::vector<float> &state,const std::vector<uint16_t> &query,const std::vector<uint16_t> &key,const std::vector<uint16_t> &value,const std::vector<float> &forget,const std::vector<float> &write,std::vector<float> *state_out,std::vector<uint16_t> *output_out,cudaStream_t stream)
{
	const uint32_t rows=shape->sequences*shape->rows_per_sequence, slot_bytes=shape->heads*DR_KEY*DR_VALUE*4u;
	std::vector<uint32_t> index(shape->sequences),begin(shape->sequences+1u);
	uint8_t *device_state; uint16_t *device_output;
	for (uint32_t sequence=0u; sequence<shape->sequences; sequence++) { index[sequence]=shape->sequences-1u-sequence; begin[sequence]=sequence*shape->rows_per_sequence; }
	begin[shape->sequences]=rows;
	uint32_t *device_index=Upload(index),*device_begin=Upload(begin);
	uint16_t *device_query=Upload(query),*device_key=Upload(key),*device_value=Upload(value);
	float *device_forget=Upload(forget),*device_write=Upload(write);
	CUDA(cudaMalloc(&device_state,state.size()*4u));
	CUDA(cudaMemcpy(device_state,state.data(),state.size()*4u,cudaMemcpyHostToDevice));
	CUDA(cudaMalloc(&device_output,(uint64_t)rows*shape->heads*DR_VALUE*2u));
	CUDA(cudaMemset(device_output,0xff,(uint64_t)rows*shape->heads*DR_VALUE*2u));
	if (DR_KEY*COLUMNS*4u > 48u*1024u)
		CUDA(cudaFuncSetAttribute((const void *)LmDeltaRuleKernel<DR_THREADS,DR_KEY,DR_VALUE,float,COLUMNS>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)(DR_KEY*COLUMNS*4u)));
	LmDeltaRuleKernel<DR_THREADS,DR_KEY,DR_VALUE,float,COLUMNS><<<dim3(shape->sequences,shape->heads,DR_VALUE/COLUMNS),DR_THREADS,DR_KEY*COLUMNS*4u,stream>>>(device_state,slot_bytes,device_index,device_begin,0,device_query,device_key,device_value,device_forget,device_write,device_output,shape->heads,1u,shape->sequences,shape->commit,0);
	CUDA(cudaPeekAtLastError());
	CUDA(cudaStreamSynchronize(stream));
	*state_out=Download((const float *)device_state,state.size());
	*output_out=Download(device_output,(size_t)rows*shape->heads*DR_VALUE);
	CUDA(cudaFree(device_state)); CUDA(cudaFree(device_output)); CUDA(cudaFree(device_index)); CUDA(cudaFree(device_begin));
	CUDA(cudaFree(device_query)); CUDA(cudaFree(device_key)); CUDA(cudaFree(device_value)); CUDA(cudaFree(device_forget)); CUDA(cudaFree(device_write));
}

static void Case(const DeltaCase *shape,cudaStream_t stream)
{
	const uint32_t rows=shape->sequences*shape->rows_per_sequence;
	std::vector<float> state((uint64_t)shape->sequences*shape->heads*DR_KEY*DR_VALUE),forget((uint64_t)rows*shape->heads*DR_KEY),write((uint64_t)rows*shape->heads);
	std::vector<uint16_t> query((uint64_t)rows*shape->heads*DR_KEY),key(query.size()),value((uint64_t)rows*shape->heads*DR_VALUE);
	std::vector<float> state_whole,state_split,state_narrow;
	std::vector<uint16_t> output_whole,output_split,output_narrow;
	for (auto &x : state) x=Signed()*0.5f;
	for (auto &x : forget) x=0.5f+0.5f*(Random()%1024u)/1024.0f;
	for (auto &x : write) x=(Random()%1024u)/1024.0f;
	for (auto &x : query) x=Bf16(Signed());
	for (auto &x : key) x=Bf16(Signed());
	for (auto &x : value) x=Bf16(Signed()*2.0f);
	Run<DR_VALUE>(shape,state,query,key,value,forget,write,&state_whole,&output_whole,stream);
	Run<16u>(shape,state,query,key,value,forget,write,&state_split,&output_split,stream);
	Run<8u>(shape,state,query,key,value,forget,write,&state_narrow,&output_narrow,stream);
	REQUIRE(memcmp(state_whole.data(),state_split.data(),state_whole.size()*4u) == 0);
	REQUIRE(memcmp(state_whole.data(),state_narrow.data(),state_whole.size()*4u) == 0);
	REQUIRE(memcmp(output_whole.data(),output_split.data(),output_whole.size()*2u) == 0);
	REQUIRE(memcmp(output_whole.data(),output_narrow.data(),output_whole.size()*2u) == 0);
	if (shape->commit == 0u)
		REQUIRE(memcmp(state.data(),state_whole.data(),state.size()*4u) == 0);
	printf("PASS sequences=%u heads=%u rows_per_sequence=%u commit=%u\n",shape->sequences,shape->heads,shape->rows_per_sequence,shape->commit);
}

int main(int argc,char **argv)
{
	static const DeltaCase cases[]={{1u,4u,1u,1u},{8u,4u,1u,1u},{2u,4u,3u,1u},{3u,2u,2u,0u}};
	cudaStream_t stream;
	if (argc < 2 || strcmp(argv[1],"--run") != 0)
	{
		fprintf(stderr,"usage: test_delta_rule_columns --run\n");
		return 2;
	}
	CUDA(cudaStreamCreate(&stream));
	for (uint32_t index=0u; index<sizeof(cases)/sizeof(cases[0]); index++)
		Case(&cases[index],stream);
	printf("test_delta_rule_columns PASS\n");
	return 0;
}
