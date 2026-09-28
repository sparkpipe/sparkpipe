#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "inference/kernels/head.cuh"

#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { fprintf(stderr,"FAIL line=%d cuda=%s call=%s\n",__LINE__,cudaGetErrorString(e),#call); exit(1); } } while(0)
#define REQUIRE(test) do { if(!(test)) { fprintf(stderr,"FAIL line=%d test=%s\n",__LINE__,#test); exit(1); } } while(0)

#define STAGE_THREADS 256u
#define STAGE_TILE 128u

template<uint32_t THREADS, uint32_t TILE>
__global__ __launch_bounds__(THREADS, 1) void ScalarGreedyKernel(const uint16_t *normed, const uint16_t *weight, const uint32_t *token_ids, float *score, uint32_t *token, uint32_t hidden, uint32_t vocabulary)
{
	LmHeadCandidateScalarBody<THREADS, TILE>(normed, weight, token_ids, score, token, hidden, vocabulary, LmHeadGreedy());
}

template<uint32_t THREADS, uint32_t TILE>
__global__ __launch_bounds__(THREADS, 1) void ScalarSampledKernel(const uint16_t *normed, const uint16_t *weight, float *score, uint32_t *token, uint32_t hidden, uint32_t vocabulary, const LmHeadSampling sampling)
{
	LmHeadCandidateScalarBody<THREADS, TILE>(normed, weight, 0, score, token, hidden, vocabulary, LmHeadSampler{sampling});
}

static uint32_t random_state = 20260928u;

static uint32_t Random()
{
	random_state ^= random_state << 13u;
	random_state ^= random_state >> 17u;
	random_state ^= random_state << 5u;
	return random_state;
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

static void Compare(const float *score_a,const uint32_t *token_a,const float *score_b,const uint32_t *token_b,size_t count,const char *name)
{
	std::vector<float> sa=Download(score_a,count),sb=Download(score_b,count);
	std::vector<uint32_t> ta=Download(token_a,count),tb=Download(token_b,count);
	for (size_t index=0u; index<count; index++)
		if (memcmp(&sa[index],&sb[index],4u) != 0 || ta[index] != tb[index])
		{
			fprintf(stderr,"MISMATCH case=%s candidate=%zu staged=%.9g/%u scalar=%.9g/%u\n",name,index,sa[index],ta[index],sb[index],tb[index]);
			exit(1);
		}
}

static void Case(uint32_t hidden,uint32_t vocabulary,uint32_t rows,uint32_t use_ids,uint32_t ties,cudaStream_t stream)
{
	const uint32_t tiles=(vocabulary+STAGE_TILE-1u)/STAGE_TILE;
	std::vector<uint16_t> weight((uint64_t)vocabulary*hidden),normed((uint64_t)rows*hidden);
	std::vector<uint32_t> ids(vocabulary);
	std::vector<SparkRowSampling> rules(rows);
	std::vector<uint32_t> positions(rows);
	uint16_t *device_weight,*device_normed;
	uint32_t *device_ids,*device_positions,*token_a,*token_b;
	SparkRowSampling *device_rules;
	float *score_a,*score_b;
	LmHeadSampling sampling;
	char name[96];
	for (auto &value : weight) value=Bf16(((int32_t)(Random()%2049u)-1024)/16384.0f);
	for (auto &value : normed) value=Bf16(((int32_t)(Random()%2049u)-1024)/256.0f);
	if (ties != 0u)
		for (uint32_t token=1u; token<vocabulary; token+=97u)
			memcpy(&weight[(uint64_t)token*hidden],&weight[0],hidden*2u);
	for (uint32_t index=0u; index<vocabulary; index++) ids[index]=use_ids ? (index*7919u)%vocabulary : index;
	memset(rules.data(),0,rules.size()*sizeof(SparkRowSampling));
	for (uint32_t row=0u; row<rows; row++) { rules[row].inverse_temperature=row%2u ? 1.25f : 0.0f; rules[row].seed=0x5eed0000u+row; positions[row]=100u+row; }
	device_weight=Upload(weight); device_normed=Upload(normed); device_ids=Upload(ids); device_rules=Upload(rules); device_positions=Upload(positions);
	CUDA(cudaMalloc(&score_a,(uint64_t)rows*tiles*4u)); CUDA(cudaMalloc(&score_b,(uint64_t)rows*tiles*4u));
	CUDA(cudaMalloc(&token_a,(uint64_t)rows*tiles*4u)); CUDA(cudaMalloc(&token_b,(uint64_t)rows*tiles*4u));
	LmHeadCandidateKernel<STAGE_THREADS,STAGE_TILE><<<dim3(tiles,rows),STAGE_THREADS,0,stream>>>(device_normed,device_weight,use_ids ? device_ids : 0,score_a,token_a,rows,hidden,vocabulary);
	ScalarGreedyKernel<STAGE_THREADS,STAGE_TILE><<<dim3(tiles,rows),STAGE_THREADS,0,stream>>>(device_normed,device_weight,use_ids ? device_ids : 0,score_b,token_b,hidden,vocabulary);
	CUDA(cudaStreamSynchronize(stream));
	snprintf(name,sizeof(name),"greedy hidden=%u vocabulary=%u rows=%u ids=%u ties=%u",hidden,vocabulary,rows,use_ids,ties);
	Compare(score_a,token_a,score_b,token_b,(size_t)rows*tiles,name);
	sampling.rules=device_rules; sampling.positions=device_positions; sampling.rows=rows; sampling.token_offset=4096u;
	LmHeadSampledCandidateKernel<STAGE_THREADS,STAGE_TILE><<<dim3(tiles,rows),STAGE_THREADS,0,stream>>>(device_normed,device_weight,score_a,token_a,hidden,vocabulary,sampling);
	ScalarSampledKernel<STAGE_THREADS,STAGE_TILE><<<dim3(tiles,rows),STAGE_THREADS,0,stream>>>(device_normed,device_weight,score_b,token_b,hidden,vocabulary,sampling);
	CUDA(cudaStreamSynchronize(stream));
	snprintf(name,sizeof(name),"sampled hidden=%u vocabulary=%u rows=%u ties=%u",hidden,vocabulary,rows,ties);
	Compare(score_a,token_a,score_b,token_b,(size_t)rows*tiles,name);
	printf("PASS %s\n",name);
	CUDA(cudaFree(device_weight)); CUDA(cudaFree(device_normed)); CUDA(cudaFree(device_ids)); CUDA(cudaFree(device_rules)); CUDA(cudaFree(device_positions));
	CUDA(cudaFree(score_a)); CUDA(cudaFree(score_b)); CUDA(cudaFree(token_a)); CUDA(cudaFree(token_b));
}

static void Time(uint32_t hidden,uint32_t vocabulary,cudaStream_t stream)
{
	const uint32_t tiles=(vocabulary+STAGE_TILE-1u)/STAGE_TILE, repeats=20u;
	uint16_t *weight,*normed;
	float *score,ms[2];
	uint32_t *token,repeat,variant;
	cudaEvent_t start,stop;
	CUDA(cudaMalloc(&weight,(uint64_t)vocabulary*hidden*2u)); CUDA(cudaMemset(weight,0x11,(uint64_t)vocabulary*hidden*2u));
	CUDA(cudaMalloc(&normed,hidden*2u)); CUDA(cudaMemset(normed,0x22,hidden*2u));
	CUDA(cudaMalloc(&score,tiles*4u)); CUDA(cudaMalloc(&token,tiles*4u));
	CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&stop));
	for (variant=0u; variant<2u; variant++)
	{
		CUDA(cudaEventRecord(start,stream));
		for (repeat=0u; repeat<repeats; repeat++)
			if (variant == 0u)
				LmHeadCandidateKernel<STAGE_THREADS,STAGE_TILE><<<dim3(tiles,1u),STAGE_THREADS,0,stream>>>(normed,weight,0,score,token,1u,hidden,vocabulary);
			else
				ScalarGreedyKernel<STAGE_THREADS,STAGE_TILE><<<dim3(tiles,1u),STAGE_THREADS,0,stream>>>(normed,weight,0,score,token,hidden,vocabulary);
		CUDA(cudaEventRecord(stop,stream)); CUDA(cudaEventSynchronize(stop));
		CUDA(cudaEventElapsedTime(&ms[variant],start,stop));
	}
	printf("TIME head hidden=%u vocabulary=%u staged_us=%.1f scalar_us=%.1f staged_gbps=%.0f\n",hidden,vocabulary,1000.0f*ms[0]/repeats,1000.0f*ms[1]/repeats,(double)vocabulary*hidden*2.0/(1e6*ms[0]/repeats));
}

int main(int argc,char **argv)
{
	cudaStream_t stream;
	if (argc < 2 || (strcmp(argv[1],"--run") != 0 && strcmp(argv[1],"--time") != 0))
	{
		fprintf(stderr,"usage: test_head_candidate_stage --run|--time\n");
		return 2;
	}
	CUDA(cudaStreamCreate(&stream));
	if (strcmp(argv[1],"--time") == 0)
	{
		Time(4096u,9680u,stream);
		return 0;
	}
	Case(4096u,9680u,1u,0u,0u,stream);
	Case(4096u,9680u,3u,0u,1u,stream);
	Case(4096u,1000u,2u,1u,1u,stream);
	Case(192u,77u,2u,0u,0u,stream);
	Case(200u,300u,1u,0u,1u,stream);
	Case(64u,128u,1u,1u,0u,stream);
	printf("test_head_candidate_stage PASS\n");
	return 0;
}
