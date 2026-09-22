#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>
#include "modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_cuda.cu"
#include "tests/fixtures/glm5_next_hc_mix_baseline.cuh"

#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { \
    fprintf(stderr,"FAIL line=%d cuda=%s call=%s\n",__LINE__,cudaGetErrorString(e),#call); \
    exit(1); } } while(0)
#define REQUIRE(test) do { if(!(test)) { \
    fprintf(stderr,"FAIL line=%d test=%s\n",__LINE__,#test); exit(1); } } while(0)

static uint32_t random_state = 719u;
static constexpr uint32_t flat = 16384u,mixes = 24u,matrices = 90u;

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
    return (uint16_t)(bits >> 16u);
}

static void Launch(bool production,cudaStream_t stream,const uint16_t *input,
    const float *weight,float *output,uint32_t rows)
{
    if (production)
        Glm5NextHcMixKernel<<<dim3(rows,GLM5_NEXT_HC_MIX_BLOCKS),256u,4096u*sizeof(float),stream>>>(
            input,weight,output,rows,flat,mixes,1e-5f);
    else
        Glm5NextHcMixBaselineKernel<<<rows,256u,4096u*sizeof(float),stream>>>(
            input,weight,output,rows,flat,mixes,1e-5f);
    CUDA(cudaPeekAtLastError());
}

static void Timing(cudaStream_t stream,uint16_t *input,float *weight,float *output,
    bool production,uint32_t distinct)
{
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    cudaEvent_t begin,end;
    CUDA(cudaEventCreate(&begin));
    CUDA(cudaEventCreate(&end));
    CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    for (uint32_t i=0u; i<90u; i++)
        Launch(production,stream,input,weight+(uint64_t)(i%distinct)*flat*mixes,output,1u);
    CUDA(cudaStreamEndCapture(stream,&graph));
    CUDA(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0u));
    std::vector<float> values;
    for (uint32_t i=0u; i<10u; i++)
    {
        CUDA(cudaEventRecord(begin,stream));
        CUDA(cudaGraphLaunch(executable,stream));
        CUDA(cudaEventRecord(end,stream));
        CUDA(cudaEventSynchronize(end));
        float ms;
        CUDA(cudaEventElapsedTime(&ms,begin,end));
        if (i >= 2u) values.push_back(ms);
    }
    std::sort(values.begin(),values.end());
    printf("TIMING production=%u rows=1 distinct_weights=%u launches=90 warmups=2 samples=8 "
           "min_ms=%.6f median_ms=%.6f max_ms=%.6f\n",production,distinct,
           values.front(),0.5f*(values[3]+values[4]),values.back());
    CUDA(cudaGraphExecDestroy(executable));
    CUDA(cudaGraphDestroy(graph));
    CUDA(cudaEventDestroy(begin));
    CUDA(cudaEventDestroy(end));
}

int main(int argc,char **argv)
{
    if (argc != 2 || strcmp(argv[1],"--run") != 0)
    {
        puts("usage: test_glm5_next_hc_mix --run");
        return 2;
    }
    alarm(60);
    setvbuf(stdout,nullptr,_IOLBF,0);
    CUDA(cudaSetDevice(0));
    cudaStream_t stream;
    CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    uint16_t *input;
    float *weights,*output;
    CUDA(cudaMalloc(&input,5u*flat*sizeof(uint16_t)));
    CUDA(cudaMalloc(&weights,(uint64_t)matrices*flat*mixes*sizeof(float)));
    CUDA(cudaMalloc(&output,5u*mixes*sizeof(float)));
    std::vector<uint16_t> host_input(5u*flat);
    std::vector<float> host_weights((uint64_t)matrices*flat*mixes);
    std::vector<float> baseline(5u*mixes),actual(5u*mixes);
    for (uint32_t dataset=0u; dataset<3u; dataset++)
    {
        for (size_t i=0u; i<host_input.size(); i++)
        {
            float value=((int32_t)(Random()%2049u)-1024)/512.0f;
            if (dataset == 1u) value=std::ldexp(value,(int)(Random()%17u)-8);
            if (dataset == 2u) value=(i%4u<2u ? 1.0f : -1.0f)*(i%13u==0u ? 256.0f : 1.0f);
            host_input[i]=Bf16(value);
        }
        for (size_t i=0u; i<host_weights.size(); i++)
        {
            float value=((int32_t)(Random()%2049u)-1024)/32768.0f;
            if (dataset == 1u) value=std::ldexp(value,(int)(Random()%17u)-8);
            if (dataset == 2u) value=(i%2u ? 1.0f : -1.0f)*(i%17u==0u ? 0.001f : 1.0f);
            host_weights[i]=value;
        }
        CUDA(cudaMemcpy(input,host_input.data(),host_input.size()*sizeof(uint16_t),cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(weights,host_weights.data(),host_weights.size()*sizeof(float),cudaMemcpyHostToDevice));
        for (uint32_t rows : {1u,3u,5u})
        {
            Launch(false,stream,input,weights,output,rows);
            CUDA(cudaStreamSynchronize(stream));
            CUDA(cudaMemcpy(baseline.data(),output,rows*mixes*sizeof(float),cudaMemcpyDeviceToHost));
            CUDA(cudaMemset(output,0xff,rows*mixes*sizeof(float)));
            Launch(true,stream,input,weights,output,rows);
            CUDA(cudaStreamSynchronize(stream));
            CUDA(cudaMemcpy(actual.data(),output,rows*mixes*sizeof(float),cudaMemcpyDeviceToHost));
            for (uint32_t i=0u; i<rows*mixes; i++)
            {
                REQUIRE(std::isfinite(actual[i]) && std::isfinite(baseline[i]));
                if (memcmp(&actual[i],&baseline[i],sizeof(float)) != 0)
                {
                    fprintf(stderr,"MISMATCH dataset=%u rows=%u output=%u actual=%.9g baseline=%.9g\n",
                        dataset,rows,i,actual[i],baseline[i]);
                    return 1;
                }
            }
            printf("PASS bitwise dataset=%u rows=%u outputs=%u\n",dataset,rows,rows*mixes);
        }
    }
    for (uint32_t distinct : {1u,90u})
        for (bool production : {false,true})
            Timing(stream,input,weights,output,production,distinct);
    CUDA(cudaFree(output));
    CUDA(cudaFree(weights));
    CUDA(cudaFree(input));
    CUDA(cudaStreamDestroy(stream));
    puts("PASS hc_mix_probe numerical_cases=9 baseline=frozen_pre_partition production_kernel=actual");
    return 0;
}
