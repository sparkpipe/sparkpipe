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
static constexpr uint32_t flat = GLM5_NEXT_HC_FLAT, mixes = GLM5_NEXT_HC_MIX, hc = GLM5_NEXT_HC, hidden = GLM5_NEXT_HIDDEN, matrices = 90u, max_rows = 5u;

typedef struct HcPlanes
{
    float *mixes,*pre,*post,*comb;
    uint16_t *collapsed,*snapshot;
}
HcPlanes;

typedef struct HcHost
{
    std::vector<float> mixes,pre,post,comb;
    std::vector<uint16_t> collapsed,snapshot;
}
HcHost;

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

static float Bf16Value(uint16_t value)
{
    uint32_t bits = (uint32_t)value << 16u;
    float out;
    memcpy(&out,&bits,sizeof(out));
    return out;
}

static void Launch(bool production,cudaStream_t stream,const uint16_t *input,const float *weight,const float *scale,const float *base,const HcPlanes *planes,uint32_t rows)
{
    if (production)
        Glm5NextHcSiteKernel<<<rows*GLM5_NEXT_HC_CLUSTER,GLM5_NEXT_LAYER_THREADS,0u,stream>>>(input,weight,scale,base,planes->mixes,planes->pre,planes->post,planes->comb,planes->collapsed,planes->snapshot);
    else
    {
        Glm5NextHcMixBaselineKernel<<<rows,256u,4096u*sizeof(float),stream>>>(input,weight,planes->mixes,rows,flat,mixes,GLM5_NEXT_RMS_EPSILON);
        Glm5NextHcSinkhornBaselineKernel<<<(rows+63u)/64u,64u,0u,stream>>>(planes->mixes,scale,base,rows,hc,GLM5_NEXT_HC_SINKHORN_ITERATIONS,GLM5_NEXT_HC_EPSILON,planes->pre,planes->post,planes->comb);
        Glm5NextHcPreReduceBaselineKernel<<<rows,GLM5_NEXT_LAYER_THREADS,0u,stream>>>(input,planes->pre,planes->collapsed,planes->snapshot,rows,hc,hidden);
    }
    CUDA(cudaPeekAtLastError());
}

static void Fetch(const HcPlanes *planes,uint32_t rows,HcHost *host)
{
    host->mixes.resize(rows*mixes); host->pre.resize(rows*hc); host->post.resize(rows*hc); host->comb.resize(rows*hc*hc);
    host->collapsed.resize(rows*hidden); host->snapshot.resize(rows*flat);
    CUDA(cudaMemcpy(host->mixes.data(),planes->mixes,host->mixes.size()*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(host->pre.data(),planes->pre,host->pre.size()*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(host->post.data(),planes->post,host->post.size()*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(host->comb.data(),planes->comb,host->comb.size()*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(host->collapsed.data(),planes->collapsed,host->collapsed.size()*sizeof(uint16_t),cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(host->snapshot.data(),planes->snapshot,host->snapshot.size()*sizeof(uint16_t),cudaMemcpyDeviceToHost));
}

static void Bitwise(const void *actual,const void *expected,size_t bytes,const char *name,uint32_t dataset,uint32_t rows)
{
    if (memcmp(actual,expected,bytes) == 0)
        return;
    fprintf(stderr,"MISMATCH plane=%s dataset=%u rows=%u is not bitwise equal to the frozen kernel fed the same mixes\n",name,dataset,rows);
    exit(1);
}

static double MixError(const HcHost *actual,const std::vector<uint16_t> &input,const std::vector<float> &weights,uint32_t rows)
{
    double worst = 0.0, total, magnitude, square, value, inverse, error;
    for (uint32_t row=0u; row<rows; row++)
    {
        square = 0.0;
        for (uint32_t i=0u; i<flat; i++)
            square += (double)Bf16Value(input[row*flat+i])*Bf16Value(input[row*flat+i]);
        inverse = 1.0/std::sqrt(square/flat+GLM5_NEXT_RMS_EPSILON);
        for (uint32_t mix=0u; mix<mixes; mix++)
        {
            total = 0.0; magnitude = 0.0;
            for (uint32_t i=0u; i<flat; i++)
            {
                value = (double)weights[(uint64_t)mix*flat+i]*Bf16Value(input[row*flat+i]);
                total += value; magnitude += std::fabs(value);
            }
            error = std::fabs(actual->mixes[row*mixes+mix]-total*inverse)/(magnitude*inverse+1e-30);
            REQUIRE(std::isfinite(actual->mixes[row*mixes+mix]) && error <= 1e-5);
            worst = std::max(worst,error);
        }
    }
    return worst;
}

static void Compare(const HcHost *actual,const HcHost *replayed,const std::vector<uint16_t> &input,const std::vector<float> &weights,uint32_t dataset,uint32_t rows)
{
    double error = MixError(actual,input,weights,rows);
    Bitwise(actual->pre.data(),replayed->pre.data(),actual->pre.size()*sizeof(float),"pre",dataset,rows);
    Bitwise(actual->post.data(),replayed->post.data(),actual->post.size()*sizeof(float),"post",dataset,rows);
    Bitwise(actual->comb.data(),replayed->comb.data(),actual->comb.size()*sizeof(float),"comb",dataset,rows);
    Bitwise(actual->collapsed.data(),replayed->collapsed.data(),actual->collapsed.size()*sizeof(uint16_t),"collapsed",dataset,rows);
    Bitwise(actual->snapshot.data(),replayed->snapshot.data(),actual->snapshot.size()*sizeof(uint16_t),"snapshot",dataset,rows);
    printf("PASS dataset=%u rows=%u mix_error_over_l1=%.3g sinkhorn_and_collapse=bitwise_vs_frozen_on_same_mixes\n",dataset,rows,error);
}

static void Replay(cudaStream_t stream,const uint16_t *input,const float *scale,const float *base,const HcPlanes *actual,const HcPlanes *replayed,uint32_t rows)
{
    Glm5NextHcSinkhornBaselineKernel<<<(rows+63u)/64u,64u,0u,stream>>>(actual->mixes,scale,base,rows,hc,GLM5_NEXT_HC_SINKHORN_ITERATIONS,GLM5_NEXT_HC_EPSILON,replayed->pre,replayed->post,replayed->comb);
    Glm5NextHcPreReduceBaselineKernel<<<rows,GLM5_NEXT_LAYER_THREADS,0u,stream>>>(input,replayed->pre,replayed->collapsed,replayed->snapshot,rows,hc,hidden);
    CUDA(cudaPeekAtLastError());
}

static void Timing(cudaStream_t stream,const uint16_t *input,const float *weight,const float *scale,const float *base,const HcPlanes *planes,bool production,uint32_t distinct)
{
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    cudaEvent_t begin,end;
    std::vector<float> values;
    float ms;
    CUDA(cudaEventCreate(&begin));
    CUDA(cudaEventCreate(&end));
    CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    for (uint32_t i=0u; i<90u; i++)
        Launch(production,stream,input,weight+(uint64_t)(i%distinct)*flat*mixes,scale,base,planes,1u);
    CUDA(cudaStreamEndCapture(stream,&graph));
    CUDA(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0u));
    for (uint32_t i=0u; i<10u; i++)
    {
        CUDA(cudaEventRecord(begin,stream));
        CUDA(cudaGraphLaunch(executable,stream));
        CUDA(cudaEventRecord(end,stream));
        CUDA(cudaEventSynchronize(end));
        CUDA(cudaEventElapsedTime(&ms,begin,end));
        if (i >= 2u) values.push_back(ms);
    }
    std::sort(values.begin(),values.end());
    printf("TIMING production=%u rows=1 distinct_weights=%u sites=90 warmups=2 samples=8 min_ms=%.6f median_ms=%.6f max_ms=%.6f\n",
        production,distinct,values.front(),0.5f*(values[3]+values[4]),values.back());
    CUDA(cudaGraphExecDestroy(executable));
    CUDA(cudaGraphDestroy(graph));
    CUDA(cudaEventDestroy(begin));
    CUDA(cudaEventDestroy(end));
}

static void AllocatePlanes(HcPlanes *planes)
{
    CUDA(cudaMalloc(&planes->mixes,max_rows*mixes*sizeof(float)));
    CUDA(cudaMalloc(&planes->pre,max_rows*hc*sizeof(float)));
    CUDA(cudaMalloc(&planes->post,max_rows*hc*sizeof(float)));
    CUDA(cudaMalloc(&planes->comb,max_rows*hc*hc*sizeof(float)));
    CUDA(cudaMalloc(&planes->collapsed,max_rows*hidden*sizeof(uint16_t)));
    CUDA(cudaMalloc(&planes->snapshot,max_rows*flat*sizeof(uint16_t)));
}

static void Fill(uint32_t dataset,std::vector<uint16_t> *input,std::vector<float> *weights,std::vector<float> *scale,std::vector<float> *base)
{
    for (size_t i=0u; i<input->size(); i++)
    {
        float value=((int32_t)(Random()%2049u)-1024)/512.0f;
        if (dataset == 1u) value=std::ldexp(value,(int)(Random()%17u)-8);
        if (dataset == 2u) value=(i%4u<2u ? 1.0f : -1.0f)*(i%13u==0u ? 256.0f : 1.0f);
        (*input)[i]=Bf16(value);
    }
    for (size_t i=0u; i<weights->size(); i++)
    {
        float value=((int32_t)(Random()%2049u)-1024)/32768.0f;
        if (dataset == 1u) value=std::ldexp(value,(int)(Random()%17u)-8);
        if (dataset == 2u) value=(i%2u ? 1.0f : -1.0f)*(i%17u==0u ? 0.001f : 1.0f);
        (*weights)[i]=value;
    }
    for (size_t i=0u; i<scale->size(); i++)
        (*scale)[i]=0.25f+(float)(Random()%1024u)/1024.0f;
    for (size_t i=0u; i<base->size(); i++)
        (*base)[i]=((int32_t)(Random()%2049u)-1024)/1024.0f;
}

int main(int argc,char **argv)
{
    cudaStream_t stream;
    uint16_t *input;
    float *weights,*scale,*base;
    HcPlanes baseline_planes,actual_planes;
    HcHost baseline,actual;
    std::vector<uint16_t> host_input(max_rows*flat);
    std::vector<float> host_weights((uint64_t)matrices*flat*mixes),host_scale(3u),host_base(mixes);
    if (argc != 2 || strcmp(argv[1],"--run") != 0)
    {
        puts("usage: test_glm5_next_hc_mix --run");
        return 2;
    }
    alarm(120);
    setvbuf(stdout,nullptr,_IOLBF,0);
    CUDA(cudaSetDevice(0));
    CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    CUDA(cudaMalloc(&input,host_input.size()*sizeof(uint16_t)));
    CUDA(cudaMalloc(&weights,host_weights.size()*sizeof(float)));
    CUDA(cudaMalloc(&scale,host_scale.size()*sizeof(float)));
    CUDA(cudaMalloc(&base,host_base.size()*sizeof(float)));
    AllocatePlanes(&baseline_planes);
    AllocatePlanes(&actual_planes);
    for (uint32_t dataset=0u; dataset<3u; dataset++)
    {
        Fill(dataset,&host_input,&host_weights,&host_scale,&host_base);
        CUDA(cudaMemcpy(input,host_input.data(),host_input.size()*sizeof(uint16_t),cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(weights,host_weights.data(),host_weights.size()*sizeof(float),cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(scale,host_scale.data(),host_scale.size()*sizeof(float),cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(base,host_base.data(),host_base.size()*sizeof(float),cudaMemcpyHostToDevice));
        for (uint32_t rows : {1u,3u,5u})
        {
            CUDA(cudaMemsetAsync(actual_planes.collapsed,0xff,max_rows*hidden*sizeof(uint16_t),stream));
            Launch(true,stream,input,weights,scale,base,&actual_planes,rows);
            Replay(stream,input,scale,base,&actual_planes,&baseline_planes,rows);
            CUDA(cudaStreamSynchronize(stream));
            Fetch(&baseline_planes,rows,&baseline);
            Fetch(&actual_planes,rows,&actual);
            Compare(&actual,&baseline,host_input,host_weights,dataset,rows);
        }
    }
    for (uint32_t distinct : {1u,90u})
        for (bool production : {false,true})
            Timing(stream,input,weights,scale,base,&actual_planes,production,distinct);
    CUDA(cudaStreamDestroy(stream));
    puts("PASS hc_site_probe numerical_cases=9 mixes=f64_reference sinkhorn_collapse=frozen_bitwise production_kernel=actual_fused_cluster");
    return 0;
}
