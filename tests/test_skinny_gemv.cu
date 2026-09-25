#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>
#include "modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_cuda.cu"

#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { \
    fprintf(stderr,"FAIL line=%d cuda=%s call=%s\n",__LINE__,cudaGetErrorString(e),#call); \
    exit(1); } } while(0)
#define REQUIRE(test) do { if(!(test)) { \
    fprintf(stderr,"FAIL line=%d test=%s\n",__LINE__,#test); exit(1); } } while(0)

typedef struct DenseShape
{
    const char *name;
    uint32_t input,output,f32;
}
DenseShape;

static const DenseShape dense_shapes[] =
{
    {"kda_qkv_beta",4096u,1540u,0u},{"kda_decay_gate_down",4096u,256u,0u},{"kda_decay_up",128u,512u,0u},{"kda_out",512u,4096u,0u},
    {"mla_q_a",4096u,1536u,0u},{"mla_q_b",1536u,1024u,0u},{"mla_kv_a",4096u,512u,0u},{"index_q",1536u,4096u,0u},
    {"index_k",4096u,128u,0u},{"index_head",4096u,32u,0u},{"attn_out",1024u,4096u,0u},{"router",4096u,288u,1u},
    {"shared_gate_up",4096u,256u,0u},{"shared_down",128u,4096u,0u},{"dense_gate_up",4096u,1536u,0u},{"dense_down",768u,4096u,0u},
    {"mtp_eh_proj",8192u,4096u,0u},{"odd_tail",96u,37u,0u},{"odd_small_k",128u,4097u,1u}
};

static uint32_t random_state = 20260924u;

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

static float Bf16Value(uint16_t value)
{
    uint32_t bits=(uint32_t)value<<16u;
    float out;
    memcpy(&out,&bits,sizeof(out));
    return out;
}

static float E4m3Value(uint8_t value)
{
    int32_t exponent=(value>>3u)&15, mantissa=value&7;
    float magnitude=exponent == 0 ? std::ldexp((float)mantissa,-9) : std::ldexp(1.0f+mantissa/8.0f,exponent-7);
    return (value & 0x80u) ? -magnitude : magnitude;
}

static void Check(double actual,double expected,double magnitude,const char *name,uint32_t row,uint32_t neuron)
{
    double bound=std::ldexp(std::fabs(expected),-8)+1e-5*magnitude+1e-30;
    if (std::isfinite(actual) && std::fabs(actual-expected) <= bound)
        return;
    fprintf(stderr,"MISMATCH shape=%s row=%u neuron=%u actual=%.9g expected=%.9g bound=%.3g\n",name,row,neuron,actual,expected,bound);
    exit(1);
}

static void DenseCase(const DenseShape *shape,uint32_t rows,cudaStream_t stream)
{
    std::vector<uint16_t> weight((uint64_t)shape->output*shape->input),activation((uint64_t)rows*shape->input),out16((uint64_t)rows*shape->output);
    std::vector<float> out32((uint64_t)rows*shape->output);
    uint16_t *device_weight,*device_activation,*device_out16;
    float *device_out32;
    double total,magnitude,term;
    for (auto &value : weight) value=Bf16(Signed()*0.0625f);
    for (auto &value : activation) value=Bf16(Signed()*4.0f);
    CUDA(cudaMalloc(&device_weight,weight.size()*2u));
    CUDA(cudaMalloc(&device_activation,activation.size()*2u));
    CUDA(cudaMalloc(&device_out16,out16.size()*2u));
    CUDA(cudaMalloc(&device_out32,out32.size()*4u));
    CUDA(cudaMemcpy(device_weight,weight.data(),weight.size()*2u,cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(device_activation,activation.data(),activation.size()*2u,cudaMemcpyHostToDevice));
    REQUIRE(LmSkinnyDense<LmBf16Format>(device_weight,device_activation,shape->f32 ? 0 : device_out16,shape->f32 ? device_out32 : 0,rows,shape->input,shape->output,0u,0u,stream) == LM_LAUNCH_OK);
    CUDA(cudaStreamSynchronize(stream));
    CUDA(cudaMemcpy(out16.data(),device_out16,out16.size()*2u,cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(out32.data(),device_out32,out32.size()*4u,cudaMemcpyDeviceToHost));
    for (uint32_t row=0u; row<rows; row++)
        for (uint32_t neuron=0u; neuron<shape->output; neuron++)
        {
            total=0.0; magnitude=0.0;
            for (uint32_t k=0u; k<shape->input; k++)
            {
                term=(double)Bf16Value(weight[(uint64_t)neuron*shape->input+k])*Bf16Value(activation[(uint64_t)row*shape->input+k]);
                total+=term; magnitude+=std::fabs(term);
            }
            Check(shape->f32 ? out32[(uint64_t)row*shape->output+neuron] : Bf16Value(out16[(uint64_t)row*shape->output+neuron]),total,magnitude,shape->name,row,neuron);
        }
    CUDA(cudaFree(device_weight)); CUDA(cudaFree(device_activation)); CUDA(cudaFree(device_out16)); CUDA(cudaFree(device_out32));
}

static void ExpertReference(const std::vector<uint8_t> &weight,const std::vector<float> &scale,const std::vector<uint16_t> &activation,uint32_t expert,uint32_t neuron,uint32_t input,uint32_t output,uint32_t row,double *total,double *magnitude)
{
    double term;
    *total=0.0; *magnitude=0.0;
    for (uint32_t k=0u; k<input; k++)
    {
        term=(double)E4m3Value(weight[((uint64_t)expert*output+neuron)*input+k])*scale[((uint64_t)expert*output+neuron)*(input/128u)+k/128u]*Bf16Value(activation[(uint64_t)row*input+k]);
        *total+=term; *magnitude+=std::fabs(term);
    }
}

static void ExpertCase(uint32_t input,uint32_t output,uint32_t tokens,uint32_t activation_packed,cudaStream_t stream)
{
    const uint32_t experts=288u, top_k=8u, pairs=tokens*top_k, activation_rows=activation_packed ? pairs : tokens;
    std::vector<uint8_t> weight((uint64_t)experts*output*input);
    std::vector<float> scale((uint64_t)experts*output*(input/128u));
    std::vector<uint16_t> activation((uint64_t)activation_rows*input),out((uint64_t)pairs*output);
    std::vector<uint32_t> route(pairs),packed(pairs);
    uint8_t *device_weight; float *device_scale; uint16_t *device_activation,*device_out; uint32_t *device_route,*device_packed;
    double total,magnitude;
    for (auto &value : weight) { value=(uint8_t)(Random()&0xffu); if ((value&0x7fu) == 0x7fu) value^=1u; }
    for (auto &value : scale) value=std::ldexp(1.0f+(Random()%256u)/256.0f,-10);
    for (auto &value : activation) value=Bf16(Signed()*4.0f);
    for (uint32_t pair=0u; pair<pairs; pair++) { route[pair]=pair == 1u ? route[0] : Random()%experts; packed[pair]=(pair*5u+3u)%pairs; }
    CUDA(cudaMalloc(&device_weight,weight.size())); CUDA(cudaMalloc(&device_scale,scale.size()*4u)); CUDA(cudaMalloc(&device_activation,activation.size()*2u));
    CUDA(cudaMalloc(&device_out,out.size()*2u)); CUDA(cudaMalloc(&device_route,pairs*4u)); CUDA(cudaMalloc(&device_packed,pairs*4u));
    CUDA(cudaMemcpy(device_weight,weight.data(),weight.size(),cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(device_scale,scale.data(),scale.size()*4u,cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(device_activation,activation.data(),activation.size()*2u,cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(device_route,route.data(),pairs*4u,cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(device_packed,packed.data(),pairs*4u,cudaMemcpyHostToDevice));
    REQUIRE(LmSkinnyExperts<LmFp8>(device_weight,LmWeightCodecScaleTensor<SPARK_WEIGHT_CODEC_FP8_E4M3>(device_scale,experts,output,input),device_activation,device_out,device_route,device_packed,pairs,top_k,activation_packed,input,output,stream) == LM_LAUNCH_OK);
    CUDA(cudaStreamSynchronize(stream));
    CUDA(cudaMemcpy(out.data(),device_out,out.size()*2u,cudaMemcpyDeviceToHost));
    for (uint32_t pair=0u; pair<pairs; pair++)
        for (uint32_t neuron=0u; neuron<output; neuron++)
        {
            ExpertReference(weight,scale,activation,route[pair],neuron,input,output,activation_packed ? packed[pair] : pair/top_k,&total,&magnitude);
            Check(Bf16Value(out[(uint64_t)packed[pair]*output+neuron]),total,magnitude,activation_packed ? "expert_w2" : "expert_w1",pair,neuron);
        }
    CUDA(cudaFree(device_weight)); CUDA(cudaFree(device_scale)); CUDA(cudaFree(device_activation)); CUDA(cudaFree(device_out)); CUDA(cudaFree(device_route)); CUDA(cudaFree(device_packed));
}

typedef struct GroupedRoute
{
    uint32_t *route,*offset,*packed,*source,*prefix_up,*prefix_down;
    std::vector<uint32_t> host_offset,host_source;
}
GroupedRoute;

static void GroupedRouteBuild(GroupedRoute *route,uint32_t tokens,uint32_t input,uint32_t output,uint32_t skew,cudaStream_t stream)
{
    const uint32_t experts=288u,top_k=8u,pairs=tokens*top_k;
    std::vector<uint32_t> expert(pairs);
    for (uint32_t pair=0u; pair<pairs; pair++)
        expert[pair]=(pair%top_k) == 0u && pair/top_k < skew ? 5u : Random()%experts;
    CUDA(cudaMalloc(&route->route,pairs*4u)); CUDA(cudaMalloc(&route->offset,(experts+1u)*4u)); CUDA(cudaMalloc(&route->packed,pairs*4u));
    CUDA(cudaMalloc(&route->source,pairs*4u)); CUDA(cudaMalloc(&route->prefix_up,(experts+1u)*4u)); CUDA(cudaMalloc(&route->prefix_down,(experts+1u)*4u));
    CUDA(cudaMemcpy(route->route,expert.data(),pairs*4u,cudaMemcpyHostToDevice));
    REQUIRE((LmRouteBuild<256u,288u>(route->route,tokens,pairs,top_k,route->offset,route->packed,route->source,input == 4096u ? output : input,input == 4096u ? input : output,GLM5_NEXT_LAYER_TILE_N,route->prefix_up,route->prefix_down,stream)) == LM_LAUNCH_OK);
    CUDA(cudaStreamSynchronize(stream));
    route->host_offset.resize(experts+1u); route->host_source.resize(pairs);
    CUDA(cudaMemcpy(route->host_offset.data(),route->offset,(experts+1u)*4u,cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(route->host_source.data(),route->source,pairs*4u,cudaMemcpyDeviceToHost));
}

static void GroupedRouteFree(GroupedRoute *route)
{
    CUDA(cudaFree(route->route)); CUDA(cudaFree(route->offset)); CUDA(cudaFree(route->packed));
    CUDA(cudaFree(route->source)); CUDA(cudaFree(route->prefix_up)); CUDA(cudaFree(route->prefix_down));
}

static void GroupedExpertCase(uint32_t input,uint32_t output,uint32_t tokens,uint32_t activation_packed,uint32_t skew,cudaStream_t stream)
{
    const uint32_t experts=288u, top_k=8u, pairs=tokens*top_k, activation_rows=activation_packed ? pairs : tokens;
    std::vector<uint8_t> weight((uint64_t)experts*output*input);
    std::vector<float> scale((uint64_t)experts*output*(input/128u));
    std::vector<uint16_t> activation((uint64_t)activation_rows*input),out((uint64_t)pairs*output);
    uint8_t *device_weight; float *device_scale; uint16_t *device_activation,*device_out;
    GroupedRoute route;
    double total,magnitude;
    uint32_t widest=0u;
    for (auto &value : weight) { value=(uint8_t)(Random()&0xffu); if ((value&0x7fu) == 0x7fu) value^=1u; }
    for (auto &value : scale) value=std::ldexp(1.0f+(Random()%256u)/256.0f,-10);
    for (auto &value : activation) value=Bf16(Signed()*4.0f);
    GroupedRouteBuild(&route,tokens,input,output,skew,stream);
    CUDA(cudaMalloc(&device_weight,weight.size())); CUDA(cudaMalloc(&device_scale,scale.size()*4u)); CUDA(cudaMalloc(&device_activation,activation.size()*2u)); CUDA(cudaMalloc(&device_out,out.size()*2u));
    CUDA(cudaMemcpy(device_weight,weight.data(),weight.size(),cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(device_scale,scale.data(),scale.size()*4u,cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(device_activation,activation.data(),activation.size()*2u,cudaMemcpyHostToDevice));
    CUDA(cudaMemset(device_out,0xff,out.size()*2u));
    std::vector<uint16_t> single((uint64_t)pairs*output);
    REQUIRE((LmSkinnyGroupedExpertsWith<LmFp8,1u>(device_weight,LmWeightCodecScaleTensor<SPARK_WEIGHT_CODEC_FP8_E4M3>(device_scale,experts,output,input),device_activation,device_out,route.offset,route.source,experts,pairs,activation_packed,input,output,stream)) == LM_LAUNCH_OK);
    CUDA(cudaStreamSynchronize(stream));
    CUDA(cudaMemcpy(single.data(),device_out,single.size()*2u,cudaMemcpyDeviceToHost));
    CUDA(cudaMemset(device_out,0xff,out.size()*2u));
    REQUIRE(LmSkinnyGroupedExperts<LmFp8>(device_weight,LmWeightCodecScaleTensor<SPARK_WEIGHT_CODEC_FP8_E4M3>(device_scale,experts,output,input),device_activation,device_out,route.offset,route.source,experts,pairs,activation_packed,input,output,stream) == LM_LAUNCH_OK);
    CUDA(cudaStreamSynchronize(stream));
    CUDA(cudaMemcpy(out.data(),device_out,out.size()*2u,cudaMemcpyDeviceToHost));
    REQUIRE(memcmp(single.data(),out.data(),out.size()*2u) == 0);
    for (uint32_t expert=0u; expert<experts; expert++)
    {
        widest=std::max(widest,route.host_offset[expert+1u]-route.host_offset[expert]);
        for (uint32_t row=route.host_offset[expert]; row<route.host_offset[expert+1u]; row++)
            for (uint32_t neuron=0u; neuron<output; neuron++)
            {
                ExpertReference(weight,scale,activation,expert,neuron,input,output,activation_packed ? row : route.host_source[row],&total,&magnitude);
                Check(Bf16Value(out[(uint64_t)row*output+neuron]),total,magnitude,activation_packed ? "grouped_w2" : "grouped_w1",row,neuron);
            }
    }
    REQUIRE(route.host_offset[experts] == pairs);
    REQUIRE(skew == 0u || widest >= skew);
    GroupedRouteFree(&route);
    CUDA(cudaFree(device_weight)); CUDA(cudaFree(device_scale)); CUDA(cudaFree(device_activation)); CUDA(cudaFree(device_out));
}

static float GroupedTime(uint32_t neurons,uint32_t input,uint32_t output,uint32_t tokens,uint32_t activation_packed,const uint8_t *weight,const float *scale,const uint16_t *activation,uint16_t *out,const GroupedRoute *route,uint32_t multiprocessors,cudaStream_t stream)
{
    const uint32_t experts=288u,top_k=8u,pairs=tokens*top_k;
    cudaGraph_t graph; cudaGraphExec_t executable; cudaEvent_t begin,end;
    std::vector<float> samples;
    LmGemmArguments gemm;
    float ms;
    memset(&gemm,0,sizeof(gemm));
    gemm.scale_a=LmScaleTensorNone(); gemm.scale_b=LmWeightCodecScaleTensor<SPARK_WEIGHT_CODEC_FP8_E4M3>(scale,experts,output,input);
    gemm.prefix_built=1u; gemm.group_row_offset=route->offset; gemm.group_tile_prefix=input == 4096u ? route->prefix_up : route->prefix_down; gemm.output_bf16=out;
    gemm.source_row_map=activation_packed ? 0 : route->source; gemm.source_row_count=tokens;
    CUDA(cudaEventCreate(&begin)); CUDA(cudaEventCreate(&end));
    CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    for (uint32_t i=0u; i<16u; i++)
        if (neurons == 1u)
            REQUIRE((LmSkinnyGroupedExpertsWith<LmFp8,1u>(weight,gemm.scale_b,activation,out,route->offset,route->source,experts,pairs,activation_packed,input,output,stream)) == LM_LAUNCH_OK);
        else if (neurons == 2u)
            REQUIRE((LmSkinnyGroupedExpertsWith<LmFp8,2u>(weight,gemm.scale_b,activation,out,route->offset,route->source,experts,pairs,activation_packed,input,output,stream)) == LM_LAUNCH_OK);
        else if (neurons != 0u)
            REQUIRE(LmSkinnyGroupedExperts<LmFp8>(weight,gemm.scale_b,activation,out,route->offset,route->source,experts,pairs,activation_packed,input,output,stream) == LM_LAUNCH_OK);
        else if (activation_packed)
            REQUIRE((LmGemmWeightOnlyLaunch<LmFp8,GLM5_NEXT_LAYER_TILE_N,GLM5_NEXT_LAYER_STAGES,GLM5_NEXT_LAYER_WARPS>(&gemm,activation,weight,pairs,tokens,top_k,experts,input,output,multiprocessors,true,stream)) == LM_LAUNCH_OK);
        else
            REQUIRE((LmGemmWeightOnlyIndirectLaunch<LmFp8,GLM5_NEXT_LAYER_TILE_N,GLM5_NEXT_LAYER_STAGES,GLM5_NEXT_LAYER_WARPS>(&gemm,activation,weight,pairs,tokens,top_k,experts,input,output,multiprocessors,stream)) == LM_LAUNCH_OK);
    CUDA(cudaStreamEndCapture(stream,&graph));
    CUDA(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0u));
    for (uint32_t i=0u; i<8u; i++)
    {
        CUDA(cudaEventRecord(begin,stream)); CUDA(cudaGraphLaunch(executable,stream)); CUDA(cudaEventRecord(end,stream));
        CUDA(cudaEventSynchronize(end)); CUDA(cudaEventElapsedTime(&ms,begin,end));
        if (i >= 2u) samples.push_back(ms/16.0f);
    }
    std::sort(samples.begin(),samples.end());
    CUDA(cudaGraphExecDestroy(executable)); CUDA(cudaGraphDestroy(graph)); CUDA(cudaEventDestroy(begin)); CUDA(cudaEventDestroy(end));
    return samples[samples.size()/2u];
}

static void GroupedTiming(uint32_t input,uint32_t output,uint32_t tokens,uint32_t activation_packed,uint32_t multiprocessors,cudaStream_t stream)
{
    const uint32_t experts=288u,pairs=tokens*8u;
    const uint64_t expert_bytes=(uint64_t)output*input+(uint64_t)output*(input/128u)*4u;
    uint8_t *weight; float *scale; uint16_t *activation,*out;
    GroupedRoute route;
    uint32_t distinct=0u;
    float skinny,single,pair,tensor;
    GroupedRouteBuild(&route,tokens,input,output,0u,stream);
    for (uint32_t expert=0u; expert<experts; expert++)
        distinct+=route.host_offset[expert+1u] != route.host_offset[expert] ? 1u : 0u;
    CUDA(cudaMalloc(&weight,(uint64_t)experts*output*input)); CUDA(cudaMemset(weight,0x38,(uint64_t)experts*output*input));
    CUDA(cudaMalloc(&scale,(uint64_t)experts*output*(input/128u)*4u)); CUDA(cudaMemset(scale,0,(uint64_t)experts*output*(input/128u)*4u));
    CUDA(cudaMalloc(&activation,(uint64_t)pairs*input*2u)); CUDA(cudaMemset(activation,0x3c,(uint64_t)pairs*input*2u));
    CUDA(cudaMalloc(&out,(uint64_t)pairs*output*2u));
    skinny=GroupedTime(LM_SKINNY_GROUPED_NEURONS,input,output,tokens,activation_packed,weight,scale,activation,out,&route,multiprocessors,stream);
    single=GroupedTime(1u,input,output,tokens,activation_packed,weight,scale,activation,out,&route,multiprocessors,stream);
    pair=GroupedTime(2u,input,output,tokens,activation_packed,weight,scale,activation,out,&route,multiprocessors,stream);
    tensor=GroupedTime(0u,input,output,tokens,activation_packed,weight,scale,activation,out,&route,multiprocessors,stream);
    printf("TIMING grouped_experts %s tokens=%u distinct=%u skinny_ms=%.4f skinny_gbps=%.1f one_neuron_ms=%.4f one_neuron_gbps=%.1f two_neuron_ms=%.4f two_neuron_gbps=%.1f tensor_core_ms=%.4f tensor_core_gbps=%.1f speedup_vs_one_neuron=%.2f speedup_vs_tensor_core=%.2f\n",activation_packed ? "w2" : "w1",tokens,distinct,skinny,distinct*expert_bytes/(skinny*1e6),single,distinct*expert_bytes/(single*1e6),pair,distinct*expert_bytes/(pair*1e6),tensor,distinct*expert_bytes/(tensor*1e6),single/skinny,tensor/skinny);
    GroupedRouteFree(&route);
    CUDA(cudaFree(weight)); CUDA(cudaFree(scale)); CUDA(cudaFree(activation)); CUDA(cudaFree(out));
}

static int32_t TensorCoreLinear(const uint16_t *activation,const uint16_t *weight,uint16_t *output,const uint32_t *row_offset,uint32_t *tile_prefix,uint32_t input,uint32_t neurons,cudaStream_t stream)
{
    LmGemmArguments gemm;
    memset(&gemm,0,sizeof(gemm));
    gemm.scale_a=LmScaleTensorNone(); gemm.scale_b=LmScaleTensorNone();
    gemm.group_row_offset=row_offset; gemm.group_tile_prefix=tile_prefix; gemm.output_bf16=output;
    return LmGemmLaunch<LmBf16Format,GLM5_NEXT_LAYER_TILE_N,LmBf16Format::kTileK,GLM5_NEXT_LAYER_STAGES,GLM5_NEXT_LAYER_WARPS>(&gemm,activation,weight,1u,1u,1u,1u,input,neurons,48u,false,stream);
}

static double Timing(const DenseShape *shape,bool skinny,uint16_t *weights,uint16_t *activation,uint16_t *output,const uint32_t *row_offset,uint32_t *tile_prefix,uint32_t matrices,cudaStream_t stream)
{
    const uint64_t bytes=(uint64_t)shape->input*shape->output*2u;
    cudaGraph_t graph; cudaGraphExec_t executable; cudaEvent_t begin,end;
    std::vector<float> values;
    float ms;
    CUDA(cudaEventCreate(&begin)); CUDA(cudaEventCreate(&end));
    CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    for (uint32_t i=0u; i<matrices; i++)
        REQUIRE((skinny ? LmSkinnyDense<LmBf16Format>(weights+(uint64_t)i*shape->input*shape->output,activation,output,0,1u,shape->input,shape->output,0u,0u,stream)
            : TensorCoreLinear(activation,weights+(uint64_t)i*shape->input*shape->output,output,row_offset,tile_prefix,shape->input,shape->output,stream)) == LM_LAUNCH_OK);
    CUDA(cudaStreamEndCapture(stream,&graph));
    CUDA(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0u));
    for (uint32_t i=0u; i<10u; i++)
    {
        CUDA(cudaEventRecord(begin,stream)); CUDA(cudaGraphLaunch(executable,stream)); CUDA(cudaEventRecord(end,stream));
        CUDA(cudaEventSynchronize(end)); CUDA(cudaEventElapsedTime(&ms,begin,end));
        if (i >= 2u) values.push_back(ms);
    }
    std::sort(values.begin(),values.end());
    CUDA(cudaGraphExecDestroy(executable)); CUDA(cudaGraphDestroy(graph)); CUDA(cudaEventDestroy(begin)); CUDA(cudaEventDestroy(end));
    return (double)bytes*matrices/(0.5*(values[3]+values[4])*1e-3)/1e9;
}

static void TimeShape(const DenseShape *shape,cudaStream_t stream)
{
    const uint32_t matrices=std::max(8u,(uint32_t)((512ull<<20)/((uint64_t)shape->input*shape->output*2u)));
    const uint32_t offsets[2]={0u,1u};
    uint16_t *weights,*activation,*output; uint32_t *row_offset,*tile_prefix;
    double skinny,tensor;
    CUDA(cudaMalloc(&weights,(uint64_t)matrices*shape->input*shape->output*2u)); CUDA(cudaMemset(weights,0x3c,(uint64_t)matrices*shape->input*shape->output*2u));
    CUDA(cudaMalloc(&activation,shape->input*2u)); CUDA(cudaMemset(activation,0x3c,shape->input*2u));
    CUDA(cudaMalloc(&output,shape->output*4u)); CUDA(cudaMalloc(&row_offset,8u)); CUDA(cudaMalloc(&tile_prefix,8u));
    CUDA(cudaMemcpy(row_offset,offsets,8u,cudaMemcpyHostToDevice));
    skinny=Timing(shape,true,weights,activation,output,row_offset,tile_prefix,matrices,stream);
    tensor=Timing(shape,false,weights,activation,output,row_offset,tile_prefix,matrices,stream);
    printf("TIMING shape=%s k=%u n=%u matrices=%u skinny_gbps=%.1f tensor_core_gbps=%.1f speedup=%.2f\n",shape->name,shape->input,shape->output,matrices,skinny,tensor,skinny/tensor);
    CUDA(cudaFree(weights)); CUDA(cudaFree(activation)); CUDA(cudaFree(output)); CUDA(cudaFree(row_offset)); CUDA(cudaFree(tile_prefix));
}

static float TopkTime(bool warp,const float *logits,uint32_t *indices,float *values,const float *bias,cudaStream_t stream)
{
    cudaGraph_t graph; cudaGraphExec_t executable; cudaEvent_t begin,end;
    std::vector<float> samples;
    float ms;
    CUDA(cudaEventCreate(&begin)); CUDA(cudaEventCreate(&end));
    CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    for (uint32_t i=0u; i<42u; i++)
        if (warp)
            CUDA((LmTopkRouteLaunch<256u,8u,true,LM_TOPK_SCORE_SIGMOID>(1u,logits,288u,indices,values,bias,0,2.5f,stream)));
        else
            LmTopkSmallKernel<256u,8u,true,1u,1u,LM_TOPK_SCORE_SIGMOID><<<1u,256u,2u*LM_TOPK_SMALL_LIMIT*sizeof(uint32_t),stream>>>(logits,288u,indices,values,bias,0,2.5f);
    CUDA(cudaStreamEndCapture(stream,&graph));
    CUDA(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0u));
    for (uint32_t i=0u; i<10u; i++)
    {
        CUDA(cudaEventRecord(begin,stream)); CUDA(cudaGraphLaunch(executable,stream)); CUDA(cudaEventRecord(end,stream));
        CUDA(cudaEventSynchronize(end)); CUDA(cudaEventElapsedTime(&ms,begin,end));
        if (i >= 2u) samples.push_back(ms);
    }
    std::sort(samples.begin(),samples.end());
    CUDA(cudaGraphExecDestroy(executable)); CUDA(cudaGraphDestroy(graph)); CUDA(cudaEventDestroy(begin)); CUDA(cudaEventDestroy(end));
    return 0.5f*(samples[3]+samples[4]);
}

static void TopkCase(cudaStream_t stream)
{
    const uint32_t rows=4u,n=288u,k=8u;
    std::vector<float> logits(rows*n),bias(n),values_small(rows*k),values_warp(rows*k);
    std::vector<uint32_t> indices_small(rows*k),indices_warp(rows*k);
    float *device_logits,*device_bias,*device_values; uint32_t *device_indices;
    for (auto &value : logits) value=Signed()*6.0f;
    for (auto &value : bias) value=Signed()*0.5f;
    CUDA(cudaMalloc(&device_logits,logits.size()*4u)); CUDA(cudaMalloc(&device_bias,bias.size()*4u));
    CUDA(cudaMalloc(&device_values,rows*k*4u)); CUDA(cudaMalloc(&device_indices,rows*k*4u));
    CUDA(cudaMemcpy(device_logits,logits.data(),logits.size()*4u,cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(device_bias,bias.data(),bias.size()*4u,cudaMemcpyHostToDevice));
    LmTopkSmallKernel<256u,8u,true,1u,1u,LM_TOPK_SCORE_SIGMOID><<<rows,256u,2u*LM_TOPK_SMALL_LIMIT*sizeof(uint32_t),stream>>>(device_logits,n,device_indices,device_values,device_bias,0,2.5f);
    CUDA(cudaStreamSynchronize(stream));
    CUDA(cudaMemcpy(indices_small.data(),device_indices,rows*k*4u,cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(values_small.data(),device_values,rows*k*4u,cudaMemcpyDeviceToHost));
    CUDA((LmTopkRouteLaunch<256u,8u,true,LM_TOPK_SCORE_SIGMOID>(rows,device_logits,n,device_indices,device_values,device_bias,0,2.5f,stream)));
    CUDA(cudaStreamSynchronize(stream));
    CUDA(cudaMemcpy(indices_warp.data(),device_indices,rows*k*4u,cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(values_warp.data(),device_values,rows*k*4u,cudaMemcpyDeviceToHost));
    REQUIRE(indices_small == indices_warp);
    REQUIRE(memcmp(values_small.data(),values_warp.data(),rows*k*4u) == 0);
    printf("PASS router top-8 of 288: warp kernel indices and renormalised weights bitwise equal to the bitonic kernel rows=%u\n",rows);
    printf("TIMING router_topk calls=42 bitonic_ms=%.4f warp_ms=%.4f\n",TopkTime(false,device_logits,device_indices,device_values,device_bias,stream),TopkTime(true,device_logits,device_indices,device_values,device_bias,stream));
    CUDA(cudaFree(device_logits)); CUDA(cudaFree(device_bias)); CUDA(cudaFree(device_values)); CUDA(cudaFree(device_indices));
}

int main(int argc,char **argv)
{
    cudaStream_t stream;
    uint32_t multiprocessors;
    if (argc != 2 || strcmp(argv[1],"--run") != 0)
    {
        puts("usage: test_skinny_gemv --run");
        return 2;
    }
    alarm(600);
    setvbuf(stdout,nullptr,_IOLBF,0);
    CUDA(cudaSetDevice(0));
    {
        cudaDeviceProp properties;
        CUDA(cudaGetDeviceProperties(&properties,0));
        printf("DEVICE sm=%d.%d multiprocessors=%d l2_bytes=%d persisting_l2_max_bytes=%d\n",properties.major,properties.minor,properties.multiProcessorCount,properties.l2CacheSize,properties.persistingL2CacheMaxSize);
        multiprocessors=(uint32_t)properties.multiProcessorCount;
    }
    CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    for (const DenseShape &shape : dense_shapes)
        for (uint32_t rows : {1u,2u,3u,4u,5u,8u})
            DenseCase(&shape,rows,stream);
    puts("PASS skinny dense bf16 shapes=19 rows=1,2,3,4,5,8 reference=f64");
    for (uint32_t tokens : {1u,2u,8u})
    {
        ExpertCase(4096u,256u,tokens,0u,stream);
        ExpertCase(128u,4096u,tokens,1u,stream);
    }
    puts("PASS skinny grouped fp8 w1/w2 tokens=1,2,8 duplicate_expert=yes reference=f64");
    for (uint32_t tokens : {9u,32u,96u})
    {
        GroupedExpertCase(4096u,256u,tokens,0u,tokens >= 32u ? 20u : 0u,stream);
        GroupedExpertCase(128u,4096u,tokens,1u,tokens >= 32u ? 20u : 0u,stream);
    }
    puts("PASS skinny per-expert fp8 w1/w2 tokens=9,32,96 rows_per_expert_up_to=20 real_route_build=yes reference=f64 neuron_blocked_equals_one_neuron=bitwise");
    REQUIRE(LmSkinnyGroupedExperts<LmFp8>((const void *)16,LmScaleTensorNone(),(const uint16_t *)16,(uint16_t *)16,(const uint32_t *)16,(const uint32_t *)16,288u,288u*16u+1u,0u,4096u,256u,stream) == LM_LAUNCH_ERR_SHAPE);
    REQUIRE(LmSkinnyGroupedExperts<LmFp8>((const void *)16,LmScaleTensorNone(),(const uint16_t *)16,(uint16_t *)16,(const uint32_t *)16,0,288u,64u,0u,4096u,256u,stream) == LM_LAUNCH_ERR_SHAPE);
    puts("PASS per-expert skinny declines more than 16 rows per expert on average and a missing token map");
    REQUIRE(LmSkinnyDense<LmBf16Format>((const void *)8,(const uint16_t *)16,(uint16_t *)16,0,9u,4096u,16u,0u,0u,stream) == LM_LAUNCH_ERR_SHAPE);
    REQUIRE(LmSkinnyDense<LmBf16Format>((const void *)8,(const uint16_t *)16,(uint16_t *)16,0,1u,4096u,16u,0u,0u,stream) == LM_LAUNCH_ERR_SHAPE);
    REQUIRE(LmSkinnyExperts<LmFp8>((const void *)16,LmScaleTensorNone(),(const uint16_t *)16,(uint16_t *)16,(const uint32_t *)16,(const uint32_t *)16,9u*8u,8u,0u,4096u,256u,stream) == LM_LAUNCH_ERR_SHAPE);
    puts("PASS skinny declines rows>8, more than eight routed tokens and misaligned weights so callers fall back to the tensor-core GEMM");
    for (const DenseShape &shape : dense_shapes)
        if ((shape.input % LmBf16Format::kTileK) == 0u)
            TimeShape(&shape,stream);
    for (uint32_t tokens : {9u,32u,64u,256u})
    {
        GroupedTiming(4096u,256u,tokens,0u,multiprocessors,stream);
        GroupedTiming(128u,4096u,tokens,1u,multiprocessors,stream);
    }
    TopkCase(stream);
    CUDA(cudaStreamDestroy(stream));
    puts("PASS skinny_gemv_probe");
    return 0;
}
