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

int main(int argc,char **argv)
{
    cudaStream_t stream;
    if (argc != 2 || strcmp(argv[1],"--run") != 0)
    {
        puts("usage: test_skinny_gemv --run");
        return 2;
    }
    alarm(600);
    setvbuf(stdout,nullptr,_IOLBF,0);
    CUDA(cudaSetDevice(0));
    CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    for (const DenseShape &shape : dense_shapes)
        for (uint32_t rows : {1u,2u,3u,4u})
            DenseCase(&shape,rows,stream);
    puts("PASS skinny dense bf16 shapes=19 rows=1..4 reference=f64");
    for (uint32_t tokens : {1u,2u})
    {
        ExpertCase(4096u,256u,tokens,0u,stream);
        ExpertCase(128u,4096u,tokens,1u,stream);
    }
    puts("PASS skinny grouped fp8 w1/w2 tokens=1..2 duplicate_expert=yes reference=f64");
    REQUIRE(LmSkinnyDense<LmBf16Format>((const void *)8,(const uint16_t *)16,(uint16_t *)16,0,5u,4096u,16u,0u,0u,stream) == LM_LAUNCH_ERR_SHAPE);
    REQUIRE(LmSkinnyDense<LmBf16Format>((const void *)8,(const uint16_t *)16,(uint16_t *)16,0,1u,4096u,16u,0u,0u,stream) == LM_LAUNCH_ERR_SHAPE);
    REQUIRE(LmSkinnyExperts<LmFp8>((const void *)16,LmScaleTensorNone(),(const uint16_t *)16,(uint16_t *)16,(const uint32_t *)16,(const uint32_t *)16,5u*8u,8u,0u,4096u,256u,stream) == LM_LAUNCH_ERR_SHAPE);
    puts("PASS skinny declines rows>4, more than four routed tokens and misaligned weights so callers fall back to the tensor-core GEMM");
    for (const DenseShape &shape : dense_shapes)
        if ((shape.input % LmBf16Format::kTileK) == 0u)
            TimeShape(&shape,stream);
    CUDA(cudaStreamDestroy(stream));
    puts("PASS skinny_gemv_probe");
    return 0;
}
