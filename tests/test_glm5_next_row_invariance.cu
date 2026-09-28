#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <unistd.h>
#include "modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_cuda.cu"

#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { fprintf(stderr,"FAIL line=%d cuda=%s call=%s\n",__LINE__,cudaGetErrorString(e),#call); exit(1); } } while(0)
#define REQUIRE(test) do { if(!(test)) { fprintf(stderr,"FAIL line=%d test=%s\n",__LINE__,#test); exit(1); } } while(0)

#define ROWEQ_POOL 64u
#define ROWEQ_COMPOSITIONS 3u
#define ROWEQ_TP 16u
#define ROWEQ_HEADS (GLM5_NEXT_ATTN_HEADS / ROWEQ_TP)
#define ROWEQ_CAPACITY 128u
#define ROWEQ_SPLIT_THRESHOLD 64u
#define ROWEQ_PAGE 64u
#define ROWEQ_RUN 4u

static const uint32_t roweq_waves[] = {1u,2u,8u,17u,64u};

static const std::vector<std::string> roweq_known_breaks =
{
};

typedef std::vector<std::vector<uint8_t>> RowBytes;

typedef struct Family
{
    std::string name;
    std::vector<std::string> sites;
    std::function<void(const std::vector<uint32_t> &,std::vector<RowBytes> &)> run;
}
Family;

typedef struct Tally
{
    uint32_t changed,total,first_row,first_wave_index;
}
Tally;

static uint32_t random_state = 20260928u;

static uint32_t Random()
{
    random_state ^= random_state << 13u;
    random_state ^= random_state >> 17u;
    random_state ^= random_state << 5u;
    return random_state;
}

__global__ static void FillBf16Kernel(uint16_t *out,uint64_t count,uint64_t seed,float scale)
{
    for (uint64_t index=blockIdx.x*(uint64_t)blockDim.x+threadIdx.x; index<count; index+=(uint64_t)gridDim.x*blockDim.x)
    {
        const uint64_t bits=LmSplitMix64(seed^LmSplitMix64(index));
        out[index]=LmFloatToBf16(((float)(bits>>40u)/8388608.0f-1.0f)*scale);
    }
}

__global__ static void FillFp8Kernel(uint8_t *out,uint64_t count,uint64_t seed)
{
    for (uint64_t index=blockIdx.x*(uint64_t)blockDim.x+threadIdx.x; index<count; index+=(uint64_t)gridDim.x*blockDim.x)
    {
        uint8_t value=(uint8_t)(LmSplitMix64(seed^LmSplitMix64(index))&0xffu);
        out[index]=(value&0x7fu) == 0x7fu ? (uint8_t)(value^1u) : value;
    }
}

__global__ static void FillScaleKernel(float *out,uint64_t count,uint64_t seed)
{
    for (uint64_t index=blockIdx.x*(uint64_t)blockDim.x+threadIdx.x; index<count; index+=(uint64_t)gridDim.x*blockDim.x)
        out[index]=ldexpf(1.0f+(float)(LmSplitMix64(seed^LmSplitMix64(index))%256u)/256.0f,-10);
}

template<class T> static T *Allocate(uint64_t count)
{
    T *device;
    CUDA(cudaMalloc(&device,std::max<uint64_t>(count,1u)*sizeof(T)));
    CUDA(cudaMemset(device,0,std::max<uint64_t>(count,1u)*sizeof(T)));
    return device;
}

static uint16_t *RandomBf16(uint64_t count,float scale)
{
    uint16_t *device=Allocate<uint16_t>(count);
    FillBf16Kernel<<<1024,256>>>(device,count,((uint64_t)Random()<<32u)|Random(),scale);
    CUDA(cudaPeekAtLastError());
    return device;
}

static uint8_t *RandomFp8(uint64_t count)
{
    uint8_t *device=Allocate<uint8_t>(count);
    FillFp8Kernel<<<1024,256>>>(device,count,((uint64_t)Random()<<32u)|Random());
    CUDA(cudaPeekAtLastError());
    return device;
}

static float *RandomScale(uint64_t count)
{
    float *device=Allocate<float>(count);
    FillScaleKernel<<<1024,256>>>(device,count,((uint64_t)Random()<<32u)|Random());
    CUDA(cudaPeekAtLastError());
    return device;
}

template<class T> static T *Upload(const std::vector<T> &host)
{
    T *device=Allocate<T>(host.size());
    if (!host.empty()) CUDA(cudaMemcpy(device,host.data(),host.size()*sizeof(T),cudaMemcpyHostToDevice));
    return device;
}

static void Gather(void *wave,const void *pool,uint64_t row_bytes,const std::vector<uint32_t> &members)
{
    for (uint32_t row=0u; row<members.size(); row++)
        CUDA(cudaMemcpy((uint8_t *)wave+(uint64_t)row*row_bytes,(const uint8_t *)pool+(uint64_t)members[row]*row_bytes,row_bytes,cudaMemcpyDeviceToDevice));
}

static RowBytes Scatter(const void *wave,uint64_t row_bytes,uint64_t row_stride_bytes,uint32_t rows)
{
    std::vector<uint8_t> host((uint64_t)rows*row_stride_bytes);
    RowBytes out(rows);
    CUDA(cudaDeviceSynchronize());
    CUDA(cudaMemcpy(host.data(),wave,host.size(),cudaMemcpyDeviceToHost));
    for (uint32_t row=0u; row<rows; row++)
        out[row].assign(host.begin()+(uint64_t)row*row_stride_bytes,host.begin()+(uint64_t)row*row_stride_bytes+row_bytes);
    return out;
}

static void RowOffset(uint32_t *device,uint32_t rows)
{
    const uint32_t window[2]={0u,rows};
    CUDA(cudaMemcpy(device,window,sizeof(window),cudaMemcpyHostToDevice));
}

typedef struct DenseShape
{
    const char *name;
    uint32_t input,output;
}
DenseShape;

static const DenseShape roweq_dense_shapes[] =
{
    {"mla_q_a",GLM5_NEXT_HIDDEN,GLM5_NEXT_QUERY_A_DIM},
    {"mla_q_b",GLM5_NEXT_QUERY_A_DIM,ROWEQ_HEADS*(GLM5_NEXT_QK_NOPE_DIM+GLM5_NEXT_ROPE_DIM)},
    {"mla_kv_a",GLM5_NEXT_HIDDEN,GLM5_NEXT_LATENT_ROW},
    {"attn_out",ROWEQ_HEADS*GLM5_NEXT_VALUE_DIM,GLM5_NEXT_HIDDEN},
    {"index_q",GLM5_NEXT_QUERY_A_DIM,GLM5_NEXT_DSA_QUERY_DIM},
    {"index_k",GLM5_NEXT_HIDDEN,GLM5_NEXT_DSA_INDEX_DIM},
    {"index_head",GLM5_NEXT_HIDDEN,GLM5_NEXT_DSA_INDEX_HEADS},
    {"shared_gate_up",GLM5_NEXT_HIDDEN,2u*GLM5_NEXT_EXPERT_INTERMEDIATE/ROWEQ_TP},
    {"shared_down",GLM5_NEXT_EXPERT_INTERMEDIATE/ROWEQ_TP,GLM5_NEXT_HIDDEN},
};

static Family DenseLinearFamily()
{
    Family family;
    std::vector<uint16_t *> weights,pools;
    uint16_t *activation=Allocate<uint16_t>((uint64_t)ROWEQ_POOL*GLM5_NEXT_HIDDEN*2u),*output=Allocate<uint16_t>((uint64_t)ROWEQ_POOL*GLM5_NEXT_HIDDEN*2u);
    family.name="dense_linear";
    for (const DenseShape &shape : roweq_dense_shapes)
    {
        family.sites.push_back(shape.name);
        weights.push_back(RandomBf16((uint64_t)shape.input*shape.output,0.0625f));
        pools.push_back(RandomBf16((uint64_t)ROWEQ_POOL*shape.input,4.0f));
    }
    family.run=[=](const std::vector<uint32_t> &members,std::vector<RowBytes> &sites)
    {
        const uint32_t rows=(uint32_t)members.size();
        sites.clear();
        for (uint32_t index=0u; index<sizeof(roweq_dense_shapes)/sizeof(roweq_dense_shapes[0]); index++)
        {
            const DenseShape &shape=roweq_dense_shapes[index];
            Gather(activation,pools[index],(uint64_t)shape.input*2u,members);
            REQUIRE(Glm5NextLaunchBf16LinearRows(activation,weights[index],output,rows,shape.input,shape.output,shape.output,0u,0) == LM_LAUNCH_OK);
            sites.push_back(Scatter(output,(uint64_t)shape.output*2u,(uint64_t)shape.output*2u,rows));
        }
    };
    return family;
}

static Glm5NextLayerBuffers MlpBuffers(uint32_t *row_offset,uint32_t *tile_prefix)
{
    Glm5NextLayerBuffers buffers;
    memset(&buffers,0,sizeof(buffers));
    buffers.dense_row_offset=row_offset;
    buffers.dense_tile_prefix=tile_prefix;
    buffers.tp_degree=ROWEQ_TP;
    buffers.hc_collapsed_bf16=Allocate<uint16_t>((uint64_t)ROWEQ_POOL*GLM5_NEXT_HIDDEN);
    buffers.hidden_bf16=Allocate<uint16_t>((uint64_t)ROWEQ_POOL*GLM5_NEXT_HC*GLM5_NEXT_HIDDEN);
    buffers.residual_bf16=Allocate<uint16_t>((uint64_t)ROWEQ_POOL*GLM5_NEXT_HIDDEN);
    buffers.normed_bf16=Allocate<uint16_t>((uint64_t)ROWEQ_POOL*GLM5_NEXT_HIDDEN);
    buffers.attention_out_bf16=Allocate<uint16_t>((uint64_t)ROWEQ_POOL*GLM5_NEXT_HIDDEN);
    buffers.mlp_norm_weight=RandomBf16(GLM5_NEXT_HIDDEN,1.0f);
    return buffers;
}

static Family DenseMlpFamily(uint32_t multiprocessors)
{
    Family family;
    uint32_t *row_offset=Allocate<uint32_t>(2u),*tile_prefix=Allocate<uint32_t>(2u);
    Glm5NextLayerBuffers buffers=MlpBuffers(row_offset,tile_prefix);
    uint16_t *pool=RandomBf16((uint64_t)ROWEQ_POOL*GLM5_NEXT_HIDDEN,2.0f);
    buffers.dense_intermediate=GLM5_NEXT_DENSE_INTERMEDIATE/ROWEQ_TP;
    buffers.dense_gate_up_rows=2u*buffers.dense_intermediate;
    buffers.dense_gate_up_fused=1u;
    buffers.dense_gate_weight=RandomBf16((uint64_t)buffers.dense_gate_up_rows*GLM5_NEXT_HIDDEN,0.0625f);
    buffers.dense_up_weight=buffers.dense_gate_weight;
    buffers.dense_down_weight=RandomBf16((uint64_t)GLM5_NEXT_HIDDEN*buffers.dense_intermediate,0.0625f);
    buffers.gate_up_bf16=Allocate<uint16_t>((uint64_t)ROWEQ_POOL*buffers.dense_gate_up_rows);
    buffers.intermediate_bf16=Allocate<uint16_t>((uint64_t)ROWEQ_POOL*buffers.dense_gate_up_rows);
    family.name="dense_mlp";
    family.sites={"output"};
    family.run=[=](const std::vector<uint32_t> &members,std::vector<RowBytes> &sites)
    {
        const uint32_t rows=(uint32_t)members.size();
        RowOffset(row_offset,rows);
        Gather(buffers.hc_collapsed_bf16,pool,GLM5_NEXT_HIDDEN*2u,members);
        REQUIRE(Glm5NextLayerDenseMlp(&buffers,rows,multiprocessors,0) == LM_LAUNCH_OK);
        sites={Scatter(buffers.attention_out_bf16,GLM5_NEXT_HIDDEN*2u,GLM5_NEXT_HIDDEN*2u,rows)};
    };
    return family;
}

static Family MoeFamily(uint32_t multiprocessors)
{
    Family family;
    uint32_t *row_offset=Allocate<uint32_t>(2u),*tile_prefix=Allocate<uint32_t>(2u);
    Glm5NextLayerBuffers buffers=MlpBuffers(row_offset,tile_prefix);
    uint16_t *pool=RandomBf16((uint64_t)ROWEQ_POOL*GLM5_NEXT_HIDDEN,2.0f);
    const uint32_t pairs=ROWEQ_POOL*GLM5_NEXT_TOP_K;
    buffers.expert_w1_rows=2u*GLM5_NEXT_EXPERT_INTERMEDIATE/ROWEQ_TP;
    buffers.expert_intermediate=GLM5_NEXT_EXPERT_INTERMEDIATE/ROWEQ_TP;
    buffers.shared_gate_up_rows=buffers.expert_w1_rows;
    buffers.shared_intermediate=buffers.expert_intermediate;
    buffers.router_weight=RandomBf16((uint64_t)GLM5_NEXT_EXPERTS*GLM5_NEXT_HIDDEN,0.0625f);
    buffers.router_correction_bias=RandomScale(GLM5_NEXT_EXPERTS);
    buffers.expert_w1_weight=RandomFp8((uint64_t)GLM5_NEXT_EXPERTS*buffers.expert_w1_rows*GLM5_NEXT_HIDDEN);
    buffers.expert_w1_scale=RandomScale((uint64_t)GLM5_NEXT_EXPERTS*buffers.expert_w1_rows*(GLM5_NEXT_HIDDEN/GLM5_NEXT_FP8_SCALE_BLOCK));
    buffers.expert_w2_weight=RandomFp8((uint64_t)GLM5_NEXT_EXPERTS*GLM5_NEXT_HIDDEN*buffers.expert_intermediate);
    buffers.expert_w2_scale=RandomScale((uint64_t)GLM5_NEXT_EXPERTS*GLM5_NEXT_HIDDEN*(buffers.expert_intermediate/GLM5_NEXT_FP8_SCALE_BLOCK));
    buffers.shared_gate_up_weight=RandomBf16((uint64_t)buffers.shared_gate_up_rows*GLM5_NEXT_HIDDEN,0.0625f);
    buffers.shared_down_weight=RandomBf16((uint64_t)GLM5_NEXT_HIDDEN*buffers.shared_intermediate,0.0625f);
    buffers.router_logits=Allocate<float>((uint64_t)ROWEQ_POOL*GLM5_NEXT_EXPERTS);
    buffers.route_expert=Allocate<uint32_t>(pairs);
    buffers.route_weight=Allocate<float>(pairs);
    buffers.route_source_token=Allocate<uint32_t>(pairs);
    buffers.route_packed_row=Allocate<uint32_t>(pairs);
    buffers.group_row_offset=Allocate<uint32_t>(GLM5_NEXT_EXPERTS+1u);
    buffers.group_tile_prefix_w1=Allocate<uint32_t>(GLM5_NEXT_EXPERTS+1u);
    buffers.group_tile_prefix_w2=Allocate<uint32_t>(GLM5_NEXT_EXPERTS+1u);
    buffers.gate_up_bf16=Allocate<uint16_t>((uint64_t)pairs*buffers.expert_w1_rows);
    buffers.intermediate_bf16=Allocate<uint16_t>((uint64_t)pairs*buffers.expert_w1_rows);
    buffers.expert_out_bf16=Allocate<uint16_t>((uint64_t)pairs*GLM5_NEXT_HIDDEN);
    buffers.shared_out_bf16=Allocate<uint16_t>((uint64_t)ROWEQ_POOL*GLM5_NEXT_HIDDEN);
    family.name="moe";
    family.sites={"router_logits","route","output"};
    family.run=[=](const std::vector<uint32_t> &members,std::vector<RowBytes> &sites)
    {
        const uint32_t rows=(uint32_t)members.size();
        RowOffset(row_offset,rows);
        Gather(buffers.hc_collapsed_bf16,pool,GLM5_NEXT_HIDDEN*2u,members);
        REQUIRE(Glm5NextLayerMoeRoute<GLM5_NEXT_EXPERT_WEIGHT_CODEC>(&buffers,rows,rows*GLM5_NEXT_TOP_K,multiprocessors,0) == LM_LAUNCH_OK);
        RowBytes logits=Scatter(buffers.router_logits,GLM5_NEXT_EXPERTS*4u,GLM5_NEXT_EXPERTS*4u,rows);
        RowBytes experts=Scatter(buffers.route_expert,GLM5_NEXT_TOP_K*4u,GLM5_NEXT_TOP_K*4u,rows);
        RowBytes weights=Scatter(buffers.route_weight,GLM5_NEXT_TOP_K*4u,GLM5_NEXT_TOP_K*4u,rows);
        for (uint32_t row=0u; row<rows; row++)
            experts[row].insert(experts[row].end(),weights[row].begin(),weights[row].end());
        REQUIRE(Glm5NextLayerMoeExperts<GLM5_NEXT_EXPERT_WEIGHT_CODEC>(&buffers,rows,rows*GLM5_NEXT_TOP_K,multiprocessors,0) == LM_LAUNCH_OK);
        sites={logits,experts,Scatter(buffers.attention_out_bf16,GLM5_NEXT_HIDDEN*2u,GLM5_NEXT_HIDDEN*2u,rows)};
    };
    return family;
}

static Family AttentionFamily(uint32_t multiprocessors)
{
    Family family;
    const uint32_t sequences=ROWEQ_POOL/ROWEQ_RUN,width=SPARK_GLM5_NEXT_MODEL_INDEX_OUTPUT_WIDTH;
    std::vector<uint32_t> last(sequences),table,selection((uint64_t)ROWEQ_POOL*width);
    uint32_t longest=0u;
    for (uint32_t sequence=0u; sequence<sequences; sequence++)
    {
        last[sequence]=sequence%4u == 0u ? ROWEQ_RUN+Random()%(ROWEQ_SPLIT_THRESHOLD-ROWEQ_RUN) : sequence%4u == 3u ? GLM5_NEXT_DSA_SELECTED+Random()%1000u : ROWEQ_RUN+Random()%1500u;
        longest=std::max(longest,last[sequence]);
    }
    for (uint32_t row=0u; row<ROWEQ_POOL; row++)
    {
        const uint32_t position=last[row/ROWEQ_RUN]-ROWEQ_RUN+row%ROWEQ_RUN;
        for (uint32_t slot=0u; slot<width; slot++)
            selection[(uint64_t)row*width+slot]=slot%97u == 13u ? 0xffffffffu : Random()%(position+1u+(slot%5u == 0u ? 8u : 0u));
    }
    const uint32_t pages=(longest+ROWEQ_PAGE-1u)/ROWEQ_PAGE;
    for (uint32_t page=0u; page<sequences*pages; page++) table.push_back((page*7919u)%(sequences*pages));
    uint16_t *kv=RandomBf16((uint64_t)sequences*pages*ROWEQ_PAGE*GLM5_NEXT_LATENT_ROW,1.0f);
    uint16_t *queries=RandomBf16((uint64_t)ROWEQ_POOL*ROWEQ_HEADS*GLM5_NEXT_LATENT,0.25f);
    uint16_t *query=Allocate<uint16_t>((uint64_t)ROWEQ_POOL*ROWEQ_HEADS*GLM5_NEXT_LATENT),*output=Allocate<uint16_t>((uint64_t)ROWEQ_POOL*ROWEQ_HEADS*GLM5_NEXT_LATENT);
    uint32_t *device_table=Upload(table),*pool_selection=Upload(selection),*wave_selection=Allocate<uint32_t>((uint64_t)ROWEQ_POOL*width);
    uint32_t *contexts=Allocate<uint32_t>(sequences),*sequence=Allocate<uint32_t>(ROWEQ_POOL),*position=Allocate<uint32_t>(ROWEQ_POOL);
    const uint64_t partial_blocks=SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_BLOCKS(ROWEQ_CAPACITY,ROWEQ_HEADS);
    float *partials=Allocate<float>(partial_blocks*SPARK_GLM5_NEXT_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_FLOATS);
    LmKvAccessError *error=Allocate<LmKvAccessError>(1u);
    LmKvView view;
    REQUIRE(LmKvViewInitialize(&view,(uint8_t *)kv,device_table,pages,sequences,sequences*pages,error) == 0);
    family.name="attention";
    family.sites={"latent"};
    family.run=[=](const std::vector<uint32_t> &members,std::vector<RowBytes> &sites)
    {
        const uint32_t rows=(uint32_t)members.size();
        std::vector<uint32_t> positions(rows),sequence_rows(rows),wave_contexts(sequences,1u);
        uint32_t context=0u;
        LmKvAccessError host_error;
        for (uint32_t row=0u; row<rows; row++)
        {
            sequence_rows[row]=members[row]/ROWEQ_RUN;
            positions[row]=last[sequence_rows[row]]-ROWEQ_RUN+members[row]%ROWEQ_RUN;
            wave_contexts[sequence_rows[row]]=std::max(wave_contexts[sequence_rows[row]],positions[row]+1u);
            context=std::max(context,positions[row]+1u);
        }
        CUDA(cudaMemcpy(sequence,sequence_rows.data(),rows*4u,cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(position,positions.data(),rows*4u,cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(contexts,wave_contexts.data(),sequences*4u,cudaMemcpyHostToDevice));
        Gather(query,queries,(uint64_t)ROWEQ_HEADS*GLM5_NEXT_LATENT*2u,members);
        Gather(wave_selection,pool_selection,(uint64_t)width*4u,members);
        for (uint32_t grouping : {multiprocessors,1u,1u<<20u})
        {
            CUDA((LmLatentAttentionHeadsLaunch<Glm5NextKv,GLM5_NEXT_LATENT>(query,view,sequence,contexts,context > GLM5_NEXT_DSA_SELECTED ? wave_selection : 0,width,ROWEQ_HEADS,0.0625f,output,position,rows,SparkGlm5NextAttentionPositionBound(context),GLM5_NEXT_DSA_SELECTED,partials,partial_blocks,grouping,0)));
            RowBytes bytes=Scatter(output,(uint64_t)ROWEQ_HEADS*GLM5_NEXT_LATENT*2u,(uint64_t)ROWEQ_HEADS*GLM5_NEXT_LATENT*2u,rows);
            if (grouping == multiprocessors)
                sites={bytes};
            else if (bytes != sites[0])
            {
                fprintf(stderr,"FAIL attention heads per block changed bits: wave=%u grouping_multiprocessors=%u\n",rows,grouping);
                exit(1);
            }
        }
        CUDA(cudaMemcpy(&host_error,error,sizeof(host_error),cudaMemcpyDeviceToHost));
        REQUIRE(host_error.error_code == LM_KV_ACCESS_ERROR_NONE);
    };
    return family;
}

typedef struct HeadState
{
    SparkGlm5NextCudaWave *wave;
    SparkGlm5NextExecutionSlot *slot;
    uint16_t *hidden;
    std::vector<SparkRowSampling> rules;
}
HeadState;

static HeadState HeadSetup(uint32_t sampled_every,uint32_t twin_rows)
{
    const uint32_t vocabulary=GLM5_NEXT_VOCAB/ROWEQ_TP,tiles=(vocabulary+GLM5_NEXT_HEAD_TILE-1u)/GLM5_NEXT_HEAD_TILE;
    HeadState state;
    SparkGlm5NextLayerWeights *layer=new SparkGlm5NextLayerWeights();
    uint32_t *ordinal=new uint32_t(UINT32_MAX);
    uint16_t *head=RandomBf16((uint64_t)vocabulary*GLM5_NEXT_HIDDEN,0.05f);
    if (twin_rows != 0u)
        CUDA(cudaMemcpy2D(head+GLM5_NEXT_HIDDEN,2u*GLM5_NEXT_HIDDEN*2u,head,2u*GLM5_NEXT_HIDDEN*2u,GLM5_NEXT_HIDDEN*2u,vocabulary/2u,cudaMemcpyDeviceToDevice));
    state.wave=new SparkGlm5NextCudaWave();
    state.slot=new SparkGlm5NextExecutionSlot();
    state.hidden=RandomBf16((uint64_t)ROWEQ_POOL*GLM5_NEXT_HC*GLM5_NEXT_HIDDEN,1.0f);
    state.slot->hidden_bf16=Allocate<uint16_t>((uint64_t)ROWEQ_POOL*GLM5_NEXT_HC*GLM5_NEXT_HIDDEN);
    state.slot->residual_bf16=Allocate<uint16_t>((uint64_t)ROWEQ_POOL*GLM5_NEXT_HIDDEN);
    state.slot->hc_mean_bf16=Allocate<uint16_t>((uint64_t)ROWEQ_POOL*GLM5_NEXT_HIDDEN);
    state.slot->normed_bf16=Allocate<uint16_t>((uint64_t)ROWEQ_POOL*GLM5_NEXT_HIDDEN);
    state.slot->head_candidate_score=Allocate<float>((uint64_t)ROWEQ_POOL*tiles);
    state.slot->head_candidate_token=Allocate<uint32_t>((uint64_t)ROWEQ_POOL*tiles);
    state.slot->output_score=Allocate<float>(ROWEQ_POOL);
    state.slot->output_token=Allocate<uint32_t>(ROWEQ_POOL);
    state.slot->head_maxloc_u64=Allocate<uint64_t>(ROWEQ_POOL);
    state.slot->positions=Allocate<uint32_t>(ROWEQ_POOL);
    state.slot->row_sampling=Allocate<SparkRowSampling>(ROWEQ_POOL);
    state.slot->head_certified_scratch=Allocate<uint8_t>(ROWEQ_POOL*SparkHeadCertifiedFp8ScratchBytes(vocabulary,GLM5_NEXT_HIDDEN));
    state.slot->head_certified_candidates=Allocate<uint32_t>(vocabulary);
    state.slot->head_screened_count=Allocate<uint32_t>(1u);
    state.slot->head_certified_token=Allocate<uint32_t>(ROWEQ_POOL);
    state.slot->head_certified_score=Allocate<float>(ROWEQ_POOL);
    uint8_t *payload=Allocate<uint8_t>((uint64_t)vocabulary*GLM5_NEXT_HIDDEN);
    float *scale=Allocate<float>((uint64_t)vocabulary*GLM5_NEXT_HIDDEN/32u),*bound=Allocate<float>((uint64_t)vocabulary*GLM5_NEXT_HIDDEN/32u);
    REQUIRE(SparkGlm5NextLaunchHeadCertifiedQuantize(0,head,payload,scale,bound,vocabulary,GLM5_NEXT_HIDDEN) == cudaSuccess);
    state.wave->slot=state.slot;
    state.wave->layers=layer;
    state.wave->layer_count=1u;
    state.wave->tp_degree=ROWEQ_TP;
    state.wave->owns_final_head=1u;
    state.wave->kda_ordinal_by_local_layer=state.wave->index_ordinal_by_local_layer=ordinal;
    state.wave->final_norm_bf16=RandomBf16(GLM5_NEXT_HIDDEN,1.0f);
    state.wave->lm_head_bf16=head;
    state.wave->head_certified_fp8_payload=payload;
    state.wave->head_certified_fp8_scale_f32=scale;
    state.wave->head_certified_fp8_norm_f32=bound;
    for (uint32_t row=0u; row<ROWEQ_POOL; row++)
        state.rules.push_back(SparkSamplingRule(sampled_every != 0u && row%sampled_every == 0u ? 0.7f+0.1f*(float)(row%3u) : 0.0f,1000u+row));
    return state;
}

static Family HeadFamily(const char *name,uint32_t sampled_every,uint32_t twin_rows)
{
    Family family;
    HeadState state=HeadSetup(sampled_every,twin_rows);
    SparkRowSampling *host_rules=new SparkRowSampling[ROWEQ_POOL];
    family.name=name;
    family.sites={"maxloc","token"};
    family.run=[=](const std::vector<uint32_t> &members,std::vector<RowBytes> &sites)
    {
        const uint32_t rows=(uint32_t)members.size();
        std::vector<SparkRowSampling> rules(rows);
        std::vector<uint32_t> positions(rows);
        uint32_t sampled=0u;
        for (uint32_t row=0u; row<rows; row++)
        {
            rules[row]=state.rules[members[row]];
            positions[row]=500u+members[row]*3u;
            sampled|=rules[row].inverse_temperature != 0.0f ? 1u : 0u;
        }
        CUDA(cudaMemcpy(state.slot->row_sampling,rules.data(),rows*sizeof(SparkRowSampling),cudaMemcpyHostToDevice));
        std::copy(rules.begin(),rules.end(),host_rules);
        state.wave->host_row_sampling=host_rules;
        CUDA(cudaMemcpy(state.slot->positions,positions.data(),rows*4u,cudaMemcpyHostToDevice));
        Gather(state.slot->hidden_bf16,state.hidden,(uint64_t)GLM5_NEXT_HC*GLM5_NEXT_HIDDEN*2u,members);
        state.wave->row_count=rows;
        state.wave->sampled=sampled;
        REQUIRE(SparkGlm5NextRunHead(state.wave) == LM_LAUNCH_OK);
        sites={Scatter(state.slot->head_maxloc_u64,8u,8u,rows),Scatter(state.slot->head_maxloc_u64,4u,8u,rows)};
        for (uint32_t row=0u; twin_rows != 0u && row<rows; row++)
        {
            uint32_t inverted;
            memcpy(&inverted,sites[1][row].data(),4u);
            if (((UINT32_MAX-inverted)&1u) != 0u)
            {
                fprintf(stderr,"FAIL head tie: pool row %u picked token %u over its lower twin at wave %u\n",members[row],UINT32_MAX-inverted,rows);
                exit(1);
            }
        }
    };
    return family;
}

static std::vector<uint32_t> Composition(uint32_t wave)
{
    std::vector<uint32_t> order(ROWEQ_POOL);
    for (uint32_t row=0u; row<ROWEQ_POOL; row++) order[row]=row;
    for (uint32_t row=ROWEQ_POOL-1u; row>0u; row--) std::swap(order[row],order[Random()%(row+1u)]);
    order.resize(wave);
    return order;
}

static void Measure(const Family &family,std::map<std::string,Tally> &tallies,std::vector<std::string> &keys)
{
    std::vector<RowBytes> reference,sites;
    reference.assign(family.sites.size(),RowBytes(ROWEQ_POOL));
    for (uint32_t row=0u; row<ROWEQ_POOL; row++)
    {
        family.run({row},sites);
        REQUIRE(sites.size() == family.sites.size());
        for (uint32_t site=0u; site<sites.size(); site++) reference[site][row]=sites[site][0];
    }
    for (uint32_t wave : roweq_waves)
    {
        if (wave == 1u) continue;
        for (uint32_t site=0u; site<family.sites.size(); site++)
        {
            const std::string key=family.name+"."+family.sites[site]+"@"+std::to_string(wave);
            tallies[key]={0u,0u,UINT32_MAX,0u};
            keys.push_back(key);
        }
        for (uint32_t trial=0u; trial<(wave == ROWEQ_POOL ? 1u : ROWEQ_COMPOSITIONS); trial++)
        {
            const std::vector<uint32_t> members=Composition(wave);
            std::vector<RowBytes> again;
            family.run(members,sites);
            family.run(members,again);
            if (again != sites)
            {
                fprintf(stderr,"FAIL row invariance: family=%s wave=%u is not repeatable, so wave dependence cannot be separated from nondeterminism\n",family.name.c_str(),wave);
                exit(1);
            }
            for (uint32_t site=0u; site<family.sites.size(); site++)
            {
                Tally &tally=tallies[family.name+"."+family.sites[site]+"@"+std::to_string(wave)];
                for (uint32_t row=0u; row<wave; row++)
                {
                    tally.total++;
                    if (sites[site][row] != reference[site][members[row]])
                    {
                        tally.changed++;
                        if (tally.first_row == UINT32_MAX) { tally.first_row=members[row]; tally.first_wave_index=row; }
                    }
                }
            }
        }
    }
}

int main(int argc,char **argv)
{
    cudaDeviceProp properties;
    std::map<std::string,Tally> tallies;
    std::vector<std::string> keys;
    std::set<std::string> known(roweq_known_breaks.begin(),roweq_known_breaks.end());
    uint32_t regressions=0u,stale=0u,breaks=0u;
    bool enforce;
    if (argc != 2 || (strcmp(argv[1],"--run") != 0 && strcmp(argv[1],"--report") != 0))
    {
        puts("usage: test_glm5_next_row_invariance --run|--report");
        return 2;
    }
    enforce=strcmp(argv[1],"--run") == 0;
    alarm(1800);
    setvbuf(stdout,nullptr,_IOLBF,0);
    CUDA(cudaSetDevice(0));
    CUDA(cudaGetDeviceProperties(&properties,0));
    const uint32_t multiprocessors=(uint32_t)properties.multiProcessorCount;
    std::vector<Family> families;
    families.push_back(DenseLinearFamily());
    families.push_back(DenseMlpFamily(multiprocessors));
    families.push_back(MoeFamily(multiprocessors));
    families.push_back(AttentionFamily(multiprocessors));
    families.push_back(HeadFamily("head_greedy",0u,0u));
    families.push_back(HeadFamily("head_sampled",1u,0u));
    families.push_back(HeadFamily("head_mixed",2u,0u));
    families.push_back(HeadFamily("head_ties",0u,1u));
    CUDA(cudaDeviceSynchronize());
    for (const Family &family : families)
        Measure(family,tallies,keys);
    for (const std::string &key : keys)
    {
        const Tally &tally=tallies[key];
        const bool broken=tally.changed != 0u,listed=known.count(key) != 0u;
        breaks+=broken ? 1u : 0u;
        if (broken && !listed) regressions++;
        if (!broken && listed) stale++;
        printf("ROWEQ cell=%s changed_rows=%u/%u verdict=%s%s",key.c_str(),tally.changed,tally.total,broken ? "BREAK" : "EQUAL",broken && !listed ? " unlisted" : !broken && listed ? " listed_but_equal" : "");
        if (broken) printf(" first_pool_row=%u at_wave_index=%u",tally.first_row,tally.first_wave_index);
        putchar('\n');
    }
    for (const std::string &key : known)
        if (tallies.count(key) == 0u)
        {
            printf("ROWEQ cell=%s verdict=UNMEASURED listed\n",key.c_str());
            stale++;
        }
    printf("ROWEQ summary cells=%zu breaks=%u known=%zu unlisted_breaks=%u listed_but_equal=%u multiprocessors=%u\n",keys.size(),breaks,known.size(),regressions,stale,multiprocessors);
    if (enforce && (regressions != 0u || stale != 0u))
    {
        puts("FAIL row invariance: every break must be listed in roweq_known_breaks and every listed cell must still break; the list may only shrink");
        return 1;
    }
    puts(enforce ? "PASS glm5_next row invariance matrix matches roweq_known_breaks" : "REPORT glm5_next row invariance matrix");
    return 0;
}
