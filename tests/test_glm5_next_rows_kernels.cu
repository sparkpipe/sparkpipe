#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>
#include "modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_cuda.cu"

#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { fprintf(stderr,"FAIL line=%d cuda=%s call=%s\n",__LINE__,cudaGetErrorString(e),#call); exit(1); } } while(0)
#define REQUIRE(test) do { if(!(test)) { fprintf(stderr,"FAIL line=%d test=%s\n",__LINE__,#test); exit(1); } } while(0)

static uint32_t random_state = 20260926u;

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

static float Bf16Value(uint16_t value)
{
    uint32_t bits=(uint32_t)value<<16u;
    float out;
    memcpy(&out,&bits,sizeof(out));
    return out;
}

static float Signed()
{
    return ((int32_t)(Random()%2049u)-1024)/1024.0f;
}

template<class T> static T *Upload(const std::vector<T> &host)
{
    T *device;
    CUDA(cudaMalloc(&device,std::max<size_t>(host.size(),1u)*sizeof(T)));
    if (!host.empty()) CUDA(cudaMemcpy(device,host.data(),host.size()*sizeof(T),cudaMemcpyHostToDevice));
    return device;
}

template<class T> static std::vector<T> Download(const T *device,size_t count)
{
    std::vector<T> host(count);
    CUDA(cudaMemcpy(host.data(),device,count*sizeof(T),cudaMemcpyDeviceToHost));
    return host;
}

static void HeadRun(bool rows_kernel,const uint16_t *normed,const uint16_t *weight,uint32_t rows,uint32_t vocabulary,uint32_t *token,float *score,cudaStream_t stream)
{
    const uint32_t tiles=(vocabulary+GLM5_NEXT_HEAD_TILE-1u)/GLM5_NEXT_HEAD_TILE;
    float *candidate_score; uint32_t *candidate_token;
    CUDA(cudaMalloc(&candidate_score,(uint64_t)rows*tiles*4u)); CUDA(cudaMalloc(&candidate_token,(uint64_t)rows*tiles*4u));
    if (rows_kernel)
        LmHeadCandidateRowsKernel<GLM5_NEXT_LAYER_THREADS,GLM5_NEXT_HEAD_TILE,GLM5_NEXT_HEAD_ROWS><<<dim3(tiles,(rows+GLM5_NEXT_HEAD_ROWS-1u)/GLM5_NEXT_HEAD_ROWS),GLM5_NEXT_LAYER_THREADS,0,stream>>>(normed,weight,0,candidate_score,candidate_token,rows,GLM5_NEXT_HIDDEN,vocabulary);
    else
        LmHeadCandidateKernel<GLM5_NEXT_LAYER_THREADS,GLM5_NEXT_HEAD_TILE><<<dim3(tiles,rows),GLM5_NEXT_LAYER_THREADS,0,stream>>>(normed,weight,0,candidate_score,candidate_token,rows,GLM5_NEXT_HIDDEN,vocabulary);
    CUDA(cudaPeekAtLastError());
    LmHeadCommitKernel<GLM5_NEXT_LAYER_THREADS><<<rows,GLM5_NEXT_LAYER_THREADS,0,stream>>>(candidate_score,candidate_token,tiles,token,score,rows);
    CUDA(cudaPeekAtLastError());
    CUDA(cudaStreamSynchronize(stream));
    CUDA(cudaFree(candidate_score)); CUDA(cudaFree(candidate_token));
}

static float HeadTime(bool rows_kernel,const uint16_t *normed,const uint16_t *weight,uint32_t rows,uint32_t vocabulary,uint32_t *token,float *score,cudaStream_t stream)
{
    cudaEvent_t begin,end;
    float ms;
    CUDA(cudaEventCreate(&begin)); CUDA(cudaEventCreate(&end));
    HeadRun(rows_kernel,normed,weight,rows,vocabulary,token,score,stream);
    CUDA(cudaEventRecord(begin,stream));
    HeadRun(rows_kernel,normed,weight,rows,vocabulary,token,score,stream);
    CUDA(cudaEventRecord(end,stream));
    CUDA(cudaEventSynchronize(end)); CUDA(cudaEventElapsedTime(&ms,begin,end));
    CUDA(cudaEventDestroy(begin)); CUDA(cudaEventDestroy(end));
    return ms;
}

static void HeadCase(uint32_t rows,uint32_t vocabulary,bool timing,cudaStream_t stream)
{
    std::vector<uint16_t> normed((uint64_t)rows*GLM5_NEXT_HIDDEN),weight((uint64_t)vocabulary*GLM5_NEXT_HIDDEN);
    uint16_t *device_normed,*device_weight; uint32_t *token_a,*token_b; float *score_a,*score_b;
    for (auto &value : normed) value=Bf16(Signed());
    for (auto &value : weight) value=Bf16(Signed()*0.05f);
    device_normed=Upload(normed); device_weight=Upload(weight);
    CUDA(cudaMalloc(&token_a,rows*4u)); CUDA(cudaMalloc(&token_b,rows*4u)); CUDA(cudaMalloc(&score_a,rows*4u)); CUDA(cudaMalloc(&score_b,rows*4u));
    HeadRun(false,device_normed,device_weight,rows,vocabulary,token_a,score_a,stream);
    HeadRun(true,device_normed,device_weight,rows,vocabulary,token_b,score_b,stream);
    std::vector<uint32_t> ta=Download(token_a,rows),tb=Download(token_b,rows);
    std::vector<float> sa=Download(score_a,rows),sb=Download(score_b,rows);
    for (uint32_t row=0u; row<rows; row++)
        if (memcmp(&sa[row],&sb[row],4u) == 0 && ta[row] != tb[row])
            printf("NOTE head exact score tie rows=%u row=%u tokens=%u/%u score=%.9g\n",rows,row,ta[row],tb[row],sa[row]);
        else if (ta[row] != tb[row] || memcmp(&sa[row],&sb[row],4u) != 0)
        {
            fprintf(stderr,"HEAD-MISMATCH rows=%u row=%u token=%u/%u score=%.9g/%.9g\n",rows,row,ta[row],tb[row],sa[row],sb[row]);
            exit(1);
        }
    printf("PASS head rows kernel rows=%u vocabulary=%u tokens_and_scores_bitwise_equal=yes\n",rows,vocabulary);
    if (timing)
        printf("TIMING head rows=%u vocabulary=%u per_row_kernel_ms=%.3f rows_kernel_ms=%.3f speedup=%.1f\n",rows,vocabulary,HeadTime(false,device_normed,device_weight,rows,vocabulary,token_a,score_a,stream),HeadTime(true,device_normed,device_weight,rows,vocabulary,token_b,score_b,stream),HeadTime(false,device_normed,device_weight,rows,vocabulary,token_a,score_a,stream)/HeadTime(true,device_normed,device_weight,rows,vocabulary,token_b,score_b,stream));
    CUDA(cudaFree(device_normed)); CUDA(cudaFree(device_weight)); CUDA(cudaFree(token_a)); CUDA(cudaFree(token_b)); CUDA(cudaFree(score_a)); CUDA(cudaFree(score_b));
}

typedef struct AttentionCase
{
    uint32_t rows,heads,pages_per_sequence,selected;
    std::vector<uint16_t> pool,query;
    std::vector<uint32_t> table,contexts,positions,selection;
}
AttentionCase;

static void AttentionReference(const AttentionCase &item,uint32_t row,uint32_t head,std::vector<double> &out)
{
    const uint32_t latent=GLM5_NEXT_LATENT,count=item.selected != 0u ? item.selected : item.contexts[row];
    std::vector<double> scores;
    std::vector<uint32_t> used;
    double maximum=-INFINITY,sum=0.0;
    for (uint32_t step=0u; step<count; step++)
    {
        uint32_t position=item.selected != 0u ? item.selection[(uint64_t)row*item.selected+step] : step;
        if (position > item.positions[row]) continue;
        uint32_t page=item.table[(uint64_t)row*item.pages_per_sequence+position/64u];
        const uint16_t *slot=&item.pool[((uint64_t)page*64u+position%64u)*latent];
        double score=0.0;
        for (uint32_t e=0u; e<latent; e++) score+=(double)Bf16Value(item.query[((uint64_t)row*item.heads+head)*latent+e])*Bf16Value(slot[e]);
        score*=0.0625;
        scores.push_back(score); used.push_back(position); maximum=std::max(maximum,score);
    }
    out.assign(latent,0.0);
    for (size_t i=0u; i<scores.size(); i++)
    {
        double weight=std::exp(scores[i]-maximum);
        uint32_t page=item.table[(uint64_t)row*item.pages_per_sequence+used[i]/64u];
        const uint16_t *slot=&item.pool[((uint64_t)page*64u+used[i]%64u)*latent];
        sum+=weight;
        for (uint32_t e=0u; e<latent; e++) out[e]+=weight*Bf16Value(slot[e]);
    }
    for (auto &value : out) value/=sum;
}

typedef struct AttentionDevice
{
    uint16_t *pool,*query,*output;
    uint32_t *table,*contexts,*positions,*selection,*sequences;
    float *partials;
    LmKvAccessError *error;
    LmKvView view;
}
AttentionDevice;

static void AttentionUpload(const AttentionCase &item,AttentionDevice *device)
{
    std::vector<uint32_t> sequence(item.rows);
    for (uint32_t row=0u; row<item.rows; row++) sequence[row]=row;
    device->pool=Upload(item.pool); device->query=Upload(item.query);
    device->table=Upload(item.table); device->contexts=Upload(item.contexts); device->positions=Upload(item.positions); device->selection=Upload(item.selection); device->sequences=Upload(sequence);
    CUDA(cudaMalloc(&device->error,sizeof(*device->error))); CUDA(cudaMemset(device->error,0,sizeof(*device->error)));
    CUDA(cudaMalloc(&device->output,(uint64_t)item.rows*item.heads*GLM5_NEXT_LATENT*2u));
    CUDA(cudaMalloc(&device->partials,(uint64_t)item.rows*item.heads*LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS*(GLM5_NEXT_LATENT+2u)*4u));
    REQUIRE(LmKvViewInitialize(&device->view,(uint8_t *)device->pool,device->table,item.pages_per_sequence,item.rows,item.rows*item.pages_per_sequence,device->error) == 0);
}

static void AttentionFree(AttentionDevice *device)
{
    CUDA(cudaFree(device->pool)); CUDA(cudaFree(device->query)); CUDA(cudaFree(device->table)); CUDA(cudaFree(device->contexts)); CUDA(cudaFree(device->positions)); CUDA(cudaFree(device->selection)); CUDA(cudaFree(device->sequences));
    CUDA(cudaFree(device->error)); CUDA(cudaFree(device->output)); CUDA(cudaFree(device->partials));
}

static void AttentionLaunch(const AttentionCase &item,const AttentionDevice *device,bool heads_kernel,uint32_t multiprocessors,cudaStream_t stream)
{
    const uint32_t bound=item.selected != 0u ? item.selected : *std::max_element(item.contexts.begin(),item.contexts.end()),blocks=item.rows*item.heads*LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS;
    const uint32_t *selection=item.selected != 0u ? device->selection : 0;
    if (heads_kernel)
        CUDA((LmLatentAttentionHeadsLaunch<Glm5NextKv,GLM5_NEXT_LATENT>(device->query,device->view,device->sequences,device->contexts,selection,item.selected,item.heads,0.0625f,device->output,device->positions,item.rows,bound,64u,device->partials,blocks,multiprocessors,stream)));
    else
        CUDA((LmLatentAttentionDecodeSplitLaunch<Glm5NextKv,GLM5_NEXT_ATTN_THREADS,GLM5_NEXT_LATENT,GLM5_NEXT_ROPE_DIM>(device->query,0,device->view,device->sequences,device->contexts,selection,item.selected,item.heads,0.0625f,device->output,device->positions,item.rows,bound,64u,device->partials,blocks,multiprocessors,stream)));
}

static std::vector<uint16_t> AttentionRun(const AttentionCase &item,bool heads_kernel,uint32_t multiprocessors,cudaStream_t stream)
{
    AttentionDevice device;
    LmKvAccessError error;
    AttentionUpload(item,&device);
    AttentionLaunch(item,&device,heads_kernel,multiprocessors,stream);
    CUDA(cudaStreamSynchronize(stream));
    CUDA(cudaMemcpy(&error,device.error,sizeof(error),cudaMemcpyDeviceToHost));
    REQUIRE(error.error_code == LM_KV_ACCESS_ERROR_NONE);
    std::vector<uint16_t> result=Download(device.output,(size_t)item.rows*item.heads*GLM5_NEXT_LATENT);
    AttentionFree(&device);
    return result;
}

static float AttentionTime(const AttentionCase &item,const AttentionDevice *device,bool heads_kernel,uint32_t multiprocessors,cudaStream_t stream)
{
    cudaEvent_t begin,end;
    float ms;
    CUDA(cudaEventCreate(&begin)); CUDA(cudaEventCreate(&end));
    AttentionLaunch(item,device,heads_kernel,multiprocessors,stream);
    CUDA(cudaEventRecord(begin,stream));
    for (uint32_t repeat=0u; repeat<5u; repeat++) AttentionLaunch(item,device,heads_kernel,multiprocessors,stream);
    CUDA(cudaEventRecord(end,stream));
    CUDA(cudaEventSynchronize(end)); CUDA(cudaEventElapsedTime(&ms,begin,end));
    CUDA(cudaEventDestroy(begin)); CUDA(cudaEventDestroy(end));
    return ms/5.0f;
}

static void AttentionBuild(AttentionCase *item,uint32_t rows,uint32_t heads,uint32_t maximum_context,uint32_t selected,bool full_context)
{
    item->rows=rows; item->heads=heads; item->selected=selected;
    item->pages_per_sequence=(maximum_context+63u)/64u;
    item->pool.resize((uint64_t)rows*item->pages_per_sequence*64u*GLM5_NEXT_LATENT);
    item->query.resize((uint64_t)rows*heads*GLM5_NEXT_LATENT);
    item->table.resize((uint64_t)rows*item->pages_per_sequence);
    for (auto &value : item->pool) value=Bf16(Signed());
    for (auto &value : item->query) value=Bf16(Signed()*0.25f);
    for (uint32_t page=0u; page<item->table.size(); page++) item->table[page]=(uint32_t)item->table.size()-1u-page;
    for (uint32_t row=0u; row<rows; row++)
    {
        uint32_t context=row == 0u || full_context ? maximum_context : 1u+Random()%maximum_context;
        item->contexts.push_back(context); item->positions.push_back(context-1u);
        for (uint32_t step=0u; step<selected; step++)
            item->selection.push_back(step%13u == 5u ? 0xffffffffu : Random()%context);
    }
}

static void AttentionTiming(uint32_t rows,uint32_t context,uint32_t multiprocessors,cudaStream_t stream)
{
    AttentionCase item;
    AttentionDevice device;
    float old_ms,heads_ms;
    AttentionBuild(&item,rows,4u,context,0u,true);
    AttentionUpload(item,&device);
    old_ms=AttentionTime(item,&device,false,multiprocessors,stream);
    heads_ms=AttentionTime(item,&device,true,multiprocessors,stream);
    printf("TIMING latent attention rows=%u heads=4 context=%u per_head_kernel_ms=%.3f all_heads_kernel_ms=%.3f speedup=%.1f kv_gbps=%.0f\n",rows,context,old_ms,heads_ms,old_ms/heads_ms,(double)rows*context*GLM5_NEXT_LATENT*2.0/(heads_ms*1.0e6));
    AttentionFree(&device);
}

static void AttentionCaseRun(uint32_t rows,uint32_t heads,uint32_t maximum_context,uint32_t selected,uint32_t multiprocessors,cudaStream_t stream)
{
    const uint32_t latent=GLM5_NEXT_LATENT;
    AttentionCase item;
    std::vector<double> reference;
    double worst=0.0,worst_old=0.0;
    AttentionBuild(&item,rows,heads,maximum_context,selected,false);
    std::vector<uint16_t> fresh=AttentionRun(item,true,multiprocessors,stream),old=AttentionRun(item,false,multiprocessors,stream);
    for (uint32_t row=0u; row<rows; row++)
        for (uint32_t head=0u; head<heads; head++)
        {
            AttentionReference(item,row,head,reference);
            for (uint32_t e=0u; e<latent; e++)
            {
                uint64_t index=((uint64_t)row*heads+head)*latent+e;
                worst=std::max(worst,std::fabs(Bf16Value(fresh[index])-reference[e]));
                worst_old=std::max(worst_old,std::fabs(Bf16Value(old[index])-reference[e]));
            }
        }
    if (worst > 1.0e-2)
    {
        fprintf(stderr,"ATTENTION-MISMATCH rows=%u heads=%u context=%u selected=%u worst=%.6f old_kernel_worst=%.6f\n",rows,heads,maximum_context,selected,worst,worst_old);
        exit(1);
    }
    printf("PASS all-heads latent attention rows=%u heads=%u context=%u selected=%u multiprocessors=%u worst_abs=%.6f old_kernel_worst_abs=%.6f reference=f64\n",rows,heads,maximum_context,selected,multiprocessors,worst,worst_old);
}

int main(int argc,char **argv)
{
    cudaStream_t stream;
    cudaDeviceProp properties;
    if (argc != 2 || strcmp(argv[1],"--run") != 0)
    {
        puts("usage: test_glm5_next_rows_kernels --run");
        return 2;
    }
    alarm(900);
    setvbuf(stdout,nullptr,_IOLBF,0);
    CUDA(cudaSetDevice(0));
    CUDA(cudaGetDeviceProperties(&properties,0));
    CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    for (uint32_t rows : {2u,15u,16u,17u,40u})
        HeadCase(rows,1000u,false,stream);
    HeadCase(256u,9680u,true,stream);
    HeadCase(8u,9680u,true,stream);
    for (uint32_t heads : {1u,2u,4u})
    {
        AttentionCaseRun(3u,heads,700u,0u,(uint32_t)properties.multiProcessorCount,stream);
        AttentionCaseRun(2u,heads,3000u,256u,(uint32_t)properties.multiProcessorCount,stream);
        AttentionCaseRun(64u,heads,300u,0u,(uint32_t)properties.multiProcessorCount,stream);
        AttentionCaseRun(200u,heads,100u,0u,(uint32_t)properties.multiProcessorCount,stream);
    }
    AttentionTiming(8u,1024u,(uint32_t)properties.multiProcessorCount,stream);
    AttentionTiming(256u,1024u,(uint32_t)properties.multiProcessorCount,stream);
    CUDA(cudaStreamDestroy(stream));
    puts("PASS glm5_next row kernels: head rows kernel equals the per-row kernel bitwise; all-heads latent attention matches an f64 reference");
    return 0;
}
