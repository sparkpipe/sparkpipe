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

typedef struct IndexCase
{
    uint32_t rows,pages_per_sequence,pools;
    uint8_t *pool;
    uint32_t *page_table,*sequence_of_row,*context_length,*row_positions;
    uint16_t *query,*head_weight;
    float *ape,*reference,*local,*gathered,*permuted;
    LmKvAccessError *error;
    LmKvView view;
}
IndexCase;

static uint32_t random_state = 20260925u;

static uint32_t Random()
{
    random_state ^= random_state << 13u;
    random_state ^= random_state >> 17u;
    random_state ^= random_state << 5u;
    return random_state;
}

static uint16_t RandomBf16(float scale)
{
    float value=(((int32_t)(Random()%2049u)-1024)/1024.0f)*scale;
    uint32_t bits;
    memcpy(&bits,&value,sizeof(bits));
    return (uint16_t)(bits>>16u);
}

template<class T> static T *Upload(const std::vector<T> &host)
{
    T *device;
    CUDA(cudaMalloc(&device,host.size()*sizeof(T)));
    CUDA(cudaMemcpy(device,host.data(),host.size()*sizeof(T),cudaMemcpyHostToDevice));
    return device;
}

static void IndexCaseBuild(IndexCase *item,const std::vector<uint32_t> &contexts)
{
    uint32_t rows=(uint32_t)contexts.size(),maximum=0u,pages,page;
    std::vector<uint8_t> pool;
    std::vector<uint32_t> table,sequence(rows),context(rows),position(rows);
    std::vector<uint16_t> query((uint64_t)rows*GLM5_NEXT_DSA_INDEX_HEADS*GLM5_NEXT_DSA_INDEX_DIM),weight((uint64_t)rows*GLM5_NEXT_DSA_INDEX_HEADS);
    std::vector<float> ape(GLM5_NEXT_DSA_KPOOL*GLM5_NEXT_DSA_INDEX_DIM);
    for (uint32_t row=0u; row<rows; row++)
    {
        maximum=std::max(maximum,contexts[row]); sequence[row]=row; context[row]=contexts[row]; position[row]=contexts[row]-1u;
    }
    item->rows=rows; item->pools=SparkGlm5NextIndexCpPools(maximum);
    item->pages_per_sequence=(maximum+GLM5_NEXT_KV_PAGE_SLOTS-1u)/GLM5_NEXT_KV_PAGE_SLOTS;
    pages=rows*item->pages_per_sequence;
    table.resize(pages); pool.resize((uint64_t)pages*Glm5NextIndexKv::kPageBytes);
    for (page=0u; page<pages; page++) table[page]=pages-1u-page;
    for (uint64_t index=0u; index<pool.size()/2u; index++) { uint16_t value=RandomBf16(1.0f); memcpy(&pool[index*2u],&value,2u); }
    for (auto &value : query) value=RandomBf16(1.0f);
    for (auto &value : weight) value=RandomBf16(1.0f);
    for (auto &value : ape) value=(((int32_t)(Random()%2049u)-1024)/4096.0f);
    item->pool=Upload(pool); item->page_table=Upload(table); item->sequence_of_row=Upload(sequence); item->context_length=Upload(context); item->row_positions=Upload(position);
    item->query=Upload(query); item->head_weight=Upload(weight); item->ape=Upload(ape);
    CUDA(cudaMalloc(&item->error,sizeof(LmKvAccessError))); CUDA(cudaMemset(item->error,0,sizeof(LmKvAccessError)));
    REQUIRE(LmKvViewInitialize(&item->view,item->pool,item->page_table,item->pages_per_sequence,rows,pages,item->error) == 0);
    CUDA(cudaMalloc(&item->reference,(uint64_t)rows*item->pools*sizeof(float)));
    CUDA(cudaMalloc(&item->permuted,(uint64_t)rows*item->pools*sizeof(float)));
    CUDA(cudaMalloc(&item->local,(uint64_t)rows*SPARK_GLM5_NEXT_INDEX_CP_SEQUENCE_FLOATS*sizeof(float)));
    CUDA(cudaMalloc(&item->gathered,(uint64_t)16u*rows*SPARK_GLM5_NEXT_INDEX_CP_SEQUENCE_FLOATS*sizeof(float)));
}

static void IndexCaseFree(IndexCase *item)
{
    CUDA(cudaFree(item->pool)); CUDA(cudaFree(item->page_table)); CUDA(cudaFree(item->sequence_of_row)); CUDA(cudaFree(item->context_length)); CUDA(cudaFree(item->row_positions));
    CUDA(cudaFree(item->query)); CUDA(cudaFree(item->head_weight)); CUDA(cudaFree(item->ape)); CUDA(cudaFree(item->error));
    CUDA(cudaFree(item->reference)); CUDA(cudaFree(item->permuted)); CUDA(cudaFree(item->local)); CUDA(cudaFree(item->gathered));
}

static void IndexScore(const IndexCase *item,uint32_t rank,uint32_t degree,float *output,cudaStream_t stream)
{
    uint32_t stride=SparkGlm5NextIndexCpLocalStride(item->pools,degree);
    Glm5NextPoolScoreKernel<GLM5_NEXT_LAYER_THREADS,GLM5_NEXT_DSA_INDEX_DIM,GLM5_NEXT_DSA_KPOOL,GLM5_NEXT_DSA_INDEX_HEADS><<<dim3(stride,item->rows),GLM5_NEXT_LAYER_THREADS,0,stream>>>(
        item->query,item->head_weight,item->view,item->sequence_of_row,item->context_length,item->row_positions,item->ape,item->pools,rank,degree,stride,GLM5_NEXT_DSA_INDEX_SCALE,GLM5_NEXT_DSA_INDEX_HEAD_WEIGHT_SCALE,output);
    CUDA(cudaPeekAtLastError());
}

static void IndexCpCase(const std::vector<uint32_t> &contexts,uint32_t degree,cudaStream_t stream)
{
    IndexCase item;
    LmKvAccessError error;
    uint32_t stride;
    uint64_t peer_floats;
    std::vector<float> reference,permuted;
    IndexCaseBuild(&item,contexts);
    stride=SparkGlm5NextIndexCpLocalStride(item.pools,degree);
    peer_floats=(uint64_t)SparkGlm5NextIndexCpGatherSequences(item.rows,stride)*SPARK_GLM5_NEXT_INDEX_CP_SEQUENCE_FLOATS;
    REQUIRE(SparkGlm5NextIndexCpGatherSequences(item.rows,stride) <= item.rows);
    IndexScore(&item,0u,1u,item.reference,stream);
    CUDA(cudaMemsetAsync(item.gathered,0xff,(uint64_t)degree*peer_floats*sizeof(float),stream));
    for (uint32_t rank=0u; rank<degree; rank++)
    {
        IndexScore(&item,rank,degree,item.local,stream);
        CUDA(cudaMemcpyAsync(item.gathered+rank*peer_floats,item.local,(uint64_t)item.rows*stride*sizeof(float),cudaMemcpyDeviceToDevice,stream));
    }
    Glm5NextPoolPermuteKernel<GLM5_NEXT_LAYER_THREADS><<<item.rows,GLM5_NEXT_LAYER_THREADS,0,stream>>>(item.gathered,item.permuted,item.pools,stride,peer_floats,degree);
    CUDA(cudaPeekAtLastError());
    CUDA(cudaStreamSynchronize(stream));
    CUDA(cudaMemcpy(&error,item.error,sizeof(error),cudaMemcpyDeviceToHost));
    REQUIRE(error.error_code == LM_KV_ACCESS_ERROR_NONE);
    reference.resize((uint64_t)item.rows*item.pools); permuted.resize(reference.size());
    CUDA(cudaMemcpy(reference.data(),item.reference,reference.size()*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(permuted.data(),item.permuted,permuted.size()*sizeof(float),cudaMemcpyDeviceToHost));
    for (uint64_t index=0u; index<reference.size(); index++)
        if (memcmp(&reference[index],&permuted[index],sizeof(float)) != 0)
        {
            fprintf(stderr,"MISMATCH degree=%u row=%llu pool=%llu reference=%.9g context_parallel=%.9g\n",degree,(unsigned long long)(index/item.pools),(unsigned long long)(index%item.pools),reference[index],permuted[index]);
            exit(1);
        }
    printf("PASS index context parallel degree=%u rows=%u pools=%u local_stride=%u bitwise_equal=yes\n",degree,item.rows,item.pools,stride);
    IndexCaseFree(&item);
}

int main(int argc,char **argv)
{
    cudaStream_t stream;
    if (argc != 2 || strcmp(argv[1],"--run") != 0)
    {
        puts("usage: test_glm5_next_index_cp --run");
        return 2;
    }
    alarm(600);
    setvbuf(stdout,nullptr,_IOLBF,0);
    CUDA(cudaSetDevice(0));
    CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    for (uint32_t degree : {2u,3u,16u})
    {
        IndexCpCase({2049u},degree,stream);
        IndexCpCase({9001u,4100u,20000u},degree,stream);
    }
    CUDA(cudaStreamDestroy(stream));
    puts("PASS glm5_next index context parallel: owned-pool scores gathered and permuted equal the replicated scores bit for bit");
    return 0;
}
