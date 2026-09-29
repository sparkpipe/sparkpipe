#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_cuda.cu"

#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { fprintf(stderr,"FAIL line=%d cuda=%s call=%s\n",__LINE__,cudaGetErrorString(e),#call); exit(1); } } while(0)
#define REQUIRE(test) do { if(!(test)) { fprintf(stderr,"FAIL line=%d test=%s\n",__LINE__,#test); exit(1); } } while(0)

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

template<class Geometry, uint32_t THREADS, uint32_t LATENT, uint32_t ROPE>
__global__ __launch_bounds__(THREADS, 1)
void ReferenceSplitKernel(
    const uint16_t *__restrict__ query_latent_bf16,
    const uint16_t *__restrict__ query_rope_bf16,
    LmKvView cache,
    const uint32_t *__restrict__ sequence_of_row,
    const uint32_t *__restrict__ context_length,
    const uint32_t *__restrict__ selected_positions,
    uint32_t selected_count,
    uint32_t heads,
    uint32_t partitions,
    float qk_scale,
    float *__restrict__ partials,
    const uint32_t *__restrict__ row_position)
{
    static_assert(
        LATENT <= 8u * THREADS,
        "the latent must fit the per-thread accumulator");
    __shared__ float reduction[THREADS / LM_WARP_LANES];
    __shared__ float shared_query[LATENT + ROPE];
    float accumulator[8];
    uint32_t row = blockIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t partition = blockIdx.z;
    uint32_t index;
    uint32_t step;
    uint32_t position_count;
    uint32_t first_position;
    uint32_t last_position;
    uint32_t partition_span;
    uint32_t sequence = sequence_of_row[row];
    uint64_t latent_base = ((uint64_t)row * heads + head) * LATENT;
    uint64_t rope_base = ((uint64_t)row * heads + head) * ROPE;
    uint64_t partial_base;
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    if (!LmKvViewIsConfigured(cache) || sequence >= cache.sequence_count)
    {
        LmKvReportRequiredAccessFailure(
            cache,
            !LmKvViewIsConfigured(cache)
                ? LM_KV_ACCESS_ERROR_INVALID_VIEW
                : LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE,
            LM_KV_ACCESS_READ,
            row,
            sequence,
            0xffffffffu,
            0xffffffffu);
        return;
    }
    if (partition >= partitions)
    {
        return;
    }

    for (index = 0u; index < 8u; ++index)
    {
        accumulator[index] = 0.0f;
    }
    for (index = threadIdx.x; index < LATENT + ROPE; index += THREADS)
    {
        shared_query[index] = index < LATENT
            ? LmBf16ToFloat(query_latent_bf16[latent_base + index])
            : LmBf16ToFloat(query_rope_bf16[rope_base + index - LATENT]);
    }
    __syncthreads();

    position_count = selected_positions != 0
        ? selected_count
        : context_length[sequence];
    partition_span = (position_count + partitions - 1u) / partitions;
    first_position = partition * partition_span;
    last_position = first_position + partition_span;
    if (last_position > position_count)
    {
        last_position = position_count;
    }
    partial_base = (((uint64_t)row * heads + head) * partitions + partition) *
                   (LATENT + 2u);
    if (first_position >= last_position)
    {
        if (threadIdx.x == 0u)
        {
            partials[partial_base] = -INFINITY;
            partials[partial_base + 1u] = 0.0f;
        }
        for (index = threadIdx.x; index < LATENT; index += THREADS)
        {
            partials[partial_base + 2u + index] = 0.0f;
        }
        return;
    }
    for (step = first_position; step < last_position; ++step)
    {
        uint32_t position = selected_positions != 0
            ? selected_positions[(row * selected_count) + step]
            : step;
        const uint8_t *slot;
        float score = 0.0f;
        float scaled_previous;
        float scaled_current;
        float previous_max;

        if (row_position != 0 && position > row_position[row])
        {
            continue;
        }
        slot = LmKvSlotRequired<Geometry>(
            cache, sequence, position, row, LM_KV_ACCESS_READ);
        if (slot == 0)
        {
            if (threadIdx.x == 0u)
            {
                partials[partial_base] = running_max;
                partials[partial_base + 1u] = running_sum;
            }
            for (index = 0u; index < 8u; ++index)
            {
                uint32_t element = (index * THREADS) + threadIdx.x;

                if (element < LATENT)
                {
                    partials[partial_base + 2u + element] = accumulator[index];
                }
            }
            return;
        }
        for (index = threadIdx.x; index < LATENT + ROPE; index += THREADS)
        {
            score += shared_query[index] *
                LmBf16ToFloat(((const uint16_t *)slot)[index]);
        }
        score = LmBlockSum<THREADS>(score, reduction) * qk_scale;
        previous_max = running_max;
        running_max = fmaxf(running_max, score);
        scaled_previous = __expf(previous_max - running_max);
        scaled_current = __expf(score - running_max);
        running_sum = (running_sum * scaled_previous) + scaled_current;
        for (index = 0u; index < 8u; ++index)
        {
            uint32_t element = (index * THREADS) + threadIdx.x;

            if (element < LATENT)
            {
                accumulator[index] =
                    (accumulator[index] * scaled_previous) +
                    (scaled_current *
                        LmBf16ToFloat(((const uint16_t *)slot)[element]));
            }
        }
    }
    if (threadIdx.x == 0u)
    {
        partials[partial_base] = running_max;
        partials[partial_base + 1u] = running_sum;
    }
    for (index = 0u; index < 8u; ++index)
    {
        uint32_t element = (index * THREADS) + threadIdx.x;

        if (element < LATENT)
        {
            partials[partial_base + 2u + element] = accumulator[index];
        }
    }
}

typedef struct AttentionCase
{
    uint32_t rows,heads,pages_per_sequence,selected;
    std::vector<uint16_t> pool,query;
    std::vector<uint32_t> table,contexts,positions,selection;
}
AttentionCase;

typedef struct AttentionDevice
{
    uint16_t *pool,*query;
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
    CUDA(cudaMalloc(&device->partials,(uint64_t)item.rows*item.heads*LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS*(GLM5_NEXT_LATENT+2u)*4u));
    REQUIRE(LmKvViewInitialize(&device->view,(uint8_t *)device->pool,device->table,item.pages_per_sequence,item.rows,item.rows*item.pages_per_sequence,device->error) == 0);
}

static void AttentionFree(AttentionDevice *device)
{
    CUDA(cudaFree(device->pool)); CUDA(cudaFree(device->query)); CUDA(cudaFree(device->table)); CUDA(cudaFree(device->contexts)); CUDA(cudaFree(device->positions)); CUDA(cudaFree(device->selection)); CUDA(cudaFree(device->sequences));
    CUDA(cudaFree(device->error)); CUDA(cudaFree(device->partials));
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

static std::vector<float> SplitRun(const AttentionCase &item,uint32_t partitions,bool reference,LmKvAccessError *error,float *milliseconds,cudaStream_t stream)
{
    AttentionDevice device;
    const uint32_t *selection;
    const uint64_t floats=(uint64_t)item.rows*item.heads*partitions*(GLM5_NEXT_LATENT+2u);
    const uint32_t repeats=milliseconds != 0 ? 20u : 1u;
    cudaEvent_t begin,end;
    AttentionUpload(item,&device);
    selection=item.selected != 0u ? device.selection : 0;
    CUDA(cudaMemset(device.partials,0x7f,floats*4u));
    CUDA(cudaEventCreate(&begin)); CUDA(cudaEventCreate(&end));
    CUDA(cudaEventRecord(begin,stream));
    for (uint32_t repeat=0u; repeat<repeats; repeat++)
    {
        if (reference)
            ReferenceSplitKernel<Glm5NextKv,GLM5_NEXT_ATTN_THREADS,GLM5_NEXT_LATENT,GLM5_NEXT_ROPE_DIM><<<dim3(item.rows,item.heads,partitions),GLM5_NEXT_ATTN_THREADS,0,stream>>>(device.query,0,device.view,device.sequences,device.contexts,selection,item.selected,item.heads,partitions,0.0625f,device.partials,device.positions);
        else
            LmLatentAttentionDecodeSplitKernel<Glm5NextKv,GLM5_NEXT_ATTN_THREADS,GLM5_NEXT_LATENT,GLM5_NEXT_ROPE_DIM><<<dim3(item.rows,item.heads,partitions),GLM5_NEXT_ATTN_THREADS,0,stream>>>(device.query,0,device.view,device.sequences,device.contexts,selection,item.selected,item.heads,partitions,0.0625f,device.partials,device.positions);
    }
    CUDA(cudaEventRecord(end,stream));
    CUDA(cudaPeekAtLastError());
    CUDA(cudaStreamSynchronize(stream));
    if (milliseconds != 0) { CUDA(cudaEventElapsedTime(milliseconds,begin,end)); *milliseconds/=(float)repeats; }
    CUDA(cudaMemcpy(error,device.error,sizeof(*error),cudaMemcpyDeviceToHost));
    std::vector<float> result=Download(device.partials,floats);
    CUDA(cudaEventDestroy(begin)); CUDA(cudaEventDestroy(end));
    AttentionFree(&device);
    return result;
}

static void Case(uint32_t rows,uint32_t heads,uint32_t context,uint32_t selected,uint32_t partitions,bool full,bool break_page,cudaStream_t stream)
{
    AttentionCase item;
    LmKvAccessError error_reference,error_grouped;
    AttentionBuild(&item,rows,heads,context,selected,full);
    if (break_page)
        item.table[item.table.size()/2u]=0xffffffffu;
    std::vector<float> reference=SplitRun(item,partitions,true,&error_reference,0,stream),grouped=SplitRun(item,partitions,false,&error_grouped,0,stream);
    REQUIRE(reference.size() == grouped.size());
    for (size_t index=0u; index<reference.size(); index++)
        if (memcmp(&reference[index],&grouped[index],4u) != 0)
        {
            fprintf(stderr,"MISMATCH rows=%u heads=%u context=%u selected=%u partitions=%u float=%zu reference=%.9g grouped=%.9g\n",rows,heads,context,selected,partitions,index,reference[index],grouped[index]);
            exit(1);
        }
    REQUIRE(memcmp(&error_reference,&error_grouped,sizeof(error_reference)) == 0);
    REQUIRE(break_page == (error_reference.error_code != LM_KV_ACCESS_ERROR_NONE));
    printf("PASS rows=%u heads=%u context=%u selected=%u partitions=%u full=%u broken_page=%u\n",rows,heads,context,selected,partitions,full ? 1u : 0u,break_page ? 1u : 0u);
}

static void Time(uint32_t context,uint32_t partitions,cudaStream_t stream)
{
    AttentionCase item;
    LmKvAccessError error;
    float reference_ms,grouped_ms;
    AttentionBuild(&item,1u,4u,context,0u,true);
    SplitRun(item,partitions,true,&error,&reference_ms,stream);
    SplitRun(item,partitions,false,&error,&grouped_ms,stream);
    printf("TIME split rows=1 heads=4 context=%u partitions=%u reference_us=%.1f grouped_us=%.1f\n",context,partitions,1000.0f*reference_ms,1000.0f*grouped_ms);
}

int main(int argc,char **argv)
{
    cudaStream_t stream;
    if (argc < 2 || (strcmp(argv[1],"--run") != 0 && strcmp(argv[1],"--time") != 0))
    {
        fprintf(stderr,"usage: test_latent_attention_split_group --run|--time\n");
        return 2;
    }
    CUDA(cudaStreamCreate(&stream));
    if (strcmp(argv[1],"--time") == 0)
    {
        Time(1024u,16u,stream);
        Time(4096u,16u,stream);
        return 0;
    }
    Case(1u,4u,1024u,0u,16u,true,false,stream);
    Case(1u,4u,1000u,0u,16u,true,false,stream);
    Case(3u,4u,700u,0u,5u,false,false,stream);
    Case(2u,4u,3000u,256u,16u,false,false,stream);
    Case(2u,4u,3000u,256u,1u,false,false,stream);
    Case(4u,2u,90u,0u,16u,false,false,stream);
    Case(1u,4u,2048u,0u,16u,true,true,stream);
    Case(2u,4u,777u,300u,7u,false,true,stream);
    printf("test_latent_attention_split_group PASS\n");
    return 0;
}
