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

static uint32_t random_state = 20260928u;

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

struct Rank
{
    SparkGlm5NextExecutionSlot slot;
    SparkGlm5NextCudaWave wave;
    Glm5NextLayerBuffers buffers;
    uint8_t *kv_pool,*index_pool;
    LmKvAccessError *error;
};

struct Case
{
    uint32_t degree,rows,capacity,max_context,pages_per_sequence,pages,heads,dsa_layer;
    std::vector<uint32_t> contexts;
    uint32_t *page_table,*sequence_of_row,*context_length,*positions,*selected;
    uint16_t *kv_rows,*index_rows,*index_query,*index_head_weight;
    float *index_ape;
    std::vector<Rank> ranks;
    SparkGlm5NextLayerWeights layers[SPARK_GLM5_NEXT_MODEL_LAYER_COUNT];
    uint32_t index_ordinal[SPARK_GLM5_NEXT_MODEL_LAYER_COUNT];
    uint32_t kda_ordinal[SPARK_GLM5_NEXT_MODEL_LAYER_COUNT];
};

static uint32_t FirstDsaLayer()
{
    for (uint32_t layer=0u; layer<SPARK_GLM5_NEXT_MODEL_LAYER_COUNT; layer++)
        if (!SPARK_GLM5_NEXT_MODEL_LAYER_IS_KDA(layer) && index_ordinal_of(nullptr,layer,layer) != UINT32_MAX) return layer;
    REQUIRE(false);
    return 0u;
}

static void RankBuild(Case *item,uint32_t rank,uint32_t shard)
{
    Rank &r=item->ranks[rank];
    uint32_t degree=shard!=0u ? item->degree : 1u,rows=item->rows;
    uint64_t kv_page=shard!=0u ? SparkKvShardPageBytes(SparkGlm5NextKvShardLatent(rank,degree),GLM5_NEXT_KV_PAGE_SLOTS,Glm5NextKv::kSlotBytes) : Glm5NextKv::kPageBytes;
    uint64_t index_page=shard!=0u ? SparkKvShardPageBytes(SparkGlm5NextKvShardIndex(rank,degree),GLM5_NEXT_KV_PAGE_SLOTS,Glm5NextIndexKv::kSlotBytes) : Glm5NextIndexKv::kPageBytes;
    uint32_t cp=item->degree;
    memset(&r.slot,0,sizeof(r.slot));memset(&r.wave,0,sizeof(r.wave));
    CUDA(cudaMalloc(&r.kv_pool,(uint64_t)item->pages*kv_page));CUDA(cudaMemset(r.kv_pool,0xa5,(uint64_t)item->pages*kv_page));
    CUDA(cudaMalloc(&r.index_pool,(uint64_t)item->pages*index_page));CUDA(cudaMemset(r.index_pool,0x5a,(uint64_t)item->pages*index_page));
    CUDA(cudaMalloc(&r.error,sizeof(LmKvAccessError)));CUDA(cudaMemset(r.error,0,sizeof(LmKvAccessError)));
    CUDA(cudaStreamCreateWithFlags((cudaStream_t *)&r.slot.stream,cudaStreamNonBlocking));
    r.slot.kv_access_error=r.error;
    r.slot.selected_positions=item->selected;
    CUDA(cudaMalloc(&r.slot.query_latent_bf16,(uint64_t)rows*SPARK_GLM5_NEXT_MODEL_HEAD_COUNT*GLM5_NEXT_LATENT*2u));
    CUDA(cudaMalloc(&r.slot.attention_latent_bf16,(uint64_t)rows*SPARK_GLM5_NEXT_MODEL_HEAD_COUNT*GLM5_NEXT_LATENT*2u));
    CUDA(cudaMalloc(&r.slot.selection_scores_f32,(uint64_t)rows*SparkGlm5NextIndexCpPools(item->max_context)*sizeof(float)));
    CUDA(cudaMalloc(&r.slot.selected_pools,(uint64_t)rows*(GLM5_NEXT_DSA_SELECTED/GLM5_NEXT_DSA_KPOOL)*sizeof(uint32_t)));
    CUDA(cudaMalloc(&r.slot.index_local_scores_f32,(uint64_t)rows*SPARK_GLM5_NEXT_INDEX_CP_SEQUENCE_FLOATS*sizeof(float)));
    CUDA(cudaMalloc(&r.slot.index_gathered_scores_f32,(uint64_t)cp*rows*SPARK_GLM5_NEXT_INDEX_CP_SEQUENCE_FLOATS*sizeof(float)));
    CUDA(cudaMalloc(&r.slot.kv_shard_query_gathered_bf16,(uint64_t)item->degree*SparkGlm5NextKvShardQueryStride(rows,item->degree)*2u));
    CUDA(cudaMalloc(&r.slot.kv_shard_partials_f32,(uint64_t)item->degree*SparkGlm5NextKvShardPartialStrideCapacity(item->degree,item->capacity)*sizeof(float)));
    CUDA(cudaMalloc(&r.slot.kv_shard_partials_received_f32,(uint64_t)item->degree*SparkGlm5NextKvShardPartialStrideCapacity(item->degree,item->capacity)*sizeof(float)));
    r.wave.first_layer_index=0u;r.wave.layer_count=SPARK_GLM5_NEXT_MODEL_LAYER_COUNT;r.wave.layers=item->layers;
    r.wave.tp_degree=item->degree;r.wave.tp_rank=rank;r.wave.index_cp_degree=cp;r.wave.kv_shard=shard;
    r.wave.row_count=rows;r.wave.maximum_context=item->max_context;r.wave.resident_sequence_capacity=rows;r.wave.max_sequence_positions=item->max_context;
    r.wave.pages_per_sequence=item->pages_per_sequence;r.wave.physical_page_count=item->pages;r.wave.slot=&r.slot;
    r.wave.kv_cache=r.kv_pool;r.wave.kv_layer_stride_bytes=(uint64_t)item->pages*kv_page;
    r.wave.index_cache=r.index_pool;r.wave.index_layer_stride_bytes=(uint64_t)item->pages*index_page;
    r.wave.index_ordinal_by_local_layer=item->index_ordinal;r.wave.kda_ordinal_by_local_layer=item->kda_ordinal;r.wave.page_table=item->page_table;r.wave.multiprocessor_count=48u;r.wave.execution_row_capacity=item->capacity;
    SparkGlm5NextBindLayer(&r.wave,item->dsa_layer,&r.buffers);
    r.buffers.sequence_of_row=item->sequence_of_row;r.buffers.context_length=item->context_length;
    r.buffers.positions=item->positions;r.buffers.row_positions=item->positions;
    r.buffers.kv_slot_bf16=item->kv_rows;r.buffers.index_packed_bf16=item->index_rows;
    r.buffers.index_query_bf16=item->index_query;r.buffers.index_head_weight_bf16=item->index_head_weight;r.buffers.index_compress_ape=item->index_ape;
    r.buffers.qk_scale=SPARK_GLM5_NEXT_MODEL_MLA_QK_SCALE;
    r.buffers.selected_positions=item->selected;r.buffers.selected_position_count=SPARK_GLM5_NEXT_MODEL_INDEX_OUTPUT_WIDTH;
    REQUIRE(r.buffers.kv_shard_active==shard);
    REQUIRE(shard==0u || (r.buffers.kv_shard.degree==item->degree && r.buffers.kv_shard.rank==rank && r.buffers.kv_shard.grain==1u && r.buffers.index_shard.grain==GLM5_NEXT_DSA_KPOOL));
    REQUIRE(shard==0u || r.buffers.shard_query_stride==SparkGlm5NextKvShardQueryStride(rows,item->degree));
    REQUIRE(shard==0u || (r.buffers.shard_row_capacity==item->capacity && r.buffers.shard_partial_stride==SparkGlm5NextKvShardPartialStride(rows,item->degree,item->capacity)));
}

static void RankFree(Rank &r)
{
    CUDA(cudaStreamDestroy((cudaStream_t)r.slot.stream));
    CUDA(cudaFree(r.kv_pool));CUDA(cudaFree(r.index_pool));CUDA(cudaFree(r.error));
    CUDA(cudaFree(r.slot.query_latent_bf16));CUDA(cudaFree(r.slot.attention_latent_bf16));CUDA(cudaFree(r.slot.selection_scores_f32));CUDA(cudaFree(r.slot.selected_pools));
    CUDA(cudaFree(r.slot.index_local_scores_f32));CUDA(cudaFree(r.slot.index_gathered_scores_f32));
    CUDA(cudaFree(r.slot.kv_shard_query_gathered_bf16));CUDA(cudaFree(r.slot.kv_shard_partials_f32));CUDA(cudaFree(r.slot.kv_shard_partials_received_f32));
}

static void Fill(Case *item)
{
    uint32_t rows=item->rows,maximum=0u;
    for (uint32_t c : item->contexts) maximum=std::max(maximum,c);
    item->max_context=maximum;item->heads=SparkGlm5NextKvShardHeads(item->degree);item->dsa_layer=FirstDsaLayer();
    memset(item->layers,0,sizeof(item->layers));
    for (uint32_t layer=0u; layer<SPARK_GLM5_NEXT_MODEL_LAYER_COUNT; layer++) { item->index_ordinal[layer]=index_ordinal_of(nullptr,layer,layer);item->kda_ordinal[layer]=UINT32_MAX; }
    item->pages_per_sequence=(maximum+GLM5_NEXT_KV_PAGE_SLOTS-1u)/GLM5_NEXT_KV_PAGE_SLOTS;item->pages=rows*item->pages_per_sequence;
    std::vector<uint32_t> table(item->pages),sequence(rows),context(rows),position(rows);
    for (uint32_t page=0u; page<item->pages; page++) table[page]=(page*7u+3u)%item->pages;
    for (uint32_t row=0u; row<rows; row++) { sequence[row]=row;context[row]=item->contexts[row];position[row]=item->contexts[row]-1u; }
    std::vector<uint32_t> selected((uint64_t)rows*SPARK_GLM5_NEXT_MODEL_INDEX_OUTPUT_WIDTH,0xffffffffu);
    for (uint32_t row=0u; row<rows; row++)
    {
        std::vector<uint32_t> order(item->contexts[row]);
        for (uint32_t i=0u; i<order.size(); i++) order[i]=i;
        for (uint32_t i=(uint32_t)order.size(); i>1u; i--) std::swap(order[i-1u],order[Random()%i]);
        for (uint32_t i=0u; i<GLM5_NEXT_DSA_SELECTED && i<order.size(); i++) selected[(uint64_t)row*SPARK_GLM5_NEXT_MODEL_INDEX_OUTPUT_WIDTH+i]=order[i];
    }
    item->page_table=Upload(table);item->sequence_of_row=Upload(sequence);item->context_length=Upload(context);item->positions=Upload(position);item->selected=Upload(selected);
    std::vector<uint16_t> index_query((uint64_t)rows*GLM5_NEXT_DSA_INDEX_HEADS*GLM5_NEXT_DSA_INDEX_DIM),index_weight((uint64_t)rows*GLM5_NEXT_DSA_INDEX_HEADS);
    std::vector<float> ape(GLM5_NEXT_DSA_KPOOL*GLM5_NEXT_DSA_INDEX_DIM);
    for (auto &v : index_query) v=RandomBf16(1.0f);
    for (auto &v : index_weight) v=RandomBf16(1.0f);
    for (auto &v : ape) v=(((int32_t)(Random()%2049u)-1024)/4096.0f);
    item->index_query=Upload(index_query);item->index_head_weight=Upload(index_weight);item->index_ape=Upload(ape);
    CUDA(cudaMalloc(&item->kv_rows,(uint64_t)maximum*GLM5_NEXT_LATENT_ROW*2u));
    CUDA(cudaMalloc(&item->index_rows,(uint64_t)maximum*SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION*2u));
}

static void StoreAllPositions(Case *item)
{
    uint32_t maximum=item->max_context;
    std::vector<uint32_t> seq(maximum),pos(maximum);
    std::vector<uint16_t> kv((uint64_t)maximum*GLM5_NEXT_LATENT_ROW),index((uint64_t)maximum*SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION);
    uint32_t *seq_d,*pos_d;
    CUDA(cudaMalloc(&seq_d,maximum*sizeof(uint32_t)));CUDA(cudaMalloc(&pos_d,maximum*sizeof(uint32_t)));
    for (uint32_t row=0u; row<item->rows; row++)
    {
        uint32_t count=item->contexts[row];
        uint32_t saved=random_state;
        random_state=0x9e3779b9u*(row+1u);
        for (uint32_t p=0u; p<count; p++) { seq[p]=row;pos[p]=p; }
        for (auto &v : kv) v=RandomBf16(1.0f);
        for (auto &v : index) v=RandomBf16(1.0f);
        random_state=saved;
        CUDA(cudaMemcpy(item->kv_rows,kv.data(),(uint64_t)count*GLM5_NEXT_LATENT_ROW*2u,cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(item->index_rows,index.data(),(uint64_t)count*SPARK_GLM5_NEXT_MODEL_INDEX_PACKED_TOKEN_DIMENSION*2u,cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(seq_d,seq.data(),count*sizeof(uint32_t),cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(pos_d,pos.data(),count*sizeof(uint32_t),cudaMemcpyHostToDevice));
        for (Rank &r : item->ranks)
        {
            Glm5NextLayerBuffers b=r.buffers;
            b.sequence_of_row=seq_d;b.positions=pos_d;
            REQUIRE(Glm5NextLayerLatentStore(&b,count,(cudaStream_t)r.slot.stream)==LM_LAUNCH_OK);
            REQUIRE(Glm5NextLayerIndexStore(&b,count,(cudaStream_t)r.slot.stream)==LM_LAUNCH_OK);
            CUDA(cudaStreamSynchronize((cudaStream_t)r.slot.stream));
        }
    }
    CUDA(cudaFree(seq_d));CUDA(cudaFree(pos_d));
}

static void CheckNoAccessError(Case *item)
{
    for (Rank &r : item->ranks)
    {
        LmKvAccessError error;
        CUDA(cudaMemcpy(&error,r.error,sizeof(error),cudaMemcpyDeviceToHost));
        if (error.error_code != LM_KV_ACCESS_ERROR_NONE)
        {
            fprintf(stderr,"FAIL kv access error code=%u\n",(unsigned)error.error_code);
            exit(1);
        }
    }
}

static void CheckOwnedSlots(Case *shard,Case *replica)
{
    uint64_t checked=0u;
    std::vector<uint8_t> full_kv=Download(replica->ranks[0].kv_pool,(uint64_t)replica->pages*Glm5NextKv::kPageBytes);
    std::vector<uint8_t> full_index=Download(replica->ranks[0].index_pool,(uint64_t)replica->pages*Glm5NextIndexKv::kPageBytes);
    std::vector<uint32_t> table=Download(shard->page_table,shard->pages);
    for (uint32_t rank=0u; rank<shard->degree; rank++)
    {
        SparkKvShard latent=SparkGlm5NextKvShardLatent(rank,shard->degree),index=SparkGlm5NextKvShardIndex(rank,shard->degree);
        uint64_t kv_page=SparkKvShardPageBytes(latent,GLM5_NEXT_KV_PAGE_SLOTS,Glm5NextKv::kSlotBytes),index_page=SparkKvShardPageBytes(index,GLM5_NEXT_KV_PAGE_SLOTS,Glm5NextIndexKv::kSlotBytes);
        REQUIRE(kv_page*shard->degree==Glm5NextKv::kPageBytes && index_page*shard->degree==Glm5NextIndexKv::kPageBytes);
        std::vector<uint8_t> kv=Download(shard->ranks[rank].kv_pool,(uint64_t)shard->pages*kv_page),ix=Download(shard->ranks[rank].index_pool,(uint64_t)shard->pages*index_page);
        for (uint32_t row=0u; row<shard->rows; row++)
            for (uint32_t p=0u; p<shard->contexts[row]; p++)
            {
                uint32_t physical=table[(uint64_t)row*shard->pages_per_sequence+p/GLM5_NEXT_KV_PAGE_SLOTS];
                if (SparkKvShardOwns(latent,p))
                {
                    REQUIRE(memcmp(kv.data()+physical*kv_page+(uint64_t)SparkKvShardSlotInPage(latent,GLM5_NEXT_KV_PAGE_SLOTS,p)*Glm5NextKv::kSlotBytes,
                        full_kv.data()+physical*(uint64_t)Glm5NextKv::kPageBytes+(uint64_t)(p%GLM5_NEXT_KV_PAGE_SLOTS)*Glm5NextKv::kSlotBytes,Glm5NextKv::kSlotBytes)==0);
                    checked++;
                }
                if (SparkKvShardOwns(index,p))
                {
                    REQUIRE(memcmp(ix.data()+physical*index_page+(uint64_t)SparkKvShardSlotInPage(index,GLM5_NEXT_KV_PAGE_SLOTS,p)*Glm5NextIndexKv::kSlotBytes,
                        full_index.data()+physical*(uint64_t)Glm5NextIndexKv::kPageBytes+(uint64_t)(p%GLM5_NEXT_KV_PAGE_SLOTS)*Glm5NextIndexKv::kSlotBytes,Glm5NextIndexKv::kSlotBytes)==0);
                    checked++;
                }
            }
    }
    uint64_t expected=0u;
    for (uint32_t c : shard->contexts) expected+=2u*(uint64_t)c;
    REQUIRE(checked==expected);
}

static void CheckIndexScores(Case *shard,Case *replica)
{
    uint32_t rows=shard->rows,pools=SparkGlm5NextIndexCpPools(shard->max_context),stride=SparkGlm5NextIndexCpLocalStride(pools,shard->degree);
    uint64_t peer=(uint64_t)SparkGlm5NextIndexCpGatherSequences(rows,stride)*SPARK_GLM5_NEXT_INDEX_CP_SEQUENCE_FLOATS;
    Glm5NextLayerBuffers reference=replica->ranks[0].buffers;
    reference.index_owner_degree=1u;reference.index_owner_rank=0u;
    REQUIRE(Glm5NextLayerIndexPoolScore(&reference,rows,shard->max_context,(cudaStream_t)replica->ranks[0].slot.stream)==LM_LAUNCH_OK);
    CUDA(cudaStreamSynchronize((cudaStream_t)replica->ranks[0].slot.stream));
    for (Rank &r : shard->ranks)
    {
        REQUIRE(Glm5NextLayerIndexPoolScore(&r.buffers,rows,shard->max_context,(cudaStream_t)r.slot.stream)==LM_LAUNCH_OK);
        CUDA(cudaStreamSynchronize((cudaStream_t)r.slot.stream));
    }
    float *gathered=shard->ranks[0].slot.index_gathered_scores_f32;
    for (uint32_t rank=0u; rank<shard->degree; rank++)
        CUDA(cudaMemcpy(gathered+rank*peer,shard->ranks[rank].slot.index_local_scores_f32,(uint64_t)rows*stride*sizeof(float),cudaMemcpyDeviceToDevice));
    CUDA(cudaDeviceSynchronize());
    Glm5NextPoolPermuteKernel<GLM5_NEXT_LAYER_THREADS><<<rows,GLM5_NEXT_LAYER_THREADS>>>(gathered,shard->ranks[0].slot.selection_scores_f32,pools,stride,peer,shard->degree);
    CUDA(cudaDeviceSynchronize());
    std::vector<float> want=Download(replica->ranks[0].slot.selection_scores_f32,(uint64_t)rows*pools),got=Download(shard->ranks[0].slot.selection_scores_f32,(uint64_t)rows*pools);
    REQUIRE(memcmp(want.data(),got.data(),want.size()*sizeof(float))==0);
}

static void Attention(Case *item,uint32_t oracle,Case *replica,std::vector<uint16_t> &merged)
{
    uint32_t rows=item->rows,degree=item->degree,heads=item->heads;
    uint64_t qstride=SparkGlm5NextKvShardQueryStride(rows,degree),pstride=SparkGlm5NextKvShardPartialStride(rows,degree,item->capacity);
    std::vector<std::vector<uint16_t>> query(degree);
    uint32_t saved=random_state;
    random_state=0x1234567u+rows*31u+degree;
    for (uint32_t rank=0u; rank<degree; rank++)
    {
        query[rank].assign((uint64_t)rows*heads*GLM5_NEXT_LATENT,0u);
        for (auto &v : query[rank]) v=RandomBf16(1.0f);
        CUDA(cudaMemcpy(item->ranks[rank].slot.query_latent_bf16,query[rank].data(),query[rank].size()*2u,cudaMemcpyHostToDevice));
    }
    random_state=saved;
    for (uint32_t dest=0u; dest<degree; dest++)
        for (uint32_t src=0u; src<degree; src++)
            CUDA(cudaMemcpy(item->ranks[dest].slot.kv_shard_query_gathered_bf16+src*qstride,item->ranks[src].slot.query_latent_bf16,qstride*2u,cudaMemcpyDeviceToDevice));
    CUDA(cudaDeviceSynchronize());
    for (uint32_t rank=0u; rank<degree; rank++)
    {
        Rank &r=item->ranks[rank];
        if (oracle!=0u)
        {
            LmKvShardReplicaView view;
            REQUIRE(LmKvShardReplicaViewInitialize<Glm5NextKv>(&view,replica->ranks[0].buffers.cache,r.buffers.kv_shard)==0);
            uint32_t listed=item->max_context>GLM5_NEXT_DSA_SELECTED;
            CUDA((LmLatentShardPartialLaunch<Glm5NextKv,LmKvShardReplicaView,GLM5_NEXT_LATENT,GLM5_NEXT_ROPE_DIM>(view,r.buffers.shard_query_gathered_bf16,qstride,heads,
                item->sequence_of_row,item->context_length,item->positions,listed ? item->selected : nullptr,listed ? SPARK_GLM5_NEXT_MODEL_INDEX_OUTPUT_WIDTH : 0u,
                GLM5_NEXT_DSA_SELECTED,r.buffers.qk_scale,r.slot.kv_shard_partials_f32,pstride,rows,(cudaStream_t)r.slot.stream)));
        }
        else
            REQUIRE(Glm5NextLayerAttentionShardPartial(&r.buffers,rows,item->max_context,(cudaStream_t)r.slot.stream)==LM_LAUNCH_OK);
        CUDA(cudaStreamSynchronize((cudaStream_t)r.slot.stream));
    }
    for (uint32_t dest=0u; dest<degree; dest++)
        for (uint32_t src=0u; src<degree; src++)
            CUDA(cudaMemcpy(item->ranks[dest].slot.kv_shard_partials_received_f32+src*pstride,item->ranks[src].slot.kv_shard_partials_f32+dest*pstride,pstride*sizeof(float),cudaMemcpyDeviceToDevice));
    CUDA(cudaDeviceSynchronize());
    merged.clear();
    for (uint32_t rank=0u; rank<degree; rank++)
    {
        Rank &r=item->ranks[rank];
        REQUIRE(Glm5NextLayerAttentionShardMergeLatent(&r.buffers,rows,(cudaStream_t)r.slot.stream)==LM_LAUNCH_OK);
        CUDA(cudaStreamSynchronize((cudaStream_t)r.slot.stream));
        std::vector<uint16_t> out=Download(r.slot.attention_latent_bf16,(uint64_t)rows*heads*GLM5_NEXT_LATENT);
        merged.insert(merged.end(),out.begin(),out.end());
    }
}

static float Bf16Value(uint16_t bits)
{
    uint32_t word=(uint32_t)bits<<16u;
    float value;
    memcpy(&value,&word,sizeof(value));
    return value;
}

static void CheckProductionKernel(Case *item,Case *replica,const std::vector<uint16_t> &merged,double *worst_scaled,size_t *differing)
{
    uint32_t rows=item->rows,heads=item->heads,listed=item->max_context>GLM5_NEXT_DSA_SELECTED;
    uint64_t per_rank=(uint64_t)rows*heads*GLM5_NEXT_LATENT;
    const Glm5NextLayerBuffers &reference=replica->ranks[0].buffers;
    uint16_t *output;
    CUDA(cudaMalloc(&output,per_rank*2u));
    *worst_scaled=0.0;*differing=0u;
    for (uint32_t rank=0u; rank<item->degree; rank++)
    {
        const uint16_t *query=item->ranks[rank].slot.query_latent_bf16;
        cudaStream_t stream=(cudaStream_t)replica->ranks[0].slot.stream;
        if (LmLatentAttentionHeadsSupported(heads)!=0u)
            CUDA((LmLatentAttentionHeadsLaunch<Glm5NextKv,GLM5_NEXT_LATENT>(query,reference.cache,item->sequence_of_row,item->context_length,
                listed ? item->selected : nullptr,listed ? SPARK_GLM5_NEXT_MODEL_INDEX_OUTPUT_WIDTH : 0u,heads,reference.qk_scale,output,item->positions,rows,
                SparkGlm5NextAttentionPositionBound(item->max_context),0u,nullptr,0u,48u,stream)));
        else
            CUDA((LmLatentAttentionDecodeSplitLaunch<Glm5NextKv,GLM5_NEXT_ATTN_THREADS,GLM5_NEXT_LATENT,GLM5_NEXT_ROPE_DIM>(query,nullptr,reference.cache,item->sequence_of_row,item->context_length,
                listed ? item->selected : nullptr,listed ? SPARK_GLM5_NEXT_MODEL_INDEX_OUTPUT_WIDTH : 0u,heads,reference.qk_scale,output,item->positions,rows,
                SparkGlm5NextAttentionPositionBound(item->max_context),0u,nullptr,0u,48u,stream)));
        CUDA(cudaStreamSynchronize(stream));
        std::vector<uint16_t> want=Download(output,per_rank);
        for (uint64_t vector=0u; vector<per_rank; vector+=GLM5_NEXT_LATENT)
        {
            double scale=0.0;
            for (uint32_t e=0u; e<GLM5_NEXT_LATENT; e++) scale=std::max(scale,(double)fabsf(Bf16Value(want[vector+e])));
            for (uint32_t e=0u; e<GLM5_NEXT_LATENT; e++)
            {
                uint16_t got=merged[rank*per_rank+vector+e];
                double gap=fabs((double)Bf16Value(want[vector+e])-(double)Bf16Value(got));
                *worst_scaled=std::max(*worst_scaled,scale>0.0 ? gap/scale : gap);
                *differing+=got!=want[vector+e] ? 1u : 0u;
            }
        }
    }
    CUDA(cudaFree(output));
    if (*worst_scaled>1.0/128.0 || *differing*100u>merged.size())
    {
        fprintf(stderr,"FAIL degree=%u rows=%u max_context=%u sharded output vs the replicated production kernel: worst gap %.3e of the head's largest value (bound one bf16 ulp, 7.8e-3), differing=%zu of %zu (bound 1%%)\n",item->degree,rows,item->max_context,*worst_scaled,*differing,merged.size());
        exit(1);
    }
}

static void ShardCase(uint32_t degree,uint32_t capacity,const std::vector<uint32_t> &contexts)
{
    Case shard,replica;
    shard.capacity=replica.capacity=capacity;
    shard.degree=replica.degree=degree;shard.rows=replica.rows=(uint32_t)contexts.size();shard.contexts=replica.contexts=contexts;
    uint32_t seed=random_state;
    Fill(&shard);random_state=seed;Fill(&replica);
    REQUIRE(SparkGlm5NextKvShardFits(degree,capacity)!=0u && shard.rows<=capacity);
    shard.ranks.resize(degree);replica.ranks.resize(1u);
    for (uint32_t rank=0u; rank<degree; rank++) RankBuild(&shard,rank,1u);
    RankBuild(&replica,0u,0u);
    StoreAllPositions(&shard);StoreAllPositions(&replica);
    CheckOwnedSlots(&shard,&replica);
    if (shard.max_context>GLM5_NEXT_DSA_SELECTED) CheckIndexScores(&shard,&replica);
    std::vector<uint16_t> sharded,oracle;
    Case oracle_case=shard;
    Attention(&shard,0u,&replica,sharded);
    Attention(&oracle_case,1u,&replica,oracle);
    std::vector<uint16_t> again;
    Attention(&shard,0u,&replica,again);
    REQUIRE(again==sharded);
    double worst_scaled;
    size_t differing;
    CheckProductionKernel(&shard,&replica,sharded,&worst_scaled,&differing);
    CheckNoAccessError(&shard);CheckNoAccessError(&replica);
    REQUIRE(sharded.size()==oracle.size());
    size_t mismatch=0u;
    for (size_t i=0u; i<sharded.size(); i++) if (sharded[i]!=oracle[i]) mismatch++;
    if (mismatch!=0u) { fprintf(stderr,"FAIL degree=%u rows=%u max_context=%u mismatched_bf16=%zu of %zu\n",degree,shard.rows,shard.max_context,mismatch,sharded.size());exit(1); }
    uint64_t kv_rank=(uint64_t)shard.pages*SparkKvShardPageBytes(SparkGlm5NextKvShardLatent(0u,degree),GLM5_NEXT_KV_PAGE_SLOTS,Glm5NextKv::kSlotBytes);
    uint64_t index_rank=(uint64_t)shard.pages*SparkKvShardPageBytes(SparkGlm5NextKvShardIndex(0u,degree),GLM5_NEXT_KV_PAGE_SLOTS,Glm5NextIndexKv::kSlotBytes);
    printf("PASS kv shard degree=%u rows=%u max_context=%u latent_bytes_per_rank=%llu of %llu index_bytes_per_rank=%llu of %llu capacity=%u exchange=%s query_sequences=%u partial_sequences=%u merged_bf16=%zu bitwise_equal_oracle=yes repeat_identical=yes index_scores_bitwise=%s production_kernel_worst_scaled=%.2e production_kernel_differing=%zu\n",
        degree,shard.rows,shard.max_context,(unsigned long long)kv_rank,(unsigned long long)((uint64_t)shard.pages*Glm5NextKv::kPageBytes),
        (unsigned long long)index_rank,(unsigned long long)((uint64_t)shard.pages*Glm5NextIndexKv::kPageBytes),
        capacity,SparkGlm5NextKvShardPartialWide(shard.rows,degree,capacity) ? "wide" : "narrow",
        SparkGlm5NextKvShardQuerySequences(shard.rows,degree),SparkGlm5NextKvShardPartialSequences(shard.rows,degree,capacity),sharded.size(),
        shard.max_context>GLM5_NEXT_DSA_SELECTED ? "yes" : "n/a",worst_scaled,differing);
    for (Rank &r : shard.ranks) RankFree(r);
    RankFree(replica.ranks[0]);
}

int main(int argc,char **argv)
{
    if (argc != 2 || strcmp(argv[1],"--run") != 0)
    {
        puts("usage: test_glm5_next_kv_shard --run");
        return 2;
    }
    alarm(900);
    setvbuf(stdout,nullptr,_IOLBF,0);
    CUDA(cudaSetDevice(0));
    REQUIRE(SparkGlm5NextKvShardFits(16u,128u)==1u && SparkGlm5NextKvShardFits(8u,128u)==1u && SparkGlm5NextKvShardFits(4u,1u)==0u && SparkGlm5NextKvShardFits(3u,1u)==0u);
    ShardCase(16u,8u,{1000u});
    ShardCase(16u,1u,{5000u});
    ShardCase(16u,8u,{100u,2049u,3000u,4096u,64u,1u,2048u,8191u});
    ShardCase(16u,64u,{100u,2049u,3000u,4096u,64u,1u,2048u,8191u});
    ShardCase(8u,4u,{1024u,3333u});
    std::vector<uint32_t> wide(64u);
    for (uint32_t i=0u; i<64u; i++) wide[i]=1024u-(i*13u)%700u;
    ShardCase(16u,64u,wide);
    puts("PASS glm5_next kv shard: the module's sharded store, index scoring, partials and merge equal the replicated-storage oracle bit for bit with 1/tp of the KV per rank, and stay within one bf16 ulp of the replicated production attention kernel");
    return 0;
}
