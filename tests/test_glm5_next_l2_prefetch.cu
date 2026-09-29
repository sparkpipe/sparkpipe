#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_cuda.cu"

#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { \
    fprintf(stderr,"FAIL line=%d cuda=%s call=%s\n",__LINE__,cudaGetErrorString(e),#call); \
    exit(1); } } while(0)
#define REQUIRE(test) do { if(!(test)) { \
    fprintf(stderr,"FAIL line=%d test=%s\n",__LINE__,#test); exit(1); } } while(0)

static constexpr uint32_t test_layers = 5u, routed_dsa_layer = 3u;
static constexpr uint64_t weight_bytes = 16ull << 20;

typedef struct PrefetchFixture
{
    SparkGlm5NextLayerWeights layers[test_layers];
    SparkGlm5NextExecutionSlot slot;
    uint32_t no_ordinal[test_layers];
    SparkGlm5NextCudaWave wave;
    uint8_t *weights;
    uint32_t *marks;
    uint64_t *flag_host;
    CUdeviceptr flag_device;
    cudaStream_t stream;
}
PrefetchFixture;

typedef struct RoundNodes
{
    cudaGraphNode_t publish,request,wait,guard,combine;
}
RoundNodes;

static __global__ void TestMarkKernel(uint32_t *marks,uint32_t index)
{
    marks[index] += 1u;
}

static void BindWeights(SparkGlm5NextLayerWeights *weight,uint8_t *base)
{
    weight->hc_attn_fn_f32 = base;
    weight->hc_ffn_fn_f32 = base + 16u;
    weight->attn_norm_bf16 = base + 32u;
    weight->post_attn_norm_bf16 = base + 48u;
    weight->q_a_bf16 = base + 64u;
    weight->index_k_bf16 = base + 80u;
    weight->index_compress_gate_bf16 = base + 96u;
    weight->index_head_bf16 = base + 112u;
    weight->router_bf16 = base + 128u;
    weight->shared_gate_up_bf16 = base + 144u;
    weight->kda_qkv_beta_bf16 = base + 160u;
    weight->kda_decay_gate_down_bf16 = base + 176u;
    weight->dense_gate_up_bf16 = base + 192u;
}

static void FixtureCreate(PrefetchFixture *fixture)
{
    uint32_t layer;
    memset(fixture,0,sizeof(*fixture));
    CUDA(cudaMalloc((void **)&fixture->weights,weight_bytes + 4096u));
    CUDA(cudaMemset(fixture->weights,0,weight_bytes + 4096u));
    CUDA(cudaMalloc((void **)&fixture->marks,64u * sizeof(uint32_t)));
    CUDA(cudaMemset(fixture->marks,0,64u * sizeof(uint32_t)));
    CUDA(cudaHostAlloc((void **)&fixture->flag_host,sizeof(uint64_t),cudaHostAllocMapped));
    *fixture->flag_host = 1ull;
    CUDA(cudaHostGetDevicePointer((void **)&fixture->flag_device,fixture->flag_host,0u));
    CUDA(cudaStreamCreateWithFlags(&fixture->stream,cudaStreamNonBlocking));
    for (layer=0u; layer<test_layers; layer++)
    {
        BindWeights(&fixture->layers[layer],fixture->weights);
        fixture->no_ordinal[layer] = UINT32_MAX;
    }
    fixture->slot.stream = fixture->stream;
    fixture->wave.first_layer_index = 0u;
    fixture->wave.layer_count = test_layers;
    fixture->wave.tp_degree = 16u;
    fixture->wave.row_count = 1u;
    fixture->wave.layers = fixture->layers;
    fixture->wave.slot = &fixture->slot;
    fixture->wave.kda_ordinal_by_local_layer = fixture->no_ordinal;
    fixture->wave.index_ordinal_by_local_layer = fixture->no_ordinal;
}

static cudaGraphNode_t CaptureTail(cudaStream_t stream)
{
    cudaStreamCaptureStatus capture;
    cudaGraph_t graph;
    const cudaGraphNode_t *dependencies;
    size_t count;
    CUDA(cudaStreamGetCaptureInfo(stream,&capture,0,&graph,&dependencies,0,&count));
    REQUIRE(capture == cudaStreamCaptureStatusActive && count == 1u);
    return(dependencies[0]);
}

static cudaGraphNode_t Mark(PrefetchFixture *fixture,uint32_t index)
{
    TestMarkKernel<<<1,1,0,fixture->stream>>>(fixture->marks,index);
    CUDA(cudaPeekAtLastError());
    return(CaptureTail(fixture->stream));
}

static cudaGraphNode_t WaitValue(PrefetchFixture *fixture)
{
    CUstreamCaptureStatus capture;
    CUstreamBatchMemOpParams operation;
    CUDA_BATCH_MEM_OP_NODE_PARAMS params;
    const CUgraphNode *dependencies;
    size_t count;
    cuuint64_t id;
    CUgraph graph;
    CUgraphNode wait;
    memset(&operation,0,sizeof(operation));
    memset(&params,0,sizeof(params));
    operation.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
    operation.waitValue.address = fixture->flag_device;
    operation.waitValue.value64 = 1u;
    operation.waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
    REQUIRE(cuStreamGetCaptureInfo((CUstream)fixture->stream,&capture,&id,&graph,&dependencies,0,&count) == CUDA_SUCCESS);
    REQUIRE(cuCtxGetCurrent(&params.ctx) == CUDA_SUCCESS);
    params.count = 1u;
    params.paramArray = &operation;
    REQUIRE(cuGraphAddBatchMemOpNode(&wait,graph,dependencies,count,&params) == CUDA_SUCCESS);
    REQUIRE(cuStreamUpdateCaptureDependencies((CUstream)fixture->stream,&wait,0,1u,CU_STREAM_SET_CAPTURE_DEPENDENCIES) == CUDA_SUCCESS);
    return((cudaGraphNode_t)wait);
}

static RoundNodes HardwareRound(PrefetchFixture *fixture,uint32_t first_mark)
{
    RoundNodes round;
    round.publish = Mark(fixture,first_mark);
    round.request = Mark(fixture,first_mark + 1u);
    round.wait = WaitValue(fixture);
    round.guard = Mark(fixture,first_mark + 2u);
    round.combine = Mark(fixture,first_mark + 3u);
    return(round);
}

static size_t Dependencies(cudaGraphNode_t node,CUgraphNode *out,size_t capacity)
{
    size_t count = 0u;
    REQUIRE(cuGraphNodeGetDependencies((CUgraphNode)node,0,0,&count) == CUDA_SUCCESS);
    REQUIRE(count <= capacity);
    if ( count != 0u )
        REQUIRE(cuGraphNodeGetDependencies((CUgraphNode)node,out,0,&count) == CUDA_SUCCESS);
    return(count);
}

static uint32_t PrefetchNodes(cudaGraph_t graph,cudaGraphNode_t *found)
{
    cudaGraphNode_t nodes[64];
    CUgraphNodeType type;
    cudaKernelNodeParams params;
    size_t count = 64u,index;
    uint32_t prefetch = 0u;
    CUDA(cudaGraphGetNodes(graph,nodes,&count));
    REQUIRE(count < 64u);
    for (index=0u; index<count; index++)
    {
        REQUIRE(cuGraphNodeGetType((CUgraphNode)nodes[index],&type) == CUDA_SUCCESS);
        if ( type != CU_GRAPH_NODE_TYPE_KERNEL )
            continue;
        CUDA(cudaGraphKernelNodeGetParams(nodes[index],&params));
        if ( params.func != (void *)Glm5NextL2PrefetchKernel )
            continue;
        REQUIRE(params.gridDim.x == GLM5_NEXT_L2_PREFETCH_BLOCKS && params.blockDim.x == GLM5_NEXT_L2_PREFETCH_THREADS);
        *found = nodes[index];
        prefetch++;
    }
    return(prefetch);
}

static void TestPlans(PrefetchFixture *fixture)
{
    Glm5NextL2PrefetchPlan plan;
    const uint64_t row = (uint64_t)GLM5_NEXT_HIDDEN * sizeof(uint16_t);
    const uint64_t fn_bytes = (uint64_t)GLM5_NEXT_HC_MIX * GLM5_NEXT_HC_FLAT * sizeof(float);
    uint32_t range;
    uint64_t total = 0u;
    REQUIRE(Glm5NextL2PrefetchPlanFor(&fixture->wave,routed_dsa_layer,SPARK_GLM5_NEXT_L2_SITE_ATTENTION_REDUCE,&plan) == LM_LAUNCH_OK);
    REQUIRE(plan.count == 4u && plan.base[0] == fixture->weights + 16u && plan.bytes[0] == fn_bytes);
    REQUIRE(plan.base[1] == fixture->weights + 48u && plan.bytes[1] == row);
    REQUIRE(plan.base[2] == fixture->weights + 128u && plan.bytes[2] == (uint64_t)GLM5_NEXT_EXPERTS * row);
    REQUIRE(plan.base[3] == fixture->weights + 144u);
    REQUIRE(Glm5NextL2PrefetchPlanFor(&fixture->wave,routed_dsa_layer - 1u,SPARK_GLM5_NEXT_L2_SITE_MLP_REDUCE,&plan) == LM_LAUNCH_OK);
    REQUIRE(plan.count >= 3u && plan.base[0] == fixture->weights && plan.bytes[0] == fn_bytes && plan.base[1] == fixture->weights + 32u && plan.base[2] == fixture->weights + 64u);
    for (range=0u; range<plan.count; range++)
    {
        REQUIRE(plan.bytes[range] % 16u == 0u);
        total += plan.bytes[range];
    }
    REQUIRE(total == plan.total && plan.total == GLM5_NEXT_L2_PREFETCH_BYTES);
    REQUIRE(Glm5NextL2PrefetchPlanFor(&fixture->wave,0u,SPARK_GLM5_NEXT_L2_SITE_BEGIN,&plan) == LM_LAUNCH_OK);
    REQUIRE(plan.count >= 3u && plan.base[0] == fixture->weights && plan.base[2] == fixture->weights + 160u);
    REQUIRE(Glm5NextL2PrefetchPlanFor(&fixture->wave,test_layers - 1u,SPARK_GLM5_NEXT_L2_SITE_MLP_REDUCE,&plan) == LM_LAUNCH_OK && plan.count == 0u);
    REQUIRE(Glm5NextL2PrefetchPlanFor(&fixture->wave,test_layers,SPARK_GLM5_NEXT_L2_SITE_ATTENTION_REDUCE,&plan) == LM_LAUNCH_ERR_SHAPE);
    REQUIRE(Glm5NextL2PrefetchPlanFor(&fixture->wave,0u,SPARK_GLM5_NEXT_L2_SITE_BEGIN + 1u,&plan) == LM_LAUNCH_ERR_SHAPE);
    fixture->layers[routed_dsa_layer].router_bf16 = fixture->weights + 8u;
    REQUIRE(Glm5NextL2PrefetchPlanFor(&fixture->wave,routed_dsa_layer,SPARK_GLM5_NEXT_L2_SITE_ATTENTION_REDUCE,&plan) == LM_LAUNCH_ERR_SHAPE);
    fixture->layers[routed_dsa_layer].router_bf16 = fixture->weights + 128u;
}

static void TestPlacement(PrefetchFixture *fixture)
{
    RoundNodes round;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    cudaGraphNode_t next,prefetch = 0;
    CUgraphNode dependencies[4];
    uint32_t placed = 7u,marks[8];
    size_t count;
    CUDA(cudaStreamBeginCapture(fixture->stream,cudaStreamCaptureModeGlobal));
    (void)Mark(fixture,0u);
    round = HardwareRound(fixture,1u);
    REQUIRE(SparkGlm5NextL2PrefetchAfterRound(&fixture->wave,routed_dsa_layer,SPARK_GLM5_NEXT_L2_SITE_ATTENTION_REDUCE,&placed) == LM_LAUNCH_OK && placed == 1u);
    TestMarkKernel<<<1,1,0,fixture->stream>>>(fixture->marks,5u);
    CUDA(cudaPeekAtLastError());
    {
        cudaStreamCaptureStatus capture;
        cudaGraph_t capturing;
        const cudaGraphNode_t *tail;
        CUDA(cudaStreamGetCaptureInfo(fixture->stream,&capture,0,&capturing,&tail,0,&count));
        REQUIRE(count == 1u);
        next = tail[0];
    }
    CUDA(cudaStreamEndCapture(fixture->stream,&graph));
    REQUIRE(PrefetchNodes(graph,&prefetch) == 1u);
    count = Dependencies(prefetch,dependencies,4u);
    REQUIRE(count == 1u && dependencies[0] == (CUgraphNode)round.request);
    count = Dependencies(round.wait,dependencies,4u);
    REQUIRE(count == 1u && dependencies[0] == (CUgraphNode)round.request);
    count = Dependencies(round.request,dependencies,4u);
    REQUIRE(count == 1u && dependencies[0] == (CUgraphNode)round.publish);
    count = Dependencies(next,dependencies,4u);
    REQUIRE(count == 2u);
    REQUIRE((dependencies[0] == (CUgraphNode)round.combine && dependencies[1] == (CUgraphNode)prefetch) ||
            (dependencies[1] == (CUgraphNode)round.combine && dependencies[0] == (CUgraphNode)prefetch));
    CUDA(cudaGraphInstantiate(&exec,graph,0));
    CUDA(cudaGraphLaunch(exec,fixture->stream));
    CUDA(cudaStreamSynchronize(fixture->stream));
    CUDA(cudaMemcpy(marks,fixture->marks,sizeof(marks),cudaMemcpyDeviceToHost));
    REQUIRE(marks[0] == 1u && marks[1] == 1u && marks[2] == 1u && marks[3] == 1u && marks[4] == 1u && marks[5] == 1u);
    CUDA(cudaGraphExecDestroy(exec));
    CUDA(cudaGraphDestroy(graph));
    CUDA(cudaMemset(fixture->marks,0,64u * sizeof(uint32_t)));
}

static void TestNoPlacement(PrefetchFixture *fixture)
{
    cudaGraph_t graph;
    cudaGraphNode_t prefetch = 0;
    uint32_t placed = 7u,mark;
    REQUIRE(SparkGlm5NextL2PrefetchAfterRound(&fixture->wave,routed_dsa_layer,SPARK_GLM5_NEXT_L2_SITE_ATTENTION_REDUCE,&placed) == LM_LAUNCH_OK && placed == 0u);
    REQUIRE(SparkGlm5NextL2PrefetchAfterRound(&fixture->wave,routed_dsa_layer,SPARK_GLM5_NEXT_L2_SITE_ATTENTION_REDUCE,0) == LM_LAUNCH_ERR_SHAPE);
    CUDA(cudaStreamBeginCapture(fixture->stream,cudaStreamCaptureModeGlobal));
    for (mark=0u; mark<6u; mark++)
        (void)Mark(fixture,mark);
    placed = 7u;
    REQUIRE(SparkGlm5NextL2PrefetchAfterRound(&fixture->wave,routed_dsa_layer,SPARK_GLM5_NEXT_L2_SITE_ATTENTION_REDUCE,&placed) == LM_LAUNCH_OK && placed == 0u);
    (void)HardwareRound(fixture,6u);
    placed = 7u;
    REQUIRE(SparkGlm5NextL2PrefetchAfterRound(&fixture->wave,test_layers - 1u,SPARK_GLM5_NEXT_L2_SITE_MLP_REDUCE,&placed) == LM_LAUNCH_OK && placed == 0u);
    (void)WaitValue(fixture);
    for (mark=10u; mark<10u + GLM5_NEXT_L2_PREFETCH_SEARCH; mark++)
        (void)Mark(fixture,mark);
    placed = 7u;
    REQUIRE(SparkGlm5NextL2PrefetchAfterRound(&fixture->wave,routed_dsa_layer,SPARK_GLM5_NEXT_L2_SITE_ATTENTION_REDUCE,&placed) == LM_LAUNCH_OK && placed == 0u);
    CUDA(cudaStreamEndCapture(fixture->stream,&graph));
    REQUIRE(PrefetchNodes(graph,&prefetch) == 0u);
    CUDA(cudaGraphDestroy(graph));
}

int main(int argc,char **argv)
{
    PrefetchFixture fixture;
    if (argc != 2 || strcmp(argv[1],"--run") != 0)
    {
        puts("usage: test_glm5_next_l2_prefetch --run");
        return(2);
    }
    FixtureCreate(&fixture);
    TestPlans(&fixture);
    TestPlacement(&fixture);
    TestNoPlacement(&fixture);
    CUDA(cudaStreamDestroy(fixture.stream));
    CUDA(cudaFreeHost(fixture.flag_host));
    CUDA(cudaFree(fixture.marks));
    CUDA(cudaFree(fixture.weights));
    puts("PASS glm5_next l2 prefetch plans, placement after the pre-wait request node, join into the next node, no placement without a peer wait");
    return(0);
}
