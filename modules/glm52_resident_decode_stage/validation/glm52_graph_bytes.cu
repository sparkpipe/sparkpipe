#define main SparkGlm52ValidatorMain
#include "spark_glm52_resident_decode_stage_cuda_validation.cu"
#undef main

#include <unistd.h>

#include "sparkpipe/spark_glm52_graph_regime.h"

#define GRAPH_BYTES_EXECS 8u
#define GRAPH_BYTES_LAYERS SPARK_GLM52_MODEL_LAYER_COUNT
#define GRAPH_BYTES_SPLIT_THRESHOLD 64u
#define GRAPH_BYTES_POSITION 200u

typedef struct GraphBytesRig
{
	SparkGlm52ValFixture fixture;
	SparkGlm52LayerWeights layers[GRAPH_BYTES_LAYERS];
	uint32_t ordinals[GRAPH_BYTES_LAYERS];
	uint32_t *host_words;
	uint16_t *boundary;
	uint16_t *gather;
} GraphBytesRig;

static uint64_t GraphBytesRss(void)
{
	FILE *file = fopen("/proc/self/statm","r");
	unsigned long pages = 0ul,resident = 0ul;
	if ( file == 0 )
		return(0u);
	if ( fscanf(file,"%lu %lu",&pages,&resident) != 2 )
		resident = 0ul;
	fclose(file);
	return((uint64_t)resident * (uint64_t)sysconf(_SC_PAGESIZE));
}

static void GraphBytesBuild(GraphBytesRig *rig,uint32_t split)
{
	SparkGlm52CudaWave *wave = &rig->fixture.wave;
	uint32_t layer;
	SparkGlm52ValBuildWave(&rig->fixture,0u,1u,GRAPH_BYTES_POSITION);
	for (layer=0u; layer<GRAPH_BYTES_LAYERS; layer++)
	{
		rig->layers[layer] = rig->fixture.weights;
		rig->ordinals[layer] = 0u;
	}
	rig->host_words[0] = 1u;
	rig->host_words[1] = 0u;
	rig->host_words[2] = GRAPH_BYTES_POSITION;
	wave->layers = rig->layers;
	wave->layer_count = GRAPH_BYTES_LAYERS;
	wave->index_ordinal_by_local_layer = rig->ordinals;
	wave->host_token_ids = rig->host_words;
	wave->host_resident_slots = rig->host_words + 1;
	wave->host_positions = rig->host_words + 2;
	wave->tp_degree = 16u;
	wave->tp_rank = 0u;
	wave->owns_final_head = 0u;
	wave->hidden_output_bf16 = rig->boundary;
	wave->boundary_row_offset = 0u;
	wave->decode_split_context_threshold = GRAPH_BYTES_SPLIT_THRESHOLD;
	wave->max_sequence_positions = SPARK_GLM52_VALIDATION_DSA_CONTEXT;
	wave->projection_split = split;
	rig->fixture.slot.projection_gather_bf16 = split != 0u ? rig->gather : 0;
}

static int GraphBytesWalk(GraphBytesRig *rig,uint32_t split)
{
	SparkGlm52CudaWave *wave = &rig->fixture.wave;
	uint32_t layer;
	if ( SparkGlm52LaunchCudaWaveBegin(wave) != 0 )
		return(1);
	for (layer=0u; layer<GRAPH_BYTES_LAYERS; layer++)
	{
		if ( split != 0u ? (SparkGlm52LaunchCudaLayerAttentionProject(wave,layer) != 0 || SparkGlm52LaunchCudaLayerAttentionCore(wave,layer) != 0) :
			SparkGlm52LaunchCudaLayerAttention(wave,layer) != 0 )
			return(2);
		if ( SparkGlm52LaunchCudaLayerMlp(wave,layer) != 0 )
			return(3);
	}
	return(SparkGlm52LaunchCudaWaveHead(wave) != 0 ? 4 : 0);
}

static int GraphBytesCapture(GraphBytesRig *rig,uint32_t split,cudaGraphExec_t *exec,size_t *nodes)
{
	cudaStream_t stream = rig->fixture.stream;
	cudaGraph_t graph = 0;
	uint32_t bound;
	GraphBytesBuild(rig,split);
	(void)SparkGlm52GraphRegime(GRAPH_BYTES_POSITION + 1u,GRAPH_BYTES_SPLIT_THRESHOLD,rig->fixture.wave.max_sequence_positions,&bound);
	rig->fixture.wave.maximum_context = bound;
	if ( cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal) != cudaSuccess )
		return(1);
	if ( GraphBytesWalk(rig,split) != 0 )
	{
		(void)cudaStreamEndCapture(stream,&graph);
		return(2);
	}
	if ( cudaStreamEndCapture(stream,&graph) != cudaSuccess || graph == 0 )
		return(3);
	if ( cudaGraphGetNodes(graph,0,nodes) != cudaSuccess )
		return(4);
	if ( cudaGraphInstantiate(exec,graph,0) != cudaSuccess || cudaGraphUpload(*exec,stream) != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess )
		return(5);
	(void)cudaGraphDestroy(graph);
	return(0);
}

int main(int argc,char **argv)
{
	GraphBytesRig *rig;
	cudaGraphExec_t execs[GRAPH_BYTES_EXECS];
	size_t nodes = 0u,free_before,free_after,total;
	uint64_t rss_before,rss_after;
	uint32_t split,index,count;
	int status;
	split = argc > 1 ? (uint32_t)atoi(argv[1]) : 1u;
	count = argc > 2 ? (uint32_t)atoi(argv[2]) : GRAPH_BYTES_EXECS;
	if ( count == 0u || count > GRAPH_BYTES_EXECS )
		return(1);
	rig = (GraphBytesRig *)calloc(1u,sizeof(*rig));
	if ( rig == 0 || SparkGlm52ValFixtureSetup(&rig->fixture) != 0 )
		return(1);
	rig->fixture.split_threshold = GRAPH_BYTES_SPLIT_THRESHOLD;
	rig->fixture.split_partials = (float *)SparkGlm52ValAllocZeroed(SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_BYTES(1u,SPARK_GLM52_MODEL_HEAD_COUNT));
	if ( rig->fixture.split_partials == 0 ||
		cudaHostAlloc((void **)&rig->host_words,4u * sizeof(uint32_t),cudaHostAllocPortable) != cudaSuccess ||
		cudaMalloc((void **)&rig->boundary,2u * SPARK_GLM52_VHIDDEN * sizeof(uint16_t)) != cudaSuccess ||
		cudaMalloc((void **)&rig->gather,SPARK_GLM52_VHIDDEN * sizeof(uint16_t)) != cudaSuccess ||
		cudaMemset(rig->gather,0,SPARK_GLM52_VHIDDEN * sizeof(uint16_t)) != cudaSuccess ||
		SparkGlm52ValResetStreams(&rig->fixture) != 0 )
		return(1);
	for (index=0u; index<count; index++)
	{
		if ( cudaDeviceSynchronize() != cudaSuccess || cudaMemGetInfo(&free_before,&total) != cudaSuccess )
			return(1);
		rss_before = GraphBytesRss();
		status = GraphBytesCapture(rig,split,&execs[index],&nodes);
		if ( status != 0 )
		{
			fprintf(stderr,"glm52_graph_bytes FAIL exec=%u status=%d cuda=%s\n",index,status,cudaGetErrorString(cudaGetLastError()));
			return(1);
		}
		if ( cudaMemGetInfo(&free_after,&total) != cudaSuccess )
			return(1);
		rss_after = GraphBytesRss();
		printf("glm52_graph_bytes exec=%u projections=%s nodes=%zu device_free_delta=%lld rss_delta=%lld\n",index,split != 0u ? "split" : "replicated",nodes,
			(long long)free_before - (long long)free_after,(long long)rss_after - (long long)rss_before);
	}
	for (index=0u; index<count; index++)
		(void)cudaGraphExecDestroy(execs[index]);
	SparkGlm52ValFixtureDestroy(&rig->fixture);
	printf("glm52_graph_bytes PASS\n");
	return(0);
}
