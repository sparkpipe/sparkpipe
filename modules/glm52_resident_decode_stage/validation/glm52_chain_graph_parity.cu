#define main SparkGlm52ValidatorMain
#include "spark_glm52_resident_decode_stage_cuda_validation.cu"
#undef main

#include "sparkpipe/spark_glm52_graph_regime.h"

#define PARITY_STEPS 96u
#define PARITY_LAYERS 2u
#define PARITY_FIRST_LAYER (SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER - 1u)
#define PARITY_SPLIT_THRESHOLD 64u
#define PARITY_BOUNDARY (2u * SPARK_GLM52_VHIDDEN)
#define PARITY_MODE_STAGED 0u
#define PARITY_MODE_LINEAR 1u
#define PARITY_MODE_GRAPH 2u
#define PARITY_MODE_GRAPH_WRONG_REGIME 3u
#define PARITY_MODE_PROJECTION_SPLIT 4u
#define PARITY_MODE_SPLIT_DROP_RANK 5u
#define PARITY_MODE_COUNT 6u

typedef struct ParityRig
{
	SparkGlm52ValFixture fixture;
	SparkGlm52LayerWeights layers[PARITY_LAYERS];
	uint32_t ordinals[PARITY_LAYERS];
	uint32_t *host_token;
	uint32_t *host_slot;
	uint32_t *host_position;
	uint16_t *boundary;
	uint16_t *gather;
	uint16_t *local;
	uint16_t *residual_saved;
	uint16_t *outputs[PARITY_MODE_COUNT];
	uint32_t tp_degree;
	uint32_t captures[PARITY_MODE_COUNT];
	uint32_t replays[PARITY_MODE_COUNT];
} ParityRig;

static const char *const PARITY_MODE_NAMES[PARITY_MODE_COUNT] = {"staged","linear","graph","graph-wrong-regime","projection-split","split-drop-last-rank"};

static void ParityBuild(ParityRig *rig,uint32_t step)
{
	static const uint32_t tokens[5] = {1u,3u,2u,6u,5u};
	SparkGlm52CudaWave *wave = &rig->fixture.wave;
	uint32_t layer;
	SparkGlm52ValBuildWave(&rig->fixture,PARITY_FIRST_LAYER,tokens[step % 5u],step);
	for (layer=0u; layer<PARITY_LAYERS; layer++)
	{
		rig->layers[layer] = rig->fixture.weights;
		rig->ordinals[layer] = 0u;
	}
	*rig->host_token = tokens[step % 5u];
	*rig->host_slot = 0u;
	*rig->host_position = step;
	wave->layers = rig->layers;
	wave->layer_count = PARITY_LAYERS;
	wave->index_ordinal_by_local_layer = rig->ordinals;
	wave->host_token_ids = rig->host_token;
	wave->host_resident_slots = rig->host_slot;
	wave->host_positions = rig->host_position;
	wave->tp_degree = rig->tp_degree;
	wave->tp_rank = 0u;
	wave->owns_final_head = 0u;
	wave->hidden_output_bf16 = rig->boundary;
	wave->boundary_row_offset = 0u;
	wave->decode_split_context_threshold = PARITY_SPLIT_THRESHOLD;
	wave->max_sequence_positions = SPARK_GLM52_VALIDATION_DSA_CONTEXT;
}

static int ParityWalk(ParityRig *rig,uint32_t staged)
{
	SparkGlm52CudaWave *wave = &rig->fixture.wave;
	cudaStream_t stream = rig->fixture.stream;
	uint32_t layer;
	if ( SparkGlm52LaunchCudaWaveBegin(wave) != 0 || (staged != 0u && cudaStreamSynchronize(stream) != cudaSuccess) )
		return(1);
	for (layer=0u; layer<PARITY_LAYERS; layer++)
	{
		if ( SparkGlm52LaunchCudaLayerAttention(wave,layer) != 0 || (staged != 0u && cudaStreamSynchronize(stream) != cudaSuccess) )
			return(2);
		if ( SparkGlm52LaunchCudaLayerMlp(wave,layer) != 0 || (staged != 0u && cudaStreamSynchronize(stream) != cudaSuccess) )
			return(3);
	}
	if ( SparkGlm52LaunchCudaWaveHead(wave) != 0 )
		return(4);
	return(0);
}

static int ParitySplitWalk(ParityRig *rig,uint32_t ranks)
{
	SparkGlm52CudaWave *wave = &rig->fixture.wave;
	cudaStream_t stream = rig->fixture.stream;
	uint64_t gather_bytes = (uint64_t)SPARK_GLM52_VHIDDEN * sizeof(uint16_t);
	uint32_t layer,rank,width = SparkGlm52ProjectionSliceWidth(wave->tp_degree);
	wave->projection_split = 1u;
	rig->fixture.slot.projection_gather_bf16 = rig->gather;
	rig->fixture.slot.projection_local_bf16 = rig->local;
	if ( SparkGlm52LaunchCudaWaveBegin(wave) != 0 || cudaStreamSynchronize(stream) != cudaSuccess )
		return(1);
	for (layer=0u; layer<PARITY_LAYERS; layer++)
	{
		if ( cudaMemcpyAsync(rig->residual_saved,rig->fixture.residual,gather_bytes,cudaMemcpyDeviceToDevice,stream) != cudaSuccess ||
			cudaMemsetAsync(rig->gather,0,gather_bytes,stream) != cudaSuccess )
			return(5);
		for (rank=0u; rank<ranks; rank++)
		{
			wave->tp_rank = rank;
			if ( cudaMemcpyAsync(rig->fixture.residual,rig->residual_saved,gather_bytes,cudaMemcpyDeviceToDevice,stream) != cudaSuccess ||
				SparkGlm52LaunchCudaLayerAttentionProject(wave,layer) != 0 ||
				cudaMemcpyAsync(rig->gather + (uint64_t)rank * width,rig->local,(uint64_t)width * sizeof(uint16_t),cudaMemcpyDeviceToDevice,stream) != cudaSuccess )
				return(6);
		}
		wave->tp_rank = 0u;
		if ( cudaStreamSynchronize(stream) != cudaSuccess )
			return(7);
		if ( SparkGlm52LaunchCudaLayerAttentionCore(wave,layer) != 0 || cudaStreamSynchronize(stream) != cudaSuccess )
			return(8);
		if ( SparkGlm52LaunchCudaLayerMlp(wave,layer) != 0 || cudaStreamSynchronize(stream) != cudaSuccess )
			return(3);
	}
	if ( SparkGlm52LaunchCudaWaveHead(wave) != 0 )
		return(4);
	return(0);
}

static int ParityCapture(ParityRig *rig,uint32_t bound,cudaGraphExec_t *exec)
{
	cudaStream_t stream = rig->fixture.stream;
	cudaGraph_t graph = 0;
	uint32_t context = rig->fixture.wave.maximum_context;
	int site;
	rig->fixture.wave.maximum_context = bound;
	if ( cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal) != cudaSuccess )
		return(10);
	site = ParityWalk(rig,0u);
	rig->fixture.wave.maximum_context = context;
	if ( cudaStreamEndCapture(stream,&graph) != cudaSuccess || site != 0 || graph == 0 )
		return(11 + site);
	if ( cudaGraphInstantiate(exec,graph,0) != cudaSuccess )
		return(20);
	(void)cudaGraphDestroy(graph);
	return(0);
}

static int ParityRun(ParityRig *rig,uint32_t mode)
{
	cudaGraphExec_t graphs[SPARK_GLM52_GRAPH_REGIME_SELECTED] = {0,0};
	uint32_t step,regime,bound;
	int status = 0;
	if ( SparkGlm52ValResetStreams(&rig->fixture) != 0 || cudaMemset(rig->boundary,0,PARITY_BOUNDARY * sizeof(uint16_t)) != cudaSuccess )
		return(30);
	for (step=0u; status==0 && step<PARITY_STEPS; step++)
	{
		ParityBuild(rig,step);
		if ( mode == PARITY_MODE_STAGED || mode == PARITY_MODE_LINEAR )
			status = ParityWalk(rig,mode == PARITY_MODE_STAGED ? 1u : 0u);
		else if ( mode == PARITY_MODE_PROJECTION_SPLIT || mode == PARITY_MODE_SPLIT_DROP_RANK )
			status = ParitySplitWalk(rig,mode == PARITY_MODE_SPLIT_DROP_RANK ? rig->tp_degree - 1u : rig->tp_degree);
		else
		{
			regime = SparkGlm52GraphRegime(rig->fixture.wave.maximum_context,PARITY_SPLIT_THRESHOLD,rig->fixture.wave.max_sequence_positions,&bound);
			if ( mode == PARITY_MODE_GRAPH_WRONG_REGIME )
				regime = SparkGlm52GraphRegime(regime == SPARK_GLM52_GRAPH_REGIME_SPLIT ? 1u : PARITY_SPLIT_THRESHOLD,PARITY_SPLIT_THRESHOLD,rig->fixture.wave.max_sequence_positions,&bound);
			else if ( bound < rig->fixture.wave.maximum_context || SparkGlm52GraphReplayable(rig->fixture.wave.maximum_context,rig->fixture.wave.max_sequence_positions) == 0u )
				status = 40;
			if ( status == 0 && graphs[regime] == 0 )
			{
				status = ParityCapture(rig,bound,&graphs[regime]);
				rig->captures[mode]++;
			}
			if ( status == 0 && cudaGraphLaunch(graphs[regime],rig->fixture.stream) != cudaSuccess )
				status = 41;
			rig->replays[mode]++;
		}
		if ( status == 0 && cudaStreamSynchronize(rig->fixture.stream) != cudaSuccess )
			status = 42;
		if ( status == 0 && cudaMemcpy(rig->outputs[mode] + (uint64_t)step * PARITY_BOUNDARY,rig->boundary,PARITY_BOUNDARY * sizeof(uint16_t),cudaMemcpyDeviceToHost) != cudaSuccess )
			status = 43;
		if ( status == 0 && SparkGlm52ValCheckAccessError(&rig->fixture) != 0 )
			status = 44;
	}
	for (regime=0u; regime<SPARK_GLM52_GRAPH_REGIME_SELECTED; regime++)
		if ( graphs[regime] != 0 )
			(void)cudaGraphExecDestroy(graphs[regime]);
	if ( status != 0 )
		fprintf(stderr,"glm52_chain_graph_parity tp=%u mode=%s step=%u status=%d cuda=%s\n",rig->tp_degree,PARITY_MODE_NAMES[mode],step,status,cudaGetErrorString(cudaGetLastError()));
	return(status);
}

static uint32_t ParityDifferingSteps(const ParityRig *rig,uint32_t mode,uint32_t *first)
{
	uint32_t step,count = 0u;
	*first = UINT32_MAX;
	for (step=0u; step<PARITY_STEPS; step++)
		if ( memcmp(rig->outputs[mode] + (uint64_t)step * PARITY_BOUNDARY,rig->outputs[PARITY_MODE_STAGED] + (uint64_t)step * PARITY_BOUNDARY,PARITY_BOUNDARY * sizeof(uint16_t)) != 0 )
		{
			if ( *first == UINT32_MAX )
				*first = step;
			count++;
		}
	return(count);
}

static uint32_t ParityNonzero(const ParityRig *rig)
{
	uint64_t index;
	for (index=0u; index<(uint64_t)PARITY_STEPS * PARITY_BOUNDARY; index++)
		if ( (rig->outputs[PARITY_MODE_STAGED][index] & 0x7fffu) != 0u )
			return(1u);
	return(0u);
}

static int ParityRunDegree(ParityRig *rig,uint32_t tp_degree)
{
	uint32_t mode,first,differ,wrong_regime;
	int failures = 0;
	rig->tp_degree = tp_degree;
	memset(rig->captures,0,sizeof(rig->captures));
	memset(rig->replays,0,sizeof(rig->replays));
	for (mode=0u; mode<PARITY_MODE_COUNT; mode++)
		if ( ParityRun(rig,mode) != 0 )
			return(1);
	if ( ParityNonzero(rig) == 0u )
	{
		fprintf(stderr,"glm52_chain_graph_parity tp=%u FAIL staged outputs are all zero\n",tp_degree);
		return(1);
	}
	for (mode=PARITY_MODE_LINEAR; mode<PARITY_MODE_COUNT; mode++)
	{
		if ( mode == PARITY_MODE_GRAPH_WRONG_REGIME || mode == PARITY_MODE_SPLIT_DROP_RANK )
			continue;
		differ = ParityDifferingSteps(rig,mode,&first);
		printf("glm52_chain_graph_parity tp=%u mode=%s steps=%u differing_steps=%u first=%d captures=%u replays=%u %s\n",
			tp_degree,PARITY_MODE_NAMES[mode],PARITY_STEPS,differ,first == UINT32_MAX ? -1 : (int)first,rig->captures[mode],rig->replays[mode],differ == 0u ? "BIT-EXACT" : "FAIL");
		failures += differ != 0u ? 1 : 0;
	}
	if ( rig->captures[PARITY_MODE_GRAPH] != SPARK_GLM52_GRAPH_REGIME_SELECTED )
	{
		fprintf(stderr,"glm52_chain_graph_parity tp=%u FAIL expected one capture per regime, saw %u\n",tp_degree,rig->captures[PARITY_MODE_GRAPH]);
		failures++;
	}
	wrong_regime = ParityDifferingSteps(rig,PARITY_MODE_GRAPH_WRONG_REGIME,&first);
	printf("glm52_chain_graph_parity tp=%u control=graph-wrong-regime differing_steps=%u first=%d (every step replayed with the other attention regime's bound) %s\n",
		tp_degree,wrong_regime,first == UINT32_MAX ? -1 : (int)first,wrong_regime != 0u ? "SEPARATED" : "FAIL");
	failures += wrong_regime == 0u ? 1 : 0;
	if ( tp_degree > 1u )
	{
		wrong_regime = ParityDifferingSteps(rig,PARITY_MODE_SPLIT_DROP_RANK,&first);
		printf("glm52_chain_graph_parity tp=%u control=split-drop-last-rank differing_steps=%u first=%d %s\n",
			tp_degree,wrong_regime,first == UINT32_MAX ? -1 : (int)first,wrong_regime != 0u ? "SEPARATED" : "FAIL");
		failures += wrong_regime == 0u ? 1 : 0;
	}
	return(failures);
}

int main(void)
{
	ParityRig *rig;
	uint32_t mode;
	int failures = 0;
	rig = (ParityRig *)calloc(1u,sizeof(*rig));
	if ( rig == 0 )
		return(1);
	if ( SparkGlm52ValFixtureSetup(&rig->fixture) != 0 )
		return(1);
	rig->fixture.split_partials = (float *)SparkGlm52ValAllocZeroed(SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTN_SPLIT_PARTIAL_BYTES(1u,SPARK_GLM52_MODEL_HEAD_COUNT));
	if ( rig->fixture.split_partials == 0 )
		return(1);
	if ( cudaHostAlloc((void **)&rig->host_token,3u * sizeof(uint32_t),cudaHostAllocPortable) != cudaSuccess ||
		cudaMalloc((void **)&rig->boundary,PARITY_BOUNDARY * sizeof(uint16_t)) != cudaSuccess ||
		cudaMalloc((void **)&rig->gather,SPARK_GLM52_VHIDDEN * sizeof(uint16_t)) != cudaSuccess ||
		cudaMalloc((void **)&rig->local,SPARK_GLM52_VHIDDEN * sizeof(uint16_t)) != cudaSuccess ||
		cudaMalloc((void **)&rig->residual_saved,SPARK_GLM52_VHIDDEN * sizeof(uint16_t)) != cudaSuccess )
		return(1);
	rig->host_slot = rig->host_token + 1u;
	rig->host_position = rig->host_token + 2u;
	for (mode=0u; mode<PARITY_MODE_COUNT; mode++)
	{
		rig->outputs[mode] = (uint16_t *)calloc((uint64_t)PARITY_STEPS * PARITY_BOUNDARY,sizeof(uint16_t));
		if ( rig->outputs[mode] == 0 )
			return(1);
	}
	failures += ParityRunDegree(rig,1u);
	failures += ParityRunDegree(rig,16u);
	SparkGlm52ValFixtureDestroy(&rig->fixture);
	printf("glm52_chain_graph_parity %s\n",failures == 0 ? "PASS" : "FAIL");
	return(failures == 0 ? 0 : 1);
}
