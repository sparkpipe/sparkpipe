#undef NDEBUG
#define GLM5_NEXT_EXPERT_WEIGHT_CODEC 5
#define GLM5_NEXT_EXPERT_CODEC_NAME "fp8"
#define GLM5_NEXT_MODEL_REVISION "test"
#define GLM5_NEXT_CONTRACT_SHA256 "test"
#include <assert.h>
#include <errno.h>
#include "../modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_module.c"

static uint32_t STEP,FAIL_ACQUIRE,FAIL_LAUNCH,FAIL_RELEASE;
static uint8_t ADDRESS[256];
static uint32_t PAUSE_RECORDING,PAUSE_CALLS,PAUSE_FAIL_ERRNO,PAUSE_INTERRUPT_ONCE,CREATE_CALLS,CREATE_BUSY_CALLS;
static struct timespec PAUSE_REQUESTS[SPARK_GLM5_NEXT_LAZY_ATTACH_ATTEMPTS];
static SparkWeightdLazyPack ATTACHED_PACK;

int nanosleep(const struct timespec *request,struct timespec *remaining)
{
	if ( PAUSE_RECORDING == 0u )
		return(0);
	assert(request != 0 && PAUSE_CALLS < SPARK_GLM5_NEXT_LAZY_ATTACH_ATTEMPTS);
	PAUSE_REQUESTS[PAUSE_CALLS++] = *request;
	if ( request->tv_sec < 0 || request->tv_nsec < 0 || request->tv_nsec >= 1000000000l )
	{
		errno = EINVAL;
		return(-1);
	}
	if ( PAUSE_INTERRUPT_ONCE != 0u )
	{
		PAUSE_INTERRUPT_ONCE = 0u;
		assert(remaining != 0);
		remaining->tv_sec = 0;
		remaining->tv_nsec = 400000000l;
		errno = EINTR;
		return(-1);
	}
	if ( PAUSE_FAIL_ERRNO != 0u )
	{
		errno = (int)PAUSE_FAIL_ERRNO;
		return(-1);
	}
	return(0);
}

SparkStatus SparkWeightdAttachRequested(void)
{
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdLazyPackCreateChecked(const char *socket,const SparkWeightdLazyAttachRequest *request,uint64_t spine_budget,uint64_t timeout,SparkWeightdManifestCheck check,void *context,SparkWeightdLazyPack **out)
{
	(void)socket; (void)request; (void)spine_budget; (void)timeout; (void)check; (void)context;
	assert(out != 0 && *out == 0);
	CREATE_CALLS++;
	if ( CREATE_CALLS <= CREATE_BUSY_CALLS )
		return(SPARK_STATUS_BUSY);
	*out = &ATTACHED_PACK;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdLazyPackDestroy(SparkWeightdLazyPack *pack)
{
	assert(pack == &ATTACHED_PACK);
	return(SPARK_STATUS_OK);
}

static SparkStatus run_lazy_open(uint32_t busy_calls,uint32_t fail_errno,uint32_t interrupt_once)
{
	SparkGlm5NextModuleState state = {0};
	SparkGlm5NextStagePackEntry entries[1] = {0};
	SparkStatus status;
	PAUSE_CALLS = 0u;
	CREATE_CALLS = 0u;
	CREATE_BUSY_CALLS = busy_calls;
	PAUSE_FAIL_ERRNO = fail_errno;
	PAUSE_INTERRUPT_ONCE = interrupt_once;
	state.tp_degree = 1u;
	PAUSE_RECORDING = 1u;
	status = SparkGlm5NextLazyOpen(&state,"/pack",4096u,entries,1u);
	PAUSE_RECORDING = 0u;
	assert(state.lazy_pack == (status == SPARK_STATUS_OK ? &ATTACHED_PACK : 0));
	return(status);
}

static void assert_pause(uint32_t index,time_t seconds,long nanoseconds)
{
	assert(index < PAUSE_CALLS);
	assert(PAUSE_REQUESTS[index].tv_sec == seconds);
	assert(PAUSE_REQUESTS[index].tv_nsec == nanoseconds);
}

static void check_lazy_attach_retry_pauses(void)
{
	uint32_t index;
	assert(setenv(SPARK_WEIGHTD_ATTACH_ENV_SHA256,"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",1) == 0);
	assert(setenv("SPARK_WEIGHTD_EXPERT_POOL_BYTES","1073741824",1) == 0);
	(void)unsetenv("SPARK_GLM5_NEXT_PIN_EXPERTS");
	assert(run_lazy_open(3u,0u,0u) == SPARK_STATUS_OK);
	assert(CREATE_CALLS == 4u && PAUSE_CALLS == 3u);
	for (index=0u; index<PAUSE_CALLS; index++)
		assert_pause(index,1,0l);
	assert(run_lazy_open(UINT32_MAX,0u,0u) == SPARK_STATUS_BUSY);
	assert(CREATE_CALLS == SPARK_GLM5_NEXT_LAZY_ATTACH_ATTEMPTS);
	assert(PAUSE_CALLS == SPARK_GLM5_NEXT_LAZY_ATTACH_ATTEMPTS - 1u);
	for (index=0u; index<PAUSE_CALLS; index++)
		assert_pause(index,1,0l);
	assert(run_lazy_open(1u,0u,1u) == SPARK_STATUS_OK);
	assert(CREATE_CALLS == 2u && PAUSE_CALLS == 2u);
	assert_pause(0u,1,0l);
	assert_pause(1u,0,400000000l);
	assert(run_lazy_open(UINT32_MAX,EINVAL,0u) == SPARK_STATUS_INTERNAL_ERROR);
	assert(CREATE_CALLS == 1u && PAUSE_CALLS == 1u);
	assert_pause(0u,1,0l);
	(void)unsetenv(SPARK_WEIGHTD_ATTACH_ENV_SHA256);
	(void)unsetenv("SPARK_WEIGHTD_EXPERT_POOL_BYTES");
}

SparkStatus SparkWeightdMapAcquire(SparkWeightdMap *map,const SparkWeightdExpertKey *keys,uint32_t count,uint64_t *identifier,uint64_t timeout)
{
	(void)map; (void)timeout;
	assert(STEP == 0u && count == 2u);
	assert(keys[0].layer == 3u && keys[0].expert == 0u);
	assert(keys[1].layer == 3u && keys[1].expert == 17u);
	STEP = 1u;
	*identifier = 123u;
	return(FAIL_ACQUIRE != 0u ? SPARK_STATUS_IO_ERROR : SPARK_STATUS_OK);
}

SparkStatus SparkWeightdMapBeginUse(SparkWeightdMap *map,uint64_t identifier,void **address)
{
	(void)map;
	assert(STEP == 1u && identifier == 123u);
	STEP = 2u;
	*address = ADDRESS;
	return(SPARK_STATUS_OK);
}

int32_t SparkGlm5NextLaunchCudaLayerMlpExperts(const SparkGlm5NextCudaWave *wave,uint32_t local_layer)
{
	assert(STEP == 2u && local_layer == 0u);
	assert(wave->expert_lease_base == ADDRESS && wave->expert_lease_local_layer == 0u);
	STEP = 3u;
	return(FAIL_LAUNCH != 0u ? -1 : 0);
}

SparkStatus SparkWeightdMapRecordCompletion(SparkWeightdMap *map,uint64_t identifier,cudaStream_t stream)
{
	(void)map; (void)stream;
	assert(STEP == 3u && identifier == 123u);
	STEP = 4u;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdMapRelease(SparkWeightdMap *map,uint64_t identifier,uint64_t timeout)
{
	(void)map; (void)timeout;
	assert(identifier == 123u && (STEP == 4u || STEP == 5u || (FAIL_ACQUIRE != 0u && STEP == 1u)));
	STEP = 5u;
	return(FAIL_RELEASE != 0u ? SPARK_STATUS_IO_ERROR : SPARK_STATUS_OK);
}

static void check_manifest_geometry(void)
{
	SparkGlm5NextStagePackEntry entries[2] = {0};
	SparkWeightdRange ranges[1153] = {0};
	SparkWeightdRangeGroup groups[288] = {0};
	SparkWeightdManifest manifest = {0};
	SparkGlm5NextManifestContext context = {entries,2u};
	uint32_t expert,tensor,plane,index;
	for (tensor=0u; tensor<2u; tensor++)
	{
		entries[tensor].tensor_kind = SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_UP_GATE + tensor;
		entries[tensor].layer_index = 3u;
		entries[tensor].weight_codec = SPARK_WEIGHT_CODEC_FP8_E4M3;
		entries[tensor].scale_encoding = SPARK_WEIGHT_SCALE_ENCODING_F32;
		entries[tensor].group_count = 288u;
		entries[tensor].payload_offset = 4096u + (tensor * 131072u);
		entries[tensor].payload_bytes = 288u * 256u;
		entries[tensor].scale_offset = entries[tensor].payload_offset + entries[tensor].payload_bytes;
		entries[tensor].scale_bytes = 288u * 16u;
	}
	for (expert=0u; expert<288u; expert++)
	{
		groups[expert] = (SparkWeightdRangeGroup){3u,expert,expert * 4u,4u};
		for (index=0u; index<4u; index++)
		{
			tensor = index / 2u;
			plane = index % 2u;
			ranges[expert * 4u + index] = (SparkWeightdRange){.offset = (plane == 0u ? entries[tensor].payload_offset + expert * 256u : entries[tensor].scale_offset + expert * 16u),.bytes = plane == 0u ? 256u : 16u,.layer = 3u,.expert = expert,.kind = entries[tensor].tensor_kind * 2u + plane};
		}
	}
	manifest.ranges = ranges;
	manifest.groups = groups;
	manifest.range_count = 1152u;
	manifest.group_count = 288u;
	assert(SparkGlm5NextManifestCheck(&manifest,&context) == SPARK_STATUS_OK);
	ranges[69].offset++;
	assert(SparkGlm5NextManifestCheck(&manifest,&context) == SPARK_STATUS_SCHEMA_ERROR);
	ranges[69].offset--;
	ranges[69].bytes++;
	assert(SparkGlm5NextManifestCheck(&manifest,&context) == SPARK_STATUS_SCHEMA_ERROR);
	ranges[69].bytes--;
	groups[17].range_count = 3u;
	assert(SparkGlm5NextManifestCheck(&manifest,&context) == SPARK_STATUS_SCHEMA_ERROR);
	groups[17].range_count = 4u;
	manifest.group_count--;
	assert(SparkGlm5NextManifestCheck(&manifest,&context) == SPARK_STATUS_SCHEMA_ERROR);
	manifest.group_count++;
	manifest.range_count++;
	assert(SparkGlm5NextManifestCheck(&manifest,&context) == SPARK_STATUS_SCHEMA_ERROR);
	manifest.range_count--;
	assert(SparkGlm5NextManifestCheck(&manifest,&context) == SPARK_STATUS_OK);
}

static void check_codec_manifest(uint32_t codec,uint32_t encoding,uint64_t payload,uint64_t block)
{
	SparkGlm5NextStagePackEntry entries[2] = {0};
	SparkWeightdRange ranges[288u * 6u + 1u];
	SparkWeightdRangeGroup groups[288] = {0};
	SparkWeightdManifest manifest = {0};
	SparkGlm5NextManifestContext context = {entries,2u};
	SparkExpertPlanes planes[2];
	uint32_t expert,tensor,plane,count,per_expert,index,saved;
	memset(ranges,0,sizeof(ranges));
	for (tensor=0u; tensor<2u; tensor++)
	{
		entries[tensor].tensor_kind = SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_UP_GATE + tensor;
		entries[tensor].layer_index = 3u;
		entries[tensor].weight_codec = codec;
		entries[tensor].scale_encoding = encoding;
		entries[tensor].group_count = 288u;
		entries[tensor].payload_offset = 4096u + (tensor * 1048576u);
		entries[tensor].payload_bytes = 288u * payload;
		entries[tensor].scale_offset = block == 0u ? 0u : entries[tensor].payload_offset + entries[tensor].payload_bytes;
		entries[tensor].scale_bytes = block == 0u ? 0u : 288u * (block + (encoding == SPARK_WEIGHT_SCALE_ENCODING_UE4M3_F32_GLOBAL ? 4u : 0u));
		assert(SparkExpertPlanesDescribe(entries[tensor].tensor_kind,codec,encoding,288u,entries[tensor].payload_offset,entries[tensor].payload_bytes,entries[tensor].scale_offset,entries[tensor].scale_bytes,&planes[tensor]) == SPARK_STATUS_OK);
	}
	per_expert = planes[0].count * 2u;
	count = 0u;
	for (expert=0u; expert<288u; expert++)
	{
		groups[expert] = (SparkWeightdRangeGroup){3u,expert,count,per_expert};
		for (tensor=0u; tensor<2u; tensor++)
			for (plane=0u; plane<planes[tensor].count; plane++)
				ranges[count++] = (SparkWeightdRange){.offset = SparkExpertPlaneOffset(&planes[tensor].planes[plane],expert),.bytes = planes[tensor].planes[plane].bytes,.layer = 3u,.expert = expert,.kind = planes[tensor].planes[plane].kind};
	}
	manifest.ranges = ranges;
	manifest.groups = groups;
	manifest.range_count = count;
	manifest.group_count = 288u;
	assert(SparkGlm5NextManifestCheck(&manifest,&context) == SPARK_STATUS_OK);
	for (index=0u; index<per_expert; index++)
	{
		ranges[5u * per_expert + index].offset++;
		assert(SparkGlm5NextManifestCheck(&manifest,&context) == SPARK_STATUS_SCHEMA_ERROR);
		ranges[5u * per_expert + index].offset--;
		saved = ranges[7u * per_expert + index].kind;
		ranges[7u * per_expert + index].kind = 999u;
		assert(SparkGlm5NextManifestCheck(&manifest,&context) == SPARK_STATUS_SCHEMA_ERROR);
		ranges[7u * per_expert + index].kind = saved;
	}
	groups[9].range_count--;
	manifest.range_count--;
	assert(SparkGlm5NextManifestCheck(&manifest,&context) == SPARK_STATUS_SCHEMA_ERROR);
	groups[9].range_count++;
	manifest.range_count++;
	entries[1].scale_encoding = SPARK_WEIGHT_SCALE_ENCODING_F32 + (encoding == SPARK_WEIGHT_SCALE_ENCODING_F32 ? 1u : 0u);
	assert(SparkGlm5NextManifestCheck(&manifest,&context) == SPARK_STATUS_SCHEMA_ERROR);
	entries[1].scale_encoding = encoding;
	entries[0].weight_codec = 99u;
	assert(SparkGlm5NextManifestCheck(&manifest,&context) == SPARK_STATUS_UNSUPPORTED);
	entries[0].weight_codec = codec;
	assert(SparkGlm5NextManifestCheck(&manifest,&context) == SPARK_STATUS_OK);
	if ( codec == SPARK_WEIGHT_CODEC_NVFP4_E2M1 )
	{
		assert(per_expert == 6u);
		assert(ranges[2].kind == SPARK_EXPERT_PLANE_GLOBAL_KIND_BASE + SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_UP_GATE && ranges[1].kind == (SPARK_GLM5_NEXT_STAGEPACK_TENSOR_EXPERT_UP_GATE * 2u) + 1u);
		assert(ranges[2].offset == entries[0].scale_offset && ranges[2].bytes == 4u);
		assert(ranges[1].offset == entries[0].scale_offset + 288u * 4u && ranges[1].bytes == block);
		assert(288u * 6u * 42u <= SPARK_WEIGHTD_RANGE_COUNT_MAX && per_expert <= SPARK_WEIGHTD_RANGES_PER_EXPERT_MAX);
	}
	if ( codec == SPARK_WEIGHT_CODEC_BF16 )
	{
		assert(per_expert == 2u);
		entries[1].scale_offset = entries[1].payload_offset + entries[1].payload_bytes;
		entries[1].scale_bytes = 288u * 4u;
		assert(SparkGlm5NextManifestCheck(&manifest,&context) == SPARK_STATUS_SCHEMA_ERROR);
		entries[1].scale_offset = 0u;
		entries[1].scale_bytes = 0u;
		assert(SparkGlm5NextManifestCheck(&manifest,&context) == SPARK_STATUS_OK);
	}
}

int main(void)
{
	SparkGlm5NextModuleState state = {0};
	SparkGlm5NextExecutionSlot slot = {0};
	SparkGlm5NextTpChain chain = {0};
	SparkGlm5NextTpChain *recovered;
	SparkWeightdLazyPack pack = {0};
	uint32_t offsets[SPARK_GLM5_NEXT_MODEL_LAYER_COUNT * 289u] = {0},i,scenario;
	cudaEvent_t event;
	cudaStream_t stream;
	check_manifest_geometry();
	check_codec_manifest(SPARK_WEIGHT_CODEC_FP8_E4M3,SPARK_WEIGHT_SCALE_ENCODING_F32,256u,16u);
	check_codec_manifest(SPARK_WEIGHT_CODEC_BF16,SPARK_WEIGHT_SCALE_ENCODING_NONE,512u,0u);
	check_codec_manifest(SPARK_WEIGHT_CODEC_NVFP4_E2M1,SPARK_WEIGHT_SCALE_ENCODING_UE4M3_F32_GLOBAL,128u,16u);
	check_codec_manifest(SPARK_WEIGHT_CODEC_MXFP4_E2M1,SPARK_WEIGHT_SCALE_ENCODING_E8M0,128u,8u);
	check_lazy_attach_retry_pauses();
	atomic_init(&state.lazy_retained[0],0);
	assert(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking) == cudaSuccess);
	state.execution_stream = slot.stream = stream;
	assert(SparkStageModuleCudaWaitInitialize(&state.stream_wait,stream) == SPARK_STATUS_OK);
	assert(cudaEventCreateWithFlags(&event,cudaEventDisableTiming) == cudaSuccess);
	assert(cudaEventRecord(event,stream) == cudaSuccess);
	for (i=0u; i<289u; i++) offsets[3u * 289u + i] = i == 0u ? 0u : (i <= 17u ? 4u : 8u);
	state.lazy_pack = &pack;
	slot.route_ready_event = event;
	slot.route_recorded = 1u;
	slot.host_group_row_offset = offsets;
	chain.state = &state;
	chain.slot = &slot;
	chain.wave.first_layer_index = 3u;
	chain.wave.row_count = 1u;
	for (scenario=0u; scenario<4u; scenario++)
	{
		STEP = 0u;
		FAIL_ACQUIRE = scenario == 1u;
		FAIL_LAUNCH = scenario == 2u;
		FAIL_RELEASE = scenario == 3u;
		assert(SparkGlm5NextLazyExperts(&chain) == (FAIL_ACQUIRE != 0u ? SPARK_STATUS_IO_ERROR : (FAIL_LAUNCH != 0u ? SPARK_STATUS_INTERNAL_ERROR : SPARK_STATUS_OK)));
		assert(SparkGlm5NextLazyRelease(&chain) == (FAIL_RELEASE != 0u ? SPARK_STATUS_IO_ERROR : SPARK_STATUS_OK));
		assert(STEP == 5u);
		assert(chain.expert_lease == (FAIL_RELEASE != 0u ? 123u : 0u));
		assert(chain.wave.expert_lease_base == 0);
		if ( FAIL_RELEASE != 0u )
		{
			assert(chain.expert_lease_recorded == 1u);
			atomic_store(&state.lazy_retained[0],&chain);
			assert(SparkGlm5NextLazyRecoverLease(&state,0u,&recovered) == SPARK_STATUS_IO_ERROR);
			assert(recovered == 0 && atomic_load(&state.lazy_retained[0]) == &chain);
			FAIL_RELEASE = 0u;
			assert(SparkGlm5NextLazyRecoverLease(&state,0u,&recovered) == SPARK_STATUS_OK);
			assert(recovered == &chain && atomic_load(&state.lazy_retained[0]) == 0);
			assert(chain.expert_lease == 0u && chain.expert_lease_recorded == 0u);
			assert(SparkGlm5NextLazyRecoverLease(&state,0u,&recovered) == SPARK_STATUS_NOT_FOUND && recovered == 0);
		}
	}
	assert(SparkStageModuleCudaWaitDestroy(&state.stream_wait) == SPARK_STATUS_OK);
	assert(cudaEventDestroy(event) == cudaSuccess);
	assert(cudaStreamDestroy(stream) == cudaSuccess);
	puts("PASS GLM lazy dispatch ordering, partial failure ownership, attach retry pauses and fp8/bf16/nvfp4/mxfp4 manifest planes (CUDA/map stubs)");
	return(0);
}
