#define GLM5_NEXT_EXPERT_WEIGHT_CODEC 5
#define GLM5_NEXT_EXPERT_CODEC_NAME "fp8"
#define GLM5_NEXT_MODEL_REVISION "test"
#define GLM5_NEXT_CONTRACT_SHA256 "test"
#include <assert.h>
#include "../modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_module.c"

static uint32_t STEP,FAIL_ACQUIRE,FAIL_LAUNCH,FAIL_RELEASE;
static uint8_t ADDRESS[256];

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

int main(void)
{
	SparkGlm5NextModuleState state = {0};
	SparkGlm5NextExecutionSlot slot = {0};
	SparkGlm5NextTpChain chain = {0};
	SparkGlm5NextTpChain *recovered;
	SparkWeightdLazyPack pack = {0};
	uint32_t offsets[289],i,scenario;
	cudaEvent_t event;
	check_manifest_geometry();
	atomic_init(&state.lazy_retained[0],0);
	assert(cudaEventCreateWithFlags(&event,cudaEventDisableTiming) == cudaSuccess);
	assert(cudaEventRecord(event,0) == cudaSuccess);
	for (i=0u; i<289u; i++) offsets[i] = i == 0u ? 0u : (i <= 17u ? 4u : 8u);
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
	assert(cudaEventDestroy(event) == cudaSuccess);
	puts("PASS GLM lazy dispatch ordering and partial failure ownership (CUDA/map stubs)");
	return(0);
}
