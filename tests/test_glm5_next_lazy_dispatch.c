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
	assert(identifier == 123u && (STEP == 4u || (FAIL_ACQUIRE != 0u && STEP == 1u)));
	STEP = 5u;
	return(FAIL_RELEASE != 0u ? SPARK_STATUS_IO_ERROR : SPARK_STATUS_OK);
}

int main(void)
{
	SparkGlm5NextModuleState state = {0};
	SparkGlm5NextExecutionSlot slot = {0};
	SparkGlm5NextTpChain chain = {0};
	SparkWeightdLazyPack pack = {0};
	uint32_t offsets[289],i,scenario;
	cudaEvent_t event;
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
		assert(FAIL_RELEASE != 0u || chain.wave.expert_lease_base == 0);
	}
	assert(cudaEventDestroy(event) == cudaSuccess);
	puts("PASS GLM lazy dispatch ordering and partial failure ownership (CUDA/map stubs)");
	return(0);
}
