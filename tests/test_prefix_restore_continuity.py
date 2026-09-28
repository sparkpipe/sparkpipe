#!/usr/bin/env python3
"""A committed prefix restore sets the lane's continuity baseline, a mismatched frame is refused and an aborted restore is refused, identically for every resident decode family."""
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
HARNESS = r'''
#include <assert.h>
#include "modules/@family@_resident_decode_stage/source/spark_@family@_resident_decode_stage_module.c"
#define main SparkUnusedKvTestMain
#include "tests/test_kv_cache.c"
#undef main
@stubs@
static Spark@Camel@ModuleState state;
static SparkTestKvPageFixture pages;
static SparkKvLaneTransaction owners[4];
static uint32_t logical[16],physical[16],page_counts[4],mutable_pages[4];

static void fixture(void)
{
	memset(&state,0,sizeof(state));
	memset(owners,0,sizeof(owners));
	SparkTestKvPageInitialize(&pages);
	state.kv_page_cache = pages.cache;
	state.resident_sequence_capacity = 4u;
	state.pipeline_slot_count = 1u;
	state.pages_per_sequence = 4u;
	state.max_sequence_positions = 64u;
	state.kv_lane_transactions = owners;
	state.kv_lane_logical_pages = logical;
	@fixture@
	SparkStageModuleAtomicStateArrayInitialize(state.lane_states,4u);
}

static void publish_prefix(void)
{
	SparkModelDriverCacheLane lane;
	uint32_t mutable_page;
	SparkTestKvPageLane(&lane,11u,0u,0u,4u);
	SparkTestKvPagePublish(&lane,4u,71u);
	assert(SparkKvPageCacheBeginLane(&state.kv_page_cache,&lane,&mutable_page) == SPARK_STATUS_OK);
	assert(SparkKvPageCacheCompleteLane(&state.kv_page_cache,&lane) == SPARK_STATUS_OK);
	assert(SparkKvPageCacheReleaseLane(&state.kv_page_cache,0u,11u) == SPARK_STATUS_OK);
}

static SparkStatus admit(SparkModelDriverCacheLane *lane,uint32_t admission_flags)
{
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	memset(&request,0,sizeof(request));
	memset(&decision,0,sizeof(decision));
	request.descriptor_bytes = sizeof(request);
	request.program_id = 1u;
	request.submission_id = 100u + lane->sequence_id;
	request.control_generation = 1u;
	request.transaction_id = 200u + lane->sequence_id;
	request.request_generation = 1u;
	request.step_generation = 1u;
	request.request_id = lane->sequence_id;
	request.sequence_id = lane->sequence_id;
	request.sequence_position = lane->sequence_position;
	request.active_slot_count = 1u;
	request.new_token_count = 1u;
	request.admission_flags = admission_flags;
	request.cache_lanes = lane;
	request.cache_lane_count = 1u;
	return(Spark@Camel@AdmissionPredicate(&state,&request,&decision));
}

static SparkStatus continuity(uint64_t sequence,uint64_t position,uint8_t *bound,uint64_t *sequence_ids,uint64_t *next_positions)
{
	Spark@Camel@ResidentDecodeStageBatchView batch;
	Spark@Camel@ClaimedContinuityContext context;
	uint32_t slots[1] = {1u};
	uint64_t positions[1],sequences[1];
	SparkStatus status;
	positions[0] = position;
	sequences[0] = sequence;
	memset(&batch,0,sizeof(batch));
	batch.row_count = 1u;
	batch.active_sequence_count = 1u;
	batch.row_resident_slots = slots;
	batch.row_positions = positions;
	batch.row_sequence_ids = sequences;
	memset(&context,0,sizeof(context));
	context.state = &state;
	context.batch = &batch;
	context.bound = bound;
	context.sequence_ids = sequence_ids;
	context.next_positions = next_positions;
	status = SparkStageModuleIndexSetClaimAndPrepare(state.lane_states,4u,slots,1u,Spark@Camel@PrepareClaimedContinuity,&context);
	if ( status == SPARK_STATUS_OK )
		SparkStageModuleIndexSetRelease(state.lane_states,4u,slots,1u);
	return(status);
}

int main(void)
{
	SparkModelDriverCacheLane lane;
	uint8_t bound[1];
	uint64_t sequence_ids[1],next_positions[1];
	fixture();
	publish_prefix();
	SparkTestKvPageLane(&lane,12u,1u,4u,5u);
	SparkTestKvPagePrefix(&lane,4u,71u);
	assert(admit(&lane,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE) == SPARK_STATUS_OK);
	assert(admit(&lane,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT) == SPARK_STATUS_OK);
	assert(continuity(13u,4u,bound,sequence_ids,next_positions) == SPARK_STATUS_VALIDATION_FAILED);
	assert(continuity(12u,5u,bound,sequence_ids,next_positions) == SPARK_STATUS_VALIDATION_FAILED);
	assert(continuity(12u,4u,bound,sequence_ids,next_positions) == SPARK_STATUS_OK);
	assert(bound[0] == 1u && sequence_ids[0] == 12u && next_positions[0] == 5u);
	assert(admit(&lane,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT) == SPARK_STATUS_OK);
	assert(state.kv_page_cache.sequences[1].sequence_id == 0u);
	assert(continuity(12u,4u,bound,sequence_ids,next_positions) == SPARK_STATUS_VALIDATION_FAILED);
	return(0);
}
'''

ARRAY_FIXTURE = "state.kv_lane_page_count = page_counts;\n\tstate.kv_lane_mutable_page = mutable_pages;"
TP_LAUNCH_STUBS = ("cudaError_t SparkTpLaunchAccumU64Max(cudaStream_t stream,uint64_t *destination,const uint64_t *source,uint32_t count) { (void)stream;(void)destination;(void)source;(void)count;return(cudaErrorInvalidValue); }\n"
                   "cudaError_t SparkTpLaunchAddF32(cudaStream_t stream,float *destination,const void *source,uint32_t count) { (void)stream;(void)destination;(void)source;(void)count;return(cudaErrorInvalidValue); }")
TRANSACTION_FIXTURE = ("state.kv_transactions.cache = &state.kv_page_cache;\n\tstate.kv_transactions.lanes = owners;\n"
                       "\tstate.kv_transactions.logical_pages = logical;\n\tstate.kv_transactions.physical_pages = physical;\n"
                       "\tstate.kv_transactions.page_capacity = 4u;\n\tassert(pthread_mutex_init(&state.kv_mutex,0) == 0);")
FAMILIES = (
    ("glm52", "Glm52", ARRAY_FIXTURE, TP_LAUNCH_STUBS, ["-DGLM_EXPERT_WEIGHT_CODEC=5", '-DGLM_EXPERT_CODEC_NAME="fp8"', '-DGLM_MODEL_REVISION="fixture"', '-DGLM_CONTRACT_SHA256="fixture"']),
    ("ling", "Ling", ARRAY_FIXTURE, TP_LAUNCH_STUBS, ["-DLING_EXPERT_WEIGHT_CODEC=1u", '-DLING_EXPERT_CODEC_NAME="bf16"', '-DLING_MODEL_REVISION="fixture"', '-DLING_CONTRACT_SHA256="fixture"']),
    ("laguna", "Laguna", TRANSACTION_FIXTURE, "", ["-DLAGUNA_EXPERT_WEIGHT_CODEC=1", '-DLAGUNA_EXPERT_CODEC_NAME="bf16"', '-DLAGUNA_MODEL_REVISION="fixture"', '-DLAGUNA_CONTRACT_SHA256="fixture"']),
)


def run_family(family, camel, fixture, stubs, defines, directory):
    source, binary = Path(directory) / f"{family}.c", Path(directory) / family
    source.write_text(HARNESS.replace("@family@", family).replace("@Camel@", camel).replace("@fixture@", fixture).replace("@stubs@", stubs))
    includes = [".", "include", "src", "tests/cuda_stub", "model-families/common/include", f"model-families/{family}/include",
                f"modules/{family}_resident_decode_stage/include", f"modules/{family}_resident_decode_stage/source"]
    subprocess.run(["cc", "-std=c11", "-D_GNU_SOURCE", "-O2", "-ffunction-sections", "-fdata-sections",
                    "-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections",
                    *["-I" + path for path in includes], "-include", f"model-families/{family}/include/sparkpipe/spark_{family}_model.h",
                    "-DSPARK_BATCH_BUCKET=1024u", *defines, str(source), "runtime/stage_module_common.c", "cache/kv_cache.c",
                    "cache/kv_page_cache.c", "cache/kv_page_store.c", "tests/cuda_stub/cuda_runtime_stub.c", "-lpthread",
                    "-o", str(binary)], cwd=ROOT, check=True)
    subprocess.run([str(binary)], check=True)


def main():
    with tempfile.TemporaryDirectory() as directory:
        for family, camel, fixture, stubs, defines in FAMILIES:
            run_family(family, camel, fixture, stubs, defines, directory)
            print(f"PASS {family} applies a committed prefix restore, refuses a mismatched frame and refuses an aborted restore")


if __name__ == "__main__":
    main()
