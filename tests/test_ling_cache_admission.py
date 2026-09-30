#!/usr/bin/env python3
"""Ling cache admission: a failed commit leaves no partial state and an abort never replays another transaction's mutations."""
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
HARNESS = r'''
#include <assert.h>
#include "modules/ling_resident_decode_stage/source/spark_ling_resident_decode_stage_module.c"
#define main SparkUnusedKvTestMain
#include "tests/test_kv_cache.c"
#undef main
cudaError_t SparkTpLaunchAddF32(cudaStream_t stream,float *destination,const void *b,uint32_t element_count)
{
	(void)stream;
	(void)destination;
	(void)b;
	(void)element_count;
	return(cudaErrorUnknown);
}
cudaError_t SparkTpLaunchAccumU64Max(cudaStream_t stream,uint64_t *destination,const uint64_t *source,uint32_t element_count)
{
	(void)stream;
	(void)destination;
	(void)source;
	(void)element_count;
	return(cudaErrorUnknown);
}
static SparkLingModuleState state;
static SparkTestKvPageFixture pages;
static SparkModelDriverCacheLane remembered[4];
static uint32_t logical[16],page_counts[4],mutable_pages[4],mutation_flags[4];

static void fixture(void)
{
	memset(&state,0,sizeof(state));
	memset(remembered,0,sizeof(remembered));
	SparkTestKvPageInitialize(&pages);
	state.kv_page_cache = pages.cache;
	state.resident_sequence_capacity = 4u;
	state.pages_per_sequence = 4u;
	state.kv_lane_cache_lanes = remembered;
	state.kv_lane_logical_pages = logical;
	state.kv_lane_page_count = page_counts;
	state.kv_lane_mutable_page = mutable_pages;
	state.kv_lane_mutation_flags = mutation_flags;
	memset(mutation_flags,0,sizeof(mutation_flags));
}

static SparkStatus admit(SparkModelDriverCacheLane *lanes,uint32_t count,uint32_t admission_flags)
{
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	memset(&request,0,sizeof(request));
	memset(&decision,0,sizeof(decision));
	request.descriptor_bytes = sizeof(request);
	request.program_id = 1u;
	request.cache_lanes = lanes;
	request.cache_lane_count = count;
	request.admission_flags = admission_flags;
	return(SparkLingAdmissionPredicate(&state,&request,&decision));
}

static void bind_and_run(SparkModelDriverCacheLane *lane)
{
	assert(admit(lane,1u,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE) == SPARK_STATUS_OK);
	assert(admit(lane,1u,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT) == SPARK_STATUS_OK);
	assert(SparkKvPageCacheCompleteLane(&state.kv_page_cache,lane) == SPARK_STATUS_OK);
}

static void check_prepared_abort_keeps_running_sequence(void)
{
	SparkModelDriverCacheLane lane;
	fixture();
	SparkTestKvPageLane(&lane,7u,0u,0u,1u);
	bind_and_run(&lane);
	SparkTestKvPageLane(&lane,7u,0u,1u,2u);
	assert(admit(&lane,1u,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE) == SPARK_STATUS_OK);
	assert(admit(&lane,1u,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT) == SPARK_STATUS_OK);
	assert(state.kv_page_cache.sequences[0].sequence_id == 7u && state.kv_page_cache.sequences[0].next_token_position == 1u);
	assert(state.kv_page_cache.sequences[0].mutable_logical_page_index != SPARK_KV_CACHE_NO_BLOCK);
}

static void check_failed_commit_rolls_back_whole_frame(void)
{
	SparkModelDriverCacheLane lanes[3];
	uint64_t resident_blocks;
	fixture();
	SparkTestKvPageLane(&lanes[2],9u,2u,0u,1u);
	bind_and_run(&lanes[2]);
	resident_blocks = state.kv_page_cache.kv_cache_arena->resident_block_count;
	SparkTestKvPageLane(&lanes[0],5u,0u,0u,1u);
	SparkTestKvPageLane(&lanes[1],6u,1u,0u,SPARK_TEST_BLOCK_TOKENS + 1u);
	SparkTestKvPageLane(&lanes[2],9u,2u,1u,2u);
	assert(admit(lanes,3u,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE) == SPARK_STATUS_OK);
	assert(admit(lanes,3u,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT) == SPARK_STATUS_CAPACITY_EXCEEDED);
	assert(state.kv_page_cache.sequences[0].sequence_id == 0u && state.kv_page_cache.sequences[1].sequence_id == 0u);
	assert(state.kv_page_cache.sequences[0].mutable_logical_page_index == SPARK_KV_CACHE_NO_BLOCK);
	assert(state.kv_page_cache.kv_cache_arena->resident_block_count == resident_blocks);
	assert(admit(lanes,3u,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT) == SPARK_STATUS_OK);
	assert(state.kv_page_cache.sequences[2].sequence_id == 9u && state.kv_page_cache.sequences[2].next_token_position == 1u);
	assert(state.kv_page_cache.sequences[2].mutable_logical_page_index != SPARK_KV_CACHE_NO_BLOCK);
}

int main(void)
{
	check_prepared_abort_keeps_running_sequence();
	check_failed_commit_rolls_back_whole_frame();
	return(0);
}
'''


def main():
    with tempfile.TemporaryDirectory() as directory:
        source, binary = Path(directory) / "probe.c", Path(directory) / "probe"
        source.write_text(HARNESS)
        includes = [".", "include", "src", "tests/cuda_stub", "model-families/common/include",
                    "model-families/ling/include", "modules/ling_resident_decode_stage/include",
                    "modules/ling_resident_decode_stage/source"]
        subprocess.run(["cc", "-std=c11", "-D_GNU_SOURCE", "-O2", "-ffunction-sections", "-fdata-sections",
                        "-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections",
                        *["-I" + p for p in includes], "-DSPARK_BATCH_BUCKET=1024u", "-DLING_EXPERT_WEIGHT_CODEC=1u",
                        '-DLING_EXPERT_CODEC_NAME="bf16"', '-DLING_MODEL_REVISION="fixture"',
                        '-DLING_CONTRACT_SHA256="fixture"', str(source), "runtime/stage_module_common.c",
                        "cache/kv_cache.c", "cache/kv_page_cache.c", "cache/kv_page_store.c", "cache/kv_snapshot.c", "src/spark_sha256.c",
                        "tests/cuda_stub/cuda_runtime_stub.c", "-lpthread",
                        "-o", str(binary)], cwd=ROOT, check=True)
        subprocess.run([str(binary)], check=True)
    print("PASS ling cache admission rolls back failed commits and never aborts with another transaction's mutations")


if __name__ == "__main__":
    main()
