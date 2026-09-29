#!/usr/bin/env python3
"""Drive the glm52 module's client-reset admission on the host with stubbed CUDA and page cache."""
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
HARNESS = r'''
#include <assert.h>
#include "modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_module.c"

static uint32_t RELEASE_ALL_CALLS,SYNC_CALLS;
static SparkStatus RELEASE_ALL_STATUS;
static cudaError_t SYNC_STATUS = cudaSuccess;

SparkStatus SparkKvPageCacheReleaseAll(SparkKvPageCache *cache) { (void)cache; RELEASE_ALL_CALLS++; return(RELEASE_ALL_STATUS); }
SparkStatus SparkKvPageCacheReleaseLane(SparkKvPageCache *cache,uint32_t slot,uint64_t sequence) { (void)cache; (void)slot; (void)sequence; assert(0); return(SPARK_STATUS_INTERNAL_ERROR); }
SparkStatus SparkKvPageCachePrepareLane(SparkKvPageCache *cache,const SparkModelDriverCacheLane *lane,uint32_t *pages,uint32_t capacity,uint32_t *count) { (void)cache; (void)lane; (void)pages; (void)capacity; (void)count; assert(0); return(SPARK_STATUS_INTERNAL_ERROR); }
SparkStatus SparkKvPageCacheBeginLaneTransaction(SparkKvPageCache *cache,const SparkModelDriverCacheLane *lane,uint32_t *page,uint32_t *flags) { (void)cache; (void)lane; (void)page; (void)flags; assert(0); return(SPARK_STATUS_INTERNAL_ERROR); }
SparkStatus SparkKvPageCacheRollbackLaneTransaction(SparkKvPageCache *cache,const SparkModelDriverCacheLane *lane,uint32_t flags) { (void)cache; (void)lane; (void)flags; assert(0); return(SPARK_STATUS_INTERNAL_ERROR); }
cudaError_t cudaStreamSynchronize(cudaStream_t stream) { (void)stream; SYNC_CALLS++; return(SYNC_STATUS); }
cudaError_t cudaGetLastError(void) { return(cudaSuccess); }
const char *cudaGetErrorString(cudaError_t error) { (void)error; return("stub"); }
const char *cudaGetErrorName(cudaError_t error) { (void)error; return("stub"); }

static SparkGlm52ModuleState state;

static void Bind(void)
{
	uint32_t lane;
	for (lane=0u; lane<state.resident_sequence_capacity; lane++)
	{
		atomic_store(&state.lane_bound[lane],1u);
		atomic_store(&state.lane_sequence_ids[lane],100u + lane);
		atomic_store(&state.lane_next_positions[lane],7u + lane);
	}
	RELEASE_ALL_CALLS = 0u;
	SYNC_CALLS = 0u;
	RELEASE_ALL_STATUS = SPARK_STATUS_OK;
	SYNC_STATUS = cudaSuccess;
}

static uint32_t Bound(void)
{
	uint32_t lane,count = 0u;
	for (lane=0u; lane<state.resident_sequence_capacity; lane++)
		count += atomic_load(&state.lane_bound[lane]) != 0u || atomic_load(&state.lane_sequence_ids[lane]) != 0u || atomic_load(&state.lane_next_positions[lane]) != 0u ? 1u : 0u;
	return(count);
}

static uint32_t Claimed(void)
{
	uint32_t index,count = 0u;
	for (index=0u; index<state.pipeline_slot_count; index++)
		count += atomic_load(&state.slot_states[index]) != SPARK_STAGE_MODULE_SLOT_FREE ? 1u : 0u;
	for (index=0u; index<state.resident_sequence_capacity; index++)
		count += atomic_load(&state.lane_states[index]) != SPARK_STAGE_MODULE_SLOT_FREE ? 1u : 0u;
	return(count);
}

static SparkModelDriverAdmissionRequest ResetRequest(uint64_t generation)
{
	SparkModelDriverAdmissionRequest request;
	memset(&request,0,sizeof(request));
	request.descriptor_bytes = (uint32_t)sizeof(request);
	request.program_id = 7u;
	request.control_generation = generation;
	request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_RESET;
	return(request);
}

static SparkStatus Admit(const SparkModelDriverAdmissionRequest *request,SparkModelDriverAdmissionDecision *decision)
{
	memset(decision,0xa5,sizeof(*decision));
	return(SparkGlm52ModuleAdmit(&state,request,decision));
}

static void TestResetClearsEveryLane(void)
{
	SparkModelDriverAdmissionRequest request = ResetRequest(1u);
	SparkModelDriverAdmissionDecision decision;
	Bind();
	assert(Admit(&request,&decision) == SPARK_STATUS_OK);
	assert(decision.accepted == 1u && decision.rejection_reason == SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED);
	assert(SYNC_CALLS == 1u && RELEASE_ALL_CALLS == 1u && Bound() == 0u && Claimed() == 0u);
	request = ResetRequest(2u);
	Bind();
	assert(Admit(&request,&decision) == SPARK_STATUS_OK && decision.accepted == 1u && Bound() == 0u && Claimed() == 0u);
}

static void TestResetWaitsForInflightSlot(void)
{
	SparkModelDriverAdmissionRequest request = ResetRequest(3u);
	SparkModelDriverAdmissionDecision decision;
	Bind();
	atomic_store(&state.slot_states[1],SPARK_STAGE_MODULE_SLOT_CLAIMED);
	assert(Admit(&request,&decision) == SPARK_STATUS_BUSY);
	assert(decision.accepted == 0u && SYNC_CALLS == 0u && RELEASE_ALL_CALLS == 0u && Bound() == state.resident_sequence_capacity);
	assert(Claimed() == 1u);
	atomic_store(&state.slot_states[1],SPARK_STAGE_MODULE_SLOT_FREE);
	atomic_store(&state.lane_states[2],SPARK_STAGE_MODULE_SLOT_CLAIMED);
	assert(Admit(&request,&decision) == SPARK_STATUS_BUSY);
	assert(decision.accepted == 0u && RELEASE_ALL_CALLS == 0u && Bound() == state.resident_sequence_capacity && Claimed() == 1u);
	atomic_store(&state.lane_states[2],SPARK_STAGE_MODULE_SLOT_FREE);
	assert(Admit(&request,&decision) == SPARK_STATUS_OK && decision.accepted == 1u && Bound() == 0u && Claimed() == 0u);
}

static void TestResetFailures(void)
{
	SparkModelDriverAdmissionRequest request = ResetRequest(4u);
	SparkModelDriverAdmissionDecision decision;
	SparkModelDriverCacheLane lane;
	Bind();
	RELEASE_ALL_STATUS = SPARK_STATUS_BUSY;
	assert(Admit(&request,&decision) == SPARK_STATUS_BUSY);
	assert(decision.accepted == 0u && RELEASE_ALL_CALLS == 1u && Bound() == state.resident_sequence_capacity && Claimed() == 0u);
	Bind();
	SYNC_STATUS = cudaErrorUnknown;
	assert(Admit(&request,&decision) == SPARK_STATUS_IO_ERROR);
	assert(decision.accepted == 0u && RELEASE_ALL_CALLS == 0u && Claimed() == state.pipeline_slot_count + state.resident_sequence_capacity);
	SparkStageModuleAtomicStateArrayInitialize(state.slot_states,state.pipeline_slot_count);
	SparkStageModuleAtomicStateArrayInitialize(state.lane_states,state.resident_sequence_capacity);
	Bind();
	memset(&lane,0,sizeof(lane));
	request.cache_lanes = &lane;
	request.cache_lane_count = 1u;
	assert(Admit(&request,&decision) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(decision.accepted == 0u && RELEASE_ALL_CALLS == 0u && Bound() == state.resident_sequence_capacity && Claimed() == 0u);
	request = ResetRequest(0u);
	assert(Admit(&request,&decision) == SPARK_STATUS_INVALID_ARGUMENT && RELEASE_ALL_CALLS == 0u);
}

int main(void)
{
	state.pipeline_slot_count = 2u;
	state.resident_sequence_capacity = 4u;
	state.max_sequence_positions = 2048u;
	SparkStageModuleAtomicStateArrayInitialize(state.slot_states,state.pipeline_slot_count);
	SparkStageModuleAtomicStateArrayInitialize(state.lane_states,state.resident_sequence_capacity);
	TestResetClearsEveryLane();
	TestResetWaitsForInflightSlot();
	TestResetFailures();
	printf("glm52 reset: ok\n");
	return(0);
}
'''


def main():
    identity = subprocess.check_output([sys.executable, "tools/glm52_model_contract.py", "--print-build-identity", "fp8"], cwd=ROOT, text=True).split()
    with tempfile.TemporaryDirectory() as directory:
        source, binary = Path(directory) / "probe.c", Path(directory) / "probe"
        source.write_text(HARNESS)
        includes = [".", "include", "tests/cuda_stub", "model-families/common/include", "model-families/glm52/include",
                    "modules/glm52_resident_decode_stage/include", "modules/glm52_resident_decode_stage/source"]
        subprocess.run(["cc", "-std=c11", "-D_GNU_SOURCE", "-O1", "-g", "-ffunction-sections", "-fdata-sections",
                        "-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections",
                        *["-I" + p for p in includes], "-DSPARK_BATCH_BUCKET=16u", "-DGLM_EXPERT_WEIGHT_CODEC=5",
                        '-DGLM_EXPERT_CODEC_NAME="fp8"', f'-DGLM_MODEL_REVISION="{identity[0]}"', f'-DGLM_CONTRACT_SHA256="{identity[1]}"',
                        '-DGLM_MODEL_DESCRIPTION_SHA256="fixture"', "-include", "model-families/glm52/include/sparkpipe/spark_glm52_model.h",
                        str(source), "runtime/stage_module_common.c", "src/spark_status.c", "-o", str(binary), "-pthread"], cwd=ROOT, check=True)
        subprocess.run([str(binary)], check=True)
    print("PASS glm52 reset: client reset accepted, every lane unbound, waits for in-flight slots and lanes, stream and release failures stay loud")


if __name__ == "__main__":
    main()
