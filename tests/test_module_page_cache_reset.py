#!/usr/bin/env python3
"""Drive the glm52 and ling modules' client-reset admission (spark_module_reset_page_cache.h) on the host with stubbed CUDA and page cache."""
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
HARNESS = r'''
#include <assert.h>
#include MODULE_SOURCE

static uint32_t RELEASE_ALL_CALLS,SYNC_CALLS;
static SparkStatus RELEASE_ALL_STATUS;
static cudaError_t SYNC_STATUS = cudaSuccess;

SparkStatus SparkKvPageCacheReleaseAll(SparkKvPageCache *cache) { (void)cache; RELEASE_ALL_CALLS++; return(RELEASE_ALL_STATUS); }
SparkStatus SparkKvLaneTransactionsReset(SparkKvLaneTransactions *transactions) { (void)transactions; RELEASE_ALL_CALLS++; return(RELEASE_ALL_STATUS); }
SparkStatus SparkKvLaneTransactionsAdmit(SparkKvLaneTransactions *transactions,const SparkModelDriverAdmissionRequest *request) { (void)transactions; (void)request; assert(0); return(SPARK_STATUS_INTERNAL_ERROR); }
SparkStatus SparkKvPageCacheReleaseLane(SparkKvPageCache *cache,uint32_t slot,uint64_t sequence) { (void)cache; (void)slot; (void)sequence; assert(0); return(SPARK_STATUS_INTERNAL_ERROR); }
SparkStatus SparkKvPageCachePrepareLane(SparkKvPageCache *cache,const SparkModelDriverCacheLane *lane,uint32_t *pages,uint32_t capacity,uint32_t *count) { (void)cache; (void)lane; (void)pages; (void)capacity; (void)count; assert(0); return(SPARK_STATUS_INTERNAL_ERROR); }
SparkStatus SparkKvPageCacheBeginLaneTransaction(SparkKvPageCache *cache,const SparkModelDriverCacheLane *lane,uint32_t *page,uint32_t *flags) { (void)cache; (void)lane; (void)page; (void)flags; assert(0); return(SPARK_STATUS_INTERNAL_ERROR); }
SparkStatus SparkKvPageCacheRollbackLaneTransaction(SparkKvPageCache *cache,const SparkModelDriverCacheLane *lane,uint32_t flags) { (void)cache; (void)lane; (void)flags; assert(0); return(SPARK_STATUS_INTERNAL_ERROR); }
cudaError_t cudaStreamSynchronize(cudaStream_t stream) { (void)stream; SYNC_CALLS++; return(SYNC_STATUS); }
uint32_t SparkKvPageCacheSavePending(const SparkKvPageCache *cache) { (void)cache; return(0u); }
uint32_t SparkKvPageCachePrefixReady(const SparkKvPageCache *cache,const SparkModelDriverCacheIdentity *identity,uint32_t token_count) { (void)cache; (void)identity; (void)token_count; assert(0); return(0u); }
SparkStatus SparkKvPageStoreWaitForTransfers(SparkKvPageStore *store) { (void)store; assert(0); return(SPARK_STATUS_INTERNAL_ERROR); }
SparkStatus SparkKvPageStoreReadback(SparkKvPageStore *store,uint32_t logical_page_index,uint64_t generation,uintptr_t destination,uint64_t bytes) { (void)store; (void)logical_page_index; (void)generation; (void)destination; (void)bytes; assert(0); return(SPARK_STATUS_INTERNAL_ERROR); }
cudaError_t cudaGetLastError(void) { return(cudaSuccess); }
const char *cudaGetErrorString(cudaError_t error) { (void)error; return("stub"); }
const char *cudaGetErrorName(cudaError_t error) { (void)error; return("stub"); }

static MODULE_STATE state;
#define LANES MODULE_LANES(state)

static void Bind(void)
{
	uint32_t lane;
	for (lane=0u; lane<state.resident_sequence_capacity; lane++)
	{
		atomic_store(&LANES.lane_bound[lane],1u);
		atomic_store(&LANES.lane_sequence_ids[lane],100u + lane);
		atomic_store(&LANES.lane_next_positions[lane],7u + lane);
#ifdef SPARK_STAGE_KV_REGION_PAGE_MAJOR
		atomic_store(&state.kv.lane_rewind_floors[lane],3u + lane);
		atomic_store(&state.kv.lane_rewind_ceilings[lane],7u + lane);
		atomic_store(&state.kv.lane_pending_floors[lane],5u + lane);
#endif
	}
	RELEASE_ALL_CALLS = 0u;
	SYNC_CALLS = 0u;
	RELEASE_ALL_STATUS = SPARK_STATUS_OK;
	SYNC_STATUS = cudaSuccess;
}

#ifdef SPARK_STAGE_KV_REGION_PAGE_MAJOR
static void BindingSetup(void)
{
	state.kv.module_tag = MODULE_NAME;
	state.kv.resident_sequence_capacity = state.resident_sequence_capacity;
	state.kv.pipeline_slot_count = state.pipeline_slot_count;
	state.kv.pages_per_sequence = 1u;
	state.kv.lane_bound = (atomic_uchar *)calloc(state.resident_sequence_capacity,sizeof(*state.kv.lane_bound));
	state.kv.lane_sequence_ids = (atomic_ullong *)calloc(state.resident_sequence_capacity,sizeof(*state.kv.lane_sequence_ids));
	state.kv.lane_next_positions = (atomic_ullong *)calloc(state.resident_sequence_capacity,sizeof(*state.kv.lane_next_positions));
	state.kv.lane_rewind_floors = (atomic_ullong *)calloc(state.resident_sequence_capacity,sizeof(*state.kv.lane_rewind_floors));
	state.kv.lane_rewind_ceilings = (atomic_ullong *)calloc(state.resident_sequence_capacity,sizeof(*state.kv.lane_rewind_ceilings));
	state.kv.lane_pending_floors = (atomic_ullong *)calloc(state.resident_sequence_capacity,sizeof(*state.kv.lane_pending_floors));
	state.kv.page_table_shadow = (uint32_t *)calloc(state.resident_sequence_capacity,sizeof(*state.kv.page_table_shadow));
	assert(state.kv.lane_bound != 0 && state.kv.lane_sequence_ids != 0 && state.kv.lane_next_positions != 0 && state.kv.page_table_shadow != 0);
	assert(state.kv.lane_rewind_floors != 0 && state.kv.lane_rewind_ceilings != 0 && state.kv.lane_pending_floors != 0);
	assert(pthread_mutex_init(&state.kv.mutex,0) == 0);
	state.kv.mutex_initialized = 1u;
}
#endif

static uint32_t Bound(void)
{
	uint32_t lane,count = 0u;
	for (lane=0u; lane<state.resident_sequence_capacity; lane++)
	{
		uint32_t bound = atomic_load(&LANES.lane_bound[lane]) != 0u || atomic_load(&LANES.lane_sequence_ids[lane]) != 0u || atomic_load(&LANES.lane_next_positions[lane]) != 0u ? 1u : 0u;
#ifdef SPARK_STAGE_KV_REGION_PAGE_MAJOR
		bound |= atomic_load(&state.kv.lane_rewind_floors[lane]) != 0u || atomic_load(&state.kv.lane_rewind_ceilings[lane]) != 0u || atomic_load(&state.kv.lane_pending_floors[lane]) != UINT64_MAX ? 1u : 0u;
#endif
		count += bound;
	}
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
	return(MODULE_ADMIT(&state,request,decision));
}

static void TestResetClearsEveryLane(void)
{
	SparkModelDriverAdmissionRequest request = ResetRequest(1u);
	SparkModelDriverAdmissionDecision decision;
	Bind();
	assert(Admit(&request,&decision) == SPARK_STATUS_OK);
	assert(decision.accepted == 1u && decision.rejection_reason == SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED);
	assert(SYNC_CALLS == MODULE_RESET_SYNCS && RELEASE_ALL_CALLS == 1u && Bound() == 0u && Claimed() == 0u);
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
	MODULE_SETUP();
	TestResetClearsEveryLane();
	TestResetWaitsForInflightSlot();
	TestResetFailures();
	printf("%s reset: ok\n",MODULE_NAME);
	return(0);
}
'''


MODULES = {
    "glm52": {
        "source": "modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_module.c",
        "state": "SparkGlm52ModuleState", "admit": "SparkGlm52ModuleAdmit",
        "includes": ["model-families/glm52/include", "modules/glm52_resident_decode_stage/include", "modules/glm52_resident_decode_stage/source"],
        "defines": ["-DGLM_EXPERT_WEIGHT_CODEC=5", '-DGLM_EXPERT_CODEC_NAME="fp8"', '-DGLM_MODEL_DESCRIPTION_SHA256="fixture"',
                    "-include", "model-families/glm52/include/sparkpipe/spark_glm52_model.h"],
        "identity": ("GLM_MODEL_REVISION", "GLM_CONTRACT_SHA256", ["tools/glm52_model_contract.py", "--print-build-identity", "fp8"]),
        "lanes": "(s).kv",
        "setup": "BindingSetup()",
        "reset_syncs": 2,
        "sources": ["runtime/stage_kv_binding.c"],
    },
    "ling": {
        "source": "modules/ling_resident_decode_stage/source/spark_ling_resident_decode_stage_module.c",
        "state": "SparkLingModuleState", "admit": "SparkLingResidentDecodeStageAdmit",
        "includes": ["model-families/ling/include", "modules/ling_resident_decode_stage/include", "modules/ling_resident_decode_stage/source"],
        "defines": ["-DLING_EXPERT_WEIGHT_CODEC=1u", '-DLING_EXPERT_CODEC_NAME="bf16"', '-DLING_MODEL_REVISION="fixture"', '-DLING_CONTRACT_SHA256="fixture"'],
        "identity": None,
        "lanes": "(s)",
        "setup": "(void)0",
        "reset_syncs": 1,
        "sources": [],
    },
}


def run_module(name, module, directory):
    source, binary = Path(directory) / f"{name}.c", Path(directory) / name
    source.write_text(HARNESS)
    defines = list(module["defines"])
    if module["identity"] is not None:
        revision, contract, command = module["identity"]
        identity = subprocess.check_output([sys.executable, *command], cwd=ROOT, text=True).split()
        defines += [f'-D{revision}="{identity[0]}"', f'-D{contract}="{identity[1]}"']
    includes = [".", "include", "src", "tests/cuda_stub", "model-families/common/include", *module["includes"]]
    subprocess.run(["cc", "-std=c11", "-D_GNU_SOURCE", "-O1", "-g", "-ffunction-sections", "-fdata-sections",
                    "-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections",
                    *["-I" + p for p in includes], "-DSPARK_BATCH_BUCKET=16u", *defines,
                    f'-DMODULE_SOURCE="{module["source"]}"', f'-DMODULE_STATE={module["state"]}', f'-DMODULE_ADMIT={module["admit"]}',
                    f'-DMODULE_NAME="{name}"', f'-DMODULE_LANES(s)={module["lanes"]}', f'-DMODULE_SETUP()={module["setup"]}', f'-DMODULE_RESET_SYNCS={module["reset_syncs"]}u',
                    str(source), "runtime/stage_module_common.c", *module["sources"], "src/spark_status.c", "-o", str(binary), "-pthread"],
                   cwd=ROOT, check=True)
    subprocess.run([str(binary)], check=True)


def main():
    with tempfile.TemporaryDirectory() as directory:
        for name, module in MODULES.items():
            run_module(name, module, directory)
    print("PASS module page-cache reset (glm52, ling): client reset accepted, every lane unbound with its rewind window and pending verify floor cleared, waits for in-flight slots and lanes, stream and release failures stay loud")

if __name__ == "__main__":
    main()
