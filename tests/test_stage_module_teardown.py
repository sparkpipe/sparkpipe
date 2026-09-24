#!/usr/bin/env python3
import os
import re
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = (
    ('qwen4_flash', 'Qwen4Flash', 'tp_collective_initialized', 'ple_prev_context_u32', True),
    ('qwen38_max', 'Qwen38Max', 'tp_collective_initialized', 't1_stage_hidden', True),
    ('gemma4', 'Gemma4', 'tp_collective_initialized', 'slots[0].host_row_lane_indices', False),
    ('muse_glimmer', 'MuseGlimmer', 'tp_collective_initialized', 'kv_logical_to_slot', False),
)
HARNESS = r'''
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "sparkpipe/spark_weightd_lazy_pack.h"
#include "sparkpipe/spark_stage_kv_client.h"
static void TestFree(void *pointer);
#define free TestFree
#include "modules/@FAMILY@_resident_decode_stage/source/spark_@FAMILY@_resident_decode_stage_module.c"
#include "runtime/stage_module_lifecycle.c"
#undef free
static Spark@PREFIX@ModuleState *owner;
static void *sentinel;
static uint32_t fail_collective,fail_lazy,fail_slots;
static uint32_t collectives,lazy_calls,unmaps,host_frees,state_frees,ledger_frees;
static void TestFree(void *pointer)
{
    if (pointer == 0)
        return;
    assert(owner->tp_device_collective.implementation == 0);
    @CHECK_LAZY@
    if (pointer == sentinel)
        host_frees++;
    if (pointer == owner)
    {
        assert(ledger_frees == 1u && host_frees == 1u);
        state_frees++;
    }
    free(pointer);
}
void SparkTpDeviceCollectiveDestroy(SparkTpDeviceCollective *collective)
{
    assert(collective == &owner->tp_device_collective);
    assert(collective->implementation != 0 && host_frees == 0u);
    collectives++;
    if (fail_collective != 0u)
        return;
    collective->implementation = 0;
}
SparkStatus SparkWeightdLazyPackDestroy(SparkWeightdLazyPack *pack)
{
    assert(owner->tp_device_collective.implementation == 0 && host_frees == 0u);
    @CHECK_PACK@
    lazy_calls++;
    if (fail_lazy != 0u)
        return SPARK_STATUS_IO_ERROR;
    unmaps++;
    free(pack);
    return SPARK_STATUS_OK;
}
SparkStatus SparkStageModuleWaitForSlots(const char *tag,const atomic_uint *states,uint32_t count,uint64_t timeout)
{
    (void)tag;(void)states;(void)timeout;
    assert(count != 0u);
    return fail_slots != 0u ? SPARK_STATUS_BUSY : SPARK_STATUS_OK;
}
void SparkStageModuleLedgerRelease(SparkStageModuleLedger *ledger)
{
    assert(ledger == &owner->ledger && host_frees == 1u);
    assert(owner->tp_device_collective.implementation == 0);
    @CHECK_LAZY@
    ledger_frees++;
}
void SparkStageKvClientClose(SparkStageKvClient *client) { (void)client; }
void SparkStageModuleStageTimingShutdown(SparkStageModuleStageTiming *timing) { (void)timing; }
cudaError_t cudaFree(void *pointer) { assert(pointer == 0);return cudaSuccess; }
static SparkStatus TestPrepare(void *state,const SparkFirmwareModuleConfiguration *configuration,const SparkFirmwareModuleHostServices *services)
{
    (void)configuration;(void)services;
    owner=state;
    sentinel=malloc(32u);assert(sentinel != 0);
    owner->@SENTINEL@=sentinel;
    owner->@FLAG@=1u;
    owner->tp_device_collective.implementation=(void *)(uintptr_t)1u;
    @SET_LAZY@
    return SPARK_STATUS_SCHEMA_ERROR;
}
int main(void)
{
    const SparkStageModuleLifecycleOps ops = {
        .state_bytes = sizeof(*owner),
        .state_prepare = TestPrepare,
        .describe = Spark@PREFIX@ModuleDescribe,
        .state_destroy = Spark@PREFIX@ModuleStateTeardown
    };
    for (uint32_t failures=0u; failures<16u; failures++)
    {
        collectives=lazy_calls=unmaps=host_frees=state_frees=ledger_frees=0u;
        fail_collective=failures&1u;fail_lazy=failures&2u;fail_slots=(failures&8u) == 0u ? failures&4u : 0u;
        owner=calloc(1u,sizeof(*owner));assert(owner != 0);
        owner->pipeline_slot_count=(failures&8u) == 0u ? 1u : 0u;
        sentinel=malloc(32u);assert(sentinel != 0);
        owner->@SENTINEL@=sentinel;
        owner->@FLAG@=1u;
        owner->tp_device_collective.implementation=(void *)(uintptr_t)1u;
        @SET_LAZY@
        if (fail_slots != 0u)
        {
            assert(SparkStageModuleLifecycleDestroy(owner,&ops) == SPARK_STATUS_BUSY);
            assert(collectives == 0u && host_frees == 0u && ledger_frees == 0u && state_frees == 0u);
            fail_slots=0u;
        }
        if (fail_collective != 0u)
        {
            assert(SparkStageModuleLifecycleDestroy(owner,&ops) == SPARK_STATUS_BUSY);
            assert(lazy_calls == 0u && host_frees == 0u && ledger_frees == 0u && state_frees == 0u);
            assert(owner->tp_device_collective.implementation != 0);
            fail_collective=0u;
        }
        @FAIL_LAZY@
        assert(SparkStageModuleLifecycleDestroy(owner,&ops) == SPARK_STATUS_OK);
        assert(collectives == 1u+(failures&1u) && host_frees == 1u && ledger_frees == 1u && state_frees == 1u);
        @CHECK_UNMAP@
    }
    collectives=lazy_calls=unmaps=host_frees=state_frees=ledger_frees=0u;
    fail_collective=fail_lazy=fail_slots=0u;
    SparkFirmwareModuleConfiguration configuration = {.abi_version=SPARK_FIRMWARE_MODULE_ABI_VERSION,.descriptor_bytes=sizeof(configuration)};
    SparkFirmwareModuleHostServices services = {.abi_version=SPARK_FIRMWARE_MODULE_HOST_SERVICES_ABI_VERSION,.descriptor_bytes=sizeof(services)};
    void *published=(void *)(uintptr_t)1u;
    assert(SparkStageModuleLifecycleInitialize(&configuration,&services,&published,&ops) == SPARK_STATUS_SCHEMA_ERROR);
    assert(published == 0 && collectives == 1u && host_frees == 1u && ledger_frees == 1u && state_frees == 1u);
    assert(SparkStageModuleLifecycleDestroy(0,&ops) == SPARK_STATUS_OK);
    puts("PASS @FAMILY@ actual teardown ownership and retry");
    return 0;
}
'''


GLM52 = r'''#include <assert.h>
#include "modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_module.c"
static uint32_t fail_release,records[2],released[2];
SparkStatus SparkWeightdMapRecordCompletion(SparkWeightdMap *map,uint64_t id,cudaStream_t stream)
{
    assert(map == (void *)(uintptr_t)1u && stream == (void *)(uintptr_t)2u && id>=10u && id<=11u);
    assert(records[id-10u] == 0u);
    records[id-10u]++;
    return SPARK_STATUS_OK;
}
SparkStatus SparkWeightdMapRelease(SparkWeightdMap *map,uint64_t id,uint64_t timeout)
{
    (void)timeout;
    assert(map == (void *)(uintptr_t)1u && id>=10u && id<=11u && records[id-10u] == 1u);
    if (fail_release != 0u)
        return SPARK_STATUS_IO_ERROR;
    assert(released[id-10u] == 0u);
    released[id-10u]++;
    return SPARK_STATUS_OK;
}
cudaError_t cudaEventSynchronize(cudaEvent_t event) { assert(event == (void *)(uintptr_t)3u);return cudaSuccess; }
cudaError_t cudaStreamSynchronize(cudaStream_t stream) { assert(stream == (void *)(uintptr_t)2u);return cudaSuccess; }
int main(void)
{
    SparkGlm52ModuleState state = {0};
    SparkGlm52ExecutionSlot slot = {0};
    SparkWeightdLazyPack pack = {0};
    SparkGlm52TpChain chain = {0},*recovered;
    state.lazy_pack=&pack;pack.map=(void *)(uintptr_t)1u;
    slot.stream=(void *)(uintptr_t)2u;slot.expert_done_event=(void *)(uintptr_t)3u;
    for (uint32_t mask=1u; mask<4u; mask++)
    {
        memset(&chain,0,sizeof(chain));memset(records,0,sizeof(records));memset(released,0,sizeof(released));
        chain.state=&state;chain.slot=&slot;
        chain.retired_lease=(mask&1u) != 0u ? 10u : 0u;chain.retired_begun=(mask&1u) != 0u;
        chain.expert_lease=(mask&2u) != 0u ? 11u : 0u;chain.expert_lease_begun=(mask&2u) != 0u;
        state.lazy_retained[0]=&chain;fail_release=1u;recovered=0;
        assert(SparkGlm52LazyRecoverLease(&state,0u,&recovered) == SPARK_STATUS_IO_ERROR);
        assert(state.lazy_retained[0] == &chain && recovered == 0);
        assert(released[0] == 0u && released[1] == 0u);
        fail_release=0u;
        assert(SparkGlm52LazyRecoverLease(&state,0u,&recovered) == SPARK_STATUS_OK);
        assert(state.lazy_retained[0] == 0 && recovered == &chain);
        assert(chain.retired_lease == 0u && chain.expert_lease == 0u);
        assert(records[0] == ((mask&1u) != 0u) && records[1] == ((mask&2u) != 0u));
        assert(released[0] == records[0] && released[1] == records[1]);
        assert(SparkGlm52LazyRecoverLease(&state,0u,&recovered) == SPARK_STATUS_NOT_FOUND && recovered == 0);
    }
    puts("PASS glm52 retained-only, active-only and combined expert lease retry");
    return 0;
}
'''



K3 = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "sparkpipe/spark_k3_weightd_include.h"
#include "sparkpipe/spark_error_site.h"
#include "inference/llms/kimi_k3/config.h"
#define LM_LAUNCH_OK 0
@STATE@
@FUNCTIONS@
static SparkK3RunnerState state;
static SparkWeightdLazyPack pack;
static K3LayerBuffers buffers;
static uint32_t offsets[K3_EXPERTS+1u],records,releases,acquires,routes,begins,syncs;
static uint64_t map_id;
static uint32_t map_phase;
static SparkStatus fail_record,fail_release,fail_acquire,fail_begin;
static cudaError_t fail_sync;
static unsigned char address[64];
static void Reset(void)
{
    memset(&state,0,sizeof(state));memset(&pack,0,sizeof(pack));memset(&buffers,0,sizeof(buffers));
    records=releases=acquires=routes=begins=syncs=0u;map_id=0u;map_phase=0u;
    fail_record=fail_release=fail_acquire=fail_begin=SPARK_STATUS_OK;fail_sync=cudaSuccess;
    pack.map=(SparkWeightdMap *)(uintptr_t)1u;state.lazy_pack=&pack;
    state.stream=(cudaStream_t)(uintptr_t)2u;state.group_offset_host=offsets;
    state.rows=3u;state.layer_w1_offset[1]=7u;state.layer_w2_offset[1]=23u;
    buffers.group_row_offset=offsets;
}
cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
    assert(stream == state.stream);syncs++;return fail_sync;
}
cudaError_t cudaMemcpy(void *destination,const void *source,size_t bytes,cudaMemcpyKind kind)
{
    assert(destination == offsets && source == offsets && bytes == sizeof(offsets) && kind == cudaMemcpyDeviceToHost);
    return cudaSuccess;
}
SparkStatus SparkWeightdRouteKeys(uint32_t layer,const uint32_t *source,uint32_t experts,uint32_t rows,SparkWeightdExpertKey *keys,uint32_t capacity,uint32_t *count)
{
    assert(map_id == 0u && source == offsets && experts == K3_EXPERTS && rows == 3u*K3_TOP_K);
    assert(layer == 1u && capacity == SPARK_WEIGHTD_LEASE_GROUPS_MAX);(void)keys;
    routes++;*count=2u;return SPARK_STATUS_OK;
}
SparkStatus SparkWeightdMapAcquire(SparkWeightdMap *map,const SparkWeightdExpertKey *keys,uint32_t count,uint64_t *id,uint64_t timeout)
{
    assert(map == pack.map && map_id == 0u && *id == 0u && count == 2u && timeout != 0u);(void)keys;
    acquires++;map_id=100u+acquires;map_phase=1u;*id=map_id;return fail_acquire;
}
SparkStatus SparkWeightdMapBeginUse(SparkWeightdMap *map,uint64_t id,void **out)
{
    assert(map == pack.map && id == map_id && map_phase == 1u);begins++;
    if (fail_begin != SPARK_STATUS_OK) return fail_begin;
    map_phase=2u;*out=address;return SPARK_STATUS_OK;
}
SparkStatus SparkWeightdMapRecordCompletion(SparkWeightdMap *map,uint64_t id,cudaStream_t stream)
{
    assert(map == pack.map && id == map_id && map_phase == 2u && stream == state.stream);records++;
    if (fail_record != SPARK_STATUS_OK) return fail_record;
    map_phase=3u;return SPARK_STATUS_OK;
}
SparkStatus SparkWeightdMapRelease(SparkWeightdMap *map,uint64_t id,uint64_t timeout)
{
    assert(map == pack.map && id == map_id && (map_phase == 1u || map_phase == 3u) && timeout != 0u);releases++;
    if (fail_release != SPARK_STATUS_OK) return fail_release;
    map_id=0u;map_phase=0u;return SPARK_STATUS_OK;
}
static void Acquire(void)
{
    assert(SparkK3RunnerLazyAcquire(&state,1u,&buffers) == LM_LAUNCH_OK);
    assert(state.lease_identifier == map_id && state.lease_phase == SPARK_K3_LEASE_BEGUN);
    assert(buffers.expert_w1_weight == address+7u && buffers.expert_w2_weight == address+23u);
}
static void Released(void)
{
    assert(state.lease_identifier == 0u && state.lease_phase == 0u && state.lease_address == 0 && map_id == 0u);
    uint32_t before=releases;SparkK3RunnerLazyRelease(&state,1u);assert(releases == before);
}
int main(void)
{
    const SparkStatus failures[]={SPARK_STATUS_BUSY,SPARK_STATUS_IO_ERROR};
    for (uint32_t i=0u;i<2u;i++)
    {
        Reset();Acquire();fail_release=failures[i];SparkK3RunnerLazyRelease(&state,1u);
        assert(state.lease_identifier == map_id && state.lease_address == address && state.lease_phase == SPARK_K3_LEASE_RECORDED);
        assert(records == 1u && releases == 1u);
        assert(SparkK3RunnerLazyAcquire(&state,1u,&buffers) == failures[i]);
        assert(acquires == 1u && routes == 1u && records == 1u && releases == 2u);
        fail_release=SPARK_STATUS_OK;Acquire();
        assert(acquires == 2u && records == 1u && releases == 3u);
        SparkK3RunnerLazyRelease(&state,1u);assert(records == 2u);Released();
    }
    Reset();Acquire();fail_record=SPARK_STATUS_IO_ERROR;SparkK3RunnerLazyRelease(&state,1u);
    assert(state.lease_identifier == map_id && state.lease_phase == SPARK_K3_LEASE_BEGUN && state.lease_address == address);
    assert(releases == 0u && records == 1u);
    fail_record=SPARK_STATUS_OK;SparkK3RunnerLazyRelease(&state,1u);assert(records == 2u && releases == 1u);Released();
    Reset();fail_begin=SPARK_STATUS_IO_ERROR;
    assert(SparkK3RunnerLazyAcquire(&state,1u,&buffers) == SPARK_STATUS_IO_ERROR);
    assert(state.lease_identifier == map_id && state.lease_phase == SPARK_K3_LEASE_ACQUIRED);
    fail_release=SPARK_STATUS_BUSY;SparkK3RunnerLazyRelease(&state,1u);
    assert(records == 0u && state.lease_identifier == map_id && state.lease_phase == SPARK_K3_LEASE_ACQUIRED);
    fail_release=SPARK_STATUS_OK;assert(SparkK3RunnerReleaseLease(&state) == SPARK_STATUS_OK);Released();
    Reset();fail_acquire=SPARK_STATUS_IO_ERROR;fail_release=SPARK_STATUS_IO_ERROR;
    assert(SparkK3RunnerLazyAcquire(&state,1u,&buffers) == SPARK_STATUS_IO_ERROR);
    assert(state.lease_identifier == map_id && state.lease_phase == SPARK_K3_LEASE_ACQUIRED && begins == 0u && records == 0u);
    fail_release=SPARK_STATUS_OK;assert(SparkK3RunnerReleaseLease(&state) == SPARK_STATUS_OK);Released();
    Reset();Acquire();fail_sync=cudaErrorLaunchFailure;
    assert(SparkK3RunnerLazyAcquire(&state,1u,&buffers) == SPARK_STATUS_IO_ERROR);
    assert(acquires == 1u && records == 0u && releases == 0u && state.lease_identifier == map_id);
    fail_sync=cudaSuccess;SparkK3RunnerLazyRelease(&state,1u);Released();
    puts("PASS k3 actual acquisition/release bodies retain phase and owner across failures");
    return 0;
}
'''


def k3_harness():
    source = (ROOT / 'modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_runner.cu').read_text()
    def block(text, marker):
        begin = text.index(marker)
        body = text.index('{', begin)
        depth = 1
        end = body + 1
        while depth:
            depth += (text[end] == '{') - (text[end] == '}')
            end += 1
        return text[begin:end]
    stray = source[source.index('#define K3_STRAY_BIT_INDEX('):source.index('static void SparkK3RunnerStrayAccount(')]
    functions = stray + '\n'.join(block(source, marker) for marker in (
        'static void SparkK3RunnerStrayAccount(',
        'static SparkStatus SparkK3RunnerReleaseLease(',
        'static int32_t SparkK3RunnerLazyAcquire(',
        'static void SparkK3RunnerLazyRelease('))
    state = source[source.index('enum\n{\n\tSPARK_K3_LEASE_ACQUIRED'):source.index('typedef struct SparkK3RunnerState\n{')]
    for name, text, fields in (
        ('SparkK3RunnerState', source, ('lazy_pack', 'lease_identifier', 'lease_phase', 'lease_address', 'group_offset_host', 'layer_w1_offset', 'layer_w2_offset', 'rows', 'stream', 'stray_head_bits', 'stray_seen_bits', 'stray_selections', 'stray_count')),
        ('K3LayerBuffers', (ROOT / 'inference/llms/kimi_k3/layer.cuh').read_text(), ('expert_w1_weight', 'expert_w2_weight', 'group_row_offset'))):
        body = block(text, 'struct ' + name + '\n{')
        declarations = []
        for field in fields:
            matches = [line for line in body.splitlines() if re.search(r'\b' + field + r'(?:\[|;)', line)]
            if len(matches) != 1:
                raise ValueError('ambiguous production field ' + name + '.' + field)
            declarations.append(matches[0])
        state += 'typedef struct {\n' + '\n'.join(declarations) + '\n} ' + name + ';\n'
    return K3.replace('@STATE@', state).replace('@FUNCTIONS@', functions)


def main():
    with tempfile.TemporaryDirectory(prefix='spark-teardown-') as directory:
        for family, prefix, flag, sentinel, lazy in (*FAMILIES, ('glm52', '', '', '', False), ('k3', '', '', '', False)):
            harness = HARNESS
            replacements = {
                'FAMILY': family, 'PREFIX': prefix, 'FLAG': flag, 'SENTINEL': sentinel,
                'CHECK_LAZY': 'assert(owner->lazy_pack == 0);' if lazy else '',
                'CHECK_PACK': 'assert(pack == owner->lazy_pack);' if lazy else '(void)pack;',
                'SET_LAZY': 'owner->lazy_pack=calloc(1u,sizeof(*owner->lazy_pack));assert(owner->lazy_pack != 0);' if lazy else '',
                'FAIL_LAZY': r'''if (fail_lazy != 0u) {
                    assert(SparkStageModuleLifecycleDestroy(owner,&ops) == SPARK_STATUS_IO_ERROR);
                    assert(owner->tp_device_collective.implementation == 0 && owner->lazy_pack != 0);
                    assert(host_frees == 0u && ledger_frees == 0u && state_frees == 0u);
                    fail_lazy=0u;
                }''' if lazy else '',
                'CHECK_UNMAP': 'assert(unmaps == 1u && lazy_calls == 1u+((failures&2u) != 0u));' if lazy else 'assert(unmaps == 0u && lazy_calls == 0u);',
            }
            for key, value in replacements.items():
                harness = harness.replace('@' + key + '@', value)
            source, binary = Path(directory) / (family + '.c'), Path(directory) / family
            source.write_text(k3_harness() if family == 'k3' else GLM52 if family == 'glm52' else harness)
            includes = ['.', 'include', 'tests/cuda_stub', 'model-families/common/include',
                        f'model-families/{family}/include', f'model-families/{family}/include/sparkpipe', f'modules/{family}_resident_decode_stage/include',
                        f'modules/{family}_resident_decode_stage/source']
            subprocess.run([os.environ.get('CC', 'cc'), '-std=c11', '-D_GNU_SOURCE', '-O1',
                            '-ffunction-sections', '-fdata-sections',
                            '-Wl,-dead_strip' if sys.platform == 'darwin' else '-Wl,--gc-sections',
                            *['-I' + path for path in includes], '-DSPARK_LLM_MTP_LAYER_COUNT=0u',
                            f'-D{family.upper()}_MODEL_REVISION="fixture"', '-DQWEN38_MODEL_REVISION="fixture"',
                            f'-D{family.upper()}_CONTRACT_SHA256="fixture"',
                            '-DGLM_EXPERT_WEIGHT_CODEC=5', '-DGLM_EXPERT_CODEC_NAME="fp8"', '-DGLM_CONTRACT_SHA256="fixture"',
                            *shlex.split(os.environ.get('SPARK_TEST_SANITIZER_FLAGS', '')),
                            str(source), '-o', str(binary)], cwd=ROOT, check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    main()
