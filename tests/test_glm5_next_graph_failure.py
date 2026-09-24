from pathlib import Path
import os
import re
import subprocess
import tempfile

from family_source import read_source

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_module.c'
CUDA = ROOT / 'modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_cuda.cu'
MESH = ROOT / 'model-families/common/include/sparkpipe/spark_tp_mesh_kernels.cuh'


def function(source, name):
    matches = re.finditer(r'(?m)^[^\n;]*\b' + re.escape(name) + r'\s*\(', source)
    for match in matches:
        start = match.start()
        opening = source.index('{', match.end())
        if ';' in source[match.end():opening]:
            continue
        depth = 1
        end = opening + 1
        while depth:
            depth += (source[end] == '{') - (source[end] == '}')
            end += 1
        return source[start:end]
    raise AssertionError(name)


KERNEL_PREFIX = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#define __global__
struct HostDim { unsigned x; } threadIdx={0},blockIdx={0},blockDim={1};
static unsigned long long host_timer;
static unsigned long long SparkGlm5NextGlobalTimerNs(void) { return ++host_timer; }
static uint64_t SparkGlm5NextLdcvU64(const volatile void *p) { return *(const volatile uint64_t *)p; }
static unsigned long long atomicExch(unsigned long long *p,unsigned long long x)
{
    unsigned long long old=*p;
    *p=x;
    return old;
}
'''
KERNEL_MAIN = r'''
int main(void)
{
    uint64_t band[16]={0};
    unsigned long long sequence=(1ull<<32)|1ull;
    unsigned long long error=0,diag=0,arrivals[256]={0},maxloc=0;
    uint32_t token=123;
    SparkGlm5NextMeshWaitKernel(band,32,&sequence,2,0,2,&error,1,&diag,0,0,arrivals);
    assert(error==sequence && arrivals[1]==0);
    SparkGlm5NextMeshGuardKernel(&error,&maxloc);
    SparkGlm5NextHeadMaxlocUnpackKernel((const uint64_t *)&maxloc,&token,1);
    assert(error==sequence && token==UINT32_MAX);
    SparkGlm5NextMeshWaitKernel(band,32,&sequence,2,0,2,&error,1,&diag,0,0,arrivals);
    assert(error==sequence && arrivals[1]==0);
    error=0;
    band[11]=sequence;
    maxloc=(0x80000000ull<<32)|(UINT32_MAX-123u);
    SparkGlm5NextMeshWaitKernel(band,32,&sequence,2,0,2,&error,1,&diag,0,0,arrivals);
    SparkGlm5NextMeshGuardKernel(&error,&maxloc);
    SparkGlm5NextHeadMaxlocUnpackKernel((const uint64_t *)&maxloc,&token,1);
    assert(error==0 && arrivals[1]!=0 && token==123);
    uint64_t cancel=2;
    unsigned long long expected=1;
    arrivals[1]=0;
    SparkGlm5NextMeshWaitKernel(band,32,&sequence,2,0,2,&error,1,&diag,&cancel,&expected,arrivals);
    SparkGlm5NextMeshGuardKernel(&error,&maxloc);
    SparkGlm5NextHeadMaxlocUnpackKernel((const uint64_t *)&maxloc,&token,1);
    assert(error==(SPARK_TP_MESH_ERROR_CANCELLED|sequence));
    assert(arrivals[1]==0 && token==UINT32_MAX);
    puts("PASS production mesh timeout/cancel remain errors through head token unpack");
}
'''
LIFETIME_PREFIX = r'''
#include <assert.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_glm5_next_model.h"
typedef struct SparkWeightdClient { uint32_t alive,identity; } SparkWeightdClient;
typedef struct { SparkWeightdClient *client; } SparkWeightdLazyPack;
typedef struct {
    atomic_uint terminal_status;
    SparkWeightdClient *lane_client;
    SparkWeightdLazyPack *lazy_pack;
} SparkGlm5NextModuleState;
typedef struct { uint32_t *host_output_token_ids; } SparkGlm5NextExecutionSlot;
typedef struct {
    SparkGlm5NextModuleState *state;
    struct { SparkStatus status; } completion;
    uint32_t *output_token_destination;
    uint32_t burst_token_count,row_count;
} SparkGlm5NextAsyncCompletion;
typedef struct { SparkGlm5NextModuleState *state; uint32_t active; } SparkGlm5NextTpChain;
static uint32_t failed,advanced;
static SparkStatus callback_status;
static uint32_t SparkWeightdClientAlive(const SparkWeightdClient *client) { return client->alive; }
static void SparkGlm5NextTpChainFail(SparkGlm5NextTpChain *chain,SparkStatus status)
{
    assert(chain->active);
    failed++;
    callback_status=status;
}
static void SparkGlm5NextTpChainAdvance(SparkGlm5NextTpChain *chain,SparkStatus status)
{
    assert(chain->active && status==SPARK_STATUS_OK);
    advanced++;
}
'''
LIFETIME_MAIN = r'''
int main(void)
{
    SparkWeightdClient lane={1,123},weights={1,456};
    SparkWeightdLazyPack pack={&weights};
    SparkGlm5NextModuleState state={0};
    state.lane_client=&lane;
    state.lazy_pack=&pack;
    atomic_init(&state.terminal_status,SPARK_STATUS_OK);
    SparkGlm5NextTpChain chain={&state,1};
    SparkTpDeviceCollectiveCompletion completion={0};
    SparkGlm5NextModuleTpCompletion(&chain,&completion);
    assert(advanced==1 && failed==0);
    lane.alive=0;
    SparkGlm5NextModuleTpCompletion(&chain,&completion);
    assert(advanced==1 && failed==1 && callback_status==SPARK_STATUS_IO_ERROR);
    assert(lane.identity==123 && weights.identity==456);
    lane.alive=1;
    assert(SparkGlm5NextWeightdHealth(&state)==SPARK_STATUS_IO_ERROR);
    atomic_store(&state.terminal_status,SPARK_STATUS_OK);
    weights.alive=0;
    assert(SparkGlm5NextWeightdHealth(&state)==SPARK_STATUS_IO_ERROR);
    weights.alive=1;
    atomic_store(&state.terminal_status,SPARK_STATUS_OK);
    uint32_t tokens[2]={0,UINT32_MAX},destination[2]={99,99};
    SparkGlm5NextExecutionSlot slot={tokens};
    SparkGlm5NextAsyncCompletion async={0};
    async.state=&state;
    async.output_token_destination=destination;
    async.row_count=2;
    assert(SparkGlm5NextCompletionStatus(&async,&slot)==SPARK_STATUS_VALIDATION_FAILED);
    assert(destination[0]==99 && destination[1]==99);
    atomic_store(&state.terminal_status,SPARK_STATUS_OK);
    tokens[1]=123;
    assert(SparkGlm5NextCompletionStatus(&async,&slot)==SPARK_STATUS_OK);
    async.completion.status=SPARK_STATUS_IO_ERROR;
    assert(SparkGlm5NextCompletionStatus(&async,&slot)==SPARK_STATUS_IO_ERROR);
    puts("PASS production callback daemon loss fences generation; invalid output fails before publication");
}
'''


def run(source, compiler, suffix, options):
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory) / ('fixture' + suffix)
        executable = Path(directory) / 'fixture'
        fixture.write_text(source)
        command = [compiler, '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined']
        command += options + [str(fixture), '-o', str(executable)]
        subprocess.run(command, check=True)
        subprocess.run([str(executable)], check=True, timeout=10)


def main():
    mesh = MESH.read_text()
    cuda = read_source(CUDA)
    module = read_source(MODULE)
    cancelled = re.search(r'(?m)^#define SPARK_TP_MESH_ERROR_CANCELLED .*$', mesh).group()
    kernels = [function(mesh, name) for name in ['SparkGlm5NextMeshWaitKernel', 'SparkGlm5NextMeshGuardKernel']]
    kernels.append(function(cuda, 'SparkGlm5NextHeadMaxlocUnpackKernel'))
    run(KERNEL_PREFIX + cancelled + '\n' + '\n'.join(kernels) + KERNEL_MAIN,
        os.environ.get('CXX', 'c++'), '.cc', ['-std=c++17'])
    names = ['SparkGlm5NextTerminalFailure', 'SparkGlm5NextWeightdHealth',
             'SparkGlm5NextModuleTpCompletion', 'SparkGlm5NextCompletionStatus']
    lifecycle = '\n'.join(function(module, name) for name in names)
    run(LIFETIME_PREFIX + lifecycle + LIFETIME_MAIN, os.environ.get('CC', 'cc'), '.c',
        ['-std=c11', '-I', str(ROOT / 'include'), '-I', str(ROOT / 'model-families/glm5_next/include')])


if __name__ == '__main__':
    main()
