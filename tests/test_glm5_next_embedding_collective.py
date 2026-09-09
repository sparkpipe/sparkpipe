import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_module.c'
PREFIX = r'''
#include <stdio.h>
#include <string.h>
#include "sparkpipe/spark_tp_device_collective.h"
typedef struct
{
    uint32_t tp_degree,tp_collective_disabled,mtp_active;
    uint32_t tp_device_collective_initialized,tp_device_collective_hc_initialized;
    SparkTpDeviceCollective tp_device_collective,tp_device_collective_hc;
} SparkGlm5NextModuleState;
typedef struct { void *stream; } TestSlot;
typedef struct { uint32_t active_sequence_count; } TestBatch;
typedef struct
{
    SparkGlm5NextModuleState *state;
    TestSlot *slot;
    TestBatch *batch;
    uint32_t tp_hc_op_index,tp_op_index,slot_index,wave_rows;
} SparkGlm5NextTpChain;
static uint32_t advanced,enqueued;
static SparkStatus enqueue_status;
static SparkTpDeviceCollective *observed_collective;
static SparkTpDeviceCollectiveSubmission observed;
static void SparkGlm5NextTpChainAdvance(SparkGlm5NextTpChain *chain,SparkStatus status)
{
    (void)chain;
    (void)status;
    advanced++;
}
static void SparkGlm5NextModuleTpCompletion(void *context,const SparkTpDeviceCollectiveCompletion *completion)
{
    SparkGlm5NextTpChainAdvance(context,completion->status);
}
static SparkStatus SparkGlm5NextChainOrdinal(SparkGlm5NextTpChain *chain,uint32_t wide,uint32_t operation,uint64_t *ordinal)
{
    (void)chain;
    *ordinal = 100u + wide + operation;
    return(SPARK_STATUS_OK);
}
SparkStatus SparkTpDeviceCollectiveEnqueue(SparkTpDeviceCollective *collective,const SparkTpDeviceCollectiveSubmission *submission,uint32_t operation)
{
    if ( operation != SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16 )
        return(SPARK_STATUS_INVALID_ARGUMENT);
    observed_collective = collective;
    observed = *submission;
    enqueued++;
    return(enqueue_status);
}
'''
SUFFIX = r'''
int main(void)
{
    SparkGlm5NextModuleState state = {0};
    TestSlot slot = {0};
    TestBatch batch = {0};
    SparkGlm5NextTpChain chain = {0};
    SparkTpDeviceCollectiveCompletion completion = {0};
    uint16_t hidden[16] = {0};
    uint32_t rows;
    state.tp_degree = 16u;
    state.tp_device_collective_initialized = 1u;
    state.tp_device_collective_hc_initialized = 1u;
    chain.state = &state;
    chain.slot = &slot;
    chain.batch = &batch;
    slot.stream = hidden;
    for (rows=1u; rows<=3u; rows+=2u)
    {
        chain.wave_rows = rows;
        batch.active_sequence_count = 3u;
        advanced = enqueued = 0u;
        chain.tp_hc_op_index = 0u;
        if ( SparkGlm5NextModuleReduceHiddenWide(&chain,hidden,1u) != SPARK_STATUS_OK )
            return(1);
        if ( enqueued != 1u || advanced != 0u || chain.tp_hc_op_index != 1u )
            return(2);
        if ( observed_collective != &state.tp_device_collective_hc || observed.active_sequence_count != rows || observed.local_device != hidden || observed.full_device != hidden || observed.cuda_stream != slot.stream )
            return(3);
        if ( observed.completion_function == 0 || observed.completion_context != &chain )
            return(4);
        if ( observed.logical_sequence_count != 3u )
            return(8);
        observed.completion_function(observed.completion_context,&completion);
        if ( advanced != 1u )
            return(5);
    }
    state.tp_device_collective_hc_initialized = 0u;
    if ( SparkGlm5NextModuleReduceHiddenWide(&chain,hidden,1u) != SPARK_STATUS_INTERNAL_ERROR )
        return(6);
    state.tp_device_collective_hc_initialized = 1u;
    enqueue_status = SPARK_STATUS_BUSY;
    chain.tp_hc_op_index = advanced = 0u;
    if ( SparkGlm5NextModuleReduceHiddenWide(&chain,hidden,1u) != SPARK_STATUS_BUSY || chain.tp_hc_op_index != 0u || advanced != 0u )
        return(7);
    puts("PASS required embedding collective, completion ordering and enqueue failure");
    return(0);
}
'''

def main():
    source = SOURCE.read_text()
    signature = 'static SparkStatus SparkGlm5NextModuleReduceHiddenWide('
    start = source.index(signature)
    while source.find(';', start) < source.find('{', start):
        start = source.index(signature, start + len(signature))
    opening = source.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    with tempfile.TemporaryDirectory() as directory:
        fixture = pathlib.Path(directory) / 'embedding.c'
        executable = pathlib.Path(directory) / 'embedding'
        fixture.write_text(PREFIX + source[start:end] + SUFFIX)
        subprocess.run(['cc', '-std=c11', '-I', str(ROOT / 'include'), str(fixture), '-o', str(executable)], check=True)
        subprocess.run([str(executable)], check=True, timeout=10)

if __name__ == '__main__':
    main()
