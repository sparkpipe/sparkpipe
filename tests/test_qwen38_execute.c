#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_hidden_transport.h"
#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_qwen38_max_resident_decode_stage_firmware.h"

extern SparkStatus SparkQwen38ResidentDecodeStageInitialize(
    const SparkFirmwareModuleConfiguration *configuration,
    const SparkFirmwareModuleHostServices *host_services,
    void **module_state);
extern void SparkQwen38ResidentDecodeStageDestroy(void *module_state);
extern SparkStatus SparkQwen38ResidentDecodeStageExecute(void *module_state, SparkModelDriverFrame *frame);

int main(int argc, char **argv)
{
    SparkFirmwareModuleConfiguration configuration;
    SparkFirmwareModuleHostServices services;
    SparkModelDriverBuffer buffers[2];
    SparkModelDriverFrame frame;
    uint32_t input_tokens[1] = { 123u };
    uint32_t output_tokens[1] = { 0xdeadbeefu };
    void *state;
    SparkStatus status;
    if ( argc != 2 )
    {
        fprintf(stderr, "usage: test_qwen38_execute PACK\n");
        return 2;
    }
    setenv("SPARK_QWEN38_ALLOW_UNQUALIFIED_EXECUTION","1",1);
    /* Slice geometry defaults to the 1-layer GDN smoke; the environment can
     * override for other slices (e.g. a GDN+attention pack). */
    setenv("SPARK_QWEN38_STAGE_COUNT",getenv("TEST_QWEN38_STAGE_COUNT") ? getenv("TEST_QWEN38_STAGE_COUNT") : "4",1);
    setenv("SPARK_QWEN38_STAGE_INDEX",getenv("TEST_QWEN38_STAGE_INDEX") ? getenv("TEST_QWEN38_STAGE_INDEX") : "1",1);
    setenv("SPARK_QWEN38_STAGE_FIRST_LAYER",getenv("TEST_QWEN38_FIRST_LAYER") ? getenv("TEST_QWEN38_FIRST_LAYER") : "1",1);
    setenv("SPARK_QWEN38_STAGE_LAYER_COUNT",getenv("TEST_QWEN38_LAYER_COUNT") ? getenv("TEST_QWEN38_LAYER_COUNT") : "1",1);
    setenv("SPARK_QWEN38_STAGE_MAX_ACTIVE_SEQUENCES","1",1);
    setenv("SPARK_QWEN38_STAGE_PIPELINE_SLOTS","1",1);
    setenv("SPARK_QWEN38_STAGE_KV_BLOCKS","8",1);
    setenv("SPARK_QWEN38_STAGE_PACK_PATH",argv[1],1);
    memset(&configuration,0,sizeof(configuration));
    configuration.abi_version = SPARK_FIRMWARE_MODULE_ABI_VERSION;
    configuration.descriptor_bytes = sizeof(configuration);
    configuration.model_id = "qwen38-execute-test";
    configuration.model_revision = "test";
    configuration.stage_name = "qwen38";
    configuration.program_name = "resident_decode";
    configuration.operation_name = "initialize";
    memset(&services,0,sizeof(services));
    services.abi_version = SPARK_FIRMWARE_MODULE_HOST_SERVICES_ABI_VERSION;
    services.descriptor_bytes = sizeof(services);
    state = 0;
    status = SparkQwen38ResidentDecodeStageInitialize(&configuration,&services,&state);
    fprintf(stderr,"initialize status=%d state=%p\n",(int)status,state);
    if ( status != SPARK_STATUS_OK )
        return 1;
    memset(&frame,0,sizeof(frame));
    frame.active_slot_count = 1u;
    frame.new_token_count = 1u;
    frame.sequence_position = 0u;
    memset(buffers,0,sizeof(buffers));
    buffers[0].address = input_tokens;
    buffers[0].bytes = sizeof(input_tokens);
    buffers[1].address = output_tokens;
    buffers[1].bytes = sizeof(output_tokens);
    frame.buffers = buffers;
    frame.buffer_count = 2u;
    status = SparkQwen38ResidentDecodeStageExecute(state,&frame);
    fprintf(stderr,"execute[0] status=%d output_token=%u\n",(int)status,output_tokens[0]);
    if ( status == SPARK_STATUS_OK )
    {
        /* Second decode step at position 1: same slot, exercises the
         * recurrent state path with the carried conv tail and GDN state
         * (row_cold is 0, so the recurrence accumulates). */
        frame.sequence_position = 1u;
        output_tokens[0] = 0xdeadbeefu;
        status = SparkQwen38ResidentDecodeStageExecute(state,&frame);
        fprintf(stderr,"execute[1] status=%d output_token=%u\n",(int)status,output_tokens[0]);
    }
    SparkQwen38ResidentDecodeStageDestroy(state);
    fprintf(stderr,"destroy ok\n");
    return status == SPARK_STATUS_OK ? 0 : 1;
}
