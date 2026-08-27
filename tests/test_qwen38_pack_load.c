#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_hidden_transport.h"
#include "sparkpipe/spark_qwen38_max_resident_decode_stage_firmware.h"

extern SparkStatus SparkQwen38MaxResidentDecodeStageInitialize(
    const SparkFirmwareModuleConfiguration *configuration,
    const SparkFirmwareModuleHostServices *host_services,
    void **module_state);
extern void SparkQwen38MaxResidentDecodeStageDestroy(void *module_state);

int main(int argc, char **argv)
{
    SparkFirmwareModuleConfiguration configuration;
    SparkFirmwareModuleHostServices services;
    void *state;
    SparkStatus status;
    if ( argc != 2 )
    {
        fprintf(stderr, "usage: test_qwen38_pack_load PACK\n");
        return 2;
    }
    setenv("SPARK_QWEN38_MAX_ALLOW_UNQUALIFIED_EXECUTION","1",1);
    setenv("SPARK_QWEN38_MAX_STAGE_COUNT","4",1);
    setenv("SPARK_QWEN38_MAX_STAGE_INDEX","1",1);
    setenv("SPARK_QWEN38_MAX_STAGE_FIRST_LAYER","1",1);
    setenv("SPARK_QWEN38_MAX_STAGE_LAYER_COUNT","1",1);
    setenv("SPARK_QWEN38_MAX_STAGE_MAX_ACTIVE_SEQUENCES","1",1);
    setenv("SPARK_QWEN38_MAX_STAGE_PIPELINE_SLOTS","1",1);
    setenv("SPARK_QWEN38_MAX_STAGE_KV_BLOCKS","8",1);
    setenv("SPARK_QWEN38_MAX_STAGE_PACK_PATH",argv[1],1);
    memset(&configuration,0,sizeof(configuration));
    configuration.abi_version = SPARK_FIRMWARE_MODULE_ABI_VERSION;
    configuration.descriptor_bytes = sizeof(configuration);
    configuration.model_id = "qwen38-load-test";
    configuration.model_revision = "test";
    configuration.stage_name = "qwen38";
    configuration.program_name = "resident_decode";
    configuration.operation_name = "initialize";
    memset(&services,0,sizeof(services));
    services.abi_version = SPARK_FIRMWARE_MODULE_HOST_SERVICES_ABI_VERSION;
    services.descriptor_bytes = sizeof(services);
    state = 0;
    status = SparkQwen38MaxResidentDecodeStageInitialize(&configuration,&services,&state);
    fprintf(stderr,"initialize status=%d state=%p\n",(int)status,state);
    if ( status != SPARK_STATUS_OK )
        return 1;
    SparkQwen38MaxResidentDecodeStageDestroy(state);
    fprintf(stderr,"destroy ok\n");
    return 0;
}
