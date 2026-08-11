#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_tp_device_collective.h"

#ifndef SPARK_TEST_TP_DEVICE_COLLECTIVE_MODULE_PATH
#define SPARK_TEST_TP_DEVICE_COLLECTIVE_MODULE_PATH \
    "build/test_modules/libtp_device_collective_module.so"
#endif

int main(void)
{
    SparkTpDeviceCollectiveConfig configuration;
    SparkTpDeviceCollective collective;
    SparkTpDeviceCollective host_collective;
    uint8_t send_buffer[256];
    uint8_t receive_buffer[256];
    const char *hosts[4] = {"rank0","rank1","rank2","rank3"};
    uint32_t step;

    memset(&configuration,0,sizeof(configuration));
    configuration.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    configuration.tp_degree = 4u;
    configuration.tp_rank = 1u;
    configuration.local_hidden_dimension = 4u;
    configuration.max_active_sequence_count = 8u;
    configuration.connect_timeout_milli = 100u;
    configuration.operation_timeout_milli = 100u;
    configuration.control_port_base = 60000u;
    configuration.collective_identifier = 1u;
    configuration.transport_module_path =
        SPARK_TEST_TP_DEVICE_COLLECTIVE_MODULE_PATH;
    configuration.local_host = hosts[1];
    memcpy(configuration.rank_hosts,hosts,sizeof(hosts));
    {
        SparkStatus status = SparkTpDeviceCollectiveCreate(
            &configuration,&collective);
        assert(status == SPARK_STATUS_OK);
    }
    memset(send_buffer,0xaa,sizeof(send_buffer));
    memset(receive_buffer,0,sizeof(receive_buffer));
    for (step=0u; step<2u; ++step)
    {
        assert(SparkTpDeviceCollectiveExchangeBf16(
            &collective,send_buffer,receive_buffer,2u,4u << step,step,
            (void *)1) == SPARK_STATUS_OK);
    }
    assert(collective.next_operation_sequence == 3u);
    assert(SparkTpDeviceCollectiveExchangeBf16(
        &collective,send_buffer,receive_buffer,2u,2u,0u,(void *)1) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    SparkTpDeviceCollectiveDestroy(&collective);
    assert(setenv("SPARK_TEST_TP_DEVICE_COLLECTIVE_HOST_MODE","1",1) == 0);
    {
        SparkStatus status = SparkTpDeviceCollectiveCreate(
            &configuration,&host_collective);
        assert(status == SPARK_STATUS_OK);
    }
    assert(host_collective.memory_mode ==
        SPARK_TP_DEVICE_COLLECTIVE_MEMORY_MODE_MAPPED_HOST);
    assert(SparkTpDeviceCollectiveExchangeBf16(
        &host_collective,send_buffer,receive_buffer,2u,4u,0u,
        (void *)1) == SPARK_STATUS_OK);
    SparkTpDeviceCollectiveDestroy(&host_collective);
    assert(unsetenv("SPARK_TEST_TP_DEVICE_COLLECTIVE_HOST_MODE") == 0);
    return 0;
}
