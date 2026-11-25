#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_tp_mesh_kernels.cuh"
#include "inference/runner/mesh_fast_kernels.cuh"

static const SparkTpMeshLaunchers *g_registered_launchers = 0;
static uint32_t g_register_calls = 0u;

extern "C" void SparkTpDeviceCollectiveRegisterLaunchers(const SparkTpMeshLaunchers *launchers)
{
    g_registered_launchers = launchers;
    g_register_calls++;
}

int main(int argc, char **argv)
{
    const int expect_disabled = argc > 1 && strcmp(argv[1], "disabled") == 0 ? 1 : 0;
    const SparkTpMeshHardwareFunction ref_hw = reinterpret_cast<SparkTpMeshHardwareFunction>(SparkTpLaunchMeshHardware);
    const SparkTpMeshHardwareFunction fast_hw = reinterpret_cast<SparkTpMeshHardwareFunction>(SparkTpLaunchMeshHardwareFast);
    if ( g_registered_launchers == 0 )
    {
        fprintf(stderr, "FAIL tp_mesh_fast_disable_cuda: no launchers registered at startup\n");
        return 1;
    }
    if ( expect_disabled != 0 )
    {
        if ( g_registered_launchers->hardware != ref_hw || g_register_calls != 1u )
        {
            fprintf(stderr, "FAIL tp_mesh_fast_disable_cuda: expected reference launcher (calls=%u) when SPARK_MESH_FAST_DISABLE is set\n", g_register_calls);
            return 1;
        }
        SparkTpMeshFastLaunchersInstall();
        if ( g_registered_launchers->hardware != ref_hw || g_register_calls != 1u )
        {
            fprintf(stderr, "FAIL tp_mesh_fast_disable_cuda: explicit install overrode reference launcher while disabled\n");
            return 1;
        }
        printf("PASS tp_mesh_fast_disable_cuda: SPARK_MESH_FAST_DISABLE keeps reference launcher installed\n");
        return 0;
    }
    if ( g_registered_launchers->hardware != fast_hw )
    {
        fprintf(stderr, "FAIL tp_mesh_fast_disable_cuda: expected fast launcher when SPARK_MESH_FAST_DISABLE is off\n");
        return 1;
    }
    if ( setenv("SPARK_MESH_FAST_DISABLE", "1", 1) != 0 )
        return 1;
    SparkTpDeviceCollectiveRegisterLaunchers(&SparkTpMeshKernelLaunchers);
    SparkTpMeshFastLaunchersInstall();
    if ( g_registered_launchers->hardware != ref_hw )
    {
        fprintf(stderr, "FAIL tp_mesh_fast_disable_cuda: runtime SPARK_MESH_FAST_DISABLE=1 did not suppress install\n");
        return 1;
    }
    if ( setenv("SPARK_MESH_FAST_DISABLE", "0", 1) != 0 )
        return 1;
    SparkTpMeshFastLaunchersInstall();
    if ( g_registered_launchers->hardware != fast_hw )
    {
        fprintf(stderr, "FAIL tp_mesh_fast_disable_cuda: SPARK_MESH_FAST_DISABLE=0 did not allow fast install\n");
        return 1;
    }
    printf("PASS tp_mesh_fast_disable_cuda: fast launcher installs by default and respects SPARK_MESH_FAST_DISABLE\n");
    return 0;
}
