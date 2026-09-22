/* Attach-then-register reproduction for the a7 MESH-REGISTER-FAIL EINVAL.
 *
 * Replicates the resident's exact sequence against the SHARED weightd:
 *   1. SparkWeightdClientConnect + SparkWeightdClientAttachLazy (the same
 *      call weightd_warm makes; the daemon stages the mesh memfd and the
 *      client maps it at an aligned fixed window),
 *   2. cudaHostRegister(mesh_send_buffer_addr, SPARK_WEIGHTD_MESH_REGION_BYTES,
 *      cudaHostRegisterPortable | cudaHostRegisterMapped) - the exact call
 *      ring/transport/tp_device_collective.c:2061 makes in
 *      SparkTpDeviceCollectivePrepareReceiveBf16.
 *
 * A fresh-memfd control (tools/mesh_register_repro.c) registers cleanly under
 * the same queue cgroup, so an EINVAL here isolates the trigger to the
 * attach-provided mapping itself.
 *
 * usage: mesh_register_attach_repro SOCKET PACK SHA256 REVISION TOPOLOGY
 * env:   SPARK_WEIGHTD_EXPERT_POOL_BYTES (finite), SPARK_WEIGHTD_SPINE_BUDGET_BYTES
 */
#define _POSIX_C_SOURCE 200809L
#include "sparkpipe/spark_ck128.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_manifest.h"
#include <cuda_runtime.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int parse_positive(const char *text, unsigned long max, unsigned long *out)
{
    char *end;
    unsigned long value;
    if (text == 0 || text[0] < '0' || text[0] > '9') return 0;
    errno = 0;
    value = strtoul(text, &end, 10);
    if (errno != 0 || *end != '\0' || value == 0u || value > max) return 0;
    *out = value;
    return 1;
}

int main(int argc, char **argv)
{
    SparkWeightdLazyAttachRequest request;
    SparkWeightdClient *client = 0;
    SparkWeightdLazyAttachResult attached;
    SparkWeightdManifest manifest;
    struct stat pack;
    unsigned long topology, pool, spine = 4294967296ul, seconds = 120;
    SparkStatus status;
    char manifest_path[4096];
    cudaError_t err;
    void *mesh;
    uint64_t bytes;

    if (argc != 6 ||
        !parse_positive(argv[5], 65535u, &topology) ||
        getenv("SPARK_WEIGHTD_EXPERT_POOL_BYTES") == 0 ||
        !parse_positive(getenv("SPARK_WEIGHTD_EXPERT_POOL_BYTES"), 1099511627775ul, &pool))
    {
        fprintf(stderr, "usage: %s SOCKET PACK SHA256 REVISION TOPOLOGY\n"
                        "env: finite SPARK_WEIGHTD_EXPERT_POOL_BYTES, optional "
                        "SPARK_WEIGHTD_SPINE_BUDGET_BYTES\n", argv[0]);
        return 2;
    }
    if (getenv("SPARK_WEIGHTD_SPINE_BUDGET_BYTES") != 0)
        (void)parse_positive(getenv("SPARK_WEIGHTD_SPINE_BUDGET_BYTES"), 1099511627775ul, &spine);

    memset(&request, 0, sizeof(request));
    if (strlen(argv[2]) >= sizeof(request.pack_path) || strlen(argv[3]) != 64u ||
        stat(argv[2], &pack) != 0 || !S_ISREG(pack.st_mode) || pack.st_size <= 0)
    {
        fprintf(stderr, "invalid pack or identity\n");
        return 2;
    }
    memcpy(request.identity.pack_sha256, argv[3], 65u);
    strcpy(request.identity.model, "glm5_next_stage");
    strcpy(request.identity.revision, argv[4]);
    request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
    request.identity.arena_bytes = (uint64_t)pack.st_size;
    request.identity.topology = (uint32_t)topology;
    strcpy(request.pack_path, argv[2]);
    request.expert_pool_bytes = pool;
    snprintf(manifest_path, sizeof(manifest_path), "%s.experts", argv[2]);
    status = SparkWeightdManifestLoad(manifest_path, request.identity.arena_bytes, &manifest);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "expert manifest failed status=%d\n", (int)status);
        return 2;
    }

    status = SparkWeightdClientConnect(argv[1], &client, 0);
    if (status == SPARK_STATUS_OK)
    {
        memset(&attached, 0, sizeof(attached));
        status = SparkWeightdClientAttachLazy(client, &request, &attached,
            seconds * UINT64_C(1000000000));
    }
    if (status != SPARK_STATUS_OK || attached.status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "attach failed status=%d daemon=%u\n", (int)status, attached.status);
        return 1;
    }
    mesh = (void *)(uintptr_t)attached.mesh_send_buffer_addr;
    bytes = attached.mesh_send_buffer_bytes;
    printf("attach OK mesh_ready=%u mesh=%p bytes=%llu arena=%llu\n",
           attached.mesh_ready, mesh, (unsigned long long)bytes,
           (unsigned long long)attached.arena_bytes);
    if (mesh == 0 || bytes != SPARK_WEIGHTD_MESH_REGION_BYTES)
    {
        fprintf(stderr, "no usable mesh region from attach\n");
        return 1;
    }
    err = cudaFree(0);
    printf("cuda context: %s\n", cudaGetErrorString(err));
    {
        static const struct { const char *label; unsigned flags; } variants[] = {
            {"PORTABLE|MAPPED", cudaHostRegisterPortable | cudaHostRegisterMapped},
            {"PORTABLE", cudaHostRegisterPortable},
            {"default", 0u},
        };
        int failures = 0;
        for (size_t index = 0; index < sizeof(variants) / sizeof(variants[0]); index++)
        {
            err = cudaHostRegister(mesh, (size_t)bytes, variants[index].flags);
            printf("cudaHostRegister(attach mesh va, %s) -> %s (%d)\n",
                   variants[index].label, cudaGetErrorString(err), (int)err);
            if (err == cudaSuccess)
            {
                void *device = 0;
                cudaError_t mapped_err = cudaHostGetDevicePointer(&device, mesh, 0);
                printf("  cudaHostGetDevicePointer -> %s (%d) device=%p\n",
                       cudaGetErrorString(mapped_err), (int)mapped_err, device);
                cudaHostUnregister(mesh);
            }
            else
                failures++;
        }
        return failures != 0;
    }
}
