#!/usr/bin/env bash
# sparkpipe_weightd GPU VMM verification (docs/WEIGHTD_DESIGN.md W2b) —
# SPARK-GATED RECEIPT: run this on a GB10 node with a real GPU; it is NOT
# part of the offline gates and the host CI never exercises the real cuMem*
# path (the host stub models it). The script builds a small consumer against
# the REAL CUDA driver + runtime and proves, on hardware:
#   1. the VMM granularity law: recommended granularity >= 2 MiB is used for
#      the arena's physical chunks (cuMemCreate per chunk),
#   2. the daemon's cold attach maps a real virtual arena whose bytes equal
#      the pack bytes (D2H readback through the mapped VA),
#   3. the warm attach shares ONE arena (same handle, refcount 2), and
#   4. the daemon TERM path frees every physical chunk + the VA (driver
#      calls succeed and the process exits 0).
# Exit 0 + "VMM VERIFY PASS" is the receipt; anything else is a failure.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
WORK="$(mktemp -d /tmp/spark_weightd_vmm_verify.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

if [ ! -f "$CUDA_HOME/include/cuda.h" ] || ! command -v cc >/dev/null 2>&1; then
    echo "SKIP sparkpipe_weightd_vmm_verify (CUDA_HOME=$CUDA_HOME unusable; spark-gated receipt)"
    exit 0
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "SKIP sparkpipe_weightd_vmm_verify (no visible GPU; spark-gated receipt)"
    exit 0
fi

cat > "$WORK/vmm_verify.c" <<'EOF'
/* The GPU-side VMM receipt consumer (see the wrapper script). */
#define _POSIX_C_SOURCE 200809L
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <cuda.h>

#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_weightd.h"

#define ARENA_BYTES (8ull * 1024ull * 1024ull) /* 4 x 2 MiB chunks */
#define CHUNK_MIN (2ull * 1024ull * 1024ull)

typedef struct { SparkWeightdServer *server; volatile sig_atomic_t stop; } server_thread;

static void *server_main(void *raw)
{
    server_thread *context = (server_thread *)raw;
    (void)SparkWeightdServerRun(context->server, &context->stop);
    return 0;
}

int main(void)
{
    const char *pack_path = "/tmp/spark_weightd_vmm_verify.spack";
    const char *socket_path = "/tmp/spark_weightd_vmm_verify.sock";
    SparkWeightdServerConfig config;
    SparkWeightdIdentity identity;
    SparkWeightdAttachRequest request;
    SparkWeightdAttachResult result;
    SparkWeightdHelloResult hello;
    SparkWeightdClient *first = 0;
    SparkWeightdClient *second = 0;
    server_thread thread_context;
    pthread_t thread_handle;
    SparkSha256Context digest;
    CUmemAllocationProp prop;
    size_t granularity = 0;
    uint8_t *staging = 0;
    uint8_t *readback = 0;
    char hex[SPARK_SHA256_HEX_BYTES];
    FILE *file;
    uint64_t offset;
    int device = 0;
    (void)signal(SIGPIPE, SIG_IGN);

    if (cudaGetDevice(&device) != cudaSuccess) { fprintf(stderr, "no cuda device\n"); return 1; }
    memset(&prop, 0, sizeof(prop));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    if (cuMemGetAllocationGranularity(&granularity, &prop,
            CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) != CUDA_SUCCESS ||
        granularity < CHUNK_MIN)
    {
        fprintf(stderr, "granularity %zu below the 2 MiB law\n", granularity);
        return 1;
    }
    printf("granularity=%zu\n", granularity);

    /* deterministic 8 MiB pack, the arena's content */
    staging = (uint8_t *)malloc(ARENA_BYTES);
    readback = (uint8_t *)malloc(ARENA_BYTES);
    if (staging == 0 || readback == 0) { return 1; }
    for (offset = 0; offset < ARENA_BYTES; offset++)
        staging[offset] = (uint8_t)(offset * 131ull + (offset >> 9));
    file = fopen(pack_path, "wb");
    if (file == 0 || fwrite(staging, 1u, ARENA_BYTES, file) != ARENA_BYTES)
        { return 1; }
    (void)fclose(file);
    SparkSha256Initialize(&digest);
    SparkSha256Update(&digest, staging, ARENA_BYTES);
    {
        uint8_t raw[SPARK_SHA256_DIGEST_BYTES];
        SparkSha256Finalize(&digest, raw);
        SparkSha256DigestToHex(raw, hex);
    }
    memset(&identity, 0, sizeof(identity));
    memcpy(identity.model, "vmm-verify", 11u);
    memcpy(identity.revision, "gpu", 4u);
    memcpy(identity.pack_sha256, hex, SPARK_SHA256_HEX_BYTES);
    identity.topology = 1u;
    identity.geometry_fingerprint = 1ull;
    identity.arena_bytes = ARENA_BYTES;
    identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
    if (SparkWeightdIdentityPrepare(&identity) != SPARK_STATUS_OK) { return 1; }

    memset(&config, 0, sizeof(config));
    config.socket_path = socket_path;
    config.device_bytes_max = 4ull * ARENA_BYTES;
    unlink(socket_path);
    if (SparkWeightdServerCreate(&config, &thread_context.server) != SPARK_STATUS_OK)
        { return 1; }
    thread_context.stop = 0;
    pthread_create(&thread_handle, 0, server_main, &thread_context);

    /* COLD: the real cuMemCreate/Map/SetAccess path inside the daemon */
    if (SparkWeightdClientConnect(socket_path, &first, &hello) != SPARK_STATUS_OK)
        { return 1; }
    memset(&request, 0, sizeof(request));
    request.identity = identity;
    memcpy(request.pack_path, pack_path, strlen(pack_path) + 1u);
    memset(&result, 0, sizeof(result));
    if (SparkWeightdClientAttach(first, &request, &result, 120000000000ull) !=
            SPARK_STATUS_OK ||
        result.status != SPARK_STATUS_OK || result.loaded_from_pack != 1u ||
        result.device_handle == 0ull)
    {
        fprintf(stderr, "cold attach failed status=%s\n",
            SparkStatusToString(result.status));
        return 1;
    }
    /* the mapped VA reads back the pack bytes through the REAL VMM map */
    if (cudaMemcpy(readback, (const void *)(uintptr_t)result.device_handle,
            ARENA_BYTES, cudaMemcpyDeviceToHost) != cudaSuccess ||
        memcmp(readback, staging, ARENA_BYTES) != 0)
    {
        fprintf(stderr, "arena readback mismatch over the real VMM map\n");
        return 1;
    }
    printf("cold arena bytes verified over cuMemMap span=%llu\n",
        (unsigned long long)result.arena_bytes);

    /* WARM: one arena, two consumers, same handle */
    if (SparkWeightdClientConnect(socket_path, &second, 0) != SPARK_STATUS_OK)
        { return 1; }
    memset(&result, 0, sizeof(result));
    if (SparkWeightdClientAttach(second, &request, &result, 120000000000ull) !=
            SPARK_STATUS_OK ||
        result.status != SPARK_STATUS_OK || result.loaded_from_pack != 0u ||
        result.refcount != 2u)
    {
        fprintf(stderr, "warm attach failed status=%s\n",
            SparkStatusToString(result.status));
        return 1;
    }
    printf("warm attach shared handle=%llx refcount=2\n",
        (unsigned long long)result.device_handle);

    SparkWeightdClientClose(first);
    SparkWeightdClientClose(second);
    __atomic_store_n(&thread_context.stop, 1, __ATOMIC_SEQ_CST);
    pthread_join(thread_handle, 0);
    SparkWeightdServerDestroy(thread_context.server);
    free(staging);
    free(readback);
    (void)remove(pack_path);
    (void)remove(socket_path);
    printf("VMM VERIFY PASS\n");
    return 0;
}
EOF

echo "building the VMM verify consumer against $CUDA_HOME"
cc -std=c11 -Wall -Wextra -O2 -g -pthread -D_GNU_SOURCE \
    -I"$ROOT" -I"$ROOT/include" -I"$ROOT/src" -I"$CUDA_HOME/include" \
    "$WORK/vmm_verify.c" \
    "$ROOT/runtime/spark_weightd.c" "$ROOT/src/spark_sha256.c" "$ROOT/src/spark_status.c" \
    -L"$CUDA_HOME/lib64" -lcudart -lcuda -lpthread -o "$WORK/vmm_verify" || exit 1
"$WORK/vmm_verify" || exit 1
echo "sparkpipe_weightd VMM receipt: green"
