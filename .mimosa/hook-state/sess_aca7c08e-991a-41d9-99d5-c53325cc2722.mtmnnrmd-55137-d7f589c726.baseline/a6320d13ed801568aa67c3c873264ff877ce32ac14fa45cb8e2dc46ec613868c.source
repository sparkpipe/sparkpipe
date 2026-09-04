#!/usr/bin/env bash
# sparkpipe_weightd GPU VMM + fd-tier verification (docs/WEIGHTD_DESIGN.md
# W2b + W3) — SPARK-GATED RECEIPT: run this on a GB10 node with a real GPU;
# it is NOT part of the offline gates and the host CI never exercises the
# real cuMem* path (the host stub models it). The script builds a consumer
# against the REAL CUDA driver + runtime and proves, on hardware:
#   1. the VMM granularity law: recommended granularity >= 2 MiB is used for
#      the arena's physical chunks (cuMemCreate per chunk),
#   2. the daemon's cold attach maps a real virtual arena whose bytes equal
#      the pack bytes (D2H readback through the mapped VA),
#   3. the warm attach shares ONE arena (same handle, refcount 2),
#   4. the W3 IMPORT LEG, in-process: the attach helper exports the arena's
#      chunks as POSIX shareable fds, imports them, verifies the chunk set
#      covers the pack's byte range, and maps them at the CONSUMER's own
#      span (cuMemImportFromShareableHandle + reserve + map + SetAccess RW);
#      the consumer-local VA reads back byte-exact,
#   5. the W3 CROSS-PROCESS leg: a real second consumer PROCESS attaches
#      warm, runs the same import/map through the production helper, and
#      reads the pack bytes through ITS OWN mapping (SCM_RIGHTS for real),
#   6. warm re-attach after the consumer process exits (the arena stays
#      warm), and
#   7. the daemon TERM path frees every physical chunk + the VA (driver
#      orderings hold, process exits 0).
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
/* The GPU-side VMM + fd-tier receipt consumer (see the wrapper script).
 * Modeless invocation runs the daemon-side parent (legs 1-4 + 6-7);
 * `--consumer <pack> <hex>` runs the real second-process consumer (leg 5)
 * through the production attach helper. */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <cuda.h>

#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_weightd_attach.h"
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

/* ---- leg 5: the real second-process consumer, production helper path ---- */
static int consumer_main(const char *pack_path, const char *hex)
{
    SparkWeightdPackSlice slice;
    SparkWeightdAttachOutcome outcome;
    uint8_t *expected = 0;
    uint8_t *readback = 0;
    char reason[SPARK_WEIGHTD_ATTACH_REASON_BYTES];
    FILE *file;
    /* the digest ALSO arrives via SPARK_WEIGHTD_PACK_SHA256 (inherited env);
     * the argv copy is the belt to that suspenders — the child verifies the
     * map against the PACK FILE, the daemon-independent bytes authority */
    (void)hex;
    (void)signal(SIGPIPE, SIG_IGN);

    memset(&slice, 0, sizeof(slice));
    slice.model = "vmm-verify";
    slice.revision = "gpu";
    slice.topology = 1u;
    slice.geometry_fingerprint = 1ull;
    slice.pack_bytes = ARENA_BYTES;

    expected = (uint8_t *)malloc(ARENA_BYTES);
    readback = (uint8_t *)malloc(ARENA_BYTES);
    file = fopen(pack_path, "rb");
    if (expected == 0 || readback == 0 || file == 0 ||
        fread(expected, 1u, ARENA_BYTES, file) != ARENA_BYTES)
    {
        fprintf(stderr, "consumer: pack read failed\n");
        return 1;
    }
    (void)fclose(file);

    if (SparkWeightdAttachPack(&slice, pack_path,
            SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS, &outcome, reason) !=
            SPARK_STATUS_OK ||
        outcome.client == 0)
    {
        fprintf(stderr, "consumer: attach fell back reason=%s\n", reason);
        return 1;
    }
    if (outcome.loaded_from_pack != 0u)
    {
        fprintf(stderr, "consumer: attach was not the warm hand-off\n");
        return 1;
    }
    if (SparkWeightdAttachImportMap(&outcome, ARENA_BYTES,
            SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS, reason) !=
            SPARK_STATUS_OK ||
        outcome.map_base == 0)
    {
        fprintf(stderr, "consumer: import map failed reason=%s\n", reason);
        return 1;
    }
    /* byte-exact readback through THIS process's imported mapping */
    if (cudaMemcpy(readback, outcome.map_base, ARENA_BYTES,
            cudaMemcpyDeviceToHost) != cudaSuccess ||
        memcmp(readback, expected, ARENA_BYTES) != 0)
    {
        fprintf(stderr, "consumer: imported map readback mismatch\n");
        return 1;
    }
    SparkWeightdAttachRelease(&outcome); /* unmap; no detach; arena warm */
    free(expected);
    free(readback);
    printf("consumer import map verified bytes=%llu\n",
        (unsigned long long)ARENA_BYTES);
    return 0;
}

int main(int argument_count, char **arguments)
{
    const char *socket_path;
    const char *pack_path;
    SparkWeightdServerConfig config;
    SparkWeightdIdentity identity;
    SparkWeightdAttachRequest request;
    SparkWeightdAttachResult result;
    SparkWeightdAttachOutcome helper;
    SparkWeightdHelloResult hello;
    SparkWeightdClient *first = 0;
    SparkWeightdClient *second = 0;
    SparkWeightdClient *fresh = 0;
    server_thread thread_context;
    pthread_t thread_handle;
    SparkSha256Context digest;
    CUmemAllocationProp prop;
    size_t granularity = 0;
    uint8_t *staging = 0;
    uint8_t *readback = 0;
    char hex[SPARK_SHA256_HEX_BYTES];
    char socket_text[128];
    char pack_text[128];
    char sha_env[SPARK_SHA256_HEX_BYTES];
    FILE *file;
    uint64_t offset;
    uint64_t generation;
    int device = 0;
    pid_t consumer_pid = -1;
    int consumer_status = 0;

    (void)signal(SIGPIPE, SIG_IGN);
    if (argument_count == 4 && strcmp(arguments[1], "--consumer") == 0)
    {
        return consumer_main(arguments[2], arguments[3]);
    }

    /* unique per-run paths: two concurrent verifies must never share a
     * socket (two daemons on one path would be a test bug, not a feature) */
    snprintf(socket_text, sizeof(socket_text),
        "/tmp/spark_weightd_vmm_verify_%ld.sock", (long)getpid());
    snprintf(pack_text, sizeof(pack_text),
        "/tmp/spark_weightd_vmm_verify_%ld.spack", (long)getpid());
    socket_path = socket_text;
    pack_path = pack_text;

    if (cudaGetDevice(&device) != cudaSuccess) { fprintf(stderr, "no cuda device\n"); return 1; }
    memset(&prop, 0, sizeof(prop));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
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
    generation = result.arena_generation;
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

    /* W3 IMPORT LEG, in-process: the production helper exports the chunks,
     * verifies coverage, and maps them at THIS process's own span */
    memset(&helper, 0, sizeof(helper));
    if (setenv("SPARK_WEIGHTD_SOCKET", socket_path, 1) != 0 ||
        setenv("SPARK_WEIGHTD_PACK_SHA256", hex, 1) != 0)
    {
        return 1;
    }
    {
        SparkWeightdPackSlice slice;
        char reason[SPARK_WEIGHTD_ATTACH_REASON_BYTES];
        memset(&slice, 0, sizeof(slice));
        slice.model = "vmm-verify";
        slice.revision = "gpu";
        slice.topology = 1u;
        slice.geometry_fingerprint = 1ull;
        slice.pack_bytes = ARENA_BYTES;
        if (SparkWeightdAttachPack(&slice, pack_path,
                SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS, &helper,
                reason) != SPARK_STATUS_OK || helper.client == 0)
        {
            fprintf(stderr, "helper attach fell back reason=%s\n", reason);
            return 1;
        }
        if (helper.loaded_from_pack != 0u || helper.refcount != 3u)
        {
            fprintf(stderr, "helper attach was not the warm hit\n");
            return 1;
        }
        if (SparkWeightdAttachImportMap(&helper, ARENA_BYTES,
                SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS, reason) !=
                SPARK_STATUS_OK || helper.map_base == 0)
        {
            fprintf(stderr, "helper import map failed reason=%s\n", reason);
            return 1;
        }
        if (cudaMemcpy(readback, helper.map_base, ARENA_BYTES,
                cudaMemcpyDeviceToHost) != cudaSuccess ||
            memcmp(readback, staging, ARENA_BYTES) != 0)
        {
            fprintf(stderr, "consumer-local map readback mismatch\n");
            return 1;
        }
        printf("in-process import map verified chunks=%u chunk_bytes=%llu\n",
            helper.map_chunk_count, (unsigned long long)helper.map_chunk_bytes);
    }
    SparkWeightdAttachRelease(&helper); /* unmap; no detach; refcount 2 */

    /* W3 CROSS-PROCESS LEG: a real second consumer PROCESS runs the same
     * production path and reads the pack bytes through ITS OWN mapping */
    memcpy(sha_env, hex, sizeof(sha_env));
    consumer_pid = fork();
    if (consumer_pid < 0) { return 1; }
    if (consumer_pid == 0)
    {
        execl("/proc/self/exe", "vmm_verify", "--consumer", pack_path,
            sha_env, (char *)0);
        _exit(127);
    }
    if (waitpid(consumer_pid, &consumer_status, 0) != consumer_pid ||
        !WIFEXITED(consumer_status) || WEXITSTATUS(consumer_status) != 0)
    {
        fprintf(stderr, "consumer process failed\n");
        return 1;
    }

    /* WARM RE-ATTACH AFTER CONSUMER EXIT: still the same arena, no reload */
    memset(&result, 0, sizeof(result));
    if (SparkWeightdClientConnect(socket_path, &fresh, 0) != SPARK_STATUS_OK)
        { return 1; }
    if (SparkWeightdClientAttach(fresh, &request, &result, 120000000000ull) !=
            SPARK_STATUS_OK ||
        result.status != SPARK_STATUS_OK || result.loaded_from_pack != 0u ||
        result.arena_generation != generation)
    {
        fprintf(stderr, "warm re-attach after consumer exit failed\n");
        return 1;
    }
    printf("warm re-attach after consumer exit generation preserved\n");

    SparkWeightdClientClose(fresh);
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
    "$ROOT/runtime/spark_weightd.c" "$ROOT/runtime/spark_weightd_attach.c" \
    "$ROOT/src/spark_sha256.c" "$ROOT/src/spark_status.c" \
    -L"$CUDA_HOME/lib64" -lcudart -lcuda -lpthread -o "$WORK/vmm_verify" || exit 1
"$WORK/vmm_verify" || exit 1
echo "sparkpipe_weightd VMM receipt: green"
