/* weightd_smoke — the pure-lazy driver prover: ATTACH_LAZY a pack with a
 * bounded expert pool, then exercise the sanctioned lease tier: ACQUIRE a
 * spread of manifest groups (the daemon commits chunks and copies the
 * ranges H2D under an owner-scoped lease), RELEASE each after the
 * synchronous acquire copy (no GPU work here), then detach. Nothing is
 * preloaded and no whole-file pass runs; RSS stays at one expert staging
 * range.
 *
 *   weightd_smoke <pack> <model> <revision> <pool-mib> <touches>
 * requires <pack>.experts (v2 range manifest) and
 * SPARK_WEIGHTD_SOCKET / SPARK_WEIGHTD_ATTACH in the environment.
 */
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_attach.h"
#include "sparkpipe/spark_weightd_manifest.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t smoke_monotonic_ns(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
    {
        return 0ull;
    }
    return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}

static uint64_t file_bytes(const char *path)
{
    FILE *file = fopen(path, "rb");
    uint64_t bytes = 0ull;
    if (file == 0)
    {
        return 0ull;
    }
    if (fseeko(file, 0, SEEK_END) == 0)
    {
        bytes = (uint64_t)ftello(file);
    }
    (void)fclose(file);
    return bytes;
}

static int copy_bounded(char *destination, size_t capacity, const char *source)
{
    size_t bytes = strlen(source) + 1u;
    if (bytes > capacity)
    {
        return 1;
    }
    memcpy(destination, source, bytes);
    return 0;
}

static int join_bounded(char *destination, size_t capacity, const char *base,
    const char *suffix)
{
    size_t base_bytes = strlen(base);
    size_t suffix_bytes = strlen(suffix) + 1u;
    if (base_bytes + suffix_bytes > capacity)
    {
        return 1;
    }
    memcpy(destination, base, base_bytes);
    memcpy(destination + base_bytes, suffix, suffix_bytes);
    return 0;
}

int main(int argc, char **argv)
{
    static const uint64_t timeout_ns = 300000000000ull;
    char manifest_path[SPARK_WEIGHTD_PATH_BYTES + 8];
    char sha_hex[SPARK_SHA256_HEX_BYTES];
    SparkWeightdLazyAttachRequest request;
    SparkWeightdLazyAttachResult attach;
    SparkWeightdWorkingSetResult working;
    SparkWeightdDetachResult detach;
    SparkWeightdHelloResult hello;
    SparkWeightdManifest manifest;
    SparkWeightdExpertKey key;
    SparkWeightdClient *client = 0;
    uint64_t pool_bytes;
    uint64_t pool_mib;
    uint64_t touches;
    uint64_t spread;
    uint64_t index;
    uint64_t count;
    uint64_t total_ns = 0ull;
    const char *socket;

    if (argc != 6)
    {
        fprintf(stderr, "usage: weightd_smoke pack model revision pool-mib touches\n");
        return 2;
    }
    socket = getenv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET);
    if (socket == 0)
    {
        fprintf(stderr, "weightd_smoke: socket env unset\n");
        return 2;
    }
    pool_mib = strtoull(argv[4], 0, 10);
    touches = strtoull(argv[5], 0, 10);
    if (pool_mib == 0ull || touches == 0ull)
    {
        return 2;
    }
    pool_bytes = pool_mib << 20;
    if (join_bounded(manifest_path, sizeof(manifest_path), argv[1],
            ".experts") != 0 ||
        SparkSha256File(manifest_path, sha_hex) != SPARK_STATUS_OK)
    {
        fprintf(stderr, "weightd_smoke: manifest missing\n");
        return 2;
    }
    if (SparkWeightdClientConnect(socket, &client, &hello) != SPARK_STATUS_OK)
    {
        fprintf(stderr, "weightd_smoke: connect failed\n");
        return 1;
    }
    memset(&request, 0, sizeof(request));
    if (copy_bounded(request.identity.model, sizeof(request.identity.model),
            argv[2]) != 0 ||
        copy_bounded(request.identity.revision,
            sizeof(request.identity.revision), argv[3]) != 0 ||
        copy_bounded(request.identity.pack_sha256,
            sizeof(request.identity.pack_sha256), sha_hex) != 0 ||
        copy_bounded(request.pack_path, SPARK_WEIGHTD_PATH_BYTES, argv[1]) != 0)
    {
        SparkWeightdClientClose(client);
        return 2;
    }
    request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
    request.identity.topology = 16u;
    request.identity.geometry_fingerprint = 0x504F4355ull;
    request.identity.arena_bytes = file_bytes(argv[1]);
    memcpy(request.identity.pack_sha256, sha_hex, strlen(sha_hex) + 1u);
    request.expert_pool_bytes = pool_bytes;
    if (request.identity.arena_bytes == 0ull ||
        SparkWeightdIdentityPrepare(&request.identity) != SPARK_STATUS_OK)
    {
        fprintf(stderr, "weightd_smoke: identity invalid\n");
        SparkWeightdClientClose(client);
        return 1;
    }
    memset(&attach, 0, sizeof(attach));
    if (SparkWeightdClientAttachLazy(client, &request, &attach,
            timeout_ns) != SPARK_STATUS_OK ||
        attach.status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "weightd_smoke: attach_lazy failed %d\n",
            (int)attach.status);
        SparkWeightdClientClose(client);
        return 1;
    }
    if (SparkWeightdManifestLoad(manifest_path,
            request.identity.arena_bytes, &manifest) != SPARK_STATUS_OK)
    {
        fprintf(stderr, "weightd_smoke: manifest load failed\n");
        SparkWeightdClientClose(client);
        return 1;
    }
    if (manifest.group_count == 0u)
    {
        fprintf(stderr, "weightd_smoke: manifest has no expert groups\n");
        SparkWeightdManifestDestroy(&manifest);
        SparkWeightdClientClose(client);
        return 1;
    }
    count = manifest.group_count;
    spread = count > 1ull ? count - 1ull : 1ull;
    for (index = 0ull; index < touches; index++)
    {
        uint64_t group = touches > 1ull ? (index * spread) / (touches - 1ull)
            : 0ull;
        uint64_t touch_start;
        if (group >= count)
        {
            group = count - 1ull;
        }
        key.layer = manifest.groups[group].layer;
        key.expert = manifest.groups[group].expert;
        touch_start = smoke_monotonic_ns();
        memset(&working, 0, sizeof(working));
        if (SparkWeightdClientAcquire(client, attach.arena_generation, &key,
                1u, &working, timeout_ns) != SPARK_STATUS_OK ||
            working.status != SPARK_STATUS_OK)
        {
            fprintf(stderr, "weightd_smoke: acquire failed %d\n",
                (int)working.status);
            SparkWeightdManifestDestroy(&manifest);
            SparkWeightdClientClose(client);
            return 1;
        }
        /* the acquire copy is synchronous; releasing right away satisfies
         * the lease contract for a consumer with no GPU work in flight */
        (void)SparkWeightdClientRelease(client, attach.arena_generation,
            working.lease_identifier, &working, timeout_ns);
        total_ns += smoke_monotonic_ns() - touch_start;
    }
    SparkWeightdManifestDestroy(&manifest);
    memset(&detach, 0, sizeof(detach));
    (void)SparkWeightdClientDetach(client, attach.arena_generation, &detach,
        timeout_ns);
    printf("SMOKE groups=%llu touches=%llu acquire_ns=%llu pool_mib=%llu\n",
        (unsigned long long)count, (unsigned long long)touches,
        (unsigned long long)total_ns, (unsigned long long)pool_mib);
    SparkWeightdClientClose(client);
    return 0;
}
