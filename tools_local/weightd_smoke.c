/* weightd_smoke — the pure-lazy driver prover: ATTACH_LAZY a pack with a
 * bounded expert pool, ENSURE a spread of segments (exactly what a smoke
 * test touches - nothing else loads), then detach. Nothing is preloaded
 * and no whole-file pass runs; RSS stays at one expert staging range.
 *
 *   weightd_smoke <pack> <model> <revision> <pool-mib> <touches>
 * requires <pack>.experts (weightd_expert_segments) and
 * SPARK_WEIGHTD_SOCKET / SPARK_WEIGHTD_ATTACH in the environment.
 */
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_attach.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    SparkWeightdEnsureResult ensure;
    SparkWeightdDetachResult detach;
    SparkWeightdHelloResult hello;
    SparkWeightdClient *client = 0;
    uint64_t pool_bytes;
    uint64_t pool_mib;
    uint64_t touches;
    uint64_t spread;
    uint64_t index;
    uint64_t count;
    uint64_t expert;
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
    if (copy_bounded(request.identity.model, SPARK_WEIGHTD_ID_BYTES,
            argv[2]) != 0 ||
        copy_bounded(request.identity.revision, SPARK_WEIGHTD_REVISION_BYTES,
            argv[3]) != 0 ||
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
    count = attach.expert_count;
    spread = count > 1ull ? count - 1ull : 1ull;
    for (index = 0ull; index < touches; index++)
    {
        expert = touches > 1ull ? (index * spread) / (touches - 1ull) : 0ull;
        if (expert >= count)
        {
            expert = count - 1ull;
        }
        memset(&ensure, 0, sizeof(ensure));
        if (SparkWeightdClientEnsure(client, attach.arena_generation, 0u,
                (uint32_t)expert, &ensure, timeout_ns) != SPARK_STATUS_OK ||
            ensure.status != SPARK_STATUS_OK)
        {
            fprintf(stderr, "weightd_smoke: ensure failed %d\n",
                (int)ensure.status);
            SparkWeightdClientClose(client);
            return 1;
        }
        total_ns += ensure.load_ns;
    }
    memset(&detach, 0, sizeof(detach));
    (void)SparkWeightdClientDetach(client, attach.arena_generation, &detach,
        timeout_ns);
    printf("SMOKE experts=%llu touches=%llu ensure_ns=%llu pool_mib=%llu\n",
        (unsigned long long)count, (unsigned long long)touches,
        (unsigned long long)total_ns, (unsigned long long)pool_mib);
    SparkWeightdClientClose(client);
    return 0;
}