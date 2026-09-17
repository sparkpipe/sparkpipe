/* weightd_loadall — attach EVERY arm's pack lazily at once (multiple
 * instances of the same model are distinct arenas by identity), then
 * ensure a small spread of segments per arm: the all-models resident
 * precondition for routed serving. Pools are sized so the full roster
 * fits the daemon ceiling; nothing beyond the ensured segments loads.
 *
 *   weightd_loadall <roster-file> <pool-mib> <touches>
 * roster lines: "<pack-path> <model>"
 * requires SPARK_WEIGHTD_SOCKET / SPARK_WEIGHTD_ATTACH.
 */
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_attach.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LOADALL_MAX_ARMS 64u
#define LOADALL_LINE_BYTES 1200u

typedef struct LoadallArm
{
    char pack[SPARK_WEIGHTD_PATH_BYTES];
    char model[SPARK_WEIGHTD_ID_BYTES];
    char manifest[SPARK_WEIGHTD_PATH_BYTES + 8];
    uint64_t generation;
    uint64_t experts;
} LoadallArm;

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

static int split_line(char *line, const char **pack, const char **model)
{
    char *space = strchr(line, ' ');
    if (space == 0)
    {
        return 1;
    }
    *space = '\0';
    *pack = line;
    *model = space + 1;
    if ((*pack)[0] == '\0' || (*model)[0] == '\0')
    {
        return 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    static const uint64_t timeout_ns = 300000000000ull;
    static LoadallArm arms[LOADALL_MAX_ARMS];
    static char line[LOADALL_LINE_BYTES];
    SparkWeightdHelloResult hello;
    SparkWeightdClient *client = 0;
    SparkWeightdLazyAttachRequest request;
    SparkWeightdLazyAttachResult attach;
    SparkWeightdEnsureResult ensure;
    uint64_t pool_bytes;
    uint64_t touches;
    uint64_t total_ns = 0ull;
    uint64_t total_ensured = 0ull;
    uint32_t count = 0u;
    uint32_t index;
    const char *socket;
    const char *pack_text;
    const char *model_text;
    FILE *roster;

    if (argc != 4)
    {
        fprintf(stderr, "usage: weightd_loadall roster pool-mib touches\n");
        return 2;
    }
    socket = getenv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET);
    if (socket == 0)
    {
        fprintf(stderr, "weightd_loadall: socket env unset\n");
        return 2;
    }
    pool_bytes = strtoull(argv[2], 0, 10) << 20;
    touches = strtoull(argv[3], 0, 10);
    if (pool_bytes == 0ull || touches == 0ull)
    {
        return 2;
    }
    roster = fopen(argv[1], "rb");
    if (roster == 0)
    {
        fprintf(stderr, "weightd_loadall: roster unreadable\n");
        return 2;
    }
    while (fgets(line, sizeof(line), roster) != 0)
    {
        size_t length = strlen(line);
        while (length > 0u && (line[length - 1u] == '\n' ||
                line[length - 1u] == '\r' || line[length - 1u] == ' '))
        {
            line[--length] = '\0';
        }
        if (length == 0u || count >= LOADALL_MAX_ARMS)
        {
            continue;
        }
        if (split_line(line, &pack_text, &model_text) != 0)
        {
            continue;
        }
        if (strlen(pack_text) >= SPARK_WEIGHTD_PATH_BYTES ||
            strlen(model_text) >= SPARK_WEIGHTD_ID_BYTES)
        {
            continue;
        }
        memcpy(arms[count].pack, pack_text, strlen(pack_text) + 1u);
        memcpy(arms[count].model, model_text, strlen(model_text) + 1u);
        count++;
    }
    (void)fclose(roster);
    if (count == 0u)
    {
        fprintf(stderr, "weightd_loadall: roster empty\n");
        return 2;
    }
    if (SparkWeightdClientConnect(socket, &client, &hello) != SPARK_STATUS_OK)
    {
        fprintf(stderr, "weightd_loadall: connect failed\n");
        return 1;
    }
    for (index = 0u; index < count; index++)
    {
        char sha_hex[SPARK_SHA256_HEX_BYTES];
        if (join_bounded(arms[index].manifest, sizeof(arms[index].manifest),
                arms[index].pack, ".experts") != 0 ||
            SparkSha256File(arms[index].manifest, sha_hex) != SPARK_STATUS_OK)
        {
            fprintf(stderr, "weightd_loadall: manifest missing for %u\n",
                (unsigned)index);
            SparkWeightdClientClose(client);
            return 1;
        }
        memset(&request, 0, sizeof(request));
        if (copy_bounded(request.identity.model, SPARK_WEIGHTD_ID_BYTES,
                arms[index].model) != 0 ||
            copy_bounded(request.pack_path, SPARK_WEIGHTD_PATH_BYTES,
                arms[index].pack) != 0)
        {
            SparkWeightdClientClose(client);
            return 2;
        }
        copy_bounded(request.identity.revision, SPARK_WEIGHTD_REVISION_BYTES,
            "loadall");
        request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
        request.identity.topology = 16u;
        request.identity.geometry_fingerprint = 0x504F4355ull;
        request.identity.arena_bytes = file_bytes(arms[index].pack);
        memcpy(request.identity.pack_sha256, sha_hex, strlen(sha_hex) + 1u);
        request.expert_pool_bytes = pool_bytes;
        memset(&attach, 0, sizeof(attach));
        if (request.identity.arena_bytes == 0ull ||
            SparkWeightdIdentityPrepare(&request.identity) != SPARK_STATUS_OK)
        {
            fprintf(stderr, "weightd_loadall: identity invalid %u\n",
                (unsigned)index);
            SparkWeightdClientClose(client);
            return 1;
        }
        if (SparkWeightdClientAttachLazy(client, &request, &attach,
                timeout_ns) != SPARK_STATUS_OK ||
            attach.status != SPARK_STATUS_OK)
        {
            fprintf(stderr, "weightd_loadall: attach_lazy failed %d\n",
                (int)attach.status);
            SparkWeightdClientClose(client);
            return 1;
        }
        arms[index].generation = attach.arena_generation;
        arms[index].experts = attach.expert_count;
    }
    for (index = 0u; index < count; index++)
    {
        uint64_t touch_index;
        uint64_t spread = arms[index].experts > 1ull
            ? arms[index].experts - 1ull : 1ull;
        for (touch_index = 0ull; touch_index < touches; touch_index++)
        {
            uint64_t expert = touches > 1ull
                ? (touch_index * spread) / (touches - 1ull) : 0ull;
            if (expert >= arms[index].experts)
            {
                expert = arms[index].experts - 1ull;
            }
            memset(&ensure, 0, sizeof(ensure));
            if (SparkWeightdClientEnsure(client, arms[index].generation, 0u,
                    (uint32_t)expert, &ensure, timeout_ns) !=
                    SPARK_STATUS_OK ||
                ensure.status != SPARK_STATUS_OK)
            {
                fprintf(stderr, "weightd_loadall: ensure failed %d\n",
                    (int)ensure.status);
                SparkWeightdClientClose(client);
                return 1;
            }
            total_ns += ensure.load_ns;
            if (ensure.loaded != 0u)
            {
                total_ensured++;
            }
        }
    }
    printf("LOADALL arms=%u touches=%llu cold_faults=%llu ensure_ns=%llu\n",
        (unsigned)count, (unsigned long long)touches,
        (unsigned long long)total_ensured,
        (unsigned long long)total_ns);
    SparkWeightdClientClose(client);
    return 0;
}
