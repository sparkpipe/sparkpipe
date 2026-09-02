/* weightdctl — manual load/unload driver for the weightd residency daemon
 * (the operator's debug-one-step law: scripts drive the daemon directly
 * before anything automated rides it).
 *
 *   weightdctl load   <pack-path> <model> <revision>   (attach; cold-loads)
 *   weightdctl unload <pack-path> <model> <revision>   (release+detach)
 *   weightdctl status                                      (attach probe)
 *
 * The identity fields mirror what a deployment would publish; the SHA is
 * computed from the file when SPARK_WEIGHTD_PACK_SHA256 is unset (a real
 * deployment carries it from placement receipts).
 */
#include "sparkpipe/spark_weightd_attach.h"
#include "sparkpipe/spark_sha256.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static uint64_t file_bytes(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (f == 0)
        return 0;
    fseeko(f, 0, SEEK_END);
    uint64_t n = (uint64_t)ftello(f);
    fclose(f);
    return n;
}

int main(int argc, char **argv)
{
    if (argc >= 2 && strcmp(argv[1], "status") == 0)
    {
        char reason[SPARK_WEIGHTD_ATTACH_REASON_BYTES];
        SparkStatus s = SparkWeightdAttachRequested();
        printf("requested=%d socket=%s\n", s == SPARK_STATUS_OK,
               getenv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET) ? : "(unset)");
        (void)reason;
        return 0;
    }
    if (argc >= 2 && strcmp(argv[1], "reclaim") == 0)
    {
        SparkWeightdClient *client = 0;
        SparkWeightdHelloResult hello;
        SparkWeightdReclaimResult reclaim;
        const char *socket = getenv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET);
        if (socket == 0)
        {
            fprintf(stderr, "weightdctl: %s unset\n",
                SPARK_WEIGHTD_ATTACH_ENV_SOCKET);
            return 2;
        }
        if (SparkWeightdClientConnect(socket, &client, &hello) !=
            SPARK_STATUS_OK)
        {
            fprintf(stderr, "weightdctl: no daemon\n");
            return 1;
        }
        if (SparkWeightdClientReclaim(client, &reclaim,
                SPARK_WEIGHTD_ATTACH_TIMEOUT_DEFAULT_NS) != SPARK_STATUS_OK)
        {
            SparkWeightdClientClose(client);
            return 1;
        }
        printf("RECLAIM status=%u reclaimed_bytes=%llu arenas=%u resident=%llu\n",
            reclaim.status, (unsigned long long)reclaim.reclaimed_bytes,
            reclaim.arena_count, (unsigned long long)reclaim.resident_bytes);
        SparkWeightdClientClose(client);
        return 0;
    }
    if (argc != 5 || (strcmp(argv[1], "load") != 0 && strcmp(argv[1], "unload") != 0))
    {
        fprintf(stderr,
            "usage: %s load|unload <pack-path> <model> <revision>\n"
            "       %s status\n", argv[0], argv[0]);
        return 2;
    }
    const char *mode = argv[1];
    const char *pack = argv[2];
    uint64_t bytes = file_bytes(pack);
    if (bytes == 0)
    {
        fprintf(stderr, "weightdctl: cannot size %s\n", pack);
        return 2;
    }
    char digest[SPARK_SHA256_HEX_BYTES];
    if (getenv(SPARK_WEIGHTD_ATTACH_ENV_SHA256) == 0)
    {
        if (SparkSha256File(pack, digest) != SPARK_STATUS_OK)
        {
            fprintf(stderr, "weightdctl: sha256 failed on %s\n", pack);
            return 2;
        }
        setenv(SPARK_WEIGHTD_ATTACH_ENV_SHA256, digest, 1);
    }
    SparkWeightdPackSlice slice;
    memset(&slice, 0, sizeof(slice));
    slice.model = argv[3];
    slice.revision = argv[4];
    slice.topology = 0u;
    slice.geometry_fingerprint = 0ull;
    slice.pack_bytes = bytes;
    SparkWeightdAttachOutcome outcome;
    memset(&outcome, 0, sizeof(outcome));
    char reason[SPARK_WEIGHTD_ATTACH_REASON_BYTES];
    uint64_t attach_timeout_ns = 300000000000ull;
    {
        const char *timeout_env = getenv("SPARK_WEIGHTDCTL_TIMEOUT_NS");
        if (timeout_env != 0 && timeout_env[0] != '\0')
        {
            attach_timeout_ns = strtoull(timeout_env, 0, 10);
        }
        if (attach_timeout_ns == 0ull)
        {
            attach_timeout_ns = 300000000000ull;
        }
    }
    SparkStatus s = SparkWeightdAttachPack(&slice, pack,
        attach_timeout_ns, &outcome, reason);
    if (s != SPARK_STATUS_OK)
    {
        fprintf(stderr, "weightdctl: attach fault %d\n", (int)s);
        return 1;
    }
    if (outcome.client == 0)
    {
        printf("FALLBACK reason=%s (daemon path not taken)\n", reason);
        return 3;
    }
    printf("ATTACHED %s arena_bytes=%llu generation=%llu cold=%u refcount=%u\n",
           mode, (unsigned long long)outcome.arena_bytes,
           (unsigned long long)outcome.arena_generation,
           outcome.loaded_from_pack, outcome.refcount);
    SparkWeightdAttachRelease(&outcome); /* unmap; the arena itself stays
                                          * resident (unload semantics come
                                          * from the daemon's reclaim path) */
    return 0;
}
