#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "sparkpipe/spark_weightd_lazy_pack.h"

int main(int argc, char **argv)
{
    SparkWeightdLazyAttachRequest request;
    SparkWeightdLazyPack *pack;
    const char *digest;
    const char *socket;
    uint64_t spine_budget;
    struct stat info;
    SparkStatus status;

    if (argc != 2)
    {
        fprintf(stderr, "usage: %s <pack-path>\n", argv[0]);
        return 2;
    }
    if (stat(argv[1], &info) != 0)
    {
        perror("stat");
        return 3;
    }
    memset(&request, 0, sizeof(request));
    digest = getenv("SPARK_WEIGHTD_ATTACH_SHA256");
    if (digest == 0)
    {
        fprintf(stderr, "SPARK_WEIGHTD_ATTACH_SHA256 not set\n");
        return 4;
    }
    memcpy(request.identity.pack_sha256, digest, 65u);
    snprintf(request.identity.model, sizeof(request.identity.model), "%s",
        "spark.glm5_next.resident_decode_stage");
    snprintf(request.identity.revision, sizeof(request.identity.revision), "%s",
        "84c6a6aa9497188e15a635ba793b0f95a79b1033");
    request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
    request.identity.arena_bytes = (uint64_t)info.st_size;
    request.identity.topology = 16u;
    memcpy(request.pack_path, argv[1], strlen(argv[1]) + 1u);
    request.expert_pool_bytes = 4294967296ull;
    spine_budget = 8589934592ull;
    socket = getenv("SPARK_WEIGHTD_SOCKET");
    if (socket == 0)
        socket = "/tmp/spark_weightd.sock";
    fprintf(stderr, "harness: path=%s arena=%llu socket=%s\n",
        argv[1], (unsigned long long)request.identity.arena_bytes, socket);
    status = SparkWeightdLazyPackCreateChecked(socket, &request, spine_budget,
        120000000000ull, 0, 0, &pack);
    fprintf(stderr, "harness: CreateChecked status=%d\n", (int)status);
    if (status == SPARK_STATUS_OK && pack != 0)
        SparkWeightdLazyPackDestroy(pack);
    return status == SPARK_STATUS_OK ? 0 : 1;
}
