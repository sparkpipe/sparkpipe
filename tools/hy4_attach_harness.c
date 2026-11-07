#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "sparkpipe/spark_weightd_lazy_pack.h"

#define HY4_MODULE_REVISION "779242edccdedc2109a0b36b164263a88f015bfa"

int main(int argc, char **argv)
{
	SparkWeightdLazyAttachRequest request;
	SparkWeightdLazyPack *pack;
	const char *digest;
	const char *socket;
	char experts_path[4096u];
	uint64_t spine_budget;
	struct stat info;
	struct stat experts;
	SparkStatus status;

	if ( argc != 2 )
	{
		fprintf(stderr, "usage: %s <pack-path>\n", argv[0]);
		return 2;
	}
	if ( stat(argv[1], &info) != 0 )
	{
		perror("stat");
		return 3;
	}
	snprintf(experts_path, sizeof(experts_path), "%s.experts", argv[1]);
	if ( stat(experts_path, &experts) != 0 )
	{
		fprintf(stderr, "attach: experts sidecar missing for %s\n", argv[1]);
		return 3;
	}
	memset(&request, 0, sizeof(request));
	digest = getenv("SPARK_WEIGHTD_ATTACH_SHA256");
	if ( digest == 0 )
	{
		fprintf(stderr, "SPARK_WEIGHTD_ATTACH_SHA256 not set\n");
		return 4;
	}
	memcpy(request.identity.pack_sha256, digest, 65u);
	snprintf(request.identity.model, sizeof(request.identity.model), "%s",
		"spark.hy4.resident_decode_stage.fp8.tp16.v1");
	snprintf(request.identity.revision, sizeof(request.identity.revision), "%s",
		HY4_MODULE_REVISION);
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = (uint64_t)info.st_size;
	request.identity.topology = 16u;
	memcpy(request.pack_path, argv[1], strlen(argv[1]) + 1u);
	spine_budget = 17179869184ull;
	socket = getenv("SPARK_WEIGHTD_SOCKET");
	if ( socket == 0 )
		socket = "/tmp/spark_weightd.sock";
	fprintf(stderr, "attach: path=%s arena=%llu experts=%llu socket=%s sha=%s\n",
		argv[1], (unsigned long long)request.identity.arena_bytes,
		(unsigned long long)experts.st_size, socket, digest);
	status = SparkWeightdLazyPackCreateChecked(socket, &request, 0u, spine_budget,
		120000000000ull, 0, 0, &pack);
	fprintf(stderr, "attach: CreateChecked status=%d (%s)\n", (int)status,
		SparkStatusToString(status));
	if ( status == SPARK_STATUS_OK && pack != 0 )
		SparkWeightdLazyPackDestroy(pack);
	return status == SPARK_STATUS_OK ? 0 : 1;
}
