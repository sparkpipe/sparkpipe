#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_lease.h"
#include "sparkpipe/spark_weightd_manifest.h"

#define PACK_BYTES (UINT64_C(1) << 20)

static void write_manifest(const char *path, uint32_t range_count)
{
	uint32_t header[4] = {SPARK_WEIGHTD_EXPERT_MANIFEST_MAGIC, SPARK_WEIGHTD_RANGE_MANIFEST_VERSION, range_count, 0u};
	FILE *file = fopen(path, "wb");
	assert(file != 0);
	assert(fwrite(header, 1u, sizeof(header), file) == sizeof(header));
	assert(fclose(file) == 0);
}

int main(void)
{
	char path[] = "/tmp/sparkpipe-dense-manifest-XXXXXX";
	SparkWeightdManifest manifest, again;
	SparkWeightdLeaseTable *table = 0;
	uint8_t digest[32], digest_again[32];
	int fd = mkstemp(path);
	assert(fd >= 0);
	(void)close(fd);
	write_manifest(path, 0u);
	assert(SparkWeightdManifestLoad(path, PACK_BYTES, &manifest) == SPARK_STATUS_OK);
	assert(manifest.range_count == 0u && manifest.group_count == 0u);
	assert(manifest.spine_count == 1u && manifest.spine[0].offset == 0u && manifest.spine[0].bytes == PACK_BYTES);
	assert(manifest.spine_bytes == PACK_BYTES && manifest.spine_allocation_bytes >= PACK_BYTES);
	assert(SparkWeightdManifestFind(&manifest, 0u, 0u) == 0);
	assert(SparkWeightdManifestIdentity(&manifest, digest) == SPARK_STATUS_OK);
	assert(SparkWeightdManifestLoad(path, PACK_BYTES, &again) == SPARK_STATUS_OK);
	assert(SparkWeightdManifestIdentity(&again, digest_again) == SPARK_STATUS_OK && memcmp(digest, digest_again, sizeof(digest)) == 0);
	assert(SparkWeightdLeaseTableCreate(&manifest, &table) == SPARK_STATUS_OK && table != 0);
	assert(SparkWeightdLeaseTableDestroy(table) == SPARK_STATUS_OK);
	SparkWeightdManifestDestroy(&again);
	SparkWeightdManifestDestroy(&manifest);
	write_manifest(path, SPARK_WEIGHTD_RANGE_COUNT_MAX + 1u);
	assert(SparkWeightdManifestLoad(path, PACK_BYTES, &manifest) == SPARK_STATUS_CAPACITY_EXCEEDED);
	write_manifest(path, 1u);
	assert(SparkWeightdManifestLoad(path, PACK_BYTES, &manifest) != SPARK_STATUS_OK);
	assert(unlink(path) == 0);
	printf("PASS weightd dense manifest: a pack with no expert ranges is one whole-pack spine span with a stable identity and an empty lease table; a truncated range table is still refused\n");
	return(0);
}
