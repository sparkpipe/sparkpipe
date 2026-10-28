#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sparkpipe/spark_named_pack_bind.h"

#define PACK_ALIGNMENT 128u

static uint32_t failures;

static void Check(int condition, const char *message)
{
	if ( !condition )
	{
		fprintf(stderr, "FAIL test_named_pack_bind: %s\n", message);
		failures++;
	}
}

static uint64_t WritePack(const char *path, uint8_t *file, uint64_t capacity)
{
	static const char manifest[] = "{\"format\":{\"container\":\"spark_named_pack\"},\"config\":{},\"tensors\":{"
		"\"layers.0.mix\":{\"offset\":0,\"bytes\":64,\"align\":128,\"kind\":\"bf16\",\"shape\":[4,8]},"
		"\"layers.0.short\":{\"offset\":128,\"bytes\":48,\"align\":128,\"kind\":\"bf16\",\"shape\":[3,8]},"
		"\"layers.0.norm\":{\"offset\":256,\"bytes\":32,\"align\":128,\"kind\":\"f32\",\"shape\":[8]}}}";
	const uint64_t manifest_bytes = sizeof(manifest) - 1u, base = (16u + manifest_bytes + PACK_ALIGNMENT - 1u) / PACK_ALIGNMENT * PACK_ALIGNMENT;
	const uint64_t total = base + 384u;
	const uint32_t magic = SPARK_NAMED_PACK_MAGIC, version = SPARK_NAMED_PACK_VERSION;
	FILE *out;
	if ( total > capacity )
		return 0u;
	memset(file, 0, total);
	memcpy(file, &magic, 4u);
	memcpy(file + 4u, &version, 4u);
	memcpy(file + 8u, &manifest_bytes, 8u);
	memcpy(file + 16u, manifest, manifest_bytes);
	for ( uint64_t index = base; index < total; ++index )
		file[index] = (uint8_t)(index * 7u);
	out = fopen(path, "wb");
	if ( out == 0 || fwrite(file, 1u, total, out) != total || fclose(out) != 0 )
		return 0u;
	return total;
}

int main(void)
{
	static uint8_t file[4096];
	char path[] = "/tmp/sparkpipe-named-pack-bind-XXXXXX";
	SparkNamedPack pack;
	SparkWeightdLazyPack lazy;
	SparkWeightdSpan span;
	const void *bound;
	uint64_t total, offset = 0u, bytes = 0u;
	uint32_t rows = 0u;
	int fd = mkstemp(path);
	if ( fd < 0 )
		return 2;
	close(fd);
	total = WritePack(path, file, sizeof(file));
	Check(total != 0u, "the test pack is written");
	Check(SparkNamedPackOpen(path, SPARK_NAMED_PACK_MAGIC, SPARK_NAMED_PACK_VERSION, PACK_ALIGNMENT, &pack) == SPARK_STATUS_OK, "the test pack opens");
	memset(&lazy, 0, sizeof(lazy));
	memset(&span, 0, sizeof(span));
	span.offset = 0u;
	span.bytes = total;
	span.compact_offset = 0u;
	lazy.manifest.spine = &span;
	lazy.manifest.spine_count = 1u;
	lazy.manifest.spine_bytes = total;
	lazy.spine = file;
	lazy.ready = 1u;

	bound = 0;
	Check(SparkNamedPackBindSized(&pack, &lazy, "test", "layers.0.mix", "bf16", 4u * 8u * sizeof(uint16_t), &bound) == SPARK_STATUS_OK &&
		bound == file + pack.payload_base, "a tensor of exactly the geometry's bytes binds to its payload");
	bound = file;
	Check(SparkNamedPackBindSized(&pack, &lazy, "test", "layers.0.short", "bf16", 4u * 8u * sizeof(uint16_t), &bound) == SPARK_STATUS_TARGET_MISMATCH &&
		bound == file, "a tensor one row short of the geometry is refused and nothing is bound");
	Check(SparkNamedPackBindSized(&pack, &lazy, "test", "layers.0.mix", "bf16", 5u * 8u * sizeof(uint16_t), &bound) == SPARK_STATUS_TARGET_MISMATCH,
		"a geometry one row longer than the pack tensor is refused");
	Check(SparkNamedPackBindSized(&pack, &lazy, "test", "layers.0.mix", "bf16", 0u, &bound) == SPARK_STATUS_TARGET_MISMATCH,
		"a bind that names no expected size is refused, never skipped");
	Check(SparkNamedPackBindSized(&pack, &lazy, "test", "layers.0.mix", "f32", 64u, &bound) == SPARK_STATUS_PARSE_ERROR,
		"a tensor of another kind is refused");
	Check(SparkNamedPackBindSized(&pack, &lazy, "test", "layers.1.mix", "bf16", 64u, &bound) == SPARK_STATUS_PARSE_ERROR,
		"a missing tensor is refused");
	Check(SparkNamedPackBindSized(&pack, &lazy, "test", "layers.0.norm", "f32", 8u * sizeof(float), &bound) == SPARK_STATUS_OK &&
		bound == file + pack.payload_base + 256u, "a later tensor binds at its own offset");
	Check(SparkNamedPackTensorRange(&pack, "test", "layers.0.short", "bf16", &offset, &bytes, &rows) == SPARK_STATUS_OK &&
		offset == pack.payload_base + 128u && bytes == 48u && rows == 3u, "the tensor range reports its file offset, bytes and rows");
	lazy.ready = 0u;
	Check(SparkNamedPackBindSized(&pack, &lazy, "test", "layers.0.mix", "bf16", 64u, &bound) == SPARK_STATUS_PARSE_ERROR,
		"a tensor weightd holds no bytes for is refused");
	SparkNamedPackClose(&pack);
	unlink(path);
	if ( failures != 0u )
		return 1;
	printf("PASS test_named_pack_bind: a tensor binds only at exactly the geometry's bytes; one row short, no expected size, another kind, a missing tensor or no resident bytes is refused\n");
	return 0;
}
