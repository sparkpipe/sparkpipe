#define _POSIX_C_SOURCE 200809L

#include "sparkpipe/spark_ck128.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* .experts lazy-manifest of record (tools/glm5_next_experts_manifest.c
   format): 16B header (WPEX magic, version, count, reserved) + 40B
   records (reserved u32, chunk ordinal u32, offset u64, bytes u64,
   ck128 16B). The records cover the whole pack in 64 MiB chunks —
   the granularity the residentd lazy map verifies at map time. */
#define MANIFEST_MAGIC UINT32_C(0x58504557)
#define MANIFEST_VERSION 1u
#define CHUNK_BYTES (64ull * 1024ull * 1024ull)

static void put32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4u); }
static void put64(uint8_t *p, uint64_t v) { memcpy(p, &v, 8u); }

int main(int argc, char **argv)
{
	FILE *pack;
	FILE *out;
	uint8_t *slice;
	uint8_t record[40];
	uint8_t digest[16];
	unsigned long long pack_bytes;
	unsigned long long offset;
	uint32_t ordinal = 0u;
	char path[4600];

	if (argc != 2)
	{
		fprintf(stderr, "usage: %s <pack>\n", argv[0]);
		return 2;
	}
	pack = fopen(argv[1], "rb");
	if (pack == 0)
	{
		perror("open pack");
		return 1;
	}
	if (fseeko(pack, 0ull, SEEK_END) != 0)
		return 1;
	pack_bytes = (unsigned long long)ftello(pack);
	if (pack_bytes == 0ull)
	{
		fprintf(stderr, "empty pack\n");
		return 1;
	}
	rewind(pack);
	snprintf(path, sizeof(path), "%s.experts", argv[1]);
	out = fopen(path, "wb");
	if (out == 0)
	{
		perror("open manifest");
		return 1;
	}
	put32(record + 0u, MANIFEST_MAGIC);
	put32(record + 4u, MANIFEST_VERSION);
	put32(record + 8u,
	    (uint32_t)((pack_bytes + CHUNK_BYTES - 1ull) / CHUNK_BYTES));
	put32(record + 12u, 0u);
	fwrite(record, 1u, 16u, out);
	slice = malloc(CHUNK_BYTES);
	if (slice == 0)
		return 1;
	for (offset = 0ull; offset < pack_bytes; offset += CHUNK_BYTES)
	{
		unsigned long long chunk_bytes = pack_bytes - offset;
		SparkCk128Context ck;

		if (chunk_bytes > CHUNK_BYTES)
			chunk_bytes = CHUNK_BYTES;
		if (fseeko(pack, (off_t)offset, SEEK_SET) != 0 ||
		    fread(slice, 1u, (size_t)chunk_bytes, pack) !=
			(size_t)chunk_bytes)
		{
			fprintf(stderr, "read failed at %llu\n",
				(unsigned long long)offset);
			return 1;
		}
		SparkCk128Initialize(&ck);
		SparkCk128Update(&ck, slice, (size_t)chunk_bytes);
		SparkCk128Finalize(&ck, digest);
		put32(record + 0u, 0u);
		put32(record + 4u, ordinal);
		put64(record + 8u, offset);
		put64(record + 16u, chunk_bytes);
		memcpy(record + 24u, digest, 16u);
		fwrite(record, 1u, 40u, out);
		ordinal++;
	}
	free(slice);
	fclose(out);
	fclose(pack);
	printf("%s: %u chunks cover %llu bytes\n", path,
		(unsigned int)ordinal,
		(unsigned long long)pack_bytes);
	return 0;
}
