/* dsv4_flash .experts manifest producer — the placement-time sidecar for the
 * shared weightd lazy tier (SPARK_WEIGHTD_RANGE_MANIFEST_VERSION 2).
 *
 * Family-specific by design: this tool reads the dsv4 stagepack directory
 * (modules/dsv4_resident_decode_stage/source/spark_dsv4_stagepack_format.h)
 * and emits one 48-byte range record per (layer, expert, plane) for
 * EXPERTS_W1/W2/W3 with kind = tensor_kind * 2 + plane (0 payload, 1 scale),
 * mirroring the glm5_next producer. Per-expert blocks are contiguous slices
 * of each tensor entry (expert-major stacking survives TP row sharding for
 * W1/W3 and W2's expert-major-row layout); scale planes emit their own
 * ranges. Coverage must include every non-MTP layer for all three kinds or
 * the manifest is refused (fail closed).
 *
 * Publish is atomic (mkstemp + rename). The ck128 digests are the
 * placement/acceptance record; runtime Ensure compares them only when the
 * operator enables verification.
 */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "sparkpipe/spark_ck128.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_manifest.h"
#include "sparkpipe/spark_dsv4_model.h"
#include "modules/dsv4_resident_decode_stage/source/spark_dsv4_stagepack_format.h"

#define DSV4_EXPERT_KIND_COUNT 3u
#define DSV4_MANIFEST_LAYERS 43u

static uint64_t plane_bytes(uint32_t weight_format, uint64_t rows, uint64_t columns, uint32_t plane)
{
	uint64_t elements = rows * columns;
	uint64_t payload;
	if (weight_format == SPARK_DSV4_STAGEPACK_WEIGHT_FP4_E2M1)
		payload = elements / 2u;
	else if (weight_format == SPARK_DSV4_STAGEPACK_WEIGHT_F32 ||
		weight_format == SPARK_DSV4_STAGEPACK_WEIGHT_U32)
		payload = elements * 4u;
	else if (weight_format == SPARK_DSV4_STAGEPACK_WEIGHT_FP8_E4M3)
		payload = elements;
	else
		payload = elements * 2u;
	if (plane == 0u)
		return payload;
	if (weight_format == SPARK_DSV4_STAGEPACK_WEIGHT_FP4_E2M1)
		return rows * ((columns + SPARK_DSV4_STAGEPACK_FP4_SCALE_BLOCK - 1u) /
			SPARK_DSV4_STAGEPACK_FP4_SCALE_BLOCK);
	if (weight_format == SPARK_DSV4_STAGEPACK_WEIGHT_FP8_E4M3)
		return rows * ((columns + SPARK_DSV4_STAGEPACK_FP8_SCALE_BLOCK - 1u) /
			SPARK_DSV4_STAGEPACK_FP8_SCALE_BLOCK);
	return 0u;
}

static int32_t range_write(FILE *pack, FILE *out, uint32_t layer,
	uint32_t expert, uint32_t kind, uint64_t offset, uint64_t bytes)
{
	SparkCk128Context ck;
	uint8_t buffer[65536];
	uint8_t record[48] = {0};
	uint64_t remaining, piece;
	if (bytes == 0u || bytes > SPARK_WEIGHTD_EXPERT_BYTES_MAX ||
		fseeko(pack, (off_t)offset, SEEK_SET) != 0)
		return -1;
	SparkCk128Initialize(&ck);
	remaining = bytes;
	while (remaining != 0u)
	{
		piece = remaining < sizeof(buffer) ? remaining : sizeof(buffer);
		if (fread(buffer, 1u, piece, pack) != piece)
			return -2;
		SparkCk128Update(&ck, buffer, piece);
		remaining -= piece;
	}
	memcpy(record, &layer, 4u);
	memcpy(record + 4u, &expert, 4u);
	memcpy(record + 8u, &kind, 4u);
	memcpy(record + 16u, &offset, 8u);
	memcpy(record + 24u, &bytes, 8u);
	SparkCk128Finalize(&ck, record + 32u);
	return fwrite(record, 1u, sizeof(record), out) == sizeof(record) ? 0 : -3;
}

static int32_t entry_write(FILE *pack, FILE *out,
	const SparkDsv4StagePackHeader *header,
	const SparkDsv4StagePackEntry *entry, uint32_t *count)
{
	uint64_t per, offset, bytes;
	uint32_t plane, expert, kind;
	if (entry->tensor_kind != SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W1 &&
		entry->tensor_kind != SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W2 &&
		entry->tensor_kind != SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W3)
		return 0;
	if (entry->layer_index >= DSV4_MANIFEST_LAYERS)
		return 0;
	if (header->routed_expert_count == 0u)
		return -5;
	for (plane = 0u; plane < 2u; plane++)
	{
		offset = plane == 0u ? entry->payload_offset : entry->scale_offset;
		bytes = plane_bytes(entry->weight_format, entry->rows,
			entry->columns, plane);
		if (plane == 1u && bytes == 0u)
			continue;
		if (bytes == 0u ||
			(bytes % (uint64_t)header->routed_expert_count) != 0u ||
			offset > header->file_bytes ||
			bytes > header->file_bytes - offset)
			return -6;
		per = bytes / (uint64_t)header->routed_expert_count;
		kind = (entry->tensor_kind * 2u) + plane;
		for (expert = 0u; expert < header->routed_expert_count; expert++)
		{
			int32_t err;
			if (*count == SPARK_WEIGHTD_RANGE_COUNT_MAX)
				return -7;
			err = range_write(pack, out, entry->layer_index, expert, kind,
				offset + ((uint64_t)expert * per), per);
			if (err < 0)
				return err;
			*count += 1u;
		}
	}
	return 0;
}

static int32_t header_read(FILE *pack, SparkDsv4StagePackHeader *header)
{
	struct stat st;
	uint64_t directory_bytes;
	if (fstat(fileno(pack), &st) != 0 || st.st_size < 0 ||
		fread(header, 1u, sizeof(*header), pack) != sizeof(*header))
		return -11;
	if (header->magic != SPARK_DSV4_STAGEPACK_MAGIC ||
		header->format_version != SPARK_DSV4_STAGEPACK_FORMAT_VERSION ||
		header->header_bytes != SPARK_DSV4_STAGEPACK_HEADER_BYTES ||
		header->directory_entry_bytes != SPARK_DSV4_STAGEPACK_ENTRY_BYTES)
		return -12;
	if (header->file_bytes != (uint64_t)st.st_size ||
		header->routed_expert_count == 0u || header->tensor_count == 0u)
		return -13;
	if (header->first_layer_index != 0u || header->layer_count != DSV4_MANIFEST_LAYERS)
		return -26;
	directory_bytes = (uint64_t)header->tensor_count *
		SPARK_DSV4_STAGEPACK_ENTRY_BYTES;
	if (header->directory_offset < SPARK_DSV4_STAGEPACK_HEADER_BYTES ||
		header->directory_offset > header->file_bytes ||
		directory_bytes > header->file_bytes - header->directory_offset)
		return -14;
	return 0;
}

static int32_t manifest_write(FILE *pack, FILE *out,
	const SparkDsv4StagePackHeader *header)
{
	uint32_t words[4] = {0u, 0u, 0u, 0u};
	uint32_t i;
	uint32_t covered[DSV4_MANIFEST_LAYERS];
	int32_t err;
	uint8_t *directory;
	uint32_t w1_seen = 0u, w2_seen = 0u, w3_seen = 0u;

	words[0] = SPARK_WEIGHTD_EXPERT_MANIFEST_MAGIC;
	words[1] = SPARK_WEIGHTD_RANGE_MANIFEST_VERSION;
	memset(covered, 0, sizeof(covered));
	if (fwrite(words, 1u, sizeof(words), out) != sizeof(words))
		return -8;
	directory = malloc((size_t)SPARK_DSV4_STAGEPACK_ENTRY_BYTES *
		header->tensor_count);
	if (directory == 0)
		return -9;
	if (fseeko(pack, (off_t)header->directory_offset, SEEK_SET) != 0 ||
		fread(directory, 1u, (size_t)SPARK_DSV4_STAGEPACK_ENTRY_BYTES *
			header->tensor_count, pack) !=
			(size_t)SPARK_DSV4_STAGEPACK_ENTRY_BYTES * header->tensor_count)
	{
		free(directory);
		return -9;
	}
	for (i = 0u; i < header->tensor_count; i++)
	{
		SparkDsv4StagePackEntry entry;
		memcpy(&entry, directory + (size_t)i * SPARK_DSV4_STAGEPACK_ENTRY_BYTES,
			sizeof(entry));
		err = entry_write(pack, out, header, &entry, &words[2]);
		if (err < 0)
		{
			free(directory);
			return err;
		}
		if (entry.layer_index < DSV4_MANIFEST_LAYERS)
		{
			if (entry.tensor_kind == SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W1)
			{
				w1_seen |= 1u;
				covered[entry.layer_index] |= 1u;
			}
			if (entry.tensor_kind == SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W2)
			{
				w2_seen |= 1u;
				covered[entry.layer_index] |= 2u;
			}
			if (entry.tensor_kind == SPARK_DSV4_STAGEPACK_TENSOR_EXPERTS_W3)
			{
				w3_seen |= 1u;
				covered[entry.layer_index] |= 4u;
			}
		}
	}
	free(directory);
	for (i = 0u; i < DSV4_MANIFEST_LAYERS; i++)
	{
		if (covered[i] != 7u)
		{
			fprintf(stderr, "layer %u missing expert coverage mask=%u\n", i,
				covered[i]);
			return -10;
		}
	}
	if (words[2] == 0u || w1_seen == 0u || w2_seen == 0u || w3_seen == 0u)
		return -10;
	if (fseeko(out, 0, SEEK_SET) != 0 ||
		fwrite(words, 1u, sizeof(words), out) != sizeof(words))
		return -10;
	return 0;
}

static int32_t manifest_publish(FILE *pack, const char *path)
{
	SparkDsv4StagePackHeader header;
	SparkWeightdManifest manifest;
	char temporary[4096];
	char final_path[4096];
	FILE *out;
	int32_t fd, err, written;
	(void)manifest;
	written = snprintf(temporary, sizeof(temporary), "%s.partial.XXXXXX",
		path);
	if (written < 0 || (uint32_t)written >= sizeof(temporary))
		return -15;
	fd = mkstemp(temporary);
	if (fd < 0)
		return -16;
	out = fdopen(fd, "wb");
	if (out == 0)
	{
		close(fd);
		unlink(temporary);
		return -16;
	}
	err = header_read(pack, &header);
	if (err == 0)
		err = manifest_write(pack, out, &header);
	if (fclose(out) != 0 && err == 0)
		err = -17;
	if (err != 0)
	{
		unlink(temporary);
		return err;
	}
	written = snprintf(final_path, sizeof(final_path), "%s.experts", path);
	if (written < 0 || (uint32_t)written >= sizeof(final_path))
		return -15;
	if (rename(temporary, final_path) != 0)
	{
		unlink(temporary);
		return -18;
	}
	return 0;
}

int main(int argc, char **argv)
{
	FILE *pack;
	const char *pack_path;
	int32_t err;
	if (argc != 2)
	{
		fprintf(stderr, "usage: %s <pack.spstage> (writes <pack>.experts)\n",
			argc > 0 ? argv[0] : "dsv4_flash_experts_manifest");
		return 2;
	}
	pack_path = argv[1];
	pack = fopen(pack_path, "rb");
	if (pack == 0)
	{
		perror(pack_path);
		return 1;
	}
	err = manifest_publish(pack, pack_path);
	fclose(pack);
	if (err != 0)
	{
		fprintf(stderr, "manifest failed: %d\n", err);
		return 1;
	}
	printf("experts manifest written beside %s\n", pack_path);
	return 0;
}
