#define _POSIX_C_SOURCE 200809L

/* Qwen 3.8 27B dense-FFN expert manifest producer (multidev lane 1).
 *
 * The 27B is a DENSE model: the weightd expert tier is the per-layer FFN
 * (gate/up/down). Every pack layer with FFN tensors becomes exactly one
 * expert group (layer, expert 0) whose ranges are the payload and scale
 * planes of the three FFN tensors (manifest kinds 10..15 = tensor kind * 2
 * + plane, mirroring the qwen38max convention). The spine (embedding, head,
 * norms, attention/GDN projections) is the loader-computed complement and
 * stays full resolution per the quality law.
 *
 * Manifest wire format (include/sparkpipe/spark_weightd_manifest.h):
 *   16-byte header: magic, version 2, range_count, zero (u32 LE)
 *   48-byte record: layer, expert, kind, zero (u32), offset, bytes (u64),
 *   ck128[16] over the pack bytes of the range.
 *
 * Self-verifies through SparkWeightdManifestLoad before the atomic
 * link() publish; the existing output is preserved on any failure.
 */

#include "sparkpipe/spark_ck128.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_manifest.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_stagepack_format.h"

#define RANGE_MAX SPARK_WEIGHTD_RANGE_COUNT_MAX
#define RECORD_BYTES 48u

typedef struct
{
	uint32_t tensor_count;
	uint32_t first_layer;
	uint32_t layer_count;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint64_t directory_offset;
	uint64_t file_bytes;
} PackHeader;

typedef struct
{
	uint32_t layer;
	uint32_t expert;
	uint32_t kind;
	uint64_t offset;
	uint64_t bytes;
} RangeWire;

static const uint32_t ffn_kinds[3] = {
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_GATE,
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_UP,
	SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_DOWN,
};

static int32_t range_write(FILE *pack, FILE *out, uint32_t layer,
    uint32_t expert, uint32_t kind, uint64_t offset, uint64_t bytes)
{
	SparkCk128Context ck;
	uint8_t buffer[65536], record[RECORD_BYTES];
	uint64_t remaining, piece;
	if (bytes == 0u || bytes > SPARK_WEIGHTD_EXPERT_BYTES_MAX ||
		fseeko(pack, (off_t)offset, SEEK_SET) != 0)
		return(-1);
	SparkCk128Initialize(&ck);
	remaining = bytes;
	while (remaining != 0u)
	{
		piece = remaining < sizeof(buffer) ? remaining : sizeof(buffer);
		if (fread(buffer, 1u, piece, pack) != piece)
			return(-2);
		SparkCk128Update(&ck, buffer, piece);
		remaining -= piece;
	}
	memset(record, 0u, sizeof(record));
	memcpy(record, &layer, 4u);
	memcpy(record + 4u, &expert, 4u);
	memcpy(record + 8u, &kind, 4u);
	memcpy(record + 16u, &offset, 8u);
	memcpy(record + 24u, &bytes, 8u);
	SparkCk128Finalize(&ck, record + 32u);
	return(fwrite(record, 1u, sizeof(record), out) == sizeof(record) ? 0 : -3);
}

static int32_t entry_write(FILE *pack, FILE *out, const PackHeader *header,
    const SparkQwen38_27bStagePackEntry *entry, uint32_t *count)
{
	uint64_t offset, bytes, directory_end;
	uint32_t plane, kind;
	int32_t err, which;
	for (which = 0; which < 3; which++)
		if (entry->tensor_kind == ffn_kinds[which])
			break;
	if (which == 3)
		return(0);
	/* The MTP layer's FFN stays in the spine: the lane is non-speculative
	 * and one draft layer of gate/up/down is not a routing tier. */
	if (entry->layer_index == SPARK_QWEN38_27B_STAGEPACK_GLOBAL_LAYER ||
		entry->layer_index == SPARK_QWEN38_27B_STAGEPACK_MTP_LAYER)
		return(0);
	if (entry->layer_index < header->first_layer ||
		entry->layer_index >= header->first_layer + header->layer_count)
		return(-4);
	directory_end = header->directory_offset +
		((uint64_t)header->tensor_count * SPARK_QWEN38_27B_STAGEPACK_ENTRY_BYTES);
	for (plane = 0u; plane < 2u; plane++)
	{
		offset = plane == 0u ? entry->payload_offset : entry->scale_offset;
		bytes = plane == 0u ? entry->payload_bytes : entry->scale_bytes;
		if (plane == 1u && bytes == 0u)
			continue;
		if (bytes == 0u || offset < directory_end || offset > header->file_bytes ||
			bytes > (header->file_bytes - offset))
			return(-5);
		if (*count == RANGE_MAX)
			return(-6);
		kind = (entry->tensor_kind * 2u) + plane;
		err = range_write(pack, out, entry->layer_index, 0u, kind, offset, bytes);
		if (err < 0)
			return(err);
		(*count)++;
	}
	return(0);
}

static int32_t manifest_write(FILE *pack, FILE *out, const PackHeader *header,
    uint64_t *expert_bytes, uint32_t *layers, uint32_t *ffn_format)
{
	SparkQwen38_27bStagePackEntry entry;
	uint64_t seen[SPARK_QWEN38_27B_MODEL_LAYER_COUNT] = {0};
	uint32_t words[4] = {SPARK_WEIGHTD_EXPERT_MANIFEST_MAGIC,
		SPARK_WEIGHTD_RANGE_MANIFEST_VERSION, 0u, 0u};
	uint32_t i;
	int32_t err;
	if (fwrite(words, 1u, sizeof(words), out) != sizeof(words))
		return(-7);	for (i = 0u; i < header->tensor_count; i++)
	{
		if (fseeko(pack, (off_t)(header->directory_offset +
			((uint64_t)i * SPARK_QWEN38_27B_STAGEPACK_ENTRY_BYTES)), SEEK_SET) != 0 ||
			fread(&entry, 1u, sizeof(entry), pack) != sizeof(entry))
			return(-8);
		if (entry.tensor_kind != SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_GATE &&
			entry.tensor_kind != SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_UP &&
			entry.tensor_kind != SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_DOWN)
			continue;
		if (entry.layer_index == SPARK_QWEN38_27B_STAGEPACK_GLOBAL_LAYER ||
			entry.layer_index == SPARK_QWEN38_27B_STAGEPACK_MTP_LAYER)
			continue;
		if (entry.layer_index >= SPARK_QWEN38_27B_MODEL_LAYER_COUNT)
			return(-9);
		if (entry.tensor_kind == SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_GATE)
			*ffn_format = entry.weight_format;
		if (entry.layer_index >= SPARK_QWEN38_27B_MODEL_LAYER_COUNT)
			return(-9);
		seen[entry.layer_index] |=
			UINT64_C(1) << (entry.tensor_kind - SPARK_QWEN38_27B_STAGEPACK_TENSOR_FFN_GATE);
		*expert_bytes += entry.payload_bytes + entry.scale_bytes;
		err = entry_write(pack, out, header, &entry, &words[2]);
		if (err < 0)
			return(err);
	}
	for (i = 0u; i < SPARK_QWEN38_27B_MODEL_LAYER_COUNT; i++)
	{
		if (seen[i] == 0u)
			continue;
		if (seen[i] != UINT64_C(7))
			return(-10);
		(*layers)++;
	}
	/* A dense whole-stack rank pack carries FFN for every pack layer. */
	if (*layers != header->layer_count)
		return(-11);
	if (words[2] == 0u || fseeko(out, 0, SEEK_SET) != 0 ||
		fwrite(words, 1u, sizeof(words), out) != sizeof(words))
		return(-12);
	return((int32_t)words[2]);
}

static int32_t header_parse(FILE *pack, PackHeader *header)
{
	SparkQwen38_27bStagePackHeader wire;
	struct stat st;
	uint64_t directory_bytes;
	if (fstat(fileno(pack), &st) != 0 || st.st_size < 0 ||
		fread(&wire, 1u, sizeof(wire), pack) != sizeof(wire))
		return(-13);
	if (wire.magic != SPARK_QWEN38_27B_STAGEPACK_MAGIC ||
		wire.format_version != SPARK_QWEN38_27B_STAGEPACK_FORMAT_VERSION ||
		wire.header_bytes != SPARK_QWEN38_27B_STAGEPACK_HEADER_BYTES ||
		wire.directory_entry_bytes != SPARK_QWEN38_27B_STAGEPACK_ENTRY_BYTES)
		return(-14);
	if (wire.file_bytes != (uint64_t)st.st_size || wire.tensor_count == 0u)
		return(-15);
	if (wire.first_layer_index >= SPARK_QWEN38_27B_MODEL_LAYER_COUNT ||
		wire.layer_count == 0u ||
		wire.layer_count > (SPARK_QWEN38_27B_MODEL_LAYER_COUNT - wire.first_layer_index))
		return(-16);
	if (wire.tp_degree == 0u || wire.tp_rank >= wire.tp_degree)
		return(-17);
	directory_bytes = (uint64_t)wire.tensor_count *
		SPARK_QWEN38_27B_STAGEPACK_ENTRY_BYTES;
	if (wire.directory_offset < SPARK_QWEN38_27B_STAGEPACK_HEADER_BYTES ||
		wire.directory_offset > wire.file_bytes ||
		directory_bytes > (wire.file_bytes - wire.directory_offset))
		return(-18);
	header->tensor_count = wire.tensor_count;
	header->first_layer = wire.first_layer_index;
	header->layer_count = wire.layer_count;
	header->tp_degree = wire.tp_degree;
	header->tp_rank = wire.tp_rank;
	header->directory_offset = wire.directory_offset;
	header->file_bytes = wire.file_bytes;
	return(0);
}

static int32_t manifest_publish(FILE *pack, const PackHeader *header,
    const char *path, uint64_t *expert_bytes, uint32_t *layers,
    uint32_t *ffn_format, uint32_t *ranges)
{
	SparkWeightdManifest manifest;
	char temporary[4096];
	FILE *out;
	int32_t fd, err, written;
	written = snprintf(temporary, sizeof(temporary), "%s.partial.XXXXXX", path);
	if (written < 0 || (uint32_t)written >= sizeof(temporary))
		return(-19);
	fd = mkstemp(temporary);
	if (fd < 0)
		return(-20);
	out = fdopen(fd, "wb");
	if (out == 0)
	{
		close(fd);
		unlink(temporary);
		return(-21);
	}
	err = manifest_write(pack, out, header, expert_bytes, layers, ffn_format);
	if (err >= 0)
		*ranges = (uint32_t)err;
	if (fflush(out) != 0 && err >= 0)
		err = -22;
	if (err >= 0 && fsync(fd) != 0)
		err = -23;
	if (fclose(out) != 0 && err >= 0)
		err = -24;
	if (err >= 0)
	{
		if (SparkWeightdManifestLoad(temporary, header->file_bytes,
			&manifest) != SPARK_STATUS_OK)
			err = -25;
		else
			SparkWeightdManifestDestroy(&manifest);
	}
	if (err >= 0 && link(temporary, path) != 0)
		err = errno == EEXIST ? -27 : -26;
	unlink(temporary);
	return(err < 0 ? err : 0);
}

int main(int argc, char **argv)
{
	PackHeader header;
	FILE *pack;
	char path[4096];
	uint64_t expert_bytes = 0u;
	uint32_t layers = 0u, ffn_format = 0u, ranges = 0u;
	int32_t err, written;
	if (argc != 2)
	{
		fprintf(stderr, "usage: %s <pack.q38sp|pack.qwen38_27bsp>\n", argv[0]);
		return(2);
	}
	written = snprintf(path, sizeof(path), "%s.experts", argv[1]);
	if (written < 0 || (uint32_t)written >= sizeof(path))
		return(3);
	pack = fopen(argv[1], "rb");
	if (pack == 0)
		return(4);
	err = header_parse(pack, &header);
	if (err == 0)
		err = manifest_publish(pack, &header, path, &expert_bytes, &layers,
			&ffn_format, &ranges);
	fclose(pack);
	if (err < 0)
	{
		fprintf(stderr, "qwen38_27b expert manifest failed: error=%d%s (existing output is preserved)\n",
			err, err == -27 ? " (stale .experts already present; remove it to regenerate)" : "");
		return(1);
	}
	printf("published %s version=%u ranges=%u layers=%u expert_bytes=%llu spine_bytes=%llu ffn_weight_format=%u tp=%u/%u\n",
		path, (unsigned)SPARK_WEIGHTD_RANGE_MANIFEST_VERSION, ranges, layers,
		(unsigned long long)expert_bytes,
		(unsigned long long)(header.file_bytes - expert_bytes), ffn_format,
		header.tp_rank, header.tp_degree);
	return(0);
}
