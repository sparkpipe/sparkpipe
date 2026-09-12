#define _POSIX_C_SOURCE 200809L

#include "sparkpipe/spark_ck128.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_manifest.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../modules/qwen38_max_resident_decode_stage/source/spark_qwen38_max_stagepack_format.h"

#define RANGE_MAX SPARK_WEIGHTD_RANGE_COUNT_MAX
#define WIRE1_BYTES 120u
#define WIRE2_BYTES 128u

typedef struct
{
	uint32_t u32[26];
	uint64_t u64[2];
} WireHeader1;

typedef struct
{
	uint32_t u32[28];
	uint64_t u64[2];
} WireHeader2;

typedef struct
{
	uint32_t tensor_count;
	uint32_t routed_expert_count;
	uint32_t expert_intermediate;
	uint32_t hidden;
	uint32_t first_layer;
	uint32_t layer_count;
	uint32_t tp_degree;
	uint32_t tp_rank;
	uint64_t directory_offset;
	uint64_t file_bytes;
} PackHeader;

static int32_t range_write(FILE *pack, FILE *out, uint32_t layer,
    uint32_t expert, uint32_t kind, uint64_t offset, uint64_t bytes)
{
    SparkCk128Context ck;
    uint8_t buffer[65536], record[48] = {0};
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
    memcpy(record, &layer, 4u);
    memcpy(record + 4u, &expert, 4u);
    memcpy(record + 8u, &kind, 4u);
    memcpy(record + 16u, &offset, 8u);
    memcpy(record + 24u, &bytes, 8u);
    SparkCk128Finalize(&ck, record + 32u);
    return(fwrite(record, 1u, sizeof(record), out) == sizeof(record) ? 0 : -3);
}

static int32_t entry_write(FILE *pack, FILE *out, const PackHeader *header,
    const SparkQwen38MaxStagePackEntry *entry, uint32_t *count)
{
    uint64_t per, offset, bytes, directory_end;
    uint32_t plane, expert, kind, expert_rows, resident, per_rank, base;
    int32_t err;
    if (entry->tensor_kind != SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_W1 &&
        entry->tensor_kind != SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_W3 &&
        entry->tensor_kind != SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_DOWN)
        return(0);
    expert_rows = entry->tensor_kind == SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_DOWN
        ? header->hidden : header->expert_intermediate;
    if (expert_rows == 0u || entry->rows == 0u ||
        entry->rows % expert_rows != 0u || entry->payload_bytes == 0u)
        return(-5);
    resident = entry->rows / expert_rows;
    per_rank = header->routed_expert_count / header->tp_degree;
    if (resident != per_rank)
        return(-28);
    base = header->tp_rank * per_rank;
    directory_end = header->directory_offset +
        ((uint64_t)header->tensor_count * sizeof(SparkQwen38MaxStagePackEntry));
    for (plane = 0u; plane < 2u; plane++)
    {
        offset = plane == 0u ? entry->payload_offset : entry->scale_offset;
        bytes = plane == 0u ? entry->payload_bytes : entry->scale_bytes;
        if (plane == 1u && bytes == 0u)
            continue;
        if (bytes == 0u || (bytes % resident) != 0u ||
            offset < directory_end || offset > header->file_bytes ||
            bytes > (header->file_bytes - offset))
            return(-6);
        per = bytes / resident;
        kind = ((entry->tensor_kind * 2u) + plane);
        for (expert = 0u; expert < resident; expert++)
        {
            if (*count == RANGE_MAX)
                return(-7);
            err = range_write(pack, out, entry->layer_index, base + expert,
                kind, offset + ((uint64_t)expert * per), per);
            if (err < 0)
                return(err);
            *count += 1u;
        }
    }
    return(0);
}

static int32_t manifest_write(FILE *pack, FILE *out, const PackHeader *header)
{
    SparkQwen38MaxStagePackEntry entry;
    uint32_t words[4] = {SPARK_WEIGHTD_EXPERT_MANIFEST_MAGIC,
        SPARK_WEIGHTD_RANGE_MANIFEST_VERSION, 0u, 0u};
    uint32_t i, complete_layers = 0u;
    uint64_t seen[SPARK_QWEN38_MAX_MODEL_LAYER_COUNT] = {0};
    int32_t err;
    if (fwrite(words, 1u, sizeof(words), out) != sizeof(words))
        return(-8);
    for (i = 0u; i < header->tensor_count; i++)
    {
        if (fseeko(pack, (off_t)(header->directory_offset +
            ((uint64_t)i * sizeof(entry))), SEEK_SET) != 0 ||
            fread(&entry, 1u, sizeof(entry), pack) != sizeof(entry))
            return(-9);
        if (entry.tensor_kind != SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_W1 &&
            entry.tensor_kind != SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_W3 &&
            entry.tensor_kind != SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_DOWN)
            continue;
        if (entry.layer_index >= SPARK_QWEN38_MAX_MODEL_LAYER_COUNT)
            return(-23);
        seen[entry.layer_index] |=
            (UINT64_C(1) << (entry.tensor_kind
                - SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_W1));
        err = entry_write(pack, out, header, &entry, &words[2]);
        if (err < 0)
            return(err);
    }
    for (i = 0u; i < SPARK_QWEN38_MAX_MODEL_LAYER_COUNT; i++)
    {
        if (seen[i] == 0u)
            continue;
        if (seen[i] != UINT64_C(7))
            return(-24);
        complete_layers++;
    }
    if (complete_layers == 0u || words[2] == 0u ||
        fseeko(out, 0, SEEK_SET) != 0 ||
        fwrite(words, 1u, sizeof(words), out) != sizeof(words))
        return(-10);
    return(0);
}

static int32_t header_parse(FILE *pack, PackHeader *header)
{
    struct stat st;
    uint8_t raw[WIRE2_BYTES];
    uint32_t version, entry_bytes;
    uint64_t directory_bytes;
    if (fstat(fileno(pack), &st) != 0 || st.st_size < 0 ||
        fread(raw, 1u, sizeof(raw), pack) != sizeof(raw))
        return(-11);
    memcpy(&version, raw + 4u, 4u);
    memcpy(&entry_bytes, raw + 12u, 4u);
    if (version == 1u)
    {
        WireHeader1 wire;
        if (entry_bytes != sizeof(SparkQwen38MaxStagePackEntry))
            return(-12);
        memcpy(&wire, raw, WIRE1_BYTES);
        if (wire.u32[2] != WIRE1_BYTES ||
            wire.u64[0] != WIRE1_BYTES)
            return(-12);
        header->tensor_count = wire.u32[4];
        header->routed_expert_count = wire.u32[20];
        header->expert_intermediate = wire.u32[22];
        header->hidden = wire.u32[5];
        header->first_layer = wire.u32[7];
        header->layer_count = wire.u32[6];
        header->tp_degree = 1u;
        header->tp_rank = 0u;
        header->directory_offset = wire.u64[0];
        header->file_bytes = wire.u64[1];
    }
    else
    {
        WireHeader2 wire;
        memcpy(&wire, raw, WIRE2_BYTES);
        if (wire.u32[1] != 2u || wire.u32[2] != WIRE2_BYTES ||
            entry_bytes != sizeof(SparkQwen38MaxStagePackEntry) ||
            wire.u64[0] != WIRE2_BYTES)
            return(-12);
        header->tensor_count = wire.u32[4];
        header->routed_expert_count = wire.u32[20];
        header->expert_intermediate = wire.u32[22];
        header->hidden = wire.u32[5];
        header->first_layer = wire.u32[7];
        header->layer_count = wire.u32[6];
        header->tp_degree = wire.u32[26];
        header->tp_rank = wire.u32[27];
        header->directory_offset = wire.u64[0];
        header->file_bytes = wire.u64[1];
        if (header->tp_degree == 0u || header->tp_rank >= header->tp_degree)
            return(-27);
    }
    if (header->routed_expert_count == 0u ||
        header->routed_expert_count % header->tp_degree != 0u)
        return(-28);
    if (header->file_bytes != (uint64_t)st.st_size ||
        header->tensor_count == 0u)
        return(-13);
    if (header->first_layer >= SPARK_QWEN38_MAX_MODEL_LAYER_COUNT ||
        header->layer_count == 0u ||
        header->layer_count > (SPARK_QWEN38_MAX_MODEL_LAYER_COUNT -
            header->first_layer))
        return(-26);
    directory_bytes = ((uint64_t)header->tensor_count *
        sizeof(SparkQwen38MaxStagePackEntry));
    if (header->directory_offset < (version == 1u ? WIRE1_BYTES : WIRE2_BYTES) ||
        header->directory_offset > header->file_bytes ||
        directory_bytes > (header->file_bytes - header->directory_offset))
        return(-14);
    return(0);
}

static int32_t manifest_publish(FILE *pack, const PackHeader *header,
    const char *path)
{
    SparkWeightdManifest manifest;
    char temporary[4096];
    FILE *out;
    int32_t fd, err, written;
    written = snprintf(temporary, sizeof(temporary), "%s.partial.XXXXXX", path);
    if (written < 0 || (uint32_t)written >= sizeof(temporary))
        return(-15);
    fd = mkstemp(temporary);
    if (fd < 0)
        return(-16);
    out = fdopen(fd, "wb");
    if (out == 0)
    {
        close(fd);
        unlink(temporary);
        return(-17);
    }
    err = manifest_write(pack, out, header);
    if (fflush(out) != 0 && err == 0)
        err = -18;
    if (err == 0 && fsync(fd) != 0)
        err = -19;
    if (fclose(out) != 0 && err == 0)
        err = -20;
    if (err == 0)
    {
        if (SparkWeightdManifestLoad(temporary, header->file_bytes,
            &manifest) != SPARK_STATUS_OK)
            err = -21;
        else
            SparkWeightdManifestDestroy(&manifest);
    }
    if (err == 0 && link(temporary, path) != 0)
        err = -22;
    unlink(temporary);
    return(err);
}

int main(int argc, char **argv)
{
    PackHeader header;
    FILE *pack;
    char path[4096];
    int32_t err, written;
    if (argc != 2)
    {
        fprintf(stderr, "usage: %s <pack.qwen38sp>\n", argv[0]);
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
        err = manifest_publish(pack, &header, path);
    fclose(pack);
    if (err < 0)
        fprintf(stderr, "expert manifest failed: error=%d (existing output is preserved)\n", err);
    else
        printf("published %s version=%u ranges=%u\n", path,
            SPARK_WEIGHTD_RANGE_MANIFEST_VERSION, 0u);
    return(err < 0 ? 1 : 0);
}
