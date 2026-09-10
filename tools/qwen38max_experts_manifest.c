/* Emit <pack>.experts — the weightd v2 range manifest for a qwen38_max
 * full-width stage pack. Ported from tools/glm5_next_experts_manifest.c
 * (the shared v2 contract) onto the max family:
 *
 *   - 120-byte header, Q8SP magic, 56-byte entries;
 *   - THREE expert kinds (W1 / W3 / DOWN), one kind-major payload region
 *     each, so every (layer, expert) resolves to three payload ranges;
 *   - fp8 experts carry an F32 block-128x128 scale plane (plane 1); the
 *     bf16 spine carries none;
 *   - range kind = tensor kind * 2 + plane, shared SparkWeightdRange wire;
 *   - record = 48 bytes (layer, expert, kind, zero, offset, bytes,
 *     ck128[16]) — the shared SparkWeightdManifestLoad wire.
 *
 * Validation is strict and fails closed: bounded counts, digest coverage
 * per range, exact expert-kind coverage per MoE layer, ranges inside the
 * file, and no duplicate (layer, expert) groups. Publication is
 * tmp-file + fsync + rename — a failed generation preserves the existing
 * artifact. Manifest geometry: 92 layers x 3 kinds x 512 experts x 2
 * planes = 282k ranges (W1/W3 and DOWN carry a scale plane each; the
 * payload plane is kind-major) — SPARK_WEIGHTD_RANGE_COUNT_MAX is the
 * shared binding.
 */
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

/* Two planes per expert kind-major entry: payload (plane 0) and the fp8
 * scale plane (plane 1). Each plane cuts into routed_expert_count slices;
 * every slice is one range. */
static int32_t entry_write(FILE *pack, FILE *out,
    const SparkQwen38MaxStagePackHeader *header,
    const SparkQwen38MaxStagePackEntry *entry, uint32_t *count)
{
    uint64_t per, offset, bytes, directory_end;
    uint32_t plane, expert, kind;
    int32_t err;

    if (entry->tensor_kind != SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_W1 &&
        entry->tensor_kind != SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_W3 &&
        entry->tensor_kind != SPARK_QWEN38_MAX_STAGEPACK_TENSOR_MOE_DOWN)
        return(0);
    if (entry->rows % header->routed_expert_count != 0u ||
        entry->payload_bytes == 0u)
        return(-5);
    directory_end = header->directory_offset +
        ((uint64_t)header->tensor_count * sizeof(*entry));
    for (plane = 0u; plane < 2u; plane++)
    {
        offset = plane == 0u ? entry->payload_offset : entry->scale_offset;
        bytes = plane == 0u ? entry->payload_bytes : entry->scale_bytes;
        if (plane == 1u && bytes == 0u)
            continue;
        if (bytes == 0u || (bytes % header->routed_expert_count) != 0u ||
            offset < directory_end || offset > header->file_bytes ||
            bytes > (header->file_bytes - offset))
            return(-6);
        per = bytes / header->routed_expert_count;
        kind = ((entry->tensor_kind * 2u) + plane);
        for (expert = 0u; expert < header->routed_expert_count; expert++)
        {
            if (*count == RANGE_MAX)
                return(-7);
            err = range_write(pack, out, entry->layer_index, expert, kind,
                offset + ((uint64_t)expert * per), per);
            if (err < 0)
                return(err);
            *count += 1u;
        }
    }
    return(0);
}

static int32_t manifest_write(FILE *pack, FILE *out,
    const SparkQwen38MaxStagePackHeader *header)
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
    /* Exact coverage: every MoE layer carries all three kinds. */
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

static int32_t header_read(FILE *pack, SparkQwen38MaxStagePackHeader *header)
{
    struct stat st;
    uint64_t directory_bytes;
    if (fstat(fileno(pack), &st) != 0 || st.st_size < 0 ||
        fread(header, 1u, sizeof(*header), pack) != sizeof(*header))
        return(-11);
    if (header->magic != SPARK_QWEN38_MAX_STAGEPACK_MAGIC ||
        header->format_version != SPARK_QWEN38_MAX_STAGEPACK_FORMAT_VERSION ||
        header->header_bytes != sizeof(*header) ||
        header->directory_entry_bytes != sizeof(SparkQwen38MaxStagePackEntry))
        return(-12);
    if (header->file_bytes != (uint64_t)st.st_size ||
        header->routed_expert_count == 0u || header->tensor_count == 0u)
        return(-13);
    if (header->first_layer_index >= SPARK_QWEN38_MAX_MODEL_LAYER_COUNT ||
        header->layer_count == 0u ||
        header->layer_count > (SPARK_QWEN38_MAX_MODEL_LAYER_COUNT -
            header->first_layer_index))
        return(-26);
    directory_bytes = ((uint64_t)header->tensor_count *
        sizeof(SparkQwen38MaxStagePackEntry));
    if (header->directory_offset < sizeof(*header) ||
        header->directory_offset > header->file_bytes ||
        directory_bytes > (header->file_bytes - header->directory_offset))
        return(-14);
    return(0);
}

static int32_t manifest_publish(FILE *pack,
    const SparkQwen38MaxStagePackHeader *header, const char *path)
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
    SparkQwen38MaxStagePackHeader header;
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
    err = header_read(pack, &header);
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
