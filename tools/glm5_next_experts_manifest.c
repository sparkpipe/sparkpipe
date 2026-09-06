#define _POSIX_C_SOURCE 200809L

#include "sparkpipe/spark_ck128.h"
#include "sparkpipe/spark_status.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../modules/glm5_next_resident_decode_stage/source/spark_glm5_next_stagepack_format.h"

#define EXPERT_UP_GATE 22u
#define EXPERT_DOWN 23u
#define MANIFEST_MAGIC UINT32_C(0x58504557)
#define MANIFEST_VERSION 1u

static void put32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4u); }
static void put64(uint8_t *p, uint64_t v) { memcpy(p, &v, 8u); }

int main(int argc, char **argv)
{
    SparkGlm5NextStagePackHeader header;
    SparkGlm5NextStagePackEntry *entries;
    FILE *pack;
    FILE *out;
    uint8_t record[40];
    uint8_t *slice = 0;
    uint64_t slice_cap = 0u;
    uint32_t index, layer;
    uint32_t count = 0u;
    char path[4096];

    if (argc != 2)
    {
        fprintf(stderr, "usage: %s <pack.sp>\n", argv[0]);
        return 2;
    }
    pack = fopen(argv[1], "rb");
    if (pack == 0)
    {
        perror("open pack");
        return 1;
    }
    if (fread(&header, 1u, sizeof(header), pack) != sizeof(header) ||
        header.magic != SPARK_GLM5_NEXT_STAGEPACK_MAGIC)
    {
        fprintf(stderr, "bad pack header\n");
        return 1;
    }
    entries = calloc(header.tensor_count, sizeof(*entries));
    if (entries == 0)
        return 1;
    if (fseeko(pack, (off_t)header.directory_offset, SEEK_SET) != 0 ||
        fread(entries, sizeof(*entries), header.tensor_count, pack) !=
            header.tensor_count)
    {
        fprintf(stderr, "bad directory\n");
        return 1;
    }
    for (index = 0u; index < header.tensor_count; index++)
        if (entries[index].tensor_kind == EXPERT_UP_GATE ||
            entries[index].tensor_kind == EXPERT_DOWN)
            count += header.routed_expert_count;
    if (count == 0u)
    {
        fprintf(stderr, "no expert tensors\n");
        return 1;
    }
    snprintf(path, sizeof(path), "%s.experts", argv[1]);
    out = fopen(path, "wb");
    if (out == 0)
    {
        perror("open manifest");
        return 1;
    }
    put32(record + 0u, MANIFEST_MAGIC);
    put32(record + 4u, MANIFEST_VERSION);
    put32(record + 8u, count);
    put32(record + 12u, 0u);
    fwrite(record, 1u, 16u, out);
    for (index = 0u; index < header.tensor_count; index++)
    {
        SparkGlm5NextStagePackEntry *entry = &entries[index];
        uint64_t per;
        uint32_t expert;
        SparkCk128Context ck;

        if (entry->tensor_kind != EXPERT_UP_GATE &&
            entry->tensor_kind != EXPERT_DOWN)
            continue;
        layer = entry->layer_index;
        per = entry->payload_bytes / header.routed_expert_count;
        if (per == 0u || per * header.routed_expert_count !=
                entry->payload_bytes)
        {
            fprintf(stderr, "kind %u layer %u payload %llu not divisible\n",
                entry->tensor_kind, layer,
                (unsigned long long)entry->payload_bytes);
            return 1;
        }
        if (per > slice_cap)
        {
            free(slice);
            slice = malloc(per);
            slice_cap = per;
            if (slice == 0)
                return 1;
        }
        for (expert = 0u; expert < header.routed_expert_count; expert++)
        {
            uint64_t offset = entry->payload_offset +
                (uint64_t)expert * per;
            uint8_t digest[16];

            if (fseeko(pack, (off_t)offset, SEEK_SET) != 0 ||
                fread(slice, 1u, per, pack) != per)
            {
                fprintf(stderr, "read slice layer %u expert %u failed\n",
                    layer, expert);
                return 1;
            }
            SparkCk128Initialize(&ck);
            SparkCk128Update(&ck, slice, per);
            SparkCk128Finalize(&ck, digest);
            put32(record + 0u, layer);
            put32(record + 4u, expert);
            put64(record + 8u, offset);
            put64(record + 16u, per);
            memcpy(record + 24u, digest, 16u);
            fwrite(record, 1u, 40u, out);
        }
    }
    fclose(out);
    fclose(pack);
    printf("%s: %u experts\n", path, count);
    return 0;
}
