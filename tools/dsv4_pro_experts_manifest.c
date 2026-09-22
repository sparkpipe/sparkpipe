#define _POSIX_C_SOURCE 200809L

#include "sparkpipe/spark_ck128.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_manifest.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/*
 * DSV4 Pro routed-expert range manifest producer (lane 5, shared multidev).
 *
 * Writes <pack>.experts next to a deployed TP4xPP4 rank pack: the version-2
 * ck128 range manifest the lazy weightd tier requires for shared-socket
 * attach (runtime/spark_weightd.c rejects the attach without it).
 *
 * DSV4 expert tensors are tensor-parallel INSIDE every expert: the rank pack
 * carries, for each of the 384 experts, its TP4 slice of W1/W3 (rows
 * EXPERT_WIDTH/4) and its column shard of W2. The directory entry is
 * expert-uniform, so per-expert ranges are the entry sliced into 384 equal
 * spans per plane (payload, scales). The replicated DSpark draft layers
 * (layer markers 0xFFFFFFFB..FD) are full-width and are included for
 * completeness; non-speculative work never leases them.
 *
 * Usage: dsv4_pro_experts_manifest <pack> [<out>]   (default out: pack.experts)
 */

#define DSV4_PACK_MAGIC UINT32_C(0x34565344)
#define DSV4_WEIGHT_FP4 3u
#define DSV4_KIND_EXPERTS_W1 19u
#define DSV4_KIND_EXPERTS_W2 20u
#define DSV4_KIND_EXPERTS_W3 21u
#define DSV4_DRAFT_LAYER_FIRST UINT32_C(0xFFFFFFFB)
#define DSV4_DRAFT_LAYER_LAST UINT32_C(0xFFFFFFFD)

typedef struct
{
	uint32_t u32[16];
	uint64_t u64[2];
} Dsv4PackHeader;

typedef struct
{
	uint32_t kind;
	uint32_t layer;
	uint32_t weight;
	uint32_t rows;
	uint32_t columns;
	uint32_t reserved;
	uint64_t payload_offset;
	uint64_t scale_offset;
} Dsv4PackEntry;

/* Range kind = tensor kind * 2 + plane (0 payload, 1 scale). */
static int32_t range_write(FILE *pack, FILE *out, uint32_t layer,
    uint32_t expert, uint32_t kind, uint64_t offset, uint64_t bytes)
{
    SparkCk128Context ck;
    uint8_t buffer[65536], record[48] = {0};
    uint64_t remaining, piece;
    if ( bytes == 0u || bytes > SPARK_WEIGHTD_EXPERT_BYTES_MAX ||
        fseeko(pack,(off_t)offset,SEEK_SET) != 0 )
        return(-1);
    SparkCk128Initialize(&ck);
    remaining = bytes;
    while ( remaining != 0u )
    {
        piece = remaining < sizeof(buffer) ? remaining : sizeof(buffer);
        if ( fread(buffer,1u,piece,pack) != piece )
            return(-2);
        SparkCk128Update(&ck,buffer,piece);
        remaining -= piece;
    }
    memcpy(record,&layer,4u);
    memcpy(record + 4u,&expert,4u);
    memcpy(record + 8u,&kind,4u);
    memcpy(record + 16u,&offset,8u);
    memcpy(record + 24u,&bytes,8u);
    SparkCk128Finalize(&ck,record + 32u);
    return(fwrite(record,1u,sizeof(record),out) == sizeof(record) ? 0 : -3);
}

static int32_t entry_write(FILE *pack, FILE *out, const Dsv4PackHeader *header,
    const Dsv4PackEntry *entry, uint32_t experts, uint32_t *count,
    uint64_t *seen_layers)
{
    uint64_t directory_end, per[2];
    uint32_t plane, expert, kind;
    int32_t err;
    if ( entry->kind != DSV4_KIND_EXPERTS_W1 && entry->kind != DSV4_KIND_EXPERTS_W2 &&
        entry->kind != DSV4_KIND_EXPERTS_W3 )
        return(0);
    if ( entry->weight != DSV4_WEIGHT_FP4 )
        return(-4);
    if ( experts == 0u || entry->rows == 0u || entry->columns == 0u ||
        (entry->rows % experts) != 0u || (entry->columns % 32u) != 0u )
        return(-5);
    directory_end = (header->u64[0] + ((uint64_t)header->u32[8] * sizeof(*entry)));
    for ( plane = 0u; plane < 2u; plane++ )
    {
        uint64_t offset = plane == 0u ? entry->payload_offset : entry->scale_offset;
        uint64_t total = plane == 0u
            ? ((uint64_t)entry->rows * ((uint64_t)entry->columns / 2u))
            : ((uint64_t)entry->rows * ((uint64_t)entry->columns / 32u));
        if ( total == 0u || (total % experts) != 0u || offset < directory_end ||
            offset > header->u64[1] || total > (header->u64[1] - offset) )
            return(-6);
        per[plane] = total / experts;
    }
    if ( entry->scale_offset != 0u && per[1] == 0u )
        return(-7);
    if ( entry->layer < 64u )
        *seen_layers |= (UINT64_C(1) << entry->layer);
    else if ( entry->layer < DSV4_DRAFT_LAYER_FIRST || entry->layer > DSV4_DRAFT_LAYER_LAST )
        return(-8); /* unexpected layer marker */
    for ( plane = 0u; plane < 2u; plane++ )
    {
        uint64_t offset = plane == 0u ? entry->payload_offset : entry->scale_offset;
        if ( plane == 1u && offset == 0u )
            continue;
        kind = ((entry->kind * 2u) + plane);
        for ( expert = 0u; expert < experts; expert++ )
        {
            if ( *count == SPARK_WEIGHTD_RANGE_COUNT_MAX )
                return(-9);
            err = range_write(pack,out,entry->layer,expert,kind,
                (offset + ((uint64_t)expert * per[plane])),per[plane]);
            if ( err < 0 )
                return(err);
            *count += 1u;
        }
    }
    return(0);
}

static int32_t manifest_write(FILE *pack, FILE *out, const Dsv4PackHeader *header,
    uint32_t experts)
{
    Dsv4PackEntry entry;
    uint32_t words[4] = {SPARK_WEIGHTD_EXPERT_MANIFEST_MAGIC,
        SPARK_WEIGHTD_RANGE_MANIFEST_VERSION,0u,0u};
    uint32_t i, draft_entries = 0u, expert_entries = 0u;
    uint64_t seen_layers = 0u, expected_layers = 0u;
    uint32_t first = header->u32[9], count = header->u32[10];
    int32_t err;
    if ( fwrite(words,1u,sizeof(words),out) != sizeof(words) )
        return(-10);
    for ( i = first; i < (first + count); i++ )
        expected_layers |= (UINT64_C(1) << i);
    for ( i = 0u; i < header->u32[8]; i++ )
    {
        if ( fseeko(pack,(off_t)(header->u64[0] + ((uint64_t)i * sizeof(entry))),SEEK_SET) != 0 ||
            fread(&entry,1u,sizeof(entry),pack) != sizeof(entry) )
            return(-11);
        if ( entry.kind != DSV4_KIND_EXPERTS_W1 && entry.kind != DSV4_KIND_EXPERTS_W2 &&
            entry.kind != DSV4_KIND_EXPERTS_W3 )
            continue;
        if ( entry.layer >= DSV4_DRAFT_LAYER_FIRST && entry.layer <= DSV4_DRAFT_LAYER_LAST )
            draft_entries++;
        else
            expert_entries++;
        err = entry_write(pack,out,header,&entry,experts,&words[2],&seen_layers);
        if ( err < 0 )
            return(err);
    }
    /* Every backbone layer of this stage carries all three expert kinds,
       and the replicated DSpark block carries all three draft layers. */
    if ( seen_layers != expected_layers || draft_entries != 9u ||
        expert_entries != (count * 3u) || words[2] == 0u )
        return(-12);
    if ( fseeko(out,0,SEEK_SET) != 0 || fwrite(words,1u,sizeof(words),out) != sizeof(words) )
        return(-13);
    fprintf(stderr,"dsv4_pro_experts_manifest: ranges=%u groups=%u experts=%u "
        "backbone_layers=%u draft_layers=3\n",
        words[2],(words[2] / 6u),experts,count);
    return(0);
}

int main(int argument_count, char **arguments)
{
    Dsv4PackHeader header;
    struct stat info;
    FILE *pack = 0, *out = 0;
    const char *pack_path, *out_path;
    uint32_t experts;
    int32_t err;
    if ( argument_count < 2 || argument_count > 3 )
    {
        fprintf(stderr,"usage: %s <pack> [<out>]\n",arguments[0]);
        return(2);
    }
    pack_path = arguments[1];
    out_path = argument_count == 3 ? arguments[2] : 0;
    if ( stat(pack_path,&info) != 0 || !S_ISREG(info.st_mode) )
    {
        fprintf(stderr,"dsv4_pro_experts_manifest: cannot stat %s\n",pack_path);
        return(3);
    }
    pack = fopen(pack_path,"rb");
    if ( pack == 0 || fread(&header,1u,sizeof(header),pack) != sizeof(header) )
    {
        fprintf(stderr,"dsv4_pro_experts_manifest: cannot read header\n");
        if ( pack != 0 ) fclose(pack);
        return(3);
    }
    if ( header.u32[0] != DSV4_PACK_MAGIC || header.u32[11] != 61u ||
        header.u32[14] == 0u || header.u32[8] == 0u ||
        header.u64[1] != (uint64_t)info.st_size )
    {
        fprintf(stderr,"dsv4_pro_experts_manifest: not a DSV4 Pro stage pack\n");
        fclose(pack);
        return(3);
    }
    experts = header.u32[14];
    if ( (uint64_t)header.u32[8] * sizeof(Dsv4PackEntry) > header.u64[1] )
    {
        fclose(pack);
        return(3);
    }
    if ( out_path == 0 )
    {
        static char path[4096 + 16];
        if ( snprintf(path,sizeof(path),"%s.experts",pack_path) >= (int)sizeof(path) )
        {
            fclose(pack);
            return(3);
        }
        out_path = path;
    }
    out = fopen(out_path,"wb");
    if ( out == 0 )
    {
        fprintf(stderr,"dsv4_pro_experts_manifest: cannot write %s\n",out_path);
        fclose(pack);
        return(4);
    }
    err = manifest_write(pack,out,&header,experts);
    if ( err < 0 )
        fprintf(stderr,"dsv4_pro_experts_manifest: FAILED (%d)\n",(int)err);
    if ( fclose(out) != 0 )
        err = -14;
    fclose(pack);
    if ( err < 0 )
        unlink(out_path);
    return(err < 0 ? 1 : 0);
}
