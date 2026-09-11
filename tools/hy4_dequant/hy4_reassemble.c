/* hy4 lane: reassemble the 16 TP16 rank bundles into one full GGUF.
 *
 * The exact inverse of tools/hy4_tp16_shard.py: for every source tensor
 * the manifests record how each rank sliced it, so concatenating rank
 * slices in rank order reproduces the original tensor bytes. Uses the
 * rank loader for offsets; emits rank-0's header followed by all tensors
 * in offset order with rank-slab data concatenated per tensor.
 *
 * Verification is external: sha256 of the output must equal the Hub LFS
 * oid recorded in model_contracts/hy4_authoritative.json.
 *
 * Usage: hy4_reassemble <allranks_dir> <out.gguf>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "hy4_rank_loader.h"

#define RANKS 16
#define CH (1 << 24)

static void copy_range(FILE *in, long off, long nbytes, FILE *out,
                       unsigned char *buf) {
    if (fseek(in, off, SEEK_SET)) { fprintf(stderr, "seek fail\n"); exit(3); }
    long remaining = nbytes;
    while (remaining > 0) {
        long want = remaining < CH ? remaining : CH;
        long got = (long)fread(buf, 1, (size_t)want, in);
        if (got != want) { fprintf(stderr, "short read\n"); exit(3); }
        fwrite(buf, 1, (size_t)got, out);
        remaining -= got;
    }
}

int main(int argc, char **argv) {
    const char *base = argv[1];
    hy4_rank *R[RANKS];
    char dir[300], pack[400];
    for (int r = 0; r < RANKS; ++r) {
        snprintf(dir, sizeof(dir), "%s/rank-%02d", base, r);
        if (hy4_rank_open(dir, 0, &R[r])) {
            fprintf(stderr, "open rank %d failed\n", r);
            return 1;
        }
    }
    snprintf(pack, sizeof(pack), "%s/rank-00/model-ud-iq1m-tp16-rank-00.gguf", base);

    /* the output header must be the ORIGINAL source header (full tensor
     * dims); rank files carry sliced dims, so take it from the captured
     * original-header file (argv[3]) */
    if (argc != 4) { fprintf(stderr, "usage: %s allranks_dir out.gguf orig_header\n", argv[0]); return 2; }
    FILE *hf = fopen(argv[3], "rb");
    if (!hf) { fprintf(stderr, "orig header missing\n"); return 1; }
    long header_len = 5051520; /* validated: loader data_offset for rank files */
    FILE *out = fopen(argv[2], "wb");
    if (!out) { fprintf(stderr, "out open failed\n"); return 1; }
    unsigned char *hdr = malloc((size_t)header_len);
    if (fread(hdr, 1, (size_t)header_len, hf) != (size_t)header_len) return 3;
    fclose(hf);
    fwrite(hdr, 1, (size_t)header_len, out);
    fprintf(stderr, "header %ld bytes from original\n", header_len);
    (void)pack;

    /* tensors in offset order, per rank: the loader stores views in name
     * order; rank files were written in offset order with matching info
     * order, so walk rank-0 views sorted by file_offset */
    int *order = malloc(sizeof(int) * R[0]->tensor_count);
    for (int i = 0; i < R[0]->tensor_count; ++i) order[i] = i;
    for (int a = 1; a < R[0]->tensor_count; ++a) {
        int key = order[a], b = a - 1;
        while (b >= 0 && R[0]->views[order[b]].file_offset >
               R[0]->views[key].file_offset) { order[b + 1] = order[b]; b--; }
        order[b + 1] = key;
    }
    unsigned char *buf = malloc(CH);
    for (int i = 0; i < R[0]->tensor_count; ++i) {
        const hy4_tensor_view *tv = &R[0]->views[order[i]];
        long pad = (32 - tv->nbytes % 32) % 32;
        if (tv->slice_kind == 0) {
            /* replicated: one full copy (rank 0's), padded once */
            copy_range((FILE *)R[0]->file, tv->file_offset, tv->nbytes, out, buf);
            for (long p = 0; p < pad; ++p) fputc(0, out);
            continue;
        }
        /* sliced: rank slabs concatenate to the original tensor bytes with
         * no inter-slab padding (each slab is a whole-block partition) */
        for (int r = 0; r < RANKS; ++r) {
            const hy4_tensor_view *tvr = &R[r]->views[order[i]];
            copy_range((FILE *)R[r]->file, tvr->file_offset, tvr->nbytes, out, buf);
        }
        for (long p = 0; p < pad; ++p) fputc(0, out);
        if (i % 200 == 0)
            fprintf(stderr, "tensor %d/%d (%s)\n", i, R[0]->tensor_count, tv->name);
    }
    fclose(out);
    printf("REASSEMBLED %s (header %ld + %d tensors x %d ranks)\n",
           argv[2], header_len, R[0]->tensor_count, RANKS);
    printf("VERIFY: sha256 %s must equal the Hub LFS oid\n", argv[2]);
    return 0;
}
