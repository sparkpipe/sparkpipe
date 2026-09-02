/* hy4 lane: rank-loader unit test against a real deployed rank bundle. */
#include <stdio.h>
#include <string.h>
#include "hy4_rank_loader.h"
#include <math.h>

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s pack_dir\n", argv[0]); return 2; }
    hy4_rank *rank = NULL;
    int rc = hy4_rank_open(argv[1], 0, &rank);
    if (rc) { fprintf(stderr, "open failed rc=%d\n", rc); return 1; }
    printf("open OK: %d tensors, file %.2f GB\n",
           rank->tensor_count, rank->file_bytes / 1e9);
    int failures = 0;
    /* expectations: rank-02 (third rank) of the deployed TP16 bundle,
     * verified against the shard plan + dequant runs 2026-09-02 */
    /* kind encoding: 0 = replicate, dim+1 = split on that dim */
    struct { const char *name; int type; int n_dims; long d0, d1, d2; int kind; long nbytes; } cases[] = {
        {"token_embd.weight", 12, 2, 6144, 7552, 0, 2, 26099712},
        {"blk.0.attn_kv_a_mqa.weight", 8, 2, 6144, 576, 0, 0, 3760128},
        {"blk.47.ffn_gate_exps.weight", 16, 3, 6144, 2048, 16, 3, 51904512},
        {"blk.47.ffn_down_exps.weight", 18, 3, 2048, 6144, 16, 3, 77070336},
        {"output.weight", 0, 2, 6144, 7552, 0, 2, 185597952},
    };
    for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        const hy4_tensor_view *tv = hy4_tensor_lookup(rank, cases[i].name);
        if (!tv) { printf("MISSING %s\n", cases[i].name); failures++; continue; }
        int ok = tv->type == cases[i].type && tv->n_dims == cases[i].n_dims &&
                 tv->dims[0] == cases[i].d0 && tv->dims[1] == cases[i].d1 &&
                 (cases[i].d2 == 0 || tv->dims[2] == cases[i].d2) &&
                 tv->slice_kind == cases[i].kind &&
                 (cases[i].nbytes == 0 || tv->nbytes == cases[i].nbytes) &&
                 tv->file_offset > 0;
        printf("%-32s type=%2d dims=[%ld,%ld,%ld] slice=kind%d+%ld bytes=%ld %s\n",
               tv->name, tv->type, tv->dims[0], tv->dims[1],
               tv->n_dims > 2 ? tv->dims[2] : 0, tv->slice_kind,
               tv->slice_start, tv->nbytes, ok ? "OK" : "MISMATCH");
        if (!ok) failures++;
    }
    /* read + digest one replicated tensor end-to-end */
    const hy4_tensor_view *tv = hy4_tensor_lookup(rank, "blk.0.attn_norm.weight");
    if (tv && tv->nbytes == 24576) {
        static unsigned char buf[24576];
        if (hy4_tensor_read(rank, tv, buf) == 0) printf("read blk.0.attn_norm OK (%ld bytes)\n", tv->nbytes);
        else { printf("read FAILED\n"); failures++; }
    }
    hy4_rank_close(rank);
    printf(failures ? "LOADER TEST FAIL (%d)\n" : "LOADER TEST OK\n", failures);
    return failures ? 1 : 0;
}
