#ifndef HY4_RANK_LOADER_H
#define HY4_RANK_LOADER_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    char name[160];
    long dims[4];
    int n_dims;
    int type;
    int slice_kind;
    long slice_start;
    long slice_count;
    long nbytes;
    long file_offset;
} hy4_tensor_view;

typedef struct {
    void *manifest_data;
    void *gguf_handle;
    char path[1024];
    long file_bytes;
    int tensor_count;
    hy4_tensor_view *views;
    void *file;
} hy4_rank;

int hy4_rank_open(const char *pack_dir, int tolerate_sha_mismatch,
                  hy4_rank **out);
void hy4_rank_close(hy4_rank *rank);

const hy4_tensor_view *hy4_tensor_lookup(const hy4_rank *rank,
                                         const char *name);

int hy4_tensor_read(const hy4_rank *rank, const hy4_tensor_view *tv,
                    void *dst);

#endif
