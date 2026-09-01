/* hy4 lane: rank-manifest loader (hy4-tp16-shard-v1).
 *
 * Opens a deployed rank bundle (manifest.json + the rank GGUF), joins the
 * manifest's slice provenance with the GGUF tensor-info offsets, and hands
 * out tensor views for the module. Host-side; no ceph, no warm — the rank
 * file is the artifact of record on node-local NVMe.
 */
#ifndef HY4_RANK_LOADER_H
#define HY4_RANK_LOADER_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    char name[160];
    long dims[4];
    int n_dims;
    int type;          /* ggml type id as stored in the GGUF */
    int slice_kind;    /* 0 = replicate, 1..3 = split on that dim */
    long slice_start;
    long slice_count;
    long nbytes;       /* this rank's byte length for the tensor */
    long file_offset;  /* absolute file offset of the tensor data */
} hy4_tensor_view;

typedef struct {
    /* internal */
    void *manifest_data;
    void *gguf_handle;
    char path[1024];
    long file_bytes;
    int tensor_count;
    hy4_tensor_view *views;
    void *file;
} hy4_rank;

/* Opens <dir>/<gguf from manifest>, joins manifest + GGUF header. Returns 0
 * on success. tolerate_sha_mismatch != 0 skips sidecar verification. */
int hy4_rank_open(const char *pack_dir, int tolerate_sha_mismatch,
                  hy4_rank **out);
void hy4_rank_close(hy4_rank *rank);

/* Returns the view for `name`, or NULL. */
const hy4_tensor_view *hy4_tensor_lookup(const hy4_rank *rank,
                                         const char *name);

/* Reads the tensor's bytes into dst (dst must hold tv->nbytes). */
int hy4_tensor_read(const hy4_rank *rank, const hy4_tensor_view *tv,
                    void *dst);

#endif
