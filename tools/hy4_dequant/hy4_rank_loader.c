/* hy4 lane: rank-manifest loader implementation. See hy4_rank_loader.h.
 *
 * The manifest (hy4-tp16-shard-v1) is machine-generated with a fixed
 * schema, so the JSON handling is a targeted scanner, not a general
 * parser: regenerate with tools/hy4_tp16_shard.py if the schema moves.
 * GGUF header parsing mirrors the sharder (alignment 32, v3 layout).
 */
#include "hy4_rank_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --- name table from the GGUF header (the offset authority) --------------- */

typedef struct {
    char name[160];
    long offset;   /* data-section-relative, from the infos */
    int type;
} gguf_info;

static int gguf_read_u64(FILE *f, uint64_t *v) { return fread(v, 8, 1, f) == 1 ? 0 : -1; }

static int gguf_skip_string(FILE *f) {
    uint64_t len;
    if (gguf_read_u64(f, &len)) return -1;
    return fseek(f, (long)len, SEEK_CUR);
}

static uint32_t fixed_size(uint32_t t) {
    static const uint32_t sizes[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    return t < 13 ? sizes[t] : 0;
}

static int gguf_skip_value(FILE *f, uint32_t vt) {
    if (vt == 8) return gguf_skip_string(f);
    if (vt == 9) {
        uint32_t et;
        uint64_t count;
        if (fread(&et, 4, 1, f) != 1 || fread(&count, 8, 1, f) != 1) return -1;
        uint32_t sz = fixed_size(et);
        if (et == 8) {
            for (uint64_t e = 0; e < count; ++e)
                if (gguf_skip_string(f)) return -1;
            return 0;
        }
        if (sz == 0) return -1;
        return fseek(f, (long)sz * count, SEEK_CUR);
    }
    uint32_t sz = fixed_size(vt);
    if (sz == 0) return -1;
    return fseek(f, (long)sz, SEEK_CUR);
}

static int gguf_parse_infos(FILE *f, long *data_offset, gguf_info **out,
                            long *count) {
    unsigned char magic[4];
    if (fread(magic, 1, 4, f) != 4) { fprintf(stderr, "gguf: magic short at pos %ld\n", ftell(f)); return -1; }
    if (memcmp(magic, "GGUF", 4)) { fprintf(stderr, "gguf: magic bad\n"); return -1; }
    uint32_t version;
    uint64_t tensor_count, kv_count;
    if (fread(&version, 4, 1, f) != 1) { fprintf(stderr, "gguf: version short\n"); return -1; }
    if (version < 2) { fprintf(stderr, "gguf: version %u\n", version); return -1; }
    if (gguf_read_u64(f, &tensor_count)) { fprintf(stderr, "gguf: tc\n"); return -1; }
    if (gguf_read_u64(f, &kv_count)) { fprintf(stderr, "gguf: kc\n"); return -1; }
    int alignment = 32;
    for (uint64_t i = 0; i < kv_count; ++i) {
        uint64_t len;
        if (gguf_read_u64(f, &len)) { fprintf(stderr, "gguf: kv %llu key len\n", (unsigned long long)i); return -1; }
        char *key = malloc((size_t)len + 1);
        if (fread(key, 1, (size_t)len, f) != (size_t)len) { free(key); fprintf(stderr, "gguf: kv %llu key read\n", (unsigned long long)i); return -1; }
        key[len] = 0;
        uint32_t vt;
        if (fread(&vt, 4, 1, f) != 1) { free(key); fprintf(stderr, "gguf: kv %llu vt\n", (unsigned long long)i); return -1; }
        if (strcmp(key, "general.alignment") == 0 && vt == 4) {
            uint32_t v;
            if (fread(&v, 4, 1, f) != 1) { free(key); return -1; }
            alignment = (int)v;
        } else if (gguf_skip_value(f, vt)) {
            free(key); fprintf(stderr, "gguf: kv %llu (%s) skip vt%u\n", (unsigned long long)i, key, vt); return -1;
        }
        free(key);
    }
    gguf_info *infos = calloc((size_t)tensor_count, sizeof(gguf_info));
    for (uint64_t i = 0; i < tensor_count; ++i) {
        uint64_t len;
        if (gguf_read_u64(f, &len)) { fprintf(stderr, "info %llu: len\n", (unsigned long long)i); free(infos); return -1; }
        if (len >= 160) { fprintf(stderr, "info %llu: len %llu too big\n", (unsigned long long)i, (unsigned long long)len); free(infos); return -1; }
        if (fread(infos[i].name, 1, (size_t)len, f) != (size_t)len) { free(infos); return -1; }
        infos[i].name[len] = 0;
        uint32_t nd;
        if (fread(&nd, 4, 1, f) != 1) { fprintf(stderr, "info %llu: nd\n", (unsigned long long)i); free(infos); return -1; }
        if (fseek(f, (long)nd * 8, SEEK_CUR)) { free(infos); return -1; }
        uint32_t gt;
        if (fread(&gt, 4, 1, f) != 1) { fprintf(stderr, "info %llu: gt\n", (unsigned long long)i); free(infos); return -1; }
        if (gguf_read_u64(f, (uint64_t *)&infos[i].offset)) { fprintf(stderr, "info %llu: off\n", (unsigned long long)i); free(infos); return -1; }
        infos[i].type = (int)gt;
    }
    /* the data section starts AFTER the tensor-info table */
    *data_offset = ((ftell(f) + alignment - 1) / alignment) * alignment;
    *out = infos;
    *count = (long)tensor_count;
    return 0;
}

/* --- manifest scanner (fixed schema) --------------------------------------- */

static int mget_long(const char *blob, const char *key, long *out) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\":", key);
    const char *at = strstr(blob, pat);
    if (!at) return -1;
    *out = strtol(at + strlen(pat), NULL, 10);
    return 0;
}

/* helpers scoped to one manifest object */
static int jget_string_scoped(const char *obj, const char *obj_end,
                              const char *key, char *out, size_t cap) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\":", key);
    const char *at = strstr(obj, pat);
    if (!at || at >= obj_end) return -1;
    at += strlen(pat);
    while (*at == ' ') at++;
    if (*at != '"') return -1;
    at++;
    size_t n = 0;
    while (at < obj_end && *at && *at != '"' && n + 1 < cap) out[n++] = *at++;
    out[n] = 0;
    return 0;
}

static int jget_string_manifest(const char *blob, const char *key,
                                char *out, size_t cap) {
    return jget_string_scoped(blob, blob + strlen(blob), key, out, cap);
}

/* --- loader ---------------------------------------------------------------- */

static char *read_whole(const char *path, long *out_bytes) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *data = malloc((size_t)n + 1);
    if (!data || fread(data, 1, (size_t)n, f) != (size_t)n) {
        fclose(f); free(data); return NULL;
    }
    fclose(f);
    data[n] = 0;
    *out_bytes = n;
    return data;
}

int hy4_rank_open(const char *pack_dir, int tolerate_sha_mismatch,
                  hy4_rank **out) {
    (void)tolerate_sha_mismatch; /* sidecar check lives in the test harness */
    hy4_rank *rank = calloc(1, sizeof(*rank));
    snprintf(rank->path, sizeof(rank->path), "%s", pack_dir);

    char manifest_path[1100], gguf_path[1100], gguf_name[160];
    snprintf(manifest_path, sizeof(manifest_path), "%s/manifest.json", pack_dir);
    long manifest_bytes;
    char *manifest = read_whole(manifest_path, &manifest_bytes);
    if (!manifest) { free(rank); return 1; }
    if (jget_string_manifest(manifest, "gguf", gguf_name, sizeof(gguf_name))) {
        free(manifest); free(rank); return 2;
    }
    snprintf(gguf_path, sizeof(gguf_path), "%s/%s", pack_dir, gguf_name);

    rank->file = fopen(gguf_path, "rb");
    if (!rank->file) { free(manifest); free(rank); return 3; }
    fseek(rank->file, 0, SEEK_END);
    rank->file_bytes = ftell(rank->file);
    fseek(rank->file, 0, SEEK_SET);

    gguf_info *infos = NULL;
    long info_count = 0, data_offset = 0;
    if (gguf_parse_infos(rank->file, &data_offset, &infos, &info_count)) {
        fclose(rank->file); free(manifest); free(rank); return 4;
    }

    /* walk manifest tensor objects and join with infos by name */
    hy4_tensor_view *views = calloc((size_t)info_count, sizeof(hy4_tensor_view));
    const char *p = strstr(manifest, "\"tensors\":");
    if (!p) { free(infos); free(views); free(manifest); fclose(rank->file); free(rank); return 5; }
    int count = 0;
    while (count < info_count) {
        const char *obj = strchr(p, '{');
        const char *arr_end = strstr(p, "]");
        if (!obj || (arr_end && obj > arr_end)) break;
        const char *obj_end = strchr(obj, '}');
        if (!obj_end) break;
        hy4_tensor_view *tv = &views[count];
        if (jget_string_scoped(obj, obj_end, "name", tv->name, sizeof(tv->name))) break;
        {
            const char *dat = strstr(obj, "\"dims\":");
            dat = dat ? strchr(dat, '[') : NULL;
            if (!dat) break;
            dat++;
            tv->n_dims = 0;
            while (*dat && *dat != ']' && tv->n_dims < 4) {
                char *end;
                long v = strtol(dat, &end, 10);
                if (end == dat) break;
                tv->dims[tv->n_dims++] = v;
                dat = end;
                while (*dat == ' ' || *dat == ',') dat++;
            }
        }
        long bytes = 0;
        if (mget_long(obj, "bytes", &bytes)) break;
        tv->nbytes = bytes;
        {
            const char *sat = strstr(obj, "\"slice\":");
            const char *obj_end = strchr(obj, '}');
            if (sat && obj_end && sat < obj_end) {
                const char *brace = strchr(sat, '{');
                if (brace && brace < obj_end) {
                    long dim = 0, start = 0;
                    /* encode split-on-dim as dim+1 so split0 (dim 0) never
                     * collides with the replicate marker 0 */
                    if (!mget_long(brace, "dim", &dim)) tv->slice_kind = (int)dim + 1;
                    if (!mget_long(brace, "start", &start)) tv->slice_start = start;
                } else {
                    tv->slice_kind = 0; /* string "replicate" */
                }
            }
        }
        /* join: linear scan over infos (2,134 x 2,134 worst case, one-time) */
        for (long i = 0; i < info_count; ++i) {
            if (strcmp(infos[i].name, tv->name) == 0) {
                tv->type = infos[i].type;
                tv->file_offset = data_offset + infos[i].offset;
                break;
            }
        }
        count++;
        p = obj_end + 1;
    }
    rank->tensor_count = count;
    rank->views = views;
    free(infos);
    free(manifest);
    *out = rank;
    return 0;
}

void hy4_rank_close(hy4_rank *rank) {
    if (!rank) return;
    if (rank->file) fclose((FILE *)rank->file);
    free(rank->views);
    free(rank);
}

const hy4_tensor_view *hy4_tensor_lookup(const hy4_rank *rank,
                                         const char *name) {
    for (int i = 0; i < rank->tensor_count; ++i)
        if (strcmp(rank->views[i].name, name) == 0) return &rank->views[i];
    return NULL;
}

int hy4_tensor_read(const hy4_rank *rank, const hy4_tensor_view *tv,
                    void *dst) {
    FILE *f = (FILE *)rank->file;
    if (fseek(f, tv->file_offset, SEEK_SET)) return -1;
    return fread(dst, 1, (size_t)tv->nbytes, f) == (size_t)tv->nbytes ? 0 : -1;
}
