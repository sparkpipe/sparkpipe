// hy4 lane FP8 numerical rung: drives the PRODUCTION grouped-dot kernel
// (SparkHy4GemvFp8GroupedKernel from the resident-decode-stage unity
// layer, linked as a second TU) over sampled rows of the PLACED FP8
// rank pack and compares every dot against an independent double
// CPU expectation (int8 payload x exp2(scale-127) per group-32).
// Resumable per projection: done markers + next-entry checkpoint,
// ~660s self-cutoff, TSV deltas for the per-layer table.
//
// Usage: hy4_fp8_rung <pack.safetensors> <manifest> <workdir>
// Env: HY4_FP8_CKPT (checkpoint path), HY4_FP8_TSV (deltas),
//      HY4_FP8_CUTOFF (seconds, default 660).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <string>
#include <unistd.h>
#include <cuda_runtime.h>

__global__ void SparkHy4GemvFp8GroupedKernel(const uint8_t* weights,
    const uint8_t* scales, const float* x, float* y, int rows,
    int columns);

#define CHECK_CUDA(call) \
    do { cudaError_t cu_err_ = (call); if (cu_err_ != cudaSuccess) { \
        fprintf(stderr, "FP8RUNG CUDA FAIL %s: %s\n", #call, \
                cudaGetErrorString(cu_err_)); return 2; } } while (0)

struct Entry {
    int il;
    int kind;
    long payload_off;
    long scale_off;
    int experts;
    int rows;
    int cols;
    int groups;
};

static uint32_t lcg(uint32_t* s) {
    *s = *s * 1664525u + 1013904223u;
    return *s;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: hy4_fp8_rung <pack> <manifest> <workdir>\n");
        return -1;
    }
    const char* pack_path = argv[1];
    const char* manifest_path = argv[2];
    const char* workdir = argv[3];
    const char* ckpt_path = getenv("HY4_FP8_CKPT");
    const char* tsv_path = getenv("HY4_FP8_TSV");
    int cutoff = 660;
    if (const char* c = getenv("HY4_FP8_CUTOFF")) cutoff = atoi(c);
    time_t started = time(NULL);

    std::vector<Entry> entries;
    {
        FILE* f = fopen(manifest_path, "r");
        if (!f) { fprintf(stderr, "FP8RUNG NO MANIFEST\n"); return -2; }
        Entry e;
        while (fscanf(f, "%d %d %ld %ld %d %d %d %d", &e.il, &e.kind,
                      &e.payload_off, &e.scale_off, &e.experts, &e.rows,
                      &e.cols, &e.groups) == 8)
            entries.push_back(e);
        fclose(f);
    }
    if (entries.empty()) { fprintf(stderr, "FP8RUNG EMPTY MANIFEST\n"); return -3; }

    std::string done_dir = std::string(workdir) + "/done";
    int start = 0;
    if (ckpt_path) {
        FILE* f = fopen(ckpt_path, "r");
        if (f) {
            if (fscanf(f, "%d", &start) != 1) start = 0;
            fclose(f);
            fprintf(stderr, "FP8RUNG RESUME entry=%d\n", start);
        }
    }
    std::string tsv = tsv_path ? tsv_path
                               : (std::string(workdir) + "/fp8_rung.tsv");
    FILE* tsv_f = fopen(tsv.c_str(), "a");
    if (!tsv_f) { fprintf(stderr, "FP8RUNG NO TSV\n"); return -4; }

    int max_cols = 0, max_groups = 0;
    for (const Entry& e : entries) {
        if (e.cols > max_cols) max_cols = e.cols;
        if (e.groups > max_groups) max_groups = e.groups;
    }
    FILE* pack = fopen(pack_path, "rb");
    if (!pack) { fprintf(stderr, "FP8RUNG NO PACK\n"); return -5; }

    const int SAMPLE_ROWS = 8;
    uint8_t* h_w = new uint8_t[(size_t)SAMPLE_ROWS * max_cols];
    uint8_t* h_s = new uint8_t[(size_t)SAMPLE_ROWS * max_groups];
    float* h_x = new float[max_cols];
    float* h_y = new float[SAMPLE_ROWS];
    uint8_t* d_w; uint8_t* d_s; float* d_x; float* d_y;
    CHECK_CUDA(cudaMalloc(&d_w, (size_t)SAMPLE_ROWS * max_cols));
    CHECK_CUDA(cudaMalloc(&d_s, (size_t)SAMPLE_ROWS * max_groups));
    CHECK_CUDA(cudaMalloc(&d_x, (size_t)max_cols * 4));
    CHECK_CUDA(cudaMalloc(&d_y, SAMPLE_ROWS * 4));

    double global_max_abs = 0.0, global_max_rel = 0.0;
    int done_entries = 0;
    char path[512];
    for (int ei = start; ei < (int)entries.size(); ++ei) {
        Entry e = entries[ei];
        snprintf(path, sizeof(path), "%s/fp8_%d_%d.done", done_dir.c_str(),
                 e.il, e.kind);
        if (access(path, 0) == 0) continue;
        int experts[2] = {0, e.experts - 1};
        int rows_per_expert = SAMPLE_ROWS / 2;
        for (int j = 0; j < SAMPLE_ROWS; ++j) {
            int ex = experts[j / rows_per_expert];
            int rseq[4] = {0, e.rows / 3, (2 * e.rows) / 3, e.rows - 1};
            int r = rseq[j % rows_per_expert];
            off_t woff = (off_t)e.payload_off +
                         ((off_t)ex * e.rows + r) * e.cols;
            off_t soff = (off_t)e.scale_off +
                         ((off_t)ex * e.rows + r) * e.groups;
            if (fseeko(pack, woff, SEEK_SET) ||
                fread(h_w + (size_t)j * e.cols, 1, e.cols, pack) !=
                    (size_t)e.cols ||
                fseeko(pack, soff, SEEK_SET) ||
                fread(h_s + (size_t)j * e.groups, 1, e.groups, pack) !=
                    (size_t)e.groups) {
                fprintf(stderr, "FP8RUNG READ FAIL il=%d kind=%d\n", e.il,
                        e.kind);
                return -6;
            }
        }
        uint32_t seed = (uint32_t)(e.il * 20011 + e.kind * 4099 + 7);
        for (int i = 0; i < e.cols; ++i) {
            uint32_t v = lcg(&seed);
            h_x[i] = (float)((v >> 8) & 0xFFFF) / 32768.0f - 1.0f;
        }
        CHECK_CUDA(cudaMemcpy(d_w, h_w, (size_t)SAMPLE_ROWS * e.cols,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_s, h_s, (size_t)SAMPLE_ROWS * e.groups,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_x, h_x, (size_t)e.cols * 4,
                              cudaMemcpyHostToDevice));
        SparkHy4GemvFp8GroupedKernel<<<1, SAMPLE_ROWS>>>(d_w, d_s, d_x, d_y,
                                                         SAMPLE_ROWS, e.cols);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_y, d_y, SAMPLE_ROWS * 4,
                              cudaMemcpyDeviceToHost));
        double layer_max_rel = 0.0, layer_max_abs = 0.0;
        for (int j = 0; j < SAMPLE_ROWS; ++j) {
            const uint8_t* w = h_w + (size_t)j * e.cols;
            const uint8_t* s = h_s + (size_t)j * e.groups;
            double expect = 0.0;
            for (int g = 0; g < e.groups; ++g) {
                double partial = 0.0;
                for (int i = 0; i < 32; ++i)
                    partial += (double)(int8_t)w[g * 32 + i] *
                               (double)h_x[g * 32 + i];
                expect += partial * exp2((double)s[g] - 127.0);
            }
            double d = fabs((double)h_y[j] - expect);
            double rel = d / (fabs(expect) > 1e-30 ? fabs(expect) : 1e-30);
            fprintf(tsv_f, "%d\t%d\t%d\t%.9e\t%.9e\t%.6e\t%.6e\n", e.il,
                    e.kind, j, expect, (double)h_y[j], d, rel);
            if (expect == 0.0)
                fprintf(stderr, "FP8RUNG ZERO il=%d kind=%d row=%d\n", e.il,
                        e.kind, j);
            if (rel > layer_max_rel) layer_max_rel = rel;
            if (d > layer_max_abs) layer_max_abs = d;
        }
        fflush(tsv_f);
        printf("FP8RUNG il=%d kind=%s maxabs=%.6e maxrel=%.6e\n", e.il,
               e.kind == 0 ? "gate_up" : "down", layer_max_abs,
               layer_max_rel);
        fflush(stdout);
        if (layer_max_abs > global_max_abs) global_max_abs = layer_max_abs;
        if (layer_max_rel > global_max_rel) global_max_rel = layer_max_rel;
        snprintf(path, sizeof(path), "%s/fp8_%d_%d.done", done_dir.c_str(),
                 e.il, e.kind);
        FILE* df = fopen(path, "w");
        if (!df) { fprintf(stderr, "FP8RUNG MARKER FAIL\n"); return -7; }
        fclose(df);
        ++done_entries;
        if (ckpt_path) {
            FILE* cf = fopen(ckpt_path, "w");
            if (!cf) { fprintf(stderr, "FP8RUNG CKPT FAIL\n"); return -8; }
            fprintf(cf, "%d\n", ei + 1);
            fclose(cf);
        }
        if (time(NULL) - started > cutoff) {
            fprintf(stderr, "FP8RUNG CUTOFF entry=%d\n", ei + 1);
            fclose(tsv_f);
            fclose(pack);
            printf("FP8_RUNG_PARTIAL entries_done=%d maxabs=%.6e maxrel=%.6e\n",
                   done_entries, global_max_abs, global_max_rel);
            return 0;
        }
    }
    fclose(tsv_f);
    fclose(pack);
    printf("FP8_RUNG_DONE entries=%d maxabs=%.6e maxrel=%.6e\n",
           (int)entries.size(), global_max_abs, global_max_rel);
    return 0;
}
