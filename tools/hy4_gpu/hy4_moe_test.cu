// hy4 lane: GPU MoE expert-gather cell — routed-expert FFN of layer 1 as
// CUDA kernels (router logits gemv + host selection, per-expert gate/up
// gemvs, HYV4 swiglu clamp, down gemv, weighted accumulate, shared expert),
// validated against an in-harness float64 reference. The MoE input vector
// is a deterministic seeded pattern (normalizes like a real fcur); the hc
// kernels feeding it are a separate cell. Expert slabs are host-dequanted
// from their OWNER rank bundles via the vendor header.
//
// Usage: hy4_moe_test <rank00.gguf>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define _Static_assert(cond, msg) static_assert(cond, msg)
#include "hy4_iq_dequant_vendor.h"

#define CHECK_CUDA(call) \
    do { cudaError_t e = (call); if (e != cudaSuccess) { \
        printf("CUDA FAIL %s: %s\n", #call, cudaGetErrorString(e)); \
        return 1; } } while (0)

__global__ void k_gemv(const float* W, const float* x, float* y, int rows,
                       int K) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* row = W + (size_t)r * K;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc = fmaf(row[k], x[k], acc);
    y[r] = acc;
}

__global__ void k_swiglu_moe(float* g, float* u, int n, float limit) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = g[i] > limit ? limit : g[i];
    float uv = u[i] > limit ? limit : (u[i] < -limit ? -limit : u[i]);
    u[i] = (gv / (1.f + expf(-gv))) * uv;
}

__global__ void k_axpy(float* acc, const float* h, float w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += w * h[i];
}

struct TensorInfo {
    char name[128];
    uint32_t n_dims;
    uint32_t type;
    uint64_t offset;
    uint64_t nelem;
};

static int kv_size(uint32_t vt) {
    switch (vt) {
        case 0: case 1: case 7: return 1;
        case 2: case 3: return 2;
        case 4: case 5: case 6: return 4;
        case 10: case 11: case 12: return 8;
        default: return -1;
    }
}

static int parse_rank_gguf(const char* path, std::vector<TensorInfo>& out,
                           uint64_t* data_offset) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", path); return -1; }
    auto rd = [&](void* dst, size_t n) { return fread(dst, 1, n, f) == n; };
    uint32_t magic, version;
    if (!rd(&magic, 4) || !rd(&version, 4) || magic != 0x46554747u || version < 2) {
        fprintf(stderr, "not GGUFv2+\n"); fclose(f); return -1;
    }
    uint64_t tensor_count, kv_count;
    if (!rd(&tensor_count, 8) || !rd(&kv_count, 8)) { fclose(f); return -1; }
    auto rstr = [&](char* buf, uint32_t cap) -> int {
        uint64_t n;
        if (!rd(&n, 8)) return -1;
        if (buf && cap) buf[0] = 0;
        uint64_t done = 0;
        char sink[4096];
        while (done < n) {
            uint64_t take = n - done;
            if (take > sizeof(sink)) take = sizeof(sink);
            if (!rd(sink, take)) return -1;
            if (buf && done < cap - 1) {
                uint64_t keep = cap - 1 - done;
                if (keep > take) keep = take;
                memcpy(buf + done, sink, keep);
                buf[done + keep] = 0;
            }
            done += take;
        }
        return 0;
    };
    auto rvalue = [&](auto&& self, uint32_t vt) -> int {
        if (vt == 8) { char tmp[128]; return rstr(tmp, sizeof(tmp)); }
        if (vt == 9) {
            uint32_t et; uint64_t n;
            if (!rd(&et, 4) || !rd(&n, 8)) return -1;
            for (uint64_t i = 0; i < n; ++i)
                if (self(self, et)) return -1;
            return 0;
        }
        int sz = kv_size(vt);
        if (sz < 0) return -1;
        uint8_t skip[8];
        for (int i = 0; i < sz; ++i) if (!rd(skip, 1)) return -1;
        return 0;
    };
    for (uint64_t i = 0; i < kv_count; ++i) {
        char key[128];
        if (rstr(key, sizeof(key))) { fclose(f); return -1; }
        uint32_t vt;
        if (!rd(&vt, 4) || rvalue(rvalue, vt)) { fclose(f); return -1; }
    }
    for (uint64_t i = 0; i < tensor_count; ++i) {
        TensorInfo ti = {};
        if (rstr(ti.name, sizeof(ti.name))) { fclose(f); return -1; }
        if (!rd(&ti.n_dims, 4)) { fclose(f); return -1; }
        ti.nelem = 1;
        for (uint32_t d = 0; d < ti.n_dims; ++d) {
            uint64_t v;
            if (!rd(&v, 8)) { fclose(f); return -1; }
            ti.nelem *= v;
        }
        if (!rd(&ti.type, 4) || !rd(&ti.offset, 8)) { fclose(f); return -1; }
        out.push_back(ti);
    }
    long pos = ftell(f);
    *data_offset = ((uint64_t)pos + 31) / 32 * 32;
    fclose(f);
    return 0;
}

static uint16_t rd16h(const uint8_t* p) {
    return (uint16_t)(p[0] | (p[1] << 8));
}

static int block_geom(uint32_t type, int* bpe, int* bpb) {
    switch (type) {
        case 0: *bpe = 1; *bpb = 4; return 0;
        case 8: *bpe = 32; *bpb = 34; return 0;
        case 14: *bpe = 256; *bpb = 210; return 0;
        case 16: *bpe = 256; *bpb = 66; return 0;
        case 18: *bpe = 256; *bpb = 98; return 0;
        case 23: *bpe = 256; *bpb = 136; return 0;
        case 29: *bpe = 256; *bpb = 56; return 0;
        default: return -1;
    }
}

static const TensorInfo* find_tensor(const std::vector<TensorInfo>& t,
                                     const char* name) {
    for (size_t i = 0; i < t.size(); ++i)
        if (strcmp(t[i].name, name) == 0) return &t[i];
    return nullptr;
}

static int read_tensor(FILE* f, const TensorInfo* ti, uint64_t data_offset,
                       std::vector<uint8_t>& buf) {
    int bpe, bpb;
    if (block_geom(ti->type, &bpe, &bpb)) return -1;
    buf.resize(ti->nelem / bpe * bpb);
    return fseek(f, (long)(data_offset + ti->offset), SEEK_SET) == 0 &&
           fread(buf.data(), 1, buf.size(), f) == buf.size() ? 0 : -1;
}

static void dequant_q8_0(const uint8_t* src, float* dst, long nblocks) {
    for (long b = 0; b < nblocks; ++b, src += 34) {
        const float d = hy4_fp16_to_fp32(rd16h(src));
        for (int i = 0; i < 32; ++i)
            dst[(size_t)b * 32 + i] = d * (float)(int8_t)src[2 + i];
    }
}

static int check(const char* tag, const float* gpu, const double* ref,
                 long n, double tol_scale) {
    double worst = 0;
    for (long i = 0; i < n; ++i) {
        if (isnan(gpu[i]) || isinf(gpu[i])) {
            printf("%s FAIL elem %ld gpu=%f (non-finite)\n", tag, i,
                   gpu[i]);
            return 1;
        }
        double m = fabs(ref[i]) > 1.0 ? fabs(ref[i]) : 1.0;
        double d = fabs((double)gpu[i] - ref[i]);
        if (d > worst) worst = d;
        if (d > tol_scale * m) {
            printf("%s FAIL elem %ld gpu=%.8f ref=%.8f |d|=%.3g\n",
                   tag, i, gpu[i], ref[i], d);
            return 1;
        }
    }
    printf("%s PASS (max|d|=%.3g, %ld elems)\n", tag, worst, n);
    return 0;
}

static void read_dequant_expert(const char* argv0, int layer, int expert,
                                const char* which, float* dst, int rows,
                                int cols) {
    const int owner = expert / 16;
    const int li = expert % 16;
    char path[512], pat[64];
    snprintf(path, sizeof(path), "%s", argv0);
    char* rk = strstr(path, "/rank-00/");
    if (!rk) { fprintf(stderr, "rank dir pattern missing\n"); exit(1); }
    snprintf(rk, sizeof(path) - (size_t)(rk - path),
             "/rank-%02d/model-ud-iq1m-tp16-rank-%02d.gguf", owner, owner);
    std::vector<TensorInfo> ts;
    uint64_t od;
    if (parse_rank_gguf(path, ts, &od))
        exit(1);
    snprintf(pat, sizeof(pat), "blk.%d.ffn_%s_exps.weight", layer, which);
    const TensorInfo* ti = find_tensor(ts, pat);
    if (!ti) { fprintf(stderr, "missing %s\n", pat); exit(1); }
    int bpe, bpb;
    uint32_t ty = ti->type;
    if (block_geom(ty, &bpe, &bpb)) {
        fprintf(stderr, "unsupported expert type %u\n", ty); exit(1);
    }
    long slab_bytes = (long)(ti->nelem / 16 / bpe) * bpb;
    std::vector<uint8_t> raw((size_t)slab_bytes);
    FILE* f = fopen(path, "rb");
    if (!f || fseek(f, (long)(od + ti->offset + (uint64_t)li * slab_bytes),
                    SEEK_SET) ||
        fread(raw.data(), 1, raw.size(), f) != raw.size()) {
        fprintf(stderr, "slab read fail %s e%d\n", pat, expert); exit(1);
    }
    fclose(f);
    long nelem = (long)rows * cols;
    if (ty == 16) hy4_dequant_iq2_xxs(raw.data(), dst, nelem / 256);
    else if (ty == 18) hy4_dequant_iq3_xxs(raw.data(), dst, nelem / 256);
    else if (ty == 23) hy4_dequant_row_iq4_xs((const block_iq4_xs*)raw.data(),
                                              dst, nelem);
    else if (ty == 29) hy4_dequant_iq1_m(raw.data(), dst, nelem / 256);
    else { fprintf(stderr, "type %u\n", ty); exit(1); }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <rank00.gguf>\n", argv[0]);
        return 2;
    }
    std::vector<TensorInfo> tensors;
    uint64_t data_offset = 0;
    if (parse_rank_gguf(argv[1], tensors, &data_offset)) return 1;

    const int N_FF = 2048, N_EMBD = 6144, N_EXPERT = 256, N_USED = 8;
    const double ROUTER_SCALE = 2.827, LIMIT = 10.0;

    std::vector<float> h_fcur(N_EMBD);
    {
        uint32_t st = 12345;
        for (int i = 0; i < N_EMBD; ++i) {
            st = st * 1664525u + 1013904223u;
            h_fcur[i] = ((float)(st >> 8) / 16777216.0f - 0.5f) * 2.0f;
        }
    }

    std::vector<uint8_t> raw;
    const TensorInfo* ti;

    FILE* f = fopen(argv[1], "rb");
    if (!f) return 1;

    ti = find_tensor(tensors, "blk.1.ffn_gate_inp.weight");
    read_tensor(f, ti, data_offset, raw);
    std::vector<float> h_ginp(N_EXPERT * N_EMBD);
    memcpy(h_ginp.data(), raw.data(), (size_t)N_EXPERT * N_EMBD * 4);

    ti = find_tensor(tensors, "blk.1.exp_probs_b.bias");
    read_tensor(f, ti, data_offset, raw);
    std::vector<float> h_eb(N_EXPERT);
    memcpy(h_eb.data(), raw.data(), (size_t)N_EXPERT * 4);

    float *d_fcur, *d_ginp, *d_logits, *d_wg, *d_wu, *d_wd, *d_g, *d_u,
          *d_h, *d_ffn;
    CHECK_CUDA(cudaMalloc(&d_fcur, N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_ginp, (size_t)N_EXPERT * N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_logits, N_EXPERT * 4));
    CHECK_CUDA(cudaMalloc(&d_wg, (size_t)N_FF * N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_wu, (size_t)N_FF * N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_wd, (size_t)N_EMBD * N_FF * 4));
    CHECK_CUDA(cudaMalloc(&d_g, N_FF * 4));
    CHECK_CUDA(cudaMalloc(&d_u, N_FF * 4));
    CHECK_CUDA(cudaMalloc(&d_h, N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_ffn, N_EMBD * 4));
    cudaMemcpy(d_fcur, h_fcur.data(), N_EMBD * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ginp, h_ginp.data(), (size_t)N_EXPERT * N_EMBD * 4,
               cudaMemcpyHostToDevice);
    cudaMemset(d_ffn, 0, N_EMBD * 4);

    k_gemv<<<(N_EXPERT + 127) / 128, 128>>>(d_ginp, d_fcur, d_logits,
                                            N_EXPERT, N_EMBD);
    std::vector<float> logits(N_EXPERT);
    cudaMemcpy(logits.data(), d_logits, N_EXPERT * 4,
               cudaMemcpyDeviceToHost);

    std::vector<double> probs(N_EXPERT), key(N_EXPERT);
    for (int e = 0; e < N_EXPERT; ++e) {
        probs[e] = 1.0 / (1.0 + exp(-(double)logits[e]));
        key[e] = probs[e] + h_eb[e];
    }
    int sel[N_USED];
    double ww[N_USED];
    for (int k = 0; k < N_USED; ++k) {
        int best = -1;
        double bv = -1e300;
        for (int e = 0; e < N_EXPERT; ++e)
            if (key[e] > bv) { bv = key[e]; best = e; }
        key[best] = -1e300;
        sel[k] = best;
        ww[k] = probs[best];
    }
    double wsum = 0;
    for (int k = 0; k < N_USED; ++k) wsum += ww[k];
    if (wsum < 6.103515625e-5) wsum = 6.103515625e-5;
    for (int k = 0; k < N_USED; ++k) ww[k] = ww[k] / wsum * ROUTER_SCALE;
    printf("selected:");
    for (int k = 0; k < N_USED; ++k) printf(" %d", sel[k]);
    printf("\n");

    std::vector<double> ref_ffn(N_EMBD, 0.0), ref_g(N_FF), ref_u(N_FF);
    std::vector<float> wg((size_t)N_FF * N_EMBD), wu((size_t)N_FF * N_EMBD),
        wd((size_t)N_EMBD * N_FF);
    int bad = 0;
    for (int k = 0; k < N_USED; ++k) {
        const int e = sel[k];
        read_dequant_expert(argv[1], 1, e, "gate", wg.data(), N_FF, N_EMBD);
        read_dequant_expert(argv[1], 1, e, "up", wu.data(), N_FF, N_EMBD);
        read_dequant_expert(argv[1], 1, e, "down", wd.data(), N_EMBD, N_FF);
        cudaMemcpy(d_wg, wg.data(), (size_t)N_FF * N_EMBD * 4,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_wu, wu.data(), (size_t)N_FF * N_EMBD * 4,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_wd, wd.data(), (size_t)N_EMBD * N_FF * 4,
                   cudaMemcpyHostToDevice);
        k_gemv<<<(N_FF + 127) / 128, 128>>>(d_wg, d_fcur, d_g, N_FF, N_EMBD);
        k_gemv<<<(N_FF + 127) / 128, 128>>>(d_wu, d_fcur, d_u, N_FF, N_EMBD);
        k_swiglu_moe<<<(N_FF + 127) / 128, 128>>>(d_g, d_u, N_FF,
                                                  (float)LIMIT);
        k_gemv<<<(N_EMBD + 127) / 128, 128>>>(d_wd, d_u, d_h, N_EMBD, N_FF);
        k_axpy<<<(N_EMBD + 127) / 128, 128>>>(d_ffn, d_h, (float)ww[k],
                                              N_EMBD);

        for (int j = 0; j < N_FF; ++j) {
            double g = 0, u = 0;
            for (int i2 = 0; i2 < N_EMBD; ++i2) {
                g += (double)wg[(size_t)j * N_EMBD + i2] * h_fcur[i2];
                u += (double)wu[(size_t)j * N_EMBD + i2] * h_fcur[i2];
            }
            ref_g[j] = g;
            ref_u[j] = u;
        }
        for (int j = 0; j < N_FF; ++j) {
            double gv = ref_g[j] > LIMIT ? LIMIT : ref_g[j];
            double uv = ref_u[j] > LIMIT ? LIMIT
                      : (ref_u[j] < -LIMIT ? -LIMIT : ref_u[j]);
            ref_u[j] = (gv / (1.0 + exp(-gv))) * uv;
        }
        for (int i = 0; i < N_EMBD; ++i) {
            double acc = 0;
            for (int j = 0; j < N_FF; ++j)
                acc += (double)wd[(size_t)i * N_FF + j] * ref_u[j];
            ref_ffn[i] += ww[k] * acc;
        }
    }
    {
        std::vector<float> routed(N_EMBD);
        cudaMemcpy(routed.data(), d_ffn, N_EMBD * 4, cudaMemcpyDeviceToHost);
        bad += check("MOE_ROUTED", routed.data(), ref_ffn.data(), N_EMBD,
                     1e-3);
    }

    {
        ti = find_tensor(tensors, "blk.1.ffn_gate_shexp.weight");
        read_tensor(f, ti, data_offset, raw);
        std::vector<float> shg((size_t)N_FF * N_EMBD);
        hy4_dequant_row_q6_K((const block_q6_K*)raw.data(), shg.data(),
                             (long)N_FF * N_EMBD);
        ti = find_tensor(tensors, "blk.1.ffn_up_shexp.weight");
        read_tensor(f, ti, data_offset, raw);
        std::vector<float> shu((size_t)N_FF * N_EMBD);
        hy4_dequant_row_q6_K((const block_q6_K*)raw.data(), shu.data(),
                             (long)N_FF * N_EMBD);
        ti = find_tensor(tensors, "blk.1.ffn_down_shexp.weight");
        read_tensor(f, ti, data_offset, raw);
        std::vector<float> shd((size_t)N_EMBD * N_FF);
        hy4_dequant_row_q6_K((const block_q6_K*)raw.data(), shd.data(),
                             (long)N_EMBD * N_FF);
        cudaMemcpy(d_wg, shg.data(), (size_t)N_FF * N_EMBD * 4,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_wu, shu.data(), (size_t)N_FF * N_EMBD * 4,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_wd, shd.data(), (size_t)N_EMBD * N_FF * 4,
                   cudaMemcpyHostToDevice);
        k_gemv<<<(N_FF + 127) / 128, 128>>>(d_wg, d_fcur, d_g, N_FF, N_EMBD);
        k_gemv<<<(N_FF + 127) / 128, 128>>>(d_wu, d_fcur, d_u, N_FF, N_EMBD);
        k_swiglu_moe<<<(N_FF + 127) / 128, 128>>>(d_g, d_u, N_FF, 3.4e38f);
        k_gemv<<<(N_EMBD + 127) / 128, 128>>>(d_wd, d_u, d_h, N_EMBD, N_FF);
        k_axpy<<<(N_EMBD + 127) / 128, 128>>>(d_ffn, d_h, 1.0f, N_EMBD);

        for (int j = 0; j < N_FF; ++j) {
            double g = 0, u = 0;
            for (int i2 = 0; i2 < N_EMBD; ++i2) {
                g += (double)shg[(size_t)j * N_EMBD + i2] * h_fcur[i2];
                u += (double)shu[(size_t)j * N_EMBD + i2] * h_fcur[i2];
            }
            ref_g[j] = g;
            ref_u[j] = u;
        }
        for (int j = 0; j < N_FF; ++j) {
            double uv = ref_u[j] > LIMIT ? LIMIT
                      : (ref_u[j] < -LIMIT ? -LIMIT : ref_u[j]);
            ref_u[j] = (ref_g[j] / (1.0 + exp(-ref_g[j]))) * uv;
        }
        for (int i = 0; i < N_EMBD; ++i) {
            double acc = 0;
            for (int j = 0; j < N_FF; ++j)
                acc += (double)shd[(size_t)i * N_FF + j] * ref_u[j];
            ref_ffn[i] += acc;
        }
        std::vector<float> ffn(N_EMBD);
        cudaMemcpy(ffn.data(), d_ffn, N_EMBD * 4, cudaMemcpyDeviceToHost);
        bad += check("MOE_WITH_SHEXP", ffn.data(), ref_ffn.data(), N_EMBD,
                     1e-3);
    }
    printf("MOE %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
