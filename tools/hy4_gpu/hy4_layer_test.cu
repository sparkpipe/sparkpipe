// hy4 lane: FULL-LAYER GPU assembly — token 802 through layers 0 and 1
// (all 64 heads across the 16 rank-bundle head slices, dense layer-0 FFN,
// the 256-expert routed MoE + shared expert in layer 1, both hc stages),
// then float-compared against the CPU forward's dumped layer-1 stream
// state (hy4_generate dump arg, l1state.t0: 4 x 6144 floats).
//
// Usage: hy4_layer_test <allranks_dir> <cpu_dump.t0>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cerrno>
#include <vector>
#include <cuda_runtime.h>

extern "C" {
#include "hy4_rank_loader.h"
}
#define _Static_assert(cond, msg) static_assert(cond, msg)
#include "hy4_iq_dequant_vendor.h"

#define CHECK_CUDA(call) \
    do { cudaError_t e = (call); if (e != cudaSuccess) { \
        printf("CUDA FAIL %s: %s\n", #call, cudaGetErrorString(e)); \
        return 1; } } while (0)

#define N_EMBD 6144
#define HC 4
#define HEADS 64
#define RANK_HEADS 4
#define N_RANKS 16
#define HK 256
#define NOPE 192
#define ROT 64
#define KV_LORA 512
#define VD 256
#define N_FF 2048
#define N_EXPERT 256
#define N_USED 8
#define LAYERS 2

__global__ void k_gemv(const float* W, const float* x, float* y, int rows,
                       int K) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* row = W + (size_t)r * K;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc = fmaf(row[k], x[k], acc);
    y[r] = acc;
}

__global__ void k_dot(const float* a, const float* b, float* out, int n) {
    float s = 0;
    for (int i = 0; i < n; ++i) s = fmaf(a[i], b[i], s);
    out[0] = s;
}

__global__ void k_rms_sq(const float* x, float* ss, int n) {
    __shared__ float part[256];
    int tid = threadIdx.x;
    float s = 0.f;
    for (int k = tid; k < n; k += blockDim.x) s = fmaf(x[k], x[k], s);
    part[tid] = s;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        __syncthreads();
        if (tid < off) part[tid] += part[tid + off];
    }
    if (tid == 0) ss[0] = part[0];
}

__global__ void k_rms_scale(const float* x, const float* w, float* y,
                            int n, float eps, const float* ss) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float inv = rsqrtf(ss[0] / n + eps);
    float wv = w ? w[i] : 1.0f;
    y[i] = x[i] * inv * wv;
}

__global__ void k_rope(float* v, int rot, float pos) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= rot / 2) return;
    float ang = pos * powf(1e7f, -2.0f * d / rot);
    float a = v[2 * d], b = v[2 * d + 1];
    v[2 * d] = a * cosf(ang) - b * sinf(ang);
    v[2 * d + 1] = a * sinf(ang) + b * cosf(ang);
}

__global__ void k_hc_gates(const float* mixes, const float* scale,
                           const float* base, float eps, float magnitude,
                           float* pre, float* post) {
    int i = threadIdx.x;
    if (i < 4)
        pre[i] = 1.0f / (1.0f + expf(-(mixes[i] * scale[0] + base[i]))) + eps;
    else {
        int j = i - 4;
        post[j] = magnitude / (1.0f + expf(-(mixes[4 + j] * scale[1] +
                                            base[4 + j]))) + eps;
    }
}

__global__ void k_hc_reduce(const float* streams, const float* pre,
                            float* out, int n_embd, int hc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    float acc = 0;
    for (int s = 0; s < hc; ++s)
        acc += streams[(size_t)s * n_embd + i] * pre[s];
    out[i] = acc;
}

__global__ void k_hc_distribute(float* streams, const float* branch,
                                const float* post, int n_embd, int hc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    for (int s = 0; s < hc; ++s)
        streams[(size_t)s * n_embd + i] += branch[i] * post[s];
}

__global__ void k_sigmoid(float* v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = 1.0f / (1.0f + expf(-v[i]));
}

__global__ void k_swiglu(float* g, float* u, int n, float limit) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = g[i] > limit ? limit : g[i];
    float uv = u[i] > limit ? limit : (u[i] < -limit ? -limit : u[i]);
    u[i] = (gv / (1.f + expf(-gv))) * uv;
}

__global__ void k_scale(float* v, float w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] *= w;
}

__global__ void k_axpy(float* acc, const float* h, float w, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += w * h[i];
}

static uint16_t rd16h(const uint8_t* p) {
    return (uint16_t)(p[0] | (p[1] << 8));
}

static void dequant_view(const uint8_t* src, uint32_t type, float* dst,
                         long nelem) {
    switch (type) {
        case 0: memcpy(dst, src, nelem * 4); break;
        case 8: {
            long nb = nelem / 32;
            for (long b = 0; b < nb; ++b, src += 34) {
                const float d = hy4_fp16_to_fp32(rd16h(src));
                for (int i = 0; i < 32; ++i)
                    dst[b * 32 + i] = d * (float)(int8_t)src[2 + i];
            }
            break;
        }
        case 12: hy4_dequant_row_q4_K((const block_q4_K*)src, dst, nelem); break;
        case 13: hy4_dequant_row_q5_K((const block_q5_K*)src, dst, nelem); break;
        case 14: hy4_dequant_row_q6_K((const block_q6_K*)src, dst, nelem); break;
        case 16: hy4_dequant_iq2_xxs(src, dst, nelem / 256); break;
        case 18: hy4_dequant_iq3_xxs(src, dst, nelem / 256); break;
        case 23: hy4_dequant_row_iq4_xs((const block_iq4_xs*)src, dst, nelem); break;
        case 29: hy4_dequant_iq1_m(src, dst, nelem / 256); break;
        default: fprintf(stderr, "bad type %u\n", type); exit(1);
    }
}

static void load_dequant(hy4_rank* rf, const char* name, float* dst,
                         long nelem) {
    const hy4_tensor_view* tv = hy4_tensor_lookup(rf, name);
    if (!tv) { fprintf(stderr, "missing %s\n", name); exit(1); }
    std::vector<uint8_t> raw((size_t)tv->nbytes);
    if (hy4_tensor_read(rf, tv, raw.data())) {
        fprintf(stderr, "read fail %s\n", name); exit(1);
    }
    dequant_view(raw.data(), tv->type, dst, nelem);
}

static void load_f32(hy4_rank* rf, const char* name, float* dst,
                     size_t n) {
    const hy4_tensor_view* tv = hy4_tensor_lookup(rf, name);
    if (!tv || tv->type != 0) {
        fprintf(stderr, "missing f32 %s\n", name); exit(1);
    }
    std::vector<uint8_t> buf(n * 4);
    if (hy4_tensor_read(rf, tv, buf.data())) {
        fprintf(stderr, "read fail %s\n", name); exit(1);
    }
    memcpy(dst, buf.data(), n * 4);
}

static int check(const char* tag, const float* gpu, const float* ref,
                 long n, double tol_scale) {
    double worst = 0;
    for (long i = 0; i < n; ++i) {
        if (isnan(gpu[i]) || isinf(gpu[i])) {
            printf("%s FAIL elem %ld gpu=%f (non-finite)\n", tag, i,
                   gpu[i]);
            return 1;
        }
        double m = fabs(ref[i]) > 1.0 ? fabs(ref[i]) : 1.0;
        double d = fabs((double)gpu[i] - (double)ref[i]);
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

__global__ void k_mul(float* v, const float* g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] *= g[i];
}

static void load_f32_dev(hy4_rank* rf, const char* name, float* d_dst,
                         size_t n) {
    const hy4_tensor_view* tv = hy4_tensor_lookup(rf, name);
    if (!tv || tv->type != 0) {
        fprintf(stderr, "missing f32 %s\n", name); exit(1);
    }
    std::vector<uint8_t> buf(n * 4);
    if (hy4_tensor_read(rf, tv, buf.data())) {
        fprintf(stderr, "read fail %s\n", name); exit(1);
    }
    cudaMemcpy(d_dst, buf.data(), n * 4, cudaMemcpyHostToDevice);
}

static void upload_dequant(hy4_rank* rf, const char* name, float* d_dst,
                           long nelem) {
    const hy4_tensor_view* tv = hy4_tensor_lookup(rf, name);
    if (!tv) { fprintf(stderr, "missing %s\n", name); exit(1); }
    std::vector<uint8_t> raw((size_t)tv->nbytes);
    if (hy4_tensor_read(rf, tv, raw.data())) {
        fprintf(stderr, "read fail %s\n", name); exit(1);
    }
    std::vector<float> h((size_t)nelem);
    dequant_view(raw.data(), tv->type, h.data(), nelem);
    cudaMemcpy(d_dst, h.data(), (size_t)nelem * 4, cudaMemcpyHostToDevice);
}

static void hc_pre_gpu(float* d_streams, float* d_flat, float* d_hc_fn,
                       float* d_hc_sc, float* d_hc_base, float* d_mixes,
                       float* d_pre, float* d_post, float* d_red,
                       float* d_cur, float* d_ss, int n_embd, int hc) {
    int flat = n_embd * hc;
    k_rms_sq<<<1, 256>>>(d_streams, d_ss, flat);
    k_rms_scale<<<flat / 256, 256>>>(d_streams, nullptr, d_flat, flat,
                                     1e-5f, d_ss);
    k_gemv<<<1, 8>>>(d_hc_fn, d_flat, d_mixes, 8, flat);
    k_hc_gates<<<1, 8>>>(d_mixes, d_hc_sc, d_hc_base, 1e-6f, 2.0f, d_pre,
                         d_post);
    k_hc_reduce<<<n_embd / 256, 256>>>(d_streams, d_pre, d_red, n_embd, hc);
    (void)d_cur;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <allranks_dir> <cpu_dump.t0>\n", argv[0]);
        return 2;
    }
    hy4_rank* R[N_RANKS];
    char pdir[512];
    for (int r = 0; r < N_RANKS; ++r) {
        snprintf(pdir, sizeof(pdir), "%s/rank-%02d", argv[1], r);
        if (hy4_rank_open(pdir, 1, &R[r])) {
            fprintf(stderr, "open rank %d\n", r);
            return 1;
        }
    }
    FILE* df = fopen(argv[2], "rb");
    if (!df) { fprintf(stderr, "dump open failed\n"); return 1; }
    std::vector<float> ref((size_t)HC * N_EMBD);
    if (fread(ref.data(), 4, ref.size(), df) != ref.size()) {
        fprintf(stderr, "dump short\n"); return 1;
    }
    fclose(df);

    std::vector<float> h_embd(N_EMBD);
    {
        const hy4_tensor_view* tv = hy4_tensor_lookup(R[0],
                                                      "token_embd.weight");
        if (!tv) { fprintf(stderr, "missing token_embd\n"); return 1; }
        long row_bytes = tv->nbytes / 7552;
        char efile[600];
        snprintf(efile, sizeof(efile),
                 "%s/model-ud-iq1m-tp16-rank-00.gguf", R[0]->path);
        std::vector<uint8_t> rowb((size_t)row_bytes);
        FILE* ef = fopen(efile, "rb");
        if (!ef ||
            fseek(ef, (long)(tv->file_offset + (uint64_t)802 * row_bytes),
                  SEEK_SET) ||
            fread(rowb.data(), 1, rowb.size(), ef) != rowb.size()) {
            fprintf(stderr, "embed read fail\n"); return 1;
        }
        fclose(ef);
        hy4_dequant_row_q4_K((const block_q4_K*)rowb.data(), h_embd.data(),
                             N_EMBD);
    }

    float *d_streams, *d_flat, *d_mixes, *d_pre, *d_post, *d_red, *d_cur,
          *d_ss, *d_branch, *d_q, *d_qr, *d_qh_all, *d_gatev, *d_kv,
          *d_klat, *d_kpe, *d_qabs, *d_vlat, *d_attn, *d_partial, *d_acc,
          *d_an, *d_fn2, *d_hc_fn, *d_hc_sc, *d_hc_base, *d_sinks,
          *d_wqa, *d_wkva, *d_wqb, *d_wkb, *d_wvb, *d_wgate, *d_wo,
          *d_wqa_n, *d_wkva_n, *d_ginp, *d_logits, *d_wg, *d_wu, *d_wd,
          *d_ge, *d_ue, *d_he, *d_wshg, *d_wshu, *d_wshd, *d_wdg, *d_wdu,
          *d_wdd, *d_dg, *d_du, *d_qpe, *d_s, *d_ones;
    CHECK_CUDA(cudaMalloc(&d_streams, (size_t)HC * N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_flat, (size_t)HC * N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_mixes, 8 * 4));
    CHECK_CUDA(cudaMalloc(&d_pre, 4 * 4));
    CHECK_CUDA(cudaMalloc(&d_post, 4 * 4));
    CHECK_CUDA(cudaMalloc(&d_red, N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_cur, N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_ss, 4));
    CHECK_CUDA(cudaMalloc(&d_branch, N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_q, 2048 * 4));
    CHECK_CUDA(cudaMalloc(&d_qr, 2048 * 4));
    CHECK_CUDA(cudaMalloc(&d_qh_all, 1024 * 4));
    CHECK_CUDA(cudaMalloc(&d_gatev, 1024 * 4));
    CHECK_CUDA(cudaMalloc(&d_kv, 576 * 4));
    CHECK_CUDA(cudaMalloc(&d_klat, KV_LORA * 4));
    CHECK_CUDA(cudaMalloc(&d_kpe, ROT * 4));
    CHECK_CUDA(cudaMalloc(&d_qabs, KV_LORA * 4));
    CHECK_CUDA(cudaMalloc(&d_vlat, KV_LORA * 4));
    CHECK_CUDA(cudaMalloc(&d_attn, 1024 * 4));
    CHECK_CUDA(cudaMalloc(&d_partial, N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_acc, N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_an, N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_fn2, N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_hc_fn, (size_t)HC * N_EMBD * 8 * 4));
    CHECK_CUDA(cudaMalloc(&d_hc_sc, 8));
    CHECK_CUDA(cudaMalloc(&d_hc_base, 32));
    CHECK_CUDA(cudaMalloc(&d_sinks, 64 * 4));
    CHECK_CUDA(cudaMalloc(&d_wqa, (size_t)2048 * N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_wkva, (size_t)576 * N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_wqb, (size_t)1024 * 2048 * 4));
    CHECK_CUDA(cudaMalloc(&d_wkb, (size_t)4 * KV_LORA * NOPE * 4));
    CHECK_CUDA(cudaMalloc(&d_wvb, (size_t)4 * VD * KV_LORA * 4));
    CHECK_CUDA(cudaMalloc(&d_wgate, (size_t)1024 * N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_wo, (size_t)N_EMBD * 1024 * 4));
    CHECK_CUDA(cudaMalloc(&d_wqa_n, 2048 * 4));
    CHECK_CUDA(cudaMalloc(&d_wkva_n, KV_LORA * 4));
    CHECK_CUDA(cudaMalloc(&d_ginp, (size_t)N_EXPERT * N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_logits, N_EXPERT * 4));
    CHECK_CUDA(cudaMalloc(&d_wg, (size_t)N_FF * N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_wu, (size_t)N_FF * N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_wd, (size_t)N_EMBD * N_FF * 4));
    CHECK_CUDA(cudaMalloc(&d_ge, N_FF * 4));
    CHECK_CUDA(cudaMalloc(&d_ue, N_FF * 4));
    CHECK_CUDA(cudaMalloc(&d_he, N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_wshg, (size_t)N_FF * N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_wshu, (size_t)N_FF * N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_wshd, (size_t)N_EMBD * N_FF * 4));
    CHECK_CUDA(cudaMalloc(&d_wdg, (size_t)18432 * N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_wdu, (size_t)18432 * N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_wdd, (size_t)N_EMBD * 18432 * 4));
    CHECK_CUDA(cudaMalloc(&d_dg, 18432 * 4));
    CHECK_CUDA(cudaMalloc(&d_du, 18432 * 4));
    CHECK_CUDA(cudaMalloc(&d_qpe, ROT * 4));
    CHECK_CUDA(cudaMalloc(&d_s, 8));
    CHECK_CUDA(cudaMalloc(&d_ones, (size_t)HC * N_EMBD * 4));
    {
        std::vector<float> init((size_t)HC * N_EMBD);
        std::vector<float> one((size_t)HC * N_EMBD, 1.0f);
        for (int s = 0; s < HC; ++s)
            memcpy(init.data() + (size_t)s * N_EMBD, h_embd.data(),
                   N_EMBD * 4);
        cudaMemcpy(d_streams, init.data(), (size_t)HC * N_EMBD * 4,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_ones, one.data(), (size_t)HC * N_EMBD * 4,
                   cudaMemcpyHostToDevice);
    }

    char nm[160];
    std::vector<float> hhost((size_t)HC * N_EMBD * 8);
    int bad = 0;
    std::vector<float> gpu((size_t)HC * N_EMBD);
    std::vector<float> sinks_h(64);

    for (int il = 0; il < LAYERS; ++il) {
        snprintf(nm, sizeof(nm), "blk.%d.hc_attn_fn.weight", il);
        load_f32(R[0], nm, hhost.data(), hhost.size());
        cudaMemcpy(d_hc_fn, hhost.data(), hhost.size() * 4,
                   cudaMemcpyHostToDevice);
        snprintf(nm, sizeof(nm), "blk.%d.hc_attn_scale.weight", il);
        load_f32_dev(R[0], nm, d_hc_sc, 2);
        snprintf(nm, sizeof(nm), "blk.%d.hc_attn_base.weight", il);
        load_f32_dev(R[0], nm, d_hc_base, 8);
        hc_pre_gpu(d_streams, d_flat, d_hc_fn, (float*)d_hc_sc,
                   (float*)d_hc_base, d_mixes, d_pre, d_post, d_red, d_cur,
                   d_ss, N_EMBD, HC);

        snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", il);
        load_f32_dev(R[0], nm, d_an, N_EMBD);
        if (il == 0) {
            float dbg[11];
            cudaMemcpy(dbg, d_flat, 12, cudaMemcpyDeviceToHost);
            cudaMemcpy(dbg + 3, d_red, 12, cudaMemcpyDeviceToHost);
            cudaMemcpy(dbg + 6, d_mixes, 32, cudaMemcpyDeviceToHost);
            cudaMemcpy(dbg + 8, d_pre, 16, cudaMemcpyDeviceToHost);
            printf("PROBE flat[0..2]=%.4f %.4f %.4f red[0..2]=%.4f %.4f "
                   "%.4f\n", dbg[0], dbg[1], dbg[2], dbg[3], dbg[4],
                   dbg[5]);
            printf("PROBE mixes=%.2f %.2f %.2f %.2f pre=%.4f %.4f %.4f "
                   "%.4f\n", dbg[6], dbg[7], dbg[8], dbg[9], dbg[8 + 4],
                   dbg[9 + 4], dbg[10], dbg[11]);
        }
        k_rms_sq<<<1, 256>>>(d_red, d_ss, N_EMBD);
        k_rms_scale<<<N_EMBD / 256, 256>>>(d_red, (float*)d_an, d_cur,
                                           N_EMBD, 1e-5f, d_ss);

        snprintf(nm, sizeof(nm), "blk.%d.attn_q_a.weight", il);
        upload_dequant(R[0], nm, d_wqa, (long)2048 * N_EMBD);
        snprintf(nm, sizeof(nm), "blk.%d.attn_q_a_norm.weight", il);
        load_f32_dev(R[0], nm, d_wqa_n, 2048);
        k_gemv<<<(2048 + 127) / 128, 128>>>(d_wqa, d_cur, d_q, 2048, N_EMBD);
        k_rms_sq<<<1, 256>>>(d_q, d_ss, 2048);
        k_rms_scale<<<2048 / 256, 256>>>(d_q, (float*)d_wqa_n, d_qr, 2048,
                                         1e-5f, d_ss);

        snprintf(nm, sizeof(nm), "blk.%d.attn_kv_a_mqa.weight", il);
        upload_dequant(R[0], nm, d_wkva, (long)576 * N_EMBD);
        k_gemv<<<(576 + 127) / 128, 128>>>(d_wkva, d_cur, d_kv, 576, N_EMBD);
        snprintf(nm, sizeof(nm), "blk.%d.attn_kv_a_norm.weight", il);
        load_f32_dev(R[0], nm, d_wkva_n, KV_LORA);
        k_rms_sq<<<1, 256>>>(d_kv, d_ss, KV_LORA);
        k_rms_scale<<<KV_LORA / 256, 256>>>(d_kv, (float*)d_wkva_n, d_klat,
                                            KV_LORA, 1e-5f, d_ss);
        if (il == 0) {
            float dbg[6];
            cudaMemcpy(dbg, d_cur, 12, cudaMemcpyDeviceToHost);
            cudaMemcpy(dbg + 3, d_kv, 12, cudaMemcpyDeviceToHost);
            printf("PROBE cur[0..2]=%.6f %.6f %.6f kvraw[0..2]=%.6f "
                   "%.6f %.6f\n", dbg[0], dbg[1], dbg[2], dbg[3], dbg[4],
                   dbg[5]);
        }
        cudaMemcpy(d_kpe, d_kv + KV_LORA, ROT * 4, cudaMemcpyDeviceToDevice);
        k_rope<<<1, 32>>>(d_kpe, ROT, 0.0f);
        snprintf(nm, sizeof(nm), "blk.%d.attn_sinks.weight", il);
        load_f32(R[0], nm, sinks_h.data(), 64);

        cudaMemset(d_acc, 0, N_EMBD * 4);
        for (int r = 0; r < N_RANKS; ++r) {
            snprintf(nm, sizeof(nm), "blk.%d.attn_q_b.weight", il);
            upload_dequant(R[r], nm, d_wqb, (long)1024 * 2048);
            snprintf(nm, sizeof(nm), "blk.%d.attn_k_b.weight", il);
            upload_dequant(R[r], nm, d_wkb, (long)4 * KV_LORA * NOPE);
            snprintf(nm, sizeof(nm), "blk.%d.attn_v_b.weight", il);
            upload_dequant(R[r], nm, d_wvb, (long)4 * VD * KV_LORA);
            snprintf(nm, sizeof(nm), "blk.%d.attn_gate.weight", il);
            upload_dequant(R[r], nm, d_wgate, (long)1024 * N_EMBD);
            snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", il);
            upload_dequant(R[r], nm, d_wo, (long)N_EMBD * 1024);
            k_gemv<<<(1024 + 127) / 128, 128>>>(d_wqb, d_qr, d_qh_all, 1024,
                                                2048);
            k_gemv<<<(1024 + 127) / 128, 128>>>(d_wgate, d_cur, d_gatev,
                                                1024, N_EMBD);
            k_sigmoid<<<(1024 + 127) / 128, 128>>>(d_gatev, 1024);
            for (int lh = 0; lh < RANK_HEADS; ++lh) {
                k_gemv<<<(KV_LORA + 127) / 128, 128>>>(
                    d_wkb + (size_t)lh * KV_LORA * NOPE,
                    d_qh_all + lh * HK, d_qabs, KV_LORA, NOPE);
                k_dot<<<1, 1>>>(d_qabs, d_klat, d_s, KV_LORA);
                k_dot<<<1, 1>>>(d_qh_all + lh * HK + NOPE, d_kpe, d_s + 1,
                                ROT);
                cudaError_t derr = cudaDeviceSynchronize();
                if (derr != cudaSuccess) {
                    printf("dot sync fail: %s\n",
                           cudaGetErrorString(derr));
                    return 1;
                }
                cudaError_t lerr = cudaGetLastError();
                if (lerr != cudaSuccess) {
                    printf("dot launch fail: %s\n",
                           cudaGetErrorString(lerr));
                    return 1;
                }
                float sh[2];
                cudaMemcpy(sh, d_s, 8, cudaMemcpyDeviceToHost);
                float sc = (sh[0] + sh[1]) / 16.0f;
                float snk = sinks_h[r * RANK_HEADS + lh];
                float M = sc > snk ? sc : snk;
                float p = expf(sc - M) / (expf(sc - M) + expf(snk - M));
                if (il == 0 && r == 0 && lh == 0) {
                    float dbg[9];
                    cudaMemcpy(dbg, d_qh_all, 12, cudaMemcpyDeviceToHost);
                    cudaMemcpy(dbg + 3, d_qabs, 12, cudaMemcpyDeviceToHost);
                    cudaMemcpy(dbg + 6, d_klat, 12, cudaMemcpyDeviceToHost);
                    printf("PROBE h0 sc=%.6f sink=%.6f p=%.6f\n", sc, snk,
                           p);
                    printf("PROBE qh[0..2]=%.6f %.6f %.6f qabs[0..2]=%.6f "
                           "%.6f %.6f klat[0..2]=%.6f %.6f %.6f\n",
                           dbg[0], dbg[1], dbg[2], dbg[3], dbg[4], dbg[5],
                           dbg[6], dbg[7], dbg[8]);
                }
                cudaMemcpy(d_vlat, d_klat, KV_LORA * 4,
                           cudaMemcpyDeviceToDevice);
                k_scale<<<KV_LORA / 256, 256>>>(d_vlat, p, KV_LORA);
                k_gemv<<<(VD + 127) / 128, 128>>>(
                    d_wvb + (size_t)lh * VD * KV_LORA, d_vlat,
                    d_attn + lh * VD, VD, KV_LORA);
            }
            k_mul<<<(1024 + 127) / 128, 128>>>(d_attn, d_gatev, 1024);
            k_gemv<<<(N_EMBD + 127) / 128, 128>>>(d_wo, d_attn, d_partial,
                                                  N_EMBD, 1024);
            k_axpy<<<N_EMBD / 256, 256>>>(d_acc, d_partial, 1.0f, N_EMBD);
        }
        k_hc_distribute<<<N_EMBD / 256, 256>>>(d_streams, d_acc, d_post,
                                               N_EMBD, HC);
        {
            std::vector<float> st((size_t)HC * N_EMBD);
            cudaMemcpy(st.data(), d_streams, (size_t)HC * N_EMBD * 4,
                       cudaMemcpyDeviceToHost);
            double s = 0;
            for (size_t i = 0; i < st.size(); ++i) s += st[i];
            printf("GPUSUM post-attn L%d t0: %.6f first3 %.6f %.6f %.6f\n",
                   il, s, st[0], st[1], st[2]);
        }

        snprintf(nm, sizeof(nm), "blk.%d.hc_ffn_fn.weight", il);
        load_f32(R[0], nm, hhost.data(), hhost.size());
        cudaMemcpy(d_hc_fn, hhost.data(), hhost.size() * 4,
                   cudaMemcpyHostToDevice);
        snprintf(nm, sizeof(nm), "blk.%d.hc_ffn_scale.weight", il);
        load_f32_dev(R[0], nm, d_hc_sc, 2);
        snprintf(nm, sizeof(nm), "blk.%d.hc_ffn_base.weight", il);
        load_f32_dev(R[0], nm, d_hc_base, 8);
        hc_pre_gpu(d_streams, d_flat, d_hc_fn, (float*)d_hc_sc,
                   (float*)d_hc_base, d_mixes, d_pre, d_post, d_red, d_cur,
                   d_ss, N_EMBD, HC);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_norm.weight", il);
        load_f32_dev(R[0], nm, d_fn2, N_EMBD);
        k_rms_sq<<<1, 256>>>(d_red, d_ss, N_EMBD);
        k_rms_scale<<<N_EMBD / 256, 256>>>(d_red, (float*)d_fn2, d_cur,
                                           N_EMBD, 1e-5f, d_ss);

        if (il == 0) {
            snprintf(nm, sizeof(nm), "blk.%d.ffn_gate.weight", il);
            upload_dequant(R[0], nm, d_wdg, (long)18432 * N_EMBD);
            snprintf(nm, sizeof(nm), "blk.%d.ffn_up.weight", il);
            upload_dequant(R[0], nm, d_wdu, (long)18432 * N_EMBD);
            snprintf(nm, sizeof(nm), "blk.%d.ffn_down.weight", il);
            upload_dequant(R[0], nm, d_wdd, (long)N_EMBD * 18432);
            k_gemv<<<(18432 + 127) / 128, 128>>>(d_wdg, d_cur, d_dg, 18432,
                                                 N_EMBD);
            k_gemv<<<(18432 + 127) / 128, 128>>>(d_wdu, d_cur, d_du, 18432,
                                                 N_EMBD);
            k_swiglu<<<(18432 + 127) / 128, 128>>>(d_dg, d_du, 18432,
                                                   3.4e38f);
            k_gemv<<<(N_EMBD + 127) / 128, 128>>>(d_wdd, d_du, d_branch,
                                                  N_EMBD, 18432);
        } else {
            snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp.weight", il);
            upload_dequant(R[0], nm, d_ginp, (long)N_EXPERT * N_EMBD);
            k_gemv<<<(N_EXPERT + 127) / 128, 128>>>(d_ginp, d_cur, d_logits,
                                                    N_EXPERT, N_EMBD);
            std::vector<float> logits(N_EXPERT);
            cudaMemcpy(logits.data(), d_logits, N_EXPERT * 4,
                       cudaMemcpyDeviceToHost);
            std::vector<float> eb(N_EXPERT);
            snprintf(nm, sizeof(nm), "blk.%d.exp_probs_b.bias", il);
            load_f32(R[0], nm, eb.data(), N_EXPERT);
            int sel[N_USED];
            double ww[N_USED];
            {
                std::vector<double> probs(N_EXPERT), key(N_EXPERT);
                for (int e = 0; e < N_EXPERT; ++e) {
                    probs[e] = 1.0 / (1.0 + exp(-(double)logits[e]));
                    key[e] = probs[e] + eb[e];
                }
                double wsum = 0;
                for (int k = 0; k < N_USED; ++k) {
                    int best = -1;
                    double bv = -1e300;
                    for (int e = 0; e < N_EXPERT; ++e)
                        if (key[e] > bv) { bv = key[e]; best = e; }
                    key[best] = -1e300;
                    sel[k] = best;
                    ww[k] = probs[best];
                    wsum += ww[k];
                }
                if (wsum < 6.103515625e-5) wsum = 6.103515625e-5;
                for (int k = 0; k < N_USED; ++k)
                    ww[k] = ww[k] / wsum * 2.827;
            }
            cudaMemset(d_branch, 0, N_EMBD * 4);
            for (int k = 0; k < N_USED; ++k) {
                const int e = sel[k];
                const int owner = e / 16;
                const int li = e % 16;
                snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_exps.weight", il);
                const hy4_tensor_view* tvg = hy4_tensor_lookup(R[owner], nm);
                snprintf(nm, sizeof(nm), "blk.%d.ffn_up_exps.weight", il);
                const hy4_tensor_view* tvu = hy4_tensor_lookup(R[owner], nm);
                snprintf(nm, sizeof(nm), "blk.%d.ffn_down_exps.weight", il);
                const hy4_tensor_view* tvd = hy4_tensor_lookup(R[owner], nm);
                long sgb = tvg->nbytes / 16, sub = tvu->nbytes / 16,
                     sdb = tvd->nbytes / 16;
                std::vector<uint8_t> graw((size_t)sgb), uraw((size_t)sub),
                    draw((size_t)sdb);
                char ofile[600];
                snprintf(ofile, sizeof(ofile),
                         "%s/model-ud-iq1m-tp16-rank-%02d.gguf",
                         R[owner]->path, owner);
                FILE* of = fopen(ofile, "rb");
                if (!of ||
                    fseek(of, (long)(tvg->file_offset + (uint64_t)li * sgb),
                          SEEK_SET) ||
                    fread(graw.data(), 1, (size_t)sgb, of) != (size_t)sgb) {
                    printf("slab read fail e=%d owner=%d li=%d sgb=%ld "
                           "path=%s off=%lld err=%s\n", e, owner, li, sgb,
                           ofile,
                           (long long)(tvg->file_offset +
                                       (uint64_t)li * sgb),
                           strerror(errno));
                    return 1;
                }
                fseek(of, (long)(tvu->file_offset + (uint64_t)li * sub),
                      SEEK_SET);
                if (fread(uraw.data(), 1, (size_t)sub, of) != (size_t)sub) {
                    printf("slab read fail u\n"); return 1;
                }
                fseek(of, (long)(tvd->file_offset + (uint64_t)li * sdb),
                      SEEK_SET);
                if (fread(draw.data(), 1, (size_t)sdb, of) != (size_t)sdb) {
                    printf("slab read fail d\n"); return 1;
                }
                fclose(of);
                std::vector<float> wg_h((size_t)N_FF * N_EMBD);
                std::vector<float> wu_h((size_t)N_FF * N_EMBD);
                std::vector<float> wd_h((size_t)N_EMBD * N_FF);
                dequant_view(graw.data(), tvg->type, wg_h.data(),
                             (long)N_FF * N_EMBD);
                dequant_view(uraw.data(), tvu->type, wu_h.data(),
                             (long)N_FF * N_EMBD);
                dequant_view(draw.data(), tvd->type, wd_h.data(),
                             (long)N_EMBD * N_FF);
                cudaMemcpy(d_wg, wg_h.data(), (size_t)N_FF * N_EMBD * 4,
                           cudaMemcpyHostToDevice);
                cudaMemcpy(d_wu, wu_h.data(), (size_t)N_FF * N_EMBD * 4,
                           cudaMemcpyHostToDevice);
                cudaMemcpy(d_wd, wd_h.data(), (size_t)N_EMBD * N_FF * 4,
                           cudaMemcpyHostToDevice);
                k_gemv<<<(N_FF + 127) / 128, 128>>>(d_wg, d_cur, d_ge, N_FF,
                                                    N_EMBD);
                k_gemv<<<(N_FF + 127) / 128, 128>>>(d_wu, d_cur, d_ue, N_FF,
                                                    N_EMBD);
                k_swiglu<<<(N_FF + 127) / 128, 128>>>(d_ge, d_ue, N_FF,
                                                      10.0f);
                k_gemv<<<(N_EMBD + 127) / 128, 128>>>(d_wd, d_ue, d_he,
                                                      N_EMBD, N_FF);
                k_axpy<<<N_EMBD / 256, 256>>>(d_branch, d_he, (float)ww[k],
                                              N_EMBD);
            }
            snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_shexp.weight", il);
            upload_dequant(R[0], nm, d_wshg, (long)N_FF * N_EMBD);
            snprintf(nm, sizeof(nm), "blk.%d.ffn_up_shexp.weight", il);
            upload_dequant(R[0], nm, d_wshu, (long)N_FF * N_EMBD);
            snprintf(nm, sizeof(nm), "blk.%d.ffn_down_shexp.weight", il);
            upload_dequant(R[0], nm, d_wshd, (long)N_EMBD * N_FF);
            k_gemv<<<(N_FF + 127) / 128, 128>>>(d_wshg, d_cur, d_ge, N_FF,
                                                N_EMBD);
            k_gemv<<<(N_FF + 127) / 128, 128>>>(d_wshu, d_cur, d_ue, N_FF,
                                                N_EMBD);
            k_swiglu<<<(N_FF + 127) / 128, 128>>>(d_ge, d_ue, N_FF, 3.4e38f);
            k_gemv<<<(N_EMBD + 127) / 128, 128>>>(d_wshd, d_ue, d_he, N_EMBD,
                                                  N_FF);
            k_axpy<<<N_EMBD / 256, 256>>>(d_branch, d_he, 1.0f, N_EMBD);
        }
        snprintf(nm, sizeof(nm), "blk.%d.hc_ffn_scale.weight", il);
        load_f32_dev(R[0], nm, d_hc_sc, 2);
        snprintf(nm, sizeof(nm), "blk.%d.hc_ffn_base.weight", il);
        load_f32_dev(R[0], nm, d_hc_base, 8);
        k_hc_distribute<<<N_EMBD / 256, 256>>>(d_streams, d_branch, d_post,
                                               N_EMBD, HC);
        {
            std::vector<float> st((size_t)HC * N_EMBD);
            cudaMemcpy(st.data(), d_streams, (size_t)HC * N_EMBD * 4,
                       cudaMemcpyDeviceToHost);
            double s = 0;
            for (size_t i = 0; i < st.size(); ++i) s += st[i];
            printf("GPUSUM post-ffn  L%d t0: %.6f first3 %.6f %.6f %.6f\n",
                   il, s, st[0], st[1], st[2]);
        }
        if (il == 0)
            printf("L0 done\n");
    }

    cudaDeviceSynchronize();
    gpu.resize((size_t)HC * N_EMBD);
    cudaMemcpy(gpu.data(), d_streams, (size_t)HC * N_EMBD * 4,
               cudaMemcpyDeviceToHost);
    bad += check("LAYER_STATE", gpu.data(), ref.data(),
                 (long)HC * N_EMBD, 2e-3);
    printf("QLAYER %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
