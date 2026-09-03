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
#define LAYERS 78

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
__global__ void k_attn_head_multi(const float* q_abs, const float* q_pe,
                                  const float* klat, const float* kpe,
                                  float sink, int T, float scale,
                                  float* vlat) {
    float scores[64];
    float M = sink;
    for (int tk = 0; tk < T; ++tk) {
        float s = 0;
        for (int i = 0; i < 512; ++i) s += q_abs[i] * klat[tk * 512 + i];
        for (int i = 0; i < 64; ++i) s += q_pe[i] * kpe[tk * 64 + i];
        scores[tk] = s * scale;
        if (scores[tk] > M) M = scores[tk];
    }
    float denom = expf(sink - M);
    float ps[64];
    for (int tk = 0; tk < T; ++tk) {
        ps[tk] = expf(scores[tk] - M);
        denom += ps[tk];
    }
    for (int i = 0; i < 512; ++i) {
        float acc = 0;
        for (int tk = 0; tk < T; ++tk)
            acc += ps[tk] / denom * klat[tk * 512 + i];
        vlat[i] = acc;
    }
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
    cudaError_t uerr = cudaMemcpy(d_dst, h.data(), (size_t)nelem * 4,
                                  cudaMemcpyHostToDevice);
    if (uerr != cudaSuccess)
        fprintf(stderr, "UPLOAD FAIL %s nelem=%ld dst=%p err=%s\n", name,
                nelem, (void*)d_dst, cudaGetErrorString(uerr));
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <allranks_dir> <expected_top1>\n",
                argv[0]);
        return 2;
    }
    const int expected_top1 = atoi(argv[1 + 1]);
    hy4_rank* R[N_RANKS];
    char pdir[512];
    for (int r = 0; r < N_RANKS; ++r) {
        snprintf(pdir, sizeof(pdir), "%s/rank-%02d", argv[1], r);
        if (hy4_rank_open(pdir, 1, &R[r])) {
            fprintf(stderr, "open rank %d\n", r);
            return 1;
        }
    }

    const int T = 4;
    const int tokens[4] = {802, 5466, 19405, 63357};

    float *d_streams_all, *d_flat, *d_mixes, *d_pre, *d_post, *d_red,
          *d_cur, *d_ss, *d_branch, *d_q, *d_qr, *d_qh_all, *d_gatev,
          *d_kv, *d_klat, *d_kpe, *d_qabs, *d_vlat, *d_attn, *d_partial,
          *d_acc, *d_an, *d_fn2, *d_hc_fn, *d_hc_sc, *d_hc_base, *d_sinks,
          *d_cur_all, *d_qabs_all, *d_vlat_all, *d_attn_all, *d_acc_all,
          *d_wdg, *d_wdu, *d_wdd, *d_dg, *d_du, *d_hh_mix, *d_hh_pre,
          *d_coll,
          *d_wqa, *d_wkva, *d_wqb, *d_wkb, *d_wvb, *d_wgate, *d_wo,
          *d_wqa_n, *d_wkva_n, *d_ginp, *d_logits, *d_wg, *d_wu, *d_wd,
          *d_ge, *d_ue, *d_he, *d_wshg, *d_wshu, *d_wshd, *d_qpe, *d_s,
          *d_ones, *d_normed, *d_klat_all, *d_kpe_all, *d_qr_all,
          *d_gatev_all, *d_hhead_fn, *d_hh_sc, *d_hh_base, *d_onorm,
          *d_outw;
    CHECK_CUDA(cudaMalloc(&d_streams_all, (size_t)T * HC * N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_flat, (size_t)HC * N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_mixes, 8 * 4));
    CHECK_CUDA(cudaMalloc(&d_pre, 4 * 4));
    CHECK_CUDA(cudaMalloc(&d_post, 4 * 4));
    CHECK_CUDA(cudaMalloc(&d_red, (size_t)7552 * 4));
    CHECK_CUDA(cudaMalloc(&d_cur, N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_ss, 4));
    CHECK_CUDA(cudaMalloc(&d_branch, (size_t)T * N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_q, 2048 * 4));
    CHECK_CUDA(cudaMalloc(&d_qr, 2048 * 4));
    CHECK_CUDA(cudaMalloc(&d_qh_all, (size_t)T * 1024 * 4));
    CHECK_CUDA(cudaMalloc(&d_gatev_all, (size_t)T * 1024 * 4));
    CHECK_CUDA(cudaMalloc(&d_cur_all, (size_t)T * N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_qr_all, (size_t)T * 2048 * 4));
    CHECK_CUDA(cudaMalloc(&d_wdg, (size_t)18432 * N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_wdu, (size_t)18432 * N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_wdd, (size_t)N_EMBD * 18432 * 4));
    CHECK_CUDA(cudaMalloc(&d_dg, 18432 * 4));
    CHECK_CUDA(cudaMalloc(&d_du, 18432 * 4));
    CHECK_CUDA(cudaMalloc(&d_kv, 576 * 4));
    CHECK_CUDA(cudaMalloc(&d_klat, KV_LORA * 4));
    CHECK_CUDA(cudaMalloc(&d_kpe, ROT * 4));
    CHECK_CUDA(cudaMalloc(&d_qabs_all, (size_t)T * KV_LORA * 4));
    CHECK_CUDA(cudaMalloc(&d_vlat_all, (size_t)T * KV_LORA * 4));
    CHECK_CUDA(cudaMalloc(&d_attn_all, (size_t)T * 1024 * 4));
    CHECK_CUDA(cudaMalloc(&d_partial, N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_acc_all, (size_t)T * N_EMBD * 4));
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
    CHECK_CUDA(cudaMalloc(&d_qpe, ROT * 4));
    CHECK_CUDA(cudaMalloc(&d_s, 8));
    CHECK_CUDA(cudaMalloc(&d_ones, (size_t)HC * N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_normed, N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_klat_all, (size_t)LAYERS * T * KV_LORA * 4));
    CHECK_CUDA(cudaMalloc(&d_kpe_all, (size_t)LAYERS * T * ROT * 4));
    {
        std::vector<float> init((size_t)T * HC * N_EMBD);
        std::vector<float> one((size_t)HC * N_EMBD, 1.0f);
        for (int t = 0; t < T; ++t) {
            const int owner = tokens[t] / 7552;
            const int row = tokens[t] % 7552;
            const hy4_tensor_view* tv =
                hy4_tensor_lookup(R[owner], "token_embd.weight");
            long row_bytes = tv->nbytes / 7552;
            char efile[600];
            snprintf(efile, sizeof(efile),
                     "%s/model-ud-iq1m-tp16-rank-%02d.gguf", R[owner]->path,
                     owner);
            std::vector<uint8_t> rowb((size_t)row_bytes);
            FILE* ef = fopen(efile, "rb");
            if (!ef ||
                fseek(ef, (long)(tv->file_offset + (uint64_t)row * row_bytes),
                      SEEK_SET) ||
                fread(rowb.data(), 1, rowb.size(), ef) != rowb.size()) {
                printf("embed read fail t%d\n", t); return 1;
            }
            fclose(ef);
            std::vector<float> e0(N_EMBD);
            hy4_dequant_row_q4_K((const block_q4_K*)rowb.data(), e0.data(),
                                 N_EMBD);
            for (int s = 0; s < HC; ++s)
                memcpy(init.data() + ((size_t)t * HC + s) * N_EMBD,
                       e0.data(), N_EMBD * 4);
        }
        cudaMemcpy(d_streams_all, init.data(), (size_t)T * HC * N_EMBD * 4,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_ones, one.data(), (size_t)HC * N_EMBD * 4,
                   cudaMemcpyHostToDevice);
    }

    char nm[160];
    std::vector<float> hhost((size_t)HC * N_EMBD * 8);
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
        snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", il);
        load_f32_dev(R[0], nm, d_an, N_EMBD);
        snprintf(nm, sizeof(nm), "blk.%d.attn_sinks.weight", il);
        load_f32_dev(R[0], nm, d_sinks, 64);
        std::vector<float> sinks_h(64);
        {
            const hy4_tensor_view* tv = hy4_tensor_lookup(R[0], nm);
            hy4_tensor_read(R[0], tv, sinks_h.data());
        }
        snprintf(nm, sizeof(nm), "blk.%d.attn_q_a.weight", il);
        upload_dequant(R[0], nm, d_wqa, (long)2048 * N_EMBD);
        snprintf(nm, sizeof(nm), "blk.%d.attn_q_a_norm.weight", il);
        load_f32_dev(R[0], nm, d_wqa_n, 2048);
        snprintf(nm, sizeof(nm), "blk.%d.attn_kv_a_mqa.weight", il);
        upload_dequant(R[0], nm, d_wkva, (long)576 * N_EMBD);
        snprintf(nm, sizeof(nm), "blk.%d.attn_kv_a_norm.weight", il);
        load_f32_dev(R[0], nm, d_wkva_n, KV_LORA);

        for (int t = 0; t < T; ++t) {
            float* d_st = d_streams_all + (size_t)t * HC * N_EMBD;
            hc_pre_gpu(d_st, d_flat, d_hc_fn, (float*)d_hc_sc,
                       (float*)d_hc_base, d_mixes, d_pre, d_post, d_red,
                       nullptr, d_ss, N_EMBD, HC);
            k_rms_sq<<<1, 256>>>(d_red, d_ss, N_EMBD);
            k_rms_scale<<<N_EMBD / 256, 256>>>(d_red, (float*)d_an, d_cur,
                                               N_EMBD, 1e-5f, d_ss);
            cudaMemcpy(d_cur_all + (size_t)t * N_EMBD, d_cur, N_EMBD * 4,
                       cudaMemcpyDeviceToDevice);
            k_gemv<<<(2048 + 127) / 128, 128>>>(d_wqa, d_cur, d_q, 2048,
                                                N_EMBD);
            k_rms_sq<<<1, 256>>>(d_q, d_ss, 2048);
            k_rms_scale<<<2048 / 256, 256>>>(d_q, (float*)d_wqa_n,
                                             d_qr_all + (size_t)t * 2048,
                                             2048, 1e-5f, d_ss);
            k_gemv<<<(576 + 127) / 128, 128>>>(d_wkva, d_cur, d_kv, 576,
                                               N_EMBD);
            k_rms_sq<<<1, 256>>>(d_kv, d_ss, KV_LORA);
            k_rms_scale<<<KV_LORA / 256, 256>>>(d_kv, (float*)d_wkva_n,
                                                d_klat, KV_LORA, 1e-5f,
                                                d_ss);
            cudaMemcpy(d_kpe, d_kv + KV_LORA, ROT * 4,
                       cudaMemcpyDeviceToDevice);
            k_rope<<<1, 32>>>(d_kpe, ROT, (float)t);
            cudaMemcpy(d_klat_all + ((size_t)il * T + t) * KV_LORA, d_klat,
                       KV_LORA * 4, cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_kpe_all + ((size_t)il * T + t) * ROT, d_kpe,
                       ROT * 4, cudaMemcpyDeviceToDevice);
        }

        cudaMemset(d_acc_all, 0, (size_t)T * N_EMBD * 4);
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
            for (int t = 0; t < T; ++t) {
                k_gemv<<<(1024 + 127) / 128, 128>>>(
                    d_wqb, d_qr_all + (size_t)t * 2048,
                    d_qh_all + (size_t)t * 1024, 1024, 2048);
                k_gemv<<<(1024 + 127) / 128, 128>>>(
                    d_wgate, d_cur_all + (size_t)t * N_EMBD,
                    d_gatev_all + (size_t)t * 1024, 1024, N_EMBD);
                k_sigmoid<<<(1024 + 127) / 128, 128>>>(
                    d_gatev_all + (size_t)t * 1024, 1024);
            }
            for (int lh = 0; lh < RANK_HEADS; ++lh) {
                const float* kbs = d_wkb + (size_t)lh * KV_LORA * NOPE;
                const float* vbs = d_wvb + (size_t)lh * VD * KV_LORA;
                for (int t = 0; t < T; ++t) {
                    k_gemv<<<(KV_LORA + 127) / 128, 128>>>(
                        kbs, d_qh_all + (size_t)t * 1024 + lh * HK,
                        d_qabs_all + (size_t)t * KV_LORA, KV_LORA, NOPE);
                }
                for (int t = 0; t < T; ++t) {
                    int h = r * RANK_HEADS + lh;
                    float snk = sinks_h[h];
                    k_attn_head_multi<<<1, 1>>>(
                        d_qabs_all + (size_t)t * KV_LORA,
                        d_qh_all + (size_t)t * 1024 + lh * HK + NOPE,
                        d_klat_all + (size_t)il * T * KV_LORA,
                        d_kpe_all + (size_t)il * T * ROT, snk, t + 1,
                        1.0f / 16.0f, d_vlat_all + (size_t)t * KV_LORA);
                }
                for (int t = 0; t < T; ++t) {
                    k_gemv<<<(VD + 127) / 128, 128>>>(
                        vbs, d_vlat_all + (size_t)t * KV_LORA,
                        d_attn_all + (size_t)t * 1024 + lh * VD, VD,
                        KV_LORA);
                }
            }
            for (int t = 0; t < T; ++t) {
                k_mul<<<(1024 + 127) / 128, 128>>>(
                    d_attn_all + (size_t)t * 1024,
                    d_gatev_all + (size_t)t * 1024, 1024);
                k_gemv<<<(N_EMBD + 127) / 128, 128>>>(
                    d_wo, d_attn_all + (size_t)t * 1024, d_partial, N_EMBD,
                    1024);
                k_axpy<<<N_EMBD / 256, 256>>>(
                    d_acc_all + (size_t)t * N_EMBD, d_partial, 1.0f, N_EMBD);
            }
        }
        for (int t = 0; t < T; ++t) {
            float* d_st = d_streams_all + (size_t)t * HC * N_EMBD;
            k_hc_distribute<<<N_EMBD / 256, 256>>>(
                d_st, d_acc_all + (size_t)t * N_EMBD, d_post, N_EMBD, HC);
        }

        snprintf(nm, sizeof(nm), "blk.%d.hc_ffn_fn.weight", il);
        load_f32(R[0], nm, hhost.data(), hhost.size());
        cudaMemcpy(d_hc_fn, hhost.data(), hhost.size() * 4,
                   cudaMemcpyHostToDevice);
        snprintf(nm, sizeof(nm), "blk.%d.hc_ffn_scale.weight", il);
        load_f32_dev(R[0], nm, d_hc_sc, 2);
        snprintf(nm, sizeof(nm), "blk.%d.hc_ffn_base.weight", il);
        load_f32_dev(R[0], nm, d_hc_base, 8);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_norm.weight", il);
        load_f32_dev(R[0], nm, d_fn2, N_EMBD);
        for (int t = 0; t < T; ++t) {
            float* d_st = d_streams_all + (size_t)t * HC * N_EMBD;
            hc_pre_gpu(d_st, d_flat, d_hc_fn, (float*)d_hc_sc,
                       (float*)d_hc_base, d_mixes, d_pre, d_post, d_red,
                       nullptr, d_ss, N_EMBD, HC);
            k_rms_sq<<<1, 256>>>(d_red, d_ss, N_EMBD);
            k_rms_scale<<<N_EMBD / 256, 256>>>(d_red, (float*)d_fn2,
                                               d_cur_all +
                                                   (size_t)t * N_EMBD,
                                               N_EMBD, 1e-5f, d_ss);
        }

        if (il == 0) {
            snprintf(nm, sizeof(nm), "blk.%d.ffn_gate.weight", il);
            upload_dequant(R[0], nm, d_wdg, (long)18432 * N_EMBD);
            snprintf(nm, sizeof(nm), "blk.%d.ffn_up.weight", il);
            upload_dequant(R[0], nm, d_wdu, (long)18432 * N_EMBD);
            snprintf(nm, sizeof(nm), "blk.%d.ffn_down.weight", il);
            upload_dequant(R[0], nm, d_wdd, (long)N_EMBD * 18432);
            for (int t = 0; t < T; ++t) {
                k_gemv<<<(18432 + 127) / 128, 128>>>(
                    d_wdg, d_cur_all + (size_t)t * N_EMBD, d_dg, 18432,
                    N_EMBD);
                k_gemv<<<(18432 + 127) / 128, 128>>>(
                    d_wdu, d_cur_all + (size_t)t * N_EMBD, d_du, 18432,
                    N_EMBD);
                k_swiglu<<<(18432 + 127) / 128, 128>>>(d_dg, d_du, 18432,
                                                       3.4e38f);
                k_gemv<<<(N_EMBD + 127) / 128, 128>>>(
                    d_wdd, d_du, d_branch, N_EMBD, 18432);
                float* d_st = d_streams_all + (size_t)t * HC * N_EMBD;
                k_hc_distribute<<<N_EMBD / 256, 256>>>(
                    d_st, d_branch, d_post, N_EMBD, HC);
            }
        } else {
            std::vector<int> sel(T * N_USED);
            std::vector<double> ww(T * N_USED);
            snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp.weight", il);
            upload_dequant(R[0], nm, d_ginp, (long)N_EXPERT * N_EMBD);
            std::vector<float> eb(N_EXPERT);
            snprintf(nm, sizeof(nm), "blk.%d.exp_probs_b.bias", il);
            load_f32(R[0], nm, eb.data(), N_EXPERT);
            int slot_of[N_EXPERT];
            for (int e = 0; e < N_EXPERT; ++e) slot_of[e] = -1;
            int n_slots = 0;
            for (int t = 0; t < T; ++t) {
                k_gemv<<<(N_EXPERT + 127) / 128, 128>>>(
                    d_ginp, d_cur_all + (size_t)t * N_EMBD, d_logits,
                    N_EXPERT, N_EMBD);
                std::vector<float> logits(N_EXPERT);
                cudaMemcpy(logits.data(), d_logits, N_EXPERT * 4,
                           cudaMemcpyDeviceToHost);
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
                    sel[t * N_USED + k] = best;
                    ww[t * N_USED + k] = probs[best];
                    wsum += ww[t * N_USED + k];
                    if (slot_of[best] < 0) slot_of[best] = n_slots++;
                }
                if (wsum < 6.103515625e-5) wsum = 6.103515625e-5;
                for (int k = 0; k < N_USED; ++k)
                    ww[t * N_USED + k] = ww[t * N_USED + k] / wsum * 2.827;
            }
            for (int s = 0; s < n_slots; ++s) {
                int found_e = -1;
                for (int t = 0; t < T && found_e < 0; ++t)
                    for (int k = 0; k < N_USED; ++k)
                        if (slot_of[s] < 0) break;
                (void)found_e;
            }
            for (int slot = 0; slot < n_slots; ++slot) {
                int e = -1;
                for (int cand = 0; cand < N_EXPERT && e < 0; ++cand) {
                    if (slot_of[cand] == slot) e = cand;
                }
                if (e < 0) continue;
                const int owner = e / 16;
                const int li = e % 16;
                snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_exps.weight", il);
                const hy4_tensor_view* tvg =
                    hy4_tensor_lookup(R[owner], nm);
                snprintf(nm, sizeof(nm), "blk.%d.ffn_up_exps.weight", il);
                const hy4_tensor_view* tvu =
                    hy4_tensor_lookup(R[owner], nm);
                snprintf(nm, sizeof(nm), "blk.%d.ffn_down_exps.weight", il);
                const hy4_tensor_view* tvd =
                    hy4_tensor_lookup(R[owner], nm);
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
                    fseek(of, (long)(tvg->file_offset +
                                     (uint64_t)li * sgb),
                          SEEK_SET) ||
                    fread(graw.data(), 1, (size_t)sgb, of) != (size_t)sgb) {
                    printf("slab read fail\n"); return 1;
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
                for (int t = 0; t < T; ++t) {
                    int used_here = 0;
                    for (int k = 0; k < N_USED; ++k)
                        if (sel[t * N_USED + k] == e) used_here = 1;
                    if (!used_here) continue;
                    k_gemv<<<(N_FF + 127) / 128, 128>>>(
                        d_wg, d_cur_all + (size_t)t * N_EMBD, d_ge, N_FF,
                        N_EMBD);
                    k_gemv<<<(N_FF + 127) / 128, 128>>>(
                        d_wu, d_cur_all + (size_t)t * N_EMBD, d_ue, N_FF,
                        N_EMBD);
                    k_swiglu<<<(N_FF + 127) / 128, 128>>>(d_ge, d_ue, N_FF,
                                                          10.0f);
                    k_gemv<<<(N_EMBD + 127) / 128, 128>>>(
                        d_wd, d_ue, d_he, N_EMBD, N_FF);
                    float w_k = 0;
                    for (int k = 0; k < N_USED; ++k)
                        if (sel[t * N_USED + k] == e) w_k = (float)ww[t * N_USED + k];
                    k_axpy<<<N_EMBD / 256, 256>>>(
                        d_branch + (size_t)t * N_EMBD, d_he, w_k, N_EMBD);
                }
            }
            snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_shexp.weight", il);
            upload_dequant(R[0], nm, d_wshg, (long)N_FF * N_EMBD);
            snprintf(nm, sizeof(nm), "blk.%d.ffn_up_shexp.weight", il);
            upload_dequant(R[0], nm, d_wshu, (long)N_FF * N_EMBD);
            snprintf(nm, sizeof(nm), "blk.%d.ffn_down_shexp.weight", il);
            upload_dequant(R[0], nm, d_wshd, (long)N_EMBD * N_FF);
            for (int t = 0; t < T; ++t) {
                k_gemv<<<(N_FF + 127) / 128, 128>>>(
                    d_wshg, d_cur_all + (size_t)t * N_EMBD, d_ge, N_FF,
                    N_EMBD);
                k_gemv<<<(N_FF + 127) / 128, 128>>>(
                    d_wshu, d_cur_all + (size_t)t * N_EMBD, d_ue, N_FF,
                    N_EMBD);
                k_swiglu<<<(N_FF + 127) / 128, 128>>>(d_ge, d_ue, N_FF,
                                                      3.4e38f);
                k_gemv<<<(N_EMBD + 127) / 128, 128>>>(
                    d_wshd, d_ue, d_he, N_EMBD, N_FF);
                k_axpy<<<N_EMBD / 256, 256>>>(
                    d_branch + (size_t)t * N_EMBD, d_he, 1.0f, N_EMBD);
                float* d_st = d_streams_all + (size_t)t * HC * N_EMBD;
                k_hc_distribute<<<N_EMBD / 256, 256>>>(
                    d_st, d_branch + (size_t)t * N_EMBD, d_post, N_EMBD,
                    HC);
            }
        }
        if (il % 10 == 0)
            printf("layer %d done\n", il), fflush(stdout);
    }

    // final: hc_head collapse + output norm + lm_head for the last token
    int t = T - 1;
    float* d_st = d_streams_all + (size_t)t * HC * N_EMBD;
    CHECK_CUDA(cudaMalloc(&d_hh_mix, 4 * 4));
    CHECK_CUDA(cudaMalloc(&d_hh_pre, 4 * 4));
    CHECK_CUDA(cudaMalloc(&d_coll, N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_normed, N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_hhead_fn, (size_t)HC * N_EMBD * 4 * 4));
    CHECK_CUDA(cudaMalloc(&d_hh_sc, 4));
    CHECK_CUDA(cudaMalloc(&d_hh_base, 4 * 4));
    CHECK_CUDA(cudaMalloc(&d_onorm, N_EMBD * 4));
    load_f32_dev(R[0], "output_hc_fn.weight", d_hhead_fn,
                 (long)HC * HC * N_EMBD);
    load_f32_dev(R[0], "output_hc_scale.weight", d_hh_sc, 1);
    load_f32_dev(R[0], "output_hc_base.weight", d_hh_base, 4);
    k_rms_sq<<<1, 256>>>(d_st, d_ss, HC * N_EMBD);
    k_rms_scale<<<HC * N_EMBD / 256, 256>>>(d_st, nullptr, d_flat,
                                            HC * N_EMBD, 1e-5f, d_ss);
    k_gemv<<<1, 4>>>(d_hhead_fn, d_flat, d_hh_mix, 4, HC * N_EMBD);
    {
        float hm[4], hb[4], hs;
        cudaMemcpy(hm, d_hh_mix, 16, cudaMemcpyDeviceToHost);
        cudaMemcpy(&hs, d_hh_sc, 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(hb, d_hh_base, 16, cudaMemcpyDeviceToHost);
        float pre[4];
        for (int s = 0; s < HC; ++s)
            pre[s] = 1.0f / (1.0f + expf(-(hm[s] * hs + hb[s]))) + 1e-6f;
        fprintf(stderr, "HCHEADGPU t=%d: mixes %.4f %.4f %.4f %.4f | "
                "hpre %.4f %.4f %.4f %.4f\n", t, hm[0], hm[1], hm[2], hm[3],
                pre[0], pre[1], pre[2], pre[3]);
        cudaMemcpy(d_hh_pre, pre, 16, cudaMemcpyHostToDevice);
    }
    k_hc_reduce<<<N_EMBD / 256, 256>>>(d_st, (float*)d_hh_pre, d_coll,
                                       N_EMBD, HC);
    load_f32_dev(R[0], "output_norm.weight", d_onorm, N_EMBD);
    k_rms_sq<<<1, 256>>>(d_coll, d_ss, N_EMBD);
    k_rms_scale<<<N_EMBD / 256, 256>>>(d_coll, (float*)d_onorm, d_normed,
                                       N_EMBD, 1e-5f, d_ss);
    {
        float nf[3];
        cudaMemcpy(nf, d_normed, 12, cudaMemcpyDeviceToHost);
        fprintf(stderr, "L78GPU: normed %.4f %.4f %.4f\n", nf[0], nf[1],
                nf[2]);
    }

    const int vocab_per_rank = 7552;
    CHECK_CUDA(cudaMalloc(&d_outw, (size_t)vocab_per_rank * N_EMBD * 4));
    float top1_val = -1e30f;
    int top1_id = -1;
    std::vector<float> h_log(vocab_per_rank);
    for (int r = 0; r < N_RANKS; ++r) {
        upload_dequant(R[r], "output.weight", d_outw,
                       (long)vocab_per_rank * N_EMBD);
        k_gemv<<<(vocab_per_rank + 127) / 128, 128>>>(
            d_outw, d_normed, d_red, vocab_per_rank, N_EMBD);
        cudaMemcpy(h_log.data(), d_red, vocab_per_rank * 4,
                   cudaMemcpyDeviceToHost);
        for (int v = 0; v < vocab_per_rank; ++v) {
            if (h_log[v] > top1_val) {
                top1_val = h_log[v];
                top1_id = r * vocab_per_rank + v;
            }
        }
    }
    printf("TOP1 %d (expected %d) val %.4f\n", top1_id, expected_top1,
           top1_val);
    printf("FORWARD %s\n", top1_id == expected_top1 ? "PASS" : "FAIL");
    return top1_id == expected_top1 ? 0 : 1;
}
