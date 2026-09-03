// hy4 lane: GPU hyper-connection cell — hc_pre (rms over the flattened
// 24576 stream vector + hc_fn gemv + sigmoid gates + weighted reduce),
// hc_post (distribute branch*post into streams), and hc_head (collapse
// gates before output_norm), layer 0 / final, validated against an
// in-harness float64 reference on real rank-00 weights with the real
// hc_init (replicated embedding row 802). llama.cpp eval-callback golden
// for hc_mixes-0: [112.76, 76.82, 74.49, 95.14, ?, -282.97, -332.24,
// -281.27].
//
// Usage: hy4_hc_test <rank00.gguf>
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

__global__ void k_hc_gates(const float* mixes, const float* scale,
                           const float* base, float eps, float magnitude,
                           float* pre, float* post) {
    int i = threadIdx.x;
    if (i < 4) {
        pre[i] = 1.0f / (1.0f + expf(-(mixes[i] * scale[0] + base[i]))) + eps;
    } else {
        int j = i - 4;
        post[j] = magnitude / (1.0f + expf(-(mixes[4 + j] * scale[1] +
                                            base[4 + j]))) + eps;
    }
}

__global__ void k_hc_head_gates(const float* mixes, const float* scale,
                                const float* base, float eps, float* pre) {
    int i = threadIdx.x;
    if (i < 4)
        pre[i] = 1.0f / (1.0f + expf(-(mixes[i] * scale[0] + base[i]))) + eps;
}

__global__ void k_hc_reduce(const float* streams, const float* pre,
                            float* out, int n_embd, int hc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    float acc = 0;
    for (int s = 0; s < hc; ++s) acc += streams[(size_t)s * n_embd + i] * pre[s];
    out[i] = acc;
}

__global__ void k_hc_distribute(float* streams, const float* branch,
                                const float* post, int n_embd, int hc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    for (int s = 0; s < hc; ++s)
        streams[(size_t)s * n_embd + i] += branch[i] * post[s];
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

static const TensorInfo* find_tensor(const std::vector<TensorInfo>& t,
                                     const char* name) {
    for (size_t i = 0; i < t.size(); ++i)
        if (strcmp(t[i].name, name) == 0) return &t[i];
    return nullptr;
}

static int read_tensor_f32(FILE* f, const TensorInfo* ti, uint64_t off,
                           float* dst, size_t count) {
    return fseek(f, (long)(off + ti->offset), SEEK_SET) == 0 &&
           fread(dst, 4, count, f) == count ? 0 : -1;
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

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <rank00.gguf>\n", argv[0]);
        return 2;
    }
    std::vector<TensorInfo> tensors;
    uint64_t data_offset = 0;
    if (parse_rank_gguf(argv[1], tensors, &data_offset)) return 1;
    const int N_EMBD = 6144, HC = 4, FLAT = N_EMBD * HC;
    const double EPS = 1e-6, MAG = 2.0;

    FILE* f = fopen(argv[1], "rb");
    if (!f) return 1;
    std::vector<uint8_t> raw;
    const TensorInfo* ti;

    ti = find_tensor(tensors, "token_embd.weight");
    std::vector<uint8_t> embd_row((size_t)6144 / 256 * 144);
    if (fseek(f, (long)(data_offset + ti->offset +
                        (uint64_t)802 * embd_row.size()), SEEK_SET) ||
        fread(embd_row.data(), 1, embd_row.size(), f) != embd_row.size()) {
        printf("EMBD READ FAIL\n"); return 1;
    }
    std::vector<float> h_embd(6144);
    hy4_dequant_row_q4_K((const block_q4_K*)embd_row.data(), h_embd.data(),
                         6144);

    std::vector<float> h_fn(FLAT * 8), h_sc(2), h_base(8);
    ti = find_tensor(tensors, "blk.0.hc_attn_fn.weight");
    read_tensor_f32(f, ti, data_offset, h_fn.data(), h_fn.size());
    ti = find_tensor(tensors, "blk.0.hc_attn_scale.weight");
    read_tensor_f32(f, ti, data_offset, h_sc.data(), 2);
    ti = find_tensor(tensors, "blk.0.hc_attn_base.weight");
    read_tensor_f32(f, ti, data_offset, h_base.data(), 8);

    std::vector<float> h_hfn(FLAT * 4), h_hsc(1), h_hbase(4);
    ti = find_tensor(tensors, "output_hc_fn.weight");
    read_tensor_f32(f, ti, data_offset, h_hfn.data(), h_hfn.size());
    ti = find_tensor(tensors, "output_hc_scale.weight");
    read_tensor_f32(f, ti, data_offset, h_hsc.data(), 1);
    ti = find_tensor(tensors, "output_hc_base.weight");
    read_tensor_f32(f, ti, data_offset, h_hbase.data(), 4);

    std::vector<float> h_streams(FLAT);
    for (int s = 0; s < HC; ++s)
        memcpy(h_streams.data() + (size_t)s * N_EMBD, h_embd.data(),
               N_EMBD * 4);
    std::vector<double> ref_streams(FLAT);
    for (int i = 0; i < FLAT; ++i) ref_streams[i] = h_streams[i];

    float *d_streams, *d_fn, *d_sc, *d_base, *d_flat, *d_mixes, *d_pre,
          *d_post, *d_red, *d_branch, *d_ss, *d_ones, *d_hfn, *d_hsc,
          *d_hbase, *d_hmix, *d_hpre;
    CHECK_CUDA(cudaMalloc(&d_streams, FLAT * 4));
    CHECK_CUDA(cudaMalloc(&d_fn, (size_t)FLAT * 8 * 4));
    CHECK_CUDA(cudaMalloc(&d_sc, 8));
    CHECK_CUDA(cudaMalloc(&d_base, 32));
    CHECK_CUDA(cudaMalloc(&d_flat, FLAT * 4));
    CHECK_CUDA(cudaMalloc(&d_mixes, 8 * 4));
    CHECK_CUDA(cudaMalloc(&d_pre, 4 * 4));
    CHECK_CUDA(cudaMalloc(&d_post, 4 * 4));
    CHECK_CUDA(cudaMalloc(&d_red, N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_branch, N_EMBD * 4));
    CHECK_CUDA(cudaMalloc(&d_ss, 4));
    CHECK_CUDA(cudaMalloc(&d_ones, FLAT * 4));
    CHECK_CUDA(cudaMalloc(&d_hfn, (size_t)FLAT * 4 * 4));
    CHECK_CUDA(cudaMalloc(&d_hsc, 4));
    CHECK_CUDA(cudaMalloc(&d_hbase, 16));
    CHECK_CUDA(cudaMalloc(&d_hmix, 4 * 4));
    CHECK_CUDA(cudaMalloc(&d_hpre, 4 * 4));
    cudaMemcpy(d_streams, h_streams.data(), FLAT * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fn, h_fn.data(), (size_t)FLAT * 8 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sc, h_sc.data(), 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_base, h_base.data(), 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hfn, h_hfn.data(), (size_t)FLAT * 4 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hsc, h_hsc.data(), 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hbase, h_hbase.data(), 16, cudaMemcpyHostToDevice);
    std::vector<float> ones(FLAT, 1.0f);
    cudaMemcpy(d_ones, ones.data(), FLAT * 4, cudaMemcpyHostToDevice);

    int bad = 0;
    std::vector<float> gpu(FLAT);

    k_rms_sq<<<1, 256>>>(d_streams, d_ss, FLAT);
    k_rms_scale<<<FLAT / 256, 256>>>(d_streams, d_ones, d_flat, FLAT, 1e-5f,
                                     d_ss);
    k_gemv<<<1, 8>>>(d_fn, d_flat, d_mixes, 8, FLAT);
    k_hc_gates<<<1, 8>>>(d_mixes, d_sc, d_base, (float)EPS, (float)MAG,
                         d_pre, d_post);
    k_hc_reduce<<<N_EMBD / 256, 256>>>(d_streams, d_pre, d_red, N_EMBD, HC);

    cudaMemcpy(gpu.data(), d_mixes, 8 * 4, cudaMemcpyDeviceToHost);
    printf("HC_MIXES gpu   %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
           gpu[0], gpu[1], gpu[2], gpu[3], gpu[4], gpu[5], gpu[6], gpu[7]);
    printf("HC_MIXES llama 112.7601 76.8200 74.4942 95.1446 ? -282.9738 "
           "-332.2355 -281.2696\n");

    std::vector<double> ref_flat(FLAT), ref_mixes(8), ref_pre(HC),
        ref_post(HC), ref_red(N_EMBD);
    {
        double ss = 0;
        for (int i = 0; i < FLAT; ++i) ss += ref_streams[i] * ref_streams[i];
        double inv = 1.0 / sqrt(ss / FLAT + 1e-5);
        for (int i = 0; i < FLAT; ++i) ref_flat[i] = ref_streams[i] * inv;
        for (int r = 0; r < 8; ++r) {
            double acc = 0;
            for (int k = 0; k < FLAT; ++k)
                acc += (double)h_fn[(size_t)r * FLAT + k] * ref_flat[k];
            ref_mixes[r] = acc;
        }
        for (int s = 0; s < HC; ++s)
            ref_pre[s] = 1.0 / (1.0 + exp(-(ref_mixes[s] * h_sc[0] +
                                            h_base[s]))) + EPS;
        for (int s = 0; s < HC; ++s)
            ref_post[s] = MAG / (1.0 + exp(-(ref_mixes[4 + s] * h_sc[1] +
                                             h_base[4 + s]))) + EPS;
        for (int i = 0; i < N_EMBD; ++i) {
            double acc = 0;
            for (int s = 0; s < HC; ++s)
                acc += ref_streams[(size_t)s * N_EMBD + i] * ref_pre[s];
            ref_red[i] = acc;
        }
    }
    bad += check("HC_MIXES", gpu.data(), ref_mixes.data(), 8, 1e-3);
    gpu.resize(N_EMBD);
    cudaMemcpy(gpu.data(), d_red, N_EMBD * 4, cudaMemcpyDeviceToHost);
    bad += check("HC_REDUCED", gpu.data(), ref_red.data(), N_EMBD, 1e-3);

    std::vector<float> h_branch(N_EMBD);
    {
        uint32_t st = 777;
        for (int i = 0; i < N_EMBD; ++i) {
            st = st * 1664525u + 1013904223u;
            h_branch[i] = ((float)(st >> 8) / 16777216.0f - 0.5f) * 0.1f;
        }
    }
    cudaMemcpy(d_branch, h_branch.data(), N_EMBD * 4, cudaMemcpyHostToDevice);
    k_hc_distribute<<<N_EMBD / 256, 256>>>(d_streams, d_branch, d_post,
                                           N_EMBD, HC);
    for (int s = 0; s < HC; ++s)
        for (int i = 0; i < N_EMBD; ++i)
            ref_streams[(size_t)s * N_EMBD + i] +=
                (double)h_branch[i] * ref_post[s];
    gpu.resize(FLAT);
    cudaMemcpy(gpu.data(), d_streams, FLAT * 4, cudaMemcpyDeviceToHost);
    bad += check("HC_DISTRIBUTE", gpu.data(), ref_streams.data(), FLAT,
                 1e-3);

    k_rms_sq<<<1, 256>>>(d_streams, d_ss, FLAT);
    k_rms_scale<<<FLAT / 256, 256>>>(d_streams, d_ones, d_flat, FLAT, 1e-5f,
                                     d_ss);
    k_gemv<<<1, 4>>>(d_hfn, d_flat, d_hmix, 4, FLAT);
    k_hc_head_gates<<<1, 4>>>(d_hmix, d_hsc, d_hbase, (float)EPS, d_hpre);
    k_hc_reduce<<<N_EMBD / 256, 256>>>(d_streams, d_hpre, d_red, N_EMBD, HC);
    gpu.resize(N_EMBD);
    cudaMemcpy(gpu.data(), d_red, N_EMBD * 4, cudaMemcpyDeviceToHost);
    {
        std::vector<double> ref_flat2(FLAT), ref_hmix(4), ref_hpre(HC),
            ref(N_EMBD);
        double ss = 0;
        for (int i = 0; i < FLAT; ++i) ss += ref_streams[i] * ref_streams[i];
        double inv = 1.0 / sqrt(ss / FLAT + 1e-5);
        for (int i = 0; i < FLAT; ++i) ref_flat2[i] = ref_streams[i] * inv;
        for (int r = 0; r < 4; ++r) {
            double acc = 0;
            for (int k = 0; k < FLAT; ++k)
                acc += (double)h_hfn[(size_t)r * FLAT + k] * ref_flat2[k];
            ref_hmix[r] = acc;
        }
        for (int s = 0; s < HC; ++s)
            ref_hpre[s] = 1.0 / (1.0 + exp(-(ref_hmix[s] * h_hsc[0] +
                                             h_hbase[s]))) + EPS;
        for (int i = 0; i < N_EMBD; ++i) {
            double acc = 0;
            for (int s = 0; s < HC; ++s)
                acc += ref_streams[(size_t)s * N_EMBD + i] * ref_hpre[s];
            ref[i] = acc;
        }
        bad += check("HC_HEAD", gpu.data(), ref.data(), N_EMBD, 1e-3);
    }
    printf("HC %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
