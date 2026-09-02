// hy4 lane: GPU q-path chain cell — gemv + rms_norm + interleaved rope
// kernels validated against an in-harness float64 reference on real
// rank-00 weights (the exact chain the llama-diff tick proved in fp64:
// embd row 802 -> attn_norm -> q_a -> q_a_norm -> q_b head 0 -> rope).
//
// Usage: hy4_qchain_test <rank00.gguf>
// Per-stage PASS/FAIL plus a QCHAIN final verdict; tolerance is
// 1e-3 * max(1, |ref|) per element (fp32-vs-fp64 envelope — the lane
// exactness policy reserves bitwise for dequant only).
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
    y[i] = x[i] * inv * w[i];
}

__global__ void k_rope(float* v, int rot, float pos) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= rot / 2) return;
    float ang = pos * powf(1e7f, -2.0f * d / rot);
    float a = v[2 * d], b = v[2 * d + 1];
    v[2 * d] = a * cosf(ang) - b * sinf(ang);
    v[2 * d + 1] = a * sinf(ang) + b * cosf(ang);
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

static int bpb_bytes(uint32_t type) {
    switch (type) {
        case 12: return 144;
        case 13: return 176;
        case 14: return 210;
        case 16: return 66;
        case 18: return 98;
        case 23: return 136;
        case 29: return 56;
        case 8: return 34;
        default: return 0;
    }
}

static const TensorInfo* find_tensor(const std::vector<TensorInfo>& t,
                                     const char* name) {
    for (size_t i = 0; i < t.size(); ++i)
        if (strcmp(t[i].name, name) == 0) return &t[i];
    return nullptr;
}

static int block_geom(uint32_t type, int* bpe, int* bpb) {
    switch (type) {
        case 0: *bpe = 1; *bpb = 4; return 0;
        case 12: *bpe = 256; *bpb = 144; return 0;
        case 13: *bpe = 256; *bpb = 176; return 0;
        case 14: *bpe = 256; *bpb = 210; return 0;
        case 16: *bpe = 256; *bpb = 66; return 0;
        case 18: *bpe = 256; *bpb = 98; return 0;
        case 23: *bpe = 256; *bpb = 136; return 0;
        case 29: *bpe = 256; *bpb = 56; return 0;
        case 8: *bpe = 32; *bpb = 34; return 0;
        default: return -1;
    }
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

static void rope_ref(double* v, int rot, double pos) {
    for (int d = 0; d < rot / 2; ++d) {
        double ang = pos * pow(1e7, -2.0 * d / rot);
        double a = v[2 * d], b = v[2 * d + 1];
        v[2 * d] = a * cos(ang) - b * sin(ang);
        v[2 * d + 1] = a * sin(ang) + b * cos(ang);
    }
}

static int check(const char* tag, const float* gpu, const double* ref,
                 long n, double tol_scale) {
    double worst = 0;
    for (long i = 0; i < n; ++i) {
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
    fprintf(stderr, "parsed %zu tensors\n", tensors.size());

    FILE* f = fopen(argv[1], "rb");
    if (!f) return 1;

    std::vector<uint8_t> raw;
    const TensorInfo* ti;

    ti = find_tensor(tensors, "token_embd.weight");
    std::vector<uint8_t> embd_row;
    {
        int bpe, bpb;
        block_geom(12, &bpe, &bpb);
        embd_row.resize((size_t)6144 / bpe * bpb);
    }
    if (fseek(f, (long)(data_offset + ti->offset +
                        (uint64_t)802 * embd_row.size()), SEEK_SET) ||
        fread(embd_row.data(), 1, embd_row.size(), f) != embd_row.size()) {
        printf("EMBD READ FAIL\n"); return 1;
    }
    std::vector<float> h_embd(6144);
    hy4_dequant_row_q4_K((const block_q4_K*)embd_row.data(), h_embd.data(),
                         6144);

    ti = find_tensor(tensors, "blk.0.attn_norm.weight");
    read_tensor(f, ti, data_offset, raw);
    std::vector<float> h_anorm(6144);
    memcpy(h_anorm.data(), raw.data(), 6144 * 4);

    ti = find_tensor(tensors, "blk.0.attn_q_a.weight");
    read_tensor(f, ti, data_offset, raw);
    std::vector<float> h_qa((size_t)2048 * 6144);
    hy4_dequant_row_q5_K((const block_q5_K*)raw.data(), h_qa.data(),
                         (long)2048 * 6144);

    ti = find_tensor(tensors, "blk.0.attn_q_a_norm.weight");
    read_tensor(f, ti, data_offset, raw);
    std::vector<float> h_qan(2048);
    memcpy(h_qan.data(), raw.data(), 2048 * 4);

    ti = find_tensor(tensors, "blk.0.attn_q_b.weight");
    read_tensor(f, ti, data_offset, raw);
    std::vector<float> h_qb_all((size_t)1024 * 2048);
    dequant_q8_0(raw.data(), h_qb_all.data(), (long)1024 * 2048 / 32);
    std::vector<float> h_qb(h_qb_all.begin(),
                            h_qb_all.begin() + (size_t)256 * 2048);

    std::vector<double> ref_cur(6144), ref_q(2048), ref_qr(2048), ref_qh(256);
    {
        double ss = 0;
        for (int i = 0; i < 6144; ++i) ss += (double)h_embd[i] * h_embd[i];
        double inv = 1.0 / sqrt(ss / 6144.0 + 1e-5);
        for (int i = 0; i < 6144; ++i)
            ref_cur[i] = (double)h_embd[i] * inv * h_anorm[i];
        for (int r = 0; r < 2048; ++r) {
            double acc = 0;
            for (int k = 0; k < 6144; ++k)
                acc += (double)h_qa[(size_t)r * 6144 + k] * ref_cur[k];
            ref_q[r] = acc;
        }
        ss = 0;
        for (int i = 0; i < 2048; ++i) ss += ref_q[i] * ref_q[i];
        inv = 1.0 / sqrt(ss / 2048.0 + 1e-5);
        for (int i = 0; i < 2048; ++i)
            ref_qr[i] = ref_q[i] * inv * h_qan[i];
        for (int r = 0; r < 256; ++r) {
            double acc = 0;
            for (int k = 0; k < 2048; ++k)
                acc += (double)h_qb[(size_t)r * 2048 + k] * ref_qr[k];
            ref_qh[r] = acc;
        }
    }
    std::vector<double> ref_pe0(ref_qh.end() - 64, ref_qh.end());
    std::vector<double> ref_pe3 = ref_pe0;
    rope_ref(ref_pe3.data(), 64, 3.0);

    float *d_embd, *d_cur, *d_anorm, *d_q, *d_qan, *d_qr, *d_qb, *d_qh, *d_ss;
    float *d_w, *d_w2;
    CHECK_CUDA(cudaMalloc(&d_embd, 6144 * 4));
    CHECK_CUDA(cudaMalloc(&d_cur, 6144 * 4));
    CHECK_CUDA(cudaMalloc(&d_anorm, 6144 * 4));
    CHECK_CUDA(cudaMalloc(&d_q, 2048 * 4));
    CHECK_CUDA(cudaMalloc(&d_qan, 2048 * 4));
    CHECK_CUDA(cudaMalloc(&d_qr, 2048 * 4));
    CHECK_CUDA(cudaMalloc(&d_qh, 256 * 4));
    CHECK_CUDA(cudaMalloc(&d_ss, 4));
    CHECK_CUDA(cudaMalloc(&d_w, (size_t)2048 * 6144 * 4));
    CHECK_CUDA(cudaMalloc(&d_w2, (size_t)256 * 2048 * 4));
    cudaMemcpy(d_embd, h_embd.data(), 6144 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_anorm, h_anorm.data(), 6144 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qan, h_qan.data(), 2048 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_qa.data(), (size_t)2048 * 6144 * 4,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, h_qb.data(), (size_t)256 * 2048 * 4,
               cudaMemcpyHostToDevice);

    std::vector<float> gpu(6144);
    int bad = 0;

    k_rms_sq<<<1, 256>>>(d_embd, d_ss, 6144);
    k_rms_scale<<<6144 / 256, 256>>>(d_embd, d_anorm, d_cur, 6144, 1e-5f,
                                     d_ss);
    cudaMemcpy(gpu.data(), d_cur, 6144 * 4, cudaMemcpyDeviceToHost);
    bad += check("ATTN_NORM", gpu.data(), ref_cur.data(), 6144, 5e-4);

    k_gemv<<<(2048 + 127) / 128, 128>>>(d_w, d_cur, d_q, 2048, 6144);
    cudaMemcpy(gpu.data(), d_q, 2048 * 4, cudaMemcpyDeviceToHost);
    bad += check("Q_A_GEMV", gpu.data(), ref_q.data(), 2048, 1e-3);

    k_rms_sq<<<1, 256>>>(d_q, d_ss, 2048);
    k_rms_scale<<<2048 / 256, 256>>>(d_q, d_qan, d_qr, 2048, 1e-5f, d_ss);
    cudaMemcpy(gpu.data(), d_qr, 2048 * 4, cudaMemcpyDeviceToHost);
    bad += check("Q_A_NORM", gpu.data(), ref_qr.data(), 2048, 5e-4);

    k_gemv<<<(256 + 127) / 128, 128>>>(d_w2, d_qr, d_qh, 256, 2048);
    cudaMemcpy(gpu.data(), d_qh, 256 * 4, cudaMemcpyDeviceToHost);
    bad += check("Q_B_GEMV", gpu.data(), ref_qh.data(), 256, 1e-3);

    float* d_pe = d_qh + 192;
    k_rope<<<1, 32>>>(d_pe, 64, 0.0f);
    cudaMemcpy(gpu.data(), d_qh, 256 * 4, cudaMemcpyDeviceToHost);
    bad += check("ROPE_POS0", gpu.data() + 192, ref_pe0.data(), 64, 5e-4);
    const float golden[3] = {0.32866367f, -0.25685636f, -0.58516650f};
    for (int i = 0; i < 3; ++i)
        printf("GOLDEN q_pe[%d] gpu=%.8f fp64numpy=%.8f\n", i,
               gpu[192 + i], golden[i]);

    k_rope<<<1, 32>>>(d_pe, 64, 3.0f);
    cudaMemcpy(gpu.data(), d_qh, 256 * 4, cudaMemcpyDeviceToHost);
    bad += check("ROPE_POS3", gpu.data() + 192, ref_pe3.data(), 64, 5e-4);

    printf("QCHAIN %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
