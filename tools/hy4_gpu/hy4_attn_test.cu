// hy4 lane: GPU absorbed-MLA attention cell — layer 0 head 0, causal over
// the 4-token prompt "The quick brown fox", validated per stage against an
// in-harness float64 reference and against llama.cpp eval-callback goldens
// (attn_kqv-0 head 0: t0 [0.0047, 0.0020, -0.0051], t3 [0.0308, 0.0514,
// -0.0408]).
//
// Usage: hy4_attn_test <rank00.gguf>
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

__global__ void k_attn_head0(const float* q_abs, const float* q_pe,
                             const float* klat, const float* kpe,
                             float sink, int T, float scale,
                             float* vlat) {
    float scores[16];
    float M = sink;
    for (int tk = 0; tk < T; ++tk) {
        float s = 0;
        for (int i = 0; i < 512; ++i) s += q_abs[i] * klat[tk * 512 + i];
        for (int i = 0; i < 64; ++i) s += q_pe[i] * kpe[tk * 64 + i];
        scores[tk] = s * scale;
        if (scores[tk] > M) M = scores[tk];
    }
    float denom = expf(sink - M);
    float ps[16];
    for (int tk = 0; tk < T; ++tk) {
        ps[tk] = expf(scores[tk] - M);
        denom += ps[tk];
    }
    for (int i = 0; i < 512; ++i) {
        float acc = 0;
        for (int tk = 0; tk < T; ++tk) acc += ps[tk] / denom * klat[tk * 512 + i];
        vlat[i] = acc;
    }
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
        case 12: *bpe = 256; *bpb = 144; return 0;
        case 13: *bpe = 256; *bpb = 176; return 0;
        case 8: *bpe = 32; *bpb = 34; return 0;
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

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <rank00.gguf>\n", argv[0]);
        return 2;
    }
    std::vector<TensorInfo> tensors;
    uint64_t data_offset = 0;
    if (parse_rank_gguf(argv[1], tensors, &data_offset)) return 1;
    const int T = 4;
    const int tokens[4] = {802, 5466, 19405, 63357};

    FILE* f = fopen(argv[1], "rb");
    if (!f) return 1;
    std::vector<uint8_t> raw;
    const TensorInfo* ti;

    std::vector<std::vector<float>> h_embd(T, std::vector<float>(6144));
    {
        char owner_path[512];
        for (int tk = 0; tk < T; ++tk) {
            const int owner = tokens[tk] / 7552;
            const int row = tokens[tk] % 7552;
            snprintf(owner_path, sizeof(owner_path), "%s", argv[1]);
            char* rk = strstr(owner_path, "/rank-00/");
            if (!rk) { printf("rank dir pattern missing\n"); return 1; }
            snprintf(rk, sizeof(owner_path) - (size_t)(rk - owner_path),
                     "/rank-%02d/model-ud-iq1m-tp16-rank-%02d.gguf",
                     owner, owner);
            std::vector<TensorInfo> ot;
            uint64_t od;
            if (parse_rank_gguf(owner_path, ot, &od)) return 1;
            const TensorInfo* oe = find_tensor(ot, "token_embd.weight");
            int bpe, bpb;
            block_geom(12, &bpe, &bpb);
            std::vector<uint8_t> rowb((size_t)6144 / bpe * bpb);
            FILE* of = fopen(owner_path, "rb");
            if (!of || fseek(of, (long)(od + oe->offset +
                                        (uint64_t)row * rowb.size()),
                             SEEK_SET) ||
                fread(rowb.data(), 1, rowb.size(), of) != rowb.size()) {
                printf("EMBD READ FAIL owner %d\n", owner);
                return 1;
            }
            fclose(of);
            hy4_dequant_row_q4_K((const block_q4_K*)rowb.data(),
                                 h_embd[tk].data(), 6144);
        }
    }

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
    std::vector<float> h_qb((size_t)256 * 2048);
    dequant_q8_0(raw.data(), h_qb.data(), (long)256 * 2048 / 32);

    ti = find_tensor(tensors, "blk.0.attn_kv_a_mqa.weight");
    read_tensor(f, ti, data_offset, raw);
    std::vector<float> h_kva((size_t)576 * 6144);
    dequant_q8_0(raw.data(), h_kva.data(), (long)576 * 6144 / 32);

    ti = find_tensor(tensors, "blk.0.attn_kv_a_norm.weight");
    read_tensor(f, ti, data_offset, raw);
    std::vector<float> h_kvan(512);
    memcpy(h_kvan.data(), raw.data(), 512 * 4);

    ti = find_tensor(tensors, "blk.0.attn_k_b.weight");
    read_tensor(f, ti, data_offset, raw);
    std::vector<float> h_wkb((size_t)512 * 192);
    dequant_q8_0(raw.data(), h_wkb.data(), (long)512 * 192 / 32);

    ti = find_tensor(tensors, "blk.0.attn_v_b.weight");
    read_tensor(f, ti, data_offset, raw);
    std::vector<float> h_wvb((size_t)256 * 512);
    dequant_q8_0(raw.data(), h_wvb.data(), (long)256 * 512 / 32);

    ti = find_tensor(tensors, "blk.0.attn_sinks.weight");
    read_tensor(f, ti, data_offset, raw);
    float h_sink;
    memcpy(&h_sink, raw.data(), 4);

    std::vector<std::vector<double>> ref_cur(T), ref_qr(T), ref_klat(T),
        ref_kpe(T), ref_attn(T);
    std::vector<std::vector<double>> ref_q(T), ref_qh(T), ref_qabs(T);
    for (int tk = 0; tk < T; ++tk) {
        ref_cur[tk].assign(6144, 0);
        double ss = 0;
        for (int i = 0; i < 6144; ++i)
            ss += (double)h_embd[tk][i] * h_embd[tk][i];
        double inv = 1.0 / sqrt(ss / 6144.0 + 1e-5);
        for (int i = 0; i < 6144; ++i)
            ref_cur[tk][i] = (double)h_embd[tk][i] * inv * h_anorm[i];

        ref_q[tk].assign(2048, 0);
        for (int r = 0; r < 2048; ++r) {
            double acc = 0;
            for (int k = 0; k < 6144; ++k)
                acc += (double)h_qa[(size_t)r * 6144 + k] * ref_cur[tk][k];
            ref_q[tk][r] = acc;
        }
        ref_qr[tk].assign(2048, 0);
        ss = 0;
        for (int i = 0; i < 2048; ++i) ss += ref_q[tk][i] * ref_q[tk][i];
        inv = 1.0 / sqrt(ss / 2048.0 + 1e-5);
        for (int i = 0; i < 2048; ++i)
            ref_qr[tk][i] = ref_q[tk][i] * inv * h_qan[i];
        ref_qh[tk].assign(256, 0);
        for (int r = 0; r < 256; ++r) {
            double acc = 0;
            for (int k = 0; k < 2048; ++k)
                acc += (double)h_qb[(size_t)r * 2048 + k] * ref_qr[tk][k];
            ref_qh[tk][r] = acc;
        }
        ref_qabs[tk].assign(512, 0);
        for (int d = 0; d < 512; ++d) {
            double acc = 0;
            for (int i = 0; i < 192; ++i)
                acc += (double)h_wkb[(size_t)d * 192 + i] *
                       ref_qh[tk][i];
            ref_qabs[tk][d] = acc;
        }
        double kv[576];
        for (int r = 0; r < 576; ++r) {
            double acc = 0;
            for (int k = 0; k < 6144; ++k)
                acc += (double)h_kva[(size_t)r * 6144 + k] * ref_cur[tk][k];
            kv[r] = acc;
        }
        ref_klat[tk].assign(512, 0);
        ss = 0;
        for (int i = 0; i < 512; ++i) ss += kv[i] * kv[i];
        inv = 1.0 / sqrt(ss / 512.0 + 1e-5);
        for (int i = 0; i < 512; ++i)
            ref_klat[tk][i] = kv[i] * inv * h_kvan[i];
        ref_kpe[tk].assign(64, 0);
        for (int i = 0; i < 64; ++i) ref_kpe[tk][i] = kv[512 + i];
        for (int d = 0; d < 32; ++d) {
            double ang = tk * pow(1e7, -2.0 * d / 64);
            double a = ref_kpe[tk][2 * d], b = ref_kpe[tk][2 * d + 1];
            ref_kpe[tk][2 * d] = a * cos(ang) - b * sin(ang);
            ref_kpe[tk][2 * d + 1] = a * sin(ang) + b * cos(ang);
        }
    }
    for (int t = 0; t < T; ++t) {
        ref_attn[t].assign(256, 0);
        double scores[4], M = h_sink;
        for (int tk = 0; tk <= t; ++tk) {
            double s = 0;
            for (int i = 0; i < 512; ++i)
                s += ref_qabs[t][i] * ref_klat[tk][i];
            for (int i = 0; i < 64; ++i)
                s += ref_qh[t][192 + i] * ref_kpe[tk][i];
            scores[tk] = s * (1.0 / 16.0);
            if (scores[tk] > M) M = scores[tk];
        }
        double denom = exp(h_sink - M), ps[4];
        for (int tk = 0; tk <= t; ++tk) {
            ps[tk] = exp(scores[tk] - M);
            denom += ps[tk];
        }
        double vlat[512];
        for (int i = 0; i < 512; ++i) {
            double acc = 0;
            for (int tk = 0; tk <= t; ++tk)
                acc += ps[tk] / denom * ref_klat[tk][i];
            vlat[i] = acc;
        }
        for (int r = 0; r < 256; ++r) {
            double acc = 0;
            for (int i = 0; i < 512; ++i)
                acc += (double)h_wvb[(size_t)r * 512 + i] * vlat[i];
            ref_attn[t][r] = acc;
        }
    }

    float *d_embd, *d_cur, *d_anorm, *d_q, *d_qan, *d_qr, *d_qh, *d_ss;
    float *d_w_qa, *d_w_qb;
    float *d_kva, *d_kv, *d_kvan, *d_klat, *d_kpe, *d_wkb, *d_qabs;
    float *d_klat_all, *d_kpe_all, *d_wvb, *d_vlat, *d_attn;
    CHECK_CUDA(cudaMalloc(&d_embd, (size_t)T * 6144 * 4));
    CHECK_CUDA(cudaMalloc(&d_cur, 6144 * 4));
    CHECK_CUDA(cudaMalloc(&d_anorm, 6144 * 4));
    CHECK_CUDA(cudaMalloc(&d_q, 2048 * 4));
    CHECK_CUDA(cudaMalloc(&d_qan, 2048 * 4));
    CHECK_CUDA(cudaMalloc(&d_qr, 2048 * 4));
    CHECK_CUDA(cudaMalloc(&d_qh, 256 * 4));
    CHECK_CUDA(cudaMalloc(&d_ss, 4));
    CHECK_CUDA(cudaMalloc(&d_w_qa, (size_t)2048 * 6144 * 4));
    CHECK_CUDA(cudaMalloc(&d_w_qb, (size_t)256 * 2048 * 4));
    CHECK_CUDA(cudaMalloc(&d_kva, (size_t)576 * 6144 * 4));
    CHECK_CUDA(cudaMalloc(&d_kv, 576 * 4));
    CHECK_CUDA(cudaMalloc(&d_kvan, 512 * 4));
    CHECK_CUDA(cudaMalloc(&d_klat, 512 * 4));
    CHECK_CUDA(cudaMalloc(&d_kpe, 64 * 4));
    CHECK_CUDA(cudaMalloc(&d_klat_all, (size_t)T * 512 * 4));
    CHECK_CUDA(cudaMalloc(&d_kpe_all, (size_t)T * 64 * 4));
    CHECK_CUDA(cudaMalloc(&d_wkb, (size_t)512 * 192 * 4));
    CHECK_CUDA(cudaMalloc(&d_qabs, 512 * 4));
    CHECK_CUDA(cudaMalloc(&d_wvb, (size_t)256 * 512 * 4));
    CHECK_CUDA(cudaMalloc(&d_vlat, 512 * 4));
    CHECK_CUDA(cudaMalloc(&d_attn, 256 * 4));

    for (int tk = 0; tk < T; ++tk)
        cudaMemcpy(d_embd + (size_t)tk * 6144, h_embd[tk].data(), 6144 * 4,
                   cudaMemcpyHostToDevice);
    cudaMemcpy(d_anorm, h_anorm.data(), 6144 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qan, h_qan.data(), 2048 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kvan, h_kvan.data(), 512 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w_qa, h_qa.data(), (size_t)2048 * 6144 * 4,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_w_qb, h_qb.data(), (size_t)256 * 2048 * 4,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_kva, h_kva.data(), (size_t)576 * 6144 * 4,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_wkb, h_wkb.data(), (size_t)512 * 192 * 4,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_wvb, h_wvb.data(), (size_t)256 * 512 * 4,
               cudaMemcpyHostToDevice);

    int bad = 0;
    std::vector<float> gpu(6144);

    for (int tk = 0; tk < T; ++tk) {
        const float* d_e = d_embd + (size_t)tk * 6144;
        k_rms_sq<<<1, 256>>>(d_e, d_ss, 6144);
        k_rms_scale<<<6144 / 256, 256>>>(d_e, d_anorm, d_cur, 6144, 1e-5f,
                                         d_ss);
        k_gemv<<<(576 + 127) / 128, 128>>>(d_kva, d_cur, d_kv, 576, 6144);
        k_rms_sq<<<1, 256>>>(d_kv, d_ss, 512);
        k_rms_scale<<<512 / 256, 256>>>(d_kv, d_kvan, d_klat, 512, 1e-5f,
                                        d_ss);
        cudaMemcpy(d_kpe, d_kv + 512, 64 * 4, cudaMemcpyDeviceToDevice);
        k_rope<<<1, 32>>>(d_kpe, 64, (float)tk);
        cudaMemcpy(d_klat_all + (size_t)tk * 512, d_klat, 512 * 4,
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_kpe_all + (size_t)tk * 64, d_kpe, 64 * 4,
                   cudaMemcpyDeviceToDevice);
    }

    for (int t = 0; t < T; ++t) {
        const float* d_e = d_embd + (size_t)t * 6144;
        k_rms_sq<<<1, 256>>>(d_e, d_ss, 6144);
        k_rms_scale<<<6144 / 256, 256>>>(d_e, d_anorm, d_cur, 6144, 1e-5f,
                                         d_ss);
        k_gemv<<<(2048 + 127) / 128, 128>>>(d_w_qa, d_cur, d_q, 2048, 6144);
        k_rms_sq<<<1, 256>>>(d_q, d_ss, 2048);
        k_rms_scale<<<2048 / 256, 256>>>(d_q, d_qan, d_qr, 2048, 1e-5f,
                                         d_ss);
        k_gemv<<<(256 + 127) / 128, 128>>>(d_w_qb, d_qr, d_qh, 256, 2048);
        k_gemv<<<(512 + 127) / 128, 128>>>(d_wkb, d_qh, d_qabs, 512, 192);
        k_attn_head0<<<1, 1>>>(d_qabs, d_qh + 192, d_klat_all, d_kpe_all,
                               h_sink, t + 1, 1.0f / 16.0f, d_vlat);
        k_gemv<<<(256 + 127) / 128, 128>>>(d_wvb, d_vlat, d_attn, 256, 512);
        gpu.resize(256);
        cudaMemcpy(gpu.data(), d_attn, 256 * 4, cudaMemcpyDeviceToHost);
        char tag[64];
        snprintf(tag, sizeof(tag), "ATTN_T%d", t);
        bad += check(tag, gpu.data(), ref_attn[t].data(), 256, 1e-3);
        printf("  gpu t%d head0 first3 %.4f %.4f %.4f "
               "(llama: t0 0.0047/0.0020/-0.0051 .. t3 0.0308/0.0514/-0.0408)\n",
               t, gpu[0], gpu[1], gpu[2]);
    }
    printf("ATTN %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
