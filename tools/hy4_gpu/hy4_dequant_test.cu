// hy4 lane: GPU dequant bitwise verification for the remaining weight
// classes (Q4_K/Q5_K/Q6_K/IQ4_XS/IQ1_M/IQ2_XXS/IQ3_XXS) against the CPU
// vendor header on real blocks from a node-local rank pack.
//
// Usage: hy4_dequant_test <rank.gguf>
// Prints one PASS/FAIL line per class found in the pack and a final
// DEQUANT_ALL verdict. Exactness policy: GPU floats must be BITWISE equal
// to the CPU vendor dequant (no reductions -> deterministic; -fmad=false).
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define _Static_assert(cond, msg) static_assert(cond, msg)
#include "hy4_iq_dequant_vendor.h"

#define QK_K 256

__device__ uint8_t d_kmask_iq2xs[8];
__device__ uint8_t d_ksigns_iq2xs[128];
__device__ uint64_t d_iq2xxs_grid[256];
__device__ uint32_t d_iq3xxs_grid[256];
__device__ uint64_t d_iq1s_grid[2048];
__device__ int8_t d_kvalues_iq4nl[16];
__device__ float d_fp16_table[65536];

__device__ __forceinline__ float d_fp16(uint16_t h) {
    return d_fp16_table[h];
}

__device__ __forceinline__ uint32_t rd32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

__device__ __forceinline__ uint16_t rd16(const uint8_t* p) {
    return (uint16_t)(p[0] | (p[1] << 8));
}

__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* q,
                                                 uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

__global__ void k_q4k(const uint8_t* x, float* y, long nb) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= nb) return;
    const uint8_t* blk = x + i * 144;
    const float d = d_fp16(rd16(blk));
    const float mn = d_fp16(rd16(blk + 2));
    const uint8_t* scales = blk + 4;
    const uint8_t* q = blk + 16;
    float* yy = y + i * QK_K;
    int is = 0;
    for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, scales, &sc, &m);
        const float d1 = d * sc; const float m1 = mn * m;
        get_scale_min_k4(is + 1, scales, &sc, &m);
        const float d2 = d * sc; const float m2 = mn * m;
        for (int l = 0; l < 32; ++l) yy[j + l] = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l) yy[j + 32 + l] = d2 * (q[l] >> 4) - m2;
        q += 32; is += 2;
    }
}

__global__ void k_q5k(const uint8_t* x, float* y, long nb) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= nb) return;
    const uint8_t* blk = x + i * 176;
    const float d = d_fp16(rd16(blk));
    const float mn = d_fp16(rd16(blk + 2));
    const uint8_t* scales = blk + 4;
    const uint8_t* qh = blk + 16;
    const uint8_t* ql = blk + 48;
    float* yy = y + i * QK_K;
    int is = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, scales, &sc, &m);
        const float d1 = d * sc; const float m1 = mn * m;
        get_scale_min_k4(is + 1, scales, &sc, &m);
        const float d2 = d * sc; const float m2 = mn * m;
        for (int l = 0; l < 32; ++l)
            yy[j + l] = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
        for (int l = 0; l < 32; ++l)
            yy[j + 32 + l] = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
        ql += 32; is += 2;
        u1 <<= 2; u2 <<= 2;
    }
}

__global__ void k_q6k(const uint8_t* x, float* y, long nb) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= nb) return;
    const uint8_t* blk = x + i * 210;
    const float d = d_fp16(rd16(blk + 208));
    const uint8_t* ql = blk;
    const uint8_t* qh = blk + 128;
    const int8_t* sc = (const int8_t*)(blk + 192);
    float* yy = y + i * QK_K;
    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; ++l) {
            int is = l / 16;
            const int8_t q1 = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            yy[n + l] = d * sc[is + 0] * q1;
            yy[n + l + 32] = d * sc[is + 2] * q2;
            yy[n + l + 64] = d * sc[is + 4] * q3;
            yy[n + l + 96] = d * sc[is + 6] * q4;
        }
        ql += 64; qh += 32; sc += 8;
    }
}

__global__ void k_iq4xs(const uint8_t* x, float* y, long nb) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= nb) return;
    const uint8_t* blk = x + i * 136;
    const float d = d_fp16(rd16(blk));
    const uint16_t scales_h = rd16(blk + 2);
    const uint8_t* scales_l = blk + 4;
    const uint8_t* qs = blk + 8;
    float* yy = y + i * QK_K;
    for (int ib = 0; ib < QK_K / 32; ++ib) {
        const int ls = ((scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) |
                       (((scales_h >> 2 * ib) & 3) << 4);
        const float dl = d * (ls - 32);
        for (int j = 0; j < 16; ++j) {
            yy[j] = dl * d_kvalues_iq4nl[qs[j] & 0xf];
            yy[j + 16] = dl * d_kvalues_iq4nl[qs[j] >> 4];
        }
        yy += 32; qs += 16;
    }
}

__global__ void k_iq2xxs(const uint8_t* x, float* y, long nb) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= nb) return;
    const uint8_t* blk = x + i * 66;
    const float d = d_fp16(rd16(blk));
    const uint8_t* qs = blk + 2;
    float* yy = y + i * QK_K;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const uint8_t* p = qs + 4 * ib32;
        const uint32_t a0 = rd32(p);
        const uint32_t a1 = rd32(p + 4);
        const uint8_t* aux8 = p;
        const float db = d * (0.5f + (a1 >> 28)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            const uint8_t* grid = (const uint8_t*)(d_iq2xxs_grid + aux8[l]);
            const uint8_t signs = d_ksigns_iq2xs[(a1 >> 7 * l) & 127];
            for (int j = 0; j < 8; ++j)
                yy[j] = db * grid[j] * (signs & d_kmask_iq2xs[j] ? -1.f : 1.f);
            yy += 8;
        }
    }
}

__global__ void k_iq3xxs(const uint8_t* x, float* y, long nb) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= nb) return;
    const uint8_t* blk = x + i * 98;
    const float d = d_fp16(rd16(blk));
    const uint8_t* qs = blk + 2;
    const uint8_t* ss = qs + 64;
    float* yy = y + i * QK_K;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const uint32_t aux32 = rd32(ss + 4 * ib32);
        const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
        for (int l = 0; l < 4; ++l) {
            const uint8_t signs = d_ksigns_iq2xs[(aux32 >> 7 * l) & 127];
            const uint8_t* grid1 = (const uint8_t*)(d_iq3xxs_grid + qs[2 * l]);
            const uint8_t* grid2 = (const uint8_t*)(d_iq3xxs_grid + qs[2 * l + 1]);
            for (int j = 0; j < 4; ++j) {
                yy[j] = db * grid1[j] * (signs & d_kmask_iq2xs[j] ? -1.f : 1.f);
                yy[j + 4] = db * grid2[j] * (signs & d_kmask_iq2xs[j + 4] ? -1.f : 1.f);
            }
            yy += 8;
        }
        qs += 8;
    }
}

__global__ void k_iq1m(const uint8_t* x, float* y, long nb) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= nb) return;
    const uint8_t* blk = x + i * 56;
    const uint8_t* qs = blk;
    const uint8_t* qh = blk + 32;
    const uint8_t* scb = blk + 48;
    const uint16_t sc[4] = {rd16(scb), rd16(scb + 2), rd16(scb + 4),
                            rd16(scb + 6)};
    const uint16_t su = (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                                   ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
    const float d = d_fp16(su);
    float* yy = y + i * QK_K;
    for (int ib = 0; ib < 8; ++ib) {
        const float dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
        const float dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
        uint16_t idx[4];
        idx[0] = (uint16_t)(qs[0] | ((qh[0] << 8) & 0x700));
        idx[1] = (uint16_t)(qs[1] | ((qh[0] << 4) & 0x700));
        idx[2] = (uint16_t)(qs[2] | ((qh[1] << 8) & 0x700));
        idx[3] = (uint16_t)(qs[3] | ((qh[1] << 4) & 0x700));
        const float delta[4] = {
            qh[0] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA,
            qh[0] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA,
            qh[1] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA,
            qh[1] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA};
        for (int l = 0; l < 2; ++l) {
            const int8_t* grid = (const int8_t*)(d_iq1s_grid + idx[l]);
            for (int j = 0; j < 8; ++j) yy[j] = dl1 * (grid[j] + delta[l]);
            yy += 8;
        }
        for (int l = 2; l < 4; ++l) {
            const int8_t* grid = (const int8_t*)(d_iq1s_grid + idx[l]);
            for (int j = 0; j < 8; ++j) yy[j] = dl2 * (grid[j] + delta[l]);
            yy += 8;
        }
        qs += 4; qh += 2;
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

static int bpb_of(uint32_t type) {
    switch (type) {
        case 12: return 144;
        case 13: return 176;
        case 14: return 210;
        case 16: return 66;
        case 18: return 98;
        case 23: return 136;
        case 29: return 56;
        default: return 0;
    }
}

static void cpu_dequant(uint32_t type, const uint8_t* src, float* dst,
                        long nblocks) {
    switch (type) {
        case 12: hy4_dequant_row_q4_K((const block_q4_K*)src, dst, nblocks * 256); break;
        case 13: hy4_dequant_row_q5_K((const block_q5_K*)src, dst, nblocks * 256); break;
        case 14: hy4_dequant_row_q6_K((const block_q6_K*)src, dst, nblocks * 256); break;
        case 23: hy4_dequant_row_iq4_xs((const block_iq4_xs*)src, dst, nblocks * 256); break;
        case 16: hy4_dequant_iq2_xxs(src, dst, nblocks); break;
        case 18: hy4_dequant_iq3_xxs(src, dst, nblocks); break;
        case 29: hy4_dequant_iq1_m(src, dst, nblocks); break;
    }
}

typedef void (*kern_fn)(const uint8_t*, float*, long);

static kern_fn kernel_of(uint32_t type) {
    switch (type) {
        case 12: return k_q4k;
        case 13: return k_q5k;
        case 14: return k_q6k;
        case 16: return k_iq2xxs;
        case 18: return k_iq3xxs;
        case 23: return k_iq4xs;
        case 29: return k_iq1m;
    }
    return nullptr;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <rank.gguf>\n", argv[0]);
        return 2;
    }
    std::vector<TensorInfo> tensors;
    uint64_t data_offset = 0;
    if (parse_rank_gguf(argv[1], tensors, &data_offset)) return 1;
    fprintf(stderr, "parsed %zu tensors, data_offset %llu\n",
            tensors.size(), (unsigned long long)data_offset);

    {
        float fp16_table_host[65536];
        for (int h = 0; h < 65536; ++h)
            fp16_table_host[h] = hy4_fp16_to_fp32((uint16_t)h);
        if (cudaMemcpyToSymbol(d_fp16_table, fp16_table_host,
                               sizeof(fp16_table_host)) != cudaSuccess) {
            fprintf(stderr, "fp16 table upload failed\n"); return 1;
        }
    }
    if (cudaMemcpyToSymbol(d_kmask_iq2xs, kmask_iq2xs, sizeof(kmask_iq2xs)) ||
        cudaMemcpyToSymbol(d_ksigns_iq2xs, ksigns_iq2xs, sizeof(ksigns_iq2xs)) ||
        cudaMemcpyToSymbol(d_iq2xxs_grid, iq2xxs_grid, sizeof(iq2xxs_grid)) ||
        cudaMemcpyToSymbol(d_iq3xxs_grid, iq3xxs_grid, sizeof(iq3xxs_grid)) ||
        cudaMemcpyToSymbol(d_iq1s_grid, iq1s_grid, sizeof(iq1s_grid)) ||
        cudaMemcpyToSymbol(d_kvalues_iq4nl, kvalues_iq4nl,
                           sizeof(kvalues_iq4nl))) {
        fprintf(stderr, "table upload failed\n"); return 1;
    }

    FILE* f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "reopen failed\n"); return 1; }

    const uint32_t want_types[7] = {12, 13, 14, 23, 29, 16, 18};
    const long max_blocks = 512;
    int all_pass = 1;
    int classes_seen = 0;

    for (int w = 0; w < 7; ++w) {
        const uint32_t type = want_types[w];
        const int bpb = bpb_of(type);
        const TensorInfo* ti = nullptr;
        for (size_t i = 0; i < tensors.size(); ++i) {
            if (tensors[i].type == type) { ti = &tensors[i]; break; }
        }
        if (!ti) {
            printf("TYPE %u ABSENT (not in this rank pack)\n", type);
            continue;
        }
        classes_seen++;
        long total_blocks = (long)(ti->nelem / QK_K);
        long nb = total_blocks < max_blocks ? total_blocks : max_blocks;
        std::vector<uint8_t> raw(nb * bpb);
        if (fseek(f, (long)(data_offset + ti->offset), SEEK_SET) ||
            fread(raw.data(), 1, raw.size(), f) != raw.size()) {
            printf("TYPE %u READ FAIL\n", type); all_pass = 0; continue;
        }
        std::vector<float> cpu_out((size_t)nb * QK_K);
        cpu_dequant(type, raw.data(), cpu_out.data(), nb);

        uint8_t* dev_in = nullptr;
        float* dev_out = nullptr;
        if (cudaMalloc(&dev_in, raw.size()) != cudaSuccess ||
            cudaMalloc(&dev_out, cpu_out.size() * 4) != cudaSuccess) {
            printf("TYPE %u CUDA ALLOC FAIL\n", type); all_pass = 0; continue;
        }
        cudaMemcpy(dev_in, raw.data(), raw.size(), cudaMemcpyHostToDevice);
        long grid = (nb + 255) / 256;
        kernel_of(type)<<<grid, 256>>>(dev_in, dev_out, nb);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("TYPE %u CUDA RUN FAIL %s\n", type,
                   cudaGetErrorString(err));
            all_pass = 0; cudaFree(dev_in); cudaFree(dev_out); continue;
        }
        std::vector<float> gpu_out(cpu_out.size());
        cudaMemcpy(gpu_out.data(), dev_out, cpu_out.size() * 4,
                   cudaMemcpyDeviceToHost);
        cudaFree(dev_in); cudaFree(dev_out);

        long first_diff = -1;
        for (long i = 0; i < (long)cpu_out.size(); ++i) {
            if (memcmp(&cpu_out[i], &gpu_out[i], 4) != 0) {
                first_diff = i;
                break;
            }
        }
        if (first_diff < 0) {
            printf("TYPE %u blocks %ld BITWISE PASS\n", type, nb);
        } else {
            printf("TYPE %u blocks %ld BITWISE FAIL first_diff elem %ld "
                   "cpu=%.6g gpu=%.6g\n", type, nb, first_diff,
                   cpu_out[first_diff], gpu_out[first_diff]);
            all_pass = 0;
        }
    }
    fclose(f);
    printf("DEQUANT_ALL %s (%d classes)\n", all_pass ? "PASS" : "FAIL",
           classes_seen);
    return all_pass ? 0 : 1;
}
