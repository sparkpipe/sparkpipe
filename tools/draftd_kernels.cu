#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

template <typename Acc>
__device__ __forceinline__ Acc WarpSum(Acc v)
{
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float Bf16Bits(uint32_t w, int hi)
{
    return __uint_as_float(hi ? (w & 0xffff0000u) : (w << 16));
}

template <typename Acc>
__global__ void GemvBf16Kernel(const uint4 *w, const float *x, float *out, int rows, int vec_cols)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= rows) return;
    const uint4 *row = w + (size_t)warp * vec_cols;
    Acc acc = 0;
    for (int v = lane; v < vec_cols; v += 32)
    {
        uint4 q = __ldg(row + v);
        const float *xv = x + v * 8;
        uint32_t words[4] = {q.x, q.y, q.z, q.w};
        for (int k = 0; k < 4; ++k)
        {
            acc += (Acc)Bf16Bits(words[k], 0) * (Acc)xv[2 * k];
            acc += (Acc)Bf16Bits(words[k], 1) * (Acc)xv[2 * k + 1];
        }
    }
    acc = WarpSum(acc);
    if (lane == 0) out[warp] = (float)acc;
}

__device__ __forceinline__ float E4m3(uint32_t b)
{
    uint32_t s = b & 0x80u, e = (b >> 3) & 0xfu, m = b & 0x7u;
    float v = e == 0 ? (float)m * 0.001953125f : __uint_as_float(((e + 120u) << 23) | (m << 20));
    if (e == 15u && m == 7u) v = __int_as_float(0x7fc00000);
    return s ? -v : v;
}

template <typename Acc>
__global__ void GemvFp8BlockKernel(const uint8_t *codes, const float *scale, const int64_t *ids,
                                   const float *x, float *out, int rows, int cols, int x_stride, int count)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= rows * count) return;
    int item = warp / rows, r = warp % rows;
    int64_t e = ids[item];
    int blocks = cols / 128;
    const uint4 *row = (const uint4 *)(codes + ((size_t)e * rows + r) * cols);
    const float *srow = scale + ((size_t)e * (rows / 128) + r / 128) * blocks;
    const float *xv = x + (size_t)item * x_stride;
    Acc acc = 0;
    for (int v = lane; v < cols / 16; v += 32)
    {
        uint4 q = __ldg(row + v);
        float s = srow[(v * 16) / 128];
        const float *xp = xv + v * 16;
        uint32_t words[4] = {q.x, q.y, q.z, q.w};
        for (int k = 0; k < 4; ++k)
            for (int b = 0; b < 4; ++b)
                acc += (Acc)(E4m3((words[k] >> (8 * b)) & 0xffu) * s) * (Acc)xp[k * 4 + b];
    }
    acc = WarpSum(acc);
    if (lane == 0) out[(size_t)item * rows + r] = (float)acc;
}

extern "C" int spark_draftd_gemv_bf16(const void *w, const float *x, float *out, int rows, int cols, int wide,
                                      cudaStream_t stream)
{
    if (cols % 8 != 0 || rows <= 0) return -1;
    int threads = 256, warps = threads / 32;
    if (wide)
        GemvBf16Kernel<double><<<(rows + warps - 1) / warps, threads, 0, stream>>>((const uint4 *)w, x, out, rows, cols / 8);
    else
        GemvBf16Kernel<float><<<(rows + warps - 1) / warps, threads, 0, stream>>>((const uint4 *)w, x, out, rows, cols / 8);
    return (int)cudaGetLastError();
}

extern "C" int spark_draftd_gemv_fp8_block(const uint8_t *codes, const float *scale, const int64_t *ids, const float *x,
                                           float *out, int rows, int cols, int x_stride, int count, int wide,
                                           cudaStream_t stream)
{
    if (rows % 128 != 0 || cols % 128 != 0 || count <= 0) return -1;
    int threads = 256, warps = threads / 32, grid = (rows * count + warps - 1) / warps;
    if (wide)
        GemvFp8BlockKernel<double><<<grid, threads, 0, stream>>>(codes, scale, ids, x, out, rows, cols, x_stride, count);
    else
        GemvFp8BlockKernel<float><<<grid, threads, 0, stream>>>(codes, scale, ids, x, out, rows, cols, x_stride, count);
    return (int)cudaGetLastError();
}
