// qwen36_dflash2_conv_parity.cu - parity harness: the DFlash2 grouped conv
// kernel (BF16 delta + side param, faithful copy of SparkQwen36DsparkConvKernel)
// vs the numpy _grouped_conv oracle. Runs BOTH sides on synthetic inputs and
// dumps everything for the python compare (tools/qwen36_dflash2_conv_parity_check.py).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define B 8
#define H 5120
#define TAPS 2
#define SIDES 2
#define NUM_GROUPS 320
#define GROUP_SIZE 16
#define BLOCK_SIZE 8

/* Faithful copy of SparkQwen36DsparkConvKernel: delta is [B, 2 sides, 2 taps,
 * groups] BF16 (the kernel_projection output), base_bf16 is the side's [taps,H]
 * slice, side selects the stride (pos*2+side)*2+tap. */
static __global__ void ConvKernel(const __nv_bfloat16 *x_bf16, const __nv_bfloat16 *delta,
    const __nv_bfloat16 *base_bf16, __nv_bfloat16 *out_bf16,
    uint32_t block_size, uint32_t num_groups, uint32_t group_size, uint32_t side)
{
    const uint32_t pos = blockIdx.x, group = blockIdx.y;
    const uint32_t c = group * group_size + threadIdx.x;
    const uint32_t Htot = num_groups * group_size;
    if (c >= Htot) return;
    const uint32_t p = (block_size & (block_size - 1u)) == 0u ? pos & (block_size - 1u) : pos % block_size;
    const uint32_t ds = (pos * 2u + side) * 2u;
    const float x0 = __bfloat162float(x_bf16[(uint64_t)pos * Htot + c]);
    const float d0 = __bfloat162float(delta[(uint64_t)(ds + 0u) * num_groups + group]);
    float out = (__bfloat162float(base_bf16[0u * Htot + c]) + d0) * x0;
    if (p >= 1u) {
        const float x1 = __bfloat162float(x_bf16[((uint64_t)(pos - 1u)) * Htot + c]);
        const float d1 = __bfloat162float(delta[(uint64_t)(ds + 1u) * num_groups + group]);
        out += (__bfloat162float(base_bf16[1u * Htot + c]) + d1) * x1;
    }
    out_bf16[(uint64_t)pos * Htot + c] = __float2bfloat16(out);
}

static void bf16_buf_to_f32(const __nv_bfloat16 *in, float *out, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) out[i] = __bfloat162float(in[i]);
}

int main(void) {
    uint32_t s = 12345u;
    auto next = [&](){ s = s * 1664525u + 1013904223u; return (float)(s & 0xFFFF) / 65536.0f - 0.5f; };

    // x [B,H], delta [B,2,2,groups], base [2,2,H]
    __nv_bfloat16 *h_x = (__nv_bfloat16*)malloc(B * H * 2);
    __nv_bfloat16 *h_delta = (__nv_bfloat16*)malloc(B * SIDES * TAPS * NUM_GROUPS * 2);
    __nv_bfloat16 *h_base = (__nv_bfloat16*)malloc(SIDES * TAPS * H * 2);
    for (uint32_t i = 0; i < B * H; i++) h_x[i] = __float2bfloat16(next());
    for (uint32_t i = 0; i < B * SIDES * TAPS * NUM_GROUPS; i++) h_delta[i] = __float2bfloat16(next());
    for (uint32_t i = 0; i < SIDES * TAPS * H; i++) h_base[i] = __float2bfloat16(next());

    __nv_bfloat16 *d_x, *d_delta, *d_base, *d_out;
    cudaMalloc(&d_x, B * H * 2); cudaMalloc(&d_delta, B * SIDES * TAPS * NUM_GROUPS * 2);
    cudaMalloc(&d_base, SIDES * TAPS * H * 2); cudaMalloc(&d_out, B * H * 2);
    cudaMemcpy(d_x, h_x, B * H * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta, h_delta, B * SIDES * TAPS * NUM_GROUPS * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_base, h_base, SIDES * TAPS * H * 2, cudaMemcpyHostToDevice);

    __nv_bfloat16 *h_out0 = (__nv_bfloat16*)malloc(B * H * 2);
    __nv_bfloat16 *h_out1 = (__nv_bfloat16*)malloc(B * H * 2);
    dim3 grid(B, NUM_GROUPS);
    // side 0: base slice [0]
    ConvKernel<<<grid, GROUP_SIZE>>>(d_x, d_delta, d_base, d_out, BLOCK_SIZE, NUM_GROUPS, GROUP_SIZE, 0u);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out0, d_out, B * H * 2, cudaMemcpyDeviceToHost);
    // side 1: base slice [1] = base + TAPS*H
    ConvKernel<<<grid, GROUP_SIZE>>>(d_x, d_delta, d_base + TAPS * H, d_out, BLOCK_SIZE, NUM_GROUPS, GROUP_SIZE, 1u);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out1, d_out, B * H * 2, cudaMemcpyDeviceToHost);

    // dump: x, delta, base, out0, out1 (all f32)
    FILE *f = fopen("/tmp/dflash2_conv_parity.bin", "wb");
    float *buf = (float*)malloc(B * H * 4); /* max(B*H, SIDES*TAPS*H) = B*H */
    bf16_buf_to_f32(h_x, buf, B * H); fwrite(buf, 4, B * H, f);
    bf16_buf_to_f32(h_delta, buf, B * SIDES * TAPS * NUM_GROUPS); fwrite(buf, 4, B * SIDES * TAPS * NUM_GROUPS, f);
    bf16_buf_to_f32(h_base, buf, SIDES * TAPS * H); fwrite(buf, 4, SIDES * TAPS * H, f);
    bf16_buf_to_f32(h_out0, buf, B * H); fwrite(buf, 4, B * H, f);
    bf16_buf_to_f32(h_out1, buf, B * H); fwrite(buf, 4, B * H, f);
    fclose(f);
    printf("CONV_PARITY dumped x/delta/base/out0/out1 to /tmp/dflash2_conv_parity.bin (BF16 delta, both sides)\n");
    return 0;
}
