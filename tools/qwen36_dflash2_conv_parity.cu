// qwen36_dflash2_conv_parity.cu - parity harness: the DFlash2 grouped conv kernel
// vs the numpy _grouped_conv oracle. Runs on synthetic inputs, dumps out.bin.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define B 8
#define H 5120
#define TAPS 2
#define NUM_GROUPS 320
#define GROUP_SIZE 16
#define BLOCK_SIZE 8

static __global__ void ConvKernel(const __nv_bfloat16 *x_bf16, const float *delta,
    const __nv_bfloat16 *base_bf16, __nv_bfloat16 *out_bf16,
    uint32_t block_size, uint32_t num_groups, uint32_t group_size)
{
    const uint32_t pos = blockIdx.x, group = blockIdx.y;
    const uint32_t c = group * group_size + threadIdx.x;
    if (c >= num_groups * group_size) return;
    const uint32_t p = (block_size & (block_size - 1u)) == 0u ? pos & (block_size - 1u) : pos % block_size;
    const float x0 = __bfloat162float(x_bf16[(uint64_t)pos * H + c]);
    const float d0 = delta[((uint64_t)pos * 2u + 0u) * num_groups + group];
    float out = (__bfloat162float(base_bf16[0u * H + c]) + d0) * x0;
    if (p >= 1u) {
        const float x1 = __bfloat162float(x_bf16[((uint64_t)(pos - 1u)) * H + c]);
        const float d1 = delta[((uint64_t)pos * 2u + 1u) * num_groups + group];
        out += (__bfloat162float(base_bf16[1u * H + c]) + d1) * x1;
    }
    out_bf16[(uint64_t)pos * H + c] = __float2bfloat16(out);
}

static __global__ void bf16_to_f32_kernel(const __nv_bfloat16 *in, float *out, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __bfloat162float(in[i]);
}

int main(void) {
    // deterministic synthetic inputs via LCG
    uint32_t s = 12345u;
    auto next = [&](){ s = s * 1664525u + 1013904223u; return (float)(s & 0xFFFF) / 65536.0f - 0.5f; };

    __nv_bfloat16 *h_x = (__nv_bfloat16*)malloc(B * H * 2);
    float *h_delta = (float*)malloc(B * TAPS * NUM_GROUPS * 4);
    __nv_bfloat16 *h_base = (__nv_bfloat16*)malloc(TAPS * H * 2);
    for (uint32_t i = 0; i < B * H; i++) h_x[i] = __float2bfloat16(next());
    for (uint32_t i = 0; i < B * TAPS * NUM_GROUPS; i++) h_delta[i] = next();
    for (uint32_t i = 0; i < TAPS * H; i++) h_base[i] = __float2bfloat16(next());

    __nv_bfloat16 *d_x, *d_base, *d_out; float *d_delta;
    cudaMalloc(&d_x, B * H * 2); cudaMalloc(&d_delta, B * TAPS * NUM_GROUPS * 4);
    cudaMalloc(&d_base, TAPS * H * 2); cudaMalloc(&d_out, B * H * 2);
    cudaMemcpy(d_x, h_x, B * H * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta, h_delta, B * TAPS * NUM_GROUPS * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_base, h_base, TAPS * H * 2, cudaMemcpyHostToDevice);

    dim3 grid(B, NUM_GROUPS);
    ConvKernel<<<grid, GROUP_SIZE>>>(d_x, d_delta, d_base, d_out, BLOCK_SIZE, NUM_GROUPS, GROUP_SIZE);
    cudaDeviceSynchronize();

    __nv_bfloat16 *h_out = (__nv_bfloat16*)malloc(B * H * 2);
    cudaMemcpy(h_out, d_out, B * H * 2, cudaMemcpyDeviceToHost);

    // dump x (bf16->f32), delta, base, out (bf16->f32) for the numpy compare
    FILE *f = fopen("/tmp/dflash2_conv_parity.bin", "wb");
    float *buf = (float*)malloc(B * H * 4);
    for (uint32_t i = 0; i < B * H; i++) buf[i] = __bfloat162float(h_x[i]);
    fwrite(buf, 4, B * H, f);
    fwrite(h_delta, 4, B * TAPS * NUM_GROUPS, f);
    for (uint32_t i = 0; i < TAPS * H; i++) buf[i] = __bfloat162float(h_base[i]);
    fwrite(buf, 4, TAPS * H, f);
    for (uint32_t i = 0; i < B * H; i++) buf[i] = __bfloat162float(h_out[i]);
    fwrite(buf, 4, B * H, f);
    fclose(f);
    printf("CONV_PARITY dumped x/delta/base/out to /tmp/dflash2_conv_parity.bin\n");
    return 0;
}
