#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_status.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>

#define DOORBELL_HIDDEN 4096u
#define DOORBELL_PORT_BASE 57340u
#define DOORBELL_IDENTIFIER 0x444f4f52424f4f1ull

static __global__ void doorbell_combine_bf16_kernel(
    uint16_t *destination, const uint16_t *source, uint32_t elements)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements)
        return;
    float a = __uint_as_float((uint32_t)destination[index] << 16);
    float b = __uint_as_float((uint32_t)source[index] << 16);
    float sum = a + b;
    __float2bfloat16... ;
}

int main(int argc, char **argv)
{
    return 0;
}
