#include "runtime/tensor_map.h"

#include <stdio.h>
#include <string.h>

uint32_t stub_rank(void);
uint64_t stub_dim(int index);
uint32_t stub_box(int index);
uint64_t stub_stride(int index);
int stub_swizzle(void);

static int failures = 0;

static void check(int condition, const char *label)
{
    printf(condition ? "  ok   %s\n" : "  FAIL %s\n", label);
    if (!condition)
    {
        failures++;
    }
}

int main(void)
{
    static uint8_t buffer[512] __attribute__((aligned(128)));
    CUtensorMap map __attribute__((aligned(64)));
    LmTensorMapRequest request;
    int32_t status;

    printf("cuTensorMapEncodeTiled argument marshalling\n");

    printf("\nFP4 weights\n");
    memset(&request, 0, sizeof(request));
    request.global_address = buffer;
    request.rows = 4096u;
    request.columns = 6144u;
    request.groups = 256u;
    request.box_rows = 128u;
    request.box_columns = 128u;
    request.element_bits = LM_TM_BITS_NVFP4;
    status = LmTensorMapPrepare(&map, &request);
    check(status == LM_TM_ENCODE_OK, "prepare succeeds");
    check(stub_rank() == 3u, "rank 3 reaches driver");
    check(stub_dim(0) == 3072u, "innermost dimension is K/2 bytes");
    check(stub_box(0) == 64u, "box inner extent is 64 bytes");
    check(stub_stride(0) == 3072u, "row stride is K/2 bytes");
    check(stub_stride(1) == 3072ULL * 4096ULL,
        "expert stride spans one matrix");
    check(stub_swizzle() == CU_TENSOR_MAP_SWIZZLE_64B,
        "64-byte swizzle selected");

    printf("\nBF16 activations\n");
    memset(&request, 0, sizeof(request));
    request.global_address = buffer;
    request.rows = 64u;
    request.columns = 7168u;
    request.groups = 1u;
    request.box_rows = 16u;
    request.box_columns = 64u;
    request.element_bits = LM_TM_BITS_BF16;
    status = LmTensorMapPrepare(&map, &request);
    check(status == LM_TM_ENCODE_OK, "BF16 prepare succeeds");
    check(stub_rank() == 2u, "BF16 single group uses rank 2");
    check(stub_dim(0) == 14336u, "BF16 inner dimension is bytes");
    check(stub_box(0) == 128u, "BF16 tile is one 128-byte box");
    check(stub_swizzle() == CU_TENSOR_MAP_SWIZZLE_128B,
        "BF16 uses 128-byte swizzle");

    printf("\n%s (%d failing)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
