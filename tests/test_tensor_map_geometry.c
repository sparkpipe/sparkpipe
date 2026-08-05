#include "inference/kernels/tensor_map.cuh"

#include <stdio.h>
#include <string.h>

static int32_t failures = 0;

static void expect(int32_t condition, const char *label)
{
    if (condition == 0)
    {
        printf("  FAIL %s\n", label);
        failures++;
        return;
    }
    printf("  ok   %s\n", label);
}

static int32_t build(
    uint64_t rows,
    uint64_t columns,
    uint64_t groups,
    uint32_t box_rows,
    uint32_t box_columns,
    uint32_t bits,
    LmTensorMapPlan *plan)
{
    LmTensorMapRequest request;
    static uint8_t aligned_storage[256] __attribute__((aligned(128)));

    memset(&request, 0, sizeof(request));
    request.global_address = aligned_storage;
    request.rows = rows;
    request.columns = columns;
    request.groups = groups;
    request.box_rows = box_rows;
    request.box_columns = box_columns;
    request.element_bits = bits;
    return LmTensorMapPlanBuild(&request, plan);
}

int32_t main(void)
{
    LmTensorMapPlan plan;
    int32_t status;

    printf("TMA descriptor geometry\n");

    printf("\nFP8 activation tile\n");
    status = build(128u, 6144u, 1u, 16u, 128u, LM_TM_BITS_FP8, &plan);
    expect(status == LM_TM_OK, "FP8 16x128 box builds");
    expect(plan.rank == 2u, "single group uses rank 2");
    expect(plan.row_bytes == 6144u, "FP8 row is K bytes");
    expect(plan.box_dimension[0] == 128u, "FP8 box is 128 bytes wide");
    expect(plan.swizzle_bytes == 128u, "FP8 box uses 128-byte swizzle");

    printf("\nINT6 expert tile\n");
    status = build(128u, 6144u, 1u, 16u, 128u, LM_TM_BITS_INT6, &plan);
    expect(status == LM_TM_OK, "INT6 TILE_K=128 builds");
    expect(plan.box_dimension[0] == 96u, "INT6 box is 96 bytes wide");
    expect(plan.swizzle_bytes == 0u, "INT6 box uses explicit unswizzled TMA");

    printf("\nINT7 expert tile\n");
    status = build(128u, 6144u, 1u, 16u, 256u, LM_TM_BITS_INT7, &plan);
    expect(status == LM_TM_OK, "INT7 TILE_K=256 builds");
    expect(plan.box_dimension[0] == 224u, "INT7 box is 224 bytes wide");
    expect(plan.swizzle_bytes == 0u, "INT7 box uses explicit unswizzled TMA");

    printf("\nFP4 expert tile\n");
    status = build(4096u, 6144u, 256u, 128u, 128u,
        LM_TM_BITS_NVFP4, &plan);
    expect(status == LM_TM_OK, "FP4 TILE_K=128 builds");
    expect(plan.rank == 3u, "expert tensor uses rank 3");
    expect(plan.row_bytes == 3072u, "FP4 row is K/2 bytes");
    expect(plan.box_dimension[0] == 64u, "FP4 box is 64 bytes wide");
    expect(plan.swizzle_bytes == 64u, "FP4 box uses 64-byte swizzle");
    expect(plan.global_stride_bytes[1] == 3072u * 4096u,
        "expert stride spans one weight matrix");

    printf("\nBF16 activation and weight tile\n");
    status = build(64u, 7168u, 1u, 16u, 64u, LM_TM_BITS_BF16, &plan);
    expect(status == LM_TM_OK, "BF16 TILE_K=64 builds");
    expect(plan.row_bytes == 14336u, "BF16 row has two bytes per element");
    expect(plan.box_dimension[0] == 128u, "BF16 box is 128 bytes wide");
    expect(plan.swizzle_bytes == 128u, "BF16 box uses 128-byte swizzle");

    printf("\nrejections\n");
    expect(build(128u, 6144u, 1u, 16u, 128u, 5u, &plan) ==
        LM_TM_ERR_BITS, "unsupported packed width rejected");
    expect(build(128u, 6145u, 1u, 16u, 128u,
        LM_TM_BITS_NVFP4, &plan) == LM_TM_ERR_ODD_COLUMNS,
        "odd FP4 column count rejected");
    expect(build(128u, 6144u, 1u, 16u, 96u,
        LM_TM_BITS_FP8, &plan) == LM_TM_ERR_BOX_ALIGN,
        "box without exact 32/64/128-byte swizzle rejected");
    expect(build(128u, 321u, 1u, 16u, 128u,
        LM_TM_BITS_FP8, &plan) == LM_TM_ERR_ROW_SWIZZLE,
        "global row stride must remain 16-byte aligned");
    expect(build(8u, 6144u, 1u, 16u, 128u,
        LM_TM_BITS_FP8, &plan) == LM_TM_ERR_BOX_EXCEEDS,
        "box taller than tensor rejected");

    printf("\n%s (%d failing checks)\n",
        failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
