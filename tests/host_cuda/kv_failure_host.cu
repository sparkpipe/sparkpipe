#include "tests/host_cuda/lm_host_cuda.cuh"

#include <stdio.h>
#include <string.h>

LmHostDim3 blockIdx, threadIdx, blockDim, gridDim;

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/mma.cuh"
#undef LM_WARP_LANES
#define LM_WARP_LANES LM_HOST_WARP_LANES
#include "inference/kernels/kv.cuh"
#include "inference/kernels/gqa.cuh"
#include "inference/kernels/attn.cuh"

#define THREADS 1u
#define KV_HEADS 2u
#define HEAD_DIM 4u
#define VALUE_DIM 4u
#define HEADS 4u
#define PAGE_SLOTS 2u

using FailureKv = LmKvGeometry<
    KV_HEADS * (HEAD_DIM + VALUE_DIM) * sizeof(uint16_t),
    PAGE_SLOTS,
    true>;

static int32_t failures;

static void Expect(int32_t condition, const char *label)
{
    if (condition)
    {
        printf("ok %s\n", label);
    }
    else
    {
        printf("FAIL %s\n", label);
        ++failures;
    }
}

int main(void)
{
    static uint8_t pool[FailureKv::kPageBytes];
    static uint32_t page_table[1];
    static uint16_t key[KV_HEADS * HEAD_DIM];
    static uint16_t value[KV_HEADS * VALUE_DIM];
    static uint16_t query[HEADS * HEAD_DIM];
    static uint16_t output[HEADS * VALUE_DIM];
    static uint16_t latent_query[16u];
    static uint16_t latent_output[12u];
    static uint32_t sequence[1] = {0u};
    static uint32_t position[1] = {0u};
    static uint32_t context_length[1] = {1u};
    LmKvAccessError error;
    LmKvView view;

    memset(&view, 0, sizeof(view));
    Expect(
        LmKvViewInitialize(
            &view,
            pool,
            page_table,
            1u,
            1u,
            1u,
            &error) == 0,
        "canonical KV view construction validates all ownership fields");

    page_table[0] = LM_KV_PAGE_UNMAPPED;
    LmKvAccessErrorReset(&error);
    LM_HOST_LAUNCH(
        dim3(1u),
        (LmGqaKvStoreKernel<FailureKv, THREADS, KV_HEADS, HEAD_DIM, VALUE_DIM>(
            view, key, value, sequence, position, 1u)));
    Expect(
        error.error_code == LM_KV_ACCESS_ERROR_PAGE_UNMAPPED &&
            error.access_kind == LM_KV_ACCESS_WRITE &&
            error.row == 0u && error.sequence == 0u && error.position == 0u,
        "unmapped store records a terminal write failure");

    page_table[0] = 1u;
    LmKvAccessErrorReset(&error);
    LM_HOST_LAUNCH(
        dim3(1u),
        (LmGqaKvStoreKernel<FailureKv, THREADS, KV_HEADS, HEAD_DIM, VALUE_DIM>(
            view, key, value, sequence, position, 1u)));
    Expect(
        error.error_code == LM_KV_ACCESS_ERROR_POOL_PAGE_OUT_OF_RANGE &&
            error.page == 1u,
        "physical pages outside the pool fail closed");

    page_table[0] = 0u;
    position[0] = PAGE_SLOTS;
    LmKvAccessErrorReset(&error);
    LM_HOST_LAUNCH(
        dim3(1u),
        (LmGqaKvStoreKernel<FailureKv, THREADS, KV_HEADS, HEAD_DIM, VALUE_DIM>(
            view, key, value, sequence, position, 1u)));
    Expect(
        error.error_code == LM_KV_ACCESS_ERROR_PAGE_TABLE_OUT_OF_RANGE &&
            error.page == 1u,
        "logical pages outside the sequence table fail closed");

    page_table[0] = LM_KV_PAGE_UNMAPPED;
    position[0] = 0u;
    memset(output, 0x7f, sizeof(output));
    LmKvAccessErrorReset(&error);
    LM_HOST_LAUNCH(
        dim3(1u, HEADS),
        (LmGqaAttentionDecodeKernel<
            FailureKv, THREADS, KV_HEADS, HEAD_DIM, VALUE_DIM>(
                query,
                view,
                sequence,
                context_length,
                0,
                0u,
                HEADS,
                1.0f,
                output,
                0)));
    Expect(
        error.error_code == LM_KV_ACCESS_ERROR_PAGE_UNMAPPED &&
            error.access_kind == LM_KV_ACCESS_READ,
        "attention cannot turn missing history into plausible output");

    memset(latent_output, 0x7f, sizeof(latent_output));
    LmKvAccessErrorReset(&error);
    LM_HOST_LAUNCH(
        dim3(1u, 1u),
        (LmAttentionDecodeKernel<FailureKv, THREADS, 12u, 4u>(
            latent_query,
            latent_query,
            view,
            sequence,
            context_length,
            0,
            0u,
            1u,
            1.0f,
            latent_output,
            0)));
    Expect(
        error.error_code == LM_KV_ACCESS_ERROR_PAGE_UNMAPPED &&
            error.access_kind == LM_KV_ACCESS_READ,
        "latent attention cannot skip a required missing page");

    page_table[0] = 0u;
    LmKvAccessErrorReset(&error);
    LM_HOST_LAUNCH(
        dim3(1u, 3u),
        (LmGqaAttentionDecodeKernel<
            FailureKv, THREADS, KV_HEADS, HEAD_DIM, VALUE_DIM>(
                query,
                view,
                sequence,
                context_length,
                0,
                0u,
                3u,
                1.0f,
                output,
                0)));
    Expect(
        error.error_code == LM_KV_ACCESS_ERROR_INVALID_GQA_GEOMETRY,
        "query heads must be divisible by KV heads");

    printf("%s (%d failures)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
