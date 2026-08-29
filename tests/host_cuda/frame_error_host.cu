#include "tests/host_cuda/lm_host_cuda.cuh"

#include <stdio.h>
#include <string.h>

LmHostDim3 blockIdx, threadIdx, blockDim, gridDim;

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/frame_error.cuh"
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

using ProbeKv = LmKvGeometry<
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

// THE FAIL-FRAME RECEIPT (BUG_LEDGER: trap-on-corruption, sparse-attn
// bounds). Corruption lands in the per-frame error record, the kernel
// returns a bounded result, and nothing traps. These probes run the real
// kernel bodies on the CPU harness; the same injections under a live CUDA
// context - where the old code emitted PTX trap and killed the whole GPU
// context - run on spark5 as tools/hardware/spark_frame_error_probe.cu.
int main(void)
{
    static LmFrameError frame_error;

    // -- the protocol: first reporter wins, reset clears --------------------
    LmFrameErrorReset(&frame_error);
    Expect(frame_error.error_code == LM_FRAME_ERROR_NONE,
        "a reset record reads no error");
    LmFrameErrorReport(&frame_error,
        (uint32_t)LM_FRAME_ERROR_ROUTE_MAP_OUT_OF_RANGE,
        0u, 3u, 9u, 0u, 2u);
    LmFrameErrorReport(&frame_error,
        (uint32_t)LM_FRAME_ERROR_PAYLOAD_WINDOW_OUT_OF_RANGE,
        0u, 4u, 5u, 6u, 7u);
    Expect(
        frame_error.error_code == LM_FRAME_ERROR_ROUTE_MAP_OUT_OF_RANGE &&
            frame_error.row == 3u && frame_error.sequence == 9u &&
            frame_error.page == 2u,
        "first reporter wins the frame error record");
    LmFrameErrorReport((LmFrameError *)0, 1u, 0u, 0u, 0u, 0u, 0u);
    Expect(frame_error.error_code == LM_FRAME_ERROR_ROUTE_MAP_OUT_OF_RANGE,
        "reporting into an unwired slot is a no-op, not a fault");

    // -- a wild sparse position records and returns, never addresses --------
    static uint8_t pool[ProbeKv::kPageBytes];
    static uint32_t page_table[1] = {0u};
    static uint16_t latent_query[16u];
    static uint16_t latent_output[12u];
    static uint32_t sequence[1] = {0u};
    static uint32_t context_length[1] = {1u};
    static uint32_t selected_positions[2] = {0u, PAGE_SLOTS * 64u};
    LmKvAccessError access_error;
    LmKvView view;
    memset(&view, 0, sizeof(view));
    Expect(
        LmKvViewInitialize(
            &view, pool, page_table, 1u, 1u, 1u, &access_error) == 0,
        "KV view with its frame error slot constructs");
    for (uint32_t index = 0u; index < 16u; ++index)
        latent_query[index] = LmFloatToBf16(0.25f);
    // The cache slot for (sequence 0, position 0) holds the same constant,
    // so the single-position attention output is exactly the value.
    for (uint32_t index = 0u; index < ProbeKv::kPageBytes / sizeof(uint16_t); ++index)
        ((uint16_t *)pool)[index] = LmFloatToBf16(0.25f);
    memset(latent_output, 0, sizeof(latent_output));
    LmKvAccessErrorReset(&access_error);
    LM_HOST_LAUNCH(
        dim3(1u, 1u),
        (LmAttentionDecodeKernel<ProbeKv, THREADS, 12u, 4u>(
            latent_query, latent_query, view, sequence, context_length,
            selected_positions, 2u, 1u, 1.0f, latent_output, 0)));
    Expect(
        access_error.error_code == LM_KV_ACCESS_ERROR_PAGE_TABLE_OUT_OF_RANGE,
        "a wild sparse position records a KV access failure instead of "
        "forming a wild KV address");
    Expect(
        latent_output[0] == 0u,
        "the frame's output stays dead rather than plausible over a "
        "failed access");
    Expect(access_error.row == 0u && access_error.page == 64u,
        "the record carries the row and the violating page for diagnosis");

    // The in-range position still attended: the same kernel, a good
    // selection, produces the single-position softmax the reference
    // computes - the failure path did not poison the good path.
    static uint32_t good_positions[1] = {0u};
    static uint16_t attended[12u];
    LmKvAccessErrorReset(&access_error);
    LM_HOST_LAUNCH(
        dim3(1u, 1u),
        (LmAttentionDecodeKernel<ProbeKv, THREADS, 12u, 4u>(
            latent_query, latent_query, view, sequence, context_length,
            good_positions, 1u, 1u, 1.0f, attended, 0)));
    Expect(
        access_error.error_code == LM_KV_ACCESS_ERROR_NONE &&
            LmBf16ToFloat(attended[0]) > 0.24f &&
            LmBf16ToFloat(attended[0]) < 0.26f,
        "an in-bounds selection still attends normally");

    printf("%s (%d failures)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
