// l4_route_device_check.hip - on-hardware comparator for the L4 synthetic
// sealed-route fixtures produced by l4_route_driver.c (SLR4 v1 layout).
//
// Compiled only where a ROCm toolchain exists - see
// run_l4_route_validation.sh check mode for the exact hipcc command.
//
// Comparison discipline (plan section 5.3): every C2 field compares
// bit-exact - expert offsets is THE declared probe surface (hwiface_v1.md
// section 3.5), and the grouped arrays compare exactly too because this
// target's packing order is a documented fixed rule; the C3 accumulator
// compares under the fixture's abs/rel tolerances because host libm
// expf/exp2f differ from the device math library by ulps.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hip/hip_runtime_api.h>

#include "sparkpipe/spark_hw_iface.h"
#include "spark_dsv4_rocm_islands.h"

#include "l4_route_synthesis.h"

#define SLR4_CHECK_MAGIC 0x53524c34u
#define SLR4_CHECK_VERSION 1u
#define SLR4_TIMEOUT_LOOPS 200000000u

namespace {

struct Slr4Fixture
{
    Slr4CaseShape shape;
    float tolerance_abs;
    float tolerance_rel;
    uint32_t pair_count;
    uint32_t *indices_u32;
    float *weights_f32;
    uint16_t *accum_initial_bf16;
    uint32_t *expected_offsets_u32;
    uint32_t *expected_inverse_pair_u32;
    uint32_t *expected_grouped_source_token_u32;
    uint32_t *expected_grouped_expert_u32;
    uint16_t *expected_accum_bf16;
};

uint32_t ReadU32(const uint8_t **cursor)
{
    uint32_t value = (uint32_t)(*cursor)[0] |
                     ((uint32_t)(*cursor)[1] << 8u) |
                     ((uint32_t)(*cursor)[2] << 16u) |
                     ((uint32_t)(*cursor)[3] << 24u);
    *cursor += 4u;
    return value;
}

uint64_t ReadU64(const uint8_t **cursor)
{
    uint64_t low = ReadU32(cursor);
    return low | ((uint64_t)ReadU32(cursor) << 32u);
}

float ReadF32(const uint8_t **cursor)
{
    union { uint32_t bits; float value; } converter;
    converter.bits = ReadU32(cursor);
    return converter.value;
}

int LoadFixture(const char *path, Slr4Fixture *fixture)
{
    FILE *file = fopen(path, "rb");
    long length;
    uint8_t *raw;
    const uint8_t *cursor;
    size_t accum_elements;

    memset(fixture, 0, sizeof(*fixture));
    if (file == 0)
    {
        fprintf(stderr, "slr4-check: cannot open %s\n", path);
        return 1;
    }
    fseek(file, 0, SEEK_END);
    length = ftell(file);
    fseek(file, 0, SEEK_SET);
    if (length <= 0)
    {
        fprintf(stderr, "slr4-check: empty fixture %s\n", path);
        fclose(file);
        return 1;
    }
    raw = (uint8_t *)malloc((size_t)length);
    if (raw == 0 || fread(raw, 1u, (size_t)length, file) != (size_t)length)
    {
        fprintf(stderr, "slr4-check: read failure %s\n", path);
        fclose(file);
        free(raw);
        return 1;
    }
    fclose(file);

    cursor = raw;
    if (ReadU32(&cursor) != SLR4_CHECK_MAGIC ||
        ReadU32(&cursor) != SLR4_CHECK_VERSION)
    {
        fprintf(stderr, "slr4-check: bad magic/version %s\n", path);
        free(raw);
        return 1;
    }
    fixture->shape.row_count = ReadU32(&cursor);
    fixture->shape.experts_per_token = ReadU32(&cursor);
    fixture->shape.expert_count = ReadU32(&cursor);
    fixture->shape.hidden_dimension = ReadU32(&cursor);
    fixture->shape.moe_intermediate_dimension = ReadU32(&cursor);
    fixture->shape.weight_format = ReadU32(&cursor);
    fixture->shape.scale_group = ReadU32(&cursor);
    fixture->shape.swiglu_limit = ReadF32(&cursor);
    fixture->shape.seed = ReadU64(&cursor);
    fixture->tolerance_abs = ReadF32(&cursor);
    fixture->tolerance_rel = ReadF32(&cursor);
    fixture->pair_count =
        fixture->shape.row_count * fixture->shape.experts_per_token;
    accum_elements = (size_t)fixture->shape.row_count *
                     fixture->shape.hidden_dimension;

    fixture->indices_u32 =
        (uint32_t *)malloc(sizeof(uint32_t) * (fixture->pair_count + 1u));
    fixture->weights_f32 =
        (float *)malloc(sizeof(float) * (fixture->pair_count + 1u));
    fixture->accum_initial_bf16 =
        (uint16_t *)malloc(sizeof(uint16_t) * accum_elements);
    fixture->expected_offsets_u32 = (uint32_t *)malloc(
        sizeof(uint32_t) * (fixture->shape.expert_count + 1u));
    fixture->expected_inverse_pair_u32 = (uint32_t *)malloc(
        sizeof(uint32_t) * (fixture->pair_count + 1u));
    fixture->expected_grouped_source_token_u32 = (uint32_t *)malloc(
        sizeof(uint32_t) * (fixture->pair_count + 1u));
    fixture->expected_grouped_expert_u32 = (uint32_t *)malloc(
        sizeof(uint32_t) * (fixture->pair_count + 1u));
    fixture->expected_accum_bf16 =
        (uint16_t *)malloc(sizeof(uint16_t) * accum_elements);
    if (fixture->indices_u32 == 0 || fixture->weights_f32 == 0 ||
        fixture->accum_initial_bf16 == 0 ||
        fixture->expected_offsets_u32 == 0 ||
        fixture->expected_inverse_pair_u32 == 0 ||
        fixture->expected_grouped_source_token_u32 == 0 ||
        fixture->expected_grouped_expert_u32 == 0 ||
        fixture->expected_accum_bf16 == 0)
    {
        fprintf(stderr, "slr4-check: out of host memory\n");
        free(raw);
        return 1;
    }
    for (uint32_t i = 0u; i < fixture->pair_count; ++i)
        fixture->indices_u32[i] = ReadU32(&cursor);
    for (uint32_t i = 0u; i < fixture->pair_count; ++i)
        fixture->weights_f32[i] = ReadF32(&cursor);
    for (size_t i = 0; i < accum_elements; ++i)
        fixture->accum_initial_bf16[i] = ReadU32(&cursor);
    for (uint32_t e = 0u; e <= fixture->shape.expert_count; ++e)
        fixture->expected_offsets_u32[e] = ReadU32(&cursor);
    for (uint32_t i = 0u; i < fixture->pair_count; ++i)
        fixture->expected_inverse_pair_u32[i] = ReadU32(&cursor);
    for (uint32_t i = 0u; i < fixture->pair_count; ++i)
        fixture->expected_grouped_source_token_u32[i] = ReadU32(&cursor);
    for (uint32_t i = 0u; i < fixture->pair_count; ++i)
        fixture->expected_grouped_expert_u32[i] = ReadU32(&cursor);
    for (size_t i = 0; i < accum_elements; ++i)
        fixture->expected_accum_bf16[i] = ReadU32(&cursor);
    free(raw);
    return 0;
}

void FreeFixture(Slr4Fixture *fixture)
{
    free(fixture->indices_u32);
    free(fixture->weights_f32);
    free(fixture->accum_initial_bf16);
    free(fixture->expected_offsets_u32);
    free(fixture->expected_inverse_pair_u32);
    free(fixture->expected_grouped_source_token_u32);
    free(fixture->expected_grouped_expert_u32);
    free(fixture->expected_accum_bf16);
}

volatile unsigned int g_callback_seen = 0u;

void Slr4CompletionCallback(void *context)
{
    (void)context;
    g_callback_seen = 1u;
}

} // namespace

int main(int argc, char **argv)
{
    Slr4Fixture fixture;
    const Slr4CaseShape *shape;
    SparkHwQueue queue = 0;
    SparkHwEvent join_event = 0;
    SparkDsv4RocmLinearView w13_gate_view;
    SparkDsv4RocmLinearView w13_up_view;
    SparkDsv4RocmLinearView w2_down_view;
    size_t view_rows_w13, view_rows_w2;
    size_t gate_payload_bytes, up_payload_bytes, down_payload_bytes;
    size_t accum_bytes;
    uint8_t *w13_gate_buffer, *w13_up_buffer, *w2_down_buffer;
    uint16_t *hidden_host, *accum_out_host;
    uint32_t *offsets_host, *inverse_host, *grouped_source_host,
        *grouped_expert_host;
    void *indices_device = 0, *weights_device = 0, *hidden_device = 0;
    void *grouped_source_device = 0, *grouped_expert_device = 0;
    void *offsets_device = 0, *inverse_device = 0;
    void *up_scratch_device = 0, *pair_out_device = 0, *accum_device = 0;
    void *w13_gate_device = 0, *w13_up_device = 0, *w2_down_device = 0;
    unsigned int loop;
    size_t elements, violations = 0, index;
    double worst = 0.0;
    int failures = 0;
    int status_int;

    if (argc != 2)
    {
        fprintf(stderr, "usage: %s <fixture.bin>\n", argv[0]);
        return 2;
    }
    if (LoadFixture(argv[1], &fixture) != 0)
        return 2;
    shape = &fixture.shape;
    const uint32_t pair_count = fixture.pair_count;
    accum_bytes =
        sizeof(uint16_t) * (size_t)shape->row_count * shape->hidden_dimension;

    /* Queue/event through the frozen primitive family so the entry runs
     * exactly as production drives it. Completion rides the stream-ordered
     * host callback; queue_synchronize stays on the failure/timeout path
     * only (the frozen sync-discipline note). */
    if (spark_hw_queue_create(&queue, SPARK_HW_QUEUE_DEFAULT, 0) !=
            SPARK_HW_OK ||
        spark_hw_event_create(&join_event, SPARK_HW_EVENT_DISABLE_TIMING) !=
            SPARK_HW_OK)
    {
        fprintf(stderr, "slr4-check: queue/event setup failed\n");
        return 2;
    }

    /* Host-side synthesis identical to the generator's reference (shared
     * header - no drift possible). */
    view_rows_w13 = (size_t)shape->expert_count *
                    shape->moe_intermediate_dimension;
    view_rows_w2 = (size_t)shape->expert_count * shape->hidden_dimension;
    gate_payload_bytes =
        shape->weight_format == SLR4_FORMAT_BF16
            ? (size_t)view_rows_w13 * shape->hidden_dimension *
                  sizeof(uint16_t)
            : (size_t)view_rows_w13 * shape->hidden_dimension /
                  (shape->weight_format == SLR4_FORMAT_MXFP4 ? 2u : 1u);
    up_payload_bytes = gate_payload_bytes;
    down_payload_bytes =
        shape->weight_format == SLR4_FORMAT_BF16
            ? (size_t)view_rows_w2 * shape->moe_intermediate_dimension *
                  sizeof(uint16_t)
            : (size_t)view_rows_w2 * shape->moe_intermediate_dimension /
                  (shape->weight_format == SLR4_FORMAT_MXFP4 ? 2u : 1u);

    w13_gate_buffer = (uint8_t *)malloc(
        gate_payload_bytes +
        (shape->weight_format == SLR4_FORMAT_BF16
             ? 0u
             : (size_t)view_rows_w13 *
                   (shape->hidden_dimension / shape->scale_group)));
    w13_up_buffer = (uint8_t *)malloc(
        up_payload_bytes +
        (shape->weight_format == SLR4_FORMAT_BF16
             ? 0u
             : (size_t)view_rows_w13 *
                   (shape->hidden_dimension / shape->scale_group)));
    w2_down_buffer = (uint8_t *)malloc(
        down_payload_bytes +
        (shape->weight_format == SLR4_FORMAT_BF16
             ? 0u
             : (size_t)view_rows_w2 *
                   (shape->moe_intermediate_dimension / shape->scale_group)));
    hidden_host = (uint16_t *)malloc(accum_bytes);
    if (w13_gate_buffer == 0 || w13_up_buffer == 0 || w2_down_buffer == 0 ||
        hidden_host == 0)
    {
        fprintf(stderr, "slr4-check: synthesis allocation failed\n");
        return 2;
    }
    Slr4SynthesizeView(shape, shape->seed ^ 0x57313347ULL,
                       (uint32_t)view_rows_w13, shape->hidden_dimension,
                       w13_gate_buffer);
    Slr4SynthesizeView(shape, shape->seed ^ 0x57313355ULL,
                       (uint32_t)view_rows_w13, shape->hidden_dimension,
                       w13_up_buffer);
    Slr4SynthesizeView(shape, shape->seed ^ 0x5732444eULL,
                       (uint32_t)view_rows_w2,
                       shape->moe_intermediate_dimension, w2_down_buffer);
    Slr4SynthesizeHidden(shape, hidden_host);

    memset(&w13_gate_view, 0, sizeof(w13_gate_view));
    w13_gate_view.weight_format = shape->weight_format;
    w13_gate_view.rows = (uint32_t)view_rows_w13;
    w13_gate_view.columns = shape->hidden_dimension;
    w13_gate_view.scale_group = shape->scale_group;
    w13_up_view = w13_gate_view;
    w2_down_view = w13_gate_view;
    w2_down_view.rows = (uint32_t)view_rows_w2;
    w2_down_view.columns = shape->moe_intermediate_dimension;

    if (hipMalloc(&indices_device, sizeof(uint32_t) * (pair_count + 1u)) !=
            hipSuccess ||
        hipMalloc(&weights_device, sizeof(float) * (pair_count + 1u)) !=
            hipSuccess ||
        hipMalloc(&hidden_device, accum_bytes) != hipSuccess ||
        hipMalloc(&grouped_source_device,
                  sizeof(uint32_t) * (pair_count + 1u)) != hipSuccess ||
        hipMalloc(&grouped_expert_device,
                  sizeof(uint32_t) * (pair_count + 1u)) != hipSuccess ||
        hipMalloc(&offsets_device,
                  sizeof(uint32_t) * (shape->expert_count + 1u)) !=
            hipSuccess ||
        hipMalloc(&inverse_device, sizeof(uint32_t) * (pair_count + 1u)) !=
            hipSuccess ||
        hipMalloc(&up_scratch_device,
                  sizeof(uint16_t) * (size_t)pair_count *
                      shape->moe_intermediate_dimension) != hipSuccess ||
        hipMalloc(&pair_out_device,
                  sizeof(uint16_t) * (size_t)pair_count *
                      shape->hidden_dimension) != hipSuccess ||
        hipMalloc(&accum_device, accum_bytes) != hipSuccess ||
        hipMalloc(&w13_gate_device, gate_payload_bytes + (shape->weight_format == SLR4_FORMAT_BF16 ? 0u : (size_t)view_rows_w13 * (shape->hidden_dimension / shape->scale_group))) != hipSuccess ||
        hipMalloc(&w13_up_device, up_payload_bytes + (shape->weight_format == SLR4_FORMAT_BF16 ? 0u : (size_t)view_rows_w13 * (shape->hidden_dimension / shape->scale_group))) != hipSuccess ||
        hipMalloc(&w2_down_device, down_payload_bytes + (shape->weight_format == SLR4_FORMAT_BF16 ? 0u : (size_t)view_rows_w2 * (shape->moe_intermediate_dimension / shape->scale_group))) != hipSuccess)
    {
        fprintf(stderr, "slr4-check: hipMalloc failure\n");
        return 2;
    }

    /* Packed views carry their E8M0 scale region behind the payload INSIDE
     * one device allocation, matching the synthesized host layout; the
     * allocations above were sized payload+scale together. */
    w13_gate_view.payload = w13_gate_device;
    w13_up_view.payload = w13_up_device;
    w2_down_view.payload = w2_down_device;
    w13_gate_view.scale_data =
        shape->weight_format == SLR4_FORMAT_BF16
            ? 0
            : (void *)((uint8_t *)w13_gate_device + gate_payload_bytes);
    w13_up_view.scale_data =
        shape->weight_format == SLR4_FORMAT_BF16
            ? 0
            : (void *)((uint8_t *)w13_up_device + up_payload_bytes);
    w2_down_view.scale_data =
        shape->weight_format == SLR4_FORMAT_BF16
            ? 0
            : (void *)((uint8_t *)w2_down_device + down_payload_bytes);
 
    if (hipMemcpy(indices_device, fixture.indices_u32,
                  sizeof(uint32_t) * (pair_count + 1u),
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(weights_device, fixture.weights_f32,
                  sizeof(float) * (pair_count + 1u),
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(hidden_device, hidden_host, accum_bytes,
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(accum_device, fixture.accum_initial_bf16, accum_bytes,
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(w13_gate_device, w13_gate_buffer,
                  shape->weight_format == SLR4_FORMAT_BF16
                      ? gate_payload_bytes
                      : gate_payload_bytes +
                            (size_t)view_rows_w13 *
                                (shape->hidden_dimension /
                                 shape->scale_group),
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(w13_up_device, w13_up_buffer,
                  shape->weight_format == SLR4_FORMAT_BF16
                      ? up_payload_bytes
                      : up_payload_bytes +
                            (size_t)view_rows_w13 *
                                (shape->hidden_dimension /
                                 shape->scale_group),
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(w2_down_device, w2_down_buffer,
                  shape->weight_format == SLR4_FORMAT_BF16
                      ? down_payload_bytes
                      : down_payload_bytes +
                            (size_t)view_rows_w2 *
                                (shape->moe_intermediate_dimension /
                                 shape->scale_group),
                  hipMemcpyHostToDevice) != hipSuccess)
    {
        fprintf(stderr, "slr4-check: H2D copy failure\n");
        return 2;
    }

    status_int = (int)spark_dsv4_rocm_l4_moe_routed(
        queue, 0 /* fork */, join_event, 0 /* memory */, hidden_device,
        (const uint32_t *)indices_device, (const float *)weights_device,
        &w13_gate_view, &w13_up_view, &w2_down_view, shape->swiglu_limit,
        grouped_source_device, grouped_expert_device, offsets_device,
        inverse_device, up_scratch_device, pair_out_device, accum_device,
        shape->row_count, shape->expert_count, shape->experts_per_token,
        shape->hidden_dimension, shape->moe_intermediate_dimension);
    if (status_int != (int)SPARK_HW_OK)
    {
        fprintf(stderr, "slr4-check: island entry returned %d (%s)\n",
                status_int,
                SparkHwStatusToString((SparkHwStatus)status_int));
        return 2;
    }
    if (spark_hw_queue_enqueue_host_callback(queue, Slr4CompletionCallback,
                                             0) != SPARK_HW_OK)
    {
        fprintf(stderr, "slr4-check: callback enqueue failed\n");
        return 2;
    }
    for (loop = 0u; g_callback_seen == 0u && loop < SLR4_TIMEOUT_LOOPS;
         ++loop)
    {
    }
    if (g_callback_seen == 0u &&
        spark_hw_queue_synchronize(queue) != SPARK_HW_OK)
    {
        fprintf(stderr, "slr4-check: queue synchronize failed\n");
        return 2;
    }

    offsets_host =
        (uint32_t *)malloc(sizeof(uint32_t) * (shape->expert_count + 1u));
    inverse_host = (uint32_t *)malloc(sizeof(uint32_t) * (pair_count + 1u));
    grouped_source_host =
        (uint32_t *)malloc(sizeof(uint32_t) * (pair_count + 1u));
    grouped_expert_host =
        (uint32_t *)malloc(sizeof(uint32_t) * (pair_count + 1u));
    accum_out_host = (uint16_t *)malloc(accum_bytes);
    if (offsets_host == 0 || inverse_host == 0 || grouped_source_host == 0 ||
        grouped_expert_host == 0 || accum_out_host == 0 ||
        hipMemcpy(offsets_host, offsets_device,
                  sizeof(uint32_t) * (shape->expert_count + 1u),
                  hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(inverse_host, inverse_device,
                  sizeof(uint32_t) * (pair_count + 1u),
                  hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(grouped_source_host, grouped_source_device,
                  sizeof(uint32_t) * (pair_count + 1u),
                  hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(grouped_expert_host, grouped_expert_device,
                  sizeof(uint32_t) * (pair_count + 1u),
                  hipMemcpyDeviceToHost) != hipSuccess ||
        hipMemcpy(accum_out_host, accum_device, accum_bytes,
                  hipMemcpyDeviceToHost) != hipSuccess)
    {
        fprintf(stderr, "slr4-check: D2H copy failure\n");
        return 2;
    }

    if (memcmp(offsets_host, fixture.expected_offsets_u32,
               sizeof(uint32_t) * (shape->expert_count + 1u)) != 0)
    {
        fprintf(stderr, "slr4-check: C2 FAIL expert offsets\n");
        failures += 1;
    }
    if (memcmp(inverse_host, fixture.expected_inverse_pair_u32,
               sizeof(uint32_t) * pair_count) != 0)
    {
        fprintf(stderr, "slr4-check: C2 FAIL inverse map\n");
        failures += 1;
    }
    if (memcmp(grouped_source_host,
               fixture.expected_grouped_source_token_u32,
               sizeof(uint32_t) * pair_count) != 0)
    {
        fprintf(stderr, "slr4-check: C2 FAIL grouped source tokens\n");
        failures += 1;
    }
    if (memcmp(grouped_expert_host, fixture.expected_grouped_expert_u32,
               sizeof(uint32_t) * pair_count) != 0)
    {
        fprintf(stderr, "slr4-check: C2 FAIL grouped experts\n");
        failures += 1;
    }

    elements = accum_bytes / sizeof(uint16_t);
    for (index = 0; index < elements; ++index)
    {
        float got = Slr4Bf16ToFloat(accum_out_host[index]);
        float want = Slr4Bf16ToFloat(fixture.expected_accum_bf16[index]);
        double difference = (double)got - (double)want;
        double bound = (double)fixture.tolerance_abs +
                       (double)fixture.tolerance_rel * fabs((double)want);
        double magnitude = fabs(difference);
        if (!(magnitude <= bound))
            violations += 1u;
        if (magnitude > worst)
            worst = magnitude;
    }
    printf("slr4-check: C3 worst |delta| %.6g (tolerance %.3g + rel %.3g), "
           "%zu violations\n",
           worst, (double)fixture.tolerance_abs,
           (double)fixture.tolerance_rel, violations);
    if (violations != 0u)
        failures += 1;

    printf("slr4-check: %s -> %s\n", argv[1],
           failures == 0 ? "PASS" : "FAIL");

    (void)spark_hw_event_destroy(join_event);
    (void)spark_hw_queue_destroy(queue);
    FreeFixture(&fixture);
    return failures == 0 ? 0 : 1;
}
