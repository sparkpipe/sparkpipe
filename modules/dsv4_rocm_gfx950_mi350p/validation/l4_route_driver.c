/* l4_route_driver.c - synthetic sealed-route batch driver for the L4
 * grouped-MoE island (plan section 5.2 step 3; owed per
 * EVIDENCE_S7_STEP3_L4_GROUPED_MOE.md).
 *
 * Modes:
 *   l4_route_driver gen <outdir>   generate fixture cases + manifest
 *   l4_route_driver selfcheck      run reference-model invariant checks only
 *
 * The generator produces one little-endian binary fixture per case holding
 * the sealed route inputs, the initial accumulator pattern, the expected C2
 * outputs (expert offsets, inverse map, grouped source/expert arrays - all
 * bit-exact under the island's documented ordering) and the expected C3
 * post-island accumulator bytes plus the comparison tolerances. The
 * on-hardware checker (l4_route_device_check.hip) consumes these fixtures
 * against the real kernels; this host program exercises the same reference
 * model and its invariants everywhere, no GPU required.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_sha256.h"

#include "l4_route_synthesis.h"

#define SLR4_MAX_PAIRS 4096u
#define SLR4_TOLERANCE_ABS 2.0e-2f
#define SLR4_TOLERANCE_REL 1.0e-2f

typedef struct Slr4Case
{
    const char *name;
    Slr4CaseShape shape;
} Slr4Case;

/* Edge coverage: empty frame, single pair, uniform spread, heavy skew,
 * every-row-uses-every-expert (K == E boundary), prime expert count with
 * ragged groups, large sparse expert id space, FP8 block-scale views, and
 * MXFP4 nibble-packed views. All weight tensors are synthesized from the
 * case seed; fixtures stay tiny because both sides regenerate them. */
static const Slr4Case slr4_cases[] = {
    { "empty_frame", { 0u, 4u, 16u, 64u, 32u, SLR4_FORMAT_BF16, 0u,
                       0.0f, 101u } },
    { "single_pair", { 1u, 1u, 8u, 64u, 64u, SLR4_FORMAT_BF16, 0u,
                       0.0f, 102u } },
    { "uniform_spread", { 8u, 8u, 64u, 128u, 64u, SLR4_FORMAT_BF16, 0u,
                          8.0f, 103u } },
    { "heavy_skew", { 16u, 4u, 128u, 128u, 96u, SLR4_FORMAT_BF16, 0u,
                      0.0f, 104u } },
    { "full_topk_boundary", { 5u, 8u, 8u, 256u, 128u, SLR4_FORMAT_BF16, 0u,
                              4.0f, 105u } },
    { "prime_experts_ragged", { 33u, 6u, 77u, 192u, 160u, SLR4_FORMAT_BF16,
                                0u, 0.0f, 106u } },
    { "wide_expert_ids", { 4u, 2u, 1024u, 64u, 32u, SLR4_FORMAT_BF16, 0u,
                           0.0f, 107u } },
    { "fp8_block_scale", { 7u, 3u, 24u, 128u, 64u, SLR4_FORMAT_FP8, 128u,
                           8.0f, 108u } },
    { "mxfp4_nibble", { 6u, 3u, 16u, 128u, 64u, SLR4_FORMAT_MXFP4, 32u,
                        0.0f, 109u } }
};

#define SLR4_CASE_COUNT (sizeof(slr4_cases) / sizeof(slr4_cases[0]))

typedef struct Slr4CaseBuffers
{
    uint32_t pair_count;
    uint32_t *indices_u32;
    float *weights_f32;
    uint16_t *hidden_bf16;
    uint16_t *accum_initial_bf16;
    uint8_t *w13_gate_buffer;
    uint8_t *w13_up_buffer;
    uint8_t *w2_down_buffer;
    uint16_t *up_scratch_bf16;
    uint16_t *pair_out_scratch_bf16;
    /* Expected outputs (reference model). */
    uint32_t *expert_offsets_u32;
    uint32_t *inverse_pair_u32;
    uint32_t *grouped_source_token_u32;
    uint32_t *grouped_expert_u32;
    uint16_t *accum_expected_bf16;
} Slr4CaseBuffers;

static void *Slr4AllocOrDie(size_t bytes)
{
    void *buffer = malloc(bytes != 0u ? bytes : 1u);
    if (buffer == 0)
    {
        fprintf(stderr, "slr4: out of memory (%zu bytes)\n", bytes);
        exit(2);
    }
    memset(buffer, 0, bytes);
    return buffer;
}

static void Slr4BuildCase(const Slr4Case *spec, Slr4CaseBuffers *buffers)
{
    const Slr4CaseShape *shape = &spec->shape;
    uint32_t pair_count = Slr4PairCount(shape);
    size_t view_rows_w13 = (size_t)shape->expert_count *
                           shape->moe_intermediate_dimension;
    size_t view_rows_w2 = (size_t)shape->expert_count *
                          shape->hidden_dimension;

    buffers->pair_count = pair_count;
    buffers->indices_u32 =
        Slr4AllocOrDie(sizeof(uint32_t) * (pair_count + 1u));
    buffers->weights_f32 =
        Slr4AllocOrDie(sizeof(float) * (pair_count + 1u));
    buffers->hidden_bf16 = Slr4AllocOrDie(
        sizeof(uint16_t) * (size_t)shape->row_count * shape->hidden_dimension);
    buffers->accum_initial_bf16 = Slr4AllocOrDie(
        sizeof(uint16_t) * (size_t)shape->row_count * shape->hidden_dimension);
    buffers->w13_gate_buffer =
        Slr4AllocOrDie(Slr4ViewPayloadBytes(shape, (uint32_t)view_rows_w13,
                                            shape->hidden_dimension));
    buffers->w13_up_buffer =
        Slr4AllocOrDie(Slr4ViewPayloadBytes(shape, (uint32_t)view_rows_w13,
                                            shape->hidden_dimension));
    buffers->w2_down_buffer =
        Slr4AllocOrDie(Slr4ViewPayloadBytes(shape, (uint32_t)view_rows_w2,
                                            shape->moe_intermediate_dimension));
    buffers->up_scratch_bf16 = Slr4AllocOrDie(
        sizeof(uint16_t) * (size_t)pair_count *
        shape->moe_intermediate_dimension);
    buffers->pair_out_scratch_bf16 = Slr4AllocOrDie(
        sizeof(uint16_t) * (size_t)pair_count * shape->hidden_dimension);
    buffers->expert_offsets_u32 =
        Slr4AllocOrDie(sizeof(uint32_t) * (shape->expert_count + 1u));
    buffers->inverse_pair_u32 =
        Slr4AllocOrDie(sizeof(uint32_t) * (pair_count + 1u));
    buffers->grouped_source_token_u32 =
        Slr4AllocOrDie(sizeof(uint32_t) * (pair_count + 1u));
    buffers->grouped_expert_u32 =
        Slr4AllocOrDie(sizeof(uint32_t) * (pair_count + 1u));
    buffers->accum_expected_bf16 = Slr4AllocOrDie(
        sizeof(uint16_t) * (size_t)shape->row_count * shape->hidden_dimension);

    Slr4SynthesizeRoute(shape, buffers->indices_u32, buffers->weights_f32);
    Slr4SynthesizeHidden(shape, buffers->hidden_bf16);
    Slr4SynthesizeAccumulator(shape, buffers->accum_initial_bf16);
    Slr4SynthesizeView(shape, shape->seed ^ 0x57313347ULL,
                       (uint32_t)view_rows_w13, shape->hidden_dimension,
                       buffers->w13_gate_buffer);
    Slr4SynthesizeView(shape, shape->seed ^ 0x57313355ULL,
                       (uint32_t)view_rows_w13, shape->hidden_dimension,
                       buffers->w13_up_buffer);
    Slr4SynthesizeView(shape, shape->seed ^ 0x5732444eULL,
                       (uint32_t)view_rows_w2,
                       shape->moe_intermediate_dimension,
                       buffers->w2_down_buffer);

    memcpy(buffers->accum_expected_bf16, buffers->accum_initial_bf16,
           sizeof(uint16_t) * (size_t)shape->row_count *
               shape->hidden_dimension);

    Slr4ReferenceRouteGroup(buffers->indices_u32, pair_count,
                            shape->expert_count, shape->experts_per_token,
                            buffers->expert_offsets_u32,
                            buffers->inverse_pair_u32,
                            buffers->grouped_source_token_u32,
                            buffers->grouped_expert_u32);
    Slr4ReferenceMoeChain(
        shape, buffers->hidden_bf16, buffers->grouped_source_token_u32,
        buffers->grouped_expert_u32, buffers->inverse_pair_u32,
        buffers->weights_f32, buffers->w13_gate_buffer,
        buffers->w13_up_buffer, buffers->w2_down_buffer,
        buffers->up_scratch_bf16, buffers->pair_out_scratch_bf16,
        buffers->accum_expected_bf16);
}

static void Slr4FreeCase(Slr4CaseBuffers *buffers)
{
    free(buffers->indices_u32);
    free(buffers->weights_f32);
    free(buffers->hidden_bf16);
    free(buffers->accum_initial_bf16);
    free(buffers->w13_gate_buffer);
    free(buffers->w13_up_buffer);
    free(buffers->w2_down_buffer);
    free(buffers->up_scratch_bf16);
    free(buffers->pair_out_scratch_bf16);
    free(buffers->expert_offsets_u32);
    free(buffers->inverse_pair_u32);
    free(buffers->grouped_source_token_u32);
    free(buffers->grouped_expert_u32);
    free(buffers->accum_expected_bf16);
    memset(buffers, 0, sizeof(*buffers));
}

/* Invariants that must hold for ANY correct realization, checked here
 * against the reference so a broken reference cannot silently emit golden
 * fixtures. */
static int Slr4CheckCaseInvariants(const Slr4Case *spec,
                                   const Slr4CaseBuffers *buffers)
{
    const Slr4CaseShape *shape = &spec->shape;
    const uint32_t pair_count = buffers->pair_count;
    uint32_t expert, i;
    uint8_t *slot_seen;
    int failures = 0;

    /* Offsets: start at zero, non-decreasing, total at [expert_count]. */
    if (buffers->expert_offsets_u32[0] != 0u)
    {
        fprintf(stderr, "slr4: %s offsets[0] != 0\n", spec->name);
        failures += 1;
    }
    for (expert = 0u; expert < shape->expert_count; ++expert)
    {
        if (buffers->expert_offsets_u32[expert] >
            buffers->expert_offsets_u32[expert + 1u])
        {
            fprintf(stderr, "slr4: %s offsets not ascending at [%u]\n",
                    spec->name, expert);
            failures += 1;
        }
    }
    if (buffers->expert_offsets_u32[shape->expert_count] != pair_count)
    {
        fprintf(stderr, "slr4: %s offsets total %u != pair count %u\n",
                spec->name,
                buffers->expert_offsets_u32[shape->expert_count], pair_count);
        failures += 1;
    }

    slot_seen = (uint8_t *)calloc(pair_count + 1u, 1u);
    if (slot_seen == 0)
    {
        fprintf(stderr, "slr4: %s out of memory for invariant check\n",
                spec->name);
        return 1;
    }

    /* Each pair lands inside its own expert's window; grouped arrays agree;
     * the inverse map is a perfect permutation (every slot exactly once). */
    for (i = 0u; i < pair_count; ++i)
    {
        uint32_t slot = buffers->inverse_pair_u32[i];
        uint32_t owned_expert = buffers->indices_u32[i];
        uint32_t window_base, window_end;

        if (owned_expert >= shape->expert_count)
        {
            fprintf(stderr, "slr4: %s route index %u out of range\n",
                    spec->name, owned_expert);
            failures += 1;
            continue;
        }
        window_base = buffers->expert_offsets_u32[owned_expert];
        window_end = buffers->expert_offsets_u32[owned_expert + 1u];
        if (slot < window_base || slot >= window_end)
        {
            fprintf(stderr,
                    "slr4: %s pair %u (expert %u) landed in slot %u outside "
                    "[%u,%u)\n",
                    spec->name, i, owned_expert, slot, window_base,
                    window_end);
            failures += 1;
            continue;
        }
        if (buffers->grouped_expert_u32[slot] != owned_expert ||
            buffers->grouped_source_token_u32[slot] !=
                i / shape->experts_per_token)
        {
            fprintf(stderr, "slr4: %s grouped arrays disagree at pair %u\n",
                    spec->name, i);
            failures += 1;
        }
        if (slot_seen[slot] != 0u)
        {
            fprintf(stderr, "slr4: %s slot %u written twice\n", spec->name,
                    slot);
            failures += 1;
        }
        slot_seen[slot] = 1u;
    }

    /* Ascending within-expert packing order (the documented fixed rule). */
    for (i = 1u; i < pair_count; ++i)
    {
        if (buffers->indices_u32[i] == buffers->indices_u32[i - 1u] &&
            buffers->inverse_pair_u32[i] <=
                buffers->inverse_pair_u32[i - 1u])
        {
            fprintf(stderr,
                    "slr4: %s within-expert order violated at pair %u\n",
                    spec->name, i);
            failures += 1;
        }
    }

    free(slot_seen);
    return failures;
}

/* ---- Fixture emission (little-endian layout, doc in header banner) ---- */

static int Slr4WriteU32(FILE *file, uint32_t value)
{
    uint8_t raw[4];
    raw[0] = (uint8_t)value;
    raw[1] = (uint8_t)(value >> 8u);
    raw[2] = (uint8_t)(value >> 16u);
    raw[3] = (uint8_t)(value >> 24u);
    return fwrite(raw, 1u, 4u, file) == 4u ? 0 : -1;
}

static int Slr4WriteU64(FILE *file, uint64_t value)
{
    uint32_t low = (uint32_t)value;
    if (Slr4WriteU32(file, low) != 0)
        return -1;
    return Slr4WriteU32(file, (uint32_t)(value >> 32u));
}

static int Slr4WriteF32(FILE *file, float value)
{
    union { float value; uint32_t bits; } converter;
    converter.value = value;
    return Slr4WriteU32(file, converter.bits);
}

static int Slr4WriteU32Array(FILE *file, const uint32_t *values,
                             size_t count)
{
    size_t index;
    for (index = 0; index < count; ++index)
    {
        if (Slr4WriteU32(file, values[index]) != 0)
            return -1;
    }
    return 0;
}

static int Slr4WriteF32Array(FILE *file, const float *values, size_t count)
{
    size_t index;
    for (index = 0; index < count; ++index)
    {
        if (Slr4WriteF32(file, values[index]) != 0)
            return -1;
    }
    return 0;
}

static int Slr4WriteU16Array(FILE *file, const uint16_t *values,
                             size_t count)
{
    size_t index;
    for (index = 0; index < count; ++index)
    {
        uint8_t raw[2];
        raw[0] = (uint8_t)values[index];
        raw[1] = (uint8_t)(values[index] >> 8u);
        if (fwrite(raw, 1u, 2u, file) != 2u)
            return -1;
    }
    return 0;
}

static int Slr4GenerateCase(const char *outdir, const Slr4Case *spec,
                            char (*hex_out)[SPARK_SHA256_HEX_BYTES])
{
    const Slr4CaseShape *shape = &spec->shape;
    Slr4CaseBuffers buffers;
    char path[512];
    FILE *file;
    int failures = 0;

    Slr4BuildCase(spec, &buffers);
    failures += Slr4CheckCaseInvariants(spec, &buffers);
    if (failures != 0)
    {
        Slr4FreeCase(&buffers);
        return failures;
    }

    snprintf(path, sizeof(path), "%s/l4_case_%s.bin", outdir, spec->name);
    file = fopen(path, "wb");
    if (file == 0)
    {
        fprintf(stderr, "slr4: cannot open %s\n", path);
        Slr4FreeCase(&buffers);
        return 1;
    }

    (void)Slr4WriteU32(file, SLR4_MAGIC);
    (void)Slr4WriteU32(file, SLR4_VERSION);
    (void)Slr4WriteU32(file, shape->row_count);
    (void)Slr4WriteU32(file, shape->experts_per_token);
    (void)Slr4WriteU32(file, shape->expert_count);
    (void)Slr4WriteU32(file, shape->hidden_dimension);
    (void)Slr4WriteU32(file, shape->moe_intermediate_dimension);
    (void)Slr4WriteU32(file, shape->weight_format);
    (void)Slr4WriteU32(file, shape->scale_group);
    (void)Slr4WriteF32(file, shape->swiglu_limit);
    (void)Slr4WriteU64(file, shape->seed);
    (void)Slr4WriteF32(file, SLR4_TOLERANCE_ABS);
    (void)Slr4WriteF32(file, SLR4_TOLERANCE_REL);
    (void)Slr4WriteU32Array(file, buffers.indices_u32, buffers.pair_count);
    (void)Slr4WriteF32Array(file, buffers.weights_f32, buffers.pair_count);
    (void)Slr4WriteU16Array(
        file, buffers.accum_initial_bf16,
        (size_t)shape->row_count * shape->hidden_dimension);
    (void)Slr4WriteU32Array(file, buffers.expert_offsets_u32,
                            shape->expert_count + 1u);
    (void)Slr4WriteU32Array(file, buffers.inverse_pair_u32,
                            buffers.pair_count);
    (void)Slr4WriteU32Array(file, buffers.grouped_source_token_u32,
                            buffers.pair_count);
    (void)Slr4WriteU32Array(file, buffers.grouped_expert_u32,
                            buffers.pair_count);
    (void)Slr4WriteU16Array(
        file, buffers.accum_expected_bf16,
        (size_t)shape->row_count * shape->hidden_dimension);
    fclose(file);

    if (SparkSha256File(path, hex_out[0]) != SPARK_STATUS_OK)
    {
        fprintf(stderr, "slr4: sha256 failed for %s\n", path);
        failures += 1;
    }
    Slr4FreeCase(&buffers);
    return failures;
}

static int Slr4RunSelfcheck(void)
{
    size_t case_index;
    int failures = 0;
    for (case_index = 0; case_index < SLR4_CASE_COUNT; ++case_index)
    {
        Slr4CaseBuffers buffers;
        Slr4BuildCase(&slr4_cases[case_index], &buffers);
        failures += Slr4CheckCaseInvariants(&slr4_cases[case_index],
                                            &buffers);
        Slr4FreeCase(&buffers);
    }
    printf("slr4: selfcheck over %zu cases: %s\n", SLR4_CASE_COUNT,
           failures == 0 ? "OK" : "FAILED");
    return failures == 0 ? 0 : 1;
}

static int Slr4RunGenerate(const char *outdir)
{
    char manifest_path[512];
    FILE *manifest;
    size_t case_index;
    int failures = 0;

    manifest = fopen("/dev/null", "r"); /* placeholder to silence unused */
    (void)manifest;
    snprintf(manifest_path, sizeof(manifest_path), "%s/manifest.txt",
             outdir);
    manifest = fopen(manifest_path, "w");
    if (manifest == 0)
    {
        fprintf(stderr, "slr4: cannot open %s\n", manifest_path);
        return 1;
    }
    fprintf(manifest, "# L4 synthetic sealed-route fixtures (SLR4 v1)\n");

    for (case_index = 0; case_index < SLR4_CASE_COUNT; ++case_index)
    {
        char hex[SPARK_SHA256_HEX_BYTES];
        failures += Slr4GenerateCase(outdir, &slr4_cases[case_index], &hex);
        printf("slr4: %-22s %s\n", slr4_cases[case_index].name,
               failures == 0 ? hex : "GENERATION_FAILED");
        if (failures == 0)
            fprintf(manifest, "l4_case_%s.bin %s\n",
                    slr4_cases[case_index].name, hex);
    }
    fclose(manifest);
    printf("slr4: generation %s (%d failures)\n",
           failures == 0 ? "OK" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}

int main(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "selfcheck") == 0)
        return Slr4RunSelfcheck();
    if (argc == 3 && strcmp(argv[1], "gen") == 0)
        return Slr4RunGenerate(argv[2]);
    fprintf(stderr,
            "usage: l4_route_driver gen <outdir> | l4_route_driver selfcheck\n");
    return 2;
}
