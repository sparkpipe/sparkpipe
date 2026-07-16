#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_glm52_dspark_draft_backend.h"

#define SPARK_GLM52_DSPARK_VALIDATION_ANCHOR_TOKEN_ID 10397u
#define SPARK_GLM52_DSPARK_VALIDATION_LANE_COUNT 2u

typedef struct SparkGlm52DsparkValidationMetrics
{
    double maximum_absolute_error;
    double relative_l2_error;
    double cosine_similarity;
} SparkGlm52DsparkValidationMetrics;

static int32_t SparkGlm52DsparkReadFile(
    const char *path,
    void *output,
    uint64_t expected_bytes)
{
    FILE *file;
    uint64_t bytes_read;
    int trailing;

    file = fopen(path, "rb");
    if (file == 0)
        return -1;
    bytes_read = fread(output, 1u, (size_t)expected_bytes, file);
    trailing = fgetc(file);
    fclose(file);
    if (bytes_read != expected_bytes || trailing != EOF)
        return -2;
    return 0;
}

static float SparkGlm52DsparkBf16ToFloat(uint16_t value)
{
    uint32_t bits;
    float output;

    bits = (uint32_t)value << 16u;
    memcpy(&output, &bits, sizeof(output));
    return output;
}

static SparkGlm52DsparkValidationMetrics SparkGlm52DsparkCompareBf16(
    const uint16_t *actual,
    const uint16_t *expected,
    uint64_t element_count)
{
    SparkGlm52DsparkValidationMetrics metrics;
    uint64_t element_index;
    double actual_value,expected_value,error;
    double error_square_sum,expected_square_sum,actual_square_sum,dot_sum;

    memset(&metrics, 0, sizeof(metrics));
    error_square_sum = 0.0;
    expected_square_sum = 0.0;
    actual_square_sum = 0.0;
    dot_sum = 0.0;
    for (element_index=0u; element_index<element_count; ++element_index)
    {
        actual_value = SparkGlm52DsparkBf16ToFloat(actual[element_index]);
        expected_value = SparkGlm52DsparkBf16ToFloat(expected[element_index]);
        error = actual_value - expected_value;
        if (fabs(error) > metrics.maximum_absolute_error)
            metrics.maximum_absolute_error = fabs(error);
        error_square_sum += error * error;
        expected_square_sum += expected_value * expected_value;
        actual_square_sum += actual_value * actual_value;
        dot_sum += actual_value * expected_value;
    }
    metrics.relative_l2_error = sqrt(error_square_sum /
        fmax(expected_square_sum, 1e-30));
    metrics.cosine_similarity = dot_sum /
        sqrt(fmax(actual_square_sum * expected_square_sum, 1e-30));
    return metrics;
}

static int32_t SparkGlm52DsparkLoadOracle(
    const char *oracle_directory,
    uint16_t *taps,
    uint16_t *target_hidden,
    uint16_t *final_hidden,
    uint32_t *tokens,
    float *confidence)
{
    char path[1024];

    snprintf(path, sizeof(path), "%s/taps.bf16", oracle_directory);
    if (SparkGlm52DsparkReadFile(path, taps,
        SPARK_GLM52_DSPARK_DRAFT_BACKEND_FUSED_INPUT_DIMENSION *
            sizeof(uint16_t)) != 0)
        return -1;
    snprintf(path, sizeof(path), "%s/target_hidden.bf16", oracle_directory);
    if (SparkGlm52DsparkReadFile(path, target_hidden,
        SPARK_GLM52_DSPARK_HIDDEN_DIMENSION * sizeof(uint16_t)) != 0)
        return -2;
    snprintf(path, sizeof(path), "%s/final_hidden.bf16", oracle_directory);
    if (SparkGlm52DsparkReadFile(path, final_hidden,
        SPARK_GLM52_DSPARK_BLOCK_SIZE * SPARK_GLM52_DSPARK_HIDDEN_DIMENSION *
            sizeof(uint16_t)) != 0)
        return -3;
    snprintf(path, sizeof(path), "%s/tokens.u32", oracle_directory);
    if (SparkGlm52DsparkReadFile(path, tokens,
        SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT * sizeof(uint32_t)) != 0)
        return -4;
    snprintf(path, sizeof(path), "%s/confidence.f32", oracle_directory);
    if (SparkGlm52DsparkReadFile(path, confidence,
        SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT * sizeof(float)) != 0)
        return -5;
    return 0;
}

static int32_t SparkGlm52DsparkUploadTaps(
    SparkGlm52DsparkDraftBackend *backend,
    const uint16_t *taps,
    uint32_t lane_index)
{
    void *tap_outputs[SPARK_GLM52_DSPARK_AUX_LAYER_COUNT];
    uint64_t lane_stride;
    uint32_t tap_index;

    if (SparkGlm52DsparkDraftBackendTapOutputPointers(
        backend, lane_index, tap_outputs, &lane_stride) != SPARK_STATUS_OK)
        return -1;
    if (lane_stride !=
        (uint64_t)SPARK_GLM52_DSPARK_DRAFT_BACKEND_FUSED_INPUT_DIMENSION *
            sizeof(uint16_t))
        return -2;
    for (tap_index=0u; tap_index<SPARK_GLM52_DSPARK_AUX_LAYER_COUNT;
         ++tap_index)
    {
        if (cudaMemcpy(tap_outputs[tap_index],
            taps + ((uint64_t)tap_index * SPARK_GLM52_DSPARK_HIDDEN_DIMENSION),
            SPARK_GLM52_DSPARK_HIDDEN_DIMENSION * sizeof(uint16_t),
            cudaMemcpyHostToDevice) != cudaSuccess)
            return -3;
    }
    return 0;
}

static int32_t SparkGlm52DsparkValidateOutputs(
    SparkGlm52DsparkDraftBackend *backend,
    const SparkGlm52DsparkDraftResult *result,
    uint32_t lane_index,
    const uint16_t *expected_target,
    const uint16_t *expected_final,
    const uint32_t *expected_tokens,
    const float *expected_confidence)
{
    uint16_t *actual_target,*actual_final;
    SparkGlm52DsparkValidationMetrics target_metrics,final_metrics;
    uint64_t target_offset,final_offset,result_offset;
    uint32_t token_index;
    int32_t failed;

    actual_target = (uint16_t *)malloc(
        SPARK_GLM52_DSPARK_HIDDEN_DIMENSION * sizeof(uint16_t));
    actual_final = (uint16_t *)malloc(
        SPARK_GLM52_DSPARK_BLOCK_SIZE * SPARK_GLM52_DSPARK_HIDDEN_DIMENSION *
            sizeof(uint16_t));
    if (actual_target == 0 || actual_final == 0)
        return -1;
    target_offset = (uint64_t)lane_index *
        SPARK_GLM52_DSPARK_HIDDEN_DIMENSION;
    final_offset = (uint64_t)lane_index *
        SPARK_GLM52_DSPARK_BLOCK_SIZE *
        SPARK_GLM52_DSPARK_HIDDEN_DIMENSION;
    result_offset = (uint64_t)lane_index *
        SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT;
    if (cudaMemcpy(actual_target,
            backend->device_target_hidden_bf16 + target_offset,
            SPARK_GLM52_DSPARK_HIDDEN_DIMENSION * sizeof(uint16_t),
            cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(actual_final,
            backend->device_block_final_bf16 + final_offset,
            SPARK_GLM52_DSPARK_BLOCK_SIZE * SPARK_GLM52_DSPARK_HIDDEN_DIMENSION *
                sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess)
        return -2;
    target_metrics = SparkGlm52DsparkCompareBf16(
        actual_target, expected_target, SPARK_GLM52_DSPARK_HIDDEN_DIMENSION);
    final_metrics = SparkGlm52DsparkCompareBf16(
        actual_final, expected_final,
        SPARK_GLM52_DSPARK_BLOCK_SIZE * SPARK_GLM52_DSPARK_HIDDEN_DIMENSION);
    fprintf(stdout,
        "dspark_target lane=%u max_abs=%.9f rel_l2=%.9f cosine=%.9f\n",
        lane_index,
        target_metrics.maximum_absolute_error,
        target_metrics.relative_l2_error,
        target_metrics.cosine_similarity);
    fprintf(stdout,
        "dspark_final lane=%u max_abs=%.9f rel_l2=%.9f cosine=%.9f\n",
        lane_index,
        final_metrics.maximum_absolute_error,
        final_metrics.relative_l2_error,
        final_metrics.cosine_similarity);
    failed = target_metrics.relative_l2_error > 0.02 ||
        target_metrics.cosine_similarity < 0.999 ||
        final_metrics.relative_l2_error > 0.08 ||
        final_metrics.cosine_similarity < 0.995;
    for (token_index=0u;
         token_index<SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT;
         ++token_index)
    {
        fprintf(stdout,
            "dspark_token lane=%u index=%u actual=%u expected=%u confidence=%.6f expected_confidence=%.6f\n",
            lane_index, token_index, result->token_ids[token_index],
            expected_tokens[token_index],
            backend->host_confidence_f32[result_offset + token_index],
            expected_confidence[token_index]);
        if (result->token_ids[token_index] != expected_tokens[token_index] ||
            fabsf(backend->host_confidence_f32[result_offset + token_index] -
                expected_confidence[token_index]) > 0.03f)
            failed = 1;
    }
    free(actual_target);
    free(actual_final);
    return failed == 0 ? 0 : -3;
}

int main(int argc,char **argv)
{
    SparkGlm52DsparkDraftBackend backend;
    SparkGlm52DsparkDraftBackendConfiguration configuration;
    SparkGlm52DsparkDraftBackendStage
        stages[SPARK_GLM52_DSPARK_VALIDATION_LANE_COUNT];
    SparkGlm52DsparkDraftRequest
        requests[SPARK_GLM52_DSPARK_VALIDATION_LANE_COUNT];
    SparkGlm52DsparkDraftResult
        results[SPARK_GLM52_DSPARK_VALIDATION_LANE_COUNT];
    uint16_t *taps,*expected_target,*expected_final;
    uint32_t expected_tokens[SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT];
    uint32_t lane_index,result_count;
    float expected_confidence[SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT];
    cudaEvent_t start_event,stop_event;
    float stage_ms,draft_ms;
    int32_t validation_status;

    if (argc != 5)
        return 2;
    taps = (uint16_t *)malloc(
        SPARK_GLM52_DSPARK_DRAFT_BACKEND_FUSED_INPUT_DIMENSION * sizeof(uint16_t));
    expected_target = (uint16_t *)malloc(
        SPARK_GLM52_DSPARK_HIDDEN_DIMENSION * sizeof(uint16_t));
    expected_final = (uint16_t *)malloc(
        SPARK_GLM52_DSPARK_BLOCK_SIZE * SPARK_GLM52_DSPARK_HIDDEN_DIMENSION *
            sizeof(uint16_t));
    if (taps == 0 || expected_target == 0 || expected_final == 0)
        return 3;
    if (SparkGlm52DsparkLoadOracle(argv[4], taps, expected_target,
        expected_final, expected_tokens, expected_confidence) != 0)
        return 4;
    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_GLM52_DSPARK_DRAFT_BACKEND_ABI_VERSION;
    configuration.descriptor_bytes =
        SPARK_GLM52_DSPARK_DRAFT_BACKEND_CONFIGURATION_DESCRIPTOR_BYTES;
    configuration.maximum_lane_count = SPARK_GLM52_DSPARK_VALIDATION_LANE_COUNT;
    configuration.maximum_context_token_count = 16u;
    configuration.manifest_path = argv[1];
    configuration.config_path = argv[2];
    configuration.safetensors_path = argv[3];
    if (SparkGlm52DsparkDraftBackendInitialize(
        &backend, &configuration) != SPARK_STATUS_OK)
        return 5;
    for (lane_index=0u;
         lane_index<SPARK_GLM52_DSPARK_VALIDATION_LANE_COUNT;
         ++lane_index)
    {
        if (SparkGlm52DsparkUploadTaps(&backend, taps, lane_index) != 0)
            return 6;
        memset(&stages[lane_index], 0, sizeof(stages[lane_index]));
        stages[lane_index].sequence_id = lane_index + 1u;
        stages[lane_index].sequence_position = 1u;
        stages[lane_index].tap_generation = 1u;
        stages[lane_index].tap_row_index = lane_index;
        stages[lane_index].backend_lane_index = lane_index;
        stages[lane_index].token_id =
            SPARK_GLM52_DSPARK_VALIDATION_ANCHOR_TOKEN_ID;
    }
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, (cudaStream_t)backend.cuda_stream);
    if (SparkGlm52DsparkDraftBackendStageBatch(
        &backend, stages,
        SPARK_GLM52_DSPARK_VALIDATION_LANE_COUNT) != SPARK_STATUS_OK)
        return 7;
    cudaEventRecord(stop_event, (cudaStream_t)backend.cuda_stream);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&stage_ms, start_event, stop_event);
    result_count = UINT32_MAX;
    if (SparkGlm52DsparkDraftBackendTakeBatchResults(
        &backend, results, SPARK_GLM52_DSPARK_VALIDATION_LANE_COUNT,
        &result_count) != SPARK_STATUS_OK ||
        result_count != 0u)
        return 7;
    for (lane_index=0u;
         lane_index<SPARK_GLM52_DSPARK_VALIDATION_LANE_COUNT;
         ++lane_index)
    {
        memset(&requests[lane_index], 0, sizeof(requests[lane_index]));
        requests[lane_index].abi_version = SPARK_GLM52_DSPARK_ABI_VERSION;
        requests[lane_index].descriptor_bytes =
            SPARK_GLM52_DSPARK_DRAFT_REQUEST_DESCRIPTOR_BYTES;
        requests[lane_index].requested_token_count =
            SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT;
        requests[lane_index].active_sequence_index = lane_index;
        requests[lane_index].request_id = lane_index + 1u;
        requests[lane_index].sequence_id = lane_index + 1u;
        requests[lane_index].sequence_position = 1u;
        requests[lane_index].tap_generation = 1u;
    }
    cudaEventRecord(start_event, (cudaStream_t)backend.cuda_stream);
    if (SparkGlm52DsparkDraftBackendLaunchDraftBatch(
        &backend, requests,
        SPARK_GLM52_DSPARK_VALIDATION_LANE_COUNT) != SPARK_STATUS_OK)
        return 8;
    cudaEventRecord(stop_event, (cudaStream_t)backend.cuda_stream);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&draft_ms, start_event, stop_event);
    result_count = 0u;
    if (SparkGlm52DsparkDraftBackendTakeBatchResults(
        &backend, results, SPARK_GLM52_DSPARK_VALIDATION_LANE_COUNT,
        &result_count) != SPARK_STATUS_OK ||
        result_count != SPARK_GLM52_DSPARK_VALIDATION_LANE_COUNT)
        return 8;
    validation_status = 0;
    for (lane_index=0u;
         lane_index<SPARK_GLM52_DSPARK_VALIDATION_LANE_COUNT;
         ++lane_index)
    {
        if (SparkGlm52DsparkValidateOutputs(
            &backend, &results[lane_index], lane_index,
            expected_target, expected_final,
            expected_tokens, expected_confidence) != 0)
            validation_status = -1;
    }
    fprintf(stdout, "dspark_timing stage_ms=%.6f draft_ms=%.6f\n",
        stage_ms, draft_ms);
    SparkGlm52DsparkDraftBackendTeardown(&backend);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    free(taps);
    free(expected_target);
    free(expected_final);
    return validation_status == 0 ? 0 : 9;
}
