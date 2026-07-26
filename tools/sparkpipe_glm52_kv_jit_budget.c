#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_glm52_kv_cache.h"
#include "sparkpipe/spark_glm52_mtp_tree.h"
#include "sparkpipe/spark_glm52_pp13_node_context_builder.h"
#include "sparkpipe/spark_glm52_pp13_runtime.h"

typedef struct SparkGlm52KvJitBudgetToolConfiguration
{
    uint32_t active_sequence_count;
    uint32_t backing_request_count;
    uint32_t average_context_token_count;
    uint32_t generation_headroom_token_count;
    uint32_t generation_headroom_is_set;
    uint32_t physical_pool_token_capacity;
    uint32_t backing_block_capacity;
    uint32_t mtp_enabled;
} SparkGlm52KvJitBudgetToolConfiguration;

static int SparkGlm52KvJitBudgetToolParseU32(
    const char *text,
    uint32_t *value_out)
{
    char *end;
    unsigned long long value;

    if (text == 0 || text[0] == '\0' || value_out == 0)
        return -1;
    end = 0;
    value = strtoull(text,&end,10);
    if (end == 0 || end[0] != '\0' || value == 0u || value > UINT32_MAX)
        return -1;
    *value_out = (uint32_t)value;
    return 0;
}

static int SparkGlm52KvJitBudgetToolApplyArgument(
    SparkGlm52KvJitBudgetToolConfiguration *configuration,
    int argc,
    char **argv,
    int *index)
{
    uint32_t value;

    if (strcmp(argv[*index],"--no-mtp") == 0)
    {
        configuration->mtp_enabled = 0u;
        return 0;
    }
    if (*index + 1 >= argc ||
        SparkGlm52KvJitBudgetToolParseU32(argv[*index + 1],&value) != 0)
        return -1;
    if (strcmp(argv[*index],"--active") == 0)
        configuration->active_sequence_count = value;
    else if (strcmp(argv[*index],"--requests") == 0)
        configuration->backing_request_count = value;
    else if (strcmp(argv[*index],"--average-context") == 0)
        configuration->average_context_token_count = value;
    else if (strcmp(argv[*index],"--generation-headroom") == 0)
    {
        configuration->generation_headroom_token_count = value;
        configuration->generation_headroom_is_set = 1u;
    }
    else if (strcmp(argv[*index],"--kv-pool-tokens") == 0)
        configuration->physical_pool_token_capacity = value;
    else if (strcmp(argv[*index],"--nvme-blocks") == 0)
        configuration->backing_block_capacity = value;
    else
        return -1;
    *index += 1;
    return 0;
}

static double SparkGlm52KvJitBudgetToolGiB(uint64_t bytes)
{
    return (double)bytes / (double)UINT64_C(1073741824);
}

static int SparkGlm52KvJitBudgetToolPopulateKvLayout(
    uint32_t quantization_mode,
    SparkGlm52KvJitStageBudgetRequest *request)
{
    if (request == 0)
        return -1;
    if (quantization_mode ==
        SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT)
    {
        request->attention_cache_layout =
            SPARK_GLM52_KV_CACHE_LAYOUT_FULL_KEY_VALUE_FP8_E4M3;
        request->fp8_scale_block_size = SPARK_GLM52_MODEL_FP8_SCALE_BLOCK;
        return 0;
    }
    return -1;
}

int main(int argc,char **argv)
{
    SparkGlm52KvJitBudgetToolConfiguration configuration;
    SparkGlm52KvJitStageBudgetRequest request;
    SparkGlm52KvJitStageBudget budget;
    SparkGlm52StagePlan stage_plan;
    char error_buffer[256u];
    uint64_t blocks_per_request;
    uint64_t physical_block_capacity;
    uint64_t reserved_context_token_count;
    uint64_t canonical_active_blocks;
    uint64_t persistent_backing_blocks;
    uint64_t mtp_transient_blocks;
    uint64_t mtp_shadow_token_records;
    uint64_t required_active_blocks;
    uint64_t all_requests_gpu_blocks;
    uint64_t storage_token_capacity;
    uint64_t aggregate_nvme_capacity_bytes;
    uint64_t aggregate_nvme_required_bytes;
    uint64_t aggregate_mtp_shadow_bytes;
    uint64_t maximum_active_sequence_count;
    uint64_t request_cohort_count;
    uint64_t resident_cohort_capacity;
    uint32_t rank_index;
    int index;

    memset(&configuration,0,sizeof(configuration));
    configuration.active_sequence_count = 1024u;
    configuration.backing_request_count =
        SPARK_GLM52_STAGE_PLAN_PIPELINE_INFLIGHT_REQUEST_CAPACITY;
    configuration.average_context_token_count = 4096u;
    configuration.physical_pool_token_capacity = SPARK_GLM52_KV_POOL_TOKENS;
    configuration.backing_block_capacity =
        SPARK_GLM52_PP13_NODE_CONTEXT_BUILDER_DEFAULT_NVME_BLOCK_CAPACITY;
    configuration.mtp_enabled = 1u;
    for (index = 1; index < argc; ++index)
    {
        if (SparkGlm52KvJitBudgetToolApplyArgument(
                &configuration,argc,argv,&index) != 0)
        {
            fprintf(stderr,
                "usage: %s [--active n] [--requests n] "
                "[--average-context n] [--generation-headroom n] "
                "[--kv-pool-tokens n] "
                "[--nvme-blocks n] [--no-mtp]\n",argv[0]);
            return 2;
        }
    }
    if (configuration.generation_headroom_is_set == 0u)
    {
        configuration.generation_headroom_token_count =
            configuration.mtp_enabled != 0u
                ? SPARK_GLM52_MODEL_MTP_DRAFT_TOKEN_COUNT + 1u
                : 1u;
    }
    if (configuration.average_context_token_count >
            UINT32_MAX - configuration.generation_headroom_token_count ||
        configuration.physical_pool_token_capacity %
            SPARK_GLM52_KV_BLOCK_TOKENS != 0u ||
        configuration.active_sequence_count >
            SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET)
    {
        fprintf(stderr,"invalid KV budget configuration\n");
        return 2;
    }
    memset(error_buffer,0,sizeof(error_buffer));
    if (SparkGlm52Pp13RuntimeBuildFixedStagePlan(
            &stage_plan,error_buffer,sizeof(error_buffer)) != SPARK_STATUS_OK)
    {
        fprintf(stderr,"stage plan failed: %s\n",error_buffer);
        return 1;
    }
    reserved_context_token_count =
        (uint64_t)configuration.average_context_token_count +
        configuration.generation_headroom_token_count;
    blocks_per_request =
        (reserved_context_token_count +
         SPARK_GLM52_KV_BLOCK_TOKENS - 1u) /
        SPARK_GLM52_KV_BLOCK_TOKENS;
    canonical_active_blocks = blocks_per_request *
        configuration.active_sequence_count;
    persistent_backing_blocks = blocks_per_request *
        configuration.backing_request_count;
    mtp_transient_blocks = configuration.mtp_enabled != 0u
        ? (uint64_t)SPARK_GLM52_MODEL_MTP_TREE_TRANSIENT_BLOCK_COUNT *
            configuration.active_sequence_count
        : 0u;
    mtp_shadow_token_records = configuration.mtp_enabled != 0u
        ? (uint64_t)SPARK_GLM52_MODEL_MTP_TREE_SHADOW_TOKEN_COUNT *
            configuration.backing_request_count
        : 0u;
    if (mtp_shadow_token_records > UINT32_MAX ||
        (uint64_t)configuration.physical_pool_token_capacity >
            UINT32_MAX - mtp_shadow_token_records)
    {
        fprintf(stderr,"KV storage token capacity overflow\n");
        return 2;
    }
    required_active_blocks = canonical_active_blocks + mtp_transient_blocks;
    all_requests_gpu_blocks = persistent_backing_blocks + mtp_transient_blocks;
    storage_token_capacity =
        configuration.physical_pool_token_capacity + mtp_shadow_token_records;
    physical_block_capacity = configuration.physical_pool_token_capacity /
        SPARK_GLM52_KV_BLOCK_TOKENS;
    maximum_active_sequence_count = physical_block_capacity /
        (blocks_per_request +
         (configuration.mtp_enabled != 0u
            ? SPARK_GLM52_MODEL_MTP_TREE_TRANSIENT_BLOCK_COUNT : 0u));
    request_cohort_count =
        ((uint64_t)configuration.backing_request_count +
         configuration.active_sequence_count - 1u) /
        configuration.active_sequence_count;
    resident_cohort_capacity = required_active_blocks != 0u
        ? physical_block_capacity / required_active_blocks : 0u;
    printf(
        "active=%u requests=%u average_context=%u generation_headroom=%u "
        "reserved_context=%" PRIu64 " pool_tokens=%u storage_tokens=%" PRIu64 " "
        "nvme_blocks=%u canonical_active_blocks=%" PRIu64 " "
        "mtp_transient_blocks=%" PRIu64 " required_active_blocks=%" PRIu64 " "
        "persistent_backing_blocks=%" PRIu64 " mtp_persistent_blocks=0 "
        "mtp_shadow_records=%" PRIu64 " maximum_resident=%" PRIu64 " "
        "request_cohorts=%" PRIu64 " resident_cohorts=%" PRIu64 " "
        "gpu_fit=%s all_requests_gpu_fit=%s nvme_fit=%s\n",
        configuration.active_sequence_count,
        configuration.backing_request_count,
        configuration.average_context_token_count,
        configuration.generation_headroom_token_count,
        reserved_context_token_count,
        configuration.physical_pool_token_capacity,
        storage_token_capacity,
        configuration.backing_block_capacity,
        canonical_active_blocks,
        mtp_transient_blocks,
        required_active_blocks,
        persistent_backing_blocks,
        mtp_shadow_token_records,
        maximum_active_sequence_count,
        request_cohort_count,
        resident_cohort_capacity,
        required_active_blocks <=
            physical_block_capacity ? "yes" : "no",
        all_requests_gpu_blocks <= physical_block_capacity ? "yes" : "no",
        persistent_backing_blocks <= configuration.backing_block_capacity
            ? "yes" : "no");
    if (required_active_blocks ==
        configuration.physical_pool_token_capacity /
            SPARK_GLM52_KV_BLOCK_TOKENS)
    {
        printf("warning=active_gpu_pool_has_zero_block_headroom\n");
    }
    printf("rank layers dsa mtp physical_gpu_pool_GiB mtp_shadow_GiB "
        "gpu_storage_GiB record_KiB nvme_capacity_GiB nvme_required_GiB "
        "compact_selected_mla_GiB\n");
    aggregate_nvme_capacity_bytes = 0u;
    aggregate_nvme_required_bytes = 0u;
    aggregate_mtp_shadow_bytes = 0u;
    for (rank_index = 0u; rank_index < stage_plan.stage_count; ++rank_index)
    {
        uint64_t required_nvme_bytes;
        uint64_t mtp_shadow_bytes;
        uint64_t gpu_storage_bytes;
        const SparkGlm52StagePlanStage *stage;

        stage = &stage_plan.stages[rank_index];
        memset(&request,0,sizeof(request));
        request.abi_version = SPARK_GLM52_KV_JIT_STAGE_BUDGET_ABI_VERSION;
        request.descriptor_bytes =
            SPARK_GLM52_KV_JIT_STAGE_BUDGET_REQUEST_DESCRIPTOR_BYTES;
        request.first_layer_index = stage->first_layer_index;
        request.layer_count = stage->layer_count;
        request.physical_pool_token_capacity =
            configuration.physical_pool_token_capacity;
        request.backing_block_capacity = configuration.backing_block_capacity;
        request.active_sequence_count = configuration.active_sequence_count;
        request.backing_request_count = configuration.backing_request_count;
        request.selected_token_count = SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT;
        request.include_mtp_layer = configuration.mtp_enabled != 0u &&
            rank_index + 1u == stage_plan.stage_count ? 1u : 0u;
        request.block_token_count = SPARK_GLM52_KV_BLOCK_TOKENS;
        request.record_alignment_bytes =
            SPARK_GLM52_KV_JIT_DEFAULT_RECORD_ALIGNMENT;
        if (SparkGlm52KvJitBudgetToolPopulateKvLayout(
                SPARK_GLM52_PP13_RUNTIME_DEFAULT_QUANTIZATION_MODE,
                &request) != 0)
        {
            fprintf(stderr,"rank %u KV layout is unsupported\n",rank_index);
            return 1;
        }
        if (SparkGlm52KvCacheCalculateJitStageBudget(
                &request,&budget) != SPARK_STATUS_OK ||
            persistent_backing_blocks > UINT64_MAX / budget.nvme_record_bytes ||
            (budget.resident_bytes_per_token != 0u &&
             mtp_shadow_token_records >
                UINT64_MAX / budget.resident_bytes_per_token))
        {
            fprintf(stderr,"rank %u budget failed\n",rank_index);
            return 1;
        }
        required_nvme_bytes =
            persistent_backing_blocks * budget.nvme_record_bytes;
        mtp_shadow_bytes =
            mtp_shadow_token_records * budget.resident_bytes_per_token;
        if (budget.resident_pool_bytes > UINT64_MAX - mtp_shadow_bytes)
        {
            fprintf(stderr,"rank %u GPU storage budget overflow\n",rank_index);
            return 1;
        }
        gpu_storage_bytes = budget.resident_pool_bytes + mtp_shadow_bytes;
        if (aggregate_nvme_capacity_bytes >
                UINT64_MAX - budget.nvme_capacity_bytes ||
            aggregate_nvme_required_bytes >
                UINT64_MAX - required_nvme_bytes ||
            aggregate_mtp_shadow_bytes > UINT64_MAX - mtp_shadow_bytes)
        {
            fprintf(stderr,"aggregate NVMe budget overflow\n");
            return 1;
        }
        aggregate_nvme_capacity_bytes += budget.nvme_capacity_bytes;
        aggregate_nvme_required_bytes += required_nvme_bytes;
        aggregate_mtp_shadow_bytes += mtp_shadow_bytes;
        printf("%u %u:%u %u %u %.3f %.3f %.3f %.0f %.3f %.3f %.3f\n",
            rank_index,
            stage->first_layer_index,
            stage->layer_count,
            budget.local_dsa_index_layer_count,
            budget.include_mtp_layer,
            SparkGlm52KvJitBudgetToolGiB(budget.resident_pool_bytes),
            SparkGlm52KvJitBudgetToolGiB(mtp_shadow_bytes),
            SparkGlm52KvJitBudgetToolGiB(gpu_storage_bytes),
            (double)budget.nvme_record_bytes / 1024.0,
            SparkGlm52KvJitBudgetToolGiB(budget.nvme_capacity_bytes),
            SparkGlm52KvJitBudgetToolGiB(required_nvme_bytes),
            SparkGlm52KvJitBudgetToolGiB(
                budget.compact_selected_mla_working_set_bytes));
    }
    printf("aggregate_nvme_capacity_GiB=%.3f "
        "aggregate_nvme_required_GiB=%.3f "
        "aggregate_mtp_shadow_GiB=%.3f\n",
        SparkGlm52KvJitBudgetToolGiB(aggregate_nvme_capacity_bytes),
        SparkGlm52KvJitBudgetToolGiB(aggregate_nvme_required_bytes),
        SparkGlm52KvJitBudgetToolGiB(aggregate_mtp_shadow_bytes));
    return required_active_blocks <=
            physical_block_capacity &&
        persistent_backing_blocks <= configuration.backing_block_capacity
        ? 0 : 1;
}
