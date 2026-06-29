#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm52_sm121_flashinfer_b12x_moe.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_SM121_B12X_GENERATED_KERNEL_TABLE_ABI_VERSION 1u

#define SPARK_GLM52_SM121_B12X_BACKEND_KIND_MICRO 1u
#define SPARK_GLM52_SM121_B12X_BACKEND_KIND_STATIC 2u
#define SPARK_GLM52_SM121_B12X_BACKEND_KIND_DYNAMIC 3u

typedef struct SparkGlm52Sm121B12xGeneratedKernelBucket
{
    uint32_t abi_version;
    uint32_t token_upper_bound;
    uint32_t backend_kind;
    uint32_t routed_rows_capacity;
    uint32_t max_rows;
    uint32_t physical_tile_capacity;
    uint32_t task_capacity;
    uint32_t max_active_clusters;
    uint32_t static_mma_tile_m;
    uint32_t static_mma_tile_n;
    uint32_t reserved0;
    uint32_t reserved1;
    uint64_t qualified_average_microseconds;
    uint64_t qualified_p95_microseconds;
} SparkGlm52Sm121B12xGeneratedKernelBucket;

typedef struct SparkGlm52Sm121B12xGeneratedWorkspace
{
    uint32_t abi_version;
    uint32_t backend_kind;
    uint32_t routed_rows_capacity;
    uint32_t max_rows;
    uint32_t physical_tile_capacity;
    uint32_t task_capacity;
    void *row_counts_i32;
    void *token_map_i32;
    void *token_weights_fp32;
    void *packed_input_u8;
    void *packed_input_scale_u8;
    void *barrier_count_i32;
    void *barrier_epoch_i32;
    void *active_expert_count_i32;
    void *weight_expert_ids_i32;
    void *global_to_local_expert_i32;
    void *compact_topk_ids_i32;
    void *expert_write_rows_i32;
    void *expert_tile_base_i32;
    void *pair_head_i32;
    void *producers_done_count_i32;
    void *all_work_published_i32;
    void *task_head_i32;
    void *task_tail_i32;
    void *task_ready_i32;
    void *task_expert_i32;
    void *task_m_tile_i32;
    void *task_slice_begin_i32;
    void *task_slice_count_i32;
    void *task_valid_rows_i32;
    void *tile_write_count_i32;
} SparkGlm52Sm121B12xGeneratedWorkspace;

typedef struct SparkGlm52Sm121B12xGeneratedLaunchArguments
{
    uint32_t abi_version;
    uint32_t token_count;
    uint32_t maximum_token_count;
    uint32_t expert_count;
    uint32_t top_k;
    uint32_t hidden_dimension;
    uint32_t intermediate_dimension;
    uint32_t reserved0;
    const void *hidden_bf16;
    const int32_t *topk_ids_i32;
    const float *topk_weights_fp32;
    const void *w1_weight_fp4_static_view;
    const void *w1_scale_static_storage_ue4m3;
    const float *w1_alpha_fp32_by_expert;
    const float *fc2_input_scale_fp32_by_expert;
    const void *w2_weight_fp4_static_view;
    const void *w2_scale_static_storage_ue4m3;
    const float *w2_alpha_fp32_by_expert;
    void *output_bf16;
    void *user_workspace;
    uint64_t user_workspace_bytes;
    SparkGlm52Sm121B12xGeneratedWorkspace *generated_workspace;
    void *cuda_stream;
} SparkGlm52Sm121B12xGeneratedLaunchArguments;

typedef struct SparkGlm52Sm121B12xGeneratedManifest
{
    uint32_t abi_version;
    uint32_t bucket_count;
    uint32_t hidden_dimension;
    uint32_t intermediate_dimension;
    uint32_t expert_count;
    uint32_t top_k;
    uint32_t maximum_token_count;
    uint32_t cuda_architecture;
    uint64_t manifest_hash_low64;
    const SparkGlm52Sm121B12xGeneratedKernelBucket *buckets;
} SparkGlm52Sm121B12xGeneratedManifest;

extern const SparkGlm52Sm121B12xGeneratedManifest
    SparkGlm52Sm121B12xGeneratedManifestInstance;

SparkStatus SparkGlm52Sm121B12xGeneratedLaunch(
    const SparkGlm52Sm121B12xGeneratedKernelBucket *bucket,
    const SparkGlm52Sm121B12xGeneratedLaunchArguments *arguments);

#ifdef __cplusplus
}
#endif
