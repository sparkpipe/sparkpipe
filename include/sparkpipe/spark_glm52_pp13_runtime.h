#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm52_stage_plan.h"
#include "sparkpipe/spark_hidden_transport.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_PP13_RUNTIME_ABI_VERSION 1u
#define SPARK_GLM52_PP13_RUNTIME_RANK_PLAN_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52Pp13RuntimeRankPlan))
#define SPARK_GLM52_PP13_RUNTIME_STAGE_COUNT 13u
#define SPARK_GLM52_PP13_RUNTIME_LAYERS_PER_STAGE 6u
#define SPARK_GLM52_PP13_RUNTIME_HOST_NAME_BYTES 16u
#define SPARK_GLM52_PP13_RUNTIME_ROUTE_NAME_BYTES 64u
#define SPARK_GLM52_PP13_RUNTIME_PACK_PATH_BYTES 512u
#define SPARK_GLM52_PP13_RUNTIME_DEFAULT_PORT_BASE 52100u
#define SPARK_GLM52_PP13_RUNTIME_FINAL_EVENT_PORT_OFFSET 200u
#define SPARK_GLM52_PP13_RUNTIME_FINAL_EVENT_ROUTE_NAME_BYTES 64u
#define SPARK_GLM52_PP13_RUNTIME_HIDDEN_DIMENSION 6144u
#define SPARK_GLM52_PP13_RUNTIME_QUANTIZATION_MODE \
    SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT
#define SPARK_GLM52_PP13_RUNTIME_FP8_PACK_MANIFEST \
    "fp8_moe_pack_manifest.json"
#define SPARK_GLM52_PP13_RUNTIME_BF16_HIDDEN_BYTES_PER_SEQUENCE \
    (SPARK_GLM52_PP13_RUNTIME_HIDDEN_DIMENSION * \
     SPARK_HIDDEN_TRANSPORT_BF16_BYTES_PER_ELEMENT)

#define SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_PREVIOUS 0x00000001u
#define SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_NEXT 0x00000002u
#define SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_FINAL_STAGE 0x00000004u
#define SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_DENSE_PREFIX 0x00000008u
#define SPARK_GLM52_PP13_RUNTIME_RANK_KNOWN_FLAGS \
    (SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_PREVIOUS | \
     SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_HAS_NEXT | \
     SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_FINAL_STAGE | \
     SPARK_GLM52_PP13_RUNTIME_RANK_FLAG_DENSE_PREFIX)

typedef struct SparkGlm52Pp13RuntimeRankPlan
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t rank_index;
    uint32_t flags;
    uint32_t first_layer_index;
    uint32_t layer_count;
    uint32_t previous_rank_index;
    uint32_t next_rank_index;
    uint32_t listen_port;
    uint32_t next_port;
    uint32_t max_active_sequence_count;
    uint32_t hidden_dimension;
    uint32_t bytes_per_sequence;
    uint32_t quantization_mode;
    uint64_t max_packet_bytes;
    char host_name[SPARK_GLM52_PP13_RUNTIME_HOST_NAME_BYTES];
    char previous_host_name[SPARK_GLM52_PP13_RUNTIME_HOST_NAME_BYTES];
    char next_host_name[SPARK_GLM52_PP13_RUNTIME_HOST_NAME_BYTES];
    char input_route_name[SPARK_GLM52_PP13_RUNTIME_ROUTE_NAME_BYTES];
    char output_route_name[SPARK_GLM52_PP13_RUNTIME_ROUTE_NAME_BYTES];
    SparkHiddenTransportEndpoint input_endpoint;
    SparkHiddenTransportEndpoint output_endpoint;
} SparkGlm52Pp13RuntimeRankPlan;

typedef struct SparkGlm52Pp13RuntimeFinalEventRoute
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t source_rank_index;
    uint32_t sink_rank_index;
    uint32_t listen_port;
    uint32_t connect_port;
    char source_host_name[SPARK_GLM52_PP13_RUNTIME_HOST_NAME_BYTES];
    char sink_host_name[SPARK_GLM52_PP13_RUNTIME_HOST_NAME_BYTES];
    char route_name[SPARK_GLM52_PP13_RUNTIME_FINAL_EVENT_ROUTE_NAME_BYTES];
} SparkGlm52Pp13RuntimeFinalEventRoute;

#define SPARK_GLM52_PP13_RUNTIME_FINAL_EVENT_ROUTE_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52Pp13RuntimeFinalEventRoute))

SparkStatus SparkGlm52Pp13RuntimeBuildFixedStagePlan(
    SparkGlm52StagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes);

SparkStatus SparkGlm52Pp13RuntimeRankHostName(
    uint32_t rank_index,
    char *host_name,
    uint32_t host_name_bytes);

SparkStatus SparkGlm52Pp13RuntimeBuildRankPlan(
    uint32_t rank_index,
    uint32_t max_active_sequence_count,
    uint32_t port_base,
    SparkGlm52Pp13RuntimeRankPlan *rank_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes);

SparkStatus SparkGlm52Pp13RuntimeValidateRankPlan(
    const SparkGlm52Pp13RuntimeRankPlan *rank_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes);

SparkStatus SparkGlm52Pp13RuntimeBuildFp8PackPath(
    const char *pack_root,
    uint32_t layer_index,
    char *pack_path,
    uint32_t pack_path_bytes);

SparkStatus SparkGlm52Pp13RuntimeValidateStageFp8PackFiles(
    const SparkGlm52Pp13RuntimeRankPlan *rank_plan,
    const char *pack_root,
    char *error_buffer,
    uint32_t error_buffer_bytes);

SparkStatus SparkGlm52Pp13RuntimeBuildFinalEventRoute(
    uint32_t port_base,
    SparkGlm52Pp13RuntimeFinalEventRoute *route,
    char *error_buffer,
    uint32_t error_buffer_bytes);

SparkStatus SparkGlm52Pp13RuntimeValidateFinalEventRoute(
    const SparkGlm52Pp13RuntimeFinalEventRoute *route,
    char *error_buffer,
    uint32_t error_buffer_bytes);

#ifdef __cplusplus
}
#endif
