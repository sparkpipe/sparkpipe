#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm52_model.h"
#include "sparkpipe/spark_glm52_production_topology.h"
#include "sparkpipe/spark_glm52_stage_plan.h"
#include "sparkpipe/spark_hidden_transport.h"
#include "sparkpipe/spark_glm52_shape_config.h"
#include "sparkpipe/spark_tp_collective.h"
#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_RING_RUNTIME_ABI_VERSION 5u
#define SPARK_GLM52_RING_RUNTIME_RANK_PLAN_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52RingRuntimeRankPlan))
#define SPARK_GLM52_RING_RUNTIME_STAGE_COUNT \
    SPARK_GLM52_STAGE_PLAN_CURRENT_SPARK_COUNT
#define SPARK_GLM52_RING_RUNTIME_HOST_NAME_BYTES 16u
#define SPARK_GLM52_RING_RUNTIME_ROUTE_NAME_BYTES 64u
#define SPARK_GLM52_RING_RUNTIME_PACK_PATH_BYTES 512u
#define SPARK_GLM52_RING_RUNTIME_DEFAULT_PORT_BASE 52100u
#define SPARK_GLM52_RING_RUNTIME_FINAL_EVENT_PORT_OFFSET 200u
#define SPARK_GLM52_RING_RUNTIME_FINAL_EVENT_ROUTE_NAME_BYTES 64u
#define SPARK_GLM52_RING_RUNTIME_FINAL_EVENT_MAGIC 0x35454650u
#define SPARK_GLM52_RING_RUNTIME_HIDDEN_DIMENSION \
    SPARK_GLM52_MODEL_HIDDEN_DIMENSION
#define SPARK_GLM52_RING_RUNTIME_DEFAULT_QUANTIZATION_MODE \
    SPARK_GLM52_STAGE_PLAN_QUANTIZATION_FP8_E4M3_8BIT
#define SPARK_GLM52_RING_RUNTIME_FP8_PACK_MANIFEST \
    "fp8_moe_pack_manifest.json"
#define SPARK_GLM52_RING_RUNTIME_W8LUT_PACK_MANIFEST \
    "w8lut_moe_pack_manifest.json"
#define SPARK_GLM52_RING_RUNTIME_B12X_PACK_MANIFEST \
    "resident_moe_pack_manifest.json"
#define SPARK_GLM52_RING_RUNTIME_MOE_BACKEND_NONE 0u
#define SPARK_GLM52_RING_RUNTIME_MOE_BACKEND_FP8_FLASHINFER_GROUPED 1u
#define SPARK_GLM52_RING_RUNTIME_MOE_BACKEND_W8LUT_BF16_WMMA 2u
#define SPARK_GLM52_RING_RUNTIME_MOE_BACKEND_NVFP4_B12X 3u
#define SPARK_GLM52_RING_RUNTIME_BF16_HIDDEN_BYTES_PER_SEQUENCE \
    SPARK_GLM52_MODEL_HIDDEN_BF16_BYTES
#define SPARK_GLM52_RING_RUNTIME_INDEXSHARE_SIDEBAND_BYTES_PER_SEQUENCE \
    SPARK_GLM52_MODEL_DSA_SELECTED_INDEX_BYTES
#define SPARK_GLM52_RING_RUNTIME_DSPARK_TAP_SIDEBAND_BYTES_PER_SEQUENCE \
    (SPARK_GLM52_DSPARK_AUX_LAYER_COUNT * \
     SPARK_GLM52_RING_RUNTIME_BF16_HIDDEN_BYTES_PER_SEQUENCE)
#define SPARK_GLM52_RING_RUNTIME_MAX_SIDEBAND_BYTES_PER_SEQUENCE \
    (SPARK_GLM52_RING_RUNTIME_INDEXSHARE_SIDEBAND_BYTES_PER_SEQUENCE + \
     SPARK_GLM52_RING_RUNTIME_DSPARK_TAP_SIDEBAND_BYTES_PER_SEQUENCE)
#define SPARK_GLM52_RING_RUNTIME_MAX_TRANSPORT_BYTES_PER_SEQUENCE \
    (SPARK_GLM52_RING_RUNTIME_BF16_HIDDEN_BYTES_PER_SEQUENCE + \
     SPARK_GLM52_RING_RUNTIME_MAX_SIDEBAND_BYTES_PER_SEQUENCE)
#define SPARK_GLM52_RING_RUNTIME_LAYER_MAJOR_TRANSPORT_BYTES_PER_ROW \
	(SPARK_GLM52_RING_RUNTIME_BF16_HIDDEN_BYTES_PER_SEQUENCE + \
	 SPARK_GLM52_RING_RUNTIME_INDEXSHARE_SIDEBAND_BYTES_PER_SEQUENCE)
#define SPARK_GLM52_RING_RUNTIME_MAX_SPECULATIVE_ROWS_PER_LANE \
	SPARK_GLM52_MODEL_MAX_SPECULATIVE_ROWS_PER_LANE

#define SPARK_GLM52_RING_RUNTIME_RANK_FLAG_HAS_PREVIOUS 0x00000001u
#define SPARK_GLM52_RING_RUNTIME_RANK_FLAG_HAS_NEXT 0x00000002u
#define SPARK_GLM52_RING_RUNTIME_RANK_FLAG_FINAL_STAGE 0x00000004u
#define SPARK_GLM52_RING_RUNTIME_RANK_FLAG_DENSE_PREFIX 0x00000008u
#define SPARK_GLM52_RING_RUNTIME_RANK_KNOWN_FLAGS \
    (SPARK_GLM52_RING_RUNTIME_RANK_FLAG_HAS_PREVIOUS | \
     SPARK_GLM52_RING_RUNTIME_RANK_FLAG_HAS_NEXT | \
     SPARK_GLM52_RING_RUNTIME_RANK_FLAG_FINAL_STAGE | \
     SPARK_GLM52_RING_RUNTIME_RANK_FLAG_DENSE_PREFIX)

typedef struct SparkGlm52RingRuntimeRankPlan
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
    uint32_t logical_lane_capacity;
    uint32_t maximum_speculative_rows_per_lane;
    uint32_t execution_row_capacity;
    uint32_t hidden_dimension;
    uint32_t bytes_per_sequence;
    uint32_t quantization_mode;
    // Inference shape: the node serves pp_stage_index of pp_stage_count
    // pipeline stages and tp_rank of tp_degree tensor-parallel shards of
    // that stage's layers. rank_index is the linear node index
    // pp_stage_index times tp_degree plus tp_rank. Legacy RING plans carry
    // degree one, rank zero, thirteen stages, and stage index equal to
    // rank index, so existing deployments validate unchanged. The
    // configuration hash is the shape config derivation over the model
    // constants; two nodes agreeing on it agree on the whole geometry.
    uint32_t tp_degree;
    uint32_t tp_rank;
    uint32_t pp_stage_count;
    uint32_t pp_stage_index;
    uint32_t tp_collective_listen_port;
    uint32_t reserved_shape;
    uint64_t shape_configuration_hash;
    uint64_t max_packet_bytes;
    char host_name[SPARK_GLM52_RING_RUNTIME_HOST_NAME_BYTES];
    char previous_host_name[SPARK_GLM52_RING_RUNTIME_HOST_NAME_BYTES];
    char next_host_name[SPARK_GLM52_RING_RUNTIME_HOST_NAME_BYTES];
    char tp_peer_host_names[SPARK_TP_COLLECTIVE_MAX_STEPS][SPARK_GLM52_RING_RUNTIME_HOST_NAME_BYTES];
    uint32_t tp_peer_ports[SPARK_TP_COLLECTIVE_MAX_STEPS];
    char input_route_name[SPARK_GLM52_RING_RUNTIME_ROUTE_NAME_BYTES];
    char output_route_name[SPARK_GLM52_RING_RUNTIME_ROUTE_NAME_BYTES];
    SparkHiddenTransportEndpoint input_endpoint;
    SparkHiddenTransportEndpoint output_endpoint;
} SparkGlm52RingRuntimeRankPlan;

uint32_t SparkGlm52RingRuntimeDsaCandidateBucket(
    uint32_t context_token_count);
uint32_t SparkGlm52RingRuntimeExecutionRowCapacity(
    uint32_t logical_lane_capacity);

SparkStatus SparkGlm52RingRuntimeParseQuantizationMode(
    const char *name,
    uint32_t *quantization_mode_out);

const char *SparkGlm52RingRuntimeQuantizationModeName(
    uint32_t quantization_mode);

SparkStatus SparkGlm52RingRuntimeValidateFp8PlanCounts(
    uint32_t quantization_mode,
    uint32_t bound_plan_count,
    uint32_t expected_plan_count);

SparkStatus SparkGlm52RingRuntimeExpectedMoeBackendKind(
    uint32_t quantization_mode,
    uint32_t *backend_kind_out);

typedef struct SparkGlm52RingRuntimeFinalEventRoute
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t source_rank_index;
    uint32_t sink_rank_index;
    uint32_t listen_port;
    uint32_t connect_port;
    char source_host_name[SPARK_GLM52_RING_RUNTIME_HOST_NAME_BYTES];
    char sink_host_name[SPARK_GLM52_RING_RUNTIME_HOST_NAME_BYTES];
    char route_name[SPARK_GLM52_RING_RUNTIME_FINAL_EVENT_ROUTE_NAME_BYTES];
} SparkGlm52RingRuntimeFinalEventRoute;

typedef struct SparkGlm52RingRuntimeFinalEvent
{
    uint32_t magic;
    uint32_t descriptor_bytes;
    uint32_t status;
    uint32_t program_id;
    uint32_t driver_dispatch_slot;
    uint32_t accepted_token_count;
    uint32_t completion_flags;
    uint32_t token_count;
    uint32_t token_ids[SPARK_MODEL_DRIVER_COMPLETION_TOKEN_CAPACITY];
    uint32_t draft_token_count;
    uint32_t draft_token_ids[SPARK_MODEL_DRIVER_COMPLETION_DRAFT_TOKEN_CAPACITY];
    uint64_t request_id;
    uint64_t sequence_id;
    uint64_t sequence_position;
    uint64_t service_time_ns;
	uint32_t extension_flags;
	uint32_t reserved0;
	SparkGlm52DsparkDraftResult dspark_draft;
} SparkGlm52RingRuntimeFinalEvent;

#define SPARK_GLM52_RING_RUNTIME_FINAL_EVENT_FLAG_DSPARK_DRAFT 0x00000001u
#define SPARK_GLM52_RING_RUNTIME_FINAL_EVENT_KNOWN_FLAGS \
	SPARK_GLM52_RING_RUNTIME_FINAL_EVENT_FLAG_DSPARK_DRAFT

#define SPARK_GLM52_RING_RUNTIME_FINAL_EVENT_ROUTE_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52RingRuntimeFinalEventRoute))
#define SPARK_GLM52_RING_RUNTIME_FINAL_EVENT_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52RingRuntimeFinalEvent))

SparkStatus SparkGlm52RingRuntimeBuildFixedStagePlan(
    SparkGlm52StagePlan *stage_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes);

SparkStatus SparkGlm52RingRuntimeRankHostName(
    uint32_t rank_index,
    char *host_name,
    uint32_t host_name_bytes);

SparkStatus SparkGlm52RingRuntimeBuildRankPlan(
    uint32_t rank_index,
    uint32_t logical_lane_capacity,
    uint32_t port_base,
    uint32_t quantization_mode,
    SparkGlm52RingRuntimeRankPlan *rank_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes);

// Build a rank plan from the inference shape: the node holds exactly the
// layers of its PP stage and its TP shard of them, the pipeline ring links
// stage neighbors at the same TP rank, and the collective peers are the
// same-stage partners whose ranks differ in one bit, with collective ports
// allocated from tp_port_base by linear node index. Hosts come from the
// fixed host table by linear node index, so shapes needing more nodes than
// the table lists fail closed until the table grows with the hardware.
SparkStatus SparkGlm52RingRuntimeBuildShapeRankPlan(
    const SparkGlm52TpShapeDescriptor *shape,
    uint32_t logical_lane_capacity,
    uint32_t port_base,
    uint32_t tp_port_base,
    uint32_t quantization_mode,
    SparkGlm52RingRuntimeRankPlan *rank_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes);

SparkStatus SparkGlm52RingRuntimeValidateRankPlan(
    const SparkGlm52RingRuntimeRankPlan *rank_plan,
    char *error_buffer,
    uint32_t error_buffer_bytes);

SparkStatus SparkGlm52RingRuntimeBuildMoePackPath(
    const char *pack_root,
    uint32_t quantization_mode,
    uint32_t layer_index,
    uint32_t tp_degree,
    uint32_t tp_rank,
    char *pack_path,
    uint32_t pack_path_bytes);

SparkStatus SparkGlm52RingRuntimeValidateStageMoePackFiles(
    const SparkGlm52RingRuntimeRankPlan *rank_plan,
    const char *pack_root,
    char *error_buffer,
    uint32_t error_buffer_bytes);

SparkStatus SparkGlm52RingRuntimeBuildFinalEventRoute(
    uint32_t port_base,
    SparkGlm52RingRuntimeFinalEventRoute *route,
    char *error_buffer,
    uint32_t error_buffer_bytes);

SparkStatus SparkGlm52RingRuntimeValidateFinalEventRoute(
    const SparkGlm52RingRuntimeFinalEventRoute *route,
    char *error_buffer,
    uint32_t error_buffer_bytes);

#ifdef __cplusplus
}
#endif
