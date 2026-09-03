#ifndef SPARKPIPE_SPARK_FABRIC_TOPOLOGY_H
#define SPARKPIPE_SPARK_FABRIC_TOPOLOGY_H

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_FABRIC_TOPOLOGY_ABI_VERSION 1u
#define SPARK_FABRIC_TOPOLOGY_LINK_SPEED_GBPS 100u
#define SPARK_FABRIC_TOPOLOGY_MINIMUM_MTU_BYTES 1500u
#define SPARK_FABRIC_TOPOLOGY_MAXIMUM_MTU_BYTES 9600u
#define SPARK_FABRIC_TOPOLOGY_NO_NODE UINT32_MAX
#define SPARK_FABRIC_TOPOLOGY_NO_PORT UINT32_MAX

#define SPARK_FABRIC_TOPOLOGY_FLAG_DEBUG_ONLY 0x00000001u
#define SPARK_FABRIC_TOPOLOGY_FLAG_SWITCHED 0x00000002u
#define SPARK_FABRIC_TOPOLOGY_FLAG_FUTURE_ONLY 0x00000004u
#define SPARK_FABRIC_TOPOLOGY_KNOWN_FLAGS \
    (SPARK_FABRIC_TOPOLOGY_FLAG_DEBUG_ONLY | \
     SPARK_FABRIC_TOPOLOGY_FLAG_SWITCHED | \
     SPARK_FABRIC_TOPOLOGY_FLAG_FUTURE_ONLY)

typedef enum SparkFabricTopologyMode
{
    SPARK_FABRIC_TOPOLOGY_MODE_INVALID = 0,
    SPARK_FABRIC_TOPOLOGY_MODE_RING_DEBUG_SINGLE_RAIL = 1,
    SPARK_FABRIC_TOPOLOGY_MODE_SINGLE_SWITCH_SINGLE_RAIL = 2,
    SPARK_FABRIC_TOPOLOGY_MODE_DUAL_SWITCH_DUAL_RAIL_FUTURE = 3
} SparkFabricTopologyMode;

typedef struct SparkFabricTopologyConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    SparkFabricTopologyMode mode;
    uint32_t flags;
    uint32_t node_count;
    uint32_t local_node_index;
    uint32_t rail_count;
    uint32_t switch_count;
    uint32_t active_port_count;
    uint32_t link_speed_gbps;
    uint32_t mtu_bytes;
    uint32_t reserved0;
} SparkFabricTopologyConfiguration;

typedef struct SparkFabricRoute
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    SparkFabricTopologyMode topology_mode;
    uint32_t rail_index;
    uint32_t source_node_index;
    uint32_t destination_node_index;
    uint32_t source_port_index;
    uint32_t first_hop_node_index;
    uint32_t hop_count;
    uint32_t switched;
    uint32_t reserved0;
    uint32_t reserved1;
} SparkFabricRoute;

#define SPARK_FABRIC_TOPOLOGY_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkFabricTopologyConfiguration))
#define SPARK_FABRIC_ROUTE_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkFabricRoute))

SparkStatus SparkFabricTopologyValidate(
    const SparkFabricTopologyConfiguration *configuration);

SparkStatus SparkFabricTopologyResolveRoute(
    const SparkFabricTopologyConfiguration *configuration,
    uint32_t destination_node_index,
    SparkFabricRoute *route);

const char *SparkFabricTopologyModeToString(SparkFabricTopologyMode mode);

#ifdef __cplusplus
}
#endif

#endif
