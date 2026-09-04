#include "sparkpipe/spark_fabric_topology.h"

#include <string.h>

static SparkStatus SparkFabricTopologyValidateCommon(
    const SparkFabricTopologyConfiguration *configuration)
{
    if (configuration == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (configuration->abi_version != SPARK_FABRIC_TOPOLOGY_ABI_VERSION ||
        configuration->descriptor_bytes !=
            SPARK_FABRIC_TOPOLOGY_CONFIGURATION_DESCRIPTOR_BYTES)
    {
        return SPARK_STATUS_ABI_MISMATCH;
    }
    if ((configuration->flags & ~SPARK_FABRIC_TOPOLOGY_KNOWN_FLAGS) != 0u ||
        configuration->node_count < 2u ||
        configuration->local_node_index >= configuration->node_count ||
        configuration->link_speed_gbps !=
            SPARK_FABRIC_TOPOLOGY_LINK_SPEED_GBPS ||
        configuration->mtu_bytes < SPARK_FABRIC_TOPOLOGY_MINIMUM_MTU_BYTES ||
        configuration->mtu_bytes > SPARK_FABRIC_TOPOLOGY_MAXIMUM_MTU_BYTES)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (configuration->reserved0 != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkFabricTopologyValidate(
    const SparkFabricTopologyConfiguration *configuration)
{
    SparkStatus status;

    status = SparkFabricTopologyValidateCommon(configuration);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    switch (configuration->mode)
    {
        case SPARK_FABRIC_TOPOLOGY_MODE_RING_DEBUG_SINGLE_RAIL:
            if (configuration->rail_count != 1u ||
                configuration->switch_count != 0u ||
                configuration->active_port_count != 2u ||
                configuration->flags != SPARK_FABRIC_TOPOLOGY_FLAG_DEBUG_ONLY)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            return SPARK_STATUS_OK;

        case SPARK_FABRIC_TOPOLOGY_MODE_SINGLE_SWITCH_SINGLE_RAIL:
            if (configuration->rail_count != 1u ||
                configuration->switch_count != 1u ||
                configuration->active_port_count != 1u ||
                configuration->flags != SPARK_FABRIC_TOPOLOGY_FLAG_SWITCHED)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            return SPARK_STATUS_OK;

        case SPARK_FABRIC_TOPOLOGY_MODE_DUAL_SWITCH_DUAL_RAIL_FUTURE:
            if (configuration->rail_count != 2u ||
                configuration->switch_count != 2u ||
                configuration->active_port_count != 2u ||
                configuration->flags !=
                    (SPARK_FABRIC_TOPOLOGY_FLAG_SWITCHED |
                     SPARK_FABRIC_TOPOLOGY_FLAG_FUTURE_ONLY))
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            return SPARK_STATUS_UNSUPPORTED;

        default:
            return SPARK_STATUS_INVALID_ARGUMENT;
    }
}

static void SparkFabricTopologyInitializeRoute(
    const SparkFabricTopologyConfiguration *configuration,
    uint32_t destination_node_index,
    SparkFabricRoute *route)
{
    memset(route, 0, sizeof(*route));
    route->abi_version = SPARK_FABRIC_TOPOLOGY_ABI_VERSION;
    route->descriptor_bytes = SPARK_FABRIC_ROUTE_DESCRIPTOR_BYTES;
    route->topology_mode = configuration->mode;
    route->source_node_index = configuration->local_node_index;
    route->destination_node_index = destination_node_index;
    route->first_hop_node_index = SPARK_FABRIC_TOPOLOGY_NO_NODE;
    route->source_port_index = SPARK_FABRIC_TOPOLOGY_NO_PORT;
}

static SparkStatus SparkFabricTopologyResolveRingRoute(
    const SparkFabricTopologyConfiguration *configuration,
    uint32_t destination_node_index,
    SparkFabricRoute *route)
{
    uint32_t counter_clockwise_hops;
    uint32_t clockwise_hops;

    clockwise_hops =
        (destination_node_index + configuration->node_count -
         configuration->local_node_index) % configuration->node_count;
    counter_clockwise_hops =
        (configuration->local_node_index + configuration->node_count -
         destination_node_index) % configuration->node_count;

    route->rail_index = 0u;
    route->switched = 0u;
    if (clockwise_hops <= counter_clockwise_hops)
    {
        route->source_port_index = 1u;
        route->first_hop_node_index =
            (configuration->local_node_index + 1u) % configuration->node_count;
        route->hop_count = clockwise_hops;
    }
    else
    {
        route->source_port_index = 0u;
        route->first_hop_node_index =
            (configuration->local_node_index + configuration->node_count - 1u) %
            configuration->node_count;
        route->hop_count = counter_clockwise_hops;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkFabricTopologyResolveRoute(
    const SparkFabricTopologyConfiguration *configuration,
    uint32_t destination_node_index,
    SparkFabricRoute *route)
{
    SparkStatus status;

    if (route == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    status = SparkFabricTopologyValidate(configuration);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (destination_node_index >= configuration->node_count ||
        destination_node_index == configuration->local_node_index)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    SparkFabricTopologyInitializeRoute(
        configuration,
        destination_node_index,
        route);

    if (configuration->mode ==
        SPARK_FABRIC_TOPOLOGY_MODE_RING_DEBUG_SINGLE_RAIL)
    {
        return SparkFabricTopologyResolveRingRoute(
            configuration,
            destination_node_index,
            route);
    }
    if (configuration->mode ==
        SPARK_FABRIC_TOPOLOGY_MODE_SINGLE_SWITCH_SINGLE_RAIL)
    {
        route->rail_index = 0u;
        route->source_port_index = 0u;
        route->first_hop_node_index = destination_node_index;
        route->hop_count = 1u;
        route->switched = 1u;
        return SPARK_STATUS_OK;
    }
    return SPARK_STATUS_UNSUPPORTED;
}

const char *SparkFabricTopologyModeToString(SparkFabricTopologyMode mode)
{
    switch (mode)
    {
        case SPARK_FABRIC_TOPOLOGY_MODE_RING_DEBUG_SINGLE_RAIL:
            return "ring_debug_single_rail";
        case SPARK_FABRIC_TOPOLOGY_MODE_SINGLE_SWITCH_SINGLE_RAIL:
            return "single_switch_single_rail";
        case SPARK_FABRIC_TOPOLOGY_MODE_DUAL_SWITCH_DUAL_RAIL_FUTURE:
            return "dual_switch_dual_rail_future";
        default:
            return "invalid";
    }
}
