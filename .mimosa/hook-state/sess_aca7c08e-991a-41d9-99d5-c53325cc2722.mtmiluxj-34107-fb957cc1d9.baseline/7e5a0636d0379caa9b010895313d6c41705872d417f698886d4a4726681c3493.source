#include <assert.h>
#include <string.h>

#include "sparkpipe/spark_fabric_topology.h"

static SparkFabricTopologyConfiguration SparkTestRingConfiguration(void)
{
    SparkFabricTopologyConfiguration configuration;

    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_FABRIC_TOPOLOGY_ABI_VERSION;
    configuration.descriptor_bytes =
        SPARK_FABRIC_TOPOLOGY_CONFIGURATION_DESCRIPTOR_BYTES;
    configuration.mode = SPARK_FABRIC_TOPOLOGY_MODE_RING_DEBUG_SINGLE_RAIL;
    configuration.flags = SPARK_FABRIC_TOPOLOGY_FLAG_DEBUG_ONLY;
    configuration.node_count = 16u;
    configuration.local_node_index = 0u;
    configuration.rail_count = 1u;
    configuration.switch_count = 0u;
    configuration.active_port_count = 2u;
    configuration.link_speed_gbps = SPARK_FABRIC_TOPOLOGY_LINK_SPEED_GBPS;
    configuration.mtu_bytes = 9000u;
    return configuration;
}

static SparkFabricTopologyConfiguration SparkTestSwitchConfiguration(void)
{
    SparkFabricTopologyConfiguration configuration;

    configuration = SparkTestRingConfiguration();
    configuration.mode =
        SPARK_FABRIC_TOPOLOGY_MODE_SINGLE_SWITCH_SINGLE_RAIL;
    configuration.flags = SPARK_FABRIC_TOPOLOGY_FLAG_SWITCHED;
    configuration.switch_count = 1u;
    configuration.active_port_count = 1u;
    return configuration;
}

static void SparkTestRingRoutesBothDirections(void)
{
    SparkFabricTopologyConfiguration configuration;
    SparkFabricRoute route;

    configuration = SparkTestRingConfiguration();
    assert(SparkFabricTopologyValidate(&configuration) == SPARK_STATUS_OK);
    assert(SparkFabricTopologyResolveRoute(
        &configuration,
        1u,
        &route) == SPARK_STATUS_OK);
    assert(route.source_port_index == 1u);
    assert(route.first_hop_node_index == 1u);
    assert(route.hop_count == 1u);
    assert(route.switched == 0u);

    assert(SparkFabricTopologyResolveRoute(
        &configuration,
        15u,
        &route) == SPARK_STATUS_OK);
    assert(route.source_port_index == 0u);
    assert(route.first_hop_node_index == 15u);
    assert(route.hop_count == 1u);

    assert(SparkFabricTopologyResolveRoute(
        &configuration,
        8u,
        &route) == SPARK_STATUS_OK);
    assert(route.source_port_index == 1u);
    assert(route.hop_count == 8u);
}

static void SparkTestSingleSwitchRoutesAnyPeerDirectly(void)
{
    SparkFabricTopologyConfiguration configuration;
    SparkFabricRoute route;

    configuration = SparkTestSwitchConfiguration();
    assert(SparkFabricTopologyValidate(&configuration) == SPARK_STATUS_OK);
    assert(SparkFabricTopologyResolveRoute(
        &configuration,
        11u,
        &route) == SPARK_STATUS_OK);
    assert(route.rail_index == 0u);
    assert(route.source_port_index == 0u);
    assert(route.first_hop_node_index == 11u);
    assert(route.hop_count == 1u);
    assert(route.switched == 1u);
}

static void SparkTestFutureDualRailFailsClosed(void)
{
    SparkFabricTopologyConfiguration configuration;
    SparkFabricRoute route;

    configuration = SparkTestSwitchConfiguration();
    configuration.mode =
        SPARK_FABRIC_TOPOLOGY_MODE_DUAL_SWITCH_DUAL_RAIL_FUTURE;
    configuration.flags =
        SPARK_FABRIC_TOPOLOGY_FLAG_SWITCHED |
        SPARK_FABRIC_TOPOLOGY_FLAG_FUTURE_ONLY;
    configuration.rail_count = 2u;
    configuration.switch_count = 2u;
    configuration.active_port_count = 2u;
    assert(SparkFabricTopologyValidate(&configuration) ==
        SPARK_STATUS_UNSUPPORTED);
    assert(SparkFabricTopologyResolveRoute(
        &configuration,
        1u,
        &route) == SPARK_STATUS_UNSUPPORTED);
}

static void SparkTestInvalidConfigurations(void)
{
    SparkFabricTopologyConfiguration configuration;
    SparkFabricRoute route;

    configuration = SparkTestRingConfiguration();
    configuration.link_speed_gbps = 25u;
    assert(SparkFabricTopologyValidate(&configuration) ==
        SPARK_STATUS_INVALID_ARGUMENT);

    configuration = SparkTestSwitchConfiguration();
    configuration.active_port_count = 2u;
    assert(SparkFabricTopologyValidate(&configuration) ==
        SPARK_STATUS_INVALID_ARGUMENT);

    configuration = SparkTestRingConfiguration();
    assert(SparkFabricTopologyResolveRoute(
        &configuration,
        configuration.local_node_index,
        &route) == SPARK_STATUS_INVALID_ARGUMENT);
}

int main(void)
{
    SparkTestRingRoutesBothDirections();
    SparkTestSingleSwitchRoutesAnyPeerDirectly();
    SparkTestFutureDualRailFailsClosed();
    SparkTestInvalidConfigurations();
    return 0;
}
