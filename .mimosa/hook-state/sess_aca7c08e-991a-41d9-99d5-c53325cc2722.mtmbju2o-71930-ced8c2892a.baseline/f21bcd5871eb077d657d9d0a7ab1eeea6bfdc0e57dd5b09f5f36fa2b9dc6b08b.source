#ifndef SPARKPIPE_SPARK_HARDWARE_TOPOLOGY_H
#define SPARKPIPE_SPARK_HARDWARE_TOPOLOGY_H


#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_HARDWARE_TOPOLOGY_ABI_VERSION 1u

#define SPARK_HARDWARE_TOPOLOGY_FLAG_DEBUG_ONLY 0x00000001u
#define SPARK_HARDWARE_TOPOLOGY_FLAG_FUTURE_ONLY 0x00000004u

typedef enum SparkHardwareTopologyMode
{
	SPARK_HARDWARE_TOPOLOGY_MODE_INVALID = 0,
	SPARK_HARDWARE_TOPOLOGY_MODE_RING = 1,
	SPARK_HARDWARE_TOPOLOGY_MODE_SINGLE_SWITCH = 2,
	SPARK_HARDWARE_TOPOLOGY_MODE_DUAL_SWITCH = 3
} SparkHardwareTopologyMode;

typedef enum SparkHardwareFabricKind
{
	SPARK_HARDWARE_FABRIC_KIND_INVALID = 0,
	SPARK_HARDWARE_FABRIC_KIND_DIRECT = 1,
	SPARK_HARDWARE_FABRIC_KIND_SWITCH = 2
} SparkHardwareFabricKind;

typedef struct SparkHardwareNodeType
{
	const char *name;
	uint32_t memory_bandwidth_gb_s;
	uint32_t bf16_tflops;
	uint32_t fp8_tflops;
	uint32_t fp4_tflops;
	uint64_t unified_memory_bytes;
	uint32_t nvme_read_gb_s;
	uint32_t network_port_count;
} SparkHardwareNodeType;

typedef struct SparkHardwareFabric
{
	const char *name;
	SparkHardwareFabricKind kind;
	uint32_t speed_gbps;
	uint32_t latency_ns;
	uint32_t mtu_bytes;
} SparkHardwareFabric;

typedef struct SparkHardwareNodePort
{
	uint32_t fabric_index;
	const char *ipv4;
} SparkHardwareNodePort;

typedef struct SparkHardwareComputeNode
{
	const char *name;
	uint32_t node_type_index;
	uint32_t rank;
	const char *nvme_device;
	const SparkHardwareNodePort *ports;
	uint32_t port_count;
} SparkHardwareComputeNode;

typedef struct SparkHardwareTopologyProjection
{
	uint32_t rail_count;
	uint32_t switch_count;
	uint32_t active_port_count;
	uint32_t link_speed_gbps;
	uint32_t mtu_bytes;
	uint32_t flags;
} SparkHardwareTopologyProjection;

typedef struct SparkHardwareTopology
{
	const char *name;
	SparkHardwareTopologyMode mode;
	const SparkHardwareNodeType *node_types;
	uint32_t node_type_count;
	const SparkHardwareFabric *fabrics;
	uint32_t fabric_count;
	const SparkHardwareComputeNode *compute_nodes;
	uint32_t compute_node_count;
	SparkHardwareTopologyProjection projection;
} SparkHardwareTopology;

#define SPARK_HARDWARE_TOPOLOGY_REGISTRY_COUNT 3u

extern const SparkHardwareTopology spark_hardware_topology_dual_switch_16node_production;
extern const SparkHardwareTopology spark_hardware_topology_ring_13node_bringup;
extern const SparkHardwareTopology spark_hardware_topology_single_switch_16node;

extern const SparkHardwareTopology *const spark_hardware_topology_registry[SPARK_HARDWARE_TOPOLOGY_REGISTRY_COUNT];

#ifdef __cplusplus
}
#endif

#endif
