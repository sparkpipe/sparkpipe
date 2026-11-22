#include "sparkpipe/spark_clock.h"
#include "sparkpipe/spark_drafter_module.h"
#include "sparkpipe/spark_gpu.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_device.h"
#include "sparkpipe/spark_weightd_lazy_pack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef PROBE_MODULE_ABI
#define PROBE_MODULE_ABI SPARK_DRAFTER_MODULE_ABI_VERSION
#endif

typedef struct ProbeState
{
	SparkWeightdLazyPack *pack;
	uint64_t frames;
	uint64_t proposals;
} ProbeState;

static SparkStatus ProbeOpen(const SparkDrafterModuleLane *lane, void **state_out, SparkDrafterModuleBind *bind)
{
	SparkWeightdLazyAttachRequest request;
	SparkWeightdLazyPackOptions options;
	ProbeState *state;
	size_t free_bytes = 0u, total_bytes = 0u;
	uint64_t need, started;
	SparkStatus status;
	*state_out = 0;
	if ( lane == 0 || lane->abi_version != SPARK_DRAFTER_MODULE_ABI_VERSION || lane->descriptor_bytes < sizeof(*lane) )
	{
		snprintf(bind->text, sizeof(bind->text), "the lane descriptor is not drafter ABI %u", SPARK_DRAFTER_MODULE_ABI_VERSION);
		return SPARK_STATUS_ABI_MISMATCH;
	}
	if ( SparkWeightdDeviceOpen() != SPARK_STATUS_OK || SparkGpuMemGetInfo(&free_bytes, &total_bytes) != SPARK_GPU_SUCCESS )
	{
		snprintf(bind->text, sizeof(bind->text), "lane %s cannot read the device's free memory", lane->lane_name);
		return SPARK_STATUS_IO_ERROR;
	}
	bind->device_free_bytes = (uint64_t)free_bytes;
	bind->device_total_bytes = (uint64_t)total_bytes;
	need = lane->pack_bytes + 255u + lane->context_bytes;
	if ( need > (uint64_t)free_bytes )
	{
		snprintf(bind->text, sizeof(bind->text), "lane %s needs %llu bytes of device memory (pack %llu + context cap %llu), %llu of %llu free",
			lane->lane_name, (unsigned long long)need, (unsigned long long)lane->pack_bytes, (unsigned long long)lane->context_bytes,
			(unsigned long long)free_bytes, (unsigned long long)total_bytes);
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	}
	memset(&request, 0, sizeof(request));
	memset(&options, 0, sizeof(options));
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = lane->pack_bytes;
	snprintf(request.identity.model, sizeof(request.identity.model), "%s", lane->model);
	snprintf(request.identity.revision, sizeof(request.identity.revision), "%s", lane->revision);
	snprintf(request.identity.pack_sha256, sizeof(request.identity.pack_sha256), "%s", lane->pack_sha256);
	snprintf(request.pack_path, sizeof(request.pack_path), "%s", lane->pack_path);
	options.timeout_nanoseconds = lane->timeout_ns;
	state = (ProbeState *)calloc(1u, sizeof(*state));
	if ( state == 0 )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	started = SparkMonotonicNs();
	status = SparkWeightdLazyPackCreate(lane->weightd_socket, &request, &options, &state->pack);
	if ( status != SPARK_STATUS_OK )
	{
		snprintf(bind->text, sizeof(bind->text), "lane %s: the pack %s did not bind", lane->lane_name, lane->pack_path);
		free(state);
		return status;
	}
	bind->bind_ns = SparkMonotonicNs() - started;
	bind->device_bytes = state->pack->spine_allocation_bytes;
	bind->first_hold = state->pack->attached.loaded_from_pack;
	snprintf(bind->text, sizeof(bind->text), "lane %s bound %s", lane->lane_name, lane->pack_path);
	*state_out = state;
	return SPARK_STATUS_OK;
}

static SparkStatus ProbeIngest(void *opaque, const SparkTapStreamReaderFrame *frame)
{
	ProbeState *state = (ProbeState *)opaque;
	(void)frame;
	state->frames++;
	return SPARK_STATUS_OK;
}

static SparkStatus ProbePropose(void *opaque, const SparkDraftServerRequest *request, SparkDraftBridgeNode *nodes, uint32_t capacity,
	uint32_t *count)
{
	ProbeState *state = (ProbeState *)opaque;
	*count = 0u;
	if ( capacity == 0u )
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	state->proposals++;
	nodes[0].token_id = request->anchor_token_id;
	nodes[0].parent_index = SPARK_DRAFT_BRIDGE_ROOT_PARENT_INDEX;
	nodes[0].depth = 0u;
	nodes[0].source_bit = request->speculator_mask & (~request->speculator_mask + 1u);
	nodes[0].score = 1.0f;
	*count = 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus ProbeClose(void *opaque)
{
	ProbeState *state = (ProbeState *)opaque;
	SparkStatus status = SparkWeightdLazyPackDestroy(state->pack);
	free(state);
	return status;
}

static const SparkDrafterModuleInterface probe_interface = {
	PROBE_MODULE_ABI,
	(uint32_t)sizeof(SparkDrafterModuleInterface),
	"drafter-probe",
	ProbeOpen,
	ProbeIngest,
	ProbePropose,
	ProbeClose,
};

__attribute__((visibility("default"))) const SparkDrafterModuleInterface *spark_drafter_module_interface(void)
{
	return &probe_interface;
}
