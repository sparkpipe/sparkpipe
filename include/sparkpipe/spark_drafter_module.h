#pragma once

#include <stdint.h>

#include "sparkpipe/spark_draft_bridge.h"
#include "sparkpipe/spark_draft_server.h"
#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_tap_stream_reader.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_DRAFTER_MODULE_ABI_VERSION 1u
#define SPARK_DRAFTER_MODULE_INTERFACE_SYMBOL "spark_drafter_module_interface"
#define SPARK_DRAFTER_MODULE_TEXT_BYTES 512u
#define SPARK_DRAFTER_MODULE_PROPOSE_DRAFTS SPARK_STATUS_OK
#define SPARK_DRAFTER_MODULE_PROPOSE_AWAIT_TAPS SPARK_STATUS_PENDING
#define SPARK_DRAFTER_MODULE_PROPOSE_NO_CONTEXT SPARK_STATUS_NOT_FOUND

typedef struct SparkDrafterModuleLane
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	const char *lane_name;
	const char *pack_path;
	const char *model;
	const char *revision;
	const char *pack_sha256;
	const char *weightd_socket;
	uint64_t pack_bytes;
	uint64_t context_bytes;
	uint64_t timeout_ns;
} SparkDrafterModuleLane;

typedef struct SparkDrafterModuleBind
{
	uint64_t device_bytes;
	uint64_t device_free_bytes;
	uint64_t device_total_bytes;
	uint64_t bind_ns;
	uint32_t first_hold;
	uint32_t reserved;
	char text[SPARK_DRAFTER_MODULE_TEXT_BYTES];
} SparkDrafterModuleBind;

typedef struct SparkDrafterModuleInterface
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	const char *name;
	SparkStatus (*open)(const SparkDrafterModuleLane *lane, void **state, SparkDrafterModuleBind *bind);
	SparkStatus (*ingest)(void *state, const SparkTapStreamReaderFrame *frame);
	SparkStatus (*propose)(void *state, const SparkDraftServerRequest *request, SparkDraftBridgeNode *nodes, uint32_t capacity,
		uint32_t *count);
	SparkStatus (*close)(void *state);
} SparkDrafterModuleInterface;

typedef const SparkDrafterModuleInterface *(*SparkDrafterModuleGetInterface)(void);

#ifdef __cplusplus
}
#endif
