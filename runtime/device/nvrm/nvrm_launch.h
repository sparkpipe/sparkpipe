#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"
#include "runtime/device/nvrm/nvrm_cubin.h"
#include "runtime/device/nvrm/nvrm_gpu.h"
#include "runtime/device/nvrm/nvrm_manifest.h"
#include "runtime/device/nvrm/nvrm_qmd.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_NVRM_PROGRAM_PAD 4096u
#define SPARK_NVRM_LAUNCH_QMD_BYTES 512u
#define SPARK_NVRM_LOCAL_EXTRA_BYTES 0x240u

typedef struct SparkNvrmProgram
{
	SparkNvrmCubin cubin;
	SparkNvrmManifest manifest;
	SparkNvrmMemory code;
} SparkNvrmProgram;

typedef struct SparkNvrmLaunchRequest
{
	const SparkNvrmCubinKernel *kernel;
	uint32_t grid[3];
	uint32_t block[3];
	uint32_t dynamic_shared_bytes;
	const void *arguments;
	uint32_t argument_bytes;
	uint8_t *slot;
	uint64_t slot_address;
	uint32_t slot_bytes;
	uint64_t release_address;
	uint64_t release_payload;
} SparkNvrmLaunchRequest;

SparkStatus SparkNvrmProgramLoad(SparkNvrmGpu *gpu, const uint8_t *blob, uint64_t bytes, const char *manifest, uint64_t manifest_bytes,
	SparkNvrmProgram *program, const char *owner);
SparkStatus SparkNvrmProgramLoadFile(SparkNvrmGpu *gpu, const char *path, SparkNvrmProgram *program, const char *owner);
void SparkNvrmProgramRelease(SparkNvrmGpu *gpu, SparkNvrmProgram *program);
uint8_t *SparkNvrmReadFile(const char *path, uint64_t *bytes);
uint32_t SparkNvrmLaunchSlotBytes(const SparkNvrmCubinKernel *kernel);
SparkStatus SparkNvrmLaunchQueue(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, const SparkNvrmProgram *program, const SparkNvrmLaunchRequest *request,
	const char *owner);
SparkStatus SparkNvrmLaunch(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, const SparkNvrmProgram *program, const SparkNvrmLaunchRequest *request,
	uint64_t *value, const char *owner);

#ifdef __cplusplus
}
#endif
