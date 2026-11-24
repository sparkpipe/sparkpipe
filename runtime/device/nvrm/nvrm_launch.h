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
#define SPARK_NVRM_CLUSTER_GROUPS_MAX 8u
#define SPARK_NVRM_LATCH_SEGMENT 32u
#define SPARK_NVRM_PROGRAM_LATCH_FIRST 0x80000000u
#define SPARK_NVRM_LAUNCH_WORDS 4u

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
	uint32_t cluster[3];
	uint32_t dependent;
	uint32_t latch_arrive;
	uint32_t latch_wait;
	uint64_t next_qmd_address;
} SparkNvrmLaunchRequest;

SparkStatus SparkNvrmProgramLoad(SparkNvrmGpu *gpu, const uint8_t *blob, uint64_t bytes, const char *manifest, uint64_t manifest_bytes,
	SparkNvrmProgram *program, const char *owner);
SparkStatus SparkNvrmProgramLoadFile(SparkNvrmGpu *gpu, const char *path, SparkNvrmProgram *program, const char *owner);
void SparkNvrmProgramRelease(SparkNvrmGpu *gpu, SparkNvrmProgram *program);
uint8_t *SparkNvrmReadFile(const char *path, uint64_t *bytes);
uint32_t SparkNvrmLaunchSlotBytes(const SparkNvrmCubinKernel *kernel);
SparkStatus SparkNvrmKernelRunnable(const SparkNvrmCubinKernel *kernel, const char *owner);
SparkStatus SparkNvrmLaunchClusterShape(const SparkNvrmLaunchRequest *request, uint32_t cluster[3], const char *owner);
uint32_t SparkNvrmLatchSegment(void);
SparkStatus SparkNvrmLatchRangeTake(uint32_t count, uint32_t *base, const char *owner);
void SparkNvrmLatchRangeGive(uint32_t base, uint32_t count);
void SparkNvrmLatchSegmentSet(uint32_t launches);
void SparkNvrmLaunchWords(uint32_t *words, uint64_t slot_address);
SparkStatus SparkNvrmLaunchDescribe(const SparkNvrmGpu *gpu, const SparkNvrmProgram *program, const SparkNvrmLaunchRequest *request, const char *owner);
SparkStatus SparkNvrmLaunchQueue(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, const SparkNvrmProgram *program, const SparkNvrmLaunchRequest *request,
	const char *owner);
SparkStatus SparkNvrmLaunch(SparkNvrmGpu *gpu, SparkNvrmChannel *channel, const SparkNvrmProgram *program, const SparkNvrmLaunchRequest *request,
	uint64_t *value, const char *owner);

#ifdef __cplusplus
}
#endif
