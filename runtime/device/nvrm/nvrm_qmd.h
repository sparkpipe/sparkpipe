#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"
#include "runtime/device/nvrm/nvrm_cubin.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_NVRM_QMD_WORDS 96u
#define SPARK_NVRM_QMD_ALIGN 256u
#define SPARK_NVRM_BANK0_DRIVER_BYTES 0x380u
#define SPARK_NVRM_BANK_ALIGN 256u
#define SPARK_NVRM_SHARED_CONFIG_KIB_MAX 100u

typedef struct SparkNvrmQmd
{
	uint32_t word[SPARK_NVRM_QMD_WORDS];
} SparkNvrmQmd;

typedef struct SparkNvrmLaunchShape
{
	uint64_t program_address;
	uint64_t program_bytes;
	uint32_t grid[3];
	uint32_t block[3];
	uint32_t registers;
	uint32_t barriers;
	uint32_t shared_bytes;
	uint32_t local_bytes;
	uint32_t sass_version;
	uint64_t bank_address[SPARK_NVRM_CUBIN_BANKS];
	uint32_t bank_bytes[SPARK_NVRM_CUBIN_BANKS];
	uint32_t bank_mask;
	uint64_t release_address;
	uint64_t release_payload;
	uint32_t release_timestamp;
} SparkNvrmLaunchShape;

SparkStatus SparkNvrmQmdWrite(SparkNvrmQmd *qmd, uint32_t high, uint32_t low, uint64_t value, const char *owner);
uint64_t SparkNvrmQmdRead(const SparkNvrmQmd *qmd, uint32_t high, uint32_t low);
SparkStatus SparkNvrmQmdBuild(const SparkNvrmLaunchShape *shape, SparkNvrmQmd *qmd, const char *owner);
SparkStatus SparkNvrmBank0Build(const SparkNvrmCubinKernel *kernel, uint64_t shared_window, uint64_t local_window,
	const void *arguments, uint32_t argument_bytes, uint8_t *bank, uint32_t bank_bytes, const char *owner);

#ifdef __cplusplus
}
#endif
