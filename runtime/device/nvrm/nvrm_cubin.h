#pragma once

#include <stdint.h>

#include "runtime/device/nvrm/nvrm_names.h"

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_NVRM_CUBIN_BANKS 8u
#define SPARK_NVRM_CUBIN_SECTION_ALIGN 128u

typedef enum SparkNvrmRelocationKind
{
	SPARK_NVRM_RELOCATION_ADDRESS64 = 1,
	SPARK_NVRM_RELOCATION_ADDRESS_LOW32 = 2,
	SPARK_NVRM_RELOCATION_ADDRESS_HIGH32 = 3
} SparkNvrmRelocationKind;

typedef struct SparkNvrmCubinBank
{
	uint64_t image_offset;
	uint64_t bytes;
	uint32_t present;
} SparkNvrmCubinBank;

typedef struct SparkNvrmCubinKernel
{
	char name[SPARK_NVRM_SHORT_NAME_BYTES];
	const char *symbol;
	uint64_t code_offset;
	uint64_t code_bytes;
	uint32_t registers;
	uint32_t stack_bytes;
	uint32_t frame_bytes;
	uint32_t shared_bytes;
	uint32_t parameter_offset;
	uint32_t parameter_bytes;
	uint32_t barriers;
	uint32_t clustered;
	uint32_t cluster[3];
	uint32_t cluster_limit;
	SparkNvrmCubinBank bank[SPARK_NVRM_CUBIN_BANKS];
} SparkNvrmCubinKernel;

typedef struct SparkNvrmCubinRelocation
{
	uint64_t image_offset;
	uint64_t target;
	SparkNvrmRelocationKind kind;
} SparkNvrmCubinRelocation;

typedef struct SparkNvrmCubin
{
	uint8_t *image;
	uint64_t image_bytes;
	SparkNvrmCubinKernel *kernels;
	uint32_t kernel_count;
	SparkNvrmCubinRelocation *relocations;
	uint32_t relocation_count;
} SparkNvrmCubin;

SparkStatus SparkNvrmCubinParse(const uint8_t *blob, uint64_t bytes, const char *owner, SparkNvrmCubin *cubin);
const SparkNvrmCubinKernel *SparkNvrmCubinKernelNamed(const SparkNvrmCubin *cubin, const char *name);
SparkStatus SparkNvrmCubinRelocate(const SparkNvrmCubin *cubin, uint64_t base, uint8_t *image, const char *owner);
void SparkNvrmCubinRelease(SparkNvrmCubin *cubin);

#ifdef __cplusplus
}
#endif
