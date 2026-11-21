#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"
#include "runtime/device/nvrm/nvrm_cubin.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_NVRM_MANIFEST_DIGEST_BYTES 65u

typedef struct SparkNvrmParameter
{
	uint32_t align;
	uint32_t bytes;
	uint32_t offset;
} SparkNvrmParameter;

typedef struct SparkNvrmManifestKernel
{
	char *name;
	uint32_t registers;
	uint32_t barriers;
	uint32_t stack_bytes;
	uint32_t shared_bytes;
	uint32_t parameter_count;
	uint32_t parameter_capacity;
	SparkNvrmParameter *parameters;
	uint32_t max_threads;
	uint32_t clustered;
} SparkNvrmManifestKernel;

typedef struct SparkNvrmManifest
{
	char cubin_digest[SPARK_NVRM_MANIFEST_DIGEST_BYTES];
	uint64_t cubin_bytes;
	SparkNvrmManifestKernel *kernels;
	uint32_t kernel_count;
	uint32_t kernel_capacity;
} SparkNvrmManifest;

SparkStatus SparkNvrmManifestParse(const char *text, uint64_t bytes, const char *owner, SparkNvrmManifest *manifest);
const SparkNvrmManifestKernel *SparkNvrmManifestKernelNamed(const SparkNvrmManifest *manifest, const char *name);
uint32_t SparkNvrmManifestParameterEnd(const SparkNvrmManifestKernel *kernel);
SparkStatus SparkNvrmManifestCheck(const SparkNvrmManifest *manifest, const uint8_t *blob, uint64_t bytes, const SparkNvrmCubin *cubin, const char *owner);
void SparkNvrmManifestRelease(SparkNvrmManifest *manifest);

#ifdef __cplusplus
}
#endif
