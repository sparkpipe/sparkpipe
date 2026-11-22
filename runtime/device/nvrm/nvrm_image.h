#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_NVRM_IMAGE_SEMAPHORE_BYTES 256u

typedef enum SparkNvrmStepKind
{
	SPARK_NVRM_STEP_LAUNCH = 1,
	SPARK_NVRM_STEP_RECORD = 2,
	SPARK_NVRM_STEP_AWAIT = 3,
	SPARK_NVRM_STEP_HOST = 4
} SparkNvrmStepKind;

typedef struct SparkNvrmStep
{
	SparkNvrmStepKind kind;
	uint32_t node;
	uint64_t slot;
	uint32_t slot_bytes;
	uint64_t address;
	uint64_t second_address;
	uint32_t payload;
	uint32_t second;
	uint32_t chained;
	const char *name;
} SparkNvrmStep;

void SparkNvrmImageChain(SparkNvrmStep *steps, uint32_t count, uint32_t segment);
uint32_t SparkNvrmImageWords(const SparkNvrmStep *steps, uint32_t count);
void SparkNvrmImageWrite(uint32_t *words, uint64_t semaphore, uint64_t slots, SparkNvrmStep *steps, uint32_t count);
void SparkNvrmImagePatch(uint32_t *words, uint32_t method, uint64_t value);
void SparkNvrmImagePatchAddress(uint32_t *words, uint32_t method, uint64_t address);
uint64_t SparkNvrmImagePayload(const uint32_t *words, uint32_t method);

#ifdef __cplusplus
}
#endif
