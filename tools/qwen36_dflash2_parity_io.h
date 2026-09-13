// qwen36_dflash2_parity_io.h - shared IO/compare helpers for the DFlash2
// W3/W4 parity harnesses. The harnesses read the numpy oracle's dumps
// (tools/qwen36_dflash2_w34_oracle.py) and diff the kernel output against
// them; nothing here touches the kernels themselves.
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define SPARK_PARITY_MAX_DIMS 16
#define SPARK_PARITY_REPORT 6

typedef struct SparkParityDims
{
	char name[SPARK_PARITY_MAX_DIMS][32];
	uint64_t value[SPARK_PARITY_MAX_DIMS];
	int count;
} SparkParityDims;

static SparkParityDims SparkParityLoadDims(const char *directory, const char *test_case)
{
	SparkParityDims dims;
	char path[512];
	FILE *file;
	dims.count = 0;
	snprintf(path, sizeof(path), "%s/%s_dims.txt", directory, test_case);
	file = fopen(path, "r");
	if (file == NULL) { fprintf(stderr, "PARITY FAIL: cannot open %s\n", path); exit(2); }
	while (dims.count < SPARK_PARITY_MAX_DIMS &&
		fscanf(file, "%31s %llu", dims.name[dims.count], (unsigned long long *)&dims.value[dims.count]) == 2)
		dims.count++;
	fclose(file);
	return dims;
}

static uint64_t SparkParityDim(const SparkParityDims *dims, const char *name)
{
	int index;
	for (index = 0; index < dims->count; index++)
		if (strcmp(dims->name[index], name) == 0)
			return dims->value[index];
	fprintf(stderr, "PARITY FAIL: missing dim %s\n", name);
	exit(2);
}

static void *SparkParityLoad(const char *directory, const char *test_case, const char *suffix, size_t bytes)
{
	char path[512];
	FILE *file;
	void *buffer;
	size_t read;
	snprintf(path, sizeof(path), "%s/%s_%s", directory, test_case, suffix);
	file = fopen(path, "rb");
	if (file == NULL) { fprintf(stderr, "PARITY FAIL: cannot open %s\n", path); exit(2); }
	buffer = malloc(bytes);
	if (buffer == NULL) { fprintf(stderr, "PARITY FAIL: malloc %zu\n", bytes); exit(2); }
	read = fread(buffer, 1, bytes, file);
	if (read != bytes) { fprintf(stderr, "PARITY FAIL: %s short read %zu of %zu\n", path, read, bytes); exit(2); }
	fclose(file);
	return buffer;
}

static uint64_t SparkParityCompareU32(const char *label, const uint32_t *actual, const uint32_t *expect, uint64_t count)
{
	uint64_t index, mismatches = 0;
	for (index = 0; index < count; index++)
		if (actual[index] != expect[index])
		{
			if (mismatches < SPARK_PARITY_REPORT)
				printf("  %s[%llu]: kernel %u vs oracle %u\n", label,
					(unsigned long long)index, actual[index], expect[index]);
			mismatches++;
		}
	printf("%s: %llu / %llu mismatched\n", label, (unsigned long long)mismatches, (unsigned long long)count);
	return mismatches;
}

// Bitwise when exact != 0 (the lattice cases: every fp32 product and partial
// sum is exactly representable, so accumulation order cannot show), otherwise
// a relative-tolerance report.
static uint64_t SparkParityCompareF32(const char *label, const float *actual, const float *expect, uint64_t count, int exact, double tolerance)
{
	uint64_t index, mismatches = 0;
	double worst = 0.0;
	uint64_t worst_index = 0;
	for (index = 0; index < count; index++)
	{
		int bad;
		double difference = fabs((double)actual[index] - (double)expect[index]);
		double scale = fabs((double)expect[index]);
		double relative = scale > 0.0 ? difference / scale : difference;
		if (relative > worst) { worst = relative; worst_index = index; }
		bad = exact != 0
			? memcmp(&actual[index], &expect[index], sizeof(float)) != 0
			: relative > tolerance;
		if (bad)
		{
			if (mismatches < SPARK_PARITY_REPORT)
				printf("  %s[%llu]: kernel %.9g vs oracle %.9g (rel %.3g)\n", label,
					(unsigned long long)index, (double)actual[index], (double)expect[index], relative);
			mismatches++;
		}
	}
	printf("%s: %llu / %llu mismatched (%s), worst rel %.3g at %llu\n", label,
		(unsigned long long)mismatches, (unsigned long long)count,
		exact != 0 ? "bitwise" : "tolerance", worst, (unsigned long long)worst_index);
	return mismatches;
}

#define SPARK_PARITY_CUDA(call) \
	do { \
		cudaError_t parity_status = (call); \
		if (parity_status != cudaSuccess) { \
			fprintf(stderr, "PARITY FAIL: %s -> %s\n", #call, cudaGetErrorString(parity_status)); \
			exit(2); \
		} \
	} while (0)
