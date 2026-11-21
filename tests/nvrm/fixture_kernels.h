#pragma once

#include <stdint.h>

#define SPARK_NVRM_FIXTURE_TILE 1000u
#define SPARK_NVRM_FIXTURE_SCRATCH 64u
#define SPARK_NVRM_FIXTURE_BLOCK_WORDS 1024u
#define SPARK_NVRM_FIXTURE_SINK_MATCH 0x9e3779b9u
#define SPARK_NVRM_FIXTURE_STREAM_THREADS 256u

typedef struct SparkNvrmFixtureShape
{
	uint32_t rows;
	uint16_t lanes;
	uint8_t flag;
	double weight;
} SparkNvrmFixtureShape;

typedef struct SparkNvrmFixtureBlock
{
	uint32_t words[SPARK_NVRM_FIXTURE_BLOCK_WORDS];
} SparkNvrmFixtureBlock;
