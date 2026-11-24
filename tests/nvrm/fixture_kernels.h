#pragma once

#include <stdint.h>

#define SPARK_NVRM_FIXTURE_TILE 1000u
#define SPARK_NVRM_FIXTURE_SCRATCH 64u
#define SPARK_NVRM_FIXTURE_BLOCK_WORDS 1024u
#define SPARK_NVRM_FIXTURE_SINK_MATCH 0x9e3779b9u
#define SPARK_NVRM_FIXTURE_STREAM_THREADS 256u
#define SPARK_NVRM_FIXTURE_TABLE_WORDS 4u
#define SPARK_NVRM_FIXTURE_HUGE_WORDS 8189u
#define SPARK_NVRM_FIXTURE_DEEP_WORDS 1024u
#define SPARK_NVRM_FIXTURE_FRAME_WORDS 1100u
#define SPARK_NVRM_FIXTURE_DEEP_STRIDE 0x9e3779b9u
#define SPARK_NVRM_FIXTURE_LONG_TEXT \
	"_ZN29spark_nvrm_fixture_long_names21SparkNvrmLongTemplateINS_21SparkNvrmFixtureValueENS_24SparkNvrmFixtureTagAlphaENS_24SparkNvrmFixtureTagBravoEEEvPjT_"

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

typedef struct SparkNvrmFixtureHuge
{
	uint32_t words[SPARK_NVRM_FIXTURE_HUGE_WORDS];
} SparkNvrmFixtureHuge;
