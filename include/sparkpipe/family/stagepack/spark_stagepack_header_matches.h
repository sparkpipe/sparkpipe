#pragma once

static inline int32_t SPARK_FAMILY(StagePackHeaderMatches)(const SPARK_FAMILY(StagePackHeader) *file_header, const SPARK_FAMILY(StagePackHeader) *expected)
{
	return(SparkStagePackHeaderMatches(
		(const SparkStagePackHeaderCommon *)file_header,
		(const SparkStagePackHeaderCommon *)expected));
}
