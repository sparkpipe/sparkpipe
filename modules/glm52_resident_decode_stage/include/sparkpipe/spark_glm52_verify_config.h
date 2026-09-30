#pragma once

#include <stdint.h>
#include <stdlib.h>

#include "sparkpipe/spark_speculation_verify_plan.h"

#define SPARK_GLM52_VERIFY_ROWS_ENV "SPARK_GLM52_VERIFY_ROWS"
#define SPARK_GLM52_VERIFY_DRAFTER_ENV "SPARK_GLM52_VERIFY_DRAFTER"
#define SPARK_GLM52_VERIFY_ROWS_LIMIT 8u
#define SPARK_GLM52_VERIFY_LOOKUP_MIN_MATCH 3u
#define SPARK_GLM52_VERIFY_LOOKUP_MAX_MATCH 8u

static inline uint32_t SparkGlm52VerifyChainRequested(void)
{
	uint32_t rows = 0u;
	return(SparkSpeculationVerifyRowsParse(getenv(SPARK_GLM52_VERIFY_ROWS_ENV),SPARK_GLM52_VERIFY_ROWS_LIMIT,&rows) == SPARK_STATUS_OK && rows != 0u ? 1u : 0u);
}
