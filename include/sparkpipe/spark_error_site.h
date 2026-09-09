#pragma once

#include <stdint.h>
#include <stdio.h>

typedef struct SparkErrorSiteRecord
{
	int32_t code;
	const char *file;
	uint32_t line;
} SparkErrorSiteRecord;

static _Thread_local __attribute__((unused)) SparkErrorSiteRecord
    spark_last_error_site = {0,0,0};

#define SPARK_ERR_REPORT(code_value) \
	((void)fprintf(stderr,"ERRSITE %s:%d status=%d\n", \
		__FILE__,__LINE__,(int)(code_value)))

#define SPARK_FAIL(status_value) \
	do { \
		spark_last_error_site.code = (int32_t)(status_value); \
		spark_last_error_site.file = __FILE__; \
		spark_last_error_site.line = (uint32_t)__LINE__; \
		SPARK_ERR_REPORT(status_value); \
		return (status_value); \
	} while (0)

#define SPARK_RETURN(status_value) \
	do { \
		int32_t spark_ret_code = (int32_t)(status_value); \
		if (spark_ret_code != 0) { \
			spark_last_error_site.code = spark_ret_code; \
			spark_last_error_site.file = __FILE__; \
			spark_last_error_site.line = (uint32_t)__LINE__; \
			SPARK_ERR_REPORT(spark_ret_code); \
		} \
		return spark_ret_code; \
	} while (0)
