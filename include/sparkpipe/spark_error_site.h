#ifndef SPARKPIPE_SPARK_ERROR_SITE_H
#define SPARKPIPE_SPARK_ERROR_SITE_H

#include <stdint.h>
#include <stdio.h>

typedef struct SparkErrorSiteRecord
{
	int32_t error_status;
	const char *file;
	uint32_t line;
} SparkErrorSiteRecord;

extern _Thread_local SparkErrorSiteRecord spark_last_error_site;

#define SPARK_ERR_REPORT(error_status) \
	((void)fprintf(stderr,"ERRSITE %s:%d status=%d\n", \
		__FILE__,__LINE__,(int)(error_status)))

#define SPARK_FAIL(error_status) \
	do { \
		int32_t spark_fail_status = (int32_t)(error_status); \
		spark_last_error_site.error_status = spark_fail_status; \
		spark_last_error_site.file = __FILE__; \
		spark_last_error_site.line = (uint32_t)__LINE__; \
		SPARK_ERR_REPORT(spark_fail_status); \
		return spark_fail_status; \
	} while (0)

#endif
