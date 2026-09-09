#ifndef SPARKPIPE_SPARK_ERROR_SITE_H
#define SPARKPIPE_SPARK_ERROR_SITE_H

#include <stdint.h>
#include <stdio.h>

typedef struct SparkErrorSiteRecord
{
	int32_t status;
	const char *file;
	uint32_t line;
} SparkErrorSiteRecord;

extern _Thread_local SparkErrorSiteRecord spark_last_error_site;

#define SPARK_ERR_REPORT(status) \
	((void)fprintf(stderr,"ERRSITE %s:%d status=%d\n", \
		__FILE__,__LINE__,(int)(status)))

#define SPARK_FAIL(status) \
	do { \
		int32_t spark_status = (int32_t)(status); \
		spark_last_error_site.status = spark_status; \
		spark_last_error_site.file = __FILE__; \
		spark_last_error_site.line = (uint32_t)__LINE__; \
		SPARK_ERR_REPORT(spark_status); \
		return spark_status; \
	} while (0)

#endif
