#ifndef SPARKPIPE_SPARK_NAMED_PACK_BIND_H
#define SPARKPIPE_SPARK_NAMED_PACK_BIND_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_named_pack.h"
#include "sparkpipe/spark_weightd_lazy_pack.h"

static inline SparkStatus SparkNamedPackTensorRange(const SparkNamedPack *pack,const char *tag,const char *name,const char *kind,
	uint64_t *offset,uint64_t *bytes,uint32_t *rows)
{
	SparkNamedPackEntry entry;
	if ( SparkNamedPackEntryLoad(pack,name,&entry) != SPARK_STATUS_OK || strcmp(entry.kind,kind) != 0 || entry.shape_count < 1u )
	{
		fprintf(stderr,"%s: pack tensor %s missing or not %s\n",tag,name,kind);
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	}
	*offset = pack->payload_base + entry.payload_offset;
	*bytes = entry.bytes;
	if ( rows != 0 )
		*rows = entry.shape[0];
	return(SPARK_STATUS_OK);
}

static inline SparkStatus SparkNamedPackBindSized(const SparkNamedPack *pack,SparkWeightdLazyPack *lazy_pack,const char *tag,const char *name,
	const char *kind,uint64_t expected_bytes,const void **bound)
{
	uint64_t offset,bytes;
	const void *slice = 0;
	if ( SparkNamedPackTensorRange(pack,tag,name,kind,&offset,&bytes,0) != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	if ( expected_bytes == 0u || bytes != expected_bytes )
	{
		fprintf(stderr,"%s: pack tensor %s is %llu bytes, the stage geometry needs %llu\n",tag,name,(unsigned long long)bytes,
			(unsigned long long)expected_bytes);
		SPARK_FAIL(SPARK_STATUS_TARGET_MISMATCH);
	}
	if ( SparkWeightdLazyPackSlice(lazy_pack,offset,bytes,&slice) != SPARK_STATUS_OK || slice == 0 )
	{
		fprintf(stderr,"%s: weightd has no resident bytes for %s\n",tag,name);
		SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
	}
	*bound = slice;
	return(SPARK_STATUS_OK);
}

#endif
