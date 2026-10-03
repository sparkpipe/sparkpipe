#pragma once

#include <stdint.h>
#include <stdio.h>

#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "sparkpipe/spark_status.h"

static inline SparkStatus SparkModuleTpCollectiveIdentifier(const char *tag,uint32_t tp_degree,uint64_t identifier,uint32_t *disabled)
{
	*disabled = identifier == 0u ? 1u : 0u;
	if ( *disabled == 0u || tp_degree == 1u )
		return(SPARK_STATUS_OK);
#ifdef DEBUG
	fprintf(stderr,"%s DEBUG tp_collective_identifier=0 degree=%u: no collective; every reduce returns this rank's partial sums\n",tag,tp_degree);
	return(SPARK_STATUS_OK);
#else
	*disabled = 0u;
	fprintf(stderr,"%s TP-COLLECTIVE-IDENTIFIER-REQUIRED degree=%u: tp_collective_identifier is 0; a release build at TP>1 always opens its collective\n",tag,tp_degree);
	SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
#endif
}

static inline SparkStatus SparkModuleTpStandalone(const char *tag,const char *variable,uint32_t tp_degree,uint32_t *standalone)
{
	SparkStatus status;
	*standalone = 0u;
	status = SparkStageModuleEnvironmentUnsignedOrDefault(tag,variable,0u,1u,0u,standalone);
	if ( status != SPARK_STATUS_OK || *standalone == 0u || tp_degree == 1u )
		return(status);
#ifdef DEBUG
	fprintf(stderr,"%s DEBUG %s=1 degree=%u: no collective; each rank computes from its own partial sums\n",tag,variable,tp_degree);
	return(SPARK_STATUS_OK);
#else
	*standalone = 0u;
	fprintf(stderr,"%s TP-STANDALONE-REFUSED degree=%u: %s=1 skips the TP collective and is accepted only by a DEBUG build\n",tag,tp_degree,variable);
	SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
#endif
}
