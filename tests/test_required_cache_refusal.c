#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_model_serving_adapter.h"

#ifndef TEST_REFUSED_SERVING_ADAPTER_PATHS
#define TEST_REFUSED_SERVING_ADAPTER_PATHS 0
#endif

int main(void)
{
	static const char *const paths[] = { TEST_REFUSED_SERVING_ADAPTER_PATHS };
	SparkModelServingAdapterDynamicLibrary library;
	uint32_t index,refused;
	refused = 0u;
	for (index=0u; index<sizeof(paths)/sizeof(paths[0]); index++)
	{
		assert(paths[index] != 0 && paths[index][0] != '\0');
		memset(&library,0,sizeof(library));
		assert(SparkModelServingAdapterLoadInterfaceFromSharedObject(paths[index],0u,&library) == SPARK_STATUS_UNSUPPORTED);
		fprintf(stderr,"refused as required: %s\n",paths[index]);
		refused++;
	}
	assert(refused == sizeof(paths)/sizeof(paths[0]) && refused != 0u);
	fprintf(stderr,"test_required_cache_refusal: %u adapters refused\n",refused);
	return(0);
}
