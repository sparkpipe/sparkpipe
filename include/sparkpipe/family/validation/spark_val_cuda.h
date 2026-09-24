#pragma once

static int SPARK_FAMILY(ValCuda)(cudaError_t error, const char *check)
{
	if (error == cudaSuccess)
		return(0);
	fprintf(stderr,SPARK_FAMILY_STRING(SPARK_FAMILY_LOWER) "_validation failure=%s cuda=%s\n",check,cudaGetErrorString(error));
	return(1);
}
