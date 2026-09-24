#pragma once

static int SPARK_FAMILY(ValFail)(const char *check,const char *detail)
{
	fprintf(stderr,SPARK_FAMILY_STRING(SPARK_FAMILY_LOWER) "_validation failure=%s detail=%s\n",check,detail);
	return(1);
}
