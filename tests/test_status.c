#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_status.h"

int32_t main(void)
{
	if ( strcmp(SparkStatusToString(SPARK_STATUS_UNSUPPORTED),"unsupported") != 0 )
		return(1);
	if ( strcmp(SparkStatusToString((SparkStatus)UINT32_MAX),"unknown_status") != 0 )
		return(2);
	return(0);
}
