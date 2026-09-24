#pragma once

static SparkStatus SPARK_FAMILY(PackFileSize)(FILE *file,uint64_t *bytes)
{
	off_t end;
	if ( file == 0 || bytes == 0 || fseeko(file,0,SEEK_END) != 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	end = ftello(file);
	if ( end < 0 || fseeko(file,0,SEEK_SET) != 0 )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	*bytes = (uint64_t)end;
	return(SPARK_STATUS_OK);
}
